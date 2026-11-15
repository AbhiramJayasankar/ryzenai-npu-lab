"""Experiment 013: CPU and GPU baselines for the operations timed on the NPU.

Same operations and sizes as 013_npu_bench.py. Backends:
  numpy        NumPy with OpenBLAS (FP32 for float work; NumPy has no fast BF16)
  torch_cpu    PyTorch CPU (oneDNN / ATen kernels), FP32 and BF16
  torch_cuda   PyTorch on the RTX 4060, timed twice:
                 compute  inputs already on the GPU, includes launch + synchronize
                 transfer copy inputs host -> GPU, compute, copy the result back

  python 013_cpu_gpu_bench.py --threads 8 [--only cpu|gpu] [--ops gemm,...]

The thread count must be set before NumPy is imported, so it is a command-line
option that sets OPENBLAS_NUM_THREADS / OMP_NUM_THREADS first. Results are
appended to cache/013/cpu_gpu.jsonl.
"""

import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--only", choices=["cpu", "gpu", "all"], default="all")
ap.add_argument("--ops", default="")
ap.add_argument("--tag", default="")
A = ap.parse_args()
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = str(A.threads)

import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(A.threads)
torch.backends.cuda.matmul.allow_tf32 = False   # FP32 means FP32 here
torch.backends.cudnn.allow_tf32 = False
OUT = Path(__file__).resolve().parents[1] / "cache/013/cpu_gpu.jsonl"
GPU = torch.cuda.is_available() and A.only in ("gpu", "all")
CPU = A.only in ("cpu", "all")
MB = 1 << 20
R = np.random.default_rng(13)


def timed(fn, min_n=10, max_n=200, budget_s=1.0, sync=None):
    for _ in range(3):
        fn()
    if sync:
        sync()
    samples = []
    t_end = time.perf_counter() + budget_s
    while len(samples) < max_n and (len(samples) < min_n or time.perf_counter() < t_end):
        t = time.perf_counter()
        fn()
        if sync:
            sync()
        samples.append(time.perf_counter() - t)
    s = sorted(samples)
    return {"median_ms": 1e3 * statistics.median(s), "p10_ms": 1e3 * s[len(s) // 10],
            "p90_ms": 1e3 * s[(9 * len(s)) // 10], "n": len(s)}


def emit(op, size, backend, dtype, t, work, mode=""):
    row = {"op": op, "size": size, "backend": backend, "dtype": dtype, "mode": mode,
           "threads": A.threads if backend != "torch_cuda" else None, **t, "work": work,
           "tag": A.tag}
    ms = t["median_ms"]
    if "bytes" in work:
        row["gbps"] = work["bytes"] / ms / 1e6
    if "macs" in work:
        row["gmacs"] = work["macs"] / ms / 1e6
    if "elems" in work:
        row["gelems"] = work["elems"] / ms / 1e6
    print(json.dumps(row), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")


def sync_cuda():
    torch.cuda.synchronize()


def run_torch(op, size, fn, host_inputs, work, dtypes=("float32", "bfloat16"),
              cpu=True, gpu=True):
    """fn(*tensors) -> tensor. host_inputs: list of float32/int NumPy arrays."""
    for dt in dtypes:
        try:
            _run_torch_one(op, size, fn, host_inputs, work, dt, cpu, gpu)
        except Exception as e:  # e.g. a dtype without a kernel on this backend
            print(json.dumps({"op": op, "size": size, "dtype": dt, "error": repr(e)[:200]}), flush=True)


def _run_torch_one(op, size, fn, host_inputs, work, dt, cpu, gpu):
    if True:
        tdt = getattr(torch, dt)
        if CPU and cpu:
            xs = [torch.from_numpy(h).to(tdt) if h.dtype.kind == "f" else torch.from_numpy(h)
                  for h in host_inputs]
            emit(op, size, "torch_cpu", dt, timed(lambda: fn(*xs)), work)
        if GPU and gpu:
            hs = [torch.from_numpy(h).to(tdt) if h.dtype.kind == "f" else torch.from_numpy(h)
                  for h in host_inputs]
            ds = [h.cuda() for h in hs]
            emit(op, size, "torch_cuda", dt, timed(lambda: fn(*ds), sync=sync_cuda), work, "compute")

            def with_copy():
                d = [h.cuda() for h in hs]
                return fn(*d).cpu()

            emit(op, size, "torch_cuda", dt, timed(with_copy, sync=sync_cuda), work, "transfer")


def want(name):
    return not A.ops or name in A.ops.split(",")


def main():
    print(json.dumps({"threads": A.threads, "torch": torch.__version__, "numpy": np.__version__,
                      "gpu": torch.cuda.get_device_name(0) if GPU else None}), flush=True)
    if want("memcpy"):
        for n in (1024, 8192, 262144, 4 * MB, 16 * MB):
            a = np.arange(n, dtype=np.int32)
            b = np.empty_like(a)
            w = dict(bytes=8 * n, elems=n)
            if CPU:
                emit("memcpy", n, "numpy", "int32", timed(lambda: np.copyto(b, a)), w)
            ta, tb = torch.from_numpy(a), torch.from_numpy(b)
            if CPU:
                emit("memcpy", n, "torch_cpu", "int32", timed(lambda: tb.copy_(ta)), w)
            if GPU:
                da, db = ta.cuda(), tb.cuda()
                emit("memcpy", n, "torch_cuda", "int32", timed(lambda: db.copy_(da), sync=sync_cuda),
                     w, "compute")
                emit("memcpy", n, "torch_cuda", "int32",
                     timed(lambda: ta.cuda().cpu(), sync=sync_cuda), w, "transfer")
    if want("saxpy"):
        for n in (4096, 262144, 16 * MB):
            x, y = R.uniform(-4, 4, n).astype(np.float32), R.uniform(-4, 4, n).astype(np.float32)
            z = np.empty_like(x)
            w = dict(bytes=6 * n, elems=n, flops=2 * n)

            def np_saxpy():
                np.multiply(x, np.float32(3), out=z)
                np.add(z, y, out=z)

            if CPU:
                emit("saxpy", n, "numpy", "float32", timed(np_saxpy), w)
            run_torch("saxpy", n, lambda a, b: torch.add(b, a, alpha=3), [x, y], w)
    for op in ("add", "mul"):
        if not want(op):
            continue
        for n in (4096, 262144, 16 * MB):
            a, b = R.uniform(-1, 1, n).astype(np.float32), R.uniform(-1, 1, n).astype(np.float32)
            c = np.empty_like(a)
            f = np.add if op == "add" else np.multiply
            w = dict(bytes=6 * n, elems=n)
            if CPU:
                emit(op, n, "numpy", "float32", timed(lambda: f(a, b, out=c)), w)
            run_torch(op, n, torch.add if op == "add" else torch.mul, [a, b], w)
    for op in ("relu", "gelu", "exp"):
        if not want(op):
            continue
        sizes = (8192, 262144, 16 * MB) if op != "exp" else (4096, 262144, 4 * MB)
        for n in sizes:
            x = R.uniform(-3, 3, n).astype(np.float32)
            y = np.empty_like(x)
            w = dict(bytes=4 * n, elems=n)
            if CPU and op == "relu":
                emit(op, n, "numpy", "float32", timed(lambda: np.maximum(x, 0, out=y)), w)
            if CPU and op == "exp":
                emit(op, n, "numpy", "float32", timed(lambda: np.exp(x, out=y)), w)
            fn = {"relu": torch.relu, "gelu": torch.nn.functional.gelu, "exp": torch.exp}[op]
            run_torch(op, n, fn, [x], w)
    if want("softmax"):
        for n in (8192, 262144, 4 * MB):
            x = R.uniform(-4, 8, (n // 1024, 1024)).astype(np.float32)
            w = dict(bytes=4 * n, elems=n)

            def np_softmax():
                e = np.exp(x - x.max(axis=1, keepdims=True))
                return e / e.sum(axis=1, keepdims=True)

            if CPU:
                emit("softmax", n, "numpy", "float32", timed(np_softmax), w)
            run_torch("softmax", n, lambda t: torch.softmax(t, dim=1), [x], w)
    if want("max"):
        for n in (2048, 32768, MB, 16 * MB):
            xi = R.integers(-10 ** 6, 10 ** 6, n).astype(np.int32)
            xf = R.standard_normal(n).astype(np.float32)
            if CPU:
                emit("max", n, "numpy", "int32", timed(lambda: xi.max()), dict(bytes=4 * n, elems=n))
                emit("max", n, "numpy", "float32", timed(lambda: xf.max()), dict(bytes=4 * n, elems=n))
            run_torch("max", n, torch.max, [xi], dict(bytes=4 * n, elems=n), dtypes=("int32",))
            run_torch("max", n, torch.max, [xf], dict(bytes=2 * n, elems=n), dtypes=("bfloat16",))
    if want("sum"):
        for n in (1024, 32768, MB, 16 * MB):
            xi = R.integers(-500, 500, n).astype(np.int32)
            if CPU:
                emit("sum", n, "numpy", "int32", timed(lambda: xi.sum()), dict(bytes=4 * n, elems=n))
            run_torch("sum", n, torch.sum, [xi], dict(bytes=4 * n, elems=n), dtypes=("int32",))
    if want("gemv"):
        for M, K in ((256, 1024), (3072, 1024), (4096, 4096)):
            wm = (R.standard_normal((M, K)) / np.sqrt(K)).astype(np.float32)
            x = R.standard_normal(K).astype(np.float32)
            y = np.empty(M, np.float32)
            w = dict(bytes=2 * (M * K + K + M), macs=M * K)   # bytes at BF16 size
            if CPU:
                emit("gemv", f"{M}x{K}", "numpy", "float32",
                     timed(lambda: np.matmul(wm, x, out=y)), {**w, "bytes": 4 * (M * K + K + M)})
            run_torch("gemv", f"{M}x{K}", torch.mv, [wm, x], w)
    if want("mv_i16"):
        for M in (288, 1024):
            a = R.integers(-100, 100, (M, M)).astype(np.int16)
            b = R.integers(-100, 100, M).astype(np.int16)
            w = dict(macs=M * M, bytes=2 * M * M)
            if CPU:
                emit("mv_i16", f"{M}x{M}", "numpy", "int16 (weak: no BLAS)", timed(lambda: a @ b), w)
                a32, b32 = a.astype(np.float32), b.astype(np.float32)
                emit("mv_i16", f"{M}x{M}", "numpy", "float32 sgemv (exact here)",
                     timed(lambda: a32 @ b32), w)
    if want("gemm"):
        for n in (256, 512, 1024, 2048):
            a = R.standard_normal((n, n)).astype(np.float32)
            b = (R.standard_normal((n, n)) / np.sqrt(n)).astype(np.float32)
            c = np.empty((n, n), np.float32)
            w = dict(macs=n ** 3)
            if CPU:
                emit("gemm", n, "numpy", "float32", timed(lambda: np.matmul(a, b, out=c)), w)
            run_torch("gemm", n, torch.mm, [a, b], w, dtypes=("float32", "bfloat16", "float16"))
    if want("gemm_i8"):
        for n in (256, 1024, 2048):
            a = R.integers(-128, 128, (n, n)).astype(np.int8)
            b = R.integers(-128, 128, (n, n)).astype(np.int8)
            w = dict(macs=n ** 3)
            run_torch("gemm_i8", n, torch._int_mm, [a, b], w, dtypes=("int8",))
    if want("gemm_i16"):
        for n in (256, 1024, 2048):
            a = R.integers(-256, 256, (n, n)).astype(np.int16)
            b = R.integers(-256, 256, (n, n)).astype(np.int16)
            w = dict(macs=n ** 3)
            if CPU:
                emit("gemm_i16", n, "numpy", "int16 (weak: no BLAS)",
                     timed(lambda: a @ b, min_n=3, max_n=10, budget_s=3.0), w)
                ta, tb = torch.from_numpy(a.astype(np.int32)), torch.from_numpy(b.astype(np.int32))
                emit("gemm_i16", n, "torch_cpu", "int32 (no int16 GEMM kernel)",
                     timed(lambda: ta @ tb, min_n=3, max_n=10, budget_s=3.0), w)
                # FP32 holds these products and sums exactly up to 2^24: exact for K <= 256
                # with |x| < 256, close otherwise. Shown as the practical CPU route.
                a32, b32 = a.astype(np.float32), b.astype(np.float32)
                emit("gemm_i16", n, "numpy", "float32 sgemm (practical route)",
                     timed(lambda: a32 @ b32), w)
    if want("conv1x1"):
        for wd, ht, ci, co in ((8, 8, 64, 64), (32, 4, 64, 64), (32, 32, 64, 64), (32, 128, 128, 128)):
            x = R.integers(1, 20, (1, ci, ht, wd)).astype(np.float32)
            k = R.integers(50, 80, (co, ci, 1, 1)).astype(np.float32)
            w = dict(macs=ci * co * ht * wd)
            run_torch("conv1x1", f"{wd}x{ht}x{ci}x{co}", torch.nn.functional.conv2d, [x, k], w,
                      dtypes=("float32", "bfloat16", "float16"))


if __name__ == "__main__":
    main()
