"""Custom-kernel route: how fast can the Phoenix NPU run Parakeet-sized matmuls?

Uses IRON's whole-array GEMM design (programming_examples/basic/
matrix_multiplication/whole_array: 4 columns x 4 rows = 16 cores, each core
running the vectorized aie::mmul kernel from aie_kernels/aie2/mm.cc) at the
encoder's shapes, submitted directly through pyxrt (npu_direct), and checks
the result against NumPy.

  powershell -NoProfile -File scripts\\run_npu_timeout.ps1 -Script experiments\\011_iron_gemm_probe.py -Seconds 1500

Shapes: M = encoder frames padded to the tile grid (128 = 10 s, 384 = 30 s),
(K, N) = FFN linear1 (1024, 4096), FFN linear2 (4096, 1024), attention
projection (1024, 1024). Writes cache/parakeet/results/iron_gemm_probe.json.
"""

import json
import sys
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, Out
from ml_dtypes import bfloat16

import npu_direct as nd

ROOT = Path(__file__).resolve().parents[1]
WA = ROOT / "cache/iron/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
sys.path.insert(0, str(WA))
import whole_array as wa  # noqa: E402

KEEP = []  # pyxrt objects crash the process when collected; keep every program/buffer alive
DT = {"bf16": bfloat16, "f32": np.float32, "i8": np.int8, "i16": np.int16, "i32": np.int32}


def design(M, K, N, m, k, n, din, dout, bcol=0, ccol=0):
    tag = f"p011gemm_{M}_{K}_{N}_{m}_{k}_{n}_{din}_{dout}" + (f"_b{bcol}c{ccol}" if bcol or ccol else "")

    @iron.jit(tag=tag)
    def gemm(A: In, B: In, C: Out, *, tag: CompileTime[str]):
        return wa._build_design(iron.get_current_device(), M, K, N, m, k, n, 4, din, dout,
                                bcol, ccol, False, False, False)

    return gemm


def run(M, K, N, m, k, n, din, dout, bcol=0, ccol=0, repeats=20):
    """bcol/ccol: B given column-major ([N, K] row-major) / C written column-major ([N, M]).
    With A = weights^T and B = activations this streams every weight exactly once."""
    rng = np.random.default_rng(0)
    if din == "i8":
        a = rng.integers(-128, 128, (M, K), dtype=np.int8)
        b = rng.integers(-128, 128, (K, N), dtype=np.int8)
    else:
        a = rng.standard_normal((M, K)).astype(DT[din])
        b = (rng.standard_normal((K, N)) / np.sqrt(K)).astype(DT[din])
    t = time.perf_counter()
    prog = nd.Program.from_design(design(M, K, N, m, k, n, din, dout, bcol, ccol))
    compile_s = time.perf_counter() - t
    A, B, C = nd.Buffer(M * K, DT[din]), nd.Buffer(K * N, DT[din]), nd.Buffer(M * N, DT[dout])
    KEEP.extend([prog, A, B, C])
    A.write(a)
    B.write(b.T.copy() if bcol else b)
    prog(A, B, C)
    for _ in range(2):
        prog(A, B, C)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        prog(A, B, C)
        times.append(time.perf_counter() - t)
    got = C.from_device().reshape((N, M) if ccol else (M, N)).astype(np.float64)
    if ccol:
        got = got.T
    if din == "i8":
        ref = a.astype(np.int64) @ b.astype(np.int64)
        err = float(np.abs(got - ref).max())
    else:
        ref = a.astype(np.float32) @ b.astype(np.float32)
        err = float(np.abs(got - ref).max() / np.abs(ref).max())
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    cpu = []
    for _ in range(repeats):
        t = time.perf_counter()
        a32 @ b32
        cpu.append(time.perf_counter() - t)
    ms, cpu_ms = 1000 * float(np.median(times)), 1000 * float(np.median(cpu))
    macs = M * K * N
    row = {"M": M, "K": K, "N": N, "tile": [m, k, n], "dtype": f"{din}->{dout}", "b_col": bcol, "c_col": ccol,
           "npu_ms": ms,
           "npu_gmacs": macs / ms / 1e6, "numpy_fp32_ms": cpu_ms, "numpy_fp32_gmacs": macs / cpu_ms / 1e6,
           "max_err": err, "weight_mb": K * N * np.dtype(DT[din]).itemsize / 1e6, "compile_s": compile_s}
    row["weight_gbps_if_streamed_once"] = row["weight_mb"] / ms
    print(json.dumps(row), flush=True)
    return row


def main():
    rows = []
    configs = []
    for M, m in ((128, 16), (384, 48)):
        for K, N in ((1024, 4096), (4096, 1024), (1024, 1024)):
            configs.append((M, K, N, m, 64, 64, "bf16", "bf16"))
    for M, m in ((128, 16), (384, 48)):
        for K, N in ((1024, 4096), (4096, 1024)):
            configs.append((M, K, N, m, 64, 64, "i8", "i32"))
    # Weights as A (M = output features), activations as column-major B (N = 128 frames).
    for M, K in ((1024, 1024), (4096, 1024), (1024, 4096), (2048, 1024)):
        configs.append((M, K, 128, 64, 64, 32, "bf16", "f32", 1, 1))
    # Optional config indices on the command line: one process per config avoids
    # piling up hardware contexts.
    picked = [int(x) for x in sys.argv[1:]] or range(len(configs))
    out = ROOT / "cache/parakeet/results/iron_gemm_probe.json"
    if sys.argv[1:] and out.exists():
        rows = json.loads(out.read_text())
    for cfg in [configs[i] for i in picked]:
        try:
            rows.append(run(*cfg))
        except Exception as e:  # keep going: report which shapes fail to compile
            print(json.dumps({"config": cfg, "error": repr(e)[:300]}), flush=True)
            rows.append({"config": cfg, "error": repr(e)[:300]})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    nd.finish()


if __name__ == "__main__":
    main()
