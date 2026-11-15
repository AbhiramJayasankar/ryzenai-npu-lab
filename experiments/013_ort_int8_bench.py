"""Experiment 013: ONNX Runtime CPU baselines for int8 matrix multiply and 1x1 convolution.

ONNX Runtime's MLAS kernels are the usual fast int8 route on an x86 CPU
(AVX-512 VNNI on Zen 4). Run in the Ryzen AI environment, which has both
onnx and onnxruntime:

  %USERPROFILE%\\miniforge3\\envs\\ryzen-ai-1.7.0\\python.exe experiments\\013_ort_int8_bench.py --threads 8

Activations are tested as int8 (s8s8, same types as the NPU) and as uint8 (u8s8,
the layout ONNX Runtime's x86 int8 kernels are built for; in this build s8s8
matmul falls back to a path about 100x slower).

Ops:
  MatMulInteger  int8 x int8 -> int32       (same output as the NPU int8 GEMM)
  QLinearMatMul  int8 x int8 -> int8         (requantized output)
  QLinearConv    1x1, int8 activations and weights -> int8 (as the NPU conv kernel)
Appends rows to cache/013/cpu_gpu.jsonl with backend "onnxruntime_cpu".
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

OUT = Path(__file__).resolve().parents[1] / "cache/013/cpu_gpu.jsonl"
R = np.random.default_rng(13)


def timed(fn, min_n=10, max_n=200, budget_s=1.0):
    for _ in range(3):
        fn()
    samples = []
    t_end = time.perf_counter() + budget_s
    while len(samples) < max_n and (len(samples) < min_n or time.perf_counter() < t_end):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    s = sorted(samples)
    return {"median_ms": 1e3 * statistics.median(s), "p10_ms": 1e3 * s[len(s) // 10],
            "p90_ms": 1e3 * s[(9 * len(s)) // 10], "n": len(s)}


def session(model, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])


def model_of(nodes, inputs, outputs, inits=()):
    g = helper.make_graph(nodes, "g", inputs, outputs, initializer=list(inits))
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def scalar(name, value, dtype):
    return helper.make_tensor(name, dtype, [], [value])


def emit(op, size, dtype, t, work, threads):
    row = {"op": op, "size": size, "backend": "onnxruntime_cpu", "dtype": dtype, "mode": "",
           "threads": threads, **t, "work": work, "ort": ort.__version__}
    row["gmacs"] = work["macs"] / t["median_ms"] / 1e6
    print(json.dumps(row), flush=True)
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    th = ap.parse_args().threads
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for n in (256, 1024, 2048):
        for at, an, tag in ((TensorProto.INT8, np.int8, "s8s8"), (TensorProto.UINT8, np.uint8, "u8s8")):
            a = (R.integers(-128, 128, (n, n)) if an == np.int8 else R.integers(0, 256, (n, n))).astype(an)
            b = R.integers(-128, 128, (n, n)).astype(np.int8)
            w = {"macs": n ** 3}
            m = model_of([helper.make_node("MatMulInteger", ["A", "B"], ["C"])],
                         [helper.make_tensor_value_info("A", at, [n, n]),
                          helper.make_tensor_value_info("B", TensorProto.INT8, [n, n])],
                         [helper.make_tensor_value_info("C", TensorProto.INT32, [n, n])])
            s = session(m, th)
            got = s.run(None, {"A": a, "B": b})[0]
            assert np.array_equal(got.astype(np.int64), a.astype(np.int64) @ b.astype(np.int64))
            emit("gemm_i8", n, f"{tag} -> int32 (MatMulInteger)", timed(lambda: s.run(None, {"A": a, "B": b})), w, th)
            inits = [scalar("sa", 0.05, TensorProto.FLOAT), scalar("za", 0, at),
                     scalar("sb", 0.05, TensorProto.FLOAT), scalar("zb", 0, TensorProto.INT8),
                     scalar("sy", 0.05 * 0.05 * 64 * np.sqrt(n), TensorProto.FLOAT),
                     scalar("zy", 0, at)]
            m = model_of([helper.make_node("QLinearMatMul", ["A", "sa", "za", "B", "sb", "zb", "sy", "zy"], ["Y"])],
                         [helper.make_tensor_value_info("A", at, [n, n]),
                          helper.make_tensor_value_info("B", TensorProto.INT8, [n, n])],
                         [helper.make_tensor_value_info("Y", at, [n, n])], inits)
            s2 = session(m, th)
            emit("gemm_i8", n, f"{tag} -> 8-bit (QLinearMatMul)", timed(lambda: s2.run(None, {"A": a, "B": b})), w, th)
    for wd, ht, ci, co in ((8, 8, 64, 64), (32, 4, 64, 64), (32, 32, 64, 64), (32, 128, 128, 128)):
        for at, an, tag in ((TensorProto.INT8, np.int8, "s8s8"), (TensorProto.UINT8, np.uint8, "u8s8")):
            x = R.integers(1, 20, (1, ci, ht, wd)).astype(an)
            k = R.integers(50, 80, (co, ci, 1, 1)).astype(np.int8)
            inits = [scalar("sx", 1.0, TensorProto.FLOAT), scalar("zx", 0, at),
                     helper.make_tensor("W", TensorProto.INT8, k.shape, k.reshape(-1).tolist()),
                     scalar("sw", 1.0, TensorProto.FLOAT), scalar("zw", 0, TensorProto.INT8),
                     scalar("sy", 1024.0, TensorProto.FLOAT), scalar("zy", 0, at)]
            m = model_of([helper.make_node("QLinearConv", ["X", "sx", "zx", "W", "sw", "zw", "sy", "zy"], ["Y"],
                                           kernel_shape=[1, 1])],
                         [helper.make_tensor_value_info("X", at, list(x.shape))],
                         [helper.make_tensor_value_info("Y", at, [1, co, ht, wd])], inits)
            s = session(m, th)
            got = s.run(None, {"X": x})[0].astype(np.int64)
            acc = np.einsum("oi,iyx->oyx", k[:, :, 0, 0].astype(np.int64), x[0].astype(np.int64))
            ref = np.clip(np.round(acc / 1024.0), -128 if an == np.int8 else 0, 127 if an == np.int8 else 255)
            assert np.abs(got[0] - ref).max() <= 1
            emit("conv1x1", f"{wd}x{ht}x{ci}x{co}", f"{tag} -> 8-bit (QLinearConv)",
                 timed(lambda: s.run(None, {"X": x})), {"macs": ci * co * ht * wd}, th)


if __name__ == "__main__":
    main()
