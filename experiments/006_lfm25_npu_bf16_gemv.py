"""Check a fused FP32-accumulate/BF16-output LFM2.5 NPU projection."""

import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_bf16_gemv_kernel import bf16_bf16_gemv
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as ref:
        activation = ref["step1_first_projection_input"].astype(bfloat16)
        weight = ref["first_projection_weight"].astype(bfloat16)
        expected = ref["step1_first_projection_output"]
    M, K = weight.shape
    w = iron.tensor(weight, dtype=bfloat16)
    x = iron.tensor(activation, dtype=bfloat16)
    y = iron.zeros((M,), dtype=bfloat16, device="npu")

    def run():
        bf16_bf16_gemv(w, x, y, M=M, K=K, n_cores=4)

    run()
    times = []
    for _ in range(10):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    actual = y.numpy().astype(np.float32)
    result = {
        "device": "Phoenix NPU1",
        "operation": "first LFM2.5 1024->3072 projection with BF16 output",
        "cores": 4,
        "median_ms": statistics.median(times),
        "exact_fraction": float(np.mean(actual == expected)),
        "max_abs_error": float(np.max(np.abs(actual - expected))),
    }
    print(json.dumps(result, indent=2))
    if result["max_abs_error"] != 0:
        raise RuntimeError("Fused BF16 GEMV did not match the CPU reference")


if __name__ == "__main__":
    main()
