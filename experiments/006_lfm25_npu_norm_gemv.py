"""Verify RMSNorm and the first LFM2.5 GEMV in one Phoenix NPU program."""

import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_norm_gemv_kernel import norm_bf16_gemv
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as ref:
        hidden = ref["step1_hidden0"].astype(bfloat16)
        gamma = ref["first_operator_norm_weight"].astype(bfloat16)
        weight = ref["first_projection_weight"].astype(bfloat16)
        expected = ref["step1_first_projection_output"]
    M, K = weight.shape
    packed = np.concatenate(
        [np.pad(gamma, (0, 4096 - K)), weight.reshape(-1)]
    ).astype(bfloat16)
    x = iron.tensor(hidden, dtype=bfloat16)
    w = iron.tensor(packed, dtype=bfloat16)
    y = iron.zeros((M,), dtype=bfloat16, device="npu")

    def run():
        norm_bf16_gemv(x, w, y, M=M, K=K)

    run()
    times = []
    for _ in range(10):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    actual = y.numpy().astype(np.float32)
    result = {
        "device": "Phoenix NPU1",
        "operation": "single-program first RMSNorm and 1024->3072 GEMV",
        "cores": 1,
        "median_ms": statistics.median(times),
        "exact_fraction": float(np.mean(actual == expected)),
        "max_abs_error": float(np.max(np.abs(actual - expected))),
    }
    print(json.dumps(result, indent=2))
    if result["max_abs_error"] != 0:
        raise RuntimeError("Composed NPU program did not match CPU BF16 reference")


if __name__ == "__main__":
    main()
