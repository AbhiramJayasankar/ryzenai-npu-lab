"""Test a real LFM2.5 decode projection using a true NPU matrix-vector kernel."""

import argparse
import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_gemv_kernel import bf16_f32_gemv
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "cache" / "lfm25-reference-cpu.npz"
    )
    parser.add_argument("--cores", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    with np.load(args.reference.resolve(strict=True)) as ref:
        activation = ref["step1_first_projection_input"].astype(bfloat16)
        weight = ref["first_projection_weight"].astype(bfloat16)
        expected = ref["step1_first_projection_output"]
    M, K = weight.shape
    w = iron.tensor(weight, dtype=bfloat16)
    x = iron.tensor(activation, dtype=bfloat16)
    y = iron.zeros((M,), dtype=np.float32, device="npu")

    def run():
        bf16_f32_gemv(w, x, y, M=M, K=K, n_cores=args.cores)

    run()
    times = []
    for _ in range(10):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    actual = y.numpy().astype(np.float32)
    error = actual - expected
    result = {
        "operation": "LFM2.5 layer 0 1024-to-3072 decode projection",
        "device": "Phoenix NPU1",
        "kernel": "BF16 input, FP32 accumulate/output, unpadded GEMV",
        "cores": args.cores,
        "median_ms": statistics.median(times),
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_abs_error": float(np.mean(np.abs(error))),
        "within_reference_tolerance": bool(
            np.max(np.abs(error)) <= 0.01 and np.mean(np.abs(error)) <= 0.002
        ),
        "first_8_actual": actual[:8].tolist(),
        "first_8_reference": expected[:8].tolist(),
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not result["within_reference_tolerance"]:
        raise RuntimeError("GEMV result exceeded reference tolerance")


if __name__ == "__main__":
    main()
