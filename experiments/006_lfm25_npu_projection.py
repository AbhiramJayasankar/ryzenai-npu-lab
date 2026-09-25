"""Run LFM2.5's first decode projection on the Phoenix NPU.

This uses the official one-core IRON BF16 matrix kernel as an initial
correctness probe. It pads one input vector to 64 rows and is intentionally
not an efficient GEMV implementation.

Run from the repo root after activating Visual Studio Developer PowerShell
and cache/iron/mlir-aie/iron_env.ps1. The CPU reference NPZ comes from
006_lfm25_baseline.py --reference cache/lfm25-reference-cpu.npz.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "cache" / "lfm25-reference-cpu.npz"
    )
    parser.add_argument("--outputs", type=int, default=64)
    parser.add_argument("--cores", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.outputs < 64 or args.outputs > 3072 or args.outputs % (64 * args.cores):
        parser.error("--outputs must be a multiple of 64*cores up to 3072")
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected the Ryzen 9 8945HS Phoenix NPU1")
    from lfm25_matmul_kernel import bf16_f32_matmul

    with np.load(args.reference.resolve(strict=True)) as reference:
        activation = reference["step1_first_projection_input"]
        weight = reference["first_projection_weight"][: args.outputs]
        expected = reference["step1_first_projection_output"][: args.outputs]
    if activation.shape != (1024,) or weight.shape != (args.outputs, 1024):
        raise ValueError("Unexpected LFM2.5 first projection dimensions")

    # The initial matmul kernel requires a 16-row tile. Only row 0 contains
    # a model activation; row 0 of the output is the real model computation.
    host_a = np.zeros((16, 1024), dtype=bfloat16)
    host_a[0] = activation.astype(bfloat16)
    host_b = weight.T.copy().astype(bfloat16)
    a = iron.tensor(host_a, dtype=bfloat16)
    b = iron.tensor(host_b, dtype=bfloat16)
    c = iron.zeros((16, args.outputs), dtype=np.float32, device="npu")
    if type(a).__name__ != "XRTTensor":
        raise RuntimeError("Input is not an XRT NPU tensor")

    def run():
        bf16_f32_matmul(a, b, c, M=16, K=1024, N=args.outputs, n_cores=args.cores)

    run()
    timings = []
    for _ in range(10):
        start = time.perf_counter()
        run()
        timings.append((time.perf_counter() - start) * 1000)
    actual = c.numpy()[0].astype(np.float32)
    error = actual - expected
    close = bool(np.max(np.abs(error)) <= 0.01 and np.mean(np.abs(error)) <= 0.002)
    result = {
        "operation": "LFM2.5 layer 0 convolution input projection",
        "device": "Phoenix NPU1",
        "cores": args.cores,
        "inputs": "bfloat16",
        "accumulation_and_output": "float32",
        "shape_executed": [16, 1024, args.outputs],
        "useful_shape": [1, 1024, args.outputs],
        "median_ms": statistics.median(timings),
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_abs_error": float(np.mean(np.abs(error))),
        "within_reference_tolerance": close,
        "reference_max_abs": float(np.max(np.abs(expected))),
        "first_8_actual": actual[:8].tolist(),
        "first_8_reference": expected[:8].tolist(),
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not close:
        raise RuntimeError("NPU output exceeded reference tolerance")


if __name__ == "__main__":
    main()
