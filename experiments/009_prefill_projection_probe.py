"""Probe a 34-position BF16 LFM2.5 projection on the Phoenix NPU.

The 34 useful rows are padded to 48 for the 16-row AIE matrix tile. This is
one projection, not a full model prefill or a CPU versus NPU model comparison.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16

from lfm25_checkpoint import BF16Checkpoint
from lfm25_matmul_kernel import bf16_f32_matmul


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--reuse-weights", action="store_true")
    args = parser.parse_args()
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    checkpoint = BF16Checkpoint(ROOT / "cache/lfm25-230m/model.safetensors")
    weight = checkpoint.load("model.layers.0.conv.in_proj.weight")
    if weight.shape != (3072, 1024):
        raise AssertionError(weight.shape)
    rng = np.random.default_rng(9)
    host_a = np.zeros((48, 1024), dtype=bfloat16)
    host_a[:34] = rng.normal(0, 0.1, (34, 1024)).astype(bfloat16)
    host_b = weight.T.copy()
    a = iron.tensor(host_a, dtype=bfloat16)
    b = iron.tensor(host_b, dtype=bfloat16)
    c = iron.zeros((48, 3072), dtype=np.float32, device="npu")
    if type(a).__name__ != "XRTTensor":
        raise RuntimeError("Input did not become an XRT NPU tensor")

    def run():
        bf16_f32_matmul(a, b, c, M=48, K=1024, N=3072,
                        n_cores=args.cores, reuse_weights=args.reuse_weights)

    run()
    elapsed = []
    for _ in range(5):
        start = time.perf_counter()
        run()
        elapsed.append((time.perf_counter() - start) * 1000)
    actual = c.numpy()[:34]
    reference = host_a[:34].astype(np.float32) @ host_b.astype(np.float32)
    diff = np.abs(actual - reference)
    result = {
        "operation": "34 useful prompt positions in layer 0 input projection",
        "device": "Phoenix NPU1",
        "shape_executed": [48, 1024, 3072],
        "npu_cores": args.cores,
        "reuse_weights_across_positions": args.reuse_weights,
        "runs_ms": elapsed,
        "median_ms": statistics.median(elapsed),
        "max_abs_error_vs_fp32_reference": float(np.max(diff)),
        "mean_abs_error_vs_fp32_reference": float(np.mean(diff)),
    }
    print(json.dumps(result, indent=2))
    if result["max_abs_error_vs_fp32_reference"] > 0.03:
        raise AssertionError("NPU matrix output exceeded numerical tolerance")


if __name__ == "__main__":
    main()
