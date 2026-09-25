"""Compare custom Phoenix RMSNorm against LFM2.5's first decode block."""

import argparse
import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_rms_norm_kernel import rms_norm
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "cache" / "lfm25-reference-cpu.npz"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected the Phoenix NPU1")
    with np.load(args.reference.resolve(strict=True)) as ref:
        x_host = ref["step1_hidden0"].astype(bfloat16)
        g_host = ref["first_operator_norm_weight"].astype(bfloat16)
        expected = ref["step1_first_projection_input"]
    x = iron.tensor(x_host, dtype=bfloat16)
    g = iron.tensor(g_host, dtype=bfloat16)
    y = iron.zeros((1024,), dtype=bfloat16, device="npu")

    def run():
        rms_norm(x, g, y, N=1024)

    run()
    times = []
    for _ in range(20):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    actual = y.numpy().astype(np.float32)
    error = actual - expected
    result = {
        "operation": "LFM2.5 layer 0 operator RMSNorm, decode token",
        "device": "Phoenix NPU1",
        "elements": 1024,
        "median_ms": statistics.median(times),
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_abs_error": float(np.mean(np.abs(error))),
        "exact_bf16_match_fraction": float(np.mean(actual == expected)),
        "first_8_actual": actual[:8].tolist(),
        "first_8_reference": expected[:8].tolist(),
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if result["max_abs_error"] > 0.01:
        raise RuntimeError("NPU RMSNorm exceeded numerical tolerance")


if __name__ == "__main__":
    main()
