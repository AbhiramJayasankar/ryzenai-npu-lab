"""Validate a real first-block LFM2.5 FFN with all operations on Phoenix NPU."""

import argparse
import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_cast_kernel import f32_to_bf16
from lfm25_gemv_kernel import bf16_f32_gemv
from lfm25_silu_gate_kernel import silu_gate
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "cache" / "lfm25-reference-cpu.npz"
    )
    args = parser.parse_args()
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected the Phoenix NPU1")
    with np.load(args.reference.resolve(strict=True)) as ref:
        activation = ref["step1_ffn0_norm_output"].astype(bfloat16)
        weights = {
            name: ref[f"first_ffn_{name}_weight"].astype(bfloat16)
            for name in ("w1", "w2", "w3")
        }
        expected = {
            name: ref[f"step1_ffn0_{name}_output"]
            for name in ("w1", "w2", "w3")
        }
        expected_gated = ref["step1_ffn0_w2_input"]
    x = iron.tensor(activation, dtype=bfloat16)
    w = {name: iron.tensor(weight, dtype=bfloat16) for name, weight in weights.items()}
    fp32 = {
        name: iron.zeros((weight.shape[0],), dtype=np.float32, device="npu")
        for name, weight in weights.items()
    }
    bf16 = {
        name: iron.zeros((weight.shape[0],), dtype=bfloat16, device="npu")
        for name, weight in weights.items()
    }
    gated = iron.zeros((2560,), dtype=bfloat16, device="npu")

    def run():
        for name in ("w1", "w3"):
            rows, cols = weights[name].shape
            bf16_f32_gemv(w[name], x, fp32[name], M=rows, K=cols, n_cores=4)
            f32_to_bf16(fp32[name], bf16[name], N=rows)
        silu_gate(bf16["w1"], bf16["w3"], gated, N=2560)
        rows, cols = weights["w2"].shape
        bf16_f32_gemv(w["w2"], gated, fp32["w2"], M=rows, K=cols, n_cores=4)
        f32_to_bf16(fp32["w2"], bf16["w2"], N=rows)

    run()
    times = []
    for _ in range(5):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    result = {
        "operation": "LFM2.5 layer 0 FFN: w1, w3, SiLU gate, w2",
        "device": "Phoenix NPU1",
        "median_chain_ms": statistics.median(times),
    }
    for name in ("w1", "w3", "w2"):
        actual = bf16[name].numpy().astype(np.float32)
        result[f"{name}_exact_fraction"] = float(np.mean(actual == expected[name]))
        result[f"{name}_max_abs_error"] = float(np.max(np.abs(actual - expected[name])))
    gated_actual = gated.numpy().astype(np.float32)
    result["gated_exact_fraction"] = float(np.mean(gated_actual == expected_gated))
    result["gated_max_abs_error"] = float(np.max(np.abs(gated_actual - expected_gated)))
    print(json.dumps(result, indent=2))
    if any(
        result[f"{name}_max_abs_error"] > 0.015625
        for name in ("w1", "w3", "w2", "gated")
    ):
        raise RuntimeError("NPU FFN exceeded BF16 reference tolerance")


if __name__ == "__main__":
    main()
