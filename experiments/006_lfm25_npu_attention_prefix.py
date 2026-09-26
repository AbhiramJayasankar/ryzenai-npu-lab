"""Check the first LFM2.5 attention layer's Q/K/V prefix on Phoenix NPU."""

import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_attention_prefix_kernel import attention_prefix
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    with np.load(ROOT / "cache" / "lfm25-attention2-reference.npz") as reference:
        hidden = reference["hidden"].reshape(-1).astype(bfloat16)
        gamma = reference["operator_gamma"].reshape(-1).astype(bfloat16)
        matrices = [reference[f"{name}_weight"].reshape(-1).astype(bfloat16) for name in ("q", "k", "v")]
        aux = np.concatenate([reference[name].reshape(-1) for name in ("q_gamma", "k_gamma", "cos", "sin")]).astype(bfloat16)
        expected = np.concatenate([reference[name].reshape(-1) for name in ("q", "k", "v")]).astype(np.float32)
    packed_weights = np.concatenate([
        np.pad(gamma, (0, 4096 - gamma.size)),
        *matrices,
        np.pad(aux, (0, 4096 - aux.size)),
    ]).astype(bfloat16)
    h = iron.tensor(hidden, dtype=bfloat16)
    w = iron.tensor(packed_weights, dtype=bfloat16)
    output = iron.zeros((2048,), dtype=bfloat16, device="npu")
    attention_prefix(h, w, output)
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        attention_prefix(h, w, output)
        timings.append((time.perf_counter() - start) * 1000)
    actual = output.numpy().astype(np.float32)
    parts = {"q": (0, 1024), "k": (1024, 1536), "v": (1536, 2048)}
    result = {
        "device": "Phoenix NPU1",
        "operation": "attention layer 2 operator norm, Q/K/V projections, head norms and rotary position",
        "median_ms": statistics.median(timings),
        "max_abs_error": float(np.max(np.abs(actual - expected))),
        "exact_fraction": float(np.mean(actual == expected)),
        "parts": {
            name: {
                "max_abs_error": float(np.max(np.abs(actual[a:b] - expected[a:b]))),
                "exact_fraction": float(np.mean(actual[a:b] == expected[a:b])),
            }
            for name, (a, b) in parts.items()
        },
    }
    print(json.dumps(result, indent=2))
    if result["max_abs_error"] != 0:
        raise RuntimeError("Attention Q/K/V prefix exceeded BF16 tolerance")


if __name__ == "__main__":
    main()
