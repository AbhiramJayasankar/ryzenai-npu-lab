"""Compare a paired Q/K/V prefix with two established NPU prefix calls."""

import importlib
import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16

from lfm25_attention_prefix_kernel import attention_prefix
from lfm25_prefill_pair_attention_prefix_kernel import attention_prefix_pair


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    module = importlib.import_module("007_lfm25_npu_chat")
    chat = module.NPUChat("fixed64")
    with np.load(ROOT / "cache/lfm25-prompt-sequence-reference.npz") as ref:
        hidden = [ref[f"p{pos}_hidden2"].astype(bfloat16) for pos in range(2)]
    layer = chat.layers[2]
    packed_weights = []
    for pos in range(2):
        weight = layer["prefix_base"].copy()
        cos, sin = module.rotary(pos)
        weight[-4096 + 128:-4096 + 192] = cos
        weight[-4096 + 192:-4096 + 256] = sin
        packed_weights.append(weight)
    pair = iron.tensor(np.concatenate(hidden).astype(bfloat16), dtype=bfloat16)
    weights = iron.tensor(packed_weights[0], dtype=bfloat16)
    aux1 = iron.tensor(packed_weights[1][-4096:].copy(), dtype=bfloat16)
    output = iron.zeros((6144,), dtype=bfloat16, device="npu")

    def run_pair():
        attention_prefix_pair(pair, weights, aux1, output)

    run_pair()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        run_pair()
        samples.append((time.perf_counter() - start) * 1000)
    reference = []
    for pos in range(2):
        x = iron.tensor(hidden[pos], dtype=bfloat16)
        w = iron.tensor(packed_weights[pos], dtype=bfloat16)
        y = iron.zeros((3072,), dtype=bfloat16, device="npu")
        attention_prefix(x, w, y, include_hidden=True)
        reference.append(y.numpy().astype(np.float32))
    actual = output.numpy().astype(np.float32).reshape(2, 3072)
    errors = [float(np.max(np.abs(actual[pos] - reference[pos]))) for pos in range(2)]
    print(json.dumps({"device": "Phoenix NPU1", "layer": 2,
                      "median_ms": statistics.median(samples), "runs_ms": samples,
                      "max_abs_error_vs_single_prefix": errors}, indent=2))
    if max(errors) != 0:
        raise AssertionError("Paired prefix did not match established NPU prefix")


if __name__ == "__main__":
    main()
