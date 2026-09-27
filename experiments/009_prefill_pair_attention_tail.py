"""Verify two real attention-tail positions with one reused-weight NPU program."""

import importlib
import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16

from lfm25_attention_fixed64_cache_kernel import (
    CACHE_ELEMENTS, attention_context_fixed64, attention_first_fixed64,
)
from lfm25_attention_prefix_kernel import attention_prefix
from lfm25_prefill_pair_attention_tail_kernel import attention_tail_pair


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    chat_module = importlib.import_module("007_lfm25_npu_chat")
    chat = chat_module.NPUChat("fixed64")
    layer = chat.layers[2]
    with np.load(ROOT / "cache/lfm25-prompt-sequence-reference.npz") as ref:
        inputs = [ref[f"p{position}_hidden2"].astype(bfloat16)
                  for position in range(2)]
        expected = [ref[f"p{position}_hidden3"].reshape(-1)
                    for position in range(2)]
    packed_pair = iron.zeros((4096,), dtype=bfloat16, device="npu")
    cache = None
    for position in range(2):
        weights = layer["prefix_base"].copy()
        cos, sin = chat_module.rotary(position)
        weights[-4096 + 128:-4096 + 192] = cos
        weights[-4096 + 192:-4096 + 256] = sin
        hidden = iron.tensor(inputs[position], dtype=bfloat16)
        prefix_weights = iron.tensor(weights, dtype=bfloat16)
        qkv = iron.zeros((3072,), dtype=bfloat16, device="npu")
        packed = packed_pair.subview(position * 4096, (2048,))
        next_cache = iron.zeros((CACHE_ELEMENTS,), dtype=bfloat16, device="npu")
        attention_prefix(hidden, prefix_weights, qkv, include_hidden=True)
        if position == 0:
            attention_first_fixed64(qkv, packed, next_cache)
        else:
            attention_context_fixed64(qkv, cache, packed, next_cache)
        cache = next_cache
    output = iron.zeros((2048,), dtype=bfloat16, device="npu")

    def run():
        attention_tail_pair(packed_pair, layer["tail_weights"], output)

    run()
    elapsed = []
    for _ in range(5):
        start = time.perf_counter()
        run()
        elapsed.append((time.perf_counter() - start) * 1000)
    actual = output.numpy().astype(np.float32).reshape(2, 1024)
    result = {
        "device": "Phoenix NPU1", "layer": 2,
        "operation": "two full attention output+FFN positions in one program",
        "median_ms": statistics.median(elapsed), "runs_ms": elapsed,
        "position0_max_abs_error": float(np.max(np.abs(actual[0] - expected[0]))),
        "position1_max_abs_error": float(np.max(np.abs(actual[1] - expected[1]))),
    }
    print(json.dumps(result, indent=2))
    if max(result["position0_max_abs_error"], result["position1_max_abs_error"]) > 0.015625:
        raise AssertionError("Paired attention tail exceeded BF16 CPU tolerance")


if __name__ == "__main__":
    main()
