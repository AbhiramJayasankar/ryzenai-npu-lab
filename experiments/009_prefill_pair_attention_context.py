"""Check two causal attention positions and KV cache in one NPU dispatch."""

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
from lfm25_prefill_pair_attention_context_kernel import attention_pair_context_fixed64


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    module = importlib.import_module("007_lfm25_npu_chat")
    chat = module.NPUChat("fixed64")
    layer = chat.layers[2]
    with np.load(ROOT / "cache/lfm25-prompt-sequence-reference.npz") as ref:
        hidden = [ref[f"p{pos}_hidden2"].astype(bfloat16) for pos in range(2)]
    qkvs = []
    for pos in range(2):
        weights = layer["prefix_base"].copy()
        cos, sin = module.rotary(pos)
        weights[-4096 + 128:-4096 + 192] = cos
        weights[-4096 + 192:-4096 + 256] = sin
        qkv = iron.zeros((3072,), dtype=bfloat16, device="npu")
        attention_prefix(iron.tensor(hidden[pos], dtype=bfloat16),
                         iron.tensor(weights, dtype=bfloat16), qkv,
                         include_hidden=True)
        qkvs.append(qkv)
    zeros = iron.zeros((CACHE_ELEMENTS,), dtype=bfloat16, device="npu")
    qkv_pair = iron.tensor(np.concatenate([qkv.numpy() for qkv in qkvs]),
                           dtype=bfloat16)
    packed_pair = iron.zeros((4096,), dtype=bfloat16, device="npu")
    next_cache = iron.zeros((CACHE_ELEMENTS,), dtype=bfloat16, device="npu")

    def run_pair():
        attention_pair_context_fixed64(qkv_pair, zeros, packed_pair, next_cache)

    run_pair()
    elapsed = []
    for _ in range(5):
        start = time.perf_counter()
        run_pair()
        elapsed.append((time.perf_counter() - start) * 1000)
    packed0 = iron.zeros((2048,), dtype=bfloat16, device="npu")
    packed1 = iron.zeros((2048,), dtype=bfloat16, device="npu")
    cache0 = iron.zeros((CACHE_ELEMENTS,), dtype=bfloat16, device="npu")
    cache1 = iron.zeros((CACHE_ELEMENTS,), dtype=bfloat16, device="npu")
    attention_first_fixed64(qkvs[0], packed0, cache0)
    attention_context_fixed64(qkvs[1], cache0, packed1, cache1)
    actual = packed_pair.numpy().astype(np.float32).reshape(2, 2048)
    expected = np.stack([packed0.numpy(), packed1.numpy()]).astype(np.float32)
    cache_error = float(np.max(np.abs(
        next_cache.numpy().astype(np.float32) - cache1.numpy().astype(np.float32)
    )))
    result = {"device": "Phoenix NPU1", "layer": 2,
              "median_ms": statistics.median(elapsed), "runs_ms": elapsed,
              "tail_input_max_abs_error": float(np.max(np.abs(actual - expected))),
              "cache_max_abs_error": cache_error}
    print(json.dumps(result, indent=2))
    if result["tail_input_max_abs_error"] or result["cache_max_abs_error"]:
        raise AssertionError("Pair attention context did not match two NPU calls")


if __name__ == "__main__":
    main()
