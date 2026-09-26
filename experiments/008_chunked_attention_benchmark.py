"""Measure Phoenix NPU attention cost as the active KV history grows."""

import argparse
import json
import statistics
import time

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16

from lfm25_attention_chunked_cache_kernel import (
    BLOCK_ELEMENTS, HEAD_ELEMENTS, attention_context_chunked,
)


def run(blocks, repeats):
    qkv = np.zeros((3072,), dtype=bfloat16)
    qkv[1536:2048] = bfloat16(1)
    qkv_tensor = iron.tensor(qkv, dtype=bfloat16)
    cache = np.zeros((blocks, 8, HEAD_ELEMENTS), dtype=bfloat16)
    cache[:, :, 0] = bfloat16(64)
    cache[-1, :, 0] = bfloat16(63)
    old_tensor = iron.tensor(cache.reshape(-1), dtype=bfloat16)
    packed = iron.zeros((2048,), dtype=bfloat16, device="npu")
    next_cache = iron.zeros((blocks * BLOCK_ELEMENTS,), dtype=bfloat16, device="npu")
    attention_context_chunked.specialize(block_count=blocks).compile()
    elapsed = []
    for _ in range(repeats + 1):
        start = time.perf_counter()
        attention_context_chunked(qkv_tensor, old_tensor, packed, next_cache,
                                  block_count=blocks)
        values = packed.numpy()
        elapsed.append((time.perf_counter() - start) * 1000)
        if not np.isfinite(values.astype(np.float32)).all():
            raise AssertionError("Attention returned a non-finite value")
    updated = next_cache.numpy().reshape(blocks, 8, HEAD_ELEMENTS)
    if not np.all(updated[-1, :, 0] == bfloat16(64)):
        raise AssertionError("The final KV block did not append the new token")
    return {
        "blocks": blocks,
        "total_tokens": 64 * blocks,
        "cache_mebibytes_per_layer": blocks * BLOCK_ELEMENTS * 2 / (1024 * 1024),
        "warmed_median_ms_per_attention_layer": statistics.median(elapsed[1:]),
        "warmed_calls_ms": elapsed[1:],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    for blocks in args.blocks:
        print(json.dumps(run(blocks, args.repeats)), flush=True)


if __name__ == "__main__":
    main()
