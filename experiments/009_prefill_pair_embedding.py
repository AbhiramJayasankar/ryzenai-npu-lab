"""Verify and time selecting two model embeddings in one NPU table pass."""

import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16

from lfm25_checkpoint import BF16Checkpoint
from lfm25_embedding_dynamic_kernel import embedding_dma_dynamic
from lfm25_prefill_pair_embedding_kernel import embedding_pair_dynamic


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    table_data = BF16Checkpoint(
        ROOT / "cache/lfm25-230m/model.safetensors"
    ).load("model.embed_tokens.weight")
    table = iron.tensor(table_data, dtype=bfloat16)
    ids = np.array([17, 2797], dtype=np.int32)
    ids_pair = iron.tensor(ids, dtype=np.int32)
    individual_ids = [iron.tensor(np.array([x], dtype=np.int32), dtype=np.int32)
                      for x in ids]
    pair = iron.zeros((2048,), dtype=bfloat16, device="npu")
    singles = [iron.zeros((1024,), dtype=bfloat16, device="npu") for _ in ids]

    def run_pair():
        embedding_pair_dynamic(table, pair, ids_pair)

    def run_singles():
        for out, token_id in zip(singles, individual_ids):
            embedding_dma_dynamic(table, out, token_id)

    run_pair()
    run_singles()
    observed = pair.numpy().astype(np.float32).reshape(2, 1024)
    expected = table_data[ids].astype(np.float32)
    if not np.array_equal(observed, expected):
        raise AssertionError("Two-token NPU lookup differs from model embedding table")
    if any(not np.array_equal(single.numpy().astype(np.float32), expected[i])
           for i, single in enumerate(singles)):
        raise AssertionError("Single-token NPU lookup differs from model embedding table")
    timings = {}
    for label, run in (("pair", run_pair), ("two_singles", run_singles)):
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            run()
            samples.append((time.perf_counter() - start) * 1000)
        timings[label] = {"samples_ms": samples, "median_ms": statistics.median(samples)}
    print(json.dumps({"device": "Phoenix NPU1", "token_ids": ids.tolist(),
                      "exact": True, "timings": timings}, indent=2))


if __name__ == "__main__":
    main()
