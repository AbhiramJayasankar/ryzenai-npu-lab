"""Verify XRT tensor row views select embeddings without scanning the table."""

import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16

from lfm25_checkpoint import BF16Checkpoint
from lfm25_prefill_pair_data_kernel import concat_hidden_pair


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    values = BF16Checkpoint(
        ROOT / "cache/lfm25-230m/model.safetensors"
    ).load("model.embed_tokens.weight")
    table = iron.tensor(values, dtype=bfloat16)
    if type(table).__name__ != "XRTTensor":
        raise RuntimeError("Expected XRT-backed embedding table")
    ids = (17, 2797)
    views = [table.subview(token_id * 1024 * 2, (1024,)) for token_id in ids]
    pair = iron.zeros((2048,), dtype=bfloat16, device="npu")

    def run():
        concat_hidden_pair(views[0], views[1], pair)

    run()
    actual = pair.numpy().astype(np.float32).reshape(2, 1024)
    expected = values[list(ids)].astype(np.float32)
    if not np.array_equal(actual, expected):
        raise AssertionError("XRT row views selected the wrong embeddings")
    elapsed = []
    for _ in range(10):
        start = time.perf_counter()
        run()
        elapsed.append((time.perf_counter() - start) * 1000)
    print(json.dumps({"device": "Phoenix NPU1", "ids": ids,
                      "row_views_are_shared": all(view.base is table for view in views),
                      "exact": True, "samples_ms": elapsed,
                      "median_ms": statistics.median(elapsed)}, indent=2))


if __name__ == "__main__":
    main()
