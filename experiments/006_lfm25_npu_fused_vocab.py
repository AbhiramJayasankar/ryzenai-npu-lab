"""Check NPU final norm, argmax, and winning tied embedding row."""

import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_checkpoint import BF16Checkpoint
from lfm25_fused_vocab_kernel import fused_vocab
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    checkpoint = BF16Checkpoint(ROOT / "cache" / "lfm25-230m" / "model.safetensors")
    gamma = checkpoint.load("model.embedding_norm.weight")
    embedding_table = checkpoint.load("model.embed_tokens.weight")
    packed_weights = np.concatenate([gamma.reshape(-1), embedding_table.reshape(-1)]).astype(bfloat16)
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as reference:
        raw_hidden = reference["step1_hidden14_raw"].reshape(-1).astype(bfloat16)
        expected_id = int(np.argmax(reference["step1_logits"].reshape(-1)))
        expected_embedding = reference["step2_hidden0"].reshape(-1)
    hidden = iron.tensor(raw_hidden, dtype=bfloat16)
    weights = iron.tensor(packed_weights, dtype=bfloat16)
    next_embedding = iron.zeros((1024,), dtype=bfloat16, device="npu")
    next_id = iron.zeros((1,), dtype=np.int32, device="npu")
    fused_vocab(hidden, weights, next_embedding, next_id)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        fused_vocab(hidden, weights, next_embedding, next_id)
        times.append((time.perf_counter() - start) * 1000)
    actual = next_embedding.numpy().astype(np.float32)
    result = {
        "device": "Phoenix NPU1",
        "operation": "fused final norm, vocabulary argmax, next embedding",
        "median_ms": statistics.median(times),
        "npu_token": int(next_id.numpy()[0]),
        "cpu_token": expected_id,
        "embedding_max_abs_error": float(np.max(np.abs(actual - expected_embedding))),
    }
    print(json.dumps(result, indent=2))
    if result["npu_token"] != expected_id or result["embedding_max_abs_error"] != 0:
        raise RuntimeError("Fused output head disagrees with CPU reference")


if __name__ == "__main__":
    main()
