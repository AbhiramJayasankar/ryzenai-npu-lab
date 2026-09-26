"""Check NPU final norm, argmax, and winning tied embedding row."""

import argparse
import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_checkpoint import BF16Checkpoint
from lfm25_fused_vocab_kernel import fused_vocab
from lfm25_fused_vocab_4core_kernel import fused_vocab_4core
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def probe_shards(kernel):
    """Force each shard to win once, including both ends of the peer chain."""
    packed = np.zeros((1024 + 65536 * 1024,), dtype=bfloat16)
    packed[:1024] = 1
    targets = [core * 16384 + 100 + core for core in range(4)]
    for core, token in enumerate(targets):
        packed[1024 + token * 1024 + core] = 1
    weights = iron.tensor(packed, dtype=bfloat16)
    next_embedding = iron.zeros((1024,), dtype=bfloat16, device="npu")
    next_id = iron.zeros((1,), dtype=np.int32, device="npu")
    results = []
    for core, token in enumerate(targets):
        basis = np.zeros((1024,), dtype=bfloat16)
        basis[core] = 1
        hidden = iron.tensor(basis, dtype=bfloat16)
        kernel(hidden, weights, next_embedding, next_id)
        actual_row = next_embedding.numpy().astype(np.float32)
        actual_id = int(next_id.numpy()[0])
        results.append({"winning_core": core, "expected_token": token,
                        "npu_token": actual_id,
                        "embedding_max_abs_error": float(np.max(np.abs(actual_row - basis.astype(np.float32))))})
    print(json.dumps({"device": "Phoenix NPU1", "shard_probes": results}, indent=2))
    if any(item["npu_token"] != item["expected_token"] or item["embedding_max_abs_error"] != 0
           for item in results):
        raise RuntimeError("A shard winner did not survive the NPU candidate chain")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, choices=(1, 4), default=1)
    parser.add_argument("--probe-shards", action="store_true")
    args = parser.parse_args()
    kernel = fused_vocab if args.cores == 1 else fused_vocab_4core
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    if args.probe_shards:
        if args.cores != 4:
            parser.error("--probe-shards requires --cores 4")
        probe_shards(kernel)
        return
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
    kernel(hidden, weights, next_embedding, next_id)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        kernel(hidden, weights, next_embedding, next_id)
        times.append((time.perf_counter() - start) * 1000)
    actual = next_embedding.numpy().astype(np.float32)
    result = {
        "device": "Phoenix NPU1",
        "operation": "fused final norm, vocabulary argmax, next embedding",
        "cores": args.cores,
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
