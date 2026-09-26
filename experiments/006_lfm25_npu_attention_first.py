"""Validate NPU attention layer 2 for the first token of an empty prompt."""

import argparse
import json
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_attention_first_context_kernel import attention_first_context
from lfm25_attention_prefix_kernel import attention_prefix
from lfm25_attention_tail_kernel import attention_tail
from lfm25_checkpoint import BF16Checkpoint, attention_tail_data, pack_attention_tail_weights
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def first_position_prefix_weights(layer):
    with np.load(ROOT / "cache" / f"lfm25-attention{layer}-reference.npz") as ref:
        gamma = ref["operator_gamma"].reshape(-1).astype(bfloat16)
        matrices = [ref[f"{name}_weight"].reshape(-1).astype(bfloat16)
                    for name in ("q", "k", "v")]
        aux = np.concatenate([
            ref["q_gamma"].reshape(-1), ref["k_gamma"].reshape(-1),
            np.ones((64,), dtype=np.float32), np.zeros((64,), dtype=np.float32),
        ]).astype(bfloat16)
    return np.concatenate([
        np.pad(gamma, (0, 4096 - gamma.size)), *matrices,
        np.pad(aux, (0, 4096 - aux.size)),
    ]).astype(bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, choices=(2, 4, 6, 8, 10, 12), default=2)
    args = parser.parse_args()
    layer = args.layer
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    with np.load(ROOT / "cache" / "lfm25-prompt-first-reference.npz") as ref:
        initial = ref[f"hidden{layer}"].reshape(-1).astype(bfloat16)
        expected = ref[f"hidden{layer + 1}"].reshape(-1)
        keys = ref[f"attn{layer}_keys"].reshape(8, 64)
        values = ref[f"attn{layer}_values"].reshape(8, 64)
        expected_cache = np.stack([keys, values], axis=1).reshape(-1)
    checkpoint = BF16Checkpoint(ROOT / "cache" / "lfm25-230m" / "model.safetensors")
    prefix = first_position_prefix_weights(layer)
    tail = pack_attention_tail_weights(attention_tail_data(checkpoint, layer))
    hidden = iron.tensor(initial, dtype=bfloat16)
    prefix_weights = iron.tensor(prefix, dtype=bfloat16)
    tail_weights = iron.tensor(tail, dtype=bfloat16)
    qkv = iron.zeros((3072,), dtype=bfloat16, device="npu")
    packed = iron.zeros((2048,), dtype=bfloat16, device="npu")
    cache = iron.zeros((1024,), dtype=bfloat16, device="npu")
    output = iron.zeros((1024,), dtype=bfloat16, device="npu")
    attention_prefix(hidden, prefix_weights, qkv, include_hidden=True)
    attention_first_context(qkv, packed, cache)
    attention_tail(packed, tail_weights, output)
    actual_cache = cache.numpy().astype(np.float32)
    actual_output = output.numpy().astype(np.float32)
    result = {
        "device": "Phoenix NPU1",
        "operation": f"first prompt token through attention layer {layer}",
        "cache_max_abs_error": float(np.max(np.abs(actual_cache - expected_cache))),
        "hidden_max_abs_error": float(np.max(np.abs(actual_output - expected))),
    }
    print(json.dumps(result, indent=2))
    if result["cache_max_abs_error"] > 0.125 or result["hidden_max_abs_error"] > 0.125:
        raise RuntimeError("First-token attention diverged from CPU BF16 reference")


if __name__ == "__main__":
    main()
