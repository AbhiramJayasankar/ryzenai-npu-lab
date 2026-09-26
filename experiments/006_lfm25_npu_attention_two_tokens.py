"""Check two attention decode tokens with NPU-produced KV state."""

import json
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_attention_context_cache_kernel import attention_context_cache
from lfm25_attention_prefix_kernel import attention_prefix
from lfm25_attention_tail_kernel import attention_tail
from lfm25_checkpoint import BF16Checkpoint, attention_tail_data, pack_attention_tail_weights
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def prefix_weights(step):
    suffix = "" if step == 1 else f"-step{step}"
    with np.load(ROOT / "cache" / f"lfm25-attention2{suffix}-reference.npz") as reference:
        gamma = reference["operator_gamma"].reshape(-1).astype(bfloat16)
        matrices = [reference[f"{name}_weight"].reshape(-1).astype(bfloat16) for name in ("q", "k", "v")]
        aux = np.concatenate([
            reference[name].reshape(-1) for name in ("q_gamma", "k_gamma", "cos", "sin")
        ]).astype(bfloat16)
    return np.concatenate([
        np.pad(gamma, (0, 4096 - gamma.size)), *matrices,
        np.pad(aux, (0, 4096 - aux.size)),
    ]).astype(bfloat16)


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    checkpoint = BF16Checkpoint(ROOT / "cache" / "lfm25-230m" / "model.safetensors")
    tail_weight = iron.tensor(
        pack_attention_tail_weights(attention_tail_data(checkpoint, 2)),
        dtype=bfloat16,
    )
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as reference:
        hidden1 = iron.tensor(reference["step1_hidden2"].reshape(-1).astype(bfloat16), dtype=bfloat16)
        hidden2 = iron.tensor(reference["step2_hidden2"].reshape(-1).astype(bfloat16), dtype=bfloat16)
        prior_keys = reference["step0_attn2_keys"].reshape(8, 21 * 64).astype(bfloat16)
        prior_values = reference["step0_attn2_values"].reshape(8, 21 * 64).astype(bfloat16)
        expected_first_cache = np.stack([
            reference["step1_attn2_keys"].reshape(8, 22 * 64),
            reference["step1_attn2_values"].reshape(8, 22 * 64),
        ], axis=1).reshape(-1)
        expected_second_cache = np.stack([
            reference["step2_attn2_keys"].reshape(8, 23 * 64),
            reference["step2_attn2_values"].reshape(8, 23 * 64),
        ], axis=1).reshape(-1)
        expected_first_hidden = reference["step1_hidden3"].reshape(-1)
        expected_second_hidden = reference["step2_hidden3"].reshape(-1)
    old_cache = iron.tensor(np.stack([prior_keys, prior_values], axis=1).reshape(-1), dtype=bfloat16)
    first_weight = iron.tensor(prefix_weights(1), dtype=bfloat16)
    second_weight = iron.tensor(prefix_weights(2), dtype=bfloat16)
    qkv1 = iron.zeros((3072,), dtype=bfloat16, device="npu")
    qkv2 = iron.zeros((3072,), dtype=bfloat16, device="npu")
    tail1 = iron.zeros((2048,), dtype=bfloat16, device="npu")
    tail2 = iron.zeros((2048,), dtype=bfloat16, device="npu")
    output1 = iron.zeros((1024,), dtype=bfloat16, device="npu")
    output2 = iron.zeros((1024,), dtype=bfloat16, device="npu")
    cache1 = iron.zeros((8 * 2 * 22 * 64,), dtype=bfloat16, device="npu")
    cache2 = iron.zeros((8 * 2 * 23 * 64,), dtype=bfloat16, device="npu")

    attention_prefix(hidden1, first_weight, qkv1, include_hidden=True)
    attention_context_cache(qkv1, old_cache, tail1, cache1, past_length=21)
    attention_tail(tail1, tail_weight, output1)
    attention_prefix(hidden2, second_weight, qkv2, include_hidden=True)
    attention_context_cache(qkv2, cache1, tail2, cache2, past_length=22)
    attention_tail(tail2, tail_weight, output2)
    actual_first = output1.numpy().astype(np.float32)
    actual_second = output2.numpy().astype(np.float32)
    result = {
        "device": "Phoenix NPU1",
        "operation": "two attention decode tokens with NPU-produced prior KV cache",
        "first_hidden_max_abs_error": float(np.max(np.abs(actual_first - expected_first_hidden))),
        "second_hidden_max_abs_error": float(np.max(np.abs(actual_second - expected_second_hidden))),
        "first_cache_max_abs_error": float(np.max(np.abs(cache1.numpy().astype(np.float32) - expected_first_cache))),
        "second_cache_max_abs_error": float(np.max(np.abs(cache2.numpy().astype(np.float32) - expected_second_cache))),
    }
    print(json.dumps(result, indent=2))
    if result["first_hidden_max_abs_error"] > 0.015625 or result["second_hidden_max_abs_error"] > 0.015625:
        raise RuntimeError("Attention block exceeded BF16 tolerance")
    if result["first_cache_max_abs_error"] != 0 or result["second_cache_max_abs_error"] != 0:
        raise RuntimeError("NPU-generated KV cache differs from CPU reference")


if __name__ == "__main__":
    main()
