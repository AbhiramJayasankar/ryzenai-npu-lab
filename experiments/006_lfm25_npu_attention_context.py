"""Run LFM2.5 attention Q/K/V and cached-token attention on Phoenix NPU."""

import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_attention_append_cache_kernel import append_attention_cache
from lfm25_attention_context_kernel import attention_context
from lfm25_attention_prefix_kernel import attention_prefix
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    with np.load(ROOT / "cache" / "lfm25-attention2-reference.npz") as ref:
        hidden = ref["hidden"].reshape(-1).astype(bfloat16)
        gamma = ref["operator_gamma"].reshape(-1).astype(bfloat16)
        matrices = [ref[f"{name}_weight"].reshape(-1).astype(bfloat16) for name in ("q", "k", "v")]
        aux = np.concatenate([ref[name].reshape(-1) for name in ("q_gamma", "k_gamma", "cos", "sin")]).astype(bfloat16)
    weights = np.concatenate([
        np.pad(gamma, (0, 4096 - gamma.size)), *matrices,
        np.pad(aux, (0, 4096 - aux.size)),
    ]).astype(bfloat16)
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as ref:
        past_keys = ref["step0_attn2_keys"].reshape(-1).astype(bfloat16)
        past_values = ref["step0_attn2_values"].reshape(-1).astype(bfloat16)
        expected = ref["step1_attn2_context"].reshape(-1).astype(np.float32)
        expected_key = ref["step1_attn2_keys"][:, :, -1, :].reshape(-1).astype(np.float32)
        expected_value = ref["step1_attn2_values"][:, :, -1, :].reshape(-1).astype(np.float32)
        expected_cache = np.stack(
            [ref["step1_attn2_keys"].reshape(8, 22 * 64),
             ref["step1_attn2_values"].reshape(8, 22 * 64)],
            axis=1,
        ).reshape(-1).astype(np.float32)
    assert past_keys.size == past_values.size == 8 * 21 * 64
    hidden_tensor = iron.tensor(hidden, dtype=bfloat16)
    weight_tensor = iron.tensor(weights, dtype=bfloat16)
    packed_cache = np.stack(
        [past_keys.reshape(8, 21 * 64), past_values.reshape(8, 21 * 64)],
        axis=1,
    ).reshape(-1)
    cache_tensor = iron.tensor(packed_cache, dtype=bfloat16)
    qkv_tensor = iron.zeros((2048,), dtype=bfloat16, device="npu")
    context_tensor = iron.zeros((1024,), dtype=bfloat16, device="npu")
    next_cache_tensor = iron.zeros((8 * 2 * 22 * 64,), dtype=bfloat16, device="npu")
    attention_prefix(hidden_tensor, weight_tensor, qkv_tensor)
    attention_context(qkv_tensor, cache_tensor, context_tensor)
    append_attention_cache(qkv_tensor, cache_tensor, next_cache_tensor)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        attention_context(qkv_tensor, cache_tensor, context_tensor)
        times.append((time.perf_counter() - start) * 1000)
    append_times = []
    for _ in range(5):
        start = time.perf_counter()
        append_attention_cache(qkv_tensor, cache_tensor, next_cache_tensor)
        append_times.append((time.perf_counter() - start) * 1000)
    qkv = qkv_tensor.numpy().astype(np.float32)
    actual = context_tensor.numpy().astype(np.float32)
    actual_cache = next_cache_tensor.numpy().astype(np.float32)
    result = {
        "device": "Phoenix NPU1",
        "operation": "attention layer 2 cached score, softmax, value mixing",
        "past_tokens": 21,
        "context_median_ms": statistics.median(times),
        "cache_append_median_ms": statistics.median(append_times),
        "new_key_max_abs_error": float(np.max(np.abs(qkv[1024:1536] - expected_key))),
        "new_value_max_abs_error": float(np.max(np.abs(qkv[1536:] - expected_value))),
        "context_max_abs_error": float(np.max(np.abs(actual - expected))),
        "context_exact_fraction": float(np.mean(actual == expected)),
        "next_cache_max_abs_error": float(np.max(np.abs(actual_cache - expected_cache))),
    }
    print(json.dumps(result, indent=2))
    if result["new_key_max_abs_error"] != 0 or result["new_value_max_abs_error"] != 0:
        raise RuntimeError("New NPU key/value differed from CPU cache")
    if result["context_max_abs_error"] > 0.001:
        raise RuntimeError("Attention context exceeded BF16 tolerance")
    if result["next_cache_max_abs_error"] != 0:
        raise RuntimeError("NPU-updated KV cache differed from CPU reference")


if __name__ == "__main__":
    main()
