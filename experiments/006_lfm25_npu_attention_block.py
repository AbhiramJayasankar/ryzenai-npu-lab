"""Run an entire layer-2 attention decode block on Phoenix NPU."""

import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_attention_append_cache_kernel import append_attention_cache
from lfm25_attention_context_kernel import attention_context
from lfm25_attention_prefix_kernel import attention_prefix
from lfm25_attention_tail_kernel import attention_tail
from lfm25_checkpoint import BF16Checkpoint, attention_tail_data, pack_attention_tail_weights
from lfm25_pack_attention_tail_kernel import pack_attention_tail
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
    prefix_weights = np.concatenate([
        np.pad(gamma, (0, 4096 - gamma.size)), *matrices,
        np.pad(aux, (0, 4096 - aux.size)),
    ]).astype(bfloat16)
    checkpoint = BF16Checkpoint(ROOT / "cache" / "lfm25-230m" / "model.safetensors")
    tail_weights = pack_attention_tail_weights(attention_tail_data(checkpoint, 2))
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as ref:
        keys = ref["step0_attn2_keys"].reshape(8, 21 * 64).astype(bfloat16)
        values = ref["step0_attn2_values"].reshape(8, 21 * 64).astype(bfloat16)
        expected_hidden = ref["step1_hidden3"].reshape(-1).astype(np.float32)
        expected_cache = np.stack([
            ref["step1_attn2_keys"].reshape(8, 22 * 64),
            ref["step1_attn2_values"].reshape(8, 22 * 64),
        ], axis=1).reshape(-1).astype(np.float32)
        cpu_context = ref["step1_attn2_context"].reshape(-1).astype(bfloat16)
    cache = np.stack([keys, values], axis=1).reshape(-1)
    h = iron.tensor(hidden, dtype=bfloat16)
    prefix_w = iron.tensor(prefix_weights, dtype=bfloat16)
    tail_w = iron.tensor(tail_weights, dtype=bfloat16)
    old_cache = iron.tensor(cache, dtype=bfloat16)
    reference_context = iron.tensor(cpu_context, dtype=bfloat16)
    qkv = iron.zeros((2048,), dtype=bfloat16, device="npu")
    context = iron.zeros((1024,), dtype=bfloat16, device="npu")
    packed = iron.zeros((2048,), dtype=bfloat16, device="npu")
    isolated_packed = iron.zeros((2048,), dtype=bfloat16, device="npu")
    output = iron.zeros((1024,), dtype=bfloat16, device="npu")
    isolated_output = iron.zeros((1024,), dtype=bfloat16, device="npu")
    next_cache = iron.zeros((8 * 2 * 22 * 64,), dtype=bfloat16, device="npu")

    def full_block():
        attention_prefix(h, prefix_w, qkv)
        attention_context(qkv, old_cache, context)
        pack_attention_tail(h, context, packed)
        attention_tail(packed, tail_w, output)
        append_attention_cache(qkv, old_cache, next_cache)

    full_block()
    pack_attention_tail(h, reference_context, isolated_packed)
    attention_tail(isolated_packed, tail_w, isolated_output)
    tail_times = []
    for _ in range(5):
        start = time.perf_counter()
        attention_tail(packed, tail_w, output)
        tail_times.append((time.perf_counter() - start) * 1000)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        full_block()
        times.append((time.perf_counter() - start) * 1000)
    actual = output.numpy().astype(np.float32)
    isolated = isolated_output.numpy().astype(np.float32)
    actual_cache = next_cache.numpy().astype(np.float32)
    result = {
        "device": "Phoenix NPU1",
        "operation": "full attention layer 2 and KV append for one decode token",
        "attention_tail_median_ms": statistics.median(tail_times),
        "five_call_chain_median_ms": statistics.median(times),
        "hidden_max_abs_error": float(np.max(np.abs(actual - expected_hidden))),
        "hidden_exact_fraction": float(np.mean(actual == expected_hidden)),
        "isolated_tail_max_abs_error": float(np.max(np.abs(isolated - expected_hidden))),
        "isolated_tail_exact_fraction": float(np.mean(isolated == expected_hidden)),
        "next_cache_max_abs_error": float(np.max(np.abs(actual_cache - expected_cache))),
    }
    print(json.dumps(result, indent=2))
    if result["isolated_tail_max_abs_error"] > 0.015625:
        raise RuntimeError("Attention tail exceeded BF16 tolerance")
    if result["hidden_max_abs_error"] > 0.015625:
        raise RuntimeError("Full attention block exceeded BF16 tolerance")
    if result["next_cache_max_abs_error"] != 0:
        raise RuntimeError("Attention cache differs from CPU reference")


if __name__ == "__main__":
    main()
