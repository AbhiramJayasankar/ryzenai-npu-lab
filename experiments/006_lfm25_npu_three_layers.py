"""Chain LFM2.5 layers 0, 1, and 2 entirely through Phoenix NPU math."""

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
from lfm25_checkpoint import (
    BF16Checkpoint,
    attention_tail_data,
    pack_attention_tail_weights,
    pack_recurrent_weights,
    recurrent_layer_data,
)
from lfm25_pack_attention_tail_kernel import pack_attention_tail
from lfm25_pack_block_input_kernel import pack_block_input
from lfm25_single_program_block_kernel import recurrent_block
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    checkpoint = BF16Checkpoint(ROOT / "cache" / "lfm25-230m" / "model.safetensors")
    conv0 = recurrent_layer_data(checkpoint, 0)
    conv1 = recurrent_layer_data(checkpoint, 1)
    tail_weights = pack_attention_tail_weights(attention_tail_data(checkpoint, 2))
    with np.load(ROOT / "cache" / "lfm25-attention2-reference.npz") as ref:
        gamma = ref["operator_gamma"].reshape(-1).astype(bfloat16)
        matrices = [ref[f"{name}_weight"].reshape(-1).astype(bfloat16) for name in ("q", "k", "v")]
        aux = np.concatenate([ref[name].reshape(-1) for name in ("q_gamma", "k_gamma", "cos", "sin")]).astype(bfloat16)
    prefix_weights = np.concatenate([
        np.pad(gamma, (0, 4096 - gamma.size)), *matrices,
        np.pad(aux, (0, 4096 - aux.size)),
    ]).astype(bfloat16)
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as ref:
        hidden0 = ref["step1_hidden0"].reshape(-1).astype(bfloat16)
        state0 = ref["step0_conv0_state"].reshape(-1).astype(bfloat16)
        state1 = ref["step0_conv1_state"].reshape(-1).astype(bfloat16)
        expected_hidden1 = ref["step1_hidden1"].reshape(-1)
        expected_hidden2 = ref["step1_hidden2"].reshape(-1)
        expected_hidden3 = ref["step1_hidden3"].reshape(-1)
        old_keys = ref["step0_attn2_keys"].reshape(8, 21 * 64).astype(bfloat16)
        old_values = ref["step0_attn2_values"].reshape(8, 21 * 64).astype(bfloat16)
        expected_cache = np.stack([
            ref["step1_attn2_keys"].reshape(8, 22 * 64),
            ref["step1_attn2_values"].reshape(8, 22 * 64),
        ], axis=1).reshape(-1)
    packed0 = np.concatenate([
        np.pad(hidden0, (0, 6144 - hidden0.size)),
        state0,
        conv0["conv_weight"].reshape(-1),
    ]).astype(bfloat16)
    state1_and_weight = np.concatenate([
        state1, conv1["conv_weight"].reshape(-1),
    ]).astype(bfloat16)
    old_cache = np.stack([old_keys, old_values], axis=1).reshape(-1)

    input0 = iron.tensor(packed0, dtype=bfloat16)
    weights0 = iron.tensor(pack_recurrent_weights(conv0), dtype=bfloat16)
    weights1 = iron.tensor(pack_recurrent_weights(conv1), dtype=bfloat16)
    state1_initial = iron.tensor(state1_and_weight, dtype=bfloat16)
    prefix_w = iron.tensor(prefix_weights, dtype=bfloat16)
    tail_w = iron.tensor(tail_weights, dtype=bfloat16)
    cache = iron.tensor(old_cache, dtype=bfloat16)
    hidden1 = iron.zeros((1024,), dtype=bfloat16, device="npu")
    hidden2 = iron.zeros((1024,), dtype=bfloat16, device="npu")
    state0_next = iron.zeros((6144,), dtype=bfloat16, device="npu")
    state1_next = iron.zeros((6144,), dtype=bfloat16, device="npu")
    input1 = iron.zeros((12288,), dtype=bfloat16, device="npu")
    qkv = iron.zeros((2048,), dtype=bfloat16, device="npu")
    context = iron.zeros((1024,), dtype=bfloat16, device="npu")
    tail_input = iron.zeros((2048,), dtype=bfloat16, device="npu")
    hidden3 = iron.zeros((1024,), dtype=bfloat16, device="npu")
    next_cache = iron.zeros((8 * 2 * 22 * 64,), dtype=bfloat16, device="npu")

    def run_chain():
        timings = {}
        start = time.perf_counter()
        recurrent_block(input0, weights0, state0_next, hidden1)
        timings["recurrent0"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        pack_block_input(hidden1, state1_initial, input1)
        timings["pack_recurrent"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        recurrent_block(input1, weights1, state1_next, hidden2)
        timings["recurrent1"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        attention_prefix(hidden2, prefix_w, qkv)
        timings["attention_prefix"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        attention_context(qkv, cache, context)
        timings["attention_context"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        pack_attention_tail(hidden2, context, tail_input)
        timings["pack_attention"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        attention_tail(tail_input, tail_w, hidden3)
        timings["attention_tail"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        append_attention_cache(qkv, cache, next_cache)
        timings["append_cache"] = (time.perf_counter() - start) * 1000
        return timings

    start = time.perf_counter()
    run_chain()
    elapsed_ms = (time.perf_counter() - start) * 1000
    times = []
    stage_runs = []
    for _ in range(5):
        start = time.perf_counter()
        stage_runs.append(run_chain())
        times.append((time.perf_counter() - start) * 1000)
    actual1 = hidden1.numpy().astype(np.float32)
    actual2 = hidden2.numpy().astype(np.float32)
    actual3 = hidden3.numpy().astype(np.float32)
    actual_cache = next_cache.numpy().astype(np.float32)
    result = {
        "device": "Phoenix NPU1",
        "operation": "three consecutive LFM2.5 layers, 0/1 recurrent and 2 attention",
        "first_call_ms_including_compilation": elapsed_ms,
        "warmed_chain_median_ms": statistics.median(times),
        "stage_median_ms": {
            name: statistics.median(run[name] for run in stage_runs)
            for name in stage_runs[0]
        },
        "hidden1_max_abs_error": float(np.max(np.abs(actual1 - expected_hidden1))),
        "hidden2_max_abs_error": float(np.max(np.abs(actual2 - expected_hidden2))),
        "hidden3_max_abs_error": float(np.max(np.abs(actual3 - expected_hidden3))),
        "hidden3_exact_fraction": float(np.mean(actual3 == expected_hidden3)),
        "new_cache_max_abs_error": float(np.max(np.abs(actual_cache - expected_cache))),
    }
    print(json.dumps(result, indent=2))
    if max(result[f"hidden{i}_max_abs_error"] for i in (1, 2, 3)) > 0.015625:
        raise RuntimeError("NPU layer chain exceeded BF16 tolerance")
    if result["new_cache_max_abs_error"] != 0:
        raise RuntimeError("NPU KV cache differed from CPU reference")


if __name__ == "__main__":
    main()
