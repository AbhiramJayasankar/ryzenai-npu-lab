"""Run two consecutive LFM2.5 recurrent layers on Phoenix NPU."""

import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_checkpoint import BF16Checkpoint, pack_recurrent_weights, recurrent_layer_data
from lfm25_pack_block_input_kernel import pack_block_input
from lfm25_single_program_block_kernel import recurrent_block
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    checkpoint = BF16Checkpoint(ROOT / "cache" / "lfm25-230m" / "model.safetensors")
    layer0 = recurrent_layer_data(checkpoint, 0)
    layer1 = recurrent_layer_data(checkpoint, 1)
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as ref:
        hidden0 = ref["step1_hidden0"].astype(bfloat16)
        hidden0_next = ref["step2_hidden0"].astype(bfloat16)
        hidden1_next_reference = ref["step2_hidden1"].astype(bfloat16)
        state0 = ref["step0_conv0_state"].reshape(-1).astype(bfloat16)
        state1 = ref["step0_conv1_state"].reshape(-1).astype(bfloat16)
        expected_hidden1 = ref["step1_hidden1"]
        expected_hidden2 = ref["step1_hidden2"]
        expected_state0 = ref["step1_conv0_state"].reshape(-1)
        expected_state1 = ref["step1_conv1_state"].reshape(-1)
        expected_hidden1_next = ref["step2_hidden1"]
        expected_hidden2_next = ref["step2_hidden2"]
        expected_state0_next = ref["step2_conv0_state"].reshape(-1)
        expected_state1_next = ref["step2_conv1_state"].reshape(-1)
    packed0 = np.concatenate(
        [np.pad(hidden0, (0, 6144 - hidden0.size)), state0, layer0["conv_weight"].reshape(-1)]
    ).astype(bfloat16)
    packed_state1 = np.concatenate([state1, layer1["conv_weight"].reshape(-1)]).astype(bfloat16)
    input0 = iron.tensor(packed0, dtype=bfloat16)
    weights0 = iron.tensor(pack_recurrent_weights(layer0), dtype=bfloat16)
    state1_initial = iron.tensor(packed_state1, dtype=bfloat16)
    weights1 = iron.tensor(pack_recurrent_weights(layer1), dtype=bfloat16)
    output0 = iron.zeros((1024,), dtype=bfloat16, device="npu")
    state0_next = iron.zeros((6144,), dtype=bfloat16, device="npu")
    input1 = iron.zeros((12288,), dtype=bfloat16, device="npu")
    output1 = iron.zeros((1024,), dtype=bfloat16, device="npu")
    state1_next = iron.zeros((6144,), dtype=bfloat16, device="npu")

    recurrent_block(input0, weights0, state0_next, output0)
    pack_block_input(output0, state1_initial, input1)

    def run_second_layer():
        recurrent_block(input1, weights1, state1_next, output1)

    run_second_layer()
    input0_next = iron.zeros((12288,), dtype=bfloat16, device="npu")
    output0_next = iron.zeros((1024,), dtype=bfloat16, device="npu")
    state0_after_next = iron.zeros((6144,), dtype=bfloat16, device="npu")
    input1_next = iron.zeros((12288,), dtype=bfloat16, device="npu")
    output1_next = iron.zeros((1024,), dtype=bfloat16, device="npu")
    state1_after_next = iron.zeros((6144,), dtype=bfloat16, device="npu")
    pack_block_input(iron.tensor(hidden0_next, dtype=bfloat16), state0_next, input0_next)
    recurrent_block(input0_next, weights0, state0_after_next, output0_next)
    pack_block_input(output0_next, state1_next, input1_next)
    recurrent_block(input1_next, weights1, state1_after_next, output1_next)
    input1_next_isolated = iron.zeros((12288,), dtype=bfloat16, device="npu")
    output1_next_isolated = iron.zeros((1024,), dtype=bfloat16, device="npu")
    state1_after_next_isolated = iron.zeros((6144,), dtype=bfloat16, device="npu")
    pack_block_input(iron.tensor(hidden1_next_reference, dtype=bfloat16), state1_next, input1_next_isolated)
    recurrent_block(input1_next_isolated, weights1, state1_after_next_isolated, output1_next_isolated)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        run_second_layer()
        times.append((time.perf_counter() - start) * 1000)
    chain_times = []
    for _ in range(5):
        start = time.perf_counter()
        recurrent_block(input0, weights0, state0_next, output0)
        pack_block_input(output0, state1_initial, input1)
        run_second_layer()
        chain_times.append((time.perf_counter() - start) * 1000)
    actual_hidden1 = output0.numpy().astype(np.float32)
    actual_hidden2 = output1.numpy().astype(np.float32)
    actual_state0 = state0_next.numpy().astype(np.float32)
    actual_state1 = state1_next.numpy().astype(np.float32)
    actual_hidden1_next = output0_next.numpy().astype(np.float32)
    actual_hidden2_next = output1_next.numpy().astype(np.float32)
    actual_state0_next = state0_after_next.numpy().astype(np.float32)
    actual_state1_next = state1_after_next.numpy().astype(np.float32)
    actual_hidden2_next_isolated = output1_next_isolated.numpy().astype(np.float32)
    actual_state1_next_isolated = state1_after_next_isolated.numpy().astype(np.float32)
    result = {
        "device": "Phoenix NPU1",
        "operation": "two successive recurrent layers over two decode tokens",
        "second_layer_median_ms": statistics.median(times),
        "two_layer_chain_median_ms": statistics.median(chain_times),
        "layer0_exact_fraction": float(np.mean(actual_hidden1 == expected_hidden1)),
        "layer0_max_abs_error": float(np.max(np.abs(actual_hidden1 - expected_hidden1))),
        "layer1_exact_fraction": float(np.mean(actual_hidden2 == expected_hidden2)),
        "layer1_max_abs_error": float(np.max(np.abs(actual_hidden2 - expected_hidden2))),
        "layer0_state_max_abs_error": float(
            np.max(np.abs(actual_state0[:3072] - expected_state0))
        ),
        "layer1_state_max_abs_error": float(
            np.max(np.abs(actual_state1[:3072] - expected_state1))
        ),
        "next_token_layer0_exact_fraction": float(np.mean(actual_hidden1_next == expected_hidden1_next)),
        "next_token_layer0_max_abs_error": float(np.max(np.abs(actual_hidden1_next - expected_hidden1_next))),
        "next_token_layer1_exact_fraction": float(np.mean(actual_hidden2_next == expected_hidden2_next)),
        "next_token_layer1_max_abs_error": float(np.max(np.abs(actual_hidden2_next - expected_hidden2_next))),
        "next_token_layer0_state_max_abs_error": float(np.max(np.abs(actual_state0_next[:3072] - expected_state0_next))),
        "next_token_layer1_state_max_abs_error": float(np.max(np.abs(actual_state1_next[:3072] - expected_state1_next))),
        "next_token_layer1_isolated_exact_fraction": float(np.mean(actual_hidden2_next_isolated == expected_hidden2_next)),
        "next_token_layer1_isolated_max_abs_error": float(np.max(np.abs(actual_hidden2_next_isolated - expected_hidden2_next))),
        "next_token_layer1_isolated_state_max_abs_error": float(np.max(np.abs(actual_state1_next_isolated[:3072] - expected_state1_next))),
    }
    print(json.dumps(result, indent=2))
    if result["layer1_max_abs_error"] > 0.015625:
        raise RuntimeError("Second recurrent layer exceeded BF16 tolerance")
    if result["layer0_state_max_abs_error"] != 0 or result["layer1_state_max_abs_error"] != 0:
        raise RuntimeError("Recurrent state differs from CPU reference")
    if result["next_token_layer1_max_abs_error"] > 0.015625:
        raise RuntimeError("Second token exceeded BF16 tolerance")
    if result["next_token_layer0_state_max_abs_error"] != 0 or result["next_token_layer1_isolated_state_max_abs_error"] != 0:
        raise RuntimeError("Second token isolated recurrent state differs from CPU reference")


if __name__ == "__main__":
    main()
