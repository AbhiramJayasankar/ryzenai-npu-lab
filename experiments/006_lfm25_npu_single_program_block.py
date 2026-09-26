"""Check an entire LFM2.5 recurrent block in one Phoenix NPU invocation."""

import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_single_program_block_kernel import recurrent_block
from lfm25_pack_block_input_kernel import pack_block_input
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as ref:
        hidden = ref["step1_hidden0"].astype(bfloat16)
        state = ref["step0_conv0_state"].reshape(-1).astype(bfloat16)
        conv_weight = ref["conv0_depthwise_weight"].reshape(-1).astype(bfloat16)
        operator_gamma = ref["first_operator_norm_weight"].astype(bfloat16)
        ffn_gamma = ref["first_ffn_norm_weight"].astype(bfloat16)
        matrices = {
            "input": ref["first_projection_weight"].astype(bfloat16),
            "output": ref["conv0_output_projection_weight"].astype(bfloat16),
            "w1": ref["first_ffn_w1_weight"].astype(bfloat16),
            "w3": ref["first_ffn_w3_weight"].astype(bfloat16),
            "w2": ref["first_ffn_w2_weight"].astype(bfloat16),
        }
        expected_state = ref["step1_conv0_state"].reshape(-1)
        expected_block = ref["step1_hidden1"]
        hidden2 = ref["step2_hidden0"].astype(bfloat16)
        expected_state2 = ref["step2_conv0_state"].reshape(-1)
        expected_block2 = ref["step2_hidden1"]
    packed_data = np.concatenate(
        [np.pad(hidden, (0, 6144 - hidden.size)), state, conv_weight]
    ).astype(bfloat16)
    packed_weights = np.concatenate(
        [
            np.pad(operator_gamma, (0, 4096 - operator_gamma.size)),
            matrices["input"].reshape(-1),
            matrices["output"].reshape(-1),
            np.pad(ffn_gamma, (0, 4096 - ffn_gamma.size)),
            matrices["w1"].reshape(-1),
            matrices["w3"].reshape(-1),
            matrices["w2"].reshape(-1),
        ]
    ).astype(bfloat16)
    data = iron.tensor(packed_data, dtype=bfloat16)
    weights = iron.tensor(packed_weights, dtype=bfloat16)
    following = iron.zeros((6144,), dtype=bfloat16, device="npu")
    output = iron.zeros((1024,), dtype=bfloat16, device="npu")

    def run():
        recurrent_block(data, weights, following, output)

    run()
    times = []
    for _ in range(5):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    actual = output.numpy().astype(np.float32)
    actual_packed = following.numpy().astype(np.float32)
    result = {
        "device": "Phoenix NPU1",
        "operation": "complete first recurrent block in one NPU program",
        "cores": 1,
        "median_ms": statistics.median(times),
        "block_exact_fraction": float(np.mean(actual == expected_block)),
        "block_max_abs_error": float(np.max(np.abs(actual - expected_block))),
        "state_max_abs_error": float(
            np.max(np.abs(actual_packed[:3072] - expected_state))
        ),
        "persisted_weight_exact": bool(
            np.array_equal(actual_packed[3072:], conv_weight.astype(np.float32))
        ),
    }
    next_hidden = iron.tensor(hidden2, dtype=bfloat16)
    packed_data2 = iron.zeros((12288,), dtype=bfloat16, device="npu")
    following2 = iron.zeros((6144,), dtype=bfloat16, device="npu")
    output2 = iron.zeros((1024,), dtype=bfloat16, device="npu")
    pack_block_input(next_hidden, following, packed_data2)
    recurrent_block(packed_data2, weights, following2, output2)
    actual2 = output2.numpy().astype(np.float32)
    actual_packed2 = following2.numpy().astype(np.float32)
    result["second_block_exact_fraction"] = float(np.mean(actual2 == expected_block2))
    result["second_block_max_abs_error"] = float(
        np.max(np.abs(actual2 - expected_block2))
    )
    result["second_state_max_abs_error"] = float(
        np.max(np.abs(actual_packed2[:3072] - expected_state2))
    )
    result["second_weight_persisted_exact"] = bool(
        np.array_equal(actual_packed2[3072:], conv_weight.astype(np.float32))
    )
    print(json.dumps(result, indent=2))
    if result["block_max_abs_error"] > 0.015625:
        raise RuntimeError("Single-program block exceeded BF16 reference tolerance")
    if result["state_max_abs_error"] != 0 or not result["persisted_weight_exact"]:
        raise RuntimeError("Single-program recurrent state did not match")
    if result["second_block_max_abs_error"] > 0.015625:
        raise RuntimeError("Second decode block output exceeded BF16 tolerance")
    if result["second_state_max_abs_error"] != 0 or not result["second_weight_persisted_exact"]:
        raise RuntimeError("Second decode recurrent state did not match")


if __name__ == "__main__":
    main()
