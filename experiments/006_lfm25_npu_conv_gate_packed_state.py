"""Verify recurrent NPU state and weights stay packed across decode tokens."""

import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_conv_gate_packed_state_kernel import conv_gate_packed_state
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as ref:
        projection1 = ref["step1_first_projection_output"].astype(bfloat16)
        projection2 = ref["step2_first_projection_output"].astype(bfloat16)
        initial_state = ref["step0_conv0_state"].reshape(-1).astype(bfloat16)
        weight = ref["conv0_depthwise_weight"].reshape(-1).astype(bfloat16)
        expected_gate1 = ref["step1_conv0_output_projection_input"]
        expected_gate2 = ref["step2_conv0_output_projection_input"]
        expected_state1 = ref["step1_conv0_state"].reshape(-1)
        expected_state2 = ref["step2_conv0_state"].reshape(-1)
    p1 = iron.tensor(projection1, dtype=bfloat16)
    p2 = iron.tensor(projection2, dtype=bfloat16)
    packed0 = iron.tensor(np.concatenate([initial_state, weight]), dtype=bfloat16)
    packed1 = iron.zeros((6144,), dtype=bfloat16, device="npu")
    packed2 = iron.zeros((6144,), dtype=bfloat16, device="npu")
    gate1 = iron.zeros((1024,), dtype=bfloat16, device="npu")
    gate2 = iron.zeros((1024,), dtype=bfloat16, device="npu")
    conv_gate_packed_state(p1, packed0, gate1, packed1)
    conv_gate_packed_state(p2, packed1, gate2, packed2)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        conv_gate_packed_state(p2, packed1, gate2, packed2)
        times.append((time.perf_counter() - start) * 1000)
    actual_gates = [gate1.numpy().astype(np.float32), gate2.numpy().astype(np.float32)]
    actual_packed = [packed1.numpy().astype(np.float32), packed2.numpy().astype(np.float32)]
    result = {
        "device": "Phoenix NPU1",
        "operation": "two recurrent gated convolution decode tokens with persistent packed state",
        "second_token_median_ms": statistics.median(times),
        "gate_max_abs_error": [
            float(np.max(np.abs(actual_gates[0] - expected_gate1))),
            float(np.max(np.abs(actual_gates[1] - expected_gate2))),
        ],
        "state_max_abs_error": [
            float(np.max(np.abs(actual_packed[0][:3072] - expected_state1))),
            float(np.max(np.abs(actual_packed[1][:3072] - expected_state2))),
        ],
        "persisted_weight_exact": bool(
            np.array_equal(actual_packed[0][3072:], weight.astype(np.float32))
            and np.array_equal(actual_packed[1][3072:], weight.astype(np.float32))
        ),
    }
    print(json.dumps(result, indent=2))
    if any(result["gate_max_abs_error"] + result["state_max_abs_error"]):
        raise RuntimeError("Packed-state convolution differs from CPU reference")
    if not result["persisted_weight_exact"]:
        raise RuntimeError("Convolution weight changed between tokens")


if __name__ == "__main__":
    main()
