"""Verify one Phoenix program computes first-block norm, projection and conv."""

import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_norm_proj_conv_kernel import norm_proj_conv
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as ref:
        hidden = ref["step1_hidden0"].astype(bfloat16)
        gamma = ref["first_operator_norm_weight"].astype(bfloat16)
        weight = ref["first_projection_weight"].astype(bfloat16)
        state = ref["step0_conv0_state"].reshape(-1).astype(bfloat16)
        conv_weight = ref["conv0_depthwise_weight"].reshape(-1).astype(bfloat16)
        expected_gate = ref["step1_conv0_output_projection_input"]
        expected_state = ref["step1_conv0_state"].reshape(-1)
    packed_data = np.concatenate([
        np.pad(hidden, (0, 6144 - hidden.size)), state, conv_weight
    ]).astype(bfloat16)
    packed_weight = np.concatenate([
        np.pad(gamma, (0, 4096 - gamma.size)), weight.reshape(-1)
    ]).astype(bfloat16)
    data = iron.tensor(packed_data, dtype=bfloat16)
    weights = iron.tensor(packed_weight, dtype=bfloat16)
    gate = iron.zeros((1024,), dtype=bfloat16, device="npu")
    following = iron.zeros((6144,), dtype=bfloat16, device="npu")

    def run():
        norm_proj_conv(data, weights, gate, following)

    run()
    times = []
    for _ in range(10):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    actual_gate = gate.numpy().astype(np.float32)
    actual_packed = following.numpy().astype(np.float32)
    result = {
        "device": "Phoenix NPU1",
        "operation": "single-program first RMSNorm, input GEMV and recurrent gate",
        "cores": 1,
        "median_ms": statistics.median(times),
        "gate_max_abs_error": float(np.max(np.abs(actual_gate - expected_gate))),
        "state_max_abs_error": float(np.max(np.abs(actual_packed[:3072] - expected_state))),
        "weight_persisted_exact": bool(
            np.array_equal(actual_packed[3072:], conv_weight.astype(np.float32))
        ),
    }
    print(json.dumps(result, indent=2))
    if result["gate_max_abs_error"] != 0 or result["state_max_abs_error"] != 0:
        raise RuntimeError("Fused NPU path differs from CPU BF16 reference")
    if not result["weight_persisted_exact"]:
        raise RuntimeError("Convolution weight changed inside the fused path")


if __name__ == "__main__":
    main()
