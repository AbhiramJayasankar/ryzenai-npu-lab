"""Check LFM2.5's first recurrent convolution/gate on Phoenix NPU."""

import argparse
import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_conv_gate_kernel import conv_gate
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "cache" / "lfm25-reference-cpu.npz"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected the Phoenix NPU1")
    with np.load(args.reference.resolve(strict=True)) as ref:
        projection1 = ref["step1_first_projection_output"].astype(bfloat16)
        projection2 = ref["step2_first_projection_output"].astype(bfloat16)
        previous_state = ref["step0_conv0_state"].reshape(-1).astype(bfloat16)
        weight = ref["conv0_depthwise_weight"].reshape(-1).astype(bfloat16)
        expected_output1 = ref["step1_conv0_output_projection_input"]
        expected_state1 = ref["step1_conv0_state"].reshape(-1)
        expected_output2 = ref["step2_conv0_output_projection_input"]
        expected_state2 = ref["step2_conv0_state"].reshape(-1)
    p1 = iron.tensor(np.concatenate([projection1, weight]), dtype=bfloat16)
    p2 = iron.tensor(np.concatenate([projection2, weight]), dtype=bfloat16)
    state0 = iron.tensor(previous_state, dtype=bfloat16)
    y1 = iron.zeros((1024,), dtype=bfloat16, device="npu")
    state1 = iron.zeros((3072,), dtype=bfloat16, device="npu")
    y2 = iron.zeros((1024,), dtype=bfloat16, device="npu")
    state2 = iron.zeros((3072,), dtype=bfloat16, device="npu")

    def run_second_token():
        conv_gate(p2, state1, y2, state2)

    conv_gate(p1, state0, y1, state1)
    run_second_token()
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        run_second_token()
        timings.append((time.perf_counter() - start) * 1000)
    actual_output1 = y1.numpy().astype(np.float32)
    actual_state1 = state1.numpy().astype(np.float32)
    actual_output2 = y2.numpy().astype(np.float32)
    actual_state2 = state2.numpy().astype(np.float32)
    result = {
        "operation": "LFM2.5 layer 0 recurrent convolution/gate over two decode tokens",
        "device": "Phoenix NPU1",
        "second_token_median_ms": statistics.median(timings),
        "first_output_exact_fraction": float(np.mean(actual_output1 == expected_output1)),
        "first_state_exact_fraction": float(np.mean(actual_state1 == expected_state1)),
        "second_output_exact_fraction": float(np.mean(actual_output2 == expected_output2)),
        "second_state_exact_fraction": float(np.mean(actual_state2 == expected_state2)),
        "first_output_max_abs_error": float(np.max(np.abs(actual_output1 - expected_output1))),
        "first_state_max_abs_error": float(np.max(np.abs(actual_state1 - expected_state1))),
        "second_output_max_abs_error": float(np.max(np.abs(actual_output2 - expected_output2))),
        "second_state_max_abs_error": float(np.max(np.abs(actual_state2 - expected_state2))),
        "second_first_8_actual": actual_output2[:8].tolist(),
        "second_first_8_reference": expected_output2[:8].tolist(),
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if any(result[key] != 0 for key in result if key.endswith("max_abs_error")):
        raise RuntimeError("NPU recurrent convolution did not match the reference")


if __name__ == "__main__":
    main()
