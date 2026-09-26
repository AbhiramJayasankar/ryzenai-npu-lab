"""Verify that the NPU GEMV output can feed an NPU BF16 cast directly."""

import argparse
import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_cast_kernel import f32_to_bf16
from lfm25_add_kernel import bf16_add
from lfm25_conv_gate_kernel import conv_gate
from lfm25_gemv_kernel import bf16_f32_gemv
from lfm25_pack_conv_kernel import pack_conv
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "cache" / "lfm25-reference-cpu.npz"
    )
    args = parser.parse_args()
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected the Phoenix NPU1")
    with np.load(args.reference.resolve(strict=True)) as ref:
        activation = ref["step1_first_projection_input"].astype(bfloat16)
        weight = ref["first_projection_weight"].astype(bfloat16)
        expected = ref["step1_first_projection_output"]
        conv_weight = ref["conv0_depthwise_weight"].reshape(-1).astype(bfloat16)
        previous_state = ref["step0_conv0_state"].reshape(-1).astype(bfloat16)
        expected_conv = ref["step1_conv0_output_projection_input"]
        expected_state = ref["step1_conv0_state"].reshape(-1)
        out_weight = ref["conv0_output_projection_weight"].astype(bfloat16)
        expected_out = ref["step1_conv0_output_projection_output"]
        block_input = ref["step1_hidden0"].astype(bfloat16)
        expected_residual = ref["step1_conv0_residual"]
    M, K = weight.shape
    w = iron.tensor(weight, dtype=bfloat16)
    x = iron.tensor(activation, dtype=bfloat16)
    fp32 = iron.zeros((M,), dtype=np.float32, device="npu")
    bf16 = iron.zeros((M,), dtype=bfloat16, device="npu")
    cw = iron.tensor(conv_weight, dtype=bfloat16)
    state0 = iron.tensor(previous_state, dtype=bfloat16)
    packed = iron.zeros((2 * M,), dtype=bfloat16, device="npu")
    conv_output = iron.zeros((K,), dtype=bfloat16, device="npu")
    state1 = iron.zeros((M,), dtype=bfloat16, device="npu")
    ow = iron.tensor(out_weight, dtype=bfloat16)
    out_fp32 = iron.zeros((K,), dtype=np.float32, device="npu")
    out_bf16 = iron.zeros((K,), dtype=bfloat16, device="npu")
    residual_input = iron.tensor(block_input, dtype=bfloat16)
    residual = iron.zeros((K,), dtype=bfloat16, device="npu")

    def run(phase_times=None, stage_count=7):
        stages = (
            ("input_gemv", lambda: bf16_f32_gemv(w, x, fp32, M=M, K=K, n_cores=4)),
            ("input_cast", lambda: f32_to_bf16(fp32, bf16, N=M)),
            ("conv_pack", lambda: pack_conv(bf16, cw, packed)),
            ("conv_gate", lambda: conv_gate(packed, state0, conv_output, state1)),
            (
                "output_gemv",
                lambda: bf16_f32_gemv(ow, conv_output, out_fp32, M=K, K=K, n_cores=4),
            ),
            ("output_cast", lambda: f32_to_bf16(out_fp32, out_bf16, N=K)),
            ("residual_add", lambda: bf16_add(out_bf16, residual_input, residual, N=K)),
        )
        for name, stage in stages[:stage_count]:
            start = time.perf_counter()
            stage()
            if phase_times is not None:
                phase_times[name].append((time.perf_counter() - start) * 1000)

    run()
    times = []
    phase_times = {
        name: []
        for name in (
            "input_gemv", "input_cast", "conv_pack", "conv_gate",
            "output_gemv", "output_cast", "residual_add",
        )
    }
    for _ in range(10):
        start = time.perf_counter()
        run(phase_times)
        times.append((time.perf_counter() - start) * 1000)
    prefix_medians = {}
    for count in (4, 5, 6, 7):
        run(stage_count=count)
        prefix_times = []
        for _ in range(5):
            start = time.perf_counter()
            run(stage_count=count)
            prefix_times.append((time.perf_counter() - start) * 1000)
        prefix_medians[str(count)] = statistics.median(prefix_times)
    actual = bf16.numpy().astype(np.float32)
    difference = actual - expected
    conv_actual = conv_output.numpy().astype(np.float32)
    state_actual = state1.numpy().astype(np.float32)
    out_actual = out_bf16.numpy().astype(np.float32)
    residual_actual = residual.numpy().astype(np.float32)
    result = {
        "operation": "LFM2.5 layer 0 NPU input projection through conv output residual",
        "device": "Phoenix NPU1",
        "median_chain_ms": statistics.median(times),
        "median_stages_ms": {
            name: statistics.median(values) for name, values in phase_times.items()
        },
        "prefix_median_ms": prefix_medians,
        "exact_fraction": float(np.mean(actual == expected)),
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "first_8_actual": actual[:8].tolist(),
        "first_8_reference": expected[:8].tolist(),
        "conv_exact_fraction": float(np.mean(conv_actual == expected_conv)),
        "conv_max_abs_error": float(np.max(np.abs(conv_actual - expected_conv))),
        "state_exact_fraction": float(np.mean(state_actual == expected_state)),
        "state_max_abs_error": float(np.max(np.abs(state_actual - expected_state))),
        "output_projection_exact_fraction": float(np.mean(out_actual == expected_out)),
        "output_projection_max_abs_error": float(np.max(np.abs(out_actual - expected_out))),
        "residual_exact_fraction": float(np.mean(residual_actual == expected_residual)),
        "residual_max_abs_error": float(np.max(np.abs(residual_actual - expected_residual))),
    }
    print(json.dumps(result, indent=2))
    if result["max_abs_error"] > 0.03125:
        raise RuntimeError("NPU projection chain exceeded BF16 tolerance")
    if result["conv_max_abs_error"] != 0 or result["state_max_abs_error"] != 0:
        raise RuntimeError("NPU recurrent convolution chain did not match the reference")
    if result["output_projection_max_abs_error"] > 0.015625:
        raise RuntimeError("NPU convolution output projection exceeded BF16 tolerance")
    if result["residual_max_abs_error"] > 0.015625:
        raise RuntimeError("NPU residual exceeded BF16 tolerance")


if __name__ == "__main__":
    main()
