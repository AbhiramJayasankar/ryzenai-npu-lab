"""Run a complete LFM2.5 recurrent block over two decode steps on Phoenix NPU."""

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
from lfm25_rms_norm_kernel import rms_norm
from lfm25_silu_gate_kernel import silu_gate
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
        operator_norm_weight = ref["first_operator_norm_weight"].astype(bfloat16)
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
        ffn_norm_weight = ref["first_ffn_norm_weight"].astype(bfloat16)
        ffn_weights = {
            name: ref[f"first_ffn_{name}_weight"].astype(bfloat16)
            for name in ("w1", "w2", "w3")
        }
        expected_norm = ref["step1_ffn0_norm_output"]
        expected_ffn = {
            name: ref[f"step1_ffn0_{name}_output"]
            for name in ("w1", "w2", "w3")
        }
        expected_gated = ref["step1_ffn0_w2_input"]
        expected_block = ref["step1_hidden1"]
        block_input2 = ref["step2_hidden0"].astype(bfloat16)
        expected_block2 = ref["step2_hidden1"]
        expected_state2 = ref["step2_conv0_state"].reshape(-1)
        second_references = {
            "operator_norm": ref["step2_first_projection_input"],
            "input_projection": ref["step2_first_projection_output"],
            "conv_gate": ref["step2_conv0_output_projection_input"],
            "output_projection": ref["step2_conv0_output_projection_output"],
            "conv_residual": ref["step2_conv0_residual"],
            "ffn_norm": ref["step2_ffn0_norm_output"],
            "w1": ref["step2_ffn0_w1_output"],
            "w3": ref["step2_ffn0_w3_output"],
            "gated": ref["step2_ffn0_w2_input"],
            "w2": ref["step2_ffn0_w2_output"],
        }
    M, K = weight.shape
    w = iron.tensor(weight, dtype=bfloat16)
    operator_gamma = iron.tensor(operator_norm_weight, dtype=bfloat16)
    x = iron.zeros((K,), dtype=bfloat16, device="npu")
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
    ffn_gamma = iron.tensor(ffn_norm_weight, dtype=bfloat16)
    normalized = iron.zeros((K,), dtype=bfloat16, device="npu")
    fw = {
        name: iron.tensor(weight, dtype=bfloat16)
        for name, weight in ffn_weights.items()
    }
    ff32 = {
        name: iron.zeros((weight.shape[0],), dtype=np.float32, device="npu")
        for name, weight in ffn_weights.items()
    }
    fb16 = {
        name: iron.zeros((weight.shape[0],), dtype=bfloat16, device="npu")
        for name, weight in ffn_weights.items()
    }
    gated = iron.zeros((2560,), dtype=bfloat16, device="npu")
    block_output = iron.zeros((K,), dtype=bfloat16, device="npu")
    residual_input2 = iron.tensor(block_input2, dtype=bfloat16)
    state2 = iron.zeros((M,), dtype=bfloat16, device="npu")
    block_output2 = iron.zeros((K,), dtype=bfloat16, device="npu")

    def run(input_hidden, previous_state, following_state, final_output,
            phase_times=None, stage_count=17):
        stages = (
            ("operator_norm", lambda: rms_norm(input_hidden, operator_gamma, x, N=K)),
            ("input_gemv", lambda: bf16_f32_gemv(w, x, fp32, M=M, K=K, n_cores=4)),
            ("input_cast", lambda: f32_to_bf16(fp32, bf16, N=M)),
            ("conv_pack", lambda: pack_conv(bf16, cw, packed)),
            ("conv_gate", lambda: conv_gate(packed, previous_state, conv_output, following_state)),
            (
                "output_gemv",
                lambda: bf16_f32_gemv(ow, conv_output, out_fp32, M=K, K=K, n_cores=4),
            ),
            ("output_cast", lambda: f32_to_bf16(out_fp32, out_bf16, N=K)),
            ("residual_add", lambda: bf16_add(out_bf16, input_hidden, residual, N=K)),
            ("ffn_norm", lambda: rms_norm(residual, ffn_gamma, normalized, N=K)),
            (
                "w1_gemv",
                lambda: bf16_f32_gemv(fw["w1"], normalized, ff32["w1"], M=2560, K=K, n_cores=4),
            ),
            ("w1_cast", lambda: f32_to_bf16(ff32["w1"], fb16["w1"], N=2560)),
            (
                "w3_gemv",
                lambda: bf16_f32_gemv(fw["w3"], normalized, ff32["w3"], M=2560, K=K, n_cores=4),
            ),
            ("w3_cast", lambda: f32_to_bf16(ff32["w3"], fb16["w3"], N=2560)),
            ("silu_gate", lambda: silu_gate(fb16["w1"], fb16["w3"], gated, N=2560)),
            (
                "w2_gemv",
                lambda: bf16_f32_gemv(fw["w2"], gated, ff32["w2"], M=K, K=2560, n_cores=4),
            ),
            ("w2_cast", lambda: f32_to_bf16(ff32["w2"], fb16["w2"], N=K)),
            ("block_add", lambda: bf16_add(residual, fb16["w2"], final_output, N=K)),
        )
        for name, stage in stages[:stage_count]:
            start = time.perf_counter()
            stage()
            if phase_times is not None:
                phase_times[name].append((time.perf_counter() - start) * 1000)

    run(residual_input, state0, state1, block_output)
    times = []
    phase_times = {
        name: []
        for name in (
            "operator_norm", "input_gemv", "input_cast", "conv_pack", "conv_gate",
            "output_gemv", "output_cast", "residual_add",
            "ffn_norm", "w1_gemv", "w1_cast", "w3_gemv", "w3_cast",
            "silu_gate", "w2_gemv", "w2_cast", "block_add",
        )
    }
    for _ in range(10):
        start = time.perf_counter()
        run(residual_input, state0, state1, block_output, phase_times=phase_times)
        times.append((time.perf_counter() - start) * 1000)
    prefix_medians = {}
    for count in (5, 6, 7, 8):
        run(residual_input, state0, state1, block_output, stage_count=count)
        prefix_times = []
        for _ in range(5):
            start = time.perf_counter()
            run(residual_input, state0, state1, block_output, stage_count=count)
            prefix_times.append((time.perf_counter() - start) * 1000)
        prefix_medians[str(count)] = statistics.median(prefix_times)
    actual = bf16.numpy().astype(np.float32)
    norm_input_actual = x.numpy().astype(np.float32)
    difference = actual - expected
    conv_actual = conv_output.numpy().astype(np.float32)
    state_actual = state1.numpy().astype(np.float32)
    out_actual = out_bf16.numpy().astype(np.float32)
    residual_actual = residual.numpy().astype(np.float32)
    norm_actual = normalized.numpy().astype(np.float32)
    gated_actual = gated.numpy().astype(np.float32)
    block_actual = block_output.numpy().astype(np.float32)
    result = {
        "operation": "LFM2.5 layer 0 complete recurrent block on Phoenix NPU",
        "device": "Phoenix NPU1",
        "median_chain_ms": statistics.median(times),
        "operator_norm_exact_fraction": float(np.mean(norm_input_actual == activation)),
        "operator_norm_max_abs_error": float(np.max(np.abs(norm_input_actual - activation))),
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
        "ffn_norm_exact_fraction": float(np.mean(norm_actual == expected_norm)),
        "ffn_norm_max_abs_error": float(np.max(np.abs(norm_actual - expected_norm))),
        "gated_exact_fraction": float(np.mean(gated_actual == expected_gated)),
        "gated_max_abs_error": float(np.max(np.abs(gated_actual - expected_gated))),
        "block_exact_fraction": float(np.mean(block_actual == expected_block)),
        "block_max_abs_error": float(np.max(np.abs(block_actual - expected_block))),
    }
    for name in ("w1", "w3", "w2"):
        actual_ffn = fb16[name].numpy().astype(np.float32)
        result[f"{name}_exact_fraction"] = float(np.mean(actual_ffn == expected_ffn[name]))
        result[f"{name}_max_abs_error"] = float(
            np.max(np.abs(actual_ffn - expected_ffn[name]))
        )
    run(residual_input2, state1, state2, block_output2)
    second_block_actual = block_output2.numpy().astype(np.float32)
    second_state_actual = state2.numpy().astype(np.float32)
    result["second_block_exact_fraction"] = float(
        np.mean(second_block_actual == expected_block2)
    )
    result["second_block_max_abs_error"] = float(
        np.max(np.abs(second_block_actual - expected_block2))
    )
    result["second_state_exact_fraction"] = float(
        np.mean(second_state_actual == expected_state2)
    )
    result["second_state_max_abs_error"] = float(
        np.max(np.abs(second_state_actual - expected_state2))
    )
    second_tensors = {
        "operator_norm": x,
        "input_projection": bf16,
        "conv_gate": conv_output,
        "output_projection": out_bf16,
        "conv_residual": residual,
        "ffn_norm": normalized,
        "w1": fb16["w1"],
        "w3": fb16["w3"],
        "gated": gated,
        "w2": fb16["w2"],
    }
    result["second_stage_comparison"] = {
        name: {
            "exact_fraction": float(
                np.mean(tensor.numpy().astype(np.float32) == second_references[name])
            ),
            "max_abs_error": float(
                np.max(
                    np.abs(tensor.numpy().astype(np.float32) - second_references[name])
                )
            ),
        }
        for name, tensor in second_tensors.items()
    }
    print(json.dumps(result, indent=2))
    if result["max_abs_error"] > 0.03125:
        raise RuntimeError("NPU projection chain exceeded BF16 tolerance")
    if result["operator_norm_max_abs_error"] != 0:
        raise RuntimeError("NPU first operator normalization did not match")
    if result["conv_max_abs_error"] != 0 or result["state_max_abs_error"] != 0:
        raise RuntimeError("NPU recurrent convolution chain did not match the reference")
    if result["output_projection_max_abs_error"] > 0.015625:
        raise RuntimeError("NPU convolution output projection exceeded BF16 tolerance")
    if result["residual_max_abs_error"] > 0.015625:
        raise RuntimeError("NPU residual exceeded BF16 tolerance")
    if any(
        result[f"{name}_max_abs_error"] > 0.015625
        for name in ("ffn_norm", "w1", "w3", "gated", "w2", "block")
    ):
        raise RuntimeError("NPU full recurrent block exceeded BF16 tolerance")
    if result["second_block_max_abs_error"] > 0.015625 or result["second_state_max_abs_error"] != 0:
        raise RuntimeError("NPU recurrent block did not match on the second decode token")


if __name__ == "__main__":
    main()
