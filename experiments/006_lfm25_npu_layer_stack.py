"""Run one or two LFM2.5 decode tokens through all model blocks on Phoenix NPU.

The initial embedding hidden state and each layer's prompt cache are reference
fixtures. Model block arithmetic and between-block hidden transfers run on NPU.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_argmax_kernel import vocab_argmax
from lfm25_attention_context_cache_kernel import attention_context_cache
from lfm25_attention_prefix_kernel import attention_prefix
from lfm25_attention_tail_kernel import attention_tail
from lfm25_checkpoint import (
    BF16Checkpoint, attention_tail_data, pack_attention_tail_weights,
    pack_recurrent_weights, recurrent_layer_data,
)
from lfm25_pack_block_input_kernel import pack_block_input
from lfm25_bf16_gemv_kernel import bf16_bf16_gemv
from lfm25_rms_norm_kernel import rms_norm
from lfm25_single_program_block_kernel import recurrent_block
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]
LAYER_TYPES = (
    "conv", "conv", "attention", "conv", "attention", "conv", "attention",
    "conv", "attention", "conv", "attention", "conv", "attention", "conv",
)


def attention_prefix_weights(layer, step=1):
    suffix = "" if step == 1 else f"-step{step}"
    with np.load(ROOT / "cache" / f"lfm25-attention{layer}{suffix}-reference.npz") as ref:
        gamma = ref["operator_gamma"].reshape(-1).astype(bfloat16)
        matrices = [ref[f"{name}_weight"].reshape(-1).astype(bfloat16) for name in ("q", "k", "v")]
        aux = np.concatenate([
            ref[name].reshape(-1) for name in ("q_gamma", "k_gamma", "cos", "sin")
        ]).astype(bfloat16)
    return np.concatenate([
        np.pad(gamma, (0, 4096 - gamma.size)), *matrices,
        np.pad(aux, (0, 4096 - aux.size)),
    ]).astype(bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=int, choices=range(14), default=13)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--head", action="store_true", help="Also run final norm, vocabulary projection, and NPU argmax")
    parser.add_argument("--tokens", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.head and args.through != 13:
        parser.error("--head requires all 14 model blocks")
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    checkpoint = BF16Checkpoint(ROOT / "cache" / "lfm25-230m" / "model.safetensors")
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as reference:
        initial_hidden = reference["step1_hidden0"].reshape(-1).astype(bfloat16)
        second_hidden = reference["step2_hidden0"].reshape(-1).astype(bfloat16) if args.tokens == 2 else None
        expected_hidden = [
            reference["step1_hidden14_raw" if i == 13 else f"step1_hidden{i + 1}"].reshape(-1)
            for i in range(args.through + 1)
        ]
        expected_hidden2 = [
            reference["step2_hidden14_raw" if i == 13 else f"step2_hidden{i + 1}"].reshape(-1)
            for i in range(args.through + 1)
        ] if args.tokens == 2 else None
        layers = []
        for index, kind in enumerate(LAYER_TYPES[:args.through + 1]):
            info = {"index": index, "kind": kind}
            info["hidden"] = iron.zeros((1024,), dtype=bfloat16, device="npu")
            if args.tokens == 2:
                info["hidden2"] = iron.zeros((1024,), dtype=bfloat16, device="npu")
            if kind == "conv":
                data = recurrent_layer_data(checkpoint, index)
                info["weights"] = iron.tensor(pack_recurrent_weights(data), dtype=bfloat16)
                old_state = reference[f"step0_conv{index}_state"].reshape(-1).astype(bfloat16)
                state_and_weight = np.concatenate([
                    old_state, data["conv_weight"].reshape(-1),
                ]).astype(bfloat16)
                info["state_initial"] = iron.tensor(state_and_weight, dtype=bfloat16)
                info["input"] = iron.zeros((12288,), dtype=bfloat16, device="npu")
                info["state_next"] = iron.zeros((6144,), dtype=bfloat16, device="npu")
                info["expected_state"] = reference[f"step1_conv{index}_state"].reshape(-1)
                if args.tokens == 2:
                    info["input2"] = iron.zeros((12288,), dtype=bfloat16, device="npu")
                    info["state_next2"] = iron.zeros((6144,), dtype=bfloat16, device="npu")
                    info["expected_state2"] = reference[f"step2_conv{index}_state"].reshape(-1)
            else:
                info["prefix_weights"] = iron.tensor(attention_prefix_weights(index), dtype=bfloat16)
                info["tail_weights"] = iron.tensor(
                    pack_attention_tail_weights(attention_tail_data(checkpoint, index)),
                    dtype=bfloat16,
                )
                keys = reference[f"step0_attn{index}_keys"].reshape(8, 21 * 64).astype(bfloat16)
                values = reference[f"step0_attn{index}_values"].reshape(8, 21 * 64).astype(bfloat16)
                info["old_cache"] = iron.tensor(np.stack([keys, values], axis=1).reshape(-1), dtype=bfloat16)
                info["qkv"] = iron.zeros((3072,), dtype=bfloat16, device="npu")
                info["tail_input"] = iron.zeros((2048,), dtype=bfloat16, device="npu")
                info["next_cache"] = iron.zeros((8 * 2 * 22 * 64,), dtype=bfloat16, device="npu")
                info["expected_cache"] = np.stack([
                    reference[f"step1_attn{index}_keys"].reshape(8, 22 * 64),
                    reference[f"step1_attn{index}_values"].reshape(8, 22 * 64),
                ], axis=1).reshape(-1)
                if args.tokens == 2:
                    info["prefix_weights2"] = iron.tensor(attention_prefix_weights(index, 2), dtype=bfloat16)
                    info["qkv2"] = iron.zeros((3072,), dtype=bfloat16, device="npu")
                    info["tail_input2"] = iron.zeros((2048,), dtype=bfloat16, device="npu")
                    info["next_cache2"] = iron.zeros((8 * 2 * 23 * 64,), dtype=bfloat16, device="npu")
                    info["expected_cache2"] = np.stack([
                        reference[f"step2_attn{index}_keys"].reshape(8, 23 * 64),
                        reference[f"step2_attn{index}_values"].reshape(8, 23 * 64),
                    ], axis=1).reshape(-1)
            layers.append(info)
        if args.head:
            expected_normalized = reference["step1_hidden14"].reshape(-1)
            expected_logits = reference["step1_logits"].reshape(-1)
            if args.tokens == 2:
                expected_normalized2 = reference["step2_hidden14"].reshape(-1)
                expected_logits2 = reference["step2_logits"].reshape(-1)

    initial = iron.tensor(initial_hidden, dtype=bfloat16)
    initial2 = iron.tensor(second_hidden, dtype=bfloat16) if args.tokens == 2 else None
    if args.head:
        final_gamma = iron.tensor(checkpoint.load("model.embedding_norm.weight"), dtype=bfloat16)
        vocab_weights = iron.tensor(checkpoint.load("model.embed_tokens.weight"), dtype=bfloat16)
        normalized = iron.zeros((1024,), dtype=bfloat16, device="npu")
        logits = iron.zeros((65536,), dtype=bfloat16, device="npu")
        token_id = iron.zeros((1,), dtype=np.int32, device="npu")
        if args.tokens == 2:
            normalized2 = iron.zeros((1024,), dtype=bfloat16, device="npu")
            logits2 = iron.zeros((65536,), dtype=bfloat16, device="npu")
            token_id2 = iron.zeros((1,), dtype=np.int32, device="npu")

    def run_stack(step):
        hidden = initial if step == 1 else initial2
        elapsed = []
        for layer in layers:
            start = time.perf_counter()
            if layer["kind"] == "conv":
                state_in = layer["state_initial"] if step == 1 else layer["state_next"]
                packed_input = layer["input"] if step == 1 else layer["input2"]
                state_out = layer["state_next"] if step == 1 else layer["state_next2"]
                hidden_out = layer["hidden"] if step == 1 else layer["hidden2"]
                pack_block_input(hidden, state_in, packed_input)
                recurrent_block(packed_input, layer["weights"], state_out, hidden_out)
            else:
                prefix_weight = layer["prefix_weights"] if step == 1 else layer["prefix_weights2"]
                qkv = layer["qkv"] if step == 1 else layer["qkv2"]
                cache_in = layer["old_cache"] if step == 1 else layer["next_cache"]
                cache_out = layer["next_cache"] if step == 1 else layer["next_cache2"]
                tail_input = layer["tail_input"] if step == 1 else layer["tail_input2"]
                hidden_out = layer["hidden"] if step == 1 else layer["hidden2"]
                attention_prefix(hidden, prefix_weight, qkv, include_hidden=True)
                attention_context_cache(qkv, cache_in, tail_input, cache_out,
                                        past_length=20 + step)
                attention_tail(tail_input, layer["tail_weights"], hidden_out)
            elapsed.append((time.perf_counter() - start) * 1000)
            hidden = hidden_out
        if args.head:
            norm_out = normalized if step == 1 else normalized2
            logits_out = logits if step == 1 else logits2
            token_out = token_id if step == 1 else token_id2
            start = time.perf_counter()
            rms_norm(hidden, final_gamma, norm_out, N=1024)
            elapsed.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            bf16_bf16_gemv(vocab_weights, norm_out, logits_out, M=65536, K=1024, n_cores=4)
            elapsed.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            vocab_argmax(logits_out, token_out)
            elapsed.append((time.perf_counter() - start) * 1000)
        return elapsed

    def run_tokens():
        first = run_stack(1)
        second = run_stack(2) if args.tokens == 2 else None
        return first, second

    run_tokens()
    timed_runs = [run_tokens() for _ in range(args.repeats)]
    reports = []
    for layer, expected in zip(layers, expected_hidden):
        actual = layer["hidden"].numpy().astype(np.float32)
        item = {
            "layer": layer["index"],
            "kind": layer["kind"],
            "hidden_max_abs_error": float(np.max(np.abs(actual - expected))),
            "hidden_exact_fraction": float(np.mean(actual == expected)),
            "median_ms": statistics.median(run[0][layer["index"]] for run in timed_runs),
        }
        if layer["kind"] == "conv":
            state = layer["state_next"].numpy().astype(np.float32)[:3072]
            item["state_max_abs_error"] = float(np.max(np.abs(state - layer["expected_state"])))
        else:
            cache = layer["next_cache"].numpy().astype(np.float32)
            item["cache_max_abs_error"] = float(np.max(np.abs(cache - layer["expected_cache"])))
        reports.append(item)
    result = {
        "device": "Phoenix NPU1",
        "operation": f"{args.tokens} decode token(s) through model block stack",
        "through_layer": args.through,
        "tokens": args.tokens,
        "repeats": args.repeats,
        "warmed_total_median_ms": statistics.median(
            sum(first) + (sum(second) if second is not None else 0)
            for first, second in timed_runs
        ),
        "layers": reports,
    }
    if args.tokens == 2:
        second_reports = []
        for layer, expected in zip(layers, expected_hidden2):
            actual = layer["hidden2"].numpy().astype(np.float32)
            item = {
                "layer": layer["index"],
                "kind": layer["kind"],
                "hidden_max_abs_error": float(np.max(np.abs(actual - expected))),
                "hidden_exact_fraction": float(np.mean(actual == expected)),
                "median_ms": statistics.median(run[1][layer["index"]] for run in timed_runs),
            }
            if layer["kind"] == "conv":
                state = layer["state_next2"].numpy().astype(np.float32)[:3072]
                item["state_max_abs_error"] = float(np.max(np.abs(state - layer["expected_state2"])))
            else:
                cache = layer["next_cache2"].numpy().astype(np.float32)
                item["cache_max_abs_error"] = float(np.max(np.abs(cache - layer["expected_cache2"])))
            second_reports.append(item)
        result["second_token_layers"] = second_reports
    if args.head:
        actual_norm = normalized.numpy().astype(np.float32)
        actual_logits = logits.numpy().astype(np.float32)
        result["head"] = {
            "norm_max_abs_error": float(np.max(np.abs(actual_norm - expected_normalized))),
            "logits_max_abs_error": float(np.max(np.abs(actual_logits - expected_logits))),
            "logits_exact_fraction": float(np.mean(actual_logits == expected_logits)),
            "cpu_token": int(np.argmax(expected_logits)),
            "npu_token": int(token_id.numpy()[0]),
            "norm_median_ms": statistics.median(run[0][14] for run in timed_runs),
            "vocab_median_ms": statistics.median(run[0][15] for run in timed_runs),
            "argmax_median_ms": statistics.median(run[0][16] for run in timed_runs),
        }
        if args.tokens == 2:
            actual_norm2 = normalized2.numpy().astype(np.float32)
            actual_logits2 = logits2.numpy().astype(np.float32)
            result["second_token_head"] = {
                "norm_max_abs_error": float(np.max(np.abs(actual_norm2 - expected_normalized2))),
                "logits_max_abs_error": float(np.max(np.abs(actual_logits2 - expected_logits2))),
                "logits_exact_fraction": float(np.mean(actual_logits2 == expected_logits2)),
                "cpu_token": int(np.argmax(expected_logits2)),
                "npu_token": int(token_id2.numpy()[0]),
                "norm_median_ms": statistics.median(run[1][14] for run in timed_runs),
                "vocab_median_ms": statistics.median(run[1][15] for run in timed_runs),
                "argmax_median_ms": statistics.median(run[1][16] for run in timed_runs),
            }
    print(json.dumps(result, indent=2))
    if any(item["hidden_max_abs_error"] > 0.25 for item in reports):
        raise RuntimeError("Layer-stack hidden values diverged from CPU reference")
    if args.tokens == 2 and any(item["hidden_max_abs_error"] > 0.25 for item in second_reports):
        raise RuntimeError("Second-token layer stack diverged from CPU reference")
    if args.head and result["head"]["npu_token"] != result["head"]["cpu_token"]:
        raise RuntimeError("NPU selected a different next token")
    if args.head and (
        result["head"]["norm_max_abs_error"] > 0.25
        or result["head"]["logits_max_abs_error"] > 0.25
    ):
        raise RuntimeError("NPU output head exceeded BF16 tolerance")
    if args.tokens == 2 and args.head and result["second_token_head"]["npu_token"] != result["second_token_head"]["cpu_token"]:
        raise RuntimeError("NPU selected a different second token")


if __name__ == "__main__":
    main()
