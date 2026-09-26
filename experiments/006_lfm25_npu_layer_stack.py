"""Run one LFM2.5 decode token through all model blocks on Phoenix NPU.

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


def attention_prefix_weights(layer):
    with np.load(ROOT / "cache" / f"lfm25-attention{layer}-reference.npz") as ref:
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
        expected_hidden = [
            reference["step1_hidden14_raw" if i == 13 else f"step1_hidden{i + 1}"].reshape(-1)
            for i in range(args.through + 1)
        ]
        layers = []
        for index, kind in enumerate(LAYER_TYPES[:args.through + 1]):
            info = {"index": index, "kind": kind}
            info["hidden"] = iron.zeros((1024,), dtype=bfloat16, device="npu")
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
            layers.append(info)
        if args.head:
            expected_normalized = reference["step1_hidden14"].reshape(-1)
            expected_logits = reference["step1_logits"].reshape(-1)

    initial = iron.tensor(initial_hidden, dtype=bfloat16)
    if args.head:
        final_gamma = iron.tensor(checkpoint.load("model.embedding_norm.weight"), dtype=bfloat16)
        vocab_weights = iron.tensor(checkpoint.load("model.embed_tokens.weight"), dtype=bfloat16)
        normalized = iron.zeros((1024,), dtype=bfloat16, device="npu")
        logits = iron.zeros((65536,), dtype=bfloat16, device="npu")
        token_id = iron.zeros((1,), dtype=np.int32, device="npu")

    def run_stack():
        hidden = initial
        elapsed = []
        for layer in layers:
            start = time.perf_counter()
            if layer["kind"] == "conv":
                pack_block_input(hidden, layer["state_initial"], layer["input"])
                recurrent_block(layer["input"], layer["weights"],
                                layer["state_next"], layer["hidden"])
            else:
                attention_prefix(hidden, layer["prefix_weights"],
                                 layer["qkv"], include_hidden=True)
                attention_context_cache(layer["qkv"], layer["old_cache"],
                                        layer["tail_input"], layer["next_cache"])
                attention_tail(layer["tail_input"], layer["tail_weights"],
                               layer["hidden"])
            elapsed.append((time.perf_counter() - start) * 1000)
            hidden = layer["hidden"]
        if args.head:
            start = time.perf_counter()
            rms_norm(hidden, final_gamma, normalized, N=1024)
            elapsed.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            bf16_bf16_gemv(vocab_weights, normalized, logits, M=65536, K=1024, n_cores=4)
            elapsed.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            vocab_argmax(logits, token_id)
            elapsed.append((time.perf_counter() - start) * 1000)
        return elapsed

    run_stack()
    timed_runs = [run_stack() for _ in range(args.repeats)]
    reports = []
    for layer, expected in zip(layers, expected_hidden):
        actual = layer["hidden"].numpy().astype(np.float32)
        item = {
            "layer": layer["index"],
            "kind": layer["kind"],
            "hidden_max_abs_error": float(np.max(np.abs(actual - expected))),
            "hidden_exact_fraction": float(np.mean(actual == expected)),
            "median_ms": statistics.median(run[layer["index"]] for run in timed_runs),
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
        "operation": "one decode token through model block stack",
        "through_layer": args.through,
        "repeats": args.repeats,
        "warmed_total_median_ms": statistics.median(sum(run) for run in timed_runs),
        "layers": reports,
    }
    if args.head:
        actual_norm = normalized.numpy().astype(np.float32)
        actual_logits = logits.numpy().astype(np.float32)
        result["head"] = {
            "norm_max_abs_error": float(np.max(np.abs(actual_norm - expected_normalized))),
            "logits_max_abs_error": float(np.max(np.abs(actual_logits - expected_logits))),
            "logits_exact_fraction": float(np.mean(actual_logits == expected_logits)),
            "cpu_token": int(np.argmax(expected_logits)),
            "npu_token": int(token_id.numpy()[0]),
            "norm_median_ms": statistics.median(run[14] for run in timed_runs),
            "vocab_median_ms": statistics.median(run[15] for run in timed_runs),
            "argmax_median_ms": statistics.median(run[16] for run in timed_runs),
        }
    print(json.dumps(result, indent=2))
    if any(item["hidden_max_abs_error"] > 0.25 for item in reports):
        raise RuntimeError("Layer-stack hidden values diverged from CPU reference")
    if args.head and result["head"]["npu_token"] != result["head"]["cpu_token"]:
        raise RuntimeError("NPU selected a different next token")
    if args.head and (
        result["head"]["norm_max_abs_error"] > 0.25
        or result["head"]["logits_max_abs_error"] > 0.25
    ):
        raise RuntimeError("NPU output head exceeded BF16 tolerance")


if __name__ == "__main__":
    main()
