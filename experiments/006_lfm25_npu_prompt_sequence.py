"""Run teacher-forced prompt and autoregressive decode entirely on NPU."""

import argparse
import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_attention_context_cache_kernel import attention_context_cache
from lfm25_attention_first_context_kernel import attention_first_context
from lfm25_attention_prefix_kernel import attention_prefix
from lfm25_attention_tail_kernel import attention_tail
from lfm25_checkpoint import (
    BF16Checkpoint, attention_tail_data, pack_attention_tail_weights,
    pack_recurrent_weights, recurrent_layer_data,
)
from lfm25_embedding_dma_kernel import embedding_dma
from lfm25_fused_vocab_4core_kernel import fused_vocab_4core
from lfm25_pack_block_input_kernel import pack_block_input
from lfm25_single_program_block_kernel import recurrent_block
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]
LAYER_TYPES = (
    "conv", "conv", "attention", "conv", "attention", "conv", "attention",
    "conv", "attention", "conv", "attention", "conv", "attention", "conv",
)


def base_attention_weights(layer):
    with np.load(ROOT / "cache" / f"lfm25-attention{layer}-reference.npz") as ref:
        gamma = ref["operator_gamma"].reshape(-1).astype(bfloat16)
        matrices = [ref[f"{name}_weight"].reshape(-1).astype(bfloat16)
                    for name in ("q", "k", "v")]
        aux = np.concatenate([
            ref[name].reshape(-1) for name in ("q_gamma", "k_gamma", "cos", "sin")
        ]).astype(bfloat16)
    return np.concatenate([
        np.pad(gamma, (0, 4096 - gamma.size)), *matrices,
        np.pad(aux, (0, 4096 - aux.size)),
    ]).astype(bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, choices=range(1, 22), default=2)
    parser.add_argument("--decode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.decode and args.positions != 21:
        parser.error("--decode requires all 21 prompt positions")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    total_positions = args.positions + args.decode
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")

    checkpoint = BF16Checkpoint(ROOT / "cache" / "lfm25-230m" / "model.safetensors")
    embedding_table = checkpoint.load("model.embed_tokens.weight")
    table = iron.tensor(embedding_table, dtype=bfloat16)
    head_weights = iron.tensor(np.concatenate([
        checkpoint.load("model.embedding_norm.weight").reshape(-1),
        embedding_table.reshape(-1),
    ]).astype(bfloat16), dtype=bfloat16)
    head_embeddings = [iron.zeros((1024,), dtype=bfloat16, device="npu")
                       for _ in range(args.decode + 1)]
    head_tokens = [iron.zeros((1,), dtype=np.int32, device="npu")
                   for _ in range(args.decode + 1)]

    with np.load(ROOT / "cache" / "lfm25-prompt-sequence-reference.npz") as ref:
        prompt_ids = ref["prompt_ids"].reshape(-1)[:args.positions].astype(np.int32)
        cos = ref["cos"][:total_positions]
        sin = ref["sin"][:total_positions]
        expected_tokens = [
            int(np.argmax(ref[f"p{pos}_logits"]))
            for pos in range(args.positions - 1, total_positions)
        ]
        expected = [
            [{"hidden": ref[f"p{pos}_hidden{layer + 1 if layer < 13 else 14}_raw"].reshape(-1)
              if layer == 13 else ref[f"p{pos}_hidden{layer + 1}"].reshape(-1)}
             for layer in range(14)]
            for pos in range(total_positions)
        ]
        for pos in range(total_positions):
            for layer, kind in enumerate(LAYER_TYPES):
                if kind == "conv":
                    expected[pos][layer]["state"] = ref[f"p{pos}_conv{layer}_state"].reshape(-1)
                else:
                    keys = ref[f"p{pos}_attn{layer}_keys"].reshape(8, (pos + 1) * 64)
                    values = ref[f"p{pos}_attn{layer}_values"].reshape(8, (pos + 1) * 64)
                    expected[pos][layer]["cache"] = np.stack([keys, values], axis=1).reshape(-1)

    inputs = [iron.zeros((1024,), dtype=bfloat16, device="npu") for _ in range(args.positions)]
    layers = []
    for index, kind in enumerate(LAYER_TYPES):
        info = {"kind": kind,
                "hidden": [iron.zeros((1024,), dtype=bfloat16, device="npu")
                           for _ in range(total_positions)]}
        if kind == "conv":
            data = recurrent_layer_data(checkpoint, index)
            info["weights"] = iron.tensor(pack_recurrent_weights(data), dtype=bfloat16)
            state_and_weight = np.concatenate([
                np.zeros((3072,), dtype=bfloat16), data["conv_weight"].reshape(-1),
            ]).astype(bfloat16)
            info["initial_state"] = iron.tensor(state_and_weight, dtype=bfloat16)
            info["packed"] = [iron.zeros((12288,), dtype=bfloat16, device="npu")
                              for _ in range(total_positions)]
            info["state"] = [iron.zeros((6144,), dtype=bfloat16, device="npu")
                             for _ in range(total_positions)]
        else:
            base = base_attention_weights(index)
            info["prefix_weights"] = []
            for pos in range(total_positions):
                packed = base.copy()
                packed[-4096 + 128:-4096 + 192] = cos[pos].astype(bfloat16)
                packed[-4096 + 192:-4096 + 256] = sin[pos].astype(bfloat16)
                info["prefix_weights"].append(iron.tensor(packed, dtype=bfloat16))
            info["tail_weights"] = iron.tensor(
                pack_attention_tail_weights(attention_tail_data(checkpoint, index)),
                dtype=bfloat16,
            )
            info["qkv"] = [iron.zeros((3072,), dtype=bfloat16, device="npu")
                           for _ in range(total_positions)]
            info["tail_input"] = [iron.zeros((2048,), dtype=bfloat16, device="npu")
                                  for _ in range(total_positions)]
            info["cache"] = [iron.zeros((8 * 2 * (pos + 1) * 64,), dtype=bfloat16, device="npu")
                             for pos in range(total_positions)]
        layers.append(info)

    def run_sequence():
        start = time.perf_counter()
        timing = {"embedding_ms": 0.0, "recurrent_ms": 0.0,
                  "attention_ms": 0.0, "head_ms": 0.0,
                  "prompt_ms": 0.0, "decode_ms": 0.0}
        for pos in range(total_positions):
            position_start = time.perf_counter()
            if pos < args.positions:
                operation_start = time.perf_counter()
                embedding_dma(table, inputs[pos], token_id=int(prompt_ids[pos]))
                timing["embedding_ms"] += (time.perf_counter() - operation_start) * 1000
                hidden = inputs[pos]
            else:
                hidden = head_embeddings[pos - args.positions]
            for layer in layers:
                operation_start = time.perf_counter()
                output = layer["hidden"][pos]
                if layer["kind"] == "conv":
                    state_in = layer["initial_state"] if pos == 0 else layer["state"][pos - 1]
                    pack_block_input(hidden, state_in, layer["packed"][pos])
                    recurrent_block(layer["packed"][pos], layer["weights"],
                                    layer["state"][pos], output)
                else:
                    attention_prefix(hidden, layer["prefix_weights"][pos],
                                     layer["qkv"][pos], include_hidden=True)
                    if pos == 0:
                        attention_first_context(layer["qkv"][pos], layer["tail_input"][pos],
                                                layer["cache"][pos])
                    else:
                        attention_context_cache(
                            layer["qkv"][pos], layer["cache"][pos - 1],
                            layer["tail_input"][pos], layer["cache"][pos],
                            past_length=pos,
                        )
                    attention_tail(layer["tail_input"][pos], layer["tail_weights"], output)
                key = "recurrent_ms" if layer["kind"] == "conv" else "attention_ms"
                timing[key] += (time.perf_counter() - operation_start) * 1000
                hidden = output
            if pos >= args.positions - 1:
                operation_start = time.perf_counter()
                head_index = pos - (args.positions - 1)
                fused_vocab_4core(hidden, head_weights,
                                  head_embeddings[head_index], head_tokens[head_index])
                timing["head_ms"] += (time.perf_counter() - operation_start) * 1000
            phase = "prompt_ms" if pos < args.positions else "decode_ms"
            timing[phase] += (time.perf_counter() - position_start) * 1000
        timing["total_ms"] = (time.perf_counter() - start) * 1000
        return timing

    run_sequence()
    elapsed = [run_sequence() for _ in range(args.repeats)]
    positions = []
    for pos in range(total_positions):
        max_hidden = 0.0
        max_state = 0.0
        for index, layer in enumerate(layers):
            actual = layer["hidden"][pos].numpy().astype(np.float32)
            max_hidden = max(max_hidden, float(np.max(np.abs(actual - expected[pos][index]["hidden"]))))
            if layer["kind"] == "conv":
                actual_state = layer["state"][pos].numpy().astype(np.float32)[:3072]
                expected_state = expected[pos][index]["state"]
            else:
                actual_state = layer["cache"][pos].numpy().astype(np.float32)
                expected_state = expected[pos][index]["cache"]
            max_state = max(max_state, float(np.max(np.abs(actual_state - expected_state))))
        token_id = (int(prompt_ids[pos]) if pos < args.positions else
                    int(head_tokens[pos - args.positions].numpy()[0]))
        positions.append({"position": pos, "token_id": token_id,
                          "max_hidden_error": max_hidden, "max_state_error": max_state})
    head_checks = []
    for index, expected_token in enumerate(expected_tokens):
        head_checks.append({
            "cpu_token": expected_token,
            "npu_token": int(head_tokens[index].numpy()[0]),
            "next_embedding_max_abs_error": float(np.max(np.abs(
                head_embeddings[index].numpy().astype(np.float32) -
                embedding_table[expected_token].astype(np.float32)
            ))),
        })
    result = {
        "device": "Phoenix NPU1",
        "operation": "teacher-forced prompt and NPU autoregressive decode from empty state",
        "prompt_positions": args.positions,
        "decode_positions": args.decode,
        "warmed_total_median_ms": statistics.median(run["total_ms"] for run in elapsed),
        "timing_median_ms": {
            key: statistics.median(run[key] for run in elapsed)
            for key in ("prompt_ms", "decode_ms", "embedding_ms", "recurrent_ms",
                        "attention_ms", "head_ms")
        },
        "head_checks": head_checks,
        "position_checks": positions,
    }
    print(json.dumps(result, indent=2))
    if any(item["npu_token"] != item["cpu_token"] or item["next_embedding_max_abs_error"] != 0
           for item in head_checks):
        raise RuntimeError("NPU prompt/decode selected a different token")
    if any(item["max_hidden_error"] > 0.5 or item["max_state_error"] > 0.5
           for item in positions):
        raise RuntimeError("NPU prompt prefill diverged from CPU BF16 reference")


if __name__ == "__main__":
    main()
