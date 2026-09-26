"""Run the first prompt token through all LFM2.5 blocks from empty NPU state."""

import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
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


def first_position_prefix_weights(layer):
    with np.load(ROOT / "cache" / f"lfm25-attention{layer}-reference.npz") as ref:
        gamma = ref["operator_gamma"].reshape(-1).astype(bfloat16)
        matrices = [ref[f"{name}_weight"].reshape(-1).astype(bfloat16)
                    for name in ("q", "k", "v")]
        aux = np.concatenate([
            ref["q_gamma"].reshape(-1), ref["k_gamma"].reshape(-1),
            np.ones((64,), dtype=np.float32), np.zeros((64,), dtype=np.float32),
        ]).astype(bfloat16)
    return np.concatenate([
        np.pad(gamma, (0, 4096 - gamma.size)), *matrices,
        np.pad(aux, (0, 4096 - aux.size)),
    ]).astype(bfloat16)


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    checkpoint = BF16Checkpoint(ROOT / "cache" / "lfm25-230m" / "model.safetensors")
    embedding_table = checkpoint.load("model.embed_tokens.weight")
    table = iron.tensor(embedding_table, dtype=bfloat16)
    head_weights = iron.tensor(np.concatenate([
        checkpoint.load("model.embedding_norm.weight").reshape(-1),
        embedding_table.reshape(-1),
    ]).astype(bfloat16), dtype=bfloat16)
    next_embedding = iron.zeros((1024,), dtype=bfloat16, device="npu")
    next_token = iron.zeros((1,), dtype=np.int32, device="npu")
    layers = []
    with np.load(ROOT / "cache" / "lfm25-prompt-first-reference.npz") as reference:
        first_id = int(reference["first_id"].reshape(-1)[0])
        expected_next_token = int(np.argmax(reference["logits"].reshape(-1)))
        expected_hiddens = [
            reference["hidden14_raw" if i == 14 else f"hidden{i}"].reshape(-1)
            for i in range(15)
        ]
        initial = iron.zeros((1024,), dtype=bfloat16, device="npu")
        for index, kind in enumerate(LAYER_TYPES):
            info = {"index": index, "kind": kind,
                    "hidden": iron.zeros((1024,), dtype=bfloat16, device="npu")}
            if kind == "conv":
                data = recurrent_layer_data(checkpoint, index)
                info["weights"] = iron.tensor(pack_recurrent_weights(data), dtype=bfloat16)
                state_weight = np.concatenate([
                    np.zeros((3072,), dtype=bfloat16),
                    data["conv_weight"].reshape(-1),
                ]).astype(bfloat16)
                info["state_weight"] = iron.tensor(state_weight, dtype=bfloat16)
                info["packed"] = iron.zeros((12288,), dtype=bfloat16, device="npu")
                info["next_state"] = iron.zeros((6144,), dtype=bfloat16, device="npu")
                info["expected_state"] = reference[f"conv{index}_state"].reshape(-1)
            else:
                info["prefix_weights"] = iron.tensor(first_position_prefix_weights(index), dtype=bfloat16)
                info["tail_weights"] = iron.tensor(
                    pack_attention_tail_weights(attention_tail_data(checkpoint, index)),
                    dtype=bfloat16,
                )
                info["qkv"] = iron.zeros((3072,), dtype=bfloat16, device="npu")
                info["tail_input"] = iron.zeros((2048,), dtype=bfloat16, device="npu")
                info["next_cache"] = iron.zeros((1024,), dtype=bfloat16, device="npu")
                keys = reference[f"attn{index}_keys"].reshape(8, 64)
                values = reference[f"attn{index}_values"].reshape(8, 64)
                info["expected_cache"] = np.stack([keys, values], axis=1).reshape(-1)
            layers.append(info)

    def run_stack():
        start = time.perf_counter()
        embedding_dma(table, initial, token_id=first_id)
        embedding_ms = (time.perf_counter() - start) * 1000
        hidden = initial
        elapsed = []
        for layer in layers:
            start = time.perf_counter()
            if layer["kind"] == "conv":
                pack_block_input(hidden, layer["state_weight"], layer["packed"])
                recurrent_block(layer["packed"], layer["weights"],
                                layer["next_state"], layer["hidden"])
            else:
                attention_prefix(hidden, layer["prefix_weights"], layer["qkv"],
                                 include_hidden=True)
                attention_first_context(layer["qkv"], layer["tail_input"],
                                        layer["next_cache"])
                attention_tail(layer["tail_input"], layer["tail_weights"], layer["hidden"])
            elapsed.append((time.perf_counter() - start) * 1000)
            hidden = layer["hidden"]
        start = time.perf_counter()
        fused_vocab_4core(hidden, head_weights, next_embedding, next_token)
        head_ms = (time.perf_counter() - start) * 1000
        return embedding_ms, elapsed, head_ms

    run_stack()
    times = [run_stack() for _ in range(3)]
    reports = []
    for layer, expected in zip(layers, expected_hiddens[1:]):
        actual = layer["hidden"].numpy().astype(np.float32)
        item = {
            "layer": layer["index"], "kind": layer["kind"],
            "hidden_max_abs_error": float(np.max(np.abs(actual - expected))),
            "median_ms": statistics.median(run[1][layer["index"]] for run in times),
        }
        if layer["kind"] == "conv":
            state = layer["next_state"].numpy().astype(np.float32)[:3072]
            item["state_max_abs_error"] = float(np.max(np.abs(state - layer["expected_state"])))
        else:
            cache = layer["next_cache"].numpy().astype(np.float32)
            item["cache_max_abs_error"] = float(np.max(np.abs(cache - layer["expected_cache"])))
        reports.append(item)
    result = {
        "device": "Phoenix NPU1", "operation": "first prompt token from empty state",
        "token_id": first_id,
        "embedding_max_abs_error": float(np.max(np.abs(
            initial.numpy().astype(np.float32) - expected_hiddens[0]
        ))),
        "cpu_next_token": expected_next_token,
        "npu_next_token": int(next_token.numpy()[0]),
        "next_embedding_max_abs_error": float(np.max(np.abs(
            next_embedding.numpy().astype(np.float32) -
            embedding_table[expected_next_token].astype(np.float32)
        ))),
        "warmed_total_median_ms": statistics.median(
            embedding_ms + sum(layers_ms) + head_ms for embedding_ms, layers_ms, head_ms in times
        ),
        "embedding_median_ms": statistics.median(run[0] for run in times),
        "head_median_ms": statistics.median(run[2] for run in times),
        "layers": reports,
    }
    print(json.dumps(result, indent=2))
    if any(item["hidden_max_abs_error"] > 0.25 for item in reports):
        raise RuntimeError("Prompt-first hidden values diverged from CPU BF16")
    if any(item.get("state_max_abs_error", item.get("cache_max_abs_error", 0)) > 0.25
           for item in reports):
        raise RuntimeError("Prompt-first state diverged from CPU BF16")
    if result["embedding_max_abs_error"] != 0 or result["npu_next_token"] != expected_next_token:
        raise RuntimeError("Prompt-first NPU embedding or token selection diverged")
    if result["next_embedding_max_abs_error"] != 0:
        raise RuntimeError("Prompt-first NPU output embedding diverged")


if __name__ == "__main__":
    main()
