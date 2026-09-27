"""Layer-wise paired NPU prefill of the fixed 20-word LFM2.5 chat prompt."""

import importlib
import json
import statistics
import time

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16

from _009_prefill_prompt import PROMPT
from lfm25_fused_vocab_4core_kernel import fused_vocab_4core
from lfm25_prefill_pair_data_kernel import concat_hidden_pair
from lfm25_prefill_pair_attention_tail_kernel import attention_tail_pair
from lfm25_prefill_pair_attention_prefix_kernel import attention_prefix_pair
from lfm25_prefill_pair_attention_context_kernel import (
    CACHE_ELEMENTS, attention_pair_context_fixed64,
)
from lfm25_prefill_pair_direct_kernel import recurrent_pair_direct


def prefill(chat, prompt_ids, rotary_fn):
    if len(prompt_ids) % 2 or len(prompt_ids) > 64:
        raise ValueError("This probe requires an even prompt length up to 64 positions")
    chat.reset()
    stage_ms = {name: 0.0 for name in (
        "embedding", "recurrent_pair", "attention",
        "vocabulary_head",
    )}

    def measured(name, operation, *args, **kwargs):
        start = time.perf_counter()
        result = operation(*args, **kwargs)
        stage_ms[name] += (time.perf_counter() - start) * 1000
        return result

    start = time.perf_counter()
    pairs = []
    for index in range(0, len(prompt_ids), 2):
        first = chat.table.subview(prompt_ids[index] * 2048, (1024,))
        second = chat.table.subview(prompt_ids[index + 1] * 2048, (1024,))
        hidden = iron.zeros((2048,), dtype=bfloat16, device="npu")
        measured("embedding", concat_hidden_pair, first, second, hidden)
        pairs.append(hidden)

    for layer_index, layer in enumerate(chat.layers):
        output_pairs = []
        if layer["kind"] == "conv":
            previous_state = layer["initial_state"]
            for hidden_pair in pairs:
                state = iron.zeros((6144,), dtype=bfloat16, device="npu")
                output = iron.zeros((2048,), dtype=bfloat16, device="npu")
                measured("recurrent_pair", recurrent_pair_direct,
                         hidden_pair, previous_state, layer["weights"],
                         state, output)
                output_pairs.append(output)
                previous_state = state
            layer["state"] = previous_state
        else:
            for pair_index, hidden_pair in enumerate(pairs):
                pair_output = iron.zeros((2048,), dtype=bfloat16, device="npu")
                tail_pair = iron.zeros((4096,), dtype=bfloat16, device="npu")
                qkv_pair = iron.zeros((6144,), dtype=bfloat16, device="npu")
                weights0 = layer["prefix_base"].copy()
                aux1 = weights0[-4096:].copy()
                cos0, sin0 = rotary_fn(2 * pair_index)
                cos1, sin1 = rotary_fn(2 * pair_index + 1)
                weights0[-4096 + 128:-4096 + 192] = cos0
                weights0[-4096 + 192:-4096 + 256] = sin0
                aux1[128:192] = cos1
                aux1[192:256] = sin1
                measured("attention", attention_prefix_pair, hidden_pair,
                         iron.tensor(weights0, dtype=bfloat16),
                         iron.tensor(aux1, dtype=bfloat16), qkv_pair)
                previous_cache = layer["cache"]
                if previous_cache is None:
                    previous_cache = iron.zeros((CACHE_ELEMENTS,), dtype=bfloat16,
                                                device="npu")
                next_cache = iron.zeros((CACHE_ELEMENTS,), dtype=bfloat16,
                                        device="npu")
                measured("attention", attention_pair_context_fixed64,
                         qkv_pair, previous_cache, tail_pair, next_cache)
                layer["cache"] = next_cache
                measured("attention", attention_tail_pair,
                         tail_pair, layer["tail_weights"], pair_output)
                output_pairs.append(pair_output)
        pairs = output_pairs

    last = pairs[-1].subview(2048, (1024,))
    next_embedding = iron.zeros((1024,), dtype=bfloat16, device="npu")
    selected = iron.zeros((1,), dtype=np.int32, device="npu")
    measured("vocabulary_head", fused_vocab_4core, last, chat.head_weights,
             next_embedding, selected)
    token_id = int(selected.numpy()[0])
    chat.position = len(prompt_ids)
    chat.consumed_ids = list(prompt_ids)
    return ({"prompt_tokens": len(prompt_ids), "first_token_id": token_id,
             "elapsed_ms": (time.perf_counter() - start) * 1000,
             "stage_ms": stage_ms, "device": "Phoenix NPU1"},
            next_embedding)


def main():
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    chat_module = importlib.import_module("007_lfm25_npu_chat")
    chat = chat_module.NPUChat("fixed64")
    prompt_ids = chat.tokenizer.encode(
        chat_module.render_chat([{"role": "user", "content": PROMPT}]),
        add_special_tokens=False,
    ).ids
    if len(prompt_ids) != 34:
        raise AssertionError(f"Expected 34 positions, got {len(prompt_ids)}")
    warm, _ = prefill(chat, prompt_ids, chat_module.rotary)
    measured_runs = [prefill(chat, prompt_ids, chat_module.rotary)[0]
                     for _ in range(2)]
    ids = [run["first_token_id"] for run in [warm, *measured_runs]]
    if ids != [2797] * 3:
        raise AssertionError(f"NPU first token did not match CPU: {ids}")
    print(json.dumps({
        "prompt": PROMPT,
        "prompt_tokens": len(prompt_ids),
        "warmup_ms": warm["elapsed_ms"],
        "runs": measured_runs,
        "median_ms": statistics.median(run["elapsed_ms"] for run in measured_runs),
        "first_token_id": ids[0],
    }, indent=2))


if __name__ == "__main__":
    main()
