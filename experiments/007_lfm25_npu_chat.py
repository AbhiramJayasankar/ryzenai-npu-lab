"""Small interactive LFM2.5-230M chat runner with all model math on Phoenix NPU.

The default attention kernel supports 64 total positions. An experimental
chunked mode accepts up to 4096 positions, with sharply increasing latency.
In one process, chat turns reuse NPU state when the next transcript retains
the same token prefix; otherwise the runner rebuilds it from token IDs.
"""

import argparse
import json
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16
from tokenizers import Tokenizer

from lfm25_attention_context_cache_kernel import attention_context_cache
from lfm25_attention_chunked_cache_kernel import (
    BLOCK_ELEMENTS as CHUNK_BLOCK_ELEMENTS, attention_context_chunked,
)
from lfm25_attention_first_context_kernel import attention_first_context
from lfm25_attention_fixed64_cache_kernel import (
    CACHE_ELEMENTS as FIXED64_CACHE_ELEMENTS,
    attention_context_fixed64, attention_first_fixed64,
)
from lfm25_attention_prefix_kernel import attention_prefix
from lfm25_attention_tail_kernel import attention_tail
from lfm25_checkpoint import (
    BF16Checkpoint, attention_tail_data, pack_attention_tail_weights,
    pack_recurrent_weights, recurrent_layer_data,
)
from lfm25_embedding_dynamic_kernel import embedding_dma_dynamic
from lfm25_fused_vocab_4core_kernel import fused_vocab_4core
from lfm25_pack_block_input_kernel import pack_block_input
from lfm25_single_program_block_kernel import recurrent_block


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "cache" / "lfm25-230m"
KINDS = ("conv", "conv", "attention", "conv", "attention", "conv",
         "attention", "conv", "attention", "conv", "attention", "conv",
         "attention", "conv")
VOCAB = 65536
EOS_ID = 7
MAX_POSITIONS = 96
CHUNKED_CONTEXT_POSITIONS = 4096
_FREQUENCIES = np.float32(1) / np.power(
    np.float32(1_000_000), np.arange(0, 64, 2, dtype=np.float32) / np.float32(64)
)


def rotary(position):
    values = np.float32(position) * _FREQUENCIES
    phase = np.concatenate((values, values))
    return np.cos(phase).astype(bfloat16), np.sin(phase).astype(bfloat16)


def attention_prefix_base(checkpoint, layer):
    prefix = f"model.layers.{layer}."
    gamma = checkpoint.load(prefix + "operator_norm.weight").reshape(-1)
    matrices = [checkpoint.load(prefix + f"self_attn.{name}_proj.weight").reshape(-1)
                for name in ("q", "k", "v")]
    q_gamma = checkpoint.load(prefix + "self_attn.q_layernorm.weight").reshape(-1)
    k_gamma = checkpoint.load(prefix + "self_attn.k_layernorm.weight").reshape(-1)
    if gamma.size != 1024 or q_gamma.size != 64 or k_gamma.size != 64:
        raise ValueError(f"Unexpected attention weight shape at layer {layer}")
    aux = np.concatenate((q_gamma, k_gamma,
                          np.zeros((128,), dtype=bfloat16)))
    return np.concatenate((
        np.pad(gamma, (0, 4096 - gamma.size)), *matrices,
        np.pad(aux, (0, 4096 - aux.size)),
    )).astype(bfloat16)


def render_chat(messages):
    parts = ["<|startoftext|>"]
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


class NPUChat:
    def __init__(self, cache_mode="fixed64"):
        if type(iron.get_current_device()).__name__ != "NPU1":
            raise RuntimeError("This runner requires the Phoenix NPU1")
        if not (MODEL / "model.safetensors").is_file():
            raise FileNotFoundError(f"Model checkpoint missing under {MODEL}")
        self.cache_mode = cache_mode
        self.max_positions = {
            "fixed64": 64,
            "variable": MAX_POSITIONS,
            "chunked": CHUNKED_CONTEXT_POSITIONS,
        }[cache_mode]
        self.tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
        checkpoint = BF16Checkpoint(MODEL / "model.safetensors")
        embeddings = checkpoint.load("model.embed_tokens.weight")
        self.table = iron.tensor(embeddings, dtype=bfloat16)
        self.head_weights = iron.tensor(np.concatenate((
            checkpoint.load("model.embedding_norm.weight").reshape(-1),
            embeddings.reshape(-1),
        )).astype(bfloat16), dtype=bfloat16)
        self.layers = []
        for index, kind in enumerate(KINDS):
            if kind == "conv":
                data = recurrent_layer_data(checkpoint, index)
                initial_state = np.concatenate((
                    np.zeros((3072,), dtype=bfloat16),
                    data["conv_weight"].reshape(-1),
                )).astype(bfloat16)
                self.layers.append({
                    "kind": kind,
                    "weights": iron.tensor(pack_recurrent_weights(data), dtype=bfloat16),
                    "initial_state": iron.tensor(initial_state, dtype=bfloat16),
                })
            else:
                self.layers.append({
                    "kind": kind,
                    "prefix_base": attention_prefix_base(checkpoint, index),
                    "tail_weights": iron.tensor(pack_attention_tail_weights(
                        attention_tail_data(checkpoint, index)), dtype=bfloat16),
                })
        self.reset()

    def reset(self):
        self.position = 0
        self.consumed_ids = []
        for layer in self.layers:
            layer["state"] = None
            layer["cache"] = None

    def step(self, *, token_id=None, embedding=None, select_next=False):
        if (token_id is None) == (embedding is None):
            raise ValueError("Provide exactly one token ID or NPU embedding")
        if self.position >= self.max_positions:
            raise ValueError(f"NPU attention is limited to {self.max_positions} positions")
        if token_id is not None:
            if not 0 <= token_id < VOCAB:
                raise ValueError("Token ID outside vocabulary")
            token = iron.tensor(np.array([token_id], dtype=np.int32), dtype=np.int32)
            hidden = iron.zeros((1024,), dtype=bfloat16, device="npu")
            embedding_dma_dynamic(self.table, hidden, token)
        else:
            hidden = embedding

        position = self.position
        for layer in self.layers:
            output = iron.zeros((1024,), dtype=bfloat16, device="npu")
            if layer["kind"] == "conv":
                old_state = layer["initial_state"] if position == 0 else layer["state"]
                packed = iron.zeros((12288,), dtype=bfloat16, device="npu")
                new_state = iron.zeros((6144,), dtype=bfloat16, device="npu")
                pack_block_input(hidden, old_state, packed)
                recurrent_block(packed, layer["weights"], new_state, output)
                layer["state"] = new_state
            else:
                weights = layer["prefix_base"].copy()
                cos, sin = rotary(position)
                weights[-4096 + 128:-4096 + 192] = cos
                weights[-4096 + 192:-4096 + 256] = sin
                prefix_weights = iron.tensor(weights, dtype=bfloat16)
                qkv = iron.zeros((3072,), dtype=bfloat16, device="npu")
                packed_tail = iron.zeros((2048,), dtype=bfloat16, device="npu")
                if self.cache_mode == "chunked":
                    block_count = position // 64 + 1
                    cache_size = block_count * CHUNK_BLOCK_ELEMENTS
                    if position and position % 64 == 0:
                        old = layer["cache"].numpy().reshape(-1)
                        layer["cache"] = iron.tensor(np.concatenate((
                            old, np.zeros((CHUNK_BLOCK_ELEMENTS,), dtype=bfloat16),
                        )), dtype=bfloat16)
                else:
                    cache_size = (FIXED64_CACHE_ELEMENTS if self.cache_mode == "fixed64"
                                  else 8 * 2 * (position + 1) * 64)
                next_cache = iron.zeros((cache_size,), dtype=bfloat16, device="npu")
                attention_prefix(hidden, prefix_weights, qkv, include_hidden=True)
                if position == 0:
                    first_op = (attention_first_fixed64 if self.cache_mode != "variable"
                                else attention_first_context)
                    first_op(qkv, packed_tail, next_cache)
                else:
                    if self.cache_mode == "fixed64":
                        attention_context_fixed64(qkv, layer["cache"], packed_tail,
                                                  next_cache)
                    elif self.cache_mode == "chunked":
                        attention_context_chunked(qkv, layer["cache"], packed_tail,
                                                  next_cache, block_count=block_count)
                    else:
                        attention_context_cache(qkv, layer["cache"], packed_tail,
                                                next_cache, past_length=position)
                attention_tail(packed_tail, layer["tail_weights"], output)
                layer["cache"] = next_cache
            hidden = output

        self.position += 1
        if not select_next:
            return None
        next_embedding = iron.zeros((1024,), dtype=bfloat16, device="npu")
        selected = iron.zeros((1,), dtype=np.int32, device="npu")
        fused_vocab_4core(hidden, self.head_weights, next_embedding, selected)
        return int(selected.numpy()[0]), next_embedding

    def respond(self, messages, max_new_tokens):
        prompt_ids = self.tokenizer.encode(render_chat(messages),
                                           add_special_tokens=False).ids
        if len(prompt_ids) + max_new_tokens - 1 > self.max_positions:
            raise ValueError(
                f"Prompt uses {len(prompt_ids)} tokens; this runner supports "
                f"at most {self.max_positions} total processed positions. "
                "Use a shorter message or lower --max-new-tokens."
            )
        reused_state = (
            0 < len(self.consumed_ids) < len(prompt_ids)
            and prompt_ids[:len(self.consumed_ids)] == self.consumed_ids
        )
        if not reused_state:
            self.reset()
        start_index = len(self.consumed_ids)
        start = time.perf_counter()
        prediction = None
        for index in range(start_index, len(prompt_ids)):
            token = prompt_ids[index]
            prediction = self.step(token_id=token,
                                   select_next=index == len(prompt_ids) - 1)
            self.consumed_ids.append(token)
        generated = []
        for _ in range(max_new_tokens):
            selected, next_embedding = prediction
            if selected == EOS_ID:
                break
            generated.append(selected)
            if len(generated) == max_new_tokens:
                break
            prediction = self.step(embedding=next_embedding, select_next=True)
            self.consumed_ids.append(selected)
        return {
            "prompt_tokens": len(prompt_ids),
            "prompt_tokens_processed_this_turn": len(prompt_ids) - start_index,
            "reused_npu_state": reused_state,
            "generated_ids": generated,
            "response": self.tokenizer.decode(generated, skip_special_tokens=False),
            "elapsed_seconds": time.perf_counter() - start,
            "stopped_on_eos": prediction[0] == EOS_ID,
            "device": "Phoenix NPU1",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="Run one prompt; omit for interactive chat")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--cache-mode", choices=("variable", "fixed64", "chunked"),
                        default="fixed64")
    parser.add_argument("--json", action="store_true", help="Print one JSON result")
    args = parser.parse_args()
    max_positions = {"fixed64": 64, "variable": MAX_POSITIONS,
                     "chunked": CHUNKED_CONTEXT_POSITIONS}[args.cache_mode]
    if not 1 <= args.max_new_tokens <= max_positions:
        parser.error(f"--max-new-tokens must be between 1 and {max_positions}")
    if args.json and args.prompt is None:
        parser.error("--json requires --prompt")
    chat = NPUChat(args.cache_mode)
    if args.prompt is not None:
        result = chat.respond([{"role": "user", "content": args.prompt}],
                              args.max_new_tokens)
        print(json.dumps(result, indent=2) if args.json else result["response"])
        return
    print("LFM2.5-230M on Phoenix NPU. Type /quit to exit.")
    print(f"Current runtime limit: {chat.max_positions} total processed tokens in context.")
    history = []
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.lower() in ("/quit", "/exit"):
            return
        if not prompt:
            continue
        history.append({"role": "user", "content": prompt})
        while True:
            try:
                result = chat.respond(history, args.max_new_tokens)
                break
            except ValueError as exc:
                if len(history) <= 2:
                    history.pop()
                    print(exc)
                    result = None
                    break
                history = history[2:]
        if result is None:
            continue
        print("NPU:", result["response"])
        print(f"({result['elapsed_seconds']:.1f} s, "
              f"{result['prompt_tokens_processed_this_turn']} new prompt tokens, "
              f"NPU state {'reused' if result['reused_npu_state'] else 'rebuilt'})")
        history.append({"role": "assistant", "content": result["response"]})


if __name__ == "__main__":
    main()
