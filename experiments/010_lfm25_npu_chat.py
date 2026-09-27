"""Interactive LFM2.5-230M chat with all model arithmetic on the Phoenix NPU.

Uses the eight-core engine (lfm25_x8_model): prompt tokens run four at a time
per weight pass, and each reply token is one NPU submission for all 14 layers
and the vocabulary head. Greedy decoding, EOS stop, and NPU state
reuse across turns while the transcript keeps the same token prefix. Context
is limited to 4096 positions (the largest compiled KV tier).
"""

import argparse
import json
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer

import npu_direct as nd
from lfm25_x8_model import MODEL, X8Model

EOS_ID = 7


def render_chat(messages):
    parts = ["<|startoftext|>"]
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


class NPUChat:
    def __init__(self, log=print):
        self.model = X8Model(log=log)
        self.tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
        self.consumed = []

    def respond(self, messages, max_new_tokens, on_text=None):
        ids = self.tokenizer.encode(render_chat(messages), add_special_tokens=False).ids
        if len(ids) + max_new_tokens > self.model.capacity:
            raise ValueError(f"Transcript needs {len(ids) + max_new_tokens} positions; "
                             f"the NPU KV cache holds {self.model.capacity}.")
        reuse = 0 < len(self.consumed) < len(ids) and ids[:len(self.consumed)] == self.consumed
        if not reuse:
            self.model.reset()
            self.consumed = []
        start = time.perf_counter()
        pending = ids[len(self.consumed):]
        next_id = self.model.prefill(pending)
        self.consumed = list(ids)
        first_token = time.perf_counter() - start
        generated, shown = [], ""
        decode_start = time.perf_counter()
        decode_steps = 0
        while next_id != EOS_ID and len(generated) < max_new_tokens:
            generated.append(next_id)
            text = self.tokenizer.decode(generated)
            if on_text and len(text) > len(shown) and not text.endswith("�"):
                on_text(text[len(shown):])
                shown = text
            if len(generated) == max_new_tokens:
                break
            self.consumed.append(next_id)
            next_id = self.model.step(next_id)
            decode_steps += 1
        decode = time.perf_counter() - decode_start
        return {
            "response": self.tokenizer.decode(generated),
            "generated_ids": generated,
            "prompt_tokens": len(ids),
            "prompt_tokens_processed": len(pending),
            "reused_npu_state": reuse,
            "generated_tokens": len(generated),
            "time_to_first_token_s": round(first_token, 3),
            "decode_tokens_per_s": round(decode_steps / decode, 1) if decode_steps else None,
            "stopped_on_eos": next_id == EOS_ID,
            "device": "Phoenix NPU1",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="Run one prompt and exit")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--json", action="store_true", help="With --prompt: print JSON")
    args = parser.parse_args()
    chat = NPUChat(log=lambda m: print(m, file=sys.stderr, flush=True))
    if args.prompt is not None:
        stream = None if args.json else (lambda t: print(t, end="", flush=True))
        result = chat.respond([{"role": "user", "content": args.prompt}],
                              args.max_new_tokens, on_text=stream)
        print(json.dumps(result, indent=1) if args.json else "", flush=True)
        nd.finish()
    print("LFM2.5-230M on the Phoenix NPU. Type /quit to exit, /reset to clear.", flush=True)
    history = []
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in ("/quit", "/exit"):
            break
        if prompt.lower() == "/reset":
            history = []
            continue
        if not prompt:
            continue
        history.append({"role": "user", "content": prompt})
        print("NPU: ", end="", flush=True)
        try:
            result = chat.respond(history, args.max_new_tokens,
                                  on_text=lambda t: print(t, end="", flush=True))
        except ValueError as exc:
            print(exc)
            history = []
            continue
        print(f"\n({result['prompt_tokens_processed']} prompt tokens in "
              f"{result['time_to_first_token_s']} s, {result['generated_tokens']} tokens at "
              f"{result['decode_tokens_per_s']} tok/s, state "
              f"{'reused' if result['reused_npu_state'] else 'rebuilt'})", flush=True)
        history.append({"role": "assistant", "content": result["response"]})
    nd.finish()


if __name__ == "__main__":
    main()
