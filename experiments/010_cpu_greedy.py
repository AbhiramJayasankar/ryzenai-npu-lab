"""CPU BF16 greedy reference for arbitrary chat prompts.

Feeds the prompt one token at a time with a KV cache (the NPU's schedule),
then greedily generates. Writes prompt and generated token IDs to JSON so the
NPU runner can be compared token for token. Run in cache/lfm-env.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "cache" / "lfm25-230m"
OUTPUT = ROOT / "cache" / "010_cpu_greedy.json"
EOS_ID = 7
PROMPTS = [
    "What is an NPU?",
    "Explain in two sentences why the sky is blue.",
    ("Here is some context about laptop processors. The AMD Ryzen 9 8945HS "
     "combines eight Zen 4 CPU cores, a Radeon 780M integrated GPU and an XDNA "
     "neural processing unit. The NPU is a spatial array of AI Engine tiles; "
     "each compute tile has its own small local memory, and data moves between "
     "tiles and main memory through programmable DMA engines. Small language "
     "models are limited mostly by how quickly their weights can be read from "
     "memory, because each generated token needs every weight once. The GPU has "
     "more memory bandwidth than the NPU, while the NPU uses less power. "
     "Question: based only on this context, which part of the chip is likely "
     "to generate text fastest for a single user, and why?"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-tokens", type=int, default=48)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16).eval()
    results = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True,
                return_dict=True, return_tensors="pt")["input_ids"][0].tolist()
            past, generated = None, []
            start = time.perf_counter()
            for token in ids:
                out = model(torch.tensor([[token]]), past_key_values=past, use_cache=True)
                past = out.past_key_values
            prefill = time.perf_counter() - start
            next_id = int(out.logits[0, -1].argmax())
            start = time.perf_counter()
            while len(generated) < args.new_tokens and next_id != EOS_ID:
                generated.append(next_id)
                out = model(torch.tensor([[next_id]]), past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_id = int(out.logits[0, -1].argmax())
            decode = time.perf_counter() - start
            results.append({
                "prompt": prompt, "prompt_ids": ids, "generated_ids": generated,
                "text": tokenizer.decode(generated),
                "cpu_prefill_s": round(prefill, 4),
                "cpu_decode_ms_per_token": round(decode / max(len(generated), 1) * 1e3, 2),
            })
            print(json.dumps({k: v for k, v in results[-1].items()
                              if k not in ("prompt_ids", "generated_ids", "prompt")}))
    OUTPUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
