"""Optimized whole-prompt CPU BF16 prefill baseline for a fixed 20-word prompt."""

import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _009_prefill_prompt import PROMPT


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "cache" / "lfm25-230m"


def main():
    if len(PROMPT.split()) != 20:
        raise AssertionError("Prompt is not 20 words")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16,
    ).eval()
    runs = []
    with torch.inference_mode():
        for threads in (1, 2, 4, 8, 16):
            torch.set_num_threads(threads)
            for _ in range(2):
                model(ids, use_cache=True, logits_to_keep=1)
            elapsed = []
            selected = []
            for _ in range(5):
                start = time.perf_counter()
                output = model(ids, use_cache=True, logits_to_keep=1)
                selected.append(int(output.logits[0, -1].argmax()))
                elapsed.append((time.perf_counter() - start) * 1000)
            if len(set(selected)) != 1:
                raise AssertionError("CPU token choice changed across repeats")
            runs.append({"threads": threads,
                         "median_ms": statistics.median(elapsed),
                         "samples_ms": elapsed, "first_token_id": selected[0]})
    print(json.dumps({"prompt": PROMPT, "words": 20,
                      "chat_tokens": int(ids.shape[1]),
                      "dtype": "BF16", "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
