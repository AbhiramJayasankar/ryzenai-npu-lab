"""Sequential CPU BF16 reference for the chunked NPU attention check."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = Path(__file__).resolve().parents[1] / "cache" / "lfm25-230m"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16,
    ).eval()
    prompt = "Tell me about NPUs in simple terms. " * args.repetitions
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )["input_ids"][0].tolist()
    past = None
    start = time.perf_counter()
    generated = []
    with torch.inference_mode():
        for token_id in ids:
            output = model(torch.tensor([[token_id]]), past_key_values=past,
                           use_cache=True)
            past = output.past_key_values
        for step in range(args.max_new_tokens):
            selected = int(output.logits[0, -1].argmax())
            generated.append(selected)
            if selected == model.config.eos_token_id:
                break
            if step < args.max_new_tokens - 1:
                output = model(torch.tensor([[selected]]), past_key_values=past,
                               use_cache=True)
                past = output.past_key_values
    print(json.dumps({
        "prompt_tokens": len(ids), "generated_ids": generated,
        "response": tokenizer.decode(generated),
        "elapsed_seconds": time.perf_counter() - start,
        "device": "CPU, eight threads, BF16 PyTorch",
    }, indent=2))


if __name__ == "__main__":
    main()
