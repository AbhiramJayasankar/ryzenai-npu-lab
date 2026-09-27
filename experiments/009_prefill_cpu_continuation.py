"""CPU BF16 reference for the first two tokens after the 20-word prompt."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _009_prefill_prompt import PROMPT


ROOT = Path(__file__).resolve().parents[1]


def main():
    model_path = ROOT / "cache/lfm25-230m"
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
    ).eval()
    with torch.inference_mode():
        generated = model.generate(ids, max_new_tokens=2, do_sample=False)
    print(json.dumps({"prompt_tokens": int(ids.shape[1]),
                      "generated_ids": generated[0, ids.shape[1]:].tolist()}))


if __name__ == "__main__":
    main()
