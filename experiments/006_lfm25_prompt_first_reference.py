"""Capture CPU BF16 intermediate states for the first chat prompt token."""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "cache" / "lfm25-230m"
OUTPUT = ROOT / "cache" / "lfm25-prompt-first-reference.npz"
PROMPT = "Reply with one short sentence about what an NPU does."


def main():
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16,
    ).eval()
    raw_final = {}

    def capture_final_raw(_module, inputs):
        raw_final["hidden14_raw"] = inputs[0][0, -1].float().numpy().copy()

    hook = model.model.embedding_norm.register_forward_pre_hook(capture_final_raw)
    with torch.inference_mode():
        result = model(ids[:, :1], use_cache=True, output_hidden_states=True)
    hook.remove()
    saved = {
        "prompt_ids": ids.numpy(),
        "first_id": ids[:, 0].numpy(),
        "logits": result.logits[0, -1].float().numpy(),
    }
    for layer, hidden in enumerate(result.hidden_states):
        saved[f"hidden{layer}"] = hidden[0, -1].float().numpy()
    saved.update(raw_final)
    for layer, kind in enumerate(model.config.layer_types):
        cache = result.past_key_values.layers[layer]
        if kind == "conv":
            saved[f"conv{layer}_state"] = cache.conv_states[0].float().numpy()
        else:
            saved[f"attn{layer}_keys"] = cache.keys.float().numpy()
            saved[f"attn{layer}_values"] = cache.values.float().numpy()
    np.savez_compressed(OUTPUT, **saved)
    print(json.dumps({"first_id": int(ids[0, 0]), "prompt_length": ids.shape[-1],
                      "reference": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
