"""Capture teacher-forced CPU BF16 state after each prompt token."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.lfm2.modeling_lfm2 import Lfm2RotaryEmbedding


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "cache" / "lfm25-230m"
OUTPUT = ROOT / "cache" / "lfm25-prompt-sequence-reference.npz"
PROMPT = "Reply with one short sentence about what an NPU does."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, choices=range(1, 22), default=21)
    parser.add_argument("--decode", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if args.decode and args.positions != 21:
        parser.error("--decode requires the complete 21-token prompt")
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )["input_ids"]
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    total_positions = args.positions + args.decode
    dummy = torch.zeros((1, total_positions, 1024), dtype=torch.bfloat16)
    cos, sin = Lfm2RotaryEmbedding(config)(
        dummy, torch.arange(total_positions).reshape(1, total_positions)
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16,
    ).eval()
    saved = {
        "prompt_ids": ids.numpy(),
        "cos": cos[0].float().numpy(),
        "sin": sin[0].float().numpy(),
    }
    active_position = -1

    def capture_raw(_module, inputs):
        saved[f"p{active_position}_hidden14_raw"] = inputs[0][0, -1].float().numpy().copy()

    hook = model.model.embedding_norm.register_forward_pre_hook(capture_raw)
    past = None
    selected = None
    consumed_ids = []
    with torch.inference_mode():
        for position in range(total_positions):
            active_position = position
            input_token = ids[:, position:position + 1] if position < args.positions else selected
            consumed_ids.append(int(input_token[0, 0]))
            output = model(
                input_token, past_key_values=past,
                use_cache=True, output_hidden_states=True,
            )
            for layer, hidden in enumerate(output.hidden_states):
                saved[f"p{position}_hidden{layer}"] = hidden[0, -1].float().numpy().copy()
            for layer, kind in enumerate(model.config.layer_types):
                cache = output.past_key_values.layers[layer]
                if kind == "conv":
                    saved[f"p{position}_conv{layer}_state"] = (
                        cache.conv_states[0].float().numpy().copy()
                    )
                else:
                    saved[f"p{position}_attn{layer}_keys"] = cache.keys.float().numpy().copy()
                    saved[f"p{position}_attn{layer}_values"] = cache.values.float().numpy().copy()
            saved[f"p{position}_logits"] = output.logits[0, -1].float().numpy().copy()
            selected = output.logits[:, -1].argmax(-1, keepdim=True)
            past = output.past_key_values
    hook.remove()
    saved["consumed_ids"] = np.asarray(consumed_ids, dtype=np.int32)
    np.savez_compressed(OUTPUT, **saved)
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as batched:
        final_logit_error = float(np.max(np.abs(
            saved[f"p{total_positions - 1}_logits"] -
            batched[f"step{args.decode}_logits"]
        ))) if args.positions == ids.shape[-1] else None
    print(json.dumps({"prompt_length": int(ids.shape[-1]),
                      "captured_positions": args.positions,
                      "decode_positions": args.decode,
                      "consumed_ids": consumed_ids,
                      "last_logits_vs_batched_max_abs_error": final_logit_error,
                      "reference": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
