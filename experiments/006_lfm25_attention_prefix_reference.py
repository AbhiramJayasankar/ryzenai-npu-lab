"""Build a CPU BF16 reference for the first attention layer's Q/K/V prefix."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.lfm2.modeling_lfm2 import Lfm2RotaryEmbedding, apply_rotary_pos_emb


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "cache" / "lfm25-230m"
PROMPT = "Reply with one short sentence about what an NPU does."


def rms_norm(x, gamma, eps):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return gamma * y.to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--step", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    if config.layer_types[args.layer] != "full_attention":
        parser.error(f"Layer {args.layer} is not attention")
    suffix = "" if args.step == 1 else f"-step{args.step}"
    output_path = args.output or ROOT / "cache" / f"lfm25-attention{args.layer}{suffix}-reference.npz"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"]
    position = prompt_ids.shape[-1] + args.step - 1
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as reference:
        hidden = torch.tensor(reference[f"step{args.step}_hidden{args.layer}"], dtype=torch.bfloat16).reshape(1, 1, 1024)
    prefix = f"model.layers.{args.layer}."
    with safe_open(MODEL / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        get = lambda name: checkpoint.get_tensor(prefix + name)
        operator_gamma = get("operator_norm.weight")
        q_weight = get("self_attn.q_proj.weight")
        k_weight = get("self_attn.k_proj.weight")
        v_weight = get("self_attn.v_proj.weight")
        q_gamma = get("self_attn.q_layernorm.weight")
        k_gamma = get("self_attn.k_layernorm.weight")
    with torch.inference_mode():
        normalized = rms_norm(hidden, operator_gamma, config.norm_eps)
        q_raw = F.linear(normalized, q_weight).reshape(1, 1, 16, 64)
        k_raw = F.linear(normalized, k_weight).reshape(1, 1, 8, 64)
        v = F.linear(normalized, v_weight).reshape(1, 1, 8, 64)
        q_norm = rms_norm(q_raw, q_gamma, config.norm_eps)
        k_norm = rms_norm(k_raw, k_gamma, config.norm_eps)
        cos, sin = Lfm2RotaryEmbedding(config)(hidden, torch.tensor([[position]]))
        q, k = apply_rotary_pos_emb(
            q_norm.transpose(1, 2), k_norm.transpose(1, 2), cos, sin
        )
    tensors = {
        "hidden": hidden,
        "operator_gamma": operator_gamma,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "v_weight": v_weight,
        "q_gamma": q_gamma,
        "k_gamma": k_gamma,
        "cos": cos,
        "sin": sin,
        "q_raw": q_raw,
        "k_raw": k_raw,
        "q_norm": q_norm,
        "k_norm": k_norm,
        "q": q,
        "k": k,
        "v": v,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **{name: value.float().numpy() for name, value in tensors.items()})
    print(json.dumps({"layer": args.layer, "step": args.step, "position": position, "q_shape": list(q.shape), "k_shape": list(k.shape), "v_shape": list(v.shape), "reference": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
