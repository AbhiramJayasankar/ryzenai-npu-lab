"""Save exact PyTorch BF16 intermediates for one recurrent decode layer."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]


def norm(x, gamma):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-5)
    return gamma * y.to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    prefix = f"model.layers.{args.layer}."
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as reference:
        h = torch.tensor(reference[f"step1_hidden{args.layer}"], dtype=torch.bfloat16).reshape(1, 1, 1024)
        prior = torch.tensor(reference[f"step0_conv{args.layer}_state"], dtype=torch.bfloat16).reshape(1, 1024, 3)
        expected_key = "step1_hidden14_raw" if args.layer == 13 else f"step1_hidden{args.layer + 1}"
        expected_hidden = reference[expected_key].reshape(-1)
    with safe_open(ROOT / "cache" / "lfm25-230m" / "model.safetensors", framework="pt", device="cpu") as checkpoint, torch.inference_mode():
        get = lambda name: checkpoint.get_tensor(prefix + name)
        op_norm = norm(h, get("operator_norm.weight"))
        input_proj = F.linear(op_norm, get("conv.in_proj.weight"))
        b, c, x = input_proj.transpose(-1, -2).chunk(3, dim=-2)
        state = torch.cat((prior[:, :, 1:], b * x), dim=-1)
        conv = torch.sum(state * get("conv.conv.weight")[:, 0, :], dim=-1).unsqueeze(-1)
        conv_gated = (c * conv).transpose(-1, -2).contiguous()
        output_proj = F.linear(conv_gated, get("conv.out_proj.weight"))
        residual = h + output_proj
        ffn_norm = norm(residual, get("ffn_norm.weight"))
        w1 = F.linear(ffn_norm, get("feed_forward.w1.weight"))
        w3 = F.linear(ffn_norm, get("feed_forward.w3.weight"))
        gated = F.silu(w1) * w3
        w2 = F.linear(gated, get("feed_forward.w2.weight"))
        final = residual + w2
    arrays = {
        "op_norm": op_norm, "input_proj": input_proj, "state": state,
        "conv_gated": conv_gated, "output_proj": output_proj,
        "residual": residual, "ffn_norm": ffn_norm, "w1": w1,
        "w3": w3, "gated": gated, "w2": w2, "final": final,
    }
    output = ROOT / "cache" / f"lfm25-recurrent{args.layer}-debug.npz"
    np.savez_compressed(output, **{k: v.float().numpy().reshape(-1) for k, v in arrays.items()})
    error = float(np.max(np.abs(final.float().numpy().reshape(-1) - expected_hidden)))
    print(json.dumps({"layer": args.layer, "cpu_final_max_abs_error": error, "reference": str(output)}, indent=2))
    if error != 0:
        raise RuntimeError("Debug CPU intermediates do not match the full-model fixture")


if __name__ == "__main__":
    main()
