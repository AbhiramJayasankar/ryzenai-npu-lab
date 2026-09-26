"""Inspect FFN SiLU input ranges in recurrent decode layers."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]


def norm(x, gamma):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean() + 1e-5)
    return gamma * y.to(torch.bfloat16)


def main():
    report = []
    with safe_open(ROOT / "cache" / "lfm25-230m" / "model.safetensors", framework="pt", device="cpu") as checkpoint, np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as reference, torch.inference_mode():
        for layer in (0, 1, 3, 5, 7, 9, 11, 13):
            prefix = f"model.layers.{layer}."
            get = lambda name: checkpoint.get_tensor(prefix + name)
            h = torch.tensor(reference[f"step1_hidden{layer}"], dtype=torch.bfloat16)
            state = torch.tensor(reference[f"step0_conv{layer}_state"], dtype=torch.bfloat16).reshape(1024, 3)
            b, c, x = F.linear(norm(h, get("operator_norm.weight")), get("conv.in_proj.weight")).chunk(3)
            state = torch.cat([state[:, 1:], (b * x)[:, None]], dim=1)
            conv = torch.sum(state * get("conv.conv.weight")[:, 0, :], dim=1)
            residual = h + F.linear(c * conv, get("conv.out_proj.weight"))
            w1 = F.linear(norm(residual, get("ffn_norm.weight")), get("feed_forward.w1.weight"))
            values = w1.float().numpy()
            report.append({
                "layer": layer,
                "w1_min": float(np.min(values)),
                "w1_max": float(np.max(values)),
                "w1_abs_gt_0_75": int(np.sum(np.abs(values) > 0.75)),
                "w1_abs_gt_2": int(np.sum(np.abs(values) > 2.0)),
                "w1_abs_gt_4": int(np.sum(np.abs(values) > 4.0)),
            })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
