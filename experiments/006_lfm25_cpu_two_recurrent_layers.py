"""Matched PyTorch BF16 CPU baseline for two LFM2.5 recurrent decode layers."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]


def load_layer(checkpoint, index):
    prefix = f"model.layers.{index}."
    names = {
        "operator_gamma": "operator_norm.weight",
        "input": "conv.in_proj.weight",
        "output": "conv.out_proj.weight",
        "ffn_gamma": "ffn_norm.weight",
        "w1": "feed_forward.w1.weight",
        "w3": "feed_forward.w3.weight",
        "w2": "feed_forward.w2.weight",
        "conv_weight": "conv.conv.weight",
    }
    layer = {key: checkpoint.get_tensor(prefix + name) for key, name in names.items()}
    layer["conv_weight"] = layer["conv_weight"][:, 0, :]
    return layer


def rms_norm(x, gamma):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6)
    return gamma * y.to(torch.bfloat16)


def recurrent_layer(hidden, previous_state, weights):
    normalized = rms_norm(hidden, weights["operator_gamma"])
    b, c, x = F.linear(normalized, weights["input"]).transpose(-1, -2).chunk(3, dim=-2)
    bx = b * x
    state = torch.cat((previous_state[:, :, 1:], bx), dim=-1)
    conv = torch.sum(state * weights["conv_weight"], dim=-1).unsqueeze(-1)
    conv_out = F.linear((c * conv).transpose(-1, -2).contiguous(), weights["output"])
    residual = hidden + conv_out
    ffn_input = rms_norm(residual, weights["ffn_gamma"])
    mlp = F.linear(
        F.silu(F.linear(ffn_input, weights["w1"]))
        * F.linear(ffn_input, weights["w3"]),
        weights["w2"],
    )
    return residual + mlp, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with safe_open(ROOT / "cache" / "lfm25-230m" / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        layer0 = load_layer(checkpoint, 0)
        layer1 = load_layer(checkpoint, 1)
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as reference:
        hidden0 = torch.tensor(reference["step1_hidden0"], dtype=torch.bfloat16).reshape(1, 1, 1024)
        state0 = torch.tensor(reference["step0_conv0_state"], dtype=torch.bfloat16).reshape(1, 1024, 3)
        state1 = torch.tensor(reference["step0_conv1_state"], dtype=torch.bfloat16).reshape(1, 1024, 3)
        expected_hidden1 = reference["step1_hidden1"]
        expected_hidden2 = reference["step1_hidden2"]
        expected_state0 = reference["step1_conv0_state"].reshape(1024, 3)
        expected_state1 = reference["step1_conv1_state"].reshape(1024, 3)

    def two_layers():
        hidden1, next_state0 = recurrent_layer(hidden0, state0, layer0)
        hidden2, next_state1 = recurrent_layer(hidden1, state1, layer1)
        return hidden1, hidden2, next_state0, next_state1

    with torch.inference_mode():
        for _ in range(10):
            actual = two_layers()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            actual = two_layers()
            times.append((time.perf_counter() - start) * 1000)
    actual_np = [tensor.float().numpy().reshape(want.shape) for tensor, want in zip(actual, (expected_hidden1, expected_hidden2, expected_state0, expected_state1))]
    expected = [expected_hidden1, expected_hidden2, expected_state0, expected_state1]
    errors = [float(np.max(np.abs(got - want))) for got, want in zip(actual_np, expected)]
    result = {
        "device": "AMD Ryzen 9 8945HS CPU",
        "torch": torch.__version__,
        "dtype": "bfloat16",
        "threads": args.threads,
        "repeats": args.repeats,
        "two_layer_median_ms": statistics.median(times),
        "two_layer_p10_ms": sorted(times)[len(times) // 10],
        "max_abs_errors_hidden1_hidden2_state0_state1": errors,
        "exact_fractions_hidden1_hidden2_state0_state1": [
            float(np.mean(got == want)) for got, want in zip(actual_np, expected)
        ],
    }
    print(json.dumps(result, indent=2))
    if any(error > 0.0625 for error in errors):
        raise RuntimeError("CPU baseline does not match the reference fixture")


if __name__ == "__main__":
    main()
