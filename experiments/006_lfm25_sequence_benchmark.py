"""Matched 21-position prompt plus two decode positions on CPU or GPU."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "cache" / "lfm25-230m"
REFERENCE = ROOT / "cache" / "lfm25-prompt-sequence-reference.npz"


def run(model, ids, device):
    past = None
    times = []
    selected = []
    with torch.inference_mode():
        for token in ids:
            input_id = torch.tensor([[int(token)]], dtype=torch.long, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = model(input_id, past_key_values=past, use_cache=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - start) * 1000)
            if len(times) >= 21:
                selected.append(int(output.logits[0, -1].argmax()))
            past = output.past_key_values
    return times, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.repeats < 1:
        parser.error("--cpu-threads and --repeats must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    with np.load(REFERENCE) as ref:
        ids = ref["consumed_ids"].reshape(-1)
        if len(ids) != 23:
            raise RuntimeError("Create the 21+2 position reference first")
        expected = [int(np.argmax(ref[f"p{pos}_logits"])) for pos in (20, 21, 22)]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16,
    ).eval().to(device)
    run(model, ids, device)
    samples = [run(model, ids, device) for _ in range(args.repeats)]
    result = {
        "device": args.device,
        "cpu_threads": args.cpu_threads if device.type == "cpu" else None,
        "model": "LiquidAI/LFM2.5-230M",
        "dtype": "bfloat16",
        "positions": len(ids),
        "repeats": args.repeats,
        "prompt_median_ms": statistics.median(sum(times[:21]) for times, _ in samples),
        "two_decode_median_ms": statistics.median(sum(times[21:]) for times, _ in samples),
        "total_median_ms": statistics.median(sum(times) for times, _ in samples),
        "cpu_reference_selected_tokens": expected,
        "selected_tokens": samples[-1][1],
    }
    print(json.dumps(result, indent=2))
    if device.type == "cpu" and result["selected_tokens"] != expected:
        raise RuntimeError("CPU reference sequence selected different tokens")


if __name__ == "__main__":
    main()
