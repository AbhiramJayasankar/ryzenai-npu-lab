"""CPU comparison for the 34-position first-layer projection probe."""

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]


def main():
    model = ROOT / "cache/lfm25-230m/model.safetensors"
    with safe_open(model, framework="pt", device="cpu") as checkpoint:
        weight = checkpoint.get_tensor("model.layers.0.conv.in_proj.weight")
    rng = np.random.default_rng(9)
    host_a = rng.normal(0, 0.1, (34, 1024)).astype(np.float32)
    a = torch.from_numpy(host_a).to(torch.bfloat16).float()
    b = weight.float().T.contiguous()
    results = []
    for threads in (1, 2, 4, 8, 16):
        torch.set_num_threads(threads)
        with torch.inference_mode():
            for _ in range(3):
                torch.mm(a, b)
            elapsed = []
            for _ in range(20):
                start = time.perf_counter()
                torch.mm(a, b)
                elapsed.append((time.perf_counter() - start) * 1000)
        results.append({"threads": threads, "median_ms": statistics.median(elapsed)})
    print(json.dumps({
        "operation": "34 useful prompt positions in layer 0 input projection",
        "shape_executed": [34, 1024, 3072],
        "input_values": "BF16 rounded, converted to FP32 for matched FP32 output",
        "runs": results,
    }, indent=2))


if __name__ == "__main__":
    main()
