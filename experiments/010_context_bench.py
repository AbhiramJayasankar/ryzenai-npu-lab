"""Decode step time versus context length on the eight-core NPU engine.

Fills the KV cache with an arbitrary token sequence (prompt-style steps
without the head), then times full decode steps (with the head) at selected
positions. Arithmetic is real; the text is not meaningful.
"""

import json
import statistics
import time

import numpy as np

import npu_direct as nd
from lfm25_x8_model import X8Model

POINTS = (32, 64, 128, 256, 512, 1024, 2048, 3072, 4090)


def main():
    model = X8Model(log=lambda m: print(m, flush=True))
    rng = np.random.default_rng(0)
    tokens = rng.integers(1000, 30000, size=max(POINTS) + 8)
    results = {}
    fill = []
    for position in range(max(POINTS) + 1):
        if position in POINTS:
            samples = []
            for _ in range(3):
                t = time.perf_counter()
                model.step(int(tokens[position]))
                samples.append(time.perf_counter() - t)
                model.position -= 1  # re-run the same position
            results[position] = round(statistics.median(samples) * 1e3, 2)
            print(position, results[position], "ms per decode step", flush=True)
        t = time.perf_counter()
        model.step(int(tokens[position]), select=False)
        fill.append(time.perf_counter() - t)
    print(json.dumps({"decode_step_ms": results,
                      "mean_prompt_step_ms": round(statistics.mean(fill) * 1e3, 2)}), flush=True)
    nd.finish()


if __name__ == "__main__":
    main()
