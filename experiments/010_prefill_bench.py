"""Time to first token versus prompt length (batched prompt processing).

Uses arbitrary token IDs; the arithmetic is real, the text is not meaningful.
"""

import json
import time

import numpy as np

import npu_direct as nd
from lfm25_x8_model import X8Model

LENGTHS = (32, 128, 512, 1024, 2048, 4000)


def main():
    model = X8Model(log=lambda m: print(m, flush=True))
    tokens = np.random.default_rng(0).integers(1000, 30000, size=max(LENGTHS)).tolist()
    results = {}
    for n in LENGTHS:
        model.reset()
        start = time.perf_counter()
        model.prefill(tokens[:n])
        seconds = time.perf_counter() - start
        results[n] = {"seconds": round(seconds, 3), "ms_per_token": round(seconds / n * 1e3, 2)}
        print(n, results[n], flush=True)
    print(json.dumps(results), flush=True)
    nd.finish()


if __name__ == "__main__":
    main()
