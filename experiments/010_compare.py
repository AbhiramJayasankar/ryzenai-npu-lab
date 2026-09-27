"""Compare NPU greedy generation with the CPU BF16 reference token for token.

Uses cache/010_cpu_greedy.json from 010_cpu_greedy.py (same prompt IDs).
"""

import argparse
import json
import time
from pathlib import Path

import npu_direct as nd
from lfm25_x8_model import X8Model

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "cache" / "010_cpu_greedy.json"
EOS_ID = 7


def generate(model, prompt_ids, max_new):
    model.reset()
    start = time.perf_counter()
    next_id = model.prefill(prompt_ids)
    prefill = time.perf_counter() - start
    generated = []
    start = time.perf_counter()
    while len(generated) < max_new and next_id != EOS_ID:
        generated.append(next_id)
        next_id = model.step(next_id)
    decode = time.perf_counter() - start
    return generated, prefill, decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", type=int, nargs="+", default=[64, 256])
    args = parser.parse_args()
    cases = json.loads(REFERENCE.read_text())
    model = X8Model(tiers=args.tiers, log=lambda m: print(m, flush=True))
    for case in cases:
        ids = case["prompt_ids"]
        expected = case["generated_ids"]
        got, prefill, decode = generate(model, ids, len(expected))
        first_diff = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), None)
        print(json.dumps({
            "prompt_tokens": len(ids), "generated": len(got),
            "match": got == expected, "first_difference_at": first_diff,
            "npu_prefill_s": round(prefill, 4),
            "npu_decode_ms_per_token": round(decode / max(len(got), 1) * 1e3, 2),
            "cpu_prefill_s": case["cpu_prefill_s"],
            "cpu_decode_ms_per_token": case["cpu_decode_ms_per_token"],
        }), flush=True)
    nd.finish()


if __name__ == "__main__":
    main()
