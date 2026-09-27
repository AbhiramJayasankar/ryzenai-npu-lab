"""Check the eight-core NPU decode path against the CPU BF16 fixture.

Replays the fixture's 23 consumed tokens (21 prompt + 2 generated). With
--check, compares the final hidden state, every conv state and every KV cache
entry after each position; always compares every selected token.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

import lfm25_x8 as x8
import npu_direct as nd
from lfm25_x8_model import X8Model

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "cache" / "lfm25-prompt-sequence-reference.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=23)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tiers", type=int, nargs="+", default=[64])
    parser.add_argument("--prefill-no-head", action="store_true",
                        help="time prompt positions without the vocabulary head")
    args = parser.parse_args()
    ref = np.load(FIXTURE)
    ids = [int(i) for i in ref["consumed_ids"]][:args.positions]
    prompt_len = int(ref["prompt_ids"].shape[-1])
    model = X8Model(tiers=args.tiers, log=lambda m: print(m, flush=True))

    worst = {"hidden": 0.0, "state": 0.0, "kv": 0.0}

    def check(p):
        f = lambda a: np.asarray(a).astype(np.float32)
        worst["hidden"] = max(worst["hidden"], float(np.max(np.abs(
            f(model.hidden()) - ref[f"p{p}_hidden14_raw"]))))
        conv_i = att_i = 0
        for layer, kind in enumerate(x8.KINDS):
            if kind == "conv":
                err = np.abs(f(model.conv_state(conv_i)) - ref[f"p{p}_conv{layer}_state"])
                worst["state"] = max(worst["state"], float(err.max()))
                conv_i += 1
            else:
                keys, values = model.kv(att_i, p)
                for got, name in ((keys, "keys"), (values, "values")):
                    err = np.abs(f(got) - ref[f"p{p}_attn{layer}_{name}"][0])
                    worst["kv"] = max(worst["kv"], float(err.max()))
                att_i += 1

    runs = []
    for repeat in range(args.repeats):
        model.reset()
        mismatches, per_position = [], []
        start = time.perf_counter()
        for p, token in enumerate(ids):
            select = not (args.prefill_no_head and p < prompt_len - 1)
            t = time.perf_counter()
            selected = model.step(token, select=select)
            per_position.append(time.perf_counter() - t)
            if selected is not None:
                expected = int(np.argmax(ref[f"p{p}_logits"]))
                if selected != expected:
                    mismatches.append((p, selected, expected))
            if args.check and repeat == 0:
                check(p)
        elapsed = time.perf_counter() - start
        runs.append({
            "seconds": round(elapsed, 4),
            "ms_per_position": round(elapsed / len(ids) * 1e3, 2),
            "median_step_ms": round(statistics.median(per_position) * 1e3, 3),
            "token_mismatches": mismatches,
        })
    print(json.dumps({"positions": len(ids), "runs": runs, "max_errors": worst}, indent=1),
          flush=True)
    nd.finish()


if __name__ == "__main__":
    main()
