"""Check batched prompt processing against the CPU BF16 fixture.

Processes fixture positions 0-19 in batches of four, comparing every token's
final hidden state, the conv states and the KV caches after each batch, then
decodes positions 20-22 with the single-token program and compares the
selected tokens. Also times batched versus single-token prompt steps.
"""

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
    ref = np.load(FIXTURE)
    ids = [int(i) for i in ref["consumed_ids"]]
    model = X8Model(tiers=(64,), log=lambda m: print(m, flush=True))
    f = lambda a: np.asarray(a).astype(np.float32)
    worst = {"hidden": 0.0, "state": 0.0, "kv": 0.0}
    model.reset()
    for start in range(0, 20, x8.PB):
        model.prefill_batch(ids[start:start + x8.PB])
        for t in range(x8.PB):
            err = np.abs(f(model.prefill_hidden(t)) - ref[f"p{start + t}_hidden14_raw"])
            worst["hidden"] = max(worst["hidden"], float(err.max()))
        p = start + x8.PB - 1
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
    selections = [(p, model.step(ids[p]), int(np.argmax(ref[f"p{p}_logits"])))
                  for p in range(20, 23)]

    batched, single = [], []
    for _ in range(5):
        model.reset()
        t = time.perf_counter()
        for start in range(0, 20, x8.PB):
            model.prefill_batch(ids[start:start + x8.PB])
        batched.append((time.perf_counter() - t) / 20)
        model.reset()
        t = time.perf_counter()
        for token in ids[:20]:
            model.step(token, select=False)
        single.append((time.perf_counter() - t) / 20)
    print(json.dumps({
        "max_errors_after_batches": worst,
        "decode_selections_(position,npu,cpu)": selections,
        "ms_per_prompt_token_batched": round(statistics.median(batched) * 1e3, 3),
        "ms_per_prompt_token_single": round(statistics.median(single) * 1e3, 3),
    }, indent=1), flush=True)
    nd.finish()


if __name__ == "__main__":
    main()
