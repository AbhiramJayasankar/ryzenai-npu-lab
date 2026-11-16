"""Experiment 014, lead 3: what 8-bit weights would do to Parakeet's accuracy.

CPU only, no NPU. Runs the 24 Conformer layers in NumPy (the layer code of
experiment 011, `parakeet_hybrid.conformer_layer`) on the 210-utterance
LibriSpeech set, with the ten linear layers of every block computed in
several number formats that an NPU int8 path could use:

  fp32      FP32 weights and activations (reference for the encoder error)
  bf16      BF16 weights and activations, FP32 sums: what the 012 engine computes
  w8c       int8 weights, one FP32 scale per output channel; BF16 activations;
            FP32 sums, scale applied after the sum (the NPU would unpack int8 to
            BF16, which is exact, and reuse the BF16 matrix unit)
  w8g128    int8 weights, one scale per output channel and group of 128 inputs
  w8a8t     int8 weights (per channel) x int8 activations with one scale per
            frame (row), int32 sums (exact here, in float64)
  w8a8s     the same with one activation scale for the whole matrix

Scales are plain FP32 values (absolute maximum / 127), not powers of two.
Everything else (LayerNorm, attention scores and softmax, the depthwise
convolution, Swish, GLU, residuals, the position tables) stays FP32. The
subsampling prefix runs in ONNX Runtime FP32 and the TDT decoder in ONNX
Runtime, unchanged.

The work is done layer by layer over all utterances, so only one layer's
weights are in memory at a time. Writes cache/014/int8_parakeet.json.

  cache\\iron\\mlir-aie\\ironenv\\Scripts\\python.exe experiments\\014_int8_parakeet.py [--limit N]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "8"  # no NPU here: let BLAS use the cores

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_hybrid as ph  # noqa: E402
import parakeet_pipeline as pp  # noqa: E402

OUT = pp.ROOT / "cache" / "014" / "int8_parakeet.json"
NAMES = [n for n, _r, _c in ph.LINEARS]
VARIANTS = ["fp32", "bf16", "w8c", "w8g128", "w8a8t", "w8a8s"]


def bf(x):
    return x.astype(bfloat16).astype(np.float32)


def quant_w(w, group=None):
    """Symmetric int8 of w [K, N] with a scale per output column (and per group of
    `group` rows of K). Returns the int8 values as float32 and the dequantized weight."""
    k, n = w.shape
    g = group or k
    wg = w.reshape(k // g, g, n)
    s = np.abs(wg).max(axis=1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    q = np.clip(np.rint(wg / s), -127, 127)
    return q.reshape(k, n).astype(np.float32), (q * s).reshape(k, n).astype(np.float32), s


def quant_a(x, per_row):
    m = np.abs(x).max(axis=1, keepdims=True) if per_row else np.abs(x).max(keepdims=True)
    s = (m / 127.0).astype(np.float64)
    s[s == 0] = 1.0
    return np.clip(np.rint(x / s), -127, 127), s


class QuantMatmul:
    """Matmul backend for conformer_layer: mm(layer, name, x) and mm.multi(...)."""

    def __init__(self, kind):
        self.kind = kind
        self.w = {}
        self.stats = {"act_absmax": {}, "w_err": {}}

    def load(self, layer_weights):
        self.w = {}
        for n in NAMES:
            w = layer_weights[n].astype(np.float32)
            if self.kind == "fp32":
                self.w[n] = w
            elif self.kind == "bf16":
                self.w[n] = bf(w)
            elif self.kind == "w8g128":
                self.w[n] = quant_w(w, 128)[1]
            else:
                q, deq, s = quant_w(w)
                self.w[n] = deq if self.kind == "w8c" else (q.astype(np.float64), s.reshape(1, -1))
            if self.kind in ("w8c", "w8g128"):
                self.stats["w_err"].setdefault(n, []).append(
                    float(np.sqrt(np.mean((self.w[n] - w) ** 2) / np.mean(w ** 2))))

    def __call__(self, layer, name, x):
        w = self.w[name]
        if self.kind == "fp32":
            return x @ w
        if self.kind in ("bf16", "w8c", "w8g128"):
            return bf(x) @ w
        q, sw = w
        xq, sx = quant_a(x.astype(np.float64), self.kind == "w8a8t")
        a = self.stats["act_absmax"].setdefault(name, [0.0, 0.0])
        a[0] = max(a[0], float(np.abs(x).max()))
        return ((xq @ q) * sx * sw).astype(np.float32)

    def multi(self, layer, names, x):
        return [self(layer, n, x) for n in names]


def rel_err(ref, got):
    a, b = ref.astype(np.float64).ravel(), got.astype(np.float64).ravel()
    return float(np.sum((a - b) ** 2)), float(np.sum(a ** 2)), float(a @ b), float(b @ b)


def summarize_err(parts):
    d, r, ab, bb = (sum(p[i] for p in parts) for i in range(4))
    return {"rel_rms": round(float(np.sqrt(d / r)), 5), "cosine": round(float(ab / np.sqrt(r * bb)), 6)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    variants = args.variants.split(",")
    if "fp32" not in variants:
        variants = ["fp32"] + variants
    items = json.loads((pp.CACHE / "eval.json").read_text())[:args.limit]
    fp32_ref = {r["id"]: r["hyp"] for r in json.loads((pp.CACHE / "results" / "cpu_fp32.json").read_text())["items"]}

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 8
    prefix = ort.InferenceSession(str(pp.CACHE / "models" / "encoder_dyn_prefix.onnx"), so,
                                  providers=["CPUExecutionProvider"])
    model = pp.Parakeet("encoder-model.int8.onnx")  # preprocessor and decoder only; its encoder is unused
    t0 = time.perf_counter()
    utts = []
    for x in items:
        feats, lens = model.features(pp.load_audio(pp.ROOT / x["wav"]))
        hidden, att_mask, pad_mask = prefix.run(["hidden", "att_mask", "pad_mask"],
                                                {"audio_signal": feats, "length": lens})
        utts.append({"id": x["id"], "ref": x["text"], "masked": att_mask[0, 0].copy(),
                     "valid": int((~pad_mask).sum()), "t": hidden.shape[1]})
        utts[-1]["x"] = {v: hidden[0].copy() for v in variants}
    print(f"prefix for {len(utts)} utterances in {time.perf_counter() - t0:.1f} s, "
          f"{sum(u['t'] for u in utts)} frames", flush=True)

    mms = {v: QuantMatmul(v) for v in variants}
    layer_err = {v: [] for v in variants}
    for layer in range(ph.LAYERS):
        t1 = time.perf_counter()
        w = ph.load_layer(layer)
        for v in variants:
            mms[v].load(w)
        parts = {v: [] for v in variants}
        for u in utts:
            pos = ph.pos_for(u["t"])[layer]
            for v in variants:
                u["x"][v] = ph.conformer_layer(u["x"][v], w, pos, u["masked"], u["valid"], mms[v], layer)
            n = u["valid"]
            for v in variants:
                parts[v].append(rel_err(u["x"]["fp32"][:n], u["x"][v][:n]))
        for v in variants:
            layer_err[v].append(summarize_err(parts[v]))
        print(f"layer {layer:2d} {time.perf_counter() - t1:6.1f} s  " +
              "  ".join(f"{v} {layer_err[v][-1]['rel_rms']:.4f}" for v in variants if v != "fp32"), flush=True)
        del w

    results = {"utterances": len(utts), "frames": sum(u["t"] for u in utts), "variants": {}}
    for v in variants:
        rows = []
        for u in utts:
            out = u["x"][v].T[None]
            rows.append({"id": u["id"], "ref": u["ref"], "hyp": model.decode(out, np.array([u["valid"]]))})
        wer, errors, words = pp.wer([r["ref"] for r in rows], [r["hyp"] for r in rows])
        same = sum(r["hyp"] == fp32_ref.get(r["id"]) for r in rows)
        diff, _e, _n = pp.wer([fp32_ref[r["id"]] for r in rows], [r["hyp"] for r in rows])
        res = {"wer": round(wer, 5), "errors": errors, "words": words,
               "identical_to_onnx_fp32": same, "word_diff_vs_onnx_fp32": round(diff, 5),
               "after_2_layers": layer_err[v][1], "output": layer_err[v][-1],
               "per_layer_rel_rms": [e["rel_rms"] for e in layer_err[v]]}
        if mms[v].stats["w_err"]:
            res["weight_rel_rms"] = {n: round(float(np.mean(e)), 5) for n, e in mms[v].stats["w_err"].items()}
        if mms[v].stats["act_absmax"]:
            res["act_absmax"] = {n: round(a[0], 2) for n, a in mms[v].stats["act_absmax"].items()}
        results["variants"][v] = res
        print(v, json.dumps({k: res[k] for k in ("wer", "errors", "identical_to_onnx_fp32",
                                                 "after_2_layers", "output")}), flush=True)
        results["variants"][v]["items"] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT} in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
