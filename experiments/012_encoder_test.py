"""Experiment 012: the full 24-layer encoder on the NPU vs ONNX Runtime FP32.

For the clips of cache/parakeet/reference_dyn.npz that fit the program's
frame bucket: subsampling prefix on the CPU (ONNX Runtime), 24 Conformer
layers in split NPU submissions, TDT decoding on the CPU. Reports the encoder
output error against the FP32 encoder, the transcript, and NPU time.

  powershell -NoProfile -File scripts\\run_npu_timeout.ps1 -Script experiments\\012_encoder_test.py -Seconds 3600

Read 012_handoff.md and obtain owner approval before any hardware run.
Use "1 24 --clip 0 --repeats 1" for exactly one full encoder run; a larger
repeat count is a separately approved soak, never an automatic hang retry.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

import npu_direct as nd
import pk_model as pm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_pipeline as pp  # noqa: E402


def err(ref, got):
    a, b = ref.ravel().astype(np.float64), got.ravel().astype(np.float64)
    return {"cosine": round(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), 6),
            "rel_rms": round(float(np.sqrt(np.mean((a - b) ** 2) / np.mean(a ** 2))), 5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nblk", type=int, nargs="?", default=1)
    parser.add_argument("n_layers", type=int, nargs="?", default=24)
    parser.add_argument("--clip", type=int, help="run only this reference clip (zero based)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="consecutive runs per clip; stop on any error, never retry")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    nblk, n_layers = args.nblk, args.n_layers
    # Validate clip selection before constructing anything that opens the NPU.
    ref = np.load(pp.CACHE / "reference_dyn.npz")
    nclips = sum(1 for k in ref.files if k.startswith("feats"))
    if args.clip is not None and not 0 <= args.clip < nclips:
        parser.error(f"--clip must be in [0, {nclips})")
    t = time.perf_counter()
    enc = pm.PkEncoder(nblk=nblk, n_layers=n_layers)
    print(f"setup {time.perf_counter() - t:.1f} s (pack/load {enc.pack_s:.1f}, compile {enc.compile_s:.1f}), "
          f"{len(enc.phases)} phases", flush=True)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    prefix = ort.InferenceSession(str(pp.CACHE / "models" / "encoder_dyn_prefix.onnx"), so,
                                  providers=["CPUExecutionProvider"])
    model = pp.Parakeet("encoder-model.int8.onnx")  # preprocessor and decoder only
    rows = []
    clips = range(nclips) if args.clip is None else [args.clip]
    for c in clips:
        hidden = prefix.run(["hidden"], {"audio_signal": ref[f"feats{c}"], "length": ref[f"len{c}"]})[0][0]
        n = hidden.shape[0]
        if n > enc.lay.TPAD:
            if args.clip is not None:
                raise ValueError(f"clip {c} has {n} frames, exceeds bucket {enc.lay.TPAD}")
            continue
        times = []
        baseline = None
        for repeat in range(args.repeats):
            print(f"clip {c}: starting encoder run {repeat + 1}/{args.repeats}", flush=True)
            x = enc.run(hidden, n)
            times.append(enc.npu_s)
            if not np.isfinite(x).all():
                raise RuntimeError("nonfinite encoder output; stopping all runs")
            if baseline is None:
                baseline = x.copy()
            elif not np.array_equal(x, baseline):
                raise RuntimeError("encoder output changed on identical input; stopping all runs")
            print(f"clip {c}: completed encoder run {repeat + 1}/{args.repeats}, "
                  f"{1000 * enc.npu_s:.2f} ms", flush=True)
        out = x.T[None]
        text = model.decode(out, np.array([n]))
        row = {"clip": str(ref[f"name{c}"]), "frames": n, "npu_ms": round(1000 * min(times), 1),
               **err(ref[f"enc{c}"][0, :, :n], out[0]), "same_text": text == str(ref[f"text{c}"]),
               "text": text}
        rows.append(row)
        print(json.dumps(row), flush=True)
    nd.finish()


if __name__ == "__main__":
    main()
