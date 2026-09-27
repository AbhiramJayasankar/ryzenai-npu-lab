"""Experiment 012: the full 24-layer encoder on the NPU vs ONNX Runtime FP32.

For the clips of cache/parakeet/reference_dyn.npz that fit the program's
frame bucket: subsampling prefix on the CPU (ONNX Runtime), 24 Conformer
layers in one NPU submission, TDT decoding on the CPU. Reports the encoder
output error against the FP32 encoder, the transcript, and NPU time.

  powershell -NoProfile -File scripts\\run_npu_timeout.ps1 -Script experiments\\012_encoder_test.py -Seconds 3600
"""

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
    nblk = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    t = time.perf_counter()
    enc = pm.PkEncoder(nblk=nblk, n_layers=n_layers)
    print(f"setup {time.perf_counter() - t:.1f} s (pack/load {enc.pack_s:.1f}, compile {enc.compile_s:.1f}), "
          f"{len(enc.phases)} phases", flush=True)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    prefix = ort.InferenceSession(str(pp.CACHE / "models" / "encoder_dyn_prefix.onnx"), so,
                                  providers=["CPUExecutionProvider"])
    model = pp.Parakeet("encoder-model.int8.onnx")  # preprocessor and decoder only
    ref = np.load(pp.CACHE / "reference_dyn.npz")
    rows = []
    for c in range(sum(1 for k in ref.files if k.startswith("feats"))):
        hidden = prefix.run(["hidden"], {"audio_signal": ref[f"feats{c}"], "length": ref[f"len{c}"]})[0][0]
        n = hidden.shape[0]
        if n > enc.lay.TPAD:
            continue
        times = []
        for _ in range(3):
            x = enc.run(hidden, n)
            times.append(enc.npu_s)
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
