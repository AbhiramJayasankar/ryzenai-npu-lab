"""Debug helper: run the first k phases (one layer) of a frame-block program once.
  run_npu_timeout.ps1 -Script experiments/012_hang_probe.py -ScriptArgs "<nblk> <k> [clip]"
Input: random frames (100*nblk valid), or clip i of cache/parakeet/reference_dyn.npz
(real subsampling output). One run only: never loop this over hang candidates.
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

import npu_direct as nd
import pk_model as pm

nblk, k = int(sys.argv[1]), int(sys.argv[2])
if len(sys.argv) > 3:
    ref = np.load(pm.CACHE / "reference_dyn.npz")
    c = int(sys.argv[3])
    so = ort.SessionOptions()
    so.log_severity_level = 3
    prefix = ort.InferenceSession(str(pm.CACHE / "models" / "encoder_dyn_prefix.onnx"), so,
                                  providers=["CPUExecutionProvider"])
    hidden = prefix.run(["hidden"], {"audio_signal": ref[f"feats{c}"], "length": ref[f"len{c}"]})[0][0]
else:
    hidden = np.random.default_rng(0).standard_normal((100 * nblk, 1024)).astype(np.float32)
if len(sys.argv) > 4:
    hidden = hidden * float(sys.argv[4])
print(f"input frames {hidden.shape[0]}, max |x| {np.abs(hidden).max():.1f}", flush=True)
enc = pm.PkEncoder(nblk=nblk, n_layers=1, max_phases=k)
x = enc.run(hidden, hidden.shape[0])
print(f"nblk {nblk} phases {k}: ok, {1000 * enc.npu_s:.2f} ms, finite {np.isfinite(x).all()}", flush=True)
nd.finish()
