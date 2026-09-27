"""Validate the hybrid Parakeet encoder (Conformer linears on the NPU) against ONNX Runtime FP32.

  powershell -NoProfile -File scripts\\run_npu_timeout.ps1 -Script experiments\\011_hybrid_validate.py -Seconds 1800

Needs cache/parakeet/reference_T1000.npz and reference_dyn.npz
(011_reference_outputs.py), the split FP32 layer 0-1 block and prefixes
(011_split.py), exported weights and position tables (011_export_weights.py,
011_export_pos.py). Steps:
 1. layers 0-1 (fixed 10 s shape): ORT FP32 block vs NumPy FP32 layers (checks the
    re-implementation), vs NumPy with BF16-rounded matmul inputs (expected NPU
    numerics), vs the NPU hybrid;
 2. the full hybrid encoder (dynamic-length prefix, 24 layers) on each clip of
    reference_dyn.npz (4 clips < 10 s and a 30.6 s one): error vs the ORT FP32
    encoder output, transcript vs the FP32 transcript, timing.
Errors are measured over the real (unpadded) frames.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_hybrid as ph  # noqa: E402  (first: sets OPENBLAS_NUM_THREADS before NumPy loads BLAS)

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import parakeet_pipeline as pp  # noqa: E402

import npu_direct as nd  # noqa: E402

M = pp.CACHE / "models"


def err(ref, got):
    a, b = ref.ravel().astype(np.float64), got.ravel().astype(np.float64)
    return {"cosine": round(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), 6),
            "rel_rms": round(float(np.sqrt(np.mean((a - b) ** 2) / np.mean(a ** 2))), 5),
            "max_abs": round(float(np.abs(a - b).max()), 4)}


def main():
    ref = np.load(pp.CACHE / "reference_T1000.npz")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    prefix = ort.InferenceSession(str(M / "encoder_T1000_prefix.onnx"), so, providers=["CPUExecutionProvider"])
    block = ort.InferenceSession(str(M / "encoder_T1000_0_1.onnx"), so, providers=["CPUExecutionProvider"])
    pos = ph.pos_for(125)
    result = {}

    hidden, att_mask, pad_mask = prefix.run(["hidden", "att_mask", "pad_mask"],
                                            {"audio_signal": ref["feats0"], "length": ref["len0"]})
    valid = int((~pad_mask).sum())
    assert np.array_equal(att_mask[0, 0], ~(~pad_mask[0, 0][:, None] & ~pad_mask[0, 0][None, :])), "mask layout"
    ort_out = block.run(None, {"hidden": hidden, "att_mask": att_mask, "pad_mask": pad_mask})[0][0]
    two = [ph.load_layer(0), ph.load_layer(1)]
    for label, mm in (("numpy_fp32", ph.CpuMatmul(two)), ("numpy_bf16_inputs", ph.CpuMatmul(two, bf16=True))):
        x = hidden[0]
        for i in range(2):
            x = ph.conformer_layer(x, two[i], pos[i], att_mask[0, 0], valid, mm, i)
        result[f"layers01_{label}_vs_ort"] = err(ort_out[:valid], x[:valid])
    del two

    weights = ph.NpuWeights()
    enc = ph.HybridEncoder(M / "encoder_dyn_prefix.onnx", weights)
    npu = enc.mms[128]
    result["npu_setup"] = {"pack_weights_s": round(weights.pack_s, 1),
                           "compile_or_cache_s": {r: round(m.compile_s, 1) for r, m in enc.mms.items()}}
    x = hidden[0]
    for i in range(2):
        x = ph.conformer_layer(x, weights.small[i], pos[i], att_mask[0, 0], valid, npu, i)
    result["layers01_npu_vs_ort"] = err(ort_out[:valid], x[:valid])
    print(json.dumps(result, indent=1), flush=True)

    ref = np.load(pp.CACHE / "reference_dyn.npz")
    n_clips = sum(1 for k in ref.files if k.startswith("feats"))
    model = pp.Parakeet(enc)
    for c in range(n_clips):  # warm-up every bucket used
        enc.run(None, {"audio_signal": ref[f"feats{c}"], "length": ref[f"len{c}"]})
    clips = []
    for c in range(n_clips):
        feed = {"audio_signal": ref[f"feats{c}"], "length": ref[f"len{c}"]}
        t = time.perf_counter()
        for m in enc.mms.values():
            m.reset()
        out, out_len = enc.run(None, feed)
        enc_s = time.perf_counter() - t
        npu = enc.last_mm
        n = int(out_len[0])
        text = model.decode(out, out_len)
        clips.append({"clip": str(ref[f"name{c}"]), "valid_frames": n, "gemm_rows": npu.rows,
                      "vs_ort_fp32": err(ref[f"enc{c}"][0, :, :n], out[0, :, :n]),
                      "encoder_ms": round(1000 * enc_s, 1), "npu_gemm_ms": round(1000 * npu.npu_s, 1),
                      "gemm_calls": npu.calls, "text": text, "fp32_text": str(ref[f"text{c}"]),
                      "same_text": text == str(ref[f"text{c}"])})
        print(json.dumps(clips[-1]), flush=True)
    result["clips"] = clips
    (pp.CACHE / "results").mkdir(parents=True, exist_ok=True)
    (pp.CACHE / "results" / "hybrid_validate.json").write_text(json.dumps(result, indent=1))
    nd.finish()


if __name__ == "__main__":
    main()
