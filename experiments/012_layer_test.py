"""Experiment 012: op-by-op check of the all-NPU encoder on layer 0.

Runs the encoder program truncated after phase k (one compiled program per
checkpoint) on test.wav's subsampling output and compares the field the last
phase wrote with a NumPy mirror of the same computation that rounds to BF16
at the same points (FP32 math in between). Errors are over the real frames.

  powershell -NoProfile -File scripts\\run_npu_timeout.ps1 -Script experiments\\012_layer_test.py -ScriptArgs "1 2 3" -Seconds 1800

Phase indices of layer 0 (after the initial LayerNorm, index 0):
  1 FFN1 W1+Swish, 2 FFN1 W2, 3 LN, 4 q|k|v, 5-12 position scores (8 heads),
  13 attention, 14 out, 15 LN, 16 pointwise1, 17 conv module, 18 pointwise2,
  19 LN, 20 FFN2 W1+Swish, 21 FFN2 W2, 22 final LN (whole layer).
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from ml_dtypes import bfloat16

import npu_direct as nd
import pk_engine as pk
import pk_model as pm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_pipeline as pp  # noqa: E402

KEEP = []


def r(x):
    return np.asarray(x, np.float32).astype(bfloat16).astype(np.float32)


def ln(x, w, name):
    mu = x.mean(-1, keepdims=True)
    v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return r((x - mu) / np.sqrt(v + 1e-5) * w[f"{name}.w"] + w[f"{name}.b"])


def sig(x):
    return 1 / (1 + np.exp(-x))


def mirror(hidden, valid, TPAD):
    """Layer 0 with BF16 rounding at the NPU's field boundaries; returns the
    field values after every phase (dict phase -> (field, array))."""
    w = pm.load_layer(0)
    w1 = pm.load_layer(1)
    pos = pm.pos_tables(TPAD)[0]
    s = pm.SCALE
    x = np.zeros((TPAD, 1024), np.float32)
    x[:hidden.shape[0]] = r(hidden)
    out = {}
    a = ln(x, w, "norm_feed_forward1")
    out[0] = ("a", a)
    h = r((lambda z: z * sig(z))(r(a) @ r(w["feed_forward1.w1"])))
    out[1] = ("big", h)
    y = r(h @ r(w["feed_forward1.w2"]))
    out[2] = ("y", y)
    x = r(x + 0.5 * y)
    a = ln(x, w, "norm_self_att")
    out[3] = ("a", a)
    qkv_w = np.concatenate([w["att.wq"] * s, w["att.wk"], w["att.wv"]], axis=1)
    bias = np.concatenate([w["att.bias_v"].reshape(-1) * s, np.zeros(2048, np.float32)])
    qkv = r(a @ r(qkv_w) + r(bias))
    out[4] = ("big", qkv)
    q, k, v = qkv[:, :1024], qkv[:, 1024:2048], qkv[:, 2048:]
    bd = np.stack([r(q[:, h * 128:(h + 1) * 128] @ r(pos[h])) for h in range(8)])
    out[12] = ("bd", bd)
    delta = r((w["att.bias_u"] - w["att.bias_v"]) * s)
    o = np.zeros((TPAD, 1024), np.float32)
    for h in range(8):
        qu = r(q[:, h * 128:(h + 1) * 128] + delta[h])
        sc = qu @ k[:, h * 128:(h + 1) * 128].T
        i = np.arange(TPAD)[:, None]
        j = np.arange(TPAD)[None, :]
        sc = sc + bd[h][i, TPAD - 1 - i + j]
        sc[:, valid:] = -np.inf
        p = np.exp(sc - sc.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        o[:, h * 128:(h + 1) * 128] = r(p @ v[:, h * 128:(h + 1) * 128])
    out[13] = ("o", o)
    y = r(o @ r(w["att.wout"]))
    out[14] = ("y", y)
    x = r(x + y)
    a = ln(x, w, "norm_conv")
    out[15] = ("a", a)
    c = r(a @ r(w["conv.pw1"]))
    out[16] = ("big", c)
    g = r(c[:, :1024] * sig(c[:, 1024:]))
    g[valid:] = 0
    gp = np.pad(g, ((4, 4), (0, 0)))
    dw = r(w["conv.dw"])
    yc = sum(gp[j:j + TPAD] * dw[:, j] for j in range(9)) + r(w["conv.dw_b"])
    sconv = r(yc * sig(yc))
    out[17] = ("o", sconv)
    y = r(sconv @ r(w["conv.pw2"]))
    out[18] = ("y", y)
    x = r(x + y)
    a = ln(x, w, "norm_feed_forward2")
    out[19] = ("a", a)
    h = r((lambda z: z * sig(z))(a @ r(w["feed_forward2.w1"])))
    out[20] = ("big", h)
    y = r(h @ r(w["feed_forward2.w2"]))
    out[21] = ("y", y)
    x = ln(r(x + 0.5 * y), w, "norm_out")
    out[22] = ("x", x)
    out["a22"] = ln(x, w1, "norm_feed_forward1")
    return out


def compare(ref, got):
    a, b = ref.ravel().astype(np.float64), got.ravel().astype(np.float64)
    return {"cosine": round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)), 6),
            "rel_rms": round(float(np.sqrt(np.mean((a - b) ** 2) / (np.mean(a ** 2) + 1e-30))), 5),
            "max_abs": round(float(np.abs(a - b).max()), 4)}


def main():
    ks = [int(v) for v in sys.argv[1:]] or [0, 1, 2, 3, 4, 12, 13, 14, 15, 16, 17, 18, 22]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    prefix = ort.InferenceSession(str(pp.CACHE / "models" / "encoder_dyn_prefix.onnx"), so,
                                  providers=["CPUExecutionProvider"])
    m = pp.Parakeet("encoder-model.int8.onnx")  # only its preprocessor is used
    feats, lens = m.features(pp.load_audio(pp.ROOT.parent / "parakeet-stt" / "test.wav"))
    hidden = prefix.run(["hidden"], {"audio_signal": feats, "length": lens})[0][0]
    valid = hidden.shape[0]
    ref = mirror(hidden, valid, 128)
    for k in ks:
        enc = pm.PkEncoder(nblk=1, n_layers=1, max_phases=k + 1)
        KEEP.append(enc)
        enc.run(hidden, valid)
        key = 12 if 5 <= k <= 12 else k
        field, want = ref[key]
        if field == "bd":
            got = enc.bd()[:, :valid, :256]
            want = want[:, :valid, :256]
            res = compare(want[:k - 4], got[:k - 4])
        else:
            got = enc.field(field)
            n = want.shape[1]
            res = compare(want[:valid], got[:valid, :n])
        print(json.dumps({"phase": k, "field": field, "npu_ms": round(1000 * enc.npu_s, 2), **res}),
              flush=True)
    nd.finish()


if __name__ == "__main__":
    main()
