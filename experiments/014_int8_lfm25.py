"""Experiment 014, lead 3 (LLM half): what 8-bit weights would do to LFM2.5-230M.

CPU only, no NPU. A self-contained PyTorch implementation of the LFM2
forward pass (system Python has PyTorch but not `transformers`, so the
checkpoint is read directly from model.safetensors). Number formats:

  fp32act the BF16 weights with FP32 activations (no BF16 rounding between
          operations): shows how far BF16 rounding alone moves the results
  bf16    the checkpoint's BF16 weights, BF16 activations at the model's
          rounding points, FP32 sums inside every matrix product (what the
          experiment 010 NPU engine and PyTorch BF16 compute)
  w8c     int8 weights, one FP32 scale per output row; BF16 activations, FP32
          sums, scale applied after the sum (an NPU would unpack int8 to BF16,
          which is exact, and reuse its BF16 multiply path)
  w8g64   int8 weights, one scale per output row and group of 64 inputs
  w8g32   the same with groups of 32
  w4g32   int4 weights, groups of 32 (an extra point, for the int8 x int4 mode)

All projections (conv in/out, q/k/v/out, FFN w1/w2/w3) and the tied
embedding table (used both for the input lookup and for the vocabulary head;
its scale per row serves both) are quantized. Norms and the 3-tap conv
weights stay BF16.

Checks:
  1. the 23-position fixture of experiment 006 (teacher-forced CPU BF16 from
     transformers): final hidden state error and argmax token per position
  2. teacher-forced next-token loss on 7.7K tokens of text (from
     014_llm_tokenize.py): perplexity, top-1 agreement with bf16, KL to bf16
  3. greedy generation, 48 tokens, for the three prompts of 010_cpu_greedy.json:
     agreement with transformers' tokens and with this file's bf16

  python experiments\\014_int8_lfm25.py
Writes cache/014/int8_lfm25.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "cache" / "lfm25-230m" / "model.safetensors"
FIXTURE = ROOT / "cache" / "lfm25-prompt-sequence-reference.npz"
OUT = ROOT / "cache" / "014" / "int8_lfm25.json"
CFG = json.loads((ROOT / "cache" / "lfm25-230m" / "config.json").read_text())
D, HEADS, KV, HD = 1024, 16, 8, 64
EPS = CFG["norm_eps"]
THETA = CFG["rope_parameters"]["rope_theta"]
KINDS = CFG["layer_types"]
BF = torch.bfloat16
ACT = BF  # activation dtype at the rounding points; FP32 for the "fp32act" reference
VARIANTS = {"bf16": None, "fp32act": None, "w8c": (8, None), "w8g64": (8, 64), "w8g32": (8, 32), "w4g32": (4, 32)}
PROJ = ["conv.in_proj", "conv.out_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.out_proj", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"]


def read_checkpoint():
    with CKPT.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    start = 8 + n
    out = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        assert spec["dtype"] == "BF16", name
        a, b = spec["data_offsets"]
        raw = np.fromfile(CKPT, dtype=np.uint16, count=(b - a) // 2, offset=start + a)
        out[name] = torch.from_numpy(raw.astype(np.int16)).view(BF).reshape(spec["shape"]).clone()
    return out


class Weight:
    """A [out, in] matrix: BF16, or int (as int8 storage) with FP32 scales per row/group."""

    def __init__(self, w, fmt):
        self.fmt = fmt
        if fmt is None:
            self.w = w
            return
        bits, group = fmt
        qmax = 2 ** (bits - 1) - 1
        o, i = w.shape
        g = group or i
        wf = w.float().reshape(o, i // g, g)
        s = wf.abs().amax(-1, keepdim=True) / qmax
        s[s == 0] = 1.0
        self.q = torch.clamp(torch.round(wf / s), -qmax, qmax).to(torch.int8)
        self.s = s
        self.err = float(((self.q.float() * s - wf) ** 2).mean().sqrt() / wf.pow(2).mean().sqrt())

    def dense(self, rows=None):
        """FP32 dequantized weight (optionally a row slice)."""
        if self.fmt is None:
            w = self.w if rows is None else self.w[rows]
            return w.float()
        q = self.q if rows is None else self.q[rows]
        s = self.s if rows is None else self.s[rows]
        return (q.float() * s).reshape(q.shape[0], -1)

    def linear(self, x):
        """BF16 activations in, FP32 sum, one BF16 rounding of the result."""
        xf = x.float()
        if self.fmt is not None and self.fmt[1] is None:  # per-row scale after the int sum
            return (F.linear(xf, self.q.reshape(self.q.shape[0], -1).float()) * self.s.reshape(1, -1)).to(ACT)
        return F.linear(xf, self.dense()).to(ACT)

    def rows(self, ids):
        return self.dense(ids).to(ACT)


def rms(x, w):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return w * xf.to(ACT)


def rope(n):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.int64).float() / HD))
    f = torch.arange(n).float()[:, None] * inv[None]
    emb = torch.cat([f, f], -1)
    return emb.cos().to(ACT), emb.sin().to(ACT)


def rot(x):
    a, b = x[..., :HD // 2], x[..., HD // 2:]
    return torch.cat([-b, a], -1)


class Model:
    def __init__(self, ck, fmt):
        self.fmt = fmt
        self.emb = Weight(ck["model.embed_tokens.weight"], fmt)
        self.norm = ck["model.embedding_norm.weight"]
        self.layers = []
        errs = []
        for i, kind in enumerate(KINDS):
            p = f"model.layers.{i}."
            L = {"kind": kind, "op_norm": ck[p + "operator_norm.weight"], "ffn_norm": ck[p + "ffn_norm.weight"]}
            for name in PROJ:
                if p + name + ".weight" in ck:
                    L[name] = Weight(ck[p + name + ".weight"], fmt)
                    if fmt:
                        errs.append(L[name].err)
            if kind == "conv":
                L["conv_w"] = ck[p + "conv.conv.weight"]
            else:
                L["q_ln"] = ck[p + "self_attn.q_layernorm.weight"]
                L["k_ln"] = ck[p + "self_attn.k_layernorm.weight"]
            self.layers.append(L)
        self.weight_rel_rms = round(float(np.mean(errs)), 5) if errs else 0.0
        self.emb_rel_rms = round(self.emb.err, 5) if fmt else 0.0

    def forward(self, ids, logits_for="all"):
        """Teacher-forced causal pass over ids (list). Returns (hidden before the
        final norm [T, D], logits FP32 [T or 1, V])."""
        t = len(ids)
        x = self.emb.rows(torch.tensor(ids))
        cos, sin = rope(t)
        for L in self.layers:
            h = rms(x, L["op_norm"])
            if L["kind"] == "conv":
                bcx = L["conv.in_proj"].linear(h)
                b, c, xx = bcx.split(D, -1)
                bx = (b * xx).T[None]  # [1, D, T]
                conv = F.conv1d(bx.float(), L["conv_w"].float(), padding=2, groups=D)[..., :t].to(ACT)
                y = c * conv[0].T
                h = L["conv.out_proj"].linear(y)
            else:
                q = rms(L["self_attn.q_proj"].linear(h).view(t, HEADS, HD), L["q_ln"])
                k = rms(L["self_attn.k_proj"].linear(h).view(t, KV, HD), L["k_ln"])
                v = L["self_attn.v_proj"].linear(h).view(t, KV, HD)
                q = q * cos[:, None] + rot(q) * sin[:, None]
                k = k * cos[:, None] + rot(k) * sin[:, None]
                q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
                k = k.repeat_interleave(HEADS // KV, 0)
                v = v.repeat_interleave(HEADS // KV, 0)
                a = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0]
                h = L["self_attn.out_proj"].linear(a.transpose(0, 1).reshape(t, D))
            x = x + h
            f = rms(x, L["ffn_norm"])
            g = F.silu(L["feed_forward.w1"].linear(f)) * L["feed_forward.w3"].linear(f)
            x = x + L["feed_forward.w2"].linear(g)
        hid = x
        n = rms(x if logits_for == "all" else x[-1:], self.norm)
        logits = self._head(n)
        return hid, logits

    def _head(self, n):
        out = []
        for r in range(0, 65536, 8192):
            if self.fmt is None:
                out.append(F.linear(n.float(), self.emb.w[r:r + 8192].float()))
            elif self.fmt[1] is not None:  # group scales: dequantized slice
                out.append(F.linear(n.float(), self.emb.dense(slice(r, r + 8192))))
            else:  # per-row scale applied after the sum
                out.append(F.linear(n.float(), self.emb.q[r:r + 8192].reshape(8192, -1).float())
                           * self.emb.s[r:r + 8192].reshape(1, -1))
        return torch.cat(out, -1)

    def greedy(self, prompt, n):
        ids = list(prompt)
        for _ in range(n):
            _h, lg = self.forward(ids, logits_for="last")
            ids.append(int(lg[-1].argmax()))
        return ids[len(prompt):]


def main():
    global ACT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    names = ap.parse_args().variants.split(",")
    torch.set_num_threads(8)
    t0 = time.perf_counter()
    ck = read_checkpoint()
    fx = np.load(FIXTURE)
    fx_ids = [int(i) for i in fx["consumed_ids"]]
    greedy_ref = json.loads((ROOT / "cache" / "010_cpu_greedy.json").read_text())
    texts = json.loads((ROOT / "cache" / "014" / "llm_text_ids.json").read_text())
    print(f"checkpoint read in {time.perf_counter() - t0:.1f} s; {sum(len(x['ids']) for x in texts)} text tokens",
          flush=True)
    results, base = {}, {}
    with torch.inference_mode():
        for name, fmt in VARIANTS.items():
            if name not in names:
                continue
            ACT = torch.float32 if name == "fp32act" else BF
            t1 = time.perf_counter()
            m = Model(ck, fmt)
            r = {"weight_rel_rms": m.weight_rel_rms, "embedding_rel_rms": m.emb_rel_rms}
            # 1. fixture
            hid, lg = m.forward(fx_ids)
            ref_h = np.stack([fx[f"p{p}_hidden14_raw"] for p in range(len(fx_ids))])
            got = hid.float().numpy()
            r["fixture"] = {
                "hidden_max_abs_err": round(float(np.abs(got - ref_h).max()), 4),
                "hidden_rel_rms": round(float(np.sqrt(((got - ref_h) ** 2).mean() / (ref_h ** 2).mean())), 5),
                "argmax_mismatch_positions": [p for p in range(len(fx_ids))
                                              if int(lg[p].argmax()) != int(np.argmax(fx[f"p{p}_logits"]))]}
            if name == "bf16":
                base["fx_hid"] = got
            else:
                b = base["fx_hid"]
                r["fixture"]["hidden_rel_rms_vs_bf16"] = round(float(np.sqrt(((got - b) ** 2).mean() / (b ** 2).mean())), 5)
            # 2. text loss
            nll = n_tok = agree = 0.0
            kl = 0.0
            for k, x in enumerate(texts):
                _h, lg = m.forward(x["ids"])
                lp = torch.log_softmax(lg[:-1], -1)
                tgt = torch.tensor(x["ids"][1:])
                nll += float(-lp[torch.arange(len(tgt)), tgt].sum())
                n_tok += len(tgt)
                top = lg[:-1].argmax(-1)
                if name != "bf16":  # bf16 distribution recomputed (all of them do not fit in RAM)
                    mine, ACT = ACT, BF  # the reference always runs in BF16
                    _h, blg = ref_model.forward(x["ids"])
                    ACT = mine
                    blp = torch.log_softmax(blg[:-1], -1)
                    agree += float((top == blg[:-1].argmax(-1)).sum())
                    kl += float((blp.exp() * (blp - lp)).sum())
                    del blg, blp
            r["text"] = {"tokens": int(n_tok), "perplexity": round(float(np.exp(nll / n_tok)), 4)}
            if name != "bf16":
                r["text"]["top1_agreement_with_bf16"] = round(agree / n_tok, 5)
                r["text"]["mean_kl_to_bf16"] = round(kl / n_tok, 6)
            # 3. greedy
            g = []
            for k, x in enumerate(greedy_ref):
                gen = m.greedy(x["prompt_ids"], len(x["generated_ids"]))
                same_hf = next((i for i, (a, b) in enumerate(zip(gen, x["generated_ids"])) if a != b), len(gen))
                row = {"prompt": x["prompt"], "tokens": len(gen), "first_diff_vs_transformers": same_hf}
                if name == "bf16":
                    base.setdefault("gen", []).append(gen)
                else:
                    row["first_diff_vs_bf16"] = next((i for i, (a, b) in enumerate(zip(gen, base["gen"][k])) if a != b),
                                                     len(gen))
                row["ids"] = gen
                g.append(row)
            r["greedy"] = g
            r["seconds"] = round(time.perf_counter() - t1, 1)
            results[name] = r
            if name == "bf16":
                ref_model = m
            print(name, json.dumps({k: v for k, v in r.items() if k != "greedy"}),
                  [(x["first_diff_vs_transformers"], x.get("first_diff_vs_bf16")) for x in g], flush=True)
            if name != "bf16":
                del m
    OUT.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    old.update(results)
    OUT.write_text(json.dumps({k: old[k] for k in VARIANTS if k in old}, indent=1))
    print(f"wrote {OUT} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
