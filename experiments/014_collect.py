"""Experiment 014: gather the raw numbers into experiments/014_results.json.

Inputs (all in the ignored cache/014/): power_summary.json from
014_power_summary.py, int8_parakeet.json from 014_int8_parakeet.py,
int8_lfm25.json from 014_int8_lfm25.py. Per-utterance transcripts and
generated token ids are left out; they stay in the cache files.

  python experiments\\014_collect.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "cache" / "014"

# Lead 2: the byte count behind the encoder's GEMM rate (see 014_headroom_leads.md).
LINEAR_WEIGHTS = {  # (K, N) of the 8 weight GEMMs of one Conformer layer (qkv counted as one)
    "ffn1.w1": (1024, 4096), "ffn1.w2": (4096, 1024), "att.qkv": (1024, 3072), "att.out": (1024, 1024),
    "conv.pw1": (1024, 2048), "conv.pw2": (1024, 1024), "ffn2.w1": (1024, 4096), "ffn2.w2": (4096, 1024)}


def lead2():
    weights = sum(k * n for k, n in LINEAR_WEIGHTS.values())
    w_bytes = 2 * weights
    rows = []
    for frames, ms in ((128, 3.7), (256, 6.3), (384, 9.6)):  # experiment 012 measurements
        blocks = frames // 128
        ddr = blocks * 2 * w_bytes  # W once per block, X re-read: as many bytes as W
        rows.append({"frames": frames, "measured_ms": ms, "macs": weights * frames,
                     "tmacs": round(weights * frames / ms / 1e9, 3),
                     "ddr_bytes_read": ddr, "ddr_gb_per_s": round(ddr / ms / 1e6, 1)})
    return {"weights_per_layer": weights, "weight_bytes_bf16": w_bytes, "gemm_phases": rows,
            "per_core_stream_limit_macs_per_cycle": 64,
            "estimate_x_reuse_2_chunks_ms_per_layer_128": [2.6, 3.0],
            "estimate_encoder_saving_ms": [17, 26],
            "taka_et_al": {"cores": 16, "bf16_tile": "96x56x96", "bf16_output": True, "power_mode": "turbo",
                           "clock_ghz": 1.0, "ddr_gb_per_s": 15, "bf16_tmacs": 1.56, "int8_tmacs": 3.26,
                           "best_size": "4224x4032x4224 (bf16)", "code_published": False}}


def strip(d, keys):
    return {k: v for k, v in d.items() if k not in keys}


def main():
    out = {"experiment": "014", "date": "2026-11-16"}
    p = DIR / "power_summary.json"
    if p.exists():
        out["lead1_power_modes"] = json.loads(p.read_text())
    out["lead2_gemm"] = lead2()
    p = DIR / "int8_parakeet.json"
    if p.exists():
        d = json.loads(p.read_text())
        out["lead3_parakeet"] = {"utterances": d["utterances"], "frames": d["frames"],
                                 "variants": {k: strip(v, {"items"}) for k, v in d["variants"].items()}}
    p = DIR / "int8_lfm25.json"
    if p.exists():
        d = json.loads(p.read_text())
        out["lead3_lfm25"] = {k: {**strip(v, {"greedy"}),
                                  "greedy": [strip(g, {"ids"}) for g in v["greedy"]]} for k, v in d.items()}
    (ROOT / "experiments" / "014_results.json").write_text(json.dumps(out, indent=1))
    print("wrote experiments/014_results.json")


if __name__ == "__main__":
    main()
