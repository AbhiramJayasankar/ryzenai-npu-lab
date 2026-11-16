"""Experiment 014, lead 1: pull the key numbers out of the power-mode logs
(cache/014/power_<label>.log, written by 014_power_modes.ps1) into one table.

  python experiments\\014_power_summary.py default performance turbo default_after
Prints a table and writes cache/014/power_summary.json.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "cache" / "014"


def jobs(text):
    """Split a log into {job: output}, keeping the last run of each job."""
    out = {}
    for m in re.finditer(r"^=== (\w+) \[.*?\] \((\d+) s.*?\)\n(.*?)(?=^=== |^### |\Z)", text, re.S | re.M):
        out[m.group(1)] = m.group(3)
    return out


def first_json(s, key):
    """First one-line JSON object containing key. The timeout wrapper's output is
    wrapped at 120 columns, so join continuation lines until the object parses."""
    lines = s.splitlines()
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("{"):
            continue
        buf = line.strip()
        for j in range(i + 1, min(i + 40, len(lines)) + 1):
            try:
                d = json.loads(buf)
                if key in buf:
                    return d
                break
            except json.JSONDecodeError:
                if j < len(lines):
                    buf += lines[j].rstrip("\r")
    return None


def parse(label):
    text = (DIR / f"power_{label}.log").read_text(errors="replace")
    head = re.findall(r"^### .*$", text, re.M)
    j = jobs(text)
    r = {"label": label, "header": head}
    if "health" in j:
        blob = j["health"][j["health"].index("{"):]
        h = json.loads(blob[:blob.rindex("}") + 1])
        r["health"] = {"median_step_ms": h["runs"][0]["median_step_ms"], "max_errors": h["max_errors"],
                       "mismatches": h["runs"][0]["token_mismatches"]}
    if "decode" in j:
        blob = j["decode"][j["decode"].index("{"):]
        h = json.loads(blob[:blob.rindex("}") + 1])
        r["decode"] = {"median_step_ms": [x["median_step_ms"] for x in h["runs"]],
                       "ms_per_position": [x["ms_per_position"] for x in h["runs"]],
                       "mismatches": [x["token_mismatches"] for x in h["runs"]]}
    for k in ("bw4", "bw8"):
        if k in j:
            d = first_json(j[k], "gb_per_s")
            r[k] = {"median_ms": d["median_ms"], "gb_per_s": d["gb_per_s"], "check": d["check"]}
    for k in ("dma64m", "core1_16m", "gemm_bf16", "gemm_i8"):
        if k in j:
            d = first_json(j[k], '"case"')
            r[k] = {"run_ms": round(d["run"]["median_ms"], 4), "p10": round(d["run"]["p10_ms"], 4),
                    "p90": round(d["run"]["p90_ms"], 4), "ok": d["ok"],
                    "gbps": round(d.get("run_gbps", 0), 2), "gmacs": round(d.get("run_gmacs", 0), 1)}
    if "call" in j:
        for line in j["call"].splitlines():
            if '"part": "per_call"' in line or '"part": "call"' in line or '"wait"' in line:
                pass
        rows = [d for d in (first_json(c, "\"mode\"") for c in re.split(r"(?m)^(?={\"mode\")", j["call"])) if d]
        r["call"] = {row["part"]: {kk: (round(vv["median_us"], 1) if isinstance(vv, dict) and "median_us" in vv
                                        else None) for kk, vv in row.items() if isinstance(vv, dict)}
                     for row in rows}
    if "vbench" in j:
        r["vbench"] = {m.group(1).strip(): int(m.group(2))
                       for m in re.finditer(r"^(\w[\w ]*?)\s+max rel err \S+\s+~\s+(\d+) cycles", j["vbench"], re.M)}
    if "encoder" in j:
        e = j["encoder"]
        r["encoder"] = {"runs_ms": [float(x) for x in re.findall(r"completed encoder run \d+/\d+, ([\d.]+) ms", e)]}
        d = first_json(e, '"clip"')
        if d:
            r["encoder"].update({k: d[k] for k in ("frames", "npu_ms", "cosine", "rel_rms", "same_text")})
    if "context" in j:
        d = first_json(j["context"], "decode_step_ms")
        if d:
            r["context"] = d
    return r


def main():
    labels = sys.argv[1:] or ["default", "performance"]
    rows = [parse(x) for x in labels if (DIR / f"power_{x}.log").exists()]
    DIR.joinpath("power_summary.json").write_text(json.dumps(rows, indent=1))

    def g(r, *path):
        for p in path:
            if r is None:
                return None
            r = r.get(p) if isinstance(r, dict) else (r[p] if isinstance(r, list) and len(r) > p else None)
        return r
    lines = [
        ("LLM decode median step ms (--check)", ("health", "median_step_ms")),
        ("LLM decode median step ms (repeats)", ("decode", "median_step_ms")),
        ("LLM hidden max err", ("health", "max_errors", "hidden")),
        ("bw probe 4 streams GB/s (iron.jit)", ("bw4", "gb_per_s")),
        ("bw probe 8 streams GB/s (iron.jit)", ("bw8", "gb_per_s")),
        ("DMA copy 64 MB run ms", ("dma64m", "run_ms")),
        ("DMA copy 64 MB GB/s (r+w)", ("dma64m", "gbps")),
        ("1 core copy 16 MB GB/s (r+w)", ("core1_16m", "gbps")),
        ("GEMM bf16->f32 1024 ms", ("gemm_bf16", "run_ms")),
        ("GEMM i8 1024 ms", ("gemm_i8", "run_ms")),
        ("vbench exp16 cycles", ("vbench", "exp16")),
        ("vbench vmul cycles", ("vbench", "vmul")),
        ("encoder 1-block runs ms", ("encoder", "runs_ms")),
        ("encoder cosine", ("encoder", "cosine")),
        ("encoder same text", ("encoder", "same_text")),
        ("context 32 / 1024 / 4090 ms", None),
    ]
    print(f"{'':40s}" + "".join(f"{r['label']:>22s}" for r in rows))
    for name, path in lines:
        if path is None:
            vals = [f"{g(r, 'context', 'decode_step_ms', '32')}/{g(r, 'context', 'decode_step_ms', '1024')}/"
                    f"{g(r, 'context', 'decode_step_ms', '4090')}" for r in rows]
        else:
            vals = [str(g(r, *path)) for r in rows]
        print(f"{name:40s}" + "".join(f"{v:>22s}" for v in vals))
    for r in rows:
        if "call" in r:
            print(r["label"], "call:", json.dumps(r["call"])[:600])


if __name__ == "__main__":
    main()
