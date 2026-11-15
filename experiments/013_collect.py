"""Experiment 013: gather raw results into experiments/013_results.json and print
comparison tables (NPU vs best CPU vs GPU) used in 013_raw_capabilities.md.

  python experiments/013_collect.py [--md]
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "cache/013"


def jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def last_per_case(rows):
    d = {}
    for r in rows:
        d[r["case"]] = r
    return d


# NPU case -> (cpu/gpu op, size key) for comparison
def match(case):
    m = re.match(r"memcpy_dma_(\w+)$", case)
    sizes = {"4k": 1024, "32k": 8192, "1m": 262144, "16m": 4 << 20, "64m": 16 << 20}
    if m:
        return "memcpy", sizes[m.group(1)]
    m = re.match(r"memcpy_core\d_(\w+)$", case)
    if m:
        return "memcpy", sizes[m.group(1)]
    n = {"1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192, "32k": 32768, "256k": 262144,
         "1m": 1 << 20, "4m": 4 << 20, "16m": 16 << 20}
    m = re.match(r"(saxpy|add|mul|relu|gelu|exp|softmax)(?:_bf16)?_(\w+)$", case)
    if m:
        return m.group(1), n[m.group(2)]
    m = re.match(r"(max|sum)_(i32|bf16)_\w+_(\w+)$", case)
    if m:
        return m.group(1), n[m.group(3)]
    m = re.match(r"gemv_bf16_\dcore_(\d+x\d+)$", case)
    if m:
        return "gemv", m.group(1)
    m = re.match(r"mv_i16_1core_(\d+)$", case)
    if m:
        return "mv_i16", f"{m.group(1)}x{m.group(1)}"
    m = re.match(r"gemm_(bf16f32|bf16|i8|i16)_(wa|1core)_(\d+)$", case)
    if m:
        op = {"bf16f32": "gemm", "bf16": "gemm", "i8": "gemm_i8", "i16": "gemm_i16"}[m.group(1)]
        return op, int(m.group(3))
    m = re.match(r"conv1x1_i8_(\d+)x(\d+)x(\d+)$", case)
    if m:
        w, h, c = m.groups()
        return "conv1x1", f"{w}x{h}x{c}x{c}"
    return None, None


def env():
    e = {}
    for name in ("environment.json",):
        p = C / name
        if p.exists():
            e.update(json.loads(p.read_text()))
    return e


def parse_vbench():
    p = C / "vbench.txt"
    out = []
    if p.exists():
        for line in p.read_text(errors="ignore").splitlines():
            m = re.match(r"(.+?)\s+max rel err (\S+)\s+~\s*(\d+) cycles per vector", line.strip())
            if m:
                out.append({"routine": m.group(1).strip(), "max_rel_err": float(m.group(2)),
                            "cycles_per_16_lane_vector_at_1ghz_assumed": int(m.group(3))})
    return out


def parse_validate(name):
    p = C / name
    if not p.exists():
        return None
    t = p.read_text(errors="ignore")
    m = re.search(r"Average (latency|throughput): ([\d.]+) (\S+)", t)
    pm = re.search(r"Power Mode\s*:\s*(\S+)", t)
    return {"value": float(m.group(2)), "unit": m.group(3), "power_mode_during_test": pm.group(1) if pm else None} if m else None


def main():
    npu_all = jsonl(C / "npu.jsonl")
    npu = last_per_case(npu_all)
    last_index = {r["case"]: i for i, r in enumerate(npu_all)}
    superseded = [r for i, r in enumerate(npu_all) if last_index[r["case"]] != i]
    base = jsonl(C / "cpu_gpu.jsonl")
    res = {
        "experiment": "013 raw NPU capabilities",
        "environment": env(),
        "npu": list(npu.values()),
        "npu_superseded_runs": superseded,  # earlier runs of a case that was rerun
        "cpu_gpu": base,
        "call_overhead": jsonl(C / "call_overhead.jsonl"),
        "vbench_012": parse_vbench(),
        "xrt_smi_validate": {"latency": parse_validate("xrt_validate_latency.txt"),
                             "throughput": parse_validate("xrt_validate_throughput.txt")},
    }
    (ROOT / "experiments/013_results.json").write_text(json.dumps(res, indent=1))

    def best(op, size, pred):
        rows = [r for r in base if r.get("op") == op and r.get("size") == size and "median_ms" in r and pred(r)]
        return min(rows, key=lambda r: r["median_ms"]) if rows else None

    print("| NPU case | NPU run ms | NPU e2e ms | best CPU ms (backend, dtype, threads) | GPU compute ms | GPU with copies ms | ok |")
    print("|---|---:|---:|---|---:|---:|---|")
    for case, r in npu.items():
        op, size = match(case)
        cpu = best(op, size, lambda x: x["backend"] != "torch_cuda")
        gc = best(op, size, lambda x: x["backend"] == "torch_cuda" and x["mode"] == "compute")
        gt = best(op, size, lambda x: x["backend"] == "torch_cuda" and x["mode"] == "transfer")
        f = lambda x: f"{x['median_ms']:.4g}" if x else "-"
        cdesc = f"{f(cpu)} ({cpu['backend']}, {cpu['dtype']}, {cpu.get('threads')})" if cpu else "-"
        gdesc = f"{f(gc)} ({gc['dtype']})" if gc else "-"
        tdesc = f"{f(gt)} ({gt['dtype']})" if gt else "-"
        print(f"| {case} | {r['run']['median_ms']:.4g} | {r['e2e']['median_ms']:.4g} | {cdesc} | {gdesc} | {tdesc} | {r['ok']} |")


if __name__ == "__main__":
    main()
