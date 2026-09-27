"""Read-only audit of Parakeet's straight-line cached IRON task sequence.

Usage: python experiments/012_cache_audit.py <cache-dir-or-aie.mlir> [...]
No IRON, XRT, or hardware imports. Counts starts between group waits/frees,
not actual queue occupancy (DMA progress is asynchronous and unknowable here).
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def audit(path):
    path = Path(path)
    if path.is_dir():
        path = path / "aie.mlir"
    text = path.read_text()
    configs, groups, starts, awaits = {}, [], [], []
    closing = False

    def finish():
        if starts:
            counts = Counter(configs[t][0] for t in starts)
            drains = [i for i, t in enumerate(starts) if configs[t][0].startswith("cj_")]
            late = bool(drains) and any(
                not configs[t][0].startswith("cj_") for t in starts[:max(drains)])
            groups.append(dict(starts=len(starts), per_channel=dict(counts),
                               fills_before_drains=late,
                               waits=[configs[t][0] for t in awaits],
                               first_line=configs[starts[0]][1]))

    in_sequence = False
    for line_no, line in enumerate(text.splitlines(), 1):
        if "aie.runtime_sequence(" in line:
            in_sequence = True
        if not in_sequence:
            continue
        cfg = re.search(r"(%\w+) = aiex.dma_configure_task_for @(\w+)", line)
        if cfg:
            if closing:
                finish()
                starts, awaits, closing = [], [], False
            configs[cfg[1]] = (cfg[2].removesuffix("_shim_alloc"), line_no)
        op = re.search(r"aiex.dma_(start|await|free)_task\((%\w+)\)", line)
        if op:
            if op[1] == "start":
                starts.append(op[2])
            else:
                closing = True
                if op[1] == "await":
                    awaits.append(op[2])
    finish()
    if not groups:
        raise ValueError(f"No supported straight-line tasks in {path}")
    peaks = {}
    for g in groups:
        for key, count in g["per_channel"].items():
            peaks[key] = max(peaks.get(key, 0), count)
    oversized = [dict(group=i, **g) for i, g in enumerate(groups)
                 if max(g["per_channel"].values()) > 4]
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                groups=len(groups), starts=sum(g["starts"] for g in groups), peaks=peaks,
                groups_over_queue_bound=len(oversized),
                groups_with_fills_before_drains=sum(g["fills_before_drains"] for g in groups),
                input_only_groups=sum(not any(k.startswith("cj_") for k in g["per_channel"])
                                      for g in groups),
                first_oversized=oversized[0] if oversized else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        print(json.dumps(audit(path), indent=2))
