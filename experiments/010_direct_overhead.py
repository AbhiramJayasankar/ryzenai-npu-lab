"""Per-call overhead of direct XRT submission and sub-buffer argument offsets."""

import importlib
import json
import statistics
import time

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16

import npu_direct as nd

probe_mod = importlib.import_module("010_bandwidth_probe")


def timed(fn, repeats=20):
    fn()
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return statistics.median(samples) * 1e3


def main():
    results = {}
    for total_mb in (4, 64):
        total = total_mb * 1024 * 1024 // 2
        design = probe_mod.build(8, 4096, total, 262144 if total_mb > 4 else 262144 // 4)
        w_i = iron.tensor(np.ones((total,), dtype=bfloat16), dtype=bfloat16)
        o_i = iron.zeros((128,), dtype=bfloat16, device="npu")
        prog = nd.Program.from_design(design, w_i, o_i)
        # Parent buffer with a marker region: the probe reports the first
        # element of the last object each core consumed.
        parent = nd.Buffer(total + 4096)
        parent.array[:] = bfloat16(1.0)
        parent.array[total:] = bfloat16(3.0)
        parent.to_device()
        out = nd.Buffer(128)
        results[f"direct_{total_mb}mb_ms"] = timed(lambda: prog(parent, out))
        full_value = float(out.from_device()[112])
        view = parent.view(4096, total)
        prog(view, out)
        view_value = float(out.from_device()[112])
        results[f"subbuffer_{total_mb}mb_values"] = [full_value, view_value]
        results[f"iron_{total_mb}mb_ms"] = timed(lambda: design(w_i, o_i), repeats=5)
    print(json.dumps(results, indent=1))
    nd.finish()


if __name__ == "__main__":
    main()
