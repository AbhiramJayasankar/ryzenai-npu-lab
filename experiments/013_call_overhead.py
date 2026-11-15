"""Experiment 013 part D: where the fixed cost of one NPU call goes.

Modes (one per process; do not mix the IRON run path and direct pyxrt):

  direct  Time each pyxrt step of a call to AMD's 4 KB DMA-only memcpy design
          (basic/memcpy, 1 column, 1 channel, no core): one-time setup steps,
          then per call: Python binding overhead, kernel(...) submission,
          run.wait(), a reused pyxrt.run object, buffer syncs of 4 KB-64 MB.
  iron        Same design called the IRON way (@iron.jit with iron tensors),
              plus a cProfile listing (cache/013/iron_profile.txt).
  iron_timed  The IRON call again, with wall-clock timers wrapped around
              IRON's own steps (specialize, runtime.load, runtime.run).
  insts [KB]  The 010 bandwidth probe moving the same 8 MB in 1-64 fills per
              core, to see the cost per instruction / DMA task / wait. The
              xclbins differ per variant, so pass one fill size per process.

  powershell -NoProfile -File scripts\\run_npu_timeout.ps1 -Script experiments\\013_call_overhead.py -ScriptArgs direct -Seconds 300
"""

import cProfile
import importlib
import io
import json
import pstats
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cache/013/call_overhead.jsonl"
sys.path.insert(0, str(Path(__file__).resolve().parent))
KEEP = []


def stats(samples):
    s = sorted(samples)
    return {"median_us": 1e6 * statistics.median(s), "p10_us": 1e6 * s[len(s) // 10],
            "p90_us": 1e6 * s[(9 * len(s)) // 10], "min_us": 1e6 * s[0], "n": len(s)}


def emit(row):
    print(json.dumps(row), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")


def memcpy_design():
    bench = importlib.import_module("013_npu_bench")
    mod = bench.load("basic/memcpy/memcpy.py", "ex_memcpy")
    return mod


def direct():
    import pyxrt
    import npu_direct as nd
    mod = memcpy_design()
    d = mod.memcpy.compilable.specialize(size=1024, num_columns=1, num_channels=1, bypass=True)
    xclbin_path, insts_path = (Path(p) for p in d.compile())
    once = {}
    t = time.perf_counter(); dev = nd.device(); once["open_device_ms"] = 1e3 * (time.perf_counter() - t)
    t = time.perf_counter(); xb = pyxrt.xclbin(str(xclbin_path)); once["read_xclbin_ms"] = 1e3 * (time.perf_counter() - t)
    t = time.perf_counter(); dev.register_xclbin(xb); once["register_xclbin_ms"] = 1e3 * (time.perf_counter() - t)
    t = time.perf_counter(); ctx = pyxrt.hw_context(dev, xb.get_uuid()); once["create_hw_context_ms"] = 1e3 * (time.perf_counter() - t)
    t = time.perf_counter(); kern = pyxrt.kernel(ctx, "MLIR_AIE"); once["create_kernel_ms"] = 1e3 * (time.perf_counter() - t)
    insts = np.fromfile(insts_path, dtype=np.uint32)
    t = time.perf_counter()
    ibo = pyxrt.bo(dev, insts.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    ibo.write(insts.tobytes(), 0)
    ibo.sync(nd.TO_DEVICE, insts.nbytes, 0)
    once["insts_bo_alloc_write_sync_ms"] = 1e3 * (time.perf_counter() - t)
    t = time.perf_counter()
    a = nd.Buffer(1024, np.int32)
    b = nd.Buffer(1024, np.int32)
    once["alloc_two_4kb_bos_ms"] = 1e3 * (time.perf_counter() - t)
    KEEP.extend([dev, xb, ctx, kern, ibo, a, b])
    a.write(np.arange(1, 1025, dtype=np.int32))
    once["insts_bytes"] = int(insts.nbytes)
    emit({"mode": "direct", "part": "one_time", **once})

    def call():
        r = kern(3, ibo, insts.nbytes, a.bo, b.bo)
        st = r.wait()
        assert st == pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED, st
        return r

    for _ in range(20):
        call()
    b.array[:] = 0
    b.to_device()
    call()
    ok = bool(np.array_equal(b.from_device(), np.arange(1, 1025, dtype=np.int32)))
    N = 400
    total, submit, wait = [], [], []
    for _ in range(N):
        t0 = time.perf_counter()
        r = kern(3, ibo, insts.nbytes, a.bo, b.bo)
        t1 = time.perf_counter()
        r.wait()
        t2 = time.perf_counter()
        total.append(t2 - t0); submit.append(t1 - t0); wait.append(t2 - t1)
    emit({"mode": "direct", "part": "call_split", "correct": ok,
          "total": stats(total), "submit_returns": stats(submit), "wait": stats(wait)})

    # A pre-built run object: arguments set once, then start()/wait() per call.
    run = pyxrt.run(kern)
    run.set_arg(0, 3); run.set_arg(1, ibo); run.set_arg(2, insts.nbytes)
    run.set_arg(3, a.bo); run.set_arg(4, b.bo)
    for _ in range(20):
        run.start(); run.wait()
    reuse, rs, rw = [], [], []
    for _ in range(N):
        t0 = time.perf_counter(); run.start(); t1 = time.perf_counter(); run.wait(); t2 = time.perf_counter()
        reuse.append(t2 - t0); rs.append(t1 - t0); rw.append(t2 - t1)
    emit({"mode": "direct", "part": "reused_run", "total": stats(reuse),
          "start_returns": stats(rs), "wait": stats(rw)})

    # Python -> pyxrt binding cost for a trivial call (no driver work)
    py = []
    for _ in range(2000):
        t0 = time.perf_counter(); kern.group_id(1); py.append(time.perf_counter() - t0)
    empty = []
    for _ in range(2000):
        t0 = time.perf_counter(); empty.append(time.perf_counter() - t0)
    emit({"mode": "direct", "part": "python_binding", "trivial_pyxrt_call": stats(py),
          "empty_timer_pair": stats(empty)})

    # Buffer syncs. For host_only BOs on this integrated NPU the memory is shared;
    # sync is a cache flush / invalidate, so its cost should scale with size.
    for kb in (4, 64, 1024, 16384, 65536):
        n = kb * 256
        buf = nd.Buffer(n, np.int32)
        KEEP.append(buf)
        buf.array[:] = 1
        to_s, from_s = [], []
        for _ in range(30 if kb < 16384 else 10):
            t0 = time.perf_counter(); buf.bo.sync(nd.TO_DEVICE, n * 4, 0); to_s.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); buf.bo.sync(nd.FROM_DEVICE, n * 4, 0); from_s.append(time.perf_counter() - t0)
        emit({"mode": "direct", "part": "bo_sync", "kb": kb, "to_device": stats(to_s),
              "from_device": stats(from_s), "to_device_gbps": n * 4 / statistics.median(to_s) / 1e9})
    nd.finish()


def iron_path():
    import aie.iron as iron
    mod = memcpy_design()
    a = iron.arange(1, 1025, dtype=np.int32, device="npu")
    b = iron.zeros_like(a)
    kw = dict(size=1024, num_columns=1, num_channels=1, bypass=True)
    for _ in range(10):
        mod.memcpy(a, b, **kw)
    ok = bool(np.array_equal(b.numpy(), np.arange(1, 1025, dtype=np.int32)))
    samples = []
    for _ in range(200):
        t0 = time.perf_counter(); mod.memcpy(a, b, **kw); samples.append(time.perf_counter() - t0)
    emit({"mode": "iron", "part": "call", "correct": ok, "total": stats(samples)})
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(200):
        mod.memcpy(a, b, **kw)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(45)
    rows = []
    for (fname, line, func), (cc, nc, tt, ct, _callers) in ps.stats.items():
        rows.append({"func": f"{Path(fname).name}:{line}:{func}", "calls": nc,
                     "cum_us_per_call": 1e6 * ct / 200, "self_us_per_call": 1e6 * tt / 200})
    rows.sort(key=lambda r: -r["cum_us_per_call"])
    emit({"mode": "iron", "part": "profile_top", "rows": rows[:60]})
    (ROOT / "cache/013/iron_profile.txt").write_text(s.getvalue())
    import npu_direct as nd
    nd.finish()


def iron_timed():
    """IRON call path with wall-clock timers around IRON's own steps (cProfile's
    attribution was misleading because of its own overhead and C++ calls)."""
    import aie.iron as iron
    from aie.utils import callabledesign as cd
    from aie.utils.hostruntime.xrtruntime import hostruntime as hr
    from aie.utils.hostruntime.xrtruntime import tensor as xt
    from aie.utils import npukernel as nk
    mod = memcpy_design()
    acc = {}

    def wrap(owner, name, label):
        f = getattr(owner, name)

        def w(*a, **k):
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                acc[label] = acc.get(label, 0.0) + time.perf_counter() - t0
        setattr(owner, name, w)

    wrap(cd.CallableDesign, "_build_compilable", "specialize (new CompilableDesign per call)")
    wrap(nk.NPUKernel, "__call__", "NPUKernel.__call__ (load + run)")
    wrap(hr.CachedXRTRuntime, "load", "runtime.load (file checks, context/kernel cache lookup)")
    wrap(hr.XRTHostRuntime, "run", "runtime.run (arg syncs, BO checks, kernel start + wait)")
    wrap(hr.XRTHostRuntime, "check_device_consistency", "check_device_consistency")
    wrap(xt.XRTTensor, "_sync_to_device", "tensor sync to device")
    a = iron.arange(1, 1025, dtype=np.int32, device="npu")
    b = iron.zeros_like(a)
    kw = dict(size=1024, num_columns=1, num_channels=1, bypass=True)
    for _ in range(10):
        mod.memcpy(a, b, **kw)
    acc.clear()
    N = 300
    t0 = time.perf_counter()
    for _ in range(N):
        mod.memcpy(a, b, **kw)
    total = time.perf_counter() - t0
    ok = bool(np.array_equal(b.numpy(), np.arange(1, 1025, dtype=np.int32)))
    emit({"mode": "iron_timed", "correct": ok, "mean_total_us": 1e6 * total / N,
          "mean_us_per_call": {k: 1e6 * v / N for k, v in acc.items()}})
    import npu_direct as nd
    nd.finish()


def insts():
    import npu_direct as nd
    probe = importlib.import_module("010_bandwidth_probe")
    total = 8 * 1024 * 1024 // 2       # 8 MB of BF16
    fills_kb = (1024, 256, 64, 16)     # 1, 4, 16, 64 fills per core (8 cores)
    if len(sys.argv) > 2:              # xclbins differ per variant: one per process
        fills_kb = tuple(int(x) for x in sys.argv[2:])
    progs = []
    for f in fills_kb:
        design = probe.build(8, 4096, total, f * 1024 // 2)
        x, i = nd.compile_design(design)
        progs.append((f, Path(x), Path(i)))
    base = progs[0][1].read_bytes()
    prog = nd.Program(progs[0][1])
    w = nd.Buffer(total)
    w.array[:] = 1.0
    w.to_device()
    o = nd.Buffer(128)
    KEEP.extend([prog, w, o])
    for f, x, i in progs:
        if x.read_bytes() != base:
            emit({"mode": "insts", "fill_kb": f, "skipped": "xclbin differs from the 1-fill one"})
            continue
        e = prog.entry(i)
        KEEP.append(e)
        for _ in range(5):
            e(w, o)
        o.array[:] = 0
        o.to_device()
        e(w, o)
        ok = bool(np.all(o.from_device().astype(np.float32) == 1.0))
        smp = []
        for _ in range(60):
            t0 = time.perf_counter(); e(w, o); smp.append(time.perf_counter() - t0)
        st = stats(smp)
        emit({"mode": "insts", "fill_kb": f, "fills_per_core": 1024 // f, "insts_bytes": e.insts_bytes,
              "correct": ok, **st, "gbps": total * 2 / (st["median_us"] * 1e-6) / 1e9})
    nd.finish()


if __name__ == "__main__":
    {"direct": direct, "iron": iron_path, "iron_timed": iron_timed, "insts": insts}[sys.argv[1]]()
