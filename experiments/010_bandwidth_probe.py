"""Measure how fast Phoenix NPU cores can pull weight bytes from DDR.

Each core consumes a private shard of one large BF16 buffer through its own
shim DMA channel. The core only touches one value per object, so the result
is a data-movement ceiling for weight-streaming GEMV, not a compute figure.
"""

import argparse
import json
import statistics
import time

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from ml_dtypes import bfloat16


def build(streams, object_elems, total_elems, fill_elems):
    per_core = total_elems // streams
    objects = per_core // object_elems
    fills = per_core // fill_elems
    assert per_core % fill_elems == 0 and fill_elems % object_elems == 0

    tag = f"s{streams}_o{object_elems}_t{total_elems}_f{fill_elems}"

    @iron.jit(tag=tag)
    def probe(weights: In, out: Out, *, tag: CompileTime[str]):
        w_ty = np.ndarray[(total_elems,), np.dtype[bfloat16]]
        obj_ty = np.ndarray[(object_elems,), np.dtype[bfloat16]]
        small_ty = np.ndarray[(16,), np.dtype[bfloat16]]
        out_ty = np.ndarray[(16 * streams,), np.dtype[bfloat16]]
        placements = []
        for i in range(streams):
            col = i % 4
            row = 2 + i // 4
            placements.append((col, row, i // 4))
        fifos = [ObjectFifo(obj_ty, name=f"bw_w{i}", depth=2) for i in range(streams)]
        cols = sorted({c for c, _r, _ch in placements})
        out_fifos, subs = {}, {}
        for c in cols:
            members = [i for i, pl in enumerate(placements) if pl[0] == c]
            ty = np.ndarray[(16 * len(members),), np.dtype[bfloat16]]
            out_fifos[c] = ObjectFifo(ty, name=f"bw_out{c}", depth=1)
            parts = out_fifos[c].prod(tile=Tile(c, 1)).join(
                [16 * k for k in range(len(members))], obj_types=[small_ty] * len(members),
                names=[f"bw_o{i}" for i in members])
            for k, i in enumerate(members):
                subs[i] = parts[k]

        def core_fn(w_in, o_out, acc):
            acc[0] = 0.0
            for _ in range_(objects):
                w = w_in.acquire(1)
                acc[0] = w[0]
                w_in.release(1)
            o = o_out.acquire(1)
            for k in range_(16):
                o[k] = acc[0]
            o_out.release(1)

        workers = [
            Worker(core_fn, [fifos[i].cons(), subs[i].prod(),
                             Buffer(np.ndarray[(1,), np.dtype[bfloat16]], name=f"bw_acc{i}")],
                   tile=Tile(c, r))
            for i, (c, r, _ch) in enumerate(placements)
        ]
        taps = [[TensorAccessPattern((total_elems,), i * per_core + f * fill_elems,
                                     [1, 1, fill_elems // 1024, 1024], [0, 0, 1024, 1])
                 for f in range(fills)] for i in range(streams)]

        def sequence(w, o, prods, out_cons):
            done = TaskGroup()
            start = 0
            for c, cons in zip(cols, out_cons):
                n = 16 * sum(1 for pl in placements if pl[0] == c)
                cons.drain(o, tap=TensorAccessPattern((16 * streams,), start, [1, 1, 1, n],
                                                      [0, 0, 0, 1]), group=done, wait=True)
                start += n
            for f in range(fills):
                g = TaskGroup()
                for i in range(streams):
                    prods[i].fill(w, tap=taps[i][f], group=g, wait=True)
                g.finish()
            done.finish()

        rt = Runtime(sequence, [w_ty, out_ty,
                                [fifos[i].prod(tile=Tile(placements[i][0], 0))
                                 for i in range(streams)],
                                [out_fifos[c].cons(tile=Tile(c, 0)) for c in cols]])
        return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

    return probe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--object-kb", type=int, default=8)
    p.add_argument("--total-mb", type=int, default=64)
    p.add_argument("--fill-kb", type=int, default=2048)
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    total = a.total_mb * 1024 * 1024 // 2
    obj = a.object_kb * 1024 // 2
    fill = a.fill_kb * 1024 // 2
    probe = build(a.streams, obj, total, fill)
    w = iron.tensor(np.ones((total,), dtype=bfloat16), dtype=bfloat16)
    o = iron.zeros((16 * a.streams,), dtype=bfloat16, device="npu")
    probe(w, o)
    times = []
    for _ in range(a.repeats):
        t = time.perf_counter()
        probe(w, o)
        times.append(time.perf_counter() - t)
    med = statistics.median(times)
    vals = o.numpy().astype(np.float32)
    print(json.dumps({
        "streams": a.streams, "object_kb": a.object_kb, "fill_kb": a.fill_kb,
        "total_mb": a.total_mb, "median_ms": round(med * 1e3, 3),
        "gb_per_s": round(a.total_mb * 1024 * 1024 / med / 1e9, 2),
        "check": float(vals[0]), "expected": 1.0,
    }))


if __name__ == "__main__":
    main()
