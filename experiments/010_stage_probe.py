"""Cost of the layer program's stage structure without model arithmetic.

Same shape as an eight-core recurrent layer: per stage, each core receives a
gathered activation from DDR and its weight slice, touches every object, and
writes one output object that the next stage gathers. Compares one stage
holding all weights against the real four-stage split.
"""

import argparse
import json
import statistics
import time

import aie.iron as iron
import numpy as np
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from ml_dtypes import bfloat16

import lfm25_x8 as x8
import npu_direct as nd

REAL = [(2, 770), (2, 256), (4, 1282), (5, 640)]  # (activation objs, weight objs)


def design(stages):
    tag = "stageprobe_" + "_".join(f"{a}x{w}" for a, w in stages)
    per_core = sum(w for _a, w in stages) * x8.OBJ
    w_len = x8.CORES * per_core

    @iron.jit(tag=tag)
    def probe(weights: In, io: Out, *, tag: CompileTime[str]):
        ins, outs = x8._fifos("sp")

        def core_fn(inp, out, buf):
            for n_act, n_w in stages:
                for _ in range_(n_act):
                    o = inp.acquire(1)
                    buf[0] = o[0]
                    inp.release(1)
                for _ in range_(n_w):
                    o = inp.acquire(1)
                    buf[0] = o[0]
                    inp.release(1)
                o = out.acquire(1)
                o[0] = buf[0]
                out.release(1)

        workers = [Worker(core_fn, [ins[k].cons(), outs[k].prod(),
                                    Buffer(np.ndarray[(16,), np.dtype[bfloat16]], name=f"sp_b{k}")],
                          tile=Tile(*x8._place(k))) for k in range(x8.CORES)]

        def sequence(w, io, prods, conss):
            off = 0
            for s, (n_act, n_w) in enumerate(stages):
                g = TaskGroup()
                for k in range(x8.CORES):
                    prods[k].fill(io, tap=x8._gather(x8.IO_LEN, 1 + s % 4, 64 * n_act), group=g)
                    prods[k].fill(w, tap=x8._linear(w_len, k * per_core + off, n_w * x8.OBJ), group=g)
                    conss[k].drain(io, tap=x8._linear(x8.IO_LEN, x8.SLOT(1 + (s + 1) % 4, k), x8.OBJ),
                                   group=g, wait=True)
                g.finish()
                off += n_w * x8.OBJ

        rt = Runtime(sequence, [np.ndarray[(w_len,), np.dtype[bfloat16]],
                                np.ndarray[(x8.IO_LEN,), np.dtype[bfloat16]],
                                [ins[k].prod(tile=Tile(x8._place(k)[0], 0)) for k in range(x8.CORES)],
                                [outs[k].cons(tile=Tile(x8._place(k)[0], 0)) for k in range(x8.CORES)]])
        return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

    return design_len(probe, w_len)


def design_len(probe, w_len):
    return probe, w_len


def main():
    results = {}
    total_w = sum(w for _a, w in REAL)
    variants = {
        "real_4_stages": REAL,
        "one_stage": [(2, total_w)],
        "sixteen_stages": [(2, total_w // 16)] * 16,
    }
    keep = []  # pyxrt objects crash if destroyed mid-process; keep them alive
    for name, stages in variants.items():
        probe, w_len = design(stages)
        prog = nd.Program.from_design(probe)
        keep.append(prog)
        w = nd.Buffer(w_len)
        io = nd.Buffer(x8.IO_LEN)
        keep += [w, io]
        prog(w, io)
        samples = []
        for _ in range(20):
            t = time.perf_counter()
            prog(w, io)
            samples.append(time.perf_counter() - t)
        results[name] = round(statistics.median(samples) * 1e3, 3)
        print(name, results[name], flush=True)
    results["bytes_mb"] = round(8 * total_w * 1024 / 1e6, 2)
    print(json.dumps(results, indent=1), flush=True)
    nd.finish()


if __name__ == "__main__":
    main()
