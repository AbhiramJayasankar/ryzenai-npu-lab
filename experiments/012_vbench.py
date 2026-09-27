"""Time and check the engine's vector math on one NPU core (kernels/pk/pk_vbench.cc).

Per routine: max error vs NumPy on 1024 inputs in [-12, 12], and cycles per
16-lane vector from the difference between 1 and 201 repeats (1 GHz clock
assumed for the cycle estimate).
"""

import time

import aie.iron as iron
import numpy as np
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.utils import config

import npu_direct as nd
import pk_engine as pk

IN_T = np.ndarray[(1040,), np.dtype[np.float32]]
OUT_T = np.ndarray[(1024,), np.dtype[np.float32]]


@iron.jit(tag="vbench_" + pk.KVER)
def vbench(inp: In, out: Out, *, tag: iron.CompileTime[str]):
    k = ExternalFunction("pk_vbench", source_file=str(pk.KDIR / "pk_vbench.cc"),
                         arg_types=[IN_T, OUT_T], compile_flags=["-O2", "-DPK_F=1"],
                         include_dirs=[config.cxx_header_path(), str(pk.KDIR)])
    fi, fo = ObjectFifo(IN_T, name="vb_in"), ObjectFifo(OUT_T, name="vb_out")

    def core(i, o, kern):
        a, b = i.acquire(1), o.acquire(1)
        kern(a, b)
        i.release(1)
        o.release(1)

    w = Worker(core, [fi.cons(), fo.prod(), k], stack_size=2048)

    def seq(a, b, p, c):
        p.fill(a)
        c.drain(b, wait=True)

    rt = Runtime(seq, [IN_T, OUT_T, fi.prod(), fo.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def main():
    prog = nd.Program.from_design(vbench)
    x = np.linspace(-12, 12, 1024).astype(np.float32)
    i, o = nd.Buffer(1040, np.float32), nd.Buffer(1024, np.float32)
    sig = 1 / (1 + np.exp(-x.astype(np.float64)))
    ref = {0: np.exp(np.minimum(x, 0).astype(np.float64)), 1: sig, 2: x * sig,
           3: x.astype(np.float64) ** 2, 4: x}
    names = {0: "exp16", 1: "sigmoid", 2: "swish", 3: "vmul", 4: "bf16 round trip"}
    for op in range(5):
        times = {}
        for reps in (1, 201):
            buf = np.zeros(1040, np.float32)
            buf[:1024], buf[1024], buf[1025] = x, op, reps
            i.write(buf)
            best = 1e9
            for _ in range(3):
                t = time.perf_counter()
                prog(i, o)
                best = min(best, time.perf_counter() - t)
            times[reps] = best
        got = np.asarray(o.from_device(), np.float64)
        err = np.abs(got - ref[op]) / np.maximum(np.abs(ref[op]), 1e-3)
        cyc = (times[201] - times[1]) / 200 / 64 * 1e9
        print(f"{names[op]:16s} max rel err {err.max():.2e}  ~{cyc:6.0f} cycles per vector", flush=True)
    nd.finish()


if __name__ == "__main__":
    main()
