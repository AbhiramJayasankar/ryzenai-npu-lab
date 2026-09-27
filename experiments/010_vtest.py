"""Run kernels/x8/x8_vtest.cc on one NPU core and print its FP32 results."""

import numpy as np
import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.utils import config

import lfm25_x8 as x8
import npu_direct as nd

IN_T = np.ndarray[(16,), np.dtype[np.float32]]
OUT_T = np.ndarray[(128,), np.dtype[np.float32]]


@iron.jit(tag=x8.KVER)
def vtest(inp: In, out: Out, *, tag: iron.CompileTime[str]):
    k = ExternalFunction("x8_vtest", source_file=str(x8.KDIR / "x8_vtest.cc"),
                         arg_types=[IN_T, OUT_T],
                         include_dirs=[config.cxx_header_path(), str(x8.KDIR)])
    fi, fo = ObjectFifo(IN_T, name="vt_in"), ObjectFifo(OUT_T, name="vt_out")

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
    prog = nd.Program.from_design(vtest)
    x = np.array([-3, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 3, 8, -8, 0.25, -0.25, 2, -2, 20],
                 dtype=np.float32)
    i, o = nd.Buffer(16, np.float32), nd.Buffer(128, np.float32)
    i.write(x)
    prog(i, o)
    r = np.asarray(o.from_device()).reshape(8, 16)
    p = np.maximum(x, 0)
    a, b = np.exp(x - p), np.exp(-p)
    expect = [p, a, b, a + b, (a + b) * 8 / 17, 1 / (a + b), x * x, 2 * x]
    names = ["max(x,0)", "exp(x-p)", "exp(-p)", "a+b", "d*8/17", "recip", "x*x", "x*2"]
    np.set_printoptions(precision=5, suppress=True, linewidth=150)
    for n, got, e in zip(names, r, expect):
        print(f"{n:10s} maxerr {np.abs(got - e).max():.3g}  got {got[:6]}")
    nd.finish()


if __name__ == "__main__":
    main()
