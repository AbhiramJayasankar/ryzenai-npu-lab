"""NPU SiLU and gated multiplication for LFM2.5 feed-forward layers."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_silu_gate.cc"


@iron.jit
def silu_gate(w1: In, w3: In, output: Out, *, N: CompileTime[int]):
    if N != 2560:
        raise ValueError("The first LFM2.5 FFN uses width 2560")
    vec_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    w1_fifo = ObjectFifo(vec_ty, name="silu_w1", depth=1)
    w3_fifo = ObjectFifo(vec_ty, name="silu_w3", depth=1)
    output_fifo = ObjectFifo(vec_ty, name="silu_output", depth=1)
    kernel = ExternalFunction(
        "lfm25_silu_gate",
        source_file=str(SOURCE),
        arg_types=[vec_ty, vec_ty, vec_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(a_fifo, b_fifo, y_fifo, op):
        a = a_fifo.acquire(1)
        b = b_fifo.acquire(1)
        y = y_fifo.acquire(1)
        op(a, b, y, N)
        a_fifo.release(1)
        b_fifo.release(1)
        y_fifo.release(1)

    worker = Worker(core_fn, [w1_fifo.cons(), w3_fifo.cons(), output_fifo.prod(), kernel])

    def sequence(a, b, y, a_prod, b_prod, y_cons):
        a_prod.fill(a)
        b_prod.fill(b)
        y_cons.drain(y, wait=True)

    runtime = Runtime(
        sequence,
        [vec_ty, vec_ty, vec_ty, w1_fifo.prod(), w3_fifo.prod(), output_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
