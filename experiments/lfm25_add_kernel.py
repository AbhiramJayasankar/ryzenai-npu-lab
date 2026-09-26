"""NPU BF16 residual addition for LFM2.5 hidden vectors."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_bf16_add.cc"


@iron.jit
def bf16_add(left: In, right: In, output: Out, *, N: CompileTime[int]):
    if N <= 0 or N > 4096 or N % 64:
        raise ValueError("N must be positive, at most 4096, and divisible by 64")
    vec_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    left_fifo = ObjectFifo(vec_ty, name="add_left", depth=1)
    right_fifo = ObjectFifo(vec_ty, name="add_right", depth=1)
    output_fifo = ObjectFifo(vec_ty, name="add_output", depth=1)
    kernel = ExternalFunction(
        "lfm25_bf16_add",
        source_file=str(SOURCE),
        arg_types=[vec_ty, vec_ty, vec_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(l_fifo, r_fifo, y_fifo, op):
        l = l_fifo.acquire(1)
        r = r_fifo.acquire(1)
        y = y_fifo.acquire(1)
        op(l, r, y, N)
        l_fifo.release(1)
        r_fifo.release(1)
        y_fifo.release(1)

    worker = Worker(
        core_fn, [left_fifo.cons(), right_fifo.cons(), output_fifo.prod(), kernel]
    )

    def sequence(l, r, y, l_prod, r_prod, y_cons):
        l_prod.fill(l)
        r_prod.fill(r)
        y_cons.drain(y, wait=True)

    runtime = Runtime(
        sequence,
        [vec_ty, vec_ty, vec_ty, left_fifo.prod(), right_fifo.prod(), output_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
