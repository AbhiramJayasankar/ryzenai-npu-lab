"""Phoenix AIE2 one-vector RMSNorm with real LFM2.5 weights."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_rms_norm.cc"


@iron.jit
def rms_norm(input0: In, gamma0: In, output: Out, *, N: CompileTime[int]):
    if N != 1024:
        raise ValueError("Initial LFM2.5 kernel expects width 1024")
    vec_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    in_fifo = ObjectFifo(vec_ty, name="rms_input")
    gamma_fifo = ObjectFifo(vec_ty, name="rms_gamma")
    out_fifo = ObjectFifo(vec_ty, name="rms_output")
    kernel = ExternalFunction(
        "lfm25_rms_norm",
        source_file=str(SOURCE),
        arg_types=[vec_ty, vec_ty, vec_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(x_fifo, g_fifo, y_fifo, op):
        x = x_fifo.acquire(1)
        g = g_fifo.acquire(1)
        y = y_fifo.acquire(1)
        op(x, g, y, N)
        x_fifo.release(1)
        g_fifo.release(1)
        y_fifo.release(1)

    worker = Worker(
        core_fn, [in_fifo.cons(), gamma_fifo.cons(), out_fifo.prod(), kernel]
    )

    def sequence(x, g, y, x_prod, g_prod, y_cons):
        x_prod.fill(x)
        g_prod.fill(g)
        y_cons.drain(y, wait=True)

    runtime = Runtime(
        sequence,
        [vec_ty, vec_ty, vec_ty, in_fifo.prod(), gamma_fifo.prod(), out_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
