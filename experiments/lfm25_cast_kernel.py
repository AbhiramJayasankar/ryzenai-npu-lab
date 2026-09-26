"""NPU-side FP32 to BF16 conversion for decode projection outputs."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_f32_to_bf16.cc"


@iron.jit
def f32_to_bf16(input0: In, output: Out, *, N: CompileTime[int]):
    if N <= 0 or N > 8192 or N % 64:
        raise ValueError("N must be positive, at most 8192, and divisible by 64")
    in_ty = np.ndarray[(N,), np.dtype[np.float32]]
    out_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    in_fifo = ObjectFifo(in_ty, name="cast_input", depth=1)
    out_fifo = ObjectFifo(out_ty, name="cast_output", depth=1)
    kernel = ExternalFunction(
        "lfm25_f32_to_bf16",
        source_file=str(SOURCE),
        arg_types=[in_ty, out_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(x_fifo, y_fifo, op):
        x = x_fifo.acquire(1)
        y = y_fifo.acquire(1)
        op(x, y, N)
        x_fifo.release(1)
        y_fifo.release(1)

    worker = Worker(core_fn, [in_fifo.cons(), out_fifo.prod(), kernel])

    def sequence(x, y, x_prod, y_cons):
        x_prod.fill(x)
        y_cons.drain(y, wait=True)

    runtime = Runtime(sequence, [in_ty, out_ty, in_fifo.prod(), out_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
