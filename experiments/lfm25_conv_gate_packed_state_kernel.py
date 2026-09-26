"""Recurrent convolution with NPU-resident state and depthwise weights."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_conv_gate_packed_state.cc"


@iron.jit
def conv_gate_packed_state(
    projection: In, state_and_weight: In, output: Out, next_state_and_weight: Out
):
    vec_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    triple_ty = np.ndarray[(3072,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(6144,), np.dtype[bfloat16]]
    projection_fifo = ObjectFifo(triple_ty, name="conv_projection", depth=1)
    state_fifo = ObjectFifo(packed_ty, name="conv_state_weight", depth=1)
    output_fifo = ObjectFifo(vec_ty, name="conv_gated", depth=1)
    next_fifo = ObjectFifo(packed_ty, name="conv_next_state_weight", depth=1)
    kernel = ExternalFunction(
        "lfm25_conv_gate_packed_state",
        source_file=str(SOURCE),
        arg_types=[triple_ty, packed_ty, vec_ty, packed_ty],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(p_fifo, s_fifo, y_fifo, n_fifo, op):
        p = p_fifo.acquire(1)
        s = s_fifo.acquire(1)
        y = y_fifo.acquire(1)
        n = n_fifo.acquire(1)
        op(p, s, y, n)
        p_fifo.release(1)
        s_fifo.release(1)
        y_fifo.release(1)
        n_fifo.release(1)

    worker = Worker(
        core_fn,
        [projection_fifo.cons(), state_fifo.cons(), output_fifo.prod(), next_fifo.prod(), kernel],
    )

    def sequence(p, s, y, n, p_prod, s_prod, y_cons, n_cons):
        p_prod.fill(p)
        s_prod.fill(s)
        y_cons.drain(y, wait=True)
        n_cons.drain(n, wait=True)

    runtime = Runtime(
        sequence,
        [
            triple_ty, packed_ty, vec_ty, packed_ty,
            projection_fifo.prod(), state_fifo.prod(), output_fifo.cons(), next_fifo.cons(),
        ],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
