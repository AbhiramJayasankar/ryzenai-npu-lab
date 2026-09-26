"""Combine NPU-resident projection and convolution weights on the NPU."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_pack_conv.cc"


@iron.jit
def pack_conv(projection: In, weight: In, packed: Out):
    triple_ty = np.ndarray[(3072,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(6144,), np.dtype[bfloat16]]
    projection_fifo = ObjectFifo(triple_ty, name="pack_projection", depth=1)
    weight_fifo = ObjectFifo(triple_ty, name="pack_weight", depth=1)
    packed_fifo = ObjectFifo(packed_ty, name="pack_output", depth=1)
    kernel = ExternalFunction(
        "lfm25_pack_conv",
        source_file=str(SOURCE),
        arg_types=[triple_ty, triple_ty, packed_ty],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(p_fifo, w_fifo, y_fifo, op):
        p = p_fifo.acquire(1)
        w = w_fifo.acquire(1)
        y = y_fifo.acquire(1)
        op(p, w, y)
        p_fifo.release(1)
        w_fifo.release(1)
        y_fifo.release(1)

    worker = Worker(
        core_fn,
        [projection_fifo.cons(), weight_fifo.cons(), packed_fifo.prod(), kernel],
    )

    def sequence(p, w, y, p_prod, w_prod, y_cons):
        p_prod.fill(p)
        w_prod.fill(w)
        y_cons.drain(y, wait=True)

    runtime = Runtime(
        sequence,
        [
            triple_ty,
            triple_ty,
            packed_ty,
            projection_fifo.prod(),
            weight_fifo.prod(),
            packed_fifo.cons(),
        ],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
