"""NPU-side packing of hidden and attention context for the block tail."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_pack_attention_tail.cc"


@iron.jit
def pack_attention_tail(hidden: In, context: In, packed: Out):
    vector_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(2048,), np.dtype[bfloat16]]
    hidden_fifo = ObjectFifo(vector_ty, name="tail_pack_hidden", depth=1)
    context_fifo = ObjectFifo(vector_ty, name="tail_pack_context", depth=1)
    packed_fifo = ObjectFifo(packed_ty, name="tail_pack_output", depth=1)
    kernel = ExternalFunction(
        "lfm25_pack_attention_tail", source_file=str(SOURCE),
        arg_types=[vector_ty, vector_ty, packed_ty],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(hidden_in, context_in, packed_out, op):
        h = hidden_in.acquire(1)
        c = context_in.acquire(1)
        y = packed_out.acquire(1)
        op(h, c, y)
        hidden_in.release(1)
        context_in.release(1)
        packed_out.release(1)

    worker = Worker(
        core_fn,
        [hidden_fifo.cons(), context_fifo.cons(), packed_fifo.prod(), kernel],
    )

    def sequence(h, c, y, h_prod, c_prod, y_cons):
        h_prod.fill(h)
        c_prod.fill(c)
        y_cons.drain(y, wait=True)

    runtime = Runtime(
        sequence,
        [vector_ty, vector_ty, packed_ty,
         hidden_fifo.prod(), context_fifo.prod(), packed_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
