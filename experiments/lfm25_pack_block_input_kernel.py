"""NPU-side packing of a new hidden vector and persistent recurrent state."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_pack_block_input.cc"


@iron.jit
def pack_block_input(hidden: In, state_and_weight: In, packed: Out):
    hidden_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    state_ty = np.ndarray[(6144,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(12288,), np.dtype[bfloat16]]
    hidden_fifo = ObjectFifo(hidden_ty, name="block_pack_hidden", depth=1)
    state_fifo = ObjectFifo(state_ty, name="block_pack_state", depth=1)
    output_fifo = ObjectFifo(packed_ty, name="block_pack_output", depth=1)
    kernel = ExternalFunction(
        "lfm25_pack_block_input",
        source_file=str(SOURCE),
        arg_types=[hidden_ty, state_ty, packed_ty],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(hidden_in, state_in, out, op):
        h = hidden_in.acquire(1)
        s = state_in.acquire(1)
        y = out.acquire(1)
        op(h, s, y)
        hidden_in.release(1)
        state_in.release(1)
        out.release(1)

    worker = Worker(
        core_fn, [hidden_fifo.cons(), state_fifo.cons(), output_fifo.prod(), kernel]
    )

    def sequence(h, s, y, h_prod, s_prod, y_cons):
        h_prod.fill(h)
        s_prod.fill(s)
        y_cons.drain(y, wait=True)

    runtime = Runtime(
        sequence,
        [
            hidden_ty, state_ty, packed_ty,
            hidden_fifo.prod(), state_fifo.prod(), output_fifo.cons(),
        ],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
