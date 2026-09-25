"""Phoenix AIE2 one-token recurrent convolution/gating kernel."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_conv_gate.cc"


@iron.jit
def conv_gate(projection_and_weight: In, state: In, output: Out, next_state: Out):
    vec_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    triple_ty = np.ndarray[(3072,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(6144,), np.dtype[bfloat16]]
    projection_fifo = ObjectFifo(packed_ty, name="projection_and_weight")
    state_fifo = ObjectFifo(triple_ty, name="previous_state")
    output_fifo = ObjectFifo(vec_ty, name="gated_output")
    next_state_fifo = ObjectFifo(triple_ty, name="next_state")
    kernel = ExternalFunction(
        "lfm25_conv_gate",
        source_file=str(SOURCE),
        arg_types=[packed_ty, triple_ty, vec_ty, triple_ty],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(proj_in, state_in, out, state_out, op):
        proj = proj_in.acquire(1)
        prev = state_in.acquire(1)
        y = out.acquire(1)
        following = state_out.acquire(1)
        op(proj, prev, y, following)
        proj_in.release(1)
        state_in.release(1)
        out.release(1)
        state_out.release(1)

    worker = Worker(
        core_fn,
        [
            projection_fifo.cons(),
            state_fifo.cons(),
            output_fifo.prod(),
            next_state_fifo.prod(),
            kernel,
        ],
    )

    def sequence(proj, prev, y, following, p, s, o, ns):
        p.fill(proj)
        s.fill(prev)
        o.drain(y, wait=True)
        ns.drain(following, wait=True)

    runtime = Runtime(
        sequence,
        [
            packed_ty,
            triple_ty,
            vec_ty,
            triple_ty,
            projection_fifo.prod(),
            state_fifo.prod(),
            output_fifo.cons(),
            next_state_fifo.cons(),
        ],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
