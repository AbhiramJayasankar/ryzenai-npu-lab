"""NPU-side append of a decode token's key/value to the prior KV cache."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_attention_append_cache.cc"
PAST_HEAD = 2 * 21 * 64
NEXT_HEAD = 2 * 22 * 64


@iron.jit
def append_attention_cache(qkv: In, prior_cache: In, next_cache: Out):
    qkv_ty = np.ndarray[(2048,), np.dtype[bfloat16]]
    prior_ty = np.ndarray[(8 * PAST_HEAD,), np.dtype[bfloat16]]
    next_ty = np.ndarray[(8 * NEXT_HEAD,), np.dtype[bfloat16]]
    old_head_ty = np.ndarray[(PAST_HEAD,), np.dtype[bfloat16]]
    new_head_ty = np.ndarray[(NEXT_HEAD,), np.dtype[bfloat16]]
    qkv_fifo = ObjectFifo(qkv_ty, name="append_qkv", depth=1)
    prior_fifo = ObjectFifo(old_head_ty, name="append_prior", depth=1)
    next_fifo = ObjectFifo(new_head_ty, name="append_next", depth=1)
    kernel = ExternalFunction(
        "lfm25_attention_append_cache", source_file=str(SOURCE),
        arg_types=[qkv_ty, old_head_ty, new_head_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(qkv_in, prior_in, next_out, op):
        qkv_data = qkv_in.acquire(1)
        for head in range_(8):
            prior = prior_in.acquire(1)
            next_state = next_out.acquire(1)
            op(qkv_data, prior, next_state, head)
            prior_in.release(1)
            next_out.release(1)
        qkv_in.release(1)

    worker = Worker(core_fn, [qkv_fifo.cons(), prior_fifo.cons(), next_fifo.prod(), kernel])
    input_taps = [
        TensorAccessPattern((8 * PAST_HEAD,), head * PAST_HEAD,
                            [1, 1, 1, PAST_HEAD], [0, 0, 0, 1])
        for head in range(8)
    ]
    output_taps = [
        TensorAccessPattern((8 * NEXT_HEAD,), head * NEXT_HEAD,
                            [1, 1, 1, NEXT_HEAD], [0, 0, 0, 1])
        for head in range(8)
    ]

    def sequence(q, prior, next_state, q_prod, prior_prod, next_cons):
        first = TaskGroup()
        q_prod.fill(q, group=first, wait=True)
        first.finish()
        for input_tap, output_tap in zip(input_taps, output_taps):
            group = TaskGroup()
            next_cons.drain(next_state, tap=output_tap, group=group, wait=True)
            prior_prod.fill(prior, tap=input_tap, group=group, wait=True)
            group.finish()

    runtime = Runtime(
        sequence,
        [qkv_ty, prior_ty, next_ty,
         qkv_fifo.prod(), prior_fifo.prod(), next_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
