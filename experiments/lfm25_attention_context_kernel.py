"""NPU attention score, softmax, and value mixing for one decode token."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_attention_context_head.cc"
HEAD_CACHE = 2 * 21 * 64
ALL_CACHE = 8 * HEAD_CACHE


@iron.jit
def attention_context(qkv: In, past_cache: In, context: Out):
    qkv_ty = np.ndarray[(2048,), np.dtype[bfloat16]]
    cache_ty = np.ndarray[(ALL_CACHE,), np.dtype[bfloat16]]
    head_ty = np.ndarray[(HEAD_CACHE,), np.dtype[bfloat16]]
    context_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    qkv_fifo = ObjectFifo(qkv_ty, name="context_qkv", depth=1)
    cache_fifo = ObjectFifo(head_ty, name="context_cache", depth=1)
    context_fifo = ObjectFifo(context_ty, name="context_output", depth=1)
    kernel = ExternalFunction(
        "lfm25_attention_context_head", source_file=str(SOURCE),
        arg_types=[qkv_ty, head_ty, context_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(qkv_in, cache_in, context_out, op):
        qkv_data = qkv_in.acquire(1)
        result = context_out.acquire(1)
        for head in range_(8):
            cache = cache_in.acquire(1)
            op(qkv_data, cache, result, head)
            cache_in.release(1)
        qkv_in.release(1)
        context_out.release(1)

    worker = Worker(
        core_fn,
        [qkv_fifo.cons(), cache_fifo.cons(), context_fifo.prod(), kernel],
    )
    taps = [
        TensorAccessPattern((ALL_CACHE,), head * HEAD_CACHE,
                            [1, 1, 1, HEAD_CACHE], [0, 0, 0, 1])
        for head in range(8)
    ]

    def sequence(q, cache, output, q_prod, cache_prod, output_cons):
        first = TaskGroup()
        q_prod.fill(q, group=first, wait=True)
        first.finish()
        output_group = TaskGroup()
        output_cons.drain(output, group=output_group, wait=True)
        for tap in taps:
            group = TaskGroup()
            cache_prod.fill(cache, tap=tap, group=group, wait=True)
            group.finish()
        output_group.finish()

    runtime = Runtime(
        sequence,
        [qkv_ty, cache_ty, context_ty,
         qkv_fifo.prod(), cache_fifo.prod(), context_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
