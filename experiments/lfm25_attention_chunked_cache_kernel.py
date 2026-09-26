"""Attention over a growing KV cache, streamed in 64-token blocks."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_attention_chunked_head.cc"
CAPACITY = 64
HEAD_ELEMENTS = 2 + 2 * CAPACITY * 64
BLOCK_ELEMENTS = 8 * HEAD_ELEMENTS


@iron.jit
def attention_context_chunked(qkv_and_hidden: In, past_cache: In,
                              packed_tail_input: Out, next_cache: Out,
                              *, block_count: CompileTime[int] = 1):
    if not 1 <= block_count <= 64:
        raise ValueError("The experimental 4K context needs 1-64 KV blocks")
    qkv_ty = np.ndarray[(3072,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(2048,), np.dtype[bfloat16]]
    cache_ty = np.ndarray[(block_count * BLOCK_ELEMENTS,), np.dtype[bfloat16]]
    head_ty = np.ndarray[(HEAD_ELEMENTS,), np.dtype[bfloat16]]
    context_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    accumulator_ty = np.ndarray[(132,), np.dtype[np.float32]]
    qkv_fifo = ObjectFifo(qkv_ty, name="chunked_qkv", depth=1)
    old_fifo = ObjectFifo(head_ty, name="chunked_old", depth=1)
    packed_fifo = ObjectFifo(packed_ty, name="chunked_tail", depth=1)
    next_fifo = ObjectFifo(head_ty, name="chunked_next", depth=1)
    context_buffer = Buffer(context_ty, name="chunked_context")
    accumulator_buffer = Buffer(accumulator_ty, name="chunked_accumulator")
    op = ExternalFunction(
        "lfm25_attention_chunked_head", source_file=str(SOURCE),
        arg_types=[qkv_ty, head_ty, head_ty, context_ty, accumulator_ty,
                   np.int32, np.int32, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(qkv_in, old_in, tail_out, next_out, context, accumulator,
                context_op):
        qkv = qkv_in.acquire(1)
        for head in range_(8):
            for block in range_(block_count):
                old = old_in.acquire(1)
                next_state = next_out.acquire(1)
                context_op(qkv, old, next_state, context, accumulator,
                           head, block, block_count)
                old_in.release(1)
                next_out.release(1)
        tail = tail_out.acquire(1)
        for i in range_(1024):
            tail[i] = qkv[2048 + i]
            tail[1024 + i] = context[i]
        qkv_in.release(1)
        tail_out.release(1)

    worker = Worker(core_fn, [qkv_fifo.cons(), old_fifo.cons(),
                              packed_fifo.prod(), next_fifo.prod(),
                              context_buffer, accumulator_buffer, op],
                    stack_size=4096)
    taps = [TensorAccessPattern(
        (block_count * BLOCK_ELEMENTS,), block * BLOCK_ELEMENTS + head * HEAD_ELEMENTS,
        [1, 1, 1, HEAD_ELEMENTS], [0, 0, 0, 1],
    ) for head in range(8) for block in range(block_count)]

    def sequence(qkv, old, packed, new, qkv_prod, old_prod, packed_cons, new_cons):
        first = TaskGroup()
        qkv_prod.fill(qkv, group=first, wait=True)
        first.finish()
        packed_group = TaskGroup()
        packed_cons.drain(packed, group=packed_group, wait=True)
        for tap in taps:
            group = TaskGroup()
            new_cons.drain(new, tap=tap, group=group, wait=True)
            old_prod.fill(old, tap=tap, group=group, wait=True)
            group.finish()
        packed_group.finish()

    runtime = Runtime(sequence, [qkv_ty, cache_ty, packed_ty, cache_ty,
                                 qkv_fifo.prod(), old_fifo.prod(),
                                 packed_fifo.cons(), next_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
