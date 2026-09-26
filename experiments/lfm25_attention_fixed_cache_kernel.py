"""One compiled attention context program for cache lengths up to 32."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_attention_fixed_head.cc"
CAPACITY = 32
HEAD_ELEMENTS = 2 + 2 * CAPACITY * 64
CACHE_ELEMENTS = 8 * HEAD_ELEMENTS


def _cache_taps():
    return [
        TensorAccessPattern(
            (CACHE_ELEMENTS,), head * HEAD_ELEMENTS,
            [1, 1, 1, HEAD_ELEMENTS], [0, 0, 0, 1],
        ) for head in range(8)
    ]


@iron.jit
def attention_first_fixed(qkv_and_hidden: In, packed_tail_input: Out,
                          first_cache: Out):
    qkv_ty = np.ndarray[(3072,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(2048,), np.dtype[bfloat16]]
    cache_ty = np.ndarray[(CACHE_ELEMENTS,), np.dtype[bfloat16]]
    head_ty = np.ndarray[(HEAD_ELEMENTS,), np.dtype[bfloat16]]
    context_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    qkv_fifo = ObjectFifo(qkv_ty, name="first_fixed_qkv", depth=1)
    packed_fifo = ObjectFifo(packed_ty, name="first_fixed_tail", depth=1)
    cache_fifo = ObjectFifo(head_ty, name="first_fixed_cache", depth=1)
    context_buffer = Buffer(context_ty, name="first_fixed_context")
    op = ExternalFunction(
        "lfm25_attention_fixed_first_head", source_file=str(SOURCE),
        arg_types=[qkv_ty, head_ty, context_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(qkv_in, tail_out, cache_out, context, first_op):
        qkv = qkv_in.acquire(1)
        for head in range_(8):
            next_head = cache_out.acquire(1)
            first_op(qkv, next_head, context, head)
            cache_out.release(1)
        tail = tail_out.acquire(1)
        for i in range_(1024):
            tail[i] = qkv[2048 + i]
            tail[1024 + i] = context[i]
        qkv_in.release(1)
        tail_out.release(1)

    worker = Worker(core_fn, [qkv_fifo.cons(), packed_fifo.prod(),
                              cache_fifo.prod(), context_buffer, op])

    def sequence(qkv, packed, cache, qkv_prod, packed_cons, cache_cons):
        first = TaskGroup()
        qkv_prod.fill(qkv, group=first, wait=True)
        first.finish()
        output_group = TaskGroup()
        packed_cons.drain(packed, group=output_group, wait=True)
        for tap in _cache_taps():
            group = TaskGroup()
            cache_cons.drain(cache, tap=tap, group=group, wait=True)
            group.finish()
        output_group.finish()

    runtime = Runtime(sequence, [qkv_ty, packed_ty, cache_ty,
                                 qkv_fifo.prod(), packed_fifo.cons(), cache_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


@iron.jit
def attention_context_fixed(qkv_and_hidden: In, past_cache: In,
                            packed_tail_input: Out, next_cache: Out):
    qkv_ty = np.ndarray[(3072,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(2048,), np.dtype[bfloat16]]
    cache_ty = np.ndarray[(CACHE_ELEMENTS,), np.dtype[bfloat16]]
    head_ty = np.ndarray[(HEAD_ELEMENTS,), np.dtype[bfloat16]]
    context_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    qkv_fifo = ObjectFifo(qkv_ty, name="context_fixed_qkv", depth=1)
    old_fifo = ObjectFifo(head_ty, name="context_fixed_old", depth=1)
    packed_fifo = ObjectFifo(packed_ty, name="context_fixed_tail", depth=1)
    next_fifo = ObjectFifo(head_ty, name="context_fixed_next", depth=1)
    context_buffer = Buffer(context_ty, name="context_fixed_context")
    op = ExternalFunction(
        "lfm25_attention_fixed_head", source_file=str(SOURCE),
        arg_types=[qkv_ty, head_ty, head_ty, context_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(qkv_in, old_in, tail_out, next_out, context, context_op):
        qkv = qkv_in.acquire(1)
        for head in range_(8):
            old = old_in.acquire(1)
            next_head = next_out.acquire(1)
            context_op(qkv, old, next_head, context, head)
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
                              context_buffer, op], stack_size=4096)
    taps = _cache_taps()

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
