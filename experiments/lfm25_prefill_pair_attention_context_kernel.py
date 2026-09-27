"""Causal attention for two positions with one KV cache read and write."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels/lfm25_attention_fixed64_pair_head.cc"
CAPACITY = 64
HEAD_ELEMENTS = 2 + 2 * CAPACITY * 64
CACHE_ELEMENTS = 8 * HEAD_ELEMENTS


@iron.jit
def attention_pair_context_fixed64(qkv_pair: In, past_cache: In,
                                   tail_pair: Out, next_cache: Out):
    qkv_ty = np.ndarray[(6144,), np.dtype[bfloat16]]
    tail_ty = np.ndarray[(4096,), np.dtype[bfloat16]]
    cache_ty = np.ndarray[(CACHE_ELEMENTS,), np.dtype[bfloat16]]
    head_ty = np.ndarray[(HEAD_ELEMENTS,), np.dtype[bfloat16]]
    context_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    qkv_fifo = ObjectFifo(qkv_ty, name="pair_context_qkv", depth=1)
    old_fifo = ObjectFifo(head_ty, name="pair_context_old", depth=1)
    tail_fifo = ObjectFifo(tail_ty, name="pair_context_tail", depth=1)
    next_fifo = ObjectFifo(head_ty, name="pair_context_next", depth=1)
    context0 = Buffer(context_ty, name="pair_context0")
    context1 = Buffer(context_ty, name="pair_context1")
    op = ExternalFunction(
        "lfm25_attention_fixed64_pair_head", source_file=str(SOURCE),
        arg_types=[qkv_ty, head_ty, head_ty, context_ty, np.int32, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(qkv_in, old_in, tail_out, next_out,
                ctx0, ctx1, context_op):
        pair = qkv_in.acquire(1)
        for head in range_(8):
            old = old_in.acquire(1)
            updated = next_out.acquire(1)
            context_op(pair, old, updated, ctx0, head, 0)
            context_op(pair, updated, updated, ctx1, head, 1)
            old_in.release(1)
            next_out.release(1)
        tail = tail_out.acquire(1)
        for i in range_(1024):
            tail[i] = pair[2048 + i]
            tail[1024 + i] = ctx0[i]
            tail[2048 + i] = pair[3072 + 2048 + i]
            tail[3072 + i] = ctx1[i]
        qkv_in.release(1)
        tail_out.release(1)

    worker = Worker(core_fn, [qkv_fifo.cons(), old_fifo.cons(),
                              tail_fifo.prod(), next_fifo.prod(),
                              context0, context1, op],
                    stack_size=4096)
    taps = [TensorAccessPattern(
        (CACHE_ELEMENTS,), head * HEAD_ELEMENTS,
        [1, 1, 1, HEAD_ELEMENTS], [0, 0, 0, 1],
    ) for head in range(8)]

    def sequence(qkv, old, tail, new, qkv_prod, old_prod, tail_cons, new_cons):
        first = TaskGroup()
        qkv_prod.fill(qkv, group=first, wait=True)
        first.finish()
        tail_group = TaskGroup()
        tail_cons.drain(tail, group=tail_group, wait=True)
        for tap in taps:
            group = TaskGroup()
            new_cons.drain(new, tap=tap, group=group, wait=True)
            old_prod.fill(old, tap=tap, group=group, wait=True)
            group.finish()
        tail_group.finish()

    runtime = Runtime(sequence, [qkv_ty, cache_ty, tail_ty, cache_ty,
                                 qkv_fifo.prod(), old_fifo.prod(),
                                 tail_fifo.cons(), next_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
