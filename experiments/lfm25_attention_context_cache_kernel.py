"""Fused attention context, KV append, and attention-tail input packing."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
@iron.jit
def attention_context_cache(
    qkv_and_hidden: In,
    past_cache: In,
    packed_tail_input: Out,
    next_cache: Out,
    *,
    past_length: CompileTime[int] = 21,
):
    if not 1 <= past_length < 96:
        raise ValueError("Current attention kernel supports 1-95 prior tokens")
    past_head = 2 * past_length * 64
    next_head = 2 * (past_length + 1) * 64
    qkv_hidden_ty = np.ndarray[(3072,), np.dtype[bfloat16]]
    past_ty = np.ndarray[(8 * past_head,), np.dtype[bfloat16]]
    old_head_ty = np.ndarray[(past_head,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(2048,), np.dtype[bfloat16]]
    context_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    next_ty = np.ndarray[(8 * next_head,), np.dtype[bfloat16]]
    next_head_ty = np.ndarray[(next_head,), np.dtype[bfloat16]]
    qkv_fifo = ObjectFifo(qkv_hidden_ty, name="fused_context_qkv", depth=1)
    old_fifo = ObjectFifo(old_head_ty, name="fused_context_old", depth=1)
    packed_fifo = ObjectFifo(packed_ty, name="fused_context_tail", depth=1)
    next_fifo = ObjectFifo(next_head_ty, name="fused_context_next", depth=1)
    context_buffer = Buffer(context_ty, name="fused_context_buffer")
    context_op = ExternalFunction(
        "lfm25_attention_context_head_dynamic",
        source_file=str(KERNEL_DIR / "lfm25_attention_context_head.cc"),
        arg_types=[qkv_hidden_ty, old_head_ty, context_ty, np.int32, np.int32],
        include_dirs=[config.cxx_header_path()],
    )
    append_op = ExternalFunction(
        "lfm25_attention_append_cache_dynamic",
        source_file=str(KERNEL_DIR / "lfm25_attention_append_cache.cc"),
        arg_types=[qkv_hidden_ty, old_head_ty, next_head_ty, np.int32, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(qkv_in, old_in, tail_out, next_out, context,
                compute_context, append_cache):
        qkv = qkv_in.acquire(1)
        for head in range_(8):
            old = old_in.acquire(1)
            next_state = next_out.acquire(1)
            compute_context(qkv, old, context, head, past_length)
            append_cache(qkv, old, next_state, head, past_length)
            old_in.release(1)
            next_out.release(1)
        tail = tail_out.acquire(1)
        for i in range_(1024):
            tail[i] = qkv[2048 + i]
            tail[1024 + i] = context[i]
        qkv_in.release(1)
        tail_out.release(1)

    worker = Worker(
        core_fn,
        [qkv_fifo.cons(), old_fifo.cons(), packed_fifo.prod(), next_fifo.prod(),
         context_buffer, context_op, append_op],
        stack_size=4096,
    )
    old_taps = [
        TensorAccessPattern((8 * past_head,), head * past_head,
                            [1, 1, 1, past_head], [0, 0, 0, 1])
        for head in range(8)
    ]
    next_taps = [
        TensorAccessPattern((8 * next_head,), head * next_head,
                            [1, 1, 1, next_head], [0, 0, 0, 1])
        for head in range(8)
    ]

    def sequence(qkv, old, packed, new, qkv_prod, old_prod, packed_cons, new_cons):
        first = TaskGroup()
        qkv_prod.fill(qkv, group=first, wait=True)
        first.finish()
        packed_group = TaskGroup()
        packed_cons.drain(packed, group=packed_group, wait=True)
        for old_tap, next_tap in zip(old_taps, next_taps):
            group = TaskGroup()
            new_cons.drain(new, tap=next_tap, group=group, wait=True)
            old_prod.fill(old, tap=old_tap, group=group, wait=True)
            group.finish()
        packed_group.finish()

    runtime = Runtime(
        sequence,
        [qkv_hidden_ty, past_ty, packed_ty, next_ty,
         qkv_fifo.prod(), old_fifo.prod(), packed_fifo.cons(), next_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
