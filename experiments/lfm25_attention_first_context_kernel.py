"""NPU attention context and initial KV cache for a prompt's first token."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_attention_first_context.cc"


@iron.jit
def attention_first_context(qkv_and_hidden: In, packed_tail_input: Out,
                            first_cache: Out):
    qkv_ty = np.ndarray[(3072,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(2048,), np.dtype[bfloat16]]
    cache_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    qkv_fifo = ObjectFifo(qkv_ty, name="first_attention_qkv", depth=1)
    packed_fifo = ObjectFifo(packed_ty, name="first_attention_tail", depth=1)
    cache_fifo = ObjectFifo(cache_ty, name="first_attention_cache", depth=1)
    op = ExternalFunction(
        "lfm25_attention_first_context", source_file=str(SOURCE),
        arg_types=[qkv_ty, packed_ty, cache_ty],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(qkv_in, packed_out, cache_out, kernel):
        qkv = qkv_in.acquire(1)
        packed = packed_out.acquire(1)
        cache = cache_out.acquire(1)
        kernel(qkv, packed, cache)
        qkv_in.release(1)
        packed_out.release(1)
        cache_out.release(1)

    worker = Worker(core_fn, [qkv_fifo.cons(), packed_fifo.prod(), cache_fifo.prod(), op])

    def sequence(qkv, packed, cache, qkv_prod, packed_cons, cache_cons):
        qkv_prod.fill(qkv)
        packed_cons.drain(packed, wait=True)
        cache_cons.drain(cache, wait=True)

    runtime = Runtime(sequence, [qkv_ty, packed_ty, cache_ty,
                                 qkv_fifo.prod(), packed_fifo.cons(), cache_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
