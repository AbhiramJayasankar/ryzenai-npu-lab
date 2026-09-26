"""NPU-side argmax of a 65,536-entry BF16 vocabulary vector."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


SOURCE = Path(__file__).resolve().parent / "kernels" / "lfm25_argmax_chunk.cc"


@iron.jit
def vocab_argmax(logits: In, token_id: Out):
    logits_ty = np.ndarray[(65536,), np.dtype[bfloat16]]
    chunk_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    result_ty = np.ndarray[(1,), np.dtype[np.int32]]
    best_ty = np.ndarray[(1,), np.dtype[np.float32]]
    logits_fifo = ObjectFifo(chunk_ty, name="vocab_logits", depth=1)
    result_fifo = ObjectFifo(result_ty, name="vocab_token", depth=1)
    best = Buffer(best_ty, name="vocab_best_value")
    index = Buffer(result_ty, name="vocab_best_index")
    kernel = ExternalFunction(
        "lfm25_argmax_chunk", source_file=str(SOURCE),
        arg_types=[chunk_ty, best_ty, result_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(input_fifo, output_fifo, best_value, best_index, op):
        best_value[0] = -3.4028235e38
        best_index[0] = 0
        for chunk in range_(64):
            values = input_fifo.acquire(1)
            op(values, best_value, best_index, chunk * 1024)
            input_fifo.release(1)
        result = output_fifo.acquire(1)
        result[0] = best_index[0]
        output_fifo.release(1)

    worker = Worker(
        core_fn,
        [logits_fifo.cons(), result_fifo.prod(), best, index, kernel],
    )
    taps = [
        TensorAccessPattern((65536,), chunk * 1024,
                            [1, 1, 1, 1024], [0, 0, 0, 1])
        for chunk in range(64)
    ]

    def sequence(values, result, values_prod, result_cons):
        final = TaskGroup()
        result_cons.drain(result, group=final, wait=True)
        for tap in taps:
            group = TaskGroup()
            values_prod.fill(values, tap=tap, group=group, wait=True)
            group.finish()
        final.finish()

    runtime = Runtime(
        sequence,
        [logits_ty, result_ty, logits_fifo.prod(), result_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
