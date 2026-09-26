"""Select a token embedding with NPU DMA, without CPU embedding arithmetic."""

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from ml_dtypes import bfloat16


@iron.jit
def embedding_dma(table: In, embedding: Out, *, token_id: CompileTime[int]):
    if not 0 <= token_id < 65536:
        raise ValueError("Token ID outside vocabulary")
    table_ty = np.ndarray[(65536, 1024), np.dtype[bfloat16]]
    row_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    row_fifo = ObjectFifo(row_ty, name="embedding_selected_row", depth=1)
    output_fifo = ObjectFifo(row_ty, name="embedding_output", depth=1)

    def core_fn(row_in, row_out):
        selected = row_in.acquire(1)
        result = row_out.acquire(1)
        for col in range_(1024):
            result[col] = selected[col]
        row_in.release(1)
        row_out.release(1)

    worker = Worker(core_fn, [row_fifo.cons(), output_fifo.prod()])
    tap = TensorAccessPattern(
        (65536, 1024), token_id * 1024,
        [1, 1, 1, 1024], [0, 0, 0, 1],
    )

    def sequence(weights, out, row_prod, output_cons):
        row_prod.fill(weights, tap=tap)
        output_cons.drain(out, wait=True)

    runtime = Runtime(sequence, [table_ty, row_ty, row_fifo.prod(), output_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
