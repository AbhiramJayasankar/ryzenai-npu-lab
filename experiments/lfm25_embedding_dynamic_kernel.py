"""Select a prompt token's embedding on NPU from a runtime token tensor.

Phoenix's static instruction format cannot encode a runtime DMA offset, so
this proof streams the table past one compute tile. It is a correctness
fallback; a faster dynamic selection path is future work.
"""

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.dialects.scf import if_
from aie.iron import Buffer, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from ml_dtypes import bfloat16


VOCAB = 65536
HIDDEN = 1024
ROWS_PER_BLOCK = 1024


@iron.jit
def embedding_dma_dynamic(table: In, embedding: Out, token_id: In):
    table_ty = np.ndarray[(VOCAB, HIDDEN), np.dtype[bfloat16]]
    row_ty = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    token_ty = np.ndarray[(1,), np.dtype[np.int32]]
    token_fifo = ObjectFifo(token_ty, name="dynamic_embedding_token", depth=1)
    row_fifo = ObjectFifo(row_ty, name="dynamic_embedding_row", depth=1)
    output_fifo = ObjectFifo(row_ty, name="dynamic_embedding_output", depth=1)
    chosen = Buffer(row_ty, name="dynamic_embedding_chosen")

    def core_fn(token_in, row_in, row_out, selected):
        token = token_in.acquire(1)
        for index in range_(VOCAB):
            row = row_in.acquire(1)
            with if_(index == token[0]):
                for col in range_(HIDDEN):
                    selected[col] = row[col]
            row_in.release(1)
        result = row_out.acquire(1)
        for col in range_(HIDDEN):
            result[col] = selected[col]
        token_in.release(1)
        row_out.release(1)

    worker = Worker(core_fn, [token_fifo.cons(), row_fifo.cons(),
                              output_fifo.prod(), chosen])
    block_taps = [TensorAccessPattern(
        (VOCAB, HIDDEN), block * ROWS_PER_BLOCK * HIDDEN,
        [1, 1, ROWS_PER_BLOCK, HIDDEN], [0, 0, HIDDEN, 1],
    ) for block in range(VOCAB // ROWS_PER_BLOCK)]

    def sequence(weights, out, selected_id, row_prod, output_cons, token_prod):
        first = TaskGroup()
        token_prod.fill(selected_id, group=first, wait=True)
        first.finish()
        output_group = TaskGroup()
        output_cons.drain(out, group=output_group, wait=True)
        for tap in block_taps:
            group = TaskGroup()
            row_prod.fill(weights, tap=tap, group=group, wait=True)
            group.finish()
        output_group.finish()

    runtime = Runtime(sequence, [table_ty, row_ty, token_ty,
                                 row_fifo.prod(), output_fifo.cons(),
                                 token_fifo.prod()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
