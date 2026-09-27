"""Select two prompt embeddings in one NPU table scan."""

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
def embedding_pair_dynamic(table: In, pair: Out, token_ids: In):
    table_ty = np.ndarray[(VOCAB, HIDDEN), np.dtype[bfloat16]]
    pair_ty = np.ndarray[(2 * HIDDEN,), np.dtype[bfloat16]]
    row_ty = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    ids_ty = np.ndarray[(2,), np.dtype[np.int32]]
    ids_fifo = ObjectFifo(ids_ty, name="pair_embedding_ids", depth=1)
    row_fifo = ObjectFifo(row_ty, name="pair_embedding_rows", depth=1)
    output_fifo = ObjectFifo(pair_ty, name="pair_embedding_result", depth=1)
    chosen = Buffer(pair_ty, name="pair_embedding_chosen")

    def core_fn(ids_in, rows_in, out, selected):
        ids = ids_in.acquire(1)
        for index in range_(VOCAB):
            row = rows_in.acquire(1)
            with if_(index == ids[0]):
                for col in range_(HIDDEN):
                    selected[col] = row[col]
            with if_(index == ids[1]):
                for col in range_(HIDDEN):
                    selected[HIDDEN + col] = row[col]
            rows_in.release(1)
        result = out.acquire(1)
        for col in range_(2 * HIDDEN):
            result[col] = selected[col]
        ids_in.release(1)
        out.release(1)

    worker = Worker(core_fn, [ids_fifo.cons(), row_fifo.cons(),
                              output_fifo.prod(), chosen])
    block_taps = [TensorAccessPattern(
        (VOCAB, HIDDEN), block * ROWS_PER_BLOCK * HIDDEN,
        [1, 1, ROWS_PER_BLOCK, HIDDEN], [0, 0, HIDDEN, 1],
    ) for block in range(VOCAB // ROWS_PER_BLOCK)]

    def sequence(weights, output, ids, row_prod, output_cons, ids_prod):
        first = TaskGroup()
        ids_prod.fill(ids, group=first, wait=True)
        first.finish()
        output_group = TaskGroup()
        output_cons.drain(output, group=output_group, wait=True)
        for tap in block_taps:
            group = TaskGroup()
            row_prod.fill(weights, tap=tap, group=group, wait=True)
            group.finish()
        output_group.finish()

    runtime = Runtime(sequence, [table_ty, pair_ty, ids_ty,
                                 row_fifo.prod(), output_fifo.cons(), ids_fifo.prod()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
