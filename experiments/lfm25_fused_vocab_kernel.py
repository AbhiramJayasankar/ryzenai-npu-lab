"""One-core NPU final norm, vocab score, argmax, next embedding in one pass."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
VOCAB = 65536
HIDDEN = 1024
ROWS_PER_BLOCK = 1024
PACKED_LEN = HIDDEN + VOCAB * HIDDEN


@iron.jit
def fused_vocab(hidden: In, packed_gamma_and_embedding: In,
                next_embedding: Out, token_id: Out):
    hidden_ty = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(PACKED_LEN,), np.dtype[bfloat16]]
    row_ty = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    index_ty = np.ndarray[(1,), np.dtype[np.int32]]
    score_ty = np.ndarray[(1,), np.dtype[np.float32]]
    hidden_fifo = ObjectFifo(hidden_ty, name="fused_vocab_hidden", depth=1)
    weight_fifo = ObjectFifo(row_ty, name="fused_vocab_weight", depth=1)
    embedding_fifo = ObjectFifo(row_ty, name="fused_vocab_embedding", depth=1)
    index_fifo = ObjectFifo(index_ty, name="fused_vocab_index", depth=1)
    activation = Buffer(hidden_ty, name="fused_vocab_normalized")
    best_row = Buffer(row_ty, name="fused_vocab_best_row")
    best_score = Buffer(score_ty, name="fused_vocab_best_score")
    best_index = Buffer(index_ty, name="fused_vocab_best_index")
    include = [config.cxx_header_path()]
    norm = ExternalFunction(
        "lfm25_rms_norm", source_file=str(KERNEL_DIR / "lfm25_rms_norm.cc"),
        arg_types=[hidden_ty, row_ty, hidden_ty, np.int32],
        include_dirs=include,
    )
    score = ExternalFunction(
        "lfm25_vocab_score_keep_row",
        source_file=str(KERNEL_DIR / "lfm25_vocab_score_keep_row.cc"),
        arg_types=[row_ty, hidden_ty, row_ty, score_ty, index_ty, np.int32],
        include_dirs=include,
    )

    def core_fn(hidden_in, weight_in, embedding_out, index_out,
                act, winner, best, index, norm_op, score_op):
        h = hidden_in.acquire(1)
        gamma = weight_in.acquire(1)
        norm_op(h, gamma, act, HIDDEN)
        hidden_in.release(1)
        weight_in.release(1)
        best[0] = -3.4028235e38
        index[0] = 0
        for row_index in range_(VOCAB):
            row = weight_in.acquire(1)
            score_op(row, act, winner, best, index, row_index)
            weight_in.release(1)
        next_row = embedding_out.acquire(1)
        next_id = index_out.acquire(1)
        for col in range_(HIDDEN):
            next_row[col] = winner[col]
        next_id[0] = index[0]
        embedding_out.release(1)
        index_out.release(1)

    worker = Worker(
        core_fn,
        [hidden_fifo.cons(), weight_fifo.cons(), embedding_fifo.prod(),
         index_fifo.prod(), activation, best_row, best_score, best_index,
         norm, score],
    )
    gamma_tap = TensorAccessPattern(
        (PACKED_LEN,), 0, [1, 1, 1, HIDDEN], [0, 0, 0, 1]
    )
    blocks = [
        TensorAccessPattern(
            (PACKED_LEN,), HIDDEN + block * ROWS_PER_BLOCK * HIDDEN,
            [1, 1, ROWS_PER_BLOCK, HIDDEN], [0, 0, HIDDEN, 1],
        )
        for block in range(VOCAB // ROWS_PER_BLOCK)
    ]

    def sequence(h, packed, embedding, token, h_prod, w_prod,
                 embedding_cons, token_cons):
        first = TaskGroup()
        h_prod.fill(h, group=first, wait=True)
        w_prod.fill(packed, tap=gamma_tap, group=first, wait=True)
        first.finish()
        outputs = TaskGroup()
        embedding_cons.drain(embedding, group=outputs, wait=True)
        token_cons.drain(token, group=outputs, wait=True)
        for tap in blocks:
            group = TaskGroup()
            w_prod.fill(packed, tap=tap, group=group, wait=True)
            group.finish()
        outputs.finish()

    runtime = Runtime(
        sequence,
        [hidden_ty, packed_ty, row_ty, index_ty,
         hidden_fifo.prod(), weight_fifo.prod(),
         embedding_fifo.cons(), index_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
