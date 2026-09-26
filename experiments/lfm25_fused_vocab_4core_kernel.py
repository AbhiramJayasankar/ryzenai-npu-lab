"""Four adjacent NPU cores scan vocabulary shards and pass the winner along."""

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
CORES = 4
ROWS_PER_CORE = VOCAB // CORES
ROWS_PER_BLOCK = 1024
PACKED_LEN = HIDDEN + VOCAB * HIDDEN


@iron.jit
def fused_vocab_4core(hidden: In, packed_gamma_and_embedding: In,
                      next_embedding: Out, token_id: Out):
    hidden_ty = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(PACKED_LEN,), np.dtype[bfloat16]]
    row_ty = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    candidate_ty = np.ndarray[(HIDDEN + 4,), np.dtype[bfloat16]]
    index_ty = np.ndarray[(1,), np.dtype[np.int32]]
    score_ty = np.ndarray[(1,), np.dtype[np.float32]]
    hidden_fifos = [ObjectFifo(hidden_ty, name=f"four_vocab_hidden_{i}", depth=1) for i in range(CORES)]
    weight_fifos = [ObjectFifo(row_ty, name=f"four_vocab_weight_{i}", depth=1) for i in range(CORES)]
    candidate_fifos = [ObjectFifo(candidate_ty, name=f"four_vocab_candidate_{i}", depth=1) for i in range(CORES - 1)]
    embedding_fifo = ObjectFifo(row_ty, name="four_vocab_embedding", depth=1)
    index_fifo = ObjectFifo(index_ty, name="four_vocab_index", depth=1)
    include = [config.cxx_header_path()]
    norm = ExternalFunction(
        "lfm25_rms_norm", source_file=str(KERNEL_DIR / "lfm25_rms_norm.cc"),
        arg_types=[hidden_ty, row_ty, hidden_ty, np.int32], include_dirs=include,
    )
    score = ExternalFunction(
        "lfm25_vocab_score_keep_row",
        source_file=str(KERNEL_DIR / "lfm25_vocab_score_keep_row.cc"),
        arg_types=[row_ty, hidden_ty, row_ty, score_ty, index_ty, np.int32],
        include_dirs=include,
    )
    pack = ExternalFunction(
        "lfm25_vocab_pack_candidate",
        source_file=str(KERNEL_DIR / "lfm25_vocab_pack_candidate.cc"),
        arg_types=[row_ty, score_ty, index_ty, candidate_ty], include_dirs=include,
    )
    accept = ExternalFunction(
        "lfm25_vocab_accept_candidate",
        source_file=str(KERNEL_DIR / "lfm25_vocab_accept_candidate.cc"),
        arg_types=[candidate_ty, row_ty, score_ty, index_ty], include_dirs=include,
    )

    def scan(h_in, w_in, activation, winner, best_score, best_index,
             norm_op, score_op, start):
        h = h_in.acquire(1)
        gamma = w_in.acquire(1)
        norm_op(h, gamma, activation, HIDDEN)
        h_in.release(1)
        w_in.release(1)
        best_score[0] = -3.4028235e38
        best_index[0] = 0
        for local_row in range_(ROWS_PER_CORE):
            row = w_in.acquire(1)
            score_op(row, activation, winner, best_score, best_index, start + local_row)
            w_in.release(1)

    def first_fn(h_in, w_in, candidate_out,
                 activation, winner, best_score, best_index,
                 norm_op, score_op, pack_op):
        scan(h_in, w_in, activation, winner, best_score, best_index,
             norm_op, score_op, 0)
        out = candidate_out.acquire(1)
        pack_op(winner, best_score, best_index, out)
        candidate_out.release(1)

    def middle_fn(h_in, w_in, candidate_in, candidate_out,
                  activation, winner, best_score, best_index,
                  norm_op, score_op, pack_op, accept_op, start):
        scan(h_in, w_in, activation, winner, best_score, best_index,
             norm_op, score_op, start)
        previous = candidate_in.acquire(1)
        accept_op(previous, winner, best_score, best_index)
        candidate_in.release(1)
        out = candidate_out.acquire(1)
        pack_op(winner, best_score, best_index, out)
        candidate_out.release(1)

    def last_fn(h_in, w_in, candidate_in, embedding_out, index_out,
                activation, winner, best_score, best_index,
                norm_op, score_op, accept_op):
        scan(h_in, w_in, activation, winner, best_score, best_index,
             norm_op, score_op, 3 * ROWS_PER_CORE)
        previous = candidate_in.acquire(1)
        accept_op(previous, winner, best_score, best_index)
        candidate_in.release(1)
        next_row = embedding_out.acquire(1)
        next_id = index_out.acquire(1)
        for col in range_(HIDDEN):
            next_row[col] = winner[col]
        next_id[0] = best_index[0]
        embedding_out.release(1)
        index_out.release(1)

    def make_worker(_row, col):
        state = [
            Buffer(hidden_ty, name=f"four_vocab_activation_{col}"),
            Buffer(row_ty, name=f"four_vocab_best_row_{col}"),
            Buffer(score_ty, name=f"four_vocab_best_score_{col}"),
            Buffer(index_ty, name=f"four_vocab_best_index_{col}"),
            norm, score,
        ]
        if col == 0:
            return Worker(first_fn, [hidden_fifos[col].cons(), weight_fifos[col].cons(),
                                     candidate_fifos[col].prod(), *state, pack])
        if col == CORES - 1:
            return Worker(last_fn, [hidden_fifos[col].cons(), weight_fifos[col].cons(),
                                    candidate_fifos[col - 1].cons(),
                                    embedding_fifo.prod(), index_fifo.prod(), *state, accept])
        return Worker(middle_fn, [hidden_fifos[col].cons(), weight_fifos[col].cons(),
                                  candidate_fifos[col - 1].cons(), candidate_fifos[col].prod(),
                                  *state, pack, accept, col * ROWS_PER_CORE])

    workers = Worker.grid(1, CORES, make_worker)
    gamma_tap = TensorAccessPattern((PACKED_LEN,), 0, [1, 1, 1, HIDDEN], [0, 0, 0, 1])
    block_taps = [
        [TensorAccessPattern(
            (PACKED_LEN,), HIDDEN + (core * ROWS_PER_CORE + block * ROWS_PER_BLOCK) * HIDDEN,
            [1, 1, ROWS_PER_BLOCK, HIDDEN], [0, 0, HIDDEN, 1],
        ) for block in range(ROWS_PER_CORE // ROWS_PER_BLOCK)]
        for core in range(CORES)
    ]

    def sequence(h, packed, embedding, token, h_prods, w_prods, embedding_cons, token_cons):
        first = TaskGroup()
        for core in range(CORES):
            h_prods[core].fill(h, group=first, wait=True)
            w_prods[core].fill(packed, tap=gamma_tap, group=first, wait=True)
        first.finish()
        outputs = TaskGroup()
        embedding_cons.drain(embedding, group=outputs, wait=True)
        token_cons.drain(token, group=outputs, wait=True)
        for block in range(ROWS_PER_CORE // ROWS_PER_BLOCK):
            group = TaskGroup()
            for core in range(CORES):
                w_prods[core].fill(packed, tap=block_taps[core][block], group=group, wait=True)
            group.finish()
        outputs.finish()

    runtime = Runtime(sequence, [
        hidden_ty, packed_ty, row_ty, index_ty,
        [fifo.prod() for fifo in hidden_fifos],
        [fifo.prod() for fifo in weight_fifos],
        embedding_fifo.cons(), index_fifo.cons(),
    ])
    return Program(iron.get_current_device(), runtime,
                   workers=[worker for row in workers for worker in row]).resolve_program()
