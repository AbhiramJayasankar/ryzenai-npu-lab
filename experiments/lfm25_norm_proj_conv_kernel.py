"""One Phoenix program for first-block norm, projection and recurrent gate."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron import Buffer, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
K = 1024
M = 3072
TILE = 64
GAMMA_PADDED = TILE * TILE
DATA_CHUNK = 6144
WEIGHT_LEN = GAMMA_PADDED + M * K


@iron.jit
def norm_proj_conv(
    packed_hidden_and_state: In,
    packed_gamma_and_projection_weight: In,
    gate_output: Out,
    next_state_and_weight: Out,
):
    data_ty = np.ndarray[(2 * DATA_CHUNK,), np.dtype[bfloat16]]
    data_chunk_ty = np.ndarray[(DATA_CHUNK,), np.dtype[bfloat16]]
    weight_ty = np.ndarray[(WEIGHT_LEN,), np.dtype[bfloat16]]
    weight_tile_ty = np.ndarray[(TILE * TILE,), np.dtype[bfloat16]]
    hidden_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    projection_ty = np.ndarray[(M,), np.dtype[bfloat16]]
    state_ty = np.ndarray[(DATA_CHUNK,), np.dtype[bfloat16]]
    accumulator_ty = np.ndarray[(TILE,), np.dtype[np.float32]]

    data_fifo = ObjectFifo(data_chunk_ty, name="npc_data", depth=1)
    weight_fifo = ObjectFifo(weight_tile_ty, name="npc_weight", depth=1)
    gate_fifo = ObjectFifo(hidden_ty, name="npc_gate", depth=1)
    next_state_fifo = ObjectFifo(state_ty, name="npc_next_state", depth=1)
    normalized = Buffer(hidden_ty, name="npc_normalized")
    projection = Buffer(projection_ty, name="npc_projection")
    accumulator = Buffer(accumulator_ty, name="npc_accumulator")

    norm = ExternalFunction(
        "lfm25_rms_norm",
        source_file=str(KERNEL_DIR / "lfm25_rms_norm.cc"),
        arg_types=[data_chunk_ty, weight_tile_ty, hidden_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )
    gemv = ExternalFunction(
        "lfm25_bf16_gemv_offset",
        source_file=str(KERNEL_DIR / "lfm25_bf16_gemv_offset.cc"),
        arg_types=[weight_tile_ty, hidden_ty, accumulator_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )
    store = ExternalFunction(
        "lfm25_cast_store_3072",
        source_file=str(KERNEL_DIR / "lfm25_cast_store_3072.cc"),
        arg_types=[accumulator_ty, projection_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )
    conv = ExternalFunction(
        "lfm25_conv_gate_packed_state",
        source_file=str(KERNEL_DIR / "lfm25_conv_gate_packed_state.cc"),
        arg_types=[projection_ty, data_chunk_ty, hidden_ty, state_ty],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(data_in, weights_in, gate_out, state_out, norm_buf, proj_buf,
                accum_buf, norm_op, gemv_op, store_op, conv_op):
        hidden = data_in.acquire(1)
        gamma = weights_in.acquire(1)
        norm_op(hidden, gamma, norm_buf, K)
        weights_in.release(1)
        data_in.release(1)
        for output_tile in range_(M // TILE):
            for i in range_(TILE):
                accum_buf[i] = 0.0
            for input_tile in range_(K // TILE):
                w = weights_in.acquire(1)
                gemv_op(w, norm_buf, accum_buf, input_tile * TILE)
                weights_in.release(1)
            store_op(accum_buf, proj_buf, output_tile * TILE)
        previous_state = data_in.acquire(1)
        gate = gate_out.acquire(1)
        next_state = state_out.acquire(1)
        conv_op(proj_buf, previous_state, gate, next_state)
        data_in.release(1)
        gate_out.release(1)
        state_out.release(1)

    worker = Worker(
        core_fn,
        [
            data_fifo.cons(), weight_fifo.cons(), gate_fifo.prod(), next_state_fifo.prod(),
            normalized, projection, accumulator, norm, gemv, store, conv,
        ],
    )

    hidden_tap = TensorAccessPattern(
        (2 * DATA_CHUNK,), 0, [1, 1, 1, DATA_CHUNK], [0, 0, 0, 1]
    )
    state_tap = TensorAccessPattern(
        (2 * DATA_CHUNK,), DATA_CHUNK, [1, 1, 1, DATA_CHUNK], [0, 0, 0, 1]
    )
    gamma_tap = TensorAccessPattern(
        (WEIGHT_LEN,), 0, [1, 1, 1, GAMMA_PADDED], [0, 0, 0, 1]
    )
    original_taps = TensorTiler2D.group_tiler(
        (M, K), (TILE, TILE), (1, K // TILE)
    )
    projection_taps = [
        TensorAccessPattern(
            (WEIGHT_LEN,), GAMMA_PADDED + tap.offset, tap.sizes, tap.strides
        )
        for tap in original_taps
    ]

    def sequence(data, weights, gate, next_state, data_prod, weight_prod,
                 gate_cons, state_cons):
        initial = TaskGroup()
        data_prod.fill(data, tap=hidden_tap, group=initial, wait=True)
        weight_prod.fill(weights, tap=gamma_tap, group=initial, wait=True)
        initial.finish()
        outputs = TaskGroup()
        data_prod.fill(data, tap=state_tap, group=outputs, wait=True)
        gate_cons.drain(gate, group=outputs, wait=True)
        state_cons.drain(next_state, group=outputs, wait=True)
        for group_index in range(M // TILE):
            group = TaskGroup()
            weight_prod.fill(
                weights, tap=projection_taps[group_index], group=group, wait=True
            )
            group.finish()
        outputs.finish()

    runtime = Runtime(
        sequence,
        [
            data_ty, weight_ty, hidden_ty, state_ty,
            data_fifo.prod(), weight_fifo.prod(), gate_fifo.cons(), next_state_fifo.cons(),
        ],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
