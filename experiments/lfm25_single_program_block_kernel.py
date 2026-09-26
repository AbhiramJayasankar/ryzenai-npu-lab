"""First LFM2.5 recurrent block as one streamed one-core Phoenix program."""

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
HIDDEN = 1024
PROJECTION = 3072
FFN = 2560
TILE = 64
DATA_CHUNK = 6144
WEIGHT_TILE = TILE * TILE
MATRICES = (
    ("input", PROJECTION, HIDDEN),
    ("output", HIDDEN, HIDDEN),
    ("w1", FFN, HIDDEN),
    ("w3", FFN, HIDDEN),
    ("w2", HIDDEN, FFN),
)
WEIGHT_LEN = 2 * WEIGHT_TILE + sum(rows * cols for _name, rows, cols in MATRICES)


@iron.jit
def recurrent_block(
    packed_hidden_and_state: In,
    packed_weights: In,
    next_state_and_weight: Out,
    block_output: Out,
):
    data_ty = np.ndarray[(2 * DATA_CHUNK,), np.dtype[bfloat16]]
    data_chunk_ty = np.ndarray[(DATA_CHUNK,), np.dtype[bfloat16]]
    weight_ty = np.ndarray[(WEIGHT_LEN,), np.dtype[bfloat16]]
    weight_tile_ty = np.ndarray[(WEIGHT_TILE,), np.dtype[bfloat16]]
    hidden_ty = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    activation_ty = np.ndarray[(FFN,), np.dtype[bfloat16]]
    projection_ty = np.ndarray[(PROJECTION,), np.dtype[bfloat16]]
    state_ty = np.ndarray[(DATA_CHUNK,), np.dtype[bfloat16]]
    accum_ty = np.ndarray[(TILE,), np.dtype[np.float32]]
    bf16_tile_ty = np.ndarray[(TILE,), np.dtype[bfloat16]]

    data_fifo = ObjectFifo(data_chunk_ty, name="block_data", depth=1)
    weight_fifo = ObjectFifo(weight_tile_ty, name="block_weight", depth=1)
    state_fifo = ObjectFifo(state_ty, name="block_next_state", depth=1)
    output_fifo = ObjectFifo(hidden_ty, name="block_final", depth=1)

    activation = Buffer(activation_ty, name="block_activation")
    projection = Buffer(projection_ty, name="block_projection")
    residual = Buffer(hidden_ty, name="block_residual")
    accum = Buffer(accum_ty, name="block_accum")
    w1_tile = Buffer(bf16_tile_ty, name="block_w1_tile")
    w3_tile = Buffer(bf16_tile_ty, name="block_w3_tile")
    gate_tile = Buffer(bf16_tile_ty, name="block_gate_tile")

    include = [config.cxx_header_path()]
    norm_op = ExternalFunction(
        "lfm25_rms_norm",
        source_file=str(KERNEL_DIR / "lfm25_rms_norm.cc"),
        arg_types=[hidden_ty, weight_tile_ty, activation_ty, np.int32],
        include_dirs=include,
    )
    gemv_op = ExternalFunction(
        "lfm25_bf16_gemv_offset",
        source_file=str(KERNEL_DIR / "lfm25_bf16_gemv_offset.cc"),
        arg_types=[weight_tile_ty, activation_ty, accum_ty, np.int32],
        include_dirs=include,
    )
    gemv_w2_op = ExternalFunction(
        "lfm25_bf16_gemv_offset_3072",
        source_file=str(KERNEL_DIR / "lfm25_bf16_gemv_offset_3072.cc"),
        arg_types=[weight_tile_ty, projection_ty, accum_ty, np.int32],
        include_dirs=include,
    )
    store_proj_op = ExternalFunction(
        "lfm25_cast_store_3072",
        source_file=str(KERNEL_DIR / "lfm25_cast_store_3072.cc"),
        arg_types=[accum_ty, projection_ty, np.int32],
        include_dirs=include,
    )
    store_final_op = ExternalFunction(
        "lfm25_cast_store_1024",
        source_file=str(KERNEL_DIR / "lfm25_cast_store_1024.cc"),
        arg_types=[accum_ty, hidden_ty, np.int32],
        include_dirs=include,
    )
    cast_tile_op = ExternalFunction(
        "lfm25_f32_to_bf16",
        source_file=str(KERNEL_DIR / "lfm25_f32_to_bf16.cc"),
        arg_types=[accum_ty, bf16_tile_ty, np.int32],
        include_dirs=include,
    )
    conv_op = ExternalFunction(
        "lfm25_conv_gate_packed_state",
        source_file=str(KERNEL_DIR / "lfm25_conv_gate_packed_state.cc"),
        arg_types=[projection_ty, data_chunk_ty, activation_ty, state_ty],
        include_dirs=include,
    )
    add_projection_op = ExternalFunction(
        "lfm25_add_projection_inplace",
        source_file=str(KERNEL_DIR / "lfm25_add_projection_inplace.cc"),
        arg_types=[hidden_ty, projection_ty],
        include_dirs=include,
    )
    add_output_op = ExternalFunction(
        "lfm25_add_output_inplace",
        source_file=str(KERNEL_DIR / "lfm25_add_output_inplace.cc"),
        arg_types=[hidden_ty, hidden_ty],
        include_dirs=include,
    )
    silu_gate_op = ExternalFunction(
        "lfm25_silu_gate",
        source_file=str(KERNEL_DIR / "lfm25_silu_gate.cc"),
        arg_types=[bf16_tile_ty, bf16_tile_ty, bf16_tile_ty, np.int32],
        include_dirs=include,
    )

    def core_fn(data_in, weights_in, state_out, final_out, act, proj, res, acc,
                w1_part, w3_part, gate_part, norm, gemv, gemv_w2, store_proj,
                store_final, cast_tile, conv, add_proj, add_output, silu_gate):
        hidden = data_in.acquire(1)
        for i in range_(HIDDEN):
            res[i] = hidden[i]
        gamma = weights_in.acquire(1)
        norm(res, gamma, act, HIDDEN)
        weights_in.release(1)
        data_in.release(1)

        # Convolution input projection: 1024 -> 3072.
        for output_tile in range_(PROJECTION // TILE):
            for i in range_(TILE):
                acc[i] = 0.0
            for input_tile in range_(HIDDEN // TILE):
                w = weights_in.acquire(1)
                gemv(w, act, acc, input_tile * TILE)
                weights_in.release(1)
            store_proj(acc, proj, output_tile * TILE)

        old_state = data_in.acquire(1)
        new_state = state_out.acquire(1)
        conv(proj, old_state, act, new_state)
        data_in.release(1)
        state_out.release(1)

        # Convolution output projection: 1024 -> 1024, then residual.
        for output_tile in range_(HIDDEN // TILE):
            for i in range_(TILE):
                acc[i] = 0.0
            for input_tile in range_(HIDDEN // TILE):
                w = weights_in.acquire(1)
                gemv(w, act, acc, input_tile * TILE)
                weights_in.release(1)
            store_proj(acc, proj, output_tile * TILE)
        add_proj(res, proj)

        gamma = weights_in.acquire(1)
        norm(res, gamma, act, HIDDEN)
        weights_in.release(1)

        # Interleave W1 and W3 per output tile; retain only gated activations.
        for output_tile in range_(FFN // TILE):
            for i in range_(TILE):
                acc[i] = 0.0
            for input_tile in range_(HIDDEN // TILE):
                w = weights_in.acquire(1)
                gemv(w, act, acc, input_tile * TILE)
                weights_in.release(1)
            cast_tile(acc, w1_part, TILE)
            for i in range_(TILE):
                acc[i] = 0.0
            for input_tile in range_(HIDDEN // TILE):
                w = weights_in.acquire(1)
                gemv(w, act, acc, input_tile * TILE)
                weights_in.release(1)
            cast_tile(acc, w3_part, TILE)
            silu_gate(w1_part, w3_part, gate_part, TILE)
            for i in range_(TILE):
                proj[output_tile * TILE + i] = gate_part[i]

        final = final_out.acquire(1)
        for output_tile in range_(HIDDEN // TILE):
            for i in range_(TILE):
                acc[i] = 0.0
            for input_tile in range_(FFN // TILE):
                w = weights_in.acquire(1)
                gemv_w2(w, proj, acc, input_tile * TILE)
                weights_in.release(1)
            store_final(acc, final, output_tile * TILE)
        add_output(final, res)
        final_out.release(1)

    worker = Worker(
        core_fn,
        [
            data_fifo.cons(), weight_fifo.cons(), state_fifo.prod(), output_fifo.prod(),
            activation, projection, residual, accum, w1_tile, w3_tile, gate_tile,
            norm_op, gemv_op, gemv_w2_op, store_proj_op, store_final_op,
            cast_tile_op, conv_op, add_projection_op, add_output_op, silu_gate_op,
        ],
    )

    hidden_tap = TensorAccessPattern(
        (2 * DATA_CHUNK,), 0, [1, 1, 1, DATA_CHUNK], [0, 0, 0, 1]
    )
    state_tap = TensorAccessPattern(
        (2 * DATA_CHUNK,), DATA_CHUNK, [1, 1, 1, DATA_CHUNK], [0, 0, 0, 1]
    )

    offsets = {}
    offset = 0
    offsets["operator_gamma"] = offset
    offset += WEIGHT_TILE
    for name in ("input", "output"):
        rows, cols = next((r, c) for n, r, c in MATRICES if n == name)
        offsets[name] = offset
        offset += rows * cols
    offsets["ffn_gamma"] = offset
    offset += WEIGHT_TILE
    for name in ("w1", "w3", "w2"):
        rows, cols = next((r, c) for n, r, c in MATRICES if n == name)
        offsets[name] = offset
        offset += rows * cols
    assert offset == WEIGHT_LEN

    gamma_taps = {
        name: TensorAccessPattern(
            (WEIGHT_LEN,), offsets[name], [1, 1, 1, WEIGHT_TILE], [0, 0, 0, 1]
        )
        for name in ("operator_gamma", "ffn_gamma")
    }
    matrix_taps = {}
    for name, rows, cols in MATRICES:
        original = TensorTiler2D.group_tiler(
            (rows, cols), (TILE, TILE), (1, cols // TILE)
        )
        matrix_taps[name] = [
            TensorAccessPattern(
                (WEIGHT_LEN,), offsets[name] + tap.offset, tap.sizes, tap.strides
            )
            for tap in original
        ]

    def fill_weight_group(weight, weight_prod, tap):
        group = TaskGroup()
        weight_prod.fill(weight, tap=tap, group=group, wait=True)
        group.finish()

    def sequence(data, weight, next_state, final, data_prod, weight_prod,
                 state_cons, final_cons):
        initial = TaskGroup()
        data_prod.fill(data, tap=hidden_tap, group=initial, wait=True)
        weight_prod.fill(weight, tap=gamma_taps["operator_gamma"], group=initial, wait=True)
        initial.finish()
        outputs = TaskGroup()
        data_prod.fill(data, tap=state_tap, group=outputs, wait=True)
        state_cons.drain(next_state, group=outputs, wait=True)
        final_cons.drain(final, group=outputs, wait=True)
        for tap in matrix_taps["input"]:
            fill_weight_group(weight, weight_prod, tap)
        for tap in matrix_taps["output"]:
            fill_weight_group(weight, weight_prod, tap)
        fill_weight_group(weight, weight_prod, gamma_taps["ffn_gamma"])
        for w1_tap, w3_tap in zip(matrix_taps["w1"], matrix_taps["w3"]):
            fill_weight_group(weight, weight_prod, w1_tap)
            fill_weight_group(weight, weight_prod, w3_tap)
        for tap in matrix_taps["w2"]:
            fill_weight_group(weight, weight_prod, tap)
        outputs.finish()

    runtime = Runtime(
        sequence,
        [
            data_ty, weight_ty, state_ty, hidden_ty,
            data_fifo.prod(), weight_fifo.prod(), state_fifo.cons(), output_fifo.cons(),
        ],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
