"""One-core NPU program for attention output projection, residual and FFN."""

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
FFN = 2560
TILE = 64
WEIGHT_TILE = TILE * TILE
MATRICES = (
    ("output", HIDDEN, HIDDEN),
    ("w1", FFN, HIDDEN),
    ("w3", FFN, HIDDEN),
    ("w2", HIDDEN, FFN),
)
WEIGHT_LEN = WEIGHT_TILE + sum(rows * cols for _, rows, cols in MATRICES)


@iron.jit
def attention_tail(packed_hidden_and_context: In, packed_weights: In, block_output: Out):
    packed_ty = np.ndarray[(2 * HIDDEN,), np.dtype[bfloat16]]
    weight_ty = np.ndarray[(WEIGHT_LEN,), np.dtype[bfloat16]]
    weight_tile_ty = np.ndarray[(WEIGHT_TILE,), np.dtype[bfloat16]]
    hidden_ty = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    activation_ty = np.ndarray[(FFN,), np.dtype[bfloat16]]
    projection_ty = np.ndarray[(3072,), np.dtype[bfloat16]]
    accum_ty = np.ndarray[(TILE,), np.dtype[np.float32]]
    bf16_tile_ty = np.ndarray[(TILE,), np.dtype[bfloat16]]

    data_fifo = ObjectFifo(packed_ty, name="tail_data", depth=1)
    weight_fifo = ObjectFifo(weight_tile_ty, name="tail_weight", depth=1)
    output_fifo = ObjectFifo(hidden_ty, name="tail_output", depth=1)
    activation = Buffer(activation_ty, name="tail_activation")
    projection = Buffer(projection_ty, name="tail_projection")
    residual = Buffer(hidden_ty, name="tail_residual")
    accum = Buffer(accum_ty, name="tail_accum")
    w1_tile = Buffer(bf16_tile_ty, name="tail_w1")
    w3_tile = Buffer(bf16_tile_ty, name="tail_w3")
    gate_tile = Buffer(bf16_tile_ty, name="tail_gate")
    include = [config.cxx_header_path()]
    norm = ExternalFunction(
        "lfm25_rms_norm", source_file=str(KERNEL_DIR / "lfm25_rms_norm.cc"),
        arg_types=[hidden_ty, weight_tile_ty, activation_ty, np.int32],
        include_dirs=include,
    )
    gemv = ExternalFunction(
        "lfm25_bf16_gemv_offset",
        source_file=str(KERNEL_DIR / "lfm25_bf16_gemv_offset.cc"),
        arg_types=[weight_tile_ty, activation_ty, accum_ty, np.int32],
        include_dirs=include,
    )
    gemv_w2 = ExternalFunction(
        "lfm25_bf16_gemv_offset_3072",
        source_file=str(KERNEL_DIR / "lfm25_bf16_gemv_offset_3072.cc"),
        arg_types=[weight_tile_ty, projection_ty, accum_ty, np.int32],
        include_dirs=include,
    )
    store_proj = ExternalFunction(
        "lfm25_cast_store_3072",
        source_file=str(KERNEL_DIR / "lfm25_cast_store_3072.cc"),
        arg_types=[accum_ty, projection_ty, np.int32],
        include_dirs=include,
    )
    store_final = ExternalFunction(
        "lfm25_cast_store_1024",
        source_file=str(KERNEL_DIR / "lfm25_cast_store_1024.cc"),
        arg_types=[accum_ty, hidden_ty, np.int32],
        include_dirs=include,
    )
    cast_tile = ExternalFunction(
        "lfm25_f32_to_bf16", source_file=str(KERNEL_DIR / "lfm25_f32_to_bf16.cc"),
        arg_types=[accum_ty, bf16_tile_ty, np.int32], include_dirs=include,
    )
    add_proj = ExternalFunction(
        "lfm25_add_projection_inplace",
        source_file=str(KERNEL_DIR / "lfm25_add_projection_inplace.cc"),
        arg_types=[hidden_ty, projection_ty], include_dirs=include,
    )
    add_output = ExternalFunction(
        "lfm25_add_output_inplace",
        source_file=str(KERNEL_DIR / "lfm25_add_output_inplace.cc"),
        arg_types=[hidden_ty, hidden_ty], include_dirs=include,
    )
    silu_gate = ExternalFunction(
        "lfm25_silu_gate", source_file=str(KERNEL_DIR / "lfm25_silu_gate.cc"),
        arg_types=[bf16_tile_ty, bf16_tile_ty, bf16_tile_ty, np.int32],
        include_dirs=include,
    )

    def core_fn(data_in, weights_in, final_out, act, proj, res, acc,
                w1_part, w3_part, gate_part, norm_op, gemv_op, gemv_w2_op,
                store_proj_op, store_final_op, cast_op, add_proj_op,
                add_output_op, silu_op):
        data = data_in.acquire(1)
        for i in range_(HIDDEN):
            res[i] = data[i]
            act[i] = data[HIDDEN + i]
        data_in.release(1)

        for output_tile in range_(HIDDEN // TILE):
            for i in range_(TILE):
                acc[i] = 0.0
            for input_tile in range_(HIDDEN // TILE):
                w = weights_in.acquire(1)
                gemv_op(w, act, acc, input_tile * TILE)
                weights_in.release(1)
            store_proj_op(acc, proj, output_tile * TILE)
        add_proj_op(res, proj)

        gamma = weights_in.acquire(1)
        norm_op(res, gamma, act, HIDDEN)
        weights_in.release(1)

        for output_tile in range_(FFN // TILE):
            for i in range_(TILE):
                acc[i] = 0.0
            for input_tile in range_(HIDDEN // TILE):
                w = weights_in.acquire(1)
                gemv_op(w, act, acc, input_tile * TILE)
                weights_in.release(1)
            cast_op(acc, w1_part, TILE)
            for i in range_(TILE):
                acc[i] = 0.0
            for input_tile in range_(HIDDEN // TILE):
                w = weights_in.acquire(1)
                gemv_op(w, act, acc, input_tile * TILE)
                weights_in.release(1)
            cast_op(acc, w3_part, TILE)
            silu_op(w1_part, w3_part, gate_part, TILE)
            for i in range_(TILE):
                proj[output_tile * TILE + i] = gate_part[i]

        final = final_out.acquire(1)
        for output_tile in range_(HIDDEN // TILE):
            for i in range_(TILE):
                acc[i] = 0.0
            for input_tile in range_(FFN // TILE):
                w = weights_in.acquire(1)
                gemv_w2_op(w, proj, acc, input_tile * TILE)
                weights_in.release(1)
            store_final_op(acc, final, output_tile * TILE)
        add_output_op(final, res)
        final_out.release(1)

    worker = Worker(
        core_fn,
        [data_fifo.cons(), weight_fifo.cons(), output_fifo.prod(), activation,
         projection, residual, accum, w1_tile, w3_tile, gate_tile, norm, gemv,
         gemv_w2, store_proj, store_final, cast_tile, add_proj, add_output, silu_gate],
    )

    offsets = {"output": 0, "ffn_gamma": HIDDEN * HIDDEN}
    offset = offsets["ffn_gamma"] + WEIGHT_TILE
    for name in ("w1", "w3", "w2"):
        rows, cols = next((r, c) for n, r, c in MATRICES if n == name)
        offsets[name] = offset
        offset += rows * cols
    assert offset == WEIGHT_LEN
    gamma_tap = TensorAccessPattern(
        (WEIGHT_LEN,), offsets["ffn_gamma"],
        [1, 1, 1, WEIGHT_TILE], [0, 0, 0, 1],
    )
    matrix_taps = {}
    for name, rows, cols in MATRICES:
        source = TensorTiler2D.group_tiler((rows, cols), (TILE, TILE), (1, cols // TILE))
        matrix_taps[name] = [
            TensorAccessPattern((WEIGHT_LEN,), offsets[name] + tap.offset, tap.sizes, tap.strides)
            for tap in source
        ]

    def fill_weight(weight, producer, tap):
        group = TaskGroup()
        producer.fill(weight, tap=tap, group=group, wait=True)
        group.finish()

    def sequence(data, weight, final, data_prod, weight_prod, final_cons):
        initial = TaskGroup()
        data_prod.fill(data, group=initial, wait=True)
        initial.finish()
        output_group = TaskGroup()
        final_cons.drain(final, group=output_group, wait=True)
        for tap in matrix_taps["output"]:
            fill_weight(weight, weight_prod, tap)
        fill_weight(weight, weight_prod, gamma_tap)
        for w1_tap, w3_tap in zip(matrix_taps["w1"], matrix_taps["w3"]):
            fill_weight(weight, weight_prod, w1_tap)
            fill_weight(weight, weight_prod, w3_tap)
        for tap in matrix_taps["w2"]:
            fill_weight(weight, weight_prod, tap)
        output_group.finish()

    runtime = Runtime(
        sequence,
        [packed_ty, weight_ty, hidden_ty,
         data_fifo.prod(), weight_fifo.prod(), output_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
