"""Two attention positions through one fused output projection and FFN."""

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
def attention_tail_pair(packed_hidden_and_context: In, packed_weights: In,
                        block_output: Out):
    packed_ty = np.ndarray[(4 * HIDDEN,), np.dtype[bfloat16]]
    single_packed_ty = np.ndarray[(2 * HIDDEN,), np.dtype[bfloat16]]
    weight_ty = np.ndarray[(WEIGHT_LEN,), np.dtype[bfloat16]]
    weight_tile_ty = np.ndarray[(WEIGHT_TILE,), np.dtype[bfloat16]]
    hidden_ty = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    activation_ty = hidden_ty
    projection_ty = np.ndarray[(3072,), np.dtype[bfloat16]]
    accum_ty = np.ndarray[(TILE,), np.dtype[np.float32]]
    bf16_tile_ty = np.ndarray[(TILE,), np.dtype[bfloat16]]

    data_fifo = ObjectFifo(single_packed_ty, name="tail_data", depth=1)
    weight_fifo = ObjectFifo(weight_tile_ty, name="tail_weight", depth=2)
    output_fifo = ObjectFifo(hidden_ty, name="tail_output", depth=2)
    activation0 = Buffer(activation_ty, name="tail_activation0")
    activation1 = Buffer(activation_ty, name="tail_activation1")
    projection0 = Buffer(projection_ty, name="tail_projection0")
    projection1 = Buffer(projection_ty, name="tail_projection1")
    residual0 = Buffer(hidden_ty, name="tail_residual0")
    residual1 = Buffer(hidden_ty, name="tail_residual1")
    accum0 = Buffer(accum_ty, name="tail_accum0")
    accum1 = Buffer(accum_ty, name="tail_accum1")
    w1_tile0 = Buffer(bf16_tile_ty, name="tail_w1_0")
    w1_tile1 = Buffer(bf16_tile_ty, name="tail_w1_1")
    w3_tile0 = Buffer(bf16_tile_ty, name="tail_w3_0")
    w3_tile1 = Buffer(bf16_tile_ty, name="tail_w3_1")
    gate_tile0 = Buffer(bf16_tile_ty, name="tail_gate0")
    gate_tile1 = Buffer(bf16_tile_ty, name="tail_gate1")
    include = [config.cxx_header_path()]
    norm = ExternalFunction(
        "lfm25_rms_norm", source_file=str(KERNEL_DIR / "lfm25_rms_norm.cc"),
        arg_types=[hidden_ty, weight_tile_ty, hidden_ty, np.int32],
        include_dirs=include,
    )
    gemv = ExternalFunction(
        "lfm25_bf16_gemv_offset",
        source_file=str(KERNEL_DIR / "lfm25_bf16_gemv_offset.cc"),
        arg_types=[weight_tile_ty, hidden_ty, accum_ty, np.int32],
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

    def core_fn(data_in, weights_in, final_out,
                act0, act1, proj0, proj1, res0, res1, acc0, acc1,
                w1_0, w1_1, w3_0, w3_1, gate0, gate1,
                norm_op, gemv_op, gemv_w2_op, store_proj_op, store_final_op,
                cast_op, add_proj_op, add_output_op, silu_op):
        data0 = data_in.acquire(1)
        for i in range_(HIDDEN):
            res0[i] = data0[i]
            act0[i] = data0[HIDDEN + i]
        data_in.release(1)
        data1 = data_in.acquire(1)
        for i in range_(HIDDEN):
            res1[i] = data1[i]
            act1[i] = data1[HIDDEN + i]
        data_in.release(1)

        for output_tile in range_(HIDDEN // TILE):
            for i in range_(TILE):
                acc0[i] = 0.0
                acc1[i] = 0.0
            for input_tile in range_(HIDDEN // TILE):
                w = weights_in.acquire(1)
                gemv_op(w, act0, acc0, input_tile * TILE)
                gemv_op(w, act1, acc1, input_tile * TILE)
                weights_in.release(1)
            store_proj_op(acc0, proj0, output_tile * TILE)
            store_proj_op(acc1, proj1, output_tile * TILE)
        add_proj_op(res0, proj0)
        add_proj_op(res1, proj1)

        gamma = weights_in.acquire(1)
        norm_op(res0, gamma, act0, HIDDEN)
        norm_op(res1, gamma, act1, HIDDEN)
        weights_in.release(1)

        for output_tile in range_(FFN // TILE):
            for i in range_(TILE):
                acc0[i] = 0.0
                acc1[i] = 0.0
            for input_tile in range_(HIDDEN // TILE):
                w = weights_in.acquire(1)
                gemv_op(w, act0, acc0, input_tile * TILE)
                gemv_op(w, act1, acc1, input_tile * TILE)
                weights_in.release(1)
            cast_op(acc0, w1_0, TILE)
            cast_op(acc1, w1_1, TILE)
            for i in range_(TILE):
                acc0[i] = 0.0
                acc1[i] = 0.0
            for input_tile in range_(HIDDEN // TILE):
                w = weights_in.acquire(1)
                gemv_op(w, act0, acc0, input_tile * TILE)
                gemv_op(w, act1, acc1, input_tile * TILE)
                weights_in.release(1)
            cast_op(acc0, w3_0, TILE)
            cast_op(acc1, w3_1, TILE)
            silu_op(w1_0, w3_0, gate0, TILE)
            silu_op(w1_1, w3_1, gate1, TILE)
            for i in range_(TILE):
                proj0[output_tile * TILE + i] = gate0[i]
                proj1[output_tile * TILE + i] = gate1[i]

        finals = final_out.acquire(2)
        for output_tile in range_(HIDDEN // TILE):
            for i in range_(TILE):
                acc0[i] = 0.0
                acc1[i] = 0.0
            for input_tile in range_(FFN // TILE):
                w = weights_in.acquire(1)
                gemv_w2_op(w, proj0, acc0, input_tile * TILE)
                gemv_w2_op(w, proj1, acc1, input_tile * TILE)
                weights_in.release(1)
            store_final_op(acc0, finals[0], output_tile * TILE)
            store_final_op(acc1, finals[1], output_tile * TILE)
        add_output_op(finals[0], res0)
        add_output_op(finals[1], res1)
        final_out.release(2)

    worker = Worker(
        core_fn,
        [data_fifo.cons(), weight_fifo.cons(), output_fifo.prod(),
         activation0, activation1, projection0, projection1,
         residual0, residual1, accum0, accum1,
         w1_tile0, w1_tile1, w3_tile0, w3_tile1, gate_tile0, gate_tile1,
         norm, gemv,
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
    input_taps = [TensorAccessPattern(
        (4 * HIDDEN,), position * 2 * HIDDEN,
        [1, 1, 1, 2 * HIDDEN], [0, 0, 0, 1],
    ) for position in range(2)]
    output_taps = [TensorAccessPattern(
        (2 * HIDDEN,), position * HIDDEN,
        [1, 1, 1, HIDDEN], [0, 0, 0, 1],
    ) for position in range(2)]

    def sequence(data, weight, final, data_prod, weight_prod, final_cons):
        initial = TaskGroup()
        for tap in input_taps:
            data_prod.fill(data, tap=tap, group=initial, wait=True)
        initial.finish()
        output_group = TaskGroup()
        for tap in output_taps:
            final_cons.drain(final, tap=tap, group=output_group, wait=True)
        taps = list(matrix_taps["output"])
        taps.append(gamma_tap)
        for w1_tap, w3_tap in zip(matrix_taps["w1"], matrix_taps["w3"]):
            taps.extend((w1_tap, w3_tap))
        taps.extend(matrix_taps["w2"])
        # The FIFO preserves tile order; completing the second transfer also
        # means the first tile has been consumed before both tasks are freed.
        for index in range(0, len(taps), 2):
            group = TaskGroup()
            pair = taps[index:index + 2]
            for pair_index, tap in enumerate(pair):
                weight_prod.fill(weight, tap=tap, group=group,
                                 wait=pair_index == len(pair) - 1)
            group.finish()
        output_group.finish()

    runtime = Runtime(
        sequence,
        [packed_ty, weight_ty, np.ndarray[(2 * HIDDEN,), np.dtype[bfloat16]],
         data_fifo.prod(), weight_fifo.prod(), output_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
