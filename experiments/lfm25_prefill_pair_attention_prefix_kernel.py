"""Two attention prefix positions sharing streamed Q/K/V weights."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 64
WEIGHT_TILE = 4096
MATRICES = (("q", 1024), ("k", 512), ("v", 512))
WEIGHT_LEN = 2 * WEIGHT_TILE + sum(rows * 1024 for _, rows in MATRICES)


@iron.jit
def attention_prefix_pair(hidden_pair: In, packed_weights_and_aux0: In,
                          aux1: In, qkv_pair: Out):
    hidden_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    hidden_pair_ty = np.ndarray[(2048,), np.dtype[bfloat16]]
    weights_ty = np.ndarray[(WEIGHT_LEN,), np.dtype[bfloat16]]
    weight_tile_ty = np.ndarray[(WEIGHT_TILE,), np.dtype[bfloat16]]
    qkv_ty = np.ndarray[(3072,), np.dtype[bfloat16]]
    qkv_pair_ty = np.ndarray[(6144,), np.dtype[bfloat16]]
    activation_ty = hidden_ty
    accum_ty = np.ndarray[(TILE,), np.dtype[np.float32]]
    bf16_tile_ty = np.ndarray[(TILE,), np.dtype[bfloat16]]

    hidden_fifo = ObjectFifo(hidden_ty, name="attention_hidden", depth=1)
    weight_fifo = ObjectFifo(weight_tile_ty, name="attention_weight", depth=1)
    qkv_fifo = ObjectFifo(qkv_ty, name="attention_qkv", depth=2)
    activation0 = Buffer(activation_ty, name="attention_activation0")
    activation1 = Buffer(activation_ty, name="attention_activation1")
    qkv0 = Buffer(qkv_ty, name="attention_qkv0")
    qkv1 = Buffer(qkv_ty, name="attention_qkv1")
    original0 = Buffer(hidden_ty, name="attention_original0")
    original1 = Buffer(hidden_ty, name="attention_original1")
    accum0 = Buffer(accum_ty, name="attention_accum0")
    accum1 = Buffer(accum_ty, name="attention_accum1")
    tile0 = Buffer(bf16_tile_ty, name="attention_tile0")
    tile1 = Buffer(bf16_tile_ty, name="attention_tile1")
    include = [config.cxx_header_path()]
    norm_op = ExternalFunction(
        "lfm25_rms_norm", source_file=str(KERNEL_DIR / "lfm25_rms_norm.cc"),
        arg_types=[hidden_ty, weight_tile_ty, hidden_ty, np.int32],
        include_dirs=include,
    )
    gemv_op = ExternalFunction(
        "lfm25_bf16_gemv_offset",
        source_file=str(KERNEL_DIR / "lfm25_bf16_gemv_offset.cc"),
        arg_types=[weight_tile_ty, hidden_ty, accum_ty, np.int32],
        include_dirs=include,
    )
    cast_op = ExternalFunction(
        "lfm25_f32_to_bf16", source_file=str(KERNEL_DIR / "lfm25_f32_to_bf16.cc"),
        arg_types=[accum_ty, bf16_tile_ty, np.int32], include_dirs=include,
    )
    head_op = ExternalFunction(
        "lfm25_attention_norm_rope",
        source_file=str(KERNEL_DIR / "lfm25_attention_norm_rope.cc"),
        arg_types=[qkv_ty, weight_tile_ty], include_dirs=include,
    )

    def core_fn(hidden_in, weights_in, qkv_out,
                act0, act1, local0, local1, saved0, saved1,
                acc0, acc1, part0, part1, norm, gemv, cast, head):
        gamma = weights_in.acquire(1)
        h0 = hidden_in.acquire(1)
        for i in range_(1024):
            saved0[i] = h0[i]
        norm(h0, gamma, act0, 1024)
        hidden_in.release(1)
        h1 = hidden_in.acquire(1)
        for i in range_(1024):
            saved1[i] = h1[i]
        norm(h1, gamma, act1, 1024)
        hidden_in.release(1)
        weights_in.release(1)
        for output_tile in range_(2048 // TILE):
            for i in range_(TILE):
                acc0[i] = 0.0
                acc1[i] = 0.0
            for input_tile in range_(1024 // TILE):
                w = weights_in.acquire(1)
                gemv(w, act0, acc0, input_tile * TILE)
                gemv(w, act1, acc1, input_tile * TILE)
                weights_in.release(1)
            cast(acc0, part0, TILE)
            cast(acc1, part1, TILE)
            for i in range_(TILE):
                local0[output_tile * TILE + i] = part0[i]
                local1[output_tile * TILE + i] = part1[i]
        aux0 = weights_in.acquire(1)
        head(local0, aux0)
        weights_in.release(1)
        aux1 = weights_in.acquire(1)
        head(local1, aux1)
        weights_in.release(1)
        out = qkv_out.acquire(2)
        for i in range_(2048):
            out[0][i] = local0[i]
            out[1][i] = local1[i]
        for i in range_(1024):
            out[0][2048 + i] = saved0[i]
            out[1][2048 + i] = saved1[i]
        qkv_out.release(2)

    worker = Worker(
        core_fn,
        [hidden_fifo.cons(), weight_fifo.cons(), qkv_fifo.prod(),
         activation0, activation1, qkv0, qkv1, original0, original1,
         accum0, accum1, tile0, tile1,
         norm_op, gemv_op, cast_op, head_op],
    )
    offsets = {"gamma": 0, "aux": WEIGHT_LEN - WEIGHT_TILE}
    offset = WEIGHT_TILE
    for name, rows in MATRICES:
        offsets[name] = offset
        offset += rows * 1024
    assert offset == offsets["aux"]
    gamma_tap = TensorAccessPattern((WEIGHT_LEN,), 0, [1, 1, 1, WEIGHT_TILE], [0, 0, 0, 1])
    aux_tap = TensorAccessPattern((WEIGHT_LEN,), offsets["aux"], [1, 1, 1, WEIGHT_TILE], [0, 0, 0, 1])
    matrix_taps = {}
    for name, rows in MATRICES:
        source = TensorTiler2D.group_tiler((rows, 1024), (TILE, TILE), (1, 1024 // TILE))
        matrix_taps[name] = [
            TensorAccessPattern((WEIGHT_LEN,), offsets[name] + tap.offset, tap.sizes, tap.strides)
            for tap in source
        ]
    hidden_taps = [TensorAccessPattern(
        (2048,), index * 1024,
        [1, 1, 1, 1024], [0, 0, 0, 1],
    ) for index in range(2)]
    qkv_taps = [TensorAccessPattern(
        (6144,), index * 3072,
        [1, 1, 1, 3072], [0, 0, 0, 1],
    ) for index in range(2)]

    def fill_weight(weight, producer, tap):
        group = TaskGroup()
        producer.fill(weight, tap=tap, group=group, wait=True)
        group.finish()

    def sequence(h, weights, aux_second, result, h_prod, w_prod, qkv_cons):
        initial = TaskGroup()
        w_prod.fill(weights, tap=gamma_tap, group=initial, wait=True)
        for tap in hidden_taps:
            h_prod.fill(h, tap=tap, group=initial, wait=True)
        initial.finish()
        output_group = TaskGroup()
        for tap in qkv_taps:
            qkv_cons.drain(result, tap=tap, group=output_group, wait=True)
        for name, _ in MATRICES:
            for tap in matrix_taps[name]:
                fill_weight(weights, w_prod, tap)
        fill_weight(weights, w_prod, aux_tap)
        fill_weight(aux_second, w_prod, TensorAccessPattern(
            (WEIGHT_TILE,), 0, [1, 1, 1, WEIGHT_TILE], [0, 0, 0, 1],
        ))
        output_group.finish()

    runtime = Runtime(
        sequence,
        [hidden_pair_ty, weights_ty, weight_tile_ty, qkv_pair_ty,
         hidden_fifo.prod(), weight_fifo.prod(), qkv_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
