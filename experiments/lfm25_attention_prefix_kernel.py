"""One-core NPU program for operator norm, Q/K/V projection, head norm, RoPE."""

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
TILE = 64
WEIGHT_TILE = 4096
MATRICES = (("q", 1024), ("k", 512), ("v", 512))
WEIGHT_LEN = 2 * WEIGHT_TILE + sum(rows * 1024 for _, rows in MATRICES)


@iron.jit
def attention_prefix(hidden: In, packed_weights: In, qkv_output: Out):
    hidden_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    weights_ty = np.ndarray[(WEIGHT_LEN,), np.dtype[bfloat16]]
    weight_tile_ty = np.ndarray[(WEIGHT_TILE,), np.dtype[bfloat16]]
    qkv_ty = np.ndarray[(2048,), np.dtype[bfloat16]]
    activation_ty = np.ndarray[(2560,), np.dtype[bfloat16]]
    accum_ty = np.ndarray[(TILE,), np.dtype[np.float32]]
    bf16_tile_ty = np.ndarray[(TILE,), np.dtype[bfloat16]]

    hidden_fifo = ObjectFifo(hidden_ty, name="attention_hidden", depth=1)
    weight_fifo = ObjectFifo(weight_tile_ty, name="attention_weight", depth=1)
    qkv_fifo = ObjectFifo(qkv_ty, name="attention_qkv", depth=1)
    activation = Buffer(activation_ty, name="attention_activation")
    qkv = Buffer(qkv_ty, name="attention_qkv_buffer")
    accum = Buffer(accum_ty, name="attention_accum")
    tile = Buffer(bf16_tile_ty, name="attention_tile")
    include = [config.cxx_header_path()]
    norm_op = ExternalFunction(
        "lfm25_rms_norm", source_file=str(KERNEL_DIR / "lfm25_rms_norm.cc"),
        arg_types=[hidden_ty, weight_tile_ty, activation_ty, np.int32],
        include_dirs=include,
    )
    gemv_op = ExternalFunction(
        "lfm25_bf16_gemv_offset",
        source_file=str(KERNEL_DIR / "lfm25_bf16_gemv_offset.cc"),
        arg_types=[weight_tile_ty, activation_ty, accum_ty, np.int32],
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

    def core_fn(hidden_in, weights_in, qkv_out, act, local_qkv, acc, part,
                norm, gemv, cast, head):
        h = hidden_in.acquire(1)
        gamma = weights_in.acquire(1)
        norm(h, gamma, act, 1024)
        weights_in.release(1)
        hidden_in.release(1)
        for output_tile in range_(2048 // TILE):
            for i in range_(TILE):
                acc[i] = 0.0
            for input_tile in range_(1024 // TILE):
                w = weights_in.acquire(1)
                gemv(w, act, acc, input_tile * TILE)
                weights_in.release(1)
            cast(acc, part, TILE)
            for i in range_(TILE):
                local_qkv[output_tile * TILE + i] = part[i]
        aux = weights_in.acquire(1)
        head(local_qkv, aux)
        weights_in.release(1)
        out = qkv_out.acquire(1)
        for i in range_(2048):
            out[i] = local_qkv[i]
        qkv_out.release(1)

    worker = Worker(
        core_fn,
        [hidden_fifo.cons(), weight_fifo.cons(), qkv_fifo.prod(), activation,
         qkv, accum, tile, norm_op, gemv_op, cast_op, head_op],
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

    def fill_weight(weight, producer, tap):
        group = TaskGroup()
        producer.fill(weight, tap=tap, group=group, wait=True)
        group.finish()

    def sequence(h, weights, result, h_prod, w_prod, qkv_cons):
        initial = TaskGroup()
        h_prod.fill(h, group=initial, wait=True)
        w_prod.fill(weights, tap=gamma_tap, group=initial, wait=True)
        initial.finish()
        output_group = TaskGroup()
        qkv_cons.drain(result, group=output_group, wait=True)
        for name, _ in MATRICES:
            for tap in matrix_taps[name]:
                fill_weight(weights, w_prod, tap)
        fill_weight(weights, w_prod, aux_tap)
        output_group.finish()

    runtime = Runtime(
        sequence,
        [hidden_ty, weights_ty, qkv_ty,
         hidden_fifo.prod(), weight_fifo.prod(), qkv_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
