"""One Phoenix IRON program for RMSNorm and a real BF16 decode GEMV."""

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


@iron.jit
def norm_bf16_gemv(
    hidden: In,
    packed_gamma_and_weight: In,
    output: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
):
    if K != 1024 or M % 64:
        raise ValueError("The first experiment needs K=1024 and M divisible by 64")
    tile = 64
    gamma_padded = tile * tile
    packed_len = gamma_padded + M * K
    hidden_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(packed_len,), np.dtype[bfloat16]]
    output_ty = np.ndarray[(M,), np.dtype[bfloat16]]
    w_tile_ty = np.ndarray[(tile * tile,), np.dtype[bfloat16]]
    y_tile_ty = np.ndarray[(tile,), np.dtype[bfloat16]]
    accum_ty = np.ndarray[(tile,), np.dtype[np.float32]]

    hidden_fifo = ObjectFifo(hidden_ty, name="fused_hidden", depth=1)
    weight_fifo = ObjectFifo(w_tile_ty, name="fused_weight", depth=2)
    output_fifo = ObjectFifo(y_tile_ty, name="fused_output", depth=1)
    normed = Buffer(hidden_ty, name="fused_normalized")
    accum = Buffer(accum_ty, name="fused_accum")

    norm = ExternalFunction(
        "lfm25_rms_norm",
        source_file=str(KERNEL_DIR / "lfm25_rms_norm.cc"),
        arg_types=[hidden_ty, w_tile_ty, hidden_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )
    gemv = ExternalFunction(
        "lfm25_bf16_gemv_offset",
        source_file=str(KERNEL_DIR / "lfm25_bf16_gemv_offset.cc"),
        arg_types=[w_tile_ty, hidden_ty, accum_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )
    cast = ExternalFunction(
        "lfm25_f32_to_bf16",
        source_file=str(KERNEL_DIR / "lfm25_f32_to_bf16.cc"),
        arg_types=[accum_ty, y_tile_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(hidden_in, weights_in, output_out, norm_buffer, acc_buffer,
                norm_op, gemv_op, cast_op):
        x = hidden_in.acquire(1)
        gamma = weights_in.acquire(1)
        norm_op(x, gamma, norm_buffer, K)
        weights_in.release(1)
        hidden_in.release(1)
        for _ in range_(M // tile):
            y = output_out.acquire(1)
            for i in range_(tile):
                acc_buffer[i] = 0.0
            for k_index in range_(K // tile):
                w = weights_in.acquire(1)
                gemv_op(w, norm_buffer, acc_buffer, k_index * tile)
                weights_in.release(1)
            cast_op(acc_buffer, y, tile)
            output_out.release(1)

    worker = Worker(
        core_fn,
        [
            hidden_fifo.cons(), weight_fifo.cons(), output_fifo.prod(),
            normed, accum, norm, gemv, cast,
        ],
    )

    gamma_tap = TensorAccessPattern(
        (packed_len,), 0, [1, 1, 1, gamma_padded], [0, 0, 0, 1]
    )
    weight_taps_original = TensorTiler2D.group_tiler(
        (M, K), (tile, tile), (1, K // tile)
    )
    weight_taps = [
        TensorAccessPattern(
            (packed_len,), gamma_padded + tap.offset, tap.sizes, tap.strides
        )
        for tap in weight_taps_original
    ]
    output_taps = TensorTiler2D.simple_tiler((1, M), (1, tile))

    def sequence(x, packed, y, x_prod, w_prod, y_cons):
        initial = TaskGroup()
        x_prod.fill(x, group=initial)
        w_prod.fill(packed, tap=gamma_tap, group=initial)
        initial.finish()
        for group_index in range(M // tile):
            group = TaskGroup()
            w_prod.fill(packed, tap=weight_taps[group_index], group=group)
            y_cons.drain(y, tap=output_taps[group_index], group=group, wait=True)
            group.finish()

    runtime = Runtime(
        sequence,
        [
            hidden_ty, packed_ty, output_ty,
            hidden_fifo.prod(), weight_fifo.prod(), output_fifo.cons(),
        ],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
