"""Four-core BF16 GEMV with FP32 accumulation and BF16 output in one NPU program."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorTiler2D
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16


KERNEL_DIR = Path(__file__).resolve().parent / "kernels"


@iron.jit
def bf16_bf16_gemv(
    weights: In,
    activation: In,
    output: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    n_cores: CompileTime[int] = 4,
):
    m = k = 64
    if M % (m * n_cores) or K % k or n_cores not in (1, 2, 4):
        raise ValueError("Needs M divisible by 64*cores, K divisible by 64")
    weight_ty = np.ndarray[(M, K), np.dtype[bfloat16]]
    activation_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    output_ty = np.ndarray[(M,), np.dtype[bfloat16]]
    w_tile_ty = np.ndarray[(m * k,), np.dtype[bfloat16]]
    x_tile_ty = np.ndarray[(k,), np.dtype[bfloat16]]
    accum_ty = np.ndarray[(m,), np.dtype[np.float32]]
    y_tile_ty = np.ndarray[(m,), np.dtype[bfloat16]]
    gemv = ExternalFunction(
        "lfm25_bf16_gemv",
        source_file=str(KERNEL_DIR / "lfm25_bf16_gemv.cc"),
        arg_types=[w_tile_ty, x_tile_ty, accum_ty],
        include_dirs=[config.cxx_header_path()],
    )
    cast = ExternalFunction(
        "lfm25_f32_to_bf16",
        source_file=str(KERNEL_DIR / "lfm25_f32_to_bf16.cc"),
        arg_types=[accum_ty, y_tile_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )
    weight_fifos = [ObjectFifo(w_tile_ty, name=f"bf16_weight_{i}") for i in range(n_cores)]
    activation_fifos = [
        ObjectFifo(x_tile_ty, name=f"bf16_activation_{i}") for i in range(n_cores)
    ]
    output_fifos = [ObjectFifo(y_tile_ty, name=f"bf16_output_{i}") for i in range(n_cores)]

    def core_fn(w_fifo, x_fifo, y_fifo, accum, gemv_op, cast_op):
        for _ in range_(M // (m * n_cores)):
            y = y_fifo.acquire(1)
            for i in range_(m):
                accum[i] = 0.0
            for _ in range_(K // k):
                w = w_fifo.acquire(1)
                x = x_fifo.acquire(1)
                gemv_op(w, x, accum)
                w_fifo.release(1)
                x_fifo.release(1)
            cast_op(accum, y, m)
            y_fifo.release(1)

    workers = Worker.grid(
        1,
        n_cores,
        lambda _row, col: Worker(
            core_fn,
            [
                weight_fifos[col].cons(),
                activation_fifos[col].cons(),
                output_fifos[col].prod(),
                Buffer(accum_ty, name=f"bf16_accum_{col}"),
                gemv,
                cast,
            ],
        ),
    )
    weight_taps = TensorTiler2D.group_tiler((M, K), (m, k), (1, K // k))
    activation_tap = TensorTiler2D.group_tiler((1, K), (1, k), (1, K // k))[0]
    output_taps = TensorTiler2D.simple_tiler((1, M), (1, m))

    def sequence(w, x, y, w_prods, x_prods, y_conses):
        for group_index in range(M // (m * n_cores)):
            group = TaskGroup()
            for col in range(n_cores):
                tile_index = group_index * n_cores + col
                w_prods[col].fill(w, tap=weight_taps[tile_index], group=group)
                x_prods[col].fill(x, tap=activation_tap, group=group)
                y_conses[col].drain(
                    y, tap=output_taps[tile_index], group=group, wait=True
                )
            group.finish()

    runtime = Runtime(
        sequence,
        [
            weight_ty,
            activation_ty,
            output_ty,
            [fifo.prod() for fifo in weight_fifos],
            [fifo.prod() for fifo in activation_fifos],
            [fifo.cons() for fifo in output_fifos],
        ],
    )
    return Program(
        iron.get_current_device(), runtime, workers=[worker for row in workers for worker in row]
    ).resolve_program()
