# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Adapted from IRON's 03_matrix_multiplication_single_core example to keep
# BF16 inputs while accumulating and returning FP32 outputs, then place
# independent output-column strips on multiple AIE cores.
"""BF16-input, FP32-output matrix multiplication for Phoenix."""

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker, kernels
from aie.iron.controlflow import range_
from ml_dtypes import bfloat16


@iron.jit
def bf16_f32_matmul(
    input0: In,
    input1: In,
    output: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    n_cores: CompileTime[int] = 1,
):
    m, k, n = 16, 64, 64
    if M != m or K % k or N % (n * n_cores) or n_cores not in (1, 2, 4):
        raise ValueError("Needs M=16, K%64=0, N%(64*n_cores)=0, cores in 1/2/4")
    matmul = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=bfloat16,
        output_dtype=np.float32,
        vectorized=True,
    )
    r, s, t = matmul.mac_dims
    A_ty = np.ndarray[(M, K), np.dtype[bfloat16]]
    B_ty = np.ndarray[(K, N), np.dtype[bfloat16]]
    C_ty = np.ndarray[(M, N), np.dtype[np.float32]]
    a_ty = np.ndarray[(m * k,), np.dtype[bfloat16]]
    b_ty = np.ndarray[(k * n,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(m * n,), np.dtype[np.float32]]

    tap_A = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
    tap_B = TensorTiler2D.group_tiler((k, n), (s, t), (k // s, n // t))[0]
    tap_C = TensorAccessPattern(
        tensor_dims=(m, n),
        offset=0,
        sizes=[m // r, r, n // t, t],
        strides=[r * n, t, r * t, 1],
    )
    fifo_A_L3L2 = []
    fifo_B_L3L2 = []
    fifo_C_L2L3 = []
    core_inputs = []
    for col in range(n_cores):
        a_l3l2 = ObjectFifo(a_ty, name=f"A_L3L2_{col}")
        a_l2l1 = a_l3l2.cons().forward(
            dims_to_stream=tap_A.transformation_dims, name=f"A_L2L1_{col}"
        )
        b_l3l2 = ObjectFifo(b_ty, name=f"B_L3L2_{col}")
        b_l2l1 = b_l3l2.cons().forward(
            dims_to_stream=tap_B.transformation_dims, name=f"B_L2L1_{col}"
        )
        c_l1l2 = ObjectFifo(c_ty, name=f"C_L1L2_{col}")
        c_l2l3 = c_l1l2.cons().forward(
            dims_to_stream=list(tap_C.transformation_dims), name=f"C_L2L3_{col}"
        )
        fifo_A_L3L2.append(a_l3l2)
        fifo_B_L3L2.append(b_l3l2)
        fifo_C_L2L3.append(c_l2l3)
        core_inputs.append([a_l2l1.cons(), b_l2l1.cons(), c_l1l2.prod(), matmul])

    def core_fn(of_a, of_b, of_c, kernel):
        for _ in range_(N // (n * n_cores)):
            elem_out = of_c.acquire(1)
            for i in range_(m * n):
                elem_out[i] = 0
            for _ in range_(K // k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                kernel(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    workers = Worker.grid(
        1, n_cores, lambda _row, col: Worker(core_fn, core_inputs[col])
    )
    a_tap = TensorTiler2D.group_tiler((M, K), (m, k), (1, K // k))[0]
    # A single descriptor cannot express the entire 6 MB weight matrix.
    # Stream one 64-column strip per task group within one NPU invocation.
    b_taps = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K // k, 1), tile_group_col_major=True
    )
    c_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, 1))

    def sequence(A, B, C, a_prods, b_prods, c_conses):
        for tile_group in range(N // (n * n_cores)):
            group = TaskGroup()
            for col in range(n_cores):
                tile_col = tile_group * n_cores + col
                a_prods[col].fill(A, tap=a_tap, group=group)
                b_prods[col].fill(B, tap=b_taps[tile_col], group=group)
                c_conses[col].drain(C, tap=c_taps[tile_col], group=group, wait=True)
            group.finish()

    runtime = Runtime(
        sequence,
        [
            A_ty,
            B_ty,
            C_ty,
            [f.prod() for f in fifo_A_L3L2],
            [f.prod() for f in fifo_B_L3L2],
            [f.cons() for f in fifo_C_L2L3],
        ],
    )
    return Program(
        iron.get_current_device(), runtime, workers=[w for row in workers for w in row]
    ).resolve_program()
