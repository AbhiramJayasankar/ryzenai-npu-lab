"""Experiment 013: time basic operations on the Phoenix NPU and check them against NumPy.

Every case reuses a design that already exists: AMD's IRON programming
examples (unchanged; only their own size/type parameters are set) or a repo
kernel that ran cleanly in an earlier experiment. Programs are compiled by
IRON and submitted directly through pyxrt (npu_direct), so the time per call
is the NPU's, not IRON's Python wrapper's.

One case per process (several hardware contexts in one process can crash
pyxrt). Run through the timeout wrapper:

  powershell -NoProfile -File scripts\\run_npu_timeout.ps1 -Script experiments\\013_npu_bench.py -ScriptArgs "gemm_bf16_wa_1024" -Seconds 600

  python 013_npu_bench.py --list              list cases
  python 013_npu_bench.py --compile a b ...   compile only (no hardware context)
  python 013_npu_bench.py CASE                run one case, append to cache/013/npu.jsonl

Two times are reported per case:
  run_ms  inputs already in the XRT buffers: kernel start + wait (what an
          application that keeps its data in NPU-visible memory pays)
  e2e_ms  copy NumPy inputs into the buffers, sync, run, sync the output back
          and copy it out (comparable to a GPU call that includes transfers)
"""

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "cache/iron/mlir-aie/programming_examples"
OUT = ROOT / "cache/013/npu.jsonl"
sys.path.insert(0, str(Path(__file__).resolve().parent))

KEEP = []  # pyxrt objects crash the process when collected late


def load(rel, name):
    """Import an AMD example file under a unique module name."""
    path = EX / rel
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rng():
    return np.random.default_rng(13)


def bf(x):
    return np.asarray(x, np.float32).astype(bfloat16)


# ---------------------------------------------------------------------------
# Case builders. Each returns a dict:
#   design   CompilableDesign (already specialized)
#   args     list of (dtype, numpy array or None for outputs, count)
#   check    fn(list of output arrays) -> dict with at least "ok"
#   work     dict: bytes (DDR bytes read+written), macs, elems
#   source   where the design comes from
# ---------------------------------------------------------------------------

def c_memcpy(size, cols, chans, bypass):
    mod = load("basic/memcpy/memcpy.py", "ex_memcpy")
    d = mod.memcpy.compilable.specialize(size=size, num_columns=cols, num_channels=chans,
                                          bypass=bypass)
    a = np.arange(1, size + 1, dtype=np.int32)

    def check(outs):
        return {"ok": bool(np.array_equal(outs[0], a)), "mismatches": int((outs[0] != a).sum())}

    return dict(design=d, args=[(np.int32, a, size), (np.int32, None, size)], check=check,
                work=dict(bytes=8 * size, elems=size),
                source=f"AMD basic/memcpy (cols={cols}, chans={chans}, bypass={bypass})")


def c_saxpy():
    mod = load("getting_started/01_SAXPY/saxpy.py", "ex_saxpy")
    n = 4096
    d = mod.saxpy.compilable.specialize(N=n, element_type=bfloat16)
    r = rng()
    x, y = bf(r.uniform(-4, 4, n)), bf(r.uniform(-4, 4, n))
    ref = (3 * x.astype(np.float32) + y.astype(np.float32))

    def check(outs):
        got = outs[0].astype(np.float32)
        err = np.abs(got - ref)
        tol = np.abs(ref) * 2 ** -7 + 2 ** -10   # one BF16 rounding of the result
        return {"ok": bool((err <= tol).all()), "max_abs_err": float(err.max())}

    return dict(design=d, args=[(bfloat16, x, n), (bfloat16, y, n), (bfloat16, None, n)],
                check=check, work=dict(bytes=6 * n, elems=n, flops=2 * n),
                source="AMD getting_started/01_SAXPY (1 core, z = 3x + y, N fixed at 4096)")


def c_eltwise(op, n):
    mod = load("ml/eltwise/eltwise.py", "ex_eltwise")
    d = mod.eltwise.compilable.specialize(size=n, num_channels=1, op=op)
    r = rng()
    a, b = bf(r.uniform(-1, 1, n)), bf(r.uniform(-1, 1, n))
    exact = (np.add if op == "add" else np.multiply)(a.astype(np.float32), b.astype(np.float32))
    rne = exact.astype(bfloat16).astype(np.float32)                  # round to nearest even
    trunc = (exact.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)  # round toward zero
    ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(exact), 1e-30))) - 7)

    def check(outs):
        got = outs[0].astype(np.float32)
        err = np.abs(got - exact)
        return {"ok": bool((err <= ulp).all()), "max_err_in_ulp": float((err / ulp).max()),
                "frac_equal_round_nearest_even": float((got == rne).mean()),
                "frac_equal_truncation": float((got == trunc).mean()),
                "tolerance": "within 1 BF16 ulp of the exact result"}

    return dict(design=d, args=[(bfloat16, a, n), (bfloat16, b, n), (bfloat16, None, n)],
                check=check, work=dict(bytes=6 * n, elems=n),
                source=f"AMD ml/eltwise op={op} (4 columns x 1 core)")


def c_unary(op, n):
    mod = load("ml/eltwise_unary/eltwise_unary.py", "ex_unary")
    d = mod.eltwise_unary.compilable.specialize(size=n, num_channels=2, op=op)
    r = rng()
    x = bf(r.uniform(-3, 3, n))
    ref_fn, kw = mod._VERIFY_CFG[op]
    ref = np.asarray(ref_fn(x)).astype(np.float32)
    rtol, atol = kw.get("rtol", 0.0), kw.get("atol", 0.0)

    def check(outs):
        got = outs[0].astype(np.float32)
        bad = np.abs(got - ref) > atol + rtol * np.abs(ref)
        return {"ok": bool(not bad.any()), "max_abs_err": float(np.abs(got - ref).max()),
                "tolerance": f"rtol={rtol} atol={atol} (AMD's)"}

    return dict(design=d, args=[(bfloat16, x, n), (bfloat16, None, n)], check=check,
                work=dict(bytes=4 * n, elems=n),
                source=f"AMD ml/eltwise_unary op={op} (4 columns x 2 cores)")


def c_exp(n):
    mod = load("basic/vector_exp/vector_exp.py", "ex_exp")
    from aie.iron import kernels
    d = mod.vector_exp.compilable.specialize(N=n)
    r = rng()
    x = bf(r.uniform(-8, 8, n))
    ref = np.asarray(kernels.bf16_exp_ref(x)).astype(np.float32)

    def check(outs):
        got = outs[0].astype(np.float32)
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)
        return {"ok": bool((rel <= 0.128).all()), "max_rel_err": float(rel.max()),
                "median_rel_err": float(np.median(rel)), "tolerance": "rel 12.8% (AMD's)"}

    return dict(design=d, args=[(bfloat16, x, n), (bfloat16, None, n)], check=check,
                work=dict(bytes=4 * n, elems=n), source="AMD basic/vector_exp (4 cores, BF16 exp)")


def c_softmax(n):
    mod = load("ml/softmax/softmax.py", "ex_softmax")
    from aie.iron import kernels
    d = mod.softmax.compilable.specialize(size=n, num_channels=2)
    r = rng()
    x = bf(r.uniform(-4, 8, n))
    ref = np.asarray(kernels.softmax_ref(x, tile_size=1024)).astype(np.float32)

    def check(outs):
        got = outs[0].astype(np.float32)
        bad = np.abs(got - ref) > 0.128 * np.abs(ref)
        return {"ok": bool(not bad.any()), "max_abs_err": float(np.abs(got - ref).max()),
                "tolerance": "rtol 0.128 (AMD's)"}

    return dict(design=d, args=[(bfloat16, x, n), (bfloat16, None, n)], check=check,
                work=dict(bytes=4 * n, elems=n),
                source="AMD ml/softmax (softmax over each 1024-element row, 4 columns x 2 cores)")


def c_reduce_max_1core(n, dtype):
    mod = load("basic/vector_reduce_max/single_core_designs/vector_reduce_max.py", "ex_rmax1")
    d = mod.vector_reduce_max.compilable.specialize(num_elements=n, dtype=dtype)
    r = rng()
    x = (r.integers(-10 ** 6, 10 ** 6, n).astype(np.int32) if dtype == np.int32
         else bf(r.standard_normal(n)))
    nout = 4 // np.dtype(dtype).itemsize

    def check(outs):
        got = outs[0][0]
        return {"ok": bool(got == x.max()), "got": float(got), "expected": float(x.max())}

    return dict(design=d, args=[(dtype, x, n), (dtype, None, nout)], check=check,
                work=dict(bytes=n * np.dtype(dtype).itemsize, elems=n),
                source=f"AMD basic/vector_reduce_max single_core ({np.dtype(dtype).name})")


def c_reduce_max_8core(nbytes, dtype_str):
    mod = load("basic/vector_reduce_max/multi_column_designs/row_wise_vector_reduce_max.py",
               "ex_rmax8")
    d = mod.vector_reduce_max.compilable.specialize(in1_size=nbytes, out_size=4,
                                                    dtype_str=dtype_str)
    dtype = np.int32 if dtype_str == "i32" else bfloat16
    n = nbytes // np.dtype(dtype).itemsize
    r = rng()
    x = (r.integers(-10 ** 6, 10 ** 6, n).astype(np.int32) if dtype == np.int32
         else bf(r.standard_normal(n)))
    nout = 4 // np.dtype(dtype).itemsize

    def check(outs):
        got = outs[0][0]
        return {"ok": bool(got == x.max()), "got": float(got), "expected": float(x.max())}

    return dict(design=d, args=[(dtype, x, n), (dtype, None, nout)], check=check,
                work=dict(bytes=nbytes, elems=n),
                source=f"AMD basic/vector_reduce_max row_wise (8 cores, {dtype_str})")


def c_reduce_add(n):
    mod = load("basic/vector_reduce_add/vector_reduce_add.py", "ex_radd")
    d = mod.vector_reduce_add.compilable.specialize(num_elements=n)
    x = np.arange(n, dtype=np.int32) - n // 2
    x = (x * 37) % 1001 - 500

    x = x.astype(np.int32)

    def check(outs):
        return {"ok": bool(outs[0][0] == x.sum()), "got": int(outs[0][0]), "expected": int(x.sum())}

    return dict(design=d, args=[(np.int32, x, n), (np.int32, None, 1)], check=check,
                work=dict(bytes=4 * n, elems=n), source="AMD basic/vector_reduce_add (1 core, int32)")


def c_scalar_mul(nbytes):
    mod = load("basic/vector_scalar_mul/vector_scalar_mul.py", "ex_smul")
    d = mod.vector_scalar_mul.compilable.specialize(in1_size=nbytes, int_bit_width=16)
    n = nbytes // 2
    x = (np.arange(n) % 1000).astype(np.int16)
    f = np.array([3], np.int32)
    ref = (x.astype(np.int32) * 3).astype(np.int16)

    def check(outs):
        return {"ok": bool(np.array_equal(outs[0], ref))}

    # AMD's argument order is (input, output, factor)
    return dict(design=d, args=[(np.int16, x, n), (np.int16, None, n), (np.int32, f, 1)],
                check=check, work=dict(bytes=4 * n, elems=n),
                source="AMD basic/vector_scalar_mul (1 core, int16)")


def c_gemv_bf16(M, K, cores):
    from lfm25_bf16_gemv_kernel import bf16_bf16_gemv
    d = bf16_bf16_gemv.compilable.specialize(M=M, K=K, n_cores=cores)
    r = rng()
    w = bf(r.standard_normal((M, K)) / np.sqrt(K))
    x = bf(r.standard_normal(K))
    ref = w.astype(np.float32) @ x.astype(np.float32)

    def check(outs):
        got = outs[0].astype(np.float32)
        err = float(np.abs(got - ref).max() / np.abs(ref).max())
        return {"ok": err < 1e-2, "max_err_rel_to_max": err}

    return dict(design=d, args=[(bfloat16, w, M * K), (bfloat16, x, K), (bfloat16, None, M)],
                check=check, work=dict(bytes=2 * (M * K + K + M), macs=M * K),
                source=f"repo lfm25_bf16_gemv_kernel (experiment 006, {cores} cores, FP32 accumulate)")


def c_mv_i16(M, K):
    mod = load("basic/matrix_multiplication/matrix_vector/matrix_vector.py", "ex_mv")
    d = mod.matrix_vector.compilable.specialize(M=M, K=K, m=32, k=32)
    r = rng()
    a = r.integers(-100, 100, (M, K)).astype(np.int16)
    b = r.integers(-100, 100, K).astype(np.int16)
    ref = a.astype(np.int64) @ b.astype(np.int64)

    def check(outs):
        return {"ok": bool(np.array_equal(outs[0].astype(np.int64), ref))}

    return dict(design=d, args=[(np.int16, a, M * K), (np.int16, b, K), (np.int32, None, M)],
                check=check, work=dict(bytes=2 * M * K + 2 * K + 4 * M, macs=M * K),
                source="AMD basic/matrix_multiplication/matrix_vector (1 core, int16 -> int32)")


def c_gemm(M, K, N, m, k, n, din, dout, whole):
    from aie.iron import CompileTime, In, Out
    import aie.iron as iron
    dt = {"bf16": bfloat16, "f32": np.float32, "i8": np.int8, "i16": np.int16, "i32": np.int32}
    if whole:
        wa = load("basic/matrix_multiplication/whole_array/whole_array.py", "ex_wa")
        tag = f"p013wa_{M}_{K}_{N}_{m}_{k}_{n}_{din}_{dout}"

        @iron.jit(tag=tag)
        def gemm(A: In, B: In, C: Out, *, tag: CompileTime[str]):
            return wa._build_design(iron.get_current_device(), M, K, N, m, k, n, 4, din, dout,
                                    0, 0, False, False, False)

        d = gemm.compilable
        src = f"AMD whole_array matmul (16 cores), tile {m}x{k}x{n}"
    else:
        sc = load("basic/matrix_multiplication/single_core/single_core.py", "ex_sc")
        d = sc.single_core.compilable.specialize(M=M, K=K, N=N, m=m, k=k, n=n,
                                                 dtype_in_str=din, dtype_out_str=dout)
        src = f"AMD single_core matmul (1 core), tile {m}x{k}x{n}"
    r = rng()
    if din in ("i8", "i16"):
        lim = 128 if din == "i8" else 256
        a = r.integers(-lim, lim, (M, K)).astype(dt[din])
        b = r.integers(-lim, lim, (K, N)).astype(dt[din])
        ref = a.astype(np.int64) @ b.astype(np.int64)
    else:
        a = bf(r.standard_normal((M, K)))
        b = bf(r.standard_normal((K, N)) / np.sqrt(K))
        ref = a.astype(np.float32) @ b.astype(np.float32)

    def check(outs):
        got = outs[0].reshape(M, N)
        if din in ("i8", "i16"):
            return {"ok": bool(np.array_equal(got.astype(np.int64), ref))}
        err = float(np.abs(got.astype(np.float32) - ref).max() / np.abs(ref).max())
        return {"ok": err < 1e-2, "max_err_rel_to_max": err}

    isz, osz = np.dtype(dt[din]).itemsize, np.dtype(dt[dout]).itemsize
    return dict(design=d, args=[(dt[din], a, M * K), (dt[din], b, K * N), (dt[dout], None, M * N)],
                check=check, work=dict(bytes=isz * (M * K + K * N) + osz * M * N, macs=M * K * N),
                source=src + f", {din} -> {dout}")


def c_conv(width, height, ci, co):
    mod = load("ml/conv2d/conv2d.py", "ex_conv")
    d = mod.conv2d.compilable.specialize(width=width, height=height, in_channels=ci,
                                         out_channels=co, scale=10, fuse_relu=False)
    r = rng()
    x = r.integers(1, 20, (ci, height, width)).astype(np.int8)        # CYX
    w = r.integers(50, 80, (co, ci)).astype(np.int8)                  # OI (1x1 kernel)
    # AMD's layouts (aie.utils.ml.DataShaper tokens, written out in NumPy because
    # that module imports torch): input YCXC8, weights OIYXI8O8, output YCXD.
    x_l = x.reshape(ci // 8, 8, height, width).transpose(2, 0, 3, 1).reshape(-1)
    w_l = w.reshape(co // 8, 8, ci // 8, 8).transpose(0, 2, 3, 1).reshape(-1)
    acc = np.einsum("oi,iyx->oyx", w.astype(np.int64), x.astype(np.int64))
    ref = np.clip(np.round(acc / 1024.0), -128, 127)                  # scale = 10 -> >> 10

    def check(outs):
        o = outs[0].reshape(height, co // 8, width, 8)
        got = o.transpose(1, 3, 0, 2).reshape(co, height, width).astype(np.int64)
        diff = np.abs(got - ref)
        return {"ok": bool(diff.max() <= 1), "max_lsb_diff": int(diff.max()),
                "exact_fraction": float((diff == 0).mean()),
                "reference": "round(acc / 2^10), clipped to int8"}

    n_in, n_out = ci * height * width, co * height * width
    return dict(design=d, args=[(np.int8, np.ascontiguousarray(x_l), n_in),
                                (np.int8, np.ascontiguousarray(w_l), ci * co), (np.int8, None, n_out)],
                check=check, work=dict(bytes=n_in + ci * co + n_out, macs=ci * co * height * width),
                source="AMD ml/conv2d (1x1 conv, int8, 1 core)")


MB = 1 << 20
CASES = {
    # pure data movement (int32 elements)
    "memcpy_dma_4k": lambda: c_memcpy(1024, 1, 1, True),
    "memcpy_dma_32k": lambda: c_memcpy(8192, 4, 2, True),
    "memcpy_dma_1m": lambda: c_memcpy(262144, 4, 2, True),
    "memcpy_dma_64m": lambda: c_memcpy(16 * MB, 4, 2, True),
    "memcpy_core1_4k": lambda: c_memcpy(1024, 1, 1, False),
    "memcpy_core1_1m": lambda: c_memcpy(262144, 1, 1, False),
    "memcpy_core1_16m": lambda: c_memcpy(4 * MB, 1, 1, False),
    "memcpy_core8_64m": lambda: c_memcpy(16 * MB, 4, 2, False),
    # elementwise BF16
    "saxpy_4k": c_saxpy,
    "add_bf16_4k": lambda: c_eltwise("add", 4096),
    "add_bf16_256k": lambda: c_eltwise("add", 262144),
    "add_bf16_16m": lambda: c_eltwise("add", 16 * MB),
    "mul_bf16_4k": lambda: c_eltwise("mul", 4096),
    "mul_bf16_256k": lambda: c_eltwise("mul", 262144),
    "mul_bf16_16m": lambda: c_eltwise("mul", 16 * MB),
    "relu_bf16_8k": lambda: c_unary("relu", 8192),
    "relu_bf16_256k": lambda: c_unary("relu", 262144),
    "relu_bf16_16m": lambda: c_unary("relu", 16 * MB),
    "gelu_bf16_8k": lambda: c_unary("gelu", 8192),
    "gelu_bf16_256k": lambda: c_unary("gelu", 262144),
    "gelu_bf16_16m": lambda: c_unary("gelu", 16 * MB),
    "scale_i16_4k": lambda: c_scalar_mul(8192),
    # nonlinear
    "exp_bf16_4k": lambda: c_exp(4096),
    "exp_bf16_256k": lambda: c_exp(262144),
    "exp_bf16_4m": lambda: c_exp(4 * MB),
    "softmax_bf16_8k": lambda: c_softmax(8192),
    "softmax_bf16_256k": lambda: c_softmax(262144),
    "softmax_bf16_4m": lambda: c_softmax(4 * MB),
    # reductions
    "max_i32_1core_2k": lambda: c_reduce_max_1core(2048, np.int32),
    "max_bf16_1core_2k": lambda: c_reduce_max_1core(2048, bfloat16),
    "max_i32_8core_32k": lambda: c_reduce_max_8core(32768 * 4, "i32"),
    "max_i32_8core_1m": lambda: c_reduce_max_8core(MB * 4, "i32"),
    "max_i32_8core_16m": lambda: c_reduce_max_8core(16 * MB * 4, "i32"),
    "max_bf16_8core_16m": lambda: c_reduce_max_8core(16 * MB * 2, "bf16"),
    "sum_i32_1core_1k": lambda: c_reduce_add(1024),
    # matrix-vector (dot products)
    "gemv_bf16_4core_256x1024": lambda: c_gemv_bf16(256, 1024, 4),
    "gemv_bf16_4core_3072x1024": lambda: c_gemv_bf16(3072, 1024, 4),
    "gemv_bf16_4core_4096x4096": lambda: c_gemv_bf16(4096, 4096, 4),
    "gemv_bf16_1core_3072x1024": lambda: c_gemv_bf16(3072, 1024, 1),
    "mv_i16_1core_288": lambda: c_mv_i16(288, 288),
    "mv_i16_1core_1024": lambda: c_mv_i16(1024, 1024),
    # matrix-matrix
    "gemm_bf16_1core_256": lambda: c_gemm(256, 256, 256, 64, 64, 64, "bf16", "bf16", False),
    "gemm_bf16_1core_512": lambda: c_gemm(512, 512, 512, 64, 64, 64, "bf16", "bf16", False),
    "gemm_bf16_wa_256": lambda: c_gemm(256, 256, 256, 32, 64, 64, "bf16", "bf16", True),
    "gemm_bf16_wa_1024": lambda: c_gemm(1024, 1024, 1024, 64, 64, 64, "bf16", "bf16", True),
    "gemm_bf16_wa_2048": lambda: c_gemm(2048, 2048, 2048, 64, 64, 64, "bf16", "bf16", True),
    "gemm_i8_wa_256": lambda: c_gemm(256, 256, 256, 32, 64, 64, "i8", "i32", True),
    "gemm_i8_wa_1024": lambda: c_gemm(1024, 1024, 1024, 64, 64, 64, "i8", "i32", True),
    "gemm_i8_wa_2048": lambda: c_gemm(2048, 2048, 2048, 64, 64, 64, "i8", "i32", True),
    "gemm_i16_wa_256": lambda: c_gemm(256, 256, 256, 32, 32, 32, "i16", "i32", True),
    "gemm_i16_wa_1024": lambda: c_gemm(1024, 1024, 1024, 32, 32, 32, "i16", "i32", True),
    "gemm_i16_wa_2048": lambda: c_gemm(2048, 2048, 2048, 32, 32, 32, "i16", "i32", True),
    # BF16 inputs with FP32 output: the accumulator stays FP32 across K tiles
    "gemm_bf16f32_1core_256": lambda: c_gemm(256, 256, 256, 64, 64, 32, "bf16", "f32", False),
    "gemm_bf16f32_1core_512": lambda: c_gemm(512, 512, 512, 64, 64, 32, "bf16", "f32", False),
    "gemm_bf16f32_wa_256": lambda: c_gemm(256, 256, 256, 32, 64, 64, "bf16", "f32", True),
    "gemm_bf16f32_wa_1024": lambda: c_gemm(1024, 1024, 1024, 64, 64, 32, "bf16", "f32", True),
    "gemm_bf16f32_wa_2048": lambda: c_gemm(2048, 2048, 2048, 64, 64, 32, "bf16", "f32", True),
    # int8 1x1 convolution
    "conv1x1_i8_8x8x64": lambda: c_conv(8, 8, 64, 64),        # wrong output, see 013 doc
    "conv1x1_i8_32x4x64": lambda: c_conv(32, 4, 64, 64),
    "conv1x1_i8_32x32x64": lambda: c_conv(32, 32, 64, 64),
    "conv1x1_i8_32x128x128": lambda: c_conv(32, 128, 128, 128),
}


def timed(fn, min_n=10, max_n=200, budget_s=1.0):
    for _ in range(3):
        fn()
    samples = []
    t_end = time.perf_counter() + budget_s
    while len(samples) < max_n and (len(samples) < min_n or time.perf_counter() < t_end):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    s = sorted(samples)
    return {"median_ms": 1e3 * statistics.median(s), "p10_ms": 1e3 * s[len(s) // 10],
            "p90_ms": 1e3 * s[(9 * len(s)) // 10], "n": len(s)}


def compile_case(name):
    c = CASES[name]()
    t = time.perf_counter()
    xclbin, insts = (Path(p) for p in c["design"].compile())
    check_arg_sizes(name, c)
    return c, xclbin, insts, time.perf_counter() - t


def check_arg_sizes(name, c):
    """Refuse to run if a host buffer is smaller or larger than the runtime
    sequence argument it is bound to (a too-small buffer lets the NPU DMA write
    past its end)."""
    exp = getattr(c["design"], "_expected_tensor_sizes", None)
    if not exp:
        raise RuntimeError(f"{name}: no runtime-sequence argument sizes to check against")
    got = [count for _dtype, _data, count in c["args"]]
    if list(exp[:len(got)]) != got or len(exp) != len(got):
        raise RuntimeError(f"{name}: buffer element counts {got} != design arguments {list(exp)}")


def run_case(name):
    import npu_direct as nd
    c, xclbin, insts, compile_s = compile_case(name)
    prog = nd.Program(xclbin, insts)
    bufs, srcs = [], []
    for dtype, data, count in c["args"]:
        b = nd.Buffer(count, dtype)
        if data is not None:
            b.write(np.asarray(data).reshape(-1))
        else:
            b.array[:] = 0
            b.to_device()
        bufs.append(b)
        srcs.append(None if data is None else np.ascontiguousarray(np.asarray(data).reshape(-1)))
    KEEP.extend([prog, bufs])
    outs = [i for i, s in enumerate(srcs) if s is None]

    prog(*bufs)
    result = c["check"]([bufs[i].from_device().copy() for i in outs])

    run = timed(lambda: prog(*bufs))

    def e2e():
        for b, s in zip(bufs, srcs):
            if s is not None:
                b.array[:] = s
                b.to_device()
        prog(*bufs)
        return [bufs[i].from_device().copy() for i in outs]

    end = timed(e2e)
    # re-check after all timed calls
    result_after = c["check"]([bufs[i].from_device().copy() for i in outs])
    w = c["work"]
    row = {"case": name, "source": c["source"], "ok": bool(result["ok"] and result_after["ok"]),
           "check": result, "run": run, "e2e": end, "work": w,
           "insts_bytes": prog.default.insts_bytes, "compile_or_cache_s": round(compile_s, 2),
           "xclbin": str(xclbin)}
    ms = run["median_ms"]
    if "bytes" in w:
        row["run_gbps"] = w["bytes"] / ms / 1e6
    if "macs" in w:
        row["run_gmacs"] = w["macs"] / ms / 1e6
    if "elems" in w:
        row["run_gelems"] = w["elems"] / ms / 1e6
    print(json.dumps(row), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")
    nd.finish(0 if row["ok"] else 2)


def main():
    a = sys.argv[1:]
    if not a or a[0] == "--list":
        print("\n".join(CASES))
        return
    if a[0] == "--compile":
        import npu_direct as nd
        names = a[1:] or list(CASES)
        for n in names:
            try:
                _c, x, i, s = compile_case(n)
                print(json.dumps({"case": n, "compiled": True, "s": round(s, 1),
                                  "arg_counts": list(_c["design"]._expected_tensor_sizes),
                                  "insts_bytes": Path(i).stat().st_size, "xclbin": str(x)}), flush=True)
            except Exception as e:  # report and continue
                import traceback
                traceback.print_exc()
                print(json.dumps({"case": n, "compiled": False, "error": repr(e)[:400]}), flush=True)
        nd.finish()
    run_case(a[0])


if __name__ == "__main__":
    main()
