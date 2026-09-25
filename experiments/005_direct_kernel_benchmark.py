"""Run AMD IRON example kernels on the NPU and compare with NumPy CPU calls.

Run inside the separate Python 3.13 IRON environment, after activating its
iron_env.ps1 and Visual Studio Developer PowerShell. See 005_low_level_access.md.
This is a small exploratory benchmark, not a hardware throughput or power test.
"""

import argparse
import sys
import time
from pathlib import Path
from statistics import median

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "cache" / "iron" / "mlir-aie"


def milliseconds(call, runs):
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000)
    return median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir-aie-root", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    examples = args.mlir_aie_root.resolve(strict=True) / "programming_examples" / "getting_started"
    sys.path.insert(0, str(examples / "01_SAXPY"))
    sys.path.insert(0, str(examples / "03_matrix_multiplication_single_core"))
    from saxpy import saxpy
    from matrix_multiplication_single_core import matrix_multiplication_single_core

    x = iron.arange(4096, dtype=bfloat16, device="npu")
    y = iron.arange(4096, dtype=bfloat16, device="npu")
    z = iron.zeros_like(x)
    if type(iron.get_current_device()).__name__ != "NPU1" or type(x).__name__ != "XRTTensor":
        raise RuntimeError("IRON did not select this laptop's NPU1 through XRT")
    saxpy_call = lambda: saxpy(x, y, z, N=4096, element_type=bfloat16)
    for _ in range(5):
        saxpy_call()
    npu_ms = milliseconds(saxpy_call, 100)
    host_x, host_y = x.numpy(), y.numpy()
    cpu_ms = milliseconds(lambda: 3 * host_x + host_y, 100)
    print(
        f"SAXPY 4096 BF16: NPU {npu_ms:.3f} ms, NumPy CPU {cpu_ms:.3f} ms, "
        f"equal={np.array_equal(z.numpy(), 3 * host_x + host_y)}",
        flush=True,
    )

    for size in (256, 512):
        a = iron.randint(0, 256, (size, size), dtype=np.int16, device="npu")
        b = iron.randint(0, 256, (size, size), dtype=np.int16, device="npu")
        c = iron.zeros(size * size, dtype=np.int16, device="npu")
        host_a, host_b = a.numpy(), b.numpy()
        cpu_output = np.empty((size, size), dtype=np.int16)
        npu_call = lambda: matrix_multiplication_single_core(
            a, b, c, M=size, K=size, N=size, element_type=np.int16
        )
        for _ in range(3):
            npu_call()
        npu_ms = milliseconds(npu_call, 20)
        cpu_ms = milliseconds(lambda: np.matmul(host_a, host_b, out=cpu_output), 20)
        print(
            f"Matmul {size}x{size} int16, one NPU core: NPU {npu_ms:.3f} ms, "
            f"NumPy CPU {cpu_ms:.3f} ms, "
            f"equal={np.array_equal(c.numpy().reshape(size, size), cpu_output)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
