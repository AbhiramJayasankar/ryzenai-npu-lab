# Experiment 005: Run a custom program directly through XRT

## Question

Can the Ryzen 9 8945HS's Phoenix NPU run kernels we define ourselves, without the Ryzen AI ONNX model compiler? And is a larger custom computation worthwhile after dispatch overhead?

**Yes for the two kernels tested here.** AMD's open [IRON / MLIR-AIE](https://github.com/Xilinx/mlir-aie) toolchain compiled the kernels to `final.xclbin` plus `insts.bin`, and XRT ran them on `NPU1`. This uses XRT and the NPU driver; it does not bypass the driver or firmware.

## Machine and isolated setup

- Hardware: Phoenix NPU, PCI `1022:1502`; XRT reported `NPU Phoenix`, five columns, and `NPU1`.
- Existing driver `32.0.203.280`, firmware `1.5.5.391`, XRT `2.19.0`; Windows 11 build `26200`.
- Python `3.13.5` in a separate IRON virtual environment; Visual Studio 2022 Developer PowerShell.
- IRON source release `v1.4.3`, commit `95b3d1ccc0bfe5183bae1fa9014cfdc1fb4d96c8`, matching `mlir_aie 1.4.3` wheel; `llvm-aie 22.0.0.2026090701+3e93bf7b` pinned by that release.
- XRT Windows SDK release `2.21.75`, ZIP SHA-256 `ccc244c2c423588972ade76142cdc01049477aaa39a35be97e782b97eb7c5295`; IRON wheel SHA-256 `c2f580931b653f1e03c0286d744bcc19726a9422548c8b1e78ab20148b84742d`. Both checksums matched their GitHub release asset digests.
- The toolchain and SDK were placed under ignored `cache/iron/`. The working Ryzen AI 1.7 environment and NPU driver were not replaced.

The [current native Windows guide](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWinNative.md) recommends a newer XRT/driver than this laptop currently has. The specific IRON 1.4.3 examples below nevertheless compiled and ran successfully here. That observation does not establish compatibility for every IRON feature or future release.

## Proof of NPU execution

AMD's `programming_examples/getting_started/01_SAXPY/saxpy.py` compiled and printed `PASS!`. Its inputs were `XRTTensor`, the selected device was `NPU1`, `pyxrt.device(0)` opened, and IRON emitted `final.xclbin` and `insts.bin`. The kernel computes `Z = 3X + Y` over 4,096 BF16 elements using one AI Engine tile and matched the CPU reference.

AMD's `03_matrix_multiplication_single_core` example AOT-compiled two INT16 matrix sizes, `256×256` and `512×512`, emitted separate binaries, ran both on the NPU, and printed `PASS` for each. It uses one NPU core and moves tiles of the matrices through the NPU memory hierarchy.

These results show that a custom BF16 arithmetic kernel can run on this Phoenix NPU even though AMD's [Ryzen AI 1.7 model compatibility table](https://ryzenai.docs.amd.com/en/1.7/relnotes.html) does not list the **BF16 ONNX model flow** for Phoenix/Hawk Point. A working custom kernel does not mean an arbitrary BF16 model will compile.

## Exploratory timing

[`005_direct_kernel_benchmark.py`](005_direct_kernel_benchmark.py) reuses the example kernels and allocated tensors, warms up, times the NPU call, then times NumPy on the CPU with the same input data. Compilation, initial tensor allocation, and final output copy are outside the timed calls. The NPU call includes the Python wrapper, XRT dispatch, in-kernel data movement, and completion. The CPU matrix output is preallocated. Correctness was checked after timing.

| Work | Warm NPU call median | NumPy CPU median | Correct? | Timed calls |
| --- | ---: | ---: | --- | ---: |
| SAXPY, 4,096 BF16 values | 0.758 ms | 0.018 ms | Yes | 100 |
| 256×256 INT16 matrix multiplication, one NPU core | 1.681 ms | 9.909 ms | Yes | 20 |
| 512×512 INT16 matrix multiplication, one NPU core | 5.470 ms | 134.316 ms | Yes | 20 |

The tiny SAXPY job is far faster on the CPU; fixed call overhead is large relative to its arithmetic. We have not separated the Python wrapper, XRT dispatch, and data-movement costs. The larger INT16 matrix jobs are faster on the NPU than **this NumPy INT16 implementation**. NumPy INT16 matrix multiplication is not necessarily the fastest CPU implementation, so these ratios must not be presented as a general CPU-versus-NPU speedup. Power was not measured; faster latency does not by itself prove lower energy use.

The built-in `xrt-smi validate --run latency --verbose` passed and reported an average of `90.3 µs` for its own 20-byte instruction sequence over 10,000 iterations. Its GEMM throughput test was **skipped** on Phoenix, so it did not establish peak compute throughput. `xrt-smi examine` showed the NPU back in Default power mode after validation.

## Reproduce on this laptop

The ignored `cache/iron/mlir-aie` checkout already contains the IRON source and separate environment. Open PowerShell from the repo root:

```powershell
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64
Set-Location cache\iron\mlir-aie
. .\iron_env.ps1
& .\ironenv\Scripts\python.exe 'C:\Users\abhir\Desktop\projects\ryzenai-npu-lab\experiments\005_direct_kernel_benchmark.py'
```

For a fresh installation, follow IRON's [native Windows setup guide](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWinNative.md), but pin source and wheel to the same release. The rolling wheel feed supplied 1.3.4 during this experiment while the source checkout was newer, causing an API mismatch; pinning both to 1.4.3 resolved it. The Windows setup also needed `ironenv/Lib/site-packages/mlir_aie/bin/llvm-objcopy.exe` on `PATH` when preparing the Peano wheel. The official source and SDK binaries remain outside Git.

## Next performance questions

1. Try a multi-core matrix kernel and compare it against a well-optimized CPU baseline at the same numeric precision and output semantics.
2. Separate compilation, allocation, host-to-NPU transfer, dispatch, computation, and output-copy time.
3. Measure power with a reliable sensor before claiming energy efficiency.
4. Test whether a useful operator from a vision or text model can be expressed as a custom kernel, then assess whole-pipeline latency rather than kernel time alone.
