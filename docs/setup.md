# Setup

How the test laptop is configured, and what you need to rerun the
experiments. Nothing here is installed by the repository: models, toolchains,
virtual environments and compiled NPU programs live in the ignored `cache/`
folder and in `~/.npu/cache`.

## Hardware

| Component | Value |
| --- | --- |
| Laptop CPU | AMD Ryzen 9 8945HS ("Hawk Point"), 8 cores / 16 threads |
| NPU | first-generation XDNA ("NPU1", Phoenix), `PCI\VEN_1022&DEV_1502`, five columns; the engines here use 4 columns × 4 compute tiles |
| GPU (for comparisons) | NVIDIA RTX 4060 Laptop, 8 GB |
| RAM | 16 GB |

The same NPU is in Ryzen 7040 and 8040 series mobile processors with Ryzen AI
and in Ryzen 8000G desktop APUs. Ryzen AI 300 ("Strix", XDNA 2) has a
different array, so the engines here would need changes to run there.

## Software

| Component | Version | Used for |
| --- | --- | --- |
| Windows 11 | build 26200 | |
| NPU driver / firmware | 32.0.203.280 at the start, 32.0.203.376 from experiment 012 / 1.5.5.391 | everything on the NPU |
| Ryzen AI Software | 1.7.0 in `C:\Program Files\RyzenAI\1.7.0`, Miniforge env `ryzen-ai-1.7.0` (Python 3.12) | experiments 001–004 and the 011 Vitis AI tests |
| IRON / MLIR-AIE | 1.4.3, source commit `95b3d1cc`, with the matching `mlir_aie` wheel and its pinned `llvm-aie` (Peano) | custom kernels, experiments 005–012 |
| XRT Windows SDK | 2.21.75 | submitting programs to the NPU (`pyxrt`) |
| Visual Studio 2022 | Community, C++ workload | IRON's host tools |
| Reference runtime | `cache/lfm-env`: PyTorch 2.8.0, Transformers 5.5.4 | CPU/GPU reference outputs for the LLM |
| ONNX Runtime | 1.30 (CPU in the IRON venv; `onnxruntime-gpu` for the GPU baseline) | Parakeet front end, decoder and baselines |

Exact checksums of the IRON wheel and XRT SDK are in
[experiment 005](../experiments/005_low_level_access.md).

## Installing IRON on Windows

Follow IRON's [native Windows setup guide](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWinNative.md)
into `cache/iron/mlir-aie`, with two changes learned the hard way:

* Pin the source checkout and the wheel to the same release (1.4.3). The
  rolling wheel feed served 1.3.4 while the source was newer, and the APIs
  did not match.
* Put `ironenv/Lib/site-packages/mlir_aie/bin/llvm-objcopy.exe` on `PATH`
  while preparing the Peano wheel.

Then run any IRON script through the wrapper, which loads the Visual Studio
developer shell and `iron_env.ps1` first:

```powershell
powershell -NoProfile -File .\scripts\iron_python.ps1 experiments\010_x8_validate.py --check
```

Start each run in a fresh PowerShell process: the developer shell appends to
`PATH` every time it loads and eventually fails with "The input line is too
long". Do not name a PowerShell script parameter `$Prompt`, because the
venv's `Activate.ps1` overwrites it.

## Models

* **LFM2.5-230M:** [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)
  at revision `40cb2ad3`, in `cache/lfm25-230m/`. Details and the SHA-256 are
  in the [experiment 006 handoff](../experiments/006_handoff.md).
* **Parakeet TDT 0.6B v2:** the ONNX export used by onnx-asr, prepared with
  `011_export_weights.py`, `011_export_pos.py` and `011_split.py`; see
  [experiment 011](../experiments/011_parakeet_npu.md).

## Checking the NPU

`scripts/check_npu.ps1` prints the NPU device and driver version (it only
reads system information). To list ONNX Runtime's execution providers in the
Ryzen AI environment:

```powershell
& "$env:USERPROFILE\miniforge3\envs\ryzen-ai-1.7.0\python.exe" -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

It should list `VitisAIExecutionProvider`, `DmlExecutionProvider` and
`CPUExecutionProvider`. The laptop also has an older Conda installation, so
call the Miniforge environment's Python by its full path, not through an
unqualified `conda`.

## Safety: NPU hangs

On this driver an NPU program that hangs makes Windows reset the device, and
twice that ended in a blue screen. Before running a new or changed NPU
program, check it offline (for the Parakeet encoder,
`experiments/012_schedule_check.py`) and read the
[experiment 012 handoff](../experiments/012_handoff.md). Reboot after a hang
reset before running anything else: until then, creating a hardware context
fails with `0xc01e0009`.
