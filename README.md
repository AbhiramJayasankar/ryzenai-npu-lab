# Ryzen AI NPU Lab: run an LLM and custom kernels on the AMD XDNA NPU

Hands-on, reproducible experiments for the **AMD Ryzen AI NPU** (XDNA,
AIE-ML tiles) on Windows 11: a small language model running **entirely on the
NPU**, custom BF16/INT16 kernels written with AMD's open **IRON / MLIR-AIE**
toolchain and **XRT**, and vision models (YOLOv8, ResNet50, MobileNetV2 and
more) through **Ryzen AI Software** and ONNX Runtime's Vitis AI execution
provider. Every result is measured against the CPU (and the GPU where
relevant), with scripts to reproduce it.

**Tested on:** Ryzen 9 8945HS ("Hawk Point") laptop, NPU driver
32.0.203.280, Ryzen AI Software 1.7.0, IRON/MLIR-AIE 1.4.3. The same
first-generation XDNA NPU ("NPU1", Phoenix) is in Ryzen 7040 and 8040
series mobile processors with Ryzen AI and in Ryzen 8000G desktop APUs, so the
code should apply there; Ryzen AI 300 "Strix" (XDNA 2 / NPU2) has a different
array and would need changes.

**What you can learn or run here**

* Chat with **Liquid AI LFM2.5-230M** on the NPU at **58 tokens/s**, with all
  model arithmetic on the NPU in BF16 ([experiment 010](experiments/010_x8_engine.md)).
* How the NPU actually performs for LLM inference: DDR bandwidth (~27 GB/s
  measured), program-memory and DMA limits, NPU vs CPU vs GPU, and why an
  optimized CPU runtime (llama.cpp) is still faster for decode.
* Writing and running custom AI Engine kernels through IRON/XRT, bypassing
  the Ryzen AI ONNX compiler ([experiment 005](experiments/005_low_level_access.md)).
* Speech-to-text with **NVIDIA Parakeet TDT 0.6B v2** on the NPU: why the
  Ryzen AI route fails for a Conformer on Phoenix, and a hybrid encoder with
  custom IRON GEMMs that matches the FP32 WER at CPU speed with far less CPU
  load ([experiment 011](experiments/011_parakeet_npu.md)); the whole encoder on
  the NPU at 185 ms per 10 s of audio, 2-2.7x the CPU with the FP32 WER
  ([experiment 012](experiments/012_all_npu_encoder.md)).
* Checking whether an ONNX model really runs on the NPU or falls back to the
  CPU ([experiment 004](experiments/004_why_models_run.md)), plus a live
  YOLOv8 webcam NPU vs CPU benchmark ([experiment 002](experiments/002_webcam_yolov8.md)).

**Current result:** The [eight-core LFM2.5-230M engine](experiments/010_x8_engine.md)
runs every model calculation on the Phoenix NPU in BF16 at **17.2 ms per
generated token (58 tokens/s)**, ~26× faster than this lab's earlier NPU
path. It beats PyTorch BF16 on 8 CPU threads (~33 tokens/s) but not an
optimized CPU runtime: llama.cpp on the same CPU decodes this model at 91
tokens/s in F16 and 160 tokens/s in Q8_0.
Prompts run four tokens per weight pass at 4.6 ms per token; a 161-token
prompt reaches its first reply token in 0.82 s. Decode is at the NPU's
measured ~27 GB/s DDR read ceiling. Outputs match the CPU BF16 reference
token for token except at exact BF16 ties. Chat with
`& .\scripts\chat_lfm25_x8.ps1` from this repository's root (4096-position
context). To continue the work, start with the
[experiment 010 handoff](experiments/010_handoff.md).

The earlier [short-context chat runner](experiments/007_lfm25_chat.md) and
the [experiment 006 handoff](experiments/006_handoff.md) record the first
all-NPU proof, setup, speed, and power comparison.
An [experimental chunked context path](experiments/008_long_context_limits.md)
crosses the earlier 64/96-token ceiling, but measured latency makes long
NPU-only chats impractical for now.
The [short-prompt prefill experiment](experiments/009_prefill.md) measures an
optimized CPU baseline and a four-core batched NPU projection. The projection
improves with weight reuse, but whole-model NPU prefill has not beaten CPU.

Reproducible experiments on the NPU in an AMD Ryzen 9 8945HS laptop. This repository will contain our own scripts, measurements, and conclusions. AMD's [RyzenAI-SW examples](https://github.com/amd/RyzenAI-SW) are a reference, not part of this repository.

## Starting point

- Windows detects the `NPU Compute Accelerator Device` (`PCI\VEN_1022&DEV_1502`, Phoenix/Hawk Point family).
- Installed NPU driver at the start of this project: `32.0.203.280`.
- Windows 11 build `26200`, Visual Studio 2022, and Conda are present. Visual Studio includes CMake `3.31.6`, although `cmake` is not currently on the terminal PATH.
- AMD Ryzen AI Software `1.7.0` is installed at `C:\Program Files\RyzenAI\1.7.0`.
- Miniforge Conda `26.7.2` has a dedicated `ryzen-ai-1.7.0` environment with Python `3.12.11`, Ryzen AI `1.7.0`, and ONNX Runtime `1.23.2.dev20260117`.
- ONNX Runtime lists `VitisAIExecutionProvider`, `DmlExecutionProvider`, and `CPUExecutionProvider`.
- AMD's bundled quicktest model ran successfully on the NPU; see [experiment 001](experiments/001_quicktest.md).
- A [live YOLOv8 webcam experiment](experiments/002_webcam_yolov8.md) now measures NPU and CPU inference on the same frames. Power use still needs a reliable sensor.
- [Five more AMD demo models](experiments/003_model_compatibility.md) ran on the NPU: MobileNetV2, ResNet50, nano-YOLOX, RetinaFace, and PointPainting segmentation. The NPU was faster for four in short same-input comparisons; RetinaFace was slightly slower.
- [Experiment 004](experiments/004_why_models_run.md) is a hands-on guide to model operators, quantization, and CPU/NPU assignment. It demonstrates why listing an NPU execution provider does not prove that the NPU performed inference.
- [Experiment 005](experiments/005_low_level_access.md) confirms that custom BF16 and INT16 programs can run directly through IRON/XRT on this Phoenix NPU. Its exploratory matrix benchmark shows the benefit of larger jobs and the overhead of tiny ones.
- [Experiment 006](experiments/006_lfm25_full_npu.md) runs LFM2.5-230M model arithmetic entirely on the NPU for a fixed 21-token prompt and two generated tokens. NPU token selections match the CPU reference. Battery measurements find lower incremental whole-laptop watts for NPU, but far higher energy per pass because this prototype is much slower than CPU/GPU.

Run `scripts/check_npu.ps1` in PowerShell to check the device and driver again. The script only reads system information.

On the setup laptop, use Miniforge's environment Python directly from PowerShell:

```powershell
& "$env:USERPROFILE\miniforge3\envs\ryzen-ai-1.7.0\python.exe" -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The laptop also has an older Conda-compatible installation, so an unqualified `conda` command may select that installation instead of Miniforge.

## Where to continue

For speech-to-text, [experiment 011](experiments/011_parakeet_npu.md) ends
with measured limits and estimates for an all-NPU Parakeet encoder.

For the LFM2.5 engine, start with the
[experiment 010 handoff](experiments/010_handoff.md): design, hardware
budgets, debugging workflow and ranked next steps (INT8 weights, attention
bookkeeping, longer context, power). The earlier vision experiments remain
available for model compatibility and operator placement study.

Experiment notes and measurements will live in [`experiments/`](experiments/README.md). Keep downloaded models, datasets, caches, and secrets out of Git.

## Keywords

AMD NPU, Ryzen AI NPU, XDNA, XDNA NPU LLM, AMD Phoenix NPU, Hawk Point NPU,
Ryzen 7040 NPU, Ryzen 8040 NPU, Ryzen 8000G NPU, Ryzen 9 8945HS, Ryzen 7
8840HS, Ryzen 7 7840HS, AI Engine, AIE-ML, AIE2, NPU1, IRON, MLIR-AIE,
Peano, llvm-aie, XRT, pyxrt, Ryzen AI Software, Vitis AI execution provider,
ONNX Runtime NPU, run LLM on AMD NPU, local LLM on Ryzen AI, NPU inference
Windows 11, LFM2, LFM2.5, Liquid AI, small language model on NPU, BF16 on
NPU, NPU vs CPU vs GPU benchmark, NPU memory bandwidth, custom NPU kernels,
YOLOv8 on NPU, Parakeet on NPU, speech-to-text on AMD NPU, ASR on Ryzen AI,
Conformer encoder on NPU.
