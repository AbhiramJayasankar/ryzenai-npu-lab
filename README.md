# Ryzen AI NPU Lab

Reproducible experiments on the NPU in an AMD Ryzen 9 8945HS laptop. This repository will contain our own scripts, measurements, and conclusions. AMD's [RyzenAI-SW examples](https://github.com/amd/RyzenAI-SW) are a reference, not part of this repository.

## Starting point

- Windows detects the `NPU Compute Accelerator Device` (`PCI\VEN_1022&DEV_1502`, Phoenix/Hawk Point family).
- Installed NPU driver at the start of this project: `32.0.203.280`.
- Windows 11 build `26200`, Visual Studio 2022, and Conda are present. Visual Studio includes CMake `3.31.6`, although `cmake` is not currently on the terminal PATH.
- AMD Ryzen AI Software `1.7.0` is installed at `C:\Program Files\RyzenAI\1.7.0`.
- Miniforge Conda `26.7.2` has a dedicated `ryzen-ai-1.7.0` environment with Python `3.12.11`, Ryzen AI `1.7.0`, and ONNX Runtime `1.23.2.dev20260117`.
- ONNX Runtime lists `VitisAIExecutionProvider`, `DmlExecutionProvider`, and `CPUExecutionProvider`.
- No NPU performance or model compatibility results have been measured yet.

Run `scripts/check_npu.ps1` in PowerShell to check the device and driver again. The script only reads system information.

On the setup laptop, use Miniforge's environment Python directly from PowerShell:

```powershell
& "$env:USERPROFILE\miniforge3\envs\ryzen-ai-1.7.0\python.exe" -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The laptop also has an older Conda-compatible installation, so an unqualified `conda` command may select that installation instead of Miniforge.

## Next milestone

Confirm that a model actually runs on this Phoenix/Hawk Point NPU using [AMD's Windows instructions](https://ryzenai.docs.amd.com/en/1.7/inst.html). AMD's guide calls for the `X1` target on this hardware. After that, measure a small model on CPU and NPU with the same inputs and record latency, throughput, and operator placement. Listing the Vitis AI provider confirms that the runtime is installed; it does not yet prove model execution on the NPU.

Experiment notes and measurements will live in [`experiments/`](experiments/README.md). Keep downloaded models, datasets, caches, and secrets out of Git.
