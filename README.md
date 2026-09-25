# Ryzen AI NPU Lab

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

Run `scripts/check_npu.ps1` in PowerShell to check the device and driver again. The script only reads system information.

On the setup laptop, use Miniforge's environment Python directly from PowerShell:

```powershell
& "$env:USERPROFILE\miniforge3\envs\ryzen-ai-1.7.0\python.exe" -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The laptop also has an older Conda-compatible installation, so an unqualified `conda` command may select that installation instead of Miniforge.

## Next milestone

Measure power use with a reliable sensor and evaluate task accuracy on labeled data. AMD's guide calls for the `X1` target on this hardware. Quicktest, live YOLOv8, and five additional quantized vision models have now confirmed NPU execution. Inference speed varies by model.

Experiment notes and measurements will live in [`experiments/`](experiments/README.md). Keep downloaded models, datasets, caches, and secrets out of Git.
