# Ryzen AI NPU Lab

Reproducible experiments on the NPU in an AMD Ryzen 9 8945HS laptop. This repository will contain our own scripts, measurements, and conclusions. AMD's [RyzenAI-SW examples](https://github.com/amd/RyzenAI-SW) are a reference, not part of this repository.

## Starting point

- Windows detects the `NPU Compute Accelerator Device` (`PCI\VEN_1022&DEV_1502`, Phoenix/Hawk Point family).
- Installed NPU driver at the start of this project: `32.0.203.280`.
- Windows 11 build `26200`, Visual Studio 2022, and Conda are present. Visual Studio includes CMake `3.31.6`, although `cmake` is not currently on the terminal PATH.
- The Ryzen AI software runtime and a dedicated Conda environment have not yet been set up for this project.
- No NPU performance or model compatibility results have been measured yet.

Run `scripts/check_npu.ps1` in PowerShell to check the device and driver again. The script only reads system information.

## Next milestone

Install a supported Ryzen AI release using [AMD's Windows instructions](https://ryzenai.docs.amd.com/en/1.7/inst.html). The installer download requires an AMD sign-in and license acceptance. Then confirm that a model actually runs on this Phoenix/Hawk Point NPU. AMD's guide calls for the `X1` target on this hardware. After that, measure a small model on CPU and NPU with the same inputs and record latency, throughput, and operator placement.

Experiment notes and measurements will live in [`experiments/`](experiments/README.md). Keep downloaded models, datasets, caches, and secrets out of Git.
