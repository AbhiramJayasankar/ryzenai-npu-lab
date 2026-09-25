# Experiment 001: AMD quicktest on the 8945HS NPU

Date: 2026-09-25

## Setup

- Hardware: Ryzen 9 8945HS, Phoenix/Hawk Point NPU (`PCI\VEN_1022&DEV_1502`).
- NPU driver: `32.0.203.280`.
- Software: Ryzen AI `1.7.0`, Miniforge environment `ryzen-ai-1.7.0`, Python `3.12.11`, ONNX Runtime `1.23.2.dev20260117`.
- Model: AMD's bundled `quicktest/test_model.onnx`, SHA-256 `7A78C7E85BAC3A0681E3D2F77E69093A761E1C8F62F7FC65FF5CE3E1B82C5BA3`. The model stays in AMD's installation directory.
- Input: seeded random float32 tensor, shape `(1, 3, 32, 32)`.
- Vitis AI provider options: `target=X1`, Phoenix `4x4.xclbin`, `xlnx_enable_py3_round=0` per [AMD's 1.7.0 installation test](https://ryzenai.docs.amd.com/en/1.7/inst.html#npu-offloading-with-session-options).

## Run

From the repository root in PowerShell:

```powershell
& "$env:USERPROFILE\miniforge3\envs\ryzen-ai-1.7.0\python.exe" .\experiments\001_quicktest.py
```

## Result

The script exited successfully. The runtime reported:

```text
[Vitis AI EP] No. of Operators :
   NPU   398
   VITIS_EP_CPU   2
[Vitis AI EP] No. of Subgraphs :
   NPU   1
Actually running on NPU   1
Session providers: ['VitisAIExecutionProvider', 'CPUExecutionProvider']
Output shape: (1, 10)
Output finite: True
Test passed
```

Session creation took 13.212 s, including initial compilation. One inference took 0.006 s. These are single-run observations and are **not** benchmark results. This test establishes that the bundled model executes on the NPU, with two operators handled by the Vitis AI CPU path. The next experiment should use repeated runs and a CPU baseline to measure latency and throughput.
