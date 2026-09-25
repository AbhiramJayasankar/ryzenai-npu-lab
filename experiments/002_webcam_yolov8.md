# Experiment 002: live YOLOv8 detection on NPU and CPU

Date: 2026-09-25

## Model and setup

- Hardware: Ryzen 9 8945HS (Hawk Point), `PCI\VEN_1022&DEV_1502`, NPU driver `32.0.203.280`.
- Runtime: Ryzen AI `1.7.0`, Python `3.12.11`, ONNX Runtime Vitis AI `1.23.2.dev20260117`.
- Camera: `USB2.0 FHD UVC WebCam`, captured at 640 × 480.
- Model: AMD's INT8 YOLOv8 `DetectionModel_int.onnx` from the [RyzenAI-SW YOLOv8 tutorial](https://github.com/amd/RyzenAI-SW/tree/bb02ad48b6ed5c304ffd1ddb005d2fed5d4a6046/tutorial/yolov8). SHA-256: `1F65C211A5F147E7B95D33D2B68334854E01C5E125B4111FE3A99248827C4EA7`. The original tutorial targets Ryzen AI 1.2; this run verifies the model also works with the installed 1.7 runtime.
- NPU options: `target=X1`, Phoenix `4x4.xclbin`, `xlnx_enable_py3_round=0`.
- Model input: letterboxed RGB float32 image, 640 × 640. Detection boxes and non-maximum suppression run on the CPU after inference.

The 104 MB model and COCO labels are excluded from Git. On this laptop, `scripts/setup_yolo_model.ps1` copies them from the existing AMD checkout into `models/yolov8/` and verifies the model hash. Pass `-AmdCheckout` if the checkout is elsewhere.

## Run

From the lab repo root in PowerShell:

```powershell
.\scripts\setup_yolo_model.ps1
& "$env:USERPROFILE\miniforge3\envs\ryzen-ai-1.7.0\python.exe" .\experiments\002_webcam_yolov8.py benchmark --frames 30 --warmup 5
& "$env:USERPROFILE\miniforge3\envs\ryzen-ai-1.7.0\python.exe" .\experiments\002_webcam_yolov8.py live --seconds 20
```

The live window displays labeled boxes. Press **Q** to close it early. Webcam frames remain in memory and are not saved. The benchmark saves timing data, without frames, to ignored `cache/yolov8_benchmark.json`.

## Results

The model compiled and ran on the NPU. AMD's runtime reported 1,281 operators on one NPU subgraph, 16 on the Vitis AI CPU path, and `Actually running on NPU 1`. On AMD's sample image it detected three people and two ties; the boxes visually aligned with those objects.

Thirty webcam frames were captured and preprocessed once. Both providers received the **same input tensors**, with five warm-up runs each. Timing covers only `session.run`; camera capture, resizing, box decoding, drawing, and session creation are excluded.

| Provider | Median inference | 90th percentile | Inference-only rate |
| --- | ---: | ---: | ---: |
| NPU | 35.117 ms | 35.615 ms | 28.48 frames/s |
| CPU | 263.963 ms | 273.172 ms | 3.79 frames/s |

For this model and setup, the NPU's median inference was approximately **7.5× faster** than the CPU's. The 20-second NPU live preview displayed 409 annotated frames, approximately 20 frames/s end to end. That preview included capture, preprocessing, inference, decoding, drawing, and display.

NPU session creation, including compilation, took approximately 35–42 seconds across runs; CPU session creation took 0.30 seconds in the benchmark. The NPU figures therefore describe repeated inference after compilation, not the first result after launch.

Power use was **not measured**. This laptop exposed no usable NPU power reading: `xrt-smi` was unavailable, and Windows' Power Meter counter returned zero. A power-efficiency claim would need a reliable external meter or another validated energy sensor.
