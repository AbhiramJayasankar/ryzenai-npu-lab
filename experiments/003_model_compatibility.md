# Experiment 003: Which vision models actually run on this NPU?

## Scope and source

Hardware: Ryzen 9 8945HS (Phoenix/Hawk Point NPU), NPU driver `32.0.203.280`, Ryzen AI Software `1.7.0`, Windows 11 build `26200`. ONNX Runtime in the `ryzen-ai-1.7.0` Conda environment used `VitisAIExecutionProvider`, target `X1`, and the Phoenix `4x4.xclbin` firmware. CPU comparisons used `CPUExecutionProvider` on the same model and input.

The five models below came from AMD's [multi-model demo](https://github.com/amd/RyzenAI-SW/tree/bb02ad48b6ed5c304ffd1ddb005d2fed5d4a6046/demo/multi_model) [resource archive](https://www.xilinx.com/bin/public/openDownload?filename=resource_multi_model_demo.zip). Tested ZIP SHA-256: `A84DF575902C9F29B7A04E1A5B7B40D0E85393114912932D0CB59777476104CB`. Model files and video are kept out of Git.

## Method

[`003_model_compatibility.py`](003_model_compatibility.py) creates one deterministic float32 tensor per model input (random values in `[0, 1)`). It warms each session twice, then measures eight inference calls. The table reports the median of those calls; session creation and preprocessing are excluded. The script checks that outputs are finite and compares CPU/NPU output shape and cosine similarity. Vitis AI's compilation log confirmed **one subgraph actually running on the NPU for each of the five models**. Its operator counts are compilation statistics, not a percentage of wall time.

| Model and task | Input shape | NPU ops / Vitis CPU ops | CPU median | NPU median | CPU ÷ NPU | Lowest output cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV2, classification | `1×224×224×3` | 660 / 3 | 12.874 ms | 3.324 ms | 3.87× | 0.9967 |
| ResNet50, classification | `1×3×224×224` | 393 / 2 | 36.881 ms | 6.008 ms | 6.14× | 0.9911 |
| nano-YOLOX, object detection | `1×3×416×416` | 723 / 4 | 10.873 ms | 7.468 ms | 1.46× | 0.9970 |
| RetinaFace, face detection | `1×3×320×320` | 356 / 4 | 4.189 ms | 4.767 ms | 0.88× | 0.9581 |
| PointPainting FPN, segmentation | `1×3×320×576` | 283 / 2 | 51.539 ms | 8.126 ms | 6.34× | 0.9926 |

All outputs had the expected shapes and finite values. MobileNetV2 also had five plain CPU operators in Vitis AI's report, in addition to its three `VITIS_EP_CPU` operators. For the other four models the report listed only the Vitis CPU operators shown. **RetinaFace ran on the NPU but was slightly slower than CPU in this short test.** NPU execution does not guarantee a speedup.

An additional older quantized ResNet50 already present in the original `onnx-benchmark` checkout also ran: 396 NPU operators, two Vitis CPU operators, one NPU subgraph; CPU 33.827 ms, NPU 5.755 ms over six measured calls, output cosine 0.9830. Its source file was `RyzenAI-SW/onnx-benchmark/models/resnet50/resnet50_fp32_XINT8.onnx` at AMD checkout commit `bb02ad48b6ed5c304ffd1ddb005d2fed5d4a6046`. This was a separate preliminary check, so its timing should not be treated as part of the uniform five-model sweep.

## Real video frame check

[`003_segmentation_preview.py`](003_segmentation_preview.py) ran the PointPainting segmentation model on frame 120 of AMD's `seg_512_288.avi`. With simple RGB/255 preprocessing, the NPU output was finite, and the CPU and NPU agreed on the highest-scoring class at **99.2589% of pixels**. The generated side-by-side preview is in ignored `cache/segmentation_preview.jpg`; the NPU mask visibly covers the parked cars. This establishes that the two providers behave similarly on one real frame. It does **not** establish semantic accuracy against ground-truth labels, because the preprocessing is only an approximation of AMD's demo pipeline and we did not evaluate a labeled dataset. The one-frame timings were 45.959 ms CPU and 12.107 ms NPU, excluding session creation and preprocessing; use the eight-call table above for a more stable timing comparison.

## Reproduce

Download the [AMD resource archive](https://www.xilinx.com/bin/public/openDownload?filename=resource_multi_model_demo.zip) to `cache/resource_multi_model_demo.zip`. It is about 1.33 GB. From the repository root in PowerShell:

```powershell
.\scripts\setup_multi_models.ps1
$python = "$env:USERPROFILE\miniforge3\envs\ryzen-ai-1.7.0\python.exe"
Get-ChildItem models\amd_multi_model\*.onnx | ForEach-Object {
    & $python experiments\003_model_compatibility.py $_.FullName
}
& $python experiments\003_segmentation_preview.py
```

The setup script verifies the archive checksum and extracts only the five models and short segmentation video. The Python script writes per-model JSON under ignored `cache/model_compatibility/`. To inspect NPU assignment, capture the Vitis AI info log and look for `Actually running on NPU 1`; a successful provider name alone is not proof of offloading.

## Conclusion and limits

This laptop ran quantized CNN models for image classification, object detection, face detection, and segmentation on the NPU. The prior [YOLOv8 webcam test](002_webcam_yolov8.md) also showed object detection on actual camera frames. This test set does not establish that every INT8 ONNX model is compatible, nor does it establish model accuracy on task datasets. Performance depends heavily on the model: four of the five AMD demo models were faster on the NPU in these conditions, while RetinaFace was slightly slower. Power draw was not measured.
