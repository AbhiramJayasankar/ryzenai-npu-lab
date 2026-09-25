# Experiments

[Experiment 001](001_quicktest.md) confirms that AMD's bundled quicktest model executes on the Phoenix/Hawk Point NPU. The [script](001_quicktest.py) reproduces the run.

[Experiment 002](002_webcam_yolov8.md) runs live YOLOv8 object detection on the webcam and compares NPU and CPU inference on the same frames. The [script](002_webcam_yolov8.py) supports a timed live preview and a repeatable benchmark.

[Experiment 003](003_model_compatibility.md) tests five more quantized vision models: classification, object detection, face detection, and segmentation. It records NPU offloading, CPU/NPU output agreement, and inference timings. A separate [preview script](003_segmentation_preview.py) compares segmentation masks on one real video frame.

[Experiment 004](004_why_models_run.md) explains ONNX operators, quantization, compilation, and CPU/NPU partitioning through a controlled ResNet50 test. Its [script](004_operator_assignment.py) shows how one Phoenix/Hawk Point provider setting changes the same model from CPU-only execution to 393 NPU-assigned nodes.

[Experiment 005](005_low_level_access.md) runs custom BF16 and INT16 kernels on the Phoenix NPU through the open IRON/XRT toolchain. It verifies correctness and compares warm calls for small and larger jobs with the [benchmark script](005_direct_kernel_benchmark.py).

For each experiment, record the model and source, software and driver versions, input shape, CPU/NPU settings, warm-up and measurement method, operator placement, raw results, and a short conclusion. Commit small result files and notes; keep downloaded models and datasets outside Git.
