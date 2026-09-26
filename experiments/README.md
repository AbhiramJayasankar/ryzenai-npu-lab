# Experiments

[Experiment 001](001_quicktest.md) confirms that AMD's bundled quicktest model executes on the Phoenix/Hawk Point NPU. The [script](001_quicktest.py) reproduces the run.

[Experiment 002](002_webcam_yolov8.md) runs live YOLOv8 object detection on the webcam and compares NPU and CPU inference on the same frames. The [script](002_webcam_yolov8.py) supports a timed live preview and a repeatable benchmark.

[Experiment 003](003_model_compatibility.md) tests five more quantized vision models: classification, object detection, face detection, and segmentation. It records NPU offloading, CPU/NPU output agreement, and inference timings. A separate [preview script](003_segmentation_preview.py) compares segmentation masks on one real video frame.

[Experiment 004](004_why_models_run.md) explains ONNX operators, quantization, compilation, and CPU/NPU partitioning through a controlled ResNet50 test. Its [script](004_operator_assignment.py) shows how one Phoenix/Hawk Point provider setting changes the same model from CPU-only execution to 393 NPU-assigned nodes.

[Experiment 005](005_low_level_access.md) runs custom BF16 and INT16 kernels on the Phoenix NPU through the open IRON/XRT toolchain. It verifies correctness and compares warm calls for small and larger jobs with the [benchmark script](005_direct_kernel_benchmark.py).

[Experiment 006](006_lfm25_full_npu.md) tracks the all-NPU LFM2.5-230M implementation, CPU/GPU baselines, and verified custom model operations on the Phoenix NPU.
For the present state and a self-contained continuation guide, start with the [experiment 006 handoff](006_handoff.md).

[Experiment 007](007_lfm25_chat.md) packages the model into a short-context interactive NPU chat runner and tests new prompts against sequential CPU BF16 token selections.

[Experiment 008](008_long_context_limits.md) tests NPU-only chunked attention beyond the old 64/96-position limit and explains why a 4K-token chat remains too slow in the current implementation.

For each experiment, record the model and source, software and driver versions, input shape, CPU/NPU settings, warm-up and measurement method, operator placement, raw results, and a short conclusion. Commit small result files and notes; keep downloaded models and datasets outside Git.
