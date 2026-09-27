# Experiments

[Experiment 001](001_quicktest.md) confirms that AMD's bundled quicktest model executes on the Phoenix/Hawk Point NPU. The [script](001_quicktest.py) reproduces the run.

[Experiment 002](002_webcam_yolov8.md) runs live YOLOv8 object detection on the webcam and compares NPU and CPU inference on the same frames. The [script](002_webcam_yolov8.py) supports a timed live preview and a repeatable benchmark.

[Experiment 003](003_model_compatibility.md) tests five more quantized vision models: classification, object detection, face detection, and segmentation. It records NPU offloading, CPU/NPU output agreement, and inference timings. A separate [preview script](003_segmentation_preview.py) compares segmentation masks on one real video frame.

[Experiment 004](004_why_models_run.md) explains ONNX operators, quantization, compilation, and CPU/NPU partitioning through a controlled ResNet50 test. Its [script](004_operator_assignment.py) shows how one Phoenix/Hawk Point provider setting changes the same model from CPU-only execution to 393 NPU-assigned nodes.

[Experiment 005](005_low_level_access.md) runs custom BF16 and INT16 kernels on the Phoenix NPU through the open IRON/XRT toolchain. It verifies correctness and compares warm calls for small and larger jobs with the [benchmark script](005_direct_kernel_benchmark.py).

[Experiment 006](006_lfm25_full_npu.md) tracks the all-NPU LFM2.5-230M implementation, CPU/GPU baselines, and verified custom model operations on the Phoenix NPU.
Its continuation guide for that path is the [experiment 006 handoff](006_handoff.md); the current engine is experiment 010.

[Experiment 007](007_lfm25_chat.md) packages the model into a short-context interactive NPU chat runner and tests new prompts against sequential CPU BF16 token selections.

[Experiment 008](008_long_context_limits.md) tests NPU-only chunked attention beyond the old 64/96-position limit and explains why a 4K-token chat remains too slow in the current implementation.

[Experiment 009](009_prefill.md) measures a 20-word prompt against batched CPU BF16 prefill, then tests a correct four-core batched NPU projection with weight reuse. It records the remaining whole-model prefill gap and the DMA sharing needed for further tile parallelism.

[Experiment 010](010_x8_engine.md) rebuilds the engine around the NPU's measured DDR bandwidth: eight cores with one DMA stream each, one core program for every layer and the vocabulary head, and one submission per token. Decode runs at 17.2 ms per token (at the ~27 GB/s ceiling) and prompts at 4.6 ms per token, matching the CPU BF16 reference; it includes the probes, validation scripts and a chat runner. Continue from the [experiment 010 handoff](010_handoff.md).

[Experiment 011](011_parakeet_npu.md) tests NVIDIA Parakeet TDT 0.6B v2 speech-to-text on the NPU. The Ryzen AI / Vitis AI route fails for this Conformer model on Phoenix: only XINT8 reaches the NPU, it splits every layer into about ten pieces, runs slower than the CPU and loses accuracy. A hybrid encoder with all linear layers as custom IRON BF16 GEMMs on the NPU matches the FP32 WER (2.19% on 210 LibriSpeech utterances) at CPU speed (520 ms per 10 s window) while keeping about 2 CPU cores busy instead of 8; the GPU is 20x faster.

[Experiment 012](012_all_npu_encoder.md) runs all 24 Conformer layers of the Parakeet encoder on the NPU with one custom 16-core IRON program, for utterances up to 30.7 s. On 210 LibriSpeech utterances the NPU's WER equals FP32 exactly (2.19%) at 185 ms per 10 s window, 2 to 2.7 times faster than the CPU (the RTX 4060 is 7 times faster still). Intermittent hangs, which caused two Windows crashes, were traced to shim task-queue overflow and fixed. See the [experiment 012 handoff](012_handoff.md) before running new NPU programs.

For each experiment, record the model and source, software and driver versions, input shape, CPU/NPU settings, warm-up and measurement method, operator placement, raw results, and a short conclusion. Commit small result files and notes; keep downloaded models and datasets outside Git.
