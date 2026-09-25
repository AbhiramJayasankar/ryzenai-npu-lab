# Experiments

[Experiment 001](001_quicktest.md) confirms that AMD's bundled quicktest model executes on the Phoenix/Hawk Point NPU. The [script](001_quicktest.py) reproduces the run.

[Experiment 002](002_webcam_yolov8.md) runs live YOLOv8 object detection on the webcam and compares NPU and CPU inference on the same frames. The [script](002_webcam_yolov8.py) supports a timed live preview and a repeatable benchmark.

For each experiment, record the model and source, software and driver versions, input shape, CPU/NPU settings, warm-up and measurement method, operator placement, raw results, and a short conclusion. Commit small result files and notes; keep downloaded models and datasets outside Git.
