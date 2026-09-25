"""YOLOv8 object detection on the 8945HS NPU, with a CPU comparison.

The INT8 model and COCO labels belong in models/yolov8/ (ignored by Git).
Camera frames remain in memory. Benchmark mode writes timing data only.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "yolov8" / "DetectionModel_int.onnx"
LABELS = ROOT / "models" / "yolov8" / "coco.names"
FIRMWARE = Path(r"C:\Program Files\RyzenAI\1.7.0\voe-4.0-win_amd64\xclbins\phoenix\4x4.xclbin")
IMAGE_SIZE = 640


def make_session(provider):
    options = ort.SessionOptions()
    options.log_severity_level = 1 if provider == "npu" else 3
    if provider == "npu":
        ep = "VitisAIExecutionProvider"
        ep_options = {
            "target": "X1",
            "xlnx_enable_py3_round": "0",
            "xclbin": str(FIRMWARE),
            "cache_dir": str(ROOT / "cache" / "yolov8"),
            "cache_key": "yolov8_phx_x1",
        }
        (ROOT / "cache" / "yolov8").mkdir(parents=True, exist_ok=True)
    else:
        ep = "CPUExecutionProvider"
        ep_options = {}
    start = time.perf_counter()
    session = ort.InferenceSession(
        str(MODEL), providers=[ep], provider_options=[ep_options], sess_options=options
    )
    print(f"{provider.upper()} session: {time.perf_counter() - start:.2f} s", flush=True)
    print(f"{provider.upper()} providers: {session.get_providers()}", flush=True)
    return session


def prepare(frame):
    height, width = frame.shape[:2]
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    target_width = round(width * scale)
    target_height = round(height * scale)
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    left = (IMAGE_SIZE - target_width) // 2
    top = (IMAGE_SIZE - target_height) // 2
    padded = cv2.copyMakeBorder(
        resized,
        top,
        IMAGE_SIZE - target_height - top,
        left,
        IMAGE_SIZE - target_width - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, scale, left, top


def infer(session, tensor):
    start = time.perf_counter()
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    return outputs, (time.perf_counter() - start) * 1000


def detect(outputs, frame_shape, scale, left, top, confidence=0.35):
    predictions = outputs[0][0].T  # 8400 rows: cx, cy, w, h, then 80 class scores.
    class_ids = predictions[:, 4:].argmax(axis=1)
    scores = predictions[np.arange(len(predictions)), class_ids + 4]
    keep = np.flatnonzero(scores >= confidence)
    if not len(keep):
        return []
    selected = predictions[keep, :4]
    x = selected[:, 0] - selected[:, 2] / 2
    y = selected[:, 1] - selected[:, 3] / 2
    boxes = np.column_stack((x, y, selected[:, 2], selected[:, 3]))
    chosen = cv2.dnn.NMSBoxes(boxes.tolist(), scores[keep].tolist(), confidence, 0.45)
    detections = []
    height, width = frame_shape[:2]
    for index in np.asarray(chosen).reshape(-1):
        cx, cy, bw, bh = selected[index]
        x1 = int(np.clip((cx - bw / 2 - left) / scale, 0, width - 1))
        y1 = int(np.clip((cy - bh / 2 - top) / scale, 0, height - 1))
        x2 = int(np.clip((cx + bw / 2 - left) / scale, 0, width - 1))
        y2 = int(np.clip((cy + bh / 2 - top) / scale, 0, height - 1))
        detections.append((x1, y1, x2, y2, int(class_ids[keep[index]]), float(scores[keep[index]])))
    return detections


def annotate(frame, detections, labels):
    result = frame.copy()
    for x1, y1, x2, y2, class_id, score in detections:
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 220, 50), 2)
        label = labels[class_id] if class_id < len(labels) else str(class_id)
        cv2.putText(result, f"{label} {score:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 50), 2)
    return result


def open_camera(index):
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return capture


def percentile90(values):
    return float(np.percentile(values, 90))


def benchmark(args):
    capture = open_camera(args.camera)
    frames = []
    try:
        for _ in range(args.frames):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped providing frames")
            frames.append(frame)
    finally:
        capture.release()
    tensors = [prepare(frame)[0] for frame in frames]
    print(f"Captured {len(frames)} camera frames in memory; none were saved.", flush=True)

    sessions = {provider: make_session(provider) for provider in ("npu", "cpu")}
    results = {}
    for provider, session in sessions.items():
        for i in range(args.warmup):
            infer(session, tensors[i % len(tensors)])
        elapsed = [infer(session, tensor)[1] for tensor in tensors]
        results[provider] = {
            "median_inference_ms": round(median(elapsed), 3),
            "p90_inference_ms": round(percentile90(elapsed), 3),
            "inference_only_fps": round(1000 / median(elapsed), 2),
            "samples_ms": [round(value, 3) for value in elapsed],
        }
        print(
            f"{provider.upper()}: median {results[provider]['median_inference_ms']:.3f} ms, "
            f"p90 {results[provider]['p90_inference_ms']:.3f} ms, "
            f"inference-only {results[provider]['inference_only_fps']:.2f} fps",
            flush=True,
        )

    sample, scale, left, top = prepare(frames[0])
    for provider, session in sessions.items():
        outputs, _ = infer(session, sample)
        detections = detect(outputs, frames[0].shape, scale, left, top)
        print(f"{provider.upper()} detections on first frame: {len(detections)}", flush=True)

    report = {
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "frame_count": len(frames),
        "warmup_count": args.warmup,
        "camera_resolution": list(frames[0].shape[:2][::-1]),
        "input_resolution": [IMAGE_SIZE, IMAGE_SIZE],
        "measure": "session.run only; camera capture and image processing excluded",
        "results": results,
    }
    report_path = ROOT / "cache" / "yolov8_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Timing details: {report_path}", flush=True)


def live(args):
    labels = LABELS.read_text(encoding="utf-8").splitlines()
    session = make_session("npu")
    capture = open_camera(args.camera)
    deadline = time.monotonic() + args.seconds
    count = 0
    try:
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped providing frames")
            tensor, scale, left, top = prepare(frame)
            outputs, latency_ms = infer(session, tensor)
            detections = detect(outputs, frame.shape, scale, left, top)
            annotated = annotate(frame, detections, labels)
            cv2.putText(annotated, f"NPU {latency_ms:.1f} ms  |  Q to quit", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 50), 2)
            cv2.imshow("Ryzen AI NPU object detection", annotated)
            count += 1
            if count % 30 == 0:
                print(f"Displayed {count} frames; latest inference {latency_ms:.1f} ms; "
                      f"{len(detections)} detections", flush=True)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    print(f"Live demo ended after {count} frames.", flush=True)


def sample(args):
    labels = LABELS.read_text(encoding="utf-8").splitlines()
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise FileNotFoundError(args.image)
    tensor, scale, left, top = prepare(frame)
    session = make_session(args.provider)
    outputs, latency_ms = infer(session, tensor)
    detections = detect(outputs, frame.shape, scale, left, top)
    for _, _, _, _, class_id, score in detections:
        print(f"{labels[class_id]}: {score:.3f}", flush=True)
    output_path = ROOT / "cache" / "yolov8_sample_annotated.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotate(frame, detections, labels))
    print(f"{len(detections)} detections; inference {latency_ms:.2f} ms; image {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "live", "sample"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--provider", choices=("cpu", "npu"), default="npu")
    args = parser.parse_args()
    if not MODEL.is_file() or not LABELS.is_file() or not FIRMWARE.is_file():
        raise FileNotFoundError("Place AMD's DetectionModel_int.onnx and coco.names in models/yolov8; "
                                "Ryzen AI 1.7.0 firmware must also be installed.")
    if args.frames < 1 or args.warmup < 0 or args.seconds < 1:
        parser.error("frames and seconds must be positive; warmup cannot be negative")
    if args.mode == "sample" and args.image is None:
        parser.error("sample mode requires --image")
    if args.mode == "benchmark":
        benchmark(args)
    elif args.mode == "live":
        live(args)
    else:
        sample(args)


if __name__ == "__main__":
    main()
