"""Compare CPU and NPU segmentation on a frame from AMD's demo video.

The color overlay shows predicted class IDs only. It is not a labeled accuracy
evaluation; this simple RGB/255 preprocessing is an approximation of AMD's
closed demo pipeline.
"""

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "amd_multi_model" / "pointpainting-nus-FPN_int.onnx"
VIDEO = ROOT / "models" / "amd_multi_model" / "seg_512_288.avi"
FIRMWARE = Path(r"C:\Program Files\RyzenAI\1.7.0\voe-4.0-win_amd64\xclbins\phoenix\4x4.xclbin")


def infer(provider, tensor):
    options = ort.SessionOptions()
    options.log_severity_level = 1 if provider == "VitisAIExecutionProvider" else 3
    provider_options = ({
        "target": "X1",
        "xlnx_enable_py3_round": "0",
        "xclbin": str(FIRMWARE),
    } if provider == "VitisAIExecutionProvider" else {})
    session = ort.InferenceSession(
        str(MODEL), providers=[provider], provider_options=[provider_options], sess_options=options
    )
    start = perf_counter()
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    print(f"{provider} inference: {(perf_counter() - start) * 1000:.3f} ms", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=120)
    args = parser.parse_args()
    if not MODEL.is_file() or not VIDEO.is_file() or not FIRMWARE.is_file():
        raise FileNotFoundError("Run scripts/setup_multi_models.ps1 and install Ryzen AI 1.7.0 first")

    capture = cv2.VideoCapture(str(VIDEO))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Could not read video frame {args.frame}")
    image = cv2.resize(frame, (576, 320))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    cpu = infer("CPUExecutionProvider", tensor)
    npu = infer("VitisAIExecutionProvider", tensor)
    cpu_mask = cpu.argmax(axis=1)[0]
    npu_mask = npu.argmax(axis=1)[0]
    print(f"Output shape: {npu.shape}; finite: {bool(np.isfinite(npu).all())}", flush=True)
    print(f"CPU/NPU pixel agreement: {np.mean(cpu_mask == npu_mask):.4%}", flush=True)
    print(f"NPU class IDs and pixel counts: {dict(zip(*np.unique(npu_mask, return_counts=True)))}", flush=True)

    palette = np.random.default_rng(5).integers(40, 256, size=(11, 3), dtype=np.uint8)
    overlay = cv2.addWeighted(image, 0.58, palette[npu_mask], 0.42, 0)
    path = ROOT / "cache" / "segmentation_preview.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate((image, overlay), axis=1))
    print(f"Preview: {path}", flush=True)


if __name__ == "__main__":
    main()
