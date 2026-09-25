"""Run AMD's bundled quicktest model on a Phoenix/Hawk Point NPU.

Use the Python executable from the Ryzen AI 1.7.0 Miniforge environment.
The model and firmware stay in AMD's installation directory; the compiler
cache stays in this repository's ignored cache/ directory.
"""

import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


INSTALL_DIR = Path(
    os.environ.get("RYZEN_AI_INSTALLATION_PATH", r"C:\Program Files\RyzenAI\1.7.0")
)
MODEL = INSTALL_DIR / "quicktest" / "test_model.onnx"
XCLBIN = INSTALL_DIR / "voe-4.0-win_amd64" / "xclbins" / "phoenix" / "4x4.xclbin"
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "quicktest"


def main() -> None:
    if not MODEL.is_file() or not XCLBIN.is_file():
        raise FileNotFoundError(f"Missing AMD test model or NPU firmware: {MODEL}, {XCLBIN}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    options = ort.SessionOptions()
    options.log_severity_level = 1  # AMD's offloading report is logged at info level.
    provider_options = {
        "target": "X1",
        "xlnx_enable_py3_round": "0",
        "xclbin": str(XCLBIN),
        "cache_dir": str(CACHE_DIR),
        "cache_key": "amd_quicktest_phx_x1",
        "log_level": "info",
    }

    print(f"Model: {MODEL}", flush=True)
    print(f"Firmware: {XCLBIN}", flush=True)
    print("Requested provider: VitisAIExecutionProvider (target X1)", flush=True)
    start = time.perf_counter()
    session = ort.InferenceSession(
        str(MODEL),
        sess_options=options,
        providers=["VitisAIExecutionProvider"],
        provider_options=[provider_options],
    )
    print(f"Session providers: {session.get_providers()}", flush=True)
    print(f"Session creation: {time.perf_counter() - start:.3f} s", flush=True)

    image = np.random.default_rng(0).random((1, 3, 32, 32), dtype=np.float32)
    start = time.perf_counter()
    outputs = session.run(None, {"input": image})
    print(f"Inference: {time.perf_counter() - start:.3f} s", flush=True)
    print(f"Output shape: {outputs[0].shape}", flush=True)
    print(f"Output finite: {bool(np.isfinite(outputs[0]).all())}", flush=True)
    if outputs[0].shape != (1, 10) or not np.isfinite(outputs[0]).all():
        raise RuntimeError("AMD quicktest produced an invalid output")
    print("Test passed", flush=True)


if __name__ == "__main__":
    main()
