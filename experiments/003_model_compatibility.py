"""Check whether an INT8 ONNX model executes on this NPU and compare CPU output.

Inputs are deterministic synthetic tensors. This checks execution and numerical
agreement, not model accuracy on a real dataset. Read the Vitis AI info log to
confirm that at least one subgraph was actually assigned to the NPU.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import median

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = Path(r"C:\Program Files\RyzenAI\1.7.0\voe-4.0-win_amd64\xclbins\phoenix\4x4.xclbin")


def create_session(model, provider):
    options = ort.SessionOptions()
    options.log_severity_level = 1 if provider == "npu" else 3
    if provider == "npu":
        ep = "VitisAIExecutionProvider"
        provider_options = {
            "target": "X1",
            "xlnx_enable_py3_round": "0",
            "xclbin": str(FIRMWARE),
        }
    else:
        ep = "CPUExecutionProvider"
        provider_options = {}
    start = time.perf_counter()
    session = ort.InferenceSession(
        str(model), providers=[ep], provider_options=[provider_options], sess_options=options
    )
    seconds = time.perf_counter() - start
    print(f"{provider.upper()} session created in {seconds:.3f} s; providers={session.get_providers()}", flush=True)
    return session, seconds


def make_inputs(session):
    rng = np.random.default_rng(0)
    tensors = {}
    for meta in session.get_inputs():
        shape = tuple(dim if isinstance(dim, int) and dim > 0 else 1 for dim in meta.shape)
        if meta.type != "tensor(float)":
            raise ValueError(f"Unsupported test input type for {meta.name}: {meta.type}")
        tensors[meta.name] = rng.random(shape, dtype=np.float32)
        print(f"Input {meta.name}: {shape}, float32", flush=True)
    return tensors


def run(session, tensors, count):
    for _ in range(2):
        session.run(None, tensors)
    times = []
    for _ in range(count):
        start = time.perf_counter()
        outputs = session.run(None, tensors)
        times.append((time.perf_counter() - start) * 1000)
    return outputs, times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--runs", type=int, default=8)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    model = args.model.resolve(strict=True)
    if not FIRMWARE.is_file():
        raise FileNotFoundError(FIRMWARE)

    cpu, cpu_create = create_session(model, "cpu")
    tensors = make_inputs(cpu)
    cpu_outputs, cpu_times = run(cpu, tensors, args.runs)
    print(f"CPU median inference: {median(cpu_times):.3f} ms", flush=True)

    npu, npu_create = create_session(model, "npu")
    npu_outputs, npu_times = run(npu, tensors, args.runs)
    print(f"NPU median inference: {median(npu_times):.3f} ms", flush=True)

    comparisons = []
    for cpu_output, npu_output in zip(cpu_outputs, npu_outputs):
        a, b = cpu_output.astype(np.float64).ravel(), npu_output.astype(np.float64).ravel()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cosine = float(np.dot(a, b) / denom) if denom else None
        comparison = {
            "shape": list(cpu_output.shape),
            "cpu_finite": bool(np.isfinite(cpu_output).all()),
            "npu_finite": bool(np.isfinite(npu_output).all()),
            "cosine_similarity": cosine,
            "max_absolute_difference": float(np.max(np.abs(a - b))),
        }
        comparisons.append(comparison)
        print(f"Output comparison: {comparison}", flush=True)
    if not all(c["cpu_finite"] and c["npu_finite"] for c in comparisons):
        raise RuntimeError("A model output contains non-finite values")

    report = {
        "model": model.name,
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "runs": args.runs,
        "input": {name: list(value.shape) for name, value in tensors.items()},
        "cpu_create_s": round(cpu_create, 3),
        "npu_create_s": round(npu_create, 3),
        "cpu_median_ms": round(median(cpu_times), 3),
        "npu_median_ms": round(median(npu_times), 3),
        "outputs": comparisons,
        "note": "Synthetic inputs; execution and output comparison only, not task accuracy",
    }
    path = ROOT / "cache" / "model_compatibility" / f"{model.stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {path}", flush=True)


if __name__ == "__main__":
    main()
