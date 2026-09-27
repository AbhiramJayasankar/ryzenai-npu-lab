"""Hybrid NPU encoder latency and its breakdown at fixed input lengths.

  powershell -NoProfile -File scripts\\run_npu_timeout.ps1 -Script experiments\\011_hybrid_bench.py -Seconds 900

Same input as 011_encoder_bench.py (test.wav tiled to 10/20/30 s, features
from nemo128.onnx), so the numbers compare directly with the CPU and GPU
encoder benchmarks. Median of 10 warm calls, split into: ORT prefix
(subsampling), NPU GEMM wall time (submission + wait), BF16 staging + sync,
result readback, and the remaining CPU NumPy work. `cpu_cores_busy` is
process CPU time / wall time.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_hybrid as ph  # noqa: E402  (first: sets OPENBLAS_NUM_THREADS before NumPy loads BLAS)

import numpy as np  # noqa: E402
import parakeet_pipeline as pp  # noqa: E402

import npu_direct as nd  # noqa: E402


def main():
    seconds = [float(x) for x in sys.argv[1:]] or [10.0, 20.0, 30.0]
    weights = ph.NpuWeights()
    enc = ph.HybridEncoder(pp.CACHE / "models" / "encoder_dyn_prefix.onnx", weights)
    model = pp.Parakeet(enc)
    base = pp.load_audio(pp.ROOT.parent / "parakeet-stt" / "test.wav")
    print("ready", flush=True)
    results = []
    for sec in seconds:
        n = int(sec * pp.SR)
        feats, lens = model.features(np.tile(base, n // len(base) + 1)[:n])
        feed = {"audio_signal": feats, "length": lens}
        for _ in range(2):
            enc.run(None, feed)
        rows = []
        for _ in range(10):
            for m in enc.mms.values():
                m.reset()
            enc.timing = {"prefix": 0.0, "layers": 0.0}
            w, c = time.perf_counter(), time.process_time()
            enc.run(None, feed)
            wall, cpu = time.perf_counter() - w, time.process_time() - c
            mm = enc.last_mm
            other = wall - enc.timing["prefix"] - mm.npu_s - mm.stage_s - mm.read_s
            rows.append({"total_ms": 1000 * wall, "prefix_ms": 1000 * enc.timing["prefix"],
                         "npu_gemm_ms": 1000 * mm.npu_s, "stage_ms": 1000 * mm.stage_s,
                         "readback_ms": 1000 * mm.read_s, "cpu_numpy_ms": 1000 * other,
                         "cpu_cores_busy": cpu / wall})
        med = {k: round(float(np.median([r[k] for r in rows])), 2) for k in rows[0]}
        med.update({"seconds": sec, "gemm_rows": enc.last_mm.rows, "gemm_calls": enc.last_mm.calls,
                    "ms_per_audio_s": round(med["total_ms"] / sec, 1)})
        results.append(med)
        print(json.dumps(med), flush=True)
    (pp.CACHE / "results" / "hybrid_bench.json").write_text(json.dumps(results, indent=1))
    nd.finish()


if __name__ == "__main__":
    main()
