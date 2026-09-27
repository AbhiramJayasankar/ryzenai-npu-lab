"""Transcribe the experiment-011 LibriSpeech subset with the hybrid NPU encoder (IRON env).

  powershell -NoProfile -File scripts\\run_npu_timeout.ps1 -Script experiments\\011_hybrid_transcribe.py -Seconds 3600

Same pipeline, clips, warm-up and summary as 011_transcribe_set.py; writes
cache/parakeet/results/npu_hybrid.json.
"""

import importlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_hybrid as ph  # noqa: E402
import parakeet_pipeline as pp  # noqa: E402

import npu_direct as nd  # noqa: E402

ts = importlib.import_module("011_transcribe_set")


def main():
    t = time.perf_counter()
    weights = ph.NpuWeights()
    enc = ph.HybridEncoder(pp.CACHE / "models" / "encoder_dyn_prefix.onnx", weights)
    model = pp.Parakeet(enc)
    load_s = time.perf_counter() - t
    print(f"loaded in {load_s:.1f} s", flush=True)
    items = json.loads((pp.CACHE / "eval.json").read_text())
    for x in items[:3]:
        model.transcribe(pp.load_audio(pp.ROOT / x["wav"]))
    results = []
    for i, x in enumerate(items):
        text, t = model.transcribe(pp.load_audio(pp.ROOT / x["wav"]))
        results.append({"id": x["id"], "seconds": x["seconds"], "ref": x["text"], "hyp": text, "t": t})
        if i % 50 == 0:
            print(f"{i:4d} {x['seconds']:5.1f}s enc {t['enc'] * 1000:7.1f} ms  {text[:70]}", flush=True)
    ref_items = json.loads((pp.CACHE / "results" / "cpu_fp32.json").read_text())["items"]
    summary = ts.summarize(results, ref_items)
    summary.update({"tag": "npu_hybrid", "provider": "NPU (IRON BF16 GEMM) + CPU NumPy",
                    "encoder": "hybrid", "load_s": load_s})
    (pp.CACHE / "results" / "npu_hybrid.json").write_text(json.dumps({"summary": summary, "items": results},
                                                                      indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    nd.finish()


if __name__ == "__main__":
    main()
