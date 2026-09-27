"""Transcribe the experiment-011 LibriSpeech subset with one encoder/provider; report WER and speed.

Examples (repository root):
  # GPU and CPU baselines, parakeet-stt venv
  ..\\parakeet-stt\\.venv\\Scripts\\python experiments\\011_transcribe_set.py --provider cuda --tag gpu_fp32
  ..\\parakeet-stt\\.venv\\Scripts\\python experiments\\011_transcribe_set.py --provider cpu --tag cpu_fp32
  ..\\parakeet-stt\\.venv\\Scripts\\python experiments\\011_transcribe_set.py --provider cpu --encoder encoder-model.int8.onnx --tag cpu_int8dyn
  # NPU, ryzen-ai-1.7.0 env
  python experiments\\011_transcribe_set.py --provider vitisai --encoder <static qdq model> --fixed-frames 2000 --tag npu_...

Writes cache/parakeet/results/<tag>.json (per-utterance text and stage timings)
and prints corpus WER against the LibriSpeech references and, when
cache/parakeet/results/cpu_fp32.json exists, word disagreement with the FP32
CPU transcripts.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_pipeline as pp  # noqa: E402

PROVIDERS = {"cpu": ["CPUExecutionProvider"], "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
             "vitisai": ["VitisAIExecutionProvider", "CPUExecutionProvider"],
             "dml": ["DmlExecutionProvider", "CPUExecutionProvider"]}
PHOENIX_XCLBIN = r"C:\Program Files\RyzenAI\1.7.0\voe-4.0-win_amd64\xclbins\phoenix\4x4.xclbin"


def vitis_options(tag, cache_dir):
    return [{"target": "X1", "xclbin": PHOENIX_XCLBIN, "xlnx_enable_py3_round": "0",
             "enable_cache_file_io_in_mem": "0", "cache_dir": str(cache_dir), "cache_key": tag}, {}]


def summarize(items, ref_items=None):
    wer, err, words = pp.wer([x["ref"] for x in items], [x["hyp"] for x in items])
    audio = sum(x["seconds"] for x in items)
    out = {"utterances": len(items), "audio_s": audio, "wer": wer, "errors": err, "words": words}
    for stage in ("pre", "enc", "dec", "total"):
        out[f"{stage}_s"] = sum(x["t"][stage] for x in items)
    out["rtfx"] = audio / out["total_s"]
    out["enc_rtfx"] = audio / out["enc_s"]
    if ref_items:
        ref = {x["id"]: x["hyp"] for x in ref_items}
        common = [x for x in items if x["id"] in ref]
        d, _, n = pp.wer([ref[x["id"]] for x in common], [x["hyp"] for x in common])
        out["word_diff_vs_cpu_fp32"] = d
        out["identical_vs_cpu_fp32"] = sum(ref[x["id"]] == x["hyp"] for x in common)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=PROVIDERS, default="cpu")
    ap.add_argument("--encoder", default="encoder-model.onnx")
    ap.add_argument("--fixed-frames", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--set", default="eval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    import onnxruntime as ort
    if args.provider == "cuda":
        ort.preload_dlls()
    ort.set_default_logger_severity(3)
    options = None
    if args.provider == "vitisai":
        options = vitis_options(args.tag, pp.CACHE / "vitis")
    elif args.provider == "cuda":  # every new input length would otherwise rerun cuDNN's exhaustive search
        options = [{"cudnn_conv_algo_search": "HEURISTIC"}, {}]

    t = time.perf_counter()
    model = pp.Parakeet(args.encoder, PROVIDERS[args.provider], options, fixed_frames=args.fixed_frames,
                        threads=args.threads)
    load_s = time.perf_counter() - t
    print(f"loaded in {load_s:.1f} s, encoder providers {model.enc.get_providers()}", flush=True)

    items = json.loads((pp.CACHE / f"{args.set}.json").read_text())[: args.limit]
    for x in items[:3]:  # warm-up
        model.transcribe(pp.load_audio(pp.ROOT / x["wav"]))
    results = []
    for i, x in enumerate(items):
        text, t = model.transcribe(pp.load_audio(pp.ROOT / x["wav"]))
        results.append({"id": x["id"], "seconds": x["seconds"], "ref": x["text"], "hyp": text, "t": t})
        if i % 50 == 0:
            print(f"{i:4d} {x['seconds']:5.1f}s enc {t['enc'] * 1000:7.1f} ms  {text[:70]}", flush=True)

    ref_path = pp.CACHE / "results" / "cpu_fp32.json"
    ref_items = json.loads(ref_path.read_text())["items"] if ref_path.exists() and args.tag != "cpu_fp32" else None
    summary = summarize(results, ref_items)
    summary.update({"tag": args.tag, "provider": args.provider, "encoder": args.encoder,
                    "fixed_frames": args.fixed_frames, "threads": args.threads, "load_s": load_s,
                    "encoder_providers": model.enc.get_providers()})
    (pp.CACHE / "results").mkdir(parents=True, exist_ok=True)
    (pp.CACHE / "results" / f"{args.tag}.json").write_text(json.dumps({"summary": summary, "items": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
