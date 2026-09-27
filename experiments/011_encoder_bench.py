"""Encoder-only latency at fixed input lengths (warm, repeated calls, median).

  python experiments\\011_encoder_bench.py --provider cpu --encoder encoder-model.onnx --seconds 5 10 20 30
  python experiments\\011_encoder_bench.py --provider vitisai --encoder <static model> --seconds 10 --tag npu_x

The input is the log-mel features of a real clip (test.wav tiled to length),
so values are realistic. For static encoders pass one --seconds value that
matches the model's fixed frame count (frames = seconds * 100).
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_pipeline as pp  # noqa: E402

ts = importlib.import_module("011_transcribe_set")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=ts.PROVIDERS, default="cpu")
    ap.add_argument("--encoder", default="encoder-model.onnx")
    ap.add_argument("--seconds", type=float, nargs="+", default=[5, 10, 20, 30])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--static", action="store_true", help="model has fixed shape: feed exactly its frames")
    ap.add_argument("--tag", default="bench")
    args = ap.parse_args()

    import onnxruntime as ort
    if args.provider == "cuda":
        ort.preload_dlls()
    options = None
    if args.provider == "vitisai":
        options = ts.vitis_options(args.tag, pp.CACHE / "vitis")
    elif args.provider == "cuda":
        options = [{"cudnn_conv_algo_search": "HEURISTIC"}, {}]
    t = time.perf_counter()
    model = pp.Parakeet(args.encoder, ts.PROVIDERS[args.provider], options, threads=args.threads)
    print(f"load {time.perf_counter() - t:.1f} s, providers {model.enc.get_providers()}", flush=True)
    names = [i.name for i in model.enc.get_inputs()]

    base = pp.load_audio(pp.ROOT.parent / "parakeet-stt" / "test.wav")
    rows = []
    for sec in args.seconds:
        n = int(sec * pp.SR)
        audio = np.tile(base, n // len(base) + 1)[:n]
        feats, lens = model.features(audio)
        frames = int(round(sec * 100))
        feats = feats[:, :, :frames]
        if feats.shape[2] < frames:
            feats = np.pad(feats, ((0, 0), (0, 0), (0, frames - feats.shape[2])))
        feed = {"audio_signal": feats}
        if "length" in names:
            feed["length"] = np.array([frames], np.int64)
        for _ in range(3):
            model.enc.run(None, feed)
        times = []
        c = time.process_time()
        for _ in range(args.repeats):
            t = time.perf_counter()
            model.enc.run(None, feed)
            times.append(time.perf_counter() - t)
        busy = (time.process_time() - c) / sum(times)  # CPU cores kept busy by this process
        ms = 1000 * float(np.median(times))
        rows.append({"seconds": sec, "median_ms": ms, "min_ms": 1000 * min(times), "ms_per_audio_s": ms / sec,
                     "cpu_cores_busy": busy})
        print(f"{sec:5.1f} s audio: median {ms:8.1f} ms  min {1000 * min(times):8.1f} ms  "
              f"({ms / sec:6.1f} ms per audio second, {busy:.1f} CPU cores busy)", flush=True)
    out = pp.CACHE / "results" / f"bench_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"provider": args.provider, "encoder": args.encoder, "threads": args.threads,
                               "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
