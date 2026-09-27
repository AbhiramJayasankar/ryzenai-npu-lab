"""Experiment 012: transcribe the utterances that did not fit one 10.24 s window
without chunking, with the 2-block (20.5 s) or 3-block (30.7 s) program.

  run_npu_timeout.ps1 -Script experiments/012_transcribe_long.py -ScriptArgs "2" -Seconds 3600
  run_npu_timeout.ps1 -Script experiments/012_transcribe_long.py -ScriptArgs "3" -Seconds 3600
  python experiments/012_transcribe_long.py --combine      (no NPU: WER tables)

nblk=2 takes every utterance that 012_transcribe.py had to chunk and whose
encoder frames fit 256; nblk=3 the rest (> 256 frames). Each utterance runs
once; any NPU error stops the script (no retries). Writes
cache/parakeet/results/npu_long_B<nblk>.json. --combine merges them with the
single-window results of npu_full.json into npu_whole.json: every utterance
transcribed whole on the NPU, compared with the CPU FP32 run of experiment 011.
"""

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_pipeline as pp  # noqa: E402

ts = importlib.import_module("011_transcribe_set")
RES = pp.CACHE / "results"


def combine():
    single = [r for r in json.loads((RES / "npu_full.json").read_text())["items"] if r["chunks"] == 1]
    long = [r for b in (2, 3) for r in json.loads((RES / f"npu_long_B{b}.json").read_text())["items"]]
    rows = single + long
    fp32 = json.loads((RES / "cpu_fp32.json").read_text())["items"]
    summ = ts.summarize([{**r, "t": {k: r["t"][k] for k in ("pre", "enc", "dec", "total")}} for r in rows],
                        fp32)
    ref = {r["id"]: r for r in fp32}
    for label, sub in (("one_block", single), ("two_block", [r for r in long if r["nblk"] == 2]),
                       ("three_block", [r for r in long if r["nblk"] == 3])):
        w, e, n = pp.wer([r["ref"] for r in sub], [r["hyp"] for r in sub])
        wf, ef, _ = pp.wer([r["ref"] for r in sub], [ref[r["id"]]["hyp"] for r in sub])
        summ[label] = {"utterances": len(sub), "wer_npu": w, "errors_npu": e, "wer_cpu_fp32": wf,
                       "errors_cpu_fp32": ef, "words": n,
                       "npu_ms_median": 1000 * float(np.median([r["t"]["npu"] for r in sub]))}
    (RES / "npu_whole.json").write_text(json.dumps({"summary": summ, "items": rows}, indent=1))
    print(json.dumps(summ, indent=1))


def main():
    if sys.argv[1:] == ["--combine"]:
        return combine()
    import onnxruntime as ort

    import npu_direct as nd
    import pk_model as pm

    nblk = int(sys.argv[1])
    so = ort.SessionOptions()
    so.log_severity_level = 3
    prefix = ort.InferenceSession(str(pp.CACHE / "models" / "encoder_dyn_prefix.onnx"), so,
                                  providers=["CPUExecutionProvider"])
    model = pp.Parakeet("encoder-model.int8.onnx")  # preprocessor and decoder only
    chunked = {r["id"] for r in json.loads((RES / "npu_full.json").read_text())["items"] if r["chunks"] > 1}
    todo = []
    for x in json.loads((pp.CACHE / "eval.json").read_text()):
        if x["id"] not in chunked:
            continue
        feats, lens = model.features(pp.load_audio(pp.ROOT / x["wav"]))
        hidden = prefix.run(["hidden"], {"audio_signal": feats, "length": lens})[0][0]
        n = hidden.shape[0]
        if (nblk == 2 and n <= 256) or (nblk == 3 and n > 256):
            todo.append((x, feats, lens, hidden))
    print(f"{len(todo)} utterances for the {nblk}-block program", flush=True)
    t = time.perf_counter()
    enc = pm.PkEncoder(nblk=nblk)
    print(f"NPU encoder ready in {time.perf_counter() - t:.1f} s", flush=True)
    rows = []
    for i, (x, feats, lens, hidden) in enumerate(todo):
        t0 = time.perf_counter()
        out = enc.run(hidden, hidden.shape[0])
        t1 = time.perf_counter()
        text = model.decode(out.T[None], np.array([hidden.shape[0]]))
        t2 = time.perf_counter()
        rows.append({"id": x["id"], "seconds": x["seconds"], "frames": hidden.shape[0], "nblk": nblk,
                     "chunks": 1, "ref": x["text"], "hyp": text,
                     "t": {"pre": 0.0, "enc": t1 - t0, "npu": enc.npu_s, "dec": t2 - t1, "total": t2 - t0}})
        print(f"{i:3d} {x['seconds']:5.1f}s {hidden.shape[0]} frames npu {1000 * enc.npu_s:6.1f} ms  "
              f"{text[:60]}", flush=True)
    (RES / f"npu_long_B{nblk}.json").write_text(json.dumps({"items": rows}, indent=1))
    nd.finish()


if __name__ == "__main__":
    main()
