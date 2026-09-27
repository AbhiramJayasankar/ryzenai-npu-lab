"""Experiment 012: WER and speed of the all-NPU encoder (1-block program, <= 128
encoder frames = 10.24 s) on the 210-utterance LibriSpeech subset.

Utterances longer than the program are cut into equal chunks of at most 10 s
of audio, each transcribed on its own; the chunk texts are joined. The same
chunking is run with the ONNX Runtime FP32 encoder on the CPU, so the effect
of chunking and the effect of the NPU can be told apart. Runs every utterance
once, in one process: if the NPU ever hangs the script stops (no retries).

  powershell -NoProfile -File scripts\\run_npu_timeout.ps1 -Script experiments\\012_transcribe.py -Seconds 3600

Writes cache/parakeet/results/npu_full.json and cpu_fp32_chunked.json.
"""

import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

import npu_direct as nd
import pk_model as pm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_pipeline as pp  # noqa: E402

ts = importlib.import_module("011_transcribe_set")
MAX_S = 10.0  # chunk length limit (the program holds 10.24 s)


def chunks(audio):
    n = math.ceil(len(audio) / (MAX_S * pp.SR))
    step = math.ceil(len(audio) / n)
    return [audio[i:i + step] for i in range(0, len(audio), step)]


def main():
    so = ort.SessionOptions()
    so.log_severity_level = 3
    prefix = ort.InferenceSession(str(pp.CACHE / "models" / "encoder_dyn_prefix.onnx"), so,
                                  providers=["CPUExecutionProvider"])
    cpu = pp.Parakeet("encoder-model.onnx")  # FP32 encoder on the CPU; its preprocessor/decoder serve both
    t = time.perf_counter()
    enc = pm.PkEncoder(nblk=1)
    load_s = time.perf_counter() - t
    print(f"NPU encoder ready in {load_s:.1f} s", flush=True)

    def npu_text(audio):
        t0 = time.perf_counter()
        feats, lens = cpu.features(audio)
        t1 = time.perf_counter()
        hidden = prefix.run(["hidden"], {"audio_signal": feats, "length": lens})[0][0]
        c0 = time.process_time()
        x = enc.run(hidden, hidden.shape[0])
        cpu_busy = time.process_time() - c0
        t2 = time.perf_counter()
        text = cpu.decode(x.T[None], np.array([hidden.shape[0]]))
        t3 = time.perf_counter()
        return text, {"pre": t1 - t0, "enc": t2 - t1, "npu": enc.npu_s, "dec": t3 - t2,
                      "total": t3 - t0, "enc_cpu_s": cpu_busy}

    items = json.loads((pp.CACHE / "eval.json").read_text())
    npu_text(pp.load_audio(pp.ROOT / items[0]["wav"]))  # warm-up
    npu_rows, cpu_rows = [], []
    for i, x in enumerate(items):
        audio = pp.load_audio(pp.ROOT / x["wav"])
        parts = chunks(audio)
        texts, tim = [], {}
        for p in parts:
            text, tt = npu_text(p)
            texts.append(text)
            for k, v in tt.items():
                tim[k] = tim.get(k, 0.0) + v
        npu_rows.append({"id": x["id"], "seconds": x["seconds"], "chunks": len(parts), "ref": x["text"],
                         "hyp": " ".join(texts), "t": tim})
        if len(parts) > 1:  # CPU FP32 with the same chunking
            ctexts, ctim = [], {}
            for p in parts:
                text, tt = cpu.transcribe(p)
                ctexts.append(text)
                for k, v in tt.items():
                    ctim[k] = ctim.get(k, 0.0) + v
            cpu_rows.append({"id": x["id"], "seconds": x["seconds"], "chunks": len(parts), "ref": x["text"],
                             "hyp": " ".join(ctexts), "t": ctim})
        if i % 25 == 0:
            print(f"{i:4d} {x['seconds']:5.1f}s {len(parts)} chunk(s) npu {1000 * tim['npu']:6.1f} ms  "
                  f"{npu_rows[-1]['hyp'][:60]}", flush=True)

    fp32 = {r["id"]: r for r in json.loads((pp.CACHE / "results" / "cpu_fp32.json").read_text())["items"]}
    out = {}
    for name, rows in (("npu_full", npu_rows), ("cpu_fp32_chunked", cpu_rows)):
        summ = ts.summarize(rows, list(fp32.values()))
        summ.update({"tag": name, "load_s": load_s if name == "npu_full" else None})
        if name == "npu_full":
            summ["npu_s"] = sum(r["t"]["npu"] for r in rows)
            summ["enc_cpu_s"] = sum(r["t"]["enc_cpu_s"] for r in rows)
            short = [r for r in rows if r["chunks"] == 1]
            long = [r for r in rows if r["chunks"] > 1]
            for label, sub in (("single_window", short), ("chunked", long)):
                w, e, n = pp.wer([r["ref"] for r in sub], [r["hyp"] for r in sub])
                wf, ef, nf = pp.wer([r["ref"] for r in sub], [fp32[r["id"]]["hyp"] for r in sub])
                summ[label] = {"utterances": len(sub), "wer_npu": w, "errors_npu": e,
                               "wer_cpu_fp32_unchunked": wf, "errors_cpu_fp32_unchunked": ef, "words": n}
            summ["npu_ms_per_window_median"] = 1000 * float(np.median(
                [r["t"]["npu"] for r in short]))
        (pp.CACHE / "results" / f"{name}.json").write_text(json.dumps({"summary": summ, "items": rows},
                                                                      indent=1))
        out[name] = summ
    print(json.dumps(out, indent=1), flush=True)
    nd.finish()


if __name__ == "__main__":
    main()
