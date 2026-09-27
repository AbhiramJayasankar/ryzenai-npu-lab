"""Compile one quantized encoder block with the Vitis AI EP; report NPU/CPU operator
assignment, output error against the FP32 block on CPU, and warm latency.

  python experiments\\011_npu_block.py --qmodel cache\\parakeet\\models\\encoder_T1000_0_1_XINT8.onnx ^
      --fp32 cache\\parakeet\\models\\encoder_T1000_0_1.onnx --upstream cache\\parakeet\\models\\encoder_T1000_prefix.onnx

The input is test.wav (7.8 s, zero-padded to the block's frame count) pushed
through the FP32 upstream models. Also times the quantized model on the CPU
EP (QDQ emulation) and the FP32 block on the CPU for reference.
"""

import argparse
import importlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_pipeline as pp  # noqa: E402

ts = importlib.import_module("011_transcribe_set")
REPORT = "vitisai_ep_report.json"


def block_inputs(upstream, frames, wav):
    pre = pp.session(pp.MODEL_DIR / "nemo128.onnx", ["CPUExecutionProvider"])
    audio = pp.load_audio(wav)
    feats, lens = pre.run(None, {"waveforms": audio[None], "waveforms_lens": np.array([len(audio)], np.int64)})
    t = min(int(lens[0]), frames)
    padded = np.zeros((1, 128, frames), np.float32)
    padded[:, :, :t] = feats[:, :, :t]
    batch = {"audio_signal": padded, "length": np.array([t], np.int64)}
    for m in upstream:
        sess = pp.session(m, ["CPUExecutionProvider"])
        outs = dict(zip([o.name for o in sess.get_outputs()],
                        sess.run(None, {i.name: batch[i.name] for i in sess.get_inputs()})))
        batch = outs if "hidden" in outs else {**batch, "hidden": outs["hidden_out"]}
    return batch


def timed(sess, feed, repeats):
    for _ in range(2):
        out = sess.run(None, feed)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = sess.run(None, feed)
        times.append(time.perf_counter() - t)
    return out, 1000 * float(np.median(times))


def compare(a, b, valid):
    """Error over the real (unpadded) frames only; padded frames are masked downstream."""
    a, b = a[:, :valid].ravel().astype(np.float64), b[:, :valid].ravel().astype(np.float64)
    return {"cosine": float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), "max_abs": float(np.abs(a - b).max()),
            "rel_rms": float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(a ** 2)))}


def summarize_profile(path, runs):
    """Per-run milliseconds by op type from an ORT profile (NPU partitions show as their fused op)."""
    events = json.loads(Path(path).read_text())
    by = Counter()
    for e in events:
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
            by[e["args"].get("op_name", "?")] += e["dur"]
    return {k: round(v / runs / 1000, 2) for k, v in by.most_common()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qmodel", type=Path, required=True)
    ap.add_argument("--fp32", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, nargs="*", default=[])
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--wav", type=Path, default=pp.ROOT.parent / "parakeet-stt" / "test.wav")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--profile", action="store_true", help="ORT profile of the NPU session (per-node times)")
    args = ap.parse_args()

    import onnxruntime as ort
    raw = Counter(n.op_type for n in onnx.load(str(args.qmodel), load_external_data=False).graph.node)
    print("quantized graph:", dict(raw.most_common()), flush=True)
    feed_all = block_inputs(args.upstream, args.frames, args.wav)
    fp32 = pp.session(args.fp32, ["CPUExecutionProvider"])
    feed = {i.name: feed_all[i.name] for i in fp32.get_inputs()}
    ref, fp32_ms = timed(fp32, feed, args.repeats)
    valid = int((~feed_all["pad_mask"]).sum())
    result = {"qmodel": args.qmodel.name, "fp32_cpu_ms": fp32_ms}

    if not args.skip_cpu:
        qcpu = pp.session(args.qmodel, ["CPUExecutionProvider"])
        out, ms = timed(qcpu, feed, args.repeats)
        result["qdq_cpu_ms"] = ms
        result["qdq_cpu_vs_fp32"] = compare(ref[0], out[0], valid)
        del qcpu

    os.environ["XLNX_ONNX_EP_REPORT_FILE"] = REPORT
    key = args.qmodel.stem
    cache_dir = pp.CACHE / "vitis"
    t = time.perf_counter()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if args.profile:
        so.enable_profiling = True
        so.profile_file_prefix = str(pp.CACHE / "results" / f"profile_{key}")
    npu = ort.InferenceSession(str(args.qmodel), so, providers=["VitisAIExecutionProvider"],
                               provider_options=ts.vitis_options(key, cache_dir)[:1])
    result["compile_s"] = time.perf_counter() - t
    report = json.loads((cache_dir / key / REPORT).read_text())
    result["assignment"] = {r["name"]: r["nodeNum"] for r in report["deviceStat"]}
    result["cpu_op_types"] = dict(Counter(r["opType"] for r in report["nodeStat"] if r["device"] != "NPU"))
    result["npu_op_types"] = dict(Counter(r["opType"] for r in report["nodeStat"] if r["device"] == "NPU"))
    print(json.dumps(result, indent=1), flush=True)
    out, ms = timed(npu, feed, args.repeats)
    result["npu_ms"] = ms
    result["npu_vs_fp32"] = compare(ref[0], out[0], valid)
    if args.profile:
        result["profile"] = summarize_profile(npu.end_profiling(), args.repeats + 2)
    print(json.dumps(result, indent=1))
    out_path = pp.CACHE / "results" / f"block_{key}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
