"""How fast is one Conformer-sized MatMul through the Vitis AI EP on Phoenix?

Builds [1, M, K] x [K, N] models (Parakeet FFN linear1: K=1024, N=4096),
quantizes them with Quark XINT8 (the only scheme that reached the Phoenix NPU
in block tests), and times NPU vs CPU FP32 vs CPU QDQ for several M (M =
encoder frames = seconds * 12.5). Reports effective GMAC/s.

  python experiments\\011_ep_matmul_probe.py --m 125 375 1000
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_pipeline as pp  # noqa: E402

ts = importlib.import_module("011_transcribe_set")


class Reader:
    def __init__(self, m, k, n=8):
        rng = np.random.default_rng(0)
        self.data = [{"x": rng.standard_normal((1, m, k)).astype(np.float32)} for _ in range(n)]
        self.it = iter(self.data)

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter(self.data)


def timed(sess, feed, repeats=20):
    for _ in range(3):
        sess.run(None, feed)
    t = []
    for _ in range(repeats):
        s = time.perf_counter()
        sess.run(None, feed)
        t.append(time.perf_counter() - s)
    return 1000 * float(np.median(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[125, 375, 1000])
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--n", type=int, default=4096)
    args = ap.parse_args()
    import onnxruntime as ort
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config.custom_config import XINT8_QCONFIG

    out_dir = pp.CACHE / "models" / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((args.k, args.n)) / np.sqrt(args.k)).astype(np.float32)
    os.environ["XLNX_ONNX_EP_REPORT_FILE"] = "vitisai_ep_report.json"
    rows = []
    for m in args.m:
        name = f"matmul_{m}x{args.k}x{args.n}"
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"])], name,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, m, args.k])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, m, args.n])],
            [numpy_helper.from_array(w, "w")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        f32 = out_dir / f"{name}.onnx"
        q = out_dir / f"{name}_XINT8.onnx"
        onnx.save(model, str(f32))
        ModelQuantizer(XINT8_QCONFIG).quantize_model(str(f32), str(q), calibration_data_reader=Reader(m, args.k))
        feed = Reader(m, args.k).data[0]
        cpu = pp.session(f32, ["CPUExecutionProvider"])
        qcpu = pp.session(q, ["CPUExecutionProvider"])
        so = ort.SessionOptions()
        so.log_severity_level = 3
        npu = ort.InferenceSession(str(q), so, providers=["VitisAIExecutionProvider"],
                                   provider_options=ts.vitis_options(name, pp.CACHE / "vitis")[:1])
        report = json.loads((pp.CACHE / "vitis" / name / "vitisai_ep_report.json").read_text())
        on_npu = {r["name"]: r["nodeNum"] for r in report["deviceStat"]}.get("NPU", 0)
        ref = cpu.run(None, feed)[0]
        got = npu.run(None, feed)[0]
        cos = float((ref.ravel() @ got.ravel()) / np.linalg.norm(ref) / np.linalg.norm(got))
        macs = m * args.k * args.n
        row = {"m": m, "npu_nodes": on_npu, "cosine_npu_vs_fp32": cos}
        for label, sess in (("cpu_fp32", cpu), ("cpu_qdq", qcpu), ("npu", npu)):
            ms = timed(sess, feed)
            row[f"{label}_ms"] = ms
            row[f"{label}_gmacs"] = macs / ms / 1e6
        rows.append(row)
        print(json.dumps(row), flush=True)
    (pp.CACHE / "results" / "ep_matmul_probe.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
