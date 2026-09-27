"""Export the projected relative-position tables for long inputs from the dynamic encoder.

Each layer's attention uses p = linear_pos(pos_emb(T)), [1, 8, 128, 2T-1],
which depends only on the frame count T. This extracts the sub-graph from
(audio_signal, length) to the 24 per-layer p tensors of the original dynamic
export, runs it once for the longest supported input (default 3072 feature
frames = 30.7 s = 384 encoder frames) and saves cache/parakeet/weights/pos_T<frames>.npz.
parakeet_hybrid.pos_for() slices shorter lengths out of it (centred window).

  python experiments\\011_export_pos.py --frames 3072
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx.external_data_helper
import onnxruntime as ort
from onnx.utils import Extractor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_pipeline as pp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3072)
    args = ap.parse_args()
    src = pp.MODEL_DIR / "encoder-model.onnx"
    model = onnx.load(str(src), load_external_data=False)
    names = [f"/layers.{i}/self_attn/Transpose_3_output_0" for i in range(24)]
    for n in names:
        model.graph.value_info.append(onnx.helper.make_tensor_value_info(
            n, onnx.TensorProto.FLOAT, ["one", 8, 128, "positions"]))
    sub = Extractor(model).extract_model([i.name for i in model.graph.input], names)
    onnx.external_data_helper.load_external_data_for_model(sub, str(src.parent))
    for init in sub.graph.initializer:
        init.ClearField("data_location")
        del init.external_data[:]
    print(f"position sub-graph: {len(sub.graph.node)} nodes")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(sub.SerializeToString(), so, providers=["CPUExecutionProvider"])
    feeds = {"audio_signal": np.zeros((1, 128, args.frames), np.float32),
             "length": np.array([args.frames], np.int64)}
    feeds = {i.name: feeds[i.name] for i in sess.get_inputs()}
    outs = sess.run(None, feeds)
    out = pp.CACHE / "weights" / f"pos_T{args.frames}.npz"
    np.savez(out, **{f"p{i}": o for i, o in enumerate(outs)})
    print("wrote", out, outs[0].shape)
    # consistency with the T=1000 export (a centred slice must match)
    small = pp.CACHE / "weights" / "pos_T1000.npz"
    if small.exists():
        a = np.load(small)["p5"][0]
        b = outs[5][0]
        c, w = b.shape[-1] // 2, a.shape[-1] // 2
        print("max |p(T=1000) - slice of p(long)|:", float(np.abs(a - b[:, :, c - w:c + w + 1]).max()))


if __name__ == "__main__":
    main()
