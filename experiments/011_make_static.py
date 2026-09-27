"""Make a fixed-shape FP32 Parakeet encoder for the NPU compiler.

The exported encoder has dynamic batch and time axes. Vitis AI needs static
shapes, so this fixes audio_signal to [1, 128, FRAMES] and length to [1], then
lets ONNX Runtime's basic (provider-independent) optimizer fold the Shape /
Gather / Concat chains that only computed sizes. With --full-length the
`length` input is removed and replaced by the constant FRAMES, so the
attention/convolution masks fold to constants too (the caller must then feed
exactly FRAMES real frames, or accept that padding is attended to).

  python experiments\\011_make_static.py --frames 1000
Output: cache/parakeet/models/encoder_T<FRAMES>[_full].onnx (+ .data)
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_pipeline as pp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--full-length", action="store_true")
    args = ap.parse_args()
    out_dir = pp.CACHE / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"encoder_T{args.frames}{'_full' if args.full_length else ''}"
    fixed = out_dir / f"{name}_fixed.onnx"
    final = out_dir / f"{name}.onnx"

    model = onnx.load(str(pp.MODEL_DIR / "encoder-model.onnx"))  # loads the 2.4 GB external data
    g = model.graph
    for inp in g.input:
        dims = inp.type.tensor_type.shape.dim
        for d, v in zip(dims, [1, 128, args.frames] if inp.name == "audio_signal" else [1]):
            d.ClearField("dim_param")
            d.dim_value = v
    for out in g.output:
        del out.type.tensor_type.shape.dim[:]
    if args.full_length:
        g.input.remove(next(i for i in g.input if i.name == "length"))
        g.initializer.append(numpy_helper.from_array(np.array([args.frames], np.int64), "length"))
    onnx.save(model, str(fixed), save_as_external_data=True, all_tensors_to_one_file=True,
              location=fixed.name + ".data", size_threshold=1024)
    del model

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(final)
    so.add_session_config_entry("session.optimized_model_external_initializers_file_name", final.name + ".data")
    so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "1024")
    ort.InferenceSession(str(fixed), so, providers=["CPUExecutionProvider"])
    fixed.unlink()
    Path(str(fixed) + ".data").unlink()

    g = onnx.load(str(final), load_external_data=False).graph
    print(final, len(g.node), "nodes", Counter(n.op_type for n in g.node).most_common())


if __name__ == "__main__":
    main()
