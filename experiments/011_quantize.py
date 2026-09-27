"""Statically quantize the fixed-shape Parakeet encoder with AMD Quark (QDQ) for the Vitis AI EP.

  python experiments\\011_quantize.py --model cache\\parakeet\\models\\encoder_T1000.onnx --config XINT8 --calib 24

Calibration inputs are log-mel features of utterances from the calibration
speakers (cache/parakeet/calib.json, disjoint from the evaluation speakers),
cropped or zero-padded to the model's fixed frame count.
Output: <model stem>_<config>.onnx next to the input model.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_pipeline as pp  # noqa: E402


class FeatureReader:
    def __init__(self, n, frames, full_length, upstream=()):
        pre = pp.session(pp.MODEL_DIR / "nemo128.onnx", ["CPUExecutionProvider"])
        chain = [pp.session(m, ["CPUExecutionProvider"]) for m in upstream]
        items = json.loads((pp.CACHE / "calib.json").read_text())
        items = [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]
        self.batches = []
        for x in items:
            audio = pp.load_audio(pp.ROOT / x["wav"])
            feats, lens = pre.run(None, {"waveforms": audio[None], "waveforms_lens": np.array([len(audio)], np.int64)})
            t = min(int(lens[0]), frames)
            padded = np.zeros((1, 128, frames), np.float32)
            padded[:, :, :t] = feats[:, :, :t]
            batch = {"audio_signal": padded}
            if not full_length:
                batch["length"] = np.array([t], np.int64)
            for sess in chain:  # FP32 prefix / earlier layer blocks -> this block's inputs
                feed = {i.name: batch[i.name] for i in sess.get_inputs()}
                outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feed)))
                if "hidden" in outs:
                    batch = outs
                else:
                    batch = {**batch, "hidden": outs["hidden_out"]}
            self.batches.append(batch)
        self.it = iter(self.batches)

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter(self.batches)


def make_config(name, extra):
    from quark.onnx import QConfig
    from quark.onnx.quantization.config import custom_config as cc

    base = getattr(cc, f"{name}_QCONFIG")
    return QConfig(global_config=base.global_config, layer_type_config=base.layer_type_config,
                   exclude=base.exclude, algo_config=base.algo_config, use_external_data_format=True,
                   **{**base.extra_options, **extra})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--config", default="XINT8")
    ap.add_argument("--calib", type=int, default=24)
    ap.add_argument("--extra", default="{}", help="JSON dict of extra Quark options")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--frames", type=int, default=None, help="for layer blocks: fixed feature frames")
    ap.add_argument("--upstream", type=Path, nargs="*", default=[],
                    help="FP32 prefix and earlier blocks that produce a layer block's inputs")
    args = ap.parse_args()
    from quark.onnx import ModelQuantizer

    inputs = {i.name: i for i in onnx.load(str(args.model), load_external_data=False).graph.input}
    if "audio_signal" in inputs:
        frames = inputs["audio_signal"].type.tensor_type.shape.dim[2].dim_value
    else:
        frames = args.frames
    reader = FeatureReader(args.calib, frames, "length" not in inputs and not args.upstream, args.upstream)
    out = args.model.with_name(f"{args.model.stem}_{args.config}{args.suffix}.onnx")
    t = time.perf_counter()
    ModelQuantizer(make_config(args.config, json.loads(args.extra))).quantize_model(
        str(args.model), str(out), calibration_data_reader=reader)
    print(f"quantized to {out} in {time.perf_counter() - t:.0f} s")


if __name__ == "__main__":
    main()
