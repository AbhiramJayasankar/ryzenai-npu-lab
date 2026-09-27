"""Save FP32 ONNX Runtime encoder outputs for a few clips (reference for the hybrid NPU encoder).

  python experiments\\011_reference_outputs.py --model cache\\parakeet\\models\\encoder_T1000.onnx

Clips: test.wav plus the first three evaluation utterances that fit the fixed
length. Writes cache/parakeet/reference_T<frames>.npz with, per clip, the padded
features, the true length, the full encoder output and the FP32 transcript.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_pipeline as pp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--clips", type=int, default=3)
    args = ap.parse_args()
    import onnx
    path = args.model if args.model.exists() else pp.MODEL_DIR / args.model
    frames = onnx.load(str(path), load_external_data=False).graph.input[0].type.tensor_type.shape.dim[2].dim_value
    model = pp.Parakeet(path, fixed_frames=frames or None)
    wavs = [pp.ROOT.parent / "parakeet-stt" / "test.wav"]
    items = json.loads((pp.CACHE / "eval.json").read_text())
    for x in items:
        if len(wavs) > args.clips:
            break
        if x["seconds"] < 10:
            wavs.append(pp.ROOT / x["wav"])
    if not frames:
        wavs.append(pp.ROOT / max(items, key=lambda x: x["seconds"])["wav"])
    out = {}
    for i, wav in enumerate(wavs):
        feats, lens = model.features(pp.load_audio(wav))
        enc, enc_len = model.encode(feats, lens)
        text = model.decode(enc, enc_len)
        out[f"feats{i}"], out[f"len{i}"], out[f"enc{i}"] = model.pad(feats), lens, enc
        out[f"text{i}"] = np.array(text)
        out[f"name{i}"] = np.array(Path(wav).stem)
        print(Path(wav).stem, enc.shape, int(enc_len[0]), text)
    np.savez(pp.CACHE / (f"reference_T{frames}.npz" if frames else "reference_dyn.npz"), **out)


if __name__ == "__main__":
    main()
