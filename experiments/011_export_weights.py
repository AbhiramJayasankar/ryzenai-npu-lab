"""Export the 24 Conformer layers' weights from a fixed-shape FP32 encoder to NumPy files.

  python experiments\\011_export_weights.py --model cache\\parakeet\\models\\encoder_T1000.onnx

Writes cache/parakeet/weights/layer_XX.npz (shape-independent weights, FP32)
and cache/parakeet/weights/pos_T<frames>.npz (the per-layer projected relative
position embeddings, which ONNX Runtime constant-folded for this frame count),
and prints the scalar constants the hybrid layer relies on so they can be checked.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cache" / "parakeet" / "weights"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--pos-only", action="store_true", help="only write pos_T<frames>.npz")
    args = ap.parse_args()
    model = onnx.load(str(args.model))
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    frames = g.input[0].type.tensor_type.shape.dim[2].dim_value
    OUT.mkdir(parents=True, exist_ok=True)

    def arr(name):
        return numpy_helper.to_array(inits[name])

    nodes = {n.name: n for n in g.node}
    pos = {}
    for layer in range(24):
        pre = f"/layers.{layer}/"

        def w(node, idx=1):
            return arr(nodes[pre + node].input[idx])

        d = {}
        for ln in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            d[f"{ln}.w"] = w(f"{ln}/LayerNormalization", 1)
            d[f"{ln}.b"] = w(f"{ln}/LayerNormalization", 2)
        for ff in ("feed_forward1", "feed_forward2"):
            d[f"{ff}.w1"] = w(f"{ff}/linear1/MatMul")
            d[f"{ff}.w2"] = w(f"{ff}/linear2/MatMul")
        for proj in ("q", "k", "v", "out"):
            d[f"att.w{proj}"] = w(f"self_attn/linear_{proj}/MatMul")
        d["att.bias_u"] = w("self_attn/Add", 1)
        d["att.bias_v"] = w("self_attn/Add_1", 1)
        d["conv.pw1"] = w("conv/pointwise_conv1/Conv")[:, :, 0].T.copy()  # [1024, 2048]
        d["conv.dw"] = w("conv/depthwise_conv/Conv")[:, 0, :]  # [1024, 9], batch norm folded in
        d["conv.dw_b"] = w("conv/depthwise_conv/Conv", 2)
        d["conv.pw2"] = w("conv/pointwise_conv2/Conv")[:, :, 0].T.copy()  # [1024, 1024]
        if not args.pos_only:
            np.savez(OUT / f"layer_{layer:02d}.npz", **d)
        pos[f"p{layer}"] = w("self_attn/MatMul")  # [1, 8, 128, 2T'-1]
        if layer == 0:
            consts = {"ff_scale": w("Mul"), "att_div": w("self_attn/Div"),
                      "mask_fill": w("self_attn/Where"), "prob_fill": w("self_attn/Where_1"),
                      "conv_fill": w("conv/Where")}
            eps = [a.f for a in nodes[pre + "norm_out/LayerNormalization"].attribute if a.name == "epsilon"]
            print("constants:", {k: float(v) for k, v in consts.items()}, "eps", eps)
            print("rel-shift pad:", w("self_attn/Pad"), "dw pad:", w("conv/depthwise_conv/Pad"))
            print({k: v.shape for k, v in d.items()}, "p", pos["p0"].shape)
    np.savez(OUT / f"pos_T{frames}.npz", **pos)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
