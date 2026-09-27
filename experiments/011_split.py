"""Split a fixed-shape Parakeet encoder into a CPU prefix and Conformer-layer blocks.

Why: Quark cannot quantize one ONNX file above protobuf's 2 GB limit on this
Windows build (ModelProto.ByteSize raises), and small blocks compile and test
faster on the NPU. Each block of layers [a, b] takes
  hidden   [1, T/8, 1024]                (output of the previous block)
  att_mask [1, 1, T/8, T/8] bool         (True = padded pair, masked)
  pad_mask [1, 1, T/8] bool              (True = padded frame, zeroed before depthwise conv)
and returns the hidden state after layer b. The prefix (subsampling convs and
mask construction, ~1% of encoder FLOPs) maps audio_signal/length to
hidden0 + both masks. The final output is hidden_23 transposed to [1, 1024, T/8].

  python experiments\\011_split.py --model cache\\parakeet\\models\\encoder_T1000.onnx --blocks 0-1
  python experiments\\011_split.py --model ... --blocks prefix 0-11 12-23
"""

import argparse
from pathlib import Path

import onnx
import onnx.external_data_helper
from onnx.utils import Extractor

HIDDEN0 = "/pre_encode/out/Add_output_0"
ATT_MASK = "/layers.0/self_attn/Unsqueeze_35_output_0"
PAD_MASK = "/layers.0/conv/Unsqueeze_output_0"


def layer_out(i):
    return f"/layers.{i}/norm_out/LayerNormalization_output_0"


def rename(model, mapping):
    for node in model.graph.node:
        for k, name in enumerate(node.input):
            if name in mapping:
                node.input[k] = mapping[name]
        for k, name in enumerate(node.output):
            if name in mapping:
                node.output[k] = mapping[name]
    for vi in list(model.graph.input) + list(model.graph.output):
        if vi.name in mapping:
            vi.name = mapping[vi.name]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--blocks", nargs="+", required=True)
    ap.add_argument("--out-stem", default=None, help="output name stem (default: model stem)")
    args = ap.parse_args()

    # Load graph only (weights stay external) so shape inference can serialize it.
    model = onnx.load(str(args.model), load_external_data=False)
    # Shape inference cannot see external weights, so declare the boundary tensors.
    t = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    if t:
        for _ in range(3):
            t = (t - 1) // 2 + 1
    else:  # dynamic-length encoder: symbolic frame count
        t = "frames"
    f, b = onnx.TensorProto.FLOAT, onnx.TensorProto.BOOL
    declared = {HIDDEN0: (f, [1, t, 1024]), ATT_MASK: (b, [1, 1, t, t]), PAD_MASK: (b, [1, 1, t])}
    declared.update({layer_out(i): (f, [1, t, 1024]) for i in range(24)})
    known = {v.name for v in model.graph.value_info}
    for name, (dtype, shape) in declared.items():
        if name not in known:
            model.graph.value_info.append(onnx.helper.make_tensor_value_info(name, dtype, shape))
    if not isinstance(t, int):  # batch dim is symbolic too in the dynamic export
        for v in model.graph.value_info:
            if v.name in declared:
                v.type.tensor_type.shape.dim[0].dim_param = "batch"
    ex = Extractor(model)
    for block in args.blocks:
        if block == "prefix":
            inputs = [i.name for i in model.graph.input]
            outputs = [HIDDEN0, ATT_MASK, PAD_MASK]
            mapping = {HIDDEN0: "hidden", ATT_MASK: "att_mask", PAD_MASK: "pad_mask"}
        else:
            a, b = map(int, block.split("-"))
            inputs = [HIDDEN0 if a == 0 else layer_out(a - 1), ATT_MASK, PAD_MASK]
            outputs = [layer_out(b)]
            mapping = {inputs[0]: "hidden", ATT_MASK: "att_mask", PAD_MASK: "pad_mask", outputs[0]: "hidden_out"}
        sub = ex.extract_model(inputs, outputs)
        rename(sub, mapping)
        onnx.external_data_helper.load_external_data_for_model(sub, str(args.model.parent))
        for init in sub.graph.initializer:  # now embedded: drop the external-file references
            init.ClearField("data_location")
            del init.external_data[:]
        stem = args.out_stem or args.model.stem
        out = args.model.parent / f"{stem}_{block.replace('-', '_')}.onnx" if args.out_stem is None else             Path(args.out_stem).parent / f"{Path(args.out_stem).name}_{block.replace('-', '_')}.onnx"
        size = sum(t.ByteSize() for t in sub.graph.initializer)
        if size > 1.8e9:
            onnx.save(sub, str(out), save_as_external_data=True, location=out.name + ".data")
        else:
            onnx.save(sub, str(out))
        print(f"{out.name}: {len(sub.graph.node)} nodes, {size / 1e6:.0f} MB weights, "
              f"inputs {[i.name for i in sub.graph.input]}")


if __name__ == "__main__":
    main()
