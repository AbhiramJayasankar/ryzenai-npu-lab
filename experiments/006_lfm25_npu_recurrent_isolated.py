"""Check one recurrent layer with matching CPU reference input and state."""

import argparse
import json
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_checkpoint import BF16Checkpoint, pack_recurrent_weights, recurrent_layer_data
from lfm25_single_program_block_kernel import recurrent_block
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    if args.layer not in (0, 1, 3, 5, 7, 9, 11, 13):
        parser.error("Expected a recurrent layer index")
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    checkpoint = BF16Checkpoint(ROOT / "cache" / "lfm25-230m" / "model.safetensors")
    weights = recurrent_layer_data(checkpoint, args.layer)
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as reference:
        hidden = reference[f"step1_hidden{args.layer}"].reshape(-1).astype(bfloat16)
        state = reference[f"step0_conv{args.layer}_state"].reshape(-1).astype(bfloat16)
        expected_key = "step1_hidden14_raw" if args.layer == 13 else f"step1_hidden{args.layer + 1}"
        expected_hidden = reference[expected_key].reshape(-1)
        expected_state = reference[f"step1_conv{args.layer}_state"].reshape(-1)
    packed = np.concatenate([
        np.pad(hidden, (0, 6144 - hidden.size)),
        state,
        weights["conv_weight"].reshape(-1),
    ]).astype(bfloat16)
    input_tensor = iron.tensor(packed, dtype=bfloat16)
    weight_tensor = iron.tensor(pack_recurrent_weights(weights), dtype=bfloat16)
    state_tensor = iron.zeros((6144,), dtype=bfloat16, device="npu")
    output_tensor = iron.zeros((1024,), dtype=bfloat16, device="npu")
    recurrent_block(input_tensor, weight_tensor, state_tensor, output_tensor)
    actual = output_tensor.numpy().astype(np.float32)
    actual_state = state_tensor.numpy().astype(np.float32)[:3072]
    result = {
        "layer": args.layer,
        "hidden_max_abs_error": float(np.max(np.abs(actual - expected_hidden))),
        "hidden_exact_fraction": float(np.mean(actual == expected_hidden)),
        "state_max_abs_error": float(np.max(np.abs(actual_state - expected_state))),
    }
    print(json.dumps(result, indent=2))
    if result["hidden_max_abs_error"] > 0.015625:
        raise RuntimeError("Isolated recurrent layer exceeds BF16 tolerance")


if __name__ == "__main__":
    main()
