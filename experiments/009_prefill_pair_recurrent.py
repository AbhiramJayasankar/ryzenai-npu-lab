"""Verify two consecutive LFM2.5 recurrent positions in one NPU program."""

import argparse
import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16

from lfm25_checkpoint import BF16Checkpoint, pack_recurrent_weights, recurrent_layer_data
from lfm25_prefill_pair_block_kernel import recurrent_pair_block
from lfm25_prefill_pair_data_kernel import pack_recurrent_pair
from lfm25_prefill_pair_direct_kernel import recurrent_pair_direct


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-pack", action="store_true")
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()
    if args.npu_pack and args.direct:
        parser.error("Choose one pair data path")
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    with np.load(ROOT / "cache/lfm25-reference-cpu.npz") as reference:
        hidden0 = reference["step1_hidden0"].astype(bfloat16)
        hidden1 = reference["step2_hidden0"].astype(bfloat16)
        initial_state = reference["step0_conv0_state"].reshape(-1).astype(bfloat16)
        expected0 = reference["step1_hidden1"].reshape(-1)
        expected1 = reference["step2_hidden1"].reshape(-1)
        expected_state = reference["step2_conv0_state"].reshape(-1)
    checkpoint = BF16Checkpoint(ROOT / "cache/lfm25-230m/model.safetensors")
    layer = recurrent_layer_data(checkpoint, 0)
    packed = np.concatenate((
        np.pad(hidden0, (0, 6144 - 1024)),
        np.pad(hidden1, (0, 6144 - 1024)),
        initial_state,
        layer["conv_weight"].reshape(-1),
    )).astype(bfloat16)
    if args.npu_pack or args.direct:
        pair = iron.tensor(np.concatenate((hidden0, hidden1)).astype(bfloat16),
                           dtype=bfloat16)
        state = iron.tensor(np.concatenate((
            initial_state, layer["conv_weight"].reshape(-1),
        )).astype(bfloat16), dtype=bfloat16)
        data = iron.zeros((18432,), dtype=bfloat16, device="npu") if args.npu_pack else None
    else:
        data = iron.tensor(packed, dtype=bfloat16)
    weights = iron.tensor(pack_recurrent_weights(layer), dtype=bfloat16)
    final_state = iron.zeros((6144,), dtype=bfloat16, device="npu")
    outputs = iron.zeros((2048,), dtype=bfloat16, device="npu")

    def run():
        if args.direct:
            recurrent_pair_direct(pair, state, weights, final_state, outputs)
        elif args.npu_pack:
            pack_recurrent_pair(pair, state, data)
            recurrent_pair_block(data, weights, final_state, outputs)
        else:
            recurrent_pair_block(data, weights, final_state, outputs)

    run()
    elapsed = []
    for _ in range(5):
        start = time.perf_counter()
        run()
        elapsed.append((time.perf_counter() - start) * 1000)
    actual = outputs.numpy().astype(np.float32).reshape(2, 1024)
    actual_state = final_state.numpy().astype(np.float32)
    result = {
        "device": "Phoenix NPU1",
        "operation": "one complete recurrent layer for two consecutive positions",
        "npu_pack": args.npu_pack,
        "direct": args.direct,
        "runs_ms": elapsed,
        "median_ms": statistics.median(elapsed),
        "position0_max_abs_error": float(np.max(np.abs(actual[0] - expected0))),
        "position1_max_abs_error": float(np.max(np.abs(actual[1] - expected1))),
        "final_state_max_abs_error": float(np.max(np.abs(actual_state[:3072] - expected_state))),
        "weight_persisted": bool(np.array_equal(
            actual_state[3072:], layer["conv_weight"].reshape(-1).astype(np.float32)
        )),
    }
    print(json.dumps(result, indent=2))
    if max(result["position0_max_abs_error"], result["position1_max_abs_error"]) > 0.015625:
        raise AssertionError("Recurrent pair output exceeded BF16 tolerance")
    if result["final_state_max_abs_error"] != 0 or not result["weight_persisted"]:
        raise AssertionError("Recurrent pair state mismatch")


if __name__ == "__main__":
    main()
