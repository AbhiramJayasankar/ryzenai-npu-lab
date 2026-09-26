"""Check LFM2.5 final RMSNorm and tied vocabulary projection on NPU."""

import argparse
import json
import statistics
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from lfm25_argmax_kernel import vocab_argmax
from lfm25_bf16_gemv_kernel import bf16_bf16_gemv
from lfm25_checkpoint import BF16Checkpoint
from lfm25_rms_norm_kernel import rms_norm
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, choices=(1024, 65536), default=1024)
    args = parser.parse_args()
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    checkpoint = BF16Checkpoint(ROOT / "cache" / "lfm25-230m" / "model.safetensors")
    gamma = checkpoint.load("model.embedding_norm.weight")
    weight = checkpoint.load("model.embed_tokens.weight")[:args.rows]
    with np.load(ROOT / "cache" / "lfm25-reference-cpu.npz") as reference:
        raw = reference["step1_hidden14_raw"].reshape(-1).astype(bfloat16)
        expected_norm = reference["step1_hidden14"].reshape(-1)
        expected_logits = reference["step1_logits"].reshape(-1)[:args.rows]
    x = iron.tensor(raw, dtype=bfloat16)
    g = iron.tensor(gamma, dtype=bfloat16)
    w = iron.tensor(weight, dtype=bfloat16)
    normalized = iron.zeros((1024,), dtype=bfloat16, device="npu")
    logits = iron.zeros((args.rows,), dtype=bfloat16, device="npu")
    token_id = iron.zeros((1,), dtype=np.int32, device="npu") if args.rows == 65536 else None

    def run():
        rms_norm(x, g, normalized, N=1024)
        bf16_bf16_gemv(w, normalized, logits, M=args.rows, K=1024, n_cores=4)
        if token_id is not None:
            vocab_argmax(logits, token_id)

    run()
    times = []
    for _ in range(3):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    actual_norm = normalized.numpy().astype(np.float32)
    actual_logits = logits.numpy().astype(np.float32)
    result = {
        "device": "Phoenix NPU1",
        "operation": "final normalization and tied vocabulary projection",
        "vocabulary_rows": args.rows,
        "median_ms": statistics.median(times),
        "norm_max_abs_error": float(np.max(np.abs(actual_norm - expected_norm))),
        "logits_max_abs_error": float(np.max(np.abs(actual_logits - expected_logits))),
        "logits_exact_fraction": float(np.mean(actual_logits == expected_logits)),
        "npu_argmax_host_readback": int(np.argmax(actual_logits)),
        "cpu_argmax": int(np.argmax(expected_logits)),
        "npu_argmax": int(token_id.numpy()[0]) if token_id is not None else None,
    }
    print(json.dumps(result, indent=2))
    if result["norm_max_abs_error"] != 0:
        raise RuntimeError("NPU final norm differs from CPU reference")
    if result["logits_max_abs_error"] > 0.125:
        raise RuntimeError("NPU vocabulary logits exceed BF16 tolerance")
    if token_id is not None and result["npu_argmax"] != result["cpu_argmax"]:
        raise RuntimeError("NPU argmax selected a different token")


if __name__ == "__main__":
    main()
