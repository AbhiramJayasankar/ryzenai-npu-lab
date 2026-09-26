"""Compile variable-length attention programs into the ignored IRON disk cache."""

import argparse
import time

import aie.iron as iron

from lfm25_attention_context_cache_kernel import attention_context_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=int, default=96,
                        help="Prepare positions below this limit (default: 96)")
    args = parser.parse_args()
    if not 2 <= args.through <= 96:
        parser.error("--through must be between 2 and 96")
    if type(iron.get_current_device()).__name__ != "NPU1":
        raise RuntimeError("Expected Phoenix NPU1")
    start = time.perf_counter()
    for length in range(1, args.through):
        attention_context_cache.specialize(past_length=length).compile()
        if length % 10 == 0 or length == args.through - 1:
            print(f"Prepared past length {length}/{args.through - 1} "
                  f"({time.perf_counter() - start:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
