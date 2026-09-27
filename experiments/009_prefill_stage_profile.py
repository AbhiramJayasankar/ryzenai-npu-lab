"""Break down the present token-wise NPU prefill without changing its math."""

import importlib
import json
import time

from _009_prefill_prompt import PROMPT


TARGETS = (
    "pack_block_input", "recurrent_block", "attention_prefix",
    "attention_first_fixed64", "attention_context_fixed64", "attention_tail",
    "fused_vocab_4core",
)


def main():
    module = importlib.import_module("007_lfm25_npu_chat")
    elapsed = {name: 0.0 for name in TARGETS}
    counts = {name: 0 for name in TARGETS}
    for name in TARGETS:
        operation = getattr(module, name)

        def wrapper(*args, _name=name, _operation=operation, **kwargs):
            start = time.perf_counter()
            result = _operation(*args, **kwargs)
            elapsed[_name] += (time.perf_counter() - start) * 1000
            counts[_name] += 1
            return result

        setattr(module, name, wrapper)
    chat = module.NPUChat("fixed64")
    result = chat.respond([{"role": "user", "content": PROMPT}], 1)
    if result["generated_ids"] != [2797]:
        raise AssertionError(result["generated_ids"])
    print(json.dumps({"prompt_tokens": result["prompt_tokens"],
                      "total_ms": result["elapsed_seconds"] * 1000,
                      "stage_ms": elapsed, "stage_calls": counts,
                      "first_token_id": result["generated_ids"][0]}, indent=2))


if __name__ == "__main__":
    main()
