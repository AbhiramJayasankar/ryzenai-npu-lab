"""Time first-token prefill for the same 20-word chat prompt as the CPU baseline."""

import importlib
import json
import statistics

from _009_prefill_prompt import PROMPT


def main():
    chat = importlib.import_module("007_lfm25_npu_chat").NPUChat("fixed64")
    results = []
    for _ in range(3):
        result = chat.respond([{"role": "user", "content": PROMPT}], 1)
        results.append(result)
    token_ids = [run["generated_ids"] for run in results]
    if len(set(map(tuple, token_ids))) != 1:
        raise AssertionError("NPU token choice changed across repeats")
    print(json.dumps({
        "prompt": PROMPT,
        "chat_tokens": results[0]["prompt_tokens"],
        "device": results[0]["device"],
        "runs_ms": [run["elapsed_seconds"] * 1000 for run in results],
        "median_ms": statistics.median(run["elapsed_seconds"] * 1000 for run in results),
        "first_token_id": token_ids[0][0] if token_ids[0] else None,
    }, indent=2))


if __name__ == "__main__":
    main()
