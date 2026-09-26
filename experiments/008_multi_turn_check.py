"""Verify that chunked NPU chat preserves state across the 64-token boundary."""

import importlib
import json


NPUChat = importlib.import_module("007_lfm25_npu_chat").NPUChat


def main():
    chat = NPUChat("chunked")
    history = [{"role": "user", "content": "Tell me about NPUs in simple terms. " * 5}]
    first = chat.respond(history, 4)
    history.append({"role": "assistant", "content": first["response"]})
    history.append({"role": "user", "content": "What is one limitation?"})
    second = chat.respond(history, 4)
    if not second["reused_npu_state"]:
        raise AssertionError("Second turn replayed rather than reusing NPU state")
    if second["prompt_tokens"] <= 64:
        raise AssertionError("Second turn did not cross the old context limit")
    print(json.dumps({"first": first, "second": second}, indent=2))


if __name__ == "__main__":
    main()
