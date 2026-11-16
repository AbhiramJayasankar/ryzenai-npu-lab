"""Experiment 014, lead 3 (LLM half), step 1: tokenize the evaluation text for
014_int8_lfm25.py. Runs in the IRON venv, which has the `tokenizers` package
(the system Python that has PyTorch does not).

Text: docs/writeup.md (technical English, split into paragraphs of at most
500 tokens) and the 210 LibriSpeech transcripts of experiment 011 in sentence
case (prose). Writes cache/014/llm_text_ids.json.

  cache\\iron\\mlir-aie\\ironenv\\Scripts\\python.exe experiments\\014_llm_tokenize.py
"""

import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
BOS = 1


def main():
    tok = Tokenizer.from_file(str(ROOT / "cache" / "lfm25-230m" / "tokenizer.json"))
    docs = []
    paras = [p.strip() for p in (ROOT / "docs" / "writeup.md").read_text(encoding="utf-8").split("\n\n")]
    buf = ""
    for p in paras:
        if not p:
            continue
        cand = (buf + "\n\n" + p) if buf else p
        if len(tok.encode(cand, add_special_tokens=False).ids) > 500 and buf:
            docs.append(("writeup", buf))
            buf = p
        else:
            buf = cand
    if buf:
        docs.append(("writeup", buf))
    items = json.loads((ROOT / "cache" / "parakeet" / "eval.json").read_text())
    speech = [x["text"].lower().capitalize() + "." for x in items]
    for i in range(0, len(speech), 10):  # 21 passages of 10 sentences
        docs.append(("librispeech", " ".join(speech[i:i + 10])))
    out = [{"source": s, "ids": [BOS] + tok.encode(t, add_special_tokens=False).ids} for s, t in docs]
    path = ROOT / "cache" / "014" / "llm_text_ids.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"{len(out)} passages, {sum(len(x['ids']) for x in out)} tokens -> {path}")


if __name__ == "__main__":
    main()
