# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers>=0.20", "huggingface_hub>=0.34"]
# ///
"""Compare the tokenizers of T-lite-it-2.1 and Qwen3-8B on this project's Russian texts.

CONCEPT.md §10 assumes the pair differs only in fine-tuning. The T-lite card mentions an
"optimized tokenizer"; this measures how different it is: vocabulary overlap, how many
token ids map to the same string, and token counts for the call scripts, chat and e-mail.

    uv run tools/compare_tokenizers.py      -> results/llm/tokenizers.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"t-lite": "t-tech/T-lite-it-2.1", "qwen3-8b": "Qwen/Qwen3-8B"}


def texts() -> dict[str, str]:
    out = {}
    for script in sorted((ROOT / "data" / "scripts").glob("*.txt")):
        lines = [re.sub(r"\[\[(.+?)\|\w+\]\]", r"\1", ln.split(":", 1)[1]) for ln in script.read_text().splitlines()
                 if ln[:1] in "OC"]
        out[script.stem] = " ".join(s.strip() for s in lines)
    chat = json.loads((ROOT / "data" / "cases" / "demo" / "chat.json").read_text())
    out["demo-chat"] = " ".join(m["text"] for m in chat["messages"])
    out["demo-email"] = json.loads((ROOT / "results" / "emails" / "demo.email-1.json").read_text())["body"]
    return out


def main() -> None:
    toks = {k: Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json")) for k, repo in MODELS.items()}
    vocab = {k: t.get_vocab() for k, t in toks.items()}
    a, b = vocab["t-lite"], vocab["qwen3-8b"]
    same_id = sum(1 for s, i in a.items() if b.get(s) == i)
    report = {
        "vocab_size": {k: len(v) for k, v in vocab.items()},
        "shared_strings": len(a.keys() & b.keys()),
        "same_string_same_id": same_id,
        "only_in_t_lite_sample": sorted(a.keys() - b.keys(), key=len, reverse=True)[:30],
        "token_counts": {
            name: {k: len(t.encode(text).ids) for k, t in toks.items()} | {"chars": len(text)}
            for name, text in texts().items()
        },
    }
    totals = {k: sum(v[k] for v in report["token_counts"].values()) for k in MODELS}
    report["token_totals"] = totals
    out = ROOT / "results" / "llm" / "tokenizers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("vocab_size", "shared_strings", "same_string_same_id", "token_totals")}))


if __name__ == "__main__":
    main()
