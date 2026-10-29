"""Compare zllm tokenization with Hugging Face `tokenizers` on the pinned corpus.

For every line of tools/corpus/tokenizer.jsonl, the token ids from zllm (vocab
read from GGUF) must equal the ids from the pinned Hugging Face tokenizer.json,
and decoding those ids must give the same text as Hugging Face's decoder.
Runs on the range-read GGUF header (what CI fetches) and on the full model GGUF when present.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tokenizers import Tokenizer

from common import Failures, fixture_paths, parse_args

CORPUS = Path(__file__).parent / "corpus" / "tokenizer.jsonl"


def main() -> int:
    args = parse_args(__doc__)
    paths = fixture_paths(args.fixtures)
    hf = Tokenizer.from_file(str(paths.hf_tokenizer))
    # Split on LF only: corpus strings may contain U+2028, NEL and friends.
    texts = [json.loads(line)["text"] for line in CORPUS.read_text(encoding="utf-8").split("\n") if line]
    fail = Failures()

    models = [m for m in (paths.gguf_header, paths.gguf) if m.is_file()]
    if paths.gguf_header not in models:
        raise SystemExit(f"missing {paths.gguf_header}; run scripts/fixtures.sh vocab")

    for model in models:
        out = subprocess.run([args.zllm, "tokenize", str(model), str(CORPUS)], check=True, capture_output=True).stdout
        results = [json.loads(line) for line in out.decode("utf-8").split("\n") if line]
        fail.expect_eq(f"{model.name}: line count", len(results), len(texts))
        matched = 0
        n_tokens = 0
        for i, (text, got) in enumerate(zip(texts, results, strict=False)):
            want = hf.encode(text).ids
            n_tokens += len(want)
            if got["ids"] == want:
                matched += 1
            else:
                fail.expect_eq(f"{model.name}: line {i + 1} ids for {text[:40]!r}", got["ids"], want)
            fail.expect_eq(f"{model.name}: line {i + 1} decode", got["decoded"], hf.decode(want, skip_special_tokens=False))
        print(f"tokenizer: {model.name}: {matched}/{len(texts)} texts, {n_tokens} tokens match Hugging Face")

    return fail.report()


if __name__ == "__main__":
    sys.exit(main())
