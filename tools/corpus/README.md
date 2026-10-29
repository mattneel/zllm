# Tokenizer parity corpus

`tokenizer.jsonl` holds one `{"source": ..., "text": ...}` object per line. Lines are split on LF only, because the texts themselves contain U+2028, NEL and other line breaks. The corpus is pinned: change it only in a commit of its own, and rerun `zig build check-tokenizer` when you do.

| `source` | Contents |
|---|---|
| `llama.cpp` | The 51 inputs from `models/ggml-vocab-qwen35.gguf.inp` at the pinned llama.cpp commit (MIT). |
| `curated` | Targeted cases: contractions, every pre-tokenizer whitespace branch, Unicode line and space separators, numbers in several scripts, NFC edge cases (composition, reordering, exclusions, Hangul, singletons), a range of scripts, emoji ZWJ sequences, code, and added/special tokens, including partial and adjacent matches. |
| `document` | Three long real documents: the v0.1 spec, the Qwen chat template and a Zig source file. Each is frozen at the time it was added. |
