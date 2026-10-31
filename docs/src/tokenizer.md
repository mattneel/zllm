# Tokenization and encode throughput

zllm loads tokenizers from GGUF `tokenizer.ggml.*` metadata, not from a
separate `tokenizer.json`. The same tokenizer API serves the native CLI and
the browser core. Its output ids retain the GGUF/Hugging Face vocabulary
indices: merge ranks choose operations, but never replace token ids.
The supported byte-level BPE families are Qwen 3.5, Qwen 3, Llama 3 and
Granite 4, with their model-specific pre-tokenization and normalization.
This is not a general loader for every Hugging Face tokenizer family.

## Engine design

The BPE core uses a [toks-inspired](https://github.com/actual-computer/toks)
separation of immutable vocabulary/merge data and per-encode scratch. It
is a Zig implementation, not a linked toks runtime, and does not inherit
toks' platform-specific assembly, model support, or published performance.

- Loading builds a flat open-addressed table keyed by adjacent token ids.
  Entries retain both the merge rank and the resulting vocabulary id.
- Each pre-tokenized piece begins as byte-token symbols with linked
  neighbors. After a merge only the affected neighboring candidates need
  a fresh table lookup. Short pieces scan cached ranks; longer pieces use
  a minimum heap with generation counters to discard stale candidates.
  Equal-rank candidates merge leftmost first.
- Added/control tokens are matched on the original text, leftmost-longest,
  with candidates bucketed by their first byte. NFC normalization, when
  required by the model, applies only to the intervening ordinary text.
  Segments already trivially NFC borrow their validated input instead of
  allocating and copying an unchanged normalized string.
- Llama's `ignore_merges` path checks a raw-byte vocabulary lookup before
  applying BPE. It does not expand each piece into the GPT-2 byte alphabet.
- Decoding copies from vocabulary bytes decoded and packed at load time.
  Added tokens retain their literal text; BOS insertion follows the model.

The core uses portable Zig integer and slice operations, rather than
native SIMD instructions, external regex libraries, or a C tokenizer ABI.
The representation does not require 64-bit pointers. Model token strings
borrow the GGUF storage, which must outlive the tokenizer. The public
`Tokenizer.load`, `encode`, and `decode` allocator/output-list contract
remains shared across native and wasm32 callers.

## Inspect exact ids

```sh
scripts/fixtures.sh vocab
zig build -Doptimize=ReleaseFast
zig-out/bin/zllm tokenize \
  .fixtures/models/Qwen3.5-0.8B-BF16.header.gguf \
  tools/corpus/tokenizer.jsonl
```

The command emits one `{"ids": [...], "decoded": ...}` object per corpus
record. The metadata-only header works because zllm does not need tensor
weights for tokenization. To compare with the pinned Hugging Face files,
run `zig build check-tokenizer`; see [validation](validation.md).

## Measure zllm

```sh
zig-out/bin/zllm tokenize-bench \
  .fixtures/models/Qwen3.5-0.8B-BF16.header.gguf \
  tools/corpus/tokenizer.jsonl
```

The command accepts a full GGUF or a metadata-only header, consistently
with `tokenize`. It emits one JSON object containing:

| Field | Meaning |
|---|---|
| `engine` | `zllm` |
| `corpus_records` | Number of independent input texts |
| `corpus_bytes` | Sum of the decoded texts' UTF-8 byte lengths for one pass |
| `corpus_tokens` | Number of ids produced by one warm pass |
| `repetitions` | Timed complete corpus passes; excludes warmup |
| `elapsed_ns` | Time spent in the repeated encode loop |
| `bytes`, `tokens` | Total timed input bytes and output ids |
| `mb_per_second` | `bytes / seconds / 1,000,000` (decimal MB, not MiB) |
| `tokens_per_second` | `tokens / seconds` |

The corpus is JSONL, with one object containing a string `text` per LF-
delimited record. Other fields are ignored and empty lines are skipped.
Each string is encoded separately, including empty strings and its exact
embedded whitespace, NUL, or Unicode line separators. Texts are never
joined, trimmed, or split with Unicode-aware `splitlines()`. An absent
corpus or one with zero total text bytes is rejected.

File reading, JSON parsing, model/tokenizer loading, one warm corpus pass,
and JSON output happen outside timing. The output id list retains its
capacity between calls. Any scratch allocation performed by the ordinary
`encode` API remains inside timing; this is not a precomputed-token cache.
The loop runs complete corpus passes until at least 0.5 seconds elapse,
so a single very large pass can extend the sample beyond that target.

## Compare Hugging Face and llama.cpp

```sh
scripts/fixtures.sh model
scripts/fixtures.sh llama
uv run --project tools tools/tokenize_bench.py \
  .fixtures/models/Qwen3.5-0.8B-BF16.gguf \
  .fixtures/hf/Qwen3.5-0.8B/tokenizer.json
```

Use a **full GGUF** for this comparison. The pinned llama.cpp loader checks
tensor file bounds even with `vocab_only`; it rejects zllm's range-read
metadata headers. Neither benchmark includes model loading time.

The harness emits one JSON report with the corpus/tokenizer SHA-256,
platform/Python versions, and per-engine results. It checks the ids of
every independent text against Hugging Face before accepting the
comparison: malformed ids or inconsistent timing/count/rate fields fail;
valid but different ids are reported as `ids_match_hf: false`, with
one-based `mismatched_records`, and cause exit status 1. Such rates are
observations of different tokenization, not equivalent-work speedups.

Hugging Face uses the pinned `tokenizers` Python binding, calling
`Tokenizer.encode(text).ids` for each record with its normal postprocessor
and added-token handling. Its measured time includes Python-call and id-
list conversion overhead. This is the public API comparison, not a claim
about Rust-only throughput or batched/parallel `encode_batch` performance.

The pinned `llama-tokenize` tool performs just one encode and has no warm
repeat timing mode. Timing its subprocess would conflate startup/loading
with encode. Instead the harness compiles `tools/tokenize-bench.cpp`
against `.fixtures/llama.cpp/build/bin/libllama`, building the CPU library
with CMake if missing. The helper loads only the vocabulary, warms once,
and times repeated `llama_tokenize` calls with `add_special=true` and
`parse_special=true`. Length-prefixed UTF-8 input preserves record
boundaries, embedded NULs and empty strings. Compilation, process startup,
IPC, id checking, and result serialization are outside its inner clock.

A missing checkout/compiler or failed build is reported as
`status: "unavailable"` with its reason and **no throughput**. A helper
runtime failure fails the command rather than inventing a result.
`--skip-llama` records an explicit omission. `--cxx`, `--llama-cpp`, and
`--cache` select the compiler, checkout, and helper-output directory.

To keep before/after measurements reproducible, preserve the old CLI
binary and invoke the unchanged benchmark interface on each build:

```sh
uv run --project tools tools/tokenize_bench.py \
  .fixtures/models/Llama-3.2-1B-Instruct-BF16.gguf \
  .fixtures/hf/llama/tokenizer.json \
  --zllm /tmp/zllm-tokenizer-before --label before
uv run --project tools tools/tokenize_bench.py \
  .fixtures/models/Llama-3.2-1B-Instruct-BF16.gguf \
  .fixtures/hf/llama/tokenizer.json \
  --zllm zig-out/bin/zllm --label after
```

These commands print reports to stdout; save them without losing the exit
status. Use the same build mode, model, corpus, and machine, and avoid
concurrent benchmarks. Half-second samples are quick observations, not a
statistical regression threshold. Fixture-free mutations of malformed
corpora, mismatched ids, fabricated work counts/rates, nonfinite values,
and subprocess-timing boundaries live in `tools/test_tokenize_bench.py`.

## Measured corpus results

Recorded on 2026-10-31, on an AMD Ryzen 9 9955HX3D under x86-64 WSL2
Linux `7.2.12-locietta-WSL2-xanmod1`. zllm used Zig
`0.18.0-dev.738`, `ReleaseFast`; Hugging Face used Python 3.14.3 and
`tokenizers` 0.23.3. llama.cpp used the pinned
`e09a78a25dec3f42175ad59ecc540a54bed0b9b2` CPU library with the C++17
`-O3` benchmark helper. The corpus contained 140 texts and 26,525 UTF-8
bytes; its SHA-256 was
`9eccf8d3fb13b3d99f2b593b1a04a45a3aaecdd7954978d55cc97d2cbae3f2c6`.

Each cell below is **MB/s / million tokens/s**, from one warmed sample of
at least half a second. The old zllm binary was preserved before replacing
the engine. Both old/new zllm samples used GGUF metadata headers; the
external-reference samples used the matching full GGUFs. Inputs and encode semantics were
the same; no model-loading time is included. These are separate, unpinned
samples on a shared machine, not confidence intervals or universal speed
claims.

| Model tokenizer | Old zllm | New zllm | Hugging Face Python API | llama.cpp |
|---|---:|---:|---:|---:|
| Qwen 3.5 0.8B | 13.681 / 4.054 | 37.891 / 11.228 | 4.940 / 1.464 | 4.647 / 1.416 **different ids** |
| Qwen 3 0.6B | 14.427 / 4.018 | 39.598 / 11.028 | 4.193 / 1.168 | 4.642 / 1.331 **different ids** |
| Llama 3.2 1B | 2.704 / 0.801 | 88.444 / 26.205 | 4.382 / 1.298 | 9.262 / 2.744 |
| Granite 4.2 3B | 7.370 / 2.404 | 50.245 / 16.389 | 3.125 / 1.019 | 4.990 / 1.628 |

The observed new/old throughput ratios are 2.77×, 2.74×, 32.71×, and
6.82× respectively. The new samples completed 715 / 747 / 1,668 / 948
passes in 500,528,471 / 500,384,931 / 500,245,210 / 500,457,172 ns.
The old samples completed 258 / 272 / 51 / 139 passes in
500,215,884 / 500,094,240 / 500,322,358 / 500,276,075 ns. These counts
and the per-pass sizes below allow the rounded rates to be recomputed.

Both old and new zllm outputs matched every Hugging Face id on this corpus.
Hugging Face produced 7,860 / 7,387 / 7,859 / 8,652 tokens per pass in
table order. llama.cpp matched Llama and Granite exactly, but produced
8,084 and 7,605 tokens for the Qwen tokenizers. Both Qwen comparisons
differed on records 47–49 and 82–88, the corpus's decomposed/composed
Unicode normalization probes. Those llama.cpp rates are retained as
observations, explicitly **not** an equivalent-tokenization comparison.
