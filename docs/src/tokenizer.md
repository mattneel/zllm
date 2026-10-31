# Tokenization and encode throughput

zllm loads tokenizers from GGUF `tokenizer.ggml.*` metadata, not from a
separate `tokenizer.json`. The same tokenizer API serves the native CLI and
the browser core. Its output ids retain the GGUF/Hugging Face vocabulary
indices: merge ranks choose operations, but never replace token ids.
The supported byte-level BPE families are Qwen 3.5, Qwen 3, Llama 3 and
Granite 4. Nanbeige's SentencePiece scored-BPE and Gemma4's BPE also use
their own normalization, initial-symbol, added-token and decoder profiles.
This is not a general loader for every Hugging Face tokenizer family, nor
does tokenizer support imply forward-pass support for those model weights.

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

### SentencePiece-style BPE profiles

Nanbeige 4.3's GGUF identifies its tokenizer as `model=llama`,
`pre=default`. Its HF JSON has no normalizer; Metaspace replaces ASCII
spaces with `▁` without splitting, and adds a dummy `▁` only when the
ordinary segment starts at the beginning of the original input. Text
after an added token does not acquire a new dummy prefix. Its 166,912-entry
GGUF vocabulary includes HF's 166,100 base entries, added ids through
166,106 and 805 trailing padding entries.

Nanbeige's GGUF stores scores rather than an explicit merge table. The
engine reconstructs HF's merge ranks from valid vocabulary splits:
within each output token, splits are ordered by left/right token id;
globally they are ordered by descending score and then descending
Unicode lengths of the left and right symbols. This reconstructs all
296,888 merges of the pinned HF tokenizer, rather than approximating them
with a score-only merge heap. The ordinary merge engine still emits
original vocabulary ids. The `<|im_start|>` BOS id 166,100 agrees with
the HF postprocessor, including on empty inputs.

Gemma4's 262,144-entry GGUF uses `model=gemma4` and provides explicit
merges. Its normalizer replaces spaces with `▁`, without a dummy prefix
or NFC. Initial symbols are Unicode scalars, with byte-token fallback
when a scalar has no ordinary vocabulary token. Added tokens are found
before normalization and retain their literal spelling. The pinned
original Google HF JSON postprocessor adds no BOS; `Tokenizer.encode`
and CLI `tokenize` follow that raw-tokenizer behavior despite GGUF's
`add_bos_token=true` flag.

Inference uses the separate `Tokenizer.encodePrompt` entry point on
both native and WASM paths. It honors GGUF's BOS policy and avoids
duplicating a BOS already present literally at the start of a rendered
chat template. For Gemma4, raw `a` encodes as `[a]`, while inference
`a` and rendered `<bos>a` both encode as `[bos, a]`. Nanbeige's raw
encoding still duplicates an explicit leading `<|im_start|>` as HF's
postprocessor does; prompt encoding avoids that duplicate. Tokenizer
benchmarks and HF parity checks exercise raw `encode`, not inference
prompt policy.

Both decoders replace vocabulary `▁` with spaces **before** byte
fallback, then fuse token text. Consecutive byte-fallback tokens must be
decoded as one run: a valid UTF-8 run becomes text; an invalid run emits
one U+FFFD per byte, rather than retaining a valid suffix. For example,
byte tokens `FF C3 A9` decode to three replacements, while `C3 A9`
decode to `é`. Bytes `E2 96 81` decode to a literal `▁`, not a space,
because replacement precedes fallback. Nanbeige additionally strips one
leading ASCII space from the **whole decoded stream**, not each token
or ordinary segment. Its prepended BOS means a following dummy-prefix
space is retained. Independent per-token decoding is not equivalent to
decoding a complete id sequence for these profiles.
Streaming callers use `src/tokenizer/stream.zig`'s `Stream.push` rather
than repeatedly invoking `Tokenizer.decode` on one id. The stream holds
fallback runs until a non-byte token or `flush`, and tracks whether the
initial-space rule has already been applied. Call `flush` when generation
ends without a stop so a final fallback run is not lost.

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

Without an explicit input variant, use a **full GGUF** for this comparison.
The pinned llama.cpp loader checks tensor file bounds even with
`vocab_only`; the original range-read headers are unavailable to that
loader. Neither benchmark includes model loading time.

The Nanbeige/Gemma4 `vocab` fixtures deliberately contain no weights
(Nanbeige's separate full-model target is `scripts/fixtures.sh nanbeige`).
To measure llama.cpp's vocabulary tokenizer without fetching weights,
opt in to an explicit derived input:

```sh
scripts/fixtures.sh vocab
uv run --project tools tools/tokenize_bench.py \
  .fixtures/models/Nanbeige4.3-33B-Q5_K_M.header.gguf \
  .fixtures/hf/nanbeige/tokenizer.json --llama-metadata-only
uv run --project tools tools/tokenize_bench.py \
  .fixtures/models/gemma-4-31B-it-Q4_K_M.header.gguf \
  .fixtures/hf/gemma4/tokenizer.json --llama-metadata-only
```

`--llama-metadata-only` creates a temporary copy with only the little-endian
tensor-count `u64` at byte offset 8 changed to zero. All original metadata
and remaining bytes are preserved; no tensors or weights are synthesized,
and the pinned original is never modified. The temporary file is removed
after the helper exits. zllm still uses the original header. The llama.cpp
result records `input_variant: "metadata-only"` so these runs cannot be
mistaken for measurements of the original GGUF. Mismatched ids still
produce exit status 1; the option does not change BOS or other metadata
to force agreement.

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
subprocess-timing boundaries, and metadata-only derivation invariants
live in `tools/test_tokenize_bench.py`.

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
first four external-reference samples used the matching full GGUFs;
Nanbeige and Gemma4 used the explicitly derived metadata-only llama.cpp
input described above. No model-loading time is included. These are
separate, unpinned samples on a shared machine, not confidence intervals
or universal speed claims.

| Model tokenizer | Old zllm | New zllm | Hugging Face Python API | llama.cpp |
|---|---:|---:|---:|---:|
| Qwen 3.5 0.8B | 13.681 / 4.054 | 37.891 / 11.228 | 4.940 / 1.464 | 4.647 / 1.416 **different ids** |
| Qwen 3 0.6B | 14.427 / 4.018 | 39.598 / 11.028 | 4.193 / 1.168 | 4.642 / 1.331 **different ids** |
| Llama 3.2 1B | 2.704 / 0.801 | 88.444 / 26.205 | 4.382 / 1.298 | 9.262 / 2.744 |
| Granite 4.2 3B | 7.370 / 2.404 | 50.245 / 16.389 | 3.125 / 1.019 | 4.990 / 1.628 |
| Nanbeige 4.3 33B | unsupported | 12.245 / 4.110 | 6.040 / 2.028 | 3.926 / 1.320 **different ids; derived metadata-only** |
| Gemma 4 31B IT | unsupported | 11.056 / 3.503 | 5.586 / 1.770 | 3.585 / 1.155 **different ids; derived metadata-only** |

For the first four models, the observed new/old throughput ratios are 2.77×, 2.74×, 32.71×, and
6.82× respectively. The new samples completed 715 / 747 / 1,668 / 948
passes in 500,528,471 / 500,384,931 / 500,245,210 / 500,457,172 ns.
The old samples completed 258 / 272 / 51 / 139 passes in
500,215,884 / 500,094,240 / 500,322,358 / 500,276,075 ns. These counts
and the per-pass sizes below allow the rounded rates to be recomputed.

For the first four models, both old and new zllm outputs matched every
Hugging Face id on this corpus. Hugging Face produced
7,860 / 7,387 / 7,859 / 8,652 tokens per pass in table order.
llama.cpp matched Llama and Granite exactly, but produced
8,084 and 7,605 tokens for the Qwen tokenizers. Both Qwen comparisons
differed on records 47–49 and 82–88, the corpus's decomposed/composed
Unicode normalization probes. Those llama.cpp rates are retained as
observations, explicitly **not** an equivalent-tokenization comparison.

Nanbeige and Gemma4 have no old-zllm throughput: those tokenizer families
were unsupported, not measured at zero speed. New zllm matched all 140
HF id sequences and decoded texts for each, producing 8,904 and 8,404
tokens per pass. Their zllm samples completed 231 / 209 passes in
500,407,493 / 501,409,729 ns; the paired HF samples completed 114 / 106
passes in 500,640,478 / 503,355,039 ns.

The original header-only inputs remain **unavailable** to the pinned
llama.cpp loader; the table's new llama.cpp rates are measured only on
the explicit metadata-only variants, not invented full-model results.
Those samples completed 74 / 68 passes in 500,006,917 / 503,117,042 ns.
Nanbeige's llama.cpp output had 8,921 tokens per pass and mismatched 23
records. Gemma4 had 8,544 tokens and mismatched all 140 records: llama.cpp
honors the GGUF BOS flag, unlike the pinned raw HF tokenizer. Neither
llama.cpp rate represents equivalent tokenization work or an exact-id
comparison speedup.
