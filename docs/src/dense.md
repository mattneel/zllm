# Dense transformers

Qwen 3, Llama 3, Nanbeige 4.3 and Granite 4 use the GGUF architectures
`qwen3`, `llama` (also Nanbeige) and `granite`. These are dense decoder-only
transformers: every layer uses full causal attention, not the
recurrent/attention schedule of the
[Qwen 3.5 hybrid family](qwen35.md), and not
[Gyro-Coder's mixture of experts](gyro.md). Every layer is

    x += attn(rmsnorm(x) · attn_norm)        GQA softmax attention
    x += ffn(rmsnorm(x) · ffn_norm)          SwiGLU: down(silu(gate(h)) · up(h))

with the logits `output(rmsnorm(x) · output_norm)`, where `output` is
`token_embd` when the file has no `output.weight`. `src/graph/dense.zig`
reads what differs between the three from the GGUF metadata; llama.cpp's
`src/models/{qwen3,llama,granite}.cpp` are the authority.

| | Qwen 3 | Llama | Granite |
|---|---|---|---|
| Q and K | per-head RMSNorm (`attn_q_norm`, `attn_k_norm`) before RoPE | as projected | as projected |
| RoPE pairs | NeoX: (i, i + n_rot/2) | normal: (2i, 2i + 1) | normal |
| RoPE angle | pos · base^(−2i/n_rot) | divided by `rope_freqs[i]` when present (Llama 3.1+ scaling) | as Llama |
| K·Q scale | 1/√head_dim | 1/√head_dim | `attention.scale` |
| Multipliers | none | none | embeddings × `embedding_scale`; each branch × `residual_scale` before its add; logits ÷ `logit_scale` |

For Granite, absent or zero attention, embedding and residual scales mean
no extra multiplier. The current parser requires a nonzero `logit_scale`
and uses its reciprocal. The equations above show the unscaled case.

Projection biases are outside the supported dense-model subset. The CPU
reference checks for them in the first layer and refuses that file rather
than implementing the biased graph. The pinned GPU fixtures have no such
biases; do not treat the architecture name as a promise of arbitrary
checkpoint compatibility.

## Supported files and evidence

The pinned small-model coverage is BF16 Qwen3-0.6B,
Llama-3.2-1B-Instruct and Granite 4.2 3B. Granite 4.2 30B also has a
recorded Q4_K_M validation. These fixture results establish particular
graphs and tensor formats, not every model in those families.

- [Dense-model validation](validation.md#dense-models) records CPU
  comparisons with llama.cpp and GPU comparisons with the CPU reference.
- [Granite 4.2 30B](validation.md#granite-42-30b) covers the large mixed
  quantization file and its activation-rounding differences.
- [Tokenizer validation](validation.md#tokenizer) covers the different
  Qwen, Llama and Granite pre-tokenizers; a shared attention graph does not
  imply a shared tokenizer.
- [Weight formats](precision.md#weight-formats) distinguishes executable
  tensor types from types the GGUF reader can only index.

## On the GPU

The dense models run as Qwen 3.5's step program with every layer full
attention (`plan/qwen35.zig`, `Model.uploadDense`): no DeltaNet state, no MTP
block. Its attention takes three switches the dense models set:

- `rope_append`: query heads `head_dim` apart rather than interleaved with
  output gates (`gated`), Q/K RMSNorm on or off (`qk_norm`), normal or NeoX
  pairs (`normal`).
- `attn_split` and `attn_combine`: the same query layout, and no output
  gate (`gated`).
- The scale is the model's.

RoPE angles come from a table the host builds. Llama 3's `rope_freqs` is a
tensor in the file's data, which the browser does not hold in memory, so
the load reads it back through the GPU (`write_file`, then `read_buffer`)
before it builds the program.

The program has no embedding, residual or logit multiplier. Granite 4.2's
are all 1; a model with others (Granite 3) is refused on the GPU. The CPU
reference runs them.

The same dense plan runs through WebGPU (browser or native Dawn) and
Agility's native D3D12 implementation. It has the shared plan's prefill
and decode paths, sampler, prefix reuse and device recovery, but no MTP
block. KV storage can be f32, f16 or FP8 E4M3; f16 remains the generation
default. The lower FP8 diagnostic bounds are not authorization to change
that default; see [precision](precision.md#quality-gated-precision-ladder)
and [GPU forward validation](validation.md#gpu-forward-pass).

## Nanbeige 4.3 33B

**Implementation and chat are exercised, but the full model quality gate
does not pass.** On both native backends, `check-quant` passes all 200/200
strict decode positions (worst relative L2 4.52e-5), but fails ten
first-layer comparisons: GPU prefill versus the rounded CPU reference has
worst sequence median relative L2 4.11e-3, and rounded reference versus
llama.cpp 1.67e-3, against the unchanged 1e-4 bound. Activation-rounding
sensitivity ratios pass (at most 1.09), but that does not excuse the failed
absolute gate. The scalar/active Q8_0 and RMSNorm discrepancies were
corrected; the residual first-layer discrepancy remains unresolved.

Fetch the opt-in full fixture with `scripts/fixtures.sh nanbeige`:
`AtomicChat/Nanbeige4.3-33B-GGUF` revision
`560e151e11dfdd4b008d394f6e16bf196a952bd0`,
`Nanbeige4.3-33B-AD-Q4_K_M.gguf` (21,161,159,680 bytes), SHA-256
`409c92c4355c81928f0f2de5d49481fe7e4d0fa4d381aa28eddee0dc4b358c49`.
It is deliberately not downloaded by `all` or `dense`.

The file declares `general.architecture=llama`: 52 layers, embedding width
7,168, FFN width 22,016, 56 query heads and 8 KV heads, each 128 wide.
Its 128 rotary dimensions use normal-pair RoPE with base 120,000,000;
there is no rope-scaling metadata or `rope_freqs.weight` tensor to add.
The output is **untied**: `token_embd.weight` is IQ4_XS and the separate
`output.weight` is Q6_K. Other matrices mix Q4_K, Q5_K, Q6_K and Q8_0;
norm vectors are F32. No alternate graph or architecture alias is needed.

The shared plan uploads the embedding table in row chunks and selects the
IQ4_XS embedding decoder on WebGPU and Agility. It does not require an
IQ4_XS projection kernel. The 166,912-entry SentencePiece vocabulary and
the file's own chat template use the existing tokenizer and Jinja paths.

The browser does not map the entire model into wasm memory: it range-reads
the header and streams bounded upload pieces. File offsets remain
JavaScript safe integers, so the 21 GB file size is not itself a wasm32
limit. Total GPU memory remains a separate constraint: an 8,192-token F16
KV cache alone uses 1,744,830,464 bytes, in addition to laid-out weights,
activation buffers and upload staging. Remote caching also needs enough
OPFS quota; insufficient quota falls back to uncached HTTP ranges.

The file's tensor index implies 21,406,085,120 bytes (19.936 GiB) of laid-out
resident weights before allocator overhead, versus 21,157,154,816 bytes
of tensor payload on disk. These are calculated layout sizes, not measured
peak device memory.

On 2026-10-31, the RTX 5090 Laptop (24,463 MiB total device memory) produced
the following `zllm benchmark` observations, over 1,024 prompt tokens and
128 greedy decode tokens. The native quality ladder selected **F32 KV**:

| Host | Context | Prefill tok/s | Decode tok/s | Sampled peak device MiB |
|---|---:|---:|---:|---:|
| Dawn D3D12 | 8,192 | 36.78 | 9.72 | 23,667 |
| Agility | 8,192 | 22.68 | 12.30 | 23,865 |
| Dawn D3D12 | 4,096 | 194.15 | 22.78 | 23,447 |
| Agility | 4,096 | 33.20 | 21.39 | 23,561 |

The 8K context is close to the device-memory limit; lowering capacity to
4K preserves F32 KV while reducing its allocation. The rates are measured,
not a claim of equivalent scheduler chunking: Dawn's 4K prefill used 33
steps, Agility's 129. Loads from the WSL file took 99.5–145.1 seconds.
Memory is sampled every 200 ms with `nvidia-smi` and includes desktop and
other processes; it is not an isolated allocation counter.

Native chat produced identical text on both backends at greedy sampling,
with 160 output tokens limited by `--max-tokens` (23.3 / 21.7 tok/s).
Chrome 156 loaded the complete file from uncached localhost HTTP ranges in
52.50 seconds at its default 8K/F16 context, answered `2 + 2` with `4` and
an EOS stop, and a separate 128-token chat sample decoded at 22.64 tok/s.
Its sampled peak device memory was 23,618 MiB (986 MiB before loading).
This verifies a 21 GB browser load, not a Hugging Face network benchmark
or approval of the failed model-parity gate. The demo dropdown includes
the pinned file with that quality warning.

The complete structured observations are in
[`benchmarks/nanbeige4.3-33b.json`](https://github.com/mattneel/zllm/blob/master/benchmarks/nanbeige4.3-33b.json);
the arithmetic diagnosis and unchanged-bound regression results are in
[`benchmarks/ggml-runtime-fidelity.json`](https://github.com/mattneel/zllm/blob/master/benchmarks/ggml-runtime-fidelity.json).


## Performance

The recorded `zllm benchmark` result for Granite 4.2 30B at Q4_K_M
(17.7 GB), on the RTX 5090 Laptop through Dawn D3D12, is prefill
209 tok/s over 1,024 tokens and decode 25.1 tok/s.
A decode step takes 40.7 ms, 82% of it in the Q4_K and Q6_K matrix-vector
multiplies (about 560 GB/s over 13 GB of weights). Loading took 106 s, nearly
all of it reading the file from the WSL disk through `\\wsl.localhost`.
Its checks are in [Validation](validation.md#granite-42-30b).

## Chat

Qwen 3.5–3.8 and the Gyro-Coder chat in zllm's fixed Qwen template
([Generation](generation.md)). The dense models chat in their own
(`tokenizer.chat_template` in the GGUF), which `src/chat/jinja.zig` renders as
transformers' `apply_chat_template` does ([Validation](validation.md#chat-template)):
the messages in the OpenAI form, the tools offered, `add_generation_prompt`,
the BOS and EOS tokens' text, `enable_thinking` and `reasoning_effort` when
set, and in the browser any `chat_template_kwargs`, as vLLM takes them.
`strftime_now` formats the current UTC date (Llama 3's "Today Date"); the
browser's shim sends its clock with each request. A model without a template
refuses chat requests (`NoChatTemplate`).
