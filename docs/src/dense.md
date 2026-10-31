# Dense transformers

Qwen 3, Llama 3 and Granite 4 use the GGUF architectures `qwen3`, `llama`
and `granite`. These are dense decoder-only transformers: every layer uses
full causal attention, not the recurrent/attention schedule of the
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
