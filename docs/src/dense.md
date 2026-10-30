# Dense transformers

Qwen 3, Llama 3 and Granite 4 (GGUF `general.architecture` `qwen3`, `llama`
and `granite`) are dense decoder-only transformers. Every layer is

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

A Granite multiplier of 0 or absent means none, as in llama.cpp. llama.cpp
also accepts projection biases in these graphs; none of the supported
models has them, and zllm refuses a file that does rather than ignore them.

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

## Chat

zllm's chat format is Qwen 3.5's; `chat` and the browser's `messages`
requests refuse other architectures (`NoChatTemplate`). Give them a prompt
with `generate`, or token ids.
