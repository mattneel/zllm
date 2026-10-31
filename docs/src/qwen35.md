# Qwen 3.5 hybrid: GGUF semantics

The `qwen35` path covers the Qwen 3.5, 3.6 and 3.8 hybrid family: Gated
DeltaNet layers alternate with full softmax attention, with a dense SwiGLU
FFN after either mixer. The loader selects this path by GGUF
`general.architecture`, not the model's display name. Qwen3.8-Flash-Next
Gyro-Coder is a different architecture, `qwen4exp`; see
[Gyro-Coder](gyro.md). Qwen 3's all-attention graph is described in
[Dense transformers](dense.md).

The pinned evidence covers Qwen3.5-0.8B BF16 and the Qwen3.8-27B mixed
Q3_K_M file. This is not a claim that every checkpoint carrying a Qwen
family name or every GGUF quantization is supported. See
[precision](precision.md#weight-formats) for accepted weight types and
[validation pins](validation.md#pins) for the exact files.

This page fixes what the `qwen35` GGUF tensors mean, as written by the pinned
llama.cpp converter (`conversion/qwen.py`, classes `Qwen3NextModel`,
`_LinearAttentionVReorderBase` and `Qwen3_5TextModel`) and consumed by its
graph (`src/models/qwen35.cpp`, `delta-net-base.cpp`). The converter changes
some weights relative to the Hugging Face checkpoint. Implementing the Hugging
Face equations directly on GGUF weights applies those changes twice, and two
implementations that share the mistake still agree with each other. That's
why the reference is checked against llama.cpp itself
([reference-forward validation](validation.md#reference-forward-pass)).

The implementation is `src/graph/qwen35.zig` (hyperparameters),
`src/ref/qwen35.zig` (the f32 reference) and `src/plan/qwen35.zig` (the
GPU step program shared by WebGPU and Agility).

## Metadata

The values below describe the pinned 0.8B fixture, not constants for every
model in the family. Dimensions and the layer schedule come from metadata.

| Key (`qwen35.` prefix) | 0.8B | Meaning |
|---|---|---|
| `block_count` | 25 | Decoder layers **plus** MTP layers |
| `nextn_predict_layers` | 1 | Trailing MTP (`blk.N.nextn.*`) layers, not run by the main pass |
| `embedding_length` | 1024 | Model width |
| `feed_forward_length` | 3584 | SwiGLU hidden width |
| `attention.head_count` / `head_count_kv` | 8 / 2 | Full-attention query / KV heads (GQA) |
| `attention.key_length` = `value_length` | 256 | Attention head dim |
| `attention.layer_norm_rms_epsilon` | 1e-6 | Every RMSNorm, and the q/k L2 norm |
| `rope.dimension_count` | 64 | Rotated dims per head (`partial_rotary_factor` 0.25) |
| `rope.freq_base` | 1e7 | RoPE base |
| `rope.dimension_sections` | [11, 11, 10, 0] | MRoPE sections (see RoPE below) |
| `ssm.conv_kernel` | 4 | DeltaNet causal conv width |
| `ssm.state_size` | 128 | DeltaNet per-head key/value dim |
| `ssm.group_count` | 16 | DeltaNet key heads |
| `ssm.time_step_rank` | 16 | DeltaNet value heads |
| `ssm.inner_size` | 2048 | value heads × head dim |
| `full_attention_interval` | 4 | Layer *i* is full attention iff (*i* + 1) mod 4 = 0 |
| `attention.recurrent_layers` | absent | Optional per-layer bool array; overrides the interval rule |

For the 0.8B fixture, the tokenizer is `tokenizer.ggml.model = gpt2` with
`pre = qwen35`, `add_bos_token = false` and EOS `<|im_end|>` (248046).
Its LM head is tied: there is no `output.weight`, so logits use
`token_embd.weight`. The loader also accepts a separate `output.weight`.

## Converter transforms

| Hugging Face | GGUF | Transform |
|---|---|---|
| `linear_attn.A_log` | `blk.i.ssm_a` | **−exp(A_log)**, stored F32 |
| `linear_attn.dt_bias` | `blk.i.ssm_dt.bias` | renamed |
| `linear_attn.conv1d.weight` [C, 1, K] | `blk.i.ssm_conv1d.weight` ne [K, C] | squeezed; taps contiguous per channel |
| every `*norm.weight` **except** `linear_attn.norm.weight` | `attn_norm`, `post_attention_norm`, `attn_q_norm`, `attn_k_norm`, `output_norm` | **+1** (Hugging Face's `Qwen3_5RMSNorm` scales by 1 + w) |
| `linear_attn.norm.weight` | `blk.i.ssm_norm.weight` | unchanged (gated norm scales by w) |
| `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, the V channels of `conv1d`, and `out_proj` columns | same tensors | **V heads reordered grouped → tiled** when key heads ≠ value heads |

In the V-head reorder, Hugging Face groups value heads by key head
(`[k0v0, k0v1, k1v0, k1v1]`) and the converter tiles them
(`[k0v0, k1v0, k0v1, k1v1]`). At runtime value head *j* then reads key head
*j* mod `group_count`: llama.cpp's `iq1 = iv1 % neq1`, and the same in
`ref`. For the 0.8B model the key and value head counts are equal (16), so the
reorder is the identity. Larger models with more value heads depend on it.

Norms and small recurrent parameters (`*_norm`, `ssm_a`, `ssm_dt.bias`,
`ssm_conv1d`) are **F32 even in quantized files**. Never infer a tensor's type
from the file name; read the per-tensor type from the index.

## Layer

Every decoder layer is pre-norm with two residuals:

$$
\begin{aligned}
x &\leftarrow x + \text{mixer}(\text{rms}(x) \odot w_\text{attn\_norm}) \\
x &\leftarrow x + W_\text{down}\big(\text{silu}(W_\text{gate}\,h) \odot W_\text{up}\,h\big),\quad h = \text{rms}(x) \odot w_\text{post\_attention\_norm}
\end{aligned}
$$

with $\text{rms}(x) = x / \sqrt{\text{mean}(x^2) + \varepsilon}$. The mixer is
full attention or Gated DeltaNet, according to the layer schedule. After the
last layer comes $\text{rms}(x) \odot w_\text{output\_norm}$, then the LM head
(tied for the 0.8B fixture).

### Full attention (layers 3, 7, 11, …)

1. `attn_q` produces `2·head_dim` values per head, **interleaved by head**:
   `[q_0 (256), gate_0 (256), q_1, gate_1, …]`. `attn_k` and `attn_v` produce
   `head_dim` per KV head.
2. RMSNorm each query head with `attn_q_norm`, and each key head with
   `attn_k_norm` (weights over `head_dim`).
3. RoPE on q and k (below).
4. Causal softmax attention, scaled by $1/\sqrt{\text{head\_dim}}$. Query head *h*
   uses KV head ⌊*h* / (`head_count` / `head_count_kv`)⌋.
5. Multiply by **sigmoid(gate)**, then `attn_output`.

### RoPE

The rope type is IMROPE (interleaved multimodal RoPE). For text, all position
components equal the token position. In `ggml_mrope_cache_init` with
sections [11, 11, 10, 0], every rotary pair then gets the same angle as plain
**NeoX** RoPE: the first `n_rot` = 64 dims rotate as pairs (*i*, *i* + 32)
by $p \cdot \text{base}^{-2i/64}$, and dims 64–255 pass through unchanged.

### Gated DeltaNet (all other layers)

Let *H* = 16 heads of dim *d* = 128.

1. `attn_qkv` produces the conv input `[q (H·d), k (H·d), v (H·d)]` = 6144
   channels. `attn_gate` produces *z* (H·d).
2. Run a causal depthwise conv over time, per channel: four taps on the current
   and previous three inputs (zero before the sequence start), then **SiLU**.
3. Split into q, k, v per head. **L2-normalize** q and k per head:
   $x / \sqrt{\sum x^2 + \varepsilon}$.
4. Compute $\beta = \sigma(W_\beta h)$ and
   $g = \text{softplus}(W_\alpha h + \text{ssm\_dt.bias}) \cdot \text{ssm\_a}$
   (negative, since `ssm_a` already holds −exp(A_log)).
5. Apply the delta rule per head with state $S \in \mathbb{R}^{d \times d}$ (key × value),
   starting at zero:

$$
S \leftarrow e^{g} S, \qquad
\delta = \beta\,(v - S^\top k), \qquad
S \leftarrow S + k\,\delta^\top, \qquad
o = S^\top q / \sqrt{d}
$$

6. Gated norm per head: $\text{rms}(o) \odot w_\text{ssm\_norm} \odot \text{silu}(z)$.
   It is **SiLU**, not sigmoid.
7. Apply `ssm_out`.

## Verified against llama.cpp

These semantics are checked by `zig build check-forward`. For the pinned
0.8B, the recorded result is 816/818 top-1 agreement with llama.cpp
(99.76%). Deliberately dropping the sigmoid attention output gate reduces
it to 2.4%, demonstrating that agreement is sensitive to the graph rather
than just the file format. See the
[reference forward pass](validation.md#reference-forward-pass) for the
per-layer and final-state criteria.

The quantized 27B needs a separate comparison: its GPU prefill path rounds
multiply operands to 8-bit blocks, while the ordinary CPU reference keeps
f32 activations. The rounded reference separates that difference from a
semantic error; the generic llama.cpp model thresholds do not hold for this
file. See [the 27B validation](validation.md#the-27b), not just the small
BF16 model's result.

## GPU execution and state

The GPU program has distinct prefill and decode multiply paths. Both run
the same attention and recurrence semantics, with weights decoded in the
shader and f32 accumulation. Full-attention layers keep a KV cache;
DeltaNet layers keep an f32 recurrent matrix and causal convolution
history. Changing the KV format does not quantize the DeltaNet state.

Generation defaults to an f16 KV cache. f32 and FP8 E4M3 caches are
explicit choices; FP8 support is not an automatic model-quality approval
or a default change. See [precision](precision.md#kv-cache-formats) and
[GPU forward validation](validation.md#gpu-forward-pass).

The engine checkpoints recurrent state for prefix reuse and recomputes
from a valid checkpoint when needed. It does not rewind DeltaNet by merely
discarding attention cache positions. See
[generation state](generation.md#state-and-prefix-reuse).

## Multi-token prediction

Trailing `nextn` layers are excluded from the main decoder layer count.
When the file carries the supported single MTP block, the loader binds it
separately: the final hidden state at a position and the next token's
embedding produce a draft for the following token. The engine can verify
that draft against the main model rather than treating it as an unchecked
output.

MTP support and default speculation are separate decisions. The block must
be present and the device's storage-offset alignment must allow its
verification layout. The engine enables speculation by default only when
the draft's weight cost is small enough relative to the main model;
`--speculate` and `--no-speculate` control it explicitly. A small model can
have a valid MTP block without benefiting from using it. See
[speculative decoding](generation.md#speculative-decoding) and
[engine validation](validation.md#engine).
