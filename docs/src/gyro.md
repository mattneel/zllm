# Qwen3.8-Flash-Next (Gyro-Coder)

Gyro-Coder uses GGUF architecture `qwen4exp`. It is not the `qwen35`
hybrid graph with larger dimensions: it adds hyper-connected residual
streams, routed and shared experts, sparse attention and a hashed
per-layer embedding (PLE) table. Its reference is AgentionAI's runtime,
not stock llama.cpp, which cannot load this file.

The recorded model is `Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf`: a
125B mixture-of-experts model with 256 experts per layer, in a 75.8 GB
file. The file name is not the routed experts' tensor type: the GPU loader
requires `TQ2_T` expert tensors. The exact model revision, reference fork
and results are in
[Gyro-Coder validation](validation.md#qwen38-flash-next-gyro-coder).

## Implementation and execution

| Source | Responsibility |
|---|---|
| `src/graph/qwen4exp.zig` | Metadata, dimensions, layer schedule and supported graph constraints |
| `src/ref/gyro/` | Trellis decoders, codebook and Hadamard rotation |
| `src/ref/qwen4exp/model.zig` | Reference tensor binding into the mapped GGUF |
| `src/ref/qwen4exp/ple.zig` | PLE row hashing and token history |
| `src/ref/qwen4exp/engine.zig` | CPU reference graph |
| `src/plan/qwen4exp.zig` | Weight upload, GPU steps, checkpoints and PLE file reads |
| `src/kernels/` and `src/agility/kernels/` | WGSL and HLSL implementations |

The GPU program runs on WebGPU and Agility through the common engine.
It uses decode kernels only, including for prompt ingestion, in steps of
up to 4 tokens. Its context cap is 32,768 positions; that is an
implementation cap, not a claim about every context advertised upstream.
It has no MTP block or speculative decoding path.

## Hyper-connections

The released model maintains four residual streams. Token embeddings
initialize identical copies, but subsequent layer outputs change them
independently. Before attention and before the expert FFN, a learned
low-rank mixer forms the block's input:

- RMS-normalize each stream with its own weight row.
- Project the concatenated streams down, apply SiLU after dividing by the
  stream count, then project up to per-stream gates.
- Take the mean of the normalized streams weighted by sigmoid gates.
- Compute separate injection weights from the normalized streams. Add
  the block output back to each residual stream, scaled by
  `2 * sigmoid(inject / stream_count)`.

The final output uses a hyper-connection mixer without residual
injection, then `output.weight`. These divisions and gates are model
semantics, not optional fusion optimizations. On the GPU, injection rows
share the down projection's multiply; the fusion pass may fold the SiLU
operation into the up projection. Fused and unfused programs must remain
bitwise identical in the
[Gyro validation gate](validation.md#qwen38-flash-next-gyro-coder).

## Recurrent and sparse-attention layers

Metadata selects the recurrent/attention schedule. Recurrent layers use
Gated DeltaNet with causal convolution, SiLU, L2-normalized queries and
keys, decay and delta updates as in the
[hybrid model](qwen35.md#gated-deltanet-all-other-layers). A critical
difference is the final recurrent output gate: Gyro-Coder uses
**sigmoid**, not Qwen 3.5's SiLU.

Attention layers use Q/K RMSNorm, text-position NeoX RoPE, GQA and a
sigmoid output gate interleaved with each query head. A learned indexer
chooses which cached positions each query may attend to:

- Raw indexer keys enter a separate f16 cache.
- Completed blocks are averaged, normalized and rotated at the block's
  start position. Indexer queries are normalized and rotated at the
  query position.
- Each block's score is the sum over indexer heads of `relu(q · key)`.
  Higher scores win; ties use the lower block index.
- The query keeps its incomplete tail and then cells from the ranked
  blocks up to the metadata-derived width, `top_k + compress_ratio - 1`.
  The last selected block can be partial.

For the released model this width is 2,051 visible cells. Below it,
attention selects every causal position. Beyond it, sparse selection is
part of the graph, not an optional approximation chosen by zllm. Passing
only a short prompt would not establish that path's correctness. The
recorded 6,430-token retrieval prompt exercises it natively and in Chrome;
see [the long-prompt result](validation.md#qwen38-flash-next-gyro-coder)
and the [sparse kernel checks](validation.md#gpu-kernels).

The attention KV format is selectable independently of the indexer's
f16 key cache. See [precision](precision.md#kv-cache-formats); FP8
implementation support alone is not evidence of Gyro model-quality parity.

## Routed experts and Gyro storage

The router computes a softmax, selects the metadata-specified experts and
renormalizes their weights, applying `expert_weights_scale` when present
and nontrivial. Selected experts run a SwiGLU FFN. A separate shared
expert contributes through a sigmoid gate, and the routed and shared
outputs are summed before residual injection.

Routed expert weights are stored in a Hadamard-rotated basis with sign
vectors declared in `prism.hadamard.*`. The input must be rotated into
that basis before the gate/up products, and the intermediate activation
before the down product. Decoding the trellis bytes without the required
rotation does not implement the model.

The CPU decoder supports all eight Gyro types: `TQ2_T`, `TQK6`, `TQK7`,
`TQP5`, `TQP6`, `TQP7`, `TQK5` and `TQK4`. The GPU expert path currently
supports only `TQ2_T`. General GGUF type recognition and CPU decoding are
not promises of a GPU kernel for every Gyro quantization. See
[weight formats](precision.md#weight-formats).

## PLE and model memory

The 50.7 GiB PLE table remains in the model file rather than occupying
GPU memory. Each token hashes its token history to 16 table rows, reads
those Q8_0 rows, concatenates their embeddings, and projects keys and
values. A signed-root sigmoid gate compares the normalized key with each
stream's normalized hidden state. Gated values and a causal dilated
convolution update the residual streams before the layer's main mixer.

Missing history and history before an EOS boundary use the configured EOS
value. The token history and the PLE convolution history therefore belong
to the sequence state: checkpoints preserve them alongside DeltaNet's
state. Reusing an attention cache without restoring this history would
change the graph.

The GPU loader currently requires the released PLE arrangement: a single
PLE layer on a recurrent layer, the supported n-gram/head layout, Q8_0
rows, and matching metadata for row offsets, ranges and embedding widths.
It reads rows rounded up to whole GPU words. Another tensor must follow
the PLE table so that the final row's padded read stays inside the file;
the loader checks this layout requirement.

About 21 GB of weights are uploaded for this model, excluding the PLE
table. That is not the complete runtime memory requirement: caches,
recurrent state, checkpoints and scratch buffers are additional. File
access remains necessary during generation, natively through the model
mapping and in the browser through the selected `File`.

Because the host computes PLE row locations, it must know each sampled
token before recording its next step. This program cannot chain samples
entirely on the GPU as the `qwen35` plan can; decode keeps one batch in
flight. See [scheduling](generation.md#scheduling) and
[state reuse](generation.md#state-and-prefix-reuse).

## Constraints and validation scope

The loader requires matching key/value dimensions, compatible grouped
query and recurrent head counts, supported rotary dimensions and sections,
and valid PLE placement. The GPU path also requires compression ratios
within the kernel's supported range and the expected expert rotation
metadata. The architecture name alone does not establish compatibility
with another `qwen4exp` checkpoint.

The recorded short-sequence GPU gate checks every layer's selected
experts, residual streams and logits, exact top-1 agreement, step splits
and fused/unfused equivalence. The CPU reference reproduces the reference
runtime's greedy tokens, but its f32 operands do not reproduce every
intermediate activation of that runtime's rounded multiplies. These are
different claims; the detailed
[validation table](validation.md#qwen38-flash-next-gyro-coder) records both.
