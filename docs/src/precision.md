# Precision and quality gates

Precision is a property of each tensor class and execution path, not just
the GGUF file name. Stored weights, temporary multiply operands, KV caches
and recurrent state can all use different formats. zllm aims to produce
tokens as fast and accurately as the supported hardware allows; a smaller
format is useful only when its numerical behavior is understood and its
model-quality gate is satisfied.

This page separates implemented formats and current defaults from the
quality-gated selection policy described in [spec §9](spec.md#9-state-and-prefix-reuse).

## Current behavior

| Tensor or operation | Implemented behavior | Current selection |
|---|---|---|
| Model weights | BF16, F16 and the supported GGUF quantizations; Gyro expert formats have separate constraints | Read each tensor's type from the GGUF; no automatic weight-quality sweep or requantization |
| Dense and hybrid decode multiplies | Decode stored weights and multiply with f32 activations, accumulating in f32 | Weight-specific kernel, with load-time tuning of quantized decode variants |
| BF16/F16 prefill multiplies | Widen stored weights and use f32 operands and accumulation | Weight-specific kernel |
| Quantized prefill multiplies | Round operands to Q8_K blocks for K-quants or Q8_0 blocks for Q8_0 weights; accumulate in f32 | Part of the existing quantized prefill path, not an adaptive precision choice |
| Gyro-Coder float-activation Q8_0 and grouped TQ2_T prefill | Original f32 values on scalar paths; FP16 activation high/residual parts, exact FP16 weight operands, and two f32-accumulating matrix products on supported Agility devices | Native LinAlg operation support selects the matrix variant, otherwise scalar HLSL; unchanged reference bounds and per-path bitwise split-step gates |
| Attention KV | f32, f16, FP8 E4M3FN | f16 for generation and browser loading; `gpu-forward` defaults to f32 for comparisons |
| DeltaNet recurrent state and convolution history | f32 | Not changed by `--kv` |
| Gyro sparse indexer keys / pooled blocks | f16 / f32 | Separate from the attention KV choice |

CPU forward passes are correctness references, not a CPU serving backend.
The ordinary hybrid/dense reference uses f32 activations and accumulation.
Its `--llama-activations` mode deliberately reproduces the reference
runtime's multiply-operand rounding for quantized model comparisons.
Gyro-Coder's CPU reference uses f32 operands but f16 attention and indexer
caches; do not assume that every reference graph has an f32 KV cache.

## Weight formats

The generic GPU embedding, decode and prefill kernels support:

- BF16 and F16;
- Q8_0;
- Q3_K, Q4_K, Q5_K and Q6_K.

A model file can mix these types. Small norm and recurrent parameter
tensors can remain F32 even when its large matrices are quantized. F32
parameter-vector support does not imply generic F32 GPU matrix kernels.
The ordinary CPU reference can load F32, F16 and BF16 matrices as well as
the supported quantized matrices.

GGUF parsing is broader than executable model support: the reader knows
the block layout of other GGML types so it can index their offsets and
sizes. The model loader and kernel dispatch still reject unsupported
matrix types. In particular, recognizing Q2_K, IQ or MXFP4 in the GGUF
index is not a claim that the generic GPU plan can infer with them.

The `Q4_K_M` and `Q3_K_M` names describe file-level mixtures, not one
uniform tensor type. The pinned 27B file mixes Q3_K, Q4_K, Q5_K, Q6_K
and Q8_0. The CPU reference decodes quantized panels as needed instead of
materializing the entire model as f32.

GPU upload can change the byte layout without changing the represented
weights: Q6_K blocks are padded, Q3_K blocks are repacked, and Q8_0 rows
split values from scales. These are load-time layouts for the kernels,
not additional lossy quantizations. See
[quantized-weight validation](validation.md#quantized-weights) for
bit-exact decoder checks, multiply checks and activation-rounding tests.

### Gyro expert weights

[Gyro-Coder](gyro.md#routed-experts-and-gyro-storage) uses AgentionAI's
Gyro trellis formats and Hadamard-rotated expert weights. Its CPU decoder
supports `TQ2_T`, `TQK6`, `TQK7`, `TQP5`, `TQP6`, `TQP7`, `TQK5` and
`TQK4`. Its GPU routed-expert kernel currently requires `TQ2_T`, including
the codebook and rotation signs. The PLE table requires Q8_0 rows. A
`TQ2_0` file name does not override these per-tensor checks.

Agility's LinAlg prefill variants decode weights into bounded shared-memory
tiles, not persistent dense FP16 matrices. Activations are split into a
rounded FP16 high part and an FP16 residual. Weight operands are exact
signed-byte integers for Q8_0 or exact FP16 codebook entries for TQ2_T.
Two matrix products accumulate high×weight and residual×weight in f32;
each original quantization block's scale is applied after its dot product.
This avoids rounding scaled weights to FP16. It is not an automatic
precision ladder or permission to relax a quality gate. Scalar HLSL and
WGSL retain their existing arithmetic.

## KV cache formats

The shared `KvFormat` has three choices:

| Choice | Storage | Intended use |
|---|---|---|
| `f32` | f32 cached values | High-precision diagnostic baseline; larger cache |
| `f16` | Packed binary16 values, explicitly rounded to nearest with ties to even | Current generation default |
| `fp8` | E4M3FN bytes, four values per word, with an f32 scale for each position and KV head | Explicit lower-precision option; model quality must be assessed separately |

FP8 uses the smallest power-of-two row scale that puts the row's maximum
magnitude within E4M3FN's finite range, 448. K and V have their own row
scales. Encoding rounds to nearest, ties to even; attention decodes the
bytes and reapplies the scale. The scales take additional memory, so the
cache is not just the raw byte count of its values. This is software
encoding and decoding, not a requirement for native FP8 matrix hardware.

`rope_append` writes the chosen format. Dense, wide-group and sparse
attention readers consume it. The CPU oracle is `src/ref/fp8.zig`; WGSL
uses `src/kernels/fp8.wgsl`, and Agility uses
`src/agility/kernels/fp8.hlsli`.

### FP8 on each backend

| Backend | Implementation | Evidence and limitation |
|---|---|---|
| Browser WebGPU | Shared WGSL FP8 writer and attention readers | The browser uses the same kernel source and format; this is not a separate published browser FP8 model-quality measurement |
| Native Dawn WebGPU | Shared WGSL FP8 writer and attention readers | Recorded exact encoding and model comparisons on the tested native hosts; see [GPU forward validation](validation.md#gpu-forward-pass) |
| Agility, native D3D12 | HLSL counterparts using the same buffer/parameter contract | Commit `ecec4af` records the parity checks and FP8 model comparisons on Agility; see also [Agility](agility.md#kernels) |

Availability on a backend is not a claim that every adapter has been
measured, nor that FP8 is acceptable for every model. The encoding gate
requires bytes and power-of-two scales to match the CPU oracle exactly;
attention is checked against a higher-precision computation. Full-model
comparisons then measure the rounding error's downstream effects.

### Selecting a cache explicitly

Native generation and chat accept `--kv f16`, `--kv f32` or `--kv fp8`:

```sh
zllm generate model.gguf "Explain the function." --kv f16
zllm chat model.gguf --kv fp8
```

The browser's load options accept `kv: "f16"`, `"f32"` or `"fp8"`.
Changing this option does not change weight storage, quantized-prefill
operand rounding, DeltaNet state or Gyro indexer-key storage. An f32 KV
cache is therefore not a global exact-precision mode.

## Quality-gated precision ladder

The spec's intended policy orders candidate formats by speed and keeps
the fastest supported rung that passes the model's quality gate. For KV
the proposed order is **FP8 → f16 → f32**. Similar selection is intended
for prefill operands and weight storage. The choice should be recorded
with the plan so that a cached plan preserves the quality decision, not
merely a fast kernel configuration.

This automatic quality-gated ladder is **planned policy, not current
load-time behavior**. Today:

- the caller selects KV explicitly, otherwise generation uses f16;
- weights keep the formats declared by the file;
- quantized prefill has its existing operand-rounding path;
- the plan cache stores tuned kernel variants, not a quality-approved
  choice of KV, weight or activation precision;
- `--precision exact`, mentioned in the spec, is not an implemented CLI
  option. `--kv f16` is implemented, but opts into a cache format rather
  than disabling all lower-precision arithmetic.

A quality gate must distinguish these questions:

1. **Does the implementation encode and compute the chosen format
   correctly?** Use exact decoder/encoding comparisons and kernel
   tolerances: [GPU kernels](validation.md#gpu-kernels) and
   [quantized weights](validation.md#quantized-weights).
2. **Does the full graph match an independent runtime when arithmetic is
   comparable?** Use the
   [reference forward pass](validation.md#reference-forward-pass),
   [dense models](validation.md#dense-models),
   [the quantized 27B](validation.md#the-27b) or
   [Gyro-Coder](validation.md#qwen38-flash-next-gyro-coder), as appropriate.
3. **Is the precision loss acceptable for this model and workload?** This
   requires the model-quality decision; passing a diagnostic FP8 bound is
   not sufficient to authorize a new default.

The current FP8 diagnostic gate requires at least 95% top-1 agreement
aggregated across the checked texts and models, with cosine similarity at
least 0.975 at every checked position. For f16 the corresponding bounds are
99% and 0.99998. These are deliberately different gates, not per-model
quality approvals. The recorded D3D12 small-model FP8 result is
1,231/1,264 top-1 agreement,
versus the f32 reference, and commit `ecec4af` records 1,233/1,264 on
Agility. These results establish an implemented FP8 path, not a
quality-approved FP8 default. The small models remain on f16. See
[GPU forward validation](validation.md#gpu-forward-pass) for each model
and the scope of the comparisons.

Quantized-prefill results also need care: rounding operands to 8-bit
blocks changes outputs even when every kernel is correct. The 27B's
ordinary reference versus llama.cpp does not meet the generic model
thresholds; a rounded reference isolates that arithmetic difference.
Do not describe the existing comparison as proof of lossless quantization,
or silently relax a future default-selection gate to the diagnostic one.
