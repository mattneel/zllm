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
| BF16/F16 prefill multiplies | Scalar f32, or FP16 single-product / activation-high-and-residual LinAlg products with f32 accumulation | Native quality-gated operand ladder on capable Agility; scalar on Dawn |
| Quantized prefill multiplies | Round operands to Q8_K blocks for K-quants or Q8_0 blocks for Q8_0 weights; accumulate in f32 | Part of the existing quantized prefill path, not an adaptive precision choice |
| Gyro-Coder float-activation Q8_0 and grouped TQ2_T prefill | Scalar f32; FP16-rounded activations and scaled weights (single product); or compensated high/residual activation products with exact unscaled weight operands | Native quality-gated operand ladder on capable Agility; scalar HLSL fallback |
| Attention KV | f32, f16, FP8 E4M3FN | Native quality ladder FP8 → f16 → f32; browser remains explicit, default f16; `gpu-forward` defaults f32 |
| DeltaNet recurrent state and convolution history | f32 | Not changed by `--kv` |
| Gyro sparse indexer keys / pooled blocks | f16 / f32 | Separate from the attention KV choice |

CPU forward passes are correctness references, not a CPU serving backend.
The ordinary hybrid/dense reference uses f32 activations and matrix
accumulation; RMSNorm accumulates f32 squares in f64, matching the pinned
CPU runtime. Its `--llama-activations` mode deliberately reproduces that
runtime's active x86 multiply-operand rounding for quantized comparisons,
including ties-to-even Q8_0 rather than the scalar file quantizer's rule.
Gyro-Coder's CPU reference defaults to f16 attention and indexer caches.
`forward --kv f32` selects f32 attention KV for the ladder oracle while
retaining the model's fixed f16 indexer storage.

## Weight formats

The generic GPU embedding, decode and prefill kernels support:

- BF16 and F16;
- Q8_0;
- Q3_K, Q4_K, Q5_K and Q6_K.

IQ4_XS is additionally supported for untied embedding tables, including
Nanbeige 4.3's. Its CPU decoder follows ggml's nonlinear 16-entry lookup
table and signed six-bit sub-block scales. GPU upload pads each 136-byte
block to 144 bytes; both backends decode the same values. IQ4_XS GPU
matrix-vector and matrix-matrix projections are not implemented: a tied
IQ4_XS output table is rejected, rather than dispatched to an absent kernel.

A model file can mix these types. Small norm and recurrent parameter
tensors can remain F32 even when its large matrices are quantized. F32
parameter-vector support does not imply generic F32 GPU matrix kernels.
The ordinary CPU reference can load F32, F16 and BF16 matrices as well as
the supported quantized matrices.

GGUF parsing is broader than executable model support: the reader knows
the block layout of other GGML types so it can index their offsets and
sizes. The model loader and kernel dispatch still reject unsupported
matrix types. In particular, recognizing Q2_K, other IQ formats or MXFP4
in the GGUF index is not a claim that the generic GPU plan can infer with them.

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

Agility's LinAlg variants decode into bounded shared-memory tiles, not
persistent dense FP16 matrices. The compensated Q8_0/TQ2_T rung splits
activations into rounded FP16 high and residual parts. Signed-byte integers
and FP16 codebook entries are exact weight operands; each original block's
scale is applied after its two f32-accumulating dot products. The faster
single-product rung instead rounds activations and scaled/dequantized
weights to FP16 and omits correction products. BF16/F16 variants use the
same activation choices with weights converted to FP16. Every approximate
rung is gated at model load; its existence is not a quality approval.

## KV cache formats

The shared `KvFormat` has three choices:

| Choice | Storage | Intended use |
|---|---|---|
| `f32` | f32 cached values | High-precision diagnostic baseline; larger cache |
| `f16` | Packed binary16 values, explicitly rounded to nearest with ties to even | Middle native ladder rung; explicit browser default |
| `fp8` | E4M3FN bytes, four values per word, with an f32 scale for each position and KV head | First native ladder rung, accepted only by the model gate |

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

Native generation, chat and benchmarks default to a deterministic load-time
ladder. Each class is ordered fastest-first:

- attention KV: **FP8 → f16 → f32**;
- prefill GEMM operands: **FP16 single product → FP16 compensated two
  products → scalar f32**;
- weight storage: **stored** only. There are no unimplemented weight rungs
  or automatic persistent requantization.

The loader runs a fixed coding/language text (the first 64 model-tokenized
positions) through the exact device path, with f32 KV and scalar operands.
It tries KV candidates first with scalar operands, then supported operand
candidates with the chosen KV. The latter comparison uses the same exact
logits, so the combined choice must pass, not merely its isolated classes.
No tokens are sampled. Unsupported operand rungs are skipped; Dawn currently
has only scalar-f32 prefill. Browser loading still uses its explicit KV
option and does not run the native calibration.

The model policy in `src/precision.zig` requires top-1 agreement ≥99%,
mean **KL(P_exact || P_candidate) ≤0.001 nats**, and maximum per-position
relative logits L2 ≤0.1. Reductions and stable log-softmax use deterministic
f64 arithmetic. Nonfinite logits reject the rung. On a 64-position probe,
99% requires all 64 top-1 predictions to agree. The KL bound limits mean
total-variation distance to at most approximately 2.24% by Pinsker's
inequality and Jensen's inequality; the L2 gate also catches changes that
distribution normalization can hide. Measured per-rung results and the
policy's separation between accepted and rejected candidates are recorded
in [Performance](performance.md).

This is a short calibration gate, not a guarantee about every future prompt
or long-context workload. The existing strict kernel/model correctness
gates are separate and are not relaxed.

The native JSON plan cache records selected rungs and gate observations
alongside tuning. A valid hit restores the same choice and skips the gate.
The key includes requested overrides, model/header identity, native operand
capability, all backend shader variants and the versioned gate policy.
`--no-tune` skips only the decode sweep, not quality selection; cache metadata
distinguishes a quality-only entry from completed empty tuning.

`--kv f16|f32|fp8` opts the KV class out and forces that format. Its operand
gate compares against scalar arithmetic with the same forced cache format.
`--precision exact` overrides even a forced KV option and selects f32 KV
plus scalar-f32 operands. Stored weight types, model-fixed indexer storage,
and the existing non-Gyro quantized-prefill arithmetic are not rewritten.
`ZLLM_KV` and `ZLLM_PRECISION=auto|exact` provide the corresponding native
defaults; command-line flags take precedence. The load log and benchmark
JSON report chosen rungs.

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
quality-approved FP8 default. Native selection now measures each model's
calibration instead of treating those aggregate diagnostic results as
permission. See [GPU forward validation](validation.md#gpu-forward-pass).

Quantized-prefill results also need care: rounding operands to 8-bit
blocks changes outputs even when every kernel is correct. The 27B's
ordinary reference versus llama.cpp does not meet the generic model
thresholds; a rounded reference isolates that arithmetic difference.
Do not describe the existing comparison as proof of lossless quantization,
or silently relax a future default-selection gate to the diagnostic one.
