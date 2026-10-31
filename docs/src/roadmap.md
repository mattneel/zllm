# Roadmap

The engine's goal is portable inference that uses the hardware well while
preserving model quality. The [status table](introduction.md#status) and
[backend matrix](backends.md) describe current implementation. This page
lists remaining work and future scope, not release commitments or measured
capabilities.

## Complete the current execution path

| Area | Implemented | Remaining distinction from the design |
|---|---|---|
| Fusion | Hand-written fused kernels and a graph fusion pass for supported producer/consumer fragments | The pass is not a general implementation of every rule in the spec |
| Tuning | Load-time decode-multiply variant sweeps; persistent cache | Not a sweep of every kernel and every fused program |
| Plan cache | Device/model/build-keyed tuning choices | Not serialized GPU pipelines or the full future precision policy |
| Precision | Explicit KV formats, including FP8, with reference/model checks | Automatic fastest-passing-rung selection and persistence of quality decisions are not implemented |
| Serving | Single-sequence scheduler, browser async generation and OpenAI-style chat chunks | No HTTP server, multi-sequence batching or general native C API |
| Platforms | Browser WebGPU, native Dawn and native Velocity | Validation and performance coverage must be established per adapter and backend |

See [Architecture](architecture.md), [Precision](precision.md), and
[Validation](validation.md) for the exact implementation boundaries.

## Backend and model targets

CUDA is a requested backend target. The intended direction is PTX kernel
templates, CUDA Graphs, and NVFP4/FP8 tensor-core paths. There is no CUDA
backend or validated CUDA implementation in this repository today. These
mechanisms are roadmap intent, not a description of existing code.

Gemma 4 and Nanbeige 4.3 are requested next model targets. They have no
implemented graph or model gates in the repository today. They must meet
the same tokenizer, CPU reference, kernel and model validation requirements
as the existing families. There are no measured results for them here.

## Image generation after the LLMs

The [spec's image scope](spec.md#1-scope) names Z-Image as the first
text-to-image target. It would reuse the core, backends and kernels and add:

- The engine's Qwen3 text encoder to produce conditioning hidden states.
- A single-stream diffusion transformer denoiser.
- A VAE decoder.
- A flow-matching Euler scheduler with classifier-free guidance.
- An OpenAI-style image interface.

These are planned components, not a working image pipeline. Each new
kernel must have a CPU reference and backend checks before a performance
claim is meaningful.

## Video with audio after images

The spec then names MiniMax-H3: text, frames or references to video with
native stereo audio. That scope adds video and audio VAE decoding and
staged loading. The text encoder is freed before the denoiser loads, as
the spec describes. MiniMax-H3 weights use its Community License; the
engine's MIT license does not replace a model's weight license.

There is no implemented video-generation pipeline or benchmark here.

## Scope boundaries

The original single-user scope excludes training, concurrent sequences,
multi-GPU inference, vision/audio input to the LLM path, grammar-constrained
decoding and a CPU inference backend. The CPU implementation remains a
validation reference. Later image/video plans do not make those features
available to today's text-generation API.

The [spec](spec.md) retains its original milestone history and later scope
amendments. It is useful design context, not a substitute for the current
implementation and validation chapters.
