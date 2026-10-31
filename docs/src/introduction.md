# zllm

zllm is a portable inference engine. It aims to deliver tokens as fast and
as accurately as the hardware allows, on every supported platform. This is
a design goal, not a claim that every device has been benchmarked.

One Zig core loads GGUF models, builds graphs and execution plans, applies
fusion, tunes kernels, caches plans, schedules work and samples on the GPU.
It also owns tokenization and chat-template rendering. Backends sit behind
a platform layer: WebGPU through Dawn natively or the browser through wasm,
and Agility for native Direct3D 12. CUDA is a roadmap target, not an
available backend.

**[Try the browser demo](https://mattneel.github.io/zllm/demo/).** It uses
the same core as the native CLI. CI builds the book, demo and wasm from the
same checkout before publishing them together. See [Browser](browser.md)
for model loading, caching and the JavaScript API.

## Status

This table describes implemented paths, not a promise that every model,
format and device combination has passed validation. The
[validation record](validation.md) names the fixtures, gates and tested
hosts. The [backend matrix](backends.md) distinguishes available features
from gaps and planned work.

| Area | Current state | Details |
|---|---|---|
| Hybrid models | Qwen 3.5/3.6/3.8 graph and GPU execution paths; reference and GPU gates on pinned fixtures | [GGUF semantics](qwen35.md) |
| Dense models | Qwen 3, Llama 3 and Granite 4, including model tokenizers and GGUF chat templates | [Dense transformers](dense.md) |
| Gyro-Coder | Qwen3.8-Flash-Next MoE, hyper-connections, sparse attention and streamed per-layer embeddings | [Gyro-Coder](gyro.md) |
| WebGPU | Native Dawn and browser wasm hosts execute the shared command stream | [WebGPU](gpu.md) |
| Agility | Native D3D12 backend, HLSL kernels, LinAlg capability queries and DirectStorage integration | [Agility](agility.md) |
| Planning | Kernel identity and contracts, fusion pass, decode-multiply tuning and persistent plan cache | [Architecture](architecture.md) |
| Precision | Quantized GGUF weights, FP8/f16/f32 KV, and a cached native model-quality ladder for KV and supported prefill operands | [Precision](precision.md) |
| Generation | Single-sequence scheduler, prefix reuse, GPU sampling, MTP speculation and device-lost rebuilding | [Generation and serving](generation.md) |
| API | Native CLI and browser generation/OpenAI-style chat interface; not an HTTP serving daemon | [Browser API](browser.md#api) |
| Future scope | CUDA, additional model families, then image and video generation | [Roadmap](roadmap.md) |

## Accuracy before speed

The CPU f32 reference is a correctness oracle, not a CPU inference backend.
Kernel checks compare GPU results with CPU computations. Model gates compare
the GPU with the reference, and the reference with pinned llama.cpp or the
model's reference runtime. Fusion and lower-precision paths have their own
gates. Passing one fixture does not establish quality for every model.

See [Validation](validation.md) for the detailed thresholds and results,
and [Precision](precision.md) for the distinction between available formats
and the planned quality-gated ladder.

## Reading this book

Start with [Getting started](getting-started.md) to build and run locally.
Read [Architecture](architecture.md) and [Backends](backends.md) for the
execution model. [Models](models.md) explains architecture-specific support.
[Generation and serving](generation.md) and [Browser](browser.md) describe
the user-facing interfaces.

[Performance](performance.md) separates dated measurements from targets.
[Roadmap](roadmap.md) separates implemented work from future scope.
[Contributing](contributing.md) lists the development workflow.
The [spec](spec.md) is the original design document, including its later
precision and generation-scope amendments; it is not a current support
matrix.
