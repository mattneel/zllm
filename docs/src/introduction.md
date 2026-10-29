# zllm

zllm runs LLM inference on WebGPU from one Zig codebase. It aims to be an
engine of vLLM's class, running as efficiently as WebGPU allows.

- **GGUF in, tokens out.** It loads stock GGUF files directly, with no
  per-model compile step and no converted weight format.
- **One core, two hosts.** The same Zig core targets `wasm32-freestanding`
  behind a thin TypeScript shim in the browser, and Dawn natively for tests.
- **Kernels shaped to your GPU.** WGSL is generated from templates, then
  autotuned, fused and cached for each GPU and model.
- **Hybrid models first.** The first targets are Qwen 3.5, 3.6 and 3.8
  (Gated DeltaNet plus full attention), from 0.8B to 27B.

## Status

Milestone **M0** is complete: a GGUF reader, the Qwen tokenizer and a CPU f32
reference forward pass for Qwen 3.5 0.8B. Each one is checked against a pinned
reference ([Validation](validation.md)). None of it runs on a GPU yet; the
next milestone (M1) brings up the Dawn backend and the core kernels, checked
against this reference.

| Milestone | Scope | State |
|---|---|---|
| M0 | GGUF reader, tokenizer, CPU reference forward pass (0.8B) | done |
| M1 | Dawn backend, core kernels at parity | |
| M2 | Scheduler, state, sampler; native CLI chat | |
| M3 | wasm build, JS shim, browser streaming | |
| M4 | Templates, tuning, plan cache | |
| M5 | Fuser | |
| M6 | 27B on an RTX 5090 Laptop meets the performance bar | |
| M7 | Llama and Qwen3 dense, MTP speculative decoding | |

## Reading this book

- [Spec](spec.md): the v0.1 design, verbatim.
- [Architecture](architecture.md): modules, the async command model and the
  validation chain.
- [Qwen 3.5 GGUF semantics](qwen35.md): what each tensor means, including the
  converter transforms.
- [Validation](validation.md): pins, checks, thresholds and current results.
- [Contributing](contributing.md): toolchain, layout and commit conventions.
