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

Milestones **M0** and **M1** are complete.

- **M0:** a GGUF reader, the Qwen tokenizer and a CPU f32 reference forward
  pass for Qwen 3.5 0.8B, each checked against a pinned reference.
- **M1:** the Dawn GPU backend, the command stream and eleven WGSL kernels.
  The full forward pass runs on the GPU and matches the CPU reference on all
  818 eval positions, both on an RTX 5090 through D3D12 and on llvmpipe.

See [Validation](validation.md) and [GPU backend](gpu.md). Next is M2: the
scheduler, persistent state (KV cache and DeltaNet state across steps), GPU
sampling and a native command-line chat.

| Milestone | Scope | State |
|---|---|---|
| M0 | GGUF reader, tokenizer, CPU reference forward pass (0.8B) | done |
| M1 | Dawn backend, core kernels at parity | done |
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
- [GPU backend](gpu.md): hosts, the command stream, kernels and weight layout.
- [Contributing](contributing.md): toolchain, layout and commit conventions.
