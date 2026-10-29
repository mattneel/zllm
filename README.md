# zllm

[![CI](https://github.com/mattneel/zllm/actions/workflows/ci.yml/badge.svg)](https://github.com/mattneel/zllm/actions/workflows/ci.yml)
[![Book](https://img.shields.io/badge/book-mattneel.github.io%2Fzllm-blue)](https://mattneel.github.io/zllm/)

LLM inference on WebGPU from one Zig codebase, built to the standard of a server engine like vLLM.

- **GGUF in, tokens out.** It loads stock GGUF files directly, with no per-model compile step and no converted weight format.
- **One core, two backends.** The same Zig core targets `wasm32-freestanding` behind a thin TypeScript shim in the browser, and Dawn natively for tests.
- **Kernels shaped to your GPU.** WGSL is generated from templates, then autotuned, fused and cached for each GPU and model.
- **Hybrid models first.** The first targets are Qwen 3.5, 3.6 and 3.8 (Gated DeltaNet plus full attention), from 0.8B to 27B.
- **OpenAI-style streaming API.** It's meant as a drop-in `engineFactory` for WebLLM.

> **Status: M2 done, so `zllm chat` runs Qwen 3.5 0.8B on the GPU natively.** M0 delivered the GGUF reader, the Qwen tokenizer and a CPU f32 reference forward pass, checked against llama.cpp and Hugging Face. M1 put the forward pass on the GPU through Dawn, matching the reference on all 818 eval positions. M2 added generation:
>
> - an engine that schedules steps under a 150 ms budget;
> - KV-cache and DeltaNet-state reuse, bitwise exact against fresh runs;
> - a GPU sampler with vLLM's semantics;
> - the fixed Qwen chat template.
>
> M4's decode kernels and M5's fusion rules have started: generation now runs at about 285 tok/s and prefill at about 2,890 tok/s on an RTX 5090 Laptop, up from 25–34 and 1,560 at M2. Next is M3: the browser. See [Generation](https://mattneel.github.io/zllm/generation.html), [Validation](https://mattneel.github.io/zllm/validation.html) and the [roadmap](https://mattneel.github.io/zllm/introduction.html#status).

## Building

The Zig version is pinned in `build.zig.zon` as `minimum_zig_version`. [anyzig](https://github.com/marler8997/anyzig) selects it automatically.

```sh
zig build test        # unit tests
zig fmt --check .     # formatting
```

The parity checks need the pinned fixtures (GGUF, Hugging Face tokenizer, llama.cpp, Dawn), each verified by SHA-256 or commit hash, plus [uv](https://docs.astral.sh/uv/):

```sh
scripts/fixtures.sh          # downloads into .fixtures/ (gitignored)
zig build check-gguf         # GGUF reader vs llama.cpp gguf-dump
zig build check-tokenizer    # tokenizer vs Hugging Face
zig build check-chat         # chat prompts vs the pinned Jinja chat template
zig build check-forward      # CPU reference vs llama.cpp
zig build gpu-check          # every GPU kernel vs a CPU computation
zig build check-gpu          # GPU forward pass vs the CPU reference, step splits, engine
```

To chat (after `scripts/fixtures.sh model` and `scripts/fixtures.sh dawn`):

```sh
ZLLM_DAWN=.fixtures/dawn/libwebgpu_dawn.so zig-out/bin/zllm chat .fixtures/models/Qwen3.5-0.8B-BF16.gguf
scripts/d3d12.sh chat "$(wslpath -w .fixtures/models/Qwen3.5-0.8B-BF16.gguf)"   # WSL, Windows GPU
```

Add `-Dgpu-host=d3d12` to the GPU checks to run them on a Windows GPU from WSL (see [GPU backend](https://mattneel.github.io/zllm/gpu.html)).

## Documentation

The book is at **[mattneel.github.io/zllm](https://mattneel.github.io/zllm/)** and contains the spec, the architecture, the GGUF tensor semantics for each model, and the validation contract. To build it locally:

```sh
(cd web && bun install && bun run build:book)   # bundles the book's TypeScript
mdbook serve docs
```

See [Contributing](https://mattneel.github.io/zllm/contributing.html) for the toolchain, the checks and the commit conventions.

## Funding

If zllm is useful to you, consider [sponsoring @mattneel](https://github.com/sponsors/mattneel).
