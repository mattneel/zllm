# zllm

[![CI](https://github.com/mattneel/zllm/actions/workflows/ci.yml/badge.svg)](https://github.com/mattneel/zllm/actions/workflows/ci.yml)
[![Book](https://img.shields.io/badge/book-mattneel.github.io%2Fzllm-blue)](https://mattneel.github.io/zllm/)

LLM inference on WebGPU from one Zig codebase, built to the standard of a server engine like vLLM.

- **GGUF in, tokens out.** It loads stock GGUF files directly, with no per-model compile step and no converted weight format.
- **One core, two backends.** The same Zig core targets `wasm32-freestanding` behind a thin TypeScript shim in the browser, and Dawn natively for tests.
- **Kernels shaped to your GPU.** WGSL is generated from templates, then autotuned, fused and cached for each GPU and model.
- **Hybrid models first.** The first targets are Qwen 3.5, 3.6 and 3.8 (Gated DeltaNet plus full attention), from 0.8B to 27B.
- **OpenAI-style streaming API.** It's meant as a drop-in `engineFactory` for WebLLM.

> **Status: pre-M0.** None of this runs yet. Milestone M0 covers the GGUF reader, the tokenizer and a CPU f32 reference forward pass for Qwen 3.5 0.8B, checked for parity against a pinned llama.cpp build. See the [roadmap](https://mattneel.github.io/zllm/spec.html#13-milestones).

## Building

The Zig version is pinned in `build.zig.zon` as `minimum_zig_version`. [anyzig](https://github.com/marler8997/anyzig) selects it automatically.

```sh
zig build test        # unit tests
zig fmt --check .     # formatting
```

Parity tests need the pinned fixtures: the GGUF, the Hugging Face tokenizer and llama.cpp. All of them are checked by SHA-256 or commit hash.

```sh
scripts/fixtures.sh   # downloads into .fixtures/ (gitignored)
```

## Documentation

The book is at **[mattneel.github.io/zllm](https://mattneel.github.io/zllm/)** and contains the spec, the architecture, the GGUF tensor semantics for each model, and the validation contract. To build it locally:

```sh
mdbook serve docs
```

## Funding

If zllm is useful to you, consider [sponsoring @mattneel](https://github.com/sponsors/mattneel).
