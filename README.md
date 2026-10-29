# zllm

[![CI](https://github.com/mattneel/zllm/actions/workflows/ci.yml/badge.svg)](https://github.com/mattneel/zllm/actions/workflows/ci.yml)
[![Book](https://img.shields.io/badge/book-mattneel.github.io%2Fzllm-blue)](https://mattneel.github.io/zllm/)

LLM inference on WebGPU from one Zig codebase, built to the standard of a server engine like vLLM.

- **GGUF in, tokens out.** It loads stock GGUF files directly, with no per-model compile step and no converted weight format.
- **One core, two backends.** The same Zig core targets `wasm32-freestanding` behind a thin TypeScript shim in the browser, and Dawn natively for tests.
- **Kernels shaped to your GPU.** WGSL is generated from templates, then autotuned, fused and cached for each GPU and model.
- **Hybrid models first.** The first targets are Qwen 3.5, 3.6 and 3.8 (Gated DeltaNet plus full attention), from 0.8B to 27B.
- **OpenAI-style streaming API.** It's meant as a drop-in `engineFactory` for WebLLM.

> **Status: M0 done; nothing runs on a GPU yet.** The GGUF reader, the Qwen tokenizer and a CPU f32 reference forward pass for Qwen 3.5 0.8B are in place, each checked against a pinned reference: llama.cpp gguf-dump, Hugging Face `tokenizers`, and llama.cpp's forward pass (99.76% top-1, final-state cosine ≥ 0.9998). See [Validation](https://mattneel.github.io/zllm/validation.html) and the [roadmap](https://mattneel.github.io/zllm/introduction.html#status).

## Building

The Zig version is pinned in `build.zig.zon` as `minimum_zig_version`. [anyzig](https://github.com/marler8997/anyzig) selects it automatically.

```sh
zig build test        # unit tests
zig fmt --check .     # formatting
```

The parity checks need the pinned fixtures (GGUF, Hugging Face tokenizer, llama.cpp), each verified by SHA-256 or commit hash, plus [uv](https://docs.astral.sh/uv/):

```sh
scripts/fixtures.sh          # downloads into .fixtures/ (gitignored)
zig build check-gguf         # GGUF reader vs llama.cpp gguf-dump
zig build check-tokenizer    # tokenizer vs Hugging Face
zig build check-forward      # CPU reference vs llama.cpp
```

## Documentation

The book is at **[mattneel.github.io/zllm](https://mattneel.github.io/zllm/)** and contains the spec, the architecture, the GGUF tensor semantics for each model, and the validation contract. To build it locally:

```sh
(cd web && bun install && bun run build:book)   # bundles the book's TypeScript
mdbook serve docs
```

See [Contributing](https://mattneel.github.io/zllm/contributing.html) for the toolchain, the checks and the commit conventions.

## Funding

If zllm is useful to you, consider [sponsoring @mattneel](https://github.com/sponsors/mattneel).
