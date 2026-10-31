# zllm

[![CI](https://github.com/mattneel/zllm/actions/workflows/ci.yml/badge.svg)](https://github.com/mattneel/zllm/actions/workflows/ci.yml)
[![Book](https://img.shields.io/badge/book-mattneel.github.io%2Fzllm-blue)](https://mattneel.github.io/zllm/)

LLM inference on WebGPU from one Zig codebase, built to the standard of a server engine like vLLM.

- **GGUF in, tokens out.** It loads stock GGUF files directly, with no per-model compile step and no converted weight format.
- **One core, two backends.** The same Zig core targets `wasm32-freestanding` behind a thin TypeScript shim in the browser, and Dawn natively for tests.
- **Kernels shaped to your GPU.** WGSL is generated from templates, then autotuned, fused and cached for each GPU and model.
- **Hybrid models first.** The first targets are Qwen 3.5, 3.6 and 3.8 (Gated DeltaNet plus full attention), from 0.8B to 27B.
- **OpenAI-style streaming API.** `chat.completions.create` in the browser, with the model cached on disk after the first load.

> **Status: M3 done, so Qwen 3.5 0.8B runs in the browser on WebGPU, at native speed.** M0 delivered the GGUF reader, the Qwen tokenizer and a CPU f32 reference forward pass, checked against llama.cpp and Hugging Face. M1 put the forward pass on the GPU through Dawn, matching the reference on all 818 eval positions. M2 added generation:
>
> - an engine that schedules steps under a 150 ms budget;
> - KV-cache and DeltaNet-state reuse, bitwise exact against fresh runs;
> - a GPU sampler with vLLM's semantics;
> - the fixed Qwen chat template.
>
> M3 builds the core to `zllm.wasm` behind the `zllm.js` shim: Chrome generates the same tokens as the native host at 252–295 tok/s on an RTX 5090 Laptop, through `generate` and OpenAI-style `chat.completions.create`, and reloads the cached model in about a second.
>
> M6 is done: Qwen 3.8 27B at Q3_K_M runs on the GPU, matching the CPU reference and, up to its activation rounding, llama.cpp. On an RTX 5090 Laptop it decodes at about 30 tok/s (bar: 20), prefills at about 320 tok/s (bar: 300), loads in Chrome from a local file in 15.3 s (bar: 40 s), and soaks for 10 minutes with no submission over 250 ms. M7 has started: speculative decoding with the model's MTP block takes the 27B's native decode to 37–41 tok/s, bitwise the same output.
>
> Qwen3.8-Flash-Next Gyro-Coder, a 125B mixture-of-experts model in AgentionAI's Gyro trellis formats (75.8 GB), runs natively and in Chrome with up to 32,768 tokens of context (its sparse attention included): about 21 GB of weights on the GPU, its 50.7 GiB per-layer embedding table read from the file per token. Greedy, it reproduces its reference runtime's tokens, natively and in Chrome, where it loads from the picked file in about 20 s; on an RTX 5090 Laptop it decodes at 35–38 tok/s natively and 21–22 in Chrome.
>
> Dense transformers run too: Qwen 3, Llama 3 and Granite 4 (`qwen3`, `llama`, `granite`), with their tokenizers. Their CPU reference matches llama.cpp, and the GPU matches the reference, on Qwen3-0.6B, Llama-3.2-1B and Granite 4.2 3B. They chat in their own GGUF templates, rendered by zllm's Jinja interpreter exactly as transformers renders them. Granite 4.2 30B at Q4_K_M matches the CPU reference and, up to its activation rounding, llama.cpp; on an RTX 5090 Laptop it prefills at 209 tok/s and decodes at 25 tok/s. See [Dense transformers](https://mattneel.github.io/zllm/dense.html).
>
> M4's decode kernels and M5's fusion rules have started: generation runs at about 285 tok/s and prefill at about 2,890 tok/s, up from 25–34 and 1,560 at M2. See [Browser](https://mattneel.github.io/zllm/browser.html), [Generation](https://mattneel.github.io/zllm/generation.html), [Validation](https://mattneel.github.io/zllm/validation.html) and the [roadmap](https://mattneel.github.io/zllm/introduction.html#status).

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

In the browser (Chrome or another browser with WebGPU, after `scripts/fixtures.sh model`), with [Bun](https://bun.sh):

```sh
zig build wasm                   # zig-out/web/zllm.wasm
cd web && bun install && bun run dev   # chat demo on http://localhost:8080/
```

Add `-Dgpu-host=d3d12` to the GPU checks to run them on a Windows GPU from WSL (see [GPU backend](https://mattneel.github.io/zllm/gpu.html)).

zllm runs on more than one GPU backend: WebGPU (Dawn natively, and the browser), and Velocity, a native Direct3D 12 backend with its own HLSL kernels, LinAlg wave matrices and DirectStorage (see [Velocity](https://mattneel.github.io/zllm/velocity.html); its kernel port is in progress). `ZLLM_BACKEND=velocity` selects it.

## Documentation

The book is at **[mattneel.github.io/zllm](https://mattneel.github.io/zllm/)** and contains the spec, the architecture, the GGUF tensor semantics for each model, and the validation contract. To build it locally:

```sh
(cd web && bun install && bun run build:book)   # bundles the book's TypeScript
mdbook serve docs
```

See [Contributing](https://mattneel.github.io/zllm/contributing.html) for the toolchain, the checks and the commit conventions.

## Funding

If zllm is useful to you, consider [sponsoring @mattneel](https://github.com/sponsors/mattneel).

## License

MIT. See [LICENSE](LICENSE).
