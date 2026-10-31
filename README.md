# zllm

[![CI](https://github.com/mattneel/zllm/actions/workflows/ci.yml/badge.svg)](https://github.com/mattneel/zllm/actions/workflows/ci.yml)

zllm is a portable inference engine. It aims to deliver tokens as fast and
as accurately as the hardware allows, on every supported platform.

**[Try the live browser demo](https://mattneel.github.io/zllm/demo/)** ·
**[Read the book](https://mattneel.github.io/zllm/)**

- **One Zig core:** GGUF loading, graphs, execution plans, fusion, tuning,
  plan caching, scheduling, GPU sampling, tokenizers and chat templates.
- **Platform backends:** WebGPU through Dawn natively and wasm in the
  browser; Velocity for native D3D12 with Agility SDK and DirectStorage
  integration. CUDA is planned, not implemented.
- **Models:** Qwen 3.5/3.6/3.8 hybrid; Qwen 3, Llama 3 and Granite 4 dense;
  Qwen3.8-Flash-Next Gyro-Coder with MoE, hyper-connections, sparse attention
  and per-layer embeddings.
- **Accuracy gates:** GPU kernels are checked against CPU references;
  model gates compare with the reference and pinned upstream runtimes.
  Available lower-precision formats are not a blanket quality guarantee.
- **Shared browser API:** streaming `generate` and OpenAI-style
  `chat.completions.create`. CI builds the published demo and book from the
  same checkout.

See the [status](https://mattneel.github.io/zllm/introduction.html#status),
[backend capability matrix](https://mattneel.github.io/zllm/backends.html),
[precision guide](https://mattneel.github.io/zllm/precision.html), and
[dated performance results](https://mattneel.github.io/zllm/performance.html).
Coverage is fixture- and device-specific; the documentation distinguishes
implemented paths, validated configurations and roadmap goals.

## Quick start

The Zig version is pinned by `minimum_zig_version` in `build.zig.zon`.
[anyzig](https://github.com/marler8997/anyzig) selects it automatically.
Build the CLI and fetch the pinned model and Dawn runtime:

```sh
zig build -Doptimize=ReleaseSafe
scripts/fixtures.sh model
scripts/fixtures.sh dawn
ZLLM_DAWN=.fixtures/dawn/libwebgpu_dawn.so \
  zig-out/bin/zllm chat .fixtures/models/Qwen3.5-0.8B-BF16.gguf
```

The command above uses Linux Dawn. For Windows/WSL and Velocity, see
[Getting started](https://mattneel.github.io/zllm/getting-started.html).
`ZLLM_BACKEND=webgpu` is the native default; `ZLLM_BACKEND=velocity`
requires a Windows build with Velocity enabled and its SDK/runtime files.

For the local browser demo, with [Bun](https://bun.sh):

```sh
zig build wasm
(cd web && bun install --frozen-lockfile && bun run dev)
```

Open `http://localhost:8080/`. Load a local GGUF or use the demo's model
URLs. See [Browser](https://mattneel.github.io/zllm/browser.html) for the
API, weight cache and device-lost recovery.

## Documentation and development

The [book](https://mattneel.github.io/zllm/) covers
[architecture](https://mattneel.github.io/zllm/architecture.html),
[models](https://mattneel.github.io/zllm/models.html),
[generation](https://mattneel.github.io/zllm/generation.html),
[validation](https://mattneel.github.io/zllm/validation.html), and the
[roadmap](https://mattneel.github.io/zllm/roadmap.html), including planned
CUDA, model families, Z-Image and MiniMax-H3 work.

To build the book locally:

```sh
(cd web && bun install --frozen-lockfile && bun run build:book)
mdbook build docs
```

Read [Contributing](https://mattneel.github.io/zllm/contributing.html) for
the toolchain, checks and commit conventions. The CPU reference is a
correctness oracle, not a CPU inference backend. GPU performance changes
must pass their accuracy gates.

## Funding

If zllm is useful to you, consider [sponsoring @mattneel](https://github.com/sponsors/mattneel).

## License

MIT. See [LICENSE](LICENSE). Model weights retain their own licenses.
