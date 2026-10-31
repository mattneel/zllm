# Getting started

Run commands from the repository root unless a block says otherwise.
The [live demo](https://mattneel.github.io/zllm/demo/) needs no local build.
Use a WebGPU-capable browser, such as Chrome or Edge.

## Toolchain

The Zig version is pinned as `minimum_zig_version` in `build.zig.zon`.
[anyzig](https://github.com/marler8997/anyzig) selects it automatically.
Use [Bun](https://bun.sh) for the browser host and
[uv](https://docs.astral.sh/uv/) for the reference-check harness.
See [Contributing](contributing.md#toolchain) for the full toolchain.

Build the native CLI with safety checks and optimization:

```sh
zig build -Doptimize=ReleaseSafe
```

This installs `zig-out/bin/zllm`. The default backend is WebGPU through
Dawn. The CPU reference commands are for validation, not CPU chat.

## Fixtures

`scripts/fixtures.sh` downloads pinned files into `.fixtures/` and checks
their digests or commit revisions. Fetch only what the chosen workflow needs:

```sh
scripts/fixtures.sh model   # Qwen3.5-0.8B-BF16.gguf
scripts/fixtures.sh dawn    # native WebGPU runtime
```

On Linux, Dawn also needs a Vulkan loader and a usable Vulkan driver.
The software driver llvmpipe is useful for kernel validation; it is not a
performance target. Under WSL, the fixture script stages the Windows Dawn
and DXC DLLs when the Windows toolchain is available.

Other fixture selectors include `vocab` (tokenizers and GGUF headers),
`dense` (dense-model files), `model27b`, `llama` (the reference runtime),
and `agility` (the Windows backend SDKs). Model files can be large; the
fixture script is the authority for their names, revisions and digests.

## Native CLI on Dawn

On Linux:

```sh
ZLLM_DAWN=.fixtures/dawn/libwebgpu_dawn.so \
  zig-out/bin/zllm chat .fixtures/models/Qwen3.5-0.8B-BF16.gguf
```

For direct text generation:

```sh
ZLLM_DAWN=.fixtures/dawn/libwebgpu_dawn.so \
  zig-out/bin/zllm generate .fixtures/models/Qwen3.5-0.8B-BF16.gguf \
  'Explain grouped-query attention.' --max-tokens 128
```

`ZLLM_DAWN` names the runtime library. Without it, the host uses the
platform's default library name. On Windows that is `webgpu_dawn.dll`.
See [Generation and serving](generation.md) for sampling, context, KV
formats, speculation and chat controls.

To run on the Windows GPU from WSL, after staging Dawn:

```sh
scripts/d3d12.sh chat "$(wslpath -w .fixtures/models/Qwen3.5-0.8B-BF16.gguf)"
```

The wrapper builds the Windows executable and runs it from the staged
Windows directory. Model paths passed to it must be Windows paths.

## Choosing a backend

| Selection | Meaning | Requirement |
|---|---|---|
| `ZLLM_BACKEND=webgpu` | Native Dawn; the default | Dawn runtime library |
| `ZLLM_BACKEND=agility` | Native Direct3D 12, without Dawn execution | Windows build with `-Dagility` and staged SDK/runtime files |
| Browser | WebGPU through the wasm host | WebGPU-capable browser; no native backend environment variable |

For Agility from WSL:

```sh
scripts/fixtures.sh dawn
scripts/fixtures.sh agility
ZLLM_BACKEND=agility scripts/d3d12.sh chat \
  "$(wslpath -w .fixtures/models/Qwen3.5-0.8B-BF16.gguf)"
```

The wrapper includes Agility when the fetched Agility SDK is present.
`ZLLM_ADAPTER` selects a Agility adapter index. For build prerequisites
and capability limits, read [Agility](agility.md) and
[Backends](backends.md). CUDA is not a selectable backend.

## Browser development

Fetch the model fixture if loading locally. Then build wasm and start the
development server:

```sh
zig build wasm
(cd web && bun install --frozen-lockfile && bun run dev)
```

Open `http://localhost:8080/`. The demo can stream a model from its URL or
load a local GGUF. URL-loaded weights can be cached in the origin private
file system. Local files and remote URLs have different loading costs;
see [Browser](browser.md) rather than treating either as a universal load
time.

## Documentation and validation

To build the book:

```sh
(cd web && bun install --frozen-lockfile && bun run build:book)
mdbook build docs
```

`mdbook serve docs` serves the book locally. The published demo is bundled
separately by CI; a plain mdBook build does not build the demo.

For the correctness workflow, start with
[Contributing](contributing.md#checks) and the detailed
[Validation](validation.md) contract. Do not treat a successful build or a
readable chat response as evidence of kernel or model parity.
