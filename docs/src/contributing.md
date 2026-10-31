# Contributing

## Toolchain

| Tool | Pinned by | Used for |
|---|---|---|
| Zig `0.18.0-dev.738+ae9a8acc7` | `minimum_zig_version` in `build.zig.zon` ([anyzig](https://github.com/marler8997/anyzig) or [mlugg/setup-zig](https://github.com/mlugg/setup-zig) picks it up) | everything in `src/` |
| [uv](https://docs.astral.sh/uv/) | `tools/uv.lock` | parity harness in `tools/` |
| [Bun](https://bun.sh) | `web/bun.lock` | browser-side TypeScript in `web/` |
| [mdBook](https://rust-lang.github.io/mdBook/) 0.5 | CI workflow | this book |
| [Dawn](https://dawn.googlesource.com/dawn) | `scripts/fixtures.sh` (prebuilt release) | GPU host; on Linux also `libvulkan1` and `mesa-vulkan-drivers` for llvmpipe |

## Layout

| Path | Contents |
|---|---|
| `src/` | Shared Zig core, CPU reference, platform layer and backend implementations |
| `tools/` | Parity harness, reference dumper (`llama-dump`), corpora, table generators |
| `web/` | Browser WebGPU host, `zllm.js` facade, demo and book scripts |
| `docs/` | This book; GitHub Pages serves it from `master` |
| `scripts/fixtures.sh` | Pinned fixture downloads |
| `scripts/d3d12.sh` | Builds the Windows binary and runs it on the D3D12 host from WSL |
| `third_party/dawn` | The vendored `webgpu.h` matching the pinned Dawn release |
| `src/velocity/` | Native D3D12 host, HLSL kernels and Windows SDK interface |
| `benchmarks/` | Committed per-device benchmark records |

## Checks

CI runs formatting, unit tests, wasm compilation, tokenizer/chat parity,
the fast GPU tier on Dawn/llvmpipe, TypeScript checks/tests and the book
build. The full model gates need their fixtures and run separately. For a
change that touches all these areas, the corresponding commands are:

```sh
zig fmt --check . && zig build test
(cd web && bun run check)
(cd web && bun run test)
scripts/fixtures.sh vocab && zig build check-tokenizer && zig build check-chat
scripts/fixtures.sh dawn && scripts/fixtures.sh model
zig build check-fast -Dengine-check=false   # CI's llvmpipe tier
(cd web && bun run build:book) && mdbook build docs
```

For changes to `gguf`, `graph` or `ref`, also run the fixture-backed checks
`zig build check-gguf`, `zig build check-dequant` and
`zig build check-forward`. For changes to execution or kernels, run
`zig build check-gpu`, plus the affected quantized and Gyro-Coder model
gates. See [Validation](validation.md) for fixtures, options and thresholds.

`zig build check-fast` is an iteration tier: kernel checks followed by the
first eval text through both kernel paths, step splits and every KV format,
plus a short Gyro-Coder check when `-Dgyro-model` is set. It does not replace
the full gates.

CI caches the model and vocabulary fixtures and `.fixtures/cache`, the CPU
reference outputs. The reference cache is keyed by the reference, GGUF and
graph source, CLI source, build files and eval corpus; changing those
inputs recomputes the reference.

CI passes `-Dengine-check=false` on llvmpipe. The software renderer makes
engine-check expensive and its prefix-reuse rounding can exceed the
bound. This is an explicit coverage gap, not a relaxed bound.
Engine-check still runs on real GPUs with `d3d12` and `velocity`.

Run GPU work one job at a time. Use `-Dgpu-host=d3d12` for Dawn on a Windows
GPU from WSL, or `-Dgpu-host=velocity` for Velocity. The default native host
uses Dawn; llvmpipe model/reference checks are not real-GPU performance
evidence. Backend-specific gates matter even when hosts share the core.

For kernel performance, `zllm profile <model.gguf> [tokens]` (or
`scripts/d3d12.sh profile ...`) gives each kernel's GPU time from timestamp
queries, for prefill and decode steps. `zllm bench` times whole decode
steps, and each kernel class's share by leaving it out.

Keep performance evidence separate from targets. Record the backend,
adapter, model revision, context, precision and commit alongside results.
See [Performance](performance.md). New kernels need a CPU reference and
checks on each executing backend; fusion and precision changes need their
own model gates.

For documentation-only changes, build the book and check local page and
fragment links. Preserve existing linked headings when reorganizing
chapters. The demo is bundled separately by CI after the book build.

## TypeScript

- `tsconfig.json` is strict, plus `noUncheckedIndexedAccess`,
  `exactOptionalPropertyTypes` and the rest of the `no*` family.
- Biome lints and formats. `any`, `@ts-ignore` and non-null `!` are errors.
  If a non-null assertion really is unavoidable, it needs a
  `biome-ignore` comment explaining why.
- Use static imports for modules known when you write the code.
- Narrow `unknown` with type guards, never casts.

## Git

- Use [Conventional Commits](https://www.conventionalcommits.org/):
  `type(scope): summary`, where type is `feat`, `fix`, `test`, `docs`,
  `chore`, `build`, `ci`, `refactor` or `perf`, and scope is the module or
  area (`gguf`, `tokenizer`, `ref`, `web`, `book`, …).
- Make one logical change per commit. Every commit must pass `zig fmt
  --check` and `zig build test` on its own.
- Keep history linear: rebase, never merge. Squash WIP and fixup commits
  before pushing, and never rewrite history that's already pushed.
- Never commit binaries or model files. Fixtures are fetched and pinned by
  digest; generated files are gitignored unless they are source, such as
  `src/tokenizer/unicode_data.zig`.
