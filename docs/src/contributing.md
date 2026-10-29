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
| `src/` | The Zig core: one directory per module in the spec's component table, plus `ref/` (CPU reference). Modules not started yet hold only a `README.md`. |
| `tools/` | Parity harness, reference dumper (`llama-dump`), corpora, table generators |
| `web/` | Browser-side TypeScript: the book's scripts now, the wasm shim and `zllm.js` from M3 |
| `docs/` | This book; GitHub Pages serves it from `master` |
| `scripts/fixtures.sh` | Pinned fixture downloads |
| `scripts/d3d12.sh` | Builds the Windows binary and runs it on the D3D12 host from WSL |
| `third_party/dawn` | The vendored `webgpu.h` matching the pinned Dawn release |

## Checks

Run these before every commit; CI runs the same:

```sh
zig fmt --check . && zig build test
(cd web && bun run check)
scripts/fixtures.sh vocab && zig build check-tokenizer && zig build check-chat
scripts/fixtures.sh dawn && zig build gpu-check
(cd web && bun run build:book) && mdbook build docs
```

For changes to `gguf`, `graph` or `ref`, also run the fixture-backed checks
`zig build check-gguf` and `zig build check-forward`. For changes to `gpu`,
`kernels`, `plan`, `sampler`, `state` or `sched`, run `zig build check-gpu`.
These need the 1.5 GB model and a llama.cpp build (`scripts/fixtures.sh`), so
CI does not run them.

Run `check-gpu` on a real GPU: natively on Linux, or with
`-Dgpu-host=d3d12` on WSL. On llvmpipe it is correct but takes about half an
hour. llvmpipe is there for `gpu-check`, which takes seconds and is what CI
runs.

For kernel performance, `zllm bench <model.gguf>` (or `scripts/d3d12.sh
bench ...`) times decode steps and attributes them to kernel classes.

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
