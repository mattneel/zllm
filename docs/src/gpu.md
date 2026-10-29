# GPU backend

The GPU path has three layers. The core records a **command stream**, a
**host** executes it on WebGPU, and **kernels** are WGSL compute shaders the
stream launches. M1 provides all three, with Dawn as the host, and runs the
full Qwen 3.5 forward pass on the GPU at parity with the CPU reference.

## Hosts

| Host | WebGPU | Adapter here | Used for |
|---|---|---|---|
| Native Linux | Dawn → Vulkan | llvmpipe (Mesa's software renderer) | CI, kernel correctness |
| Windows, run from WSL | Dawn → D3D12 | RTX 5090 Laptop | Local parity and timing |
| Browser (M3) | Chrome's Dawn → D3D12/Metal/Vulkan | | The product |

Dawn comes from the pinned prebuilt release. Its static archive is wrapped
into a shared library (`libwebgpu_dawn.so`, `webgpu_dawn.dll`) that exports
exactly the `webgpu.h` C API, and zllm loads it at run time. zllm never links
Dawn's C++, and the vendored header in `third_party/dawn` fixes the ABI.

```sh
scripts/fixtures.sh dawn          # Linux .so; under WSL with Visual Studio, also the DLL
zig build gpu-check               # every kernel vs a CPU computation (llvmpipe)
zig build gpu-check -Dgpu-host=d3d12
zig build check-gpu               # full forward pass vs the CPU reference
zig build check-gpu -Dgpu-host=d3d12
```

The D3D12 host needs Visual Studio (for `link.exe`) and the Windows SDK (for
DXC's `dxcompiler.dll`). Windows won't load DLLs from a `\\wsl.localhost`
path, so `scripts/fixtures.sh` stages the DLLs in `%LOCALAPPDATA%\zllm\gpu`.
`scripts/d3d12.sh` builds the Windows executable, copies it there and runs
it. On WSL, Dawn's Vulkan backend rejects Dozen (Vulkan on D3D12), so the
native Linux host falls back to llvmpipe.

## Command stream

The core never calls WebGPU. It appends commands to a `Recorder`
(`src/gpu/cmd.zig`), a host executes the batch, and the core consumes one
completion per `submit` and per `read_buffer`. The browser host will interpret
the same words, crossing the wasm/JS boundary once per batch rather than once
per dispatch.

- **Encoding:** little-endian `u32` words. Each command starts with
  `opcode | (words << 8)`, so a host can bounds-check every command.
- **Object ids:** objects are named by ids the core assigns, densely per kind.
- **Addresses:** buffer contents, WGSL source and readback destinations are
  core-memory addresses, two words each; in wasm32 the high word is 0.
- **Batching:** commands between two `submit`s go into one fresh command
  buffer, and consecutive dispatches share one compute pass.

| Op | Arguments |
|---|---|
| `create_buffer` | id, size, usage (WebGPU `GPUBufferUsage` bits) |
| `write_buffer` | buffer, offset, data, length |
| `create_shader` | id, WGSL source (entry point `main`) |
| `create_pipeline` | id, shader |
| `create_bind_group` | id, pipeline, group, then (binding, buffer, offset, size)… |
| `dispatch` | pipeline, workgroups x/y/z, bind groups |
| `copy_buffer` | src, offset, dst, offset, size |
| `read_buffer` | token, buffer, offset, size, destination |
| `submit` | token |
| `destroy` | kind, id |

## Kernels

Kernels live in `src/kernels/*.wgsl`, each with a uniform `Params` struct
mirrored in `kernels.zig`. Weights stay BF16 on the GPU, packed two per `u32`
and widened in the shader; activations and accumulation are f32.

| Kernel | Work | Notes |
|---|---|---|
| `embed_bf16` | Token embedding gather | Per table chunk |
| `rmsnorm` | Row RMSNorm | One workgroup per row |
| `matmul_bf16` | x · Wᵀ | 16 tokens × 64 outputs per workgroup; tiles staged in workgroup memory |
| `silu_mul`, `add` | SwiGLU, residual | Elementwise |
| `qk_norm_rope` | Per-head Q/K RMSNorm, then partial NeoX RoPE | Reads a host-built (cos, sin) table: WGSL trig is only accurate near [−π, π] |
| `attention` | Causal GQA with the sigmoid output gate | Scores in workgroup memory (≤ 4096 tokens) |
| `conv_silu`, `l2norm_heads`, `gates` | DeltaNet front end | |
| `delta_rule` | Gated delta rule, then SiLU(z)-gated RMSNorm | One workgroup per value head, one thread per state column |

These are plain WGSL in M1, written to be correct and reasonably fast. M4
replaces them with templates that are tuned per GPU.

## Weight layout

`src/plan/qwen35.zig` records the forward pass. Every storage binding must
fit `maxStorageBufferBindingSize`, which is 128 MiB on llvmpipe and 2 GiB on
the 5090:

- **Weight matrices** are split by rows into chunks; each chunk is its own
  buffer and its own dispatch.
- **Matrix multiplies** also split the batch into token groups whose input
  and output each fit one binding, at offsets that satisfy
  `minStorageBufferOffsetAlignment`. The 248,320-row vocabulary projection
  needs both splits on llvmpipe.
- **DeltaNet state:** each DeltaNet layer has its own recurrent state buffer,
  which M2 keeps across steps.
