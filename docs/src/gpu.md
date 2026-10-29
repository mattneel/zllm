# GPU backend

The GPU path has three layers. The core records a **command stream**, a
**host** executes it on WebGPU, and **kernels** are WGSL compute shaders the
stream launches. M1 provides all three, with Dawn as the host, and runs the
full Qwen 3.5 forward pass on the GPU at parity with the CPU reference. M2
turns the forward pass into a **step program**: prebuilt launches that
process any number of tokens against a KV cache and the DeltaNet state, and
sample the next token on the GPU ([Generation](generation.md)).

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
zig build check-gpu               # forward pass vs the CPU reference, step splits, engine
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
| `clear_buffer` | buffer, offset, size (zero-fill, ordered with dispatches) |

## Kernels

Kernels live in `src/kernels/*.wgsl`, each with a uniform `Params` struct
mirrored in `kernels.zig`. Weights stay BF16 on the GPU, packed two per `u32`
and widened in the shader; activations and accumulation are f32.

| Kernel | Work | Notes |
|---|---|---|
| `embed_bf16` | Token embedding gather | Per table chunk |
| `rmsnorm` | Row RMSNorm | One workgroup per row |
| `matmul_bf16` | x · Wᵀ, prefill | 16 tokens × 64 outputs per workgroup; tiles staged in workgroup memory |
| `matvec_bf16` | x · Wᵀ, decode | 8 rows per workgroup, each row's dot product over 32 lanes reading 16-byte weight vectors, then a fixed tree |
| `silu_mul`, `add` | SwiGLU, residual | Elementwise |
| `qk_norm_rope` | Per-head Q/K RMSNorm, then partial NeoX RoPE at positions pos0 + t | Reads a host-built (cos, sin) table: WGSL trig is only accurate near [−π, π] |
| `kv_append` | K and V rows into the cache at pos0 + t | f16 pairs (`pack2x16float`) or f32 |
| `attention` | Causal GQA over the cache, with the sigmoid output gate, prefill | One workgroup per (head, query); online softmax over 256-key tiles; no length limit |
| `attn_split`, `attn_combine` | The same, decode (flash-decoding) | One workgroup per (KV head, 64 cache positions, token) scores the block for the whole GQA group and takes a local softmax; the combine merges the blocks |
| `conv_silu` | DeltaNet causal conv | One thread per channel; reads and advances the last 3 inputs |
| `l2norm_heads`, `gates` | DeltaNet front end | |
| `delta_rule` | Gated delta rule | Each thread keeps 16 rows of one state column in registers for the whole step; a head spans 4 workgroups |
| `gated_norm` | The SiLU(z)-gated RMSNorm after it | One workgroup per (head, token) |
| `mark_tokens` | The prompt's token set, for the repetition penalty | |
| `sample` | Bias, penalties, temperature, min-p, top-k, top-p, draw | In phases, one dispatch each; see [Generation](generation.md#sampling) |

These are plain WGSL. M4 makes them templates tuned per GPU; the decode
kernels above are its first part, measured with `zllm bench`
([Generation](generation.md#performance)).

## Step program

`src/plan/qwen35.zig` builds everything a step needs once, as the spec's
execution plan asks:

- **Buffers:** activations sized for the largest step (512 tokens), a KV
  cache per attention layer sized for the context, every DeltaNet layer's
  recurrent state and conv history in one buffer, checkpoint copies of that
  buffer, and the sampler's buffers.
- **Launches:** every dispatch with its bind group and parameters, which are
  packed into shared 64 KiB uniform buffers.

A step is then one `write_buffer` of a 16-byte `Step` uniform
(`m` tokens, first position `pos0`, the sampler's draw `u`), the token ids
for prefill, and a replay of the launch list:

- **Step-dependent kernels:** kernels that mix tokens or use positions
  (RoPE, the cache append, attention, the convolution, the delta rule, the
  sampler) read the `Step` uniform.
- **Row-local kernels:** the rest take capacity-sized parameters, and their
  grid covers only the step's `m` rows.
- **Logits:** the final norm's last rows are copied into the vocabulary
  projection's input, so decoding projects one row, not 512.

One `submit` per step keeps the step uniform unambiguous: queued writes land
before the submission they precede.

### Two kernel paths

Steps of at most 4 tokens, decoding, run the **decode kernels**:
`matvec_bf16` for every multiply, and split attention. Longer steps run the
**prefill kernels**. The plan prebuilds every multiply and attention both
ways, and a step picks one (`plan.Path`).

Within a path, every row's arithmetic is the same whatever the step's size.
A query walks the same key tiles from position 0, and each multiply output
reduces in the same order, so splitting a sequence into steps of any size
gives bitwise identical results. The two paths reduce in different orders,
so a position processed through one differs from the same position through
the other by rounding, about 1e-5 relative. `check-gpu` checks both paths
against the CPU reference, and each path's splits bitwise.

Every storage binding must fit `maxStorageBufferBindingSize`, which is
128 MiB on llvmpipe and 2 GiB on the 5090:

- **Weight matrices** are split by rows into chunks; each chunk is its own
  buffer and its own dispatch.
- **The vocabulary projection** runs over at most `logit_rows` rows at a
  time (1 when generating). A check that needs every position's logits gets
  them in passes.
- **Recurrent state** is bound per layer at offsets aligned to
  `minStorageBufferOffsetAlignment`.
