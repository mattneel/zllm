# Velocity (native D3D12)

Velocity is zllm's native Direct3D 12 backend. It runs the same command
stream as the WebGPU backend ([GPU backend](gpu.md)), but with its own
hand-written HLSL kernels. It reaches D3D12 features WebGPU does not expose:

- LinAlg wave and cooperative matrices (Shader Model 6.10, Agility SDK
  preview).
- DirectStorage reads from the model file straight into GPU buffers.

## Platform layer

The core never calls a GPU API. It records a command stream that names
kernels by identity (`create_kernel`). A backend supplies:

- each kernel's implementation (its *artifact*);
- a device;
- a host that executes the stream.

`src/gpu/host.zig` is the platform layer: `gpu.Device` and `gpu.Host` over
every backend in `src/backend/backend.zig`. Every backend implements every
kernel and feature where it makes sense. Where it does not, the capability
matrix says so, with the reason.

| | WebGPU | Velocity |
|---|---|---|
| Kernels | WGSL (`src/kernels/`) | HLSL (`src/velocity/kernels/`) |
| Compiled by | Dawn or the browser, at `create_kernel` | DXC (`dxcompiler.dll`, pinned preview), at `create_kernel` |
| Host | `src/gpu/dawn_host.zig`; `web/src/zllm/gpu.ts` | `src/velocity/host.zig` over `host.cpp` |
| Weights | `write_file` from a mapping or HTTP range reads | `write_file` through DirectStorage |

## The native layer

`src/velocity/host.cpp` is a C interface (`velocity.h`) over D3D12 with
DirectStorage. It is built by `zig c++` for `x86_64-windows-gnu`.

- **Bindless:** every buffer has a raw SRV and a raw UAV in one
  shader-visible heap.
- **Bindings:** one root signature of 60 root constants. A kernel's binding
  *i* is three of them: the SRV index, the UAV index and the byte offset,
  for up to 19 bindings. The last three hold the dispatch's grid, which
  `num_groups()` returns (WGSL's `num_workgroups`, which HLSL lacks).
  `kernels/common.hlsli` reads them.
- **Device:** comes from the Agility SDK runtime through a device factory,
  with experimental shader models enabled on that factory only. Dawn in the
  same process keeps the system D3D12 runtime.

The Zig host keeps WebGPU's semantics:

- `write_buffer` and `write_file` land before the next submit's work.
- Dispatches are ordered by UAV barriers.
- Each `submit` blocks until the GPU has finished it.

Copies, clears, uploads and readbacks are the host's own kernels
(`copy.hlsl`, `fill.hlsl`), so every GPU access in a submission is a
compute UAV access. A submission's uploads run as a command list of their
own. Buffers return to the common state between the two lists, so kernels
read uploaded weights through SRVs (`ro`) and everything else through UAVs
(`rw`).

## Kernels

`src/velocity/kernels.zig` lists the kernels Velocity implements. Each one
keeps its WGSL counterpart's contract:

- the params block at binding 0;
- the same buffers at the same bindings;
- the same grid;
- the same algorithm and reduction order, so results match WebGPU's to
  the checks' tolerances, bitwise where the checks are bitwise.

Kernels are composed from parts the way the WGSL is: half-precision and
quantized types' decoders, the attention masks, and f16 rounding are
shared fragments (`.hlsli`). The quantized decode multiply is a template
(`matvec_quant.hlsl.hbs`) over the same variants WebGPU's tuner explores.

Velocity implements every kernel; a unit test holds it to that. A backend
that lacked one would mark it missing: dispatching it fails with
`error.UnsupportedKernel`, `gpu-check` reports `MISSING`, and models refuse
to load.

On the RTX 5090 Laptop, Velocity passes every parity gate:

| Gate | Result on Velocity |
|---|---|
| `gpu-check` | 69/69 kernel cases |
| `check-gpu` (Qwen3.5-0.8B, dense models) | prefill 1264/1264 and decode 320/320 top-1, max rel L2 1.4e-5; f16 KV 818/818; engine-check 15/15, including speculative decoding, pipelined decode and recovery after a lost device |
| `check-quant` (Granite 4.2 30B Q4_K_M) | decode 200/200 top-1, max rel L2 3.5e-5; prefill 870/890 against llama.cpp, the reference's own figure |
| `check-gyro` (Qwen3.8-Flash-Next Gyro-Coder) | 0 expert sets differ; logits median rel L2 1.7e-5; 66/66 top-1, 40/40 greedy; step splits and fusion bitwise identical; the 6,430-token prompt answers correctly |

Its speed is WebGPU's, since the kernels are the same algorithms: the
Gyro-Coder prefills at 66 tok/s and decodes at 28 tok/s. Velocity-specific
kernels (LinAlg wave matrices on the tensor cores, wave reductions, DP4a)
are the next step.

## Building and running

Velocity builds for Windows only, against pinned SDKs:

- Agility SDK 1.722.0-preview;
- DirectStorage 1.3.0;
- DXC 1.10.2605.37.

```sh
scripts/fixtures.sh dawn        # the D3D12 host stage
scripts/fixtures.sh velocity    # Velocity's SDKs, and its DLLs in the stage
ZLLM_BACKEND=velocity scripts/d3d12.sh gpu-info
ZLLM_BACKEND=velocity scripts/d3d12.sh gpu-check
```

`scripts/d3d12.sh` adds `-Dvelocity` when the SDKs are present.
`-Dgpu-host=velocity` runs the parity checks (`gpu-check`, `check-gpu`,
`check-quant`, `check-gyro`) on Velocity.

`ZLLM_BACKEND` selects the backend for every GPU command: `webgpu` (the
default) or `velocity`. `ZLLM_ADAPTER` picks Velocity's adapter, in DXGI's
high-performance order.

Velocity needs Windows Developer Mode for the experimental shader models.
