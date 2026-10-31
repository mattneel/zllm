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
  *i* is three of them: the SRV index, the UAV index and the byte offset.
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
- the same grid.

Dispatching a kernel Velocity lacks fails with `error.UnsupportedKernel`.
`gpu-check` reports such checks as `MISSING`, and models refuse to load on
Velocity until every kernel is ported.

Ported: `rmsnorm`. The rest of the 47 kernels are in progress, Gyro-Coder's
set first.

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

`ZLLM_BACKEND` selects the backend for every GPU command: `webgpu` (the
default) or `velocity`. `ZLLM_ADAPTER` picks Velocity's adapter, in DXGI's
high-performance order.

Velocity needs Windows Developer Mode for the experimental shader models.
