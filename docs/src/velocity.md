# Velocity (native D3D12)

Velocity is zllm's native Direct3D 12 backend. It runs the same Zig plans
and command stream as the WebGPU backend ([GPU backend](gpu.md)), with
hand-written HLSL kernels compiled by DXC.

DirectStorage model-file reads are implemented. The host also queries
LinAlg wave and matrix capabilities through the preview Agility SDK, but
the current inference kernels do not use LinAlg matrix intrinsics or
tensor-core compute. Native wave reductions and lane exchange are already
used in the routed-expert and router kernels. See the
[backend matrix](backends.md) for implemented features and remaining gaps.

## Platform layer

The core never calls a GPU API. It records a command stream that names
kernels by identity (`create_kernel`). A backend supplies:

- each kernel's implementation (its *artifact*);
- a device;
- a host that executes the stream.

`src/gpu/host.zig` is the native platform layer: `gpu.Device` and `gpu.Host`
over the identities in `src/backend/backend.zig`. Graphs, plans, fusion
legality, tuning, the scheduler and sampling stay in the Zig core. The
backend supplies an artifact for each kernel identity, not a separate
model implementation.

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
- Dispatches are ordered by UAV barriers where buffer-range hazards require them.
- Each `submit` blocks until the GPU has finished it.

Each kernel's contract (`Kernel.contract`) comes from its WGSL's binding
declarations: how many bindings, and which it writes (`var<storage,
read_write>`). `create_kernel` carries the write set. A unit test holds every
HLSL kernel to it: a kernel may write only the bindings its contract
declares. The host orders operations by it: a UAV barrier goes between two
operations only where the later one touches a buffer range one since the
last barrier wrote, or writes a range one of them read. While profiling,
every operation is serialized, so each dispatch's time is its own. On the
RTX 5090 the elided barriers are not measurable: decode chains dependent
kernels, and a barrier between dependent dispatches costs next to nothing.

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
- the defined operation and numerical checks; native reductions may differ
  from WGSL within the checks' tolerances, while bitwise gates still apply
  to the paths they compare.

Kernels are composed from parts the way the WGSL is: half-precision and
quantized types' decoders, the attention masks, and f16 rounding are
shared fragments (`.hlsli`). The quantized decode multiply is a template
(`matvec_quant.hlsl.hbs`) over the same variants WebGPU's tuner explores.

Velocity implements every current kernel identity; the source contains a
coverage test for that catalog. A missing artifact is explicit:
`kernels.Set.missing` records it, dispatch fails with
`error.UnsupportedKernel`, `gpu-check` reports `MISSING`, and native model
loads call `requireAll()` to reject an incomplete backend.

The load-time tuner renders the same twelve `MatvecVariant` candidates
from the HLSL template. The shared fuser can fold `hc_act` into the
quantized multiply's prologue under its single-reader and layout rules.
Native tuning is cached through the same JSON cache as Dawn, with
`velocity` as the runtime identity. The current key hashes the WGSL
catalog, not the HLSL catalog: an HLSL-only change with an unchanged
version can reuse an old tuning entry. See [Plan cache](gpu.md#plan-cache).

### Native optimizations

Velocity is no longer only a translation of the initial WGSL algorithms:

- `gyro_matvec` computes 32 rows per workgroup. It stages both the input
  and the codebook in groupshared memory, exchanges stream bytes with
  `WaveReadLaneAt`, and reduces rows with `WaveActiveSum` in 32-lane waves.
  WGSL keeps the codebook in storage because staging it alongside the
  input would exceed WebGPU's guaranteed workgroup memory.
- `moe_logits` assigns one workgroup per router row, reads vectors and
  reduces with wave sums. WGSL uses a workgroup tree.
- Both backends repack Q8_0 matrix rows at load into i8 values followed
  by f16 scales, padded for aligned vector reads. Per-step PLE reads
  retain their original file layout.
- Kernel contracts let the host omit barriers between independent accesses;
  profiling deliberately serializes operations for isolated timings.

LinAlg wave/cooperative matrix kernels remain future work, not a property
of these wave reductions. `host.cpp` queries Shader Model 6.10, the
LinAlg tier and matrix-operation support flags. `host.zig` can target
`cs_6_10` when a shader requests it; the current catalog uses `cs_6_6`.

### Recorded validation and timing

The following table preserves the existing RTX 5090 Laptop validation
record from the full HLSL port. Counts describe that revision's gates, not
a test run of the current checkout:

| Gate | Recorded result on Velocity |
|---|---|
| `gpu-check` | 69/69 kernel cases at the full-port revision, before the later FP8 cases |
| `check-gpu` (Qwen3.5-0.8B, dense models) | prefill 1264/1264 and decode 320/320 top-1, max rel L2 1.4e-5; f16 KV 818/818; engine-check 15/15, including speculation, pipelined decode and lost-device recovery |
| `check-quant` (Granite 4.2 30B Q4_K_M) | decode 200/200 top-1, max rel L2 3.5e-5; prefill 870/890 against llama.cpp, the reference's own figure |
| `check-gyro` (Qwen3.8-Flash-Next Gyro-Coder) | 0 expert sets differ; logits median rel L2 1.7e-5; 66/66 top-1, 40/40 greedy; step splits and fusion bitwise identical; the 6,430-token prompt answers correctly |

FP8 KV was added after the full port (`ecec4af`). The cache encoding is
E4M3FN with a power-of-two scale per position and KV head, implemented in
`fp8.hlsli` and checked byte-for-byte against the CPU encoder. That commit
records 1233/1264 top-1 agreement with the f32 reference on Velocity's
small-model evaluation, versus 1231/1264 on Dawn. f16 remains the default.
FP8 KV storage does not imply FP8 matrix compute or automatic
quality-gated precision selection.

The original full-port timing was 66 tok/s prefill and 28 tok/s decode on
Gyro-Coder. Later commit messages record these narrower changes on the
RTX 5090; they are historical measurements, not a combined benchmark or
a promise for another adapter:

| Change | Recorded result | Commit |
|---|---|---|
| Split Q8_0 rows and vectorized decode tasks | After the 6,430-token prompt: Velocity 30.6 → 40.5 tok/s; Dawn 35.9 tok/s | `823d6de` |
| More rows per expert workgroup, staged codebook and wave operations | Routed experts on Velocity: 4.7 → 3.2 ms per decode step | `d406cae` |
| One word load per f16 scale | Velocity decode: 23.4–23.7 → 22.2–22.5 ms per step | `30362f6` |
| One workgroup per router row | Velocity routing: 1.56 → 0.3 ms per decode step | `1713119` |

For quality gates and their tolerances, see [Validation](validation.md).

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

Windows Developer Mode is needed to enable the experimental shader-model
features used by LinAlg. The host attempts to enable them on its device
factory, records whether that succeeds, and queries the capabilities.
The current Zig device-open path does not set `require_linalg`; the
implemented `cs_6_6` kernels do not require LinAlg support. The preview
SDK setup alone is not evidence that matrix kernels are running.
