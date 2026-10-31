# Backends

zllm shares its Zig inference core across platforms. Backends provide the
kernel implementations, device access and command-stream execution. The
implemented backend identities in `src/backend/backend.zig` are `webgpu`
and `agility`; browser WebGPU and native Dawn are two hosts of `webgpu`,
not different inference engines.

The goal is to use each supported platform well without giving up the
reference checks. A shared kernel identity fixes parameters and bindings,
not the instruction sequence. Agility can use native wave operations where
the portable WGSL path uses a workgroup reduction.

## Capability matrix

**Implemented** means a path exists in the repository. It does not mean that
every adapter or operating system has been exercised. **Planned** is not an
available backend, option or performance claim.

| Capability | Browser WebGPU | Native Dawn | Agility | CUDA (planned) |
|---|---|---|---|---|
| Availability | Implemented: wasm core and TypeScript WebGPU host | Implemented: native core and dynamically loaded Dawn C API | Implemented: Windows D3D12, opt-in build | Not implemented; no backend identity, host or kernel catalog |
| GPU APIs | Browser-selected D3D12, Metal or Vulkan path | Dawn-selected D3D12, Vulkan or Metal path | D3D12 through the Agility SDK device factory | No runtime path |
| Kernel artifacts | Hand-written WGSL and rendered WGSL templates | Same WGSL catalog as the browser | Hand-written HLSL and templates, compiled by DXC to DXIL | No CUDA/PTX artifacts |
| Current kernel coverage | Full `kernels.Kernel` catalog | Full `kernels.Kernel` catalog | Full catalog in `src/agility/kernels.zig` | None |
| Supported model plans | Shared Qwen 3.5, dense and Qwen3.8-Flash-Next programs | Same programs | Same programs | Not available |
| Plan-built fusions | Implemented: stacked projections, norm/activation prologues, residual adds and model-specific combined kernels | Same shared plan construction | Same contracts with HLSL implementations | Not available |
| Fusion pass | Implemented: `hc_act` into quantized decode matvec, under single-reader and layout rules | Same pass | Same pass and HLSL prologue | Not available |
| Load-time tuning | Implemented: quantized decode matvec variants, correctness check before timing | Same sweep | Same sweep over HLSL variants | No catalog or sweep path |
| Persistent tuning cache | OPFS under `zllm/plans/` | Native JSON cache directory | Same native cache, separate runtime identity | No cache integration |
| f32 / f16 / FP8 KV | All implemented; f16 default | All implemented; f16 default | All implemented; f16 default | Not available |
| Automatic precision ladder | Not implemented; cache format is explicit | Same | Same | Not available |
| Native tensor-core / matrix-intrinsic kernels | Not implemented in current WGSL kernels | Not implemented in current WGSL kernels | Not implemented in current HLSL inference kernels; host queries LinAlg capabilities | No implementation |
| Native wave optimizations | Current WGSL kernels do not use subgroup operations | Same WGSL path | Implemented in `gyro_matvec` and `moe_logits` | No implementation |
| Model-file uploads | HTTP ranges or a local `File`, served by the shim | Ranges from a file mapping | DirectStorage range reads into GPU buffers | No file-upload path |
| DirectStorage | No: browser has no native DirectStorage interface | No: Dawn host uses mapped bytes and queue writes | Implemented in `write_file` | Not integrated |
| Device-lost recovery | Failed request rejects; next request rebuilds on a new owned device | Runtime rebuilds before returning the lost-device error | Same native runtime rebuild, with D3D12 error mapping and removed-device teardown | No error or recovery path |
| CUDA Graphs | Not applicable to this API | Not applicable to this API | Not applicable to this API | Not implemented |

The matrix describes backend capabilities. Model-specific restrictions still
apply: Qwen3.8-Flash-Next uses decode kernels in steps of at most four tokens,
reads PLE rows from the file for every step, and cannot chain sampled tokens
entirely on the GPU. It has no MTP speculation path. The Qwen 3.5 program can
use prefill kernels, pipelined decode and an MTP block when the model and
binding limits permit them. See [Generation](generation.md) and
[Gyro-Coder](gyro.md).

## Shared contracts

`src/gpu/cmd.zig` records kernel identity, variant, binding write set and
backend artifact in `create_kernel`. `Kernel.contract()` in
`src/kernels/kernels.zig` derives the binding count and write set from the
WGSL declarations. The Zig parameter structs and binding order are shared
with the HLSL implementations. Agility uses these contracts to place UAV
barriers only for conflicting buffer ranges.

A missing artifact is explicit, not a silent fallback. `kernels.Set.missing`
tracks it, dispatch fails with `error.UnsupportedKernel`, and the native
loader's `requireAll()` rejects incomplete catalogs. The Agility source
contains tests for complete kernel coverage and permitted HLSL writes.
Neither backend falls back to CPU inference for a missing kernel.

## Tuning, fusion and cache limits

The tuner currently searches the twelve `MatvecVariant` combinations for
quantized decode multiplies. It does not search every kernel, choose weight
formats, select tensor-core algorithms or discover arbitrary fusions. The
fuser handles the supported hyper-connection activation pattern; other
fusions are expressed directly by the plans. Tuning runs before program
construction, without a post-fusion sweep.

The plan cache is a JSON record of non-default variants, not compiled WGSL,
DXIL or serialized launch lists. A cache hit avoids the sweep, not weight
upload or pipeline creation. Native Dawn and Agility use different runtime
strings in the key; the browser supplies its runtime and adapter strings.

The key hashes the zllm version, the adapter, the runtime, the plan limits,
the GGUF header and index, the backend, and that backend's own kernel
sources (`kernels.backendSource`: WGSL and its templates on WebGPU, HLSL on
Agility), so a change to any kernel a backend runs invalidates its tuning.
It does not hash the model's tensor payload. The cache stores neither
precision choices nor an automated model-quality verdict.

## Precision and hardware features

FP8 KV storage is implemented in portable shader arithmetic on all three
hosts. It is E4M3FN packed four values per word, with an f32 power-of-two
scale per position and KV head, separately for K and V. This reduces cache
storage; it is not FP8 tensor-core compute. Attention decodes the values for
f32 arithmetic. f16 remains the default, and `--kv fp8` is an explicit
native option. See [Precision](precision.md) for the implemented formats
and the distinction from the planned quality-gated ladder.

Agility queries Shader Model 6.10 and LinAlg matrix capabilities and can
compile a shader that asks for `cs_6_10`. The current inference catalog does
not use LinAlg wave/cooperative matrix intrinsics. Its native optimizations
use wave reductions, lane exchange and a staged Gyro codebook instead. SDK
and capability plumbing should not be read as completed tensor-core kernels.

CUDA is a planned backend, not an alternative spelling of Agility. The
current backend enum, build integration and artifact selector contain no
CUDA path. There are no implemented PTX templates, CUDA Graphs, NVFP4 kernels
or CUDA FP8 compute kernels to document. No support or speed claim for
those mechanisms follows from the existing FP8 KV implementation.

The [roadmap](roadmap.md) separates the requested CUDA direction from the
implemented backends. PTX templates, CUDA Graphs and NVIDIA low-precision
compute belong to that future work, not this capability inventory.

## Platform evidence and selection

The existing validation record covers native Dawn on Linux llvmpipe and
Windows D3D12, browser Chrome on D3D12, and Agility on Windows with the RTX
5090 Laptop. Metal and hardware Vulkan are intended WebGPU paths, but the
existing documentation does not establish equivalent runs on those systems.
A software Vulkan adapter is useful for correctness checks, not evidence of
hardware Vulkan performance.

Native commands use `ZLLM_BACKEND=webgpu` by default or
`ZLLM_BACKEND=agility` when built with Agility support. `ZLLM_ADAPTER`
selects a Agility adapter in DXGI high-performance order. The browser always
uses WebGPU; selecting Agility is not a browser option.

For command encoding and kernels, see [GPU backend](gpu.md). For the native
D3D12 implementation and SDK setup, see [Agility](agility.md). Test gates
and recorded numerical results are in [Validation](validation.md).
