# kernels

WGSL compute kernels and their launch parameters (`kernels.zig`). Each kernel
is one `<name>.wgsl` file with a `main` entry point and a uniform `Params`
struct at binding 0, mirrored by an `extern struct` in `kernels.zig`. Weights
stay bf16 on the GPU (two per `u32`); activations and accumulation are f32.

`check.zig` checks every kernel on seeded random inputs against a float64 CPU
computation of the same formula: `zllm gpu-check` on any Dawn host
(`scripts/d3d12.sh gpu-check` for the Windows D3D12 host).

These are plain WGSL for M1. Templates with parameter schemas and cost models
replace them in M4; see the [spec](../../docs/src/spec.md#4-kernel-templates).
