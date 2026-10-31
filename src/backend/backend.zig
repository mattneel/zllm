//! The platforms zllm runs on. The core is the same on every one: graphs,
//! plans, the fuser, the scheduler, the sampler, tuning and the plan cache
//! record a command stream (gpu/cmd.zig) that names kernels by identity
//! (kernels.Kernel, with the params struct and bindings that fix their
//! contract). A backend supplies, for every kernel, its own hand-written
//! implementation (its artifact: WGSL on WebGPU, DXIL on Velocity), its
//! fragments for the fuser and its variant catalogs for tuning, and a host
//! that executes the stream.
//!
//! Every backend implements every feature and kernel where it makes sense;
//! where it does not, the capability matrix (docs) says so, with the reason.

pub const Backend = enum {
    /// WebGPU: Dawn natively, the browser through the JS shim. WGSL.
    webgpu,
    /// Velocity: native Direct3D 12 with the Agility SDK (LinAlg and wave
    /// matrices where the device has them) and DirectStorage. HLSL/DXIL.
    velocity,
};
