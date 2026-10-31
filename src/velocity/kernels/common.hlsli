// zllm's HLSL kernel header for Velocity, prefixed to every kernel's source.
//
// Velocity binds bindlessly: the root signature is 60 32-bit root constants at
// b0, and every buffer has a raw SRV and a raw UAV in one shader-visible heap.
// Binding i of a kernel (its WGSL @binding(i): 0 is the params block, then the
// kernel's buffers in contract order) is three constants: the buffer's SRV
// index, its UAV index, and the binding's byte offset into the buffer, for
// up to 19 bindings. Constants 57..59 hold the dispatch's grid (WGSL's
// num_workgroups), which HLSL has no builtin for.
//
// Use ro(i) only for buffers that only uploads write (write_buffer,
// write_file: weights, tables): D3D12 promotes a buffer to one access state
// per command list, and the host runs a submission's uploads as a list of
// their own, so uploaded buffers start the kernels' list in the common
// state. Use rw(i) for everything kernels, copies or clears write. Waves
// have 32 lanes on the devices Velocity targets; kernels that depend on it
// say so.

cbuffer Bindings : register(b0) { uint4 bindings[15]; }

uint binding_word(uint i) { return bindings[i >> 2][i & 3]; }
ByteAddressBuffer ro(uint i) { return ResourceDescriptorHeap[NonUniformResourceIndex(binding_word(3 * i))]; }
RWByteAddressBuffer rw(uint i) { return ResourceDescriptorHeap[NonUniformResourceIndex(binding_word(3 * i + 1))]; }
/// Byte offset of binding i into its buffer.
uint base(uint i) { return binding_word(3 * i + 2); }
/// The dispatch's group counts (WGSL's num_workgroups).
uint3 num_groups() { return uint3(bindings[14][1], bindings[14][2], bindings[14][3]); }

/// The params block's u32 word `w` (binding 0).
uint param(uint w) { return rw(0).Load(base(0) + 4 * w); }
float paramf(uint w) { return asfloat(param(w)); }

/// f32 element `e` of binding i.
float ldf(uint i, uint e) { return asfloat(rw(i).Load(base(i) + 4 * e)); }
void stf(uint i, uint e, float v) { rw(i).Store(base(i) + 4 * e, asuint(v)); }
uint ldu(uint i, uint e) { return rw(i).Load(base(i) + 4 * e); }
void stu(uint i, uint e, uint v) { rw(i).Store(base(i) + 4 * e, v); }
/// u32 word `w` of read-only binding i, and as f32.
uint ldw(uint i, uint w) { return ro(i).Load(base(i) + 4 * w); }
float ldwf(uint i, uint w) { return asfloat(ldw(i, w)); }
