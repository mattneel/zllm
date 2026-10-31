// Host-internal: copies u32 words between two buffers through their UAVs
// (either may be written in the same submission). Serves the command
// stream's copy_buffer, and write_buffer and read_buffer through host-memory
// staging. Internal kernels take root constants directly, not the binding
// layout of common.hlsli: source UAV, source byte offset, destination UAV,
// destination byte offset, word count. Grid-strided over at most 65535
// groups, so any word count fits one dispatch.

cbuffer Copy : register(b0) { uint src, src_offset, dst, dst_offset, words; }

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    RWByteAddressBuffer s = ResourceDescriptorHeap[src];
    RWByteAddressBuffer d = ResourceDescriptorHeap[dst];
    for (uint w = gid.x * 256 + lid.x; w < words; w += 256 * 65535) d.Store(dst_offset + 4 * w, s.Load(src_offset + 4 * w));
}
