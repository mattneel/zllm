// Host-internal: zero-fills u32 words (the command stream's clear_buffer).
// Root constants: destination UAV, byte offset, word count. Grid-strided like
// copy.hlsl.

cbuffer Fill : register(b0) { uint dst, dst_offset, words; }

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    RWByteAddressBuffer d = ResourceDescriptorHeap[dst];
    for (uint w = gid.x * 256 + lid.x; w < words; w += 256 * 65535) d.Store(dst_offset + 4 * w, 0);
}
