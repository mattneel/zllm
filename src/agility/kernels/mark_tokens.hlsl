// Mark the prompt-token set consumed by the sampler's repetition penalty
// (kernels/mark_tokens.wgsl).
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint i = id.x + id.y * num_groups().x * 256u;
    if (i < param(11)) stu(2, ldu(1, i), 1u);
}
