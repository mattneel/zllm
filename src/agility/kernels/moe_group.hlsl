// Stable original routed slots per expert; same contract as moe_group.wgsl.
// Params: n_used, n_expert, max_slots, pad. Final binding is Step.
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint expert = gid.x;
    if (expert >= param(1)) return;
    uint n = 0u;
    for (uint slot = 0u; slot < ldu(4, 0) * param(0); slot++) {
        if (ldu(1, slot) == expert) {
            stu(3, expert * param(2) + n, slot);
            n++;
        }
    }
    stu(2, expert, n);
}
