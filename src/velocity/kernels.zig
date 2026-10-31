//! Velocity's kernels: for each kernels.Kernel it implements, its HLSL,
//! composed the way kernels.zig composes the WGSL: kernels/common.hlsli,
//! then the kernel's parts from kernels/ (see `composed`). Each keeps its
//! WGSL counterpart's contract (params block, bindings, grid) and is written
//! for Direct3D 12. Kernels not listed in `implemented` are missing on
//! Velocity (kernels.Set.missing).

const std = @import("std");
const kernels = @import("../kernels/kernels.zig");
const Kernel = kernels.Kernel;

const common = @embedFile("kernels/common.hlsli");

/// The kernels Velocity implements.
const implemented = [_]Kernel{
    .rmsnorm,
    .matmul_bf16,
    .matmul_f16,
    .matvec_bf16,
    .matvec_f16,
    .embed_bf16,
    .embed_f16,
    .matmul_reduce,
    .silu_mul,
    .repad,
    .mark_tokens,
    .rope_append,
    .attn_split,
    .attn_split_wide,
    .attn_split_sparse,
    .attn_combine,
    .qsa_keys,
    .qsa_pool,
    .qsa_score,
    .qsa_select,
    .gyro_matvec,
    .moe_logits,
    .moe_route,
    .rotate128,
    .moe_combine,
    .hc_act,
    .hc_mix,
    .hc_combine,
    .ple_mix,
    .ple_conv,
    .delta_prep,
    .delta_rule,
    .gated_norm,
    .draft,
    .sample,
    .matvec_q8_0,
    .matvec_q3_k,
    .matvec_q4_k,
    .matvec_q5_k,
    .matvec_q6_k,
    .matmul_q8_0,
    .matmul_q3_k,
    .matmul_q4_k,
    .matmul_q5_k,
    .matmul_q6_k,
    .embed_q8_0,
    .embed_q3_k,
    .embed_q4_k,
    .embed_q5_k,
    .embed_q6_k,
    .quantize_q8,
    .repack_q3_k,
};

/// Kernel `k`'s HLSL (for a template, kernels.Kernel.isTemplate, the
/// template's source), or null where Velocity does not implement it.
pub fn source(k: Kernel) ?[]const u8 {
    inline for (implemented) |i| {
        if (k == i) return comptime common ++ composed(@tagName(i));
    }
    return null;
}

/// A kernel's parts, as kernels.zig's `composed` for the WGSL:
///   <op>_bf16, <op>_f16   half/<h>.hlsli, <op>_half.hlsl
///   <op>_q<type>          quant/common.hlsli, quant/<type>.hlsli, for
///                         matvec quant/<type>_dot.hlsli and the template
///                         matvec_quant.hlsl.hbs, else <op>_quant.hlsl
///   attn_split[_wide]     #define MAX_GROUP, attn_dense.hlsli, attn_split.hlsl
///   attn_split_sparse     #define MAX_GROUP, attn_sparse.hlsli, attn_split.hlsl
///   rope_append, qsa_keys f16_round.hlsli, <name>.hlsl
///   anything else         <name>.hlsl
fn composed(comptime name: []const u8) []const u8 {
    @setEvalBranchQuota(10_000);
    for (std.enums.values(kernels.QuantOp)) |op| {
        for ([_][]const u8{ "bf16", "f16" }) |h| {
            if (std.mem.eql(u8, name, @tagName(op) ++ "_" ++ h)) {
                return @embedFile("kernels/half/" ++ h ++ ".hlsli") ++ @embedFile("kernels/" ++ @tagName(op) ++ "_half.hlsl");
            }
        }
    }
    for (std.enums.values(kernels.QuantOp)) |op| {
        const prefix = @tagName(op) ++ "_q";
        if (std.mem.startsWith(u8, name, prefix)) {
            const t = name[@tagName(op).len + 1 ..];
            const dot = if (op == .matvec) @embedFile("kernels/quant/" ++ t ++ "_dot.hlsli") else "";
            const body = if (op == .matvec) "matvec_quant.hlsl.hbs" else @tagName(op) ++ "_quant.hlsl";
            return @embedFile("kernels/quant/common.hlsli") ++ @embedFile("kernels/quant/" ++ t ++ ".hlsli") ++ dot ++ @embedFile("kernels/" ++ body);
        }
    }
    const group = std.fmt.comptimePrint;
    if (std.mem.eql(u8, name, "attn_split")) return group("#define MAX_GROUP {d}\n", .{kernels.attn_max_group}) ++ @embedFile("kernels/attn_dense.hlsli") ++ @embedFile("kernels/attn_split.hlsl");
    if (std.mem.eql(u8, name, "attn_split_wide")) return group("#define MAX_GROUP {d}\n", .{kernels.attn_max_group_wide}) ++ @embedFile("kernels/attn_dense.hlsli") ++ @embedFile("kernels/attn_split.hlsl");
    if (std.mem.eql(u8, name, "attn_split_sparse")) return group("#define MAX_GROUP {d}\n", .{kernels.attn_max_group_wide}) ++ @embedFile("kernels/attn_sparse.hlsli") ++ @embedFile("kernels/attn_split.hlsl");
    if (std.mem.eql(u8, name, "rope_append") or std.mem.eql(u8, name, "qsa_keys")) return @embedFile("kernels/f16_round.hlsli") ++ @embedFile("kernels/" ++ name ++ ".hlsl");
    return @embedFile("kernels/" ++ name ++ ".hlsl");
}

test "Velocity implements every kernel" {
    for (std.enums.values(Kernel)) |k| {
        const s = source(k) orelse return error.TestUnexpectedResult;
        try std.testing.expect(s.len > common.len);
    }
}
