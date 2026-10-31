//! Agility's kernels: for each kernels.Kernel it implements, its HLSL,
//! composed the way kernels.zig composes the WGSL: kernels/common.hlsli,
//! then the kernel's parts from kernels/ (see `composed`). Each keeps its
//! WGSL counterpart's contract (params block, bindings, grid) and is written
//! for Direct3D 12. Kernels not listed in `implemented` are missing on
//! Agility (kernels.Set.missing).

const std = @import("std");
const kernels = @import("../kernels/kernels.zig");
const Kernel = kernels.Kernel;

const common = @embedFile("kernels/common.hlsli");

/// The kernels Agility implements.
const implemented = [_]Kernel{
    .rmsnorm,
    .matmul_bf16,
    .matmul_f16,
    .matmul_q8_0_float,
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
    .gyro_matmul,
    .moe_group,
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
    .repack_q8_0,
};

/// Kernel `k`'s HLSL (for a template, kernels.Kernel.isTemplate, the
/// template's source), or null where Agility does not implement it.
pub fn source(k: Kernel) ?[]const u8 {
    inline for (implemented) |i| {
        if (k == i) return comptime common ++ composed(@tagName(i));
    }
    return null;
}

/// The queried wave32, 16×16×16 FP16-input/FP32-accumulate operation.
/// Emulated operands are not a native tensor-core implementation.
pub const LinAlgCaps = struct {
    shader_model: u32,
    tier: u32,
    wave_flags: u32,
    experimental: bool,

    pub fn supported(caps: LinAlgCaps) bool {
        return caps.experimental and caps.shader_model >= 0x6a and
            caps.tier >= 0x10 and caps.wave_flags & 0x7 == 0x1;
    }
};

/// A device-selected implementation of the same identity, grid and contract.
/// Selection happens at pipeline creation, after the native capability query.
pub fn linAlgSource(k: Kernel) ?[]const u8 {
    return switch (k) {
        .matmul_q8_0_float => common ++ @embedFile("kernels/matmul_q8_0_float_linalg.hlsl"),
        .gyro_matmul => common ++ @embedFile("kernels/gyro_matmul_linalg.hlsl"),
        else => null,
    };
}

/// Hash both possible implementations: capability selection must not hide a
/// shader change from the persistent tuning cache.
pub fn cacheSource(k: Kernel) []const u8 {
    inline for (implemented) |i| {
        if (k == i) return comptime (source(i).? ++ (linAlgSource(i) orelse ""));
    }
    return "";
}

test "LinAlg selection requires every native operation capability" {
    const supported: LinAlgCaps = .{ .shader_model = 0x6a, .tier = 0x10, .wave_flags = 1, .experimental = true };
    try std.testing.expect(supported.supported());
    var mutated = supported;
    mutated.shader_model = 0x69;
    try std.testing.expect(!mutated.supported());
    mutated = supported;
    mutated.tier = 0;
    try std.testing.expect(!mutated.supported());
    mutated = supported;
    mutated.experimental = false;
    try std.testing.expect(!mutated.supported());
    for ([_]u32{ 0, 2, 3, 5, 7, 8 }) |flags| {
        mutated = supported;
        mutated.wave_flags = flags;
        try std.testing.expect(!mutated.supported());
    }
}

test "LinAlg variants preserve the complete WGSL write set" {
    for (std.enums.values(Kernel)) |k| {
        if (linAlgSource(k)) |code| {
            try std.testing.expectEqual(k.contract().writes, hlslWrites(code[common.len..]));
            try std.testing.expect(std.mem.indexOf(u8, code, "\n// agility: cs_6_10\n") != null);
        }
    }
}

/// A kernel's parts, as kernels.zig's `composed` for the WGSL:
///   <op>_bf16, <op>_f16   half/<h>.hlsli, <op>_half.hlsl
///   <op>_q<type>          quant/common.hlsli, quant/<type>.hlsli, for
///                         matvec quant/<type>_dot.hlsli and the template
///                         matvec_quant.hlsl.hbs, else <op>_quant.hlsl
///   attn_split[_wide]     #define MAX_GROUP, fp8.hlsli, attn_dense.hlsli,
///                         attn_split.hlsl
///   attn_split_sparse     #define MAX_GROUP, fp8.hlsli, attn_sparse.hlsli,
///                         attn_split.hlsl
///   rope_append           f16_round.hlsli, fp8.hlsli, rope_append.hlsl
///   qsa_keys              f16_round.hlsli, qsa_keys.hlsl
///   anything else         <name>.hlsl
fn composed(comptime name: []const u8) []const u8 {
    @setEvalBranchQuota(10_000);
    if (std.mem.eql(u8, name, "matmul_q8_0_float")) return @embedFile("kernels/matmul_q8_0_float.hlsl");
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
    if (std.mem.eql(u8, name, "attn_split")) return group("#define MAX_GROUP {d}\n", .{kernels.attn_max_group}) ++ @embedFile("kernels/fp8.hlsli") ++ @embedFile("kernels/attn_dense.hlsli") ++ @embedFile("kernels/attn_split.hlsl");
    if (std.mem.eql(u8, name, "attn_split_wide")) return group("#define MAX_GROUP {d}\n", .{kernels.attn_max_group_wide}) ++ @embedFile("kernels/fp8.hlsli") ++ @embedFile("kernels/attn_dense.hlsli") ++ @embedFile("kernels/attn_split.hlsl");
    if (std.mem.eql(u8, name, "attn_split_sparse")) return group("#define MAX_GROUP {d}\n", .{kernels.attn_max_group_wide}) ++ @embedFile("kernels/fp8.hlsli") ++ @embedFile("kernels/attn_sparse.hlsli") ++ @embedFile("kernels/attn_split.hlsl");
    if (std.mem.eql(u8, name, "rope_append")) return @embedFile("kernels/f16_round.hlsli") ++ @embedFile("kernels/fp8.hlsli") ++ @embedFile("kernels/rope_append.hlsl");
    if (std.mem.eql(u8, name, "qsa_keys")) return @embedFile("kernels/f16_round.hlsli") ++ @embedFile("kernels/qsa_keys.hlsl");
    return @embedFile("kernels/" ++ name ++ ".hlsl");
}

test "Agility implements every kernel" {
    for (std.enums.values(Kernel)) |k| {
        const s = source(k) orelse return error.TestUnexpectedResult;
        try std.testing.expect(s.len > common.len);
    }
}

/// The bindings `hlsl` writes: stf(N, ...), stu(N, ...), and
/// rw(N).Store* / rw(N).Interlocked* with a literal binding N (the way
/// every kernel past common.hlsli writes).
fn hlslWrites(hlsl: []const u8) u32 {
    var writes: u32 = 0;
    for ([_][]const u8{ "stf(", "stu(", "rw(" }) |call| {
        var at: usize = 0;
        while (std.mem.indexOfPos(u8, hlsl, at, call)) |i| {
            at = i + call.len;
            if (i > 0 and (std.ascii.isAlphanumeric(hlsl[i - 1]) or hlsl[i - 1] == '_')) continue;
            var end = at;
            while (end < hlsl.len and std.ascii.isDigit(hlsl[end])) end += 1;
            if (end == at) continue;
            const rest = hlsl[end..];
            const write = if (call[1] == 'w')
                std.mem.startsWith(u8, rest, ").Store") or std.mem.startsWith(u8, rest, ").Interlocked")
            else
                std.mem.startsWith(u8, rest, ",");
            if (write) writes |= @as(u32, 1) << @intCast(std.fmt.parseInt(u5, hlsl[at..end], 10) catch unreachable);
        }
    }
    return writes;
}

test "every kernel writes only the bindings its WGSL contract declares read_write" {
    for (std.enums.values(Kernel)) |k| {
        const writes = hlslWrites(source(k).?[common.len..]);
        const allowed = k.contract().writes;
        if (writes & ~allowed != 0) {
            std.debug.print("{t}: HLSL writes bindings 0x{x}, its contract allows 0x{x}\n", .{ k, writes, allowed });
            return error.TestUnexpectedResult;
        }
        try std.testing.expect(writes != 0);
    }
}

test hlslWrites {
    try std.testing.expectEqual(@as(u32, 0b1010), hlslWrites("stf(3, i, v); x = ldf(2, i); stu(1, 0, w); rw(2).Load(0);"));
    try std.testing.expectEqual(@as(u32, 0b100), hlslWrites("rw(2).InterlockedAdd(0, 1, o); mystf(5, 1);"));
}

test "grouped Gyro HLSL writes exactly its WGSL write set" {
    inline for (.{ Kernel.moe_group, Kernel.gyro_matmul }) |k| {
        try std.testing.expectEqual(k.contract().writes, hlslWrites(source(k).?[common.len..]));
    }
}
