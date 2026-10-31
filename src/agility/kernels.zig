//! Agility's kernels: for each kernels.Kernel it implements, its HLSL,
//! composed the way kernels.zig composes the WGSL: kernels/common.hlsli,
//! then the kernel's parts from kernels/ (see `composed`). Each keeps its
//! WGSL counterpart's contract (params block, bindings, grid) and is written
//! for Direct3D 12. Kernels not listed in `implemented` are missing on
//! Agility (kernels.Set.missing).

const std = @import("std");
const kernels = @import("../kernels/kernels.zig");
const Kernel = kernels.Kernel;
const precision = @import("../precision.zig");

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

/// The compensated implementation of the same identity, grid and contract.
/// Kept separate from the explicit runtime rung selection below.
pub fn linAlgSource(k: Kernel) ?[]const u8 {
    return switch (k) {
        .matmul_q8_0_float => common ++ @embedFile("kernels/matmul_q8_0_float_linalg.hlsl"),
        .gyro_matmul => common ++ @embedFile("kernels/gyro_matmul_linalg.hlsl"),
        .matmul_bf16 => common ++ "#define SINGLE_PRODUCT 0\n" ++ @embedFile("kernels/half/bf16.hlsli") ++ @embedFile("kernels/matmul_half_linalg.hlsl"),
        .matmul_f16 => common ++ "#define SINGLE_PRODUCT 0\n" ++ @embedFile("kernels/half/f16.hlsli") ++ @embedFile("kernels/matmul_half_linalg.hlsl"),
        else => null,
    };
}

/// The only kernel identities changed by prefill precision selection.
pub const prefill_kernels = [_]Kernel{ .matmul_q8_0_float, .gyro_matmul, .matmul_bf16, .matmul_f16 };

/// Reserved command-stream variants. Valid matvec IDs have nonzero lanes
/// in bits 8..15; these IDs keep their entire low half zero.
pub fn prefillVariant(rung: precision.Prefill) u32 {
    return switch (rung) {
        .scalar_f32 => 0,
        .fp16_single => 0xfffc0000,
        .fp16_compensated => 0xffff0000,
    };
}

pub fn prefillFromVariant(id: u32) ?precision.Prefill {
    return switch (id) {
        0 => .scalar_f32,
        0xfffc0000 => .fp16_single,
        0xffff0000 => .fp16_compensated,
        else => null,
    };
}

/// Validate identities independently of shader compilation. Unsupported
/// native operands fail closed rather than falling back to scalar source.
pub fn validatePrefillVariant(k: Kernel, variant: u32, caps: LinAlgCaps) error{ BadCommand, UnsupportedPrefill }!void {
    if (linAlgSource(k) != null) {
        const rung = prefillFromVariant(variant) orelse return error.BadCommand;
        if (rung != .scalar_f32 and !caps.supported()) return error.UnsupportedPrefill;
    } else if (prefillFromVariant(variant)) |rung| {
        if (rung != .scalar_f32) return error.BadCommand;
    }
}

/// Real FP16 matrix implementations; scalar is always the original source.
pub fn prefillSource(k: Kernel, rung: precision.Prefill) ?[]const u8 {
    return switch (rung) {
        .scalar_f32 => source(k),
        .fp16_compensated => linAlgSource(k),
        .fp16_single => switch (k) {
            .matmul_q8_0_float => common ++ @embedFile("kernels/matmul_q8_0_float_linalg_single.hlsl"),
            .gyro_matmul => common ++ @embedFile("kernels/gyro_matmul_linalg_single.hlsl"),
            .matmul_bf16 => common ++ "#define SINGLE_PRODUCT 1\n" ++ @embedFile("kernels/half/bf16.hlsli") ++ @embedFile("kernels/matmul_half_linalg.hlsl"),
            .matmul_f16 => common ++ "#define SINGLE_PRODUCT 1\n" ++ @embedFile("kernels/half/f16.hlsli") ++ @embedFile("kernels/matmul_half_linalg.hlsl"),
            else => null,
        },
    };
}

/// Hash every implementation: selecting another rung must not hide a shader
/// change from the persistent tuning cache.
pub fn cacheSource(k: Kernel) []const u8 {
    inline for (implemented) |i| {
        if (k == i) return comptime (source(i).? ++
            (prefillSource(i, .fp16_single) orelse "") ++
            (prefillSource(i, .fp16_compensated) orelse ""));
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
        for ([_]precision.Prefill{ .fp16_single, .fp16_compensated }) |rung| {
            if (prefillSource(k, rung)) |code| {
                try std.testing.expectEqual(k.contract().writes, hlslWrites(code[common.len..]));
                try std.testing.expect(std.mem.indexOf(u8, code, "\n// agility: cs_6_10\n") != null);
            }
        }
    }
}

test "prefill variants are distinct and reject mutated identities" {
    for ([_]precision.Prefill{ .scalar_f32, .fp16_single, .fp16_compensated }) |rung| {
        const id = prefillVariant(rung);
        try std.testing.expectEqual(rung, prefillFromVariant(id).?);
        for (0..32) |bit| {
            try std.testing.expect(prefillFromVariant(id ^ (@as(u32, 1) << @intCast(bit))) == null);
        }
        for (kernels.MatvecVariant.candidates) |candidate| {
            for (0..256) |act_div| {
                var v = candidate;
                v.act_div = @intCast(act_div);
                if (rung != .scalar_f32) try std.testing.expect(id != v.id());
            }
        }
    }
}

test "native prefill rejects unsupported capabilities and variant mutations" {
    const native: LinAlgCaps = .{ .shader_model = 0x6a, .tier = 0x10, .wave_flags = 1, .experimental = true };
    var unsupported = native;
    unsupported.experimental = false;
    for (prefill_kernels) |k| {
        try validatePrefillVariant(k, 0, unsupported);
        for ([_]precision.Prefill{ .fp16_single, .fp16_compensated }) |rung| {
            const variant = prefillVariant(rung);
            try validatePrefillVariant(k, variant, native);
            try std.testing.expectError(error.UnsupportedPrefill, validatePrefillVariant(k, variant, unsupported));
            try std.testing.expectError(error.BadCommand, validatePrefillVariant(.rmsnorm, variant, native));
            for (0..32) |bit| {
                try std.testing.expectError(error.BadCommand, validatePrefillVariant(k, variant ^ (@as(u32, 1) << @intCast(bit)), native));
            }
        }
    }
    // Decode variants retain their independent command-stream namespace.
    for (kernels.MatvecVariant.candidates) |v| try validatePrefillVariant(.matvec_q8_0, v.id(), unsupported);
}

test "prefill cache source commits every rung and changes with each source" {
    for (prefill_kernels) |k| {
        const cached = cacheSource(k);
        var offset: usize = 0;
        for ([_]precision.Prefill{ .scalar_f32, .fp16_single, .fp16_compensated }) |rung| {
            const code = prefillSource(k, rung).?;
            try std.testing.expectEqualSlices(u8, code, cached[offset..][0..code.len]);
            const mutated = try std.testing.allocator.dupe(u8, cached);
            defer std.testing.allocator.free(mutated);
            mutated[offset + code.len - 1] ^= 1;
            try std.testing.expect(std.hash.Wyhash.hash(0, cached) != std.hash.Wyhash.hash(0, mutated));
            offset += code.len;
        }
        try std.testing.expectEqual(cached.len, offset);
    }
}

test "single-product sources round scaled weights without correction products" {
    for ([_]Kernel{ .matmul_q8_0_float, .gyro_matmul }) |k| {
        const single = prefillSource(k, .fp16_single).?;
        const compensated = prefillSource(k, .fp16_compensated).?;
        try std.testing.expect(!std.mem.eql(u8, single, compensated));
        try std.testing.expect(std.mem.indexOf(u8, single, "Matrix<ComponentType::F32") != null);
        try std.testing.expect(std.mem.indexOf(u8, single, "= (half)v;") != null);
        try std.testing.expect(std.mem.indexOf(u8, single, "(half)(scale *") != null);
        try std.testing.expect(std.mem.indexOf(u8, single, "_lo") == null);
        const mma = "MultiplyAccumulate(";
        const first = std.mem.indexOf(u8, single, mma).?;
        try std.testing.expect(std.mem.indexOfPos(u8, single, first + mma.len, mma) == null);
    }
    const dense = prefillSource(.matmul_q8_0_float, .fp16_single).?;
    try std.testing.expect(std.mem.indexOf(u8, dense, "SK = TK + 8u") != null);
    try std.testing.expect(std.mem.indexOf(u8, dense, "wg.z * param(7) + t * out_stride + col0 + j") != null);
    const expert = prefillSource(.gyro_matmul, .fp16_single).?;
    try std.testing.expect(std.mem.indexOf(u8, expert, "#define ROWS 64") != null);
    try std.testing.expect(std.mem.indexOf(u8, expert, "if ((wg.x & 1u) != 0u) return;") != null);
    try std.testing.expect(std.mem.indexOf(u8, expert, "slot * param(5) + wg.z * param(6) + j") != null);
}

test "half-weight prefill specializes real single and compensated products" {
    for ([_]Kernel{ .matmul_bf16, .matmul_f16 }) |k| {
        const single = prefillSource(k, .fp16_single).?;
        const compensated = prefillSource(k, .fp16_compensated).?;
        try std.testing.expect(std.mem.indexOf(u8, single, "#define SINGLE_PRODUCT 1\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, compensated, "#define SINGLE_PRODUCT 0\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, single, "(half)half_lo(word)") != null);
        try std.testing.expect(std.mem.indexOf(u8, single, "(half)half_hi(word)") != null);
        try std.testing.expect(std.mem.indexOf(u8, single, "#if !SINGLE_PRODUCT\n            TileA a_lo") != null);
        try std.testing.expect(std.mem.indexOf(u8, single, "t * out_stride + col0 + j") != null);
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
    if (std.mem.eql(u8, name, "rmsnorm") or std.mem.eql(u8, name, "quantize_q8")) {
        return @embedFile("kernels/norm.hlsli") ++ @embedFile("kernels/" ++ name ++ ".hlsl");
    }
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
