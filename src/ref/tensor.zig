//! Dequantization of GGUF tensors to f32 for the CPU reference.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");

pub const Error = error{ UnsupportedTensorType, MissingTensor, BadShape } || Allocator.Error;

/// Whether `dequantize` handles `t`.
pub fn supported(t: gguf.TensorType) bool {
    return switch (t) {
        .f32, .f16, .bf16, .q8_0, .q3_k, .q4_k, .q5_k, .q6_k, .iq4_xs => true,
        else => false,
    };
}

/// Decodes `bytes` of type `t` into `out` (one f32 per element). The
/// quantized formats follow ggml's dequantize_row_* (ggml-quants.c) in
/// layout and in the order of every f32 operation, so results match ggml
/// bit for bit.
pub fn dequantize(t: gguf.TensorType, bytes: []const u8, out: []f32) error{UnsupportedTensorType}!void {
    switch (t) {
        .q8_0 => for (0..out.len / 32) |i| q8_0(bytes[i * 34 ..][0..34], out[i * 32 ..][0..32]),
        .q3_k => for (0..out.len / 256) |i| q3K(bytes[i * 110 ..][0..110], out[i * 256 ..][0..256]),
        .q4_k => for (0..out.len / 256) |i| q4K(bytes[i * 144 ..][0..144], out[i * 256 ..][0..256]),
        .q5_k => for (0..out.len / 256) |i| q5K(bytes[i * 176 ..][0..176], out[i * 256 ..][0..256]),
        .q6_k => for (0..out.len / 256) |i| q6K(bytes[i * 210 ..][0..210], out[i * 256 ..][0..256]),
        .iq4_xs => for (0..out.len / 256) |i| iq4Xs(bytes[i * 136 ..][0..136], out[i * 256 ..][0..256]),
        .f32 => for (out, 0..) |*o, i| {
            o.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
        },
        .f16 => for (out, 0..) |*o, i| {
            const h: f16 = @bitCast(std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little));
            o.* = h;
        },
        // bfloat16 is the high half of an IEEE f32.
        .bf16 => for (out, 0..) |*o, i| {
            const b: u32 = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
            o.* = @bitCast(b << 16);
        },
        else => return error.UnsupportedTensorType,
    }
}

fn half(b: *const [2]u8) f32 {
    return @as(f16, @bitCast(std.mem.readInt(u16, b, .little)));
}

/// block_q8_0: d (f16), qs[32] (i8).
fn q8_0(b: *const [34]u8, y: *[32]f32) void {
    const d = half(b[0..2]);
    for (y, b[2..34]) |*o, q| o.* = @as(f32, @floatFromInt(@as(i8, @bitCast(q)))) * d;
}

/// block_q3_K: hmask[32], qs[64], scales[12] (6-bit, packed), d (f16).
fn q3K(b: *const [110]u8, y: *[256]f32) void {
    const hm = b[0..32];
    const d_all = half(b[108..110]);
    // The 16 scales: low 4 bits from scales[0..8], high 2 bits from [8..12].
    var aux: [4]u32 = undefined;
    for (&aux, 0..) |*a, i| a.* = if (i < 3) std.mem.readInt(u32, b[96 + 4 * i ..][0..4], .little) else 0;
    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;
    const tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const scales: [16]i8 = @bitCast(aux);

    var out: usize = 0;
    var is: usize = 0;
    var m: u8 = 1;
    for (0..2) |n| {
        const q = b[32 + n * 32 ..][0..32];
        var shift: u3 = 0;
        for (0..4) |_| {
            for (0..2) |half_| {
                const dl = d_all * @as(f32, @floatFromInt(@as(i32, scales[is]) - 32));
                is += 1;
                for (0..16) |l| {
                    const lo: i8 = @intCast((q[l + 16 * half_] >> shift) & 3);
                    const v = lo - @as(i8, if (hm[l + 16 * half_] & m != 0) 0 else 4);
                    y[out] = dl * @as(f32, @floatFromInt(v));
                    out += 1;
                }
            }
            shift +%= 2;
            m <<= 1;
        }
    }
}

/// The 6-bit scale and min of sub-block j from a K-quant's 12 scale bytes.
fn scaleMinK4(j: usize, q: *const [12]u8) [2]u8 {
    if (j < 4) return .{ q[j] & 63, q[j + 4] & 63 };
    return .{ (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4) };
}

/// block_q4_K: d, dmin (f16), scales[12], qs[128] (4-bit).
fn q4K(b: *const [144]u8, y: *[256]f32) void {
    const d = half(b[0..2]);
    const min = half(b[2..4]);
    const scales = b[4..16];
    for (0..4) |j| {
        const q = b[16 + 32 * j ..][0..32];
        const s1 = scaleMinK4(2 * j, scales);
        const s2 = scaleMinK4(2 * j + 1, scales);
        const d1 = d * @as(f32, @floatFromInt(s1[0]));
        const m1 = min * @as(f32, @floatFromInt(s1[1]));
        const d2 = d * @as(f32, @floatFromInt(s2[0]));
        const m2 = min * @as(f32, @floatFromInt(s2[1]));
        for (0..32) |l| y[64 * j + l] = d1 * @as(f32, @floatFromInt(q[l] & 0xF)) - m1;
        for (0..32) |l| y[64 * j + 32 + l] = d2 * @as(f32, @floatFromInt(q[l] >> 4)) - m2;
    }
}

/// block_q5_K: d, dmin (f16), scales[12], qh[32] (high bits), qs[128].
fn q5K(b: *const [176]u8, y: *[256]f32) void {
    const d = half(b[0..2]);
    const min = half(b[2..4]);
    const scales = b[4..16];
    const qh = b[16..48];
    for (0..4) |j| {
        const ql = b[48 + 32 * j ..][0..32];
        const bit_lo: u8 = @as(u8, 1) << @intCast(2 * j);
        const bit_hi: u8 = @as(u8, 2) << @intCast(2 * j);
        const s1 = scaleMinK4(2 * j, scales);
        const s2 = scaleMinK4(2 * j + 1, scales);
        const d1 = d * @as(f32, @floatFromInt(s1[0]));
        const m1 = min * @as(f32, @floatFromInt(s1[1]));
        const d2 = d * @as(f32, @floatFromInt(s2[0]));
        const m2 = min * @as(f32, @floatFromInt(s2[1]));
        for (0..32) |l| y[64 * j + l] = d1 * @as(f32, @floatFromInt((ql[l] & 0xF) + @as(u8, if (qh[l] & bit_lo != 0) 16 else 0))) - m1;
        for (0..32) |l| y[64 * j + 32 + l] = d2 * @as(f32, @floatFromInt((ql[l] >> 4) + @as(u8, if (qh[l] & bit_hi != 0) 16 else 0))) - m2;
    }
}

/// block_q6_K: ql[128] (low 4 bits), qh[64] (high 2 bits), scales[16]
/// (i8), d (f16).
fn q6K(b: *const [210]u8, y: *[256]f32) void {
    const d = half(b[208..210]);
    for (0..2) |n| {
        const ql = b[64 * n ..][0..64];
        const qh = b[128 + 32 * n ..][0..32];
        const sc: *const [8]i8 = @ptrCast(b[192 + 8 * n ..][0..8]);
        const out = y[128 * n ..][0..128];
        for (0..32) |l| {
            const is = l / 16;
            const q1: i8 = @as(i8, @intCast((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4))) - 32;
            const q2: i8 = @as(i8, @intCast((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4))) - 32;
            const q3: i8 = @as(i8, @intCast((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4))) - 32;
            const q4: i8 = @as(i8, @intCast((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4))) - 32;
            out[l] = d * @as(f32, @floatFromInt(sc[is])) * @as(f32, @floatFromInt(q1));
            out[l + 32] = d * @as(f32, @floatFromInt(sc[is + 2])) * @as(f32, @floatFromInt(q2));
            out[l + 64] = d * @as(f32, @floatFromInt(sc[is + 4])) * @as(f32, @floatFromInt(q3));
            out[l + 96] = d * @as(f32, @floatFromInt(sc[is + 6])) * @as(f32, @floatFromInt(q4));
        }
    }
}

/// block_iq4_xs: d (f16), scales_h (u16), scales_l[4], qs[128].
/// Each 32-element group has a signed six-bit scale, biased by 32,
/// and nonlinear four-bit quants (low nibbles precede high nibbles).
fn iq4Xs(b: *const [136]u8, y: *[256]f32) void {
    const values = [_]i8{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
    const d = half(b[0..2]);
    const scales_h = std.mem.readInt(u16, b[2..4], .little);
    for (0..8) |group| {
        const low = (b[4 + group / 2] >> @as(u3, @intCast(4 * (group % 2)))) & 15;
        const high = (scales_h >> @as(u4, @intCast(2 * group))) & 3;
        const scale = @as(i32, low | (high << 4)) - 32;
        const dl = d * @as(f32, @floatFromInt(scale));
        const qs = b[8 + 16 * group ..][0..16];
        for (qs, 0..) |q, j| {
            y[32 * group + j] = dl * @as(f32, @floatFromInt(values[q & 15]));
            y[32 * group + 16 + j] = dl * @as(f32, @floatFromInt(values[q >> 4]));
        }
    }
}

/// Loads tensor `name` as f32 after checking its GGML shape. Caller owns the result.
pub fn load(gpa: Allocator, file: gguf.File, file_bytes: []const u8, name: []const u8, shape: []const u64) Error![]f32 {
    const info = file.tensor(name) orelse return error.MissingTensor;
    if (!std.mem.eql(u64, info.shape(), shape)) return error.BadShape;
    const out = try gpa.alloc(f32, @intCast(info.elemCount()));
    errdefer gpa.free(out);
    try dequantize(info.type, file.tensorBytes(file_bytes, info), out);
    return out;
}

test "dequantize f32, f16 and bf16" {
    var out: [2]f32 = undefined;

    const f32_bytes = std.mem.toBytes([2]u32{ @bitCast(@as(f32, 1.5)), @bitCast(@as(f32, -2)) });
    try dequantize(.f32, &f32_bytes, &out);
    try std.testing.expectEqualSlices(f32, &.{ 1.5, -2 }, &out);

    const f16_bytes = std.mem.toBytes([2]u16{ @bitCast(@as(f16, 0.25)), @bitCast(@as(f16, -65504)) });
    try dequantize(.f16, &f16_bytes, &out);
    try std.testing.expectEqualSlices(f32, &.{ 0.25, -65504 }, &out);

    // bf16 0x3FC0 = 1.5, 0xC000 = -2.
    const bf16_bytes = [_]u8{ 0xC0, 0x3F, 0x00, 0xC0 };
    try dequantize(.bf16, &bf16_bytes, &out);
    try std.testing.expectEqualSlices(f32, &.{ 1.5, -2 }, &out);

    try std.testing.expectError(error.UnsupportedTensorType, dequantize(.q2_k, &bf16_bytes, &out));
}

test "IQ4_XS covers every nonlinear code, signed scale and block boundary" {
    try std.testing.expect(supported(.iq4_xs));
    const layout = gguf.TensorType.iq4_xs.layout().?;
    try std.testing.expectEqual(@as(u32, 256), layout.block_elems);
    try std.testing.expectEqual(@as(u32, 136), layout.block_bytes);
    const want_values = [_]f32{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
    var bytes: [2 * 136]u8 = undefined;
    var out: [2 * 256]f32 = undefined;
    const ds = [_]f16{ 1.5, -0.25 };
    for (0..64) |code| {
        @memset(&bytes, 0);
        for (ds, 0..) |d, block| {
            const b = bytes[136 * block ..][0..136];
            std.mem.writeInt(u16, b[0..2], @bitCast(d), .little);
            var scales_h: u16 = 0;
            for (0..8) |group| {
                const s: u8 = @intCast((code + 7 * group) % 64);
                b[4 + group / 2] |= (s & 15) << @as(u3, @intCast(4 * (group % 2)));
                scales_h |= @as(u16, s >> 4) << @as(u4, @intCast(2 * group));
                for (0..16) |j| b[8 + 16 * group + j] = @as(u8, @intCast(j)) | (@as(u8, @intCast(15 - j)) << 4);
            }
            std.mem.writeInt(u16, b[2..4], scales_h, .little);
        }
        try dequantize(.iq4_xs, &bytes, &out);
        for (ds, 0..) |d, block| for (0..8) |group| {
            const scale: i32 = @as(i32, @intCast((code + 7 * group) % 64)) - 32;
            const dl = @as(f32, d) * @as(f32, @floatFromInt(scale));
            for (0..32) |j| {
                const value = want_values[if (j < 16) j else 31 - j];
                const want: f32 = dl * value;
                try std.testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(out[256 * block + 32 * group + j])));
            }
        };
    }
}

test "IQ4_XS individual packed-bit mutations affect only their group or lane" {
    // d = 0.5; all scale codes = 33 (signed +1); all quants = 8 (value +1).
    var base: [136]u8 = @splat(0x88);
    base[0..8].* = .{ 0, 0x38, 0xaa, 0xaa, 0x11, 0x11, 0x11, 0x11 };
    var out: [256]f32 = undefined;
    try dequantize(.iq4_xs, &base, &out);
    for (out) |value| try std.testing.expectEqual(@as(f32, 0.5), value);
    for (2..136) |byte| for (0..8) |bit| {
        var mutated = base;
        mutated[byte] ^= @as(u8, 1) << @as(u3, @intCast(bit));
        try dequantize(.iq4_xs, &mutated, &out);
        const group = if (byte < 4)
            ((byte - 2) * 8 + bit) / 2
        else if (byte < 8)
            (byte - 4) * 2 + bit / 4
        else
            (byte - 8) / 16;
        const lane = (byte -| 8) % 16 + (bit / 4) * 16;
        var changed: usize = 0;
        for (out, 0..) |value, i| {
            const affected = i / 32 == group and (byte < 8 or i % 32 == lane);
            try std.testing.expectEqual(affected, value != 0.5);
            changed += @intFromBool(value != 0.5);
        }
        try std.testing.expectEqual(@as(usize, if (byte < 8) 32 else 1), changed);
    };
}
