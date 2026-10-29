//! Dequantization of GGUF tensors to f32 for the CPU reference.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");

pub const Error = error{ UnsupportedTensorType, MissingTensor, BadShape } || Allocator.Error;

/// Whether `dequantize` handles `t`.
pub fn supported(t: gguf.TensorType) bool {
    return switch (t) {
        .f32, .f16, .bf16, .q8_0, .q3_k, .q4_k, .q5_k, .q6_k => true,
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
