//! The Gyro trellis tensor formats (AgentionAI SPEC.md 1.1, upstream
//! revision 587a5269848eb07996c9c1433f9ad638604eb4d8) and their block
//! Hadamard rotation. Lifted from gyro-dx, whose decoder matched the
//! upstream C reference bit for bit; the codebook and formulas derive from
//! the Apache-2.0 reference, see third_party/agention-gyro/NOTICE.
//!
//! A block is 128 weights: a binary16 scale, then a circular bitstream of
//! steps. Each step's 16-bit state (a window of the stream) keeps its low 15
//! bits, hashed by a fixed multiply into one or two indices of a 2,048-pair
//! codebook; a weight is scale × codebook value, one f32 product.

const std = @import("std");
const gguf = @import("../../gguf/gguf.zig");

pub const block_weights = 128;
/// agention.trellis.state_bits, the only value the format defines.
pub const state_bits = 15;

/// 2,048 pairs of binary16 values, little-endian (SHA-256 ee481f15…e588).
const codebook_bytes = @embedFile("codebook.f16");

/// The codebook as f32, 4,096 values.
pub const codebook: [4096]f32 = blk: {
    @setEvalBranchQuota(100000);
    var table: [4096]f32 = undefined;
    for (&table, 0..) |*v, i| {
        const half = @as(u16, codebook_bytes[i * 2]) | (@as(u16, codebook_bytes[i * 2 + 1]) << 8);
        v.* = @floatCast(@as(f16, @bitCast(half)));
    }
    break :blk table;
};

pub const Info = struct {
    /// Bits each step consumes.
    k: u8,
    /// 32 steps of two pairs (TQ2_T, TQK*) or 64 of one (TQP*).
    steps: u8,
    block_bytes: u16,
};

/// The format of tensor type `t`, or null for a non-Gyro type.
pub fn info(t: gguf.TensorType) ?Info {
    return switch (t) {
        .tq2_t => .{ .k = 8, .steps = 32, .block_bytes = 34 },
        .tqk6 => .{ .k = 6, .steps = 32, .block_bytes = 26 },
        .tqk7 => .{ .k = 7, .steps = 32, .block_bytes = 30 },
        .tqp5 => .{ .k = 5, .steps = 64, .block_bytes = 42 },
        .tqp6 => .{ .k = 6, .steps = 64, .block_bytes = 50 },
        .tqp7 => .{ .k = 7, .steps = 64, .block_bytes = 58 },
        .tqk5 => .{ .k = 5, .steps = 32, .block_bytes = 22 },
        .tqk4 => .{ .k = 4, .steps = 32, .block_bytes = 18 },
        else => null,
    };
}

/// The 16-bit state of `step`: for TQ2_T the byte before the step's and the
/// step's own; otherwise the 16 stream bits from (steps − 1 − step)·k,
/// wrapping around the stream.
fn state(t: gguf.TensorType, in: Info, stream: []const u8, step: usize) u32 {
    if (t == .tq2_t) return (@as(u32, stream[(step + 31) % 32]) << 8) | stream[step];
    const offset: usize = (@as(usize, in.steps) - 1 - step) * in.k;
    const byte = offset >> 3;
    const word = @as(u32, stream[byte]) |
        (@as(u32, stream[(byte + 1) % stream.len]) << 8) |
        (@as(u32, stream[(byte + 2) % stream.len]) << 16);
    return (word >> @as(u5, @intCast(offset & 7))) & 0xffff;
}

/// Decodes one block of type `t` into 128 weights, in the stored basis.
pub fn decodeBlock(t: gguf.TensorType, block: []const u8, out: *[block_weights]f32) void {
    const in = info(t).?;
    const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
    const stream = block[2..in.block_bytes];
    const per_step: usize = if (in.steps == 64) 2 else 4;
    for (0..in.steps) |step| {
        const h = (state(t, in, stream, step) & 0x7fff) *% @as(u32, 0x9e3779b1);
        const a = (h >> 21) * 2;
        out[step * per_step + 0] = scale * codebook[a];
        out[step * per_step + 1] = scale * codebook[a + 1];
        if (per_step == 4) {
            const b = ((h >> 10) & 2047) * 2;
            out[step * per_step + 2] = scale * codebook[b];
            out[step * per_step + 3] = scale * codebook[b + 1];
        }
    }
}

/// T(x) = H_b (signs · x) along a row, in blocks of `block` (a power of
/// two): the basis rotated tensors are stored in, for activations.
pub fn rotate(x: []f32, block: usize, signs: []const f32) void {
    @setFloatMode(.strict);
    for (x, signs) |*v, s| v.* *= s;
    hadamard(x, block);
}

/// T⁻¹(x) = signs · H_b(x): a decoded row back in the plain basis.
pub fn unrotate(x: []f32, block: usize, signs: []const f32) void {
    @setFloatMode(.strict);
    hadamard(x, block);
    for (x, signs) |*v, s| v.* *= s;
}

/// The normalized Sylvester-Walsh-Hadamard transform of each block.
fn hadamard(x: []f32, block: usize) void {
    @setFloatMode(.strict);
    std.debug.assert(std.math.isPowerOfTwo(block) and x.len % block == 0);
    const norm: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(block)));
    var start: usize = 0;
    while (start < x.len) : (start += block) {
        const b = x[start..][0..block];
        var width: usize = 1;
        while (width < block) : (width *= 2) {
            var s: usize = 0;
            while (s < block) : (s += 2 * width) {
                for (0..width) |j| {
                    const p = b[s + j];
                    const q = b[s + j + width];
                    b[s + j] = p + q;
                    b[s + j + width] = p - q;
                }
            }
        }
        for (b) |*v| v.* *= norm;
    }
}

test "codebook is the normative table" {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(codebook_bytes, &digest, .{});
    var hex: [64]u8 = undefined;
    _ = try std.fmt.bufPrint(&hex, "{x}", .{&digest});
    try std.testing.expectEqualStrings("ee481f151d8947ce8b08817fdf89d0a56d647b6435ceb3df0f079ea748e0e588", &hex);
}

// oracle.bin: outputs of the upstream C reference (tq_ref.c), generated by
// gyro-dx's tests/gyro_vectors_generate.py (SHA-256 8f739c3e…818c): 512
// blocks over all eight types, scales covering both signs, zero,
// subnormals and the largest finite value, then rotations at widths 640
// and 2,560.
test "every block and rotation matches the C reference, bit for bit" {
    const fixture = @embedFile("oracle.bin");
    var pos: usize = 0;
    const Cursor = struct {
        fn take(f: []const u8, p: *usize, n: usize) []const u8 {
            defer p.* += n;
            return f[p.*..][0..n];
        }
        fn word(f: []const u8, p: *usize) u32 {
            return std.mem.readInt(u32, take(f, p, 4)[0..4], .little);
        }
    };
    try std.testing.expectEqualStrings("GYROV1\x00\x00", Cursor.take(fixture, &pos, 8));
    const cases = Cursor.word(fixture, &pos);
    try std.testing.expectEqual(@as(u32, 512), cases);
    for (0..cases) |_| {
        const t: gguf.TensorType = @fromBackingInt(@intCast(Cursor.word(fixture, &pos)));
        const size = Cursor.word(fixture, &pos);
        const encoded = Cursor.take(fixture, &pos, size);
        try std.testing.expectEqual(@as(u32, info(t).?.block_bytes), size);
        var decoded: [block_weights]f32 = undefined;
        decodeBlock(t, encoded, &decoded);
        for (decoded) |v| try std.testing.expectEqual(Cursor.word(fixture, &pos), @as(u32, @bitCast(v)));
    }
    const rotations = Cursor.word(fixture, &pos);
    try std.testing.expectEqual(@as(u32, 2), rotations);
    for (0..rotations) |_| {
        const width = Cursor.word(fixture, &pos);
        const block = Cursor.word(fixture, &pos);
        const values = try std.testing.allocator.alloc(f32, width);
        defer std.testing.allocator.free(values);
        const signs = try std.testing.allocator.alloc(f32, width);
        defer std.testing.allocator.free(signs);
        for (values) |*v| v.* = @bitCast(Cursor.word(fixture, &pos));
        for (signs, Cursor.take(fixture, &pos, width)) |*s, byte| s.* = @floatFromInt(@as(i8, @bitCast(byte)));
        rotate(values, block, signs);
        for (values) |v| try std.testing.expectEqual(Cursor.word(fixture, &pos), @as(u32, @bitCast(v)));
        unrotate(values, block, signs);
        for (values) |v| try std.testing.expectEqual(Cursor.word(fixture, &pos), @as(u32, @bitCast(v)));
    }
    try std.testing.expectEqual(fixture.len, pos);
}
