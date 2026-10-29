//! Dequantization of GGUF tensors to f32 for the CPU reference.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");

pub const Error = error{ UnsupportedTensorType, MissingTensor, BadShape } || Allocator.Error;

/// Decodes `bytes` of type `t` into `out` (one f32 per element).
pub fn dequantize(t: gguf.TensorType, bytes: []const u8, out: []f32) error{UnsupportedTensorType}!void {
    switch (t) {
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

    try std.testing.expectError(error.UnsupportedTensorType, dequantize(.q4_k, &bf16_bytes, &out));
}
