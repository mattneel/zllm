//! Loads GGUF tensors for the reference models: vectors as f32 slices,
//! matrices packed for `matmul`, all owned by the model's arena.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const tensor = @import("tensor.zig");
const Matrix = @import("matmul.zig").Matrix;
const Pool = @import("pool.zig").Pool;

pub const Loader = struct {
    /// Owns everything the model keeps.
    arena: Allocator,
    pool: *Pool,
    file: gguf.File,
    bytes: []const u8,

    pub fn global(l: *Loader, name: []const u8, shape: []const u64) tensor.Error![]f32 {
        return tensor.load(l.arena, l.file, l.bytes, name, shape);
    }

    pub fn get(l: *Loader, comptime fmt: []const u8, layer: usize, shape: []const u64) tensor.Error![]f32 {
        var buf: [96]u8 = undefined;
        return l.global(std.fmt.bufPrint(&buf, fmt, .{layer}) catch unreachable, shape);
    }

    /// A GGML {k, n} weight as a packed n×k `Matrix`.
    pub fn globalMatrix(l: *Loader, name: []const u8, shape: *const [2]u64) tensor.Error!Matrix {
        const info = l.file.tensor(name) orelse return error.MissingTensor;
        if (!std.mem.eql(u64, info.shape(), shape)) return error.BadShape;
        const n: usize = @intCast(shape[1]);
        const k: usize = @intCast(shape[0]);
        return Matrix.init(l.arena, l.pool, info.type, l.file.tensorBytes(l.bytes, info), n, k);
    }

    pub fn matrix(l: *Loader, comptime fmt: []const u8, layer: usize, shape: *const [2]u64) tensor.Error!Matrix {
        var buf: [96]u8 = undefined;
        return l.globalMatrix(std.fmt.bufPrint(&buf, fmt, .{layer}) catch unreachable, shape);
    }

    pub fn has(l: *Loader, comptime fmt: []const u8, layer: usize) bool {
        var buf: [96]u8 = undefined;
        return l.file.tensor(std.fmt.bufPrint(&buf, fmt, .{layer}) catch unreachable) != null;
    }
};
