//! Matrix multiply for the CPU reference, the one tuned kernel.
//!
//! Weights are repacked once at load into panels of `panel` output rows,
//! stored k-major: panel p holds w[p·panel + c][i] at `(i · panel + c)`.
//! The microkernel is an outer product: for each i it broadcasts one
//! activation per token and multiplies it with the panel's 32-wide weight
//! row, accumulating `block_rows` tokens × 32 outputs in 16 AVX-512
//! registers. There are no horizontal reductions. A thread walks its panels;
//! each 128 KiB panel stays in L2 while every token block streams past.
//!
//! Accumulation is in f32, as in the rest of the reference; only the
//! summation order differs from a naive dot product.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Pool = @import("pool.zig").Pool;
const gguf = @import("../gguf/gguf.zig");
const tensor = @import("tensor.zig");

const vec_len = 16;
const V = @Vector(vec_len, f32);

/// Output rows per packed panel: two AVX-512 vectors.
pub const panel = 2 * vec_len;
/// Tokens per microkernel call. block_rows × 2 accumulators = 16 registers.
const block_rows = 8;

const alignment: std.mem.Alignment = .@"64";

/// An n×k weight matrix (row j = the weights of output j) in packed panels.
/// Columns past `n` in the last panel are zero.
pub const Matrix = struct {
    data: []align(64) f32,
    n: usize,
    k: usize,

    pub fn panels(m: Matrix) usize {
        return std.math.divCeil(usize, m.n, panel) catch unreachable;
    }

    pub const InitError = error{UnsupportedTensorType} || Allocator.Error;

    /// Dequantizes an n×k GGML tensor of type `t` (row j contiguous in
    /// `bytes`) straight into packed panels, one panel per pool task.
    pub fn init(gpa: Allocator, pool: *Pool, t: gguf.TensorType, bytes: []const u8, n: usize, k: usize) InitError!Matrix {
        const layout = t.layout() orelse return error.UnsupportedTensorType;
        if (!tensor.supported(t)) return error.UnsupportedTensorType;
        std.debug.assert(k % layout.block_elems == 0);
        std.debug.assert(bytes.len == n * k / layout.block_elems * layout.block_bytes);

        const n_panels = std.math.divCeil(usize, n, panel) catch unreachable;
        const data = try gpa.alignedAlloc(f32, alignment, n_panels * panel * k);
        const m: Matrix = .{ .data = data, .n = n, .k = k };
        const Ctx = struct {
            m: Matrix,
            t: gguf.TensorType,
            layout: gguf.TensorType.Layout,
            bytes: []const u8,

            /// Row chunks of `chunk` elements: big enough to amortize the
            /// dequantizer call, small enough for the stack, and a multiple
            /// of every block size.
            const chunk = 256;

            fn run(c: *const @This(), first: usize, last: usize) void {
                const kk = c.m.k;
                const row_bytes = kk / c.layout.block_elems * c.layout.block_bytes;
                var buf: [chunk]f32 = undefined;
                for (first..last) |p| {
                    const dst = c.m.data[p * panel * kk ..][0 .. panel * kk];
                    for (0..panel) |col| {
                        const j = p * panel + col;
                        if (j >= c.m.n) {
                            for (0..kk) |i| dst[i * panel + col] = 0;
                            continue;
                        }
                        const src = c.bytes[j * row_bytes ..][0..row_bytes];
                        var start: usize = 0;
                        while (start < kk) : (start += chunk) {
                            const len = @min(chunk, kk - start);
                            const b0 = start / c.layout.block_elems * c.layout.block_bytes;
                            tensor.dequantize(c.t, src[b0..], buf[0..len]) catch unreachable;
                            for (buf[0..len], start..) |v, i| dst[i * panel + col] = v;
                        }
                    }
                }
            }
        };
        std.debug.assert(Ctx.chunk % layout.block_elems == 0);
        const ctx: Ctx = .{ .m = m, .t = t, .layout = layout, .bytes = bytes };
        pool.parallelFor(n_panels, 1, 1, &ctx, Ctx.run);
        return m;
    }

    pub fn deinit(m: *Matrix, gpa: Allocator) void {
        gpa.free(m.data);
        m.* = undefined;
    }

    /// Copies row j (the k weights of output j) into `out`.
    pub fn row(m: Matrix, j: usize, out: []f32) void {
        std.debug.assert(j < m.n and out.len == m.k);
        const base = m.data[(j / panel) * panel * m.k + j % panel ..];
        for (out, 0..) |*o, i| o.* = base[i * panel];
    }
};

/// out[t][j] = Σ_i x[t][i] · w[j][i] for t < rows: x is rows×k, out is rows×n.
/// Splits the panels across the pool.
pub fn matmul(pool: *Pool, out: []f32, x: []const f32, w: *const Matrix, rows: usize) void {
    std.debug.assert(x.len == rows * w.k and out.len == rows * w.n);
    const Ctx = struct {
        out: []f32,
        x: []const f32,
        w: *const Matrix,
        rows: usize,

        /// Panels [first, last). Each 128 KiB panel stays in L2 while every
        /// token block streams past it.
        fn run(c: *const @This(), first: usize, last: usize) void {
            for (first..last) |p| {
                var t: usize = 0;
                while (t < c.rows) : (t += block_rows) {
                    switch (@min(block_rows, c.rows - t)) {
                        inline 1...block_rows => |r| microkernel(r, c, p, t),
                        else => unreachable,
                    }
                }
            }
        }
    };
    const ctx: Ctx = .{ .out = out, .x = x, .w = w, .rows = rows };
    pool.parallelFor(w.panels(), 1, 1, &ctx, Ctx.run);
}

/// out[t0 + a][p·panel + c] for a < R, c < panel (clipped to n).
inline fn microkernel(comptime R: usize, c: anytype, p: usize, t0: usize) void {
    const k = c.w.k;
    // Many-item pointers: bounds are established above, and the hot loop
    // must stay free of per-access checks in safe builds.
    // Panels start at multiples of panel·k floats (128·k bytes) from a
    // 64-byte-aligned base.
    const wp: [*]align(64) const f32 = @alignCast(c.w.data[p * panel * k ..].ptr);
    var xp: [R][*]const f32 = undefined;
    inline for (0..R) |a| xp[a] = c.x[(t0 + a) * k ..].ptr;

    var acc: [R][2]V = @splat(@splat(@splat(0)));
    for (0..k) |i| {
        const w0: V = wp[i * panel ..][0..vec_len].*;
        const w1: V = wp[i * panel + vec_len ..][0..vec_len].*;
        inline for (0..R) |a| {
            const xb: V = @splat(xp[a][i]);
            acc[a][0] += xb * w0;
            acc[a][1] += xb * w1;
        }
    }

    const n = c.w.n;
    const j0 = p * panel;
    const cols = @min(panel, n - j0);
    inline for (0..R) |a| {
        const dst = c.out[(t0 + a) * n + j0 ..][0..cols];
        if (cols == panel) {
            dst[0..vec_len].* = acc[a][0];
            dst[vec_len..panel].* = acc[a][1];
        } else {
            const lanes: [panel]f32 = @bitCast(acc[a]);
            @memcpy(dst, lanes[0..cols]);
        }
    }
}

test "matmul matches the naive product for every remainder and pool size" {
    const gpa = std.testing.allocator;
    const n = 70; // two full panels and a partial one
    const k = 37;
    const max_rows = 2 * block_rows + 1;
    const x = try gpa.alloc(f32, max_rows * k);
    defer gpa.free(x);
    const w = try gpa.alloc(f32, n * k);
    defer gpa.free(w);
    var prng: std.Random.DefaultPrng = .init(1);
    const r = prng.random();
    for (x) |*v| v.* = r.float(f32) - 0.5;
    for (w) |*v| v.* = r.float(f32) - 0.5;
    const out = try gpa.alloc(f32, max_rows * n);
    defer gpa.free(out);

    for ([_]usize{ 1, 2, 7 }) |threads| {
        const pool = try Pool.init(gpa, std.testing.io, threads);
        defer pool.deinit();
        var m = try Matrix.init(gpa, pool, .f32, std.mem.sliceAsBytes(w), n, k);
        defer m.deinit(gpa);

        var got_row: [k]f32 = undefined;
        for (0..n) |j| {
            m.row(j, &got_row);
            try std.testing.expectEqualSlices(f32, w[j * k ..][0..k], &got_row);
        }

        // Every row remainder of the token block, including more than one block.
        for (1..max_rows + 1) |rows| {
            matmul(pool, out[0 .. rows * n], x[0 .. rows * k], &m, rows);
            for (0..rows) |t| for (0..n) |j| {
                var want: f32 = 0;
                for (0..k) |i| want += x[t * k + i] * w[j * k + i];
                try std.testing.expectApproxEqAbs(want, out[t * n + j], 1e-5);
            };
        }
    }
}
