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
//!
//! Quantized weights are not packed at load: in f32 a 27B model would need
//! 108 GB. They stay in the file mapping, and each thread decodes a panel
//! into its own buffer just before using it.

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

/// An n×k weight matrix (row j = the weights of output j): packed panels
/// for f32, f16 and bf16 (columns past `n` in the last panel are zero), or
/// for quantized types the raw rows, decoded panel by panel as they are used.
pub const Matrix = struct {
    data: []align(64) f32,
    n: usize,
    k: usize,
    /// The GGUF type the weights came in.
    type: gguf.TensorType = .f32,
    /// Quantized rows (row j contiguous) when not packed.
    quant: ?struct { type: gguf.TensorType, bytes: []const u8 } = null,

    pub fn panels(m: Matrix) usize {
        return std.math.divCeil(usize, m.n, panel) catch unreachable;
    }

    pub const InitError = error{UnsupportedTensorType} || Allocator.Error;

    /// An n×k GGML tensor of type `t` (row j contiguous in `bytes`):
    /// dequantized straight into packed panels, one panel per pool task, or
    /// for a quantized type kept as `bytes`, which must outlive the Matrix.
    pub fn init(gpa: Allocator, pool: *Pool, t: gguf.TensorType, bytes: []const u8, n: usize, k: usize) InitError!Matrix {
        const layout = t.layout() orelse return error.UnsupportedTensorType;
        if (!tensor.supported(t)) return error.UnsupportedTensorType;
        std.debug.assert(k % layout.block_elems == 0);
        std.debug.assert(bytes.len == n * k / layout.block_elems * layout.block_bytes);
        switch (t) {
            .f32, .f16, .bf16 => {},
            else => return .{ .data = &.{}, .n = n, .k = k, .type = t, .quant = .{ .type = t, .bytes = bytes } },
        }

        const n_panels = std.math.divCeil(usize, n, panel) catch unreachable;
        const data = try gpa.alignedAlloc(f32, alignment, n_panels * panel * k);
        const m: Matrix = .{ .data = data, .n = n, .k = k, .type = t };
        const Ctx = struct {
            m: Matrix,
            t: gguf.TensorType,
            bytes: []const u8,

            fn run(c: *const @This(), first: usize, last: usize) void {
                for (first..last) |p| packPanel(c.t, c.bytes, c.m.n, c.m.k, p, c.m.data[p * panel * c.m.k ..][0 .. panel * c.m.k]);
            }
        };
        const ctx: Ctx = .{ .m = m, .t = t, .bytes = bytes };
        pool.parallelFor(n_panels, 1, 1, &ctx, Ctx.run);
        return m;
    }

    /// Panel p of the matrix, packed k-major (column c of the panel at
    /// i · panel + c): from `data`, or decoded into `buf`.
    fn panelData(m: *const Matrix, p: usize, buf: []align(64) f32) []align(64) const f32 {
        const q = m.quant orelse return @alignCast(m.data[p * panel * m.k ..][0 .. panel * m.k]);
        packPanel(q.type, q.bytes, m.n, m.k, p, buf);
        return buf;
    }

    pub fn deinit(m: *Matrix, gpa: Allocator) void {
        gpa.free(m.data);
        m.* = undefined;
    }

    /// Copies row j (the k weights of output j) into `out`.
    pub fn row(m: Matrix, j: usize, out: []f32) void {
        std.debug.assert(j < m.n and out.len == m.k);
        if (m.quant) |q| {
            const l = q.type.layout().?;
            const row_bytes = m.k / l.block_elems * l.block_bytes;
            tensor.dequantize(q.type, q.bytes[j * row_bytes ..][0..row_bytes], out) catch unreachable;
            return;
        }
        const base = m.data[(j / panel) * panel * m.k + j % panel ..];
        for (out, 0..) |*o, i| o.* = base[i * panel];
    }
};

/// Decodes rows [p·panel, p·panel + panel) of an n×k tensor of type `t` into
/// `dst`, packed k-major; rows past n are zero.
fn packPanel(t: gguf.TensorType, bytes: []const u8, n: usize, k: usize, p: usize, dst: []f32) void {
    const layout = t.layout().?;
    const row_bytes = k / layout.block_elems * layout.block_bytes;
    // Row pieces of `chunk` elements: big enough to amortize the decoder
    // call, small enough for the stack, a multiple of every block size.
    const chunk = 256;
    var buf: [chunk]f32 = undefined;
    for (0..panel) |col| {
        const j = p * panel + col;
        if (j >= n) {
            for (0..k) |i| dst[i * panel + col] = 0;
            continue;
        }
        const src = bytes[j * row_bytes ..][0..row_bytes];
        var start: usize = 0;
        while (start < k) : (start += chunk) {
            const len = @min(chunk, k - start);
            const b0 = start / layout.block_elems * layout.block_bytes;
            tensor.dequantize(t, src[b0..], buf[0..len]) catch unreachable;
            for (buf[0..len], start..) |v, i| dst[i * panel + col] = v;
        }
    }
}

/// Each thread's panel buffer for quantized matrices, grown as needed and
/// kept for the life of the thread.
threadlocal var panel_buf: []align(64) f32 = &.{};

fn threadPanel(k: usize) []align(64) f32 {
    if (panel_buf.len < panel * k) {
        if (panel_buf.len > 0) std.heap.page_allocator.free(panel_buf);
        panel_buf = std.heap.page_allocator.alignedAlloc(f32, alignment, panel * k) catch @panic("out of memory for a weight panel");
    }
    return panel_buf[0 .. panel * k];
}

/// How a multiply sees its input.
pub const Activations = enum {
    /// As given, f32.
    f32,
    /// Rounded as llama.cpp's CPU backend rounds them for each weight type's
    /// dot product (`vec_dot_type`): to bf16 for bf16 weights, to Q8_0 blocks
    /// for Q8_0 and to Q8_K blocks for the K-quants. A diagnostic: it shows
    /// how much of a difference from llama.cpp that rounding explains.
    llama_cpp,
};

/// out[t][j] = Σ_i x[t][i] · w[j][i] for t < rows: x is rows×k, out is rows×n.
/// Splits the panels across the pool.
pub fn matmul(pool: *Pool, out: []f32, x_in: []const f32, w: *const Matrix, rows: usize, act: Activations) void {
    std.debug.assert(x_in.len == rows * w.k and out.len == rows * w.n);
    var rounded: []f32 = &.{};
    defer if (rounded.len > 0) std.heap.page_allocator.free(rounded);
    const x = switch (act) {
        .f32 => x_in,
        .llama_cpp => blk: {
            rounded = std.heap.page_allocator.dupe(f32, x_in) catch @panic("out of memory for activations");
            for (0..rows) |r| roundLikeLlama(w.type, rounded[r * w.k ..][0..w.k]);
            break :blk rounded;
        },
    };
    const Ctx = struct {
        out: []f32,
        x: []const f32,
        w: *const Matrix,
        rows: usize,

        /// Panels [first, last). Each 128 KiB panel stays in L2 while every
        /// token block streams past it.
        fn run(c: *const @This(), first: usize, last: usize) void {
            for (first..last) |p| {
                const wp = c.w.panelData(p, if (c.w.quant != null) threadPanel(c.w.k) else &.{});
                var t: usize = 0;
                while (t < c.rows) : (t += block_rows) {
                    switch (@min(block_rows, c.rows - t)) {
                        inline 1...block_rows => |r| microkernel(r, c, wp, p, t),
                        else => unreachable,
                    }
                }
            }
        }
    };
    const ctx: Ctx = .{ .out = out, .x = x, .w = w, .rows = rows };
    pool.parallelFor(w.panels(), 1, 1, &ctx, Ctx.run);
}

/// Rounds one row of activations as ggml's CPU backend quantizes them for a
/// dot product with weights of type `t` (quantize_row_q8_0_ref,
/// quantize_row_q8_K_ref, GGML_FP32_TO_BF16, f16), back to f32; f32
/// weights take f32 activations.
pub fn roundLikeLlama(t: gguf.TensorType, x: []f32) void {
    switch (t) {
        .q8_0 => {
            var b: usize = 0;
            while (b < x.len) : (b += 32) {
                const blk = x[b..][0..32];
                var amax: f32 = 0;
                for (blk) |v| amax = @max(amax, @abs(v));
                const d = amax / 127;
                const id: f32 = if (d != 0) 1 / d else 0;
                const d16: f32 = @as(f16, @floatCast(d));
                for (blk) |*v| v.* = @round(v.* * id) * d16;
            }
        },
        .q3_k, .q4_k, .q5_k, .q6_k => {
            var b: usize = 0;
            while (b < x.len) : (b += 256) {
                const blk = x[b..][0..256];
                var amax: f32 = 0;
                var max: f32 = 0;
                for (blk) |v| if (@abs(v) > amax) {
                    amax = @abs(v);
                    max = v;
                };
                if (amax == 0) {
                    @memset(blk, 0);
                    continue;
                }
                const iscale = -127.0 / max;
                const d = 1 / iscale;
                for (blk) |*v| v.* = @min(127, @round(iscale * v.*)) * d;
            }
        },
        .f16 => for (x) |*v| {
            v.* = @as(f16, @floatCast(v.*));
        },
        .bf16 => for (x) |*v| {
            // Round to nearest even, as GGML_FP32_TO_BF16.
            const bits: u32 = @bitCast(v.*);
            const r = bits +% (0x7fff + ((bits >> 16) & 1));
            v.* = @bitCast(r & 0xffff0000);
        },
        else => {},
    }
}

/// out[t0 + a][p·panel + c] for a < R, c < panel (clipped to n), from the
/// panel's packed weights `panel_w`.
inline fn microkernel(comptime R: usize, c: anytype, panel_w: []align(64) const f32, p: usize, t0: usize) void {
    const k = c.w.k;
    // Many-item pointers: bounds are established above, and the hot loop
    // must stay free of per-access checks in safe builds.
    const wp: [*]align(64) const f32 = panel_w.ptr;
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
            matmul(pool, out[0 .. rows * n], x[0 .. rows * k], &m, rows, .f32);
            for (0..rows) |t| for (0..n) |j| {
                var want: f32 = 0;
                for (0..k) |i| want += x[t * k + i] * w[j * k + i];
                try std.testing.expectApproxEqAbs(want, out[t * n + j], 1e-5);
            };
        }
    }
}

test "quantized matrices decode panels on the fly" {
    const gpa = std.testing.allocator;
    const n = 70; // two full panels and a partial one
    const k = 64;
    const rows = 9;
    var prng: std.Random.DefaultPrng = .init(2);
    const r = prng.random();
    // Q8_0 blocks: f16 scale, then 32 random int8 quants.
    const bytes = try gpa.alloc(u8, n * k / 32 * 34);
    defer gpa.free(bytes);
    r.bytes(bytes);
    for (0..n * k / 32) |b| std.mem.writeInt(u16, bytes[b * 34 ..][0..2], @bitCast(@as(f16, @floatCast(r.float(f32) - 0.5))), .little);
    const w = try gpa.alloc(f32, n * k);
    defer gpa.free(w);
    try tensor.dequantize(.q8_0, bytes, w);
    const x = try gpa.alloc(f32, rows * k);
    defer gpa.free(x);
    for (x) |*v| v.* = r.float(f32) - 0.5;
    const out = try gpa.alloc(f32, rows * n);
    defer gpa.free(out);

    const pool = try Pool.init(gpa, std.testing.io, 3);
    defer pool.deinit();
    var m = try Matrix.init(gpa, pool, .q8_0, bytes, n, k);
    defer m.deinit(gpa);
    try std.testing.expect(m.quant != null);

    var got_row: [k]f32 = undefined;
    m.row(n - 1, &got_row);
    try std.testing.expectEqualSlices(f32, w[(n - 1) * k ..][0..k], &got_row);

    matmul(pool, out, x, &m, rows, .f32);
    for (0..rows) |t| for (0..n) |j| {
        var want: f32 = 0;
        for (0..k) |i| want += x[t * k + i] * w[j * k + i];
        try std.testing.expectApproxEqAbs(want, out[t * n + j], 1e-4);
    };
}
