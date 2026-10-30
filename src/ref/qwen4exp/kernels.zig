//! CPU kernels for the Qwen4Exp reference. Weights stay in their GGUF
//! encoding and are decoded a row at a time (the routed experts alone would
//! take 112 GiB as f16); activations are f32. Lifted from gyro-dx.

const std = @import("std");
const gguf = @import("../../gguf/gguf.zig");
const gyro = @import("../gyro/gyro.zig");
const Pool = @import("../pool.zig").Pool;

/// Widest row a weight has here (the hyper-connection streams: 4 × 2,560).
pub const max_cols = 16384;

/// `rows` outputs of `cols` inputs each, in the stored encoding.
pub const Matrix = struct {
    type: gguf.TensorType,
    cols: usize,
    rows: usize,
    row_bytes: usize,
    data: []const u8,

    pub fn row(m: Matrix, r: usize) []const u8 {
        return m.data[r * m.row_bytes ..][0..m.row_bytes];
    }

    /// Rows [start, start + count).
    pub fn slice(m: Matrix, start: usize, count: usize) Matrix {
        var s = m;
        s.rows = count;
        s.data = m.data[start * m.row_bytes ..][0 .. count * m.row_bytes];
        return s;
    }
};

/// Whether the reference can multiply weights of type `t`.
pub fn supported(t: gguf.TensorType) bool {
    return switch (t) {
        .f32, .f16, .bf16, .q8_0 => true,
        else => gyro.info(t) != null,
    };
}

const V8 = @Vector(8, f32);

fn f16At(row: []const u8, i: usize) f32 {
    return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, row[i * 2 ..][0..2], .little))));
}

fn bf16At(row: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, row[i * 2 ..][0..2], .little)) << 16);
}

/// Decodes one row into f32, in the stored basis.
pub fn decodeRow(t: gguf.TensorType, row: []const u8, out: []f32) void {
    switch (t) {
        .f32 => for (out, 0..) |*o, i| {
            o.* = @bitCast(row[i * 4 ..][0..4].*);
        },
        .f16 => for (out, 0..) |*o, i| {
            o.* = f16At(row, i);
        },
        .bf16 => for (out, 0..) |*o, i| {
            o.* = bf16At(row, i);
        },
        .q8_0 => for (0..out.len / 32) |b| {
            const block = row[b * 34 ..][0..34];
            const d = f16At(block, 0);
            for (0..32) |j| out[b * 32 + j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(block[2 + j]))));
        },
        else => {
            const in = gyro.info(t).?;
            for (0..out.len / gyro.block_weights) |b| {
                gyro.decodeBlock(t, row[b * in.block_bytes ..][0..in.block_bytes], out[b * gyro.block_weights ..][0..gyro.block_weights]);
            }
        },
    }
}

pub fn dotDense(a: []const f32, b: []const f32) f32 {
    @setFloatMode(.optimized);
    var acc: V8 = @splat(0);
    var i: usize = 0;
    while (i + 8 <= a.len) : (i += 8) acc += @as(V8, a[i..][0..8].*) * @as(V8, b[i..][0..8].*);
    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

/// One stored row · x.
pub fn dotRow(t: gguf.TensorType, row: []const u8, x: []const f32) f32 {
    @setFloatMode(.optimized);
    switch (t) {
        .q8_0 => {
            var total: f32 = 0;
            for (0..x.len / 32) |b| {
                const block = row[b * 34 ..][0..34];
                const q: @Vector(32, i8) = @bitCast(block[2..34].*);
                const w: @Vector(32, f32) = @floatFromInt(q);
                total += f16At(block, 0) * @reduce(.Add, w * @as(@Vector(32, f32), x[b * 32 ..][0..32].*));
            }
            return total;
        },
        else => {
            var buf: [max_cols]f32 align(64) = undefined;
            decodeRow(t, row, buf[0..x.len]);
            return dotDense(buf[0..x.len], x);
        },
    }
}

const MatmulCtx = struct { m: Matrix, x: []const f32, y: []f32, n: usize };

fn matmulRange(c: *const MatmulCtx, start: usize, end: usize) void {
    if (c.n == 1) {
        for (start..end) |r| c.y[r] = dotRow(c.m.type, c.m.row(r), c.x);
        return;
    }
    // Several rows of x: decode each weight row once.
    var buf: [max_cols]f32 align(64) = undefined;
    const w = buf[0..c.m.cols];
    for (start..end) |r| {
        decodeRow(c.m.type, c.m.row(r), w);
        for (0..c.n) |t| c.y[t * c.m.rows + r] = dotDense(w, c.x[t * c.m.cols ..][0..c.m.cols]);
    }
}

/// y[t][r] = Σ_i W[r][i] x[t][i] for `n` rows of x (n × cols) into y (n × rows).
pub fn matmul(pool: *Pool, m: Matrix, x: []const f32, y: []f32, n: usize) void {
    std.debug.assert(x.len >= n * m.cols and y.len >= n * m.rows and m.cols <= max_cols);
    const ctx: MatmulCtx = .{ .m = m, .x = x, .y = y, .n = n };
    pool.parallelFor(m.rows, 1, 8, &ctx, matmulRange);
}

pub fn rmsNorm(out: []f32, x: []const f32, weight: ?[]const f32, eps: f32) void {
    @setFloatMode(.optimized);
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const scale = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    if (weight) |w| {
        for (out, x, w) |*o, v, g| o.* = v * scale * g;
    } else {
        for (out, x) |*o, v| o.* = v * scale;
    }
}

/// llama.cpp's build_gdn_l2_norm: rms_norm(x, eps / n) / sqrt(n).
pub fn l2Norm(x: []f32, eps: f32) void {
    @setFloatMode(.optimized);
    const n: f32 = @floatFromInt(x.len);
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const scale = (1.0 / @sqrt(ss / n + eps / n)) / @sqrt(n);
    for (x) |*v| v.* *= scale;
}

pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

pub inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// ggml_softplus: log(1 + exp(x)), linear above 20.
pub inline fn softplus(x: f32) f32 {
    return if (x > 20.0) x else @log(1.0 + @exp(x));
}

pub fn softmaxInPlace(x: []f32) void {
    var m: f32 = -std.math.inf(f32);
    for (x) |v| m = @max(m, v);
    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - m);
        sum += v.*;
    }
    for (x) |*v| v.* /= sum;
}

/// NeoX rotary embedding of a head's first `n_rot` values: pairs (i, i + n_rot/2).
pub fn ropeNeox(head: []f32, n_rot: usize, pos: f32, freq_base: f32) void {
    const half = n_rot / 2;
    const theta_scale = std.math.pow(f32, freq_base, -2.0 / @as(f32, @floatFromInt(n_rot)));
    var theta = pos;
    for (0..half) |i| {
        const c = @cos(theta);
        const s = @sin(theta);
        const a = head[i];
        const b = head[i + half];
        head[i] = a * c - b * s;
        head[i + half] = a * s + b * c;
        theta *= theta_scale;
    }
}

test "dotRow agrees with decodeRow for every type the model uses" {
    var prng: std.Random.DefaultPrng = .init(7);
    const r = prng.random();
    var x: [256]f32 = undefined;
    for (&x) |*v| v.* = r.float(f32) - 0.5;
    var row: [256 * 4]u8 = undefined;
    for ([_]gguf.TensorType{ .f32, .bf16, .q8_0, .tq2_t, .tqp6 }) |t| {
        const l = t.layout().?;
        const bytes = row[0 .. 256 / l.block_elems * l.block_bytes];
        r.bytes(bytes);
        // Finite scales and values.
        if (t == .f32) {
            for (0..256) |i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(r.float(f32) - 0.5), .little);
        } else if (t == .bf16) {
            for (0..256) |i| std.mem.writeInt(u16, bytes[i * 2 ..][0..2], @truncate(@as(u32, @bitCast(r.float(f32) - 0.5)) >> 16), .little);
        } else {
            var b: usize = 0;
            while (b < bytes.len) : (b += l.block_bytes) std.mem.writeInt(u16, bytes[b..][0..2], @bitCast(@as(f16, 0.03125)), .little);
        }
        var dense: [256]f32 = undefined;
        decodeRow(t, bytes, &dense);
        var want: f32 = 0;
        for (dense, x) |w, v| want += w * v;
        try std.testing.expectApproxEqAbs(want, dotRow(t, bytes, &x), 1e-4);
    }
}
