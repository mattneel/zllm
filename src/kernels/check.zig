//! GPU kernel checks: each kernel on seeded random inputs against a float64
//! CPU computation of the same formula (the sampler against its CPU
//! reference). Shapes are chosen to hit tile and chunk remainders, and
//! stateful kernels run as two steps. Run with `zllm gpu-check` on any Dawn
//! host.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cmd = @import("../gpu/cmd.zig");
const Host = @import("../gpu/host.zig").Host;
const kernels = @import("kernels.zig");
const sampler = @import("../sampler/sampler.zig");
const P = kernels.params;
const gguf = @import("../gguf/gguf.zig");
const ref_tensor = @import("../ref/tensor.zig");
const ref_matmul = @import("../ref/matmul.zig");

pub const Result = struct {
    name: []const u8,
    err: f64,
    tolerance: f64,
    /// Sampler checks: draws made, and how many picked the reference's token.
    trials: u32 = 0,
    exact: u32 = 0,
};

/// Runs every check, reporting one line per check to `out`. Returns whether
/// all passed.
pub fn run(gpa: Allocator, host: *Host, out: *std.Io.Writer) !bool {
    var ctx: Ctx = .{ .gpa = gpa, .host = host, .rec = .{} };
    defer ctx.rec.deinit(gpa);
    ctx.set = try kernels.Set.create(gpa, &ctx.rec);
    try ctx.flush();

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    ctx.arena = arena_state.allocator();
    ctx.rand = prng.random();

    var ok = true;
    inline for (.{
        checkRmsNorm,        checkMatmul,          checkMatmulChunked,     checkMatvec,         checkMatvecNorm,
        checkMatvecSiluPair, checkSiluMul,         checkEmbed,             checkRopeAppendF32,  checkRopeAppendF16,
        checkAttentionF32,   checkAttentionF16,    checkAttnDecode,        checkDeltaPrep,      checkDeltaRule,
        checkSampleGreedy,   checkSampleTopKTopP,  checkSampleMinP,        checkSampleTopP,     checkSamplePlain,
        checkSampleTopK1,    checkSampleLargeTopK, checkSampleTopKSkipped, checkSampleVocab,    Quant(.q8_0).matvec,
        Quant(.q8_0).matmul, Quant(.q8_0).embed,   Quant(.q3_k).matvec,    Quant(.q3_k).matmul, Quant(.q3_k).embed,
        Quant(.q4_k).matvec, Quant(.q4_k).matmul,  Quant(.q4_k).embed,     Quant(.q5_k).matvec, Quant(.q5_k).matmul,
        Quant(.q5_k).embed,  Quant(.q6_k).matvec,  Quant(.q6_k).matmul,    Quant(.q6_k).embed,  checkQuantTiles,
        checkQuantSiluPair,  checkQuantizeQ8K,     checkQuantizeQ80,       checkQuantizeNorm,   checkQuantizeSilu,
    }) |f| {
        const r = try f(&ctx);
        const pass = r.err <= r.tolerance;
        ok = ok and pass;
        const verdict = if (pass) "PASS" else "FAIL";
        if (r.trials > 0) {
            try out.print("{s} {s:<30} max draw deviation {e:.2} (tolerance {e:.0}); {d}/{d} draws exact\n", .{ verdict, r.name, r.err, r.tolerance, r.exact, r.trials });
        } else {
            try out.print("{s} {s:<30} max rel err {e:.2} (tolerance {e:.0})\n", .{ verdict, r.name, r.err, r.tolerance });
        }
        _ = arena_state.reset(.retain_capacity);
    }
    return ok;
}

const Ctx = struct {
    gpa: Allocator,
    arena: Allocator = undefined,
    host: *Host,
    rec: cmd.Recorder,
    set: kernels.Set = undefined,
    rand: std.Random = undefined,
    token: u32 = 0,

    fn launcher(c: *Ctx) kernels.Launcher {
        return .{ .gpa = c.gpa, .arena = c.arena, .rec = &c.rec, .set = &c.set };
    }

    fn flush(c: *Ctx) !void {
        var completions: std.ArrayList(cmd.Completion) = .empty;
        defer completions.deinit(c.gpa);
        c.token += 1;
        try c.rec.submit(c.gpa, c.token);
        c.host.execute(c.rec.batch(), &completions) catch |err| {
            if (err == error.GpuError or err == error.DeviceLost) std.log.err("{s}", .{c.host.d.errorMessage()});
            return err;
        };
        c.rec.clear();
    }

    fn random(c: *Ctx, n: usize, scale: f32) ![]f32 {
        const v = try c.arena.alloc(f32, n);
        for (v) |*x| x.* = (c.rand.float(f32) * 2 - 1) * scale;
        return v;
    }

    /// Random bf16 values packed two per u32, and their f32 values.
    fn randomBf16(c: *Ctx, n: usize, scale: f32) !struct { words: []u32, values: []f32 } {
        std.debug.assert(n % 2 == 0);
        const words = try c.arena.alloc(u32, n / 2);
        const values = try c.arena.alloc(f32, n);
        for (values, 0..) |*v, i| {
            const f = (c.rand.float(f32) * 2 - 1) * scale;
            const bits: u16 = @truncate(@as(u32, @bitCast(f)) >> 16);
            v.* = @bitCast(@as(u32, bits) << 16);
            if (i % 2 == 0) words[i / 2] = bits else words[i / 2] |= @as(u32, bits) << 16;
        }
        return .{ .words = words, .values = values };
    }

    fn upload(c: *Ctx, data: anytype) !kernels.Arg {
        const bytes = std.mem.sliceAsBytes(data);
        const size = std.mem.alignForward(u64, bytes.len, 4);
        const buf = try c.rec.createBuffer(c.gpa, size, .{ .storage = true, .copy_dst = true, .copy_src = true });
        try c.rec.writeBuffer(c.gpa, buf, 0, bytes);
        return .{ .buffer = buf, .size = size };
    }

    fn output(c: *Ctx, n: usize) !kernels.Arg {
        const size = n * 4;
        const buf = try c.rec.createBuffer(c.gpa, size, .{ .storage = true, .copy_src = true, .copy_dst = true });
        return .{ .buffer = buf, .size = size };
    }

    fn uniform(c: *Ctx, value: anytype) !kernels.Arg {
        const bytes = try c.arena.dupe(u8, std.mem.asBytes(&value));
        const buf = try c.rec.createBuffer(c.gpa, bytes.len, .{ .uniform = true, .copy_dst = true });
        try c.rec.writeBuffer(c.gpa, buf, 0, bytes);
        return .{ .buffer = buf, .size = bytes.len };
    }

    fn step(c: *Ctx, m: u32, pos0: u32, u: f32) !kernels.Arg {
        return c.uniform(P.Step{ .m = m, .pos0 = pos0, .u = u });
    }

    /// Reads `arg` back as f32 after running everything recorded so far.
    fn read(c: *Ctx, arg: kernels.Arg) ![]f32 {
        return std.mem.bytesAsSlice(f32, try c.readBytes(arg));
    }

    fn readWords(c: *Ctx, arg: kernels.Arg) ![]u32 {
        return std.mem.bytesAsSlice(u32, try c.readBytes(arg));
    }

    fn readBytes(c: *Ctx, arg: kernels.Arg) ![]align(4) u8 {
        const dst = try c.arena.alignedAlloc(u8, .@"4", @intCast(arg.size));
        c.token += 1;
        try c.rec.readBuffer(c.gpa, c.token, arg.buffer, arg.offset, dst);
        try c.flush();
        return dst;
    }
};

/// max_i |got_i - want_i| / max_i |want_i|: error relative to the output scale.
fn relErr(got: []const f32, want: []const f64) f64 {
    std.debug.assert(got.len == want.len);
    var max_err: f64 = 0;
    var scale: f64 = 0;
    for (got, want) |g, w| {
        max_err = @max(max_err, @abs(@as(f64, g) - w));
        scale = @max(scale, @abs(w));
    }
    return max_err / @max(scale, 1e-30);
}

fn checkRmsNorm(c: *Ctx) !Result {
    const rows = 7;
    const dim = 1000;
    const eps = 1e-6;
    const x = try c.random(rows * dim, 3);
    const w = try c.random(dim, 1);
    const y = try c.output(rows * dim);
    try c.launcher().launch(.rmsnorm, P.RmsNorm{ .rows = rows, .dim = dim, .eps = eps }, &.{ try c.upload(x), try c.upload(w), y }, .{ rows, 1, 1 });
    const got = try c.read(y);

    const want = try c.arena.alloc(f64, rows * dim);
    for (0..rows) |r| {
        var ss: f64 = 0;
        for (x[r * dim ..][0..dim]) |v| ss += @as(f64, v) * v;
        const scale = 1 / @sqrt(ss / dim + eps);
        for (0..dim) |i| want[r * dim + i] = x[r * dim + i] * scale * w[i];
    }
    return .{ .name = "rmsnorm", .err = relErr(got, want), .tolerance = 1e-6 };
}

/// With `accumulate`, the output starts random and the product adds to it.
fn matmulCase(c: *Ctx, name: []const u8, rows: u32, n: u32, k: u32, split: ?u32, accumulate: bool) !Result {
    const x = try c.random(rows * k, 1);
    const w = try c.randomBf16(n * k, 1);
    const init = try c.random(rows * n, if (accumulate) 4 else 0);
    const out = try c.upload(init);
    const xa = try c.upload(x);
    const l = c.launcher();
    // Optionally split the weights into two chunks, as large matrices are.
    var col0: u32 = 0;
    for ([_]u32{ split orelse n, n - (split orelse n) }) |cols| {
        if (cols == 0) continue;
        const wa = try c.upload(w.words[col0 * k / 2 ..][0 .. cols * k / 2]);
        try l.launch(.matmul_bf16, P.MatmulBf16{ .rows = rows, .n = cols, .k = k, .col0 = col0, .out_stride = n, .accumulate = @intFromBool(accumulate) }, &.{ xa, wa, out }, .{
            std.math.divCeil(u32, cols, kernels.matmul_tile_n) catch unreachable,
            std.math.divCeil(u32, rows, kernels.matmul_tile_t) catch unreachable,
            1,
        });
        col0 += cols;
    }
    const got = try c.read(out);

    const want = try c.arena.alloc(f64, rows * n);
    for (0..rows) |t| for (0..n) |j| {
        var s: f64 = init[t * n + j];
        for (0..k) |i| s += @as(f64, x[t * k + i]) * w.values[j * k + i];
        want[t * n + j] = s;
    };
    return .{ .name = name, .err = relErr(got, want), .tolerance = 1e-5 };
}

fn checkMatmul(c: *Ctx) !Result {
    // 19 tokens and 100 outputs leave partial tiles in both dimensions.
    return matmulCase(c, "matmul_bf16", 19, 100, 1024, null, false);
}

fn checkMatmulChunked(c: *Ctx) !Result {
    return matmulCase(c, "matmul_bf16 (2 chunks, +=)", 33, 150, 96, 70, true);
}

fn checkMatvec(c: *Ctx) !Result {
    // 3 tokens, 100 outputs (a partial workgroup of rows), split into two
    // chunks; k = 4000 is too wide to stage and leaves a partial pass over
    // the 32 lanes.
    return matvecCase(c, "matvec_bf16 (2 chunks)", 4000, 37, false);
}

fn checkMatvecNorm(c: *Ctx) !Result {
    return matvecCase(c, "matvec_bf16 norm+= (2 chunks)", 1000, 37, true);
}

/// out (+)= rmsnorm(x) · w for 3 tokens and 100 outputs in two chunks; with
/// `fused`, the RMSNorm prologue and accumulate epilogue are on.
fn matvecCase(c: *Ctx, name: []const u8, k: u32, split: u32, fused: bool) !Result {
    const rows = 3;
    const n = 100;
    const eps = 1e-6;
    const x = try c.random(rows * k, 3);
    const nw = try c.random(k, 1);
    const w = try c.randomBf16(n * k, 1);
    const init = try c.random(rows * n, if (fused) 4 else 0);
    const out = try c.upload(init);
    const xa = try c.upload(x);
    const na = try c.upload(nw);
    var col0: u32 = 0;
    for ([_]u32{ split, n - split }) |cols| {
        const wa = try c.upload(w.words[col0 * k / 2 ..][0 .. cols * k / 2]);
        try c.launcher().launch(.matvec_bf16, P.MatvecBf16{
            .n = cols,
            .k = k,
            .col0 = col0,
            .out_stride = n,
            .eps = eps,
            .flags = .{ .norm = fused, .accumulate = fused },
        }, &.{ xa, wa, out, na }, .{ std.math.divCeil(u32, cols, kernels.matvec_rows) catch unreachable, rows, 1 });
        col0 += cols;
    }
    const got = try c.read(out);
    const want = try c.arena.alloc(f64, rows * n);
    const xn = try c.arena.alloc(f64, k);
    for (0..rows) |t| {
        normRow(xn, x[t * k ..][0..k], if (fused) nw else null, eps);
        for (0..n) |j| {
            var s: f64 = init[t * n + j];
            for (0..k) |i| s += xn[i] * w.values[j * k + i];
            want[t * n + j] = s;
        }
    }
    return .{ .name = name, .err = relErr(got, want), .tolerance = 1e-5 };
}

/// out = rmsnorm(x) · w when `w` is given, else x.
fn normRow(out: []f64, x: []const f32, w: ?[]const f32, eps: f64) void {
    var ss: f64 = 0;
    for (x) |v| ss += @as(f64, v) * v;
    const scale = if (w != null) 1 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps) else 1;
    for (out, x, 0..) |*o, v, i| o.* = v * scale * if (w) |ww| ww[i] else 1;
}

/// The FFN up-projection: act = silu(rmsnorm(x) · gate) · (rmsnorm(x) · up),
/// gate and up stacked in one matrix.
fn checkMatvecSiluPair(c: *Ctx) !Result {
    const rows = 3;
    const ff = 100;
    const k = 1024;
    const eps = 1e-6;
    const x = try c.random(rows * k, 3);
    const nw = try c.random(k, 1);
    const w = try c.randomBf16(2 * ff * k, 1);
    const out = try c.output(rows * ff);
    try c.launcher().launch(.matvec_bf16, P.MatvecBf16{
        .n = ff,
        .k = k,
        .col0 = 0,
        .out_stride = ff,
        .eps = eps,
        .flags = .{ .norm = true, .silu_pair = true },
        .half = ff,
    }, &.{ try c.upload(x), try c.upload(w.words), out, try c.upload(nw) }, .{ std.math.divCeil(u32, ff, kernels.matvec_rows) catch unreachable, rows, 1 });
    const got = try c.read(out);
    const want = try c.arena.alloc(f64, rows * ff);
    const xn = try c.arena.alloc(f64, k);
    for (0..rows) |t| {
        normRow(xn, x[t * k ..][0..k], nw, eps);
        for (0..ff) |j| {
            var g: f64 = 0;
            var u: f64 = 0;
            for (0..k) |i| {
                g += xn[i] * w.values[j * k + i];
                u += xn[i] * w.values[(ff + j) * k + i];
            }
            want[t * ff + j] = g * sigmoid64(g) * u;
        }
    }
    return .{ .name = "matvec_bf16 norm+silu pair", .err = relErr(got, want), .tolerance = 1e-5 };
}

fn checkSiluMul(c: *Ctx) !Result {
    // 7 rows of a fused gate|up projection, 10,001 wide.
    const rows = 7;
    const width = 10_001;
    const n = rows * width;
    const gu = try c.random(2 * n, 6);
    const act = try c.output(n);
    try c.launcher().launch(.silu_mul, P.SiluMul{ .count = n, .width = width }, &.{ try c.upload(gu), act }, kernels.Launcher.elementwise(n));
    const got = try c.read(act);
    const want = try c.arena.alloc(f64, n);
    for (0..rows) |t| for (0..width) |j| {
        const g: f64 = gu[t * 2 * width + j];
        want[t * width + j] = g * sigmoid64(g) * gu[t * 2 * width + width + j];
    };
    return .{ .name = "silu_mul", .err = relErr(got, want), .tolerance = 1e-6 };
}

/// Random blocks of quantized type `t` for `rows` rows of `k` weights: every
/// byte random except the f16 super-scales, which are finite; plus their
/// values decoded on the CPU (ref.tensor, bitwise as ggml).
fn quantRows(c: *Ctx, t: gguf.TensorType, rows: usize, k: usize) !struct { bytes: []u8, values: []f32 } {
    const l = t.layout().?;
    const n_blocks = rows * k / l.block_elems;
    const bytes = try c.arena.alloc(u8, n_blocks * l.block_bytes);
    c.rand.bytes(bytes);
    const scales: []const usize = switch (t) {
        .q8_0 => &.{0},
        .q3_k => &.{108},
        .q4_k, .q5_k => &.{ 0, 2 },
        .q6_k => &.{208},
        else => unreachable,
    };
    for (0..n_blocks) |b| for (scales) |off| {
        const v: f16 = @floatCast((c.rand.float(f32) * 2 - 1) * 0.05);
        std.mem.writeInt(u16, bytes[b * l.block_bytes + off ..][0..2], @bitCast(v), .little);
    };
    const values = try c.arena.alloc(f32, rows * k);
    try ref_tensor.dequantize(t, bytes, values);
    return .{ .bytes = bytes, .values = values };
}

/// Uploads raw GGUF blocks of type `t` in the layout the quantized kernels
/// read: as they are, or for a padded type through `repad`, as the model
/// loader does.
fn gpuWeights(c: *Ctx, t: gguf.TensorType, raw: []const u8) !kernels.Arg {
    const l = t.layout().?;
    const stride = kernels.blockStride(t);
    const blocks = raw.len / l.block_bytes;
    const padded = try c.arena.alloc(u8, std.mem.alignForward(usize, raw.len, 16));
    @memset(padded, 0);
    @memcpy(padded[0..raw.len], raw);
    const src = try c.upload(padded);
    if (stride == l.block_bytes) return src;
    const out = try c.output(std.mem.alignForward(usize, blocks * stride, 16) / 4);
    try c.launcher().launch(.repad, P.Repad{ .blocks = @intCast(blocks), .src_bytes = l.block_bytes, .dst_bytes = stride }, &.{ src, out }, kernels.Launcher.elementwise(blocks * stride / 4));
    return out;
}

const Act = struct { qx: kernels.Arg, dx: kernels.Arg };

/// x (rows × k) rounded to the 8-bit blocks the prefill multiply takes for
/// weights of type `t` (quantize_q8).
fn quantizeX(c: *Ctx, t: gguf.TensorType, x: kernels.Arg, rows: u32, k: u32) !Act {
    const block = kernels.actBlock(t);
    const act: Act = .{ .qx = try c.output(rows * k / 4), .dx = try c.output(rows * k / block) };
    try c.launcher().launch(.quantize_q8, P.QuantizeQ8{ .k = k, .block = block }, &.{ x, act.qx, act.dx, x }, .{ k / 256, rows, 1 });
    return act;
}

/// quantize_q8 against the reference's rounding (roundLikeLlama), value
/// for value: rows of random values, one whose largest magnitude is
/// negative, one with a tie between +amax and −amax, and a zero row.
fn checkQuantize(c: *Ctx, t: gguf.TensorType, name: []const u8) !Result {
    const rows = 6;
    const k = 512;
    const x = try c.random(rows * k, 3);
    x[2 * k + 77] = -9.5;
    x[3 * k + 5] = -7.25;
    x[3 * k + 40] = 7.25;
    @memset(x[4 * k ..][0..k], 0);
    const act = try quantizeX(c, t, try c.upload(x), rows, k);
    const words = try c.read(act.qx);
    const d = try c.read(act.dx);
    const want = try c.arena.dupe(f32, x);
    for (0..rows) |r| ref_matmul.roundLikeLlama(t, want[r * k ..][0..k]);
    const block = kernels.actBlock(t);
    var worst: f64 = 0;
    for (want, 0..) |v, i| {
        const word: u32 = @bitCast(words[i / 4]);
        const q: i8 = @bitCast(@as(u8, @truncate(word >> @intCast(8 * (i % 4)))));
        const got = @as(f32, @floatFromInt(q)) * d[i / block];
        worst = @max(worst, @abs(@as(f64, got) - v) / @max(@abs(@as(f64, v)), 1e-30));
    }
    return .{ .name = name, .err = worst, .tolerance = 0 };
}

/// quantize_q8 computing its input (RMSNorm, SiLU pair) before rounding:
/// every value must be a rounding of the f64 result to its block's step,
/// within half a step (the GPU's sums may differ from f64 by ulps).
fn checkQuantizeFused(c: *Ctx, mode: u32, name: []const u8) !Result {
    const rows = 5;
    const k = 768;
    const width: usize = if (mode == 2) 2 * k else k;
    const x = try c.random(rows * width, 2);
    const nw = try c.random(k, 1);
    const eps: f32 = 1e-6;
    const act: Act = .{ .qx = try c.output(rows * k / 4), .dx = try c.output(rows * k / 256) };
    const xa = try c.upload(x);
    try c.launcher().launch(.quantize_q8, P.QuantizeQ8{ .k = k, .block = 256, .mode = mode, .eps = eps }, &.{ xa, act.qx, act.dx, try c.upload(nw) }, .{ k / 256, rows, 1 });
    const words = try c.read(act.qx);
    const d = try c.read(act.dx);
    var worst: f64 = 0;
    for (0..rows) |r| {
        var ss: f64 = 0;
        for (x[r * width ..][0..k]) |v| ss += @as(f64, v) * v;
        const scale = 1 / @sqrt(ss / k + eps);
        for (0..k) |i| {
            const want: f64 = if (mode == 1) x[r * width + i] * scale * nw[i] else blk: {
                const g: f64 = x[r * width + i];
                break :blk g / (1 + @exp(-g)) * x[r * width + k + i];
            };
            const j = r * k + i;
            const word: u32 = @bitCast(words[j / 4]);
            const q: i8 = @bitCast(@as(u8, @truncate(word >> @intCast(8 * (j % 4)))));
            const step: f64 = @abs(d[j / 256]);
            worst = @max(worst, @abs(@as(f64, @floatFromInt(q)) * d[j / 256] - want) / (step / 2));
        }
    }
    return .{ .name = name, .err = worst, .tolerance = 1.0001 };
}

fn checkQuantizeNorm(c: *Ctx) !Result {
    return checkQuantizeFused(c, 1, "quantize_q8 of RMSNorm (steps/2)");
}

fn checkQuantizeSilu(c: *Ctx) !Result {
    return checkQuantizeFused(c, 2, "quantize_q8 of SiLU pair (steps/2)");
}

fn checkQuantizeQ8K(c: *Ctx) !Result {
    return checkQuantize(c, .q4_k, "quantize_q8 Q8_K (256)");
}

fn checkQuantizeQ80(c: *Ctx) !Result {
    return checkQuantize(c, .q8_0, "quantize_q8 Q8_0 (32)");
}

/// The quantized kernels for type `t`: the decode multiply with its norm
/// prologue and accumulate epilogue, the prefill multiply accumulating, and
/// the embedding gather over two table chunks.
fn Quant(comptime t: gguf.TensorType) type {
    return struct {
        fn matvec(c: *Ctx) !Result {
            return quantMatvec(c, t, "matvec " ++ @tagName(t) ++ " norm+=", 1024, false);
        }

        /// The prefill multiply accumulating, over more than one tile each
        /// way: whole, and split over k into partials that matmul_reduce
        /// adds up (kernels.matmulSlices), on x rounded by quantize_q8.
        fn matmul(c: *Ctx) !Result {
            const rows = 70;
            const n = 150;
            const k = 1024;
            const s = kernels.matmulSlices(n, k);
            std.debug.assert(s.count > 1);
            const x = try c.upload(try c.random(rows * k, 1));
            // The multiply sees x rounded to 8-bit blocks, as the
            // reference's llama.cpp mode rounds it.
            const xs = try c.read(x);
            for (0..rows) |r| ref_matmul.roundLikeLlama(t, xs[r * k ..][0..k]);
            const act = try quantizeX(c, t, x, rows, k);
            const w = try quantRows(c, t, n, k);
            const wa = try gpuWeights(c, t, w.bytes);
            const init = try c.random(rows * n, 4);
            const whole = try c.upload(init);
            const split = try c.upload(init);
            const partials = try c.output(s.count * rows * n);
            const grid: [3]u32 = .{ std.math.divCeil(u32, n, kernels.matmul_quant_tile_n) catch unreachable, std.math.divCeil(u32, rows, kernels.matmul_quant_tile_t) catch unreachable, 1 };
            const l = c.launcher();
            const kernel = kernels.Kernel.quant(.matmul, t).?;
            try l.launch(kernel, P.MatmulQuant{ .rows = rows, .n = n, .k = k, .col0 = 0, .out_stride = n, .accumulate = 1, .k_slice = k }, &.{ act.qx, wa, whole, act.dx }, grid);
            try l.launch(kernel, P.MatmulQuant{ .rows = rows, .n = n, .k = k, .col0 = 0, .out_stride = n, .k_slice = s.k_slice, .slice_stride = rows * n }, &.{ act.qx, wa, partials, act.dx }, .{ grid[0], grid[1], s.count });
            try l.launch(.matmul_reduce, P.MatmulReduce{ .rows = rows, .n = n, .slices = s.count, .col0 = 0, .out_stride = n, .accumulate = 1, .slice_stride = rows * n }, &.{ partials, split }, kernels.Launcher.elementwise(rows * n));
            const want = try c.arena.alloc(f64, rows * n);
            for (0..rows) |tk| for (0..n) |j| {
                var sum: f64 = init[tk * n + j];
                for (0..k) |i| sum += @as(f64, xs[tk * k + i]) * w.values[j * k + i];
                want[tk * n + j] = sum;
            };
            const err = @max(relErr(try c.read(whole), want), relErr(try c.read(split), want));
            return .{ .name = "matmul " ++ @tagName(t) ++ " += (whole, split k)", .err = err, .tolerance = 1e-5 };
        }

        fn embed(c: *Ctx) !Result {
            const vocab = 50;
            const dim = 512;
            const split = 30;
            const l = t.layout().?;
            const row_bytes = dim / l.block_elems * l.block_bytes;
            const table = try quantRows(c, t, vocab, dim);
            const tokens = [_]u32{ 0, 29, 30, 49, 7, 30, 12 };
            const out = try c.output(tokens.len * dim);
            const ta = try c.upload(&tokens);
            for ([_][2]u32{ .{ 0, split }, .{ split, vocab - split } }) |chunk| {
                const bytes = table.bytes[chunk[0] * row_bytes ..][0 .. chunk[1] * row_bytes];
                try c.launcher().launch(kernels.Kernel.quant(.embed, t).?, P.EmbedBf16{ .dim = dim, .row0 = chunk[0], .chunk_rows = chunk[1] }, &.{ ta, try gpuWeights(c, t, bytes), out }, .{ tokens.len, 1, 1 });
            }
            const got = try c.read(out);
            const want = try c.arena.alloc(f64, tokens.len * dim);
            for (tokens, 0..) |tok, i| for (0..dim) |e| {
                want[i * dim + e] = table.values[tok * dim + e];
            };
            if (t == .q3_k) std.debug.print("got {any}\nwant {any}\n", .{ got[0..40], want[0..40] });
            if (t == .q3_k) std.debug.print("got {any}\nwant {any}\n", .{ got[0..40], want[0..40] });
            if (t == .q3_k) std.debug.print("got {any}\nwant {any}\n", .{ got[0..40], want[0..40] });
            if (t == .q3_k) std.debug.print("bytes {any}\ngot {any}\nwant {any}\n", .{ table.bytes[0..110], got[0..4], want[0..4] });
            return .{ .name = "embed " ++ @tagName(t) ++ " (2 chunks)", .err = relErr(got, want), .tolerance = 1e-6 };
        }
    };
}

/// The decode multiply on quantized weights: 3 tokens, 100 outputs (a
/// partial workgroup of rows), norm prologue and accumulate epilogue; or,
/// with `pair`, the SiLU pair over stacked gate and up rows.
fn quantMatvec(c: *Ctx, comptime t: gguf.TensorType, name: []const u8, k: u32, pair: bool) !Result {
    const rows = 3;
    const n = 100;
    const eps = 1e-6;
    const x = try c.random(rows * k, 3);
    const nw = try c.random(k, 1);
    const w = try quantRows(c, t, if (pair) 2 * n else n, k);
    const init = try c.random(rows * n, if (pair) 0 else 4);
    const out = try c.upload(init);
    try c.launcher().launch(kernels.Kernel.quant(.matvec, t).?, P.MatvecBf16{
        .n = n,
        .k = k,
        .col0 = 0,
        .out_stride = n,
        .eps = eps,
        .flags = .{ .norm = true, .accumulate = !pair, .silu_pair = pair },
        .half = if (pair) n else 0,
    }, &.{ try c.upload(x), try gpuWeights(c, t, w.bytes), out, try c.upload(nw) }, .{ std.math.divCeil(u32, n, kernels.matvec_quant_rows) catch unreachable, rows, 1 });
    const got = try c.read(out);
    const want = try c.arena.alloc(f64, rows * n);
    const xn = try c.arena.alloc(f64, k);
    for (0..rows) |tk| {
        normRow(xn, x[tk * k ..][0..k], nw, eps);
        for (0..n) |j| {
            var g: f64 = 0;
            var u: f64 = 0;
            for (0..k) |i| {
                g += xn[i] * w.values[j * k + i];
                if (pair) u += xn[i] * w.values[(n + j) * k + i];
            }
            want[tk * n + j] = if (pair) g * sigmoid64(g) * u else init[tk * n + j] + g;
        }
    }
    return .{ .name = name, .err = relErr(got, want), .tolerance = 1e-5 };
}

fn checkQuantTiles(c: *Ctx) !Result {
    // Two staged tiles of x, as the 27B's 5,120-wide inputs take three.
    return quantMatvec(c, .q4_k, "matvec q4_k norm+= (2 tiles)", 4096, false);
}

fn checkQuantSiluPair(c: *Ctx) !Result {
    return quantMatvec(c, .q3_k, "matvec q3_k norm+silu pair", 1024, true);
}

fn checkEmbed(c: *Ctx) !Result {
    const vocab = 50;
    const dim = 64;
    const split = 30;
    const table = try c.randomBf16(vocab * dim, 1);
    const tokens = [_]u32{ 0, 29, 30, 49, 7, 30, 12 };
    const out = try c.output(tokens.len * dim);
    const ta = try c.upload(&tokens);
    const l = c.launcher();
    for ([_][2]u32{ .{ 0, split }, .{ split, vocab - split } }) |chunk| {
        const words = table.words[chunk[0] * dim / 2 ..][0 .. chunk[1] * dim / 2];
        try l.launch(.embed_bf16, P.EmbedBf16{ .dim = dim, .row0 = chunk[0], .chunk_rows = chunk[1] }, &.{ ta, try c.upload(words), out }, .{ tokens.len, 1, 1 });
    }
    const got = try c.read(out);
    const want = try c.arena.alloc(f64, tokens.len * dim);
    for (tokens, 0..) |tok, t| for (0..dim) |i| {
        want[t * dim + i] = table.values[tok * dim + i];
    };
    return .{ .name = "embed_bf16 (2 chunks)", .err = relErr(got, want), .tolerance = 0 };
}

fn sigmoid64(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// head ← rmsnorm(head) · w, then NeoX RoPE at `pos` over the first n_rot
/// dims, in float64.
fn normRope(head: []f64, w: []const f32, pos: usize, n_rot: u32, base: f64, eps: f64) void {
    var ss: f64 = 0;
    for (head) |v| ss += v * v;
    const scale = 1 / @sqrt(ss / @as(f64, @floatFromInt(head.len)) + eps);
    for (head, w) |*v, wi| v.* = v.* * scale * wi;
    const half = n_rot / 2;
    for (0..half) |i| {
        const theta = @as(f64, @floatFromInt(pos)) * std.math.pow(f64, base, -2.0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n_rot)));
        const x0 = head[i];
        const x1 = head[i + half];
        head[i] = x0 * @cos(theta) - x1 * @sin(theta);
        head[i + half] = x0 * @sin(theta) + x1 * @cos(theta);
    }
}

/// rope_append on 5 tokens at positions 7..11 of a fused Q|K|V projection:
/// 3 query heads of 32 interleaved with gates, 2 KV heads, rotation over the
/// first 16 dims, rows padded past V. Checks the whole projection (Q heads
/// rotated in place, gates, K and V untouched) and the cache (K normalized
/// and rotated, V copied, earlier positions untouched).
fn ropeAppendCase(c: *Ctx, name: []const u8, half: bool) !Result {
    const tokens = 5;
    const heads = 3;
    const kv = 2;
    const hd = 32;
    const n_rot = 16;
    const pos0 = 7;
    const eps = 1e-6;
    const base = 1e7;
    const width = kv * hd;
    const k_off = heads * 2 * hd;
    const v_off = k_off + width;
    const row = v_off + width + 8;
    const x = try c.random(tokens * row, 2);
    const qw = try c.random(hd, 1);
    const kw = try c.random(hd, 1);
    const table = try kernels.ropeTable(c.arena, pos0 + tokens, n_rot, base);
    const xa = try c.upload(x);
    const words: u32 = if (half) width / 2 else width;
    const kc = try c.output((pos0 + tokens) * words);
    const vc = try c.output((pos0 + tokens) * words);
    try c.launcher().launch(.rope_append, P.RopeAppend{
        .heads = heads,
        .kv_heads = kv,
        .head_dim = hd,
        .n_rot = n_rot,
        .row = row,
        .k_off = k_off,
        .v_off = v_off,
        .f16 = @intFromBool(half),
        .eps = eps,
    }, &.{ xa, try c.upload(qw), try c.upload(kw), try c.upload(table), kc, vc, try c.step(tokens, pos0, 0) }, .{ heads + kv, tokens, 1 });
    const ck = try decodeCache(c, try c.readWords(kc), half);
    const cv = try decodeCache(c, try c.readWords(vc), half);
    for (ck[0 .. pos0 * width], cv[0 .. pos0 * width]) |a, b| if (a != 0 or b != 0) return .{ .name = name, .err = 1, .tolerance = 0 };
    const cached = tokens * width;
    const got = try c.arena.alloc(f32, x.len + 2 * cached);
    @memcpy(got[0..x.len], try c.read(xa));
    @memcpy(got[x.len..][0..cached], ck[pos0 * width ..]);
    @memcpy(got[x.len + cached ..], cv[pos0 * width ..]);

    const want = try c.arena.alloc(f64, got.len);
    for (want[0..x.len], x) |*o, v| o.* = v;
    for (0..tokens) |t| {
        for (0..heads) |h| normRope(want[t * row + h * 2 * hd ..][0..hd], qw, pos0 + t, n_rot, base, eps);
        const kd = want[x.len + t * width ..][0..width];
        const vd = want[x.len + cached + t * width ..][0..width];
        for (kd, vd, x[t * row + k_off ..][0..width], x[t * row + v_off ..][0..width]) |*a, *b, kv_, vv| {
            a.* = kv_;
            b.* = vv;
        }
        for (0..kv) |g| normRope(kd[g * hd ..][0..hd], kw, pos0 + t, n_rot, base, eps);
    }
    // The table is f32 from the reference formula, so angles carry f32
    // error; the f16 cache rounds to 2^-11 relative.
    return .{ .name = name, .err = relErr(got, want), .tolerance = if (half) 1e-3 else 1e-5 };
}

fn checkRopeAppendF32(c: *Ctx) !Result {
    return ropeAppendCase(c, "rope_append (f32 cache)", false);
}

fn checkRopeAppendF16(c: *Ctx) !Result {
    return ropeAppendCase(c, "rope_append (f16 cache)", true);
}

/// A cache of random values for `total` positions, as the kernels store it:
/// f16 pairs or f32 bits. `values` are rounded to what the cache holds.
fn cacheWords(c: *Ctx, values: []f32, half: bool) ![]u32 {
    if (!half) return @constCast(std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(values)));
    const words = try c.arena.alloc(u32, values.len / 2);
    for (words, 0..) |*w, i| {
        const lo: f16 = @floatCast(values[2 * i]);
        const hi: f16 = @floatCast(values[2 * i + 1]);
        values[2 * i] = lo;
        values[2 * i + 1] = hi;
        w.* = @as(u32, @as(u16, @bitCast(lo))) | (@as(u32, @as(u16, @bitCast(hi))) << 16);
    }
    return words;
}

/// The Qwen 3.5 attention shape: 8 query heads over 2 KV heads, head_dim
/// 256, queries in rows padded past their gates (q_row).
const AttnShape = struct {
    const heads = 8;
    const kv = 2;
    const hd = 256;
    const width = kv * hd;
    const q_row = heads * 2 * hd + 96;
    const scale: f32 = 1.0 / 16.0;
};

/// Gated attention of `second` queries at positions first.. over the cache,
/// in float64.
fn attnWant(c: *Ctx, qg: []const f32, k: []const f32, v: []const f32, first: u32, second: u32) ![]f64 {
    const S = AttnShape;
    const want = try c.arena.alloc(f64, second * S.heads * S.hd);
    const s = try c.arena.alloc(f64, first + second);
    for (0..second) |t| for (0..S.heads) |h| {
        const kvh = h / (S.heads / S.kv);
        const keys = first + t + 1;
        const q = qg[t * S.q_row + h * 2 * S.hd ..][0..S.hd];
        var max: f64 = -std.math.inf(f64);
        for (0..keys) |j| {
            var dot: f64 = 0;
            for (q, k[j * S.width + kvh * S.hd ..][0..S.hd]) |a, b| dot += @as(f64, a) * b;
            s[j] = dot * S.scale;
            max = @max(max, s[j]);
        }
        var sum: f64 = 0;
        for (s[0..keys]) |*e| {
            e.* = @exp(e.* - max);
            sum += e.*;
        }
        for (0..S.hd) |d| {
            var o: f64 = 0;
            for (0..keys) |j| o += s[j] * v[j * S.width + kvh * S.hd + d];
            want[(t * S.heads + h) * S.hd + d] = o / sum * sigmoid64(qg[t * S.q_row + h * 2 * S.hd + S.hd + d]);
        }
    };
    return want;
}

/// Prefill attention for 17 queries after 270 cached positions: 287 keys
/// span two of the kernel's 256-key tiles.
fn attentionCase(c: *Ctx, name: []const u8, half: bool) !Result {
    const S = AttnShape;
    const first = 270;
    const second = 17;
    const total = first + second;
    const k = try c.random(total * S.width, 1);
    const v = try c.random(total * S.width, 1);
    const qg = try c.random(second * S.q_row, 1);
    const out = try c.output(second * S.heads * S.hd);
    try c.launcher().launch(.attention, P.Attention{ .heads = S.heads, .kv_heads = S.kv, .head_dim = S.hd, .f16 = @intFromBool(half), .scale = S.scale, .q_row = S.q_row }, &.{
        try c.upload(qg), try c.upload(try cacheWords(c, k, half)), try c.upload(try cacheWords(c, v, half)), out, try c.step(second, first, 0),
    }, .{ S.heads, second, 1 });
    const got = try c.read(out);
    return .{ .name = name, .err = relErr(got, try attnWant(c, qg, k, v, first, second)), .tolerance = 1e-5 };
}

fn checkAttentionF32(c: *Ctx) !Result {
    return attentionCase(c, "attention (f32 cache)", false);
}

fn checkAttentionF16(c: *Ctx) !Result {
    return attentionCase(c, "attention (f16 cache)", true);
}

/// Decode attention: 3 queries after 150 cached positions, through
/// attn_split (three 64-position blocks, the last partial) and attn_combine.
fn checkAttnDecode(c: *Ctx) !Result {
    const S = AttnShape;
    const first = 150;
    const second = 3;
    const total = first + second;
    const k = try c.random(total * S.width, 1);
    const v = try c.random(total * S.width, 1);
    const qg = try c.random(second * S.q_row, 1);
    const out = try c.output(second * S.heads * S.hd);
    const max_blocks = 4;
    const part = try c.output(second * S.heads * max_blocks * (S.hd + 2));
    const l = c.launcher();
    const step = try c.step(second, first, 0);
    const params: P.AttnDecode = .{ .heads = S.heads, .kv_heads = S.kv, .head_dim = S.hd, .f16 = 1, .scale = S.scale, .max_blocks = max_blocks, .q_row = S.q_row };
    const qa = try c.upload(qg);
    const grid: kernels.Grid = .{ .key_blocks = .{ .x = S.kv, .block = kernels.attn_block } };
    try l.launch(.attn_split, params, &.{ qa, try c.upload(try cacheWords(c, k, true)), try c.upload(try cacheWords(c, v, true)), part, step }, grid.resolve(.{ .rows = second, .keys = total }));
    try l.launch(.attn_combine, params, &.{ qa, part, out, step }, .{ S.heads, second, 1 });
    const got = try c.read(out);
    return .{ .name = "attn_split + combine", .err = relErr(got, try attnWant(c, qg, k, v, first, second)), .tolerance = 1e-5 };
}

/// Cache words as f32 values.
fn decodeCache(c: *Ctx, words: []const u32, half: bool) ![]f32 {
    if (!half) return @constCast(std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(words)));
    const values = try c.arena.alloc(f32, words.len * 2);
    for (words, 0..) |w, i| {
        values[2 * i] = @as(f16, @bitCast(@as(u16, @truncate(w))));
        values[2 * i + 1] = @as(f16, @bitCast(@as(u16, @truncate(w >> 16))));
    }
    return values;
}

/// delta_prep over 12 tokens of a fused DeltaNet projection (2 key heads, 4
/// value heads of 128) in steps of 5, 2, 1 and 4 tokens, with delta_rule
/// advancing the conv history between them: steps shorter than the history
/// shift it. Checks the conv output (SiLU, L2 norm on q and k) and the gates
/// of every token. α spans softplus's linear cut at 20.
fn checkDeltaPrep(c: *Ctx) !Result {
    const steps = [_]u32{ 5, 2, 1, 4 };
    const tokens = 12;
    const hk = 2;
    const hv = 4;
    const d = 128;
    const taps = 4;
    const eps = 1e-6;
    const key_channels = 2 * hk * d;
    const channels = key_channels + hv * d;
    const inner = hv * d;
    const beta_off = channels + inner;
    const alpha_off = beta_off + hv;
    const row = alpha_off + hv;
    const proj = try c.random(tokens * row, 2);
    for (0..tokens) |t| for (proj[t * row + alpha_off ..][0..hv]) |*v| {
        v.* *= 12;
    };
    const w = try c.random(channels * taps, 1);
    const dt = try c.random(hv, 1);
    const a = try c.random(hv, 1);
    for (a) |*v| v.* = -@exp(v.*);
    const wa = try c.upload(w);
    const dta = try c.upload(dt);
    const aa = try c.upload(a);
    const hist = try c.output((taps - 1) * channels);
    const state = try c.output(hv * d * d);
    const per = channels + 2 * hv;
    const got = try c.arena.alloc(f32, tokens * per);
    var t0: u32 = 0;
    for (steps) |m| {
        const pa = try c.upload(proj[t0 * row ..][0 .. m * row]);
        const conv = try c.output(m * channels);
        const g = try c.output(m * hv);
        const beta = try c.output(m * hv);
        const step = try c.step(m, t0, 0);
        const l = c.launcher();
        try l.launch(.delta_prep, P.DeltaPrep{
            .channels = channels,
            .taps = taps,
            .key_channels = key_channels,
            .head = d,
            .row = row,
            .v_heads = hv,
            .beta_off = beta_off,
            .alpha_off = alpha_off,
            .eps = eps,
        }, &.{ pa, wa, hist, conv, dta, aa, g, beta }, .{ channels / d + 1, m, 1 });
        try l.launch(.delta_rule, P.DeltaRule{ .k_heads = hk, .v_heads = hv, .dim = d, .conv_stride = channels, .taps = taps, .row = row }, &.{
            conv, g, beta, state, try c.output(m * inner), step, pa, hist,
        }, .{ hv, std.math.divCeil(u32, d, kernels.delta_cols) catch unreachable, 1 });
        const cv = try c.read(conv);
        const gv = try c.read(g);
        const bv = try c.read(beta);
        for (0..m) |t| {
            const dst = got[(t0 + t) * per ..][0..per];
            @memcpy(dst[0..channels], cv[t * channels ..][0..channels]);
            @memcpy(dst[channels..][0..hv], gv[t * hv ..][0..hv]);
            @memcpy(dst[channels + hv ..][0..hv], bv[t * hv ..][0..hv]);
        }
        t0 += m;
    }

    const want = try c.arena.alloc(f64, tokens * per);
    for (0..tokens) |t| {
        const dst = want[t * per ..][0..per];
        for (0..channels) |ch| {
            var sum: f64 = 0;
            for (0..taps) |tap| {
                const back = taps - 1 - tap;
                if (t >= back) sum += @as(f64, w[ch * taps + tap]) * proj[(t - back) * row + ch];
            }
            dst[ch] = sum * sigmoid64(sum);
        }
        for (0..2 * hk) |h| {
            const head = dst[h * d ..][0..d];
            var ss: f64 = 0;
            for (head) |v| ss += v * v;
            for (head) |*v| v.* /= @sqrt(ss + eps);
        }
        for (0..hv) |j| {
            const xv = @as(f64, proj[t * row + alpha_off + j]) + dt[j];
            dst[channels + j] = (if (xv > 20) xv else @log(1 + @exp(xv))) * a[j];
            dst[channels + hv + j] = sigmoid64(proj[t * row + beta_off + j]);
        }
    }
    return .{ .name = "delta_prep (4 steps)", .err = relErr(got, want), .tolerance = 1e-5 };
}

fn checkDeltaRule(c: *Ctx) !Result {
    // 4 value heads over 2 key heads (tiled mapping), dim 128, 37 tokens in
    // two steps of 20 and 17 that carry the state between them. gated_norm
    // reads z from rows of a wider projection.
    const tokens = 37;
    const hk = 2;
    const hv = 4;
    const d = 128;
    const eps = 1e-6;
    const key_dim = hk * d;
    const inner = hv * d;
    const stride = 2 * key_dim + inner;
    const z_off = 40;
    const z_row = z_off + inner + 24;
    const conv = try c.random(tokens * stride, 1);
    // q, k arrive L2-normalized.
    for (0..tokens) |t| for (0..2 * hk) |h| {
        const head = conv[t * stride + h * d ..][0..d];
        var ss: f32 = 0;
        for (head) |v| ss += v * v;
        for (head) |*v| v.* /= @sqrt(ss);
    };
    const g = try c.random(tokens * hv, 1);
    for (g) |*v| v.* = -@abs(v.*); // decays in (e^-1, 1]
    const beta = try c.random(tokens * hv, 1);
    for (beta) |*v| v.* = @abs(v.*);
    const zp = try c.random(tokens * z_row, 2);
    const norm_w = try c.random(d, 1);
    const state = try c.output(hv * d * d);
    const wn = try c.upload(norm_w);
    // One tap: no conv history to advance.
    const hist = try c.output(1);
    const got = try c.arena.alloc(f32, tokens * inner);
    var t0: u32 = 0;
    for ([_]u32{ 20, 17 }) |m| {
        const out = try c.output(m * inner);
        const za = try c.upload(zp[t0 * z_row ..][0 .. m * z_row]);
        const l = c.launcher();
        try l.launch(.delta_rule, P.DeltaRule{
            .k_heads = hk,
            .v_heads = hv,
            .dim = d,
            .conv_stride = stride,
            .taps = 1,
            .row = z_row,
        }, &.{
            try c.upload(conv[t0 * stride ..][0 .. m * stride]), try c.upload(g[t0 * hv ..][0 .. m * hv]),
            try c.upload(beta[t0 * hv ..][0 .. m * hv]),         state,
            out,                                                 try c.step(m, t0, 0),
            za,                                                  hist,
        }, .{ hv, std.math.divCeil(u32, d, kernels.delta_cols) catch unreachable, 1 });
        try l.launch(.gated_norm, P.GatedNorm{ .v_heads = hv, .dim = d, .eps = eps, .z_row = z_row, .z_off = z_off }, &.{ out, za, wn }, .{ hv, m, 1 });
        @memcpy(got[t0 * inner ..][0 .. m * inner], try c.read(out));
        t0 += m;
    }

    const want = try c.arena.alloc(f64, tokens * inner);
    const s = try c.arena.alloc(f64, d * d);
    const o = try c.arena.alloc(f64, d);
    for (0..hv) |j| {
        @memset(s, 0);
        const kh = j % hk;
        for (0..tokens) |t| {
            const row = conv[t * stride ..][0..stride];
            const q = row[kh * d ..][0..d];
            const k = row[key_dim + kh * d ..][0..d];
            const v = row[2 * key_dim + j * d ..][0..d];
            const decay = @exp(@as(f64, g[t * hv + j]));
            for (s) |*x| x.* *= decay;
            for (0..d) |b| {
                var stk: f64 = 0;
                for (0..d) |a| stk += s[a * d + b] * k[a];
                const delta = (v[b] - stk) * beta[t * hv + j];
                for (0..d) |a| s[a * d + b] += k[a] * delta;
            }
            var ss: f64 = 0;
            for (0..d) |b| {
                var acc: f64 = 0;
                for (0..d) |a| acc += s[a * d + b] * q[a];
                o[b] = acc / @sqrt(@as(f64, d));
                ss += o[b] * o[b];
            }
            const r = 1 / @sqrt(ss / d + eps);
            for (0..d) |b| {
                const zz: f64 = zp[t * z_row + z_off + j * d + b];
                want[t * inner + j * d + b] = o[b] * r * norm_w[b] * zz * sigmoid64(zz);
            }
        }
    }
    return .{ .name = "delta_rule+norm (2 steps)", .err = relErr(got, want), .tolerance = 1e-5 };
}

/// Gaussian logits (σ = 2): every threshold (top-k, top-p, min-p) cuts
/// through a dense band of tokens, so moving it changes the draws.
fn sampleLogits(c: *Ctx, vocab: u32) ![]f32 {
    const logits = try c.arena.alloc(f32, vocab);
    for (logits) |*l| l.* = c.rand.floatNorm(f32) * 2;
    return logits;
}

/// Runs the GPU sampler `trials` times at seeded draws. Each token must be
/// the CPU reference's draw up to f32 rounding: `sampler.deviation`, the
/// distance from the draw to the token's CDF interval with the min-p and
/// top-p cuts allowed to move by the same relative amount, at most
/// `draw_tolerance`. Also checks the stop flag, the next-step token and the
/// count bump on every trial.
fn sampleCase(c: *Ctx, name: []const u8, vocab: u32, p: sampler.Params, trials: u32) !Result {
    const logits = try sampleLogits(c, vocab);
    const prompt = try c.arena.alloc(bool, vocab);
    const prompt_words = try c.arena.alloc(u32, vocab);
    const counts = try c.arena.alloc(u32, vocab);
    for (prompt, prompt_words, counts) |*in, *word, *n| {
        in.* = c.rand.uintLessThan(u32, 20) == 0;
        word.* = @intFromBool(in.*);
        n.* = if (c.rand.uintLessThan(u32, 30) == 0) 1 + c.rand.uintLessThan(u32, 3) else 0;
    }
    var stops: [3]u32 = undefined;
    for (&stops) |*s| s.* = c.rand.uintLessThan(u32, vocab);
    // Make greedy and top-1 runs stop: the greedy token is a stop token.
    stops[2] = try sampler.reference(c.arena, logits, .{ .temperature = 0, .repetition_penalty = p.repetition_penalty, .presence_penalty = p.presence_penalty, .frequency_penalty = p.frequency_penalty, .logit_bias = p.logit_bias }, .{ .prompt = prompt, .counts = counts }, 0);

    var bias_words: [kernels.max_bias][2]u32 = undefined;
    const biases = p.biasWords(&bias_words);
    const block = p.gpu(vocab, 0, &stops);
    const params = try c.uniform(block);
    var steps_buf: [kernels.max_sample_steps]kernels.SampleStep = undefined;
    const steps = kernels.sampleSteps(block, &steps_buf);
    const la = try c.upload(logits);
    const ba = try c.upload(if (biases.len > 0) biases else &[_][2]u32{.{ 0, 0 }});
    const pa = try c.upload(prompt_words);
    const ca = try c.output(vocab);
    const work = try c.output(vocab);
    const state = try c.output(kernels.sampleStateWords(vocab));
    const next = try c.output(4);

    var draws: sampler.Draws = .init(p.seed);
    var worst: f64 = 0;
    var exact: u32 = 0;
    for (0..trials) |_| {
        const u = draws.next();
        try c.rec.writeBuffer(c.gpa, ca.buffer, 0, std.mem.sliceAsBytes(counts));
        const step = try c.step(1, 0, u);
        for (steps) |ss| {
            const groups: u32 = if (ss.phase.wide()) kernels.sampleGroups(vocab) else 1;
            try c.launcher().launch(.sample, P.SamplePhase{ .phase = @backingInt(ss.phase), .round = ss.round }, &.{
                la, ba, pa, ca, work, state, next, step, params,
            }, .{ groups, 1, 1 });
        }
        const got = try c.readWords(state);
        const got_next = try c.readWords(next);
        const got_counts = try c.readWords(ca);
        const want = try sampler.reference(c.arena, logits, p, .{ .prompt = prompt, .counts = counts }, u);
        const token = got[0];
        const stop = std.mem.indexOfScalar(u32, &stops, token) != null;
        const consistent = token < vocab and got[1] == @intFromBool(stop) and got_next[0] == token and
            got_counts[token] == counts[token] + 1;
        if (!consistent) return .{ .name = name, .err = 1, .tolerance = draw_tolerance, .trials = trials };
        if (token == want) exact += 1;
        worst = @max(worst, try sampler.deviation(c.arena, logits, p, .{ .prompt = prompt, .counts = counts }, u, token, draw_tolerance));
    }
    return .{ .name = name, .err = worst, .tolerance = draw_tolerance, .trials = trials, .exact = exact };
}

/// f32 accumulation over the vocabulary moves CDF positions by a few 1e-8
/// of the mass (4.4e-8 is the largest seen), enough to cross into a
/// neighbouring token's interval, which average 3e-5 of the mass at 32k
/// tokens.
const draw_tolerance = 1e-6;

const test_bias = [_]sampler.Bias{ .{ .token = 3, .bias = 9 }, .{ .token = 17, .bias = -50 }, .{ .token = 900, .bias = 4.5 } };

fn checkSampleGreedy(c: *Ctx) !Result {
    return sampleCase(c, "sample greedy+penalties", 32000, .{
        .temperature = 0,
        .repetition_penalty = 1.3,
        .presence_penalty = 0.5,
        .frequency_penalty = 0.2,
        .logit_bias = &test_bias,
    }, 4);
}

fn checkSampleTopKTopP(c: *Ctx) !Result {
    return sampleCase(c, "sample k=40 p=0.9", 32000, .{ .temperature = 0.7, .top_k = 40, .top_p = 0.9, .seed = 1 }, 16);
}

fn checkSampleMinP(c: *Ctx) !Result {
    return sampleCase(c, "sample min-p 0.05", 32000, .{ .min_p = 0.05, .seed = 2 }, 16);
}

fn checkSampleTopP(c: *Ctx) !Result {
    return sampleCase(c, "sample top-p 0.5", 32000, .{ .temperature = 1.3, .top_p = 0.5, .seed = 3 }, 16);
}

fn checkSamplePlain(c: *Ctx) !Result {
    return sampleCase(c, "sample T=1", 32000, .{ .seed = 4, .logit_bias = &test_bias, .repetition_penalty = 1.1 }, 16);
}

fn checkSampleTopK1(c: *Ctx) !Result {
    return sampleCase(c, "sample top-k 1", 32000, .{ .top_k = 1, .seed = 5 }, 4);
}

fn checkSampleLargeTopK(c: *Ctx) !Result {
    // Past the candidate cap: radix select, then top-p by bisection.
    return sampleCase(c, "sample k=2000 p=0.8", 32000, .{ .top_k = 2000, .top_p = 0.8, .seed = 7 }, 8);
}

fn checkSampleTopKSkipped(c: *Ctx) !Result {
    // min-p leaves fewer than k tokens, so top-k removes nothing.
    return sampleCase(c, "sample min-p 0.3 k=500", 32000, .{ .min_p = 0.3, .top_k = 500, .seed = 8 }, 8);
}

fn checkSampleVocab(c: *Ctx) !Result {
    return sampleCase(c, "sample Qwen vocab", 248320, .{
        .temperature = 0.8,
        .top_k = 20,
        .top_p = 0.95,
        .min_p = 0.01,
        .repetition_penalty = 1.05,
        .presence_penalty = 0.3,
        .seed = 6,
    }, 3);
}
