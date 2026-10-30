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
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    var ctx: Ctx = .{ .gpa = gpa, .host = host, .rec = .{}, .arena = arena_state.allocator(), .rand = prng.random() };
    defer ctx.rec.deinit(gpa);
    ctx.set = try kernels.Set.create(gpa, ctx.arena, &ctx.rec);
    try ctx.flush();

    var ok = true;
    inline for (.{
        checkRmsNorm,        checkMatmul,         checkMatmulChunked,   checkMatvec,            checkMatvecNorm,
        checkMatvecSiluPair, checkMatmulF16,      checkMatvecNormF16,   checkMatvecSiluPairF16, checkEmbedF16,
        checkSiluMul,        checkEmbed,          checkRopeAppendF32,   checkRopeAppendF16,     checkRopeAppendRounding,
        checkAttentionF32,   checkAttentionF16,   checkAttentionWide,   checkAttnDecode,        checkAttnDecodeWide,
        checkDeltaPrep,      checkDeltaKeep,      checkDeltaRule,       checkGatedNormSigmoid,  checkDraft,
        checkGyroMatvec,     checkMoeRoute,       checkRotate128,       checkMoeCombine,        checkHc,
        checkPle,            checkSampleGreedy,   checkSampleTopKTopP,  checkSampleMinP,        checkSampleTopP,
        checkSamplePlain,    checkSampleTopK1,    checkSampleLargeTopK, checkSampleTopKSkipped, checkSampleVocab,
        Quant(.q8_0).matvec, Quant(.q8_0).matmul, Quant(.q8_0).embed,   checkEmbedStride,       Quant(.q3_k).matvec,
        Quant(.q3_k).matmul, Quant(.q3_k).embed,  Quant(.q4_k).matvec,  Quant(.q4_k).matmul,    Quant(.q4_k).embed,
        Quant(.q5_k).matvec, Quant(.q5_k).matmul, Quant(.q5_k).embed,   Quant(.q6_k).matvec,    Quant(.q6_k).matmul,
        Quant(.q6_k).embed,  checkQuantTiles,     checkQuantSiluPair,   checkQuantizeQ8K,       checkQuantizeQ80,
        checkQuantizeNorm,   checkQuantizeSilu,
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
    /// n random half-precision values of type `t` (BF16 or F16), packed two
    /// per word, and the values they hold.
    fn randomHalf(c: *Ctx, t: gguf.TensorType, n: usize, scale: f32) !struct { words: []u32, values: []f32 } {
        std.debug.assert(n % 2 == 0);
        const words = try c.arena.alloc(u32, n / 2);
        const values = try c.arena.alloc(f32, n);
        for (values, 0..) |*v, i| {
            const f = (c.rand.float(f32) * 2 - 1) * scale;
            const bits: u16 = switch (t) {
                .bf16 => @truncate(@as(u32, @bitCast(f)) >> 16),
                .f16 => @bitCast(@as(f16, @floatCast(f))),
                else => unreachable,
            };
            v.* = switch (t) {
                .bf16 => @bitCast(@as(u32, bits) << 16),
                else => @as(f16, @bitCast(bits)),
            };
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
/// The larger error; a NaN is infinitely large (@max would pass it by).
fn worse(a: f64, b: f64) f64 {
    return if (std.math.isNan(b)) std.math.inf(f64) else @max(a, b);
}

/// Max |got − want| over max |want|; infinite if any output is not finite.
fn relErr(got: []const f32, want: []const f64) f64 {
    std.debug.assert(got.len == want.len);
    var max_err: f64 = 0;
    var scale: f64 = 0;
    for (got, want) |g, w| {
        if (!std.math.isFinite(g)) return std.math.inf(f64);
        max_err = @max(max_err, @abs(@as(f64, g) - w));
        scale = @max(scale, @abs(w));
    }
    return max_err / @max(scale, 1e-30);
}

// A NaN output must fail a check, not vanish into @max.
test relErr {
    const nan = std.math.nan(f32);
    try std.testing.expect(std.math.isInf(relErr(&.{ 1, nan }, &.{ 1, 2 })));
    try std.testing.expectEqual(0.5, relErr(&.{ 1, 1 }, &.{ 1, 2 }));
    try std.testing.expect(std.math.isInf(worse(0, std.math.nan(f64))));
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
fn matmulCase(c: *Ctx, comptime wt: gguf.TensorType, name: []const u8, rows: u32, n: u32, k: u32, split: ?u32, accumulate: bool) !Result {
    const x = try c.random(rows * k, 1);
    const w = try c.randomHalf(wt, n * k, 1);
    const init = try c.random(rows * n, if (accumulate) 4 else 0);
    const out = try c.upload(init);
    const xa = try c.upload(x);
    const l = c.launcher();
    // Optionally split the weights into two chunks, as large matrices are.
    var col0: u32 = 0;
    for ([_]u32{ split orelse n, n - (split orelse n) }) |cols| {
        if (cols == 0) continue;
        const wa = try c.upload(w.words[col0 * k / 2 ..][0 .. cols * k / 2]);
        try l.launch(comptime kernels.Kernel.weight(.matmul, wt).?, P.MatmulBf16{ .rows = rows, .n = cols, .k = k, .col0 = col0, .out_stride = n, .accumulate = @intFromBool(accumulate) }, &.{ xa, wa, out }, .{
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
    return matmulCase(c, .bf16, "matmul_bf16", 19, 100, 1024, null, false);
}

fn checkMatmulF16(c: *Ctx) !Result {
    return matmulCase(c, .f16, "matmul_f16", 19, 100, 1024, null, false);
}

fn checkMatmulChunked(c: *Ctx) !Result {
    return matmulCase(c, .bf16, "matmul_bf16 (2 chunks, +=)", 33, 150, 96, 70, true);
}

fn checkMatvec(c: *Ctx) !Result {
    // 3 tokens, 100 outputs (a partial workgroup of rows), split into two
    // chunks; k = 4000 is too wide to stage and leaves a partial pass over
    // the 32 lanes.
    return matvecCase(c, .bf16, "matvec_bf16 (2 chunks)", 4000, 37, false);
}

fn checkMatvecNorm(c: *Ctx) !Result {
    return matvecCase(c, .bf16, "matvec_bf16 norm+= (2 chunks)", 1000, 37, true);
}

fn checkMatvecNormF16(c: *Ctx) !Result {
    return matvecCase(c, .f16, "matvec_f16 norm+= (2 chunks)", 1000, 37, true);
}

/// out (+)= rmsnorm(x) · w for 3 tokens and 100 outputs in two chunks; with
/// `fused`, the RMSNorm prologue and accumulate epilogue are on.
fn matvecCase(c: *Ctx, comptime wt: gguf.TensorType, name: []const u8, k: u32, split: u32, fused: bool) !Result {
    const rows = 3;
    const n = 100;
    const eps = 1e-6;
    const x = try c.random(rows * k, 3);
    const nw = try c.random(k, 1);
    const w = try c.randomHalf(wt, n * k, 1);
    const init = try c.random(rows * n, if (fused) 4 else 0);
    const out = try c.upload(init);
    const xa = try c.upload(x);
    const na = try c.upload(nw);
    var col0: u32 = 0;
    for ([_]u32{ split, n - split }) |cols| {
        const wa = try c.upload(w.words[col0 * k / 2 ..][0 .. cols * k / 2]);
        try c.launcher().launch(comptime kernels.Kernel.weight(.matvec, wt).?, P.MatvecBf16{
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
    return siluPairCase(c, .bf16, "matvec_bf16 norm+silu pair");
}

fn checkMatvecSiluPairF16(c: *Ctx) !Result {
    return siluPairCase(c, .f16, "matvec_f16 norm+silu pair");
}

fn siluPairCase(c: *Ctx, comptime wt: gguf.TensorType, name: []const u8) !Result {
    const rows = 3;
    const ff = 100;
    const k = 1024;
    const eps = 1e-6;
    const x = try c.random(rows * k, 3);
    const nw = try c.random(k, 1);
    const w = try c.randomHalf(wt, 2 * ff * k, 1);
    const out = try c.output(rows * ff);
    try c.launcher().launch(comptime kernels.Kernel.weight(.matvec, wt).?, P.MatvecBf16{
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
    return .{ .name = name, .err = relErr(got, want), .tolerance = 1e-5 };
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
/// read: as they are, or padded or repacked by Launcher.layout, as the
/// model loader does.
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
    try c.launcher().layout(t, src, out, blocks);
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
        worst = worse(worst, @abs(@as(f64, got) - v) / @max(@abs(@as(f64, v)), 1e-30));
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
            worst = worse(worst, @abs(@as(f64, @floatFromInt(q)) * d[j / 256] - want) / (step / 2));
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
            return .{ .name = "embed " ++ @tagName(t) ++ " (2 chunks)", .err = relErr(got, want), .tolerance = 1e-6 };
        }
    };
}

/// Q8_0 rows of 160 values (170 bytes) stored 172 bytes apart, with 2 bytes
/// of junk after each, as the Gyro-Coder's PLE rows arrive from the file.
fn checkEmbedStride(c: *Ctx) !Result {
    const rows = 16;
    const dim = 160;
    const row_bytes = 170;
    const stride = 172;
    const table = try quantRows(c, .q8_0, rows, dim);
    const bytes = try c.arena.alloc(u8, rows * stride);
    c.rand.bytes(bytes);
    for (0..rows) |r| @memcpy(bytes[r * stride ..][0..row_bytes], table.bytes[r * row_bytes ..][0..row_bytes]);
    const tokens = [_]u32{ 0, 15, 7, 8, 1 };
    const out = try c.output(tokens.len * dim);
    try c.launcher().launch(.embed_q8_0, P.EmbedBf16{ .dim = dim, .row0 = 0, .chunk_rows = rows, .stride = stride }, &.{ try c.upload(&tokens), try c.upload(bytes), out }, .{ tokens.len, 1, 1 });
    const got = try c.read(out);
    const want = try c.arena.alloc(f64, tokens.len * dim);
    for (tokens, 0..) |tok, i| for (0..dim) |e| {
        want[i * dim + e] = table.values[tok * dim + e];
    };
    return .{ .name = "embed q8_0, rows 172 bytes apart", .err = relErr(got, want), .tolerance = 1e-6 };
}

/// The decode multiply on quantized weights: 3 tokens, 100 outputs (a
/// partial workgroup of rows), norm prologue and accumulate epilogue; or,
/// with `pair`, the SiLU pair over stacked gate and up rows. Every variant
/// tuning may pick (MatvecVariant.candidates); the error is the worst.
fn quantMatvec(c: *Ctx, comptime t: gguf.TensorType, name: []const u8, k: u32, pair: bool) !Result {
    const rows = 3;
    const n = 100;
    const eps = 1e-6;
    const x = try c.random(rows * k, 3);
    const nw = try c.random(k, 1);
    const w = try quantRows(c, t, if (pair) 2 * n else n, k);
    const init = try c.random(rows * n, if (pair) 0 else 4);
    const kernel = kernels.Kernel.quant(.matvec, t).?;
    const xa = try c.upload(x);
    const wa = try gpuWeights(c, t, w.bytes);
    const na = try c.upload(nw);
    var got: [kernels.MatvecVariant.candidates.len][]f32 = undefined;
    for (kernels.MatvecVariant.candidates, &got) |v, *g| {
        const out = try c.upload(init);
        try c.launcher().launchPipeline(try c.set.matvec(c.gpa, c.arena, &c.rec, kernel, v), P.MatvecBf16{
            .n = n,
            .k = k,
            .col0 = 0,
            .out_stride = n,
            .eps = eps,
            .flags = .{ .norm = true, .accumulate = !pair, .silu_pair = pair },
            .half = if (pair) n else 0,
        }, &.{ xa, wa, out, na }, .{ std.math.divCeil(u32, n, v.groupRows()) catch unreachable, rows, 1 });
        g.* = try c.read(out);
    }
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
    var worst: f64 = 0;
    for (got) |g| worst = @max(worst, relErr(g, want));
    return .{ .name = name, .err = worst, .tolerance = 1e-5 };
}

fn checkQuantTiles(c: *Ctx) !Result {
    // Two staged tiles of x, as the 27B's 5,120-wide inputs take three.
    return quantMatvec(c, .q4_k, "matvec q4_k norm+= (2 tiles)", 4096, false);
}

fn checkQuantSiluPair(c: *Ctx) !Result {
    return quantMatvec(c, .q3_k, "matvec q3_k norm+silu pair", 1024, true);
}

fn checkEmbed(c: *Ctx) !Result {
    return embedCase(c, .bf16, "embed_bf16 (2 chunks)");
}

fn checkEmbedF16(c: *Ctx) !Result {
    return embedCase(c, .f16, "embed_f16 (2 chunks)");
}

fn embedCase(c: *Ctx, comptime wt: gguf.TensorType, name: []const u8) !Result {
    const vocab = 50;
    const dim = 64;
    const split = 30;
    const table = try c.randomHalf(wt, vocab * dim, 1);
    const tokens = [_]u32{ 0, 29, 30, 49, 7, 30, 12 };
    const out = try c.output(tokens.len * dim);
    const ta = try c.upload(&tokens);
    const l = c.launcher();
    for ([_][2]u32{ .{ 0, split }, .{ split, vocab - split } }) |chunk| {
        const words = table.words[chunk[0] * dim / 2 ..][0 .. chunk[1] * dim / 2];
        try l.launch(comptime kernels.Kernel.weight(.embed, wt).?, P.EmbedBf16{ .dim = dim, .row0 = chunk[0], .chunk_rows = chunk[1] }, &.{ ta, try c.upload(words), out }, .{ tokens.len, 1, 1 });
    }
    const got = try c.read(out);
    const want = try c.arena.alloc(f64, tokens.len * dim);
    for (tokens, 0..) |tok, t| for (0..dim) |i| {
        want[t * dim + i] = table.values[tok * dim + i];
    };
    return .{ .name = name, .err = relErr(got, want), .tolerance = 0 };
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
    // error; the f16 cache rounds to nearest, half a step: at most 2^-11
    // relative.
    return .{ .name = name, .err = relErr(got, want), .tolerance = if (half) 5e-4 else 1e-5 };
}

fn checkRopeAppendF32(c: *Ctx) !Result {
    return ropeAppendCase(c, "rope_append (f32 cache)", false);
}

fn checkRopeAppendF16(c: *Ctx) !Result {
    return ropeAppendCase(c, "rope_append (f16 cache)", true);
}

/// The f16 cache must hold the f32 cache's values rounded to nearest, ties
/// to even, as llama.cpp's cache does (the reference); WGSL lets
/// pack2x16float round either way. Both caches from one input, whose V
/// (copied as it is) holds ties, subnormals, overflow and non-finite values
/// and random values over exponents −28 to 17; err is the fraction of the
/// f16 cache's values that differ.
fn checkRopeAppendRounding(c: *Ctx) !Result {
    const tokens = 64;
    const kv = 2;
    const hd = 256;
    const width = kv * hd;
    const row = 2 * width + 2 * hd;
    const x = try c.random(tokens * row, 2);
    const inf = std.math.inf(f32);
    const edges = [_]f32{
        1 + 0x1p-11,       1 + 3 * 0x1p-11, -(1 + 0x1p-11), 0x1p-25,  3 * 0x1p-25, 0x1p-24, 0x1.8p-24, 0x1p-14 - 0x1p-25,
        0x1p-14 - 0x1p-24, 65504,           65519,          65520,    65536,       inf,     -inf,      std.math.nan(f32),
        -0.0,              0,               0x1p-30,        -0x1p-20,
    };
    for (0..tokens) |t| for (x[t * row + 2 * hd + width ..][0..width], 0..) |*v, j| {
        v.* = if (j < edges.len) edges[j] else std.math.ldexp(v.*, @as(i32, @intCast((j + t) % 46)) - 28);
    };
    const ones = try c.arena.alloc(f32, hd);
    @memset(ones, 1);
    const table = try kernels.ropeTable(c.arena, tokens, 64, 1e7);
    const params: P.RopeAppend = .{ .heads = 1, .kv_heads = kv, .head_dim = hd, .n_rot = 64, .row = row, .k_off = 2 * hd, .v_off = 2 * hd + width, .f16 = 0, .eps = 1e-6 };
    var words: [2][]const u32 = undefined;
    for (&words, [_]u32{ 0, 1 }) |*out, half| {
        var p = params;
        p.f16 = half;
        const n: u32 = if (half == 1) width / 2 else width;
        const kc = try c.output(tokens * n);
        const vc = try c.output(tokens * n);
        try c.launcher().launch(.rope_append, p, &.{ try c.upload(x), try c.upload(ones), try c.upload(ones), try c.upload(table), kc, vc, try c.step(tokens, 0, 0) }, .{ 1 + kv, tokens, 1 });
        const kw = try c.readWords(kc);
        const vw = try c.readWords(vc);
        out.* = try std.mem.concat(c.arena, u32, &.{ kw, vw });
    }
    const exact: []const f32 = @ptrCast(words[0]);
    var differ: usize = 0;
    for (words[1], 0..) |w, i| {
        for ([_]u16{ @truncate(w), @truncate(w >> 16) }, 0..) |got, j| {
            const value: f16 = @floatCast(exact[2 * i + j]);
            const want: u16 = @bitCast(value);
            const nans = std.math.isNan(value) and std.math.isNan(@as(f16, @bitCast(got)));
            differ += @intFromBool(got != want and !nans);
        }
    }
    return .{ .name = "rope_append f16 rounds to nearest", .err = @as(f64, @floatFromInt(differ)) / @as(f64, @floatFromInt(exact.len)), .tolerance = 0 };
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

/// An attention shape: `heads` query heads over 2 KV heads, head_dim 256,
/// queries in rows padded past their gates (q_row). Qwen 3.5's 0.8B has 8
/// heads; 24 (Qwen3.8-Flash-Next) takes attn_split_wide.
fn AttnShape(comptime n_heads: u32) type {
    return struct {
        const heads = n_heads;
        const kv = 2;
        const hd = 256;
        const width = kv * hd;
        const q_row = heads * 2 * hd + 96;
        const scale: f32 = 1.0 / 16.0;
        const split = kernels.attnSplit(heads / kv);
    };
}

/// Gated attention of `second` queries at positions first.. over the cache,
/// in float64.
fn attnWant(c: *Ctx, comptime S: type, qg: []const f32, k: []const f32, v: []const f32, first: u32, second: u32) ![]f64 {
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

/// Prefill attention for 70 queries after 270 cached positions, as a step
/// runs it: attn_split and attn_combine over chunks of attn_rows tokens (32,
/// 32, 6), each over 64-position blocks of the 340 keys, the last partial.
fn attentionCase(c: *Ctx, comptime S: type, name: []const u8, half: bool) !Result {
    const first = 270;
    const second = 70;
    const total = first + second;
    const k = try c.random(total * S.width, 1);
    const v = try c.random(total * S.width, 1);
    const qg = try c.random(second * S.q_row, 1);
    const out = try c.output(second * S.heads * S.hd);
    const max_blocks = std.math.divCeil(u32, total, kernels.attn_block) catch unreachable;
    const part = try c.output(kernels.attn_rows * S.heads * max_blocks * (S.hd + 2));
    const qa = try c.upload(qg);
    const ka = try c.upload(try cacheWords(c, k, half));
    const va = try c.upload(try cacheWords(c, v, half));
    const step = try c.step(second, first, 0);
    const l = c.launcher();
    var row0: u32 = 0;
    while (row0 < second) : (row0 += kernels.attn_rows) {
        const params: P.Attention = .{ .heads = S.heads, .kv_heads = S.kv, .head_dim = S.hd, .f16 = @intFromBool(half), .scale = S.scale, .max_blocks = max_blocks, .q_row = S.q_row, .row0 = row0 };
        const split: kernels.Grid = .{ .key_blocks = .{ .x = S.kv, .block = kernels.attn_block, .row0 = row0, .max_rows = kernels.attn_rows } };
        const combine: kernels.Grid = .{ .rows_y_from = .{ .x = S.heads, .row0 = row0, .max_rows = kernels.attn_rows } };
        const e: kernels.Extent = .{ .rows = second, .keys = total };
        try l.launch(S.split, params, &.{ qa, ka, va, part, step }, split.resolve(e));
        try l.launch(.attn_combine, params, &.{ qa, part, out, step }, combine.resolve(e));
    }
    const got = try c.read(out);
    return .{ .name = name, .err = relErr(got, try attnWant(c, S, qg, k, v, first, second)), .tolerance = 1e-5 };
}

fn checkAttentionF32(c: *Ctx) !Result {
    return attentionCase(c, AttnShape(8), "attention, 3 chunks (f32 cache)", false);
}

fn checkAttentionF16(c: *Ctx) !Result {
    return attentionCase(c, AttnShape(8), "attention, 3 chunks (f16 cache)", true);
}

fn checkAttentionWide(c: *Ctx) !Result {
    return attentionCase(c, AttnShape(24), "attention, 3 chunks, groups of 12", true);
}

/// Decode attention: 3 queries after 150 cached positions, through
/// attn_split (three 64-position blocks, the last partial) and attn_combine.
fn attnDecodeCase(c: *Ctx, comptime S: type, name: []const u8) !Result {
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
    const params: P.Attention = .{ .heads = S.heads, .kv_heads = S.kv, .head_dim = S.hd, .f16 = 1, .scale = S.scale, .max_blocks = max_blocks, .q_row = S.q_row };
    const qa = try c.upload(qg);
    const grid: kernels.Grid = .{ .key_blocks = .{ .x = S.kv, .block = kernels.attn_block } };
    try l.launch(S.split, params, &.{ qa, try c.upload(try cacheWords(c, k, true)), try c.upload(try cacheWords(c, v, true)), part, step }, grid.resolve(.{ .rows = second, .keys = total }));
    try l.launch(.attn_combine, params, &.{ qa, part, out, step }, .{ S.heads, second, 1 });
    const got = try c.read(out);
    return .{ .name = name, .err = relErr(got, try attnWant(c, S, qg, k, v, first, second)), .tolerance = 1e-5 };
}

fn checkAttnDecode(c: *Ctx) !Result {
    return attnDecodeCase(c, AttnShape(8), "attn_split + combine");
}

fn checkAttnDecodeWide(c: *Ctx) !Result {
    return attnDecodeCase(c, AttnShape(24), "attn_split + combine, groups of 12");
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
            conv, g, beta, state, try c.output(m * inner), step, pa, hist, try c.output(1),
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

/// delta_rule's kept state and conv history: a step of 4 tokens keeping
/// row 1 must leave in `kept` exactly (bitwise) the state and history that a
/// step of the first 2 of those tokens leaves, after the same first step of
/// 5 tokens. The history (3 slots) then mixes slots from before the step with
/// the step's own inputs.
fn checkDeltaKeep(c: *Ctx) !Result {
    const hk = 2;
    const hv = 4;
    const d = 128;
    const taps = 4;
    const key_dim = hk * d;
    const channels = 2 * key_dim + hv * d;
    const row = channels + 8;
    const first = 5;
    const tokens = first + 4;
    const conv = try c.random(tokens * channels, 1);
    for (0..tokens) |t| for (0..2 * hk) |h| {
        const head = conv[t * channels + h * d ..][0..d];
        var ss: f32 = 0;
        for (head) |v| ss += v * v;
        for (head) |*v| v.* /= @sqrt(ss);
    };
    const g = try c.random(tokens * hv, 1);
    for (g) |*v| v.* = -@abs(v.*);
    const beta = try c.random(tokens * hv, 1);
    for (beta) |*v| v.* = @abs(v.*);
    const proj = try c.random(tokens * row, 1);
    const state_floats = hv * d * d;
    const hist_floats = (taps - 1) * channels;
    const grid: [3]u32 = .{ hv, std.math.divCeil(u32, d, kernels.delta_cols) catch unreachable, 1 };

    // Runs the first step, then `m` tokens keeping `keep`; returns the state
    // and history buffers and the kept buffer.
    const Run = struct { state: kernels.Arg, hist: kernels.Arg, kept: kernels.Arg };
    var runs: [2]Run = undefined;
    for (&runs, [_]u32{ 4, 2 }, [_]u32{ 2, 0 }) |*r, m, keep| {
        r.* = .{ .state = try c.output(state_floats), .hist = try c.output(hist_floats), .kept = try c.output(state_floats + hist_floats) };
        var t0: u32 = 0;
        for ([_]u32{ first, m }, [_]u32{ 0, keep }) |n, k| {
            const l = c.launcher();
            try l.launch(.delta_rule, P.DeltaRule{
                .k_heads = hk,
                .v_heads = hv,
                .dim = d,
                .conv_stride = channels,
                .taps = taps,
                .row = row,
                .kept_hist = state_floats,
            }, &.{
                try c.upload(conv[t0 * channels ..][0 .. n * channels]), try c.upload(g[t0 * hv ..][0 .. n * hv]),
                try c.upload(beta[t0 * hv ..][0 .. n * hv]),             r.state,
                try c.output(n * hv * d),                                try c.uniform(P.Step{ .m = n, .pos0 = t0, .u = 0, .keep = k }),
                try c.upload(proj[t0 * row ..][0 .. n * row]),           r.hist,
                r.kept,
            }, grid);
            t0 += n;
        }
    }
    const kept = try c.readWords(runs[0].kept);
    const want_state = try c.readWords(runs[1].state);
    const want_hist = try c.readWords(runs[1].hist);
    const same = std.mem.eql(u32, kept[0..state_floats], want_state) and std.mem.eql(u32, kept[state_floats..], want_hist);
    return .{ .name = "delta_rule kept row (bitwise)", .err = if (same) 0 else 1, .tolerance = 0 };
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
            try c.output(1),
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

/// The draft: the first index of the maximum of the second of two rows of
/// the Qwen vocabulary, written to slot 1; the maximum is planted twice, so
/// the lower index must win.
fn checkDraft(c: *Ctx) !Result {
    const vocab = 248320;
    const logits = try c.random(2 * vocab, 4);
    const row = logits[vocab..];
    for (row) |*v| v.* = @min(v.*, 10);
    row[700] = 11;
    row[123456] = 11;
    logits[5] = 99; // the other row's maximum
    const out = try c.output(2);
    try c.launcher().launch(.draft, P.Draft{ .vocab = vocab, .row = vocab, .slot = 1 }, &.{ try c.upload(logits), out }, .{ 1, 1, 1 });
    const got = (try c.readWords(out))[1];
    return .{ .name = "draft (argmax, ties to the lower id)", .err = @floatFromInt(@intFromBool(got != 700)), .tolerance = 0 };
}

/// Routed-expert products on TQ2_T weights: 3 experts of 80 rows (a
/// partial row group), 2 tokens routed to 2 experts each. Gate and up
/// (2,560 inputs, both projections in one dispatch, x per token), then down
/// (640 inputs, x per slot), against the CPU reference's row decoder.
fn checkGyroMatvec(c: *Ctx) !Result {
    const q4 = @import("../ref/qwen4exp/kernels.zig");
    const book = try c.upload(@import("../ref/gyro/gyro.zig").codebook_bytes);
    const rows = 80;
    const n_expert = 3;
    const tokens = 2;
    const used = 2;
    const ids = [_]u32{ 2, 0, 1, 2 };
    var worst: f64 = 0;
    for ([_]u32{ 2560, 640 }, [_]u32{ 2, 1 }) |cols, projections| {
        var w: [2][]u8 = undefined;
        for (&w) |*bytes| {
            const n_blocks = n_expert * rows * cols / 128;
            bytes.* = try c.arena.alloc(u8, std.mem.alignForward(usize, n_blocks * 34, 4));
            c.rand.bytes(bytes.*);
            for (0..n_blocks) |b| std.mem.writeInt(u16, bytes.*[b * 34 ..][0..2], @bitCast(@as(f16, @floatCast((c.rand.float(f32) * 2 - 1) * 0.05))), .little);
        }
        const per_slot = projections == 1;
        const x_rows: usize = if (per_slot) tokens * used else tokens;
        const x = try c.random(x_rows * cols, 1);
        const out = try c.output(tokens * used * 2 * rows);
        try c.launcher().launch(.gyro_matvec, P.GyroMatvec{
            .rows = rows,
            .cols = cols,
            .n_used = used,
            .x_per_slot = @intFromBool(per_slot),
            .x_stride = cols,
            .out_stride = 2 * rows,
            .out_z = rows,
        }, &.{ try c.upload(w[0]), try c.upload(w[1]), try c.upload(x), out, try c.upload(&ids), book }, .{ std.math.divCeil(u32, rows, 8) catch unreachable, tokens * used, projections });
        const got = try c.read(out);
        const want = try c.arena.alloc(f64, got.len);
        @memset(want, 0);
        const row_bytes = cols / 128 * 34;
        for (0..tokens * used) |slot| for (0..projections) |z| for (0..rows) |j| {
            const xr = x[(if (per_slot) slot else slot / used) * cols ..][0..cols];
            const off = (ids[slot] * rows + j) * row_bytes;
            want[slot * 2 * rows + z * rows + j] = q4.dotRow(.tq2_t, w[z][off..][0..row_bytes], xr);
        };
        // Unwritten up rows of the down pass stay 0 on both sides.
        worst = @max(worst, relErr(got, want));
    }
    return .{ .name = "gyro_matvec tq2_t (routed, gate+up, down)", .err = worst, .tolerance = 1e-5 };
}

/// Routing for 3 tokens over 256 experts, the model's top 10, through
/// moe_logits and moe_route: softmax of router · x, the 10 largest (ids
/// exact), their normalized weights, and the shared gate's logit.
fn checkMoeRoute(c: *Ctx) !Result {
    const n_expert = 256;
    const used = 10;
    const cols = 2560;
    const tokens = 3;
    const router = try c.random(n_expert * cols, 0.05);
    const shared = try c.random(cols, 0.05);
    const x = try c.random(tokens * cols, 1);
    const ids = try c.output(tokens * used);
    const weights = try c.output(tokens * used);
    const logit = try c.output(tokens);
    const logits = try c.output(tokens * (n_expert + 1));
    const params: P.MoeRoute = .{ .n_expert = n_expert, .n_used = used, .cols = cols, .x_stride = cols };
    const l = c.launcher();
    try l.launch(.moe_logits, params, &.{ try c.upload(router), try c.upload(shared), try c.upload(x), logits }, .{ (n_expert + 1 + 7) / 8, tokens, 1 });
    try l.launch(.moe_route, params, &.{ logits, ids, weights, logit }, .{ tokens, 1, 1 });
    const got_ids = try c.readWords(ids);
    const got_w = try c.read(weights);
    const got_l = try c.read(logit);
    const want_w = try c.arena.alloc(f64, tokens * used);
    const want_l = try c.arena.alloc(f64, tokens);
    var ids_ok = true;
    for (0..tokens) |t| {
        const xr = x[t * cols ..][0..cols];
        var probs: [n_expert]f64 = undefined;
        var m: f64 = -std.math.inf(f64);
        for (&probs, 0..) |*pr, e| {
            var s: f64 = 0;
            for (router[e * cols ..][0..cols], xr) |w, v| s += @as(f64, w) * v;
            pr.* = s;
            m = @max(m, s);
        }
        var sum: f64 = 0;
        for (&probs) |*pr| {
            pr.* = @exp(pr.* - m);
            sum += pr.*;
        }
        for (&probs) |*pr| pr.* /= sum;
        var total: f64 = 0;
        for (0..used) |k| {
            var best: usize = 0;
            for (probs, 0..) |pr, e| if (pr > probs[best]) {
                best = e;
            };
            ids_ok = ids_ok and got_ids[t * used + k] == best;
            want_w[t * used + k] = probs[best];
            total += probs[best];
            probs[best] = -1;
        }
        for (want_w[t * used ..][0..used]) |*w| w.* /= total;
        var s: f64 = 0;
        for (shared, xr) |w, v| s += @as(f64, w) * v;
        want_l[t] = s;
    }
    const err = @max(relErr(got_w, want_w), relErr(got_l, want_l));
    return .{ .name = "moe_route (256 experts, top 10, shared gate)", .err = if (ids_ok) err else 1, .tolerance = 1e-5 };
}

/// rotate128 against the reference rotation: rows of 2,560 as they are, and
/// rows of 640 from silu(gate) · up.
fn checkRotate128(c: *Ctx) !Result {
    const gyro_ref = @import("../ref/gyro/gyro.zig");
    var worst: f64 = 0;
    for ([_]u32{ 2560, 640 }, [_]u32{ 0, 1 }) |width, pair| {
        const rows = 3;
        const x_stride = width * (1 + pair);
        const x = try c.random(rows * x_stride, 2);
        const signs = try c.arena.alloc(f32, width);
        for (signs) |*s| s.* = if (c.rand.boolean()) 1 else -1;
        const y = try c.output(rows * width);
        try c.launcher().launch(.rotate128, P.Rotate128{ .width = width, .x_stride = x_stride, .y_stride = width, .silu_pair = pair }, &.{ try c.upload(x), try c.upload(signs), y }, .{ width / 128, rows, 1 });
        const got = try c.read(y);
        const want = try c.arena.alloc(f64, rows * width);
        const row = try c.arena.alloc(f32, width);
        for (0..rows) |r| {
            const xr = x[r * x_stride ..][0..x_stride];
            for (row, 0..) |*v, i| v.* = if (pair == 1) xr[i] / (1 + @exp(-xr[i])) * xr[width + i] else xr[i];
            gyro_ref.rotate(row, 128, signs);
            for (want[r * width ..][0..width], row) |*w, v| w.* = v;
        }
        worst = @max(worst, relErr(got, want));
    }
    return .{ .name = "rotate128 (plain, silu pair)", .err = worst, .tolerance = 1e-6 };
}

fn checkMoeCombine(c: *Ctx) !Result {
    const used = 10;
    const dim = 2560;
    const tokens = 2;
    const down = try c.random(tokens * used * dim, 1);
    const w = try c.random(tokens * used, 1);
    const shared = try c.random(tokens * dim, 1);
    const logit = try c.random(tokens, 3);
    const out = try c.output(tokens * dim);
    try c.launcher().launch(.moe_combine, P.MoeCombine{ .n_used = used, .dim = dim }, &.{ try c.upload(down), try c.upload(w), try c.upload(shared), try c.upload(logit), out }, .{ dim / 256, tokens, 1 });
    const got = try c.read(out);
    const want = try c.arena.alloc(f64, tokens * dim);
    for (0..tokens) |t| for (0..dim) |i| {
        var acc: f64 = 0;
        for (0..used) |k| acc += @as(f64, w[t * used + k]) * down[(t * used + k) * dim + i];
        want[t * dim + i] = acc + sigmoid64(logit[t]) * shared[t * dim + i];
    };
    return .{ .name = "moe_combine", .err = relErr(got, want), .tolerance = 1e-6 };
}

/// The hyper-connection mixer's own kernels for 3 tokens of 4 streams:
/// the streams' norms (rmsnorm with a weight row per stream), the low-rank
/// activation, the mix, and the combine into the residual streams.
fn checkHc(c: *Ctx) !Result {
    const hc = 4;
    const e = 256;
    const tokens = 3;
    const rank = 32;
    const eps = 1e-6;
    const res = try c.random(tokens * hc * e, 2);
    const wn = try c.random(hc * e, 1);
    const lo = try c.random(tokens * rank, 3);
    const gate = try c.random(tokens * hc * e, 3);
    const block = try c.random(tokens * e, 1);
    const inject = try c.random(tokens * hc, 4);
    const xn = try c.output(tokens * hc * e);
    const lo_a = try c.upload(lo);
    const mixed = try c.output(tokens * e);
    const res_a = try c.upload(res);
    const l = c.launcher();
    try l.launch(.rmsnorm, P.RmsNorm{ .rows = tokens * hc, .dim = e, .eps = eps, .w_rows = hc }, &.{ res_a, try c.upload(wn), xn }, .{ tokens * hc, 1, 1 });
    try l.launch(.hc_act, P.HcAct{ .count = tokens * rank, .hc = hc }, &.{lo_a}, .{ 1, 1, 1 });
    try l.launch(.hc_mix, P.Hc{ .dim = e, .hc = hc }, &.{ xn, try c.upload(gate), mixed }, .{ 1, tokens, 1 });
    try l.launch(.hc_combine, P.Hc{ .dim = e, .hc = hc }, &.{ try c.upload(block), try c.upload(inject), res_a }, .{ 1, tokens, 1 });
    const got = .{ try c.read(xn), try c.read(lo_a), try c.read(mixed), try c.read(res_a) };
    const want_xn = try c.arena.alloc(f64, tokens * hc * e);
    const want_lo = try c.arena.alloc(f64, tokens * rank);
    const want_mix = try c.arena.alloc(f64, tokens * e);
    const want_res = try c.arena.alloc(f64, tokens * hc * e);
    for (0..tokens * hc) |r| {
        var ss: f64 = 0;
        for (res[r * e ..][0..e]) |v| ss += @as(f64, v) * v;
        const s = 1 / @sqrt(ss / e + eps);
        for (0..e) |i| want_xn[r * e + i] = res[r * e + i] * s * wn[(r % hc) * e + i];
    }
    for (want_lo, lo) |*w, v| {
        const a = @as(f64, v) / hc;
        w.* = a * sigmoid64(a);
    }
    for (0..tokens) |t| for (0..e) |i| {
        var acc: f64 = 0;
        for (0..hc) |ch| acc += want_xn[(t * hc + ch) * e + i] * sigmoid64(gate[(t * hc + ch) * e + i]);
        want_mix[t * e + i] = acc / hc;
        for (0..hc) |ch| want_res[(t * hc + ch) * e + i] = res[(t * hc + ch) * e + i] + block[t * e + i] * 2 * sigmoid64(inject[t * hc + ch] / hc);
    };
    const err = @max(@max(relErr(got[0], want_xn), relErr(got[1], want_lo)), @max(relErr(got[2], want_mix), relErr(got[3], want_res)));
    return .{ .name = "hyper-connection norm, act, mix, combine", .err = err, .tolerance = 1e-6 };
}

/// The PLE block's gate and convolution for 8 tokens of 4 streams, in steps
/// of 5 and 3: the second step's taps reach into the history the first left.
fn checkPle(c: *Ctx) !Result {
    const hc = 4;
    const e = 128;
    const n = hc * e;
    const taps = 4;
    const dil = 3;
    const eps = 1e-6;
    const tokens = 8;
    const key = try c.random(tokens * n, 2);
    const hidden0 = try c.random(tokens * n, 2);
    const value = try c.random(tokens * e, 1);
    const wk = try c.random(n, 1);
    const wq = try c.random(n, 1);
    const wc = try c.random(n, 1);
    const conv = try c.random(n * taps, 1);
    const hist = try c.output((taps - 1) * dil * n);
    const got = try c.arena.alloc(f32, tokens * n);
    var t0: u32 = 0;
    for ([_]u32{ 5, 3 }) |m| {
        const gated = try c.output(m * n);
        const normed = try c.output(m * n);
        const hid = try c.upload(hidden0[t0 * n ..][0 .. m * n]);
        const l = c.launcher();
        try l.launch(.ple_mix, P.PleMix{ .dim = e, .hc = hc, .eps = eps }, &.{
            try c.upload(key[t0 * n ..][0 .. m * n]), hid,              try c.upload(value[t0 * e ..][0 .. m * e]),
            try c.upload(wk),                         try c.upload(wq), try c.upload(wc),
            gated,                                    normed,
        }, .{ hc, m, 1 });
        try l.launch(.ple_conv, P.PleConv{ .channels = n, .taps = taps, .dilation = dil }, &.{ gated, normed, try c.upload(conv), hist, hid, try c.step(m, t0, 0) }, .{ n / 256, 1, 1 });
        @memcpy(got[t0 * n ..][0 .. m * n], try c.read(hid));
        t0 += m;
    }
    const want = try c.arena.alloc(f64, tokens * n);
    const normed_all = try c.arena.alloc(f64, tokens * n);
    const gated_all = try c.arena.alloc(f64, tokens * n);
    for (0..tokens) |t| for (0..hc) |ch| {
        const seg = (t * hc + ch) * e;
        var sk: f64 = 0;
        var sh: f64 = 0;
        for (0..e) |i| {
            sk += @as(f64, key[seg + i]) * key[seg + i];
            sh += @as(f64, hidden0[seg + i]) * hidden0[seg + i];
        }
        sk = 1 / @sqrt(sk / e + eps);
        sh = 1 / @sqrt(sh / e + eps);
        var d: f64 = 0;
        for (0..e) |i| d += key[seg + i] * sk * wk[ch * e + i] * hidden0[seg + i] * sh * wq[ch * e + i];
        const s = d / @sqrt(@as(f64, e));
        const sgn: f64 = if (s > 0) 1 else if (s < 0) -1 else 0;
        const gate = sigmoid64(sgn * @sqrt(@max(@abs(s), 1e-6)));
        var sv: f64 = 0;
        for (0..e) |i| {
            const g = value[t * e + i] * gate;
            gated_all[seg + i] = g;
            sv += g * g;
        }
        const sg = 1 / @sqrt(sv / e + eps);
        for (0..e) |i| normed_all[seg + i] = gated_all[seg + i] * sg * wc[ch * e + i];
    };
    for (0..tokens) |t| for (0..n) |ch| {
        var acc: f64 = 0;
        for (0..taps) |k| {
            const back = (taps - 1 - k) * dil;
            if (t >= back) acc += conv[ch * taps + k] * normed_all[(t - back) * n + ch];
        }
        want[t * n + ch] = hidden0[t * n + ch] + gated_all[t * n + ch] + acc * sigmoid64(acc);
    };
    return .{ .name = "ple_mix + ple_conv (steps of 5 and 3)", .err = relErr(got, want), .tolerance = 1e-5 };
}

/// gated_norm's sigmoid gate (Qwen4Exp's DeltaNet): 3 tokens of 4 heads.
fn checkGatedNormSigmoid(c: *Ctx) !Result {
    const hv = 4;
    const d = 128;
    const tokens = 3;
    const eps = 1e-6;
    const z_off = 16;
    const z_row = z_off + hv * d + 8;
    const o = try c.random(tokens * hv * d, 2);
    const zp = try c.random(tokens * z_row, 3);
    const w = try c.random(d, 1);
    const oa = try c.upload(o);
    try c.launcher().launch(.gated_norm, P.GatedNorm{ .v_heads = hv, .dim = d, .eps = eps, .z_row = z_row, .z_off = z_off, .sigmoid = 1 }, &.{ oa, try c.upload(zp), try c.upload(w) }, .{ hv, tokens, 1 });
    const got = try c.read(oa);
    const want = try c.arena.alloc(f64, tokens * hv * d);
    for (0..tokens) |t| for (0..hv) |j| {
        const head = o[(t * hv + j) * d ..][0..d];
        var ss: f64 = 0;
        for (head) |v| ss += @as(f64, v) * v;
        const r = 1 / @sqrt(ss / d + eps);
        for (0..d) |b| want[(t * hv + j) * d + b] = head[b] * r * w[b] * sigmoid64(zp[t * z_row + z_off + j * d + b]);
    };
    return .{ .name = "gated_norm, sigmoid gate", .err = relErr(got, want), .tolerance = 1e-6 };
}

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
