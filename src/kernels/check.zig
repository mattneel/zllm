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
        checkRmsNorm,         checkMatmul,            checkMatmulChunked, checkSiluMul,      checkAdd,
        checkEmbed,           checkQkNormRope,        checkKvAppendF16,   checkAttentionF32, checkAttentionF16,
        checkConvSilu,        checkL2Norm,            checkGates,         checkDeltaRule,    checkSampleGreedy,
        checkSampleTopKTopP,  checkSampleMinP,        checkSampleTopP,    checkSamplePlain,  checkSampleTopK1,
        checkSampleLargeTopK, checkSampleTopKSkipped, checkSampleVocab,
    }) |f| {
        const r = try f(&ctx);
        const pass = r.err <= r.tolerance;
        ok = ok and pass;
        const verdict = if (pass) "PASS" else "FAIL";
        if (r.trials > 0) {
            try out.print("{s} {s:<24} max draw deviation {e:.2} (tolerance {e:.0}); {d}/{d} draws exact\n", .{ verdict, r.name, r.err, r.tolerance, r.exact, r.trials });
        } else {
            try out.print("{s} {s:<24} max rel err {e:.2} (tolerance {e:.0})\n", .{ verdict, r.name, r.err, r.tolerance });
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

fn matmulCase(c: *Ctx, name: []const u8, rows: u32, n: u32, k: u32, split: ?u32) !Result {
    const x = try c.random(rows * k, 1);
    const w = try c.randomBf16(n * k, 1);
    const out = try c.output(rows * n);
    const xa = try c.upload(x);
    const l = c.launcher();
    // Optionally split the weights into two chunks, as large matrices are.
    var col0: u32 = 0;
    for ([_]u32{ split orelse n, n - (split orelse n) }) |cols| {
        if (cols == 0) continue;
        const wa = try c.upload(w.words[col0 * k / 2 ..][0 .. cols * k / 2]);
        try l.launch(.matmul_bf16, P.MatmulBf16{ .rows = rows, .n = cols, .k = k, .col0 = col0, .out_stride = n }, &.{ xa, wa, out }, .{
            std.math.divCeil(u32, cols, kernels.matmul_tile_n) catch unreachable,
            std.math.divCeil(u32, rows, kernels.matmul_tile_t) catch unreachable,
            1,
        });
        col0 += cols;
    }
    const got = try c.read(out);

    const want = try c.arena.alloc(f64, rows * n);
    for (0..rows) |t| for (0..n) |j| {
        var s: f64 = 0;
        for (0..k) |i| s += @as(f64, x[t * k + i]) * w.values[j * k + i];
        want[t * n + j] = s;
    };
    return .{ .name = name, .err = relErr(got, want), .tolerance = 1e-5 };
}

fn checkMatmul(c: *Ctx) !Result {
    // 19 tokens and 100 outputs leave partial tiles in both dimensions.
    return matmulCase(c, "matmul_bf16", 19, 100, 1024, null);
}

fn checkMatmulChunked(c: *Ctx) !Result {
    return matmulCase(c, "matmul_bf16 (2 chunks)", 33, 150, 96, 70);
}

fn checkSiluMul(c: *Ctx) !Result {
    const n = 70_001;
    const gate = try c.random(n, 6);
    const up = try c.random(n, 2);
    const ga = try c.upload(gate);
    try c.launcher().launch(.silu_mul, P.Count{ .count = n }, &.{ ga, try c.upload(up) }, kernels.Launcher.elementwise(n));
    const got = try c.read(ga);
    const want = try c.arena.alloc(f64, n);
    for (want, gate, up) |*o, g, u| o.* = @as(f64, g) / (1 + @exp(-@as(f64, g))) * u;
    return .{ .name = "silu_mul", .err = relErr(got, want), .tolerance = 1e-6 };
}

fn checkAdd(c: *Ctx) !Result {
    const n = 4099;
    const x = try c.random(n, 1);
    const y = try c.random(n, 1);
    const xa = try c.upload(x);
    try c.launcher().launch(.add, P.Count{ .count = n }, &.{ xa, try c.upload(y) }, kernels.Launcher.elementwise(n));
    const got = try c.read(xa);
    const want = try c.arena.alloc(f64, n);
    for (want, x, y) |*o, a, b| o.* = @as(f32, a + b);
    return .{ .name = "add", .err = relErr(got, want), .tolerance = 0 };
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

fn checkQkNormRope(c: *Ctx) !Result {
    // Q layout: 3 heads of 32 interleaved with gates (head stride 64), 5 tokens,
    // rotation over the first 16 dims, positions starting at 7.
    const tokens = 5;
    const heads = 3;
    const hd = 32;
    const n_rot = 16;
    const pos0 = 7;
    const eps = 1e-6;
    const base = 1e7;
    const row = heads * 2 * hd;
    const x = try c.random(tokens * row, 2);
    const w = try c.random(hd, 1);
    const table = try kernels.ropeTable(c.arena, pos0 + tokens, n_rot, base);
    const xa = try c.upload(x);
    try c.launcher().launch(.qk_norm_rope, P.QkNormRope{
        .heads = heads,
        .head_dim = hd,
        .n_rot = n_rot,
        .row_stride = row,
        .head_stride = 2 * hd,
        .eps = eps,
    }, &.{ xa, try c.upload(w), try c.upload(table), try c.step(tokens, pos0, 0) }, .{ heads, tokens, 1 });
    const got = try c.read(xa);

    const want = try c.arena.alloc(f64, x.len);
    for (want, x) |*o, v| o.* = v; // gates and other heads' halves pass through
    for (0..tokens) |t| for (0..heads) |h| {
        const head = want[t * row + h * 2 * hd ..][0..hd];
        var ss: f64 = 0;
        for (head) |v| ss += v * v;
        const scale = 1 / @sqrt(ss / hd + eps);
        for (head, w) |*v, wi| v.* = v.* * scale * wi;
        const pos: f64 = @floatFromInt(pos0 + t);
        for (0..n_rot / 2) |i| {
            const theta = pos * std.math.pow(f64, base, -2.0 * @as(f64, @floatFromInt(i)) / n_rot);
            const x0 = head[i];
            const x1 = head[i + n_rot / 2];
            head[i] = x0 * @cos(theta) - x1 * @sin(theta);
            head[i + n_rot / 2] = x0 * @sin(theta) + x1 * @cos(theta);
        }
    };
    // The table is f32 from the reference formula, so angles carry f32 error.
    return .{ .name = "qk_norm_rope", .err = relErr(got, want), .tolerance = 1e-5 };
}

/// Two steps of kv_append, then attention for the second step's queries,
/// checked against the values the cache actually holds.
fn attentionCase(c: *Ctx, name: []const u8, half: bool) !Result {
    // Qwen 3.5 shape: 8 query heads over 2 KV heads, head_dim 256. 287 keys
    // span two of the kernel's 256-key tiles.
    const first = 270;
    const second = 17;
    const total = first + second;
    const heads = 8;
    const kv = 2;
    const hd = 256;
    const width = kv * hd;
    const scale: f32 = 1.0 / 16.0;
    const k = try c.random(total * width, 1);
    const v = try c.random(total * width, 1);
    const qg = try c.random(second * heads * 2 * hd, 1);
    const words: u32 = if (half) width / 2 else width;
    const kc = try c.output(total * words);
    const vc = try c.output(total * words);
    const out = try c.output(second * heads * hd);
    const l = c.launcher();
    const append: P.KvAppend = .{ .width = width, .f16 = @intFromBool(half) };
    const step1 = try c.step(first, 0, 0);
    try l.launch(.kv_append, append, &.{ try c.upload(k[0 .. first * width]), try c.upload(v[0 .. first * width]), kc, vc, step1 }, kernels.Launcher.elementwise(first * width / 2));
    const step2 = try c.step(second, first, 0);
    try l.launch(.kv_append, append, &.{ try c.upload(k[first * width ..]), try c.upload(v[first * width ..]), kc, vc, step2 }, kernels.Launcher.elementwise(second * width / 2));
    try l.launch(.attention, P.Attention{ .heads = heads, .kv_heads = kv, .head_dim = hd, .f16 = @intFromBool(half), .scale = scale }, &.{
        try c.upload(qg), kc, vc, out, step2,
    }, .{ heads, second, 1 });
    const got = try c.read(out);
    const ck = try decodeCache(c, try c.readWords(kc), half);
    const cv = try decodeCache(c, try c.readWords(vc), half);

    const want = try c.arena.alloc(f64, second * heads * hd);
    const s = try c.arena.alloc(f64, total);
    for (0..second) |t| for (0..heads) |h| {
        const kvh = h / (heads / kv);
        const keys = first + t + 1;
        const q = qg[t * heads * 2 * hd + h * 2 * hd ..][0..hd];
        var max: f64 = -std.math.inf(f64);
        for (0..keys) |j| {
            var dot: f64 = 0;
            for (q, ck[j * width + kvh * hd ..][0..hd]) |a, b| dot += @as(f64, a) * b;
            s[j] = dot * scale;
            max = @max(max, s[j]);
        }
        var sum: f64 = 0;
        for (s[0..keys]) |*e| {
            e.* = @exp(e.* - max);
            sum += e.*;
        }
        for (0..hd) |d| {
            var o: f64 = 0;
            for (0..keys) |j| o += s[j] * cv[j * width + kvh * hd + d];
            want[(t * heads + h) * hd + d] = o / sum * sigmoid64(qg[t * heads * 2 * hd + h * 2 * hd + hd + d]);
        }
    };
    return .{ .name = name, .err = relErr(got, want), .tolerance = 1e-5 };
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

fn checkAttentionF32(c: *Ctx) !Result {
    return attentionCase(c, "attention (f32 cache)", false);
}

fn checkAttentionF16(c: *Ctx) !Result {
    return attentionCase(c, "attention (f16 cache)", true);
}

/// kv_append's f16 rounding: within one f16 step (2^-10 relative) of the input.
fn checkKvAppendF16(c: *Ctx) !Result {
    const tokens = 5;
    const pos0 = 3;
    const width = 64;
    const k = try c.random(tokens * width, 1);
    const v = try c.random(tokens * width, 1);
    const kc = try c.output((pos0 + tokens) * width / 2);
    const vc = try c.output((pos0 + tokens) * width / 2);
    try c.launcher().launch(.kv_append, P.KvAppend{ .width = width, .f16 = 1 }, &.{
        try c.upload(k), try c.upload(v), kc, vc, try c.step(tokens, pos0, 0),
    }, kernels.Launcher.elementwise(tokens * width / 2));
    const ck = try decodeCache(c, try c.readWords(kc), true);
    const cv = try decodeCache(c, try c.readWords(vc), true);
    const got = try c.arena.alloc(f32, 2 * tokens * width);
    @memcpy(got[0 .. tokens * width], ck[pos0 * width ..]);
    @memcpy(got[tokens * width ..], cv[pos0 * width ..]);
    const want = try c.arena.alloc(f64, 2 * tokens * width);
    for (want[0 .. tokens * width], k) |*o, x| o.* = x;
    for (want[tokens * width ..], v) |*o, x| o.* = x;
    // Positions before pos0 stay untouched (zero).
    for (ck[0 .. pos0 * width]) |x| if (x != 0) return .{ .name = "kv_append (f16)", .err = 1, .tolerance = 1e-3 };
    return .{ .name = "kv_append (f16)", .err = relErr(got, want), .tolerance = 1e-3 };
}

fn checkConvSilu(c: *Ctx) !Result {
    // Two steps of 5 and 4 tokens: the second reads the history the first left.
    const steps = [_]u32{ 5, 4 };
    const tokens = 9;
    const channels = 300;
    const taps = 4;
    const x = try c.random(tokens * channels, 2);
    const w = try c.random(channels * taps, 1);
    const hist = try c.output((taps - 1) * channels);
    const wa = try c.upload(w);
    const got = try c.arena.alloc(f32, tokens * channels);
    var t0: u32 = 0;
    for (steps) |m| {
        const out = try c.output(m * channels);
        try c.launcher().launch(.conv_silu, P.ConvSilu{ .channels = channels, .taps = taps }, &.{
            try c.upload(x[t0 * channels ..][0 .. m * channels]), wa, hist, out, try c.step(m, t0, 0),
        }, kernels.Launcher.elementwise(channels));
        @memcpy(got[t0 * channels ..][0 .. m * channels], try c.read(out));
        t0 += m;
    }

    const want = try c.arena.alloc(f64, tokens * channels);
    for (0..tokens) |t| for (0..channels) |ch| {
        var sum: f64 = 0;
        for (0..taps) |tap| {
            const back = taps - 1 - tap;
            if (t >= back) sum += @as(f64, w[ch * taps + tap]) * x[(t - back) * channels + ch];
        }
        want[t * channels + ch] = sum * sigmoid64(sum);
    };
    return .{ .name = "conv_silu (2 steps)", .err = relErr(got, want), .tolerance = 1e-6 };
}

fn checkL2Norm(c: *Ctx) !Result {
    // Normalize the first 4 heads of 128 in rows of 640 (a trailing region
    // stands in for the value heads and must stay untouched).
    const tokens = 6;
    const heads = 4;
    const dim = 128;
    const row = 640;
    const eps = 1e-6;
    const x = try c.random(tokens * row, 3);
    const xa = try c.upload(x);
    try c.launcher().launch(.l2norm_heads, P.L2NormHeads{ .heads = heads, .dim = dim, .row_stride = row, .eps = eps }, &.{xa}, .{ heads, tokens, 1 });
    const got = try c.read(xa);

    const want = try c.arena.alloc(f64, x.len);
    for (want, x) |*o, v| o.* = v;
    for (0..tokens) |t| for (0..heads) |h| {
        const head = want[t * row + h * dim ..][0..dim];
        var ss: f64 = 0;
        for (head) |v| ss += v * v;
        for (head) |*v| v.* /= @sqrt(ss + eps);
    };
    return .{ .name = "l2norm_heads", .err = relErr(got, want), .tolerance = 1e-6 };
}

fn checkGates(c: *Ctx) !Result {
    // Inputs span softplus's linear cut at 20.
    const tokens = 11;
    const heads = 16;
    const n = tokens * heads;
    const beta = try c.random(n, 6);
    const g = try c.random(n, 25);
    const dt = try c.random(heads, 1);
    const a = try c.random(heads, 1);
    for (a) |*v| v.* = -@exp(v.*);
    const ba = try c.upload(beta);
    const ga = try c.upload(g);
    try c.launcher().launch(.gates, P.Gates{ .count = n, .heads = heads }, &.{ ba, ga, try c.upload(dt), try c.upload(a) }, kernels.Launcher.elementwise(n));
    const got_beta = try c.read(ba);
    const got_g = try c.read(ga);

    const want = try c.arena.alloc(f64, 2 * n);
    const got = try c.arena.alloc(f32, 2 * n);
    for (0..n) |i| {
        want[i] = sigmoid64(beta[i]);
        const xv = @as(f64, g[i]) + dt[i % heads];
        want[n + i] = (if (xv > 20) xv else @log(1 + @exp(xv))) * a[i % heads];
    }
    @memcpy(got[0..n], got_beta);
    @memcpy(got[n..], got_g);
    return .{ .name = "gates", .err = relErr(got, want), .tolerance = 1e-6 };
}

fn checkDeltaRule(c: *Ctx) !Result {
    // 4 value heads over 2 key heads (tiled mapping), dim 128, 37 tokens in
    // two steps of 20 and 17 that carry the state between them.
    const tokens = 37;
    const hk = 2;
    const hv = 4;
    const d = 128;
    const eps = 1e-6;
    const key_dim = hk * d;
    const inner = hv * d;
    const stride = 2 * key_dim + inner;
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
    const z = try c.random(tokens * inner, 2);
    const norm_w = try c.random(d, 1);
    const state = try c.output(hv * d * d);
    const wn = try c.upload(norm_w);
    const got = try c.arena.alloc(f32, tokens * inner);
    var t0: u32 = 0;
    for ([_]u32{ 20, 17 }) |m| {
        const out = try c.output(m * inner);
        try c.launcher().launch(.delta_rule, P.DeltaRule{
            .k_heads = hk,
            .v_heads = hv,
            .dim = d,
            .conv_stride = stride,
            .eps = eps,
        }, &.{
            try c.upload(conv[t0 * stride ..][0 .. m * stride]), try c.upload(g[t0 * hv ..][0 .. m * hv]),
            try c.upload(beta[t0 * hv ..][0 .. m * hv]),         try c.upload(z[t0 * inner ..][0 .. m * inner]),
            wn,                                                  state,
            out,                                                 try c.step(m, t0, 0),
        }, .{ hv, 1, 1 });
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
                const zz: f64 = z[t * inner + j * d + b];
                want[t * inner + j * d + b] = o[b] * r * norm_w[b] * zz * sigmoid64(zz);
            }
        }
    }
    return .{ .name = "delta_rule (2 steps)", .err = relErr(got, want), .tolerance = 1e-5 };
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
