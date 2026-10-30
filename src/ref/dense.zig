//! CPU f32 reference forward pass for the dense transformers (Qwen 3,
//! Llama, Granite; see graph/dense.zig for what differs between them).
//!
//! Teacher-forced like the Qwen 3.5 reference: one call runs a whole token
//! sequence from an empty state and reports each layer's output, the final
//! normed hidden state and the logits under llama.cpp's tensor names.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const graph = @import("../graph/dense.zig");
const tensor = @import("tensor.zig");
const ops = @import("ops.zig");
const mm = @import("matmul.zig");
const Matrix = mm.Matrix;
const Pool = @import("pool.zig").Pool;
const Loader = @import("loader.zig").Loader;
const qwen35 = @import("qwen35.zig");

pub const Hparams = graph.Hparams;
pub const Observer = qwen35.Model.Observer;

const Layer = struct {
    attn_norm: []f32,
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
    /// Per-head Q and K RMSNorm weights (Qwen 3).
    q_norm: ?[]f32,
    k_norm: ?[]f32,
    ffn_norm: []f32,
    ffn_gate: Matrix,
    ffn_up: Matrix,
    ffn_down: Matrix,
};

pub const Model = struct {
    hp: Hparams,
    /// Embeddings (row j = token j), also the LM head when tied.
    tok_embd: Matrix,
    /// Separate LM head, or null when tied to `tok_embd`.
    output: ?Matrix,
    output_norm: []f32,
    /// Per-pair RoPE frequency divisors (`rope_freqs.weight`), if any.
    rope_freqs: ?[]f32,
    layers: []Layer,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(m: *Model) void {
        m.arena.deinit();
        m.* = undefined;
    }

    pub const LoadError = Hparams.Error || tensor.Error || error{UnsupportedTensor};

    pub fn load(gpa: Allocator, pool: *Pool, file: gguf.File, bytes: []const u8) LoadError!Model {
        const hp = try Hparams.fromGguf(file);
        // Projection biases are optional in llama.cpp's graphs; none of the
        // supported models has them, so a file that does is refused rather
        // than run without them.
        for ([_][]const u8{ "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down" }) |w| {
            var buf: [64]u8 = undefined;
            if (file.tensor(std.fmt.bufPrint(&buf, "blk.0.{s}.bias", .{w}) catch unreachable) != null) return error.UnsupportedTensor;
        }
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        var l: Loader = .{ .arena = arena.allocator(), .pool = pool, .file = file, .bytes = bytes };

        const e: u64 = hp.n_embd;
        const ff: u64 = hp.n_ff;
        const hd: u64 = hp.head_dim;
        const kv_out = hd * hp.n_head_kv;
        const layers = try l.arena.alloc(Layer, hp.n_layer);
        for (layers, 0..) |*layer, i| {
            layer.* = .{
                .attn_norm = try l.get("blk.{d}.attn_norm.weight", i, &.{e}),
                .wq = try l.matrix("blk.{d}.attn_q.weight", i, &.{ e, hd * hp.n_head }),
                .wk = try l.matrix("blk.{d}.attn_k.weight", i, &.{ e, kv_out }),
                .wv = try l.matrix("blk.{d}.attn_v.weight", i, &.{ e, kv_out }),
                .wo = try l.matrix("blk.{d}.attn_output.weight", i, &.{ hd * hp.n_head, e }),
                .q_norm = if (hp.qk_norm) try l.get("blk.{d}.attn_q_norm.weight", i, &.{hd}) else null,
                .k_norm = if (hp.qk_norm) try l.get("blk.{d}.attn_k_norm.weight", i, &.{hd}) else null,
                .ffn_norm = try l.get("blk.{d}.ffn_norm.weight", i, &.{e}),
                .ffn_gate = try l.matrix("blk.{d}.ffn_gate.weight", i, &.{ e, ff }),
                .ffn_up = try l.matrix("blk.{d}.ffn_up.weight", i, &.{ e, ff }),
                .ffn_down = try l.matrix("blk.{d}.ffn_down.weight", i, &.{ ff, e }),
            };
        }
        const vocab: u64 = hp.vocab;
        return .{
            .hp = hp,
            .tok_embd = try l.globalMatrix("token_embd.weight", &.{ e, vocab }),
            .output = if (file.tensor("output.weight") != null) try l.globalMatrix("output.weight", &.{ e, vocab }) else null,
            .output_norm = try l.global("output_norm.weight", &.{e}),
            .rope_freqs = if (file.tensor("rope_freqs.weight") != null) try l.global("rope_freqs.weight", &.{hp.n_rot / 2}) else null,
            .layers = layers,
            .arena = arena,
        };
    }

    /// Runs `tokens` from an empty state. Reports `l_out-<i>` after each
    /// layer, then `result_norm` and `result_output` (logits).
    pub fn forward(m: *const Model, gpa: Allocator, tokens: []const u32, pool: *Pool, obs: Observer, act: mm.Activations) !void {
        const hp = m.hp;
        const n = tokens.len;
        const e: usize = hp.n_embd;

        const x = try gpa.alloc(f32, n * e);
        defer gpa.free(x);
        const h = try gpa.alloc(f32, n * e);
        defer gpa.free(h);
        const mix = try gpa.alloc(f32, n * e);
        defer gpa.free(mix);

        for (tokens, 0..) |tok, t| {
            if (tok >= hp.vocab) return error.TokenOutOfRange;
            m.tok_embd.row(tok, x[t * e ..][0..e]);
        }
        if (hp.embd_scale != 1) for (x) |*v| {
            v.* *= hp.embd_scale;
        };

        var scratch_arena: std.heap.ArenaAllocator = .init(gpa);
        defer scratch_arena.deinit();
        const scratch = scratch_arena.allocator();

        var name_buf: [32]u8 = undefined;
        for (m.layers, 0..) |*layer, il| {
            _ = scratch_arena.reset(.retain_capacity);
            normRows(h, x, layer.attn_norm, hp.rms_eps);
            try attention(scratch, m, layer, h, mix, n, pool, act);
            addScaled(x, mix, hp.residual_scale);

            normRows(h, x, layer.ffn_norm, hp.rms_eps);
            try ffn(scratch, hp, layer, h, mix, n, pool, act);
            addScaled(x, mix, hp.residual_scale);

            const name = std.fmt.bufPrint(&name_buf, "l_out-{d}", .{il}) catch unreachable;
            try obs.emit(obs.ctx, name, x, e, n);
        }

        normRows(h, x, m.output_norm, hp.rms_eps);
        try obs.emit(obs.ctx, "result_norm", h, e, n);

        const vocab: usize = hp.vocab;
        const logits = try gpa.alloc(f32, n * vocab);
        defer gpa.free(logits);
        mm.matmul(pool, logits, h, if (m.output) |*o| o else &m.tok_embd, n, act);
        if (hp.logit_mul != 1) for (logits) |*v| {
            v.* *= hp.logit_mul;
        };
        try obs.emit(obs.ctx, "result_output", logits, vocab, n);
    }
};

/// x += s * y; llama.cpp skips the multiply when s is 1.
fn addScaled(x: []f32, y: []const f32, s: f32) void {
    if (s == 1) {
        for (x, y) |*a, b| a.* += b;
    } else {
        for (x, y) |*a, b| a.* += b * s;
    }
}

/// out[t] = rmsnorm(in[t]) * weight for each row.
fn normRows(out: []f32, in: []const f32, weight: []const f32, eps: f32) void {
    const d = weight.len;
    @memcpy(out, in);
    var t: usize = 0;
    while (t * d < out.len) : (t += 1) ops.rmsNorm(out[t * d ..][0..d], weight, eps);
}

fn ffn(gpa: Allocator, hp: Hparams, layer: *const Layer, h: []const f32, out: []f32, n: usize, pool: *Pool, act: mm.Activations) !void {
    const ff: usize = hp.n_ff;
    const gate = try gpa.alloc(f32, n * ff);
    const up = try gpa.alloc(f32, n * ff);
    mm.matmul(pool, gate, h, &layer.ffn_gate, n, act);
    mm.matmul(pool, up, h, &layer.ffn_up, n, act);
    const Act = struct {
        gate: []f32,
        up: []const f32,
        ff: usize,
        fn run(c: *const @This(), first: usize, last: usize) void {
            const lo = first * c.ff;
            const hi = last * c.ff;
            for (c.gate[lo..hi], c.up[lo..hi]) |*g, u| g.* = ops.silu(g.*) * u;
        }
    };
    pool.parallelFor(n, 1, 1, &Act{ .gate = gate, .up = up, .ff = ff }, Act.run);
    mm.matmul(pool, out, gate, &layer.ffn_down, n, act);
}

/// Causal GQA softmax attention: Q/K/V projections, optional per-head Q/K
/// RMSNorm, RoPE, softmax(K·Q × attn_scale)·V, output projection.
fn attention(gpa: Allocator, m: *const Model, w: *const Layer, h: []const f32, out: []f32, n: usize, pool: *Pool, act: mm.Activations) !void {
    const hp = m.hp;
    const hd: usize = hp.head_dim;
    const q = try gpa.alloc(f32, n * hd * hp.n_head);
    const k = try gpa.alloc(f32, n * hd * hp.n_head_kv);
    const v = try gpa.alloc(f32, n * hd * hp.n_head_kv);
    const attn = try gpa.alloc(f32, n * hd * hp.n_head);
    mm.matmul(pool, q, h, &w.wq, n, act);
    mm.matmul(pool, k, h, &w.wk, n, act);
    mm.matmul(pool, v, h, &w.wv, n, act);

    const Ctx = struct {
        m: *const Model,
        w: *const Layer,
        q: []f32,
        k: []f32,
        v: []const f32,
        attn: []f32,
        n: usize,

        fn head(c: *const @This(), buf: []f32, t: usize, heads: usize, hh: usize) []f32 {
            const hd_ = c.m.hp.head_dim;
            return buf[(t * heads + hh) * hd_ ..][0..hd_];
        }

        fn normRopeOne(c: *const @This(), x: []f32, norm: ?[]const f32, pos: u32) void {
            const p = c.m.hp;
            if (norm) |wn| ops.rmsNorm(x, wn, p.rms_eps);
            switch (p.rope) {
                .neox => ops.rope(x, pos, p.n_rot, p.rope_base),
                .normal => ops.ropeNormal(x, pos, p.n_rot, p.rope_base, c.m.rope_freqs),
            }
        }

        /// Tokens [first, last): Q/K norms and RoPE in place.
        fn normRope(c: *const @This(), first: usize, last: usize) void {
            const p = c.m.hp;
            for (first..last) |t| {
                const pos: u32 = @intCast(t);
                for (0..p.n_head) |hh| c.normRopeOne(c.head(c.q, t, p.n_head, hh), c.w.q_norm, pos);
                for (0..p.n_head_kv) |kvh| c.normRopeOne(c.head(c.k, t, p.n_head_kv, kvh), c.w.k_norm, pos);
            }
        }

        /// Units [first, last) of (query head, token) pairs. Two passes over
        /// the keys (max, then exp-weighted sum) need no score buffer.
        fn attend(c: *const @This(), first: usize, last: usize) void {
            const p = c.m.hp;
            const hd_ = p.head_dim;
            for (first..last) |u| {
                const hh = u / c.n;
                const t = u % c.n;
                const kvh = hh / p.group();
                const qh = c.q[(t * p.n_head + hh) * hd_ ..][0..hd_];
                const kv = p.n_head_kv;

                var max: f32 = -std.math.inf(f32);
                for (0..t + 1) |j| max = @max(max, ops.dot(qh, c.k[(j * kv + kvh) * hd_ ..][0..hd_]) * p.attn_scale);
                const o = c.attn[(t * p.n_head + hh) * hd_ ..][0..hd_];
                @memset(o, 0);
                var sum: f32 = 0;
                for (0..t + 1) |j| {
                    const s = @exp(ops.dot(qh, c.k[(j * kv + kvh) * hd_ ..][0..hd_]) * p.attn_scale - max);
                    sum += s;
                    for (o, c.v[(j * kv + kvh) * hd_ ..][0..hd_]) |*a, b| a.* += s * b;
                }
                for (o) |*a| a.* /= sum;
            }
        }
    };
    const ctx: Ctx = .{ .m = m, .w = w, .q = q, .k = k, .v = v, .attn = attn, .n = n };
    pool.parallelFor(n, 1, 1, &ctx, Ctx.normRope);
    pool.parallelFor(hp.n_head * n, 1, 1, &ctx, Ctx.attend);
    mm.matmul(pool, out, attn, &w.wo, n, act);
}
