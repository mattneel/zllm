//! CPU f32 reference forward pass for the Qwen 3.5 hybrid model.
//!
//! Teacher-forced: one call processes a whole token sequence from an empty
//! state and reports activations for every position. Weights are decoded to
//! f32 (f32, f16 and bf16 matrices repacked for `matmul` at load, quantized
//! ones panel by panel as they are used); every operation accumulates in
//! f32.
//! See docs/src/qwen35.md for the tensor semantics this code assumes.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const graph = @import("../graph/qwen35.zig");
const tensor = @import("tensor.zig");
const ops = @import("ops.zig");
const mm = @import("matmul.zig");
const Matrix = mm.Matrix;
const Pool = @import("pool.zig").Pool;
const Loader = @import("loader.zig").Loader;

const vec_len = 16;
const V = @Vector(vec_len, f32);
/// Widest DeltaNet head the recurrence supports, in vectors (256 floats).
const max_head_vecs = 16;

pub const Hparams = graph.Hparams;

const Attention = struct {
    /// Query and output gate, interleaved per head: [q_h (head_dim), gate_h (head_dim)].
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
    q_norm: []f32,
    k_norm: []f32,
};

const DeltaNet = struct {
    /// Fused q, k, v input projection: [q (k_heads·d), k (k_heads·d), v (v_heads·d)].
    wqkv: Matrix,
    /// Output gate z.
    wgate: Matrix,
    /// Depthwise causal conv taps, `conv_kernel` per channel.
    conv: []f32,
    dt_bias: []f32,
    /// Stored as -exp(A_log) by the converter.
    a: []f32,
    wbeta: Matrix,
    walpha: Matrix,
    norm: []f32,
    wout: Matrix,
};

const Layer = struct {
    attn_norm: []f32,
    post_norm: []f32,
    ffn_gate: Matrix,
    ffn_up: Matrix,
    ffn_down: Matrix,
    mixer: union(graph.LayerKind) {
        delta_net: DeltaNet,
        attention: Attention,
    },
};

/// The multi-token prediction (NextN) block: from position t's final hidden
/// state and token t+1, logits for token t+2 (llama.cpp's qwen35 graph_mtp).
const Mtp = struct {
    enorm: []f32,
    hnorm: []f32,
    /// [enorm(embedding), hnorm(hidden)] → n_embd.
    eh_proj: Matrix,
    /// A full-attention decoder block with its own cache.
    block: Layer,
    head_norm: []f32,
};

pub const Model = struct {
    hp: Hparams,
    /// The MTP block, for models that have one.
    mtp: ?Mtp = null,
    /// Embeddings (row j = token j), also the LM head when tied.
    tok_embd: Matrix,
    /// Separate LM head, or null when tied to `tok_embd`.
    output: ?Matrix,
    output_norm: []f32,
    layers: []Layer,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(m: *Model) void {
        m.arena.deinit();
        m.* = undefined;
    }

    pub const LoadError = Hparams.Error || tensor.Error || error{UnsupportedShape};

    pub fn load(gpa: Allocator, pool: *Pool, file: gguf.File, bytes: []const u8) LoadError!Model {
        const hp = try Hparams.fromGguf(file);
        // The DeltaNet recurrence keeps a head's δ and o in vector registers.
        if (hp.gdn_head_dim % vec_len != 0 or hp.gdn_head_dim / vec_len > max_head_vecs) return error.UnsupportedShape;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        const e: u64 = hp.n_embd;
        const ff: u64 = hp.n_ff;
        const vocab: u64 = hp.vocab;
        var l: Loader = .{ .arena = a, .pool = pool, .file = file, .bytes = bytes };

        const layers = try a.alloc(Layer, hp.n_layer);
        for (layers, hp.kinds(), 0..) |*layer, kind, i| {
            layer.* = .{
                .attn_norm = try l.get("blk.{d}.attn_norm.weight", i, &.{e}),
                .post_norm = try l.get("blk.{d}.post_attention_norm.weight", i, &.{e}),
                .ffn_gate = try l.matrix("blk.{d}.ffn_gate.weight", i, &.{ e, ff }),
                .ffn_up = try l.matrix("blk.{d}.ffn_up.weight", i, &.{ e, ff }),
                .ffn_down = try l.matrix("blk.{d}.ffn_down.weight", i, &.{ ff, e }),
                .mixer = switch (kind) {
                    .attention => .{ .attention = try loadAttention(&l, hp, i) },
                    .delta_net => blk: {
                        const d: u64 = hp.gdn_head_dim;
                        const key_dim = d * hp.gdn_k_heads;
                        const conv_dim = 2 * key_dim + hp.gdn_inner;
                        break :blk .{ .delta_net = .{
                            .wqkv = try l.matrix("blk.{d}.attn_qkv.weight", i, &.{ e, conv_dim }),
                            .wgate = try l.matrix("blk.{d}.attn_gate.weight", i, &.{ e, hp.gdn_inner }),
                            .conv = try l.get("blk.{d}.ssm_conv1d.weight", i, &.{ hp.conv_kernel, conv_dim }),
                            .dt_bias = try l.get("blk.{d}.ssm_dt.bias", i, &.{hp.gdn_v_heads}),
                            .a = try l.get("blk.{d}.ssm_a", i, &.{hp.gdn_v_heads}),
                            .wbeta = try l.matrix("blk.{d}.ssm_beta.weight", i, &.{ e, hp.gdn_v_heads }),
                            .walpha = try l.matrix("blk.{d}.ssm_alpha.weight", i, &.{ e, hp.gdn_v_heads }),
                            .norm = try l.get("blk.{d}.ssm_norm.weight", i, &.{d}),
                            .wout = try l.matrix("blk.{d}.ssm_out.weight", i, &.{ hp.gdn_inner, e }),
                        } };
                    },
                },
            };
        }

        const mtp: ?Mtp = if (hp.n_layer_nextn == 1) blk: {
            const i: usize = hp.n_layer;
            break :blk .{
                .enorm = try l.get("blk.{d}.nextn.enorm.weight", i, &.{e}),
                .hnorm = try l.get("blk.{d}.nextn.hnorm.weight", i, &.{e}),
                .eh_proj = try l.matrix("blk.{d}.nextn.eh_proj.weight", i, &.{ 2 * e, e }),
                .block = .{
                    .attn_norm = try l.get("blk.{d}.attn_norm.weight", i, &.{e}),
                    .post_norm = try l.get("blk.{d}.post_attention_norm.weight", i, &.{e}),
                    .ffn_gate = try l.matrix("blk.{d}.ffn_gate.weight", i, &.{ e, ff }),
                    .ffn_up = try l.matrix("blk.{d}.ffn_up.weight", i, &.{ e, ff }),
                    .ffn_down = try l.matrix("blk.{d}.ffn_down.weight", i, &.{ ff, e }),
                    .mixer = .{ .attention = try loadAttention(&l, hp, i) },
                },
                // The model's output norm where the block has no own.
                .head_norm = if (l.has("blk.{d}.nextn.shared_head_norm.weight", i))
                    try l.get("blk.{d}.nextn.shared_head_norm.weight", i, &.{e})
                else
                    try l.global("output_norm.weight", &.{e}),
            };
        } else null;

        return .{
            .hp = hp,
            .mtp = mtp,
            .tok_embd = try l.globalMatrix("token_embd.weight", &.{ e, vocab }),
            .output = if (file.tensor("output.weight") != null)
                try l.globalMatrix("output.weight", &.{ e, vocab })
            else
                null,
            .output_norm = try l.global("output_norm.weight", &.{e}),
            .layers = layers,
            .arena = arena,
        };
    }

    /// Receives activations as row-major [n_tokens][dim] f32.
    pub const Observer = struct {
        ctx: *anyopaque,
        emit: *const fn (ctx: *anyopaque, name: []const u8, data: []const f32, dim: usize, n_tokens: usize) anyerror!void,
    };

    /// Runs `tokens` from an empty state. Reports `l_out-<i>` after each
    /// layer, then `result_norm` and `result_output` (logits), matching the
    /// llama.cpp tensor names. With `mtp` (and an MTP block), then also
    /// `mtp_logits`: row i is the block's prediction of token i + 2 from
    /// position i's final hidden state and token i + 1 (at position i + 1).
    pub fn forward(m: *const Model, gpa: Allocator, tokens: []const u32, pool: *Pool, obs: Observer, act: mm.Activations, mtp: bool) !void {
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

        // Per-layer temporaries come from one arena reset between layers, so
        // after the first layer no buffer is freshly mapped and zero-faulted.
        var scratch_arena: std.heap.ArenaAllocator = .init(gpa);
        defer scratch_arena.deinit();
        const scratch = scratch_arena.allocator();

        var name_buf: [32]u8 = undefined;
        for (m.layers, 0..) |*layer, il| {
            _ = scratch_arena.reset(.retain_capacity);
            normRows(h, x, layer.attn_norm, hp.rms_eps);
            switch (layer.mixer) {
                .attention => |*att| try attention(scratch, hp, att, h, mix, n, 0, pool, act),
                .delta_net => |*dn| try deltaNet(scratch, hp, dn, h, mix, n, pool, act),
            }
            for (x, mix) |*a, b| a.* += b;

            normRows(h, x, layer.post_norm, hp.rms_eps);
            try ffn(scratch, hp, layer, h, mix, n, pool, act);
            for (x, mix) |*a, b| a.* += b;

            const name = std.fmt.bufPrint(&name_buf, "l_out-{d}", .{il}) catch unreachable;
            try obs.emit(obs.ctx, name, x, e, n);
        }

        normRows(h, x, m.output_norm, hp.rms_eps);
        try obs.emit(obs.ctx, "result_norm", h, e, n);

        const vocab: usize = hp.vocab;
        const logits = try gpa.alloc(f32, n * vocab);
        defer gpa.free(logits);
        const head = if (m.output) |*o| o else &m.tok_embd;
        mm.matmul(pool, logits, h, head, n, act);
        try obs.emit(obs.ctx, "result_output", logits, vocab, n);

        const block = m.mtp orelse return;
        if (!mtp or n < 2) return;
        // Rows i = 0 .. n − 2: [enorm(embed(t_{i+1})), hnorm(h_i)].
        const r = n - 1;
        _ = scratch_arena.reset(.retain_capacity);
        const cat = try scratch.alloc(f32, r * 2 * e);
        for (0..r) |i| {
            const row = cat[i * 2 * e ..][0 .. 2 * e];
            m.tok_embd.row(tokens[i + 1], row[0..e]);
            ops.rmsNorm(row[0..e], block.enorm, hp.rms_eps);
            @memcpy(row[e..], h[i * e ..][0..e]);
            ops.rmsNorm(row[e..], block.hnorm, hp.rms_eps);
        }
        const mx = x[0 .. r * e];
        mm.matmul(pool, mx, cat, &block.eh_proj, r, act);
        const mh = h[0 .. r * e];
        const mmix = mix[0 .. r * e];
        normRows(mh, mx, block.block.attn_norm, hp.rms_eps);
        try attention(scratch, hp, &block.block.mixer.attention, mh, mmix, r, 1, pool, act);
        for (mx, mmix) |*a, b| a.* += b;
        normRows(mh, mx, block.block.post_norm, hp.rms_eps);
        try ffn(scratch, hp, &block.block, mh, mmix, r, pool, act);
        for (mx, mmix) |*a, b| a.* += b;
        normRows(mh, mx, block.head_norm, hp.rms_eps);
        mm.matmul(pool, logits[0 .. r * vocab], mh, head, r, act);
        try obs.emit(obs.ctx, "mtp_logits", logits[0 .. r * vocab], vocab, r);
    }
};

/// Layer `i`'s gated full attention.
fn loadAttention(l: *Loader, hp: Hparams, i: usize) tensor.Error!Attention {
    const e: u64 = hp.n_embd;
    const hd: u64 = hp.head_dim;
    const kv_out = hd * hp.n_head_kv;
    return .{
        .wq = try l.matrix("blk.{d}.attn_q.weight", i, &.{ e, 2 * hd * hp.n_head }),
        .wk = try l.matrix("blk.{d}.attn_k.weight", i, &.{ e, kv_out }),
        .wv = try l.matrix("blk.{d}.attn_v.weight", i, &.{ e, kv_out }),
        .wo = try l.matrix("blk.{d}.attn_output.weight", i, &.{ hd * hp.n_head, e }),
        .q_norm = try l.get("blk.{d}.attn_q_norm.weight", i, &.{hd}),
        .k_norm = try l.get("blk.{d}.attn_k_norm.weight", i, &.{hd}),
    };
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

/// Gated full attention: joint Q+gate projection, per-head Q/K RMSNorm,
/// partial NeoX RoPE, causal GQA softmax attention, sigmoid output gate.
/// Row t is position pos0 + t.
fn attention(gpa: Allocator, hp: Hparams, w: *const Attention, h: []const f32, out: []f32, n: usize, pos0: u32, pool: *Pool, act: mm.Activations) !void {
    const hd: usize = hp.head_dim;
    const n_head: usize = hp.n_head;
    const n_kv: usize = hp.n_head_kv;

    const qg = try gpa.alloc(f32, n * 2 * hd * n_head);
    const k = try gpa.alloc(f32, n * hd * n_kv);
    const v = try gpa.alloc(f32, n * hd * n_kv);
    const attn = try gpa.alloc(f32, n * hd * n_head);
    mm.matmul(pool, qg, h, &w.wq, n, act);
    mm.matmul(pool, k, h, &w.wk, n, act);
    mm.matmul(pool, v, h, &w.wv, n, act);

    const Ctx = struct {
        hp: Hparams,
        w: *const Attention,
        qg: []f32,
        k: []f32,
        v: []const f32,
        attn: []f32,
        n: usize,
        pos0: u32,

        fn q(c: *const @This(), t: usize, hh: usize) []f32 {
            const hd_ = c.hp.head_dim;
            return c.qg[(t * c.hp.n_head + hh) * 2 * hd_ ..][0..hd_];
        }
        fn gate(c: *const @This(), t: usize, hh: usize) []const f32 {
            const hd_ = c.hp.head_dim;
            return c.qg[(t * c.hp.n_head + hh) * 2 * hd_ + hd_ ..][0..hd_];
        }
        fn kvRow(c: *const @This(), buf: []const f32, t: usize, kvh: usize) []const f32 {
            const hd_ = c.hp.head_dim;
            return buf[(t * c.hp.n_head_kv + kvh) * hd_ ..][0..hd_];
        }

        /// Tokens [first, last): Q/K RMSNorm and RoPE in place.
        fn normRope(c: *const @This(), first: usize, last: usize) void {
            const hd_ = c.hp.head_dim;
            for (first..last) |t| {
                const pos: u32 = c.pos0 + @as(u32, @intCast(t));
                for (0..c.hp.n_head) |hh| {
                    ops.rmsNorm(c.q(t, hh), c.w.q_norm, c.hp.rms_eps);
                    ops.rope(c.q(t, hh), pos, c.hp.n_rot, c.hp.rope_base);
                }
                for (0..c.hp.n_head_kv) |kvh| {
                    const kh = c.k[(t * c.hp.n_head_kv + kvh) * hd_ ..][0..hd_];
                    ops.rmsNorm(kh, c.w.k_norm, c.hp.rms_eps);
                    ops.rope(kh, pos, c.hp.n_rot, c.hp.rope_base);
                }
            }
        }

        /// Units [first, last) of (query head, token) pairs: causal softmax
        /// attention for one query, then the sigmoid output gate. Two passes
        /// over the keys (max, then exp-weighted sum) need no score buffer.
        fn attend(c: *const @This(), first: usize, last: usize) void {
            const hd_ = c.hp.head_dim;
            const group = c.hp.n_head / c.hp.n_head_kv;
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd_)));
            for (first..last) |u| {
                const hh = u / c.n;
                const t = u % c.n;
                const kvh = hh / group;
                const qh = c.q(t, hh);

                var max: f32 = -std.math.inf(f32);
                for (0..t + 1) |j| max = @max(max, ops.dot(qh, c.kvRow(c.k, j, kvh)) * scale);

                const o = c.attn[(t * c.hp.n_head + hh) * hd_ ..][0..hd_];
                @memset(o, 0);
                var sum: f32 = 0;
                for (0..t + 1) |j| {
                    const p = @exp(ops.dot(qh, c.kvRow(c.k, j, kvh)) * scale - max);
                    sum += p;
                    for (o, c.kvRow(c.v, j, kvh)) |*a, b| a.* += p * b;
                }
                for (o, c.gate(t, hh)) |*a, g| a.* = a.* / sum * ops.sigmoid(g);
            }
        }
    };
    const ctx: Ctx = .{ .hp = hp, .w = w, .qg = qg, .k = k, .v = v, .attn = attn, .n = n, .pos0 = pos0 };
    pool.parallelFor(n, 1, 1, &ctx, Ctx.normRope);
    pool.parallelFor(n_head * n, 1, 1, &ctx, Ctx.attend);
    mm.matmul(pool, out, attn, &w.wo, n, act);
}

/// Gated DeltaNet: causal depthwise conv + SiLU over the fused q/k/v
/// projection, L2-normalized q and k, the gated delta rule recurrence, then
/// per-head RMSNorm gated by SiLU(z).
fn deltaNet(gpa: Allocator, hp: Hparams, w: *const DeltaNet, h: []const f32, out: []f32, n: usize, pool: *Pool, act: mm.Activations) !void {
    const d: usize = hp.gdn_head_dim;
    const hv: usize = hp.gdn_v_heads;
    const inner: usize = hp.gdn_inner;
    const conv_dim = 2 * d * hp.gdn_k_heads + inner;

    const qkv = try gpa.alloc(f32, n * conv_dim);
    const z = try gpa.alloc(f32, n * inner);
    const beta = try gpa.alloc(f32, n * hv);
    const g = try gpa.alloc(f32, n * hv);
    mm.matmul(pool, qkv, h, &w.wqkv, n, act);
    mm.matmul(pool, z, h, &w.wgate, n, act);
    mm.matmul(pool, beta, h, &w.wbeta, n, act);
    mm.matmul(pool, g, h, &w.walpha, n, act);
    for (beta) |*b| b.* = ops.sigmoid(b.*);
    for (0..n) |t| for (0..hv) |j| {
        const gi = &g[t * hv + j];
        gi.* = ops.softplus(gi.* + w.dt_bias[j]) * w.a[j];
    };

    const conv = try gpa.alloc(f32, n * conv_dim);
    const state = try gpa.alloc(f32, hv * d * d);
    @memset(state, 0);
    const o_all = try gpa.alloc(f32, n * inner);

    const Ctx = struct {
        hp: Hparams,
        w: *const DeltaNet,
        qkv: []const f32,
        conv: []f32,
        z: []const f32,
        beta: []const f32,
        g: []const f32,
        state: []f32,
        o_all: []f32,
        n: usize,

        /// Tokens [first, last): causal depthwise conv (zero history) + SiLU,
        /// then L2-normalize each q and k head.
        fn convNorm(c: *const @This(), first: usize, last: usize) void {
            const dd = c.hp.gdn_head_dim;
            const key_dim = dd * c.hp.gdn_k_heads;
            const cd = 2 * key_dim + c.hp.gdn_inner;
            const kernel: usize = c.hp.conv_kernel;
            for (first..last) |t| {
                const row = c.conv[t * cd ..][0..cd];
                for (row, 0..) |*o, ch| {
                    var sum: f32 = 0;
                    for (0..kernel) |tap| {
                        const back = kernel - 1 - tap;
                        if (t >= back) sum += c.w.conv[ch * kernel + tap] * c.qkv[(t - back) * cd + ch];
                    }
                    o.* = ops.silu(sum);
                }
                for (0..2 * c.hp.gdn_k_heads) |i| ops.l2Norm(row[i * dd ..][0..dd], c.hp.rms_eps);
            }
        }

        /// Value heads [first, last): the delta rule over all tokens, then
        /// the SiLU(z)-gated RMSNorm. The head width is dispatched at compile
        /// time so δ and o live in registers (`Model.load` checks the width).
        fn recur(c: *const @This(), first: usize, last: usize) void {
            switch (c.hp.gdn_head_dim / vec_len) {
                inline 1...max_head_vecs => |nv| for (first..last) |j| c.recurHead(nv, j),
                else => unreachable,
            }
        }

        fn recurHead(c: *const @This(), comptime nv: usize, j: usize) void {
            const D = nv * vec_len;
            const hk = c.hp.gdn_k_heads;
            const hv_ = c.hp.gdn_v_heads;
            const key_dim = D * hk;
            const cd = 2 * key_dim + c.hp.gdn_inner;
            const inner_ = c.hp.gdn_inner;
            const scale: V = @splat(1.0 / @sqrt(@as(f32, D)));
            // s[a * D + b] = S[a][b], a over keys, b over values. A many-item
            // pointer keeps per-access bounds checks out of the hot loops.
            const s: [*]f32 = c.state[j * D * D ..][0 .. D * D].ptr;
            for (0..c.n) |t| {
                const row = c.conv[t * cd ..][0..cd];
                // Value heads map onto key heads by tiling (j mod k_heads);
                // the converter reorders value heads to match.
                const q = row[(j % hk) * D ..][0..D];
                const kk = row[key_dim + (j % hk) * D ..][0..D];
                const v = row[2 * key_dim + j * D ..][0..D];
                const decay: V = @splat(@exp(c.g[t * hv_ + j]));
                const b: V = @splat(c.beta[t * hv_ + j]);

                // S ← e^g S, accumulating Sᵀk.
                var stk: [nv]V = @splat(@splat(0));
                for (0..D) |a| {
                    const ka: V = @splat(kk[a]);
                    inline for (0..nv) |i| {
                        const p = s[a * D + i * vec_len ..][0..vec_len];
                        const x = p.* * decay;
                        p.* = x;
                        stk[i] += x * ka;
                    }
                }
                // δ = (v − Sᵀk) · β
                var delta: [nv]V = undefined;
                inline for (0..nv) |i| delta[i] = (@as(V, v[i * vec_len ..][0..vec_len].*) - stk[i]) * b;
                // S ← S + k δᵀ, accumulating o = Sᵀq.
                var o: [nv]V = @splat(@splat(0));
                for (0..D) |a| {
                    const ka: V = @splat(kk[a]);
                    const qa: V = @splat(q[a]);
                    inline for (0..nv) |i| {
                        const p = s[a * D + i * vec_len ..][0..vec_len];
                        const x = p.* + ka * delta[i];
                        p.* = x;
                        o[i] += x * qa;
                    }
                }

                const oh = c.o_all[t * inner_ + j * D ..][0..D];
                inline for (0..nv) |i| oh[i * vec_len ..][0..vec_len].* = o[i] * scale;
                ops.rmsNorm(oh, c.w.norm, c.hp.rms_eps);
                for (oh, c.z[t * inner_ + j * D ..][0..D]) |*ob, zz| ob.* *= ops.silu(zz);
            }
        }
    };
    const ctx: Ctx = .{
        .hp = hp,
        .w = w,
        .qkv = qkv,
        .conv = conv,
        .z = z,
        .beta = beta,
        .g = g,
        .state = state,
        .o_all = o_all,
        .n = n,
    };
    pool.parallelFor(n, 1, 1, &ctx, Ctx.convNorm);
    pool.parallelFor(hv, 1, 1, &ctx, Ctx.recur);
    mm.matmul(pool, out, o_all, &w.wout, n, act);
}
