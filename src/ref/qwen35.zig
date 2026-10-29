//! CPU f32 reference forward pass for the Qwen 3.5 hybrid model.
//!
//! Teacher-forced: one call processes a whole token sequence from an empty
//! state and reports activations for every position. All weights are
//! dequantized to f32 at load; every operation accumulates in f32.
//! See docs/src/qwen35.md for the tensor semantics this code assumes.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const graph = @import("../graph/qwen35.zig");
const tensor = @import("tensor.zig");
const ops = @import("ops.zig");

pub const Hparams = graph.Hparams;

const Attention = struct {
    /// Query and output gate, interleaved per head: [q_h (head_dim), gate_h (head_dim)].
    wq: []f32,
    wk: []f32,
    wv: []f32,
    wo: []f32,
    q_norm: []f32,
    k_norm: []f32,
};

const DeltaNet = struct {
    /// Fused q, k, v input projection: [q (k_heads·d), k (k_heads·d), v (v_heads·d)].
    wqkv: []f32,
    /// Output gate z.
    wgate: []f32,
    /// Depthwise causal conv taps, `conv_kernel` per channel.
    conv: []f32,
    dt_bias: []f32,
    /// Stored as -exp(A_log) by the converter.
    a: []f32,
    wbeta: []f32,
    walpha: []f32,
    norm: []f32,
    wout: []f32,
};

const Layer = struct {
    attn_norm: []f32,
    post_norm: []f32,
    ffn_gate: []f32,
    ffn_up: []f32,
    ffn_down: []f32,
    mixer: union(graph.LayerKind) {
        delta_net: DeltaNet,
        attention: Attention,
    },
};

pub const Model = struct {
    hp: Hparams,
    tok_embd: []f32,
    /// Separate LM head, or null when tied to `tok_embd`.
    output: ?[]f32,
    output_norm: []f32,
    layers: []Layer,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(m: *Model) void {
        m.arena.deinit();
        m.* = undefined;
    }

    pub const LoadError = Hparams.Error || tensor.Error;

    pub fn load(gpa: Allocator, file: gguf.File, bytes: []const u8) LoadError!Model {
        const hp = try Hparams.fromGguf(file);
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        const e: u64 = hp.n_embd;
        const ff: u64 = hp.n_ff;
        const vocab: u64 = hp.vocab;
        var l: Loader = .{ .gpa = a, .file = file, .bytes = bytes };

        const layers = try a.alloc(Layer, hp.n_layer);
        for (layers, hp.kinds(), 0..) |*layer, kind, i| {
            layer.* = .{
                .attn_norm = try l.get("blk.{d}.attn_norm.weight", i, &.{e}),
                .post_norm = try l.get("blk.{d}.post_attention_norm.weight", i, &.{e}),
                .ffn_gate = try l.get("blk.{d}.ffn_gate.weight", i, &.{ e, ff }),
                .ffn_up = try l.get("blk.{d}.ffn_up.weight", i, &.{ e, ff }),
                .ffn_down = try l.get("blk.{d}.ffn_down.weight", i, &.{ ff, e }),
                .mixer = switch (kind) {
                    .attention => blk: {
                        const hd: u64 = hp.head_dim;
                        const q_out = 2 * hd * hp.n_head;
                        const kv_out = hd * hp.n_head_kv;
                        break :blk .{ .attention = .{
                            .wq = try l.get("blk.{d}.attn_q.weight", i, &.{ e, q_out }),
                            .wk = try l.get("blk.{d}.attn_k.weight", i, &.{ e, kv_out }),
                            .wv = try l.get("blk.{d}.attn_v.weight", i, &.{ e, kv_out }),
                            .wo = try l.get("blk.{d}.attn_output.weight", i, &.{ hd * hp.n_head, e }),
                            .q_norm = try l.get("blk.{d}.attn_q_norm.weight", i, &.{hd}),
                            .k_norm = try l.get("blk.{d}.attn_k_norm.weight", i, &.{hd}),
                        } };
                    },
                    .delta_net => blk: {
                        const d: u64 = hp.gdn_head_dim;
                        const key_dim = d * hp.gdn_k_heads;
                        const conv_dim = 2 * key_dim + hp.gdn_inner;
                        break :blk .{ .delta_net = .{
                            .wqkv = try l.get("blk.{d}.attn_qkv.weight", i, &.{ e, conv_dim }),
                            .wgate = try l.get("blk.{d}.attn_gate.weight", i, &.{ e, hp.gdn_inner }),
                            .conv = try l.get("blk.{d}.ssm_conv1d.weight", i, &.{ hp.conv_kernel, conv_dim }),
                            .dt_bias = try l.get("blk.{d}.ssm_dt.bias", i, &.{hp.gdn_v_heads}),
                            .a = try l.get("blk.{d}.ssm_a", i, &.{hp.gdn_v_heads}),
                            .wbeta = try l.get("blk.{d}.ssm_beta.weight", i, &.{ e, hp.gdn_v_heads }),
                            .walpha = try l.get("blk.{d}.ssm_alpha.weight", i, &.{ e, hp.gdn_v_heads }),
                            .norm = try l.get("blk.{d}.ssm_norm.weight", i, &.{d}),
                            .wout = try l.get("blk.{d}.ssm_out.weight", i, &.{ hp.gdn_inner, e }),
                        } };
                    },
                },
            };
        }

        return .{
            .hp = hp,
            .tok_embd = try l.global("token_embd.weight", &.{ e, vocab }),
            .output = if (file.tensor("output.weight") != null)
                try l.global("output.weight", &.{ e, vocab })
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
    /// llama.cpp tensor names.
    pub fn forward(m: *const Model, gpa: Allocator, tokens: []const u32, threads: usize, obs: Observer) !void {
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
            @memcpy(x[t * e ..][0..e], m.tok_embd[tok * e ..][0..e]);
        }

        var name_buf: [32]u8 = undefined;
        for (m.layers, 0..) |*layer, il| {
            normRows(h, x, layer.attn_norm, hp.rms_eps);
            switch (layer.mixer) {
                .attention => |*att| try attention(gpa, hp, att, h, mix, n, threads),
                .delta_net => |*dn| try deltaNet(gpa, hp, dn, h, mix, n, threads),
            }
            for (x, mix) |*a, b| a.* += b;

            normRows(h, x, layer.post_norm, hp.rms_eps);
            try ffn(gpa, hp, layer, h, mix, n, threads);
            for (x, mix) |*a, b| a.* += b;

            const name = std.fmt.bufPrint(&name_buf, "l_out-{d}", .{il}) catch unreachable;
            try obs.emit(obs.ctx, name, x, e, n);
        }

        normRows(h, x, m.output_norm, hp.rms_eps);
        try obs.emit(obs.ctx, "result_norm", h, e, n);

        const vocab: usize = hp.vocab;
        const logits = try gpa.alloc(f32, n * vocab);
        defer gpa.free(logits);
        ops.matmul(logits, h, m.output orelse m.tok_embd, n, vocab, e, threads);
        try obs.emit(obs.ctx, "result_output", logits, vocab, n);
    }
};

const Loader = struct {
    gpa: Allocator,
    file: gguf.File,
    bytes: []const u8,

    fn global(l: *Loader, name: []const u8, shape: []const u64) tensor.Error![]f32 {
        return tensor.load(l.gpa, l.file, l.bytes, name, shape);
    }

    fn get(l: *Loader, comptime fmt: []const u8, layer: usize, shape: []const u64) tensor.Error![]f32 {
        var buf: [96]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, fmt, .{layer}) catch unreachable;
        return tensor.load(l.gpa, l.file, l.bytes, name, shape);
    }
};

/// out[t] = rmsnorm(in[t]) * weight for each row.
fn normRows(out: []f32, in: []const f32, weight: []const f32, eps: f32) void {
    const d = weight.len;
    @memcpy(out, in);
    var t: usize = 0;
    while (t * d < out.len) : (t += 1) ops.rmsNorm(out[t * d ..][0..d], weight, eps);
}

fn ffn(gpa: Allocator, hp: Hparams, layer: *const Layer, h: []const f32, out: []f32, n: usize, threads: usize) !void {
    const e: usize = hp.n_embd;
    const ff: usize = hp.n_ff;
    const gate = try gpa.alloc(f32, n * ff);
    defer gpa.free(gate);
    const up = try gpa.alloc(f32, n * ff);
    defer gpa.free(up);
    ops.matmul(gate, h, layer.ffn_gate, n, ff, e, threads);
    ops.matmul(up, h, layer.ffn_up, n, ff, e, threads);
    for (gate, up) |*g, u| g.* = ops.silu(g.*) * u;
    ops.matmul(out, gate, layer.ffn_down, n, e, ff, threads);
}

/// Gated full attention: joint Q+gate projection, per-head Q/K RMSNorm,
/// partial NeoX RoPE, causal GQA softmax attention, sigmoid output gate.
fn attention(gpa: Allocator, hp: Hparams, w: *const Attention, h: []const f32, out: []f32, n: usize, threads: usize) !void {
    const e: usize = hp.n_embd;
    const hd: usize = hp.head_dim;
    const n_head: usize = hp.n_head;
    const n_kv: usize = hp.n_head_kv;
    const q_w = 2 * hd * n_head;
    const kv_w = hd * n_kv;

    const qg = try gpa.alloc(f32, n * q_w);
    defer gpa.free(qg);
    const k = try gpa.alloc(f32, n * kv_w);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, n * kv_w);
    defer gpa.free(v);
    ops.matmul(qg, h, w.wq, n, q_w, e, threads);
    ops.matmul(k, h, w.wk, n, kv_w, e, threads);
    ops.matmul(v, h, w.wv, n, kv_w, e, threads);

    for (0..n) |t| {
        const pos: u32 = @intCast(t);
        for (0..n_head) |hh| {
            const q = qg[t * q_w + hh * 2 * hd ..][0..hd];
            ops.rmsNorm(q, w.q_norm, hp.rms_eps);
            ops.rope(q, pos, hp.n_rot, hp.rope_base);
        }
        for (0..n_kv) |hh| {
            const kh = k[t * kv_w + hh * hd ..][0..hd];
            ops.rmsNorm(kh, w.k_norm, hp.rms_eps);
            ops.rope(kh, pos, hp.n_rot, hp.rope_base);
        }
    }

    const attn = try gpa.alloc(f32, n * hd * n_head);
    defer gpa.free(attn);
    const scores = try gpa.alloc(f32, n);
    defer gpa.free(scores);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const group = n_head / n_kv;
    for (0..n) |t| {
        for (0..n_head) |hh| {
            const q = qg[t * q_w + hh * 2 * hd ..][0..hd];
            const kvh = hh / group;
            const s = scores[0 .. t + 1];
            for (s, 0..) |*sc, j| sc.* = ops.dot(q, k[j * kv_w + kvh * hd ..][0..hd]) * scale;
            ops.softmax(s);
            const o = attn[(t * n_head + hh) * hd ..][0..hd];
            @memset(o, 0);
            for (s, 0..) |p, j| {
                const vj = v[j * kv_w + kvh * hd ..][0..hd];
                for (o, vj) |*a, b| a.* += p * b;
            }
            const gate = qg[t * q_w + hh * 2 * hd + hd ..][0..hd];
            for (o, gate) |*a, g| a.* *= ops.sigmoid(g);
        }
    }
    ops.matmul(out, attn, w.wo, n, e, hd * n_head, threads);
}

/// Gated DeltaNet: causal depthwise conv + SiLU over the fused q/k/v
/// projection, L2-normalized q and k, the gated delta rule recurrence, then
/// per-head RMSNorm gated by SiLU(z).
fn deltaNet(gpa: Allocator, hp: Hparams, w: *const DeltaNet, h: []const f32, out: []f32, n: usize, threads: usize) !void {
    const e: usize = hp.n_embd;
    const d: usize = hp.gdn_head_dim;
    const hk: usize = hp.gdn_k_heads;
    const hv: usize = hp.gdn_v_heads;
    const inner: usize = hp.gdn_inner;
    const key_dim = d * hk;
    const conv_dim = 2 * key_dim + inner;
    const kernel: usize = hp.conv_kernel;

    const qkv = try gpa.alloc(f32, n * conv_dim);
    defer gpa.free(qkv);
    const z = try gpa.alloc(f32, n * inner);
    defer gpa.free(z);
    const beta = try gpa.alloc(f32, n * hv);
    defer gpa.free(beta);
    const g = try gpa.alloc(f32, n * hv);
    defer gpa.free(g);
    ops.matmul(qkv, h, w.wqkv, n, conv_dim, e, threads);
    ops.matmul(z, h, w.wgate, n, inner, e, threads);
    ops.matmul(beta, h, w.wbeta, n, hv, e, threads);
    ops.matmul(g, h, w.walpha, n, hv, e, threads);
    for (beta) |*b| b.* = ops.sigmoid(b.*);
    for (0..n) |t| for (0..hv) |j| {
        const gi = &g[t * hv + j];
        gi.* = ops.softplus(gi.* + w.dt_bias[j]) * w.a[j];
    };

    // Causal conv over time per channel, zero history, then SiLU.
    const conv = try gpa.alloc(f32, n * conv_dim);
    defer gpa.free(conv);
    for (0..n) |t| for (0..conv_dim) |c| {
        var sum: f32 = 0;
        for (0..kernel) |tap| {
            const back = kernel - 1 - tap;
            if (t >= back) sum += w.conv[c * kernel + tap] * qkv[(t - back) * conv_dim + c];
        }
        conv[t * conv_dim + c] = ops.silu(sum);
    };

    const state = try gpa.alloc(f32, hv * d * d);
    defer gpa.free(state);
    @memset(state, 0);
    const delta = try gpa.alloc(f32, d);
    defer gpa.free(delta);
    const o_all = try gpa.alloc(f32, n * inner);
    defer gpa.free(o_all);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    for (0..n) |t| {
        const row = conv[t * conv_dim ..][0..conv_dim];
        for (0..hk) |i| {
            ops.l2Norm(row[i * d ..][0..d], hp.rms_eps);
            ops.l2Norm(row[key_dim + i * d ..][0..d], hp.rms_eps);
        }
        for (0..hv) |j| {
            // Value heads map onto key heads by tiling (j mod k_heads); the
            // converter reorders value heads to match.
            const q = row[(j % hk) * d ..][0..d];
            const kk = row[key_dim + (j % hk) * d ..][0..d];
            const v = row[2 * key_dim + j * d ..][0..d];
            // s[a * d + b] = S[a][b], a over keys, b over values.
            const s = state[j * d * d ..][0 .. d * d];
            const decay = @exp(g[t * hv + j]);
            const b = beta[t * hv + j];
            for (s) |*x| x.* *= decay;
            // delta[b] = (v[b] - Σ_a S[a][b] k[a]) · β
            @memset(delta, 0);
            for (0..d) |a| {
                const sa = s[a * d ..][0..d];
                for (delta, sa) |*dl, x| dl.* += x * kk[a];
            }
            for (delta, v) |*dl, vb| dl.* = (vb - dl.*) * b;
            // S[a][b] += k[a] · delta[b]; o[b] = Σ_a S[a][b] q[a] / sqrt(d)
            const o = o_all[t * inner + j * d ..][0..d];
            @memset(o, 0);
            for (0..d) |a| {
                const sa = s[a * d ..][0..d];
                for (sa, delta) |*x, dl| x.* += kk[a] * dl;
                for (o, sa) |*ob, x| ob.* += x * q[a];
            }
            for (o) |*ob| ob.* *= scale;

            ops.rmsNorm(o, w.norm, hp.rms_eps);
            const zh = z[t * inner + j * d ..][0..d];
            for (o, zh) |*ob, zz| ob.* *= ops.silu(zz);
        }
    }
    ops.matmul(out, o_all, w.wout, n, e, inner, threads);
}
