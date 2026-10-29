//! GPU execution plan for the Qwen 3.5 hybrid model.
//!
//! `Model` uploads the weights; `Program` builds everything a step needs
//! once (buffers, persistent state, prebuilt launches) and records steps:
//! m tokens at position pos0, optionally projecting the last rows to logits
//! and sampling the next token on the GPU. Prefill is a step with many
//! tokens, decode a step with one.
//!
//! Layout: matrices stay bf16 and are split by rows into chunks that each fit
//! one storage binding (`maxStorageBufferBindingSize`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const graph = @import("../graph/qwen35.zig");
const cmd = @import("../gpu/cmd.zig");
const kernels = @import("../kernels/kernels.zig");
const Arg = kernels.Arg;
const P = kernels.params;
const sampler = @import("../sampler/sampler.zig");

pub const Hparams = graph.Hparams;

pub const Limits = struct {
    /// maxStorageBufferBindingSize
    max_binding: u64,
    /// minStorageBufferOffsetAlignment
    offset_align: u64,
};

/// A bf16 matrix of `n` rows × `k` (row j = output j's weights), split by
/// rows into chunks.
pub const Matrix = struct {
    n: u32,
    k: u32,
    chunks: []const Chunk,

    /// `type`: bf16 or a quantized type (kernels.Kernel.quant).
    pub const Chunk = struct { arg: Arg, row0: u32, rows: u32, type: gguf.TensorType };
};

const Attention = struct {
    /// Q (with the output gates) | K | V, stacked by rows.
    wqkv: Matrix,
    wo: Matrix,
    q_norm: Arg,
    k_norm: Arg,
};

const DeltaNet = struct {
    /// qkv | z (the output gate) | β | α, stacked by rows.
    wproj: Matrix,
    wout: Matrix,
    conv: Arg,
    dt_bias: Arg,
    a: Arg,
    norm: Arg,
};

const Layer = struct {
    attn_norm: Arg,
    post_norm: Arg,
    /// gate | up, stacked by rows.
    ffn_gu: Matrix,
    ffn_down: Matrix,
    mixer: union(graph.LayerKind) {
        delta_net: DeltaNet,
        attention: Attention,
    },
};

/// Weights resident on the GPU.
pub const Model = struct {
    hp: Hparams,
    tok_embd: Matrix,
    output: ?Matrix,
    output_norm: Arg,
    layers: []const Layer,

    pub const UploadError = Hparams.Error || Allocator.Error || error{ MissingTensor, BadShape, UnsupportedTensorType, TensorTooLarge };

    /// Calls f(context, w) for every multiplied weight (not the embedding
    /// table).
    pub fn eachMatrix(m: *const Model, context: anytype, comptime f: fn (@TypeOf(context), *const Matrix) void) void {
        if (m.output) |*o| f(context, o);
        for (m.layers) |*l| {
            f(context, &l.ffn_gu);
            f(context, &l.ffn_down);
            switch (l.mixer) {
                .attention => |*w| {
                    f(context, &w.wqkv);
                    f(context, &w.wo);
                },
                .delta_net => |*w| {
                    f(context, &w.wproj);
                    f(context, &w.wout);
                },
            }
        }
    }

    /// Records buffer creation and an upload from the model file
    /// (`write_file`) for every weight; the host reads the file. `arena` owns
    /// the returned layout.
    pub fn upload(arena: Allocator, gpa: Allocator, rec: *cmd.Recorder, set: *const kernels.Set, file: gguf.File, limits: Limits) UploadError!Model {
        const hp = try Hparams.fromGguf(file);
        var u: Upload = .{ .arena = arena, .gpa = gpa, .rec = rec, .set = set, .file = file, .limits = limits };
        const e: u64 = hp.n_embd;
        const ff: u64 = hp.n_ff;

        const layers = try arena.alloc(Layer, hp.n_layer);
        for (layers, hp.kinds(), 0..) |*layer, kind, i| {
            layer.* = .{
                .attn_norm = try u.vector("blk.{d}.attn_norm.weight", i, e),
                .post_norm = try u.vector("blk.{d}.post_attention_norm.weight", i, e),
                .ffn_gu = try u.stacked(i, e, &.{ .{ "blk.{d}.ffn_gate.weight", ff }, .{ "blk.{d}.ffn_up.weight", ff } }),
                .ffn_down = try u.matrix("blk.{d}.ffn_down.weight", i, ff, e),
                .mixer = switch (kind) {
                    .attention => blk: {
                        const hd: u64 = hp.head_dim;
                        break :blk .{ .attention = .{
                            .wqkv = try u.stacked(i, e, &.{
                                .{ "blk.{d}.attn_q.weight", 2 * hd * hp.n_head },
                                .{ "blk.{d}.attn_k.weight", hd * hp.n_head_kv },
                                .{ "blk.{d}.attn_v.weight", hd * hp.n_head_kv },
                            }),
                            .wo = try u.matrix("blk.{d}.attn_output.weight", i, hd * hp.n_head, e),
                            .q_norm = try u.vector("blk.{d}.attn_q_norm.weight", i, hd),
                            .k_norm = try u.vector("blk.{d}.attn_k_norm.weight", i, hd),
                        } };
                    },
                    .delta_net => blk: {
                        const conv_dim: u64 = 2 * hp.gdn_head_dim * hp.gdn_k_heads + hp.gdn_inner;
                        break :blk .{ .delta_net = .{
                            .wproj = try u.stacked(i, e, &.{
                                .{ "blk.{d}.attn_qkv.weight", conv_dim },
                                .{ "blk.{d}.attn_gate.weight", hp.gdn_inner },
                                .{ "blk.{d}.ssm_beta.weight", hp.gdn_v_heads },
                                .{ "blk.{d}.ssm_alpha.weight", hp.gdn_v_heads },
                            }),
                            .wout = try u.matrix("blk.{d}.ssm_out.weight", i, hp.gdn_inner, e),
                            .conv = try u.tensor("blk.{d}.ssm_conv1d.weight", i, &.{ hp.conv_kernel, conv_dim }),
                            .dt_bias = try u.vector("blk.{d}.ssm_dt.bias", i, hp.gdn_v_heads),
                            .a = try u.vector("blk.{d}.ssm_a", i, hp.gdn_v_heads),
                            .norm = try u.vector("blk.{d}.ssm_norm.weight", i, hp.gdn_head_dim),
                        } };
                    },
                },
            };
        }

        return .{
            .hp = hp,
            .tok_embd = try u.matrix("token_embd.weight", null, e, hp.vocab),
            .output = if (file.tensor("output.weight") != null) try u.matrix("output.weight", null, e, hp.vocab) else null,
            .output_norm = try u.vector("output_norm.weight", null, e),
            .layers = layers,
        };
    }
};

const Upload = struct {
    arena: Allocator,
    gpa: Allocator,
    rec: *cmd.Recorder,
    /// For `repad`.
    set: *const kernels.Set,
    file: gguf.File,
    limits: Limits,

    fn info(u: *Upload, comptime fmt: []const u8, layer: ?usize, shape: []const u64, t: gguf.TensorType) !*const gguf.TensorInfo {
        var buf: [96]u8 = undefined;
        const name = if (comptime std.mem.indexOf(u8, fmt, "{d}") != null)
            std.fmt.bufPrint(&buf, fmt, .{layer.?}) catch unreachable
        else
            fmt;
        const ti = u.file.tensor(name) orelse return error.MissingTensor;
        if (!std.mem.eql(u64, ti.shape(), shape)) return error.BadShape;
        if (ti.type != t) return error.UnsupportedTensorType;
        return ti;
    }

    /// A matrix weight: bf16, or a type the quantized kernels decode.
    fn weightInfo(u: *Upload, comptime fmt: []const u8, layer: ?usize, shape: []const u64) !*const gguf.TensorInfo {
        var buf: [96]u8 = undefined;
        const name = if (comptime std.mem.indexOf(u8, fmt, "{d}") != null)
            std.fmt.bufPrint(&buf, fmt, .{layer.?}) catch unreachable
        else
            fmt;
        const ti = u.file.tensor(name) orelse return error.MissingTensor;
        if (!std.mem.eql(u64, ti.shape(), shape)) return error.BadShape;
        if (ti.type != .bf16 and kernels.Kernel.quant(.matvec, ti.type) == null) return error.UnsupportedTensorType;
        return ti;
    }

    /// Where `ti`'s bytes start in the file.
    fn fileOffset(u: *const Upload, ti: *const gguf.TensorInfo) u64 {
        return u.file.data_offset + ti.offset;
    }

    /// An f32 tensor of any shape.
    fn tensor(u: *Upload, comptime fmt: []const u8, layer: ?usize, shape: []const u64) !Arg {
        const ti = try u.info(fmt, layer, shape, .f32);
        const size = ti.byteSize();
        if (size > u.limits.max_binding) return error.TensorTooLarge;
        const buf = try u.rec.createBuffer(u.gpa, size, .{ .storage = true, .copy_dst = true });
        try u.rec.writeFile(u.gpa, buf, 0, u.fileOffset(ti), size);
        return .{ .buffer = buf, .size = size };
    }

    fn vector(u: *Upload, comptime fmt: []const u8, layer: ?usize, n: u64) !Arg {
        return u.tensor(fmt, layer, &.{n});
    }

    /// A GGML {k, n} weight (bf16 or quantized), split into chunks of whole
    /// rows.
    fn matrix(u: *Upload, comptime fmt: []const u8, layer: ?usize, k: u64, n: u64) !Matrix {
        return u.stacked(layer, k, &.{.{ fmt, n }});
    }

    /// GGML {k, n_i} weights sharing an input, stacked by rows into one
    /// matrix (part i's rows follow part i − 1's) and split into chunks of
    /// whole rows. Consecutive parts of one weight type share chunks, each
    /// part's bytes written at their place; a change of type starts new
    /// chunks, since a chunk is decoded by one kernel.
    fn stacked(u: *Upload, layer: ?usize, k: u64, parts: anytype) !Matrix {
        const Span = struct { offset: u64, rows: u64, type: gguf.TensorType };
        var spans: [parts.len]Span = undefined;
        var n: u64 = 0;
        inline for (parts, &spans) |part, *span| {
            const ti = try u.weightInfo(part[0], layer, &.{ k, part[1] });
            span.* = .{ .offset = u.fileOffset(ti), .rows = part[1], .type = ti.type };
            n += part[1];
        }
        var chunks: std.ArrayList(Matrix.Chunk) = .empty;
        var first: usize = 0;
        var run_row0: u64 = 0;
        while (first < spans.len) {
            // A run of parts of one type.
            var end = first + 1;
            while (end < spans.len and spans[end].type == spans[first].type) end += 1;
            var run_rows: u64 = 0;
            for (spans[first..end]) |s| run_rows += s.rows;
            try u.chunkRun(&chunks, k, spans[first].type, spans[first..end], run_row0, run_rows);
            run_row0 += run_rows;
            first = end;
        }
        return .{ .n = @intCast(n), .k = @intCast(k), .chunks = chunks.items };
    }

    /// Chunks of a run of parts of type `t` covering matrix rows
    /// [run_row0, run_row0 + run_rows). Quantized blocks whose size is not a
    /// multiple of 16 (Q3_K, Q6_K) are laid out padded
    /// (kernels.blockStride): the rows go to a staging buffer first, then
    /// `repad` places each block, and the staging buffer is dropped after
    /// its own submit, so at most one chunk's worth is ever extra.
    fn chunkRun(u: *Upload, chunks: *std.ArrayList(Matrix.Chunk), k: u64, t: gguf.TensorType, spans: anytype, run_row0: u64, run_rows: u64) !void {
        const l = t.layout().?;
        const blocks_per_row = k / l.block_elems;
        const file_row = blocks_per_row * l.block_bytes;
        const stride: u64 = kernels.blockStride(t);
        const gpu_row = blocks_per_row * stride;
        const repad = stride != l.block_bytes;
        const rows_per_chunk = @min(run_rows, u.limits.max_binding / gpu_row);
        if (rows_per_chunk == 0) return error.TensorTooLarge;
        var row0: u64 = 0;
        while (row0 < run_rows) : (row0 += rows_per_chunk) {
            const rows = @min(rows_per_chunk, run_rows - row0);
            // Storage bindings are whole words; the quantized kernels read
            // 16-byte vectors.
            const size = std.mem.alignForward(u64, rows * gpu_row, 16);
            const buf = try u.rec.createBuffer(u.gpa, size, .{ .storage = true, .copy_dst = true });
            const raw_size = std.mem.alignForward(u64, rows * file_row, 4);
            const raw = if (repad) try u.rec.createBuffer(u.gpa, raw_size, .{ .storage = true, .copy_dst = true }) else buf;
            // Copy the parts' rows that fall in [row0, row0 + rows) of the run.
            var part_row0: u64 = 0;
            for (spans) |span| {
                const lo = @max(row0, part_row0);
                const hi = @min(row0 + rows, part_row0 + span.rows);
                if (lo < hi) {
                    try u.rec.writeFile(u.gpa, raw, (lo - row0) * file_row, span.offset + (lo - part_row0) * file_row, (hi - lo) * file_row);
                }
                part_row0 += span.rows;
            }
            if (repad) {
                const blocks = rows * blocks_per_row;
                const launcher: kernels.Launcher = .{ .gpa = u.gpa, .arena = u.arena, .rec = u.rec, .set = u.set };
                try launcher.launch(.repad, P.Repad{ .blocks = @intCast(blocks), .src_bytes = l.block_bytes, .dst_bytes = @intCast(stride) }, &.{
                    .{ .buffer = raw, .size = raw_size },
                    .{ .buffer = buf, .size = size },
                }, kernels.Launcher.elementwise(@intCast(blocks * stride / 4)));
                try u.rec.submit(u.gpa, 0);
                try u.rec.destroy(u.gpa, .buffer, @backingInt(raw));
            }
            try chunks.append(u.arena, .{ .arg = .{ .buffer = buf, .size = size }, .row0 = @intCast(run_row0 + row0), .rows = @intCast(rows), .type = t });
        }
    }
};

pub const KvFormat = enum(u32) { f32 = 0, f16 = 1 };

pub const Config = struct {
    /// Most tokens one step processes (one prefill chunk).
    max_tokens: u32 = 512,
    /// Positions the KV cache and the RoPE table hold.
    context: u32 = 8192,
    /// Rows of logits one head pass computes: 1 for generation; more lets a
    /// check read every position's logits in fewer passes.
    logit_rows: u32 = 1,
    kv: KvFormat = .f16,
    /// DeltaNet checkpoint slots.
    checkpoints: u32 = 4,
};

/// One entry of the step body.
pub const Item = union(enum) {
    /// The same launch on both kernel paths.
    launch: kernels.Launch,
    /// Different launches per kernel path.
    fork: Fork,
    /// The final RMSNorm into `h`: the prefill head reads it; the decode head
    /// folds it into its projection, so decode runs it only for a readback.
    final_norm: kernels.Launch,
    /// The end of a layer, where a check may read the residual stream back.
    layer_end: u32,
};

/// Launches that differ by kernel path: the prefill kernels unfused around
/// tiled matrix multiplies, the decode kernels fused into matrix-vector
/// multiplies.
pub const Fork = struct {
    prefill: []const kernels.Launch,
    decode: []const kernels.Launch,

    pub fn pick(f: Fork, path: Path) []const kernels.Launch {
        return if (path == .decode) f.decode else f.prefill;
    }
};

/// Which kernels a step runs: the prefill kernels, or the decode kernels
/// (fused matrix-vector multiplies, split attention) for steps of at most
/// `kernels.decode_max_rows` tokens. Each is bitwise independent of how a
/// sequence is split into steps, but the two round differently.
pub const Path = enum {
    prefill,
    decode,

    pub fn of(rows: u32) Path {
        return if (rows <= kernels.decode_max_rows) .decode else .prefill;
    }
};

/// Everything a step needs, built once: activation buffers sized for
/// `max_tokens`, the KV caches, the DeltaNet state and its checkpoints, the
/// sampler's buffers, and every launch with its bind group. A step then
/// writes one uniform (and its token ids) and replays the launches with
/// grids sized to its token count.
///
/// Records keep pointers into the Program (staging for uploads), so it must
/// not move while a recorded batch is pending.
pub const Program = struct {
    hp: Hparams,
    cfg: Config,
    step: Arg,
    token_ids: Arg,
    x: Arg,
    h: Arg,
    lm_in: Arg,
    logits: Arg,
    /// Every DeltaNet layer's recurrent state and conv history, back to back.
    recurrent: Arg,
    checkpoints: []const cmd.Buffer,
    sample_params: Arg,
    bias: Arg,
    prompt_ids: Arg,
    prompt_mask: Arg,
    counts: Arg,
    /// The sampler's state; the drawn token and its stop flag come first.
    sample_state: Arg,

    body: []const Item,
    /// The vocabulary projection: on the prefill path from `lm_in` (rows
    /// copied from `h`), on the decode path from `x` with the final norm.
    head: Fork,
    mark: kernels.Launch,
    /// One launch per sampler phase and round.
    sample: std.EnumArray(kernels.SamplePhase, [4]kernels.Launch),
    /// The phases the current request's draws run: the first `sample_len`.
    sample_steps: [kernels.max_sample_steps]kernels.SampleStep = undefined,
    sample_len: usize = 0,
    /// Tokens per tile of the prefill multiplies: a step's partial tile
    /// costs as much as a whole one (kernels.matmulTiles).
    prefill_tile: u32,

    // Upload staging, alive until the batch that writes it has executed.
    step_value: P.Step = .{ .m = 0, .pos0 = 0, .u = 0 },
    sample_value: P.Sample = undefined,
    token_stage: []u32,
    prompt_stage: []u32,
    bias_stage: [][2]u32,

    pub const BuildError = Allocator.Error || error{ BadConfig, TooLarge };

    /// Records creation of every buffer and bind group. `arena` owns the
    /// result and the parameter staging, and must outlive the Program.
    pub fn build(arena: Allocator, gpa: Allocator, rec: *cmd.Recorder, set: *const kernels.Set, m: *const Model, cfg: Config, limits: Limits) BuildError!Program {
        const hp = m.hp;
        if (cfg.max_tokens == 0 or cfg.logit_rows == 0 or cfg.context < cfg.max_tokens or cfg.checkpoints == 0) return error.BadConfig;
        if (hp.head_dim > kernels.max_head_dim or hp.gdn_head_dim > kernels.max_delta_head or hp.n_embd % 4 != 0) return error.BadConfig;
        if (hp.n_head % hp.n_head_kv != 0 or hp.n_head / hp.n_head_kv > kernels.attn_max_group) return error.BadConfig;
        const e: u64 = hp.n_embd;
        const cap: u64 = cfg.max_tokens;
        const conv_dim: u64 = 2 * hp.gdn_head_dim * hp.gdn_k_heads + hp.gdn_inner;
        const kv_dim: u64 = hp.head_dim * hp.n_head_kv;

        var a: Alloc = .{ .gpa = gpa, .rec = rec, .limits = limits };
        const hd: u64 = hp.head_dim;
        // Row widths of the fused projections.
        const qkv_row: u64 = 2 * hd * hp.n_head + 2 * kv_dim;
        const proj_row: u64 = conv_dim + hp.gdn_inner + 2 * @as(u64, hp.gdn_v_heads);
        const x = try a.storage(cap * e);
        const h = try a.storage(cap * e);
        const gu = try a.storage(cap * 2 * hp.n_ff);
        const act = try a.storage(cap * hp.n_ff);
        const qkv = try a.storage(cap * qkv_row);
        const att = try a.storage(cap * hd * hp.n_head);
        const proj = try a.storage(cap * proj_row);
        const conv = try a.storage(cap * conv_dim);
        const beta = try a.storage(cap * hp.gdn_v_heads);
        const g = try a.storage(cap * hp.gdn_v_heads);
        const o = try a.storage(cap * hp.gdn_inner);
        const token_ids = try a.storage(cap);
        const lm_in = try a.storage(@as(u64, cfg.logit_rows) * e);
        const logits = try a.storage(@as(u64, cfg.logit_rows) * hp.vocab);

        const rope_table = try kernels.ropeTable(arena, cfg.context, hp.n_rot, hp.rope_base);
        const rope = try a.storage(rope_table.len);
        try rec.writeBuffer(gpa, rope.buffer, 0, std.mem.sliceAsBytes(rope_table));
        const step = try a.buffer(@sizeOf(P.Step), .{ .uniform = true, .copy_dst = true });

        // Recurrent state: per DeltaNet layer, the state then the conv
        // history, each at an aligned offset so it can be bound alone.
        const state_bytes: u64 = 4 * @as(u64, hp.gdn_v_heads) * hp.gdn_head_dim * hp.gdn_head_dim;
        const hist_bytes: u64 = 4 * @as(u64, hp.conv_kernel - 1) * conv_dim;
        const al = limits.offset_align;
        const per_layer = std.mem.alignForward(u64, state_bytes, al) + std.mem.alignForward(u64, hist_bytes, al);
        var n_delta: u64 = 0;
        for (hp.kinds()) |kind| n_delta += @intFromBool(kind == .delta_net);
        const recurrent = try a.buffer(@max(per_layer * n_delta, 4), .{ .storage = true, .copy_src = true, .copy_dst = true });
        const checkpoints = try arena.alloc(cmd.Buffer, cfg.checkpoints);
        for (checkpoints) |*cp| cp.* = (try a.buffer(recurrent.size, .{ .copy_src = true, .copy_dst = true })).buffer;

        const kv_bytes: u64 = if (cfg.kv == .f16) 2 else 4;
        const kv_layer = @as(u64, cfg.context) * kv_dim * kv_bytes;

        const sample_params = try a.buffer(@sizeOf(P.Sample), .{ .uniform = true, .copy_dst = true });
        const bias = try a.storage(2 * kernels.max_bias);
        const prompt_ids = try a.storage(cfg.context);
        const prompt_mask = try a.storage(hp.vocab);
        const counts = try a.storage(hp.vocab);
        const work = try a.storage(hp.vocab);
        // Decode attention's per-block partials: (max, sum, output) per
        // (token, head, block of cache positions).
        const max_blocks: u32 = std.math.divCeil(u32, cfg.context, kernels.attn_block) catch unreachable;
        const attn_part = try a.storage(@as(u64, kernels.decode_max_rows) * hp.n_head * max_blocks * (hp.head_dim + 2));
        const sample_state = try a.storage(kernels.sampleStateWords(hp.vocab));

        var b: kernels.Builder = .{ .gpa = gpa, .arena = arena, .rec = rec, .set = set };
        defer b.deinit();
        var body: std.ArrayList(Item) = .empty;
        const capn: u32 = cfg.max_tokens;
        const norm: P.RmsNorm = .{ .rows = capn, .dim = hp.n_embd, .eps = hp.rms_eps };
        const mm: Mm = .{ .b = &b, .arena = arena, .cap = capn, .eps = hp.rms_eps, .partials = try a.storage(Mm.partialFloats(m, cap)) };

        for (m.tok_embd.chunks) |ch| {
            try body.append(arena, .{ .launch = try b.add(weightKernel(.embed, ch.type), P.EmbedBf16{ .dim = hp.n_embd, .row0 = ch.row0, .chunk_rows = ch.rows }, &.{ token_ids, ch.arg, x }, .rows_x) });
        }
        var delta_i: u64 = 0;
        for (m.layers, 0..) |*layer, il| {
            const pre_norm = try b.add(.rmsnorm, norm, &.{ x, layer.attn_norm, h }, .rows_x);
            switch (layer.mixer) {
                .attention => |*w| {
                    const kc = try a.buffer(kv_layer, .{ .storage = true });
                    const vc = try a.buffer(kv_layer, .{ .storage = true });
                    const half: u32 = @backingInt(cfg.kv);
                    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
                    const q_row: u32 = @intCast(qkv_row);
                    try body.append(arena, .{ .fork = .{
                        .prefill = try concat(arena, &.{ &.{pre_norm}, try mm.tiled(qkv, h, &w.wqkv, false) }),
                        .decode = try mm.vector(qkv, x, &w.wqkv, .{ .norm = layer.attn_norm }),
                    } });
                    try body.append(arena, .{ .launch = try b.add(.rope_append, P.RopeAppend{
                        .heads = hp.n_head,
                        .kv_heads = hp.n_head_kv,
                        .head_dim = hp.head_dim,
                        .n_rot = hp.n_rot,
                        .row = q_row,
                        .k_off = @intCast(2 * hd * hp.n_head),
                        .v_off = @intCast(2 * hd * hp.n_head + kv_dim),
                        .f16 = half,
                        .eps = hp.rms_eps,
                    }, &.{ qkv, w.q_norm, w.k_norm, rope, kc, vc, step }, .{ .rows_y = hp.n_head + hp.n_head_kv }) });
                    const decode: P.AttnDecode = .{ .heads = hp.n_head, .kv_heads = hp.n_head_kv, .head_dim = hp.head_dim, .f16 = half, .scale = scale, .max_blocks = max_blocks, .q_row = q_row };
                    try body.append(arena, .{ .fork = .{
                        .prefill = try concat(arena, &.{
                            &.{try b.add(.attention, P.Attention{
                                .heads = hp.n_head,
                                .kv_heads = hp.n_head_kv,
                                .head_dim = hp.head_dim,
                                .f16 = half,
                                .scale = scale,
                                .q_row = q_row,
                            }, &.{ qkv, kc, vc, att, step }, .{ .rows_y = hp.n_head })},
                            try mm.tiled(x, att, &w.wo, true),
                        }),
                        .decode = try concat(arena, &.{
                            &.{
                                try b.add(.attn_split, decode, &.{ qkv, kc, vc, attn_part, step }, .{ .key_blocks = .{ .x = hp.n_head_kv, .block = kernels.attn_block } }),
                                try b.add(.attn_combine, decode, &.{ qkv, attn_part, att, step }, .{ .rows_y = hp.n_head }),
                            },
                            try mm.vector(x, att, &w.wo, .{ .accumulate = true }),
                        }),
                    } });
                },
                .delta_net => |*w| {
                    const base = delta_i * per_layer;
                    delta_i += 1;
                    const state: Arg = .{ .buffer = recurrent.buffer, .offset = base, .size = state_bytes };
                    const hist: Arg = .{ .buffer = recurrent.buffer, .offset = base + std.mem.alignForward(u64, state_bytes, al), .size = hist_bytes };
                    const d = hp.gdn_head_dim;
                    try body.append(arena, .{ .fork = .{
                        .prefill = try concat(arena, &.{ &.{pre_norm}, try mm.tiled(proj, h, &w.wproj, false) }),
                        .decode = try mm.vector(proj, x, &w.wproj, .{ .norm = layer.attn_norm }),
                    } });
                    try body.append(arena, .{ .launch = try b.add(.delta_prep, P.DeltaPrep{
                        .channels = @intCast(conv_dim),
                        .taps = hp.conv_kernel,
                        .key_channels = 2 * d * hp.gdn_k_heads,
                        .head = d,
                        .row = @intCast(proj_row),
                        .v_heads = hp.gdn_v_heads,
                        .beta_off = @intCast(conv_dim + hp.gdn_inner),
                        .alpha_off = @intCast(conv_dim + hp.gdn_inner + hp.gdn_v_heads),
                        .eps = hp.rms_eps,
                    }, &.{ proj, w.conv, hist, conv, w.dt_bias, w.a, g, beta }, .{ .rows_y = @intCast(conv_dim / d + 1) }) });
                    try body.append(arena, .{ .launch = try b.add(.delta_rule, P.DeltaRule{
                        .k_heads = hp.gdn_k_heads,
                        .v_heads = hp.gdn_v_heads,
                        .dim = d,
                        .conv_stride = @intCast(conv_dim),
                        .taps = hp.conv_kernel,
                        .row = @intCast(proj_row),
                    }, &.{ conv, g, beta, state, o, step, proj, hist }, .{ .fixed = .{ hp.gdn_v_heads, std.math.divCeil(u32, d, kernels.delta_cols) catch unreachable, 1 } }) });
                    try body.append(arena, .{ .launch = try b.add(.gated_norm, P.GatedNorm{
                        .v_heads = hp.gdn_v_heads,
                        .dim = d,
                        .eps = hp.rms_eps,
                        .z_row = @intCast(proj_row),
                        .z_off = @intCast(conv_dim),
                    }, &.{ o, proj, w.norm }, .{ .rows_y = hp.gdn_v_heads }) });
                    try body.append(arena, .{ .fork = .{
                        .prefill = try mm.tiled(x, o, &w.wout, true),
                        .decode = try mm.vector(x, o, &w.wout, .{ .accumulate = true }),
                    } });
                },
            }
            const ff = hp.n_ff;
            try body.append(arena, .{ .fork = .{
                .prefill = try concat(arena, &.{
                    &.{try b.add(.rmsnorm, norm, &.{ x, layer.post_norm, h }, .rows_x)},
                    try mm.tiled(gu, h, &layer.ffn_gu, false),
                    &.{try b.add(.silu_mul, P.SiluMul{ .count = capn * ff, .width = ff }, &.{ gu, act }, .{ .elems = ff })},
                    try mm.tiled(x, act, &layer.ffn_down, true),
                }),
                .decode = try concat(arena, &.{
                    try mm.vector(act, x, &layer.ffn_gu, .{ .norm = layer.post_norm, .silu_pair = ff }),
                    try mm.vector(x, act, &layer.ffn_down, .{ .accumulate = true }),
                }),
            } });
            try body.append(arena, .{ .layer_end = @intCast(il) });
        }
        try body.append(arena, .{ .final_norm = try b.add(.rmsnorm, norm, &.{ x, m.output_norm, h }, .rows_x) });

        const lm = if (m.output) |*out| out else &m.tok_embd;
        const head: Fork = .{
            .prefill = try mm.tiled(logits, lm_in, lm, false),
            .decode = try mm.vector(logits, x, lm, .{ .norm = m.output_norm }),
        };

        const mark = try b.addWith(.mark_tokens, sample_params, &.{ prompt_ids, prompt_mask }, .{ .elems = 1 });
        var sample: std.EnumArray(kernels.SamplePhase, [4]kernels.Launch) = undefined;
        for (std.enums.values(kernels.SamplePhase)) |phase| {
            const groups: u32 = if (phase.wide()) kernels.sampleGroups(hp.vocab) else 1;
            const rounds: u32 = if (phase == .hist or phase == .digit) 4 else 1;
            for (sample.getPtr(phase), 0..) |*l, r| {
                // Phases with one round reuse round 0's launch.
                if (r >= rounds) {
                    l.* = sample.get(phase)[0];
                    continue;
                }
                l.* = try b.add(.sample, P.SamplePhase{ .phase = @backingInt(phase), .round = @intCast(r) }, &.{
                    logits, bias, prompt_mask, counts, work, sample_state, token_ids, step, sample_params,
                }, .{ .fixed = .{ groups, 1, 1 } });
            }
        }
        try b.finish();

        return .{
            .hp = hp,
            .cfg = cfg,
            .prefill_tile = prefillTile(m),
            .step = step,
            .token_ids = token_ids,
            .x = x,
            .h = h,
            .lm_in = lm_in,
            .logits = logits,
            .recurrent = recurrent,
            .checkpoints = checkpoints,
            .sample_params = sample_params,
            .bias = bias,
            .prompt_ids = prompt_ids,
            .prompt_mask = prompt_mask,
            .counts = counts,
            .sample_state = sample_state,
            .body = body.items,
            .head = head,
            .mark = mark,
            .sample = sample,
            .token_stage = try arena.alloc(u32, cfg.max_tokens),
            .prompt_stage = try arena.alloc(u32, cfg.context),
            .bias_stage = try arena.alloc([2]u32, kernels.max_bias),
        };
    }

    /// Zeroes the DeltaNet state: the next step starts a sequence at
    /// position 0. (Cache rows are only read at positions a step has written.)
    pub fn recordReset(p: *const Program, gpa: Allocator, rec: *cmd.Recorder) Allocator.Error!void {
        try rec.clearBuffer(gpa, p.recurrent.buffer, 0, p.recurrent.size);
    }

    pub fn recordSave(p: *const Program, gpa: Allocator, rec: *cmd.Recorder, slot: u32) Allocator.Error!void {
        try rec.copyBuffer(gpa, p.recurrent.buffer, 0, p.checkpoints[slot], 0, p.recurrent.size);
    }

    pub fn recordRestore(p: *const Program, gpa: Allocator, rec: *cmd.Recorder, slot: u32) Allocator.Error!void {
        try rec.copyBuffer(gpa, p.checkpoints[slot], 0, p.recurrent.buffer, 0, p.recurrent.size);
    }

    /// Sets up the sampler for a request: its parameters, the prompt's token
    /// set for the repetition penalty, and zero output counts.
    pub fn recordRequest(p: *Program, gpa: Allocator, rec: *cmd.Recorder, params: sampler.Params, prompt: []const u32, stops: []const u32) Allocator.Error!void {
        std.debug.assert(prompt.len <= p.cfg.context);
        p.sample_value = params.gpu(p.hp.vocab, @intCast(prompt.len), stops);
        p.sample_len = kernels.sampleSteps(p.sample_value, &p.sample_steps).len;
        try rec.writeBuffer(gpa, p.sample_params.buffer, 0, std.mem.asBytes(&p.sample_value));
        const biases = params.biasWords(p.bias_stage);
        if (biases.len > 0) try rec.writeBuffer(gpa, p.bias.buffer, 0, std.mem.sliceAsBytes(biases));
        @memcpy(p.prompt_stage[0..prompt.len], prompt);
        if (prompt.len > 0) try rec.writeBuffer(gpa, p.prompt_ids.buffer, 0, std.mem.sliceAsBytes(p.prompt_stage[0..prompt.len]));
        try rec.clearBuffer(gpa, p.counts.buffer, 0, p.counts.size);
        try rec.clearBuffer(gpa, p.prompt_mask.buffer, 0, p.prompt_mask.size);
        if (prompt.len > 0) try p.mark.record(gpa, rec, .{ .rows = @intCast(prompt.len) });
    }

    pub const Step = struct {
        /// Position of the step's first token.
        pos0: u32,
        /// Token ids to process; null: the one token the previous step sampled.
        tokens: ?[]const u32,
        /// Project this many of the step's last rows to logits.
        logit_rows: u32 = 0,
        /// Sample from the last row with this draw (needs logit_rows == 1),
        /// reading back (token, is_stop) into `sampled`.
        draw: ?f32 = null,
        sampled: ?*[2]u32 = null,
        /// Activation readbacks for checks.
        readback: ?Readback = null,
        /// The kernels to use; by default the decode kernels for steps of at
        /// most `kernels.decode_max_rows` tokens, else the prefill kernels.
        path: ?Path = null,
    };

    /// Where a step's readbacks land, row-major [rows][dim]: `layer_out[i]`
    /// gets l_out-i and `result_norm` the final norm (m rows each), `logits`
    /// the step's logit rows. Each is optional. Must outlive execution.
    pub const Readback = struct {
        layer_out: []const []f32 = &.{},
        result_norm: ?[]f32 = null,
        logits: ?[]f32 = null,
    };

    pub fn recordStep(p: *Program, gpa: Allocator, rec: *cmd.Recorder, s: Step) Allocator.Error!void {
        const m: u32 = if (s.tokens) |t| @intCast(t.len) else 1;
        const path = s.path orelse Path.of(m);
        std.debug.assert(path == .prefill or m <= kernels.decode_max_rows);
        const e: usize = p.hp.n_embd;
        std.debug.assert(m > 0 and m <= p.cfg.max_tokens and s.pos0 + m <= p.cfg.context);
        std.debug.assert(s.logit_rows <= m and (s.draw == null or s.logit_rows == 1));

        p.step_value = .{ .m = m, .pos0 = s.pos0, .u = s.draw orelse 0 };
        try rec.writeBuffer(gpa, p.step.buffer, 0, std.mem.asBytes(&p.step_value));
        if (s.tokens) |t| {
            @memcpy(p.token_stage[0..t.len], t);
            try rec.writeBuffer(gpa, p.token_ids.buffer, 0, std.mem.sliceAsBytes(p.token_stage[0..t.len]));
        }

        // The decode head projects x itself, with the final norm fused, when
        // it covers every row of the step in one pass; otherwise rows of the
        // final norm are copied to lm_in for the prefill head.
        const fused_head = path == .decode and s.logit_rows == m and m <= p.cfg.logit_rows;
        const want_norm = s.readback != null and s.readback.?.result_norm != null;
        const ext: kernels.Extent = .{ .rows = m, .keys = s.pos0 + m };
        for (p.body) |item| switch (item) {
            .launch => |l| try l.record(gpa, rec, ext),
            .fork => |f| for (f.pick(path)) |l| try l.record(gpa, rec, ext),
            .final_norm => |l| if (path == .prefill or want_norm or (s.logit_rows > 0 and !fused_head)) try l.record(gpa, rec, ext),
            .layer_end => |il| if (s.readback) |r| if (r.layer_out.len > 0) {
                try rec.readBuffer(gpa, read_token, p.x.buffer, 0, std.mem.sliceAsBytes(r.layer_out[il][0 .. m * e]));
            },
        };
        if (s.readback) |r| if (r.result_norm) |dst| try rec.readBuffer(gpa, read_token, p.h.buffer, 0, std.mem.sliceAsBytes(dst[0 .. m * e]));

        // Logit rows in passes of at most cfg.logit_rows.
        const vocab: usize = p.hp.vocab;
        var r0 = m - s.logit_rows;
        while (r0 < m) {
            const n = @min(p.cfg.logit_rows, m - r0);
            if (fused_head) {
                for (p.head.decode) |l| try l.record(gpa, rec, .{ .rows = n });
            } else {
                try rec.copyBuffer(gpa, p.h.buffer, r0 * e * 4, p.lm_in.buffer, 0, n * e * 4);
                for (p.head.prefill) |l| try l.record(gpa, rec, .{ .rows = n });
            }
            if (s.readback) |r| if (r.logits) |dst| {
                const at = (r0 - (m - s.logit_rows)) * vocab;
                try rec.readBuffer(gpa, read_token, p.logits.buffer, 0, std.mem.sliceAsBytes(dst[at..][0 .. n * vocab]));
            };
            r0 += n;
        }

        if (s.draw != null) {
            std.debug.assert(p.sample_len > 0);
            for (p.sample_steps[0..p.sample_len]) |ss| try p.sample.get(ss.phase)[ss.round].record(gpa, rec, .{ .rows = 1 });
            try rec.readBuffer(gpa, read_token, p.sample_state.buffer, 0, std.mem.asBytes(s.sampled.?));
        }
    }
};

/// Completion token of every read a Program records.
pub const read_token = 0xffff_fffe;

/// Records buffer creation, checking each fits one binding.
const Alloc = struct {
    gpa: Allocator,
    rec: *cmd.Recorder,
    limits: Limits,

    fn buffer(a: *Alloc, size: u64, usage: cmd.Usage) Program.BuildError!Arg {
        if (usage.storage and size > a.limits.max_binding) return error.TooLarge;
        return .{ .buffer = try a.rec.createBuffer(a.gpa, size, usage), .size = size };
    }

    /// `count` f32 or u32 values.
    fn storage(a: *Alloc, count: u64) Program.BuildError!Arg {
        return a.buffer(count * 4, .{ .storage = true, .copy_src = true, .copy_dst = true });
    }
};

/// The largest token tile among the model's prefill multiplies.
fn prefillTile(m: *const Model) u32 {
    const Tile = struct {
        tokens: u32 = 1,
        fn add(s: *@This(), w: *const Matrix) void {
            for (w.chunks) |ch| s.tokens = @max(s.tokens, kernels.matmulTiles(ch.type)[1]);
        }
    };
    var tile: Tile = .{};
    m.eachMatrix(&tile, Tile.add);
    return tile.tokens;
}

/// Builds a weight's multiply, one launch per chunk, either way.
const Mm = struct {
    b: *kernels.Builder,
    arena: Allocator,
    cap: u32,
    eps: f32,
    /// Partial sums of the prefill multiplies split over k
    /// (partialFloats).
    partials: Arg,

    /// Floats of `partials` for steps of `cap` tokens: the largest split
    /// chunk's slices × rows × cap, at least one.
    fn partialFloats(m: *const Model, cap: u64) u64 {
        const Most = struct {
            cap: u64,
            floats: u64 = 1,
            fn add(s: *@This(), w: *const Matrix) void {
                for (w.chunks) |ch| {
                    if (ch.type == .bf16) continue;
                    const slices = kernels.matmulSlices(ch.rows, w.k);
                    if (slices.count > 1) s.floats = @max(s.floats, @as(u64, slices.count) * ch.rows * s.cap);
                }
            }
        };
        var most: Most = .{ .cap = cap };
        m.eachMatrix(&most, Most.add);
        return most.floats;
    }

    /// What the decode multiply fuses around it.
    const Fuse = struct {
        /// RMSNorm the input with these weights first.
        norm: ?Arg = null,
        /// out += result.
        accumulate: bool = false,
        /// The weight is gate | up, each this many rows: out = silu(gate) · up.
        silu_pair: ?u32 = null,
    };

    /// out[t][j] (+)= x[t] · w[j] for the rows of a prefill step: tiled;
    /// quantized chunks with too few column tiles split over k into
    /// `partials` and are added up by matmul_reduce.
    fn tiled(mm: Mm, out: Arg, x: Arg, w: *const Matrix, accumulate: bool) Allocator.Error![]const kernels.Launch {
        std.debug.assert(w.k % kernels.matmul_tile_k == 0);
        var list: std.ArrayList(kernels.Launch) = .empty;
        for (w.chunks) |ch| {
            const tiles = kernels.matmulTiles(ch.type);
            const x_tiles = std.math.divCeil(u32, ch.rows, tiles[0]) catch unreachable;
            if (ch.type == .bf16) {
                try list.append(mm.arena, try mm.b.add(.matmul_bf16, P.MatmulBf16{
                    .rows = mm.cap,
                    .n = ch.rows,
                    .k = w.k,
                    .col0 = ch.row0,
                    .out_stride = w.n,
                    .accumulate = @intFromBool(accumulate),
                }, &.{ x, ch.arg, out }, .{ .tiles_y = .{ .x = x_tiles, .tile = tiles[1] } }));
                continue;
            }
            const s = kernels.matmulSlices(ch.rows, w.k);
            const kernel = weightKernel(.matmul, ch.type);
            if (s.count == 1) {
                try list.append(mm.arena, try mm.b.add(kernel, P.MatmulQuant{
                    .rows = mm.cap,
                    .n = ch.rows,
                    .k = w.k,
                    .col0 = ch.row0,
                    .out_stride = w.n,
                    .accumulate = @intFromBool(accumulate),
                    .k_slice = w.k,
                }, &.{ x, ch.arg, out }, .{ .tiles_y = .{ .x = x_tiles, .tile = tiles[1] } }));
                continue;
            }
            const stride = mm.cap * ch.rows;
            std.debug.assert(@as(u64, s.count) * stride * 4 <= mm.partials.size);
            try list.append(mm.arena, try mm.b.add(kernel, P.MatmulQuant{
                .rows = mm.cap,
                .n = ch.rows,
                .k = w.k,
                .col0 = 0,
                .out_stride = ch.rows,
                .k_slice = s.k_slice,
                .slice_stride = stride,
            }, &.{ x, ch.arg, mm.partials }, .{ .tiles_y = .{ .x = x_tiles, .tile = tiles[1], .z = s.count } }));
            try list.append(mm.arena, try mm.b.add(.matmul_reduce, P.MatmulReduce{
                .rows = mm.cap,
                .n = ch.rows,
                .slices = s.count,
                .col0 = ch.row0,
                .out_stride = w.n,
                .accumulate = @intFromBool(accumulate),
                .slice_stride = stride,
            }, &.{ mm.partials, out }, .{ .elems = ch.rows }));
        }
        return list.items;
    }

    /// The same for a decode step, matrix-vector, with `fuse` folded in.
    fn vector(mm: Mm, out: Arg, x: Arg, w: *const Matrix, fuse: Fuse) Program.BuildError![]const kernels.Launch {
        if (fuse.silu_pair) |half| {
            // Both halves of a row pair must be in one binding.
            if (w.chunks.len != 1) return error.TooLarge;
            std.debug.assert(w.n == 2 * half);
        }
        const flags: P.MatvecBf16.Flags = .{ .norm = fuse.norm != null, .accumulate = fuse.accumulate, .silu_pair = fuse.silu_pair != null };
        const list = try mm.arena.alloc(kernels.Launch, w.chunks.len);
        for (w.chunks, list) |ch, *l| {
            // The bf16 kernel normalizes only staged inputs.
            if (ch.type == .bf16 and fuse.norm != null and w.k > kernels.matvec_max_staged) return error.BadConfig;
            const n = fuse.silu_pair orelse ch.rows;
            l.* = try mm.b.add(weightKernel(.matvec, ch.type), P.MatvecBf16{
                .n = n,
                .k = w.k,
                .col0 = if (fuse.silu_pair != null) 0 else ch.row0,
                .out_stride = if (fuse.silu_pair) |half| half else w.n,
                .eps = mm.eps,
                .flags = flags,
                .half = fuse.silu_pair orelse 0,
            }, &.{
                x,                       ch.arg, out,
                // Unread without the norm; the weight is a read-only stand-in.
                fuse.norm orelse ch.arg,
            }, .{ .rows_y = std.math.divCeil(u32, n, if (ch.type == .bf16) kernels.matvec_rows else kernels.matvec_quant_rows) catch unreachable });
        }
        return list;
    }
};

/// The kernel for operation `op` on a weight chunk of type `t` (bf16, or a
/// quantized type Upload.weightInfo accepted).
fn weightKernel(comptime op: kernels.QuantOp, t: gguf.TensorType) kernels.Kernel {
    if (t == .bf16) return switch (op) {
        .matvec => .matvec_bf16,
        .matmul => .matmul_bf16,
        .embed => .embed_bf16,
    };
    return kernels.Kernel.quant(op, t).?;
}

fn concat(arena: Allocator, parts: []const []const kernels.Launch) Allocator.Error![]const kernels.Launch {
    return std.mem.concat(arena, kernels.Launch, parts);
}
