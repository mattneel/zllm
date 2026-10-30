//! The Qwen4Exp weights the reference graph reads, bound in place in the
//! mapped GGUF file (nothing is copied or decoded at load). Lifted from
//! gyro-dx's model.zig.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../../gguf/gguf.zig");
const kernels = @import("kernels.zig");
const ple = @import("ple.zig");
const Matrix = kernels.Matrix;

pub const arch = "qwen4exp";

pub const Error = Allocator.Error || ple.Error || error{ WrongArchitecture, MissingTensor, BadShape, UnsupportedTensorType, Unsupported };

pub const Hparams = struct {
    n_layer: u32,
    n_embd: u32,
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    n_rot: u32,
    rope_base: f32,
    eps: f32,
    n_expert: u32,
    n_expert_used: u32,
    n_ff_exp: u32,
    n_ff_shexp: u32,
    ssm_conv: u32,
    ssm_state: u32,
    ssm_k_heads: u32,
    ssm_v_heads: u32,
    ssm_inner: u32,
    /// Hyper-connection streams, and the rank of their mixing.
    hc: u32,
    hc_rank: u32,
    idx_heads: u32,
    idx_dim: u32,
    idx_top_k: u32,
    ple_layer: u32,
    ple_head_dim: u32,
    ple_conv: u32,
    ple_dilation: u32,
    vocab: u32,
    expert_weights_scale: f32,

    pub fn hcDim(h: Hparams) u32 {
        return h.hc * h.n_embd;
    }
    pub fn convChannels(h: Hparams) u32 {
        return h.ssm_state * h.ssm_k_heads * 2 + h.ssm_state * h.ssm_v_heads;
    }
};

pub const Hc = struct {
    norm: []const f32,
    down: Matrix,
    up: Matrix,
    inject: ?Matrix,
};

pub const Linear = struct {
    qkv: Matrix,
    z: Matrix,
    beta: Matrix,
    alpha: Matrix,
    out: Matrix,
    conv: []const f32,
    dt_bias: []const f32,
    a: []const f32,
    norm: []const f32,
};

pub const Full = struct {
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
    q_norm: []const f32,
    k_norm: []const f32,
    idx_q: Matrix,
    idx_k: Matrix,
    idx_q_norm: []const f32,
    idx_k_norm: []const f32,
    compress_ratio: u32,
};

pub const Ple = struct {
    key: Matrix,
    value: Matrix,
    norm_key: []const f32,
    norm_query: []const f32,
    norm_conv: []const f32,
    conv: []const f32,
};

pub const Layer = struct {
    hc_attn: Hc,
    hc_ffn: Hc,
    mixer: union(enum) { linear: Linear, full: Full },
    ple: ?Ple,
    router: Matrix,
    gate_exps: Matrix,
    up_exps: Matrix,
    down_exps: Matrix,
    /// Sign vectors of rotated expert tensors (prism.hadamard): stored as
    /// W·D·H, they take H·D·x.
    gate_signs: ?[]const f32,
    up_signs: ?[]const f32,
    down_signs: ?[]const f32,
    sh_gate: Matrix,
    sh_up: Matrix,
    sh_down: Matrix,
    sh_gate_inp: []const f32,

    pub fn expertGate(l: *const Layer, h: Hparams, e: usize) Matrix {
        return l.gate_exps.slice(e * h.n_ff_exp, h.n_ff_exp);
    }
    pub fn expertUp(l: *const Layer, h: Hparams, e: usize) Matrix {
        return l.up_exps.slice(e * h.n_ff_exp, h.n_ff_exp);
    }
    pub fn expertDown(l: *const Layer, h: Hparams, e: usize) Matrix {
        return l.down_exps.slice(e * h.n_embd, h.n_embd);
    }
};

pub const Model = struct {
    h: Hparams,
    ple_config: ple.Config,
    token_embd: Matrix,
    output: Matrix,
    /// The final mix of the streams, as a hyper-connection without inject.
    hc_head: Hc,
    ple_table: Matrix,
    layers: []Layer,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(m: *Model) void {
        m.arena.deinit();
        m.* = undefined;
    }

    /// Binds `file` (header) over `bytes` (the whole mapped file).
    pub fn load(gpa: Allocator, file: gguf.File, bytes: []const u8) Error!Model {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        var b: Binder = .{ .file = file, .bytes = bytes, .arena = arena.allocator() };
        var m = try b.bind();
        m.arena = arena;
        return m;
    }
};

const Binder = struct {
    file: gguf.File,
    bytes: []const u8,
    arena: Allocator,
    /// prism.hadamard: rotated tensor names, sign widths and values.
    rotated: []const []const u8 = &.{},
    sign_widths: []const u32 = &.{},
    signs: []const f32 = &.{},

    fn value(b: *const Binder, key: []const u8) Error!gguf.Value {
        return b.file.get(key) orelse error.MissingMetadata;
    }

    fn num(b: *const Binder, comptime key: []const u8) Error!u32 {
        return switch (try b.value(arch ++ "." ++ key)) {
            .u32 => |x| x,
            .i32 => |x| std.math.cast(u32, x) orelse error.BadMetadata,
            .u64 => |x| std.math.cast(u32, x) orelse error.BadMetadata,
            else => error.BadMetadata,
        };
    }

    fn float(b: *const Binder, comptime key: []const u8) Error!f32 {
        return switch (try b.value(arch ++ "." ++ key)) {
            .f32 => |x| x,
            else => error.BadMetadata,
        };
    }

    /// An integer array (i32 or u32 elements) as u32s.
    fn ints(b: *const Binder, key: []const u8) Error![]u32 {
        const v = try b.value(key);
        if (v != .array) return error.BadMetadata;
        const a = v.array;
        const out = try b.arena.alloc(u32, @intCast(a.len));
        for (out, 0..) |*o, i| o.* = switch (a.elem_type) {
            .u32 => a.get(u32, i),
            .i32 => std.math.cast(u32, @as(i32, @bitCast(a.get(u32, i)))) orelse return error.BadMetadata,
            else => return error.BadMetadata,
        };
        return out;
    }

    fn tensor(b: *const Binder, name: []const u8) Error!*const gguf.TensorInfo {
        return b.file.tensor(name) orelse {
            std.log.err("missing tensor {s}", .{name});
            return error.MissingTensor;
        };
    }

    fn f32s(b: *const Binder, name: []const u8, count: usize) Error![]const f32 {
        const t = try b.tensor(name);
        if (t.type != .f32) return error.UnsupportedTensorType;
        if (t.elemCount() != count) return error.BadShape;
        const bytes = b.file.tensorBytes(b.bytes, t);
        if (@intFromPtr(bytes.ptr) % 4 != 0) return error.BadShape;
        return @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0..count];
    }

    /// A {cols, rows…} weight as `rows` rows of `cols`.
    fn matrix(b: *const Binder, name: []const u8, cols: usize, rows: usize) Error!Matrix {
        const t = try b.tensor(name);
        if (!kernels.supported(t.type)) return error.UnsupportedTensorType;
        if (t.dims[0] != cols or t.elemCount() != @as(u64, cols) * rows) {
            std.log.err("tensor {s} has shape {any}, expected [{d}, {d}]", .{ name, t.shape(), cols, rows });
            return error.BadShape;
        }
        const l = t.type.layout().?;
        if (cols % l.block_elems != 0) return error.BadShape;
        return .{ .type = t.type, .cols = cols, .rows = rows, .row_bytes = cols / l.block_elems * l.block_bytes, .data = b.file.tensorBytes(b.bytes, t) };
    }

    /// The sign vector of `name` at `width` if it is rotated, else null.
    fn signsOf(b: *const Binder, name: []const u8, width: usize) Error!?[]const f32 {
        for (b.rotated) |r| {
            if (!std.mem.eql(u8, r, name)) continue;
            var off: usize = 0;
            for (b.sign_widths) |w| {
                if (w == width) return b.signs[off..][0..width];
                off += w;
            }
            return error.Unsupported;
        }
        return null;
    }

    fn layerName(b: *const Binder, il: usize, comptime suffix: []const u8) Error![]const u8 {
        return std.fmt.allocPrint(b.arena, "blk.{d}." ++ suffix, .{il});
    }

    fn hc(b: *Binder, il: usize, comptime which: []const u8, h: Hparams) Error!Hc {
        const hd = h.hcDim();
        return .{
            .norm = try b.f32s(try b.layerName(il, "hc_" ++ which ++ "_norm.weight"), hd),
            .down = try b.matrix(try b.layerName(il, "hc_" ++ which ++ "_down.weight"), hd, h.hc_rank),
            .up = try b.matrix(try b.layerName(il, "hc_" ++ which ++ "_up.weight"), h.hc_rank, hd),
            .inject = try b.matrix(try b.layerName(il, "hc_" ++ which ++ "_inject.weight"), hd, h.hc),
        };
    }

    fn bind(b: *Binder) Error!Model {
        const arch_name = try b.value("general.architecture");
        if (arch_name != .string or !std.mem.eql(u8, arch_name.string, arch)) return error.WrongArchitecture;
        const ple_layers = try b.ints(arch ++ ".ple.layers");
        if (ple_layers.len != 1) return error.Unsupported;
        const embd = try b.tensor("token_embd.weight");
        var h: Hparams = .{
            .n_layer = try b.num("block_count"),
            .n_embd = try b.num("embedding_length"),
            .n_head = try b.num("attention.head_count"),
            .n_head_kv = try b.num("attention.head_count_kv"),
            .head_dim = try b.num("attention.key_length"),
            .n_rot = try b.num("rope.dimension_count"),
            .rope_base = try b.float("rope.freq_base"),
            .eps = try b.float("attention.layer_norm_rms_epsilon"),
            .n_expert = try b.num("expert_count"),
            .n_expert_used = try b.num("expert_used_count"),
            .n_ff_exp = try b.num("expert_feed_forward_length"),
            .n_ff_shexp = try b.num("expert_shared_feed_forward_length"),
            .ssm_conv = try b.num("ssm.conv_kernel"),
            .ssm_state = try b.num("ssm.state_size"),
            .ssm_k_heads = try b.num("ssm.group_count"),
            .ssm_v_heads = try b.num("ssm.time_step_rank"),
            .ssm_inner = try b.num("ssm.inner_size"),
            .hc = try b.num("hyper_connection.count"),
            .hc_rank = try b.num("hyper_connection.low_rank"),
            .idx_heads = try b.num("attention.indexer.head_count"),
            .idx_dim = try b.num("attention.indexer.key_length"),
            .idx_top_k = try b.num("attention.indexer.top_k"),
            .ple_layer = ple_layers[0],
            .ple_head_dim = try b.num("embedding_length_per_layer_input"),
            .ple_conv = try b.num("ple.conv_kernel"),
            .ple_dilation = try b.num("ple.ngram_size"),
            .vocab = std.math.cast(u32, embd.dims[1]) orelse return error.BadShape,
            .expert_weights_scale = 0,
        };
        if (b.file.get(arch ++ ".expert_weights_scale") != null) h.expert_weights_scale = try b.float("expert_weights_scale");
        if (h.head_dim != try b.num("attention.value_length")) return error.Unsupported;
        if (h.ssm_inner != h.ssm_state * h.ssm_v_heads or h.ssm_v_heads % h.ssm_k_heads != 0) return error.Unsupported;
        if (h.n_head % h.n_head_kv != 0 or h.n_rot > h.head_dim or h.n_rot % 2 != 0) return error.Unsupported;
        if (ple.heads * h.ple_head_dim != h.n_embd or h.hc < 2 or h.hcDim() > kernels.max_cols) return error.Unsupported;
        if (h.n_expert > 512 or h.ple_layer >= h.n_layer) return error.Unsupported;
        // Text positions are the same in every M-RoPE section, so it
        // reduces to NeoX ordering over the sections' total.
        var section_sum: u32 = 0;
        for (try b.ints(arch ++ ".rope.dimension_sections")) |s| section_sum += s;
        if (section_sum * 2 != h.n_rot) return error.Unsupported;

        const ratios = try b.ints(arch ++ ".attention.compress_ratios");
        if (ratios.len < h.n_layer) return error.BadMetadata;
        const interval = if (b.file.get(arch ++ ".full_attention_interval") != null) try b.num("full_attention_interval") else 4;

        // Rotated weights store W·D·H.
        const names = try b.value("prism.hadamard.weight_names");
        if (names != .array or names.array.elem_type != .string) return error.BadMetadata;
        var it = names.array.iterator();
        var rotated: std.ArrayList([]const u8) = .empty;
        while (it.nextString()) |s| try rotated.append(b.arena, s);
        b.rotated = rotated.items;
        b.sign_widths = try b.ints("prism.hadamard.sign_widths");
        const sv = try b.value("prism.hadamard.sign_values");
        if (sv != .array or sv.array.elem_type != .i32) return error.BadMetadata;
        const signs = try b.arena.alloc(f32, @intCast(sv.array.len));
        for (signs, 0..) |*s, i| s.* = @floatFromInt(@as(i32, @bitCast(sv.array.get(u32, i))));
        b.signs = signs;
        const block = try b.value("prism.hadamard.block_size");
        if (!(block == .u32 and block.u32 == 128) and !(block == .i32 and block.i32 == 128)) return error.Unsupported;
        for (b.rotated) |name| {
            const expert = std.mem.endsWith(u8, name, ".ffn_gate_exps.weight") or std.mem.endsWith(u8, name, ".ffn_up_exps.weight") or std.mem.endsWith(u8, name, ".ffn_down_exps.weight");
            if (!expert) return error.Unsupported;
        }

        const hd = h.hcDim();
        const table = try b.tensor("per_layer_token_embd.weight");
        const m: Model = .{
            .h = h,
            .ple_config = try ple.Config.fromGguf(b.file),
            .token_embd = try b.matrix("token_embd.weight", h.n_embd, h.vocab),
            .output = if (b.file.tensor("output.weight") != null) try b.matrix("output.weight", h.n_embd, h.vocab) else try b.matrix("token_embd.weight", h.n_embd, h.vocab),
            .hc_head = .{
                .norm = try b.f32s("output_hc_norm.weight", hd),
                .down = try b.matrix("output_hc_down.weight", hd, h.hc_rank),
                .up = try b.matrix("output_hc_up.weight", h.hc_rank, hd),
                .inject = null,
            },
            .ple_table = try b.matrix("per_layer_token_embd.weight", h.ple_head_dim, @intCast(table.dims[1])),
            .layers = try b.arena.alloc(Layer, h.n_layer),
            .arena = undefined,
        };
        for (m.layers, 0..) |*l, il| {
            const recurrent = (il + 1) % interval != 0;
            l.hc_attn = try b.hc(il, "attn", h);
            l.hc_ffn = try b.hc(il, "ffn", h);
            if (recurrent) {
                const ch = h.convChannels();
                l.mixer = .{ .linear = .{
                    .qkv = try b.matrix(try b.layerName(il, "attn_qkv.weight"), h.n_embd, ch),
                    .z = try b.matrix(try b.layerName(il, "attn_gate.weight"), h.n_embd, h.ssm_inner),
                    .beta = try b.matrix(try b.layerName(il, "ssm_beta.weight"), h.n_embd, h.ssm_v_heads),
                    .alpha = try b.matrix(try b.layerName(il, "ssm_alpha.weight"), h.n_embd, h.ssm_v_heads),
                    .out = try b.matrix(try b.layerName(il, "ssm_out.weight"), h.ssm_inner, h.n_embd),
                    .conv = try b.f32s(try b.layerName(il, "ssm_conv1d.weight"), h.ssm_conv * ch),
                    .dt_bias = try b.f32s(try b.layerName(il, "ssm_dt.bias"), h.ssm_v_heads),
                    .a = try b.f32s(try b.layerName(il, "ssm_a"), h.ssm_v_heads),
                    .norm = try b.f32s(try b.layerName(il, "ssm_norm.weight"), h.ssm_state),
                } };
            } else {
                if (ratios[il] == 0) return error.Unsupported;
                l.mixer = .{ .full = .{
                    .wq = try b.matrix(try b.layerName(il, "attn_q.weight"), h.n_embd, h.head_dim * h.n_head * 2),
                    .wk = try b.matrix(try b.layerName(il, "attn_k.weight"), h.n_embd, h.head_dim * h.n_head_kv),
                    .wv = try b.matrix(try b.layerName(il, "attn_v.weight"), h.n_embd, h.head_dim * h.n_head_kv),
                    .wo = try b.matrix(try b.layerName(il, "attn_output.weight"), h.head_dim * h.n_head, h.n_embd),
                    .q_norm = try b.f32s(try b.layerName(il, "attn_q_norm.weight"), h.head_dim),
                    .k_norm = try b.f32s(try b.layerName(il, "attn_k_norm.weight"), h.head_dim),
                    .idx_q = try b.matrix(try b.layerName(il, "indexer.q_proj.weight"), h.n_embd, h.idx_heads * h.idx_dim),
                    .idx_k = try b.matrix(try b.layerName(il, "indexer.k_proj.weight"), h.n_embd, h.idx_dim),
                    .idx_q_norm = try b.f32s(try b.layerName(il, "indexer.q_norm.weight"), h.idx_dim),
                    .idx_k_norm = try b.f32s(try b.layerName(il, "indexer.k_norm.weight"), h.idx_dim),
                    .compress_ratio = ratios[il],
                } };
            }
            l.ple = if (il == h.ple_layer) blk: {
                if (!recurrent) return error.Unsupported;
                break :blk .{
                    .key = try b.matrix(try b.layerName(il, "ple_key.weight"), h.n_embd, hd),
                    .value = try b.matrix(try b.layerName(il, "ple_value.weight"), h.n_embd, h.n_embd),
                    .norm_key = try b.f32s(try b.layerName(il, "ple_norm_key.weight"), hd),
                    .norm_query = try b.f32s(try b.layerName(il, "ple_norm_query.weight"), hd),
                    .norm_conv = try b.f32s(try b.layerName(il, "ple_norm_conv.weight"), hd),
                    .conv = try b.f32s(try b.layerName(il, "ple_conv1d.weight"), h.ple_conv * hd),
                };
            } else null;
            l.router = try b.matrix(try b.layerName(il, "ffn_gate_inp.weight"), h.n_embd, h.n_expert);
            const gate = try b.layerName(il, "ffn_gate_exps.weight");
            const up = try b.layerName(il, "ffn_up_exps.weight");
            const down = try b.layerName(il, "ffn_down_exps.weight");
            l.gate_exps = try b.matrix(gate, h.n_embd, h.n_ff_exp * h.n_expert);
            l.up_exps = try b.matrix(up, h.n_embd, h.n_ff_exp * h.n_expert);
            l.down_exps = try b.matrix(down, h.n_ff_exp, h.n_embd * h.n_expert);
            l.gate_signs = try b.signsOf(gate, h.n_embd);
            l.up_signs = try b.signsOf(up, h.n_embd);
            l.down_signs = try b.signsOf(down, h.n_ff_exp);
            // Gate and up share one rotated input.
            if ((l.gate_signs == null) != (l.up_signs == null)) return error.Unsupported;
            if (l.gate_signs) |g| if (!std.mem.eql(f32, g, l.up_signs.?)) return error.Unsupported;
            l.sh_gate = try b.matrix(try b.layerName(il, "ffn_gate_shexp.weight"), h.n_embd, h.n_ff_shexp);
            l.sh_up = try b.matrix(try b.layerName(il, "ffn_up_shexp.weight"), h.n_embd, h.n_ff_shexp);
            l.sh_down = try b.matrix(try b.layerName(il, "ffn_down_shexp.weight"), h.n_ff_shexp, h.n_embd);
            l.sh_gate_inp = try b.f32s(try b.layerName(il, "ffn_gate_inp_shexp.weight"), h.n_embd);
        }
        return m;
    }
};
