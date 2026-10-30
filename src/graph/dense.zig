//! Dense decoder-only transformers (GGUF `general.architecture` = `qwen3`,
//! `llama` or `granite`): every layer is RMSNorm → GQA softmax attention →
//! residual → RMSNorm → SwiGLU FFN → residual. The architectures differ in
//! a handful of switches, all read here (llama.cpp's graphs in
//! src/models/{qwen3,llama,granite}.cpp are the authority):
//!
//! - Qwen 3: per-head Q and K RMSNorm before RoPE; NeoX RoPE (pairs i and
//!   i + n_rot/2).
//! - Llama: normal RoPE (pairs 2i and 2i + 1), divided per pair by the
//!   `rope_freqs` factors when the file has them (Llama 3.1+ scaling).
//! - Granite: normal RoPE; embeddings × `embedding_scale`, each branch
//!   × `residual_scale` before its residual add, K·Q × `attention.scale`
//!   instead of 1/√d, logits ÷ `logit_scale`. llama.cpp skips each when 0.
//!
//! The LM head is `output.weight`, or `token_embd.weight` when tied.

const std = @import("std");
const gguf = @import("../gguf/gguf.zig");

pub const Arch = enum {
    qwen3,
    llama,
    granite,

    pub fn fromName(name: []const u8) ?Arch {
        return std.meta.stringToEnum(Arch, name);
    }
};

pub const Rope = enum {
    /// Pairs (i, i + n_rot/2): Qwen.
    neox,
    /// Pairs (2i, 2i + 1): Llama, Granite.
    normal,
};

pub const Hparams = struct {
    arch: Arch,
    vocab: u32,
    n_embd: u32,
    n_ff: u32,
    n_layer: u32,
    rms_eps: f32,
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    /// Rotary dimensions per head; the rest pass through unrotated.
    n_rot: u32,
    rope_base: f32,
    rope: Rope,
    /// Per-head Q and K RMSNorm (`attn_q_norm`, `attn_k_norm`).
    qk_norm: bool,
    /// Multiplies K·Q before the softmax.
    attn_scale: f32,
    /// Multiplies the token embeddings.
    embd_scale: f32,
    /// Multiplies each attention and FFN output before its residual add.
    residual_scale: f32,
    /// Multiplies the logits (1 / GGUF `logit_scale`).
    logit_mul: f32,

    pub const Error = error{ WrongArchitecture, MissingMetadata, BadMetadata };

    pub fn fromGguf(f: gguf.File) Error!Hparams {
        const name = getString(f, "general.architecture") orelse return error.MissingMetadata;
        const a = Arch.fromName(name) orelse return error.WrongArchitecture;
        var keys: Keys = .{ .f = f, .arch = name };

        const n_embd = try keys.u32_("embedding_length");
        const n_head = try keys.u32_("attention.head_count");
        if (n_head == 0 or n_embd % n_head != 0) return error.BadMetadata;
        const head_dim = keys.u32_("attention.key_length") catch (n_embd / n_head);
        var h: Hparams = .{
            .arch = a,
            .vocab = 0,
            .n_embd = n_embd,
            .n_ff = try keys.u32_("feed_forward_length"),
            .n_layer = try keys.u32_("block_count"),
            .rms_eps = try keys.f32_("attention.layer_norm_rms_epsilon"),
            .n_head = n_head,
            .n_head_kv = keys.u32_("attention.head_count_kv") catch n_head,
            .head_dim = head_dim,
            .n_rot = keys.u32_("rope.dimension_count") catch head_dim,
            .rope_base = keys.f32_("rope.freq_base") catch 10000,
            .rope = if (a == .qwen3) .neox else .normal,
            .qk_norm = a == .qwen3,
            .attn_scale = 1 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            .embd_scale = 1,
            .residual_scale = 1,
            .logit_mul = 1,
        };
        if (a == .granite) {
            // llama.cpp: a missing or zero multiplier is no multiplier.
            if (nonzero(keys.f32_("attention.scale") catch 0)) |s| h.attn_scale = s;
            if (nonzero(keys.f32_("embedding_scale") catch 0)) |s| h.embd_scale = s;
            if (nonzero(keys.f32_("residual_scale") catch 0)) |s| h.residual_scale = s;
            h.logit_mul = 1 / (nonzero(try keys.f32_("logit_scale")) orelse return error.BadMetadata);
        }

        const tokens = f.get("tokenizer.ggml.tokens") orelse return error.MissingMetadata;
        if (tokens != .array) return error.BadMetadata;
        h.vocab = std.math.cast(u32, tokens.array.len) orelse return error.BadMetadata;

        if ((keys.u32_("attention.value_length") catch h.head_dim) != h.head_dim) return error.BadMetadata;
        if (h.n_head_kv == 0 or h.n_head % h.n_head_kv != 0) return error.BadMetadata;
        if (h.n_rot > h.head_dim or h.n_rot % 2 != 0) return error.BadMetadata;
        if (h.n_layer == 0) return error.BadMetadata;
        return h;
    }

    /// Query heads sharing one KV head.
    pub fn group(h: Hparams) u32 {
        return h.n_head / h.n_head_kv;
    }
};

fn nonzero(x: f32) ?f32 {
    return if (x == 0) null else x;
}

/// `<arch>.<key>` lookups.
const Keys = struct {
    f: gguf.File,
    arch: []const u8,

    fn get(k: *Keys, comptime name: []const u8) ?gguf.Value {
        var buf: [96]u8 = undefined;
        const full = std.fmt.bufPrint(&buf, "{s}." ++ name, .{k.arch}) catch return null;
        return k.f.get(full);
    }

    fn u32_(k: *Keys, comptime name: []const u8) Hparams.Error!u32 {
        return switch (k.get(name) orelse return error.MissingMetadata) {
            .u32 => |x| x,
            .i32 => |x| std.math.cast(u32, x) orelse error.BadMetadata,
            .u64 => |x| std.math.cast(u32, x) orelse error.BadMetadata,
            else => error.BadMetadata,
        };
    }

    fn f32_(k: *Keys, comptime name: []const u8) Hparams.Error!f32 {
        return switch (k.get(name) orelse return error.MissingMetadata) {
            .f32 => |x| x,
            else => error.BadMetadata,
        };
    }
};

fn getString(f: gguf.File, k: []const u8) ?[]const u8 {
    const v = f.get(k) orelse return null;
    return if (v == .string) v.string else null;
}
