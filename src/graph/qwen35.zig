//! Qwen 3.5 hybrid architecture (GGUF `general.architecture = qwen35`):
//! hyperparameters and layer schedule read from GGUF metadata.
//!
//! Tensor semantics (what the converter already applied to the stored
//! weights) are documented in docs/src/qwen35.md.

const std = @import("std");
const gguf = @import("../gguf/gguf.zig");

pub const arch = "qwen35";

pub const LayerKind = enum {
    /// Gated DeltaNet (linear attention with a recurrent state).
    delta_net,
    /// Full softmax attention with a KV cache.
    attention,
};

pub const max_layers = 128;

pub const Hparams = struct {
    vocab: u32,
    n_embd: u32,
    n_ff: u32,
    /// Decoder layers run by the main forward pass. GGUF `block_count` also
    /// counts the trailing MTP (`nextn`) layers, which are excluded here.
    n_layer: u32,
    n_layer_nextn: u32,
    rms_eps: f32,

    // Full attention.
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    /// Rotary dimensions per head; the rest pass through unrotated.
    n_rot: u32,
    rope_base: f32,
    rope_sections: [4]u32,

    // Gated DeltaNet.
    conv_kernel: u32,
    /// Per-head key (and value) width; GGUF `ssm.state_size`.
    gdn_head_dim: u32,
    /// GGUF `ssm.group_count`.
    gdn_k_heads: u32,
    /// GGUF `ssm.time_step_rank`.
    gdn_v_heads: u32,
    /// `gdn_v_heads * gdn_head_dim`; GGUF `ssm.inner_size`.
    gdn_inner: u32,

    layer_kinds: [max_layers]LayerKind,

    pub const Error = error{ WrongArchitecture, MissingMetadata, BadMetadata };

    pub fn fromGguf(f: gguf.File) Error!Hparams {
        const a = getString(f, "general.architecture") orelse return error.MissingMetadata;
        if (!std.mem.eql(u8, a, arch)) return error.WrongArchitecture;

        const block_count = try getU32(f, "block_count");
        const nextn = getU32(f, "nextn_predict_layers") catch 0;
        if (nextn >= block_count or block_count - nextn > max_layers) return error.BadMetadata;

        var h: Hparams = .{
            .vocab = 0,
            .n_embd = try getU32(f, "embedding_length"),
            .n_ff = try getU32(f, "feed_forward_length"),
            .n_layer = block_count - nextn,
            .n_layer_nextn = nextn,
            .rms_eps = try getF32(f, "attention.layer_norm_rms_epsilon"),
            .n_head = try getU32(f, "attention.head_count"),
            .n_head_kv = try getU32(f, "attention.head_count_kv"),
            .head_dim = try getU32(f, "attention.key_length"),
            .n_rot = try getU32(f, "rope.dimension_count"),
            .rope_base = try getF32(f, "rope.freq_base"),
            .rope_sections = @splat(0),
            .conv_kernel = try getU32(f, "ssm.conv_kernel"),
            .gdn_head_dim = try getU32(f, "ssm.state_size"),
            .gdn_k_heads = try getU32(f, "ssm.group_count"),
            .gdn_v_heads = try getU32(f, "ssm.time_step_rank"),
            .gdn_inner = try getU32(f, "ssm.inner_size"),
            .layer_kinds = undefined,
        };

        const tokens = f.get("tokenizer.ggml.tokens") orelse return error.MissingMetadata;
        if (tokens != .array) return error.BadMetadata;
        h.vocab = std.math.cast(u32, tokens.array.len) orelse return error.BadMetadata;

        if (try getU32(f, "attention.value_length") != h.head_dim) return error.BadMetadata;
        if (h.n_head_kv == 0 or h.n_head % h.n_head_kv != 0) return error.BadMetadata;
        if (h.n_rot > h.head_dim or h.n_rot % 2 != 0) return error.BadMetadata;
        if (h.gdn_k_heads == 0 or h.gdn_v_heads % h.gdn_k_heads != 0) return error.BadMetadata;
        if (h.gdn_inner != h.gdn_v_heads * h.gdn_head_dim) return error.BadMetadata;

        if (f.get(key(arch, "rope.dimension_sections"))) |v| {
            if (v != .array or v.array.len != 4) return error.BadMetadata;
            for (&h.rope_sections, 0..) |*s, i| s.* = v.array.get(u32, i);
        }

        // Explicit per-layer flags when present, else every
        // `full_attention_interval`-th layer is full attention.
        if (f.get(key(arch, "attention.recurrent_layers"))) |v| {
            if (v != .array or v.array.len < h.n_layer) return error.BadMetadata;
            for (h.layer_kinds[0..h.n_layer], 0..) |*k, i| {
                k.* = if (v.array.bytes[i] != 0) .delta_net else .attention;
            }
        } else {
            const interval = getU32(f, "full_attention_interval") catch 4;
            if (interval == 0) return error.BadMetadata;
            for (h.layer_kinds[0..h.n_layer], 0..) |*k, i| {
                k.* = if ((i + 1) % interval == 0) .attention else .delta_net;
            }
        }
        return h;
    }

    pub fn kinds(h: *const Hparams) []const LayerKind {
        return h.layer_kinds[0..h.n_layer];
    }
};

fn key(comptime a: []const u8, comptime k: []const u8) []const u8 {
    return a ++ "." ++ k;
}

fn getString(f: gguf.File, k: []const u8) ?[]const u8 {
    const v = f.get(k) orelse return null;
    return if (v == .string) v.string else null;
}

fn getU32(f: gguf.File, comptime k: []const u8) Hparams.Error!u32 {
    const v = f.get(key(arch, k)) orelse return error.MissingMetadata;
    return switch (v) {
        .u32 => |x| x,
        .i32 => |x| std.math.cast(u32, x) orelse error.BadMetadata,
        .u64 => |x| std.math.cast(u32, x) orelse error.BadMetadata,
        else => error.BadMetadata,
    };
}

fn getF32(f: gguf.File, comptime k: []const u8) Hparams.Error!f32 {
    const v = f.get(key(arch, k)) orelse return error.MissingMetadata;
    return switch (v) {
        .f32 => |x| x,
        else => error.BadMetadata,
    };
}
