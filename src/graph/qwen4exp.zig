//! Qwen4Exp (Qwen3.8-Flash-Next) hyperparameters from GGUF metadata alone,
//! as the browser has only the header: what the reference and the GPU
//! program both build from.

const std = @import("std");
const gguf = @import("../gguf/gguf.zig");

pub const arch = "qwen4exp";

pub const Error = error{ WrongArchitecture, MissingMetadata, BadMetadata, MissingTensor, Unsupported };

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

    /// Layer kinds: every `interval`-th layer has full attention.
    interval: u32,

    pub fn recurrent(h: Hparams, il: usize) bool {
        return (il + 1) % h.interval != 0;
    }

    pub fn hcDim(h: Hparams) u32 {
        return h.hc * h.n_embd;
    }
    pub fn convChannels(h: Hparams) u32 {
        return h.ssm_state * h.ssm_k_heads * 2 + h.ssm_state * h.ssm_v_heads;
    }

    pub fn fromGguf(f: gguf.File) Error!Hparams {
        const name = f.get("general.architecture") orelse return error.MissingMetadata;
        if (name != .string or !std.mem.eql(u8, name.string, arch)) return error.WrongArchitecture;
        const embd = f.tensor("token_embd.weight") orelse return error.MissingTensor;
        const ple_layers = f.get(arch ++ ".ple.layers") orelse return error.MissingMetadata;
        if (ple_layers != .array or ple_layers.array.len != 1) return error.Unsupported;
        if (ple_layers.array.elem_type != .i32 and ple_layers.array.elem_type != .u32) return error.BadMetadata;
        var h: Hparams = .{
            .n_layer = try num(f, "block_count"),
            .n_embd = try num(f, "embedding_length"),
            .n_head = try num(f, "attention.head_count"),
            .n_head_kv = try num(f, "attention.head_count_kv"),
            .head_dim = try num(f, "attention.key_length"),
            .n_rot = try num(f, "rope.dimension_count"),
            .rope_base = try float(f, "rope.freq_base"),
            .eps = try float(f, "attention.layer_norm_rms_epsilon"),
            .n_expert = try num(f, "expert_count"),
            .n_expert_used = try num(f, "expert_used_count"),
            .n_ff_exp = try num(f, "expert_feed_forward_length"),
            .n_ff_shexp = try num(f, "expert_shared_feed_forward_length"),
            .ssm_conv = try num(f, "ssm.conv_kernel"),
            .ssm_state = try num(f, "ssm.state_size"),
            .ssm_k_heads = try num(f, "ssm.group_count"),
            .ssm_v_heads = try num(f, "ssm.time_step_rank"),
            .ssm_inner = try num(f, "ssm.inner_size"),
            .hc = try num(f, "hyper_connection.count"),
            .hc_rank = try num(f, "hyper_connection.low_rank"),
            .idx_heads = try num(f, "attention.indexer.head_count"),
            .idx_dim = try num(f, "attention.indexer.key_length"),
            .idx_top_k = try num(f, "attention.indexer.top_k"),
            .ple_layer = ple_layers.array.get(u32, 0),
            .ple_head_dim = try num(f, "embedding_length_per_layer_input"),
            .ple_conv = try num(f, "ple.conv_kernel"),
            .ple_dilation = try num(f, "ple.ngram_size"),
            .vocab = std.math.cast(u32, embd.dims[1]) orelse return error.BadMetadata,
            .expert_weights_scale = if (f.get(arch ++ ".expert_weights_scale") != null) try float(f, "expert_weights_scale") else 0,
            .interval = if (f.get(arch ++ ".full_attention_interval") != null) try num(f, "full_attention_interval") else 4,
        };
        if (h.head_dim != try num(f, "attention.value_length")) return error.Unsupported;
        if (h.ssm_inner != h.ssm_state * h.ssm_v_heads or h.ssm_v_heads % h.ssm_k_heads != 0) return error.Unsupported;
        if (h.n_head % h.n_head_kv != 0 or h.n_rot > h.head_dim or h.n_rot % 2 != 0) return error.Unsupported;
        if (16 * h.ple_head_dim != h.n_embd or h.hc < 2 or h.interval == 0) return error.Unsupported;
        if (h.n_expert > 512 or h.ple_layer >= h.n_layer or !h.recurrent(h.ple_layer)) return error.Unsupported;
        // Text positions are the same in every M-RoPE section, so it
        // reduces to NeoX ordering over the sections' total.
        const sections = f.get(arch ++ ".rope.dimension_sections") orelse return error.MissingMetadata;
        if (sections != .array) return error.BadMetadata;
        var sum: u32 = 0;
        for (0..@intCast(sections.array.len)) |k| sum += sections.array.get(u32, k);
        if (sum * 2 != h.n_rot) return error.Unsupported;
        return h;
    }
};

fn num(f: gguf.File, comptime key: []const u8) Error!u32 {
    return switch (f.get(arch ++ "." ++ key) orelse return error.MissingMetadata) {
        .u32 => |x| x,
        .i32 => |x| std.math.cast(u32, x) orelse error.BadMetadata,
        .u64 => |x| std.math.cast(u32, x) orelse error.BadMetadata,
        else => error.BadMetadata,
    };
}

fn float(f: gguf.File, comptime key: []const u8) Error!f32 {
    return switch (f.get(arch ++ "." ++ key) orelse return error.MissingMetadata) {
        .f32 => |x| x,
        else => error.BadMetadata,
    };
}
