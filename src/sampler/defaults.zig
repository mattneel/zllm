//! A model's default sampling: what a request leaves unset takes.
//!
//! The base is the architecture's model-card settings, for chat with a
//! reasoning block and without: Qwen 3.5's for `qwen35` (Qwen 3.5, 3.6 and
//! 3.8 dense), plain sampling for the rest. The GGUF's own recommendation,
//! llama.cpp's `general.sampling.*` keys, then replaces the fields it sets,
//! in every mode; raw continuations take it over plain sampling, as vLLM
//! applies a model's generation config to every request. Of those keys,
//! zllm's sampler takes temp, top_k, top_p, min_p and penalty_repeat; the
//! others (sequence, penalty_last_n, xtc_*, mirostat*) are ignored, as is a
//! value out of the sampler's range.

const std = @import("std");
const gguf = @import("../gguf/gguf.zig");
const Params = @import("sampler.zig").Params;

pub const Defaults = struct {
    /// Chat with a reasoning block, and without.
    thinking: Params,
    instruct: Params,
    /// Raw continuations (a prompt or token ids, no chat template).
    raw: Params,

    /// Qwen 3.5's model card, for text tasks.
    const qwen_thinking: Params = .{ .temperature = 1.0, .top_p = 0.95, .top_k = 20, .presence_penalty = 1.5 };
    const qwen_instruct: Params = .{ .temperature = 1.0, .top_p = 1.0, .top_k = 20, .presence_penalty = 2.0 };

    pub fn fromGguf(f: gguf.File) Defaults {
        const arch = f.get("general.architecture");
        const qwen = arch != null and arch.? == .string and std.mem.eql(u8, arch.?.string, "qwen35");
        return .{
            .thinking = recommended(f, if (qwen) qwen_thinking else .{}),
            .instruct = recommended(f, if (qwen) qwen_instruct else .{}),
            .raw = recommended(f, .{}),
        };
    }

    pub fn chat(d: Defaults, thinking: bool) Params {
        return if (thinking) d.thinking else d.instruct;
    }
};

/// `base` with the fields the GGUF recommends in range.
fn recommended(f: gguf.File, base: Params) Params {
    var p = base;
    if (float(f, "temp")) |v| if (v >= 0) {
        p.temperature = v;
    };
    if (int(f, "top_k")) |v| {
        p.top_k = v;
    }
    if (float(f, "top_p")) |v| if (v > 0 and v <= 1) {
        p.top_p = v;
    };
    if (float(f, "min_p")) |v| if (v >= 0 and v <= 1) {
        p.min_p = v;
    };
    if (float(f, "penalty_repeat")) |v| if (v > 0) {
        p.repetition_penalty = v;
    };
    return p;
}

fn float(f: gguf.File, comptime key: []const u8) ?f32 {
    const v: f32 = switch (f.get("general.sampling." ++ key) orelse return null) {
        .f32 => |x| x,
        .f64 => |x| @floatCast(x),
        else => return null,
    };
    return if (std.math.isFinite(v)) v else null;
}

fn int(f: gguf.File, comptime key: []const u8) ?u32 {
    return switch (f.get("general.sampling." ++ key) orelse return null) {
        .i32 => |x| std.math.cast(u32, x),
        .u32 => |x| x,
        .i64 => |x| std.math.cast(u32, x),
        .u64 => |x| std.math.cast(u32, x),
        else => null,
    };
}

fn testFile(metadata: []gguf.KeyValue) gguf.File {
    return .{ .version = 3, .alignment = 32, .metadata = metadata, .tensors = &.{}, .data_offset = 0 };
}

test "the GGUF's recommendation replaces what it sets, on the model card" {
    var md = [_]gguf.KeyValue{
        .{ .key = "general.architecture", .value = .{ .string = "qwen35" } },
        .{ .key = "general.sampling.temp", .value = .{ .f32 = 0.6 } },
        .{ .key = "general.sampling.top_k", .value = .{ .i32 = 40 } },
    };
    const d: Defaults = .fromGguf(testFile(&md));
    // The card's top_p and presence penalty stay; temp and top_k change.
    try std.testing.expectEqual(@as(f32, 0.6), d.thinking.temperature);
    try std.testing.expectEqual(@as(u32, 40), d.thinking.top_k);
    try std.testing.expectEqual(@as(f32, 0.95), d.thinking.top_p);
    try std.testing.expectEqual(@as(f32, 1.5), d.thinking.presence_penalty);
    try std.testing.expectEqual(@as(f32, 2.0), d.chat(false).presence_penalty);
    // Raw continuations: the recommendation over plain sampling.
    try std.testing.expectEqual(@as(f32, 0.6), d.raw.temperature);
    try std.testing.expectEqual(@as(u32, 40), d.raw.top_k);
    try std.testing.expectEqual(@as(f32, 0), d.raw.presence_penalty);
}

test "an architecture without a card takes only the GGUF's recommendation" {
    var md = [_]gguf.KeyValue{
        .{ .key = "general.architecture", .value = .{ .string = "qwen4exp" } },
        .{ .key = "general.sampling.top_p", .value = .{ .f32 = 0.95 } },
        // Out of the sampler's range: ignored.
        .{ .key = "general.sampling.min_p", .value = .{ .f32 = 2 } },
        // Not one the sampler takes.
        .{ .key = "general.sampling.penalty_last_n", .value = .{ .i32 = 64 } },
    };
    const d: Defaults = .fromGguf(testFile(&md));
    for ([_]Params{ d.thinking, d.instruct, d.raw }) |p| {
        try std.testing.expectEqual(@as(f32, 0.95), p.top_p);
        try std.testing.expectEqual(@as(f32, 0), p.min_p);
        try std.testing.expectEqual(@as(f32, 0), p.presence_penalty);
        try std.testing.expectEqual(@as(u32, 0), p.top_k);
    }
}

test "without a recommendation, the card for chat and plain sampling raw" {
    var md = [_]gguf.KeyValue{.{ .key = "general.architecture", .value = .{ .string = "qwen35" } }};
    const d: Defaults = .fromGguf(testFile(&md));
    try std.testing.expectEqual(Defaults.qwen_thinking.presence_penalty, d.thinking.presence_penalty);
    try std.testing.expectEqual(Defaults.qwen_instruct.top_p, d.instruct.top_p);
    try std.testing.expectEqual(@as(u32, 0), d.raw.top_k);
}
