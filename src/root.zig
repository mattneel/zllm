//! zllm: LLM inference on WebGPU from one Zig codebase.

pub const version = @import("version.zig").zllm;

pub const gguf = @import("gguf/gguf.zig");
pub const tokenizer = @import("tokenizer/tokenizer.zig");
pub const graph = @import("graph/graph.zig");
pub const ref = @import("ref/ref.zig");
pub const gpu = @import("gpu/gpu.zig");
pub const tmpl = @import("tmpl/tmpl.zig");
pub const kernels = @import("kernels/kernels.zig");
pub const tune = @import("tune/tune.zig");
pub const kernel_check = @import("kernels/check.zig");
pub const plan = @import("plan/plan.zig");
pub const fuse = @import("fuse/fuse.zig");
pub const backend = @import("backend/backend.zig");
pub const sampler = @import("sampler/sampler.zig");
pub const state = @import("state/state.zig");
pub const sched = @import("sched/sched.zig");
pub const chat = @import("chat/chat.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
    _ = @import("tokenizer/unicode.zig");
    _ = @import("tokenizer/pretokenize.zig");
    _ = @import("sampler/defaults.zig");
    _ = @import("plan/cache.zig");
    _ = ref.ops;
    _ = ref.tensor;
    _ = ref.gyro;
    _ = ref.qwen4exp.ple;
    _ = ref.qwen4exp.kernels;
    _ = ref.qwen4exp.engine;
    _ = @import("ref/pool.zig");
    _ = ref.matmul;
    _ = gpu.cmd;
    _ = @import("tokenizer/stream.zig");
    _ = @import("chat/jinja.zig");
    _ = @import("fuse/fuse.zig");
    _ = @import("velocity/kernels.zig");
    _ = @import("velocity/hazards.zig");
    _ = @import("ref/fp8.zig");
}
