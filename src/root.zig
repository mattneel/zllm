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
pub const precision = @import("precision.zig");
pub const chat = @import("chat/chat.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
    _ = @import("tokenizer/unicode.zig");
    _ = @import("tokenizer/pretokenize.zig");
    _ = @import("tokenizer/bpe.zig");
    _ = @import("tokenizer/gemma.zig");
    _ = @import("tokenizer/sentencepiece.zig");
    _ = @import("sampler/defaults.zig");
    _ = @import("plan/cache.zig");
    _ = @import("plan/common.zig");
    _ = @import("graph/dense.zig");
    _ = @import("precision.zig");
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
    _ = @import("agility/kernels.zig");
    _ = @import("kernels/check.zig");
    _ = @import("agility/hazards.zig");
    _ = @import("agility/linalg_test.zig");
    _ = @import("agility/storage.zig");
    _ = @import("ref/fp8.zig");
}
