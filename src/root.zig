//! zllm: LLM inference on WebGPU from one Zig codebase.

pub const gguf = @import("gguf/gguf.zig");
pub const tokenizer = @import("tokenizer/tokenizer.zig");
pub const graph = @import("graph/graph.zig");
pub const ref = @import("ref/ref.zig");
pub const gpu = @import("gpu/gpu.zig");
pub const kernels = @import("kernels/kernels.zig");
pub const kernel_check = @import("kernels/check.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
    _ = @import("tokenizer/unicode.zig");
    _ = @import("tokenizer/pretokenize.zig");
    _ = ref.ops;
    _ = ref.tensor;
    _ = @import("ref/pool.zig");
    _ = ref.matmul;
    _ = gpu.cmd;
}
