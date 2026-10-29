//! zllm: LLM inference on WebGPU from one Zig codebase.

pub const gguf = @import("gguf/gguf.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
