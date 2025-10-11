//! zLLM - Zero-allocation Large Language Models in Zig
//!
//! A high-performance LLM inference and training library following Tiger Style:
//! - Safe: Fixed-size buffers, explicit bounds, no recursion
//! - Fast: SIMD operations, cache-friendly layout, zero allocations
//! - Clear: Comptime configuration, explicit errors, self-documenting
//!
//! Usage:
//! ```zig
//! const zllm = @import("zllm");
//!
//! // All configuration is done at compile time
//! const model_config = zllm.models.GPT2Config{
//!     .vocab_size = 50257,
//!     .n_layers = 12,
//!     .n_heads = 12,
//!     .d_model = 768,
//! };
//! ```

const std = @import("std");

pub const core = @import("core");
pub const models = @import("models");
pub const tokenizers = @import("tokenizers");
pub const training = @import("training");

test {
    std.testing.refAllDecls(@This());
}
