//! Model architectures for zLLM
//!
//! Supported architectures:
//! - GPT-2 family (124M, 355M, 774M, 1.5B)
//! - LLaMA family (7B, 13B, 70B)
//!
//! Performance targets:
//! - GPT-2 124M: 45 tokens/sec (M2, INT8)
//! - LLaMA 7B: 8 tokens/sec (M2, INT4)
//!
//! Each model is configured at compile time using comptime parameters
//! to eliminate runtime overhead and enable aggressive optimization.

const std = @import("std");
const core = @import("core");

/// Error set for model operations
pub const ModelError = error{
    UnsupportedArchitecture,
    InvalidConfiguration,
    CheckpointCorrupted,
    InvalidTokenId,
    ContextLengthExceeded,
};

test {
    std.testing.refAllDecls(@This());
}

test "models module compiles" {
    try std.testing.expect(true);
}
