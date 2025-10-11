//! Training utilities for zLLM
//!
//! Provides:
//! - Optimizers (SGD, Adam, AdamW)
//! - Learning rate schedules
//! - Gradient computation
//! - Checkpoint management
//!
//! All training operations use pre-allocated buffers
//! and fixed-size structures for deterministic memory usage.

const std = @import("std");
const core = @import("core");

/// Error set for training operations
pub const TrainingError = error{
    GradientOverflow,
    InvalidLearningRate,
    CheckpointFailed,
    OutOfMemory,
};

test {
    std.testing.refAllDecls(@This());
}

test "training module compiles" {
    try std.testing.expect(true);
}
