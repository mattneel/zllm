//! zLLM Command-Line Interface
//!
//! Provides commands for:
//! - Model inference
//! - Model training
//! - Checkpoint conversion
//! - Benchmarking

const std = @import("std");
const zllm = @import("zllm");

pub fn main() !void {
    std.debug.print("zLLM CLI v0.1.0\n", .{});
    std.debug.print("Usage: zllm <command> [options]\n", .{});
    std.debug.print("\nCommands:\n", .{});
    std.debug.print("  generate  Generate text from a prompt\n", .{});
    std.debug.print("  train     Train a model\n", .{});
    std.debug.print("  convert   Convert model checkpoints\n", .{});
    std.debug.print("  bench     Run benchmarks\n", .{});
}
