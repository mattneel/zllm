//! CPU f32 reference implementation: the correctness oracle for GPU kernels.
//! Test and validation use only; zllm has no CPU inference path.

pub const ops = @import("ops.zig");
pub const tensor = @import("tensor.zig");
pub const qwen35 = @import("qwen35.zig");
