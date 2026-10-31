//! GPU layer: the command stream the core emits, and the platform layer
//! (host.zig) whose devices and hosts run it natively on every backend. The
//! browser's host is web/src/zllm/gpu.ts.

pub const cmd = @import("cmd.zig");
pub const dawn = @import("dawn.zig");
const host = @import("host.zig");
pub const Backend = host.Backend;
pub const Device = host.Device;
pub const Host = host.Host;
pub const Open = host.Open;
pub const Limits = host.Limits;
pub const Adapter = host.Adapter;
pub const Timings = @import("timings.zig").Timings;
