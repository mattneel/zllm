//! GPU layer: the command stream the core emits, and the hosts that run it.
//! `dawn` + `host` are the native host (tests and benchmarks); the browser
//! host arrives with M3.

pub const cmd = @import("cmd.zig");
pub const dawn = @import("dawn.zig");
pub const Host = @import("host.zig").Host;
