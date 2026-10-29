//! Run-time library loading: `std.DynLib` (dlopen) on POSIX, kernel32 on
//! Windows, which std no longer covers.

const std = @import("std");
const builtin = @import("builtin");

pub const Error = error{ LibraryNotFound, NameTooLong };

pub const Lib = if (builtin.os.tag == .windows) Windows else Posix;

const Posix = struct {
    inner: std.DynLib,

    pub fn open(path: []const u8) Error!Posix {
        return .{ .inner = std.DynLib.open(path) catch return error.LibraryNotFound };
    }

    pub fn close(l: *Posix) void {
        l.inner.close();
    }

    pub fn lookup(l: *Posix, comptime T: type, name: [:0]const u8) ?T {
        return l.inner.lookup(T, name);
    }
};

const Windows = struct {
    module: *anyopaque,

    const HMODULE = *anyopaque;
    extern "kernel32" fn LoadLibraryW(name: [*:0]const u16) callconv(.winapi) ?HMODULE;
    extern "kernel32" fn GetProcAddress(module: HMODULE, name: [*:0]const u8) callconv(.winapi) ?*const anyopaque;
    extern "kernel32" fn FreeLibrary(module: HMODULE) callconv(.winapi) i32;

    /// A bare name searches the executable's directory first, then the
    /// system directories.
    pub fn open(path: []const u8) Error!Windows {
        var wide: [std.fs.max_path_bytes:0]u16 = undefined;
        const n = std.unicode.wtf8ToWtf16Le(&wide, path) catch return error.NameTooLong;
        if (n >= wide.len) return error.NameTooLong;
        wide[n] = 0;
        return .{ .module = LoadLibraryW(wide[0..n :0]) orelse return error.LibraryNotFound };
    }

    pub fn close(l: *Windows) void {
        _ = FreeLibrary(l.module);
    }

    pub fn lookup(l: *Windows, comptime T: type, name: [:0]const u8) ?T {
        return @ptrCast(@alignCast(GetProcAddress(l.module, name) orelse return null));
    }
};
