//! Dawn host: the native WebGPU implementation behind the `webgpu.h` C API.
//!
//! zllm loads Dawn at run time (`libwebgpu_dawn.so`, `webgpu_dawn.dll`) and
//! resolves the C functions it uses into `Procs`, so the binary never links
//! Dawn's C++ and the vendored header (third_party/dawn) pins the ABI. The
//! native host may block: every asynchronous WebGPU call here is waited on.

const std = @import("std");
pub const c = @import("webgpu");
const dynlib = @import("dynlib.zig");

/// Function pointers resolved from the Dawn library. A field `Foo` is the
/// C function `wgpuFoo`; add fields as the host needs more of the API.
pub const Procs = struct {
    CreateInstance: Proc(c.WGPUProcCreateInstance),
    InstanceRequestAdapter: Proc(c.WGPUProcInstanceRequestAdapter),
    InstanceWaitAny: Proc(c.WGPUProcInstanceWaitAny),
    InstanceRelease: Proc(c.WGPUProcInstanceRelease),
    AdapterGetInfo: Proc(c.WGPUProcAdapterGetInfo),
    AdapterInfoFreeMembers: Proc(c.WGPUProcAdapterInfoFreeMembers),
    AdapterGetLimits: Proc(c.WGPUProcAdapterGetLimits),
    AdapterHasFeature: Proc(c.WGPUProcAdapterHasFeature),
    AdapterRequestDevice: Proc(c.WGPUProcAdapterRequestDevice),
    AdapterRelease: Proc(c.WGPUProcAdapterRelease),
    DeviceGetQueue: Proc(c.WGPUProcDeviceGetQueue),
    DeviceGetLimits: Proc(c.WGPUProcDeviceGetLimits),
    DeviceCreateBuffer: Proc(c.WGPUProcDeviceCreateBuffer),
    DeviceCreateShaderModule: Proc(c.WGPUProcDeviceCreateShaderModule),
    DeviceCreateComputePipeline: Proc(c.WGPUProcDeviceCreateComputePipeline),
    DeviceCreateBindGroup: Proc(c.WGPUProcDeviceCreateBindGroup),
    DeviceCreateCommandEncoder: Proc(c.WGPUProcDeviceCreateCommandEncoder),
    DeviceRelease: Proc(c.WGPUProcDeviceRelease),
    QueueWriteBuffer: Proc(c.WGPUProcQueueWriteBuffer),
    QueueSubmit: Proc(c.WGPUProcQueueSubmit),
    QueueOnSubmittedWorkDone: Proc(c.WGPUProcQueueOnSubmittedWorkDone),
    QueueRelease: Proc(c.WGPUProcQueueRelease),
    BufferMapAsync: Proc(c.WGPUProcBufferMapAsync),
    BufferGetConstMappedRange: Proc(c.WGPUProcBufferGetConstMappedRange),
    BufferUnmap: Proc(c.WGPUProcBufferUnmap),
    BufferDestroy: Proc(c.WGPUProcBufferDestroy),
    BufferRelease: Proc(c.WGPUProcBufferRelease),
    ShaderModuleRelease: Proc(c.WGPUProcShaderModuleRelease),
    ComputePipelineGetBindGroupLayout: Proc(c.WGPUProcComputePipelineGetBindGroupLayout),
    ComputePipelineRelease: Proc(c.WGPUProcComputePipelineRelease),
    BindGroupLayoutRelease: Proc(c.WGPUProcBindGroupLayoutRelease),
    BindGroupRelease: Proc(c.WGPUProcBindGroupRelease),
    CommandEncoderBeginComputePass: Proc(c.WGPUProcCommandEncoderBeginComputePass),
    CommandEncoderClearBuffer: Proc(c.WGPUProcCommandEncoderClearBuffer),
    CommandEncoderCopyBufferToBuffer: Proc(c.WGPUProcCommandEncoderCopyBufferToBuffer),
    CommandEncoderFinish: Proc(c.WGPUProcCommandEncoderFinish),
    CommandEncoderRelease: Proc(c.WGPUProcCommandEncoderRelease),
    CommandBufferRelease: Proc(c.WGPUProcCommandBufferRelease),
    ComputePassEncoderSetPipeline: Proc(c.WGPUProcComputePassEncoderSetPipeline),
    ComputePassEncoderSetBindGroup: Proc(c.WGPUProcComputePassEncoderSetBindGroup),
    ComputePassEncoderDispatchWorkgroups: Proc(c.WGPUProcComputePassEncoderDispatchWorkgroups),
    ComputePassEncoderEnd: Proc(c.WGPUProcComputePassEncoderEnd),
    ComputePassEncoderRelease: Proc(c.WGPUProcComputePassEncoderRelease),

    fn load(lib: *dynlib.Lib) error{MissingSymbol}!Procs {
        var p: Procs = undefined;
        const info = @typeInfo(Procs).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(p, name) = lib.lookup(T, "wgpu" ++ name) orelse return error.MissingSymbol;
        }
        return p;
    }
};

/// The non-optional function pointer type behind a `WGPUProc*` typedef.
fn Proc(comptime T: type) type {
    return @typeInfo(T).optional.child;
}

pub fn str(s: []const u8) c.WGPUStringView {
    return .{ .data = s.ptr, .length = s.len };
}

pub fn slice(s: c.WGPUStringView) []const u8 {
    const p = s.data orelse return "";
    return if (s.length == std.math.maxInt(usize)) std.mem.span(p) else p[0..s.length];
}

pub const Error = error{
    LibraryNotFound,
    MissingSymbol,
    NoInstance,
    NoAdapter,
    NoDevice,
    DeviceLost,
    /// A WebGPU validation or internal error; see `Device.error_message`.
    GpuError,
};

/// Library name the dynamic loader searches for by default.
pub const default_library = switch (@import("builtin").os.tag) {
    .windows => "webgpu_dawn.dll",
    .macos => "libwebgpu_dawn.dylib",
    else => "libwebgpu_dawn.so",
};

pub const Device = struct {
    lib: dynlib.Lib,
    p: Procs,
    instance: c.WGPUInstance,
    adapter: c.WGPUAdapter,
    device: c.WGPUDevice,
    queue: c.WGPUQueue,
    limits: c.WGPULimits,
    has_f16: bool,

    /// First uncaptured error since the last `takeError`.
    error_message: [1024]u8 = undefined,
    error_len: usize = 0,
    errored: bool = false,
    lost: bool = false,

    /// Opens Dawn and the highest-performance adapter. The device requests
    /// the adapter's full limits and shader-f16 when available. The result
    /// is heap-allocated because Dawn callbacks hold a pointer to it.
    pub fn init(gpa: std.mem.Allocator, library_path: []const u8) (Error || std.mem.Allocator.Error)!*Device {
        const d = try gpa.create(Device);
        errdefer gpa.destroy(d);
        d.* = .{
            .lib = dynlib.Lib.open(library_path) catch return error.LibraryNotFound,
            .p = undefined,
            .instance = null,
            .adapter = null,
            .device = null,
            .queue = null,
            .limits = std.mem.zeroes(c.WGPULimits),
            .has_f16 = false,
        };
        errdefer d.lib.close();
        d.p = try Procs.load(&d.lib);
        const p = &d.p;

        const instance_features = [_]c.WGPUInstanceFeatureName{c.WGPUInstanceFeatureName_TimedWaitAny};
        var idesc = std.mem.zeroes(c.WGPUInstanceDescriptor);
        idesc.requiredFeatureCount = instance_features.len;
        idesc.requiredFeatures = &instance_features;
        d.instance = p.CreateInstance(&idesc) orelse return error.NoInstance;
        errdefer p.InstanceRelease(d.instance);

        var opts = std.mem.zeroes(c.WGPURequestAdapterOptions);
        opts.powerPreference = c.WGPUPowerPreference_HighPerformance;
        var adapter_result: Result(c.WGPUAdapter) = .{};
        var acb = std.mem.zeroes(c.WGPURequestAdapterCallbackInfo);
        acb.mode = c.WGPUCallbackMode_WaitAnyOnly;
        acb.callback = Result(c.WGPUAdapter).onAdapter;
        acb.userdata1 = &adapter_result;
        d.wait(p.InstanceRequestAdapter(d.instance, &opts, acb));
        d.adapter = adapter_result.value orelse return error.NoAdapter;
        errdefer p.AdapterRelease(d.adapter);

        var limits = std.mem.zeroes(c.WGPULimits);
        _ = p.AdapterGetLimits(d.adapter, &limits);
        d.has_f16 = p.AdapterHasFeature(d.adapter, c.WGPUFeatureName_ShaderF16) != 0;
        const features = [_]c.WGPUFeatureName{c.WGPUFeatureName_ShaderF16};

        var ddesc = std.mem.zeroes(c.WGPUDeviceDescriptor);
        ddesc.requiredFeatureCount = if (d.has_f16) 1 else 0;
        ddesc.requiredFeatures = &features;
        ddesc.requiredLimits = &limits;
        ddesc.uncapturedErrorCallbackInfo.callback = onUncapturedError;
        ddesc.uncapturedErrorCallbackInfo.userdata1 = d;
        ddesc.deviceLostCallbackInfo.mode = c.WGPUCallbackMode_AllowSpontaneous;
        ddesc.deviceLostCallbackInfo.callback = onDeviceLost;
        ddesc.deviceLostCallbackInfo.userdata1 = d;
        var device_result: Result(c.WGPUDevice) = .{};
        var dcb = std.mem.zeroes(c.WGPURequestDeviceCallbackInfo);
        dcb.mode = c.WGPUCallbackMode_WaitAnyOnly;
        dcb.callback = Result(c.WGPUDevice).onDevice;
        dcb.userdata1 = &device_result;
        d.wait(p.AdapterRequestDevice(d.adapter, &ddesc, dcb));
        d.device = device_result.value orelse return error.NoDevice;
        d.queue = p.DeviceGetQueue(d.device);
        _ = p.DeviceGetLimits(d.device, &d.limits);
        return d;
    }

    pub fn deinit(d: *Device, gpa: std.mem.Allocator) void {
        d.p.QueueRelease(d.queue);
        d.p.DeviceRelease(d.device);
        d.p.AdapterRelease(d.adapter);
        d.p.InstanceRelease(d.instance);
        d.lib.close();
        gpa.destroy(d);
    }

    pub fn wait(d: *Device, future: c.WGPUFuture) void {
        var wait_info: c.WGPUFutureWaitInfo = .{ .future = future, .completed = 0 };
        _ = d.p.InstanceWaitAny(d.instance, 1, &wait_info, std.math.maxInt(u64));
    }

    /// Returns `error.GpuError` (or `DeviceLost`) if an error was reported
    /// since the last call, and clears it.
    pub fn takeError(d: *Device) Error!void {
        if (d.lost) return error.DeviceLost;
        if (!d.errored) return;
        d.errored = false;
        return error.GpuError;
    }

    pub fn errorMessage(d: *const Device) []const u8 {
        return d.error_message[0..d.error_len];
    }

    pub const Info = struct {
        vendor: []const u8,
        architecture: []const u8,
        device: []const u8,
        description: []const u8,
        backend: []const u8,
        adapter_type: []const u8,
    };

    /// Adapter description; strings stay valid until `freeInfo`.
    pub fn info(d: *Device, raw: *c.WGPUAdapterInfo) Info {
        raw.* = std.mem.zeroes(c.WGPUAdapterInfo);
        _ = d.p.AdapterGetInfo(d.adapter, raw);
        return .{
            .vendor = slice(raw.vendor),
            .architecture = slice(raw.architecture),
            .device = slice(raw.device),
            .description = slice(raw.description),
            .backend = switch (raw.backendType) {
                c.WGPUBackendType_D3D12 => "d3d12",
                c.WGPUBackendType_Metal => "metal",
                c.WGPUBackendType_Vulkan => "vulkan",
                c.WGPUBackendType_D3D11 => "d3d11",
                c.WGPUBackendType_OpenGL, c.WGPUBackendType_OpenGLES => "opengl",
                c.WGPUBackendType_Null => "null",
                else => "unknown",
            },
            .adapter_type = switch (raw.adapterType) {
                c.WGPUAdapterType_DiscreteGPU => "discrete",
                c.WGPUAdapterType_IntegratedGPU => "integrated",
                c.WGPUAdapterType_CPU => "cpu",
                else => "unknown",
            },
        };
    }

    pub fn freeInfo(d: *Device, raw: c.WGPUAdapterInfo) void {
        d.p.AdapterInfoFreeMembers(raw);
    }

    fn onUncapturedError(_: [*c]const c.WGPUDevice, _: c.WGPUErrorType, message: c.WGPUStringView, userdata1: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
        const d: *Device = @ptrCast(@alignCast(userdata1.?));
        if (d.errored) return;
        const m = slice(message);
        const n = @min(m.len, d.error_message.len);
        @memcpy(d.error_message[0..n], m[0..n]);
        d.error_len = n;
        d.errored = true;
    }

    fn onDeviceLost(_: [*c]const c.WGPUDevice, reason: c.WGPUDeviceLostReason, message: c.WGPUStringView, userdata1: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
        // Destroying the device on shutdown also reports a loss.
        if (reason == c.WGPUDeviceLostReason_Destroyed or reason == c.WGPUDeviceLostReason_CallbackCancelled) return;
        const d: *Device = @ptrCast(@alignCast(userdata1.?));
        d.lost = true;
        const m = slice(message);
        const n = @min(m.len, d.error_message.len);
        @memcpy(d.error_message[0..n], m[0..n]);
        d.error_len = n;
    }
};

/// Receives the object from a request callback.
fn Result(comptime T: type) type {
    return struct {
        value: T = null,

        fn onAdapter(status: c.WGPURequestAdapterStatus, adapter: c.WGPUAdapter, message: c.WGPUStringView, userdata1: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
            const r: *@This() = @ptrCast(@alignCast(userdata1.?));
            if (status == c.WGPURequestAdapterStatus_Success) r.value = adapter else std.log.err("adapter: {s}", .{slice(message)});
        }

        fn onDevice(status: c.WGPURequestDeviceStatus, device: c.WGPUDevice, message: c.WGPUStringView, userdata1: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
            const r: *@This() = @ptrCast(@alignCast(userdata1.?));
            if (status == c.WGPURequestDeviceStatus_Success) r.value = device else std.log.err("device: {s}", .{slice(message)});
        }
    };
}
