const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Velocity, the native D3D12 backend: Windows only, against the SDKs
    // scripts/fixtures.sh velocity fetches.
    const velocity = b.option(bool, "velocity", "Build the Velocity backend (Windows; needs scripts/fixtures.sh velocity)") orelse false;
    if (velocity and target.result.os.tag != .windows) @panic("-Dvelocity needs a Windows target (-Dtarget=x86_64-windows-gnu)");
    const velocity_sdk: std.Build.LazyPath = if (b.option([]const u8, "velocity-sdk", "Velocity SDK directory (default: .fixtures/velocity)")) |p|
        .{ .cwd_relative = p }
    else
        b.path(".fixtures/velocity");
    const options = b.addOptions();
    options.addOption(bool, "velocity", velocity);
    const platform: Platform = .{ .options = options.createModule(), .velocity = if (velocity) velocity_sdk else null };

    const mod = b.addModule("zllm", zllmOptions(b, target, optimize));
    mod.addImport("webgpu", webgpuModule(b, target, optimize));
    addPlatform(b, mod, platform);
    const exe = addCli(b, mod);
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.addPassthruArgs();
    b.step("run", "Run the zllm CLI").dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run unit tests (no fixtures needed)");
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mod })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = exe.root_module })).step);

    // zllm.wasm for the browser shim (web/src/zllm), installed as
    // zig-out/web/zllm.wasm. Safety checks stay on by default.
    const wasm_target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding });
    const wasm_optimize = b.option(std.builtin.OptimizeMode, "wasm-optimize", "Optimize mode for zllm.wasm (default: safe)") orelse .safe;
    const wasm_options = b.addOptions();
    wasm_options.addOption(bool, "velocity", false);
    const wasm = b.addExecutable(.{
        .name = "zllm",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/api/wasm.zig"),
            .target = wasm_target,
            .optimize = wasm_optimize,
            .imports = &.{.{ .name = "zllm", .module = b.createModule(.{
                .root_source_file = b.path("src/root.zig"),
                .target = wasm_target,
                .optimize = wasm_optimize,
                .imports = &.{.{ .name = "build_options", .module = wasm_options.createModule() }},
            }) }},
        }),
    });
    wasm.entry = .disabled;
    wasm.rdynamic = true;
    const wasm_install = b.addInstallArtifact(wasm, .{ .dest_dir = .{ .override = .{ .custom = "web" } } });
    b.step("wasm", "Build zllm.wasm for the browser").dependOn(&wasm_install.step);

    // Parity checks against pinned references. They need `scripts/fixtures.sh`
    // and run through the uv-managed harness in tools/. The CLI they drive is
    // always ReleaseSafe: fast enough for the reference forward pass, with
    // safety checks kept on.
    const check_mod = b.createModule(zllmOptions(b, target, .ReleaseSafe));
    check_mod.addImport("webgpu", webgpuModule(b, target, .ReleaseSafe));
    addPlatform(b, check_mod, platform);
    const check_exe = addCli(b, check_mod);
    const fixtures: std.Build.LazyPath = if (b.option([]const u8, "fixtures", "Fixture directory (default: .fixtures)")) |p|
        .{ .cwd_relative = p }
    else
        b.path(".fixtures");
    _ = addCheck(b, check_exe, fixtures, "check-gguf", "Compare the GGUF index with llama.cpp gguf-dump", "check_gguf.py");
    _ = addCheck(b, check_exe, fixtures, "check-tokenizer", "Compare tokenization with Hugging Face tokenizers", "check_tokenizer.py");
    _ = addCheck(b, check_exe, fixtures, "check-chat", "Compare chat prompts with the pinned Jinja template", "check_chat.py");
    _ = addCheck(b, check_exe, fixtures, "check-forward", "Compare the reference forward pass with llama.cpp", "check_forward.py");
    _ = addCheck(b, check_exe, fixtures, "check-dequant", "Compare CPU dequantization with ggml's, bit for bit", "check_dequant.py");

    // Where GPU checks run: `native` is this build on whatever adapter Dawn
    // finds (llvmpipe in CI); `d3d12` is the Windows build run from WSL
    // through scripts/d3d12.sh, on Dawn; `velocity` is that build on the
    // Velocity backend.
    const gpu_host = b.option(enum { native, d3d12, velocity }, "gpu-host", "Where GPU checks run (default: native)") orelse .native;
    const gpu_run = addCheck(b, check_exe, fixtures, "check-gpu", "Compare the GPU forward pass with the CPU reference", "check_gpu.py");
    gpu_run.addArgs(&.{ "--host", @tagName(gpu_host) });
    const quant_run = addCheck(b, check_exe, fixtures, "check-quant", "Compare the GPU forward pass on the quantized 27B with llama.cpp", "check_quant.py");
    quant_run.addArgs(&.{ "--host", @tagName(gpu_host) });
    if (b.option([]const u8, "quant-model", "Quantized GGUF for check-quant (default: the 27B fixture)")) |m| quant_run.addArgs(&.{ "--model", m });
    // The Gyro-Coder is read in place (75.8 GB): its path is an option.
    const gyro_run = addCheck(b, check_exe, fixtures, "check-gyro", "Compare the Qwen3.8-Flash-Next Gyro-Coder on the GPU with the CPU reference and its runtime's greedy tokens", "check_gyro.py");
    gyro_run.addArgs(&.{ "--host", @tagName(gpu_host), "--model", b.option([]const u8, "gyro-model", "Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf for check-gyro") orelse "" });

    const kernel_step = b.step("gpu-check", "Check every GPU kernel against a CPU computation");
    switch (gpu_host) {
        .native => {
            // The Dawn path is only known at run time, so set it in a shell.
            const run = b.addSystemCommand(&.{ "sh", "-c", "ZLLM_DAWN=\"$1/dawn/libwebgpu_dawn.so\" exec \"$2\" gpu-check", "sh" });
            run.addDirectoryArg(fixtures);
            run.addArtifactArg(check_exe);
            run.has_side_effects = true;
            kernel_step.dependOn(&run.step);
        },
        .d3d12, .velocity => {
            const run = b.addSystemCommand(&.{"bash"});
            run.addFileArg(b.path("scripts/d3d12.sh"));
            run.addArg("gpu-check");
            run.setEnvironmentVariable("ZLLM_BACKEND", if (gpu_host == .velocity) "velocity" else "webgpu");
            run.has_side_effects = true;
            kernel_step.dependOn(&run.step);
        },
    }
}

/// Options for the zllm library module; it also imports `webgpuModule`.
/// Native builds link libc so the GPU host can load Dawn with
/// dlopen/LoadLibrary.
fn zllmOptions(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) std.Build.Module.CreateOptions {
    return .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    };
}

/// The backends a build has: its build options, and Velocity's SDK when
/// Velocity is built.
const Platform = struct {
    options: *std.Build.Module,
    velocity: ?std.Build.LazyPath,
};

/// Adds the platform to a zllm module: the build options, and with
/// Velocity its native layer (src/velocity/host.cpp), its C interface as
/// the module `velocity_c`, and the D3D12, DXGI and DirectStorage libraries.
fn addPlatform(b: *std.Build, mod: *std.Build.Module, platform: Platform) void {
    mod.addImport("build_options", platform.options);
    const sdk = platform.velocity orelse return;
    const includes = [_]std.Build.LazyPath{
        b.path("src/velocity"),
        sdk.path(b, "agility/build/native/include"),
        sdk.path(b, "directstorage/native/include"),
        sdk.path(b, "dxc/inc"),
    };
    for (includes) |p| mod.addIncludePath(p);
    mod.addCSourceFile(.{ .file = b.path("src/velocity/host.cpp"), .flags = &.{ "-std=c++20", "-fms-extensions" }, .language = .cpp });
    mod.link_libcpp = true;
    mod.linkSystemLibrary("d3d12", .{});
    mod.linkSystemLibrary("dxgi", .{});
    mod.addObjectFile(sdk.path(b, "directstorage/native/lib/x64/dstorage.lib"));
    const tc = b.addTranslateC(.{
        .root_source_file = b.path("src/velocity/velocity.h"),
        .target = mod.resolved_target.?,
        .optimize = mod.optimize.?,
    });
    mod.addImport("velocity_c", tc.createModule());
}

/// Zig bindings for the vendored Dawn `webgpu.h`. Only the `WGPUProc*`
/// typedefs are used: Dawn is loaded at run time (see src/gpu/dawn.zig).
fn webgpuModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const tc = b.addTranslateC(.{
        .root_source_file = b.path("third_party/dawn/include/webgpu/webgpu.h"),
        .target = target,
        .optimize = optimize,
    });
    tc.addIncludePath(b.path("third_party/dawn/include"));
    tc.defineCMacro("WGPU_SKIP_DECLARATIONS", null);
    return tc.createModule();
}

/// The zllm CLI over `lib`, built with the library module's target and mode.
fn addCli(b: *std.Build, lib: *std.Build.Module) *std.Build.Step.Compile {
    return b.addExecutable(.{
        .name = "zllm",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = lib.resolved_target,
            .optimize = lib.optimize,
            .imports = &.{.{ .name = "zllm", .module = lib }},
        }),
    });
}

fn addCheck(
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    fixtures: std.Build.LazyPath,
    name: []const u8,
    description: []const u8,
    script: []const u8,
) *std.Build.Step.Run {
    const run = b.addSystemCommand(&.{ "uv", "run", "--frozen", "--project" });
    run.addDirectoryArg(b.path("tools"));
    run.addArg("python");
    run.addFileArg(b.path(b.fmt("tools/{s}", .{script})));
    run.addArg("--zllm");
    run.addArtifactArg(exe);
    run.addArg("--fixtures");
    run.addDirectoryArg(fixtures);
    run.has_side_effects = true;
    b.step(name, description).dependOn(&run.step);
    return run;
}
