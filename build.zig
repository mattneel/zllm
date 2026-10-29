const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("zllm", zllmOptions(b, target, optimize));
    mod.addImport("webgpu", webgpuModule(b, target, optimize));
    const exe = addCli(b, mod);
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.addPassthruArgs();
    b.step("run", "Run the zllm CLI").dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run unit tests (no fixtures needed)");
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mod })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = exe.root_module })).step);

    // Parity checks against pinned references. They need `scripts/fixtures.sh`
    // and run through the uv-managed harness in tools/. The CLI they drive is
    // always ReleaseSafe: fast enough for the reference forward pass, with
    // safety checks kept on.
    const check_mod = b.createModule(zllmOptions(b, target, .ReleaseSafe));
    check_mod.addImport("webgpu", webgpuModule(b, target, .ReleaseSafe));
    const check_exe = addCli(b, check_mod);
    const fixtures: std.Build.LazyPath = if (b.option([]const u8, "fixtures", "Fixture directory (default: .fixtures)")) |p|
        .{ .cwd_relative = p }
    else
        b.path(".fixtures");
    _ = addCheck(b, check_exe, fixtures, "check-gguf", "Compare the GGUF index with llama.cpp gguf-dump", "check_gguf.py");
    _ = addCheck(b, check_exe, fixtures, "check-tokenizer", "Compare tokenization with Hugging Face tokenizers", "check_tokenizer.py");
    _ = addCheck(b, check_exe, fixtures, "check-forward", "Compare the reference forward pass with llama.cpp", "check_forward.py");

    // GPU checks run on a Dawn host: `native` is this build on whatever
    // adapter Dawn finds (llvmpipe in CI); `d3d12` is the Windows build run
    // from WSL through scripts/d3d12.sh.
    const gpu_host = b.option(enum { native, d3d12 }, "gpu-host", "Dawn host for GPU checks (default: native)") orelse .native;
    const gpu_run = addCheck(b, check_exe, fixtures, "check-gpu", "Compare the GPU forward pass with the CPU reference", "check_gpu.py");
    gpu_run.addArgs(&.{ "--host", @tagName(gpu_host) });

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
        .d3d12 => {
            const run = b.addSystemCommand(&.{"bash"});
            run.addFileArg(b.path("scripts/d3d12.sh"));
            run.addArg("gpu-check");
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
