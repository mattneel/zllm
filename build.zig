const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("zllm", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
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
    const check_exe = addCli(b, b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    }));
    const fixtures: std.Build.LazyPath = if (b.option([]const u8, "fixtures", "Fixture directory (default: .fixtures)")) |p|
        .{ .cwd_relative = p }
    else
        b.path(".fixtures");
    addCheck(b, check_exe, fixtures, "check-gguf", "Compare the GGUF index with llama.cpp gguf-dump", "check_gguf.py");
    addCheck(b, check_exe, fixtures, "check-tokenizer", "Compare tokenization with Hugging Face tokenizers", "check_tokenizer.py");
    addCheck(b, check_exe, fixtures, "check-forward", "Compare the reference forward pass with llama.cpp", "check_forward.py");
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
) void {
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
}
