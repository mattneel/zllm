const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Configuration options
    const enable_simd = b.option(bool, "simd", "Enable SIMD optimizations") orelse true;
    const max_threads = b.option(u32, "max-threads", "Maximum number of threads") orelse 0;

    // Core library module
    const zllm_core = b.addModule("zllm-core", .{
        .root_source_file = b.path("src/core/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Model architectures module
    const zllm_models = b.addModule("zllm-models", .{
        .root_source_file = b.path("src/models/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    zllm_models.addImport("core", zllm_core);

    // Tokenizers module
    const zllm_tokenizers = b.addModule("zllm-tokenizers", .{
        .root_source_file = b.path("src/tokenizers/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Training module
    const zllm_training = b.addModule("zllm-training", .{
        .root_source_file = b.path("src/training/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    zllm_training.addImport("core", zllm_core);

    // Main zllm module that re-exports everything
    const zllm = b.addModule("zllm", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    zllm.addImport("core", zllm_core);
    zllm.addImport("models", zllm_models);
    zllm.addImport("tokenizers", zllm_tokenizers);
    zllm.addImport("training", zllm_training);

    // Add build options
    const options = b.addOptions();
    options.addOption(bool, "enable_simd", enable_simd);
    options.addOption(u32, "max_threads", max_threads);
    zllm_core.addOptions("build_options", options);

    // Tests
    const core_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/core/lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    core_tests.root_module.addOptions("build_options", options);

    const models_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/models/lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    models_tests.root_module.addImport("core", zllm_core);

    const training_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/training/lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    training_tests.root_module.addImport("core", zllm_core);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(core_tests).step);
    test_step.dependOn(&b.addRunArtifact(models_tests).step);
    test_step.dependOn(&b.addRunArtifact(training_tests).step);

    // Examples
    const example_generate = b.addExecutable(.{
        .name = "generate",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/generate.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    example_generate.root_module.addImport("zllm", zllm);
    b.installArtifact(example_generate);

    const example_train = b.addExecutable(.{
        .name = "train",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/train.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    example_train.root_module.addImport("zllm", zllm);
    b.installArtifact(example_train);

    const example_step = b.step("examples", "Build examples");
    example_step.dependOn(&example_generate.step);
    example_step.dependOn(&example_train.step);

    const run_generate = b.addRunArtifact(example_generate);
    const run_step = b.step("run", "Run generation example");
    run_step.dependOn(&run_generate.step);

    // Benchmarks
    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    bench_exe.root_module.addImport("zllm", zllm);

    const bench_step = b.step("bench", "Run benchmarks");
    const run_bench = b.addRunArtifact(bench_exe);
    bench_step.dependOn(&run_bench.step);

    // CLI tool
    const cli = b.addExecutable(.{
        .name = "zllm",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    cli.root_module.addImport("zllm", zllm);
    b.installArtifact(cli);

    // Format checking
    const fmt_step = b.step("fmt", "Check formatting");
    const fmt = b.addFmt(.{
        .paths = &.{
            "src",
            "examples",
            "bench",
            "build.zig",
        },
        .check = true,
    });
    fmt_step.dependOn(&fmt.step);

    // CI step that runs everything
    const ci_step = b.step("ci", "Run all CI checks");
    ci_step.dependOn(test_step);
    ci_step.dependOn(fmt_step);
    ci_step.dependOn(example_step);
    ci_step.dependOn(bench_step);
}
