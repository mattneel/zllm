# Performance

The goal is to use each device well without weakening correctness. The
numbers below are historical measurements, not cross-platform guarantees.
A faster kernel is useful only after its reference gate passes.

## Recorded results

All rows below use an RTX 5090 Laptop GPU. Native rows use Dawn's D3D12
host, not Agility. Dates identify the commit that recorded the result;
the repository does not separately timestamp every execution. These
snapshots predate later kernel and precision changes.

| Recorded in | Model and workload | Result | Evidence |
|---|---|---|---|
| 2026-10-29, `eea4f84` | Qwen 3.5 0.8B BF16, native generation, f16 KV, short-context greedy decode | 285–295 tok/s | [Generation performance](generation.md#performance) |
| 2026-10-29, `eea4f84` | Same model, native prefill of a 2,457-token prompt | 2,882–2,896 tok/s | [Generation performance](generation.md#performance) |
| 2026-10-29, `2acacac` | Same model, Chrome, greedy decode with two steps in flight | 252–295 tok/s | [Browser execution](browser.md#two-steps-in-flight) |
| 2026-10-29, `a4dd813` | Qwen 3.8 27B Q3_K_M, native benchmark | About 320 tok/s prefill; 30 tok/s decode | Commit message; committed per-device JSON |
| 2026-10-29, `bfa5353` | Same 27B, native code continuation, MTP speculation | 37–41 tok/s, versus 29–31 without speculation | [Speculative decoding](generation.md#speculative-decoding) |
| 2026-10-29, `a3f49e4` | Same 27B, Chrome, local-file load | 15.3 s | [Browser 27B](browser.md#the-27b) |
| 2026-10-30, `2a2cadf` | Gyro-Coder, greedy decode | 35–38 tok/s native; 21.4–22.2 tok/s Chrome | [Gyro-Coder](gyro.md) and commit message |
| 2026-10-30, `8cac630` | Granite 4.2 30B Q4_K_M, native benchmark | 209 tok/s prefill; 25.1 tok/s decode | [Dense performance](dense.md#performance) |
| 2026-10-30, `7f494e0` | Gyro-Coder, Chrome load with tuning versus cached tuning | 29.4 s first load; 21.8 s cached-tuning load | [Browser plan cache](browser.md#tuning-and-the-plan-cache) |

The [generation chapter](generation.md#performance) retains the detailed
optimization history, context lengths and sampling settings. Browser
measurements distinguish HTTP loading, OPFS caching and picked files.
These are different workloads; they should not be compared as though they
were one benchmark.

## Gyro-Coder Agility prefill cutover

Measured on 2026-10-31 with the RTX 5090 Laptop, native Agility,
`Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf` (75,796,732,256 bytes),
8,192 context positions, f16 KV and the default tuned matvecs. The
baseline is an immutable worktree of `a1fe958`; both versions ran these
same commands, with only one GPU job at a time:

```sh
ZLLM_BACKEND=agility scripts/d3d12.sh benchmark 'C:\models\Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf'
ZLLM_BACKEND=agility scripts/d3d12.sh profile 'C:\models\Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf' 4
```

| Workload | Decode-only baseline | Tiled/grouped prefill |
|---|---:|---:|
| `benchmark`: 1,024-token prompt | 92.91 tok/s, 257 steps | 107.29 tok/s, 65 steps |
| `benchmark`: greedy decode | 52.73 tok/s | 54.41 tok/s |
| `benchmark`: plain decode | 51.56 tok/s | 54.47 tok/s |
| `profile ... 4`: unprofiled 4-token step | 40.70 ms (98.28 tok/s) | 41.23 ms (97.02 tok/s) |
| Same profile's one-token decode | 21.88 ms | 19.64 ms |

The end-to-end prefill improvement is **15.5% (1.155×)**, not a claim of
250 tok/s. The remaining gap to that comparison point is 142.71 tok/s,
requiring another 2.33× over this benchmark snapshot. Four-token steps
deliberately stay on the decode path, so the required `profile ... 4`
comparison does not measure the new prefill kernels.

An additional `profile ... 64` run exercises the new path: 462.11 ms per
unprofiled 64-token step (138.50 tok/s), at positions 64–448. Its GPU
timestamp sum is 349.36 ms; per-dispatch profiling and wall-clock timing
are different measurements. The top phases are:

| Phase | GPU ms per 64-token step | Timestamp share |
|---|---:|---:|
| Grouped TQ2_T experts (`gyro_matmul`) | 224.19 | 64.2% |
| Float-activation Q8_0 dense projections | 94.81 | 27.1% |
| Sequential DeltaNet recurrence | 8.58 | 2.5% |
| Expert grouping | 4.57 | 1.3% |
| Sparse attention | 4.16 | 1.2% |

Grouped gate/up alone accounts for 166.41 ms (47.6%); grouped down for
57.77 ms (16.5%). Expert matmul, then dense projection throughput, remain
the principal gap rather than routing or the PLE kernels. These are
ordinary compute shaders, not a tensor-core performance claim. Both
Agility and Dawn D3D12 `check-fast` gates passed before recording these
results, with unchanged Gyro reference bounds and bitwise chunking checks
separately within each path; see [Gyro-Coder](gyro.md).

## Per-device records

`benchmarks/rtx5090-laptop-d3d12.jsonl` stores the committed native benchmark
records. Each line names the adapter and model, load phases, a random
prefill prompt and greedy decode. The benchmark protocol uses a
1,024-token prompt and 128 generated tokens. Records with `decode_plain`
compare default speculation with one token per step. Earlier records have
only `decode`. See `benchmarks/README.md` in the repository.

To record a result after the appropriate fixtures and CLI are installed:

```sh
ZLLM_DAWN=.fixtures/dawn/libwebgpu_dawn.so \
  zig-out/bin/zllm benchmark .fixtures/models/Qwen3.5-0.8B-BF16.gguf
```

Use `profile` for per-kernel timestamp measurements and `bench` for decode
step attribution. They answer different questions from the end-to-end
`benchmark` command. Record the backend, adapter, model revision, context,
KV format, tuning state and commit when comparing runs.

## Targets are not measurements

The [spec](spec.md) sets a 27B Q3_K target on an RTX 5090 Laptop in Chrome:
at least 20 tok/s decode and 300 tok/s prefill, warm load within 40 s,
and no submission over 250 ms. The scheduler aims for a 150 ms submission
budget. These numbers are design and acceptance thresholds, not measured
performance on every adapter.

There is no committed performance table here for Agility tensor-core
execution, CUDA, every browser/driver combination, or the future image and
video models. Do not infer a speedup from an exposed hardware capability.
Read [Backends](backends.md) for what is actually executed and
[Validation](validation.md) for accuracy and soak-test evidence.
