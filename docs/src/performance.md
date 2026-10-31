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
