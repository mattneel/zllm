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

## Agility LinAlg prefill

On the same RTX 5090 Laptop and model described above, a fresh run of
`668ea20` and the LinAlg cutover used the same commands, 8,192-position
context, f16 KV, and default load-time matvec tuning. Only one D3D12 job
ran at a time. The new shader catalog invalidated the tuning cache, so the
after benchmark includes a fresh sweep; tuning time is outside throughput.

```sh
ZLLM_BACKEND=agility scripts/d3d12.sh benchmark 'C:\models\Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf'
ZLLM_BACKEND=agility scripts/d3d12.sh profile 'C:\models\Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf' 64
```

| Workload | Scalar `668ea20`, remeasured | Compensated LinAlg |
|---|---:|---:|
| 1,024-token benchmark prefill, 65 steps | 111.54 tok/s | 118.71 tok/s |
| Greedy decode, 128 tokens | 55.17 tok/s | 54.49 tok/s |
| Plain decode, 128 tokens | 55.07 tok/s | 53.48 tok/s |
| Unprofiled 64-token step at positions 64–448 | 470.18 ms | 378.30 ms |
| Same step's GPU timestamp sum | 347.468 ms | 252.774 ms |

The end-to-end prefill gain is **6.43% (1.064×)**. It remains
**131.29 tok/s short of 250 tok/s**, requiring another **2.106×**.
The 64-token profile reaches 169.18 tok/s, but that is a different workload
from the benchmark and must not substitute for its 118.71 tok/s result.
Decode kernels are unchanged; the measured decode rates above are not a
decode speedup claim.

The final 64-token GPU phase table is:

| Kernel | GPU ms per step | Timestamp share |
|---|---:|---:|
| `gyro_matmul` | 157.948 | 62.5% |
| `matmul_q8_0_float` | 65.272 | 25.8% |
| `delta_rule` | 8.371 | 3.3% |
| `moe_group` | 4.447 | 1.8% |
| `attn_split_sparse` | 4.201 | 1.7% |
| `moe_logits` | 3.144 | 1.2% |
| `matmul_bf16` | 2.814 | 1.1% |
| `matmul_reduce` | 1.439 | 0.6% |
| `rmsnorm` | 0.847 | 0.3% |
| `hc_combine` | 0.731 | 0.3% |
| `moe_route` | 0.602 | 0.2% |
| `delta_prep` | 0.589 | 0.2% |
| `rotate128` | 0.529 | 0.2% |
| `hc_mix` | 0.386 | 0.2% |
| `gated_norm` | 0.342 | 0.1% |
| `moe_combine` | 0.282 | 0.1% |
| `attn_combine` | 0.188 | 0.1% |
| `hc_act` | 0.183 | 0.1% |
| `silu_mul` | 0.111 | 0.0% |
| `rope_append` | 0.103 | 0.0% |
| `qsa_keys` | 0.049 | 0.0% |
| `qsa_pool` | 0.048 | 0.0% |
| `embed_q8_0` | 0.046 | 0.0% |
| `ple_conv` | 0.044 | 0.0% |
| `qsa_score` | 0.021 | 0.0% |
| `qsa_select` | 0.020 | 0.0% |
| `ple_mix` | 0.015 | 0.0% |
| **Sum** | **252.774** | |

Expert GPU time fell 28.8%, and dense float-Q8 GPU time fell 31.2%.
Expert gate/up still costs 108.684 ms and down 49.263 ms; together with
dense projections these remain 88.3% of timestamps. Profiling serializes
dispatches and its timestamp sum excludes wall-clock overhead.

### Precision experiments and acceptance

The retained implementation uses two FP16 matrix products: activation
high/residual parts multiplied by exact signed-byte or codebook values,
with original per-block scales applied afterward and f32 accumulation.
The initially tested three-product decomposition of both operands was
correct but slower. A deliberately uncompensated single-product experiment
was faster but failed the unchanged model gate and was removed:

| Implementation | Prefill tok/s | Decode tok/s | Expert-set flips | Streams max rel L2 | Logits median / max rel L2 |
|---|---:|---:|---:|---:|---:|
| Initial three-product compensation | 83.78 | 54.06 | 0 | 1.36e-4 | 1.97e-5 / 1.57e-4 |
| Single-product FP16, rejected | 132.52 | 54.52 | 85 | 7.23e-2 | 4.75e-3 / 7.58e-2 |
| Final two-product compensation | 118.71 | 54.49 | 0 | 1.39e-4 | 2.02e-5 / 1.63e-4 |

All three retained 66/66 top-1 agreement and 40/40 greedy tokens on this
66-position reference, illustrating why those checks alone are insufficient.
The single-product experiment failed the existing stream/logit bounds and
routing agreement despite bitwise step-split invariance. No tolerance was
loosened.

The final implementation passed `zig fmt --check .`, `zig build`,
`zig build test`, `zig build wasm`, standalone Agility `gpu-check`, and
`check-fast` on both Agility and Dawn D3D12 with the Gyro model. Each path's
split-step and fusion checks remain bitwise. Agility decode has zero expert
flips, stream max rel L2 1.49e-4 and logits median/max 1.82e-5/1.32e-4.
Dawn's unchanged prefill path has zero flips, stream max 1.12e-4 and logits
median/max 1.86e-5/1.47e-4. See [Precision](precision.md) for operand details.

## Removing Agility's per-row PLE storage waits

After `91e7a65`, a 64-token step took 378.30 ms while the profiled kernels
summed to 252.774 ms: an apparent 125.526 ms gap. That subtraction compares
one unprofiled warm-up step with six profiled steps at later positions, so
it is not by itself a measurement of GPU idle time.

PIX-free attribution used temporary `QueryPerformanceCounter` timers around
DirectStorage enqueue, storage-fence waits, command-list recording,
`ExecuteCommandLists`, and the final compute-fence wait. Additional GPU
timestamps bracketed each complete compute list, including barriers rather
than only individual dispatches. The diagnostic code was removed before
the final gates and measurements.

The culprit was **1,024 separate DirectStorage submissions**: 16 PLE table
rows for each of 64 tokens. The host waited for the previous storage
ticket after every row, preventing compute submission while the GPU was
idle. Routing and expert grouping do not read back during these prefill
steps; neither model programs nor descriptors are rebuilt per step.

Six profiled steps gave these mean diagnostic times:

| Measurement | Per-row submissions | Batched submissions |
|---|---:|---:|
| Storage submissions / original row ranges per step | 1,024 / 1,024 | 3 / 1,024 |
| CPU storage-fence waits | 112.016 ms | 10.834 ms |
| CPU storage enqueue calls | 2.893 ms | 6.781 ms |
| Native command-list recording | 1.693 ms | 1.584 ms |
| Command-list submission | 0.126 ms | 0.089 ms |
| Final compute-fence wait | 252.762 ms | 287.672 ms |
| Complete compute-list GPU span | 252.610 ms | 287.502 ms |

The final fence wait includes GPU execution and must not be counted as
additional idle time. Before batching, the full-list GPU span exceeded the
249.331 ms dispatch sum by only 3.279 ms: barriers were not the missing
125 ms. The diagnostic run after batching had a different GPU execution
time; its 275.834 ms dispatch sum and 307.46 ms wall time left 31.626 ms
outside dispatches, versus 121.519 ms before. Enqueue calls can block when
the bounded storage queue fills, so batching shifts some remaining storage
time into enqueue rather than eliminating file I/O.

The fix stages consecutive disjoint ranges for the same destination in a
fixed 510-entry batch, retaining the existing 32 MiB request-size limit
and two half-queue batches in flight. Buffer changes and non-monotonic or
overlapping destinations end a batch. Overlapping writes wait for the
earlier batch; pending reads flush before compute or a file change.
No shader, plan, precision gate or PLE row byte changes.

With diagnostic timers removed, the same model, commands, f16 KV,
8,192-position context and cached tuning gave:

| Measurement | Before (`91e7a65`) | Batched PLE reads |
|---|---:|---:|
| Unprofiled 64-token step | 378.30 ms | **277.56 ms** |
| Profiled wall time, mean of six steps | 382.91 ms | 270.16 ms |
| GPU kernel timestamp sum | 252.774 ms | 252.323 ms |
| Unprofiled-step minus kernel-sum comparison | 125.526 ms | 25.237 ms |
| Profiled wall minus kernel sum | 130.136 ms | 17.837 ms |
| 1,024-token benchmark prefill, 65 steps | 118.71 tok/s | **154.07 tok/s** |
| Greedy decode, 128 tokens | 54.49 tok/s | **61.53 tok/s** |
| Plain decode, 128 tokens | 53.48 tok/s | **61.31 tok/s** |

Prefill improves **29.78%**, and the unprofiled 64-token step falls
**100.74 ms (26.63%)**. The remaining benchmark gap to 250 tok/s is
95.93 tok/s, or another 1.623×. GPU arithmetic is unchanged; most of the
wall-time gain comes from avoiding repeated CPU storage-fence waits.

`zig fmt --check .`, `zig build`, `zig build test`, `zig build wasm`, and
both Agility and Dawn D3D12 `check-fast` passed. Unit tests cover range
preservation, the 510-request boundary, buffer changes, overlap/reversal
mutations, and all 1,024 PLE rows reaching exactly three batches. The
unchanged Gyro checks retain zero routing flips, 66/66 top-1 agreement and
40/40 greedy tokens; prefill splits of seven and one token and unfused
execution remain bitwise identical within each backend's prefill path.
Agility's prefill logits median/max rel L2 remain 2.02e-5/1.63e-4.

## Larger LinAlg expert tiles

The reference `gyro-dx` checkout uses `gemm_q8` tiles of 16 tokens by
64 outputs with KT64 and four computing waves. Its single-FP16 operand
path is not interchangeable with zllm's unchanged accuracy gates.
Its expert `moe_gyro` path does **not** group tokens by expert: the host
dispatches `n_tok * n_used` selected slots, and each workgroup decodes
32 output rows for one slot using scalar wave reductions. zllm already
shares decoded weights across 16 routed slots per expert.

The larger expert tile increases output rows from 32 to 64 so all four
waves compute, reusing activation and codebook staging across twice as many
outputs. It preserves the shared dispatch grid by returning odd x groups.
Wave exchange removes the second trellis-byte load; wave broadcasts and
direct matrix-coordinate output stores replace shared scale/result/slot
staging. Both activation products and the complete 128-K block order are
unchanged. This is not a lower-precision path.

Same benchmark protocol, model, context, f16 KV and adapter as above:

| Measurement | Before (`917438f`) | 64-output expert tile |
|---|---:|---:|
| 1,024-token prefill, 65 steps | 154.07 tok/s | 160.64 tok/s |
| Greedy decode | 61.53 tok/s | 61.81 tok/s |
| Plain decode | 61.31 tok/s | 62.11 tok/s |
| Unprofiled 64-token step | 277.56 ms | 248.61 ms |
| Profiled wall time | 270.16 ms | 268.27 ms |

| 64-token phase | Before GPU ms / share | 64-output expert GPU ms / share |
|---|---:|---:|
| Grouped experts | 156.969 / 62.2% | 148.957 / 60.9% |
| Dense float Q8_0 | 65.126 / 25.8% | 66.064 / 27.0% |
| DeltaNet recurrence | 8.901 / 3.5% | 8.547 / 3.5% |
| Expert grouping | 4.450 / 1.8% | 4.443 / 1.8% |
| Sparse attention | 4.216 / 1.7% | 4.209 / 1.7% |
| Router logits | 3.244 / 1.3% | 3.179 / 1.3% |
| BF16 matmul | 2.848 / 1.1% | 2.764 / 1.1% |
| Other kernels | 6.569 | 6.338 |
| **Sum** | **252.323** | **244.501** |

Grouped down falls from 48.953 to 42.195 ms; gate/up changes from 108.016
to 106.763 ms. Prefill throughput improves 4.26%; the remaining gap to
250 tok/s is 89.36 tok/s. Decode kernels are unchanged. The differing
warm-up and profiled step times are reported separately, not used to
substitute for the end-to-end benchmark.

The expert tile passed `zig fmt --check .`, `zig build`, `zig build test`,
`zig build wasm`, standalone Agility `gpu-check`, and `check-fast` on both
Agility and Dawn D3D12. GPU checks now include 69 expert output rows and
67 dense token rows, crossing the larger tile boundaries and their tails.
Gyro routing has zero flips, logits median/max rel L2 remain
2.02e-5/1.63e-4, and split/unfused prefill results remain bitwise identical.

## Bank-aware dense LinAlg staging

The retained dense optimization keeps TT16/TN64/KT64 rather than reducing
the number of active groups with a larger virtual tile. Shared activation
and weight rows have a stride of 72 half elements (144 bytes), retaining
16-byte alignment while avoiding the original power-of-two bank mapping.
Direct matrix-coordinate output stores remove the 4 KiB shared result
plane and its barrier. The resulting shared allocation is 14 KiB versus
16.5 KiB before. Both activation products and the per-32-K scaling order
are unchanged.

| Measurement | Before (`548444e`) | Padded shared rows / direct stores |
|---|---:|---:|
| 1,024-token prefill, 65 steps | 160.64 tok/s | 178.04 tok/s |
| Greedy decode | 61.81 tok/s | 61.73 tok/s |
| Plain decode | 62.11 tok/s | 61.11 tok/s |
| Unprofiled 64-token step | 248.61 ms | 228.95 ms |
| Profiled wall time | 268.27 ms | 239.31 ms |

| 64-token phase | Before GPU ms / share | Padded dense GPU ms / share |
|---|---:|---:|
| Grouped experts | 148.957 / 60.9% | 140.337 / 64.0% |
| Dense float Q8_0 | 66.064 / 27.0% | 48.337 / 22.0% |
| DeltaNet recurrence | 8.547 / 3.5% | 9.354 / 4.3% |
| Expert grouping | 4.443 / 1.8% | 4.629 / 2.1% |
| Sparse attention | 4.209 / 1.7% | 4.199 / 1.9% |
| Router logits | 3.179 / 1.3% | 3.279 / 1.5% |
| BF16 matmul | 2.764 / 1.1% | 2.869 / 1.3% |
| Other kernels | 6.338 | 6.419 |
| **Sum** | **244.501** | **219.423** |

Prefill improves 10.83% and dense GPU time falls 26.8%. An adjacent
remeasurement of the old dense kernel gave 64.684 ms (versus 48.337 ms,
a 25.3% reduction), with unchanged expert code at 135.705 ms. That control
also recorded 149.71 tok/s prefill and 57.63 tok/s decode: device timing and
the newly keyed tuning sweeps vary, so small decode movements should not be
attributed to a changed decode algorithm. Remaining gap to 250 tok/s:
71.96 tok/s, requiring another 1.404×.

A virtual 32-token by 128-output candidate passed the numerical checks but
regressed to 119.96 tok/s and 82.961 ms dense GPU time; it was removed.
Explicitly unrolling dynamic matrix-element loops caused an NVIDIA driver
pipeline-compilation crash on the tested driver and was also removed.
Larger tiles and fewer source-level loops are not automatic speedups.

The padded layout passed the unchanged standalone Agility `gpu-check`,
`check-fast` on Agility and Dawn D3D12, and fmt/build/test/wasm gates.
Agility's Gyro logits median/max rel L2 remain 2.02e-5/1.63e-4, with
zero routing flips, 66/66 top-1, 40/40 greedy and bitwise split/unfused
prefill results. No precision gate or tolerance changed.

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

The Agility LinAlg table above covers this specific adapter and workload,
not CUDA, every browser/driver combination, or future image and video
models. Do not infer a speedup from an exposed hardware capability.
Read [Backends](backends.md) for what is actually executed and
[Validation](validation.md) for accuracy and soak-test evidence.
