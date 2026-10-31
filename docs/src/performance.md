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

## Matched browser prefill

On 2026-10-31, native Dawn D3D12 and Chrome **156.0.0.0** processed the
same Qwen 3.5 0.8B BF16 seed-1 token prefix on the RTX 5090 Laptop GPU.
Both used 8,192 context positions, f16 KV, stored weights, scalar-f32
prefill, and disabled tuning/speculation. The timed workload is six
64-token steps at positions 64–448, after a 64-token warmup, with
**254 dispatches per step**. Browser scheduler feedback was controlled
only in the measurement harness; real timings were recorded separately.

| Matched, profiled prefill | Native Dawn | Chrome WebGPU |
|---|---:|---:|
| Wall ms per 64-token step | 29.480 | 30.433 |
| Prefill tok/s | 2,171 | 2,103 |
| Summed GPU dispatch ms per step | 26.706 | 26.957 |
| Wall minus GPU sum, ms per step | 2.774 | 3.476 |

Both profiles split dispatches into individual compute passes. The browser
kernel sum is **0.94%** higher, not the large difference initially seen
after idle periods. Its fixed-chunk wall time fell from roughly 48 ms to
**28.494 ms unprofiled** after forty consecutive requests. GPU telemetry
observed a 1,117–2,640 MHz graphics-clock ramp. These clocks were not
locked, so this is not a clock-normalized shader-compiler comparison.

There is no separate browser chunk planner: both hosts use the shared
timing-adaptive scheduler. The earlier 1,024-token, four-native-versus-five-
browser-step result mixed different schedules and boost states.
Measured sustained browser replay is 0.367 ms/step, including 0.144 ms
of API-call CPU. These prefill steps have no model `mapAsync` readback.
The residual includes asynchronous submission/completion and profiling
costs, which are not independently identified here.

Warmed WebGPU commands now replay without per-command promise yields;
pending compilation still waits in order. Short-run replay means of
0.478 versus 0.450 ms do not demonstrate a material throughput gain.
The shared scheduler and shader arithmetic remain unchanged.
See [Browser execution](browser.md#matched-08b-prefill) for the command,
limits and methodology; `benchmarks/qwen35-08b-browser-prefill.json`
contains the committed per-step evidence and exact token prefix.

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

A separate double-buffered expert experiment used four compute waves and
four producer waves, retaining the same 32 KiB shared allocation and
original 128-K high/residual accumulation by consuming two 64-K slices.
It passed standalone `gpu-check`, but was rejected: with the padded dense
kernel below, 1,024-token prefill fell from 178.04 to 163.93 tok/s and
expert GPU time rose from 140.337 to 153.030 ms per 64-token step.
Dense GPU time was 46.753 ms in that experiment, so its regression was
localized to the expert implementation. Decode was 62.18 tok/s.
The four-wave, single-buffered expert remains the production kernel;
the producer-wave experiment is not retained.

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

## Quality-gated native precision ladder

The native loader now probes a fixed 64-position coding/language sequence,
without sampling, against device scalar-f32 operands and f32 attention KV.
It walks KV FP8/f16/f32 first, then supported single-product/compensated/
scalar prefill operands with the selected KV. The combined choice is gated.
Weights remain stored. Dawn has no FP16 matrix rung; Agility exposes real
BF16/F16, float-Q8_0 and grouped TQ2_T LinAlg variants.

On the RTX 5090 Laptop, Agility's isolated rung measurements are:

| Model / changed rung (other class exact) | Top-1 | Mean KL, nats | Max logits rel L2 | Quality result |
|---|---:|---:|---:|---|
| Qwen3.5-0.8B BF16 / FP8 KV | 63/64 | 6.654813e-4 | 4.221842e-2 | Reject: agreement |
| Qwen3.5-0.8B BF16 / f16 KV | 64/64 | 3.765059e-8 | 3.068764e-4 | Pass |
| Qwen3.5-0.8B BF16 / single-product FP16 | 64/64 | 4.928418e-7 | 7.080552e-4 | Pass |
| Qwen3.5-0.8B BF16 / compensated FP16 | 64/64 | 8.882968e-11 | 7.817629e-6 | Pass |
| Gyro-Coder / FP8 KV | 62/64 | 2.063113e-2 | 1.822745e-1 | Reject |
| Gyro-Coder / f16 KV | 62/64 | 4.219834e-3 | 9.464887e-2 | Reject |
| Gyro-Coder / single-product FP16 | 64/64 | 4.823378e-3 | 1.000559e-1 | Reject |
| Gyro-Coder / compensated FP16 | 64/64 | 3.875763e-11 | 2.803597e-6 | Pass |

The policy is top-1 ≥99%, mean KL(P_exact || P_candidate) ≤0.001 and
maximum relative logits L2 ≤0.1. The KL limit separates Gyro's materially
shifted FP8/single-product distributions from compensated arithmetic;
agreement independently rejects Qwen's FP8 candidate even though its mean
KL passes. No existing strict implementation tolerance changes.

Agility therefore chooses **f16 KV / single-product operands** for the
0.8B checkpoint and **f32 KV / compensated operands** for Gyro. The combined
0.8B choice measures 64/64, KL 5.079502e-7 and maximum relative L2 6.951167e-4
against the exact device reference. Cold quality calibration took 1.498 s
and 4.054 s respectively, including CPU metric reduction, not model-file
upload or decode tuning. A valid plan-cache hit skips this work.

`precision-check` audits lower rungs for reporting even after one passes;
ordinary loading stops at the first passing rung. It exports the same
probe tokens and chosen logits for the independent CPU ladder check.
The native benchmark records selected rungs and quality-load time;
`--precision exact` forces f32 KV and scalar-f32 operands.

Independent CPU checks use a true f32 attention-cache oracle on both models.
Gyro's legacy f16 CPU mode is retained for its unchanged strict gates;
comparing its f16 reference with the new f32 choice would instead measure
the KV-format change (62/64 and KL 0.0038077 on this probe), not backend
implementation error. The optional CPU f32 cache fixes that mismatch
without relaxing a bound; the sparse indexer remains f16.

| Backend / model | Chosen KV / prefill | CPU top-1 | CPU mean KL | CPU max rel L2 |
|---|---|---:|---:|---:|
| Agility / Qwen3.5-0.8B | f16 / single | 64/64 | 5.081023e-7 | 6.957935e-4 |
| Agility / Gyro-Coder | f32 / compensated | 64/64 | 1.19016e-11 | 4.18495e-6 |
| Dawn D3D12 / Qwen3.5-0.8B | f16 / scalar | 64/64 | 3.719850e-8 | 3.066005e-4 |
| Dawn D3D12 / Gyro-Coder | f32 / scalar | 64/64 | 4.221473e-11 | 4.897768e-6 |

Dawn's FP8 rung also rejects: Qwen 63/64, KL 6.643438e-4, max relative
L2 4.221790e-2; Gyro 62/64, KL 2.098070e-2, max relative L2 1.779240e-1.
Native FP16 operand rungs are unsupported on Dawn, not scalar aliases.

The same 1,024-token/128-decode-token benchmark at context 8192 gives:

| Backend / mode | KV / prefill | Prefill tok/s | Decode tok/s | Plain decode tok/s |
|---|---|---:|---:|---:|
| Agility / default ladder | f32 / compensated | 167.59 | 57.92 | 58.40 |
| Agility / `--precision exact` | f32 / scalar | 131.01 | 60.87 | 57.90 |
| Dawn D3D12 / default ladder | f32 / scalar | 137.35 | 47.76 | 46.28 |
| Dawn D3D12 / `--precision exact` | f32 / scalar | 149.41 | 47.61 | 49.37 |

Agility's measured prefill improvement over exact is 27.92%. The earlier
178.04 tok/s result used explicitly chosen f16 KV; the new default rejects
that rung on its probe, so it is not the same precision policy. The
default's remaining gap to 250 tok/s is 82.41 tok/s. These are individual
runs with distinct policy/tuning cache keys; both exact loads retuned nine
of eleven decode shapes. Dawn's selected algorithms are identical in the
two modes, so its spread is not an algorithmic precision improvement.
Default benchmark loads restored their recorded choices with zero quality
milliseconds and reported a precision-cache hit.

`zig fmt --check .`, build/test/wasm, the Python harness mutations,
standalone Agility `gpu-check` and `check-fast` on Agility and Dawn D3D12
all pass. `check-fast` now serializes kernel checks, strict exact forward
parity, the CPU ladder check, and Gyro's strict f16-cache cases. Agility's
strict Gyro prefill has zero routing flips, logits median/max relative
L2 1.73e-5/1.24e-4 and bitwise split/unfused results; no old gate changed.

## Agility decode matvec study

On 2026-10-31, the RTX 5090 Laptop ran Gyro-Coder with context 8,192,
f32 KV, compensated prefill operands and stored weights. Before and after
used the same model and commands, with one D3D12 job at a time:

```sh
ZLLM_BACKEND=agility scripts/d3d12.sh benchmark 'C:\models\Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf'
ZLLM_BACKEND=agility scripts/d3d12.sh profile 'C:\models\Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf' 4
```

Both benchmark runs restored their tuning and precision caches. These are
individual runs, not confidence intervals; profile positions 29–45 and
the benchmark's 1,024-token context are different workloads.

| Measurement | Before | Retained Q8_0 cutover |
|---|---:|---:|
| 1,024-token benchmark prefill | 158.43 tok/s | 171.65 tok/s |
| Greedy decode, 128 tokens | 58.61 tok/s; 17.061 ms/token | 62.64 tok/s; 15.965 ms/token |
| Plain decode, 128 tokens | 59.45 tok/s; 16.822 ms/token | 61.10 tok/s; 16.367 ms/token |
| `profile ... 4`, unprofiled four-token step | 36.24 ms | 31.71 ms |
| Same profile's unprofiled one-token decode | 19.87 ms | 18.11 ms |
| One-token decode GPU timestamp sum | 17.573 ms | 16.894 ms |

Greedy and plain decode improve **6.87%** and **2.78%**, respectively.
Q8_0 decode GPU time falls **6.28%**; its four-token time falls **19.57%**.
Large prefill kernels are unchanged, so the benchmark prefill movement is
not attributed to this decode optimization. No expert-kernel speedup is
claimed.

The retained change transposes the Q8_0 shared activation plane: each
lane's four float4 reads become contiguous across the wave rather than
64-byte-strided. Aligned 16-byte weight loads, arithmetic and allocation
sizes remain unchanged. The finish uses the original descending tree via
explicit wave lane reads, not an unspecified wave sum. A 64-lane row
exchanges its high half through shared memory once before that tree.
With tuning disabled to hold WG256/32 lanes/one row fixed, Q8_0 decode
measured 10.136 → 9.749 ms for the transpose, then 9.341 ms with the tree;
four-token Q8_0 measured 18.548 → 16.192 → 14.908 ms.

### Decode kernel table

Dispatch counts are unchanged. Other kernels' timing movements are not
claimed as code improvements:

| Kernel | Dispatches/step | Before GPU ms | After GPU ms |
|---|---:|---:|---:|
| `matvec_q8_0` | 389 | 9.591 | 8.989 |
| `gyro_matvec` | 96 | 3.130 | 3.175 |
| `rmsnorm` | 97 | 0.906 | 0.904 |
| `attn_split_sparse` | 12 | 0.621 | 0.621 |
| `moe_route` | 48 | 0.581 | 0.548 |
| `delta_rule` | 36 | 0.430 | 0.462 |
| `hc_combine` | 96 | 0.382 | 0.303 |
| `rotate128` | 96 | 0.371 | 0.389 |
| `moe_logits` | 48 | 0.328 | 0.326 |
| `hc_mix` | 97 | 0.240 | 0.233 |
| `sample` | 12 | 0.186 | 0.182 |
| `delta_prep` | 36 | 0.156 | 0.154 |
| `moe_combine` | 48 | 0.144 | 0.150 |
| `matvec_bf16` | 12 | 0.119 | 0.114 |
| `gated_norm` | 36 | 0.102 | 0.102 |
| `qsa_keys` | 12 | 0.073 | 0.047 |
| `rope_append` | 12 | 0.056 | 0.053 |
| `qsa_pool` | 12 | 0.043 | 0.031 |
| `attn_combine` | 12 | 0.037 | 0.036 |
| `qsa_select` | 12 | 0.022 | 0.022 |
| `qsa_score` | 12 | 0.021 | 0.021 |
| `ple_mix` | 1 | 0.015 | 0.015 |
| `embed_q8_0` | 2 | 0.011 | 0.011 |
| `ple_conv` | 1 | 0.007 | 0.008 |
| **Sum** | **1,235** | **17.573** | **16.894** |

### Per-shape catalog

The existing sixteen-candidate sweep remains; no RTX-specific heuristic
or new candidate survived the measurements. The retained run's Q8_0
catalog is below. `n` is the full matrix's row count, including both
halves of fused gate/up weights; the launch may produce half that many
outputs. Default rows are shown explicitly even though the cache omits them.

| n | k | WG | Lanes/row | Rows/lane group |
|---:|---:|---:|---:|---:|
| 248,320 | 2,560 | 128 | 32 | 2 |
| 320 | 10,240 | 256 | 64 | 1 |
| 10,240 | 320 | 256 | 8 | 1 |
| 324 | 10,240 | 256 | 64 | 1 |
| 16,480 | 2,560 | 256 | 16 | 1 |
| 2,560 | 6,144 | 256 | 32 | 1 |
| 1,280 | 2,560 | 256 | 32 | 1 |
| 2,560 | 640 | 256 | 16 | 1 |
| 10,240 | 2,560 | 256 | 16 | 2 |
| 2,560 | 2,560 | 128 | 16 | 1 |
| 13,312 | 2,560 | 256 | 16 | 1 |

These are measured cache entries for this device/model/source combination,
not portable defaults. Changing HLSL invalidates the existing source-keyed
cache. WGSL and its variant contracts remain unchanged.

### Rejected experiments

Trials below used `profile ... 4`; times are the decode section unless
noted. Uncontrolled catalog sweeps and device timing vary, so small
movements were not sufficient reason to retain extra code.

| Experiment | Observed GPU ms/step | Disposition |
|---|---:|---|
| Q8_0 direct global activation loads, no staged tile | Q8_0 10.934; four-token 27.201 | Reject: baseline 9.591 / 16.690 |
| Three signed-byte residual digits with DP4a | Q8_0 9.344; four-token 16.672 | Reject: no clear gain over scalar trials, which reached 7.897 |
| Wave tree without transposed staging | Q8_0 8.046 | No isolated win against that 7.897 scalar trial; retain only the later fixed-variant combined win above |
| 36 candidates: add WG64 and four rows/lane group | Q8_0 9.494; tuning 7.791 s | Reject: none of the eleven shapes selected a new candidate |
| Four interleaved expert-row accumulators | Experts 3.111 | Reject: immaterial against 3.130 baseline |
| WG1024, one expert row per wave | Experts 3.816 | Reject |
| Vector-load/vector-store expert activation staging | Experts 4.482 | Reject |
| Pad one scalar per 32 expert activations | Experts 3.250 | Reject |
| Transpose expert activations and rotate layer matrices during tuning | Q8_0 8.994; experts 3.085 | Reject combination: benchmark 50.90 greedy / 58.64 plain tok/s |

The DP4a prototype passed the unchanged 1e-5 kernel gate: Q8_0 norm
relative L2 7.44e-7 and long tile-tail/pair relative L2 6.49e-7, including
bitwise token splits. It was removed for performance before running the
full model gate; it is not claimed to have passed that gate. Expert
activation transposition alone was not a repeatable decode win and was
also removed. The final expert shader and tuner are unchanged.

### Dispatch graph versus gyro-dx

Source inspection of `gyro-dx/shaders/engine/mv.hlsl` found the same
aligned 16-byte-per-lane Q8_0 weight loads; its expert kernel in
`moe.hlsl` already matches zllm's 32-row/four-rows-per-wave organization
closely. Merely substituting its wave sum or increasing load width was
not a demonstrated improvement.

For a resident-expert, one-token, logits-producing step with this model's
36 recurrent and 12 full-attention layers, source-derived counts are:

| Implementation | Actual D3D12 dispatches | Global UAV barriers |
|---|---:|---:|
| gyro-dx | 1,556 + P + 2S | 966 + P + S |
| zllm Agility, normal hazard elision | 1,231 | 1,115 |

`P` counts full-attention layers completing a pooled indexer block and
`S` counts those needing sparse selection; each is 0–12. Thus gyro-dx's
range is 1,556–1,592 dispatches and 966–990 barriers. zllm's figure
includes eight compute-copy dispatches beyond its 1,223 model kernels.
The stochastic sampler used by `profile` adds twelve launches: counters
observed 1,241 operations in the main compute list, two upload copies in
the preceding list, and 1,127 normal barriers. Profiling forces 1,240
barriers in the main list to isolate dispatch times.

Temporary timestamps around the **complete** main compute list, including
copies and barriers, measured the retained kernel/catalog combination:

| Diagnostic measurement | Normal one-token step | Mean of 16 profiled steps |
|---|---:|---:|
| Wall time, including host work | 18.07 ms | 18.62 ms |
| Complete main-list GPU span | 15.930 ms | 17.200 ms |
| Sum of individually timed model kernels | Not instrumented | 16.815 ms |
| GPU span outside those kernel intervals | Not isolated | 0.386 ms |
| Wall time outside the main-list GPU span | 2.140 ms | 1.420 ms |

The 0.386 ms includes six untimed compute-copy operations, barriers and
timestamp/dispatch gaps; it is **not** an isolated barrier-cost estimate.
The wall remainder also includes uploads, DirectStorage dependencies,
command handling and diagnostic logging, not just GPU idle time. No
comparison of one unprofiled step with a later kernel sum is used to
attribute idle time. All temporary counters and timestamps were removed;
the original host implementation was restored exactly.


Both implementations re-record command lists every step; neither replays
a D3D12 bundle. Both keep pipelines and bindless descriptors. Resident
gyro-dx submits one compute list and waits once; zllm submits an upload
list plus the main list and waits for the final ticket. Its hazard
tracking already elides independent accesses, but its four-dispatch
hyperconnection mixer has more boundaries than gyro-dx's fused two-pass
`hc_pre`/`hc_post`. Conversely, zllm has five recurrent dispatches versus
twelve, ten full-attention dispatches versus at least seventeen, and nine
FFN dispatches versus thirteen. Fewer total launches do not imply fewer
barriers.

Nonresident gyro-dx is a different dispatch graph: it removes the 48
resident `moe_resolve` launches but synchronizes after every layer's
routing. Prediction disabled/enabled gives 1,508/1,602 + P + 2S
dispatches and 966/1,013 + P + S barrier calls, respectively, with
**49 compute submit/wait pairs per token** before any storage-miss waits.
These are source-derived branches, not the established settings of the
external 17.6 ms measurement.


The cited gyro-dx 17.6 ms/step uses another harness; it is not used as the
before baseline above or claimed to have identical KV, context, sampling
and residency settings. No speculative barrier removal was retained.

### Acceptance

The retained implementation passed `zig fmt --check .`, `zig build`,
`zig build test`, `zig build wasm`, standalone Agility `gpu-check`, and
both required `check-fast` runs with the Gyro model on Agility and Dawn
D3D12. No gate or tolerance changed. The new Q8_0 4,160-wide norm/SiLU
tile-tail case has relative L2 2.29e-7 at the existing 1e-5 kernel bound.
Every supported quantized matvec candidate now checks multi-token versus
single-token results bitwise; a fixture-free mutation test rejects
missing, reordered and rounded checker outputs.

Both strict Gyro runs retain zero expert-set flips, 66/66 top-1 and
40/40 greedy agreement, with bitwise split-step and unfused results:

| Backend/path, strict f16-KV reference | Streams max rel L2 | Logits median / max rel L2 |
|---|---:|---:|
| Agility prefill | 1.11e-4 | 1.73e-5 / 1.24e-4 |
| Agility decode | 1.49e-4 | 1.82e-5 / 1.32e-4 |
| Dawn D3D12 prefill | 1.12e-4 | 1.86e-5 / 1.47e-4 |
| Dawn D3D12 decode | 1.36e-4 | 1.97e-5 / 1.53e-4 |


## Nanbeige memory-aware ladder and packed-dot prefill

The 2026-10-31 remeasurement uses
`Nanbeige4.3-33B-AD-Q4_K_M.gguf` (21,161,159,680 bytes; fixture revision and
SHA-256 in `benchmarks/nanbeige4.3-33b.json`) on the RTX 5090 Laptop GPU.
Each native benchmark warms up, then prefills the same seed-7 1,024-token
prompt and greedily decodes 128 tokens; default load-time tuning remains
enabled. Only one GPU job ran at a time, through `scripts/d3d12.sh`.

The ladder now counts actual aligned resident weight and program buffers
at the requested context and reserves **5% of the device budget, minimum
512 MiB**. It skips non-fitting rungs before quality trials. The cache key
includes context and budget; JSON retains the estimates. The gate, its
thresholds and rung order are unchanged. See [Precision](precision.md#memory-aware-selections-on-the-rtx-5090-laptop)
for every measured selection and the disclosed quality-limited fallbacks.

| Backend / context | KV before → after | Prefill tok/s before → after | Decode tok/s before → after | Peak device MiB before → after |
|---|---|---:|---:|---:|
| Dawn D3D12 / 8,192 | f32 → f16, warned fallback | 36.78 → **197.82** | 9.72 → **23.10** | 23,667 → **23,435** |
| Agility / 8,192 | f32 → FP8, warned fallback | 22.68 → **204.30** | 12.30 → **21.22** | 23,865 → **22,716** |
| Dawn D3D12 / 4,096 | f32 → f32 | 194.15 → **194.92** | 22.78 → **22.34** | 23,447 → **23,399** |
| Agility / 4,096 | f32 → f16, warned fallback | 33.20 → **202.13** | 21.39 → **22.45** | 23,561 → **22,678** |

All after selections use scalar-f32 prefill operands. These Nanbeige
fallbacks are **not quality passes**: both f16 and FP8 fail the unchanged
calibration gate, and the most accurate fitting rung is used with a clear
warning. Dawn's physical-heap bound and Agility's live DXGI budget differ,
so their fitting rungs need not agree.

Peaks are `nvidia-smi` whole-device used memory sampled every 200 ms,
including desktop/other processes and load-time allocations; they are not
process-only allocation measurements. Before/after baselines are retained
in JSON. Residency/eviction can limit the observed peak, so the small Dawn
peak difference is not the f32-to-f16 allocation saving. Browser f16's
previous 22.6 decode tok/s remains a separate browser workload, not a
substitute for these native results.

### Agility 4,096 prefill cause

The residual anomaly was **not memory pressure**. Agility's ordinary
quantized matmul helper expanded each packed signed-byte dot into two
`int4` unpacks, four multiplies and three additions, while WGSL used
`dot4I8Packed`. The retained HLSL helper uses the exact SM 6.4
`dot4add_i8packed(a, b, 0)` operation; the existing Agility baseline is
SM 6.6. This changes no quantizer, rounding or precision policy.

A controlled rollback changed **only that helper** back to its manual
implementation while retaining the memory-aware ladder, 4,096 context,
f16 KV, scalar operands, identical allocation estimates and DXGI budget:

| Controlled Agility run | 1,024-token prefill | Steps | Decode | Peak device MiB |
|---|---:|---:|---:|---:|
| Memory fix + old manual packed dot | 33.96 tok/s | 126 | 22.59 tok/s | 22,678 |
| Memory fix + native packed dot | **202.13 tok/s** | **33** | 22.45 tok/s | **22,678** |

The **5.95×** prefill gain at identical peak memory isolates the integer-dot
implementation. The slow fixed-cost 32-token tiles also kept the existing
submission scheduler on smaller partial chunks, multiplying the wasted
tile work; no submission-budget threshold was relaxed. The temporary
rollback was removed. A source regression test requires the native signed
packed-dot intrinsic for all five ordinary quantized matmul identities.

Acceptance passed `zig fmt --check .`, `zig build`, `zig build test`,
`zig build wasm`, and `check-fast` on both Agility and Dawn D3D12 with
`-Dgyro-model=/mnt/c/models/Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf`.
The new non-fitting-rung unit test also killed an actual mutation replacing
its fit predicate with unconditional acceptance; production filtering was
restored. Strict Gyro checks retain zero routing flips, 66/66 top-1,
40/40 greedy agreement, and bitwise split/unfused results on both backends.

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
