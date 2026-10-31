# GPU backend

The GPU path has three layers: the shared Zig core records a **command
stream**, a backend **host** executes it, and backend-specific **kernels**
perform the work. WebGPU uses WGSL in the browser and through native Dawn;
[Velocity](velocity.md) uses hand-written HLSL on native D3D12. Both run the
same model plans, scheduler, sampler, tuner and fusion legality rules.

The plans are **step programs**: buffers, bindings and launch lists built
once, then replayed for prefill or decode against KV caches and recurrent
state. They cover Qwen 3.5, the supported dense models and
Qwen3.8-Flash-Next. See [Architecture](architecture.md),
[Generation](generation.md) and the [backend matrix](backends.md).

## Hosts

The existing documentation records these exercised host paths:

| Host | GPU API path | Recorded adapter | Use |
|---|---|---|---|
| Native Linux | Dawn → Vulkan | llvmpipe (Mesa's software renderer) | Kernel and forward-pass correctness |
| Windows, run from WSL | Dawn → D3D12 | RTX 5090 Laptop | Native parity and timing |
| Browser | Chrome WebGPU → D3D12 | RTX 5090 Laptop | Browser inference and parity |
| Native Windows | Velocity → D3D12 | RTX 5090 Laptop | HLSL parity, DirectStorage and native optimization |

WebGPU also provides Metal and hardware Vulkan paths, but the existing
validation record does not establish runs on those adapters. The current
WGSL kernels use portable compute and packed integer operations, not native
tensor-core or matrix intrinsics. Velocity's capabilities and remaining
matrix-kernel work are described separately in [Velocity](velocity.md).

Dawn comes from the pinned prebuilt release. Its static archive is wrapped
into a shared library (`libwebgpu_dawn.so`, `webgpu_dawn.dll`) that exports
exactly the `webgpu.h` C API, and zllm loads it at run time. zllm never links
Dawn's C++, and the vendored header in `third_party/dawn` fixes the ABI.

```sh
scripts/fixtures.sh dawn          # Linux .so; under WSL with Visual Studio, also the DLL
zig build gpu-check               # every kernel vs a CPU computation (llvmpipe)
zig build gpu-check -Dgpu-host=d3d12
zig build check-gpu               # forward pass vs the CPU reference, step splits, engine
zig build check-gpu -Dgpu-host=d3d12
```

The D3D12 host needs Visual Studio (for `link.exe`) and the Windows SDK (for
DXC's `dxcompiler.dll`). Windows won't load DLLs from a `\\wsl.localhost`
path, so `scripts/fixtures.sh` stages the DLLs in `%LOCALAPPDATA%\zllm\gpu`.
`scripts/d3d12.sh` builds the Windows executable, copies it there and runs
it. On WSL, Dawn's Vulkan backend rejects Dozen (Vulkan on D3D12), so the
native Linux host falls back to llvmpipe.

## Command stream

The core never calls a GPU API. It appends commands to a `Recorder`
(`src/gpu/cmd.zig`), a host executes the batch, and the core consumes one
completion per `submit` and per `read_buffer`. The native platform interface
(`src/gpu/host.zig`) selects Dawn or Velocity. The browser host
(`web/src/zllm/gpu.ts`) interprets the same words, crossing the wasm/JS
boundary once per batch rather than once per dispatch.

- **Encoding:** little-endian `u32` words. Each command starts with
  `opcode | (words << 8)`, so a host can bounds-check every command.
- **Object ids:** objects are named by ids the core assigns, densely per kind.
- **Addresses:** buffer contents, kernel artifacts and readback destinations are
  core-memory addresses, two words each; in wasm32 the high word is 0.
- **Batching:** commands between two `submit`s go into one fresh command
  buffer, and consecutive dispatches share one compute pass.

| Op | Arguments |
|---|---|
| `create_buffer` | id, size, usage (WebGPU `GPUBufferUsage` bits) |
| `write_buffer` | buffer, offset, data, length |
| `create_kernel` | id, kernel, variant, the bindings its contract writes, artifact (the backend's implementation: WGSL with entry point `main` on WebGPU, HLSL on Velocity; empty where the backend lacks the kernel) |
| `create_bind_group` | id, pipeline, group, then (binding, buffer, offset, size)… |
| `dispatch` | pipeline, workgroups x/y/z, bind groups |
| `copy_buffer` | src, offset, dst, offset, size |
| `read_buffer` | token, buffer, offset, size, destination |
| `submit` | token |
| `destroy` | kind, id |
| `clear_buffer` | buffer, offset, size (zero-fill, ordered with dispatches) |
| `write_file` | buffer, offset, file offset, size |

Weights never pass through core memory. Each upload is a `write_file`: the
host copies a range of the model file it opened into the buffer. Natively
that range comes from a mapping of the file; in the browser it comes from
range reads of a URL or a local file. The core needs only the GGUF header.

Dawn stages each queued write in system memory until the GPU has copied
it. A load is one batch, so the native host flushes after every 256 MiB of
writes: it submits what is encoded and waits for the previous flush, which
keeps staging under 512 MiB while reading continues. Unflushed, the
Gyro-Coder's 20 GB of weights held 17 GB of staging at once, which Windows
counts as shared GPU memory, on top of 22 GB of dedicated memory.

## Kernels

The WebGPU implementations live in `src/kernels/*.wgsl`. Kernel identities,
uniform parameter structs and binding contracts are in `kernels.zig`;
Velocity supplies the corresponding HLSL artifacts. Weights retain their
BF16, F16 or supported quantized values and are decoded in the shader.
Activations and accumulation are f32, except that quantized prefill
multiplies round their input to 8 bits as llama.cpp does (below).

Three weight layouts change at load:

- Q6_K blocks grow from 210 to 224 bytes, keeping 16-byte vector loads aligned.
- Q3_K blocks are repacked from 110 to 128 bytes: eight self-contained
  16-byte units with values, high bits, scales and `d`.
- Q8_0 matrix rows put their i8 values first, then their f16 scales, padded
  to 16 bytes. Decode tasks read 16 values in one aligned load. PLE rows
  read directly from the file keep the original 34-byte blocks.

| Kernel | Work | Notes |
|---|---|---|
| `embed_bf16`, `embed_f16` | Token embedding gather | Per table chunk |
| `rmsnorm` | Row RMSNorm | One workgroup per row |
| `matmul_bf16`, `matmul_f16` | x · Wᵀ, prefill | 16 tokens × 64 outputs per workgroup; tiles staged in workgroup memory. Optionally adds into the output (the residual) |
| `matvec_bf16`, `matvec_f16` | x · Wᵀ, decode | The half-precision kernels are `matvec_half.wgsl`, `matmul_half.wgsl` and `embed_half.wgsl`, composed with `half/bf16.wgsl` or `half/f16.wgsl`, which widen a word's two values. 8 rows per workgroup, each row's dot product over 32 lanes reading 16-byte weight vectors, then a fixed tree. Optionally RMSNorms the staged input first, adds into the output, or computes SiLU(gate) · up over stacked gate and up rows |
| `silu_mul` | SwiGLU over the stacked gate and up projection, prefill | Elementwise |
| `rope_append` | Q/K RMSNorm (optional), partial RoPE (NeoX pairs, or the dense models' normal pairs) at positions pos0 + t, and the K/V cache write | One workgroup per (head, token) of the stacked Q, K, V projection, whose query heads are interleaved with output gates (Qwen 3.5) or not (the dense models). Reads a host-built (cos, sin) table: WGSL trig is only accurate near [−π, π]. The cache holds f32, f16 pairs, or scaled E4M3FN FP8 |
| `attn_split`, `attn_combine` | Causal GQA over the cache, with the sigmoid output gate where the model has one (flash-decoding) | One workgroup per (KV head, 64 cache positions, token) scores the block for the whole GQA group and takes a local softmax (groups of up to 8 query heads; `attn_split_wide` takes up to 12, whose queries fill WebGPU's 16 KB of workgroup memory, which costs the 27B's prefill 2% if used for its groups of 6); the combine merges the blocks. Prefill steps run them in chunks of 32 tokens, so the partials hold one chunk; a token's result depends only on its position. The 27B's prefill attention at position 5,000 dropped from 96 to 61 ms a step when this replaced a kernel with one workgroup per (head, query) |
| `qsa_keys`, `qsa_pool`, `qsa_score`, `qsa_select`, `attn_split_sparse` | Qwen3.8-Flash-Next's sparse attention: the indexer's queries (norm, RoPE) and keys (an f16 cache), each block of `ratio` keys pooled (mean, norm, RoPE at its start), every block's score Σ_heads relu(q · block), the top blocks, and attention over the selected cells | Exact selection: the scores are never negative, so their bits order as their values; one workgroup per row finds the K-th largest bit by bit and breaks ties by index with a scan. `attn_split_sparse` is `attn_split_wide` with an inclusion test composed in (`attn_sparse.wgsl`; `attn_dense.wgsl` for the others). An excluded key weighs 0, even where a whole 64-key block is excluded |
| `delta_prep` | DeltaNet front end: causal conv and SiLU, L2 norm of q and k, β and decay gates | One workgroup per (head of channels, token); inputs before the step come from the conv history |
| `delta_rule` | Gated delta rule | Each thread keeps 16 rows of one state column in registers for the whole step; a head spans 4 workgroups. Also advances the conv history, and can keep the state and history as after one row of the step (for rolling back a rejected speculative token) |
| `gated_norm` | The SiLU(z)-gated RMSNorm after it; sigmoid(z)-gated for Qwen4Exp | One workgroup per (head, token); z read from the stacked projection |
| `matvec_<q>`, `matmul_<q>`, `embed_<q>` | The multiplies and embedding gather on quantized weights: `q8_0`, `q3_k`, `q4_k`, `q5_k`, `q6_k` | `matvec_quant.wgsl.hbs`, `matmul_quant.wgsl` and `embed_quant.wgsl`, composed with their type's decoder. The decode template's threads, threads per row and rows per lane group are `MatvecVariant` parameters, with 12 candidates. It decodes the stored quantized values, stages x in 2,048-float tiles and keeps the norm, SiLU and residual fusions. Q8_0 splits each 32-weight block between two lanes with one aligned 16-byte value load each; K-quants use their own `quant/<q>_dot.wgsl` decomposition. Multi-token steps pair inputs in 1,024-float tiles and dot decoded weights with both (`block_dot2`), preserving each token's one-step sums. The prefill multiply uses 8-bit activation blocks and exact integer sub-block sums (`dot4I8Packed`), then applies scales: 32 tokens × 128 outputs per workgroup, 4 × 8 per thread, with the next unit fetched and decoded while the current one computes |
| `quantize_q8` | Rounds a quantized prefill multiply's input to 8-bit blocks, computing it first where it is an RMSNorm or the SiLU pair, so those kernels don't run | As llama.cpp's CPU backend does for the weight type: Q8_K blocks of 256 for the K-quants, Q8_0 blocks of 32 for Q8_0. Bitwise as the reference's `--llama-activations` mode |
| `matmul_reduce` | Adds up a quantized prefill multiply split over k | A matrix with too few column tiles to fill the GPU (a 5,120-row projection has 40) splits k into slices, aiming at 256 workgroups; each slice writes partial sums, and this adds them into the output |
| `repad`, `repack_q3_k`, `repack_q8_0` | Lay out Q6_K (padded) and Q3_K (repacked) blocks, and Q8_0 split rows (the row's values as i8, then its scales as f16, padded to 16 bytes), at load | Each chunk's rows go to a staging buffer, freed after the chunk's own submit |
| `mark_tokens` | The prompt's token set, for the repetition penalty | |
| `sample` | Bias, penalties, temperature, min-p, top-k, top-p, draw | In phases, one dispatch each; see [Generation](generation.md#sampling). One more phase undoes the last draw's output count (a rejected speculative round's second sample) |
| `draft` | The speculative draft: the first index of the maximum of a row of logits, into the next step's token input | One workgroup |
| `gyro_matvec` | Routed-expert products on Gyro TQ2_T weights (Gyro-Coder) | The expert comes from the router's GPU buffer, with no readback. A workgroup computes 32 rows; a lane per trellis step decodes four weights from the 2,048-pair codebook; gate and up share a dispatch. Velocity stages the codebook and uses lane exchange and wave sums; WGSL reads the codebook from storage to stay within the guaranteed workgroup-memory limit |
| `moe_logits`, `moe_route` | Expert routing: router · x (f32) and the shared expert's gate logit, then softmax, top-k (ties to the lower index) and normalized weights | `moe_logits` assigns a workgroup per router row with vector loads, then reduces with a tree in WGSL or wave sums in HLSL. `moe_route` performs selection in one workgroup per token |
| `rotate128` | H₁₂₈ · (signs · x) per 128-value block: the basis Gyro expert weights are stored in. Optionally of silu(gate) · up | 64 butterflies a stage in workgroup memory |
| `moe_combine` | The expert block's output: the routed experts' weighted sum plus the gated shared expert | |
| `hc_act`, `hc_mix`, `hc_combine` | The hyper-connection mixer around Q8_0 down, up and inject products: silu(lo / hc); the streams' gated mean into a block's input; a block's output added to each stream, weighted 2·sigmoid(inject / hc) | The streams' norms are `rmsnorm` with a weight row per stream. `hc_act` runs fused, as the up multiply's staging prologue (`MatvecVariant.act_div`), where the up weights are quantized: one dispatch and one buffer fewer per mix, 97 per decode step (Gyro-Coder: 29.5 → 28.4 ms) |
| `ple_mix`, `ple_conv` | The PLE block: per stream, the signed-root sigmoid gate of normed key · query, the gated value and its norm; then the causal dilated convolution (4 taps, dilation 3) into the residual streams | The convolution runs a step's tokens in order against a 9-row history, which it advances as `delta_rule` does the conv history |

The quantized decode multiplies are hand-written templates rendered by
`src/tmpl`, not generated shader algorithms. Load-time tuning selects their
launch variants; the rest of the catalog is not yet generally autotuned.
`zllm bench` measures step throughput
([Generation](generation.md#performance)).

### Kernel identities and contracts

`create_kernel` names a `kernels.Kernel` and its variant, carries the binding
write set, and supplies the backend artifact. `Kernel.contract()` derives
the binding count and writes from WGSL declarations. Every implementation
keeps the same parameter layout and binding order; a native implementation
may use a different reduction strategy within the checks' tolerances.

Velocity's catalog has implementations for every current kernel. Missing
implementations are represented by `kernels.Set.missing`, not replaced by
another backend: dispatch fails with `error.UnsupportedKernel`, and the
native model loader calls `requireAll()` before loading. Contract tests
check that HLSL writes only bindings declared writable. See
[Velocity's ordering rules](velocity.md#the-native-layer).

## Tuning

After the weights are on the GPU and before the step program is built,
`src/tune` sweeps the quantized decode multiply's variants for
each weight shape in the model, the MTP block's included:

- **Candidates:** the 12 `MatvecVariant`s, less those past the device's
  limits, the default first.
- **Correctness first:** each candidate's output on the shape's real weights
  must be within 1e-5 relative L2 of the default variant's; one that is not
  is dropped.
- **Timing:** 2 more warm-up batches, then the median of 7. Batches are
  timed from submission to completion, which every host measures (the
  spec's fallback for adapters without timestamp queries), and repeat the
  dispatch to last about a millisecond each.
- **Steps:** two tokens where the engine speculates by default, else one.

The winners go into the step program's `Tuning`. This is a sweep of
quantized decode matvecs, not a search over all operators or precisions.
The program then selects the tuned variant while building launches,
including launches with fused prologues. There is no post-fusion sweep.

The following measurements are the earlier WebGPU tuning record on the
RTX 5090 Laptop, not a benchmark of the current revision: all 16 of the
27B's shapes took another variant, the K-quants 8 or 16 threads per row
instead of 32 and Q8_0 two rows per lane group. Speculative decode went
from 40.7–41.5 tok/s to 47.0–47.3; one token per step went from 30.0–30.4
to 31.3–31.9. Tuned logits remained close to the CPU reference (6.4e-6
relative L2 at most, against 6.2e-6). That sweep took 13 s. `--no-tune`
skips the sweep and native tuning-cache lookup.

### Device lost

A lost device (a driver reset, a GPU hang, a removed adapter) fails the
request in flight with `error.DeviceLost`. The shared native runtime then
rebuilds before returning: a new device on the selected backend, uploaded
weights, tuning from the plan cache when available, and a fresh engine with
no resident sequence (`Runtime.recover`). This applies to Dawn and
Velocity; Velocity maps D3D12 removed/hung/reset errors to the typed error
and allows teardown of a removed device. An unavailable cache means a new
sweep, not preserved tuning in GPU memory.

`zllm chat` reports the loss and asks for the message again. The existing
Dawn measurement rebuilt the 0.8B on the RTX 5090 Laptop in 2.2 s.
`engine-check` forces device loss and checks the typed error, cached rebuild,
and bitwise tokens and logits after recovery. The browser rejects the lost
request and rebuilds before the next request, using the same model source
and cached tuning ([Browser](browser.md#device-lost)).

### Plan cache

The CLI keeps the sweep's non-default variants in the plan cache
(`src/plan/cache.zig`): one JSON file named by its key, in
`%LOCALAPPDATA%\zllm\plans` on Windows and `$XDG_CACHE_HOME/zllm/plans`
(or `~/.cache/zllm/plans`) elsewhere. `ZLLM_PLAN_CACHE` names another
directory, or disables the native cache when empty.

The SHA-256 key covers the adapter strings and API, the runtime (`dawn` or
`velocity` natively), plan binding limits, the model's GGUF header and
tensor index, the zllm version, and every `Kernel.source()` WGSL/template
source. It does not hash tensor payloads or Velocity's HLSL sources. Thus
an HLSL-only change with an unchanged zllm version can leave a Velocity
cache valid under the current key.

A valid hit skips tuning. A different key or malformed/unknown variant is
ignored and a fresh sweep replaces it. The cache is not compiled pipelines
or a serialized step program; it holds no precision decisions. Loads still
create pipelines and upload weights. The earlier Gyro-Coder cache record
skipped 2.5 s of tuning with the same decode output and rate.

The browser tunes at load too, through the same sweep, between the
weights and the step program. The shim gets the key from the core
(`zllm_plan_key`, with the adapter's strings from `GPUDevice.adapterInfo`
and the browser's user agent as the runtime) and keeps the file in the
origin private file system, under `zllm/plans/`. In Chrome the Gyro-Coder's
sweep adds 8 s to its first load and none to later ones.

## Step program

`src/plan/qwen35.zig` builds the Qwen 3.5 and supported dense models'
steps once; `src/plan/qwen4exp.zig` builds Qwen3.8-Flash-Next's. The shared
`plan.Program` interface lets the scheduler drive either. The Qwen 3.5
program includes:

- **Buffers:** activations sized for the largest step (512 tokens), a KV
  cache per attention layer sized for the context, every DeltaNet layer's
  recurrent state and conv history in one buffer, checkpoint copies of that
  buffer, and the sampler's buffers. With MTP, also the MTP block's own KV
  cache, and a second copy of the recurrent buffer that `delta_rule` fills
  with the state as after a step's kept row (`Step.keep`): rolling back to
  that row is one buffer copy. A speculative round records the MTP block's
  step over the last step's positions, its draft head over the last row
  only and the `draft` argmax, then the verify step: the token and the
  draft through the decode kernels, both rows' logits, and the sampler twice
  (the second launch set reads logits row 1 with its own draw).
- **Launches:** every dispatch with its bind group and parameters, which are
  packed into shared 64 KiB uniform buffers.

A step is then one `write_buffer` of a 16-byte `Step` uniform
(`m` tokens, first position `pos0`, the sampler's draw `u`, the row to keep),
the token ids for prefill, and a replay of the launch list:

- **Step-dependent kernels:** kernels that mix tokens across the step or
  use positions (RoPE and the cache write, attention, the delta rule and the
  conv history it advances, the sampler) read the `Step` uniform.
- **Row-local kernels:** the rest take capacity-sized parameters, and their
  grid covers only the step's `m` rows.
- **Logits:** decoding projects the residual stream directly, with the
  final norm in the multiply's prologue. Otherwise the final norm's last rows
  are copied into the projection's input, so a step projects only the rows
  it needs, not 512.

One `submit` per step keeps the step uniform unambiguous: queued writes land
before the submission they precede.

### Two kernel paths

Steps of at most 4 tokens, decoding, run the **decode kernels**:
the half-precision or quantized `matvec` kernels for each weight chunk,
with supported fusions, and split attention. Longer steps run the
**prefill kernels**. The plan prebuilds the
launches that differ both ways (`plan.qwen35.Fork`), and a step picks one
(`plan.qwen35.Path`).

Within a path, every row's arithmetic is the same whatever the step's size.
A query walks the same key tiles from position 0, and each multiply output
reduces in the same order, so splitting a sequence into steps of any size
gives bitwise identical results. The two paths reduce in different orders,
so a position processed through one differs from the same position through
the other by rounding, about 1e-5 relative. `check-gpu` checks both paths
against the CPU reference, and each path's splits bitwise.

Every storage binding must fit `maxStorageBufferBindingSize`, which is
128 MiB on llvmpipe and 2 GiB on the 5090:

- **Weight matrices** are split by rows into chunks; each chunk is its own
  buffer and its own dispatch.
- **The vocabulary projection** runs over at most `logit_rows` rows at a
  time (1 when generating). A check that needs every position's logits gets
  them in passes.
- **Recurrent state** is bound per layer at offsets aligned to
  `minStorageBufferOffsetAlignment`.
- **The stacked gate and up matrix** must fit one binding, so the decode
  multiply can read both rows of a pair. The plan fails to build otherwise.
- **Weight types** are per chunk. Stacked weights share chunks while their
  parts have one type and start new ones where it changes; each launch
  picks its kernel from its chunk's type.

### Fusion

The plans construct several fusions directly. Weights that share an input
are stacked by rows when uploaded, so each set runs as one multiply, and
later kernels read their part of the output at a row stride:

- Q (with its output gates), K and V
- the DeltaNet projections: qkv, z, β and α
- the FFN's gate and up

| Rule | Decode kernels | Prefill kernels |
|---|---|---|
| RMSNorm into the next multiply | In the decode multiply's prologue: each workgroup reduces the row's sum of squares itself | Half-precision: separate `rmsnorm`; quantized: can be folded into `quantize_q8` |
| Gate, up and SiLU-multiply | One decode multiply over row pairs | One multiply, then `silu_mul`, or SiLU folded into `quantize_q8` for the next quantized multiply |
| Residual add into the output and down projections | The multiply adds into the residual stream | The same |
| RoPE and the cache write after QKV | `rope_append` | The same |
| DeltaNet conv, gates and L2 norm before the scan | `delta_prep` | The same |
| Final norm and the vocabulary projection | Norm in the projection's prologue | Half-precision: separate `rmsnorm`; quantized: norm can be folded into activation quantization |

A fusion that needs workgroups to synchronize is not legal, which rules out
three more:

- **RoPE inside the QKV multiply:** a head's norm spans the output rows of
  32 workgroups.
- **The gated norm inside the scan:** a head's state spans 4 workgroups.
- **Top-k inside the vocabulary projection:** the sampler's selection runs
  over the whole vocabulary.

The existing Qwen3.5-0.8B fusion record reduced a decode layer to 7
dispatches and a step from 441 to 182. Those counts are model-specific,
not counts for every supported architecture.

There is also an implemented fusion pass in `src/fuse/fuse.zig`. The
Gyro-Coder hyper-connection mixer describes `hc_act` and its following
decode multiply as buffer operations. `fuse.lower` folds
`silu(x / hc)` into the quantized matvec's staged-input prologue only when:

- the activation output has one reader, with no external `live_out` reader;
- that reader is a later decode multiply over full vector rows;
- every weight chunk has the quantized prologue (half-precision chunks
  prevent this fusion);
- the multiply has no competing norm/activation prologue.

Unfused operations lower to ordinary launches. `plan.Config.fuse` and
`gpu-forward --no-fuse` select the comparison path; `check-gyro` compares
fused and unfused records bitwise. This is a focused pass, not arbitrary
cross-workgroup fusion or a complete implementation of every planned rule.

## Qwen3.8-Flash-Next

`src/plan/qwen4exp.zig` is the Gyro-Coder's step program. It is built from
the parts it shares with Qwen 3.5's (`src/plan/common.zig`: weight upload,
buffers, the multiplies' launches, the sampler). The engine drives either
through `plan.Program`. This program runs the decode kernels only, in steps
of up to 4 tokens, with up to 32,768 positions of context. Its attention
is sparse: past 2,051 visible cells (top-k 2,048 + ratio 4 − 1) each query
attends to its tail and the 4-cell blocks its indexer scores highest, about
2,051 cells in all, whatever the position. Each attention layer has its
indexer's weights (query heads and key, one BF16 multiply), an f16 key
cache and its pooled blocks; the rows' scores and selections are shared.

- **Weights:** the dense Q8_0 matrices as Qwen 3.5 uploads them, stacked
  where they share an input (the DeltaNet projections; Q, K and V; the
  shared expert's gate and up; the hyper-connection matrices). The routed
  experts go up as stored, TQ2_T, read by `gyro_matvec` at the ids the
  router wrote; then the router, the codebook and the Hadamard sign
  vectors. Everything but the PLE table is about 21 GB of GPU memory.
- **PLE:** the 50.7 GiB n-gram table stays in the file. Each step hashes
  its tokens' histories (the two tokens before each) and records a
  `write_file` of each of their 16 Q8_0 rows, which the host serves from
  the file: the mapping natively, the picked `File` in the browser. GPU
  writes are whole words, so each 170-byte row is read with the 2 bytes
  after it, 172 bytes apart, and `embed_q8_0` decodes them at that stride.
  The history belongs to the program: a reset clears it, and checkpoints
  save it with the recurrent state.
- **State:** every DeltaNet layer's recurrent state and conv history, and
  the PLE convolution's history, in one buffer that checkpoints copy.
- **A layer:** the four streams' norms and the low-rank mix into the
  block's input (`hc_act`, `hc_mix`), the mixer (DeltaNet with a sigmoid
  gate, or attention with `attn_split_wide`), `hc_combine` back into the
  streams; again for the experts: `moe_logits`, `moe_route`, `rotate128`, the gate and up
  experts, `rotate128` of their SiLU pair, the down experts, the shared
  expert, `moe_combine`, `hc_combine`.

The earlier WebGPU measurement on the RTX 5090 Laptop recorded a warm
`zllm generate` load of 16 to 24 s (weights 11 to 15 s, tuning 5 to 7 s),
prefill at 73 to 77 tok/s and decode at 35 to 37 tok/s, one step per batch.
These are historical results, before subsequent Q8_0 and routed-expert
optimizations, not current backend comparisons. The model occupied about
21 of the card's 24 GB. With another 20 GB process on the GPU, Windows
moved buffers into shared system memory and decode fell to 6 tok/s.

See [Gyro-Coder](gyro.md) for the model's data path and
[Velocity](velocity.md#kernels) for later native-kernel changes.

Each hyper-connection mixer's down projection carries the stream weights'
(inject) rows: 324 rows in one multiply instead of 320, then 4 rows of
10,240 in a single workgroup (38 µs, 96 times a step, 3.7 ms).
