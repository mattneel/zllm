# GPU backend

The GPU path has three layers. The core records a **command stream**, a
**host** executes it on WebGPU, and **kernels** are WGSL compute shaders the
stream launches. M1 provides all three, with Dawn as the host, and runs the
full Qwen 3.5 forward pass on the GPU at parity with the CPU reference. M2
turns the forward pass into a **step program**: prebuilt launches that
process any number of tokens against a KV cache and the DeltaNet state, and
sample the next token on the GPU ([Generation](generation.md)).

## Hosts

| Host | WebGPU | Adapter here | Used for |
|---|---|---|---|
| Native Linux | Dawn → Vulkan | llvmpipe (Mesa's software renderer) | CI, kernel correctness |
| Windows, run from WSL | Dawn → D3D12 | RTX 5090 Laptop | Local parity and timing |
| Browser | Chrome's Dawn → D3D12/Metal/Vulkan | RTX 5090 Laptop (D3D12) | The product ([Browser](browser.md)) |

The spec targets D3D12, Metal and Vulkan. D3D12 runs on the 5090 and
Vulkan on llvmpipe in CI; Metal and hardware Vulkan haven't been run, for
want of the hardware. The kernels use only core WebGPU plus `shader-f16`
where available and the packed 4×8 integer builtins, which all three
backends offer.

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

The core never calls WebGPU. It appends commands to a `Recorder`
(`src/gpu/cmd.zig`), a host executes the batch, and the core consumes one
completion per `submit` and per `read_buffer`. The browser host
(`web/src/zllm/gpu.ts`) interprets the same words, crossing the wasm/JS
boundary once per batch rather than once per dispatch.

- **Encoding:** little-endian `u32` words. Each command starts with
  `opcode | (words << 8)`, so a host can bounds-check every command.
- **Object ids:** objects are named by ids the core assigns, densely per kind.
- **Addresses:** buffer contents, WGSL source and readback destinations are
  core-memory addresses, two words each; in wasm32 the high word is 0.
- **Batching:** commands between two `submit`s go into one fresh command
  buffer, and consecutive dispatches share one compute pass.

| Op | Arguments |
|---|---|
| `create_buffer` | id, size, usage (WebGPU `GPUBufferUsage` bits) |
| `write_buffer` | buffer, offset, data, length |
| `create_shader` | id, WGSL source (entry point `main`) |
| `create_pipeline` | id, shader |
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

Kernels live in `src/kernels/*.wgsl`, each with a uniform `Params` struct
mirrored in `kernels.zig`. Weights stay in their GGUF format on the GPU,
BF16, F16 or quantized, and are decoded in the shader; activations and
accumulation are f32, except that the quantized prefill multiplies round
their input to 8 bits as llama.cpp does (below). Two changes to the
format, made at load: Q6_K blocks (210 bytes) are padded to 224, so every
K-quant block starts on a 16-byte boundary and is read in 16-byte vectors;
and Q3_K blocks (110 bytes) are repacked into eight 16-byte units of 32
weights, each holding its 2-bit values, high bits, two scales and d, so a
unit decodes from one load (128 bytes per block).

| Kernel | Work | Notes |
|---|---|---|
| `embed_bf16`, `embed_f16` | Token embedding gather | Per table chunk |
| `rmsnorm` | Row RMSNorm | One workgroup per row |
| `matmul_bf16`, `matmul_f16` | x · Wᵀ, prefill | 16 tokens × 64 outputs per workgroup; tiles staged in workgroup memory. Optionally adds into the output (the residual) |
| `matvec_bf16`, `matvec_f16` | x · Wᵀ, decode | The half-precision kernels are `matvec_half.wgsl`, `matmul_half.wgsl` and `embed_half.wgsl`, composed with `half/bf16.wgsl` or `half/f16.wgsl`, which widen a word's two values. 8 rows per workgroup, each row's dot product over 32 lanes reading 16-byte weight vectors, then a fixed tree. Optionally RMSNorms the staged input first, adds into the output, or computes SiLU(gate) · up over stacked gate and up rows |
| `silu_mul` | SwiGLU over the stacked gate and up projection, prefill | Elementwise |
| `rope_append` | Q/K RMSNorm, partial NeoX RoPE at positions pos0 + t, and the K/V cache write | One workgroup per (head, token) of the stacked Q, K, V projection. Reads a host-built (cos, sin) table: WGSL trig is only accurate near [−π, π]. The cache holds f16 pairs (`pack2x16float`) or f32 |
| `attn_split`, `attn_combine` | Causal GQA over the cache, with the sigmoid output gate (flash-decoding) | One workgroup per (KV head, 64 cache positions, token) scores the block for the whole GQA group and takes a local softmax (groups of up to 8 query heads; `attn_split_wide` takes up to 12, whose queries fill WebGPU's 16 KB of workgroup memory, which costs the 27B's prefill 2% if used for its groups of 6); the combine merges the blocks. Prefill steps run them in chunks of 32 tokens, so the partials hold one chunk; a token's result depends only on its position. The 27B's prefill attention at position 5,000 dropped from 96 to 61 ms a step when this replaced a kernel with one workgroup per (head, query) |
| `qsa_keys`, `qsa_pool`, `qsa_score`, `qsa_select`, `attn_split_sparse` | Qwen3.8-Flash-Next's sparse attention: the indexer's queries (norm, RoPE) and keys (an f16 cache), each block of `ratio` keys pooled (mean, norm, RoPE at its start), every block's score Σ_heads relu(q · block), the top blocks, and attention over the selected cells | Exact selection: the scores are never negative, so their bits order as their values; one workgroup per row finds the K-th largest bit by bit and breaks ties by index with a scan. `attn_split_sparse` is `attn_split_wide` with an inclusion test composed in (`attn_sparse.wgsl`; `attn_dense.wgsl` for the others). An excluded key weighs 0, even where a whole 64-key block is excluded |
| `delta_prep` | DeltaNet front end: causal conv and SiLU, L2 norm of q and k, β and decay gates | One workgroup per (head of channels, token); inputs before the step come from the conv history |
| `delta_rule` | Gated delta rule | Each thread keeps 16 rows of one state column in registers for the whole step; a head spans 4 workgroups. Also advances the conv history, and can keep the state and history as after one row of the step (for rolling back a rejected speculative token) |
| `gated_norm` | The SiLU(z)-gated RMSNorm after it; sigmoid(z)-gated for Qwen4Exp | One workgroup per (head, token); z read from the stacked projection |
| `matvec_<q>`, `matmul_<q>`, `embed_<q>` | The multiplies and embedding gather on quantized weights: `q8_0`, `q3_k`, `q4_k`, `q5_k`, `q6_k` | `matvec_quant.wgsl.hbs`, `matmul_quant.wgsl` and `embed_quant.wgsl`, composed at compile time with `quant/<q>.wgsl`. The decode multiply is a template (`src/tmpl`): its threads, threads per row and rows per lane group are parameters (`MatvecVariant`), 12 candidates, which gpu-check runs all of; the defaults render the kernel as it was. That file decodes a row's 32-weight units as ggml's `dequantize_row_*` does, value for value, 16 weights from 16-byte loads at a time. The decode multiply keeps `matvec_bf16`'s fusions, stages x in 2,048-float tiles, and splits each block among 4–8 lanes (`quant/<q>_dot.wgsl`), consecutive lanes on consecutive vectors. A step of several tokens takes them in pairs: both tokens' x in 1,024-float tiles, each decoded weight vector dotted with both (`block_dot2`), so two tokens cost much less than two steps; each token's sums are bitwise those of a step of one. The prefill multiply takes x rounded to 8-bit blocks and sums each weight sub-block's products exactly in integers (`dot4I8Packed`), then scales them, as llama.cpp's `vec_dot` does: 32 tokens × 128 outputs per workgroup, 4 × 8 per thread, the next unit fetched and decoded while the current one computes |
| `quantize_q8` | Rounds a quantized prefill multiply's input to 8-bit blocks, computing it first where it is an RMSNorm or the SiLU pair, so those kernels don't run | As llama.cpp's CPU backend does for the weight type: Q8_K blocks of 256 for the K-quants, Q8_0 blocks of 32 for Q8_0. Bitwise as the reference's `--llama-activations` mode |
| `matmul_reduce` | Adds up a quantized prefill multiply split over k | A matrix with too few column tiles to fill the GPU (a 5,120-row projection has 40) splits k into slices, aiming at 256 workgroups; each slice writes partial sums, and this adds them into the output |
| `repad`, `repack_q3_k` | Lay out Q6_K (padded) and Q3_K (repacked) blocks at load | Each chunk's rows go to a staging buffer, freed after the chunk's own submit |
| `mark_tokens` | The prompt's token set, for the repetition penalty | |
| `sample` | Bias, penalties, temperature, min-p, top-k, top-p, draw | In phases, one dispatch each; see [Generation](generation.md#sampling). One more phase undoes the last draw's output count (a rejected speculative round's second sample) |
| `draft` | The speculative draft: the first index of the maximum of a row of logits, into the next step's token input | One workgroup |
| `gyro_matvec` | Routed-expert products on Gyro TQ2_T weights (Gyro-Coder) | Lifted from gyro-dx: the expert comes from a GPU buffer the router wrote, no readback; a lane per trellis step decodes four weights from the 2,048-pair codebook; gate and up in one dispatch |
| `moe_logits`, `moe_route` | Expert routing: router · x (f32) and the shared expert's gate logit, then a token per workgroup: softmax, top-k (ties to the lower index), normalized weights | Router rows read coalesced, a lane group per row, 8 rows per workgroup. Computed in `moe_route`'s one workgroup per token, the product took 1.08 ms a layer, 52 of the 87 ms of a decode step |
| `rotate128` | H₁₂₈ · (signs · x) per 128-value block: the basis Gyro expert weights are stored in. Optionally of silu(gate) · up | 64 butterflies a stage in workgroup memory |
| `moe_combine` | The expert block's output: the routed experts' weighted sum plus the gated shared expert | |
| `hc_act`, `hc_mix`, `hc_combine` | The hyper-connection mixer around Q8_0 down, up and inject products: silu(lo / hc); the streams' gated mean into a block's input; a block's output added to each stream, weighted 2·sigmoid(inject / hc) | The streams' norms are `rmsnorm` with a weight row per stream |
| `ple_mix`, `ple_conv` | The PLE block: per stream, the signed-root sigmoid gate of normed key · query, the gated value and its norm; then the causal dilated convolution (4 taps, dilation 3) into the residual streams | The convolution runs a step's tokens in order against a 9-row history, which it advances as `delta_rule` does the conv history |

These are plain WGSL. M4 makes them templates tuned per GPU; the decode
kernels above are its first part, measured with `zllm bench`
([Generation](generation.md#performance)).

## Tuning

After the weights are on the GPU and before the step program is built,
`src/tune` sweeps the quantized decode multiply's variants (spec §5) for
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

The winners go into the step program's `Tuning`. On the RTX 5090 Laptop, all
16 of the 27B's shapes take another variant: the K-quants 8 or 16 threads
per row instead of 32, Q8_0 two rows per lane group. Decoding speculatively
goes from 40.7–41.5 tok/s to 47.0–47.3, one token per step from 30.0–30.4 to
31.3–31.9; tuned, the 27B's decode logits stay as close to the CPU reference
(6.4e-6 relative L2 at most, against 6.2e-6). The sweep takes 13 s;
`--no-tune` skips it.

### Device lost

A lost device (a driver reset, a GPU hang, a removed adapter) fails the
request in flight with `error.DeviceLost` (spec §8). The native runtime then
rebuilds itself before returning: a new Dawn device, the weights read again
from the file mapping, the tuning from the plan cache, a fresh engine with
no resident sequence (`Runtime.recover`). `zllm chat` reports it and takes
the message again. On the RTX 5090 Laptop the 0.8B rebuilds in 2.2 s.
`engine-check` forces a loss through Dawn's `wgpuDeviceForceLoss` and checks
the typed error, a rebuild from the plan cache, and the same prompt's
tokens and logits, bitwise, after it. The browser does the same
([Browser](browser.md#device-lost)).

### Plan cache

The CLI keeps what a sweep found in the plan cache (spec §7,
`src/plan/cache.zig`): one JSON file of the variants that differ from the
default, named by its key, in `%LOCALAPPDATA%\zllm\plans` on Windows and
`$XDG_CACHE_HOME/zllm/plans` (or `~/.cache/zllm/plans`) elsewhere;
`ZLLM_PLAN_CACHE` names another directory, or none when empty. The key is a
SHA-256 over everything that could change the result: the adapter (vendor,
architecture, device, description, which carries the driver version on most
backends, and backend), the runtime, the device limits the plan uses, the
model's GGUF header and tensor index, the zllm version, and every kernel's
source. A load that finds its key's file uses it instead of sweeping; a file
under another key, or one naming an unknown variant or type, is ignored, and
the sweep's result replaces it. On the Gyro-Coder the cached load skips
2.5 s of tuning and decodes as fast, with the same output.

The browser tunes at load too, through the same sweep, between the
weights and the step program. The shim gets the key from the core
(`zllm_plan_key`, with the adapter's strings from `GPUDevice.adapterInfo`
and the browser's user agent as the runtime) and keeps the file in the
origin private file system, under `zllm/plans/`. In Chrome the Gyro-Coder's
sweep adds 8 s to its first load and none to later ones.

## Step program

`src/plan/qwen35.zig` builds everything a step needs once, as the spec's
execution plan asks:

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
`matvec_bf16` for every multiply, fused with what surrounds it, and split
attention. Longer steps run the **prefill kernels**. The plan prebuilds the
launches that differ both ways (`plan.Fork`), and a step picks one
(`plan.Path`).

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

The plan applies the spec's fusion rules by hand. Weights that share an
input are stacked by rows when uploaded, so each set runs as one multiply,
and later kernels read their part of the output at a row stride:

- Q (with its output gates), K and V
- the DeltaNet projections: qkv, z, β and α
- the FFN's gate and up

| Rule | Decode kernels | Prefill kernels |
|---|---|---|
| RMSNorm into the next multiply | In `matvec_bf16`'s prologue: each workgroup reduces the row's sum of squares itself | A separate `rmsnorm`, which every tile reads |
| Gate, up and SiLU-multiply | One `matvec_bf16` over row pairs | One multiply, then `silu_mul` |
| Residual add into the output and down projections | The multiply adds into the residual stream | The same |
| RoPE and the cache write after QKV | `rope_append` | The same |
| DeltaNet conv, gates and L2 norm before the scan | `delta_prep` | The same |
| Final norm and the vocabulary projection | Norm in the projection's prologue | A separate `rmsnorm` |

A fusion that needs workgroups to synchronize is not legal, which rules out
three more:

- **RoPE inside the QKV multiply:** a head's norm spans the output rows of
  32 workgroups.
- **The gated norm inside the scan:** a head's state spans 4 workgroups.
- **Top-k inside the vocabulary projection:** the sampler's selection runs
  over the whole vocabulary.

A decode layer is 7 dispatches. A decode step is 182, down from 441.

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

On the RTX 5090 Laptop, `zllm generate` loads it warm in 16 to 24 s
(weights 11 to 15 s, tuning 5 to 7 s), prefills at 73 to 77 tok/s and
decodes at 35 to 37 tok/s, one step per batch. The model takes about 21 of
the card's 24 GB: with another 20 GB process on the GPU (a second copy of
the model in a browser), Windows moved buffers into shared system memory
and decoding fell to 6 tok/s.

Each hyper-connection mixer's down projection carries the stream weights'
(inject) rows: 324 rows in one multiply instead of 320, then 4 rows of
10,240 in a single workgroup (38 µs, 96 times a step, 3.7 ms).
