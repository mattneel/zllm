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
| `moe_route` | Expert routing, a token per workgroup: router · x (f32), softmax, top-k (ties to the lower index), normalized weights; and the shared expert's gate logit | Router rows read coalesced, a lane group per row |
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
buffers, the multiplies' launches). It runs the decode kernels only, in
steps of up to 4 tokens, and caps the context at 2,048. Up to 2,051 visible
cells the sparse attention's indexer selects every cell, so dense attention
computes the same thing.

- **Weights:** the dense Q8_0 matrices as Qwen 3.5 uploads them, stacked
  where they share an input (the DeltaNet projections; Q, K and V; the
  shared expert's gate and up; the hyper-connection matrices). The routed
  experts go up as stored, TQ2_T, read by `gyro_matvec` at the ids the
  router wrote; then the router, the codebook and the Hadamard sign
  vectors. Everything but the PLE table is about 21 GB of GPU memory.
- **PLE:** the 50.7 GiB n-gram table stays in the file. For each token the
  host hashes its history, reads its 16 Q8_0 rows (16 × 170 bytes) and
  sends them with the step; `embed_q8_0` decodes them as a 16-row table.
- **A layer:** the four streams' norms and the low-rank mix into the
  block's input (`hc_act`, `hc_mix`), the mixer (DeltaNet with a sigmoid
  gate, or attention with `attn_split_wide`), `hc_combine` back into the
  streams; again for the experts: `moe_route`, `rotate128`, the gate and up
  experts, `rotate128` of their SiLU pair, the down experts, the shared
  expert, `moe_combine`, `hc_combine`.

Through `zllm gpu-forward` on the RTX 5090 Laptop, a warm load takes 14 to
20 s, and a token through all 48 layers about 90 ms. Each step there is its
own synchronous submission, with 96 readbacks for the checks
([Validation](validation.md#qwen38-flash-next-gyro-coder)).
