# Generation

M2 turns the GPU forward pass into text generation: an engine that schedules
steps under a time budget, keeps the model's state between requests, samples
on the GPU, and streams text. `zllm chat` and `zllm generate` drive it on the
Dawn host.

```sh
zllm chat model.gguf                        # thinking on; --no-think to skip it
zllm chat model.gguf --system "Be brief." --temperature 0.7
zllm generate model.gguf "Once upon a time" --max-tokens 200 --seed 1
scripts/d3d12.sh chat 'C:\path\to\model.gguf'   # on the Windows GPU, from WSL
```

## Engine

`src/sched/sched.zig` is a state machine that never blocks. Its host calls
`start` with a request, then repeatedly `next`, which returns one of four
events:

- **`submit`:** execute this batch, then call `complete` with its GPU time.
- **`token`:** a sampled token and the text it released.
- **`wait`:** complete the oldest outstanding batch first.
- **`done`:** the finish reason (`stop`, `length` or `abort`) and usage.

The native host executes and completes each batch before calling `next`
again, so it never sees `wait`. The browser host executes asynchronously
and keeps two batches in flight: decoding records the next step before the
previous step's token is read back, since the sampler hands each token to
the next step on the GPU. The resident sequence holds a placeholder until
the token arrives. When a step ends the request, the step after it has
already processed the final token, which stays resident; that step's own
sample is discarded. This hides Chrome's readback latency
([Browser](browser.md#two-steps-in-flight)).

`cancel` ends a request after the batch in flight: `done` reports `abort`,
and the tokens processed so far stay resident, so the next request reuses
them as after any other finish.

```mermaid
stateDiagram-v2
  [*] --> Prefill: start (restore or reset state, sampler setup)
  Prefill --> Prefill: chunk under the budget
  Prefill --> Emit: last prompt token, sampled
  Decode --> Emit: one token, sampled
  Emit --> Decode: no stop
  Emit --> Done: stop token, stop string, max tokens, context full, cancel
  Done --> [*]
```

## Scheduling

Every submission is one step, and every step aims for the spec's 150 ms
budget:

- **Prefill:** a chunk is the largest token count whose predicted cost fits
  90% of the budget, at most 512. The prediction is linear,
  fixed + per_token · m:
  - **Fixed part:** measured on one-token steps. Before the first request,
    the host runs two calibration steps on a scratch state; the first only
    warms up the pipelines.
  - **Per-token part:** taken from the latest prefill step, because it grows
    with position (attention). That growth is also why the target is 90%.
  - **Ramp:** a chunk grows at most 4× past the step its rate came from,
    since the rate also rises with the chunk size. Before any prefill step
    has been measured, the first chunk is as many tokens as one-token steps
    fit in 90% of the budget, at most 64. A prefill token never costs more
    than a whole decode step, so this holds even for the 27B, whose
    one-token step is far slower.
- **Slow GPUs:** where a one-token step alone takes more than half the
  budget, smaller chunks would only add steps that each pay that cost. The
  target becomes twice the fixed cost instead. On llvmpipe, where one step
  costs about a second, the plain rule cut chunks to one or two tokens.
- **Tiles:** the prefill multiplies compute whole tiles of tokens (16 for
  BF16 weights, 32 for quantized ones), and a partial tile costs as much as
  a whole one. So the per-token rate counts a step's tokens in whole tiles,
  and chunks are whole tiles. Where not even one tile fits the target, the
  chunk is one tile as long as its prediction stays under 90% of the
  spec's 250 ms hard cap. The 27B's 32-token steps take about 116 ms; with
  8-token chunks, each step cost as much as a whole tile and prefill ran
  at 25 tok/s.
- **Checkpoint boundaries:** chunks also end at multiples of 512 and just
  before the prompt's last token. That token then runs as its own step, which
  measures the fixed cost and produces the logits to sample.
- **Alignment:** otherwise chunks end only at multiples of 8. Steps of at
  most 4 tokens run the decode kernels, which round differently from the
  prefill kernels ([GPU backend](gpu.md#step-program)). Every aligned step
  is longer than that, and the prefill kernels give the same result however
  a prompt is split, so the kernels each prompt position goes through depend
  on the prompt alone, not on step timings. The alignment was 64 until the
  27B: its 64-token steps alone took seconds, past Windows' 2 s GPU
  watchdog.
- **Decode:** one token per step. The sampler writes the token straight into
  the next step's input on the GPU, so the CPU only reads back the token id
  and its stop flag.

## State and prefix reuse

The resident sequence is every token the GPU has processed. The KV cache
holds their rows, and the DeltaNet state is the state after the last one.
`src/state/state.zig` decides how much of a new prompt to reuse:

| Case | Action | Prefill |
|---|---|---|
| The resident sequence is a prefix of the prompt | Continue from the current state | The rest of the prompt |
| They share a prefix | Restore the latest checkpoint inside it | From the checkpoint |
| Nothing usable, or no resident sequence | Zero the DeltaNet state | The whole prompt |

- **Rewinding:** attention rows rewind for free, since rows past the
  sequence are never read. The DeltaNet state cannot rewind, so the engine
  copies it into one of 4 checkpoint slots every 512 tokens and just before
  each prompt's last token, evicting the oldest slot when all are full.
- **The last-token checkpoint:** it serves three cases with one replayed
  token: regenerating the same prompt, editing what follows it, and
  continuing a conversation whose reply was re-rendered differently.

Reuse is exact where it can be. A reused prefix gives logits bitwise
identical to processing the whole prompt from scratch whenever both runs put
each position through the same kernels. That covers checkpoint restores, a
repeated prompt and a reset. One case differs: a prompt that extends a turn
reuses the reply's tokens, which went through the decode kernels, while a
fresh run would prefill them. There the logits agree within 1e-4 relative
L2, and 3e-5 is measured ([Validation](validation.md#engine)).

KV is one contiguous f16 cache per attention layer, sized for the context
(8192 tokens by default, 12 KiB per token for the 0.8B model). The spec's
256-token pages matter once several sequences share a pool, which a
single-sequence engine doesn't need.

## Sampling

The sampler (`src/kernels/sample.wgsl`) follows vLLM's v1 sampler
(`vllm-project/vllm@0c81a03`, `vllm/v1/sample/sampler.py`) step for step:

1. Logit bias.
2. Penalties:
   - Repetition: over prompt and output tokens, a logit l becomes l / r if
     positive, else l · r.
   - Frequency and presence: over output tokens only, l −= frequency · count
     + presence · [count > 0].
3. Greedy argmax (first index) if the temperature is below 1e-5.
4. Temperature.
5. Min-p: drop p < min_p · p_max.
6. Top-k: keep values ≥ the k-th largest.
7. Top-p: keep a token while the probability ranked above it is < p.
8. The draw.

Each filter keeps a top segment of the values, so the kept set is always one
threshold on order-preserving keys, combined with the min-p test. The kernel
runs in phases, one dispatch each, chosen per request:

| Phase | Workgroups | Work |
|---|---|---|
| prep | one per 1,024 tokens | Bias (binary search in a token-sorted list), penalties, temperature; each workgroup's maximum |
| reduce | 1 | The maximum and its first index; greedy finishes here |
| top-k | 4 × (one per 1,024 tokens, then 1) | Radix select: 8-bit histograms of the keys, then the digit holding the k-th largest |
| top-p, after a top-k of at most 1,024 | one per 1,024 tokens, then 1 | Gather the candidates, sort them in workgroup memory, cut at p, draw in token order |
| draw | one per 1,024 tokens, then 1 | Each workgroup's kept mass in token order, then the inverse CDF |
| top-p otherwise | 32 × (one per 1,024 tokens, then 1) | Bisection on the key threshold |

With the chat defaults (top-k 20, top-p 0.95) a draw is 12 dispatches and
0.2 ms on the RTX 5090. Stop tokens are checked on the GPU. Two differences
from vLLM, both deliberate:

- **Ties:** tokens tied at a top-k or top-p threshold are all kept. vLLM
  splits them by sort order.
- **Draw:** an inverse CDF in token-id order at one uniform u per step, drawn
  on the CPU from the request's seed. vLLM uses an exponential race, so
  seeded streams differ.

`src/sampler/sampler.zig` holds the request parameters, with vLLM's range
checks, and a CPU reference sampler that `gpu-check` compares the kernel
against.

## Chat format

`zllm chat` formats conversations with the fixed Qwen chat template,
[froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
@ `855bffc4`, v22.5. `src/chat/chat.zig` transcribes it for text
conversations without tools, including:

- **Thinking control:** the `enable_thinking` and `reasoning_effort`
  options, and `<|think_off|>`-style tags in system and user messages.
- **Reasoning:** split out of earlier replies or passed separately.
- **System prompts:** leading system messages merged into one.
- **Whitespace:** trimmed like Python's `str.strip`.

`check-chat` compares the transcription with the Jinja template, rendered as
Hugging Face `transformers` renders it.

The template keeps earlier turns' reasoning (`preserve_thinking`), which
matters for reuse. Each new prompt re-renders the previous reply exactly as
the model generated it, so the whole conversation stays resident, and a turn
prefills only the new message. The stock Qwen template drops old reasoning,
which would force recomputing every earlier reply.

Text streams through `src/tokenizer/stream.zig`. It holds back bytes until
they form complete UTF-8 and cannot be the start of a stop string. The stop
string itself is not output, as in vLLM by default.

`chat` defaults to the Qwen 3.5 model card's text-task sampling:

| Mode | Temperature | top_p | top_k | Presence penalty |
|---|---|---|---|---|
| Thinking | 1.0 | 0.95 | 20 | 1.5 |
| `--no-think` | 1.0 | 1.0 | 20 | 2.0 |

`generate` samples plainly unless told otherwise, as vLLM does for a model
without a `generation_config.json`.

## Performance

Measured with `zllm generate` on an RTX 5090 Laptop (D3D12), f16 KV:

| Phase | M2 | Now |
|---|---|---|
| Prefill, a 2,457-token prompt | 1,562–1,567 tok/s | 2,882–2,896 tok/s, in 12 steps |
| Decode, greedy, short context | 34 tok/s | 285–295 tok/s |
| Decode, greedy, after the 2,457-token prompt | 27 tok/s | 270 tok/s |
| Decode, top-k 20 + top-p 0.95 | 25 tok/s | 268 tok/s |

`zllm bench` attributes a decode step to kernel classes by ablation: the
step replayed without each class. It drove the decode kernels and their
fusion, step by step, at position 1,056:

| Change | ms per decode step |
|---|---|
| M2 kernels | 42.6 |
| Matrix-vector multiplies for decode (23.7 → 2.0 ms) | 18.9 |
| Sampler in phases across the GPU (10.6 → 0.2 ms) | 8.1 |
| Decode attention split over cache blocks (3.0 → 0.45 ms) | 5.7 |
| DeltaNet state in registers (1.2 → 0.35 ms) | 5.0 |
| Matrix-vector activations in workgroup memory | 4.8–4.9 |
| Decode attention reads V once per GQA group (0.66 → 0.34–0.40 ms) | 4.7 |
| Fusion: 441 → 182 dispatches per step ([GPU backend](gpu.md#fusion)) | 3.9 |

What remains, of about 3.9 ms per step:

- **Weights:** a decode step streams 1.5 GB of BF16 weights, about 1.7 ms
  at this GPU's bandwidth. The vocabulary projection runs at that floor
  (0.5 GB in 0.54 ms). The body's projections reach about 60% of it (1.0 GB
  in 1.7 ms), so they are the next lever.
- **Dispatches:** each dependent dispatch costs 2.4–3.8 µs on this GPU
  through D3D12, depending on its work, so the 182 cost 0.4–0.7 ms.
- **Attention and the sampler:** about 0.3 and 0.2 ms.

Synchronizing a submission costs 0.1–0.2 ms.

### The 27B

Qwen 3.8 27B at Q3_K_M (13.75 GB) on the same GPU, `zllm generate`, greedy:

| Phase | First run | Now | Spec bar |
|---|---|---|---|
| Prefill, a 901-token prompt | 25 tok/s, in 114 steps | 256–257 tok/s, in 31 steps | 300 tok/s |
| Decode | 6 tok/s | 26–27 tok/s | 20 tok/s |

The quantized kernels made the difference ([GPU backend](gpu.md#kernels)).
`zllm bench` at position 288 measured the steps:

| Change | ms per decode step |
|---|---|
| Byte-wise decode multiply | 175 |
| x staged in workgroup tiles | 50 |
| Blocks split among lanes, 8 rows per workgroup | 46 |
| Q3_K and Q6_K padded; 16-byte loads; four weights decoded at once | 38 |

A decode step reads about 12.2 GB of weights, 13.7 ms at this GPU's
bandwidth. The Q3_K multiplies, 6.6 GB, take about 19 ms: 11 ms of loads,
4 ms of decoding and 3.5 ms reading x from workgroup memory.

Prefill steps of the 27B are 32 tokens, about 116 ms each:

| Prefill multiply | tok/s, end to end |
|---|---|
| f32, register tiles, split over k; tile-sized chunks | 154 |
| Input rounded to 8 bits, integer dot products (`dot4I8Packed`) | 227 |
| The next unit fetched while the current one computes | 247–250 |
| Rounding fused with the RMSNorm and SiLU that produce its input | 256–257 |

In f32 the multiply-adds alone took about 66 ms of a step. Rounding the
input to 8-bit blocks, as llama.cpp's CPU backend does, makes each weight
sub-block's dot product exact in integers, at four products per
instruction. The GPU is then checked against the reference that rounds
the same way ([Validation](validation.md#the-27b)). Decode keeps f32
inputs: it is bound by reading weights, not by arithmetic.

The multiply now takes about 98 ms of a step. It issues about three
instructions per four products: the integer dot product, a separate add
(WGSL's `dot4I8Packed` has no accumulator operand) and the sub-block
scaling. That is near this GPU's issue rate. Rounding the input happens in the
same kernel that computes it (RMSNorm, SiLU), and the rest of the step
takes about 36 ms.
