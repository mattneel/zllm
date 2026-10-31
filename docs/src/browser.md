# Browser

The browser runs the same Zig inference core as the native CLI, compiled
for `wasm32-freestanding`. `zllm.js` (`web/src/zllm`) executes its command
batches on WebGPU, reads weights from a GGUF source, and exposes streaming
generation and an OpenAI-style chat facade. Inference runs locally; the
browser is one host for the portable engine, not a separate model runtime.

**[Try the live demo](https://mattneel.github.io/zllm/demo/)** in a browser
with WebGPU, such as Chrome or Edge. Pick a pinned model from the list,
paste a GGUF URL, or open a local file. The default is Qwen3.5-0.8B BF16
from Hugging Face. URL loads can use the browser's private file system for
later visits, subject to validators and storage quota.

The `book` job in `.github/workflows/ci.yml` builds the book, `zllm.wasm`
and the demo bundle from the same checkout, then uploads `docs/book` as one
Pages artifact. Deployment runs after the CI jobs pass on a push to
`master`. This keeps a published book and its demo on the same commit; it
does not mean the demo has a GPU browser test in CI.

```sh
zig build wasm            # zig-out/web/zllm.wasm
cd web
bun install --frozen-lockfile
bun run dev               # demo on http://localhost:8080/, models from ../.fixtures/models
bun run build             # dist/zllm.js and dist/zllm.wasm
```

Use the published HTTPS demo or the localhost development server. Loading
requires an available WebGPU adapter and enough device memory for the
selected model's resident weights, state and context.
Models served from another origin must permit the browser's range requests
through CORS and expose `Content-Range`. URL caching also needs an exposed
`ETag` or `Last-Modified`; a URL without one still loads, but is not cached.
`urlSource` requires HTTP 206 responses and checks each returned range's
length. A local file avoids these network requirements.

## API

```ts
import * as zllm from "./zllm.js";

const messages: zllm.ChatMessage[] = [
  { role: "user", content: "Explain grouped-query attention." },
];

const engine = await zllm.load({
  model: "/models/Qwen3.5-0.8B-BF16.gguf", // URL (range requests), File, Blob or FileSystemFileHandle
  onProgress: ({ loaded, total }) => console.log(loaded / total),
  // cache: false, to skip the weight cache
});

for await (const chunk of engine.generate({ prompt: "Once upon a time", maxTokens: 200 })) {
  console.log(chunk.text); // the last chunk carries finish and usage
}

for await (const chunk of engine.chat.completions.create({ messages, stream: true })) {
  const delta = chunk.choices[0].delta; // reasoning_content, then content
}
const reply = await engine.chat.completions.create({ messages, enable_thinking: false });
// A model with its own template (Llama, Granite, ...) also takes vLLM's
// chat_template_kwargs:
await engine.chat.completions.create({ messages, chat_template_kwargs: { date_string: "1 Nov 2026" } });

// Tool calling, in the OpenAI format.
const tools: zllm.Tool[] = [{
  type: "function",
  function: {
    name: "get_weather",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  },
}];
const turn = await engine.chat.completions.create({ messages, tools });
// The app executes calls; zllm only generates and parses them.
const call = turn.choices[0].message.tool_calls?.[0];
if (call) {
  messages.push(turn.choices[0].message);
  messages.push({ role: "tool", tool_call_id: call.id, content: "18°C, clear" });
}

engine.tokenize("text"); engine.detokenize(ids);
engine.interruptGenerate(); // or pass `signal`, or stop reading the stream
engine.stats(); await engine.unload();
await zllm.cacheUsage(); await zllm.clearCache();
```

The facade's public types are exported from `web/src/zllm/index.ts`.
It is an in-process API, not an HTTP server or the complete OpenAI API.

| Area | Options and results |
|---|---|
| `load` | `model`: URL, `File`/`Blob`, `FileSystemFileHandle`, or a custom `ModelSource` with `size` and async `read(offset, length)` |
| Device and memory | Optional `device`, `context` (default 8192), and `kv: "f16" \| "f32" \| "fp8"` (default f16) |
| Loading | Optional `wasm` URL, `cache`, `tune`, and `onProgress` |
| `generate` | `prompt` text or `tokens` ids; `maxTokens`, `sampling`, `stop`, `logprobs`, `speculate`, `signal` |
| `sampling` | `temperature`, `topK`, `topP`, `minP`, repetition/presence/frequency penalties, `logitBias`, `seed` |
| Chat | `messages`, `stream`, `max_tokens` or `max_completion_tokens`, snake-case sampling controls, thinking/template options, tools, logprobs, `signal` |
| Completion | Final chunk includes finish reason and usage; raw usage includes prompt, cached and completion tokens, prefill steps, prefill and decode milliseconds |
| `stats()` | Adapter, model metadata and sampling defaults, load time, weight-cache and tuning source, last usage, recovery count |

Raw generation returns token ids and text on each chunk; text may be empty
while UTF-8 or a possible stop string is pending. The final chunk has no
tokens and carries `finish` and `usage`. Both generation and chat default
to at most 4096 output tokens, subject to context capacity. Chat uses
`max_completion_tokens` when both output-limit fields are present.

- **Requests** run one at a time; a second waits for the first. What a
  request leaves unset takes the model's defaults, as `zllm chat` and
  `zllm generate` do ([Generation](generation.md#sampling)); `stats().model.sampling`
  lists them.
- **Chat** renders the fixed Qwen template inside the core, the template
  `check-chat` pins, or the model's own ([Dense transformers](dense.md#chat)).
  Each reply is split into `reasoning_content` and `content`, as vLLM's
  Qwen3 reasoning parser does: from inside the `<think>` block where the
  prompt opens one (the core reports it), from a `<think>` the reply opens
  itself (Qwen 3), else all content (Llama). An assistant turn sent
  back with its `reasoning_content` re-renders exactly as generated, so the
  whole conversation stays resident and a new turn prefills only its new
  message.
- **Tools** are offered as the template offers them, in the system turn
  with its XML call format (or as the model's own template offers them).
  An assistant turn's `tool_calls` (arguments as JSON text, decoded first,
  as vLLM does) and `tool` messages render as the template renders them.
  A reply's calls become `tool_calls`, with `finish_reason`
  `"tool_calls"`, parsed in the format the model writes them
  (`stats().model.tool_format`, from its template), as vLLM's parser of
  that name does:
  - `xml` (Qwen 3.5–3.8, Granite 4): `<tool_call><function=…>` blocks,
    parameter values converted by the tool's schema type;
  - `hermes` (Qwen 3): `<tool_call>` blocks of JSON `{"name", "arguments"}`;
  - `llama3_json` (Llama 3): a reply that is JSON `{"name", "parameters"}`
    objects, optionally after `<|python_tag|>`, separated by `;`.

  A call that doesn't parse leaves the reply as text. Streamed text stops at
  the first call, and the calls arrive in one `tool_calls` delta at the end.
  A normally stopped reply with parsed calls reports `"tool_calls"`; a
  length limit or abort keeps its own finish reason. `tool_choice: "none"`
  leaves the tools out. The facade accepts `"auto"` and `"none"`, not
  `"required"` or a named-function constraint; grammar-constrained decoding
  is not implemented.
- **Logprobs:** `logprobs: true` with `top_logprobs` (0 to 20) returns
  each generated token's log-probability under the model's raw logits and
  its most likely alternatives, in OpenAI's `choices[].logprobs.content`
  (`generate` takes `logprobs` and puts them on each chunk). In a stream,
  a token's entry comes with its first chunk, or an empty one while its
  text is held back. Token text comes from detokenizing the single token,
  so a token that ends inside a UTF-8 character shows as U+FFFD.
- **Cancelling** (`interruptGenerate`, the request's `AbortSignal`, or a
  consumer that stops reading) drains the work already in flight and ends
  the request with finish reason `abort`. A signal already aborted before
  the request starts throws instead. An early stream consumer does not
  receive the final chunk, but the facade still drains the request before
  releasing the next queued one.
- **Template options:** `chat_template_kwargs` passes extra values to the
  model's own template. For a non-streaming Granite reply,
  `force_nonempty_content: true` returns reasoning as content if the reply
  never reaches content and makes no tool calls.
- **Lifetime:** `unload()` cancels active generation, waits for the queue,
  and releases the model's buffers and core allocations. Create a new
  engine to load another model. `cacheUsage()` counts complete model files;
  `clearCache()` removes the `zllm` OPFS directory, including cached tuning.

## How it works

```mermaid
sequenceDiagram
  participant JS as zllm.js
  participant W as zllm.wasm
  participant G as WebGPU
  JS->>W: zllm_open(header prefix)
  JS->>W: zllm_load(limits)
  loop setup, calibration, then each request
    JS->>W: zllm_next()
    W-->>JS: submit (command batch)
    JS->>G: encode and submit; write_file range reads
    JS->>W: zllm_next() (a second batch may follow)
    W-->>JS: wait
    G-->>JS: work done, reads mapped
    JS->>W: zllm_complete(GPU ms)
    W-->>JS: token / done
  end
```

- **The core sees only the GGUF header.** The shim reads a prefix of the file
  (8 MiB, doubled until it parses). Every weight upload is a `write_file`
  command naming a range of the file. The shim reads it from the source in
  32 MiB pieces, six at a time, and writes it straight to the GPU buffer.
  The original browser report recorded about 70 MiB in the core; this is
  an observation, not a memory ceiling for every model and context.
- **The executor** (`gpu.ts`) is the browser twin of `src/gpu/dawn_host.zig`: one command
  encoder per submit, one compute pass per run of dispatches, pipelines
  compiled with `createComputePipelineAsync` and awaited at first use.
- **Exports** (`src/api/wasm.zig`) take and return plain numbers. Requests
  arrive as JSON (raw text, token ids, or chat messages), and results land
  in one fixed block of wasm memory. A negative return leaves an error
  message there.

The timings below are historical hardware observations, not performance
guarantees or new measurements of the current checkout. Source commits are
identified where the repository records them; their dates are commit
dates, not independently recorded benchmark execution dates. Native
comparisons in these tables use Dawn's D3D12 host, not Agility.

### Weight cache

A URL model can be kept in the origin private file system (OPFS), so later
loads read it from disk when caching is available (`web/src/zllm/cache.ts`).

- **Key:** a hash of the URL, the server's `ETag` or `Last-Modified`, and the
  file size. A changed validator or size produces a new key. This relies on
  the server updating its validator when bytes change. URLs served without
  a validator aren't cached, since a change couldn't be detected. The probe
  that reads the validator bypasses
  the HTTP cache, which would otherwise answer it with an old one; weight
  reads bypass it too, so the bytes aren't stored twice.
- **Filling:** a first load writes every range it reads into the entry,
  which is committed, and marked complete, only after the load succeeds.
  An interrupted load leaves an incomplete entry that later loads ignore and
  replace. Filling a new version of a URL removes the old one.
- **Room:** an entry is only started when the storage estimate has room for
  the whole file. The page requests persistent storage, but the browser
  decides whether to grant it. Local files aren't cached: they are on disk
  already.

| Qwen 3.5 0.8B BF16 (1.5 GB), Chrome | Load | Model requests |
|---|---|---|
| Uncached, over HTTP from localhost | 3.9–4.1 s | 248 |
| First load, filling the cache | 4.8–5.9 s | 248 |
| Cached | 1.0–1.2 s | 1 (the validator probe) |

The cached load's output was identical to the native host's after each of:
a fill, a reload, a changed file (refilled, one entry left), and a fill
interrupted halfway.

Source: `4714c18` (2026-10-29), which documented these cache measurements.
They use localhost HTTP, not a measured Hugging Face download.

### Tuning and the plan cache

Loading tunes the decode multiplies for the GPU as the native CLI does
([GPU backend](gpu.md#tuning)), unless the plan cache holds the tuning for
this GPU, browser, model and build: the core computes the key, and the shim
keeps the cache's files in OPFS under `zllm/plans/` and passes the one for
the key to the load. `tune: false` skips tuning, and `stats().tuning` says
where it came from (`cached`, `swept` or `off`). In Chrome the Gyro-Coder's
first load sweeps for 8 s (29.4 s in all); later loads read the tuning from
the cache (21.8 s).

Source: `7f494e0` (2026-10-30). This cache stores tuning parameters; it does
not preserve live GPU buffers or compiled pipelines. A cache hit still
loads weights, creates pipelines and the step program, and calibrates.

### Device lost

A request in flight when the device is lost (a driver reset, a GPU hang,
or `engine.device.destroy()`) fails with the exported `DeviceLostError`.
It is not replayed automatically. The next request first attempts to load
the model again from the same source on a new device owned by the engine,
using cached tuning when available. Even if the app supplied the original
device, recovery creates a new one. The resident sequence is gone, so the
new request prefills its prompt again. Recovery can itself fail if the
source or a usable device is no longer available.

`engine.device` exposes the current device; `stats().recoveries` counts
successful rebuilds. In the Chrome 0.8B checks recorded in `7929127`
(2026-10-30), destroying the device during a request failed that request
with `DeviceLostError`, and the next greedy request reproduced the earlier
tokens. Loss between requests was checked too. These are exercised
recovery cases, not a promise that every driver failure is recoverable.

### Two steps in flight

A mapped readback in Chrome resolves about 3 ms after its submit, even on
an idle GPU, while `onSubmittedWorkDone` takes 0.1 ms. Waiting for each
token's readback before submitting the next step held decoding at 188 tok/s.

The engine doesn't need the token on the CPU to continue. The sampler
writes it where the next decode step reads it, on the GPU. So the shim keeps
two batches in flight: on `submit` it submits and asks for the next batch
at once, and on `wait` it completes the oldest
([Generation](generation.md#engine)). The readback latency then overlaps the
next step's GPU work.

| Chrome, RTX 5090 Laptop (D3D12) | One step in flight | Two in flight | Native CLI |
|---|---|---|---|
| Decode, greedy, 256 tokens | 188 tok/s | 252–295 tok/s | 285–295 tok/s |
| Prefill, 2,458-token prompt | | 2,239–2,960 tok/s | 2,882–2,896 tok/s |

Loading takes 3.9–4.1 s over HTTP from localhost, and 8.1 s from a local
file read through the `\\wsl` share. The first request after load runs its
prefill pipelines for the first time, which costs about 130 ms more; later
ones don't.

Source: `2acacac` (2026-10-29), the browser execution and readback report.

### Matched 0.8B prefill

The 2026-10-31 comparison uses Chrome **156.0.0.0**, the RTX 5090 Laptop
GPU, and `Qwen3.5-0.8B-BF16.gguf` (1,557,662,496 bytes). Both hosts use
8,192 context positions, f16 KV, stored BF16 weights, scalar-f32 prefill,
and no tuning or speculation. Chrome reports the NVIDIA Blackwell adapter;
its maximum storage binding is 2,147,483,644 bytes, maximum buffer is
2,147,483,648 bytes, and binding alignment is 256 bytes. Both programs
allow steps of up to 512 tokens: these limits do not explain this result.

Native and wasm already use the same `sched.Engine.chunkSize` policy.
Chunks target 90% of the 150 ms submission budget, follow the measured
fixed and per-token costs, and stop at checkpoint and prompt boundaries.
Consequently the same 1,024-token prompt took four native prefill steps
but five browser steps in the initial run. That is intended timing
adaptation, not a second browser planner or a different precision choice.
The initial 3,472 versus 2,087 tok/s result does not isolate host overhead.

For a like-for-like comparison, native `profile` and the browser process
the same seed-1 random-token prefix: a 64-token warmup, then six 64-token
steps covering positions 64–448. Each timed step has **254 dispatches**.
The browser-only measurement harness controls scheduler feedback to hold
this schedule while recording real elapsed times separately; those
controlled feedback values are **not** the timings below. Its extra final
sampled token is excluded. Production scheduling is unchanged.

| Matched 64-token steps, per step | Native Dawn D3D12 | Chrome WebGPU |
|---|---:|---:|
| Profiled wall time | 29.480 ms | 30.433 ms |
| Profiled prefill throughput | 2,171 tok/s | 2,103 tok/s |
| Sum of dispatch timestamp intervals | 26.706 ms | 26.957 ms |
| Wall minus dispatch sum | 2.774 ms | 3.476 ms |
| Dispatches | 254 | 254 |

Both profilers split consecutive dispatches into separate compute passes.
The browser GPU sum is only **0.94%** higher; its profiled wall time is
**3.23%** higher. These are profiling results, not unprofiled throughput.
Chrome timestamps were quantized in 65,536 ns increments.

GPU boost state matters much more than that remaining kernel difference.
After idle periods, browser steps initially took 43–48 ms and a profile
summed to about 45 ms of GPU work. Forty back-to-back requests brought the
last six requests to **28.494 ms per unprofiled step** (2,246 tok/s);
the immediately following profile produced the table above. Concurrent
`nvidia-smi` telemetry observed graphics clocks rising from 1,117 to
2,640 MHz during sustained work. Clocks were not locked. The initial
large kernel gap is therefore not evidence of worse shader translation;
the sustained measurements do not establish a compiler regression.

The sustained unprofiled browser steps spent **0.367 ms** replaying
commands, including **0.144 ms** inside measured WebGPU API calls:
about **0.222 ms** remained in JS replay and its bookkeeping. These steps
have **no model readback or `mapAsync`**: completion uses
`onSubmittedWorkDone`. Profiling adds a timestamp-buffer readback.
The profiled residual includes submission, completion delivery, copies
and profiling overhead; it cannot be split into exclusive asynchronous
IPC and callback costs from these timers.

Ready command replay now stays synchronous until submission or a genuinely
pending pipeline compilation. It no longer creates/awaits promises for
each warmed dispatch; cold compilation still suspends dependent commands
in order and propagates failures. Short-run replay means changed from
0.478 to 0.450 ms per step, within timing noise. No material prefill
throughput gain is claimed for this change; the GPU work dominates.

Reproduce the native side with:

```bash
ZLLM_BACKEND=webgpu scripts/d3d12.sh profile \
  "$(wslpath -w "$PWD/.fixtures/models/Qwen3.5-0.8B-BF16.gguf")" \
  64 0 --no-tune --kv f16
```

The browser used `bun src/dev/serve.ts --port 8080` from `web/`, the local
model URL, and the wasm build. `benchmarks/qwen35-08b-browser-prefill.json`
records the exact token prefix, controlled-feedback protocol, settings,
per-step observations, and cold-to-sustained sequence. See also
[Performance](performance.md#matched-browser-prefill).

### The 27B

Qwen 3.8 27B at Q3_K_M (13.75 GB) runs in Chrome on the same GPU:

| Phase | Chrome |
|---|---|
| Load, uncached, over HTTP from localhost | 25.8 s |
| Load from a local file | 15.3 s |
| Decode, 3,016 tokens of a chat reply | 27.3 tok/s |

The design spec's warm-load target is 40 s. On the recorded machine Chrome
gave the origin a 12.3 GB quota, with 53 GB of disk free, so the 13.75 GB
file did not fit and the cache declined it. The uncached localhost load
and local-file load above are different paths; neither is evidence of an
OPFS cache hit for this model.

A local file needs no weight-cache copy. An app can retain a
`FileSystemFileHandle` and reopen it, subject to browser permissions. These
observations were documented in `81244f5` and `a3f49e4` (2026-10-29).

### Qwen3.8-Flash-Next Gyro-Coder

The 125B Gyro-Coder (75.8 GB) runs in Chrome from the picked file. Only
its 20 GB of weights go to the GPU; its 50.7 GiB per-layer embedding table
stays in the file, and each step reads its tokens' 16 rows apiece through
`write_file`, as the weights were read at load
([GPU backend](gpu.md#qwen38-flash-next)). On the RTX 5090 Laptop:

| Phase | Chrome | Native CLI |
|---|---|---|
| Load from a local file (weights 20 GB) | 19.4–21.4 s | 22 s, with 2–7 s of tuning |
| Decode, greedy | 21.4–22.2 tok/s | 35–38 tok/s |

Chrome's queued writes don't pile up during the load: its shared GPU
memory stays at 0.14 GB. Every decode step still pays for recording its
1,100 dispatches in JavaScript and a mapped readback. The host must know
each token before its step, to read its rows, so the shim keeps one batch
in flight here, not two
([Generation](generation.md#scheduling)). Greedy, from the reference
runtime's prompt, Chrome gives that runtime's 40 tokens; from a chat
prompt, the same 128 tokens as `zllm generate`.

The browser figures were documented in `47ec8f0` and `2a2cadf`
(2026-10-30); cached tuning was added in `7f494e0` later that day.
They predate subsequent native and WGSL kernel changes, so the table is a
recorded comparison, not the latest backend ranking. See [Gyro-Coder](gyro.md)
for the model and [Performance](performance.md) for broader context.

## Validation

- **Same output as native.** In Chrome on the RTX 5090, greedy generation
  from a prompt gives the same 256 tokens as `zllm generate` on the native
  D3D12 host, loaded both over HTTP and from a local file. The Gyro-Coder
  gives the same 128 tokens, and its reference runtime's 40.
- **The engine logic the browser adds** is checked natively by
  `engine-check` ([Validation](validation.md#engine)). One scenario runs a
  request pipelined, with batches completed only on `wait`; it must sample
  the same tokens with bitwise identical logits. Others cancel requests,
  then check that the next prompt reuses exactly what was processed.
- **The wasm build** compiles in CI, and the shim type-checks under the
  strict `tsc` and Biome configuration. The WebGPU path itself runs only in a
  browser with a GPU, so CI doesn't exercise it.

## Not yet

- **Coverage is model-specific:** the engine handles the supported Qwen
  hybrid and dense architectures, Llama, Granite and the Gyro-Coder.
  Dense models already render their own GGUF chat templates with the
  core's Jinja interpreter; the fixed Qwen template is not the only path.
  A model selector entry is not evidence that every browser or device can
  load that model. See [Backends](backends.md) and [Validation](validation.md).
- **One resident sequence:** requests queue, rather than run as a
  multi-user batch. Multiple GPUs and grammar-constrained output are not
  implemented.
- **Precision selection:** f16 is the default KV format. FP8 is available
  by explicit request, but automatic model-quality-gated precision
  selection is not implemented; see [Precision](precision.md).
- **Cache scope:** the weight cache and tuning cache exist. Caching a
  complete compiled execution plan or browser GPU pipelines is not
  implemented.
