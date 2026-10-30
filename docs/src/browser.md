# Browser

M3 runs the engine in the browser. `zllm.wasm` is the same Zig core the
native CLI uses, built for `wasm32-freestanding`. `zllm.js` (`web/src/zllm`)
is the shim around it: it runs the core's command batches on WebGPU, streams
weights from the model file, and exposes the spec's JavaScript API.

```sh
zig build wasm            # zig-out/web/zllm.wasm
cd web
bun run dev               # demo on http://localhost:8080/, models from ../.fixtures/models
bun run build             # dist/zllm.js and dist/zllm.wasm
```

## API

```ts
import * as zllm from "./zllm.js";

const engine = await zllm.load({
  model: "/models/Qwen3.5-0.8B-BF16.gguf", // URL (range requests), File, Blob or FileSystemFileHandle
  onProgress: ({ loaded, total }) => console.log(loaded / total),
  // cache: false, to skip the weight cache
});

for await (const chunk of engine.generate({ prompt: "Once upon a time", maxTokens: 200 })) {
  out.textContent += chunk.text; // the last chunk carries finish and usage
}

for await (const chunk of engine.chat.completions.create({ messages, stream: true })) {
  const delta = chunk.choices[0].delta; // reasoning_content, then content
}
const reply = await engine.chat.completions.create({ messages, enable_thinking: false });
// A model with its own template (Llama, Granite, ...) also takes vLLM's
// chat_template_kwargs:
await engine.chat.completions.create({ messages, chat_template_kwargs: { date_string: "1 Nov 2026" } });

// Tool calling, in the OpenAI format.
const tools = [{ type: "function", function: { name: "get_weather", parameters: { /* JSON Schema */ } } }];
const turn = await engine.chat.completions.create({ messages, tools });
// turn.choices[0]: finish_reason "tool_calls", message.tool_calls [{ id, type, function: { name, arguments } }]
messages.push(turn.choices[0].message, { role: "tool", tool_call_id: id, content: "18°C, clear" });

engine.tokenize("text"); engine.detokenize(ids);
engine.interruptGenerate(); // or pass `signal`, or stop reading the stream
engine.stats(); await engine.unload();
await zllm.cacheUsage(); await zllm.clearCache();
```

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
  `tool_choice: "none"` leaves the tools out; forcing a call (`"required"`
  or a named function) needs grammar-constrained decoding, which the spec
  leaves out of v1.
- **Logprobs:** `logprobs: true` with `top_logprobs` (0 to 20) returns
  each generated token's log-probability under the model's raw logits and
  its most likely alternatives, in OpenAI's `choices[].logprobs.content`
  (`generate` takes `logprobs` and puts them on each chunk). In a stream,
  a token's entry comes with its first chunk, or an empty one while its
  text is held back. Token text comes from detokenizing the single token,
  so a token that ends inside a UTF-8 character shows as U+FFFD.
- **Cancelling** (`interruptGenerate`, the request's `AbortSignal`, or a
  consumer that stops reading) ends the request after the GPU step in
  flight, with finish reason `abort`.

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
  32 MiB pieces, six at a time, and writes it straight to the GPU buffer. The
  core holds about 70 MiB.
- **The executor** (`gpu.ts`) is the browser twin of `host.zig`: one command
  encoder per submit, one compute pass per run of dispatches, pipelines
  compiled with `createComputePipelineAsync` and awaited at first use.
- **Exports** (`src/api/wasm.zig`) take and return plain numbers. Requests
  arrive as JSON (raw text, token ids, or chat messages), and results land
  in one fixed block of wasm memory. A negative return leaves an error
  message there.

### Weight cache

A model loaded from a URL is kept in the origin private file system (OPFS),
so later loads read it from disk (`web/src/zllm/cache.ts`).

- **Key:** a hash of the URL, the server's `ETag` or `Last-Modified`, and the
  file size. A changed file has a new validator and so a new key, and an
  entry is never stale. URLs served without a validator aren't cached, since
  a change couldn't be detected. The probe that reads the validator bypasses
  the HTTP cache, which would otherwise answer it with an old one; weight
  reads bypass it too, so the bytes aren't stored twice.
- **Filling:** a first load writes every range it reads into the entry,
  which is committed, and marked complete, only after the load succeeds.
  An interrupted load leaves an incomplete entry that later loads ignore and
  replace. Filling a new version of a URL removes the old one.
- **Room:** an entry is only started when the storage estimate has room for
  the whole file, and the page asks for persistent storage so the browser
  doesn't evict it under pressure. Local files aren't cached: they are on
  disk already.

| Qwen 3.5 0.8B BF16 (1.5 GB), Chrome | Load | Model requests |
|---|---|---|
| Uncached, over HTTP from localhost | 3.9–4.1 s | 248 |
| First load, filling the cache | 4.8–5.9 s | 248 |
| Cached | 1.0–1.2 s | 1 (the validator probe) |

The cached load's output was identical to the native host's after each of:
a fill, a reload, a changed file (refilled, one entry left), and a fill
interrupted halfway.

### Tuning and the plan cache

Loading tunes the decode multiplies for the GPU as the native CLI does
([GPU backend](gpu.md#tuning)), unless the plan cache holds the tuning for
this GPU, browser, model and build: the core computes the key, and the shim
keeps the cache's files in OPFS under `zllm/plans/` and passes the one for
the key to the load. `tune: false` skips tuning, and `stats().tuning` says
where it came from (`cached`, `swept` or `off`). In Chrome the Gyro-Coder's
first load sweeps for 8 s (29.4 s in all); later loads read the tuning from
the cache (21.8 s).

### Device lost

A request in flight when the device is lost (a driver reset, a GPU hang,
`engine.device.destroy()`) fails with `DeviceLostError`. The next request
first loads the model again on a new device of the engine's own, from the
same source, with the tuning from the plan cache, and then runs; the
resident sequence is gone. `stats().recoveries` counts the rebuilds. In
Chrome, with the 0.8B, a device destroyed in the middle of a request fails
it with `DeviceLostError`, and the next greedy request gives the same tokens
as before the loss; so does one after a loss between requests.

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

### The 27B

Qwen 3.8 27B at Q3_K_M (13.75 GB) runs in Chrome on the same GPU:

| Phase | Chrome |
|---|---|
| Load, uncached, over HTTP from localhost | 25.8 s |
| Load from a local file | 15.3 s |
| Decode, 3,016 tokens of a chat reply | 27.3 tok/s |

The spec's bar for a warm load is 40 s, from the weight cache. On this
machine Chrome gives the origin a 12.3 GB quota, which follows the disk's
free space (53 GB), so the 13.75 GB file doesn't fit and the cache
declines it. Loading it uncached over localhost already takes 25.8 s.

A model picked as a local file needs no cache: its bytes are on disk
already. Loaded that way, the 27B takes 15.3 s, the warm load the spec's
40 s bar is about, since the API takes a `FileSystemFileHandle` an app can
keep and reopen.

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

- **Model and template coverage:** Qwen 3.5 and 3.8 GGUFs (BF16, F16 and the
  quantized types; the Gyro-Coder up to 32,768 tokens of context), with the
  fixed Qwen template. Rendering a GGUF's own chat template with
  `@huggingface/jinja` is still to come.
- **Plan cache:** a plan cache (M4) would skip setup work that a cached
  load still does: compiling pipelines and calibrating.
