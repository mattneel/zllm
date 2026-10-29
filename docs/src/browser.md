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

- **Requests** run one at a time; a second waits for the first. `generate`
  samples plainly unless told otherwise (temperature 1). `chat` defaults to
  the model card's settings, as `zllm chat` does.
- **Chat** renders the fixed Qwen template inside the core, the template
  `check-chat` pins. Each reply is split into `reasoning_content` and
  `content`, as vLLM's Qwen3 reasoning parser does. An assistant turn sent
  back with its `reasoning_content` re-renders exactly as generated, so the
  whole conversation stays resident and a new turn prefills only its new
  message.
- **Tools** are offered as the template offers them, in the system turn
  with its XML call format. An assistant turn's `tool_calls` (arguments
  as JSON text, decoded first, as vLLM does) and `tool` messages render
  as the template renders them. A reply's `<tool_call>` blocks become
  `tool_calls`, with `finish_reason` `"tool_calls"`. Parameter values are
  converted by the tool's schema type, as vLLM's Qwen3 XML parser does.
  Streamed text stops at the first call, and the calls arrive in one
  `tool_calls` delta at the end. `tool_choice: "none"` leaves the tools
  out; forcing a call (`"required"` or a named function) isn't supported.
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
| Decode, 3,016 tokens of a chat reply | 27.3 tok/s |

The spec's bar for a warm load is 40 s, from the weight cache. On this
machine Chrome gives the origin a 12.3 GB quota, which follows the disk's
free space (53 GB), so the 13.75 GB file doesn't fit and the cache
declines it. Loading it uncached over localhost already takes 25.8 s.

## Validation

- **Same output as native.** In Chrome on the RTX 5090, greedy generation
  from a prompt gives the same 256 tokens as `zllm generate` on the native
  D3D12 host, loaded both over HTTP and from a local file.
- **The engine logic the browser adds** is checked natively by
  `engine-check` ([Validation](validation.md#engine)). One scenario runs a
  request pipelined, with batches completed only on `wait`; it must sample
  the same tokens with bitwise identical logits. Others cancel requests,
  then check that the next prompt reuses exactly what was processed.
- **The wasm build** compiles in CI, and the shim type-checks under the
  strict `tsc` and Biome configuration. The WebGPU path itself runs only in a
  browser with a GPU, so CI doesn't exercise it.

## Not yet

- **Model and template coverage:** only the Qwen 3.5 BF16 GGUFs, with the
  fixed Qwen template. Rendering a GGUF's own chat template with
  `@huggingface/jinja` is still to come.
- **Plan cache:** a plan cache (M4) would skip setup work that a cached
  load still does: compiling pipelines and calibrating.
