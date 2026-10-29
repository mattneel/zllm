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
});

for await (const chunk of engine.generate({ prompt: "Once upon a time", maxTokens: 200 })) {
  out.textContent += chunk.text; // the last chunk carries finish and usage
}

for await (const chunk of engine.chat.completions.create({ messages, stream: true })) {
  const delta = chunk.choices[0].delta; // reasoning_content, then content
}
const reply = await engine.chat.completions.create({ messages, enable_thinking: false });

engine.tokenize("text"); engine.detokenize(ids);
engine.interruptGenerate(); // or pass `signal`, or stop reading the stream
engine.stats(); await engine.unload();
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
  `@huggingface/jinja`, tool-call parsers and WebLLM's `engineFactory` are
  still to come.
- **Weight cache:** weights are fetched on every load; an OPFS or IndexedDB
  cache is one of the spec's deferred decisions.
- **Plan cache:** a plan cache (M4) would skip setup work on reload.
