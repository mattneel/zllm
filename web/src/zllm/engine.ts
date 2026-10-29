// The zllm engine in the browser: zllm.wasm drives generation; this file
// feeds it the model header, runs its command batches on WebGPU and turns
// its events into async iterables.

import { Event, type Exports, FINISH, type FinishReason, NEED_MORE, Out } from "./abi";
import { type Cached, openCached } from "./cache";
import { Executor, type LoadProgress, planLimits, requestDevice } from "./gpu";
import { fileSource, type ModelSource, urlSource } from "./source";
import { argumentsValue, type Tool, type ToolCall, ToolCallSplitter } from "./tools";

/** First read of the model file for its header; doubled until it parses. */
const HEADER_PREFIX = 8 << 20;

export interface LoadOptions {
  /** A URL served with range requests, a local file, or any `ModelSource`. */
  model: string | URL | Blob | FileSystemFileHandle | ModelSource;
  /** Where zllm.wasm is; by default next to this module. */
  wasm?: string | URL;
  /** Context capacity in tokens (default 8192). */
  context?: number;
  /** KV cache format (default f16). */
  kv?: "f16" | "f32";
  /**
   * Keep the model in the browser's origin private file system, so later
   * loads read it from disk (default true; URLs only, see cache.ts).
   */
  cache?: boolean;
  /** An existing device; by default the high-performance adapter's. */
  device?: GPUDevice;
  onProgress?: (p: LoadProgress) => void;
}

export interface Sampling {
  temperature?: number;
  topK?: number;
  topP?: number;
  minP?: number;
  repetitionPenalty?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  /** Seed for the draws; random when omitted. */
  seed?: number;
  /** Token id to logit bias. */
  logitBias?: Record<number, number>;
}

export interface GenerateRequest {
  /** Raw text to continue, or token ids. */
  prompt?: string;
  tokens?: number[];
  maxTokens?: number;
  /** Plain sampling (temperature 1) where unset. */
  sampling?: Sampling;
  /** Strings that end generation; not part of the output. */
  stop?: string[];
  /**
   * Report each token's log-probability and this many most likely tokens'
   * (0 to 20), under the model's raw logits.
   */
  logprobs?: number;
  signal?: AbortSignal;
}

/** A token's log-probability, with the most likely tokens'. */
export interface TokenLogprob {
  logprob: number;
  top: { token: number; logprob: number }[];
}

export interface Usage {
  promptTokens: number;
  /** Prompt tokens served from the resident sequence instead of prefill. */
  cachedTokens: number;
  completionTokens: number;
  prefillSteps: number;
  prefillMs: number;
  decodeMs: number;
}

export interface GenerateChunk {
  /** Text released by this token (may be empty while UTF-8 or a stop string is pending). */
  text: string;
  tokens: number[];
  /** With `logprobs`. */
  logprob?: TokenLogprob;
  /** Set on the last chunk, with `usage`. */
  finish?: FinishReason;
  usage?: Usage;
}

export interface ChatMessage {
  role: "system" | "developer" | "user" | "assistant" | "tool";
  /** Null for an assistant turn that only called tools. */
  content: string | null;
  /** An assistant turn's reasoning, as returned in `reasoning_content`. */
  reasoning_content?: string;
  /** An assistant turn's calls. */
  tool_calls?: ToolCall[];
  /** A tool turn's call (accepted; the template doesn't show it). */
  tool_call_id?: string;
}

/** Why a chat reply ended: as generation, or because it called tools. */
export type ChatFinishReason = FinishReason | "tool_calls";

export interface ChatRequest {
  messages: ChatMessage[];
  stream?: boolean;
  max_tokens?: number;
  max_completion_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  repetition_penalty?: number;
  seed?: number;
  stop?: string | string[];
  logit_bias?: Record<string, number>;
  /** Qwen thinking (default on), and its effort ("none" turns it off). */
  enable_thinking?: boolean;
  reasoning_effort?: string;
  /** Each content token's log-probability, and `top_logprobs` alternatives. */
  logprobs?: boolean;
  top_logprobs?: number;
  /** Functions the model may call. */
  tools?: Tool[];
  /** "auto" (default): call tools or not; "none": tools are left out. */
  tool_choice?: "auto" | "none";
  signal?: AbortSignal;
}

export interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: {
    index: 0;
    delta: {
      role?: "assistant";
      content?: string;
      reasoning_content?: string;
      tool_calls?: (ToolCall & { index: number })[];
    };
    logprobs?: { content: ChatLogprob[] } | null;
    finish_reason: ChatFinishReason | null;
  }[];
  usage?: OpenAIUsage;
}

export interface ChatCompletion {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: {
    index: 0;
    message: {
      role: "assistant";
      content: string | null;
      reasoning_content?: string;
      tool_calls?: ToolCall[];
    };
    logprobs?: { content: ChatLogprob[] } | null;
    finish_reason: ChatFinishReason;
  }[];
  usage: OpenAIUsage;
}

/** A generated token's log-probability, as OpenAI reports it. */
export interface ChatLogprob {
  token: string;
  logprob: number;
  bytes: number[];
  top_logprobs: { token: string; logprob: number; bytes: number[] }[];
}

export interface OpenAIUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details: { cached_tokens: number };
}

/** A request as zllm_start reads it (src/api/wasm.zig `RequestJson`). */
interface RequestJson extends SamplingJson {
  prompt?: string;
  tokens?: number[];
  messages?: {
    role: ChatMessage["role"];
    content: string;
    reasoning: string | null;
    tool_calls: { function: { name: string; arguments: unknown } }[];
  }[];
  tools?: Tool[];
  enable_thinking?: boolean;
  reasoning_effort?: string | null;
  logprobs?: number;
  max_tokens: number;
  stop: string[];
}

interface SamplingJson {
  temperature?: number;
  top_k?: number;
  top_p?: number;
  min_p?: number;
  repetition_penalty?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  seed?: number;
  logit_bias?: { token: number; bias: number }[];
}

export interface Stats {
  adapter: string;
  model: Record<string, unknown>;
  loadMs: number;
  /** Where the weights came from: the cache, the network while filling it, or uncached. */
  cache: "hit" | "filled" | "off";
  /** The last request's usage. */
  last: Usage | null;
}

export async function load(opts: LoadOptions): Promise<Engine> {
  const t0 = performance.now();
  const opened = await openSource(opts.model);
  const cached: Cached = opts.cache === false ? { none: opened } : await openCached(opened);
  const source = "hit" in cached ? cached.hit : "fill" in cached ? cached.fill.source : cached.none;
  try {
    const engine = await loadFrom(source, opts);
    if ("fill" in cached) {
      await cached.fill.commit();
    }
    engine.cache = "hit" in cached ? "hit" : "fill" in cached ? "filled" : "off";
    engine.loadMs = performance.now() - t0;
    return engine;
  } catch (e) {
    if ("fill" in cached) {
      await cached.fill.discard();
    }
    throw e;
  }
}

async function loadFrom(source: ModelSource, opts: LoadOptions): Promise<Engine> {
  const wasmUrl = opts.wasm ?? new URL("./zllm.wasm", import.meta.url);
  const { instance } = await WebAssembly.instantiateStreaming(fetch(wasmUrl), {
    env: {
      zllm_log(ptr: number, len: number): void {
        console.debug(`zllm: ${readText(instance.exports as unknown as Exports, ptr, len)}`);
      },
    },
  });
  const x = instance.exports as unknown as Exports;

  // The header: read a prefix of the file until it parses.
  for (let len = Math.min(HEADER_PREFIX, source.size); ; len = Math.min(2 * len, source.size)) {
    const bytes = await source.read(0, len);
    const ptr = x.zllm_alloc(len);
    if (ptr === 0) {
      throw new Error("zllm: out of memory for the model header");
    }
    new Uint8Array(x.memory.buffer, ptr, len).set(bytes);
    const r = x.zllm_open(ptr, len);
    if (r !== NEED_MORE || len === source.size) {
      check(x, r);
      break;
    }
  }

  const { device, adapter } = opts.device
    ? { device: opts.device, adapter: "WebGPU device" }
    : await requestDevice();
  const exec = new Executor(device, x.memory, source);
  exec.onProgress = opts.onProgress ?? null;
  callJson(x, x.zllm_load, {
    context: opts.context ?? 8192,
    kv: opts.kv ?? "f16",
    ...planLimits(device),
  });
  const engine = new Engine(x, exec, adapter);
  // The setup batch (buffers, weights, pipelines) and calibration.
  for await (const _ of engine.drive()) {
    // Loading emits no tokens.
  }
  return engine;
}

async function openSource(model: LoadOptions["model"]): Promise<ModelSource> {
  if (typeof model === "string" || model instanceof URL) {
    return urlSource(String(model));
  }
  if (model instanceof Blob) {
    return fileSource(model);
  }
  if ("getFile" in model) {
    return fileSource(await model.getFile());
  }
  return model;
}

export class Engine {
  readonly #x: Exports;
  readonly #exec: Executor;
  readonly #adapter: string;
  /** Requests run one at a time; each waits for the previous. */
  #queue: Promise<unknown> = Promise.resolve();
  #last: Usage | null = null;
  #unloaded = false;
  /**
   * Completions of submitted batches, oldest first. Kept here, not in a
   * `drive` call: a consumer that stops early abandons its generator, and
   * the drain that follows must still complete those batches.
   */
  readonly #inFlight: Promise<number>[] = [];
  loadMs = 0;
  cache: Stats["cache"] = "off";

  readonly chat = {
    completions: {
      create: ((req: ChatRequest) =>
        req.stream === true ? this.#chatStream(req) : this.#chatOnce(req)) as {
        (req: ChatRequest & { stream: true }): AsyncIterable<ChatCompletionChunk>;
        (req: ChatRequest): Promise<ChatCompletion>;
      },
    },
  };

  constructor(x: Exports, exec: Executor, adapter: string) {
    this.#x = x;
    this.#exec = exec;
    this.#adapter = adapter;
  }

  /** Continues `prompt` (or `tokens`), streaming each sampled token. */
  generate(req: GenerateRequest): AsyncIterable<GenerateChunk> {
    const body: RequestJson = {
      max_tokens: req.maxTokens ?? 4096,
      stop: req.stop ?? [],
      ...samplingJson(req.sampling ?? {}),
    };
    if (req.logprobs !== undefined) {
      body.logprobs = req.logprobs;
    }
    if (req.tokens !== undefined) {
      body.tokens = req.tokens;
    } else {
      body.prompt = req.prompt ?? "";
    }
    return this.#request(body, req.signal);
  }

  tokenize(text: string): number[] {
    this.#live();
    const bytes = new TextEncoder().encode(text);
    const ptr = this.#copyIn(bytes);
    try {
      check(this.#x, this.#x.zllm_tokenize(ptr, bytes.length));
    } finally {
      this.#x.zllm_free(ptr, Math.max(bytes.length, 1));
    }
    const out = this.#out();
    return Array.from(
      new Uint32Array(
        this.#x.memory.buffer,
        out.getUint32(Out.ptr, true),
        out.getUint32(Out.count, true),
      ),
    );
  }

  detokenize(ids: number[]): string {
    this.#live();
    const words = new Uint32Array(ids);
    const ptr = this.#copyIn(new Uint8Array(words.buffer));
    try {
      check(this.#x, this.#x.zllm_detokenize(ptr, ids.length));
    } finally {
      this.#x.zllm_free(ptr, Math.max(words.byteLength, 1));
    }
    const out = this.#out();
    return readText(this.#x, out.getUint32(Out.ptr, true), out.getUint32(Out.len, true));
  }

  /** Ends the running request after its current GPU step; it finishes with "abort". */
  interruptGenerate(): void {
    if (!this.#unloaded) {
      this.#x.zllm_cancel();
    }
  }

  stats(): Stats {
    this.#live();
    check(this.#x, this.#x.zllm_info());
    const out = this.#out();
    const info = readText(this.#x, out.getUint32(Out.ptr, true), out.getUint32(Out.len, true));
    return {
      adapter: this.#adapter,
      model: JSON.parse(info),
      loadMs: this.loadMs,
      cache: this.cache,
      last: this.#last,
    };
  }

  /** Frees the model: its GPU buffers and the core's memory. */
  async unload(): Promise<void> {
    if (this.#unloaded) {
      return;
    }
    this.interruptGenerate();
    await this.#queue.catch(() => undefined);
    this.#unloaded = true;
    this.#x.zllm_unload();
    this.#exec.destroyAll();
  }

  /**
   * Runs the engine until it idles: executes its batches, up to two in
   * flight, yields tokens, and returns at the request's end. Callers hold
   * the queue.
   */
  async *drive(): AsyncGenerator<GenerateChunk, void> {
    const x = this.#x;
    const inFlight = this.#inFlight;
    for (;;) {
      const ev = x.zllm_next();
      check(x, ev);
      const out = this.#out();
      switch (ev) {
        case Event.submit: {
          const { done } = await this.#exec.execute(
            out.getUint32(Out.ptr, true),
            out.getUint32(Out.count, true),
          );
          inFlight.push(done);
          break;
        }
        case Event.wait: {
          const oldest = inFlight.shift();
          if (oldest === undefined) {
            throw new Error("zllm: waiting with no batch in flight");
          }
          check(x, x.zllm_complete(await oldest));
          break;
        }
        case Event.token: {
          const chunk: GenerateChunk = {
            text: readText(x, out.getUint32(Out.ptr, true), out.getUint32(Out.len, true)),
            tokens: [out.getUint32(Out.token, true)],
          };
          const logprob = out.getFloat32(Out.logprob, true);
          if (!Number.isNaN(logprob)) {
            const at = out.getUint32(Out.topPtr, true);
            const view = new DataView(x.memory.buffer, at, 8 * out.getUint32(Out.topCount, true));
            const top: TokenLogprob["top"] = [];
            for (let i = 0; i < view.byteLength; i += 8) {
              top.push({ token: view.getUint32(i, true), logprob: view.getFloat32(i + 4, true) });
            }
            chunk.logprob = { logprob, top };
          }
          yield chunk;
          break;
        }
        case Event.done: {
          const usage: Usage = {
            promptTokens: out.getUint32(Out.promptTokens, true),
            cachedTokens: out.getUint32(Out.cachedTokens, true),
            completionTokens: out.getUint32(Out.completionTokens, true),
            prefillSteps: out.getUint32(Out.prefillSteps, true),
            prefillMs: out.getFloat64(Out.prefillMs, true),
            decodeMs: out.getFloat64(Out.decodeMs, true),
          };
          this.#last = usage;
          const finish = FINISH[out.getUint32(Out.finish, true)];
          if (finish === undefined) {
            throw new Error("zllm: unknown finish reason");
          }
          yield { text: "", tokens: [], finish, usage };
          return;
        }
        case Event.idle:
          return;
        default:
          throw new Error(`zllm: unknown event ${ev}`);
      }
    }
  }

  /** Queues a request and streams it. */
  async *#request(
    body: RequestJson,
    signal: AbortSignal | undefined,
  ): AsyncGenerator<GenerateChunk> {
    this.#live();
    const { promise: turn, resolve: release } = Promise.withResolvers<void>();
    const previous = this.#queue;
    this.#queue = previous.then(() => turn);
    await previous.catch(() => undefined);
    const onAbort = (): void => this.interruptGenerate();
    try {
      signal?.throwIfAborted();
      signal?.addEventListener("abort", onAbort);
      body.seed ??= Math.floor(Math.random() * 2 ** 53);
      callJson(this.#x, this.#x.zllm_start, body);
      let finished = false;
      try {
        for await (const chunk of this.drive()) {
          finished = chunk.finish !== undefined;
          yield chunk;
        }
      } finally {
        // A consumer that stops early cancels; the engine still runs to idle.
        if (!finished) {
          this.#x.zllm_cancel();
          for await (const _ of this.drive()) {
            // Drain to the abort.
          }
        }
      }
    } finally {
      signal?.removeEventListener("abort", onAbort);
      release();
    }
  }

  #chatBody(req: ChatRequest): RequestJson {
    const sampling: Sampling = {};
    if (req.temperature !== undefined) sampling.temperature = req.temperature;
    if (req.top_p !== undefined) sampling.topP = req.top_p;
    if (req.top_k !== undefined) sampling.topK = req.top_k;
    if (req.min_p !== undefined) sampling.minP = req.min_p;
    if (req.presence_penalty !== undefined) sampling.presencePenalty = req.presence_penalty;
    if (req.frequency_penalty !== undefined) sampling.frequencyPenalty = req.frequency_penalty;
    if (req.repetition_penalty !== undefined) sampling.repetitionPenalty = req.repetition_penalty;
    if (req.seed !== undefined) sampling.seed = req.seed;
    if (req.logit_bias !== undefined) {
      sampling.logitBias = Object.fromEntries(
        Object.entries(req.logit_bias).map(([k, v]) => [Number(k), v]),
      );
    }
    const thinking = req.enable_thinking ?? true;
    return {
      messages: req.messages.map((m) => ({
        role: m.role,
        content: m.content ?? "",
        reasoning: m.reasoning_content ?? null,
        tool_calls: (m.tool_calls ?? []).map((c) => ({
          function: { name: c.function.name, arguments: argumentsValue(c.function.arguments) },
        })),
      })),
      ...(offersTools(req) ? { tools: req.tools } : {}),
      enable_thinking: thinking,
      reasoning_effort: req.reasoning_effort ?? null,
      ...(req.logprobs === true ? { logprobs: req.top_logprobs ?? 0 } : {}),
      max_tokens: req.max_completion_tokens ?? req.max_tokens ?? 4096,
      stop: typeof req.stop === "string" ? [req.stop] : (req.stop ?? []),
      ...samplingJson(sampling),
    };
  }

  async *#chatStream(req: ChatRequest): AsyncGenerator<ChatCompletionChunk> {
    const id = `chatcmpl-${crypto.randomUUID()}`;
    const created = Math.floor(Date.now() / 1000);
    const model = "zllm";
    const chunk = (
      delta: ChatCompletionChunk["choices"][number]["delta"],
      finish: ChatFinishReason | null,
    ): ChatCompletionChunk => ({
      id,
      object: "chat.completion.chunk",
      created,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
    yield chunk({ role: "assistant", content: "" }, null);
    const split = new ReasoningSplitter(req.enable_thinking ?? true);
    // Content passes through the tool-call splitter when tools are offered.
    const calls = offersTools(req) ? new ToolCallSplitter(req.tools ?? []) : null;
    function* parts(
      deltas: Iterable<{ content?: string; reasoning_content?: string }>,
    ): Generator<ChatCompletionChunk> {
      for (const d of deltas) {
        if (d.content !== undefined && calls !== null) {
          for (const text of calls.push(d.content)) {
            yield chunk({ content: text }, null);
          }
        } else {
          yield chunk(d, null);
        }
      }
    }
    for await (const c of this.#request(this.#chatBody(req), req.signal)) {
      const out = [...parts(split.push(c.text))];
      if (c.logprob !== undefined) {
        // The token's logprobs ride on its first chunk, or on an empty one
        // while its text is held back.
        const first = out[0] ?? chunk({}, null);
        if (out.length === 0) {
          out.push(first);
        }
        const choice = first.choices[0];
        if (choice !== undefined) {
          choice.logprobs = { content: [this.#chatLogprob(c.tokens[0] ?? 0, c.logprob)] };
        }
      }
      yield* out;
      if (c.finish !== undefined && c.usage !== undefined) {
        yield* parts(split.flush());
        let finish: ChatFinishReason = c.finish;
        if (calls !== null) {
          const { calls: made, rest } = calls.finish();
          if (rest !== "") {
            yield chunk({ content: rest }, null);
          }
          if (made.length > 0) {
            yield chunk({ tool_calls: made.map((call, index) => ({ index, ...call })) }, null);
            if (finish === "stop") {
              finish = "tool_calls";
            }
          }
        }
        yield { ...chunk({}, finish), usage: openaiUsage(c.usage) };
      }
    }
  }

  async #chatOnce(req: ChatRequest): Promise<ChatCompletion> {
    let content = "";
    let reasoning = "";
    const toolCalls: ToolCall[] = [];
    const logprobs: ChatLogprob[] = [];
    let last: ChatCompletionChunk | null = null;
    for await (const c of this.#chatStream(req)) {
      const delta = c.choices[0]?.delta;
      logprobs.push(...(c.choices[0]?.logprobs?.content ?? []));
      content += delta?.content ?? "";
      reasoning += delta?.reasoning_content ?? "";
      for (const { index: _, ...call } of delta?.tool_calls ?? []) {
        toolCalls.push(call);
      }
      last = c;
    }
    const finish = last?.choices[0]?.finish_reason;
    if (last === null || finish === null || finish === undefined || last.usage === undefined) {
      throw new Error("zllm: the request ended without finishing");
    }
    const message: ChatCompletion["choices"][number]["message"] = {
      role: "assistant",
      content: content === "" && toolCalls.length > 0 ? null : content,
    };
    if (reasoning !== "") {
      message.reasoning_content = reasoning;
    }
    if (toolCalls.length > 0) {
      message.tool_calls = toolCalls;
    }
    const choice: ChatCompletion["choices"][number] = { index: 0, message, finish_reason: finish };
    if (req.logprobs === true) {
      choice.logprobs = { content: logprobs };
    }
    return {
      id: last.id,
      object: "chat.completion",
      created: last.created,
      model: last.model,
      choices: [choice],
      usage: last.usage,
    };
  }

  /** A token's logprobs as OpenAI reports them. */
  #chatLogprob(id: number, lp: TokenLogprob): ChatLogprob {
    const text = (token: number) => this.detokenize([token]);
    const bytes = (s: string) => Array.from(new TextEncoder().encode(s));
    const token = text(id);
    return {
      token,
      logprob: lp.logprob,
      bytes: bytes(token),
      top_logprobs: lp.top.map((t) => {
        const s = text(t.token);
        return { token: s, logprob: t.logprob, bytes: bytes(s) };
      }),
    };
  }

  #live(): void {
    if (this.#unloaded) {
      throw new Error("zllm: engine unloaded");
    }
  }

  #out(): DataView {
    return new DataView(this.#x.memory.buffer, this.#x.zllm_out());
  }

  /** Copies bytes into a fresh core allocation; the caller frees it. */
  #copyIn(bytes: Uint8Array): number {
    const ptr = this.#x.zllm_alloc(Math.max(bytes.length, 1));
    if (ptr === 0) {
      throw new Error("zllm: out of memory");
    }
    new Uint8Array(this.#x.memory.buffer, ptr, bytes.length).set(bytes);
    return ptr;
  }
}

/**
 * Splits a Qwen thinking reply into reasoning (up to `</think>`) and
 * content, as vLLM's Qwen3 reasoning parser does. The chat template opens
 * the `<think>` block in the prompt, so the reply starts inside it.
 */
class ReasoningSplitter {
  static readonly END = "</think>";
  #inReasoning: boolean;
  #held = "";
  #contentStarted = false;

  constructor(thinking: boolean) {
    this.#inReasoning = thinking;
  }

  *push(text: string): Generator<{ content?: string; reasoning_content?: string }> {
    if (!this.#inReasoning) {
      yield* this.#content(text);
      return;
    }
    this.#held += text;
    const end = this.#held.indexOf(ReasoningSplitter.END);
    if (end >= 0) {
      const reasoning = this.#held.slice(0, end);
      const rest = this.#held.slice(end + ReasoningSplitter.END.length);
      this.#held = "";
      this.#inReasoning = false;
      if (reasoning !== "") {
        yield { reasoning_content: reasoning };
      }
      yield* this.#content(rest);
      return;
    }
    // Hold back a tail that could start "</think>".
    let keep = 0;
    for (let n = Math.min(ReasoningSplitter.END.length - 1, this.#held.length); n > 0; n--) {
      if (ReasoningSplitter.END.startsWith(this.#held.slice(-n))) {
        keep = n;
        break;
      }
    }
    const ready = this.#held.slice(0, this.#held.length - keep);
    this.#held = this.#held.slice(this.#held.length - keep);
    if (ready !== "") {
      yield { reasoning_content: ready };
    }
  }

  *flush(): Generator<{ content?: string; reasoning_content?: string }> {
    if (this.#held !== "") {
      yield this.#inReasoning ? { reasoning_content: this.#held } : { content: this.#held };
      this.#held = "";
    }
  }

  /** Content, without the blank lines the template puts after `</think>`. */
  *#content(text: string): Generator<{ content?: string }> {
    const out = this.#contentStarted ? text : text.replace(/^\n+/, "");
    if (out !== "") {
      this.#contentStarted = true;
      yield { content: out };
    }
  }
}

/** Whether the request offers tools to the model. */
function offersTools(req: ChatRequest): boolean {
  return (req.tools?.length ?? 0) > 0 && req.tool_choice !== "none";
}

function samplingJson(s: Sampling): SamplingJson {
  const out: SamplingJson = {};
  if (s.temperature !== undefined) out.temperature = s.temperature;
  if (s.topK !== undefined) out.top_k = s.topK;
  if (s.topP !== undefined) out.top_p = s.topP;
  if (s.minP !== undefined) out.min_p = s.minP;
  if (s.repetitionPenalty !== undefined) out.repetition_penalty = s.repetitionPenalty;
  if (s.presencePenalty !== undefined) out.presence_penalty = s.presencePenalty;
  if (s.frequencyPenalty !== undefined) out.frequency_penalty = s.frequencyPenalty;
  if (s.seed !== undefined) out.seed = s.seed;
  if (s.logitBias !== undefined) {
    out.logit_bias = Object.entries(s.logitBias).map(([id, bias]) => ({
      token: Number(id),
      bias,
    }));
  }
  return out;
}

function openaiUsage(u: Usage): OpenAIUsage {
  return {
    prompt_tokens: u.promptTokens,
    completion_tokens: u.completionTokens,
    total_tokens: u.promptTokens + u.completionTokens,
    prompt_tokens_details: { cached_tokens: u.cachedTokens },
  };
}

/** Calls a JSON-taking export with `value` encoded into core memory. */
function callJson(x: Exports, f: (ptr: number, len: number) => number, value: unknown): void {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  const ptr = x.zllm_alloc(Math.max(bytes.length, 1));
  if (ptr === 0) {
    throw new Error("zllm: out of memory");
  }
  new Uint8Array(x.memory.buffer, ptr, bytes.length).set(bytes);
  try {
    check(x, f(ptr, bytes.length));
  } finally {
    x.zllm_free(ptr, Math.max(bytes.length, 1));
  }
}

/** Throws the core's error message for a negative result. */
function check(x: Exports, r: number): void {
  if (r >= 0) {
    return;
  }
  const out = new DataView(x.memory.buffer, x.zllm_out());
  throw new Error(
    `zllm: ${readText(x, out.getUint32(Out.ptr, true), out.getUint32(Out.len, true))}`,
  );
}

function readText(x: Exports, ptr: number, len: number): string {
  return new TextDecoder().decode(new Uint8Array(x.memory.buffer, ptr, len));
}
