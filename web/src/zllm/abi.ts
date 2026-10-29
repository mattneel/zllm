// The contract between this shim and zllm.wasm (src/api/wasm.zig): the
// exports, the result block the core fills, and the command stream opcodes
// (src/gpu/cmd.zig).

/** Exports of zllm.wasm. Pointers and lengths are byte offsets into `memory`. */
export interface Exports {
  readonly memory: WebAssembly.Memory;
  /** Allocates `len` bytes; 0 when out of memory. */
  zllm_alloc(len: number): number;
  zllm_free(ptr: number, len: number): void;
  /**
   * Parses the GGUF header from `len` bytes at `ptr`, taking ownership of
   * the allocation: `NEED_MORE` asks for a longer prefix of the file.
   */
  zllm_open(ptr: number, len: number): number;
  /** Builds the model and step program from a JSON `LoadOptions` at `ptr`. */
  zllm_load(ptr: number, len: number): number;
  /** Advances the engine: one of `Event`, or a negative error. */
  zllm_next(): number;
  /** Reports that the oldest batch in flight finished after `ms` GPU milliseconds. */
  zllm_complete(ms: number): number;
  /** Starts a request from a JSON `RequestJson` at `ptr`. */
  zllm_start(ptr: number, len: number): number;
  /** Ends the current request after the batch in flight. */
  zllm_cancel(): void;
  /** Tokenizes UTF-8 text: the ids land at `Out.ptr`, `Out.count` of them. */
  zllm_tokenize(ptr: number, len: number): number;
  /** Decodes `count` u32 ids: the UTF-8 text lands at `Out.ptr`, `Out.len`. */
  zllm_detokenize(ptr: number, count: number): number;
  /** Model and configuration as JSON at `Out.ptr`, `Out.len`. */
  zllm_info(): number;
  /** Frees the model, program and engine. GPU objects are the host's to drop. */
  zllm_unload(): void;
  /** Address of the result block (`Out`). */
  zllm_out(): number;
}

/** `zllm_next` results. */
export const Event = {
  /** Execute the batch at `Out.ptr` (`Out.count` u32 words); `zllm_next` may follow before it completes. */
  submit: 0,
  /** A sampled token (`Out.token`) and the text it released (`Out.ptr`, `Out.len`). */
  token: 1,
  /** The request ended: `Out.finish` and the usage fields. */
  done: 2,
  /** Nothing to do: loading finished, or no request is running. */
  idle: 3,
  /** Complete the oldest batch in flight (`zllm_complete`) before anything else. */
  wait: 4,
} as const;

/** `zllm_open` result asking for more of the file. */
export const NEED_MORE = 1;

export const FINISH = ["stop", "length", "abort"] as const;
export type FinishReason = (typeof FINISH)[number];

/** Field offsets in the result block (extern struct Out in wasm.zig). */
export const Out = {
  token: 0,
  finish: 4,
  ptr: 8,
  len: 12,
  count: 16,
  promptTokens: 20,
  cachedTokens: 24,
  completionTokens: 28,
  prefillSteps: 32,
  topCount: 36,
  prefillMs: 40,
  decodeMs: 48,
  logprob: 56,
  topPtr: 60,
} as const;

/** Command stream opcodes (src/gpu/cmd.zig `Op`). */
export const Op = {
  createBuffer: 1,
  writeBuffer: 2,
  createShader: 3,
  createPipeline: 4,
  createBindGroup: 5,
  dispatch: 6,
  copyBuffer: 7,
  readBuffer: 8,
  submit: 9,
  destroy: 10,
  clearBuffer: 11,
  writeFile: 12,
} as const;

/** `destroy` kinds (src/gpu/cmd.zig `Kind`). */
export const Kind = { buffer: 0, shader: 1, pipeline: 2, bindGroup: 3 } as const;
