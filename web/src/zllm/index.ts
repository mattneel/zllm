// zllm.js: LLM inference on WebGPU, driven by zllm.wasm.
//
//   const engine = await zllm.load({ model: url, onProgress });
//   for await (const chunk of engine.generate({ prompt, maxTokens: 256 })) show(chunk.text);
//   const reply = await engine.chat.completions.create({ messages });

export type { FinishReason } from "./abi";
export { type CacheUsage, cacheUsage, clearCache } from "./cache";
export {
  type ChatCompletion,
  type ChatCompletionChunk,
  type ChatFinishReason,
  type ChatLogprob,
  type ChatMessage,
  type ChatRequest,
  Engine,
  type GenerateChunk,
  type GenerateRequest,
  type LoadOptions,
  load,
  type OpenAIUsage,
  type Sampling,
  type Stats,
  type TokenLogprob,
  type Usage,
} from "./engine";
export { DeviceLostError, type LoadProgress } from "./gpu";
export { fileSource, type ModelSource, urlSource } from "./source";
export type { Tool, ToolCall } from "./tools";
