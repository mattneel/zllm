// zllm.js: LLM inference on WebGPU, driven by zllm.wasm.
//
//   const engine = await zllm.load({ model: url, onProgress });
//   for await (const chunk of engine.generate({ prompt, maxTokens: 256 })) show(chunk.text);
//   const reply = await engine.chat.completions.create({ messages });

export type { FinishReason } from "./abi";
export {
  type ChatCompletion,
  type ChatCompletionChunk,
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
  type Usage,
} from "./engine";
export type { LoadProgress } from "./gpu";
export { fileSource, type ModelSource, urlSource } from "./source";
