# api

The browser entry point: `wasm.zig` builds to `zllm.wasm` (`zig build wasm`,
installed as `zig-out/web/zllm.wasm`). Its exports drive the engine through
the TypeScript shim in `web/src/zllm`, which owns the GPU and the model file.

See the [spec](../../docs/src/spec.md).
