# gpu

The command stream the core emits (`cmd.zig`) and the native host that runs it
(`dawn.zig`, `host.zig`). The core never calls WebGPU: it records commands,
a host executes them and reports completions. The browser host, a TypeScript
shim interpreting the same words, arrives in M3.

Dawn is loaded at run time (`dynlib.zig`) through the C API of the vendored
`third_party/dawn` header; see the [GPU backend](../../docs/src/gpu.md) page.
