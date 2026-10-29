# fuse

Not started; planned for M5.

Graph fusion pass. A fusion is legal only when its reduction domains are owned by one workgroup and it fits device limits; fused kernels are checked layer by layer against the unfused path.

See the [spec](../../docs/src/spec.md) and [architecture](../../docs/src/architecture.md).
