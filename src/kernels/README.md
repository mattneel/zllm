# kernels

Not started; planned for M1.

WGSL kernel templates (`*.wgsl.hbs`), each with a parameter schema checked against device limits, a cost model (FLOPs, bytes), and a reference variant. Shapes go in uniforms; parameters are baked into the WGSL.

See the [spec](../../docs/src/spec.md) and [architecture](../../docs/src/architecture.md).
