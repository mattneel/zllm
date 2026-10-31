# fuse

The fusion pass (M5), `fuse.zig`. A step program describes a stretch of ops as a small graph over buffers (`fuse.Op`); `fuse.lower` fuses where legal and lowers the rest to their own launches.

- **Generated from hand-written fragments:** a fused kernel is the consumer's hand-written kernel template rendered with the producer as a fragment (today the quantized decode multiply's staging prologue, `MatvecVariant.act_div`). Each backend writes its fragments in its own language; fusing never leaves hand-written code.
- **Legality is backend-independent, here:** a producer fuses into its consumer only if its output has that one reader, in the graph and outside it (`Options.live_out`), the consumer's kernel has the fragment for every weight chunk (the prologue: quantized weights only), and the shapes fit the fragment's layout (whole vectors per row).
- **Checked:** `zig build test` pins the legality rules; `gpu-check` the fused kernel against float64; `check-gyro` the whole program fused against unfused (`gpu-forward --no-fuse`), every layer's records bitwise.

Fusions so far: `hc_act` into the up multiply of the Gyro-Coder's hyper-connection mixer (97 dispatches a decode step).
