# Benchmarks

One file per device, one `zllm benchmark` record per line (the spec's
per-device benchmark JSON): load phases in milliseconds, prefill of a
1,024-token random prompt and greedy decode of 128 tokens, through the
engine as `zllm generate` runs them. `decode` is the engine's default
(speculative where the model's MTP block pays, with the drafts accepted),
`decode_plain` one token per step; records before speculative decoding have
`decode` only.

```sh
zllm benchmark <model.gguf>                  # native Dawn
scripts/d3d12.sh benchmark <windows path>    # D3D12 from WSL
```
