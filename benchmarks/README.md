# Benchmarks

One file per device, one `zllm benchmark` record per line (the spec's
per-device benchmark JSON): load phases in milliseconds, prefill of a
1,024-token random prompt and greedy decode of 128 tokens, through the
engine as `zllm generate` runs them.

```sh
zllm benchmark <model.gguf>                  # native Dawn
scripts/d3d12.sh benchmark <windows path>    # D3D12 from WSL
```
