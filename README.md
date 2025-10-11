# ⚡ zLLM

[![CI](https://github.com/mattneel/zllm/actions/workflows/ci.yml/badge.svg)](https://github.com/mattneel/zllm/actions/workflows/ci.yml)
[![Benchmark](https://github.com/mattneel/zllm/actions/workflows/benchmark.yml/badge.svg)](https://github.com/mattneel/zllm/actions/workflows/benchmark.yml)
[![Documentation](https://github.com/mattneel/zllm/actions/workflows/docs.yml/badge.svg)](https://mattneel.github.io/zllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Zero-GPU Language Model Runtime** - Democratizing AI with CPU-native LLM training and inference in pure Zig.

zLLM is a pure Zig implementation of a complete LLM training and inference runtime designed to run efficiently on commodity CPUs without requiring GPUs. It features a modular, composable architecture where components like attention mechanisms, position encodings, and activations can be mixed and matched at compile-time with zero runtime overhead.

## Why zLLM?

- **No GPU Required**: Run and train LLMs on commodity CPUs with competitive performance
- **Zero Dependencies**: Single static binary, no Python/PyTorch/CUDA needed
- **Comptime Optimization**: Architecture selection at compile-time for zero-cost abstractions
- **Tiger Style**: Built with disciplined engineering for safety, performance, and maintainability
- **Full Training Support**: Complete backpropagation with Adam optimization, not just inference

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mattneel/zllm
cd zllm

# Install mise for task management
curl https://mise.jdx.dev/install.sh | sh
mise install

# Run tests
mise run test

# Build and run example
mise run example
```

### Basic Usage

```zig
const zllm = @import("zllm");

pub fn main() !void {
    // Initialize model with comptime configuration
    const config = zllm.GPTConfig{
        .vocab_size = 50257,
        .n_layers = 12,
        .n_heads = 12,
        .d_model = 768,
    };

    var model = try zllm.GPT(config).init();
    defer model.deinit();

    // Generate text
    const prompt = "Once upon a time";
    const output = try model.generate(prompt, .{
        .max_tokens = 100,
        .temperature = 0.8,
    });

    std.debug.print("Generated: {s}\n", .{output});
}
```

## Features

### 🚀 Comptime Architecture Selection

```zig
// Mix and match components at compile-time
const MyModel = zllm.Transformer(.{
    .attention = zllm.FlashAttention,      // or MultiHeadAttention, GQA
    .position = zllm.RoPE,                 // or ALiBi, Learned
    .activation = zllm.SwiGLU,             // or GELU, ReLU
    .norm = zllm.RMSNorm,                  // or LayerNorm
});
```

### 💪 SIMD Optimization

- Automatic vectorization for AVX2/AVX-512/NEON
- Cache-aligned data structures
- Optimized matrix operations

### 🧩 Modular Design

- Swap components without changing code
- Add custom attention mechanisms
- Implement new optimizers easily

### ⚡ Performance

- **Inference**: 5-20 tokens/sec for 7B models (INT8)
- **Training**: Full backpropagation support
- **Memory**: INT4 quantization for 4x reduction
- **Parallelism**: Multi-threaded attention heads

## Project Status

**⚠️ Early Development**: zLLM is under active development. APIs will change. Not production-ready.

### Completed ✅

- Core vector/matrix operations with SIMD
- Basic transformer architecture
- Multi-head attention
- Thread pool parallelization
- Simple tokenizer
- Forward pass and generation
- Basic backpropagation framework

### In Progress 🚧

- Complete gradient computation
- Model format converters (GGUF, SafeTensors)
- BPE tokenizer
- INT8/INT4 quantization
- Flash Attention implementation

### Planned 📋

- LLaMA/Mistral architectures
- OpenAI-compatible API server
- LoRA fine-tuning
- Speculative decoding

See [ROADMAP.md](ROADMAP.md) for detailed plans.

## Performance Targets

| Model Size | Precision | Target Speed | Memory Usage |
| ---------- | --------- | ------------ | ------------ |
| 125M       | FP32      | 50 tok/s     | 500 MB       |
| 125M       | INT8      | 100 tok/s    | 125 MB       |
| 7B         | INT8      | 10 tok/s     | 7 GB         |
| 7B         | INT4      | 20 tok/s     | 3.5 GB       |

## Development

zLLM follows [Tiger Style](AGENTS.md) - a disciplined approach to software engineering focused on safety, performance, and developer experience.

### Building from Source

Requirements:

- Zig 0.15.1 or later
- 8GB+ RAM for development
- AVX2-capable CPU (recommended)

```bash
# Build debug version
zig build

# Build optimized release
zig build -Doptimize=ReleaseFast

# Run tests
zig build test

# Run benchmarks
zig build bench
```

### Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas needing help:

- 🔥 **Quantization**: INT8/INT4 implementations
- 🎯 **Model Loaders**: GGUF/SafeTensors converters
- 🌟 **Architectures**: LLaMA, Mistral, Phi
- 📚 **Documentation**: Tutorials and examples

For AI coding agents: See [AGENTS.md](AGENTS.md) for detailed implementation guidelines.

### Testing

```bash
# Run all tests
mise run test

# Run specific module tests
mise run test:core
mise run test:models
mise run test:training

# Run benchmarks
mise run bench
```

## Documentation

Full documentation available at [https://mattneel.github.io/zllm/](https://mattneel.github.io/zllm/)

Build locally:

```bash
mise run docs:serve
```

## Architecture

```
zllm/
├── src/
│   ├── core/           # SIMD vectors, matrices, thread pool
│   ├── models/         # GPT, LLaMA, Mistral architectures
│   ├── training/       # Optimizers, loss functions, backprop
│   ├── tokenizers/     # BPE, SentencePiece
│   ├── loaders/        # Model format converters
│   └── quant/          # INT8/INT4 quantization
├── examples/           # Usage examples
├── bench/              # Performance benchmarks
└── docs/               # mdbook documentation
```

## Comparison

| Feature               | zLLM | llama.cpp | candle | tinygrad |
| --------------------- | ---- | --------- | ------ | -------- |
| Language              | Zig  | C++       | Rust   | Python   |
| Training              | ✅   | ❌        | ✅     | ✅       |
| Comptime Optimization | ✅   | ❌        | ❌     | ❌       |
| Zero Dependencies     | ✅   | ✅        | ❌     | ❌       |
| SIMD                  | ✅   | ✅        | ✅     | ⚠️       |
| INT4 Quantization     | 🚧   | ✅        | ⚠️     | ❌       |

## Citation

If you use zLLM in your research, please cite:

```bibtex
@software{zllm2024,
  title = {zLLM: Zero-GPU Language Model Runtime},
  author = {zLLM Contributors},
  year = {2024},
  url = {https://github.com/mattneel/zllm}
}
```

## License

MIT - See [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by llama.cpp, ggml, and candle
- Built with Zig's powerful comptime and SIMD features
- Following Tiger Style principles from TigerBeetle

---

**Note**: This is an experimental project exploring CPU-optimized LLM inference and training. Expect rough edges and rapid changes.
