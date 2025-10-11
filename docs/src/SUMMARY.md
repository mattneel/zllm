# Summary

[Introduction](./introduction.md)

# User Guide

- [Getting Started](./guide/getting_started.md)
  - [Installation](./guide/installation.md)
  - [Quick Start](./guide/quickstart.md)
  - [Basic Usage](./guide/basic_usage.md)
- [Core Concepts](./guide/concepts.md)
  - [Comptime Architecture](./guide/comptime.md)
  - [SIMD Optimization](./guide/simd.md)
  - [Memory Management](./guide/memory.md)
- [Model Architectures](./guide/models.md)
  - [GPT Models](./guide/models/gpt.md)
  - [LLaMA Models](./guide/models/llama.md)
  - [Custom Models](./guide/models/custom.md)
- [Training](./guide/training.md)
  - [Data Preparation](./guide/training/data.md)
  - [Training Loop](./guide/training/loop.md)
  - [Fine-tuning](./guide/training/finetuning.md)
- [Inference](./guide/inference.md)
  - [Text Generation](./guide/inference/generation.md)
  - [Sampling Strategies](./guide/inference/sampling.md)
  - [Performance Tuning](./guide/inference/performance.md)

# Developer Guide

- [Architecture Overview](./dev/architecture.md)
- [Building from Source](./dev/building.md)
- [Module System](./dev/modules.md)
  - [Core Module](./dev/modules/core.md)
  - [Models Module](./dev/modules/models.md)
  - [Training Module](./dev/modules/training.md)
  - [Tokenizers Module](./dev/modules/tokenizers.md)
- [Writing Components](./dev/components.md)
  - [Custom Attention](./dev/components/attention.md)
  - [Custom Activations](./dev/components/activation.md)
  - [Custom Optimizers](./dev/components/optimizer.md)
- [Testing](./dev/testing.md)
- [Benchmarking](./dev/benchmarking.md)
- [Contributing](./dev/contributing.md)

# API Reference

- [Core API](./api/core.md)
  - [Vector Operations](./api/core/vector.md)
  - [Matrix Operations](./api/core/matrix.md)
  - [Thread Pool](./api/core/thread_pool.md)
- [Model API](./api/models.md)
  - [Transformer](./api/models/transformer.md)
  - [Attention Mechanisms](./api/models/attention.md)
  - [Position Encodings](./api/models/position.md)
- [Training API](./api/training.md)
  - [Optimizers](./api/training/optimizers.md)
  - [Loss Functions](./api/training/loss.md)
  - [Trainer](./api/training/trainer.md)
- [Tokenizer API](./api/tokenizers.md)

# Examples

- [Hello World](./examples/hello_world.md)
- [Training GPT-2](./examples/train_gpt2.md)
- [Custom Architecture](./examples/custom_arch.md)
- [Quantization](./examples/quantization.md)
- [Model Conversion](./examples/conversion.md)

# Appendix

- [Benchmarks](./appendix/benchmarks.md)
- [Model Formats](./appendix/formats.md)
- [Troubleshooting](./appendix/troubleshooting.md)
- [FAQ](./appendix/faq.md)
- [Glossary](./appendix/glossary.md)
- [License](./appendix/license.md)
