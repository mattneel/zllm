# 📁 zLLM Project Structure

```
zllm/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # Main CI pipeline
│   │   ├── benchmark.yml       # Performance regression tests
│   │   ├── release.yml         # Release automation
│   │   └── docs.yml           # Documentation deployment
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── model_support.md
│   └── dependabot.yml
│
├── src/
│   ├── core/                   # Core primitives and operations
│   │   ├── lib.zig            # Module exports
│   │   ├── vector.zig         # SIMD vector operations
│   │   ├── matrix.zig         # Matrix operations
│   │   ├── tensor.zig         # N-dimensional tensors
│   │   ├── simd.zig           # SIMD utilities
│   │   ├── thread_pool.zig    # Parallel execution
│   │   └── memory.zig         # Memory management
│   │
│   ├── models/                 # Model architectures
│   │   ├── lib.zig
│   │   ├── transformer.zig    # Base transformer
│   │   ├── gpt.zig           # GPT-2/3 architecture
│   │   ├── llama.zig         # LLaMA family
│   │   ├── mistral.zig       # Mistral with sliding window
│   │   ├── phi.zig           # Microsoft Phi models
│   │   └── components/
│   │       ├── attention.zig  # MHA, GQA, MQA, Flash
│   │       ├── position.zig   # RoPE, ALiBi, learned
│   │       ├── activation.zig # GELU, SwiGLU, ReLU
│   │       └── norm.zig       # LayerNorm, RMSNorm
│   │
│   ├── tokenizers/            # Text tokenization
│   │   ├── lib.zig
│   │   ├── bpe.zig           # Byte-Pair Encoding
│   │   ├── sentencepiece.zig # SentencePiece
│   │   ├── wordpiece.zig     # WordPiece
│   │   └── vocab.zig         # Vocabulary management
│   │
│   ├── training/              # Training infrastructure
│   │   ├── lib.zig
│   │   ├── optimizer.zig     # Adam, SGD, AdamW
│   │   ├── loss.zig          # Loss functions
│   │   ├── trainer.zig       # Training loop
│   │   ├── gradient.zig      # Backpropagation
│   │   ├── scheduler.zig     # Learning rate schedules
│   │   └── checkpoint.zig    # Model checkpointing
│   │
│   ├── quant/                 # Quantization
│   │   ├── lib.zig
│   │   ├── int8.zig          # INT8 quantization
│   │   ├── int4.zig          # INT4 quantization
│   │   ├── dynamic.zig       # Dynamic quantization
│   │   └── qat.zig           # Quantization-aware training
│   │
│   ├── loaders/               # Model format converters
│   │   ├── lib.zig
│   │   ├── gguf.zig          # GGUF format (llama.cpp)
│   │   ├── safetensors.zig   # SafeTensors format
│   │   ├── pytorch.zig       # PyTorch checkpoints
│   │   └── zllm.zig          # Native zLLM format
│   │
│   ├── inference/             # Inference optimizations
│   │   ├── lib.zig
│   │   ├── kv_cache.zig      # KV cache management
│   │   ├── sampling.zig      # Token sampling strategies
│   │   ├── speculative.zig   # Speculative decoding
│   │   └── batch.zig         # Batch inference
│   │
│   ├── server/                # API server
│   │   ├── main.zig          # Server entry point
│   │   ├── routes.zig        # API routes
│   │   ├── openai.zig        # OpenAI compatibility
│   │   └── websocket.zig     # Streaming support
│   │
│   ├── cli/                   # Command-line interface
│   │   ├── main.zig          # CLI entry point
│   │   ├── commands.zig      # CLI commands
│   │   └── repl.zig          # Interactive REPL
│   │
│   └── lib.zig               # Main library exports
│
├── examples/                  # Example applications
│   ├── generate.zig          # Text generation
│   ├── train.zig             # Training from scratch
│   ├── finetune.zig          # Fine-tuning
│   ├── chat.zig             # Interactive chat
│   ├── embeddings.zig        # Generate embeddings
│   └── quantize.zig          # Quantize models
│
├── tests/                     # Tests
│   ├── integration.zig       # Integration tests
│   ├── e2e/                  # End-to-end tests
│   │   ├── generation.zig
│   │   ├── training.zig
│   │   └── conversion.zig
│   └── fixtures/             # Test data
│       ├── tiny_model.json
│       └── test_text.txt
│
├── bench/                     # Benchmarks
│   ├── bench.zig             # Main benchmark suite
│   ├── matrix_ops.zig        # Matrix operation benchmarks
│   ├── attention.zig         # Attention benchmarks
│   ├── inference.zig         # Inference benchmarks
│   └── comparison/           # Comparative benchmarks
│       ├── llama_cpp.zig
│       └── baseline.json
│
├── scripts/                   # Utility scripts
│   ├── convert_model.py      # Model conversion
│   ├── download_models.py    # Model downloading
│   ├── compare_benchmarks.py # Benchmark comparison
│   ├── lint_comptime.py      # Comptime linting
│   ├── process_docs.py       # Documentation processing
│   └── package_release.py    # Release packaging
│
├── docs/                      # Documentation
│   ├── architecture.md       # Architecture overview
│   ├── getting_started.md    # Quick start guide
│   ├── api/                  # API documentation
│   │   ├── core.md
│   │   ├── models.md
│   │   └── training.md
│   ├── guides/               # User guides
│   │   ├── custom_model.md
│   │   ├── training.md
│   │   ├── quantization.md
│   │   └── deployment.md
│   └── blog/                 # Technical blog posts
│       ├── comptime_optimization.md
│       └── cpu_inference.md
│
├── models/                    # Model storage (gitignored)
│   └── .gitkeep
│
├── data/                      # Training data (gitignored)
│   └── tiny_shakespeare.txt  # Example dataset
│
├── .gitignore
```
