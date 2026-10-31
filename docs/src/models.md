# Models

Support is architecture-specific. Reading a GGUF file does not mean the
engine can execute every architecture or tensor layout it contains.
The graph builder, weight loader, tokenizer, chat template and GPU program
must agree on the model's semantics.

| Family | Execution model | Guide |
|---|---|---|
| Qwen 3.5/3.6/3.8 hybrid | Gated DeltaNet interleaved with full attention; optional MTP block | [Qwen GGUF semantics](qwen35.md) |
| Qwen 3 | Dense attention and SwiGLU, with Q/K normalization | [Dense transformers](dense.md) |
| Llama 3 | Dense attention and SwiGLU with Llama tokenizer and RoPE semantics | [Dense transformers](dense.md) |
| Nanbeige 4.3 33B | Existing `llama` graph, SentencePiece and IQ4_XS embeddings; native/browser chat exercised, first-layer model gate still fails | [Dense transformers](dense.md#nanbeige-43-33b) |
| Granite 4 | Dense attention with Granite scaling and tokenizer conventions | [Dense transformers](dense.md) |
| Qwen3.8-Flash-Next Gyro-Coder | Mixture of experts, hyper-connections, sparse attention and per-layer embeddings in Gyro formats | [Gyro-Coder](gyro.md) |

The pinned fixtures and their measured parity are listed in
[Validation](validation.md). A family-level implementation is not a claim
that every checkpoint, quantization or converter revision has been tested.
See [Precision](precision.md) for format support and quality constraints.

## Loading and chat

The native CLI reads a GGUF file. The browser can read a remote URL or a
local file, and can cache remote model bytes in OPFS. Large-model loading
also depends on GPU buffer limits and the architecture's weight-streaming
path; see [Backends](backends.md) and [Browser](browser.md).

Qwen hybrid models and Gyro-Coder use the fixed Qwen text chat template.
Dense models use their GGUF's chat template, rendered by the Zig Jinja
interpreter. Tokenization and rendering stay in the shared core rather
than being reimplemented by each host. See
[Chat format](generation.md#chat-format).

Gemma 4 remains a requested roadmap target, not a supported model family.
Image and video models are future scope. See [Roadmap](roadmap.md).
