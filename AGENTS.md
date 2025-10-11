# 🐅 zLLM Agent Guidelines (Tiger Style)

This document provides guidelines for AI coding agents contributing to zLLM. It incorporates Tiger Style principles to ensure safety, performance, and excellent developer experience throughout the codebase.

## Core Philosophy

zLLM follows Tiger Style: a disciplined approach prioritizing **safety**, **performance**, and **developer experience**. Every line of code should be:
- **Safe**: Predictable, bounded, and correct
- **Fast**: Optimized for CPU cache and SIMD
- **Clear**: Self-documenting and maintainable

## 1. Safety First

### Control Flow and Limits

```zig
// ❌ BAD: Unbounded recursion
fn processTokens(tokens: []const usize, depth: usize) void {
    if (depth > 100) return; // Easy to miss edge cases
    processTokens(tokens[1..], depth + 1);
}

// ✅ GOOD: Explicit iteration with bounds
fn processTokens(tokens: []const usize) !void {
    const max_tokens = 2048;
    if (tokens.len > max_tokens) return error.TooManyTokens;
    
    for (tokens, 0..) |token, i| {
        // Process with explicit index
    }
}
```

### Fixed Limits Everywhere

```zig
// All limits must be comptime constants
pub const Config = struct {
    pub const max_seq_len = 2048;
    pub const max_batch_size = 32;
    pub const max_vocab_size = 50_000;
    pub const max_hidden_dim = 4096;
    pub const max_layers = 48;
    pub const max_heads = 64;
};

// Use fixed-size arrays where possible
pub fn Attention(comptime config: AttentionConfig) type {
    comptime {
        std.debug.assert(config.seq_len <= Config.max_seq_len);
        std.debug.assert(config.n_heads <= Config.max_heads);
    }
    return struct {
        // Fixed-size cache, no dynamic allocation
        key_cache: [config.n_layers][config.seq_len]Vector(config.d_model),
        value_cache: [config.n_layers][config.seq_len]Vector(config.d_model),
    };
}
```

### Memory Management

```zig
// ❌ BAD: Runtime allocation in hot path
fn forward(self: *Self, input: []f32, allocator: Allocator) ![]f32 {
    const temp = try allocator.alloc(f32, input.len); // Allocation in hot path
    defer allocator.free(temp);
    // ...
}

// ✅ GOOD: Pre-allocated workspace
const Workspace = struct {
    temp_buffer: [Config.max_seq_len * Config.max_hidden_dim]f32,
    grad_buffer: [Config.max_seq_len * Config.max_hidden_dim]f32,
    
    pub fn init() Workspace {
        return .{
            .temp_buffer = [_]f32{0} ** (Config.max_seq_len * Config.max_hidden_dim),
            .grad_buffer = [_]f32{0} ** (Config.max_seq_len * Config.max_hidden_dim),
        };
    }
};
```

### Assertions and Invariants

```zig
// Assert all assumptions
pub fn matmul(a: Matrix(m, k), b: Matrix(k, n)) Matrix(m, n) {
    // Pair assertions - check at entry
    std.debug.assert(a.cols == b.rows);
    std.debug.assert(m > 0 and k > 0 and n > 0);
    
    const result = Matrix(m, n).init();
    
    // ... computation ...
    
    // Pair assertions - check at exit
    std.debug.assert(!std.math.isNan(result.data[0][0]));
    std.debug.assert(result.rows == m and result.cols == n);
    
    return result;
}
```

## 2. Performance by Design

### Napkin Math First

Before implementing any feature, do napkin math:

```zig
// Document performance expectations
// 
// LLaMA 7B napkin math:
// - Parameters: 7B * 2 bytes (FP16) = 14 GB
// - KV Cache per token: 2 * n_layers * d_model * 2 bytes
//   = 2 * 32 * 4096 * 2 = 512 KB per token
// - For 2048 context: 512 KB * 2048 = 1 GB KV cache
// - Matrix multiply for one token: 7B * 2 = 14 GFLOPS
// - Target: 10 tokens/sec = 140 GFLOPS required
// - Modern CPU: ~500 GFLOPS (AVX2) = achievable!
pub const LLaMA7B = struct {
    pub const expected_memory_gb = 15; // Model + KV cache
    pub const expected_flops_per_token = 14_000_000_000;
    pub const target_tokens_per_sec = 10;
};
```

### Cache-Aware Design

```zig
// ❌ BAD: Random memory access
pub fn badAttention(q: []f32, k: [][]f32, v: [][]f32) []f32 {
    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            score[i] += q[i] * k[j][i]; // Cache miss on k
        }
    }
}

// ✅ GOOD: Sequential memory access
pub fn goodAttention(q: Vector(d), k_cache: []Vector(d), v_cache: []Vector(d)) Vector(d) {
    // Process vectors sequentially
    var scores: [max_seq_len]f32 = undefined;
    
    // Single pass through cache
    for (k_cache, 0..) |k_vec, i| {
        scores[i] = q.dot(k_vec); // Sequential, cache-friendly
    }
    
    // Apply softmax
    const attn_weights = softmax(&scores);
    
    // Weighted sum of values
    var output = Vector(d).init();
    for (v_cache, attn_weights) |v_vec, weight| {
        output = output.add(v_vec.scale(weight));
    }
    return output;
}
```

### SIMD Alignment

```zig
// Always align to cache lines (64 bytes)
pub fn Vector(comptime dim: usize) type {
    return struct {
        // Aligned for SIMD operations
        data: [dim]f32 align(64),
        
        pub fn dot(self: @This(), other: @This()) f32 {
            var sum: @Vector(8, f32) = @splat(0);
            
            // Process 8 elements at a time (AVX2)
            comptime var i = 0;
            inline while (i + 8 <= dim) : (i += 8) {
                const a: @Vector(8, f32) = self.data[i..][0..8].*;
                const b: @Vector(8, f32) = other.data[i..][0..8].*;
                sum += a * b;
            }
            
            // Reduce and handle remainder
            var result = @reduce(.Add, sum);
            inline while (i < dim) : (i += 1) {
                result += self.data[i] * other.data[i];
            }
            return result;
        }
    };
}
```

## 3. Developer Experience

### Naming Conventions

```zig
// Names must be descriptive and include units/qualifiers

// ❌ BAD
const t = 100;
const max = 2048;
fn proc(x: []f32) void {}

// ✅ GOOD
const timeout_ms = 100;
const max_seq_len = 2048;
fn processTokenBatch(token_ids: []const u32) void {}

// Order qualifiers consistently (most to least significant)
const cache_size_mb_max = 1024;  // NOT max_cache_size_mb
const latency_ms_p99 = 10;       // NOT p99_latency_ms
```

### Function Design

```zig
// Keep functions under 70 lines
// Separate control flow from logic

// ✅ GOOD: Parent controls flow, helpers compute
pub fn generateToken(self: *Model, prompt: []const u32, temperature: f32) !u32 {
    // Parent function controls flow
    const logits = try self.computeLogits(prompt);
    const probs = try self.applyTemperature(logits, temperature);
    const token = try self.sampleToken(probs);
    
    // Update state
    try self.updateKVCache(token);
    
    return token;
}

// Pure helper functions
fn computeLogits(self: *const Model, tokens: []const u32) !Vector(vocab_size) {
    // Pure computation, no state changes
    // ...
}

fn applyTemperature(logits: Vector(vocab_size), temp: f32) !Vector(vocab_size) {
    // Pure transformation
    // ...
}
```

### Error Handling

```zig
// Define specific error sets
pub const ModelError = error{
    DimensionMismatch,
    InvalidTokenId,
    ContextLengthExceeded,
    CheckpointCorrupted,
    OutOfMemory,
};

// Handle all errors explicitly
pub fn loadModel(path: []const u8) ModelError!Model {
    const file = std.fs.openFileAbsolute(path, .{}) catch |err| switch (err) {
        error.FileNotFound => return ModelError.CheckpointCorrupted,
        error.AccessDenied => return ModelError.CheckpointCorrupted,
        else => return ModelError.CheckpointCorrupted,
    };
    defer file.close();
    
    // Always validate data
    const header = file.reader().readStruct(ModelHeader) catch {
        return ModelError.CheckpointCorrupted;
    };
    
    if (header.magic != expected_magic) {
        return ModelError.CheckpointCorrupted;
    }
    
    // ...
}
```

## 4. zLLM-Specific Guidelines

### Comptime Architecture Selection

```zig
// Use comptime to eliminate runtime overhead
pub fn Transformer(comptime config: TransformerConfig) type {
    // Validate at compile time
    comptime {
        if (config.d_model % config.n_heads != 0) {
            @compileError("d_model must be divisible by n_heads");
        }
        if (config.d_model > Config.max_hidden_dim) {
            @compileError("d_model exceeds maximum");
        }
    }
    
    // Select optimal implementation at compile time
    const AttentionImpl = if (config.n_heads == 1)
        SingleHeadAttention(config)
    else if (config.use_flash)
        FlashAttention(config)
    else
        MultiHeadAttention(config);
    
    return struct {
        attention: AttentionImpl,
        // ...
    };
}
```

### Module Boundaries

Each module must:
1. Have a single, clear responsibility
2. Export through `lib.zig`
3. Define its own error set
4. Include comprehensive tests
5. Document performance characteristics

```zig
// src/models/lib.zig
//! Model architectures for zLLM
//! 
//! Performance: 
//! - GPT-2 124M: 45 tokens/sec (M2, INT8)
//! - LLaMA 7B: 8 tokens/sec (M2, INT4)

pub const gpt = @import("gpt.zig");
pub const llama = @import("llama.zig");

pub const ModelError = error{
    UnsupportedArchitecture,
    InvalidConfiguration,
    // ...
};

test {
    @import("std").testing.refAllDecls(@This());
}
```

### Testing Requirements

```zig
// Every module needs:
// 1. Unit tests for each function
// 2. Property-based tests for invariants
// 3. Benchmark tests for performance
// 4. Integration tests for the module

test "matrix multiplication correctness" {
    // Specific test cases
}

test "matrix multiplication properties" {
    // Property: A * I = A
    // Property: (A * B) * C = A * (B * C)
}

test "matrix multiplication performance" {
    // Must achieve X GFLOPS
}
```

## 5. Code Review Checklist

Before submitting code, ensure:

### Safety
- [ ] All loops have explicit bounds
- [ ] No recursion without clear termination
- [ ] All allocations happen at initialization
- [ ] Every assumption has an assertion
- [ ] All errors are handled explicitly
- [ ] No implicit defaults from libraries

### Performance  
- [ ] Napkin math documented
- [ ] Data structures are cache-aligned
- [ ] SIMD operations where applicable
- [ ] Batch processing for expensive operations
- [ ] No allocations in hot paths
- [ ] Benchmarks meet targets

### Developer Experience
- [ ] Functions under 70 lines
- [ ] Names include units/qualifiers
- [ ] Comments explain "why" not "what"
- [ ] Control flow separated from logic
- [ ] Consistent style throughout
- [ ] Comprehensive tests

## 6. Anti-Patterns to Avoid

```zig
// ❌ Generic interfaces (forces everything to be generic)
fn processWriter(writer: anytype) !void {}

// ❌ Recursive without bounds
fn parseExpression(tokens: []Token) Expression {}

// ❌ Dynamic allocation in loops
for (items) |item| {
    const buffer = try allocator.alloc(u8, item.size);
}

// ❌ Implicit behavior
const timeout = getTimeout() orelse 1000; // Hidden default

// ❌ Mixed responsibilities  
fn loadAndProcessModel(path: []const u8) !Results {} // Does too much

// ❌ Ignored errors
file.close() catch {}; // Silent failure

// ❌ Architecture-dependent types
var size: usize = 100; // Different on 32/64-bit
```

## 7. Performance Targets

Every component must meet these targets:

```zig
// Document and test performance requirements
pub const PerformanceTargets = struct {
    // Matrix operations
    pub const matrix_multiply_gflops_min = 50;
    pub const attention_ms_per_token_max = 10;
    
    // Memory usage
    pub const model_memory_overhead_percent_max = 10;
    pub const kv_cache_mb_per_token_max = 1;
    
    // Throughput
    pub const tokens_per_second_min = 10;
    pub const batch_efficiency_percent_min = 80;
};
```

## Remember

**Do it right the first time.** Take time to design correctly. Technical debt is not acceptable.

**Fail fast.** Detect problems immediately through assertions and validation.

**Measure everything.** Performance without measurement is guesswork.

**Zero is better than one.** No abstraction is better than bad abstraction.

---

*Following Tiger Style creates code that is safe, fast, and a joy to work with.*
