//! Tokenizers for zLLM
//!
//! Supported tokenization schemes:
//! - BPE (Byte Pair Encoding) - GPT-2, GPT-3
//! - SentencePiece - LLaMA, T5
//! - WordPiece - BERT
//!
//! All tokenizers use fixed-size buffers and pre-allocated
//! vocabulary tables for zero-allocation operation.

const std = @import("std");

/// Error set for tokenizer operations
pub const TokenizerError = error{
    InvalidVocabulary,
    TokenTooLong,
    InvalidUtf8,
    VocabularyTooLarge,
};

test {
    std.testing.refAllDecls(@This());
}

test "tokenizers module compiles" {
    try std.testing.expect(true);
}
