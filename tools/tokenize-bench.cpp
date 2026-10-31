// Encode-only llama.cpp comparison. Python supplies length-prefixed UTF-8 records.
#include "llama.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<std::string> read_corpus(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open corpus");
    std::vector<std::string> texts;
    for (;;) {
        unsigned char raw[4];
        in.read(reinterpret_cast<char *>(raw), 4);
        if (in.gcount() == 0 && in.eof()) break;
        if (in.gcount() != 4) throw std::runtime_error("truncated record length");
        const uint32_t size = uint32_t(raw[0]) | uint32_t(raw[1]) << 8 |
                              uint32_t(raw[2]) << 16 | uint32_t(raw[3]) << 24;
        if (size > uint32_t(std::numeric_limits<int32_t>::max())) throw std::runtime_error("record too long");
        std::string text(size, '\0');
        in.read(text.data(), size);
        if (!in) throw std::runtime_error("truncated record text");
        texts.push_back(std::move(text));
    }
    return texts;
}

static int32_t encode(const llama_vocab * vocab, const std::string & text, std::vector<llama_token> & ids) {
    int32_t count = llama_tokenize(vocab, text.data(), int32_t(text.size()), ids.data(), int32_t(ids.size()), true, true);
    if (count < 0) {
        ids.resize(size_t(-int64_t(count)));
        count = llama_tokenize(vocab, text.data(), int32_t(text.size()), ids.data(), int32_t(ids.size()), true, true);
    }
    if (count < 0) throw std::runtime_error("tokenization failed after resizing");
    return count;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: llama-tokenize-bench <model.gguf> <records.bin>\n";
        return 2;
    }
    llama_model * model = nullptr;
    try {
        const auto texts = read_corpus(argv[2]);
        uint64_t corpus_bytes = 0;
        for (const auto & text : texts) corpus_bytes += text.size();
        if (texts.empty() || corpus_bytes == 0) throw std::runtime_error("empty corpus");
        llama_backend_init();
        auto params = llama_model_default_params();
        params.vocab_only = true;
        model = llama_model_load_from_file(argv[1], params);
        if (!model) throw std::runtime_error("could not load model vocabulary");
        const auto * vocab = llama_model_get_vocab(model);
        std::vector<llama_token> ids;
        std::vector<std::vector<llama_token>> expected;
        uint64_t corpus_tokens = 0;
        // One warm pass also retains ids for exact cross-engine parity checks.
        for (const auto & text : texts) {
            const auto count = encode(vocab, text, ids);
            corpus_tokens += count;
            expected.emplace_back(ids.begin(), ids.begin() + count);
        }
        uint64_t repetitions = 0, tokens = 0;
        int64_t elapsed_ns = 0;
        const auto start = std::chrono::steady_clock::now();
        while (elapsed_ns < 500000000) {
            for (const auto & text : texts) tokens += encode(vocab, text, ids);
            ++repetitions;
            elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        }
        const uint64_t bytes = corpus_bytes * repetitions;
        const double seconds = double(elapsed_ns) / 1e9;
        std::cout << std::setprecision(17)
                  << "{\"engine\":\"llama.cpp\",\"corpus_records\":" << texts.size()
                  << ",\"corpus_bytes\":" << corpus_bytes << ",\"corpus_tokens\":" << corpus_tokens
                  << ",\"repetitions\":" << repetitions << ",\"elapsed_ns\":" << elapsed_ns
                  << ",\"bytes\":" << bytes << ",\"tokens\":" << tokens
                  << ",\"mb_per_second\":" << double(bytes) / seconds / 1e6
                  << ",\"tokens_per_second\":" << double(tokens) / seconds << ",\"ids\":[";
        for (size_t i = 0; i < expected.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << '[';
            for (size_t j = 0; j < expected[i].size(); ++j) {
                if (j) std::cout << ',';
                std::cout << expected[i][j];
            }
            std::cout << ']';
        }
        std::cout << "]}\n";
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        if (model) llama_model_free(model);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
