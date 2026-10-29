// llama-dump: run one teacher-forced forward pass with the pinned llama.cpp
// and write selected activations for every position.
//
//   llama-dump <model.gguf> <tokens.i32> <out.bin>
//
// tokens.i32 is a flat little-endian int32 array. The whole sequence is one
// batch and one ubatch, so every captured tensor covers all positions. The KV
// cache is f32 so llama.cpp does not round attention inputs to f16.
//
// out.bin is a sequence of records:
//   u32 name_len, name bytes, u32 n_dims, i64 ne[n_dims], f32 data[prod(ne)]
// with ne in GGML order (ne[0] contiguous). Captured tensors: l_out-<layer>,
// result_norm and result_output (logits).

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Capture {
    FILE * out;
    std::vector<uint8_t> buf;
    int n_written = 0;
};

bool wanted(const char * name) {
    return std::strncmp(name, "l_out-", 6) == 0 || std::strcmp(name, "result_norm") == 0 ||
           std::strcmp(name, "result_output") == 0;
}

bool on_eval(ggml_tensor * t, bool ask, void * user) {
    if (ask) {
        return wanted(t->name);
    }
    if (!wanted(t->name)) {
        return true;
    }
    auto * cap = static_cast<Capture *>(user);
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        std::fprintf(stderr, "llama-dump: %s is not contiguous f32\n", t->name);
        return false;
    }
    const size_t n = ggml_nbytes(t);
    cap->buf.resize(n);
    ggml_backend_tensor_get(t, cap->buf.data(), 0, n);

    const uint32_t name_len = std::strlen(t->name);
    const uint32_t n_dims = 2;
    const int64_t ne[2] = {t->ne[0], t->ne[1] * t->ne[2] * t->ne[3]};
    std::fwrite(&name_len, sizeof name_len, 1, cap->out);
    std::fwrite(t->name, 1, name_len, cap->out);
    std::fwrite(&n_dims, sizeof n_dims, 1, cap->out);
    std::fwrite(ne, sizeof ne, 1, cap->out);
    std::fwrite(cap->buf.data(), 1, n, cap->out);
    cap->n_written += 1;
    return true;
}

std::vector<llama_token> read_tokens(const char * path) {
    std::vector<llama_token> tokens;
    FILE * f = std::fopen(path, "rb");
    if (f == nullptr) {
        return tokens;
    }
    int32_t t;
    while (std::fread(&t, sizeof t, 1, f) == 1) {
        tokens.push_back(t);
    }
    std::fclose(f);
    return tokens;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <tokens.i32> <out.bin>\n", argv[0]);
        return 2;
    }
    const std::vector<llama_token> tokens = read_tokens(argv[2]);
    if (tokens.empty()) {
        std::fprintf(stderr, "llama-dump: no tokens in %s\n", argv[2]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        return 1;
    }

    Capture cap;
    cap.out = std::fopen(argv[3], "wb");
    if (cap.out == nullptr) {
        return 1;
    }

    const uint32_t n = static_cast<uint32_t>(tokens.size());
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n;
    cparams.n_batch = n;
    cparams.n_ubatch = n;
    cparams.n_seq_max = 1;
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.cb_eval = on_eval;
    cparams.cb_eval_user_data = &cap;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        return 1;
    }

    llama_batch batch = llama_batch_init(static_cast<int32_t>(n), 0, 1);
    for (uint32_t i = 0; i < n; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = static_cast<int32_t>(n);

    const int32_t rc = llama_decode(ctx, batch);
    std::fclose(cap.out);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    if (rc != 0) {
        std::fprintf(stderr, "llama-dump: llama_decode failed: %d\n", rc);
        return 1;
    }
    std::fprintf(stderr, "llama-dump: %u tokens, %d tensors\n", n, cap.n_written);
    return 0;
}
