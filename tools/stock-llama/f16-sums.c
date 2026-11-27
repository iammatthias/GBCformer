// Whether a ggml build's CPU backend sums an f16 matrix's products in f16 or in f32.
// One row of 64 weights of 16,352 against inputs of 1: the sum is 1,046,528. A row of 32
// such weights and 32 of -16,352, four of each sign in turn, sums to 0. A backend that
// accumulates in f16 lanes gives inf for the first and NaN for the second; in f32 lanes
// both are exact. Each row runs against 1 token (ggml_vec_dot_f16) and 16 tokens
// (llamafile's tinyBLAS), with the weights stored as f16 and as f32.
// cc -O2 -I<llama.cpp>/ggml/include f16-sums.c -L<build>/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,<build>/bin -o f16-sums
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>

static float product(enum ggml_type type, int tokens, int balanced) {
    struct ggml_init_params ip = { 16 * 1024 * 1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, type, 64, 1);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, tokens);
    for (int c = 0; c < 64; c++) {
        float v = balanced && c % 8 >= 4 ? -16352.0f : 16352.0f;
        if (type == GGML_TYPE_F16) {
            ((ggml_fp16_t *) w->data)[c] = ggml_fp32_to_fp16(v);
        } else {
            ((float *) w->data)[c] = v;
        }
    }
    for (int i = 0; i < 64 * tokens; i++) {
        ((float *) x->data)[i] = 1.0f;
    }
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    struct ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);
    ggml_graph_compute_with_ctx(ctx, g, 1);
    float out = ((float *) y->data)[0];
    ggml_free(ctx);
    return out;
}

int main(void) {
    for (int balanced = 0; balanced < 2; balanced++) {
        for (int tokens = 1; tokens <= 16; tokens += 15) {
            printf("%s, %2d token%s: f16 weights %g, f32 weights %g\n", balanced ? "32 of 16352 and 32 of -16352 (sum 0)      " : "64 of 16352 (sum 1046528)", tokens, tokens == 1 ? " " : "s", product(GGML_TYPE_F16, tokens, balanced), product(GGML_TYPE_F32, tokens, balanced));
        }
    }
    return 0;
}
