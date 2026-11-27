// A few C functions over llama.h, so the Rust side never mirrors llama.cpp's structs.
// The context has an f16 KV cache (STOCK_KV=f32 for f32) and flash attention on
// (STOCK_FA=0 turns it off). The sequences share one unified KV buffer, which with flash
// attention lets each sequence of a batch attend to its own cells only (llama.cpp's
// kv_rows).

#include "llama.h"
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml-backend.h"
#include <math.h>
#include <dlfcn.h>

// STOCK_NANCHECK: every tensor a decode computes is scanned, and the first one holding a
// value that is not finite is named once (a diagnostic, off by default).
static int nan_reported = 0;
static float * last_norm = NULL;
static long long last_norm_d = 0;
static void gate_row(struct ggml_tensor * t, size_t i);
static bool nan_cb(struct ggml_tensor * t, bool ask, void * user) {
    (void) user;
    if (ask) {
        return !nan_reported && t->type == GGML_TYPE_F32;
    }
    if (nan_reported || t->type != GGML_TYPE_F32) {
        return true;
    }
    size_t n = ggml_nelements(t);
    float * buf = malloc(n * sizeof(float));
    ggml_backend_tensor_get(t, buf, 0, n * sizeof(float));
    if (strncmp(t->name, "ffn_norm", 8) == 0) {
        free(last_norm);
        last_norm = malloc(n * sizeof(float));
        memcpy(last_norm, buf, n * sizeof(float));
        last_norm_d = t->ne[0];
    }
    for (size_t i = 0; i < n && strncmp(t->name, "ffn_swiglu", 10) == 0; i++) {
        if (fabsf(buf[i]) > 65504.0f) {
            fprintf(stderr, "NANCHECK: %s holds %f beyond f16 at unit %lld, token %lld\n", t->name, buf[i], (long long) (i % t->ne[0]), (long long) (i / t->ne[0]));
            if (last_norm) {
                long long tok = (long long) (i / t->ne[0]);
                fprintf(stderr, "NANCHECK: ffn_norm row of token %lld (nonzero channels):", tok);
                for (long long c = 0; c < last_norm_d; c++) {
                    float v = last_norm[tok * last_norm_d + c];
                    if (v != 0.0f) fprintf(stderr, " %lld:%g", c, v);
                }
                fprintf(stderr, "\n");
            }
            nan_reported = 1;
            free(buf);
            return true;
        }
    }
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(buf[i])) {
            fprintf(stderr, "NANCHECK: %s (op %s, %lld x %lld x %lld) holds %f at element %zu of %zu; src0 %s, src1 %s\n", t->name, ggml_op_desc(t), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], buf[i], i, n, t->src[0] ? t->src[0]->name : "-", t->src[1] ? t->src[1]->name : "-");
            gate_row(t, i);
            nan_reported = 1;
            break;
        }
    }
    free(buf);
    return true;
}


// The products behind element `i` of a matmul of an f16 weight matrix with the last
// `ffn_norm`: the row's exact sum in f64, the sums of its positive and of its negative
// products, and its largest products. A backend that sums in f32 holds the exact sum; one
// that sums in f16 lanes (ggml's CPU backend on ARM with FP16 arithmetic) overflows when
// a lane's partial sum passes 65,504.
static void gate_row(struct ggml_tensor * t, size_t i) {
    struct ggml_tensor * w = t->src[0];
    if (!w || w->type != GGML_TYPE_F16 || !last_norm || !t->src[1] || strncmp(t->src[1]->name, "ffn_norm", 8) != 0 || w->ne[0] != last_norm_d) {
        return;
    }
    long long unit = (long long) (i % t->ne[0]), tok = (long long) (i / t->ne[0]), d = last_norm_d;
    ggml_fp16_t * row = malloc(d * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(w, row, unit * d * sizeof(ggml_fp16_t), d * sizeof(ggml_fp16_t));
    const float * x = last_norm + tok * d;
    double sum = 0.0, pos = 0.0, neg = 0.0, top_x = 0.0;
    long long terms = 0;
    for (long long c = 0; c < d; c++) {
        double p = (double) ggml_fp16_to_fp32(row[c]) * (double) x[c];
        if (fabs((double) x[c]) > top_x) top_x = fabs((double) x[c]);
        if (p != 0.0) terms++;
        sum += p;
        if (p > 0) pos += p; else neg += p;
    }
    fprintf(stderr, "NANCHECK: row of unit %lld at token %lld: %lld nonzero products, exact sum %.1f, positive products %.1f, negative products %.1f, largest input %.1f; products by channel (channel:weight*input):", unit, tok, terms, sum, pos, neg, top_x);
    for (long long c = 0; c < d; c++) {
        double wv = (double) ggml_fp16_to_fp32(row[c]);
        if (wv * (double) x[c] != 0.0) fprintf(stderr, " %lld:%g*%g", c, wv, (double) x[c]);
    }
    fprintf(stderr, "\n");
    free(row);
}

// STOCK_TENSORS=<file>: every tensor of one decode written as a line of name, op, shape,
// count of values that are not finite, largest magnitude, sum and a hash of its f32 values
// (a diagnostic, off by default). The decode is the first of more than 16 tokens after
// STOCK_TENSORS_SKIP such decodes on the stream STOCK_TENSORS_STREAM names (any stream
// when unset). Two runs' files diff to the first tensor two backends compute differently.
static FILE * dump_file = NULL;
static int dump_active = 0;
static int dump_skip = 0;
static char dump_wanted[16] = "";
static char dump_stream[16] = "";
static bool dump_cb(struct ggml_tensor * t, bool ask, void * user) {
    (void) user;
    if (ask || !dump_active) {
        return dump_active;
    }
    size_t n = ggml_nelements(t);
    fprintf(dump_file, "%s %s %s %lldx%lldx%lldx%lld", t->name[0] ? t->name : "-", ggml_op_desc(t), ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    if (t->type == GGML_TYPE_F32) {
        float * buf = malloc(n * sizeof(float));
        ggml_backend_tensor_get(t, buf, 0, n * sizeof(float));
        size_t bad = 0, first_bad = n, big = 0;
        double sum = 0.0;
        float top = 0.0f;
        unsigned long long hash = 1469598103934665603ull;
        for (size_t i = 0; i < n; i++) {
            float v = buf[i] == 0.0f ? 0.0f : buf[i];
            if (!isfinite(v)) {
                bad++;
                if (first_bad == n) first_bad = i;
            } else {
                sum += v;
                if (fabsf(v) > top) top = fabsf(v);
                if (fabsf(v) > 65504.0f) big++;
            }
            unsigned int bits;
            memcpy(&bits, &v, 4);
            hash = (hash ^ bits) * 1099511628211ull;
        }
        char first[24] = "-";
        if (first_bad != n) snprintf(first, sizeof first, "%zu", first_bad);
        fprintf(dump_file, " bad %zu first %s beyond-f16 %zu top %g sum %.6f hash %016llx", bad, first, big, top, sum, hash);
        free(buf);
    }
    fprintf(dump_file, " src %s %s\n", t->src[0] ? (t->src[0]->name[0] ? t->src[0]->name : ggml_op_desc(t->src[0])) : "-", t->src[1] ? (t->src[1]->name[0] ? t->src[1]->name : ggml_op_desc(t->src[1])) : "-");
    return true;
}

// Both diagnostics on one decode: the callback slot holds one function.
static bool nan_dump_cb(struct ggml_tensor * t, bool ask, void * user) {
    if (ask) {
        return nan_cb(t, true, user) | dump_cb(t, true, user);
    }
    nan_cb(t, false, user);
    dump_cb(t, false, user);
    return true;
}

struct shim {
    struct llama_model * model;
    struct llama_context * ctx;
    struct llama_batch batch;
    int n_batch;
};

static int quiet_level = GGML_LOG_LEVEL_WARN;

static void log_cb(enum ggml_log_level level, const char * text, void * user) {
    (void) user;
    if (level >= quiet_level && level != GGML_LOG_LEVEL_CONT) {
        fputs(text, stderr);
    }
}

// One model shared by every context opened on the same path.
static struct llama_model * shared_model = NULL;
static char shared_path[4096];
static int shared_users = 0;

struct shim * shim_open(const char * path, int n_ctx, int n_batch, int n_seq_max, int n_threads, int verbose) {
    quiet_level = verbose ? GGML_LOG_LEVEL_INFO : GGML_LOG_LEVEL_WARN;
    llama_log_set(log_cb, NULL);
    llama_backend_init();
    const char * gpu = getenv("STOCK_GPU_LAYERS");
    int n_gpu = gpu ? atoi(gpu) : 0;
    struct llama_model * model = NULL;
    if (shared_model && strcmp(shared_path, path) == 0) {
        model = shared_model;
    } else {
        struct llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = n_gpu;
        model = llama_model_load_from_file(path, mp);
        if (!model) {
            return NULL;
        }
        shared_model = model;
        snprintf(shared_path, sizeof shared_path, "%s", path);
    }
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = n_batch;
    cp.n_seq_max = n_seq_max;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    const char * kv = getenv("STOCK_KV");
    enum ggml_type t = (kv && strcmp(kv, "f32") == 0) ? GGML_TYPE_F32 : GGML_TYPE_F16;
    cp.type_k = t;
    cp.type_v = t;
    const char * fa = getenv("STOCK_FA");
    cp.flash_attn_type = (fa && strcmp(fa, "0") == 0) ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv = n_gpu > 0;
    cp.op_offload = n_gpu > 0;
    cp.kv_unified = true;
    cp.no_perf = true;
    if (getenv("STOCK_NANCHECK")) {
        cp.cb_eval = nan_cb;
        cp.cb_eval_user_data = NULL;
    }
    if (getenv("STOCK_TENSORS") && !dump_file) {
        dump_file = fopen(getenv("STOCK_TENSORS"), "w");
        if (!dump_file) {
            fprintf(stderr, "STOCK_TENSORS: cannot write %s\n", getenv("STOCK_TENSORS"));
            exit(1);
        }
        dump_skip = getenv("STOCK_TENSORS_SKIP") ? atoi(getenv("STOCK_TENSORS_SKIP")) : 0;
        snprintf(dump_wanted, sizeof dump_wanted, "%s", getenv("STOCK_TENSORS_STREAM") ? getenv("STOCK_TENSORS_STREAM") : "");
        cp.cb_eval = getenv("STOCK_NANCHECK") ? nan_dump_cb : dump_cb;
        cp.cb_eval_user_data = NULL;
    }
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return NULL;
    }
    shared_users++;
    struct shim * s = calloc(1, sizeof(struct shim));
    s->model = model;
    s->ctx = ctx;
    s->n_batch = n_batch;
    s->batch = llama_batch_init(n_batch, 0, 2);
    return s;
}

int shim_n_vocab(struct shim * s) {
    return llama_vocab_n_tokens(llama_model_get_vocab(s->model));
}

// Decodes `n` tokens. Token i sits at pos[i] in sequence seq[i], and in sequence also[i]
// too unless that is negative; logits[i] asks for its output row. Returns llama_decode's
// status. llama_decode copies a token's sequences into an unordered_set and masks its
// attention by the first the set yields, which for {0, 1} under libc++ is the one listed
// second (Runtime::split_pass relies on it, and a flip would fail every check).
int shim_decode(struct shim * s, int n, const int * tokens, const int * pos, const int * seq, const int * also, const signed char * logits) {
    if (n > s->n_batch) {
        return -100;
    }
    struct llama_batch * b = &s->batch;
    b->n_tokens = n;
    for (int i = 0; i < n; i++) {
        b->token[i] = tokens[i];
        b->pos[i] = pos[i];
        b->seq_id[i][0] = seq[i];
        b->seq_id[i][1] = also[i];
        b->n_seq_id[i] = also[i] < 0 ? 1 : 2;
        b->logits[i] = logits[i];
    }
    if (dump_file && n > 16 && (!dump_wanted[0] || strcmp(dump_wanted, dump_stream) == 0)) {
        dump_active = dump_skip-- == 0;
    }
    int status = llama_decode(s->ctx, *b);
    if (dump_active) {
        llama_synchronize(s->ctx);
        fflush(dump_file);
        dump_active = 0;
    }
    return status;
}

// Names the stream of an overlay checkpoint that the next decodes run, as the GGUF's
// `gbcformer.stream.<name>.*` keys slice it, or the whole model with NULL. Returns 1 when
// the runtime runs that slice: it carries `stream-slices.patch` and the GGUF names the
// stream. A runtime without the patch runs the whole model and returns 0.
int shim_stream(const char * name) {
    static int (*set)(const char *) = NULL;
    static int looked = 0;
    if (!looked) {
        set = (int (*)(const char *)) dlsym(RTLD_DEFAULT, "llama_gbcformer_set_stream");
        looked = 1;
    }
    snprintf(dump_stream, sizeof dump_stream, "%s", name ? name : "");
    return set ? set(name) >= 0 : 0;
}

// The most tokens of a batch that `stream-slices.patch` runs on the whole model whatever
// stream is named, or -1 from a runtime that does not export it.
int shim_fence(void) {
    int (*fence)(void) = (int (*)(void)) dlsym(RTLD_DEFAULT, "llama_gbcformer_fence");
    return fence ? fence() : -1;
}

// Whether the last decode's logit rows ran a stream's slice (1) or the whole model (0), as
// the graph that computed them reports it (`stored-runs.patch`), or -1 from a runtime
// that cannot say.
int shim_sliced(struct shim * s) {
    static int32_t (*ran)(struct llama_context *) = NULL;
    static int looked = 0;
    if (!looked) {
        ran = (int32_t (*)(struct llama_context *)) dlsym(RTLD_DEFAULT, "llama_get_logits_stream");
        looked = 1;
    }
    return ran ? ran(s->ctx) >= 0 : -1;
}


// Which of the four functions this tree's patches export the loaded runtime has, one bit
// each in the order of `PATCH_SYMBOLS` (src/cpp.rs): 0 from unpatched llama.cpp.
int shim_patch_symbols(void) {
    static const char * names[] = {"llama_gbcformer_set_stream", "llama_gbcformer_fence", "llama_get_logits_stream", "llama_get_logits_cols"};
    int found = 0;
    for (int i = 0; i < 4; i++) {
        found |= (dlsym(RTLD_DEFAULT, names[i]) != NULL) << i;
    }
    return found;
}

// The file the process loaded `llama_decode` from, or "" when the loader cannot say.
const char * shim_lib(void) {
    Dl_info info;
    return dladdr((void *) llama_decode, &info) && info.dli_fname ? info.dli_fname : "";
}

const float * shim_logits(struct shim * s, int i) {
    return llama_get_logits_ith(s->ctx, i);
}

// The width of the last decode's logit rows, and in *first the token id their first
// column scores. A runtime with `stored-runs.patch` scores only the rows of the LM head
// the GGUF stores, or the run its stream can decode; any other scores every token from 0.
int shim_logits_cols(struct shim * s, int * first) {
    static int32_t (*cols)(struct llama_context *, int32_t *) = NULL;
    static int looked = 0;
    if (!looked) {
        cols = (int32_t (*)(struct llama_context *, int32_t *)) dlsym(RTLD_DEFAULT, "llama_get_logits_cols");
        looked = 1;
    }
    if (!cols) {
        *first = 0;
        return shim_n_vocab(s);
    }
    int32_t f = 0;
    int32_t n = cols(s->ctx, &f);
    *first = f;
    return n;
}

int shim_seq_rm(struct shim * s, int seq, int p0, int p1) {
    return llama_memory_seq_rm(llama_get_memory(s->ctx), seq, p0, p1);
}

void shim_seq_add(struct shim * s, int seq, int p0, int p1, int delta) {
    llama_memory_seq_add(llama_get_memory(s->ctx), seq, p0, p1, delta);
}

void shim_seq_cp(struct shim * s, int src, int dst, int p0, int p1) {
    llama_memory_seq_cp(llama_get_memory(s->ctx), src, dst, p0, p1);
}

int shim_seq_pos_max(struct shim * s, int seq) {
    return llama_memory_seq_pos_max(llama_get_memory(s->ctx), seq);
}

int shim_seq_pos_min(struct shim * s, int seq) {
    return llama_memory_seq_pos_min(llama_get_memory(s->ctx), seq);
}

// Saves sequence `seq`'s cells (positions, keys and values) to a file, and loads them
// back. Each returns 0 on failure.
int shim_seq_save(struct shim * s, const char * path, int seq) {
    return llama_state_seq_save_file(s->ctx, path, seq, NULL, 0) != 0;
}

int shim_seq_load(struct shim * s, const char * path, int seq) {
    llama_token none[1];
    size_t n = 0;
    return llama_state_seq_load_file(s->ctx, path, seq, none, 1, &n) != 0;
}

// Saves every sequence of the context (each cell with the sequences it is in), and loads
// them back into a context opened the same way. Each returns 0 on failure.
int shim_state_save(struct shim * s, const char * path) {
    return llama_state_save_file(s->ctx, path, NULL, 0) ? 1 : 0;
}

int shim_state_load(struct shim * s, const char * path) {
    llama_token none[1];
    size_t n = 0;
    return llama_state_load_file(s->ctx, path, none, 1, &n) ? 1 : 0;
}

void shim_close(struct shim * s) {
    llama_batch_free(s->batch);
    llama_free(s->ctx);
    if (--shared_users == 0) {
        llama_model_free(shared_model);
        shared_model = NULL;
    }
    free(s);
}
