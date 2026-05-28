#include "ds4_gpu.h"
#include "ds4_source_formats.h"
#include "engine/layer_execute.h"
#include "engine/layer_state.h"

#include <errno.h>
#include <float.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures;

typedef struct {
    const unsigned char *ptr;
    uint64_t size;
    int fd;
} model_map;

typedef ds4_v100_bound_matrix bound_matrix;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_v100_integrated_layer_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/cuda_v100_integrated_layer_smoke --index FILE --model FILE [--layer N] [--router-token N] [--position N]\n");
}

static int parse_int_arg(const char *s, const char *name, int max_v) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || !*s || !end || *end != '\0' || v < 0 || v > max_v) {
        fprintf(stderr, "cuda_v100_integrated_layer_smoke: invalid %s: %s\n",
                name,
                s ? s : "(null)");
        exit(2);
    }
    return (int)v;
}

static int map_model_file(const char *path, model_map *out) {
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "cuda_v100_integrated_layer_smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "cuda_v100_integrated_layer_smoke: cannot stat %s\n", path);
        close(fd);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "cuda_v100_integrated_layer_smoke: cannot mmap %s: %s\n",
                path,
                strerror(errno));
        close(fd);
        return 1;
    }
    out->ptr = (const unsigned char *)p;
    out->size = (uint64_t)st.st_size;
    out->fd = fd;
    return 0;
}

static void unmap_model_file(model_map *m) {
    if (!m) return;
    if (m->ptr) munmap((void *)m->ptr, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

static void fill_hidden(float *hidden, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        int a = (int)(i % 43u) - 21;
        int b = (int)((i * 23u) % 37u) - 18;
        hidden[i] = ((float)a * 0.00029f) + ((float)b * 0.00007f);
    }
}

static const uint8_t *matrix_host_ptr(const model_map *model, const bound_matrix *m) {
    const ds4_v100_tensor_binding *b = &m->binding;
    if (b->source_offset + m->rel > model->size ||
        m->bytes > model->size - b->source_offset - m->rel) {
        return NULL;
    }
    return model->ptr + b->source_offset + m->rel;
}

static int upload_matrix(ds4_gpu_arena *arena,
                         const model_map *model,
                         const bound_matrix *m,
                         unsigned char **scratch,
                         uint64_t *scratch_bytes) {
    const uint8_t *src = matrix_host_ptr(model, m);
    if (!src) return 1;
    if (*scratch_bytes < m->bytes) {
        unsigned char *p = (unsigned char *)realloc(*scratch, (size_t)m->bytes);
        if (!p) return 1;
        *scratch = p;
        *scratch_bytes = m->bytes;
    }
    memcpy(*scratch, src, (size_t)m->bytes);
    return ds4_gpu_arena_upload(arena,
                                ds4_v100_bound_matrix_arena_offset(m),
                                *scratch,
                                m->bytes);
}

static void cpu_f8_matmul_rows(const bound_matrix *m,
                               const model_map *model,
                               uint32_t first_row,
                               uint32_t rows,
                               const float *x,
                               float *out) {
    const uint8_t *base = matrix_host_ptr(model, m);
    char err[128];
    if (!base || first_row > m->rows || rows > m->rows - first_row) {
        failures++;
        return;
    }
    for (uint32_t r = 0; r < rows; r++) {
        err[0] = '\0';
        const uint8_t *row = base + (uint64_t)(first_row + r) * m->row_bytes;
        if (ds4_src_f8_e4m3_b128_row_dot(&out[r], row, x, m->cols,
                                         err, sizeof(err)) != 0) {
            fprintf(stderr, "cuda_v100_integrated_layer_smoke: %s\n", err);
            failures++;
            out[r] = 0.0f;
        }
    }
}

static void cpu_f8_matmul(const bound_matrix *m,
                          const model_map *model,
                          const float *x,
                          float *out) {
    cpu_f8_matmul_rows(m, model, 0, m->rows, x, out);
}

static void cpu_mxfp4_matmul(const bound_matrix *m,
                             const model_map *model,
                             const float *x,
                             float *out) {
    const uint8_t *base = matrix_host_ptr(model, m);
    char err[128];
    if (!base) {
        failures++;
        return;
    }
    for (uint32_t r = 0; r < m->rows; r++) {
        err[0] = '\0';
        if (ds4_src_mxfp4_row_dot(&out[r],
                                  base + (uint64_t)r * m->row_bytes,
                                  x,
                                  m->cols,
                                  err,
                                  sizeof(err)) != 0) {
            fprintf(stderr, "cuda_v100_integrated_layer_smoke: %s\n", err);
            failures++;
            out[r] = 0.0f;
        }
    }
}

static void cpu_f32_matmul(const bound_matrix *m,
                           const model_map *model,
                           const float *x,
                           float *out) {
    const float *base = (const float *)(const void *)matrix_host_ptr(model, m);
    if (!base) {
        failures++;
        return;
    }
    for (uint32_t r = 0; r < m->rows; r++) {
        const float *row = (const float *)(const void *)((const uint8_t *)base + (uint64_t)r * m->row_bytes);
        double acc = 0.0;
        for (uint32_t c = 0; c < m->cols; c++) acc += (double)row[c] * (double)x[c];
        out[r] = (float)acc;
    }
}

static float softplus_ref(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static float swiglu_ref(float gate, float up, float clamp, float weight) {
    if (clamp > 1.0e-6f) {
        if (gate > clamp) gate = clamp;
        if (up > clamp) up = clamp;
        if (up < -clamp) up = -clamp;
    }
    return (gate / (1.0f + expf(-gate))) * up * weight;
}

static int router_score_better_ref(float a_score, uint32_t a_idx,
                                   float b_score, uint32_t b_idx) {
    return a_score > b_score || (a_score == b_score && a_idx < b_idx);
}

static void cpu_router_select(const ds4_v100_layer_state *state,
                              const model_map *model,
                              const float logits[256],
                              uint32_t token,
                              int32_t selected[6],
                              float weights[6]) {
    float probs[256];
    for (uint32_t e = 0; e < 256; e++) probs[e] = sqrtf(softplus_ref(logits[e]));
    if (state->router_kind == DS4_V100_ROUTER_HASH && state->has_hash_router) {
        const ds4_v100_tensor_binding *hash_b = &state->router_hash;
        if (hash_b->n_shape_dims != 2 || hash_b->shape[0] != 6 ||
            token >= hash_b->shape[1] ||
            hash_b->source_offset > model->size ||
            hash_b->byte_length > model->size - hash_b->source_offset) {
            failures++;
            return;
        }
        const int32_t *hash = (const int32_t *)(const void *)(model->ptr + hash_b->source_offset);
        const int32_t *row = hash + (uint64_t)token * 6u;
        for (uint32_t i = 0; i < 6; i++) selected[i] = row[i];
    } else if (state->router_kind == DS4_V100_ROUTER_BIAS && state->has_bias_router) {
        const ds4_v100_tensor_binding *bias_b = &state->router_bias;
        if (bias_b->n_shape_dims != 1 || bias_b->shape[0] != 256 ||
            bias_b->source_offset > model->size ||
            bias_b->byte_length > model->size - bias_b->source_offset) {
            failures++;
            return;
        }
        const float *bias = (const float *)(const void *)(model->ptr + bias_b->source_offset);
        float best_scores[6];
        for (uint32_t i = 0; i < 6; i++) {
            selected[i] = -1;
            best_scores[i] = -FLT_MAX;
        }
        for (uint32_t e = 0; e < 256; e++) {
            const float score = probs[e] + bias[e];
            for (uint32_t k = 0; k < 6; k++) {
                if (selected[k] < 0 ||
                    router_score_better_ref(score,
                                            e,
                                            best_scores[k],
                                            (uint32_t)selected[k])) {
                    for (uint32_t j = 5; j > k; j--) {
                        selected[j] = selected[j - 1];
                        best_scores[j] = best_scores[j - 1];
                    }
                    selected[k] = (int32_t)e;
                    best_scores[k] = score;
                    break;
                }
            }
        }
    } else {
        failures++;
        return;
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < 6; i++) {
        const int32_t e = selected[i];
        weights[i] = (e >= 0 && e < 256) ? probs[(uint32_t)e] : 0.0f;
        sum += weights[i];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t i = 0; i < 6; i++) weights[i] = weights[i] / sum * 1.5f;
}

static void cpu_rms_norm_weight(float *out,
                                const float *x,
                                const model_map *model,
                                const ds4_v100_tensor_binding *weight,
                                uint32_t n,
                                float eps) {
    if (!weight || weight->source_offset > model->size ||
        (uint64_t)n * sizeof(float) > model->size - weight->source_offset) {
        failures++;
        return;
    }
    const float *w = (const float *)(const void *)(model->ptr + weight->source_offset);
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint32_t i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

static void cpu_head_rms_norm(float *x, uint32_t n_head, uint32_t head_dim) {
    for (uint32_t h = 0; h < n_head; h++) {
        float *head = x + (uint64_t)h * head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; i++) ss += (double)head[i] * (double)head[i];
        const float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + DS4_V100_RMS_EPS);
        for (uint32_t i = 0; i < head_dim; i++) head[i] *= scale;
    }
}

static uint32_t state_ratio(const ds4_v100_layer_state *state) {
    if (state->layer_class == DS4_V100_LAYER_RATIO_4) return 4u;
    if (state->layer_class == DS4_V100_LAYER_RATIO_128) return 128u;
    return 0u;
}

static float rope_yarn_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static float rope_yarn_corr_dim(int n_dims, uint64_t n_ctx_orig, float n_rot, float base) {
    const float pi = 3.14159265358979323846f;
    return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * pi)) /
           (2.0f * logf(base));
}

static void rope_yarn_corr_dims(int n_dims,
                                uint64_t n_ctx_orig,
                                float freq_base,
                                float beta_fast,
                                float beta_slow,
                                float dims[2]) {
    const float start = floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    const float end = ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = fmaxf(0.0f, start);
    dims[1] = fminf((float)(n_dims - 1), end);
}

static void cpu_rope_tail(float *x,
                          uint32_t n_head,
                          uint32_t head_dim,
                          uint32_t n_rot,
                          uint32_t pos,
                          const ds4_v100_layer_state *state,
                          bool inverse) {
    const uint32_t n_nope = head_dim - n_rot;
    const bool compressed = state_ratio(state) != 0;
    const float freq_base = compressed ? 160000.0f : 10000.0f;
    const float freq_scale = compressed ? (1.0f / 16.0f) : 1.0f;
    const float ext_factor = compressed ? 1.0f : 0.0f;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const float sin_sign = inverse ? -1.0f : 1.0f;
    float attn_factor = 1.0f;
    float corr_dims[2] = {0.0f, 0.0f};
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
        rope_yarn_corr_dims((int)n_rot, 65536u, freq_base, 32.0f, 1.0f, corr_dims);
    }

    for (uint32_t h = 0; h < n_head; h++) {
        float *tail = x + (uint64_t)h * head_dim + n_nope;
        float theta_extrap = (float)pos;
        for (uint32_t i = 0; i < n_rot; i += 2) {
            const float theta_interp = freq_scale * theta_extrap;
            float theta = theta_interp;
            float mscale = attn_factor;
            if (ext_factor != 0.0f) {
                const float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)i) * ext_factor;
                theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }
            const float c = cosf(theta) * mscale;
            const float s = sin_sign * sinf(theta) * mscale;
            const float x0 = tail[i + 0];
            const float x1 = tail[i + 1];
            tail[i + 0] = x0 * c - x1 * s;
            tail[i + 1] = x0 * s + x1 * c;
            theta_extrap *= theta_scale;
        }
    }
}

static void cpu_attention_decode(float *heads,
                                 const model_map *model,
                                 const ds4_v100_layer_state *state,
                                 const float *q,
                                 const float *raw_kv,
                                 uint32_t n_raw,
                                 const float *comp_kv,
                                 const float *comp_mask,
                                 uint32_t n_comp) {
    const float *sinks = (const float *)(const void *)(model->ptr + state->attn_sinks.source_offset);
    const float scale = 1.0f / sqrtf((float)DS4_V100_HEAD_DIM);
    for (uint32_t h = 0; h < DS4_V100_N_HEAD; h++) {
        const float *qh = q + (uint64_t)h * DS4_V100_HEAD_DIM;
        float scores[16];
        float max_s = sinks[h];
        uint32_t idx = 0;
        for (uint32_t r = 0; r < n_raw; r++, idx++) {
            const float *kv = raw_kv + (uint64_t)r * DS4_V100_HEAD_DIM;
            double dot = 0.0;
            for (uint32_t d = 0; d < DS4_V100_HEAD_DIM; d++) dot += (double)qh[d] * (double)kv[d];
            scores[idx] = (float)dot * scale;
            if (scores[idx] > max_s) max_s = scores[idx];
        }
        for (uint32_t c = 0; c < n_comp; c++, idx++) {
            const float add = comp_mask ? comp_mask[c] : 0.0f;
            if (add <= -1.0e20f) {
                scores[idx] = -1.0e30f;
            } else {
                const float *kv = comp_kv + (uint64_t)c * DS4_V100_HEAD_DIM;
                double dot = 0.0;
                for (uint32_t d = 0; d < DS4_V100_HEAD_DIM; d++) dot += (double)qh[d] * (double)kv[d];
                scores[idx] = (float)dot * scale + add;
                if (scores[idx] > max_s) max_s = scores[idx];
            }
        }

        float denom = expf(sinks[h] - max_s);
        for (uint32_t i = 0; i < n_raw + n_comp; i++) {
            scores[i] = expf(scores[i] - max_s);
            denom += scores[i];
        }
        float *oh = heads + (uint64_t)h * DS4_V100_HEAD_DIM;
        for (uint32_t d = 0; d < DS4_V100_HEAD_DIM; d++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < n_raw; r++) {
                acc += raw_kv[(uint64_t)r * DS4_V100_HEAD_DIM + d] * scores[r];
            }
            for (uint32_t c = 0; c < n_comp; c++) {
                acc += comp_kv[(uint64_t)c * DS4_V100_HEAD_DIM + d] * scores[n_raw + c];
            }
            oh[d] = acc / denom;
        }
    }
}

static void cpu_grouped_attention_output(float *out,
                                         const model_map *model,
                                         const ds4_v100_layer_state *state,
                                         const float *heads,
                                         float *low) {
    memset(low, 0, (uint64_t)state->attention_output_rank * sizeof(float));
    for (uint32_t g = 0; g < DS4_V100_OUT_GROUPS; g++) {
        cpu_f8_matmul_rows(&state->attn_output_a,
                           model,
                           g * DS4_V100_OUT_GROUP_RANK,
                           DS4_V100_OUT_GROUP_RANK,
                           heads + (uint64_t)g * DS4_V100_OUT_GROUP_DIM,
                           low + (uint64_t)g * DS4_V100_OUT_GROUP_RANK);
    }
    cpu_f8_matmul(&state->attn_output_b, model, low, out);
}

static void cpu_swiglu(float *out,
                       const float *gate,
                       const float *up,
                       uint32_t n,
                       float weight) {
    for (uint32_t i = 0; i < n; i++) out[i] = swiglu_ref(gate[i], up[i], 10.0f, weight);
}

static void cpu_add(float *out, const float *a, const float *b, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static void expect_close_vector(const float *got,
                                const float *want,
                                uint32_t n,
                                const char *label,
                                float base_tol,
                                float rel_tol) {
    float max_abs = 0.0f;
    uint32_t max_i = 0;
    for (uint32_t i = 0; i < n; i++) {
        float d = fabsf(got[i] - want[i]);
        if (d > max_abs) {
            max_abs = d;
            max_i = i;
        }
    }
    const float tol = base_tol + rel_tol * fabsf(want[max_i]);
    if (max_abs > tol) {
        fprintf(stderr,
                "cuda_v100_integrated_layer_smoke: %s max_abs %.8g at %u got %.8g expected %.8g tol %.8g\n",
                label,
                max_abs,
                max_i,
                got[max_i],
                want[max_i],
                tol);
        failures++;
    }
}

static void expect_finite_nonzero_vector(const float *got,
                                         uint32_t n,
                                         const char *label) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) {
            fprintf(stderr,
                    "cuda_v100_integrated_layer_smoke: %s non-finite at %u: %.8g\n",
                    label,
                    i,
                    got[i]);
            failures++;
            return;
        }
        ss += (double)got[i] * (double)got[i];
    }
    if (ss <= 0.0) {
        fprintf(stderr, "cuda_v100_integrated_layer_smoke: %s is all zero\n", label);
        failures++;
    }
}

int main(int argc, char **argv) {
    const char *index = NULL;
    const char *model_path = NULL;
    int layer = 2;
    int router_token = 16;
    int position = 16;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(argv[i], "--index") && i + 1 < argc) {
            index = argv[++i];
        } else if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "--layer") && i + 1 < argc) {
            layer = parse_int_arg(argv[++i], "--layer", 42);
        } else if (!strcmp(argv[i], "--router-token") && i + 1 < argc) {
            router_token = parse_int_arg(argv[++i], "--router-token", 129279);
        } else if (!strcmp(argv[i], "--position") && i + 1 < argc) {
            position = parse_int_arg(argv[++i], "--position", 1048575);
        } else {
            usage(stderr);
            return 2;
        }
    }
    if (!index || !model_path) {
        usage(stderr);
        return 2;
    }

    int devices = ds4_gpu_device_count();
    if (devices < 1) {
        fprintf(stderr, "cuda_v100_integrated_layer_smoke: no CUDA devices visible\n");
        return 1;
    }

    model_map model;
    if (map_model_file(model_path, &model)) return 1;
    check(ds4_gpu_set_model_fd(model.fd), "model fd");

    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    opts.pack_index_path = index;
    opts.kv_ctx_tokens = 1048576;
    opts.kv_active_slots = 1;

    char err[512] = {0};
    ds4_v100_context *ctx = NULL;
    if (ds4_v100_context_open(&ctx, &opts, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_integrated_layer_smoke: %s\n", err);
        unmap_model_file(&model);
        return 1;
    }
    ds4_v100_layer_state state;
    if (ds4_v100_layer_state_init(&state, ctx, layer, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_integrated_layer_smoke: %s\n", err);
        ds4_v100_context_close(ctx);
        unmap_model_file(&model);
        return 1;
    }
    if (!((state.router_kind == DS4_V100_ROUTER_HASH && state.has_hash_router) ||
          (state.router_kind == DS4_V100_ROUTER_BIAS && state.has_bias_router))) {
        fprintf(stderr, "cuda_v100_integrated_layer_smoke: layer %d has unsupported router metadata\n",
                layer);
        ds4_v100_context_close(ctx);
        unmap_model_file(&model);
        return 1;
    }

    const uint32_t hidden = state.hidden_size;
    const uint32_t q_rank = state.q_lora_rank;
    const uint32_t q_width = state.q_width;
    const uint32_t kv_width = state.kv_latent_width;
    const uint32_t out_rank = state.attention_output_rank;
    const uint32_t mid = state.intermediate_size;
    const uint32_t n_raw = 3;
    const uint32_t n_comp = 3;

    float *hidden_x = (float *)calloc(hidden, sizeof(float));
    float *attn_norm = (float *)calloc(hidden, sizeof(float));
    float *q_a = (float *)calloc(q_rank, sizeof(float));
    float *q_a_norm = (float *)calloc(q_rank, sizeof(float));
    float *q = (float *)calloc(q_width, sizeof(float));
    float *kv_raw = (float *)calloc(kv_width, sizeof(float));
    float *kv = (float *)calloc(kv_width, sizeof(float));
    float *raw_kv = (float *)calloc((size_t)n_raw * kv_width, sizeof(float));
    float *comp_kv = (float *)calloc((size_t)n_comp * kv_width, sizeof(float));
    float comp_mask[3] = {0.0f, -1.0e30f, 0.0f};
    float *heads = (float *)calloc(q_width, sizeof(float));
    float *low = (float *)calloc(out_rank, sizeof(float));
    float *attn_out = (float *)calloc(hidden, sizeof(float));
    float *residual = (float *)calloc(hidden, sizeof(float));
    float *ffn_norm = (float *)calloc(hidden, sizeof(float));
    float logits[256];
    int32_t selected[6] = {0};
    float weights[6] = {0};
    float *r_gate = (float *)calloc(mid, sizeof(float));
    float *r_up = (float *)calloc(mid, sizeof(float));
    float *r_mid = (float *)calloc(mid, sizeof(float));
    float *r_out = (float *)calloc(hidden, sizeof(float));
    float *s_gate = (float *)calloc(mid, sizeof(float));
    float *s_up = (float *)calloc(mid, sizeof(float));
    float *s_mid = (float *)calloc(mid, sizeof(float));
    float *s_out = (float *)calloc(hidden, sizeof(float));
    float *ffn_delta = (float *)calloc(hidden, sizeof(float));
    float *next_cpu = (float *)calloc(hidden, sizeof(float));
    float *next_gpu = (float *)calloc(hidden, sizeof(float));
    float *hidden_hc = (float *)calloc((size_t)DS4_V100_N_HC * hidden, sizeof(float));
    float *next_hc_gpu = (float *)calloc((size_t)DS4_V100_N_HC * hidden, sizeof(float));
    check(hidden_x && attn_norm && q_a && q_a_norm && q && kv_raw && kv &&
              raw_kv && comp_kv && heads && low && attn_out && residual &&
              ffn_norm && r_gate && r_up && r_mid && r_out && s_gate &&
              s_up && s_mid && s_out && ffn_delta && next_cpu && next_gpu &&
              hidden_hc && next_hc_gpu,
          "host buffer allocation");

    bound_matrix gate_routes[6];
    bound_matrix up_routes[6];
    bound_matrix down_routes[6];
    if (!failures) {
        fill_hidden(hidden_x, hidden);
        for (uint32_t h = 0; h < DS4_V100_N_HC; h++) {
            memcpy(hidden_hc + (uint64_t)h * hidden,
                   hidden_x,
                   (size_t)hidden * sizeof(hidden_x[0]));
        }
        cpu_rms_norm_weight(attn_norm, hidden_x, &model, &state.attn_norm, hidden, DS4_V100_RMS_EPS);
        cpu_f8_matmul(&state.attn_q_a, &model, attn_norm, q_a);
        cpu_rms_norm_weight(q_a_norm, q_a, &model, &state.attn_q_a_norm, q_rank, DS4_V100_RMS_EPS);
        cpu_f8_matmul(&state.attn_q_b, &model, q_a_norm, q);
        cpu_head_rms_norm(q, DS4_V100_N_HEAD, DS4_V100_HEAD_DIM);
        cpu_rope_tail(q, DS4_V100_N_HEAD, DS4_V100_HEAD_DIM, DS4_V100_N_ROT,
                      (uint32_t)position, &state, false);
        cpu_f8_matmul(&state.attn_kv_latent, &model, attn_norm, kv_raw);
        cpu_rms_norm_weight(kv, kv_raw, &model, &state.attn_kv_a_norm, kv_width, DS4_V100_RMS_EPS);
        cpu_rope_tail(kv, 1, DS4_V100_HEAD_DIM, DS4_V100_N_ROT,
                      (uint32_t)position, &state, false);

        for (uint32_t d = 0; d < kv_width; d++) {
            const float pattern = ((float)((int)(d % 29u) - 14)) * 0.00013f;
            raw_kv[d] = kv[d] * 0.25f + pattern;
            raw_kv[(uint64_t)kv_width + d] = kv[d] * 0.5f - pattern;
            raw_kv[(uint64_t)2u * kv_width + d] = kv[d];
            comp_kv[d] = kv[d] * 0.75f + pattern * 0.5f;
            comp_kv[(uint64_t)kv_width + d] = raw_kv[d] - 0.5f * raw_kv[(uint64_t)kv_width + d];
            comp_kv[(uint64_t)2u * kv_width + d] = kv[d] * -0.25f + pattern;
        }

        cpu_attention_decode(heads, &model, &state, q, raw_kv, n_raw, comp_kv, comp_mask, n_comp);
        cpu_rope_tail(heads, DS4_V100_N_HEAD, DS4_V100_HEAD_DIM, DS4_V100_N_ROT,
                      (uint32_t)position, &state, true);
        cpu_grouped_attention_output(attn_out, &model, &state, heads, low);
        cpu_add(residual, hidden_x, attn_out, hidden);
        cpu_rms_norm_weight(ffn_norm, residual, &model, &state.ffn_norm, hidden, DS4_V100_RMS_EPS);

        cpu_f32_matmul(&state.router, &model, ffn_norm, logits);
        cpu_router_select(&state, &model, logits, (uint32_t)router_token, selected, weights);
        for (uint32_t route = 0; route < 6; route++) {
            check(selected[route] >= 0 && selected[route] < 256, "CPU selected invalid expert");
            ds4_v100_route_matrices route_views;
            check(ds4_v100_layer_state_route_matrices(&state,
                                                      (uint32_t)selected[route],
                                                      &route_views,
                                                      err,
                                                      sizeof(err)) == 0,
                  "route matrices");
            gate_routes[route] = route_views.gate;
            up_routes[route] = route_views.up;
            down_routes[route] = route_views.down;
        }

        for (uint32_t route = 0; route < 6; route++) {
            cpu_mxfp4_matmul(&gate_routes[route], &model, ffn_norm, r_gate);
            cpu_mxfp4_matmul(&up_routes[route], &model, ffn_norm, r_up);
            cpu_swiglu(r_mid, r_gate, r_up, mid, weights[route]);
            cpu_mxfp4_matmul(&down_routes[route], &model, r_mid, r_out);
            for (uint32_t i = 0; i < hidden; i++) ffn_delta[i] += r_out[i];
        }
        cpu_f8_matmul(&state.shared_gate, &model, ffn_norm, s_gate);
        cpu_f8_matmul(&state.shared_up, &model, ffn_norm, s_up);
        cpu_swiglu(s_mid, s_gate, s_up, mid, 1.0f);
        cpu_f8_matmul(&state.shared_down, &model, s_mid, s_out);
        for (uint32_t i = 0; i < hidden; i++) {
            ffn_delta[i] += s_out[i];
            next_cpu[i] = residual[i] + ffn_delta[i];
        }
    }

    uint64_t attention_span = 0;
    uint64_t ffn_span = 0;
    check(ds4_v100_layer_state_attention_arena_span(&state,
                                                    &attention_span,
                                                    err,
                                                    sizeof(err)) == 0,
          "attention arena span");
    check(ds4_v100_layer_state_ffn_arena_span(&state,
                                              selected,
                                              6,
                                              &ffn_span,
                                              err,
                                              sizeof(err)) == 0,
          "ffn arena span");
    const uint64_t arena_bytes = attention_span > ffn_span ? attention_span : ffn_span;

    ds4_gpu_arena *arena = NULL;
    check(ds4_gpu_arena_open(&arena, state.owning_gpu, arena_bytes) == 0, "arena open");
    check(arena && ds4_gpu_arena_is_device_memory(arena), "arena is device memory");
    unsigned char *scratch = NULL;
    uint64_t scratch_bytes = 0;
    if (arena && !failures) {
        const bound_matrix *fixed[] = {
            &state.attn_q_a,
            &state.attn_q_b,
            &state.attn_kv_latent,
            &state.attn_output_a,
            &state.attn_output_b,
            &state.router,
            &state.shared_gate,
            &state.shared_up,
            &state.shared_down,
        };
        for (uint32_t i = 0; i < sizeof(fixed) / sizeof(fixed[0]); i++) {
            check(upload_matrix(arena, &model, fixed[i], &scratch, &scratch_bytes) == 0,
                  "upload fixed matrix");
        }
        if (state.has_attention_compressor) {
            const bound_matrix *compressor[] = {
                &state.attn_compressor_kv,
                &state.attn_compressor_gate,
            };
            for (uint32_t i = 0; i < sizeof(compressor) / sizeof(compressor[0]); i++) {
                check(upload_matrix(arena, &model, compressor[i], &scratch, &scratch_bytes) == 0,
                      "upload attention compressor matrix");
            }
        }
        if (state.has_indexer) {
            const bound_matrix *indexer[] = {
                &state.indexer_attn_q_b,
                &state.indexer_proj,
                &state.indexer_compressor_kv,
                &state.indexer_compressor_gate,
            };
            for (uint32_t i = 0; i < sizeof(indexer) / sizeof(indexer[0]); i++) {
                check(upload_matrix(arena, &model, indexer[i], &scratch, &scratch_bytes) == 0,
                      "upload indexer matrix");
            }
        }
        for (uint32_t route = 0; route < 6; route++) {
            const bound_matrix *routed[] = {&gate_routes[route], &up_routes[route], &down_routes[route]};
            for (uint32_t i = 0; i < sizeof(routed) / sizeof(routed[0]); i++) {
                check(upload_matrix(arena, &model, routed[i], &scratch, &scratch_bytes) == 0,
                      "upload routed matrix");
            }
        }
    }

    ds4_gpu_tensor *hidden_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *raw_t = ds4_gpu_tensor_alloc((uint64_t)n_raw * kv_width * sizeof(float));
    ds4_gpu_tensor *comp_t = ds4_gpu_tensor_alloc((uint64_t)n_comp * kv_width * sizeof(float));
    ds4_gpu_tensor *mask_t = ds4_gpu_tensor_alloc((uint64_t)n_comp * sizeof(float));
    ds4_gpu_tensor *next_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *hidden_hc_t = ds4_gpu_tensor_alloc((uint64_t)DS4_V100_N_HC * hidden * sizeof(float));
    ds4_gpu_tensor *next_hc_t = ds4_gpu_tensor_alloc((uint64_t)DS4_V100_N_HC * hidden * sizeof(float));
    const uint32_t cache_raw_cap = 128u;
    const uint32_t cache_comp_cap = 4u;
    const uint32_t cache_ratio = state.compress_ratio;
    const uint32_t cache_coff = cache_ratio == 4u ? 2u : 1u;
    const uint32_t attn_state_rows = cache_ratio ? cache_coff * cache_ratio : 0u;
    const uint32_t attn_state_width = cache_coff * DS4_V100_HEAD_DIM;
    const uint32_t index_state_rows = state.has_indexer ? 2u * cache_ratio : 0u;
    const uint32_t index_state_width = 2u * DS4_V100_INDEXER_HEAD_DIM;
    const uint32_t cache_index_top_k = 1u;
    ds4_gpu_tensor *cache_raw_t = ds4_gpu_tensor_alloc((uint64_t)cache_raw_cap * kv_width * sizeof(float));
    ds4_gpu_tensor *cache_attn_state_kv_t = cache_ratio
        ? ds4_gpu_tensor_alloc((uint64_t)attn_state_rows * attn_state_width * sizeof(float))
        : NULL;
    ds4_gpu_tensor *cache_attn_state_score_t = cache_ratio
        ? ds4_gpu_tensor_alloc((uint64_t)attn_state_rows * attn_state_width * sizeof(float))
        : NULL;
    ds4_gpu_tensor *cache_attn_comp_t = cache_ratio
        ? ds4_gpu_tensor_alloc((uint64_t)cache_comp_cap * kv_width * sizeof(float))
        : NULL;
    ds4_gpu_tensor *cache_index_state_kv_t = state.has_indexer
        ? ds4_gpu_tensor_alloc((uint64_t)index_state_rows * index_state_width * sizeof(float))
        : NULL;
    ds4_gpu_tensor *cache_index_state_score_t = state.has_indexer
        ? ds4_gpu_tensor_alloc((uint64_t)index_state_rows * index_state_width * sizeof(float))
        : NULL;
    ds4_gpu_tensor *cache_index_comp_t = state.has_indexer
        ? ds4_gpu_tensor_alloc((uint64_t)cache_comp_cap * DS4_V100_INDEXER_HEAD_DIM * sizeof(float))
        : NULL;
    ds4_gpu_tensor *cache_index_topk_t = state.has_indexer
        ? ds4_gpu_tensor_alloc((uint64_t)cache_index_top_k * sizeof(uint32_t))
        : NULL;
    check(hidden_t && raw_t && comp_t && mask_t && next_t && hidden_hc_t && next_hc_t && cache_raw_t,
          "device tensor allocation");
    if (cache_ratio) {
        check(cache_attn_state_kv_t && cache_attn_state_score_t && cache_attn_comp_t,
              "attention decode-cache tensor allocation");
    }
    if (state.has_indexer) {
        check(cache_index_state_kv_t && cache_index_state_score_t && cache_index_comp_t &&
                  cache_index_topk_t,
              "indexer decode-cache tensor allocation");
    }

    ds4_v100_layer_execute_report report;
    memset(&report, 0, sizeof(report));
    ds4_v100_layer_execute_report hc_report;
    memset(&hc_report, 0, sizeof(hc_report));
    int hc_executed = 0;
    int cache_executed = 0;
    if (!failures) {
        check(ds4_gpu_tensor_write(hidden_t, 0, hidden_x, (uint64_t)hidden * sizeof(float)),
              "hidden upload");
        check(ds4_gpu_tensor_write(hidden_hc_t,
                                   0,
                                   hidden_hc,
                                   (uint64_t)DS4_V100_N_HC * hidden * sizeof(float)),
              "hidden hc upload");
        check(ds4_gpu_tensor_write(raw_t, 0, raw_kv, (uint64_t)n_raw * kv_width * sizeof(float)),
              "raw kv upload");
        check(ds4_gpu_tensor_write(comp_t, 0, comp_kv, (uint64_t)n_comp * kv_width * sizeof(float)),
              "compressed kv upload");
        check(ds4_gpu_tensor_write(mask_t, 0, comp_mask, (uint64_t)n_comp * sizeof(float)),
              "compressed mask upload");

        ds4_v100_layer_execute_config cfg = {
            .model_map = model.ptr,
            .model_size = model.size,
            .arena = arena,
            .router_token = (uint32_t)router_token,
            .position = (uint32_t)position,
            .raw_kv = raw_t,
            .n_raw = n_raw,
            .raw_cap = n_raw,
            .raw_start = 0,
            .compressed_kv = comp_t,
            .n_compressed = n_comp,
            .compressed_mask = mask_t,
            .use_compressed_mask = true,
        };
        err[0] = '\0';
        check(ds4_v100_layer_execute_decode(&state,
                                            &cfg,
                                            hidden_t,
                                            next_t,
                                            &report,
                                            err,
                                            sizeof(err)) == 0,
              err[0] ? err : "layer execute decode");
        check(ds4_gpu_tensor_read(next_t, 0, next_gpu, (uint64_t)hidden * sizeof(float)),
              "next hidden read");
        for (uint32_t route = 0; route < 6; route++) {
            if (report.selected_experts[route] != selected[route]) {
                fprintf(stderr,
                        "cuda_v100_integrated_layer_smoke: route %u selected %d expected %d\n",
                        route,
                        report.selected_experts[route],
                        selected[route]);
                failures++;
            }
            const float tol = 2e-5f + 2e-5f * fabsf(weights[route]);
            if (fabsf(report.route_weights[route] - weights[route]) > tol) {
                fprintf(stderr,
                        "cuda_v100_integrated_layer_smoke: route %u weight %.8g expected %.8g\n",
                        route,
                        report.route_weights[route],
                        weights[route]);
                failures++;
            }
        }
        expect_close_vector(next_gpu, next_cpu, hidden, "integrated next hidden", 0.35f, 0.05f);

        const int hc_failures_before = failures;
        err[0] = '\0';
        check(ds4_v100_layer_execute_hc_decode(&state,
                                               &cfg,
                                               hidden_hc_t,
                                               next_hc_t,
                                               &hc_report,
                                               err,
                                               sizeof(err)) == 0,
              err[0] ? err : "HC layer execute decode");
        check(ds4_gpu_tensor_read(next_hc_t,
                                  0,
                                  next_hc_gpu,
                                  (uint64_t)DS4_V100_N_HC * hidden * sizeof(float)),
              "next HC read");
        expect_finite_nonzero_vector(next_hc_gpu,
                                     DS4_V100_N_HC * hidden,
                                     "integrated next HC");
        check(hc_report.routes == state.routes_per_token, "HC route count");
        for (uint32_t route = 0; route < hc_report.routes; route++) {
            check(hc_report.selected_experts[route] >= 0 &&
                      (uint32_t)hc_report.selected_experts[route] < state.routed_experts,
                  "HC selected expert range");
        }
        hc_executed = failures == hc_failures_before;

        if (state.compress_ratio == 4u && state.has_attention_compressor && state.has_indexer) {
            const int cache_failures_before = failures;
            check(ds4_gpu_tensor_write(hidden_t, 0, hidden_x, (uint64_t)hidden * sizeof(float)),
                  "cache hidden reset");
            check(ds4_gpu_tensor_fill_f32(cache_raw_t, 0.0f, (uint64_t)cache_raw_cap * kv_width),
                  "raw cache zero");
            check(ds4_gpu_tensor_fill_f32(cache_attn_state_kv_t,
                                          0.0f,
                                          (uint64_t)attn_state_rows * attn_state_width),
                  "attention state KV zero");
            check(ds4_gpu_tensor_fill_f32(cache_attn_state_score_t,
                                          -1.0e30f,
                                          (uint64_t)attn_state_rows * attn_state_width),
                  "attention state score init");
            check(ds4_gpu_tensor_fill_f32(cache_attn_comp_t,
                                          0.0f,
                                          (uint64_t)cache_comp_cap * kv_width),
                  "attention compressed cache zero");
            check(ds4_gpu_tensor_fill_f32(cache_index_state_kv_t,
                                          0.0f,
                                          (uint64_t)index_state_rows * index_state_width),
                  "indexer state KV zero");
            check(ds4_gpu_tensor_fill_f32(cache_index_state_score_t,
                                          -1.0e30f,
                                          (uint64_t)index_state_rows * index_state_width),
                  "indexer state score init");
            check(ds4_gpu_tensor_fill_f32(cache_index_comp_t,
                                          0.0f,
                                          (uint64_t)cache_comp_cap * DS4_V100_INDEXER_HEAD_DIM),
                  "indexer compressed cache zero");

            ds4_v100_layer_decode_cache decode_cache = {
                .raw_kv = cache_raw_t,
                .raw_cap = cache_raw_cap,
                .raw_window = 128u,
                .attn_state_kv = cache_attn_state_kv_t,
                .attn_state_score = cache_attn_state_score_t,
                .attn_comp_kv = cache_attn_comp_t,
                .attn_comp_cap = cache_comp_cap,
                .n_attn_comp = 0,
                .index_state_kv = cache_index_state_kv_t,
                .index_state_score = cache_index_state_score_t,
                .index_comp_kv = cache_index_comp_t,
                .index_comp_cap = cache_comp_cap,
                .n_index_comp = 0,
                .indexer_topk = cache_index_topk_t,
                .indexer_top_k = cache_index_top_k,
            };
            for (uint32_t pos = 0; pos < 8u && !failures; pos++) {
                ds4_v100_layer_execute_config cache_cfg = {
                    .model_map = model.ptr,
                    .model_size = model.size,
                    .arena = arena,
                    .router_token = (uint32_t)router_token,
                    .position = pos,
                    .decode_cache = &decode_cache,
                };
                ds4_v100_layer_execute_report cache_report;
                memset(&cache_report, 0, sizeof(cache_report));
                err[0] = '\0';
                check(ds4_v100_layer_execute_decode(&state,
                                                    &cache_cfg,
                                                    hidden_t,
                                                    next_t,
                                                    &cache_report,
                                                    err,
                                                    sizeof(err)) == 0,
                      err[0] ? err : "decode-cache layer execute");
                check(cache_report.routes == state.routes_per_token,
                      "decode-cache route count");
                check(ds4_gpu_tensor_copy(hidden_t,
                                          0,
                                          next_t,
                                          0,
                                          (uint64_t)hidden * sizeof(float)),
                      "decode-cache next hidden copy");
            }
            check(decode_cache.n_attn_comp == 2u, "decode-cache attention compressed count");
            check(decode_cache.n_index_comp == 2u, "decode-cache indexer compressed count");
            check(ds4_gpu_tensor_read(cache_attn_comp_t,
                                      0,
                                      comp_kv,
                                      2u * (uint64_t)kv_width * sizeof(float)),
                  "decode-cache attention compressed read");
            expect_finite_nonzero_vector(comp_kv,
                                         2u * kv_width,
                                         "decode-cache attention compressed rows");
            check(ds4_gpu_tensor_read(cache_index_comp_t,
                                      0,
                                      comp_kv,
                                      2u * (uint64_t)DS4_V100_INDEXER_HEAD_DIM * sizeof(float)),
                  "decode-cache indexer compressed read");
            expect_finite_nonzero_vector(comp_kv,
                                         2u * DS4_V100_INDEXER_HEAD_DIM,
                                         "decode-cache indexer compressed rows");
            uint32_t topk_host = UINT32_MAX;
            check(ds4_gpu_tensor_read(cache_index_topk_t,
                                      0,
                                      &topk_host,
                                      sizeof(topk_host)),
                  "decode-cache top-k read");
            check(topk_host < decode_cache.n_index_comp, "decode-cache top-k range");
            cache_executed = failures == cache_failures_before;
        }
    }

    printf("cuda_v100_integrated_layer_smoke: layer=%d token=%d pos=%d expert0=%d hc=%s cache=%s gpu=%d arena_bytes=%" PRIu64 " hidden=%u raw=%u comp=%u %s\n",
           layer,
           router_token,
           position,
           selected[0],
           hc_executed ? "ok" : "skip",
           cache_executed ? "ok" : "skip",
           state.owning_gpu,
           arena_bytes,
           hidden,
           n_raw,
           n_comp,
           failures ? "FAIL" : "ok");

    ds4_gpu_tensor_free(cache_index_topk_t);
    ds4_gpu_tensor_free(cache_index_comp_t);
    ds4_gpu_tensor_free(cache_index_state_score_t);
    ds4_gpu_tensor_free(cache_index_state_kv_t);
    ds4_gpu_tensor_free(cache_attn_comp_t);
    ds4_gpu_tensor_free(cache_attn_state_score_t);
    ds4_gpu_tensor_free(cache_attn_state_kv_t);
    ds4_gpu_tensor_free(cache_raw_t);
    ds4_gpu_tensor_free(next_hc_t);
    ds4_gpu_tensor_free(hidden_hc_t);
    ds4_gpu_tensor_free(next_t);
    ds4_gpu_tensor_free(mask_t);
    ds4_gpu_tensor_free(comp_t);
    ds4_gpu_tensor_free(raw_t);
    ds4_gpu_tensor_free(hidden_t);
    ds4_gpu_arena_close(arena);
    free(scratch);
    free(next_gpu);
    free(next_hc_gpu);
    free(hidden_hc);
    free(next_cpu);
    free(ffn_delta);
    free(s_out);
    free(s_mid);
    free(s_up);
    free(s_gate);
    free(r_out);
    free(r_mid);
    free(r_up);
    free(r_gate);
    free(ffn_norm);
    free(residual);
    free(attn_out);
    free(low);
    free(heads);
    free(comp_kv);
    free(raw_kv);
    free(kv);
    free(kv_raw);
    free(q);
    free(q_a_norm);
    free(q_a);
    free(attn_norm);
    free(hidden_x);
    ds4_v100_context_close(ctx);
    unmap_model_file(&model);
    return failures ? 1 : 0;
}
