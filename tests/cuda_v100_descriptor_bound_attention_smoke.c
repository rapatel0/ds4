#include "ds4_gpu.h"
#include "ds4_source_formats.h"
#include "ds4_v100_layer_state.h"

#include <errno.h>
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

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_v100_descriptor_bound_attention_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/cuda_v100_descriptor_bound_attention_smoke --index FILE --model FILE [--layer N]\n");
}

static int parse_int_arg(const char *s, const char *name, int max_v) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || !*s || !end || *end != '\0' || v < 0 || v > max_v) {
        fprintf(stderr, "cuda_v100_descriptor_bound_attention_smoke: invalid %s: %s\n",
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
        fprintf(stderr, "cuda_v100_descriptor_bound_attention_smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "cuda_v100_descriptor_bound_attention_smoke: cannot stat %s\n", path);
        close(fd);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "cuda_v100_descriptor_bound_attention_smoke: cannot mmap %s: %s\n",
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
        int a = (int)(i % 41u) - 20;
        int b = (int)((i * 19u) % 31u) - 15;
        hidden[i] = ((float)a * 0.00031f) + ((float)b * 0.00009f);
    }
}

static const uint8_t *matrix_host_ptr(const model_map *model,
                                      const ds4_v100_bound_matrix *m) {
    const ds4_v100_tensor_binding *b = &m->binding;
    if (b->source_offset + m->rel > model->size ||
        m->bytes > model->size - b->source_offset - m->rel) {
        return NULL;
    }
    return model->ptr + b->source_offset + m->rel;
}

static int upload_matrix(ds4_gpu_arena *arena,
                         const model_map *model,
                         const ds4_v100_bound_matrix *m,
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

static void cpu_f8_matmul(const ds4_v100_bound_matrix *m,
                          const model_map *model,
                          const float *x,
                          float *out) {
    const uint8_t *base = matrix_host_ptr(model, m);
    char err[128];
    for (uint32_t r = 0; r < m->rows; r++) {
        err[0] = '\0';
        if (ds4_src_f8_e4m3_b128_row_dot(&out[r],
                                         base + (uint64_t)r * m->row_bytes,
                                         x,
                                         m->cols,
                                         err,
                                         sizeof(err)) != 0) {
            fprintf(stderr, "cuda_v100_descriptor_bound_attention_smoke: %s\n", err);
            failures++;
            out[r] = 0.0f;
        }
    }
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

static void cpu_add(float *out, const float *a, const float *b, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static ds4_gpu_source_row_view source_view(const ds4_v100_bound_matrix *m) {
    ds4_gpu_source_row_view v;
    char err[128];
    err[0] = '\0';
    if (ds4_v100_bound_matrix_source_view(m, &v, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_descriptor_bound_attention_smoke: %s\n", err);
        failures++;
        memset(&v, 0, sizeof(v));
    }
    return v;
}

static void expect_close_vector(const float *got,
                                const float *want,
                                uint32_t n,
                                const char *label) {
    float max_abs = 0.0f;
    uint32_t max_i = 0;
    for (uint32_t i = 0; i < n; i++) {
        float d = fabsf(got[i] - want[i]);
        if (d > max_abs) {
            max_abs = d;
            max_i = i;
        }
    }
    const float tol = 0.08f + 0.02f * fabsf(want[max_i]);
    if (max_abs > tol) {
        fprintf(stderr,
                "cuda_v100_descriptor_bound_attention_smoke: %s max_abs %.8g at %u got %.8g expected %.8g tol %.8g\n",
                label,
                max_abs,
                max_i,
                got[max_i],
                want[max_i],
                tol);
        failures++;
    }
}

int main(int argc, char **argv) {
    const char *index = NULL;
    const char *model_path = NULL;
    int layer = 2;

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
        fprintf(stderr, "cuda_v100_descriptor_bound_attention_smoke: no CUDA devices visible\n");
        return 1;
    }

    model_map model;
    if (map_model_file(model_path, &model)) return 1;

    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    opts.pack_index_path = index;
    opts.kv_ctx_tokens = 1048576;
    opts.kv_active_slots = 1;

    char err[512] = {0};
    ds4_v100_context *ctx = NULL;
    if (ds4_v100_context_open(&ctx, &opts, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_descriptor_bound_attention_smoke: %s\n", err);
        unmap_model_file(&model);
        return 1;
    }
    ds4_v100_layer_state state;
    if (ds4_v100_layer_state_init(&state, ctx, layer, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_descriptor_bound_attention_smoke: %s\n", err);
        ds4_v100_context_close(ctx);
        unmap_model_file(&model);
        return 1;
    }

    const uint32_t hidden = state.hidden_size;
    const uint32_t q_rank = state.q_lora_rank;
    const uint32_t q_width = state.q_width;
    const uint32_t kv_width = state.kv_latent_width;
    const uint32_t out_rank = state.attention_output_rank;
    uint64_t arena_bytes = 0;
    check(ds4_v100_layer_state_attention_arena_span(&state,
                                                    &arena_bytes,
                                                    err,
                                                    sizeof(err)) == 0,
          "attention arena span");

    float *hidden_x = (float *)calloc(hidden, sizeof(float));
    float *attn_norm = (float *)calloc(hidden, sizeof(float));
    float *q_a = (float *)calloc(q_rank, sizeof(float));
    float *q_a_norm = (float *)calloc(q_rank, sizeof(float));
    float *q_full = (float *)calloc(q_width, sizeof(float));
    float *kv_latent = (float *)calloc(kv_width, sizeof(float));
    float *out_a = (float *)calloc(out_rank, sizeof(float));
    float *attn_out = (float *)calloc(hidden, sizeof(float));
    float *residual = (float *)calloc(hidden, sizeof(float));
    float *ffn_norm = (float *)calloc(hidden, sizeof(float));
    float *gpu = (float *)calloc(q_width, sizeof(float));
    check(hidden_x && attn_norm && q_a && q_a_norm && q_full && kv_latent &&
              out_a && attn_out && residual && ffn_norm && gpu,
          "host buffer allocation");
    if (!failures) {
        fill_hidden(hidden_x, hidden);
        cpu_rms_norm_weight(attn_norm, hidden_x, &model, &state.attn_norm, hidden, 1e-6f);
        cpu_f8_matmul(&state.attn_q_a, &model, attn_norm, q_a);
        cpu_rms_norm_weight(q_a_norm, q_a, &model, &state.attn_q_a_norm, q_rank, 1e-6f);
        cpu_f8_matmul(&state.attn_q_b, &model, q_a_norm, q_full);
        cpu_f8_matmul(&state.attn_kv_latent, &model, attn_norm, kv_latent);
        cpu_f8_matmul(&state.attn_output_a, &model, q_full, out_a);
        cpu_f8_matmul(&state.attn_output_b, &model, out_a, attn_out);
        cpu_add(residual, hidden_x, attn_out, hidden);
        cpu_rms_norm_weight(ffn_norm, residual, &model, &state.ffn_norm, hidden, 1e-6f);
    }

    ds4_gpu_arena *arena = NULL;
    check(ds4_gpu_arena_open(&arena, state.owning_gpu, arena_bytes) == 0, "arena open");
    check(arena && ds4_gpu_arena_is_device_memory(arena), "arena is device memory");
    unsigned char *scratch = NULL;
    uint64_t scratch_bytes = 0;
    const ds4_v100_bound_matrix *mats[] = {
        &state.attn_q_a,
        &state.attn_q_b,
        &state.attn_kv_latent,
        &state.attn_output_a,
        &state.attn_output_b,
    };
    if (arena && !failures) {
        for (uint32_t i = 0; i < sizeof(mats) / sizeof(mats[0]); i++) {
            check(upload_matrix(arena, &model, mats[i], &scratch, &scratch_bytes) == 0,
                  "upload attention matrix");
        }
    }

    ds4_gpu_tensor *hidden_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *attn_norm_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *q_a_t = ds4_gpu_tensor_alloc((uint64_t)q_rank * sizeof(float));
    ds4_gpu_tensor *q_a_norm_t = ds4_gpu_tensor_alloc((uint64_t)q_rank * sizeof(float));
    ds4_gpu_tensor *q_full_t = ds4_gpu_tensor_alloc((uint64_t)q_width * sizeof(float));
    ds4_gpu_tensor *kv_t = ds4_gpu_tensor_alloc((uint64_t)kv_width * sizeof(float));
    ds4_gpu_tensor *out_a_t = ds4_gpu_tensor_alloc((uint64_t)out_rank * sizeof(float));
    ds4_gpu_tensor *attn_out_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *residual_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *ffn_norm_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    check(hidden_t && attn_norm_t && q_a_t && q_a_norm_t && q_full_t && kv_t &&
              out_a_t && attn_out_t && residual_t && ffn_norm_t,
          "device tensor allocation");

    if (!failures) {
        ds4_gpu_source_row_view q_a_v = source_view(&state.attn_q_a);
        ds4_gpu_source_row_view q_b_v = source_view(&state.attn_q_b);
        ds4_gpu_source_row_view kv_v = source_view(&state.attn_kv_latent);
        ds4_gpu_source_row_view out_a_v = source_view(&state.attn_output_a);
        ds4_gpu_source_row_view out_b_v = source_view(&state.attn_output_b);

        check(ds4_gpu_tensor_write(hidden_t, 0, hidden_x, (uint64_t)hidden * sizeof(float)),
              "hidden upload");
        check(ds4_gpu_rms_norm_weight_tensor(attn_norm_t,
                                             hidden_t,
                                             model.ptr,
                                             model.size,
                                             state.attn_norm.source_offset,
                                             hidden,
                                             1e-6f),
              "attn rms norm");
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_f32(arena, &q_a_v, attn_norm_t, q_a_t) == 0,
              "q_a matmul");
        check(ds4_gpu_rms_norm_weight_tensor(q_a_norm_t,
                                             q_a_t,
                                             model.ptr,
                                             model.size,
                                             state.attn_q_a_norm.source_offset,
                                             q_rank,
                                             1e-6f),
              "q_a rms norm");
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_f32(arena, &q_b_v, q_a_norm_t, q_full_t) == 0,
              "q_b matmul");
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_f32(arena, &kv_v, attn_norm_t, kv_t) == 0,
              "kv latent matmul");
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_f32(arena, &out_a_v, q_full_t, out_a_t) == 0,
              "attention output a matmul");
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_f32(arena, &out_b_v, out_a_t, attn_out_t) == 0,
              "attention output b matmul");
        check(ds4_gpu_add_tensor(residual_t, hidden_t, attn_out_t, hidden),
              "residual add");
        check(ds4_gpu_rms_norm_weight_tensor(ffn_norm_t,
                                             residual_t,
                                             model.ptr,
                                             model.size,
                                             state.ffn_norm.source_offset,
                                             hidden,
                                             1e-6f),
              "ffn rms norm");

        check(ds4_gpu_tensor_read(q_a_t, 0, gpu, (uint64_t)q_rank * sizeof(float)),
              "q_a read");
        expect_close_vector(gpu, q_a, q_rank, "q_a");
        check(ds4_gpu_tensor_read(q_full_t, 0, gpu, (uint64_t)q_width * sizeof(float)),
              "q_full read");
        expect_close_vector(gpu, q_full, q_width, "q_full");
        check(ds4_gpu_tensor_read(kv_t, 0, gpu, (uint64_t)kv_width * sizeof(float)),
              "kv read");
        expect_close_vector(gpu, kv_latent, kv_width, "kv_latent");
        check(ds4_gpu_tensor_read(attn_out_t, 0, gpu, (uint64_t)hidden * sizeof(float)),
              "attention output read");
        expect_close_vector(gpu, attn_out, hidden, "attention_output");
        check(ds4_gpu_tensor_read(ffn_norm_t, 0, gpu, (uint64_t)hidden * sizeof(float)),
              "ffn norm read");
        expect_close_vector(gpu, ffn_norm, hidden, "ffn_norm");
    }

    printf("cuda_v100_descriptor_bound_attention_smoke: layer=%d gpu=%d arena_bytes=%" PRIu64 " hidden=%u q=%u kv=%u out_rank=%u %s\n",
           layer,
           state.owning_gpu,
           arena_bytes,
           hidden,
           q_width,
           kv_width,
           out_rank,
           failures ? "FAIL" : "ok");

    ds4_gpu_tensor_free(ffn_norm_t);
    ds4_gpu_tensor_free(residual_t);
    ds4_gpu_tensor_free(attn_out_t);
    ds4_gpu_tensor_free(out_a_t);
    ds4_gpu_tensor_free(kv_t);
    ds4_gpu_tensor_free(q_full_t);
    ds4_gpu_tensor_free(q_a_norm_t);
    ds4_gpu_tensor_free(q_a_t);
    ds4_gpu_tensor_free(attn_norm_t);
    ds4_gpu_tensor_free(hidden_t);
    ds4_gpu_arena_close(arena);
    free(scratch);
    free(gpu);
    free(ffn_norm);
    free(residual);
    free(attn_out);
    free(out_a);
    free(kv_latent);
    free(q_full);
    free(q_a_norm);
    free(q_a);
    free(attn_norm);
    free(hidden_x);
    ds4_v100_context_close(ctx);
    unmap_model_file(&model);
    return failures ? 1 : 0;
}
