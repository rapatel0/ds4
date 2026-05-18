#include "ds4_gpu.h"
#include "ds4_source_formats.h"
#include "ds4_v100_context.h"

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

typedef struct {
    ds4_v100_tensor_binding binding;
    uint32_t rows;
    uint32_t cols;
    uint64_t row_bytes;
    uint64_t bytes;
    uint64_t rel;
} bound_matrix;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/cuda_v100_descriptor_bound_ffn_smoke --index FILE --model FILE [--layer N] [--expert N]\n");
}

static int parse_int_arg(const char *s, const char *name, int max_v) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || !*s || !end || *end != '\0' || v < 0 || v > max_v) {
        fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: invalid %s: %s\n",
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
        fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: cannot stat %s\n", path);
        close(fd);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: cannot mmap %s: %s\n",
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

static float swiglu_ref(float gate, float up, float clamp, float weight) {
    if (clamp > 1.0e-6f) {
        if (gate > clamp) gate = clamp;
        if (up > clamp) up = clamp;
        if (up < -clamp) up = -clamp;
    }
    return (gate / (1.0f + expf(-gate))) * up * weight;
}

static void fill_hidden(float *hidden, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        int a = (int)(i % 37u) - 18;
        int b = (int)((i * 17u) % 29u) - 14;
        hidden[i] = ((float)a * 0.00037f) + ((float)b * 0.00011f);
    }
}

static int require_binding(ds4_v100_context *ctx,
                           int layer,
                           const char *suffix,
                           ds4_v100_tensor_binding *out,
                           char *err,
                           size_t errlen) {
    if (ds4_v100_context_require_layer_tensor_binding(ctx, layer, suffix, out, err, errlen)) {
        fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: %s\n", err);
        return 1;
    }
    return 0;
}

static int make_expert_matrix(const ds4_v100_tensor_binding *b,
                              uint32_t expert,
                              bound_matrix *out) {
    if (!b || !out || b->n_shape_dims != 3 || expert >= b->shape[2]) return 1;
    memset(out, 0, sizeof(*out));
    out->binding = *b;
    out->cols = (uint32_t)b->shape[0];
    out->rows = (uint32_t)b->shape[1];
    out->row_bytes = ds4_src_mxfp4_row_bytes(out->cols);
    out->bytes = (uint64_t)out->rows * out->row_bytes;
    out->rel = (uint64_t)expert * out->bytes;
    if (out->rel > b->byte_length || out->bytes > b->byte_length - out->rel) return 1;
    return 0;
}

static int make_shared_matrix(const ds4_v100_tensor_binding *b, bound_matrix *out) {
    if (!b || !out || b->n_shape_dims != 2) return 1;
    memset(out, 0, sizeof(*out));
    out->binding = *b;
    out->cols = (uint32_t)b->shape[0];
    out->rows = (uint32_t)b->shape[1];
    out->row_bytes = ds4_src_f8_e4m3_b128_row_bytes(out->cols);
    out->bytes = (uint64_t)out->rows * out->row_bytes;
    out->rel = 0;
    if (out->bytes != b->byte_length) return 1;
    return 0;
}

static const uint8_t *matrix_host_ptr(const model_map *model, const bound_matrix *m) {
    const ds4_v100_tensor_binding *b = &m->binding;
    if (b->source_offset + m->rel > model->size ||
        m->bytes > model->size - b->source_offset - m->rel) {
        return NULL;
    }
    return model->ptr + b->source_offset + m->rel;
}

static uint64_t matrix_arena_offset(const bound_matrix *m) {
    return m->binding.shard_offset + m->rel;
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
    return ds4_gpu_arena_upload(arena, matrix_arena_offset(m), *scratch, m->bytes);
}

static void cpu_mxfp4_matmul(const bound_matrix *m,
                             const model_map *model,
                             const float *x,
                             float *out) {
    const uint8_t *base = matrix_host_ptr(model, m);
    char err[128];
    for (uint32_t r = 0; r < m->rows; r++) {
        err[0] = '\0';
        if (ds4_src_mxfp4_row_dot(&out[r],
                                  base + (uint64_t)r * m->row_bytes,
                                  x,
                                  m->cols,
                                  err,
                                  sizeof(err)) != 0) {
            fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: %s\n", err);
            failures++;
            out[r] = 0.0f;
        }
    }
}

static void cpu_f8_matmul(const bound_matrix *m,
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
            fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: %s\n", err);
            failures++;
            out[r] = 0.0f;
        }
    }
}

static void cpu_swiglu(float *out,
                       const float *gate,
                       const float *up,
                       uint32_t n,
                       float weight) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] = swiglu_ref(gate[i], up[i], 10.0f, weight);
    }
}

static ds4_gpu_source_row_view source_view(const bound_matrix *m) {
    ds4_gpu_source_row_view v;
    memset(&v, 0, sizeof(v));
    v.arena_offset = matrix_arena_offset(m);
    v.byte_length = m->bytes;
    v.rows = m->rows;
    v.cols = m->cols;
    v.row_stride_bytes = (uint32_t)m->row_bytes;
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
    const float tol = 0.08f + 0.015f * fabsf(want[max_i]);
    if (max_abs > tol) {
        fprintf(stderr,
                "cuda_v100_descriptor_bound_ffn_smoke: %s max_abs %.8g at %u got %.8g expected %.8g tol %.8g\n",
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
    int expert = 0;

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
        } else if (!strcmp(argv[i], "--expert") && i + 1 < argc) {
            expert = parse_int_arg(argv[++i], "--expert", 255);
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
        fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: no CUDA devices visible\n");
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
        fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: %s\n", err);
        unmap_model_file(&model);
        return 1;
    }

    ds4_v100_tensor_binding gate_b;
    ds4_v100_tensor_binding up_b;
    ds4_v100_tensor_binding down_b;
    ds4_v100_tensor_binding shared_gate_b;
    ds4_v100_tensor_binding shared_up_b;
    ds4_v100_tensor_binding shared_down_b;
    if (require_binding(ctx, layer, "ffn_gate_exps.weight", &gate_b, err, sizeof(err)) ||
        require_binding(ctx, layer, "ffn_up_exps.weight", &up_b, err, sizeof(err)) ||
        require_binding(ctx, layer, "ffn_down_exps.weight", &down_b, err, sizeof(err)) ||
        require_binding(ctx, layer, "ffn_gate_shexp.weight", &shared_gate_b, err, sizeof(err)) ||
        require_binding(ctx, layer, "ffn_up_shexp.weight", &shared_up_b, err, sizeof(err)) ||
        require_binding(ctx, layer, "ffn_down_shexp.weight", &shared_down_b, err, sizeof(err))) {
        ds4_v100_context_close(ctx);
        unmap_model_file(&model);
        return 1;
    }

    bound_matrix gate;
    bound_matrix up;
    bound_matrix down;
    bound_matrix shared_gate;
    bound_matrix shared_up;
    bound_matrix shared_down;
    check(make_expert_matrix(&gate_b, (uint32_t)expert, &gate) == 0, "gate expert matrix");
    check(make_expert_matrix(&up_b, (uint32_t)expert, &up) == 0, "up expert matrix");
    check(make_expert_matrix(&down_b, (uint32_t)expert, &down) == 0, "down expert matrix");
    check(make_shared_matrix(&shared_gate_b, &shared_gate) == 0, "shared gate matrix");
    check(make_shared_matrix(&shared_up_b, &shared_up) == 0, "shared up matrix");
    check(make_shared_matrix(&shared_down_b, &shared_down) == 0, "shared down matrix");
    check(gate.cols == shared_gate.cols && up.cols == shared_up.cols, "gate/up hidden dims");
    check(down.rows == shared_down.rows && down.cols == shared_down.cols, "down dims");
    if (failures) {
        ds4_v100_context_close(ctx);
        unmap_model_file(&model);
        return 1;
    }

    const uint32_t hidden = gate.cols;
    const uint32_t mid = gate.rows;
    uint64_t arena_bytes = 0;
    const bound_matrix *all[] = { &gate, &up, &down, &shared_gate, &shared_up, &shared_down };
    for (uint32_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) {
        uint64_t end = matrix_arena_offset(all[i]) + all[i]->bytes;
        if (end > arena_bytes) arena_bytes = end;
    }

    ds4_gpu_arena *arena = NULL;
    check(ds4_gpu_arena_open(&arena, gate_b.owning_gpu, arena_bytes) == 0, "arena open");
    check(arena && ds4_gpu_arena_is_device_memory(arena), "arena is device memory");
    unsigned char *scratch = NULL;
    uint64_t scratch_bytes = 0;
    if (arena) {
        for (uint32_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) {
            check(upload_matrix(arena, &model, all[i], &scratch, &scratch_bytes) == 0,
                  "upload matrix");
        }
    }

    float *hidden_x = (float *)calloc(hidden, sizeof(float));
    float *r_gate = (float *)calloc(mid, sizeof(float));
    float *r_up = (float *)calloc(mid, sizeof(float));
    float *r_mid = (float *)calloc(mid, sizeof(float));
    float *r_out = (float *)calloc(hidden, sizeof(float));
    float *s_gate = (float *)calloc(mid, sizeof(float));
    float *s_up = (float *)calloc(mid, sizeof(float));
    float *s_mid = (float *)calloc(mid, sizeof(float));
    float *s_out = (float *)calloc(hidden, sizeof(float));
    float *cpu_out = (float *)calloc(hidden, sizeof(float));
    float *gpu_out = (float *)calloc(hidden, sizeof(float));
    check(hidden_x && r_gate && r_up && r_mid && r_out && s_gate && s_up && s_mid &&
              s_out && cpu_out && gpu_out,
          "host buffer allocation");
    if (!failures) {
        fill_hidden(hidden_x, hidden);
        cpu_mxfp4_matmul(&gate, &model, hidden_x, r_gate);
        cpu_mxfp4_matmul(&up, &model, hidden_x, r_up);
        cpu_swiglu(r_mid, r_gate, r_up, mid, 1.0f);
        cpu_mxfp4_matmul(&down, &model, r_mid, r_out);

        cpu_f8_matmul(&shared_gate, &model, hidden_x, s_gate);
        cpu_f8_matmul(&shared_up, &model, hidden_x, s_up);
        cpu_swiglu(s_mid, s_gate, s_up, mid, 1.0f);
        cpu_f8_matmul(&shared_down, &model, s_mid, s_out);

        for (uint32_t i = 0; i < hidden; i++) cpu_out[i] = r_out[i] + s_out[i];
    }

    ds4_gpu_tensor *hidden_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *gate_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *up_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *mid_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *route_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *sgate_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *sup_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *smid_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *shared_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *sum_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    check(hidden_t && gate_t && up_t && mid_t && route_t && sgate_t && sup_t &&
              smid_t && shared_t && sum_t,
          "device tensor allocation");

    if (!failures) {
        ds4_gpu_source_row_view gate_v = source_view(&gate);
        ds4_gpu_source_row_view up_v = source_view(&up);
        ds4_gpu_source_row_view down_v = source_view(&down);
        ds4_gpu_source_row_view shared_gate_v = source_view(&shared_gate);
        ds4_gpu_source_row_view shared_up_v = source_view(&shared_up);
        ds4_gpu_source_row_view shared_down_v = source_view(&shared_down);

        check(ds4_gpu_tensor_write(hidden_t, 0, hidden_x, (uint64_t)hidden * sizeof(float)),
              "hidden upload");
        check(ds4_gpu_arena_mxfp4_matmul_f32(arena, &gate_v, hidden_t, gate_t) == 0,
              "routed gate matmul");
        check(ds4_gpu_arena_mxfp4_matmul_f32(arena, &up_v, hidden_t, up_t) == 0,
              "routed up matmul");
        check(ds4_gpu_swiglu_tensor(mid_t, gate_t, up_t, mid, 10.0f, 1.0f),
              "routed swiglu");
        check(ds4_gpu_arena_mxfp4_matmul_f32(arena, &down_v, mid_t, route_t) == 0,
              "routed down matmul");

        check(ds4_gpu_arena_f8_e4m3_b128_matmul_f32(arena, &shared_gate_v, hidden_t, sgate_t) == 0,
              "shared gate matmul");
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_f32(arena, &shared_up_v, hidden_t, sup_t) == 0,
              "shared up matmul");
        check(ds4_gpu_swiglu_tensor(smid_t, sgate_t, sup_t, mid, 10.0f, 1.0f),
              "shared swiglu");
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_f32(arena, &shared_down_v, smid_t, shared_t) == 0,
              "shared down matmul");
        check(ds4_gpu_add_tensor(sum_t, route_t, shared_t, hidden), "ffn add");
        check(ds4_gpu_tensor_read(sum_t, 0, gpu_out, (uint64_t)hidden * sizeof(float)),
              "ffn read");
        expect_close_vector(gpu_out, cpu_out, hidden, "descriptor-bound ffn");
    }

    printf("cuda_v100_descriptor_bound_ffn_smoke: layer=%d expert=%d gpu=%d arena_bytes=%" PRIu64 " hidden=%u mid=%u max_source_offset=%" PRIu64 " %s\n",
           layer,
           expert,
           gate_b.owning_gpu,
           arena_bytes,
           hidden,
           mid,
           up.binding.source_offset + up.rel + up.bytes,
           failures ? "FAIL" : "ok");

    ds4_gpu_tensor_free(sum_t);
    ds4_gpu_tensor_free(shared_t);
    ds4_gpu_tensor_free(smid_t);
    ds4_gpu_tensor_free(sup_t);
    ds4_gpu_tensor_free(sgate_t);
    ds4_gpu_tensor_free(route_t);
    ds4_gpu_tensor_free(mid_t);
    ds4_gpu_tensor_free(up_t);
    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(hidden_t);
    ds4_gpu_arena_close(arena);
    free(scratch);
    free(gpu_out);
    free(cpu_out);
    free(s_out);
    free(s_mid);
    free(s_up);
    free(s_gate);
    free(r_out);
    free(r_mid);
    free(r_up);
    free(r_gate);
    free(hidden_x);
    ds4_v100_context_close(ctx);
    unmap_model_file(&model);
    return failures ? 1 : 0;
}
