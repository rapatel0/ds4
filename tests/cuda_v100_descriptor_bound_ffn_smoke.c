#include "ds4_gpu.h"
#include "ds4_source_formats.h"
#include "engine/layer_state.h"

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

typedef ds4_bound_matrix bound_matrix;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/cuda_v100_descriptor_bound_ffn_smoke --index FILE --model FILE [--layer N] [--router-token N]\n");
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

static float softplus_ref(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static void fill_hidden(float *hidden, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        int a = (int)(i % 37u) - 18;
        int b = (int)((i * 17u) % 29u) - 14;
        hidden[i] = ((float)a * 0.00037f) + ((float)b * 0.00011f);
    }
}

static const uint8_t *matrix_host_ptr(const model_map *model, const bound_matrix *m) {
    const ds4_tensor_binding *b = &m->binding;
    if (b->source_offset + m->rel > model->size ||
        m->bytes > model->size - b->source_offset - m->rel) {
        return NULL;
    }
    return model->ptr + b->source_offset + m->rel;
}

static uint64_t matrix_arena_offset(const bound_matrix *m) {
    return ds4_bound_matrix_arena_offset(m);
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

static void cpu_hash_router_select(const ds4_tensor_binding *hash_b,
                                   const model_map *model,
                                   const float logits[256],
                                   uint32_t token,
                                   int32_t selected[6],
                                   float weights[6],
                                   float probs[256]) {
    for (uint32_t e = 0; e < 256; e++) probs[e] = sqrtf(softplus_ref(logits[e]));
    if (!hash_b || hash_b->n_shape_dims != 2 || hash_b->shape[0] != 6 ||
        token >= hash_b->shape[1] ||
        hash_b->source_offset > model->size ||
        hash_b->byte_length > model->size - hash_b->source_offset) {
        failures++;
        return;
    }
    const int32_t *hash = (const int32_t *)(const void *)(model->ptr + hash_b->source_offset);
    const int32_t *row = hash + (uint64_t)token * 6u;
    float sum = 0.0f;
    for (uint32_t i = 0; i < 6; i++) {
        selected[i] = row[i];
        const int32_t e = selected[i];
        weights[i] = (e >= 0 && e < 256) ? probs[(uint32_t)e] : 0.0f;
        sum += weights[i];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t i = 0; i < 6; i++) weights[i] = weights[i] / sum * 1.5f;
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
    char err[128];
    err[0] = '\0';
    if (ds4_bound_matrix_source_view(m, &v, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: %s\n", err);
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
    int router_token = 16;

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

    ds4_context_options opts;
    ds4_context_options_init(&opts);
    opts.pack_index_path = index;
    opts.kv_ctx_tokens = 1048576;
    opts.kv_active_slots = 1;

    char err[512] = {0};
    ds4_context *ctx = NULL;
    if (ds4_context_open(&ctx, &opts, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: %s\n", err);
        unmap_model_file(&model);
        return 1;
    }

    ds4_layer_state layer_state;
    if (ds4_layer_state_init(&layer_state, ctx, layer, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: %s\n", err);
        ds4_context_close(ctx);
        unmap_model_file(&model);
        return 1;
    }
    if (layer_state.router_kind != DS4_V100_ROUTER_HASH || !layer_state.has_hash_router) {
        fprintf(stderr, "cuda_v100_descriptor_bound_ffn_smoke: layer %d is not a hash-router layer\n",
                layer);
        ds4_context_close(ctx);
        unmap_model_file(&model);
        return 1;
    }

    bound_matrix gate_routes[6];
    bound_matrix up_routes[6];
    bound_matrix down_routes[6];
    bound_matrix shared_gate = layer_state.shared_gate;
    bound_matrix shared_up = layer_state.shared_up;
    bound_matrix shared_down = layer_state.shared_down;
    bound_matrix router = layer_state.router;
    const ds4_tensor_binding *hash_b = &layer_state.router_hash;
    check(router.cols == shared_gate.cols && router.rows == 256, "router dimensions");
    check(shared_gate.cols == shared_up.cols, "shared gate/up hidden dims");
    check(shared_down.cols == shared_gate.rows, "shared down mid dim");
    if (failures) {
        ds4_context_close(ctx);
        unmap_model_file(&model);
        return 1;
    }

    const uint32_t hidden = router.cols;
    const uint32_t mid = shared_gate.rows;
    float *hidden_x = (float *)calloc(hidden, sizeof(float));
    float cpu_logits[256];
    float cpu_probs[256];
    int32_t cpu_selected[6];
    float cpu_weights[6];
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
        cpu_f32_matmul(&router, &model, hidden_x, cpu_logits);
        cpu_hash_router_select(hash_b,
                               &model,
                               cpu_logits,
                               (uint32_t)router_token,
                               cpu_selected,
                               cpu_weights,
                               cpu_probs);
        for (uint32_t route = 0; route < 6; route++) {
            uint32_t expert = (uint32_t)cpu_selected[route];
            check(expert < 256, "router selected invalid expert");
            ds4_route_matrices route_views;
            check(ds4_layer_state_route_matrices(&layer_state,
                                                      expert,
                                                      &route_views,
                                                      err,
                                                      sizeof(err)) == 0,
                  "route matrices");
            gate_routes[route] = route_views.gate;
            up_routes[route] = route_views.up;
            down_routes[route] = route_views.down;
            check(gate_routes[route].cols == hidden && gate_routes[route].rows == mid,
                  "route gate dimensions");
            check(down_routes[route].rows == hidden && down_routes[route].cols == mid,
                  "route down dimensions");
        }
    }

    uint64_t arena_bytes = 0;
    check(ds4_layer_state_ffn_arena_span(&layer_state,
                                              cpu_selected,
                                              6,
                                              &arena_bytes,
                                              err,
                                              sizeof(err)) == 0,
          "ffn arena span");

    ds4_gpu_arena *arena = NULL;
    check(ds4_gpu_arena_open(&arena, layer_state.owning_gpu, arena_bytes) == 0, "arena open");
    check(arena && ds4_gpu_arena_is_device_memory(arena), "arena is device memory");
    unsigned char *scratch = NULL;
    uint64_t scratch_bytes = 0;
    const bound_matrix *fixed[] = { &router, &shared_gate, &shared_up, &shared_down };
    if (arena && !failures) {
        for (uint32_t i = 0; i < sizeof(fixed) / sizeof(fixed[0]); i++) {
            check(upload_matrix(arena, &model, fixed[i], &scratch, &scratch_bytes) == 0,
                  "upload fixed matrix");
        }
        for (uint32_t route = 0; route < 6; route++) {
            const bound_matrix *routed[] = { &gate_routes[route], &up_routes[route], &down_routes[route] };
            for (uint32_t i = 0; i < sizeof(routed) / sizeof(routed[0]); i++) {
                check(upload_matrix(arena, &model, routed[i], &scratch, &scratch_bytes) == 0,
                      "upload routed matrix");
            }
        }
    }

    if (!failures) {
        for (uint32_t route = 0; route < 6; route++) {
            cpu_mxfp4_matmul(&gate_routes[route], &model, hidden_x, r_gate);
            cpu_mxfp4_matmul(&up_routes[route], &model, hidden_x, r_up);
            cpu_swiglu(r_mid, r_gate, r_up, mid, cpu_weights[route]);
            cpu_mxfp4_matmul(&down_routes[route], &model, r_mid, r_out);
            for (uint32_t i = 0; i < hidden; i++) cpu_out[i] += r_out[i];
        }

        cpu_f8_matmul(&shared_gate, &model, hidden_x, s_gate);
        cpu_f8_matmul(&shared_up, &model, hidden_x, s_up);
        cpu_swiglu(s_mid, s_gate, s_up, mid, 1.0f);
        cpu_f8_matmul(&shared_down, &model, s_mid, s_out);

        for (uint32_t i = 0; i < hidden; i++) cpu_out[i] += s_out[i];
    }

    ds4_gpu_tensor *hidden_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *router_t = ds4_gpu_tensor_alloc(256u * sizeof(float));
    ds4_gpu_tensor *probs_t = ds4_gpu_tensor_alloc(256u * sizeof(float));
    ds4_gpu_tensor *selected_t = ds4_gpu_tensor_alloc(6u * sizeof(int32_t));
    ds4_gpu_tensor *weights_t = ds4_gpu_tensor_alloc(6u * sizeof(float));
    ds4_gpu_tensor *gate_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *up_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *mid_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *route_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *accum_a = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *accum_b = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *sgate_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *sup_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *smid_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *shared_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *sum_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    check(hidden_t && router_t && probs_t && selected_t && weights_t &&
              gate_t && up_t && mid_t && route_t && accum_a && accum_b &&
              sgate_t && sup_t && smid_t && shared_t && sum_t,
          "device tensor allocation");

    if (!failures) {
        ds4_gpu_source_row_view router_v = source_view(&router);
        ds4_gpu_source_row_view shared_gate_v = source_view(&shared_gate);
        ds4_gpu_source_row_view shared_up_v = source_view(&shared_up);
        ds4_gpu_source_row_view shared_down_v = source_view(&shared_down);
        int32_t gpu_selected[6];
        float gpu_weights[6];

        check(ds4_gpu_tensor_write(hidden_t, 0, hidden_x, (uint64_t)hidden * sizeof(float)),
              "hidden upload");
        check(ds4_gpu_arena_f32_matmul_f32(arena, &router_v, hidden_t, router_t) == 0,
              "router f32 matmul");
        check(ds4_gpu_router_select_tensor(selected_t,
                                           weights_t,
                                           probs_t,
                                           model.ptr,
                                           model.size,
                                           0,
                                           hash_b->source_offset,
                                           (uint32_t)hash_b->shape[1],
                                           (uint32_t)router_token,
                                           0,
                                           0,
                                           false,
                                           true,
                                           router_t),
              "router select");
        check(ds4_gpu_tensor_read(selected_t, 0, gpu_selected, sizeof(gpu_selected)),
              "selected read");
        check(ds4_gpu_tensor_read(weights_t, 0, gpu_weights, sizeof(gpu_weights)),
              "weights read");
        for (uint32_t route = 0; route < 6; route++) {
            if (gpu_selected[route] != cpu_selected[route]) {
                fprintf(stderr,
                        "cuda_v100_descriptor_bound_ffn_smoke: route %u selected %d expected %d\n",
                        route,
                        gpu_selected[route],
                        cpu_selected[route]);
                failures++;
            }
            const float tol = 2e-5f + 2e-5f * fabsf(cpu_weights[route]);
            if (fabsf(gpu_weights[route] - cpu_weights[route]) > tol) {
                fprintf(stderr,
                        "cuda_v100_descriptor_bound_ffn_smoke: route %u weight %.8g expected %.8g\n",
                        route,
                        gpu_weights[route],
                        cpu_weights[route]);
                failures++;
            }
        }
        check(ds4_gpu_tensor_fill_f32(accum_a, 0.0f, hidden), "route accum zero");
        ds4_gpu_tensor *accum = accum_a;
        ds4_gpu_tensor *next = accum_b;
        for (uint32_t route = 0; route < 6; route++) {
            ds4_gpu_source_row_view gate_v = source_view(&gate_routes[route]);
            ds4_gpu_source_row_view up_v = source_view(&up_routes[route]);
            ds4_gpu_source_row_view down_v = source_view(&down_routes[route]);
            check(ds4_gpu_arena_mxfp4_matmul_f32(arena, &gate_v, hidden_t, gate_t) == 0,
                  "routed gate matmul");
            check(ds4_gpu_arena_mxfp4_matmul_f32(arena, &up_v, hidden_t, up_t) == 0,
                  "routed up matmul");
            check(ds4_gpu_swiglu_tensor(mid_t, gate_t, up_t, mid, 10.0f, gpu_weights[route]),
                  "routed swiglu");
            check(ds4_gpu_arena_mxfp4_matmul_f32(arena, &down_v, mid_t, route_t) == 0,
                  "routed down matmul");
            check(ds4_gpu_add_tensor(next, accum, route_t, hidden), "route accumulate");
            ds4_gpu_tensor *tmp = accum;
            accum = next;
            next = tmp;
        }

        check(ds4_gpu_arena_f8_e4m3_b128_matmul_f32(arena, &shared_gate_v, hidden_t, sgate_t) == 0,
              "shared gate matmul");
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_f32(arena, &shared_up_v, hidden_t, sup_t) == 0,
              "shared up matmul");
        check(ds4_gpu_swiglu_tensor(smid_t, sgate_t, sup_t, mid, 10.0f, 1.0f),
              "shared swiglu");
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_f32(arena, &shared_down_v, smid_t, shared_t) == 0,
              "shared down matmul");
        check(ds4_gpu_add_tensor(sum_t, accum, shared_t, hidden), "ffn add");
        check(ds4_gpu_tensor_read(sum_t, 0, gpu_out, (uint64_t)hidden * sizeof(float)),
              "ffn read");
        expect_close_vector(gpu_out, cpu_out, hidden, "descriptor-bound ffn");
    }

    printf("cuda_v100_descriptor_bound_ffn_smoke: layer=%d token=%d expert0=%d gpu=%d arena_bytes=%" PRIu64 " hidden=%u mid=%u max_source_offset=%" PRIu64 " %s\n",
           layer,
           router_token,
           cpu_selected[0],
           layer_state.owning_gpu,
           arena_bytes,
           hidden,
           mid,
           up_routes[0].binding.source_offset + up_routes[0].rel + up_routes[0].bytes,
           failures ? "FAIL" : "ok");

    ds4_gpu_tensor_free(sum_t);
    ds4_gpu_tensor_free(shared_t);
    ds4_gpu_tensor_free(smid_t);
    ds4_gpu_tensor_free(sup_t);
    ds4_gpu_tensor_free(sgate_t);
    ds4_gpu_tensor_free(accum_b);
    ds4_gpu_tensor_free(accum_a);
    ds4_gpu_tensor_free(route_t);
    ds4_gpu_tensor_free(mid_t);
    ds4_gpu_tensor_free(up_t);
    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(weights_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(probs_t);
    ds4_gpu_tensor_free(router_t);
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
    ds4_context_close(ctx);
    unmap_model_file(&model);
    return failures ? 1 : 0;
}
