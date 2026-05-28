#include "ds4_source_formats.h"
#include "engine/scheduler.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
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

enum {
    TOPK = 5,
    N_HC = DS4_V100_HC_ROWS,
    HIDDEN = DS4_V100_HC_COLS,
};

typedef struct {
    const unsigned char *ptr;
    uint64_t size;
    int fd;
} model_map;

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_v100_output_head_parity_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/cuda_v100_output_head_parity_smoke --index FILE --model FILE\n");
}

static int map_model_file(const char *path, model_map *out) {
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "cuda_v100_output_head_parity_smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "cuda_v100_output_head_parity_smoke: cannot stat %s\n", path);
        close(fd);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "cuda_v100_output_head_parity_smoke: cannot mmap %s: %s\n",
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

static float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static void insert_topk(uint32_t *tokens, float *logits, uint32_t token, float logit) {
    for (uint32_t i = 0; i < TOPK; i++) {
        if (tokens[i] == UINT32_MAX || logit > logits[i]) {
            for (uint32_t j = TOPK - 1; j > i; j--) {
                tokens[j] = tokens[j - 1];
                logits[j] = logits[j - 1];
            }
            tokens[i] = token;
            logits[i] = logit;
            return;
        }
    }
}

static void deterministic_hc(float *hc) {
    const uint64_t n = (uint64_t)N_HC * HIDDEN;
    for (uint64_t i = 0; i < n; i++) {
        int v = (int)((i * 131u + i / 17u) % 257u) - 128;
        hc[i] = (float)v * 0.0029296875f;
    }
}

static void cpu_rms_norm_plain(float *out, const float *x, uint64_t n) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float scale = 1.0f / sqrtf((float)(ss / (double)n) + 1.0e-6f);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale;
}

static void cpu_rms_norm_weight(float *out, const float *x, const float *w, uint64_t n) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float scale = 1.0f / sqrtf((float)(ss / (double)n) + 1.0e-6f);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

static void cpu_f32_matvec(float *out,
                           const float *w,
                           const float *x,
                           uint64_t in_dim,
                           uint64_t out_dim) {
    for (uint64_t r = 0; r < out_dim; r++) {
        const float *row = w + r * in_dim;
        double acc = 0.0;
        for (uint64_t c = 0; c < in_dim; c++) {
            acc += (double)row[c] * (double)x[c];
        }
        out[r] = (float)acc;
    }
}

static void cpu_hc_weighted_sum(float *out, const float *hc, const float *weights) {
    for (uint32_t c = 0; c < HIDDEN; c++) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < N_HC; h++) {
            acc += hc[(uint64_t)h * HIDDEN + c] * weights[h];
        }
        out[c] = acc;
    }
}

static void cpu_output_topk(const model_map *model,
                            const ds4_v100_tensor_binding *hc_head_fn,
                            const ds4_v100_tensor_binding *hc_head_scale,
                            const ds4_v100_tensor_binding *hc_head_base,
                            const ds4_v100_tensor_binding *output_norm,
                            const ds4_v100_tensor_binding *output_weight,
                            const float *hc,
                            uint32_t *tokens,
                            float *logits) {
    const uint64_t hc_values = (uint64_t)N_HC * HIDDEN;
    float *hc_norm = (float *)malloc((size_t)hc_values * sizeof(float));
    float pre[N_HC];
    float weights[N_HC];
    float *embd = (float *)malloc((size_t)HIDDEN * sizeof(float));
    float *norm = (float *)malloc((size_t)HIDDEN * sizeof(float));
    if (!hc_norm || !embd || !norm) {
        check(0, "cpu output allocation");
        free(norm);
        free(embd);
        free(hc_norm);
        return;
    }

    cpu_rms_norm_plain(hc_norm, hc, hc_values);
    cpu_f32_matvec(pre,
                   (const float *)(const void *)(model->ptr + hc_head_fn->source_offset),
                   hc_norm,
                   hc_values,
                   N_HC);

    const float *scale = (const float *)(const void *)(model->ptr + hc_head_scale->source_offset);
    const float *base = (const float *)(const void *)(model->ptr + hc_head_base->source_offset);
    for (uint32_t i = 0; i < N_HC; i++) {
        weights[i] = sigmoid_stable(pre[i] * scale[0] + base[i]) + 1.0e-6f;
    }
    cpu_hc_weighted_sum(embd, hc, weights);
    cpu_rms_norm_weight(norm,
                        embd,
                        (const float *)(const void *)(model->ptr + output_norm->source_offset),
                        HIDDEN);

    for (uint32_t i = 0; i < TOPK; i++) {
        tokens[i] = UINT32_MAX;
        logits[i] = -FLT_MAX;
    }
    const uint32_t vocab = (uint32_t)output_weight->shape[1];
    const uint16_t *w = (const uint16_t *)(const void *)(model->ptr + output_weight->source_offset);
    for (uint32_t r = 0; r < vocab; r++) {
        const uint16_t *row = w + (uint64_t)r * HIDDEN;
        double acc = 0.0;
        for (uint32_t c = 0; c < HIDDEN; c++) {
            acc += (double)ds4_src_bf16_to_f32(row[c]) * (double)norm[c];
        }
        insert_topk(tokens, logits, r, (float)acc);
    }

    free(norm);
    free(embd);
    free(hc_norm);
}

int main(int argc, char **argv) {
    const char *index = NULL;
    const char *model_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--index") && i + 1 < argc) {
            index = argv[++i];
        } else if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(stdout);
            return 0;
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
    if (devices < DS4_V100_EXPECTED_GPUS) {
        fprintf(stderr,
                "cuda_v100_output_head_parity_smoke: need 8 CUDA devices, got %d\n",
                devices);
        return 1;
    }

    model_map model;
    if (map_model_file(model_path, &model)) return 1;
    check(ds4_gpu_set_model_fd(model.fd), "model fd");

    char err[512] = {0};
    ds4_v100_context_options ctx_opts;
    ds4_v100_context_options_init(&ctx_opts);
    ctx_opts.pack_index_path = index;
    ds4_v100_context *ctx = NULL;
    if (ds4_v100_context_open(&ctx, &ctx_opts, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_output_head_parity_smoke: %s\n", err);
        unmap_model_file(&model);
        return 1;
    }

    ds4_v100_tensor_binding hc_head_fn;
    ds4_v100_tensor_binding hc_head_scale;
    ds4_v100_tensor_binding hc_head_base;
    ds4_v100_tensor_binding output_norm;
    ds4_v100_tensor_binding output_weight;
    if (ds4_v100_context_lookup_tensor_binding(ctx, "hc_head_fn", &hc_head_fn, err, sizeof(err)) ||
        ds4_v100_context_lookup_tensor_binding(ctx, "hc_head_scale", &hc_head_scale, err, sizeof(err)) ||
        ds4_v100_context_lookup_tensor_binding(ctx, "hc_head_base", &hc_head_base, err, sizeof(err)) ||
        ds4_v100_context_lookup_tensor_binding(ctx, "output_norm.weight", &output_norm, err, sizeof(err)) ||
        ds4_v100_context_output_head_binding(ctx, &output_weight, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_output_head_parity_smoke: %s\n", err);
        ds4_v100_context_close(ctx);
        unmap_model_file(&model);
        return 1;
    }

    ds4_v100_stage_scheduler_options opts;
    ds4_v100_stage_scheduler_options_init(&opts);
    opts.pack_index_path = index;
    opts.model_map = model.ptr;
    opts.model_size = model.size;
    opts.stage_id = DS4_V100_EXPECTED_GPUS - 1;

    ds4_v100_stage_scheduler *sched = NULL;
    if (ds4_v100_stage_scheduler_open(&sched, &opts, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_output_head_parity_smoke: %s\n", err);
        ds4_v100_context_close(ctx);
        unmap_model_file(&model);
        return 1;
    }

    const uint64_t hc_values = (uint64_t)N_HC * HIDDEN;
    float *hc = (float *)malloc((size_t)hc_values * sizeof(float));
    check(hc != NULL, "host HC allocation");
    if (hc) deterministic_hc(hc);

    uint32_t cpu_tokens[TOPK];
    float cpu_logits[TOPK];
    uint32_t gpu_tokens[TOPK];
    float gpu_logits[TOPK];
    for (uint32_t i = 0; i < TOPK; i++) {
        cpu_tokens[i] = gpu_tokens[i] = UINT32_MAX;
        cpu_logits[i] = gpu_logits[i] = 0.0f;
    }
    if (hc) {
        cpu_output_topk(&model,
                        &hc_head_fn,
                        &hc_head_scale,
                        &hc_head_base,
                        &output_norm,
                        &output_weight,
                        hc,
                        cpu_tokens,
                        cpu_logits);
        check(ds4_v100_stage_scheduler_write_hc(sched,
                                                hc,
                                                hc_values * sizeof(float)) != 0,
              "scheduler HC write");
        err[0] = '\0';
        check(ds4_v100_stage_scheduler_select_topk(sched,
                                                   gpu_tokens,
                                                   gpu_logits,
                                                   TOPK,
                                                   err,
                                                   sizeof(err)) == 0,
              err[0] ? err : "scheduler output topk");
    }

    for (uint32_t i = 0; i < TOPK; i++) {
        if (cpu_tokens[i] != gpu_tokens[i]) {
            fprintf(stderr,
                    "cuda_v100_output_head_parity_smoke: top%u token got %" PRIu32
                    " expected %" PRIu32 "\n",
                    i + 1,
                    gpu_tokens[i],
                    cpu_tokens[i]);
            failures++;
        }
        if (fabsf(cpu_logits[i] - gpu_logits[i]) > 5.0e-3f) {
            fprintf(stderr,
                    "cuda_v100_output_head_parity_smoke: top%u logit got %.8g expected %.8g\n",
                    i + 1,
                    gpu_logits[i],
                    cpu_logits[i]);
            failures++;
        }
    }

    printf("cuda_v100_output_head_parity_smoke: cpu_top1=%" PRIu32
           " gpu_top1=%" PRIu32 " cpu_logit=%.8g gpu_logit=%.8g %s\n",
           cpu_tokens[0],
           gpu_tokens[0],
           cpu_logits[0],
           gpu_logits[0],
           failures ? "FAIL" : "ok");

    free(hc);
    ds4_v100_stage_scheduler_close(sched);
    ds4_v100_context_close(ctx);
    unmap_model_file(&model);
    return failures ? 1 : 0;
}
