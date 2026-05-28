#include "ds4_source_formats.h"
#include "engine/context.h"
#include "engine/mtp.h"

#include <errno.h>
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
#include <fcntl.h>

enum {
    MTP_LOGITS_N_EMBD = 4096,
    MTP_LOGITS_N_HC = 4,
    MTP_LOGITS_HC_DIM = MTP_LOGITS_N_EMBD * MTP_LOGITS_N_HC,
    MTP_LOGITS_MAX_TOPK = 16,
};

#define MTP_LOGITS_RMS_EPS 1.0e-6f
#define MTP_LOGITS_HC_EPS  1.0e-6f

typedef struct {
    const char *model;
    const char *mtp_model;
    const char *pack_index;
    const char *report_path;
    int gpu;
    int require_gpus;
    int reserve_mib;
    uint32_t top_k;
    double logit_tol;
} options;

typedef struct {
    const unsigned char *ptr;
    uint64_t size;
    int fd;
} model_map;

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-v100-mtp-logits-smoke --model FILE --mtp-model FILE --pack-index FILE [options]\n"
            "\n"
            "Options:\n"
            "  --index FILE            Alias for --pack-index\n"
            "  --gpu N                 Upload and execute on CUDA device N. Default: 7\n"
            "  --require-gpus N        Require at least N visible CUDA devices\n"
            "  --reserve-mib N         Require this much free memory after upload. Default: 4096\n"
            "  --top-k N               Number of logits candidates to compare. Default: 5\n"
            "  --logit-tol F           Max allowed selected-logit delta. Default: 0.02\n"
            "  --report FILE           Write detailed report to FILE instead of stdout\n");
}

static const char *need_arg(int *i, int argc, char **argv, const char *arg) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: %s requires an argument\n", arg);
        exit(2);
    }
    return argv[++*i];
}

static int parse_int(const char *s, const char *arg) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || !end || *end || v < 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: bad integer for %s: %s\n", arg, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double(const char *s, const char *arg) {
    errno = 0;
    char *end = NULL;
    double v = strtod(s, &end);
    if (errno || !s[0] || !end || *end || !(v >= 0.0)) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: bad float for %s: %s\n", arg, s);
        exit(2);
    }
    return v;
}

static options parse_options(int argc, char **argv) {
    options opt;
    memset(&opt, 0, sizeof(opt));
    opt.gpu = 7;
    opt.reserve_mib = 4096;
    opt.top_k = 5;
    opt.logit_tol = 0.02;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "--model")) {
            opt.model = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-model")) {
            opt.mtp_model = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--pack-index") || !strcmp(arg, "--index")) {
            opt.pack_index = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--report")) {
            opt.report_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--gpu")) {
            opt.gpu = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--require-gpus")) {
            opt.require_gpus = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--reserve-mib")) {
            opt.reserve_mib = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--top-k")) {
            opt.top_k = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--logit-tol")) {
            opt.logit_tol = parse_double(need_arg(&i, argc, argv, arg), arg);
        } else {
            fprintf(stderr, "ds4-v100-mtp-logits-smoke: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }
    if (!opt.model || !opt.model[0] ||
        !opt.mtp_model || !opt.mtp_model[0] ||
        !opt.pack_index || !opt.pack_index[0] ||
        opt.top_k == 0 || opt.top_k > MTP_LOGITS_MAX_TOPK) {
        usage(stderr);
        exit(2);
    }
    return opt;
}

static int map_model_file(const char *path, model_map *out) {
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr,
                "ds4-v100-mtp-logits-smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: cannot stat %s\n", path);
        close(fd);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr,
                "ds4-v100-mtp-logits-smoke: cannot mmap %s: %s\n",
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

static void insert_topk(uint32_t *tokens,
                        float *logits,
                        uint32_t top_k,
                        uint32_t token,
                        float logit) {
    for (uint32_t i = 0; i < top_k; i++) {
        if (tokens[i] == UINT32_MAX || logit > logits[i]) {
            for (uint32_t j = top_k - 1; j > i; j--) {
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
    const uint64_t n = (uint64_t)MTP_LOGITS_N_HC * MTP_LOGITS_N_EMBD;
    for (uint64_t i = 0; i < n; i++) {
        int v = (int)((i * 197u + i / 13u + 17u) % 509u) - 254;
        hc[i] = (float)v * 0.001953125f;
    }
}

static void cpu_rms_norm_plain(float *out, const float *x, uint64_t n) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float scale = 1.0f / sqrtf((float)(ss / (double)n) + MTP_LOGITS_RMS_EPS);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale;
}

static void cpu_rms_norm_weight(float *out,
                                const float *x,
                                const float *w,
                                uint64_t n) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float scale = 1.0f / sqrtf((float)(ss / (double)n) + MTP_LOGITS_RMS_EPS);
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
    for (uint32_t c = 0; c < MTP_LOGITS_N_EMBD; c++) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < MTP_LOGITS_N_HC; h++) {
            acc += hc[(uint64_t)h * MTP_LOGITS_N_EMBD + c] * weights[h];
        }
        out[c] = acc;
    }
}

static const void *tensor_bytes(const ds4_v100_mtp_sidecar *sidecar,
                                const char *name,
                                uint64_t bytes) {
    const ds4_mtp_sidecar_tensor_info *t =
        ds4_v100_mtp_sidecar_tensor(sidecar, name);
    const unsigned char *map = (const unsigned char *)ds4_v100_mtp_sidecar_map(sidecar);
    uint64_t size = ds4_v100_mtp_sidecar_size(sidecar);
    if (!t || !map || t->source_offset > size || bytes > size - t->source_offset ||
        bytes > t->byte_length) {
        return NULL;
    }
    return map + t->source_offset;
}

static int output_bf16_view_from_binding(const ds4_v100_tensor_binding *b,
                                         ds4_gpu_bf16_matrix_view *out,
                                         char *err,
                                         size_t errlen) {
    if (!b || !out) {
        snprintf(err, errlen, "missing output binding");
        return 1;
    }
    if (!b->source_dtype || strcmp(b->source_dtype, "bf16") != 0 ||
        b->n_shape_dims != 2 ||
        b->shape[0] != MTP_LOGITS_N_EMBD ||
        b->shape[1] == 0 ||
        b->shape[1] > UINT32_MAX ||
        b->byte_length != b->shape[0] * b->shape[1] * sizeof(uint16_t)) {
        snprintf(err, errlen, "invalid output.weight bf16 binding");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    out->arena_offset = 0;
    out->byte_length = b->byte_length;
    out->rows = (uint32_t)b->shape[1];
    out->cols = (uint32_t)b->shape[0];
    out->row_stride_elements = (uint32_t)b->shape[0];
    return 0;
}

static int cpu_mtp_logits_topk(const model_map *base,
                               const ds4_v100_mtp_sidecar *sidecar,
                               const ds4_v100_tensor_binding *output_weight,
                               const float *hc,
                               uint32_t top_k,
                               uint32_t *tokens,
                               float *logits) {
    const float *hc_head_fn =
        (const float *)tensor_bytes(sidecar,
                                    "mtp.0.hc_head_fn.weight",
                                    (uint64_t)MTP_LOGITS_N_HC *
                                        MTP_LOGITS_HC_DIM * sizeof(float));
    const float *hc_head_scale =
        (const float *)tensor_bytes(sidecar,
                                    "mtp.0.hc_head_scale.weight",
                                    sizeof(float));
    const float *hc_head_base =
        (const float *)tensor_bytes(sidecar,
                                    "mtp.0.hc_head_base.weight",
                                    (uint64_t)MTP_LOGITS_N_HC * sizeof(float));
    const float *norm_weight =
        (const float *)tensor_bytes(sidecar,
                                    "mtp.0.norm.weight",
                                    (uint64_t)MTP_LOGITS_N_EMBD * sizeof(float));
    if (!base || !base->ptr || !sidecar || !output_weight || !hc_head_fn ||
        !hc_head_scale || !hc_head_base || !norm_weight ||
        output_weight->source_offset > base->size ||
        output_weight->byte_length > base->size - output_weight->source_offset) {
        return 1;
    }

    float *hc_norm = (float *)malloc((size_t)MTP_LOGITS_HC_DIM * sizeof(float));
    float *embd = (float *)malloc((size_t)MTP_LOGITS_N_EMBD * sizeof(float));
    float *norm = (float *)malloc((size_t)MTP_LOGITS_N_EMBD * sizeof(float));
    if (!hc_norm || !embd || !norm) {
        free(norm);
        free(embd);
        free(hc_norm);
        return 1;
    }

    float pre[MTP_LOGITS_N_HC];
    float weights[MTP_LOGITS_N_HC];
    cpu_rms_norm_plain(hc_norm, hc, MTP_LOGITS_HC_DIM);
    cpu_f32_matvec(pre,
                   hc_head_fn,
                   hc_norm,
                   MTP_LOGITS_HC_DIM,
                   MTP_LOGITS_N_HC);
    for (uint32_t i = 0; i < MTP_LOGITS_N_HC; i++) {
        weights[i] = sigmoid_stable(pre[i] * hc_head_scale[0] + hc_head_base[i]) +
                     MTP_LOGITS_HC_EPS;
    }
    cpu_hc_weighted_sum(embd, hc, weights);
    cpu_rms_norm_weight(norm, embd, norm_weight, MTP_LOGITS_N_EMBD);

    for (uint32_t i = 0; i < top_k; i++) {
        tokens[i] = UINT32_MAX;
        logits[i] = -FLT_MAX;
    }

    const uint32_t vocab = (uint32_t)output_weight->shape[1];
    const uint16_t *w =
        (const uint16_t *)(const void *)(base->ptr + output_weight->source_offset);
    for (uint32_t r = 0; r < vocab; r++) {
        const uint16_t *row = w + (uint64_t)r * MTP_LOGITS_N_EMBD;
        double acc = 0.0;
        for (uint32_t c = 0; c < MTP_LOGITS_N_EMBD; c++) {
            acc += (double)ds4_src_bf16_to_f32(row[c]) * (double)norm[c];
        }
        insert_topk(tokens, logits, top_k, r, (float)acc);
    }

    free(norm);
    free(embd);
    free(hc_norm);
    return 0;
}

static int arena_upload_chunks(ds4_gpu_arena *arena,
                               uint64_t offset,
                               const void *src,
                               uint64_t bytes) {
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    uint64_t done = 0;
    while (done < bytes) {
        uint64_t n = bytes - done;
        if (n > chunk) n = chunk;
        if (ds4_gpu_arena_upload(arena,
                                 offset + done,
                                 (const unsigned char *)src + done,
                                 n) != 0) {
            return 1;
        }
        done += n;
    }
    return 0;
}

static void topk_from_logits(const float *all_logits,
                             uint32_t vocab,
                             uint32_t top_k,
                             uint32_t *tokens,
                             float *logits) {
    for (uint32_t i = 0; i < top_k; i++) {
        tokens[i] = UINT32_MAX;
        logits[i] = -FLT_MAX;
    }
    for (uint32_t i = 0; i < vocab; i++) {
        if (isfinite(all_logits[i])) {
            insert_topk(tokens, logits, top_k, i, all_logits[i]);
        }
    }
}

int main(int argc, char **argv) {
    options opt = parse_options(argc, argv);
    FILE *report = stdout;
    if (opt.report_path) {
        report = fopen(opt.report_path, "w");
        if (!report) {
            fprintf(stderr,
                    "ds4-v100-mtp-logits-smoke: cannot open report %s: %s\n",
                    opt.report_path,
                    strerror(errno));
            return 1;
        }
    }

    int rc = 1;
    char err[512];
    err[0] = '\0';
    model_map base_map;
    memset(&base_map, 0, sizeof(base_map));
    base_map.fd = -1;
    ds4_v100_context *ctx = NULL;
    ds4_v100_mtp_sidecar *sidecar = NULL;
    ds4_gpu_arena *output_arena = NULL;
    ds4_gpu_tensor *hc_d = NULL;
    ds4_gpu_tensor *hc_norm_d = NULL;
    ds4_gpu_tensor *head_pre_d = NULL;
    ds4_gpu_tensor *head_weights_d = NULL;
    ds4_gpu_tensor *embd_d = NULL;
    ds4_gpu_tensor *norm_d = NULL;
    ds4_gpu_tensor *logits_d = NULL;
    float *hc = NULL;
    float *gpu_all_logits = NULL;
    uint32_t *cpu_tokens = NULL;
    uint32_t *gpu_tokens = NULL;
    float *cpu_logits = NULL;
    float *gpu_logits = NULL;

    fprintf(report, "model\t%s\n", opt.model);
    fprintf(report, "mtp_model\t%s\n", opt.mtp_model);
    fprintf(report, "pack_index\t%s\n", opt.pack_index);
    fprintf(report, "gpu\t%d\n", opt.gpu);
    fprintf(report, "require_gpus\t%d\n", opt.require_gpus);
    fprintf(report, "reserve_mib\t%d\n", opt.reserve_mib);
    fprintf(report, "top_k\t%" PRIu32 "\n", opt.top_k);
    fprintf(report, "logit_tol\t%.8g\n", opt.logit_tol);

    if (map_model_file(opt.model, &base_map) != 0) goto done;

    ds4_v100_context_options ctx_opts;
    ds4_v100_context_options_init(&ctx_opts);
    ctx_opts.pack_index_path = opt.pack_index;
    if (ds4_v100_context_open(&ctx, &ctx_opts, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: %s\n", err);
        goto done;
    }

    ds4_v100_tensor_binding output_weight;
    if (ds4_v100_context_output_head_binding(ctx, &output_weight, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: %s\n", err);
        goto done;
    }
    ds4_gpu_bf16_matrix_view output_view;
    if (output_bf16_view_from_binding(&output_weight, &output_view, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: %s\n", err);
        goto done;
    }
    if (output_weight.source_offset > base_map.size ||
        output_weight.byte_length > base_map.size - output_weight.source_offset) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: output.weight outside model map\n");
        goto done;
    }

    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: ds4_gpu_init failed\n");
        goto done;
    }
    int n_dev = ds4_gpu_device_count();
    fprintf(report, "visible_cuda_devices\t%d\n", n_dev);
    if (opt.require_gpus > 0 && n_dev < opt.require_gpus) {
        fprintf(stderr,
                "ds4-v100-mtp-logits-smoke: need %d CUDA devices, got %d\n",
                opt.require_gpus,
                n_dev);
        goto done;
    }
    if (!ds4_gpu_set_device(opt.gpu)) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: failed to set gpu%d\n", opt.gpu);
        goto done;
    }

    ds4_v100_mtp_sidecar_options mtp_opts;
    ds4_v100_mtp_sidecar_options_init(&mtp_opts);
    mtp_opts.mtp_path = opt.mtp_model;
    mtp_opts.gpu = opt.gpu;
    mtp_opts.require_device_arena = true;
    if (ds4_v100_mtp_sidecar_open(&sidecar, &mtp_opts, report, err, sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-logits-smoke: %s\n",
                err[0] ? err : "failed to open MTP sidecar");
        goto done;
    }

    ds4_gpu_source_row_view hc_fn_view;
    ds4_gpu_source_row_view hc_scale_view;
    ds4_gpu_source_row_view hc_base_view;
    ds4_gpu_source_row_view norm_view;
    if (ds4_v100_mtp_sidecar_f32_matrix_view(sidecar,
                                             "mtp.0.hc_head_fn.weight",
                                             &hc_fn_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.hc_head_scale.weight",
                                             &hc_scale_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.hc_head_base.weight",
                                             &hc_base_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.norm.weight",
                                             &norm_view,
                                             err,
                                             sizeof(err)) != 0) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: %s\n", err);
        goto done;
    }
    if (hc_fn_view.rows != MTP_LOGITS_N_HC ||
        hc_fn_view.cols != MTP_LOGITS_HC_DIM ||
        hc_scale_view.cols != 1 ||
        hc_base_view.cols != MTP_LOGITS_N_HC ||
        norm_view.cols != MTP_LOGITS_N_EMBD) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: invalid MTP output tensor shapes\n");
        goto done;
    }

    if (ds4_gpu_arena_open(&output_arena, opt.gpu, output_weight.byte_length) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-logits-smoke: failed to allocate output arena bytes=%" PRIu64 "\n",
                output_weight.byte_length);
        goto done;
    }
    if (!ds4_gpu_arena_is_device_memory(output_arena)) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: output arena is not device memory\n");
        goto done;
    }
    if (arena_upload_chunks(output_arena,
                            0,
                            base_map.ptr + output_weight.source_offset,
                            output_weight.byte_length) != 0) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: output.weight upload failed\n");
        goto done;
    }
    const uint64_t reserve_bytes = (uint64_t)opt.reserve_mib * 1024ull * 1024ull;
    const uint64_t free_after = ds4_gpu_arena_free_after_upload_bytes(output_arena);
    fprintf(report, "output_weight_bytes\t%" PRIu64 "\n", output_weight.byte_length);
    fprintf(report, "output_vocab\t%" PRIu32 "\n", output_view.rows);
    fprintf(report, "reserve_bytes\t%" PRIu64 "\n", reserve_bytes);
    fprintf(report, "free_after_output_upload_bytes\t%" PRIu64 "\n", free_after);
    if (free_after < reserve_bytes) {
        fprintf(stderr,
                "ds4-v100-mtp-logits-smoke: free_after_output_upload %" PRIu64
                " below reserve %" PRIu64 "\n",
                free_after,
                reserve_bytes);
        goto done;
    }

    const uint64_t hc_bytes = (uint64_t)MTP_LOGITS_HC_DIM * sizeof(float);
    const uint64_t embd_bytes = (uint64_t)MTP_LOGITS_N_EMBD * sizeof(float);
    const uint64_t weights_bytes = (uint64_t)MTP_LOGITS_N_HC * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)output_view.rows * sizeof(float);

    hc = (float *)malloc((size_t)hc_bytes);
    gpu_all_logits = (float *)malloc((size_t)logits_bytes);
    cpu_tokens = (uint32_t *)malloc((size_t)opt.top_k * sizeof(uint32_t));
    gpu_tokens = (uint32_t *)malloc((size_t)opt.top_k * sizeof(uint32_t));
    cpu_logits = (float *)malloc((size_t)opt.top_k * sizeof(float));
    gpu_logits = (float *)malloc((size_t)opt.top_k * sizeof(float));
    if (!hc || !gpu_all_logits || !cpu_tokens || !gpu_tokens ||
        !cpu_logits || !gpu_logits) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: host allocation failed\n");
        goto done;
    }
    deterministic_hc(hc);

    if (cpu_mtp_logits_topk(&base_map,
                            sidecar,
                            &output_weight,
                            hc,
                            opt.top_k,
                            cpu_tokens,
                            cpu_logits) != 0) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: CPU MTP logits oracle failed\n");
        goto done;
    }

    hc_d = ds4_gpu_tensor_alloc(hc_bytes);
    hc_norm_d = ds4_gpu_tensor_alloc(hc_bytes);
    head_pre_d = ds4_gpu_tensor_alloc(weights_bytes);
    head_weights_d = ds4_gpu_tensor_alloc(weights_bytes);
    embd_d = ds4_gpu_tensor_alloc(embd_bytes);
    norm_d = ds4_gpu_tensor_alloc(embd_bytes);
    logits_d = ds4_gpu_tensor_alloc(logits_bytes);
    if (!hc_d || !hc_norm_d || !head_pre_d || !head_weights_d ||
        !embd_d || !norm_d || !logits_d) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: device tensor allocation failed\n");
        goto done;
    }
    if (!ds4_gpu_tensor_write(hc_d, 0, hc, hc_bytes) ||
        !ds4_gpu_rms_norm_plain_tensor(hc_norm_d,
                                       hc_d,
                                       MTP_LOGITS_HC_DIM,
                                       MTP_LOGITS_RMS_EPS) ||
        ds4_gpu_arena_f32_matmul_f32(ds4_v100_mtp_sidecar_arena(sidecar),
                                     &hc_fn_view,
                                     hc_norm_d,
                                     head_pre_d) != 0 ||
        ds4_gpu_arena_output_hc_weights_tensor(ds4_v100_mtp_sidecar_arena(sidecar),
                                               &hc_scale_view,
                                               &hc_base_view,
                                               head_weights_d,
                                               head_pre_d,
                                               MTP_LOGITS_N_HC,
                                               MTP_LOGITS_HC_EPS) != 0 ||
        !ds4_gpu_hc_weighted_sum_tensor(embd_d,
                                        hc_d,
                                        head_weights_d,
                                        MTP_LOGITS_N_EMBD,
                                        MTP_LOGITS_N_HC) ||
        ds4_gpu_arena_f32_rms_norm_f32(ds4_v100_mtp_sidecar_arena(sidecar),
                                       &norm_view,
                                       embd_d,
                                       norm_d,
                                       MTP_LOGITS_N_EMBD,
                                       1,
                                       MTP_LOGITS_RMS_EPS) != 0 ||
        ds4_gpu_arena_bf16_matmul_f32(output_arena,
                                      &output_view,
                                      norm_d,
                                      logits_d) != 0 ||
        !ds4_gpu_tensor_read(logits_d, 0, gpu_all_logits, logits_bytes)) {
        fprintf(stderr, "ds4-v100-mtp-logits-smoke: GPU MTP logits sequence failed\n");
        goto done;
    }

    topk_from_logits(gpu_all_logits,
                     output_view.rows,
                     opt.top_k,
                     gpu_tokens,
                     gpu_logits);

    double max_abs = 0.0;
    int failures = 0;
    for (uint32_t i = 0; i < opt.top_k; i++) {
        double delta = fabs((double)cpu_logits[i] - (double)gpu_logits[i]);
        if (delta > max_abs) max_abs = delta;
        fprintf(report,
                "mtp_logits_topk\trank=%" PRIu32
                "\tcpu_token=%" PRIu32 "\tgpu_token=%" PRIu32
                "\tcpu_logit=%.9g\tgpu_logit=%.9g\tdelta=%.9g\n",
                i + 1,
                cpu_tokens[i],
                gpu_tokens[i],
                cpu_logits[i],
                gpu_logits[i],
                delta);
        if (cpu_tokens[i] != gpu_tokens[i] || delta > opt.logit_tol) {
            failures++;
        }
    }

    fprintf(report,
            "mtp_logits_summary\ttop1_cpu=%" PRIu32
            "\ttop1_gpu=%" PRIu32 "\tmax_abs=%.9g\t%s\n",
            cpu_tokens[0],
            gpu_tokens[0],
            max_abs,
            failures ? "FAIL" : "PASS");
    printf("mtp_logits_smoke: cpu_top1=%" PRIu32
           " gpu_top1=%" PRIu32 " max_abs=%.9g %s\n",
           cpu_tokens[0],
           gpu_tokens[0],
           max_abs,
           failures ? "FAIL" : "PASS");
    if (failures) goto done;

    rc = 0;

done:
    ds4_gpu_tensor_free(logits_d);
    ds4_gpu_tensor_free(norm_d);
    ds4_gpu_tensor_free(embd_d);
    ds4_gpu_tensor_free(head_weights_d);
    ds4_gpu_tensor_free(head_pre_d);
    ds4_gpu_tensor_free(hc_norm_d);
    ds4_gpu_tensor_free(hc_d);
    ds4_gpu_arena_close(output_arena);
    ds4_v100_mtp_sidecar_close(sidecar);
    ds4_v100_context_close(ctx);
    unmap_model_file(&base_map);
    free(gpu_logits);
    free(cpu_logits);
    free(gpu_tokens);
    free(cpu_tokens);
    free(gpu_all_logits);
    free(hc);
    ds4_gpu_cleanup();
    if (report && report != stdout) fclose(report);
    return rc;
}
