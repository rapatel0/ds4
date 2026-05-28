#include "engine/mtp.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    MTP_PREFIX_HC_ROWS = 4,
};

#define MTP_PREFIX_RMS_EPS 1.0e-6f

typedef struct {
    const char *mtp_model;
    const char *report_path;
    int gpu;
    int require_gpus;
    int reserve_mib;
    double max_abs_tol;
} options;

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-v100-mtp-prefix-smoke --mtp-model FILE [options]\n"
            "\n"
            "Options:\n"
            "  --gpu N                 Upload and execute on CUDA device N. Default: 7\n"
            "  --require-gpus N        Require at least N visible CUDA devices\n"
            "  --reserve-mib N         Require this much free memory after upload. Default: 4096\n"
            "  --max-abs-tol F         Max allowed absolute output delta. Default: 1e-5\n"
            "  --report FILE           Write report to FILE instead of stdout\n");
}

static const char *need_arg(int *i, int argc, char **argv, const char *arg) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: %s requires an argument\n", arg);
        exit(2);
    }
    return argv[++*i];
}

static int parse_int(const char *s, const char *arg) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || !end || *end || v < 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: bad integer for %s: %s\n", arg, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double(const char *s, const char *arg) {
    errno = 0;
    char *end = NULL;
    double v = strtod(s, &end);
    if (errno || !s[0] || !end || *end || !(v >= 0.0)) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: bad float for %s: %s\n", arg, s);
        exit(2);
    }
    return v;
}

static options parse_options(int argc, char **argv) {
    options opt;
    memset(&opt, 0, sizeof(opt));
    opt.gpu = 7;
    opt.reserve_mib = 4096;
    opt.max_abs_tol = 1e-5;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "--mtp-model")) {
            opt.mtp_model = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--report")) {
            opt.report_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--gpu")) {
            opt.gpu = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--require-gpus")) {
            opt.require_gpus = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--reserve-mib")) {
            opt.reserve_mib = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--max-abs-tol")) {
            opt.max_abs_tol = parse_double(need_arg(&i, argc, argv, arg), arg);
        } else {
            fprintf(stderr, "ds4-v100-mtp-prefix-smoke: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }
    if (!opt.mtp_model || !opt.mtp_model[0]) {
        usage(stderr);
        exit(2);
    }
    return opt;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void fill_activation(float *x, uint64_t n_tok, uint32_t cols, uint32_t salt) {
    for (uint64_t tok = 0; tok < n_tok; tok++) {
        for (uint32_t c = 0; c < cols; c++) {
            uint32_t v = (uint32_t)((uint64_t)c * 17u + tok * 31u + salt * 43u);
            int centered = (int)(v % 257u) - 128;
            x[tok * cols + c] = (float)centered / 97.0f;
        }
    }
}

static int compare_outputs(const float *got,
                           const float *ref,
                           uint64_t n,
                           double *max_abs_out,
                           double *max_rel_out,
                           uint64_t *max_i_out) {
    double max_abs = 0.0;
    double max_rel = 0.0;
    uint64_t max_i = 0;
    for (uint64_t i = 0; i < n; i++) {
        double diff = fabs((double)got[i] - (double)ref[i]);
        double denom = fabs((double)ref[i]);
        double rel = denom > 1e-12 ? diff / denom : diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_i = i;
        }
        if (rel > max_rel) max_rel = rel;
    }
    if (max_abs_out) *max_abs_out = max_abs;
    if (max_rel_out) *max_rel_out = max_rel;
    if (max_i_out) *max_i_out = max_i;
    return 0;
}

static void rms_norm_weight_host(float *out,
                                 const float *x,
                                 const float *weight,
                                 uint32_t n,
                                 uint32_t rows,
                                 float eps) {
    for (uint32_t row = 0; row < rows; row++) {
        const float *xr = x + (uint64_t)row * n;
        float *orow = out + (uint64_t)row * n;
        double ss = 0.0;
        for (uint32_t i = 0; i < n; i++) ss += (double)xr[i] * (double)xr[i];
        const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
        for (uint32_t i = 0; i < n; i++) orow[i] = xr[i] * scale * weight[i];
    }
}

static float f16_to_f32_host(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127u - 15u + 1u;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float f = 0.0f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void quantize_q8_0_activation_host(const float *x,
                                          int8_t *xq,
                                          float *scale,
                                          uint64_t n) {
    const uint64_t blocks = (n + 31u) / 32u;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = n - i0 < 32u ? n - i0 : 32u;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) {
            const float ax = fabsf(x[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scale[b] = d;
        for (uint64_t i = 0; i < bn; i++) {
            int v = (int)lrintf(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        for (uint64_t i = bn; i < 32u; i++) xq[i0 + i] = 0;
    }
}

static int32_t dot_i8_host(const int8_t *a, const int8_t *b, uint64_t n) {
    int32_t sum = 0;
    for (uint64_t i = 0; i < n; i++) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

static int matmul_q8_0_host(float *out,
                            const unsigned char *weight,
                            uint32_t in_dim,
                            uint32_t out_dim,
                            const float *x,
                            uint64_t n_tok) {
    const uint64_t blocks = ((uint64_t)in_dim + 31u) / 32u;
    const uint64_t row_stride = blocks * 34u;
    int8_t *xq = (int8_t *)malloc((size_t)(n_tok * blocks * 32u));
    float *xscale = (float *)malloc((size_t)(n_tok * blocks * sizeof(float)));
    if (!xq || !xscale) {
        free(xq);
        free(xscale);
        return 1;
    }
    for (uint64_t tok = 0; tok < n_tok; tok++) {
        quantize_q8_0_activation_host(x + tok * in_dim,
                                      xq + tok * blocks * 32u,
                                      xscale + tok * blocks,
                                      in_dim);
    }
    for (uint64_t tok = 0; tok < n_tok; tok++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            const unsigned char *wrow = weight + (uint64_t)row * row_stride;
            float acc = 0.0f;
            for (uint64_t b = 0; b < blocks; b++) {
                uint16_t scale_bits = 0;
                memcpy(&scale_bits, wrow + b * 34u, sizeof(scale_bits));
                const int8_t *qs = (const int8_t *)(wrow + b * 34u + 2u);
                const uint64_t i0 = b * 32u;
                const uint64_t n = in_dim - i0 < 32u ? in_dim - i0 : 32u;
                acc += f16_to_f32_host(scale_bits) *
                       xscale[tok * blocks + b] *
                       (float)dot_i8_host(qs, xq + tok * blocks * 32u + i0, n);
            }
            out[tok * out_dim + row] = acc;
        }
    }
    free(xscale);
    free(xq);
    return 0;
}

static unsigned char *copy_sidecar_tensor(ds4_v100_mtp_sidecar *sidecar,
                                          const ds4_mtp_sidecar_tensor_info *tensor) {
    if (!sidecar || !tensor || tensor->byte_length > SIZE_MAX) return NULL;
    unsigned char *copy = (unsigned char *)malloc((size_t)tensor->byte_length);
    if (!copy) return NULL;
    memcpy(copy,
           (const unsigned char *)ds4_v100_mtp_sidecar_map(sidecar) + tensor->source_offset,
           (size_t)tensor->byte_length);
    return copy;
}

static int compare_tensor_pair(const char *label,
                               const ds4_gpu_tensor *got_t,
                               const ds4_gpu_tensor *ref_t,
                               uint64_t values,
                               double max_abs_tol,
                               FILE *report) {
    const uint64_t bytes = values * sizeof(float);
    float *got = (float *)malloc((size_t)bytes);
    float *ref = (float *)malloc((size_t)bytes);
    if (!got || !ref) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: host compare allocation failed for %s\n", label);
        free(got);
        free(ref);
        return 1;
    }
    if (!ds4_gpu_tensor_read(got_t, 0, got, bytes) ||
        !ds4_gpu_tensor_read(ref_t, 0, ref, bytes)) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: compare readback failed for %s\n", label);
        free(got);
        free(ref);
        return 1;
    }

    double max_abs = 0.0;
    double max_rel = 0.0;
    uint64_t max_i = 0;
    compare_outputs(got, ref, values, &max_abs, &max_rel, &max_i);
    fprintf(report,
            "mtp_prefix_chain_tensor\t%s\tvalues=%" PRIu64
            "\tmax_abs=%.9g\tmax_rel=%.9g\tmax_i=%" PRIu64 "\t%s\n",
            label,
            values,
            max_abs,
            max_rel,
            max_i,
            max_abs <= max_abs_tol ? "PASS" : "FAIL");

    free(got);
    free(ref);
    if (max_abs > max_abs_tol) {
        fprintf(stderr,
                "ds4-v100-mtp-prefix-smoke: %s max_abs %.9g exceeds %.9g\n",
                label,
                max_abs,
                max_abs_tol);
        return 1;
    }
    return 0;
}

static int run_projection(ds4_v100_mtp_sidecar *sidecar,
                          const char *name,
                          uint64_t n_tok,
                          uint32_t salt,
                          double max_abs_tol,
                          FILE *report) {
    char err[512] = {0};
    ds4_gpu_source_row_view view;
    if (ds4_v100_mtp_sidecar_q8_0_view(sidecar, name, &view, err, sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-prefix-smoke: %s\n",
                err[0] ? err : "failed to bind Q8_0 view");
        return 1;
    }

    const ds4_mtp_sidecar_tensor_info *tensor =
        ds4_v100_mtp_sidecar_tensor(sidecar, name);
    const uint64_t x_values = n_tok * (uint64_t)view.cols;
    const uint64_t y_values = n_tok * (uint64_t)view.rows;
    unsigned char *weight_copy = (unsigned char *)malloc((size_t)tensor->byte_length);
    float *x = (float *)malloc((size_t)(x_values * sizeof(float)));
    float *arena_out = (float *)malloc((size_t)(y_values * sizeof(float)));
    float *ref_out = (float *)malloc((size_t)(y_values * sizeof(float)));
    ds4_gpu_tensor *x_t = NULL;
    ds4_gpu_tensor *arena_t = NULL;
    ds4_gpu_tensor *ref_t = NULL;
    if (!weight_copy || !x || !arena_out || !ref_out) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: host allocation failed\n");
        free(weight_copy);
        free(x);
        free(arena_out);
        free(ref_out);
        return 1;
    }
    memcpy(weight_copy,
           (const unsigned char *)ds4_v100_mtp_sidecar_map(sidecar) + tensor->source_offset,
           (size_t)tensor->byte_length);
    fill_activation(x, n_tok, view.cols, salt);

    int rc = 1;
    const uint64_t x_bytes = x_values * sizeof(float);
    const uint64_t y_bytes = y_values * sizeof(float);
    x_t = ds4_gpu_tensor_alloc(x_bytes);
    arena_t = ds4_gpu_tensor_alloc(y_bytes);
    ref_t = ds4_gpu_tensor_alloc(y_bytes);
    if (!x_t || !arena_t || !ref_t) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: device tensor allocation failed\n");
        goto done;
    }
    if (!ds4_gpu_tensor_write(x_t, 0, x, x_bytes)) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: activation upload failed\n");
        goto done;
    }

    double t0 = now_ms();
    if (ds4_gpu_arena_q8_0_matmul_f32(ds4_v100_mtp_sidecar_arena(sidecar),
                                      &view,
                                      x_t,
                                      arena_t,
                                      n_tok) != 0 ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: resident Q8_0 matmul failed for %s\n", name);
        goto done;
    }
    double t1 = now_ms();
    if (!ds4_gpu_matmul_q8_0_tensor(ref_t,
                                    weight_copy,
                                    tensor->byte_length,
                                    0,
                                    view.cols,
                                    view.rows,
                                    x_t,
                                    n_tok) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: mapped Q8_0 reference failed for %s\n", name);
        goto done;
    }
    double t2 = now_ms();

    if (!ds4_gpu_tensor_read(arena_t, 0, arena_out, y_bytes) ||
        !ds4_gpu_tensor_read(ref_t, 0, ref_out, y_bytes)) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: output readback failed for %s\n", name);
        goto done;
    }

    double max_abs = 0.0;
    double max_rel = 0.0;
    uint64_t max_i = 0;
    compare_outputs(arena_out, ref_out, y_values, &max_abs, &max_rel, &max_i);
    fprintf(report,
            "mtp_prefix_tensor\t%s\tdtype=%s\trows=%u\tcols=%u\trow_stride=%u"
            "\tn_tok=%" PRIu64 "\tsource_offset=%" PRIu64 "\tresident_offset=%" PRIu64
            "\tbytes=%" PRIu64 "\tarena_ms=%.3f\treference_ms=%.3f"
            "\tmax_abs=%.9g\tmax_rel=%.9g\tmax_i=%" PRIu64 "\t%s\n",
            name,
            tensor->dtype,
            view.rows,
            view.cols,
            view.row_stride_bytes,
            n_tok,
            tensor->source_offset,
            tensor->resident_offset,
            tensor->byte_length,
            t1 - t0,
            t2 - t1,
            max_abs,
            max_rel,
            max_i,
            max_abs <= max_abs_tol ? "PASS" : "FAIL");
    if (max_abs > max_abs_tol) {
        fprintf(stderr,
                "ds4-v100-mtp-prefix-smoke: %s max_abs %.9g exceeds %.9g\n",
                name,
                max_abs,
                max_abs_tol);
        goto done;
    }

    rc = 0;

done:
    ds4_gpu_tensor_free(ref_t);
    ds4_gpu_tensor_free(arena_t);
    ds4_gpu_tensor_free(x_t);
    free(ref_out);
    free(arena_out);
    free(x);
    free(weight_copy);
    return rc;
}

static int run_prefix_chain(ds4_v100_mtp_sidecar *sidecar,
                            double max_abs_tol,
                            FILE *report) {
    char err[512] = {0};
    ds4_gpu_source_row_view enorm_view;
    ds4_gpu_source_row_view hnorm_view;
    ds4_gpu_source_row_view e_proj_view;
    ds4_gpu_source_row_view h_proj_view;

    if (ds4_v100_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.enorm.weight",
                                             &enorm_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.hnorm.weight",
                                             &hnorm_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_q8_0_view(sidecar,
                                       "mtp.0.e_proj.weight",
                                       &e_proj_view,
                                       err,
                                       sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_q8_0_view(sidecar,
                                       "mtp.0.h_proj.weight",
                                       &h_proj_view,
                                       err,
                                       sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-prefix-smoke: %s\n",
                err[0] ? err : "failed to bind prefix chain views");
        return 1;
    }

    const uint32_t n_embd = e_proj_view.cols;
    if (e_proj_view.rows != n_embd ||
        h_proj_view.rows != n_embd ||
        h_proj_view.cols != n_embd ||
        enorm_view.cols != n_embd ||
        hnorm_view.cols != n_embd) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: inconsistent prefix tensor dimensions\n");
        return 1;
    }

    const ds4_mtp_sidecar_tensor_info *enorm_tensor =
        ds4_v100_mtp_sidecar_tensor(sidecar, "mtp.0.enorm.weight");
    const ds4_mtp_sidecar_tensor_info *hnorm_tensor =
        ds4_v100_mtp_sidecar_tensor(sidecar, "mtp.0.hnorm.weight");
    const ds4_mtp_sidecar_tensor_info *e_proj_tensor =
        ds4_v100_mtp_sidecar_tensor(sidecar, "mtp.0.e_proj.weight");
    const ds4_mtp_sidecar_tensor_info *h_proj_tensor =
        ds4_v100_mtp_sidecar_tensor(sidecar, "mtp.0.h_proj.weight");
    unsigned char *enorm_copy = copy_sidecar_tensor(sidecar, enorm_tensor);
    unsigned char *hnorm_copy = copy_sidecar_tensor(sidecar, hnorm_tensor);
    unsigned char *e_proj_copy = copy_sidecar_tensor(sidecar, e_proj_tensor);
    unsigned char *h_proj_copy = copy_sidecar_tensor(sidecar, h_proj_tensor);

    const uint64_t row_values = n_embd;
    const uint64_t hc_values = (uint64_t)MTP_PREFIX_HC_ROWS * n_embd;
    const uint64_t row_bytes = row_values * sizeof(float);
    const uint64_t hc_bytes = hc_values * sizeof(float);
    float *embed = (float *)malloc((size_t)row_bytes);
    float *prev_hc = (float *)malloc((size_t)hc_bytes);
    float *ref_enorm_host = (float *)malloc((size_t)row_bytes);
    float *ref_hnorm_host = (float *)malloc((size_t)hc_bytes);
    float *ref_eproj_host = (float *)malloc((size_t)row_bytes);
    float *ref_eproj_hc_host = (float *)malloc((size_t)hc_bytes);
    float *ref_hproj_host = (float *)malloc((size_t)hc_bytes);
    float *ref_prefix_host = (float *)malloc((size_t)hc_bytes);

    ds4_gpu_tensor *embed_t = NULL;
    ds4_gpu_tensor *prev_hc_t = NULL;
    ds4_gpu_tensor *arena_enorm = NULL;
    ds4_gpu_tensor *arena_eproj = NULL;
    ds4_gpu_tensor *arena_eproj_hc = NULL;
    ds4_gpu_tensor *arena_hnorm = NULL;
    ds4_gpu_tensor *arena_hproj = NULL;
    ds4_gpu_tensor *arena_prefix = NULL;
    ds4_gpu_tensor *ref_enorm = NULL;
    ds4_gpu_tensor *ref_eproj = NULL;
    ds4_gpu_tensor *ref_eproj_hc = NULL;
    ds4_gpu_tensor *ref_hnorm = NULL;
    ds4_gpu_tensor *ref_hproj = NULL;
    ds4_gpu_tensor *ref_prefix = NULL;

    int rc = 1;
    if (!enorm_copy || !hnorm_copy || !e_proj_copy || !h_proj_copy ||
        !embed || !prev_hc || !ref_enorm_host || !ref_hnorm_host ||
        !ref_eproj_host || !ref_eproj_hc_host || !ref_hproj_host ||
        !ref_prefix_host) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: prefix chain host allocation failed\n");
        goto done;
    }
    fill_activation(embed, 1, n_embd, 23);
    fill_activation(prev_hc, MTP_PREFIX_HC_ROWS, n_embd, 29);
    rms_norm_weight_host(ref_enorm_host,
                         embed,
                         (const float *)enorm_copy,
                         n_embd,
                         1,
                         MTP_PREFIX_RMS_EPS);
    rms_norm_weight_host(ref_hnorm_host,
                         prev_hc,
                         (const float *)hnorm_copy,
                         n_embd,
                         MTP_PREFIX_HC_ROWS,
                         MTP_PREFIX_RMS_EPS);
    if (matmul_q8_0_host(ref_eproj_host,
                         e_proj_copy,
                         e_proj_view.cols,
                         e_proj_view.rows,
                         ref_enorm_host,
                         1) != 0) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: host e_proj reference failed\n");
        goto done;
    }
    for (uint32_t row = 0; row < MTP_PREFIX_HC_ROWS; row++) {
        memcpy(ref_eproj_hc_host + (uint64_t)row * n_embd,
               ref_eproj_host,
               (size_t)row_bytes);
    }
    if (matmul_q8_0_host(ref_hproj_host,
                         h_proj_copy,
                         h_proj_view.cols,
                         h_proj_view.rows,
                         ref_hnorm_host,
                         MTP_PREFIX_HC_ROWS) != 0) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: host h_proj reference failed\n");
        goto done;
    }
    for (uint64_t i = 0; i < hc_values; i++) {
        ref_prefix_host[i] = ref_eproj_hc_host[i] + ref_hproj_host[i];
    }

    embed_t = ds4_gpu_tensor_alloc(row_bytes);
    prev_hc_t = ds4_gpu_tensor_alloc(hc_bytes);
    arena_enorm = ds4_gpu_tensor_alloc(row_bytes);
    arena_eproj = ds4_gpu_tensor_alloc(row_bytes);
    arena_eproj_hc = ds4_gpu_tensor_alloc(hc_bytes);
    arena_hnorm = ds4_gpu_tensor_alloc(hc_bytes);
    arena_hproj = ds4_gpu_tensor_alloc(hc_bytes);
    arena_prefix = ds4_gpu_tensor_alloc(hc_bytes);
    ref_enorm = ds4_gpu_tensor_alloc(row_bytes);
    ref_eproj = ds4_gpu_tensor_alloc(row_bytes);
    ref_eproj_hc = ds4_gpu_tensor_alloc(hc_bytes);
    ref_hnorm = ds4_gpu_tensor_alloc(hc_bytes);
    ref_hproj = ds4_gpu_tensor_alloc(hc_bytes);
    ref_prefix = ds4_gpu_tensor_alloc(hc_bytes);
    if (!embed_t || !prev_hc_t || !arena_enorm || !arena_eproj ||
        !arena_eproj_hc || !arena_hnorm || !arena_hproj || !arena_prefix ||
        !ref_enorm || !ref_eproj || !ref_eproj_hc || !ref_hnorm ||
        !ref_hproj || !ref_prefix) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: prefix chain device allocation failed\n");
        goto done;
    }
    if (!ds4_gpu_tensor_write(embed_t, 0, embed, row_bytes) ||
        !ds4_gpu_tensor_write(prev_hc_t, 0, prev_hc, hc_bytes) ||
        !ds4_gpu_tensor_write(ref_enorm, 0, ref_enorm_host, row_bytes) ||
        !ds4_gpu_tensor_write(ref_eproj, 0, ref_eproj_host, row_bytes) ||
        !ds4_gpu_tensor_write(ref_eproj_hc, 0, ref_eproj_hc_host, hc_bytes) ||
        !ds4_gpu_tensor_write(ref_hnorm, 0, ref_hnorm_host, hc_bytes) ||
        !ds4_gpu_tensor_write(ref_hproj, 0, ref_hproj_host, hc_bytes) ||
        !ds4_gpu_tensor_write(ref_prefix, 0, ref_prefix_host, hc_bytes)) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: prefix chain activation upload failed\n");
        goto done;
    }

    double t0 = now_ms();
    if (ds4_gpu_arena_f32_rms_norm_f32(ds4_v100_mtp_sidecar_arena(sidecar),
                                       &enorm_view,
                                       embed_t,
                                       arena_enorm,
                                       n_embd,
                                       1,
                                       MTP_PREFIX_RMS_EPS) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(ds4_v100_mtp_sidecar_arena(sidecar),
                                      &e_proj_view,
                                      arena_enorm,
                                      arena_eproj,
                                      1) != 0 ||
        !ds4_gpu_repeat_hc_tensor(arena_eproj_hc,
                                  arena_eproj,
                                  n_embd,
                                  MTP_PREFIX_HC_ROWS) ||
        ds4_gpu_arena_f32_rms_norm_f32(ds4_v100_mtp_sidecar_arena(sidecar),
                                       &hnorm_view,
                                       prev_hc_t,
                                       arena_hnorm,
                                       n_embd,
                                       MTP_PREFIX_HC_ROWS,
                                       MTP_PREFIX_RMS_EPS) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(ds4_v100_mtp_sidecar_arena(sidecar),
                                      &h_proj_view,
                                      arena_hnorm,
                                      arena_hproj,
                                      MTP_PREFIX_HC_ROWS) != 0 ||
        !ds4_gpu_add_tensor(arena_prefix,
                            arena_eproj_hc,
                            arena_hproj,
                            (uint32_t)hc_values) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: resident prefix chain failed\n");
        goto done;
    }
    double t1 = now_ms();

    if (!ds4_gpu_synchronize()) goto done;
    double t2 = now_ms();
    const double chain_abs_tol = max_abs_tol < 1e-2 ? 1e-2 : max_abs_tol;

    fprintf(report,
            "mtp_prefix_chain\trows=%d\tcols=%u\trms_eps=%.9g"
            "\tarena_ms=%.3f\treference_ms=%.3f\tcpu_q8_abs_tol=%.9g\n",
            MTP_PREFIX_HC_ROWS,
            n_embd,
            (double)MTP_PREFIX_RMS_EPS,
            t1 - t0,
            t2 - t1,
            chain_abs_tol);
    if (compare_tensor_pair("enorm", arena_enorm, ref_enorm, row_values, max_abs_tol, report) ||
        compare_tensor_pair("e_proj", arena_eproj, ref_eproj, row_values, chain_abs_tol, report) ||
        compare_tensor_pair("e_proj_hc", arena_eproj_hc, ref_eproj_hc, hc_values, chain_abs_tol, report) ||
        compare_tensor_pair("hnorm_hc", arena_hnorm, ref_hnorm, hc_values, max_abs_tol, report) ||
        compare_tensor_pair("h_proj_hc", arena_hproj, ref_hproj, hc_values, chain_abs_tol, report) ||
        compare_tensor_pair("mtp_input_hc", arena_prefix, ref_prefix, hc_values, chain_abs_tol, report)) {
        goto done;
    }

    fprintf(report, "mtp_prefix_chain\tPASS\n");
    rc = 0;

done:
    ds4_gpu_tensor_free(ref_prefix);
    ds4_gpu_tensor_free(ref_hproj);
    ds4_gpu_tensor_free(ref_hnorm);
    ds4_gpu_tensor_free(ref_eproj_hc);
    ds4_gpu_tensor_free(ref_eproj);
    ds4_gpu_tensor_free(ref_enorm);
    ds4_gpu_tensor_free(arena_prefix);
    ds4_gpu_tensor_free(arena_hproj);
    ds4_gpu_tensor_free(arena_hnorm);
    ds4_gpu_tensor_free(arena_eproj_hc);
    ds4_gpu_tensor_free(arena_eproj);
    ds4_gpu_tensor_free(arena_enorm);
    ds4_gpu_tensor_free(prev_hc_t);
    ds4_gpu_tensor_free(embed_t);
    free(prev_hc);
    free(embed);
    free(ref_prefix_host);
    free(ref_hproj_host);
    free(ref_eproj_hc_host);
    free(ref_eproj_host);
    free(ref_hnorm_host);
    free(ref_enorm_host);
    free(h_proj_copy);
    free(e_proj_copy);
    free(hnorm_copy);
    free(enorm_copy);
    return rc;
}

int main(int argc, char **argv) {
    options opt = parse_options(argc, argv);
    FILE *report = stdout;
    if (opt.report_path && opt.report_path[0]) {
        report = fopen(opt.report_path, "w");
        if (!report) {
            fprintf(stderr,
                    "ds4-v100-mtp-prefix-smoke: cannot open report %s\n",
                    opt.report_path);
            return 1;
        }
    }

    int rc = 1;
    char err[512] = {0};
    ds4_v100_mtp_sidecar *sidecar = NULL;
    int device_count = ds4_gpu_device_count();
    fprintf(report, "visible_devices\t%d\n", device_count);
    fprintf(report, "target_gpu\t%d\n", opt.gpu);
    fprintf(report, "reserve_mib\t%d\n", opt.reserve_mib);
    fprintf(report, "max_abs_tol\t%.9g\n", opt.max_abs_tol);
    ds4_gpu_print_topology_report(report);

    if (opt.require_gpus > 0 && device_count < opt.require_gpus) {
        fprintf(stderr,
                "ds4-v100-mtp-prefix-smoke: visible devices %d < required %d\n",
                device_count,
                opt.require_gpus);
        goto done;
    }
    if (device_count <= 0) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: no CUDA devices visible\n");
        goto done;
    }
    if (opt.gpu >= device_count) {
        fprintf(stderr,
                "ds4-v100-mtp-prefix-smoke: target gpu %d outside visible device count %d\n",
                opt.gpu,
                device_count);
        goto done;
    }
    if (!ds4_gpu_set_device(opt.gpu)) {
        fprintf(stderr, "ds4-v100-mtp-prefix-smoke: failed to select gpu %d\n", opt.gpu);
        goto done;
    }

    setenv("DS4_CUDA_NO_Q8_F16_CACHE", "1", 0);
    setenv("DS4_CUDA_NO_Q8_F32_CACHE", "1", 0);

    ds4_v100_mtp_sidecar_options sidecar_opts;
    ds4_v100_mtp_sidecar_options_init(&sidecar_opts);
    sidecar_opts.mtp_path = opt.mtp_model;
    sidecar_opts.gpu = opt.gpu;
    sidecar_opts.require_device_arena = true;
    if (ds4_v100_mtp_sidecar_open(&sidecar, &sidecar_opts, report, err, sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-prefix-smoke: %s\n",
                err[0] ? err : "MTP sidecar open failed");
        goto done;
    }

    uint64_t reserve_bytes = (uint64_t)opt.reserve_mib * 1024ull * 1024ull;
    uint64_t free_after =
        ds4_gpu_arena_free_after_upload_bytes(ds4_v100_mtp_sidecar_arena(sidecar));
    fprintf(report, "reserve_bytes\t%" PRIu64 "\n", reserve_bytes);
    if (free_after < reserve_bytes) {
        fprintf(stderr,
                "ds4-v100-mtp-prefix-smoke: gpu %d free bytes %" PRIu64
                " below reserve %" PRIu64 "\n",
                opt.gpu,
                free_after,
                reserve_bytes);
        goto done;
    }

    if (run_projection(sidecar,
                       "mtp.0.e_proj.weight",
                       1,
                       11,
                       opt.max_abs_tol,
                       report) != 0) {
        goto done;
    }
    if (run_projection(sidecar,
                       "mtp.0.h_proj.weight",
                       4,
                       19,
                       opt.max_abs_tol,
                       report) != 0) {
        goto done;
    }
    if (run_prefix_chain(sidecar, opt.max_abs_tol, report) != 0) {
        goto done;
    }

    fprintf(report, "mtp_prefix_smoke\tPASS\n");
    if (opt.report_path && opt.report_path[0]) {
        printf("mtp_prefix_smoke\tPASS\treport=%s\n", opt.report_path);
    }
    rc = 0;

done:
    ds4_v100_mtp_sidecar_close(sidecar);
    if (report && report != stdout) fclose(report);
    return rc;
}
