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
    MTP_Q4K_QK_K = 256,
    MTP_Q4K_ROUTES = 6,
};

#define MTP_Q4K_SWIGLU_CLAMP 10.0f

typedef struct {
    const char *mtp_model;
    const char *report_path;
    int gpu;
    int require_gpus;
    int reserve_mib;
    double max_abs_tol;
} options;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[MTP_Q4K_QK_K / 2];
} host_block_q4_K;

typedef struct {
    float d;
    int8_t qs[MTP_Q4K_QK_K];
    int16_t bsums[MTP_Q4K_QK_K / 16];
} host_block_q8_K;

typedef char host_block_q4_k_size[(sizeof(host_block_q4_K) == 144) ? 1 : -1];
typedef char host_block_q8_k_size[(sizeof(host_block_q8_K) == 292) ? 1 : -1];

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-v100-mtp-q4k-smoke --mtp-model FILE [options]\n"
            "\n"
            "Options:\n"
            "  --gpu N                 Upload and execute on CUDA device N. Default: 7\n"
            "  --require-gpus N        Require at least N visible CUDA devices\n"
            "  --reserve-mib N         Require this much free memory after upload. Default: 4096\n"
            "  --max-abs-tol F         Max allowed absolute output delta. Default: 0.05\n"
            "  --report FILE           Write report to FILE instead of stdout\n");
}

static const char *need_arg(int *i, int argc, char **argv, const char *arg) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-v100-mtp-q4k-smoke: %s requires an argument\n", arg);
        exit(2);
    }
    return argv[++*i];
}

static int parse_int(const char *s, const char *arg) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || !end || *end || v < 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-v100-mtp-q4k-smoke: bad integer for %s: %s\n", arg, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double(const char *s, const char *arg) {
    errno = 0;
    char *end = NULL;
    double v = strtod(s, &end);
    if (errno || !s[0] || !end || *end || !(v >= 0.0)) {
        fprintf(stderr, "ds4-v100-mtp-q4k-smoke: bad float for %s: %s\n", arg, s);
        exit(2);
    }
    return v;
}

static options parse_options(int argc, char **argv) {
    options opt;
    memset(&opt, 0, sizeof(opt));
    opt.gpu = 7;
    opt.reserve_mib = 4096;
    opt.max_abs_tol = 0.05;
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
            fprintf(stderr, "ds4-v100-mtp-q4k-smoke: unknown option: %s\n", arg);
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

static void fill_activation(float *x, uint32_t cols, uint32_t salt) {
    for (uint32_t c = 0; c < cols; c++) {
        uint32_t v = c * 17u + salt * 43u;
        int centered = (int)(v % 257u) - 128;
        x[c] = (float)centered / 97.0f;
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

static float silu_host(float x) {
    return x / (1.0f + expf(-x));
}

static void quantize_q8_K_host(const float *x, host_block_q8_K *y, uint32_t k) {
    const uint32_t nb = k / MTP_Q4K_QK_K;
    for (uint32_t b = 0; b < nb; b++) {
        float max = 0.0f;
        float amax = 0.0f;
        const float *xb = x + (uint64_t)b * MTP_Q4K_QK_K;
        for (uint32_t j = 0; j < MTP_Q4K_QK_K; j++) {
            const float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax;
                max = xb[j];
            }
        }
        if (amax == 0.0f) {
            y[b].d = 0.0f;
            memset(y[b].qs, 0, sizeof(y[b].qs));
            memset(y[b].bsums, 0, sizeof(y[b].bsums));
            continue;
        }
        const float iscale = -127.0f / max;
        for (uint32_t j = 0; j < MTP_Q4K_QK_K; j++) {
            int v = (int)lrintf(iscale * xb[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (uint32_t j = 0; j < MTP_Q4K_QK_K / 16; j++) {
            int sum = 0;
            for (uint32_t i = 0; i < 16; i++) sum += y[b].qs[j * 16u + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.0f / iscale;
    }
}

static void q4_K_get_scale_min_host(uint32_t j,
                                    const uint8_t *scales,
                                    uint8_t *d_out,
                                    uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

static int32_t dot_q4_32_host(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    for (uint32_t i = 0; i < 32u; i++) {
        const int32_t v = (int32_t)((qs[i] >> shift) & 0x0fu);
        sum += v * (int32_t)q8[i];
    }
    return sum;
}

static float dot_q4_K_q8_K_block_host(const host_block_q4_K *x,
                                      const host_block_q8_K *y) {
    const float xd = f16_to_f32_host(x->d);
    const float xmin = f16_to_f32_host(x->dmin);
    int isum = 0;
    int summs = 0;
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc = 0;
        uint8_t m = 0;
        q4_K_get_scale_min_host(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dot_q4_32_host(x->qs + byte_off,
                                         y->qs + j * 32u,
                                         shift);
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

static double compare_outputs(const float *got,
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
    return max_abs;
}

static int q4k_reference(float *out,
                         const unsigned char *map,
                         const ds4_mtp_sidecar_tensor_info *gate_tensor,
                         const ds4_mtp_sidecar_tensor_info *up_tensor,
                         const ds4_mtp_sidecar_tensor_info *down_tensor,
                         const ds4_gpu_q4_k_expert_view *gate_view,
                         const ds4_gpu_q4_k_expert_view *up_view,
                         const ds4_gpu_q4_k_expert_view *down_view,
                         const int32_t *selected,
                         const float *weights,
                         const float *x,
                         float clamp) {
    const uint32_t in_dim = gate_view->cols;
    const uint32_t mid_dim = gate_view->rows;
    const uint32_t out_dim = down_view->rows;
    const uint32_t xq_blocks = in_dim / MTP_Q4K_QK_K;
    const uint32_t midq_blocks = mid_dim / MTP_Q4K_QK_K;
    host_block_q8_K *xq = (host_block_q8_K *)malloc((size_t)xq_blocks * sizeof(*xq));
    host_block_q8_K *midq = (host_block_q8_K *)malloc((size_t)MTP_Q4K_ROUTES *
                                                       midq_blocks * sizeof(*midq));
    float *mid = (float *)malloc((size_t)MTP_Q4K_ROUTES * mid_dim * sizeof(*mid));
    if (!xq || !midq || !mid) {
        free(mid);
        free(midq);
        free(xq);
        return 1;
    }

    const unsigned char *gate_base = map + gate_tensor->source_offset;
    const unsigned char *up_base = map + up_tensor->source_offset;
    const unsigned char *down_base = map + down_tensor->source_offset;

    quantize_q8_K_host(x, xq, in_dim);
    for (uint32_t slot = 0; slot < MTP_Q4K_ROUTES; slot++) {
        const uint32_t expert = selected[slot] < 0 ? 0u : (uint32_t)selected[slot];
        if (expert >= gate_view->experts) {
            free(mid);
            free(midq);
            free(xq);
            return 1;
        }
        for (uint32_t row = 0; row < mid_dim; row++) {
            const host_block_q4_K *gr = (const host_block_q4_K *)(
                    gate_base + (uint64_t)expert * gate_view->expert_stride_bytes +
                    (uint64_t)row * gate_view->row_stride_bytes);
            const host_block_q4_K *ur = (const host_block_q4_K *)(
                    up_base + (uint64_t)expert * up_view->expert_stride_bytes +
                    (uint64_t)row * up_view->row_stride_bytes);
            float gate = 0.0f;
            float up = 0.0f;
            for (uint32_t b = 0; b < xq_blocks; b++) {
                gate += dot_q4_K_q8_K_block_host(gr + b, xq + b);
                up += dot_q4_K_q8_K_block_host(ur + b, xq + b);
            }
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            mid[(uint64_t)slot * mid_dim + row] =
                silu_host(gate) * up * weights[slot];
        }
        quantize_q8_K_host(mid + (uint64_t)slot * mid_dim,
                           midq + (uint64_t)slot * midq_blocks,
                           mid_dim);
    }

    for (uint32_t row = 0; row < out_dim; row++) {
        float total = 0.0f;
        for (uint32_t slot = 0; slot < MTP_Q4K_ROUTES; slot++) {
            const uint32_t expert = selected[slot] < 0 ? 0u : (uint32_t)selected[slot];
            const host_block_q4_K *wr = (const host_block_q4_K *)(
                    down_base + (uint64_t)expert * down_view->expert_stride_bytes +
                    (uint64_t)row * down_view->row_stride_bytes);
            float acc = 0.0f;
            for (uint32_t b = 0; b < midq_blocks; b++) {
                acc += dot_q4_K_q8_K_block_host(wr + b,
                                                midq + (uint64_t)slot * midq_blocks + b);
            }
            total += acc;
        }
        out[row] = total;
    }

    free(mid);
    free(midq);
    free(xq);
    return 0;
}

static int run_q4k(ds4_mtp_sidecar *sidecar,
                   double max_abs_tol,
                   FILE *report) {
    char err[512] = {0};
    ds4_gpu_q4_k_expert_view gate_view;
    ds4_gpu_q4_k_expert_view up_view;
    ds4_gpu_q4_k_expert_view down_view;
    if (ds4_mtp_sidecar_q4_k_expert_view(sidecar,
                                               "mtp.0.ffn_gate_exps.weight",
                                               &gate_view,
                                               err,
                                               sizeof(err)) != 0 ||
        ds4_mtp_sidecar_q4_k_expert_view(sidecar,
                                               "mtp.0.ffn_up_exps.weight",
                                               &up_view,
                                               err,
                                               sizeof(err)) != 0 ||
        ds4_mtp_sidecar_q4_k_expert_view(sidecar,
                                               "mtp.0.ffn_down_exps.weight",
                                               &down_view,
                                               err,
                                               sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-q4k-smoke: %s\n",
                err[0] ? err : "failed to bind Q4_K expert views");
        return 1;
    }
    if (gate_view.cols != 4096u || gate_view.rows != 2048u ||
        up_view.cols != gate_view.cols || up_view.rows != gate_view.rows ||
        down_view.cols != gate_view.rows || down_view.rows != gate_view.cols ||
        gate_view.experts != 256u || up_view.experts != 256u ||
        down_view.experts != 256u) {
        fprintf(stderr, "ds4-v100-mtp-q4k-smoke: unexpected Q4_K MTP expert layout\n");
        return 1;
    }

    const ds4_mtp_sidecar_tensor_info *gate_tensor =
        ds4_mtp_sidecar_tensor(sidecar, "mtp.0.ffn_gate_exps.weight");
    const ds4_mtp_sidecar_tensor_info *up_tensor =
        ds4_mtp_sidecar_tensor(sidecar, "mtp.0.ffn_up_exps.weight");
    const ds4_mtp_sidecar_tensor_info *down_tensor =
        ds4_mtp_sidecar_tensor(sidecar, "mtp.0.ffn_down_exps.weight");

    const uint32_t in_dim = gate_view.cols;
    const uint32_t mid_dim = gate_view.rows;
    const uint32_t out_dim = down_view.rows;
    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
    const uint64_t mid_values = (uint64_t)MTP_Q4K_ROUTES * mid_dim;
    const uint64_t down_values = (uint64_t)MTP_Q4K_ROUTES * out_dim;
    float *x = (float *)malloc((size_t)x_bytes);
    float *got = (float *)malloc((size_t)out_bytes);
    float *ref = (float *)malloc((size_t)out_bytes);
    const int32_t selected[MTP_Q4K_ROUTES] = {0, 17, 63, 127, 191, 255};
    const float route_weights[MTP_Q4K_ROUTES] = {0.39f, 0.31f, 0.25f, 0.21f, 0.18f, 0.16f};
    ds4_gpu_tensor *x_t = NULL;
    ds4_gpu_tensor *selected_t = NULL;
    ds4_gpu_tensor *weights_t = NULL;
    ds4_gpu_tensor *out_t = NULL;
    ds4_gpu_tensor *gate_tmp = NULL;
    ds4_gpu_tensor *up_tmp = NULL;
    ds4_gpu_tensor *mid_tmp = NULL;
    ds4_gpu_tensor *down_tmp = NULL;

    int rc = 1;
    if (!x || !got || !ref) {
        fprintf(stderr, "ds4-v100-mtp-q4k-smoke: host allocation failed\n");
        goto done;
    }
    fill_activation(x, in_dim, 37);

    x_t = ds4_gpu_tensor_alloc(x_bytes);
    selected_t = ds4_gpu_tensor_alloc(sizeof(selected));
    weights_t = ds4_gpu_tensor_alloc(sizeof(route_weights));
    out_t = ds4_gpu_tensor_alloc(out_bytes);
    gate_tmp = ds4_gpu_tensor_alloc(mid_values * sizeof(float));
    up_tmp = ds4_gpu_tensor_alloc(mid_values * sizeof(float));
    mid_tmp = ds4_gpu_tensor_alloc(mid_values * sizeof(float));
    down_tmp = ds4_gpu_tensor_alloc(down_values * sizeof(float));
    if (!x_t || !selected_t || !weights_t || !out_t || !gate_tmp ||
        !up_tmp || !mid_tmp || !down_tmp) {
        fprintf(stderr, "ds4-v100-mtp-q4k-smoke: device tensor allocation failed\n");
        goto done;
    }
    if (!ds4_gpu_tensor_write(x_t, 0, x, x_bytes) ||
        !ds4_gpu_tensor_write(selected_t, 0, selected, sizeof(selected)) ||
        !ds4_gpu_tensor_write(weights_t, 0, route_weights, sizeof(route_weights))) {
        fprintf(stderr, "ds4-v100-mtp-q4k-smoke: device tensor upload failed\n");
        goto done;
    }

    double t0 = now_ms();
    if (ds4_gpu_arena_q4_k_routed_moe_one_f32(
                ds4_mtp_sidecar_arena(sidecar),
                &gate_view,
                &up_view,
                &down_view,
                out_t,
                gate_tmp,
                up_tmp,
                mid_tmp,
                down_tmp,
                selected_t,
                weights_t,
                x_t,
                MTP_Q4K_ROUTES,
                MTP_Q4K_SWIGLU_CLAMP) != 0 ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "ds4-v100-mtp-q4k-smoke: resident Q4_K routed MoE failed\n");
        goto done;
    }
    double t1 = now_ms();
    if (!ds4_gpu_tensor_read(out_t, 0, got, out_bytes)) {
        fprintf(stderr, "ds4-v100-mtp-q4k-smoke: output readback failed\n");
        goto done;
    }

    double t2 = now_ms();
    if (q4k_reference(ref,
                      (const unsigned char *)ds4_mtp_sidecar_map(sidecar),
                      gate_tensor,
                      up_tensor,
                      down_tensor,
                      &gate_view,
                      &up_view,
                      &down_view,
                      selected,
                      route_weights,
                      x,
                      MTP_Q4K_SWIGLU_CLAMP) != 0) {
        fprintf(stderr, "ds4-v100-mtp-q4k-smoke: host reference failed\n");
        goto done;
    }
    double t3 = now_ms();

    double max_abs = 0.0;
    double max_rel = 0.0;
    uint64_t max_i = 0;
    compare_outputs(got, ref, out_dim, &max_abs, &max_rel, &max_i);

    fprintf(report,
            "mtp_q4k_tensor\tgate\tdtype=%s\texperts=%u\trows=%u\tcols=%u"
            "\trow_stride=%u\texpert_stride=%" PRIu64 "\tbytes=%" PRIu64 "\n",
            gate_tensor->dtype,
            gate_view.experts,
            gate_view.rows,
            gate_view.cols,
            gate_view.row_stride_bytes,
            gate_view.expert_stride_bytes,
            gate_tensor->byte_length);
    fprintf(report,
            "mtp_q4k_tensor\tup\tdtype=%s\texperts=%u\trows=%u\tcols=%u"
            "\trow_stride=%u\texpert_stride=%" PRIu64 "\tbytes=%" PRIu64 "\n",
            up_tensor->dtype,
            up_view.experts,
            up_view.rows,
            up_view.cols,
            up_view.row_stride_bytes,
            up_view.expert_stride_bytes,
            up_tensor->byte_length);
    fprintf(report,
            "mtp_q4k_tensor\tdown\tdtype=%s\texperts=%u\trows=%u\tcols=%u"
            "\trow_stride=%u\texpert_stride=%" PRIu64 "\tbytes=%" PRIu64 "\n",
            down_tensor->dtype,
            down_view.experts,
            down_view.rows,
            down_view.cols,
            down_view.row_stride_bytes,
            down_view.expert_stride_bytes,
            down_tensor->byte_length);
    fprintf(report,
            "mtp_q4k_routes\tselected=%d,%d,%d,%d,%d,%d"
            "\tweights=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\tweight_sum=%.9g\n",
            selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
            route_weights[0], route_weights[1], route_weights[2],
            route_weights[3], route_weights[4], route_weights[5],
            route_weights[0] + route_weights[1] + route_weights[2] +
            route_weights[3] + route_weights[4] + route_weights[5]);
    fprintf(report,
            "mtp_q4k_routed\tin_dim=%u\tmid_dim=%u\tout_dim=%u\troutes=%u"
            "\tclamp=%.9g\tarena_ms=%.3f\treadback_ms=%.3f\treference_ms=%.3f"
            "\tmax_abs=%.9g\tmax_rel=%.9g\tmax_i=%" PRIu64 "\ttol=%.9g\t%s\n",
            in_dim,
            mid_dim,
            out_dim,
            MTP_Q4K_ROUTES,
            (double)MTP_Q4K_SWIGLU_CLAMP,
            t1 - t0,
            t2 - t1,
            t3 - t2,
            max_abs,
            max_rel,
            max_i,
            max_abs_tol,
            max_abs <= max_abs_tol ? "PASS" : "FAIL");
    if (max_abs > max_abs_tol) {
        fprintf(stderr,
                "ds4-v100-mtp-q4k-smoke: max_abs %.9g exceeds %.9g\n",
                max_abs,
                max_abs_tol);
        goto done;
    }

    fprintf(report, "mtp_q4k_smoke\tPASS\n");
    rc = 0;

done:
    ds4_gpu_tensor_free(down_tmp);
    ds4_gpu_tensor_free(mid_tmp);
    ds4_gpu_tensor_free(up_tmp);
    ds4_gpu_tensor_free(gate_tmp);
    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(weights_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(x_t);
    free(ref);
    free(got);
    free(x);
    return rc;
}

int main(int argc, char **argv) {
    options opt = parse_options(argc, argv);
    FILE *report = stdout;
    if (opt.report_path && opt.report_path[0]) {
        report = fopen(opt.report_path, "w");
        if (!report) {
            fprintf(stderr,
                    "ds4-v100-mtp-q4k-smoke: cannot open report %s\n",
                    opt.report_path);
            return 1;
        }
    }

    int rc = 1;
    char err[512] = {0};
    ds4_mtp_sidecar *sidecar = NULL;
    int device_count = ds4_gpu_device_count();
    fprintf(report, "visible_devices\t%d\n", device_count);
    fprintf(report, "target_gpu\t%d\n", opt.gpu);
    fprintf(report, "reserve_mib\t%d\n", opt.reserve_mib);
    fprintf(report, "max_abs_tol\t%.9g\n", opt.max_abs_tol);
    ds4_gpu_print_topology_report(report);

    if (opt.require_gpus > 0 && device_count < opt.require_gpus) {
        fprintf(stderr,
                "ds4-v100-mtp-q4k-smoke: visible devices %d < required %d\n",
                device_count,
                opt.require_gpus);
        goto done;
    }
    if (device_count <= 0) {
        fprintf(stderr, "ds4-v100-mtp-q4k-smoke: no CUDA devices visible\n");
        goto done;
    }
    if (opt.gpu >= device_count) {
        fprintf(stderr,
                "ds4-v100-mtp-q4k-smoke: target gpu %d outside visible device count %d\n",
                opt.gpu,
                device_count);
        goto done;
    }
    if (!ds4_gpu_set_device(opt.gpu)) {
        fprintf(stderr, "ds4-v100-mtp-q4k-smoke: failed to select gpu %d\n", opt.gpu);
        goto done;
    }

    ds4_mtp_sidecar_options sidecar_opts;
    ds4_mtp_sidecar_options_init(&sidecar_opts);
    sidecar_opts.mtp_path = opt.mtp_model;
    sidecar_opts.gpu = opt.gpu;
    sidecar_opts.require_device_arena = true;
    if (ds4_mtp_sidecar_open(&sidecar, &sidecar_opts, report, err, sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-q4k-smoke: %s\n",
                err[0] ? err : "MTP sidecar open failed");
        goto done;
    }

    uint64_t reserve_bytes = (uint64_t)opt.reserve_mib * 1024ull * 1024ull;
    uint64_t free_after =
        ds4_gpu_arena_free_after_upload_bytes(ds4_mtp_sidecar_arena(sidecar));
    fprintf(report, "reserve_bytes\t%" PRIu64 "\n", reserve_bytes);
    if (free_after < reserve_bytes) {
        fprintf(stderr,
                "ds4-v100-mtp-q4k-smoke: gpu %d free bytes %" PRIu64
                " below reserve %" PRIu64 "\n",
                opt.gpu,
                free_after,
                reserve_bytes);
        goto done;
    }

    if (run_q4k(sidecar, opt.max_abs_tol, report) != 0) goto done;
    if (opt.report_path && opt.report_path[0]) {
        printf("mtp_q4k_smoke\tPASS\treport=%s\n", opt.report_path);
    }
    rc = 0;

done:
    ds4_mtp_sidecar_close(sidecar);
    if (report && report != stdout) fclose(report);
    return rc;
}
