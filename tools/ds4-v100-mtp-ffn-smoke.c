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
#include <time.h>

enum {
    MTP_FFN_N_EMBD = 4096,
    MTP_FFN_N_HC = 4,
    MTP_FFN_HC_DIM = MTP_FFN_N_EMBD * MTP_FFN_N_HC,
    MTP_FFN_HC_MIX = 2 * MTP_FFN_N_HC + MTP_FFN_N_HC * MTP_FFN_N_HC,
    MTP_FFN_N_EXPERT = 256,
    MTP_FFN_N_ROUTE = 6,
    MTP_FFN_N_FF_EXP = 2048,
    MTP_FFN_QK_K = 256,
    MTP_FFN_HC_SINKHORN_ITERS = 20,
};

#define MTP_FFN_RMS_EPS 1.0e-6f
#define MTP_FFN_HC_EPS  1.0e-6f
#define MTP_FFN_ROUTED_SWIGLU_CLAMP 10.0f

typedef struct {
    const char *mtp_model;
    const char *report_path;
    int gpu;
    int require_gpus;
    int reserve_mib;
    double max_abs_tol;
    double route_tol;
} options;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[MTP_FFN_QK_K / 2];
} host_block_q4_K;

typedef struct {
    float d;
    int8_t qs[MTP_FFN_QK_K];
    int16_t bsums[MTP_FFN_QK_K / 16];
} host_block_q8_K;

typedef char host_block_q4_k_size[(sizeof(host_block_q4_K) == 144) ? 1 : -1];
typedef char host_block_q8_k_size[(sizeof(host_block_q8_K) == 292) ? 1 : -1];

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-v100-mtp-ffn-smoke --mtp-model FILE [options]\n"
            "\n"
            "Options:\n"
            "  --gpu N                 Upload and execute on CUDA device N. Default: 7\n"
            "  --require-gpus N        Require at least N visible CUDA devices\n"
            "  --reserve-mib N         Require this much free memory after upload. Default: 4096\n"
            "  --max-abs-tol F         Max allowed output delta. Default: 0.10\n"
            "  --route-tol F           Max allowed route-weight delta. Default: 1e-5\n"
            "  --report FILE           Write report to FILE instead of stdout\n");
}

static const char *need_arg(int *i, int argc, char **argv, const char *arg) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-v100-mtp-ffn-smoke: %s requires an argument\n", arg);
        exit(2);
    }
    return argv[++*i];
}

static int parse_int(const char *s, const char *arg) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || !end || *end || v < 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-v100-mtp-ffn-smoke: bad integer for %s: %s\n", arg, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double(const char *s, const char *arg) {
    errno = 0;
    char *end = NULL;
    double v = strtod(s, &end);
    if (errno || !s[0] || !end || *end || !(v >= 0.0)) {
        fprintf(stderr, "ds4-v100-mtp-ffn-smoke: bad float for %s: %s\n", arg, s);
        exit(2);
    }
    return v;
}

static options parse_options(int argc, char **argv) {
    options opt;
    memset(&opt, 0, sizeof(opt));
    opt.gpu = 7;
    opt.reserve_mib = 4096;
    opt.max_abs_tol = 0.10;
    opt.route_tol = 1.0e-5;
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
        } else if (!strcmp(arg, "--route-tol")) {
            opt.route_tol = parse_double(need_arg(&i, argc, argv, arg), arg);
        } else {
            fprintf(stderr, "ds4-v100-mtp-ffn-smoke: unknown option: %s\n", arg);
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

static void fill_hc_state(float *x) {
    for (uint32_t i = 0; i < MTP_FFN_HC_DIM; i++) {
        uint32_t v = i * 17u + 101u;
        int centered = (int)(v % 257u) - 128;
        x[i] = (float)centered / 113.0f;
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

static float sigmoid_host(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(x);
    return z / (1.0f + z);
}

static float softplus_host(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static float silu_host(float x) {
    return x * sigmoid_host(x);
}

static void rms_norm_plain_host(float *out, const float *x, uint32_t n, float eps) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) sum += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(sum / (double)n) + eps);
    for (uint32_t i = 0; i < n; i++) out[i] = x[i] * scale;
}

static void rms_norm_weight_host(float *out,
                                 const float *x,
                                 const float *weight,
                                 uint32_t n,
                                 float eps) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) sum += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(sum / (double)n) + eps);
    for (uint32_t i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}

static void matmul_f32_host(float *out,
                            const float *weight,
                            uint32_t rows,
                            uint32_t cols,
                            const float *x) {
    for (uint32_t r = 0; r < rows; r++) {
        const float *row = weight + (uint64_t)r * cols;
        double acc = 0.0;
        for (uint32_t c = 0; c < cols; c++) acc += (double)row[c] * (double)x[c];
        out[r] = (float)acc;
    }
}

static void swiglu_host(float *out,
                        const float *gate,
                        const float *up,
                        uint32_t n,
                        float clamp,
                        float weight) {
    for (uint32_t i = 0; i < n; i++) {
        float g = gate[i];
        float u = up[i];
        if (clamp > 1.0e-6f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        out[i] = silu_host(g) * u * weight;
    }
}

static void hc_split_sinkhorn_host(float *out,
                                   const float *mix,
                                   const float *scale,
                                   const float *base,
                                   int n_hc,
                                   int iters,
                                   float eps) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < n_hc; i++) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = sigmoid_host(z) + eps;
    }
    for (int i = 0; i < n_hc; i++) {
        const int off = n_hc + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f * sigmoid_host(z);
    }

    float c[16];
    for (int dst = 0; dst < n_hc; dst++) {
        float row_max = -FLT_MAX;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const int off = 2 * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }
        float row_sum = 0.0f;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }
        const float inv = 1.0f / row_sum;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }

    for (int src = 0; src < n_hc; src++) {
        float sum = 0.0f;
        for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];
        const float inv = 1.0f / (sum + eps);
        for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
    }

    for (int iter = 1; iter < iters; iter++) {
        for (int dst = 0; dst < n_hc; dst++) {
            float sum = 0.0f;
            for (int src = 0; src < n_hc; src++) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (int src = 0; src < n_hc; src++) c[src + dst * n_hc] *= inv;
        }
        for (int src = 0; src < n_hc; src++) {
            float sum = 0.0f;
            for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
        }
    }

    for (int i = 0; i < n_hc * n_hc; i++) out[2 * n_hc + i] = c[i];
}

static void hc_weighted_sum_host(float *out,
                                 const float *x,
                                 const float *weights,
                                 uint32_t n_embd,
                                 uint32_t n_hc) {
    for (uint32_t d = 0; d < n_embd; d++) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < n_hc; h++) {
            acc += x[(uint64_t)h * n_embd + d] * weights[h];
        }
        out[d] = acc;
    }
}

static void hc_expand_add_host(float *out_hc,
                               const float *block_out,
                               const float *block_add,
                               const float *residual_hc,
                               const float *split,
                               uint32_t n_embd,
                               uint32_t n_hc) {
    const float *post = split + n_hc;
    const float *comb = split + 2u * n_hc;
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        for (uint32_t d = 0; d < n_embd; d++) {
            float acc = (block_out[d] + block_add[d]) * post[dst];
            for (uint32_t src = 0; src < n_hc; src++) {
                acc += comb[dst + src * n_hc] *
                       residual_hc[(uint64_t)src * n_embd + d];
            }
            out_hc[(uint64_t)dst * n_embd + d] = acc;
        }
    }
}

static void router_select_host(int32_t selected[MTP_FFN_N_ROUTE],
                               float weights[MTP_FFN_N_ROUTE],
                               float probs[MTP_FFN_N_EXPERT],
                               const float *logits,
                               const float *bias) {
    for (uint32_t i = 0; i < MTP_FFN_N_EXPERT; i++) {
        probs[i] = sqrtf(softplus_host(logits[i]));
    }
    for (uint32_t i = 0; i < MTP_FFN_N_ROUTE; i++) selected[i] = -1;
    for (uint32_t e = 0; e < MTP_FFN_N_EXPERT; e++) {
        const float score = probs[e] + bias[e];
        for (uint32_t j = 0; j < MTP_FFN_N_ROUTE; j++) {
            if (selected[j] < 0 ||
                score > probs[(uint32_t)selected[j]] + bias[(uint32_t)selected[j]]) {
                for (uint32_t k = MTP_FFN_N_ROUTE - 1; k > j; k--) {
                    selected[k] = selected[k - 1];
                }
                selected[j] = (int32_t)e;
                break;
            }
        }
    }
    float sum = 0.0f;
    for (uint32_t j = 0; j < MTP_FFN_N_ROUTE; j++) {
        weights[j] = probs[(uint32_t)selected[j]];
        sum += weights[j];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t j = 0; j < MTP_FFN_N_ROUTE; j++) {
        weights[j] = weights[j] / sum * 1.5f;
    }
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

static void quantize_q8_K_host(const float *x, host_block_q8_K *y, uint32_t k) {
    const uint32_t nb = k / MTP_FFN_QK_K;
    for (uint32_t b = 0; b < nb; b++) {
        float max = 0.0f;
        float amax = 0.0f;
        const float *xb = x + (uint64_t)b * MTP_FFN_QK_K;
        for (uint32_t j = 0; j < MTP_FFN_QK_K; j++) {
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
        for (uint32_t j = 0; j < MTP_FFN_QK_K; j++) {
            int v = (int)lrintf(iscale * xb[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (uint32_t j = 0; j < MTP_FFN_QK_K / 16; j++) {
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
    const uint32_t xq_blocks = in_dim / MTP_FFN_QK_K;
    const uint32_t midq_blocks = mid_dim / MTP_FFN_QK_K;
    host_block_q8_K *xq = (host_block_q8_K *)malloc((size_t)xq_blocks * sizeof(*xq));
    host_block_q8_K *midq = (host_block_q8_K *)malloc((size_t)MTP_FFN_N_ROUTE *
                                                       midq_blocks * sizeof(*midq));
    float *mid = (float *)malloc((size_t)MTP_FFN_N_ROUTE * mid_dim * sizeof(*mid));
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
    for (uint32_t slot = 0; slot < MTP_FFN_N_ROUTE; slot++) {
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
        for (uint32_t slot = 0; slot < MTP_FFN_N_ROUTE; slot++) {
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

static double compare_outputs(const float *got,
                              const float *ref,
                              uint64_t n,
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
    if (max_rel_out) *max_rel_out = max_rel;
    if (max_i_out) *max_i_out = max_i;
    return max_abs;
}

static const unsigned char *tensor_bytes(const ds4_v100_mtp_sidecar *sidecar,
                                         const char *name,
                                         const ds4_mtp_sidecar_tensor_info **info_out) {
    const ds4_mtp_sidecar_tensor_info *t =
        ds4_v100_mtp_sidecar_tensor(sidecar, name);
    if (info_out) *info_out = t;
    if (!t) return NULL;
    return (const unsigned char *)ds4_v100_mtp_sidecar_map(sidecar) + t->source_offset;
}

static int run_ffn(ds4_v100_mtp_sidecar *sidecar,
                   double max_abs_tol,
                   double route_tol,
                   FILE *report) {
    char err[512] = {0};
    const unsigned char *map = (const unsigned char *)ds4_v100_mtp_sidecar_map(sidecar);
    ds4_gpu_arena *arena = ds4_v100_mtp_sidecar_arena(sidecar);

    ds4_gpu_source_row_view hc_fn_view;
    ds4_gpu_source_row_view hc_scale_view;
    ds4_gpu_source_row_view hc_base_view;
    ds4_gpu_source_row_view ffn_norm_view;
    ds4_gpu_source_row_view router_view;
    ds4_gpu_source_row_view bias_view;
    ds4_gpu_source_row_view shared_gate_view;
    ds4_gpu_source_row_view shared_up_view;
    ds4_gpu_source_row_view shared_down_view;
    ds4_gpu_q4_k_expert_view q4_gate_view;
    ds4_gpu_q4_k_expert_view q4_up_view;
    ds4_gpu_q4_k_expert_view q4_down_view;

    if (ds4_v100_mtp_sidecar_f32_matrix_view(sidecar,
                                             "mtp.0.hc_ffn_fn.weight",
                                             &hc_fn_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.hc_ffn_scale.weight",
                                             &hc_scale_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.hc_ffn_base.weight",
                                             &hc_base_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.ffn_norm.weight",
                                             &ffn_norm_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_f32_matrix_view(sidecar,
                                             "mtp.0.ffn_gate_inp.weight",
                                             &router_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.exp_probs_b.bias",
                                             &bias_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_q8_0_view(sidecar,
                                       "mtp.0.ffn_gate_shexp.weight",
                                       &shared_gate_view,
                                       err,
                                       sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_q8_0_view(sidecar,
                                       "mtp.0.ffn_up_shexp.weight",
                                       &shared_up_view,
                                       err,
                                       sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_q8_0_view(sidecar,
                                       "mtp.0.ffn_down_shexp.weight",
                                       &shared_down_view,
                                       err,
                                       sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_q4_k_expert_view(sidecar,
                                               "mtp.0.ffn_gate_exps.weight",
                                               &q4_gate_view,
                                               err,
                                               sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_q4_k_expert_view(sidecar,
                                               "mtp.0.ffn_up_exps.weight",
                                               &q4_up_view,
                                               err,
                                               sizeof(err)) != 0 ||
        ds4_v100_mtp_sidecar_q4_k_expert_view(sidecar,
                                               "mtp.0.ffn_down_exps.weight",
                                               &q4_down_view,
                                               err,
                                               sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-ffn-smoke: %s\n",
                err[0] ? err : "failed to bind MTP FFN views");
        return 1;
    }

    if (hc_fn_view.rows != MTP_FFN_HC_MIX || hc_fn_view.cols != MTP_FFN_HC_DIM ||
        hc_scale_view.cols != 3u || hc_base_view.cols != MTP_FFN_HC_MIX ||
        ffn_norm_view.cols != MTP_FFN_N_EMBD ||
        router_view.rows != MTP_FFN_N_EXPERT || router_view.cols != MTP_FFN_N_EMBD ||
        bias_view.cols != MTP_FFN_N_EXPERT ||
        shared_gate_view.rows != MTP_FFN_N_FF_EXP || shared_gate_view.cols != MTP_FFN_N_EMBD ||
        shared_up_view.rows != MTP_FFN_N_FF_EXP || shared_up_view.cols != MTP_FFN_N_EMBD ||
        shared_down_view.rows != MTP_FFN_N_EMBD || shared_down_view.cols != MTP_FFN_N_FF_EXP ||
        q4_gate_view.experts != MTP_FFN_N_EXPERT ||
        q4_gate_view.rows != MTP_FFN_N_FF_EXP || q4_gate_view.cols != MTP_FFN_N_EMBD ||
        q4_up_view.experts != MTP_FFN_N_EXPERT ||
        q4_up_view.rows != MTP_FFN_N_FF_EXP || q4_up_view.cols != MTP_FFN_N_EMBD ||
        q4_down_view.experts != MTP_FFN_N_EXPERT ||
        q4_down_view.rows != MTP_FFN_N_EMBD || q4_down_view.cols != MTP_FFN_N_FF_EXP) {
        fprintf(stderr, "ds4-v100-mtp-ffn-smoke: unexpected MTP FFN tensor layout\n");
        return 1;
    }

    const ds4_mtp_sidecar_tensor_info *q4_gate_tensor = NULL;
    const ds4_mtp_sidecar_tensor_info *q4_up_tensor = NULL;
    const ds4_mtp_sidecar_tensor_info *q4_down_tensor = NULL;
    const float *hc_fn = (const float *)tensor_bytes(sidecar, "mtp.0.hc_ffn_fn.weight", NULL);
    const float *hc_scale = (const float *)tensor_bytes(sidecar, "mtp.0.hc_ffn_scale.weight", NULL);
    const float *hc_base = (const float *)tensor_bytes(sidecar, "mtp.0.hc_ffn_base.weight", NULL);
    const float *ffn_norm_w = (const float *)tensor_bytes(sidecar, "mtp.0.ffn_norm.weight", NULL);
    const float *router_w = (const float *)tensor_bytes(sidecar, "mtp.0.ffn_gate_inp.weight", NULL);
    const float *router_bias = (const float *)tensor_bytes(sidecar, "mtp.0.exp_probs_b.bias", NULL);
    const unsigned char *shared_gate =
        tensor_bytes(sidecar, "mtp.0.ffn_gate_shexp.weight", NULL);
    const unsigned char *shared_up =
        tensor_bytes(sidecar, "mtp.0.ffn_up_shexp.weight", NULL);
    const unsigned char *shared_down =
        tensor_bytes(sidecar, "mtp.0.ffn_down_shexp.weight", NULL);
    (void)tensor_bytes(sidecar, "mtp.0.ffn_gate_exps.weight", &q4_gate_tensor);
    (void)tensor_bytes(sidecar, "mtp.0.ffn_up_exps.weight", &q4_up_tensor);
    (void)tensor_bytes(sidecar, "mtp.0.ffn_down_exps.weight", &q4_down_tensor);
    if (!hc_fn || !hc_scale || !hc_base || !ffn_norm_w || !router_w ||
        !router_bias || !shared_gate || !shared_up || !shared_down ||
        !q4_gate_tensor || !q4_up_tensor || !q4_down_tensor) {
        fprintf(stderr, "ds4-v100-mtp-ffn-smoke: missing sidecar tensor bytes\n");
        return 1;
    }

    const uint64_t embd_bytes = (uint64_t)MTP_FFN_N_EMBD * sizeof(float);
    const uint64_t hc_bytes = (uint64_t)MTP_FFN_HC_DIM * sizeof(float);
    const uint64_t mid_bytes = (uint64_t)MTP_FFN_N_FF_EXP * sizeof(float);
    const uint64_t mix_bytes = (uint64_t)MTP_FFN_HC_MIX * sizeof(float);
    const uint64_t route_i32_bytes = (uint64_t)MTP_FFN_N_ROUTE * sizeof(int32_t);
    const uint64_t route_f32_bytes = (uint64_t)MTP_FFN_N_ROUTE * sizeof(float);
    const uint64_t probs_bytes = (uint64_t)MTP_FFN_N_EXPERT * sizeof(float);
    const uint64_t q4_mid_values = (uint64_t)MTP_FFN_N_ROUTE * MTP_FFN_N_FF_EXP;
    const uint64_t q4_down_values = (uint64_t)MTP_FFN_N_ROUTE * MTP_FFN_N_EMBD;

    float *after_hc = (float *)malloc((size_t)hc_bytes);
    float *got_routed = (float *)malloc((size_t)embd_bytes);
    float *got_shared = (float *)malloc((size_t)embd_bytes);
    float *got_ffn = (float *)malloc((size_t)embd_bytes);
    float *got_next_hc = (float *)malloc((size_t)hc_bytes);
    float *ref_hc_norm = (float *)malloc((size_t)hc_bytes);
    float *ref_ffn_cur = (float *)malloc((size_t)embd_bytes);
    float *ref_ffn_norm = (float *)malloc((size_t)embd_bytes);
    float *ref_router_logits = (float *)malloc((size_t)probs_bytes);
    float *ref_router_probs = (float *)malloc((size_t)probs_bytes);
    float *ref_routed = (float *)malloc((size_t)embd_bytes);
    float *ref_shared_gate = (float *)malloc((size_t)mid_bytes);
    float *ref_shared_up = (float *)malloc((size_t)mid_bytes);
    float *ref_shared_mid = (float *)malloc((size_t)mid_bytes);
    float *ref_shared = (float *)malloc((size_t)embd_bytes);
    float *ref_ffn = (float *)malloc((size_t)embd_bytes);
    float *ref_next_hc = (float *)malloc((size_t)hc_bytes);
    int32_t ref_selected[MTP_FFN_N_ROUTE];
    int32_t got_selected[MTP_FFN_N_ROUTE];
    float ref_weights[MTP_FFN_N_ROUTE];
    float got_weights[MTP_FFN_N_ROUTE];
    float ref_hc_mix[MTP_FFN_HC_MIX];
    float ref_ffn_split[MTP_FFN_HC_MIX];

    ds4_gpu_tensor *after_hc_t = NULL;
    ds4_gpu_tensor *hc_norm_t = NULL;
    ds4_gpu_tensor *hc_mix_t = NULL;
    ds4_gpu_tensor *ffn_cur_t = NULL;
    ds4_gpu_tensor *ffn_split_t = NULL;
    ds4_gpu_tensor *ffn_norm_t = NULL;
    ds4_gpu_tensor *router_logits_t = NULL;
    ds4_gpu_tensor *router_probs_t = NULL;
    ds4_gpu_tensor *selected_t = NULL;
    ds4_gpu_tensor *weights_t = NULL;
    ds4_gpu_tensor *routed_t = NULL;
    ds4_gpu_tensor *q4_gate_tmp_t = NULL;
    ds4_gpu_tensor *q4_up_tmp_t = NULL;
    ds4_gpu_tensor *q4_mid_tmp_t = NULL;
    ds4_gpu_tensor *q4_down_tmp_t = NULL;
    ds4_gpu_tensor *shared_gate_t = NULL;
    ds4_gpu_tensor *shared_up_t = NULL;
    ds4_gpu_tensor *shared_mid_t = NULL;
    ds4_gpu_tensor *shared_t = NULL;
    ds4_gpu_tensor *ffn_t = NULL;
    ds4_gpu_tensor *next_hc_t = NULL;

    int rc = 1;
    if (!after_hc || !got_routed || !got_shared || !got_ffn || !got_next_hc ||
        !ref_hc_norm || !ref_ffn_cur || !ref_ffn_norm || !ref_router_logits ||
        !ref_router_probs || !ref_routed || !ref_shared_gate || !ref_shared_up ||
        !ref_shared_mid || !ref_shared || !ref_ffn || !ref_next_hc) {
        fprintf(stderr, "ds4-v100-mtp-ffn-smoke: host allocation failed\n");
        goto done;
    }
    fill_hc_state(after_hc);

    after_hc_t = ds4_gpu_tensor_alloc(hc_bytes);
    hc_norm_t = ds4_gpu_tensor_alloc(hc_bytes);
    hc_mix_t = ds4_gpu_tensor_alloc(mix_bytes);
    ffn_cur_t = ds4_gpu_tensor_alloc(embd_bytes);
    ffn_split_t = ds4_gpu_tensor_alloc(mix_bytes);
    ffn_norm_t = ds4_gpu_tensor_alloc(embd_bytes);
    router_logits_t = ds4_gpu_tensor_alloc(probs_bytes);
    router_probs_t = ds4_gpu_tensor_alloc(probs_bytes);
    selected_t = ds4_gpu_tensor_alloc(route_i32_bytes);
    weights_t = ds4_gpu_tensor_alloc(route_f32_bytes);
    routed_t = ds4_gpu_tensor_alloc(embd_bytes);
    q4_gate_tmp_t = ds4_gpu_tensor_alloc(q4_mid_values * sizeof(float));
    q4_up_tmp_t = ds4_gpu_tensor_alloc(q4_mid_values * sizeof(float));
    q4_mid_tmp_t = ds4_gpu_tensor_alloc(q4_mid_values * sizeof(float));
    q4_down_tmp_t = ds4_gpu_tensor_alloc(q4_down_values * sizeof(float));
    shared_gate_t = ds4_gpu_tensor_alloc(mid_bytes);
    shared_up_t = ds4_gpu_tensor_alloc(mid_bytes);
    shared_mid_t = ds4_gpu_tensor_alloc(mid_bytes);
    shared_t = ds4_gpu_tensor_alloc(embd_bytes);
    ffn_t = ds4_gpu_tensor_alloc(embd_bytes);
    next_hc_t = ds4_gpu_tensor_alloc(hc_bytes);
    if (!after_hc_t || !hc_norm_t || !hc_mix_t || !ffn_cur_t || !ffn_split_t ||
        !ffn_norm_t || !router_logits_t || !router_probs_t || !selected_t ||
        !weights_t || !routed_t || !q4_gate_tmp_t || !q4_up_tmp_t ||
        !q4_mid_tmp_t || !q4_down_tmp_t || !shared_gate_t || !shared_up_t ||
        !shared_mid_t || !shared_t || !ffn_t || !next_hc_t) {
        fprintf(stderr, "ds4-v100-mtp-ffn-smoke: device tensor allocation failed\n");
        goto done;
    }
    if (!ds4_gpu_tensor_write(after_hc_t, 0, after_hc, hc_bytes)) {
        fprintf(stderr, "ds4-v100-mtp-ffn-smoke: HC upload failed\n");
        goto done;
    }

    double t0 = now_ms();
    if (!ds4_gpu_rms_norm_plain_tensor(hc_norm_t,
                                       after_hc_t,
                                       MTP_FFN_HC_DIM,
                                       MTP_FFN_RMS_EPS) ||
        ds4_gpu_arena_f32_matmul_f32(arena,
                                     &hc_fn_view,
                                     hc_norm_t,
                                     hc_mix_t) != 0 ||
        ds4_gpu_arena_hc_split_weighted_sum_tensor(arena,
                                                   &hc_scale_view,
                                                   &hc_base_view,
                                                   ffn_cur_t,
                                                   ffn_split_t,
                                                   hc_mix_t,
                                                   after_hc_t,
                                                   MTP_FFN_N_EMBD,
                                                   MTP_FFN_N_HC,
                                                   MTP_FFN_HC_SINKHORN_ITERS,
                                                   MTP_FFN_HC_EPS) != 0 ||
        ds4_gpu_arena_f32_rms_norm_f32(arena,
                                       &ffn_norm_view,
                                       ffn_cur_t,
                                       ffn_norm_t,
                                       MTP_FFN_N_EMBD,
                                       1,
                                       MTP_FFN_RMS_EPS) != 0 ||
        ds4_gpu_arena_f32_matmul_f32(arena,
                                     &router_view,
                                     ffn_norm_t,
                                     router_logits_t) != 0 ||
        ds4_gpu_arena_router_select_bias_tensor(arena,
                                                &bias_view,
                                                selected_t,
                                                weights_t,
                                                router_probs_t,
                                                router_logits_t) != 0 ||
        ds4_gpu_arena_q4_k_routed_moe_one_f32(arena,
                                              &q4_gate_view,
                                              &q4_up_view,
                                              &q4_down_view,
                                              routed_t,
                                              q4_gate_tmp_t,
                                              q4_up_tmp_t,
                                              q4_mid_tmp_t,
                                              q4_down_tmp_t,
                                              selected_t,
                                              weights_t,
                                              ffn_norm_t,
                                              MTP_FFN_N_ROUTE,
                                              MTP_FFN_ROUTED_SWIGLU_CLAMP) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(arena,
                                      &shared_gate_view,
                                      ffn_norm_t,
                                      shared_gate_t,
                                      1) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(arena,
                                      &shared_up_view,
                                      ffn_norm_t,
                                      shared_up_t,
                                      1) != 0 ||
        !ds4_gpu_swiglu_tensor(shared_mid_t,
                               shared_gate_t,
                               shared_up_t,
                               MTP_FFN_N_FF_EXP,
                               0.0f,
                               1.0f) ||
        ds4_gpu_arena_q8_0_matmul_f32(arena,
                                      &shared_down_view,
                                      shared_mid_t,
                                      shared_t,
                                      1) != 0 ||
        !ds4_gpu_add_tensor(ffn_t,
                            shared_t,
                            routed_t,
                            MTP_FFN_N_EMBD) ||
        !ds4_gpu_hc_expand_add_split_tensor(next_hc_t,
                                            shared_t,
                                            routed_t,
                                            after_hc_t,
                                            ffn_split_t,
                                            MTP_FFN_N_EMBD,
                                            MTP_FFN_N_HC) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "ds4-v100-mtp-ffn-smoke: resident MTP FFN execution failed\n");
        goto done;
    }
    double t1 = now_ms();

    if (!ds4_gpu_tensor_read(selected_t, 0, got_selected, sizeof(got_selected)) ||
        !ds4_gpu_tensor_read(weights_t, 0, got_weights, sizeof(got_weights)) ||
        !ds4_gpu_tensor_read(routed_t, 0, got_routed, embd_bytes) ||
        !ds4_gpu_tensor_read(shared_t, 0, got_shared, embd_bytes) ||
        !ds4_gpu_tensor_read(ffn_t, 0, got_ffn, embd_bytes) ||
        !ds4_gpu_tensor_read(next_hc_t, 0, got_next_hc, hc_bytes)) {
        fprintf(stderr, "ds4-v100-mtp-ffn-smoke: output readback failed\n");
        goto done;
    }
    double t2 = now_ms();

    rms_norm_plain_host(ref_hc_norm, after_hc, MTP_FFN_HC_DIM, MTP_FFN_RMS_EPS);
    matmul_f32_host(ref_hc_mix, hc_fn, MTP_FFN_HC_MIX, MTP_FFN_HC_DIM, ref_hc_norm);
    hc_split_sinkhorn_host(ref_ffn_split,
                           ref_hc_mix,
                           hc_scale,
                           hc_base,
                           MTP_FFN_N_HC,
                           MTP_FFN_HC_SINKHORN_ITERS,
                           MTP_FFN_HC_EPS);
    hc_weighted_sum_host(ref_ffn_cur,
                         after_hc,
                         ref_ffn_split,
                         MTP_FFN_N_EMBD,
                         MTP_FFN_N_HC);
    rms_norm_weight_host(ref_ffn_norm,
                         ref_ffn_cur,
                         ffn_norm_w,
                         MTP_FFN_N_EMBD,
                         MTP_FFN_RMS_EPS);
    matmul_f32_host(ref_router_logits,
                    router_w,
                    MTP_FFN_N_EXPERT,
                    MTP_FFN_N_EMBD,
                    ref_ffn_norm);
    router_select_host(ref_selected,
                       ref_weights,
                       ref_router_probs,
                       ref_router_logits,
                       router_bias);
    if (q4k_reference(ref_routed,
                      map,
                      q4_gate_tensor,
                      q4_up_tensor,
                      q4_down_tensor,
                      &q4_gate_view,
                      &q4_up_view,
                      &q4_down_view,
                      ref_selected,
                      ref_weights,
                      ref_ffn_norm,
                      MTP_FFN_ROUTED_SWIGLU_CLAMP) != 0 ||
        matmul_q8_0_host(ref_shared_gate,
                         shared_gate,
                         MTP_FFN_N_EMBD,
                         MTP_FFN_N_FF_EXP,
                         ref_ffn_norm,
                         1) != 0 ||
        matmul_q8_0_host(ref_shared_up,
                         shared_up,
                         MTP_FFN_N_EMBD,
                         MTP_FFN_N_FF_EXP,
                         ref_ffn_norm,
                         1) != 0) {
        fprintf(stderr, "ds4-v100-mtp-ffn-smoke: CPU gate/up reference failed\n");
        goto done;
    }
    swiglu_host(ref_shared_mid,
                ref_shared_gate,
                ref_shared_up,
                MTP_FFN_N_FF_EXP,
                0.0f,
                1.0f);
    if (matmul_q8_0_host(ref_shared,
                         shared_down,
                         MTP_FFN_N_FF_EXP,
                         MTP_FFN_N_EMBD,
                         ref_shared_mid,
                         1) != 0) {
        fprintf(stderr, "ds4-v100-mtp-ffn-smoke: CPU shared-down reference failed\n");
        goto done;
    }
    for (uint32_t i = 0; i < MTP_FFN_N_EMBD; i++) {
        ref_ffn[i] = ref_shared[i] + ref_routed[i];
    }
    hc_expand_add_host(ref_next_hc,
                       ref_shared,
                       ref_routed,
                       after_hc,
                       ref_ffn_split,
                       MTP_FFN_N_EMBD,
                       MTP_FFN_N_HC);
    double t3 = now_ms();

    int selected_match = 1;
    double route_max_abs = 0.0;
    for (uint32_t i = 0; i < MTP_FFN_N_ROUTE; i++) {
        if (got_selected[i] != ref_selected[i]) selected_match = 0;
        const double diff = fabs((double)got_weights[i] - (double)ref_weights[i]);
        if (diff > route_max_abs) route_max_abs = diff;
    }
    double routed_rel = 0.0;
    double shared_rel = 0.0;
    double ffn_rel = 0.0;
    double hc_rel = 0.0;
    uint64_t routed_i = 0;
    uint64_t shared_i = 0;
    uint64_t ffn_i = 0;
    uint64_t hc_i = 0;
    const double routed_abs = compare_outputs(got_routed,
                                              ref_routed,
                                              MTP_FFN_N_EMBD,
                                              &routed_rel,
                                              &routed_i);
    const double shared_abs = compare_outputs(got_shared,
                                              ref_shared,
                                              MTP_FFN_N_EMBD,
                                              &shared_rel,
                                              &shared_i);
    const double ffn_abs = compare_outputs(got_ffn,
                                           ref_ffn,
                                           MTP_FFN_N_EMBD,
                                           &ffn_rel,
                                           &ffn_i);
    const double hc_abs = compare_outputs(got_next_hc,
                                          ref_next_hc,
                                          MTP_FFN_HC_DIM,
                                          &hc_rel,
                                          &hc_i);

    fprintf(report,
            "mtp_ffn_tensor\thc_ffn_fn\trows=%u\tcols=%u\tbytes=%" PRIu64 "\n",
            hc_fn_view.rows,
            hc_fn_view.cols,
            hc_fn_view.byte_length);
    fprintf(report,
            "mtp_ffn_tensor\tffn_gate_inp\trows=%u\tcols=%u\tbytes=%" PRIu64 "\n",
            router_view.rows,
            router_view.cols,
            router_view.byte_length);
    fprintf(report,
            "mtp_ffn_tensor\tshared_q8\tgate_rows=%u\tgate_cols=%u\tdown_rows=%u\tdown_cols=%u\n",
            shared_gate_view.rows,
            shared_gate_view.cols,
            shared_down_view.rows,
            shared_down_view.cols);
    fprintf(report,
            "mtp_ffn_routes\tselected=%d,%d,%d,%d,%d,%d"
            "\tref_selected=%d,%d,%d,%d,%d,%d"
            "\tweights=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g"
            "\tref_weights=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g"
            "\tweight_max_abs=%.9g\ttol=%.9g\t%s\n",
            got_selected[0], got_selected[1], got_selected[2],
            got_selected[3], got_selected[4], got_selected[5],
            ref_selected[0], ref_selected[1], ref_selected[2],
            ref_selected[3], ref_selected[4], ref_selected[5],
            got_weights[0], got_weights[1], got_weights[2],
            got_weights[3], got_weights[4], got_weights[5],
            ref_weights[0], ref_weights[1], ref_weights[2],
            ref_weights[3], ref_weights[4], ref_weights[5],
            route_max_abs,
            route_tol,
            selected_match && route_max_abs <= route_tol ? "PASS" : "FAIL");
    fprintf(report,
            "mtp_ffn_compare\trouted\tmax_abs=%.9g\tmax_rel=%.9g\tmax_i=%" PRIu64 "\ttol=%.9g\t%s\n",
            routed_abs,
            routed_rel,
            routed_i,
            max_abs_tol,
            routed_abs <= max_abs_tol ? "PASS" : "FAIL");
    fprintf(report,
            "mtp_ffn_compare\tshared\tmax_abs=%.9g\tmax_rel=%.9g\tmax_i=%" PRIu64 "\ttol=%.9g\t%s\n",
            shared_abs,
            shared_rel,
            shared_i,
            max_abs_tol,
            shared_abs <= max_abs_tol ? "PASS" : "FAIL");
    fprintf(report,
            "mtp_ffn_compare\tffn_out\tmax_abs=%.9g\tmax_rel=%.9g\tmax_i=%" PRIu64 "\ttol=%.9g\t%s\n",
            ffn_abs,
            ffn_rel,
            ffn_i,
            max_abs_tol,
            ffn_abs <= max_abs_tol ? "PASS" : "FAIL");
    fprintf(report,
            "mtp_ffn_compare\tnext_hc\tmax_abs=%.9g\tmax_rel=%.9g\tmax_i=%" PRIu64 "\ttol=%.9g\t%s\n",
            hc_abs,
            hc_rel,
            hc_i,
            max_abs_tol,
            hc_abs <= max_abs_tol ? "PASS" : "FAIL");
    fprintf(report,
            "mtp_ffn_timing\tarena_ms=%.3f\treadback_ms=%.3f\treference_ms=%.3f\n",
            t1 - t0,
            t2 - t1,
            t3 - t2);

    if (!selected_match || route_max_abs > route_tol ||
        routed_abs > max_abs_tol || shared_abs > max_abs_tol ||
        ffn_abs > max_abs_tol || hc_abs > max_abs_tol) {
        fprintf(stderr,
                "ds4-v100-mtp-ffn-smoke: comparison failed"
                " route=%.9g routed=%.9g shared=%.9g ffn=%.9g hc=%.9g\n",
                route_max_abs,
                routed_abs,
                shared_abs,
                ffn_abs,
                hc_abs);
        goto done;
    }

    fprintf(report, "mtp_ffn_smoke\tPASS\n");
    rc = 0;

done:
    ds4_gpu_tensor_free(next_hc_t);
    ds4_gpu_tensor_free(ffn_t);
    ds4_gpu_tensor_free(shared_t);
    ds4_gpu_tensor_free(shared_mid_t);
    ds4_gpu_tensor_free(shared_up_t);
    ds4_gpu_tensor_free(shared_gate_t);
    ds4_gpu_tensor_free(q4_down_tmp_t);
    ds4_gpu_tensor_free(q4_mid_tmp_t);
    ds4_gpu_tensor_free(q4_up_tmp_t);
    ds4_gpu_tensor_free(q4_gate_tmp_t);
    ds4_gpu_tensor_free(routed_t);
    ds4_gpu_tensor_free(weights_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(router_probs_t);
    ds4_gpu_tensor_free(router_logits_t);
    ds4_gpu_tensor_free(ffn_norm_t);
    ds4_gpu_tensor_free(ffn_split_t);
    ds4_gpu_tensor_free(ffn_cur_t);
    ds4_gpu_tensor_free(hc_mix_t);
    ds4_gpu_tensor_free(hc_norm_t);
    ds4_gpu_tensor_free(after_hc_t);
    free(ref_next_hc);
    free(ref_ffn);
    free(ref_shared);
    free(ref_shared_mid);
    free(ref_shared_up);
    free(ref_shared_gate);
    free(ref_routed);
    free(ref_router_probs);
    free(ref_router_logits);
    free(ref_ffn_norm);
    free(ref_ffn_cur);
    free(ref_hc_norm);
    free(got_next_hc);
    free(got_ffn);
    free(got_shared);
    free(got_routed);
    free(after_hc);
    return rc;
}

int main(int argc, char **argv) {
    options opt = parse_options(argc, argv);
    FILE *report = stdout;
    if (opt.report_path && opt.report_path[0]) {
        report = fopen(opt.report_path, "w");
        if (!report) {
            fprintf(stderr,
                    "ds4-v100-mtp-ffn-smoke: cannot open report %s\n",
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
    fprintf(report, "route_tol\t%.9g\n", opt.route_tol);
    ds4_gpu_print_topology_report(report);

    if (opt.require_gpus > 0 && device_count < opt.require_gpus) {
        fprintf(stderr,
                "ds4-v100-mtp-ffn-smoke: visible devices %d < required %d\n",
                device_count,
                opt.require_gpus);
        goto done;
    }
    if (device_count <= 0) {
        fprintf(stderr, "ds4-v100-mtp-ffn-smoke: no CUDA devices visible\n");
        goto done;
    }
    if (opt.gpu >= device_count) {
        fprintf(stderr,
                "ds4-v100-mtp-ffn-smoke: target gpu %d outside visible device count %d\n",
                opt.gpu,
                device_count);
        goto done;
    }
    if (!ds4_gpu_set_device(opt.gpu)) {
        fprintf(stderr, "ds4-v100-mtp-ffn-smoke: failed to select gpu %d\n", opt.gpu);
        goto done;
    }

    ds4_v100_mtp_sidecar_options sidecar_opts;
    ds4_v100_mtp_sidecar_options_init(&sidecar_opts);
    sidecar_opts.mtp_path = opt.mtp_model;
    sidecar_opts.gpu = opt.gpu;
    sidecar_opts.require_device_arena = true;
    if (ds4_v100_mtp_sidecar_open(&sidecar, &sidecar_opts, report, err, sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-ffn-smoke: %s\n",
                err[0] ? err : "MTP sidecar open failed");
        goto done;
    }

    uint64_t reserve_bytes = (uint64_t)opt.reserve_mib * 1024ull * 1024ull;
    uint64_t free_after =
        ds4_gpu_arena_free_after_upload_bytes(ds4_v100_mtp_sidecar_arena(sidecar));
    fprintf(report, "reserve_bytes\t%" PRIu64 "\n", reserve_bytes);
    if (free_after < reserve_bytes) {
        fprintf(stderr,
                "ds4-v100-mtp-ffn-smoke: gpu %d free bytes %" PRIu64
                " below reserve %" PRIu64 "\n",
                opt.gpu,
                free_after,
                reserve_bytes);
        goto done;
    }

    if (run_ffn(sidecar, opt.max_abs_tol, opt.route_tol, report) != 0) goto done;
    if (opt.report_path && opt.report_path[0]) {
        printf("mtp_ffn_smoke\tPASS\treport=%s\n", opt.report_path);
    }
    rc = 0;

done:
    ds4_v100_mtp_sidecar_close(sidecar);
    if (report && report != stdout) fclose(report);
    return rc;
}
