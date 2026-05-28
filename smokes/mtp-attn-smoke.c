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
    MTP_ATTN_N_EMBD = 4096,
    MTP_ATTN_N_HC = 4,
    MTP_ATTN_HC_DIM = MTP_ATTN_N_EMBD * MTP_ATTN_N_HC,
    MTP_ATTN_HC_MIX = 2 * MTP_ATTN_N_HC + MTP_ATTN_N_HC * MTP_ATTN_N_HC,
    MTP_ATTN_N_HEAD = 64,
    MTP_ATTN_HEAD_DIM = 512,
    MTP_ATTN_N_ROT = 64,
    MTP_ATTN_RAW_CAP = 128,
    MTP_ATTN_Q_LORA = 1024,
    MTP_ATTN_OUT_GROUPS = 8,
    MTP_ATTN_OUT_GROUP_DIM = 4096,
    MTP_ATTN_OUT_GROUP_RANK = 1024,
    MTP_ATTN_OUT_LOW_DIM = MTP_ATTN_OUT_GROUPS * MTP_ATTN_OUT_GROUP_RANK,
    MTP_ATTN_HC_SINKHORN_ITERS = 20,
};

#define MTP_ATTN_RMS_EPS 1.0e-6f
#define MTP_ATTN_HC_EPS  1.0e-6f

typedef struct {
    const char *mtp_model;
    const char *report_path;
    int gpu;
    int require_gpus;
    int reserve_mib;
    double max_abs_tol;
    double integrated_max_abs_tol;
} options;

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-v100-mtp-attn-smoke --mtp-model FILE [options]\n"
            "\n"
            "Options:\n"
            "  --gpu N                 Upload and execute on CUDA device N. Default: 7\n"
            "  --require-gpus N        Require at least N visible CUDA devices\n"
            "  --reserve-mib N         Require this much free memory after upload. Default: 4096\n"
            "  --max-abs-tol F         Max allowed attention-head delta. Default: 0.002\n"
            "  --integrated-max-abs-tol F\n"
            "                          Max allowed integrated output delta. Default: 0.5\n"
            "  --report FILE           Write report to FILE instead of stdout\n");
}

static const char *need_arg(int *i, int argc, char **argv, const char *arg) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: %s requires an argument\n", arg);
        exit(2);
    }
    return argv[++*i];
}

static int parse_int(const char *s, const char *arg) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || !end || *end || v < 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: bad integer for %s: %s\n", arg, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double(const char *s, const char *arg) {
    errno = 0;
    char *end = NULL;
    double v = strtod(s, &end);
    if (errno || !s[0] || !end || *end || !(v >= 0.0)) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: bad float for %s: %s\n", arg, s);
        exit(2);
    }
    return v;
}

static options parse_options(int argc, char **argv) {
    options opt;
    memset(&opt, 0, sizeof(opt));
    opt.gpu = 7;
    opt.reserve_mib = 4096;
    opt.max_abs_tol = 0.002;
    opt.integrated_max_abs_tol = 0.5;
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
        } else if (!strcmp(arg, "--integrated-max-abs-tol")) {
            opt.integrated_max_abs_tol = parse_double(need_arg(&i, argc, argv, arg), arg);
        } else {
            fprintf(stderr, "ds4-v100-mtp-attn-smoke: unknown option: %s\n", arg);
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

static uint16_t f32_to_f16_host(float f) {
    uint32_t bits = 0;
    memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) return (uint16_t)(sign | 0x7e00u);
        return (uint16_t)(sign | 0x7c00u);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

static void f16_round_inplace_host(float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) x[i] = f16_to_f32_host(f32_to_f16_host(x[i]));
}

static float dsv4_e4m3fn_value_host(int i) {
    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

static float dsv4_e4m3fn_dequant_host(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_host(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - dsv4_e4m3fn_value_host(best));
        const float next_diff = fabsf(ax - dsv4_e4m3fn_value_host(best + 1));
        if (next_diff < best_diff ||
            (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best++;
        }
    }
    return sign * dsv4_e4m3fn_value_host(best);
}

static void dsv4_fp8_kv_quantize_row_inplace_host(float *x,
                                                  uint32_t head_dim,
                                                  uint32_t n_rot) {
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 64; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        for (uint32_t i = 0; i < 64; i++) {
            float v = x[off + i] / scale;
            if (v > 448.0f) v = 448.0f;
            if (v < -448.0f) v = -448.0f;
            x[off + i] = dsv4_e4m3fn_dequant_host(v) * scale;
        }
    }
}

static void fill_q(float *q) {
    for (uint32_t h = 0; h < MTP_ATTN_N_HEAD; h++) {
        for (uint32_t d = 0; d < MTP_ATTN_HEAD_DIM; d++) {
            const uint32_t v = h * 131u + d * 17u + 23u;
            const int centered = (int)(v % 257u) - 128;
            q[(uint64_t)h * MTP_ATTN_HEAD_DIM + d] = (float)centered / 2048.0f;
        }
    }
}

static void fill_kv_for_pos(float *kv, uint32_t pos) {
    for (uint32_t d = 0; d < MTP_ATTN_HEAD_DIM; d++) {
        const uint32_t v = pos * 193u + d * 29u + 71u;
        const int centered = (int)(v % 383u) - 191;
        kv[d] = (float)centered / 1536.0f;
    }
}

static void fill_hc_state(float *x) {
    for (uint32_t i = 0; i < MTP_ATTN_HC_DIM; i++) {
        const uint32_t v = i * 19u + 157u;
        const int centered = (int)(v % 293u) - 146;
        x[i] = (float)centered / 127.0f;
    }
}

static const ds4_mtp_sidecar_tensor_info *need_tensor(
        const ds4_mtp_sidecar *sidecar,
        const char *name) {
    const ds4_mtp_sidecar_tensor_info *t =
        ds4_mtp_sidecar_tensor(sidecar, name);
    if (!t) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: missing MTP tensor %s\n", name);
    }
    return t;
}

static int source_view_rows(const ds4_gpu_source_row_view *src,
                            uint32_t row0,
                            uint32_t rows,
                            ds4_gpu_source_row_view *out) {
    if (!src || !out || rows == 0 || row0 > src->rows || rows > src->rows - row0 ||
        src->row_stride_bytes == 0) {
        return 1;
    }
    const uint64_t skip = (uint64_t)row0 * src->row_stride_bytes;
    const uint64_t byte_length = (uint64_t)rows * src->row_stride_bytes;
    if (skip > src->byte_length || byte_length > src->byte_length - skip) {
        return 1;
    }
    *out = *src;
    out->arena_offset += skip;
    out->byte_length = byte_length;
    out->rows = rows;
    return 0;
}

static void attention_ref(float *out,
                          const float *q,
                          const float *raw_cache,
                          const float *sinks,
                          uint32_t n_raw,
                          uint32_t raw_start) {
    const float scale = 1.0f / sqrtf((float)MTP_ATTN_HEAD_DIM);
    float scores[MTP_ATTN_RAW_CAP];
    for (uint32_t h = 0; h < MTP_ATTN_N_HEAD; h++) {
        const float *qh = q + (uint64_t)h * MTP_ATTN_HEAD_DIM;
        float max_s = sinks[h];
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = (raw_start + r) % MTP_ATTN_RAW_CAP;
            const float *kv = raw_cache + (uint64_t)row * MTP_ATTN_HEAD_DIM;
            float dot = 0.0f;
            for (uint32_t d = 0; d < MTP_ATTN_HEAD_DIM; d++) dot += qh[d] * kv[d];
            scores[r] = dot * scale;
            if (scores[r] > max_s) max_s = scores[r];
        }
        float denom = expf(sinks[h] - max_s);
        for (uint32_t r = 0; r < n_raw; r++) {
            scores[r] = expf(scores[r] - max_s);
            denom += scores[r];
        }
        float *oh = out + (uint64_t)h * MTP_ATTN_HEAD_DIM;
        for (uint32_t d = 0; d < MTP_ATTN_HEAD_DIM; d++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < n_raw; r++) {
                const uint32_t row = (raw_start + r) % MTP_ATTN_RAW_CAP;
                acc += raw_cache[(uint64_t)row * MTP_ATTN_HEAD_DIM + d] * scores[r];
            }
            oh[d] = acc / denom;
        }
    }
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

static int compare_device_to_host(const char *label,
                                  const ds4_gpu_tensor *got_t,
                                  const float *ref,
                                  uint64_t values,
                                  double tol,
                                  FILE *report,
                                  double *max_abs_out) {
    const uint64_t bytes = values * sizeof(float);
    float *got = (float *)malloc((size_t)bytes);
    if (!got) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: compare allocation failed for %s\n", label);
        return 1;
    }
    if (!ds4_gpu_tensor_read(got_t, 0, got, bytes)) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: compare readback failed for %s\n", label);
        free(got);
        return 1;
    }
    double max_rel = 0.0;
    uint64_t max_i = 0;
    const double max_abs = compare_outputs(got, ref, values, &max_rel, &max_i);
    if (max_abs_out) *max_abs_out = max_abs;
    fprintf(report,
            "mtp_attn_integrated\t%s\tvalues=%" PRIu64
            "\tmax_abs=%.9g\tmax_rel=%.9g\tmax_i=%" PRIu64 "\t%s\n",
            label,
            values,
            max_abs,
            max_rel,
            max_i,
            max_abs <= tol ? "PASS" : "FAIL");
    free(got);
    if (max_abs > tol) {
        fprintf(stderr,
                "ds4-v100-mtp-attn-smoke: integrated %s max_abs %.9g exceeds %.9g\n",
                label,
                max_abs,
                tol);
        return 1;
    }
    return 0;
}

static const unsigned char *tensor_bytes(const ds4_mtp_sidecar *sidecar,
                                         const char *name) {
    const ds4_mtp_sidecar_tensor_info *t = need_tensor(sidecar, name);
    if (!t) return NULL;
    return (const unsigned char *)ds4_mtp_sidecar_map(sidecar) + t->source_offset;
}

static float sigmoid_host(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(x);
    return z / (1.0f + z);
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

static void head_rms_norm_host(float *x,
                               uint32_t n_head,
                               uint32_t head_dim,
                               float eps) {
    for (uint32_t h = 0; h < n_head; h++) {
        rms_norm_plain_host(x + (uint64_t)h * head_dim,
                            x + (uint64_t)h * head_dim,
                            head_dim,
                            eps);
    }
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
        for (int src = 0; src < n_hc; src++) c[src + dst * n_hc] /= row_sum;
    }
    for (int it = 0; it < iters; it++) {
        for (int src = 0; src < n_hc; src++) {
            float col_sum = 0.0f;
            for (int dst = 0; dst < n_hc; dst++) col_sum += c[src + dst * n_hc];
            const float inv = col_sum != 0.0f ? 1.0f / col_sum : 0.0f;
            for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
        }
        for (int dst = 0; dst < n_hc; dst++) {
            float row_sum = 0.0f;
            for (int src = 0; src < n_hc; src++) row_sum += c[src + dst * n_hc];
            const float inv = row_sum != 0.0f ? 1.0f / row_sum : 0.0f;
            for (int src = 0; src < n_hc; src++) c[src + dst * n_hc] *= inv;
        }
    }
    for (int dst = 0; dst < n_hc; dst++) {
        for (int src = 0; src < n_hc; src++) {
            out[2 * n_hc + src + dst * n_hc] = c[src + dst * n_hc];
        }
    }
}

static void hc_weighted_sum_host(float *out,
                                 const float *residual_hc,
                                 const float *split,
                                 uint32_t n_embd,
                                 uint32_t n_hc) {
    for (uint32_t d = 0; d < n_embd; d++) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < n_hc; h++) {
            acc += residual_hc[(uint64_t)h * n_embd + d] * split[h];
        }
        out[d] = acc;
    }
}

static void hc_expand_split_host(float *out_hc,
                                 const float *block_out,
                                 const float *residual_hc,
                                 const float *split,
                                 uint32_t n_embd,
                                 uint32_t n_hc) {
    const float *post = split + n_hc;
    const float *comb = split + 2u * n_hc;
    for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
        for (uint32_t d = 0; d < n_embd; d++) {
            float acc = block_out[d] * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = acc;
        }
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

static int raw_row_visible(uint32_t raw_start, uint32_t n_raw, uint32_t row) {
    for (uint32_t r = 0; r < n_raw; r++) {
        if (((raw_start + r) % MTP_ATTN_RAW_CAP) == row) return 1;
    }
    return 0;
}

static int grouped_output_arena(ds4_mtp_sidecar *sidecar,
                                const ds4_gpu_source_row_view *out_a,
                                const ds4_gpu_source_row_view *out_b,
                                const ds4_gpu_tensor *heads,
                                ds4_gpu_tensor *low,
                                ds4_gpu_tensor *out) {
    ds4_gpu_arena *arena = ds4_mtp_sidecar_arena(sidecar);
    if (!arena || !out_a || !out_b || !heads || !low || !out) return 1;
    if (out_a->rows != MTP_ATTN_OUT_LOW_DIM ||
        out_a->cols != MTP_ATTN_OUT_GROUP_DIM ||
        out_b->rows != MTP_ATTN_N_EMBD ||
        out_b->cols != MTP_ATTN_OUT_LOW_DIM) {
        return 1;
    }
    if (!ds4_gpu_tensor_fill_f32(low, 0.0f, MTP_ATTN_OUT_LOW_DIM)) return 1;
    for (uint32_t g = 0; g < MTP_ATTN_OUT_GROUPS; g++) {
        ds4_gpu_source_row_view group_view;
        if (source_view_rows(out_a,
                             g * MTP_ATTN_OUT_GROUP_RANK,
                             MTP_ATTN_OUT_GROUP_RANK,
                             &group_view) != 0) {
            return 1;
        }
        ds4_gpu_tensor *head_view = ds4_gpu_tensor_view(
                heads,
                (uint64_t)g * MTP_ATTN_OUT_GROUP_DIM * sizeof(float),
                (uint64_t)MTP_ATTN_OUT_GROUP_DIM * sizeof(float));
        ds4_gpu_tensor *low_view = ds4_gpu_tensor_view(
                low,
                (uint64_t)g * MTP_ATTN_OUT_GROUP_RANK * sizeof(float),
                (uint64_t)MTP_ATTN_OUT_GROUP_RANK * sizeof(float));
        if (!head_view || !low_view) {
            ds4_gpu_tensor_free(low_view);
            ds4_gpu_tensor_free(head_view);
            return 1;
        }
        const int failed = ds4_gpu_arena_q8_0_matmul_f32(
                arena,
                &group_view,
                head_view,
                low_view,
                1);
        ds4_gpu_tensor_free(low_view);
        ds4_gpu_tensor_free(head_view);
        if (failed) return 1;
    }
    return ds4_gpu_arena_q8_0_matmul_f32(arena, out_b, low, out, 1);
}

static int run_integrated_attention(ds4_mtp_sidecar *sidecar,
                                    double max_abs_tol,
                                    FILE *report) {
    char err[512] = {0};
    ds4_gpu_arena *arena = ds4_mtp_sidecar_arena(sidecar);

    ds4_gpu_source_row_view hc_fn_view;
    ds4_gpu_source_row_view hc_scale_view;
    ds4_gpu_source_row_view hc_base_view;
    ds4_gpu_source_row_view attn_norm_view;
    ds4_gpu_source_row_view q_a_view;
    ds4_gpu_source_row_view q_a_norm_view;
    ds4_gpu_source_row_view q_b_view;
    ds4_gpu_source_row_view kv_view;
    ds4_gpu_source_row_view kv_norm_view;
    ds4_gpu_source_row_view sinks_view;
    ds4_gpu_source_row_view out_a_view;
    ds4_gpu_source_row_view out_b_view;

    if (ds4_mtp_sidecar_f32_matrix_view(sidecar,
                                             "mtp.0.hc_attn_fn.weight",
                                             &hc_fn_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.hc_attn_scale.weight",
                                             &hc_scale_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.hc_attn_base.weight",
                                             &hc_base_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.attn_norm.weight",
                                             &attn_norm_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_mtp_sidecar_q8_0_view(sidecar,
                                       "mtp.0.attn_q_a.weight",
                                       &q_a_view,
                                       err,
                                       sizeof(err)) != 0 ||
        ds4_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.attn_q_a_norm.weight",
                                             &q_a_norm_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_mtp_sidecar_q8_0_view(sidecar,
                                       "mtp.0.attn_q_b.weight",
                                       &q_b_view,
                                       err,
                                       sizeof(err)) != 0 ||
        ds4_mtp_sidecar_q8_0_view(sidecar,
                                       "mtp.0.attn_kv.weight",
                                       &kv_view,
                                       err,
                                       sizeof(err)) != 0 ||
        ds4_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.attn_kv_a_norm.weight",
                                             &kv_norm_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.attn_sinks.weight",
                                             &sinks_view,
                                             err,
                                             sizeof(err)) != 0 ||
        ds4_mtp_sidecar_q8_0_view(sidecar,
                                       "mtp.0.attn_output_a.weight",
                                       &out_a_view,
                                       err,
                                       sizeof(err)) != 0 ||
        ds4_mtp_sidecar_q8_0_view(sidecar,
                                       "mtp.0.attn_output_b.weight",
                                       &out_b_view,
                                       err,
                                       sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-attn-smoke: %s\n",
                err[0] ? err : "failed to bind integrated attention views");
        return 1;
    }

    if (!arena ||
        hc_fn_view.rows != MTP_ATTN_HC_MIX ||
        hc_fn_view.cols != MTP_ATTN_HC_DIM ||
        hc_scale_view.cols != 3u ||
        hc_base_view.cols != MTP_ATTN_HC_MIX ||
        attn_norm_view.cols != MTP_ATTN_N_EMBD ||
        q_a_view.rows != MTP_ATTN_Q_LORA ||
        q_a_view.cols != MTP_ATTN_N_EMBD ||
        q_a_norm_view.cols != MTP_ATTN_Q_LORA ||
        q_b_view.rows != MTP_ATTN_N_HEAD * MTP_ATTN_HEAD_DIM ||
        q_b_view.cols != MTP_ATTN_Q_LORA ||
        kv_view.rows != MTP_ATTN_HEAD_DIM ||
        kv_view.cols != MTP_ATTN_N_EMBD ||
        kv_norm_view.cols != MTP_ATTN_HEAD_DIM ||
        sinks_view.cols != MTP_ATTN_N_HEAD ||
        out_a_view.rows != MTP_ATTN_OUT_LOW_DIM ||
        out_a_view.cols != MTP_ATTN_OUT_GROUP_DIM ||
        out_b_view.rows != MTP_ATTN_N_EMBD ||
        out_b_view.cols != MTP_ATTN_OUT_LOW_DIM) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: unexpected integrated attention layout\n");
        return 1;
    }

    const float *hc_fn = (const float *)tensor_bytes(sidecar, "mtp.0.hc_attn_fn.weight");
    const float *hc_scale = (const float *)tensor_bytes(sidecar, "mtp.0.hc_attn_scale.weight");
    const float *hc_base = (const float *)tensor_bytes(sidecar, "mtp.0.hc_attn_base.weight");
    const float *attn_norm = (const float *)tensor_bytes(sidecar, "mtp.0.attn_norm.weight");
    const unsigned char *q_a_w = tensor_bytes(sidecar, "mtp.0.attn_q_a.weight");
    const float *q_a_norm = (const float *)tensor_bytes(sidecar, "mtp.0.attn_q_a_norm.weight");
    const unsigned char *q_b_w = tensor_bytes(sidecar, "mtp.0.attn_q_b.weight");
    const unsigned char *kv_w = tensor_bytes(sidecar, "mtp.0.attn_kv.weight");
    const float *kv_norm = (const float *)tensor_bytes(sidecar, "mtp.0.attn_kv_a_norm.weight");
    const float *sinks = (const float *)tensor_bytes(sidecar, "mtp.0.attn_sinks.weight");
    const unsigned char *out_a_w = tensor_bytes(sidecar, "mtp.0.attn_output_a.weight");
    const unsigned char *out_b_w = tensor_bytes(sidecar, "mtp.0.attn_output_b.weight");
    if (!hc_fn || !hc_scale || !hc_base || !attn_norm || !q_a_w ||
        !q_a_norm || !q_b_w || !kv_w || !kv_norm || !sinks ||
        !out_a_w || !out_b_w) {
        return 1;
    }

    const uint64_t hc_bytes = (uint64_t)MTP_ATTN_HC_DIM * sizeof(float);
    const uint64_t embd_bytes = (uint64_t)MTP_ATTN_N_EMBD * sizeof(float);
    const uint64_t mix_bytes = (uint64_t)MTP_ATTN_HC_MIX * sizeof(float);
    const uint64_t q_lora_bytes = (uint64_t)MTP_ATTN_Q_LORA * sizeof(float);
    const uint64_t heads_values = (uint64_t)MTP_ATTN_N_HEAD * MTP_ATTN_HEAD_DIM;
    const uint64_t heads_bytes = heads_values * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)MTP_ATTN_HEAD_DIM * sizeof(float);
    const uint64_t raw_bytes =
        (uint64_t)MTP_ATTN_RAW_CAP * MTP_ATTN_HEAD_DIM * sizeof(float);
    const uint64_t low_bytes = (uint64_t)MTP_ATTN_OUT_LOW_DIM * sizeof(float);

    float *input_hc = (float *)malloc((size_t)hc_bytes);
    float *ref_hc_norm = (float *)malloc((size_t)hc_bytes);
    float *ref_hc_mix = (float *)malloc((size_t)mix_bytes);
    float *ref_attn_cur = (float *)malloc((size_t)embd_bytes);
    float *ref_attn_split = (float *)malloc((size_t)mix_bytes);
    float *ref_attn_norm = (float *)malloc((size_t)embd_bytes);
    float *ref_q = (float *)malloc((size_t)q_lora_bytes);
    float *ref_q_norm = (float *)malloc((size_t)q_lora_bytes);
    float *ref_q_heads = (float *)malloc((size_t)heads_bytes);
    float *ref_kv = (float *)malloc((size_t)kv_bytes);
    float *ref_raw = (float *)calloc(1, (size_t)raw_bytes);
    float *ref_heads = (float *)malloc((size_t)heads_bytes);
    float *ref_low = (float *)malloc((size_t)low_bytes);
    float *ref_attn_out = (float *)malloc((size_t)embd_bytes);
    float *ref_next_hc = (float *)malloc((size_t)hc_bytes);
    ds4_gpu_tensor *input_hc_t = NULL;
    ds4_gpu_tensor *hc_norm_a = NULL;
    ds4_gpu_tensor *hc_mix_a = NULL;
    ds4_gpu_tensor *attn_cur_a = NULL;
    ds4_gpu_tensor *attn_split_a = NULL;
    ds4_gpu_tensor *attn_norm_a = NULL;
    ds4_gpu_tensor *q_a = NULL;
    ds4_gpu_tensor *q_norm_a = NULL;
    ds4_gpu_tensor *q_heads_a = NULL;
    ds4_gpu_tensor *kv_a = NULL;
    ds4_gpu_tensor *raw_a = NULL;
    ds4_gpu_tensor *heads_a = NULL;
    ds4_gpu_tensor *low_a = NULL;
    ds4_gpu_tensor *attn_out_a = NULL;
    ds4_gpu_tensor *next_hc_a = NULL;

    int rc = 1;
    double global_max_abs = 0.0;
    if (!input_hc || !ref_hc_norm || !ref_hc_mix || !ref_attn_cur ||
        !ref_attn_split || !ref_attn_norm || !ref_q || !ref_q_norm ||
        !ref_q_heads || !ref_kv || !ref_raw || !ref_heads || !ref_low ||
        !ref_attn_out || !ref_next_hc) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: integrated host allocation failed\n");
        goto done;
    }
    fill_hc_state(input_hc);
    rms_norm_plain_host(ref_hc_norm,
                        input_hc,
                        MTP_ATTN_HC_DIM,
                        MTP_ATTN_RMS_EPS);
    matmul_f32_host(ref_hc_mix,
                    hc_fn,
                    MTP_ATTN_HC_MIX,
                    MTP_ATTN_HC_DIM,
                    ref_hc_norm);
    hc_split_sinkhorn_host(ref_attn_split,
                           ref_hc_mix,
                           hc_scale,
                           hc_base,
                           MTP_ATTN_N_HC,
                           MTP_ATTN_HC_SINKHORN_ITERS,
                           MTP_ATTN_HC_EPS);
    hc_weighted_sum_host(ref_attn_cur,
                         input_hc,
                         ref_attn_split,
                         MTP_ATTN_N_EMBD,
                         MTP_ATTN_N_HC);
    rms_norm_weight_host(ref_attn_norm,
                         ref_attn_cur,
                         attn_norm,
                         MTP_ATTN_N_EMBD,
                         MTP_ATTN_RMS_EPS);
    if (matmul_q8_0_host(ref_q,
                         q_a_w,
                         MTP_ATTN_N_EMBD,
                         MTP_ATTN_Q_LORA,
                         ref_attn_norm,
                         1) != 0) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: host q_a reference failed\n");
        goto done;
    }
    rms_norm_weight_host(ref_q_norm,
                         ref_q,
                         q_a_norm,
                         MTP_ATTN_Q_LORA,
                         MTP_ATTN_RMS_EPS);
    if (matmul_q8_0_host(ref_q_heads,
                         q_b_w,
                         MTP_ATTN_Q_LORA,
                         MTP_ATTN_N_HEAD * MTP_ATTN_HEAD_DIM,
                         ref_q_norm,
                         1) != 0) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: host q_b reference failed\n");
        goto done;
    }
    head_rms_norm_host(ref_q_heads,
                       MTP_ATTN_N_HEAD,
                       MTP_ATTN_HEAD_DIM,
                       MTP_ATTN_RMS_EPS);
    if (matmul_q8_0_host(ref_kv,
                         kv_w,
                         MTP_ATTN_N_EMBD,
                         MTP_ATTN_HEAD_DIM,
                         ref_attn_norm,
                         1) != 0) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: host kv reference failed\n");
        goto done;
    }
    rms_norm_weight_host(ref_kv,
                         ref_kv,
                         kv_norm,
                         MTP_ATTN_HEAD_DIM,
                         MTP_ATTN_RMS_EPS);
    dsv4_fp8_kv_quantize_row_inplace_host(ref_kv, MTP_ATTN_HEAD_DIM, MTP_ATTN_N_ROT);
    f16_round_inplace_host(ref_kv, MTP_ATTN_HEAD_DIM);
    memcpy(ref_raw, ref_kv, (size_t)kv_bytes);
    attention_ref(ref_heads, ref_q_heads, ref_raw, sinks, 1, 0);
    for (uint32_t g = 0; g < MTP_ATTN_OUT_GROUPS; g++) {
        if (matmul_q8_0_host(ref_low + (uint64_t)g * MTP_ATTN_OUT_GROUP_RANK,
                             out_a_w + (uint64_t)g * MTP_ATTN_OUT_GROUP_RANK *
                                       out_a_view.row_stride_bytes,
                             MTP_ATTN_OUT_GROUP_DIM,
                             MTP_ATTN_OUT_GROUP_RANK,
                             ref_heads + (uint64_t)g * MTP_ATTN_OUT_GROUP_DIM,
                             1) != 0) {
            fprintf(stderr, "ds4-v100-mtp-attn-smoke: host grouped output-a reference failed\n");
            goto done;
        }
    }
    if (matmul_q8_0_host(ref_attn_out,
                         out_b_w,
                         MTP_ATTN_OUT_LOW_DIM,
                         MTP_ATTN_N_EMBD,
                         ref_low,
                         1) != 0) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: host output-b reference failed\n");
        goto done;
    }
    hc_expand_split_host(ref_next_hc,
                         ref_attn_out,
                         input_hc,
                         ref_attn_split,
                         MTP_ATTN_N_EMBD,
                         MTP_ATTN_N_HC);

#define ALLOC_TENSOR(name, bytes)                                                   \
    do {                                                                            \
        name = ds4_gpu_tensor_alloc((bytes));                                       \
        if (!(name)) {                                                              \
            fprintf(stderr,                                                         \
                    "ds4-v100-mtp-attn-smoke: failed to allocate %s\n",             \
                    #name);                                                         \
            goto done;                                                              \
        }                                                                           \
    } while (0)

    ALLOC_TENSOR(input_hc_t, hc_bytes);
    ALLOC_TENSOR(hc_norm_a, hc_bytes);
    ALLOC_TENSOR(hc_mix_a, mix_bytes);
    ALLOC_TENSOR(attn_cur_a, embd_bytes);
    ALLOC_TENSOR(attn_split_a, mix_bytes);
    ALLOC_TENSOR(attn_norm_a, embd_bytes);
    ALLOC_TENSOR(q_a, q_lora_bytes);
    ALLOC_TENSOR(q_norm_a, q_lora_bytes);
    ALLOC_TENSOR(q_heads_a, heads_bytes);
    ALLOC_TENSOR(kv_a, kv_bytes);
    ALLOC_TENSOR(raw_a, raw_bytes);
    ALLOC_TENSOR(heads_a, heads_bytes);
    ALLOC_TENSOR(low_a, low_bytes);
    ALLOC_TENSOR(attn_out_a, embd_bytes);
    ALLOC_TENSOR(next_hc_a, hc_bytes);

#undef ALLOC_TENSOR

    if (!ds4_gpu_tensor_write(input_hc_t, 0, input_hc, hc_bytes) ||
        !ds4_gpu_tensor_fill_f32(raw_a, 0.0f, MTP_ATTN_RAW_CAP * MTP_ATTN_HEAD_DIM)) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: integrated initialization failed\n");
        goto done;
    }

    const double t0 = now_ms();
    if (!ds4_gpu_rms_norm_plain_tensor(hc_norm_a,
                                       input_hc_t,
                                       MTP_ATTN_HC_DIM,
                                       MTP_ATTN_RMS_EPS)) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: integrated HC plain norm failed\n");
        goto done;
    }
    if (ds4_gpu_arena_f32_matmul_f32(arena, &hc_fn_view, hc_norm_a, hc_mix_a) != 0) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: integrated HC function failed\n");
        goto done;
    }
    if (ds4_gpu_arena_hc_split_weighted_sum_tensor(arena,
                                                  &hc_scale_view,
                                                  &hc_base_view,
                                                  attn_cur_a,
                                                  attn_split_a,
                                                  hc_mix_a,
                                                  input_hc_t,
                                                  MTP_ATTN_N_EMBD,
                                                  MTP_ATTN_N_HC,
                                                  MTP_ATTN_HC_SINKHORN_ITERS,
                                                  MTP_ATTN_HC_EPS) != 0) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: integrated HC split failed\n");
        goto done;
    }
    if (ds4_gpu_arena_f32_rms_norm_f32(arena,
                                       &attn_norm_view,
                                       attn_cur_a,
                                       attn_norm_a,
                                       MTP_ATTN_N_EMBD,
                                       1,
                                       MTP_ATTN_RMS_EPS) != 0) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: integrated attention norm failed\n");
        goto done;
    }
    if (ds4_gpu_arena_q8_0_matmul_f32(arena, &q_a_view, attn_norm_a, q_a, 1) != 0) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: integrated q_a projection failed\n");
        goto done;
    }
    if (ds4_gpu_arena_f32_rms_norm_f32(arena,
                                       &q_a_norm_view,
                                       q_a,
                                       q_norm_a,
                                       MTP_ATTN_Q_LORA,
                                       1,
                                       MTP_ATTN_RMS_EPS) != 0) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: integrated q_a norm failed\n");
        goto done;
    }
    if (ds4_gpu_arena_q8_0_matmul_f32(arena, &q_b_view, q_norm_a, q_heads_a, 1) != 0 ||
        !ds4_gpu_head_rms_norm_tensor(q_heads_a,
                                      1,
                                      MTP_ATTN_N_HEAD,
                                      MTP_ATTN_HEAD_DIM,
                                      MTP_ATTN_RMS_EPS)) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: integrated q_b/head norm failed\n");
        goto done;
    }
    if (ds4_gpu_arena_q8_0_matmul_f32(arena, &kv_view, attn_norm_a, kv_a, 1) != 0 ||
        ds4_gpu_arena_f32_rms_norm_f32(arena,
                                       &kv_norm_view,
                                       kv_a,
                                       kv_a,
                                       MTP_ATTN_HEAD_DIM,
                                       1,
                                       MTP_ATTN_RMS_EPS) != 0) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: integrated kv projection/norm failed\n");
        goto done;
    }
    if (!ds4_gpu_kv_fp8_store_raw_tensor(kv_a,
                                         raw_a,
                                         MTP_ATTN_RAW_CAP,
                                         0,
                                         MTP_ATTN_HEAD_DIM,
                                         MTP_ATTN_N_ROT)) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: integrated raw KV store failed\n");
        goto done;
    }
    if (ds4_gpu_arena_attention_decode_heads_tensor(arena,
                                                   &sinks_view,
                                                   heads_a,
                                                   q_heads_a,
                                                   raw_a,
                                                   1,
                                                   MTP_ATTN_RAW_CAP,
                                                   0,
                                                   NULL,
                                                   0,
                                                   NULL,
                                                   0,
                                                   MTP_ATTN_N_HEAD,
                                                   MTP_ATTN_HEAD_DIM) != 0) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: integrated attention decode failed\n");
        goto done;
    }
    if (grouped_output_arena(sidecar,
                             &out_a_view,
                             &out_b_view,
                             heads_a,
                             low_a,
                             attn_out_a) != 0) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: integrated grouped output failed\n");
        goto done;
    }
    if (!ds4_gpu_hc_expand_split_tensor(next_hc_a,
                                        attn_out_a,
                                        input_hc_t,
                                        attn_split_a,
                                        MTP_ATTN_N_EMBD,
                                        MTP_ATTN_N_HC) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: integrated HC expand failed\n");
        goto done;
    }

    double max_abs = 0.0;
    if (compare_device_to_host("q_heads", q_heads_a, ref_q_heads, heads_values,
                               max_abs_tol, report, &max_abs) != 0) goto done;
    if (max_abs > global_max_abs) global_max_abs = max_abs;
    if (compare_device_to_host("kv_row", kv_a, ref_kv, MTP_ATTN_HEAD_DIM,
                               max_abs_tol, report, &max_abs) != 0) goto done;
    if (max_abs > global_max_abs) global_max_abs = max_abs;
    if (compare_device_to_host("heads", heads_a, ref_heads, heads_values,
                               max_abs_tol, report, &max_abs) != 0) goto done;
    if (max_abs > global_max_abs) global_max_abs = max_abs;
    if (compare_device_to_host("attn_out", attn_out_a, ref_attn_out, MTP_ATTN_N_EMBD,
                               max_abs_tol, report, &max_abs) != 0) goto done;
    if (max_abs > global_max_abs) global_max_abs = max_abs;
    if (compare_device_to_host("next_hc", next_hc_a, ref_next_hc, MTP_ATTN_HC_DIM,
                               max_abs_tol, report, &max_abs) != 0) goto done;
    if (max_abs > global_max_abs) global_max_abs = max_abs;

    const double t1 = now_ms();
    fprintf(report,
            "mtp_attn_integrated_summary\tarena_cpu_ms=%.3f"
            "\tglobal_max_abs=%.9g\tPASS\n",
            t1 - t0,
            global_max_abs);
    rc = 0;

done:
    ds4_gpu_tensor_free(next_hc_a);
    ds4_gpu_tensor_free(attn_out_a);
    ds4_gpu_tensor_free(low_a);
    ds4_gpu_tensor_free(heads_a);
    ds4_gpu_tensor_free(raw_a);
    ds4_gpu_tensor_free(kv_a);
    ds4_gpu_tensor_free(q_heads_a);
    ds4_gpu_tensor_free(q_norm_a);
    ds4_gpu_tensor_free(q_a);
    ds4_gpu_tensor_free(attn_norm_a);
    ds4_gpu_tensor_free(attn_split_a);
    ds4_gpu_tensor_free(attn_cur_a);
    ds4_gpu_tensor_free(hc_mix_a);
    ds4_gpu_tensor_free(hc_norm_a);
    ds4_gpu_tensor_free(input_hc_t);
    free(ref_next_hc);
    free(ref_attn_out);
    free(ref_low);
    free(ref_heads);
    free(ref_raw);
    free(ref_kv);
    free(ref_q_heads);
    free(ref_q_norm);
    free(ref_q);
    free(ref_attn_norm);
    free(ref_attn_split);
    free(ref_attn_cur);
    free(ref_hc_mix);
    free(ref_hc_norm);
    free(input_hc);
    return rc;
}

static int run_attention(ds4_mtp_sidecar *sidecar,
                         double max_abs_tol,
                         FILE *report) {
    char err[512] = {0};
    ds4_gpu_source_row_view sinks_view;
    if (ds4_mtp_sidecar_f32_vector_view(sidecar,
                                             "mtp.0.attn_sinks.weight",
                                             &sinks_view,
                                             err,
                                             sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-attn-smoke: %s\n",
                err[0] ? err : "failed to bind MTP attn sinks view");
        return 1;
    }
    if (sinks_view.rows != 1u || sinks_view.cols != MTP_ATTN_N_HEAD) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: unexpected attn_sinks layout\n");
        return 1;
    }

    const ds4_mtp_sidecar_tensor_info *sinks_tensor =
        ds4_mtp_sidecar_tensor(sidecar, "mtp.0.attn_sinks.weight");
    const float *sinks = (const float *)(
            (const unsigned char *)ds4_mtp_sidecar_map(sidecar) +
            sinks_tensor->source_offset);

    const uint64_t heads_values = (uint64_t)MTP_ATTN_N_HEAD * MTP_ATTN_HEAD_DIM;
    const uint64_t heads_bytes = heads_values * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)MTP_ATTN_HEAD_DIM * sizeof(float);
    const uint64_t raw_bytes =
        (uint64_t)MTP_ATTN_RAW_CAP * MTP_ATTN_HEAD_DIM * sizeof(float);

    float *q = (float *)malloc((size_t)heads_bytes);
    float *kv = (float *)malloc((size_t)kv_bytes);
    float *raw_cache = (float *)calloc(1, (size_t)raw_bytes);
    float *got = (float *)malloc((size_t)heads_bytes);
    float *ref = (float *)malloc((size_t)heads_bytes);
    ds4_gpu_tensor *q_t = NULL;
    ds4_gpu_tensor *kv_t = NULL;
    ds4_gpu_tensor *raw_t = NULL;
    ds4_gpu_tensor *heads_t = NULL;
    if (!q || !kv || !raw_cache || !got || !ref) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: host allocation failed\n");
        free(ref);
        free(got);
        free(raw_cache);
        free(kv);
        free(q);
        return 1;
    }

    int rc = 1;
    fill_q(q);
    q_t = ds4_gpu_tensor_alloc(heads_bytes);
    kv_t = ds4_gpu_tensor_alloc(kv_bytes);
    raw_t = ds4_gpu_tensor_alloc(raw_bytes);
    heads_t = ds4_gpu_tensor_alloc(heads_bytes);
    if (!q_t || !kv_t || !raw_t || !heads_t) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: device tensor allocation failed\n");
        goto done;
    }
    if (!ds4_gpu_tensor_write(q_t, 0, q, heads_bytes) ||
        !ds4_gpu_tensor_fill_f32(raw_t, 0.0f, MTP_ATTN_RAW_CAP * MTP_ATTN_HEAD_DIM)) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: device initialization failed\n");
        goto done;
    }

    const uint32_t check_positions[] = {0, 1, 127, 128, 129};
    uint32_t next_check = 0;
    double global_max_abs = 0.0;
    double global_max_rel = 0.0;
    uint64_t global_max_i = 0;
    const double t0 = now_ms();
    for (uint32_t pos = 0; pos <= 129; pos++) {
        fill_kv_for_pos(kv, pos);
        float stored[MTP_ATTN_HEAD_DIM];
        memcpy(stored, kv, sizeof(stored));
        dsv4_fp8_kv_quantize_row_inplace_host(stored, MTP_ATTN_HEAD_DIM, MTP_ATTN_N_ROT);
        f16_round_inplace_host(stored, MTP_ATTN_HEAD_DIM);
        const uint32_t raw_row = pos % MTP_ATTN_RAW_CAP;
        memcpy(raw_cache + (uint64_t)raw_row * MTP_ATTN_HEAD_DIM, stored, sizeof(stored));

        if (!ds4_gpu_tensor_write(kv_t, 0, kv, kv_bytes) ||
            !ds4_gpu_kv_fp8_store_raw_tensor(kv_t,
                                             raw_t,
                                             MTP_ATTN_RAW_CAP,
                                             raw_row,
                                             MTP_ATTN_HEAD_DIM,
                                             MTP_ATTN_N_ROT) ||
            !ds4_gpu_synchronize()) {
            fprintf(stderr, "ds4-v100-mtp-attn-smoke: raw KV store failed at pos %u\n", pos);
            goto done;
        }

        if (next_check >= sizeof(check_positions) / sizeof(check_positions[0]) ||
            pos != check_positions[next_check]) {
            continue;
        }

        const uint32_t n_raw = pos + 1u < MTP_ATTN_RAW_CAP
            ? pos + 1u
            : MTP_ATTN_RAW_CAP;
        const uint32_t raw_start = (pos + 1u - n_raw) % MTP_ATTN_RAW_CAP;
        if (ds4_gpu_arena_attention_decode_heads_tensor(
                    ds4_mtp_sidecar_arena(sidecar),
                    &sinks_view,
                    heads_t,
                    q_t,
                    raw_t,
                    n_raw,
                    MTP_ATTN_RAW_CAP,
                    raw_start,
                    NULL,
                    0,
                    NULL,
                    0,
                    MTP_ATTN_N_HEAD,
                    MTP_ATTN_HEAD_DIM) != 0 ||
            !ds4_gpu_synchronize()) {
            fprintf(stderr, "ds4-v100-mtp-attn-smoke: arena attention failed at pos %u\n", pos);
            goto done;
        }

        attention_ref(ref, q, raw_cache, sinks, n_raw, raw_start);
        if (!ds4_gpu_tensor_read(heads_t, 0, got, heads_bytes)) {
            fprintf(stderr, "ds4-v100-mtp-attn-smoke: head readback failed at pos %u\n", pos);
            goto done;
        }
        double max_rel = 0.0;
        uint64_t max_i = 0;
        const double max_abs = compare_outputs(got, ref, heads_values, &max_rel, &max_i);
        if (max_abs > global_max_abs) {
            global_max_abs = max_abs;
            global_max_rel = max_rel;
            global_max_i = max_i;
        }
        fprintf(report,
                "mtp_attn_step\tpos=%u\traw_row=%u\tn_raw=%u\traw_start=%u"
                "\tmax_abs=%.9g\tmax_rel=%.9g\tmax_i=%" PRIu64 "\t%s\n",
                pos,
                raw_row,
                n_raw,
                raw_start,
                max_abs,
                max_rel,
                max_i,
                max_abs <= max_abs_tol ? "PASS" : "FAIL");
        if (pos == 129u) {
            const int current_visible = raw_row_visible(raw_start, n_raw, raw_row);
            const int oldest_visible = raw_row_visible(raw_start, n_raw, raw_start);
            fprintf(report,
                    "mtp_attn_wrap\tpos=129\traw_start=%u\tcurrent_row=%u"
                    "\tcurrent_visible=%d\toldest_row=%u\toldest_visible=%d\t%s\n",
                    raw_start,
                    raw_row,
                    current_visible,
                    raw_start,
                    oldest_visible,
                    raw_start == 2u &&
                    raw_row == 1u &&
                    current_visible &&
                    oldest_visible ? "PASS" : "FAIL");
            if (raw_start != 2u ||
                raw_row != 1u ||
                !current_visible ||
                !oldest_visible) {
                goto done;
            }
        }
        if (max_abs > max_abs_tol) {
            fprintf(stderr,
                    "ds4-v100-mtp-attn-smoke: pos %u max_abs %.9g exceeds %.9g\n",
                    pos,
                    max_abs,
                    max_abs_tol);
            goto done;
        }
        next_check++;
    }
    const double t1 = now_ms();
    fprintf(report,
            "mtp_attn_summary\tchecks=%u\tarena_ms=%.3f\tglobal_max_abs=%.9g"
            "\tglobal_max_rel=%.9g\tglobal_max_i=%" PRIu64 "\tPASS\n",
            next_check,
            t1 - t0,
            global_max_abs,
            global_max_rel,
            global_max_i);
    fprintf(report, "mtp_attn_raw_smoke\tPASS\n");
    rc = 0;

done:
    ds4_gpu_tensor_free(heads_t);
    ds4_gpu_tensor_free(raw_t);
    ds4_gpu_tensor_free(kv_t);
    ds4_gpu_tensor_free(q_t);
    free(ref);
    free(got);
    free(raw_cache);
    free(kv);
    free(q);
    return rc;
}

int main(int argc, char **argv) {
    options opt = parse_options(argc, argv);
    FILE *report = stdout;
    if (opt.report_path && opt.report_path[0]) {
        report = fopen(opt.report_path, "w");
        if (!report) {
            fprintf(stderr,
                    "ds4-v100-mtp-attn-smoke: cannot open report %s\n",
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
    fprintf(report, "integrated_max_abs_tol\t%.9g\n", opt.integrated_max_abs_tol);
    ds4_gpu_print_topology_report(report);

    if (opt.require_gpus > 0 && device_count < opt.require_gpus) {
        fprintf(stderr,
                "ds4-v100-mtp-attn-smoke: visible devices %d < required %d\n",
                device_count,
                opt.require_gpus);
        goto done;
    }
    if (device_count <= 0) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: no CUDA devices visible\n");
        goto done;
    }
    if (opt.gpu >= device_count) {
        fprintf(stderr,
                "ds4-v100-mtp-attn-smoke: target gpu %d outside visible device count %d\n",
                opt.gpu,
                device_count);
        goto done;
    }
    if (!ds4_gpu_set_device(opt.gpu)) {
        fprintf(stderr, "ds4-v100-mtp-attn-smoke: failed to select gpu %d\n", opt.gpu);
        goto done;
    }

    ds4_mtp_sidecar_options sidecar_opts;
    ds4_mtp_sidecar_options_init(&sidecar_opts);
    sidecar_opts.mtp_path = opt.mtp_model;
    sidecar_opts.gpu = opt.gpu;
    sidecar_opts.require_device_arena = true;
    if (ds4_mtp_sidecar_open(&sidecar, &sidecar_opts, report, err, sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-attn-smoke: %s\n",
                err[0] ? err : "MTP sidecar open failed");
        goto done;
    }

    uint64_t reserve_bytes = (uint64_t)opt.reserve_mib * 1024ull * 1024ull;
    uint64_t free_after =
        ds4_gpu_arena_free_after_upload_bytes(ds4_mtp_sidecar_arena(sidecar));
    fprintf(report, "reserve_bytes\t%" PRIu64 "\n", reserve_bytes);
    if (free_after < reserve_bytes) {
        fprintf(stderr,
                "ds4-v100-mtp-attn-smoke: gpu %d free bytes %" PRIu64
                " below reserve %" PRIu64 "\n",
                opt.gpu,
                free_after,
                reserve_bytes);
        goto done;
    }

    if (run_attention(sidecar, opt.max_abs_tol, report) != 0) goto done;
    if (run_integrated_attention(sidecar, opt.integrated_max_abs_tol, report) != 0) goto done;
    fprintf(report, "mtp_attn_smoke\tPASS\n");
    if (opt.report_path && opt.report_path[0]) {
        printf("mtp_attn_smoke\tPASS\treport=%s\n", opt.report_path);
    }
    rc = 0;

done:
    ds4_mtp_sidecar_close(sidecar);
    if (report && report != stdout) fclose(report);
    return rc;
}
