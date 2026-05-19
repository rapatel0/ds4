#include "ds4_v100_mtp.h"

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
    MTP_ATTN_N_HEAD = 64,
    MTP_ATTN_HEAD_DIM = 512,
    MTP_ATTN_N_ROT = 64,
    MTP_ATTN_RAW_CAP = 128,
};

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
            "Usage: ds4-v100-mtp-attn-smoke --mtp-model FILE [options]\n"
            "\n"
            "Options:\n"
            "  --gpu N                 Upload and execute on CUDA device N. Default: 7\n"
            "  --require-gpus N        Require at least N visible CUDA devices\n"
            "  --reserve-mib N         Require this much free memory after upload. Default: 4096\n"
            "  --max-abs-tol F         Max allowed attention-head delta. Default: 0.002\n"
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

static int raw_row_visible(uint32_t raw_start, uint32_t n_raw, uint32_t row) {
    for (uint32_t r = 0; r < n_raw; r++) {
        if (((raw_start + r) % MTP_ATTN_RAW_CAP) == row) return 1;
    }
    return 0;
}

static int run_attention(ds4_v100_mtp_sidecar *sidecar,
                         double max_abs_tol,
                         FILE *report) {
    char err[512] = {0};
    ds4_gpu_source_row_view sinks_view;
    if (ds4_v100_mtp_sidecar_f32_vector_view(sidecar,
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
        ds4_v100_mtp_sidecar_tensor(sidecar, "mtp.0.attn_sinks.weight");
    const float *sinks = (const float *)(
            (const unsigned char *)ds4_v100_mtp_sidecar_map(sidecar) +
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
                    ds4_v100_mtp_sidecar_arena(sidecar),
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
    fprintf(report, "mtp_attn_smoke\tPASS\n");
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
    ds4_v100_mtp_sidecar *sidecar = NULL;
    int device_count = ds4_gpu_device_count();
    fprintf(report, "visible_devices\t%d\n", device_count);
    fprintf(report, "target_gpu\t%d\n", opt.gpu);
    fprintf(report, "reserve_mib\t%d\n", opt.reserve_mib);
    fprintf(report, "max_abs_tol\t%.9g\n", opt.max_abs_tol);
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

    ds4_v100_mtp_sidecar_options sidecar_opts;
    ds4_v100_mtp_sidecar_options_init(&sidecar_opts);
    sidecar_opts.mtp_path = opt.mtp_model;
    sidecar_opts.gpu = opt.gpu;
    sidecar_opts.require_device_arena = true;
    if (ds4_v100_mtp_sidecar_open(&sidecar, &sidecar_opts, report, err, sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-attn-smoke: %s\n",
                err[0] ? err : "MTP sidecar open failed");
        goto done;
    }

    uint64_t reserve_bytes = (uint64_t)opt.reserve_mib * 1024ull * 1024ull;
    uint64_t free_after =
        ds4_gpu_arena_free_after_upload_bytes(ds4_v100_mtp_sidecar_arena(sidecar));
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
    if (opt.report_path && opt.report_path[0]) {
        printf("mtp_attn_smoke\tPASS\treport=%s\n", opt.report_path);
    }
    rc = 0;

done:
    ds4_v100_mtp_sidecar_close(sidecar);
    if (report && report != stdout) fclose(report);
    return rc;
}
