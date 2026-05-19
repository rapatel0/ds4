#include "ds4_v100_mtp.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
