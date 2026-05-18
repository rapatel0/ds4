#include "ds4_v100_mtp.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *mtp_model;
    const char *report_path;
    int gpu;
    int require_gpus;
    int reserve_mib;
    bool allow_host_stub;
} options;

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-v100-mtp-residency-smoke --mtp-model FILE [options]\n"
            "\n"
            "Options:\n"
            "  --gpu N                 Upload the MTP sidecar to CUDA device N. Default: 7\n"
            "  --require-gpus N        Require at least N visible CUDA devices\n"
            "  --reserve-mib N         Require this much free memory after upload. Default: 4096\n"
            "  --report FILE           Write report to FILE instead of stdout\n"
            "  --allow-host-stub       Allow the local non-CUDA arena stub\n");
}

static const char *need_arg(int *i, int argc, char **argv, const char *arg) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-v100-mtp-residency-smoke: %s requires an argument\n", arg);
        exit(2);
    }
    return argv[++*i];
}

static int parse_int(const char *s, const char *arg) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || !end || *end || v < 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-v100-mtp-residency-smoke: bad integer for %s: %s\n", arg, s);
        exit(2);
    }
    return (int)v;
}

static options parse_options(int argc, char **argv) {
    options opt;
    memset(&opt, 0, sizeof(opt));
    opt.gpu = 7;
    opt.reserve_mib = 4096;
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
        } else if (!strcmp(arg, "--allow-host-stub")) {
            opt.allow_host_stub = true;
        } else {
            fprintf(stderr, "ds4-v100-mtp-residency-smoke: unknown option: %s\n", arg);
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

int main(int argc, char **argv) {
    options opt = parse_options(argc, argv);
    FILE *report = stdout;
    if (opt.report_path && opt.report_path[0]) {
        report = fopen(opt.report_path, "w");
        if (!report) {
            fprintf(stderr,
                    "ds4-v100-mtp-residency-smoke: cannot open report %s\n",
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
    ds4_gpu_print_topology_report(report);

    if (opt.require_gpus > 0 && device_count < opt.require_gpus) {
        fprintf(stderr,
                "ds4-v100-mtp-residency-smoke: visible devices %d < required %d\n",
                device_count,
                opt.require_gpus);
        goto done;
    }
    if (device_count > 0 && opt.gpu >= device_count) {
        fprintf(stderr,
                "ds4-v100-mtp-residency-smoke: target gpu %d outside visible device count %d\n",
                opt.gpu,
                device_count);
        goto done;
    }

    ds4_v100_mtp_sidecar_options sidecar_opts;
    ds4_v100_mtp_sidecar_options_init(&sidecar_opts);
    sidecar_opts.mtp_path = opt.mtp_model;
    sidecar_opts.gpu = opt.gpu;
    sidecar_opts.require_device_arena = !opt.allow_host_stub;
    if (ds4_v100_mtp_sidecar_open(&sidecar, &sidecar_opts, report, err, sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-residency-smoke: %s\n",
                err[0] ? err : "MTP sidecar residency failed");
        goto done;
    }

    if (device_count > 0) {
        uint64_t reserve_bytes = (uint64_t)opt.reserve_mib * 1024ull * 1024ull;
        uint64_t free_after =
            ds4_gpu_arena_free_after_upload_bytes(ds4_v100_mtp_sidecar_arena(sidecar));
        fprintf(report, "reserve_bytes\t%" PRIu64 "\n", reserve_bytes);
        if (free_after < reserve_bytes) {
            fprintf(stderr,
                    "ds4-v100-mtp-residency-smoke: gpu %d free bytes %" PRIu64
                    " below reserve %" PRIu64 "\n",
                    opt.gpu,
                    free_after,
                    reserve_bytes);
            goto done;
        }
    }

    fprintf(report, "mtp_residency_smoke\tPASS\n");
    if (opt.report_path && opt.report_path[0]) {
        printf("mtp_residency_smoke\tPASS\treport=%s\n", opt.report_path);
    }
    rc = 0;

done:
    ds4_v100_mtp_sidecar_close(sidecar);
    if (report && report != stdout) fclose(report);
    return rc;
}
