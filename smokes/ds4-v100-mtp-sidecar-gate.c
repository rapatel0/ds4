#include "ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tools/ds4-v100-mtp-sidecar-gate --mtp-model FILE [options]\n"
            "\n"
            "Options:\n"
            "  --mtp-model FILE   DeepSeek-V4 Flash MTP sidecar GGUF\n"
            "  --report FILE      Write the tensor/layout report to FILE\n"
            "  --help             Show this help\n");
}

int main(int argc, char **argv) {
    const char *mtp_model = NULL;
    const char *report_path = NULL;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--mtp-model")) {
            if (++i >= argc) {
                fprintf(stderr, "ds4-v100-mtp-sidecar-gate: --mtp-model requires a value\n");
                return 2;
            }
            mtp_model = argv[i];
        } else if (!strcmp(arg, "--report")) {
            if (++i >= argc) {
                fprintf(stderr, "ds4-v100-mtp-sidecar-gate: --report requires a value\n");
                return 2;
            }
            report_path = argv[i];
        } else if (!strcmp(arg, "--help")) {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "ds4-v100-mtp-sidecar-gate: unknown option: %s\n", arg);
            usage(stderr);
            return 2;
        }
    }

    if (!mtp_model || !mtp_model[0]) {
        fprintf(stderr, "ds4-v100-mtp-sidecar-gate: --mtp-model is required\n");
        usage(stderr);
        return 2;
    }

    FILE *report = stdout;
    if (report_path && report_path[0]) {
        report = fopen(report_path, "w");
        if (!report) {
            perror("ds4-v100-mtp-sidecar-gate: cannot open report");
            return 1;
        }
    }

    char err[512];
    const int rc = ds4_mtp_sidecar_report(mtp_model, report, err, sizeof(err));
    if (report_path && report) {
        if (fclose(report) != 0) {
            perror("ds4-v100-mtp-sidecar-gate: cannot close report");
            return 1;
        }
    }
    if (rc != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-sidecar-gate: %s\n",
                err[0] ? err : "MTP sidecar validation failed");
        return 1;
    }

    if (report_path && report_path[0]) {
        printf("mtp_sidecar\tPASS\treport=%s\n", report_path);
    }
    return 0;
}
