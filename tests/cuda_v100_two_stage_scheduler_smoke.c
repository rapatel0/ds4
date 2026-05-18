#include "ds4_v100_scheduler.h"

#include <errno.h>
#include <fcntl.h>
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

typedef struct {
    const unsigned char *ptr;
    uint64_t size;
    int fd;
} model_map;

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_v100_two_stage_scheduler_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/cuda_v100_two_stage_scheduler_smoke --index FILE --model FILE [--token N] [--position N]\n");
}

static int parse_int_arg(const char *s, const char *name, int max_v) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || !*s || !end || *end != '\0' || v < 0 || v > max_v) {
        fprintf(stderr, "cuda_v100_two_stage_scheduler_smoke: invalid %s: %s\n",
                name,
                s ? s : "(null)");
        exit(2);
    }
    return (int)v;
}

static int map_model_file(const char *path, model_map *out) {
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "cuda_v100_two_stage_scheduler_smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "cuda_v100_two_stage_scheduler_smoke: cannot stat %s\n", path);
        close(fd);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "cuda_v100_two_stage_scheduler_smoke: cannot mmap %s: %s\n",
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

static void expect_finite_nonzero(const float *x, uint64_t n, const char *label) {
    double sum_abs = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(x[i])) {
            fprintf(stderr,
                    "cuda_v100_two_stage_scheduler_smoke: %s non-finite at %" PRIu64 "\n",
                    label,
                    i);
            failures++;
            return;
        }
        sum_abs += fabs((double)x[i]);
    }
    if (sum_abs == 0.0) {
        fprintf(stderr, "cuda_v100_two_stage_scheduler_smoke: %s all zero\n", label);
        failures++;
    }
}

int main(int argc, char **argv) {
    const char *index = NULL;
    const char *model_path = NULL;
    int token = 16;
    int position = 16;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--index") && i + 1 < argc) {
            index = argv[++i];
        } else if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "--token") && i + 1 < argc) {
            token = parse_int_arg(argv[++i], "--token", 200000);
        } else if (!strcmp(argv[i], "--position") && i + 1 < argc) {
            position = parse_int_arg(argv[++i], "--position", 2000000);
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
    if (devices < 2) {
        fprintf(stderr,
                "cuda_v100_two_stage_scheduler_smoke: need at least 2 CUDA devices, got %d\n",
                devices);
        return 1;
    }

    model_map model;
    if (map_model_file(model_path, &model)) return 1;
    check(ds4_gpu_set_model_fd(model.fd), "model fd");

    ds4_v100_stage_scheduler_options opts0;
    ds4_v100_stage_scheduler_options_init(&opts0);
    opts0.pack_index_path = index;
    opts0.model_map = model.ptr;
    opts0.model_size = model.size;
    opts0.stage_id = 0;

    ds4_v100_stage_scheduler_options opts1 = opts0;
    opts1.stage_id = 1;

    char err[512] = {0};
    ds4_v100_stage_scheduler *stage0 = NULL;
    ds4_v100_stage_scheduler *stage1 = NULL;
    if (ds4_v100_stage_scheduler_open(&stage0, &opts0, err, sizeof(err)) ||
        ds4_v100_stage_scheduler_open(&stage1, &opts1, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_two_stage_scheduler_smoke: %s\n", err);
        ds4_v100_stage_scheduler_close(stage1);
        ds4_v100_stage_scheduler_close(stage0);
        unmap_model_file(&model);
        return 1;
    }

    ds4_v100_stage_scheduler_report report0;
    ds4_v100_stage_scheduler_report report1;
    memset(&report0, 0, sizeof(report0));
    memset(&report1, 0, sizeof(report1));
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_decode_token(stage0,
                                                (uint32_t)token,
                                                (uint32_t)position,
                                                &report0,
                                                err,
                                                sizeof(err)) == 0,
          err[0] ? err : "stage 0 decode");
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_handoff(stage1, stage0, err, sizeof(err)) == 0,
          err[0] ? err : "stage 0 to stage 1 handoff");
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_decode_hc(stage1,
                                             (uint32_t)token,
                                             (uint32_t)position,
                                             &report1,
                                             err,
                                             sizeof(err)) == 0,
          err[0] ? err : "stage 1 decode");

    const uint64_t hc_values = (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS;
    float *hc = (float *)calloc((size_t)hc_values, sizeof(float));
    check(hc != NULL, "host HC allocation");
    if (hc) {
        check(ds4_v100_stage_scheduler_read_hc(stage1,
                                               hc,
                                               hc_values * sizeof(float)) != 0,
              "stage 1 HC read");
        expect_finite_nonzero(hc, hc_values, "two-stage scheduler HC");
    }
    check(report0.layers_executed == 6, "stage 0 executed six layers");
    check(report1.layers_executed == 6, "stage 1 executed six layers");

    printf("cuda_v100_two_stage_scheduler_smoke: stage0=%d-%d gpu0=%d stage1=%d-%d gpu1=%d token=%d pos=%d uploaded0=%" PRIu64 " uploaded1=%" PRIu64 " expert0=%d expert1=%d %s\n",
           report0.first_layer,
           report0.last_layer,
           report0.gpu,
           report1.first_layer,
           report1.last_layer,
           report1.gpu,
           token,
           position,
           report0.uploaded_bytes,
           report1.uploaded_bytes,
           report0.last_layer_report.selected_experts[0],
           report1.last_layer_report.selected_experts[0],
           failures ? "FAIL" : "ok");

    free(hc);
    ds4_v100_stage_scheduler_close(stage1);
    ds4_v100_stage_scheduler_close(stage0);
    unmap_model_file(&model);
    return failures ? 1 : 0;
}
