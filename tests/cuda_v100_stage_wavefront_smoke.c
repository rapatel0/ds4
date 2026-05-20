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
        fprintf(stderr, "cuda_v100_stage_wavefront_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/cuda_v100_stage_wavefront_smoke --index FILE --model FILE "
            "[--token0 N] [--token1 N] [--position0 N] [--position1 N]\n");
}

static int parse_int_arg(const char *s, const char *name, int max_v) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || !*s || !end || *end != '\0' || v < 0 || v > max_v) {
        fprintf(stderr,
                "cuda_v100_stage_wavefront_smoke: invalid %s: %s\n",
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
        fprintf(stderr,
                "cuda_v100_stage_wavefront_smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "cuda_v100_stage_wavefront_smoke: cannot stat %s\n", path);
        close(fd);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr,
                "cuda_v100_stage_wavefront_smoke: cannot mmap %s: %s\n",
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

static void close_scheds(ds4_v100_stage_scheduler **scheds, int n) {
    if (!scheds) return;
    for (int i = n - 1; i >= 0; i--) {
        ds4_v100_stage_scheduler_close(scheds[i]);
        scheds[i] = NULL;
    }
}

static double max_abs_diff(const float *a, const float *b, uint64_t n) {
    double out = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > out) out = d;
    }
    return out;
}

static int open_two_stages(ds4_v100_stage_scheduler **scheds,
                           const char *index,
                           const model_map *model,
                           char *err,
                           size_t errlen) {
    ds4_v100_stage_scheduler_options opts;
    ds4_v100_stage_scheduler_options_init(&opts);
    opts.pack_index_path = index;
    opts.model_map = model->ptr;
    opts.model_size = model->size;
    opts.kv_active_slots = 2;
    opts.attn_comp_cap = 64;
    opts.index_comp_cap = 64;
    for (int i = 0; i < 2; i++) {
        opts.stage_id = i;
        if (ds4_v100_stage_scheduler_open(&scheds[i], &opts, err, errlen)) {
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *index = NULL;
    const char *model_path = NULL;
    uint32_t token0 = 16;
    uint32_t token1 = 926;
    uint32_t position0 = 16;
    uint32_t position1 = 18;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--index") && i + 1 < argc) {
            index = argv[++i];
        } else if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "--token0") && i + 1 < argc) {
            token0 = (uint32_t)parse_int_arg(argv[++i], "--token0", 200000);
        } else if (!strcmp(argv[i], "--token1") && i + 1 < argc) {
            token1 = (uint32_t)parse_int_arg(argv[++i], "--token1", 200000);
        } else if (!strcmp(argv[i], "--position0") && i + 1 < argc) {
            position0 = (uint32_t)parse_int_arg(argv[++i], "--position0", 2000000);
        } else if (!strcmp(argv[i], "--position1") && i + 1 < argc) {
            position1 = (uint32_t)parse_int_arg(argv[++i], "--position1", 2000000);
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
                "cuda_v100_stage_wavefront_smoke: need at least 2 CUDA devices, got %d\n",
                devices);
        return 1;
    }

    setenv("DS4_V100_BATCH_LAYER_FFN", "0", 1);

    model_map model;
    if (map_model_file(model_path, &model)) return 1;
    check(ds4_gpu_set_model_fd(model.fd), "model fd");

    ds4_v100_stage_scheduler *serial[2] = {0};
    ds4_v100_stage_scheduler *wave[2] = {0};
    char err[512] = {0};

    const uint64_t hc_values = (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS;
    const uint64_t hc_bytes = hc_values * sizeof(float);
    float *serial0 = (float *)malloc((size_t)hc_bytes);
    float *serial1 = (float *)malloc((size_t)hc_bytes);
    float *wave0 = (float *)malloc((size_t)hc_bytes);
    float *wave1 = (float *)malloc((size_t)hc_bytes);
    if (!serial0 || !serial1 || !wave0 || !wave1) {
        fprintf(stderr, "cuda_v100_stage_wavefront_smoke: host allocation failed\n");
        failures++;
        goto cleanup;
    }

    if (open_two_stages(serial, index, &model, err, sizeof(err))) {
        fprintf(stderr,
                "cuda_v100_stage_wavefront_smoke: serial open failed: %s\n",
                err[0] ? err : "open");
        failures++;
        goto cleanup;
    }

    const uint32_t tokens[2] = { token0, token1 };
    const uint32_t positions[2] = { position0, position1 };
    ds4_v100_stage_scheduler_report reports[2];
    memset(reports, 0, sizeof(reports));
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_decode_token_batch(serial[0],
                                                      tokens,
                                                      positions,
                                                      2,
                                                      reports,
                                                      err,
                                                      sizeof(err)) == 0,
          err[0] ? err : "serial stage0 decode");
    check(ds4_gpu_synchronize(), "serial stage0 synchronize");
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_handoff_batch(serial[1],
                                                 serial[0],
                                                 2,
                                                 err,
                                                 sizeof(err)) == 0,
          err[0] ? err : "serial handoff");
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_decode_hc_batch(serial[1],
                                                   tokens,
                                                   positions,
                                                   2,
                                                   reports,
                                                   err,
                                                   sizeof(err)) == 0,
          err[0] ? err : "serial stage1 decode");
    check(ds4_gpu_synchronize(), "serial stage1 synchronize");
    check(ds4_v100_stage_scheduler_read_hc_slot(serial[1], 0, serial0, hc_bytes),
          "serial slot0 read");
    check(ds4_v100_stage_scheduler_read_hc_slot(serial[1], 1, serial1, hc_bytes),
          "serial slot1 read");
    close_scheds(serial, 2);

    if (open_two_stages(wave, index, &model, err, sizeof(err))) {
        fprintf(stderr,
                "cuda_v100_stage_wavefront_smoke: wavefront open failed: %s\n",
                err[0] ? err : "open");
        failures++;
        goto cleanup;
    }

    const uint32_t token0_arr[1] = { token0 };
    const uint32_t token1_arr[1] = { token1 };
    const uint32_t pos0_arr[1] = { position0 };
    const uint32_t pos1_arr[1] = { position1 };

    err[0] = '\0';
    check(ds4_v100_stage_scheduler_decode_token_slot_span(wave[0],
                                                          0,
                                                          token0_arr,
                                                          pos0_arr,
                                                          1,
                                                          reports,
                                                          err,
                                                          sizeof(err)) == 0,
          err[0] ? err : "wave slot0 stage0 decode");
    check(ds4_gpu_synchronize(), "wave slot0 stage0 synchronize");
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_handoff_slot_span(wave[1],
                                                     wave[0],
                                                     0,
                                                     1,
                                                     err,
                                                     sizeof(err)) == 0,
          err[0] ? err : "wave slot0 handoff");

    err[0] = '\0';
    check(ds4_v100_stage_scheduler_decode_hc_slot_span(wave[1],
                                                       0,
                                                       token0_arr,
                                                       pos0_arr,
                                                       1,
                                                       reports,
                                                       err,
                                                       sizeof(err)) == 0,
          err[0] ? err : "wave slot0 stage1 decode");
    check(ds4_gpu_synchronize(), "wave slot0 stage1 synchronize");

    err[0] = '\0';
    int rc = ds4_v100_stage_scheduler_decode_token_slot_span(wave[0],
                                                             1,
                                                             token1_arr,
                                                             pos1_arr,
                                                             1,
                                                             reports,
                                                             err,
                                                             sizeof(err));
    check(rc == 0, err[0] ? err : "wave slot1 stage0 decode");
    check(ds4_gpu_synchronize(), "wave slot1 stage0 synchronize");

    err[0] = '\0';
    check(ds4_v100_stage_scheduler_handoff_slot_span(wave[1],
                                                     wave[0],
                                                     1,
                                                     1,
                                                     err,
                                                     sizeof(err)) == 0,
          err[0] ? err : "wave slot1 handoff");
    err[0] = '\0';
    rc = ds4_v100_stage_scheduler_decode_hc_slot_span(wave[1],
                                                      1,
                                                      token1_arr,
                                                      pos1_arr,
                                                      1,
                                                      reports,
                                                      err,
                                                      sizeof(err));
    check(rc == 0, err[0] ? err : "wave slot1 stage1 decode");
    check(ds4_gpu_synchronize(), "wave slot1 stage1 synchronize");
    check(ds4_v100_stage_scheduler_read_hc_slot(wave[1], 0, wave0, hc_bytes),
          "wave slot0 read");
    check(ds4_v100_stage_scheduler_read_hc_slot(wave[1], 1, wave1, hc_bytes),
          "wave slot1 read");

    const double max0 = max_abs_diff(serial0, wave0, hc_values);
    const double max1 = max_abs_diff(serial1, wave1, hc_values);
    check(max0 <= 1.0e-5, "slot0 wavefront parity");
    check(max1 <= 1.0e-5, "slot1 wavefront parity");

    if (failures == 0) {
        printf("cuda_v100_stage_wavefront_smoke: token0=%" PRIu32
               " token1=%" PRIu32 " max_abs_slot0=%.9g max_abs_slot1=%.9g ok\n",
               token0,
               token1,
               max0,
               max1);
    }

cleanup:
    free(wave1);
    free(wave0);
    free(serial1);
    free(serial0);
    close_scheds(wave, 2);
    close_scheds(serial, 2);
    unmap_model_file(&model);
    return failures == 0 ? 0 : 1;
}
