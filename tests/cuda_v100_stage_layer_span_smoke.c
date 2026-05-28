#include "engine/scheduler.h"

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
        fprintf(stderr, "cuda_v100_stage_layer_span_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/cuda_v100_stage_layer_span_smoke --index FILE --model FILE "
            "[--turbomind-index FILE] [--shard-dir DIR] "
            "[--token0 N] [--token1 N] [--position0 N] [--position1 N]\n");
}

static int parse_int_arg(const char *s, const char *name, int max_v) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || !*s || !end || *end != '\0' || v < 0 || v > max_v) {
        fprintf(stderr,
                "cuda_v100_stage_layer_span_smoke: invalid %s: %s\n",
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
                "cuda_v100_stage_layer_span_smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "cuda_v100_stage_layer_span_smoke: cannot stat %s\n", path);
        close(fd);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr,
                "cuda_v100_stage_layer_span_smoke: cannot mmap %s: %s\n",
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

static int stage_range(int stage_id, int *first, int *last) {
    int f = -1;
    int l = -1;
    for (int layer = 0; layer < DS4_V100_N_LAYERS; layer++) {
        if (ds4_v100_stage_for_layer(layer) != stage_id) continue;
        if (f < 0) f = layer;
        l = layer;
    }
    if (f < 0 || l < f) return 1;
    *first = f;
    *last = l;
    return 0;
}

static void check_report(const ds4_v100_stage_scheduler_report *r,
                         int first_layer,
                         int last_layer,
                         const char *label) {
    char msg[160];
    snprintf(msg, sizeof(msg), "%s first layer", label);
    check(r && r->first_layer == first_layer, msg);
    snprintf(msg, sizeof(msg), "%s last layer", label);
    check(r && r->last_layer == last_layer, msg);
    snprintf(msg, sizeof(msg), "%s layer count", label);
    check(r && r->layers_executed == (uint32_t)(last_layer - first_layer + 1), msg);
}

static int open_two_stages(ds4_v100_stage_scheduler **scheds,
                           const char *index,
                           const char *turbomind_index,
                           const char *shard_dir,
                           const model_map *model,
                           char *err,
                           size_t errlen) {
    ds4_v100_stage_scheduler_options opts;
    ds4_v100_stage_scheduler_options_init(&opts);
    opts.pack_index_path = index;
    opts.turbomind_pack_index_path = turbomind_index;
    opts.shard_dir = shard_dir;
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
    const char *turbomind_index = NULL;
    const char *shard_dir = NULL;
    const char *model_path = NULL;
    uint32_t token0 = 16;
    uint32_t token1 = 926;
    uint32_t position0 = 16;
    uint32_t position1 = 18;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--index") && i + 1 < argc) {
            index = argv[++i];
        } else if (!strcmp(argv[i], "--turbomind-index") && i + 1 < argc) {
            turbomind_index = argv[++i];
        } else if (!strcmp(argv[i], "--shard-dir") && i + 1 < argc) {
            shard_dir = argv[++i];
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
                "cuda_v100_stage_layer_span_smoke: need at least 2 CUDA devices, got %d\n",
                devices);
        return 1;
    }

    int s0_first = -1;
    int s0_last = -1;
    int s1_first = -1;
    int s1_last = -1;
    if (stage_range(0, &s0_first, &s0_last) || stage_range(1, &s1_first, &s1_last)) {
        fprintf(stderr, "cuda_v100_stage_layer_span_smoke: failed to resolve stage ranges\n");
        return 1;
    }
    const int s0_mid = s0_first + (s0_last - s0_first) / 2;
    const int s1_mid = s1_first + (s1_last - s1_first) / 2;

    model_map model;
    if (map_model_file(model_path, &model)) return 1;
    check(ds4_gpu_set_model_fd(model.fd), "model fd");

    ds4_v100_stage_scheduler *full[2] = {0};
    ds4_v100_stage_scheduler *seg[2] = {0};
    char err[512] = {0};

    const uint64_t hc_values = (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS;
    const uint64_t hc_bytes = hc_values * sizeof(float);
    float *full0 = (float *)malloc((size_t)hc_bytes);
    float *full1 = (float *)malloc((size_t)hc_bytes);
    float *seg0 = (float *)malloc((size_t)hc_bytes);
    float *seg1 = (float *)malloc((size_t)hc_bytes);
    if (!full0 || !full1 || !seg0 || !seg1) {
        fprintf(stderr, "cuda_v100_stage_layer_span_smoke: host allocation failed\n");
        failures++;
        goto cleanup;
    }

    const uint32_t tokens[2] = { token0, token1 };
    const uint32_t positions[2] = { position0, position1 };
    ds4_v100_stage_scheduler_report reports[2];

    if (open_two_stages(full, index, turbomind_index, shard_dir, &model, err, sizeof(err))) {
        fprintf(stderr,
                "cuda_v100_stage_layer_span_smoke: full open failed: %s\n",
                err[0] ? err : "open");
        failures++;
        goto cleanup;
    }

    memset(reports, 0, sizeof(reports));
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_decode_token_batch(full[0],
                                                      tokens,
                                                      positions,
                                                      2,
                                                      reports,
                                                      err,
                                                      sizeof(err)) == 0,
          err[0] ? err : "full stage0 decode");
    check_report(&reports[0], s0_first, s0_last, "full stage0 slot0");
    check(ds4_gpu_synchronize(), "full stage0 synchronize");
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_handoff_batch(full[1],
                                                 full[0],
                                                 2,
                                                 err,
                                                 sizeof(err)) == 0,
          err[0] ? err : "full handoff");
    memset(reports, 0, sizeof(reports));
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_decode_hc_batch(full[1],
                                                   tokens,
                                                   positions,
                                                   2,
                                                   reports,
                                                   err,
                                                   sizeof(err)) == 0,
          err[0] ? err : "full stage1 decode");
    check_report(&reports[0], s1_first, s1_last, "full stage1 slot0");
    check(ds4_gpu_synchronize(), "full stage1 synchronize");
    check(ds4_v100_stage_scheduler_read_hc_slot(full[1], 0, full0, hc_bytes),
          "full slot0 read");
    check(ds4_v100_stage_scheduler_read_hc_slot(full[1], 1, full1, hc_bytes),
          "full slot1 read");
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_reset(full[0], err, sizeof(err)) == 0,
          err[0] ? err : "reset stage0");
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_reset(full[1], err, sizeof(err)) == 0,
          err[0] ? err : "reset stage1");
    check(ds4_gpu_synchronize(), "post-reset synchronize");
    seg[0] = full[0];
    seg[1] = full[1];
    full[0] = NULL;
    full[1] = NULL;

    if (getenv("DS4_V100_STAGE_LAYER_SPAN_SECOND_FULL")) {
        memset(reports, 0, sizeof(reports));
        err[0] = '\0';
        check(ds4_v100_stage_scheduler_decode_token_batch(seg[0],
                                                          tokens,
                                                          positions,
                                                          2,
                                                          reports,
                                                          err,
                                                          sizeof(err)) == 0,
              err[0] ? err : "second full stage0 decode");
        check(ds4_gpu_synchronize(), "second full stage0 synchronize");
        err[0] = '\0';
        check(ds4_v100_stage_scheduler_handoff_batch(seg[1],
                                                     seg[0],
                                                     2,
                                                     err,
                                                     sizeof(err)) == 0,
              err[0] ? err : "second full handoff");
        memset(reports, 0, sizeof(reports));
        err[0] = '\0';
        check(ds4_v100_stage_scheduler_decode_hc_batch(seg[1],
                                                       tokens,
                                                       positions,
                                                       2,
                                                       reports,
                                                       err,
                                                       sizeof(err)) == 0,
              err[0] ? err : "second full stage1 decode");
        check(ds4_gpu_synchronize(), "second full stage1 synchronize");
        goto read_segmented;
    }

    memset(reports, 0, sizeof(reports));
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_decode_token_layer_span(seg[0],
                                                           0,
                                                           tokens,
                                                           positions,
                                                           2,
                                                           s0_first,
                                                           s0_mid,
                                                           reports,
                                                           err,
                                                           sizeof(err)) == 0,
          err[0] ? err : "segmented stage0 first span");
    check_report(&reports[0], s0_first, s0_mid, "segmented stage0 first span slot0");
    if (s0_mid < s0_last) {
        memset(reports, 0, sizeof(reports));
        err[0] = '\0';
        check(ds4_v100_stage_scheduler_decode_hc_layer_span(seg[0],
                                                            0,
                                                            tokens,
                                                            positions,
                                                            2,
                                                            s0_mid + 1,
                                                            s0_last,
                                                            reports,
                                                            err,
                                                            sizeof(err)) == 0,
              err[0] ? err : "segmented stage0 second span");
        check_report(&reports[0], s0_mid + 1, s0_last, "segmented stage0 second span slot0");
    }
    check(ds4_gpu_synchronize(), "segmented stage0 synchronize");
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_handoff_batch(seg[1],
                                                 seg[0],
                                                 2,
                                                 err,
                                                 sizeof(err)) == 0,
          err[0] ? err : "segmented handoff");

    memset(reports, 0, sizeof(reports));
    err[0] = '\0';
    check(ds4_v100_stage_scheduler_decode_hc_layer_span(seg[1],
                                                        0,
                                                        tokens,
                                                        positions,
                                                        2,
                                                        s1_first,
                                                        s1_mid,
                                                        reports,
                                                        err,
                                                        sizeof(err)) == 0,
          err[0] ? err : "segmented stage1 first span");
    check_report(&reports[0], s1_first, s1_mid, "segmented stage1 first span slot0");
    if (s1_mid < s1_last) {
        memset(reports, 0, sizeof(reports));
        err[0] = '\0';
        check(ds4_v100_stage_scheduler_decode_hc_layer_span(seg[1],
                                                            0,
                                                            tokens,
                                                            positions,
                                                            2,
                                                            s1_mid + 1,
                                                            s1_last,
                                                            reports,
                                                            err,
                                                            sizeof(err)) == 0,
              err[0] ? err : "segmented stage1 second span");
        check_report(&reports[0], s1_mid + 1, s1_last, "segmented stage1 second span slot0");
    }
    check(ds4_gpu_synchronize(), "segmented stage1 synchronize");

read_segmented:
    check(ds4_v100_stage_scheduler_read_hc_slot(seg[1], 0, seg0, hc_bytes),
          "segmented slot0 read");
    check(ds4_v100_stage_scheduler_read_hc_slot(seg[1], 1, seg1, hc_bytes),
          "segmented slot1 read");

    const double max0 = max_abs_diff(full0, seg0, hc_values);
    const double max1 = max_abs_diff(full1, seg1, hc_values);
    const double parity_threshold = 3.0e-2;
    printf("cuda_v100_stage_layer_span_smoke: "
           "stage0=[%d,%d] stage1=[%d,%d] token0=%" PRIu32
           " token1=%" PRIu32 " max_abs_slot0=%.9g max_abs_slot1=%.9g threshold=%.9g\n",
           s0_first,
           s0_last,
           s1_first,
           s1_last,
           token0,
           token1,
           max0,
           max1,
           parity_threshold);
    check(max0 <= parity_threshold, "slot0 layer-span parity");
    check(max1 <= parity_threshold, "slot1 layer-span parity");

    if (failures == 0) {
        printf("cuda_v100_stage_layer_span_smoke: ok\n");
    }

cleanup:
    free(seg1);
    free(seg0);
    free(full1);
    free(full0);
    close_scheds(seg, 2);
    close_scheds(full, 2);
    unmap_model_file(&model);
    return failures == 0 ? 0 : 1;
}
