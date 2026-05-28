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

enum {
    PATH_BUF_SIZE = 4096,
};

typedef struct {
    const unsigned char *ptr;
    uint64_t size;
    int fd;
} model_map;

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_v100_stage_scheduler_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/cuda_v100_stage_scheduler_smoke --index FILE "
            "[--model FILE | --shard-dir DIR | --appliance-dir DIR] "
            "[--tm-index FILE] "
            "[--stage N] [--token N] [--position N] [--slots N] [--ctx N] "
            "[--expect-tm-layers N] [--expect-tp2-layers N]\n");
}

static int parse_int_arg(const char *s, const char *name, int max_v) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || !*s || !end || *end != '\0' || v < 0 || v > max_v) {
        fprintf(stderr, "cuda_v100_stage_scheduler_smoke: invalid %s: %s\n",
                name,
                s ? s : "(null)");
        exit(2);
    }
    return (int)v;
}

static void join_path(char *dst, size_t dst_size, const char *dir, const char *base) {
    int n = snprintf(dst, dst_size, "%s/%s", dir, base);
    if (n < 0 || (size_t)n >= dst_size) {
        fprintf(stderr, "cuda_v100_stage_scheduler_smoke: path too long: %s/%s\n",
                dir ? dir : "(null)",
                base ? base : "(null)");
        exit(2);
    }
}

static int map_model_file(const char *path, model_map *out) {
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "cuda_v100_stage_scheduler_smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "cuda_v100_stage_scheduler_smoke: cannot stat %s\n", path);
        close(fd);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "cuda_v100_stage_scheduler_smoke: cannot mmap %s: %s\n",
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
                    "cuda_v100_stage_scheduler_smoke: %s non-finite at %" PRIu64 "\n",
                    label,
                    i);
            failures++;
            return;
        }
        sum_abs += fabs((double)x[i]);
    }
    if (sum_abs == 0.0) {
        fprintf(stderr, "cuda_v100_stage_scheduler_smoke: %s all zero\n", label);
        failures++;
    }
}

int main(int argc, char **argv) {
    const char *index = NULL;
    const char *tm_index = NULL;
    const char *model_path = NULL;
    const char *shard_dir = NULL;
    char appliance_index[PATH_BUF_SIZE];
    char appliance_tm_index[PATH_BUF_SIZE];
    memset(appliance_index, 0, sizeof(appliance_index));
    memset(appliance_tm_index, 0, sizeof(appliance_tm_index));
    int stage = 0;
    int token = 16;
    int position = 16;
    int slots = 1;
    uint64_t ctx = 1048576ULL;
    int expect_tm_layers = -1;
    int expect_tp2_layers = -1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--index") && i + 1 < argc) {
            index = argv[++i];
        } else if (!strcmp(argv[i], "--tm-index") && i + 1 < argc) {
            tm_index = argv[++i];
        } else if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "--shard-dir") && i + 1 < argc) {
            shard_dir = argv[++i];
        } else if (!strcmp(argv[i], "--appliance-dir") && i + 1 < argc) {
            shard_dir = argv[++i];
            join_path(appliance_index, sizeof(appliance_index), shard_dir, "pack-index.tsv");
            join_path(appliance_tm_index, sizeof(appliance_tm_index), shard_dir, "turbomind-pack-index.tsv");
            index = appliance_index;
            tm_index = appliance_tm_index;
        } else if (!strcmp(argv[i], "--stage") && i + 1 < argc) {
            stage = parse_int_arg(argv[++i], "--stage", 7);
        } else if (!strcmp(argv[i], "--token") && i + 1 < argc) {
            token = parse_int_arg(argv[++i], "--token", 200000);
        } else if (!strcmp(argv[i], "--position") && i + 1 < argc) {
            position = parse_int_arg(argv[++i], "--position", 2000000);
        } else if (!strcmp(argv[i], "--slots") && i + 1 < argc) {
            slots = parse_int_arg(argv[++i], "--slots", DS4_V100_SCHED_MAX_SLOTS);
        } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            ctx = (uint64_t)parse_int_arg(argv[++i], "--ctx", 2000000);
        } else if (!strcmp(argv[i], "--expect-tm-layers") && i + 1 < argc) {
            expect_tm_layers = parse_int_arg(argv[++i], "--expect-tm-layers",
                                             DS4_V100_N_LAYERS);
        } else if (!strcmp(argv[i], "--expect-tp2-layers") && i + 1 < argc) {
            expect_tp2_layers = parse_int_arg(argv[++i], "--expect-tp2-layers",
                                              DS4_V100_N_LAYERS);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(stdout);
            return 0;
        } else {
            usage(stderr);
            return 2;
        }
    }
    if (!index || (!model_path && !shard_dir)) {
        usage(stderr);
        return 2;
    }

    int devices = ds4_gpu_device_count();
    if (devices < 1) {
        fprintf(stderr, "cuda_v100_stage_scheduler_smoke: no CUDA devices visible\n");
        return 1;
    }

    model_map model;
    memset(&model, 0, sizeof(model));
    model.fd = -1;
    if (model_path) {
        if (map_model_file(model_path, &model)) return 1;
        check(ds4_gpu_set_model_fd(model.fd), "model fd");
    }

    ds4_v100_stage_scheduler_options opts;
    ds4_v100_stage_scheduler_options_init(&opts);
    opts.pack_index_path = index;
    opts.turbomind_pack_index_path = tm_index;
    opts.shard_dir = shard_dir;
    opts.model_map = model.ptr;
    opts.model_size = model.size;
    opts.stage_id = stage;
    opts.kv_active_slots = (uint64_t)slots;
    opts.kv_ctx_tokens = ctx;

    char err[512] = {0};
    ds4_v100_stage_scheduler *sched = NULL;
    if (ds4_v100_stage_scheduler_open(&sched, &opts, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_stage_scheduler_smoke: %s\n", err);
        unmap_model_file(&model);
        return 1;
    }

    ds4_v100_stage_scheduler_report reports[DS4_V100_SCHED_MAX_SLOTS];
    memset(reports, 0, sizeof(reports));
    const uint32_t n_slots = (uint32_t)slots;
    uint32_t batch_tokens[DS4_V100_SCHED_MAX_SLOTS];
    uint32_t batch_positions[DS4_V100_SCHED_MAX_SLOTS];
    for (uint32_t i = 0; i < n_slots; i++) {
        batch_tokens[i] = (uint32_t)token;
        batch_positions[i] = (uint32_t)position;
    }
    err[0] = '\0';
    if (n_slots == 1) {
        check(ds4_v100_stage_scheduler_decode_token(sched,
                                                    (uint32_t)token,
                                                    (uint32_t)position,
                                                    &reports[0],
                                                    err,
                                                    sizeof(err)) == 0,
              err[0] ? err : "stage scheduler decode token");
    } else {
        check(ds4_v100_stage_scheduler_decode_token_batch(sched,
                                                          batch_tokens,
                                                          batch_positions,
                                                          n_slots,
                                                          reports,
                                                          err,
                                                          sizeof(err)) == 0,
              err[0] ? err : "stage scheduler decode token batch");
    }

    const uint64_t hc_values = (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS;
    float *hc = (float *)calloc((size_t)hc_values, sizeof(float));
    check(hc != NULL, "host HC allocation");
    if (hc) {
        check(ds4_v100_stage_scheduler_read_hc(sched,
                                               hc,
                                               hc_values * sizeof(float)) != 0,
              "stage scheduler HC read");
        expect_finite_nonzero(hc, hc_values, "stage scheduler HC");
    }
    for (uint32_t i = 0; i < n_slots; i++) {
        check(reports[i].layers_executed > 0, "stage executed at least one layer");
        check(reports[i].last_layer_report.routes == 6, "last layer reported six routes");
        if (expect_tm_layers >= 0) {
            check(reports[i].turbomind_routed_layers_executed == (uint32_t)expect_tm_layers,
                  "stage used expected TurboMind routed layers");
        } else if (shard_dir) {
            check(reports[i].turbomind_routed_layers_executed > 0,
                  "appliance stage used TurboMind routed layers");
        }
        if (expect_tp2_layers >= 0) {
            check(reports[i].turbomind_tp2_routed_layers_executed == (uint32_t)expect_tp2_layers,
                  "stage used expected TP2 routed layers");
        }
    }

    printf("cuda_v100_stage_scheduler_smoke: stage=%d gpu=%d layers=%d-%d executed=%u tm_layers=%u tp2_layers=%u token=%d pos=%d slots=%u arena_bytes=%" PRIu64 " uploaded_tensors=%" PRIu64 " uploaded_bytes=%" PRIu64 " ffn_ms=%.3f total_ms=%.3f expert0=%d %s\n",
           reports[0].stage_id,
           reports[0].gpu,
           reports[0].first_layer,
           reports[0].last_layer,
           reports[0].layers_executed,
           reports[0].turbomind_routed_layers_executed,
           reports[0].turbomind_tp2_routed_layers_executed,
           token,
           position,
           n_slots,
           reports[0].arena_bytes,
           reports[0].uploaded_tensors,
           reports[0].uploaded_bytes,
           reports[0].timing_ffn_ms,
           reports[0].timing_total_ms,
           reports[0].last_layer_report.selected_experts[0],
           failures ? "FAIL" : "ok");

    free(hc);
    ds4_v100_stage_scheduler_close(sched);
    unmap_model_file(&model);
    return failures ? 1 : 0;
}
