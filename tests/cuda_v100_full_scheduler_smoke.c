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
        fprintf(stderr, "cuda_v100_full_scheduler_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/cuda_v100_full_scheduler_smoke --index FILE "
            "[--model FILE | --shard-dir DIR | --appliance-dir DIR] [--tm-index FILE] "
            "[--token N] [--position N] [--stages N] [--slots N] [--ctx N] "
            "[--expect-tm-layers N]\n");
}

static int parse_int_arg(const char *s, const char *name, int max_v) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || !*s || !end || *end != '\0' || v < 0 || v > max_v) {
        fprintf(stderr, "cuda_v100_full_scheduler_smoke: invalid %s: %s\n",
                name,
                s ? s : "(null)");
        exit(2);
    }
    return (int)v;
}

static void join_path(char *dst, size_t dst_size, const char *dir, const char *base) {
    int n = snprintf(dst, dst_size, "%s/%s", dir, base);
    if (n < 0 || (size_t)n >= dst_size) {
        fprintf(stderr, "cuda_v100_full_scheduler_smoke: path too long: %s/%s\n",
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
        fprintf(stderr, "cuda_v100_full_scheduler_smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "cuda_v100_full_scheduler_smoke: cannot stat %s\n", path);
        close(fd);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "cuda_v100_full_scheduler_smoke: cannot mmap %s: %s\n",
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
                    "cuda_v100_full_scheduler_smoke: %s non-finite at %" PRIu64 "\n",
                    label,
                    i);
            failures++;
            return;
        }
        sum_abs += fabs((double)x[i]);
    }
    if (sum_abs == 0.0) {
        fprintf(stderr, "cuda_v100_full_scheduler_smoke: %s all zero\n", label);
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
    int token = 16;
    int position = 16;
    int stages = DS4_V100_EXPECTED_GPUS;
    int slots = 1;
    uint64_t ctx = 1048576ULL;
    int expect_tm_layers = -1;

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
        } else if (!strcmp(argv[i], "--token") && i + 1 < argc) {
            token = parse_int_arg(argv[++i], "--token", 200000);
        } else if (!strcmp(argv[i], "--position") && i + 1 < argc) {
            position = parse_int_arg(argv[++i], "--position", 2000000);
        } else if (!strcmp(argv[i], "--stages") && i + 1 < argc) {
            stages = parse_int_arg(argv[++i], "--stages", DS4_V100_EXPECTED_GPUS);
        } else if (!strcmp(argv[i], "--slots") && i + 1 < argc) {
            slots = parse_int_arg(argv[++i], "--slots", DS4_V100_SCHED_MAX_SLOTS);
        } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            ctx = (uint64_t)parse_int_arg(argv[++i], "--ctx", 2000000);
        } else if (!strcmp(argv[i], "--expect-tm-layers") && i + 1 < argc) {
            expect_tm_layers = parse_int_arg(argv[++i], "--expect-tm-layers",
                                             DS4_V100_N_LAYERS);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(stdout);
            return 0;
        } else {
            usage(stderr);
            return 2;
        }
    }
    if (!index || (!model_path && !shard_dir) || stages <= 0) {
        usage(stderr);
        return 2;
    }

    int devices = ds4_gpu_device_count();
    if (devices < stages) {
        fprintf(stderr,
                "cuda_v100_full_scheduler_smoke: need at least %d CUDA devices, got %d\n",
                stages,
                devices);
        return 1;
    }

    model_map model;
    memset(&model, 0, sizeof(model));
    model.fd = -1;
    if (model_path) {
        if (map_model_file(model_path, &model)) return 1;
        check(ds4_gpu_set_model_fd(model.fd), "model fd");
    }

    ds4_v100_stage_scheduler *scheds[DS4_V100_EXPECTED_GPUS];
    ds4_v100_stage_scheduler_report reports[DS4_V100_EXPECTED_GPUS];
    memset(scheds, 0, sizeof(scheds));
    memset(reports, 0, sizeof(reports));
    uint32_t layers_executed = 0;
    uint32_t tm_layers_executed = 0;
    uint64_t uploaded_bytes = 0;
    uint64_t uploaded_tensors = 0;
    const uint32_t n_slots = (uint32_t)slots;

    ds4_v100_stage_scheduler_options opts;
    ds4_v100_stage_scheduler_options_init(&opts);
    opts.pack_index_path = index;
    opts.turbomind_pack_index_path = tm_index;
    opts.shard_dir = shard_dir;
    opts.model_map = model.ptr;
    opts.model_size = model.size;
    opts.kv_active_slots = (uint64_t)slots;
    opts.kv_ctx_tokens = ctx;

    char err[512] = {0};
    for (int i = 0; i < stages; i++) {
        opts.stage_id = i;
        if (ds4_v100_stage_scheduler_open(&scheds[i], &opts, err, sizeof(err))) {
            fprintf(stderr,
                    "cuda_v100_full_scheduler_smoke: open stage %d failed: %s\n",
                    i,
                    err[0] ? err : "scheduler open");
            failures++;
            goto cleanup;
        }
    }

    ds4_v100_stage_scheduler_report batch_reports[DS4_V100_SCHED_MAX_SLOTS];
    uint32_t batch_tokens[DS4_V100_SCHED_MAX_SLOTS];
    uint32_t batch_positions[DS4_V100_SCHED_MAX_SLOTS];
    for (uint32_t i = 0; i < n_slots; i++) {
        batch_tokens[i] = (uint32_t)token;
        batch_positions[i] = (uint32_t)position;
    }

    err[0] = '\0';
    memset(batch_reports, 0, sizeof(batch_reports));
    if (n_slots == 1) {
        check(ds4_v100_stage_scheduler_decode_token(scheds[0],
                                                    (uint32_t)token,
                                                    (uint32_t)position,
                                                    &reports[0],
                                                    err,
                                                    sizeof(err)) == 0,
              err[0] ? err : "stage 0 decode");
    } else {
        check(ds4_v100_stage_scheduler_decode_token_batch(scheds[0],
                                                          batch_tokens,
                                                          batch_positions,
                                                          n_slots,
                                                          batch_reports,
                                                          err,
                                                          sizeof(err)) == 0,
              err[0] ? err : "stage 0 decode batch");
        reports[0] = batch_reports[0];
    }

    for (int i = 1; i < stages && failures == 0; i++) {
        err[0] = '\0';
        if (n_slots == 1) {
            check(ds4_v100_stage_scheduler_handoff(scheds[i], scheds[i - 1], err, sizeof(err)) == 0,
                  err[0] ? err : "stage handoff");
        } else {
            check(ds4_v100_stage_scheduler_handoff_batch(scheds[i],
                                                         scheds[i - 1],
                                                         n_slots,
                                                         err,
                                                         sizeof(err)) == 0,
                  err[0] ? err : "stage handoff batch");
        }
        err[0] = '\0';
        if (n_slots == 1) {
            check(ds4_v100_stage_scheduler_decode_hc(scheds[i],
                                                     (uint32_t)token,
                                                     (uint32_t)position,
                                                     &reports[i],
                                                     err,
                                                     sizeof(err)) == 0,
                  err[0] ? err : "stage decode");
        } else {
            memset(batch_reports, 0, sizeof(batch_reports));
            check(ds4_v100_stage_scheduler_decode_hc_batch(scheds[i],
                                                           batch_tokens,
                                                           batch_positions,
                                                           n_slots,
                                                           batch_reports,
                                                           err,
                                                           sizeof(err)) == 0,
                  err[0] ? err : "stage decode batch");
            reports[i] = batch_reports[0];
        }
    }

    for (int i = 0; i < stages; i++) {
        const int expected_layers = reports[i].last_layer - reports[i].first_layer + 1;
        char msg[128];
        snprintf(msg, sizeof(msg), "stage %d executed assigned layers", i);
        check(expected_layers > 0 &&
                  reports[i].layers_executed == (uint32_t)expected_layers,
              msg);
        layers_executed += reports[i].layers_executed;
        tm_layers_executed += reports[i].turbomind_routed_layers_executed;
        uploaded_bytes += reports[i].uploaded_bytes;
        uploaded_tensors += reports[i].uploaded_tensors;
    }
    if (stages == DS4_V100_EXPECTED_GPUS) {
        check(layers_executed == DS4_V100_N_LAYERS, "full scheduler executed 43 layers");
        check(reports[stages - 1].last_layer == DS4_V100_N_LAYERS - 1,
              "full scheduler reached final layer");
    }
    if (expect_tm_layers >= 0) {
        check(tm_layers_executed == (uint32_t)expect_tm_layers,
              "full appliance smoke used expected TurboMind routed layers");
    } else if (shard_dir) {
        check(tm_layers_executed > 0, "full appliance smoke used TurboMind routed layers");
    }

    const uint64_t hc_values = (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS;
    float *hc = (float *)calloc((size_t)hc_values, sizeof(float));
    check(hc != NULL, "host HC allocation");
    if (hc && stages > 0) {
        check(ds4_v100_stage_scheduler_read_hc(scheds[stages - 1],
                                               hc,
                                               hc_values * sizeof(float)) != 0,
              "final HC read");
        expect_finite_nonzero(hc, hc_values, "full scheduler HC");
    }
    free(hc);

cleanup:
    printf("cuda_v100_full_scheduler_smoke: stages=%d token=%d pos=%d slots=%u layers=%" PRIu32
           " tm_layers=%" PRIu32 " last=%d-%d gpu=%d uploaded_tensors=%" PRIu64 " uploaded_bytes=%" PRIu64
           " expert_last=%d %s\n",
           stages,
           token,
           position,
           n_slots,
           layers_executed,
           tm_layers_executed,
           reports[stages - 1].first_layer,
           reports[stages - 1].last_layer,
           reports[stages - 1].gpu,
           uploaded_tensors,
           uploaded_bytes,
           reports[stages - 1].last_layer_report.selected_experts[0],
           failures ? "FAIL" : "ok");

    for (int i = stages - 1; i >= 0; i--) ds4_v100_stage_scheduler_close(scheds[i]);
    unmap_model_file(&model);
    return failures ? 1 : 0;
}
