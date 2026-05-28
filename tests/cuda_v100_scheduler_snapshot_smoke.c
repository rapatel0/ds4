#include "engine/scheduler.h"

#include "ds4_gpu.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
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
    TOP_K = 5,
};

typedef struct {
    const unsigned char *ptr;
    uint64_t size;
    int fd;
} model_map;

static int failures;

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/cuda_v100_scheduler_snapshot_smoke "
            "--index FILE --model FILE [--token N] [--steps N] [--ctx N]\n");
}

static int parse_int_arg(const char *s, const char *name, int min_v, int max_v) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || !*s || !end || *end != '\0' || v < min_v || v > max_v) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: invalid %s: %s\n", name, s ? s : "(null)");
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
                "cuda_v100_scheduler_snapshot_smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: cannot stat %s\n", path);
        close(fd);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr,
                "cuda_v100_scheduler_snapshot_smoke: cannot mmap %s: %s\n",
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

static int feed_token(ds4_v100_stage_scheduler **scheds,
                      uint32_t token,
                      uint32_t pos,
                      char *err,
                      size_t errlen) {
    ds4_v100_stage_scheduler_report report;
    memset(&report, 0, sizeof(report));
    if (ds4_v100_stage_scheduler_decode_token(scheds[0],
                                              token,
                                              pos,
                                              &report,
                                              err,
                                              errlen)) {
        return 1;
    }
    for (int stage = 1; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        if (ds4_v100_stage_scheduler_handoff(scheds[stage],
                                             scheds[stage - 1],
                                             err,
                                             errlen)) {
            return 1;
        }
        memset(&report, 0, sizeof(report));
        if (ds4_v100_stage_scheduler_decode_hc(scheds[stage],
                                               token,
                                               pos,
                                               &report,
                                               err,
                                               errlen)) {
            return 1;
        }
    }
    return ds4_gpu_synchronize() ? 0 : 1;
}

static int select_topk(ds4_v100_stage_scheduler **scheds,
                       uint32_t *tokens,
                       float *logits,
                       char *err,
                       size_t errlen) {
    if (ds4_v100_stage_scheduler_select_topk(scheds[DS4_V100_EXPECTED_GPUS - 1],
                                             tokens,
                                             logits,
                                             TOP_K,
                                             err,
                                             errlen)) {
        return 1;
    }
    return ds4_gpu_synchronize() ? 0 : 1;
}

static int compare_topk(const char *label,
                        const uint32_t *a_tokens,
                        const float *a_logits,
                        const uint32_t *b_tokens,
                        const float *b_logits,
                        double tol,
                        double *max_delta_out) {
    double max_delta = 0.0;
    for (uint32_t i = 0; i < TOP_K; i++) {
        const double delta = fabs((double)a_logits[i] - (double)b_logits[i]);
        if (delta > max_delta) max_delta = delta;
        if (a_tokens[i] != b_tokens[i] || delta > tol) {
            fprintf(stderr,
                    "cuda_v100_scheduler_snapshot_smoke: %s mismatch rank=%" PRIu32
                    " token_a=%" PRIu32 " token_b=%" PRIu32
                    " logit_a=%.9g logit_b=%.9g delta=%.9g tol=%.9g\n",
                    label,
                    i + 1,
                    a_tokens[i],
                    b_tokens[i],
                    a_logits[i],
                    b_logits[i],
                    delta,
                    tol);
            failures++;
        }
    }
    if (max_delta_out) *max_delta_out = max_delta;
    return failures ? 1 : 0;
}

static double max_abs_delta(const float *a, const float *b, uint64_t n) {
    double max_delta = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        const double delta = fabs((double)a[i] - (double)b[i]);
        if (delta > max_delta) max_delta = delta;
    }
    return max_delta;
}

int main(int argc, char **argv) {
    const char *index = NULL;
    const char *model_path = NULL;
    uint32_t token = 16;
    uint32_t steps = 8;
    uint64_t ctx = 4096;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--index")) {
            if (++i >= argc) {
                usage(stderr);
                return 2;
            }
            index = argv[i];
        } else if (!strcmp(arg, "--model")) {
            if (++i >= argc) {
                usage(stderr);
                return 2;
            }
            model_path = argv[i];
        } else if (!strcmp(arg, "--token")) {
            if (++i >= argc) {
                usage(stderr);
                return 2;
            }
            token = (uint32_t)parse_int_arg(argv[i], "--token", 0, INT32_MAX);
        } else if (!strcmp(arg, "--steps")) {
            if (++i >= argc) {
                usage(stderr);
                return 2;
            }
            steps = (uint32_t)parse_int_arg(argv[i], "--steps", 1, 512);
        } else if (!strcmp(arg, "--ctx")) {
            if (++i >= argc) {
                usage(stderr);
                return 2;
            }
            ctx = (uint64_t)parse_int_arg(argv[i], "--ctx", 1, INT32_MAX);
        } else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: unknown option: %s\n", arg);
            usage(stderr);
            return 2;
        }
    }
    if (!index || !model_path) {
        usage(stderr);
        return 2;
    }

    model_map model;
    memset(&model, 0, sizeof(model));
    ds4_v100_stage_scheduler *scheds[DS4_V100_EXPECTED_GPUS] = {0};
    ds4_v100_stage_scheduler_snapshot *snaps[DS4_V100_EXPECTED_GPUS] = {0};
    char err[512] = {0};
    uint32_t before_tokens[TOP_K], restored_tokens[TOP_K], mutated_tokens[TOP_K], replay_tokens[TOP_K];
    float before_logits[TOP_K], restored_logits[TOP_K], mutated_logits[TOP_K], replay_logits[TOP_K];
    double restore_delta = 0.0;
    double replay_delta = 0.0;
    double hc_mutate_delta = 0.0;
    double hc_restore_delta = 0.0;
    uint64_t snapshot_bytes = 0;
    const uint64_t hc_values = (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS;
    const uint64_t hc_bytes = hc_values * sizeof(float);
    float *before_hc = (float *)malloc((size_t)hc_bytes);
    float *mutated_hc = (float *)malloc((size_t)hc_bytes);
    float *restored_hc = (float *)malloc((size_t)hc_bytes);
    if (!before_hc || !mutated_hc || !restored_hc) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: host HC allocation failed\n");
        failures++;
        goto cleanup;
    }

    if (!ds4_gpu_init()) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: ds4_gpu_init failed\n");
        return 1;
    }
    if (ds4_gpu_device_count() < DS4_V100_EXPECTED_GPUS) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: need %d CUDA devices\n", DS4_V100_EXPECTED_GPUS);
        failures++;
        goto cleanup;
    }
    if (map_model_file(model_path, &model)) {
        failures++;
        goto cleanup;
    }
    if (!ds4_gpu_set_model_fd(model.fd)) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: failed to register model fd\n");
        failures++;
        goto cleanup;
    }

    ds4_v100_stage_scheduler_options opts;
    ds4_v100_stage_scheduler_options_init(&opts);
    opts.pack_index_path = index;
    opts.model_map = model.ptr;
    opts.model_size = model.size;
    opts.kv_ctx_tokens = ctx;
    opts.attn_comp_cap = 64;
    opts.index_comp_cap = 64;
    opts.indexer_top_k = 1;
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        opts.stage_id = stage;
        err[0] = '\0';
        if (ds4_v100_stage_scheduler_open(&scheds[stage], &opts, err, sizeof(err))) {
            fprintf(stderr,
                    "cuda_v100_scheduler_snapshot_smoke: open stage %d failed: %s\n",
                    stage,
                    err[0] ? err : "open failed");
            failures++;
            goto cleanup;
        }
    }

    err[0] = '\0';
    for (uint32_t pos = 0; pos < steps; pos++) {
        err[0] = '\0';
        if (feed_token(scheds, token + pos, pos, err, sizeof(err))) {
            fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: feed token failed at pos=%" PRIu32 ": %s\n",
                    pos,
                    err);
            failures++;
            goto cleanup;
        }
    }
    err[0] = '\0';
    if (select_topk(scheds, before_tokens, before_logits, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: select before failed: %s\n", err);
        failures++;
        goto cleanup;
    }
    if (!ds4_v100_stage_scheduler_read_hc(scheds[DS4_V100_EXPECTED_GPUS - 1], before_hc, hc_bytes)) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: read before HC failed\n");
        failures++;
        goto cleanup;
    }
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        err[0] = '\0';
        if (ds4_v100_stage_scheduler_snapshot_create(scheds[stage],
                                                     &snaps[stage],
                                                     err,
                                                     sizeof(err))) {
            fprintf(stderr,
                    "cuda_v100_scheduler_snapshot_smoke: snapshot stage %d failed: %s\n",
                    stage,
                    err[0] ? err : "snapshot failed");
            failures++;
            goto cleanup;
        }
        snapshot_bytes += ds4_v100_stage_scheduler_snapshot_bytes(snaps[stage]);
    }

    const uint32_t next_token = before_tokens[0];
    err[0] = '\0';
    if (feed_token(scheds, next_token, steps, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: mutate feed failed: %s\n", err);
        failures++;
        goto cleanup;
    }
    err[0] = '\0';
    if (select_topk(scheds, mutated_tokens, mutated_logits, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: select mutated failed: %s\n", err);
        failures++;
        goto cleanup;
    }
    if (!ds4_v100_stage_scheduler_read_hc(scheds[DS4_V100_EXPECTED_GPUS - 1], mutated_hc, hc_bytes)) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: read mutated HC failed\n");
        failures++;
        goto cleanup;
    }
    hc_mutate_delta = max_abs_delta(before_hc, mutated_hc, hc_values);
    if (hc_mutate_delta <= 1.0e-7) {
        fprintf(stderr,
                "cuda_v100_scheduler_snapshot_smoke: mutation did not change HC state max_delta=%.9g\n",
                hc_mutate_delta);
        failures++;
    }

    for (int stage = DS4_V100_EXPECTED_GPUS - 1; stage >= 0; stage--) {
        err[0] = '\0';
        if (ds4_v100_stage_scheduler_snapshot_restore(scheds[stage],
                                                      snaps[stage],
                                                      err,
                                                      sizeof(err))) {
            fprintf(stderr,
                    "cuda_v100_scheduler_snapshot_smoke: restore stage %d failed: %s\n",
                    stage,
                    err[0] ? err : "restore failed");
            failures++;
            goto cleanup;
        }
    }
    if (!ds4_gpu_synchronize()) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: restore synchronize failed\n");
        failures++;
        goto cleanup;
    }

    err[0] = '\0';
    if (select_topk(scheds, restored_tokens, restored_logits, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: select restored failed: %s\n", err);
        failures++;
        goto cleanup;
    }
    if (!ds4_v100_stage_scheduler_read_hc(scheds[DS4_V100_EXPECTED_GPUS - 1], restored_hc, hc_bytes)) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: read restored HC failed\n");
        failures++;
        goto cleanup;
    }
    hc_restore_delta = max_abs_delta(before_hc, restored_hc, hc_values);
    if (hc_restore_delta > 1.0e-5) {
        fprintf(stderr,
                "cuda_v100_scheduler_snapshot_smoke: restored HC mismatch max_delta=%.9g\n",
                hc_restore_delta);
        failures++;
    }
    compare_topk("restore", before_tokens, before_logits, restored_tokens, restored_logits, 1.0e-5, &restore_delta);

    err[0] = '\0';
    if (feed_token(scheds, next_token, steps, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: replay feed failed: %s\n", err);
        failures++;
        goto cleanup;
    }
    err[0] = '\0';
    if (select_topk(scheds, replay_tokens, replay_logits, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_scheduler_snapshot_smoke: select replay failed: %s\n", err);
        failures++;
        goto cleanup;
    }
    compare_topk("replay", mutated_tokens, mutated_logits, replay_tokens, replay_logits, 1.0e-5, &replay_delta);

cleanup:
    printf("cuda_v100_scheduler_snapshot_smoke: token=%" PRIu32
           " steps=%" PRIu32 " next=%" PRIu32 " snapshot_bytes=%" PRIu64
           " before_top1=%" PRIu32 " restored_top1=%" PRIu32
           " replay_top1=%" PRIu32 " hc_mutate_delta=%.9g hc_restore_delta=%.9g"
           " restore_delta=%.9g replay_delta=%.9g %s\n",
           token,
           steps,
           before_tokens[0],
           snapshot_bytes,
           before_tokens[0],
           restored_tokens[0],
           replay_tokens[0],
           hc_mutate_delta,
           hc_restore_delta,
           restore_delta,
           replay_delta,
           failures ? "FAIL" : "PASS");
    for (int stage = DS4_V100_EXPECTED_GPUS - 1; stage >= 0; stage--) {
        ds4_v100_stage_scheduler_snapshot_free(snaps[stage]);
        ds4_v100_stage_scheduler_close(scheds[stage]);
    }
    unmap_model_file(&model);
    ds4_gpu_cleanup();
    free(restored_hc);
    free(mutated_hc);
    free(before_hc);
    return failures ? 1 : 0;
}
