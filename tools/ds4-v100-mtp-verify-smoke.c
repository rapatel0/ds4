#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4-v100-mtp-forward-common.h"
#include "ds4_v100_mtp.h"
#include "ds4_v100_scheduler.h"

#include <errno.h>
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
#include <fcntl.h>

enum {
    MTP_VERIFY_MAX_TOPK = 16,
    MTP_VERIFY_RAW_CAP = 128,
    MTP_VERIFY_HEAD_DIM = 512,
};

typedef struct {
    const unsigned char *ptr;
    uint64_t size;
    int fd;
} model_map;

typedef struct {
    const char *model;
    const char *mtp_model;
    const char *pack_index;
    const char *prompt_file;
    const char *report_path;
    int gpu;
    int require_gpus;
    int reserve_mib;
    uint32_t top_k;
    uint64_t ctx;
} verify_options;

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-v100-mtp-verify-smoke --model FILE --mtp-model FILE --pack-index FILE [options]\n"
            "\n"
            "Options:\n"
            "  --index FILE            Alias for --pack-index\n"
            "  --prompt-file FILE      Prompt to tokenize. Default: short_reasoning_plain.txt\n"
            "  --gpu N                 MTP sidecar/draft-state GPU. Default: 7\n"
            "  --require-gpus N        Require at least N visible CUDA devices\n"
            "  --reserve-mib N         Require this much free memory after sidecar upload. Default: 4096\n"
            "  --top-k N               Number of target candidates to compare. Default: 5\n"
            "  --ctx N                 Scheduler KV context tokens. Default: 1048576\n"
            "  --report FILE           Write detailed report to FILE instead of stdout\n");
}

static const char *need_arg(int *i, int argc, char **argv, const char *arg) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: %s requires an argument\n", arg);
        exit(2);
    }
    return argv[++*i];
}

static int parse_int(const char *s, const char *arg) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || !s[0] || !end || *end || v < 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: bad integer for %s: %s\n", arg, s ? s : "(null)");
        exit(2);
    }
    return (int)v;
}

static uint64_t parse_u64(const char *s, const char *arg) {
    errno = 0;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || !s || !s[0] || !end || *end) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: bad integer for %s: %s\n", arg, s ? s : "(null)");
        exit(2);
    }
    return (uint64_t)v;
}

static verify_options parse_options(int argc, char **argv) {
    verify_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.prompt_file = "tests/test-vectors/prompts/short_reasoning_plain.txt";
    opt.gpu = 7;
    opt.reserve_mib = 4096;
    opt.top_k = 5;
    opt.ctx = 1048576;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "--model")) {
            opt.model = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-model")) {
            opt.mtp_model = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--pack-index") || !strcmp(arg, "--index")) {
            opt.pack_index = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            opt.prompt_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--report")) {
            opt.report_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--gpu")) {
            opt.gpu = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--require-gpus")) {
            opt.require_gpus = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--reserve-mib")) {
            opt.reserve_mib = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--top-k")) {
            opt.top_k = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx")) {
            opt.ctx = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else {
            fprintf(stderr, "ds4-v100-mtp-verify-smoke: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }
    if (!opt.model || !opt.model[0] ||
        !opt.mtp_model || !opt.mtp_model[0] ||
        !opt.pack_index || !opt.pack_index[0] ||
        !opt.prompt_file || !opt.prompt_file[0] ||
        opt.top_k < 2 || opt.top_k > MTP_VERIFY_MAX_TOPK ||
        opt.ctx == 0) {
        usage(stderr);
        exit(2);
    }
    return opt;
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *buf = (char *)malloc((size_t)len + 1u);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)len, fp);
    fclose(fp);
    if (got != (size_t)len) {
        free(buf);
        return NULL;
    }
    buf[len] = '\0';
    return buf;
}

static int map_model_file(const char *path, model_map *out) {
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: cannot stat %s\n", path);
        close(fd);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: cannot mmap %s: %s\n",
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

static int feed_prompt(ds4_v100_stage_scheduler **scheds,
                       const ds4_tokens *prompt,
                       char *err,
                       size_t errlen) {
    for (int pos = 0; pos < prompt->len; pos++) {
        if (prompt->v[pos] < 0) {
            snprintf(err, errlen, "negative prompt token at position %d", pos);
            return 1;
        }
        if (feed_token(scheds,
                       (uint32_t)prompt->v[pos],
                       (uint32_t)pos,
                       err,
                       errlen)) {
            return 1;
        }
    }
    return 0;
}

static int select_topk(ds4_v100_stage_scheduler **scheds,
                       uint32_t *tokens,
                       float *logits,
                       uint32_t k,
                       char *err,
                       size_t errlen) {
    if (ds4_v100_stage_scheduler_select_topk(scheds[DS4_V100_EXPECTED_GPUS - 1],
                                             tokens,
                                             logits,
                                             k,
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
                        uint32_t k,
                        double tol,
                        FILE *report,
                        double *max_delta_out) {
    double max_delta = 0.0;
    int failures = 0;
    for (uint32_t i = 0; i < k; i++) {
        const double delta = fabs((double)a_logits[i] - (double)b_logits[i]);
        if (delta > max_delta) max_delta = delta;
        fprintf(report,
                "mtp_rollback_compare\t%s\trank=%" PRIu32
                "\ta_token=%" PRIu32 "\tb_token=%" PRIu32
                "\ta_logit=%.9g\tb_logit=%.9g\tdelta=%.9g\ttol=%.9g\n",
                label,
                i + 1,
                a_tokens[i],
                b_tokens[i],
                a_logits[i],
                b_logits[i],
                delta,
                tol);
        if (a_tokens[i] != b_tokens[i] || delta > tol) failures++;
    }
    if (max_delta_out) *max_delta_out = max_delta;
    return failures ? 1 : 0;
}

static int open_schedulers(ds4_v100_stage_scheduler **scheds,
                           const verify_options *opt,
                           const model_map *model,
                           char *err,
                           size_t errlen) {
    ds4_v100_stage_scheduler_options sopts;
    ds4_v100_stage_scheduler_options_init(&sopts);
    sopts.pack_index_path = opt->pack_index;
    sopts.model_map = model->ptr;
    sopts.model_size = model->size;
    sopts.kv_ctx_tokens = opt->ctx;
    sopts.attn_comp_cap = 64;
    sopts.index_comp_cap = 64;
    sopts.indexer_top_k = 512;
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        sopts.stage_id = stage;
        err[0] = '\0';
        if (ds4_v100_stage_scheduler_open(&scheds[stage], &sopts, err, errlen)) {
            return 1;
        }
    }
    return 0;
}

static void close_schedulers(ds4_v100_stage_scheduler **scheds) {
    for (int stage = DS4_V100_EXPECTED_GPUS - 1; stage >= 0; stage--) {
        ds4_v100_stage_scheduler_close(scheds[stage]);
        scheds[stage] = NULL;
    }
}

static void free_snapshots(ds4_v100_stage_scheduler_snapshot **snaps) {
    for (int stage = DS4_V100_EXPECTED_GPUS - 1; stage >= 0; stage--) {
        ds4_v100_stage_scheduler_snapshot_free(snaps[stage]);
        snaps[stage] = NULL;
    }
}

static int create_snapshots(ds4_v100_stage_scheduler **scheds,
                            ds4_v100_stage_scheduler_snapshot **snaps,
                            uint64_t *bytes_out,
                            char *err,
                            size_t errlen) {
    uint64_t bytes = 0;
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        err[0] = '\0';
        if (ds4_v100_stage_scheduler_snapshot_create(scheds[stage],
                                                     &snaps[stage],
                                                     err,
                                                     errlen)) {
            return 1;
        }
        bytes += ds4_v100_stage_scheduler_snapshot_bytes(snaps[stage]);
    }
    if (bytes_out) *bytes_out = bytes;
    return 0;
}

static int restore_snapshots(ds4_v100_stage_scheduler **scheds,
                             ds4_v100_stage_scheduler_snapshot **snaps,
                             char *err,
                             size_t errlen) {
    for (int stage = DS4_V100_EXPECTED_GPUS - 1; stage >= 0; stage--) {
        err[0] = '\0';
        if (ds4_v100_stage_scheduler_snapshot_restore(scheds[stage],
                                                      snaps[stage],
                                                      err,
                                                      errlen)) {
            return 1;
        }
    }
    return ds4_gpu_synchronize() ? 0 : 1;
}

static double max_abs_f32(const float *x, uint32_t n) {
    double m = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        double a = fabs((double)x[i]);
        if (a > m) m = a;
    }
    return m;
}

int main(int argc, char **argv) {
    verify_options opt = parse_options(argc, argv);
    FILE *report = stdout;
    if (opt.report_path) {
        report = fopen(opt.report_path, "w");
        if (!report) {
            fprintf(stderr,
                    "ds4-v100-mtp-verify-smoke: cannot open report %s: %s\n",
                    opt.report_path,
                    strerror(errno));
            return 1;
        }
    }

    int rc = 1;
    char err[512] = {0};
    model_map model;
    memset(&model, 0, sizeof(model));
    model.fd = -1;
    ds4_engine *tok_engine = NULL;
    ds4_tokens prompt = {0};
    ds4_v100_context *ctx = NULL;
    ds4_v100_mtp_sidecar *sidecar = NULL;
    ds4_v100_mtp_forward *mtp_forward = NULL;
    ds4_gpu_tensor *mtp_raw = NULL;
    ds4_gpu_tensor *mtp_raw_snapshot = NULL;
    ds4_v100_stage_scheduler *scheds[DS4_V100_EXPECTED_GPUS] = {0};
    ds4_v100_stage_scheduler_snapshot *snaps[DS4_V100_EXPECTED_GPUS] = {0};
    float *committed_embed = NULL;
    float *post_commit_hc = NULL;
    uint32_t after_prompt_tokens[MTP_VERIFY_MAX_TOPK];
    uint32_t post_t_tokens[MTP_VERIFY_MAX_TOPK];
    uint32_t mtp_tokens[MTP_VERIFY_MAX_TOPK];
    uint32_t restored_tokens[MTP_VERIFY_MAX_TOPK];
    uint32_t continued_tokens[MTP_VERIFY_MAX_TOPK];
    uint32_t replay_tokens[MTP_VERIFY_MAX_TOPK];
    float after_prompt_logits[MTP_VERIFY_MAX_TOPK];
    float post_t_logits[MTP_VERIFY_MAX_TOPK];
    float mtp_logits[MTP_VERIFY_MAX_TOPK];
    float restored_logits[MTP_VERIFY_MAX_TOPK];
    float continued_logits[MTP_VERIFY_MAX_TOPK];
    float replay_logits[MTP_VERIFY_MAX_TOPK];
    ds4_v100_mtp_forward_report mtp_fwd_report;
    memset(&mtp_fwd_report, 0, sizeof(mtp_fwd_report));
    double restore_delta = 0.0;
    double replay_delta = 0.0;
    double mtp_raw_restore_max_abs = DBL_MAX;
    uint64_t snapshot_bytes = 0;
    uint32_t mtp_n_raw = 0;
    uint32_t mtp_n_raw_snapshot = 0;

    fprintf(report, "model\t%s\n", opt.model);
    fprintf(report, "mtp_model\t%s\n", opt.mtp_model);
    fprintf(report, "pack_index\t%s\n", opt.pack_index);
    fprintf(report, "prompt_file\t%s\n", opt.prompt_file);
    fprintf(report, "gpu\t%d\n", opt.gpu);
    fprintf(report, "require_gpus\t%d\n", opt.require_gpus);
    fprintf(report, "reserve_mib\t%d\n", opt.reserve_mib);
    fprintf(report, "top_k\t%" PRIu32 "\n", opt.top_k);
    fprintf(report, "ctx\t%" PRIu64 "\n", opt.ctx);

    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: ds4_gpu_init failed\n");
        goto done;
    }
    int n_dev = ds4_gpu_device_count();
    fprintf(report, "visible_cuda_devices\t%d\n", n_dev);
    if (opt.require_gpus > 0 && n_dev < opt.require_gpus) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: need %d CUDA devices, got %d\n",
                opt.require_gpus,
                n_dev);
        goto done;
    }
    if (map_model_file(opt.model, &model)) goto done;
    if (!ds4_gpu_set_model_fd(model.fd)) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: failed to register model fd\n");
        goto done;
    }

    ds4_engine_options eopts;
    memset(&eopts, 0, sizeof(eopts));
    eopts.model_path = opt.model;
    eopts.backend = DS4_BACKEND_CPU;
    eopts.inspect_only = true;
    eopts.n_threads = 1;
    if (ds4_engine_open(&tok_engine, &eopts) != 0) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: tokenizer engine open failed\n");
        goto done;
    }
    char *prompt_text = read_file(opt.prompt_file);
    if (!prompt_text) goto done;
    ds4_encode_chat_prompt(tok_engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    free(prompt_text);
    if (prompt.len <= 0) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: empty prompt tokenization\n");
        goto done;
    }
    fprintf(report, "prompt_tokens\t%d\n", prompt.len);

    ds4_v100_context_options ctx_opts;
    ds4_v100_context_options_init(&ctx_opts);
    ctx_opts.pack_index_path = opt.pack_index;
    if (ds4_v100_context_open(&ctx, &ctx_opts, err, sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: context open failed: %s\n",
                err[0] ? err : "context open");
        goto done;
    }
    ds4_v100_tensor_binding output_weight;
    if (ds4_v100_context_output_head_binding(ctx,
                                             &output_weight,
                                             err,
                                             sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: output binding failed: %s\n",
                err[0] ? err : "output binding");
        goto done;
    }

    ds4_v100_mtp_sidecar_options mtp_opts;
    ds4_v100_mtp_sidecar_options_init(&mtp_opts);
    mtp_opts.mtp_path = opt.mtp_model;
    mtp_opts.gpu = opt.gpu;
    mtp_opts.require_device_arena = true;
    if (ds4_v100_mtp_sidecar_open(&sidecar, &mtp_opts, report, err, sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: %s\n",
                err[0] ? err : "failed to open MTP sidecar");
        goto done;
    }
    ds4_gpu_arena *mtp_arena = ds4_v100_mtp_sidecar_arena(sidecar);
    const uint64_t reserve_bytes = (uint64_t)opt.reserve_mib * 1024ull * 1024ull;
    const uint64_t free_after_upload = ds4_gpu_arena_free_after_upload_bytes(mtp_arena);
    fprintf(report, "mtp_uploaded_bytes\t%" PRIu64 "\n", ds4_v100_mtp_sidecar_uploaded_bytes(sidecar));
    fprintf(report, "mtp_arena_bytes\t%" PRIu64 "\n", ds4_gpu_arena_bytes(mtp_arena));
    fprintf(report, "mtp_arena_kind\t%s\n", ds4_gpu_arena_memory_kind(mtp_arena));
    fprintf(report, "mtp_free_after_upload_bytes\t%" PRIu64 "\n", free_after_upload);
    fprintf(report, "reserve_bytes\t%" PRIu64 "\n", reserve_bytes);
    if (free_after_upload < reserve_bytes) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: free_after_upload %" PRIu64
                " below reserve %" PRIu64 "\n",
                free_after_upload,
                reserve_bytes);
        goto done;
    }

    if (open_schedulers(scheds, &opt, &model, err, sizeof(err))) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: open scheduler failed: %s\n",
                err[0] ? err : "open scheduler");
        goto done;
    }
    if (ds4_v100_mtp_forward_open(&mtp_forward,
                                  sidecar,
                                  model.ptr,
                                  model.size,
                                  &output_weight,
                                  opt.gpu,
                                  err,
                                  sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: MTP forward open failed: %s\n",
                err[0] ? err : "MTP forward open");
        goto done;
    }
    if (feed_prompt(scheds, &prompt, err, sizeof(err))) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: prompt decode failed: %s\n",
                err[0] ? err : "prompt decode");
        goto done;
    }
    if (select_topk(scheds, after_prompt_tokens, after_prompt_logits, opt.top_k, err, sizeof(err))) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: after-prompt select failed: %s\n",
                err[0] ? err : "select");
        goto done;
    }

    const uint32_t committed_token = after_prompt_tokens[0];
    const uint32_t committed_pos = (uint32_t)prompt.len;
    if (feed_token(scheds, committed_token, committed_pos, err, sizeof(err))) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: committed decode failed: %s\n",
                err[0] ? err : "committed decode");
        goto done;
    }
    if (select_topk(scheds, post_t_tokens, post_t_logits, opt.top_k, err, sizeof(err))) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: post-commit select failed: %s\n",
                err[0] ? err : "select");
        goto done;
    }

    const uint64_t embed_bytes =
        (uint64_t)DS4_V100_MTP_FORWARD_N_EMBD * sizeof(float);
    const uint64_t hc_bytes =
        (uint64_t)DS4_V100_MTP_FORWARD_HC_VALUES * sizeof(float);
    committed_embed = (float *)malloc((size_t)embed_bytes);
    post_commit_hc = (float *)malloc((size_t)hc_bytes);
    if (!committed_embed || !post_commit_hc) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: MTP input allocation failed\n");
        goto done;
    }
    if (ds4_v100_stage_scheduler_read_token_embedding_f32(scheds[0],
                                                          committed_token,
                                                          committed_embed,
                                                          DS4_V100_MTP_FORWARD_N_EMBD,
                                                          err,
                                                          sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: committed embedding read failed: %s\n",
                err[0] ? err : "embedding");
        goto done;
    }
    if (!ds4_v100_stage_scheduler_read_hc(scheds[DS4_V100_EXPECTED_GPUS - 1],
                                          post_commit_hc,
                                          hc_bytes)) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: post-commit HC read failed\n");
        goto done;
    }
    if (ds4_v100_mtp_forward_run_host(mtp_forward,
                                      committed_embed,
                                      post_commit_hc,
                                      committed_pos,
                                      opt.top_k,
                                      mtp_tokens,
                                      mtp_logits,
                                      &mtp_fwd_report,
                                      err,
                                      sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: native MTP forward failed: %s\n",
                err[0] ? err : "MTP forward");
        goto done;
    }
    if (mtp_tokens[0] == UINT32_MAX) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: native MTP produced no finite draft\n");
        goto done;
    }
    const uint32_t target_top1 = post_t_tokens[0];
    const bool mtp_accept = mtp_tokens[0] == target_top1;
    for (uint32_t i = 0; i < opt.top_k; i++) {
        fprintf(report,
                "mtp_verify_topk\trank=%" PRIu32
                "\ttarget_token=%" PRIu32 "\tmtp_token=%" PRIu32
                "\ttarget_logit=%.9g\tmtp_logit=%.9g\n",
                i + 1,
                post_t_tokens[i],
                mtp_tokens[i],
                post_t_logits[i],
                mtp_logits[i]);
    }
    fprintf(report,
            "mtp_verify_decision\tcommitted_token=%" PRIu32
            "\tcommitted_pos=%" PRIu32
            "\ttarget_top1=%" PRIu32 "\tmtp_top1=%" PRIu32
            "\taccepted=%s\traw_row=%" PRIu32 "\tn_raw=%" PRIu32
            "\toutput_vocab=%" PRIu32
            "\toutput_weight_bytes=%" PRIu64
            "\tfree_after_output_upload_bytes=%" PRIu64 "\n",
            committed_token,
            committed_pos,
            target_top1,
            mtp_tokens[0],
            mtp_accept ? "true" : "false",
            mtp_fwd_report.raw_row,
            mtp_fwd_report.n_raw,
            mtp_fwd_report.output_vocab,
            mtp_fwd_report.output_weight_bytes,
            mtp_fwd_report.free_after_output_upload_bytes);
    if (mtp_fwd_report.free_after_output_upload_bytes < reserve_bytes) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: free_after_output_upload %" PRIu64
                " below reserve %" PRIu64 "\n",
                mtp_fwd_report.free_after_output_upload_bytes,
                reserve_bytes);
        goto done;
    }

    if (create_snapshots(scheds, snaps, &snapshot_bytes, err, sizeof(err))) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: target snapshot failed: %s\n",
                err[0] ? err : "snapshot");
        goto done;
    }

    const uint64_t raw_values = (uint64_t)MTP_VERIFY_RAW_CAP * MTP_VERIFY_HEAD_DIM;
    const uint64_t raw_bytes = raw_values * sizeof(float);
    if (!ds4_gpu_set_device(opt.gpu)) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: failed to set gpu%d\n", opt.gpu);
        goto done;
    }
    mtp_raw = ds4_gpu_tensor_alloc(raw_bytes);
    mtp_raw_snapshot = ds4_gpu_tensor_alloc(raw_bytes);
    if (!mtp_raw || !mtp_raw_snapshot ||
        !ds4_gpu_tensor_fill_f32(mtp_raw, 0.0f, raw_values) ||
        !ds4_gpu_tensor_copy(mtp_raw_snapshot, 0, mtp_raw, 0, raw_bytes) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: MTP raw state allocation failed\n");
        goto done;
    }
    mtp_n_raw = 0;
    mtp_n_raw_snapshot = mtp_n_raw;

    const uint32_t bad_draft = !mtp_accept && mtp_tokens[0] != target_top1
        ? mtp_tokens[0]
        : (post_t_tokens[1] != target_top1 ? post_t_tokens[1]
                                           : (target_top1 == 0 ? 1u : target_top1 - 1u));
    const bool accept = (bad_draft == target_top1);
    if (accept) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: reject draft unexpectedly equals target\n");
        goto done;
    }
    if (feed_token(scheds, bad_draft, committed_pos + 1u, err, sizeof(err))) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: rejected draft mutation failed: %s\n",
                err[0] ? err : "rejected draft mutation");
        goto done;
    }
    if (!ds4_gpu_set_device(opt.gpu) ||
        !ds4_gpu_tensor_fill_f32(mtp_raw, 0.25f, raw_values) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: MTP raw mutation failed\n");
        goto done;
    }
    mtp_n_raw = 1;

    if (restore_snapshots(scheds, snaps, err, sizeof(err))) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: target restore failed: %s\n",
                err[0] ? err : "restore");
        goto done;
    }
    if (!ds4_gpu_set_device(opt.gpu) ||
        !ds4_gpu_tensor_copy(mtp_raw, 0, mtp_raw_snapshot, 0, raw_bytes) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: MTP raw restore failed\n");
        goto done;
    }
    mtp_n_raw = mtp_n_raw_snapshot;
    float raw_probe[16] = {0};
    if (!ds4_gpu_tensor_read(mtp_raw, 0, raw_probe, sizeof(raw_probe))) {
        fprintf(stderr, "ds4-v100-mtp-verify-smoke: MTP raw probe read failed\n");
        goto done;
    }
    mtp_raw_restore_max_abs = max_abs_f32(raw_probe, 16);

    if (select_topk(scheds, restored_tokens, restored_logits, opt.top_k, err, sizeof(err))) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: restored select failed: %s\n",
                err[0] ? err : "select");
        goto done;
    }
    int failures = 0;
    if (compare_topk("restore",
                     post_t_tokens,
                     post_t_logits,
                     restored_tokens,
                     restored_logits,
                     opt.top_k,
                     1.0e-5,
                     report,
                     &restore_delta) != 0) {
        failures++;
    }
    if (mtp_n_raw != 0 || mtp_raw_restore_max_abs > 1.0e-7) {
        fprintf(report,
                "mtp_rollback_mtp_raw_restore\tmtp_n_raw=%" PRIu32
                "\tmax_abs=%.9g\tFAIL\n",
                mtp_n_raw,
                mtp_raw_restore_max_abs);
        failures++;
    } else {
        fprintf(report,
                "mtp_rollback_mtp_raw_restore\tmtp_n_raw=%" PRIu32
                "\tmax_abs=%.9g\tPASS\n",
                mtp_n_raw,
                mtp_raw_restore_max_abs);
    }

    if (feed_token(scheds, target_top1, committed_pos + 1u, err, sizeof(err))) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: continued decode failed: %s\n",
                err[0] ? err : "continued decode");
        goto done;
    }
    if (select_topk(scheds, continued_tokens, continued_logits, opt.top_k, err, sizeof(err))) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: continued select failed: %s\n",
                err[0] ? err : "select");
        goto done;
    }

    if (restore_snapshots(scheds, snaps, err, sizeof(err))) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: clean replay restore failed: %s\n",
                err[0] ? err : "restore");
        goto done;
    }
    if (feed_token(scheds, target_top1, committed_pos + 1u, err, sizeof(err)) ||
        select_topk(scheds, replay_tokens, replay_logits, opt.top_k, err, sizeof(err))) {
        fprintf(stderr,
                "ds4-v100-mtp-verify-smoke: clean replay failed: %s\n",
                err[0] ? err : "clean replay");
        goto done;
    }
    if (compare_topk("clean_replay",
                     continued_tokens,
                     continued_logits,
                     replay_tokens,
                     replay_logits,
                     opt.top_k,
                     1.0e-5,
                     report,
                     &replay_delta) != 0) {
        failures++;
    }

    const double margin = (double)post_t_logits[0] - (double)post_t_logits[1];
    fprintf(report,
            "mtp_rollback_decision\tcommitted_token=%" PRIu32
            "\ttarget_top1=%" PRIu32 "\tmtp_top1=%" PRIu32
            "\tmtp_accepted=%s\trejected_draft=%" PRIu32
            "\tforced_reject=%s\tmargin=%.9g\n",
            committed_token,
            target_top1,
            mtp_tokens[0],
            mtp_accept ? "true" : "false",
            bad_draft,
            mtp_accept ? "true" : "false",
            margin);
    fprintf(report,
            "mtp_verify_summary\tcommitted_token=%" PRIu32
            "\tcommitted_pos=%" PRIu32
            "\ttarget_top1=%" PRIu32 "\tmtp_top1=%" PRIu32
            "\taccepted=%s\traw_row=%" PRIu32 "\tn_raw=%" PRIu32
            "\tsnapshot_bytes=%" PRIu64
            "\trestore_delta=%.9g\treplay_delta=%.9g\t%s\n",
            committed_token,
            committed_pos,
            target_top1,
            mtp_tokens[0],
            mtp_accept ? "true" : "false",
            mtp_fwd_report.raw_row,
            mtp_fwd_report.n_raw,
            snapshot_bytes,
            restore_delta,
            replay_delta,
            failures ? "FAIL" : "PASS");
    fprintf(report,
            "mtp_rollback_summary\tsnapshot_bytes=%" PRIu64
            "\tmtp_raw_bytes=%" PRIu64
            "\trestore_delta=%.9g\treplay_delta=%.9g"
            "\tcontinued_top1=%" PRIu32 "\treplay_top1=%" PRIu32
            "\t%s\n",
            snapshot_bytes,
            raw_bytes,
            restore_delta,
            replay_delta,
            continued_tokens[0],
            replay_tokens[0],
            failures ? "FAIL" : "PASS");
    printf("mtp_verify_smoke: prompt_tokens=%d committed=%" PRIu32
           " target_top1=%" PRIu32 " mtp_top1=%" PRIu32
           " mtp_accepted=%s rejected=%" PRIu32
           " snapshot_bytes=%" PRIu64
           " restore_delta=%.9g replay_delta=%.9g mtp_raw_max_abs=%.9g %s\n",
           prompt.len,
           committed_token,
           target_top1,
           mtp_tokens[0],
           mtp_accept ? "true" : "false",
           bad_draft,
           snapshot_bytes,
           restore_delta,
           replay_delta,
           mtp_raw_restore_max_abs,
           failures ? "FAIL" : "PASS");
    if (failures == 0) rc = 0;

done:
    free(post_commit_hc);
    free(committed_embed);
    free_snapshots(snaps);
    close_schedulers(scheds);
    ds4_gpu_tensor_free(mtp_raw_snapshot);
    ds4_gpu_tensor_free(mtp_raw);
    ds4_v100_mtp_forward_close(mtp_forward);
    ds4_v100_mtp_sidecar_close(sidecar);
    ds4_v100_context_close(ctx);
    ds4_tokens_free(&prompt);
    ds4_engine_close(tok_engine);
    unmap_model_file(&model);
    ds4_gpu_cleanup();
    if (report && report != stdout) fclose(report);
    return rc;
}
