#include "ds4_v100_replay.h"

#include "ds4_gpu.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

struct ds4_v100_replay {
    ds4_engine *tokenizer;
    const unsigned char *model_map;
    uint64_t model_size;
    int model_fd;
    ds4_v100_stage_scheduler *scheds[DS4_V100_EXPECTED_GPUS];
    ds4_v100_replay_options opts;
    double open_ms[DS4_V100_EXPECTED_GPUS];
    double open_total_ms;
    bool used;
};

static double replay_now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static int replay_error(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg ? msg : "V100 replay error");
    return 1;
}

static int replay_errorf(char *err, size_t errlen, const char *fmt, const char *value) {
    if (err && errlen) snprintf(err, errlen, fmt, value ? value : "(null)");
    return 1;
}

static int map_model_file(ds4_v100_replay *rt, const char *path, char *err, size_t errlen) {
    rt->model_fd = open(path, O_RDONLY);
    if (rt->model_fd < 0) {
        if (err && errlen) {
            snprintf(err, errlen, "cannot open model %s: %s", path, strerror(errno));
        }
        return 1;
    }
    struct stat st;
    if (fstat(rt->model_fd, &st) != 0 || st.st_size <= 0) {
        if (err && errlen) snprintf(err, errlen, "cannot stat model %s", path);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, rt->model_fd, 0);
    if (p == MAP_FAILED) {
        if (err && errlen) {
            snprintf(err, errlen, "cannot mmap model %s: %s", path, strerror(errno));
        }
        return 1;
    }
    rt->model_map = (const unsigned char *)p;
    rt->model_size = (uint64_t)st.st_size;
    return 0;
}

void ds4_v100_replay_options_init(ds4_v100_replay_options *opts) {
    if (!opts) return;
    memset(opts, 0, sizeof(*opts));
    opts->kv_ctx_tokens = 1048576;
    opts->attn_comp_cap = 64;
    opts->index_comp_cap = 64;
    opts->indexer_top_k = 512;
}

void ds4_v100_replay_open_counters(const ds4_v100_replay *rt,
                                   ds4_v100_replay_counters *out) {
    if (!rt || !out) return;
    memset(out, 0, sizeof(*out));
    out->open_total_ms = rt->open_total_ms;
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        out->open_ms[i] = rt->open_ms[i];
    }
}

const void *ds4_v100_replay_model_map(const ds4_v100_replay *rt) {
    return rt ? rt->model_map : NULL;
}

uint64_t ds4_v100_replay_model_size(const ds4_v100_replay *rt) {
    return rt ? rt->model_size : 0;
}

void ds4_v100_replay_close(ds4_v100_replay *rt) {
    if (!rt) return;
    for (int i = DS4_V100_EXPECTED_GPUS - 1; i >= 0; i--) {
        ds4_v100_stage_scheduler_close(rt->scheds[i]);
    }
    if (rt->model_map) munmap((void *)rt->model_map, (size_t)rt->model_size);
    if (rt->model_fd >= 0) close(rt->model_fd);
    ds4_engine_close(rt->tokenizer);
    free(rt);
}

int ds4_v100_replay_reset(ds4_v100_replay *rt, char *err, size_t errlen) {
    if (!rt) return replay_error(err, errlen, "missing V100 replay reset input");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (ds4_v100_stage_scheduler_reset(rt->scheds[i], err, errlen)) return 1;
    }
    rt->used = false;
    return 0;
}

typedef struct {
    ds4_v100_stage_scheduler_options opts;
    ds4_v100_stage_scheduler *sched;
    double open_ms;
    int rc;
    char err[512];
} replay_open_worker;

static void *replay_open_worker_main(void *arg) {
    replay_open_worker *w = (replay_open_worker *)arg;
    if (!w) return NULL;
    const double t0 = replay_now_ms();
    w->rc = ds4_v100_stage_scheduler_open(&w->sched,
                                          &w->opts,
                                          w->err,
                                          sizeof(w->err));
    w->open_ms = replay_now_ms() - t0;
    return NULL;
}

static int replay_open_stages_serial(ds4_v100_replay *rt,
                                     const ds4_v100_stage_scheduler_options *base,
                                     char *err,
                                     size_t errlen) {
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        ds4_v100_stage_scheduler_options sopts = *base;
        sopts.stage_id = stage;
        const double t0 = replay_now_ms();
        if (ds4_v100_stage_scheduler_open(&rt->scheds[stage], &sopts, err, errlen)) {
            return 1;
        }
        rt->open_ms[stage] = replay_now_ms() - t0;
    }
    return 0;
}

static int replay_open_stages_parallel(ds4_v100_replay *rt,
                                       const ds4_v100_stage_scheduler_options *base,
                                       char *err,
                                       size_t errlen) {
    replay_open_worker workers[DS4_V100_EXPECTED_GPUS];
    pthread_t threads[DS4_V100_EXPECTED_GPUS];
    bool created[DS4_V100_EXPECTED_GPUS];
    memset(workers, 0, sizeof(workers));
    memset(threads, 0, sizeof(threads));
    memset(created, 0, sizeof(created));

    int create_failed = -1;
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        workers[stage].opts = *base;
        workers[stage].opts.stage_id = stage;
        if (pthread_create(&threads[stage],
                           NULL,
                           replay_open_worker_main,
                           &workers[stage]) != 0) {
            create_failed = stage;
            snprintf(workers[stage].err,
                     sizeof(workers[stage].err),
                     "failed to create stage-open worker %d",
                     stage);
            workers[stage].rc = 1;
            break;
        }
        created[stage] = true;
    }

    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        if (created[stage]) {
            (void)pthread_join(threads[stage], NULL);
        }
    }

    int first_failed = create_failed;
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        if (workers[stage].rc != 0 && first_failed < 0) first_failed = stage;
    }
    if (first_failed >= 0) {
        if (err && errlen) {
            snprintf(err,
                     errlen,
                     "stage %d parallel open failed: %s",
                     first_failed,
                     workers[first_failed].err[0] ? workers[first_failed].err : "unknown error");
        }
        for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
            ds4_v100_stage_scheduler_close(workers[stage].sched);
        }
        return 1;
    }

    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        rt->scheds[stage] = workers[stage].sched;
        rt->open_ms[stage] = workers[stage].open_ms;
        workers[stage].sched = NULL;
    }
    return 0;
}

int ds4_v100_replay_open(ds4_v100_replay **out,
                         const ds4_v100_replay_options *opts,
                         char *err,
                         size_t errlen) {
    if (!out) return replay_error(err, errlen, "missing V100 replay output");
    *out = NULL;
    if (!opts || !opts->model_path || !opts->pack_index_path) {
        return replay_error(err, errlen, "missing V100 replay model or pack index");
    }
    const int devices = ds4_gpu_device_count();
    if (devices < DS4_V100_EXPECTED_GPUS) {
        if (err && errlen) {
            snprintf(err,
                     errlen,
                     "V100 replay requires %d CUDA devices, got %d",
                     DS4_V100_EXPECTED_GPUS,
                     devices);
        }
        return 1;
    }

    ds4_v100_replay *rt = (ds4_v100_replay *)calloc(1, sizeof(*rt));
    if (!rt) return replay_error(err, errlen, "failed to allocate V100 replay");
    rt->model_fd = -1;
    rt->opts = *opts;

    const double open0 = replay_now_ms();
    ds4_engine_options eopts;
    memset(&eopts, 0, sizeof(eopts));
    eopts.model_path = opts->model_path;
    eopts.backend = DS4_BACKEND_CPU;
    eopts.inspect_only = true;
    eopts.n_threads = 1;
    if (ds4_engine_open(&rt->tokenizer, &eopts) != 0) {
        ds4_v100_replay_close(rt);
        return replay_errorf(err, errlen, "tokenizer engine open failed for %s", opts->model_path);
    }
    if (map_model_file(rt, opts->model_path, err, errlen)) {
        ds4_v100_replay_close(rt);
        return 1;
    }
    if (!ds4_gpu_set_model_fd(rt->model_fd)) {
        ds4_v100_replay_close(rt);
        return replay_error(err, errlen, "failed to register V100 replay model fd");
    }

    ds4_v100_stage_scheduler_options sopts;
    ds4_v100_stage_scheduler_options_init(&sopts);
    sopts.pack_index_path = opts->pack_index_path;
    sopts.model_map = rt->model_map;
    sopts.model_size = rt->model_size;
    sopts.kv_ctx_tokens = opts->kv_ctx_tokens;
    sopts.attn_comp_cap = opts->attn_comp_cap;
    sopts.index_comp_cap = opts->index_comp_cap;
    sopts.indexer_top_k = opts->indexer_top_k;
    sopts.fp8_kv_cache = opts->fp8_kv_cache;
    int open_rc = opts->serial_open
        ? replay_open_stages_serial(rt, &sopts, err, errlen)
        : replay_open_stages_parallel(rt, &sopts, err, errlen);
    if (open_rc) {
        if (!opts->serial_open && err && errlen && err[0] == '\0') {
            snprintf(err, errlen, "parallel stage open failed");
        }
        ds4_v100_replay_close(rt);
        return 1;
    }
    rt->open_total_ms = replay_now_ms() - open0;
    *out = rt;
    return 0;
}

void ds4_v100_replay_encode_prompt(ds4_v100_replay *rt,
                                   const char *system,
                                   const char *prompt,
                                   ds4_think_mode think_mode,
                                   ds4_tokens *out) {
    if (!rt || !out) return;
    ds4_encode_chat_prompt(rt->tokenizer,
                           system ? system : "",
                           prompt ? prompt : "",
                           think_mode,
                           out);
}

static void counters_add_report(ds4_v100_replay_counters *c,
                                int stage,
                                const ds4_v100_stage_scheduler_report *r) {
    if (!c || !r || stage < 0 || stage >= DS4_V100_EXPECTED_GPUS) return;
    if (c->arena_bytes[stage] == 0) {
        c->uploaded_tensors += r->uploaded_tensors;
        c->uploaded_bytes += r->uploaded_bytes;
    }
    c->arena_bytes[stage] = r->arena_bytes;
    c->layers_executed += r->layers_executed;
}

static int replay_feed_token(ds4_v100_replay *rt,
                             uint32_t token,
                             uint32_t position,
                             ds4_v100_replay_counters *counters,
                             double *bucket_ms,
                             char *err,
                             size_t errlen) {
    ds4_v100_stage_scheduler_report report;
    memset(&report, 0, sizeof(report));
    double t0 = replay_now_ms();
    if (ds4_v100_stage_scheduler_decode_token(rt->scheds[0],
                                              token,
                                              position,
                                              &report,
                                              err,
                                              errlen)) {
        return 1;
    }
    if (!ds4_gpu_synchronize()) return replay_error(err, errlen, "stage 0 synchronize failed");
    double dt = replay_now_ms() - t0;
    if (counters) counters->stage_decode_ms[0] += dt;
    if (bucket_ms) *bucket_ms += dt;
    counters_add_report(counters, 0, &report);

    for (int stage = 1; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        t0 = replay_now_ms();
        if (ds4_v100_stage_scheduler_handoff(rt->scheds[stage],
                                             rt->scheds[stage - 1],
                                             err,
                                             errlen)) {
            return 1;
        }
        dt = replay_now_ms() - t0;
        if (counters) counters->handoff_ms[stage - 1] += dt;
        if (bucket_ms) *bucket_ms += dt;

        memset(&report, 0, sizeof(report));
        t0 = replay_now_ms();
        if (ds4_v100_stage_scheduler_decode_hc(rt->scheds[stage],
                                               token,
                                               position,
                                               &report,
                                               err,
                                               errlen)) {
            return 1;
        }
        if (!ds4_gpu_synchronize()) return replay_error(err, errlen, "stage synchronize failed");
        dt = replay_now_ms() - t0;
        if (counters) counters->stage_decode_ms[stage] += dt;
        if (bucket_ms) *bucket_ms += dt;
        counters_add_report(counters, stage, &report);
    }
    return 0;
}

static int replay_select_token(ds4_v100_replay *rt,
                               ds4_v100_replay_output *out,
                               ds4_v100_replay_counters *counters,
                               char *err,
                               size_t errlen) {
    uint32_t token = UINT32_MAX;
    float logit = 0.0f;
    double t0 = replay_now_ms();
    if (ds4_v100_stage_scheduler_select_token(rt->scheds[DS4_V100_EXPECTED_GPUS - 1],
                                              &token,
                                              &logit,
                                              err,
                                              errlen)) {
        return 1;
    }
    if (counters) counters->output_head_ms += replay_now_ms() - t0;

    memset(out, 0, sizeof(*out));
    out->token = token;
    out->logit = logit;
    t0 = replay_now_ms();
    out->text = ds4_token_text(rt->tokenizer, (int)token, &out->text_len);
    if (counters) counters->token_text_ms += replay_now_ms() - t0;
    if (!out->text) return replay_error(err, errlen, "failed to decode selected token text");
    return 0;
}

int ds4_v100_replay_generate(ds4_v100_replay *rt,
                             const ds4_tokens *prompt,
                             uint32_t max_tokens,
                             ds4_v100_replay_output *outputs,
                             uint32_t output_cap,
                             uint32_t *out_count,
                             ds4_v100_replay_counters *counters,
                             char *err,
                             size_t errlen) {
    if (out_count) *out_count = 0;
    if (!rt || !prompt || prompt->len <= 0 || !outputs || output_cap == 0 || max_tokens == 0) {
        return replay_error(err, errlen, "missing V100 replay generation input");
    }
    if (max_tokens > output_cap) return replay_error(err, errlen, "V100 replay output buffer too small");
    if (rt->used) {
        return replay_error(err,
                            errlen,
                            "V100 replay runtime is one-shot; reopen it for another prompt");
    }
    if (rt->opts.kv_ctx_tokens && (uint64_t)prompt->len + max_tokens > rt->opts.kv_ctx_tokens) {
        return replay_error(err, errlen, "V100 replay prompt exceeds configured context");
    }

    ds4_v100_replay_counters local;
    ds4_v100_replay_counters *c = counters ? counters : &local;
    memset(c, 0, sizeof(*c));
    c->prompt_tokens = (uint32_t)prompt->len;
    c->open_total_ms = rt->open_total_ms;
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) c->open_ms[i] = rt->open_ms[i];

    const double total0 = replay_now_ms();
    for (int pos = 0; pos < prompt->len; pos++) {
        if (prompt->v[pos] < 0) return replay_error(err, errlen, "negative prompt token");
        if (replay_feed_token(rt,
                              (uint32_t)prompt->v[pos],
                              (uint32_t)pos,
                              c,
                              &c->prompt_replay_ms,
                              err,
                              errlen)) {
            return 1;
        }
        c->total_input_tokens++;
    }

    uint32_t n_out = 0;
    for (uint32_t i = 0; i < max_tokens; i++) {
        if (replay_select_token(rt, &outputs[n_out], c, err, errlen)) return 1;
        n_out++;
        if (i + 1 == max_tokens) break;
        const uint32_t next_pos = (uint32_t)prompt->len + i;
        if (replay_feed_token(rt,
                              outputs[n_out - 1].token,
                              next_pos,
                              c,
                              &c->continuation_decode_ms,
                              err,
                              errlen)) {
            return 1;
        }
        c->total_input_tokens++;
    }

    c->generated_tokens = n_out;
    c->total_ms = replay_now_ms() - total0;
    rt->used = true;
    if (out_count) *out_count = n_out;
    return 0;
}

int ds4_v100_replay_read_token_embedding_f32(ds4_v100_replay *rt,
                                             uint32_t token,
                                             float *dst,
                                             uint64_t dst_values,
                                             char *err,
                                             size_t errlen) {
    if (!rt || !rt->scheds[0] || !dst) {
        return replay_error(err, errlen, "missing V100 replay embedding read input");
    }
    return ds4_v100_stage_scheduler_read_token_embedding_f32(rt->scheds[0],
                                                             token,
                                                             dst,
                                                             dst_values,
                                                             err,
                                                             errlen);
}

int ds4_v100_replay_read_output_hc(ds4_v100_replay *rt,
                                   float *dst,
                                   uint64_t bytes,
                                   char *err,
                                   size_t errlen) {
    if (!rt || !rt->scheds[DS4_V100_EXPECTED_GPUS - 1] || !dst) {
        return replay_error(err, errlen, "missing V100 replay HC read input");
    }
    if (!ds4_v100_stage_scheduler_read_hc(rt->scheds[DS4_V100_EXPECTED_GPUS - 1],
                                          dst,
                                          bytes)) {
        return replay_error(err, errlen, "V100 replay HC read failed");
    }
    return 0;
}

void ds4_v100_replay_output_free(ds4_v100_replay_output *out) {
    if (!out) return;
    free(out->text);
    memset(out, 0, sizeof(*out));
}
