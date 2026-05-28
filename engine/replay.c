#include "engine/replay.h"

#include "ds4_gpu.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

typedef struct replay_pipeline_runtime replay_pipeline_runtime;
typedef struct replay_mailbox_runtime replay_mailbox_runtime;

struct ds4_v100_replay {
    ds4_engine *tokenizer;
    const unsigned char *model_map;
    uint64_t model_size;
    int model_fd;
    ds4_v100_stage_scheduler *scheds[DS4_V100_EXPECTED_GPUS];
    ds4_gpu_event *stage_ready[DS4_V100_EXPECTED_GPUS][DS4_V100_SCHED_MAX_SLOTS];
    replay_pipeline_runtime *pipeline;
    replay_mailbox_runtime *mailbox;
    ds4_v100_replay_options opts;
    double open_ms[DS4_V100_EXPECTED_GPUS];
    double open_total_ms;
    bool used;
};

struct ds4_v100_replay_snapshot {
    ds4_v100_stage_scheduler_snapshot *stages[DS4_V100_EXPECTED_GPUS];
    uint64_t bytes;
};

static void replay_pipeline_runtime_close(replay_pipeline_runtime *p);
static int replay_pipeline_runtime_open(ds4_v100_replay *rt, char *err, size_t errlen);
static void replay_mailbox_runtime_close(replay_mailbox_runtime *m);
static int replay_mailbox_runtime_open(ds4_v100_replay *rt, char *err, size_t errlen);
static int replay_open_stage_ready_events(ds4_v100_replay *rt, char *err, size_t errlen);
static void replay_close_stage_ready_events(ds4_v100_replay *rt);

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

static void replay_init_counters(ds4_v100_replay *rt,
                                 uint32_t prompt_tokens,
                                 ds4_v100_replay_counters *c) {
    if (!rt || !c) return;
    memset(c, 0, sizeof(*c));
    c->prompt_tokens = prompt_tokens;
    c->open_total_ms = rt->open_total_ms;
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) c->open_ms[i] = rt->open_ms[i];
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
    opts->kv_active_slots = 1;
    opts->suppress_router_readback = true;
    opts->wavefront_decode = false;
    opts->async_pipeline_decode = false;
    opts->async_handoff = false;
    opts->async_event_handoff = false;
    opts->async_pipeline_mode = DS4_V100_REPLAY_ASYNC_PIPELINE_OFF;
}

static ds4_v100_replay_async_pipeline_mode
replay_async_pipeline_mode(const ds4_v100_replay_options *opts) {
    if (!opts) return DS4_V100_REPLAY_ASYNC_PIPELINE_OFF;
    if (opts->async_pipeline_mode != DS4_V100_REPLAY_ASYNC_PIPELINE_OFF) {
        return opts->async_pipeline_mode;
    }
    return opts->async_pipeline_decode
        ? DS4_V100_REPLAY_ASYNC_PIPELINE_PER_STEP
        : DS4_V100_REPLAY_ASYNC_PIPELINE_OFF;
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
    replay_pipeline_runtime_close(rt->pipeline);
    replay_mailbox_runtime_close(rt->mailbox);
    replay_close_stage_ready_events(rt);
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

static void replay_close_stage_ready_events(ds4_v100_replay *rt) {
    if (!rt) return;
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        for (uint32_t slot = 0; slot < DS4_V100_SCHED_MAX_SLOTS; slot++) {
            ds4_gpu_event_free(rt->stage_ready[stage][slot]);
            rt->stage_ready[stage][slot] = NULL;
        }
    }
}

static int replay_open_stage_ready_events(ds4_v100_replay *rt, char *err, size_t errlen) {
    if (!rt || !rt->opts.async_event_handoff ||
        replay_async_pipeline_mode(&rt->opts) != DS4_V100_REPLAY_ASYNC_PIPELINE_PER_STEP) {
        return 0;
    }
    uint64_t slots64 = rt->opts.kv_active_slots ? rt->opts.kv_active_slots : 1;
    if (slots64 > DS4_V100_SCHED_MAX_SLOTS) slots64 = DS4_V100_SCHED_MAX_SLOTS;
    const uint32_t slots = (uint32_t)slots64;
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        for (uint32_t slot = 0; slot < slots; slot++) {
            rt->stage_ready[stage][slot] = ds4_gpu_event_create(stage);
            if (!rt->stage_ready[stage][slot]) {
                replay_close_stage_ready_events(rt);
                return replay_error(err, errlen, "failed to create V100 replay stage event");
            }
        }
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
    sopts.turbomind_pack_index_path = opts->turbomind_pack_index_path;
    sopts.shard_dir = opts->shard_dir;
    sopts.model_map = rt->model_map;
    sopts.model_size = rt->model_size;
    sopts.kv_ctx_tokens = opts->kv_ctx_tokens;
    sopts.attn_comp_cap = opts->attn_comp_cap;
    sopts.index_comp_cap = opts->index_comp_cap;
    sopts.indexer_top_k = opts->indexer_top_k;
    sopts.fp8_kv_cache = opts->fp8_kv_cache;
    sopts.kv_active_slots = opts->kv_active_slots ? opts->kv_active_slots : 1;
    sopts.suppress_router_readback = opts->suppress_router_readback;
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
    if (replay_open_stage_ready_events(rt, err, errlen)) {
        ds4_v100_replay_close(rt);
        return 1;
    }
    switch (replay_async_pipeline_mode(opts)) {
    case DS4_V100_REPLAY_ASYNC_PIPELINE_PERSISTENT:
        if (replay_pipeline_runtime_open(rt, err, errlen)) {
            ds4_v100_replay_close(rt);
            return 1;
        }
        break;
    case DS4_V100_REPLAY_ASYNC_PIPELINE_MAILBOX:
        if (replay_mailbox_runtime_open(rt, err, errlen)) {
            ds4_v100_replay_close(rt);
            return 1;
        }
        break;
    case DS4_V100_REPLAY_ASYNC_PIPELINE_PER_STEP:
    case DS4_V100_REPLAY_ASYNC_PIPELINE_OFF:
    default:
        break;
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
    c->stage_hc_attn_ms[stage] += r->timing_hc_attn_ms;
    c->stage_attention_ms[stage] += r->timing_attention_ms;
    c->stage_attn_proj_ms[stage] += r->timing_attn_proj_ms;
    c->stage_attn_cache_ms[stage] += r->timing_attn_cache_ms;
    c->stage_attn_softmax_ms[stage] += r->timing_attn_softmax_ms;
    c->stage_attn_inverse_rope_ms[stage] += r->timing_attn_inverse_rope_ms;
    c->stage_attn_output_ms[stage] += r->timing_attn_output_ms;
    c->stage_hc_ffn_ms[stage] += r->timing_hc_ffn_ms;
    c->stage_ffn_ms[stage] += r->timing_ffn_ms;
    c->stage_hc_final_ms[stage] += r->timing_hc_final_ms;
    c->stage_profile_total_ms[stage] += r->timing_total_ms;
}

static int replay_handoff_slot_span(ds4_v100_replay *rt,
                                    int stage,
                                    uint32_t slot_start,
                                    uint32_t n_slots,
                                    char *err,
                                    size_t errlen) {
    if (!rt || stage <= 0 || stage >= DS4_V100_EXPECTED_GPUS) {
        return replay_error(err, errlen, "missing replay handoff endpoint");
    }
    if (rt->opts.async_handoff) {
        return ds4_v100_stage_scheduler_handoff_slot_span_async(rt->scheds[stage],
                                                               rt->scheds[stage - 1],
                                                               slot_start,
                                                               n_slots,
                                                               err,
                                                               errlen);
    }
    return ds4_v100_stage_scheduler_handoff_slot_span(rt->scheds[stage],
                                                     rt->scheds[stage - 1],
                                                     slot_start,
                                                     n_slots,
                                                     err,
                                                     errlen);
}

static int replay_handoff_slot_span_after_event(ds4_v100_replay *rt,
                                                int stage,
                                                uint32_t slot_start,
                                                uint32_t n_slots,
                                                const ds4_gpu_event *event,
                                                char *err,
                                                size_t errlen) {
    if (!event) return replay_handoff_slot_span(rt, stage, slot_start, n_slots, err, errlen);
    if (!rt || stage <= 0 || stage >= DS4_V100_EXPECTED_GPUS) {
        return replay_error(err, errlen, "missing replay event handoff endpoint");
    }
    return ds4_v100_stage_scheduler_handoff_slot_span_after_event_async(
        rt->scheds[stage],
        rt->scheds[stage - 1],
        slot_start,
        n_slots,
        event,
        err,
        errlen);
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
        if (replay_handoff_slot_span(rt, stage, 0, 1, err, errlen)) {
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

typedef struct {
    uint32_t prompt_idx;
    uint32_t prompt_len;
} replay_batch_slot_plan;

static int replay_feed_token_batch(ds4_v100_replay *rt,
                                   const uint32_t *tokens,
                                   const uint32_t *positions,
                                   uint32_t n_slots,
                                   ds4_v100_replay_counters *counters,
                                   double *bucket_ms,
                                   char *err,
                                   size_t errlen) {
    ds4_v100_stage_scheduler_report reports[DS4_V100_SCHED_MAX_SLOTS];
    memset(reports, 0, sizeof(reports));
    double t0 = replay_now_ms();
    if (ds4_v100_stage_scheduler_decode_token_batch(rt->scheds[0],
                                                    tokens,
                                                    positions,
                                                    n_slots,
                                                    reports,
                                                    err,
                                                    errlen)) {
        return 1;
    }
    if (!ds4_gpu_synchronize()) return replay_error(err, errlen, "stage 0 synchronize failed");
    double dt = replay_now_ms() - t0;
    if (counters) counters->stage_decode_ms[0] += dt;
    if (bucket_ms) *bucket_ms += dt;
    counters_add_report(counters, 0, &reports[0]);

    for (int stage = 1; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        t0 = replay_now_ms();
        if (replay_handoff_slot_span(rt, stage, 0, n_slots, err, errlen)) {
            return 1;
        }
        dt = replay_now_ms() - t0;
        if (counters) counters->handoff_ms[stage - 1] += dt;
        if (bucket_ms) *bucket_ms += dt;

        memset(reports, 0, sizeof(reports));
        t0 = replay_now_ms();
        if (ds4_v100_stage_scheduler_decode_hc_batch(rt->scheds[stage],
                                                     tokens,
                                                     positions,
                                                     n_slots,
                                                     reports,
                                                     err,
                                                     errlen)) {
            return 1;
        }
        if (!ds4_gpu_synchronize()) return replay_error(err, errlen, "stage synchronize failed");
        dt = replay_now_ms() - t0;
        if (counters) counters->stage_decode_ms[stage] += dt;
        if (bucket_ms) *bucket_ms += dt;
        counters_add_report(counters, stage, &reports[0]);
    }
    return 0;
}

static int replay_sync_all_stages(char *err, size_t errlen) {
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        if (!ds4_gpu_set_device(stage)) {
            return replay_error(err, errlen, "wavefront set-device failed");
        }
        if (!ds4_gpu_synchronize()) {
            return replay_error(err, errlen, "wavefront synchronize failed");
        }
    }
    return 0;
}

static int replay_feed_token_batch_wavefront(ds4_v100_replay *rt,
                                             const uint32_t *tokens,
                                             const uint32_t *positions,
                                             uint32_t n_slots,
                                             ds4_v100_replay_counters *counters,
                                             double *bucket_ms,
                                             char *err,
                                             size_t errlen) {
    if (!rt || !tokens || !positions || n_slots == 0 ||
        n_slots > DS4_V100_SCHED_MAX_SLOTS) {
        return replay_error(err, errlen, "missing V100 replay wavefront input");
    }

    const double total0 = replay_now_ms();
    for (uint32_t diag = 0; diag < n_slots + DS4_V100_EXPECTED_GPUS - 1; diag++) {
        int stage_hi = (int)diag;
        if (stage_hi >= DS4_V100_EXPECTED_GPUS) stage_hi = DS4_V100_EXPECTED_GPUS - 1;
        int stage_lo = (int)diag - (int)n_slots + 1;
        if (stage_lo < 0) stage_lo = 0;

        for (int stage = stage_hi; stage >= stage_lo; stage--) {
            const uint32_t slot = diag - (uint32_t)stage;
            ds4_v100_stage_scheduler_report report;
            memset(&report, 0, sizeof(report));
            const double t0 = replay_now_ms();

            if (stage == 0) {
                if (ds4_v100_stage_scheduler_decode_token_slot_span(rt->scheds[0],
                                                                     slot,
                                                                     &tokens[slot],
                                                                     &positions[slot],
                                                                     1,
                                                                     &report,
                                                                     err,
                                                                     errlen)) {
                    return 1;
                }
            } else {
                if (replay_handoff_slot_span(rt,
                                             stage,
                                             slot,
                                             1,
                                             err,
                                             errlen) ||
                    ds4_v100_stage_scheduler_decode_hc_slot_span(rt->scheds[stage],
                                                                 slot,
                                                                 &tokens[slot],
                                                                 &positions[slot],
                                                                 1,
                                                                 &report,
                                                                 err,
                                                                 errlen)) {
                    return 1;
                }
            }

            if (counters) {
                counters->stage_decode_ms[stage] += replay_now_ms() - t0;
                if (slot == 0) counters_add_report(counters, stage, &report);
            }
        }
    }

    if (replay_sync_all_stages(err, errlen)) return 1;
    if (bucket_ms) *bucket_ms += replay_now_ms() - total0;
    return 0;
}

typedef struct {
    replay_pipeline_runtime *pipeline;
    int stage;
} replay_pipeline_worker;

struct replay_pipeline_runtime {
    ds4_v100_replay *rt;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_t threads[DS4_V100_EXPECTED_GPUS];
    bool thread_created[DS4_V100_EXPECTED_GPUS];
    replay_pipeline_worker workers[DS4_V100_EXPECTED_GPUS];
    bool stop;
    bool active;
    bool failed;
    uint64_t generation;
    uint32_t completed_workers;
    const uint32_t *tokens;
    const uint32_t *positions;
    uint32_t n_slots;
    bool done[DS4_V100_EXPECTED_GPUS][DS4_V100_SCHED_MAX_SLOTS];
    char err[512];
    ds4_v100_stage_scheduler_report reports[DS4_V100_EXPECTED_GPUS][DS4_V100_SCHED_MAX_SLOTS];
    double stage_decode_ms[DS4_V100_EXPECTED_GPUS];
    double handoff_ms[DS4_V100_EXPECTED_GPUS - 1];
    double worker_wait_ms[DS4_V100_EXPECTED_GPUS];
    double sync_ms[DS4_V100_EXPECTED_GPUS];
};

static void replay_pipeline_fail(replay_pipeline_runtime *p, const char *msg) {
    pthread_mutex_lock(&p->mu);
    if (!p->failed) {
        p->failed = true;
        snprintf(p->err, sizeof(p->err), "%s", msg ? msg : "async pipeline failed");
    }
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
}

static bool replay_pipeline_job_live(replay_pipeline_runtime *p, uint64_t generation) {
    pthread_mutex_lock(&p->mu);
    const bool live = !p->stop && p->active &&
        p->generation == generation && !p->failed;
    pthread_mutex_unlock(&p->mu);
    return live;
}

static bool replay_pipeline_wait_prev(replay_pipeline_runtime *p,
                                      int stage,
                                      uint32_t slot,
                                      uint64_t generation) {
    if (stage == 0) return true;
    pthread_mutex_lock(&p->mu);
    while (!p->stop && p->active && p->generation == generation &&
           !p->failed && !p->done[stage - 1][slot]) {
        pthread_cond_wait(&p->cv, &p->mu);
    }
    const bool ok = !p->stop && p->active &&
        p->generation == generation && !p->failed;
    pthread_mutex_unlock(&p->mu);
    return ok;
}

static void replay_pipeline_mark_done(replay_pipeline_runtime *p,
                                      int stage,
                                      uint32_t slot,
                                      uint64_t generation) {
    pthread_mutex_lock(&p->mu);
    if (p->active && p->generation == generation && !p->failed) {
        p->done[stage][slot] = true;
    }
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
}

static void replay_pipeline_worker_done(replay_pipeline_runtime *p,
                                        uint64_t generation) {
    pthread_mutex_lock(&p->mu);
    if (p->active && p->generation == generation) {
        p->completed_workers++;
        if (p->completed_workers >= DS4_V100_EXPECTED_GPUS) {
            p->active = false;
        }
    }
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
}

static void *replay_pipeline_worker_main(void *arg) {
    replay_pipeline_worker *w = (replay_pipeline_worker *)arg;
    if (!w || !w->pipeline) return NULL;
    replay_pipeline_runtime *p = w->pipeline;
    const int stage = w->stage;
    uint64_t seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (!p->stop && (!p->active || p->generation == seen_generation)) {
            pthread_cond_wait(&p->cv, &p->mu);
        }
        if (p->stop) {
            pthread_mutex_unlock(&p->mu);
            break;
        }
        seen_generation = p->generation;
        pthread_mutex_unlock(&p->mu);

        char local_err[512] = {0};
        for (uint32_t slot = 0; slot < p->n_slots; slot++) {
            if (!replay_pipeline_job_live(p, seen_generation)) {
                break;
            }
            const double wait0 = replay_now_ms();
            const bool prev_ready =
                replay_pipeline_wait_prev(p, stage, slot, seen_generation);
            p->worker_wait_ms[stage] += replay_now_ms() - wait0;
            if (!prev_ready) {
                break;
            }

            ds4_v100_stage_scheduler_report report;
            memset(&report, 0, sizeof(report));
            double t0 = replay_now_ms();
            if (stage == 0) {
                if (ds4_v100_stage_scheduler_decode_token_slot_span(
                        p->rt->scheds[0],
                        slot,
                        &p->tokens[slot],
                        &p->positions[slot],
                        1,
                        &report,
                        local_err,
                        sizeof(local_err))) {
                    replay_pipeline_fail(p, local_err);
                    break;
                }
            } else {
                if (replay_handoff_slot_span(p->rt,
                                             stage,
                                             slot,
                                             1,
                                             local_err,
                                             sizeof(local_err))) {
                    replay_pipeline_fail(p, local_err);
                    break;
                }
                p->handoff_ms[stage - 1] += replay_now_ms() - t0;
                t0 = replay_now_ms();
                if (ds4_v100_stage_scheduler_decode_hc_slot_span(
                        p->rt->scheds[stage],
                        slot,
                        &p->tokens[slot],
                        &p->positions[slot],
                        1,
                        &report,
                        local_err,
                        sizeof(local_err))) {
                    replay_pipeline_fail(p, local_err);
                    break;
                }
            }
            const double sync0 = replay_now_ms();
            if (!ds4_gpu_set_device(stage) || !ds4_gpu_synchronize()) {
                replay_pipeline_fail(p, "async pipeline synchronize failed");
                break;
            }
            p->sync_ms[stage] += replay_now_ms() - sync0;
            p->stage_decode_ms[stage] += replay_now_ms() - t0;
            p->reports[stage][slot] = report;
            replay_pipeline_mark_done(p, stage, slot, seen_generation);
        }

        replay_pipeline_worker_done(p, seen_generation);
    }
    return NULL;
}

static int replay_pipeline_runtime_open(ds4_v100_replay *rt, char *err, size_t errlen) {
    if (!rt) return replay_error(err, errlen, "missing async pipeline runtime input");
    if (rt->pipeline) return 0;

    replay_pipeline_runtime *p =
        (replay_pipeline_runtime *)calloc(1, sizeof(*p));
    if (!p) return replay_error(err, errlen, "failed to allocate async pipeline runtime");
    p->rt = rt;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);

    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        p->workers[stage].pipeline = p;
        p->workers[stage].stage = stage;
        if (pthread_create(&p->threads[stage],
                           NULL,
                           replay_pipeline_worker_main,
                           &p->workers[stage]) != 0) {
            if (err && errlen) {
                snprintf(err, errlen, "failed to create persistent async pipeline worker %d", stage);
            }
            replay_pipeline_runtime_close(p);
            return 1;
        }
        p->thread_created[stage] = true;
    }
    rt->pipeline = p;
    return 0;
}

static void replay_pipeline_runtime_close(replay_pipeline_runtime *p) {
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    p->stop = true;
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);

    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        if (p->thread_created[stage]) {
            (void)pthread_join(p->threads[stage], NULL);
        }
    }
    pthread_cond_destroy(&p->cv);
    pthread_mutex_destroy(&p->mu);
    free(p);
}

static int replay_pipeline_runtime_dispatch(replay_pipeline_runtime *p,
                                            const uint32_t *tokens,
                                            const uint32_t *positions,
                                            uint32_t n_slots,
                                            ds4_v100_replay_counters *counters,
                                            double *bucket_ms,
                                            char *err,
                                            size_t errlen) {
    if (!p || !tokens || !positions || n_slots == 0 ||
        n_slots > DS4_V100_SCHED_MAX_SLOTS) {
        return replay_error(err, errlen, "missing async pipeline dispatch input");
    }

    const double total0 = replay_now_ms();
    pthread_mutex_lock(&p->mu);
    if (p->active) {
        pthread_mutex_unlock(&p->mu);
        return replay_error(err, errlen, "async pipeline dispatch while active");
    }
    memset(p->done, 0, sizeof(p->done));
    memset(p->reports, 0, sizeof(p->reports));
    memset(p->stage_decode_ms, 0, sizeof(p->stage_decode_ms));
    memset(p->handoff_ms, 0, sizeof(p->handoff_ms));
    memset(p->worker_wait_ms, 0, sizeof(p->worker_wait_ms));
    memset(p->sync_ms, 0, sizeof(p->sync_ms));
    p->err[0] = '\0';
    p->failed = false;
    p->tokens = tokens;
    p->positions = positions;
    p->n_slots = n_slots;
    p->completed_workers = 0;
    p->generation++;
    p->active = true;
    pthread_cond_broadcast(&p->cv);
    const uint64_t generation = p->generation;
    const double setup_ms = replay_now_ms() - total0;
    const double wait0 = replay_now_ms();
    while (p->active && p->generation == generation) {
        pthread_cond_wait(&p->cv, &p->mu);
    }
    const double host_wait_ms = replay_now_ms() - wait0;
    const bool failed = p->failed;
    char local_err[512] = {0};
    if (failed) snprintf(local_err, sizeof(local_err), "%s", p->err);
    pthread_mutex_unlock(&p->mu);

    if (failed) {
        if (err && errlen) {
            snprintf(err, errlen, "%s", local_err[0] ? local_err : "async pipeline failed");
        }
        return 1;
    }

    const double complete0 = replay_now_ms();
    if (replay_sync_all_stages(err, errlen)) return 1;
    const double complete_ms = replay_now_ms() - complete0;
    const double total_ms = replay_now_ms() - total0;
    if (bucket_ms) *bucket_ms += total_ms;
    if (counters) {
        counters->async_pipeline_dispatches++;
        counters->async_pipeline_total_ms += total_ms;
        counters->async_pipeline_setup_ms += setup_ms;
        counters->async_pipeline_host_wait_ms += host_wait_ms;
        counters->async_pipeline_complete_ms += complete_ms;
        for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
            counters->stage_decode_ms[stage] += p->stage_decode_ms[stage];
            counters->async_pipeline_worker_wait_ms[stage] += p->worker_wait_ms[stage];
            counters->async_pipeline_sync_ms[stage] += p->sync_ms[stage];
            counters_add_report(counters, stage, &p->reports[stage][0]);
        }
        for (int stage = 1; stage < DS4_V100_EXPECTED_GPUS; stage++) {
            counters->handoff_ms[stage - 1] += p->handoff_ms[stage - 1];
        }
    }
    return 0;
}

typedef struct {
    replay_mailbox_runtime *mailbox;
    int stage;
} replay_mailbox_worker;

struct replay_mailbox_runtime {
    ds4_v100_replay *rt;
    pthread_mutex_t mu;
    pthread_cond_t stage_cv[DS4_V100_EXPECTED_GPUS];
    pthread_cond_t host_cv;
    pthread_t threads[DS4_V100_EXPECTED_GPUS];
    bool thread_created[DS4_V100_EXPECTED_GPUS];
    replay_mailbox_worker workers[DS4_V100_EXPECTED_GPUS];
    bool stop;
    bool active;
    bool failed;
    uint64_t generation;
    const uint32_t *tokens;
    const uint32_t *positions;
    uint32_t n_slots;
    uint64_t done_generation[DS4_V100_EXPECTED_GPUS][DS4_V100_SCHED_MAX_SLOTS];
    uint32_t done_count[DS4_V100_EXPECTED_GPUS];
    char err[512];
    ds4_v100_stage_scheduler_report reports[DS4_V100_EXPECTED_GPUS][DS4_V100_SCHED_MAX_SLOTS];
    double stage_decode_ms[DS4_V100_EXPECTED_GPUS];
    double handoff_ms[DS4_V100_EXPECTED_GPUS - 1];
    double worker_wait_ms[DS4_V100_EXPECTED_GPUS];
    double sync_ms[DS4_V100_EXPECTED_GPUS];
};

static void replay_mailbox_signal_all_locked(replay_mailbox_runtime *m) {
    if (!m) return;
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        pthread_cond_broadcast(&m->stage_cv[stage]);
    }
    pthread_cond_broadcast(&m->host_cv);
}

static void replay_mailbox_fail(replay_mailbox_runtime *m, const char *msg) {
    pthread_mutex_lock(&m->mu);
    if (!m->failed) {
        m->failed = true;
        m->active = false;
        snprintf(m->err, sizeof(m->err), "%s", msg ? msg : "async mailbox pipeline failed");
    }
    replay_mailbox_signal_all_locked(m);
    pthread_mutex_unlock(&m->mu);
}

static bool replay_mailbox_wait_generation(replay_mailbox_runtime *m,
                                           int stage,
                                           uint64_t *seen_generation) {
    pthread_mutex_lock(&m->mu);
    while (!m->stop && (!m->active || m->generation == *seen_generation)) {
        pthread_cond_wait(&m->stage_cv[stage], &m->mu);
    }
    if (m->stop) {
        pthread_mutex_unlock(&m->mu);
        return false;
    }
    *seen_generation = m->generation;
    pthread_mutex_unlock(&m->mu);
    return true;
}

static bool replay_mailbox_wait_prev(replay_mailbox_runtime *m,
                                     int stage,
                                     uint32_t slot,
                                     uint64_t generation) {
    if (stage == 0) return true;
    pthread_mutex_lock(&m->mu);
    while (!m->stop && m->active && m->generation == generation &&
           !m->failed &&
           m->done_generation[stage - 1][slot] != generation) {
        pthread_cond_wait(&m->stage_cv[stage], &m->mu);
    }
    const bool ok = !m->stop && m->active &&
        m->generation == generation && !m->failed;
    pthread_mutex_unlock(&m->mu);
    return ok;
}

static void replay_mailbox_mark_done(replay_mailbox_runtime *m,
                                     int stage,
                                     uint32_t slot,
                                     uint64_t generation) {
    pthread_mutex_lock(&m->mu);
    if (m->active && m->generation == generation && !m->failed) {
        m->done_generation[stage][slot] = generation;
        m->done_count[stage]++;
        if (stage + 1 < DS4_V100_EXPECTED_GPUS) {
            pthread_cond_signal(&m->stage_cv[stage + 1]);
        } else if (m->done_count[stage] >= m->n_slots) {
            m->active = false;
            pthread_cond_broadcast(&m->host_cv);
        }
    }
    pthread_mutex_unlock(&m->mu);
}

static void *replay_mailbox_worker_main(void *arg) {
    replay_mailbox_worker *w = (replay_mailbox_worker *)arg;
    if (!w || !w->mailbox) return NULL;
    replay_mailbox_runtime *m = w->mailbox;
    const int stage = w->stage;
    uint64_t seen_generation = 0;

    for (;;) {
        if (!replay_mailbox_wait_generation(m, stage, &seen_generation)) {
            break;
        }

        char local_err[512] = {0};
        for (uint32_t slot = 0; slot < m->n_slots; slot++) {
            const double wait0 = replay_now_ms();
            const bool prev_ready =
                replay_mailbox_wait_prev(m, stage, slot, seen_generation);
            m->worker_wait_ms[stage] += replay_now_ms() - wait0;
            if (!prev_ready) {
                break;
            }

            ds4_v100_stage_scheduler_report report;
            memset(&report, 0, sizeof(report));
            double t0 = replay_now_ms();
            if (stage == 0) {
                if (ds4_v100_stage_scheduler_decode_token_slot_span(
                        m->rt->scheds[0],
                        slot,
                        &m->tokens[slot],
                        &m->positions[slot],
                        1,
                        &report,
                        local_err,
                        sizeof(local_err))) {
                    replay_mailbox_fail(m, local_err);
                    break;
                }
            } else {
                if (replay_handoff_slot_span(m->rt,
                                             stage,
                                             slot,
                                             1,
                                             local_err,
                                             sizeof(local_err))) {
                    replay_mailbox_fail(m, local_err);
                    break;
                }
                m->handoff_ms[stage - 1] += replay_now_ms() - t0;
                t0 = replay_now_ms();
                if (ds4_v100_stage_scheduler_decode_hc_slot_span(
                        m->rt->scheds[stage],
                        slot,
                        &m->tokens[slot],
                        &m->positions[slot],
                        1,
                        &report,
                        local_err,
                        sizeof(local_err))) {
                    replay_mailbox_fail(m, local_err);
                    break;
                }
            }
            const double sync0 = replay_now_ms();
            if (!ds4_gpu_set_device(stage) || !ds4_gpu_synchronize()) {
                replay_mailbox_fail(m, "async mailbox pipeline synchronize failed");
                break;
            }
            m->sync_ms[stage] += replay_now_ms() - sync0;
            m->stage_decode_ms[stage] += replay_now_ms() - t0;
            m->reports[stage][slot] = report;
            replay_mailbox_mark_done(m, stage, slot, seen_generation);
        }
    }
    return NULL;
}

static int replay_mailbox_runtime_open(ds4_v100_replay *rt, char *err, size_t errlen) {
    if (!rt) return replay_error(err, errlen, "missing async mailbox runtime input");
    if (rt->mailbox) return 0;

    replay_mailbox_runtime *m =
        (replay_mailbox_runtime *)calloc(1, sizeof(*m));
    if (!m) return replay_error(err, errlen, "failed to allocate async mailbox runtime");
    m->rt = rt;
    pthread_mutex_init(&m->mu, NULL);
    pthread_cond_init(&m->host_cv, NULL);
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        pthread_cond_init(&m->stage_cv[stage], NULL);
    }

    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        m->workers[stage].mailbox = m;
        m->workers[stage].stage = stage;
        if (pthread_create(&m->threads[stage],
                           NULL,
                           replay_mailbox_worker_main,
                           &m->workers[stage]) != 0) {
            if (err && errlen) {
                snprintf(err, errlen, "failed to create mailbox async pipeline worker %d", stage);
            }
            replay_mailbox_runtime_close(m);
            return 1;
        }
        m->thread_created[stage] = true;
    }
    rt->mailbox = m;
    return 0;
}

static void replay_mailbox_runtime_close(replay_mailbox_runtime *m) {
    if (!m) return;
    pthread_mutex_lock(&m->mu);
    m->stop = true;
    replay_mailbox_signal_all_locked(m);
    pthread_mutex_unlock(&m->mu);

    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        if (m->thread_created[stage]) {
            (void)pthread_join(m->threads[stage], NULL);
        }
    }
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        pthread_cond_destroy(&m->stage_cv[stage]);
    }
    pthread_cond_destroy(&m->host_cv);
    pthread_mutex_destroy(&m->mu);
    free(m);
}

static int replay_mailbox_runtime_dispatch(replay_mailbox_runtime *m,
                                           const uint32_t *tokens,
                                           const uint32_t *positions,
                                           uint32_t n_slots,
                                           ds4_v100_replay_counters *counters,
                                           double *bucket_ms,
                                           char *err,
                                           size_t errlen) {
    if (!m || !tokens || !positions || n_slots == 0 ||
        n_slots > DS4_V100_SCHED_MAX_SLOTS) {
        return replay_error(err, errlen, "missing async mailbox dispatch input");
    }

    const double total0 = replay_now_ms();
    pthread_mutex_lock(&m->mu);
    if (m->active) {
        pthread_mutex_unlock(&m->mu);
        return replay_error(err, errlen, "async mailbox dispatch while active");
    }
    memset(m->done_generation, 0, sizeof(m->done_generation));
    memset(m->done_count, 0, sizeof(m->done_count));
    memset(m->reports, 0, sizeof(m->reports));
    memset(m->stage_decode_ms, 0, sizeof(m->stage_decode_ms));
    memset(m->handoff_ms, 0, sizeof(m->handoff_ms));
    memset(m->worker_wait_ms, 0, sizeof(m->worker_wait_ms));
    memset(m->sync_ms, 0, sizeof(m->sync_ms));
    m->err[0] = '\0';
    m->failed = false;
    m->tokens = tokens;
    m->positions = positions;
    m->n_slots = n_slots;
    m->generation++;
    m->active = true;
    pthread_cond_signal(&m->stage_cv[0]);
    const uint64_t generation = m->generation;
    const double setup_ms = replay_now_ms() - total0;
    const double wait0 = replay_now_ms();
    while (m->active && m->generation == generation && !m->failed) {
        pthread_cond_wait(&m->host_cv, &m->mu);
    }
    const double host_wait_ms = replay_now_ms() - wait0;
    const bool failed = m->failed;
    char local_err[512] = {0};
    if (failed) snprintf(local_err, sizeof(local_err), "%s", m->err);
    pthread_mutex_unlock(&m->mu);

    if (failed) {
        if (err && errlen) {
            snprintf(err,
                     errlen,
                     "%s",
                     local_err[0] ? local_err : "async mailbox pipeline failed");
        }
        return 1;
    }

    const double complete0 = replay_now_ms();
    if (replay_sync_all_stages(err, errlen)) return 1;
    const double complete_ms = replay_now_ms() - complete0;
    const double total_ms = replay_now_ms() - total0;
    if (bucket_ms) *bucket_ms += total_ms;
    if (counters) {
        counters->async_pipeline_dispatches++;
        counters->async_pipeline_total_ms += total_ms;
        counters->async_pipeline_setup_ms += setup_ms;
        counters->async_pipeline_host_wait_ms += host_wait_ms;
        counters->async_pipeline_complete_ms += complete_ms;
        for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
            counters->stage_decode_ms[stage] += m->stage_decode_ms[stage];
            counters->async_pipeline_worker_wait_ms[stage] += m->worker_wait_ms[stage];
            counters->async_pipeline_sync_ms[stage] += m->sync_ms[stage];
            counters_add_report(counters, stage, &m->reports[stage][0]);
        }
        for (int stage = 1; stage < DS4_V100_EXPECTED_GPUS; stage++) {
            counters->handoff_ms[stage - 1] += m->handoff_ms[stage - 1];
        }
    }
    return 0;
}

typedef struct {
    ds4_v100_replay *rt;
    const uint32_t *tokens;
    const uint32_t *positions;
    uint32_t n_slots;
    bool event_handoff;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    bool done[DS4_V100_EXPECTED_GPUS][DS4_V100_SCHED_MAX_SLOTS];
    bool failed;
    char err[512];
    ds4_v100_stage_scheduler_report reports[DS4_V100_EXPECTED_GPUS][DS4_V100_SCHED_MAX_SLOTS];
    double stage_decode_ms[DS4_V100_EXPECTED_GPUS];
    double handoff_ms[DS4_V100_EXPECTED_GPUS - 1];
    double worker_wait_ms[DS4_V100_EXPECTED_GPUS];
    double sync_ms[DS4_V100_EXPECTED_GPUS];
} replay_step_pipeline_batch;

typedef struct {
    replay_step_pipeline_batch *batch;
    int stage;
} replay_step_pipeline_worker;

static void replay_step_pipeline_fail(replay_step_pipeline_batch *b, const char *msg) {
    pthread_mutex_lock(&b->mu);
    if (!b->failed) {
        b->failed = true;
        snprintf(b->err, sizeof(b->err), "%s", msg ? msg : "async per-step pipeline failed");
    }
    pthread_cond_broadcast(&b->cv);
    pthread_mutex_unlock(&b->mu);
}

static bool replay_step_pipeline_wait_prev(replay_step_pipeline_batch *b,
                                           int stage,
                                           uint32_t slot) {
    if (stage == 0) return true;
    pthread_mutex_lock(&b->mu);
    while (!b->failed && !b->done[stage - 1][slot]) {
        pthread_cond_wait(&b->cv, &b->mu);
    }
    const bool ok = !b->failed;
    pthread_mutex_unlock(&b->mu);
    return ok;
}

static void replay_step_pipeline_mark_done(replay_step_pipeline_batch *b,
                                           int stage,
                                           uint32_t slot) {
    pthread_mutex_lock(&b->mu);
    b->done[stage][slot] = true;
    pthread_cond_broadcast(&b->cv);
    pthread_mutex_unlock(&b->mu);
}

static uint32_t replay_async_slot_chunk(const replay_step_pipeline_batch *b) {
    if (!b) return 1;
    const char *env = getenv("DS4_V100_ASYNC_SLOT_CHUNK");
    if (!env || !env[0]) env = getenv("DS4_ASYNC_SLOT_CHUNK");
    if (!env || !env[0]) return 1;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || *end != '\0' || v == 0) return 1;
    if (v > DS4_V100_SCHED_MAX_SLOTS) v = DS4_V100_SCHED_MAX_SLOTS;
    return (uint32_t)v;
}

static uint32_t replay_env_u32_clamped(const char *name,
                                       uint32_t default_value,
                                       uint32_t max_value) {
    const char *env = name ? getenv(name) : NULL;
    if (!env || !env[0]) return default_value;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || *end != '\0') return default_value;
    if (v > max_value) v = max_value;
    return (uint32_t)v;
}

static uint32_t replay_async_ready_chunk_max(void) {
    return replay_env_u32_clamped("DS4_V100_ASYNC_READY_CHUNK_MAX",
                                  0,
                                  DS4_V100_SCHED_MAX_SLOTS);
}

static uint32_t replay_async_ready_stage0_chunk_max(uint32_t ready_chunk_max) {
    uint32_t v = replay_env_u32_clamped("DS4_V100_ASYNC_READY_STAGE0_CHUNK_MAX",
                                        1,
                                        DS4_V100_SCHED_MAX_SLOTS);
    if (ready_chunk_max && v > ready_chunk_max) v = ready_chunk_max;
    return v ? v : 1u;
}

static uint32_t replay_async_ready_wait_us(void) {
    return replay_env_u32_clamped("DS4_V100_ASYNC_READY_WAIT_US", 0, 1000000u);
}

static bool replay_env_enabled(const char *name) {
    const char *env = name ? getenv(name) : NULL;
    if (!env || !env[0]) return false;
    if (!strcmp(env, "0") ||
        !strcmp(env, "false") ||
        !strcmp(env, "False") ||
        !strcmp(env, "off") ||
        !strcmp(env, "Off") ||
        !strcmp(env, "no") ||
        !strcmp(env, "No")) {
        return false;
    }
    return true;
}

static bool replay_async_layer_wavefront_enabled(void) {
    return replay_env_enabled("DS4_V100_ASYNC_LAYER_WAVEFRONT");
}

static bool replay_async_ffn_wavefront_enabled(void) {
    return replay_env_enabled("DS4_V100_ASYNC_FFN_WAVEFRONT");
}

static bool replay_async_ffn_wavefront_verbose(void) {
    return replay_env_enabled("DS4_V100_ASYNC_FFN_WAVEFRONT_VERBOSE");
}

static uint32_t replay_async_layer_wavefront_chunk(void) {
    uint32_t v = replay_env_u32_clamped("DS4_V100_ASYNC_LAYER_WAVEFRONT_CHUNK",
                                        2,
                                        DS4_V100_SCHED_MAX_SLOTS);
    return v ? v : 1u;
}

static uint32_t replay_async_ffn_wavefront_chunk(void) {
    uint32_t v = replay_env_u32_clamped("DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK",
                                        2,
                                        DS4_V100_SCHED_MAX_SLOTS);
    return v ? v : 1u;
}

static void replay_ready_deadline_from_now(uint32_t wait_us, struct timespec *out) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t nsec = (uint64_t)tv.tv_usec * 1000ull + (uint64_t)wait_us * 1000ull;
    out->tv_sec = tv.tv_sec + (time_t)(nsec / 1000000000ull);
    out->tv_nsec = (long)(nsec % 1000000000ull);
}

static uint32_t replay_step_pipeline_collect_ready_chunk(replay_step_pipeline_batch *b,
                                                        int stage,
                                                        uint32_t slot,
                                                        uint32_t max_chunk,
                                                        uint32_t wait_us) {
    if (!b || max_chunk <= 1u || slot >= b->n_slots) return 1;
    uint32_t remaining = b->n_slots - slot;
    if (max_chunk > remaining) max_chunk = remaining;
    if (stage == 0) {
        const uint32_t stage0_max = replay_async_ready_stage0_chunk_max(max_chunk);
        return stage0_max < max_chunk ? stage0_max : max_chunk;
    }

    struct timespec deadline;
    if (wait_us) replay_ready_deadline_from_now(wait_us, &deadline);
    uint32_t chunk = 1;
    while (chunk < max_chunk) {
        const uint32_t next_slot = slot + chunk;
        pthread_mutex_lock(&b->mu);
        while (!b->failed && !b->done[stage - 1][next_slot] && wait_us) {
            const int wait_rc = pthread_cond_timedwait(&b->cv, &b->mu, &deadline);
            if (wait_rc == ETIMEDOUT) break;
        }
        const bool ready = !b->failed && b->done[stage - 1][next_slot];
        pthread_mutex_unlock(&b->mu);
        if (!ready) break;
        chunk++;
    }
    return chunk;
}

static bool replay_stage_layer_range(int stage, int *first_layer, int *last_layer) {
    if (!first_layer || !last_layer) return false;
    int first = -1;
    int last = -1;
    for (int layer = 0; layer < DS4_V100_N_LAYERS; layer++) {
        if (ds4_v100_stage_for_layer(layer) != stage) continue;
        if (first < 0) first = layer;
        last = layer;
    }
    if (first < 0 || last < first) return false;
    *first_layer = first;
    *last_layer = last;
    return true;
}

static bool replay_step_pipeline_prev_done_now(replay_step_pipeline_batch *b,
                                               int stage,
                                               uint32_t slot) {
    if (!b || slot >= b->n_slots) return false;
    if (stage == 0) return true;
    pthread_mutex_lock(&b->mu);
    const bool ready = !b->failed && b->done[stage - 1][slot];
    pthread_mutex_unlock(&b->mu);
    return ready;
}

static void replay_step_pipeline_report_accum(
    ds4_v100_stage_scheduler_report *dst,
    const ds4_v100_stage_scheduler_report *src) {
    if (!dst || !src) return;
    if (dst->layers_executed == 0) {
        *dst = *src;
        return;
    }
    dst->last_layer = src->last_layer;
    dst->layers_executed += src->layers_executed;
    dst->last_layer_report = src->last_layer_report;
    dst->turbomind_routed_layers_executed += src->turbomind_routed_layers_executed;
    dst->turbomind_tp2_routed_layers_executed += src->turbomind_tp2_routed_layers_executed;
    dst->timing_tp2_copy_in_ms += src->timing_tp2_copy_in_ms;
    dst->timing_tp2_owner_ms += src->timing_tp2_owner_ms;
    dst->timing_tp2_peer_ms += src->timing_tp2_peer_ms;
    dst->timing_tp2_copy_out_ms += src->timing_tp2_copy_out_ms;
    dst->timing_tp2_reduce_ms += src->timing_tp2_reduce_ms;
    dst->timing_tp2_total_ms += src->timing_tp2_total_ms;
    dst->timing_hc_attn_ms += src->timing_hc_attn_ms;
    dst->timing_attention_ms += src->timing_attention_ms;
    dst->timing_attn_proj_ms += src->timing_attn_proj_ms;
    dst->timing_attn_cache_ms += src->timing_attn_cache_ms;
    dst->timing_attn_softmax_ms += src->timing_attn_softmax_ms;
    dst->timing_attn_inverse_rope_ms += src->timing_attn_inverse_rope_ms;
    dst->timing_attn_output_ms += src->timing_attn_output_ms;
    dst->timing_hc_ffn_ms += src->timing_hc_ffn_ms;
    dst->timing_ffn_ms += src->timing_ffn_ms;
    dst->timing_hc_final_ms += src->timing_hc_final_ms;
    dst->timing_total_ms += src->timing_total_ms;
}

static bool replay_step_pipeline_find_layer_group(const int *next_layer,
                                                  const bool *complete,
                                                  uint32_t n_slots,
                                                  int first_layer,
                                                  int last_layer,
                                                  uint32_t max_chunk,
                                                  uint32_t min_chunk,
                                                  uint32_t *out_slot,
                                                  uint32_t *out_chunk,
                                                  int *out_layer) {
    if (!next_layer || !complete || !out_slot || !out_chunk || !out_layer) return false;
    if (max_chunk == 0) max_chunk = 1;
    if (min_chunk == 0) min_chunk = 1;
    for (int layer = first_layer + 1; layer <= last_layer; layer++) {
        for (uint32_t slot = 0; slot < n_slots;) {
            if (complete[slot] || next_layer[slot] != layer) {
                slot++;
                continue;
            }
            const uint32_t start = slot;
            uint32_t chunk = 0;
            while (slot < n_slots &&
                   !complete[slot] &&
                   next_layer[slot] == layer &&
                   chunk < max_chunk) {
                slot++;
                chunk++;
            }
            if (chunk >= min_chunk) {
                *out_slot = start;
                *out_chunk = chunk;
                *out_layer = layer;
                return true;
            }
        }
    }
    return false;
}

static bool replay_step_pipeline_find_startable_slot(replay_step_pipeline_batch *b,
                                                     int stage,
                                                     const bool *started,
                                                     const bool *complete,
                                                     uint32_t *out_slot) {
    if (!b || !started || !complete || !out_slot) return false;
    for (uint32_t slot = 0; slot < b->n_slots; slot++) {
        if (started[slot] || complete[slot]) continue;
        if (replay_step_pipeline_prev_done_now(b, stage, slot)) {
            *out_slot = slot;
            return true;
        }
    }
    return false;
}

static int replay_step_pipeline_execute_layer_span(replay_step_pipeline_batch *b,
                                                   int stage,
                                                   uint32_t slot,
                                                   uint32_t chunk,
                                                   int layer,
                                                   int last_layer,
                                                   bool start_layer,
                                                   int *next_layer,
                                                   bool *complete,
                                                   uint32_t *completed,
                                                   char *local_err,
                                                   size_t local_errlen) {
    if (!b || !next_layer || !complete || !completed || chunk == 0) {
        snprintf(local_err, local_errlen, "missing layer-wavefront execution input");
        return 1;
    }
    ds4_v100_stage_scheduler_report reports[DS4_V100_SCHED_MAX_SLOTS];
    memset(reports, 0, sizeof(reports));

    double handoff0 = replay_now_ms();
    if (start_layer && stage > 0) {
        for (uint32_t rel = 0; rel < chunk; rel++) {
            const uint32_t s = slot + rel;
            const ds4_gpu_event *prev_event = b->event_handoff
                ? b->rt->stage_ready[stage - 1][s]
                : NULL;
            if (b->event_handoff && !prev_event) {
                snprintf(local_err, local_errlen, "missing layer-wavefront source event");
                return 1;
            }
            if (replay_handoff_slot_span_after_event(b->rt,
                                                     stage,
                                                     s,
                                                     1,
                                                     prev_event,
                                                     local_err,
                                                     local_errlen)) {
                return 1;
            }
        }
        b->handoff_ms[stage - 1] += replay_now_ms() - handoff0;
    }

    const double decode0 = replay_now_ms();
    int rc = 0;
    if (start_layer && stage == 0) {
        rc = ds4_v100_stage_scheduler_decode_token_layer_span(
            b->rt->scheds[0],
            slot,
            &b->tokens[slot],
            &b->positions[slot],
            chunk,
            layer,
            layer,
            reports,
            local_err,
            local_errlen);
    } else {
        rc = ds4_v100_stage_scheduler_decode_hc_layer_span(
            b->rt->scheds[stage],
            slot,
            &b->tokens[slot],
            &b->positions[slot],
            chunk,
            layer,
            layer,
            reports,
            local_err,
            local_errlen);
    }
    if (rc) return 1;
    b->stage_decode_ms[stage] += replay_now_ms() - decode0;

    bool completed_now = false;
    for (uint32_t rel = 0; rel < chunk; rel++) {
        const uint32_t s = slot + rel;
        replay_step_pipeline_report_accum(&b->reports[stage][s], &reports[rel]);
        next_layer[s] = layer + 1;
        if (layer >= last_layer && !complete[s]) {
            complete[s] = true;
            completed_now = true;
            (*completed)++;
        }
    }
    if (!completed_now) return 0;

    if (b->event_handoff) {
        for (uint32_t rel = 0; rel < chunk; rel++) {
            const uint32_t s = slot + rel;
            if (layer < last_layer) continue;
            ds4_gpu_event *ready_event = b->rt->stage_ready[stage][s];
            if (!ready_event || !ds4_gpu_event_record(ready_event)) {
                snprintf(local_err,
                         local_errlen,
                         "layer-wavefront event record failed");
                return 1;
            }
        }
    } else {
        const double sync0 = replay_now_ms();
        if (!ds4_gpu_set_device(stage) || !ds4_gpu_synchronize()) {
            snprintf(local_err, local_errlen, "layer-wavefront synchronize failed");
            return 1;
        }
        b->sync_ms[stage] += replay_now_ms() - sync0;
    }
    for (uint32_t rel = 0; rel < chunk; rel++) {
        const uint32_t s = slot + rel;
        if (layer >= last_layer) replay_step_pipeline_mark_done(b, stage, s);
    }
    return 0;
}

static void *replay_step_pipeline_worker_layer_wavefront(void *arg) {
    replay_step_pipeline_worker *w = (replay_step_pipeline_worker *)arg;
    if (!w || !w->batch) return NULL;
    replay_step_pipeline_batch *b = w->batch;
    const int stage = w->stage;
    char local_err[512] = {0};
    int first_layer = -1;
    int last_layer = -1;
    if (!replay_stage_layer_range(stage, &first_layer, &last_layer)) {
        replay_step_pipeline_fail(b, "failed to resolve layer-wavefront stage range");
        return NULL;
    }

    bool started[DS4_V100_SCHED_MAX_SLOTS] = {0};
    bool complete[DS4_V100_SCHED_MAX_SLOTS] = {0};
    int next_layer[DS4_V100_SCHED_MAX_SLOTS] = {0};
    uint32_t completed = 0;
    const uint32_t max_chunk = replay_async_layer_wavefront_chunk();

    while (completed < b->n_slots) {
        if (b->failed) break;
        uint32_t slot = 0;
        uint32_t chunk = 0;
        int layer = -1;
        bool start_layer = false;

        if (replay_step_pipeline_find_layer_group(next_layer,
                                                  complete,
                                                  b->n_slots,
                                                  first_layer,
                                                  last_layer,
                                                  max_chunk,
                                                  max_chunk,
                                                  &slot,
                                                  &chunk,
                                                  &layer)) {
            start_layer = false;
        } else if (replay_step_pipeline_find_startable_slot(b,
                                                           stage,
                                                           started,
                                                           complete,
                                                           &slot)) {
            chunk = 1;
            layer = first_layer;
            start_layer = true;
        } else if (replay_step_pipeline_find_layer_group(next_layer,
                                                        complete,
                                                        b->n_slots,
                                                        first_layer,
                                                        last_layer,
                                                        max_chunk,
                                                        2,
                                                        &slot,
                                                        &chunk,
                                                        &layer) ||
                   replay_step_pipeline_find_layer_group(next_layer,
                                                        complete,
                                                        b->n_slots,
                                                        first_layer,
                                                        last_layer,
                                                        1,
                                                        1,
                                                        &slot,
                                                        &chunk,
                                                        &layer)) {
            start_layer = false;
        } else {
            uint32_t wait_slot = UINT32_MAX;
            for (uint32_t s = 0; s < b->n_slots; s++) {
                if (!started[s] && !complete[s]) {
                    wait_slot = s;
                    break;
                }
            }
            if (wait_slot == UINT32_MAX) {
                replay_step_pipeline_fail(b, "layer-wavefront made no progress");
                break;
            }
            const double wait0 = replay_now_ms();
            if (!replay_step_pipeline_wait_prev(b, stage, wait_slot)) {
                break;
            }
            b->worker_wait_ms[stage] += replay_now_ms() - wait0;
            slot = wait_slot;
            chunk = 1;
            layer = first_layer;
            start_layer = true;
        }

        if (start_layer) {
            for (uint32_t rel = 0; rel < chunk; rel++) {
                started[slot + rel] = true;
                next_layer[slot + rel] = first_layer;
            }
        }
        if (replay_step_pipeline_execute_layer_span(b,
                                                    stage,
                                                    slot,
                                                    chunk,
                                                    layer,
                                                    last_layer,
                                                    start_layer,
                                                    next_layer,
                                                    complete,
                                                    &completed,
                                                    local_err,
                                                    sizeof(local_err))) {
            replay_step_pipeline_fail(b, local_err);
            break;
        }
    }
    return NULL;
}

static int replay_step_pipeline_execute_ffn_wavefront_layer(
                                                   replay_step_pipeline_batch *b,
                                                   int stage,
                                                   uint32_t slot,
                                                   uint32_t chunk,
                                                   int layer,
                                                   int last_layer,
                                                   bool start_layer,
                                                   int *next_layer,
                                                   bool *complete,
                                                   uint32_t *completed,
                                                   char *local_err,
                                                   size_t local_errlen) {
    if (!b || !next_layer || !complete || !completed || chunk == 0) {
        snprintf(local_err, local_errlen, "missing FFN-wavefront execution input");
        return 1;
    }
    ds4_v100_stage_scheduler_report reports[DS4_V100_SCHED_MAX_SLOTS];
    memset(reports, 0, sizeof(reports));

    double handoff0 = replay_now_ms();
    if (start_layer && stage > 0) {
        for (uint32_t rel = 0; rel < chunk; rel++) {
            const uint32_t s = slot + rel;
            const ds4_gpu_event *prev_event = b->event_handoff
                ? b->rt->stage_ready[stage - 1][s]
                : NULL;
            if (b->event_handoff && !prev_event) {
                snprintf(local_err, local_errlen, "missing FFN-wavefront source event");
                return 1;
            }
            if (replay_handoff_slot_span_after_event(b->rt,
                                                     stage,
                                                     s,
                                                     1,
                                                     prev_event,
                                                     local_err,
                                                     local_errlen)) {
                return 1;
            }
        }
        b->handoff_ms[stage - 1] += replay_now_ms() - handoff0;
    }

    const double decode0 = replay_now_ms();
    int rc = 0;
    if (start_layer && stage == 0) {
        rc = ds4_v100_stage_scheduler_decode_token_layer_span(
            b->rt->scheds[0],
            slot,
            &b->tokens[slot],
            &b->positions[slot],
            chunk,
            layer,
            layer,
            reports,
            local_err,
            local_errlen);
    } else {
        rc = ds4_v100_stage_scheduler_decode_hc_ffn_microbatch_layer(
            b->rt->scheds[stage],
            slot,
            &b->tokens[slot],
            &b->positions[slot],
            chunk,
            layer,
            reports,
            local_err,
            local_errlen);
    }
    if (rc) return 1;
    b->stage_decode_ms[stage] += replay_now_ms() - decode0;
    if (replay_async_ffn_wavefront_verbose()) {
        uint32_t route_total = 0;
        uint32_t tm_slots = 0;
        for (uint32_t rel = 0; rel < chunk; rel++) {
            route_total += reports[rel].last_layer_report.routes;
            if (reports[rel].last_layer_report.turbomind_routed) tm_slots++;
        }
        fprintf(stderr,
                "ds4: FFN wavefront stage=%d layer=%d start_slot=%u slots=%u "
                "routes=%u tm_slots=%u start_layer=%u\n",
                stage,
                layer,
                slot,
                chunk,
                route_total,
                tm_slots,
                start_layer ? 1u : 0u);
    }

    bool completed_now = false;
    for (uint32_t rel = 0; rel < chunk; rel++) {
        const uint32_t s = slot + rel;
        replay_step_pipeline_report_accum(&b->reports[stage][s], &reports[rel]);
        next_layer[s] = layer + 1;
        if (layer >= last_layer && !complete[s]) {
            complete[s] = true;
            completed_now = true;
            (*completed)++;
        }
    }
    if (!completed_now) return 0;

    if (b->event_handoff) {
        for (uint32_t rel = 0; rel < chunk; rel++) {
            const uint32_t s = slot + rel;
            if (layer < last_layer) continue;
            ds4_gpu_event *ready_event = b->rt->stage_ready[stage][s];
            if (!ready_event || !ds4_gpu_event_record(ready_event)) {
                snprintf(local_err,
                         local_errlen,
                         "FFN-wavefront event record failed");
                return 1;
            }
        }
    } else {
        const double sync0 = replay_now_ms();
        if (!ds4_gpu_set_device(stage) || !ds4_gpu_synchronize()) {
            snprintf(local_err, local_errlen, "FFN-wavefront synchronize failed");
            return 1;
        }
        b->sync_ms[stage] += replay_now_ms() - sync0;
    }
    for (uint32_t rel = 0; rel < chunk; rel++) {
        const uint32_t s = slot + rel;
        if (layer >= last_layer) replay_step_pipeline_mark_done(b, stage, s);
    }
    return 0;
}

static void *replay_step_pipeline_worker_ffn_wavefront(void *arg) {
    replay_step_pipeline_worker *w = (replay_step_pipeline_worker *)arg;
    if (!w || !w->batch) return NULL;
    replay_step_pipeline_batch *b = w->batch;
    const int stage = w->stage;
    char local_err[512] = {0};
    int first_layer = -1;
    int last_layer = -1;
    if (!replay_stage_layer_range(stage, &first_layer, &last_layer)) {
        replay_step_pipeline_fail(b, "failed to resolve FFN-wavefront stage range");
        return NULL;
    }

    bool started[DS4_V100_SCHED_MAX_SLOTS] = {0};
    bool complete[DS4_V100_SCHED_MAX_SLOTS] = {0};
    int next_layer[DS4_V100_SCHED_MAX_SLOTS] = {0};
    uint32_t completed = 0;
    const uint32_t max_chunk = replay_async_ffn_wavefront_chunk();

    while (completed < b->n_slots) {
        if (b->failed) break;
        uint32_t slot = 0;
        uint32_t chunk = 0;
        int layer = -1;
        bool start_layer = false;

        if (replay_step_pipeline_find_layer_group(next_layer,
                                                  complete,
                                                  b->n_slots,
                                                  first_layer,
                                                  last_layer,
                                                  max_chunk,
                                                  max_chunk,
                                                  &slot,
                                                  &chunk,
                                                  &layer)) {
            start_layer = false;
        } else if (replay_step_pipeline_find_startable_slot(b,
                                                           stage,
                                                           started,
                                                           complete,
                                                           &slot)) {
            chunk = 1;
            layer = first_layer;
            start_layer = true;
        } else if (replay_step_pipeline_find_layer_group(next_layer,
                                                        complete,
                                                        b->n_slots,
                                                        first_layer,
                                                        last_layer,
                                                        max_chunk,
                                                        2,
                                                        &slot,
                                                        &chunk,
                                                        &layer) ||
                   replay_step_pipeline_find_layer_group(next_layer,
                                                        complete,
                                                        b->n_slots,
                                                        first_layer,
                                                        last_layer,
                                                        1,
                                                        1,
                                                        &slot,
                                                        &chunk,
                                                        &layer)) {
            start_layer = false;
        } else {
            uint32_t wait_slot = UINT32_MAX;
            for (uint32_t s = 0; s < b->n_slots; s++) {
                if (!started[s] && !complete[s]) {
                    wait_slot = s;
                    break;
                }
            }
            if (wait_slot == UINT32_MAX) {
                replay_step_pipeline_fail(b, "FFN-wavefront made no progress");
                break;
            }
            const double wait0 = replay_now_ms();
            if (!replay_step_pipeline_wait_prev(b, stage, wait_slot)) {
                break;
            }
            b->worker_wait_ms[stage] += replay_now_ms() - wait0;
            slot = wait_slot;
            chunk = 1;
            layer = first_layer;
            start_layer = true;
        }

        if (start_layer) {
            for (uint32_t rel = 0; rel < chunk; rel++) {
                started[slot + rel] = true;
                next_layer[slot + rel] = first_layer;
            }
        }
        if (replay_step_pipeline_execute_ffn_wavefront_layer(b,
                                                             stage,
                                                             slot,
                                                             chunk,
                                                             layer,
                                                             last_layer,
                                                             start_layer,
                                                             next_layer,
                                                             complete,
                                                             &completed,
                                                             local_err,
                                                             sizeof(local_err))) {
            replay_step_pipeline_fail(b, local_err);
            break;
        }
    }
    return NULL;
}

static void *replay_step_pipeline_worker_main(void *arg) {
    if (replay_async_ffn_wavefront_enabled()) {
        return replay_step_pipeline_worker_ffn_wavefront(arg);
    }
    if (replay_async_layer_wavefront_enabled()) {
        return replay_step_pipeline_worker_layer_wavefront(arg);
    }
    replay_step_pipeline_worker *w = (replay_step_pipeline_worker *)arg;
    if (!w || !w->batch) return NULL;
    replay_step_pipeline_batch *b = w->batch;
    const int stage = w->stage;
    char local_err[512] = {0};
    const uint32_t slot_chunk = replay_async_slot_chunk(b);
    const uint32_t ready_chunk_max = replay_async_ready_chunk_max();
    const uint32_t ready_wait_us = replay_async_ready_wait_us();

    for (uint32_t slot = 0; slot < b->n_slots;) {
        uint32_t chunk = 1;
        const double wait0 = replay_now_ms();
        bool prev_ready = true;
        if (ready_chunk_max) {
            if (!replay_step_pipeline_wait_prev(b, stage, slot)) {
                prev_ready = false;
            } else {
                chunk = replay_step_pipeline_collect_ready_chunk(b,
                                                                 stage,
                                                                 slot,
                                                                 ready_chunk_max,
                                                                 ready_wait_us);
            }
        } else {
            chunk = slot_chunk;
            if (chunk > b->n_slots - slot) chunk = b->n_slots - slot;
            for (uint32_t rel = 0; rel < chunk; rel++) {
                if (!replay_step_pipeline_wait_prev(b, stage, slot + rel)) {
                    prev_ready = false;
                    break;
                }
            }
        }
        b->worker_wait_ms[stage] += replay_now_ms() - wait0;
        if (!prev_ready) break;

        memset(&b->reports[stage][slot], 0, chunk * sizeof(b->reports[stage][slot]));
        double t0 = replay_now_ms();
        if (stage == 0) {
            if (ds4_v100_stage_scheduler_decode_token_slot_span(
                    b->rt->scheds[0],
                    slot,
                    &b->tokens[slot],
                    &b->positions[slot],
                    chunk,
                    &b->reports[stage][slot],
                    local_err,
                    sizeof(local_err))) {
                replay_step_pipeline_fail(b, local_err);
                break;
            }
        } else {
            const ds4_gpu_event *prev_event = b->event_handoff
                ? b->rt->stage_ready[stage - 1][slot]
                : NULL;
            if (b->event_handoff && !prev_event) {
                replay_step_pipeline_fail(b, "missing async per-step source event");
                break;
            }
            if (replay_handoff_slot_span_after_event(b->rt,
                                                     stage,
                                                     slot,
                                                     chunk,
                                                     prev_event,
                                                     local_err,
                                                     sizeof(local_err))) {
                replay_step_pipeline_fail(b, local_err);
                break;
            }
            b->handoff_ms[stage - 1] += replay_now_ms() - t0;
            t0 = replay_now_ms();
            if (ds4_v100_stage_scheduler_decode_hc_slot_span(
                    b->rt->scheds[stage],
                    slot,
                    &b->tokens[slot],
                    &b->positions[slot],
                    chunk,
                    &b->reports[stage][slot],
                    local_err,
                    sizeof(local_err))) {
                replay_step_pipeline_fail(b, local_err);
                break;
            }
        }
        if (b->event_handoff) {
            for (uint32_t rel = 0; rel < chunk; rel++) {
                ds4_gpu_event *ready_event = b->rt->stage_ready[stage][slot + rel];
                if (!ready_event || !ds4_gpu_event_record(ready_event)) {
                    replay_step_pipeline_fail(b, "async per-step pipeline event record failed");
                    break;
                }
            }
            if (b->failed) break;
        } else {
            const double sync0 = replay_now_ms();
            if (!ds4_gpu_set_device(stage) || !ds4_gpu_synchronize()) {
                replay_step_pipeline_fail(b, "async per-step pipeline synchronize failed");
                break;
            }
            b->sync_ms[stage] += replay_now_ms() - sync0;
        }
        b->stage_decode_ms[stage] += replay_now_ms() - t0;
        for (uint32_t rel = 0; rel < chunk; rel++) {
            replay_step_pipeline_mark_done(b, stage, slot + rel);
        }
        slot += chunk;
    }
    return NULL;
}

static int replay_feed_token_batch_async_per_step(ds4_v100_replay *rt,
                                                  const uint32_t *tokens,
                                                  const uint32_t *positions,
                                                  uint32_t n_slots,
                                                  ds4_v100_replay_counters *counters,
                                                  double *bucket_ms,
                                                  char *err,
                                                  size_t errlen) {
    if (!rt || !tokens || !positions || n_slots == 0 ||
        n_slots > DS4_V100_SCHED_MAX_SLOTS) {
        return replay_error(err, errlen, "missing V100 replay async per-step input");
    }
    if (n_slots == 1) {
        return replay_feed_token_batch(rt, tokens, positions, n_slots, counters, bucket_ms, err, errlen);
    }

    const double total0 = replay_now_ms();
    replay_step_pipeline_batch batch;
    memset(&batch, 0, sizeof(batch));
    batch.rt = rt;
    batch.tokens = tokens;
    batch.positions = positions;
    batch.n_slots = n_slots;
    batch.event_handoff = rt->opts.async_event_handoff;
    pthread_mutex_init(&batch.mu, NULL);
    pthread_cond_init(&batch.cv, NULL);

    pthread_t threads[DS4_V100_EXPECTED_GPUS];
    replay_step_pipeline_worker workers[DS4_V100_EXPECTED_GPUS];
    memset(threads, 0, sizeof(threads));
    memset(workers, 0, sizeof(workers));

    int created = 0;
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        workers[stage].batch = &batch;
        workers[stage].stage = stage;
        if (pthread_create(&threads[stage],
                           NULL,
                           replay_step_pipeline_worker_main,
                           &workers[stage]) != 0) {
            replay_step_pipeline_fail(&batch, "failed to create async per-step worker");
            break;
        }
        created++;
    }
    const double setup_ms = replay_now_ms() - total0;
    const double wait0 = replay_now_ms();
    for (int i = 0; i < created; i++) {
        (void)pthread_join(threads[i], NULL);
    }
    const double host_wait_ms = replay_now_ms() - wait0;

    const bool failed = batch.failed;
    int rc = failed ? 1 : 0;
    double complete_ms = 0.0;
    if (!failed) {
        const double complete0 = replay_now_ms();
        if (replay_sync_all_stages(err, errlen)) {
            rc = 1;
        }
        complete_ms = replay_now_ms() - complete0;
    } else if (err && errlen) {
        snprintf(err,
                 errlen,
                 "%s",
                 batch.err[0] ? batch.err : "async per-step pipeline failed");
    }

    const double total_ms = replay_now_ms() - total0;
    if (rc == 0) {
        if (bucket_ms) *bucket_ms += total_ms;
        if (counters) {
            counters->async_pipeline_dispatches++;
            counters->async_pipeline_total_ms += total_ms;
            counters->async_pipeline_setup_ms += setup_ms;
            counters->async_pipeline_host_wait_ms += host_wait_ms;
            counters->async_pipeline_complete_ms += complete_ms;
            for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
                counters->stage_decode_ms[stage] += batch.stage_decode_ms[stage];
                counters->async_pipeline_worker_wait_ms[stage] += batch.worker_wait_ms[stage];
                counters->async_pipeline_sync_ms[stage] += batch.sync_ms[stage];
                counters_add_report(counters, stage, &batch.reports[stage][0]);
            }
            for (int stage = 1; stage < DS4_V100_EXPECTED_GPUS; stage++) {
                counters->handoff_ms[stage - 1] += batch.handoff_ms[stage - 1];
            }
        }
    }

    pthread_cond_destroy(&batch.cv);
    pthread_mutex_destroy(&batch.mu);
    return rc;
}

static int replay_feed_token_batch_async_pipeline(ds4_v100_replay *rt,
                                                  const uint32_t *tokens,
                                                  const uint32_t *positions,
                                                  uint32_t n_slots,
                                                  ds4_v100_replay_counters *counters,
                                                  double *bucket_ms,
                                                  char *err,
                                                  size_t errlen) {
    if (!rt || !tokens || !positions || n_slots == 0 ||
        n_slots > DS4_V100_SCHED_MAX_SLOTS) {
        return replay_error(err, errlen, "missing V100 replay async pipeline input");
    }
    if (n_slots == 1) {
        return replay_feed_token_batch(rt, tokens, positions, n_slots, counters, bucket_ms, err, errlen);
    }
    if (!rt->pipeline) {
        return replay_error(err, errlen, "async pipeline runtime is not open");
    }
    return replay_pipeline_runtime_dispatch(
        rt->pipeline, tokens, positions, n_slots, counters, bucket_ms, err, errlen);
}

static int replay_feed_token_batch_async_mailbox(ds4_v100_replay *rt,
                                                 const uint32_t *tokens,
                                                 const uint32_t *positions,
                                                 uint32_t n_slots,
                                                 ds4_v100_replay_counters *counters,
                                                 double *bucket_ms,
                                                 char *err,
                                                 size_t errlen) {
    if (!rt || !tokens || !positions || n_slots == 0 ||
        n_slots > DS4_V100_SCHED_MAX_SLOTS) {
        return replay_error(err, errlen, "missing V100 replay async mailbox input");
    }
    if (n_slots == 1) {
        return replay_feed_token_batch(rt, tokens, positions, n_slots, counters, bucket_ms, err, errlen);
    }
    if (!rt->mailbox) {
        return replay_error(err, errlen, "async mailbox runtime is not open");
    }
    return replay_mailbox_runtime_dispatch(
        rt->mailbox, tokens, positions, n_slots, counters, bucket_ms, err, errlen);
}

static int replay_feed_token_batch_async_selected(ds4_v100_replay *rt,
                                                  const uint32_t *tokens,
                                                  const uint32_t *positions,
                                                  uint32_t n_slots,
                                                  ds4_v100_replay_counters *counters,
                                                  double *bucket_ms,
                                                  char *err,
                                                  size_t errlen) {
    switch (replay_async_pipeline_mode(rt ? &rt->opts : NULL)) {
    case DS4_V100_REPLAY_ASYNC_PIPELINE_PERSISTENT:
        return replay_feed_token_batch_async_pipeline(
            rt, tokens, positions, n_slots, counters, bucket_ms, err, errlen);
    case DS4_V100_REPLAY_ASYNC_PIPELINE_PER_STEP:
        return replay_feed_token_batch_async_per_step(
            rt, tokens, positions, n_slots, counters, bucket_ms, err, errlen);
    case DS4_V100_REPLAY_ASYNC_PIPELINE_MAILBOX:
        return replay_feed_token_batch_async_mailbox(
            rt, tokens, positions, n_slots, counters, bucket_ms, err, errlen);
    case DS4_V100_REPLAY_ASYNC_PIPELINE_OFF:
    default:
        return replay_feed_token_batch(
            rt, tokens, positions, n_slots, counters, bucket_ms, err, errlen);
    }
}

static int replay_feed_token_batch_selected(ds4_v100_replay *rt,
                                            const uint32_t *tokens,
                                            const uint32_t *positions,
                                            uint32_t n_slots,
                                            ds4_v100_replay_counters *counters,
                                            double *bucket_ms,
                                            char *err,
                                            size_t errlen) {
    if (rt && replay_async_pipeline_mode(&rt->opts) != DS4_V100_REPLAY_ASYNC_PIPELINE_OFF &&
        n_slots > 1) {
        return replay_feed_token_batch_async_selected(
            rt, tokens, positions, n_slots, counters, bucket_ms, err, errlen);
    }
    if (rt && rt->opts.wavefront_decode) {
        return replay_feed_token_batch_wavefront(
            rt, tokens, positions, n_slots, counters, bucket_ms, err, errlen);
    }
    return replay_feed_token_batch(rt, tokens, positions, n_slots, counters, bucket_ms, err, errlen);
}

static int replay_select_token_slot(ds4_v100_replay *rt,
                                    uint32_t slot,
                                    ds4_v100_replay_output *out,
                                    ds4_v100_replay_counters *counters,
                                    char *err,
                                    size_t errlen) {
    uint32_t token = UINT32_MAX;
    float logit = 0.0f;
    double t0 = replay_now_ms();
    if (ds4_v100_stage_scheduler_select_token_slot(rt->scheds[DS4_V100_EXPECTED_GPUS - 1],
                                                   slot,
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

static int replay_select_token_batch(ds4_v100_replay *rt,
                                     const replay_batch_slot_plan *plan,
                                     uint32_t n_slots,
                                     uint32_t step,
                                     ds4_v100_replay_output *outputs,
                                     uint32_t output_stride,
                                     ds4_v100_replay_counters *counters,
                                     char *err,
                                     size_t errlen) {
    if (!rt || !plan || !outputs || n_slots == 0 || n_slots > DS4_V100_SCHED_MAX_SLOTS) {
        return replay_error(err, errlen, "missing V100 replay output batch input");
    }
    if (n_slots == 1) {
        const uint32_t prompt_idx = plan[0].prompt_idx;
        return replay_select_token_slot(rt,
                                        0,
                                        &outputs[(uint64_t)prompt_idx * output_stride + step],
                                        counters,
                                        err,
                                        errlen);
    }

    uint32_t tokens[DS4_V100_SCHED_MAX_SLOTS];
    float logits[DS4_V100_SCHED_MAX_SLOTS];
    double t0 = replay_now_ms();
    if (ds4_v100_stage_scheduler_select_token_batch(rt->scheds[DS4_V100_EXPECTED_GPUS - 1],
                                                    0,
                                                    n_slots,
                                                    tokens,
                                                    logits,
                                                    err,
                                                    errlen)) {
        return 1;
    }
    if (counters) counters->output_head_ms += replay_now_ms() - t0;

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        const uint32_t prompt_idx = plan[slot].prompt_idx;
        ds4_v100_replay_output *out =
            &outputs[(uint64_t)prompt_idx * output_stride + step];
        memset(out, 0, sizeof(*out));
        out->token = tokens[slot];
        out->logit = logits[slot];
        t0 = replay_now_ms();
        out->text = ds4_token_text(rt->tokenizer, (int)tokens[slot], &out->text_len);
        if (counters) counters->token_text_ms += replay_now_ms() - t0;
        if (!out->text) return replay_error(err, errlen, "failed to decode selected token text");
    }
    return 0;
}

int ds4_v100_replay_begin_generation(ds4_v100_replay *rt,
                                      uint32_t prompt_tokens,
                                      ds4_v100_replay_counters *counters,
                                      char *err,
                                      size_t errlen) {
    if (!rt || !counters) return replay_error(err, errlen, "missing V100 replay begin input");
    if (rt->used) {
        return replay_error(err,
                            errlen,
                            "V100 replay runtime is one-shot; reopen it for another prompt");
    }
    replay_init_counters(rt, prompt_tokens, counters);
    return 0;
}

int ds4_v100_replay_feed_token_at_position(ds4_v100_replay *rt,
                                            uint32_t token,
                                            uint32_t position,
                                            ds4_v100_replay_counters *counters,
                                            double *bucket_ms,
                                            char *err,
                                            size_t errlen) {
    if (!rt) return replay_error(err, errlen, "missing V100 replay feed input");
    return replay_feed_token(rt, token, position, counters, bucket_ms, err, errlen);
}

int ds4_v100_replay_select_current_token(ds4_v100_replay *rt,
                                          ds4_v100_replay_output *out,
                                          ds4_v100_replay_counters *counters,
                                          char *err,
                                          size_t errlen) {
    if (!rt || !out) return replay_error(err, errlen, "missing V100 replay select input");
    return replay_select_token(rt, out, counters, err, errlen);
}

void ds4_v100_replay_snapshot_free(ds4_v100_replay_snapshot *snapshot) {
    if (!snapshot) return;
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        ds4_v100_stage_scheduler_snapshot_free(snapshot->stages[stage]);
    }
    free(snapshot);
}

uint64_t ds4_v100_replay_snapshot_bytes(
    const ds4_v100_replay_snapshot *snapshot) {
    return snapshot ? snapshot->bytes : 0;
}

int ds4_v100_replay_snapshot_create(ds4_v100_replay *rt,
                                    ds4_v100_replay_snapshot **out,
                                    char *err,
                                    size_t errlen) {
    if (!out) return replay_error(err, errlen, "missing V100 replay snapshot output");
    *out = NULL;
    if (!rt) return replay_error(err, errlen, "missing V100 replay snapshot input");
    if (rt->opts.kv_active_slots != 1) {
        if (err && errlen) {
            snprintf(err,
                     errlen,
                     "V100 replay snapshot currently requires kv_active_slots=1, got %" PRIu64,
                     rt->opts.kv_active_slots);
        }
        return 1;
    }

    ds4_v100_replay_snapshot *snap =
        (ds4_v100_replay_snapshot *)calloc(1, sizeof(*snap));
    if (!snap) return replay_error(err, errlen, "failed to allocate V100 replay snapshot");

    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        if (!rt->scheds[stage]) {
            ds4_v100_replay_snapshot_free(snap);
            return replay_error(err, errlen, "missing V100 replay scheduler for snapshot");
        }
        if (ds4_v100_stage_scheduler_snapshot_create(rt->scheds[stage],
                                                     &snap->stages[stage],
                                                     err,
                                                     errlen)) {
            ds4_v100_replay_snapshot_free(snap);
            return 1;
        }
        snap->bytes += ds4_v100_stage_scheduler_snapshot_bytes(snap->stages[stage]);
    }
    *out = snap;
    return 0;
}

int ds4_v100_replay_snapshot_restore(ds4_v100_replay *rt,
                                     const ds4_v100_replay_snapshot *snapshot,
                                     char *err,
                                     size_t errlen) {
    if (!rt || !snapshot) {
        return replay_error(err, errlen, "missing V100 replay snapshot restore input");
    }
    if (rt->opts.kv_active_slots != 1) {
        if (err && errlen) {
            snprintf(err,
                     errlen,
                     "V100 replay snapshot restore currently requires kv_active_slots=1, got %" PRIu64,
                     rt->opts.kv_active_slots);
        }
        return 1;
    }
    for (int stage = 0; stage < DS4_V100_EXPECTED_GPUS; stage++) {
        if (!rt->scheds[stage] || !snapshot->stages[stage]) {
            return replay_error(err, errlen, "missing V100 replay scheduler snapshot");
        }
        if (ds4_v100_stage_scheduler_snapshot_restore(rt->scheds[stage],
                                                      snapshot->stages[stage],
                                                      err,
                                                      errlen)) {
            return 1;
        }
    }
    return 0;
}

int ds4_v100_replay_verify_token_block(
    ds4_v100_replay *rt,
    const uint32_t *tokens,
    const uint32_t *positions,
    uint32_t n_tokens,
    uint32_t accepted_prefix_len,
    ds4_v100_replay_output *outputs,
    uint32_t output_cap,
    ds4_v100_replay_target_block_report *report,
    ds4_v100_replay_counters *counters,
    char *err,
    size_t errlen) {
    if (report) memset(report, 0, sizeof(*report));
    if (!rt || !tokens || !positions || n_tokens == 0 || !outputs ||
        output_cap < n_tokens) {
        return replay_error(err, errlen, "missing V100 replay target block input");
    }
    if (rt->opts.kv_active_slots != 1) {
        if (err && errlen) {
            snprintf(err,
                     errlen,
                     "V100 replay target block currently requires kv_active_slots=1, got %" PRIu64,
                     rt->opts.kv_active_slots);
        }
        return 1;
    }
    if (accepted_prefix_len > n_tokens) accepted_prefix_len = n_tokens;

    const double t0 = replay_now_ms();
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (replay_feed_token(rt,
                              tokens[i],
                              positions[i],
                              counters,
                              counters ? &counters->continuation_decode_ms : NULL,
                              err,
                              errlen)) {
            return 1;
        }
        if (counters) counters->total_input_tokens++;
        if (replay_select_token(rt, &outputs[i], counters, err, errlen)) {
            return 1;
        }
    }

    if (report) {
        report->target_forwards = n_tokens;
        report->accepted_prefix_len = accepted_prefix_len;
        report->target_tokens_verified = n_tokens;
        report->effective_output_tokens = accepted_prefix_len;
        report->speculative_saves =
            report->effective_output_tokens > report->target_forwards
                ? report->effective_output_tokens - report->target_forwards
                : 0u;
        report->verify_ms = replay_now_ms() - t0;
    }
    return 0;
}

void ds4_v100_replay_finish_generation(ds4_v100_replay *rt,
                                        uint32_t generated_tokens,
                                        double total_ms,
                                        ds4_v100_replay_counters *counters) {
    if (counters) {
        counters->generated_tokens = generated_tokens;
        counters->total_ms = total_ms;
    }
    if (rt) rt->used = true;
}

static void replay_sort_batch_plan(replay_batch_slot_plan *plan, uint32_t n) {
    for (uint32_t i = 1; i < n; i++) {
        replay_batch_slot_plan key = plan[i];
        uint32_t j = i;
        while (j > 0 && plan[j - 1].prompt_len < key.prompt_len) {
            plan[j] = plan[j - 1];
            j--;
        }
        plan[j] = key;
    }
}

int ds4_v100_replay_generate_batch(ds4_v100_replay *rt,
                                   const ds4_tokens *prompts,
                                   uint32_t n_prompts,
                                   uint32_t max_tokens,
                                   ds4_v100_replay_output *outputs,
                                   uint32_t output_stride,
                                   uint32_t *out_counts,
                                   ds4_v100_replay_counters *counters,
                                   char *err,
                                   size_t errlen) {
    if (!rt || !prompts || !outputs || n_prompts == 0 || max_tokens == 0 ||
        output_stride < max_tokens || n_prompts > DS4_V100_SCHED_MAX_SLOTS) {
        return replay_error(err, errlen, "missing V100 replay batch generation input");
    }
    if (n_prompts > rt->opts.kv_active_slots) {
        return replay_error(err, errlen, "batch prompt count exceeds configured slots");
    }
    if (rt->used) {
        return replay_error(err,
                            errlen,
                            "V100 replay runtime is one-shot; reopen it for another prompt");
    }

    ds4_v100_replay_counters local;
    ds4_v100_replay_counters *c = counters ? counters : &local;
    replay_init_counters(rt, 0, c);

    replay_batch_slot_plan plan[DS4_V100_SCHED_MAX_SLOTS] = {0};
    uint32_t max_prompt_len = 0;
    uint32_t total_prompt_len = 0;
    for (uint32_t i = 0; i < n_prompts; i++) {
        if (prompts[i].len <= 0) {
            return replay_error(err, errlen, "batch prompt must contain at least one token");
        }
        if (rt->opts.kv_ctx_tokens &&
            (uint64_t)prompts[i].len + max_tokens > rt->opts.kv_ctx_tokens) {
            return replay_error(err, errlen, "V100 replay prompt exceeds configured context");
        }
        plan[i].prompt_idx = i;
        plan[i].prompt_len = (uint32_t)prompts[i].len;
        if (plan[i].prompt_len > max_prompt_len) max_prompt_len = plan[i].prompt_len;
        total_prompt_len += plan[i].prompt_len;
    }
    replay_sort_batch_plan(plan, n_prompts);
    c->prompt_tokens = total_prompt_len;

    const double total0 = replay_now_ms();
    uint32_t batch_tokens[DS4_V100_SCHED_MAX_SLOTS];
    uint32_t batch_positions[DS4_V100_SCHED_MAX_SLOTS];
    for (uint32_t pos = 0; pos < max_prompt_len; pos++) {
        uint32_t active = 0;
        while (active < n_prompts && plan[active].prompt_len > pos) {
            const ds4_tokens *p = &prompts[plan[active].prompt_idx];
            if (p->v[pos] < 0) return replay_error(err, errlen, "negative prompt token");
            batch_tokens[active] = (uint32_t)p->v[pos];
            batch_positions[active] = pos;
            active++;
        }
        if (active == 0) continue;
        if (replay_feed_token_batch_selected(rt,
                                             batch_tokens,
                                             batch_positions,
                                             active,
                                             c,
                                             &c->prompt_replay_ms,
                                             err,
                                             errlen)) {
            return 1;
        }
        c->total_input_tokens += active;
    }

    for (uint32_t i = 0; i < n_prompts; i++) {
        if (out_counts) out_counts[i] = 0;
    }

    if (replay_select_token_batch(rt,
                                  plan,
                                  n_prompts,
                                  0,
                                  outputs,
                                  output_stride,
                                  c,
                                  err,
                                  errlen)) {
        return 1;
    }
    for (uint32_t slot = 0; slot < n_prompts; slot++) {
        if (out_counts) out_counts[plan[slot].prompt_idx] = 1;
    }

    for (uint32_t step = 1; step < max_tokens; step++) {
        for (uint32_t slot = 0; slot < n_prompts; slot++) {
            const uint32_t prompt_idx = plan[slot].prompt_idx;
            const ds4_v100_replay_output *row =
                &outputs[(uint64_t)prompt_idx * output_stride];
            batch_tokens[slot] = row[step - 1].token;
            batch_positions[slot] = plan[slot].prompt_len + step - 1;
        }
        if (replay_feed_token_batch_selected(rt,
                                             batch_tokens,
                                             batch_positions,
                                             n_prompts,
                                             c,
                                             &c->continuation_decode_ms,
                                             err,
                                             errlen)) {
            return 1;
        }
        c->total_input_tokens += n_prompts;
        if (replay_select_token_batch(rt,
                                      plan,
                                      n_prompts,
                                      step,
                                      outputs,
                                      output_stride,
                                      c,
                                      err,
                                      errlen)) {
            return 1;
        }
        for (uint32_t slot = 0; slot < n_prompts; slot++) {
            if (out_counts) out_counts[plan[slot].prompt_idx] = step + 1;
        }
    }
    c->generated_tokens = n_prompts * max_tokens;
    c->total_ms = replay_now_ms() - total0;
    rt->used = true;
    return 0;
}

int ds4_v100_replay_generate_first_token_batch(
    ds4_v100_replay *rt,
    const ds4_tokens *prompts,
    uint32_t n_prompts,
    ds4_v100_replay_output *outputs,
    ds4_v100_replay_counters *counters,
    char *err,
    size_t errlen) {
    return ds4_v100_replay_generate_batch(rt,
                                          prompts,
                                          n_prompts,
                                          1,
                                          outputs,
                                          1,
                                          NULL,
                                          counters,
                                          err,
                                          errlen);
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
    replay_init_counters(rt, (uint32_t)prompt->len, c);

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
    return ds4_v100_replay_read_output_hc_slot(rt, 0, dst, bytes, err, errlen);
}

int ds4_v100_replay_read_output_hc_slot(ds4_v100_replay *rt,
                                        uint32_t slot,
                                        float *dst,
                                        uint64_t bytes,
                                        char *err,
                                        size_t errlen) {
    if (!rt || !rt->scheds[DS4_V100_EXPECTED_GPUS - 1] || !dst) {
        return replay_error(err, errlen, "missing V100 replay HC read input");
    }
    if (!ds4_v100_stage_scheduler_read_hc_slot(rt->scheds[DS4_V100_EXPECTED_GPUS - 1],
                                               slot,
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
