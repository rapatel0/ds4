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
