enum {
    DS4_V100_SCHED_UPLOAD_CHUNK = 8 * 1024 * 1024,
};

typedef struct {
    ds4_layer_decode_cache cache;
    ds4_gpu_tensor *raw_kv;
    ds4_gpu_tensor *attn_state_kv;
    ds4_gpu_tensor *attn_state_score;
    ds4_gpu_tensor *attn_comp_kv;
    ds4_gpu_tensor *index_state_kv;
    ds4_gpu_tensor *index_state_score;
    ds4_gpu_tensor *index_comp_kv;
    ds4_gpu_tensor *indexer_topk;
} scheduler_layer_cache;

typedef struct {
    unsigned char *data;
    uint64_t bytes;
} scheduler_snapshot_tensor;

typedef struct {
    uint32_t n_attn_comp;
    uint32_t n_index_comp;
    scheduler_snapshot_tensor raw_kv;
    scheduler_snapshot_tensor attn_state_kv;
    scheduler_snapshot_tensor attn_state_score;
    scheduler_snapshot_tensor attn_comp_kv;
    scheduler_snapshot_tensor index_state_kv;
    scheduler_snapshot_tensor index_state_score;
    scheduler_snapshot_tensor index_comp_kv;
    scheduler_snapshot_tensor indexer_topk;
} scheduler_layer_cache_snapshot;

struct ds4_stage_scheduler_snapshot {
    int stage_id;
    int gpu;
    int layer_begin;
    int layer_end;
    uint32_t active_slots;
    int cur_hc_slot;
    scheduler_snapshot_tensor cur_hc;
    scheduler_layer_cache_snapshot layers[DS4_V100_N_LAYERS];
    uint64_t captured_bytes;
};

struct ds4_stage_scheduler {
    ds4_context *ctx;
    ds4_pack *pack;
    ds4_gpu_arena *arena;
    ds4_gpu_arena *tp2_owner_arena;
    ds4_gpu_arena *tp2_peer_arena;
    ds4_stage_info stage;
    ds4_tensor_binding token_embedding;
    ds4_tensor_binding hc_head_fn;
    ds4_tensor_binding hc_head_base;
    ds4_tensor_binding hc_head_scale;
    ds4_tensor_binding output_norm;
    ds4_tensor_binding output_weight;
    ds4_layer_state states[DS4_V100_N_LAYERS];
    scheduler_layer_cache caches[DS4_V100_N_LAYERS * DS4_V100_SCHED_MAX_SLOTS];
    ds4_gpu_tensor *hc_a[DS4_V100_SCHED_MAX_SLOTS];
    ds4_gpu_tensor *hc_b[DS4_V100_SCHED_MAX_SLOTS];
    ds4_gpu_tensor *cur_hc[DS4_V100_SCHED_MAX_SLOTS];
    ds4_gpu_tensor *output_hc_batch;
    ds4_gpu_tensor *output_hc_norm;
    ds4_gpu_tensor *output_head_pre;
    ds4_gpu_tensor *output_head_weights;
    ds4_gpu_tensor *output_embd;
    ds4_gpu_tensor *output_norm_scratch;
    ds4_gpu_tensor *output_logits;
    ds4_gpu_tensor *tp2_peer_input;
    ds4_gpu_tensor *tp2_peer_selected;
    ds4_gpu_tensor *tp2_peer_weights;
    ds4_gpu_tensor *tp2_peer_out;
    ds4_gpu_tensor *tp2_peer_recv;
    uint32_t output_scratch_vocab;
    uint32_t output_scratch_slots;
    uint32_t tp2_scratch_slots;
    ds4_layer_batch_scratch batch_scratch;
    uint64_t uploaded_tensors;
    uint64_t uploaded_bytes;
    const void *model_map;
    uint64_t model_size;
    bool model_map_uses_shard_offsets;
    int shard_fd;
    void *shard_map;
    uint64_t shard_size;
    char shard_path[1024];
    uint32_t active_slots;
    uint32_t raw_cap;
    uint32_t raw_window;
    uint32_t attn_comp_cap;
    uint32_t index_comp_cap;
    uint32_t indexer_top_k;
    bool fp8_kv_cache;
    bool suppress_router_readback;
    int tp2_layer;
    int tp2_layer_count;
    int tp2_requested_peer_gpu;
    char tp2_shard_dir[1024];
    uint64_t tp2_uploaded_bytes;
};

static int scheduler_error(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg ? msg : "scheduler error");
    return 1;
}

static int scheduler_errorf(char *err, size_t errlen, const char *fmt, int value) {
    if (err && errlen) snprintf(err, errlen, fmt, value);
    return 1;
}

static int scheduler_errorf_u64(char *err, size_t errlen, const char *fmt, uint64_t value) {
    if (err && errlen) snprintf(err, errlen, fmt, value);
    return 1;
}

static int scheduler_errorf_u32(char *err, size_t errlen, const char *fmt, uint32_t value) {
    if (err && errlen) snprintf(err, errlen, fmt, value);
    return 1;
}

static const char *scheduler_first_env(const char *a, const char *b) {
    const char *v = a ? getenv(a) : NULL;
    if (v && v[0]) return v;
    v = b ? getenv(b) : NULL;
    return (v && v[0]) ? v : NULL;
}

static bool scheduler_env_enabled(const char *name) {
    const char *v = name ? getenv(name) : NULL;
    if (!v || !v[0]) return false;
    if (!strcmp(v, "0") || !strcmp(v, "false") || !strcmp(v, "False") ||
        !strcmp(v, "no") || !strcmp(v, "No") || !strcmp(v, "off") ||
        !strcmp(v, "Off")) {
        return false;
    }
    return true;
}

static bool scheduler_debug_hc_finite_enabled(void) {
    return scheduler_env_enabled("DS4_V100_DEBUG_HC_FINITE");
}

static bool scheduler_env_disabled(const char *name) {
    const char *v = name ? getenv(name) : NULL;
    if (!v || !v[0]) return false;
    return !scheduler_env_enabled(name);
}

static bool scheduler_debug_hc_finite_layer_checks_enabled(void) {
    if (!scheduler_debug_hc_finite_enabled()) return false;
    if (scheduler_env_disabled("DS4_V100_DEBUG_HC_FINITE_LAYER_CHECKS")) return false;
    return true;
}

static bool scheduler_debug_hc_finite_pre_output_enabled(void) {
    if (!scheduler_debug_hc_finite_enabled()) return false;
    if (scheduler_env_disabled("DS4_V100_DEBUG_HC_FINITE_PRE_OUTPUT")) return false;
    return true;
}

static int scheduler_hc_finite_error(const ds4_stage_scheduler *sched,
                                     const char *phase,
                                     int layer,
                                     uint32_t slot,
                                     uint32_t token,
                                     uint32_t position,
                                     uint64_t index,
                                     float value,
                                     char *err,
                                     size_t errlen) {
    char msg[512];
    const int n = snprintf(msg,
                           sizeof(msg),
                           "HC non-finite: phase=%s stage=%d gpu=%d layer=%d slot=%u token=%u position=%u index=%" PRIu64 " value=%g",
                           phase ? phase : "unknown",
                           sched ? sched->stage.stage_id : -1,
                           sched ? sched->stage.gpu : -1,
                           layer,
                           slot,
                           token,
                           position,
                           index,
                           value);
    if (err && errlen) {
        if (n < 0) {
            snprintf(err, errlen, "HC non-finite diagnostic formatting failed");
        } else {
            snprintf(err, errlen, "%s", msg);
        }
    }
    fprintf(stderr, "ds4-v100-scheduler: %s\n", msg);
    return 1;
}

static int scheduler_check_hc_finite(const ds4_stage_scheduler *sched,
                                     const ds4_gpu_tensor *hc,
                                     const char *phase,
                                     int layer,
                                     uint32_t slot,
                                     uint32_t token,
                                     uint32_t position,
                                     char *err,
                                     size_t errlen) {
    if (!hc) {
        return scheduler_errorf(err, errlen, "missing HC tensor for finite check slot %d",
                                (int)slot);
    }
    const uint64_t hc_values = (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS;
    const uint64_t hc_bytes = hc_values * sizeof(float);
    float *host = (float *)malloc((size_t)hc_bytes);
    if (!host) {
        return scheduler_error(err, errlen, "failed to allocate HC finite check buffer");
    }
    if (!ds4_gpu_tensor_read(hc, 0, host, hc_bytes)) {
        free(host);
        return scheduler_errorf(err, errlen, "HC finite check readback failed for slot %d",
                                (int)slot);
    }
    for (uint64_t i = 0; i < hc_values; i++) {
        if (!isfinite(host[i])) {
            const float value = host[i];
            free(host);
            return scheduler_hc_finite_error(sched,
                                             phase,
                                             layer,
                                             slot,
                                             token,
                                             position,
                                             i,
                                             value,
                                             err,
                                             errlen);
        }
    }
    free(host);
    return 0;
}

static int scheduler_parse_i32_value(const char *s,
                                     const char *name,
                                     int min_value,
                                     int max_value,
                                     int *out,
                                     char *err,
                                     size_t errlen) {
    if (!s || !s[0] || !out) {
        return scheduler_error(err, errlen, "missing integer scheduler option");
    }
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end != '\0' || v < min_value || v > max_value) {
        if (err && errlen) {
            snprintf(err,
                     errlen,
                     "invalid %s: %s",
                     name ? name : "scheduler integer",
                     s);
        }
        return 1;
    }
    *out = (int)v;
    return 0;
}

static int scheduler_join_path(char *dst,
                               size_t dst_size,
                               const char *dir,
                               const char *base,
                               char *err,
                               size_t errlen) {
    if (!dst || dst_size == 0 || !dir || !dir[0] || !base || !base[0]) {
        return scheduler_error(err, errlen, "missing scheduler path component");
    }
    int n = snprintf(dst, dst_size, "%s/%s", dir, base);
    if (n < 0 || (size_t)n >= dst_size) {
        return scheduler_error(err, errlen, "scheduler path is too long");
    }
    return 0;
}

static uint64_t scheduler_model_offset(const ds4_stage_scheduler *sched,
                                       const ds4_tensor_binding *b) {
    if (!b) return 0;
    return (sched && sched->model_map_uses_shard_offsets) ? b->shard_offset : b->source_offset;
}

static int scheduler_activate_model_source(const ds4_stage_scheduler *sched,
                                           char *err,
                                           size_t errlen) {
    if (!sched || sched->shard_fd < 0) return 0;
    if (!ds4_gpu_set_model_fd(sched->shard_fd)) {
        return scheduler_errorf(err, errlen, "failed to activate shard fd for gpu%d",
                                sched->stage.gpu);
    }
    return 0;
}

static scheduler_layer_cache *scheduler_cache_slot(ds4_stage_scheduler *sched,
                                                   int layer,
                                                   uint32_t slot) {
    if (!sched || layer < 0 || layer >= DS4_V100_N_LAYERS ||
        slot >= DS4_V100_SCHED_MAX_SLOTS) {
        return NULL;
    }
    return &sched->caches[(size_t)layer * DS4_V100_SCHED_MAX_SLOTS + slot];
}

static const scheduler_layer_cache *scheduler_cache_slot_const(
    const ds4_stage_scheduler *sched,
    int layer,
    uint32_t slot) {
    if (!sched || layer < 0 || layer >= DS4_V100_N_LAYERS ||
        slot >= DS4_V100_SCHED_MAX_SLOTS) {
        return NULL;
    }
    return &sched->caches[(size_t)layer * DS4_V100_SCHED_MAX_SLOTS + slot];
}

typedef struct {
    const ds4_stage_scheduler *sched;
    ds4_stage_scheduler_checkpoint_fn checkpoint_fn;
    void *checkpoint_user;
    uint32_t token;
    uint32_t position;
} scheduler_layer_checkpoint_user;

static int scheduler_layer_checkpoint(
    const ds4_layer_execute_checkpoint *layer_cp,
    void *user,
    char *err,
    size_t errlen) {
    scheduler_layer_checkpoint_user *u = (scheduler_layer_checkpoint_user *)user;
    if (!u || !u->sched || !u->checkpoint_fn || !layer_cp) {
        return scheduler_error(err, errlen, "missing scheduler layer checkpoint input");
    }
    ds4_stage_scheduler_checkpoint cp = {
        .stage_id = u->sched->stage.stage_id,
        .gpu = u->sched->stage.gpu,
        .layer = layer_cp->layer,
        .kind = layer_cp->kind,
        .position = u->position,
        .token = u->token,
        .hc = layer_cp->hc,
        .hc_bytes = layer_cp->hc_bytes,
    };
    return u->checkpoint_fn(&cp, u->checkpoint_user, err, errlen);
}

void ds4_stage_scheduler_options_init(ds4_stage_scheduler_options *opts) {
    if (!opts) return;
    memset(opts, 0, sizeof(*opts));
    opts->stage_id = 0;
    opts->raw_cap = DS4_V100_SWA_ROWS;
    opts->raw_window = DS4_V100_SWA_ROWS;
    opts->attn_comp_cap = 4;
    opts->index_comp_cap = 4;
    opts->indexer_top_k = 1;
    opts->kv_ctx_tokens = 1048576;
    opts->kv_active_slots = 1;
}

static void free_layer_cache(scheduler_layer_cache *lc) {
    if (!lc) return;
    ds4_gpu_tensor_free(lc->indexer_topk);
    ds4_gpu_tensor_free(lc->index_comp_kv);
    ds4_gpu_tensor_free(lc->index_state_score);
    ds4_gpu_tensor_free(lc->index_state_kv);
    ds4_gpu_tensor_free(lc->attn_comp_kv);
    ds4_gpu_tensor_free(lc->attn_state_score);
    ds4_gpu_tensor_free(lc->attn_state_kv);
    ds4_gpu_tensor_free(lc->raw_kv);
    memset(lc, 0, sizeof(*lc));
}

static void free_output_head_scratch(ds4_stage_scheduler *sched) {
    if (!sched) return;
    ds4_gpu_tensor_free(sched->output_logits);
    ds4_gpu_tensor_free(sched->output_norm_scratch);
    ds4_gpu_tensor_free(sched->output_embd);
    ds4_gpu_tensor_free(sched->output_head_weights);
    ds4_gpu_tensor_free(sched->output_head_pre);
    ds4_gpu_tensor_free(sched->output_hc_norm);
    ds4_gpu_tensor_free(sched->output_hc_batch);
    sched->output_logits = NULL;
    sched->output_norm_scratch = NULL;
    sched->output_embd = NULL;
    sched->output_head_weights = NULL;
    sched->output_head_pre = NULL;
    sched->output_hc_norm = NULL;
    sched->output_hc_batch = NULL;
    sched->output_scratch_vocab = 0;
    sched->output_scratch_slots = 0;
}

void ds4_stage_scheduler_close(ds4_stage_scheduler *sched) {
    if (!sched) return;
    for (uint32_t slot = 0; slot < DS4_V100_SCHED_MAX_SLOTS; slot++) {
        ds4_gpu_tensor_free(sched->hc_b[slot]);
        ds4_gpu_tensor_free(sched->hc_a[slot]);
    }
    for (int layer = 0; layer < DS4_V100_N_LAYERS; layer++) {
        for (uint32_t slot = 0; slot < DS4_V100_SCHED_MAX_SLOTS; slot++) {
            free_layer_cache(scheduler_cache_slot(sched, layer, slot));
        }
    }
    ds4_gpu_tensor_free(sched->tp2_peer_recv);
    ds4_gpu_tensor_free(sched->tp2_peer_out);
    ds4_gpu_tensor_free(sched->tp2_peer_weights);
    ds4_gpu_tensor_free(sched->tp2_peer_selected);
    ds4_gpu_tensor_free(sched->tp2_peer_input);
    free_output_head_scratch(sched);
    ds4_layer_batch_scratch_free(&sched->batch_scratch);
    ds4_gpu_arena_close(sched->tp2_peer_arena);
    ds4_gpu_arena_close(sched->tp2_owner_arena);
    ds4_gpu_arena_close(sched->arena);
    if (sched->shard_map && sched->shard_map != MAP_FAILED) {
        munmap(sched->shard_map, (size_t)sched->shard_size);
    }
    if (sched->shard_fd >= 0) close(sched->shard_fd);
    ds4_pack_close(sched->pack);
    ds4_context_close(sched->ctx);
    free(sched);
}

typedef struct {
    ds4_stage_scheduler *sched;
    const unsigned char *model;
    unsigned char *chunk;
    char *err;
    size_t errlen;
} upload_stage_ud;

static int map_stage_shard(ds4_stage_scheduler *sched,
                           const char *shard_dir,
                           char *err,
                           size_t errlen) {
    if (!sched || !shard_dir || !shard_dir[0]) {
        return scheduler_error(err, errlen, "missing stage shard directory");
    }
    int n = snprintf(sched->shard_path,
                     sizeof(sched->shard_path),
                     "%s/gpu%d.weights",
                     shard_dir,
                     sched->stage.gpu);
    if (n < 0 || (size_t)n >= sizeof(sched->shard_path)) {
        return scheduler_error(err, errlen, "stage shard path is too long");
    }
    sched->shard_fd = open(sched->shard_path, O_RDONLY);
    if (sched->shard_fd < 0) {
        if (err && errlen) {
            snprintf(err, errlen, "cannot open %s: %s", sched->shard_path, strerror(errno));
        }
        return 1;
    }
    struct stat st;
    if (fstat(sched->shard_fd, &st) != 0 || st.st_size <= 0) {
        if (err && errlen) {
            snprintf(err, errlen, "cannot stat %s: %s", sched->shard_path, strerror(errno));
        }
        return 1;
    }
    sched->shard_size = (uint64_t)st.st_size;
    sched->shard_map = mmap(NULL,
                            (size_t)sched->shard_size,
                            PROT_READ,
                            MAP_PRIVATE,
                            sched->shard_fd,
                            0);
    if (sched->shard_map == MAP_FAILED) {
        sched->shard_map = NULL;
        if (err && errlen) {
            snprintf(err, errlen, "cannot mmap %s: %s", sched->shard_path, strerror(errno));
        }
        return 1;
    }
    sched->model_map = sched->shard_map;
    sched->model_size = sched->shard_size;
    sched->model_map_uses_shard_offsets = true;
    return 0;
}

static int upload_stage_entry(const ds4_pack_entry *e, void *ud_ptr) {
    upload_stage_ud *ud = (upload_stage_ud *)ud_ptr;
    ds4_stage_scheduler *sched = ud->sched;
    if (e->owning_gpu != sched->stage.gpu) return 0;
    if (e->source_offset > sched->model_size ||
        e->byte_length > sched->model_size - e->source_offset) {
        scheduler_errorf(ud->err, ud->errlen, "pack entry outside model map for gpu%d",
                         sched->stage.gpu);
        return 1;
    }
    uint64_t done = 0;
    while (done < e->byte_length) {
        uint64_t n = e->byte_length - done;
        if (n > DS4_V100_SCHED_UPLOAD_CHUNK) n = DS4_V100_SCHED_UPLOAD_CHUNK;
        memcpy(ud->chunk, ud->model + e->source_offset + done, (size_t)n);
        if (ds4_gpu_arena_upload(sched->arena, e->shard_offset + done, ud->chunk, n) != 0) {
            scheduler_errorf(ud->err, ud->errlen, "stage arena upload failed on gpu%d",
                             sched->stage.gpu);
            return 1;
        }
        done += n;
    }
    sched->uploaded_tensors++;
    sched->uploaded_bytes += e->byte_length;
    return 0;
}

static int upload_stage_shard(ds4_stage_scheduler *sched,
                              char *err,
                              size_t errlen) {
    if (!sched || !sched->shard_map || sched->shard_size == 0) {
        return scheduler_error(err, errlen, "missing mapped appliance shard");
    }
    uint64_t done = 0;
    const unsigned char *src = (const unsigned char *)sched->shard_map;
    while (done < sched->shard_size) {
        uint64_t n = sched->shard_size - done;
        if (n > DS4_V100_SCHED_UPLOAD_CHUNK) n = DS4_V100_SCHED_UPLOAD_CHUNK;
        if (ds4_gpu_arena_upload(sched->arena, done, src + done, n) != 0) {
            return scheduler_errorf(err, errlen, "stage shard upload failed on gpu%d",
                                    sched->stage.gpu);
        }
        done += n;
    }
    sched->uploaded_tensors++;
    sched->uploaded_bytes += sched->shard_size;
    return 0;
}

static int open_upload_file_arena(ds4_gpu_arena **out,
                                  int gpu,
                                  const char *path,
                                  uint64_t *uploaded_bytes,
                                  char *err,
                                  size_t errlen) {
    if (!out || gpu < 0 || !path || !path[0]) {
        return scheduler_error(err, errlen, "missing overlay arena input");
    }
    *out = NULL;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (err && errlen) snprintf(err, errlen, "cannot open %s: %s", path, strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        if (err && errlen) snprintf(err, errlen, "cannot stat %s: %s", path, strerror(errno));
        close(fd);
        return 1;
    }
    const uint64_t bytes = (uint64_t)st.st_size;
    ds4_gpu_arena *arena = NULL;
    if (ds4_gpu_arena_open(&arena, gpu, bytes) != 0) {
        close(fd);
        return scheduler_errorf(err, errlen, "failed to open TP2 overlay arena on gpu%d", gpu);
    }
    unsigned char *chunk = (unsigned char *)malloc(DS4_V100_SCHED_UPLOAD_CHUNK);
    if (!chunk) {
        ds4_gpu_arena_close(arena);
        close(fd);
        return scheduler_error(err, errlen, "failed to allocate TP2 upload chunk");
    }
    uint64_t done = 0;
    while (done < bytes) {
        uint64_t want = bytes - done;
        if (want > DS4_V100_SCHED_UPLOAD_CHUNK) want = DS4_V100_SCHED_UPLOAD_CHUNK;
        ssize_t got = read(fd, chunk, (size_t)want);
        if (got <= 0 || (uint64_t)got != want) {
            if (err && errlen) {
                snprintf(err, errlen, "short read while uploading %s", path);
            }
            free(chunk);
            ds4_gpu_arena_close(arena);
            close(fd);
            return 1;
        }
        if (ds4_gpu_arena_upload(arena, done, chunk, (uint64_t)got) != 0) {
            if (err && errlen) snprintf(err, errlen, "TP2 overlay upload failed on gpu%d", gpu);
            free(chunk);
            ds4_gpu_arena_close(arena);
            close(fd);
            return 1;
        }
        done += (uint64_t)got;
    }
    free(chunk);
    close(fd);
    if (uploaded_bytes) *uploaded_bytes = bytes;
    *out = arena;
    return 0;
}

static int upload_stage_weights(ds4_stage_scheduler *sched,
                                char *err,
                                size_t errlen) {
    if (sched && sched->shard_map) return upload_stage_shard(sched, err, errlen);
    unsigned char *chunk = (unsigned char *)malloc(DS4_V100_SCHED_UPLOAD_CHUNK);
    if (!chunk) return scheduler_error(err, errlen, "failed to allocate stage upload chunk");
    upload_stage_ud ud = {
        .sched = sched,
        .model = (const unsigned char *)sched->model_map,
        .chunk = chunk,
        .err = err,
        .errlen = errlen,
    };
    int rc = ds4_pack_for_each(sched->pack, upload_stage_entry, &ud);
    free(chunk);
    return rc != 0;
}

static int scheduler_load_tp2_env(ds4_stage_scheduler *sched,
                                  char *err,
                                  size_t errlen) {
    if (!sched) return scheduler_error(err, errlen, "missing scheduler TP2 env input");
    sched->tp2_layer = -1;
    sched->tp2_layer_count = 1;
    sched->tp2_requested_peer_gpu = -1;

    const char *layer_env = getenv("DS4_V100_TP_EP_LAYER_FIRST");
    if (!layer_env || !layer_env[0]) {
        layer_env = scheduler_first_env("DS4_V100_TP_ROUTED_FFN_LAYER", "DS4_V100_TP2_LAYER");
    }
    const bool enabled =
        scheduler_env_enabled("DS4_V100_TP_ROUTED_FFN") ||
        scheduler_env_enabled("DS4_V100_TP2_ROUTED_FFN") ||
        (layer_env && strcmp(layer_env, "-1") != 0);
    if (!enabled) return 0;
    if (!layer_env) {
        return scheduler_error(err,
                               errlen,
                               "TP/EP routed FFN requires DS4_V100_TP_EP_LAYER_FIRST");
    }
    if (scheduler_parse_i32_value(layer_env,
                                  "TP2 routed FFN layer",
                                  -1,
                                  DS4_V100_N_LAYERS - 1,
                                  &sched->tp2_layer,
                                  err,
                                  errlen)) {
        return 1;
    }
    if (sched->tp2_layer < 0) return 0;

    const char *count_env = getenv("DS4_V100_TP_EP_LAYER_COUNT");
    if (!count_env || !count_env[0]) {
        count_env = scheduler_first_env("DS4_V100_TP_ROUTED_FFN_LAYER_COUNT",
                                        "DS4_V100_TP2_LAYER_COUNT");
    }
    if (count_env &&
        scheduler_parse_i32_value(count_env,
                                  "TP2 routed FFN layer count",
                                  1,
                                  DS4_V100_N_LAYERS,
                                  &sched->tp2_layer_count,
                                  err,
                                  errlen)) {
        return 1;
    }
    if (sched->tp2_layer + sched->tp2_layer_count > DS4_V100_N_LAYERS) {
        return scheduler_error(err,
                               errlen,
                               "TP2 routed FFN span exceeds layer range");
    }

    const char *dir_env =
        getenv("DS4_V100_TP_EP_SHARD_DIR");
    if (!dir_env || !dir_env[0]) {
        dir_env = scheduler_first_env("DS4_V100_TP_ROUTED_FFN_SHARD_DIR", "DS4_V100_TP2_SHARD_DIR");
    }
    if (!dir_env) {
        return scheduler_error(err,
                               errlen,
                               "TP/EP routed FFN requires DS4_V100_TP_EP_SHARD_DIR");
    }
    if (strlen(dir_env) >= sizeof(sched->tp2_shard_dir)) {
        return scheduler_error(err, errlen, "TP2 routed FFN shard dir is too long");
    }
    snprintf(sched->tp2_shard_dir, sizeof(sched->tp2_shard_dir), "%s", dir_env);

    const char *peer_env =
        getenv("DS4_V100_TP_EP_PEER");
    if (!peer_env || !peer_env[0]) {
        peer_env = scheduler_first_env("DS4_V100_TP_ROUTED_FFN_PEER_GPU", "DS4_V100_TP2_PEER_GPU");
    }
    if (peer_env &&
        scheduler_parse_i32_value(peer_env,
                                  "TP2 routed FFN peer GPU",
                                  0,
                                  DS4_V100_EXPECTED_GPUS - 1,
                                  &sched->tp2_requested_peer_gpu,
                                  err,
                                  errlen)) {
        return 1;
    }
    return 0;
}

static int scheduler_setup_tp2_overlay(ds4_stage_scheduler *sched,
                                       char *err,
                                       size_t errlen) {
    if (!sched || sched->tp2_layer < 0) return 0;
    const int span_first = sched->tp2_layer;
    const int span_last = sched->tp2_layer + sched->tp2_layer_count - 1;
    const int local_first =
        span_first > sched->stage.layer_begin ? span_first : sched->stage.layer_begin;
    const int local_last =
        span_last < sched->stage.layer_end ? span_last : sched->stage.layer_end;
    if (local_first > local_last) {
        return 0;
    }
    if (span_first < sched->stage.layer_begin || span_last > sched->stage.layer_end) {
        if (err && errlen) {
            snprintf(err,
                     errlen,
                     "TP2 routed FFN span [%d,%d] crosses stage range [%d,%d]",
                     span_first,
                     span_last,
                     sched->stage.layer_begin,
                     sched->stage.layer_end);
        }
        return 1;
    }

    ds4_layer_state *first_state = NULL;
    int peer_gpu = -1;
    const char *owner_shard = NULL;
    const char *peer_shard = NULL;
    for (int layer = local_first; layer <= local_last; layer++) {
        ds4_layer_state *state = &sched->states[layer];
        if (!state->has_turbomind_tp2_routed) {
            return scheduler_errorf(err, errlen, "TP2 routed FFN layer %d has no TP2 bindings", layer);
        }
        if (state->owning_gpu != sched->stage.gpu ||
            state->turbomind_tp2_gate_up_binding[0].owning_gpu != sched->stage.gpu ||
            state->turbomind_tp2_down_binding[0].owning_gpu != sched->stage.gpu) {
            return scheduler_errorf(err, errlen, "TP2 routed FFN owner mismatch for layer %d", layer);
        }
        if (state->turbomind_tp2_peer_gpu < 0 ||
            state->turbomind_tp2_peer_gpu == sched->stage.gpu) {
            return scheduler_errorf(err, errlen, "TP2 routed FFN invalid peer GPU for layer %d", layer);
        }
        if (peer_gpu < 0) {
            first_state = state;
            peer_gpu = state->turbomind_tp2_peer_gpu;
            owner_shard = state->turbomind_tp2_gate_up_binding[0].shard_file;
            peer_shard = state->turbomind_tp2_gate_up_binding[1].shard_file;
        } else if (state->turbomind_tp2_peer_gpu != peer_gpu) {
            return scheduler_errorf(err, errlen, "TP2 routed FFN peer mismatch for layer %d", layer);
        }
        if (strcmp(state->turbomind_tp2_gate_up_binding[0].shard_file, owner_shard) != 0 ||
            strcmp(state->turbomind_tp2_down_binding[0].shard_file, owner_shard) != 0 ||
            strcmp(state->turbomind_tp2_gate_up_binding[1].shard_file, peer_shard) != 0 ||
            strcmp(state->turbomind_tp2_down_binding[1].shard_file, peer_shard) != 0) {
            return scheduler_errorf(err, errlen, "TP2 routed FFN shard mismatch for layer %d", layer);
        }
    }
    if (!first_state || peer_gpu < 0) {
        return scheduler_error(err, errlen, "TP2 routed FFN span has no local layers");
    }
    if (sched->tp2_requested_peer_gpu >= 0 &&
        sched->tp2_requested_peer_gpu != peer_gpu) {
        return scheduler_errorf(err, errlen, "TP2 routed FFN peer mismatch for layer %d", local_first);
    }
    if (!ds4_gpu_enable_peer_access(sched->stage.gpu, peer_gpu)) {
        return scheduler_errorf(err, errlen, "TP2 peer access enable failed for layer %d", local_first);
    }

    char owner_path[1024];
    char peer_path[1024];
    if (scheduler_join_path(owner_path,
                            sizeof(owner_path),
                            sched->tp2_shard_dir,
                            owner_shard,
                            err,
                            errlen) ||
        scheduler_join_path(peer_path,
                            sizeof(peer_path),
                            sched->tp2_shard_dir,
                            peer_shard,
                            err,
                            errlen)) {
        return 1;
    }
    uint64_t owner_bytes = 0;
    uint64_t peer_bytes = 0;
    if (open_upload_file_arena(&sched->tp2_owner_arena,
                               sched->stage.gpu,
                               owner_path,
                               &owner_bytes,
                               err,
                               errlen) ||
        open_upload_file_arena(&sched->tp2_peer_arena,
                               peer_gpu,
                               peer_path,
                               &peer_bytes,
                               err,
                               errlen)) {
        return 1;
    }
    sched->tp2_uploaded_bytes = owner_bytes + peer_bytes;

    const uint64_t hidden_bytes =
        (uint64_t)sched->active_slots * first_state->hidden_size * sizeof(float);
    const uint64_t route_i32_bytes =
        (uint64_t)sched->active_slots * first_state->routes_per_token * sizeof(int32_t);
    const uint64_t route_f32_bytes =
        (uint64_t)sched->active_slots * first_state->routes_per_token * sizeof(float);
    if (!ds4_gpu_set_device(peer_gpu)) {
        return scheduler_errorf(err, errlen, "failed to set TP2 peer gpu%d", peer_gpu);
    }
    sched->tp2_peer_input = ds4_gpu_tensor_alloc(hidden_bytes);
    sched->tp2_peer_selected = ds4_gpu_tensor_alloc(route_i32_bytes);
    sched->tp2_peer_weights = ds4_gpu_tensor_alloc(route_f32_bytes);
    sched->tp2_peer_out = ds4_gpu_tensor_alloc(hidden_bytes);
    if (!ds4_gpu_set_device(sched->stage.gpu)) {
        return scheduler_errorf(err, errlen, "failed to restore TP2 owner gpu%d", sched->stage.gpu);
    }
    sched->tp2_peer_recv = ds4_gpu_tensor_alloc(hidden_bytes);
    if (!sched->tp2_peer_input ||
        !sched->tp2_peer_selected ||
        !sched->tp2_peer_weights ||
        !sched->tp2_peer_out ||
        !sched->tp2_peer_recv) {
        return scheduler_errorf(err, errlen, "TP2 routed FFN scratch allocation failed for layer %d", local_first);
    }
    sched->tp2_scratch_slots = sched->active_slots;
    return 0;
}

static int alloc_layer_cache(ds4_stage_scheduler *sched,
                             int layer,
                             uint32_t slot,
                             char *err,
                             size_t errlen) {
    const ds4_layer_state *state = &sched->states[layer];
    scheduler_layer_cache *lc = scheduler_cache_slot(sched, layer, slot);
    if (!lc) {
        return scheduler_errorf(err, errlen, "invalid cache slot %d", (int)slot);
    }
    const uint32_t kv_width = state->kv_latent_width;
    const uint32_t raw_cap = sched->raw_cap ? sched->raw_cap : DS4_V100_SWA_ROWS;
    lc->raw_kv = ds4_gpu_tensor_alloc((uint64_t)raw_cap * kv_width * sizeof(float));
    if (!lc->raw_kv) return scheduler_errorf(err, errlen, "raw KV allocation failed for layer %d", layer);
    if (!ds4_gpu_tensor_fill_f32(lc->raw_kv, 0.0f, (uint64_t)raw_cap * kv_width)) {
        return scheduler_errorf(err, errlen, "raw KV init failed for layer %d", layer);
    }
    lc->cache.raw_kv = lc->raw_kv;
    lc->cache.raw_cap = raw_cap;
    lc->cache.raw_window = sched->raw_window ? sched->raw_window : DS4_V100_SWA_ROWS;

    if (state->compress_ratio == 0) return 0;

    const uint32_t coff = state->compress_ratio == 4u ? 2u : 1u;
    const uint32_t attn_state_rows = coff * state->compress_ratio;
    const uint32_t attn_state_width = coff * DS4_V100_HEAD_DIM;
    const uint32_t attn_comp_cap = sched->attn_comp_cap ? sched->attn_comp_cap : 1u;
    lc->attn_state_kv =
        ds4_gpu_tensor_alloc((uint64_t)attn_state_rows * attn_state_width * sizeof(float));
    lc->attn_state_score =
        ds4_gpu_tensor_alloc((uint64_t)attn_state_rows * attn_state_width * sizeof(float));
    lc->attn_comp_kv =
        ds4_gpu_tensor_alloc((uint64_t)attn_comp_cap * kv_width * sizeof(float));
    if (!lc->attn_state_kv || !lc->attn_state_score || !lc->attn_comp_kv) {
        return scheduler_errorf(err, errlen, "attention cache allocation failed for layer %d", layer);
    }
    if (!ds4_gpu_tensor_fill_f32(lc->attn_state_kv,
                                 0.0f,
                                 (uint64_t)attn_state_rows * attn_state_width) ||
        !ds4_gpu_tensor_fill_f32(lc->attn_state_score,
                                 -1.0e30f,
                                 (uint64_t)attn_state_rows * attn_state_width) ||
        !ds4_gpu_tensor_fill_f32(lc->attn_comp_kv,
                                 0.0f,
                                 (uint64_t)attn_comp_cap * kv_width)) {
        return scheduler_errorf(err, errlen, "attention cache init failed for layer %d", layer);
    }
    lc->cache.attn_state_kv = lc->attn_state_kv;
    lc->cache.attn_state_score = lc->attn_state_score;
    lc->cache.attn_comp_kv = lc->attn_comp_kv;
    lc->cache.attn_comp_cap = attn_comp_cap;

    if (state->compress_ratio != 4u) return 0;

    const uint32_t index_state_rows = 2u * state->compress_ratio;
    const uint32_t index_state_width = 2u * DS4_V100_INDEXER_HEAD_DIM;
    const uint32_t index_comp_cap = sched->index_comp_cap ? sched->index_comp_cap : 1u;
    const uint32_t indexer_top_k = sched->indexer_top_k ? sched->indexer_top_k : 1u;
    lc->index_state_kv =
        ds4_gpu_tensor_alloc((uint64_t)index_state_rows * index_state_width * sizeof(float));
    lc->index_state_score =
        ds4_gpu_tensor_alloc((uint64_t)index_state_rows * index_state_width * sizeof(float));
    lc->index_comp_kv =
        ds4_gpu_tensor_alloc((uint64_t)index_comp_cap * DS4_V100_INDEXER_HEAD_DIM * sizeof(float));
    lc->indexer_topk =
        ds4_gpu_tensor_alloc((uint64_t)indexer_top_k * sizeof(uint32_t));
    if (!lc->index_state_kv || !lc->index_state_score || !lc->index_comp_kv || !lc->indexer_topk) {
        return scheduler_errorf(err, errlen, "indexer cache allocation failed for layer %d", layer);
    }
    if (!ds4_gpu_tensor_fill_f32(lc->index_state_kv,
                                 0.0f,
                                 (uint64_t)index_state_rows * index_state_width) ||
        !ds4_gpu_tensor_fill_f32(lc->index_state_score,
                                 -1.0e30f,
                                 (uint64_t)index_state_rows * index_state_width) ||
        !ds4_gpu_tensor_fill_f32(lc->index_comp_kv,
                                 0.0f,
                                 (uint64_t)index_comp_cap * DS4_V100_INDEXER_HEAD_DIM)) {
        return scheduler_errorf(err, errlen, "indexer cache init failed for layer %d", layer);
    }
    lc->cache.index_state_kv = lc->index_state_kv;
    lc->cache.index_state_score = lc->index_state_score;
    lc->cache.index_comp_kv = lc->index_comp_kv;
    lc->cache.index_comp_cap = index_comp_cap;
    lc->cache.indexer_topk = lc->indexer_topk;
    lc->cache.indexer_top_k = indexer_top_k;
    return 0;
}

static int reset_layer_cache(ds4_stage_scheduler *sched,
                             int layer,
                             uint32_t slot,
                             char *err,
                             size_t errlen) {
    const ds4_layer_state *state = &sched->states[layer];
    scheduler_layer_cache *lc = scheduler_cache_slot(sched, layer, slot);
    if (!lc) {
        return scheduler_errorf(err, errlen, "invalid cache slot %d", (int)slot);
    }
    const uint32_t kv_width = state->kv_latent_width;
    const uint32_t raw_cap = sched->raw_cap ? sched->raw_cap : DS4_V100_SWA_ROWS;
    if (!lc->raw_kv ||
        !ds4_gpu_tensor_fill_f32(lc->raw_kv, 0.0f, (uint64_t)raw_cap * kv_width)) {
        return scheduler_errorf(err, errlen, "raw KV reset failed for layer %d", layer);
    }
    lc->cache.n_attn_comp = 0;
    lc->cache.n_index_comp = 0;
    if (state->compress_ratio == 0) return 0;

    const uint32_t coff = state->compress_ratio == 4u ? 2u : 1u;
    const uint32_t attn_state_rows = coff * state->compress_ratio;
    const uint32_t attn_state_width = coff * DS4_V100_HEAD_DIM;
    const uint32_t attn_comp_cap = sched->attn_comp_cap ? sched->attn_comp_cap : 1u;
    if (!lc->attn_state_kv || !lc->attn_state_score || !lc->attn_comp_kv ||
        !ds4_gpu_tensor_fill_f32(lc->attn_state_kv,
                                 0.0f,
                                 (uint64_t)attn_state_rows * attn_state_width) ||
        !ds4_gpu_tensor_fill_f32(lc->attn_state_score,
                                 -1.0e30f,
                                 (uint64_t)attn_state_rows * attn_state_width) ||
        !ds4_gpu_tensor_fill_f32(lc->attn_comp_kv,
                                 0.0f,
                                 (uint64_t)attn_comp_cap * kv_width)) {
        return scheduler_errorf(err, errlen, "attention cache reset failed for layer %d", layer);
    }
    if (state->compress_ratio != 4u) return 0;

    const uint32_t index_state_rows = 2u * state->compress_ratio;
    const uint32_t index_state_width = 2u * DS4_V100_INDEXER_HEAD_DIM;
    const uint32_t index_comp_cap = sched->index_comp_cap ? sched->index_comp_cap : 1u;
    if (!lc->index_state_kv || !lc->index_state_score || !lc->index_comp_kv ||
        !ds4_gpu_tensor_fill_f32(lc->index_state_kv,
                                 0.0f,
                                 (uint64_t)index_state_rows * index_state_width) ||
        !ds4_gpu_tensor_fill_f32(lc->index_state_score,
                                 -1.0e30f,
                                 (uint64_t)index_state_rows * index_state_width) ||
        !ds4_gpu_tensor_fill_f32(lc->index_comp_kv,
                                 0.0f,
                                 (uint64_t)index_comp_cap * DS4_V100_INDEXER_HEAD_DIM)) {
        return scheduler_errorf(err, errlen, "indexer cache reset failed for layer %d", layer);
    }
    return 0;
}

int ds4_stage_scheduler_reset(ds4_stage_scheduler *sched,
                                   char *err,
                                   size_t errlen) {
    if (!sched) return scheduler_error(err, errlen, "missing scheduler reset input");
    if (!ds4_gpu_set_device(sched->stage.gpu)) {
        return scheduler_errorf(err, errlen, "failed to set scheduler reset device gpu%d",
                                sched->stage.gpu);
    }
    for (int layer = sched->stage.layer_begin; layer <= sched->stage.layer_end; layer++) {
        for (uint32_t slot = 0; slot < sched->active_slots; slot++) {
            if (reset_layer_cache(sched, layer, slot, err, errlen)) return 1;
        }
    }
    const uint64_t hc_values = (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS;
    for (uint32_t slot = 0; slot < sched->active_slots; slot++) {
        if (!ds4_gpu_tensor_fill_f32(sched->hc_a[slot], 0.0f, hc_values) ||
            !ds4_gpu_tensor_fill_f32(sched->hc_b[slot], 0.0f, hc_values)) {
            return scheduler_errorf(err, errlen, "scheduler HC reset failed for slot %d",
                                    (int)slot);
        }
        sched->cur_hc[slot] = sched->hc_a[slot];
    }
    return 0;
}
