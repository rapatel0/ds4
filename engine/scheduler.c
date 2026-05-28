#include "engine/scheduler.h"

#include "ds4_pack.h"
#include "ds4_source_formats.h"

#include <float.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    DS4_V100_SCHED_UPLOAD_CHUNK = 8 * 1024 * 1024,
};

typedef struct {
    ds4_v100_layer_decode_cache cache;
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

struct ds4_v100_stage_scheduler_snapshot {
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

struct ds4_v100_stage_scheduler {
    ds4_v100_context *ctx;
    ds4_pack *pack;
    ds4_gpu_arena *arena;
    ds4_gpu_arena *tp2_owner_arena;
    ds4_gpu_arena *tp2_peer_arena;
    ds4_v100_stage_info stage;
    ds4_v100_tensor_binding token_embedding;
    ds4_v100_tensor_binding hc_head_fn;
    ds4_v100_tensor_binding hc_head_base;
    ds4_v100_tensor_binding hc_head_scale;
    ds4_v100_tensor_binding output_norm;
    ds4_v100_tensor_binding output_weight;
    ds4_v100_layer_state states[DS4_V100_N_LAYERS];
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
    ds4_v100_layer_batch_scratch batch_scratch;
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

static int scheduler_hc_finite_error(const ds4_v100_stage_scheduler *sched,
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

static int scheduler_check_hc_finite(const ds4_v100_stage_scheduler *sched,
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

static uint64_t scheduler_model_offset(const ds4_v100_stage_scheduler *sched,
                                       const ds4_v100_tensor_binding *b) {
    if (!b) return 0;
    return (sched && sched->model_map_uses_shard_offsets) ? b->shard_offset : b->source_offset;
}

static int scheduler_activate_model_source(const ds4_v100_stage_scheduler *sched,
                                           char *err,
                                           size_t errlen) {
    if (!sched || sched->shard_fd < 0) return 0;
    if (!ds4_gpu_set_model_fd(sched->shard_fd)) {
        return scheduler_errorf(err, errlen, "failed to activate shard fd for gpu%d",
                                sched->stage.gpu);
    }
    return 0;
}

static scheduler_layer_cache *scheduler_cache_slot(ds4_v100_stage_scheduler *sched,
                                                   int layer,
                                                   uint32_t slot) {
    if (!sched || layer < 0 || layer >= DS4_V100_N_LAYERS ||
        slot >= DS4_V100_SCHED_MAX_SLOTS) {
        return NULL;
    }
    return &sched->caches[(size_t)layer * DS4_V100_SCHED_MAX_SLOTS + slot];
}

static const scheduler_layer_cache *scheduler_cache_slot_const(
    const ds4_v100_stage_scheduler *sched,
    int layer,
    uint32_t slot) {
    if (!sched || layer < 0 || layer >= DS4_V100_N_LAYERS ||
        slot >= DS4_V100_SCHED_MAX_SLOTS) {
        return NULL;
    }
    return &sched->caches[(size_t)layer * DS4_V100_SCHED_MAX_SLOTS + slot];
}

typedef struct {
    const ds4_v100_stage_scheduler *sched;
    ds4_v100_stage_scheduler_checkpoint_fn checkpoint_fn;
    void *checkpoint_user;
    uint32_t token;
    uint32_t position;
} scheduler_layer_checkpoint_user;

static int scheduler_layer_checkpoint(
    const ds4_v100_layer_execute_checkpoint *layer_cp,
    void *user,
    char *err,
    size_t errlen) {
    scheduler_layer_checkpoint_user *u = (scheduler_layer_checkpoint_user *)user;
    if (!u || !u->sched || !u->checkpoint_fn || !layer_cp) {
        return scheduler_error(err, errlen, "missing scheduler layer checkpoint input");
    }
    ds4_v100_stage_scheduler_checkpoint cp = {
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

void ds4_v100_stage_scheduler_options_init(ds4_v100_stage_scheduler_options *opts) {
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

static void free_output_head_scratch(ds4_v100_stage_scheduler *sched) {
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

void ds4_v100_stage_scheduler_close(ds4_v100_stage_scheduler *sched) {
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
    ds4_v100_layer_batch_scratch_free(&sched->batch_scratch);
    ds4_gpu_arena_close(sched->tp2_peer_arena);
    ds4_gpu_arena_close(sched->tp2_owner_arena);
    ds4_gpu_arena_close(sched->arena);
    if (sched->shard_map && sched->shard_map != MAP_FAILED) {
        munmap(sched->shard_map, (size_t)sched->shard_size);
    }
    if (sched->shard_fd >= 0) close(sched->shard_fd);
    ds4_pack_close(sched->pack);
    ds4_v100_context_close(sched->ctx);
    free(sched);
}

typedef struct {
    ds4_v100_stage_scheduler *sched;
    const unsigned char *model;
    unsigned char *chunk;
    char *err;
    size_t errlen;
} upload_stage_ud;

static int map_stage_shard(ds4_v100_stage_scheduler *sched,
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
    ds4_v100_stage_scheduler *sched = ud->sched;
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

static int upload_stage_shard(ds4_v100_stage_scheduler *sched,
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

static int upload_stage_weights(ds4_v100_stage_scheduler *sched,
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

static int scheduler_load_tp2_env(ds4_v100_stage_scheduler *sched,
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
        scheduler_env_enabled("DS4_V100_TP_EP_ROUTED_FFN") ||
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

static int scheduler_setup_tp2_overlay(ds4_v100_stage_scheduler *sched,
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

    ds4_v100_layer_state *first_state = NULL;
    int peer_gpu = -1;
    const char *owner_shard = NULL;
    const char *peer_shard = NULL;
    for (int layer = local_first; layer <= local_last; layer++) {
        ds4_v100_layer_state *state = &sched->states[layer];
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

static int alloc_layer_cache(ds4_v100_stage_scheduler *sched,
                             int layer,
                             uint32_t slot,
                             char *err,
                             size_t errlen) {
    const ds4_v100_layer_state *state = &sched->states[layer];
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

static int reset_layer_cache(ds4_v100_stage_scheduler *sched,
                             int layer,
                             uint32_t slot,
                             char *err,
                             size_t errlen) {
    const ds4_v100_layer_state *state = &sched->states[layer];
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

int ds4_v100_stage_scheduler_reset(ds4_v100_stage_scheduler *sched,
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

static void snapshot_tensor_free(scheduler_snapshot_tensor *snap) {
    if (!snap) return;
    free(snap->data);
    memset(snap, 0, sizeof(*snap));
}

static int snapshot_tensor_capture(scheduler_snapshot_tensor *snap,
                                   const ds4_gpu_tensor *tensor,
                                   uint64_t *captured_bytes,
                                   char *err,
                                   size_t errlen) {
    if (!snap) return scheduler_error(err, errlen, "missing snapshot tensor");
    memset(snap, 0, sizeof(*snap));
    if (!tensor) return 0;
    const uint64_t bytes = ds4_gpu_tensor_bytes(tensor);
    if (bytes == 0) return 0;
    if (bytes > (uint64_t)SIZE_MAX) {
        return scheduler_errorf_u64(err, errlen, "snapshot tensor too large: %" PRIu64, bytes);
    }
    snap->data = (unsigned char *)malloc((size_t)bytes);
    if (!snap->data) {
        return scheduler_errorf_u64(err, errlen, "failed to allocate snapshot tensor bytes=%" PRIu64, bytes);
    }
    snap->bytes = bytes;
    if (!ds4_gpu_tensor_read(tensor, 0, snap->data, bytes)) {
        snapshot_tensor_free(snap);
        return scheduler_error(err, errlen, "snapshot tensor read failed");
    }
    if (captured_bytes) *captured_bytes += bytes;
    return 0;
}

static int snapshot_tensor_restore(const scheduler_snapshot_tensor *snap,
                                   ds4_gpu_tensor *tensor,
                                   char *err,
                                   size_t errlen) {
    if (!snap || snap->bytes == 0) return 0;
    if (!tensor || ds4_gpu_tensor_bytes(tensor) != snap->bytes) {
        return scheduler_error(err, errlen, "snapshot tensor restore size mismatch");
    }
    if (!ds4_gpu_tensor_write(tensor, 0, snap->data, snap->bytes)) {
        return scheduler_error(err, errlen, "snapshot tensor write failed");
    }
    return 0;
}

static void layer_snapshot_free(scheduler_layer_cache_snapshot *snap) {
    if (!snap) return;
    snapshot_tensor_free(&snap->raw_kv);
    snapshot_tensor_free(&snap->attn_state_kv);
    snapshot_tensor_free(&snap->attn_state_score);
    snapshot_tensor_free(&snap->attn_comp_kv);
    snapshot_tensor_free(&snap->index_state_kv);
    snapshot_tensor_free(&snap->index_state_score);
    snapshot_tensor_free(&snap->index_comp_kv);
    snapshot_tensor_free(&snap->indexer_topk);
    memset(snap, 0, sizeof(*snap));
}

int ds4_v100_stage_scheduler_snapshot_create(
    const ds4_v100_stage_scheduler *sched,
    ds4_v100_stage_scheduler_snapshot **out,
    char *err,
    size_t errlen) {
    if (!out) return scheduler_error(err, errlen, "missing scheduler snapshot output");
    *out = NULL;
    if (!sched || !sched->cur_hc[0]) {
        return scheduler_error(err, errlen, "missing scheduler snapshot input");
    }
    if (sched->active_slots != 1) {
        return scheduler_errorf_u32(err,
                                    errlen,
                                    "scheduler snapshot currently requires active_slots=1, got %u",
                                    sched->active_slots);
    }
    if (!ds4_gpu_set_device(sched->stage.gpu)) {
        return scheduler_errorf(err, errlen, "failed to set scheduler snapshot device gpu%d",
                                sched->stage.gpu);
    }

    ds4_v100_stage_scheduler_snapshot *snap =
        (ds4_v100_stage_scheduler_snapshot *)calloc(1, sizeof(*snap));
    if (!snap) return scheduler_error(err, errlen, "failed to allocate scheduler snapshot");
    snap->stage_id = sched->stage.stage_id;
    snap->gpu = sched->stage.gpu;
    snap->layer_begin = sched->stage.layer_begin;
    snap->layer_end = sched->stage.layer_end;
    snap->active_slots = sched->active_slots;
    snap->cur_hc_slot = sched->cur_hc[0] == sched->hc_b[0] ? 1 : 0;

    if (snapshot_tensor_capture(&snap->cur_hc,
                                sched->cur_hc[0],
                                &snap->captured_bytes,
                                err,
                                errlen)) {
        ds4_v100_stage_scheduler_snapshot_free(snap);
        return 1;
    }

    for (int layer = sched->stage.layer_begin; layer <= sched->stage.layer_end; layer++) {
        const scheduler_layer_cache *lc = scheduler_cache_slot_const(sched, layer, 0);
        scheduler_layer_cache_snapshot *ls = &snap->layers[layer];
        if (!lc) {
            ds4_v100_stage_scheduler_snapshot_free(snap);
            return scheduler_error(err, errlen, "snapshot cache lookup failed");
        }
        ls->n_attn_comp = lc->cache.n_attn_comp;
        ls->n_index_comp = lc->cache.n_index_comp;
        if (snapshot_tensor_capture(&ls->raw_kv, lc->raw_kv, &snap->captured_bytes, err, errlen) ||
            snapshot_tensor_capture(&ls->attn_state_kv, lc->attn_state_kv, &snap->captured_bytes, err, errlen) ||
            snapshot_tensor_capture(&ls->attn_state_score, lc->attn_state_score, &snap->captured_bytes, err, errlen) ||
            snapshot_tensor_capture(&ls->attn_comp_kv, lc->attn_comp_kv, &snap->captured_bytes, err, errlen) ||
            snapshot_tensor_capture(&ls->index_state_kv, lc->index_state_kv, &snap->captured_bytes, err, errlen) ||
            snapshot_tensor_capture(&ls->index_state_score, lc->index_state_score, &snap->captured_bytes, err, errlen) ||
            snapshot_tensor_capture(&ls->index_comp_kv, lc->index_comp_kv, &snap->captured_bytes, err, errlen) ||
            snapshot_tensor_capture(&ls->indexer_topk, lc->indexer_topk, &snap->captured_bytes, err, errlen)) {
            ds4_v100_stage_scheduler_snapshot_free(snap);
            return 1;
        }
    }
    *out = snap;
    return 0;
}

int ds4_v100_stage_scheduler_snapshot_restore(
    ds4_v100_stage_scheduler *sched,
    const ds4_v100_stage_scheduler_snapshot *snap,
    char *err,
    size_t errlen) {
    if (!sched || !snap) return scheduler_error(err, errlen, "missing scheduler snapshot restore input");
    if (sched->stage.stage_id != snap->stage_id ||
        sched->stage.gpu != snap->gpu ||
        sched->stage.layer_begin != snap->layer_begin ||
        sched->stage.layer_end != snap->layer_end) {
        return scheduler_error(err, errlen, "scheduler snapshot restore stage mismatch");
    }
    if (sched->active_slots != snap->active_slots || sched->active_slots != 1) {
        return scheduler_error(err, errlen, "scheduler snapshot restore active-slot mismatch");
    }
    if (!ds4_gpu_set_device(sched->stage.gpu)) {
        return scheduler_errorf(err, errlen, "failed to set scheduler restore device gpu%d",
                                sched->stage.gpu);
    }
    ds4_gpu_tensor *hc = snap->cur_hc_slot ? sched->hc_b[0] : sched->hc_a[0];
    if (snapshot_tensor_restore(&snap->cur_hc, hc, err, errlen)) return 1;
    sched->cur_hc[0] = hc;

    for (int layer = sched->stage.layer_begin; layer <= sched->stage.layer_end; layer++) {
        scheduler_layer_cache *lc = scheduler_cache_slot(sched, layer, 0);
        const scheduler_layer_cache_snapshot *ls = &snap->layers[layer];
        if (!lc) return scheduler_error(err, errlen, "snapshot restore cache lookup failed");
        lc->cache.n_attn_comp = ls->n_attn_comp;
        lc->cache.n_index_comp = ls->n_index_comp;
        if (snapshot_tensor_restore(&ls->raw_kv, lc->raw_kv, err, errlen) ||
            snapshot_tensor_restore(&ls->attn_state_kv, lc->attn_state_kv, err, errlen) ||
            snapshot_tensor_restore(&ls->attn_state_score, lc->attn_state_score, err, errlen) ||
            snapshot_tensor_restore(&ls->attn_comp_kv, lc->attn_comp_kv, err, errlen) ||
            snapshot_tensor_restore(&ls->index_state_kv, lc->index_state_kv, err, errlen) ||
            snapshot_tensor_restore(&ls->index_state_score, lc->index_state_score, err, errlen) ||
            snapshot_tensor_restore(&ls->index_comp_kv, lc->index_comp_kv, err, errlen) ||
            snapshot_tensor_restore(&ls->indexer_topk, lc->indexer_topk, err, errlen)) {
            return 1;
        }
    }
    return 0;
}

uint64_t ds4_v100_stage_scheduler_snapshot_bytes(
    const ds4_v100_stage_scheduler_snapshot *snapshot) {
    return snapshot ? snapshot->captured_bytes : 0;
}

void ds4_v100_stage_scheduler_snapshot_free(
    ds4_v100_stage_scheduler_snapshot *snapshot) {
    if (!snapshot) return;
    snapshot_tensor_free(&snapshot->cur_hc);
    for (int layer = 0; layer < DS4_V100_N_LAYERS; layer++) {
        layer_snapshot_free(&snapshot->layers[layer]);
    }
    free(snapshot);
}

int ds4_v100_stage_scheduler_open(ds4_v100_stage_scheduler **out,
                                  const ds4_v100_stage_scheduler_options *opts,
                                  char *err,
                                  size_t errlen) {
    if (!out) return scheduler_error(err, errlen, "missing scheduler output");
    *out = NULL;
    if (!opts || !opts->pack_index_path ||
        (!opts->shard_dir && (!opts->model_map || opts->model_size == 0))) {
        return scheduler_error(err, errlen, "missing scheduler options");
    }
    if (opts->turbomind_pack_index_path && !opts->shard_dir) {
        return scheduler_error(err, errlen,
                               "TurboMind appliance scheduling requires shard_dir");
    }
    if (opts->stage_id < 0 || opts->stage_id >= DS4_V100_EXPECTED_GPUS) {
        return scheduler_errorf(err, errlen, "invalid scheduler stage %d", opts->stage_id);
    }

    ds4_v100_stage_scheduler *sched =
        (ds4_v100_stage_scheduler *)calloc(1, sizeof(*sched));
    if (!sched) return scheduler_error(err, errlen, "failed to allocate scheduler");
    sched->shard_fd = -1;
    sched->tp2_layer = -1;
    sched->tp2_layer_count = 1;
    sched->tp2_requested_peer_gpu = -1;
    if (scheduler_load_tp2_env(sched, err, errlen)) {
        ds4_v100_stage_scheduler_close(sched);
        return 1;
    }
    sched->active_slots = (uint32_t)(opts->kv_active_slots ? opts->kv_active_slots : 1u);
    if (sched->active_slots == 0 || sched->active_slots > DS4_V100_SCHED_MAX_SLOTS) {
        ds4_v100_stage_scheduler_close(sched);
        return scheduler_errorf_u32(err,
                                    errlen,
                                    "kv_active_slots must be in [1,%u]",
                                    DS4_V100_SCHED_MAX_SLOTS);
    }
    sched->model_map = opts->model_map;
    sched->model_size = opts->model_size;
    sched->model_map_uses_shard_offsets = false;
    sched->raw_cap = opts->raw_cap ? opts->raw_cap : DS4_V100_SWA_ROWS;
    sched->raw_window = opts->raw_window ? opts->raw_window : DS4_V100_SWA_ROWS;
    sched->attn_comp_cap = opts->attn_comp_cap ? opts->attn_comp_cap : 1u;
    sched->index_comp_cap = opts->index_comp_cap ? opts->index_comp_cap : 1u;
    sched->indexer_top_k = opts->indexer_top_k ? opts->indexer_top_k : 1u;
    sched->fp8_kv_cache = opts->fp8_kv_cache;
    sched->suppress_router_readback = opts->suppress_router_readback;
    ds4_v100_layer_batch_scratch_init(&sched->batch_scratch);

    ds4_v100_context_options ctx_opts;
    ds4_v100_context_options_init(&ctx_opts);
    ctx_opts.pack_index_path = opts->pack_index_path;
    ctx_opts.turbomind_pack_index_path = opts->turbomind_pack_index_path;
    ctx_opts.kv_ctx_tokens = opts->kv_ctx_tokens ? opts->kv_ctx_tokens : 1048576;
    ctx_opts.kv_active_slots = sched->active_slots;
    if (ds4_v100_context_open(&sched->ctx, &ctx_opts, err, errlen) ||
        ds4_pack_open(&sched->pack, opts->pack_index_path, err, errlen)) {
        ds4_v100_stage_scheduler_close(sched);
        return 1;
    }

    const ds4_v100_stage_info *stage =
        ds4_v100_context_stage(sched->ctx, opts->stage_id);
    if (!stage) {
        ds4_v100_stage_scheduler_close(sched);
        return scheduler_errorf(err, errlen, "missing context stage %d", opts->stage_id);
    }
    sched->stage = *stage;
    if (opts->shard_dir && map_stage_shard(sched, opts->shard_dir, err, errlen)) {
        ds4_v100_stage_scheduler_close(sched);
        return 1;
    }
    if (sched->stage.owns_token_embedding &&
        ds4_v100_context_lookup_tensor_binding(sched->ctx,
                                               "token_embd.weight",
                                               &sched->token_embedding,
                                               err,
                                               errlen)) {
        ds4_v100_stage_scheduler_close(sched);
        return 1;
    }
    if (sched->stage.owns_output_head) {
        if (ds4_v100_context_lookup_tensor_binding(sched->ctx,
                                                   "hc_head_fn",
                                                   &sched->hc_head_fn,
                                                   err,
                                                   errlen) ||
            ds4_v100_context_lookup_tensor_binding(sched->ctx,
                                                   "hc_head_base",
                                                   &sched->hc_head_base,
                                                   err,
                                                   errlen) ||
            ds4_v100_context_lookup_tensor_binding(sched->ctx,
                                                   "hc_head_scale",
                                                   &sched->hc_head_scale,
                                                   err,
                                                   errlen) ||
            ds4_v100_context_lookup_tensor_binding(sched->ctx,
                                                   "output_norm.weight",
                                                   &sched->output_norm,
                                                   err,
                                                   errlen) ||
            ds4_v100_context_output_head_binding(sched->ctx,
                                                 &sched->output_weight,
                                                 err,
                                                 errlen)) {
            ds4_v100_stage_scheduler_close(sched);
            return 1;
        }
    }

    uint64_t arena_bytes = sched->stage.arena_bytes;
    if (sched->shard_size > arena_bytes) arena_bytes = sched->shard_size;
    sched->stage.arena_bytes = arena_bytes;
    if (arena_bytes == 0 ||
        ds4_gpu_arena_open(&sched->arena, sched->stage.gpu, arena_bytes) != 0) {
        ds4_v100_stage_scheduler_close(sched);
        return scheduler_errorf(err, errlen, "failed to open resident arena for gpu%d",
                                sched->stage.gpu);
    }
    if (upload_stage_weights(sched, err, errlen)) {
        ds4_v100_stage_scheduler_close(sched);
        return 1;
    }
    if (!ds4_gpu_set_device(sched->stage.gpu)) {
        ds4_v100_stage_scheduler_close(sched);
        return scheduler_errorf(err, errlen, "failed to set scheduler device gpu%d",
                                sched->stage.gpu);
    }

    for (int layer = sched->stage.layer_begin; layer <= sched->stage.layer_end; layer++) {
        if (ds4_v100_layer_state_init(&sched->states[layer], sched->ctx, layer, err, errlen)) {
            ds4_v100_stage_scheduler_close(sched);
            return 1;
        }
        for (uint32_t slot = 0; slot < sched->active_slots; slot++) {
            if (alloc_layer_cache(sched, layer, slot, err, errlen)) {
                ds4_v100_stage_scheduler_close(sched);
                return 1;
            }
        }
    }

    if (scheduler_setup_tp2_overlay(sched, err, errlen)) {
        ds4_v100_stage_scheduler_close(sched);
        return 1;
    }

    const uint64_t hc_bytes =
        (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS * sizeof(float);
    for (uint32_t slot = 0; slot < sched->active_slots; slot++) {
        sched->hc_a[slot] = ds4_gpu_tensor_alloc(hc_bytes);
        sched->hc_b[slot] = ds4_gpu_tensor_alloc(hc_bytes);
        if (!sched->hc_a[slot] || !sched->hc_b[slot]) {
            ds4_v100_stage_scheduler_close(sched);
            return scheduler_errorf(err, errlen, "failed to allocate scheduler HC slot %d",
                                    (int)slot);
        }
        sched->cur_hc[slot] = sched->hc_a[slot];
    }
    *out = sched;
    return 0;
}

static int scheduler_validate_slot_span_args(const ds4_v100_stage_scheduler *sched,
                                             const uint32_t *tokens,
                                             const uint32_t *positions,
                                             uint32_t slot_start,
                                             uint32_t n_slots,
                                             char *err,
                                             size_t errlen) {
    if (!sched) return scheduler_error(err, errlen, "missing scheduler");
    if (!tokens || !positions) return scheduler_error(err, errlen, "missing scheduler batch inputs");
    if (slot_start >= sched->active_slots) {
        return scheduler_errorf_u32(err,
                                    errlen,
                                    "slot start must be in [0,active_slots), got %u",
                                    slot_start);
    }
    if (n_slots == 0 || n_slots > sched->active_slots - slot_start) {
        return scheduler_errorf_u32(err,
                                    errlen,
                                    "batch slot span exceeds active slots, got %u",
                                    n_slots);
    }
    if (n_slots > DS4_V100_SCHED_MAX_SLOTS ||
        slot_start > DS4_V100_SCHED_MAX_SLOTS ||
        n_slots > DS4_V100_SCHED_MAX_SLOTS - slot_start) {
        return scheduler_errorf_u32(err,
                                    errlen,
                                    "batch slots exceed scheduler max: %u",
                                    n_slots);
    }
    return 0;
}

static bool scheduler_use_layer_batch(void) {
    const char *env = getenv("DS4_V100_BATCH_LAYER_FFN");
    return !(env && (strcmp(env, "0") == 0 ||
                     strcmp(env, "off") == 0 ||
                     strcmp(env, "false") == 0));
}

static int scheduler_validate_layer_span_args(
    const ds4_v100_stage_scheduler *sched,
    int first_layer,
    int last_layer,
    char *err,
    size_t errlen) {
    if (!sched) return scheduler_error(err, errlen, "missing scheduler");
    if (first_layer > last_layer) {
        if (err && errlen) {
            snprintf(err,
                     errlen,
                     "invalid layer span [%d,%d]",
                     first_layer,
                     last_layer);
        }
        return 1;
    }
    if (first_layer < sched->stage.layer_begin ||
        last_layer > sched->stage.layer_end) {
        if (err && errlen) {
            snprintf(err,
                     errlen,
                     "layer span [%d,%d] outside stage range [%d,%d]",
                     first_layer,
                     last_layer,
                     sched->stage.layer_begin,
                     sched->stage.layer_end);
        }
        return 1;
    }
    return 0;
}

int ds4_v100_stage_scheduler_decode_hc_batch(
    ds4_v100_stage_scheduler *sched,
    const uint32_t *tokens,
    const uint32_t *positions,
    uint32_t n_slots,
    ds4_v100_stage_scheduler_report *reports,
    char *err,
    size_t errlen) {
    return ds4_v100_stage_scheduler_decode_hc_slot_span(
        sched, 0, tokens, positions, n_slots, reports, err, errlen);
}

int ds4_v100_stage_scheduler_decode_hc_slot_span(
    ds4_v100_stage_scheduler *sched,
    uint32_t slot_start,
    const uint32_t *tokens,
    const uint32_t *positions,
    uint32_t n_slots,
    ds4_v100_stage_scheduler_report *reports,
    char *err,
    size_t errlen) {
    if (!sched) return scheduler_error(err, errlen, "missing scheduler");
    return ds4_v100_stage_scheduler_decode_hc_layer_span(
        sched,
        slot_start,
        tokens,
        positions,
        n_slots,
        sched->stage.layer_begin,
        sched->stage.layer_end,
        reports,
        err,
        errlen);
}

int ds4_v100_stage_scheduler_decode_hc_layer_span(
    ds4_v100_stage_scheduler *sched,
    uint32_t slot_start,
    const uint32_t *tokens,
    const uint32_t *positions,
    uint32_t n_slots,
    int first_layer,
    int last_layer,
    ds4_v100_stage_scheduler_report *reports,
    char *err,
    size_t errlen) {
    if (scheduler_validate_slot_span_args(
            sched, tokens, positions, slot_start, n_slots, err, errlen)) {
        return 1;
    }
    if (scheduler_validate_layer_span_args(sched, first_layer, last_layer, err, errlen)) {
        return 1;
    }
    if (!ds4_gpu_set_device(sched->stage.gpu)) {
        return scheduler_errorf(err, errlen, "failed to set scheduler device gpu%d",
                                sched->stage.gpu);
    }
    if (scheduler_activate_model_source(sched, err, errlen)) return 1;

    ds4_gpu_tensor *cur[DS4_V100_SCHED_MAX_SLOTS];
    ds4_gpu_tensor *next[DS4_V100_SCHED_MAX_SLOTS];
    ds4_v100_layer_execute_report last[DS4_V100_SCHED_MAX_SLOTS];
    double timing_hc_attn_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_attention_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_attn_proj_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_attn_cache_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_attn_softmax_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_attn_inverse_rope_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_attn_output_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_hc_ffn_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_ffn_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_hc_final_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_total_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_tp2_copy_in_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_tp2_owner_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_tp2_peer_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_tp2_copy_out_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_tp2_reduce_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    double timing_tp2_total_ms[DS4_V100_SCHED_MAX_SLOTS] = {0};
    uint32_t turbomind_layers[DS4_V100_SCHED_MAX_SLOTS] = {0};
    uint32_t turbomind_tp2_layers[DS4_V100_SCHED_MAX_SLOTS] = {0};
    memset(last, 0, sizeof(last));
    for (uint32_t rel = 0; rel < n_slots; rel++) {
        const uint32_t slot = slot_start + rel;
        if (!sched->cur_hc[slot]) {
            return scheduler_errorf(err, errlen, "missing scheduler HC input for slot %d",
                                    (int)slot);
        }
        cur[rel] = sched->cur_hc[slot];
        next[rel] = cur[rel] == sched->hc_a[slot] ? sched->hc_b[slot] : sched->hc_a[slot];
    }

    uint32_t executed = 0;
    const bool use_layer_batch = scheduler_use_layer_batch();
    const bool debug_hc_finite = scheduler_debug_hc_finite_layer_checks_enabled();
    for (int layer = first_layer; layer <= last_layer; layer++) {
        ds4_v100_layer_execute_config cfgs[DS4_V100_SCHED_MAX_SLOTS];
        const ds4_gpu_tensor *hidden_hc[DS4_V100_SCHED_MAX_SLOTS];
        ds4_gpu_tensor *next_hc[DS4_V100_SCHED_MAX_SLOTS];
        for (uint32_t rel = 0; rel < n_slots; rel++) {
            const uint32_t slot = slot_start + rel;
            scheduler_layer_cache *lc = scheduler_cache_slot(sched, layer, slot);
            if (!lc) {
                return scheduler_errorf(err, errlen, "missing decode cache for layer %d",
                                        layer);
            }
            cfgs[rel] = (ds4_v100_layer_execute_config) {
                .model_map = sched->model_map,
                .model_size = sched->model_size,
                .model_map_uses_shard_offsets = sched->model_map_uses_shard_offsets,
                .arena = sched->arena,
                .batch_scratch = use_layer_batch ? &sched->batch_scratch : NULL,
                .router_token = tokens[rel],
                .position = positions[rel],
                .decode_cache = &lc->cache,
                .fp8_kv_cache = sched->fp8_kv_cache,
                .suppress_router_readback = sched->suppress_router_readback,
                .tp2_layer = sched->tp2_layer,
                .tp2_layer_count = sched->tp2_layer_count,
                .tp2_owner_arena = sched->tp2_owner_arena,
                .tp2_peer_arena = sched->tp2_peer_arena,
                .tp2_peer_input = sched->tp2_peer_input,
                .tp2_peer_selected = sched->tp2_peer_selected,
                .tp2_peer_weights = sched->tp2_peer_weights,
                .tp2_peer_out = sched->tp2_peer_out,
                .tp2_peer_recv = sched->tp2_peer_recv,
                .tp2_scratch_slots = sched->tp2_scratch_slots,
            };
            memset(&last[rel], 0, sizeof(last[rel]));
            hidden_hc[rel] = cur[rel];
            next_hc[rel] = next[rel];
        }
        if (use_layer_batch && n_slots > 1) {
            if (ds4_v100_layer_execute_hc_decode_batch(&sched->states[layer],
                                                       cfgs,
                                                       hidden_hc,
                                                       next_hc,
                                                       n_slots,
                                                       last,
                                                       err,
                                                       errlen)) {
                if (err && errlen) {
                    if (err[0]) {
                        char inner[256];
                        snprintf(inner, sizeof(inner), "%s", err);
                        snprintf(err, errlen, "batch layer %d decode failed: %s", layer, inner);
                    } else {
                        scheduler_errorf(err, errlen, "batch layer decode failed at layer %d", layer);
                    }
                }
                return 1;
            }
        } else {
            for (uint32_t rel = 0; rel < n_slots; rel++) {
                if (ds4_v100_layer_execute_hc_decode(&sched->states[layer],
                                                     &cfgs[rel],
                                                     cur[rel],
                                                     next[rel],
                                                     &last[rel],
                                                     err,
                                                     errlen)) {
                    if (err && errlen) {
                        if (err[0]) {
                            char inner[256];
                            snprintf(inner, sizeof(inner), "%s", err);
                            snprintf(err, errlen, "layer %d decode failed: %s", layer, inner);
                        } else {
                            scheduler_errorf(err, errlen, "layer decode failed at layer %d", layer);
                        }
                    }
                    return 1;
                }
            }
        }
        for (uint32_t rel = 0; rel < n_slots; rel++) {
            timing_hc_attn_ms[rel] += last[rel].timing_hc_attn_ms;
            timing_attention_ms[rel] += last[rel].timing_attention_ms;
            timing_attn_proj_ms[rel] += last[rel].timing_attn_proj_ms;
            timing_attn_cache_ms[rel] += last[rel].timing_attn_cache_ms;
            timing_attn_softmax_ms[rel] += last[rel].timing_attn_softmax_ms;
            timing_attn_inverse_rope_ms[rel] += last[rel].timing_attn_inverse_rope_ms;
            timing_attn_output_ms[rel] += last[rel].timing_attn_output_ms;
            timing_hc_ffn_ms[rel] += last[rel].timing_hc_ffn_ms;
            timing_ffn_ms[rel] += last[rel].timing_ffn_ms;
            timing_hc_final_ms[rel] += last[rel].timing_hc_final_ms;
            timing_total_ms[rel] += last[rel].timing_total_ms;
            timing_tp2_copy_in_ms[rel] += last[rel].timing_tp2_copy_in_ms;
            timing_tp2_owner_ms[rel] += last[rel].timing_tp2_owner_ms;
            timing_tp2_peer_ms[rel] += last[rel].timing_tp2_peer_ms;
            timing_tp2_copy_out_ms[rel] += last[rel].timing_tp2_copy_out_ms;
            timing_tp2_reduce_ms[rel] += last[rel].timing_tp2_reduce_ms;
            timing_tp2_total_ms[rel] += last[rel].timing_tp2_total_ms;
            turbomind_layers[rel] += last[rel].turbomind_routed ? 1u : 0u;
            turbomind_tp2_layers[rel] += last[rel].turbomind_tp2_routed ? 1u : 0u;
            ds4_gpu_tensor *tmp = cur[rel];
            cur[rel] = next[rel];
            next[rel] = tmp;
            if (debug_hc_finite &&
                scheduler_check_hc_finite(sched,
                                          cur[rel],
                                          use_layer_batch && n_slots > 1 ? "decode-layer-batch" : "decode-layer-slot",
                                          layer,
                                          slot_start + rel,
                                          tokens[rel],
                                          positions[rel],
                                          err,
                                          errlen)) {
                return 1;
            }
        }
        executed++;
    }
    for (uint32_t rel = 0; rel < n_slots; rel++) {
        const uint32_t slot = slot_start + rel;
        sched->cur_hc[slot] = cur[rel];
        if (reports) {
            ds4_v100_stage_scheduler_report *r = &reports[rel];
            memset(r, 0, sizeof(*r));
            r->stage_id = sched->stage.stage_id;
            r->gpu = sched->stage.gpu;
            r->first_layer = first_layer;
            r->last_layer = last_layer;
            r->layers_executed = executed;
            r->position = positions[rel];
            r->token = tokens[rel];
            r->arena_bytes = ds4_gpu_arena_bytes(sched->arena);
            r->uploaded_tensors = sched->uploaded_tensors;
            r->uploaded_bytes = sched->uploaded_bytes;
            r->last_layer_report = last[rel];
            r->timing_tp2_copy_in_ms = timing_tp2_copy_in_ms[rel];
            r->timing_tp2_owner_ms = timing_tp2_owner_ms[rel];
            r->timing_tp2_peer_ms = timing_tp2_peer_ms[rel];
            r->timing_tp2_copy_out_ms = timing_tp2_copy_out_ms[rel];
            r->timing_tp2_reduce_ms = timing_tp2_reduce_ms[rel];
            r->timing_tp2_total_ms = timing_tp2_total_ms[rel];
            r->turbomind_routed_layers_executed = turbomind_layers[rel];
            r->turbomind_tp2_routed_layers_executed = turbomind_tp2_layers[rel];
            r->timing_hc_attn_ms = timing_hc_attn_ms[rel];
            r->timing_attention_ms = timing_attention_ms[rel];
            r->timing_attn_proj_ms = timing_attn_proj_ms[rel];
            r->timing_attn_cache_ms = timing_attn_cache_ms[rel];
            r->timing_attn_softmax_ms = timing_attn_softmax_ms[rel];
            r->timing_attn_inverse_rope_ms = timing_attn_inverse_rope_ms[rel];
            r->timing_attn_output_ms = timing_attn_output_ms[rel];
            r->timing_hc_ffn_ms = timing_hc_ffn_ms[rel];
            r->timing_ffn_ms = timing_ffn_ms[rel];
            r->timing_hc_final_ms = timing_hc_final_ms[rel];
            r->timing_total_ms = timing_total_ms[rel];
        }
    }
    return 0;
}

int ds4_v100_stage_scheduler_decode_hc_ffn_microbatch_layer(
    ds4_v100_stage_scheduler *sched,
    uint32_t slot_start,
    const uint32_t *tokens,
    const uint32_t *positions,
    uint32_t n_slots,
    int layer,
    ds4_v100_stage_scheduler_report *reports,
    char *err,
    size_t errlen) {
    if (scheduler_validate_slot_span_args(
            sched, tokens, positions, slot_start, n_slots, err, errlen)) {
        return 1;
    }
    if (scheduler_validate_layer_span_args(sched, layer, layer, err, errlen)) {
        return 1;
    }
    if (!ds4_gpu_set_device(sched->stage.gpu)) {
        return scheduler_errorf(err, errlen, "failed to set scheduler device gpu%d",
                                sched->stage.gpu);
    }
    if (scheduler_activate_model_source(sched, err, errlen)) return 1;

    ds4_gpu_tensor *cur[DS4_V100_SCHED_MAX_SLOTS];
    ds4_gpu_tensor *next[DS4_V100_SCHED_MAX_SLOTS];
    ds4_v100_layer_execute_config cfgs[DS4_V100_SCHED_MAX_SLOTS];
    ds4_v100_layer_prepared_ffn prepared[DS4_V100_SCHED_MAX_SLOTS];
    ds4_v100_layer_execute_report layer_reports[DS4_V100_SCHED_MAX_SLOTS];
    memset(prepared, 0, sizeof(prepared));
    memset(layer_reports, 0, sizeof(layer_reports));

    for (uint32_t rel = 0; rel < n_slots; rel++) {
        const uint32_t slot = slot_start + rel;
        scheduler_layer_cache *lc = scheduler_cache_slot(sched, layer, slot);
        if (!lc) {
            return scheduler_errorf(err, errlen, "missing decode cache for layer %d",
                                    layer);
        }
        if (!sched->cur_hc[slot]) {
            return scheduler_errorf(err, errlen, "missing scheduler HC input for slot %d",
                                    (int)slot);
        }
        cur[rel] = sched->cur_hc[slot];
        next[rel] = cur[rel] == sched->hc_a[slot] ? sched->hc_b[slot] : sched->hc_a[slot];
        cfgs[rel] = (ds4_v100_layer_execute_config) {
            .model_map = sched->model_map,
            .model_size = sched->model_size,
            .model_map_uses_shard_offsets = sched->model_map_uses_shard_offsets,
            .arena = sched->arena,
            .batch_scratch = &sched->batch_scratch,
            .router_token = tokens[rel],
            .position = positions[rel],
            .decode_cache = &lc->cache,
            .fp8_kv_cache = sched->fp8_kv_cache,
            .suppress_router_readback = sched->suppress_router_readback,
            .tp2_layer = sched->tp2_layer,
            .tp2_layer_count = sched->tp2_layer_count,
            .tp2_owner_arena = sched->tp2_owner_arena,
            .tp2_peer_arena = sched->tp2_peer_arena,
            .tp2_peer_input = sched->tp2_peer_input,
            .tp2_peer_selected = sched->tp2_peer_selected,
            .tp2_peer_weights = sched->tp2_peer_weights,
            .tp2_peer_out = sched->tp2_peer_out,
            .tp2_peer_recv = sched->tp2_peer_recv,
            .tp2_scratch_slots = sched->tp2_scratch_slots,
        };
        if (ds4_v100_layer_execute_hc_prepare_ffn(&sched->states[layer],
                                                  &cfgs[rel],
                                                  cur[rel],
                                                  next[rel],
                                                  rel,
                                                  &prepared[rel],
                                                  err,
                                                  errlen)) {
            if (err && errlen && err[0]) {
                char inner[256];
                snprintf(inner, sizeof(inner), "%s", err);
                snprintf(err,
                         errlen,
                         "FFN microbatch prepare failed at layer %d slot %u: %s",
                         layer,
                         slot,
                         inner);
            }
            return 1;
        }
    }

    if (ds4_v100_layer_execute_hc_finish_ffn_batch(&sched->states[layer],
                                                   cfgs,
                                                   prepared,
                                                   n_slots,
                                                   layer_reports,
                                                   err,
                                                   errlen)) {
        if (err && errlen && err[0]) {
            char inner[256];
            snprintf(inner, sizeof(inner), "%s", err);
            snprintf(err,
                     errlen,
                     "FFN microbatch finish failed at layer %d: %s",
                     layer,
                     inner);
        }
        return 1;
    }

    for (uint32_t rel = 0; rel < n_slots; rel++) {
        const uint32_t slot = slot_start + rel;
        sched->cur_hc[slot] = next[rel];
        if (reports) {
            ds4_v100_stage_scheduler_report *r = &reports[rel];
            memset(r, 0, sizeof(*r));
            r->stage_id = sched->stage.stage_id;
            r->gpu = sched->stage.gpu;
            r->first_layer = layer;
            r->last_layer = layer;
            r->layers_executed = 1;
            r->position = positions[rel];
            r->token = tokens[rel];
            r->arena_bytes = ds4_gpu_arena_bytes(sched->arena);
            r->uploaded_tensors = sched->uploaded_tensors;
            r->uploaded_bytes = sched->uploaded_bytes;
            r->last_layer_report = layer_reports[rel];
            r->timing_tp2_copy_in_ms = layer_reports[rel].timing_tp2_copy_in_ms;
            r->timing_tp2_owner_ms = layer_reports[rel].timing_tp2_owner_ms;
            r->timing_tp2_peer_ms = layer_reports[rel].timing_tp2_peer_ms;
            r->timing_tp2_copy_out_ms = layer_reports[rel].timing_tp2_copy_out_ms;
            r->timing_tp2_reduce_ms = layer_reports[rel].timing_tp2_reduce_ms;
            r->timing_tp2_total_ms = layer_reports[rel].timing_tp2_total_ms;
            r->turbomind_routed_layers_executed =
                layer_reports[rel].turbomind_routed ? 1u : 0u;
            r->turbomind_tp2_routed_layers_executed =
                layer_reports[rel].turbomind_tp2_routed ? 1u : 0u;
        }
    }
    return 0;
}

int ds4_v100_stage_scheduler_decode_hc(ds4_v100_stage_scheduler *sched,
                                       uint32_t token,
                                       uint32_t position,
                                       ds4_v100_stage_scheduler_report *report,
                                       char *err,
                                       size_t errlen) {
    const uint32_t tokens[1] = { token };
    const uint32_t positions[1] = { position };
    return ds4_v100_stage_scheduler_decode_hc_batch(
        sched, tokens, positions, 1, report, err, errlen);
}

int ds4_v100_stage_scheduler_decode_token_batch(
    ds4_v100_stage_scheduler *sched,
    const uint32_t *tokens,
    const uint32_t *positions,
    uint32_t n_slots,
    ds4_v100_stage_scheduler_report *reports,
    char *err,
    size_t errlen) {
    return ds4_v100_stage_scheduler_decode_token_slot_span(
        sched, 0, tokens, positions, n_slots, reports, err, errlen);
}

int ds4_v100_stage_scheduler_decode_token_slot_span(
    ds4_v100_stage_scheduler *sched,
    uint32_t slot_start,
    const uint32_t *tokens,
    const uint32_t *positions,
    uint32_t n_slots,
    ds4_v100_stage_scheduler_report *reports,
    char *err,
    size_t errlen) {
    if (!sched) return scheduler_error(err, errlen, "missing scheduler");
    return ds4_v100_stage_scheduler_decode_token_layer_span(
        sched,
        slot_start,
        tokens,
        positions,
        n_slots,
        sched->stage.layer_begin,
        sched->stage.layer_end,
        reports,
        err,
        errlen);
}

int ds4_v100_stage_scheduler_decode_token_layer_span(
    ds4_v100_stage_scheduler *sched,
    uint32_t slot_start,
    const uint32_t *tokens,
    const uint32_t *positions,
    uint32_t n_slots,
    int first_layer,
    int last_layer,
    ds4_v100_stage_scheduler_report *reports,
    char *err,
    size_t errlen) {
    if (!sched) return scheduler_error(err, errlen, "missing scheduler");
    if (!sched->stage.owns_token_embedding) {
        return scheduler_error(err, errlen, "decode-token requires token-embedding stage");
    }
    if (scheduler_validate_slot_span_args(
            sched, tokens, positions, slot_start, n_slots, err, errlen)) {
        return 1;
    }
    if (scheduler_validate_layer_span_args(sched, first_layer, last_layer, err, errlen)) {
        return 1;
    }
    if (first_layer != sched->stage.layer_begin) {
        if (err && errlen) {
            snprintf(err,
                     errlen,
                     "decode-token layer span must start at stage first layer %d",
                     sched->stage.layer_begin);
        }
        return 1;
    }
    if (!ds4_gpu_set_device(sched->stage.gpu)) {
        return scheduler_errorf(err, errlen, "failed to set scheduler device gpu%d",
                                sched->stage.gpu);
    }
    if (scheduler_activate_model_source(sched, err, errlen)) return 1;
    if (sched->token_embedding.n_shape_dims != 2 ||
        sched->token_embedding.shape[0] != DS4_V100_HC_COLS) {
        return scheduler_error(err, errlen, "token embedding shape does not match HC width");
    }
    const uint32_t n_vocab = (uint32_t)sched->token_embedding.shape[1];
    for (uint32_t rel = 0; rel < n_slots; rel++) {
        const uint32_t slot = slot_start + rel;
        const uint32_t token = tokens[rel];
        if (token >= n_vocab) return scheduler_error(err, errlen, "token outside embedding vocab");
        if (!ds4_gpu_embed_token_hc_tensor(sched->hc_a[slot],
                                           sched->model_map,
                                           sched->model_size,
                                           scheduler_model_offset(sched, &sched->token_embedding),
                                           n_vocab,
                                           token,
                                           DS4_V100_HC_COLS,
                                           DS4_V100_HC_ROWS)) {
            return scheduler_errorf(err, errlen, "token embedding HC seed failed for slot %d",
                                    (int)slot);
        }
        sched->cur_hc[slot] = sched->hc_a[slot];
    }
    return ds4_v100_stage_scheduler_decode_hc_layer_span(
        sched,
        slot_start,
        tokens,
        positions,
        n_slots,
        first_layer,
        last_layer,
        reports,
        err,
        errlen);
}

int ds4_v100_stage_scheduler_decode_token(ds4_v100_stage_scheduler *sched,
                                          uint32_t token,
                                          uint32_t position,
                                          ds4_v100_stage_scheduler_report *report,
                                          char *err,
                                          size_t errlen) {
    const uint32_t tokens[1] = { token };
    const uint32_t positions[1] = { position };
    return ds4_v100_stage_scheduler_decode_token_batch(
        sched, tokens, positions, 1, report, err, errlen);
}

int ds4_v100_stage_scheduler_decode_token_checkpoints(
    ds4_v100_stage_scheduler *sched,
    uint32_t token,
    uint32_t position,
    ds4_v100_stage_scheduler_report *report,
    ds4_v100_stage_scheduler_checkpoint_fn checkpoint_fn,
    void *checkpoint_user,
    char *err,
    size_t errlen) {
    if (!sched) return scheduler_error(err, errlen, "missing scheduler");
    if (sched->active_slots != 1) {
        return scheduler_errorf_u32(err,
                                    errlen,
                                    "checkpoint decode currently requires active_slots=1, got %u",
                                    sched->active_slots);
    }
    if (!sched->stage.owns_token_embedding) {
        return scheduler_error(err, errlen, "decode-token requires token-embedding stage");
    }
    if (!ds4_gpu_set_device(sched->stage.gpu)) {
        return scheduler_errorf(err, errlen, "failed to set scheduler device gpu%d",
                                sched->stage.gpu);
    }
    if (scheduler_activate_model_source(sched, err, errlen)) return 1;
    if (sched->token_embedding.n_shape_dims != 2 ||
        sched->token_embedding.shape[0] != DS4_V100_HC_COLS) {
        return scheduler_error(err, errlen, "token embedding shape does not match HC width");
    }
    const uint32_t n_vocab = (uint32_t)sched->token_embedding.shape[1];
    if (token >= n_vocab) return scheduler_error(err, errlen, "token outside embedding vocab");
    if (!ds4_gpu_embed_token_hc_tensor(sched->hc_a[0],
                                       sched->model_map,
                                       sched->model_size,
                                       scheduler_model_offset(sched, &sched->token_embedding),
                                       n_vocab,
                                       token,
                                       DS4_V100_HC_COLS,
                                       DS4_V100_HC_ROWS)) {
        return scheduler_error(err, errlen, "token embedding HC seed failed");
    }
    sched->cur_hc[0] = sched->hc_a[0];
    if (checkpoint_fn) {
        const uint64_t hc_bytes =
            (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS * sizeof(float);
        ds4_v100_stage_scheduler_checkpoint cp = {
            .stage_id = sched->stage.stage_id,
            .gpu = sched->stage.gpu,
            .layer = -1,
            .kind = DS4_V100_HC_CHECKPOINT_SEED,
            .position = position,
            .token = token,
            .hc = sched->cur_hc[0],
            .hc_bytes = hc_bytes,
        };
        if (checkpoint_fn(&cp, checkpoint_user, err, errlen)) return 1;
    }
    return ds4_v100_stage_scheduler_decode_hc_checkpoints(sched,
                                                          token,
                                                          position,
                                                          report,
                                                          checkpoint_fn,
                                                          checkpoint_user,
                                                          err,
                                                          errlen);
}

int ds4_v100_stage_scheduler_handoff_batch(ds4_v100_stage_scheduler *dst,
                                           const ds4_v100_stage_scheduler *src,
                                           uint32_t n_slots,
                                           char *err,
                                           size_t errlen) {
    return ds4_v100_stage_scheduler_handoff_slot_span(dst, src, 0, n_slots, err, errlen);
}

static int scheduler_handoff_slot_span_impl(ds4_v100_stage_scheduler *dst,
                                            const ds4_v100_stage_scheduler *src,
                                            uint32_t slot_start,
                                            uint32_t n_slots,
                                            bool async_copy,
                                            const ds4_gpu_event *event,
                                            char *err,
                                            size_t errlen) {
    if (!dst || !src) {
        return scheduler_error(err, errlen, "missing scheduler handoff endpoint");
    }
    if (slot_start >= dst->active_slots || slot_start >= src->active_slots) {
        return scheduler_errorf_u32(err,
                                    errlen,
                                    "handoff slot start must be in [0,active_slots), got %u",
                                    slot_start);
    }
    if (n_slots == 0 ||
        n_slots > dst->active_slots - slot_start ||
        n_slots > src->active_slots - slot_start) {
        return scheduler_errorf_u32(err,
                                    errlen,
                                    "handoff slot span exceeds active slots, got %u",
                                    n_slots);
    }
    if (n_slots > DS4_V100_SCHED_MAX_SLOTS ||
        slot_start > DS4_V100_SCHED_MAX_SLOTS ||
        n_slots > DS4_V100_SCHED_MAX_SLOTS - slot_start) {
        return scheduler_errorf_u32(err,
                                    errlen,
                                    "handoff slots exceed scheduler max: %u",
                                    n_slots);
    }
    const uint64_t hc_bytes =
        (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS * sizeof(float);
    for (uint32_t rel = 0; rel < n_slots; rel++) {
        const uint32_t slot = slot_start + rel;
        if (!src->cur_hc[slot] || !dst->hc_a[slot]) {
            return scheduler_errorf(err, errlen, "missing handoff HC slot %d", (int)slot);
        }
        int ok = 0;
        if (event) {
            ok = ds4_gpu_tensor_copy_async_after_event(dst->hc_a[slot],
                                                       0,
                                                       src->cur_hc[slot],
                                                       0,
                                                       hc_bytes,
                                                       event);
        } else if (async_copy) {
            ok = ds4_gpu_tensor_copy_async(dst->hc_a[slot], 0, src->cur_hc[slot], 0, hc_bytes);
        } else {
            ok = ds4_gpu_tensor_copy(dst->hc_a[slot], 0, src->cur_hc[slot], 0, hc_bytes);
        }
        if (!ok) {
            return scheduler_errorf(err, errlen, "scheduler HC handoff copy failed for slot %d",
                                    (int)slot);
        }
        dst->cur_hc[slot] = dst->hc_a[slot];
    }
    return 0;
}

int ds4_v100_stage_scheduler_handoff_slot_span(ds4_v100_stage_scheduler *dst,
                                               const ds4_v100_stage_scheduler *src,
                                               uint32_t slot_start,
                                               uint32_t n_slots,
                                               char *err,
                                               size_t errlen) {
    return scheduler_handoff_slot_span_impl(dst, src, slot_start, n_slots, false, NULL, err, errlen);
}

int ds4_v100_stage_scheduler_handoff_slot_span_async(ds4_v100_stage_scheduler *dst,
                                                     const ds4_v100_stage_scheduler *src,
                                                     uint32_t slot_start,
                                                     uint32_t n_slots,
                                                     char *err,
                                                     size_t errlen) {
    return scheduler_handoff_slot_span_impl(dst, src, slot_start, n_slots, true, NULL, err, errlen);
}

int ds4_v100_stage_scheduler_handoff_slot_span_after_event_async(
    ds4_v100_stage_scheduler *dst,
    const ds4_v100_stage_scheduler *src,
    uint32_t slot_start,
    uint32_t n_slots,
    const ds4_gpu_event *event,
    char *err,
    size_t errlen) {
    if (!event) {
        return scheduler_error(err, errlen, "missing scheduler handoff event");
    }
    return scheduler_handoff_slot_span_impl(dst, src, slot_start, n_slots, true, event, err, errlen);
}

int ds4_v100_stage_scheduler_handoff(ds4_v100_stage_scheduler *dst,
                                     const ds4_v100_stage_scheduler *src,
                                     char *err,
                                     size_t errlen) {
    return ds4_v100_stage_scheduler_handoff_batch(dst, src, 1, err, errlen);
}

int ds4_v100_stage_scheduler_decode_hc_checkpoints(
    ds4_v100_stage_scheduler *sched,
    uint32_t token,
    uint32_t position,
    ds4_v100_stage_scheduler_report *report,
    ds4_v100_stage_scheduler_checkpoint_fn checkpoint_fn,
    void *checkpoint_user,
    char *err,
    size_t errlen) {
    if (!sched || !sched->cur_hc[0]) {
        return scheduler_error(err, errlen, "missing scheduler HC input");
    }
    if (sched->active_slots != 1) {
        return scheduler_errorf_u32(err,
                                    errlen,
                                    "checkpoint decode currently requires active_slots=1, got %u",
                                    sched->active_slots);
    }
    if (!ds4_gpu_set_device(sched->stage.gpu)) {
        return scheduler_errorf(err, errlen, "failed to set scheduler device gpu%d",
                                sched->stage.gpu);
    }
    if (scheduler_activate_model_source(sched, err, errlen)) return 1;
    ds4_gpu_tensor *cur = sched->cur_hc[0];
    ds4_gpu_tensor *next = cur == sched->hc_a[0] ? sched->hc_b[0] : sched->hc_a[0];
    ds4_v100_layer_execute_report last;
    uint32_t turbomind_layers = 0;
    uint32_t turbomind_tp2_layers = 0;
    memset(&last, 0, sizeof(last));
    uint32_t executed = 0;
    for (int layer = sched->stage.layer_begin; layer <= sched->stage.layer_end; layer++) {
        scheduler_layer_cache *lc = scheduler_cache_slot(sched, layer, 0);
        if (!lc) return scheduler_errorf(err, errlen, "missing decode cache for layer %d", layer);
        scheduler_layer_checkpoint_user layer_checkpoint_user = {
            .sched = sched,
            .checkpoint_fn = checkpoint_fn,
            .checkpoint_user = checkpoint_user,
            .token = token,
            .position = position,
        };
        ds4_v100_layer_execute_config cfg = {
            .model_map = sched->model_map,
            .model_size = sched->model_size,
            .model_map_uses_shard_offsets = sched->model_map_uses_shard_offsets,
            .arena = sched->arena,
            .router_token = token,
            .position = position,
            .decode_cache = &lc->cache,
            .fp8_kv_cache = sched->fp8_kv_cache,
            .suppress_router_readback = sched->suppress_router_readback,
            .tp2_layer = sched->tp2_layer,
            .tp2_layer_count = sched->tp2_layer_count,
            .tp2_owner_arena = sched->tp2_owner_arena,
            .tp2_peer_arena = sched->tp2_peer_arena,
            .tp2_peer_input = sched->tp2_peer_input,
            .tp2_peer_selected = sched->tp2_peer_selected,
            .tp2_peer_weights = sched->tp2_peer_weights,
            .tp2_peer_out = sched->tp2_peer_out,
            .tp2_peer_recv = sched->tp2_peer_recv,
            .tp2_scratch_slots = sched->tp2_scratch_slots,
            .checkpoint_layer = layer,
            .checkpoint_fn = checkpoint_fn ? scheduler_layer_checkpoint : NULL,
            .checkpoint_user = &layer_checkpoint_user,
        };
        memset(&last, 0, sizeof(last));
        if (ds4_v100_layer_execute_hc_decode(&sched->states[layer],
                                             &cfg,
                                             cur,
                                             next,
                                             &last,
                                             err,
                                             errlen)) {
            return 1;
        }
        ds4_gpu_tensor *tmp = cur;
        cur = next;
        next = tmp;
        executed++;
        turbomind_layers += last.turbomind_routed ? 1u : 0u;
        turbomind_tp2_layers += last.turbomind_tp2_routed ? 1u : 0u;
        if (checkpoint_fn) {
            const uint64_t hc_bytes =
                (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS * sizeof(float);
            ds4_v100_stage_scheduler_checkpoint cp = {
                .stage_id = sched->stage.stage_id,
                .gpu = sched->stage.gpu,
                .layer = layer,
                .kind = DS4_V100_HC_CHECKPOINT_LAYER_FINAL,
                .position = position,
                .token = token,
                .hc = cur,
                .hc_bytes = hc_bytes,
                .layer_report = last,
            };
            if (checkpoint_fn(&cp, checkpoint_user, err, errlen)) return 1;
        }
    }
    sched->cur_hc[0] = cur;
    if (report) {
        memset(report, 0, sizeof(*report));
        report->stage_id = sched->stage.stage_id;
        report->gpu = sched->stage.gpu;
        report->first_layer = sched->stage.layer_begin;
        report->last_layer = sched->stage.layer_end;
        report->layers_executed = executed;
        report->position = position;
        report->token = token;
        report->arena_bytes = ds4_gpu_arena_bytes(sched->arena);
        report->uploaded_tensors = sched->uploaded_tensors;
        report->uploaded_bytes = sched->uploaded_bytes;
        report->last_layer_report = last;
        report->turbomind_routed_layers_executed = turbomind_layers;
        report->turbomind_tp2_routed_layers_executed = turbomind_tp2_layers;
    }
    return 0;
}

int ds4_v100_stage_scheduler_read_hc(const ds4_v100_stage_scheduler *sched,
                                     void *dst,
                                     uint64_t bytes) {
    return ds4_v100_stage_scheduler_read_hc_slot(sched, 0, dst, bytes);
}

int ds4_v100_stage_scheduler_read_hc_slot(const ds4_v100_stage_scheduler *sched,
                                          uint32_t slot,
                                          void *dst,
                                          uint64_t bytes) {
    if (!sched || slot >= sched->active_slots || !sched->cur_hc[slot] || !dst) return 0;
    const uint64_t hc_bytes =
        (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS * sizeof(float);
    if (bytes > hc_bytes) return 0;
    return ds4_gpu_tensor_read(sched->cur_hc[slot], 0, dst, bytes);
}

int ds4_v100_stage_scheduler_read_token_embedding_f32(
    const ds4_v100_stage_scheduler *sched,
    uint32_t token,
    float *dst,
    uint64_t dst_values,
    char *err,
    size_t errlen) {
    if (!sched || !dst) {
        return scheduler_error(err, errlen, "missing scheduler token embedding input");
    }
    const ds4_v100_tensor_binding *b = &sched->token_embedding;
    if (!b->source_dtype || strcmp(b->source_dtype, "bf16") != 0 ||
        b->n_shape_dims != 2 ||
        b->shape[0] != DS4_V100_HC_COLS ||
        b->shape[1] == 0 ||
        dst_values != DS4_V100_HC_COLS ||
        b->byte_length != b->shape[0] * b->shape[1] * sizeof(uint16_t)) {
        return scheduler_error(err, errlen, "invalid token_embd.weight bf16 binding");
    }
    if (token >= b->shape[1]) {
        return scheduler_error(err, errlen, "token outside embedding vocab");
    }
    if (scheduler_activate_model_source(sched, err, errlen)) return 1;
    const uint64_t row_bytes = b->shape[0] * sizeof(uint16_t);
    const uint64_t row_offset = scheduler_model_offset(sched, b) + (uint64_t)token * row_bytes;
    if (row_offset > sched->model_size ||
        row_bytes > sched->model_size - row_offset) {
        return scheduler_error(err, errlen, "token embedding row outside model map");
    }
    const uint16_t *row =
        (const uint16_t *)((const unsigned char *)sched->model_map + row_offset);
    if (ds4_src_bf16_row_to_f32(dst, row, DS4_V100_HC_COLS) != 0) {
        return scheduler_error(err, errlen, "token embedding bf16 decode failed");
    }
    return 0;
}

int ds4_v100_stage_scheduler_write_hc(ds4_v100_stage_scheduler *sched,
                                      const void *src,
                                      uint64_t bytes) {
    if (!sched || !sched->hc_a[0] || !src) return 0;
    const uint64_t hc_bytes =
        (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS * sizeof(float);
    if (bytes != hc_bytes) return 0;
    if (!ds4_gpu_set_device(sched->stage.gpu)) return 0;
    if (!ds4_gpu_tensor_write(sched->hc_a[0], 0, src, bytes)) return 0;
    sched->cur_hc[0] = sched->hc_a[0];
    return 1;
}

static int output_bf16_view(const ds4_v100_tensor_binding *b,
                            ds4_gpu_bf16_matrix_view *out,
                            char *err,
                            size_t errlen) {
    if (!b || !out) return scheduler_error(err, errlen, "missing output binding");
    if (!b->source_dtype || strcmp(b->source_dtype, "bf16") != 0 ||
        b->n_shape_dims != 2 ||
        b->shape[0] != DS4_V100_HC_COLS ||
        b->shape[1] == 0 ||
        b->byte_length != b->shape[0] * b->shape[1] * sizeof(uint16_t)) {
        return scheduler_error(err, errlen, "invalid output.weight bf16 binding");
    }
    memset(out, 0, sizeof(*out));
    out->arena_offset = b->shard_offset;
    out->byte_length = b->byte_length;
    out->rows = (uint32_t)b->shape[1];
    out->cols = (uint32_t)b->shape[0];
    out->row_stride_elements = (uint32_t)b->shape[0];
    return 0;
}

static void insert_topk(uint32_t *tokens,
                        float *logits,
                        uint32_t k,
                        uint32_t token,
                        float logit) {
    for (uint32_t i = 0; i < k; i++) {
        if (tokens[i] == UINT32_MAX || logit > logits[i]) {
            for (uint32_t j = k - 1; j > i; j--) {
                tokens[j] = tokens[j - 1];
                logits[j] = logits[j - 1];
            }
            tokens[i] = token;
            logits[i] = logit;
            return;
        }
    }
}

static bool output_head_fastpath_enabled(void) {
    const char *disabled = getenv("DS4_V100_DISABLE_OUTPUT_HEAD_FASTPATH");
    if (disabled && disabled[0] &&
        strcmp(disabled, "0") != 0 &&
        strcmp(disabled, "false") != 0 &&
        strcmp(disabled, "FALSE") != 0) {
        return false;
    }
    const char *v = getenv("DS4_V100_ENABLE_OUTPUT_HEAD_FASTPATH");
    if (v && v[0] &&
        (strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "FALSE") == 0)) {
        return false;
    }
    return true;
}

static bool output_head_batch_enabled(void) {
    const char *disable = getenv("DS4_V100_DISABLE_OUTPUT_HEAD_BATCH");
    if (disable && disable[0] &&
        strcmp(disable, "0") != 0 &&
        strcmp(disable, "false") != 0 &&
        strcmp(disable, "FALSE") != 0) {
        return false;
    }
    const char *enable = getenv("DS4_V100_ENABLE_OUTPUT_HEAD_BATCH");
    return enable && enable[0] &&
        strcmp(enable, "0") != 0 &&
        strcmp(enable, "false") != 0 &&
        strcmp(enable, "FALSE") != 0;
}

static int ensure_output_head_scratch(ds4_v100_stage_scheduler *sched,
                                      uint32_t n_vocab,
                                      uint32_t n_slots,
                                      char *err,
                                      size_t errlen) {
    if (!sched || !sched->stage.owns_output_head || n_vocab == 0 || n_slots == 0 ||
        n_slots > DS4_V100_SCHED_MAX_SLOTS) {
        return scheduler_error(err, errlen, "missing output-head scratch input");
    }
    if (sched->output_hc_batch &&
        sched->output_hc_norm &&
        sched->output_head_pre &&
        sched->output_head_weights &&
        sched->output_embd &&
        sched->output_norm_scratch &&
        sched->output_logits &&
        sched->output_scratch_vocab == n_vocab &&
        sched->output_scratch_slots == n_slots) {
        return 0;
    }

    free_output_head_scratch(sched);
    if (!ds4_gpu_set_device(sched->stage.gpu)) {
        return scheduler_errorf(err, errlen, "failed to set output-head scratch device gpu%d",
                                sched->stage.gpu);
    }

    const uint64_t hc_values = (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS;
    sched->output_hc_batch = ds4_gpu_tensor_alloc(hc_values * n_slots * sizeof(float));
    sched->output_hc_norm = ds4_gpu_tensor_alloc(hc_values * n_slots * sizeof(float));
    sched->output_head_pre = ds4_gpu_tensor_alloc((uint64_t)DS4_V100_HC_ROWS * n_slots * sizeof(float));
    sched->output_head_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_V100_HC_ROWS * n_slots * sizeof(float));
    sched->output_embd = ds4_gpu_tensor_alloc((uint64_t)DS4_V100_HC_COLS * n_slots * sizeof(float));
    sched->output_norm_scratch = ds4_gpu_tensor_alloc((uint64_t)DS4_V100_HC_COLS * n_slots * sizeof(float));
    sched->output_logits = ds4_gpu_tensor_alloc((uint64_t)n_vocab * n_slots * sizeof(float));
    if (!sched->output_hc_batch ||
        !sched->output_hc_norm ||
        !sched->output_head_pre ||
        !sched->output_head_weights ||
        !sched->output_embd ||
        !sched->output_norm_scratch ||
        !sched->output_logits) {
        free_output_head_scratch(sched);
        return scheduler_error(err, errlen, "failed to allocate output-head scratch");
    }
    sched->output_scratch_vocab = n_vocab;
    sched->output_scratch_slots = n_slots;
    return 0;
}

static int scheduler_select_top1_fastpath(ds4_v100_stage_scheduler *sched,
                                          uint32_t slot,
                                          const ds4_gpu_bf16_matrix_view *output_v,
                                          uint32_t n_vocab,
                                          uint32_t *tokens,
                                          float *out_logits,
                                          char *err,
                                          size_t errlen) {
    if (ensure_output_head_scratch(sched, n_vocab, 1, err, errlen)) return 1;

    const uint64_t hc_values = (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS;
    const uint64_t hc_bytes = hc_values * sizeof(float);
    if (!ds4_gpu_tensor_copy(sched->output_hc_batch, 0, sched->cur_hc[slot], 0, hc_bytes)) {
        return scheduler_error(err, errlen, "output-head fast HC pack failed");
    }
    if (!ds4_gpu_rms_norm_plain_tensor(sched->output_hc_norm,
                                       sched->output_hc_batch,
                                       (uint32_t)hc_values,
                                       1.0e-6f) ||
        !ds4_gpu_matmul_f32_tensor(sched->output_head_pre,
                                   sched->model_map,
                                   sched->model_size,
                                   scheduler_model_offset(sched, &sched->hc_head_fn),
                                   hc_values,
                                   DS4_V100_HC_ROWS,
                                   sched->output_hc_norm,
                                   1) ||
        !ds4_gpu_output_hc_weights_tensor(sched->output_head_weights,
                                          sched->output_head_pre,
                                          sched->model_map,
                                          sched->model_size,
                                          scheduler_model_offset(sched, &sched->hc_head_scale),
                                          scheduler_model_offset(sched, &sched->hc_head_base),
                                          DS4_V100_HC_ROWS,
                                          1.0e-6f) ||
        !ds4_gpu_hc_weighted_sum_tensor(sched->output_embd,
                                        sched->output_hc_batch,
                                        sched->output_head_weights,
                                        DS4_V100_HC_COLS,
                                        DS4_V100_HC_ROWS) ||
        !ds4_gpu_rms_norm_weight_tensor(sched->output_norm_scratch,
                                        sched->output_embd,
                                        sched->model_map,
                                        sched->model_size,
                                        scheduler_model_offset(sched, &sched->output_norm),
                                        DS4_V100_HC_COLS,
                                        1.0e-6f) ||
        ds4_gpu_arena_bf16_matmul_f32(sched->arena,
                                      output_v,
                                      sched->output_norm_scratch,
                                      sched->output_logits) != 0 ||
        !ds4_gpu_top1_f32_tensor(sched->output_logits,
                                 n_vocab,
                                 &tokens[0],
                                 &out_logits[0])) {
        return scheduler_error(err, errlen, "output-head fast selected-token sequence failed");
    }
    if (!isfinite(out_logits[0])) {
        return scheduler_error(err, errlen, "output-head logits contained non-finite values");
    }
    return 0;
}

int ds4_v100_stage_scheduler_select_topk_slot(ds4_v100_stage_scheduler *sched,
                                              uint32_t slot,
                                              uint32_t *tokens,
                                              float *out_logits,
                                              uint32_t k,
                                              char *err,
                                              size_t errlen) {
    if (!sched || slot >= sched->active_slots || !sched->cur_hc[slot] ||
        !tokens || !out_logits || k == 0) {
        return scheduler_error(err, errlen, "missing scheduler output-head input");
    }
    for (uint32_t i = 0; i < k; i++) {
        tokens[i] = UINT32_MAX;
        out_logits[i] = -FLT_MAX;
    }
    if (!sched->stage.owns_output_head) {
        return scheduler_error(err, errlen, "select-token requires output-head stage");
    }
    if (!ds4_gpu_set_device(sched->stage.gpu)) {
        return scheduler_errorf(err, errlen, "failed to set scheduler device gpu%d",
                                sched->stage.gpu);
    }
    if (scheduler_activate_model_source(sched, err, errlen)) return 1;

    const uint64_t hc_values = (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS;
    if (sched->hc_head_fn.n_shape_dims != 2 ||
        sched->hc_head_fn.shape[0] != hc_values ||
        sched->hc_head_fn.shape[1] != DS4_V100_HC_ROWS ||
        sched->hc_head_base.n_shape_dims != 1 ||
        sched->hc_head_base.shape[0] != DS4_V100_HC_ROWS ||
        sched->hc_head_scale.n_shape_dims != 1 ||
        sched->hc_head_scale.shape[0] != 1 ||
        sched->output_norm.n_shape_dims != 1 ||
        sched->output_norm.shape[0] != DS4_V100_HC_COLS ||
        sched->output_weight.owning_gpu != sched->stage.gpu) {
        return scheduler_error(err, errlen, "invalid output-head descriptor shapes");
    }

    ds4_gpu_bf16_matrix_view output_v;
    if (output_bf16_view(&sched->output_weight, &output_v, err, errlen)) return 1;
    const uint32_t n_vocab = output_v.rows;
    if (k == 1 && output_head_fastpath_enabled()) {
        return scheduler_select_top1_fastpath(sched,
                                              slot,
                                              &output_v,
                                              n_vocab,
                                              tokens,
                                              out_logits,
                                              err,
                                              errlen);
    }
    const uint64_t logits_bytes = (uint64_t)n_vocab * sizeof(float);

    ds4_gpu_tensor *hc_norm = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *head_pre = ds4_gpu_tensor_alloc(DS4_V100_HC_ROWS * sizeof(float));
    ds4_gpu_tensor *head_weights = ds4_gpu_tensor_alloc(DS4_V100_HC_ROWS * sizeof(float));
    ds4_gpu_tensor *output_embd = ds4_gpu_tensor_alloc(DS4_V100_HC_COLS * sizeof(float));
    ds4_gpu_tensor *output_norm = ds4_gpu_tensor_alloc(DS4_V100_HC_COLS * sizeof(float));
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(logits_bytes);
    float *host_logits = (float *)malloc((size_t)logits_bytes);
    int rc = 1;

    if (!hc_norm || !head_pre || !head_weights || !output_embd ||
        !output_norm || !logits || !host_logits) {
        scheduler_error(err, errlen, "failed to allocate output-head tensors");
        goto done;
    }

    if (!ds4_gpu_rms_norm_plain_tensor(hc_norm,
                                       sched->cur_hc[slot],
                                       (uint32_t)hc_values,
                                       1.0e-6f) ||
        !ds4_gpu_matmul_f32_tensor(head_pre,
                                   sched->model_map,
                                   sched->model_size,
                                   scheduler_model_offset(sched, &sched->hc_head_fn),
                                   hc_values,
                                   DS4_V100_HC_ROWS,
                                   hc_norm,
                                   1) ||
        !ds4_gpu_output_hc_weights_tensor(head_weights,
                                          head_pre,
                                          sched->model_map,
                                          sched->model_size,
                                          scheduler_model_offset(sched, &sched->hc_head_scale),
                                          scheduler_model_offset(sched, &sched->hc_head_base),
                                          DS4_V100_HC_ROWS,
                                          1.0e-6f) ||
        !ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                        sched->cur_hc[slot],
                                        head_weights,
                                        DS4_V100_HC_COLS,
                                        DS4_V100_HC_ROWS) ||
        !ds4_gpu_rms_norm_weight_tensor(output_norm,
                                        output_embd,
                                        sched->model_map,
                                        sched->model_size,
                                        scheduler_model_offset(sched, &sched->output_norm),
                                        DS4_V100_HC_COLS,
                                        1.0e-6f) ||
        ds4_gpu_arena_bf16_matmul_f32(sched->arena,
                                      &output_v,
                                      output_norm,
                                      logits) != 0 ||
        !ds4_gpu_tensor_read(logits, 0, host_logits, logits_bytes)) {
        scheduler_error(err, errlen, "output-head selected-token sequence failed");
        goto done;
    }

    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = host_logits[i];
        if (!isfinite(v)) {
            scheduler_error(err, errlen, "output-head logits contained non-finite values");
            goto done;
        }
        insert_topk(tokens, out_logits, k, i, v);
    }
    rc = 0;

done:
    free(host_logits);
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(head_weights);
    ds4_gpu_tensor_free(head_pre);
    ds4_gpu_tensor_free(hc_norm);
    return rc;
}

int ds4_v100_stage_scheduler_select_topk(ds4_v100_stage_scheduler *sched,
                                         uint32_t *tokens,
                                         float *out_logits,
                                         uint32_t k,
                                         char *err,
                                         size_t errlen) {
    return ds4_v100_stage_scheduler_select_topk_slot(sched, 0, tokens, out_logits, k, err, errlen);
}

int ds4_v100_stage_scheduler_select_token_slot(ds4_v100_stage_scheduler *sched,
                                               uint32_t slot,
                                               uint32_t *token,
                                               float *logit,
                                               char *err,
                                               size_t errlen) {
    if (!token) return scheduler_error(err, errlen, "missing selected-token output");
    uint32_t top_token = UINT32_MAX;
    float top_logit = 0.0f;
    int rc = ds4_v100_stage_scheduler_select_topk_slot(sched,
                                                       slot,
                                                       &top_token,
                                                       &top_logit,
                                                       1,
                                                       err,
                                                       errlen);
    if (rc == 0) {
        *token = top_token;
        if (logit) *logit = top_logit;
    }
    return rc;
}

int ds4_v100_stage_scheduler_select_token_batch(ds4_v100_stage_scheduler *sched,
                                                uint32_t slot_start,
                                                uint32_t n_slots,
                                                uint32_t *tokens,
                                                float *logits,
                                                char *err,
                                                size_t errlen) {
    if (!sched || !tokens || !logits || n_slots == 0 ||
        slot_start >= sched->active_slots ||
        n_slots > sched->active_slots - slot_start ||
        n_slots > DS4_V100_SCHED_MAX_SLOTS) {
        return scheduler_error(err, errlen, "missing scheduler output-head batch input");
    }
    if (scheduler_debug_hc_finite_pre_output_enabled()) {
        if (!ds4_gpu_set_device(sched->stage.gpu)) {
            return scheduler_errorf(err, errlen, "failed to set scheduler device gpu%d",
                                    sched->stage.gpu);
        }
        for (uint32_t rel = 0; rel < n_slots; rel++) {
            const uint32_t slot = slot_start + rel;
            if (scheduler_check_hc_finite(sched,
                                          sched->cur_hc[slot],
                                          "pre-output-head",
                                          -1,
                                          slot,
                                          UINT32_MAX,
                                          UINT32_MAX,
                                          err,
                                          errlen)) {
                return 1;
            }
        }
    }
    if (n_slots == 1 ||
        !output_head_fastpath_enabled() ||
        !output_head_batch_enabled()) {
        for (uint32_t rel = 0; rel < n_slots; rel++) {
            if (ds4_v100_stage_scheduler_select_token_slot(sched,
                                                           slot_start + rel,
                                                           &tokens[rel],
                                                           &logits[rel],
                                                           err,
                                                           errlen)) {
                return 1;
            }
        }
        return 0;
    }
    if (!sched->stage.owns_output_head) {
        return scheduler_error(err, errlen, "select-token batch requires output-head stage");
    }
    if (!ds4_gpu_set_device(sched->stage.gpu)) {
        return scheduler_errorf(err, errlen, "failed to set scheduler device gpu%d",
                                sched->stage.gpu);
    }
    if (scheduler_activate_model_source(sched, err, errlen)) return 1;

    const uint64_t hc_values = (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS;
    const uint64_t hc_bytes = hc_values * sizeof(float);
    if (sched->hc_head_fn.n_shape_dims != 2 ||
        sched->hc_head_fn.shape[0] != hc_values ||
        sched->hc_head_fn.shape[1] != DS4_V100_HC_ROWS ||
        sched->hc_head_base.n_shape_dims != 1 ||
        sched->hc_head_base.shape[0] != DS4_V100_HC_ROWS ||
        sched->hc_head_scale.n_shape_dims != 1 ||
        sched->hc_head_scale.shape[0] != 1 ||
        sched->output_norm.n_shape_dims != 1 ||
        sched->output_norm.shape[0] != DS4_V100_HC_COLS ||
        sched->output_weight.owning_gpu != sched->stage.gpu) {
        return scheduler_error(err, errlen, "invalid output-head descriptor shapes");
    }

    ds4_gpu_bf16_matrix_view output_v;
    if (output_bf16_view(&sched->output_weight, &output_v, err, errlen)) return 1;
    const uint32_t n_vocab = output_v.rows;
    if (ensure_output_head_scratch(sched, n_vocab, n_slots, err, errlen)) return 1;

    for (uint32_t rel = 0; rel < n_slots; rel++) {
        const uint32_t slot = slot_start + rel;
        if (!sched->cur_hc[slot]) {
            return scheduler_errorf(err, errlen, "missing scheduler HC input for slot %d",
                                    (int)slot);
        }
        if (!ds4_gpu_tensor_copy(sched->output_hc_batch,
                                 (uint64_t)rel * hc_bytes,
                                 sched->cur_hc[slot],
                                 0,
                                 hc_bytes)) {
            return scheduler_errorf(err, errlen, "output-head batch HC pack failed for slot %d",
                                    (int)slot);
        }
    }

    if (!ds4_gpu_rms_norm_plain_rows_tensor(sched->output_hc_norm,
                                            sched->output_hc_batch,
                                            (uint32_t)hc_values,
                                            n_slots,
                                            1.0e-6f) ||
        !ds4_gpu_matmul_f32_tensor(sched->output_head_pre,
                                   sched->model_map,
                                   sched->model_size,
                                   scheduler_model_offset(sched, &sched->hc_head_fn),
                                   hc_values,
                                   DS4_V100_HC_ROWS,
                                   sched->output_hc_norm,
                                   n_slots) ||
        !ds4_gpu_output_hc_weights_tensor(sched->output_head_weights,
                                          sched->output_head_pre,
                                          sched->model_map,
                                          sched->model_size,
                                          scheduler_model_offset(sched, &sched->hc_head_scale),
                                          scheduler_model_offset(sched, &sched->hc_head_base),
                                          DS4_V100_HC_ROWS,
                                          1.0e-6f) ||
        !ds4_gpu_hc_weighted_sum_tensor(sched->output_embd,
                                        sched->output_hc_batch,
                                        sched->output_head_weights,
                                        DS4_V100_HC_COLS,
                                        DS4_V100_HC_ROWS) ||
        !ds4_gpu_rms_norm_weight_rows_tensor(sched->output_norm_scratch,
                                             sched->output_embd,
                                             sched->model_map,
                                             sched->model_size,
                                             scheduler_model_offset(sched, &sched->output_norm),
                                             DS4_V100_HC_COLS,
                                             n_slots,
                                             1.0e-6f) ||
        ds4_gpu_arena_bf16_matmul_f32_rows(sched->arena,
                                           &output_v,
                                           sched->output_norm_scratch,
                                           n_slots,
                                           sched->output_logits) != 0 ||
        !ds4_gpu_top1_f32_rows_tensor(sched->output_logits,
                                      n_slots,
                                      n_vocab,
                                      tokens,
                                      logits)) {
        return scheduler_error(err, errlen, "output-head batch selected-token sequence failed");
    }
    for (uint32_t rel = 0; rel < n_slots; rel++) {
        if (!isfinite(logits[rel])) {
            return scheduler_error(err, errlen, "output-head batch logits contained non-finite values");
        }
    }
    return 0;
}

int ds4_v100_stage_scheduler_select_token(ds4_v100_stage_scheduler *sched,
                                          uint32_t *token,
                                          float *logit,
                                          char *err,
                                          size_t errlen) {
    return ds4_v100_stage_scheduler_select_token_slot(sched, 0, token, logit, err, errlen);
}
