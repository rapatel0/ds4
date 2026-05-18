#include "ds4_v100_scheduler.h"

#include "ds4_pack.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

struct ds4_v100_stage_scheduler {
    ds4_v100_context *ctx;
    ds4_pack *pack;
    ds4_gpu_arena *arena;
    ds4_v100_stage_info stage;
    ds4_v100_tensor_binding token_embedding;
    ds4_v100_layer_state states[DS4_V100_N_LAYERS];
    scheduler_layer_cache caches[DS4_V100_N_LAYERS];
    ds4_gpu_tensor *hc_a;
    ds4_gpu_tensor *hc_b;
    ds4_gpu_tensor *cur_hc;
    uint64_t uploaded_tensors;
    uint64_t uploaded_bytes;
    const void *model_map;
    uint64_t model_size;
    uint32_t raw_cap;
    uint32_t raw_window;
    uint32_t attn_comp_cap;
    uint32_t index_comp_cap;
    uint32_t indexer_top_k;
};

static int scheduler_error(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg ? msg : "scheduler error");
    return 1;
}

static int scheduler_errorf(char *err, size_t errlen, const char *fmt, int value) {
    if (err && errlen) snprintf(err, errlen, fmt, value);
    return 1;
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

void ds4_v100_stage_scheduler_close(ds4_v100_stage_scheduler *sched) {
    if (!sched) return;
    ds4_gpu_tensor_free(sched->hc_b);
    ds4_gpu_tensor_free(sched->hc_a);
    for (int i = 0; i < DS4_V100_N_LAYERS; i++) free_layer_cache(&sched->caches[i]);
    ds4_gpu_arena_close(sched->arena);
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

static int upload_stage_weights(ds4_v100_stage_scheduler *sched,
                                char *err,
                                size_t errlen) {
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

static int alloc_layer_cache(ds4_v100_stage_scheduler *sched,
                             int layer,
                             char *err,
                             size_t errlen) {
    const ds4_v100_layer_state *state = &sched->states[layer];
    scheduler_layer_cache *lc = &sched->caches[layer];
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

int ds4_v100_stage_scheduler_open(ds4_v100_stage_scheduler **out,
                                  const ds4_v100_stage_scheduler_options *opts,
                                  char *err,
                                  size_t errlen) {
    if (!out) return scheduler_error(err, errlen, "missing scheduler output");
    *out = NULL;
    if (!opts || !opts->pack_index_path || !opts->model_map || opts->model_size == 0) {
        return scheduler_error(err, errlen, "missing scheduler options");
    }
    if (opts->stage_id < 0 || opts->stage_id >= DS4_V100_EXPECTED_GPUS) {
        return scheduler_errorf(err, errlen, "invalid scheduler stage %d", opts->stage_id);
    }

    ds4_v100_stage_scheduler *sched =
        (ds4_v100_stage_scheduler *)calloc(1, sizeof(*sched));
    if (!sched) return scheduler_error(err, errlen, "failed to allocate scheduler");
    sched->model_map = opts->model_map;
    sched->model_size = opts->model_size;
    sched->raw_cap = opts->raw_cap ? opts->raw_cap : DS4_V100_SWA_ROWS;
    sched->raw_window = opts->raw_window ? opts->raw_window : DS4_V100_SWA_ROWS;
    sched->attn_comp_cap = opts->attn_comp_cap ? opts->attn_comp_cap : 1u;
    sched->index_comp_cap = opts->index_comp_cap ? opts->index_comp_cap : 1u;
    sched->indexer_top_k = opts->indexer_top_k ? opts->indexer_top_k : 1u;

    ds4_v100_context_options ctx_opts;
    ds4_v100_context_options_init(&ctx_opts);
    ctx_opts.pack_index_path = opts->pack_index_path;
    ctx_opts.kv_ctx_tokens = opts->kv_ctx_tokens ? opts->kv_ctx_tokens : 1048576;
    ctx_opts.kv_active_slots = 1;
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
    if (!sched->stage.owns_token_embedding) {
        ds4_v100_stage_scheduler_close(sched);
        return scheduler_error(err, errlen, "decode-token scheduler currently requires token-embedding stage");
    }
    if (ds4_v100_context_lookup_tensor_binding(sched->ctx,
                                               "token_embd.weight",
                                               &sched->token_embedding,
                                               err,
                                               errlen)) {
        ds4_v100_stage_scheduler_close(sched);
        return 1;
    }

    const uint64_t arena_bytes = ds4_pack_arena_bytes(sched->pack, sched->stage.gpu);
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

    for (int layer = sched->stage.layer_begin; layer <= sched->stage.layer_end; layer++) {
        if (ds4_v100_layer_state_init(&sched->states[layer], sched->ctx, layer, err, errlen) ||
            alloc_layer_cache(sched, layer, err, errlen)) {
            ds4_v100_stage_scheduler_close(sched);
            return 1;
        }
    }

    const uint64_t hc_bytes =
        (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS * sizeof(float);
    sched->hc_a = ds4_gpu_tensor_alloc(hc_bytes);
    sched->hc_b = ds4_gpu_tensor_alloc(hc_bytes);
    if (!sched->hc_a || !sched->hc_b) {
        ds4_v100_stage_scheduler_close(sched);
        return scheduler_error(err, errlen, "failed to allocate scheduler HC tensors");
    }
    sched->cur_hc = sched->hc_a;
    *out = sched;
    return 0;
}

int ds4_v100_stage_scheduler_decode_token(ds4_v100_stage_scheduler *sched,
                                          uint32_t token,
                                          uint32_t position,
                                          ds4_v100_stage_scheduler_report *report,
                                          char *err,
                                          size_t errlen) {
    if (!sched) return scheduler_error(err, errlen, "missing scheduler");
    if (sched->token_embedding.n_shape_dims != 2 ||
        sched->token_embedding.shape[0] != DS4_V100_HC_COLS) {
        return scheduler_error(err, errlen, "token embedding shape does not match HC width");
    }
    const uint32_t n_vocab = (uint32_t)sched->token_embedding.shape[1];
    if (token >= n_vocab) return scheduler_error(err, errlen, "token outside embedding vocab");
    if (!ds4_gpu_embed_token_hc_tensor(sched->hc_a,
                                       sched->model_map,
                                       sched->model_size,
                                       sched->token_embedding.source_offset,
                                       n_vocab,
                                       token,
                                       DS4_V100_HC_COLS,
                                       DS4_V100_HC_ROWS)) {
        return scheduler_error(err, errlen, "token embedding HC seed failed");
    }

    ds4_gpu_tensor *cur = sched->hc_a;
    ds4_gpu_tensor *next = sched->hc_b;
    ds4_v100_layer_execute_report last;
    memset(&last, 0, sizeof(last));
    uint32_t executed = 0;
    for (int layer = sched->stage.layer_begin; layer <= sched->stage.layer_end; layer++) {
        ds4_v100_layer_execute_config cfg = {
            .model_map = sched->model_map,
            .model_size = sched->model_size,
            .arena = sched->arena,
            .router_token = token,
            .position = position,
            .decode_cache = &sched->caches[layer].cache,
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
    }
    sched->cur_hc = cur;
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
    }
    return 0;
}

int ds4_v100_stage_scheduler_read_hc(const ds4_v100_stage_scheduler *sched,
                                     void *dst,
                                     uint64_t bytes) {
    if (!sched || !sched->cur_hc || !dst) return 0;
    const uint64_t hc_bytes =
        (uint64_t)DS4_V100_HC_ROWS * DS4_V100_HC_COLS * sizeof(float);
    if (bytes > hc_bytes) return 0;
    return ds4_gpu_tensor_read(sched->cur_hc, 0, dst, bytes);
}
