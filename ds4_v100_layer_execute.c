#include "ds4_v100_layer_execute.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static int exec_error(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return 1;
}

static int source_view(const ds4_v100_bound_matrix *m,
                       ds4_gpu_source_row_view *out,
                       char *err,
                       size_t errlen) {
    return ds4_v100_bound_matrix_source_view(m, out, err, errlen);
}

static int source_view_rows(const ds4_v100_bound_matrix *m,
                            uint32_t first_row,
                            uint32_t rows,
                            ds4_gpu_source_row_view *out,
                            char *err,
                            size_t errlen) {
    if (!m || !out) return exec_error(err, errlen, "missing grouped source view");
    if (first_row > m->rows || rows > m->rows - first_row) {
        return exec_error(err, errlen,
                          "grouped source rows outside matrix: first=%u rows=%u matrix_rows=%u",
                          first_row,
                          rows,
                          m->rows);
    }
    memset(out, 0, sizeof(*out));
    out->arena_offset = ds4_v100_bound_matrix_arena_offset(m) +
                        (uint64_t)first_row * m->row_bytes;
    out->byte_length = (uint64_t)rows * m->row_bytes;
    out->rows = rows;
    out->cols = m->cols;
    out->row_stride_bytes = (uint32_t)m->row_bytes;
    return 0;
}

static uint32_t layer_ratio(const ds4_v100_layer_state *state) {
    if (!state) return 0;
    switch (state->layer_class) {
    case DS4_V100_LAYER_RATIO_4: return 4u;
    case DS4_V100_LAYER_RATIO_128: return 128u;
    case DS4_V100_LAYER_SWA_ONLY:
    default: return 0;
    }
}

static int rope_tail_layer_tensor(ds4_gpu_tensor *x,
                                  uint32_t n_head,
                                  uint32_t head_dim,
                                  uint32_t pos,
                                  const ds4_v100_layer_state *state,
                                  bool inverse) {
    const uint32_t ratio = layer_ratio(state);
    const bool compressed = ratio != 0;
    const float freq_base = compressed ? 160000.0f : 10000.0f;
    const float freq_scale = compressed ? (1.0f / 16.0f) : 1.0f;
    const float ext_factor = compressed ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    return ds4_gpu_rope_tail_tensor(x,
                                    1,
                                    n_head,
                                    head_dim,
                                    DS4_V100_N_ROT,
                                    pos,
                                    compressed ? 65536u : 0u,
                                    inverse,
                                    freq_base,
                                    freq_scale,
                                    ext_factor,
                                    attn_factor,
                                    32.0f,
                                    1.0f);
}

typedef struct {
    const ds4_gpu_tensor *raw_kv;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    const ds4_gpu_tensor *compressed_kv;
    uint32_t n_compressed;
    const ds4_gpu_tensor *compressed_mask;
    uint32_t use_compressed_mask;
    const ds4_gpu_tensor *indexed_topk;
    uint32_t indexed_top_k;
    bool use_indexed_attention;
} ds4_v100_attention_inputs;

static float compressed_attn_factor(void) {
    return 1.0f / (1.0f + 0.1f * logf(16.0f));
}

static int grouped_attention_output(const ds4_v100_layer_state *state,
                                    const ds4_v100_layer_execute_config *cfg,
                                    const ds4_gpu_tensor *heads,
                                    ds4_gpu_tensor *low,
                                    ds4_gpu_tensor *out,
                                    char *err,
                                    size_t errlen) {
    if (state->attn_output_a.rows != DS4_V100_OUT_GROUPS * DS4_V100_OUT_GROUP_RANK ||
        state->attn_output_a.cols != DS4_V100_OUT_GROUP_DIM ||
        state->attn_output_b.rows != state->hidden_size ||
        state->attn_output_b.cols != DS4_V100_OUT_GROUPS * DS4_V100_OUT_GROUP_RANK) {
        return exec_error(err, errlen, "attention grouped output dimensions do not match DS4");
    }
    if (!heads || !low || !out ||
        ds4_gpu_tensor_bytes(heads) < (uint64_t)DS4_V100_N_HEAD * DS4_V100_HEAD_DIM * sizeof(float) ||
        ds4_gpu_tensor_bytes(low) < (uint64_t)DS4_V100_OUT_GROUPS * DS4_V100_OUT_GROUP_RANK * sizeof(float) ||
        ds4_gpu_tensor_bytes(out) < (uint64_t)state->hidden_size * sizeof(float)) {
        return exec_error(err, errlen, "attention grouped output tensor is too small");
    }

    for (uint32_t g = 0; g < DS4_V100_OUT_GROUPS; g++) {
        ds4_gpu_source_row_view a_view;
        if (source_view_rows(&state->attn_output_a,
                             g * DS4_V100_OUT_GROUP_RANK,
                             DS4_V100_OUT_GROUP_RANK,
                             &a_view,
                             err,
                             errlen)) {
            return 1;
        }

        ds4_gpu_tensor *head_view = ds4_gpu_tensor_view(
                heads,
                (uint64_t)g * DS4_V100_OUT_GROUP_DIM * sizeof(float),
                (uint64_t)DS4_V100_OUT_GROUP_DIM * sizeof(float));
        ds4_gpu_tensor *low_view = ds4_gpu_tensor_view(
                low,
                (uint64_t)g * DS4_V100_OUT_GROUP_RANK * sizeof(float),
                (uint64_t)DS4_V100_OUT_GROUP_RANK * sizeof(float));
        if (!head_view || !low_view) {
            ds4_gpu_tensor_free(low_view);
            ds4_gpu_tensor_free(head_view);
            return exec_error(err, errlen, "failed to create grouped output tensor views");
        }
        const int rc = ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena,
                                                             &a_view,
                                                             head_view,
                                                             low_view);
        ds4_gpu_tensor_free(low_view);
        ds4_gpu_tensor_free(head_view);
        if (rc != 0) return exec_error(err, errlen, "attention output_a grouped matmul failed");
    }

    ds4_gpu_source_row_view b_view;
    if (source_view(&state->attn_output_b, &b_view, err, errlen)) return 1;
    if (ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena, &b_view, low, out) != 0) {
        return exec_error(err, errlen, "attention output_b matmul failed");
    }
    return 0;
}

static int validate_decode_cache(const ds4_v100_layer_state *state,
                                 const ds4_v100_layer_decode_cache *cache,
                                 char *err,
                                 size_t errlen) {
    if (!cache) return 0;
    if (!cache->raw_kv || cache->raw_cap == 0) {
        return exec_error(err, errlen, "decode cache requires raw KV storage");
    }
    if (ds4_gpu_tensor_bytes(cache->raw_kv) <
        (uint64_t)cache->raw_cap * DS4_V100_HEAD_DIM * sizeof(float)) {
        return exec_error(err, errlen, "decode cache raw KV tensor is too small");
    }
    if (state->compress_ratio == 0) return 0;
    const uint32_t coff = state->compress_ratio == 4u ? 2u : 1u;
    const uint32_t comp_width = coff * DS4_V100_HEAD_DIM;
    const uint32_t comp_state_rows = coff * state->compress_ratio;
    if (!state->has_attention_compressor ||
        !cache->attn_state_kv || !cache->attn_state_score || !cache->attn_comp_kv ||
        cache->attn_comp_cap == 0 ||
        ds4_gpu_tensor_bytes(cache->attn_state_kv) <
            (uint64_t)comp_state_rows * comp_width * sizeof(float) ||
        ds4_gpu_tensor_bytes(cache->attn_state_score) <
            (uint64_t)comp_state_rows * comp_width * sizeof(float) ||
        ds4_gpu_tensor_bytes(cache->attn_comp_kv) <
            (uint64_t)cache->attn_comp_cap * DS4_V100_HEAD_DIM * sizeof(float)) {
        return exec_error(err, errlen, "decode cache attention compressor tensors are invalid");
    }
    if (cache->n_attn_comp > cache->attn_comp_cap) {
        return exec_error(err, errlen, "decode cache attention compressed count exceeds capacity");
    }
    if (state->compress_ratio == 4u) {
        const uint32_t index_width = 2u * DS4_V100_INDEXER_HEAD_DIM;
        const uint32_t index_state_rows = 2u * state->compress_ratio;
        const uint32_t top_k = cache->indexer_top_k ? cache->indexer_top_k : DS4_V100_INDEXER_TOP_K;
        if (!state->has_indexer ||
            !cache->index_state_kv || !cache->index_state_score || !cache->index_comp_kv ||
            !cache->indexer_topk ||
            cache->index_comp_cap == 0 ||
            top_k == 0 ||
            top_k > DS4_V100_INDEXER_TOP_K ||
            ds4_gpu_tensor_bytes(cache->index_state_kv) <
                (uint64_t)index_state_rows * index_width * sizeof(float) ||
            ds4_gpu_tensor_bytes(cache->index_state_score) <
                (uint64_t)index_state_rows * index_width * sizeof(float) ||
            ds4_gpu_tensor_bytes(cache->index_comp_kv) <
                (uint64_t)cache->index_comp_cap * DS4_V100_INDEXER_HEAD_DIM * sizeof(float) ||
            ds4_gpu_tensor_bytes(cache->indexer_topk) < (uint64_t)top_k * sizeof(uint32_t)) {
            return exec_error(err, errlen, "decode cache ratio-4 indexer tensors are invalid");
        }
        if (cache->n_index_comp > cache->index_comp_cap) {
            return exec_error(err, errlen, "decode cache indexer compressed count exceeds capacity");
        }
    }
    return 0;
}

static int validate_execute_common(const ds4_v100_layer_state *state,
                                   const ds4_v100_layer_execute_config *cfg,
                                   char *err,
                                   size_t errlen) {
    if (!state || !cfg) {
        return exec_error(err, errlen, "missing layer executor argument");
    }
    if (!cfg->arena || !cfg->model_map || cfg->model_size == 0) {
        return exec_error(err, errlen, "missing layer executor arena or model map");
    }
    if (state->q_width != DS4_V100_N_HEAD * DS4_V100_HEAD_DIM ||
        state->hidden_size != DS4_V100_OUT_GROUP_DIM ||
        state->routes_per_token == 0 ||
        state->routes_per_token > 6) {
        return exec_error(err, errlen, "unsupported DS4 V100 layer dimensions");
    }
    const uint32_t kv_width = state->kv_latent_width;
    const uint32_t n_raw = cfg->n_raw ? cfg->n_raw : 1u;
    const uint32_t raw_cap = cfg->raw_cap ? cfg->raw_cap : n_raw;
    if (cfg->raw_kv &&
        ds4_gpu_tensor_bytes(cfg->raw_kv) < (uint64_t)raw_cap * kv_width * sizeof(float)) {
        return exec_error(err, errlen, "raw KV tensor is too small");
    }
    if (cfg->n_compressed &&
        (!cfg->compressed_kv ||
         ds4_gpu_tensor_bytes(cfg->compressed_kv) <
             (uint64_t)cfg->n_compressed * kv_width * sizeof(float))) {
        return exec_error(err, errlen, "compressed KV tensor is too small");
    }
    if (cfg->use_compressed_mask &&
        (!cfg->compressed_mask ||
         ds4_gpu_tensor_bytes(cfg->compressed_mask) <
             (uint64_t)cfg->n_compressed * sizeof(float))) {
        return exec_error(err, errlen, "compressed mask tensor is too small");
    }
    if (validate_decode_cache(state, cfg->decode_cache, err, errlen)) {
        return 1;
    }
    return 0;
}

static int prepare_decode_cache_attention(
        const ds4_v100_layer_state          *state,
        const ds4_v100_layer_execute_config *cfg,
        const ds4_gpu_tensor                *attn_norm,
        const ds4_gpu_tensor                *q_a_norm,
        ds4_gpu_tensor                      *kv,
        ds4_v100_attention_inputs           *inputs,
        char                                *err,
        size_t                               errlen) {
    ds4_v100_layer_decode_cache *cache = cfg->decode_cache;
    if (!cache) return 0;

    const uint32_t raw_window = cache->raw_window ? cache->raw_window : 128u;
    const uint32_t raw_cap = cache->raw_cap;
    uint32_t n_raw = cfg->position + 1u < raw_window ? cfg->position + 1u : raw_window;
    if (n_raw > raw_cap) n_raw = raw_cap;
    const uint32_t raw_row = cfg->position % raw_cap;
    const uint32_t raw_start = (cfg->position + raw_cap + 1u - n_raw) % raw_cap;

    int raw_ok = 0;
    if (cfg->fp8_kv_cache) {
        raw_ok = ds4_gpu_kv_fp8_store_raw_tensor(kv,
                                                 cache->raw_kv,
                                                 raw_cap,
                                                 raw_row,
                                                 DS4_V100_HEAD_DIM,
                                                 DS4_V100_N_ROT);
    } else {
        raw_ok = ds4_gpu_store_raw_kv_tensor(cache->raw_kv,
                                             kv,
                                             raw_cap,
                                             raw_row,
                                             DS4_V100_HEAD_DIM);
    }
    if (!raw_ok) {
        return exec_error(err, errlen, "decode cache raw KV update failed");
    }
    inputs->raw_kv = cache->raw_kv;
    inputs->n_raw = n_raw;
    inputs->raw_cap = raw_cap;
    inputs->raw_start = raw_start;

    if (state->compress_ratio == 0u) return 0;

    const uint32_t ratio = state->compress_ratio;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t comp_width = coff * DS4_V100_HEAD_DIM;
    const bool emit = ((cfg->position + 1u) % ratio) == 0u;
    if (emit && cache->n_attn_comp >= cache->attn_comp_cap) {
        return exec_error(err, errlen, "decode cache attention compressed capacity exceeded");
    }

    ds4_gpu_bf16_matrix_view comp_kv_v;
    ds4_gpu_bf16_matrix_view comp_sc_v;
    if (ds4_v100_bound_matrix_bf16_view(&state->attn_compressor_kv,
                                        &comp_kv_v,
                                        err,
                                        errlen) ||
        ds4_v100_bound_matrix_bf16_view(&state->attn_compressor_gate,
                                        &comp_sc_v,
                                        err,
                                        errlen)) {
        return 1;
    }

    ds4_gpu_tensor *comp_kv_cur = ds4_gpu_tensor_alloc((uint64_t)comp_width * sizeof(float));
    ds4_gpu_tensor *comp_sc_cur = ds4_gpu_tensor_alloc((uint64_t)comp_width * sizeof(float));
    int rc = 1;
    if (!comp_kv_cur || !comp_sc_cur) {
        exec_error(err, errlen, "failed to allocate attention compressor scratch");
        goto done;
    }
    if (ds4_gpu_arena_bf16_matmul_f32(cfg->arena, &comp_kv_v, attn_norm, comp_kv_cur) != 0 ||
        ds4_gpu_arena_bf16_matmul_f32(cfg->arena, &comp_sc_v, attn_norm, comp_sc_cur) != 0) {
        exec_error(err, errlen, "attention compressor projection failed");
        goto done;
    }
    if (!ds4_gpu_compressor_update_tensor(comp_kv_cur,
                                          comp_sc_cur,
                                          cache->attn_state_kv,
                                          cache->attn_state_score,
                                          cache->attn_comp_kv,
                                          cfg->model_map,
                                          cfg->model_size,
                                          state->attn_compressor_ape.binding.source_offset,
                                          0,
                                          state->attn_compressor_norm.source_offset,
                                          0,
                                          DS4_V100_HEAD_DIM,
                                          ratio,
                                          cfg->position,
                                          cache->n_attn_comp,
                                          DS4_V100_N_ROT,
                                          65536u,
                                          160000.0f,
                                          1.0f / 16.0f,
                                          1.0f,
                                          compressed_attn_factor(),
                                          32.0f,
                                          1.0f,
                                          DS4_V100_RMS_EPS)) {
        exec_error(err, errlen, "attention compressor update failed");
        goto done;
    }
    if (emit) {
        ds4_gpu_tensor *row = ds4_gpu_tensor_view(
                cache->attn_comp_kv,
                (uint64_t)cache->n_attn_comp * DS4_V100_HEAD_DIM * sizeof(float),
                (uint64_t)DS4_V100_HEAD_DIM * sizeof(float));
        if (!row) {
            exec_error(err, errlen, "failed to view emitted attention compressed row");
            goto done;
        }
        int ok = 1;
        if (cfg->fp8_kv_cache) {
            ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(row, 1, DS4_V100_HEAD_DIM, DS4_V100_N_ROT);
        }
        if (ok) ok = ds4_gpu_f16_round_tensor(row, DS4_V100_HEAD_DIM);
        ds4_gpu_tensor_free(row);
        if (!ok) {
            exec_error(err, errlen, "attention compressed row KV round-trip failed");
            goto done;
        }
        cache->n_attn_comp++;
    }

    inputs->compressed_kv = cache->attn_comp_kv;
    inputs->n_compressed = cache->n_attn_comp;
    inputs->compressed_mask = NULL;
    inputs->use_compressed_mask = 0;

    if (ratio == 4u) {
        const uint32_t index_width = 2u * DS4_V100_INDEXER_HEAD_DIM;
        if (emit && cache->n_index_comp >= cache->index_comp_cap) {
            exec_error(err, errlen, "decode cache indexer compressed capacity exceeded");
            goto done;
        }
        ds4_gpu_bf16_matrix_view index_kv_v;
        ds4_gpu_bf16_matrix_view index_sc_v;
        if (ds4_v100_bound_matrix_bf16_view(&state->indexer_compressor_kv,
                                            &index_kv_v,
                                            err,
                                            errlen) ||
            ds4_v100_bound_matrix_bf16_view(&state->indexer_compressor_gate,
                                            &index_sc_v,
                                            err,
                                            errlen)) {
            goto done;
        }
        ds4_gpu_tensor *index_kv_cur = ds4_gpu_tensor_alloc((uint64_t)index_width * sizeof(float));
        ds4_gpu_tensor *index_sc_cur = ds4_gpu_tensor_alloc((uint64_t)index_width * sizeof(float));
        if (!index_kv_cur || !index_sc_cur) {
            ds4_gpu_tensor_free(index_sc_cur);
            ds4_gpu_tensor_free(index_kv_cur);
            exec_error(err, errlen, "failed to allocate indexer compressor scratch");
            goto done;
        }
        const int projected =
            ds4_gpu_arena_bf16_matmul_f32(cfg->arena, &index_kv_v, attn_norm, index_kv_cur) == 0 &&
            ds4_gpu_arena_bf16_matmul_f32(cfg->arena, &index_sc_v, attn_norm, index_sc_cur) == 0;
        if (!projected ||
            !ds4_gpu_compressor_update_tensor(index_kv_cur,
                                              index_sc_cur,
                                              cache->index_state_kv,
                                              cache->index_state_score,
                                              cache->index_comp_kv,
                                              cfg->model_map,
                                              cfg->model_size,
                                              state->indexer_compressor_ape.binding.source_offset,
                                              0,
                                              state->indexer_compressor_norm.source_offset,
                                              0,
                                              DS4_V100_INDEXER_HEAD_DIM,
                                              ratio,
                                              cfg->position,
                                              cache->n_index_comp,
                                              DS4_V100_N_ROT,
                                              65536u,
                                              160000.0f,
                                              1.0f / 16.0f,
                                              1.0f,
                                              compressed_attn_factor(),
                                              32.0f,
                                              1.0f,
                                              DS4_V100_RMS_EPS)) {
            ds4_gpu_tensor_free(index_sc_cur);
            ds4_gpu_tensor_free(index_kv_cur);
            exec_error(err, errlen, projected ? "indexer compressor update failed" :
                       "indexer compressor projection failed");
            goto done;
        }
        ds4_gpu_tensor_free(index_sc_cur);
        ds4_gpu_tensor_free(index_kv_cur);
        if (emit) {
            ds4_gpu_tensor *row = ds4_gpu_tensor_view(
                    cache->index_comp_kv,
                    (uint64_t)cache->n_index_comp * DS4_V100_INDEXER_HEAD_DIM * sizeof(float),
                    (uint64_t)DS4_V100_INDEXER_HEAD_DIM * sizeof(float));
            const int ok = row && ds4_gpu_f16_round_tensor(row, DS4_V100_INDEXER_HEAD_DIM);
            ds4_gpu_tensor_free(row);
            if (!ok) {
                exec_error(err, errlen, "indexer compressed row F16 round-trip failed");
                goto done;
            }
            cache->n_index_comp++;
        }

        const uint32_t top_k = cache->indexer_top_k ? cache->indexer_top_k : DS4_V100_INDEXER_TOP_K;
        if (cache->n_index_comp > top_k) {
            ds4_gpu_source_row_view index_q_v;
            ds4_gpu_bf16_matrix_view index_w_v;
            if (source_view(&state->indexer_attn_q_b, &index_q_v, err, errlen) ||
                ds4_v100_bound_matrix_bf16_view(&state->indexer_proj, &index_w_v, err, errlen)) {
                goto done;
            }
            ds4_gpu_tensor *indexer_q = ds4_gpu_tensor_alloc(
                    (uint64_t)DS4_V100_INDEXER_HEAD * DS4_V100_INDEXER_HEAD_DIM * sizeof(float));
            ds4_gpu_tensor *indexer_w = ds4_gpu_tensor_alloc(
                    (uint64_t)DS4_V100_INDEXER_HEAD * sizeof(float));
            ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(
                    (uint64_t)cache->n_index_comp * sizeof(float));
            if (!indexer_q || !indexer_w || !scores) {
                ds4_gpu_tensor_free(scores);
                ds4_gpu_tensor_free(indexer_w);
                ds4_gpu_tensor_free(indexer_q);
                exec_error(err, errlen, "failed to allocate indexer top-k scratch");
                goto done;
            }
            const float scale = 1.0f / sqrtf((float)(DS4_V100_INDEXER_HEAD * DS4_V100_INDEXER_HEAD_DIM));
            const int ok =
                ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena, &index_q_v, q_a_norm, indexer_q) == 0 &&
                rope_tail_layer_tensor(indexer_q,
                                       DS4_V100_INDEXER_HEAD,
                                       DS4_V100_INDEXER_HEAD_DIM,
                                       cfg->position,
                                       state,
                                       false) &&
                ds4_gpu_arena_bf16_matmul_f32(cfg->arena, &index_w_v, attn_norm, indexer_w) == 0 &&
                ds4_gpu_indexer_score_one_tensor(scores,
                                                 indexer_q,
                                                 indexer_w,
                                                 cache->index_comp_kv,
                                                 cache->n_index_comp,
                                                 DS4_V100_INDEXER_HEAD,
                                                 DS4_V100_INDEXER_HEAD_DIM,
                                                 scale) &&
                ds4_gpu_indexer_topk_tensor(cache->indexer_topk,
                                            scores,
                                            cache->n_index_comp,
                                            1,
                                            top_k);
            ds4_gpu_tensor_free(scores);
            ds4_gpu_tensor_free(indexer_w);
            ds4_gpu_tensor_free(indexer_q);
            if (!ok) {
                exec_error(err, errlen, "indexer top-k sequence failed");
                goto done;
            }
            inputs->indexed_topk = cache->indexer_topk;
            inputs->indexed_top_k = top_k;
            inputs->use_indexed_attention = true;
        }
    }

    rc = 0;

done:
    ds4_gpu_tensor_free(comp_sc_cur);
    ds4_gpu_tensor_free(comp_kv_cur);
    return rc;
}

static int execute_attention_output(const ds4_v100_layer_state *state,
                                    const ds4_v100_layer_execute_config *cfg,
                                    const ds4_gpu_tensor *hidden,
                                    ds4_gpu_tensor *attn_out,
                                    char *err,
                                    size_t errlen) {
    const uint32_t hidden_n = state->hidden_size;
    const uint32_t q_rank = state->q_lora_rank;
    const uint32_t q_width = state->q_width;
    const uint32_t kv_width = state->kv_latent_width;
    const uint32_t out_rank = state->attention_output_rank;
    const uint32_t n_raw = cfg->n_raw ? cfg->n_raw : 1u;
    const uint32_t raw_cap = cfg->raw_cap ? cfg->raw_cap : n_raw;
    const uint32_t raw_start = cfg->raw_start;

    if (!hidden || !attn_out ||
        ds4_gpu_tensor_bytes(hidden) < (uint64_t)hidden_n * sizeof(float) ||
        ds4_gpu_tensor_bytes(attn_out) < (uint64_t)hidden_n * sizeof(float)) {
        return exec_error(err, errlen, "attention body tensors are too small");
    }

    ds4_gpu_source_row_view q_a_v;
    ds4_gpu_source_row_view q_b_v;
    ds4_gpu_source_row_view kv_v;
    if (source_view(&state->attn_q_a, &q_a_v, err, errlen) ||
        source_view(&state->attn_q_b, &q_b_v, err, errlen) ||
        source_view(&state->attn_kv_latent, &kv_v, err, errlen)) {
        return 1;
    }

    ds4_gpu_tensor *attn_norm = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
    ds4_gpu_tensor *q_a = ds4_gpu_tensor_alloc((uint64_t)q_rank * sizeof(float));
    ds4_gpu_tensor *q_a_norm = ds4_gpu_tensor_alloc((uint64_t)q_rank * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)q_width * sizeof(float));
    ds4_gpu_tensor *kv_raw = ds4_gpu_tensor_alloc((uint64_t)kv_width * sizeof(float));
    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc((uint64_t)kv_width * sizeof(float));
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc((uint64_t)q_width * sizeof(float));
    ds4_gpu_tensor *low = ds4_gpu_tensor_alloc((uint64_t)out_rank * sizeof(float));

    int rc = 1;
    if (!attn_norm || !q_a || !q_a_norm || !q || !kv_raw || !kv || !heads || !low) {
        exec_error(err, errlen, "failed to allocate attention body tensors");
        goto done;
    }

    if (!ds4_gpu_rms_norm_weight_tensor(attn_norm,
                                        hidden,
                                        cfg->model_map,
                                        cfg->model_size,
                                        state->attn_norm.source_offset,
                                        hidden_n,
                                        DS4_V100_RMS_EPS) ||
        ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena, &q_a_v, attn_norm, q_a) != 0 ||
        !ds4_gpu_rms_norm_weight_tensor(q_a_norm,
                                        q_a,
                                        cfg->model_map,
                                        cfg->model_size,
                                        state->attn_q_a_norm.source_offset,
                                        q_rank,
                                        DS4_V100_RMS_EPS) ||
        ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena, &q_b_v, q_a_norm, q) != 0 ||
        !ds4_gpu_head_rms_norm_tensor(q, 1, DS4_V100_N_HEAD, DS4_V100_HEAD_DIM, DS4_V100_RMS_EPS) ||
        !rope_tail_layer_tensor(q,
                                DS4_V100_N_HEAD,
                                DS4_V100_HEAD_DIM,
                                cfg->position,
                                state,
                                false) ||
        ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena, &kv_v, attn_norm, kv_raw) != 0 ||
        !ds4_gpu_rms_norm_weight_tensor(kv,
                                        kv_raw,
                                        cfg->model_map,
                                        cfg->model_size,
                                        state->attn_kv_a_norm.source_offset,
                                        kv_width,
                                        DS4_V100_RMS_EPS) ||
        !rope_tail_layer_tensor(kv,
                                1,
                                DS4_V100_HEAD_DIM,
                                cfg->position,
                                state,
                                false)) {
        exec_error(err, errlen, "attention projection sequence failed");
        goto done;
    }

    ds4_v100_attention_inputs attn_inputs = {
        .raw_kv = cfg->raw_kv ? cfg->raw_kv : kv,
        .n_raw = n_raw,
        .raw_cap = raw_cap,
        .raw_start = raw_start,
        .compressed_kv = cfg->compressed_kv,
        .n_compressed = cfg->n_compressed,
        .compressed_mask = cfg->compressed_mask,
        .use_compressed_mask = cfg->use_compressed_mask ? 1u : 0u,
        .indexed_topk = NULL,
        .indexed_top_k = 0,
        .use_indexed_attention = false,
    };
    if (prepare_decode_cache_attention(state,
                                       cfg,
                                       attn_norm,
                                       q_a_norm,
                                       kv,
                                       &attn_inputs,
                                       err,
                                       errlen)) {
        goto done;
    }

    int attention_ok = 0;
    if (attn_inputs.use_indexed_attention) {
        attention_ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                heads,
                cfg->model_map,
                cfg->model_size,
                state->attn_sinks.source_offset,
                q,
                attn_inputs.raw_kv,
                attn_inputs.compressed_kv,
                attn_inputs.indexed_topk,
                1,
                cfg->position,
                attn_inputs.n_raw,
                attn_inputs.raw_cap,
                attn_inputs.raw_start,
                attn_inputs.n_compressed,
                attn_inputs.indexed_top_k,
                cfg->decode_cache && cfg->decode_cache->raw_window ? cfg->decode_cache->raw_window : 128u,
                state->compress_ratio,
                DS4_V100_N_HEAD,
                DS4_V100_HEAD_DIM);
    } else {
        attention_ok = ds4_gpu_attention_decode_heads_tensor(heads,
                                                             cfg->model_map,
                                                             cfg->model_size,
                                                             state->attn_sinks.source_offset,
                                                             q,
                                                             attn_inputs.raw_kv,
                                                             attn_inputs.n_raw,
                                                             attn_inputs.raw_cap,
                                                             attn_inputs.raw_start,
                                                             attn_inputs.compressed_kv,
                                                             attn_inputs.n_compressed,
                                                             attn_inputs.compressed_mask,
                                                             attn_inputs.use_compressed_mask,
                                                             DS4_V100_N_HEAD,
                                                             DS4_V100_HEAD_DIM);
    }
    if (!attention_ok) {
        exec_error(err, errlen, "attention softmax failed");
        goto done;
    }
    if (!rope_tail_layer_tensor(heads,
                                DS4_V100_N_HEAD,
                                DS4_V100_HEAD_DIM,
                                cfg->position,
                                state,
                                true)) {
        exec_error(err, errlen, "attention inverse rope failed");
        goto done;
    }

    if (grouped_attention_output(state, cfg, heads, low, attn_out, err, errlen)) {
        goto done;
    }
    rc = 0;

done:
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(kv_raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(q_a_norm);
    ds4_gpu_tensor_free(q_a);
    ds4_gpu_tensor_free(attn_norm);
    return rc;
}

static int execute_ffn_delta(const ds4_v100_layer_state *state,
                             const ds4_v100_layer_execute_config *cfg,
                             const ds4_gpu_tensor *ffn_input,
                             ds4_gpu_tensor *ffn_delta,
                             ds4_v100_layer_execute_report *report,
                             char *err,
                             size_t errlen) {
    const bool hash_mode = state->router_kind == DS4_V100_ROUTER_HASH && state->has_hash_router;
    const bool bias_mode = state->router_kind == DS4_V100_ROUTER_BIAS && state->has_bias_router;
    if (!hash_mode && !bias_mode) {
        return exec_error(err, errlen, "layer executor requires hash or bias router metadata");
    }

    const uint32_t hidden = state->hidden_size;
    const uint32_t mid = state->intermediate_size;
    ds4_gpu_source_row_view router_v;
    ds4_gpu_source_row_view shared_gate_v;
    ds4_gpu_source_row_view shared_up_v;
    ds4_gpu_source_row_view shared_down_v;
    if (source_view(&state->router, &router_v, err, errlen) ||
        source_view(&state->shared_gate, &shared_gate_v, err, errlen) ||
        source_view(&state->shared_up, &shared_up_v, err, errlen) ||
        source_view(&state->shared_down, &shared_down_v, err, errlen)) {
        return 1;
    }

    ds4_gpu_tensor *router_t = ds4_gpu_tensor_alloc(256u * sizeof(float));
    ds4_gpu_tensor *probs_t = ds4_gpu_tensor_alloc(256u * sizeof(float));
    ds4_gpu_tensor *selected_t = ds4_gpu_tensor_alloc(6u * sizeof(int32_t));
    ds4_gpu_tensor *weights_t = ds4_gpu_tensor_alloc(6u * sizeof(float));
    ds4_gpu_tensor *mid_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *route_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *accum_a = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *accum_b = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *shared_gate_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *shared_up_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *shared_mid_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *shared_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));

    int rc = 1;
    int32_t selected[6] = {0};
    float weights[6] = {0};
    if (!router_t || !probs_t || !selected_t || !weights_t ||
        !mid_t || !route_t || !accum_a || !accum_b ||
        !shared_gate_t || !shared_up_t || !shared_mid_t || !shared_t) {
        exec_error(err, errlen, "failed to allocate FFN executor tensors");
        goto done;
    }

    if (ds4_gpu_arena_f32_matmul_f32(cfg->arena, &router_v, ffn_input, router_t) != 0) {
        exec_error(err, errlen, "router matmul failed");
        goto done;
    }
    if (!ds4_gpu_router_select_tensor(selected_t,
                                      weights_t,
                                      probs_t,
                                      cfg->model_map,
                                      cfg->model_size,
                                      bias_mode ? state->router_bias.source_offset : 0,
                                      hash_mode ? state->router_hash.source_offset : 0,
                                      hash_mode ? (uint32_t)state->router_hash.shape[1] : 0,
                                      cfg->router_token,
                                      0,
                                      0,
                                      bias_mode,
                                      hash_mode,
                                      router_t)) {
        exec_error(err, errlen, "router select failed");
        goto done;
    }
    if (!ds4_gpu_tensor_read(selected_t, 0, selected, sizeof(selected)) ||
        !ds4_gpu_tensor_read(weights_t, 0, weights, sizeof(weights))) {
        exec_error(err, errlen, "router readback failed");
        goto done;
    }
    for (uint32_t route = 0; route < state->routes_per_token; route++) {
        if (selected[route] < 0 || (uint32_t)selected[route] >= state->routed_experts) {
            exec_error(err, errlen, "router selected invalid expert %d", selected[route]);
            goto done;
        }
    }

    if (!ds4_gpu_tensor_fill_f32(accum_a, 0.0f, hidden)) {
        exec_error(err, errlen, "route accumulator fill failed");
        goto done;
    }
    ds4_gpu_tensor *accum = accum_a;
    ds4_gpu_tensor *next = accum_b;
    for (uint32_t route = 0; route < state->routes_per_token; route++) {
        ds4_v100_route_matrices route_mats;
        ds4_gpu_source_row_view gate_v;
        ds4_gpu_source_row_view up_v;
        ds4_gpu_source_row_view down_v;
        if (ds4_v100_layer_state_route_matrices(state,
                                                (uint32_t)selected[route],
                                                &route_mats,
                                                err,
                                                errlen) ||
            source_view(&route_mats.gate, &gate_v, err, errlen) ||
            source_view(&route_mats.up, &up_v, err, errlen) ||
            source_view(&route_mats.down, &down_v, err, errlen)) {
            goto done;
        }
        if (ds4_gpu_arena_mxfp4_pair_swiglu_f32(cfg->arena,
                                                &gate_v,
                                                &up_v,
                                                ffn_input,
                                                mid_t,
                                                10.0f,
                                                weights[route]) != 0 ||
            ds4_gpu_arena_mxfp4_matmul_f32(cfg->arena, &down_v, mid_t, route_t) != 0 ||
            !ds4_gpu_add_tensor(next, accum, route_t, hidden)) {
            exec_error(err, errlen, "routed FFN route %u failed", route);
            goto done;
        }
        ds4_gpu_tensor *tmp = accum;
        accum = next;
        next = tmp;
    }

    if (ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena, &shared_gate_v, ffn_input, shared_gate_t) != 0 ||
        ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena, &shared_up_v, ffn_input, shared_up_t) != 0 ||
        !ds4_gpu_swiglu_tensor(shared_mid_t, shared_gate_t, shared_up_t, mid, 10.0f, 1.0f) ||
        ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena, &shared_down_v, shared_mid_t, shared_t) != 0 ||
        !ds4_gpu_add_tensor(ffn_delta, accum, shared_t, hidden)) {
        exec_error(err, errlen, "shared FFN failed");
        goto done;
    }

    if (report) {
        memset(report, 0, sizeof(*report));
        report->routes = state->routes_per_token;
        for (uint32_t i = 0; i < state->routes_per_token && i < 6u; i++) {
            report->selected_experts[i] = selected[i];
            report->route_weights[i] = weights[i];
        }
    }
    rc = 0;

done:
    ds4_gpu_tensor_free(shared_t);
    ds4_gpu_tensor_free(shared_mid_t);
    ds4_gpu_tensor_free(shared_up_t);
    ds4_gpu_tensor_free(shared_gate_t);
    ds4_gpu_tensor_free(accum_b);
    ds4_gpu_tensor_free(accum_a);
    ds4_gpu_tensor_free(route_t);
    ds4_gpu_tensor_free(mid_t);
    ds4_gpu_tensor_free(weights_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(probs_t);
    ds4_gpu_tensor_free(router_t);
    return rc;
}

int ds4_v100_layer_execute_decode(
        const ds4_v100_layer_state          *state,
        const ds4_v100_layer_execute_config *cfg,
        const ds4_gpu_tensor                *hidden,
        ds4_gpu_tensor                      *next_hidden,
        ds4_v100_layer_execute_report       *report,
        char                                *err,
        size_t                               errlen) {
    if (!hidden || !next_hidden) return exec_error(err, errlen, "missing hidden tensor");
    if (validate_execute_common(state, cfg, err, errlen)) return 1;
    const uint32_t hidden_n = state->hidden_size;
    if (ds4_gpu_tensor_bytes(hidden) < (uint64_t)hidden_n * sizeof(float) ||
        ds4_gpu_tensor_bytes(next_hidden) < (uint64_t)hidden_n * sizeof(float)) {
        return exec_error(err, errlen, "hidden tensor is too small");
    }

    ds4_gpu_tensor *attn_out = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
    ds4_gpu_tensor *residual = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
    ds4_gpu_tensor *ffn_norm = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
    ds4_gpu_tensor *ffn_delta = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));

    int rc = 1;
    if (!attn_out || !residual || !ffn_norm || !ffn_delta) {
        exec_error(err, errlen, "failed to allocate layer executor tensors");
        goto done;
    }

    if (execute_attention_output(state, cfg, hidden, attn_out, err, errlen) ||
        !ds4_gpu_add_tensor(residual, hidden, attn_out, hidden_n) ||
        !ds4_gpu_rms_norm_weight_tensor(ffn_norm,
                                        residual,
                                        cfg->model_map,
                                        cfg->model_size,
                                        state->ffn_norm.source_offset,
                                        hidden_n,
                                        DS4_V100_RMS_EPS) ||
        execute_ffn_delta(state, cfg, ffn_norm, ffn_delta, report, err, errlen) ||
        !ds4_gpu_add_tensor(next_hidden, residual, ffn_delta, hidden_n)) {
        if (err && err[0] == '\0') exec_error(err, errlen, "layer residual/FFN sequence failed");
        goto done;
    }

    rc = 0;

done:
    ds4_gpu_tensor_free(ffn_delta);
    ds4_gpu_tensor_free(ffn_norm);
    ds4_gpu_tensor_free(residual);
    ds4_gpu_tensor_free(attn_out);
    return rc;
}

int ds4_v100_layer_execute_hc_decode(
        const ds4_v100_layer_state          *state,
        const ds4_v100_layer_execute_config *cfg,
        const ds4_gpu_tensor                *hidden_hc,
        ds4_gpu_tensor                      *next_hidden_hc,
        ds4_v100_layer_execute_report       *report,
        char                                *err,
        size_t                               errlen) {
    if (!hidden_hc || !next_hidden_hc) return exec_error(err, errlen, "missing HC tensor");
    if (validate_execute_common(state, cfg, err, errlen)) return 1;
    const uint32_t hidden_n = state->hidden_size;
    const uint64_t hc_values = (uint64_t)DS4_V100_N_HC * hidden_n;
    const uint64_t hc_bytes = hc_values * sizeof(float);
    if (hidden_n != DS4_V100_OUT_GROUP_DIM ||
        ds4_gpu_tensor_bytes(hidden_hc) < hc_bytes ||
        ds4_gpu_tensor_bytes(next_hidden_hc) < hc_bytes) {
        return exec_error(err, errlen, "HC tensor is too small");
    }
    if (state->hc_attn_fn.n_shape_dims != 2 ||
        state->hc_ffn_fn.n_shape_dims != 2 ||
        state->hc_attn_fn.shape[0] != hc_values ||
        state->hc_ffn_fn.shape[0] != hc_values ||
        state->hc_attn_fn.shape[1] != DS4_V100_HC_MIX ||
        state->hc_ffn_fn.shape[1] != DS4_V100_HC_MIX) {
        return exec_error(err, errlen, "HC control dimensions do not match DS4");
    }

    ds4_gpu_tensor *hc_norm = ds4_gpu_tensor_alloc(hc_bytes);
    ds4_gpu_tensor *hc_mix = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
    ds4_gpu_tensor *attn_split = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
    ds4_gpu_tensor *ffn_split = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
    ds4_gpu_tensor *attn_cur = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
    ds4_gpu_tensor *attn_out = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
    ds4_gpu_tensor *after_attn_hc = ds4_gpu_tensor_alloc(hc_bytes);
    ds4_gpu_tensor *ffn_cur = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
    ds4_gpu_tensor *ffn_norm = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
    ds4_gpu_tensor *ffn_delta = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));

    int rc = 1;
    if (!hc_norm || !hc_mix || !attn_split || !ffn_split || !attn_cur ||
        !attn_out || !after_attn_hc || !ffn_cur || !ffn_norm || !ffn_delta) {
        exec_error(err, errlen, "failed to allocate HC layer executor tensors");
        goto done;
    }

    if (!ds4_gpu_rms_norm_plain_tensor(hc_norm, hidden_hc, (uint32_t)hc_values, DS4_V100_RMS_EPS) ||
        !ds4_gpu_matmul_f32_tensor(hc_mix,
                                   cfg->model_map,
                                   cfg->model_size,
                                   state->hc_attn_fn.source_offset,
                                   hc_values,
                                   DS4_V100_HC_MIX,
                                   hc_norm,
                                   1) ||
        !ds4_gpu_hc_split_weighted_sum_tensor(attn_cur,
                                              attn_split,
                                              hc_mix,
                                              hidden_hc,
                                              cfg->model_map,
                                              cfg->model_size,
                                              state->hc_attn_scale.source_offset,
                                              state->hc_attn_base.source_offset,
                                              hidden_n,
                                              DS4_V100_N_HC,
                                              DS4_V100_HC_SINKHORN_ITERS,
                                              DS4_V100_RMS_EPS) ||
        execute_attention_output(state, cfg, attn_cur, attn_out, err, errlen) ||
        !ds4_gpu_hc_expand_split_tensor(after_attn_hc,
                                        attn_out,
                                        hidden_hc,
                                        attn_split,
                                        hidden_n,
                                        DS4_V100_N_HC) ||
        !ds4_gpu_rms_norm_plain_tensor(hc_norm, after_attn_hc, (uint32_t)hc_values, DS4_V100_RMS_EPS) ||
        !ds4_gpu_matmul_f32_tensor(hc_mix,
                                   cfg->model_map,
                                   cfg->model_size,
                                   state->hc_ffn_fn.source_offset,
                                   hc_values,
                                   DS4_V100_HC_MIX,
                                   hc_norm,
                                   1) ||
        !ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur,
                                              ffn_split,
                                              hc_mix,
                                              after_attn_hc,
                                              cfg->model_map,
                                              cfg->model_size,
                                              state->hc_ffn_scale.source_offset,
                                              state->hc_ffn_base.source_offset,
                                              hidden_n,
                                              DS4_V100_N_HC,
                                              DS4_V100_HC_SINKHORN_ITERS,
                                              DS4_V100_RMS_EPS) ||
        !ds4_gpu_rms_norm_weight_tensor(ffn_norm,
                                        ffn_cur,
                                        cfg->model_map,
                                        cfg->model_size,
                                        state->ffn_norm.source_offset,
                                        hidden_n,
                                        DS4_V100_RMS_EPS) ||
        execute_ffn_delta(state, cfg, ffn_norm, ffn_delta, report, err, errlen) ||
        !ds4_gpu_hc_expand_split_tensor(next_hidden_hc,
                                        ffn_delta,
                                        after_attn_hc,
                                        ffn_split,
                                        hidden_n,
                                        DS4_V100_N_HC)) {
        if (err && err[0] == '\0') exec_error(err, errlen, "HC layer execution sequence failed");
        goto done;
    }

    if (cfg->checkpoint_fn) {
        ds4_v100_layer_execute_checkpoint cp = {
            .layer = cfg->checkpoint_layer,
            .kind = DS4_V100_HC_CHECKPOINT_AFTER_ATTN,
            .hc = after_attn_hc,
            .hc_bytes = hc_bytes,
        };
        if (cfg->checkpoint_fn(&cp, cfg->checkpoint_user, err, errlen)) goto done;
    }

    rc = 0;

done:
    ds4_gpu_tensor_free(ffn_delta);
    ds4_gpu_tensor_free(ffn_norm);
    ds4_gpu_tensor_free(ffn_cur);
    ds4_gpu_tensor_free(after_attn_hc);
    ds4_gpu_tensor_free(attn_out);
    ds4_gpu_tensor_free(attn_cur);
    ds4_gpu_tensor_free(ffn_split);
    ds4_gpu_tensor_free(attn_split);
    ds4_gpu_tensor_free(hc_mix);
    ds4_gpu_tensor_free(hc_norm);
    return rc;
}
