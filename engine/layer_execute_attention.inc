static int validate_decode_cache(const ds4_layer_state *state,
                                 const ds4_layer_decode_cache *cache,
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

static int validate_execute_common(const ds4_layer_state *state,
                                   const ds4_layer_execute_config *cfg,
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
        const ds4_layer_state          *state,
        const ds4_layer_execute_config *cfg,
        const ds4_gpu_tensor                *attn_norm,
        const ds4_gpu_tensor                *q_a_norm,
        ds4_gpu_tensor                      *kv,
        ds4_attention_inputs           *inputs,
        char                                *err,
        size_t                               errlen) {
    ds4_layer_decode_cache *cache = cfg->decode_cache;
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
    if (ds4_bound_matrix_bf16_view(&state->attn_compressor_kv,
                                        &comp_kv_v,
                                        err,
                                        errlen) ||
        ds4_bound_matrix_bf16_view(&state->attn_compressor_gate,
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
                                          model_offset_for_matrix(cfg, &state->attn_compressor_ape),
                                          0,
                                          model_offset_for_binding(cfg, &state->attn_compressor_norm),
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
        if (ds4_bound_matrix_bf16_view(&state->indexer_compressor_kv,
                                            &index_kv_v,
                                            err,
                                            errlen) ||
            ds4_bound_matrix_bf16_view(&state->indexer_compressor_gate,
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
                                              model_offset_for_matrix(cfg, &state->indexer_compressor_ape),
                                              0,
                                              model_offset_for_binding(cfg, &state->indexer_compressor_norm),
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
                ds4_bound_matrix_bf16_view(&state->indexer_proj, &index_w_v, err, errlen)) {
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

static int execute_attention_output(const ds4_layer_state *state,
                                    const ds4_layer_execute_config *cfg,
                                    const ds4_gpu_tensor *hidden,
                                    ds4_gpu_tensor *attn_out,
                                    ds4_layer_execute_report *report,
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

    int rc = 1;
    ds4_gpu_tensor *attn_norm = NULL;
    ds4_gpu_tensor *q_a = NULL;
    ds4_gpu_tensor *q_a_norm = NULL;
    ds4_gpu_tensor *q = NULL;
    ds4_gpu_tensor *kv_raw = NULL;
    ds4_gpu_tensor *kv = NULL;
    ds4_gpu_tensor *heads = NULL;
    ds4_gpu_tensor *low = NULL;
    const bool use_scratch =
        cfg->batch_scratch != NULL && single_slot_attention_scratch_enabled();
    if (use_scratch &&
        ensure_batch_scratch(cfg->batch_scratch,
                             hidden_n,
                             state->intermediate_size,
                             state->routes_per_token,
                             q_rank,
                             q_width,
                             kv_width,
                             out_rank,
                             err,
                             errlen)) {
        goto done;
    }

    attn_norm = use_scratch ? cfg->batch_scratch->attn_norm_view[0]
        : ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
    q_a = use_scratch ? cfg->batch_scratch->attn_q_a_view[0]
        : ds4_gpu_tensor_alloc((uint64_t)q_rank * sizeof(float));
    q_a_norm = use_scratch ? cfg->batch_scratch->attn_q_a_norm_view[0]
        : ds4_gpu_tensor_alloc((uint64_t)q_rank * sizeof(float));
    q = use_scratch ? cfg->batch_scratch->attn_q_view[0]
        : ds4_gpu_tensor_alloc((uint64_t)q_width * sizeof(float));
    kv_raw = use_scratch ? cfg->batch_scratch->attn_kv_raw_view[0]
        : ds4_gpu_tensor_alloc((uint64_t)kv_width * sizeof(float));
    kv = use_scratch ? cfg->batch_scratch->attn_kv_view[0]
        : ds4_gpu_tensor_alloc((uint64_t)kv_width * sizeof(float));
    heads = use_scratch ? cfg->batch_scratch->attn_heads_view[0]
        : ds4_gpu_tensor_alloc((uint64_t)q_width * sizeof(float));
    low = use_scratch ? cfg->batch_scratch->attn_low_view[0]
        : ds4_gpu_tensor_alloc((uint64_t)out_rank * sizeof(float));

    if (!attn_norm || !q_a || !q_a_norm || !q || !kv_raw || !kv || !heads || !low) {
        exec_error(err, errlen, "failed to allocate attention body tensors");
        goto done;
    }

    const bool detail_profile =
        report &&
        env_flag_enabled("DS4_V100_PROFILE_DECODE") &&
        env_flag_enabled("DS4_V100_PROFILE_ATTENTION_DETAIL");
    double detail_last_ms = 0.0;
    double timing_attn_proj_ms = 0.0;
    double timing_attn_cache_ms = 0.0;
    double timing_attn_softmax_ms = 0.0;
    double timing_attn_inverse_rope_ms = 0.0;
    double timing_attn_output_ms = 0.0;
    if (detail_profile) {
        if (!ds4_gpu_synchronize()) {
            exec_error(err, errlen, "attention detail profile start sync failed");
            goto done;
        }
        detail_last_ms = monotonic_ms();
    }

    const bool fused_q_norm_rope = env_flag_enabled("DS4_V100_FUSED_Q_NORM_ROPE");
    if (!ds4_gpu_rms_norm_weight_tensor(attn_norm,
                                        hidden,
                                        cfg->model_map,
                                        cfg->model_size,
                                        model_offset_for_binding(cfg, &state->attn_norm),
                                        hidden_n,
                                        DS4_V100_RMS_EPS)) {
        exec_error(err, errlen, "attention projection sequence failed");
        goto done;
    }
    if (ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena, &q_a_v, attn_norm, q_a) != 0) {
        exec_error(err, errlen, "attention projection sequence failed");
        goto done;
    }
    if (!ds4_gpu_rms_norm_weight_tensor(q_a_norm,
                                        q_a,
                                        cfg->model_map,
                                        cfg->model_size,
                                        model_offset_for_binding(cfg, &state->attn_q_a_norm),
                                        q_rank,
                                        DS4_V100_RMS_EPS)) {
        exec_error(err, errlen, "attention projection sequence failed");
        goto done;
    }
    if (ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena, &q_b_v, q_a_norm, q) != 0) {
        exec_error(err, errlen, "attention projection sequence failed");
        goto done;
    }
    if (!(fused_q_norm_rope
              ? q_head_norm_rope_tail_layer_tensor(q,
                                                   DS4_V100_N_HEAD,
                                                   DS4_V100_HEAD_DIM,
                                                   cfg->position,
                                                   state)
              : (ds4_gpu_head_rms_norm_tensor(q,
                                              1,
                                              DS4_V100_N_HEAD,
                                              DS4_V100_HEAD_DIM,
                                              DS4_V100_RMS_EPS) &&
                 rope_tail_layer_tensor(q,
                                        DS4_V100_N_HEAD,
                                        DS4_V100_HEAD_DIM,
                                        cfg->position,
                                        state,
                                        false)))) {
        exec_error(err, errlen, "attention projection sequence failed");
        goto done;
    }
    if (ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena, &kv_v, attn_norm, kv_raw) != 0) {
        exec_error(err, errlen, "attention projection sequence failed");
        goto done;
    }
    if (!ds4_gpu_rms_norm_weight_tensor(kv,
                                        kv_raw,
                                        cfg->model_map,
                                        cfg->model_size,
                                        model_offset_for_binding(cfg, &state->attn_kv_a_norm),
                                        kv_width,
                                        DS4_V100_RMS_EPS)) {
        exec_error(err, errlen, "attention projection sequence failed");
        goto done;
    }
    if (!rope_tail_layer_tensor(kv,
                                1,
                                DS4_V100_HEAD_DIM,
                                cfg->position,
                                state,
                                false)) {
        exec_error(err, errlen, "attention projection sequence failed");
        goto done;
    }
    if (detail_profile && !profile_mark(&detail_last_ms, &timing_attn_proj_ms)) {
        exec_error(err, errlen, "attention projection detail profile sync failed");
        goto done;
    }

    ds4_attention_inputs attn_inputs = {
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
    if (detail_profile && !profile_mark(&detail_last_ms, &timing_attn_cache_ms)) {
        exec_error(err, errlen, "attention cache detail profile sync failed");
        goto done;
    }

    int attention_ok = 0;
    if (attn_inputs.use_indexed_attention) {
        attention_ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                heads,
                cfg->model_map,
                cfg->model_size,
                model_offset_for_binding(cfg, &state->attn_sinks),
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
                                                             model_offset_for_binding(cfg, &state->attn_sinks),
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
    if (detail_profile && !profile_mark(&detail_last_ms, &timing_attn_softmax_ms)) {
        exec_error(err, errlen, "attention softmax detail profile sync failed");
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
    if (detail_profile && !profile_mark(&detail_last_ms, &timing_attn_inverse_rope_ms)) {
        exec_error(err, errlen, "attention inverse rope detail profile sync failed");
        goto done;
    }

    if (grouped_attention_output(state, cfg, heads, low, attn_out, err, errlen)) {
        goto done;
    }
    if (detail_profile && !profile_mark(&detail_last_ms, &timing_attn_output_ms)) {
        exec_error(err, errlen, "attention output detail profile sync failed");
        goto done;
    }
    if (detail_profile) {
        report->timing_attn_proj_ms = timing_attn_proj_ms;
        report->timing_attn_cache_ms = timing_attn_cache_ms;
        report->timing_attn_softmax_ms = timing_attn_softmax_ms;
        report->timing_attn_inverse_rope_ms = timing_attn_inverse_rope_ms;
        report->timing_attn_output_ms = timing_attn_output_ms;
    }
    rc = 0;

done:
    if (!use_scratch) {
        ds4_gpu_tensor_free(low);
        ds4_gpu_tensor_free(heads);
        ds4_gpu_tensor_free(kv);
        ds4_gpu_tensor_free(kv_raw);
        ds4_gpu_tensor_free(q);
        ds4_gpu_tensor_free(q_a_norm);
        ds4_gpu_tensor_free(q_a);
        ds4_gpu_tensor_free(attn_norm);
    }
    return rc;
}

static int execute_attention_output_batch(const ds4_layer_state *state,
                                          const ds4_layer_execute_config *cfgs,
                                          const ds4_gpu_tensor *const *hidden,
                                          ds4_gpu_tensor *const *attn_out,
                                          uint32_t n_slots,
                                          char *err,
                                          size_t errlen) {
    if (!state || !cfgs || !hidden || !attn_out ||
        n_slots == 0 || n_slots > DS4_V100_LAYER_MAX_BATCH) {
        return exec_error(err, errlen, "invalid attention batch inputs");
    }
    const bool use_attention_projection_batch =
        env_flag_enabled("DS4_V100_ENABLE_BATCH_ATTN_PROJ") &&
        (n_slots == 4u ||
         n_slots == 8u ||
         n_slots == 16u ||
         env_flag_enabled("DS4_V100_ENABLE_BATCH_ATTN_PROJ_ANY"));
    if (n_slots == 1 || !use_attention_projection_batch) {
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            if (execute_attention_output(state,
                                         &cfgs[slot],
                                         hidden[slot],
                                         attn_out[slot],
                                         NULL,
                                         err,
                                         errlen)) {
                return 1;
            }
        }
        return 0;
    }

    const uint32_t hidden_n = state->hidden_size;
    const uint32_t q_rank = state->q_lora_rank;
    const uint32_t q_width = state->q_width;
    const uint32_t kv_width = state->kv_latent_width;
    const uint32_t out_rank = state->attention_output_rank;
    if (hidden_n != DS4_V100_OUT_GROUP_DIM ||
        q_width != DS4_V100_N_HEAD * DS4_V100_HEAD_DIM ||
        out_rank != DS4_V100_OUT_GROUPS * DS4_V100_OUT_GROUP_RANK) {
        return exec_error(err, errlen, "attention batch dimensions do not match DS4");
    }
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!hidden[slot] || !attn_out[slot] ||
            ds4_gpu_tensor_bytes(hidden[slot]) < (uint64_t)hidden_n * sizeof(float) ||
            ds4_gpu_tensor_bytes(attn_out[slot]) < (uint64_t)hidden_n * sizeof(float) ||
            cfgs[slot].arena != cfgs[0].arena ||
            cfgs[slot].model_map != cfgs[0].model_map ||
            cfgs[slot].model_size != cfgs[0].model_size ||
            cfgs[slot].model_map_uses_shard_offsets != cfgs[0].model_map_uses_shard_offsets ||
            cfgs[slot].batch_scratch != cfgs[0].batch_scratch) {
            return exec_error(err, errlen, "attention batch cfgs must share arena/model");
        }
    }

    ds4_gpu_source_row_view q_a_v;
    ds4_gpu_source_row_view q_b_v;
    ds4_gpu_source_row_view kv_v;
    if (source_view(&state->attn_q_a, &q_a_v, err, errlen) ||
        source_view(&state->attn_q_b, &q_b_v, err, errlen) ||
        source_view(&state->attn_kv_latent, &kv_v, err, errlen)) {
        return 1;
    }

    const bool use_scratch = cfgs[0].batch_scratch != NULL;
    const bool batch_output_a =
        use_scratch &&
        n_slots == 16u &&
        env_flag_enabled("DS4_V100_BATCH_ATTN_OUTPUT_A");
    const bool batch_output_b =
        use_scratch &&
        n_slots == 16u &&
        env_flag_enabled("DS4_V100_BATCH_ATTN_OUTPUT_B");
    const bool fused_q_norm_rope = env_flag_enabled("DS4_V100_FUSED_Q_NORM_ROPE");
    if (use_scratch && ensure_batch_scratch(cfgs[0].batch_scratch,
                                            hidden_n,
                                            state->intermediate_size,
                                            state->routes_per_token,
                                            q_rank,
                                            q_width,
                                            kv_width,
                                            out_rank,
                                            err,
                                            errlen)) {
        return 1;
    }
    ds4_gpu_tensor *input_ptrs_t = use_scratch ? cfgs[0].batch_scratch->attn_input_ptrs
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * sizeof(void *));
    ds4_gpu_tensor *attn_norm_batch = use_scratch ? cfgs[0].batch_scratch->attn_norm_batch
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * hidden_n * sizeof(float));
    ds4_gpu_tensor *q_a_batch = use_scratch ? cfgs[0].batch_scratch->attn_q_a_batch
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * q_rank * sizeof(float));
    ds4_gpu_tensor *q_a_norm_batch = use_scratch ? cfgs[0].batch_scratch->attn_q_a_norm_batch
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * q_rank * sizeof(float));
    ds4_gpu_tensor *q_batch = use_scratch ? cfgs[0].batch_scratch->attn_q_batch
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * q_width * sizeof(float));
    ds4_gpu_tensor *kv_raw_batch = use_scratch ? cfgs[0].batch_scratch->attn_kv_raw_batch
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * kv_width * sizeof(float));
    ds4_gpu_tensor *kv_batch = use_scratch ? cfgs[0].batch_scratch->attn_kv_batch
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * kv_width * sizeof(float));
    ds4_gpu_tensor *heads_batch = use_scratch ? cfgs[0].batch_scratch->attn_heads_batch
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * q_width * sizeof(float));
    ds4_gpu_tensor *low_batch = use_scratch ? cfgs[0].batch_scratch->attn_low_batch
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * out_rank * sizeof(float));
    ds4_gpu_tensor *attn_norm_views[DS4_V100_LAYER_MAX_BATCH] = {0};

    int rc = 1;
    if (!input_ptrs_t || !attn_norm_batch || !q_a_batch || !q_a_norm_batch ||
        !q_batch || !kv_raw_batch || !kv_batch || !heads_batch || !low_batch) {
        exec_error(err, errlen, "failed to allocate attention batch tensors");
        goto done;
    }
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        attn_norm_views[slot] = use_scratch
            ? cfgs[0].batch_scratch->attn_norm_view[slot]
            : ds4_gpu_tensor_view(attn_norm_batch,
                                  (uint64_t)slot * hidden_n * sizeof(float),
                                  (uint64_t)hidden_n * sizeof(float));
        if (!attn_norm_views[slot]) {
            exec_error(err, errlen, "failed to create attention norm batch view");
            goto done;
        }
        if (!ds4_gpu_rms_norm_weight_tensor(attn_norm_views[slot],
                                            hidden[slot],
                                            cfgs[slot].model_map,
                                            cfgs[slot].model_size,
                                            model_offset_for_binding(&cfgs[slot], &state->attn_norm),
                                            hidden_n,
                                            DS4_V100_RMS_EPS)) {
            exec_error(err, errlen, "attention batch input norm failed");
            goto done;
        }
    }
    if (!ds4_gpu_tensor_write_f32_row_ptrs(input_ptrs_t,
                                           (const ds4_gpu_tensor *const *)attn_norm_views,
                                           n_slots,
                                           (uint64_t)hidden_n * sizeof(float)) ||
        ds4_gpu_arena_f8_e4m3_b128_matmul_batch_ptr_table_f32(cfgs[0].arena,
                                                              &q_a_v,
                                                              input_ptrs_t,
                                                              n_slots,
                                                              q_a_batch) != 0 ||
        ds4_gpu_arena_f8_e4m3_b128_matmul_batch_ptr_table_f32(cfgs[0].arena,
                                                              &kv_v,
                                                              input_ptrs_t,
                                                              n_slots,
                                                              kv_raw_batch) != 0 ||
        !ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(q_a_norm_batch,
                                               q_a_batch,
                                               cfgs[0].model_map,
                                               cfgs[0].model_size,
                                               model_offset_for_binding(&cfgs[0], &state->attn_q_a_norm),
                                               q_rank,
                                               kv_batch,
                                               kv_raw_batch,
                                               model_offset_for_binding(&cfgs[0], &state->attn_kv_a_norm),
                                               kv_width,
                                               n_slots,
                                               DS4_V100_RMS_EPS) ||
        ds4_gpu_arena_f8_e4m3_b128_matmul_batch_f32(cfgs[0].arena,
                                                    &q_b_v,
                                                    q_a_norm_batch,
                                                    n_slots,
                                                    q_batch) != 0) {
        exec_error(err, errlen, "attention batch projection sequence failed");
        goto done;
    }

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        ds4_gpu_tensor *q_view = use_scratch ? cfgs[0].batch_scratch->attn_q_view[slot]
            : ds4_gpu_tensor_view(q_batch,
                                  (uint64_t)slot * q_width * sizeof(float),
                                  (uint64_t)q_width * sizeof(float));
        ds4_gpu_tensor *q_a_norm_view = use_scratch ? cfgs[0].batch_scratch->attn_q_a_norm_view[slot]
            : ds4_gpu_tensor_view(q_a_norm_batch,
                                  (uint64_t)slot * q_rank * sizeof(float),
                                  (uint64_t)q_rank * sizeof(float));
        ds4_gpu_tensor *kv_view = use_scratch ? cfgs[0].batch_scratch->attn_kv_view[slot]
            : ds4_gpu_tensor_view(kv_batch,
                                  (uint64_t)slot * kv_width * sizeof(float),
                                  (uint64_t)kv_width * sizeof(float));
        ds4_gpu_tensor *heads_view = use_scratch ? cfgs[0].batch_scratch->attn_heads_view[slot]
            : ds4_gpu_tensor_view(heads_batch,
                                  (uint64_t)slot * q_width * sizeof(float),
                                  (uint64_t)q_width * sizeof(float));
        ds4_gpu_tensor *low_view = use_scratch ? cfgs[0].batch_scratch->attn_low_view[slot]
            : ds4_gpu_tensor_view(low_batch,
                                  (uint64_t)slot * out_rank * sizeof(float),
                                  (uint64_t)out_rank * sizeof(float));
        if (!q_view || !q_a_norm_view || !kv_view || !heads_view || !low_view) {
            if (!use_scratch) {
                ds4_gpu_tensor_free(low_view);
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(kv_view);
                ds4_gpu_tensor_free(q_a_norm_view);
                ds4_gpu_tensor_free(q_view);
            }
            exec_error(err, errlen, "failed to create attention batch tensor views");
            goto done;
        }

        const uint32_t n_raw = cfgs[slot].n_raw ? cfgs[slot].n_raw : 1u;
        const uint32_t raw_cap = cfgs[slot].raw_cap ? cfgs[slot].raw_cap : n_raw;
        ds4_attention_inputs attn_inputs = {
            .raw_kv = cfgs[slot].raw_kv ? cfgs[slot].raw_kv : kv_view,
            .n_raw = n_raw,
            .raw_cap = raw_cap,
            .raw_start = cfgs[slot].raw_start,
            .compressed_kv = cfgs[slot].compressed_kv,
            .n_compressed = cfgs[slot].n_compressed,
            .compressed_mask = cfgs[slot].compressed_mask,
            .use_compressed_mask = cfgs[slot].use_compressed_mask ? 1u : 0u,
            .indexed_topk = NULL,
            .indexed_top_k = 0,
            .use_indexed_attention = false,
        };
        int attention_ok = 0;
        if (!(fused_q_norm_rope
                ? q_head_norm_rope_tail_layer_tensor(q_view,
                                                     DS4_V100_N_HEAD,
                                                     DS4_V100_HEAD_DIM,
                                                     cfgs[slot].position,
                                                     state)
                : (ds4_gpu_head_rms_norm_tensor(q_view,
                                                1,
                                                DS4_V100_N_HEAD,
                                                DS4_V100_HEAD_DIM,
                                                DS4_V100_RMS_EPS) &&
                   rope_tail_layer_tensor(q_view,
                                          DS4_V100_N_HEAD,
                                          DS4_V100_HEAD_DIM,
                                          cfgs[slot].position,
                                          state,
                                          false))) ||
            !rope_tail_layer_tensor(kv_view,
                                    1,
                                    DS4_V100_HEAD_DIM,
                                    cfgs[slot].position,
                                    state,
                                    false) ||
            prepare_decode_cache_attention(state,
                                           &cfgs[slot],
                                           attn_norm_views[slot],
                                           q_a_norm_view,
                                           kv_view,
                                           &attn_inputs,
                                           err,
                                           errlen)) {
            if (!use_scratch) {
                ds4_gpu_tensor_free(low_view);
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(kv_view);
                ds4_gpu_tensor_free(q_a_norm_view);
                ds4_gpu_tensor_free(q_view);
            }
            goto done;
        }
        if (attn_inputs.use_indexed_attention) {
            attention_ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                    heads_view,
                    cfgs[slot].model_map,
                    cfgs[slot].model_size,
                    model_offset_for_binding(&cfgs[slot], &state->attn_sinks),
                    q_view,
                    attn_inputs.raw_kv,
                    attn_inputs.compressed_kv,
                    attn_inputs.indexed_topk,
                    1,
                    cfgs[slot].position,
                    attn_inputs.n_raw,
                    attn_inputs.raw_cap,
                    attn_inputs.raw_start,
                    attn_inputs.n_compressed,
                    attn_inputs.indexed_top_k,
                    cfgs[slot].decode_cache && cfgs[slot].decode_cache->raw_window ? cfgs[slot].decode_cache->raw_window : 128u,
                    state->compress_ratio,
                    DS4_V100_N_HEAD,
                    DS4_V100_HEAD_DIM);
        } else {
            attention_ok = ds4_gpu_attention_decode_heads_tensor(heads_view,
                                                                 cfgs[slot].model_map,
                                                                 cfgs[slot].model_size,
                                                                 model_offset_for_binding(&cfgs[slot], &state->attn_sinks),
                                                                 q_view,
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
        if (!attention_ok ||
            !rope_tail_layer_tensor(heads_view,
                                    DS4_V100_N_HEAD,
                                    DS4_V100_HEAD_DIM,
                                    cfgs[slot].position,
                                    state,
                                    true) ||
            (!batch_output_a &&
             (batch_output_b
                ? grouped_attention_output_a(state,
                                             &cfgs[slot],
                                             heads_view,
                                             low_view,
                                             err,
                                             errlen)
                : grouped_attention_output(state,
                                           &cfgs[slot],
                                           heads_view,
                                           low_view,
                                           attn_out[slot],
                                           err,
                                           errlen)))) {
            if (err && err[0] == '\0') exec_error(err, errlen, "attention batch slot failed");
            if (!use_scratch) {
                ds4_gpu_tensor_free(low_view);
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(kv_view);
                ds4_gpu_tensor_free(q_a_norm_view);
                ds4_gpu_tensor_free(q_view);
            }
            goto done;
        }
        if (!use_scratch) {
            ds4_gpu_tensor_free(low_view);
            ds4_gpu_tensor_free(heads_view);
            ds4_gpu_tensor_free(kv_view);
            ds4_gpu_tensor_free(q_a_norm_view);
            ds4_gpu_tensor_free(q_view);
        }
    }

    if (batch_output_a) {
        ds4_gpu_source_row_view a_view;
        if (source_view(&state->attn_output_a, &a_view, err, errlen) ||
            ds4_gpu_arena_f8_e4m3_b128_matmul_grouped_batch_f32(
                    cfgs[0].arena,
                    &a_view,
                    heads_batch,
                    n_slots,
                    DS4_V100_OUT_GROUPS,
                    DS4_V100_OUT_GROUP_RANK,
                    DS4_V100_OUT_GROUP_DIM,
                    low_batch) != 0) {
            exec_error(err, errlen, "attention output_a grouped batch matmul failed");
            goto done;
        }
    }

    if (batch_output_b) {
        ds4_gpu_source_row_view b_view;
        if (source_view(&state->attn_output_b, &b_view, err, errlen) ||
            ds4_gpu_arena_f8_e4m3_b128_matmul_batch_f32(cfgs[0].arena,
                                                        &b_view,
                                                        low_batch,
                                                        n_slots,
                                                        cfgs[0].batch_scratch->attn_out_batch) != 0) {
            exec_error(err, errlen, "attention output_b batch matmul failed");
            goto done;
        }
    } else if (batch_output_a) {
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            if (grouped_attention_output_b(state,
                                           &cfgs[slot],
                                           cfgs[0].batch_scratch->attn_low_view[slot],
                                           attn_out[slot],
                                           err,
                                           errlen)) {
                goto done;
            }
        }
    }

    rc = 0;

done:
    if (!use_scratch) {
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            ds4_gpu_tensor_free(attn_norm_views[slot]);
        }
        ds4_gpu_tensor_free(low_batch);
        ds4_gpu_tensor_free(heads_batch);
        ds4_gpu_tensor_free(kv_batch);
        ds4_gpu_tensor_free(kv_raw_batch);
        ds4_gpu_tensor_free(q_batch);
        ds4_gpu_tensor_free(q_a_norm_batch);
        ds4_gpu_tensor_free(q_a_batch);
        ds4_gpu_tensor_free(attn_norm_batch);
        ds4_gpu_tensor_free(input_ptrs_t);
    }
    return rc;
}
