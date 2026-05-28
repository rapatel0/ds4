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

static uint64_t model_offset_for_binding(const ds4_v100_layer_execute_config *cfg,
                                         const ds4_v100_tensor_binding *b) {
    if (!b) return 0;
    return (cfg && cfg->model_map_uses_shard_offsets) ? b->shard_offset : b->source_offset;
}

static uint64_t model_offset_for_matrix(const ds4_v100_layer_execute_config *cfg,
                                        const ds4_v100_bound_matrix *m) {
    return m ? model_offset_for_binding(cfg, &m->binding) : 0;
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

static void free_tensor_array(ds4_gpu_tensor **tensors, uint32_t n) {
    if (!tensors) return;
    for (uint32_t i = 0; i < n; i++) {
        ds4_gpu_tensor_free(tensors[i]);
        tensors[i] = NULL;
    }
}

void ds4_v100_layer_batch_scratch_init(ds4_v100_layer_batch_scratch *scratch) {
    if (!scratch) return;
    memset(scratch, 0, sizeof(*scratch));
}

void ds4_v100_layer_batch_scratch_free(ds4_v100_layer_batch_scratch *scratch) {
    if (!scratch) return;
    free_tensor_array(scratch->hc_norm, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->hc_mix, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->attn_split, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->ffn_split, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->attn_cur, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->attn_out, DS4_V100_LAYER_MAX_BATCH);
    ds4_gpu_tensor_free(scratch->attn_out_batch);
    free_tensor_array(scratch->after_attn_hc, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->ffn_cur, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->ffn_norm, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->ffn_delta, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->ffn_shared_batch_view, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->ffn_routed_out_view, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->attn_low_view, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->attn_heads_view, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->attn_kv_view, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->attn_kv_raw_view, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->attn_q_view, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->attn_q_a_norm_view, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->attn_q_a_view, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->attn_norm_view, DS4_V100_LAYER_MAX_BATCH);
    ds4_gpu_tensor_free(scratch->attn_low_batch);
    ds4_gpu_tensor_free(scratch->attn_heads_batch);
    ds4_gpu_tensor_free(scratch->attn_kv_batch);
    ds4_gpu_tensor_free(scratch->attn_kv_raw_batch);
    ds4_gpu_tensor_free(scratch->attn_q_batch);
    ds4_gpu_tensor_free(scratch->attn_q_a_norm_batch);
    ds4_gpu_tensor_free(scratch->attn_q_a_batch);
    ds4_gpu_tensor_free(scratch->attn_norm_batch);
    ds4_gpu_tensor_free(scratch->attn_input_ptrs);
    ds4_gpu_tensor_free(scratch->ffn_shared);
    ds4_gpu_tensor_free(scratch->ffn_shared_batch);
    ds4_gpu_tensor_free(scratch->ffn_shared_mid_batch);
    ds4_gpu_tensor_free(scratch->ffn_shared_mid);
    ds4_gpu_tensor_free(scratch->ffn_shared_up);
    ds4_gpu_tensor_free(scratch->ffn_shared_gate);
    ds4_gpu_tensor_free(scratch->ffn_routed_out);
    ds4_gpu_tensor_free(scratch->ffn_routed_mid);
    ds4_gpu_tensor_free(scratch->ffn_input_ptrs);
    ds4_gpu_tensor_free(scratch->ffn_tokens);
    ds4_gpu_tensor_free(scratch->ffn_weights);
    ds4_gpu_tensor_free(scratch->ffn_selected);
    ds4_gpu_tensor_free(scratch->ffn_probs);
    ds4_gpu_tensor_free(scratch->ffn_router);
    ds4_gpu_tensor_free(scratch->ffn_delta_batch);
    ds4_gpu_tensor_free(scratch->ffn_norm_batch);
    memset(scratch, 0, sizeof(*scratch));
}

static int alloc_tensor_slot(ds4_gpu_tensor **slot,
                             uint64_t bytes,
                             char *err,
                             size_t errlen,
                             const char *name) {
    if (!slot) return exec_error(err, errlen, "missing scratch slot");
    *slot = ds4_gpu_tensor_alloc(bytes);
    if (!*slot) return exec_error(err, errlen, "failed to allocate %s", name);
    return 0;
}

static int ensure_batch_scratch(ds4_v100_layer_batch_scratch *scratch,
                                uint32_t hidden,
                                uint32_t mid,
                                uint32_t routes,
                                uint32_t q_rank,
                                uint32_t q_width,
                                uint32_t kv_width,
                                uint32_t out_rank,
                                char *err,
                                size_t errlen) {
    if (!scratch) return 0;
    if (scratch->hidden == hidden &&
        scratch->intermediate == mid &&
        scratch->routes == routes &&
        scratch->q_rank == q_rank &&
        scratch->q_width == q_width &&
        scratch->kv_width == kv_width &&
        scratch->out_rank == out_rank &&
        scratch->max_slots == DS4_V100_LAYER_MAX_BATCH &&
        scratch->ffn_norm_batch &&
        scratch->ffn_delta_batch &&
        scratch->ffn_shared) {
        return 0;
    }

    ds4_v100_layer_batch_scratch_free(scratch);
    const uint64_t hc_bytes =
        (uint64_t)DS4_V100_N_HC * hidden * sizeof(float);
    const uint64_t hidden_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t mid_bytes = (uint64_t)mid * sizeof(float);
    const uint64_t q_rank_bytes = (uint64_t)q_rank * sizeof(float);
    const uint64_t q_width_bytes = (uint64_t)q_width * sizeof(float);
    const uint64_t kv_width_bytes = (uint64_t)kv_width * sizeof(float);
    const uint64_t out_rank_bytes = (uint64_t)out_rank * sizeof(float);

    for (uint32_t slot = 0; slot < DS4_V100_LAYER_MAX_BATCH; slot++) {
        if (alloc_tensor_slot(&scratch->hc_norm[slot], hc_bytes, err, errlen, "batch hc_norm") ||
            alloc_tensor_slot(&scratch->hc_mix[slot],
                              DS4_V100_HC_MIX * sizeof(float),
                              err,
                              errlen,
                              "batch hc_mix") ||
            alloc_tensor_slot(&scratch->attn_split[slot],
                              DS4_V100_HC_MIX * sizeof(float),
                              err,
                              errlen,
                              "batch attn_split") ||
            alloc_tensor_slot(&scratch->ffn_split[slot],
                              DS4_V100_HC_MIX * sizeof(float),
                              err,
                              errlen,
                              "batch ffn_split") ||
            alloc_tensor_slot(&scratch->attn_cur[slot],
                              hidden_bytes,
                              err,
                              errlen,
                              "batch attn_cur") ||
            alloc_tensor_slot(&scratch->after_attn_hc[slot],
                              hc_bytes,
                              err,
                              errlen,
                              "batch after_attn_hc") ||
            alloc_tensor_slot(&scratch->ffn_cur[slot],
                              hidden_bytes,
                              err,
                              errlen,
                              "batch ffn_cur")) {
            ds4_v100_layer_batch_scratch_free(scratch);
            return 1;
        }
    }

    const uint64_t max_slots = DS4_V100_LAYER_MAX_BATCH;
    if (alloc_tensor_slot(&scratch->attn_out_batch,
                          max_slots * hidden_bytes,
                          err,
                          errlen,
                          "batch attn_out_batch") ||
        alloc_tensor_slot(&scratch->ffn_norm_batch,
                          max_slots * hidden_bytes,
                          err,
                          errlen,
                          "batch ffn_norm_batch") ||
        alloc_tensor_slot(&scratch->ffn_delta_batch,
                          max_slots * hidden_bytes,
                          err,
                          errlen,
                          "batch ffn_delta_batch") ||
        alloc_tensor_slot(&scratch->ffn_router,
                          max_slots * 256u * sizeof(float),
                          err,
                          errlen,
                          "batch ffn_router") ||
        alloc_tensor_slot(&scratch->ffn_probs,
                          max_slots * 256u * sizeof(float),
                          err,
                          errlen,
                          "batch ffn_probs") ||
        alloc_tensor_slot(&scratch->ffn_selected,
                          max_slots * routes * sizeof(int32_t),
                          err,
                          errlen,
                          "batch ffn_selected") ||
        alloc_tensor_slot(&scratch->ffn_weights,
                          max_slots * routes * sizeof(float),
                          err,
                          errlen,
                          "batch ffn_weights") ||
        alloc_tensor_slot(&scratch->ffn_tokens,
                          max_slots * sizeof(int32_t),
                          err,
                          errlen,
                          "batch ffn_tokens") ||
        alloc_tensor_slot(&scratch->ffn_input_ptrs,
                          max_slots * sizeof(void *),
                          err,
                          errlen,
                          "batch ffn_input_ptrs") ||
        alloc_tensor_slot(&scratch->ffn_routed_mid,
                          max_slots * routes * mid_bytes,
                          err,
                          errlen,
                          "batch ffn_routed_mid") ||
        alloc_tensor_slot(&scratch->ffn_routed_out,
                          max_slots * hidden_bytes,
                          err,
                          errlen,
                          "batch ffn_routed_out") ||
        alloc_tensor_slot(&scratch->ffn_shared_gate,
                          mid_bytes,
                          err,
                          errlen,
                          "batch ffn_shared_gate") ||
        alloc_tensor_slot(&scratch->ffn_shared_up,
                          mid_bytes,
                          err,
                          errlen,
                          "batch ffn_shared_up") ||
        alloc_tensor_slot(&scratch->ffn_shared_mid,
                          mid_bytes,
                          err,
                          errlen,
                          "batch ffn_shared_mid") ||
        alloc_tensor_slot(&scratch->ffn_shared,
                          hidden_bytes,
                          err,
                          errlen,
                          "batch ffn_shared") ||
        alloc_tensor_slot(&scratch->ffn_shared_mid_batch,
                          max_slots * mid_bytes,
                          err,
                          errlen,
                          "batch ffn_shared_mid_batch") ||
        alloc_tensor_slot(&scratch->ffn_shared_batch,
                          max_slots * hidden_bytes,
                          err,
                          errlen,
                          "batch ffn_shared_batch") ||
        alloc_tensor_slot(&scratch->attn_input_ptrs,
                          max_slots * sizeof(void *),
                          err,
                          errlen,
                          "batch attn_input_ptrs") ||
        alloc_tensor_slot(&scratch->attn_norm_batch,
                          max_slots * hidden_bytes,
                          err,
                          errlen,
                          "batch attn_norm_batch") ||
        alloc_tensor_slot(&scratch->attn_q_a_batch,
                          max_slots * q_rank_bytes,
                          err,
                          errlen,
                          "batch attn_q_a_batch") ||
        alloc_tensor_slot(&scratch->attn_q_a_norm_batch,
                          max_slots * q_rank_bytes,
                          err,
                          errlen,
                          "batch attn_q_a_norm_batch") ||
        alloc_tensor_slot(&scratch->attn_q_batch,
                          max_slots * q_width_bytes,
                          err,
                          errlen,
                          "batch attn_q_batch") ||
        alloc_tensor_slot(&scratch->attn_kv_raw_batch,
                          max_slots * kv_width_bytes,
                          err,
                          errlen,
                          "batch attn_kv_raw_batch") ||
        alloc_tensor_slot(&scratch->attn_kv_batch,
                          max_slots * kv_width_bytes,
                          err,
                          errlen,
                          "batch attn_kv_batch") ||
        alloc_tensor_slot(&scratch->attn_heads_batch,
                          max_slots * q_width_bytes,
                          err,
                          errlen,
                          "batch attn_heads_batch") ||
        alloc_tensor_slot(&scratch->attn_low_batch,
                          max_slots * out_rank_bytes,
                          err,
                          errlen,
                          "batch attn_low_batch")) {
        ds4_v100_layer_batch_scratch_free(scratch);
        return 1;
    }

    scratch->hidden = hidden;
    scratch->intermediate = mid;
    scratch->routes = routes;
    scratch->q_rank = q_rank;
    scratch->q_width = q_width;
    scratch->kv_width = kv_width;
    scratch->out_rank = out_rank;
    scratch->max_slots = DS4_V100_LAYER_MAX_BATCH;
    for (uint32_t slot = 0; slot < DS4_V100_LAYER_MAX_BATCH; slot++) {
        scratch->ffn_norm[slot] =
            ds4_gpu_tensor_view(scratch->ffn_norm_batch,
                                (uint64_t)slot * hidden_bytes,
                                hidden_bytes);
        scratch->ffn_delta[slot] =
            ds4_gpu_tensor_view(scratch->ffn_delta_batch,
                                (uint64_t)slot * hidden_bytes,
                                hidden_bytes);
        scratch->attn_out[slot] =
            ds4_gpu_tensor_view(scratch->attn_out_batch,
                                (uint64_t)slot * hidden_bytes,
                                hidden_bytes);
        scratch->ffn_routed_out_view[slot] =
            ds4_gpu_tensor_view(scratch->ffn_routed_out,
                                (uint64_t)slot * hidden_bytes,
                                hidden_bytes);
        scratch->ffn_shared_batch_view[slot] =
            ds4_gpu_tensor_view(scratch->ffn_shared_batch,
                                (uint64_t)slot * hidden_bytes,
                                hidden_bytes);
        scratch->attn_norm_view[slot] =
            ds4_gpu_tensor_view(scratch->attn_norm_batch,
                                (uint64_t)slot * hidden_bytes,
                                hidden_bytes);
        scratch->attn_q_a_view[slot] =
            ds4_gpu_tensor_view(scratch->attn_q_a_batch,
                                (uint64_t)slot * q_rank_bytes,
                                q_rank_bytes);
        scratch->attn_q_a_norm_view[slot] =
            ds4_gpu_tensor_view(scratch->attn_q_a_norm_batch,
                                (uint64_t)slot * q_rank_bytes,
                                q_rank_bytes);
        scratch->attn_q_view[slot] =
            ds4_gpu_tensor_view(scratch->attn_q_batch,
                                (uint64_t)slot * q_width_bytes,
                                q_width_bytes);
        scratch->attn_kv_raw_view[slot] =
            ds4_gpu_tensor_view(scratch->attn_kv_raw_batch,
                                (uint64_t)slot * kv_width_bytes,
                                kv_width_bytes);
        scratch->attn_kv_view[slot] =
            ds4_gpu_tensor_view(scratch->attn_kv_batch,
                                (uint64_t)slot * kv_width_bytes,
                                kv_width_bytes);
        scratch->attn_heads_view[slot] =
            ds4_gpu_tensor_view(scratch->attn_heads_batch,
                                (uint64_t)slot * q_width_bytes,
                                q_width_bytes);
        scratch->attn_low_view[slot] =
            ds4_gpu_tensor_view(scratch->attn_low_batch,
                                (uint64_t)slot * out_rank_bytes,
                                out_rank_bytes);
        if (!scratch->ffn_norm[slot] ||
            !scratch->ffn_delta[slot] ||
            !scratch->attn_out[slot] ||
            !scratch->ffn_routed_out_view[slot] ||
            !scratch->ffn_shared_batch_view[slot] ||
            !scratch->attn_norm_view[slot] ||
            !scratch->attn_q_a_view[slot] ||
            !scratch->attn_q_a_norm_view[slot] ||
            !scratch->attn_q_view[slot] ||
            !scratch->attn_kv_raw_view[slot] ||
            !scratch->attn_kv_view[slot] ||
            !scratch->attn_heads_view[slot] ||
            !scratch->attn_low_view[slot]) {
            ds4_v100_layer_batch_scratch_free(scratch);
            return exec_error(err, errlen, "failed to allocate batch scratch views");
        }
    }
    return 0;
}

static bool env_flag_enabled(const char *name) {
    const char *v = getenv(name);
    if (!v || !*v) return false;
    return strcmp(v, "0") != 0 &&
           strcmp(v, "false") != 0 &&
           strcmp(v, "FALSE") != 0 &&
           strcmp(v, "off") != 0 &&
           strcmp(v, "OFF") != 0;
}

static bool env_flag_default_enabled(const char *name) {
    const char *v = getenv(name);
    if (!v || !*v) return true;
    return strcmp(v, "0") != 0 &&
           strcmp(v, "false") != 0 &&
           strcmp(v, "FALSE") != 0 &&
           strcmp(v, "off") != 0 &&
           strcmp(v, "OFF") != 0;
}

static bool single_slot_batch_scratch_enabled(void) {
    return env_flag_enabled("DS4_V100_SINGLE_SLOT_BATCH_SCRATCH") ||
           env_flag_enabled("DS4_V100_TURBOMIND_GRAPH");
}

static bool single_slot_attention_scratch_enabled(void) {
    if (single_slot_batch_scratch_enabled()) return true;
    const char *v = getenv("DS4_V100_SINGLE_SLOT_ATTN_SCRATCH");
    if (v && *v) return env_flag_enabled("DS4_V100_SINGLE_SLOT_ATTN_SCRATCH");
    return true;
}

static double monotonic_ms(void);
static int profile_mark(double *last_ms, double *bucket_ms);

static bool tp2_async_input_enabled(void) {
    return env_flag_enabled("DS4_V100_TP2_ASYNC_INPUT") ||
           env_flag_enabled("DS4_V100_TP_ROUTED_FFN_ASYNC_INPUT") ||
           env_flag_enabled("DS4_V100_TP_EP_ASYNC_INPUT");
}

static bool tp2_verbose_enabled(void) {
    return env_flag_enabled("DS4_V100_TP2_VERBOSE") ||
           env_flag_enabled("DS4_V100_TP_EP_VERBOSE");
}

static bool tp2_parallel_halves_enabled(void) {
    return env_flag_enabled("DS4_V100_TP2_PARALLEL_HALVES") ||
           env_flag_enabled("DS4_V100_TP_EP_PARALLEL_HALVES");
}

static bool tp2_routed_enabled(const ds4_v100_layer_state *state,
                               const ds4_v100_layer_execute_config *cfg,
                               uint32_t n_slots,
                               bool use_fused_gate_up) {
    if (!state || !cfg || !use_fused_gate_up || !state->has_turbomind_tp2_routed) {
        return false;
    }
    const int tp2_count = cfg->tp2_layer_count > 0 ? cfg->tp2_layer_count : 1;
    if (cfg->tp2_layer < 0 ||
        state->layer_id < cfg->tp2_layer ||
        state->layer_id >= cfg->tp2_layer + tp2_count ||
        !cfg->tp2_owner_arena ||
        !cfg->tp2_peer_arena ||
        !cfg->tp2_peer_input ||
        !cfg->tp2_peer_selected ||
        !cfg->tp2_peer_weights ||
        !cfg->tp2_peer_out ||
        !cfg->tp2_peer_recv ||
        cfg->tp2_scratch_slots < n_slots) {
        return false;
    }
    const uint64_t hidden_values = (uint64_t)n_slots * state->hidden_size;
    const uint64_t route_values = (uint64_t)n_slots * state->routes_per_token;
    return ds4_gpu_tensor_bytes(cfg->tp2_peer_input) >= hidden_values * sizeof(float) &&
           ds4_gpu_tensor_bytes(cfg->tp2_peer_out) >= hidden_values * sizeof(float) &&
           ds4_gpu_tensor_bytes(cfg->tp2_peer_recv) >= hidden_values * sizeof(float) &&
           ds4_gpu_tensor_bytes(cfg->tp2_peer_selected) >= route_values * sizeof(int32_t) &&
           ds4_gpu_tensor_bytes(cfg->tp2_peer_weights) >= route_values * sizeof(float);
}

typedef struct {
    const ds4_gpu_arena *arena;
    const ds4_gpu_turbomind_mxfp4_matrix_view *gate_up;
    const ds4_gpu_turbomind_mxfp4_matrix_view *down;
    uint32_t hidden;
    uint32_t mid;
    uint32_t n_total_experts;
    const ds4_gpu_tensor *selected;
    const ds4_gpu_tensor *weights;
    uint32_t routes;
    const ds4_gpu_tensor *x;
    uint32_t n_slots;
    ds4_gpu_tensor *out;
    int rc;
} ds4_tp2_routed_worker;

static void *tp2_routed_worker_main(void *arg) {
    ds4_tp2_routed_worker *w = (ds4_tp2_routed_worker *)arg;
    if (!w) return NULL;
    w->rc = ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_f32(
        w->arena,
        w->gate_up,
        w->down,
        w->hidden,
        w->mid,
        w->n_total_experts,
        w->selected,
        w->weights,
        w->routes,
        w->x,
        w->n_slots,
        w->out);
    return NULL;
}

static int execute_turbomind_tp2_routed(const ds4_v100_layer_state *state,
                                        const ds4_v100_layer_execute_config *cfg,
                                        const ds4_gpu_tensor *selected,
                                        const ds4_gpu_tensor *weights,
                                        const ds4_gpu_tensor *x,
                                        uint32_t n_slots,
                                        ds4_gpu_tensor *out,
                                        ds4_v100_layer_execute_report *tp2_report,
                                        char *err,
                                        size_t errlen) {
    const uint32_t hidden = state->hidden_size;
    const uint32_t routes = state->routes_per_token;
    const uint64_t hidden_bytes = (uint64_t)n_slots * hidden * sizeof(float);
    const uint64_t route_i32_bytes = (uint64_t)n_slots * routes * sizeof(int32_t);
    const uint64_t route_f32_bytes = (uint64_t)n_slots * routes * sizeof(float);
    const bool async_input = tp2_async_input_enabled();
    const bool profile = tp2_verbose_enabled();
    double profile_last_ms = 0.0;
    double copy_in_ms = 0.0;
    double owner_ms = 0.0;
    double peer_ms = 0.0;
    double copy_out_ms = 0.0;
    double reduce_ms = 0.0;
    const bool parallel_halves = tp2_parallel_halves_enabled();
    if (profile) {
        if (!ds4_gpu_set_device(ds4_gpu_arena_gpu(cfg->tp2_owner_arena)) ||
            !ds4_gpu_synchronize()) {
            return exec_error(err, errlen, "TP2 profile sync failed");
        }
        profile_last_ms = monotonic_ms();
    }
    const int copied = async_input
        ? ds4_gpu_tensor_copy_async(cfg->tp2_peer_input, 0, x, 0, hidden_bytes) &&
          ds4_gpu_tensor_copy_async(cfg->tp2_peer_selected, 0, selected, 0, route_i32_bytes) &&
          ds4_gpu_tensor_copy_async(cfg->tp2_peer_weights, 0, weights, 0, route_f32_bytes)
        : ds4_gpu_tensor_copy(cfg->tp2_peer_input, 0, x, 0, hidden_bytes) &&
          ds4_gpu_tensor_copy(cfg->tp2_peer_selected, 0, selected, 0, route_i32_bytes) &&
          ds4_gpu_tensor_copy(cfg->tp2_peer_weights, 0, weights, 0, route_f32_bytes);
    if (!copied) {
        return exec_error(err, errlen, "TP2 routed input peer copy failed");
    }
    if (profile && !profile_mark(&profile_last_ms, &copy_in_ms)) {
        return exec_error(err, errlen, "TP2 routed input profile failed");
    }
    if (parallel_halves) {
        ds4_tp2_routed_worker peer = {
            .arena = cfg->tp2_peer_arena,
            .gate_up = &state->turbomind_tp2_gate_up_view[1],
            .down = &state->turbomind_tp2_down_view[1],
            .hidden = hidden,
            .mid = state->intermediate_size / 2u,
            .n_total_experts = state->routed_experts,
            .selected = cfg->tp2_peer_selected,
            .weights = cfg->tp2_peer_weights,
            .routes = routes,
            .x = cfg->tp2_peer_input,
            .n_slots = n_slots,
            .out = cfg->tp2_peer_out,
            .rc = 1,
        };
        pthread_t peer_thread;
        const int thread_rc = pthread_create(&peer_thread, NULL, tp2_routed_worker_main, &peer);
        if (thread_rc != 0) {
            return exec_error(err, errlen, "TP2 peer thread create failed");
        }
        const int owner_rc = ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_f32(
            cfg->tp2_owner_arena,
            &state->turbomind_tp2_gate_up_view[0],
            &state->turbomind_tp2_down_view[0],
            hidden,
            state->intermediate_size / 2u,
            state->routed_experts,
            selected,
            weights,
            routes,
            x,
            n_slots,
            out);
        const int join_rc = pthread_join(peer_thread, NULL);
        if (owner_rc != 0) {
            return exec_error(err, errlen, "TP2 owner routed FFN failed");
        }
        if (join_rc != 0) {
            return exec_error(err, errlen, "TP2 peer thread join failed");
        }
        if (peer.rc != 0) {
            return exec_error(err, errlen, "TP2 peer routed FFN failed");
        }
        if (profile && !profile_mark(&profile_last_ms, &owner_ms)) {
            return exec_error(err, errlen, "TP2 parallel routed profile failed");
        }
    } else {
        if (ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_f32(
                cfg->tp2_owner_arena,
                &state->turbomind_tp2_gate_up_view[0],
                &state->turbomind_tp2_down_view[0],
                hidden,
                state->intermediate_size / 2u,
                state->routed_experts,
                selected,
                weights,
                routes,
                x,
                n_slots,
                out) != 0) {
            return exec_error(err, errlen, "TP2 owner routed FFN failed");
        }
        if (profile && !profile_mark(&profile_last_ms, &owner_ms)) {
            return exec_error(err, errlen, "TP2 owner profile failed");
        }
        if (ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_f32(
                cfg->tp2_peer_arena,
                &state->turbomind_tp2_gate_up_view[1],
                &state->turbomind_tp2_down_view[1],
                hidden,
                state->intermediate_size / 2u,
                state->routed_experts,
                cfg->tp2_peer_selected,
                cfg->tp2_peer_weights,
                routes,
                cfg->tp2_peer_input,
                n_slots,
                cfg->tp2_peer_out) != 0) {
            return exec_error(err, errlen, "TP2 peer routed FFN failed");
        }
        if (profile && !profile_mark(&profile_last_ms, &peer_ms)) {
            return exec_error(err, errlen, "TP2 peer profile failed");
        }
    }
    if (!ds4_gpu_tensor_copy(cfg->tp2_peer_recv, 0, cfg->tp2_peer_out, 0, hidden_bytes)) {
        return exec_error(err, errlen, "TP2 routed output peer copy failed");
    }
    if (profile && !profile_mark(&profile_last_ms, &copy_out_ms)) {
        return exec_error(err, errlen, "TP2 routed output profile failed");
    }
    if (!ds4_gpu_set_device(ds4_gpu_arena_gpu(cfg->tp2_owner_arena)) ||
        !ds4_gpu_add_tensor(out, out, cfg->tp2_peer_recv, (uint32_t)((uint64_t)n_slots * hidden))) {
        return exec_error(err, errlen, "TP2 routed partial sum failed");
    }
    if (profile && !profile_mark(&profile_last_ms, &reduce_ms)) {
        return exec_error(err, errlen, "TP2 routed reduce profile failed");
    }
    if (tp2_report) {
        tp2_report->timing_tp2_copy_in_ms = copy_in_ms;
        tp2_report->timing_tp2_owner_ms = owner_ms;
        tp2_report->timing_tp2_peer_ms = peer_ms;
        tp2_report->timing_tp2_copy_out_ms = copy_out_ms;
        tp2_report->timing_tp2_reduce_ms = reduce_ms;
        tp2_report->timing_tp2_total_ms = copy_in_ms + owner_ms + peer_ms + copy_out_ms + reduce_ms;
    }
    if (profile) {
        fprintf(stderr,
                "ds4: TP/EP routed boundary layer=%d slots=%u routes=%u async_input=%u parallel_halves=%u "
                "copy_in_ms=%.4f owner_ms=%.4f peer_ms=%.4f copy_out_ms=%.4f reduce_ms=%.4f total_ms=%.4f\n",
                state->layer_id,
                n_slots,
                routes,
                async_input ? 1u : 0u,
                parallel_halves ? 1u : 0u,
                copy_in_ms,
                owner_ms,
                peer_ms,
                copy_out_ms,
                reduce_ms,
                copy_in_ms + owner_ms + peer_ms + copy_out_ms + reduce_ms);
    }
    return 0;
}

static bool has_turbomind_separate_gate_up(const ds4_v100_layer_state *state) {
    return state &&
           state->turbomind_gate_view.experts_packed != 0 &&
           state->turbomind_up_view.experts_packed != 0;
}

static bool ffn_inputs_match_scratch_norm(
        const ds4_v100_layer_batch_scratch *scratch,
        const ds4_gpu_tensor *const *ffn_inputs,
        uint32_t n_slots) {
    if (!scratch || !ffn_inputs || n_slots == 0 || n_slots > DS4_V100_LAYER_MAX_BATCH) {
        return false;
    }
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (ffn_inputs[slot] != scratch->ffn_norm[slot]) return false;
    }
    return true;
}

static bool ffn_outputs_match_scratch_delta(
        const ds4_v100_layer_batch_scratch *scratch,
        ds4_gpu_tensor *const *ffn_deltas,
        uint32_t n_slots) {
    if (!scratch || !ffn_deltas || n_slots == 0 || n_slots > DS4_V100_LAYER_MAX_BATCH) {
        return false;
    }
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (ffn_deltas[slot] != scratch->ffn_delta[slot]) return false;
    }
    return true;
}

static int ensure_cached_ffn_input_ptrs(
        ds4_v100_layer_batch_scratch *scratch,
        const ds4_gpu_tensor *const *ffn_inputs,
        uint32_t n_slots,
        uint64_t min_row_bytes,
        char *err,
        size_t errlen) {
    if (!scratch || !scratch->ffn_input_ptrs || !ffn_inputs) {
        return exec_error(err, errlen, "missing FFN input pointer scratch");
    }
    if (scratch->ffn_input_ptrs_valid &&
        scratch->ffn_input_ptrs_slots == n_slots &&
        scratch->ffn_input_ptrs_min_row_bytes >= min_row_bytes) {
        return 0;
    }
    if (!ds4_gpu_tensor_write_f32_row_ptrs(scratch->ffn_input_ptrs,
                                           ffn_inputs,
                                           n_slots,
                                           min_row_bytes)) {
        return exec_error(err, errlen, "FFN input pointer table upload failed");
    }
    scratch->ffn_input_ptrs_valid = 1;
    scratch->ffn_input_ptrs_slots = n_slots;
    scratch->ffn_input_ptrs_min_row_bytes = min_row_bytes;
    return 0;
}

static double monotonic_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int profile_mark(double *last_ms, double *bucket_ms) {
    if (!last_ms || !bucket_ms) return 1;
    if (!ds4_gpu_synchronize()) return 0;
    const double now = monotonic_ms();
    if (*last_ms > 0.0 && now >= *last_ms) {
        *bucket_ms += now - *last_ms;
    }
    *last_ms = now;
    return 1;
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

static int q_head_norm_rope_tail_layer_tensor(ds4_gpu_tensor *x,
                                              uint32_t n_head,
                                              uint32_t head_dim,
                                              uint32_t pos,
                                              const ds4_v100_layer_state *state) {
    const uint32_t ratio = layer_ratio(state);
    const bool compressed = ratio != 0;
    const float freq_base = compressed ? 160000.0f : 10000.0f;
    const float freq_scale = compressed ? (1.0f / 16.0f) : 1.0f;
    const float ext_factor = compressed ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    return ds4_gpu_head_rms_norm_rope_tail_tensor(x,
                                                  1,
                                                  n_head,
                                                  head_dim,
                                                  DS4_V100_N_ROT,
                                                  pos,
                                                  compressed ? 65536u : 0u,
                                                  false,
                                                  freq_base,
                                                  freq_scale,
                                                  ext_factor,
                                                  attn_factor,
                                                  32.0f,
                                                  1.0f,
                                                  DS4_V100_RMS_EPS);
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

    if (env_flag_enabled("DS4_V100_SINGLE_SLOT_ATTN_OUTPUT_A_HMMA") &&
        env_flag_enabled("DS4_CUDA_F8_HMMA_GROUPED_ATTN_O_SINGLE")) {
        ds4_gpu_source_row_view a_view;
        if (source_view(&state->attn_output_a, &a_view, err, errlen)) return 1;
        if (ds4_gpu_arena_f8_e4m3_b128_matmul_grouped_batch_f32(
                    cfg->arena,
                    &a_view,
                    heads,
                    1,
                    DS4_V100_OUT_GROUPS,
                    DS4_V100_OUT_GROUP_RANK,
                    DS4_V100_OUT_GROUP_DIM,
                    low) != 0) {
            return exec_error(err, errlen, "attention output_a single HMMA matmul failed");
        }
    } else if (!env_flag_enabled("DS4_V100_DISABLE_GROUPED_ATTN_OUTPUT_A")) {
        ds4_gpu_source_row_view a_view;
        if (source_view(&state->attn_output_a, &a_view, err, errlen)) return 1;
        if (ds4_gpu_arena_f8_e4m3_b128_matmul_grouped_f32(
                    cfg->arena,
                    &a_view,
                    heads,
                    DS4_V100_OUT_GROUPS,
                    DS4_V100_OUT_GROUP_RANK,
                    DS4_V100_OUT_GROUP_DIM,
                    low) != 0) {
            return exec_error(err, errlen, "attention output_a grouped matmul failed");
        }
    } else {
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
    }

    ds4_gpu_source_row_view b_view;
    if (source_view(&state->attn_output_b, &b_view, err, errlen)) return 1;
    if (ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena, &b_view, low, out) != 0) {
        return exec_error(err, errlen, "attention output_b matmul failed");
    }
    return 0;
}

static int grouped_attention_output_b(const ds4_v100_layer_state *state,
                                      const ds4_v100_layer_execute_config *cfg,
                                      const ds4_gpu_tensor *low,
                                      ds4_gpu_tensor *out,
                                      char *err,
                                      size_t errlen) {
    if (state->attn_output_b.rows != state->hidden_size ||
        state->attn_output_b.cols != DS4_V100_OUT_GROUPS * DS4_V100_OUT_GROUP_RANK) {
        return exec_error(err, errlen, "attention output_b dimensions do not match DS4");
    }
    if (!low || !out ||
        ds4_gpu_tensor_bytes(low) < (uint64_t)DS4_V100_OUT_GROUPS * DS4_V100_OUT_GROUP_RANK * sizeof(float) ||
        ds4_gpu_tensor_bytes(out) < (uint64_t)state->hidden_size * sizeof(float)) {
        return exec_error(err, errlen, "attention output_b tensor is too small");
    }
    ds4_gpu_source_row_view b_view;
    if (source_view(&state->attn_output_b, &b_view, err, errlen)) return 1;
    if (ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena, &b_view, low, out) != 0) {
        return exec_error(err, errlen, "attention output_b matmul failed");
    }
    return 0;
}

static int grouped_attention_output_a(const ds4_v100_layer_state *state,
                                      const ds4_v100_layer_execute_config *cfg,
                                      const ds4_gpu_tensor *heads,
                                      ds4_gpu_tensor *low,
                                      char *err,
                                      size_t errlen) {
    if (state->attn_output_a.rows != DS4_V100_OUT_GROUPS * DS4_V100_OUT_GROUP_RANK ||
        state->attn_output_a.cols != DS4_V100_OUT_GROUP_DIM) {
        return exec_error(err, errlen, "attention grouped output_a dimensions do not match DS4");
    }
    if (!heads || !low ||
        ds4_gpu_tensor_bytes(heads) < (uint64_t)DS4_V100_N_HEAD * DS4_V100_HEAD_DIM * sizeof(float) ||
        ds4_gpu_tensor_bytes(low) < (uint64_t)DS4_V100_OUT_GROUPS * DS4_V100_OUT_GROUP_RANK * sizeof(float)) {
        return exec_error(err, errlen, "attention grouped output_a tensor is too small");
    }

    if (env_flag_enabled("DS4_V100_SINGLE_SLOT_ATTN_OUTPUT_A_HMMA") &&
        env_flag_enabled("DS4_CUDA_F8_HMMA_GROUPED_ATTN_O_SINGLE")) {
        ds4_gpu_source_row_view a_view;
        if (source_view(&state->attn_output_a, &a_view, err, errlen)) return 1;
        if (ds4_gpu_arena_f8_e4m3_b128_matmul_grouped_batch_f32(
                    cfg->arena,
                    &a_view,
                    heads,
                    1,
                    DS4_V100_OUT_GROUPS,
                    DS4_V100_OUT_GROUP_RANK,
                    DS4_V100_OUT_GROUP_DIM,
                    low) != 0) {
            return exec_error(err, errlen, "attention output_a single HMMA matmul failed");
        }
        return 0;
    }

    if (!env_flag_enabled("DS4_V100_DISABLE_GROUPED_ATTN_OUTPUT_A")) {
        ds4_gpu_source_row_view a_view;
        if (source_view(&state->attn_output_a, &a_view, err, errlen)) return 1;
        if (ds4_gpu_arena_f8_e4m3_b128_matmul_grouped_f32(
                    cfg->arena,
                    &a_view,
                    heads,
                    DS4_V100_OUT_GROUPS,
                    DS4_V100_OUT_GROUP_RANK,
                    DS4_V100_OUT_GROUP_DIM,
                    low) != 0) {
            return exec_error(err, errlen, "attention output_a grouped matmul failed");
        }
        return 0;
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
            return exec_error(err, errlen, "failed to create grouped output_a tensor views");
        }
        const int rc = ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena,
                                                             &a_view,
                                                             head_view,
                                                             low_view);
        ds4_gpu_tensor_free(low_view);
        ds4_gpu_tensor_free(head_view);
        if (rc != 0) return exec_error(err, errlen, "attention output_a grouped matmul failed");
    }
    return 0;
}
