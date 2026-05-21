#include "ds4_v100_layer_execute.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    free_tensor_array(scratch->attn_q_view, DS4_V100_LAYER_MAX_BATCH);
    free_tensor_array(scratch->attn_q_a_norm_view, DS4_V100_LAYER_MAX_BATCH);
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
        scratch->attn_q_a_norm_view[slot] =
            ds4_gpu_tensor_view(scratch->attn_q_a_norm_batch,
                                (uint64_t)slot * q_rank_bytes,
                                q_rank_bytes);
        scratch->attn_q_view[slot] =
            ds4_gpu_tensor_view(scratch->attn_q_batch,
                                (uint64_t)slot * q_width_bytes,
                                q_width_bytes);
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
            !scratch->attn_q_a_norm_view[slot] ||
            !scratch->attn_q_view[slot] ||
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

static bool tp2_routed_enabled(const ds4_v100_layer_state *state,
                               const ds4_v100_layer_execute_config *cfg,
                               uint32_t n_slots,
                               bool use_fused_gate_up) {
    if (!state || !cfg || !use_fused_gate_up || !state->has_turbomind_tp2_routed) {
        return false;
    }
    if (cfg->tp2_layer != state->layer_id ||
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

static int execute_turbomind_tp2_routed(const ds4_v100_layer_state *state,
                                        const ds4_v100_layer_execute_config *cfg,
                                        const ds4_gpu_tensor *selected,
                                        const ds4_gpu_tensor *weights,
                                        const ds4_gpu_tensor *x,
                                        uint32_t n_slots,
                                        ds4_gpu_tensor *out,
                                        char *err,
                                        size_t errlen) {
    const uint32_t hidden = state->hidden_size;
    const uint32_t routes = state->routes_per_token;
    const uint64_t hidden_bytes = (uint64_t)n_slots * hidden * sizeof(float);
    const uint64_t route_i32_bytes = (uint64_t)n_slots * routes * sizeof(int32_t);
    const uint64_t route_f32_bytes = (uint64_t)n_slots * routes * sizeof(float);
    if (!ds4_gpu_tensor_copy(cfg->tp2_peer_input, 0, x, 0, hidden_bytes) ||
        !ds4_gpu_tensor_copy(cfg->tp2_peer_selected, 0, selected, 0, route_i32_bytes) ||
        !ds4_gpu_tensor_copy(cfg->tp2_peer_weights, 0, weights, 0, route_f32_bytes)) {
        return exec_error(err, errlen, "TP2 routed input peer copy failed");
    }
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
    if (!ds4_gpu_tensor_copy(cfg->tp2_peer_recv, 0, cfg->tp2_peer_out, 0, hidden_bytes)) {
        return exec_error(err, errlen, "TP2 routed output peer copy failed");
    }
    if (!ds4_gpu_set_device(ds4_gpu_arena_gpu(cfg->tp2_owner_arena)) ||
        !ds4_gpu_add_tensor(out, out, cfg->tp2_peer_recv, (uint32_t)((uint64_t)n_slots * hidden))) {
        return exec_error(err, errlen, "TP2 routed partial sum failed");
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
                                        model_offset_for_binding(cfg, &state->attn_norm),
                                        hidden_n,
                                        DS4_V100_RMS_EPS) ||
        ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena, &q_a_v, attn_norm, q_a) != 0 ||
        !ds4_gpu_rms_norm_weight_tensor(q_a_norm,
                                        q_a,
                                        cfg->model_map,
                                        cfg->model_size,
                                        model_offset_for_binding(cfg, &state->attn_q_a_norm),
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
                                        model_offset_for_binding(cfg, &state->attn_kv_a_norm),
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

static int execute_attention_output_batch(const ds4_v100_layer_state *state,
                                          const ds4_v100_layer_execute_config *cfgs,
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
        ds4_v100_attention_inputs attn_inputs = {
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
        if (!ds4_gpu_head_rms_norm_tensor(q_view, 1, DS4_V100_N_HEAD, DS4_V100_HEAD_DIM, DS4_V100_RMS_EPS) ||
            !rope_tail_layer_tensor(q_view,
                                    DS4_V100_N_HEAD,
                                    DS4_V100_HEAD_DIM,
                                    cfgs[slot].position,
                                    state,
                                    false) ||
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
    const uint32_t routes = state->routes_per_token;
    if (routes == 0 || routes > 6u) {
        return exec_error(err, errlen, "unsupported route count %u", routes);
    }
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

    const bool use_single_shared_pair =
        env_flag_enabled("DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE");
    int rc = 1;
    const bool use_scratch =
        cfg->batch_scratch != NULL && single_slot_batch_scratch_enabled();
    ds4_gpu_tensor *router_t = NULL;
    ds4_gpu_tensor *probs_t = NULL;
    ds4_gpu_tensor *selected_t = NULL;
    ds4_gpu_tensor *weights_t = NULL;
    ds4_gpu_tensor *routed_mid_t = NULL;
    ds4_gpu_tensor *accum_a = NULL;
    ds4_gpu_tensor *shared_gate_t = NULL;
    ds4_gpu_tensor *shared_up_t = NULL;
    ds4_gpu_tensor *shared_mid_t = NULL;
    ds4_gpu_tensor *shared_t = NULL;
    bool used_tp2 = false;
    if (use_scratch &&
        ensure_batch_scratch(cfg->batch_scratch,
                             hidden,
                             mid,
                             routes,
                             state->q_lora_rank,
                             state->q_width,
                             state->kv_latent_width,
                             state->attention_output_rank,
                             err,
                             errlen)) {
        goto done;
    }
    router_t = use_scratch ? cfg->batch_scratch->ffn_router
        : ds4_gpu_tensor_alloc(256u * sizeof(float));
    probs_t = use_scratch ? cfg->batch_scratch->ffn_probs
        : ds4_gpu_tensor_alloc(256u * sizeof(float));
    selected_t = use_scratch ? cfg->batch_scratch->ffn_selected
        : ds4_gpu_tensor_alloc(6u * sizeof(int32_t));
    weights_t = use_scratch ? cfg->batch_scratch->ffn_weights
        : ds4_gpu_tensor_alloc(6u * sizeof(float));
    routed_mid_t = use_scratch ? cfg->batch_scratch->ffn_routed_mid
        : ds4_gpu_tensor_alloc((uint64_t)routes * mid * sizeof(float));
    accum_a = use_scratch ? cfg->batch_scratch->ffn_routed_out_view[0]
        : ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    shared_gate_t = use_single_shared_pair ? NULL
        : (use_scratch ? cfg->batch_scratch->ffn_shared_gate
                       : ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float)));
    shared_up_t = use_single_shared_pair ? NULL
        : (use_scratch ? cfg->batch_scratch->ffn_shared_up
                       : ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float)));
    shared_mid_t = use_scratch ? cfg->batch_scratch->ffn_shared_mid
        : ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    shared_t = use_scratch ? cfg->batch_scratch->ffn_shared
        : ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));

    int32_t selected[6] = {0};
    float weights[6] = {0};
    const bool readback_routes = !cfg->suppress_router_readback;
    if (!router_t || !probs_t || !selected_t || !weights_t ||
        !routed_mid_t || !accum_a ||
        (!use_single_shared_pair && (!shared_gate_t || !shared_up_t)) ||
        !shared_mid_t || !shared_t) {
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
                                      bias_mode ? model_offset_for_binding(cfg, &state->router_bias) : 0,
                                      hash_mode ? model_offset_for_binding(cfg, &state->router_hash) : 0,
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
    if (readback_routes) {
        if (!ds4_gpu_tensor_read(selected_t, 0, selected, sizeof(selected)) ||
            !ds4_gpu_tensor_read(weights_t, 0, weights, sizeof(weights))) {
            exec_error(err, errlen, "router readback failed");
            goto done;
        }
        for (uint32_t route = 0; route < routes; route++) {
            if (selected[route] < 0 || (uint32_t)selected[route] >= state->routed_experts) {
                exec_error(err, errlen, "router selected invalid expert %d", selected[route]);
                goto done;
            }
        }
    }

    if (state->has_turbomind_routed) {
        const bool use_fused_gate_up =
            state->has_turbomind_fused_gate_up &&
            env_flag_default_enabled("DS4_V100_TURBOMIND_FUSED_GATE_UP");
        if (!use_fused_gate_up && !has_turbomind_separate_gate_up(state)) {
            exec_error(err, errlen,
                       "TurboMind fused gate_up disabled but no separate gate/up tensors are bound");
            goto done;
        }
        int tm_rc = 1;
        const bool use_tp2 = tp2_routed_enabled(state, cfg, 1, use_fused_gate_up);
        if (use_tp2) {
            used_tp2 = true;
            tm_rc = execute_turbomind_tp2_routed(state,
                                                 cfg,
                                                 selected_t,
                                                 weights_t,
                                                 ffn_input,
                                                 1,
                                                 accum_a,
                                                 err,
                                                 errlen);
        } else {
            tm_rc = use_fused_gate_up
                ? ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_f32(
                  cfg->arena,
                  &state->turbomind_gate_up_view,
                  &state->turbomind_down_view,
                  hidden,
                  mid,
                  state->routed_experts,
                  selected_t,
                  weights_t,
                  routes,
                  ffn_input,
                  1,
                  accum_a)
                : ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_f32(
                  cfg->arena,
                  &state->turbomind_gate_view,
                  &state->turbomind_up_view,
                  &state->turbomind_down_view,
                  hidden,
                  mid,
                  state->routed_experts,
                  selected_t,
                  weights_t,
                  routes,
                  ffn_input,
                  1,
                  accum_a);
        }
        if (tm_rc != 0) {
            if (!err || !errlen || !err[0]) {
                exec_error(err, errlen, "TurboMind routed FFN failed");
            }
            goto done;
        }
    } else {
        ds4_v100_route_matrices route0;
        if (ds4_v100_layer_state_route_matrices(state, 0, &route0, err, errlen)) {
            goto done;
        }
        if (route0.gate.bytes != route0.up.bytes ||
            route0.gate.row_bytes != route0.up.row_bytes ||
            route0.gate.rows != route0.up.rows ||
            route0.gate.cols != route0.up.cols) {
            exec_error(err, errlen, "routed gate/up MXFP4 layouts differ");
            goto done;
        }
        if (ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_f32(
                cfg->arena,
                state->routed_gate_binding.shard_offset,
                state->routed_gate_binding.byte_length,
                state->routed_up_binding.shard_offset,
                state->routed_up_binding.byte_length,
                state->routed_down_binding.shard_offset,
                state->routed_down_binding.byte_length,
                route0.gate.bytes,
                (uint32_t)route0.gate.row_bytes,
                route0.down.bytes,
                (uint32_t)route0.down.row_bytes,
                hidden,
                mid,
                state->routed_experts,
                selected_t,
                weights_t,
                routes,
                ffn_input,
                routed_mid_t,
                accum_a) != 0) {
            exec_error(err, errlen, "grouped routed FFN failed");
            goto done;
        }
    }

    ds4_gpu_tensor *accum = accum_a;
    const bool use_shared_down_add =
        env_flag_enabled("DS4_V100_F8_SHARED_DOWN_ADD");

    int shared_rc = 0;
    if (use_single_shared_pair) {
        shared_rc =
            ds4_gpu_arena_f8_e4m3_b128_pair_swiglu_f32(cfg->arena,
                                                       &shared_gate_v,
                                                       &shared_up_v,
                                                       ffn_input,
                                                       shared_mid_t,
                                                       10.0f,
                                                       1.0f) != 0;
    } else {
        shared_rc =
            ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena,
                                                 &shared_gate_v,
                                                 ffn_input,
                                                 shared_gate_t) != 0 ||
            ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena,
                                                 &shared_up_v,
                                                 ffn_input,
                                                 shared_up_t) != 0 ||
            !ds4_gpu_swiglu_tensor(shared_mid_t,
                                  shared_gate_t,
                                  shared_up_t,
                                  mid,
                                  10.0f,
                                  1.0f);
    }
    if (!shared_rc) {
        if (use_shared_down_add) {
            shared_rc =
                ds4_gpu_arena_f8_e4m3_b128_matmul_add_f32(cfg->arena,
                                                          &shared_down_v,
                                                          shared_mid_t,
                                                          accum,
                                                          ffn_delta) != 0;
        } else {
            shared_rc =
                ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena,
                                                     &shared_down_v,
                                                     shared_mid_t,
                                                     shared_t) != 0 ||
                !ds4_gpu_add_tensor(ffn_delta, accum, shared_t, hidden);
        }
    }
    if (shared_rc) {
        exec_error(err, errlen, "shared FFN failed");
        goto done;
    }

    if (report) {
        memset(report, 0, sizeof(*report));
        report->routes = routes;
        report->turbomind_routed = state->has_turbomind_routed ? 1u : 0u;
        report->turbomind_tp2_routed = used_tp2 ? 1u : 0u;
        if (readback_routes) {
            for (uint32_t i = 0; i < routes && i < 6u; i++) {
                report->selected_experts[i] = selected[i];
                report->route_weights[i] = weights[i];
            }
        }
    }
    rc = 0;

done:
    if (!use_scratch) {
        ds4_gpu_tensor_free(shared_t);
        ds4_gpu_tensor_free(shared_mid_t);
        ds4_gpu_tensor_free(shared_up_t);
        ds4_gpu_tensor_free(shared_gate_t);
        ds4_gpu_tensor_free(accum_a);
        ds4_gpu_tensor_free(routed_mid_t);
        ds4_gpu_tensor_free(weights_t);
        ds4_gpu_tensor_free(selected_t);
        ds4_gpu_tensor_free(probs_t);
        ds4_gpu_tensor_free(router_t);
    }
    return rc;
}

static int execute_ffn_delta_batch(const ds4_v100_layer_state *state,
                                   const ds4_v100_layer_execute_config *cfgs,
                                   const ds4_gpu_tensor *const *ffn_inputs,
                                   ds4_gpu_tensor *const *ffn_deltas,
                                   uint32_t n_slots,
                                   ds4_v100_layer_execute_report *reports,
                                   char *err,
                                   size_t errlen) {
    if (!state || !cfgs || !ffn_inputs || !ffn_deltas ||
        n_slots == 0 || n_slots > DS4_V100_LAYER_MAX_BATCH) {
        return exec_error(err, errlen, "invalid FFN batch inputs");
    }
    if (n_slots == 1) {
        return execute_ffn_delta(state,
                                 &cfgs[0],
                                 ffn_inputs[0],
                                 ffn_deltas[0],
                                 reports ? &reports[0] : NULL,
                                 err,
                                 errlen);
    }

    const bool hash_mode = state->router_kind == DS4_V100_ROUTER_HASH && state->has_hash_router;
    const bool bias_mode = state->router_kind == DS4_V100_ROUTER_BIAS && state->has_bias_router;
    if (!hash_mode && !bias_mode) {
        return exec_error(err, errlen, "layer executor requires hash or bias router metadata");
    }

    const uint32_t hidden = state->hidden_size;
    const uint32_t mid = state->intermediate_size;
    const uint32_t routes = state->routes_per_token;
    if (routes == 0 || routes > 6u) {
        return exec_error(err, errlen, "unsupported route count %u", routes);
    }
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!ffn_inputs[slot] || !ffn_deltas[slot] ||
            ds4_gpu_tensor_bytes(ffn_inputs[slot]) < (uint64_t)hidden * sizeof(float) ||
            ds4_gpu_tensor_bytes(ffn_deltas[slot]) < (uint64_t)hidden * sizeof(float)) {
            return exec_error(err, errlen, "FFN batch tensor is too small");
        }
        if (cfgs[slot].arena != cfgs[0].arena ||
            cfgs[slot].model_map != cfgs[0].model_map ||
            cfgs[slot].model_size != cfgs[0].model_size ||
            cfgs[slot].batch_scratch != cfgs[0].batch_scratch) {
            return exec_error(err, errlen, "FFN batch cfgs must share arena/model");
        }
    }

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

    const bool use_scratch = cfgs[0].batch_scratch != NULL;
    if (use_scratch && ensure_batch_scratch(cfgs[0].batch_scratch,
                                            hidden,
                                            mid,
                                            routes,
                                            state->q_lora_rank,
                                            state->q_width,
                                            state->kv_latent_width,
                                            state->attention_output_rank,
                                            err,
                                            errlen)) {
        return 1;
    }
    const bool batch_shared_f8 = env_flag_enabled("DS4_V100_BATCH_SHARED_F8");
    const bool use_single_shared_pair =
        env_flag_enabled("DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE");
    const bool use_shared_down_add =
        env_flag_enabled("DS4_V100_F8_SHARED_DOWN_ADD");
    ds4_gpu_tensor *router_t = use_scratch ? cfgs[0].batch_scratch->ffn_router
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * 256u * sizeof(float));
    ds4_gpu_tensor *probs_t = use_scratch ? cfgs[0].batch_scratch->ffn_probs
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * 256u * sizeof(float));
    ds4_gpu_tensor *selected_t = use_scratch ? cfgs[0].batch_scratch->ffn_selected
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * routes * sizeof(int32_t));
    ds4_gpu_tensor *weights_t = use_scratch ? cfgs[0].batch_scratch->ffn_weights
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * routes * sizeof(float));
    ds4_gpu_tensor *tokens_t = use_scratch ? cfgs[0].batch_scratch->ffn_tokens
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * sizeof(int32_t));
    ds4_gpu_tensor *input_ptrs_t = use_scratch ? cfgs[0].batch_scratch->ffn_input_ptrs
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * sizeof(void *));
    ds4_gpu_tensor *routed_mid_t = use_scratch ? cfgs[0].batch_scratch->ffn_routed_mid
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * routes * mid * sizeof(float));
    ds4_gpu_tensor *routed_out_t = use_scratch ? cfgs[0].batch_scratch->ffn_routed_out
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * hidden * sizeof(float));
    ds4_gpu_tensor *shared_gate_t = use_scratch ? cfgs[0].batch_scratch->ffn_shared_gate
        : ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *shared_up_t = use_scratch ? cfgs[0].batch_scratch->ffn_shared_up
        : ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *shared_mid_t = use_scratch ? cfgs[0].batch_scratch->ffn_shared_mid
        : ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *shared_t = use_scratch ? cfgs[0].batch_scratch->ffn_shared
        : ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *shared_mid_batch_t = use_scratch ? cfgs[0].batch_scratch->ffn_shared_mid_batch
        : (batch_shared_f8 ? ds4_gpu_tensor_alloc((uint64_t)n_slots * mid * sizeof(float)) : NULL);
    ds4_gpu_tensor *shared_batch_t = use_scratch ? cfgs[0].batch_scratch->ffn_shared_batch
        : (batch_shared_f8 ? ds4_gpu_tensor_alloc((uint64_t)n_slots * hidden * sizeof(float)) : NULL);

    int rc = 1;
    int32_t tokens[DS4_V100_LAYER_MAX_BATCH] = {0};
    int32_t selected[DS4_V100_LAYER_MAX_BATCH * 6u] = {0};
    float weights[DS4_V100_LAYER_MAX_BATCH * 6u] = {0};
    bool readback_routes = false;
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!cfgs[slot].suppress_router_readback) {
            readback_routes = true;
            break;
        }
    }
    if (!router_t || !probs_t || !selected_t || !weights_t || !tokens_t || !input_ptrs_t ||
        !routed_mid_t || !routed_out_t ||
        !shared_gate_t || !shared_up_t || !shared_mid_t || !shared_t ||
        (batch_shared_f8 && (!shared_mid_batch_t || !shared_batch_t))) {
        exec_error(err, errlen, "failed to allocate FFN batch tensors");
        goto done;
    }

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        tokens[slot] = (int32_t)cfgs[slot].router_token;
        ds4_gpu_tensor *router_view =
            ds4_gpu_tensor_view(router_t,
                                (uint64_t)slot * 256u * sizeof(float),
                                256u * sizeof(float));
        if (!router_view) {
            exec_error(err, errlen, "FFN batch router view failed");
            goto done;
        }
        const int router_rc =
            ds4_gpu_arena_f32_matmul_f32(cfgs[slot].arena,
                                         &router_v,
                                         ffn_inputs[slot],
                                         router_view);
        ds4_gpu_tensor_free(router_view);
        if (router_rc != 0) {
            exec_error(err, errlen, "FFN batch router matmul failed");
            goto done;
        }
    }
    if (!ds4_gpu_tensor_write(tokens_t,
                              0,
                              tokens,
                              (uint64_t)n_slots * sizeof(int32_t))) {
        exec_error(err, errlen, "FFN batch token upload failed");
        goto done;
    }
    if (!ds4_gpu_router_select_batch_tensor(selected_t,
                                            weights_t,
                                            probs_t,
                                            cfgs[0].model_map,
                                            cfgs[0].model_size,
                                            bias_mode ? model_offset_for_binding(&cfgs[0], &state->router_bias) : 0,
                                            hash_mode ? model_offset_for_binding(&cfgs[0], &state->router_hash) : 0,
                                            hash_mode ? (uint32_t)state->router_hash.shape[1] : 0,
                                            0,
                                            0,
                                            bias_mode,
                                            hash_mode,
                                            router_t,
                                            tokens_t,
                                            n_slots)) {
        exec_error(err, errlen, "FFN batch router select failed");
        goto done;
    }
    if (readback_routes) {
        if (!ds4_gpu_tensor_read(selected_t,
                                 0,
                                 selected,
                                 (uint64_t)n_slots * routes * sizeof(int32_t)) ||
            !ds4_gpu_tensor_read(weights_t,
                                 0,
                                 weights,
                                 (uint64_t)n_slots * routes * sizeof(float))) {
            exec_error(err, errlen, "FFN batch router readback failed");
            goto done;
        }
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            for (uint32_t route = 0; route < routes; route++) {
                const int32_t expert = selected[(uint64_t)slot * routes + route];
                if (expert < 0 || (uint32_t)expert >= state->routed_experts) {
                    exec_error(err, errlen, "FFN batch selected invalid expert %d", expert);
                    goto done;
                }
            }
        }
    }

    const uint64_t ffn_input_min_bytes = (uint64_t)hidden * sizeof(float);
    const bool use_cached_input_ptrs =
        use_scratch &&
        input_ptrs_t == cfgs[0].batch_scratch->ffn_input_ptrs &&
        ffn_inputs_match_scratch_norm(cfgs[0].batch_scratch, ffn_inputs, n_slots);
    if (use_cached_input_ptrs &&
        ensure_cached_ffn_input_ptrs(cfgs[0].batch_scratch,
                                     ffn_inputs,
                                     n_slots,
                                     ffn_input_min_bytes,
                                     err,
                                     errlen)) {
        goto done;
    }

    const bool use_fused_gate_up =
        state->has_turbomind_fused_gate_up &&
        env_flag_default_enabled("DS4_V100_TURBOMIND_FUSED_GATE_UP");
    const bool use_tp2 =
        use_scratch &&
        use_cached_input_ptrs &&
        cfgs[0].batch_scratch->ffn_norm_batch &&
        tp2_routed_enabled(state, &cfgs[0], n_slots, use_fused_gate_up);
    const bool use_direct_delta =
        env_flag_enabled("DS4_V100_FFN_DIRECT_DELTA") &&
        batch_shared_f8 &&
        state->has_turbomind_routed &&
        !use_tp2 &&
        use_scratch &&
        use_cached_input_ptrs &&
        ffn_outputs_match_scratch_delta(cfgs[0].batch_scratch, ffn_deltas, n_slots) &&
        cfgs[0].batch_scratch->ffn_norm_batch &&
        cfgs[0].batch_scratch->ffn_delta_batch;
    if (use_direct_delta) {
        if (!use_fused_gate_up && !has_turbomind_separate_gate_up(state)) {
            exec_error(err, errlen,
                       "TurboMind fused gate_up disabled but no separate gate/up tensors are bound");
            goto done;
        }
        if (ds4_gpu_arena_f8_e4m3_b128_pair_swiglu_batch_ptr_table_f32(
                cfgs[0].arena,
                &shared_gate_v,
                &shared_up_v,
                input_ptrs_t,
                n_slots,
                shared_mid_batch_t,
                10.0f,
                1.0f) != 0 ||
            ds4_gpu_arena_f8_e4m3_b128_matmul_batch_f32(cfgs[0].arena,
                                                        &shared_down_v,
                                                        shared_mid_batch_t,
                                                        n_slots,
                                                        cfgs[0].batch_scratch->ffn_delta_batch) != 0) {
            exec_error(err, errlen, "shared FFN direct-delta batch failed");
            goto done;
        }
        const int tm_rc = use_fused_gate_up
            ? ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_accum_f32(
                  cfgs[0].arena,
                  &state->turbomind_gate_up_view,
                  &state->turbomind_down_view,
                  hidden,
                  mid,
                  state->routed_experts,
                  selected_t,
                  weights_t,
                  routes,
                  cfgs[0].batch_scratch->ffn_norm_batch,
                  n_slots,
                  cfgs[0].batch_scratch->ffn_delta_batch)
            : ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_accum_f32(
                  cfgs[0].arena,
                  &state->turbomind_gate_view,
                  &state->turbomind_up_view,
                  &state->turbomind_down_view,
                  hidden,
                  mid,
                  state->routed_experts,
                  selected_t,
                  weights_t,
                  routes,
                  cfgs[0].batch_scratch->ffn_norm_batch,
                  n_slots,
                  cfgs[0].batch_scratch->ffn_delta_batch);
        if (tm_rc != 0) {
            exec_error(err, errlen, "TurboMind routed FFN direct-delta batch failed");
            goto done;
        }
        goto fill_reports;
    }

    if (state->has_turbomind_routed) {
        if (!use_fused_gate_up && !has_turbomind_separate_gate_up(state)) {
            exec_error(err, errlen,
                       "TurboMind fused gate_up disabled but no separate gate/up tensors are bound");
            goto done;
        }
        int tm_rc = 1;
        if (use_tp2) {
            tm_rc = execute_turbomind_tp2_routed(state,
                                                 &cfgs[0],
                                                 selected_t,
                                                 weights_t,
                                                 cfgs[0].batch_scratch->ffn_norm_batch,
                                                 n_slots,
                                                 routed_out_t,
                                                 err,
                                                 errlen);
        } else if (use_cached_input_ptrs) {
            tm_rc = use_fused_gate_up
                ? ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_batch_ptr_table_f32(
                      cfgs[0].arena,
                      &state->turbomind_gate_up_view,
                      &state->turbomind_down_view,
                      hidden,
                      mid,
                      state->routed_experts,
                      selected_t,
                      weights_t,
                      routes,
                      input_ptrs_t,
                      n_slots,
                      routed_out_t)
                : ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_batch_ptr_table_f32(
                      cfgs[0].arena,
                      &state->turbomind_gate_view,
                      &state->turbomind_up_view,
                      &state->turbomind_down_view,
                      hidden,
                      mid,
                      state->routed_experts,
                      selected_t,
                      weights_t,
                      routes,
                      input_ptrs_t,
                      n_slots,
                      routed_out_t);
        } else {
            tm_rc = use_fused_gate_up
                ? ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_batch_ptrs_f32(
                      cfgs[0].arena,
                      &state->turbomind_gate_up_view,
                      &state->turbomind_down_view,
                      hidden,
                      mid,
                      state->routed_experts,
                      selected_t,
                      weights_t,
                      routes,
                      input_ptrs_t,
                      ffn_inputs,
                      n_slots,
                      routed_out_t)
                : ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_batch_ptrs_f32(
                      cfgs[0].arena,
                      &state->turbomind_gate_view,
                      &state->turbomind_up_view,
                      &state->turbomind_down_view,
                      hidden,
                      mid,
                      state->routed_experts,
                      selected_t,
                      weights_t,
                      routes,
                      input_ptrs_t,
                      ffn_inputs,
                      n_slots,
                      routed_out_t);
        }
        if (tm_rc != 0) {
            if (!err || !errlen || !err[0]) {
                exec_error(err, errlen, "TurboMind routed FFN batch failed");
            }
            goto done;
        }
    } else {
        ds4_v100_route_matrices route0;
        if (ds4_v100_layer_state_route_matrices(state, 0, &route0, err, errlen)) {
            goto done;
        }
        if (route0.gate.bytes != route0.up.bytes ||
            route0.gate.row_bytes != route0.up.row_bytes ||
            route0.gate.rows != route0.up.rows ||
            route0.gate.cols != route0.up.cols) {
            exec_error(err, errlen, "routed gate/up MXFP4 layouts differ");
            goto done;
        }
        if (ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_batch_ptrs_f32(
                cfgs[0].arena,
                state->routed_gate_binding.shard_offset,
                state->routed_gate_binding.byte_length,
                state->routed_up_binding.shard_offset,
                state->routed_up_binding.byte_length,
                state->routed_down_binding.shard_offset,
                state->routed_down_binding.byte_length,
                route0.gate.bytes,
                (uint32_t)route0.gate.row_bytes,
                route0.down.bytes,
                (uint32_t)route0.down.row_bytes,
                hidden,
                mid,
                state->routed_experts,
                selected_t,
                weights_t,
                routes,
                input_ptrs_t,
                ffn_inputs,
                n_slots,
                routed_mid_t,
                routed_out_t) != 0) {
            exec_error(err, errlen, "grouped routed FFN batch failed");
            goto done;
        }
    }

    if (batch_shared_f8) {
        const bool use_batch_shared_down_add =
            use_shared_down_add &&
            use_scratch &&
            ffn_outputs_match_scratch_delta(cfgs[0].batch_scratch, ffn_deltas, n_slots) &&
            cfgs[0].batch_scratch->ffn_delta_batch;
        if (ds4_gpu_arena_f8_e4m3_b128_pair_swiglu_batch_ptr_table_f32(
                cfgs[0].arena,
                &shared_gate_v,
                &shared_up_v,
                input_ptrs_t,
                n_slots,
                shared_mid_batch_t,
                10.0f,
                1.0f) != 0) {
            exec_error(err, errlen, "shared FFN batch failed");
            goto done;
        }
        if (use_batch_shared_down_add) {
            if (ds4_gpu_arena_f8_e4m3_b128_matmul_batch_add_f32(
                    cfgs[0].arena,
                    &shared_down_v,
                    shared_mid_batch_t,
                    n_slots,
                    routed_out_t,
                    cfgs[0].batch_scratch->ffn_delta_batch) != 0) {
                exec_error(err, errlen, "shared FFN batch fused-add failed");
                goto done;
            }
        } else {
            if (ds4_gpu_arena_f8_e4m3_b128_matmul_batch_f32(cfgs[0].arena,
                                                            &shared_down_v,
                                                            shared_mid_batch_t,
                                                            n_slots,
                                                            shared_batch_t) != 0) {
                exec_error(err, errlen, "shared FFN batch failed");
                goto done;
            }
            for (uint32_t slot = 0; slot < n_slots; slot++) {
                ds4_gpu_tensor *routed_view = use_scratch
                    ? cfgs[0].batch_scratch->ffn_routed_out_view[slot]
                    : ds4_gpu_tensor_view(routed_out_t,
                                          (uint64_t)slot * hidden * sizeof(float),
                                          (uint64_t)hidden * sizeof(float));
                ds4_gpu_tensor *shared_view = use_scratch
                    ? cfgs[0].batch_scratch->ffn_shared_batch_view[slot]
                    : ds4_gpu_tensor_view(shared_batch_t,
                                          (uint64_t)slot * hidden * sizeof(float),
                                          (uint64_t)hidden * sizeof(float));
                if (!routed_view || !shared_view) {
                    if (!use_scratch) {
                        ds4_gpu_tensor_free(shared_view);
                        ds4_gpu_tensor_free(routed_view);
                    }
                    exec_error(err, errlen, "FFN batch output view failed");
                    goto done;
                }
                const int shared_rc = !ds4_gpu_add_tensor(ffn_deltas[slot],
                                                          routed_view,
                                                          shared_view,
                                                          hidden);
                if (!use_scratch) {
                    ds4_gpu_tensor_free(shared_view);
                    ds4_gpu_tensor_free(routed_view);
                }
                if (shared_rc) {
                    exec_error(err, errlen, "shared FFN batch slot %u failed", slot);
                    goto done;
                }
            }
        }
    } else {
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            ds4_gpu_tensor *routed_view = use_scratch
                ? cfgs[0].batch_scratch->ffn_routed_out_view[slot]
                : ds4_gpu_tensor_view(routed_out_t,
                                      (uint64_t)slot * hidden * sizeof(float),
                                      (uint64_t)hidden * sizeof(float));
            if (!routed_view) {
                if (!use_scratch) ds4_gpu_tensor_free(routed_view);
                exec_error(err, errlen, "FFN batch routed output view failed");
                goto done;
            }
            int shared_rc = 0;
            if (use_single_shared_pair) {
                shared_rc =
                    ds4_gpu_arena_f8_e4m3_b128_pair_swiglu_f32(cfgs[slot].arena,
                                                               &shared_gate_v,
                                                               &shared_up_v,
                                                               ffn_inputs[slot],
                                                               shared_mid_t,
                                                               10.0f,
                                                               1.0f) != 0;
            } else {
                shared_rc =
                    ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfgs[slot].arena,
                                                         &shared_gate_v,
                                                         ffn_inputs[slot],
                                                         shared_gate_t) != 0 ||
                    ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfgs[slot].arena,
                                                         &shared_up_v,
                                                         ffn_inputs[slot],
                                                         shared_up_t) != 0 ||
                    !ds4_gpu_swiglu_tensor(shared_mid_t,
                                          shared_gate_t,
                                          shared_up_t,
                                          mid,
                                          10.0f,
                                          1.0f);
            }
            if (!shared_rc) {
                if (use_shared_down_add) {
                    shared_rc =
                        ds4_gpu_arena_f8_e4m3_b128_matmul_add_f32(cfgs[slot].arena,
                                                                  &shared_down_v,
                                                                  shared_mid_t,
                                                                  routed_view,
                                                                  ffn_deltas[slot]) != 0;
                } else {
                    shared_rc =
                        ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfgs[slot].arena,
                                                             &shared_down_v,
                                                             shared_mid_t,
                                                             shared_t) != 0 ||
                        !ds4_gpu_add_tensor(ffn_deltas[slot], routed_view, shared_t, hidden);
                }
            }
            if (!use_scratch) ds4_gpu_tensor_free(routed_view);
            if (shared_rc) {
                exec_error(err, errlen, "shared FFN batch slot %u failed", slot);
                goto done;
            }
        }
    }

fill_reports:
    if (reports) {
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            ds4_v100_layer_execute_report *report = &reports[slot];
            memset(report, 0, sizeof(*report));
            report->routes = routes;
            report->turbomind_routed = state->has_turbomind_routed ? 1u : 0u;
            report->turbomind_tp2_routed = use_tp2 ? 1u : 0u;
            if (readback_routes) {
                for (uint32_t i = 0; i < routes && i < 6u; i++) {
                    report->selected_experts[i] = selected[(uint64_t)slot * routes + i];
                    report->route_weights[i] = weights[(uint64_t)slot * routes + i];
                }
            }
        }
    }

    rc = 0;

done:
    if (!use_scratch) {
        ds4_gpu_tensor_free(shared_batch_t);
        ds4_gpu_tensor_free(shared_mid_batch_t);
        ds4_gpu_tensor_free(shared_t);
        ds4_gpu_tensor_free(shared_mid_t);
        ds4_gpu_tensor_free(shared_up_t);
        ds4_gpu_tensor_free(shared_gate_t);
        ds4_gpu_tensor_free(routed_out_t);
        ds4_gpu_tensor_free(routed_mid_t);
        ds4_gpu_tensor_free(input_ptrs_t);
        ds4_gpu_tensor_free(tokens_t);
        ds4_gpu_tensor_free(weights_t);
        ds4_gpu_tensor_free(selected_t);
        ds4_gpu_tensor_free(probs_t);
        ds4_gpu_tensor_free(router_t);
    }
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
                                        model_offset_for_binding(cfg, &state->ffn_norm),
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

    int rc = 1;
    const bool use_scratch =
        cfg->batch_scratch != NULL && single_slot_batch_scratch_enabled();
    ds4_gpu_tensor *hc_norm = NULL;
    ds4_gpu_tensor *hc_mix = NULL;
    ds4_gpu_tensor *attn_split = NULL;
    ds4_gpu_tensor *ffn_split = NULL;
    ds4_gpu_tensor *attn_cur = NULL;
    ds4_gpu_tensor *attn_out = NULL;
    ds4_gpu_tensor *after_attn_hc = NULL;
    ds4_gpu_tensor *ffn_cur = NULL;
    ds4_gpu_tensor *ffn_norm = NULL;
    ds4_gpu_tensor *ffn_delta = NULL;
    if (use_scratch &&
        ensure_batch_scratch(cfg->batch_scratch,
                             hidden_n,
                             state->intermediate_size,
                             state->routes_per_token,
                             state->q_lora_rank,
                             state->q_width,
                             state->kv_latent_width,
                             state->attention_output_rank,
                             err,
                             errlen)) {
        goto done;
    }
    if (use_scratch) {
        hc_norm = cfg->batch_scratch->hc_norm[0];
        hc_mix = cfg->batch_scratch->hc_mix[0];
        attn_split = cfg->batch_scratch->attn_split[0];
        ffn_split = cfg->batch_scratch->ffn_split[0];
        attn_cur = cfg->batch_scratch->attn_cur[0];
        attn_out = cfg->batch_scratch->attn_out[0];
        after_attn_hc = cfg->batch_scratch->after_attn_hc[0];
        ffn_cur = cfg->batch_scratch->ffn_cur[0];
        ffn_norm = cfg->batch_scratch->ffn_norm[0];
        ffn_delta = cfg->batch_scratch->ffn_delta[0];
    } else {
        hc_norm = ds4_gpu_tensor_alloc(hc_bytes);
        hc_mix = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
        attn_split = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
        ffn_split = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
        attn_cur = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
        attn_out = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
        after_attn_hc = ds4_gpu_tensor_alloc(hc_bytes);
        ffn_cur = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
        ffn_norm = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
        ffn_delta = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
    }
    if (!hc_norm || !hc_mix || !attn_split || !ffn_split || !attn_cur ||
        !attn_out || !after_attn_hc || !ffn_cur || !ffn_norm || !ffn_delta) {
        exec_error(err, errlen, "failed to allocate HC layer executor tensors");
        goto done;
    }

    if (!ds4_gpu_rms_norm_plain_tensor(hc_norm, hidden_hc, (uint32_t)hc_values, DS4_V100_RMS_EPS) ||
        !ds4_gpu_matmul_f32_tensor(hc_mix,
                                   cfg->model_map,
                                   cfg->model_size,
                                   model_offset_for_binding(cfg, &state->hc_attn_fn),
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
                                              model_offset_for_binding(cfg, &state->hc_attn_scale),
                                              model_offset_for_binding(cfg, &state->hc_attn_base),
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
                                   model_offset_for_binding(cfg, &state->hc_ffn_fn),
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
                                              model_offset_for_binding(cfg, &state->hc_ffn_scale),
                                              model_offset_for_binding(cfg, &state->hc_ffn_base),
                                              hidden_n,
                                              DS4_V100_N_HC,
                                              DS4_V100_HC_SINKHORN_ITERS,
                                              DS4_V100_RMS_EPS) ||
        !ds4_gpu_rms_norm_weight_tensor(ffn_norm,
                                        ffn_cur,
                                        cfg->model_map,
                                        cfg->model_size,
                                        model_offset_for_binding(cfg, &state->ffn_norm),
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
    if (!use_scratch) {
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
    }
    return rc;
}

int ds4_v100_layer_execute_hc_decode_batch(
        const ds4_v100_layer_state           *state,
        const ds4_v100_layer_execute_config  *cfgs,
        const ds4_gpu_tensor *const          *hidden_hc,
        ds4_gpu_tensor *const                *next_hidden_hc,
        uint32_t                              n_slots,
        ds4_v100_layer_execute_report        *reports,
        char                                 *err,
        size_t                                errlen) {
    if (!state || !cfgs || !hidden_hc || !next_hidden_hc ||
        n_slots == 0 || n_slots > DS4_V100_LAYER_MAX_BATCH) {
        return exec_error(err, errlen, "invalid HC batch inputs");
    }
    if (n_slots == 1) {
        return ds4_v100_layer_execute_hc_decode(state,
                                                &cfgs[0],
                                                hidden_hc[0],
                                                next_hidden_hc[0],
                                                reports ? &reports[0] : NULL,
                                                err,
                                                errlen);
    }
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (cfgs[slot].checkpoint_fn) {
            for (uint32_t i = 0; i < n_slots; i++) {
                if (ds4_v100_layer_execute_hc_decode(state,
                                                     &cfgs[i],
                                                     hidden_hc[i],
                                                     next_hidden_hc[i],
                                                     reports ? &reports[i] : NULL,
                                                     err,
                                                     errlen)) {
                    return 1;
                }
            }
            return 0;
        }
    }

    const uint32_t hidden_n = state->hidden_size;
    const uint64_t hc_values = (uint64_t)DS4_V100_N_HC * hidden_n;
    const uint64_t hc_bytes = hc_values * sizeof(float);
    if (hidden_n != DS4_V100_OUT_GROUP_DIM ||
        state->hc_attn_fn.n_shape_dims != 2 ||
        state->hc_ffn_fn.n_shape_dims != 2 ||
        state->hc_attn_fn.shape[0] != hc_values ||
        state->hc_ffn_fn.shape[0] != hc_values ||
        state->hc_attn_fn.shape[1] != DS4_V100_HC_MIX ||
        state->hc_ffn_fn.shape[1] != DS4_V100_HC_MIX) {
        return exec_error(err, errlen, "HC batch control dimensions do not match DS4");
    }

    ds4_gpu_tensor *hc_norm[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *hc_mix[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *attn_split[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *ffn_split[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *attn_cur[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *attn_out[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *after_attn_hc[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *ffn_cur[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *ffn_norm[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *ffn_delta[DS4_V100_LAYER_MAX_BATCH] = {0};

    int rc = 1;
    const bool use_scratch = cfgs[0].batch_scratch != NULL;
    if (use_scratch &&
        ensure_batch_scratch(cfgs[0].batch_scratch,
                             hidden_n,
                             state->intermediate_size,
                             state->routes_per_token,
                             state->q_lora_rank,
                             state->q_width,
                             state->kv_latent_width,
                             state->attention_output_rank,
                             err,
                             errlen)) {
        goto done;
    }
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!hidden_hc[slot] || !next_hidden_hc[slot] ||
            ds4_gpu_tensor_bytes(hidden_hc[slot]) < hc_bytes ||
            ds4_gpu_tensor_bytes(next_hidden_hc[slot]) < hc_bytes ||
            validate_execute_common(state, &cfgs[slot], err, errlen)) {
            goto done;
        }
        if (cfgs[slot].arena != cfgs[0].arena ||
            cfgs[slot].model_map != cfgs[0].model_map ||
            cfgs[slot].model_size != cfgs[0].model_size ||
            cfgs[slot].batch_scratch != cfgs[0].batch_scratch) {
            exec_error(err, errlen, "HC batch cfgs must share arena/model");
            goto done;
        }
        if (use_scratch) {
            hc_norm[slot] = cfgs[0].batch_scratch->hc_norm[slot];
            hc_mix[slot] = cfgs[0].batch_scratch->hc_mix[slot];
            attn_split[slot] = cfgs[0].batch_scratch->attn_split[slot];
            ffn_split[slot] = cfgs[0].batch_scratch->ffn_split[slot];
            attn_cur[slot] = cfgs[0].batch_scratch->attn_cur[slot];
            attn_out[slot] = cfgs[0].batch_scratch->attn_out[slot];
            after_attn_hc[slot] = cfgs[0].batch_scratch->after_attn_hc[slot];
            ffn_cur[slot] = cfgs[0].batch_scratch->ffn_cur[slot];
            ffn_norm[slot] = cfgs[0].batch_scratch->ffn_norm[slot];
            ffn_delta[slot] = cfgs[0].batch_scratch->ffn_delta[slot];
        } else {
            hc_norm[slot] = ds4_gpu_tensor_alloc(hc_bytes);
            hc_mix[slot] = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
            attn_split[slot] = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
            ffn_split[slot] = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
            attn_cur[slot] = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
            attn_out[slot] = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
            after_attn_hc[slot] = ds4_gpu_tensor_alloc(hc_bytes);
            ffn_cur[slot] = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
            ffn_norm[slot] = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
            ffn_delta[slot] = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
        }
        if (!hc_norm[slot] || !hc_mix[slot] || !attn_split[slot] || !ffn_split[slot] ||
            !attn_cur[slot] || !attn_out[slot] || !after_attn_hc[slot] ||
            !ffn_cur[slot] || !ffn_norm[slot] || !ffn_delta[slot]) {
            exec_error(err, errlen, "failed to allocate HC batch tensors");
            goto done;
        }
    }

    const bool profile = env_flag_enabled("DS4_V100_PROFILE_DECODE");
    double profile_last_ms = 0.0;
    double timing_hc_attn_ms = 0.0;
    double timing_attention_ms = 0.0;
    double timing_hc_ffn_ms = 0.0;
    double timing_ffn_ms = 0.0;
    double timing_hc_final_ms = 0.0;
    if (profile) {
        if (!ds4_gpu_synchronize()) {
            exec_error(err, errlen, "HC batch profile start sync failed");
            goto done;
        }
        profile_last_ms = monotonic_ms();
    }

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!ds4_gpu_rms_norm_plain_tensor(hc_norm[slot],
                                           hidden_hc[slot],
                                           (uint32_t)hc_values,
                                           DS4_V100_RMS_EPS) ||
            !ds4_gpu_matmul_f32_tensor(hc_mix[slot],
                                       cfgs[slot].model_map,
                                       cfgs[slot].model_size,
                                       model_offset_for_binding(&cfgs[slot], &state->hc_attn_fn),
                                       hc_values,
                                       DS4_V100_HC_MIX,
                                       hc_norm[slot],
                                       1) ||
            !ds4_gpu_hc_split_weighted_sum_tensor(attn_cur[slot],
                                                  attn_split[slot],
                                                  hc_mix[slot],
                                                  hidden_hc[slot],
                                                  cfgs[slot].model_map,
                                                  cfgs[slot].model_size,
                                                  model_offset_for_binding(&cfgs[slot], &state->hc_attn_scale),
                                                  model_offset_for_binding(&cfgs[slot], &state->hc_attn_base),
                                                  hidden_n,
                                                  DS4_V100_N_HC,
                                                  DS4_V100_HC_SINKHORN_ITERS,
                                                  DS4_V100_RMS_EPS)) {
            if (err && err[0] == '\0') exec_error(err, errlen, "HC batch attention prep failed");
            goto done;
        }
        if (profile && !profile_mark(&profile_last_ms, &timing_hc_attn_ms)) {
            exec_error(err, errlen, "HC batch attention prep profile sync failed");
            goto done;
        }
    }

    if (execute_attention_output_batch(state,
                                       cfgs,
                                       (const ds4_gpu_tensor *const *)attn_cur,
                                       attn_out,
                                       n_slots,
                                       err,
                                       errlen)) {
        goto done;
    }
    if (profile && !profile_mark(&profile_last_ms, &timing_attention_ms)) {
        exec_error(err, errlen, "HC batch attention profile sync failed");
        goto done;
    }

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!ds4_gpu_hc_expand_split_tensor(after_attn_hc[slot],
                                            attn_out[slot],
                                            hidden_hc[slot],
                                            attn_split[slot],
                                            hidden_n,
                                            DS4_V100_N_HC) ||
            !ds4_gpu_rms_norm_plain_tensor(hc_norm[slot],
                                           after_attn_hc[slot],
                                           (uint32_t)hc_values,
                                           DS4_V100_RMS_EPS) ||
            !ds4_gpu_matmul_f32_tensor(hc_mix[slot],
                                       cfgs[slot].model_map,
                                       cfgs[slot].model_size,
                                       model_offset_for_binding(&cfgs[slot], &state->hc_ffn_fn),
                                       hc_values,
                                       DS4_V100_HC_MIX,
                                       hc_norm[slot],
                                       1) ||
            !ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur[slot],
                                                  ffn_split[slot],
                                                  hc_mix[slot],
                                                  after_attn_hc[slot],
                                                  cfgs[slot].model_map,
                                                  cfgs[slot].model_size,
                                                  model_offset_for_binding(&cfgs[slot], &state->hc_ffn_scale),
                                                  model_offset_for_binding(&cfgs[slot], &state->hc_ffn_base),
                                                  hidden_n,
                                                  DS4_V100_N_HC,
                                                  DS4_V100_HC_SINKHORN_ITERS,
                                                  DS4_V100_RMS_EPS) ||
            !ds4_gpu_rms_norm_weight_tensor(ffn_norm[slot],
                                            ffn_cur[slot],
                                            cfgs[slot].model_map,
                                            cfgs[slot].model_size,
                                            model_offset_for_binding(&cfgs[slot], &state->ffn_norm),
                                            hidden_n,
                                            DS4_V100_RMS_EPS)) {
            if (err && err[0] == '\0') exec_error(err, errlen, "HC batch FFN prep failed");
            goto done;
        }
        if (profile && !profile_mark(&profile_last_ms, &timing_hc_ffn_ms)) {
            exec_error(err, errlen, "HC batch FFN prep profile sync failed");
            goto done;
        }
    }

    if (execute_ffn_delta_batch(state,
                                cfgs,
                                (const ds4_gpu_tensor *const *)ffn_norm,
                                ffn_delta,
                                n_slots,
                                reports,
                                err,
                                errlen)) {
        goto done;
    }
    if (profile && !profile_mark(&profile_last_ms, &timing_ffn_ms)) {
        exec_error(err, errlen, "HC batch FFN profile sync failed");
        goto done;
    }

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!ds4_gpu_hc_expand_split_tensor(next_hidden_hc[slot],
                                            ffn_delta[slot],
                                            after_attn_hc[slot],
                                            ffn_split[slot],
                                            hidden_n,
                                            DS4_V100_N_HC)) {
            exec_error(err, errlen, "HC batch final expansion failed");
            goto done;
        }
    }
    if (profile && !profile_mark(&profile_last_ms, &timing_hc_final_ms)) {
        exec_error(err, errlen, "HC batch final profile sync failed");
        goto done;
    }
    if (reports && profile) {
        const double timing_total_ms = timing_hc_attn_ms +
                                       timing_attention_ms +
                                       timing_hc_ffn_ms +
                                       timing_ffn_ms +
                                       timing_hc_final_ms;
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            reports[slot].timing_hc_attn_ms = timing_hc_attn_ms;
            reports[slot].timing_attention_ms = timing_attention_ms;
            reports[slot].timing_hc_ffn_ms = timing_hc_ffn_ms;
            reports[slot].timing_ffn_ms = timing_ffn_ms;
            reports[slot].timing_hc_final_ms = timing_hc_final_ms;
            reports[slot].timing_total_ms = timing_total_ms;
        }
    }

    rc = 0;

done:
    if (!use_scratch) {
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            ds4_gpu_tensor_free(ffn_delta[slot]);
            ds4_gpu_tensor_free(ffn_norm[slot]);
            ds4_gpu_tensor_free(ffn_cur[slot]);
            ds4_gpu_tensor_free(after_attn_hc[slot]);
            ds4_gpu_tensor_free(attn_out[slot]);
            ds4_gpu_tensor_free(attn_cur[slot]);
            ds4_gpu_tensor_free(ffn_split[slot]);
            ds4_gpu_tensor_free(attn_split[slot]);
            ds4_gpu_tensor_free(hc_mix[slot]);
            ds4_gpu_tensor_free(hc_norm[slot]);
        }
    }
    return rc;
}
