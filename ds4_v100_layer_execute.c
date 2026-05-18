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
    return 0;
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

    const ds4_gpu_tensor *raw_kv = cfg->raw_kv ? cfg->raw_kv : kv;
    if (!ds4_gpu_attention_decode_heads_tensor(heads,
                                               cfg->model_map,
                                               cfg->model_size,
                                               state->attn_sinks.source_offset,
                                               q,
                                               raw_kv,
                                               n_raw,
                                               raw_cap,
                                               raw_start,
                                               cfg->compressed_kv,
                                               cfg->n_compressed,
                                               cfg->compressed_mask,
                                               cfg->use_compressed_mask ? 1u : 0u,
                                               DS4_V100_N_HEAD,
                                               DS4_V100_HEAD_DIM)) {
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
    if (state->router_kind != DS4_V100_ROUTER_HASH || !state->has_hash_router) {
        return exec_error(err, errlen, "Sprint 019 layer executor currently supports hash-router layers only");
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
    ds4_gpu_tensor *gate_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *up_t = ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
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
        !gate_t || !up_t || !mid_t || !route_t || !accum_a || !accum_b ||
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
                                      0,
                                      state->router_hash.source_offset,
                                      (uint32_t)state->router_hash.shape[1],
                                      cfg->router_token,
                                      0,
                                      0,
                                      false,
                                      true,
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
        if (ds4_gpu_arena_mxfp4_matmul_f32(cfg->arena, &gate_v, ffn_input, gate_t) != 0 ||
            ds4_gpu_arena_mxfp4_matmul_f32(cfg->arena, &up_v, ffn_input, up_t) != 0 ||
            !ds4_gpu_swiglu_tensor(mid_t, gate_t, up_t, mid, 10.0f, weights[route]) ||
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
    ds4_gpu_tensor_free(up_t);
    ds4_gpu_tensor_free(gate_t);
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
