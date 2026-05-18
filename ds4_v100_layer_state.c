#include "ds4_v100_layer_state.h"
#include "ds4_source_formats.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static int state_error(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return 1;
}

const char *ds4_v100_router_kind_name(ds4_v100_router_kind kind) {
    switch (kind) {
    case DS4_V100_ROUTER_HASH: return "hash";
    case DS4_V100_ROUTER_BIAS: return "bias";
    case DS4_V100_ROUTER_UNKNOWN:
    default: return "unknown";
    }
}

static int dtype_is(const ds4_v100_tensor_binding *b, const char *dtype) {
    return b && b->source_dtype && strcmp(b->source_dtype, dtype) == 0;
}

static int bind_required(const ds4_v100_context *ctx,
                         int layer_id,
                         const char *suffix,
                         ds4_v100_tensor_binding *out,
                         char *err,
                         size_t errlen) {
    if (ds4_v100_context_require_layer_tensor_binding(ctx, layer_id, suffix, out, err, errlen)) {
        return 1;
    }
    return 0;
}

static int bind_optional(const ds4_v100_context *ctx,
                         int layer_id,
                         const char *suffix,
                         ds4_v100_tensor_binding *out) {
    char err[256];
    memset(out, 0, sizeof(*out));
    return ds4_v100_context_require_layer_tensor_binding(
        ctx, layer_id, suffix, out, err, sizeof(err)) == 0;
}

static int make_f32_matrix(const ds4_v100_tensor_binding *b,
                           ds4_v100_bound_matrix *out,
                           const char *label,
                           char *err,
                           size_t errlen) {
    if (!b || !out || b->n_shape_dims != 2 || !dtype_is(b, "f32")) {
        return state_error(err, errlen, "%s must be a 2D f32 tensor", label);
    }
    ds4_v100_tensor_binding binding = *b;
    memset(out, 0, sizeof(*out));
    out->binding = binding;
    out->cols = (uint32_t)binding.shape[0];
    out->rows = (uint32_t)binding.shape[1];
    out->planes = 1;
    out->row_bytes = (uint64_t)out->cols * sizeof(float);
    out->bytes = (uint64_t)out->rows * out->row_bytes;
    if (out->bytes != binding.byte_length) {
        return state_error(err, errlen,
                           "%s byte length mismatch: expected %" PRIu64 " got %" PRIu64,
                           label,
                           out->bytes,
                           binding.byte_length);
    }
    return 0;
}

static int make_f8_matrix(const ds4_v100_tensor_binding *b,
                          ds4_v100_bound_matrix *out,
                          const char *label,
                          char *err,
                          size_t errlen) {
    if (!b || !out || b->n_shape_dims != 2 || !dtype_is(b, "f8_e4m3_b128")) {
        return state_error(err, errlen, "%s must be a 2D f8_e4m3_b128 tensor", label);
    }
    ds4_v100_tensor_binding binding = *b;
    memset(out, 0, sizeof(*out));
    out->binding = binding;
    out->cols = (uint32_t)binding.shape[0];
    out->rows = (uint32_t)binding.shape[1];
    out->planes = 1;
    out->row_bytes = ds4_src_f8_e4m3_b128_row_bytes(out->cols);
    out->bytes = (uint64_t)out->rows * out->row_bytes;
    if (out->bytes != binding.byte_length) {
        return state_error(err, errlen,
                           "%s byte length mismatch: expected %" PRIu64 " got %" PRIu64,
                           label,
                           out->bytes,
                           binding.byte_length);
    }
    return 0;
}

static int make_mxfp4_expert_matrix(const ds4_v100_tensor_binding *b,
                                    uint32_t expert,
                                    ds4_v100_bound_matrix *out,
                                    const char *label,
                                    char *err,
                                    size_t errlen) {
    if (!b || !out || b->n_shape_dims != 3 || !dtype_is(b, "mxfp4")) {
        return state_error(err, errlen, "%s must be a 3D mxfp4 expert tensor", label);
    }
    ds4_v100_tensor_binding binding = *b;
    if (expert >= binding.shape[2]) {
        return state_error(err, errlen,
                           "%s expert %u outside expert count %" PRIu64,
                           label,
                           expert,
                           binding.shape[2]);
    }
    memset(out, 0, sizeof(*out));
    out->binding = binding;
    out->cols = (uint32_t)binding.shape[0];
    out->rows = (uint32_t)binding.shape[1];
    out->planes = (uint32_t)binding.shape[2];
    out->row_bytes = ds4_src_mxfp4_row_bytes(out->cols);
    out->bytes = (uint64_t)out->rows * out->row_bytes;
    out->rel = (uint64_t)expert * out->bytes;
    if (out->rel > binding.byte_length || out->bytes > binding.byte_length - out->rel) {
        return state_error(err, errlen, "%s expert span overflows binding", label);
    }
    return 0;
}

static int check_same_owner(const ds4_v100_layer_state *state,
                            const ds4_v100_tensor_binding *b,
                            const char *label,
                            char *err,
                            size_t errlen) {
    if (!b || b->owning_gpu != state->owning_gpu) {
        return state_error(err, errlen,
                           "%s owner mismatch: got gpu%d expected gpu%d",
                           label,
                           b ? b->owning_gpu : -1,
                           state->owning_gpu);
    }
    return 0;
}

static int validate_router_binding(ds4_v100_layer_state *state,
                                   char *err,
                                   size_t errlen) {
    if (state->router.rows != 256 || state->router.cols == 0) {
        return state_error(err, errlen,
                           "router dimensions must be [hidden x 256], got rows=%u cols=%u",
                           state->router.rows,
                           state->router.cols);
    }
    if (state->has_hash_router) {
        if (!dtype_is(&state->router_hash, "i32") ||
            state->router_hash.n_shape_dims != 2 ||
            state->router_hash.shape[0] != 6) {
            return state_error(err, errlen, "hash router must be i32 [6 x tokens]");
        }
        state->router_kind = DS4_V100_ROUTER_HASH;
        state->routes_per_token = 6;
        state->router_token_capacity = (uint32_t)state->router_hash.shape[1];
        return 0;
    }
    if (state->has_bias_router) {
        if (!dtype_is(&state->router_bias, "f32") ||
            state->router_bias.n_shape_dims != 1 ||
            state->router_bias.shape[0] != 256) {
            return state_error(err, errlen, "bias router must be f32 [256]");
        }
        state->router_kind = DS4_V100_ROUTER_BIAS;
        state->routes_per_token = 6;
        state->router_token_capacity = 0;
        return 0;
    }
    return state_error(err, errlen, "layer has neither hash nor bias router metadata");
}

int ds4_v100_layer_state_init(ds4_v100_layer_state *out,
                              const ds4_v100_context *ctx,
                              int layer_id,
                              char *err,
                              size_t errlen) {
    if (!out) return state_error(err, errlen, "missing layer state output");
    memset(out, 0, sizeof(*out));
    if (!ctx) return state_error(err, errlen, "missing V100 context");

    const ds4_v100_layer_info *li = ds4_v100_context_layer(ctx, layer_id);
    if (!li) return state_error(err, errlen, "invalid layer %d", layer_id);
    const ds4_v100_stage_info *stage = ds4_v100_context_stage(ctx, li->stage_id);
    if (!stage) return state_error(err, errlen, "missing stage for layer %d", layer_id);

    out->layer_id = layer_id;
    out->stage_id = li->stage_id;
    out->owning_gpu = stage->gpu;
    out->layer_class = li->layer_class;
    out->kv_view = li->kv_view;

    if (bind_required(ctx, layer_id, "ffn_gate_exps.weight", &out->routed_gate_binding, err, errlen) ||
        bind_required(ctx, layer_id, "ffn_up_exps.weight", &out->routed_up_binding, err, errlen) ||
        bind_required(ctx, layer_id, "ffn_down_exps.weight", &out->routed_down_binding, err, errlen) ||
        bind_required(ctx, layer_id, "ffn_gate_shexp.weight", &out->shared_gate.binding, err, errlen) ||
        bind_required(ctx, layer_id, "ffn_up_shexp.weight", &out->shared_up.binding, err, errlen) ||
        bind_required(ctx, layer_id, "ffn_down_shexp.weight", &out->shared_down.binding, err, errlen) ||
        bind_required(ctx, layer_id, "ffn_gate_inp.weight", &out->router.binding, err, errlen) ||
        bind_required(ctx, layer_id, "ffn_norm.weight", &out->ffn_norm, err, errlen) ||
        bind_required(ctx, layer_id, "hc_ffn_fn", &out->hc_ffn_fn, err, errlen) ||
        bind_required(ctx, layer_id, "hc_ffn_base", &out->hc_ffn_base, err, errlen) ||
        bind_required(ctx, layer_id, "hc_ffn_scale", &out->hc_ffn_scale, err, errlen)) {
        return 1;
    }

    out->has_hash_router = bind_optional(ctx, layer_id, "ffn_gate_tid2eid", &out->router_hash);
    out->has_bias_router = bind_optional(ctx, layer_id, "exp_probs_b", &out->router_bias);

    if (make_f32_matrix(&out->router.binding, &out->router, "ffn_gate_inp.weight", err, errlen) ||
        make_f8_matrix(&out->shared_gate.binding, &out->shared_gate, "ffn_gate_shexp.weight", err, errlen) ||
        make_f8_matrix(&out->shared_up.binding, &out->shared_up, "ffn_up_shexp.weight", err, errlen) ||
        make_f8_matrix(&out->shared_down.binding, &out->shared_down, "ffn_down_shexp.weight", err, errlen) ||
        validate_router_binding(out, err, errlen)) {
        return 1;
    }

    const ds4_v100_tensor_binding *owned[] = {
        &out->routed_gate_binding,
        &out->routed_up_binding,
        &out->routed_down_binding,
        &out->shared_gate.binding,
        &out->shared_up.binding,
        &out->shared_down.binding,
        &out->router.binding,
        &out->ffn_norm,
        &out->hc_ffn_fn,
        &out->hc_ffn_base,
        &out->hc_ffn_scale,
    };
    const char *labels[] = {
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
        "ffn_gate_inp.weight",
        "ffn_norm.weight",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
    };
    for (uint32_t i = 0; i < sizeof(owned) / sizeof(owned[0]); i++) {
        if (check_same_owner(out, owned[i], labels[i], err, errlen)) return 1;
    }

    if (out->routed_gate_binding.n_shape_dims != 3 ||
        out->routed_up_binding.n_shape_dims != 3 ||
        out->routed_down_binding.n_shape_dims != 3) {
        return state_error(err, errlen, "routed expert tensors must be 3D");
    }
    out->hidden_size = out->router.cols;
    out->intermediate_size = out->shared_gate.rows;
    out->routed_experts = (uint32_t)out->routed_gate_binding.shape[2];
    if (out->routed_up_binding.shape[2] != out->routed_experts ||
        out->routed_down_binding.shape[2] != out->routed_experts) {
        return state_error(err, errlen, "routed expert counts do not match");
    }
    if (out->routed_gate_binding.shape[0] != out->hidden_size ||
        out->routed_up_binding.shape[0] != out->hidden_size ||
        out->routed_gate_binding.shape[1] != out->intermediate_size ||
        out->routed_up_binding.shape[1] != out->intermediate_size ||
        out->routed_down_binding.shape[0] != out->intermediate_size ||
        out->routed_down_binding.shape[1] != out->hidden_size) {
        return state_error(err, errlen, "routed expert dimensions do not match router/shared dims");
    }
    if (out->shared_up.cols != out->hidden_size ||
        out->shared_up.rows != out->intermediate_size ||
        out->shared_down.cols != out->intermediate_size ||
        out->shared_down.rows != out->hidden_size) {
        return state_error(err, errlen, "shared expert dimensions do not match router dims");
    }
    return 0;
}

int ds4_v100_layer_state_route_matrices(const ds4_v100_layer_state *state,
                                        uint32_t expert,
                                        ds4_v100_route_matrices *out,
                                        char *err,
                                        size_t errlen) {
    if (!state || !out) return state_error(err, errlen, "missing route matrix output");
    memset(out, 0, sizeof(*out));
    if (expert >= state->routed_experts) {
        return state_error(err, errlen,
                           "expert %u outside routed expert count %u",
                           expert,
                           state->routed_experts);
    }
    if (make_mxfp4_expert_matrix(&state->routed_gate_binding, expert, &out->gate,
                                 "ffn_gate_exps.weight", err, errlen) ||
        make_mxfp4_expert_matrix(&state->routed_up_binding, expert, &out->up,
                                 "ffn_up_exps.weight", err, errlen) ||
        make_mxfp4_expert_matrix(&state->routed_down_binding, expert, &out->down,
                                 "ffn_down_exps.weight", err, errlen)) {
        return 1;
    }
    return 0;
}

uint64_t ds4_v100_bound_matrix_arena_offset(const ds4_v100_bound_matrix *matrix) {
    if (!matrix) return 0;
    return matrix->binding.shard_offset + matrix->rel;
}

static int grow_span_for_matrix(const ds4_v100_bound_matrix *matrix, uint64_t *span) {
    const uint64_t start = ds4_v100_bound_matrix_arena_offset(matrix);
    if (matrix->bytes > UINT64_MAX - start) return 1;
    const uint64_t end = start + matrix->bytes;
    if (end > *span) *span = end;
    return 0;
}

int ds4_v100_layer_state_ffn_arena_span(const ds4_v100_layer_state *state,
                                        const int32_t *selected_experts,
                                        uint32_t n_selected,
                                        uint64_t *out_bytes,
                                        char *err,
                                        size_t errlen) {
    if (!state || !out_bytes) return state_error(err, errlen, "missing arena span output");
    uint64_t span = 0;
    if (grow_span_for_matrix(&state->router, &span) ||
        grow_span_for_matrix(&state->shared_gate, &span) ||
        grow_span_for_matrix(&state->shared_up, &span) ||
        grow_span_for_matrix(&state->shared_down, &span)) {
        return state_error(err, errlen, "fixed FFN arena span overflow");
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        if (!selected_experts) return state_error(err, errlen, "missing selected experts");
        if (selected_experts[i] < 0 || (uint32_t)selected_experts[i] >= state->routed_experts) {
            return state_error(err, errlen, "selected expert %d outside routed expert count",
                               selected_experts[i]);
        }
        ds4_v100_route_matrices route;
        if (ds4_v100_layer_state_route_matrices(state,
                                                (uint32_t)selected_experts[i],
                                                &route,
                                                err,
                                                errlen)) {
            return 1;
        }
        if (grow_span_for_matrix(&route.gate, &span) ||
            grow_span_for_matrix(&route.up, &span) ||
            grow_span_for_matrix(&route.down, &span)) {
            return state_error(err, errlen, "routed FFN arena span overflow");
        }
    }
    *out_bytes = span;
    return 0;
}

int ds4_v100_bound_matrix_source_view(const ds4_v100_bound_matrix *matrix,
                                      ds4_gpu_source_row_view *out,
                                      char *err,
                                      size_t errlen) {
    if (!matrix || !out) return state_error(err, errlen, "missing source row view output");
    if (matrix->rows == 0 || matrix->cols == 0 || matrix->row_bytes == 0 || matrix->bytes == 0) {
        return state_error(err, errlen, "invalid bound matrix dimensions");
    }
    memset(out, 0, sizeof(*out));
    out->arena_offset = ds4_v100_bound_matrix_arena_offset(matrix);
    out->byte_length = matrix->bytes;
    out->rows = matrix->rows;
    out->cols = matrix->cols;
    out->row_stride_bytes = (uint32_t)matrix->row_bytes;
    return 0;
}
