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

static int make_bf16_matrix(const ds4_v100_tensor_binding *b,
                            ds4_v100_bound_matrix *out,
                            const char *label,
                            char *err,
                            size_t errlen) {
    if (!b || !out || b->n_shape_dims != 2 || !dtype_is(b, "bf16")) {
        return state_error(err, errlen, "%s must be a 2D bf16 tensor", label);
    }
    ds4_v100_tensor_binding binding = *b;
    memset(out, 0, sizeof(*out));
    out->binding = binding;
    out->cols = (uint32_t)binding.shape[0];
    out->rows = (uint32_t)binding.shape[1];
    out->planes = 1;
    out->row_bytes = (uint64_t)out->cols * sizeof(uint16_t);
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

static ds4_gpu_turbomind_mxfp4_matrix_view tm_gpu_view(
        const ds4_v100_turbomind_binding *b) {
    ds4_gpu_turbomind_mxfp4_matrix_view v;
    memset(&v, 0, sizeof(v));
    if (!b) return v;
    v.n = b->n;
    v.k = b->k;
    v.weight_offset = b->weight_offset;
    v.scale_offset = b->scale_offset;
    v.weight_bytes_per_expert = b->weight_bytes_per_expert;
    v.scale_bytes_per_expert = b->scale_bytes_per_expert;
    v.k_pack = b->k_pack;
    v.weight_stride = b->weight_stride;
    v.scale_stride = b->scale_stride;
    v.experts_packed = b->experts_packed;
    v.experts_total = b->experts_total;
    if (b->runtime_layout &&
        !strcmp(b->runtime_layout, "turbomind_mxfp4_grouped_gate_up_interleaved")) {
        v.flags |= DS4_GPU_TURBOMIND_MXFP4_GATE_UP_INTERLEAVED;
    }
    return v;
}

static int make_turbomind_routed_binding(const ds4_v100_turbomind_binding *b,
                                         ds4_v100_tensor_binding *out,
                                         const char *label,
                                         char *err,
                                         size_t errlen) {
    if (!b || !out || b->n_shape_dims != 3 ||
        !b->source_dtype || strcmp(b->source_dtype, "mxfp4") != 0) {
        return state_error(err, errlen, "%s must be a 3D TurboMind MXFP4 tensor", label);
    }
    if (b->experts_packed < b->experts_total) {
        return state_error(err, errlen,
                           "%s TurboMind binding packs only %u/%u experts",
                           label,
                           b->experts_packed,
                           b->experts_total);
    }
    memset(out, 0, sizeof(*out));
    out->semantic_tensor_id = b->semantic_tensor_id;
    out->source_name = b->source_name;
    out->source_dtype = b->source_dtype;
    out->source_shape = b->source_shape;
    out->runtime_layout = b->runtime_layout;
    out->kernel_family = b->kernel_family;
    out->shard_file = b->shard_file;
    out->owning_gpu = b->owning_gpu;
    out->layer_id = b->layer_id;
    out->scale_offset = -1;
    out->source_offset = 0;
    out->byte_length = b->source_byte_length;
    out->shard_offset = b->source_shard_offset;
    out->policy = b->policy;
    out->n_shape_dims = b->n_shape_dims;
    for (uint32_t i = 0; i < b->n_shape_dims && i < DS4_V100_MAX_SHAPE_DIMS; i++) {
        out->shape[i] = b->shape[i];
    }
    return 0;
}

static int make_turbomind_fused_gate_up_synthetic_binding(
        const ds4_v100_turbomind_binding *b,
        ds4_v100_tensor_binding *out,
        const char *label,
        char *err,
        size_t errlen) {
    if (make_turbomind_routed_binding(b, out, label, err, errlen)) return 1;
    if (b->n == 0 || (b->n % 2u) != 0 || b->n_shape_dims != 3 ||
        b->shape[1] != b->n || b->shape[0] != b->k) {
        return state_error(err, errlen, "%s must be a fused gate_up tensor with even N", label);
    }
    out->shape[1] = (uint64_t)b->n / 2u;
    out->byte_length = b->source_byte_length / 2u;
    return 0;
}

static int bind_routed_expert_tensors(ds4_v100_layer_state *out,
                                      const ds4_v100_context *ctx,
                                      int layer_id,
                                      char *err,
                                      size_t errlen) {
    char tm_err[256];
    memset(tm_err, 0, sizeof(tm_err));
    int g = ds4_v100_context_require_layer_turbomind_binding(
        ctx, layer_id, "ffn_gate_exps.weight", &out->turbomind_gate_binding,
        tm_err, sizeof(tm_err));
    int u = ds4_v100_context_require_layer_turbomind_binding(
        ctx, layer_id, "ffn_up_exps.weight", &out->turbomind_up_binding,
        tm_err, sizeof(tm_err));
    int gu = ds4_v100_context_require_layer_turbomind_binding(
        ctx, layer_id, "ffn_gate_up_exps.weight", &out->turbomind_gate_up_binding,
        tm_err, sizeof(tm_err));
    int d = ds4_v100_context_require_layer_turbomind_binding(
        ctx, layer_id, "ffn_down_exps.weight", &out->turbomind_down_binding,
        tm_err, sizeof(tm_err));
    const bool has_separate_gate_up = !g && !u;
    const bool has_partial_separate_gate_up = (!g) != (!u);
    const bool has_fused_gate_up = !gu;
    if (!d && (has_separate_gate_up || has_fused_gate_up)) {
        if (has_partial_separate_gate_up) {
            return state_error(err, errlen,
                               "partial TurboMind routed expert binding for layer %d",
                               layer_id);
        }
        out->has_turbomind_routed = true;
        out->has_turbomind_fused_gate_up = has_fused_gate_up;
        ds4_v100_tensor_binding fused_gate_binding;
        ds4_v100_tensor_binding fused_up_binding;
        memset(&fused_gate_binding, 0, sizeof(fused_gate_binding));
        memset(&fused_up_binding, 0, sizeof(fused_up_binding));
        if (has_fused_gate_up) {
            if (make_turbomind_fused_gate_up_synthetic_binding(
                    &out->turbomind_gate_up_binding,
                    &fused_gate_binding,
                    "ffn_gate_up_exps.weight", err, errlen) ||
                make_turbomind_fused_gate_up_synthetic_binding(
                    &out->turbomind_gate_up_binding,
                    &fused_up_binding,
                    "ffn_gate_up_exps.weight", err, errlen)) {
                return 1;
            }
            out->turbomind_gate_up_view = tm_gpu_view(&out->turbomind_gate_up_binding);
        }
        if (has_separate_gate_up) {
            if (make_turbomind_routed_binding(&out->turbomind_gate_binding,
                                              &out->routed_gate_binding,
                                              "ffn_gate_exps.weight", err, errlen) ||
                make_turbomind_routed_binding(&out->turbomind_up_binding,
                                              &out->routed_up_binding,
                                              "ffn_up_exps.weight", err, errlen)) {
                return 1;
            }
            out->turbomind_gate_view = tm_gpu_view(&out->turbomind_gate_binding);
            out->turbomind_up_view = tm_gpu_view(&out->turbomind_up_binding);
        } else if (has_fused_gate_up) {
            out->routed_gate_binding = fused_gate_binding;
            out->routed_up_binding = fused_up_binding;
        }
        if (make_turbomind_routed_binding(&out->turbomind_down_binding,
                                          &out->routed_down_binding,
                                          "ffn_down_exps.weight", err, errlen)) {
            return 1;
        }
        out->turbomind_down_view = tm_gpu_view(&out->turbomind_down_binding);
        return 0;
    }

    return bind_required(ctx, layer_id, "ffn_gate_exps.weight",
                         &out->routed_gate_binding, err, errlen) ||
           bind_required(ctx, layer_id, "ffn_up_exps.weight",
                         &out->routed_up_binding, err, errlen) ||
           bind_required(ctx, layer_id, "ffn_down_exps.weight",
                         &out->routed_down_binding, err, errlen);
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

    if (bind_routed_expert_tensors(out, ctx, layer_id, err, errlen) ||
        bind_required(ctx, layer_id, "ffn_gate_shexp.weight", &out->shared_gate.binding, err, errlen) ||
        bind_required(ctx, layer_id, "ffn_up_shexp.weight", &out->shared_up.binding, err, errlen) ||
        bind_required(ctx, layer_id, "ffn_down_shexp.weight", &out->shared_down.binding, err, errlen) ||
        bind_required(ctx, layer_id, "ffn_gate_inp.weight", &out->router.binding, err, errlen) ||
        bind_required(ctx, layer_id, "attn_q_a.weight", &out->attn_q_a.binding, err, errlen) ||
        bind_required(ctx, layer_id, "attn_q_b.weight", &out->attn_q_b.binding, err, errlen) ||
        bind_required(ctx, layer_id, "attn_kv_latent.weight", &out->attn_kv_latent.binding, err, errlen) ||
        bind_required(ctx, layer_id, "attn_output_a.weight", &out->attn_output_a.binding, err, errlen) ||
        bind_required(ctx, layer_id, "attn_output_b.weight", &out->attn_output_b.binding, err, errlen) ||
        bind_required(ctx, layer_id, "attn_norm.weight", &out->attn_norm, err, errlen) ||
        bind_required(ctx, layer_id, "attn_q_a_norm.weight", &out->attn_q_a_norm, err, errlen) ||
        bind_required(ctx, layer_id, "attn_kv_a_norm.weight", &out->attn_kv_a_norm, err, errlen) ||
        bind_required(ctx, layer_id, "attn_sinks", &out->attn_sinks, err, errlen) ||
        bind_required(ctx, layer_id, "hc_attn_fn", &out->hc_attn_fn, err, errlen) ||
        bind_required(ctx, layer_id, "hc_attn_base", &out->hc_attn_base, err, errlen) ||
        bind_required(ctx, layer_id, "hc_attn_scale", &out->hc_attn_scale, err, errlen) ||
        bind_required(ctx, layer_id, "ffn_norm.weight", &out->ffn_norm, err, errlen) ||
        bind_required(ctx, layer_id, "hc_ffn_fn", &out->hc_ffn_fn, err, errlen) ||
        bind_required(ctx, layer_id, "hc_ffn_base", &out->hc_ffn_base, err, errlen) ||
        bind_required(ctx, layer_id, "hc_ffn_scale", &out->hc_ffn_scale, err, errlen)) {
        return 1;
    }

    out->has_hash_router = bind_optional(ctx, layer_id, "ffn_gate_tid2eid", &out->router_hash);
    out->has_bias_router = bind_optional(ctx, layer_id, "exp_probs_b", &out->router_bias);

    if (make_f32_matrix(&out->router.binding, &out->router, "ffn_gate_inp.weight", err, errlen) ||
        make_f8_matrix(&out->attn_q_a.binding, &out->attn_q_a, "attn_q_a.weight", err, errlen) ||
        make_f8_matrix(&out->attn_q_b.binding, &out->attn_q_b, "attn_q_b.weight", err, errlen) ||
        make_f8_matrix(&out->attn_kv_latent.binding, &out->attn_kv_latent, "attn_kv_latent.weight", err, errlen) ||
        make_f8_matrix(&out->attn_output_a.binding, &out->attn_output_a, "attn_output_a.weight", err, errlen) ||
        make_f8_matrix(&out->attn_output_b.binding, &out->attn_output_b, "attn_output_b.weight", err, errlen) ||
        make_f8_matrix(&out->shared_gate.binding, &out->shared_gate, "ffn_gate_shexp.weight", err, errlen) ||
        make_f8_matrix(&out->shared_up.binding, &out->shared_up, "ffn_up_shexp.weight", err, errlen) ||
        make_f8_matrix(&out->shared_down.binding, &out->shared_down, "ffn_down_shexp.weight", err, errlen) ||
        validate_router_binding(out, err, errlen)) {
        return 1;
    }

    out->compress_ratio = out->layer_class == DS4_V100_LAYER_RATIO_4 ? 4u :
                          out->layer_class == DS4_V100_LAYER_RATIO_128 ? 128u : 0u;
    out->compressor_width = out->compress_ratio == 4u ? 2u * DS4_V100_HEAD_DIM :
                            out->compress_ratio == 128u ? DS4_V100_HEAD_DIM : 0u;
    out->indexer_q_width = out->compress_ratio == 4u ? 64u * DS4_V100_INDEXER_HEAD_DIM : 0u;
    out->indexer_proj_width = out->compress_ratio == 4u ? 64u : 0u;
    out->indexer_compressor_width = out->compress_ratio == 4u ? 2u * DS4_V100_INDEXER_HEAD_DIM : 0u;

    if (out->compress_ratio != 0) {
        ds4_v100_tensor_binding comp_ape;
        ds4_v100_tensor_binding comp_kv;
        ds4_v100_tensor_binding comp_gate;
        if (bind_required(ctx, layer_id, "attn_compress_ape", &comp_ape, err, errlen) ||
            bind_required(ctx, layer_id, "attn_compress_kv.weight", &comp_kv, err, errlen) ||
            bind_required(ctx, layer_id, "attn_compress_gate.weight", &comp_gate, err, errlen) ||
            bind_required(ctx, layer_id, "attn_compress_norm.weight", &out->attn_compressor_norm, err, errlen) ||
            make_f32_matrix(&comp_ape, &out->attn_compressor_ape, "attn_compress_ape", err, errlen) ||
            make_bf16_matrix(&comp_kv, &out->attn_compressor_kv, "attn_compress_kv.weight", err, errlen) ||
            make_bf16_matrix(&comp_gate, &out->attn_compressor_gate, "attn_compress_gate.weight", err, errlen)) {
            return 1;
        }
        out->has_attention_compressor = true;
        if (check_same_owner(out, &out->attn_compressor_ape.binding, "attn_compress_ape", err, errlen) ||
            check_same_owner(out, &out->attn_compressor_kv.binding, "attn_compress_kv.weight", err, errlen) ||
            check_same_owner(out, &out->attn_compressor_gate.binding, "attn_compress_gate.weight", err, errlen) ||
            check_same_owner(out, &out->attn_compressor_norm, "attn_compress_norm.weight", err, errlen)) {
            return 1;
        }
    }

    if (out->compress_ratio == 4u) {
        ds4_v100_tensor_binding index_q_b;
        ds4_v100_tensor_binding index_proj;
        ds4_v100_tensor_binding index_ape;
        ds4_v100_tensor_binding index_kv;
        ds4_v100_tensor_binding index_gate;
        if (bind_required(ctx, layer_id, "indexer.attn_q_b.weight", &index_q_b, err, errlen) ||
            bind_required(ctx, layer_id, "indexer.proj.weight", &index_proj, err, errlen) ||
            bind_required(ctx, layer_id, "indexer.compress_ape", &index_ape, err, errlen) ||
            bind_required(ctx, layer_id, "indexer.compress_kv.weight", &index_kv, err, errlen) ||
            bind_required(ctx, layer_id, "indexer.compress_gate.weight", &index_gate, err, errlen) ||
            bind_required(ctx, layer_id, "indexer.compress_norm.weight", &out->indexer_compressor_norm, err, errlen) ||
            make_f8_matrix(&index_q_b, &out->indexer_attn_q_b, "indexer.attn_q_b.weight", err, errlen) ||
            make_bf16_matrix(&index_proj, &out->indexer_proj, "indexer.proj.weight", err, errlen) ||
            make_f32_matrix(&index_ape, &out->indexer_compressor_ape, "indexer.compress_ape", err, errlen) ||
            make_bf16_matrix(&index_kv, &out->indexer_compressor_kv, "indexer.compress_kv.weight", err, errlen) ||
            make_bf16_matrix(&index_gate, &out->indexer_compressor_gate, "indexer.compress_gate.weight", err, errlen)) {
            return 1;
        }
        out->has_indexer = true;
        if (check_same_owner(out, &out->indexer_attn_q_b.binding, "indexer.attn_q_b.weight", err, errlen) ||
            check_same_owner(out, &out->indexer_proj.binding, "indexer.proj.weight", err, errlen) ||
            check_same_owner(out, &out->indexer_compressor_ape.binding, "indexer.compress_ape", err, errlen) ||
            check_same_owner(out, &out->indexer_compressor_kv.binding, "indexer.compress_kv.weight", err, errlen) ||
            check_same_owner(out, &out->indexer_compressor_gate.binding, "indexer.compress_gate.weight", err, errlen) ||
            check_same_owner(out, &out->indexer_compressor_norm, "indexer.compress_norm.weight", err, errlen)) {
            return 1;
        }
    }

    const ds4_v100_tensor_binding *owned[] = {
        &out->routed_gate_binding,
        &out->routed_up_binding,
        &out->routed_down_binding,
        &out->shared_gate.binding,
        &out->shared_up.binding,
        &out->shared_down.binding,
        &out->router.binding,
        &out->attn_q_a.binding,
        &out->attn_q_b.binding,
        &out->attn_kv_latent.binding,
        &out->attn_output_a.binding,
        &out->attn_output_b.binding,
        &out->attn_norm,
        &out->attn_q_a_norm,
        &out->attn_kv_a_norm,
        &out->attn_sinks,
        &out->hc_attn_fn,
        &out->hc_attn_base,
        &out->hc_attn_scale,
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
        "attn_q_a.weight",
        "attn_q_b.weight",
        "attn_kv_latent.weight",
        "attn_output_a.weight",
        "attn_output_b.weight",
        "attn_norm.weight",
        "attn_q_a_norm.weight",
        "attn_kv_a_norm.weight",
        "attn_sinks",
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
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
    out->q_lora_rank = out->attn_q_a.rows;
    out->q_width = out->attn_q_b.rows;
    out->kv_latent_width = out->attn_kv_latent.rows;
    out->attention_output_rank = out->attn_output_a.rows;
    out->intermediate_size = out->shared_gate.rows;
    if (out->attn_q_a.cols != out->hidden_size ||
        out->attn_q_b.cols != out->q_lora_rank ||
        out->attn_kv_latent.cols != out->hidden_size ||
        out->attn_output_a.cols != out->hidden_size ||
        out->attn_output_b.cols != out->attention_output_rank ||
        out->attn_output_b.rows != out->hidden_size ||
        out->q_lora_rank != 1024 ||
        out->q_width != 32768 ||
        out->kv_latent_width != DS4_V100_HEAD_DIM ||
        out->attention_output_rank != 8192) {
        return state_error(err, errlen, "attention projection dimensions do not match DS4 layer state");
    }
    if (!dtype_is(&out->attn_norm, "f32") ||
        !dtype_is(&out->attn_q_a_norm, "f32") ||
        !dtype_is(&out->attn_kv_a_norm, "f32") ||
        !dtype_is(&out->attn_sinks, "f32") ||
        out->attn_norm.n_shape_dims != 1 || out->attn_norm.shape[0] != out->hidden_size ||
        out->attn_q_a_norm.n_shape_dims != 1 || out->attn_q_a_norm.shape[0] != out->q_lora_rank ||
        out->attn_kv_a_norm.n_shape_dims != 1 || out->attn_kv_a_norm.shape[0] != out->kv_latent_width ||
        out->attn_sinks.n_shape_dims != 1 || out->attn_sinks.shape[0] != 64) {
        return state_error(err, errlen, "attention control dimensions do not match DS4 layer state");
    }
    if (out->has_attention_compressor) {
        if (out->attn_compressor_ape.rows != out->compress_ratio ||
            out->attn_compressor_ape.cols != out->compressor_width ||
            out->attn_compressor_kv.rows != out->compressor_width ||
            out->attn_compressor_kv.cols != out->hidden_size ||
            out->attn_compressor_gate.rows != out->compressor_width ||
            out->attn_compressor_gate.cols != out->hidden_size ||
            !dtype_is(&out->attn_compressor_norm, "f32") ||
            out->attn_compressor_norm.n_shape_dims != 1 ||
            out->attn_compressor_norm.shape[0] != DS4_V100_HEAD_DIM) {
            return state_error(err, errlen, "attention compressor dimensions do not match DS4 layer state");
        }
    }
    if (out->has_indexer) {
        if (out->indexer_attn_q_b.rows != out->indexer_q_width ||
            out->indexer_attn_q_b.cols != out->q_lora_rank ||
            out->indexer_proj.rows != out->indexer_proj_width ||
            out->indexer_proj.cols != out->hidden_size ||
            out->indexer_compressor_ape.rows != out->compress_ratio ||
            out->indexer_compressor_ape.cols != out->indexer_compressor_width ||
            out->indexer_compressor_kv.rows != out->indexer_compressor_width ||
            out->indexer_compressor_kv.cols != out->hidden_size ||
            out->indexer_compressor_gate.rows != out->indexer_compressor_width ||
            out->indexer_compressor_gate.cols != out->hidden_size ||
            !dtype_is(&out->indexer_compressor_norm, "f32") ||
            out->indexer_compressor_norm.n_shape_dims != 1 ||
            out->indexer_compressor_norm.shape[0] != DS4_V100_INDEXER_HEAD_DIM) {
            return state_error(err, errlen, "indexer dimensions do not match DS4 layer state");
        }
    }
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
    if (state->has_turbomind_routed) {
        return state_error(err, errlen,
                           "source MXFP4 expert matrix view is unavailable for TurboMind-bound layer %d",
                           state->layer_id);
    }
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

static int grow_span_for_turbomind_view(const ds4_gpu_turbomind_mxfp4_matrix_view *view,
                                        uint64_t *span) {
    if (!view || !span) return 1;
    uint64_t weight_end =
        view->weight_offset + (uint64_t)view->experts_packed * view->weight_bytes_per_expert;
    uint64_t scale_end =
        view->scale_offset + (uint64_t)view->experts_packed * view->scale_bytes_per_expert;
    if (weight_end < view->weight_offset || scale_end < view->scale_offset) return 1;
    if (weight_end > *span) *span = weight_end;
    if (scale_end > *span) *span = scale_end;
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
    if (state->has_turbomind_routed) {
        for (uint32_t i = 0; i < n_selected; i++) {
            if (!selected_experts) return state_error(err, errlen, "missing selected experts");
            if (selected_experts[i] < 0 || (uint32_t)selected_experts[i] >= state->routed_experts) {
                return state_error(err, errlen, "selected expert %d outside routed expert count",
                                   selected_experts[i]);
            }
        }
        if (state->has_turbomind_fused_gate_up &&
            grow_span_for_turbomind_view(&state->turbomind_gate_up_view, &span)) {
            return state_error(err, errlen, "TurboMind routed FFN arena span overflow");
        }
        if ((state->turbomind_gate_view.experts_packed || state->turbomind_up_view.experts_packed) &&
            (grow_span_for_turbomind_view(&state->turbomind_gate_view, &span) ||
             grow_span_for_turbomind_view(&state->turbomind_up_view, &span))) {
            return state_error(err, errlen, "TurboMind routed FFN arena span overflow");
        }
        if (grow_span_for_turbomind_view(&state->turbomind_down_view, &span)) {
            return state_error(err, errlen, "TurboMind routed FFN arena span overflow");
        }
        *out_bytes = span;
        return 0;
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

int ds4_v100_layer_state_attention_arena_span(const ds4_v100_layer_state *state,
                                              uint64_t *out_bytes,
                                              char *err,
                                              size_t errlen) {
    if (!state || !out_bytes) return state_error(err, errlen, "missing attention arena span output");
    uint64_t span = 0;
    if (grow_span_for_matrix(&state->attn_q_a, &span) ||
        grow_span_for_matrix(&state->attn_q_b, &span) ||
        grow_span_for_matrix(&state->attn_kv_latent, &span) ||
        grow_span_for_matrix(&state->attn_output_a, &span) ||
        grow_span_for_matrix(&state->attn_output_b, &span)) {
        return state_error(err, errlen, "attention arena span overflow");
    }
    if (state->has_attention_compressor &&
        (grow_span_for_matrix(&state->attn_compressor_ape, &span) ||
         grow_span_for_matrix(&state->attn_compressor_kv, &span) ||
         grow_span_for_matrix(&state->attn_compressor_gate, &span))) {
        return state_error(err, errlen, "attention compressor arena span overflow");
    }
    if (state->has_indexer &&
        (grow_span_for_matrix(&state->indexer_attn_q_b, &span) ||
         grow_span_for_matrix(&state->indexer_proj, &span) ||
         grow_span_for_matrix(&state->indexer_compressor_ape, &span) ||
         grow_span_for_matrix(&state->indexer_compressor_kv, &span) ||
         grow_span_for_matrix(&state->indexer_compressor_gate, &span))) {
        return state_error(err, errlen, "indexer arena span overflow");
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

int ds4_v100_bound_matrix_bf16_view(const ds4_v100_bound_matrix *matrix,
                                    ds4_gpu_bf16_matrix_view *out,
                                    char *err,
                                    size_t errlen) {
    if (!matrix || !out) return state_error(err, errlen, "missing bf16 matrix view output");
    if (!dtype_is(&matrix->binding, "bf16")) {
        return state_error(err, errlen, "matrix is not bf16");
    }
    if (matrix->rows == 0 || matrix->cols == 0 || matrix->row_bytes == 0 || matrix->bytes == 0) {
        return state_error(err, errlen, "invalid bf16 matrix dimensions");
    }
    memset(out, 0, sizeof(*out));
    out->arena_offset = ds4_v100_bound_matrix_arena_offset(matrix);
    out->byte_length = matrix->bytes;
    out->rows = matrix->rows;
    out->cols = matrix->cols;
    out->row_stride_elements = matrix->cols;
    return 0;
}
