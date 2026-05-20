#ifndef DS4_V100_LAYER_STATE_H
#define DS4_V100_LAYER_STATE_H

#include "ds4_gpu.h"
#include "ds4_v100_context.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DS4_V100_ROUTER_UNKNOWN = 0,
    DS4_V100_ROUTER_HASH,
    DS4_V100_ROUTER_BIAS,
} ds4_v100_router_kind;

typedef struct {
    ds4_v100_tensor_binding binding;
    uint32_t rows;
    uint32_t cols;
    uint32_t planes;
    uint64_t row_bytes;
    uint64_t bytes;
    uint64_t rel;
} ds4_v100_bound_matrix;

typedef struct {
    ds4_v100_bound_matrix gate;
    ds4_v100_bound_matrix up;
    ds4_v100_bound_matrix down;
} ds4_v100_route_matrices;

typedef struct {
    int layer_id;
    int stage_id;
    int owning_gpu;
    ds4_v100_layer_class layer_class;
    ds4_v100_layer_kv_view kv_view;

    uint32_t hidden_size;
    uint32_t q_lora_rank;
    uint32_t q_width;
    uint32_t kv_latent_width;
    uint32_t attention_output_rank;
    uint32_t compress_ratio;
    uint32_t compressor_width;
    uint32_t indexer_q_width;
    uint32_t indexer_proj_width;
    uint32_t indexer_compressor_width;
    uint32_t intermediate_size;
    uint32_t routed_experts;
    uint32_t routes_per_token;
    uint32_t router_token_capacity;
    ds4_v100_router_kind router_kind;

    ds4_v100_bound_matrix router;
    ds4_v100_tensor_binding router_hash;
    ds4_v100_tensor_binding router_bias;
    bool has_hash_router;
    bool has_bias_router;

    ds4_v100_bound_matrix attn_q_a;
    ds4_v100_bound_matrix attn_q_b;
    ds4_v100_bound_matrix attn_kv_latent;
    ds4_v100_bound_matrix attn_output_a;
    ds4_v100_bound_matrix attn_output_b;
    ds4_v100_tensor_binding attn_norm;
    ds4_v100_tensor_binding attn_q_a_norm;
    ds4_v100_tensor_binding attn_kv_a_norm;
    ds4_v100_tensor_binding attn_sinks;
    ds4_v100_bound_matrix attn_compressor_ape;
    ds4_v100_bound_matrix attn_compressor_kv;
    ds4_v100_bound_matrix attn_compressor_gate;
    ds4_v100_tensor_binding attn_compressor_norm;
    bool has_attention_compressor;
    ds4_v100_bound_matrix indexer_attn_q_b;
    ds4_v100_bound_matrix indexer_proj;
    ds4_v100_bound_matrix indexer_compressor_ape;
    ds4_v100_bound_matrix indexer_compressor_kv;
    ds4_v100_bound_matrix indexer_compressor_gate;
    ds4_v100_tensor_binding indexer_compressor_norm;
    bool has_indexer;
    ds4_v100_tensor_binding hc_attn_fn;
    ds4_v100_tensor_binding hc_attn_base;
    ds4_v100_tensor_binding hc_attn_scale;

    ds4_v100_tensor_binding routed_gate_binding;
    ds4_v100_tensor_binding routed_up_binding;
    ds4_v100_tensor_binding routed_down_binding;
    bool has_turbomind_routed;
    bool has_turbomind_fused_gate_up;
    ds4_v100_turbomind_binding turbomind_gate_binding;
    ds4_v100_turbomind_binding turbomind_up_binding;
    ds4_v100_turbomind_binding turbomind_gate_up_binding;
    ds4_v100_turbomind_binding turbomind_down_binding;
    ds4_gpu_turbomind_mxfp4_matrix_view turbomind_gate_view;
    ds4_gpu_turbomind_mxfp4_matrix_view turbomind_up_view;
    ds4_gpu_turbomind_mxfp4_matrix_view turbomind_gate_up_view;
    ds4_gpu_turbomind_mxfp4_matrix_view turbomind_down_view;
    ds4_v100_bound_matrix shared_gate;
    ds4_v100_bound_matrix shared_up;
    ds4_v100_bound_matrix shared_down;

    ds4_v100_tensor_binding ffn_norm;
    ds4_v100_tensor_binding hc_ffn_fn;
    ds4_v100_tensor_binding hc_ffn_base;
    ds4_v100_tensor_binding hc_ffn_scale;
} ds4_v100_layer_state;

const char *ds4_v100_router_kind_name(ds4_v100_router_kind kind);

int ds4_v100_layer_state_init(ds4_v100_layer_state *out,
                              const ds4_v100_context *ctx,
                              int layer_id,
                              char *err,
                              size_t errlen);

int ds4_v100_layer_state_route_matrices(const ds4_v100_layer_state *state,
                                        uint32_t expert,
                                        ds4_v100_route_matrices *out,
                                        char *err,
                                        size_t errlen);

int ds4_v100_layer_state_ffn_arena_span(const ds4_v100_layer_state *state,
                                        const int32_t *selected_experts,
                                        uint32_t n_selected,
                                        uint64_t *out_bytes,
                                        char *err,
                                        size_t errlen);

int ds4_v100_layer_state_attention_arena_span(const ds4_v100_layer_state *state,
                                              uint64_t *out_bytes,
                                              char *err,
                                              size_t errlen);

uint64_t ds4_v100_bound_matrix_arena_offset(const ds4_v100_bound_matrix *matrix);

int ds4_v100_bound_matrix_source_view(const ds4_v100_bound_matrix *matrix,
                                      ds4_gpu_source_row_view *out,
                                      char *err,
                                      size_t errlen);

int ds4_v100_bound_matrix_bf16_view(const ds4_v100_bound_matrix *matrix,
                                    ds4_gpu_bf16_matrix_view *out,
                                    char *err,
                                    size_t errlen);

#ifdef __cplusplus
}
#endif

#endif
