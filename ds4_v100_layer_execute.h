#ifndef DS4_V100_LAYER_EXECUTE_H
#define DS4_V100_LAYER_EXECUTE_H

#include "ds4_gpu.h"
#include "ds4_v100_layer_state.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_V100_RMS_EPS 1.0e-6f
#define DS4_V100_N_HEAD 64u
#define DS4_V100_N_ROT 64u
#define DS4_V100_N_HC 4u
#define DS4_V100_HC_MIX 24u
#define DS4_V100_HC_SINKHORN_ITERS 20u
#define DS4_V100_INDEXER_HEAD 64u
#define DS4_V100_INDEXER_TOP_K 512u
#define DS4_V100_OUT_GROUPS 8u
#define DS4_V100_OUT_GROUP_DIM 4096u
#define DS4_V100_OUT_GROUP_RANK 1024u
#define DS4_V100_LAYER_MAX_BATCH 256u

typedef struct {
    ds4_gpu_tensor *raw_kv;
    uint32_t raw_cap;
    uint32_t raw_window;

    ds4_gpu_tensor *attn_state_kv;
    ds4_gpu_tensor *attn_state_score;
    ds4_gpu_tensor *attn_comp_kv;
    uint32_t attn_comp_cap;
    uint32_t n_attn_comp;

    ds4_gpu_tensor *index_state_kv;
    ds4_gpu_tensor *index_state_score;
    ds4_gpu_tensor *index_comp_kv;
    uint32_t index_comp_cap;
    uint32_t n_index_comp;

    ds4_gpu_tensor *indexer_topk;
    uint32_t indexer_top_k;
} ds4_v100_layer_decode_cache;

enum {
    DS4_V100_HC_CHECKPOINT_LAYER_FINAL = 0,
    DS4_V100_HC_CHECKPOINT_AFTER_ATTN = 1,
    DS4_V100_HC_CHECKPOINT_SEED = 2,
};

typedef struct {
    int layer;
    int kind;
    const ds4_gpu_tensor *hc;
    uint64_t hc_bytes;
} ds4_v100_layer_execute_checkpoint;

typedef int (*ds4_v100_layer_execute_checkpoint_fn)(
    const ds4_v100_layer_execute_checkpoint *checkpoint,
    void *user,
    char *err,
    size_t errlen);

typedef struct {
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t routes;
    uint32_t q_rank;
    uint32_t q_width;
    uint32_t kv_width;
    uint32_t out_rank;
    uint32_t max_slots;

    ds4_gpu_tensor *hc_norm[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *hc_mix[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *attn_split[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *ffn_split[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *attn_cur[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *attn_out_batch;
    ds4_gpu_tensor *attn_out[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *after_attn_hc[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *ffn_cur[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *ffn_norm[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *ffn_delta[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *ffn_norm_batch;
    ds4_gpu_tensor *ffn_delta_batch;

    ds4_gpu_tensor *ffn_router;
    ds4_gpu_tensor *ffn_probs;
    ds4_gpu_tensor *ffn_selected;
    ds4_gpu_tensor *ffn_weights;
    ds4_gpu_tensor *ffn_tokens;
    ds4_gpu_tensor *ffn_input_ptrs;
    uint32_t ffn_input_ptrs_valid;
    uint32_t ffn_input_ptrs_slots;
    uint64_t ffn_input_ptrs_min_row_bytes;
    ds4_gpu_tensor *ffn_routed_mid;
    ds4_gpu_tensor *ffn_routed_out;
    ds4_gpu_tensor *ffn_routed_out_view[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *ffn_shared_gate;
    ds4_gpu_tensor *ffn_shared_up;
    ds4_gpu_tensor *ffn_shared_mid;
    ds4_gpu_tensor *ffn_shared;
    ds4_gpu_tensor *ffn_shared_mid_batch;
    ds4_gpu_tensor *ffn_shared_batch;
    ds4_gpu_tensor *ffn_shared_batch_view[DS4_V100_LAYER_MAX_BATCH];

    ds4_gpu_tensor *attn_input_ptrs;
    ds4_gpu_tensor *attn_norm_batch;
    ds4_gpu_tensor *attn_q_a_batch;
    ds4_gpu_tensor *attn_q_a_norm_batch;
    ds4_gpu_tensor *attn_q_batch;
    ds4_gpu_tensor *attn_kv_raw_batch;
    ds4_gpu_tensor *attn_kv_batch;
    ds4_gpu_tensor *attn_heads_batch;
    ds4_gpu_tensor *attn_low_batch;
    ds4_gpu_tensor *attn_norm_view[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *attn_q_a_norm_view[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *attn_q_view[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *attn_kv_view[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *attn_heads_view[DS4_V100_LAYER_MAX_BATCH];
    ds4_gpu_tensor *attn_low_view[DS4_V100_LAYER_MAX_BATCH];
} ds4_v100_layer_batch_scratch;

void ds4_v100_layer_batch_scratch_init(ds4_v100_layer_batch_scratch *scratch);
void ds4_v100_layer_batch_scratch_free(ds4_v100_layer_batch_scratch *scratch);

typedef struct {
    const void *model_map;
    uint64_t model_size;
    bool model_map_uses_shard_offsets;
    ds4_gpu_arena *arena;
    ds4_v100_layer_batch_scratch *batch_scratch;

    uint32_t router_token;
    uint32_t position;

    const ds4_gpu_tensor *raw_kv;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;

    const ds4_gpu_tensor *compressed_kv;
    uint32_t n_compressed;
    const ds4_gpu_tensor *compressed_mask;
    bool use_compressed_mask;

    ds4_v100_layer_decode_cache *decode_cache;
    bool fp8_kv_cache;
    bool suppress_router_readback;

    int tp2_layer;
    ds4_gpu_arena *tp2_owner_arena;
    ds4_gpu_arena *tp2_peer_arena;
    ds4_gpu_tensor *tp2_peer_input;
    ds4_gpu_tensor *tp2_peer_selected;
    ds4_gpu_tensor *tp2_peer_weights;
    ds4_gpu_tensor *tp2_peer_out;
    ds4_gpu_tensor *tp2_peer_recv;
    uint32_t tp2_scratch_slots;

    int checkpoint_layer;
    ds4_v100_layer_execute_checkpoint_fn checkpoint_fn;
    void *checkpoint_user;
} ds4_v100_layer_execute_config;

typedef struct {
    int32_t selected_experts[6];
    float route_weights[6];
    uint32_t routes;
    uint32_t turbomind_routed;
    uint32_t turbomind_tp2_routed;
    double timing_tp2_copy_in_ms;
    double timing_tp2_owner_ms;
    double timing_tp2_peer_ms;
    double timing_tp2_copy_out_ms;
    double timing_tp2_reduce_ms;
    double timing_tp2_total_ms;
    double timing_hc_attn_ms;
    double timing_attention_ms;
    double timing_hc_ffn_ms;
    double timing_ffn_ms;
    double timing_hc_final_ms;
    double timing_total_ms;
} ds4_v100_layer_execute_report;

int ds4_v100_layer_execute_decode(
        const ds4_v100_layer_state          *state,
        const ds4_v100_layer_execute_config *cfg,
        const ds4_gpu_tensor                *hidden,
        ds4_gpu_tensor                      *next_hidden,
        ds4_v100_layer_execute_report       *report,
        char                                *err,
        size_t                               errlen);

int ds4_v100_layer_execute_hc_decode(
        const ds4_v100_layer_state          *state,
        const ds4_v100_layer_execute_config *cfg,
        const ds4_gpu_tensor                *hidden_hc,
        ds4_gpu_tensor                      *next_hidden_hc,
        ds4_v100_layer_execute_report       *report,
        char                                *err,
        size_t                               errlen);

int ds4_v100_layer_execute_hc_decode_batch(
        const ds4_v100_layer_state           *state,
        const ds4_v100_layer_execute_config  *cfgs,
        const ds4_gpu_tensor *const          *hidden_hc,
        ds4_gpu_tensor *const                *next_hidden_hc,
        uint32_t                              n_slots,
        ds4_v100_layer_execute_report        *reports,
        char                                 *err,
        size_t                                errlen);

#ifdef __cplusplus
}
#endif

#endif
