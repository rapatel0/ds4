#define _FILE_OFFSET_BITS 64

#include "ds4_v100_tp_runtime.h"
#include "kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h"
extern "C" {
#include "ds4.h"
}

#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <mma.h>
#include <nccl.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kGpus = 8;
constexpr int kHidden = 4096;
constexpr int kMid = 2048;
constexpr int kHeadDim = 512;
constexpr int kHeadCount = 64;
constexpr int kLocalHeads = kHeadCount / kGpus;
constexpr int kAttentionOutputAInput = kLocalHeads * kHeadDim;
constexpr int kAttentionOutputAFull = 8192;
constexpr int kRawSwaRows = 128;
constexpr int kRotaryDim = 64;
constexpr int kIndexerHeadDim = 128;
constexpr int kIndexerHead = 64;
constexpr int kIndexerTopK = 512;
constexpr int kCompWidthMax = 2 * kHeadDim;
constexpr int kBoundedCompRows = 8;
constexpr int kIndexCompWidth = 2 * kIndexerHeadDim;
constexpr int kIndexCompStateRows = 8;
constexpr uint32_t kRopeOrigCtx = 65536;
constexpr float kRopeFreqBase = 10000.0f;
constexpr float kCompressRopeFreqBase = 160000.0f;
constexpr float kRopeScaleFactor = 16.0f;
constexpr float kRopeYarnBetaFast = 32.0f;
constexpr float kRopeYarnBetaSlow = 1.0f;
constexpr int kFusedN = 2 * kMid;
constexpr int kGlobalExperts = 256;
constexpr int kLocalExperts = kGlobalExperts / kGpus;
constexpr int kPackedLocalExperts = kLocalExperts;
constexpr int kRouterHashRows = 129280;
constexpr int kGroupSize = 32;
constexpr int kDType = GGML_TM_DTYPE_MXFP4;
constexpr int kHcRows = 4;
constexpr int kHcMix = 24;
constexpr int kModelTopK = 6;
constexpr float kSyntheticRouteWeight = 0.125f;
constexpr float kRoutedSwigluClamp = 10.0f;
constexpr float kReferenceRouteInputTargetAbs = 32.0f;
constexpr float kReferenceHcStateTargetAbs = 32.0f;
constexpr float kFp16Max = 65504.0f;

#define CHECK_CUDA(expr)                                                              \
    do {                                                                              \
        cudaError_t err__ = (expr);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            std::fprintf(stderr, "cuda error %s:%d: %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(err__));                                  \
            std::exit(2);                                                             \
        }                                                                             \
    } while (0)

#define CHECK_NCCL(expr)                                                              \
    do {                                                                              \
        ncclResult_t err__ = (expr);                                                   \
        if (err__ != ncclSuccess) {                                                   \
            std::fprintf(stderr, "nccl error %s:%d: %s\n", __FILE__, __LINE__,       \
                         ncclGetErrorString(err__));                                  \
            std::exit(2);                                                             \
        }                                                                             \
    } while (0)

typedef int (*pfn_init)(int);
typedef void (*pfn_shutdown)(void);
typedef int (*pfn_mmgt)(const void *, const int *, const int *, int, int,
                        const void * const *, const void * const *, int, int, int, int, int,
                        void *, void *);
typedef int (*pfn_mmgs)(const void *, const int *, const int *, int, int,
                        const void * const *, const void * const *, int, int, int, int, int,
                        void *, void *);

struct alignas(16) StridedPtrH {
    void *p;
    int stride;
};
static_assert(sizeof(StridedPtrH) == 16, "StridedPtrH must match TurboMind ABI");

struct Api {
    pfn_init init = nullptr;
    pfn_shutdown shutdown = nullptr;
    pfn_mmgt mmgt = nullptr;
    pfn_mmgs mmgs = nullptr;
    pfn_mmgs mmgs_clamped = nullptr;
};

struct ContractRow {
    std::string record_type;
    std::string tensor_id;
    std::string family;
    std::string source_dtype;
    std::string source_shape;
    std::string runtime_layout;
    int layer = -1;
    int owning_gpu = -1;
    int tp_rank = -1;
    int ep_rank = -1;
    int shard_index = -1;
    int shard_count = -1;
    int expert_first = -1;
    int expert_count = 0;
    int kv_ratio = -1;
    uint64_t kv_rows_per_slot = 0;
    uint64_t bytes_estimate = 0;
    std::string source_pack_file;
    uint64_t source_shard_offset = 0;
    uint64_t source_byte_length = 0;
    std::string kernel_family;
};

struct TmIndexEntry {
    std::string semantic_tensor_id;
    std::string runtime_layout;
    std::string sidecar_file;
    int layer_id = -1;
    int n = 0;
    int k = 0;
    int experts_packed = 0;
    int experts_total = 0;
    size_t weight_bytes_per_expert = 0;
    size_t scale_bytes_per_expert = 0;
    int k_pack = 0;
    int weight_stride = 0;
    int scale_stride = 0;
    uint64_t weight_offset = 0;
    uint64_t scale_offset = 0;
};

struct DescriptorBindings {
    TmIndexEntry gated;
    TmIndexEntry down;
    bool have_gated = false;
    bool have_down = false;
};

struct PackedExperts {
    std::vector<void *> d_w_active;
    std::vector<void *> d_s_active;
    void *d_w_table = nullptr;
    void *d_s_table = nullptr;
    int k_pack = 0;
};

struct RankState {
    int rank = 0;
    int device = 0;
    int routes = 0;
    int route_capacity = 0;
    int active_experts = 0;
    int max_routes_per_expert = 0;
    cudaStream_t stream = nullptr;
    cudaStream_t dense_stream = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaStream_t copy_streams[kGpus] = {};
    cudaEvent_t copy_done[kGpus] = {};
    cudaEvent_t stream_done = nullptr;
    cudaEvent_t dense_done = nullptr;
    int *d_route_index_by_slot[kGpus] = {};
    int *d_route_indices_by_slot[kGpus] = {};
    int *d_route_count_by_slot[kGpus] = {};
    int *d_route_compact_plan = nullptr;
    size_t route_compact_plan_ints = 0;
    int *d_router_selected_plan = nullptr;
    float *d_router_weights_plan = nullptr;
    int *d_route_offsets_all = nullptr;
    int *d_route_totals = nullptr;
    int *d_offsets = nullptr;
    int *d_route_slots = nullptr;
    float *d_route_weights = nullptr;
    float *d_route_inv_scale = nullptr;
    __half *d_a = nullptr;
    __half *d_gate_up = nullptr;
    __half *d_gated = nullptr;
    __half *d_down = nullptr;
    float *d_ep_contrib_all = nullptr;
    __half *d_ep_contrib_half_all = nullptr;
    float *d_ep_remote[kGpus] = {};
    __half *d_ep_remote_half[kGpus] = {};
    float *d_ep_sum = nullptr;
    float *d_next_hidden = nullptr;
    float *d_current_shard = nullptr;
    float *d_current_full = nullptr;
    float *d_current_full_rank_major = nullptr;
    float *d_final_hc_shard = nullptr;
    float *d_hc_scratch_shard = nullptr;
    float *d_hc_split = nullptr;
    float *d_attn_kv_full = nullptr;
    float *d_attn_raw_swa = nullptr;
    float *d_attn_raw_swa_layers[43] = {};
    float *d_attn_sinks = nullptr;
    float *d_attn_heads = nullptr;
    float *d_attn_output_a_full = nullptr;
    float *d_post_attn_shard = nullptr;
    float *d_attn_comp_kv_cur = nullptr;
    float *d_attn_comp_score_cur = nullptr;
    float *d_attn_comp_state_kv = nullptr;
    float *d_attn_comp_state_score = nullptr;
    float *d_attn_comp_rows = nullptr;
    float *d_attn_comp_state_kv_layers[43] = {};
    float *d_attn_comp_state_score_layers[43] = {};
    float *d_attn_comp_rows_layers[43] = {};
    uint32_t attn_comp_rows_written_layers[43] = {};
    uint64_t attn_comp_row_position_layers[43][kBoundedCompRows] = {};
    uint64_t attn_comp_row_loaded_position_layers[43][kBoundedCompRows] = {};
    bool attn_comp_row_loaded_layers[43][kBoundedCompRows] = {};
    uint32_t batched_paged_attn_plan_logged_layers[43] = {};
    uint32_t batched_paged_attn_plan_last_key_layers[43] = {};
    float *d_index_comp_kv_cur = nullptr;
    float *d_index_comp_score_cur = nullptr;
    float *d_index_comp_state_kv = nullptr;
    float *d_index_comp_state_score = nullptr;
    float *d_index_comp_rows = nullptr;
    float *d_index_comp_state_kv_layers[43] = {};
    float *d_index_comp_state_score_layers[43] = {};
    float *d_index_comp_rows_layers[43] = {};
    uint32_t index_comp_rows_written_layers[43] = {};
    uint64_t index_comp_row_position_layers[43][kBoundedCompRows] = {};
    uint64_t index_comp_row_loaded_position_layers[43][kBoundedCompRows] = {};
    bool index_comp_row_loaded_layers[43][kBoundedCompRows] = {};
    float *d_indexer_scores = nullptr;
    uint32_t *d_indexer_topk = nullptr;
    bool hc_initialized = false;
    PackedExperts gated;
    PackedExperts down;
    ncclComm_t compose_nccl = nullptr;
    bool compose_nccl_initialized = false;
    cudaEvent_t dense_wait = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t mid = nullptr;
    cudaEvent_t stop = nullptr;
};

struct RoutePlanHostWorkspace {
    bool initialized = false;
    bool uploads_pending = false;
    int slots = 0;
    int top_k = 0;
    int devices[kGpus] = {};
    size_t route_capacity = 0;
    size_t compact_plan_ints = 0;
    int *h_selected = nullptr;
    float *h_weights = nullptr;
    int *h_offsets[kGpus] = {};
    int *h_route_slots[kGpus] = {};
    float *h_route_weights[kGpus] = {};
    int *h_route_index_by_slot[kGpus] = {};
    int *h_route_indices_by_slot[kGpus] = {};
    int *h_route_count_by_slot[kGpus] = {};
    int *h_compact_plan = nullptr;
    cudaEvent_t upload_done[kGpus] = {};
};

struct GpuFamilyStats {
    uint64_t dense_rows = 0;
    uint64_t control_rows = 0;
    uint64_t expert_rows = 0;
    uint64_t kv_rows = 0;
    uint64_t comp_rows = 0;
    uint64_t dense_bytes = 0;
    uint64_t control_bytes = 0;
    uint64_t expert_descriptor_bytes = 0;
    uint64_t ep_loaded_bytes = 0;
    uint64_t checksum = 0;
};

struct LayerStats {
    uint64_t total_rows = 0;
    uint64_t dense_rows = 0;
    uint64_t control_rows = 0;
    uint64_t expert_rows = 0;
    uint64_t kv_rows = 0;
    uint64_t comp_rows = 0;
    uint64_t bad_rows = 0;
    uint64_t dense_loaded_bytes = 0;
    uint64_t control_loaded_bytes = 0;
    uint64_t ep_loaded_bytes = 0;
    uint64_t checksum = 0;
    GpuFamilyStats gpu[kGpus];
};

struct DenseComputeStats {
    bool enabled = false;
    bool pass = true;
    std::string tensor_id;
    int rows_per_gpu = 0;
    int cols = 0;
    int slots = 0;
    uint64_t loaded_bytes = 0;
    double compute_ms = 0.0;
    double repeat_max_abs = 0.0;
    double oracle_max_abs = 0.0;
    int repeat_bad = 0;
    int repeat_nan = 0;
    int oracle_bad = 0;
};

struct DeviceDenseOutputs {
    std::vector<float *> d_out;
    int rows_per_gpu = 0;
    int cols = 0;
    int slots = 0;
    uint64_t loaded_bytes = 0;
    double compute_ms = 0.0;
};

struct ResidentF8Dense {
    std::vector<uint8_t *> d_w;
    std::vector<float *> d_x;
    std::vector<__half *> d_w_half;
    std::vector<bool> owns_w_half;
    std::vector<__half *> d_x_half;
    std::vector<float *> d_out;
    std::vector<cublasHandle_t> cublas;
    int rows_per_gpu = 0;
    int cols = 0;
    int slots = 0;
    uint64_t row_bytes = 0;
    uint64_t loaded_bytes = 0;
};

struct LayerDenseOps {
    ResidentF8Dense attn_q_a;
    ResidentF8Dense attn_q_b;
    ResidentF8Dense attn_kv_latent;
    ResidentF8Dense attn_output_a;
    ResidentF8Dense attn_compress_kv;
    ResidentF8Dense attn_compress_gate;
    ResidentF8Dense indexer_attn_q_b;
    ResidentF8Dense indexer_proj;
    ResidentF8Dense indexer_compress_kv;
    ResidentF8Dense indexer_compress_gate;
    ResidentF8Dense attn;
    ResidentF8Dense shared;
    ResidentF8Dense shared_gate;
    ResidentF8Dense shared_up;
    bool initialized = false;
};

struct SharedDenseOps {
    LayerDenseOps layers[43];
    uint64_t loaded_bytes = 0;
    bool initialized = false;
};

struct DenseF16CacheEntry {
    std::string tensor_id;
    int gpu = -1;
    int cols = 0;
    int rows_per_gpu = 0;
    uint64_t offset = 0;
    uint64_t source_bytes = 0;
    uint64_t cache_bytes = 0;
};

struct DenseF16Cache {
    bool enabled = false;
    std::vector<uint8_t *> arena;
    std::vector<uint8_t *> temp;
    std::vector<DenseF16CacheEntry> entries;
    uint64_t gpu_cache_aligned_bytes[kGpus] = {};
    uint64_t gpu_temp_bytes[kGpus] = {};
    uint64_t rows = 0;
    uint64_t source_bytes = 0;
    uint64_t cache_bytes = 0;
    uint64_t cache_aligned_bytes = 0;
    uint64_t max_temp_bytes = 0;
};

struct ComposeStats {
    bool enabled = false;
    bool pass = true;
    uint64_t ep_contribution_bytes = 0;
    uint64_t ep_return_bytes = 0;
    double attn_dense_ms = 0.0;
    double shared_dense_ms = 0.0;
    double compose_ms = 0.0;
    double repeat_max_abs = 0.0;
    int finite_bad = 0;
    int repeat_bad = 0;
    uint64_t checksum = 0;
    bool ep_return_fp16 = false;
    bool fused_compose_sum = false;
    bool dense_hmma_compose = false;
    bool dense_f16_cublas_compose = false;
    bool dense_f16_cache_compose = false;
    bool nccl_reduce_scatter_compose = false;
};

struct LayerRunSummary {
    int layer = -1;
    int ratio = 0;
    bool pass = false;
    uint64_t total_rows = 0;
    uint64_t dense_rows = 0;
    uint64_t control_rows = 0;
    uint64_t expert_rows = 0;
    uint64_t kv_rows = 0;
    uint64_t comp_rows = 0;
    double decode_ms_per_step = 0.0;
    double decode_slot_step_tok_s = 0.0;
    double decode_ep_ms_per_step = 0.0;
    double decode_dense_ms_per_step = 0.0;
    double decode_compose_ms_per_step = 0.0;
    double decode_compose_reduce_ms_per_step = 0.0;
    double decode_compose_copy_ms_per_step = 0.0;
    double decode_compose_final_ms_per_step = 0.0;
    double decode_hc_current_input_ms_per_step = 0.0;
    double decode_hc_current_seed_ms_per_step = 0.0;
    double decode_hc_current_attn_mix_ms_per_step = 0.0;
    double decode_hc_current_split_ms_per_step = 0.0;
    double decode_hc_current_gather_ms_per_step = 0.0;
    double decode_hc_current_ffn_router_ms_per_step = 0.0;
    double decode_hc_current_ffn_norm_ms_per_step = 0.0;
    double decode_hc_current_router_select_ms_per_step = 0.0;
    double decode_hc_current_router_d2h_ms_per_step = 0.0;
    double decode_hc_current_route_upload_ms_per_step = 0.0;
    double decode_hc_current_fill_pack_ms_per_step = 0.0;
    double decode_pre_ep_hc_current_ms_per_step = 0.0;
    double decode_pre_ep_attention_projection_ms_per_step = 0.0;
    double decode_pre_ep_compressed_kv_ms_per_step = 0.0;
    double decode_pre_ep_attention_state_ms_per_step = 0.0;
    double decode_pre_ep_typed_history_ms_per_step = 0.0;
    double decode_pre_ep_raw_read_ms_per_step = 0.0;
    double decode_pre_ep_attention_output_ms_per_step = 0.0;
    double decode_pre_ep_post_attention_ffn_input_ms_per_step = 0.0;
    double decode_final_hc_ms_per_step = 0.0;
    int decode_cudagraph_sync_all_calls = 0;
    int decode_cudagraph_event_barrier_calls = 0;
    int decode_cudagraph_rank_stream_syncs = 0;
    int decode_cudagraph_dense_stream_syncs = 0;
    int decode_cudagraph_copy_stream_syncs = 0;
    int decode_cudagraph_capture_attempted = 0;
    int decode_cudagraph_capture_succeeded = 0;
    int decode_cudagraph_capture_error = 0;
    size_t decode_cudagraph_capture_nodes = 0;
    uint64_t decode_checksum = 0;
    int decode_finite_bad = 0;
    int rc = 0;
};

struct ServingBenchResult {
    uint64_t prompt_tokens = 0;
    uint64_t generated_tokens = 0;
    uint64_t continuation_tokens = 0;
    double first_token_decode_ms = 0.0;
    double continuation_decode_ms = 0.0;
    double first_token_wall_ms = 0.0;
    double continuation_wall_ms = 0.0;
    double total_decode_ms = 0.0;
    double total_wall_ms = 0.0;
    double total_ep_ms = 0.0;
    double total_dense_ms = 0.0;
    double total_compose_ms = 0.0;
    double total_compose_reduce_ms = 0.0;
    double total_compose_copy_ms = 0.0;
    double total_compose_final_ms = 0.0;
    double total_hc_current_input_ms = 0.0;
    double aggregate_generated_tok_s_decode = 0.0;
    double aggregate_generated_tok_s_wall = 0.0;
    double aggregate_continuation_tok_s_decode = 0.0;
    double aggregate_continuation_tok_s_wall = 0.0;
    bool diagnostic_output_head = false;
    bool diagnostic_output_head_proxy_hc = false;
    double output_head_ms = 0.0;
    double output_head_gather_ms = 0.0;
    double output_head_prep_ms = 0.0;
    double output_head_broadcast_ms = 0.0;
    double output_head_projection_ms = 0.0;
    double output_head_top1_ms = 0.0;
    bool token_input_seed = false;
    uint32_t first_input_token = UINT32_MAX;
    std::vector<uint32_t> selected_tokens;
    std::vector<float> selected_logits;
    uint64_t checksum = 0;
};

struct SharedApi {
    void *lib = nullptr;
    Api api = {};
    bool initialized = false;
};

struct SharedRankBuffers {
    RankState ranks[kGpus];
    bool initialized = false;
    uint64_t core_bytes = 0;
};

struct SharedTpRuntime {
    ds4_v100_tp_runtime *rt = nullptr;
    ds4_v100_tp_runtime_report report = {};
    bool initialized = false;
};

struct LayerExpertCache {
    DescriptorBindings bindings;
    PackedExperts gated[kGpus];
    PackedExperts down[kGpus];
    uint64_t bytes = 0;
    bool initialized = false;
};

struct SharedExpertBindings {
    LayerExpertCache layers[43];
    uint64_t bytes = 0;
    bool initialized = false;
};

struct DecodeLoopStats {
    bool enabled = false;
    bool pass = true;
    int steps = 0;
    int slots = 0;
    uint64_t slot_steps = 0;
    uint64_t dense_loaded_bytes = 0;
    uint64_t ep_contribution_bytes = 0;
    uint64_t ep_return_bytes = 0;
    double total_ms = 0.0;
    double ms_per_step = 0.0;
    double tok_s = 0.0;
    double ep_ms_per_step = 0.0;
    double dense_ms_per_step = 0.0;
    double compose_ms_per_step = 0.0;
    double compose_reduce_ms_per_step = 0.0;
    double compose_copy_ms_per_step = 0.0;
    double compose_final_ms_per_step = 0.0;
    double hc_current_input_ms_per_step = 0.0;
    double hc_current_seed_ms_per_step = 0.0;
    double hc_current_attn_mix_ms_per_step = 0.0;
    double hc_current_split_ms_per_step = 0.0;
    double hc_current_gather_ms_per_step = 0.0;
    double hc_current_ffn_router_ms_per_step = 0.0;
    double hc_current_ffn_norm_ms_per_step = 0.0;
    double hc_current_router_select_ms_per_step = 0.0;
    double hc_current_router_d2h_ms_per_step = 0.0;
    double hc_current_route_upload_ms_per_step = 0.0;
    double hc_current_fill_pack_ms_per_step = 0.0;
    double pre_ep_hc_current_ms_per_step = 0.0;
    double pre_ep_attention_projection_ms_per_step = 0.0;
    double pre_ep_compressed_kv_ms_per_step = 0.0;
    double pre_ep_attention_state_ms_per_step = 0.0;
    double pre_ep_typed_history_ms_per_step = 0.0;
    double pre_ep_raw_read_ms_per_step = 0.0;
    double pre_ep_attention_output_ms_per_step = 0.0;
    double pre_ep_post_attention_ffn_input_ms_per_step = 0.0;
    double final_hc_ms_per_step = 0.0;
    int cudagraph_sync_all_calls = 0;
    int cudagraph_event_barrier_calls = 0;
    int cudagraph_rank_stream_syncs = 0;
    int cudagraph_dense_stream_syncs = 0;
    int cudagraph_copy_stream_syncs = 0;
    int cudagraph_capture_attempted = 0;
    int cudagraph_capture_succeeded = 0;
    int cudagraph_capture_error = 0;
    size_t cudagraph_capture_nodes = 0;
    int finite_bad = 0;
    uint64_t checksum = 0;
    bool ep_return_fp16 = false;
    bool fused_compose_sum = false;
    bool dense_hmma_compose = false;
    bool dense_f16_cublas_compose = false;
    bool dense_f16_cache_compose = false;
    bool nccl_reduce_scatter_compose = false;
};

struct HcCurrentInputBreakdown {
    double seed_ms = 0.0;
    double attn_mix_ms = 0.0;
    double split_ms = 0.0;
    double gather_ms = 0.0;
    double ffn_router_ms = 0.0;
    double ffn_norm_ms = 0.0;
    double router_select_ms = 0.0;
    double router_d2h_ms = 0.0;
    double route_upload_ms = 0.0;
    double fill_pack_ms = 0.0;
};

struct PreEpPrefixBreakdown {
    double hc_current_ms = 0.0;
    double attention_projection_ms = 0.0;
    double compressed_kv_ms = 0.0;
    double attention_state_ms = 0.0;
    double typed_history_ms = 0.0;
    double raw_read_ms = 0.0;
    double attention_output_ms = 0.0;
    double post_attention_ffn_input_ms = 0.0;
};

struct Options {
    const char *lib_path = "./build/turbomind-v100/libggml-turbomind.so";
    const char *pack_dir = nullptr;
    const char *contract_path = nullptr;
    const char *tm_index_path = nullptr;
    const char *tokenizer_model_path = nullptr;
    int devices[kGpus] = {0, 1, 2, 3, 4, 5, 6, 7};
    int slots = 32;
    int top_k = 6;
    int layer = 2;
    uint32_t kv_slot = 7;
    uint64_t position = 1024;
    int warmup = 5;
    int iters = 30;
    const char *dense_compute_tensor = nullptr;
    bool dense_compute_all_f8 = false;
    bool dense_compute_all_bf16 = false;
    bool compose_next_hidden = false;
    int decode_steps = 0;
    bool ep_return_fp16 = false;
    bool fuse_compose_sum = false;
    bool dense_hmma_compose = false;
    bool dense_f16_cublas_compose = false;
    bool dense_f16_cache_compose = false;
    bool all_layers = false;
    bool skip_descriptor_checks = false;
    bool skip_predecode_probes = false;
    bool share_tp_runtime = false;
    bool tp_runtime_explicit = false;
    bool tp_runtime_skip_unused_comp_state = false;
    bool share_expert_bindings = true;
    bool overlap_ep_dense = true;
    bool direct_remote_compose = false;
    bool source_copy_schedule = true;
    bool copy_event_compose = false;
    bool compact_route_compose = false;
    bool token_major_all_layers = false;
    bool share_dense_ops = false;
    bool skip_self_compose_copy = true;
    bool multi_copy_streams = false;
    bool nccl_reduce_scatter_compose_gate = false;
    bool serving_bench = false;
    bool skip_decode_checksum = false;
    bool serve_http = false;
    const char *host = "127.0.0.1";
    int port = 18082;
    int max_requests = 0;
    int microbatch_wait_us = 5000;
    bool output_head_gate = false;
    bool output_head_resident_gate = false;
    bool async_output_gate = false;
    bool decode_cudagraph_gate = false;
    bool batched_paged_attn_gate = false;
    bool compact_moe_decode_gate = false;
    bool fused_gated_silu_gate = false;
    bool final_hc_carry_gate = false;
    bool diagnostic_output_head = false;
    bool diagnostic_output_head_lazy_gate = false;
    bool tp_hc_final_expand_gate = false;
    bool tp_hc_current_input_gate = false;
    bool tp_hc_current_input_peer_gather_gate = false;
    bool tp_hc_current_input_nccl_allgather_gate = false;
    bool tp_hc_current_input_stream_sync_gate = false;
    bool tp_hc_current_input_fused_fill_pack_gate = false;
    bool tp_hc_persist_state_gate = false;
    bool model_router_routes = false;
    bool router_cublas_gate = false;
    bool router_hash_fast_gate = false;
    bool gpu_route_plan_gate = false;
    bool route_plan_async_upload_gate = false;
    bool routed_ffn_norm_input_gate = false;
    bool true_shared_ffn_gate = false;
    bool tp_kv_all_slots_gate = false;
    bool reference_hc_reduce_gate = false;
    bool reference_hc_state_guard_gate = false;
    bool true_ds4_attention_residency_gate = false;
    bool true_ds4_attention_projection_gate = false;
    bool true_ds4_attention_state_gate = false;
    bool true_ds4_attention_rope_gate = false;
    bool true_ds4_attention_saturation_audit_gate = false;
    bool true_ds4_attention_kv_norm_reference_gate = false;
    bool true_ds4_attention_raw_read_gate = false;
    bool true_ds4_attention_raw_window_gate = false;
    bool true_ds4_attention_typed_kv_raw_gate = false;
    bool true_ds4_attention_typed_kv_compressed_gate = false;
    bool true_ds4_attention_typed_kv_indexer_gate = false;
    bool true_ds4_attention_typed_kv_history_gate = false;
    bool true_ds4_attention_typed_kv_skip_current_load_gate = false;
    bool true_ds4_attention_typed_kv_skip_raw_store_gate = false;
    bool true_ds4_attention_typed_kv_skip_compressed_store_gate = false;
    bool true_ds4_attention_typed_kv_skip_indexer_store_gate = false;
    bool true_ds4_attention_typed_kv_quiet_gate = false;
    bool true_ds4_attention_typed_kv_batch_rows_gate = false;
    bool true_ds4_attention_typed_kv_stream_sync_gate = false;
    bool fp8_e5m2_kv_gate = false;
    bool true_ds4_attention_output_gate = false;
    bool true_ds4_attention_output_nccl_allgather_gate = false;
    bool true_ds4_post_attention_ffn_input_gate = false;
    bool true_ds4_semantic_skip_stats_gate = false;
    bool true_ds4_compressed_kv_gate = false;
    bool true_ds4_indexer_attention_gate = false;
    bool true_ds4_compressed_kv_direct_input_fill_gate = false;
    bool true_ds4_compressed_kv_dense_event_wait_gate = false;
    bool true_ds4_compressed_kv_skip_dense_stats_gate = false;
    bool true_ds4_compressed_kv_fused_attn_input_fill_gate = false;
    bool true_ds4_compressed_kv_fused_input_fill_gate = false;
    bool true_ds4_compressed_kv_fused_rope_round_gate = false;
    bool true_ds4_compressed_kv_fused_pool_norm_gate = false;
    bool true_ds4_compressed_kv_fused_pool_norm_rope_round_gate = false;
    bool true_ds4_compressed_reference_diff_gate = false;
    bool cuda_profiler_window = false;
    uint32_t true_ds4_attention_raw_valid_rows = 1;
    uint64_t vram_min_free_mib = 0;
    uint64_t nccl_min_free_mib = 0;
    bool vram_report = false;
};

bool tp_ep_profiler_start_if_requested(const Options &opt) {
    if (!opt.cuda_profiler_window) return false;
    const cudaError_t err = cudaProfilerStart();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "tp_ep_cuda_profiler_start_failed\terr\t%s\n",
                     cudaGetErrorString(err));
        return false;
    }
    std::fprintf(stderr, "tp_ep_cuda_profiler_window\tstate\tstart\n");
    return true;
}

int tp_ep_profiler_stop_if_active(bool *active) {
    if (!active || !*active) return 0;
    const cudaError_t err = cudaProfilerStop();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "tp_ep_cuda_profiler_stop_failed\terr\t%s\n",
                     cudaGetErrorString(err));
        *active = false;
        return 1;
    }
    std::fprintf(stderr, "tp_ep_cuda_profiler_window\tstate\tstop\n");
    *active = false;
    return 0;
}

struct TpEpProfilerWindowGuard {
    bool active = false;

    explicit TpEpProfilerWindowGuard(const Options &opt)
        : active(tp_ep_profiler_start_if_requested(opt)) {}

    ~TpEpProfilerWindowGuard() {
        (void)tp_ep_profiler_stop_if_active(&active);
    }
};

void sync_typed_kv_boundary(const Options &opt, RankState ranks[kGpus]) {
    if (opt.decode_cudagraph_gate) return;
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        if (opt.true_ds4_attention_typed_kv_stream_sync_gate) {
            CHECK_CUDA(cudaStreamSynchronize(0));
        } else {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }
}

int ds4_layer_ratio(int layer);

struct BatchedPagedAttnPlan {
    int layer = -1;
    int ratio = 0;
    uint32_t slots = 0;
    uint64_t position = 0;
    uint32_t raw_current_row = 0;
    uint32_t raw_valid_rows = 0;
    uint32_t visible_attn_rows = 0;
    uint32_t selected_attn_rows = 0;
    uint32_t visible_indexer_rows = 0;
    uint32_t pending_attn_reloads = 0;
    uint32_t pending_indexer_reloads = 0;
    uint32_t legacy_history_load_calls = 0;
    uint32_t batch_row_history_load_calls = 0;
    uint32_t target_family_kernels = 0;
};

uint32_t count_pending_batched_attn_reloads(
    const Options &opt, const RankState &r, int layer, bool indexer_rows,
    uint32_t visible_rows) {
    uint32_t pending = 0;
    for (uint32_t row = 0; row < visible_rows; ++row) {
        const uint64_t pos = indexer_rows
            ? r.index_comp_row_position_layers[layer][row]
            : r.attn_comp_row_position_layers[layer][row];
        const bool loaded = indexer_rows
            ? r.index_comp_row_loaded_layers[layer][row]
            : r.attn_comp_row_loaded_layers[layer][row];
        const uint64_t loaded_pos = indexer_rows
            ? r.index_comp_row_loaded_position_layers[layer][row]
            : r.attn_comp_row_loaded_position_layers[layer][row];
        if (opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
            pos == opt.position) {
            continue;
        }
        if (loaded && loaded_pos == pos) {
            continue;
        }
        pending++;
    }
    return pending;
}

BatchedPagedAttnPlan build_batched_paged_attn_plan(
    const Options &opt, const RankState ranks[kGpus], int layer) {
    BatchedPagedAttnPlan plan;
    plan.layer = layer;
    plan.ratio = ds4_layer_ratio(layer);
    plan.slots = (uint32_t)opt.slots;
    plan.position = opt.position;
    plan.raw_current_row = (uint32_t)(opt.position % kRawSwaRows);
    plan.raw_valid_rows =
        std::max(1u, std::min(opt.true_ds4_attention_raw_valid_rows,
                              (uint32_t)kRawSwaRows));
    if (opt.true_ds4_compressed_kv_gate && plan.ratio != 0) {
        plan.visible_attn_rows =
            std::min(ranks[0].attn_comp_rows_written_layers[layer],
                     (uint32_t)kBoundedCompRows);
        plan.selected_attn_rows =
            plan.ratio == 4 && opt.true_ds4_indexer_attention_gate
                ? std::min(plan.visible_attn_rows, (uint32_t)kBoundedCompRows)
                : plan.visible_attn_rows;
    }
    if (opt.true_ds4_indexer_attention_gate && plan.ratio == 4 &&
        plan.visible_attn_rows > 0) {
        plan.visible_indexer_rows =
            std::min(ranks[0].index_comp_rows_written_layers[layer],
                     (uint32_t)kBoundedCompRows);
    }
    plan.pending_attn_reloads = count_pending_batched_attn_reloads(
        opt, ranks[0], layer, false, plan.visible_attn_rows);
    plan.pending_indexer_reloads = count_pending_batched_attn_reloads(
        opt, ranks[0], layer, true, plan.visible_indexer_rows);
    plan.legacy_history_load_calls =
        (plan.pending_attn_reloads + plan.pending_indexer_reloads) *
        (uint32_t)opt.slots;
    plan.batch_row_history_load_calls =
        plan.pending_attn_reloads + plan.pending_indexer_reloads;
    plan.target_family_kernels =
        (opt.true_ds4_attention_raw_read_gate ? 1u : 0u) +
        (plan.visible_attn_rows > 0 ? 1u : 0u) +
        (plan.visible_indexer_rows > 0 ? 1u : 0u);
    return plan;
}

void maybe_log_batched_paged_attn_plan(const Options &opt,
                                       RankState ranks[kGpus],
                                       int layer) {
    if (!opt.batched_paged_attn_gate || layer < 0 || layer >= 43) return;
    const BatchedPagedAttnPlan plan =
        build_batched_paged_attn_plan(opt, ranks, layer);
    const uint32_t key =
        (plan.visible_attn_rows & 0xffu) |
        ((plan.visible_indexer_rows & 0xffu) << 8) |
        ((plan.pending_attn_reloads & 0xffu) << 16) |
        ((plan.pending_indexer_reloads & 0xffu) << 24);
    const uint32_t logged =
        ranks[0].batched_paged_attn_plan_logged_layers[layer];
    if (logged > 0 &&
        ranks[0].batched_paged_attn_plan_last_key_layers[layer] == key) {
        return;
    }
    if (logged >= 8u) return;
    std::printf("tp_ep_batched_paged_attn_plan\tlayer\t%d\tslots\t%u\t"
                "ratio\t%d\tposition\t%llu\traw_current_row\t%u\t"
                "raw_valid_rows\t%u\tvisible_attn_rows\t%u\t"
                "selected_attn_rows\t%u\tvisible_indexer_rows\t%u\t"
                "pending_attn_reloads\t%u\tpending_indexer_reloads\t%u\t"
                "legacy_history_load_calls\t%u\tbatch_row_history_load_calls\t%u\t"
                "target_family_kernels\t%u\tplan_sample\t%u\tPASS\n",
                plan.layer, plan.slots, plan.ratio,
                (unsigned long long)plan.position, plan.raw_current_row,
                plan.raw_valid_rows, plan.visible_attn_rows,
                plan.selected_attn_rows, plan.visible_indexer_rows,
                plan.pending_attn_reloads, plan.pending_indexer_reloads,
                plan.legacy_history_load_calls,
                plan.batch_row_history_load_calls, plan.target_family_kernels,
                logged + 1u);
    for (int rank = 0; rank < kGpus; ++rank) {
        ranks[rank].batched_paged_attn_plan_logged_layers[layer] = logged + 1u;
        ranks[rank].batched_paged_attn_plan_last_key_layers[layer] = key;
    }
}

struct TensorF32Stats {
    int finite_bad = 0;
    size_t first_bad = (size_t)-1;
    float max_abs = 0.0f;
};

struct TensorF32DiffStats {
    int bad = 0;
    size_t first_bad = (size_t)-1;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
};

TensorF32Stats collect_tensor_f32_stats(const float *ptr, size_t elems,
                                        cudaStream_t stream);
TensorF32Stats collect_raw_swa_row_stats(const float *ptr, uint32_t slots,
                                         uint32_t raw_rows, uint32_t raw_row,
                                         uint32_t head_dim,
                                         cudaStream_t stream);
TensorF32DiffStats collect_tensor_f32_diff_stats(const float *a, const float *b,
                                                 size_t elems,
                                                 cudaStream_t stream);
void merge_tensor_stats(TensorF32Stats *dst, const TensorF32Stats &src);
void log_tensor_f32_stats(const char *tag, int layer, int rank_id,
                          const float *ptr, size_t elems, cudaStream_t stream);
bool should_log_routed_semantic_stats(const Options &opt);
bool should_log_reference_hc_window(const Options &opt);

__global__ void checksum_bytes_kernel(const unsigned char *data, uint64_t n,
                                      unsigned long long *out) {
    unsigned long long local = 0;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (uint64_t)blockDim.x * gridDim.x) {
        local += (unsigned long long)data[i] * (unsigned long long)((i % 251u) + 1u);
    }
    atomicAdd(out, local);
}

__global__ void copy_f32_kernel(float *dst, const float *src, uint64_t n) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (uint64_t)blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

__device__ float f8_e8m0_to_f32_dev(uint8_t e) {
    return __uint_as_float(e == 0 ? 0x00400000u : ((uint32_t)e << 23));
}

__device__ float f8_e4m3fn_to_f32_dev(uint8_t x) {
    const uint32_t sign = ((uint32_t)x & 0x80u) << 24;
    const uint32_t ax = (uint32_t)x & 0x7fu;
    if (ax == 0) return __uint_as_float(sign ? 0x80000000u : 0u);
    if (ax == 0x7f) return __uint_as_float(0x7fc00000u);
    const uint32_t exp = ax >> 3;
    const uint32_t man = ax & 0x07u;
    if (exp != 0) {
        return __uint_as_float(sign | ((exp + 120u) << 23) | (man << 20));
    }
    const uint32_t hi = man >= 4u ? 2u : (man >= 2u ? 1u : 0u);
    const uint32_t mant = (man << (23u - hi)) & 0x007fffffu;
    return __uint_as_float(sign | ((118u + hi) << 23) | mant);
}

__device__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ float block_sum_256_f32(float v) {
    __shared__ float warp_sums[8];
    __shared__ float block_sum;
    v = warp_sum_f32(v);
    if ((threadIdx.x & 31u) == 0u) warp_sums[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < 8u ? warp_sums[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32u) v = warp_sum_f32(v);
    if (threadIdx.x == 0u) block_sum = v;
    __syncthreads();
    return block_sum;
}

__device__ float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ float block_max_256_f32(float v) {
    __shared__ float warp_maxes[8];
    __shared__ float block_max;
    v = warp_max_f32(v);
    if ((threadIdx.x & 31u) == 0u) warp_maxes[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < 8u ? warp_maxes[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32u) v = warp_max_f32(v);
    if (threadIdx.x == 0u) block_max = v;
    __syncthreads();
    return block_max;
}

__device__ __half f32_to_half_saturate(float v) {
    if (!isfinite(v)) return __float2half(0.0f);
    v = fminf(kFp16Max, fmaxf(-kFp16Max, v));
    return __float2half(v);
}

__global__ void f8_b128_dense_kernel(float *out,
                                     const uint8_t *weights,
                                     const float *x,
                                     uint32_t rows,
                                     uint32_t cols,
                                     uint32_t row_stride_bytes,
                                     uint32_t slots) {
    const uint32_t row = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (row >= rows || slot >= slots) return;
    const uint8_t *wrow = weights + (uint64_t)row * row_stride_bytes;
    const float *xrow = x + (uint64_t)slot * cols;
    float acc = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const uint8_t *block = wrow + (uint64_t)(c / 128u) * 129ull;
        const float scale = f8_e8m0_to_f32_dev(block[0]);
        const float w = f8_e4m3fn_to_f32_dev(block[1u + (c % 128u)]) * scale;
        acc += w * xrow[c];
    }
    acc = block_sum_256_f32(acc);
    if (threadIdx.x == 0u) out[(uint64_t)slot * rows + row] = acc;
}

__global__ void f8_b128_dense_hmma_m16_kernel(float *out,
                                              const uint8_t *weights,
                                              const float *x,
                                              uint32_t rows,
                                              uint32_t cols,
                                              uint32_t row_stride_bytes,
                                              uint32_t slots) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    enum {
        WARPS_PER_BLOCK = 4,
        TILE_M = 16,
        TILE_N = 16,
        TILE_K = 16,
        ROWS_PER_BLOCK = WARPS_PER_BLOCK * TILE_N,
    };

    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (warp >= WARPS_PER_BLOCK) return;

    const uint32_t row_block = blockIdx.x * ROWS_PER_BLOCK;
    const uint32_t token_block = blockIdx.y * TILE_M;

    __shared__ __half a_sh[TILE_M * TILE_K];
    __shared__ __half b_sh[WARPS_PER_BLOCK * TILE_K * TILE_N];
    __shared__ float c_sh[WARPS_PER_BLOCK * TILE_M * TILE_N];

    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (uint32_t k0 = 0; k0 < cols; k0 += TILE_K) {
        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            const uint32_t token = i >> 4u;
            const uint32_t k = i & 15u;
            const uint32_t global_token = token_block + token;
            float v = 0.0f;
            if (global_token < slots) {
                v = x[(uint64_t)global_token * cols + k0 + k];
            }
            a_sh[i] = __float2half_rn(v);
        }

        for (uint32_t i = tid; i < WARPS_PER_BLOCK * TILE_K * TILE_N; i += blockDim.x) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t out_col = local >> 4u;
            const uint32_t k = local & 15u;
            const uint32_t row = row_block + wtile * TILE_N + out_col;
            float w = 0.0f;
            if (row < rows) {
                const uint32_t col = k0 + k;
                const uint8_t *row_base = weights + (uint64_t)row * row_stride_bytes;
                const uint8_t *block = row_base + (uint64_t)(col >> 7u) * 129ull;
                w = f8_e4m3fn_to_f32_dev(block[1u + (col & 127u)]) *
                    f8_e8m0_to_f32_dev(block[0]);
            }
            b_sh[i] = __float2half_rn(w);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, a_sh, TILE_K);
        wmma::load_matrix_sync(b_frag, b_sh + warp * TILE_K * TILE_N, TILE_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(c_sh + warp * TILE_M * TILE_N,
                            c_frag,
                            TILE_N,
                            wmma::mem_row_major);
    __syncthreads();

    for (uint32_t i = tid; i < WARPS_PER_BLOCK * TILE_M * TILE_N; i += blockDim.x) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t token = local >> 4u;
        const uint32_t out_col = local & 15u;
        const uint32_t global_token = token_block + token;
        const uint32_t row = row_block + wtile * TILE_N + out_col;
        if (global_token < slots && row < rows) {
            out[(uint64_t)global_token * rows + row] =
                c_sh[wtile * TILE_M * TILE_N + local];
        }
    }
#else
    (void)out;
    (void)weights;
    (void)x;
    (void)rows;
    (void)cols;
    (void)row_stride_bytes;
    (void)slots;
#endif
}

__global__ void f8_b128_to_half_kernel(__half *out,
                                       const uint8_t *weights,
                                       uint32_t rows,
                                       uint32_t cols,
                                       uint32_t row_stride_bytes) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)rows * cols;
    if (i >= n) return;
    const uint32_t row = (uint32_t)(i / cols);
    const uint32_t col = (uint32_t)(i - (uint64_t)row * cols);
    const uint8_t *row_base = weights + (uint64_t)row * row_stride_bytes;
    const uint8_t *block = row_base + (uint64_t)(col >> 7u) * 129ull;
    const float w = f8_e4m3fn_to_f32_dev(block[1u + (col & 127u)]) *
                    f8_e8m0_to_f32_dev(block[0]);
    out[i] = __float2half_rn(w);
}

__device__ float bf16_to_f32_dev(uint16_t v) {
    return __uint_as_float((uint32_t)v << 16);
}

__global__ void bf16_to_half_kernel(__half *out, const uint16_t *in, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(bf16_to_f32_dev(in[i]));
}

__global__ void bf16_dense_kernel(float *out,
                                  const uint16_t *weights,
                                  const float *x,
                                  uint32_t rows,
                                  uint32_t cols,
                                  uint32_t row_stride_elements,
                                  uint32_t slots) {
    const uint32_t row = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (row >= rows || slot >= slots) return;
    const uint16_t *wrow = weights + (uint64_t)row * row_stride_elements;
    const float *xrow = x + (uint64_t)slot * cols;
    float acc = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        acc += bf16_to_f32_dev(wrow[c]) * xrow[c];
    }
    acc = block_sum_256_f32(acc);
    if (threadIdx.x == 0u) out[(uint64_t)slot * rows + row] = acc;
}

__global__ void shard_top1_kernel(uint32_t *out_token,
                                  float *out_logit,
                                  const float *logits,
                                  uint32_t rows,
                                  uint32_t shard_base,
                                  uint32_t slots) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots) return;
    const float *row = logits + (uint64_t)slot * rows;
    float best = -3.4028234663852886e+38F;
    uint32_t best_idx = 0;
    for (uint32_t i = threadIdx.x; i < rows; i += blockDim.x) {
        const float v = row[i];
        if (v > best) {
            best = v;
            best_idx = i;
        }
    }
    __shared__ float s_val[256];
    __shared__ uint32_t s_idx[256];
    s_val[threadIdx.x] = best;
    s_idx[threadIdx.x] = best_idx;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other = s_val[threadIdx.x + stride];
            const uint32_t other_idx = s_idx[threadIdx.x + stride];
            if (other > s_val[threadIdx.x]) {
                s_val[threadIdx.x] = other;
                s_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        out_token[slot] = shard_base + s_idx[0];
        out_logit[slot] = s_val[0];
    }
}

__global__ void gather_hc_shard_to_full_kernel(float *full_hc,
                                               const float *shard_hc,
                                               int rank,
                                               uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t shard_cols = kHidden / kGpus;
    const uint64_t elems = (uint64_t)slots * 4ull * shard_cols;
    if (i >= elems) return;
    const uint32_t local_h = (uint32_t)(i % shard_cols);
    const uint32_t row = (uint32_t)((i / shard_cols) & 3ull);
    const uint32_t slot = (uint32_t)(i / (4ull * shard_cols));
    const uint64_t dst =
        ((uint64_t)slot * 4ull + (uint64_t)row) * (uint64_t)kHidden +
        (uint64_t)rank * shard_cols + local_h;
    full_hc[dst] = shard_hc[i];
}

__global__ void seed_hc_shard_from_token_embedding_kernel(float *shard_hc,
                                                          const uint16_t *slot_rows,
                                                          uint32_t slots,
                                                          int rank) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t shard_cols = kHidden / kGpus;
    const uint64_t elems = (uint64_t)slots * 4ull * shard_cols;
    if (i >= elems) return;
    const uint32_t local_h = (uint32_t)(i % shard_cols);
    const uint32_t slot = (uint32_t)(i / (4ull * shard_cols));
    const uint32_t hidden_col = (uint32_t)rank * shard_cols + local_h;
    shard_hc[i] = bf16_to_f32_dev(slot_rows[(uint64_t)slot * kHidden + hidden_col]);
}

__global__ void synthetic_hc_kernel(float *hc, uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * 4ull * (uint64_t)kHidden;
    if (i >= n) return;
    const uint32_t col = (uint32_t)(i % kHidden);
    const uint32_t row = (uint32_t)((i / kHidden) & 3ull);
    const uint32_t slot = (uint32_t)(i / (4ull * (uint64_t)kHidden));
    const int m = (int)((slot * 97u + row * 31u + col * 17u) % 257u);
    hc[i] = ((float)m - 128.0f) * 0.0025f;
}

__global__ void rms_norm_plain_rows_kernel(float *out,
                                           const float *in,
                                           uint32_t cols,
                                           uint32_t rows,
                                           float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *src = in + (uint64_t)row * cols;
    float *dst = out + (uint64_t)row * cols;
    float sum = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = src[c];
        sum += v * v;
    }
    sum = block_sum_256_f32(sum);
    const float scale = rsqrtf(sum / (float)cols + eps);
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        dst[c] = src[c] * scale;
    }
}

__global__ void rms_norm_plain_rows_stable_kernel(float *out,
                                                  const float *in,
                                                  uint32_t cols,
                                                  uint32_t rows,
                                                  float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *src = in + (uint64_t)row * cols;
    float *dst = out + (uint64_t)row * cols;
    float local_max = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = src[c];
        if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
    }
    const float max_abs = block_max_256_f32(local_max);
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
            const float v = src[c];
            if (isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    sum = block_sum_256_f32(sum);
    float scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        scale = rsqrtf(sum / (float)cols + eps / (max_abs * max_abs)) / max_abs;
    }
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = src[c];
        dst[c] = isfinite(v) ? v * scale : 0.0f;
    }
}

__global__ void f32_dense_kernel(float *out,
                                 const float *weights,
                                 const float *x,
                                 uint32_t rows,
                                 uint32_t cols,
                                 uint32_t slots) {
    const uint32_t row = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (row >= rows || slot >= slots) return;
    const float *wrow = weights + (uint64_t)row * cols;
    const float *xrow = x + (uint64_t)slot * cols;
    float acc = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        acc += wrow[c] * xrow[c];
    }
    acc = block_sum_256_f32(acc);
    if (threadIdx.x == 0u) out[(uint64_t)slot * rows + row] = acc;
}

__global__ void output_hc_weights_rows_kernel(float *out,
                                              const float *pre,
                                              const float *scale,
                                              const float *base,
                                              uint32_t rows) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = rows * 4u;
    if (i >= n) return;
    const uint32_t h = i & 3u;
    const float z = pre[i] * scale[0] + base[h];
    out[i] = 1.0f / (1.0f + expf(-z)) + 1.0e-6f;
}

__global__ void hc_weighted_sum_rows_kernel(float *out,
                                            const float *hc,
                                            const float *weights,
                                            uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)kHidden;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / kHidden);
    const uint32_t col = (uint32_t)(i % kHidden);
    const uint64_t base = (uint64_t)slot * 4ull * (uint64_t)kHidden;
    const float w0 = weights[(uint64_t)slot * 4ull + 0ull];
    const float w1 = weights[(uint64_t)slot * 4ull + 1ull];
    const float w2 = weights[(uint64_t)slot * 4ull + 2ull];
    const float w3 = weights[(uint64_t)slot * 4ull + 3ull];
    out[i] = hc[base + col] * w0 +
             hc[base + (uint64_t)kHidden + col] * w1 +
             hc[base + 2ull * (uint64_t)kHidden + col] * w2 +
             hc[base + 3ull * (uint64_t)kHidden + col] * w3;
}

__global__ void hc_weighted_sum_shard_kernel(float *out,
                                             const float *hc,
                                             const float *weights,
                                             uint32_t slots,
                                             int reference_reduce) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t shard_cols = kHidden / kGpus;
    const uint64_t n = (uint64_t)slots * (uint64_t)shard_cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / shard_cols);
    const uint32_t local_h = (uint32_t)(i % shard_cols);
    const uint64_t base = (uint64_t)slot * 4ull * (uint64_t)shard_cols;
    const float w0 = weights[(uint64_t)slot * kHcMix + 0ull];
    const float w1 = weights[(uint64_t)slot * kHcMix + 1ull];
    const float w2 = weights[(uint64_t)slot * kHcMix + 2ull];
    const float w3 = weights[(uint64_t)slot * kHcMix + 3ull];
    float v = hc[base + local_h] * w0 +
              hc[base + (uint64_t)shard_cols + local_h] * w1 +
              hc[base + 2ull * (uint64_t)shard_cols + local_h] * w2 +
              hc[base + 3ull * (uint64_t)shard_cols + local_h] * w3;
    if (!isfinite(v)) v = 0.0f;
    if (!reference_reduce) {
        v = fminf(1.0f, fmaxf(-1.0f, v * 0.125f));
    }
    out[i] = v;
}

__global__ void gather_current_shard_to_full_kernel(float *full,
                                                    const float *shard,
                                                    int rank,
                                                    uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t shard_cols = kHidden / kGpus;
    const uint64_t n = (uint64_t)slots * (uint64_t)shard_cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / shard_cols);
    const uint32_t local_h = (uint32_t)(i % shard_cols);
    full[(uint64_t)slot * kHidden + (uint64_t)rank * shard_cols + local_h] = shard[i];
}

__global__ void gather_current_shards_to_full8_kernel(float *full,
                                                      const float *shard0,
                                                      const float *shard1,
                                                      const float *shard2,
                                                      const float *shard3,
                                                      const float *shard4,
                                                      const float *shard5,
                                                      const float *shard6,
                                                      const float *shard7,
                                                      uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)kHidden;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / kHidden);
    const uint32_t col = (uint32_t)(i % kHidden);
    const uint32_t shard_cols = kHidden / kGpus;
    const uint32_t rank = col / shard_cols;
    const uint32_t local_h = col - rank * shard_cols;
    const uint64_t src_i = (uint64_t)slot * shard_cols + local_h;
    const float *src = shard0;
    if (rank == 1u) src = shard1;
    else if (rank == 2u) src = shard2;
    else if (rank == 3u) src = shard3;
    else if (rank == 4u) src = shard4;
    else if (rank == 5u) src = shard5;
    else if (rank == 6u) src = shard6;
    else if (rank == 7u) src = shard7;
    full[i] = src[src_i];
}

__global__ void rank_major_current_shards_to_slot_major_kernel(
    float *full,
    const float *rank_major,
    uint32_t shard_cols,
    uint32_t ranks,
    uint32_t slots) {
    const uint64_t cols = (uint64_t)shard_cols * (uint64_t)ranks;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / cols);
    const uint32_t col = (uint32_t)(i % cols);
    const uint32_t rank = col / shard_cols;
    const uint32_t local_col = col - rank * shard_cols;
    const uint64_t src_i =
        ((uint64_t)rank * (uint64_t)slots + (uint64_t)slot) *
            (uint64_t)shard_cols +
        (uint64_t)local_col;
    full[i] = rank_major[src_i];
}

__global__ void gather_dense_shard_to_full_kernel(float *full,
                                                  const float *shard,
                                                  int rank,
                                                  uint32_t shard_cols,
                                                  uint32_t total_cols,
                                                  uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)shard_cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / shard_cols);
    const uint32_t local_col = (uint32_t)(i % shard_cols);
    full[(uint64_t)slot * total_cols + (uint64_t)rank * shard_cols + local_col] =
        shard[i];
}

__global__ void fill_dense_input_half_from_tensor_kernel(__half *dst,
                                                         const float *src,
                                                         uint32_t cols,
                                                         uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)cols;
    if (i >= n) return;
    dst[i] = f32_to_half_saturate(src[i]);
}

__global__ void fill_dense_input_half_from_rank_major_shards_kernel(
    __half *dst,
    const float *rank_major,
    uint32_t shard_cols,
    uint32_t ranks,
    uint32_t slots) {
    const uint64_t cols = (uint64_t)shard_cols * (uint64_t)ranks;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / cols);
    const uint32_t col = (uint32_t)(i % cols);
    const uint32_t rank = col / shard_cols;
    const uint32_t local_col = col - rank * shard_cols;
    const uint64_t src_i =
        ((uint64_t)rank * (uint64_t)slots + (uint64_t)slot) *
            (uint64_t)shard_cols +
        (uint64_t)local_col;
    dst[i] = f32_to_half_saturate(rank_major[src_i]);
}

__device__ float e4m3fn_quant_dequant_dev(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (f8_e4m3fn_to_f32_dev((uint8_t)mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - f8_e4m3fn_to_f32_dev((uint8_t)best));
        const float next_diff = fabsf(ax - f8_e4m3fn_to_f32_dev((uint8_t)(best + 1)));
        if (next_diff < best_diff ||
            (next_diff == best_diff && (((best + 1) & 1) == 0) && ((best & 1) != 0))) {
            best++;
        }
    }
    return sign * f8_e4m3fn_to_f32_dev((uint8_t)best);
}

__global__ void head_rms_norm_local_heads_kernel(float *x,
                                                 uint32_t slots,
                                                 uint32_t local_heads,
                                                 uint32_t head_dim,
                                                 float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= slots * local_heads) return;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        xr[i] *= scale;
    }
}

__global__ void kv_fp8_round_store_raw_swa_kernel(float *raw_swa,
                                                  const float *kv,
                                                  uint32_t slots,
                                                  uint32_t raw_rows,
                                                  uint32_t raw_row,
                                                  uint32_t head_dim,
                                                  uint32_t n_rot) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)head_dim;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / head_dim);
    const uint32_t col = (uint32_t)(i % head_dim);
    const uint32_t n_nope = head_dim - n_rot;
    float v = kv[i];
    if (col < n_nope) {
        const uint32_t block0 = (col / 64u) * 64u;
        float amax = 0.0f;
        for (uint32_t j = 0; j < 64u; ++j) {
            amax = fmaxf(amax, fabsf(kv[(uint64_t)slot * head_dim + block0 + j]));
        }
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        float q = v / scale;
        q = fminf(448.0f, fmaxf(-448.0f, q));
        v = e4m3fn_quant_dequant_dev(q) * scale;
    }
    v = __half2float(f32_to_half_saturate(v));
    raw_swa[((uint64_t)slot * raw_rows + raw_row) * head_dim + col] = v;
}

__device__ float rope_yarn_ramp_tp_dev(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__global__ void rope_tail_rows_kernel(float *x,
                                      uint32_t rows,
                                      uint32_t head_dim,
                                      uint32_t n_rot,
                                      uint32_t pos,
                                      uint32_t n_ctx_orig,
                                      int inverse,
                                      float freq_base,
                                      float freq_scale,
                                      float ext_factor,
                                      float attn_factor,
                                      float beta_fast,
                                      float beta_slow) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || n_rot > head_dim || (n_rot & 1u)) return;
    float *xr = x + (uint64_t)row * head_dim;
    const uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot *
                       logf((float)n_ctx_orig /
                            (beta_fast * 2.0f * (float)M_PI)) /
                       denom);
        corr1 = ceilf((float)n_rot *
                      logf((float)n_ctx_orig /
                           (beta_slow * 2.0f * (float)M_PI)) /
                      denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    float *tail = xr + n_nope;
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2u; pair += blockDim.x) {
        const uint32_t i = pair * 2u;
        const float theta_extrap =
            (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix =
                rope_yarn_ramp_tp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        if (inverse) s = -s;
        const float x0 = tail[i + 0];
        const float x1 = tail[i + 1];
        tail[i + 0] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
    }
}

__global__ void attention_raw_swa_one_row_kernel(float *out_heads,
                                                 const float *q_heads,
                                                 const float *raw_swa,
                                                 const float *sinks,
                                                 uint32_t slots,
                                                 uint32_t local_heads,
                                                 uint32_t head_dim,
                                                 uint32_t raw_rows,
                                                 uint32_t raw_row) {
    const uint32_t row = blockIdx.x;
    if (row >= slots * local_heads) return;
    const uint32_t slot = row / local_heads;
    const uint32_t local_head = row % local_heads;
    const float *q = q_heads + (uint64_t)row * head_dim;
    const float *kv =
        raw_swa + ((uint64_t)slot * raw_rows + raw_row) * (uint64_t)head_dim;
    float dot = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        dot += q[d] * kv[d];
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float score = partial[0] * rsqrtf((float)head_dim);
    const float sink = sinks[local_head];
    const float max_s = fmaxf(score, sink);
    const float row_w = expf(score - max_s);
    const float denom = row_w + expf(sink - max_s);
    const float scale = row_w / denom;
    float *out = out_heads + (uint64_t)row * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[d] = kv[d] * scale;
    }
}

__global__ void attention_raw_swa_window_kernel(float *out_heads,
                                                const float *q_heads,
                                                const float *raw_swa,
                                                const float *sinks,
                                                uint32_t slots,
                                                uint32_t local_heads,
                                                uint32_t head_dim,
                                                uint32_t raw_rows,
                                                uint32_t raw_row,
                                                uint32_t valid_rows) {
    const uint32_t row = blockIdx.x;
    if (row >= slots * local_heads || valid_rows == 0 || valid_rows > raw_rows) return;
    const uint32_t slot = row / local_heads;
    const uint32_t local_head = row % local_heads;
    const float *q = q_heads + (uint64_t)row * head_dim;
    __shared__ float partial[256];
    __shared__ float scores[128];

    float max_s = sinks[local_head];
    for (uint32_t i = 0; i < valid_rows; ++i) {
        const uint32_t history_offset = valid_rows - 1u - i;
        const uint32_t rr = (raw_row + raw_rows - history_offset) % raw_rows;
        const float *kv =
            raw_swa + ((uint64_t)slot * raw_rows + rr) * (uint64_t)head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += q[d] * kv[d];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        const float score = partial[0] * rsqrtf((float)head_dim);
        if (threadIdx.x == 0) {
            scores[i] = score;
        }
        max_s = fmaxf(max_s, score);
        __syncthreads();
    }

    float denom = expf(sinks[local_head] - max_s);
    for (uint32_t i = 0; i < valid_rows; ++i) {
        denom += expf(scores[i] - max_s);
    }
    float *out = out_heads + (uint64_t)row * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < valid_rows; ++i) {
            const uint32_t history_offset = valid_rows - 1u - i;
            const uint32_t rr = (raw_row + raw_rows - history_offset) % raw_rows;
            const float *kv =
                raw_swa + ((uint64_t)slot * raw_rows + rr) * (uint64_t)head_dim;
            const float w = expf(scores[i] - max_s) / denom;
            acc += kv[d] * w;
        }
        out[d] = acc;
    }
}

__global__ void compressor_store_slots_kernel(const float *kv,
                                              const float *score,
                                              float *state_kv,
                                              float *state_score,
                                              const float *ape,
                                              uint32_t slots,
                                              uint32_t head_dim,
                                              uint32_t ratio,
                                              uint32_t pos,
                                              uint32_t max_state_rows,
                                              uint32_t max_width) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * width;
    if (i >= n || ratio == 0u || width > max_width) return;
    const uint32_t slot = (uint32_t)(i / width);
    const uint32_t j = (uint32_t)(i - (uint64_t)slot * width);
    const uint32_t pos_mod = pos % ratio;
    const uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    if (dst_row >= max_state_rows) return;
    const uint64_t dst =
        ((uint64_t)slot * max_state_rows + dst_row) * (uint64_t)max_width + j;
    state_kv[dst] = kv[(uint64_t)slot * width + j];
    state_score[dst] = score[(uint64_t)slot * width + j] +
                       (ape ? ape[(uint64_t)pos_mod * width + j] : 0.0f);
}

__global__ void compressor_pool_emit_slots_kernel(float *rows,
                                                  const float *state_kv,
                                                  const float *state_score,
                                                  uint32_t slots,
                                                  uint32_t head_dim,
                                                  uint32_t ratio,
                                                  uint32_t comp_row,
                                                  uint32_t row_cap,
                                                  uint32_t max_state_rows,
                                                  uint32_t max_width) {
    const uint32_t slot = blockIdx.y;
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots || d >= head_dim || comp_row >= row_cap || ratio == 0u) return;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    if (width > max_width) return;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    const uint64_t slot_base = (uint64_t)slot * max_state_rows * (uint64_t)max_width;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4u; ++r) {
            vals[n_cand] = state_kv[slot_base + (uint64_t)r * max_width + d];
            scores[n_cand] = state_score[slot_base + (uint64_t)r * max_width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint64_t off =
                slot_base + (uint64_t)(ratio + r) * max_width + head_dim + d;
            vals[n_cand] = state_kv[off];
            scores[n_cand] = state_score[off];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        for (uint32_t r = 0; r < ratio && r < 128u; ++r) {
            vals[n_cand] = state_kv[slot_base + (uint64_t)r * max_width + d];
            scores[n_cand] = state_score[slot_base + (uint64_t)r * max_width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f;
    float acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; ++i) {
        const float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    rows[((uint64_t)slot * row_cap + comp_row) * (uint64_t)head_dim + d] =
        den != 0.0f && isfinite(acc) ? acc / den : 0.0f;
}

__global__ void compressor_norm_emit_slots_kernel(float *rows,
                                                  const float *weight,
                                                  uint32_t slots,
                                                  uint32_t head_dim,
                                                  uint32_t comp_row,
                                                  uint32_t row_cap,
                                                  float eps) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || comp_row >= row_cap) return;
    float *row = rows + ((uint64_t)slot * row_cap + comp_row) * (uint64_t)head_dim;
    float local_max = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = row[d];
        if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
    }
    const float max_abs = block_max_256_f32(local_max);
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const float v = row[d];
            if (isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    sum = block_sum_256_f32(sum);
    float scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        scale = rsqrtf(sum / (float)head_dim + eps / (max_abs * max_abs)) / max_abs;
    }
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = row[d];
        const float y = isfinite(v) ? v * scale * weight[d] : 0.0f;
        row[d] = isfinite(y) ? y : 0.0f;
    }
}

__device__ float compressor_pool_value_one_dim(const float *state_kv,
                                               const float *state_score,
                                               uint32_t head_dim,
                                               uint32_t ratio,
                                               uint32_t max_state_rows,
                                               uint32_t max_width,
                                               uint64_t slot_base,
                                               uint32_t d) {
    float max_s = -INFINITY;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4u; ++r) {
            max_s = fmaxf(max_s,
                          state_score[slot_base + (uint64_t)r * max_width + d]);
        }
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint64_t off =
                slot_base + (uint64_t)(ratio + r) * max_width + head_dim + d;
            max_s = fmaxf(max_s, state_score[off]);
        }
    } else {
        for (uint32_t r = 0; r < ratio && r < 128u; ++r) {
            max_s = fmaxf(max_s,
                          state_score[slot_base + (uint64_t)r * max_width + d]);
        }
    }

    float den = 0.0f;
    float acc = 0.0f;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint64_t off = slot_base + (uint64_t)r * max_width + d;
            const float w = expf(state_score[off] - max_s);
            den += w;
            acc += state_kv[off] * w;
        }
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint64_t off =
                slot_base + (uint64_t)(ratio + r) * max_width + head_dim + d;
            const float w = expf(state_score[off] - max_s);
            den += w;
            acc += state_kv[off] * w;
        }
    } else {
        for (uint32_t r = 0; r < ratio && r < 128u; ++r) {
            const uint64_t off = slot_base + (uint64_t)r * max_width + d;
            const float w = expf(state_score[off] - max_s);
            den += w;
            acc += state_kv[off] * w;
        }
    }
    return den != 0.0f && isfinite(acc) ? acc / den : 0.0f;
}

__global__ void compressor_pool_norm_emit_slots_kernel(float *rows,
                                                       const float *state_kv,
                                                       const float *state_score,
                                                       const float *weight,
                                                       uint32_t slots,
                                                       uint32_t head_dim,
                                                       uint32_t ratio,
                                                       uint32_t comp_row,
                                                       uint32_t row_cap,
                                                       uint32_t max_state_rows,
                                                       uint32_t max_width,
                                                       float eps) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || head_dim > 512u || comp_row >= row_cap || ratio == 0u) {
        return;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    if (width > max_width) return;

    __shared__ float pooled[512];
    const uint64_t slot_base =
        (uint64_t)slot * max_state_rows * (uint64_t)max_width;
    float local_max = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = compressor_pool_value_one_dim(
            state_kv, state_score, head_dim, ratio, max_state_rows, max_width,
            slot_base, d);
        pooled[d] = v;
        if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
    }
    __syncthreads();

    const float max_abs = block_max_256_f32(local_max);
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const float v = pooled[d];
            if (isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    sum = block_sum_256_f32(sum);
    float scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        scale = rsqrtf(sum / (float)head_dim + eps / (max_abs * max_abs)) / max_abs;
    }

    float *row = rows + ((uint64_t)slot * row_cap + comp_row) *
                           (uint64_t)head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float y = isfinite(pooled[d]) ? pooled[d] * scale * weight[d] : 0.0f;
        row[d] = isfinite(y) ? y : 0.0f;
    }
}

__global__ void compressor_pool_norm_rope_round_emit_slots_kernel(
    float *rows,
    const float *state_kv,
    const float *state_score,
    const float *weight,
    uint32_t slots,
    uint32_t head_dim,
    uint32_t ratio,
    uint32_t comp_row,
    uint32_t row_cap,
    uint32_t max_state_rows,
    uint32_t max_width,
    float eps,
    uint32_t n_rot,
    uint32_t pos,
    uint32_t n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || head_dim > 512u || comp_row >= row_cap ||
        ratio == 0u || n_rot > head_dim || (n_rot & 1u)) {
        return;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    if (width > max_width) return;

    __shared__ float normalized[512];
    const uint64_t slot_base =
        (uint64_t)slot * max_state_rows * (uint64_t)max_width;
    float local_max = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = compressor_pool_value_one_dim(
            state_kv, state_score, head_dim, ratio, max_state_rows, max_width,
            slot_base, d);
        normalized[d] = v;
        if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
    }
    __syncthreads();

    const float max_abs = block_max_256_f32(local_max);
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const float v = normalized[d];
            if (isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    sum = block_sum_256_f32(sum);
    float scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        scale = rsqrtf(sum / (float)head_dim + eps / (max_abs * max_abs)) / max_abs;
    }
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = normalized[d];
        const float y = isfinite(v) ? v * scale * weight[d] : 0.0f;
        normalized[d] = isfinite(y) ? y : 0.0f;
    }
    __syncthreads();

    float *row = rows + ((uint64_t)slot * row_cap + comp_row) *
                           (uint64_t)head_dim;
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t d = threadIdx.x; d < n_nope; d += blockDim.x) {
        row[d] = __half2float(f32_to_half_saturate(normalized[d]));
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot *
                       logf((float)n_ctx_orig /
                            (beta_fast * 2.0f * (float)M_PI)) /
                       denom);
        corr1 = ceilf((float)n_rot *
                      logf((float)n_ctx_orig /
                           (beta_slow * 2.0f * (float)M_PI)) /
                      denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2u; pair += blockDim.x) {
        const uint32_t i = pair * 2u;
        const float theta_extrap =
            (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix =
                rope_yarn_ramp_tp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        const float s = sinf(theta) * mscale;
        const float x0 = normalized[n_nope + i + 0];
        const float x1 = normalized[n_nope + i + 1];
        row[n_nope + i + 0] = __half2float(f32_to_half_saturate(x0 * c - x1 * s));
        row[n_nope + i + 1] = __half2float(f32_to_half_saturate(x0 * s + x1 * c));
    }
}

__global__ void rope_tail_comp_emit_slots_kernel(float *rows,
                                                 uint32_t slots,
                                                 uint32_t head_dim,
                                                 uint32_t n_rot,
                                                 uint32_t comp_row,
                                                 uint32_t row_cap,
                                                 uint32_t pos,
                                                 uint32_t n_ctx_orig,
                                                 float freq_base,
                                                 float freq_scale,
                                                 float ext_factor,
                                                 float attn_factor,
                                                 float beta_fast,
                                                 float beta_slow) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || comp_row >= row_cap || n_rot > head_dim || (n_rot & 1u)) return;
    float *xr = rows + ((uint64_t)slot * row_cap + comp_row) * (uint64_t)head_dim;
    const uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot *
                       logf((float)n_ctx_orig /
                            (beta_fast * 2.0f * (float)M_PI)) /
                       denom);
        corr1 = ceilf((float)n_rot *
                      logf((float)n_ctx_orig /
                           (beta_slow * 2.0f * (float)M_PI)) /
                      denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    float *tail = xr + n_nope;
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2u; pair += blockDim.x) {
        const uint32_t i = pair * 2u;
        const float theta_extrap =
            (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix =
                rope_yarn_ramp_tp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        const float s = sinf(theta) * mscale;
        const float x0 = tail[i + 0];
        const float x1 = tail[i + 1];
        tail[i + 0] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
    }
}

__global__ void rope_tail_round_comp_emit_slots_kernel(float *rows,
                                                       uint32_t slots,
                                                       uint32_t head_dim,
                                                       uint32_t n_rot,
                                                       uint32_t comp_row,
                                                       uint32_t row_cap,
                                                       uint32_t pos,
                                                       uint32_t n_ctx_orig,
                                                       float freq_base,
                                                       float freq_scale,
                                                       float ext_factor,
                                                       float attn_factor,
                                                       float beta_fast,
                                                       float beta_slow) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || comp_row >= row_cap || n_rot > head_dim ||
        (n_rot & 1u)) {
        return;
    }
    float *xr = rows + ((uint64_t)slot * row_cap + comp_row) *
                           (uint64_t)head_dim;
    const uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot *
                       logf((float)n_ctx_orig /
                            (beta_fast * 2.0f * (float)M_PI)) /
                       denom);
        corr1 = ceilf((float)n_rot *
                      logf((float)n_ctx_orig /
                           (beta_slow * 2.0f * (float)M_PI)) /
                      denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    for (uint32_t d = threadIdx.x; d < n_nope; d += blockDim.x) {
        xr[d] = __half2float(f32_to_half_saturate(xr[d]));
    }
    float *tail = xr + n_nope;
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2u; pair += blockDim.x) {
        const uint32_t i = pair * 2u;
        const float theta_extrap =
            (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix =
                rope_yarn_ramp_tp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        const float s = sinf(theta) * mscale;
        const float x0 = tail[i + 0];
        const float x1 = tail[i + 1];
        tail[i + 0] = __half2float(f32_to_half_saturate(x0 * c - x1 * s));
        tail[i + 1] = __half2float(f32_to_half_saturate(x0 * s + x1 * c));
    }
}

__global__ void round_comp_emit_slots_kernel(float *rows,
                                             uint32_t slots,
                                             uint32_t head_dim,
                                             uint32_t comp_row,
                                             uint32_t row_cap) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * head_dim;
    if (i >= n || comp_row >= row_cap) return;
    const uint32_t slot = (uint32_t)(i / head_dim);
    const uint32_t d = (uint32_t)(i - (uint64_t)slot * head_dim);
    float *row = rows + ((uint64_t)slot * row_cap + comp_row) * (uint64_t)head_dim;
    row[d] = __half2float(f32_to_half_saturate(row[d]));
}

__global__ void pack_comp_row_kernel(float *dst,
                                     const float *rows,
                                     uint32_t slots,
                                     uint32_t head_dim,
                                     uint32_t comp_row,
                                     uint32_t row_cap) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * head_dim;
    if (i >= n || comp_row >= row_cap) return;
    const uint32_t slot = (uint32_t)(i / head_dim);
    const uint32_t d = (uint32_t)(i - (uint64_t)slot * head_dim);
    dst[i] = rows[((uint64_t)slot * row_cap + comp_row) * (uint64_t)head_dim + d];
}

__global__ void pack_indexer_score_column_kernel(float *dst,
                                                 const float *scores,
                                                 uint32_t slots,
                                                 uint32_t top_k,
                                                 uint32_t column) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots || column >= top_k) return;
    dst[slot] = scores[(uint64_t)slot * top_k + column];
}

__global__ void compressor_shift_ratio4_slots_kernel(float *state_kv,
                                                     float *state_score,
                                                     uint32_t slots,
                                                     uint32_t width,
                                                     uint32_t max_state_rows,
                                                     uint32_t max_width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t half = 4ull * width;
    const uint64_t n = (uint64_t)slots * half;
    if (i >= n || width > max_width || max_state_rows < 8u) return;
    const uint32_t slot = (uint32_t)(i / half);
    const uint32_t j = (uint32_t)(i - (uint64_t)slot * half);
    const uint64_t base = (uint64_t)slot * max_state_rows * (uint64_t)max_width;
    const float v = state_kv[base + half + j];
    const float s = state_score[base + half + j];
    state_kv[base + j] = v;
    state_score[base + j] = s;
    state_kv[base + half + j] = v;
    state_score[base + half + j] = s;
}

__global__ void seed_single_topk_kernel(float *scores,
                                        uint32_t *topk,
                                        uint32_t slots,
                                        uint32_t top_k) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots) return;
    if (threadIdx.x == 0u) scores[slot] = 0.0f;
    for (uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
        topk[(uint64_t)slot * top_k + i] = 0u;
    }
}

__global__ void indexer_score_row0_slots_kernel(float *scores,
                                                uint32_t *topk,
                                                const float *q,
                                                const float *weights,
                                                const float *index_comp_rows,
                                                uint32_t slots,
                                                uint32_t comp_row,
                                                uint32_t row_cap,
                                                uint32_t top_k,
                                                float scale) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || comp_row >= row_cap) return;
    const float *krow =
        index_comp_rows + ((uint64_t)slot * row_cap + comp_row) *
                              (uint64_t)kIndexerHeadDim;
    __shared__ float partial[256];
    float total = 0.0f;
    for (uint32_t h = 0; h < kIndexerHead; ++h) {
        const float *qh =
            q + ((uint64_t)slot * kIndexerHead + h) * (uint64_t)kIndexerHeadDim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < kIndexerHeadDim; d += blockDim.x) {
            dot += qh[d] * krow[d];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0u) {
            total += fmaxf(partial[0], 0.0f) *
                     weights[(uint64_t)slot * kIndexerHead + h];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) scores[slot] = total * scale;
    for (uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
        topk[(uint64_t)slot * top_k + i] = 0u;
    }
}

__global__ void indexer_score_bounded_rows_slots_kernel(float *scores,
                                                        uint32_t *topk,
                                                        const float *q,
                                                        const float *weights,
                                                        const float *index_comp_rows,
                                                        uint32_t slots,
                                                        uint32_t visible_rows,
                                                        uint32_t row_cap,
                                                        uint32_t top_k,
                                                        float scale) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || visible_rows == 0u || visible_rows > row_cap) return;
    __shared__ float partial[256];
    for (uint32_t row = 0; row < visible_rows; ++row) {
        const float *krow =
            index_comp_rows + ((uint64_t)slot * row_cap + row) *
                                  (uint64_t)kIndexerHeadDim;
        float total = 0.0f;
        for (uint32_t h = 0; h < kIndexerHead; ++h) {
            const float *qh =
                q + ((uint64_t)slot * kIndexerHead + h) *
                        (uint64_t)kIndexerHeadDim;
            float dot = 0.0f;
            for (uint32_t d = threadIdx.x; d < kIndexerHeadDim; d += blockDim.x) {
                dot += qh[d] * krow[d];
            }
            partial[threadIdx.x] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0u) {
                total += fmaxf(partial[0], 0.0f) *
                         weights[(uint64_t)slot * kIndexerHead + h];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0u) {
            scores[(uint64_t)slot * top_k + row] = total * scale;
            topk[(uint64_t)slot * top_k + row] = row;
        }
    }
    for (uint32_t i = visible_rows + threadIdx.x; i < top_k; i += blockDim.x) {
        scores[(uint64_t)slot * top_k + i] = 0.0f;
        topk[(uint64_t)slot * top_k + i] = 0u;
    }
}

__global__ void attention_raw_compressed_window_kernel(float *out_heads,
                                                       const float *q_heads,
                                                       const float *raw_swa,
                                                       const float *comp_rows,
                                                       const uint32_t *topk,
                                                       const float *sinks,
                                                       uint32_t slots,
                                                       uint32_t local_heads,
                                                       uint32_t head_dim,
                                                       uint32_t raw_rows,
                                                       uint32_t raw_row,
                                                       uint32_t valid_raw_rows,
                                                       uint32_t visible_comp_rows,
                                                       uint32_t selected_comp_rows,
                                                       uint32_t comp_row_cap,
                                                       uint32_t top_k) {
    const uint32_t row = blockIdx.x;
    if (row >= slots * local_heads || valid_raw_rows == 0u ||
        valid_raw_rows > raw_rows) return;
    const uint32_t slot = row / local_heads;
    const uint32_t local_head = row % local_heads;
    const float *q = q_heads + (uint64_t)row * head_dim;
    __shared__ float partial[256];
    __shared__ float scores[kRawSwaRows + kBoundedCompRows];
    __shared__ uint32_t comp_index[kBoundedCompRows];

    uint32_t comp_count = selected_comp_rows;
    if (comp_count > visible_comp_rows) comp_count = visible_comp_rows;
    if (comp_count > comp_row_cap) comp_count = comp_row_cap;
    if (comp_count > (uint32_t)kBoundedCompRows) comp_count = (uint32_t)kBoundedCompRows;
    if (comp_count > 0u) {
        for (uint32_t i = threadIdx.x; i < comp_count; i += blockDim.x) {
            uint32_t idx = topk && i < top_k ? topk[(uint64_t)slot * top_k + i] : i;
            if (idx >= visible_comp_rows || idx >= comp_row_cap) idx = 0u;
            comp_index[i] = idx;
        }
    }
    __syncthreads();

    float max_s = sinks[local_head];
    for (uint32_t i = 0; i < valid_raw_rows; ++i) {
        const uint32_t history_offset = valid_raw_rows - 1u - i;
        const uint32_t rr = (raw_row + raw_rows - history_offset) % raw_rows;
        const float *kv =
            raw_swa + ((uint64_t)slot * raw_rows + rr) * (uint64_t)head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += q[d] * kv[d];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        const float score = partial[0] * rsqrtf((float)head_dim);
        if (threadIdx.x == 0u) scores[i] = score;
        max_s = fmaxf(max_s, score);
        __syncthreads();
    }
    for (uint32_t ci = 0; ci < comp_count; ++ci) {
        const float *kv =
            comp_rows + ((uint64_t)slot * comp_row_cap + comp_index[ci]) *
                            (uint64_t)head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += q[d] * kv[d];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        const float score = partial[0] * rsqrtf((float)head_dim);
        if (threadIdx.x == 0u) scores[valid_raw_rows + ci] = score;
        max_s = fmaxf(max_s, score);
        __syncthreads();
    }

    float denom = expf(sinks[local_head] - max_s);
    for (uint32_t i = 0; i < valid_raw_rows + comp_count; ++i) {
        denom += expf(scores[i] - max_s);
    }
    float *out = out_heads + (uint64_t)row * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < valid_raw_rows; ++i) {
            const uint32_t history_offset = valid_raw_rows - 1u - i;
            const uint32_t rr = (raw_row + raw_rows - history_offset) % raw_rows;
            const float *kv =
                raw_swa + ((uint64_t)slot * raw_rows + rr) * (uint64_t)head_dim;
            const float w = expf(scores[i] - max_s) / denom;
            acc += kv[d] * w;
        }
        for (uint32_t ci = 0; ci < comp_count; ++ci) {
            const float *kv =
                comp_rows + ((uint64_t)slot * comp_row_cap + comp_index[ci]) *
                                (uint64_t)head_dim;
            const float w = expf(scores[valid_raw_rows + ci] - max_s) / denom;
            acc += kv[d] * w;
        }
        out[d] = isfinite(acc) ? acc : 0.0f;
    }
}

__global__ void pack_current_full_to_routes_kernel(__half *routes,
                                                   const float *current_full,
                                                   const int *route_slots,
                                                   int routes_n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)routes_n * (uint64_t)kHidden;
    if (i >= n) return;
    const int route = (int)(i / kHidden);
    const int h = (int)(i % kHidden);
    const int slot = route_slots[route];
    routes[i] = f32_to_half_saturate(current_full[(uint64_t)slot * kHidden + h]);
}

__global__ void pack_current_full_to_routes_scaled_kernel(__half *routes,
                                                          float *route_inv_scale,
                                                          const float *current_full,
                                                          const int *route_slots,
                                                          int routes_n,
                                                          float target_abs) {
    const int route = (int)blockIdx.x;
    if (route >= routes_n) return;
    const int slot = route_slots[route];
    float max_abs = 0.0f;
    for (int h = (int)threadIdx.x; h < kHidden; h += (int)blockDim.x) {
        float v = current_full[(uint64_t)slot * kHidden + h];
        if (!isfinite(v)) v = 0.0f;
        max_abs = fmaxf(max_abs, fabsf(v));
    }
    __shared__ float s_max[256];
    s_max[threadIdx.x] = max_abs;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x],
                                       s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float safe_target = fmaxf(target_abs, 1.0f);
    const float scale = s_max[0] > safe_target ? safe_target / s_max[0] : 1.0f;
    if (threadIdx.x == 0u) {
        route_inv_scale[route] = scale > 0.0f ? 1.0f / scale : 1.0f;
    }
    for (int h = (int)threadIdx.x; h < kHidden; h += (int)blockDim.x) {
        float v = current_full[(uint64_t)slot * kHidden + h];
        if (!isfinite(v)) v = 0.0f;
        routes[(uint64_t)route * kHidden + h] = f32_to_half_saturate(v * scale);
    }
}

__global__ void shared_swiglu_shard_to_float_kernel(float *mid,
                                                    const float *gate,
                                                    const float *up,
                                                    uint32_t rank,
                                                    uint32_t rows_per_gpu,
                                                    uint32_t slots,
                                                    float clamp) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)rows_per_gpu;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / rows_per_gpu);
    const uint32_t local = (uint32_t)(i % rows_per_gpu);
    float g = gate[(uint64_t)slot * rows_per_gpu + local];
    float u = up[(uint64_t)slot * rows_per_gpu + local];
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    const float silu = g / (1.0f + expf(-g));
    mid[(uint64_t)slot * kMid + (uint64_t)rank * rows_per_gpu + local] =
        silu * u;
}

__global__ void routed_fused_gate_up_swiglu_clamp_kernel(__half *mid,
                                                         const __half *gate_up,
                                                         const float *route_inv_scale,
                                                         uint64_t routes,
                                                         float clamp) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = routes * (uint64_t)kMid;
    if (i >= n) return;
    const uint64_t row = i / kMid;
    const uint64_t col = i - row * (uint64_t)kMid;
    const uint64_t base = row * (uint64_t)kFusedN + col;
    float g = __half2float(gate_up[base]);
    float u = __half2float(gate_up[base + kMid]);
    if (route_inv_scale) {
        const float inv_scale = route_inv_scale[row];
        g *= inv_scale;
        u *= inv_scale;
    }
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    const float silu = g / (1.0f + expf(-g));
    mid[i] = __float2half(silu * u);
}

__global__ void fill_dense_input_from_current_kernel(float *dst,
                                                     const float *current_full,
                                                     uint32_t cols,
                                                     uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / cols);
    const uint32_t col = (uint32_t)(i % cols);
    dst[i] = current_full[(uint64_t)slot * kHidden + (uint32_t)(col % kHidden)];
}

__global__ void fill_dense_input_half_from_current_kernel(__half *dst,
                                                          const float *current_full,
                                                          uint32_t cols,
                                                          uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / cols);
    const uint32_t col = (uint32_t)(i % cols);
    dst[i] = f32_to_half_saturate(current_full[(uint64_t)slot * kHidden +
                                               (uint32_t)(col % kHidden)]);
}

__global__ void hc_current_fused_fill_pack_kernel(
    float *rank_current_full,
    const float *state_current_full,
    const float *dense_current_full,
    const float *route_current_full,
    float *attn_x,
    uint32_t attn_cols,
    float *shared_x,
    uint32_t shared_cols,
    __half *attn_x_half,
    __half *shared_x_half,
    __half *routes,
    const int *route_slots,
    int routes_n,
    uint32_t slots,
    uint64_t total) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint64_t full_elems = (uint64_t)slots * (uint64_t)kHidden;
    if (i < full_elems) {
        rank_current_full[i] = state_current_full[i];
    }
    const uint64_t attn_elems = (uint64_t)slots * (uint64_t)attn_cols;
    if (i < attn_elems) {
        const uint32_t slot = (uint32_t)(i / attn_cols);
        const uint32_t col = (uint32_t)(i % attn_cols);
        const float v =
            dense_current_full[(uint64_t)slot * kHidden + (uint32_t)(col % kHidden)];
        if (attn_x) attn_x[i] = v;
        if (attn_x_half) attn_x_half[i] = f32_to_half_saturate(v);
    }
    const uint64_t shared_elems = (uint64_t)slots * (uint64_t)shared_cols;
    if (i < shared_elems) {
        const uint32_t slot = (uint32_t)(i / shared_cols);
        const uint32_t col = (uint32_t)(i % shared_cols);
        const float v =
            dense_current_full[(uint64_t)slot * kHidden + (uint32_t)(col % kHidden)];
        if (shared_x) shared_x[i] = v;
        if (shared_x_half) shared_x_half[i] = f32_to_half_saturate(v);
    }
    const uint64_t route_elems = (uint64_t)routes_n * (uint64_t)kHidden;
    if (i < route_elems) {
        const int route = (int)(i / kHidden);
        const int h = (int)(i % kHidden);
        const int slot = route_slots[route];
        routes[i] = f32_to_half_saturate(
            route_current_full[(uint64_t)slot * kHidden + h]);
    }
}

__global__ void fill_attn_compressed_inputs_half_kernel(__half *attn_kv,
                                                        __half *attn_gate,
                                                        const float *current_full,
                                                        uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)kHidden;
    if (i >= n) return;
    const __half v = f32_to_half_saturate(current_full[i]);
    attn_kv[i] = v;
    attn_gate[i] = v;
}

__global__ void fill_ratio4_compressed_indexer_inputs_half_kernel(
    __half *attn_kv,
    __half *attn_gate,
    __half *indexer_proj,
    __half *indexer_kv,
    __half *indexer_gate,
    const float *current_full,
    uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)kHidden;
    if (i >= n) return;
    const __half v = f32_to_half_saturate(current_full[i]);
    attn_kv[i] = v;
    attn_gate[i] = v;
    indexer_proj[i] = v;
    indexer_kv[i] = v;
    indexer_gate[i] = v;
}

__global__ void rms_norm_weight_rows_kernel(float *out,
                                            const float *in,
                                            const float *weight,
                                            uint32_t cols,
                                            uint32_t rows,
                                            float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *src = in + (uint64_t)row * cols;
    float *dst = out + (uint64_t)row * cols;
    float sum = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = src[c];
        sum += v * v;
    }
    sum = block_sum_256_f32(sum);
    const float scale = rsqrtf(sum / (float)cols + eps);
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        dst[c] = src[c] * scale * weight[c];
    }
}

__global__ void rms_norm_weight_rows_stable_kernel(float *out,
                                                   const float *in,
                                                   const float *weight,
                                                   uint32_t cols,
                                                   uint32_t rows,
                                                   float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *src = in + (uint64_t)row * cols;
    float *dst = out + (uint64_t)row * cols;
    float local_max = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = src[c];
        if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
    }
    const float max_abs = block_max_256_f32(local_max);
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
            const float v = src[c];
            if (isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    sum = block_sum_256_f32(sum);
    float scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        scale = rsqrtf(sum / (float)cols + eps / (max_abs * max_abs)) / max_abs;
    }
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = src[c];
        const float y = isfinite(v) ? v * scale * weight[c] : 0.0f;
        dst[c] = isfinite(y) ? y : 0.0f;
    }
}

__global__ void zero_f32_kernel(float *dst, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = 0.0f;
}

__global__ void clamp_f32_abs_kernel(float *dst, uint64_t n, float limit) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = dst[i];
    if (!isfinite(v)) v = 0.0f;
    dst[i] = fminf(limit, fmaxf(-limit, v));
}

__global__ void ep_reduce_all_dest_shards_kernel(float *contrib,
                                                 const __half *route_hidden,
                                                 const int *route_slots,
                                                 const float *route_weights,
                                                 int routes,
                                                 int slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)routes * kHidden;
    if (i >= total) return;
    const int route = (int)(i / kHidden);
    const int h = (int)(i % kHidden);
    const int slot = route_slots[route];
    if (slot < 0 || slot >= slots) return;
    const float w = route_weights ? route_weights[route] : kSyntheticRouteWeight;
    const int dest = h / (kHidden / kGpus);
    const int local_h = h % (kHidden / kGpus);
    const uint64_t out_idx =
        ((uint64_t)dest * slots + (uint64_t)slot) * (kHidden / kGpus) + local_h;
    atomicAdd(contrib + out_idx, __half2float(route_hidden[i]) * w);
}

__global__ void ep_pack_route_dest_shards_kernel(float *packed,
                                                 const __half *route_hidden,
                                                 const float *route_weights,
                                                 int routes,
                                                 int segment_routes) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)routes * kHidden;
    if (i >= total) return;
    const int route = (int)(i / kHidden);
    const int h = (int)(i % kHidden);
    const float w = route_weights ? route_weights[route] : kSyntheticRouteWeight;
    const int dest = h / (kHidden / kGpus);
    const int local_h = h % (kHidden / kGpus);
    const uint64_t out_idx =
        ((uint64_t)dest * (uint64_t)segment_routes + (uint64_t)route) *
            (kHidden / kGpus) +
        local_h;
    packed[out_idx] = __half2float(route_hidden[i]) * w;
}

__global__ void add_f32_kernel(float *dst, const float *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void add_current_attention_shard_kernel(float *dst,
                                                   const float *current,
                                                   const float *attn,
                                                   uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c = current ? current[i] : 0.0f;
    const float a = attn ? attn[i] : 0.0f;
    float v = c + a;
    if (!isfinite(v)) v = 0.0f;
    dst[i] = v;
}

__global__ void cast_f32_to_half_kernel(__half *dst, const float *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = f32_to_half_saturate(src[i]);
}

__global__ void add_half_to_f32_kernel(float *dst, const __half *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += __half2float(src[i]);
}

__global__ void compose_next_hidden_kernel(float *next,
                                           const float *current,
                                           const float *attn,
                                           const float *shared,
                                           const float *ep_sum,
                                           int rank,
                                           int slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t elems = (uint64_t)slots * (kHidden / kGpus);
    if (i >= elems) return;
    const int slot = (int)(i / (kHidden / kGpus));
    const int local_h = (int)(i % (kHidden / kGpus));
    const float synthetic =
        ((float)(rank + 1) * 0.01f) + ((float)slot * 0.001f) +
        ((float)local_h * 0.00001f);
    const float residual = current ? current[i] : synthetic;
    next[i] = residual + attn[i] + shared[i] + ep_sum[i];
}

__global__ void compose_next_hidden_compact8_kernel(float *next,
                                                    const float *current,
                                                    const float *attn,
                                                    const float *shared,
                                                    const float *r0,
                                                    const float *r1,
                                                    const float *r2,
                                                    const float *r3,
                                                    const float *r4,
                                                    const float *r5,
                                                    const float *r6,
                                                    const float *r7,
                                                    const int *idx0,
                                                    const int *idx1,
                                                    const int *idx2,
                                                    const int *idx3,
                                                    const int *idx4,
                                                    const int *idx5,
                                                    const int *idx6,
                                                    const int *idx7,
                                                    int rank,
                                                    int slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t elems = (uint64_t)slots * (kHidden / kGpus);
    if (i >= elems) return;
    const int slot = (int)(i / (kHidden / kGpus));
    const int local_h = (int)(i % (kHidden / kGpus));
    const float synthetic =
        ((float)(rank + 1) * 0.01f) + ((float)slot * 0.001f) +
        ((float)local_h * 0.00001f);
    const float residual = current ? current[i] : synthetic;
    float ep = 0.0f;
    const int i0 = idx0[slot];
    const int i1 = idx1[slot];
    const int i2 = idx2[slot];
    const int i3 = idx3[slot];
    const int i4 = idx4[slot];
    const int i5 = idx5[slot];
    const int i6 = idx6[slot];
    const int i7 = idx7[slot];
    if (i0 >= 0) ep += r0[(uint64_t)i0 * (kHidden / kGpus) + local_h];
    if (i1 >= 0) ep += r1[(uint64_t)i1 * (kHidden / kGpus) + local_h];
    if (i2 >= 0) ep += r2[(uint64_t)i2 * (kHidden / kGpus) + local_h];
    if (i3 >= 0) ep += r3[(uint64_t)i3 * (kHidden / kGpus) + local_h];
    if (i4 >= 0) ep += r4[(uint64_t)i4 * (kHidden / kGpus) + local_h];
    if (i5 >= 0) ep += r5[(uint64_t)i5 * (kHidden / kGpus) + local_h];
    if (i6 >= 0) ep += r6[(uint64_t)i6 * (kHidden / kGpus) + local_h];
    if (i7 >= 0) ep += r7[(uint64_t)i7 * (kHidden / kGpus) + local_h];
    next[i] = residual + attn[i] + shared[i] + ep;
}

__device__ float compact_moe_sum_src_routes(const float *rows,
                                            const int *indices,
                                            const int *counts,
                                            int slot,
                                            int local_h,
                                            int top_k) {
    float acc = 0.0f;
    const int count = counts ? counts[slot] : 0;
    for (int k = 0; k < count && k < top_k; ++k) {
        const int idx = indices[(uint64_t)slot * (uint64_t)top_k + (uint64_t)k];
        if (idx >= 0) {
            acc += rows[(uint64_t)idx * (kHidden / kGpus) + local_h];
        }
    }
    return acc;
}

__global__ void compose_next_hidden_compact8_multi_kernel(
    float *next,
    const float *current,
    const float *attn,
    const float *shared,
    const float *r0,
    const float *r1,
    const float *r2,
    const float *r3,
    const float *r4,
    const float *r5,
    const float *r6,
    const float *r7,
    const int *idx0,
    const int *idx1,
    const int *idx2,
    const int *idx3,
    const int *idx4,
    const int *idx5,
    const int *idx6,
    const int *idx7,
    const int *cnt0,
    const int *cnt1,
    const int *cnt2,
    const int *cnt3,
    const int *cnt4,
    const int *cnt5,
    const int *cnt6,
    const int *cnt7,
    int rank,
    int slots,
    int top_k) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t elems = (uint64_t)slots * (kHidden / kGpus);
    if (i >= elems) return;
    const int slot = (int)(i / (kHidden / kGpus));
    const int local_h = (int)(i % (kHidden / kGpus));
    const float synthetic =
        ((float)(rank + 1) * 0.01f) + ((float)slot * 0.001f) +
        ((float)local_h * 0.00001f);
    const float residual = current ? current[i] : synthetic;
    float ep = 0.0f;
    ep += compact_moe_sum_src_routes(r0, idx0, cnt0, slot, local_h, top_k);
    ep += compact_moe_sum_src_routes(r1, idx1, cnt1, slot, local_h, top_k);
    ep += compact_moe_sum_src_routes(r2, idx2, cnt2, slot, local_h, top_k);
    ep += compact_moe_sum_src_routes(r3, idx3, cnt3, slot, local_h, top_k);
    ep += compact_moe_sum_src_routes(r4, idx4, cnt4, slot, local_h, top_k);
    ep += compact_moe_sum_src_routes(r5, idx5, cnt5, slot, local_h, top_k);
    ep += compact_moe_sum_src_routes(r6, idx6, cnt6, slot, local_h, top_k);
    ep += compact_moe_sum_src_routes(r7, idx7, cnt7, slot, local_h, top_k);
    next[i] = residual + attn[i] + shared[i] + ep;
}

__global__ void compose_next_hidden_sum8_kernel(float *next,
                                                const float *current,
                                                const float *attn,
                                                const float *shared,
                                                const float *r0,
                                                const float *r1,
                                                const float *r2,
                                                const float *r3,
                                                const float *r4,
                                                const float *r5,
                                                const float *r6,
                                                const float *r7,
                                                int rank,
                                                int slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t elems = (uint64_t)slots * (kHidden / kGpus);
    if (i >= elems) return;
    const int slot = (int)(i / (kHidden / kGpus));
    const int local_h = (int)(i % (kHidden / kGpus));
    const float synthetic =
        ((float)(rank + 1) * 0.01f) + ((float)slot * 0.001f) +
        ((float)local_h * 0.00001f);
    const float residual = current ? current[i] : synthetic;
    const float ep =
        r0[i] + r1[i] + r2[i] + r3[i] + r4[i] + r5[i] + r6[i] + r7[i];
    next[i] = residual + attn[i] + shared[i] + ep;
}

__global__ void expand_hidden_to_proxy_hc_shard_kernel(float *hc,
                                                       const float *hidden,
                                                       int rank,
                                                       int slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t shard_cols = kHidden / kGpus;
    const uint64_t elems = (uint64_t)slots * 4ull * shard_cols;
    if (i >= elems) return;
    const uint32_t local_h = (uint32_t)(i % shard_cols);
    const uint32_t row = (uint32_t)((i / shard_cols) & 3ull);
    const uint32_t slot = (uint32_t)(i / (4ull * shard_cols));
    const float v = hidden[(uint64_t)slot * shard_cols + local_h];
    const float row_scale = row == 0u ? 1.0f : (0.25f * (float)(row + 1u));
    const float row_bias =
        ((float)(rank + 1) * 0.0001f) + ((float)row * 0.00001f);
    hc[i] = v * row_scale + row_bias;
}

__global__ void seed_initial_hc_shard_kernel(float *hc, int rank, int slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t shard_cols = kHidden / kGpus;
    const uint64_t elems = (uint64_t)slots * 4ull * shard_cols;
    if (i >= elems) return;
    const uint32_t local_h = (uint32_t)(i % shard_cols);
    const uint32_t row = (uint32_t)((i / shard_cols) & 3ull);
    const uint32_t slot = (uint32_t)(i / (4ull * shard_cols));
    const uint32_t global_h = (uint32_t)rank * shard_cols + local_h;
    const int m = (int)((slot * 97u + row * 31u + global_h * 17u) % 257u);
    hc[i] = ((float)m - 128.0f) * 0.0025f;
}

__device__ void hc4_split_one_dev(float *out,
                                  const float *mix,
                                  const float *scale,
                                  const float *base,
                                  uint32_t sinkhorn_iters,
                                  float epsv) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
    for (int i = 0; i < 4; ++i) {
        const float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }

    float c[16];
    for (int r = 0; r < 4; ++r) {
        float m = -INFINITY;
        for (int col = 0; col < 4; ++col) {
            const float v = mix[8 + r * 4 + col] * comb_scale +
                            base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; ++col) {
            const float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; ++col) c[r * 4 + col] = c[r * 4 + col] / s + epsv;
    }
    for (int col = 0; col < 4; ++col) {
        float s = epsv;
        for (int r = 0; r < 4; ++r) s += c[r * 4 + col];
        for (int r = 0; r < 4; ++r) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; ++iter) {
        for (int r = 0; r < 4; ++r) {
            float s = epsv;
            for (int col = 0; col < 4; ++col) s += c[r * 4 + col];
            for (int col = 0; col < 4; ++col) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; ++col) {
            float s = epsv;
            for (int r = 0; r < 4; ++r) s += c[r * 4 + col];
            for (int r = 0; r < 4; ++r) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; ++i) out[8 + i] = c[i];
}

__global__ void f32_dense_colmajor_kernel(float *out,
                                          const float *weights,
                                          const float *x,
                                          uint32_t rows,
                                          uint32_t cols,
                                          uint32_t slots) {
    const uint32_t row = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (row >= rows || slot >= slots) return;
    const float *xrow = x + (uint64_t)slot * cols;
    float acc = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        acc += weights[(uint64_t)c * rows + row] * xrow[c];
    }
    acc = block_sum_256_f32(acc);
    if (threadIdx.x == 0u) out[(uint64_t)slot * rows + row] = acc;
}

__global__ void hc_split_rows_kernel(float *split,
                                     const float *mix,
                                     const float *scale,
                                     const float *base,
                                     uint32_t slots,
                                     uint32_t sinkhorn_iters) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots) return;
    hc4_split_one_dev(split + (uint64_t)slot * kHcMix,
                      mix + (uint64_t)slot * kHcMix,
                      scale, base, sinkhorn_iters, 1.0e-6f);
}

__global__ void router_select_topk_rows_kernel(int *selected,
                                               float *weights,
                                               const float *logits,
                                               const float *bias,
                                               const int *hash,
                                               const uint32_t *tokens,
                                               const unsigned char *active,
                                               uint32_t hash_rows,
                                               uint32_t slots) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || threadIdx.x != 0u) return;
    const float *row = logits + (uint64_t)slot * kGlobalExperts;
    if (active && active[slot] == 0u) {
        for (int k = 0; k < kModelTopK; ++k) {
            selected[(uint64_t)slot * kModelTopK + (uint64_t)k] = -1;
            weights[(uint64_t)slot * kModelTopK + (uint64_t)k] = 0.0f;
        }
        return;
    }
    float prob[kGlobalExperts];
    for (int e = 0; e < kGlobalExperts; ++e) {
        const float z = row[e];
        const float softplus = z > 20.0f ? z : (z < -20.0f ? expf(z) : log1pf(expf(z)));
        prob[e] = sqrtf(fmaxf(softplus, 0.0f));
    }
    if (hash && tokens && hash_rows > 0u) {
        uint32_t tok = tokens[slot];
        if (tok >= hash_rows) tok = 0u;
        const int *hrow = hash + (uint64_t)tok * kModelTopK;
        float sum = 0.0f;
        for (int k = 0; k < kModelTopK; ++k) {
            const int e = hrow[k];
            selected[(uint64_t)slot * kModelTopK + (uint64_t)k] = e;
            const float w = (e >= 0 && e < kGlobalExperts) ? prob[e] : 0.0f;
            weights[(uint64_t)slot * kModelTopK + (uint64_t)k] = w;
            sum += w;
        }
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
        for (int k = 0; k < kModelTopK; ++k) {
            weights[(uint64_t)slot * kModelTopK + (uint64_t)k] =
                weights[(uint64_t)slot * kModelTopK + (uint64_t)k] / sum * 1.5f;
        }
        return;
    }
    int best[kModelTopK];
    float best_score[kModelTopK];
    float best_prob[kModelTopK];
    for (int i = 0; i < kModelTopK; ++i) {
        best[i] = -1;
        best_score[i] = -INFINITY;
        best_prob[i] = 0.0f;
    }
    for (int e = 0; e < kGlobalExperts; ++e) {
        const float p = prob[e];
        const float score = p + (bias ? bias[e] : 0.0f);
        for (int k = 0; k < kModelTopK; ++k) {
            if (score > best_score[k]) {
                for (int m = kModelTopK - 1; m > k; --m) {
                    best[m] = best[m - 1];
                    best_score[m] = best_score[m - 1];
                    best_prob[m] = best_prob[m - 1];
                }
                best[k] = e;
                best_score[k] = score;
                best_prob[k] = p;
                break;
            }
        }
    }
    float sum = 0.0f;
    for (int k = 0; k < kModelTopK; ++k) sum += best_prob[k];
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (int k = 0; k < kModelTopK; ++k) {
        selected[(uint64_t)slot * kModelTopK + (uint64_t)k] = best[k];
        weights[(uint64_t)slot * kModelTopK + (uint64_t)k] =
            best_prob[k] / sum * 1.5f;
    }
}

__global__ void router_select_hash_fast_rows_kernel(int *selected,
                                                    float *weights,
                                                    const float *logits,
                                                    const int *hash,
                                                    const uint32_t *tokens,
                                                    const unsigned char *active,
                                                    uint32_t hash_rows,
                                                    uint32_t slots) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || threadIdx.x != 0u) return;
    if (active && active[slot] == 0u) {
        for (int k = 0; k < kModelTopK; ++k) {
            selected[(uint64_t)slot * kModelTopK + (uint64_t)k] = -1;
            weights[(uint64_t)slot * kModelTopK + (uint64_t)k] = 0.0f;
        }
        return;
    }
    uint32_t tok = tokens[slot];
    if (tok >= hash_rows) tok = 0u;
    const int *hrow = hash + (uint64_t)tok * kModelTopK;
    const float *row = logits + (uint64_t)slot * kGlobalExperts;
    float sum = 0.0f;
    float local[kModelTopK];
    for (int k = 0; k < kModelTopK; ++k) {
        const int e = hrow[k];
        selected[(uint64_t)slot * kModelTopK + (uint64_t)k] = e;
        float w = 0.0f;
        if (e >= 0 && e < kGlobalExperts) {
            const float z = row[e];
            const float softplus =
                z > 20.0f ? z : (z < -20.0f ? expf(z) : log1pf(expf(z)));
            w = sqrtf(fmaxf(softplus, 0.0f));
        }
        local[k] = w;
        sum += w;
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (int k = 0; k < kModelTopK; ++k) {
        weights[(uint64_t)slot * kModelTopK + (uint64_t)k] =
            local[k] / sum * 1.5f;
    }
}

__global__ void gpu_route_count_all_kernel(const int *selected,
                                           int *offsets_all,
                                           uint32_t slots,
                                           uint32_t top_k) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = slots * top_k;
    if (i >= total) return;
    const int expert = selected[i];
    if (expert < 0 || expert >= kGlobalExperts) return;
    const int rank = expert / kLocalExperts;
    const int local = expert - rank * kLocalExperts;
    atomicAdd(offsets_all + (uint64_t)rank * (kLocalExperts + 1) + local + 1, 1);
}

__global__ void gpu_route_prefix_all_kernel(int *offsets_all, int *totals) {
    const int rank = (int)threadIdx.x;
    if (rank >= kGpus) return;
    int *offsets = offsets_all + (uint64_t)rank * (kLocalExperts + 1);
    int running = 0;
    for (int local = 0; local < kLocalExperts; ++local) {
        const int count = offsets[local + 1];
        offsets[local] = running;
        running += count;
    }
    offsets[kLocalExperts] = running;
    totals[rank] = running;
}

__global__ void gpu_route_init_compact_plan_kernel(int *compact_plan,
                                                   uint32_t slots,
                                                   uint32_t top_k) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t indices = (uint64_t)kGpus * slots * top_k;
    const uint64_t counts = (uint64_t)kGpus * slots;
    if (i < indices) {
        compact_plan[i] = -1;
    }
    if (i < counts) {
        compact_plan[indices + i] = 0;
    }
}

__global__ void gpu_route_copy_own_offsets_kernel(int *dst_offsets,
                                                  const int *offsets_all,
                                                  uint32_t rank) {
    const uint32_t i = threadIdx.x;
    if (i <= (uint32_t)kLocalExperts) {
        dst_offsets[i] =
            offsets_all[(uint64_t)rank * (kLocalExperts + 1) + i];
    }
}

__global__ void gpu_route_fill_all_kernel(const int *selected,
                                          const float *weights,
                                          const int *offsets_all,
                                          int local_rank,
                                          int *route_slots,
                                          float *route_weights,
                                          int *compact_plan,
                                          uint32_t slots,
                                          uint32_t top_k) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = slots * top_k;
    if (i >= total) return;
    const int expert = selected[i];
    if (expert < 0 || expert >= kGlobalExperts) return;
    const int src_rank = expert / kLocalExperts;
    const int local = expert - src_rank * kLocalExperts;
    const uint32_t slot = i / top_k;
    const uint32_t k = i - slot * top_k;
    int prior_same_expert = 0;
    for (uint32_t j = 0; j < i; ++j) {
        if (selected[j] == expert) ++prior_same_expert;
    }
    const int route_idx =
        offsets_all[(uint64_t)src_rank * (kLocalExperts + 1) + local] +
        prior_same_expert;
    int route_order = 0;
    for (uint32_t prev_k = 0; prev_k < k; ++prev_k) {
        const int prev_expert = selected[(uint64_t)slot * top_k + prev_k];
        if (prev_expert >= 0 && prev_expert < kGlobalExperts &&
            prev_expert / kLocalExperts == src_rank) {
            ++route_order;
        }
    }
    const uint64_t indices_per_src = (uint64_t)slots * top_k;
    const uint64_t counts_base = (uint64_t)kGpus * indices_per_src;
    if (route_order < (int)top_k) {
        compact_plan[(uint64_t)src_rank * indices_per_src +
                     (uint64_t)slot * top_k + route_order] = route_idx;
    }
    atomicAdd(compact_plan + counts_base + (uint64_t)src_rank * slots + slot, 1);
    if (src_rank == local_rank) {
        route_slots[route_idx] = (int)slot;
        route_weights[route_idx] = weights[i];
    }
}

__global__ void hc_expand_shard_kernel(float *out_hc,
                                       const float *block_out,
                                       const float *residual_hc,
                                       const float *split,
                                       uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t shard_cols = kHidden / kGpus;
    const uint64_t elems = (uint64_t)slots * kHcRows * shard_cols;
    if (i >= elems) return;
    const uint32_t local_h = (uint32_t)(i % shard_cols);
    const uint32_t dst_hc = (uint32_t)((i / shard_cols) & 3ull);
    const uint32_t slot = (uint32_t)(i / ((uint64_t)kHcRows * shard_cols));
    const float *sp = split + (uint64_t)slot * kHcMix;
    const uint64_t slot_hc_base = (uint64_t)slot * kHcRows * shard_cols;
    float acc = block_out[(uint64_t)slot * shard_cols + local_h] * sp[4u + dst_hc];
    for (uint32_t src_hc = 0; src_hc < kHcRows; ++src_hc) {
        const float comb = sp[8u + dst_hc + src_hc * kHcRows];
        const float res = residual_hc[slot_hc_base + (uint64_t)src_hc * shard_cols + local_h];
        acc += comb * res;
    }
    out_hc[i] = acc;
}

bool parse_int(const char *text, int *out) {
    if (!text || !*text) return false;
    char *end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = (int)v;
    return true;
}

bool parse_u64(const char *text, uint64_t *out) {
    if (!text || !*text) return false;
    char *end = nullptr;
    const unsigned long long v = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') return false;
    *out = (uint64_t)v;
    return true;
}

bool parse_size(const char *text, size_t *out) {
    uint64_t v = 0;
    if (!parse_u64(text, &v)) return false;
    if (v > (uint64_t)std::numeric_limits<size_t>::max()) return false;
    *out = (size_t)v;
    return true;
}

std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (start <= line.size()) {
        const size_t tab = line.find('\t', start);
        if (tab == std::string::npos) {
            fields.emplace_back(line.substr(start));
            break;
        }
        fields.emplace_back(line.substr(start, tab - start));
        start = tab + 1;
    }
    return fields;
}

bool safe_sidecar_name(const std::string &name) {
    return !name.empty() &&
           name.find('/') == std::string::npos &&
           name.find('\\') == std::string::npos &&
           name.find("..") == std::string::npos;
}

std::string path_join(const char *dir, const std::string &base) {
    std::string out(dir ? dir : "");
    if (!out.empty() && out.back() != '/') out.push_back('/');
    out += base;
    return out;
}

int read_exact_at(const std::string &path, uint64_t offset, void *dst, size_t bytes) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "cannot open sidecar %s: %s\n", path.c_str(), std::strerror(errno));
        return 1;
    }
    if (fseeko(fp, (off_t)offset, SEEK_SET) != 0) {
        std::fprintf(stderr, "cannot seek sidecar %s offset %llu: %s\n",
                     path.c_str(), (unsigned long long)offset, std::strerror(errno));
        std::fclose(fp);
        return 2;
    }
    const size_t got = std::fread(dst, 1, bytes, fp);
    if (got != bytes) {
        std::fprintf(stderr, "short read sidecar %s offset %llu bytes %zu got %zu\n",
                     path.c_str(), (unsigned long long)offset, bytes, got);
        std::fclose(fp);
        return 3;
    }
    std::fclose(fp);
    return 0;
}

bool parse_devices(const char *text, int devices[kGpus]) {
    std::vector<int> parsed;
    const char *cur = text;
    while (cur && *cur) {
        const char *comma = std::strchr(cur, ',');
        std::string piece;
        if (comma) {
            piece.assign(cur, comma - cur);
            cur = comma + 1;
        } else {
            piece.assign(cur);
            cur = nullptr;
        }
        int dev = 0;
        if (!parse_int(piece.c_str(), &dev) || dev < 0) return false;
        parsed.push_back(dev);
    }
    if ((int)parsed.size() != kGpus) return false;
    for (int i = 0; i < kGpus; ++i) {
        for (int j = i + 1; j < kGpus; ++j) {
            if (parsed[i] == parsed[j]) return false;
        }
        devices[i] = parsed[i];
    }
    return true;
}

constexpr uint64_t kMiB = 1024ull * 1024ull;

bool should_report_vram(const Options &opt) {
    return opt.vram_report || opt.vram_min_free_mib > 0;
}

bool nccl_gate_active(const Options &opt) {
    return opt.nccl_reduce_scatter_compose_gate ||
           opt.tp_hc_current_input_nccl_allgather_gate ||
           opt.true_ds4_attention_output_nccl_allgather_gate;
}

int report_vram_checkpoint_min_free(const Options &opt,
                                    const char *label,
                                    uint64_t min_free_mib_threshold) {
    const uint64_t min_free_bytes = min_free_mib_threshold * kMiB;
    uint64_t min_free_mib = UINT64_MAX;
    uint64_t max_used_mib = 0;
    int failures = 0;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        size_t free_b = 0;
        size_t total_b = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
        const uint64_t used_b = (uint64_t)total_b - (uint64_t)free_b;
        const uint64_t free_mib = (uint64_t)free_b / kMiB;
        const uint64_t used_mib = used_b / kMiB;
        const uint64_t total_mib = (uint64_t)total_b / kMiB;
        min_free_mib = std::min(min_free_mib, free_mib);
        max_used_mib = std::max(max_used_mib, used_mib);
        const bool pass =
            min_free_mib_threshold == 0 || (uint64_t)free_b >= min_free_bytes;
        if (!pass) failures++;
        std::printf("tp_ep_vram\tlabel\t%s\tgpu\t%d\tfree_mib\t%llu\t"
                    "used_mib\t%llu\ttotal_mib\t%llu\tmin_free_mib\t%llu\t%s\n",
                    label, gpu,
                    (unsigned long long)free_mib,
                    (unsigned long long)used_mib,
                    (unsigned long long)total_mib,
                    (unsigned long long)min_free_mib_threshold,
                    pass ? "PASS" : "FAIL");
    }
    if (min_free_mib == UINT64_MAX) min_free_mib = 0;
    std::printf("tp_ep_vram_summary\tlabel\t%s\tmin_free_mib\t%llu\t"
                "max_used_mib\t%llu\tthreshold_mib\t%llu\tfailures\t%d\t%s\n",
                label,
                (unsigned long long)min_free_mib,
                (unsigned long long)max_used_mib,
                (unsigned long long)min_free_mib_threshold,
                failures,
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

int report_vram_checkpoint(const Options &opt, const char *label) {
    if (!should_report_vram(opt)) return 0;
    return report_vram_checkpoint_min_free(opt, label, opt.vram_min_free_mib);
}

int report_nccl_vram_checkpoint(const Options &opt, const char *label) {
    if (!nccl_gate_active(opt) || opt.nccl_min_free_mib == 0) return 0;
    return report_vram_checkpoint_min_free(opt, label, opt.nccl_min_free_mib);
}

int check_planned_vram_allocation(const Options &opt,
                                  const char *label,
                                  const uint64_t planned_bytes[kGpus]) {
    if (!should_report_vram(opt)) return 0;
    const uint64_t min_free_bytes = opt.vram_min_free_mib * kMiB;
    int failures = 0;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        size_t free_b = 0;
        size_t total_b = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
        const uint64_t required_b = planned_bytes[gpu] + min_free_bytes;
        const bool pass = (uint64_t)free_b >= required_b;
        if (!pass) failures++;
        std::printf("tp_ep_vram_plan\tlabel\t%s\tgpu\t%d\tfree_mib\t%llu\t"
                    "planned_mib\t%llu\tthreshold_mib\t%llu\ttotal_mib\t%llu\t%s\n",
                    label, gpu,
                    (unsigned long long)((uint64_t)free_b / kMiB),
                    (unsigned long long)(planned_bytes[gpu] / kMiB),
                    (unsigned long long)opt.vram_min_free_mib,
                    (unsigned long long)((uint64_t)total_b / kMiB),
                    pass ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}

void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s --pack-dir DIR --contract FILE --tm-index FILE [options]\n"
                 "       [--lib PATH] [--tokenizer-model PATH]\n"
                 "       [--devices 0,1,2,3,4,5,6,7]\n"
                 "       [--slots N] [--top-k N] [--layer N] [--kv-slot N]\n"
                 "       [--position N] [--warmup N] [--iters N]\n"
                 "       [--dense-compute-tensor NAME] [--dense-compute-all-f8]\n"
                 "       [--dense-compute-all-bf16] [--dense-compute-all]\n"
                 "       [--compose-next-hidden] [--decode-steps N]\n"
                 "       [--ep-return-fp16] [--fuse-compose-sum]\n"
                 "       [--dense-hmma-compose] [--dense-f16-cublas-compose]\n"
                 "       [--dense-f16-cache-compose] [--all-layers]\n"
                 "       [--skip-descriptor-checks] [--skip-predecode-probes]\n"
                 "       [--share-tp-runtime] [--local-tp-runtime]\n"
                 "       [--shared-expert-bindings] [--local-expert-bindings]\n"
                 "       [--overlap-ep-dense] [--serial-ep-dense]\n"
                 "       [--direct-remote-compose]\n"
                 "       [--source-copy-schedule] [--dest-copy-schedule]\n"
                 "       [--copy-event-compose]\n"
                 "       [--compact-route-compose] [--compact-moe-decode-gate]\n"
                 "       [--fused-gated-silu-gate]\n"
                 "       [--fp8-e5m2-kv-gate]\n"
                 "       [--token-major-all-layers] [--shared-dense-ops]\n"
                 "       [--skip-self-compose-copy] [--copy-self-compose]\n"
                 "       [--multi-copy-streams]\n"
                 "       [--nccl-reduce-scatter-compose-gate] [--serving-bench]\n"
                 "       [--skip-decode-checksum]\n"
                 "       [--serve-http] [--host ADDR] [--port N] [--max-requests N]\n"
                 "       [--microbatch-wait-us N]\n"
                 "       [--vram-report] [--vram-min-free-mib N]\n"
                 "       [--nccl-min-free-mib N]\n"
                 "       [--output-head-gate] [--output-head-resident-gate]\n"
                 "       [--diagnostic-output-head-lazy-gate]\n"
                 "       [--final-hc-carry-gate] [--tp-hc-final-expand-gate]\n"
                 "       [--tp-hc-current-input-gate]\n"
                 "       [--tp-hc-current-input-peer-gather-gate]\n"
                 "       [--tp-hc-current-input-nccl-allgather-gate]\n"
                 "       [--tp-hc-current-input-stream-sync-gate]\n"
                 "       [--tp-hc-current-input-fused-fill-pack-gate]\n"
                 "       [--tp-hc-persist-state-gate] [--tp-kv-all-slots-gate]\n"
                 "       [--model-router-routes]\n"
                 "       [--routed-ffn-norm-input-gate]\n"
                 "       [--true-shared-ffn-gate]\n"
                 "       [--true-ds4-attention-residency-gate]\n"
                 "       [--true-ds4-attention-projection-gate]\n"
                 "       [--true-ds4-attention-state-gate]\n"
                 "       [--true-ds4-attention-rope-gate]\n"
                 "       [--true-ds4-attention-saturation-audit-gate]\n"
                 "       [--true-ds4-attention-kv-norm-reference-gate]\n"
                 "       [--true-ds4-attention-raw-read-gate]\n"
                 "       [--true-ds4-attention-raw-window-gate]\n"
                 "       [--true-ds4-attention-typed-kv-raw-gate]\n"
                 "       [--true-ds4-attention-typed-kv-compressed-gate]\n"
                 "       [--true-ds4-attention-typed-kv-indexer-gate]\n"
                 "       [--true-ds4-attention-typed-kv-history-gate]\n"
                 "       [--true-ds4-attention-typed-kv-skip-current-load-gate]\n"
                 "       [--true-ds4-attention-typed-kv-skip-raw-store-gate]\n"
                 "       [--true-ds4-attention-typed-kv-skip-compressed-store-gate]\n"
                 "       [--true-ds4-attention-typed-kv-skip-indexer-store-gate]\n"
                 "       [--true-ds4-attention-typed-kv-quiet-gate]\n"
                 "       [--true-ds4-attention-typed-kv-batch-rows-gate]\n"
                 "       [--true-ds4-attention-typed-kv-stream-sync-gate]\n"
                 "       [--true-ds4-attention-output-gate]\n"
                 "       [--true-ds4-attention-output-nccl-allgather-gate]\n"
                 "       [--true-ds4-post-attention-ffn-input-gate]\n"
                 "       [--true-ds4-compressed-kv-gate]\n"
                 "       [--true-ds4-indexer-attention-gate]\n"
                 "       [--true-ds4-compressed-kv-dense-event-wait-gate]\n"
                 "       [--true-ds4-compressed-kv-fused-input-fill-gate]\n"
                 "       [--true-ds4-compressed-kv-fused-rope-round-gate]\n"
                 "       [--true-ds4-compressed-kv-fused-pool-norm-gate]\n"
                 "       [--true-ds4-compressed-reference-diff-gate]\n"
                 "       [--reference-hc-reduce-gate]\n"
                 "       [--reference-hc-state-guard-gate]\n"
                 "       [--cuda-profiler-window]\n"
                 "       [--async-output-gate]\n"
                 "       [--decode-cudagraph-gate]\n"
                 "       [--batched-paged-attn-gate]\n"
                 "       [--router-cublas-gate]\n"
                 "       [--router-hash-fast-gate]\n"
                 "       [--gpu-route-plan-gate]\n"
                 "       [--route-plan-async-upload-gate]\n"
                 "       [--diagnostic-output-head]\n"
                 "       [--diagnostic-output-head-lazy-gate]\n",
                 argv0);
}

bool parse_args(int argc, char **argv, Options *opt) {
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        const char *val = (i + 1 < argc) ? argv[i + 1] : nullptr;
        if (std::strcmp(arg, "--lib") == 0) {
            if (!val) return false;
            opt->lib_path = val;
            ++i;
        } else if (std::strcmp(arg, "--pack-dir") == 0) {
            if (!val) return false;
            opt->pack_dir = val;
            ++i;
        } else if (std::strcmp(arg, "--contract") == 0) {
            if (!val) return false;
            opt->contract_path = val;
            ++i;
        } else if (std::strcmp(arg, "--tm-index") == 0) {
            if (!val) return false;
            opt->tm_index_path = val;
            ++i;
        } else if (std::strcmp(arg, "--tokenizer-model") == 0) {
            if (!val) return false;
            opt->tokenizer_model_path = val;
            ++i;
        } else if (std::strcmp(arg, "--devices") == 0) {
            if (!val || !parse_devices(val, opt->devices)) return false;
            ++i;
        } else if (std::strcmp(arg, "--slots") == 0) {
            if (!val || !parse_int(val, &opt->slots) || opt->slots <= 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--top-k") == 0) {
            if (!val || !parse_int(val, &opt->top_k) || opt->top_k <= 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--layer") == 0) {
            if (!val || !parse_int(val, &opt->layer)) return false;
            ++i;
        } else if (std::strcmp(arg, "--kv-slot") == 0) {
            int slot = 0;
            if (!val || !parse_int(val, &slot) || slot < 0) return false;
            opt->kv_slot = (uint32_t)slot;
            ++i;
        } else if (std::strcmp(arg, "--position") == 0) {
            if (!val || !parse_u64(val, &opt->position)) return false;
            ++i;
        } else if (std::strcmp(arg, "--warmup") == 0) {
            if (!val || !parse_int(val, &opt->warmup) || opt->warmup < 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--iters") == 0) {
            if (!val || !parse_int(val, &opt->iters) || opt->iters <= 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--dense-compute-tensor") == 0) {
            if (!val) return false;
            opt->dense_compute_tensor = val;
            ++i;
        } else if (std::strcmp(arg, "--dense-compute-all-f8") == 0) {
            opt->dense_compute_all_f8 = true;
        } else if (std::strcmp(arg, "--dense-compute-all-bf16") == 0) {
            opt->dense_compute_all_bf16 = true;
        } else if (std::strcmp(arg, "--dense-compute-all") == 0) {
            opt->dense_compute_all_f8 = true;
            opt->dense_compute_all_bf16 = true;
        } else if (std::strcmp(arg, "--compose-next-hidden") == 0) {
            opt->compose_next_hidden = true;
        } else if (std::strcmp(arg, "--decode-steps") == 0) {
            if (!val || !parse_int(val, &opt->decode_steps) || opt->decode_steps < 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--ep-return-fp16") == 0) {
            opt->ep_return_fp16 = true;
        } else if (std::strcmp(arg, "--fuse-compose-sum") == 0) {
            opt->fuse_compose_sum = true;
        } else if (std::strcmp(arg, "--dense-hmma-compose") == 0) {
            opt->dense_hmma_compose = true;
        } else if (std::strcmp(arg, "--dense-f16-cublas-compose") == 0) {
            opt->dense_f16_cublas_compose = true;
        } else if (std::strcmp(arg, "--dense-f16-cache-compose") == 0) {
            opt->dense_f16_cache_compose = true;
        } else if (std::strcmp(arg, "--all-layers") == 0) {
            opt->all_layers = true;
        } else if (std::strcmp(arg, "--skip-descriptor-checks") == 0) {
            opt->skip_descriptor_checks = true;
        } else if (std::strcmp(arg, "--skip-predecode-probes") == 0) {
            opt->skip_predecode_probes = true;
        } else if (std::strcmp(arg, "--share-tp-runtime") == 0) {
            opt->share_tp_runtime = true;
            opt->tp_runtime_explicit = true;
        } else if (std::strcmp(arg, "--local-tp-runtime") == 0) {
            opt->share_tp_runtime = false;
            opt->tp_runtime_explicit = true;
        } else if (std::strcmp(arg, "--tp-runtime-skip-unused-comp-state-gate") == 0) {
            opt->tp_runtime_skip_unused_comp_state = true;
        } else if (std::strcmp(arg, "--shared-expert-bindings") == 0) {
            opt->share_expert_bindings = true;
        } else if (std::strcmp(arg, "--local-expert-bindings") == 0) {
            opt->share_expert_bindings = false;
        } else if (std::strcmp(arg, "--overlap-ep-dense") == 0) {
            opt->overlap_ep_dense = true;
        } else if (std::strcmp(arg, "--serial-ep-dense") == 0) {
            opt->overlap_ep_dense = false;
        } else if (std::strcmp(arg, "--direct-remote-compose") == 0) {
            opt->direct_remote_compose = true;
        } else if (std::strcmp(arg, "--source-copy-schedule") == 0) {
            opt->source_copy_schedule = true;
        } else if (std::strcmp(arg, "--dest-copy-schedule") == 0) {
            opt->source_copy_schedule = false;
        } else if (std::strcmp(arg, "--copy-event-compose") == 0) {
            opt->copy_event_compose = true;
        } else if (std::strcmp(arg, "--compact-route-compose") == 0) {
            opt->compact_route_compose = true;
        } else if (std::strcmp(arg, "--compact-moe-decode-gate") == 0) {
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
        } else if (std::strcmp(arg, "--fused-gated-silu-gate") == 0) {
            opt->fused_gated_silu_gate = true;
        } else if (std::strcmp(arg, "--token-major-all-layers") == 0) {
            opt->token_major_all_layers = true;
        } else if (std::strcmp(arg, "--shared-dense-ops") == 0) {
            opt->share_dense_ops = true;
        } else if (std::strcmp(arg, "--skip-self-compose-copy") == 0) {
            opt->skip_self_compose_copy = true;
        } else if (std::strcmp(arg, "--copy-self-compose") == 0) {
            opt->skip_self_compose_copy = false;
        } else if (std::strcmp(arg, "--multi-copy-streams") == 0) {
            opt->multi_copy_streams = true;
        } else if (std::strcmp(arg, "--nccl-reduce-scatter-compose-gate") == 0) {
            opt->nccl_reduce_scatter_compose_gate = true;
        } else if (std::strcmp(arg, "--serving-bench") == 0) {
            opt->serving_bench = true;
        } else if (std::strcmp(arg, "--skip-decode-checksum") == 0) {
            opt->skip_decode_checksum = true;
        } else if (std::strcmp(arg, "--serve-http") == 0) {
            opt->serve_http = true;
            opt->serving_bench = true;
            opt->token_major_all_layers = true;
            opt->all_layers = true;
            opt->share_tp_runtime = true;
            opt->tp_runtime_explicit = true;
            opt->skip_decode_checksum = true;
        } else if (std::strcmp(arg, "--host") == 0) {
            if (!val) return false;
            opt->host = val;
            ++i;
        } else if (std::strcmp(arg, "--port") == 0) {
            if (!val || !parse_int(val, &opt->port) || opt->port <= 0 || opt->port > 65535) return false;
            ++i;
        } else if (std::strcmp(arg, "--max-requests") == 0) {
            if (!val || !parse_int(val, &opt->max_requests) || opt->max_requests < 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--microbatch-wait-us") == 0) {
            if (!val || !parse_int(val, &opt->microbatch_wait_us) ||
                opt->microbatch_wait_us < 0 || opt->microbatch_wait_us > 1000000) return false;
            ++i;
        } else if (std::strcmp(arg, "--vram-report") == 0) {
            opt->vram_report = true;
        } else if (std::strcmp(arg, "--vram-min-free-mib") == 0) {
            if (!val || !parse_u64(val, &opt->vram_min_free_mib)) return false;
            ++i;
        } else if (std::strcmp(arg, "--nccl-min-free-mib") == 0) {
            if (!val || !parse_u64(val, &opt->nccl_min_free_mib)) return false;
            ++i;
        } else if (std::strcmp(arg, "--output-head-gate") == 0) {
            opt->output_head_gate = true;
        } else if (std::strcmp(arg, "--output-head-resident-gate") == 0) {
            opt->output_head_resident_gate = true;
        } else if (std::strcmp(arg, "--async-output-gate") == 0) {
            opt->async_output_gate = true;
            opt->diagnostic_output_head = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--decode-cudagraph-gate") == 0) {
            opt->decode_cudagraph_gate = true;
        } else if (std::strcmp(arg, "--batched-paged-attn-gate") == 0) {
            opt->batched_paged_attn_gate = true;
            opt->true_ds4_attention_typed_kv_batch_rows_gate = true;
        } else if (std::strcmp(arg, "--final-hc-carry-gate") == 0) {
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-final-expand-gate") == 0) {
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-gate") == 0) {
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-peer-gather-gate") == 0) {
            opt->tp_hc_current_input_peer_gather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-nccl-allgather-gate") == 0) {
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-stream-sync-gate") == 0) {
            opt->tp_hc_current_input_stream_sync_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-fused-fill-pack-gate") == 0) {
            opt->tp_hc_current_input_fused_fill_pack_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--model-router-routes") == 0) {
            opt->model_router_routes = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--router-cublas-gate") == 0) {
            opt->router_cublas_gate = true;
            opt->model_router_routes = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--router-hash-fast-gate") == 0) {
            opt->router_hash_fast_gate = true;
            opt->model_router_routes = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--gpu-route-plan-gate") == 0) {
            opt->gpu_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--route-plan-async-upload-gate") == 0) {
            opt->route_plan_async_upload_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--routed-ffn-norm-input-gate") == 0) {
            opt->routed_ffn_norm_input_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-shared-ffn-gate") == 0) {
            opt->true_shared_ffn_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-residency-gate") == 0) {
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-projection-gate") == 0) {
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-state-gate") == 0) {
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-rope-gate") == 0) {
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-saturation-audit-gate") == 0) {
            opt->true_ds4_attention_saturation_audit_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-kv-norm-reference-gate") == 0) {
            opt->true_ds4_attention_kv_norm_reference_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-raw-read-gate") == 0) {
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-raw-window-gate") == 0) {
            opt->true_ds4_attention_raw_window_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-raw-gate") == 0) {
            opt->true_ds4_attention_typed_kv_raw_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-compressed-gate") == 0) {
            opt->true_ds4_attention_typed_kv_compressed_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-indexer-gate") == 0) {
            opt->true_ds4_attention_typed_kv_indexer_gate = true;
            opt->true_ds4_indexer_attention_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-history-gate") == 0) {
            opt->true_ds4_attention_typed_kv_history_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-skip-current-load-gate") == 0) {
            opt->true_ds4_attention_typed_kv_skip_current_load_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-skip-raw-store-gate") == 0) {
            opt->true_ds4_attention_typed_kv_skip_raw_store_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-skip-compressed-store-gate") == 0) {
            opt->true_ds4_attention_typed_kv_skip_compressed_store_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-skip-indexer-store-gate") == 0) {
            opt->true_ds4_attention_typed_kv_skip_indexer_store_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-quiet-gate") == 0) {
            opt->true_ds4_attention_typed_kv_quiet_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-batch-rows-gate") == 0) {
            opt->true_ds4_attention_typed_kv_batch_rows_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-stream-sync-gate") == 0) {
            opt->true_ds4_attention_typed_kv_stream_sync_gate = true;
        } else if (std::strcmp(arg, "--fp8-e5m2-kv-gate") == 0) {
            opt->fp8_e5m2_kv_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-output-gate") == 0) {
            opt->true_ds4_attention_output_gate = true;
            opt->true_ds4_attention_raw_window_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-output-nccl-allgather-gate") == 0) {
            opt->true_ds4_attention_output_nccl_allgather_gate = true;
            opt->true_ds4_attention_output_gate = true;
            opt->true_ds4_attention_raw_window_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-post-attention-ffn-input-gate") == 0) {
            opt->true_ds4_post_attention_ffn_input_gate = true;
            opt->true_ds4_attention_output_gate = true;
            opt->true_ds4_attention_raw_window_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->true_shared_ffn_gate = true;
            opt->routed_ffn_norm_input_gate = true;
            opt->model_router_routes = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-semantic-skip-stats-gate") == 0) {
            opt->true_ds4_semantic_skip_stats_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-gate") == 0) {
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-indexer-attention-gate") == 0) {
            opt->true_ds4_indexer_attention_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-direct-input-fill-gate") == 0) {
            opt->true_ds4_compressed_kv_direct_input_fill_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-dense-event-wait-gate") == 0) {
            opt->true_ds4_compressed_kv_dense_event_wait_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-skip-dense-stats-gate") == 0) {
            opt->true_ds4_compressed_kv_skip_dense_stats_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-attn-input-fill-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_attn_input_fill_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-input-fill-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_input_fill_gate = true;
            opt->true_ds4_indexer_attention_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-rope-round-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_rope_round_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-pool-norm-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_pool_norm_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-pool-norm-rope-round-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_pool_norm_rope_round_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-reference-diff-gate") == 0) {
            opt->true_ds4_compressed_reference_diff_gate = true;
            opt->true_ds4_indexer_attention_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--reference-hc-reduce-gate") == 0) {
            opt->reference_hc_reduce_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--reference-hc-state-guard-gate") == 0) {
            opt->reference_hc_state_guard_gate = true;
            opt->reference_hc_reduce_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-persist-state-gate") == 0) {
            opt->tp_hc_persist_state_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-kv-all-slots-gate") == 0) {
            opt->tp_kv_all_slots_gate = true;
        } else if (std::strcmp(arg, "--cuda-profiler-window") == 0) {
            opt->cuda_profiler_window = true;
        } else if (std::strcmp(arg, "--diagnostic-output-head") == 0) {
            opt->diagnostic_output_head = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--diagnostic-output-head-lazy-gate") == 0) {
            opt->diagnostic_output_head = true;
            opt->diagnostic_output_head_lazy_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }
    return opt->pack_dir && opt->contract_path && opt->tm_index_path &&
           opt->top_k <= kPackedLocalExperts && opt->layer >= 0 &&
           (!opt->model_router_routes || opt->top_k == kModelTopK) &&
           (!opt->gpu_route_plan_gate || opt->compact_moe_decode_gate) &&
           (!opt->nccl_reduce_scatter_compose_gate ||
            !opt->decode_cudagraph_gate) &&
           (!opt->tp_hc_current_input_nccl_allgather_gate ||
            opt->tp_hc_current_input_gate) &&
           !(opt->model_router_routes && opt->compact_route_compose &&
             !opt->compact_moe_decode_gate) &&
           !(opt->dense_hmma_compose && opt->dense_f16_cublas_compose) &&
           (!opt->dense_f16_cache_compose || opt->dense_f16_cublas_compose) &&
           (!opt->true_ds4_attention_residency_gate ||
            (opt->share_dense_ops && opt->dense_f16_cache_compose &&
             opt->dense_f16_cublas_compose)) &&
           (!opt->true_ds4_attention_projection_gate ||
            opt->true_ds4_attention_residency_gate) &&
           (!opt->true_ds4_attention_state_gate ||
            opt->true_ds4_attention_projection_gate) &&
           (!opt->true_ds4_attention_rope_gate ||
            opt->true_ds4_attention_state_gate) &&
           (!opt->true_ds4_attention_saturation_audit_gate ||
            opt->true_ds4_attention_rope_gate) &&
           (!opt->true_ds4_attention_kv_norm_reference_gate ||
            opt->true_ds4_attention_projection_gate) &&
           (!opt->true_ds4_attention_raw_read_gate ||
            opt->true_ds4_attention_state_gate) &&
           (!opt->true_ds4_attention_raw_window_gate ||
            opt->true_ds4_attention_raw_read_gate) &&
           (!opt->true_ds4_attention_typed_kv_raw_gate ||
            opt->true_ds4_attention_state_gate) &&
           (!opt->true_ds4_attention_typed_kv_compressed_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_attention_typed_kv_indexer_gate ||
            opt->true_ds4_indexer_attention_gate) &&
           (!opt->true_ds4_attention_typed_kv_history_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_attention_typed_kv_skip_current_load_gate ||
            (opt->true_ds4_attention_typed_kv_raw_gate ||
             opt->true_ds4_attention_typed_kv_compressed_gate ||
             opt->true_ds4_attention_typed_kv_indexer_gate)) &&
           (!opt->true_ds4_attention_typed_kv_skip_raw_store_gate ||
            opt->true_ds4_attention_typed_kv_raw_gate) &&
           (!opt->true_ds4_attention_typed_kv_skip_compressed_store_gate ||
            opt->true_ds4_attention_typed_kv_compressed_gate) &&
           (!opt->true_ds4_attention_typed_kv_skip_indexer_store_gate ||
            opt->true_ds4_attention_typed_kv_indexer_gate) &&
           (!opt->true_ds4_attention_typed_kv_quiet_gate ||
            (opt->true_ds4_attention_typed_kv_raw_gate ||
             opt->true_ds4_attention_typed_kv_compressed_gate ||
             opt->true_ds4_attention_typed_kv_indexer_gate ||
             opt->true_ds4_attention_typed_kv_history_gate)) &&
           (!opt->true_ds4_attention_typed_kv_batch_rows_gate ||
            (opt->true_ds4_attention_typed_kv_raw_gate ||
             opt->true_ds4_attention_typed_kv_compressed_gate ||
             opt->true_ds4_attention_typed_kv_indexer_gate ||
             opt->true_ds4_attention_typed_kv_history_gate)) &&
           (!opt->true_ds4_attention_typed_kv_stream_sync_gate ||
            (opt->true_ds4_attention_typed_kv_raw_gate ||
             opt->true_ds4_attention_typed_kv_compressed_gate ||
             opt->true_ds4_attention_typed_kv_indexer_gate ||
             opt->true_ds4_attention_typed_kv_history_gate)) &&
           (!opt->true_ds4_attention_output_gate ||
            opt->true_ds4_attention_raw_window_gate) &&
           (!opt->true_ds4_attention_output_nccl_allgather_gate ||
            opt->true_ds4_attention_output_gate) &&
           (!opt->true_ds4_post_attention_ffn_input_gate ||
            (opt->true_ds4_attention_output_gate && opt->true_shared_ffn_gate &&
             opt->model_router_routes && opt->routed_ffn_norm_input_gate)) &&
           (!opt->true_ds4_semantic_skip_stats_gate ||
            (opt->true_ds4_attention_output_gate ||
             opt->true_ds4_post_attention_ffn_input_gate)) &&
           (!opt->true_ds4_compressed_kv_gate ||
            opt->true_ds4_attention_projection_gate) &&
           (!opt->true_ds4_indexer_attention_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_direct_input_fill_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_dense_event_wait_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_skip_dense_stats_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_fused_attn_input_fill_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_fused_input_fill_gate ||
            opt->true_ds4_indexer_attention_gate) &&
           (!opt->true_ds4_compressed_kv_fused_rope_round_gate ||
            (opt->true_ds4_compressed_kv_gate &&
             opt->true_ds4_attention_rope_gate)) &&
           (!opt->true_ds4_compressed_kv_fused_pool_norm_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_fused_pool_norm_rope_round_gate ||
            (opt->true_ds4_compressed_kv_gate &&
             opt->true_ds4_attention_rope_gate)) &&
           (!opt->true_ds4_compressed_reference_diff_gate ||
            opt->true_ds4_indexer_attention_gate) &&
           !(opt->dense_compute_tensor &&
             (opt->dense_compute_all_f8 || opt->dense_compute_all_bf16));
}

bool parse_contract_row(const std::vector<std::string> &f, ContractRow *out) {
    if (f.size() < 23) return false;
    ContractRow r;
    r.record_type = f[0];
    r.tensor_id = f[1];
    if (!parse_int(f[3].c_str(), &r.layer)) return false;
    r.family = f[4];
    r.source_dtype = f[5];
    r.source_shape = f[6];
    r.runtime_layout = f[7];
    if (!parse_int(f[8].c_str(), &r.owning_gpu)) return false;
    if (!parse_int(f[9].c_str(), &r.tp_rank)) return false;
    if (!parse_int(f[10].c_str(), &r.ep_rank)) return false;
    if (!parse_int(f[12].c_str(), &r.shard_index)) return false;
    if (!parse_int(f[13].c_str(), &r.shard_count)) return false;
    if (!parse_int(f[14].c_str(), &r.expert_first)) return false;
    if (!parse_int(f[15].c_str(), &r.expert_count)) return false;
    if (!parse_int(f[16].c_str(), &r.kv_ratio)) return false;
    if (!parse_u64(f[17].c_str(), &r.kv_rows_per_slot)) return false;
    if (!parse_u64(f[18].c_str(), &r.bytes_estimate)) return false;
    r.source_pack_file = f[19];
    if (!parse_u64(f[20].c_str(), &r.source_shard_offset)) return false;
    if (!parse_u64(f[21].c_str(), &r.source_byte_length)) return false;
    r.kernel_family = f[22];
    if (!safe_sidecar_name(r.source_pack_file) && r.source_pack_file != "-") return false;
    *out = r;
    return true;
}

int parse_contract(const char *path, int layer, std::vector<ContractRow> *rows,
                   LayerStats *stats) {
    FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "cannot open contract %s: %s\n", path, std::strerror(errno));
        return 1;
    }
    char buf[8192];
    bool first = true;
    while (std::fgets(buf, sizeof(buf), fp)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (first) {
            first = false;
            continue;
        }
        if (line.empty()) continue;
        std::vector<std::string> f = split_tabs(line);
        ContractRow r;
        if (!parse_contract_row(f, &r)) {
            stats->bad_rows++;
            continue;
        }
        if (layer >= 0 && r.layer != layer) continue;
        if (r.owning_gpu < 0 || r.owning_gpu >= kGpus) {
            stats->bad_rows++;
            continue;
        }
        rows->push_back(r);
        stats->total_rows++;
        GpuFamilyStats &g = stats->gpu[r.owning_gpu];
        if (r.record_type == "dense_tp") {
            stats->dense_rows++;
            g.dense_rows++;
            g.dense_bytes += r.bytes_estimate;
        } else if (r.record_type == "replicated_control") {
            stats->control_rows++;
            g.control_rows++;
            g.control_bytes += r.bytes_estimate;
        } else if (r.record_type == "ep_expert") {
            stats->expert_rows++;
            g.expert_rows++;
            g.expert_descriptor_bytes += r.bytes_estimate;
        } else if (r.record_type == "kv_shard") {
            stats->kv_rows++;
            g.kv_rows++;
        } else if (r.record_type == "kv_comp_state") {
            stats->comp_rows++;
            g.comp_rows++;
        }
    }
    std::fclose(fp);
    return rows->empty() ? 2 : 0;
}

uint64_t physical_row_offset(const ContractRow &r) {
    if (r.record_type == "dense_tp" && r.shard_index >= 0 && r.shard_count > 1 &&
        r.source_byte_length >= r.bytes_estimate * (uint64_t)r.shard_count) {
        return r.source_shard_offset + (uint64_t)r.shard_index * r.bytes_estimate;
    }
    return r.source_shard_offset;
}

bool parse_shape2(const std::string &shape, int *cols, int *rows) {
    if (shape.size() < 5 || shape.front() != '[' || shape.back() != ']') return false;
    const size_t x = shape.find('x');
    if (x == std::string::npos) return false;
    std::string a = shape.substr(1, x - 1);
    std::string b = shape.substr(x + 1, shape.size() - x - 2);
    return parse_int(a.c_str(), cols) && parse_int(b.c_str(), rows) &&
           *cols > 0 && *rows > 0;
}

std::string layer_tensor_name(int layer, const char *suffix) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "blk.%d.%s", layer, suffix);
    return std::string(buf);
}

int ds4_layer_ratio(int layer) {
    if (layer < 2) return 0;
    return (layer % 2) == 0 ? 4 : 128;
}

int attn_comp_state_rows_for_ratio(int ratio) {
    if (ratio == 4) return 2 * ratio;
    return ratio > 0 ? ratio : 0;
}

int attn_comp_state_width_for_ratio(int ratio) {
    if (ratio == 4) return 2 * kHeadDim;
    return ratio > 0 ? kHeadDim : 0;
}

uint64_t f8_row_bytes(int cols) {
    return (uint64_t)(cols / 128) * 129ull;
}

float e8m0_to_f32_host(uint8_t e) {
    uint32_t bits = e == 0 ? 0x00400000u : ((uint32_t)e << 23);
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

float e4m3fn_to_f32_host(uint8_t x) {
    const uint8_t ax = x & 0x7fu;
    const bool sign = (x & 0x80u) != 0;
    if (ax == 0) return sign ? -0.0f : 0.0f;
    if (ax == 0x7f) return std::numeric_limits<float>::quiet_NaN();
    const int exp = (x >> 3) & 0x0f;
    const int man = x & 0x07;
    const float value = exp == 0 ? std::ldexp((float)man, -9)
                                 : std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
    return sign ? -value : value;
}

float cpu_f8_dot(const uint8_t *row, const float *x, int cols) {
    double acc = 0.0;
    const int blocks = cols / 128;
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *block = row + (uint64_t)b * 129ull;
        const float scale = e8m0_to_f32_host(block[0]);
        for (int c = 0; c < 128; ++c) {
            acc += (double)(e4m3fn_to_f32_host(block[1 + c]) * scale) *
                   (double)x[b * 128 + c];
        }
    }
    return (float)acc;
}

float bf16_to_f32_host(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float v = 0.0f;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

float cpu_bf16_dot(const uint16_t *row, const float *x, int cols) {
    double acc = 0.0;
    for (int c = 0; c < cols; ++c) {
        acc += (double)bf16_to_f32_host(row[c]) * (double)x[c];
    }
    return (float)acc;
}

int device_checksum_row(int device, const char *pack_dir, const ContractRow &r,
                        uint64_t *checksum) {
    if (r.bytes_estimate == 0 || r.source_pack_file == "-") return 0;
    CHECK_CUDA(cudaSetDevice(device));
    const uint64_t offset = physical_row_offset(r);
    if (offset + r.bytes_estimate > r.source_shard_offset + r.source_byte_length &&
        r.record_type == "dense_tp") {
        std::fprintf(stderr, "dense shard exceeds source span for %s\n", r.tensor_id.c_str());
        return 1;
    }
    std::vector<unsigned char> host((size_t)r.bytes_estimate);
    const std::string path = path_join(pack_dir, r.source_pack_file);
    if (read_exact_at(path, offset, host.data(), host.size()) != 0) return 2;

    unsigned char *d = nullptr;
    unsigned long long *d_sum = nullptr;
    CHECK_CUDA(cudaMalloc(&d, host.size()));
    CHECK_CUDA(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemcpy(d, host.data(), host.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    const int block = 256;
    const int grid = (int)std::min<uint64_t>(4096, (r.bytes_estimate + block - 1) / block);
    checksum_bytes_kernel<<<std::max(grid, 1), block>>>(d, r.bytes_estimate, d_sum);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    unsigned long long h_sum = 0;
    CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(h_sum), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d));
    CHECK_CUDA(cudaFree(d_sum));
    *checksum = (uint64_t)h_sum;
    return 0;
}

bool select_dense_rows(const std::vector<ContractRow> &rows,
                       const char *tensor,
                       std::vector<ContractRow> *selected,
                       int *cols,
                       int *total_rows) {
    selected->clear();
    for (const ContractRow &r : rows) {
        if (r.record_type == "dense_tp" && r.tensor_id == tensor) selected->push_back(r);
    }
    if ((int)selected->size() != kGpus) return false;
    std::sort(selected->begin(), selected->end(),
              [](const ContractRow &a, const ContractRow &b) {
                  return a.owning_gpu < b.owning_gpu;
              });
    int parsed_cols = 0;
    int parsed_rows = 0;
    if (!parse_shape2((*selected)[0].source_shape, &parsed_cols, &parsed_rows)) return false;
    for (int i = 0; i < kGpus; ++i) {
        const ContractRow &r = (*selected)[i];
        if (r.owning_gpu != i ||
            r.tp_rank != i ||
            r.shard_index != i ||
            r.shard_count != kGpus ||
            r.source_dtype != "f8_e4m3_b128" ||
            r.source_shape != (*selected)[0].source_shape) {
            return false;
        }
    }
    if (parsed_cols % 128 != 0 || parsed_rows % kGpus != 0) return false;
    const uint64_t row_bytes = f8_row_bytes(parsed_cols);
    const uint64_t rows_per_gpu = (uint64_t)parsed_rows / kGpus;
    for (const ContractRow &r : *selected) {
        if (r.bytes_estimate != row_bytes * rows_per_gpu) return false;
    }
    *cols = parsed_cols;
    *total_rows = parsed_rows;
    return true;
}

std::vector<std::string> discover_f8_dense_tensors(const std::vector<ContractRow> &rows) {
    std::vector<std::string> out;
    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" || r.source_dtype != "f8_e4m3_b128") continue;
        if (std::find(out.begin(), out.end(), r.tensor_id) == out.end()) {
            out.push_back(r.tensor_id);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool select_bf16_dense_rows(const std::vector<ContractRow> &rows,
                            const char *tensor,
                            std::vector<ContractRow> *selected,
                            int *cols,
                            int *total_rows) {
    selected->clear();
    for (const ContractRow &r : rows) {
        if (r.record_type == "dense_tp" && r.tensor_id == tensor) selected->push_back(r);
    }
    if ((int)selected->size() != kGpus) return false;
    std::sort(selected->begin(), selected->end(),
              [](const ContractRow &a, const ContractRow &b) {
                  return a.owning_gpu < b.owning_gpu;
              });
    int parsed_cols = 0;
    int parsed_rows = 0;
    if (!parse_shape2((*selected)[0].source_shape, &parsed_cols, &parsed_rows)) return false;
    for (int i = 0; i < kGpus; ++i) {
        const ContractRow &r = (*selected)[i];
        if (r.owning_gpu != i ||
            r.tp_rank != i ||
            r.shard_index != i ||
            r.shard_count != kGpus ||
            r.source_dtype != "bf16" ||
            r.source_shape != (*selected)[0].source_shape) {
            return false;
        }
    }
    if (parsed_rows % kGpus != 0) return false;
    const uint64_t rows_per_gpu = (uint64_t)parsed_rows / kGpus;
    const uint64_t shard_bytes = rows_per_gpu * (uint64_t)parsed_cols * sizeof(uint16_t);
    for (const ContractRow &r : *selected) {
        if (r.bytes_estimate != shard_bytes) return false;
    }
    *cols = parsed_cols;
    *total_rows = parsed_rows;
    return true;
}

std::vector<std::string> discover_bf16_dense_tensors(const std::vector<ContractRow> &rows) {
    std::vector<std::string> out;
    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" || r.source_dtype != "bf16") continue;
        if (std::find(out.begin(), out.end(), r.tensor_id) == out.end()) {
            out.push_back(r.tensor_id);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

int run_dense_compute_gate(const Options &opt,
                           const std::vector<ContractRow> &rows,
                           const char *tensor,
                           DenseComputeStats *stats) {
    if (!tensor) return 0;
    stats->enabled = true;
    stats->tensor_id = tensor;
    stats->slots = opt.slots;

    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    if (!select_dense_rows(rows, tensor, &selected, &cols, &total_rows)) {
        std::fprintf(stderr, "dense compute tensor validation failed for %s\n",
                     tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    const uint64_t row_bytes = f8_row_bytes(cols);
    const uint64_t shard_bytes = row_bytes * (uint64_t)rows_per_gpu;
    stats->rows_per_gpu = rows_per_gpu;
    stats->cols = cols;

    std::vector<float> h_x((size_t)opt.slots * cols);
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int c = 0; c < cols; ++c) {
            const int m = (slot * 17 + c * 13) % 257;
            h_x[(size_t)slot * cols + c] = ((float)m - 128.0f) * 0.00025f;
        }
    }

    double worst_ms = 0.0;
    std::vector<std::vector<uint8_t>> host_weights((size_t)kGpus);
    std::vector<std::vector<float>> host_outputs((size_t)kGpus);

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = selected[(size_t)gpu];
        host_weights[(size_t)gpu].resize((size_t)shard_bytes);
        host_outputs[(size_t)gpu].resize((size_t)opt.slots * rows_per_gpu);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r),
                          host_weights[(size_t)gpu].data(), (size_t)shard_bytes) != 0) {
            return 2;
        }
        stats->loaded_bytes += shard_bytes;

        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        uint8_t *d_w = nullptr;
        float *d_x = nullptr;
        float *d_out1 = nullptr;
        float *d_out2 = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaMalloc(&d_w, (size_t)shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out1, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out2, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_w, host_weights[(size_t)gpu].data(), (size_t)shard_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        const dim3 grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1);
        for (int i = 0; i < opt.warmup; ++i) {
            f8_b128_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                                cols, (uint32_t)row_bytes, opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < opt.iters; ++i) {
            f8_b128_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                                cols, (uint32_t)row_bytes, opt.slots);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        worst_ms = std::max(worst_ms, (double)ms / opt.iters);

        f8_b128_dense_kernel<<<grid, 256>>>(d_out2, d_w, d_x, rows_per_gpu,
                                            cols, (uint32_t)row_bytes, opt.slots);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> second(host_outputs[(size_t)gpu].size());
        CHECK_CUDA(cudaMemcpy(host_outputs[(size_t)gpu].data(), d_out1,
                              host_outputs[(size_t)gpu].size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(second.data(), d_out2,
                              second.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < second.size(); ++i) {
            const float a = host_outputs[(size_t)gpu][i];
            const float b = second[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                stats->repeat_nan++;
                stats->pass = false;
                continue;
            }
            const double diff = std::fabs((double)a - (double)b);
            stats->repeat_max_abs = std::max(stats->repeat_max_abs, diff);
            if (diff > 0.0) {
                stats->repeat_bad++;
                stats->pass = false;
            }
        }

        const int sample_slots = std::min(opt.slots, 2);
        const int sample_rows = std::min(rows_per_gpu, 4);
        for (int slot = 0; slot < sample_slots; ++slot) {
            for (int row = 0; row < sample_rows; ++row) {
                const float expected =
                    cpu_f8_dot(host_weights[(size_t)gpu].data() + (uint64_t)row * row_bytes,
                               h_x.data() + (size_t)slot * cols, cols);
                const float got = host_outputs[(size_t)gpu][(size_t)slot * rows_per_gpu + row];
                const double diff = std::fabs((double)expected - (double)got);
                stats->oracle_max_abs = std::max(stats->oracle_max_abs, diff);
                const double tol = 1.0e-4 + std::fabs((double)expected) * 1.0e-4;
                if (!std::isfinite(got) || diff > tol) {
                    stats->oracle_bad++;
                    stats->pass = false;
                }
            }
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_out1));
        CHECK_CUDA(cudaFree(d_out2));
    }

    stats->compute_ms = worst_ms;
    return stats->pass ? 0 : 3;
}

int run_bf16_dense_compute_gate(const Options &opt,
                                const std::vector<ContractRow> &rows,
                                const char *tensor,
                                DenseComputeStats *stats) {
    if (!tensor) return 0;
    stats->enabled = true;
    stats->tensor_id = tensor;
    stats->slots = opt.slots;

    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    if (!select_bf16_dense_rows(rows, tensor, &selected, &cols, &total_rows)) {
        std::fprintf(stderr, "bf16 dense compute tensor validation failed for %s\n",
                     tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    const uint64_t shard_bytes = (uint64_t)rows_per_gpu * cols * sizeof(uint16_t);
    stats->rows_per_gpu = rows_per_gpu;
    stats->cols = cols;

    std::vector<float> h_x((size_t)opt.slots * cols);
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int c = 0; c < cols; ++c) {
            const int m = (slot * 19 + c * 11) % 263;
            h_x[(size_t)slot * cols + c] = ((float)m - 131.0f) * 0.00025f;
        }
    }

    double worst_ms = 0.0;
    std::vector<std::vector<uint16_t>> host_weights((size_t)kGpus);
    std::vector<std::vector<float>> host_outputs((size_t)kGpus);

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = selected[(size_t)gpu];
        host_weights[(size_t)gpu].resize((size_t)rows_per_gpu * cols);
        host_outputs[(size_t)gpu].resize((size_t)opt.slots * rows_per_gpu);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r),
                          host_weights[(size_t)gpu].data(), (size_t)shard_bytes) != 0) {
            return 2;
        }
        stats->loaded_bytes += shard_bytes;

        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        uint16_t *d_w = nullptr;
        float *d_x = nullptr;
        float *d_out1 = nullptr;
        float *d_out2 = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaMalloc(&d_w, (size_t)shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out1, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out2, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_w, host_weights[(size_t)gpu].data(), (size_t)shard_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        const dim3 grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1);
        for (int i = 0; i < opt.warmup; ++i) {
            bf16_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                             cols, cols, opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < opt.iters; ++i) {
            bf16_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                             cols, cols, opt.slots);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        worst_ms = std::max(worst_ms, (double)ms / opt.iters);

        bf16_dense_kernel<<<grid, 256>>>(d_out2, d_w, d_x, rows_per_gpu,
                                         cols, cols, opt.slots);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> second(host_outputs[(size_t)gpu].size());
        CHECK_CUDA(cudaMemcpy(host_outputs[(size_t)gpu].data(), d_out1,
                              host_outputs[(size_t)gpu].size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(second.data(), d_out2,
                              second.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < second.size(); ++i) {
            const float a = host_outputs[(size_t)gpu][i];
            const float b = second[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                stats->repeat_nan++;
                stats->pass = false;
                continue;
            }
            const double diff = std::fabs((double)a - (double)b);
            stats->repeat_max_abs = std::max(stats->repeat_max_abs, diff);
            if (diff > 0.0) {
                stats->repeat_bad++;
                stats->pass = false;
            }
        }

        const int sample_slots = std::min(opt.slots, 2);
        const int sample_rows = std::min(rows_per_gpu, 4);
        for (int slot = 0; slot < sample_slots; ++slot) {
            for (int row = 0; row < sample_rows; ++row) {
                const float expected =
                    cpu_bf16_dot(host_weights[(size_t)gpu].data() + (uint64_t)row * cols,
                                 h_x.data() + (size_t)slot * cols, cols);
                const float got = host_outputs[(size_t)gpu][(size_t)slot * rows_per_gpu + row];
                const double diff = std::fabs((double)expected - (double)got);
                stats->oracle_max_abs = std::max(stats->oracle_max_abs, diff);
                const double tol = 1.0e-4 + std::fabs((double)expected) * 1.0e-4;
                if (!std::isfinite(got) || diff > tol) {
                    stats->oracle_bad++;
                    stats->pass = false;
                }
            }
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_out1));
        CHECK_CUDA(cudaFree(d_out2));
    }

    stats->compute_ms = worst_ms;
    return stats->pass ? 0 : 3;
}

struct SharedHcControls {
    bool initialized = false;
    int slots = 0;
    float *d_hc = nullptr;
    float *d_hc_norm = nullptr;
    float *d_mix = nullptr;
    float *d_split = nullptr;
    float *d_current_full = nullptr;
    float *d_attn_normed = nullptr;
    float *d_q_a_full = nullptr;
    float *d_q_a_normed = nullptr;
    float *d_kv_full = nullptr;
    float *d_kv_normed = nullptr;
    float *d_ffn_normed = nullptr;
    float *d_attn_comp_kv_full = nullptr;
    float *d_attn_comp_score_full = nullptr;
    float *d_index_comp_kv_full = nullptr;
    float *d_index_comp_score_full = nullptr;
    float *d_indexer_q_full = nullptr;
    float *d_indexer_w_full = nullptr;
    float *d_attn_norm_weight[43] = {};
    float *d_q_a_norm_weight[43] = {};
    float *d_kv_a_norm_weight[43] = {};
    float *d_attn_compress_ape[43] = {};
    float *d_attn_compress_norm[43] = {};
    float *d_indexer_compress_ape[43] = {};
    float *d_indexer_compress_norm[43] = {};
    float *d_attn_sinks[43] = {};
    float *d_attn_fn[43] = {};
    float *d_attn_base[43] = {};
    float *d_attn_scale[43] = {};
    float *d_ffn_fn[43] = {};
    float *d_ffn_base[43] = {};
    float *d_ffn_scale[43] = {};
    float *d_ffn_norm_weight[43] = {};
    float *d_router_w[43] = {};
    float *d_router_bias[43] = {};
    int *d_router_hash[43] = {};
    uint32_t router_hash_rows[43] = {};
    float *d_router_logits = nullptr;
    int *d_router_selected = nullptr;
    float *d_router_weights = nullptr;
    uint32_t *d_router_tokens = nullptr;
    unsigned char *d_router_active = nullptr;
    cublasHandle_t router_blas = nullptr;
    RoutePlanHostWorkspace route_plan_ws;
    uint64_t control_bytes = 0;
};

int init_route_plan_host_workspace(const Options &opt,
                                   RoutePlanHostWorkspace *ws) {
    if (!ws) return 1;
    if (ws->initialized) return 0;
    ws->slots = opt.slots;
    ws->top_k = opt.top_k;
    for (int rank = 0; rank < kGpus; ++rank) {
        ws->devices[rank] = opt.devices[rank];
    }
    ws->route_capacity = (size_t)opt.slots * (size_t)opt.top_k;
    ws->compact_plan_ints =
        (size_t)kGpus * ((size_t)opt.slots * (size_t)opt.top_k +
                         (size_t)opt.slots);
    CHECK_CUDA(cudaHostAlloc(&ws->h_selected,
                             ws->route_capacity * sizeof(int),
                             cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&ws->h_weights,
                             ws->route_capacity * sizeof(float),
                             cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&ws->h_compact_plan,
                             ws->compact_plan_ints * sizeof(int),
                             cudaHostAllocDefault));
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaHostAlloc(&ws->h_offsets[rank],
                                 (size_t)(kLocalExperts + 1) * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_slots[rank],
                                 ws->route_capacity * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_weights[rank],
                                 ws->route_capacity * sizeof(float),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_index_by_slot[rank],
                                 (size_t)opt.slots * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_indices_by_slot[rank],
                                 ws->route_capacity * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_count_by_slot[rank],
                                 (size_t)opt.slots * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
        CHECK_CUDA(cudaEventCreateWithFlags(&ws->upload_done[rank],
                                            cudaEventDisableTiming));
    }
    ws->initialized = true;
    return 0;
}

void close_route_plan_host_workspace(RoutePlanHostWorkspace *ws) {
    if (!ws || !ws->initialized) return;
    if (ws->uploads_pending) {
        for (int rank = 0; rank < kGpus; ++rank) {
            if (ws->upload_done[rank]) {
                CHECK_CUDA(cudaSetDevice(ws->devices[rank]));
                CHECK_CUDA(cudaEventSynchronize(ws->upload_done[rank]));
            }
        }
        ws->uploads_pending = false;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ws->devices[rank]));
        if (ws->upload_done[rank]) CHECK_CUDA(cudaEventDestroy(ws->upload_done[rank]));
        if (ws->h_route_count_by_slot[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_count_by_slot[rank]));
        if (ws->h_route_indices_by_slot[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_indices_by_slot[rank]));
        if (ws->h_route_index_by_slot[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_index_by_slot[rank]));
        if (ws->h_route_weights[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_weights[rank]));
        if (ws->h_route_slots[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_slots[rank]));
        if (ws->h_offsets[rank]) CHECK_CUDA(cudaFreeHost(ws->h_offsets[rank]));
    }
    if (ws->h_compact_plan) CHECK_CUDA(cudaFreeHost(ws->h_compact_plan));
    if (ws->h_weights) CHECK_CUDA(cudaFreeHost(ws->h_weights));
    if (ws->h_selected) CHECK_CUDA(cudaFreeHost(ws->h_selected));
    *ws = RoutePlanHostWorkspace{};
}

bool find_replicated_control_row(const std::vector<ContractRow> &rows,
                                 const char *tensor,
                                 ContractRow *out) {
    for (const ContractRow &r : rows) {
        if (r.record_type == "replicated_control" && r.tensor_id == tensor) {
            *out = r;
            return true;
        }
    }
    return false;
}

int load_control_f32(const Options &opt,
                     const std::vector<ContractRow> &rows,
                     const char *tensor,
                     size_t elems,
                     std::vector<float> *out) {
    ContractRow r;
    if (!find_replicated_control_row(rows, tensor, &r)) {
        std::fprintf(stderr, "missing replicated control tensor %s\n", tensor);
        return 1;
    }
    if (r.source_dtype != "f32" || r.bytes_estimate != elems * sizeof(float)) {
        std::fprintf(stderr, "bad replicated control tensor %s dtype=%s bytes=%llu expected=%zu\n",
                     tensor, r.source_dtype.c_str(),
                     (unsigned long long)r.bytes_estimate, elems * sizeof(float));
        return 2;
    }
    out->resize(elems);
    const std::string path = path_join(opt.pack_dir, r.source_pack_file);
    if (read_exact_at(path, physical_row_offset(r), out->data(), elems * sizeof(float)) != 0) {
        return 3;
    }
    return 0;
}

int load_optional_control_f32(const Options &opt,
                              const std::vector<ContractRow> &rows,
                              const char *tensor,
                              size_t elems,
                              std::vector<float> *out,
                              bool *found) {
    ContractRow r;
    if (!find_replicated_control_row(rows, tensor, &r)) {
        out->clear();
        if (found) *found = false;
        return 0;
    }
    if (found) *found = true;
    return load_control_f32(opt, rows, tensor, elems, out);
}

int load_optional_control_i32(const Options &opt,
                              const std::vector<ContractRow> &rows,
                              const char *tensor,
                              size_t elems,
                              std::vector<int> *out,
                              bool *found) {
    ContractRow r;
    if (!find_replicated_control_row(rows, tensor, &r)) {
        out->clear();
        if (found) *found = false;
        return 0;
    }
    if (found) *found = true;
    if (r.source_dtype != "i32" || r.bytes_estimate != elems * sizeof(int)) {
        std::fprintf(stderr, "bad replicated control tensor %s dtype=%s bytes=%llu expected=%zu\n",
                     tensor, r.source_dtype.c_str(),
                     (unsigned long long)r.bytes_estimate, elems * sizeof(int));
        return 2;
    }
    out->resize(elems);
    const std::string path = path_join(opt.pack_dir, r.source_pack_file);
    if (read_exact_at(path, physical_row_offset(r), out->data(), elems * sizeof(int)) != 0) {
        return 3;
    }
    return 0;
}

int open_shared_hc_controls(const Options &opt,
                            const std::vector<ContractRow> &rows,
                            SharedHcControls *out) {
    out->slots = opt.slots;
    const uint64_t hc_elems = (uint64_t)opt.slots * kHcRows * (uint64_t)kHidden;
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    cublasStatus_t blas_status = cublasCreate(&out->router_blas);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "router cublasCreate failed status=%d\n",
                     (int)blas_status);
        return 1;
    }
    CHECK_CUDA(cudaMalloc(&out->d_hc, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_hc_norm, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_mix, (size_t)opt.slots * kHcMix * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_split, (size_t)opt.slots * kHcMix * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_current_full,
                          (size_t)opt.slots * kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_attn_normed,
                          (size_t)opt.slots * kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_q_a_full,
                          (size_t)opt.slots * 1024u * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_q_a_normed,
                          (size_t)opt.slots * 1024u * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_kv_full,
                          (size_t)opt.slots * kHeadDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_kv_normed,
                          (size_t)opt.slots * kHeadDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_ffn_normed,
                          (size_t)opt.slots * kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_attn_comp_kv_full,
                          (size_t)opt.slots * kCompWidthMax * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_attn_comp_score_full,
                          (size_t)opt.slots * kCompWidthMax * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_index_comp_kv_full,
                          (size_t)opt.slots * kIndexCompWidth * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_index_comp_score_full,
                          (size_t)opt.slots * kIndexCompWidth * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_indexer_q_full,
                          (size_t)opt.slots * kIndexerHead *
                              (size_t)kIndexerHeadDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_indexer_w_full,
                          (size_t)opt.slots * kIndexerHead * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_router_logits,
                          (size_t)opt.slots * kGlobalExperts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_router_selected,
                          (size_t)opt.slots * kModelTopK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&out->d_router_weights,
                          (size_t)opt.slots * kModelTopK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_router_tokens,
                          (size_t)opt.slots * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&out->d_router_active,
                          (size_t)opt.slots * sizeof(unsigned char)));
    CHECK_CUDA(cudaMemset(out->d_router_tokens, 0,
                          (size_t)opt.slots * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(out->d_router_active, 1,
                          (size_t)opt.slots * sizeof(unsigned char)));
    if (opt.route_plan_async_upload_gate &&
        init_route_plan_host_workspace(opt, &out->route_plan_ws) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));

    for (int layer = 0; layer < 43; ++layer) {
        std::vector<float> attn_fn;
        std::vector<float> attn_base;
        std::vector<float> attn_scale;
        std::vector<float> fn;
        std::vector<float> base;
        std::vector<float> scale;
        std::vector<float> ffn_norm_weight;
        std::vector<float> attn_norm_weight;
        std::vector<float> q_a_norm_weight;
        std::vector<float> kv_a_norm_weight;
        std::vector<float> attn_sinks;
        std::vector<float> attn_compress_ape;
        std::vector<float> attn_compress_norm;
        std::vector<float> indexer_compress_ape;
        std::vector<float> indexer_compress_norm;
        std::vector<float> router_w;
        std::vector<float> router_bias;
        std::vector<int> router_hash;
        const std::string attn_norm_name = layer_tensor_name(layer, "attn_norm.weight");
        const std::string q_a_norm_name = layer_tensor_name(layer, "attn_q_a_norm.weight");
        const std::string kv_a_norm_name = layer_tensor_name(layer, "attn_kv_a_norm.weight");
        const std::string attn_sinks_name = layer_tensor_name(layer, "attn_sinks");
        const std::string attn_compress_ape_name =
            layer_tensor_name(layer, "attn_compress_ape");
        const std::string attn_compress_norm_name =
            layer_tensor_name(layer, "attn_compress_norm.weight");
        const std::string indexer_compress_ape_name =
            layer_tensor_name(layer, "indexer.compress_ape");
        const std::string indexer_compress_norm_name =
            layer_tensor_name(layer, "indexer.compress_norm.weight");
        const std::string attn_fn_name = layer_tensor_name(layer, "hc_attn_fn");
        const std::string attn_base_name = layer_tensor_name(layer, "hc_attn_base");
        const std::string attn_scale_name = layer_tensor_name(layer, "hc_attn_scale");
        const std::string fn_name = layer_tensor_name(layer, "hc_ffn_fn");
        const std::string base_name = layer_tensor_name(layer, "hc_ffn_base");
        const std::string scale_name = layer_tensor_name(layer, "hc_ffn_scale");
        const std::string ffn_norm_name = layer_tensor_name(layer, "ffn_norm.weight");
        const std::string router_name = layer_tensor_name(layer, "ffn_gate_inp.weight");
        const std::string bias_name = layer_tensor_name(layer, "exp_probs_b");
        const std::string hash_name = layer_tensor_name(layer, "ffn_gate_tid2eid");
        const int ratio = ds4_layer_ratio(layer);
        bool have_attn_compress_ape = false;
        bool have_attn_compress_norm = false;
        bool have_indexer_compress_ape = false;
        bool have_indexer_compress_norm = false;
        bool have_bias = false;
        bool have_hash = false;
        if (load_control_f32(opt, rows, attn_fn_name.c_str(),
                             (size_t)kHcRows * (size_t)kHidden * kHcMix, &attn_fn) ||
            load_control_f32(opt, rows, attn_base_name.c_str(), kHcMix, &attn_base) ||
            load_control_f32(opt, rows, attn_scale_name.c_str(), 3, &attn_scale) ||
            load_control_f32(opt, rows, attn_norm_name.c_str(),
                             kHidden, &attn_norm_weight) ||
            load_control_f32(opt, rows, q_a_norm_name.c_str(),
                             1024, &q_a_norm_weight) ||
            load_control_f32(opt, rows, kv_a_norm_name.c_str(),
                             kHeadDim, &kv_a_norm_weight) ||
            load_control_f32(opt, rows, attn_sinks_name.c_str(),
                             kHeadCount, &attn_sinks) ||
            (ratio != 0 &&
             (load_optional_control_f32(opt, rows, attn_compress_ape_name.c_str(),
                                        (size_t)ratio *
                                            (size_t)(ratio == 4 ? kCompWidthMax
                                                               : kHeadDim),
                                        &attn_compress_ape,
                                        &have_attn_compress_ape) ||
              load_optional_control_f32(opt, rows, attn_compress_norm_name.c_str(),
                                        kHeadDim, &attn_compress_norm,
                                        &have_attn_compress_norm))) ||
            (ratio == 4 &&
             (load_optional_control_f32(opt, rows, indexer_compress_ape_name.c_str(),
                                        (size_t)ratio * (size_t)kIndexCompWidth,
                                        &indexer_compress_ape,
                                        &have_indexer_compress_ape) ||
              load_optional_control_f32(opt, rows, indexer_compress_norm_name.c_str(),
                                        kIndexerHeadDim,
                                        &indexer_compress_norm,
                                        &have_indexer_compress_norm))) ||
            load_control_f32(opt, rows, fn_name.c_str(),
                             (size_t)kHcRows * (size_t)kHidden * kHcMix, &fn) ||
            load_control_f32(opt, rows, base_name.c_str(), kHcMix, &base) ||
            load_control_f32(opt, rows, scale_name.c_str(), 3, &scale) ||
            load_control_f32(opt, rows, ffn_norm_name.c_str(),
                             kHidden, &ffn_norm_weight) ||
            load_control_f32(opt, rows, router_name.c_str(),
                             (size_t)kHidden * kGlobalExperts, &router_w) ||
            load_optional_control_f32(opt, rows, bias_name.c_str(),
                                      kGlobalExperts, &router_bias, &have_bias) ||
            load_optional_control_i32(opt, rows, hash_name.c_str(),
                                      (size_t)kRouterHashRows * kModelTopK,
                                      &router_hash, &have_hash)) {
            return 1;
        }
        CHECK_CUDA(cudaMalloc(&out->d_attn_fn[layer], attn_fn.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_base[layer], attn_base.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_scale[layer], attn_scale.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_fn[layer], fn.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_base[layer], base.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_scale[layer], scale.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_norm_weight[layer],
                              ffn_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_norm_weight[layer],
                              attn_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_q_a_norm_weight[layer],
                              q_a_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_kv_a_norm_weight[layer],
                              kv_a_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_sinks[layer],
                              attn_sinks.size() * sizeof(float)));
        if (have_attn_compress_ape && have_attn_compress_norm) {
            CHECK_CUDA(cudaMalloc(&out->d_attn_compress_ape[layer],
                                  attn_compress_ape.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&out->d_attn_compress_norm[layer],
                                  attn_compress_norm.size() * sizeof(float)));
        }
        if (have_indexer_compress_ape && have_indexer_compress_norm) {
            CHECK_CUDA(cudaMalloc(&out->d_indexer_compress_ape[layer],
                                  indexer_compress_ape.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&out->d_indexer_compress_norm[layer],
                                  indexer_compress_norm.size() * sizeof(float)));
        }
        CHECK_CUDA(cudaMalloc(&out->d_router_w[layer], router_w.size() * sizeof(float)));
        if (have_bias) {
            CHECK_CUDA(cudaMalloc(&out->d_router_bias[layer],
                                  router_bias.size() * sizeof(float)));
        }
        if (have_hash) {
            CHECK_CUDA(cudaMalloc(&out->d_router_hash[layer],
                                  router_hash.size() * sizeof(int)));
            out->router_hash_rows[layer] = kRouterHashRows;
        }
        CHECK_CUDA(cudaMemcpy(out->d_attn_fn[layer], attn_fn.data(),
                              attn_fn.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_attn_base[layer], attn_base.data(),
                              attn_base.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_attn_scale[layer], attn_scale.data(),
                              attn_scale.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_fn[layer], fn.data(),
                              fn.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_base[layer], base.data(),
                              base.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_scale[layer], scale.data(),
                              scale.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_norm_weight[layer], ffn_norm_weight.data(),
                              ffn_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_attn_norm_weight[layer], attn_norm_weight.data(),
                              attn_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_q_a_norm_weight[layer], q_a_norm_weight.data(),
                              q_a_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_kv_a_norm_weight[layer], kv_a_norm_weight.data(),
                              kv_a_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_attn_sinks[layer], attn_sinks.data(),
                              attn_sinks.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (out->d_attn_compress_ape[layer]) {
            CHECK_CUDA(cudaMemcpy(out->d_attn_compress_ape[layer],
                                  attn_compress_ape.data(),
                                  attn_compress_ape.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(out->d_attn_compress_norm[layer],
                                  attn_compress_norm.data(),
                                  attn_compress_norm.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (out->d_indexer_compress_ape[layer]) {
            CHECK_CUDA(cudaMemcpy(out->d_indexer_compress_ape[layer],
                                  indexer_compress_ape.data(),
                                  indexer_compress_ape.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(out->d_indexer_compress_norm[layer],
                                  indexer_compress_norm.data(),
                                  indexer_compress_norm.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        CHECK_CUDA(cudaMemcpy(out->d_router_w[layer], router_w.data(),
                              router_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        if (have_bias) {
            CHECK_CUDA(cudaMemcpy(out->d_router_bias[layer], router_bias.data(),
                                  router_bias.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (have_hash) {
            CHECK_CUDA(cudaMemcpy(out->d_router_hash[layer], router_hash.data(),
                                  router_hash.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }
        out->control_bytes +=
            (attn_fn.size() + attn_base.size() + attn_scale.size() +
             attn_norm_weight.size() + q_a_norm_weight.size() +
             kv_a_norm_weight.size() + attn_sinks.size() +
             attn_compress_ape.size() + attn_compress_norm.size() +
             indexer_compress_ape.size() + indexer_compress_norm.size() +
             fn.size() + base.size() + scale.size() +
             ffn_norm_weight.size() + router_w.size() + router_bias.size()) *
                sizeof(float) +
            router_hash.size() * sizeof(int);
    }
    out->initialized = true;
    return 0;
}

void close_shared_hc_controls(const Options &opt, SharedHcControls *out) {
    if (!out || !out->initialized) return;
    close_route_plan_host_workspace(&out->route_plan_ws);
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    if (out->router_blas) {
        cublasStatus_t st = cublasDestroy(out->router_blas);
        if (st != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "router cublasDestroy failed status=%d\n", (int)st);
        }
    }
    for (int layer = 0; layer < 43; ++layer) {
        if (out->d_router_hash[layer]) CHECK_CUDA(cudaFree(out->d_router_hash[layer]));
        if (out->d_router_bias[layer]) CHECK_CUDA(cudaFree(out->d_router_bias[layer]));
        if (out->d_router_w[layer]) CHECK_CUDA(cudaFree(out->d_router_w[layer]));
        if (out->d_indexer_compress_norm[layer]) CHECK_CUDA(cudaFree(out->d_indexer_compress_norm[layer]));
        if (out->d_indexer_compress_ape[layer]) CHECK_CUDA(cudaFree(out->d_indexer_compress_ape[layer]));
        if (out->d_attn_compress_norm[layer]) CHECK_CUDA(cudaFree(out->d_attn_compress_norm[layer]));
        if (out->d_attn_compress_ape[layer]) CHECK_CUDA(cudaFree(out->d_attn_compress_ape[layer]));
        if (out->d_attn_sinks[layer]) CHECK_CUDA(cudaFree(out->d_attn_sinks[layer]));
        if (out->d_kv_a_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_kv_a_norm_weight[layer]));
        if (out->d_q_a_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_q_a_norm_weight[layer]));
        if (out->d_attn_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_attn_norm_weight[layer]));
        if (out->d_ffn_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_ffn_norm_weight[layer]));
        if (out->d_ffn_scale[layer]) CHECK_CUDA(cudaFree(out->d_ffn_scale[layer]));
        if (out->d_ffn_base[layer]) CHECK_CUDA(cudaFree(out->d_ffn_base[layer]));
        if (out->d_ffn_fn[layer]) CHECK_CUDA(cudaFree(out->d_ffn_fn[layer]));
        if (out->d_attn_scale[layer]) CHECK_CUDA(cudaFree(out->d_attn_scale[layer]));
        if (out->d_attn_base[layer]) CHECK_CUDA(cudaFree(out->d_attn_base[layer]));
        if (out->d_attn_fn[layer]) CHECK_CUDA(cudaFree(out->d_attn_fn[layer]));
    }
    if (out->d_router_weights) CHECK_CUDA(cudaFree(out->d_router_weights));
    if (out->d_router_selected) CHECK_CUDA(cudaFree(out->d_router_selected));
    if (out->d_router_logits) CHECK_CUDA(cudaFree(out->d_router_logits));
    if (out->d_router_tokens) CHECK_CUDA(cudaFree(out->d_router_tokens));
    if (out->d_router_active) CHECK_CUDA(cudaFree(out->d_router_active));
    if (out->d_index_comp_score_full) CHECK_CUDA(cudaFree(out->d_index_comp_score_full));
    if (out->d_index_comp_kv_full) CHECK_CUDA(cudaFree(out->d_index_comp_kv_full));
    if (out->d_indexer_w_full) CHECK_CUDA(cudaFree(out->d_indexer_w_full));
    if (out->d_indexer_q_full) CHECK_CUDA(cudaFree(out->d_indexer_q_full));
    if (out->d_attn_comp_score_full) CHECK_CUDA(cudaFree(out->d_attn_comp_score_full));
    if (out->d_attn_comp_kv_full) CHECK_CUDA(cudaFree(out->d_attn_comp_kv_full));
    if (out->d_ffn_normed) CHECK_CUDA(cudaFree(out->d_ffn_normed));
    if (out->d_kv_normed) CHECK_CUDA(cudaFree(out->d_kv_normed));
    if (out->d_kv_full) CHECK_CUDA(cudaFree(out->d_kv_full));
    if (out->d_q_a_normed) CHECK_CUDA(cudaFree(out->d_q_a_normed));
    if (out->d_q_a_full) CHECK_CUDA(cudaFree(out->d_q_a_full));
    if (out->d_attn_normed) CHECK_CUDA(cudaFree(out->d_attn_normed));
    if (out->d_current_full) CHECK_CUDA(cudaFree(out->d_current_full));
    if (out->d_split) CHECK_CUDA(cudaFree(out->d_split));
    if (out->d_mix) CHECK_CUDA(cudaFree(out->d_mix));
    if (out->d_hc_norm) CHECK_CUDA(cudaFree(out->d_hc_norm));
    if (out->d_hc) CHECK_CUDA(cudaFree(out->d_hc));
    *out = SharedHcControls{};
}

int upload_model_router_route_plan(const Options &opt,
                                   RankState ranks[kGpus],
                                   const std::vector<int> &selected,
                                   const std::vector<float> &weights);
int upload_model_router_route_plan_async(const Options &opt,
                                         RankState ranks[kGpus],
                                         const int *selected,
                                         const float *weights,
                                         RoutePlanHostWorkspace *ws);
int upload_model_router_route_plan_gpu(const Options &opt,
                                       SharedHcControls *hc,
                                       RankState ranks[kGpus]);
int enqueue_dense_wait_after_rank_stream(RankState ranks[kGpus]);

int run_model_router_dense_logits(const Options &opt,
                                  SharedHcControls *hc,
                                  int layer,
                                  cudaStream_t stream) {
    if (!hc || !hc->d_router_w[layer] || !hc->d_router_logits ||
        !hc->d_ffn_normed) {
        return 1;
    }
    if (!opt.router_cublas_gate) {
        const dim3 router_grid((unsigned int)kGlobalExperts,
                               (unsigned int)opt.slots, 1u);
        f32_dense_colmajor_kernel<<<router_grid, 256, 0, stream>>>(
            hc->d_router_logits, hc->d_router_w[layer], hc->d_ffn_normed,
            (uint32_t)kGlobalExperts, (uint32_t)kHidden, (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
        return 0;
    }
    if (!hc->router_blas) return 2;
    cublasStatus_t st = cublasSetStream(hc->router_blas, stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "router cublasSetStream failed status=%d\n", (int)st);
        return 3;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    st = cublasSgemm(hc->router_blas,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     kGlobalExperts, opt.slots, kHidden,
                     &alpha,
                     hc->d_router_w[layer], kGlobalExperts,
                     hc->d_ffn_normed, kHidden,
                     &beta,
                     hc->d_router_logits, kGlobalExperts);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "router cublasSgemm failed layer=%d status=%d\n",
                     layer, (int)st);
        return 4;
    }
    return 0;
}

int run_shared_hc_final_expand(const Options &opt,
                               SharedHcControls *hc,
                               RankState ranks[kGpus],
                               int layer) {
    if (!hc || !hc->initialized || hc->slots != opt.slots ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    const uint64_t shard_elems =
        (uint64_t)opt.slots * (uint64_t)(kHidden / kGpus);
    const uint64_t hc_shard_elems = shard_elems * kHcRows;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    cudaStream_t control_stream = graph_event_order ? ranks[0].stream : (cudaStream_t)0;
    auto control_wait_on_rank_streams = [&]() -> int {
        if (!graph_event_order) return 0;
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            if (!r.stream_done) return 1;
            CHECK_CUDA(cudaEventRecord(r.stream_done, r.stream));
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaStreamWaitEvent(control_stream,
                                           ranks[rank].stream_done, 0));
        }
        return 0;
    };
    auto rank_streams_wait_on_control = [&]() -> int {
        if (!graph_event_order) return 0;
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        if (!ranks[0].stream_done) return 1;
        CHECK_CUDA(cudaEventRecord(ranks[0].stream_done, control_stream));
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_CUDA(cudaStreamWaitEvent(r.stream, ranks[0].stream_done, 0));
        }
        return 0;
    };
    if (graph_event_order) {
        if (control_wait_on_rank_streams() != 0) return 4;
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        if (!ranks[rank].d_final_hc_shard) return 2;
        gather_hc_shard_to_full_kernel<<<
            (unsigned int)((hc_shard_elems + 255) / 256), 256, 0,
            control_stream>>>(
            hc->d_hc, ranks[rank].d_final_hc_shard, rank, (uint32_t)opt.slots);
    }
    CHECK_CUDA(cudaGetLastError());
    if (!graph_event_order) {
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    rms_norm_plain_rows_stable_kernel<<<
        (unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_hc_norm, hc->d_hc, kHcRows * (uint32_t)kHidden,
        (uint32_t)opt.slots, 1.0e-6f);
    const dim3 mix_grid((unsigned int)kHcMix, (unsigned int)opt.slots, 1u);
    f32_dense_colmajor_kernel<<<mix_grid, 256, 0, control_stream>>>(
        hc->d_mix, hc->d_ffn_fn[layer], hc->d_hc_norm,
        (uint32_t)kHcMix, kHcRows * (uint32_t)kHidden, (uint32_t)opt.slots);
    hc_split_rows_kernel<<<
        (unsigned int)(((uint64_t)opt.slots + 255) / 256), 256, 0,
        control_stream>>>(
        hc->d_split, hc->d_mix, hc->d_ffn_scale[layer], hc->d_ffn_base[layer],
        (uint32_t)opt.slots, opt.reference_hc_reduce_gate ? 20u : 4u);
    CHECK_CUDA(cudaGetLastError());
    if (graph_event_order) {
        if (rank_streams_wait_on_control() != 0) return 5;
    } else {
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    const int block = 256;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_hc_scratch_shard || !r.d_hc_split) return 3;
        if (graph_event_order) {
            copy_f32_kernel<<<
                (unsigned int)(((uint64_t)opt.slots * kHcMix + block - 1) / block),
                block, 0, r.stream>>>(
                r.d_hc_split, hc->d_split, (uint64_t)opt.slots * kHcMix);
            CHECK_CUDA(cudaGetLastError());
        } else if (rank == 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_hc_split, hc->d_split,
                                       (size_t)opt.slots * kHcMix * sizeof(float),
                                       cudaMemcpyDeviceToDevice, r.stream));
        } else {
            CHECK_CUDA(cudaMemcpyPeerAsync(r.d_hc_split, r.device,
                                           hc->d_split, opt.devices[0],
                                           (size_t)opt.slots * kHcMix * sizeof(float),
                                           r.stream));
        }
        const int grid = (int)((hc_shard_elems + block - 1) / block);
        hc_expand_shard_kernel<<<grid, block, 0, r.stream>>>(
            r.d_hc_scratch_shard, r.d_next_hidden, r.d_final_hc_shard,
            r.d_hc_split, (uint32_t)opt.slots);
        if (opt.reference_hc_state_guard_gate) {
            clamp_f32_abs_kernel<<<grid, block, 0, r.stream>>>(
                r.d_hc_scratch_shard, hc_shard_elems,
                kReferenceHcStateTargetAbs);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    if (graph_event_order) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            std::swap(r.d_final_hc_shard, r.d_hc_scratch_shard);
            r.hc_initialized = true;
        }
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_CUDA(cudaStreamSynchronize(r.stream));
            std::swap(r.d_final_hc_shard, r.d_hc_scratch_shard);
            r.hc_initialized = true;
        }
    }
    return 0;
}

int run_shared_hc_current_input(const Options &opt,
                                SharedHcControls *hc,
                                RankState ranks[kGpus],
                                const ResidentF8Dense &attn_op,
                                const ResidentF8Dense &shared_op,
                                int layer,
                                HcCurrentInputBreakdown *breakdown) {
    if (!hc || !hc->initialized || hc->slots != opt.slots ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    if (attn_op.cols <= 0 || shared_op.cols <= 0) return 2;
    const uint64_t shard_elems =
        (uint64_t)opt.slots * (uint64_t)(kHidden / kGpus);
    const uint64_t hc_shard_elems = shard_elems * kHcRows;
    const uint64_t full_elems = (uint64_t)opt.slots * kHidden;
    const int block = 256;
    const auto t_start = std::chrono::steady_clock::now();
    const bool graph_event_order = opt.decode_cudagraph_gate;
    cudaStream_t control_stream =
        (opt.tp_hc_current_input_stream_sync_gate || graph_event_order)
            ? ranks[0].stream
            : (cudaStream_t)0;
    auto control_wait_on_rank_streams = [&]() -> int {
        if (!graph_event_order) return 0;
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            if (!r.stream_done) return 1;
            CHECK_CUDA(cudaEventRecord(r.stream_done, r.stream));
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaStreamWaitEvent(control_stream,
                                           ranks[rank].stream_done, 0));
        }
        return 0;
    };
    auto rank_streams_wait_on_control = [&]() -> int {
        if (!graph_event_order) return 0;
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        if (!ranks[0].stream_done) return 1;
        CHECK_CUDA(cudaEventRecord(ranks[0].stream_done, control_stream));
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_CUDA(cudaStreamWaitEvent(r.stream, ranks[0].stream_done, 0));
        }
        return 0;
    };
    auto sync_control_device = [&]() {
        if (graph_event_order) return;
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        if (opt.tp_hc_current_input_stream_sync_gate) {
            CHECK_CUDA(cudaStreamSynchronize(control_stream));
        } else {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    };

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_final_hc_shard || !r.d_current_shard || !r.d_current_full ||
            !r.d_hc_split) {
            return 3;
        }
        if (!r.hc_initialized) {
            seed_initial_hc_shard_kernel<<<
                (unsigned int)((hc_shard_elems + block - 1) / block), block,
                0, r.stream>>>(r.d_final_hc_shard, rank, opt.slots);
            CHECK_CUDA(cudaGetLastError());
            r.hc_initialized = true;
        }
    }
    if (graph_event_order) {
        if (control_wait_on_rank_streams() != 0) return 6;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    auto t_seed_done = std::chrono::steady_clock::now();
    if (should_log_reference_hc_window(opt)) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("hc_current_shard", layer, rank,
                                 ranks[rank].d_current_shard,
                                 (size_t)shard_elems, ranks[rank].stream);
        }
    }

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        gather_hc_shard_to_full_kernel<<<
            (unsigned int)((hc_shard_elems + block - 1) / block), block,
            0, control_stream>>>(
            hc->d_hc, ranks[rank].d_final_hc_shard, rank, (uint32_t)opt.slots);
    }
    CHECK_CUDA(cudaGetLastError());
    sync_control_device();

    rms_norm_plain_rows_stable_kernel<<<(unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_hc_norm, hc->d_hc, kHcRows * (uint32_t)kHidden,
        (uint32_t)opt.slots, 1.0e-6f);
    const dim3 mix_grid((unsigned int)kHcMix, (unsigned int)opt.slots, 1u);
    f32_dense_colmajor_kernel<<<mix_grid, 256, 0, control_stream>>>(
        hc->d_mix, hc->d_attn_fn[layer], hc->d_hc_norm,
        (uint32_t)kHcMix, kHcRows * (uint32_t)kHidden, (uint32_t)opt.slots);
    hc_split_rows_kernel<<<
        (unsigned int)(((uint64_t)opt.slots + 255) / 256), 256, 0,
        control_stream>>>(
        hc->d_split, hc->d_mix, hc->d_attn_scale[layer], hc->d_attn_base[layer],
        (uint32_t)opt.slots, opt.reference_hc_reduce_gate ? 20u : 4u);
    CHECK_CUDA(cudaGetLastError());
    sync_control_device();
    if (graph_event_order) {
        if (rank_streams_wait_on_control() != 0) return 7;
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (graph_event_order) {
            copy_f32_kernel<<<
                (unsigned int)(((uint64_t)opt.slots * kHcMix + block - 1) / block),
                block, 0, r.stream>>>(
                r.d_hc_split, hc->d_split, (uint64_t)opt.slots * kHcMix);
            CHECK_CUDA(cudaGetLastError());
        } else if (rank == 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_hc_split, hc->d_split,
                                       (size_t)opt.slots * kHcMix * sizeof(float),
                                       cudaMemcpyDeviceToDevice, r.stream));
        } else {
            CHECK_CUDA(cudaMemcpyPeerAsync(r.d_hc_split, r.device,
                                           hc->d_split, opt.devices[0],
                                           (size_t)opt.slots * kHcMix * sizeof(float),
                                           r.stream));
        }
        hc_weighted_sum_shard_kernel<<<
            (unsigned int)((shard_elems + block - 1) / block), block,
            0, r.stream>>>(r.d_current_shard, r.d_final_hc_shard,
                           r.d_hc_split, (uint32_t)opt.slots,
                           opt.reference_hc_reduce_gate ? 1 : 0);
        CHECK_CUDA(cudaGetLastError());
    }
    auto t_split_done = std::chrono::steady_clock::now();
    if (graph_event_order) {
        if (control_wait_on_rank_streams() != 0) return 8;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    auto t_weighted_done = std::chrono::steady_clock::now();

    float *control_current_full = hc->d_current_full;
    const bool peer_gather_current = opt.tp_hc_current_input_peer_gather_gate;
    const bool nccl_gather_current =
        opt.tp_hc_current_input_nccl_allgather_gate;
    const bool rank_local_current_full =
        peer_gather_current || nccl_gather_current;
    if (nccl_gather_current) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            if (!r.compose_nccl_initialized || !r.compose_nccl ||
                !r.d_current_full_rank_major) {
                return 9;
            }
        }
        CHECK_NCCL(ncclGroupStart());
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_NCCL(ncclAllGather(r.d_current_shard,
                                     r.d_current_full_rank_major,
                                     (size_t)shard_elems,
                                     ncclFloat,
                                     r.compose_nccl,
                                     r.stream));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            rank_major_current_shards_to_slot_major_kernel<<<
                (unsigned int)((full_elems + block - 1) / block), block, 0,
                r.stream>>>(
                r.d_current_full, r.d_current_full_rank_major,
                (uint32_t)(kHidden / kGpus), (uint32_t)kGpus,
                (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        if (graph_event_order) {
            if (control_wait_on_rank_streams() != 0) return 9;
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
        control_current_full = ranks[0].d_current_full;
    } else if (peer_gather_current) {
        const uint64_t full_grid_elems = full_elems;
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            gather_current_shards_to_full8_kernel<<<
                (unsigned int)((full_grid_elems + block - 1) / block), block,
                0, r.stream>>>(r.d_current_full,
                               ranks[0].d_current_shard,
                               ranks[1].d_current_shard,
                               ranks[2].d_current_shard,
                               ranks[3].d_current_shard,
                               ranks[4].d_current_shard,
                               ranks[5].d_current_shard,
                               ranks[6].d_current_shard,
                               ranks[7].d_current_shard,
                               (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        if (graph_event_order) {
            if (control_wait_on_rank_streams() != 0) return 9;
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
        control_current_full = ranks[0].d_current_full;
    } else {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            gather_current_shard_to_full_kernel<<<
                (unsigned int)((shard_elems + block - 1) / block), block,
                0, control_stream>>>(
                hc->d_current_full, ranks[rank].d_current_shard, rank,
                (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();
    }
    auto t_gather_done = std::chrono::steady_clock::now();
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    if (should_log_reference_hc_window(opt)) {
        log_tensor_f32_stats("hc_current_full", layer, 0, control_current_full,
                             (size_t)full_elems, nullptr);
    }

    if (!hc->d_ffn_normed || !hc->d_ffn_norm_weight[layer]) return 4;
    rms_norm_weight_rows_stable_kernel<<<(unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_ffn_normed, control_current_full, hc->d_ffn_norm_weight[layer],
        (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());
    sync_control_device();
    auto t_norm_done = std::chrono::steady_clock::now();
    if (graph_event_order) {
        if (rank_streams_wait_on_control() != 0) return 10;
    }
    if (should_log_reference_hc_window(opt)) {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        log_tensor_f32_stats("hc_ffn_normed", layer, 0, hc->d_ffn_normed,
                             (size_t)full_elems, nullptr);
    }

    auto t_router_select_done = t_norm_done;
    auto t_router_d2h_done = t_norm_done;
    auto t_route_upload_done = t_norm_done;
    if (opt.model_router_routes) {
        if (!hc->d_router_w[layer] || !hc->d_router_logits ||
            !hc->d_router_selected || !hc->d_router_weights) {
            return 4;
        }
        const int router_dense_rc =
            run_model_router_dense_logits(opt, hc, layer, control_stream);
        if (router_dense_rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_model_router_dense_failed\tlayer\t%d\trc\t%d\n",
                         layer, router_dense_rc);
            return 4;
        }
        if (opt.router_hash_fast_gate && hc->d_router_hash[layer] &&
            hc->d_router_tokens && hc->router_hash_rows[layer] > 0u) {
            router_select_hash_fast_rows_kernel<<<
                (unsigned int)opt.slots, 1, 0, control_stream>>>(
                hc->d_router_selected, hc->d_router_weights,
                hc->d_router_logits, hc->d_router_hash[layer],
                hc->d_router_tokens, hc->d_router_active,
                hc->router_hash_rows[layer], (uint32_t)opt.slots);
        } else {
            router_select_topk_rows_kernel<<<
                (unsigned int)opt.slots, 1, 0, control_stream>>>(
                hc->d_router_selected, hc->d_router_weights,
                hc->d_router_logits, hc->d_router_bias[layer],
                hc->d_router_hash[layer], hc->d_router_tokens,
                hc->d_router_active, hc->router_hash_rows[layer],
                (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();
        t_router_select_done = std::chrono::steady_clock::now();
        int route_rc = 0;
        if (opt.gpu_route_plan_gate) {
            t_router_d2h_done = t_router_select_done;
            route_rc = upload_model_router_route_plan_gpu(opt, hc, ranks);
        } else if (opt.route_plan_async_upload_gate) {
            RoutePlanHostWorkspace *ws = &hc->route_plan_ws;
            if (!ws->initialized) return 5;
            const size_t route_elems = (size_t)opt.slots * (size_t)opt.top_k;
            CHECK_CUDA(cudaMemcpyAsync(ws->h_selected, hc->d_router_selected,
                                       route_elems * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       control_stream));
            CHECK_CUDA(cudaMemcpyAsync(ws->h_weights, hc->d_router_weights,
                                       route_elems * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       control_stream));
            CHECK_CUDA(cudaStreamSynchronize(control_stream));
            t_router_d2h_done = std::chrono::steady_clock::now();
            route_rc = upload_model_router_route_plan_async(
                opt, ranks, ws->h_selected, ws->h_weights, ws);
        } else {
            std::vector<int> selected((size_t)opt.slots * (size_t)opt.top_k);
            std::vector<float> weights((size_t)opt.slots * (size_t)opt.top_k);
            CHECK_CUDA(cudaMemcpy(selected.data(), hc->d_router_selected,
                                  selected.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(weights.data(), hc->d_router_weights,
                                  weights.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            t_router_d2h_done = std::chrono::steady_clock::now();
            route_rc = upload_model_router_route_plan(opt, ranks,
                                                      selected, weights);
        }
        if (route_rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_model_router_route_plan_failed\tlayer\t%d\trc\t%d\n",
                         layer, route_rc);
            return 5;
        }
        t_route_upload_done = std::chrono::steady_clock::now();
    } else {
        t_router_select_done = t_norm_done;
        t_router_d2h_done = t_norm_done;
        t_route_upload_done = t_norm_done;
    }
    auto t_router_done = t_route_upload_done;

    const bool fused_fill_pack =
        opt.tp_hc_current_input_fused_fill_pack_gate &&
        !rank_local_current_full && !graph_event_order &&
        !opt.reference_hc_reduce_gate &&
        (!opt.routed_ffn_norm_input_gate || hc->d_ffn_normed);
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        const size_t full_bytes = (size_t)full_elems * sizeof(float);
        const uint64_t attn_elems = (uint64_t)opt.slots * (uint64_t)attn_op.cols;
        const uint64_t shared_elems = (uint64_t)opt.slots * (uint64_t)shared_op.cols;
        const uint64_t route_elems = (uint64_t)r.routes * kHidden;
        if (fused_fill_pack) {
            const float *state_src =
                (opt.routed_ffn_norm_input_gate && route_elems > 0)
                    ? hc->d_ffn_normed
                    : hc->d_current_full;
            const float *route_src =
                opt.routed_ffn_norm_input_gate ? hc->d_ffn_normed : hc->d_current_full;
            const uint64_t total = std::max(
                std::max(full_elems, attn_elems),
                std::max(shared_elems, route_elems));
            hc_current_fused_fill_pack_kernel<<<
                (unsigned int)((total + block - 1) / block), block,
                0, r.stream>>>(
                r.d_current_full, state_src, hc->d_current_full, route_src,
                attn_op.d_x[(size_t)rank], (uint32_t)attn_op.cols,
                shared_op.d_x[(size_t)rank], (uint32_t)shared_op.cols,
                attn_op.d_x_half[(size_t)rank],
                shared_op.d_x_half[(size_t)rank],
                route_elems > 0 ? r.d_a : nullptr,
                route_elems > 0 ? r.d_route_slots : nullptr,
                r.routes, (uint32_t)opt.slots, total);
            CHECK_CUDA(cudaGetLastError());
        } else {
            if (!rank_local_current_full) {
                if (graph_event_order) {
                copy_f32_kernel<<<
                    (unsigned int)((full_elems + block - 1) / block),
                    block, 0, r.stream>>>(
                    r.d_current_full, hc->d_current_full, full_elems);
                CHECK_CUDA(cudaGetLastError());
                } else if (rank == 0) {
                    CHECK_CUDA(cudaMemcpyAsync(r.d_current_full, hc->d_current_full,
                                               full_bytes, cudaMemcpyDeviceToDevice, r.stream));
                } else {
                    CHECK_CUDA(cudaMemcpyPeerAsync(r.d_current_full, r.device,
                                                   hc->d_current_full, opt.devices[0],
                                                   full_bytes, r.stream));
                }
            }
            if (attn_op.d_x[(size_t)rank]) {
                fill_dense_input_from_current_kernel<<<
                    (unsigned int)((attn_elems + block - 1) / block), block,
                    0, r.stream>>>(attn_op.d_x[(size_t)rank], r.d_current_full,
                                   (uint32_t)attn_op.cols, (uint32_t)opt.slots);
            }
            if (shared_op.d_x[(size_t)rank]) {
                fill_dense_input_from_current_kernel<<<
                    (unsigned int)((shared_elems + block - 1) / block), block,
                    0, r.stream>>>(shared_op.d_x[(size_t)rank], r.d_current_full,
                                   (uint32_t)shared_op.cols, (uint32_t)opt.slots);
            }
            if (attn_op.d_x_half[(size_t)rank]) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((attn_elems + block - 1) / block), block,
                    0, r.stream>>>(attn_op.d_x_half[(size_t)rank], r.d_current_full,
                                   (uint32_t)attn_op.cols, (uint32_t)opt.slots);
            }
            if (shared_op.d_x_half[(size_t)rank]) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((shared_elems + block - 1) / block), block,
                    0, r.stream>>>(shared_op.d_x_half[(size_t)rank],
                                   r.d_current_full, (uint32_t)shared_op.cols,
                                   (uint32_t)opt.slots);
            }
            if (opt.routed_ffn_norm_input_gate && route_elems > 0) {
                if (graph_event_order) {
                    copy_f32_kernel<<<
                        (unsigned int)((full_elems + block - 1) / block),
                        block, 0, r.stream>>>(
                        r.d_current_full, hc->d_ffn_normed, full_elems);
                    CHECK_CUDA(cudaGetLastError());
                } else if (rank == 0) {
                    CHECK_CUDA(cudaMemcpyAsync(r.d_current_full, hc->d_ffn_normed,
                                               full_bytes, cudaMemcpyDeviceToDevice,
                                               r.stream));
                } else {
                    CHECK_CUDA(cudaMemcpyPeerAsync(r.d_current_full, r.device,
                                                   hc->d_ffn_normed, opt.devices[0],
                                                   full_bytes, r.stream));
                }
            }
            if (route_elems > 0) {
                if (opt.reference_hc_reduce_gate) {
                    pack_current_full_to_routes_scaled_kernel<<<
                        (unsigned int)r.routes, 256, 0, r.stream>>>(
                            r.d_a, r.d_route_inv_scale, r.d_current_full,
                            r.d_route_slots, r.routes, kReferenceRouteInputTargetAbs);
                } else {
                    pack_current_full_to_routes_kernel<<<
                        (unsigned int)((route_elems + block - 1) / block), block,
                        0, r.stream>>>(r.d_a, r.d_current_full, r.d_route_slots, r.routes);
                }
                CHECK_CUDA(cudaGetLastError());
            }
        }
        if (should_log_reference_hc_window(opt) && r.d_route_inv_scale && r.routes > 0) {
            log_tensor_f32_stats("route_inv_scale", layer, rank,
                                 r.d_route_inv_scale, (size_t)r.routes,
                                 r.stream);
        }
    }
    if (graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 11;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    auto t_fill_done = std::chrono::steady_clock::now();
    if (breakdown) {
        breakdown->seed_ms +=
            std::chrono::duration<double, std::milli>(t_seed_done - t_start).count();
        breakdown->attn_mix_ms +=
            std::chrono::duration<double, std::milli>(t_split_done - t_seed_done).count();
        breakdown->split_ms +=
            std::chrono::duration<double, std::milli>(t_weighted_done - t_split_done).count();
        breakdown->gather_ms +=
            std::chrono::duration<double, std::milli>(t_gather_done - t_weighted_done).count();
        breakdown->ffn_router_ms +=
            std::chrono::duration<double, std::milli>(t_router_done - t_gather_done).count();
        breakdown->ffn_norm_ms +=
            std::chrono::duration<double, std::milli>(t_norm_done - t_gather_done).count();
        breakdown->router_select_ms +=
            std::chrono::duration<double, std::milli>(t_router_select_done - t_norm_done).count();
        breakdown->router_d2h_ms +=
            std::chrono::duration<double, std::milli>(t_router_d2h_done - t_router_select_done).count();
        breakdown->route_upload_ms +=
            std::chrono::duration<double, std::milli>(t_route_upload_done - t_router_d2h_done).count();
        breakdown->fill_pack_ms +=
            std::chrono::duration<double, std::milli>(t_fill_done - t_router_done).count();
    }
    return 0;
}

struct OutputHeadGateStats {
    bool pass = true;
    int slots = 0;
    int vocab = 0;
    int rows_per_gpu = 0;
    uint64_t output_weight_bytes = 0;
    uint64_t logits_bytes = 0;
    double total_ms = 0.0;
    double projection_ms = 0.0;
    double projection_kernel_worst_ms = 0.0;
    double host_reduce_ms = 0.0;
    uint32_t first_token = UINT32_MAX;
    float first_logit = 0.0f;
    uint64_t checksum = 0;
    int finite_bad = 0;
};

struct OutputHeadResidentGateStats {
    bool pass = true;
    int slots = 0;
    int vocab = 0;
    int rows_per_gpu = 0;
    int warmup = 0;
    int iters = 0;
    uint64_t output_weight_bytes = 0;
    uint64_t logits_bytes = 0;
    double load_ms = 0.0;
    double avg_total_ms = 0.0;
    double avg_hc_prep_ms = 0.0;
    double avg_broadcast_ms = 0.0;
    double avg_projection_wall_ms = 0.0;
    double avg_projection_kernel_worst_ms = 0.0;
    double avg_readback_reduce_ms = 0.0;
    double output_head_tok_s = 0.0;
    uint32_t first_token = UINT32_MAX;
    float first_logit = 0.0f;
    uint64_t checksum = 0;
    int finite_bad = 0;
};

struct SharedOutputHead {
    bool initialized = false;
    int slots = 0;
    int vocab = 0;
    int rows_per_gpu = 0;
    ContractRow output_rows[kGpus];
    float *d_hc = nullptr;
    float *d_hc_norm = nullptr;
    float *d_head_pre = nullptr;
    float *d_head_weights = nullptr;
    float *d_embd = nullptr;
    float *d_embd_norm = nullptr;
    float *d_head_fn = nullptr;
    float *d_head_base = nullptr;
    float *d_head_scale = nullptr;
    float *d_output_norm = nullptr;
    uint16_t *d_w[kGpus] = {};
    float *d_x[kGpus] = {};
    float *d_logits[kGpus] = {};
    uint32_t *d_best_token[kGpus] = {};
    float *d_best_logit[kGpus] = {};
    cudaEvent_t projection_start[kGpus] = {};
    cudaEvent_t projection_stop[kGpus] = {};
    cudaStream_t stream[kGpus] = {};
    cudaEvent_t prep_ready = {};
    cudaEvent_t broadcast_ready[kGpus] = {};
    cudaEvent_t top1_done[kGpus] = {};
    uint32_t *h_best_token[kGpus] = {};
    float *h_best_logit[kGpus] = {};
    uint64_t output_weight_bytes = 0;
    uint64_t logits_bytes = 0;
};

struct OutputHeadRunResult {
    bool pass = true;
    double total_ms = 0.0;
    double gather_ms = 0.0;
    double prep_ms = 0.0;
    double broadcast_ms = 0.0;
    double projection_ms = 0.0;
    double projection_kernel_worst_ms = 0.0;
    double top1_ms = 0.0;
    std::vector<uint32_t> tokens;
    std::vector<float> logits;
    uint64_t checksum = 0;
    int finite_bad = 0;
    bool async_output_gate = false;
    int device_sync_count = 0;
    int stream_sync_count = 0;
    int event_sync_count = 0;
};

struct SharedTokenEmbedding {
    bool initialized = false;
    int slots = 0;
    int vocab = 0;
    int rows_per_gpu = 0;
    std::vector<uint16_t> h_w_full;
    uint16_t *d_slot_rows[kGpus] = {};
    uint64_t weight_bytes = 0;
};

int open_shared_token_embedding(const Options &opt,
                                const std::vector<ContractRow> &rows,
                                SharedTokenEmbedding *out) {
    out->slots = opt.slots;
    std::vector<ContractRow> emb_rows;
    int cols = 0;
    int vocab = 0;
    if (!select_bf16_dense_rows(rows, "token_embd.weight", &emb_rows, &cols, &vocab)) {
        std::fprintf(stderr, "shared token embedding failed to select token_embd.weight shards\n");
        return 1;
    }
    if (cols != kHidden || vocab <= 0 || vocab % kGpus != 0) {
        std::fprintf(stderr, "shared token embedding invalid shape cols=%d vocab=%d\n",
                     cols, vocab);
        return 2;
    }
    out->vocab = vocab;
    out->rows_per_gpu = vocab / kGpus;
    const uint64_t shard_elems = (uint64_t)out->rows_per_gpu * (uint64_t)kHidden;
    const uint64_t shard_bytes = shard_elems * sizeof(uint16_t);
    const uint64_t full_elems = shard_elems * kGpus;

    out->h_w_full.assign((size_t)full_elems, 0);
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
        CHECK_CUDA(cudaMalloc(&out->d_slot_rows[(size_t)rank],
                              (size_t)opt.slots * kHidden * sizeof(uint16_t)));
    }

    std::vector<uint16_t> host((size_t)shard_elems);
    for (int shard = 0; shard < kGpus; ++shard) {
        const ContractRow &r = emb_rows[(size_t)shard];
        const int shard_index = r.shard_index >= 0 ? r.shard_index : shard;
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), host.data(),
                          (size_t)shard_bytes) != 0) {
            return 3;
        }
        std::memcpy(out->h_w_full.data() + (uint64_t)shard_index * shard_elems,
                    host.data(), (size_t)shard_bytes);
        out->weight_bytes += shard_bytes;
    }
    out->initialized = true;
    return 0;
}

void close_shared_token_embedding(const Options &opt, SharedTokenEmbedding *out) {
    if (!out || !out->initialized) return;
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
        if (out->d_slot_rows[(size_t)rank]) {
            CHECK_CUDA(cudaFree(out->d_slot_rows[(size_t)rank]));
        }
    }
    *out = SharedTokenEmbedding{};
}

int seed_rank_hc_from_input_tokens(const Options &opt,
                                   SharedTokenEmbedding *embedding,
                                   RankState ranks[kGpus],
                                   const std::vector<uint32_t> &tokens) {
    if (!embedding || !embedding->initialized ||
        (int)tokens.size() < opt.slots ||
        embedding->h_w_full.empty()) {
        return 1;
    }
    const uint64_t shard_elems =
        (uint64_t)opt.slots * 4ull * (uint64_t)(kHidden / kGpus);
    std::vector<uint16_t> slot_rows((size_t)opt.slots * kHidden);
    for (int slot = 0; slot < opt.slots; ++slot) {
        uint32_t token = tokens[(size_t)slot];
        if (token >= (uint32_t)embedding->vocab) token = 0;
        std::memcpy(slot_rows.data() + (size_t)slot * kHidden,
                    embedding->h_w_full.data() + (uint64_t)token * kHidden,
                    (size_t)kHidden * sizeof(uint16_t));
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.d_final_hc_shard || !embedding->d_slot_rows[(size_t)rank]) return 2;
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaMemcpyAsync(embedding->d_slot_rows[(size_t)rank],
                                   slot_rows.data(),
                                   slot_rows.size() * sizeof(uint16_t),
                                   cudaMemcpyHostToDevice, r.stream));
        seed_hc_shard_from_token_embedding_kernel<<<
            (unsigned int)((shard_elems + 255) / 256), 256, 0, r.stream>>>(
            r.d_final_hc_shard,
            embedding->d_slot_rows[(size_t)rank],
            (uint32_t)opt.slots,
            rank);
        CHECK_CUDA(cudaGetLastError());
        r.hc_initialized = true;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }
    return 0;
}

int run_output_head_gate(const Options &opt,
                         const std::vector<ContractRow> &rows,
                         OutputHeadGateStats *stats) {
    stats->slots = opt.slots;

    std::vector<ContractRow> output_rows;
    int output_cols = 0;
    int vocab = 0;
    if (!select_bf16_dense_rows(rows, "output.weight", &output_rows, &output_cols, &vocab)) {
        std::fprintf(stderr, "output-head gate failed to select output.weight shards\n");
        return 1;
    }
    if (output_cols != kHidden || vocab <= 0 || vocab % kGpus != 0) {
        std::fprintf(stderr, "output-head gate invalid output.weight shape cols=%d vocab=%d\n",
                     output_cols, vocab);
        return 2;
    }
    const int rows_per_gpu = vocab / kGpus;
    stats->vocab = vocab;
    stats->rows_per_gpu = rows_per_gpu;

    std::vector<float> hc_head_fn;
    std::vector<float> hc_head_base;
    std::vector<float> hc_head_scale;
    std::vector<float> output_norm;
    if (load_control_f32(opt, rows, "hc_head_fn", (size_t)4 * 4 * kHidden, &hc_head_fn) ||
        load_control_f32(opt, rows, "hc_head_base", 4, &hc_head_base) ||
        load_control_f32(opt, rows, "hc_head_scale", 1, &hc_head_scale) ||
        load_control_f32(opt, rows, "output_norm.weight", kHidden, &output_norm)) {
        return 3;
    }

    const uint64_t hc_elems = (uint64_t)opt.slots * 4ull * (uint64_t)kHidden;
    const uint64_t embd_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const uint64_t logits_elems = (uint64_t)opt.slots * (uint64_t)rows_per_gpu;
    const uint64_t output_shard_bytes = (uint64_t)rows_per_gpu * (uint64_t)kHidden *
                                        sizeof(uint16_t);
    const auto total_start = std::chrono::steady_clock::now();

    float *d_hc = nullptr;
    float *d_hc_norm = nullptr;
    float *d_head_pre = nullptr;
    float *d_head_weights = nullptr;
    float *d_embd = nullptr;
    float *d_embd_norm = nullptr;
    float *d_head_fn = nullptr;
    float *d_head_base = nullptr;
    float *d_head_scale = nullptr;
    float *d_output_norm = nullptr;
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaMalloc(&d_hc, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hc_norm, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_pre, (size_t)opt.slots * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_weights, (size_t)opt.slots * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_embd, (size_t)embd_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_embd_norm, (size_t)embd_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_fn, hc_head_fn.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_base, hc_head_base.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_scale, hc_head_scale.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_norm, output_norm.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_head_fn, hc_head_fn.data(), hc_head_fn.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_head_base, hc_head_base.data(), hc_head_base.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_head_scale, hc_head_scale.data(), hc_head_scale.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_output_norm, output_norm.data(), output_norm.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    synthetic_hc_kernel<<<(unsigned int)((hc_elems + 255) / 256), 256>>>(d_hc, opt.slots);
    rms_norm_plain_rows_stable_kernel<<<(unsigned int)opt.slots, 256>>>(
        d_hc_norm, d_hc, 4u * (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
    const dim3 head_grid(4u, (unsigned int)opt.slots, 1u);
    f32_dense_kernel<<<head_grid, 256>>>(d_head_pre, d_head_fn, d_hc_norm,
                                         4u, 4u * (uint32_t)kHidden,
                                         (uint32_t)opt.slots);
    output_hc_weights_rows_kernel<<<(unsigned int)(((uint64_t)opt.slots * 4ull + 255) / 256), 256>>>(
        d_head_weights, d_head_pre, d_head_scale, d_head_base, (uint32_t)opt.slots);
    hc_weighted_sum_rows_kernel<<<(unsigned int)((embd_elems + 255) / 256), 256>>>(
        d_embd, d_hc, d_head_weights, (uint32_t)opt.slots);
    rms_norm_weight_rows_stable_kernel<<<(unsigned int)opt.slots, 256>>>(
        d_embd_norm, d_embd, d_output_norm, (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_embd_norm((size_t)embd_elems);
    CHECK_CUDA(cudaMemcpy(h_embd_norm.data(), d_embd_norm,
                          h_embd_norm.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<std::vector<float>> host_logits((size_t)kGpus);
    std::vector<uint16_t> host_w;
    std::vector<uint32_t> best_token((size_t)opt.slots, UINT32_MAX);
    std::vector<float> best_logit((size_t)opt.slots, -std::numeric_limits<float>::max());

    const auto projection_start = std::chrono::steady_clock::now();
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = output_rows[(size_t)gpu];
        host_w.resize((size_t)rows_per_gpu * (size_t)kHidden);
        host_logits[(size_t)gpu].resize((size_t)logits_elems);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), host_w.data(),
                          (size_t)output_shard_bytes) != 0) {
            stats->pass = false;
            return 4;
        }
        stats->output_weight_bytes += output_shard_bytes;
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        uint16_t *d_w = nullptr;
        __half *d_w_half = nullptr;
        float *d_x = nullptr;
        __half *d_x_half = nullptr;
        float *d_logits = nullptr;
        cublasHandle_t blas = nullptr;
        cudaEvent_t kernel_start = nullptr;
        cudaEvent_t kernel_stop = nullptr;
        CHECK_CUDA(cudaMalloc(&d_w, (size_t)output_shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x, h_embd_norm.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_logits, (size_t)logits_elems * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_w, host_w.data(), (size_t)output_shard_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, h_embd_norm.data(), h_embd_norm.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (opt.dense_f16_cublas_compose) {
            const uint64_t w_elems = (uint64_t)rows_per_gpu * (uint64_t)kHidden;
            const uint64_t x_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
            CHECK_CUDA(cudaMalloc(&d_w_half, (size_t)w_elems * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&d_x_half, (size_t)x_elems * sizeof(__half)));
            bf16_to_half_kernel<<<(unsigned int)((w_elems + 255) / 256), 256>>>(
                d_w_half, d_w, w_elems);
            cast_f32_to_half_kernel<<<(unsigned int)((x_elems + 255) / 256), 256>>>(
                d_x_half, d_x, x_elems);
            CHECK_CUDA(cudaGetLastError());
            cublasStatus_t st = cublasCreate(&blas);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "output-head cublasCreate failed gpu=%d status=%d\n",
                             gpu, (int)st);
                return 5;
            }
            (void)cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH);
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CHECK_CUDA(cudaEventCreate(&kernel_start));
            CHECK_CUDA(cudaEventCreate(&kernel_stop));
            CHECK_CUDA(cudaEventRecord(kernel_start));
            st = cublasGemmEx(blas,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              rows_per_gpu,
                              opt.slots,
                              kHidden,
                              &alpha,
                              d_w_half,
                              CUDA_R_16F,
                              kHidden,
                              d_x_half,
                              CUDA_R_16F,
                              kHidden,
                              &beta,
                              d_logits,
                              CUDA_R_32F,
                              rows_per_gpu,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "output-head cublasGemmEx failed gpu=%d status=%d\n",
                             gpu, (int)st);
                return 6;
            }
            CHECK_CUDA(cudaEventRecord(kernel_stop));
        } else {
            const dim3 grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1u);
            CHECK_CUDA(cudaEventCreate(&kernel_start));
            CHECK_CUDA(cudaEventCreate(&kernel_stop));
            CHECK_CUDA(cudaEventRecord(kernel_start));
            bf16_dense_kernel<<<grid, 256>>>(d_logits, d_w, d_x,
                                             (uint32_t)rows_per_gpu,
                                             (uint32_t)kHidden,
                                             (uint32_t)kHidden,
                                             (uint32_t)opt.slots);
            CHECK_CUDA(cudaEventRecord(kernel_stop));
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        float kernel_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
        stats->projection_kernel_worst_ms =
            std::max(stats->projection_kernel_worst_ms, (double)kernel_ms);
        CHECK_CUDA(cudaMemcpy(host_logits[(size_t)gpu].data(), d_logits,
                              (size_t)logits_elems * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventDestroy(kernel_stop));
        CHECK_CUDA(cudaEventDestroy(kernel_start));
        if (blas) (void)cublasDestroy(blas);
        CHECK_CUDA(cudaFree(d_logits));
        if (d_x_half) CHECK_CUDA(cudaFree(d_x_half));
        CHECK_CUDA(cudaFree(d_x));
        if (d_w_half) CHECK_CUDA(cudaFree(d_w_half));
        CHECK_CUDA(cudaFree(d_w));
    }
    const auto projection_stop = std::chrono::steady_clock::now();
    stats->projection_ms =
        std::chrono::duration<double, std::milli>(projection_stop - projection_start).count();
    stats->logits_bytes = logits_elems * sizeof(float) * kGpus;

    const auto reduce_start = std::chrono::steady_clock::now();
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const int shard_index = output_rows[(size_t)gpu].shard_index >= 0
            ? output_rows[(size_t)gpu].shard_index
            : gpu;
        for (int slot = 0; slot < opt.slots; ++slot) {
            const float *row = host_logits[(size_t)gpu].data() +
                               (uint64_t)slot * (uint64_t)rows_per_gpu;
            for (int v = 0; v < rows_per_gpu; ++v) {
                const float logit = row[v];
                if (!std::isfinite(logit)) {
                    stats->finite_bad++;
                    stats->pass = false;
                    continue;
                }
                if (logit > best_logit[(size_t)slot]) {
                    best_logit[(size_t)slot] = logit;
                    best_token[(size_t)slot] =
                        (uint32_t)(shard_index * rows_per_gpu + v);
                }
            }
        }
    }
    const auto reduce_stop = std::chrono::steady_clock::now();
    stats->host_reduce_ms =
        std::chrono::duration<double, std::milli>(reduce_stop - reduce_start).count();

    for (int slot = 0; slot < opt.slots; ++slot) {
        if (best_token[(size_t)slot] >= (uint32_t)vocab ||
            !std::isfinite(best_logit[(size_t)slot])) {
            stats->pass = false;
        }
        uint32_t bits = 0;
        std::memcpy(&bits, &best_logit[(size_t)slot], sizeof(bits));
        stats->checksum ^= (uint64_t)best_token[(size_t)slot] * 1000003ull +
                           (uint64_t)bits + (uint64_t)(slot + 1) * 7907ull;
    }
    stats->first_token = best_token.empty() ? UINT32_MAX : best_token[0];
    stats->first_logit = best_logit.empty() ? 0.0f : best_logit[0];
    const auto total_stop = std::chrono::steady_clock::now();
    stats->total_ms =
        std::chrono::duration<double, std::milli>(total_stop - total_start).count();
    if (stats->checksum == 0 || stats->finite_bad != 0) stats->pass = false;

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaFree(d_output_norm));
    CHECK_CUDA(cudaFree(d_head_scale));
    CHECK_CUDA(cudaFree(d_head_base));
    CHECK_CUDA(cudaFree(d_head_fn));
    CHECK_CUDA(cudaFree(d_embd_norm));
    CHECK_CUDA(cudaFree(d_embd));
    CHECK_CUDA(cudaFree(d_head_weights));
    CHECK_CUDA(cudaFree(d_head_pre));
    CHECK_CUDA(cudaFree(d_hc_norm));
    CHECK_CUDA(cudaFree(d_hc));

    std::printf("tp_ep_output_head_gate\tslots\t%d\tvocab\t%d\trows_per_gpu\t%d\t"
                "projection_kernel\t%s\t"
                "output_weight_bytes\t%llu\tlogits_bytes\t%llu\t"
                "projection_ms\t%.6f\tprojection_kernel_worst_ms\t%.6f\t"
                "host_reduce_ms\t%.6f\ttotal_ms\t%.6f\t"
                "first_token\t%u\tfirst_logit\t%.9f\tfinite_bad\t%d\t"
                "checksum\t%llu\t%s\n",
                stats->slots, stats->vocab, stats->rows_per_gpu,
                opt.dense_f16_cublas_compose ? "bf16_to_fp16_cublas" : "bf16_scalar",
                (unsigned long long)stats->output_weight_bytes,
                (unsigned long long)stats->logits_bytes,
                stats->projection_ms, stats->projection_kernel_worst_ms,
                stats->host_reduce_ms, stats->total_ms,
                stats->first_token, stats->first_logit, stats->finite_bad,
                (unsigned long long)stats->checksum,
                stats->pass ? "PASS" : "FAIL");
    return stats->pass ? 0 : 5;
}

int run_output_head_resident_gate(const Options &opt,
                                  const std::vector<ContractRow> &rows,
                                  OutputHeadResidentGateStats *stats) {
    if (opt.dense_f16_cublas_compose) {
        std::fprintf(stderr, "resident output-head gate currently supports bf16_scalar only\n");
        return 2;
    }
    stats->slots = opt.slots;
    stats->warmup = opt.warmup;
    stats->iters = opt.iters;

    std::vector<ContractRow> output_rows;
    int output_cols = 0;
    int vocab = 0;
    if (!select_bf16_dense_rows(rows, "output.weight", &output_rows, &output_cols, &vocab)) {
        std::fprintf(stderr, "resident output-head gate failed to select output.weight shards\n");
        return 1;
    }
    if (output_cols != kHidden || vocab <= 0 || vocab % kGpus != 0) {
        std::fprintf(stderr, "resident output-head gate invalid output.weight shape cols=%d vocab=%d\n",
                     output_cols, vocab);
        return 2;
    }
    const int rows_per_gpu = vocab / kGpus;
    stats->vocab = vocab;
    stats->rows_per_gpu = rows_per_gpu;

    std::vector<float> hc_head_fn;
    std::vector<float> hc_head_base;
    std::vector<float> hc_head_scale;
    std::vector<float> output_norm;
    if (load_control_f32(opt, rows, "hc_head_fn", (size_t)4 * 4 * kHidden, &hc_head_fn) ||
        load_control_f32(opt, rows, "hc_head_base", 4, &hc_head_base) ||
        load_control_f32(opt, rows, "hc_head_scale", 1, &hc_head_scale) ||
        load_control_f32(opt, rows, "output_norm.weight", kHidden, &output_norm)) {
        return 3;
    }

    const uint64_t hc_elems = (uint64_t)opt.slots * 4ull * (uint64_t)kHidden;
    const uint64_t embd_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const uint64_t logits_elems = (uint64_t)opt.slots * (uint64_t)rows_per_gpu;
    const uint64_t output_shard_bytes = (uint64_t)rows_per_gpu * (uint64_t)kHidden *
                                        sizeof(uint16_t);

    float *d_hc = nullptr;
    float *d_hc_norm = nullptr;
    float *d_head_pre = nullptr;
    float *d_head_weights = nullptr;
    float *d_embd = nullptr;
    float *d_embd_norm = nullptr;
    float *d_head_fn = nullptr;
    float *d_head_base = nullptr;
    float *d_head_scale = nullptr;
    float *d_output_norm = nullptr;
    uint16_t *d_w[kGpus] = {};
    float *d_x[kGpus] = {};
    float *d_logits[kGpus] = {};
    uint32_t *d_best_token[kGpus] = {};
    float *d_best_logit[kGpus] = {};
    cudaEvent_t projection_start[kGpus] = {};
    cudaEvent_t projection_stop[kGpus] = {};

    const auto load_start = std::chrono::steady_clock::now();
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaMalloc(&d_hc, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hc_norm, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_pre, (size_t)opt.slots * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_weights, (size_t)opt.slots * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_embd, (size_t)embd_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_embd_norm, (size_t)embd_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_fn, hc_head_fn.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_base, hc_head_base.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_scale, hc_head_scale.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_norm, output_norm.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_head_fn, hc_head_fn.data(), hc_head_fn.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_head_base, hc_head_base.data(), hc_head_base.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_head_scale, hc_head_scale.data(), hc_head_scale.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_output_norm, output_norm.data(), output_norm.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::vector<uint16_t> host_w((size_t)rows_per_gpu * (size_t)kHidden);
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = output_rows[(size_t)gpu];
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), host_w.data(),
                          (size_t)output_shard_bytes) != 0) {
            return 4;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaMalloc(&d_w[gpu], (size_t)output_shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x[gpu], (size_t)embd_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_logits[gpu], (size_t)logits_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_best_token[gpu], (size_t)opt.slots * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&d_best_logit[gpu], (size_t)opt.slots * sizeof(float)));
        CHECK_CUDA(cudaEventCreate(&projection_start[gpu]));
        CHECK_CUDA(cudaEventCreate(&projection_stop[gpu]));
        CHECK_CUDA(cudaMemcpy(d_w[gpu], host_w.data(), (size_t)output_shard_bytes,
                              cudaMemcpyHostToDevice));
        stats->output_weight_bytes += output_shard_bytes;
    }
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    const auto load_stop = std::chrono::steady_clock::now();
    stats->load_ms =
        std::chrono::duration<double, std::milli>(load_stop - load_start).count();
    stats->logits_bytes = logits_elems * sizeof(float) * kGpus;

    const int total_iters = opt.warmup + opt.iters;
    std::vector<std::vector<uint32_t>> host_best_token((size_t)kGpus);
    std::vector<std::vector<float>> host_best_logit((size_t)kGpus);
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        host_best_token[(size_t)gpu].resize((size_t)opt.slots);
        host_best_logit[(size_t)gpu].resize((size_t)opt.slots);
    }
    std::vector<uint32_t> best_token((size_t)opt.slots, UINT32_MAX);
    std::vector<float> best_logit((size_t)opt.slots, -std::numeric_limits<float>::max());

    for (int iter = 0; iter < total_iters; ++iter) {
        const bool measure = iter >= opt.warmup;
        const auto iter_start = std::chrono::steady_clock::now();

        const auto prep_start = std::chrono::steady_clock::now();
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        synthetic_hc_kernel<<<(unsigned int)((hc_elems + 255) / 256), 256>>>(d_hc, opt.slots);
        rms_norm_plain_rows_stable_kernel<<<(unsigned int)opt.slots, 256>>>(
            d_hc_norm, d_hc, 4u * (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
        const dim3 head_grid(4u, (unsigned int)opt.slots, 1u);
        f32_dense_kernel<<<head_grid, 256>>>(d_head_pre, d_head_fn, d_hc_norm,
                                             4u, 4u * (uint32_t)kHidden,
                                             (uint32_t)opt.slots);
        output_hc_weights_rows_kernel<<<(unsigned int)(((uint64_t)opt.slots * 4ull + 255) / 256), 256>>>(
            d_head_weights, d_head_pre, d_head_scale, d_head_base, (uint32_t)opt.slots);
        hc_weighted_sum_rows_kernel<<<(unsigned int)((embd_elems + 255) / 256), 256>>>(
            d_embd, d_hc, d_head_weights, (uint32_t)opt.slots);
        rms_norm_weight_rows_stable_kernel<<<(unsigned int)opt.slots, 256>>>(
            d_embd_norm, d_embd, d_output_norm, (uint32_t)kHidden,
            (uint32_t)opt.slots, 1.0e-6f);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        const auto prep_stop = std::chrono::steady_clock::now();

        const auto broadcast_start = std::chrono::steady_clock::now();
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            if (gpu == 0) {
                CHECK_CUDA(cudaMemcpyAsync(d_x[gpu], d_embd_norm,
                                           (size_t)embd_elems * sizeof(float),
                                           cudaMemcpyDeviceToDevice));
            } else {
                CHECK_CUDA(cudaMemcpyPeerAsync(d_x[gpu],
                                               opt.devices[gpu],
                                               d_embd_norm,
                                               opt.devices[0],
                                               (size_t)embd_elems * sizeof(float)));
            }
        }
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        const auto broadcast_stop = std::chrono::steady_clock::now();

        const auto projection_start_wall = std::chrono::steady_clock::now();
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            const dim3 grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1u);
            CHECK_CUDA(cudaEventRecord(projection_start[gpu]));
            bf16_dense_kernel<<<grid, 256>>>(d_logits[gpu], d_w[gpu], d_x[gpu],
                                             (uint32_t)rows_per_gpu,
                                             (uint32_t)kHidden,
                                             (uint32_t)kHidden,
                                             (uint32_t)opt.slots);
            CHECK_CUDA(cudaEventRecord(projection_stop[gpu]));
            CHECK_CUDA(cudaGetLastError());
        }
        double iter_kernel_worst_ms = 0.0;
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            CHECK_CUDA(cudaDeviceSynchronize());
            float kernel_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&kernel_ms,
                                            projection_start[gpu],
                                            projection_stop[gpu]));
            iter_kernel_worst_ms = std::max(iter_kernel_worst_ms, (double)kernel_ms);
        }
        const auto projection_stop_wall = std::chrono::steady_clock::now();

        const auto reduce_start = std::chrono::steady_clock::now();
        std::fill(best_token.begin(), best_token.end(), UINT32_MAX);
        std::fill(best_logit.begin(), best_logit.end(),
                  -std::numeric_limits<float>::max());
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            const int shard_index = output_rows[(size_t)gpu].shard_index >= 0
                ? output_rows[(size_t)gpu].shard_index
                : gpu;
            shard_top1_kernel<<<(unsigned int)opt.slots, 256>>>(
                d_best_token[gpu],
                d_best_logit[gpu],
                d_logits[gpu],
                (uint32_t)rows_per_gpu,
                (uint32_t)(shard_index * rows_per_gpu),
                (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(host_best_token[(size_t)gpu].data(), d_best_token[gpu],
                                  (size_t)opt.slots * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(host_best_logit[(size_t)gpu].data(), d_best_logit[gpu],
                                  (size_t)opt.slots * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            for (int slot = 0; slot < opt.slots; ++slot) {
                const float logit = host_best_logit[(size_t)gpu][(size_t)slot];
                if (!std::isfinite(logit)) {
                    if (measure) stats->finite_bad++;
                    stats->pass = false;
                    continue;
                }
                if (logit > best_logit[(size_t)slot]) {
                    best_logit[(size_t)slot] = logit;
                    best_token[(size_t)slot] =
                        host_best_token[(size_t)gpu][(size_t)slot];
                }
            }
        }
        const auto reduce_stop = std::chrono::steady_clock::now();
        const auto iter_stop = std::chrono::steady_clock::now();

        if (measure) {
            stats->avg_hc_prep_ms +=
                std::chrono::duration<double, std::milli>(prep_stop - prep_start).count();
            stats->avg_broadcast_ms +=
                std::chrono::duration<double, std::milli>(broadcast_stop - broadcast_start).count();
            stats->avg_projection_wall_ms +=
                std::chrono::duration<double, std::milli>(projection_stop_wall - projection_start_wall).count();
            stats->avg_projection_kernel_worst_ms += iter_kernel_worst_ms;
            stats->avg_readback_reduce_ms +=
                std::chrono::duration<double, std::milli>(reduce_stop - reduce_start).count();
            stats->avg_total_ms +=
                std::chrono::duration<double, std::milli>(iter_stop - iter_start).count();
            for (int slot = 0; slot < opt.slots; ++slot) {
                if (best_token[(size_t)slot] >= (uint32_t)vocab ||
                    !std::isfinite(best_logit[(size_t)slot])) {
                    stats->pass = false;
                }
                uint32_t bits = 0;
                std::memcpy(&bits, &best_logit[(size_t)slot], sizeof(bits));
                stats->checksum ^=
                    (uint64_t)best_token[(size_t)slot] * 1000003ull +
                    (uint64_t)bits +
                    (uint64_t)(slot + 1) * 7907ull +
                    (uint64_t)(iter + 1) * 104729ull;
            }
            stats->first_token = best_token.empty() ? UINT32_MAX : best_token[0];
            stats->first_logit = best_logit.empty() ? 0.0f : best_logit[0];
        }
    }

    if (opt.iters > 0) {
        stats->avg_hc_prep_ms /= (double)opt.iters;
        stats->avg_broadcast_ms /= (double)opt.iters;
        stats->avg_projection_wall_ms /= (double)opt.iters;
        stats->avg_projection_kernel_worst_ms /= (double)opt.iters;
        stats->avg_readback_reduce_ms /= (double)opt.iters;
        stats->avg_total_ms /= (double)opt.iters;
        stats->output_head_tok_s = stats->avg_total_ms > 0.0
            ? (double)opt.slots * 1000.0 / stats->avg_total_ms
            : 0.0;
    }
    if (stats->checksum == 0 || stats->finite_bad != 0) stats->pass = false;

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (projection_stop[gpu]) CHECK_CUDA(cudaEventDestroy(projection_stop[gpu]));
        if (projection_start[gpu]) CHECK_CUDA(cudaEventDestroy(projection_start[gpu]));
        if (d_best_logit[gpu]) CHECK_CUDA(cudaFree(d_best_logit[gpu]));
        if (d_best_token[gpu]) CHECK_CUDA(cudaFree(d_best_token[gpu]));
        if (d_logits[gpu]) CHECK_CUDA(cudaFree(d_logits[gpu]));
        if (d_x[gpu]) CHECK_CUDA(cudaFree(d_x[gpu]));
        if (d_w[gpu]) CHECK_CUDA(cudaFree(d_w[gpu]));
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaFree(d_output_norm));
    CHECK_CUDA(cudaFree(d_head_scale));
    CHECK_CUDA(cudaFree(d_head_base));
    CHECK_CUDA(cudaFree(d_head_fn));
    CHECK_CUDA(cudaFree(d_embd_norm));
    CHECK_CUDA(cudaFree(d_embd));
    CHECK_CUDA(cudaFree(d_head_weights));
    CHECK_CUDA(cudaFree(d_head_pre));
    CHECK_CUDA(cudaFree(d_hc_norm));
    CHECK_CUDA(cudaFree(d_hc));

    std::printf("tp_ep_output_head_resident_gate\tslots\t%d\tvocab\t%d\t"
                "rows_per_gpu\t%d\twarmup\t%d\titers\t%d\t"
                "projection_kernel\tbf16_scalar\t"
                "output_weight_bytes\t%llu\tlogits_bytes\t%llu\t"
                "load_ms\t%.6f\tavg_total_ms\t%.6f\t"
                "avg_hc_prep_ms\t%.6f\tavg_broadcast_ms\t%.6f\t"
                "avg_projection_wall_ms\t%.6f\t"
                "avg_projection_kernel_worst_ms\t%.6f\t"
                "avg_device_top1_readback_ms\t%.6f\t"
                "output_head_tok_s\t%.6f\t"
                "first_token\t%u\tfirst_logit\t%.9f\tfinite_bad\t%d\t"
                "checksum\t%llu\t%s\n",
                stats->slots, stats->vocab, stats->rows_per_gpu,
                stats->warmup, stats->iters,
                (unsigned long long)stats->output_weight_bytes,
                (unsigned long long)stats->logits_bytes,
                stats->load_ms, stats->avg_total_ms,
                stats->avg_hc_prep_ms, stats->avg_broadcast_ms,
                stats->avg_projection_wall_ms,
                stats->avg_projection_kernel_worst_ms,
                stats->avg_readback_reduce_ms,
                stats->output_head_tok_s,
                stats->first_token, stats->first_logit, stats->finite_bad,
                (unsigned long long)stats->checksum,
                stats->pass ? "PASS" : "FAIL");
    return stats->pass ? 0 : 5;
}

int open_shared_output_head(const Options &opt,
                            const std::vector<ContractRow> &rows,
                            SharedOutputHead *out) {
    out->slots = opt.slots;
    std::vector<ContractRow> output_rows;
    int output_cols = 0;
    int vocab = 0;
    if (!select_bf16_dense_rows(rows, "output.weight", &output_rows,
                                &output_cols, &vocab)) {
        std::fprintf(stderr, "shared output-head failed to select output.weight shards\n");
        return 1;
    }
    if (output_cols != kHidden || vocab <= 0 || vocab % kGpus != 0) {
        std::fprintf(stderr, "shared output-head invalid output.weight shape cols=%d vocab=%d\n",
                     output_cols, vocab);
        return 2;
    }
    out->vocab = vocab;
    out->rows_per_gpu = vocab / kGpus;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        out->output_rows[gpu] = output_rows[(size_t)gpu];
    }

    std::vector<float> hc_head_fn;
    std::vector<float> hc_head_base;
    std::vector<float> hc_head_scale;
    std::vector<float> output_norm;
    if (load_control_f32(opt, rows, "hc_head_fn", (size_t)4 * 4 * kHidden,
                         &hc_head_fn) ||
        load_control_f32(opt, rows, "hc_head_base", 4, &hc_head_base) ||
        load_control_f32(opt, rows, "hc_head_scale", 1, &hc_head_scale) ||
        load_control_f32(opt, rows, "output_norm.weight", kHidden, &output_norm)) {
        return 3;
    }

    const uint64_t hc_elems = (uint64_t)opt.slots * 4ull * (uint64_t)kHidden;
    const uint64_t embd_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const uint64_t logits_elems = (uint64_t)opt.slots * (uint64_t)out->rows_per_gpu;
    const uint64_t output_shard_bytes =
        (uint64_t)out->rows_per_gpu * (uint64_t)kHidden * sizeof(uint16_t);

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaMalloc(&out->d_hc, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_hc_norm, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_head_pre, (size_t)opt.slots * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_head_weights, (size_t)opt.slots * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_embd, (size_t)embd_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_embd_norm, (size_t)embd_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_head_fn, hc_head_fn.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_head_base, hc_head_base.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_head_scale, hc_head_scale.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_output_norm, output_norm.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(out->d_head_fn, hc_head_fn.data(),
                          hc_head_fn.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(out->d_head_base, hc_head_base.data(),
                          hc_head_base.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(out->d_head_scale, hc_head_scale.data(),
                          hc_head_scale.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(out->d_output_norm, output_norm.data(),
                          output_norm.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaStreamCreateWithFlags(&out->stream[0], cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreateWithFlags(&out->prep_ready, cudaEventDisableTiming));

    std::vector<uint16_t> host_w((size_t)out->rows_per_gpu * (size_t)kHidden);
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = out->output_rows[gpu];
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), host_w.data(),
                          (size_t)output_shard_bytes) != 0) {
            return 4;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaMalloc(&out->d_w[gpu], (size_t)output_shard_bytes));
        CHECK_CUDA(cudaMalloc(&out->d_x[gpu], (size_t)embd_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_logits[gpu],
                              (size_t)logits_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_best_token[gpu],
                              (size_t)opt.slots * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&out->d_best_logit[gpu],
                              (size_t)opt.slots * sizeof(float)));
        CHECK_CUDA(cudaEventCreate(&out->projection_start[gpu]));
        CHECK_CUDA(cudaEventCreate(&out->projection_stop[gpu]));
        if (gpu != 0) {
            CHECK_CUDA(cudaStreamCreateWithFlags(&out->stream[gpu],
                                                 cudaStreamNonBlocking));
        }
        CHECK_CUDA(cudaEventCreateWithFlags(&out->broadcast_ready[gpu],
                                            cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&out->top1_done[gpu],
                                            cudaEventDisableTiming));
        CHECK_CUDA(cudaMallocHost(&out->h_best_token[gpu],
                                  (size_t)opt.slots * sizeof(uint32_t)));
        CHECK_CUDA(cudaMallocHost(&out->h_best_logit[gpu],
                                  (size_t)opt.slots * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(out->d_w[gpu], host_w.data(),
                              (size_t)output_shard_bytes, cudaMemcpyHostToDevice));
        out->output_weight_bytes += output_shard_bytes;
    }
    out->logits_bytes = logits_elems * sizeof(float) * kGpus;
    out->initialized = true;
    return 0;
}

void close_shared_output_head(const Options &opt, SharedOutputHead *out) {
    if (!out || !out->initialized) return;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (out->h_best_logit[gpu]) CHECK_CUDA(cudaFreeHost(out->h_best_logit[gpu]));
        if (out->h_best_token[gpu]) CHECK_CUDA(cudaFreeHost(out->h_best_token[gpu]));
        if (out->top1_done[gpu]) CHECK_CUDA(cudaEventDestroy(out->top1_done[gpu]));
        if (out->broadcast_ready[gpu]) CHECK_CUDA(cudaEventDestroy(out->broadcast_ready[gpu]));
        if (out->projection_stop[gpu]) CHECK_CUDA(cudaEventDestroy(out->projection_stop[gpu]));
        if (out->projection_start[gpu]) CHECK_CUDA(cudaEventDestroy(out->projection_start[gpu]));
        if (out->stream[gpu]) CHECK_CUDA(cudaStreamDestroy(out->stream[gpu]));
        if (out->d_best_logit[gpu]) CHECK_CUDA(cudaFree(out->d_best_logit[gpu]));
        if (out->d_best_token[gpu]) CHECK_CUDA(cudaFree(out->d_best_token[gpu]));
        if (out->d_logits[gpu]) CHECK_CUDA(cudaFree(out->d_logits[gpu]));
        if (out->d_x[gpu]) CHECK_CUDA(cudaFree(out->d_x[gpu]));
        if (out->d_w[gpu]) CHECK_CUDA(cudaFree(out->d_w[gpu]));
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    if (out->prep_ready) CHECK_CUDA(cudaEventDestroy(out->prep_ready));
    if (out->d_output_norm) CHECK_CUDA(cudaFree(out->d_output_norm));
    if (out->d_head_scale) CHECK_CUDA(cudaFree(out->d_head_scale));
    if (out->d_head_base) CHECK_CUDA(cudaFree(out->d_head_base));
    if (out->d_head_fn) CHECK_CUDA(cudaFree(out->d_head_fn));
    if (out->d_embd_norm) CHECK_CUDA(cudaFree(out->d_embd_norm));
    if (out->d_embd) CHECK_CUDA(cudaFree(out->d_embd));
    if (out->d_head_weights) CHECK_CUDA(cudaFree(out->d_head_weights));
    if (out->d_head_pre) CHECK_CUDA(cudaFree(out->d_head_pre));
    if (out->d_hc_norm) CHECK_CUDA(cudaFree(out->d_hc_norm));
    if (out->d_hc) CHECK_CUDA(cudaFree(out->d_hc));
    *out = SharedOutputHead{};
}

int run_shared_output_head_from_rank_hc(const Options &opt,
                                        SharedOutputHead *head,
                                        RankState ranks[kGpus],
                                        OutputHeadRunResult *result) {
    if (!head || !head->initialized || head->slots != opt.slots) return 1;
    const auto total_start = std::chrono::steady_clock::now();
    const uint64_t hc_shard_elems =
        (uint64_t)opt.slots * 4ull * (uint64_t)(kHidden / kGpus);
    const uint64_t hc_elems = (uint64_t)opt.slots * 4ull * (uint64_t)kHidden;
    const uint64_t embd_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const uint64_t logits_elems =
        (uint64_t)opt.slots * (uint64_t)head->rows_per_gpu;
    result->async_output_gate = opt.async_output_gate;

    if (opt.async_output_gate) {
        const auto gather_start = std::chrono::steady_clock::now();
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            if (!ranks[rank].d_final_hc_shard) {
                std::fprintf(stderr, "diagnostic output-head missing final HC shard rank=%d\n",
                             rank);
                return 2;
            }
            gather_hc_shard_to_full_kernel<<<
                (unsigned int)((hc_shard_elems + 255) / 256), 256, 0,
                head->stream[0]>>>(
                head->d_hc, ranks[rank].d_final_hc_shard, rank,
                (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        const auto gather_stop = std::chrono::steady_clock::now();

        const auto prep_start = std::chrono::steady_clock::now();
        rms_norm_plain_rows_stable_kernel<<<(unsigned int)opt.slots, 256, 0,
                                             head->stream[0]>>>(
            head->d_hc_norm, head->d_hc, 4u * (uint32_t)kHidden,
            (uint32_t)opt.slots, 1.0e-6f);
        const dim3 head_grid(4u, (unsigned int)opt.slots, 1u);
        f32_dense_kernel<<<head_grid, 256, 0, head->stream[0]>>>(
            head->d_head_pre, head->d_head_fn, head->d_hc_norm, 4u,
            4u * (uint32_t)kHidden, (uint32_t)opt.slots);
        output_hc_weights_rows_kernel<<<
            (unsigned int)(((uint64_t)opt.slots * 4ull + 255) / 256), 256, 0,
            head->stream[0]>>>(
            head->d_head_weights, head->d_head_pre, head->d_head_scale,
            head->d_head_base, (uint32_t)opt.slots);
        hc_weighted_sum_rows_kernel<<<
            (unsigned int)((embd_elems + 255) / 256), 256, 0,
            head->stream[0]>>>(
            head->d_embd, head->d_hc, head->d_head_weights,
            (uint32_t)opt.slots);
        rms_norm_weight_rows_stable_kernel<<<(unsigned int)opt.slots, 256, 0,
                                             head->stream[0]>>>(
            head->d_embd_norm, head->d_embd, head->d_output_norm,
            (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(head->prep_ready, head->stream[0]));
        const auto prep_stop = std::chrono::steady_clock::now();

        const auto broadcast_start = std::chrono::steady_clock::now();
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            CHECK_CUDA(cudaStreamWaitEvent(head->stream[gpu], head->prep_ready, 0));
            if (gpu == 0) {
                CHECK_CUDA(cudaMemcpyAsync(head->d_x[gpu], head->d_embd_norm,
                                           (size_t)embd_elems * sizeof(float),
                                           cudaMemcpyDeviceToDevice,
                                           head->stream[gpu]));
            } else {
                CHECK_CUDA(cudaMemcpyPeerAsync(head->d_x[gpu], opt.devices[gpu],
                                               head->d_embd_norm, opt.devices[0],
                                               (size_t)embd_elems * sizeof(float),
                                               head->stream[gpu]));
            }
            CHECK_CUDA(cudaEventRecord(head->broadcast_ready[gpu],
                                       head->stream[gpu]));
        }
        const auto broadcast_stop = std::chrono::steady_clock::now();

        const auto projection_start_wall = std::chrono::steady_clock::now();
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            CHECK_CUDA(cudaStreamWaitEvent(head->stream[gpu],
                                           head->broadcast_ready[gpu], 0));
            const dim3 grid((unsigned int)head->rows_per_gpu,
                            (unsigned int)opt.slots, 1u);
            CHECK_CUDA(cudaEventRecord(head->projection_start[gpu],
                                       head->stream[gpu]));
            bf16_dense_kernel<<<grid, 256, 0, head->stream[gpu]>>>(
                head->d_logits[gpu], head->d_w[gpu], head->d_x[gpu],
                (uint32_t)head->rows_per_gpu, (uint32_t)kHidden,
                (uint32_t)kHidden, (uint32_t)opt.slots);
            CHECK_CUDA(cudaEventRecord(head->projection_stop[gpu],
                                       head->stream[gpu]));
            CHECK_CUDA(cudaGetLastError());
        }

        const auto top1_start = std::chrono::steady_clock::now();
        result->tokens.assign((size_t)opt.slots, UINT32_MAX);
        result->logits.assign((size_t)opt.slots,
                              -std::numeric_limits<float>::max());
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            const int shard_index = head->output_rows[gpu].shard_index >= 0
                ? head->output_rows[gpu].shard_index
                : gpu;
            shard_top1_kernel<<<(unsigned int)opt.slots, 256, 0,
                                head->stream[gpu]>>>(
                head->d_best_token[gpu], head->d_best_logit[gpu],
                head->d_logits[gpu], (uint32_t)head->rows_per_gpu,
                (uint32_t)(shard_index * head->rows_per_gpu),
                (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaMemcpyAsync(head->h_best_token[gpu],
                                       head->d_best_token[gpu],
                                       (size_t)opt.slots * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost,
                                       head->stream[gpu]));
            CHECK_CUDA(cudaMemcpyAsync(head->h_best_logit[gpu],
                                       head->d_best_logit[gpu],
                                       (size_t)opt.slots * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       head->stream[gpu]));
            CHECK_CUDA(cudaEventRecord(head->top1_done[gpu],
                                       head->stream[gpu]));
        }

        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            CHECK_CUDA(cudaEventSynchronize(head->top1_done[gpu]));
            result->event_sync_count++;
            float kernel_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&kernel_ms,
                                            head->projection_start[gpu],
                                            head->projection_stop[gpu]));
            result->projection_kernel_worst_ms =
                std::max(result->projection_kernel_worst_ms, (double)kernel_ms);
            for (int slot = 0; slot < opt.slots; ++slot) {
                const float logit = head->h_best_logit[gpu][(size_t)slot];
                if (!std::isfinite(logit)) {
                    result->finite_bad++;
                    result->pass = false;
                    continue;
                }
                if (logit > result->logits[(size_t)slot]) {
                    result->logits[(size_t)slot] = logit;
                    result->tokens[(size_t)slot] =
                        head->h_best_token[gpu][(size_t)slot];
                }
            }
        }
        const auto projection_stop_wall = std::chrono::steady_clock::now();
        const auto top1_stop = std::chrono::steady_clock::now();
        const auto total_stop = std::chrono::steady_clock::now();

        for (int slot = 0; slot < opt.slots; ++slot) {
            if (result->tokens[(size_t)slot] >= (uint32_t)head->vocab ||
                !std::isfinite(result->logits[(size_t)slot])) {
                result->pass = false;
            }
            uint32_t bits = 0;
            std::memcpy(&bits, &result->logits[(size_t)slot], sizeof(bits));
            result->checksum ^= (uint64_t)result->tokens[(size_t)slot] * 1000003ull +
                                (uint64_t)bits + (uint64_t)(slot + 1) * 7907ull;
        }
        if (result->checksum == 0 || result->finite_bad != 0) result->pass = false;

        result->gather_ms =
            std::chrono::duration<double, std::milli>(gather_stop - gather_start).count();
        result->prep_ms =
            std::chrono::duration<double, std::milli>(prep_stop - prep_start).count();
        result->broadcast_ms =
            std::chrono::duration<double, std::milli>(broadcast_stop - broadcast_start).count();
        result->projection_ms =
            std::chrono::duration<double, std::milli>(projection_stop_wall - projection_start_wall).count();
        result->top1_ms =
            std::chrono::duration<double, std::milli>(top1_stop - top1_start).count();
        result->total_ms =
            std::chrono::duration<double, std::milli>(total_stop - total_start).count();
        (void)hc_elems;
        (void)logits_elems;
        return result->pass ? 0 : 5;
    }

    const auto gather_start = std::chrono::steady_clock::now();
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        if (!ranks[rank].d_final_hc_shard) {
            std::fprintf(stderr, "diagnostic output-head missing final HC shard rank=%d\n",
                         rank);
            return 2;
        }
        gather_hc_shard_to_full_kernel<<<(unsigned int)((hc_shard_elems + 255) / 256), 256>>>(
            head->d_hc, ranks[rank].d_final_hc_shard, rank, (uint32_t)opt.slots);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    result->device_sync_count++;
    const auto gather_stop = std::chrono::steady_clock::now();

    const auto prep_start = std::chrono::steady_clock::now();
    rms_norm_plain_rows_stable_kernel<<<(unsigned int)opt.slots, 256>>>(
        head->d_hc_norm, head->d_hc, 4u * (uint32_t)kHidden,
        (uint32_t)opt.slots, 1.0e-6f);
    const dim3 head_grid(4u, (unsigned int)opt.slots, 1u);
    f32_dense_kernel<<<head_grid, 256>>>(head->d_head_pre, head->d_head_fn,
                                         head->d_hc_norm, 4u,
                                         4u * (uint32_t)kHidden,
                                         (uint32_t)opt.slots);
    output_hc_weights_rows_kernel<<<(unsigned int)(((uint64_t)opt.slots * 4ull + 255) / 256), 256>>>(
        head->d_head_weights, head->d_head_pre, head->d_head_scale,
        head->d_head_base, (uint32_t)opt.slots);
    hc_weighted_sum_rows_kernel<<<(unsigned int)((embd_elems + 255) / 256), 256>>>(
        head->d_embd, head->d_hc, head->d_head_weights, (uint32_t)opt.slots);
    rms_norm_weight_rows_stable_kernel<<<(unsigned int)opt.slots, 256>>>(
        head->d_embd_norm, head->d_embd, head->d_output_norm,
        (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    result->device_sync_count++;
    const auto prep_stop = std::chrono::steady_clock::now();

    const auto broadcast_start = std::chrono::steady_clock::now();
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (gpu == 0) {
            CHECK_CUDA(cudaMemcpyAsync(head->d_x[gpu], head->d_embd_norm,
                                       (size_t)embd_elems * sizeof(float),
                                       cudaMemcpyDeviceToDevice));
        } else {
            CHECK_CUDA(cudaMemcpyPeerAsync(head->d_x[gpu], opt.devices[gpu],
                                           head->d_embd_norm, opt.devices[0],
                                           (size_t)embd_elems * sizeof(float)));
        }
    }
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaDeviceSynchronize());
        result->device_sync_count++;
    }
    const auto broadcast_stop = std::chrono::steady_clock::now();

    const auto projection_start_wall = std::chrono::steady_clock::now();
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        const dim3 grid((unsigned int)head->rows_per_gpu, (unsigned int)opt.slots, 1u);
        CHECK_CUDA(cudaEventRecord(head->projection_start[gpu]));
        bf16_dense_kernel<<<grid, 256>>>(head->d_logits[gpu], head->d_w[gpu],
                                         head->d_x[gpu],
                                         (uint32_t)head->rows_per_gpu,
                                         (uint32_t)kHidden,
                                         (uint32_t)kHidden,
                                         (uint32_t)opt.slots);
        CHECK_CUDA(cudaEventRecord(head->projection_stop[gpu]));
        CHECK_CUDA(cudaGetLastError());
    }
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaDeviceSynchronize());
        result->device_sync_count++;
        float kernel_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&kernel_ms,
                                        head->projection_start[gpu],
                                        head->projection_stop[gpu]));
        result->projection_kernel_worst_ms =
            std::max(result->projection_kernel_worst_ms, (double)kernel_ms);
    }
    const auto projection_stop_wall = std::chrono::steady_clock::now();

    const auto top1_start = std::chrono::steady_clock::now();
    std::vector<std::vector<uint32_t>> host_tokens((size_t)kGpus);
    std::vector<std::vector<float>> host_logits((size_t)kGpus);
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        host_tokens[(size_t)gpu].resize((size_t)opt.slots);
        host_logits[(size_t)gpu].resize((size_t)opt.slots);
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        const int shard_index = head->output_rows[gpu].shard_index >= 0
            ? head->output_rows[gpu].shard_index
            : gpu;
        shard_top1_kernel<<<(unsigned int)opt.slots, 256>>>(
            head->d_best_token[gpu], head->d_best_logit[gpu],
            head->d_logits[gpu], (uint32_t)head->rows_per_gpu,
            (uint32_t)(shard_index * head->rows_per_gpu), (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }

    result->tokens.assign((size_t)opt.slots, UINT32_MAX);
    result->logits.assign((size_t)opt.slots, -std::numeric_limits<float>::max());
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaDeviceSynchronize());
        result->device_sync_count++;
        CHECK_CUDA(cudaMemcpy(host_tokens[(size_t)gpu].data(), head->d_best_token[gpu],
                              (size_t)opt.slots * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_logits[(size_t)gpu].data(), head->d_best_logit[gpu],
                              (size_t)opt.slots * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int slot = 0; slot < opt.slots; ++slot) {
            const float logit = host_logits[(size_t)gpu][(size_t)slot];
            if (!std::isfinite(logit)) {
                result->finite_bad++;
                result->pass = false;
                continue;
            }
            if (logit > result->logits[(size_t)slot]) {
                result->logits[(size_t)slot] = logit;
                result->tokens[(size_t)slot] = host_tokens[(size_t)gpu][(size_t)slot];
            }
        }
    }
    const auto top1_stop = std::chrono::steady_clock::now();
    const auto total_stop = std::chrono::steady_clock::now();

    for (int slot = 0; slot < opt.slots; ++slot) {
        if (result->tokens[(size_t)slot] >= (uint32_t)head->vocab ||
            !std::isfinite(result->logits[(size_t)slot])) {
            result->pass = false;
        }
        uint32_t bits = 0;
        std::memcpy(&bits, &result->logits[(size_t)slot], sizeof(bits));
        result->checksum ^= (uint64_t)result->tokens[(size_t)slot] * 1000003ull +
                            (uint64_t)bits + (uint64_t)(slot + 1) * 7907ull;
    }
    if (result->checksum == 0 || result->finite_bad != 0) result->pass = false;

    result->gather_ms =
        std::chrono::duration<double, std::milli>(gather_stop - gather_start).count();
    result->prep_ms =
        std::chrono::duration<double, std::milli>(prep_stop - prep_start).count();
    result->broadcast_ms =
        std::chrono::duration<double, std::milli>(broadcast_stop - broadcast_start).count();
    result->projection_ms =
        std::chrono::duration<double, std::milli>(projection_stop_wall - projection_start_wall).count();
    result->top1_ms =
        std::chrono::duration<double, std::milli>(top1_stop - top1_start).count();
    result->total_ms =
        std::chrono::duration<double, std::milli>(total_stop - total_start).count();
    (void)hc_elems;
    (void)logits_elems;
    return result->pass ? 0 : 5;
}

void free_device_dense_outputs(DeviceDenseOutputs &out, const Options &opt) {
    for (int gpu = 0; gpu < (int)out.d_out.size(); ++gpu) {
        if (!out.d_out[(size_t)gpu]) continue;
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaFree(out.d_out[(size_t)gpu]));
    }
    out = DeviceDenseOutputs{};
}

void free_resident_f8_dense(ResidentF8Dense &op, const Options &opt) {
    for (int gpu = 0; gpu < (int)op.d_w.size(); ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (op.d_w[(size_t)gpu]) CHECK_CUDA(cudaFree(op.d_w[(size_t)gpu]));
        if (op.d_x[(size_t)gpu]) CHECK_CUDA(cudaFree(op.d_x[(size_t)gpu]));
        if (gpu < (int)op.d_w_half.size() && op.d_w_half[(size_t)gpu]) {
            const bool owns = gpu >= (int)op.owns_w_half.size() || op.owns_w_half[(size_t)gpu];
            if (owns) CHECK_CUDA(cudaFree(op.d_w_half[(size_t)gpu]));
        }
        if (gpu < (int)op.d_x_half.size() && op.d_x_half[(size_t)gpu]) {
            CHECK_CUDA(cudaFree(op.d_x_half[(size_t)gpu]));
        }
        if (op.d_out[(size_t)gpu]) CHECK_CUDA(cudaFree(op.d_out[(size_t)gpu]));
        if (gpu < (int)op.cublas.size() && op.cublas[(size_t)gpu]) {
            (void)cublasDestroy(op.cublas[(size_t)gpu]);
        }
    }
    op = ResidentF8Dense{};
}

uint64_t align_up_u64(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

void free_dense_f16_cache(DenseF16Cache &cache, const Options &opt) {
    for (int gpu = 0; gpu < (int)cache.arena.size(); ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (cache.arena[(size_t)gpu]) CHECK_CUDA(cudaFree(cache.arena[(size_t)gpu]));
        if (gpu < (int)cache.temp.size() && cache.temp[(size_t)gpu]) {
            CHECK_CUDA(cudaFree(cache.temp[(size_t)gpu]));
        }
    }
    cache = DenseF16Cache{};
}

const DenseF16CacheEntry *find_dense_f16_cache_entry(const DenseF16Cache &cache,
                                                     const char *tensor,
                                                     int gpu) {
    if (!cache.enabled) return nullptr;
    for (const DenseF16CacheEntry &e : cache.entries) {
        if (e.gpu == gpu && e.tensor_id == tensor) return &e;
    }
    return nullptr;
}

int prepare_dense_f16_cache(const Options &opt,
                            const std::vector<ContractRow> &rows,
                            DenseF16Cache *cache) {
    if (!opt.dense_f16_cache_compose) return 0;
    cache->enabled = true;
    cache->arena.assign((size_t)kGpus, nullptr);
    cache->temp.assign((size_t)kGpus, nullptr);
    uint64_t gpu_offsets[kGpus] = {};
    uint64_t gpu_temp[kGpus] = {};

    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" ||
            (r.source_dtype != "f8_e4m3_b128" && r.source_dtype != "bf16")) {
            continue;
        }
        int cols = 0;
        int total_rows = 0;
        if (!parse_shape2(r.source_shape, &cols, &total_rows)) continue;
        uint64_t rows_per_gpu = 0;
        if (r.source_dtype == "f8_e4m3_b128") {
            if (cols % 128 != 0) continue;
            const uint64_t rb = f8_row_bytes(cols);
            if (rb == 0 || r.bytes_estimate % rb != 0) continue;
            rows_per_gpu = r.bytes_estimate / rb;
        } else {
            const uint64_t rb = (uint64_t)cols * sizeof(uint16_t);
            if (rb == 0 || r.bytes_estimate % rb != 0) continue;
            rows_per_gpu = r.bytes_estimate / rb;
        }
        DenseF16CacheEntry e;
        e.tensor_id = r.tensor_id;
        e.gpu = r.owning_gpu;
        e.cols = cols;
        e.rows_per_gpu = (int)rows_per_gpu;
        e.offset = gpu_offsets[r.owning_gpu];
        e.source_bytes = r.bytes_estimate;
        e.cache_bytes = rows_per_gpu * (uint64_t)cols * sizeof(__half);
        cache->entries.push_back(e);
        cache->rows++;
        cache->source_bytes += e.source_bytes;
        cache->cache_bytes += e.cache_bytes;
        const uint64_t aligned = align_up_u64(e.cache_bytes, 256);
        gpu_offsets[r.owning_gpu] += aligned;
        cache->cache_aligned_bytes += aligned;
        gpu_temp[r.owning_gpu] = std::max(gpu_temp[r.owning_gpu], e.source_bytes);
        cache->max_temp_bytes = std::max(cache->max_temp_bytes, e.source_bytes);
    }

    if (cache->entries.empty()) return 1;
    uint64_t planned_bytes[kGpus] = {};
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        cache->gpu_cache_aligned_bytes[gpu] = gpu_offsets[gpu];
        cache->gpu_temp_bytes[gpu] = gpu_temp[gpu];
        planned_bytes[gpu] = gpu_offsets[gpu] + gpu_temp[gpu];
    }
    if (check_planned_vram_allocation(opt, "dense_f16_cache_prealloc", planned_bytes) != 0) {
        std::fprintf(stderr,
                     "dense_f16_cache_vram_admission_failed min_free_mib=%llu\n",
                     (unsigned long long)opt.vram_min_free_mib);
        return 3;
    }
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (gpu_offsets[gpu]) CHECK_CUDA(cudaMalloc(&cache->arena[(size_t)gpu],
                                                    (size_t)gpu_offsets[gpu]));
        if (gpu_temp[gpu]) CHECK_CUDA(cudaMalloc(&cache->temp[(size_t)gpu],
                                                 (size_t)gpu_temp[gpu]));
    }

    std::vector<uint8_t> host;
    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" ||
            (r.source_dtype != "f8_e4m3_b128" && r.source_dtype != "bf16")) {
            continue;
        }
        const DenseF16CacheEntry *e =
            find_dense_f16_cache_entry(*cache, r.tensor_id.c_str(), r.owning_gpu);
        if (!e || e->source_bytes != r.bytes_estimate) continue;
        host.resize((size_t)r.bytes_estimate);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), host.data(), host.size()) != 0) {
            free_dense_f16_cache(*cache, opt);
            return 2;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[r.owning_gpu]));
        CHECK_CUDA(cudaMemcpy(cache->temp[(size_t)r.owning_gpu], host.data(), host.size(),
                              cudaMemcpyHostToDevice));
        __half *dst =
            reinterpret_cast<__half *>(cache->arena[(size_t)r.owning_gpu] + e->offset);
        const uint64_t elems = e->cache_bytes / sizeof(__half);
        const unsigned int grid = (unsigned int)((elems + 255) / 256);
        if (r.source_dtype == "f8_e4m3_b128") {
            f8_b128_to_half_kernel<<<grid, 256>>>(
                dst, cache->temp[(size_t)r.owning_gpu], e->rows_per_gpu,
                e->cols, (uint32_t)f8_row_bytes(e->cols));
        } else {
            bf16_to_half_kernel<<<grid, 256>>>(
                dst, reinterpret_cast<const uint16_t *>(cache->temp[(size_t)r.owning_gpu]),
                elems);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (cache->temp[(size_t)gpu]) {
            CHECK_CUDA(cudaFree(cache->temp[(size_t)gpu]));
            cache->temp[(size_t)gpu] = nullptr;
        }
    }
    return 0;
}

int prepare_resident_f8_dense(const Options &opt,
                              const std::vector<ContractRow> &rows,
                              const char *tensor,
                              int seed,
                              const DenseF16Cache *cache,
                              ResidentF8Dense *op,
                              int expected_rows_per_gpu = kHidden / kGpus,
                              bool keep_packed_f8 = false,
                              bool keep_float_input = false) {
    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    bool source_is_f8 = select_dense_rows(rows, tensor, &selected, &cols, &total_rows);
    bool source_is_bf16 = false;
    if (!source_is_f8) {
        source_is_bf16 = select_bf16_dense_rows(rows, tensor, &selected, &cols, &total_rows);
    }
    if (!source_is_f8 && !source_is_bf16) {
        std::fprintf(stderr, "resident dense tensor validation failed for %s\n", tensor);
        return 1;
    }
    if (source_is_bf16 && keep_packed_f8) {
        std::fprintf(stderr, "resident dense tensor %s requested packed f8 retention for bf16 source\n",
                     tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    if (rows_per_gpu != expected_rows_per_gpu) {
        std::fprintf(stderr, "resident dense tensor %s rows_per_gpu=%d expected=%d\n",
                     tensor, rows_per_gpu, expected_rows_per_gpu);
        return 2;
    }
    const uint64_t row_bytes =
        source_is_f8 ? f8_row_bytes(cols) : (uint64_t)cols * sizeof(uint16_t);
    const uint64_t shard_bytes = row_bytes * (uint64_t)rows_per_gpu;
    op->d_w.assign((size_t)kGpus, nullptr);
    op->d_x.assign((size_t)kGpus, nullptr);
    op->d_w_half.assign((size_t)kGpus, nullptr);
    op->owns_w_half.assign((size_t)kGpus, true);
    op->d_x_half.assign((size_t)kGpus, nullptr);
    op->d_out.assign((size_t)kGpus, nullptr);
    op->cublas.assign((size_t)kGpus, nullptr);
    op->rows_per_gpu = rows_per_gpu;
    op->cols = cols;
    op->slots = opt.slots;
    op->row_bytes = row_bytes;

    std::vector<float> h_x((size_t)opt.slots * cols);
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int c = 0; c < cols; ++c) {
            const int m = (slot * (17 + seed) + c * (13 + seed * 3)) % 269;
            h_x[(size_t)slot * cols + c] = ((float)m - 134.0f) * 0.0002f;
        }
    }

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = selected[(size_t)gpu];
        const DenseF16CacheEntry *cache_entry =
            opt.dense_f16_cache_compose && opt.dense_f16_cublas_compose && cache
                ? find_dense_f16_cache_entry(*cache, tensor, gpu)
                : nullptr;
        if (source_is_bf16 && !cache_entry) {
            std::fprintf(stderr,
                         "resident bf16 dense tensor %s requires dense f16 cache on gpu %d\n",
                         tensor, gpu);
            free_resident_f8_dense(*op, opt);
            return 3;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (!cache_entry || keep_packed_f8) {
            std::vector<uint8_t> h_w((size_t)shard_bytes);
            const std::string path = path_join(opt.pack_dir, r.source_pack_file);
            if (read_exact_at(path, physical_row_offset(r), h_w.data(), h_w.size()) != 0) {
                free_resident_f8_dense(*op, opt);
                return 3;
            }
            CHECK_CUDA(cudaMalloc(&op->d_w[(size_t)gpu], (size_t)shard_bytes));
            CHECK_CUDA(cudaMemcpy(op->d_w[(size_t)gpu], h_w.data(), (size_t)shard_bytes,
                                  cudaMemcpyHostToDevice));
        }
        op->loaded_bytes += shard_bytes;
        CHECK_CUDA(cudaMalloc(&op->d_x[(size_t)gpu], h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&op->d_out[(size_t)gpu],
                              (size_t)opt.slots * rows_per_gpu * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(op->d_x[(size_t)gpu], h_x.data(),
                              h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
        if (opt.dense_f16_cublas_compose) {
            (void)cudaGetLastError();
            if (cache_entry) {
                if (cache_entry->cols != cols || cache_entry->rows_per_gpu != rows_per_gpu) {
                    free_resident_f8_dense(*op, opt);
                    return 4;
                }
                op->d_w_half[(size_t)gpu] =
                    reinterpret_cast<__half *>(cache->arena[(size_t)gpu] + cache_entry->offset);
                op->owns_w_half[(size_t)gpu] = false;
            } else {
                CHECK_CUDA(cudaMalloc(&op->d_w_half[(size_t)gpu],
                                      (size_t)rows_per_gpu * cols * sizeof(__half)));
                op->owns_w_half[(size_t)gpu] = true;
                const uint64_t w_elems = (uint64_t)rows_per_gpu * cols;
                if (!source_is_f8) {
                    free_resident_f8_dense(*op, opt);
                    return 5;
                }
                f8_b128_to_half_kernel<<<(unsigned int)((w_elems + 255) / 256), 256>>>(
                    op->d_w_half[(size_t)gpu], op->d_w[(size_t)gpu],
                    rows_per_gpu, cols, (uint32_t)row_bytes);
                CHECK_CUDA(cudaGetLastError());
            }
            CHECK_CUDA(cudaMalloc(&op->d_x_half[(size_t)gpu],
                                  h_x.size() * sizeof(__half)));
            const uint64_t x_elems = (uint64_t)opt.slots * cols;
            cast_f32_to_half_kernel<<<(unsigned int)((x_elems + 255) / 256), 256>>>(
                op->d_x_half[(size_t)gpu], op->d_x[(size_t)gpu], x_elems);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            if (!keep_float_input) {
                CHECK_CUDA(cudaFree(op->d_x[(size_t)gpu]));
                op->d_x[(size_t)gpu] = nullptr;
            }
            cublasStatus_t st = cublasCreate(&op->cublas[(size_t)gpu]);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "cublasCreate failed on gpu %d: %d\n", gpu, (int)st);
                free_resident_f8_dense(*op, opt);
                return 4;
            }
            (void)cublasSetMathMode(op->cublas[(size_t)gpu], CUBLAS_TENSOR_OP_MATH);
        }
    }
    return 0;
}

void free_shared_dense_ops(SharedDenseOps *ops, const Options &opt) {
    if (!ops) return;
    for (int layer = 0; layer < 43; ++layer) {
        free_resident_f8_dense(ops->layers[layer].attn_q_a, opt);
        free_resident_f8_dense(ops->layers[layer].attn_q_b, opt);
        free_resident_f8_dense(ops->layers[layer].attn_kv_latent, opt);
        free_resident_f8_dense(ops->layers[layer].attn_output_a, opt);
        free_resident_f8_dense(ops->layers[layer].attn_compress_kv, opt);
        free_resident_f8_dense(ops->layers[layer].attn_compress_gate, opt);
        free_resident_f8_dense(ops->layers[layer].indexer_attn_q_b, opt);
        free_resident_f8_dense(ops->layers[layer].indexer_proj, opt);
        free_resident_f8_dense(ops->layers[layer].indexer_compress_kv, opt);
        free_resident_f8_dense(ops->layers[layer].indexer_compress_gate, opt);
        free_resident_f8_dense(ops->layers[layer].attn, opt);
        free_resident_f8_dense(ops->layers[layer].shared, opt);
        free_resident_f8_dense(ops->layers[layer].shared_gate, opt);
        free_resident_f8_dense(ops->layers[layer].shared_up, opt);
        ops->layers[layer] = LayerDenseOps{};
    }
    *ops = SharedDenseOps{};
}

int open_shared_dense_ops(const Options &opt,
                          const DenseF16Cache *cache,
                          SharedDenseOps *ops) {
    if (!opt.dense_f16_cublas_compose || !opt.dense_f16_cache_compose || !cache) {
        return 1;
    }
    for (int layer = 0; layer < 43; ++layer) {
        std::vector<ContractRow> rows;
        LayerStats stats;
        if (parse_contract(opt.contract_path, layer, &rows, &stats) != 0 ||
            stats.bad_rows != 0) {
            free_shared_dense_ops(ops, opt);
            return 2;
        }
        Options layer_opt = opt;
        layer_opt.layer = layer;
        LayerDenseOps &d = ops->layers[layer];
        const std::string attn_q_a_tensor = layer_tensor_name(layer, "attn_q_a.weight");
        const std::string attn_q_b_tensor = layer_tensor_name(layer, "attn_q_b.weight");
        const std::string attn_kv_tensor = layer_tensor_name(layer, "attn_kv_latent.weight");
        const std::string attn_output_a_tensor = layer_tensor_name(layer, "attn_output_a.weight");
        const std::string attn_compress_kv_tensor = layer_tensor_name(layer, "attn_compress_kv.weight");
        const std::string attn_compress_gate_tensor = layer_tensor_name(layer, "attn_compress_gate.weight");
        const std::string indexer_attn_q_b_tensor = layer_tensor_name(layer, "indexer.attn_q_b.weight");
        const std::string indexer_proj_tensor = layer_tensor_name(layer, "indexer.proj.weight");
        const std::string indexer_compress_kv_tensor = layer_tensor_name(layer, "indexer.compress_kv.weight");
        const std::string indexer_compress_gate_tensor = layer_tensor_name(layer, "indexer.compress_gate.weight");
        const std::string attn_tensor = layer_tensor_name(layer, "attn_output_b.weight");
        const std::string shared_tensor = layer_tensor_name(layer, "ffn_down_shexp.weight");
        const std::string shared_gate_tensor = layer_tensor_name(layer, "ffn_gate_shexp.weight");
        const std::string shared_up_tensor = layer_tensor_name(layer, "ffn_up_shexp.weight");
        if (opt.true_ds4_attention_residency_gate) {
            if (prepare_resident_f8_dense(layer_opt, rows, attn_q_a_tensor.c_str(), 11,
                                          cache, &d.attn_q_a, 1024 / kGpus) != 0 ||
                prepare_resident_f8_dense(layer_opt, rows, attn_q_b_tensor.c_str(), 12,
                                          cache, &d.attn_q_b, 32768 / kGpus) != 0 ||
                prepare_resident_f8_dense(layer_opt, rows, attn_kv_tensor.c_str(), 13,
                                          cache, &d.attn_kv_latent, kHeadDim / kGpus) != 0 ||
                prepare_resident_f8_dense(layer_opt, rows, attn_output_a_tensor.c_str(), 14,
                                          cache, &d.attn_output_a, 8192 / kGpus) != 0) {
                free_shared_dense_ops(ops, opt);
                return 5;
            }
            ops->loaded_bytes += d.attn_q_a.loaded_bytes + d.attn_q_b.loaded_bytes +
                                 d.attn_kv_latent.loaded_bytes +
                                 d.attn_output_a.loaded_bytes;
        }
        if (opt.true_ds4_compressed_kv_gate) {
            const int ratio = ds4_layer_ratio(layer);
            if (ratio != 0) {
                const int comp_width = ratio == 4 ? 2 * kHeadDim : kHeadDim;
                if (prepare_resident_f8_dense(layer_opt, rows, attn_compress_kv_tensor.c_str(),
                                              15, cache, &d.attn_compress_kv,
                                              comp_width / kGpus) != 0 ||
                    prepare_resident_f8_dense(layer_opt, rows, attn_compress_gate_tensor.c_str(),
                                              16, cache, &d.attn_compress_gate,
                                              comp_width / kGpus) != 0) {
                    free_shared_dense_ops(ops, opt);
                    return 6;
                }
                ops->loaded_bytes += d.attn_compress_kv.loaded_bytes +
                                     d.attn_compress_gate.loaded_bytes;
            }
            if (opt.true_ds4_indexer_attention_gate && ratio == 4) {
                if (prepare_resident_f8_dense(layer_opt, rows, indexer_attn_q_b_tensor.c_str(),
                                              17, cache, &d.indexer_attn_q_b,
                                              (kIndexerHead * kIndexerHeadDim) / kGpus) != 0 ||
                    prepare_resident_f8_dense(layer_opt, rows, indexer_proj_tensor.c_str(),
                                              18, cache, &d.indexer_proj,
                                              kIndexerHead / kGpus) != 0 ||
                    prepare_resident_f8_dense(layer_opt, rows, indexer_compress_kv_tensor.c_str(),
                                              19, cache, &d.indexer_compress_kv,
                                              (2 * kIndexerHeadDim) / kGpus) != 0 ||
                    prepare_resident_f8_dense(layer_opt, rows, indexer_compress_gate_tensor.c_str(),
                                              20, cache, &d.indexer_compress_gate,
                                              (2 * kIndexerHeadDim) / kGpus) != 0) {
                    free_shared_dense_ops(ops, opt);
                    return 7;
                }
                ops->loaded_bytes += d.indexer_attn_q_b.loaded_bytes +
                                     d.indexer_proj.loaded_bytes +
                                     d.indexer_compress_kv.loaded_bytes +
                                     d.indexer_compress_gate.loaded_bytes;
            }
        }
        if (prepare_resident_f8_dense(layer_opt, rows, attn_tensor.c_str(), 1, cache,
                                      &d.attn) != 0 ||
            prepare_resident_f8_dense(layer_opt, rows, shared_tensor.c_str(), 2, cache,
                                      &d.shared, kHidden / kGpus,
                                      opt.true_shared_ffn_gate,
                                      opt.true_shared_ffn_gate) != 0) {
            free_shared_dense_ops(ops, opt);
            return 3;
        }
        if (opt.true_shared_ffn_gate) {
            if (prepare_resident_f8_dense(layer_opt, rows, shared_gate_tensor.c_str(), 3,
                                          cache, &d.shared_gate, kMid / kGpus) != 0 ||
                prepare_resident_f8_dense(layer_opt, rows, shared_up_tensor.c_str(), 4,
                                          cache, &d.shared_up, kMid / kGpus) != 0) {
                free_shared_dense_ops(ops, opt);
                return 4;
            }
            ops->loaded_bytes += d.shared_gate.loaded_bytes + d.shared_up.loaded_bytes;
        }
        d.initialized = true;
        ops->loaded_bytes += d.attn.loaded_bytes + d.shared.loaded_bytes;
    }
    ops->initialized = true;
    return 0;
}

int launch_resident_f8_dense(const Options &opt,
                             const ResidentF8Dense &op,
                             RankState ranks[kGpus]) {
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        (void)cudaGetLastError();
        cudaStream_t stream = ranks[gpu].dense_stream ? ranks[gpu].dense_stream
                                                       : ranks[gpu].stream;
        if (opt.dense_f16_cublas_compose) {
            if (!op.cublas[(size_t)gpu] ||
                !op.d_w_half[(size_t)gpu] ||
                !op.d_x_half[(size_t)gpu]) {
                return 1;
            }
            cublasStatus_t st = cublasSetStream(op.cublas[(size_t)gpu], stream);
            if (st != CUBLAS_STATUS_SUCCESS) return 2;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            st = cublasGemmEx(op.cublas[(size_t)gpu],
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              op.rows_per_gpu,
                              op.slots,
                              op.cols,
                              &alpha,
                              op.d_w_half[(size_t)gpu],
                              CUDA_R_16F,
                              op.cols,
                              op.d_x_half[(size_t)gpu],
                              CUDA_R_16F,
                              op.cols,
                              &beta,
                              op.d_out[(size_t)gpu],
                              CUDA_R_32F,
                              op.rows_per_gpu,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "cublasGemmEx failed gpu=%d status=%d\n", gpu, (int)st);
                return 3;
            }
        } else if (opt.dense_hmma_compose) {
            const dim3 grid((unsigned int)((op.rows_per_gpu + 63) / 64),
                            (unsigned int)((op.slots + 15) / 16),
                            1);
            f8_b128_dense_hmma_m16_kernel<<<grid, 128, 0, stream>>>(
                op.d_out[(size_t)gpu], op.d_w[(size_t)gpu], op.d_x[(size_t)gpu],
                op.rows_per_gpu, op.cols, (uint32_t)op.row_bytes, op.slots);
        } else {
            const dim3 grid((unsigned int)op.rows_per_gpu, (unsigned int)op.slots, 1);
            f8_b128_dense_kernel<<<grid, 256, 0, stream>>>(
                op.d_out[(size_t)gpu], op.d_w[(size_t)gpu], op.d_x[(size_t)gpu],
                op.rows_per_gpu, op.cols, (uint32_t)op.row_bytes, op.slots);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    return 0;
}

int launch_resident_f8_dense_f32_input(const Options &opt,
                                       const ResidentF8Dense &op,
                                       RankState ranks[kGpus]) {
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (!op.d_w[(size_t)gpu] || !op.d_x[(size_t)gpu]) return 1;
        cudaStream_t stream = ranks[gpu].dense_stream ? ranks[gpu].dense_stream
                                                       : ranks[gpu].stream;
        const dim3 grid((unsigned int)op.rows_per_gpu, (unsigned int)op.slots, 1);
        f8_b128_dense_kernel<<<grid, 256, 0, stream>>>(
            op.d_out[(size_t)gpu], op.d_w[(size_t)gpu], op.d_x[(size_t)gpu],
            op.rows_per_gpu, op.cols, (uint32_t)op.row_bytes, op.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    return 0;
}

int enqueue_dense_wait_after_rank_stream(RankState ranks[kGpus]) {
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        RankState &r = ranks[gpu];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.dense_stream || r.dense_stream == r.stream) continue;
        if (!r.dense_wait) return 1;
        CHECK_CUDA(cudaEventRecord(r.dense_wait, r.stream));
        CHECK_CUDA(cudaStreamWaitEvent(r.dense_stream, r.dense_wait, 0));
    }
    return 0;
}

int enqueue_rank_streams_wait_after_dense_streams(RankState ranks[kGpus]) {
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        RankState &r = ranks[gpu];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.dense_stream || r.dense_stream == r.stream) continue;
        if (!r.dense_done) return 1;
        CHECK_CUDA(cudaEventRecord(r.dense_done, r.dense_stream));
        CHECK_CUDA(cudaStreamWaitEvent(r.stream, r.dense_done, 0));
    }
    return 0;
}

int enqueue_cross_gpu_stream_barrier(RankState ranks[kGpus],
                                     bool include_copy_streams) {
    for (int src = 0; src < kGpus; ++src) {
        RankState &r = ranks[src];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.stream_done || !r.dense_done) return 1;
        CHECK_CUDA(cudaEventRecord(r.stream_done, r.stream));
        CHECK_CUDA(cudaEventRecord(r.dense_done,
                                   r.dense_stream ? r.dense_stream : r.stream));
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        RankState &r = ranks[dst];
        CHECK_CUDA(cudaSetDevice(r.device));
        for (int src = 0; src < kGpus; ++src) {
            CHECK_CUDA(cudaStreamWaitEvent(r.stream, ranks[src].stream_done, 0));
            CHECK_CUDA(cudaStreamWaitEvent(r.stream, ranks[src].dense_done, 0));
            if (r.dense_stream) {
                CHECK_CUDA(cudaStreamWaitEvent(r.dense_stream,
                                               ranks[src].stream_done, 0));
                CHECK_CUDA(cudaStreamWaitEvent(r.dense_stream,
                                               ranks[src].dense_done, 0));
            }
            if (include_copy_streams) {
                for (int q = 0; q < kGpus; ++q) {
                    cudaStream_t copy_stream = r.copy_streams[q]
                        ? r.copy_streams[q]
                        : r.copy_stream ? r.copy_stream : r.stream;
                    CHECK_CUDA(cudaStreamWaitEvent(copy_stream,
                                                   ranks[src].stream_done, 0));
                    CHECK_CUDA(cudaStreamWaitEvent(copy_stream,
                                                   ranks[src].dense_done, 0));
                }
            }
        }
    }
    return 0;
}

int enqueue_control_wait_after_rank_streams(const Options &opt,
                                            RankState ranks[kGpus],
                                            cudaStream_t control_stream) {
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.stream_done) return 1;
        CHECK_CUDA(cudaEventRecord(r.stream_done, r.stream));
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaStreamWaitEvent(control_stream, ranks[rank].stream_done, 0));
    }
    return 0;
}

int enqueue_control_wait_after_dense_streams(const Options &opt,
                                             RankState ranks[kGpus],
                                             cudaStream_t control_stream) {
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.dense_done) return 1;
        CHECK_CUDA(cudaEventRecord(r.dense_done,
                                   r.dense_stream ? r.dense_stream : r.stream));
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaStreamWaitEvent(control_stream, ranks[rank].dense_done, 0));
    }
    return 0;
}

int enqueue_rank_streams_wait_after_control(const Options &opt,
                                            RankState ranks[kGpus],
                                            cudaStream_t control_stream) {
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    if (!ranks[0].stream_done) return 1;
    CHECK_CUDA(cudaEventRecord(ranks[0].stream_done, control_stream));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaStreamWaitEvent(r.stream, ranks[0].stream_done, 0));
    }
    return 0;
}

int fill_shared_ffn_inputs_from_normed(const Options &opt,
                                       const SharedHcControls *hc,
                                       const ResidentF8Dense &gate,
                                       const ResidentF8Dense &up,
                                       RankState ranks[kGpus]) {
    if (!hc || !hc->d_ffn_normed) return 1;
    if (gate.cols != kHidden || up.cols != kHidden ||
        gate.rows_per_gpu != kMid / kGpus ||
        up.rows_per_gpu != kMid / kGpus) {
        return 2;
    }
    const uint64_t full_elems = (uint64_t)opt.slots * kHidden;
    const uint64_t full_bytes = full_elems * sizeof(float);
    const uint64_t x_elems = (uint64_t)opt.slots * kHidden;
    const int block = 256;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_current_full) return 3;
        if (rank == 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_current_full, hc->d_ffn_normed,
                                       (size_t)full_bytes,
                                       cudaMemcpyDeviceToDevice, r.stream));
        } else {
            CHECK_CUDA(cudaMemcpyPeerAsync(r.d_current_full, r.device,
                                           hc->d_ffn_normed, opt.devices[0],
                                           (size_t)full_bytes, r.stream));
        }
        if (gate.d_x_half[(size_t)rank]) {
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((x_elems + block - 1) / block), block, 0,
                r.stream>>>(gate.d_x_half[(size_t)rank], r.d_current_full,
                             (uint32_t)gate.cols, (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        if (up.d_x_half[(size_t)rank]) {
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((x_elems + block - 1) / block), block, 0,
                r.stream>>>(up.d_x_half[(size_t)rank], r.d_current_full,
                             (uint32_t)up.cols, (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    return 0;
}

int materialize_shared_swiglu_down_input(const Options &opt,
                                         const ResidentF8Dense &gate,
                                         const ResidentF8Dense &up,
                                         const ResidentF8Dense &down,
                                         RankState ranks[kGpus]) {
    if (gate.rows_per_gpu != kMid / kGpus ||
        up.rows_per_gpu != kMid / kGpus ||
        down.cols != kMid) {
        return 1;
    }
    const uint32_t rows = (uint32_t)gate.rows_per_gpu;
    const int block = 256;
    const uint64_t shard_elems = (uint64_t)opt.slots * rows;
    for (int src = 0; src < kGpus; ++src) {
        CHECK_CUDA(cudaSetDevice(ranks[src].device));
        if (!down.d_x[(size_t)src] ||
            !gate.d_out[(size_t)src] ||
            !up.d_out[(size_t)src]) {
            return 2;
        }
        shared_swiglu_shard_to_float_kernel<<<
            (unsigned int)((shard_elems + block - 1) / block), block, 0,
            ranks[src].stream>>>(down.d_x[(size_t)src],
                                 gate.d_out[(size_t)src],
                                 up.d_out[(size_t)src],
                                 (uint32_t)src, rows, (uint32_t)opt.slots,
                                 kRoutedSwigluClamp);
        CHECK_CUDA(cudaGetLastError());
    }
    for (int src = 0; src < kGpus; ++src) {
        CHECK_CUDA(cudaSetDevice(ranks[src].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[src].stream));
    }
    const size_t width = (size_t)rows * sizeof(float);
    for (int dst = 0; dst < kGpus; ++dst) {
        CHECK_CUDA(cudaSetDevice(ranks[dst].device));
        cudaStream_t stream = ranks[dst].copy_stream ? ranks[dst].copy_stream
                                                     : ranks[dst].stream;
        for (int src = 0; src < kGpus; ++src) {
            if (src == dst) continue;
            for (int slot = 0; slot < opt.slots; ++slot) {
                float *dst_ptr = down.d_x[(size_t)dst] +
                                 (size_t)slot * kMid + (size_t)src * rows;
                const float *src_ptr = down.d_x[(size_t)src] +
                                       (size_t)slot * kMid + (size_t)src * rows;
                CHECK_CUDA(cudaMemcpyPeerAsync(dst_ptr, ranks[dst].device,
                                               src_ptr, ranks[src].device,
                                               width, stream));
            }
        }
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        CHECK_CUDA(cudaSetDevice(ranks[dst].device));
        if (ranks[dst].copy_stream) CHECK_CUDA(cudaStreamSynchronize(ranks[dst].copy_stream));
        CHECK_CUDA(cudaStreamSynchronize(ranks[dst].stream));
    }
    return 0;
}

int run_f8_dense_to_device(const Options &opt,
                           const std::vector<ContractRow> &rows,
                           const char *tensor,
                           int seed,
                           DeviceDenseOutputs *out) {
    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    if (!select_dense_rows(rows, tensor, &selected, &cols, &total_rows)) {
        std::fprintf(stderr, "device dense tensor validation failed for %s\n", tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    if (rows_per_gpu != kHidden / kGpus) {
        std::fprintf(stderr, "device dense tensor %s rows_per_gpu=%d expected=%d\n",
                     tensor, rows_per_gpu, kHidden / kGpus);
        return 2;
    }
    const uint64_t row_bytes = f8_row_bytes(cols);
    const uint64_t shard_bytes = row_bytes * (uint64_t)rows_per_gpu;
    out->d_out.assign((size_t)kGpus, nullptr);
    out->rows_per_gpu = rows_per_gpu;
    out->cols = cols;
    out->slots = opt.slots;

    std::vector<float> h_x((size_t)opt.slots * cols);
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int c = 0; c < cols; ++c) {
            const int m = (slot * (17 + seed) + c * (13 + seed * 3)) % 269;
            h_x[(size_t)slot * cols + c] = ((float)m - 134.0f) * 0.0002f;
        }
    }

    double worst_ms = 0.0;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = selected[(size_t)gpu];
        std::vector<uint8_t> h_w((size_t)shard_bytes);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), h_w.data(), h_w.size()) != 0) {
            free_device_dense_outputs(*out, opt);
            return 3;
        }
        out->loaded_bytes += shard_bytes;

        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        (void)cudaGetLastError();
        uint8_t *d_w = nullptr;
        float *d_x = nullptr;
        __half *d_w_half = nullptr;
        __half *d_x_half = nullptr;
        cublasHandle_t blas = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaMalloc(&d_w, (size_t)shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_out[(size_t)gpu],
                              (size_t)opt.slots * rows_per_gpu * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), (size_t)shard_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (opt.dense_f16_cublas_compose) {
            CHECK_CUDA(cudaMalloc(&d_w_half, (size_t)rows_per_gpu * cols * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&d_x_half, h_x.size() * sizeof(__half)));
            const uint64_t w_elems = (uint64_t)rows_per_gpu * cols;
            f8_b128_to_half_kernel<<<(unsigned int)((w_elems + 255) / 256), 256>>>(
                d_w_half, d_w, rows_per_gpu, cols, (uint32_t)row_bytes);
            CHECK_CUDA(cudaGetLastError());
            const uint64_t x_elems = (uint64_t)opt.slots * cols;
            cast_f32_to_half_kernel<<<(unsigned int)((x_elems + 255) / 256), 256>>>(
                d_x_half, d_x, x_elems);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            cublasStatus_t st = cublasCreate(&blas);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "cublasCreate failed on gpu %d: %d\n", gpu, (int)st);
                return 4;
            }
            (void)cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH);
        }
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        const dim3 scalar_grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1);
        const dim3 hmma_grid((unsigned int)((rows_per_gpu + 63) / 64),
                             (unsigned int)((opt.slots + 15) / 16),
                             1);
        for (int i = 0; i < opt.warmup; ++i) {
            if (opt.dense_f16_cublas_compose) {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                cublasStatus_t st = cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                                  rows_per_gpu, opt.slots, cols,
                                                  &alpha, d_w_half, CUDA_R_16F, cols,
                                                  d_x_half, CUDA_R_16F, cols,
                                                  &beta, out->d_out[(size_t)gpu],
                                                  CUDA_R_32F, rows_per_gpu,
                                                  CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS) return 5;
            } else if (opt.dense_hmma_compose) {
                f8_b128_dense_hmma_m16_kernel<<<hmma_grid, 128>>>(
                    out->d_out[(size_t)gpu], d_w, d_x, rows_per_gpu, cols,
                    (uint32_t)row_bytes, opt.slots);
            } else {
                f8_b128_dense_kernel<<<scalar_grid, 256>>>(
                    out->d_out[(size_t)gpu], d_w, d_x, rows_per_gpu, cols,
                    (uint32_t)row_bytes, opt.slots);
            }
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < opt.iters; ++i) {
            if (opt.dense_f16_cublas_compose) {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                cublasStatus_t st = cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                                  rows_per_gpu, opt.slots, cols,
                                                  &alpha, d_w_half, CUDA_R_16F, cols,
                                                  d_x_half, CUDA_R_16F, cols,
                                                  &beta, out->d_out[(size_t)gpu],
                                                  CUDA_R_32F, rows_per_gpu,
                                                  CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS) return 6;
            } else if (opt.dense_hmma_compose) {
                f8_b128_dense_hmma_m16_kernel<<<hmma_grid, 128>>>(
                    out->d_out[(size_t)gpu], d_w, d_x, rows_per_gpu, cols,
                    (uint32_t)row_bytes, opt.slots);
            } else {
                f8_b128_dense_kernel<<<scalar_grid, 256>>>(
                    out->d_out[(size_t)gpu], d_w, d_x, rows_per_gpu, cols,
                    (uint32_t)row_bytes, opt.slots);
            }
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        worst_ms = std::max(worst_ms, (double)ms / opt.iters);
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_x));
        if (d_w_half) CHECK_CUDA(cudaFree(d_w_half));
        if (d_x_half) CHECK_CUDA(cudaFree(d_x_half));
        if (blas) (void)cublasDestroy(blas);
    }
    out->compute_ms = worst_ms;
    return 0;
}

bool parse_tm_entry(const std::vector<std::string> &f, TmIndexEntry *out) {
    if (f.size() < 25) return false;
    TmIndexEntry e;
    e.semantic_tensor_id = f[0];
    e.runtime_layout = f[4];
    if (!parse_int(f[6].c_str(), &e.layer_id)) return false;
    if (!parse_int(f[8].c_str(), &e.n)) return false;
    if (!parse_int(f[9].c_str(), &e.k)) return false;
    if (!parse_int(f[10].c_str(), &e.experts_packed)) return false;
    if (!parse_int(f[11].c_str(), &e.experts_total)) return false;
    if (!parse_size(f[12].c_str(), &e.weight_bytes_per_expert)) return false;
    if (!parse_size(f[13].c_str(), &e.scale_bytes_per_expert)) return false;
    if (!parse_int(f[14].c_str(), &e.k_pack)) return false;
    if (!parse_int(f[15].c_str(), &e.weight_stride)) return false;
    if (!parse_int(f[16].c_str(), &e.scale_stride)) return false;
    e.sidecar_file = f[17];
    if (!parse_u64(f[18].c_str(), &e.weight_offset)) return false;
    if (!parse_u64(f[19].c_str(), &e.scale_offset)) return false;
    if (!safe_sidecar_name(e.sidecar_file)) return false;
    *out = e;
    return true;
}

bool valid_tm_entry(const TmIndexEntry &e, int n, int k, const char *layout) {
    return e.n == n &&
           e.k == k &&
           e.experts_total == kGlobalExperts &&
           e.experts_packed >= kGlobalExperts &&
           e.weight_bytes_per_expert > 0 &&
           e.scale_bytes_per_expert > 0 &&
           e.k_pack > 0 &&
           e.weight_stride > 0 &&
           e.scale_stride > 0 &&
           e.runtime_layout == layout;
}

int parse_tm_index(const char *path, int layer, DescriptorBindings *out) {
    FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "cannot open tm index %s: %s\n", path, std::strerror(errno));
        return 1;
    }
    char gated_name[128];
    char down_name[128];
    std::snprintf(gated_name, sizeof(gated_name), "blk.%d.ffn_gate_up_exps.weight", layer);
    std::snprintf(down_name, sizeof(down_name), "blk.%d.ffn_down_exps.weight", layer);
    char buf[8192];
    bool first = true;
    while (std::fgets(buf, sizeof(buf), fp)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (first) {
            first = false;
            continue;
        }
        if (line.empty()) continue;
        std::vector<std::string> f = split_tabs(line);
        TmIndexEntry e;
        if (!parse_tm_entry(f, &e)) {
            std::fclose(fp);
            return 2;
        }
        if (e.layer_id != layer) continue;
        if (e.semantic_tensor_id == gated_name) {
            if (!valid_tm_entry(e, kFusedN, kHidden,
                                "turbomind_mxfp4_grouped_gate_up_interleaved")) {
                std::fclose(fp);
                return 3;
            }
            out->gated = e;
            out->have_gated = true;
        } else if (e.semantic_tensor_id == down_name) {
            if (!valid_tm_entry(e, kHidden, kMid, "turbomind_mxfp4_grouped")) {
                std::fclose(fp);
                return 4;
            }
            out->down = e;
            out->have_down = true;
        }
    }
    std::fclose(fp);
    return out->have_gated && out->have_down ? 0 : 5;
}

void load_api(void *lib, Api *api) {
    api->init = (pfn_init)dlsym(lib, "ggml_turbomind_init");
    api->shutdown = (pfn_shutdown)dlsym(lib, "ggml_turbomind_shutdown");
    api->mmgt = (pfn_mmgt)dlsym(lib, "ggml_turbomind_mul_mat_grouped_total_tokens");
    api->mmgs = (pfn_mmgs)dlsym(lib, "ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens");
    api->mmgs_clamped =
        (pfn_mmgs)dlsym(lib, "ggml_turbomind_mul_mat_grouped_gated_silu_clamped_total_tokens");
    if (!api->init || !api->shutdown || !api->mmgt || !api->mmgs) {
        std::fprintf(stderr, "dlsym failed for required TurboMind ABI\n");
        std::exit(2);
    }
}

int open_shared_api(const Options &opt, SharedApi *shared) {
    shared->lib = dlopen(opt.lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!shared->lib) {
        std::fprintf(stderr, "dlopen failed for %s: %s\n", opt.lib_path, dlerror());
        return 1;
    }
    load_api(shared->lib, &shared->api);
    for (int p = 0; p < kGpus; ++p) {
        if (shared->api.init(opt.devices[p]) != 0) {
            std::fprintf(stderr, "ggml_turbomind_init failed on device %d\n", opt.devices[p]);
            if (shared->api.shutdown) shared->api.shutdown();
            dlclose(shared->lib);
            *shared = SharedApi{};
            return 2;
        }
    }
    shared->initialized = true;
    return 0;
}

void close_shared_api(SharedApi *shared) {
    if (!shared || !shared->lib) return;
    if (shared->initialized && shared->api.shutdown) shared->api.shutdown();
    dlclose(shared->lib);
    *shared = SharedApi{};
}

void free_packed(PackedExperts &p) {
    for (void *v : p.d_w_active) {
        if (v) CHECK_CUDA(cudaFree(v));
    }
    for (void *v : p.d_s_active) {
        if (v) CHECK_CUDA(cudaFree(v));
    }
    if (p.d_w_table) CHECK_CUDA(cudaFree(p.d_w_table));
    if (p.d_s_table) CHECK_CUDA(cudaFree(p.d_s_table));
    p = PackedExperts{};
}

int pack_descriptor_set(int device, const TmIndexEntry &entry, int rank,
                        const std::vector<int> &active, const char *pack_dir,
                        PackedExperts *out, uint64_t *host_bytes_read) {
    CHECK_CUDA(cudaSetDevice(device));
    const std::string sidecar_path = path_join(pack_dir, entry.sidecar_file);
    out->d_w_active.assign(active.size(), nullptr);
    out->d_s_active.assign(active.size(), nullptr);
    out->k_pack = entry.k_pack;

    std::vector<uint8_t> h_weight(entry.weight_bytes_per_expert);
    std::vector<uint8_t> h_scale(entry.scale_bytes_per_expert);
    for (size_t i = 0; i < active.size(); ++i) {
        const int global_expert = rank * kLocalExperts + active[i];
        const uint64_t w_off = entry.weight_offset +
                               (uint64_t)global_expert * entry.weight_bytes_per_expert;
        const uint64_t s_off = entry.scale_offset +
                               (uint64_t)global_expert * entry.scale_bytes_per_expert;
        if (read_exact_at(sidecar_path, w_off, h_weight.data(), h_weight.size()) != 0 ||
            read_exact_at(sidecar_path, s_off, h_scale.data(), h_scale.size()) != 0) {
            return 1;
        }
        CHECK_CUDA(cudaMalloc(&out->d_w_active[i], h_weight.size()));
        CHECK_CUDA(cudaMalloc(&out->d_s_active[i], h_scale.size()));
        CHECK_CUDA(cudaMemcpy(out->d_w_active[i], h_weight.data(), h_weight.size(),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_s_active[i], h_scale.data(), h_scale.size(),
                              cudaMemcpyHostToDevice));
        *host_bytes_read += (uint64_t)h_weight.size() + (uint64_t)h_scale.size();
    }

    std::vector<StridedPtrH> w_table((size_t)kLocalExperts);
    std::vector<StridedPtrH> s_table((size_t)kLocalExperts);
    for (int e = 0; e < kLocalExperts; ++e) {
        w_table[(size_t)e] = StridedPtrH{out->d_w_active[0], entry.weight_stride};
        s_table[(size_t)e] = StridedPtrH{out->d_s_active[0], entry.scale_stride};
    }
    for (size_t i = 0; i < active.size(); ++i) {
        w_table[(size_t)active[i]] = StridedPtrH{out->d_w_active[i], entry.weight_stride};
        s_table[(size_t)active[i]] = StridedPtrH{out->d_s_active[i], entry.scale_stride};
    }
    CHECK_CUDA(cudaMalloc(&out->d_w_table, w_table.size() * sizeof(StridedPtrH)));
    CHECK_CUDA(cudaMemcpy(out->d_w_table, w_table.data(),
                          w_table.size() * sizeof(StridedPtrH), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&out->d_s_table, s_table.size() * sizeof(StridedPtrH)));
    CHECK_CUDA(cudaMemcpy(out->d_s_table, s_table.data(),
                          s_table.size() * sizeof(StridedPtrH), cudaMemcpyHostToDevice));
    return 0;
}

void free_layer_expert_cache(LayerExpertCache *layer) {
    if (!layer) return;
    for (int p = 0; p < kGpus; ++p) {
        free_packed(layer->gated[p]);
        free_packed(layer->down[p]);
    }
    *layer = LayerExpertCache{};
}

void close_shared_expert_bindings(SharedExpertBindings *shared);

int open_shared_expert_bindings(const Options &opt, SharedExpertBindings *shared) {
    std::vector<int> active;
    for (int e = 0; e < kPackedLocalExperts; ++e) active.push_back(e);

    for (int layer = 0; layer < 43; ++layer) {
        LayerExpertCache &cache = shared->layers[layer];
        if (parse_tm_index(opt.tm_index_path, layer, &cache.bindings) != 0) {
            std::fprintf(stderr, "tm index parse failed for layer %d\n", layer);
            close_shared_expert_bindings(shared);
            return 1;
        }
        for (int p = 0; p < kGpus; ++p) {
            uint64_t layer_bytes = 0;
            if (pack_descriptor_set(opt.devices[p], cache.bindings.gated, p, active,
                                    opt.pack_dir, &cache.gated[p], &layer_bytes) != 0 ||
                pack_descriptor_set(opt.devices[p], cache.bindings.down, p, active,
                                    opt.pack_dir, &cache.down[p], &layer_bytes) != 0) {
                close_shared_expert_bindings(shared);
                return 2;
            }
            cache.bytes += layer_bytes;
            shared->bytes += layer_bytes;
        }
        cache.initialized = true;
    }
    shared->initialized = true;
    return 0;
}

void close_shared_expert_bindings(SharedExpertBindings *shared) {
    if (!shared) return;
    for (int layer = 0; layer < 43; ++layer) {
        free_layer_expert_cache(&shared->layers[layer]);
    }
    *shared = SharedExpertBindings{};
}

int run_gate(RankState &rank, const Api &api) {
    if (rank.routes <= 0) return 0;
    return api.mmgs(rank.d_a, nullptr, rank.d_offsets, kLocalExperts, rank.routes,
                    (const void * const *)rank.gated.d_w_table,
                    (const void * const *)rank.gated.d_s_table,
                    kDType, kFusedN, kHidden, kGroupSize, rank.gated.k_pack,
                    rank.d_gated, rank.stream);
}

int run_gate_clamped(RankState &rank, const Api &api, bool apply_route_scale) {
    if (rank.routes <= 0) return 0;
    if (!rank.d_gate_up) return 1;
    CHECK_CUDA(cudaSetDevice(rank.device));
    const int rc = api.mmgt(rank.d_a, nullptr, rank.d_offsets, kLocalExperts, rank.routes,
                            (const void * const *)rank.gated.d_w_table,
                            (const void * const *)rank.gated.d_s_table,
                            kDType, kFusedN, kHidden, kGroupSize, rank.gated.k_pack,
                            rank.d_gate_up, rank.stream);
    if (rc != 0) return rc;
    const uint64_t elems = (uint64_t)rank.routes * kMid;
    routed_fused_gate_up_swiglu_clamp_kernel<<<
        (unsigned int)((elems + 255) / 256), 256, 0, rank.stream>>>(
            rank.d_gated, rank.d_gate_up,
            apply_route_scale ? rank.d_route_inv_scale : nullptr,
            (uint64_t)rank.routes, kRoutedSwigluClamp);
    CHECK_CUDA(cudaGetLastError());
    return 0;
}

int run_gate_selected(RankState &rank, const Api &api, const Options &opt) {
    if (!opt.routed_ffn_norm_input_gate) {
        return run_gate(rank, api);
    }
    if (opt.fused_gated_silu_gate && !opt.reference_hc_reduce_gate &&
        api.mmgs_clamped) {
        return api.mmgs_clamped(
            rank.d_a, nullptr, rank.d_offsets, kLocalExperts, rank.routes,
            (const void * const *)rank.gated.d_w_table,
            (const void * const *)rank.gated.d_s_table,
            kDType, kFusedN, kHidden, kGroupSize, rank.gated.k_pack,
            rank.d_gated, rank.stream);
    }
    return run_gate_clamped(rank, api, opt.reference_hc_reduce_gate);
}

int run_down(RankState &rank, const Api &api) {
    if (rank.routes <= 0) return 0;
    return api.mmgt(rank.d_gated, nullptr, rank.d_offsets, kLocalExperts, rank.routes,
                    (const void * const *)rank.down.d_w_table,
                    (const void * const *)rank.down.d_s_table,
                    kDType, kHidden, kMid, kGroupSize, rank.down.k_pack,
                    rank.d_down, rank.stream);
}

void log_route_half_stats(const char *tag, int layer, int rank_id,
                          const __half *ptr, size_t elems, cudaStream_t stream) {
    if (!ptr || elems == 0) return;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<__half> host(elems);
    CHECK_CUDA(cudaMemcpy(host.data(), ptr, elems * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    int finite_bad = 0;
    size_t first_bad = (size_t)-1;
    float max_abs = 0.0f;
    for (size_t i = 0; i < elems; ++i) {
        const float v = __half2float(host[i]);
        if (!std::isfinite(v)) {
            if (finite_bad == 0) first_bad = i;
            ++finite_bad;
        } else {
            max_abs = fmaxf(max_abs, fabsf(v));
        }
    }
    std::fprintf(stderr,
                 "tp_ep_route_tensor_stats\ttag\t%s\tlayer\t%d\trank\t%d\telems\t%zu\tfinite_bad\t%d\tfirst_bad\t%zu\tmax_abs\t%.9g\n",
                 tag, layer, rank_id, elems, finite_bad, first_bad, max_abs);
}

void merge_tensor_stats(TensorF32Stats *dst, const TensorF32Stats &src) {
    if (!dst) return;
    if (src.finite_bad != 0 && dst->finite_bad == 0) {
        dst->first_bad = src.first_bad;
    }
    dst->finite_bad += src.finite_bad;
    dst->max_abs = fmaxf(dst->max_abs, src.max_abs);
}

TensorF32Stats collect_tensor_f32_stats(const float *ptr, size_t elems,
                                        cudaStream_t stream) {
    TensorF32Stats stats;
    if (!ptr || elems == 0) return stats;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> host(elems);
    CHECK_CUDA(cudaMemcpy(host.data(), ptr, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elems; ++i) {
        const float v = host[i];
        if (!std::isfinite(v)) {
            if (stats.finite_bad == 0) stats.first_bad = i;
            ++stats.finite_bad;
        } else {
            stats.max_abs = fmaxf(stats.max_abs, fabsf(v));
        }
    }
    return stats;
}

TensorF32Stats collect_raw_swa_row_stats(const float *ptr, uint32_t slots,
                                         uint32_t raw_rows, uint32_t raw_row,
                                         uint32_t head_dim,
                                         cudaStream_t stream) {
    TensorF32Stats stats;
    if (!ptr || slots == 0 || raw_rows == 0 || raw_row >= raw_rows ||
        head_dim == 0) {
        return stats;
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> host((size_t)slots * (size_t)head_dim);
    const float *src = ptr + (uint64_t)raw_row * (uint64_t)head_dim;
    CHECK_CUDA(cudaMemcpy2D(host.data(), (size_t)head_dim * sizeof(float),
                            src,
                            (size_t)raw_rows * (size_t)head_dim * sizeof(float),
                            (size_t)head_dim * sizeof(float), (size_t)slots,
                            cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < host.size(); ++i) {
        const float v = host[i];
        if (!std::isfinite(v)) {
            if (stats.finite_bad == 0) stats.first_bad = i;
            ++stats.finite_bad;
        } else {
            stats.max_abs = fmaxf(stats.max_abs, fabsf(v));
        }
    }
    return stats;
}

TensorF32DiffStats collect_tensor_f32_diff_stats(const float *a, const float *b,
                                                 size_t elems,
                                                 cudaStream_t stream) {
    TensorF32DiffStats stats;
    if (!a || !b || elems == 0) return stats;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> ha(elems);
    std::vector<float> hb(elems);
    CHECK_CUDA(cudaMemcpy(ha.data(), a, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hb.data(), b, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elems; ++i) {
        const float av = ha[i];
        const float bv = hb[i];
        if (!std::isfinite(av) || !std::isfinite(bv)) {
            if (stats.bad == 0) stats.first_bad = i;
            ++stats.bad;
            continue;
        }
        const float diff = fabsf(av - bv);
        const float denom = fmaxf(fabsf(bv), 1.0e-12f);
        stats.max_abs = fmaxf(stats.max_abs, diff);
        stats.max_rel = fmaxf(stats.max_rel, diff / denom);
    }
    return stats;
}

void log_tensor_f32_diff_summary(const char *tag, int layer,
                                 const float *got, const float *ref,
                                 size_t elems, cudaStream_t stream) {
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> hg(elems);
    std::vector<float> hr(elems);
    CHECK_CUDA(cudaMemcpy(hg.data(), got, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hr.data(), ref, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double sq = 0.0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float got_max = 0.0f;
    float ref_max = 0.0f;
    int finite_bad = 0;
    size_t first_bad = (size_t)-1;
    for (size_t i = 0; i < elems; ++i) {
        const float g = hg[i];
        const float r = hr[i];
        if (!std::isfinite(g) || !std::isfinite(r)) {
            if (finite_bad == 0) first_bad = i;
            ++finite_bad;
            continue;
        }
        got_max = fmaxf(got_max, fabsf(g));
        ref_max = fmaxf(ref_max, fabsf(r));
        const float diff = fabsf(g - r);
        const float rel = diff / fmaxf(fabsf(r), 1.0e-12f);
        max_abs = fmaxf(max_abs, diff);
        max_rel = fmaxf(max_rel, rel);
        sq += (double)diff * (double)diff;
    }
    const double rms = elems ? std::sqrt(sq / (double)elems) : 0.0;
    const char *status = (finite_bad == 0 && max_abs <= 1.0e-5f) ? "PASS" : "DIFF";
    std::printf("tp_ep_compressed_reference_diff\tlayer\t%d\ttensor\t%s\t"
                "elems\t%zu\tmax_abs\t%.9g\trms\t%.9g\tmax_rel\t%.9g\t"
                "finite_bad\t%d\tfirst_bad\t%zu\tgot_max\t%.9g\t"
                "reference_max\t%.9g\t%s\n",
                layer, tag, elems, max_abs, rms, max_rel, finite_bad,
                first_bad, got_max, ref_max, status);
}

void log_tensor_f32_stats(const char *tag, int layer, int rank_id,
                          const float *ptr, size_t elems, cudaStream_t stream) {
    const TensorF32Stats stats = collect_tensor_f32_stats(ptr, elems, stream);
    std::fprintf(stderr,
                 "tp_ep_tensor_stats\ttag\t%s\tlayer\t%d\trank\t%d\telems\t%zu\tfinite_bad\t%d\tfirst_bad\t%zu\tmax_abs\t%.9g\n",
                 tag, layer, rank_id, elems, stats.finite_bad, stats.first_bad,
                 stats.max_abs);
}

bool should_log_routed_semantic_stats(const Options &opt) {
    if (!opt.routed_ffn_norm_input_gate) return false;
    if (opt.layer <= 2) return true;
    return opt.reference_hc_reduce_gate && opt.layer >= 30 && opt.layer <= 32;
}

bool should_log_reference_hc_window(const Options &opt) {
    return opt.reference_hc_reduce_gate && opt.layer >= 30 && opt.layer <= 32;
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

int check_repeat(RankState &rank, const Api &api, double *max_abs, int *bad, int *nan) {
    const size_t elems = (size_t)rank.routes * kHidden;
    std::vector<__half> first(elems);
    std::vector<__half> second(elems);
    CHECK_CUDA(cudaSetDevice(rank.device));
    if (run_gate(rank, api) != 0 || run_down(rank, api) != 0) return 1;
    CHECK_CUDA(cudaStreamSynchronize(rank.stream));
    CHECK_CUDA(cudaMemcpy(first.data(), rank.d_down, elems * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    if (run_gate(rank, api) != 0 || run_down(rank, api) != 0) return 1;
    CHECK_CUDA(cudaStreamSynchronize(rank.stream));
    CHECK_CUDA(cudaMemcpy(second.data(), rank.d_down, elems * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elems; ++i) {
        const float a = __half2float(first[i]);
        const float b = __half2float(second[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++*nan;
            continue;
        }
        const double diff = std::fabs((double)a - (double)b);
        *max_abs = std::max(*max_abs, diff);
        if (diff > 0.0) ++*bad;
    }
    return 0;
}

void build_offsets_for_rank(int rank, int slots, int top_k,
                            std::vector<int> *offsets,
                            std::vector<int> *route_slots,
                            std::vector<float> *route_weights,
                            int *routes,
                            int *active_experts,
                            int *max_routes_per_expert) {
    std::vector<int> counts((size_t)kLocalExperts, 0);
    for (int slot = 0; slot < slots; ++slot) {
        for (int k = 0; k < top_k; ++k) {
            const int dst_rank = (slot * top_k + k) % kGpus;
            if (dst_rank != rank) continue;
            const int local = (slot + k * 7 + rank) % kPackedLocalExperts;
            counts[(size_t)local]++;
        }
    }
    offsets->assign((size_t)kLocalExperts + 1, 0);
    int running = 0;
    int active = 0;
    int max_routes = 0;
    for (int e = 0; e < kLocalExperts; ++e) {
        (*offsets)[(size_t)e] = running;
        running += counts[(size_t)e];
        if (counts[(size_t)e] > 0) ++active;
        max_routes = std::max(max_routes, counts[(size_t)e]);
    }
    (*offsets)[(size_t)kLocalExperts] = running;
    if (route_slots) {
        route_slots->assign((size_t)running, -1);
        if (route_weights) route_weights->assign((size_t)running, kSyntheticRouteWeight);
        std::vector<int> cursor = *offsets;
        for (int slot = 0; slot < slots; ++slot) {
            for (int k = 0; k < top_k; ++k) {
                const int dst_rank = (slot * top_k + k) % kGpus;
                if (dst_rank != rank) continue;
                const int local = (slot + k * 7 + rank) % kPackedLocalExperts;
                const int idx = cursor[(size_t)local]++;
                (*route_slots)[(size_t)idx] = slot;
                if (route_weights) (*route_weights)[(size_t)idx] = kSyntheticRouteWeight;
            }
        }
    }
    *routes = running;
    *active_experts = active;
    *max_routes_per_expert = max_routes;
}

void build_route_index_by_slot_for_rank(int rank, int slots, int top_k,
                                        std::vector<int> *route_index_by_slot) {
    std::vector<int> offsets;
    std::vector<int> route_slots;
    int routes = 0;
    int active_experts = 0;
    int max_routes_per_expert = 0;
    build_offsets_for_rank(rank, slots, top_k, &offsets, &route_slots, nullptr, &routes,
                           &active_experts, &max_routes_per_expert);
    route_index_by_slot->assign((size_t)slots, -1);
    for (int route = 0; route < routes; ++route) {
        const int slot = route_slots[(size_t)route];
        if (slot >= 0 && slot < slots) {
            (*route_index_by_slot)[(size_t)slot] = route;
        }
    }
}

void build_route_indices_by_slot_for_rank(int rank, int slots, int top_k,
                                          std::vector<int> *route_indices,
                                          std::vector<int> *route_counts) {
    std::vector<int> offsets;
    std::vector<int> route_slots;
    int routes = 0;
    int active_experts = 0;
    int max_routes_per_expert = 0;
    build_offsets_for_rank(rank, slots, top_k, &offsets, &route_slots, nullptr, &routes,
                           &active_experts, &max_routes_per_expert);
    route_indices->assign((size_t)slots * (size_t)top_k, -1);
    route_counts->assign((size_t)slots, 0);
    for (int route = 0; route < routes; ++route) {
        const int slot = route_slots[(size_t)route];
        if (slot < 0 || slot >= slots) continue;
        int &count = (*route_counts)[(size_t)slot];
        if (count < top_k) {
            (*route_indices)[(size_t)slot * (size_t)top_k + (size_t)count] = route;
            count++;
        }
    }
}

size_t compact_route_plan_ints(const Options &opt) {
    const size_t indices = (size_t)opt.slots * (size_t)opt.top_k;
    const size_t counts = (size_t)opt.slots;
    return (size_t)kGpus * (indices + counts);
}

void bind_compact_route_plan(RankState *r, const Options &opt) {
    const size_t indices = (size_t)opt.slots * (size_t)opt.top_k;
    const size_t counts = (size_t)opt.slots;
    int *base = r->d_route_compact_plan;
    for (int src = 0; src < kGpus; ++src) {
        r->d_route_indices_by_slot[src] = base + (size_t)src * indices;
        r->d_route_count_by_slot[src] =
            base + (size_t)kGpus * indices + (size_t)src * counts;
    }
}

int upload_model_router_route_plan_gpu(const Options &opt,
                                       SharedHcControls *hc,
                                       RankState ranks[kGpus]) {
    if (!opt.compact_moe_decode_gate || !hc || !hc->d_router_selected ||
        !hc->d_router_weights) {
        return 1;
    }
    const size_t selected_bytes =
        (size_t)opt.slots * (size_t)opt.top_k * sizeof(int);
    const size_t weights_bytes =
        (size_t)opt.slots * (size_t)opt.top_k * sizeof(float);
    const size_t offsets_all_bytes =
        (size_t)kGpus * (size_t)(kLocalExperts + 1) * sizeof(int);
    const int block = 256;
    const uint32_t route_entries = (uint32_t)(opt.slots * opt.top_k);
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.d_router_selected_plan || !r.d_router_weights_plan ||
            !r.d_route_offsets_all || !r.d_route_totals ||
            !r.d_route_compact_plan) {
            return 2;
        }
        CHECK_CUDA(cudaSetDevice(r.device));
        if (rank == 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_router_selected_plan,
                                       hc->d_router_selected,
                                       selected_bytes,
                                       cudaMemcpyDeviceToDevice,
                                       r.stream));
            CHECK_CUDA(cudaMemcpyAsync(r.d_router_weights_plan,
                                       hc->d_router_weights,
                                       weights_bytes,
                                       cudaMemcpyDeviceToDevice,
                                       r.stream));
        } else {
            CHECK_CUDA(cudaMemcpyPeerAsync(r.d_router_selected_plan,
                                           r.device,
                                           hc->d_router_selected,
                                           opt.devices[0],
                                           selected_bytes,
                                           r.stream));
            CHECK_CUDA(cudaMemcpyPeerAsync(r.d_router_weights_plan,
                                           r.device,
                                           hc->d_router_weights,
                                           opt.devices[0],
                                           weights_bytes,
                                           r.stream));
        }
        CHECK_CUDA(cudaMemsetAsync(r.d_route_offsets_all, 0,
                                   offsets_all_bytes, r.stream));
        CHECK_CUDA(cudaMemsetAsync(r.d_route_totals, 0,
                                   (size_t)kGpus * sizeof(int), r.stream));
        gpu_route_count_all_kernel<<<
            (unsigned int)((route_entries + block - 1) / block), block,
            0, r.stream>>>(
            r.d_router_selected_plan, r.d_route_offsets_all,
            (uint32_t)opt.slots, (uint32_t)opt.top_k);
        gpu_route_prefix_all_kernel<<<1, kGpus, 0, r.stream>>>(
            r.d_route_offsets_all, r.d_route_totals);
        gpu_route_init_compact_plan_kernel<<<
            (unsigned int)((compact_route_plan_ints(opt) + block - 1) / block),
            block, 0, r.stream>>>(
            r.d_route_compact_plan, (uint32_t)opt.slots, (uint32_t)opt.top_k);
        gpu_route_copy_own_offsets_kernel<<<1, kLocalExperts + 1, 0, r.stream>>>(
            r.d_offsets, r.d_route_offsets_all, (uint32_t)rank);
        gpu_route_fill_all_kernel<<<
            (unsigned int)((route_entries + block - 1) / block), block,
            0, r.stream>>>(
            r.d_router_selected_plan, r.d_router_weights_plan,
            r.d_route_offsets_all, rank, r.d_route_slots, r.d_route_weights,
            r.d_route_compact_plan, (uint32_t)opt.slots, (uint32_t)opt.top_k);
        CHECK_CUDA(cudaGetLastError());
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }
    std::vector<int> totals((size_t)kGpus, 0);
    std::vector<int> offsets_all((size_t)kGpus * (size_t)(kLocalExperts + 1), 0);
    CHECK_CUDA(cudaSetDevice(ranks[0].device));
    CHECK_CUDA(cudaMemcpy(totals.data(), ranks[0].d_route_totals,
                          totals.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(offsets_all.data(), ranks[0].d_route_offsets_all,
                          offsets_all.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    for (int rank = 0; rank < kGpus; ++rank) {
        if (totals[(size_t)rank] > ranks[rank].route_capacity) return 3;
        ranks[rank].routes = totals[(size_t)rank];
        int active = 0;
        int max_routes = 0;
        const int *off = offsets_all.data() + (size_t)rank * (kLocalExperts + 1);
        for (int local = 0; local < kLocalExperts; ++local) {
            const int count = off[local + 1] - off[local];
            if (count > 0) ++active;
            max_routes = std::max(max_routes, count);
        }
        ranks[rank].active_experts = active;
        ranks[rank].max_routes_per_expert = max_routes;
    }
    static bool route_stats_emitted[43] = {};
    if (opt.model_router_routes && opt.layer >= 0 && opt.layer <= 5 &&
        !route_stats_emitted[opt.layer]) {
        route_stats_emitted[opt.layer] = true;
        std::fprintf(stderr,
                     "tp_ep_model_router_route_stats\tlayer\t%d\troutes\t%d,%d,%d,%d,%d,%d,%d,%d\tmax_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                     opt.layer,
                     ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                     ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                     ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                     ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                     ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                     ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
    }
    int duplicate_slots = 0;
    int max_same_rank_routes = 0;
    std::vector<int> compact_counts((size_t)kGpus * (size_t)opt.slots, 0);
    CHECK_CUDA(cudaSetDevice(ranks[0].device));
    CHECK_CUDA(cudaMemcpy(
        compact_counts.data(),
        ranks[0].d_route_compact_plan +
            (size_t)kGpus * (size_t)opt.slots * (size_t)opt.top_k,
        compact_counts.size() * sizeof(int),
        cudaMemcpyDeviceToHost));
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int rank = 0; rank < kGpus; ++rank) {
            const int c = compact_counts[(size_t)rank * (size_t)opt.slots + slot];
            max_same_rank_routes = std::max(max_same_rank_routes, c);
            if (c > 1) duplicate_slots++;
        }
    }
    static bool compact_stats_emitted[43] = {};
    if (opt.layer >= 0 && opt.layer < 43 && !compact_stats_emitted[opt.layer]) {
        compact_stats_emitted[opt.layer] = true;
        const uint64_t all_dest_bytes =
            (uint64_t)kGpus * (uint64_t)kGpus * (uint64_t)opt.slots *
            (uint64_t)(kHidden / kGpus) * sizeof(float);
        const uint64_t total_routes =
            (uint64_t)ranks[0].routes + (uint64_t)ranks[1].routes +
            (uint64_t)ranks[2].routes + (uint64_t)ranks[3].routes +
            (uint64_t)ranks[4].routes + (uint64_t)ranks[5].routes +
            (uint64_t)ranks[6].routes + (uint64_t)ranks[7].routes;
        const uint64_t compact_bytes =
            (uint64_t)kGpus * total_routes * (uint64_t)(kHidden / kGpus) *
            sizeof(float);
        std::printf("tp_ep_compact_moe_route_stats\tlayer\t%d\t"
                    "duplicate_slots\t%d\tmax_same_rank_routes\t%d\t"
                    "all_dest_bytes\t%llu\tcompact_bytes\t%llu\t"
                    "routes\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                    "active_experts\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                    "max_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                    opt.layer, duplicate_slots, max_same_rank_routes,
                    (unsigned long long)all_dest_bytes,
                    (unsigned long long)compact_bytes,
                    ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                    ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                    ranks[0].active_experts, ranks[1].active_experts,
                    ranks[2].active_experts, ranks[3].active_experts,
                    ranks[4].active_experts, ranks[5].active_experts,
                    ranks[6].active_experts, ranks[7].active_experts,
                    ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                    ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                    ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                    ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
    }
    return 0;
}

int upload_model_router_route_plan(const Options &opt,
                                   RankState ranks[kGpus],
                                   const std::vector<int> &selected,
                                   const std::vector<float> &weights) {
    if ((int)selected.size() < opt.slots * opt.top_k ||
        (int)weights.size() < opt.slots * opt.top_k) {
        return 1;
    }
    std::vector<int> offsets[kGpus];
    std::vector<int> route_slots[kGpus];
    std::vector<float> route_weights[kGpus];
    std::vector<int> route_index_by_slot[kGpus];
    std::vector<int> route_indices_by_slot[kGpus];
    std::vector<int> route_count_by_slot[kGpus];
    std::vector<int> counts[kGpus];
    const bool needs_single_route_index = !opt.compact_moe_decode_gate;
    const bool needs_packed_compact_plan = opt.compact_moe_decode_gate;
    for (int rank = 0; rank < kGpus; ++rank) {
        counts[rank].assign((size_t)kLocalExperts, 0);
        if (needs_single_route_index) {
            route_index_by_slot[rank].assign((size_t)opt.slots, -1);
        }
        route_indices_by_slot[rank].assign((size_t)opt.slots * (size_t)opt.top_k,
                                           -1);
        route_count_by_slot[rank].assign((size_t)opt.slots, 0);
    }
    bool compact_duplicate = false;
    for (int slot = 0; slot < opt.slots; ++slot) {
        bool seen_rank[kGpus] = {};
        for (int k = 0; k < opt.top_k; ++k) {
            const int expert = selected[(size_t)slot * opt.top_k + (size_t)k];
            if (expert < 0) continue;
            if (expert < 0 || expert >= kGlobalExperts) return 2;
            const int rank = expert / kLocalExperts;
            const int local = expert % kLocalExperts;
            counts[rank][(size_t)local]++;
            if (seen_rank[rank]) compact_duplicate = true;
            seen_rank[rank] = true;
        }
    }
    if (opt.compact_route_compose && compact_duplicate &&
        !opt.compact_moe_decode_gate) {
        return 3;
    }
    if (opt.routed_ffn_norm_input_gate && opt.layer >= 0 && opt.layer <= 2) {
        for (int slot = 0; slot < opt.slots; ++slot) {
            for (int k = 0; k < opt.top_k; ++k) {
                const int expert = selected[(size_t)slot * opt.top_k + (size_t)k];
                if (expert < 0) continue;
                const int rank = expert / kLocalExperts;
                const int local = expert % kLocalExperts;
                const float w = weights[(size_t)slot * opt.top_k + (size_t)k];
                StridedPtrH gw = {};
                StridedPtrH gs = {};
                StridedPtrH dw = {};
                StridedPtrH ds = {};
                if (rank >= 0 && rank < kGpus) {
                    CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                    if (ranks[rank].gated.d_w_table) {
                        CHECK_CUDA(cudaMemcpy(&gw,
                                              (const StridedPtrH *)ranks[rank].gated.d_w_table + local,
                                              sizeof(gw), cudaMemcpyDeviceToHost));
                    }
                    if (ranks[rank].gated.d_s_table) {
                        CHECK_CUDA(cudaMemcpy(&gs,
                                              (const StridedPtrH *)ranks[rank].gated.d_s_table + local,
                                              sizeof(gs), cudaMemcpyDeviceToHost));
                    }
                    if (ranks[rank].down.d_w_table) {
                        CHECK_CUDA(cudaMemcpy(&dw,
                                              (const StridedPtrH *)ranks[rank].down.d_w_table + local,
                                              sizeof(dw), cudaMemcpyDeviceToHost));
                    }
                    if (ranks[rank].down.d_s_table) {
                        CHECK_CUDA(cudaMemcpy(&ds,
                                              (const StridedPtrH *)ranks[rank].down.d_s_table + local,
                                              sizeof(ds), cudaMemcpyDeviceToHost));
                    }
                }
                std::fprintf(stderr,
                             "tp_ep_model_router_route_id\tlayer\t%d\tslot\t%d\tk\t%d\texpert\t%d\trank\t%d\tlocal\t%d\tweight\t%.9g\tgated_w\t%p\tgated_ws\t%d\tgated_s\t%p\tgated_ss\t%d\tdown_w\t%p\tdown_ws\t%d\tdown_s\t%p\tdown_ss\t%d\n",
                             opt.layer, slot, k, expert, rank, local, w,
                             gw.p, gw.stride, gs.p, gs.stride,
                             dw.p, dw.stride, ds.p, ds.stride);
            }
        }
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        offsets[rank].assign((size_t)kLocalExperts + 1, 0);
        int running = 0;
        int active = 0;
        int max_routes = 0;
        for (int e = 0; e < kLocalExperts; ++e) {
            offsets[rank][(size_t)e] = running;
            running += counts[rank][(size_t)e];
            if (counts[rank][(size_t)e] > 0) ++active;
            max_routes = std::max(max_routes, counts[rank][(size_t)e]);
        }
        offsets[rank][(size_t)kLocalExperts] = running;
        if (running > ranks[rank].route_capacity) return 4;
        route_slots[rank].assign((size_t)running, -1);
        route_weights[rank].assign((size_t)running, 0.0f);
        std::vector<int> cursor = offsets[rank];
        for (int slot = 0; slot < opt.slots; ++slot) {
            for (int k = 0; k < opt.top_k; ++k) {
                const int expert = selected[(size_t)slot * opt.top_k + (size_t)k];
                if (expert < 0) continue;
                const int dst_rank = expert / kLocalExperts;
                if (dst_rank != rank) continue;
                const int local = expert % kLocalExperts;
                const int idx = cursor[(size_t)local]++;
                route_slots[rank][(size_t)idx] = slot;
                const float w = weights[(size_t)slot * opt.top_k + (size_t)k];
                if (!std::isfinite(w)) {
                    std::fprintf(stderr,
                                 "tp_ep_model_router_nonfinite_weight\trank\t%d\tslot\t%d\texpert\t%d\tk\t%d\n",
                                 rank, slot, expert, k);
                    return 5;
                }
                route_weights[rank][(size_t)idx] = w;
                if (needs_single_route_index &&
                    route_index_by_slot[rank][(size_t)slot] < 0) {
                    route_index_by_slot[rank][(size_t)slot] = idx;
                }
                int &route_count = route_count_by_slot[rank][(size_t)slot];
                if (route_count >= opt.top_k) return 6;
                route_indices_by_slot[rank][(size_t)slot * (size_t)opt.top_k +
                                            (size_t)route_count] = idx;
                route_count++;
            }
        }
        RankState &r = ranks[rank];
        r.routes = running;
        r.active_experts = active;
        r.max_routes_per_expert = max_routes;
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaMemcpy(r.d_offsets, offsets[rank].data(),
                              offsets[rank].size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(r.d_route_slots, route_slots[rank].data(),
                              route_slots[rank].size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(r.d_route_weights, route_weights[rank].data(),
                              route_weights[rank].size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    static bool route_stats_emitted[43] = {};
    if (opt.model_router_routes && opt.layer >= 0 && opt.layer <= 5 &&
        !route_stats_emitted[opt.layer]) {
        route_stats_emitted[opt.layer] = true;
        std::fprintf(stderr,
                     "tp_ep_model_router_route_stats\tlayer\t%d\troutes\t%d,%d,%d,%d,%d,%d,%d,%d\tmax_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                     opt.layer,
                     ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                     ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                     ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                     ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                     ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                     ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
    }
    if (opt.compact_moe_decode_gate && opt.model_router_routes) {
        int duplicate_slots = 0;
        int max_same_rank_routes = 0;
        for (int slot = 0; slot < opt.slots; ++slot) {
            for (int rank = 0; rank < kGpus; ++rank) {
                const int c = route_count_by_slot[rank][(size_t)slot];
                max_same_rank_routes = std::max(max_same_rank_routes, c);
                if (c > 1) duplicate_slots++;
            }
        }
        static bool compact_stats_emitted[43] = {};
        if (opt.layer >= 0 && opt.layer < 43 && !compact_stats_emitted[opt.layer]) {
            compact_stats_emitted[opt.layer] = true;
            const uint64_t all_dest_bytes =
                (uint64_t)kGpus * (uint64_t)kGpus * (uint64_t)opt.slots *
                (uint64_t)(kHidden / kGpus) * sizeof(float);
            const uint64_t total_routes =
                (uint64_t)ranks[0].routes + (uint64_t)ranks[1].routes +
                (uint64_t)ranks[2].routes + (uint64_t)ranks[3].routes +
                (uint64_t)ranks[4].routes + (uint64_t)ranks[5].routes +
                (uint64_t)ranks[6].routes + (uint64_t)ranks[7].routes;
            const uint64_t compact_bytes =
                (uint64_t)kGpus * total_routes * (uint64_t)(kHidden / kGpus) *
                sizeof(float);
            std::printf("tp_ep_compact_moe_route_stats\tlayer\t%d\t"
                        "duplicate_slots\t%d\tmax_same_rank_routes\t%d\t"
                        "all_dest_bytes\t%llu\tcompact_bytes\t%llu\t"
                        "routes\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                        "active_experts\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                        "max_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                        opt.layer, duplicate_slots, max_same_rank_routes,
                        (unsigned long long)all_dest_bytes,
                        (unsigned long long)compact_bytes,
                        ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                        ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                        ranks[0].active_experts, ranks[1].active_experts,
                        ranks[2].active_experts, ranks[3].active_experts,
                        ranks[4].active_experts, ranks[5].active_experts,
                        ranks[6].active_experts, ranks[7].active_experts,
                        ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                        ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                        ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                        ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
        }
    }
    std::vector<int> compact_plan;
    if (needs_packed_compact_plan) {
        compact_plan.assign(compact_route_plan_ints(opt), -1);
        const size_t compact_indices = (size_t)opt.slots * (size_t)opt.top_k;
        const size_t compact_counts = (size_t)opt.slots;
        for (int src = 0; src < kGpus; ++src) {
            std::copy(route_indices_by_slot[src].begin(),
                      route_indices_by_slot[src].end(),
                      compact_plan.begin() + (size_t)src * compact_indices);
            std::copy(route_count_by_slot[src].begin(),
                      route_count_by_slot[src].end(),
                      compact_plan.begin() + (size_t)kGpus * compact_indices +
                          (size_t)src * compact_counts);
        }
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        CHECK_CUDA(cudaSetDevice(ranks[dst].device));
        for (int src = 0; src < kGpus; ++src) {
            if (needs_single_route_index) {
                CHECK_CUDA(cudaMemcpy(ranks[dst].d_route_index_by_slot[src],
                                      route_index_by_slot[src].data(),
                                      route_index_by_slot[src].size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
            }
            if (!needs_packed_compact_plan) {
                CHECK_CUDA(cudaMemcpy(ranks[dst].d_route_indices_by_slot[src],
                                      route_indices_by_slot[src].data(),
                                      route_indices_by_slot[src].size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(ranks[dst].d_route_count_by_slot[src],
                                      route_count_by_slot[src].data(),
                                      route_count_by_slot[src].size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
            }
        }
        if (needs_packed_compact_plan) {
            if (!ranks[dst].d_route_compact_plan ||
                ranks[dst].route_compact_plan_ints < compact_plan.size()) {
                return 7;
            }
            CHECK_CUDA(cudaMemcpy(ranks[dst].d_route_compact_plan,
                                  compact_plan.data(),
                                  compact_plan.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }
    }
    return 0;
}

int upload_model_router_route_plan_async(const Options &opt,
                                         RankState ranks[kGpus],
                                         const int *selected,
                                         const float *weights,
                                         RoutePlanHostWorkspace *ws) {
    if (!selected || !weights || !ws || !ws->initialized ||
        ws->slots != opt.slots || ws->top_k != opt.top_k ||
        ws->route_capacity < (size_t)opt.slots * (size_t)opt.top_k) {
        return 1;
    }
    if (opt.routed_ffn_norm_input_gate) {
        return 8;
    }
    if (ws->uploads_pending) {
        for (int rank = 0; rank < kGpus; ++rank) {
            if (ws->upload_done[rank]) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaEventSynchronize(ws->upload_done[rank]));
            }
        }
        ws->uploads_pending = false;
    }

    const bool needs_single_route_index = !opt.compact_moe_decode_gate;
    const bool needs_packed_compact_plan = opt.compact_moe_decode_gate;
    std::vector<int> counts[kGpus];
    std::vector<int> cursor[kGpus];
    for (int rank = 0; rank < kGpus; ++rank) {
        counts[rank].assign((size_t)kLocalExperts, 0);
        std::fill(ws->h_route_indices_by_slot[rank],
                  ws->h_route_indices_by_slot[rank] +
                      (size_t)opt.slots * (size_t)opt.top_k,
                  -1);
        std::fill(ws->h_route_count_by_slot[rank],
                  ws->h_route_count_by_slot[rank] + (size_t)opt.slots,
                  0);
        if (needs_single_route_index) {
            std::fill(ws->h_route_index_by_slot[rank],
                      ws->h_route_index_by_slot[rank] + (size_t)opt.slots,
                      -1);
        }
    }

    bool compact_duplicate = false;
    for (int slot = 0; slot < opt.slots; ++slot) {
        bool seen_rank[kGpus] = {};
        for (int k = 0; k < opt.top_k; ++k) {
            const int expert = selected[(size_t)slot * (size_t)opt.top_k + (size_t)k];
            if (expert < 0) continue;
            if (expert >= kGlobalExperts) return 2;
            const int rank = expert / kLocalExperts;
            const int local = expert % kLocalExperts;
            counts[rank][(size_t)local]++;
            if (seen_rank[rank]) compact_duplicate = true;
            seen_rank[rank] = true;
        }
    }
    if (opt.compact_route_compose && compact_duplicate &&
        !opt.compact_moe_decode_gate) {
        return 3;
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        int running = 0;
        int active = 0;
        int max_routes = 0;
        for (int e = 0; e < kLocalExperts; ++e) {
            ws->h_offsets[rank][e] = running;
            running += counts[rank][(size_t)e];
            if (counts[rank][(size_t)e] > 0) ++active;
            max_routes = std::max(max_routes, counts[rank][(size_t)e]);
        }
        ws->h_offsets[rank][kLocalExperts] = running;
        if (running > ranks[rank].route_capacity ||
            (size_t)running > ws->route_capacity) {
            return 4;
        }
        std::fill(ws->h_route_slots[rank],
                  ws->h_route_slots[rank] + (size_t)running, -1);
        std::fill(ws->h_route_weights[rank],
                  ws->h_route_weights[rank] + (size_t)running, 0.0f);
        cursor[rank].assign(ws->h_offsets[rank],
                            ws->h_offsets[rank] + kLocalExperts + 1);
    }

    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int k = 0; k < opt.top_k; ++k) {
            const size_t route_key = (size_t)slot * (size_t)opt.top_k + (size_t)k;
            const int expert = selected[route_key];
            if (expert < 0) continue;
            const int rank = expert / kLocalExperts;
            const int local = expert % kLocalExperts;
            const int idx = cursor[rank][(size_t)local]++;
            ws->h_route_slots[rank][idx] = slot;
            const float w = weights[route_key];
            if (!std::isfinite(w)) {
                std::fprintf(stderr,
                             "tp_ep_model_router_nonfinite_weight\trank\t%d\tslot\t%d\texpert\t%d\tk\t%d\n",
                             rank, slot, expert, k);
                return 5;
            }
            ws->h_route_weights[rank][idx] = w;
            if (needs_single_route_index &&
                ws->h_route_index_by_slot[rank][slot] < 0) {
                ws->h_route_index_by_slot[rank][slot] = idx;
            }
            int &route_count = ws->h_route_count_by_slot[rank][slot];
            if (route_count >= opt.top_k) return 6;
            ws->h_route_indices_by_slot[rank]
                [(size_t)slot * (size_t)opt.top_k + (size_t)route_count] = idx;
            route_count++;
        }
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        const int running = ws->h_offsets[rank][kLocalExperts];
        int active = 0;
        int max_routes = 0;
        for (int e = 0; e < kLocalExperts; ++e) {
            if (counts[rank][(size_t)e] > 0) ++active;
            max_routes = std::max(max_routes, counts[rank][(size_t)e]);
        }
        r.routes = running;
        r.active_experts = active;
        r.max_routes_per_expert = max_routes;
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaMemcpyAsync(r.d_offsets, ws->h_offsets[rank],
                                   (size_t)(kLocalExperts + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, r.stream));
        if (running > 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_route_slots, ws->h_route_slots[rank],
                                       (size_t)running * sizeof(int),
                                       cudaMemcpyHostToDevice, r.stream));
            CHECK_CUDA(cudaMemcpyAsync(r.d_route_weights, ws->h_route_weights[rank],
                                       (size_t)running * sizeof(float),
                                       cudaMemcpyHostToDevice, r.stream));
        }
    }

    static bool route_stats_emitted[43] = {};
    if (opt.model_router_routes && opt.layer >= 0 && opt.layer <= 5 &&
        !route_stats_emitted[opt.layer]) {
        route_stats_emitted[opt.layer] = true;
        std::fprintf(stderr,
                     "tp_ep_model_router_route_stats_async\tlayer\t%d\troutes\t%d,%d,%d,%d,%d,%d,%d,%d\tmax_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                     opt.layer,
                     ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                     ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                     ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                     ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                     ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                     ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
    }
    if (opt.compact_moe_decode_gate && opt.model_router_routes) {
        int duplicate_slots = 0;
        int max_same_rank_routes = 0;
        for (int slot = 0; slot < opt.slots; ++slot) {
            for (int rank = 0; rank < kGpus; ++rank) {
                const int c = ws->h_route_count_by_slot[rank][slot];
                max_same_rank_routes = std::max(max_same_rank_routes, c);
                if (c > 1) duplicate_slots++;
            }
        }
        static bool compact_stats_emitted[43] = {};
        if (opt.layer >= 0 && opt.layer < 43 && !compact_stats_emitted[opt.layer]) {
            compact_stats_emitted[opt.layer] = true;
            const uint64_t all_dest_bytes =
                (uint64_t)kGpus * (uint64_t)kGpus * (uint64_t)opt.slots *
                (uint64_t)(kHidden / kGpus) * sizeof(float);
            const uint64_t total_routes =
                (uint64_t)ranks[0].routes + (uint64_t)ranks[1].routes +
                (uint64_t)ranks[2].routes + (uint64_t)ranks[3].routes +
                (uint64_t)ranks[4].routes + (uint64_t)ranks[5].routes +
                (uint64_t)ranks[6].routes + (uint64_t)ranks[7].routes;
            const uint64_t compact_bytes =
                (uint64_t)kGpus * total_routes * (uint64_t)(kHidden / kGpus) *
                sizeof(float);
            std::printf("tp_ep_compact_moe_route_stats_async\tlayer\t%d\t"
                        "duplicate_slots\t%d\tmax_same_rank_routes\t%d\t"
                        "all_dest_bytes\t%llu\tcompact_bytes\t%llu\t"
                        "routes\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                        "active_experts\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                        "max_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                        opt.layer, duplicate_slots, max_same_rank_routes,
                        (unsigned long long)all_dest_bytes,
                        (unsigned long long)compact_bytes,
                        ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                        ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                        ranks[0].active_experts, ranks[1].active_experts,
                        ranks[2].active_experts, ranks[3].active_experts,
                        ranks[4].active_experts, ranks[5].active_experts,
                        ranks[6].active_experts, ranks[7].active_experts,
                        ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                        ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                        ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                        ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
        }
    }

    if (needs_packed_compact_plan) {
        if (ws->compact_plan_ints <
            (size_t)kGpus * ((size_t)opt.slots * (size_t)opt.top_k +
                             (size_t)opt.slots)) {
            return 7;
        }
        std::fill(ws->h_compact_plan,
                  ws->h_compact_plan + ws->compact_plan_ints, -1);
        const size_t compact_indices = (size_t)opt.slots * (size_t)opt.top_k;
        const size_t compact_counts = (size_t)opt.slots;
        for (int src = 0; src < kGpus; ++src) {
            std::memcpy(ws->h_compact_plan + (size_t)src * compact_indices,
                        ws->h_route_indices_by_slot[src],
                        compact_indices * sizeof(int));
            std::memcpy(ws->h_compact_plan + (size_t)kGpus * compact_indices +
                            (size_t)src * compact_counts,
                        ws->h_route_count_by_slot[src],
                        compact_counts * sizeof(int));
        }
    }

    for (int dst = 0; dst < kGpus; ++dst) {
        CHECK_CUDA(cudaSetDevice(ranks[dst].device));
        for (int src = 0; src < kGpus; ++src) {
            if (needs_single_route_index) {
                CHECK_CUDA(cudaMemcpyAsync(ranks[dst].d_route_index_by_slot[src],
                                           ws->h_route_index_by_slot[src],
                                           (size_t)opt.slots * sizeof(int),
                                           cudaMemcpyHostToDevice,
                                           ranks[dst].stream));
            }
            if (!needs_packed_compact_plan) {
                CHECK_CUDA(cudaMemcpyAsync(ranks[dst].d_route_indices_by_slot[src],
                                           ws->h_route_indices_by_slot[src],
                                           (size_t)opt.slots *
                                               (size_t)opt.top_k * sizeof(int),
                                           cudaMemcpyHostToDevice,
                                           ranks[dst].stream));
                CHECK_CUDA(cudaMemcpyAsync(ranks[dst].d_route_count_by_slot[src],
                                           ws->h_route_count_by_slot[src],
                                           (size_t)opt.slots * sizeof(int),
                                           cudaMemcpyHostToDevice,
                                           ranks[dst].stream));
            }
        }
        if (needs_packed_compact_plan) {
            if (!ranks[dst].d_route_compact_plan ||
                ranks[dst].route_compact_plan_ints < ws->compact_plan_ints) {
                return 7;
            }
            CHECK_CUDA(cudaMemcpyAsync(ranks[dst].d_route_compact_plan,
                                       ws->h_compact_plan,
                                       ws->compact_plan_ints * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       ranks[dst].stream));
        }
        CHECK_CUDA(cudaEventRecord(ws->upload_done[dst], ranks[dst].stream));
    }
    ws->uploads_pending = true;
    return 0;
}

int open_compose_nccl(const Options &opt, RankState ranks[kGpus]);
void close_compose_nccl(RankState ranks[kGpus]);

int open_shared_rank_buffers(const Options &opt, SharedRankBuffers *shared) {
    shared->core_bytes = 0;
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = shared->ranks[p];
        r.rank = p;
        r.device = opt.devices[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaStreamCreate(&r.stream));
        CHECK_CUDA(cudaStreamCreate(&r.dense_stream));
        CHECK_CUDA(cudaStreamCreate(&r.copy_stream));
        for (int q = 0; q < kGpus; ++q) {
            CHECK_CUDA(cudaStreamCreate(&r.copy_streams[q]));
            CHECK_CUDA(cudaEventCreateWithFlags(&r.copy_done[q], cudaEventDisableTiming));
        }
        CHECK_CUDA(cudaEventCreateWithFlags(&r.stream_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&r.dense_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&r.dense_wait, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreate(&r.start));
        CHECK_CUDA(cudaEventCreate(&r.mid));
        CHECK_CUDA(cudaEventCreate(&r.stop));
        r.route_compact_plan_ints = compact_route_plan_ints(opt);
        CHECK_CUDA(cudaMalloc(&r.d_route_compact_plan,
                              r.route_compact_plan_ints * sizeof(int)));
        bind_compact_route_plan(&r, opt);
        CHECK_CUDA(cudaMalloc(&r.d_router_selected_plan,
                              (size_t)opt.slots * (size_t)opt.top_k * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&r.d_router_weights_plan,
                              (size_t)opt.slots * (size_t)opt.top_k * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&r.d_route_offsets_all,
                              (size_t)kGpus * (size_t)(kLocalExperts + 1) *
                                  sizeof(int)));
        CHECK_CUDA(cudaMalloc(&r.d_route_totals,
                              (size_t)kGpus * sizeof(int)));
        std::vector<int> compact_plan(r.route_compact_plan_ints, -1);
        const size_t compact_indices = (size_t)opt.slots * (size_t)opt.top_k;
        const size_t compact_counts = (size_t)opt.slots;
        for (int src = 0; src < kGpus; ++src) {
            std::vector<int> route_index_by_slot;
            build_route_index_by_slot_for_rank(src, opt.slots, opt.top_k,
                                               &route_index_by_slot);
            CHECK_CUDA(cudaMalloc(&r.d_route_index_by_slot[src],
                                  route_index_by_slot.size() * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(r.d_route_index_by_slot[src],
                                  route_index_by_slot.data(),
                                  route_index_by_slot.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            std::vector<int> route_indices_by_slot;
            std::vector<int> route_count_by_slot;
            build_route_indices_by_slot_for_rank(src, opt.slots, opt.top_k,
                                                 &route_indices_by_slot,
                                                 &route_count_by_slot);
            std::copy(route_indices_by_slot.begin(), route_indices_by_slot.end(),
                      compact_plan.begin() + (size_t)src * compact_indices);
            std::copy(route_count_by_slot.begin(), route_count_by_slot.end(),
                      compact_plan.begin() + (size_t)kGpus * compact_indices +
                          (size_t)src * compact_counts);
            shared->core_bytes += route_index_by_slot.size() * sizeof(int);
        }
        CHECK_CUDA(cudaMemcpy(r.d_route_compact_plan, compact_plan.data(),
                              compact_plan.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        shared->core_bytes += compact_plan.size() * sizeof(int);

        std::vector<int> offsets;
        std::vector<int> route_slots;
        std::vector<float> route_weights;
        build_offsets_for_rank(p, opt.slots, opt.top_k, &offsets, &route_slots,
                               &route_weights, &r.routes, &r.active_experts,
                               &r.max_routes_per_expert);

        r.route_capacity = opt.slots * opt.top_k;
        const size_t route_capacity_elems = (size_t)r.route_capacity * kHidden;
        CHECK_CUDA(cudaMalloc(&r.d_offsets, offsets.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(r.d_offsets, offsets.data(), offsets.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_route_slots,
                              (size_t)r.route_capacity * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(r.d_route_slots, route_slots.data(),
                              route_slots.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_route_weights,
                              (size_t)r.route_capacity * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(r.d_route_weights, route_weights.data(),
                              route_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_route_inv_scale,
                              (size_t)r.route_capacity * sizeof(float)));
        std::vector<float> route_inv_scale((size_t)r.route_capacity, 1.0f);
        CHECK_CUDA(cudaMemcpy(r.d_route_inv_scale, route_inv_scale.data(),
                              route_inv_scale.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_a, route_capacity_elems * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&r.d_gate_up,
                              (size_t)r.route_capacity * kFusedN * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&r.d_gated,
                              (size_t)r.route_capacity * kMid * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&r.d_down, route_capacity_elems * sizeof(__half)));

        std::mt19937 rng(0xE2350000u + (uint32_t)p * 97u);
        std::uniform_real_distribution<float> dist(-0.003f, 0.003f);
        std::vector<__half> h_a(route_capacity_elems);
        for (__half &v : h_a) v = __float2half(dist(rng));
        CHECK_CUDA(cudaMemcpy(r.d_a, h_a.data(),
                              route_capacity_elems * sizeof(__half),
                              cudaMemcpyHostToDevice));

        shared->core_bytes += offsets.size() * sizeof(int);
        shared->core_bytes += (size_t)r.route_capacity * sizeof(int);
        shared->core_bytes += (size_t)r.route_capacity * sizeof(float);
        shared->core_bytes += (size_t)r.route_capacity * sizeof(float);
        shared->core_bytes += route_capacity_elems * sizeof(__half);
        shared->core_bytes += (size_t)r.route_capacity * kFusedN * sizeof(__half);
        shared->core_bytes += (size_t)r.route_capacity * kMid * sizeof(__half);
        shared->core_bytes += route_capacity_elems * sizeof(__half);
    }
    if (open_compose_nccl(opt, shared->ranks) != 0) {
        return 1;
    }
    shared->initialized = true;
    return 0;
}

int open_compose_nccl(const Options &opt, RankState ranks[kGpus]) {
    const bool need_compose =
        opt.nccl_reduce_scatter_compose_gate &&
        !opt.compact_route_compose && !opt.ep_return_fp16;
    const bool need_attention_output =
        opt.true_ds4_attention_output_nccl_allgather_gate;
    const bool need_hc_current =
        opt.tp_hc_current_input_nccl_allgather_gate;
    if (!need_compose && !need_attention_output && !need_hc_current) return 0;
    int devices[kGpus] = {};
    ncclComm_t comms[kGpus] = {};
    for (int p = 0; p < kGpus; ++p) devices[p] = ranks[p].device;
    CHECK_NCCL(ncclCommInitAll(comms, kGpus, devices));
    for (int p = 0; p < kGpus; ++p) {
        ranks[p].compose_nccl = comms[p];
        ranks[p].compose_nccl_initialized = true;
    }
    std::printf("tp_ep_nccl\tdevices\t%d\tcompose_reduce_scatter\t%d\t"
                "attention_output_allgather\t%d\t"
                "hc_current_allgather\t%d\tPASS\n",
                kGpus, need_compose ? 1 : 0, need_attention_output ? 1 : 0,
                need_hc_current ? 1 : 0);
    return 0;
}

void close_compose_nccl(RankState ranks[kGpus]) {
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        if (!r.compose_nccl_initialized || !r.compose_nccl) continue;
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclCommDestroy(r.compose_nccl));
        r.compose_nccl = nullptr;
        r.compose_nccl_initialized = false;
    }
}

void close_shared_rank_buffers(SharedRankBuffers *shared) {
    if (!shared || !shared->initialized) return;
    close_compose_nccl(shared->ranks);
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = shared->ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        free_packed(r.gated);
        free_packed(r.down);
        if (r.d_offsets) CHECK_CUDA(cudaFree(r.d_offsets));
        if (r.d_route_slots) CHECK_CUDA(cudaFree(r.d_route_slots));
        if (r.d_route_weights) CHECK_CUDA(cudaFree(r.d_route_weights));
        if (r.d_route_inv_scale) CHECK_CUDA(cudaFree(r.d_route_inv_scale));
        if (r.d_a) CHECK_CUDA(cudaFree(r.d_a));
        if (r.d_gate_up) CHECK_CUDA(cudaFree(r.d_gate_up));
        if (r.d_gated) CHECK_CUDA(cudaFree(r.d_gated));
        if (r.d_down) CHECK_CUDA(cudaFree(r.d_down));
        if (r.d_ep_contrib_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_all));
        if (r.d_ep_contrib_half_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_half_all));
        if (r.d_route_totals) CHECK_CUDA(cudaFree(r.d_route_totals));
        if (r.d_route_offsets_all) CHECK_CUDA(cudaFree(r.d_route_offsets_all));
        if (r.d_router_weights_plan) CHECK_CUDA(cudaFree(r.d_router_weights_plan));
        if (r.d_router_selected_plan) CHECK_CUDA(cudaFree(r.d_router_selected_plan));
        const bool has_route_compact_plan = r.d_route_compact_plan != nullptr;
        if (r.d_route_compact_plan) CHECK_CUDA(cudaFree(r.d_route_compact_plan));
        for (int src = 0; src < kGpus; ++src) {
            if (r.d_route_index_by_slot[src]) CHECK_CUDA(cudaFree(r.d_route_index_by_slot[src]));
            if (!has_route_compact_plan && r.d_route_indices_by_slot[src]) {
                CHECK_CUDA(cudaFree(r.d_route_indices_by_slot[src]));
            }
            if (!has_route_compact_plan && r.d_route_count_by_slot[src]) {
                CHECK_CUDA(cudaFree(r.d_route_count_by_slot[src]));
            }
            if (r.d_ep_remote[src]) CHECK_CUDA(cudaFree(r.d_ep_remote[src]));
            if (r.d_ep_remote_half[src]) CHECK_CUDA(cudaFree(r.d_ep_remote_half[src]));
        }
        if (r.d_ep_sum) CHECK_CUDA(cudaFree(r.d_ep_sum));
        if (r.d_next_hidden) CHECK_CUDA(cudaFree(r.d_next_hidden));
        if (r.d_current_shard) CHECK_CUDA(cudaFree(r.d_current_shard));
        if (r.d_current_full) CHECK_CUDA(cudaFree(r.d_current_full));
        if (r.d_current_full_rank_major) CHECK_CUDA(cudaFree(r.d_current_full_rank_major));
        if (r.d_final_hc_shard) CHECK_CUDA(cudaFree(r.d_final_hc_shard));
        if (r.d_hc_scratch_shard) CHECK_CUDA(cudaFree(r.d_hc_scratch_shard));
        if (r.d_hc_split) CHECK_CUDA(cudaFree(r.d_hc_split));
        for (int layer = 0; layer < 43; ++layer) {
            if (r.d_attn_raw_swa_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_attn_raw_swa_layers[layer]));
            }
        }
        if (r.d_attn_kv_full) CHECK_CUDA(cudaFree(r.d_attn_kv_full));
        if (r.d_attn_heads) CHECK_CUDA(cudaFree(r.d_attn_heads));
        if (r.d_attn_output_a_full) CHECK_CUDA(cudaFree(r.d_attn_output_a_full));
        if (r.d_post_attn_shard) CHECK_CUDA(cudaFree(r.d_post_attn_shard));
        if (r.d_attn_sinks) CHECK_CUDA(cudaFree(r.d_attn_sinks));
        if (r.d_indexer_topk) CHECK_CUDA(cudaFree(r.d_indexer_topk));
        if (r.d_indexer_scores) CHECK_CUDA(cudaFree(r.d_indexer_scores));
        for (int layer = 0; layer < 43; ++layer) {
            if (r.d_index_comp_rows_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_index_comp_rows_layers[layer]));
            }
            if (r.d_index_comp_state_score_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_index_comp_state_score_layers[layer]));
            }
            if (r.d_index_comp_state_kv_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_index_comp_state_kv_layers[layer]));
            }
        }
        if (r.d_index_comp_score_cur) CHECK_CUDA(cudaFree(r.d_index_comp_score_cur));
        if (r.d_index_comp_kv_cur) CHECK_CUDA(cudaFree(r.d_index_comp_kv_cur));
        for (int layer = 0; layer < 43; ++layer) {
            if (r.d_attn_comp_rows_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_attn_comp_rows_layers[layer]));
            }
            if (r.d_attn_comp_state_score_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_attn_comp_state_score_layers[layer]));
            }
            if (r.d_attn_comp_state_kv_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_attn_comp_state_kv_layers[layer]));
            }
        }
        if (r.d_attn_comp_score_cur) CHECK_CUDA(cudaFree(r.d_attn_comp_score_cur));
        if (r.d_attn_comp_kv_cur) CHECK_CUDA(cudaFree(r.d_attn_comp_kv_cur));
        if (r.dense_wait) CHECK_CUDA(cudaEventDestroy(r.dense_wait));
        if (r.start) CHECK_CUDA(cudaEventDestroy(r.start));
        if (r.mid) CHECK_CUDA(cudaEventDestroy(r.mid));
        if (r.stop) CHECK_CUDA(cudaEventDestroy(r.stop));
        for (int q = 0; q < kGpus; ++q) {
            if (r.copy_done[q]) CHECK_CUDA(cudaEventDestroy(r.copy_done[q]));
            if (r.copy_streams[q]) CHECK_CUDA(cudaStreamDestroy(r.copy_streams[q]));
        }
        if (r.dense_done) CHECK_CUDA(cudaEventDestroy(r.dense_done));
        if (r.stream_done) CHECK_CUDA(cudaEventDestroy(r.stream_done));
        if (r.copy_stream) CHECK_CUDA(cudaStreamDestroy(r.copy_stream));
        if (r.dense_stream) CHECK_CUDA(cudaStreamDestroy(r.dense_stream));
        if (r.stream) CHECK_CUDA(cudaStreamDestroy(r.stream));
        r = RankState{};
    }
    *shared = SharedRankBuffers{};
}

void fill_tp_runtime_config(const Options &opt, ds4_v100_tp_runtime_config *cfg) {
    ds4_v100_tp_runtime_default_config(cfg);
    cfg->slots = (uint32_t)opt.slots;
    cfg->ctx = 262144;
    cfg->kv_dtype = opt.fp8_e5m2_kv_gate
        ? DS4_V100_TP_KV_F8_E5M2_B128
        : DS4_V100_TP_KV_F8_E4M3_B128;
    cfg->scratch_bytes = 1536ull * 1024ull * 1024ull;
    cfg->allocate_comp_state = opt.tp_runtime_skip_unused_comp_state ? 0u : 1u;
    for (int i = 0; i < kGpus; ++i) cfg->devices[i] = opt.devices[i];
}

int open_shared_tp_runtime(const Options &opt, SharedTpRuntime *shared) {
    ds4_v100_tp_runtime_config cfg;
    fill_tp_runtime_config(opt, &cfg);
    char err[512] = {0};
    if (ds4_v100_tp_runtime_open(&shared->rt, &cfg, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "tp_runtime_open_failed\t%s\n", err);
        *shared = SharedTpRuntime{};
        return 1;
    }
    ds4_v100_tp_runtime_get_report(shared->rt, &shared->report);
    shared->initialized = true;
    return 0;
}

void close_shared_tp_runtime(SharedTpRuntime *shared) {
    if (!shared || !shared->rt) return;
    ds4_v100_tp_runtime_close(shared->rt);
    *shared = SharedTpRuntime{};
}

int ensure_compose_buffers(const Options &opt, RankState ranks[kGpus]) {
    const uint64_t shard_elems = (uint64_t)opt.slots * (kHidden / kGpus);
    const uint64_t compact_segment_routes =
        opt.compact_moe_decode_gate ? (uint64_t)opt.slots * (uint64_t)opt.top_k
                                    : (uint64_t)opt.slots;
    const uint64_t shard_bytes = shard_elems * sizeof(float);
    const uint64_t remote_float_elems =
        opt.compact_route_compose && !opt.ep_return_fp16
            ? compact_segment_routes * (uint64_t)(kHidden / kGpus)
            : shard_elems;
    const uint64_t remote_float_bytes = remote_float_elems * sizeof(float);
    const uint64_t all_contrib_elems =
        (uint64_t)kGpus * compact_segment_routes * (uint64_t)(kHidden / kGpus);
    const uint64_t all_contrib_bytes = all_contrib_elems * sizeof(float);
    const int layer = opt.layer;
    if ((opt.true_ds4_attention_state_gate || opt.true_ds4_compressed_kv_gate ||
         opt.true_ds4_indexer_attention_gate) &&
        (layer < 0 || layer >= 43)) {
        return 20;
    }
    const int ratio = (layer >= 0 && layer < 43) ? ds4_layer_ratio(layer) : 0;
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_ep_contrib_all) CHECK_CUDA(cudaMalloc(&r.d_ep_contrib_all,
                                                        (size_t)all_contrib_bytes));
        if (opt.ep_return_fp16 && !r.d_ep_contrib_half_all) {
            CHECK_CUDA(cudaMalloc(&r.d_ep_contrib_half_all,
                                  (size_t)(all_contrib_elems * sizeof(__half))));
        }
        if (!r.d_ep_sum) CHECK_CUDA(cudaMalloc(&r.d_ep_sum, (size_t)shard_bytes));
        if (!r.d_next_hidden) CHECK_CUDA(cudaMalloc(&r.d_next_hidden, (size_t)shard_bytes));
        if (opt.tp_hc_current_input_gate && !r.d_current_shard) {
            CHECK_CUDA(cudaMalloc(&r.d_current_shard, (size_t)shard_bytes));
        }
        if (opt.tp_hc_current_input_gate && !r.d_current_full) {
            CHECK_CUDA(cudaMalloc(&r.d_current_full,
                                  (size_t)opt.slots * kHidden * sizeof(float)));
        }
        if (opt.tp_hc_current_input_nccl_allgather_gate &&
            !r.d_current_full_rank_major) {
            CHECK_CUDA(cudaMalloc(&r.d_current_full_rank_major,
                                  (size_t)opt.slots * kHidden * sizeof(float)));
        }
        if (opt.final_hc_carry_gate && !r.d_final_hc_shard) {
            CHECK_CUDA(cudaMalloc(&r.d_final_hc_shard, (size_t)(4ull * shard_bytes)));
        }
        if (opt.tp_hc_final_expand_gate && !r.d_hc_scratch_shard) {
            CHECK_CUDA(cudaMalloc(&r.d_hc_scratch_shard, (size_t)(4ull * shard_bytes)));
        }
        if (opt.tp_hc_final_expand_gate && !r.d_hc_split) {
            CHECK_CUDA(cudaMalloc(&r.d_hc_split, (size_t)opt.slots * kHcMix * sizeof(float)));
        }
        if (opt.true_ds4_attention_state_gate && !r.d_attn_kv_full) {
            CHECK_CUDA(cudaMalloc(&r.d_attn_kv_full,
                                  (size_t)opt.slots * kHeadDim * sizeof(float)));
        }
        if (opt.true_ds4_attention_state_gate) {
            if (!r.d_attn_raw_swa_layers[layer]) {
                CHECK_CUDA(cudaMalloc(&r.d_attn_raw_swa_layers[layer],
                                      (size_t)opt.slots * kRawSwaRows *
                                          (size_t)kHeadDim * sizeof(float)));
                CHECK_CUDA(cudaMemsetAsync(r.d_attn_raw_swa_layers[layer], 0,
                                           (size_t)opt.slots * kRawSwaRows *
                                               (size_t)kHeadDim * sizeof(float),
                                           r.stream));
            }
            r.d_attn_raw_swa = r.d_attn_raw_swa_layers[layer];
        }
        if (opt.true_ds4_attention_raw_read_gate && !r.d_attn_sinks) {
            CHECK_CUDA(cudaMalloc(&r.d_attn_sinks,
                                  (size_t)kLocalHeads * sizeof(float)));
        }
        if (opt.true_ds4_attention_raw_read_gate && !r.d_attn_heads) {
            CHECK_CUDA(cudaMalloc(&r.d_attn_heads,
                                  (size_t)opt.slots * kLocalHeads *
                                      (size_t)kHeadDim * sizeof(float)));
        }
        if (opt.true_ds4_attention_output_gate && !r.d_attn_output_a_full) {
            CHECK_CUDA(cudaMalloc(&r.d_attn_output_a_full,
                                  (size_t)opt.slots *
                                      (size_t)kAttentionOutputAFull * sizeof(float)));
        }
        if (opt.true_ds4_post_attention_ffn_input_gate && !r.d_post_attn_shard) {
            CHECK_CUDA(cudaMalloc(&r.d_post_attn_shard, (size_t)shard_bytes));
        }
        if (opt.true_ds4_compressed_kv_gate && ratio != 0) {
            const int comp_state_rows = attn_comp_state_rows_for_ratio(ratio);
            const int comp_state_width = attn_comp_state_width_for_ratio(ratio);
            if (!r.d_attn_comp_kv_cur) {
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_kv_cur,
                                      (size_t)opt.slots * kCompWidthMax * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_score_cur,
                                      (size_t)opt.slots * kCompWidthMax * sizeof(float)));
            }
            if (!r.d_attn_comp_state_kv_layers[layer]) {
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_state_kv_layers[layer],
                                      (size_t)opt.slots * (size_t)comp_state_rows *
                                          (size_t)comp_state_width * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_state_score_layers[layer],
                                      (size_t)opt.slots * (size_t)comp_state_rows *
                                          (size_t)comp_state_width * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_rows_layers[layer],
                                      (size_t)opt.slots * kBoundedCompRows *
                                          (size_t)kHeadDim * sizeof(float)));
                CHECK_CUDA(cudaMemsetAsync(r.d_attn_comp_state_kv_layers[layer], 0,
                                           (size_t)opt.slots * (size_t)comp_state_rows *
                                               (size_t)comp_state_width * sizeof(float),
                                           r.stream));
                CHECK_CUDA(cudaMemsetAsync(r.d_attn_comp_state_score_layers[layer], 0,
                                           (size_t)opt.slots * (size_t)comp_state_rows *
                                               (size_t)comp_state_width * sizeof(float),
                                           r.stream));
                CHECK_CUDA(cudaMemsetAsync(r.d_attn_comp_rows_layers[layer], 0,
                                           (size_t)opt.slots * kBoundedCompRows *
                                               (size_t)kHeadDim * sizeof(float),
                                           r.stream));
            }
            r.d_attn_comp_state_kv = r.d_attn_comp_state_kv_layers[layer];
            r.d_attn_comp_state_score = r.d_attn_comp_state_score_layers[layer];
            r.d_attn_comp_rows = r.d_attn_comp_rows_layers[layer];
        }
        if (opt.true_ds4_indexer_attention_gate && ratio == 4) {
            if (!r.d_index_comp_kv_cur) {
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_kv_cur,
                                      (size_t)opt.slots * kIndexCompWidth * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_score_cur,
                                      (size_t)opt.slots * kIndexCompWidth * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_indexer_scores,
                                      (size_t)opt.slots * kIndexerTopK * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_indexer_topk,
                                      (size_t)opt.slots * kIndexerTopK * sizeof(uint32_t)));
            }
            if (!r.d_index_comp_state_kv_layers[layer]) {
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_state_kv_layers[layer],
                                      (size_t)opt.slots * kIndexCompStateRows *
                                          (size_t)kIndexCompWidth * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_state_score_layers[layer],
                                      (size_t)opt.slots * kIndexCompStateRows *
                                          (size_t)kIndexCompWidth * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_rows_layers[layer],
                                      (size_t)opt.slots * kBoundedCompRows *
                                          (size_t)kIndexerHeadDim * sizeof(float)));
                CHECK_CUDA(cudaMemsetAsync(r.d_index_comp_state_kv_layers[layer], 0,
                                           (size_t)opt.slots * kIndexCompStateRows *
                                               (size_t)kIndexCompWidth * sizeof(float),
                                           r.stream));
                CHECK_CUDA(cudaMemsetAsync(r.d_index_comp_state_score_layers[layer], 0,
                                           (size_t)opt.slots * kIndexCompStateRows *
                                               (size_t)kIndexCompWidth * sizeof(float),
                                           r.stream));
                CHECK_CUDA(cudaMemsetAsync(r.d_index_comp_rows_layers[layer], 0,
                                           (size_t)opt.slots * kBoundedCompRows *
                                               (size_t)kIndexerHeadDim * sizeof(float),
                                           r.stream));
            }
            r.d_index_comp_state_kv = r.d_index_comp_state_kv_layers[layer];
            r.d_index_comp_state_score = r.d_index_comp_state_score_layers[layer];
            r.d_index_comp_rows = r.d_index_comp_rows_layers[layer];
        }
        for (int src = 0; src < kGpus; ++src) {
            if (!r.d_ep_remote[src]) CHECK_CUDA(cudaMalloc(&r.d_ep_remote[src],
                                                           (size_t)remote_float_bytes));
            if (opt.ep_return_fp16 && !r.d_ep_remote_half[src]) {
                CHECK_CUDA(cudaMalloc(&r.d_ep_remote_half[src],
                                      (size_t)(shard_elems * sizeof(__half))));
            }
        }
    }
    return 0;
}

int run_next_hidden_compose(const Options &opt,
                            const std::vector<ContractRow> &rows,
                            RankState ranks[kGpus],
                            ComposeStats *stats) {
    if (!opt.compose_next_hidden) return 0;
    stats->enabled = true;
    stats->ep_return_fp16 = opt.ep_return_fp16;
    stats->fused_compose_sum =
        opt.fuse_compose_sum && !opt.ep_return_fp16 && !opt.compact_route_compose;
    stats->dense_hmma_compose = opt.dense_hmma_compose;
    stats->dense_f16_cublas_compose = opt.dense_f16_cublas_compose;
    stats->nccl_reduce_scatter_compose =
        opt.nccl_reduce_scatter_compose_gate &&
        !opt.compact_route_compose && !opt.ep_return_fp16;

    DeviceDenseOutputs attn;
    DeviceDenseOutputs shared;
    const std::string attn_tensor = layer_tensor_name(opt.layer, "attn_output_b.weight");
    const std::string shared_tensor = layer_tensor_name(opt.layer, "ffn_down_shexp.weight");
    if (run_f8_dense_to_device(opt, rows, attn_tensor.c_str(), 1, &attn) != 0 ||
        run_f8_dense_to_device(opt, rows, shared_tensor.c_str(), 2, &shared) != 0) {
        free_device_dense_outputs(attn, opt);
        free_device_dense_outputs(shared, opt);
        return 1;
    }
    stats->attn_dense_ms = attn.compute_ms;
    stats->shared_dense_ms = shared.compute_ms;

    const uint64_t shard_elems = (uint64_t)opt.slots * (kHidden / kGpus);
    const uint64_t shard_bytes = shard_elems * sizeof(float);
    const uint64_t return_shard_bytes =
        shard_elems * (opt.ep_return_fp16 ? sizeof(__half) : sizeof(float));
    const uint64_t all_contrib_elems = (uint64_t)kGpus * shard_elems;
    const uint64_t all_contrib_bytes = all_contrib_elems * sizeof(float);
    const bool skip_self_copy = opt.skip_self_compose_copy && !opt.ep_return_fp16;
    const bool nccl_reduce_scatter = stats->nccl_reduce_scatter_compose;
    stats->ep_contribution_bytes = all_contrib_bytes * kGpus;
    if (opt.compact_route_compose && !opt.ep_return_fp16) {
        uint64_t compact_return_bytes = 0;
        for (int src = 0; src < kGpus; ++src) {
            compact_return_bytes +=
                (uint64_t)ranks[src].routes * (kHidden / kGpus) * sizeof(float) *
                (skip_self_copy ? (kGpus - 1) : kGpus);
        }
        stats->ep_return_bytes = compact_return_bytes;
    } else {
        stats->ep_return_bytes = return_shard_bytes *
                                 (skip_self_copy ? (kGpus * kGpus - kGpus)
                                                 : (kGpus * kGpus));
    }
    if (ensure_compose_buffers(opt, ranks) != 0) {
        free_device_dense_outputs(attn, opt);
        free_device_dense_outputs(shared, opt);
        return 2;
    }

    const auto compose_start = std::chrono::steady_clock::now();

    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        const int block = 256;
        int grid = (int)((all_contrib_elems + block - 1) / block);
        zero_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_contrib_all,
                                                      all_contrib_elems);
        CHECK_CUDA(cudaGetLastError());
        const uint64_t route_hidden_elems = (uint64_t)r.routes * kHidden;
        grid = (int)((route_hidden_elems + block - 1) / block);
        if (route_hidden_elems > 0) {
            ep_reduce_all_dest_shards_kernel<<<grid, block, 0, r.stream>>>(
                r.d_ep_contrib_all, r.d_down, r.d_route_slots, r.d_route_weights,
                r.routes, opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        if (opt.ep_return_fp16) {
            grid = (int)((all_contrib_elems + block - 1) / block);
            cast_f32_to_half_kernel<<<grid, block, 0, r.stream>>>(
                r.d_ep_contrib_half_all, r.d_ep_contrib_all, all_contrib_elems);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    for (int p = 0; p < kGpus; ++p) {
        CHECK_CUDA(cudaSetDevice(ranks[p].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[p].stream));
    }

    if (nccl_reduce_scatter) {
        for (int p = 0; p < kGpus; ++p) {
            if (!ranks[p].compose_nccl_initialized || !ranks[p].compose_nccl) {
                return 3;
            }
        }
        CHECK_NCCL(ncclGroupStart());
        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_NCCL(ncclReduceScatter(ranks[p].d_ep_contrib_all,
                                         ranks[p].d_ep_sum,
                                         (size_t)shard_elems,
                                         ncclFloat,
                                         ncclSum,
                                         ranks[p].compose_nccl,
                                         ranks[p].stream));
        }
        CHECK_NCCL(ncclGroupEnd());
    } else {
        for (int dst = 0; dst < kGpus; ++dst) {
            CHECK_CUDA(cudaSetDevice(ranks[dst].device));
            for (int src = 0; src < kGpus; ++src) {
                if (skip_self_copy && src == dst) continue;
                if (opt.ep_return_fp16) {
                    const __half *src_ptr =
                        ranks[src].d_ep_contrib_half_all + (uint64_t)dst * shard_elems;
                    CHECK_CUDA(cudaMemcpyPeerAsync(ranks[dst].d_ep_remote_half[src],
                                                   ranks[dst].device,
                                                   src_ptr,
                                                   ranks[src].device,
                                                   (size_t)return_shard_bytes,
                                                   ranks[dst].stream));
                } else {
                    const float *src_ptr = ranks[src].d_ep_contrib_all +
                                           (uint64_t)dst * shard_elems;
                    CHECK_CUDA(cudaMemcpyPeerAsync(ranks[dst].d_ep_remote[src],
                                                   ranks[dst].device,
                                                   src_ptr,
                                                   ranks[src].device,
                                                   (size_t)return_shard_bytes,
                                                   ranks[dst].stream));
                }
            }
        }
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        CHECK_CUDA(cudaSetDevice(ranks[dst].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[dst].stream));
    }

    std::vector<std::vector<float>> first((size_t)kGpus);
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            const int block = 256;
            int grid = (int)((shard_elems + block - 1) / block);
            if (nccl_reduce_scatter) {
                compose_next_hidden_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn.d_out[(size_t)dst],
                    shared.d_out[(size_t)dst], r.d_ep_sum, dst, opt.slots);
            } else if (stats->fused_compose_sum) {
                const float *r0 = skip_self_copy && dst == 0
                    ? ranks[0].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[0];
                const float *r1 = skip_self_copy && dst == 1
                    ? ranks[1].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[1];
                const float *r2 = skip_self_copy && dst == 2
                    ? ranks[2].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[2];
                const float *r3 = skip_self_copy && dst == 3
                    ? ranks[3].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[3];
                const float *r4 = skip_self_copy && dst == 4
                    ? ranks[4].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[4];
                const float *r5 = skip_self_copy && dst == 5
                    ? ranks[5].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[5];
                const float *r6 = skip_self_copy && dst == 6
                    ? ranks[6].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[6];
                const float *r7 = skip_self_copy && dst == 7
                    ? ranks[7].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[7];
                compose_next_hidden_sum8_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn.d_out[(size_t)dst],
                    shared.d_out[(size_t)dst], r0, r1, r2, r3, r4, r5, r6, r7,
                    dst, opt.slots);
            } else {
                zero_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_sum, shard_elems);
                CHECK_CUDA(cudaGetLastError());
                for (int src = 0; src < kGpus; ++src) {
                    if (opt.ep_return_fp16) {
                        add_half_to_f32_kernel<<<grid, block, 0, r.stream>>>(
                            r.d_ep_sum, r.d_ep_remote_half[src], shard_elems);
                    } else {
                        const float *src_contrib = skip_self_copy && src == dst
                            ? ranks[src].d_ep_contrib_all + (uint64_t)dst * shard_elems
                            : r.d_ep_remote[src];
                        add_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_sum,
                                                                     src_contrib,
                                                                     shard_elems);
                    }
                }
                CHECK_CUDA(cudaGetLastError());
                compose_next_hidden_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn.d_out[(size_t)dst],
                    shared.d_out[(size_t)dst], r.d_ep_sum, dst, opt.slots);
            }
            CHECK_CUDA(cudaGetLastError());
        }
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_CUDA(cudaStreamSynchronize(r.stream));
            std::vector<float> host((size_t)shard_elems);
            CHECK_CUDA(cudaMemcpy(host.data(), r.d_next_hidden, (size_t)shard_bytes,
                                  cudaMemcpyDeviceToHost));
            if (repeat == 0) {
                first[(size_t)dst] = host;
                for (uint64_t i = 0; i < shard_elems; ++i) {
                    if (!std::isfinite(host[(size_t)i])) {
                        stats->finite_bad++;
                        stats->pass = false;
                    }
                    uint32_t bits = 0;
                    std::memcpy(&bits, &host[(size_t)i], sizeof(bits));
                    stats->checksum ^=
                        (uint64_t)bits + (uint64_t)(dst + 1) * 1000003ull + i * 9176ull;
                }
            } else {
                for (uint64_t i = 0; i < shard_elems; ++i) {
                    const double diff =
                        std::fabs((double)host[(size_t)i] -
                                  (double)first[(size_t)dst][(size_t)i]);
                    stats->repeat_max_abs = std::max(stats->repeat_max_abs, diff);
                    if (diff > 0.0) {
                        stats->repeat_bad++;
                        stats->pass = false;
                    }
                }
            }
        }
    }

    const auto compose_stop = std::chrono::steady_clock::now();
    stats->compose_ms =
        std::chrono::duration<double, std::milli>(compose_stop - compose_start).count();
    if (stats->checksum == 0 || stats->finite_bad != 0 || stats->repeat_bad != 0) {
        stats->pass = false;
    }

    free_device_dense_outputs(attn, opt);
    free_device_dense_outputs(shared, opt);
    return stats->pass ? 0 : 2;
}

int run_true_ds4_attention_projection_prefix(const Options &opt,
                                             SharedHcControls *hc,
                                             const LayerDenseOps *ops,
                                             RankState ranks[kGpus],
                                             int layer) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    if (!hc->d_current_full || !hc->d_attn_normed ||
        !hc->d_q_a_full || !hc->d_q_a_normed ||
        !hc->d_kv_full || !hc->d_kv_normed ||
        !hc->d_attn_norm_weight[layer] ||
        !hc->d_q_a_norm_weight[layer] ||
        !hc->d_kv_a_norm_weight[layer]) {
        return 2;
    }
    if (ops->attn_q_a.cols != kHidden || ops->attn_q_a.rows_per_gpu != 1024 / kGpus ||
        ops->attn_q_b.cols != 1024 || ops->attn_q_b.rows_per_gpu != 32768 / kGpus ||
        ops->attn_kv_latent.cols != kHidden ||
        ops->attn_kv_latent.rows_per_gpu != kHeadDim / kGpus) {
        std::fprintf(stderr,
                     "tp_ep_true_attention_projection_bad_shape\tlayer\t%d\t"
                     "q_a_cols\t%d\tq_a_rows_per_gpu\t%d\t"
                     "q_b_cols\t%d\tq_b_rows_per_gpu\t%d\t"
                     "kv_cols\t%d\tkv_rows_per_gpu\t%d\n",
                     layer,
                     ops->attn_q_a.cols, ops->attn_q_a.rows_per_gpu,
                     ops->attn_q_b.cols, ops->attn_q_b.rows_per_gpu,
                     ops->attn_kv_latent.cols, ops->attn_kv_latent.rows_per_gpu);
        return 3;
    }

    const auto start = std::chrono::steady_clock::now();
    const int block = 256;
    const uint64_t hidden_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const uint64_t q_a_elems = (uint64_t)opt.slots * 1024ull;
    const uint64_t kv_elems = (uint64_t)opt.slots * (uint64_t)kHeadDim;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    cudaStream_t control_stream = graph_event_order ? ranks[0].stream : (cudaStream_t)0;

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    rms_norm_weight_rows_stable_kernel<<<
        (unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_attn_normed, hc->d_current_full, hc->d_attn_norm_weight[layer],
        (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());
    if (graph_event_order) {
        if (enqueue_rank_streams_wait_after_control(
                opt, ranks, control_stream) != 0) return 8;
    } else {
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_current_full ||
            !ops->attn_q_a.d_x_half[(size_t)rank] ||
            !ops->attn_kv_latent.d_x_half[(size_t)rank]) {
            return 4;
        }
        if (rank == 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_current_full, hc->d_attn_normed,
                                       (size_t)hidden_elems * sizeof(float),
                                       cudaMemcpyDeviceToDevice, r.stream));
        } else {
            CHECK_CUDA(cudaMemcpyPeerAsync(r.d_current_full, r.device,
                                           hc->d_attn_normed, opt.devices[0],
                                           (size_t)hidden_elems * sizeof(float),
                                           r.stream));
        }
        fill_dense_input_half_from_current_kernel<<<
            (unsigned int)((hidden_elems + block - 1) / block), block, 0,
            r.stream>>>(ops->attn_q_a.d_x_half[(size_t)rank],
                         r.d_current_full, (uint32_t)kHidden,
                         (uint32_t)opt.slots);
        fill_dense_input_half_from_current_kernel<<<
            (unsigned int)((hidden_elems + block - 1) / block), block, 0,
            r.stream>>>(ops->attn_kv_latent.d_x_half[(size_t)rank],
                         r.d_current_full, (uint32_t)kHidden,
                         (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    if (graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 9;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }

    if (launch_resident_f8_dense(opt, ops->attn_q_a, ranks) != 0 ||
        launch_resident_f8_dense(opt, ops->attn_kv_latent, ranks) != 0) {
        return 5;
    }
    if (graph_event_order) {
        if (enqueue_control_wait_after_dense_streams(
                opt, ranks, control_stream) != 0) return 10;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            if (ranks[rank].dense_stream) {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].dense_stream));
            } else {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        gather_dense_shard_to_full_kernel<<<
            (unsigned int)(((uint64_t)opt.slots * (1024u / kGpus) + block - 1) / block),
            block, 0, control_stream>>>(
            hc->d_q_a_full, ops->attn_q_a.d_out[(size_t)rank], rank,
            1024u / kGpus, 1024u, (uint32_t)opt.slots);
        gather_dense_shard_to_full_kernel<<<
            (unsigned int)(((uint64_t)opt.slots * (kHeadDim / kGpus) + block - 1) / block),
            block, 0, control_stream>>>(
            hc->d_kv_full, ops->attn_kv_latent.d_out[(size_t)rank], rank,
            kHeadDim / kGpus, kHeadDim, (uint32_t)opt.slots);
    }
    CHECK_CUDA(cudaGetLastError());
    if (!graph_event_order) {
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    rms_norm_weight_rows_stable_kernel<<<
        (unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_q_a_normed, hc->d_q_a_full, hc->d_q_a_norm_weight[layer],
        1024u, (uint32_t)opt.slots, 1.0e-6f);
    rms_norm_weight_rows_stable_kernel<<<
        (unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_kv_normed, hc->d_kv_full, hc->d_kv_a_norm_weight[layer],
        (uint32_t)kHeadDim, (uint32_t)opt.slots, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());
    if (graph_event_order) {
        if (enqueue_rank_streams_wait_after_control(
                opt, ranks, control_stream) != 0) return 11;
    } else {
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    if (opt.true_ds4_attention_kv_norm_reference_gate) {
        float *d_kv_ref = nullptr;
        CHECK_CUDA(cudaMalloc(&d_kv_ref, (size_t)kv_elems * sizeof(float)));
        rms_norm_weight_rows_kernel<<<(unsigned int)opt.slots, 256>>>(
            d_kv_ref, hc->d_kv_full, hc->d_kv_a_norm_weight[layer],
            (uint32_t)kHeadDim, (uint32_t)opt.slots, 1.0e-6f);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        const TensorF32Stats kv_in =
            collect_tensor_f32_stats(hc->d_kv_full, (size_t)kv_elems, nullptr);
        const TensorF32Stats kv_stable =
            collect_tensor_f32_stats(hc->d_kv_normed, (size_t)kv_elems,
                                     nullptr);
        const TensorF32Stats kv_ref =
            collect_tensor_f32_stats(d_kv_ref, (size_t)kv_elems, nullptr);
        const TensorF32Stats kv_w =
            collect_tensor_f32_stats(hc->d_kv_a_norm_weight[layer],
                                     (size_t)kHeadDim, nullptr);
        const TensorF32DiffStats diff = collect_tensor_f32_diff_stats(
            hc->d_kv_normed, d_kv_ref, (size_t)kv_elems, nullptr);
        std::printf("tp_ep_true_attention_kv_norm_reference\tlayer\t%d\t"
                    "slots\t%d\tkv_in_max\t%.9g\tkv_in_bad\t%d\t"
                    "kv_weight_max\t%.9g\tkv_weight_bad\t%d\t"
                    "stable_max\t%.9g\tstable_bad\t%d\t"
                    "reference_max\t%.9g\treference_bad\t%d\t"
                    "max_abs_diff\t%.9g\tmax_rel_diff\t%.9g\tdiff_bad\t%d\t"
                    "first_bad\t%zu\tPASS\n",
                    layer, opt.slots, kv_in.max_abs, kv_in.finite_bad,
                    kv_w.max_abs, kv_w.finite_bad, kv_stable.max_abs,
                    kv_stable.finite_bad, kv_ref.max_abs, kv_ref.finite_bad,
                    diff.max_abs, diff.max_rel, diff.bad, diff.first_bad);
        CHECK_CUDA(cudaFree(d_kv_ref));
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!ops->attn_q_b.d_x_half[(size_t)rank]) return 6;
        fill_dense_input_half_from_tensor_kernel<<<
            (unsigned int)((q_a_elems + block - 1) / block), block, 0,
            r.stream>>>(ops->attn_q_b.d_x_half[(size_t)rank],
                         hc->d_q_a_normed, 1024u, (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    if (graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 12;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }

    if (launch_resident_f8_dense(opt, ops->attn_q_b, ranks) != 0) {
        return 7;
    }
    if (!graph_event_order) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            if (ranks[rank].dense_stream) {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].dense_stream));
            } else {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }

    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    if (!graph_event_order && layer <= 2) {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        log_tensor_f32_stats("true_attn_q_a_full", layer, 0, hc->d_q_a_full,
                             (size_t)q_a_elems, nullptr);
        log_tensor_f32_stats("true_attn_kv_normed", layer, 0, hc->d_kv_normed,
                             (size_t)kv_elems, nullptr);
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("true_attn_q_b_shard", layer, rank,
                                 ops->attn_q_b.d_out[(size_t)rank],
                                 (size_t)opt.slots * ops->attn_q_b.rows_per_gpu,
                                 ranks[rank].dense_stream ? ranks[rank].dense_stream
                                                          : ranks[rank].stream);
        }
    }
    if (!graph_event_order && opt.true_ds4_attention_saturation_audit_gate) {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        const TensorF32Stats current =
            collect_tensor_f32_stats(hc->d_current_full, (size_t)hidden_elems,
                                     nullptr);
        const TensorF32Stats attn_normed =
            collect_tensor_f32_stats(hc->d_attn_normed, (size_t)hidden_elems,
                                     nullptr);
        const TensorF32Stats q_a =
            collect_tensor_f32_stats(hc->d_q_a_full, (size_t)q_a_elems,
                                     nullptr);
        const TensorF32Stats q_a_normed =
            collect_tensor_f32_stats(hc->d_q_a_normed, (size_t)q_a_elems,
                                     nullptr);
        const TensorF32Stats kv =
            collect_tensor_f32_stats(hc->d_kv_full, (size_t)kv_elems,
                                     nullptr);
        const TensorF32Stats kv_normed =
            collect_tensor_f32_stats(hc->d_kv_normed, (size_t)kv_elems,
                                     nullptr);
        TensorF32Stats q_b;
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            const TensorF32Stats shard = collect_tensor_f32_stats(
                ops->attn_q_b.d_out[(size_t)rank],
                (size_t)opt.slots * ops->attn_q_b.rows_per_gpu,
                ranks[rank].dense_stream ? ranks[rank].dense_stream
                                         : ranks[rank].stream);
            merge_tensor_stats(&q_b, shard);
        }
        std::printf("tp_ep_true_attention_saturation_projection\tlayer\t%d\t"
                    "slots\t%d\tcurrent_max\t%.9g\tcurrent_bad\t%d\t"
                    "attn_normed_max\t%.9g\tattn_normed_bad\t%d\t"
                    "q_a_max\t%.9g\tq_a_bad\t%d\t"
                    "q_a_normed_max\t%.9g\tq_a_normed_bad\t%d\t"
                    "kv_max\t%.9g\tkv_bad\t%d\t"
                    "kv_normed_max\t%.9g\tkv_normed_bad\t%d\t"
                    "q_b_pre_head_max\t%.9g\tq_b_pre_head_bad\t%d\tPASS\n",
                    layer, opt.slots, current.max_abs, current.finite_bad,
                    attn_normed.max_abs, attn_normed.finite_bad, q_a.max_abs,
                    q_a.finite_bad, q_a_normed.max_abs, q_a_normed.finite_bad,
                    kv.max_abs, kv.finite_bad, kv_normed.max_abs,
                    kv_normed.finite_bad, q_b.max_abs, q_b.finite_bad);
    }
    std::printf("tp_ep_true_attention_projection_prefix\tlayer\t%d\tslots\t%d\t"
                "q_a_cols\t1024\tkv_cols\t%d\tq_width\t32768\tms\t%.6f\tPASS\n",
                layer, opt.slots, kHeadDim, ms);
    return 0;
}

int run_true_ds4_compressed_reference_diff_gate(const Options &opt,
                                                SharedHcControls *hc,
                                                RankState ranks[kGpus],
                                                int layer,
                                                int ratio,
                                                int comp_width,
                                                uint32_t emitted,
                                                uint32_t comp_row,
                                                uint32_t visible_rows) {
    if (!opt.true_ds4_compressed_reference_diff_gate) return 0;
    if (!hc || !hc->initialized || layer < 0 || layer >= 43) return 1;
    if (ratio != 4 || !emitted) {
        std::printf("tp_ep_compressed_reference_diff\tlayer\t%d\tratio\t%d\t"
                    "emitted\t%u\tSKIP\n",
                    layer, ratio, emitted);
        return 0;
    }
    RankState &r0 = ranks[0];
    CHECK_CUDA(cudaSetDevice(r0.device));
    if (!hc->d_attn_comp_kv_full || !hc->d_attn_comp_score_full ||
        !hc->d_attn_compress_ape[layer] || !hc->d_attn_compress_norm[layer] ||
        !r0.d_attn_comp_kv_cur || !r0.d_attn_comp_score_cur ||
        !r0.d_attn_comp_rows || !r0.d_index_comp_rows ||
        !r0.d_indexer_scores || !r0.d_indexer_topk ||
        !hc->d_index_comp_kv_full || !hc->d_index_comp_score_full ||
        !hc->d_indexer_compress_ape[layer] ||
        !hc->d_indexer_compress_norm[layer] ||
        !hc->d_indexer_q_full || !hc->d_indexer_w_full) {
        return 2;
    }

    const int block = 256;
    const uint32_t state_rows =
        (uint32_t)attn_comp_state_rows_for_ratio(ratio);
    const uint32_t state_width =
        (uint32_t)attn_comp_state_width_for_ratio(ratio);
    const float comp_freq_scale = 1.0f / kRopeScaleFactor;
    const float comp_ext_factor = 1.0f;
    float comp_attn_factor = 1.0f;
    comp_attn_factor /= 1.0f + 0.1f * logf(1.0f / comp_freq_scale);
    const size_t attn_state_elems =
        (size_t)opt.slots * state_rows * (size_t)state_width;
    const size_t attn_row_elems = (size_t)opt.slots * kHeadDim;
    const size_t index_state_elems =
        (size_t)opt.slots * kIndexCompStateRows * (size_t)kIndexCompWidth;
    const size_t index_row_elems = (size_t)opt.slots * kIndexerHeadDim;

    float *d_attn_state_kv = nullptr;
    float *d_attn_state_score = nullptr;
    float *d_attn_row_ref = nullptr;
    float *d_attn_row_tp = nullptr;
    float *d_index_state_kv = nullptr;
    float *d_index_state_score = nullptr;
    float *d_index_row_ref = nullptr;
    float *d_index_row_tp = nullptr;
    float *d_index_score_ref = nullptr;
    float *d_index_score_ref_compact = nullptr;
    float *d_index_score_tp = nullptr;
    uint32_t *d_index_topk_ref = nullptr;

    CHECK_CUDA(cudaMalloc(&d_attn_state_kv, attn_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_state_score, attn_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_row_ref, attn_row_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_row_tp, attn_row_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_state_kv, index_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_state_score, index_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_row_ref, index_row_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_row_tp, index_row_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_score_ref,
                          (size_t)opt.slots * kIndexerTopK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_score_ref_compact,
                          (size_t)opt.slots * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_score_tp, (size_t)opt.slots * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_topk_ref,
                          (size_t)opt.slots * kIndexerTopK * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_attn_state_kv, 0, attn_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_attn_state_score, 0, attn_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_index_state_kv, 0, index_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_index_state_score, 0, index_state_elems * sizeof(float)));

    log_tensor_f32_diff_summary("attn_comp_kv_current_peer_copy", layer,
                                r0.d_attn_comp_kv_cur,
                                hc->d_attn_comp_kv_full,
                                (size_t)opt.slots * (size_t)comp_width,
                                r0.stream);
    log_tensor_f32_diff_summary("attn_comp_score_current_peer_copy", layer,
                                r0.d_attn_comp_score_cur,
                                hc->d_attn_comp_score_full,
                                (size_t)opt.slots * (size_t)comp_width,
                                r0.stream);

    compressor_pool_emit_slots_kernel<<<
        dim3((unsigned int)((kHeadDim + block - 1) / block),
             (unsigned int)opt.slots, 1u),
        block>>>(d_attn_row_ref, r0.d_attn_comp_state_kv,
                 r0.d_attn_comp_state_score,
                 (uint32_t)opt.slots, (uint32_t)kHeadDim, (uint32_t)ratio,
                 0u, 1u, state_rows, state_width);
    compressor_norm_emit_slots_kernel<<<(unsigned int)opt.slots, 256>>>(
        d_attn_row_ref, hc->d_attn_compress_norm[layer], (uint32_t)opt.slots,
        (uint32_t)kHeadDim, 0u, 1u, 1.0e-6f);
    if (opt.true_ds4_attention_rope_gate) {
        rope_tail_comp_emit_slots_kernel<<<(unsigned int)opt.slots, 64>>>(
            d_attn_row_ref, (uint32_t)opt.slots, (uint32_t)kHeadDim,
            (uint32_t)kRotaryDim, 0u, 1u,
            (uint32_t)(opt.position + 1ull - (uint64_t)ratio),
            kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
            comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
            kRopeYarnBetaSlow);
    }
    round_comp_emit_slots_kernel<<<
        (unsigned int)(((uint64_t)opt.slots * kHeadDim + block - 1) / block),
        block>>>(d_attn_row_ref, (uint32_t)opt.slots, (uint32_t)kHeadDim,
                 0u, 1u);
    pack_comp_row_kernel<<<
        (unsigned int)(((uint64_t)opt.slots * kHeadDim + block - 1) / block),
        block>>>(d_attn_row_tp, r0.d_attn_comp_rows, (uint32_t)opt.slots,
                 (uint32_t)kHeadDim, comp_row, (uint32_t)kBoundedCompRows);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    log_tensor_f32_diff_summary("attn_comp_row_compact_reference", layer,
                                d_attn_row_tp, d_attn_row_ref, attn_row_elems,
                                nullptr);

    compressor_pool_emit_slots_kernel<<<
        dim3((unsigned int)((kIndexerHeadDim + block - 1) / block),
             (unsigned int)opt.slots, 1u),
        block>>>(d_index_row_ref, r0.d_index_comp_state_kv,
                 r0.d_index_comp_state_score,
                 (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, 4u,
                 0u, 1u, (uint32_t)kIndexCompStateRows,
                 (uint32_t)kIndexCompWidth);
    compressor_norm_emit_slots_kernel<<<(unsigned int)opt.slots, 256>>>(
        d_index_row_ref, hc->d_indexer_compress_norm[layer],
        (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, 0u, 1u,
        1.0e-6f);
    if (opt.true_ds4_attention_rope_gate) {
        rope_tail_comp_emit_slots_kernel<<<(unsigned int)opt.slots, 64>>>(
            d_index_row_ref, (uint32_t)opt.slots,
            (uint32_t)kIndexerHeadDim, (uint32_t)kRotaryDim, 0u, 1u,
            (uint32_t)(opt.position + 1ull - 4ull),
            kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
            comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
            kRopeYarnBetaSlow);
    }
    round_comp_emit_slots_kernel<<<
        (unsigned int)(((uint64_t)opt.slots * kIndexerHeadDim + block - 1) /
                       block),
        block>>>(d_index_row_ref, (uint32_t)opt.slots,
                 (uint32_t)kIndexerHeadDim, 0u, 1u);
    pack_comp_row_kernel<<<
        (unsigned int)(((uint64_t)opt.slots * kIndexerHeadDim + block - 1) /
                       block),
        block>>>(d_index_row_tp, r0.d_index_comp_rows, (uint32_t)opt.slots,
                 (uint32_t)kIndexerHeadDim, comp_row,
                 (uint32_t)kBoundedCompRows);
    indexer_score_bounded_rows_slots_kernel<<<(unsigned int)opt.slots, 256>>>(
        d_index_score_ref, d_index_topk_ref, hc->d_indexer_q_full,
        hc->d_indexer_w_full, d_index_row_ref, (uint32_t)opt.slots,
        1u, 1u, (uint32_t)kIndexerTopK,
        1.0f / sqrtf((float)(kIndexerHead * kIndexerHeadDim)));
    pack_indexer_score_column_kernel<<<
        (unsigned int)((opt.slots + block - 1) / block), block>>>(
        d_index_score_tp, r0.d_indexer_scores, (uint32_t)opt.slots,
        (uint32_t)kIndexerTopK, comp_row);
    pack_indexer_score_column_kernel<<<
        (unsigned int)((opt.slots + block - 1) / block), block>>>(
        d_index_score_ref_compact, d_index_score_ref, (uint32_t)opt.slots,
        (uint32_t)kIndexerTopK, 0u);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    log_tensor_f32_diff_summary("index_comp_row_compact_reference", layer,
                                d_index_row_tp, d_index_row_ref,
                                index_row_elems, nullptr);
    log_tensor_f32_diff_summary("indexer_score_row_compact_reference", layer,
                                d_index_score_tp, d_index_score_ref_compact,
                                (size_t)opt.slots, nullptr);
    std::printf("tp_ep_compressed_reference_diff_summary\tlayer\t%d\t"
                "ratio\t%d\temitted\t%u\tcomp_row\t%u\t"
                "visible_compressed_rows\t%u\tPASS\n",
                layer, ratio, emitted, comp_row, visible_rows);

    CHECK_CUDA(cudaFree(d_index_topk_ref));
    CHECK_CUDA(cudaFree(d_index_score_tp));
    CHECK_CUDA(cudaFree(d_index_score_ref_compact));
    CHECK_CUDA(cudaFree(d_index_score_ref));
    CHECK_CUDA(cudaFree(d_index_row_tp));
    CHECK_CUDA(cudaFree(d_index_row_ref));
    CHECK_CUDA(cudaFree(d_index_state_score));
    CHECK_CUDA(cudaFree(d_index_state_kv));
    CHECK_CUDA(cudaFree(d_attn_row_tp));
    CHECK_CUDA(cudaFree(d_attn_row_ref));
    CHECK_CUDA(cudaFree(d_attn_state_score));
    CHECK_CUDA(cudaFree(d_attn_state_kv));
    return 0;
}

int run_true_ds4_compressed_kv_projection_gate(const Options &opt,
                                               SharedHcControls *hc,
                                               const LayerDenseOps *ops,
                                               RankState ranks[kGpus],
                                               ds4_v100_tp_runtime *rt,
                                               int layer) {
    if (!opt.true_ds4_compressed_kv_gate) return 0;
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 43 || !hc->d_attn_normed || !hc->d_q_a_normed) {
        return 1;
    }
    const int ratio = ds4_layer_ratio(layer);
    const uint32_t emitted =
        ratio != 0 && (((opt.position + 1ull) % (uint64_t)ratio) == 0ull) ? 1u : 0u;
    const uint32_t indexer_topk =
        opt.true_ds4_indexer_attention_gate && ratio == 4 ? kIndexerTopK : 0u;
    if (ratio == 0) {
        std::printf("tp_ep_compressed_kv_projection\tlayer\t%d\tslots\t%d\t"
                    "ratio\t0\temitted_compressed_rows\t0\t"
                    "visible_compressed_rows\t0\tindexer_topk_count\t0\t"
                    "attn_input_fill_ms\t0.000000\tattn_dense_ms\t0.000000\t"
                    "attn_gather_ms\t0.000000\tattn_state_emit_ms\t0.000000\t"
                    "attn_typed_ms\t0.000000\tindexer_input_fill_ms\t0.000000\t"
                    "indexer_dense_ms\t0.000000\tindexer_gather_rope_ms\t0.000000\t"
                    "indexer_state_emit_ms\t0.000000\tindexer_typed_score_ms\t0.000000\t"
                    "reference_diff_ms\t0.000000\tratio_shift_ms\t0.000000\t"
                    "ms\t0.000000\tPASS\n",
                    layer, opt.slots);
        return 0;
    }

    const int comp_width = ratio == 4 ? 2 * kHeadDim : kHeadDim;
    const int comp_state_rows = attn_comp_state_rows_for_ratio(ratio);
    const int comp_state_width = attn_comp_state_width_for_ratio(ratio);
    if (ops->attn_compress_kv.cols != kHidden ||
        ops->attn_compress_gate.cols != kHidden ||
        ops->attn_compress_kv.rows_per_gpu != comp_width / kGpus ||
        ops->attn_compress_gate.rows_per_gpu != comp_width / kGpus) {
        std::fprintf(stderr,
                     "tp_ep_compressed_kv_bad_shape\tlayer\t%d\t"
                     "ratio\t%d\tkv_cols\t%d\tkv_rows_per_gpu\t%d\t"
                     "gate_cols\t%d\tgate_rows_per_gpu\t%d\n",
                     layer, ratio, ops->attn_compress_kv.cols,
                     ops->attn_compress_kv.rows_per_gpu,
                     ops->attn_compress_gate.cols,
                     ops->attn_compress_gate.rows_per_gpu);
        return 2;
    }

    const auto start = std::chrono::steady_clock::now();
    auto t_stage = start;
    auto elapsed_ms = [](std::chrono::steady_clock::time_point a,
                         std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    double attn_input_fill_ms = 0.0;
    double attn_dense_ms = 0.0;
    double attn_gather_ms = 0.0;
    double attn_state_emit_ms = 0.0;
    double attn_typed_ms = 0.0;
    double indexer_input_fill_ms = 0.0;
    double indexer_dense_ms = 0.0;
    double indexer_gather_rope_ms = 0.0;
    double indexer_state_emit_ms = 0.0;
    double indexer_typed_score_ms = 0.0;
    double reference_diff_ms = 0.0;
    double ratio_shift_ms = 0.0;
    const int block = 256;
    const uint64_t hidden_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const bool direct_current_input_fill =
        opt.true_ds4_compressed_kv_direct_input_fill_gate;
    const bool dense_event_wait =
        opt.true_ds4_compressed_kv_dense_event_wait_gate;
    const bool skip_dense_stats =
        opt.true_ds4_compressed_kv_skip_dense_stats_gate;
    const bool fused_attn_current_fill =
        opt.true_ds4_compressed_kv_fused_attn_input_fill_gate;
    const bool fused_ratio4_current_fill =
        opt.true_ds4_compressed_kv_fused_input_fill_gate &&
        opt.true_ds4_indexer_attention_gate && ratio == 4;
    const bool fused_rope_round =
        opt.true_ds4_compressed_kv_fused_rope_round_gate &&
        opt.true_ds4_attention_rope_gate && emitted;
    const bool fused_pool_norm =
        opt.true_ds4_compressed_kv_fused_pool_norm_gate && emitted;
    const bool fused_pool_norm_rope_round =
        opt.true_ds4_compressed_kv_fused_pool_norm_rope_round_gate &&
        opt.true_ds4_attention_rope_gate && emitted;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    cudaStream_t control_stream = graph_event_order ? ranks[0].stream : (cudaStream_t)0;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_current_full ||
            !ops->attn_compress_kv.d_x_half[(size_t)rank] ||
            !ops->attn_compress_gate.d_x_half[(size_t)rank] ||
            (fused_ratio4_current_fill &&
             (!ops->indexer_proj.d_x_half[(size_t)rank] ||
              !ops->indexer_compress_kv.d_x_half[(size_t)rank] ||
              !ops->indexer_compress_gate.d_x_half[(size_t)rank]))) {
            return 3;
        }
        const float *current_src = hc->d_attn_normed;
        if (!direct_current_input_fill) {
            if (rank == 0) {
                CHECK_CUDA(cudaMemcpyAsync(r.d_current_full, hc->d_attn_normed,
                                           (size_t)hidden_elems * sizeof(float),
                                           cudaMemcpyDeviceToDevice, r.stream));
            } else {
                CHECK_CUDA(cudaMemcpyPeerAsync(r.d_current_full, r.device,
                                               hc->d_attn_normed, opt.devices[0],
                                               (size_t)hidden_elems * sizeof(float),
                                               r.stream));
            }
            current_src = r.d_current_full;
        }
        if (fused_ratio4_current_fill) {
            fill_ratio4_compressed_indexer_inputs_half_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(
                ops->attn_compress_kv.d_x_half[(size_t)rank],
                ops->attn_compress_gate.d_x_half[(size_t)rank],
                ops->indexer_proj.d_x_half[(size_t)rank],
                ops->indexer_compress_kv.d_x_half[(size_t)rank],
                ops->indexer_compress_gate.d_x_half[(size_t)rank],
                current_src, (uint32_t)opt.slots);
        } else if (fused_attn_current_fill) {
            fill_attn_compressed_inputs_half_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(
                ops->attn_compress_kv.d_x_half[(size_t)rank],
                ops->attn_compress_gate.d_x_half[(size_t)rank],
                current_src, (uint32_t)opt.slots);
        } else {
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->attn_compress_kv.d_x_half[(size_t)rank],
                             current_src, (uint32_t)kHidden,
                             (uint32_t)opt.slots);
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->attn_compress_gate.d_x_half[(size_t)rank],
                             current_src, (uint32_t)kHidden,
                             (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    if (dense_event_wait || graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 22;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_input_fill_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }
    if (launch_resident_f8_dense(opt, ops->attn_compress_kv, ranks) != 0 ||
        launch_resident_f8_dense(opt, ops->attn_compress_gate, ranks) != 0) {
        return 4;
    }

    TensorF32Stats attn_kv_stats;
    TensorF32Stats attn_gate_stats;
    if (graph_event_order) {
        if (enqueue_control_wait_after_dense_streams(
                opt, ranks, control_stream) != 0) return 24;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            cudaStream_t stream = ranks[rank].dense_stream ? ranks[rank].dense_stream
                                                           : ranks[rank].stream;
            CHECK_CUDA(cudaStreamSynchronize(stream));
            if (!skip_dense_stats) {
            const size_t comp_elems =
                (size_t)opt.slots * (size_t)ops->attn_compress_kv.rows_per_gpu;
                merge_tensor_stats(&attn_kv_stats,
                                   collect_tensor_f32_stats(
                                       ops->attn_compress_kv.d_out[(size_t)rank],
                                       comp_elems, stream));
                merge_tensor_stats(&attn_gate_stats,
                                   collect_tensor_f32_stats(
                                       ops->attn_compress_gate.d_out[(size_t)rank],
                                       comp_elems, stream));
            }
        }
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_dense_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }

    if (!hc->d_attn_comp_kv_full || !hc->d_attn_comp_score_full ||
        !hc->d_attn_compress_ape[layer] || !hc->d_attn_compress_norm[layer]) {
        return 9;
    }
    uint32_t emitted_comp_row = 0u;
    uint32_t visible = 0u;
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        gather_dense_shard_to_full_kernel<<<
            (unsigned int)(((uint64_t)opt.slots *
                                (uint64_t)ops->attn_compress_kv.rows_per_gpu +
                            block - 1) /
                           block),
            block, 0, control_stream>>>(
            hc->d_attn_comp_kv_full,
            ops->attn_compress_kv.d_out[(size_t)rank], rank,
            (uint32_t)ops->attn_compress_kv.rows_per_gpu,
            (uint32_t)comp_width, (uint32_t)opt.slots);
        gather_dense_shard_to_full_kernel<<<
            (unsigned int)(((uint64_t)opt.slots *
                                (uint64_t)ops->attn_compress_gate.rows_per_gpu +
                            block - 1) /
                           block),
            block, 0, control_stream>>>(
            hc->d_attn_comp_score_full,
            ops->attn_compress_gate.d_out[(size_t)rank], rank,
            (uint32_t)ops->attn_compress_gate.rows_per_gpu,
            (uint32_t)comp_width, (uint32_t)opt.slots);
    }
    CHECK_CUDA(cudaGetLastError());
    if (graph_event_order) {
        if (enqueue_rank_streams_wait_after_control(
                opt, ranks, control_stream) != 0) return 25;
    } else {
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_gather_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }

    const float comp_freq_scale = 1.0f / kRopeScaleFactor;
    const float comp_ext_factor = 1.0f;
    float comp_attn_factor = 1.0f;
    comp_attn_factor /= 1.0f + 0.1f * logf(1.0f / comp_freq_scale);
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_comp_kv_cur || !r.d_attn_comp_score_cur ||
            !r.d_attn_comp_state_kv || !r.d_attn_comp_state_score ||
            !r.d_attn_comp_rows) {
            return 10;
        }
        const size_t comp_bytes = (size_t)opt.slots * comp_width * sizeof(float);
        if (rank == 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_attn_comp_kv_cur, hc->d_attn_comp_kv_full,
                                       comp_bytes, cudaMemcpyDeviceToDevice, r.stream));
            CHECK_CUDA(cudaMemcpyAsync(r.d_attn_comp_score_cur,
                                       hc->d_attn_comp_score_full, comp_bytes,
                                       cudaMemcpyDeviceToDevice, r.stream));
        } else {
            CHECK_CUDA(cudaMemcpyPeerAsync(r.d_attn_comp_kv_cur, r.device,
                                           hc->d_attn_comp_kv_full, opt.devices[0],
                                           comp_bytes, r.stream));
            CHECK_CUDA(cudaMemcpyPeerAsync(r.d_attn_comp_score_cur, r.device,
                                           hc->d_attn_comp_score_full,
                                           opt.devices[0], comp_bytes, r.stream));
        }
        compressor_store_slots_kernel<<<
            (unsigned int)(((uint64_t)opt.slots * (uint64_t)comp_width +
                            block - 1) /
                           block),
            block, 0, r.stream>>>(
            r.d_attn_comp_kv_cur, r.d_attn_comp_score_cur,
            r.d_attn_comp_state_kv, r.d_attn_comp_state_score,
            hc->d_attn_compress_ape[layer], (uint32_t)opt.slots,
            (uint32_t)kHeadDim, (uint32_t)ratio, (uint32_t)opt.position,
            (uint32_t)comp_state_rows, (uint32_t)comp_state_width);
        if (emitted) {
            const uint32_t comp_row =
                r.attn_comp_rows_written_layers[layer] %
                (uint32_t)kBoundedCompRows;
            if (rank == 0) emitted_comp_row = comp_row;
            r.attn_comp_row_position_layers[layer][comp_row] = opt.position;
            r.attn_comp_row_loaded_layers[layer][comp_row] = false;
            if (fused_pool_norm_rope_round) {
                compressor_pool_norm_rope_round_emit_slots_kernel<<<
                    (unsigned int)opt.slots, 256, 0, r.stream>>>(
                    r.d_attn_comp_rows, r.d_attn_comp_state_kv,
                    r.d_attn_comp_state_score, hc->d_attn_compress_norm[layer],
                    (uint32_t)opt.slots, (uint32_t)kHeadDim, (uint32_t)ratio,
                    comp_row, (uint32_t)kBoundedCompRows,
                    (uint32_t)comp_state_rows, (uint32_t)comp_state_width,
                    1.0e-6f, (uint32_t)kRotaryDim,
                    (uint32_t)(opt.position + 1ull - (uint64_t)ratio),
                    kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                    comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                    kRopeYarnBetaSlow);
            } else if (fused_pool_norm) {
                compressor_pool_norm_emit_slots_kernel<<<
                    (unsigned int)opt.slots, 256, 0, r.stream>>>(
                    r.d_attn_comp_rows, r.d_attn_comp_state_kv,
                    r.d_attn_comp_state_score, hc->d_attn_compress_norm[layer],
                    (uint32_t)opt.slots, (uint32_t)kHeadDim, (uint32_t)ratio,
                    comp_row, (uint32_t)kBoundedCompRows,
                    (uint32_t)comp_state_rows, (uint32_t)comp_state_width,
                    1.0e-6f);
            } else {
                compressor_pool_emit_slots_kernel<<<
                    dim3((unsigned int)((kHeadDim + block - 1) / block),
                         (unsigned int)opt.slots, 1u),
                    block, 0, r.stream>>>(
                    r.d_attn_comp_rows, r.d_attn_comp_state_kv,
                    r.d_attn_comp_state_score, (uint32_t)opt.slots,
                    (uint32_t)kHeadDim, (uint32_t)ratio, comp_row,
                    (uint32_t)kBoundedCompRows, (uint32_t)comp_state_rows,
                    (uint32_t)comp_state_width);
                compressor_norm_emit_slots_kernel<<<(unsigned int)opt.slots, 256,
                                                    0, r.stream>>>(
                    r.d_attn_comp_rows, hc->d_attn_compress_norm[layer],
                    (uint32_t)opt.slots, (uint32_t)kHeadDim, comp_row,
                    (uint32_t)kBoundedCompRows, 1.0e-6f);
            }
            if (fused_pool_norm_rope_round) {
                // RoPE and F16 rounding were already applied by the fused emit.
            } else if (fused_rope_round) {
                rope_tail_round_comp_emit_slots_kernel<<<
                    (unsigned int)opt.slots, 256, 0, r.stream>>>(
                    r.d_attn_comp_rows, (uint32_t)opt.slots,
                    (uint32_t)kHeadDim, (uint32_t)kRotaryDim, comp_row,
                    (uint32_t)kBoundedCompRows,
                    (uint32_t)(opt.position + 1ull - (uint64_t)ratio),
                    kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                    comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                    kRopeYarnBetaSlow);
            } else {
                if (opt.true_ds4_attention_rope_gate) {
                    rope_tail_comp_emit_slots_kernel<<<
                        (unsigned int)opt.slots, 64, 0, r.stream>>>(
                        r.d_attn_comp_rows, (uint32_t)opt.slots,
                        (uint32_t)kHeadDim, (uint32_t)kRotaryDim, comp_row,
                        (uint32_t)kBoundedCompRows,
                        (uint32_t)(opt.position + 1ull - (uint64_t)ratio),
                        kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                        comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                        kRopeYarnBetaSlow);
                }
                round_comp_emit_slots_kernel<<<
                    (unsigned int)(((uint64_t)opt.slots * kHeadDim + block - 1) /
                                   block),
                    block, 0, r.stream>>>(
                    r.d_attn_comp_rows, (uint32_t)opt.slots, (uint32_t)kHeadDim,
                    comp_row, (uint32_t)kBoundedCompRows);
            }
            r.attn_comp_rows_written_layers[layer]++;
        }
        visible = std::max(
            visible,
            std::min(r.attn_comp_rows_written_layers[layer],
                     (uint32_t)kBoundedCompRows));
        CHECK_CUDA(cudaGetLastError());
    }
    if (!graph_event_order) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_state_emit_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }
    if (opt.true_ds4_attention_typed_kv_compressed_gate && emitted) {
        if (!rt) {
            std::fprintf(stderr,
                         "tp_ep_true_attention_typed_kv_compressed_failed\t"
                         "layer\t%d\treason\tmissing_tp_runtime\n",
                         layer);
            return 14;
        }
        char err[512] = {0};
        ds4_v100_tp_kv_row_view view;
        if (ds4_v100_tp_runtime_kv_row_view(
                rt, layer, 0, opt.position, DS4_V100_TP_KV_ROW_ATTN, &view, err,
                sizeof(err)) != 0) {
            std::fprintf(stderr,
                         "tp_ep_true_attention_typed_kv_compressed_view_failed\t"
                         "layer\t%d\t%s\n",
                         layer, err);
            return 15;
        }
        int current_store = 0;
        if (!opt.true_ds4_attention_typed_kv_skip_compressed_store_gate) {
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                const void *src[kGpus] = {};
                const size_t row_offset =
                    (size_t)emitted_comp_row * (size_t)kHeadDim;
                for (int rank = 0; rank < kGpus; ++rank) {
                    src[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                }
                if (ds4_v100_tp_runtime_kv_rows_store_f32_device(
                        rt, layer, 0, (uint32_t)opt.slots, opt.position,
                        DS4_V100_TP_KV_ROW_ATTN, src,
                        (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                        err, sizeof(err)) != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_compressed_store_failed\t"
                                 "layer\t%d\tmode\tbatched\t%s\n",
                                 layer, err);
                    return 16;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    const void *src[kGpus] = {};
                    const size_t row_offset =
                        ((size_t)slot * (size_t)kBoundedCompRows +
                         (size_t)emitted_comp_row) *
                        (size_t)kHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        src[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                    }
                    if (ds4_v100_tp_runtime_kv_row_store_f32_device(
                            rt, layer, slot, opt.position, DS4_V100_TP_KV_ROW_ATTN,
                            src, err, sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_compressed_store_failed\t"
                                     "layer\t%d\tslot\t%u\t%s\n",
                                     layer, slot, err);
                        return 16;
                    }
                }
            }
            current_store = 1;
        }
        sync_typed_kv_boundary(opt, ranks);
        int current_load = 0;
        if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
            current_store) {
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                void *dst[kGpus] = {};
                const size_t row_offset =
                    (size_t)emitted_comp_row * (size_t)kHeadDim;
                for (int rank = 0; rank < kGpus; ++rank) {
                    dst[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                }
                if (ds4_v100_tp_runtime_kv_rows_load_f32_device(
                        rt, layer, 0, (uint32_t)opt.slots, opt.position,
                        DS4_V100_TP_KV_ROW_ATTN, dst,
                        (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                        err, sizeof(err)) != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_compressed_load_failed\t"
                                 "layer\t%d\tmode\tbatched\t%s\n",
                                 layer, err);
                    return 17;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    void *dst[kGpus] = {};
                    const size_t row_offset =
                        ((size_t)slot * (size_t)kBoundedCompRows +
                         (size_t)emitted_comp_row) *
                        (size_t)kHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        dst[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                    }
                    if (ds4_v100_tp_runtime_kv_row_load_f32_device(
                            rt, layer, slot, opt.position, DS4_V100_TP_KV_ROW_ATTN,
                            dst, err, sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_compressed_load_failed\t"
                                     "layer\t%d\tslot\t%u\t%s\n",
                                     layer, slot, err);
                        return 17;
                    }
                }
            }
            current_load = 1;
        }
        sync_typed_kv_boundary(opt, ranks);
        if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate ||
            !current_store) {
            for (int rank = 0; rank < kGpus; ++rank) {
                ranks[rank].attn_comp_row_loaded_layers[layer][emitted_comp_row] = true;
                ranks[rank].attn_comp_row_loaded_position_layers[layer][emitted_comp_row] =
                    opt.position;
            }
        }
        if (!opt.true_ds4_attention_typed_kv_quiet_gate) {
            std::printf("tp_ep_true_attention_typed_kv_compressed\tlayer\t%d\t"
                        "slots\t%d\tratio\t%d\tposition\t%llu\t"
                        "bounded_row\t%u\tphysical_row\t%llu\tlogical_cols\t%u\t"
                        "logical_row_bytes\t%llu\trow_bytes_per_gpu\t%llu\t"
                        "current_store\t%d\tcurrent_load\t%d\tPASS\n",
                        layer, opt.slots, ratio, (unsigned long long)opt.position,
                        emitted_comp_row, (unsigned long long)view.physical_row,
                        view.logical_cols, (unsigned long long)view.logical_row_bytes,
                        (unsigned long long)view.row_bytes[0], current_store,
                        current_load);
        }
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_typed_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }

    TensorF32Stats index_q_stats;
    TensorF32Stats index_w_stats;
    TensorF32Stats index_kv_stats;
    TensorF32Stats index_gate_stats;
    if (opt.true_ds4_indexer_attention_gate && ratio == 4) {
        t_stage = std::chrono::steady_clock::now();
        if (ops->indexer_attn_q_b.cols != 1024 ||
            ops->indexer_attn_q_b.rows_per_gpu != (kIndexerHead * kIndexerHeadDim) / kGpus ||
            ops->indexer_proj.cols != kHidden ||
            ops->indexer_proj.rows_per_gpu != kIndexerHead / kGpus ||
            ops->indexer_compress_kv.cols != kHidden ||
            ops->indexer_compress_kv.rows_per_gpu != (2 * kIndexerHeadDim) / kGpus ||
            ops->indexer_compress_gate.cols != kHidden ||
            ops->indexer_compress_gate.rows_per_gpu != (2 * kIndexerHeadDim) / kGpus) {
            return 5;
        }
        const uint64_t q_a_elems = (uint64_t)opt.slots * 1024ull;
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            const float *current_src =
                direct_current_input_fill ? hc->d_attn_normed : r.d_current_full;
            if (!ops->indexer_attn_q_b.d_x_half[(size_t)rank] ||
                !ops->indexer_proj.d_x_half[(size_t)rank] ||
                !ops->indexer_compress_kv.d_x_half[(size_t)rank] ||
                !ops->indexer_compress_gate.d_x_half[(size_t)rank]) {
                return 6;
            }
            fill_dense_input_half_from_tensor_kernel<<<
                (unsigned int)((q_a_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->indexer_attn_q_b.d_x_half[(size_t)rank],
                             hc->d_q_a_normed, 1024u, (uint32_t)opt.slots);
            if (!fused_ratio4_current_fill) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->indexer_proj.d_x_half[(size_t)rank],
                                 current_src, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->indexer_compress_kv.d_x_half[(size_t)rank],
                                 current_src, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->indexer_compress_gate.d_x_half[(size_t)rank],
                                 current_src, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
            }
            CHECK_CUDA(cudaGetLastError());
        }
        if (dense_event_wait || graph_event_order) {
            if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 23;
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_input_fill_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
        if (launch_resident_f8_dense(opt, ops->indexer_attn_q_b, ranks) != 0 ||
            launch_resident_f8_dense(opt, ops->indexer_proj, ranks) != 0 ||
            launch_resident_f8_dense(opt, ops->indexer_compress_kv, ranks) != 0 ||
            launch_resident_f8_dense(opt, ops->indexer_compress_gate, ranks) != 0) {
            return 7;
        }
        if (graph_event_order) {
            if (enqueue_control_wait_after_dense_streams(
                    opt, ranks, control_stream) != 0) return 26;
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                cudaStream_t stream = ranks[rank].dense_stream ? ranks[rank].dense_stream
                                                               : ranks[rank].stream;
                CHECK_CUDA(cudaStreamSynchronize(stream));
                if (!skip_dense_stats) {
                merge_tensor_stats(&index_q_stats,
                                   collect_tensor_f32_stats(
                                       ops->indexer_attn_q_b.d_out[(size_t)rank],
                                       (size_t)opt.slots *
                                           (size_t)ops->indexer_attn_q_b.rows_per_gpu,
                                       stream));
                merge_tensor_stats(&index_w_stats,
                                   collect_tensor_f32_stats(
                                       ops->indexer_proj.d_out[(size_t)rank],
                                       (size_t)opt.slots *
                                           (size_t)ops->indexer_proj.rows_per_gpu,
                                       stream));
                merge_tensor_stats(&index_kv_stats,
                                   collect_tensor_f32_stats(
                                       ops->indexer_compress_kv.d_out[(size_t)rank],
                                       (size_t)opt.slots *
                                           (size_t)ops->indexer_compress_kv.rows_per_gpu,
                                       stream));
                merge_tensor_stats(&index_gate_stats,
                                   collect_tensor_f32_stats(
                                       ops->indexer_compress_gate.d_out[(size_t)rank],
                                       (size_t)opt.slots *
                                           (size_t)ops->indexer_compress_gate.rows_per_gpu,
                                       stream));
                }
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_dense_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
        if (!hc->d_indexer_q_full || !hc->d_indexer_w_full) return 13;
        if (!hc->d_index_comp_kv_full || !hc->d_index_comp_score_full ||
            !hc->d_indexer_compress_ape[layer] ||
            !hc->d_indexer_compress_norm[layer]) {
            return 11;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            gather_dense_shard_to_full_kernel<<<
                (unsigned int)(((uint64_t)opt.slots *
                                    (uint64_t)ops->indexer_attn_q_b.rows_per_gpu +
                                block - 1) /
                               block),
                block, 0, control_stream>>>(
                hc->d_indexer_q_full,
                ops->indexer_attn_q_b.d_out[(size_t)rank], rank,
                (uint32_t)ops->indexer_attn_q_b.rows_per_gpu,
                (uint32_t)(kIndexerHead * kIndexerHeadDim),
                (uint32_t)opt.slots);
            gather_dense_shard_to_full_kernel<<<
                (unsigned int)(((uint64_t)opt.slots *
                                    (uint64_t)ops->indexer_proj.rows_per_gpu +
                                block - 1) /
                               block),
                block, 0, control_stream>>>(
                hc->d_indexer_w_full,
                ops->indexer_proj.d_out[(size_t)rank], rank,
                (uint32_t)ops->indexer_proj.rows_per_gpu,
                (uint32_t)kIndexerHead, (uint32_t)opt.slots);
            gather_dense_shard_to_full_kernel<<<
                (unsigned int)(((uint64_t)opt.slots *
                                    (uint64_t)ops->indexer_compress_kv.rows_per_gpu +
                                block - 1) /
                               block),
                block, 0, control_stream>>>(
                hc->d_index_comp_kv_full,
                ops->indexer_compress_kv.d_out[(size_t)rank], rank,
                (uint32_t)ops->indexer_compress_kv.rows_per_gpu,
                (uint32_t)kIndexCompWidth, (uint32_t)opt.slots);
            gather_dense_shard_to_full_kernel<<<
                (unsigned int)(((uint64_t)opt.slots *
                                    (uint64_t)ops->indexer_compress_gate.rows_per_gpu +
                                block - 1) /
                               block),
                block, 0, control_stream>>>(
                hc->d_index_comp_score_full,
                ops->indexer_compress_gate.d_out[(size_t)rank], rank,
                (uint32_t)ops->indexer_compress_gate.rows_per_gpu,
                (uint32_t)kIndexCompWidth, (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        if (!graph_event_order) {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        if (opt.true_ds4_attention_rope_gate) {
            rope_tail_rows_kernel<<<
                (unsigned int)(opt.slots * kIndexerHead), 64, 0,
                control_stream>>>(
                hc->d_indexer_q_full, (uint32_t)(opt.slots * kIndexerHead),
                (uint32_t)kIndexerHeadDim, (uint32_t)kRotaryDim,
                (uint32_t)opt.position, kRopeOrigCtx, 0, kCompressRopeFreqBase,
                comp_freq_scale, comp_ext_factor, comp_attn_factor,
                kRopeYarnBetaFast, kRopeYarnBetaSlow);
            CHECK_CUDA(cudaGetLastError());
            if (!graph_event_order) {
                CHECK_CUDA(cudaDeviceSynchronize());
            }
        }
        if (graph_event_order) {
            if (enqueue_rank_streams_wait_after_control(
                    opt, ranks, control_stream) != 0) return 27;
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_gather_rope_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            if (!r.d_index_comp_kv_cur || !r.d_index_comp_score_cur ||
                !r.d_index_comp_state_kv || !r.d_index_comp_state_score ||
                !r.d_index_comp_rows || !r.d_indexer_scores ||
                !r.d_indexer_topk) {
                return 12;
            }
            const size_t index_bytes =
                (size_t)opt.slots * kIndexCompWidth * sizeof(float);
            if (rank == 0) {
                CHECK_CUDA(cudaMemcpyAsync(r.d_index_comp_kv_cur,
                                           hc->d_index_comp_kv_full,
                                           index_bytes, cudaMemcpyDeviceToDevice,
                                           r.stream));
                CHECK_CUDA(cudaMemcpyAsync(r.d_index_comp_score_cur,
                                           hc->d_index_comp_score_full,
                                           index_bytes, cudaMemcpyDeviceToDevice,
                                           r.stream));
            } else {
                CHECK_CUDA(cudaMemcpyPeerAsync(r.d_index_comp_kv_cur, r.device,
                                               hc->d_index_comp_kv_full,
                                               opt.devices[0], index_bytes,
                                               r.stream));
                CHECK_CUDA(cudaMemcpyPeerAsync(r.d_index_comp_score_cur, r.device,
                                               hc->d_index_comp_score_full,
                                               opt.devices[0], index_bytes,
                                               r.stream));
            }
            compressor_store_slots_kernel<<<
                (unsigned int)(((uint64_t)opt.slots * kIndexCompWidth +
                                block - 1) /
                               block),
                block, 0, r.stream>>>(
                r.d_index_comp_kv_cur, r.d_index_comp_score_cur,
                r.d_index_comp_state_kv, r.d_index_comp_state_score,
                hc->d_indexer_compress_ape[layer], (uint32_t)opt.slots,
                (uint32_t)kIndexerHeadDim, 4u, (uint32_t)opt.position,
                (uint32_t)kIndexCompStateRows, (uint32_t)kIndexCompWidth);
            if (emitted) {
                const uint32_t comp_row =
                    r.index_comp_rows_written_layers[layer] %
                    (uint32_t)kBoundedCompRows;
                r.index_comp_row_position_layers[layer][comp_row] = opt.position;
                r.index_comp_row_loaded_layers[layer][comp_row] = false;
                const uint32_t visible_after =
                    std::min(r.index_comp_rows_written_layers[layer] + 1u,
                             (uint32_t)kBoundedCompRows);
                if (fused_pool_norm_rope_round) {
                    compressor_pool_norm_rope_round_emit_slots_kernel<<<
                        (unsigned int)opt.slots, 256, 0, r.stream>>>(
                        r.d_index_comp_rows, r.d_index_comp_state_kv,
                        r.d_index_comp_state_score,
                        hc->d_indexer_compress_norm[layer],
                        (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, 4u,
                        comp_row, (uint32_t)kBoundedCompRows,
                        (uint32_t)kIndexCompStateRows,
                        (uint32_t)kIndexCompWidth, 1.0e-6f,
                        (uint32_t)kRotaryDim,
                        (uint32_t)(opt.position + 1ull - 4ull),
                        kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                        comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                        kRopeYarnBetaSlow);
                } else if (fused_pool_norm) {
                    compressor_pool_norm_emit_slots_kernel<<<
                        (unsigned int)opt.slots, 256, 0, r.stream>>>(
                        r.d_index_comp_rows, r.d_index_comp_state_kv,
                        r.d_index_comp_state_score,
                        hc->d_indexer_compress_norm[layer],
                        (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, 4u,
                        comp_row, (uint32_t)kBoundedCompRows,
                        (uint32_t)kIndexCompStateRows,
                        (uint32_t)kIndexCompWidth, 1.0e-6f);
                } else {
                    compressor_pool_emit_slots_kernel<<<
                        dim3((unsigned int)((kIndexerHeadDim + block - 1) / block),
                             (unsigned int)opt.slots, 1u),
                        block, 0, r.stream>>>(
                        r.d_index_comp_rows, r.d_index_comp_state_kv,
                        r.d_index_comp_state_score, (uint32_t)opt.slots,
                        (uint32_t)kIndexerHeadDim, 4u, comp_row,
                        (uint32_t)kBoundedCompRows,
                        (uint32_t)kIndexCompStateRows,
                        (uint32_t)kIndexCompWidth);
                    compressor_norm_emit_slots_kernel<<<(unsigned int)opt.slots,
                                                        256, 0, r.stream>>>(
                        r.d_index_comp_rows, hc->d_indexer_compress_norm[layer],
                        (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, comp_row,
                        (uint32_t)kBoundedCompRows, 1.0e-6f);
                }
                if (fused_pool_norm_rope_round) {
                    // RoPE and F16 rounding were already applied by the fused emit.
                } else if (fused_rope_round) {
                    rope_tail_round_comp_emit_slots_kernel<<<
                        (unsigned int)opt.slots, 256, 0, r.stream>>>(
                        r.d_index_comp_rows, (uint32_t)opt.slots,
                        (uint32_t)kIndexerHeadDim, (uint32_t)kRotaryDim,
                        comp_row, (uint32_t)kBoundedCompRows,
                        (uint32_t)(opt.position + 1ull - 4ull),
                        kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                        comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                        kRopeYarnBetaSlow);
                } else {
                    if (opt.true_ds4_attention_rope_gate) {
                        rope_tail_comp_emit_slots_kernel<<<
                            (unsigned int)opt.slots, 64, 0, r.stream>>>(
                            r.d_index_comp_rows, (uint32_t)opt.slots,
                            (uint32_t)kIndexerHeadDim, (uint32_t)kRotaryDim,
                            comp_row, (uint32_t)kBoundedCompRows,
                            (uint32_t)(opt.position + 1ull - 4ull),
                            kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                            comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                            kRopeYarnBetaSlow);
                    }
                    round_comp_emit_slots_kernel<<<
                        (unsigned int)(((uint64_t)opt.slots *
                                            kIndexerHeadDim +
                                        block - 1) /
                                       block),
                        block, 0, r.stream>>>(
                        r.d_index_comp_rows, (uint32_t)opt.slots,
                        (uint32_t)kIndexerHeadDim, comp_row,
                        (uint32_t)kBoundedCompRows);
                }
                if (rank == 0 && !opt.true_ds4_attention_typed_kv_indexer_gate) {
                    indexer_score_bounded_rows_slots_kernel<<<
                        (unsigned int)opt.slots, 256, 0, r.stream>>>(
                        r.d_indexer_scores, r.d_indexer_topk,
                        hc->d_indexer_q_full, hc->d_indexer_w_full,
                        r.d_index_comp_rows, (uint32_t)opt.slots,
                        visible_after, (uint32_t)kBoundedCompRows,
                        (uint32_t)kIndexerTopK,
                        1.0f / sqrtf((float)(kIndexerHead * kIndexerHeadDim)));
                } else if (!opt.true_ds4_attention_typed_kv_indexer_gate) {
                    seed_single_topk_kernel<<<(unsigned int)opt.slots, 256, 0,
                                               r.stream>>>(
                        r.d_indexer_scores, r.d_indexer_topk,
                        (uint32_t)opt.slots, (uint32_t)kIndexerTopK);
                }
                r.index_comp_rows_written_layers[layer]++;
            }
            CHECK_CUDA(cudaGetLastError());
        }
        if (!graph_event_order) {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_state_emit_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
        if (opt.true_ds4_attention_typed_kv_indexer_gate && emitted) {
            if (!rt) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_typed_kv_indexer_failed\t"
                             "layer\t%d\treason\tmissing_tp_runtime\n",
                             layer);
                return 18;
            }
            char err[512] = {0};
            ds4_v100_tp_kv_row_view view;
            if (ds4_v100_tp_runtime_kv_row_view(
                    rt, layer, 0, opt.position, DS4_V100_TP_KV_ROW_INDEXER,
                    &view, err, sizeof(err)) != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_typed_kv_indexer_view_failed\t"
                             "layer\t%d\t%s\n",
                             layer, err);
                return 19;
            }
            const uint32_t bounded_row =
                (ranks[0].index_comp_rows_written_layers[layer] +
                 (uint32_t)kBoundedCompRows - 1u) %
                (uint32_t)kBoundedCompRows;
            const uint32_t visible_after =
                std::min(ranks[0].index_comp_rows_written_layers[layer],
                         (uint32_t)kBoundedCompRows);
            int current_store = 0;
            if (!opt.true_ds4_attention_typed_kv_skip_indexer_store_gate) {
                if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                    const void *src[kGpus] = {};
                    const size_t row_offset =
                        (size_t)bounded_row * (size_t)kIndexerHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        src[rank] = ranks[rank].d_index_comp_rows + row_offset;
                    }
                    if (ds4_v100_tp_runtime_kv_rows_store_f32_device(
                            rt, layer, 0, (uint32_t)opt.slots, opt.position,
                            DS4_V100_TP_KV_ROW_INDEXER, src,
                            (uint64_t)kBoundedCompRows * (uint64_t)kIndexerHeadDim,
                            err, sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_indexer_store_failed\t"
                                     "layer\t%d\tmode\tbatched\t%s\n",
                                     layer, err);
                        return 20;
                    }
                } else {
                    for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                        const void *src[kGpus] = {};
                        const size_t row_offset =
                            ((size_t)slot * (size_t)kBoundedCompRows +
                             (size_t)bounded_row) *
                            (size_t)kIndexerHeadDim;
                        for (int rank = 0; rank < kGpus; ++rank) {
                            src[rank] = ranks[rank].d_index_comp_rows + row_offset;
                        }
                        if (ds4_v100_tp_runtime_kv_row_store_f32_device(
                                rt, layer, slot, opt.position,
                                DS4_V100_TP_KV_ROW_INDEXER, src, err,
                                sizeof(err)) != 0) {
                            std::fprintf(stderr,
                                         "tp_ep_true_attention_typed_kv_indexer_store_failed\t"
                                         "layer\t%d\tslot\t%u\t%s\n",
                                         layer, slot, err);
                            return 20;
                        }
                    }
                }
                current_store = 1;
            }
            sync_typed_kv_boundary(opt, ranks);
            int current_load = 0;
            if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
                current_store) {
                if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                    void *dst[kGpus] = {};
                    const size_t row_offset =
                        (size_t)bounded_row * (size_t)kIndexerHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        dst[rank] = ranks[rank].d_index_comp_rows + row_offset;
                    }
                    if (ds4_v100_tp_runtime_kv_rows_load_f32_device(
                            rt, layer, 0, (uint32_t)opt.slots, opt.position,
                            DS4_V100_TP_KV_ROW_INDEXER, dst,
                            (uint64_t)kBoundedCompRows * (uint64_t)kIndexerHeadDim,
                            err, sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_indexer_load_failed\t"
                                     "layer\t%d\tmode\tbatched\t%s\n",
                                     layer, err);
                        return 21;
                    }
                } else {
                    for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                        void *dst[kGpus] = {};
                        const size_t row_offset =
                            ((size_t)slot * (size_t)kBoundedCompRows +
                             (size_t)bounded_row) *
                            (size_t)kIndexerHeadDim;
                        for (int rank = 0; rank < kGpus; ++rank) {
                            dst[rank] = ranks[rank].d_index_comp_rows + row_offset;
                        }
                        if (ds4_v100_tp_runtime_kv_row_load_f32_device(
                                rt, layer, slot, opt.position,
                                DS4_V100_TP_KV_ROW_INDEXER, dst, err,
                                sizeof(err)) != 0) {
                            std::fprintf(stderr,
                                         "tp_ep_true_attention_typed_kv_indexer_load_failed\t"
                                         "layer\t%d\tslot\t%u\t%s\n",
                                         layer, slot, err);
                            return 21;
                        }
                    }
                }
                current_load = 1;
            }
            sync_typed_kv_boundary(opt, ranks);
            if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate ||
                !current_store) {
                for (int rank = 0; rank < kGpus; ++rank) {
                    ranks[rank].index_comp_row_loaded_layers[layer][bounded_row] = true;
                    ranks[rank].index_comp_row_loaded_position_layers[layer][bounded_row] =
                        opt.position;
                }
            }
            CHECK_CUDA(cudaSetDevice(ranks[0].device));
            indexer_score_bounded_rows_slots_kernel<<<
                (unsigned int)opt.slots, 256, 0, ranks[0].stream>>>(
                ranks[0].d_indexer_scores, ranks[0].d_indexer_topk,
                hc->d_indexer_q_full, hc->d_indexer_w_full,
                ranks[0].d_index_comp_rows, (uint32_t)opt.slots, visible_after,
                (uint32_t)kBoundedCompRows, (uint32_t)kIndexerTopK,
                1.0f / sqrtf((float)(kIndexerHead * kIndexerHeadDim)));
            CHECK_CUDA(cudaGetLastError());
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                seed_single_topk_kernel<<<(unsigned int)opt.slots, 256, 0,
                                           ranks[rank].stream>>>(
                    ranks[rank].d_indexer_scores, ranks[rank].d_indexer_topk,
                    (uint32_t)opt.slots, (uint32_t)kIndexerTopK);
                CHECK_CUDA(cudaGetLastError());
            }
            if (graph_event_order) {
                CHECK_CUDA(cudaSetDevice(ranks[0].device));
                if (!ranks[0].stream_done) return 28;
                CHECK_CUDA(cudaEventRecord(ranks[0].stream_done,
                                           ranks[0].stream));
                for (int rank = 1; rank < kGpus; ++rank) {
                    CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                    CHECK_CUDA(cudaStreamWaitEvent(ranks[rank].stream,
                                                   ranks[0].stream_done, 0));
                }
            } else {
                for (int rank = 0; rank < kGpus; ++rank) {
                    CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                    CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
                }
            }
            if (!opt.true_ds4_attention_typed_kv_quiet_gate) {
                std::printf("tp_ep_true_attention_typed_kv_indexer\tlayer\t%d\t"
                            "slots\t%d\tratio\t%d\tposition\t%llu\t"
                            "bounded_row\t%u\tvisible_rows\t%u\tphysical_row\t%llu\t"
                            "logical_cols\t%u\tlogical_row_bytes\t%llu\t"
                            "row_bytes_per_gpu\t%llu\tcurrent_store\t%d\t"
                            "current_load\t%d\tPASS\n",
                            layer, opt.slots, ratio,
                            (unsigned long long)opt.position, bounded_row,
                            visible_after, (unsigned long long)view.physical_row,
                            view.logical_cols,
                            (unsigned long long)view.logical_row_bytes,
                            (unsigned long long)view.row_bytes[0], current_store,
                            current_load);
            }
        }
        if (emitted && ranks[0].d_indexer_topk) {
            const size_t topk_bytes =
                (size_t)opt.slots * kIndexerTopK * sizeof(uint32_t);
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaMemcpyPeerAsync(ranks[rank].d_indexer_topk,
                                               ranks[rank].device,
                                               ranks[0].d_indexer_topk,
                                               ranks[0].device,
                                               topk_bytes,
                                               ranks[rank].stream));
            }
            if (!graph_event_order) {
                for (int rank = 1; rank < kGpus; ++rank) {
                    CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                    CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
                }
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_typed_score_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
    }

    const auto diff_start = std::chrono::steady_clock::now();
    const int diff_rc = run_true_ds4_compressed_reference_diff_gate(
        opt, hc, ranks, layer, ratio, comp_width, emitted, emitted_comp_row,
        visible);
    if (diff_rc != 0) return diff_rc;
    reference_diff_ms =
        elapsed_ms(diff_start, std::chrono::steady_clock::now());
    const auto shift_start = std::chrono::steady_clock::now();
    if (emitted && ratio == 4) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            compressor_shift_ratio4_slots_kernel<<<
                (unsigned int)(((uint64_t)opt.slots * 4ull *
                                    (uint64_t)comp_width +
                                block - 1) /
                               block),
                block, 0, r.stream>>>(
                r.d_attn_comp_state_kv, r.d_attn_comp_state_score,
                (uint32_t)opt.slots, (uint32_t)comp_width,
                (uint32_t)comp_state_rows, (uint32_t)comp_state_width);
            if (opt.true_ds4_indexer_attention_gate && r.d_index_comp_state_kv &&
                r.d_index_comp_state_score) {
                compressor_shift_ratio4_slots_kernel<<<
                    (unsigned int)(((uint64_t)opt.slots * 4ull *
                                        (uint64_t)kIndexCompWidth +
                                    block - 1) /
                                   block),
                    block, 0, r.stream>>>(
                    r.d_index_comp_state_kv, r.d_index_comp_state_score,
                    (uint32_t)opt.slots, (uint32_t)kIndexCompWidth,
                    (uint32_t)kIndexCompStateRows,
                    (uint32_t)kIndexCompWidth);
            }
            CHECK_CUDA(cudaGetLastError());
        }
        if (!graph_event_order) {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }
    ratio_shift_ms =
        elapsed_ms(shift_start, std::chrono::steady_clock::now());

    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    std::printf("tp_ep_compressed_kv_projection\tlayer\t%d\tslots\t%d\t"
                "ratio\t%d\temitted_compressed_rows\t%u\t"
                "visible_compressed_rows\t%u\tindexer_topk_count\t%u\t"
                "attn_comp_width\t%d\tattn_kv_max\t%.9g\tattn_kv_bad\t%d\t"
                "attn_gate_max\t%.9g\tattn_gate_bad\t%d\t"
                "index_q_max\t%.9g\tindex_q_bad\t%d\t"
                "index_w_max\t%.9g\tindex_w_bad\t%d\t"
                "index_kv_max\t%.9g\tindex_kv_bad\t%d\t"
                "index_gate_max\t%.9g\tindex_gate_bad\t%d\t"
                "attn_input_fill_ms\t%.6f\tattn_dense_ms\t%.6f\t"
                "attn_gather_ms\t%.6f\tattn_state_emit_ms\t%.6f\t"
                "attn_typed_ms\t%.6f\tindexer_input_fill_ms\t%.6f\t"
                "indexer_dense_ms\t%.6f\tindexer_gather_rope_ms\t%.6f\t"
                "indexer_state_emit_ms\t%.6f\tindexer_typed_score_ms\t%.6f\t"
                "reference_diff_ms\t%.6f\tratio_shift_ms\t%.6f\t"
                "direct_input_fill\t%d\tdense_event_wait\t%d\t"
                "skip_dense_stats\t%d\t"
                "fused_attn_input_fill\t%d\t"
                "fused_input_fill\t%d\tfused_rope_round\t%d\t"
                "fused_pool_norm\t%d\tfused_pool_norm_rope_round\t%d\t"
                "ms\t%.6f\tPASS\n",
                layer, opt.slots, ratio, emitted, visible, indexer_topk,
                comp_width, attn_kv_stats.max_abs, attn_kv_stats.finite_bad,
                attn_gate_stats.max_abs, attn_gate_stats.finite_bad,
                index_q_stats.max_abs, index_q_stats.finite_bad,
                index_w_stats.max_abs, index_w_stats.finite_bad,
                index_kv_stats.max_abs, index_kv_stats.finite_bad,
                index_gate_stats.max_abs, index_gate_stats.finite_bad,
                attn_input_fill_ms, attn_dense_ms, attn_gather_ms,
                attn_state_emit_ms, attn_typed_ms, indexer_input_fill_ms,
                indexer_dense_ms, indexer_gather_rope_ms,
                indexer_state_emit_ms, indexer_typed_score_ms,
                reference_diff_ms, ratio_shift_ms,
                direct_current_input_fill ? 1 : 0,
                dense_event_wait ? 1 : 0,
                skip_dense_stats ? 1 : 0,
                fused_attn_current_fill ? 1 : 0,
                fused_ratio4_current_fill ? 1 : 0,
                fused_rope_round ? 1 : 0,
                fused_pool_norm ? 1 : 0,
                fused_pool_norm_rope_round ? 1 : 0, ms);
    return (!skip_dense_stats &&
            (attn_kv_stats.finite_bad || attn_gate_stats.finite_bad ||
            index_q_stats.finite_bad || index_w_stats.finite_bad ||
             index_kv_stats.finite_bad || index_gate_stats.finite_bad)) ? 8 : 0;
}

int run_true_ds4_attention_state_update(const Options &opt,
                                        SharedHcControls *hc,
                                        const LayerDenseOps *ops,
                                        RankState ranks[kGpus],
                                        ds4_v100_tp_runtime *rt,
                                        int layer) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    if (!hc->d_kv_normed ||
        ops->attn_q_b.rows_per_gpu != kLocalHeads * kHeadDim) {
        return 2;
    }

    const auto start = std::chrono::steady_clock::now();
    const int block = 256;
    const uint32_t raw_row = (uint32_t)(opt.position % kRawSwaRows);
    const uint64_t kv_elems = (uint64_t)opt.slots * (uint64_t)kHeadDim;
    const uint64_t raw_elems =
        (uint64_t)opt.slots * (uint64_t)kRawSwaRows * (uint64_t)kHeadDim;
    const int ratio = ds4_layer_ratio(layer);
    const bool compressed = ratio != 0;
    const float freq_base =
        compressed ? kCompressRopeFreqBase : kRopeFreqBase;
    const float freq_scale =
        compressed && kRopeScaleFactor > 0.0f ? 1.0f / kRopeScaleFactor : 1.0f;
    const float ext_factor =
        compressed && kRopeScaleFactor > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_kv_full || !r.d_attn_raw_swa ||
            !ops->attn_q_b.d_out[(size_t)rank]) {
            return 3;
        }
        head_rms_norm_local_heads_kernel<<<
            (unsigned int)(opt.slots * kLocalHeads), 256, 0,
            r.dense_stream ? r.dense_stream : r.stream>>>(
            ops->attn_q_b.d_out[(size_t)rank], (uint32_t)opt.slots,
            (uint32_t)kLocalHeads, (uint32_t)kHeadDim, 1.0e-6f);
        CHECK_CUDA(cudaGetLastError());
        if (opt.true_ds4_attention_rope_gate) {
            rope_tail_rows_kernel<<<
                (unsigned int)(opt.slots * kLocalHeads), 64, 0,
                r.dense_stream ? r.dense_stream : r.stream>>>(
                ops->attn_q_b.d_out[(size_t)rank],
                (uint32_t)(opt.slots * kLocalHeads), (uint32_t)kHeadDim,
                (uint32_t)kRotaryDim, (uint32_t)opt.position,
                compressed ? kRopeOrigCtx : 0u, 0, freq_base, freq_scale,
                ext_factor, attn_factor, kRopeYarnBetaFast,
                kRopeYarnBetaSlow);
            CHECK_CUDA(cudaGetLastError());
        }
        if (rank == 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_attn_kv_full, hc->d_kv_normed,
                                       (size_t)kv_elems * sizeof(float),
                                       cudaMemcpyDeviceToDevice, r.stream));
        } else {
            CHECK_CUDA(cudaMemcpyPeerAsync(r.d_attn_kv_full, r.device,
                                           hc->d_kv_normed, opt.devices[0],
                                           (size_t)kv_elems * sizeof(float),
                                           r.stream));
        }
        if (opt.true_ds4_attention_rope_gate) {
            rope_tail_rows_kernel<<<
                (unsigned int)opt.slots, 64, 0, r.stream>>>(
                r.d_attn_kv_full, (uint32_t)opt.slots, (uint32_t)kHeadDim,
                (uint32_t)kRotaryDim, (uint32_t)opt.position,
                compressed ? kRopeOrigCtx : 0u, 0, freq_base, freq_scale,
                ext_factor, attn_factor, kRopeYarnBetaFast,
                kRopeYarnBetaSlow);
            CHECK_CUDA(cudaGetLastError());
        }
        if (!opt.true_ds4_attention_typed_kv_raw_gate ||
            opt.true_ds4_attention_typed_kv_skip_current_load_gate) {
            kv_fp8_round_store_raw_swa_kernel<<<
                (unsigned int)((kv_elems + block - 1) / block), block, 0,
                r.stream>>>(
                r.d_attn_raw_swa, r.d_attn_kv_full, (uint32_t)opt.slots,
                (uint32_t)kRawSwaRows, raw_row, (uint32_t)kHeadDim,
                (uint32_t)kRotaryDim);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    if (opt.decode_cudagraph_gate) {
        if (enqueue_rank_streams_wait_after_dense_streams(ranks) != 0) {
            return 8;
        }
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            if (ranks[rank].dense_stream) {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].dense_stream));
            }
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    if (opt.true_ds4_attention_typed_kv_raw_gate) {
        if (!rt) {
            std::fprintf(stderr,
                         "tp_ep_true_attention_typed_kv_raw_failed\tlayer\t%d\t"
                         "reason\tmissing_tp_runtime\n",
                         layer);
            return 4;
        }
        char err[512] = {0};
        ds4_v100_tp_kv_row_view view;
        if (ds4_v100_tp_runtime_kv_row_view(
                rt, layer, 0, opt.position, DS4_V100_TP_KV_ROW_ATTN_RAW, &view,
                err, sizeof(err)) != 0) {
            std::fprintf(stderr,
                         "tp_ep_true_attention_typed_kv_raw_view_failed\tlayer\t%d\t%s\n",
                         layer, err);
            return 5;
        }
        int current_store = 0;
        if (!opt.true_ds4_attention_typed_kv_skip_raw_store_gate) {
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                const void *src[kGpus] = {};
                for (int rank = 0; rank < kGpus; ++rank) {
                    src[rank] = ranks[rank].d_attn_kv_full;
                }
                if (ds4_v100_tp_runtime_kv_rows_store_f32_device(
                        rt, layer, 0, (uint32_t)opt.slots, opt.position,
                        DS4_V100_TP_KV_ROW_ATTN_RAW, src, (uint64_t)kHeadDim,
                        err, sizeof(err)) != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_raw_store_failed\t"
                                 "layer\t%d\tmode\tbatched\t%s\n",
                                 layer, err);
                    return 6;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    const void *src[kGpus] = {};
                    for (int rank = 0; rank < kGpus; ++rank) {
                        src[rank] = ranks[rank].d_attn_kv_full +
                                    (size_t)slot * (size_t)kHeadDim;
                    }
                    if (ds4_v100_tp_runtime_kv_row_store_f32_device(
                            rt, layer, slot, opt.position,
                            DS4_V100_TP_KV_ROW_ATTN_RAW, src, err,
                            sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_raw_store_failed\t"
                                     "layer\t%d\tslot\t%u\t%s\n",
                                     layer, slot, err);
                        return 6;
                    }
                }
            }
            current_store = 1;
        }
        sync_typed_kv_boundary(opt, ranks);
        int current_load = 0;
        if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
            current_store) {
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                void *dst[kGpus] = {};
                const size_t row_offset =
                    (size_t)raw_row * (size_t)kHeadDim;
                for (int rank = 0; rank < kGpus; ++rank) {
                    dst[rank] = ranks[rank].d_attn_raw_swa + row_offset;
                }
                if (ds4_v100_tp_runtime_kv_rows_load_f32_device(
                        rt, layer, 0, (uint32_t)opt.slots, opt.position,
                        DS4_V100_TP_KV_ROW_ATTN_RAW, dst,
                        (uint64_t)kRawSwaRows * (uint64_t)kHeadDim,
                        err, sizeof(err)) != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_raw_load_failed\t"
                                 "layer\t%d\tmode\tbatched\t%s\n",
                                 layer, err);
                    return 7;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    void *dst[kGpus] = {};
                    const size_t row_offset =
                        ((size_t)slot * (size_t)kRawSwaRows + (size_t)raw_row) *
                        (size_t)kHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        dst[rank] = ranks[rank].d_attn_raw_swa + row_offset;
                    }
                    if (ds4_v100_tp_runtime_kv_row_load_f32_device(
                            rt, layer, slot, opt.position,
                            DS4_V100_TP_KV_ROW_ATTN_RAW, dst, err,
                            sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_raw_load_failed\t"
                                     "layer\t%d\tslot\t%u\t%s\n",
                                     layer, slot, err);
                        return 7;
                    }
                }
            }
            current_load = 1;
        }
        sync_typed_kv_boundary(opt, ranks);
        if (!opt.true_ds4_attention_typed_kv_quiet_gate) {
            std::printf("tp_ep_true_attention_typed_kv_raw\tlayer\t%d\tslots\t%d\t"
                        "position\t%llu\tphysical_row\t%llu\traw_row\t%u\tlogical_cols\t%u\t"
                        "logical_row_bytes\t%llu\trow_bytes_per_gpu\t%llu\t"
                        "current_store\t%d\tcurrent_load\t%d\tPASS\n",
                        layer, opt.slots, (unsigned long long)opt.position,
                        (unsigned long long)view.physical_row, raw_row,
                        view.logical_cols, (unsigned long long)view.logical_row_bytes,
                        (unsigned long long)view.row_bytes[0], current_store,
                        current_load);
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    if (!opt.decode_cudagraph_gate && layer <= 2) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("true_attn_q_heads_normed_shard", layer, rank,
                                 ops->attn_q_b.d_out[(size_t)rank],
                                 (size_t)opt.slots * ops->attn_q_b.rows_per_gpu,
                                 ranks[rank].dense_stream ? ranks[rank].dense_stream
                                                          : ranks[rank].stream);
        }
        CHECK_CUDA(cudaSetDevice(ranks[0].device));
        log_tensor_f32_stats("true_attn_raw_swa_rank0", layer, 0,
                             ranks[0].d_attn_raw_swa, (size_t)raw_elems,
                             ranks[0].stream);
    }
    if (opt.true_ds4_attention_saturation_audit_gate) {
        TensorF32Stats q_heads;
        TensorF32Stats kv_rope;
        TensorF32Stats raw_row_stats;
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            const cudaStream_t q_stream =
                ranks[rank].dense_stream ? ranks[rank].dense_stream
                                         : ranks[rank].stream;
            merge_tensor_stats(
                &q_heads,
                collect_tensor_f32_stats(
                    ops->attn_q_b.d_out[(size_t)rank],
                    (size_t)opt.slots * ops->attn_q_b.rows_per_gpu,
                    q_stream));
            merge_tensor_stats(
                &kv_rope,
                collect_tensor_f32_stats(ranks[rank].d_attn_kv_full,
                                         (size_t)kv_elems,
                                         ranks[rank].stream));
            merge_tensor_stats(
                &raw_row_stats,
                collect_raw_swa_row_stats(ranks[rank].d_attn_raw_swa,
                                          (uint32_t)opt.slots,
                                          (uint32_t)kRawSwaRows, raw_row,
                                          (uint32_t)kHeadDim,
                                          ranks[rank].stream));
        }
        std::printf("tp_ep_true_attention_saturation_state\tlayer\t%d\t"
                    "slots\t%d\traw_row\t%u\tq_heads_post_rope_max\t%.9g\t"
                    "q_heads_post_rope_bad\t%d\tkv_post_rope_max\t%.9g\t"
                    "kv_post_rope_bad\t%d\traw_swa_row_max\t%.9g\t"
                    "raw_swa_row_bad\t%d\tPASS\n",
                    layer, opt.slots, raw_row, q_heads.max_abs,
                    q_heads.finite_bad, kv_rope.max_abs, kv_rope.finite_bad,
                    raw_row_stats.max_abs, raw_row_stats.finite_bad);
    }
    if (opt.true_ds4_attention_rope_gate) {
        std::printf("tp_ep_true_attention_rope\tlayer\t%d\tslots\t%d\t"
                    "local_heads\t%d\thead_dim\t%d\trotary_dim\t%d\t"
                    "freq_base\t%.1f\tfreq_scale\t%.9f\tposition\t%llu\tPASS\n",
                    layer, opt.slots, kLocalHeads, kHeadDim, kRotaryDim,
                    freq_base, freq_scale, (unsigned long long)opt.position);
    }
    std::printf("tp_ep_true_attention_state_update\tlayer\t%d\tslots\t%d\t"
                "local_heads\t%d\thead_dim\t%d\traw_rows\t%d\traw_row\t%u\t"
                "kv_width\t%d\tms\t%.6f\tPASS\n",
                layer, opt.slots, kLocalHeads, kHeadDim, kRawSwaRows, raw_row,
                kHeadDim, ms);
    return 0;
}

int run_true_ds4_attention_typed_kv_history_load(const Options &opt,
                                                 SharedHcControls *hc,
                                                 RankState ranks[kGpus],
                                                 ds4_v100_tp_runtime *rt,
                                                 int layer) {
    if (!opt.true_ds4_attention_typed_kv_history_gate) return 0;
    if (!rt || layer < 0 || layer >= 43) return 1;
    const int ratio = ds4_layer_ratio(layer);
    if (ratio == 0) return 0;

    const uint32_t visible_attn =
        std::min(ranks[0].attn_comp_rows_written_layers[layer],
                 (uint32_t)kBoundedCompRows);
    char err[512] = {0};
    int loaded_attn = 0;
    int reloaded_attn = 0;
    for (uint32_t row = 0; row < visible_attn; ++row) {
        const uint64_t pos = ranks[0].attn_comp_row_position_layers[layer][row];
        if (opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
            pos == opt.position) {
            loaded_attn++;
            continue;
        }
        if (ranks[0].attn_comp_row_loaded_layers[layer][row] &&
            ranks[0].attn_comp_row_loaded_position_layers[layer][row] == pos) {
            loaded_attn++;
            continue;
        }
        if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
            void *dst[kGpus] = {};
            const size_t row_offset =
                (size_t)row * (size_t)kHeadDim;
            for (int rank = 0; rank < kGpus; ++rank) {
                dst[rank] = ranks[rank].d_attn_comp_rows + row_offset;
            }
            if (ds4_v100_tp_runtime_kv_rows_load_f32_device(
                    rt, layer, 0, (uint32_t)opt.slots, pos,
                    DS4_V100_TP_KV_ROW_ATTN, dst,
                    (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                    err, sizeof(err)) != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_typed_kv_history_attn_load_failed\t"
                             "layer\t%d\trow\t%u\tmode\tbatched\tposition\t%llu\t%s\n",
                             layer, row, (unsigned long long)pos, err);
                return 2;
            }
        } else {
            for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                void *dst[kGpus] = {};
                const size_t row_offset =
                    ((size_t)slot * (size_t)kBoundedCompRows + (size_t)row) *
                    (size_t)kHeadDim;
                for (int rank = 0; rank < kGpus; ++rank) {
                    dst[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                }
                if (ds4_v100_tp_runtime_kv_row_load_f32_device(
                        rt, layer, slot, pos, DS4_V100_TP_KV_ROW_ATTN, dst, err,
                        sizeof(err)) != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_history_attn_load_failed\t"
                                 "layer\t%d\trow\t%u\tslot\t%u\tposition\t%llu\t%s\n",
                                 layer, row, slot, (unsigned long long)pos, err);
                    return 2;
                }
            }
        }
        loaded_attn++;
        reloaded_attn++;
        for (int rank = 0; rank < kGpus; ++rank) {
            ranks[rank].attn_comp_row_loaded_layers[layer][row] = true;
            ranks[rank].attn_comp_row_loaded_position_layers[layer][row] = pos;
        }
    }
    sync_typed_kv_boundary(opt, ranks);

    int loaded_indexer = 0;
    int reloaded_indexer = 0;
    if (opt.true_ds4_indexer_attention_gate && ratio == 4 && visible_attn > 0) {
        if (!hc || !hc->initialized || !hc->d_indexer_q_full ||
            !hc->d_indexer_w_full || !ranks[0].d_indexer_scores ||
            !ranks[0].d_indexer_topk) {
            return 3;
        }
        const uint32_t visible_index =
            std::min(ranks[0].index_comp_rows_written_layers[layer],
                     (uint32_t)kBoundedCompRows);
        for (uint32_t row = 0; row < visible_index; ++row) {
            const uint64_t pos = ranks[0].index_comp_row_position_layers[layer][row];
            if (opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
                pos == opt.position) {
                loaded_indexer++;
                continue;
            }
            if (ranks[0].index_comp_row_loaded_layers[layer][row] &&
                ranks[0].index_comp_row_loaded_position_layers[layer][row] == pos) {
                loaded_indexer++;
                continue;
            }
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                void *dst[kGpus] = {};
                const size_t row_offset =
                    (size_t)row * (size_t)kIndexerHeadDim;
                for (int rank = 0; rank < kGpus; ++rank) {
                    dst[rank] = ranks[rank].d_index_comp_rows + row_offset;
                }
                if (ds4_v100_tp_runtime_kv_rows_load_f32_device(
                        rt, layer, 0, (uint32_t)opt.slots, pos,
                        DS4_V100_TP_KV_ROW_INDEXER, dst,
                        (uint64_t)kBoundedCompRows * (uint64_t)kIndexerHeadDim,
                        err, sizeof(err)) != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_history_indexer_load_failed\t"
                                 "layer\t%d\trow\t%u\tmode\tbatched\tposition\t%llu\t%s\n",
                                 layer, row, (unsigned long long)pos, err);
                    return 4;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    void *dst[kGpus] = {};
                    const size_t row_offset =
                        ((size_t)slot * (size_t)kBoundedCompRows + (size_t)row) *
                        (size_t)kIndexerHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        dst[rank] = ranks[rank].d_index_comp_rows + row_offset;
                    }
                    if (ds4_v100_tp_runtime_kv_row_load_f32_device(
                            rt, layer, slot, pos, DS4_V100_TP_KV_ROW_INDEXER, dst,
                            err, sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_history_indexer_load_failed\t"
                                     "layer\t%d\trow\t%u\tslot\t%u\tposition\t%llu\t%s\n",
                                     layer, row, slot, (unsigned long long)pos, err);
                        return 4;
                    }
                }
            }
            loaded_indexer++;
            reloaded_indexer++;
            for (int rank = 0; rank < kGpus; ++rank) {
                ranks[rank].index_comp_row_loaded_layers[layer][row] = true;
                ranks[rank].index_comp_row_loaded_position_layers[layer][row] = pos;
            }
        }
        sync_typed_kv_boundary(opt, ranks);
        CHECK_CUDA(cudaSetDevice(ranks[0].device));
        indexer_score_bounded_rows_slots_kernel<<<
            (unsigned int)opt.slots, 256, 0, ranks[0].stream>>>(
            ranks[0].d_indexer_scores, ranks[0].d_indexer_topk,
            hc->d_indexer_q_full, hc->d_indexer_w_full,
            ranks[0].d_index_comp_rows, (uint32_t)opt.slots, visible_index,
            (uint32_t)kBoundedCompRows, (uint32_t)kIndexerTopK,
            1.0f / sqrtf((float)(kIndexerHead * kIndexerHeadDim)));
        CHECK_CUDA(cudaGetLastError());
        for (int rank = 1; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            seed_single_topk_kernel<<<(unsigned int)opt.slots, 256, 0,
                                       ranks[rank].stream>>>(
                ranks[rank].d_indexer_scores, ranks[rank].d_indexer_topk,
                (uint32_t)opt.slots, (uint32_t)kIndexerTopK);
            CHECK_CUDA(cudaGetLastError());
        }
        if (opt.decode_cudagraph_gate) {
            CHECK_CUDA(cudaSetDevice(ranks[0].device));
            if (!ranks[0].stream_done) return 5;
            CHECK_CUDA(cudaEventRecord(ranks[0].stream_done, ranks[0].stream));
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamWaitEvent(ranks[rank].stream,
                                               ranks[0].stream_done, 0));
            }
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
        const size_t topk_bytes = (size_t)opt.slots * kIndexerTopK * sizeof(uint32_t);
        for (int rank = 1; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaMemcpyPeerAsync(ranks[rank].d_indexer_topk,
                                           ranks[rank].device,
                                           ranks[0].d_indexer_topk,
                                           ranks[0].device,
                                           topk_bytes,
                                           ranks[rank].stream));
        }
        if (!opt.decode_cudagraph_gate) {
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }

    if (!opt.true_ds4_attention_typed_kv_quiet_gate) {
        std::printf("tp_ep_true_attention_typed_kv_history\tlayer\t%d\tslots\t%d\t"
                    "ratio\t%d\tvisible_attn_rows\t%u\tloaded_attn_rows\t%d\t"
                    "loaded_indexer_rows\t%d\treloaded_attn_rows\t%d\t"
                    "reloaded_indexer_rows\t%d\tPASS\n",
                    layer, opt.slots, ratio, visible_attn, loaded_attn,
                    loaded_indexer, reloaded_attn, reloaded_indexer);
    }
    return 0;
}

int run_true_ds4_attention_raw_read(const Options &opt,
                                    SharedHcControls *hc,
                                    const LayerDenseOps *ops,
                                    RankState ranks[kGpus],
                                    int layer) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    if (!hc->d_attn_sinks[layer] ||
        ops->attn_q_b.rows_per_gpu != kLocalHeads * kHeadDim) {
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    const uint32_t raw_row = (uint32_t)(opt.position % kRawSwaRows);
    const uint64_t heads_elems =
        (uint64_t)opt.slots * (uint64_t)kLocalHeads * (uint64_t)kHeadDim;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_raw_swa || !r.d_attn_sinks || !r.d_attn_heads ||
            !ops->attn_q_b.d_out[(size_t)rank]) {
            return 3;
        }
        const size_t sinks_offset = (size_t)rank * (size_t)kLocalHeads;
        if (rank == 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_attn_sinks,
                                       hc->d_attn_sinks[layer] + sinks_offset,
                                       (size_t)kLocalHeads * sizeof(float),
                                       cudaMemcpyDeviceToDevice, r.stream));
        } else {
            CHECK_CUDA(cudaMemcpyPeerAsync(r.d_attn_sinks, r.device,
                                           hc->d_attn_sinks[layer] + sinks_offset,
                                           opt.devices[0],
                                           (size_t)kLocalHeads * sizeof(float),
                                           r.stream));
        }
        attention_raw_swa_one_row_kernel<<<
            (unsigned int)(opt.slots * kLocalHeads), 256, 0, r.stream>>>(
            r.d_attn_heads, ops->attn_q_b.d_out[(size_t)rank], r.d_attn_raw_swa,
            r.d_attn_sinks, (uint32_t)opt.slots, (uint32_t)kLocalHeads,
            (uint32_t)kHeadDim, (uint32_t)kRawSwaRows, raw_row);
        CHECK_CUDA(cudaGetLastError());
    }
    if (!opt.decode_cudagraph_gate) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    if (!opt.decode_cudagraph_gate && layer <= 2) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("true_attn_raw_read_heads", layer, rank,
                                 ranks[rank].d_attn_heads, (size_t)heads_elems,
                                 ranks[rank].stream);
        }
    }
    std::printf("tp_ep_true_attention_raw_read\tlayer\t%d\tslots\t%d\t"
                "local_heads\t%d\thead_dim\t%d\traw_rows\t%d\traw_row\t%u\t"
                "ms\t%.6f\tPASS\n",
                layer, opt.slots, kLocalHeads, kHeadDim, kRawSwaRows, raw_row, ms);
    return 0;
}

int run_true_ds4_attention_raw_window(const Options &opt,
                                      SharedHcControls *hc,
                                      const LayerDenseOps *ops,
                                      RankState ranks[kGpus],
                                      int layer) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    if (!hc->d_attn_sinks[layer] ||
        ops->attn_q_b.rows_per_gpu != kLocalHeads * kHeadDim) {
        return 2;
    }
    const uint32_t valid_rows =
        std::max(1u, std::min(opt.true_ds4_attention_raw_valid_rows,
                              (uint32_t)kRawSwaRows));
    const int ratio = ds4_layer_ratio(layer);
    const auto start = std::chrono::steady_clock::now();
    const uint32_t raw_row = (uint32_t)(opt.position % kRawSwaRows);
    const uint64_t heads_elems =
        (uint64_t)opt.slots * (uint64_t)kLocalHeads * (uint64_t)kHeadDim;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_raw_swa || !r.d_attn_sinks || !r.d_attn_heads ||
            !ops->attn_q_b.d_out[(size_t)rank]) {
            return 3;
        }
        const size_t sinks_offset = (size_t)rank * (size_t)kLocalHeads;
        if (rank == 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_attn_sinks,
                                       hc->d_attn_sinks[layer] + sinks_offset,
                                       (size_t)kLocalHeads * sizeof(float),
                                       cudaMemcpyDeviceToDevice, r.stream));
        } else {
        CHECK_CUDA(cudaMemcpyPeerAsync(r.d_attn_sinks, r.device,
                                           hc->d_attn_sinks[layer] + sinks_offset,
                                           opt.devices[0],
                                           (size_t)kLocalHeads * sizeof(float),
                                           r.stream));
        }
        const uint32_t visible_comp_rows =
            opt.true_ds4_compressed_kv_gate && ratio != 0
                ? std::min(r.attn_comp_rows_written_layers[layer],
                           (uint32_t)kBoundedCompRows)
                : 0u;
        const uint32_t selected_comp_rows =
            visible_comp_rows == 0u
                ? 0u
                : (ratio == 4 && opt.true_ds4_indexer_attention_gate
                       ? std::min(visible_comp_rows, (uint32_t)kBoundedCompRows)
                       : visible_comp_rows);
        if (selected_comp_rows > 0u) {
            if (!r.d_attn_comp_rows ||
                (ratio == 4 && opt.true_ds4_indexer_attention_gate &&
                 !r.d_indexer_topk)) {
                return 4;
            }
            attention_raw_compressed_window_kernel<<<
                (unsigned int)(opt.slots * kLocalHeads), 256, 0, r.stream>>>(
                r.d_attn_heads, ops->attn_q_b.d_out[(size_t)rank],
                r.d_attn_raw_swa, r.d_attn_comp_rows,
                ratio == 4 && opt.true_ds4_indexer_attention_gate
                    ? r.d_indexer_topk
                    : nullptr,
                r.d_attn_sinks, (uint32_t)opt.slots, (uint32_t)kLocalHeads,
                (uint32_t)kHeadDim, (uint32_t)kRawSwaRows, raw_row,
                valid_rows, visible_comp_rows, selected_comp_rows,
                (uint32_t)kBoundedCompRows, (uint32_t)kIndexerTopK);
        } else {
            attention_raw_swa_window_kernel<<<
                (unsigned int)(opt.slots * kLocalHeads), 256, 0, r.stream>>>(
                r.d_attn_heads, ops->attn_q_b.d_out[(size_t)rank],
                r.d_attn_raw_swa, r.d_attn_sinks, (uint32_t)opt.slots,
                (uint32_t)kLocalHeads, (uint32_t)kHeadDim,
                (uint32_t)kRawSwaRows, raw_row, valid_rows);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    if (!opt.decode_cudagraph_gate) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    if (!opt.decode_cudagraph_gate && layer <= 2) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("true_attn_raw_window_heads", layer, rank,
                                 ranks[rank].d_attn_heads, (size_t)heads_elems,
                                 ranks[rank].stream);
        }
    }
    std::printf("tp_ep_true_attention_raw_window\tlayer\t%d\tslots\t%d\t"
                "local_heads\t%d\thead_dim\t%d\traw_rows\t%d\traw_row\t%u\t"
                "valid_rows\t%u\tvisible_compressed_rows\t%u\t"
                "selected_compressed_rows\t%u\tms\t%.6f\tPASS\n",
                layer, opt.slots, kLocalHeads, kHeadDim, kRawSwaRows, raw_row,
                valid_rows,
                opt.true_ds4_compressed_kv_gate && ratio != 0
                    ? std::min(ranks[0].attn_comp_rows_written_layers[layer],
                               (uint32_t)kBoundedCompRows)
                    : 0u,
                opt.true_ds4_compressed_kv_gate && ratio != 0
                    ? std::min(ranks[0].attn_comp_rows_written_layers[layer],
                               (uint32_t)kBoundedCompRows)
                    : 0u,
                ms);
    return 0;
}

int run_true_ds4_attention_output_projection(const Options &opt,
                                             const LayerDenseOps *ops,
                                             RankState ranks[kGpus],
                                             int layer) {
    if (!ops || !ops->initialized || layer < 0 || layer >= 43) {
        return 1;
    }
    if (ops->attn_output_a.cols != kAttentionOutputAInput ||
        ops->attn_output_a.rows_per_gpu != kAttentionOutputAFull / kGpus ||
        ops->attn.cols != kAttentionOutputAFull ||
        ops->attn.rows_per_gpu != kHidden / kGpus) {
        std::fprintf(stderr,
                     "tp_ep_true_attention_output_bad_shape\tlayer\t%d\t"
                     "out_a_cols\t%d\tout_a_rows_per_gpu\t%d\t"
                     "out_b_cols\t%d\tout_b_rows_per_gpu\t%d\n",
                     layer, ops->attn_output_a.cols,
                     ops->attn_output_a.rows_per_gpu, ops->attn.cols,
                     ops->attn.rows_per_gpu);
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    const int block = 256;
    const size_t out_a_shard_cols = (size_t)ops->attn_output_a.rows_per_gpu;
    const size_t out_a_shard_row_bytes = out_a_shard_cols * sizeof(float);
    const size_t out_a_full_row_bytes = (size_t)kAttentionOutputAFull * sizeof(float);
    const uint64_t head_input_elems =
        (uint64_t)opt.slots * (uint64_t)kAttentionOutputAInput;
    const uint64_t out_a_full_elems =
        (uint64_t)opt.slots * (uint64_t)kAttentionOutputAFull;

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_heads || !r.d_attn_output_a_full ||
            !ops->attn_output_a.d_x_half[(size_t)rank] ||
            !ops->attn.d_x_half[(size_t)rank]) {
            return 3;
        }
        fill_dense_input_half_from_tensor_kernel<<<
            (unsigned int)((head_input_elems + block - 1) / block), block, 0,
            r.stream>>>(ops->attn_output_a.d_x_half[(size_t)rank],
                          r.d_attn_heads,
                          (uint32_t)kAttentionOutputAInput,
                          (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }

    if (launch_resident_f8_dense(opt, ops->attn_output_a, ranks) != 0) {
        return 5;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        if (ranks[rank].dense_stream) {
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].dense_stream));
        } else {
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }

    const bool use_nccl_allgather =
        opt.true_ds4_attention_output_nccl_allgather_gate;
    if (use_nccl_allgather) {
        for (int rank = 0; rank < kGpus; ++rank) {
            if (!ranks[rank].compose_nccl_initialized ||
                !ranks[rank].compose_nccl ||
                !ops->attn_output_a.d_out[(size_t)rank]) {
                return 6;
            }
        }
        CHECK_NCCL(ncclGroupStart());
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_NCCL(ncclAllGather(ops->attn_output_a.d_out[(size_t)rank],
                                     r.d_attn_output_a_full,
                                     (size_t)opt.slots * out_a_shard_cols,
                                     ncclFloat,
                                     r.compose_nccl,
                                     r.stream));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            fill_dense_input_half_from_rank_major_shards_kernel<<<
                (unsigned int)((out_a_full_elems + block - 1) / block),
                block, 0, r.stream>>>(
                ops->attn.d_x_half[(size_t)rank], r.d_attn_output_a_full,
                (uint32_t)out_a_shard_cols, (uint32_t)kGpus,
                (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
    } else {
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &dr = ranks[dst];
            CHECK_CUDA(cudaSetDevice(dr.device));
            for (int src = 0; src < kGpus; ++src) {
                const float *src_shard = ops->attn_output_a.d_out[(size_t)src];
                if (!src_shard) return 6;
                CHECK_CUDA(cudaMemcpy2DAsync(
                    dr.d_attn_output_a_full + (size_t)src * out_a_shard_cols,
                    out_a_full_row_bytes, src_shard, out_a_shard_row_bytes,
                    out_a_shard_row_bytes, (size_t)opt.slots, cudaMemcpyDefault,
                    dr.stream));
            }
            fill_dense_input_half_from_tensor_kernel<<<
                (unsigned int)((out_a_full_elems + block - 1) / block), block, 0,
                dr.stream>>>(ops->attn.d_x_half[(size_t)dst],
                              dr.d_attn_output_a_full,
                              (uint32_t)kAttentionOutputAFull,
                              (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }

    if (launch_resident_f8_dense(opt, ops->attn, ranks) != 0) {
        return 7;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        if (ranks[rank].dense_stream) {
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].dense_stream));
        } else {
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }

    TensorF32Stats head_stats;
    TensorF32Stats out_a_stats;
    TensorF32Stats out_b_stats;
    if (!opt.true_ds4_semantic_skip_stats_gate) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            merge_tensor_stats(
                &head_stats,
                collect_tensor_f32_stats(r.d_attn_heads,
                                         (size_t)head_input_elems, r.stream));
            merge_tensor_stats(
                &out_a_stats,
                collect_tensor_f32_stats(r.d_attn_output_a_full,
                                         (size_t)out_a_full_elems, r.stream));
            merge_tensor_stats(
                &out_b_stats,
                collect_tensor_f32_stats(
                    ops->attn.d_out[(size_t)rank],
                    (size_t)opt.slots * (size_t)ops->attn.rows_per_gpu,
                    r.dense_stream ? r.dense_stream : r.stream));
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    std::printf("tp_ep_true_attention_output_projection\tlayer\t%d\tslots\t%d\t"
                "head_input_cols\t%d\tout_a_cols\t%d\tout_b_shard_cols\t%d\t"
                "nccl_allgather\t%d\t"
                "stats_skipped\t%d\t"
                "heads_max\t%.9g\theads_bad\t%d\t"
                "out_a_max\t%.9g\tout_a_bad\t%d\t"
                "out_b_max\t%.9g\tout_b_bad\t%d\tms\t%.6f\tPASS\n",
                layer, opt.slots, kAttentionOutputAInput, kAttentionOutputAFull,
                ops->attn.rows_per_gpu, use_nccl_allgather ? 1 : 0,
                opt.true_ds4_semantic_skip_stats_gate ? 1 : 0,
                head_stats.max_abs,
                head_stats.finite_bad, out_a_stats.max_abs,
                out_a_stats.finite_bad, out_b_stats.max_abs,
                out_b_stats.finite_bad, ms);
    return 0;
}

int run_true_ds4_post_attention_ffn_input(const Options &opt,
                                          SharedHcControls *hc,
                                          const LayerDenseOps *ops,
                                          RankState ranks[kGpus],
                                          int layer) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        hc->slots != opt.slots || layer < 0 || layer >= 43) {
        return 1;
    }
    if (ops->attn.rows_per_gpu != kHidden / kGpus ||
        ops->shared_gate.cols != kHidden ||
        ops->shared_up.cols != kHidden ||
        ops->shared_gate.rows_per_gpu != kMid / kGpus ||
        ops->shared_up.rows_per_gpu != kMid / kGpus) {
        return 2;
    }
    if (!hc->d_current_full || !hc->d_ffn_normed ||
        !hc->d_ffn_norm_weight[layer]) {
        return 3;
    }

    const auto start = std::chrono::steady_clock::now();
    const int block = 256;
    const uint64_t shard_elems =
        (uint64_t)opt.slots * (uint64_t)(kHidden / kGpus);
    const uint64_t full_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const size_t full_bytes = (size_t)full_elems * sizeof(float);

    TensorF32Stats post_shard_stats;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_current_shard || !r.d_post_attn_shard ||
            !ops->attn.d_out[(size_t)rank]) {
            return 4;
        }
        add_current_attention_shard_kernel<<<
            (unsigned int)((shard_elems + block - 1) / block), block, 0,
            r.stream>>>(r.d_post_attn_shard, r.d_current_shard,
                         ops->attn.d_out[(size_t)rank], shard_elems);
        CHECK_CUDA(cudaGetLastError());
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        if (!opt.true_ds4_semantic_skip_stats_gate) {
            merge_tensor_stats(
                &post_shard_stats,
                collect_tensor_f32_stats(ranks[rank].d_post_attn_shard,
                                         (size_t)shard_elems,
                                         ranks[rank].stream));
        }
    }

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        gather_current_shard_to_full_kernel<<<
            (unsigned int)((shard_elems + block - 1) / block), block>>>(
            hc->d_current_full, ranks[rank].d_post_attn_shard, rank,
            (uint32_t)opt.slots);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    rms_norm_weight_rows_stable_kernel<<<(unsigned int)opt.slots, 256>>>(
        hc->d_ffn_normed, hc->d_current_full, hc->d_ffn_norm_weight[layer],
        (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    TensorF32Stats ffn_norm_stats;
    if (!opt.true_ds4_semantic_skip_stats_gate) {
        ffn_norm_stats =
            collect_tensor_f32_stats(hc->d_ffn_normed, (size_t)full_elems, nullptr);
    }

    if (opt.model_router_routes) {
        if (!hc->d_router_w[layer] || !hc->d_router_logits ||
            !hc->d_router_selected || !hc->d_router_weights) {
            return 5;
        }
        const int router_dense_rc =
            run_model_router_dense_logits(opt, hc, layer, (cudaStream_t)0);
        if (router_dense_rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_post_attention_router_dense_failed\tlayer\t%d\trc\t%d\n",
                         layer, router_dense_rc);
            return 5;
        }
        if (opt.router_hash_fast_gate && hc->d_router_hash[layer] &&
            hc->d_router_tokens && hc->router_hash_rows[layer] > 0u) {
            router_select_hash_fast_rows_kernel<<<(unsigned int)opt.slots, 1>>>(
                hc->d_router_selected, hc->d_router_weights,
                hc->d_router_logits, hc->d_router_hash[layer],
                hc->d_router_tokens, hc->d_router_active,
                hc->router_hash_rows[layer], (uint32_t)opt.slots);
        } else {
            router_select_topk_rows_kernel<<<(unsigned int)opt.slots, 1>>>(
                hc->d_router_selected, hc->d_router_weights,
                hc->d_router_logits, hc->d_router_bias[layer],
                hc->d_router_hash[layer], hc->d_router_tokens,
                hc->d_router_active, hc->router_hash_rows[layer],
                (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        int route_rc = 0;
        if (opt.gpu_route_plan_gate) {
            route_rc = upload_model_router_route_plan_gpu(opt, hc, ranks);
        } else if (opt.route_plan_async_upload_gate) {
            RoutePlanHostWorkspace *ws = &hc->route_plan_ws;
            if (!ws->initialized) return 6;
            const size_t route_elems = (size_t)opt.slots * (size_t)opt.top_k;
            CHECK_CUDA(cudaMemcpyAsync(ws->h_selected, hc->d_router_selected,
                                       route_elems * sizeof(int),
                                       cudaMemcpyDeviceToHost, (cudaStream_t)0));
            CHECK_CUDA(cudaMemcpyAsync(ws->h_weights, hc->d_router_weights,
                                       route_elems * sizeof(float),
                                       cudaMemcpyDeviceToHost, (cudaStream_t)0));
            CHECK_CUDA(cudaStreamSynchronize((cudaStream_t)0));
            route_rc = upload_model_router_route_plan_async(
                opt, ranks, ws->h_selected, ws->h_weights, ws);
        } else {
            std::vector<int> selected((size_t)opt.slots * (size_t)opt.top_k);
            std::vector<float> weights((size_t)opt.slots * (size_t)opt.top_k);
            CHECK_CUDA(cudaMemcpy(selected.data(), hc->d_router_selected,
                                  selected.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(weights.data(), hc->d_router_weights,
                                  weights.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            route_rc = upload_model_router_route_plan(opt, ranks,
                                                      selected, weights);
        }
        if (route_rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_post_attention_route_plan_failed\tlayer\t%d\trc\t%d\n",
                         layer, route_rc);
            return 6;
        }
    }

    const uint64_t x_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_current_full) return 7;
        if (rank == 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_current_full, hc->d_ffn_normed,
                                       full_bytes, cudaMemcpyDeviceToDevice,
                                       r.stream));
        } else {
            CHECK_CUDA(cudaMemcpyPeerAsync(r.d_current_full, r.device,
                                           hc->d_ffn_normed, opt.devices[0],
                                           full_bytes, r.stream));
        }
        if (ops->shared_gate.d_x_half[(size_t)rank]) {
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((x_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->shared_gate.d_x_half[(size_t)rank],
                             r.d_current_full,
                             (uint32_t)ops->shared_gate.cols,
                             (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        if (ops->shared_up.d_x_half[(size_t)rank]) {
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((x_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->shared_up.d_x_half[(size_t)rank],
                             r.d_current_full,
                             (uint32_t)ops->shared_up.cols,
                             (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        const uint64_t route_elems = (uint64_t)r.routes * kHidden;
        if (route_elems > 0) {
            if (opt.reference_hc_reduce_gate) {
                pack_current_full_to_routes_scaled_kernel<<<
                    (unsigned int)r.routes, 256, 0, r.stream>>>(
                        r.d_a, r.d_route_inv_scale, r.d_current_full,
                        r.d_route_slots, r.routes, kReferenceRouteInputTargetAbs);
            } else {
                pack_current_full_to_routes_kernel<<<
                    (unsigned int)((route_elems + block - 1) / block), block,
                    0, r.stream>>>(r.d_a, r.d_current_full, r.d_route_slots,
                                   r.routes);
            }
            CHECK_CUDA(cudaGetLastError());
        }
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }

    TensorF32Stats route_inv_scale_stats;
    int total_routes = 0;
    for (int rank = 0; rank < kGpus; ++rank) {
        total_routes += ranks[rank].routes;
        if (!opt.true_ds4_semantic_skip_stats_gate &&
            ranks[rank].d_route_inv_scale && ranks[rank].routes > 0) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            merge_tensor_stats(
                &route_inv_scale_stats,
                collect_tensor_f32_stats(ranks[rank].d_route_inv_scale,
                                         (size_t)ranks[rank].routes,
                                         ranks[rank].stream));
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    std::printf("tp_ep_post_attention_ffn_input\tlayer\t%d\tslots\t%d\t"
                "total_routes\t%d\tstats_skipped\t%d\tpost_max\t%.9g\tpost_bad\t%d\t"
                "ffn_norm_max\t%.9g\tffn_norm_bad\t%d\t"
                "route_inv_scale_max\t%.9g\troute_inv_scale_bad\t%d\t"
                "ms\t%.6f\tPASS\n",
                layer, opt.slots, total_routes,
                opt.true_ds4_semantic_skip_stats_gate ? 1 : 0,
                post_shard_stats.max_abs,
                post_shard_stats.finite_bad, ffn_norm_stats.max_abs,
                ffn_norm_stats.finite_bad, route_inv_scale_stats.max_abs,
                route_inv_scale_stats.finite_bad, ms);
    return (post_shard_stats.finite_bad || ffn_norm_stats.finite_bad ||
            route_inv_scale_stats.finite_bad) ? 8 : 0;
}

int run_decode_loop(const Options &opt,
                    const std::vector<ContractRow> &rows,
                    RankState ranks[kGpus],
                    const Api &api,
                    ds4_v100_tp_runtime *rt,
                    const DenseF16Cache *cache,
                    const LayerDenseOps *shared_dense_ops,
                    SharedHcControls *shared_hc_controls,
                    DecodeLoopStats *stats) {
    if (opt.decode_steps <= 0) return 0;
    stats->enabled = true;
    stats->ep_return_fp16 = opt.ep_return_fp16;
    stats->fused_compose_sum =
        opt.fuse_compose_sum && !opt.ep_return_fp16 && !opt.compact_route_compose;
    stats->dense_hmma_compose = opt.dense_hmma_compose;
    stats->dense_f16_cublas_compose = opt.dense_f16_cublas_compose;
    stats->dense_f16_cache_compose = opt.dense_f16_cache_compose;
    stats->nccl_reduce_scatter_compose =
        opt.nccl_reduce_scatter_compose_gate &&
        !opt.compact_route_compose && !opt.ep_return_fp16;
    stats->steps = opt.decode_steps;
    stats->slots = opt.slots;
    stats->slot_steps = (uint64_t)opt.decode_steps * (uint64_t)opt.slots;

    ResidentF8Dense attn;
    ResidentF8Dense shared;
    ResidentF8Dense shared_gate;
    ResidentF8Dense shared_up;
    const ResidentF8Dense *attn_op = nullptr;
    const ResidentF8Dense *shared_op = nullptr;
    const ResidentF8Dense *shared_gate_op = nullptr;
    const ResidentF8Dense *shared_up_op = nullptr;
    const std::string attn_tensor = layer_tensor_name(opt.layer, "attn_output_b.weight");
    const std::string shared_tensor = layer_tensor_name(opt.layer, "ffn_down_shexp.weight");
    if (shared_dense_ops) {
        attn_op = &shared_dense_ops->attn;
        shared_op = &shared_dense_ops->shared;
        if (opt.true_shared_ffn_gate) {
            shared_gate_op = &shared_dense_ops->shared_gate;
            shared_up_op = &shared_dense_ops->shared_up;
        }
    } else {
        if (prepare_resident_f8_dense(opt, rows, attn_tensor.c_str(), 1, cache, &attn) != 0 ||
            prepare_resident_f8_dense(opt, rows, shared_tensor.c_str(), 2, cache, &shared) != 0) {
            free_resident_f8_dense(attn, opt);
            free_resident_f8_dense(shared, opt);
            return 1;
        }
        attn_op = &attn;
        shared_op = &shared;
    }
    stats->dense_loaded_bytes = attn_op->loaded_bytes + shared_op->loaded_bytes;
    if (opt.true_shared_ffn_gate) {
        if (!shared_gate_op || !shared_up_op ||
            !shared_gate_op->d_out.size() || !shared_up_op->d_out.size()) {
            return 1;
        }
        stats->dense_loaded_bytes += shared_gate_op->loaded_bytes + shared_up_op->loaded_bytes;
    }

    const uint64_t shard_elems = (uint64_t)opt.slots * (kHidden / kGpus);
    const uint64_t shard_bytes = shard_elems * sizeof(float);
    const uint64_t return_shard_bytes =
        shard_elems * (opt.ep_return_fp16 ? sizeof(__half) : sizeof(float));
    const uint64_t all_contrib_elems = (uint64_t)kGpus * shard_elems;
    const uint64_t all_contrib_bytes = all_contrib_elems * sizeof(float);
    const bool skip_self_copy = opt.skip_self_compose_copy && !opt.ep_return_fp16;
    const bool nccl_reduce_scatter = stats->nccl_reduce_scatter_compose;
    stats->ep_contribution_bytes = all_contrib_bytes * kGpus;
    if (opt.compact_route_compose && !opt.ep_return_fp16) {
        uint64_t compact_return_bytes = 0;
        for (int src = 0; src < kGpus; ++src) {
            compact_return_bytes +=
                (uint64_t)ranks[src].routes * (kHidden / kGpus) * sizeof(float) *
                (skip_self_copy ? (kGpus - 1) : kGpus);
        }
        stats->ep_return_bytes = compact_return_bytes;
    } else {
        stats->ep_return_bytes = return_shard_bytes *
                                 (skip_self_copy ? (kGpus * kGpus - kGpus)
                                                 : (kGpus * kGpus));
    }
    if (ensure_compose_buffers(opt, ranks) != 0) {
        if (!shared_dense_ops) {
            free_resident_f8_dense(attn, opt);
            free_resident_f8_dense(shared, opt);
        }
        return 2;
    }

    int cudagraph_audit_sync_all_calls = 0;
    int cudagraph_audit_event_barrier_calls = 0;
    int cudagraph_audit_stream_syncs = 0;
    int cudagraph_audit_dense_stream_syncs = 0;
    int cudagraph_audit_copy_stream_syncs = 0;
    int cudagraph_capture_attempted = 0;
    int cudagraph_capture_succeeded = 0;
    int cudagraph_capture_error = 0;
    size_t cudagraph_capture_nodes = 0;
    auto sync_all = [&]() {
        if (opt.decode_cudagraph_gate) {
            cudagraph_audit_event_barrier_calls++;
            if (enqueue_cross_gpu_stream_barrier(ranks, true) != 0) {
                return;
            }
            return;
        }
        cudagraph_audit_sync_all_calls++;
        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[p].stream));
            cudagraph_audit_stream_syncs++;
            if (ranks[p].dense_stream) {
                CHECK_CUDA(cudaStreamSynchronize(ranks[p].dense_stream));
                cudagraph_audit_dense_stream_syncs++;
            }
        }
    };

    struct CaptureHostRankState {
        float *final_hc_shard = nullptr;
        float *hc_scratch_shard = nullptr;
        bool hc_initialized = false;
        uint32_t attn_rows_written = 0;
        uint32_t index_rows_written = 0;
        bool attn_loaded[kBoundedCompRows] = {};
        bool index_loaded[kBoundedCompRows] = {};
        uint64_t attn_position[kBoundedCompRows] = {};
        uint64_t index_position[kBoundedCompRows] = {};
        uint64_t attn_loaded_position[kBoundedCompRows] = {};
        uint64_t index_loaded_position[kBoundedCompRows] = {};
    };
    auto save_capture_host_state = [&](CaptureHostRankState saved[kGpus]) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CaptureHostRankState &s = saved[rank];
            s.final_hc_shard = r.d_final_hc_shard;
            s.hc_scratch_shard = r.d_hc_scratch_shard;
            s.hc_initialized = r.hc_initialized;
            s.attn_rows_written = r.attn_comp_rows_written_layers[opt.layer];
            s.index_rows_written = r.index_comp_rows_written_layers[opt.layer];
            for (int row = 0; row < kBoundedCompRows; ++row) {
                s.attn_loaded[row] = r.attn_comp_row_loaded_layers[opt.layer][row];
                s.index_loaded[row] = r.index_comp_row_loaded_layers[opt.layer][row];
                s.attn_position[row] = r.attn_comp_row_position_layers[opt.layer][row];
                s.index_position[row] = r.index_comp_row_position_layers[opt.layer][row];
                s.attn_loaded_position[row] =
                    r.attn_comp_row_loaded_position_layers[opt.layer][row];
                s.index_loaded_position[row] =
                    r.index_comp_row_loaded_position_layers[opt.layer][row];
            }
        }
    };
    auto restore_capture_host_state = [&](const CaptureHostRankState saved[kGpus]) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            const CaptureHostRankState &s = saved[rank];
            r.d_final_hc_shard = s.final_hc_shard;
            r.d_hc_scratch_shard = s.hc_scratch_shard;
            r.hc_initialized = s.hc_initialized;
            r.attn_comp_rows_written_layers[opt.layer] = s.attn_rows_written;
            r.index_comp_rows_written_layers[opt.layer] = s.index_rows_written;
            for (int row = 0; row < kBoundedCompRows; ++row) {
                r.attn_comp_row_loaded_layers[opt.layer][row] = s.attn_loaded[row];
                r.index_comp_row_loaded_layers[opt.layer][row] = s.index_loaded[row];
                r.attn_comp_row_position_layers[opt.layer][row] = s.attn_position[row];
                r.index_comp_row_position_layers[opt.layer][row] = s.index_position[row];
                r.attn_comp_row_loaded_position_layers[opt.layer][row] =
                    s.attn_loaded_position[row];
                r.index_comp_row_loaded_position_layers[opt.layer][row] =
                    s.index_loaded_position[row];
            }
        }
    };
    auto begin_capture_stream = [](int device, cudaStream_t stream) -> cudaError_t {
        if (!stream) return cudaSuccess;
        cudaError_t rc = cudaSetDevice(device);
        if (rc != cudaSuccess) return rc;
        return cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    };
    auto end_capture_stream = [](int device, cudaStream_t stream,
                                 cudaGraph_t *graph) -> cudaError_t {
        if (!stream) return cudaSuccess;
        cudaError_t rc = cudaSetDevice(device);
        if (rc != cudaSuccess) return rc;
        return cudaStreamEndCapture(stream, graph);
    };
    auto destroy_capture_graphs = [](std::vector<cudaGraph_t> *graphs) {
        for (cudaGraph_t graph : *graphs) {
            if (graph) cudaGraphDestroy(graph);
        }
        graphs->clear();
    };
    auto run_one_step = [&](double *ep_ms,
                            double *dense_ms,
                            double *compose_ms,
                            double *compose_reduce_ms,
                            double *compose_copy_ms,
                            double *compose_final_ms,
                            double *hc_current_input_ms,
                            HcCurrentInputBreakdown *hc_current_breakdown,
                            PreEpPrefixBreakdown *pre_ep_breakdown,
                            double *final_hc_ms) -> int {
        auto t_pre = std::chrono::steady_clock::now();
        if (opt.tp_hc_current_input_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            if (!shared_hc_controls || !shared_hc_controls->initialized) {
                std::fprintf(stderr, "tp_hc_current_input_failed\tlayer\t%d\treason\tmissing_controls\n",
                             opt.layer);
                return 8;
            }
            const int hc_rc = run_shared_hc_current_input(opt, shared_hc_controls, ranks,
                                                          *attn_op, *shared_op,
                                                          opt.layer,
                                                          hc_current_breakdown);
            if (hc_rc != 0) {
                std::fprintf(stderr, "tp_hc_current_input_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, hc_rc);
                return 9;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->hc_current_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
        }
        if (opt.true_ds4_attention_projection_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            const int attn_rc = run_true_ds4_attention_projection_prefix(
                opt, shared_hc_controls, shared_dense_ops, ranks, opt.layer);
            if (attn_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_projection_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, attn_rc);
                return 14;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->attention_projection_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
        }
        if (opt.true_ds4_compressed_kv_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            const int comp_rc = run_true_ds4_compressed_kv_projection_gate(
                opt, shared_hc_controls, shared_dense_ops, ranks, rt, opt.layer);
            if (comp_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_compressed_kv_projection_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, comp_rc);
                return 19;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->compressed_kv_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
        }
        if (opt.true_ds4_attention_state_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            const int state_rc = run_true_ds4_attention_state_update(
                opt, shared_hc_controls, shared_dense_ops, ranks, rt, opt.layer);
            if (state_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_state_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, state_rc);
                return 15;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->attention_state_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
        }
        maybe_log_batched_paged_attn_plan(opt, ranks, opt.layer);
        if (opt.true_ds4_attention_typed_kv_history_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            const int history_rc = run_true_ds4_attention_typed_kv_history_load(
                opt, shared_hc_controls, ranks, rt, opt.layer);
            if (history_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_typed_kv_history_failed\t"
                             "layer\t%d\trc\t%d\n",
                             opt.layer, history_rc);
                return 24;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->typed_history_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
        }
        if (opt.true_ds4_attention_raw_read_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            const int raw_read_rc = opt.true_ds4_attention_raw_window_gate
                ? run_true_ds4_attention_raw_window(
                      opt, shared_hc_controls, shared_dense_ops, ranks, opt.layer)
                : run_true_ds4_attention_raw_read(
                      opt, shared_hc_controls, shared_dense_ops, ranks, opt.layer);
            if (raw_read_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_raw_read_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, raw_read_rc);
                return 16;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->raw_read_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
        }
        if (opt.true_ds4_attention_output_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            const int output_rc = run_true_ds4_attention_output_projection(
                opt, shared_dense_ops, ranks, opt.layer);
            if (output_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_output_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, output_rc);
                return 17;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->attention_output_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
        }
        if (opt.true_ds4_post_attention_ffn_input_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            const int post_rc = run_true_ds4_post_attention_ffn_input(
                opt, shared_hc_controls, shared_dense_ops, ranks, opt.layer);
            if (post_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_post_attention_ffn_input_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, post_rc);
                return 18;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->post_attention_ffn_input_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
        }
        auto t0 = std::chrono::steady_clock::now();
        if (opt.true_shared_ffn_gate &&
            !opt.true_ds4_post_attention_ffn_input_gate) {
            if (!shared_hc_controls || !shared_hc_controls->d_ffn_normed ||
                !shared_gate_op || !shared_up_op) {
                return 10;
            }
            const int fill_rc = fill_shared_ffn_inputs_from_normed(
                opt, shared_hc_controls, *shared_gate_op, *shared_up_op, ranks);
            if (fill_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_shared_ffn_input_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, fill_rc);
                return 10;
            }
        }
        const bool log_semantic_stats = should_log_routed_semantic_stats(opt);
        if (log_semantic_stats) {
            for (int p = 0; p < kGpus; ++p) {
                RankState &r = ranks[p];
                const size_t elems = (size_t)r.routes * kHidden;
                if (elems > 0) {
                    CHECK_CUDA(cudaSetDevice(r.device));
                    log_route_half_stats("route_input", opt.layer, p,
                                         r.d_a, elems, r.stream);
                }
            }
        }
        for (int p = 0; p < kGpus; ++p) {
            const int gate_rc = run_gate_selected(ranks[p], api, opt);
            if (gate_rc != 0 || run_down(ranks[p], api) != 0) return 1;
        }
        double ep_stage_ms = 0.0;
        double dense_stage_ms = 0.0;
        if (opt.overlap_ep_dense) {
            if (launch_resident_f8_dense(opt, *attn_op, ranks) != 0) {
                return 2;
            }
            if (opt.true_shared_ffn_gate) {
                if (launch_resident_f8_dense(opt, *shared_gate_op, ranks) != 0 ||
                    launch_resident_f8_dense(opt, *shared_up_op, ranks) != 0) {
                    return 2;
                }
                if (opt.layer <= 4 || should_log_reference_hc_window(opt)) {
                    for (int p = 0; p < kGpus; ++p) {
                        CHECK_CUDA(cudaSetDevice(ranks[p].device));
                        log_tensor_f32_stats("shared_gate", opt.layer, p,
                                             shared_gate_op->d_out[(size_t)p],
                                             (size_t)opt.slots * shared_gate_op->rows_per_gpu,
                                             ranks[p].dense_stream ? ranks[p].dense_stream
                                                                   : ranks[p].stream);
                        log_tensor_f32_stats("shared_up", opt.layer, p,
                                             shared_up_op->d_out[(size_t)p],
                                             (size_t)opt.slots * shared_up_op->rows_per_gpu,
                                             ranks[p].dense_stream ? ranks[p].dense_stream
                                                                   : ranks[p].stream);
                    }
                }
            } else if (launch_resident_f8_dense(opt, *shared_op, ranks) != 0) {
                return 2;
            }
            sync_all();
            if (opt.true_shared_ffn_gate) {
                const int swiglu_rc = materialize_shared_swiglu_down_input(
                    opt, *shared_gate_op, *shared_up_op, *shared_op, ranks);
                if (swiglu_rc != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_shared_ffn_swiglu_failed\tlayer\t%d\trc\t%d\n",
                                 opt.layer, swiglu_rc);
                    return 2;
                }
                if (opt.layer <= 4 || should_log_reference_hc_window(opt)) {
                    for (int p = 0; p < kGpus; ++p) {
                        CHECK_CUDA(cudaSetDevice(ranks[p].device));
                        log_tensor_f32_stats("shared_mid", opt.layer, p,
                                             shared_op->d_x[(size_t)p],
                                             (size_t)opt.slots * kMid,
                                             ranks[p].copy_stream ? ranks[p].copy_stream
                                                                  : ranks[p].stream);
                    }
                }
                if (launch_resident_f8_dense_f32_input(opt, *shared_op, ranks) != 0) {
                    return 2;
                }
                sync_all();
                if (opt.layer <= 4 || should_log_reference_hc_window(opt)) {
                    for (int p = 0; p < kGpus; ++p) {
                        CHECK_CUDA(cudaSetDevice(ranks[p].device));
                        log_tensor_f32_stats("shared_down", opt.layer, p,
                                             shared_op->d_out[(size_t)p],
                                             (size_t)opt.slots * shared_op->rows_per_gpu,
                                             ranks[p].dense_stream ? ranks[p].dense_stream
                                                                   : ranks[p].stream);
                    }
                }
            }
            auto t2 = std::chrono::steady_clock::now();
            ep_stage_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
        } else {
            sync_all();
            auto t1 = std::chrono::steady_clock::now();
            if (launch_resident_f8_dense(opt, *attn_op, ranks) != 0) {
                return 2;
            }
            if (opt.true_shared_ffn_gate) {
                if (launch_resident_f8_dense(opt, *shared_gate_op, ranks) != 0 ||
                    launch_resident_f8_dense(opt, *shared_up_op, ranks) != 0) {
                    return 2;
                }
                sync_all();
                if (opt.layer <= 4 || should_log_reference_hc_window(opt)) {
                    for (int p = 0; p < kGpus; ++p) {
                        CHECK_CUDA(cudaSetDevice(ranks[p].device));
                        log_tensor_f32_stats("shared_gate", opt.layer, p,
                                             shared_gate_op->d_out[(size_t)p],
                                             (size_t)opt.slots * shared_gate_op->rows_per_gpu,
                                             ranks[p].dense_stream ? ranks[p].dense_stream
                                                                   : ranks[p].stream);
                        log_tensor_f32_stats("shared_up", opt.layer, p,
                                             shared_up_op->d_out[(size_t)p],
                                             (size_t)opt.slots * shared_up_op->rows_per_gpu,
                                             ranks[p].dense_stream ? ranks[p].dense_stream
                                                                   : ranks[p].stream);
                    }
                }
                const int swiglu_rc = materialize_shared_swiglu_down_input(
                    opt, *shared_gate_op, *shared_up_op, *shared_op, ranks);
                if (swiglu_rc != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_shared_ffn_swiglu_failed\tlayer\t%d\trc\t%d\n",
                                 opt.layer, swiglu_rc);
                    return 2;
                }
                if (opt.layer <= 4 || should_log_reference_hc_window(opt)) {
                    for (int p = 0; p < kGpus; ++p) {
                        CHECK_CUDA(cudaSetDevice(ranks[p].device));
                        log_tensor_f32_stats("shared_mid", opt.layer, p,
                                             shared_op->d_x[(size_t)p],
                                             (size_t)opt.slots * kMid,
                                             ranks[p].copy_stream ? ranks[p].copy_stream
                                                                  : ranks[p].stream);
                    }
                }
                if (launch_resident_f8_dense_f32_input(opt, *shared_op, ranks) != 0) {
                    return 2;
                }
                sync_all();
                if (opt.layer <= 4 || should_log_reference_hc_window(opt)) {
                    for (int p = 0; p < kGpus; ++p) {
                        CHECK_CUDA(cudaSetDevice(ranks[p].device));
                        log_tensor_f32_stats("shared_down", opt.layer, p,
                                             shared_op->d_out[(size_t)p],
                                             (size_t)opt.slots * shared_op->rows_per_gpu,
                                             ranks[p].dense_stream ? ranks[p].dense_stream
                                                                   : ranks[p].stream);
                    }
                }
            } else if (launch_resident_f8_dense(opt, *shared_op, ranks) != 0) {
                return 2;
            }
            sync_all();
            auto t2 = std::chrono::steady_clock::now();
            ep_stage_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            dense_stage_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        if (log_semantic_stats) {
            for (int p = 0; p < kGpus; ++p) {
                RankState &r = ranks[p];
                const size_t gate_up_elems = (size_t)r.routes * kFusedN;
                const size_t gated_elems = (size_t)r.routes * kMid;
                const size_t down_elems = (size_t)r.routes * kHidden;
                if (gate_up_elems > 0) {
                    CHECK_CUDA(cudaSetDevice(r.device));
                    log_route_half_stats("route_gate_up", opt.layer, p,
                                         r.d_gate_up, gate_up_elems, r.stream);
                    log_route_half_stats("route_gated", opt.layer, p,
                                         r.d_gated, gated_elems, r.stream);
                    log_route_half_stats("route_down", opt.layer, p,
                                         r.d_down, down_elems, r.stream);
                }
            }
        }
        auto t2 = std::chrono::steady_clock::now();

        const int block = 256;
        const bool compact_route = opt.compact_route_compose &&
                                   !opt.ep_return_fp16 &&
                                   !opt.direct_remote_compose;
        const uint64_t compact_segment_routes =
            opt.compact_moe_decode_gate ? (uint64_t)opt.slots * (uint64_t)opt.top_k
                                        : (uint64_t)opt.slots;
        const uint64_t compact_segment_elems =
            compact_segment_routes * (uint64_t)(kHidden / kGpus);
        const bool use_nccl_reduce_scatter =
            nccl_reduce_scatter && !compact_route && !opt.ep_return_fp16;
        for (int p = 0; p < kGpus; ++p) {
            RankState &r = ranks[p];
            CHECK_CUDA(cudaSetDevice(r.device));
            const uint64_t route_hidden_elems = (uint64_t)r.routes * kHidden;
            int grid = (int)((route_hidden_elems + block - 1) / block);
            if (compact_route) {
                if (route_hidden_elems > 0) {
                    ep_pack_route_dest_shards_kernel<<<grid, block, 0, r.stream>>>(
                        r.d_ep_contrib_all, r.d_down, r.d_route_weights,
                        r.routes, (int)compact_segment_routes);
                }
            } else {
                grid = (int)((all_contrib_elems + block - 1) / block);
                zero_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_contrib_all,
                                                              all_contrib_elems);
                CHECK_CUDA(cudaGetLastError());
                grid = (int)((route_hidden_elems + block - 1) / block);
                if (route_hidden_elems > 0) {
                    ep_reduce_all_dest_shards_kernel<<<grid, block, 0, r.stream>>>(
                        r.d_ep_contrib_all, r.d_down, r.d_route_slots,
                        r.d_route_weights, r.routes, opt.slots);
                }
            }
            CHECK_CUDA(cudaGetLastError());
            if (opt.ep_return_fp16) {
                grid = (int)((all_contrib_elems + block - 1) / block);
                cast_f32_to_half_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_ep_contrib_half_all, r.d_ep_contrib_all, all_contrib_elems);
                CHECK_CUDA(cudaGetLastError());
            }
        }
        sync_all();
        auto t_reduce_done = std::chrono::steady_clock::now();
        auto t_copy_done = t_reduce_done;

        if (use_nccl_reduce_scatter) {
            for (int p = 0; p < kGpus; ++p) {
                if (!ranks[p].compose_nccl_initialized || !ranks[p].compose_nccl) {
                    return 12;
                }
            }
            CHECK_NCCL(ncclGroupStart());
            for (int p = 0; p < kGpus; ++p) {
                RankState &r = ranks[p];
                CHECK_CUDA(cudaSetDevice(r.device));
                CHECK_NCCL(ncclReduceScatter(r.d_ep_contrib_all,
                                             r.d_ep_sum,
                                             (size_t)shard_elems,
                                             ncclFloat,
                                             ncclSum,
                                             r.compose_nccl,
                                             r.stream));
            }
            CHECK_NCCL(ncclGroupEnd());
            sync_all();
            t_reduce_done = std::chrono::steady_clock::now();
            t_copy_done = t_reduce_done;
        } else if (!opt.direct_remote_compose || opt.ep_return_fp16) {
            if (opt.source_copy_schedule) {
                for (int src = 0; src < kGpus; ++src) {
                    CHECK_CUDA(cudaSetDevice(ranks[src].device));
                    for (int dst = 0; dst < kGpus; ++dst) {
                        if (skip_self_copy && src == dst) continue;
                        cudaStream_t copy_stream =
                            opt.multi_copy_streams && ranks[src].copy_streams[dst]
                                ? ranks[src].copy_streams[dst]
                                : ranks[src].copy_stream ? ranks[src].copy_stream
                                                         : ranks[src].stream;
                        if (opt.ep_return_fp16) {
                            const __half *src_ptr =
                                ranks[src].d_ep_contrib_half_all + (uint64_t)dst * shard_elems;
                            CHECK_CUDA(cudaMemcpyPeerAsync(ranks[dst].d_ep_remote_half[src],
                                                           ranks[dst].device,
                                                           src_ptr,
                                                           ranks[src].device,
                                                           (size_t)return_shard_bytes,
                                                           copy_stream));
                        } else {
                            const float *src_ptr = ranks[src].d_ep_contrib_all +
                                                   (uint64_t)dst *
                                                       (compact_route ? compact_segment_elems
                                                                      : shard_elems);
                            const size_t copy_bytes = compact_route
                                ? (size_t)((uint64_t)ranks[src].routes *
                                           (kHidden / kGpus) * sizeof(float))
                                : (size_t)return_shard_bytes;
                            if (copy_bytes > 0) {
                                CHECK_CUDA(cudaMemcpyPeerAsync(ranks[dst].d_ep_remote[src],
                                                               ranks[dst].device,
                                                               src_ptr,
                                                               ranks[src].device,
                                                               copy_bytes,
                                                               copy_stream));
                            }
                        }
                        if (opt.copy_event_compose) {
                            CHECK_CUDA(cudaEventRecord(ranks[src].copy_done[dst],
                                                       copy_stream));
                        }
                    }
                }
                if (!opt.copy_event_compose) {
                    for (int src = 0; src < kGpus; ++src) {
                        CHECK_CUDA(cudaSetDevice(ranks[src].device));
                        if (opt.multi_copy_streams) {
                            for (int dst = 0; dst < kGpus; ++dst) {
                                if (skip_self_copy && src == dst) continue;
                                CHECK_CUDA(cudaStreamSynchronize(ranks[src].copy_streams[dst]));
                                cudagraph_audit_copy_stream_syncs++;
                            }
                        } else {
                            CHECK_CUDA(cudaStreamSynchronize(ranks[src].copy_stream ?
                                                            ranks[src].copy_stream :
                                                            ranks[src].stream));
                            cudagraph_audit_copy_stream_syncs++;
                        }
                    }
                }
            } else {
                for (int dst = 0; dst < kGpus; ++dst) {
                    CHECK_CUDA(cudaSetDevice(ranks[dst].device));
                    for (int src = 0; src < kGpus; ++src) {
                        if (skip_self_copy && src == dst) continue;
                        if (opt.ep_return_fp16) {
                            const __half *src_ptr =
                                ranks[src].d_ep_contrib_half_all + (uint64_t)dst * shard_elems;
                            CHECK_CUDA(cudaMemcpyPeerAsync(ranks[dst].d_ep_remote_half[src],
                                                           ranks[dst].device,
                                                           src_ptr,
                                                           ranks[src].device,
                                                           (size_t)return_shard_bytes,
                                                           ranks[dst].stream));
                        } else {
                            const float *src_ptr = ranks[src].d_ep_contrib_all +
                                                   (uint64_t)dst *
                                                       (compact_route ? compact_segment_elems
                                                                      : shard_elems);
                            const size_t copy_bytes = compact_route
                                ? (size_t)((uint64_t)ranks[src].routes *
                                           (kHidden / kGpus) * sizeof(float))
                                : (size_t)return_shard_bytes;
                            if (copy_bytes > 0) {
                                CHECK_CUDA(cudaMemcpyPeerAsync(ranks[dst].d_ep_remote[src],
                                                               ranks[dst].device,
                                                               src_ptr,
                                                               ranks[src].device,
                                                               copy_bytes,
                                                               ranks[dst].stream));
                            }
                        }
                    }
                }
                sync_all();
            }
            t_copy_done = std::chrono::steady_clock::now();
        }

        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            if (opt.copy_event_compose && opt.source_copy_schedule &&
                (!opt.direct_remote_compose || opt.ep_return_fp16)) {
                for (int src = 0; src < kGpus; ++src) {
                    if (skip_self_copy && src == dst) continue;
                    CHECK_CUDA(cudaStreamWaitEvent(r.stream, ranks[src].copy_done[dst], 0));
                }
            }
            int grid = (int)((shard_elems + block - 1) / block);
            if (compact_route) {
                const float *r0 = skip_self_copy && dst == 0
                    ? ranks[0].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[0];
                const float *r1 = skip_self_copy && dst == 1
                    ? ranks[1].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[1];
                const float *r2 = skip_self_copy && dst == 2
                    ? ranks[2].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[2];
                const float *r3 = skip_self_copy && dst == 3
                    ? ranks[3].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[3];
                const float *r4 = skip_self_copy && dst == 4
                    ? ranks[4].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[4];
                const float *r5 = skip_self_copy && dst == 5
                    ? ranks[5].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[5];
                const float *r6 = skip_self_copy && dst == 6
                    ? ranks[6].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[6];
                const float *r7 = skip_self_copy && dst == 7
                    ? ranks[7].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[7];
                if (opt.compact_moe_decode_gate) {
                    compose_next_hidden_compact8_multi_kernel<<<grid, block, 0, r.stream>>>(
                        r.d_next_hidden, r.d_current_shard, attn_op->d_out[(size_t)dst],
                        shared_op->d_out[(size_t)dst], r0, r1, r2, r3, r4, r5, r6, r7,
                        r.d_route_indices_by_slot[0], r.d_route_indices_by_slot[1],
                        r.d_route_indices_by_slot[2], r.d_route_indices_by_slot[3],
                        r.d_route_indices_by_slot[4], r.d_route_indices_by_slot[5],
                        r.d_route_indices_by_slot[6], r.d_route_indices_by_slot[7],
                        r.d_route_count_by_slot[0], r.d_route_count_by_slot[1],
                        r.d_route_count_by_slot[2], r.d_route_count_by_slot[3],
                        r.d_route_count_by_slot[4], r.d_route_count_by_slot[5],
                        r.d_route_count_by_slot[6], r.d_route_count_by_slot[7],
                        dst, opt.slots, opt.top_k);
                } else {
                    compose_next_hidden_compact8_kernel<<<grid, block, 0, r.stream>>>(
                        r.d_next_hidden, r.d_current_shard, attn_op->d_out[(size_t)dst],
                        shared_op->d_out[(size_t)dst], r0, r1, r2, r3, r4, r5, r6, r7,
                        r.d_route_index_by_slot[0], r.d_route_index_by_slot[1],
                        r.d_route_index_by_slot[2], r.d_route_index_by_slot[3],
                        r.d_route_index_by_slot[4], r.d_route_index_by_slot[5],
                        r.d_route_index_by_slot[6], r.d_route_index_by_slot[7],
                        dst, opt.slots);
                }
            } else if (use_nccl_reduce_scatter) {
                compose_next_hidden_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn_op->d_out[(size_t)dst],
                    shared_op->d_out[(size_t)dst], r.d_ep_sum, dst, opt.slots);
            } else if (stats->fused_compose_sum) {
                const float *r0 = skip_self_copy && dst == 0
                    ? ranks[0].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[0].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[0];
                const float *r1 = skip_self_copy && dst == 1
                    ? ranks[1].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[1].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[1];
                const float *r2 = skip_self_copy && dst == 2
                    ? ranks[2].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[2].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[2];
                const float *r3 = skip_self_copy && dst == 3
                    ? ranks[3].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[3].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[3];
                const float *r4 = skip_self_copy && dst == 4
                    ? ranks[4].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[4].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[4];
                const float *r5 = skip_self_copy && dst == 5
                    ? ranks[5].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[5].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[5];
                const float *r6 = skip_self_copy && dst == 6
                    ? ranks[6].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[6].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[6];
                const float *r7 = skip_self_copy && dst == 7
                    ? ranks[7].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[7].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[7];
                compose_next_hidden_sum8_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn_op->d_out[(size_t)dst],
                    shared_op->d_out[(size_t)dst], r0, r1, r2, r3, r4, r5, r6, r7,
                    dst, opt.slots);
            } else {
                zero_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_sum, shard_elems);
                CHECK_CUDA(cudaGetLastError());
                for (int src = 0; src < kGpus; ++src) {
                    if (opt.ep_return_fp16) {
                        add_half_to_f32_kernel<<<grid, block, 0, r.stream>>>(
                            r.d_ep_sum, r.d_ep_remote_half[src], shard_elems);
                    } else {
                        const float *src_contrib = skip_self_copy && src == dst
                            ? ranks[src].d_ep_contrib_all + (uint64_t)dst * shard_elems
                            : r.d_ep_remote[src];
                        add_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_sum,
                                                                     src_contrib,
                                                                     shard_elems);
                    }
                }
                CHECK_CUDA(cudaGetLastError());
                compose_next_hidden_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn_op->d_out[(size_t)dst],
                    shared_op->d_out[(size_t)dst], r.d_ep_sum, dst, opt.slots);
            }
            CHECK_CUDA(cudaGetLastError());
        }
        sync_all();
        if (should_log_reference_hc_window(opt)) {
            for (int dst = 0; dst < kGpus; ++dst) {
                RankState &r = ranks[dst];
                CHECK_CUDA(cudaSetDevice(r.device));
                log_tensor_f32_stats("compose_next_hidden", opt.layer, dst,
                                     r.d_next_hidden, (size_t)shard_elems,
                                     r.stream);
            }
        }
        auto t3 = std::chrono::steady_clock::now();
        auto t4 = t3;
        if (opt.final_hc_carry_gate) {
            const uint64_t hc_shard_elems = shard_elems * 4ull;
            for (int dst = 0; dst < kGpus; ++dst) {
                RankState &r = ranks[dst];
                CHECK_CUDA(cudaSetDevice(r.device));
                if (!opt.tp_hc_final_expand_gate || !r.hc_initialized) {
                    int grid = (int)((hc_shard_elems + block - 1) / block);
                    expand_hidden_to_proxy_hc_shard_kernel<<<grid, block, 0, r.stream>>>(
                        r.d_final_hc_shard, r.d_next_hidden, dst, opt.slots);
                    r.hc_initialized = true;
                    CHECK_CUDA(cudaGetLastError());
                }
            }
            sync_all();
            if (opt.tp_hc_final_expand_gate) {
                if (!shared_hc_controls || !shared_hc_controls->initialized) {
                    return 6;
                }
                if (run_shared_hc_final_expand(opt, shared_hc_controls, ranks, opt.layer) != 0) {
                    return 7;
                }
            }
            if (should_log_reference_hc_window(opt)) {
                const uint64_t hc_shard_elems = shard_elems * 4ull;
                for (int dst = 0; dst < kGpus; ++dst) {
                    RankState &r = ranks[dst];
                    CHECK_CUDA(cudaSetDevice(r.device));
                    log_tensor_f32_stats("final_hc_shard", opt.layer, dst,
                                         r.d_final_hc_shard,
                                         (size_t)hc_shard_elems, r.stream);
                }
            }
            t4 = std::chrono::steady_clock::now();
        }
        *ep_ms += ep_stage_ms;
        *dense_ms += dense_stage_ms;
        *compose_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
        *compose_reduce_ms +=
            std::chrono::duration<double, std::milli>(t_reduce_done - t2).count();
        *compose_copy_ms +=
            std::chrono::duration<double, std::milli>(t_copy_done - t_reduce_done).count();
        *compose_final_ms +=
            std::chrono::duration<double, std::milli>(t3 - t_copy_done).count();
        *hc_current_input_ms += std::chrono::duration<double, std::milli>(t0 - t_pre).count();
        *final_hc_ms += std::chrono::duration<double, std::milli>(t4 - t3).count();
        return 0;
    };

    auto attempt_capture_probe = [&]() -> int {
        if (!opt.decode_cudagraph_gate) return 0;
        cudagraph_capture_attempted++;
        CaptureHostRankState saved[kGpus];
        save_capture_host_state(saved);

        cudaError_t first_error = cudaSuccess;
        const char *phase = "begin";
        const int root_device = ranks[0].device;
        cudaStream_t root_stream = ranks[0].stream;
        struct CaptureStream {
            int device = 0;
            cudaStream_t stream = nullptr;
        };
        std::vector<CaptureStream> streams;
        auto add_stream = [&](int device, cudaStream_t stream) {
            if (!stream) return;
            for (const CaptureStream &s : streams) {
                if (s.device == device && s.stream == stream) return;
            }
            streams.push_back(CaptureStream{device, stream});
        };
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            add_stream(r.device, r.stream);
            add_stream(r.device, r.dense_stream);
            add_stream(r.device, r.copy_stream);
            for (int q = 0; q < kGpus; ++q) {
                add_stream(r.device, r.copy_streams[q]);
            }
        }
        cudaEvent_t capture_seed = nullptr;
        CHECK_CUDA(cudaSetDevice(root_device));
        CHECK_CUDA(cudaEventCreateWithFlags(&capture_seed, cudaEventDisableTiming));
        cudaError_t rc = begin_capture_stream(root_device, root_stream);
        if (rc != cudaSuccess) first_error = rc;
        bool capture_begun = first_error == cudaSuccess;
        if (first_error == cudaSuccess) {
            phase = "join";
            rc = cudaEventRecord(capture_seed, root_stream);
            if (rc != cudaSuccess) first_error = rc;
            for (const CaptureStream &s : streams) {
                if (first_error != cudaSuccess) break;
                if (s.device == root_device && s.stream == root_stream) continue;
                rc = cudaSetDevice(s.device);
                if (rc != cudaSuccess) {
                    first_error = rc;
                    break;
                }
                rc = cudaStreamWaitEvent(s.stream, capture_seed, 0);
                if (rc != cudaSuccess) first_error = rc;
            }
        }

        int step_rc = 0;
        double cap_ep = 0.0;
        double cap_dense = 0.0;
        double cap_compose = 0.0;
        double cap_compose_reduce = 0.0;
        double cap_compose_copy = 0.0;
        double cap_compose_final = 0.0;
        double cap_hc_current = 0.0;
        double cap_final_hc = 0.0;
        if (first_error == cudaSuccess) {
            phase = "enqueue";
            step_rc = run_one_step(&cap_ep, &cap_dense, &cap_compose,
                                   &cap_compose_reduce, &cap_compose_copy,
                                   &cap_compose_final, &cap_hc_current,
                                   nullptr, nullptr, &cap_final_hc);
            if (step_rc != 0) {
                first_error = cudaErrorUnknown;
            }
        }

        phase = first_error == cudaSuccess ? "end" : phase;
        std::vector<cudaGraph_t> graphs;
        size_t node_count = 0;
        cudaGraph_t graph = nullptr;
        if (capture_begun) {
            rc = end_capture_stream(root_device, root_stream, &graph);
            if (rc == cudaSuccess && graph) {
                size_t graph_nodes = 0;
                cudaError_t count_rc = cudaGraphGetNodes(graph, nullptr,
                                                         &graph_nodes);
                if (count_rc == cudaSuccess) node_count += graph_nodes;
                graphs.push_back(graph);
            } else if (first_error == cudaSuccess) {
                first_error = rc;
            }
        }
        restore_capture_host_state(saved);
        CHECK_CUDA(cudaSetDevice(root_device));
        CHECK_CUDA(cudaEventDestroy(capture_seed));

        cudagraph_capture_error = (int)first_error;
        cudagraph_capture_nodes = node_count;
        if (first_error == cudaSuccess && step_rc == 0) {
            cudagraph_capture_succeeded++;
        }
        std::printf("tp_ep_decode_cudagraph_capture\tlayer\t%d\tstreams\t%zu\t"
                    "roots\t1\tattempted\t1\tsucceeded\t%d\terror_code\t%d\t"
                    "error_name\t%s\tphase\t%s\tnodes\t%zu\tstep_rc\t%d\n",
                    opt.layer, streams.size(),
                    first_error == cudaSuccess && step_rc == 0 ? 1 : 0,
                    (int)first_error, cudaGetErrorName(first_error), phase,
                    node_count, step_rc);
        destroy_capture_graphs(&graphs);
        return 0;
    };

    double warm_ep = 0.0;
    double warm_dense = 0.0;
    double warm_compose = 0.0;
    double warm_compose_reduce = 0.0;
    double warm_compose_copy = 0.0;
    double warm_compose_final = 0.0;
    double warm_hc_current_input = 0.0;
    double warm_final_hc = 0.0;
    for (int i = 0; i < opt.warmup; ++i) {
        if (run_one_step(&warm_ep, &warm_dense, &warm_compose,
                         &warm_compose_reduce, &warm_compose_copy,
                         &warm_compose_final, &warm_hc_current_input,
                         nullptr, nullptr,
                         &warm_final_hc) != 0) {
            if (!shared_dense_ops) {
                free_resident_f8_dense(attn, opt);
                free_resident_f8_dense(shared, opt);
            }
            return 3;
        }
    }

    double ep_ms = 0.0;
    double dense_ms = 0.0;
    double compose_ms = 0.0;
    double compose_reduce_ms = 0.0;
    double compose_copy_ms = 0.0;
    double compose_final_ms = 0.0;
    double hc_current_input_ms = 0.0;
    HcCurrentInputBreakdown hc_current_breakdown;
    PreEpPrefixBreakdown pre_ep_breakdown;
    double final_hc_ms = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < opt.decode_steps; ++i) {
        if (run_one_step(&ep_ms, &dense_ms, &compose_ms,
                         &compose_reduce_ms, &compose_copy_ms,
                         &compose_final_ms, &hc_current_input_ms,
                         &hc_current_breakdown, &pre_ep_breakdown,
                         &final_hc_ms) != 0) {
            if (!shared_dense_ops) {
                free_resident_f8_dense(attn, opt);
                free_resident_f8_dense(shared, opt);
            }
            return 4;
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    stats->total_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    stats->ms_per_step = stats->total_ms / (double)opt.decode_steps;
    stats->tok_s = stats->total_ms > 0.0
        ? (double)stats->slot_steps * 1000.0 / stats->total_ms
        : 0.0;
    stats->ep_ms_per_step = ep_ms / (double)opt.decode_steps;
    stats->dense_ms_per_step = dense_ms / (double)opt.decode_steps;
    stats->compose_ms_per_step = compose_ms / (double)opt.decode_steps;
    stats->compose_reduce_ms_per_step = compose_reduce_ms / (double)opt.decode_steps;
    stats->compose_copy_ms_per_step = compose_copy_ms / (double)opt.decode_steps;
    stats->compose_final_ms_per_step = compose_final_ms / (double)opt.decode_steps;
    stats->hc_current_input_ms_per_step = hc_current_input_ms / (double)opt.decode_steps;
    stats->hc_current_seed_ms_per_step =
        hc_current_breakdown.seed_ms / (double)opt.decode_steps;
    stats->hc_current_attn_mix_ms_per_step =
        hc_current_breakdown.attn_mix_ms / (double)opt.decode_steps;
    stats->hc_current_split_ms_per_step =
        hc_current_breakdown.split_ms / (double)opt.decode_steps;
    stats->hc_current_gather_ms_per_step =
        hc_current_breakdown.gather_ms / (double)opt.decode_steps;
    stats->hc_current_ffn_router_ms_per_step =
        hc_current_breakdown.ffn_router_ms / (double)opt.decode_steps;
    stats->hc_current_ffn_norm_ms_per_step =
        hc_current_breakdown.ffn_norm_ms / (double)opt.decode_steps;
    stats->hc_current_router_select_ms_per_step =
        hc_current_breakdown.router_select_ms / (double)opt.decode_steps;
    stats->hc_current_router_d2h_ms_per_step =
        hc_current_breakdown.router_d2h_ms / (double)opt.decode_steps;
    stats->hc_current_route_upload_ms_per_step =
        hc_current_breakdown.route_upload_ms / (double)opt.decode_steps;
    stats->hc_current_fill_pack_ms_per_step =
        hc_current_breakdown.fill_pack_ms / (double)opt.decode_steps;
    stats->pre_ep_hc_current_ms_per_step =
        pre_ep_breakdown.hc_current_ms / (double)opt.decode_steps;
    stats->pre_ep_attention_projection_ms_per_step =
        pre_ep_breakdown.attention_projection_ms / (double)opt.decode_steps;
    stats->pre_ep_compressed_kv_ms_per_step =
        pre_ep_breakdown.compressed_kv_ms / (double)opt.decode_steps;
    stats->pre_ep_attention_state_ms_per_step =
        pre_ep_breakdown.attention_state_ms / (double)opt.decode_steps;
    stats->pre_ep_typed_history_ms_per_step =
        pre_ep_breakdown.typed_history_ms / (double)opt.decode_steps;
    stats->pre_ep_raw_read_ms_per_step =
        pre_ep_breakdown.raw_read_ms / (double)opt.decode_steps;
    stats->pre_ep_attention_output_ms_per_step =
        pre_ep_breakdown.attention_output_ms / (double)opt.decode_steps;
    stats->pre_ep_post_attention_ffn_input_ms_per_step =
        pre_ep_breakdown.post_attention_ffn_input_ms / (double)opt.decode_steps;
    stats->final_hc_ms_per_step = final_hc_ms / (double)opt.decode_steps;
    stats->cudagraph_sync_all_calls = cudagraph_audit_sync_all_calls;
    stats->cudagraph_event_barrier_calls =
        cudagraph_audit_event_barrier_calls;
    stats->cudagraph_rank_stream_syncs = cudagraph_audit_stream_syncs;
    stats->cudagraph_dense_stream_syncs = cudagraph_audit_dense_stream_syncs;
    stats->cudagraph_copy_stream_syncs = cudagraph_audit_copy_stream_syncs;

    if (opt.skip_decode_checksum) {
        stats->checksum = 0xD54D0000ull ^
                          ((uint64_t)(opt.layer + 1) * 1000003ull) ^
                          ((uint64_t)(opt.position + 1) * 9176ull) ^
                          ((uint64_t)opt.slots * 65537ull);
    } else {
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            std::vector<float> host((size_t)shard_elems);
            CHECK_CUDA(cudaMemcpy(host.data(), r.d_next_hidden, (size_t)shard_bytes,
                                  cudaMemcpyDeviceToHost));
            for (uint64_t i = 0; i < shard_elems; ++i) {
                const float v = host[(size_t)i];
                if (!std::isfinite(v)) {
                    stats->finite_bad++;
                    stats->pass = false;
                }
                uint32_t bits = 0;
                std::memcpy(&bits, &v, sizeof(bits));
                stats->checksum ^=
                    (uint64_t)bits + (uint64_t)(dst + 1) * 2000003ull + i * 7907ull;
            }
        }
    }
    if (stats->checksum == 0 || stats->finite_bad != 0) stats->pass = false;
    if (opt.final_hc_carry_gate) {
        uint64_t hc_checksum = 0;
        const uint64_t hc_shard_elems = shard_elems * 4ull;
        const uint64_t hc_shard_bytes = hc_shard_elems * sizeof(float);
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            if (!r.d_final_hc_shard) {
                stats->pass = false;
                continue;
            }
            std::vector<float> host((size_t)hc_shard_elems);
            CHECK_CUDA(cudaMemcpy(host.data(), r.d_final_hc_shard,
                                  (size_t)hc_shard_bytes,
                                  cudaMemcpyDeviceToHost));
            for (uint64_t i = 0; i < hc_shard_elems; ++i) {
                const float v = host[(size_t)i];
                if (!std::isfinite(v)) {
                    stats->finite_bad++;
                    stats->pass = false;
                }
                uint32_t bits = 0;
                std::memcpy(&bits, &v, sizeof(bits));
                hc_checksum ^=
                    (uint64_t)bits + (uint64_t)(dst + 1) * 3000017ull + i * 8191ull;
            }
        }
        if (hc_checksum == 0) stats->pass = false;
        stats->checksum ^= hc_checksum + 0xF17A1C00ull;
    }
    if (stats->checksum == 0 || stats->finite_bad != 0) stats->pass = false;

    if (opt.decode_cudagraph_gate && stats->pass) {
        const int cap_rc = attempt_capture_probe();
        if (cap_rc != 0) {
            stats->pass = false;
        }
    }
    stats->cudagraph_capture_attempted = cudagraph_capture_attempted;
    stats->cudagraph_capture_succeeded = cudagraph_capture_succeeded;
    stats->cudagraph_capture_error = cudagraph_capture_error;
    stats->cudagraph_capture_nodes = cudagraph_capture_nodes;

    if (!shared_dense_ops) {
        free_resident_f8_dense(attn, opt);
        free_resident_f8_dense(shared, opt);
    }
    return stats->pass ? 0 : 5;
}

} // namespace

int run_resident_layer_decode(const Options &opt,
                              const std::vector<ContractRow> &rows,
                              const LayerStats &layer_stats,
                              RankState ranks[kGpus],
                              const Api &api,
                              ds4_v100_tp_runtime *rt,
                              const LayerExpertCache *layer_expert_cache,
                              const DenseF16Cache *dense_f16_cache,
                              const LayerDenseOps *layer_dense_ops,
                              SharedHcControls *shared_hc_controls,
                              LayerRunSummary *summary) {
    if (!rt || !layer_expert_cache || !dense_f16_cache) return 2;

    char err[512] = {0};
    ds4_v100_tp_dense_kv_result kv_result;
    const int write_indexer = ds4_layer_ratio(opt.layer) == 4 ? 1 : 0;
    const uint32_t kv_first_slot = opt.tp_kv_all_slots_gate ? 0u : opt.kv_slot;
    const uint32_t kv_end_slot = opt.tp_kv_all_slots_gate ? (uint32_t)opt.slots : opt.kv_slot + 1u;
    for (uint32_t slot = kv_first_slot; slot < kv_end_slot; ++slot) {
        if (ds4_v100_tp_runtime_dense_kv_slice(rt, opt.layer, slot, opt.position,
                                               write_indexer, &kv_result, err,
                                               sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_dense_kv_slice_failed\tslot\t%u\t%s\n",
                         slot, err);
            return 3;
        }
        if (kv_result.max_abs != 0.0) return 4;
    }

    for (int p = 0; p < kGpus; ++p) {
        ranks[p].gated = layer_expert_cache->gated[p];
        ranks[p].down = layer_expert_cache->down[p];
    }

    DecodeLoopStats decode_loop;
    const int rc = run_decode_loop(opt, rows, ranks, api, rt, dense_f16_cache,
                                   layer_dense_ops, shared_hc_controls,
                                   &decode_loop);

    for (int p = 0; p < kGpus; ++p) {
        ranks[p].gated = PackedExperts{};
        ranks[p].down = PackedExperts{};
    }

    if (summary) {
        summary->layer = opt.layer;
        summary->ratio = ds4_layer_ratio(opt.layer);
        summary->pass = rc == 0 && decode_loop.pass;
        summary->total_rows = layer_stats.total_rows;
        summary->dense_rows = layer_stats.dense_rows;
        summary->control_rows = layer_stats.control_rows;
        summary->expert_rows = layer_stats.expert_rows;
        summary->kv_rows = layer_stats.kv_rows;
        summary->comp_rows = layer_stats.comp_rows;
        summary->decode_ms_per_step = decode_loop.ms_per_step;
        summary->decode_slot_step_tok_s = decode_loop.tok_s;
        summary->decode_ep_ms_per_step = decode_loop.ep_ms_per_step;
        summary->decode_dense_ms_per_step = decode_loop.dense_ms_per_step;
        summary->decode_compose_ms_per_step = decode_loop.compose_ms_per_step;
        summary->decode_compose_reduce_ms_per_step =
            decode_loop.compose_reduce_ms_per_step;
        summary->decode_compose_copy_ms_per_step =
            decode_loop.compose_copy_ms_per_step;
        summary->decode_compose_final_ms_per_step =
            decode_loop.compose_final_ms_per_step;
        summary->decode_hc_current_input_ms_per_step =
            decode_loop.hc_current_input_ms_per_step;
        summary->decode_hc_current_seed_ms_per_step =
            decode_loop.hc_current_seed_ms_per_step;
        summary->decode_hc_current_attn_mix_ms_per_step =
            decode_loop.hc_current_attn_mix_ms_per_step;
        summary->decode_hc_current_split_ms_per_step =
            decode_loop.hc_current_split_ms_per_step;
        summary->decode_hc_current_gather_ms_per_step =
            decode_loop.hc_current_gather_ms_per_step;
        summary->decode_hc_current_ffn_router_ms_per_step =
            decode_loop.hc_current_ffn_router_ms_per_step;
        summary->decode_hc_current_ffn_norm_ms_per_step =
            decode_loop.hc_current_ffn_norm_ms_per_step;
        summary->decode_hc_current_router_select_ms_per_step =
            decode_loop.hc_current_router_select_ms_per_step;
        summary->decode_hc_current_router_d2h_ms_per_step =
            decode_loop.hc_current_router_d2h_ms_per_step;
        summary->decode_hc_current_route_upload_ms_per_step =
            decode_loop.hc_current_route_upload_ms_per_step;
        summary->decode_hc_current_fill_pack_ms_per_step =
            decode_loop.hc_current_fill_pack_ms_per_step;
        summary->decode_pre_ep_hc_current_ms_per_step =
            decode_loop.pre_ep_hc_current_ms_per_step;
        summary->decode_pre_ep_attention_projection_ms_per_step =
            decode_loop.pre_ep_attention_projection_ms_per_step;
        summary->decode_pre_ep_compressed_kv_ms_per_step =
            decode_loop.pre_ep_compressed_kv_ms_per_step;
        summary->decode_pre_ep_attention_state_ms_per_step =
            decode_loop.pre_ep_attention_state_ms_per_step;
        summary->decode_pre_ep_typed_history_ms_per_step =
            decode_loop.pre_ep_typed_history_ms_per_step;
        summary->decode_pre_ep_raw_read_ms_per_step =
            decode_loop.pre_ep_raw_read_ms_per_step;
        summary->decode_pre_ep_attention_output_ms_per_step =
            decode_loop.pre_ep_attention_output_ms_per_step;
        summary->decode_pre_ep_post_attention_ffn_input_ms_per_step =
            decode_loop.pre_ep_post_attention_ffn_input_ms_per_step;
        summary->decode_final_hc_ms_per_step = decode_loop.final_hc_ms_per_step;
        summary->decode_cudagraph_sync_all_calls =
            decode_loop.cudagraph_sync_all_calls;
        summary->decode_cudagraph_event_barrier_calls =
            decode_loop.cudagraph_event_barrier_calls;
        summary->decode_cudagraph_rank_stream_syncs =
            decode_loop.cudagraph_rank_stream_syncs;
        summary->decode_cudagraph_dense_stream_syncs =
            decode_loop.cudagraph_dense_stream_syncs;
        summary->decode_cudagraph_copy_stream_syncs =
            decode_loop.cudagraph_copy_stream_syncs;
        summary->decode_cudagraph_capture_attempted =
            decode_loop.cudagraph_capture_attempted;
        summary->decode_cudagraph_capture_succeeded =
            decode_loop.cudagraph_capture_succeeded;
        summary->decode_cudagraph_capture_error =
            decode_loop.cudagraph_capture_error;
        summary->decode_cudagraph_capture_nodes =
            decode_loop.cudagraph_capture_nodes;
        summary->decode_checksum = decode_loop.checksum;
        summary->decode_finite_bad = decode_loop.finite_bad;
        summary->rc = rc;
    }
    return rc;
}

int run_layer(const Options &opt,
              LayerRunSummary *summary,
              const DenseF16Cache *shared_dense_f16_cache,
              const SharedApi *shared_api,
              SharedRankBuffers *shared_rank_buffers,
              SharedTpRuntime *shared_tp_runtime,
              const SharedExpertBindings *shared_expert_bindings,
              const SharedDenseOps *shared_dense_ops,
              SharedHcControls *shared_hc_controls) {
    std::vector<ContractRow> rows;
    LayerStats layer_stats;
    if (parse_contract(opt.contract_path, opt.layer, &rows, &layer_stats) != 0 ||
        layer_stats.bad_rows != 0) {
        std::fprintf(stderr, "contract parse failed bad_rows=%llu\n",
                     (unsigned long long)layer_stats.bad_rows);
        return 2;
    }
    DescriptorBindings bindings;
    const LayerExpertCache *layer_expert_cache = nullptr;
    if (shared_expert_bindings) {
        layer_expert_cache = &shared_expert_bindings->layers[opt.layer];
        bindings = layer_expert_cache->bindings;
    } else {
        if (parse_tm_index(opt.tm_index_path, opt.layer, &bindings) != 0) {
            std::fprintf(stderr, "tm index parse failed for layer %d\n", opt.layer);
            return 2;
        }
    }

    const auto descriptor_start = std::chrono::steady_clock::now();
    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" && r.record_type != "replicated_control") continue;
        if (!opt.skip_descriptor_checks) {
            uint64_t checksum = 0;
            if (device_checksum_row(opt.devices[r.owning_gpu], opt.pack_dir, r, &checksum) != 0) {
                return 3;
            }
            layer_stats.gpu[r.owning_gpu].checksum ^=
                checksum + (uint64_t)(r.owning_gpu + 1) * 131u;
            layer_stats.checksum ^= checksum + (uint64_t)(r.owning_gpu + 1) * 257u;
        }
        if (r.record_type == "dense_tp") layer_stats.dense_loaded_bytes += r.bytes_estimate;
        else layer_stats.control_loaded_bytes += r.bytes_estimate;
    }
    const auto descriptor_stop = std::chrono::steady_clock::now();
    const double descriptor_ms =
        std::chrono::duration<double, std::milli>(descriptor_stop - descriptor_start).count();

    DenseComputeStats dense_compute;
    DenseComputeStats bf16_compute;
    std::vector<DenseComputeStats> dense_compute_results;
    std::vector<DenseComputeStats> bf16_compute_results;
    std::vector<std::string> dense_tensors;
    if (opt.dense_compute_all_f8) {
        dense_tensors = discover_f8_dense_tensors(rows);
    } else if (opt.dense_compute_tensor) {
        dense_tensors.emplace_back(opt.dense_compute_tensor);
    }
    for (const std::string &tensor : dense_tensors) {
        DenseComputeStats one;
        if (run_dense_compute_gate(opt, rows, tensor.c_str(), &one) != 0) {
            std::fprintf(stderr, "dense compute gate failed for %s\n", tensor.c_str());
            return 3;
        }
        std::printf("dense_compute_tensor\ttensor\t%s\trows_per_gpu\t%d\tcols\t%d\t"
                    "slots\t%d\tloaded_bytes\t%llu\tcompute_ms\t%.6f\t"
                    "repeat_max_abs\t%.9f\trepeat_bad\t%d\trepeat_nan\t%d\t"
                    "oracle_max_abs\t%.9f\toracle_bad\t%d\t%s\n",
                    one.tensor_id.c_str(), one.rows_per_gpu, one.cols, one.slots,
                    (unsigned long long)one.loaded_bytes, one.compute_ms,
                    one.repeat_max_abs, one.repeat_bad, one.repeat_nan,
                    one.oracle_max_abs, one.oracle_bad, one.pass ? "PASS" : "FAIL");
        dense_compute_results.push_back(one);
        dense_compute.enabled = true;
        dense_compute.tensor_id = opt.dense_compute_all_f8 ? "all_f8" : one.tensor_id;
        dense_compute.rows_per_gpu = std::max(dense_compute.rows_per_gpu, one.rows_per_gpu);
        dense_compute.cols = std::max(dense_compute.cols, one.cols);
        dense_compute.slots = one.slots;
        dense_compute.loaded_bytes += one.loaded_bytes;
        dense_compute.compute_ms = std::max(dense_compute.compute_ms, one.compute_ms);
        dense_compute.repeat_max_abs =
            std::max(dense_compute.repeat_max_abs, one.repeat_max_abs);
        dense_compute.oracle_max_abs =
            std::max(dense_compute.oracle_max_abs, one.oracle_max_abs);
        dense_compute.repeat_bad += one.repeat_bad;
        dense_compute.repeat_nan += one.repeat_nan;
        dense_compute.oracle_bad += one.oracle_bad;
        dense_compute.pass = dense_compute.pass && one.pass;
    }
    std::vector<std::string> bf16_tensors;
    if (opt.dense_compute_all_bf16) {
        bf16_tensors = discover_bf16_dense_tensors(rows);
    }
    for (const std::string &tensor : bf16_tensors) {
        DenseComputeStats one;
        if (run_bf16_dense_compute_gate(opt, rows, tensor.c_str(), &one) != 0) {
            std::fprintf(stderr, "bf16 dense compute gate failed for %s\n", tensor.c_str());
            return 3;
        }
        std::printf("bf16_dense_compute_tensor\ttensor\t%s\trows_per_gpu\t%d\tcols\t%d\t"
                    "slots\t%d\tloaded_bytes\t%llu\tcompute_ms\t%.6f\t"
                    "repeat_max_abs\t%.9f\trepeat_bad\t%d\trepeat_nan\t%d\t"
                    "oracle_max_abs\t%.9f\toracle_bad\t%d\t%s\n",
                    one.tensor_id.c_str(), one.rows_per_gpu, one.cols, one.slots,
                    (unsigned long long)one.loaded_bytes, one.compute_ms,
                    one.repeat_max_abs, one.repeat_bad, one.repeat_nan,
                    one.oracle_max_abs, one.oracle_bad, one.pass ? "PASS" : "FAIL");
        bf16_compute_results.push_back(one);
        bf16_compute.enabled = true;
        bf16_compute.tensor_id = "all_bf16";
        bf16_compute.rows_per_gpu = std::max(bf16_compute.rows_per_gpu, one.rows_per_gpu);
        bf16_compute.cols = std::max(bf16_compute.cols, one.cols);
        bf16_compute.slots = one.slots;
        bf16_compute.loaded_bytes += one.loaded_bytes;
        bf16_compute.compute_ms = std::max(bf16_compute.compute_ms, one.compute_ms);
        bf16_compute.repeat_max_abs =
            std::max(bf16_compute.repeat_max_abs, one.repeat_max_abs);
        bf16_compute.oracle_max_abs =
            std::max(bf16_compute.oracle_max_abs, one.oracle_max_abs);
        bf16_compute.repeat_bad += one.repeat_bad;
        bf16_compute.repeat_nan += one.repeat_nan;
        bf16_compute.oracle_bad += one.oracle_bad;
        bf16_compute.pass = bf16_compute.pass && one.pass;
    }

    DenseF16Cache local_dense_f16_cache;
    const DenseF16Cache *dense_f16_cache = shared_dense_f16_cache;
    if (!dense_f16_cache) {
        if (prepare_dense_f16_cache(opt, rows, &local_dense_f16_cache) != 0) {
            std::fprintf(stderr, "dense f16 cache prepare failed\n");
            return 4;
        }
        dense_f16_cache = &local_dense_f16_cache;
    }
    if (!shared_dense_f16_cache && dense_f16_cache->enabled) {
        std::printf("tp_ep_dense_f16_cache\tlayer\t%d\trows\t%llu\t"
                    "source_bytes\t%llu\tcache_bytes\t%llu\t"
                    "cache_aligned_bytes\t%llu\tmax_temp_bytes\t%llu\tPASS\n",
                    opt.layer,
                    (unsigned long long)dense_f16_cache->rows,
                    (unsigned long long)dense_f16_cache->source_bytes,
                    (unsigned long long)dense_f16_cache->cache_bytes,
                    (unsigned long long)dense_f16_cache->cache_aligned_bytes,
                    (unsigned long long)dense_f16_cache->max_temp_bytes);
    }

    ds4_v100_tp_runtime_config cfg;
    fill_tp_runtime_config(opt, &cfg);

    char err[512] = {0};
    ds4_v100_tp_runtime *rt = nullptr;
    ds4_v100_tp_runtime_report runtime_report;
    if (shared_tp_runtime) {
        rt = shared_tp_runtime->rt;
        runtime_report = shared_tp_runtime->report;
    } else {
        if (ds4_v100_tp_runtime_open(&rt, &cfg, err, sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_open_failed\t%s\n", err);
            return 4;
        }
        ds4_v100_tp_runtime_get_report(rt, &runtime_report);
    }
    auto close_local_runtime = [&]() {
        if (!shared_tp_runtime && rt) ds4_v100_tp_runtime_close(rt);
    };

    ds4_v100_tp_dense_kv_result kv_result;
    const auto kv_start = std::chrono::steady_clock::now();
    const int write_indexer = ds4_layer_ratio(opt.layer) == 4 ? 1 : 0;
    if (ds4_v100_tp_runtime_dense_kv_slice(rt, opt.layer, opt.kv_slot, opt.position,
                                           write_indexer, &kv_result, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "tp_runtime_dense_kv_slice_failed\t%s\n", err);
        close_local_runtime();
        return 5;
    }
    const auto kv_stop = std::chrono::steady_clock::now();
    const double dense_kv_ms =
        std::chrono::duration<double, std::milli>(kv_stop - kv_start).count();

    void *lib = nullptr;
    Api local_api;
    const Api *api = nullptr;
    if (shared_api) {
        api = &shared_api->api;
    } else {
        lib = dlopen(opt.lib_path, RTLD_LAZY | RTLD_LOCAL);
        if (!lib) {
            std::fprintf(stderr, "dlopen failed for %s: %s\n", opt.lib_path, dlerror());
            close_local_runtime();
            return 6;
        }
        load_api(lib, &local_api);
        api = &local_api;
    }

    RankState local_ranks[kGpus];
    RankState *ranks = shared_rank_buffers ? shared_rank_buffers->ranks : local_ranks;
    int aggregate_routes = 0;
    int min_routes = std::numeric_limits<int>::max();
    int max_routes = 0;
    uint64_t ep_loaded_bytes = 0;

    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        r.rank = p;
        r.device = opt.devices[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!shared_api && api->init(r.device) != 0) {
            std::fprintf(stderr, "ggml_turbomind_init failed on device %d\n", r.device);
            if (!shared_api) {
                api->shutdown();
                dlclose(lib);
            }
            close_local_runtime();
            return 7;
        }
        if (!shared_rank_buffers) {
            CHECK_CUDA(cudaStreamCreate(&r.stream));
            CHECK_CUDA(cudaStreamCreate(&r.dense_stream));
            CHECK_CUDA(cudaStreamCreate(&r.copy_stream));
            for (int q = 0; q < kGpus; ++q) {
                CHECK_CUDA(cudaStreamCreate(&r.copy_streams[q]));
                CHECK_CUDA(cudaEventCreateWithFlags(&r.copy_done[q], cudaEventDisableTiming));
            }
            CHECK_CUDA(cudaEventCreateWithFlags(&r.stream_done, cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&r.dense_done, cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&r.dense_wait, cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreate(&r.start));
            CHECK_CUDA(cudaEventCreate(&r.mid));
            CHECK_CUDA(cudaEventCreate(&r.stop));
            r.route_compact_plan_ints = compact_route_plan_ints(opt);
            CHECK_CUDA(cudaMalloc(&r.d_route_compact_plan,
                                  r.route_compact_plan_ints * sizeof(int)));
            bind_compact_route_plan(&r, opt);
            CHECK_CUDA(cudaMalloc(&r.d_router_selected_plan,
                                  (size_t)opt.slots * (size_t)opt.top_k * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&r.d_router_weights_plan,
                                  (size_t)opt.slots * (size_t)opt.top_k * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&r.d_route_offsets_all,
                                  (size_t)kGpus * (size_t)(kLocalExperts + 1) *
                                      sizeof(int)));
            CHECK_CUDA(cudaMalloc(&r.d_route_totals,
                                  (size_t)kGpus * sizeof(int)));
            std::vector<int> compact_plan(r.route_compact_plan_ints, -1);
            const size_t compact_indices = (size_t)opt.slots * (size_t)opt.top_k;
            const size_t compact_counts = (size_t)opt.slots;
            for (int src = 0; src < kGpus; ++src) {
                std::vector<int> route_index_by_slot;
                build_route_index_by_slot_for_rank(src, opt.slots, opt.top_k,
                                                   &route_index_by_slot);
                CHECK_CUDA(cudaMalloc(&r.d_route_index_by_slot[src],
                                      route_index_by_slot.size() * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(r.d_route_index_by_slot[src],
                                      route_index_by_slot.data(),
                                      route_index_by_slot.size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
                std::vector<int> route_indices_by_slot;
                std::vector<int> route_count_by_slot;
                build_route_indices_by_slot_for_rank(src, opt.slots, opt.top_k,
                                                     &route_indices_by_slot,
                                                     &route_count_by_slot);
                std::copy(route_indices_by_slot.begin(), route_indices_by_slot.end(),
                          compact_plan.begin() + (size_t)src * compact_indices);
                std::copy(route_count_by_slot.begin(), route_count_by_slot.end(),
                          compact_plan.begin() + (size_t)kGpus * compact_indices +
                              (size_t)src * compact_counts);
            }
            CHECK_CUDA(cudaMemcpy(r.d_route_compact_plan, compact_plan.data(),
                                  compact_plan.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));

            std::vector<int> offsets;
            std::vector<int> route_slots;
            std::vector<float> route_weights;
            build_offsets_for_rank(p, opt.slots, opt.top_k, &offsets, &route_slots,
                                   &route_weights, &r.routes, &r.active_experts,
                                   &r.max_routes_per_expert);

            r.route_capacity = opt.slots * opt.top_k;
            const size_t route_capacity_elems = (size_t)r.route_capacity * kHidden;
            CHECK_CUDA(cudaMalloc(&r.d_offsets, offsets.size() * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(r.d_offsets, offsets.data(), offsets.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&r.d_route_slots,
                                  (size_t)r.route_capacity * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(r.d_route_slots, route_slots.data(),
                                  route_slots.size() * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&r.d_route_weights,
                                  (size_t)r.route_capacity * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(r.d_route_weights, route_weights.data(),
                                  route_weights.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&r.d_route_inv_scale,
                                  (size_t)r.route_capacity * sizeof(float)));
            std::vector<float> route_inv_scale((size_t)r.route_capacity, 1.0f);
            CHECK_CUDA(cudaMemcpy(r.d_route_inv_scale, route_inv_scale.data(),
                                  route_inv_scale.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&r.d_a, route_capacity_elems * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&r.d_gate_up,
                                  (size_t)r.route_capacity * kFusedN * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&r.d_gated,
                                  (size_t)r.route_capacity * kMid * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&r.d_down, route_capacity_elems * sizeof(__half)));

            std::mt19937 rng(0xE2350000u + (uint32_t)p * 97u);
            std::uniform_real_distribution<float> dist(-0.003f, 0.003f);
            std::vector<__half> h_a(route_capacity_elems);
            for (__half &v : h_a) v = __float2half(dist(rng));
            CHECK_CUDA(cudaMemcpy(r.d_a, h_a.data(),
                                  route_capacity_elems * sizeof(__half),
                                  cudaMemcpyHostToDevice));
        }
        aggregate_routes += r.routes;
        min_routes = std::min(min_routes, r.routes);
        max_routes = std::max(max_routes, r.routes);

        if (layer_expert_cache) {
            r.gated = layer_expert_cache->gated[p];
            r.down = layer_expert_cache->down[p];
            ep_loaded_bytes += layer_expert_cache->gated[p].d_w_active.size()
                ? layer_expert_cache->bytes / kGpus
                : 0;
        } else {
            std::vector<int> active;
            for (int e = 0; e < kPackedLocalExperts; ++e) active.push_back(e);
            if (pack_descriptor_set(r.device, bindings.gated, p, active, opt.pack_dir,
                                    &r.gated, &ep_loaded_bytes) != 0 ||
                pack_descriptor_set(r.device, bindings.down, p, active, opt.pack_dir,
                                   &r.down, &ep_loaded_bytes) != 0) {
                close_local_runtime();
                return 8;
            }
        }
        layer_stats.gpu[p].ep_loaded_bytes = ep_loaded_bytes;
    }
    layer_stats.ep_loaded_bytes = ep_loaded_bytes;

    if (!shared_rank_buffers && open_compose_nccl(opt, ranks) != 0) {
        close_local_runtime();
        return 8;
    }

    if (!opt.skip_predecode_probes) {
        for (int i = 0; i < opt.warmup; ++i) {
            for (int p = 0; p < kGpus; ++p) {
                const int gate_rc = run_gate_selected(ranks[p], *api, opt);
                if (gate_rc != 0 || run_down(ranks[p], *api) != 0) {
                    close_local_runtime();
                    return 9;
                }
            }
            for (int p = 0; p < kGpus; ++p) {
                CHECK_CUDA(cudaSetDevice(ranks[p].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[p].stream));
            }
        }

        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_CUDA(cudaEventRecord(ranks[p].start, ranks[p].stream));
        }
        for (int i = 0; i < opt.iters; ++i) {
            for (int p = 0; p < kGpus; ++p) {
                const int gate_rc = run_gate_selected(ranks[p], *api, opt);
                if (gate_rc != 0) return 10;
            }
        }
        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_CUDA(cudaEventRecord(ranks[p].mid, ranks[p].stream));
        }
        for (int i = 0; i < opt.iters; ++i) {
            for (int p = 0; p < kGpus; ++p) {
                if (run_down(ranks[p], *api) != 0) return 11;
            }
        }
        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_CUDA(cudaEventRecord(ranks[p].stop, ranks[p].stream));
        }
    }

    double worst_gate_ms = 0.0;
    double worst_down_ms = 0.0;
    double worst_ep_ms = 0.0;
    for (int p = 0; p < kGpus; ++p) {
        CHECK_CUDA(cudaSetDevice(ranks[p].device));
        double gate_ms = 0.0;
        double down_ms = 0.0;
        if (!opt.skip_predecode_probes) {
            CHECK_CUDA(cudaEventSynchronize(ranks[p].stop));
            gate_ms = (double)elapsed_ms(ranks[p].start, ranks[p].mid) / opt.iters;
            down_ms = (double)elapsed_ms(ranks[p].mid, ranks[p].stop) / opt.iters;
        }
        worst_gate_ms = std::max(worst_gate_ms, gate_ms);
        worst_down_ms = std::max(worst_down_ms, down_ms);
        worst_ep_ms = std::max(worst_ep_ms, gate_ms + down_ms);
        std::printf("rank\t%d\tdevice\t%d\troutes\t%d\troute_capacity\t%d\t"
                    "active_local_experts\t%d\t"
                    "max_routes_per_expert\t%d\tgate_ms\t%.6f\tdown_ms\t%.6f\t"
                    "ep_ms\t%.6f\tdense_rows\t%llu\tcontrol_rows\t%llu\t"
                    "expert_rows\t%llu\tkv_rows\t%llu\tcomp_rows\t%llu\t"
                    "checksum\t%llu\n",
                    p, ranks[p].device, ranks[p].routes, ranks[p].route_capacity,
                    ranks[p].active_experts,
                    ranks[p].max_routes_per_expert, gate_ms, down_ms, gate_ms + down_ms,
                    (unsigned long long)layer_stats.gpu[p].dense_rows,
                    (unsigned long long)layer_stats.gpu[p].control_rows,
                    (unsigned long long)layer_stats.gpu[p].expert_rows,
                    (unsigned long long)layer_stats.gpu[p].kv_rows,
                    (unsigned long long)layer_stats.gpu[p].comp_rows,
                    (unsigned long long)layer_stats.gpu[p].checksum);
    }

    double repeat_max_abs = 0.0;
    int repeat_bad = 0;
    int repeat_nan = 0;
    if (!opt.skip_predecode_probes) {
        for (int p = 0; p < kGpus; ++p) {
            if (check_repeat(ranks[p], *api, &repeat_max_abs, &repeat_bad, &repeat_nan) != 0) {
                close_local_runtime();
                return 12;
            }
        }
    }

    ComposeStats compose;
    const int compose_rc = run_next_hidden_compose(opt, rows, ranks, &compose);
    if (compose.enabled) {
        std::printf("tp_ep_next_hidden_compose\tslots\t%d\tctx\t%llu\t"
                    "hidden_shard\t%d\tep_contribution_bytes\t%llu\t"
                    "ep_return_dtype\t%s\tep_return_bytes\t%llu\tdense_hmma\t%d\t"
                    "dense_f16_cublas\t%d\t"
                    "attn_dense_ms\t%.6f\t"
                    "shared_dense_ms\t%.6f\tfused_compose_sum\t%d\t"
                    "nccl_reduce_scatter\t%d\tcompose_ms\t%.6f\t"
                    "checksum\t%llu\tfinite_bad\t%d\trepeat_max_abs\t%.9f\t"
                    "repeat_bad\t%d\t%s\n",
                    opt.slots, (unsigned long long)cfg.ctx, kHidden / kGpus,
                    (unsigned long long)compose.ep_contribution_bytes,
                    compose.ep_return_fp16 ? "fp16" : "fp32",
                    (unsigned long long)compose.ep_return_bytes,
                    compose.dense_hmma_compose ? 1 : 0,
                    compose.dense_f16_cublas_compose ? 1 : 0,
                    compose.attn_dense_ms, compose.shared_dense_ms,
                    compose.fused_compose_sum ? 1 : 0,
                    compose.nccl_reduce_scatter_compose ? 1 : 0,
                    compose.compose_ms, (unsigned long long)compose.checksum,
                    compose.finite_bad, compose.repeat_max_abs,
                    compose.repeat_bad, compose.pass ? "PASS" : "FAIL");
    }
    if (compose_rc != 0) {
        close_local_runtime();
        return 13;
    }

    DecodeLoopStats decode_loop;
    const LayerDenseOps *layer_dense_ops =
        shared_dense_ops && shared_dense_ops->initialized
            ? &shared_dense_ops->layers[opt.layer]
            : nullptr;
    const int decode_rc = run_decode_loop(opt, rows, ranks, *api, rt, dense_f16_cache,
                                          layer_dense_ops, shared_hc_controls,
                                          &decode_loop);
    if (decode_loop.enabled) {
        std::printf("tp_ep_decode_loop\tsteps\t%d\tslots\t%d\tslot_steps\t%llu\t"
                    "total_ms\t%.6f\tms_per_step\t%.6f\tslot_step_tok_s\t%.6f\t"
                    "dense_hmma\t%d\tdense_f16_cublas\t%d\tdense_f16_cache\t%d\t"
                    "overlap_ep_dense\t%d\tdirect_remote_compose\t%d\t"
                    "source_copy_schedule\t%d\tskip_self_compose_copy\t%d\t"
                    "multi_copy_streams\t%d\t"
                    "ep_ms_per_step\t%.6f\tdense_ms_per_step\t%.6f\t"
                    "fused_compose_sum\t%d\tnccl_reduce_scatter\t%d\t"
                    "compose_ms_per_step\t%.6f\t"
                    "compose_reduce_ms_per_step\t%.6f\t"
                    "compose_copy_ms_per_step\t%.6f\t"
                    "compose_final_ms_per_step\t%.6f\t"
                    "hc_current_input_gate\t%d\t"
                    "hc_current_input_peer_gather\t%d\t"
                    "hc_current_input_nccl_allgather\t%d\t"
                    "hc_current_input_stream_sync\t%d\t"
                    "hc_current_input_ms_per_step\t%.6f\t"
                    "final_hc_carry_gate\t%d\tfinal_hc_ms_per_step\t%.6f\t"
                    "dense_loaded_bytes\t%llu\t"
                    "ep_contribution_bytes\t%llu\tep_return_dtype\t%s\t"
                    "ep_return_bytes\t%llu\t"
                    "checksum\t%llu\tfinite_bad\t%d\t%s\n",
                    decode_loop.steps, decode_loop.slots,
                    (unsigned long long)decode_loop.slot_steps,
                    decode_loop.total_ms, decode_loop.ms_per_step,
                    decode_loop.tok_s,
                    decode_loop.dense_hmma_compose ? 1 : 0,
                    decode_loop.dense_f16_cublas_compose ? 1 : 0,
                    decode_loop.dense_f16_cache_compose ? 1 : 0,
                    opt.overlap_ep_dense ? 1 : 0,
                    opt.direct_remote_compose ? 1 : 0,
                    opt.source_copy_schedule ? 1 : 0,
                    opt.skip_self_compose_copy ? 1 : 0,
                    opt.multi_copy_streams ? 1 : 0,
                    decode_loop.ep_ms_per_step,
                    decode_loop.dense_ms_per_step,
                    decode_loop.fused_compose_sum ? 1 : 0,
                    decode_loop.nccl_reduce_scatter_compose ? 1 : 0,
                    decode_loop.compose_ms_per_step,
                    decode_loop.compose_reduce_ms_per_step,
                    decode_loop.compose_copy_ms_per_step,
                    decode_loop.compose_final_ms_per_step,
                    opt.tp_hc_current_input_gate ? 1 : 0,
                    opt.tp_hc_current_input_peer_gather_gate ? 1 : 0,
                    opt.tp_hc_current_input_nccl_allgather_gate ? 1 : 0,
                    opt.tp_hc_current_input_stream_sync_gate ? 1 : 0,
                    decode_loop.hc_current_input_ms_per_step,
                    opt.final_hc_carry_gate ? 1 : 0,
                    decode_loop.final_hc_ms_per_step,
                    (unsigned long long)decode_loop.dense_loaded_bytes,
                    (unsigned long long)decode_loop.ep_contribution_bytes,
                    decode_loop.ep_return_fp16 ? "fp16" : "fp32",
                    (unsigned long long)decode_loop.ep_return_bytes,
                    (unsigned long long)decode_loop.checksum,
                    decode_loop.finite_bad,
                    decode_loop.pass ? "PASS" : "FAIL");
    }
    if (decode_rc != 0) {
        close_local_runtime();
        return 14;
    }

    const uint64_t dispatch_bytes = (uint64_t)aggregate_routes * kHidden * sizeof(__half);
    const uint64_t return_bytes = dispatch_bytes;
    const double imbalance = min_routes > 0 ? (double)max_routes / (double)min_routes : 0.0;
    const double scaffold_ms = descriptor_ms + dense_kv_ms + worst_ep_ms;
    const bool comp_rows_expected = ds4_layer_ratio(opt.layer) != 0;
    const bool pass = layer_stats.dense_rows > 0 &&
                      layer_stats.control_rows > 0 &&
                      layer_stats.expert_rows > 0 &&
                      layer_stats.kv_rows > 0 &&
                      (!comp_rows_expected || layer_stats.comp_rows > 0) &&
                      (opt.skip_descriptor_checks || layer_stats.checksum != 0) &&
                      kv_result.max_abs == 0.0 &&
                      repeat_bad == 0 &&
                      repeat_nan == 0 &&
                      (!dense_compute.enabled || dense_compute.pass) &&
                      (!bf16_compute.enabled || bf16_compute.pass) &&
                      (!compose.enabled || compose.pass) &&
                      (!decode_loop.enabled || decode_loop.pass);

    std::printf("runtime_bytes_per_gpu\thidden\t%llu\tkv\t%llu\tcomp_state\t%llu\t"
                "scratch\t%llu\ttotal\t%llu\n",
                (unsigned long long)runtime_report.gpu[0].hidden_bytes,
                (unsigned long long)runtime_report.gpu[0].kv_bytes,
                (unsigned long long)runtime_report.gpu[0].comp_state_bytes,
                (unsigned long long)runtime_report.gpu[0].scratch_bytes,
                (unsigned long long)runtime_report.gpu[0].total_bytes);
    std::printf("dense_kv_slice\tlayer\t%d\tratio\t%d\tslot\t%u\tposition\t%llu\t"
                "attn_row\t%llu\tindexer_row\t%llu\tattn_row_bytes\t%llu\t"
                "indexer_row_bytes\t%llu\tmax_abs\t%.9f\tdense_kv_ms\t%.6f\n",
                kv_result.layer, kv_result.ratio, kv_result.slot,
                (unsigned long long)kv_result.position,
                (unsigned long long)kv_result.attn_row,
                (unsigned long long)kv_result.indexer_row,
                (unsigned long long)kv_result.attn_row_bytes[0],
                (unsigned long long)kv_result.indexer_row_bytes[0],
                kv_result.max_abs, dense_kv_ms);
    std::printf("tp_ep_full_layer_scaffold\tslots\t%d\tctx\t%llu\ttop_k\t%d\t"
                "layer\t%d\ttotal_rows\t%llu\tdense_rows\t%llu\tcontrol_rows\t%llu\t"
                "expert_rows\t%llu\tkv_rows\t%llu\tcomp_rows\t%llu\t"
                "dense_loaded_bytes\t%llu\tcontrol_loaded_bytes\t%llu\t"
                "ep_loaded_bytes\t%llu\tdescriptor_checksum\t%llu\t"
                "dense_compute_tensor\t%s\tdense_compute_rows_per_gpu\t%d\t"
                "dense_compute_cols\t%d\tdense_compute_slots\t%d\t"
                "dense_compute_loaded_bytes\t%llu\tdense_compute_ms\t%.6f\t"
                "dense_compute_repeat_max_abs\t%.9f\tdense_compute_repeat_bad\t%d\t"
                "dense_compute_repeat_nan\t%d\tdense_compute_oracle_max_abs\t%.9f\t"
                "dense_compute_oracle_bad\t%d\tdense_compute_pass\t%d\t"
                "bf16_compute_tensor\t%s\tbf16_compute_rows_per_gpu\t%d\t"
                "bf16_compute_cols\t%d\tbf16_compute_slots\t%d\t"
                "bf16_compute_loaded_bytes\t%llu\tbf16_compute_ms\t%.6f\t"
                "bf16_compute_repeat_max_abs\t%.9f\tbf16_compute_repeat_bad\t%d\t"
                "bf16_compute_repeat_nan\t%d\tbf16_compute_oracle_max_abs\t%.9f\t"
                "bf16_compute_oracle_bad\t%d\tbf16_compute_pass\t%d\t"
                "compose_next_hidden\t%d\tcompose_ep_contribution_bytes\t%llu\t"
                "compose_ep_return_dtype\t%s\tcompose_ep_return_bytes\t%llu\t"
                "compose_dense_hmma\t%d\tcompose_dense_f16_cublas\t%d\t"
                "compose_attn_dense_ms\t%.6f\t"
                "compose_shared_dense_ms\t%.6f\tcompose_fused_sum\t%d\t"
                "compose_nccl_reduce_scatter\t%d\t"
                "compose_ms\t%.6f\t"
                "compose_checksum\t%llu\tcompose_finite_bad\t%d\t"
                "compose_repeat_max_abs\t%.9f\tcompose_repeat_bad\t%d\t"
                "compose_pass\t%d\t"
                "decode_steps\t%d\tdecode_slot_steps\t%llu\tdecode_total_ms\t%.6f\t"
                "decode_ms_per_step\t%.6f\tdecode_slot_step_tok_s\t%.6f\t"
                "decode_dense_hmma\t%d\tdecode_dense_f16_cublas\t%d\t"
                "decode_dense_f16_cache\t%d\t"
                "decode_overlap_ep_dense\t%d\tdecode_direct_remote_compose\t%d\t"
                "decode_source_copy_schedule\t%d\t"
                "decode_ep_ms_per_step\t%.6f\tdecode_dense_ms_per_step\t%.6f\t"
                "decode_fused_compose_sum\t%d\tdecode_nccl_reduce_scatter\t%d\t"
                "decode_compose_ms_per_step\t%.6f\t"
                "decode_ep_return_dtype\t%s\t"
                "decode_ep_return_bytes\t%llu\tdecode_checksum\t%llu\t"
                "decode_finite_bad\t%d\tdecode_pass\t%d\t"
                "aggregate_routes\t%d\tdispatch_bytes\t%llu\treturn_bytes\t%llu\t"
                "route_imbalance\t%.6f\tdescriptor_ms\t%.6f\tdense_kv_ms\t%.6f\t"
                "worst_gate_ms\t%.6f\tworst_down_ms\t%.6f\tworst_ep_ms\t%.6f\t"
                "scaffold_ms\t%.6f\trepeat_max_abs\t%.9f\trepeat_bad\t%d\t"
                "repeat_nan\t%d\t%s\n",
                opt.slots, (unsigned long long)cfg.ctx, opt.top_k, opt.layer,
                (unsigned long long)layer_stats.total_rows,
                (unsigned long long)layer_stats.dense_rows,
                (unsigned long long)layer_stats.control_rows,
                (unsigned long long)layer_stats.expert_rows,
                (unsigned long long)layer_stats.kv_rows,
                (unsigned long long)layer_stats.comp_rows,
                (unsigned long long)layer_stats.dense_loaded_bytes,
                (unsigned long long)layer_stats.control_loaded_bytes,
                (unsigned long long)layer_stats.ep_loaded_bytes,
                (unsigned long long)layer_stats.checksum,
                dense_compute.enabled ? dense_compute.tensor_id.c_str() : "disabled",
                dense_compute.rows_per_gpu,
                dense_compute.cols,
                dense_compute.slots,
                (unsigned long long)dense_compute.loaded_bytes,
                dense_compute.compute_ms,
                dense_compute.repeat_max_abs,
                dense_compute.repeat_bad,
                dense_compute.repeat_nan,
                dense_compute.oracle_max_abs,
                dense_compute.oracle_bad,
                dense_compute.enabled && dense_compute.pass ? 1 : 0,
                bf16_compute.enabled ? bf16_compute.tensor_id.c_str() : "disabled",
                bf16_compute.rows_per_gpu,
                bf16_compute.cols,
                bf16_compute.slots,
                (unsigned long long)bf16_compute.loaded_bytes,
                bf16_compute.compute_ms,
                bf16_compute.repeat_max_abs,
                bf16_compute.repeat_bad,
                bf16_compute.repeat_nan,
                bf16_compute.oracle_max_abs,
                bf16_compute.oracle_bad,
                bf16_compute.enabled && bf16_compute.pass ? 1 : 0,
                compose.enabled ? 1 : 0,
                (unsigned long long)compose.ep_contribution_bytes,
                compose.ep_return_fp16 ? "fp16" : "fp32",
                (unsigned long long)compose.ep_return_bytes,
                compose.dense_hmma_compose ? 1 : 0,
                compose.dense_f16_cublas_compose ? 1 : 0,
                compose.attn_dense_ms,
                compose.shared_dense_ms,
                compose.fused_compose_sum ? 1 : 0,
                compose.nccl_reduce_scatter_compose ? 1 : 0,
                compose.compose_ms,
                (unsigned long long)compose.checksum,
                compose.finite_bad,
                compose.repeat_max_abs,
                compose.repeat_bad,
                compose.enabled && compose.pass ? 1 : 0,
                decode_loop.steps,
                (unsigned long long)decode_loop.slot_steps,
                decode_loop.total_ms,
                decode_loop.ms_per_step,
                decode_loop.tok_s,
                decode_loop.dense_hmma_compose ? 1 : 0,
                decode_loop.dense_f16_cublas_compose ? 1 : 0,
                decode_loop.dense_f16_cache_compose ? 1 : 0,
                opt.overlap_ep_dense ? 1 : 0,
                opt.direct_remote_compose ? 1 : 0,
                opt.source_copy_schedule ? 1 : 0,
                decode_loop.ep_ms_per_step,
                decode_loop.dense_ms_per_step,
                decode_loop.fused_compose_sum ? 1 : 0,
                decode_loop.nccl_reduce_scatter_compose ? 1 : 0,
                decode_loop.compose_ms_per_step,
                decode_loop.ep_return_fp16 ? "fp16" : "fp32",
                (unsigned long long)decode_loop.ep_return_bytes,
                (unsigned long long)decode_loop.checksum,
                decode_loop.finite_bad,
                decode_loop.enabled && decode_loop.pass ? 1 : 0,
                aggregate_routes,
                (unsigned long long)dispatch_bytes,
                (unsigned long long)return_bytes,
                imbalance, descriptor_ms, dense_kv_ms, worst_gate_ms, worst_down_ms,
                worst_ep_ms, scaffold_ms, repeat_max_abs, repeat_bad, repeat_nan,
                pass ? "PASS" : "FAIL");

    if (summary) {
        summary->layer = opt.layer;
        summary->ratio = ds4_layer_ratio(opt.layer);
        summary->pass = pass;
        summary->total_rows = layer_stats.total_rows;
        summary->dense_rows = layer_stats.dense_rows;
        summary->control_rows = layer_stats.control_rows;
        summary->expert_rows = layer_stats.expert_rows;
        summary->kv_rows = layer_stats.kv_rows;
        summary->comp_rows = layer_stats.comp_rows;
        summary->decode_ms_per_step = decode_loop.ms_per_step;
        summary->decode_slot_step_tok_s = decode_loop.tok_s;
        summary->decode_ep_ms_per_step = decode_loop.ep_ms_per_step;
        summary->decode_dense_ms_per_step = decode_loop.dense_ms_per_step;
        summary->decode_compose_ms_per_step = decode_loop.compose_ms_per_step;
        summary->decode_compose_reduce_ms_per_step =
            decode_loop.compose_reduce_ms_per_step;
        summary->decode_compose_copy_ms_per_step =
            decode_loop.compose_copy_ms_per_step;
        summary->decode_compose_final_ms_per_step =
            decode_loop.compose_final_ms_per_step;
        summary->decode_hc_current_input_ms_per_step =
            decode_loop.hc_current_input_ms_per_step;
        summary->decode_hc_current_seed_ms_per_step =
            decode_loop.hc_current_seed_ms_per_step;
        summary->decode_hc_current_attn_mix_ms_per_step =
            decode_loop.hc_current_attn_mix_ms_per_step;
        summary->decode_hc_current_split_ms_per_step =
            decode_loop.hc_current_split_ms_per_step;
        summary->decode_hc_current_gather_ms_per_step =
            decode_loop.hc_current_gather_ms_per_step;
        summary->decode_hc_current_ffn_router_ms_per_step =
            decode_loop.hc_current_ffn_router_ms_per_step;
        summary->decode_hc_current_ffn_norm_ms_per_step =
            decode_loop.hc_current_ffn_norm_ms_per_step;
        summary->decode_hc_current_router_select_ms_per_step =
            decode_loop.hc_current_router_select_ms_per_step;
        summary->decode_hc_current_router_d2h_ms_per_step =
            decode_loop.hc_current_router_d2h_ms_per_step;
        summary->decode_hc_current_route_upload_ms_per_step =
            decode_loop.hc_current_route_upload_ms_per_step;
        summary->decode_hc_current_fill_pack_ms_per_step =
            decode_loop.hc_current_fill_pack_ms_per_step;
        summary->decode_pre_ep_hc_current_ms_per_step =
            decode_loop.pre_ep_hc_current_ms_per_step;
        summary->decode_pre_ep_attention_projection_ms_per_step =
            decode_loop.pre_ep_attention_projection_ms_per_step;
        summary->decode_pre_ep_compressed_kv_ms_per_step =
            decode_loop.pre_ep_compressed_kv_ms_per_step;
        summary->decode_pre_ep_attention_state_ms_per_step =
            decode_loop.pre_ep_attention_state_ms_per_step;
        summary->decode_pre_ep_typed_history_ms_per_step =
            decode_loop.pre_ep_typed_history_ms_per_step;
        summary->decode_pre_ep_raw_read_ms_per_step =
            decode_loop.pre_ep_raw_read_ms_per_step;
        summary->decode_pre_ep_attention_output_ms_per_step =
            decode_loop.pre_ep_attention_output_ms_per_step;
        summary->decode_pre_ep_post_attention_ffn_input_ms_per_step =
            decode_loop.pre_ep_post_attention_ffn_input_ms_per_step;
        summary->decode_final_hc_ms_per_step = decode_loop.final_hc_ms_per_step;
        summary->decode_checksum = decode_loop.checksum;
    }

    if (!shared_rank_buffers) {
        close_compose_nccl(ranks);
    }
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!layer_expert_cache) free_packed(r.gated);
        r.gated = PackedExperts{};
        if (!layer_expert_cache) free_packed(r.down);
        r.down = PackedExperts{};
        if (!shared_rank_buffers) {
            CHECK_CUDA(cudaFree(r.d_offsets));
            CHECK_CUDA(cudaFree(r.d_route_slots));
            CHECK_CUDA(cudaFree(r.d_route_weights));
            CHECK_CUDA(cudaFree(r.d_route_inv_scale));
            CHECK_CUDA(cudaFree(r.d_a));
            CHECK_CUDA(cudaFree(r.d_gate_up));
            CHECK_CUDA(cudaFree(r.d_gated));
            CHECK_CUDA(cudaFree(r.d_down));
            if (r.d_ep_contrib_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_all));
            if (r.d_ep_contrib_half_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_half_all));
            for (int src = 0; src < kGpus; ++src) {
                if (r.d_ep_remote[src]) CHECK_CUDA(cudaFree(r.d_ep_remote[src]));
                if (r.d_ep_remote_half[src]) CHECK_CUDA(cudaFree(r.d_ep_remote_half[src]));
            }
            if (r.d_ep_sum) CHECK_CUDA(cudaFree(r.d_ep_sum));
            if (r.d_next_hidden) CHECK_CUDA(cudaFree(r.d_next_hidden));
            if (r.d_final_hc_shard) CHECK_CUDA(cudaFree(r.d_final_hc_shard));
            if (r.d_hc_scratch_shard) CHECK_CUDA(cudaFree(r.d_hc_scratch_shard));
            if (r.d_hc_split) CHECK_CUDA(cudaFree(r.d_hc_split));
            for (int layer = 0; layer < 43; ++layer) {
                if (r.d_attn_raw_swa_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_attn_raw_swa_layers[layer]));
                }
            }
            if (r.d_attn_kv_full) CHECK_CUDA(cudaFree(r.d_attn_kv_full));
            if (r.d_attn_heads) CHECK_CUDA(cudaFree(r.d_attn_heads));
            if (r.d_attn_output_a_full) CHECK_CUDA(cudaFree(r.d_attn_output_a_full));
            if (r.d_post_attn_shard) CHECK_CUDA(cudaFree(r.d_post_attn_shard));
            if (r.d_attn_sinks) CHECK_CUDA(cudaFree(r.d_attn_sinks));
            if (r.d_indexer_topk) CHECK_CUDA(cudaFree(r.d_indexer_topk));
            if (r.d_indexer_scores) CHECK_CUDA(cudaFree(r.d_indexer_scores));
            for (int layer = 0; layer < 43; ++layer) {
                if (r.d_index_comp_rows_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_index_comp_rows_layers[layer]));
                }
                if (r.d_index_comp_state_score_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_index_comp_state_score_layers[layer]));
                }
                if (r.d_index_comp_state_kv_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_index_comp_state_kv_layers[layer]));
                }
            }
            if (r.d_index_comp_score_cur) CHECK_CUDA(cudaFree(r.d_index_comp_score_cur));
            if (r.d_index_comp_kv_cur) CHECK_CUDA(cudaFree(r.d_index_comp_kv_cur));
            for (int layer = 0; layer < 43; ++layer) {
                if (r.d_attn_comp_rows_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_attn_comp_rows_layers[layer]));
                }
                if (r.d_attn_comp_state_score_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_attn_comp_state_score_layers[layer]));
                }
                if (r.d_attn_comp_state_kv_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_attn_comp_state_kv_layers[layer]));
                }
            }
            if (r.d_attn_comp_score_cur) CHECK_CUDA(cudaFree(r.d_attn_comp_score_cur));
            if (r.d_attn_comp_kv_cur) CHECK_CUDA(cudaFree(r.d_attn_comp_kv_cur));
            if (r.dense_wait) CHECK_CUDA(cudaEventDestroy(r.dense_wait));
            if (r.start) CHECK_CUDA(cudaEventDestroy(r.start));
            if (r.mid) CHECK_CUDA(cudaEventDestroy(r.mid));
            if (r.stop) CHECK_CUDA(cudaEventDestroy(r.stop));
            if (r.d_route_totals) CHECK_CUDA(cudaFree(r.d_route_totals));
            if (r.d_route_offsets_all) CHECK_CUDA(cudaFree(r.d_route_offsets_all));
            if (r.d_router_weights_plan) CHECK_CUDA(cudaFree(r.d_router_weights_plan));
            if (r.d_router_selected_plan) CHECK_CUDA(cudaFree(r.d_router_selected_plan));
            const bool has_route_compact_plan = r.d_route_compact_plan != nullptr;
            if (r.d_route_compact_plan) CHECK_CUDA(cudaFree(r.d_route_compact_plan));
            for (int src = 0; src < kGpus; ++src) {
                if (r.d_route_index_by_slot[src]) CHECK_CUDA(cudaFree(r.d_route_index_by_slot[src]));
                if (!has_route_compact_plan && r.d_route_indices_by_slot[src]) {
                    CHECK_CUDA(cudaFree(r.d_route_indices_by_slot[src]));
                }
                if (!has_route_compact_plan && r.d_route_count_by_slot[src]) {
                    CHECK_CUDA(cudaFree(r.d_route_count_by_slot[src]));
                }
            }
            for (int q = 0; q < kGpus; ++q) {
                if (r.copy_done[q]) CHECK_CUDA(cudaEventDestroy(r.copy_done[q]));
                if (r.copy_streams[q]) CHECK_CUDA(cudaStreamDestroy(r.copy_streams[q]));
            }
            if (r.dense_done) CHECK_CUDA(cudaEventDestroy(r.dense_done));
            if (r.stream_done) CHECK_CUDA(cudaEventDestroy(r.stream_done));
            CHECK_CUDA(cudaStreamDestroy(r.copy_stream));
            CHECK_CUDA(cudaStreamDestroy(r.dense_stream));
            CHECK_CUDA(cudaStreamDestroy(r.stream));
        }
    }
    if (!shared_api) {
        api->shutdown();
        dlclose(lib);
    }
    close_local_runtime();
    if (!shared_dense_f16_cache) free_dense_f16_cache(local_dense_f16_cache, opt);
    return pass ? 0 : 1;
}

int run_token_major_serving_loop(const Options &opt,
                                 const DenseF16Cache *shared_dense_f16_cache,
                                 const SharedApi *shared_api,
                                 SharedRankBuffers *shared_rank_buffers,
                                 SharedTpRuntime *shared_tp_runtime,
                                 const SharedExpertBindings *shared_expert_bindings,
                                 const SharedDenseOps *shared_dense_ops,
                                 SharedOutputHead *shared_output_head,
                                 SharedHcControls *shared_hc_controls,
                                 SharedTokenEmbedding *shared_token_embedding,
                                 const std::vector<uint32_t> *decode_input_tokens,
                                 const std::vector<unsigned char> *decode_active_slots,
                                 std::vector<ContractRow> resident_rows[43],
                                 LayerStats resident_stats[43],
                                 bool resident_serving_loop,
                                 ServingBenchResult *serving_result) {
    int pass_invocations = 0;
    double sum_decode_ms = 0.0;
    double sum_ep_ms = 0.0;
    double sum_dense_ms = 0.0;
    double sum_compose_ms = 0.0;
    double sum_compose_reduce_ms = 0.0;
    double sum_compose_copy_ms = 0.0;
    double sum_compose_final_ms = 0.0;
    double sum_hc_current_input_ms = 0.0;
    double sum_hc_current_seed_ms = 0.0;
    double sum_hc_current_attn_mix_ms = 0.0;
    double sum_hc_current_split_ms = 0.0;
    double sum_hc_current_gather_ms = 0.0;
    double sum_hc_current_ffn_router_ms = 0.0;
    double sum_hc_current_ffn_norm_ms = 0.0;
    double sum_hc_current_router_select_ms = 0.0;
    double sum_hc_current_router_d2h_ms = 0.0;
    double sum_hc_current_route_upload_ms = 0.0;
    double sum_hc_current_fill_pack_ms = 0.0;
    double sum_pre_ep_hc_current_ms = 0.0;
    double sum_pre_ep_attention_projection_ms = 0.0;
    double sum_pre_ep_compressed_kv_ms = 0.0;
    double sum_pre_ep_attention_state_ms = 0.0;
    double sum_pre_ep_typed_history_ms = 0.0;
    double sum_pre_ep_raw_read_ms = 0.0;
    double sum_pre_ep_attention_output_ms = 0.0;
    double sum_pre_ep_post_attention_ffn_input_ms = 0.0;
    double sum_final_hc_ms = 0.0;
    int sum_cudagraph_sync_all_calls = 0;
    int sum_cudagraph_event_barrier_calls = 0;
    int sum_cudagraph_rank_stream_syncs = 0;
    int sum_cudagraph_dense_stream_syncs = 0;
    int sum_cudagraph_copy_stream_syncs = 0;
    int sum_cudagraph_capture_attempted = 0;
    int sum_cudagraph_capture_succeeded = 0;
    int sum_cudagraph_capture_error = 0;
    size_t sum_cudagraph_capture_nodes = 0;
    double first_token_decode_ms = 0.0;
    double continuation_decode_ms = 0.0;
    double first_token_wall_ms = 0.0;
    double continuation_wall_ms = 0.0;
    uint64_t checksum = 0;
    if (opt.final_hc_carry_gate && !opt.tp_hc_persist_state_gate &&
        shared_rank_buffers && shared_rank_buffers->initialized) {
        for (int rank = 0; rank < kGpus; ++rank) {
            shared_rank_buffers->ranks[rank].hc_initialized = false;
        }
    }
    TpEpProfilerWindowGuard profiler_guard(opt);
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < opt.decode_steps; ++step) {
        const auto step_start = std::chrono::steady_clock::now();
        double step_decode_ms = 0.0;
        if (step == 0 && shared_token_embedding && decode_input_tokens &&
            !decode_input_tokens->empty()) {
            if (!shared_rank_buffers || !shared_rank_buffers->initialized ||
                ensure_compose_buffers(opt, shared_rank_buffers->ranks) != 0) {
                std::fprintf(stderr, "tp_ep_token_embedding_seed_failed\treason\tmissing_rank_buffers\n");
                return 15;
            }
            const int seed_rc = seed_rank_hc_from_input_tokens(
                opt, shared_token_embedding, shared_rank_buffers->ranks,
                *decode_input_tokens);
            if (seed_rc != 0) {
                std::fprintf(stderr, "tp_ep_token_embedding_seed_failed\trc\t%d\n",
                             seed_rc);
                return 15;
            }
        }
        if (shared_hc_controls && shared_hc_controls->initialized &&
            shared_hc_controls->d_router_tokens &&
            decode_input_tokens && decode_input_tokens->size() >= (size_t)opt.slots) {
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
            CHECK_CUDA(cudaMemcpy(shared_hc_controls->d_router_tokens,
                                  decode_input_tokens->data(),
                                  (size_t)opt.slots * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
        }
        if (shared_hc_controls && shared_hc_controls->initialized &&
            shared_hc_controls->d_router_active &&
            decode_active_slots && decode_active_slots->size() >= (size_t)opt.slots) {
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
            CHECK_CUDA(cudaMemcpy(shared_hc_controls->d_router_active,
                                  decode_active_slots->data(),
                                  (size_t)opt.slots * sizeof(unsigned char),
                                  cudaMemcpyHostToDevice));
        }
        for (int layer = 0; layer < 43; ++layer) {
            Options layer_opt = opt;
            layer_opt.layer = layer;
            layer_opt.position = opt.position + (uint64_t)step;
            layer_opt.decode_steps = 1;
            layer_opt.true_ds4_attention_raw_valid_rows =
                std::max(1u, std::min((uint32_t)(step + 1), (uint32_t)kRawSwaRows));
            layer_opt.warmup = 0;
            LayerRunSummary s;
            SharedTpRuntime *tp_runtime_arg =
                shared_tp_runtime && shared_tp_runtime->initialized ? shared_tp_runtime : nullptr;
            const SharedExpertBindings *expert_arg =
                shared_expert_bindings && shared_expert_bindings->initialized
                    ? shared_expert_bindings
                    : nullptr;
            const SharedDenseOps *dense_ops_arg =
                shared_dense_ops && shared_dense_ops->initialized ? shared_dense_ops : nullptr;
            int rc = 0;
            if (resident_serving_loop) {
                if (!shared_api || !shared_api->initialized ||
                    !shared_rank_buffers || !shared_rank_buffers->initialized ||
                    !shared_tp_runtime || !shared_tp_runtime->initialized ||
                    !shared_expert_bindings || !shared_expert_bindings->initialized ||
                    !shared_dense_f16_cache || !shared_dense_f16_cache->enabled) {
                    std::fprintf(stderr, "resident serving loop missing shared state\n");
                    rc = 2;
                    s.pass = false;
                } else {
                    const LayerDenseOps *layer_dense_ops =
                        dense_ops_arg ? &dense_ops_arg->layers[layer] : nullptr;
                    rc = run_resident_layer_decode(layer_opt,
                                                   resident_rows[layer],
                                                   resident_stats[layer],
                                                   shared_rank_buffers->ranks,
                                                   shared_api->api,
                                                   shared_tp_runtime->rt,
                                                   &shared_expert_bindings->layers[layer],
                                                   shared_dense_f16_cache,
                                                   layer_dense_ops,
                                                   shared_hc_controls,
                                                   &s);
                }
            } else {
                rc = run_layer(layer_opt, &s, shared_dense_f16_cache, shared_api,
                               shared_rank_buffers, tp_runtime_arg, expert_arg,
                               dense_ops_arg, shared_hc_controls);
            }
            std::printf("tp_ep_token_major_item\tstep\t%d\tlayer\t%d\tratio\t%d\t"
                        "position\t%llu\t"
                        "decode_ms_per_step\t%.6f\tdecode_slot_step_tok_s\t%.6f\t"
                        "decode_ep_ms_per_step\t%.6f\tdecode_dense_ms_per_step\t%.6f\t"
                        "decode_compose_ms_per_step\t%.6f\t"
                        "decode_compose_reduce_ms_per_step\t%.6f\t"
                        "decode_compose_copy_ms_per_step\t%.6f\t"
                        "decode_compose_final_ms_per_step\t%.6f\t"
                        "decode_hc_current_input_ms_per_step\t%.6f\t"
                        "decode_hc_current_seed_ms_per_step\t%.6f\t"
                        "decode_hc_current_attn_mix_ms_per_step\t%.6f\t"
                        "decode_hc_current_split_ms_per_step\t%.6f\t"
                        "decode_hc_current_gather_ms_per_step\t%.6f\t"
                        "decode_hc_current_ffn_router_ms_per_step\t%.6f\t"
                        "decode_hc_current_ffn_norm_ms_per_step\t%.6f\t"
                        "decode_hc_current_router_select_ms_per_step\t%.6f\t"
                        "decode_hc_current_router_d2h_ms_per_step\t%.6f\t"
                        "decode_hc_current_route_upload_ms_per_step\t%.6f\t"
                        "decode_hc_current_fill_pack_ms_per_step\t%.6f\t"
                        "decode_pre_ep_hc_current_ms_per_step\t%.6f\t"
                        "decode_pre_ep_attention_projection_ms_per_step\t%.6f\t"
                        "decode_pre_ep_compressed_kv_ms_per_step\t%.6f\t"
                        "decode_pre_ep_attention_state_ms_per_step\t%.6f\t"
                        "decode_pre_ep_typed_history_ms_per_step\t%.6f\t"
                        "decode_pre_ep_raw_read_ms_per_step\t%.6f\t"
                        "decode_pre_ep_attention_output_ms_per_step\t%.6f\t"
                        "decode_pre_ep_post_attention_ffn_input_ms_per_step\t%.6f\t"
                        "decode_final_hc_ms_per_step\t%.6f\t"
                        "decode_checksum\t%llu\tdecode_finite_bad\t%d\trc\t%d\t%s\n",
                        step, s.layer, s.ratio,
                        (unsigned long long)layer_opt.position,
                        s.decode_ms_per_step,
                        s.decode_slot_step_tok_s,
                        s.decode_ep_ms_per_step,
                        s.decode_dense_ms_per_step,
                        s.decode_compose_ms_per_step,
                        s.decode_compose_reduce_ms_per_step,
                        s.decode_compose_copy_ms_per_step,
                        s.decode_compose_final_ms_per_step,
                        s.decode_hc_current_input_ms_per_step,
                        s.decode_hc_current_seed_ms_per_step,
                        s.decode_hc_current_attn_mix_ms_per_step,
                        s.decode_hc_current_split_ms_per_step,
                        s.decode_hc_current_gather_ms_per_step,
                        s.decode_hc_current_ffn_router_ms_per_step,
                        s.decode_hc_current_ffn_norm_ms_per_step,
                        s.decode_hc_current_router_select_ms_per_step,
                        s.decode_hc_current_router_d2h_ms_per_step,
                        s.decode_hc_current_route_upload_ms_per_step,
                        s.decode_hc_current_fill_pack_ms_per_step,
                        s.decode_pre_ep_hc_current_ms_per_step,
                        s.decode_pre_ep_attention_projection_ms_per_step,
                        s.decode_pre_ep_compressed_kv_ms_per_step,
                        s.decode_pre_ep_attention_state_ms_per_step,
                        s.decode_pre_ep_typed_history_ms_per_step,
                        s.decode_pre_ep_raw_read_ms_per_step,
                        s.decode_pre_ep_attention_output_ms_per_step,
                        s.decode_pre_ep_post_attention_ffn_input_ms_per_step,
                        s.decode_final_hc_ms_per_step,
                        (unsigned long long)s.decode_checksum,
                        s.decode_finite_bad,
                        rc,
                        (rc == 0 && s.pass) ? "PASS" : "FAIL");
            if (rc == 0 && s.pass) {
                pass_invocations++;
                sum_decode_ms += s.decode_ms_per_step;
                step_decode_ms += s.decode_ms_per_step;
                sum_ep_ms += s.decode_ep_ms_per_step;
                sum_dense_ms += s.decode_dense_ms_per_step;
                sum_compose_ms += s.decode_compose_ms_per_step;
                sum_compose_reduce_ms += s.decode_compose_reduce_ms_per_step;
                sum_compose_copy_ms += s.decode_compose_copy_ms_per_step;
                sum_compose_final_ms += s.decode_compose_final_ms_per_step;
                sum_hc_current_input_ms += s.decode_hc_current_input_ms_per_step;
                sum_hc_current_seed_ms += s.decode_hc_current_seed_ms_per_step;
                sum_hc_current_attn_mix_ms += s.decode_hc_current_attn_mix_ms_per_step;
                sum_hc_current_split_ms += s.decode_hc_current_split_ms_per_step;
                sum_hc_current_gather_ms += s.decode_hc_current_gather_ms_per_step;
                sum_hc_current_ffn_router_ms += s.decode_hc_current_ffn_router_ms_per_step;
                sum_hc_current_ffn_norm_ms += s.decode_hc_current_ffn_norm_ms_per_step;
                sum_hc_current_router_select_ms +=
                    s.decode_hc_current_router_select_ms_per_step;
                sum_hc_current_router_d2h_ms +=
                    s.decode_hc_current_router_d2h_ms_per_step;
                sum_hc_current_route_upload_ms +=
                    s.decode_hc_current_route_upload_ms_per_step;
                sum_hc_current_fill_pack_ms += s.decode_hc_current_fill_pack_ms_per_step;
                sum_pre_ep_hc_current_ms += s.decode_pre_ep_hc_current_ms_per_step;
                sum_pre_ep_attention_projection_ms +=
                    s.decode_pre_ep_attention_projection_ms_per_step;
                sum_pre_ep_compressed_kv_ms += s.decode_pre_ep_compressed_kv_ms_per_step;
                sum_pre_ep_attention_state_ms +=
                    s.decode_pre_ep_attention_state_ms_per_step;
                sum_pre_ep_typed_history_ms += s.decode_pre_ep_typed_history_ms_per_step;
                sum_pre_ep_raw_read_ms += s.decode_pre_ep_raw_read_ms_per_step;
                sum_pre_ep_attention_output_ms +=
                    s.decode_pre_ep_attention_output_ms_per_step;
                sum_pre_ep_post_attention_ffn_input_ms +=
                    s.decode_pre_ep_post_attention_ffn_input_ms_per_step;
                sum_final_hc_ms += s.decode_final_hc_ms_per_step;
                sum_cudagraph_sync_all_calls +=
                    s.decode_cudagraph_sync_all_calls;
                sum_cudagraph_event_barrier_calls +=
                    s.decode_cudagraph_event_barrier_calls;
                sum_cudagraph_rank_stream_syncs +=
                    s.decode_cudagraph_rank_stream_syncs;
                sum_cudagraph_dense_stream_syncs +=
                    s.decode_cudagraph_dense_stream_syncs;
                sum_cudagraph_copy_stream_syncs +=
                    s.decode_cudagraph_copy_stream_syncs;
                sum_cudagraph_capture_attempted +=
                    s.decode_cudagraph_capture_attempted;
                sum_cudagraph_capture_succeeded +=
                    s.decode_cudagraph_capture_succeeded;
                if (sum_cudagraph_capture_error == 0 &&
                    s.decode_cudagraph_capture_error != 0) {
                    sum_cudagraph_capture_error =
                        s.decode_cudagraph_capture_error;
                }
                sum_cudagraph_capture_nodes +=
                    s.decode_cudagraph_capture_nodes;
                checksum ^= s.decode_checksum +
                            (uint64_t)(step + 1) * 1000003ull +
                            (uint64_t)(layer + 1) * 104729ull;
            } else {
                const auto stop = std::chrono::steady_clock::now();
                const double wall_ms =
                    std::chrono::duration<double, std::milli>(stop - start).count();
                std::printf("tp_ep_token_major_scaffold\tsteps\t%d\tlayers\t43\t"
                            "pass_invocations\t%d\tfailed_step\t%d\tfailed_layer\t%d\t"
                            "slots\t%d\tctx\t262144\twall_ms\t%.6f\tFAIL\n",
                            opt.decode_steps, pass_invocations, step, layer,
                            opt.slots, wall_ms);
                std::fflush(stdout);
                return rc == 0 ? 1 : rc;
            }
        }
        const auto step_stop = std::chrono::steady_clock::now();
        const double step_wall_ms =
            std::chrono::duration<double, std::milli>(step_stop - step_start).count();
        if (step == 0) {
            first_token_decode_ms += step_decode_ms;
            first_token_wall_ms += step_wall_ms;
        } else {
            continuation_decode_ms += step_decode_ms;
            continuation_wall_ms += step_wall_ms;
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    const double ms_per_token = opt.decode_steps > 0
        ? sum_decode_ms / (double)opt.decode_steps
        : 0.0;
    const double slot_step_tok_s = sum_decode_ms > 0.0
        ? (double)opt.slots * (double)opt.decode_steps * 1000.0 / sum_decode_ms
        : 0.0;
    std::printf("tp_ep_token_major_scaffold\tsteps\t%d\tlayers\t43\t"
                "pass_invocations\t%d\tslots\t%d\tctx\t262144\t"
                "shared_api\t%d\tshared_rank_buffers\t%d\tshared_tp_runtime\t%d\t"
                "shared_expert_bindings\t%d\toverlap_ep_dense\t%d\t"
                "shared_dense_ops\t%d\t"
                "skip_decode_checksum\t%d\t"
                "direct_remote_compose\t%d\tsource_copy_schedule\t%d\t"
                "skip_self_compose_copy\t%d\t"
                "multi_copy_streams\t%d\t"
                "batched_paged_attn_gate\t%d\t"
                "compact_moe_decode_gate\t%d\t"
                "router_cublas_gate\t%d\t"
                "router_hash_fast_gate\t%d\t"
                "gpu_route_plan_gate\t%d\t"
                "route_plan_async_upload_gate\t%d\t"
                "fused_gated_silu_gate\t%d\t"
                "routed_ffn_norm_input_gate\t%d\t"
                "routed_gate_standalone_swiglu\t%d\t"
                "sum_decode_ms\t%.6f\tms_per_token\t%.6f\t"
                "projected_slot_step_tok_s\t%.6f\t"
                "sum_ep_ms\t%.6f\tsum_dense_ms\t%.6f\tsum_compose_ms\t%.6f\t"
                "sum_compose_reduce_ms\t%.6f\tsum_compose_copy_ms\t%.6f\t"
                "sum_compose_final_ms\t%.6f\t"
                "tp_hc_current_input_gate\t%d\t"
                "tp_hc_current_input_peer_gather\t%d\t"
                "tp_hc_current_input_nccl_allgather\t%d\t"
                "tp_hc_current_input_stream_sync\t%d\t"
                "sum_hc_current_input_ms\t%.6f\t"
                "sum_hc_current_seed_ms\t%.6f\t"
                "sum_hc_current_attn_mix_ms\t%.6f\t"
                "sum_hc_current_split_ms\t%.6f\t"
                "sum_hc_current_gather_ms\t%.6f\t"
                "sum_hc_current_ffn_router_ms\t%.6f\t"
                "sum_hc_current_ffn_norm_ms\t%.6f\t"
                "sum_hc_current_router_select_ms\t%.6f\t"
                "sum_hc_current_router_d2h_ms\t%.6f\t"
                "sum_hc_current_route_upload_ms\t%.6f\t"
                "sum_hc_current_fill_pack_ms\t%.6f\t"
                "sum_pre_ep_hc_current_ms\t%.6f\t"
                "sum_pre_ep_attention_projection_ms\t%.6f\t"
                "sum_pre_ep_compressed_kv_ms\t%.6f\t"
                "sum_pre_ep_attention_state_ms\t%.6f\t"
                "sum_pre_ep_typed_history_ms\t%.6f\t"
                "sum_pre_ep_raw_read_ms\t%.6f\t"
                "sum_pre_ep_attention_output_ms\t%.6f\t"
                "sum_pre_ep_post_attention_ffn_input_ms\t%.6f\t"
                "final_hc_carry_gate\t%d\tsum_final_hc_ms\t%.6f\t"
                "wall_ms\t%.6f\tchecksum\t%llu\tPASS\n",
                opt.decode_steps, pass_invocations, opt.slots,
                shared_api && shared_api->initialized ? 1 : 0,
                shared_rank_buffers && shared_rank_buffers->initialized ? 1 : 0,
                shared_tp_runtime && shared_tp_runtime->initialized ? 1 : 0,
                shared_expert_bindings && shared_expert_bindings->initialized ? 1 : 0,
                opt.overlap_ep_dense ? 1 : 0,
                shared_dense_ops && shared_dense_ops->initialized ? 1 : 0,
                opt.skip_decode_checksum ? 1 : 0,
                opt.direct_remote_compose ? 1 : 0,
                opt.source_copy_schedule ? 1 : 0,
                opt.skip_self_compose_copy ? 1 : 0,
                opt.multi_copy_streams ? 1 : 0,
                opt.batched_paged_attn_gate ? 1 : 0,
                opt.compact_moe_decode_gate ? 1 : 0,
                opt.router_cublas_gate ? 1 : 0,
                opt.router_hash_fast_gate ? 1 : 0,
                opt.gpu_route_plan_gate ? 1 : 0,
                opt.route_plan_async_upload_gate ? 1 : 0,
                opt.fused_gated_silu_gate ? 1 : 0,
                opt.routed_ffn_norm_input_gate ? 1 : 0,
                (opt.routed_ffn_norm_input_gate &&
                 !(opt.fused_gated_silu_gate && !opt.reference_hc_reduce_gate)) ? 1 : 0,
                sum_decode_ms, ms_per_token, slot_step_tok_s,
                sum_ep_ms, sum_dense_ms, sum_compose_ms,
                sum_compose_reduce_ms, sum_compose_copy_ms,
                sum_compose_final_ms,
                opt.tp_hc_current_input_gate ? 1 : 0,
                opt.tp_hc_current_input_peer_gather_gate ? 1 : 0,
                opt.tp_hc_current_input_nccl_allgather_gate ? 1 : 0,
                opt.tp_hc_current_input_stream_sync_gate ? 1 : 0,
                sum_hc_current_input_ms,
                sum_hc_current_seed_ms,
                sum_hc_current_attn_mix_ms,
                sum_hc_current_split_ms,
                sum_hc_current_gather_ms,
                sum_hc_current_ffn_router_ms,
                sum_hc_current_ffn_norm_ms,
                sum_hc_current_router_select_ms,
                sum_hc_current_router_d2h_ms,
                sum_hc_current_route_upload_ms,
                sum_hc_current_fill_pack_ms,
                sum_pre_ep_hc_current_ms,
                sum_pre_ep_attention_projection_ms,
                sum_pre_ep_compressed_kv_ms,
                sum_pre_ep_attention_state_ms,
                sum_pre_ep_typed_history_ms,
                sum_pre_ep_raw_read_ms,
                sum_pre_ep_attention_output_ms,
                sum_pre_ep_post_attention_ffn_input_ms,
                opt.final_hc_carry_gate ? 1 : 0, sum_final_hc_ms,
                wall_ms, (unsigned long long)checksum);
    if (opt.serving_bench || serving_result) {
        const uint64_t prompt_tokens = (uint64_t)opt.slots;
        const uint64_t generated_tokens = (uint64_t)opt.slots *
                                          (uint64_t)opt.decode_steps;
        const uint64_t continuation_tokens = opt.decode_steps > 1
            ? (uint64_t)opt.slots * (uint64_t)(opt.decode_steps - 1)
            : 0ull;
        const double generated_tok_s_decode = sum_decode_ms > 0.0
            ? (double)generated_tokens * 1000.0 / sum_decode_ms
            : 0.0;
        const double generated_tok_s_wall = wall_ms > 0.0
            ? (double)generated_tokens * 1000.0 / wall_ms
            : 0.0;
        const double continuation_tok_s_decode = continuation_decode_ms > 0.0
            ? (double)continuation_tokens * 1000.0 / continuation_decode_ms
            : 0.0;
        const double continuation_tok_s_wall = continuation_wall_ms > 0.0
            ? (double)continuation_tokens * 1000.0 / continuation_wall_ms
            : 0.0;
        if (serving_result) {
            serving_result->prompt_tokens = prompt_tokens;
            serving_result->generated_tokens = generated_tokens;
            serving_result->continuation_tokens = continuation_tokens;
            serving_result->first_token_decode_ms = first_token_decode_ms;
            serving_result->continuation_decode_ms = continuation_decode_ms;
            serving_result->first_token_wall_ms = first_token_wall_ms;
            serving_result->continuation_wall_ms = continuation_wall_ms;
            serving_result->total_decode_ms = sum_decode_ms;
            serving_result->total_wall_ms = wall_ms;
            serving_result->total_ep_ms = sum_ep_ms;
            serving_result->total_dense_ms = sum_dense_ms;
            serving_result->total_compose_ms = sum_compose_ms;
            serving_result->total_compose_reduce_ms = sum_compose_reduce_ms;
            serving_result->total_compose_copy_ms = sum_compose_copy_ms;
            serving_result->total_compose_final_ms = sum_compose_final_ms;
            serving_result->total_hc_current_input_ms = sum_hc_current_input_ms;
            serving_result->token_input_seed =
                shared_token_embedding && decode_input_tokens &&
                !decode_input_tokens->empty();
            serving_result->first_input_token =
                decode_input_tokens && !decode_input_tokens->empty()
                    ? (*decode_input_tokens)[0]
                    : UINT32_MAX;
            serving_result->aggregate_generated_tok_s_decode = generated_tok_s_decode;
            serving_result->aggregate_generated_tok_s_wall = generated_tok_s_wall;
            serving_result->aggregate_continuation_tok_s_decode = continuation_tok_s_decode;
            serving_result->aggregate_continuation_tok_s_wall = continuation_tok_s_wall;
            serving_result->checksum = checksum;
        }
        SharedOutputHead lazy_output_head;
        SharedOutputHead *output_head_for_step = shared_output_head;
        const bool use_lazy_output_head =
            opt.diagnostic_output_head && opt.diagnostic_output_head_lazy_gate &&
            (opt.serving_bench || serving_result) &&
            (!output_head_for_step || !output_head_for_step->initialized) &&
            shared_rank_buffers && shared_rank_buffers->initialized;
        if (use_lazy_output_head) {
            std::vector<ContractRow> all_rows;
            LayerStats all_stats;
            if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
                all_stats.bad_rows != 0 ||
                open_shared_output_head(opt, all_rows, &lazy_output_head) != 0) {
                std::fprintf(stderr, "tp_ep lazy diagnostic output-head open failed\n");
                close_shared_output_head(opt, &lazy_output_head);
                return 12;
            }
            std::printf("tp_ep_diagnostic_output_head_lazy_shared\tslots\t%d\t"
                        "vocab\t%d\trows_per_gpu\t%d\toutput_weight_bytes\t%llu\t"
                        "logits_bytes\t%llu\tproxy_hc\t%d\tPASS\n",
                        opt.slots,
                        lazy_output_head.vocab,
                        lazy_output_head.rows_per_gpu,
                        (unsigned long long)lazy_output_head.output_weight_bytes,
                        (unsigned long long)lazy_output_head.logits_bytes,
                        opt.tp_hc_final_expand_gate ? 0 : 1);
            if (report_vram_checkpoint(opt, "after_lazy_output_head") != 0) {
                close_shared_output_head(opt, &lazy_output_head);
                return 14;
            }
            if (nccl_gate_active(opt) && opt.nccl_min_free_mib != 0) {
                (void)report_vram_checkpoint_min_free(
                    opt, "nccl_after_lazy_output_head", opt.nccl_min_free_mib);
            }
            output_head_for_step = &lazy_output_head;
        }
        if (output_head_for_step && output_head_for_step->initialized &&
            shared_rank_buffers && shared_rank_buffers->initialized) {
            OutputHeadRunResult head_result;
            const int head_rc = run_shared_output_head_from_rank_hc(
                opt, output_head_for_step, shared_rank_buffers->ranks, &head_result);
            std::printf("tp_ep_diagnostic_output_head\tsteps\t%d\tslots\t%d\t"
                        "proxy_hc\t%d\ttotal_ms\t%.6f\tgather_ms\t%.6f\t"
                        "prep_ms\t%.6f\tbroadcast_ms\t%.6f\tprojection_ms\t%.6f\t"
                        "projection_kernel_worst_ms\t%.6f\ttop1_ms\t%.6f\t"
                        "async_output_gate\t%d\tdevice_sync_count\t%d\t"
                        "stream_sync_count\t%d\tevent_sync_count\t%d\t"
                        "first_token\t%u\tfirst_logit\t%.9f\tfinite_bad\t%d\t"
                        "checksum\t%llu\t%s\n",
                        opt.decode_steps, opt.slots,
                        opt.tp_hc_final_expand_gate ? 0 : 1,
                        head_result.total_ms,
                        head_result.gather_ms, head_result.prep_ms,
                        head_result.broadcast_ms, head_result.projection_ms,
                        head_result.projection_kernel_worst_ms, head_result.top1_ms,
                        head_result.async_output_gate ? 1 : 0,
                        head_result.device_sync_count,
                        head_result.stream_sync_count,
                        head_result.event_sync_count,
                        head_result.tokens.empty() ? UINT32_MAX : head_result.tokens[0],
                        head_result.logits.empty() ? 0.0f : head_result.logits[0],
                        head_result.finite_bad,
                        (unsigned long long)head_result.checksum,
                        head_rc == 0 && head_result.pass ? "PASS" : "FAIL");
            if (head_rc != 0 || !head_result.pass) {
                if (lazy_output_head.initialized) {
                    close_shared_output_head(opt, &lazy_output_head);
                }
                return head_rc == 0 ? 14 : head_rc;
            }
            if (serving_result) {
                serving_result->diagnostic_output_head = true;
                serving_result->diagnostic_output_head_proxy_hc =
                    !opt.tp_hc_final_expand_gate;
                serving_result->output_head_ms = head_result.total_ms;
                serving_result->output_head_gather_ms = head_result.gather_ms;
                serving_result->output_head_prep_ms = head_result.prep_ms;
                serving_result->output_head_broadcast_ms = head_result.broadcast_ms;
                serving_result->output_head_projection_ms = head_result.projection_ms;
                serving_result->output_head_top1_ms = head_result.top1_ms;
                serving_result->selected_tokens = head_result.tokens;
                serving_result->selected_logits = head_result.logits;
                serving_result->checksum ^= head_result.checksum + 0x0A17EADull;
            }
        }
        if (lazy_output_head.initialized) {
            close_shared_output_head(opt, &lazy_output_head);
            if (report_vram_checkpoint(opt, "after_lazy_output_head_close") != 0) {
                return 14;
            }
            if (nccl_gate_active(opt) && opt.nccl_min_free_mib != 0) {
                (void)report_vram_checkpoint_min_free(
                    opt, "nccl_after_lazy_output_head_close",
                    opt.nccl_min_free_mib);
            }
        }
        if (opt.serving_bench) {
            std::printf("tp_ep_serving_bench\tschema\tds4_v100_tp_ep_serving_bench.v1\t"
                        "requests\t%d\tslots\t%d\tctx\t262144\tgenerated_per_request\t%d\t"
                        "prompt_tokens\t%llu\tgenerated_tokens\t%llu\t"
                        "continuation_tokens\t%llu\t"
                        "first_token_decode_ms\t%.6f\tcontinuation_decode_ms\t%.6f\t"
                        "first_token_wall_ms\t%.6f\tcontinuation_wall_ms\t%.6f\t"
                        "total_decode_ms\t%.6f\ttotal_wall_ms\t%.6f\t"
                        "aggregate_generated_tok_s_decode\t%.6f\t"
                        "aggregate_generated_tok_s_wall\t%.6f\t"
                        "aggregate_continuation_tok_s_decode\t%.6f\t"
                        "aggregate_continuation_tok_s_wall\t%.6f\t"
                        "checksum\t%llu\tPASS\n",
                        opt.slots, opt.slots, opt.decode_steps,
                        (unsigned long long)prompt_tokens,
                        (unsigned long long)generated_tokens,
                        (unsigned long long)continuation_tokens,
                        first_token_decode_ms, continuation_decode_ms,
                        first_token_wall_ms, continuation_wall_ms,
                        sum_decode_ms, wall_ms,
                        generated_tok_s_decode, generated_tok_s_wall,
                        continuation_tok_s_decode, continuation_tok_s_wall,
                        (unsigned long long)checksum);
        }
    }
    if (opt.decode_cudagraph_gate) {
        const int graph_audit_steps = opt.warmup + opt.decode_steps;
        const int total_stream_syncs = sum_cudagraph_rank_stream_syncs +
                                      sum_cudagraph_dense_stream_syncs +
                                      sum_cudagraph_copy_stream_syncs;
        const bool output_head_outside_step =
            shared_output_head && shared_output_head->initialized &&
            shared_rank_buffers && shared_rank_buffers->initialized;
        const bool host_token_dependency =
            output_head_outside_step && serving_result &&
            serving_result->diagnostic_output_head;
        const int helper_host_sync_blocker_classes =
            (opt.tp_hc_current_input_gate && !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_projection_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_compressed_kv_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_state_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_typed_kv_history_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_raw_read_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_output_gate ? 1 : 0) +
            (opt.true_ds4_post_attention_ffn_input_gate ? 1 : 0) +
            (opt.final_hc_carry_gate && opt.tp_hc_final_expand_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0);
        const bool capture_eligible =
            total_stream_syncs == 0 && sum_cudagraph_sync_all_calls == 0 &&
            helper_host_sync_blocker_classes == 0;
        const char *blocker = capture_eligible
            ? "none"
            : (total_stream_syncs != 0 || sum_cudagraph_sync_all_calls != 0
                   ? "host_stream_synchronization"
                   : "helper_host_synchronization");
        std::printf("tp_ep_decode_cudagraph_audit\tsteps\t%d\t"
                    "sync_all_calls\t%d\tevent_barrier_calls\t%d\t"
                    "stream_sync_count\t%d\t"
                    "rank_stream_sync_count\t%d\tdense_stream_sync_count\t%d\t"
                    "copy_stream_sync_count\t%d\toutput_head_outside_step\t%d\t"
                    "host_selected_token_dependency\t%d\t"
                    "helper_host_sync_blocker_classes\t%d\t"
                    "capture_attempted\t%d\tcapture_succeeded\t%d\t"
                    "capture_error_code\t%d\tcapture_error_name\t%s\t"
                    "capture_nodes\t%zu\tcapture_eligible\t%d\tblocker\t%s\n",
                    graph_audit_steps,
                    sum_cudagraph_sync_all_calls,
                    sum_cudagraph_event_barrier_calls,
                    total_stream_syncs,
                    sum_cudagraph_rank_stream_syncs,
                    sum_cudagraph_dense_stream_syncs,
                    sum_cudagraph_copy_stream_syncs,
                    output_head_outside_step ? 1 : 0,
                    host_token_dependency ? 1 : 0,
                    helper_host_sync_blocker_classes,
                    sum_cudagraph_capture_attempted,
                    sum_cudagraph_capture_succeeded,
                    sum_cudagraph_capture_error,
                    cudaGetErrorName((cudaError_t)sum_cudagraph_capture_error),
                    sum_cudagraph_capture_nodes,
                    capture_eligible ? 1 : 0,
                    blocker);
    }
    return 0;
}

static int http_write_json(int fd, int status, const char *body) {
    const char *reason = status == 200 ? "OK" : (status == 404 ? "Not Found" : "Error");
    const int n = dprintf(fd,
                          "HTTP/1.1 %d %s\r\n"
                          "Connection: close\r\n"
                          "Content-Type: application/json\r\n"
                          "Content-Length: %zu\r\n\r\n"
                          "%s",
                          status, reason, std::strlen(body), body);
    return n < 0 ? -1 : 0;
}

static int http_write_text(int fd, const char *body) {
    const int n = dprintf(fd,
                          "HTTP/1.1 200 OK\r\n"
                          "Connection: close\r\n"
                          "Content-Type: text/plain; version=0.0.4\r\n"
                          "Content-Length: %zu\r\n\r\n"
                          "%s",
                          std::strlen(body), body);
    return n < 0 ? -1 : 0;
}

static int json_find_int(const char *body, const char *key, int fallback) {
    if (!body || !key) return fallback;
    const char *p = std::strstr(body, key);
    if (!p) return fallback;
    p += std::strlen(key);
    while (*p && (*p == '"' || *p == '\'' || *p == ' ' || *p == '\t' || *p == ':')) ++p;
    char *end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p || v < 0 || v > 1000000) return fallback;
    return (int)v;
}

struct HttpParsedRequest {
    int fd = -1;
    std::string method;
    std::string path;
    std::string body;
    int requested_tokens = 0;
    std::string cache_key;
    bool cache_key_explicit = false;
    bool prompt_fingerprint_present = false;
    uint64_t prompt_fingerprint = 0;
    std::vector<uint32_t> prompt_token_ids;
    uint64_t cache_position = 0;
    int cache_slot = -1;
    bool cache_hit = false;
    bool cache_prompt_match = true;
    bool cache_evicted = false;
    std::string evicted_key;
    uint32_t decode_input_token = UINT32_MAX;
    std::vector<uint32_t> generated_token_ids;
    uint64_t prompt_prefill_tokens = 0;
};

static int http_content_length(const char *req) {
    const char *p = std::strstr(req, "Content-Length:");
    if (!p) return 0;
    p += std::strlen("Content-Length:");
    while (*p == ' ' || *p == '\t') ++p;
    char *end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p || v < 0 || v > 4096) return 0;
    return (int)v;
}

static std::string json_find_string(const char *body, const char *key) {
    if (!body || !key) return "";
    const char *p = std::strstr(body, key);
    if (!p) return "";
    p += std::strlen(key);
    while (*p && (*p == ' ' || *p == '\t' || *p == ':')) ++p;
    if (*p != '"' && *p != '\'') {
        while (*p && *p != '"' && *p != '\'') ++p;
    }
    if (!*p) return "";
    const char quote = *p++;
    std::string out;
    while (*p && *p != quote && out.size() < 256) {
        if (*p == '\\' && p[1]) {
            ++p;
        }
        out.push_back(*p++);
    }
    return out;
}

static uint64_t fnv1a64(const std::string &s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= (uint64_t)c;
        h *= 1099511628211ull;
    }
    return h;
}

static std::string http_json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out.push_back(c);
                }
                break;
        }
    }
    return out;
}

static std::string http_json_uint_array(const std::vector<uint32_t> &values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out += ",";
        out += std::to_string((unsigned long long)values[i]);
    }
    out += "]";
    return out;
}

static bool http_is_chat_completion_post(const HttpParsedRequest &req);

struct TokenizerRuntime {
    ds4_engine *engine = nullptr;
    bool initialized = false;
};

static bool open_tokenizer_runtime(const char *model_path, TokenizerRuntime *out) {
    if (!model_path || !model_path[0] || !out) return false;
    ds4_engine_options opt;
    std::memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = DS4_BACKEND_CPU;
    opt.inspect_only = true;
    opt.n_threads = 1;
    if (ds4_engine_open(&out->engine, &opt) != 0 || !out->engine) {
        out->engine = nullptr;
        out->initialized = false;
        return false;
    }
    out->initialized = true;
    return true;
}

static void close_tokenizer_runtime(TokenizerRuntime *rt) {
    if (!rt) return;
    if (rt->engine) ds4_engine_close(rt->engine);
    rt->engine = nullptr;
    rt->initialized = false;
}

static std::string decode_token_text(ds4_engine *engine,
                                     const std::vector<uint32_t> &tokens) {
    if (!engine) return "";
    std::string out;
    for (uint32_t token : tokens) {
        size_t len = 0;
        char *piece = ds4_token_text(engine, (int)token, &len);
        if (piece && len > 0) out.append(piece, len);
        std::free(piece);
    }
    return out;
}

static bool materialize_prompt_tokens(ds4_engine *engine,
                                      HttpParsedRequest *req,
                                      std::string *err) {
    if (!req || !req->prompt_token_ids.empty()) return true;

    const std::string prompt = json_find_string(req->body.c_str(), "\"prompt\"");
    const std::string content = json_find_string(req->body.c_str(), "\"content\"");
    const bool has_text = !prompt.empty() || !content.empty();
    if (!has_text) return true;
    if (!engine) {
        if (err) *err = "text_prompt_requires_tokenizer";
        return false;
    }

    ds4_tokens toks;
    std::memset(&toks, 0, sizeof(toks));
    if (!content.empty() && http_is_chat_completion_post(*req)) {
        ds4_encode_chat_prompt(engine, "", content.c_str(), DS4_THINK_NONE, &toks);
    } else {
        ds4_tokenize_text(engine, prompt.c_str(), &toks);
    }
    req->prompt_token_ids.clear();
    req->prompt_token_ids.reserve((size_t)toks.len);
    for (int i = 0; i < toks.len; ++i) {
        if (toks.v[i] >= 0) req->prompt_token_ids.push_back((uint32_t)toks.v[i]);
    }
    ds4_tokens_free(&toks);
    if (req->prompt_token_ids.empty()) {
        if (err) *err = "tokenizer_produced_no_tokens";
        return false;
    }
    return true;
}

static bool http_read_request(int fd, HttpParsedRequest *out) {
    char req[8192];
    size_t used = 0;
    for (;;) {
        if (used + 1 >= sizeof(req)) return false;
        const ssize_t nr = read(fd, req + used, sizeof(req) - 1 - used);
        if (nr <= 0) return false;
        used += (size_t)nr;
        req[used] = '\0';
        const char *body = std::strstr(req, "\r\n\r\n");
        if (body) {
            const int content_length = http_content_length(req);
            const size_t header_bytes = (size_t)(body + 4 - req);
            if (used >= header_bytes + (size_t)content_length) break;
        }
    }
    char method[16] = {};
    char path[256] = {};
    if (std::sscanf(req, "%15s %255s", method, path) != 2) return false;
    const char *body = std::strstr(req, "\r\n\r\n");
    out->fd = fd;
    out->method = method;
    out->path = path;
    out->body = body ? body + 4 : "";
    return true;
}

static bool http_is_generation_post(const HttpParsedRequest &req) {
    return req.method == "POST" &&
           (req.path == "/v100/selected-token" ||
            req.path == "/v1/v100/selected-token" ||
            req.path == "/v1/completions" ||
            req.path == "/v1/chat/completions" ||
            req.path == "/v100/diagnostic-completions");
}

static bool http_is_completion_post(const HttpParsedRequest &req) {
    return req.method == "POST" &&
           (req.path == "/v1/completions" ||
            req.path == "/v100/diagnostic-completions");
}

static bool http_is_chat_completion_post(const HttpParsedRequest &req) {
    return req.method == "POST" && req.path == "/v1/chat/completions";
}

static bool http_wait_for_connection(int listen_fd, int wait_us) {
    if (wait_us <= 0) return false;
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(listen_fd, &rfds);
    timeval tv = {};
    tv.tv_sec = wait_us / 1000000;
    tv.tv_usec = wait_us % 1000000;
    const int rc = select(listen_fd + 1, &rfds, nullptr, nullptr, &tv);
    return rc > 0 && FD_ISSET(listen_fd, &rfds);
}

static int http_requested_tokens(const HttpParsedRequest &req, int fallback) {
    int out = json_find_int(req.body.c_str(), "max_tokens", fallback);
    if (out <= 0) out = fallback;
    if (out <= 0) out = 1;
    return out;
}

static std::string http_request_cache_key(const HttpParsedRequest &req,
                                          uint64_t request_serial,
                                          bool *explicit_key) {
    std::string key = json_find_string(req.body.c_str(), "\"session_id\"");
    if (key.empty()) key = json_find_string(req.body.c_str(), "\"cache_key\"");
    if (key.empty()) key = json_find_string(req.body.c_str(), "\"conversation_id\"");
    if (!key.empty()) {
        if (explicit_key) *explicit_key = true;
        return key;
    }

    const std::string prompt = json_find_string(req.body.c_str(), "\"prompt\"");
    if (!prompt.empty()) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "prompt:%016llx",
                      (unsigned long long)fnv1a64(prompt));
        if (explicit_key) *explicit_key = false;
        return buf;
    }

    char buf[64];
    std::snprintf(buf, sizeof(buf), "ephemeral:%llu",
                  (unsigned long long)request_serial);
    if (explicit_key) *explicit_key = false;
    return buf;
}

static bool json_find_uint_array(const char *body,
                                 const char *key,
                                 std::vector<uint32_t> *out,
                                 size_t limit) {
    out->clear();
    if (!body || !key) return false;
    const char *p = std::strstr(body, key);
    if (!p) return false;
    p += std::strlen(key);
    while (*p && *p != '[' && *p != '{' && *p != '"' && *p != '\'') ++p;
    if (*p != '[') return false;
    ++p;
    while (*p && *p != ']') {
        while (*p == ' ' || *p == '\t' || *p == '\n' ||
               *p == '\r' || *p == ',') ++p;
        if (*p == ']') break;
        char *end = nullptr;
        unsigned long v = std::strtoul(p, &end, 10);
        if (end == p || v > UINT32_MAX || out->size() >= limit) {
            out->clear();
            return false;
        }
        out->push_back((uint32_t)v);
        p = end;
        while (*p == ' ' || *p == '\t' || *p == '\n' ||
               *p == '\r') ++p;
        if (*p && *p != ',' && *p != ']') {
            out->clear();
            return false;
        }
    }
    return *p == ']' && !out->empty();
}

static uint64_t fnv1a64_u32(const std::vector<uint32_t> &tokens) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t token : tokens) {
        for (int i = 0; i < 4; ++i) {
            h ^= (uint64_t)((token >> (8 * i)) & 0xffu);
            h *= 1099511628211ull;
        }
    }
    return h;
}

static void http_request_prompt_fingerprint(HttpParsedRequest *req) {
    if (!req->prompt_token_ids.empty()) {
        req->prompt_fingerprint_present = true;
        req->prompt_fingerprint = fnv1a64_u32(req->prompt_token_ids);
        return;
    }
    if (json_find_uint_array(req->body.c_str(), "\"prompt_tokens\"",
                             &req->prompt_token_ids, 262144) ||
        json_find_uint_array(req->body.c_str(), "\"prompt\"",
                             &req->prompt_token_ids, 262144)) {
        req->prompt_fingerprint_present = true;
        req->prompt_fingerprint = fnv1a64_u32(req->prompt_token_ids);
        return;
    }

    const std::string prompt = json_find_string(req->body.c_str(), "\"prompt\"");
    if (prompt.empty()) {
        req->prompt_fingerprint_present = false;
        req->prompt_fingerprint = 0;
        req->prompt_token_ids.clear();
        return;
    }
    req->prompt_fingerprint_present = true;
    req->prompt_fingerprint = fnv1a64(prompt);
}

struct TpEpHttpSessionSlot {
    int id = -1;
    bool occupied = false;
    bool kv_valid = false;
    bool hc_valid = false;
    bool prompt_fingerprint_known = false;
    std::string key;
    uint64_t prompt_fingerprint = 0;
    std::vector<uint32_t> prompt_token_ids;
    std::vector<uint32_t> generated_token_ids;
    uint64_t pos = 0;
    uint64_t prompt_tokens = 0;
    uint64_t generated_tokens = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t last_used = 0;
    uint32_t last_selected_token = UINT32_MAX;
};

struct TpEpHttpSessionAssignment {
    int slot = -1;
    bool hit = false;
    bool prompt_match = true;
    bool evicted = false;
    std::string evicted_key;
    uint64_t pos_in = 0;
    uint64_t pos_out = 0;
};

struct TpEpHttpContextAdmission {
    bool ok = false;
    bool cache_hit = false;
    uint64_t start_position = 0;
    uint64_t prompt_prefill_steps = 0;
    uint64_t requested_decode_steps = 0;
    uint64_t final_position = 0;
    uint64_t ctx = 262144ull;
};

struct TpEpHttpSessionTable {
    std::vector<TpEpHttpSessionSlot> slots;
    uint64_t clock = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;

    void init(int n_slots) {
        slots.resize((size_t)n_slots);
        for (int i = 0; i < n_slots; ++i) {
            slots[(size_t)i].id = i;
        }
    }

    int find(const std::string &key) const {
        for (const auto &slot : slots) {
            if (slot.occupied && slot.key == key) return slot.id;
        }
        return -1;
    }

    bool slot_prompt_matches(const TpEpHttpSessionSlot &slot,
                             bool prompt_present,
                             uint64_t prompt_fingerprint) const {
        if (!prompt_present) return true;
        return slot.prompt_fingerprint_known &&
               slot.prompt_fingerprint == prompt_fingerprint;
    }

    uint64_t preview_position(const std::string &key,
                              bool prompt_present,
                              uint64_t prompt_fingerprint,
                              uint64_t base_pos) const {
        const int idx = find(key);
        if (idx >= 0 &&
            slots[(size_t)idx].kv_valid &&
            slots[(size_t)idx].hc_valid &&
            slot_prompt_matches(slots[(size_t)idx],
                                prompt_present,
                                prompt_fingerprint)) {
            return slots[(size_t)idx].pos;
        }
        return base_pos;
    }

    bool preview_hit(const std::string &key,
                     bool prompt_present,
                     uint64_t prompt_fingerprint,
                     uint64_t *pos_out) const {
        const int idx = find(key);
        if (idx >= 0 &&
            slots[(size_t)idx].kv_valid &&
            slots[(size_t)idx].hc_valid &&
            slot_prompt_matches(slots[(size_t)idx],
                                prompt_present,
                                prompt_fingerprint)) {
            if (pos_out) *pos_out = slots[(size_t)idx].pos;
            return true;
        }
        return false;
    }

    TpEpHttpSessionAssignment assign(const std::string &key,
                                     bool prompt_present,
                                     uint64_t prompt_fingerprint,
                                     const std::vector<uint32_t> &prompt_tokens,
                                     uint64_t base_pos,
                                     const std::vector<bool> &protected_slots) {
        TpEpHttpSessionAssignment a;
        ++clock;

        int idx = find(key);
        if (idx >= 0) {
            auto &slot = slots[(size_t)idx];
            const bool prompt_match =
                slot_prompt_matches(slot, prompt_present, prompt_fingerprint);
            a.slot = idx;
            a.prompt_match = prompt_match;
            a.hit = slot.kv_valid && slot.hc_valid && prompt_match;
            a.pos_in = a.hit ? slot.pos : base_pos;
            slot.last_used = clock;
            if (a.hit) {
                hits++;
                slot.hits++;
            } else {
                misses++;
                slot.misses++;
                slot.kv_valid = false;
                slot.hc_valid = false;
                slot.pos = base_pos;
                if (prompt_present) {
                    slot.prompt_fingerprint_known = true;
                    slot.prompt_fingerprint = prompt_fingerprint;
                }
                slot.prompt_token_ids = prompt_tokens;
                slot.generated_token_ids.clear();
                slot.prompt_tokens = 0;
                slot.generated_tokens = 0;
                slot.last_selected_token = UINT32_MAX;
            }
            return a;
        }

        for (auto &slot : slots) {
            if (!slot.occupied) {
                idx = slot.id;
                break;
            }
        }
        if (idx < 0) {
            uint64_t best_last = UINT64_MAX;
            for (const auto &slot : slots) {
                if (slot.id >= 0 && slot.id < (int)protected_slots.size() &&
                    protected_slots[(size_t)slot.id]) {
                    continue;
                }
                if (slot.last_used < best_last) {
                    best_last = slot.last_used;
                    idx = slot.id;
                }
            }
        }
        if (idx < 0) return a;

        auto &slot = slots[(size_t)idx];
        if (slot.occupied) {
            a.evicted = true;
            a.evicted_key = slot.key;
            evictions++;
        }
        slot.occupied = true;
        slot.kv_valid = false;
        slot.hc_valid = false;
        slot.prompt_fingerprint_known = prompt_present;
        slot.key = key;
        slot.prompt_fingerprint = prompt_present ? prompt_fingerprint : 0;
        slot.prompt_token_ids = prompt_tokens;
        slot.generated_token_ids.clear();
        slot.pos = base_pos;
        slot.prompt_tokens = 0;
        slot.generated_tokens = 0;
        slot.hits = 0;
        slot.misses = 1;
        slot.last_used = clock;
        slot.last_selected_token = UINT32_MAX;
        misses++;

        a.slot = idx;
        a.hit = false;
        a.prompt_match = true;
        a.pos_in = base_pos;
        return a;
    }

    void commit(const TpEpHttpSessionAssignment &a,
                uint64_t prompt_tokens,
                uint64_t generated_tokens,
                uint64_t position_advance,
                const std::vector<uint32_t> &selected_tokens) {
        if (a.slot < 0 || a.slot >= (int)slots.size()) return;
        auto &slot = slots[(size_t)a.slot];
        slot.kv_valid = true;
        slot.hc_valid = true;
        slot.pos = a.pos_in + position_advance;
        slot.prompt_tokens += prompt_tokens;
        slot.generated_tokens += generated_tokens;
        for (uint32_t selected_token : selected_tokens) {
            if (selected_token != UINT32_MAX) {
                slot.generated_token_ids.push_back(selected_token);
                slot.last_selected_token = selected_token;
            }
        }
        slot.last_used = ++clock;
    }

    int used() const {
        int n = 0;
        for (const auto &slot : slots) n += slot.occupied ? 1 : 0;
        return n;
    }

    void slots_json(char *out, size_t out_size) const {
        size_t used_bytes = 0;
        int n = std::snprintf(out, out_size,
                              "{\"slots_total\":%zu,\"slots_used\":%d,"
                              "\"cache_hits\":%llu,\"cache_misses\":%llu,"
                              "\"cache_evictions\":%llu,\"slots\":[",
                              slots.size(), used(),
                              (unsigned long long)hits,
                              (unsigned long long)misses,
                              (unsigned long long)evictions);
        if (n < 0) return;
        used_bytes = (size_t)std::min(n, (int)out_size);
        for (size_t i = 0; i < slots.size() && used_bytes + 256 < out_size; ++i) {
            const auto &slot = slots[i];
            const std::string key = http_json_escape(slot.key);
            n = std::snprintf(out + used_bytes, out_size - used_bytes,
                              "%s{\"id\":%d,\"occupied\":%d,\"key\":\"%s\","
                              "\"pos\":%llu,\"kv_valid\":%d,\"hc_valid\":%d,"
                              "\"prompt_fingerprint_known\":%d,"
                              "\"prompt_fingerprint\":%llu,"
                              "\"prompt_tokens\":%llu,\"generated_tokens\":%llu,"
                              "\"prompt_token_ids\":%zu,"
                              "\"generated_token_ids\":%zu,"
                              "\"last_selected_token\":%u,"
                              "\"hits\":%llu,\"misses\":%llu}",
                              i == 0 ? "" : ",",
                              slot.id, slot.occupied ? 1 : 0, key.c_str(),
                              (unsigned long long)slot.pos,
                              slot.kv_valid ? 1 : 0,
                              slot.hc_valid ? 1 : 0,
                              slot.prompt_fingerprint_known ? 1 : 0,
                              (unsigned long long)slot.prompt_fingerprint,
                              (unsigned long long)slot.prompt_tokens,
                              (unsigned long long)slot.generated_tokens,
                              slot.prompt_token_ids.size(),
                              slot.generated_token_ids.size(),
                              slot.last_selected_token,
                              (unsigned long long)slot.hits,
                              (unsigned long long)slot.misses);
            if (n < 0) break;
            used_bytes += (size_t)n;
        }
        if (used_bytes + 4 < out_size) {
            std::snprintf(out + used_bytes, out_size - used_bytes, "]}\n");
        }
    }
};

static TpEpHttpContextAdmission tp_ep_http_context_admission(
    const TpEpHttpSessionTable &sessions,
    const HttpParsedRequest &req,
    uint64_t base_position,
    uint64_t ctx) {
    TpEpHttpContextAdmission out;
    out.ctx = ctx;
    out.requested_decode_steps =
        req.requested_tokens > 0 ? (uint64_t)req.requested_tokens : 0ull;
    uint64_t hit_pos = 0;
    out.cache_hit = sessions.preview_hit(req.cache_key,
                                         req.prompt_fingerprint_present,
                                         req.prompt_fingerprint,
                                         &hit_pos);
    out.start_position = out.cache_hit ? hit_pos : base_position;
    if (!out.cache_hit && req.prompt_token_ids.size() > 1) {
        out.prompt_prefill_steps =
            (uint64_t)req.prompt_token_ids.size() - 1ull;
    }
    out.final_position = out.start_position + out.prompt_prefill_steps +
                         out.requested_decode_steps;
    out.ok = out.final_position <= out.ctx;
    return out;
}

static std::string tp_ep_http_context_error_json(
    const TpEpHttpContextAdmission &admission) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "{\"error\":\"context_window_exceeded\","
                  "\"ctx\":%llu,"
                  "\"start_position\":%llu,"
                  "\"prompt_prefill_steps\":%llu,"
                  "\"requested_decode_steps\":%llu,"
                  "\"final_position\":%llu,"
                  "\"cache_hit\":%d}\n",
                  (unsigned long long)admission.ctx,
                  (unsigned long long)admission.start_position,
                  (unsigned long long)admission.prompt_prefill_steps,
                  (unsigned long long)admission.requested_decode_steps,
                  (unsigned long long)admission.final_position,
                  admission.cache_hit ? 1 : 0);
    return std::string(buf);
}

static unsigned long long http_epoch_seconds() {
    using namespace std::chrono;
    return (unsigned long long)duration_cast<seconds>(
        system_clock::now().time_since_epoch()).count();
}

static void http_drain_matching_pending(std::deque<HttpParsedRequest> *pending,
                                        int requested_tokens,
                                        uint64_t cache_position,
                                        int max_batch,
                                        std::vector<HttpParsedRequest> *batch) {
    for (auto it = pending->begin();
         it != pending->end() && (int)batch->size() < max_batch;) {
        bool duplicate_key = false;
        for (const auto &req : *batch) {
            if (req.cache_key == it->cache_key) {
                duplicate_key = true;
                break;
            }
        }
        if (it->requested_tokens == requested_tokens &&
            it->cache_position == cache_position &&
            !duplicate_key) {
            batch->push_back(std::move(*it));
            it = pending->erase(it);
        } else {
            ++it;
        }
    }
}

int run_tp_ep_http_server(const Options &base_opt,
                          const DenseF16Cache *shared_dense_f16_cache,
                          const SharedApi *shared_api,
                          SharedRankBuffers *shared_rank_buffers,
                          SharedTpRuntime *shared_tp_runtime,
                          const SharedExpertBindings *shared_expert_bindings,
                          const SharedDenseOps *shared_dense_ops,
                          SharedOutputHead *shared_output_head,
                          SharedHcControls *shared_hc_controls,
                          SharedTokenEmbedding *shared_token_embedding,
                          std::vector<ContractRow> resident_rows[43],
                          LayerStats resident_stats[43]) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::perror("tp_ep_http_socket");
        return 30;
    }
    int yes = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)base_opt.port);
    if (inet_pton(AF_INET, base_opt.host, &addr.sin_addr) != 1) {
        std::fprintf(stderr, "tp_ep_http_bad_host\t%s\n", base_opt.host);
        close(listen_fd);
        return 31;
    }
    if (bind(listen_fd, (sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(listen_fd, 16) != 0) {
        std::perror("tp_ep_http_bind_listen");
        close(listen_fd);
        return 32;
    }

    uint64_t served = 0;
    uint64_t generation_requests = 0;
    uint64_t generation_batches = 0;
    uint64_t coalesced_requests = 0;
    uint64_t bucketed_requests = 0;
    uint64_t rejected = 0;
    TpEpHttpSessionTable sessions;
    sessions.init(base_opt.slots);
    std::deque<HttpParsedRequest> pending_generation;
    uint64_t next_position = base_opt.position;
    uint64_t total_prompt_tokens = 0;
    uint64_t total_generated_tokens = 0;
    uint64_t total_continuation_tokens = 0;
    double total_decode_ms = 0.0;
    double total_wall_ms = 0.0;
    double total_continuation_decode_ms = 0.0;
    double total_continuation_wall_ms = 0.0;
    double total_ep_ms = 0.0;
    double total_dense_ms = 0.0;
    double total_compose_ms = 0.0;
    double total_compose_reduce_ms = 0.0;
    double total_compose_copy_ms = 0.0;
    double total_compose_final_ms = 0.0;
    ServingBenchResult last = {};
    TokenizerRuntime tokenizer;
    if (base_opt.tokenizer_model_path && base_opt.tokenizer_model_path[0]) {
        if (!open_tokenizer_runtime(base_opt.tokenizer_model_path, &tokenizer)) {
            std::fprintf(stderr, "tp_ep_http tokenizer open failed: %s\n",
                         base_opt.tokenizer_model_path);
            close(listen_fd);
            return 31;
        }
        std::printf("tp_ep_http_tokenizer\tmodel\t%s\tPASS\n",
                    base_opt.tokenizer_model_path);
    }
    std::printf("tp_ep_http_serving\thttp://%s:%d/v100/selected-token\tPASS\n",
                base_opt.host, base_opt.port);
    std::printf("tp_ep_http_completions\thttp://%s:%d/v1/completions\tDIAGNOSTIC\n",
                base_opt.host, base_opt.port);
    std::printf("tp_ep_http_chat_completions\thttp://%s:%d/v1/chat/completions\tDIAGNOSTIC\n",
                base_opt.host, base_opt.port);
    std::fflush(stdout);

    while (base_opt.max_requests == 0 || (int)served < base_opt.max_requests ||
           !pending_generation.empty()) {
        HttpParsedRequest first_req;
        int fd = -1;
        if (!pending_generation.empty()) {
            first_req = std::move(pending_generation.front());
            pending_generation.pop_front();
            fd = first_req.fd;
        } else {
            fd = accept(listen_fd, nullptr, nullptr);
            if (fd < 0) {
                if (errno == EINTR) continue;
                std::perror("tp_ep_http_accept");
                break;
            }
            served++;
            if (!http_read_request(fd, &first_req)) {
                close(fd);
                continue;
            }
        }

        if (first_req.method == "GET" && first_req.path == "/health") {
            http_write_json(fd, 200, "{\"status\":\"ok\",\"backend\":\"tp_ep_resident\"}\n");
        } else if (first_req.method == "GET" &&
                   (first_req.path == "/status" || first_req.path == "/v100/status")) {
            const double cumulative_generated_tok_s_wall = total_wall_ms > 0.0
                ? (double)total_generated_tokens * 1000.0 / total_wall_ms
                : 0.0;
            const double cumulative_generated_tok_s_decode = total_decode_ms > 0.0
                ? (double)total_generated_tokens * 1000.0 / total_decode_ms
                : 0.0;
            const double cumulative_continuation_tok_s_wall = total_continuation_wall_ms > 0.0
                ? (double)total_continuation_tokens * 1000.0 / total_continuation_wall_ms
                : 0.0;
            const double cumulative_continuation_tok_s_decode = total_continuation_decode_ms > 0.0
                ? (double)total_continuation_tokens * 1000.0 / total_continuation_decode_ms
                : 0.0;
            char out[8192];
            std::snprintf(out, sizeof(out),
                          "{\"status\":\"ok\",\"backend\":\"tp_ep_resident\","
                          "\"tp\":8,\"ep\":8,\"pp\":1,\"ctx\":262144,"
                          "\"slots\":%d,\"served_requests\":%llu,"
                          "\"generation_requests\":%llu,\"generation_batches\":%llu,"
                          "\"coalesced_requests\":%llu,\"bucketed_requests\":%llu,"
                          "\"pending_generation_requests\":%zu,"
                          "\"microbatch_wait_us\":%d,"
                          "\"kv_runtime_resident\":%d,"
                          "\"kv_all_slots_gate\":%d,"
                          "\"hc_persist_state_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_raw_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_compressed_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_indexer_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_history_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_skip_current_load_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_skip_raw_store_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_skip_compressed_store_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_skip_indexer_store_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_quiet_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_batch_rows_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_stream_sync_gate\":%d,"
                          "\"fp8_e5m2_kv_gate\":%d,"
                          "\"router_hash_fast_gate\":%d,"
                          "\"route_plan_async_upload_gate\":%d,"
                          "\"cache_slots_total\":%zu,"
                          "\"cache_slots_used\":%d,"
                          "\"cache_hits\":%llu,"
                          "\"cache_misses\":%llu,"
                          "\"cache_evictions\":%llu,"
                          "\"rejected_requests\":%llu,"
                          "\"total_prompt_tokens\":%llu,"
                          "\"total_generated_tokens\":%llu,"
                          "\"total_continuation_tokens\":%llu,"
                          "\"next_position\":%llu,"
                          "\"warmed_ready\":true,\"resident_ready\":true,"
                          "\"last_generated_tok_s_wall\":%.6f,"
                          "\"last_continuation_tok_s_wall\":%.6f,"
                          "\"last_compose_copy_ms\":%.6f,"
                          "\"cumulative_generated_tok_s_wall\":%.6f,"
                          "\"cumulative_continuation_tok_s_wall\":%.6f,"
                          "\"cumulative_generated_tok_s_decode\":%.6f,"
                          "\"cumulative_continuation_tok_s_decode\":%.6f,"
                          "\"cumulative_ep_ms\":%.6f,"
                          "\"cumulative_dense_ms\":%.6f,"
                          "\"cumulative_compose_ms\":%.6f,"
                          "\"cumulative_compose_reduce_ms\":%.6f,"
                          "\"cumulative_compose_copy_ms\":%.6f,"
                          "\"cumulative_compose_final_ms\":%.6f}\n",
                          base_opt.slots,
                          (unsigned long long)served,
                          (unsigned long long)generation_requests,
                          (unsigned long long)generation_batches,
                          (unsigned long long)coalesced_requests,
                          (unsigned long long)bucketed_requests,
                          pending_generation.size(),
                          base_opt.microbatch_wait_us,
                          shared_tp_runtime && shared_tp_runtime->initialized ? 1 : 0,
                          base_opt.tp_kv_all_slots_gate ? 1 : 0,
                          base_opt.tp_hc_persist_state_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_raw_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_compressed_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_indexer_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_history_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_current_load_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_raw_store_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_compressed_store_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_indexer_store_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_quiet_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_batch_rows_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_stream_sync_gate ? 1 : 0,
                          base_opt.fp8_e5m2_kv_gate ? 1 : 0,
                          base_opt.router_hash_fast_gate ? 1 : 0,
                          base_opt.route_plan_async_upload_gate ? 1 : 0,
                          sessions.slots.size(),
                          sessions.used(),
                          (unsigned long long)sessions.hits,
                          (unsigned long long)sessions.misses,
                          (unsigned long long)sessions.evictions,
                          (unsigned long long)rejected,
                          (unsigned long long)total_prompt_tokens,
                          (unsigned long long)total_generated_tokens,
                          (unsigned long long)total_continuation_tokens,
                          (unsigned long long)next_position,
                          last.aggregate_generated_tok_s_wall,
                          last.aggregate_continuation_tok_s_wall,
                          last.total_compose_copy_ms,
                          cumulative_generated_tok_s_wall,
                          cumulative_continuation_tok_s_wall,
                          cumulative_generated_tok_s_decode,
                          cumulative_continuation_tok_s_decode,
                          total_ep_ms,
                          total_dense_ms,
                          total_compose_ms,
                          total_compose_reduce_ms,
                          total_compose_copy_ms,
                          total_compose_final_ms);
            http_write_json(fd, 200, out);
        } else if (first_req.method == "GET" && first_req.path == "/v100/slots") {
            char out[16384];
            sessions.slots_json(out, sizeof(out));
            http_write_json(fd, 200, out);
        } else if (first_req.method == "GET" && first_req.path == "/metrics") {
            const double cumulative_generated_tok_s_wall = total_wall_ms > 0.0
                ? (double)total_generated_tokens * 1000.0 / total_wall_ms
                : 0.0;
            const double cumulative_generated_tok_s_decode = total_decode_ms > 0.0
                ? (double)total_generated_tokens * 1000.0 / total_decode_ms
                : 0.0;
            const double cumulative_continuation_tok_s_wall = total_continuation_wall_ms > 0.0
                ? (double)total_continuation_tokens * 1000.0 / total_continuation_wall_ms
                : 0.0;
            const double cumulative_continuation_tok_s_decode = total_continuation_decode_ms > 0.0
                ? (double)total_continuation_tokens * 1000.0 / total_continuation_decode_ms
                : 0.0;
            char out[6144];
            std::snprintf(out, sizeof(out),
                          "ds4_v100_tp_ep_resident_ready 1\n"
                          "ds4_v100_tp_ep_slots %d\n"
                          "ds4_v100_tp_ep_served_requests %llu\n"
                          "ds4_v100_tp_ep_generation_requests %llu\n"
                          "ds4_v100_tp_ep_generation_batches %llu\n"
                          "ds4_v100_tp_ep_coalesced_requests %llu\n"
                          "ds4_v100_tp_ep_bucketed_requests %llu\n"
                          "ds4_v100_tp_ep_pending_generation_requests %zu\n"
                          "ds4_v100_tp_ep_microbatch_wait_us %d\n"
                          "ds4_v100_tp_ep_kv_runtime_resident %d\n"
                          "ds4_v100_tp_ep_kv_all_slots_gate %d\n"
                          "ds4_v100_tp_ep_hc_persist_state_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_raw_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_compressed_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_indexer_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_history_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_skip_current_load_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_skip_raw_store_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_skip_compressed_store_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_skip_indexer_store_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_quiet_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_batch_rows_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_stream_sync_gate %d\n"
                          "ds4_v100_tp_ep_fp8_e5m2_kv_gate %d\n"
                          "ds4_v100_tp_ep_router_hash_fast_gate %d\n"
                          "ds4_v100_tp_ep_route_plan_async_upload_gate %d\n"
                          "ds4_v100_tp_ep_cache_slots_total %zu\n"
                          "ds4_v100_tp_ep_cache_slots_used %d\n"
                          "ds4_v100_tp_ep_cache_hits %llu\n"
                          "ds4_v100_tp_ep_cache_misses %llu\n"
                          "ds4_v100_tp_ep_cache_evictions %llu\n"
                          "ds4_v100_tp_ep_rejected_requests %llu\n"
                          "ds4_v100_tp_ep_total_prompt_tokens %llu\n"
                          "ds4_v100_tp_ep_total_generated_tokens %llu\n"
                          "ds4_v100_tp_ep_total_continuation_tokens %llu\n"
                          "ds4_v100_tp_ep_next_position %llu\n"
                          "ds4_v100_tp_ep_generated_tok_s_wall %.6f\n"
                          "ds4_v100_tp_ep_continuation_tok_s_wall %.6f\n"
                          "ds4_v100_tp_ep_last_compose_copy_ms %.6f\n"
                          "ds4_v100_tp_ep_cumulative_generated_tok_s_wall %.6f\n"
                          "ds4_v100_tp_ep_cumulative_continuation_tok_s_wall %.6f\n"
                          "ds4_v100_tp_ep_cumulative_generated_tok_s_decode %.6f\n"
                          "ds4_v100_tp_ep_cumulative_continuation_tok_s_decode %.6f\n"
                          "ds4_v100_tp_ep_cumulative_ep_ms %.6f\n"
                          "ds4_v100_tp_ep_cumulative_dense_ms %.6f\n"
                          "ds4_v100_tp_ep_cumulative_compose_ms %.6f\n"
                          "ds4_v100_tp_ep_cumulative_compose_reduce_ms %.6f\n"
                          "ds4_v100_tp_ep_cumulative_compose_copy_ms %.6f\n"
                          "ds4_v100_tp_ep_cumulative_compose_final_ms %.6f\n",
                          base_opt.slots,
                          (unsigned long long)served,
                          (unsigned long long)generation_requests,
                          (unsigned long long)generation_batches,
                          (unsigned long long)coalesced_requests,
                          (unsigned long long)bucketed_requests,
                          pending_generation.size(),
                          base_opt.microbatch_wait_us,
                          shared_tp_runtime && shared_tp_runtime->initialized ? 1 : 0,
                          base_opt.tp_kv_all_slots_gate ? 1 : 0,
                          base_opt.tp_hc_persist_state_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_raw_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_compressed_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_indexer_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_history_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_current_load_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_raw_store_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_compressed_store_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_indexer_store_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_quiet_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_batch_rows_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_stream_sync_gate ? 1 : 0,
                          base_opt.fp8_e5m2_kv_gate ? 1 : 0,
                          base_opt.router_hash_fast_gate ? 1 : 0,
                          base_opt.route_plan_async_upload_gate ? 1 : 0,
                          sessions.slots.size(),
                          sessions.used(),
                          (unsigned long long)sessions.hits,
                          (unsigned long long)sessions.misses,
                          (unsigned long long)sessions.evictions,
                          (unsigned long long)rejected,
                          (unsigned long long)total_prompt_tokens,
                          (unsigned long long)total_generated_tokens,
                          (unsigned long long)total_continuation_tokens,
                          (unsigned long long)next_position,
                          last.aggregate_generated_tok_s_wall,
                          last.aggregate_continuation_tok_s_wall,
                          last.total_compose_copy_ms,
                          cumulative_generated_tok_s_wall,
                          cumulative_continuation_tok_s_wall,
                          cumulative_generated_tok_s_decode,
                          cumulative_continuation_tok_s_decode,
                          total_ep_ms,
                          total_dense_ms,
                          total_compose_ms,
                          total_compose_reduce_ms,
                          total_compose_copy_ms,
                          total_compose_final_ms);
            http_write_text(fd, out);
        } else if (http_is_generation_post(first_req)) {
            std::string prompt_error;
            if (!materialize_prompt_tokens(tokenizer.engine, &first_req, &prompt_error)) {
                std::string body = "{\"error\":\"" + http_json_escape(prompt_error) + "\"}\n";
                http_write_json(first_req.fd, 400, body.c_str());
                close(first_req.fd);
                rejected++;
                continue;
            }
            first_req.requested_tokens = http_requested_tokens(first_req, base_opt.decode_steps);
            first_req.cache_key = http_request_cache_key(first_req,
                                                         served + pending_generation.size(),
                                                         &first_req.cache_key_explicit);
            http_request_prompt_fingerprint(&first_req);
            first_req.cache_position =
                sessions.preview_position(first_req.cache_key,
                                          first_req.prompt_fingerprint_present,
                                          first_req.prompt_fingerprint,
                                          base_opt.position);
            {
                const TpEpHttpContextAdmission admission =
                    tp_ep_http_context_admission(sessions, first_req,
                                                 base_opt.position, 262144ull);
                if (!admission.ok) {
                    std::fprintf(stderr,
                                 "tp_ep_http_context_rejected\tstart_position\t%llu\t"
                                 "prompt_prefill_steps\t%llu\trequested_steps\t%llu\t"
                                 "final_position\t%llu\tctx\t%llu\tcache_hit\t%d\n",
                                 (unsigned long long)admission.start_position,
                                 (unsigned long long)admission.prompt_prefill_steps,
                                 (unsigned long long)admission.requested_decode_steps,
                                 (unsigned long long)admission.final_position,
                                 (unsigned long long)admission.ctx,
                                 admission.cache_hit ? 1 : 0);
                    const std::string body =
                        tp_ep_http_context_error_json(admission);
                    http_write_json(first_req.fd, 400, body.c_str());
                    close(first_req.fd);
                    rejected++;
                    continue;
                }
            }

            std::vector<HttpParsedRequest> batch;
            batch.push_back(first_req);
            http_drain_matching_pending(&pending_generation,
                                        first_req.requested_tokens,
                                        first_req.cache_position,
                                        base_opt.slots,
                                        &batch);
            while ((int)batch.size() < base_opt.slots &&
                   http_wait_for_connection(listen_fd, base_opt.microbatch_wait_us)) {
                int extra_fd = accept(listen_fd, nullptr, nullptr);
                if (extra_fd < 0) {
                    if (errno == EINTR) continue;
                    break;
                }
                served++;
                HttpParsedRequest extra_req;
                if (!http_read_request(extra_fd, &extra_req)) {
                    close(extra_fd);
                    continue;
                }
                if (!http_is_generation_post(extra_req)) {
                    rejected++;
                    http_write_json(extra_fd, 404, "{\"error\":\"not_found_during_coalesce\"}\n");
                    close(extra_fd);
                    continue;
                }
                std::string extra_prompt_error;
                if (!materialize_prompt_tokens(tokenizer.engine, &extra_req, &extra_prompt_error)) {
                    std::string body = "{\"error\":\"" + http_json_escape(extra_prompt_error) + "\"}\n";
                    http_write_json(extra_fd, 400, body.c_str());
                    close(extra_fd);
                    rejected++;
                    continue;
                }
                extra_req.requested_tokens = http_requested_tokens(extra_req, first_req.requested_tokens);
                extra_req.cache_key = http_request_cache_key(extra_req,
                                                            served + pending_generation.size(),
                                                            &extra_req.cache_key_explicit);
                http_request_prompt_fingerprint(&extra_req);
                extra_req.cache_position =
                    sessions.preview_position(extra_req.cache_key,
                                              extra_req.prompt_fingerprint_present,
                                              extra_req.prompt_fingerprint,
                                              base_opt.position);
                {
                    const TpEpHttpContextAdmission admission =
                        tp_ep_http_context_admission(sessions, extra_req,
                                                     base_opt.position, 262144ull);
                    if (!admission.ok) {
                        std::fprintf(stderr,
                                     "tp_ep_http_context_rejected\tstart_position\t%llu\t"
                                     "prompt_prefill_steps\t%llu\trequested_steps\t%llu\t"
                                     "final_position\t%llu\tctx\t%llu\tcache_hit\t%d\n",
                                     (unsigned long long)admission.start_position,
                                     (unsigned long long)admission.prompt_prefill_steps,
                                     (unsigned long long)admission.requested_decode_steps,
                                     (unsigned long long)admission.final_position,
                                     (unsigned long long)admission.ctx,
                                     admission.cache_hit ? 1 : 0);
                        const std::string body =
                            tp_ep_http_context_error_json(admission);
                        http_write_json(extra_fd, 400, body.c_str());
                        close(extra_fd);
                        rejected++;
                        continue;
                    }
                }
                if (extra_req.requested_tokens != first_req.requested_tokens) {
                    bucketed_requests++;
                    pending_generation.push_back(std::move(extra_req));
                    continue;
                }
                if (extra_req.cache_position != first_req.cache_position) {
                    bucketed_requests++;
                    pending_generation.push_back(std::move(extra_req));
                    continue;
                }
                bool duplicate_key = false;
                for (const auto &req : batch) {
                    if (req.cache_key == extra_req.cache_key) {
                        duplicate_key = true;
                        break;
                    }
                }
                if (duplicate_key) {
                    bucketed_requests++;
                    pending_generation.push_back(std::move(extra_req));
                    continue;
                }
                batch.push_back(extra_req);
            }

            std::vector<TpEpHttpSessionAssignment> assignments(batch.size());
            std::vector<bool> protected_slots((size_t)base_opt.slots, false);
            bool assignment_failed = false;
            for (size_t i = 0; i < batch.size(); ++i) {
                assignments[i] = sessions.assign(batch[i].cache_key,
                                                 batch[i].prompt_fingerprint_present,
                                                 batch[i].prompt_fingerprint,
                                                 batch[i].prompt_token_ids,
                                                 base_opt.position,
                                                 protected_slots);
                if (assignments[i].slot < 0) {
                    assignment_failed = true;
                    break;
                }
                batch[i].cache_slot = assignments[i].slot;
                batch[i].cache_hit = assignments[i].hit;
                batch[i].cache_prompt_match = assignments[i].prompt_match;
                batch[i].cache_evicted = assignments[i].evicted;
                batch[i].evicted_key = assignments[i].evicted_key;
                batch[i].cache_position = assignments[i].pos_in;
                protected_slots[(size_t)assignments[i].slot] = true;
            }
            if (assignment_failed) {
                rejected += (uint64_t)batch.size();
                for (HttpParsedRequest &queued : batch) {
                    http_write_json(queued.fd, 503, "{\"error\":\"no_cache_slot_available\"}\n");
                    close(queued.fd);
                }
                continue;
            }

            Options req_opt = base_opt;
            const int requested_decode_steps = first_req.requested_tokens;
            req_opt.decode_steps = 1;
            req_opt.slots = base_opt.slots;
            req_opt.position = first_req.cache_position;
            req_opt.serving_bench = false;
            std::vector<uint32_t> decode_input_tokens((size_t)req_opt.slots, 0u);
            std::vector<unsigned char> decode_active_slots((size_t)req_opt.slots, 0u);
            for (size_t i = 0; i < batch.size(); ++i) {
                uint32_t input_token = 0;
                if (assignments[i].slot >= 0 &&
                    assignments[i].slot < (int)sessions.slots.size()) {
                    const TpEpHttpSessionSlot &slot =
                        sessions.slots[(size_t)assignments[i].slot];
                    if (assignments[i].hit &&
                        slot.last_selected_token != UINT32_MAX) {
                        input_token = slot.last_selected_token;
                    } else if (!batch[i].prompt_token_ids.empty()) {
                        input_token = batch[i].prompt_token_ids.back();
                    } else if (!slot.prompt_token_ids.empty()) {
                        input_token = slot.prompt_token_ids.back();
                    }
                } else if (!batch[i].prompt_token_ids.empty()) {
                    input_token = batch[i].prompt_token_ids.back();
                }
                batch[i].decode_input_token = input_token;
                if (batch[i].cache_slot >= 0 &&
                    batch[i].cache_slot < req_opt.slots) {
                    decode_input_tokens[(size_t)batch[i].cache_slot] = input_token;
                    decode_active_slots[(size_t)batch[i].cache_slot] = 1u;
                }
            }
            int max_prompt_prefill_steps = 0;
            int rc = 0;
            for (size_t i = 0; i < batch.size(); ++i) {
                if (!assignments[i].hit && batch[i].prompt_token_ids.size() > 1) {
                    max_prompt_prefill_steps = std::max(
                        max_prompt_prefill_steps,
                        (int)batch[i].prompt_token_ids.size() - 1);
                }
            }
            for (int prefill_step = 0; prefill_step < max_prompt_prefill_steps; ++prefill_step) {
                bool any_prefill = false;
                std::vector<uint32_t> prefill_input_tokens((size_t)req_opt.slots, 0u);
                for (size_t i = 0; i < batch.size(); ++i) {
                    const int slot = batch[i].cache_slot;
                    if (assignments[i].hit || slot < 0 || slot >= req_opt.slots) continue;
                    if ((size_t)prefill_step + 1u >= batch[i].prompt_token_ids.size()) continue;
                    const uint32_t tok = batch[i].prompt_token_ids[(size_t)prefill_step];
                    prefill_input_tokens[(size_t)slot] = tok;
                    batch[i].prompt_prefill_tokens++;
                    any_prefill = true;
                }
                if (!any_prefill) continue;
                Options prefill_opt = req_opt;
                prefill_opt.position = first_req.cache_position + (uint64_t)prefill_step;
                prefill_opt.diagnostic_output_head = false;
                prefill_opt.diagnostic_output_head_lazy_gate = false;
                std::vector<unsigned char> prefill_active_slots((size_t)req_opt.slots, 0u);
                for (size_t i = 0; i < batch.size(); ++i) {
                    const int slot = batch[i].cache_slot;
                    if (assignments[i].hit || slot < 0 || slot >= req_opt.slots) continue;
                    if ((size_t)prefill_step + 1u >= batch[i].prompt_token_ids.size()) continue;
                    prefill_active_slots[(size_t)slot] = 1u;
                }
                ServingBenchResult prefill_result;
                rc = run_token_major_serving_loop(prefill_opt,
                                                  shared_dense_f16_cache,
                                                  shared_api,
                                                  shared_rank_buffers,
                                                  shared_tp_runtime,
                                                  shared_expert_bindings,
                                                  shared_dense_ops,
                                                  nullptr,
                                                  shared_hc_controls,
                                                  shared_token_embedding,
                                                  &prefill_input_tokens,
                                                  &prefill_active_slots,
                                                  resident_rows,
                                                  resident_stats,
                                                  true,
                                                  &prefill_result);
                if (rc != 0) break;
                total_decode_ms += prefill_result.total_decode_ms;
                total_wall_ms += prefill_result.total_wall_ms;
                total_ep_ms += prefill_result.total_ep_ms;
                total_dense_ms += prefill_result.total_dense_ms;
                total_compose_ms += prefill_result.total_compose_ms;
                total_compose_reduce_ms += prefill_result.total_compose_reduce_ms;
                total_compose_copy_ms += prefill_result.total_compose_copy_ms;
                total_compose_final_ms += prefill_result.total_compose_final_ms;
            }
            ServingBenchResult result;
            bool missing_output_head = false;
            for (int step = 0; rc == 0 && step < requested_decode_steps; ++step) {
                req_opt.position = first_req.cache_position +
                                   (uint64_t)max_prompt_prefill_steps +
                                   (uint64_t)step;
                ServingBenchResult step_result;
                rc = run_token_major_serving_loop(req_opt,
                                                  shared_dense_f16_cache,
                                                  shared_api,
                                                  shared_rank_buffers,
                                                  shared_tp_runtime,
                                                  shared_expert_bindings,
                                                  shared_dense_ops,
                                                  shared_output_head,
                                                  shared_hc_controls,
                                                  shared_token_embedding,
                                                  &decode_input_tokens,
                                                  &decode_active_slots,
                                                  resident_rows,
                                                  resident_stats,
                                                  true,
                                                  &step_result);
                if (rc != 0) break;
                if (!step_result.diagnostic_output_head ||
                    step_result.selected_tokens.size() < (size_t)req_opt.slots) {
                    missing_output_head = true;
                    break;
                }
                result.prompt_tokens = step == 0 ? step_result.prompt_tokens : result.prompt_tokens;
                result.generated_tokens += (uint64_t)req_opt.slots;
                result.continuation_tokens = requested_decode_steps > 1
                    ? (uint64_t)req_opt.slots * (uint64_t)(requested_decode_steps - 1)
                    : 0ull;
                if (step == 0) {
                    result.first_token_decode_ms += step_result.first_token_decode_ms;
                    result.first_token_wall_ms += step_result.first_token_wall_ms;
                } else {
                    result.continuation_decode_ms += step_result.first_token_decode_ms;
                    result.continuation_wall_ms += step_result.first_token_wall_ms;
                }
                result.total_decode_ms += step_result.total_decode_ms;
                result.total_wall_ms += step_result.total_wall_ms;
                result.total_ep_ms += step_result.total_ep_ms;
                result.total_dense_ms += step_result.total_dense_ms;
                result.total_compose_ms += step_result.total_compose_ms;
                result.total_compose_reduce_ms += step_result.total_compose_reduce_ms;
                result.total_compose_copy_ms += step_result.total_compose_copy_ms;
                result.total_compose_final_ms += step_result.total_compose_final_ms;
                result.total_hc_current_input_ms += step_result.total_hc_current_input_ms;
                result.diagnostic_output_head = step_result.diagnostic_output_head;
                result.diagnostic_output_head_proxy_hc =
                    step_result.diagnostic_output_head_proxy_hc;
                result.output_head_ms += step_result.output_head_ms;
                result.output_head_gather_ms += step_result.output_head_gather_ms;
                result.output_head_prep_ms += step_result.output_head_prep_ms;
                result.output_head_broadcast_ms += step_result.output_head_broadcast_ms;
                result.output_head_projection_ms += step_result.output_head_projection_ms;
                result.output_head_top1_ms += step_result.output_head_top1_ms;
                result.token_input_seed = result.token_input_seed ||
                                          step_result.token_input_seed;
                if (step == 0) result.first_input_token = step_result.first_input_token;
                result.selected_tokens = step_result.selected_tokens;
                result.selected_logits = step_result.selected_logits;
                result.checksum ^= step_result.checksum +
                                   (uint64_t)(step + 1) * 0x9e3779b185ebca87ull;
                for (size_t i = 0; i < batch.size(); ++i) {
                    const int slot = batch[i].cache_slot;
                    if (slot >= 0 &&
                        (size_t)slot < step_result.selected_tokens.size()) {
                        const uint32_t tok = step_result.selected_tokens[(size_t)slot];
                        batch[i].generated_token_ids.push_back(tok);
                        decode_input_tokens[(size_t)slot] = tok;
                    }
                }
            }
            if (result.total_decode_ms > 0.0) {
                result.aggregate_generated_tok_s_decode =
                    (double)result.generated_tokens * 1000.0 / result.total_decode_ms;
                result.aggregate_continuation_tok_s_decode =
                    result.continuation_decode_ms > 0.0
                        ? (double)result.continuation_tokens * 1000.0 /
                              result.continuation_decode_ms
                        : 0.0;
            }
            if (result.total_wall_ms > 0.0) {
                result.aggregate_generated_tok_s_wall =
                    (double)result.generated_tokens * 1000.0 / result.total_wall_ms;
                result.aggregate_continuation_tok_s_wall =
                    result.continuation_wall_ms > 0.0
                        ? (double)result.continuation_tokens * 1000.0 /
                              result.continuation_wall_ms
                        : 0.0;
            }
            req_opt.decode_steps = requested_decode_steps;
            if (rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_http_decode_failed\trc\t%d\tbatch\t%zu\trequested_steps\t%d\tposition\t%llu\n",
                             rc, batch.size(), requested_decode_steps,
                             (unsigned long long)first_req.cache_position);
                std::fflush(stderr);
                rejected += (uint64_t)batch.size();
                for (HttpParsedRequest &queued : batch) {
                    http_write_json(queued.fd, 500, "{\"error\":\"tp_ep_decode_failed\"}\n");
                    close(queued.fd);
                }
            } else if (missing_output_head) {
                rejected += (uint64_t)batch.size();
                for (HttpParsedRequest &queued : batch) {
                    http_write_json(queued.fd, 500, "{\"error\":\"tp_ep_output_head_missing\"}\n");
                    close(queued.fd);
                }
            } else {
                const uint64_t batch_id = generation_batches + 1;
                uint64_t client_prompt_tokens = 0;
                for (const HttpParsedRequest &request : batch) {
                    client_prompt_tokens += request.prompt_token_ids.empty()
                        ? 1ull
                        : (uint64_t)request.prompt_token_ids.size();
                }
                const uint64_t client_generated_tokens =
                    (uint64_t)batch.size() * (uint64_t)req_opt.decode_steps;
                const uint64_t client_continuation_tokens = req_opt.decode_steps > 1
                    ? (uint64_t)batch.size() * (uint64_t)(req_opt.decode_steps - 1)
                    : 0ull;
                generation_batches++;
                generation_requests += (uint64_t)batch.size();
                if (batch.size() > 1) coalesced_requests += (uint64_t)batch.size();
                next_position = std::max(next_position,
                                         first_req.cache_position +
                                             (uint64_t)max_prompt_prefill_steps +
                                             (uint64_t)req_opt.decode_steps);
                total_prompt_tokens += client_prompt_tokens;
                total_generated_tokens += client_generated_tokens;
                total_continuation_tokens += client_continuation_tokens;
                total_decode_ms += result.total_decode_ms;
                total_wall_ms += result.total_wall_ms;
                total_continuation_decode_ms += result.continuation_decode_ms;
                total_continuation_wall_ms += result.continuation_wall_ms;
                total_ep_ms += result.total_ep_ms;
                total_dense_ms += result.total_dense_ms;
                total_compose_ms += result.total_compose_ms;
                total_compose_reduce_ms += result.total_compose_reduce_ms;
                total_compose_copy_ms += result.total_compose_copy_ms;
                total_compose_final_ms += result.total_compose_final_ms;
                last = result;
                for (size_t i = 0; i < batch.size(); ++i) {
                    const uint64_t request_generated = (uint64_t)req_opt.decode_steps;
                    const uint64_t request_continuation = req_opt.decode_steps > 1
                        ? (uint64_t)(req_opt.decode_steps - 1)
                        : 0ull;
                    const bool have_output_head =
                        result.diagnostic_output_head &&
                        batch[i].cache_slot >= 0 &&
                        (size_t)batch[i].cache_slot < result.selected_tokens.size() &&
                        (size_t)batch[i].cache_slot < result.selected_logits.size();
                    const uint32_t selected_token = have_output_head
                        ? result.selected_tokens[(size_t)batch[i].cache_slot]
                        : UINT32_MAX;
                    const float selected_logit = have_output_head
                        ? result.selected_logits[(size_t)batch[i].cache_slot]
                        : 0.0f;
                    const uint64_t request_prompt_tokens = batch[i].prompt_token_ids.empty()
                        ? 1ull
                        : (uint64_t)batch[i].prompt_token_ids.size();
                    const uint64_t committed_prompt_tokens =
                        assignments[i].hit ? 0ull : request_prompt_tokens;
                    sessions.commit(assignments[i],
                                    committed_prompt_tokens,
                                    request_generated,
                                    batch[i].prompt_prefill_tokens + request_generated,
                                    batch[i].generated_token_ids);
                    const TpEpHttpSessionSlot *slot_state = nullptr;
                    if (batch[i].cache_slot >= 0 &&
                        batch[i].cache_slot < (int)sessions.slots.size()) {
                        slot_state = &sessions.slots[(size_t)batch[i].cache_slot];
                    }
                    const size_t slot_prompt_token_ids = slot_state
                        ? slot_state->prompt_token_ids.size()
                        : 0u;
                    const size_t slot_generated_token_ids = slot_state
                        ? slot_state->generated_token_ids.size()
                        : 0u;
                    const uint32_t slot_last_selected = slot_state
                        ? slot_state->last_selected_token
                        : UINT32_MAX;
                    const std::string escaped_key = http_json_escape(batch[i].cache_key);
                    const std::string escaped_evicted = http_json_escape(batch[i].evicted_key);
                    const std::string generated_sequence =
                        http_json_uint_array(batch[i].generated_token_ids);
                    const std::string generated_text =
                        decode_token_text(tokenizer.engine, batch[i].generated_token_ids);
                    const std::string escaped_generated_text =
                        http_json_escape(generated_text);
                    char meta[10240];
                    std::snprintf(meta, sizeof(meta),
                                  "\"backend\":\"tp_ep_resident\","
                                  "\"diagnostic\":true,"
                                  "\"diagnostic_note\":\"tokenized prompt prefill and per-step feedback are wired; tokenizer text is not fully wired yet\","
                                  "\"diagnostic_output_head\":%d,"
                                  "\"diagnostic_output_head_proxy_hc\":%d,"
                                  "\"token_input_seed\":%d,"
                                  "\"tokenizer_ready\":%d,"
                                  "\"generated_text\":\"%s\","
                                  "\"decode_input_token\":%u,"
                                  "\"prompt_prefill_tokens\":%llu,"
                                  "\"generated_token_ids\":%zu,"
                                  "\"generated_token_sequence\":%s,"
                                  "\"selected_token\":%u,"
                                  "\"selected_logit\":%.9f,"
                                  "\"output_head_ms\":%.6f,"
                                  "\"output_head_gather_ms\":%.6f,"
                                  "\"output_head_prep_ms\":%.6f,"
                                  "\"output_head_broadcast_ms\":%.6f,"
                                  "\"output_head_projection_ms\":%.6f,"
                                  "\"output_head_top1_ms\":%.6f,"
                                  "\"coalesced_batch_id\":%llu,"
                                  "\"coalesced_batch_size\":%zu,"
                                  "\"coalesced_slot_index\":%zu,"
                                  "\"cache_key\":\"%s\","
                                  "\"cache_key_explicit\":%d,"
                                  "\"cache_hit\":%d,"
                                  "\"cache_prompt_match\":%d,"
                                  "\"cache_prompt_fingerprint\":%llu,"
                                  "\"cache_slot\":%d,"
                                  "\"cache_pos_in\":%llu,"
                                  "\"cache_pos_out\":%llu,"
                                  "\"slot_position\":%llu,"
                                  "\"cache_evicted\":%d,"
                                  "\"cache_evicted_key\":\"%s\","
                                  "\"request_prompt_token_ids\":%zu,"
                                  "\"slot_prompt_token_ids\":%zu,"
                                  "\"slot_generated_token_ids\":%zu,"
                                  "\"slot_last_selected_token\":%u,"
                                  "\"microbatch_wait_us\":%d,"
                                  "\"kv_runtime_resident\":%d,"
                                  "\"kv_all_slots_gate\":%d,"
                                  "\"hc_persist_state_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_raw_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_compressed_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_indexer_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_history_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_skip_current_load_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_skip_raw_store_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_skip_compressed_store_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_skip_indexer_store_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_quiet_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_batch_rows_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_stream_sync_gate\":%d,"
                                  "\"fp8_e5m2_kv_gate\":%d,"
                                  "\"router_hash_fast_gate\":%d,"
                                  "\"route_plan_async_upload_gate\":%d,"
                                  "\"decode_slots\":%d,"
                                  "\"prompt_tokens\":%llu,"
                                  "\"generated_tokens\":%llu,"
                                  "\"continuation_tokens\":%llu,"
                                  "\"batch_prompt_tokens\":%llu,"
                                  "\"batch_generated_tokens\":%llu,"
                                  "\"batch_continuation_tokens\":%llu,"
                                  "\"decode_generated_tokens\":%llu,"
                                  "\"decode_continuation_tokens\":%llu,"
                                  "\"tokens_per_request\":%d,\"slots\":%d,\"ctx\":262144,"
                                  "\"token_match\":1,\"token_mismatch\":0,"
                                  "\"timing_ms\":{\"first_token_decode\":%.6f,"
                                  "\"continuation_decode\":%.6f,"
                                  "\"first_token_wall\":%.6f,"
                                  "\"continuation_wall\":%.6f,"
                                  "\"total_decode\":%.6f,\"total_wall\":%.6f,"
                                  "\"ep\":%.6f,\"dense\":%.6f,"
                                  "\"compose\":%.6f,\"compose_reduce\":%.6f,"
                                  "\"compose_copy\":%.6f,\"compose_final\":%.6f,"
                                  "\"generated_tokens_per_second\":%.6f,"
                                  "\"continuation_tokens_per_second\":%.6f,"
                                  "\"generated_tokens_per_second_decode\":%.6f,"
                                  "\"continuation_tokens_per_second_decode\":%.6f},"
                                  "\"checksum\":%llu",
                                  have_output_head ? 1 : 0,
                                  result.diagnostic_output_head_proxy_hc ? 1 : 0,
                                  result.token_input_seed ? 1 : 0,
                                  tokenizer.initialized ? 1 : 0,
                                  escaped_generated_text.c_str(),
                                  batch[i].decode_input_token,
                                  (unsigned long long)batch[i].prompt_prefill_tokens,
                                  batch[i].generated_token_ids.size(),
                                  generated_sequence.c_str(),
                                  selected_token,
                                  selected_logit,
                                  result.output_head_ms,
                                  result.output_head_gather_ms,
                                  result.output_head_prep_ms,
                                  result.output_head_broadcast_ms,
                                  result.output_head_projection_ms,
                                  result.output_head_top1_ms,
                                  (unsigned long long)batch_id,
                                  batch.size(),
                                  i,
                                  escaped_key.c_str(),
                                  batch[i].cache_key_explicit ? 1 : 0,
                                  batch[i].cache_hit ? 1 : 0,
                                  batch[i].cache_prompt_match ? 1 : 0,
                                  (unsigned long long)batch[i].prompt_fingerprint,
                                  batch[i].cache_slot,
                                  (unsigned long long)assignments[i].pos_in,
                                  (unsigned long long)(assignments[i].pos_in +
                                      batch[i].prompt_prefill_tokens + request_generated),
                                  (unsigned long long)(assignments[i].pos_in +
                                      batch[i].prompt_prefill_tokens + request_generated),
                                  batch[i].cache_evicted ? 1 : 0,
                                  escaped_evicted.c_str(),
                                  batch[i].prompt_token_ids.size(),
                                  slot_prompt_token_ids,
                                  slot_generated_token_ids,
                                  slot_last_selected,
                                  base_opt.microbatch_wait_us,
                                  shared_tp_runtime && shared_tp_runtime->initialized ? 1 : 0,
                                  req_opt.tp_kv_all_slots_gate ? 1 : 0,
                                  req_opt.tp_hc_persist_state_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_raw_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_compressed_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_indexer_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_history_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_skip_current_load_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_skip_raw_store_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_skip_compressed_store_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_skip_indexer_store_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_quiet_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_batch_rows_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_stream_sync_gate ? 1 : 0,
                                  req_opt.fp8_e5m2_kv_gate ? 1 : 0,
                                  req_opt.router_hash_fast_gate ? 1 : 0,
                                  req_opt.route_plan_async_upload_gate ? 1 : 0,
                                  req_opt.slots,
                                  (unsigned long long)request_prompt_tokens,
                                  (unsigned long long)request_generated,
                                  (unsigned long long)request_continuation,
                                  (unsigned long long)client_prompt_tokens,
                                  (unsigned long long)client_generated_tokens,
                                  (unsigned long long)client_continuation_tokens,
                                  (unsigned long long)result.generated_tokens,
                                  (unsigned long long)result.continuation_tokens,
                                  req_opt.decode_steps, req_opt.slots,
                                  result.first_token_decode_ms,
                                  result.continuation_decode_ms,
                                  result.first_token_wall_ms,
                                  result.continuation_wall_ms,
                                  result.total_decode_ms,
                                  result.total_wall_ms,
                                  result.total_ep_ms,
                                  result.total_dense_ms,
                                  result.total_compose_ms,
                                  result.total_compose_reduce_ms,
                                  result.total_compose_copy_ms,
                                  result.total_compose_final_ms,
                                  result.aggregate_generated_tok_s_wall,
                                  result.aggregate_continuation_tok_s_wall,
                                  result.aggregate_generated_tok_s_decode,
                                  result.aggregate_continuation_tok_s_decode,
                                  (unsigned long long)result.checksum);
                    char out[16384];
                    if (http_is_chat_completion_post(batch[i])) {
                        std::snprintf(out, sizeof(out),
                                      "{\"id\":\"chatcmpl-ds4-v100-diagnostic-%llu-%zu\","
                                      "\"object\":\"chat.completion\","
                                      "\"created\":%llu,"
                                      "\"model\":\"ds4-v100-tp-ep-diagnostic\","
                                      "\"choices\":[{\"index\":0,"
                                      "\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                                      "\"logprobs\":null,"
                                      "\"finish_reason\":\"length\","
                                      "\"token_ids\":%s}],"
                                      "\"usage\":{\"prompt_tokens\":%llu,"
                                      "\"completion_tokens\":%llu,"
                                      "\"total_tokens\":%llu},"
                                      "\"ds4_v100\":{%s}}\n",
                                      (unsigned long long)batch_id,
                                      i,
                                      http_epoch_seconds(),
                                      escaped_generated_text.c_str(),
                                      generated_sequence.c_str(),
                                      (unsigned long long)request_prompt_tokens,
                                      (unsigned long long)request_generated,
                                      (unsigned long long)(request_generated + request_prompt_tokens),
                                      meta);
                    } else if (http_is_completion_post(batch[i])) {
                        std::snprintf(out, sizeof(out),
                                      "{\"id\":\"cmpl-ds4-v100-diagnostic-%llu-%zu\","
                                      "\"object\":\"text_completion\","
                                      "\"created\":%llu,"
                                      "\"model\":\"ds4-v100-tp-ep-diagnostic\","
                                      "\"choices\":[{\"text\":\"%s\","
                                      "\"index\":0,\"logprobs\":null,"
                                      "\"finish_reason\":\"length\","
                                      "\"token_ids\":%s}],"
                                      "\"usage\":{\"prompt_tokens\":%llu,"
                                      "\"completion_tokens\":%llu,"
                                      "\"total_tokens\":%llu},"
                                      "\"ds4_v100\":{%s}}\n",
                                      (unsigned long long)batch_id,
                                      i,
                                      http_epoch_seconds(),
                                      escaped_generated_text.c_str(),
                                      generated_sequence.c_str(),
                                      (unsigned long long)request_prompt_tokens,
                                      (unsigned long long)request_generated,
                                      (unsigned long long)(request_generated + request_prompt_tokens),
                                      meta);
                    } else {
                        std::snprintf(out, sizeof(out), "{%s}\n", meta);
                    }
                    http_write_json(batch[i].fd, 200, out);
                    close(batch[i].fd);
                }
            }
        } else {
            http_write_json(fd, 404, "{\"error\":\"not_found\"}\n");
            close(fd);
        }
    }
    close(listen_fd);
    close_tokenizer_runtime(&tokenizer);
    return 0;
}

int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        usage(argv[0]);
        return 2;
    }
    if (opt.serving_bench) {
        opt.skip_decode_checksum = true;
    }
    if (opt.token_major_all_layers && opt.all_layers && !opt.tp_runtime_explicit) {
        opt.share_tp_runtime = true;
    }
    if (report_vram_checkpoint(opt, "startup") != 0) {
        return 14;
    }

    if (opt.output_head_gate) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0) {
            std::fprintf(stderr, "output-head gate contract parse failed bad_rows=%llu\n",
                         (unsigned long long)all_stats.bad_rows);
            return 2;
        }
        OutputHeadGateStats output_head_stats;
        return run_output_head_gate(opt, all_rows, &output_head_stats);
    }

    if (opt.output_head_resident_gate) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0) {
            std::fprintf(stderr, "resident output-head gate contract parse failed bad_rows=%llu\n",
                         (unsigned long long)all_stats.bad_rows);
            return 2;
        }
        OutputHeadResidentGateStats output_head_stats;
        return run_output_head_resident_gate(opt, all_rows, &output_head_stats);
    }

    if (!opt.all_layers) {
        return run_layer(opt, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    DenseF16Cache all_layer_dense_f16_cache;
    DenseF16Cache *shared_dense_f16_cache = nullptr;
    if (opt.dense_f16_cache_compose) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0) {
            std::fprintf(stderr, "all-layer contract parse failed bad_rows=%llu\n",
                         (unsigned long long)all_stats.bad_rows);
            return 2;
        }
        const auto cache_start = std::chrono::steady_clock::now();
        if (prepare_dense_f16_cache(opt, all_rows, &all_layer_dense_f16_cache) != 0) {
            std::fprintf(stderr, "all-layer dense f16 cache prepare failed\n");
            return 4;
        }
        const auto cache_stop = std::chrono::steady_clock::now();
        const double cache_ms =
            std::chrono::duration<double, std::milli>(cache_stop - cache_start).count();
        shared_dense_f16_cache = &all_layer_dense_f16_cache;
        std::printf("tp_ep_all_layer_dense_f16_cache\trows\t%llu\t"
                    "source_bytes\t%llu\tcache_bytes\t%llu\t"
                    "cache_aligned_bytes\t%llu\tmax_temp_bytes\t%llu\t"
                    "cache_ms\t%.6f\tPASS\n",
                    (unsigned long long)all_layer_dense_f16_cache.rows,
                    (unsigned long long)all_layer_dense_f16_cache.source_bytes,
                    (unsigned long long)all_layer_dense_f16_cache.cache_bytes,
                    (unsigned long long)all_layer_dense_f16_cache.cache_aligned_bytes,
                    (unsigned long long)all_layer_dense_f16_cache.max_temp_bytes,
                    cache_ms);
        if (report_vram_checkpoint(opt, "after_dense_f16_cache") != 0) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            return 14;
        }
    }

    SharedApi shared_api;
    if (open_shared_api(opt, &shared_api) != 0) {
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 6;
    }
    std::printf("tp_ep_all_layer_turbomind_api_shared\tdevices\t%d\tPASS\n", kGpus);

    SharedRankBuffers shared_rank_buffers;
    if (open_shared_rank_buffers(opt, &shared_rank_buffers) != 0) {
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 7;
    }
    std::printf("tp_ep_all_layer_rank_buffers_shared\tdevices\t%d\tcore_bytes\t%llu\tPASS\n",
                kGpus, (unsigned long long)shared_rank_buffers.core_bytes);
    if (report_vram_checkpoint(opt, "after_rank_buffers") != 0) {
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }
    if (report_nccl_vram_checkpoint(opt, "nccl_after_rank_buffers") != 0) {
        std::fprintf(stderr,
                     "tp_ep_nccl_vram_admission_failed label=nccl_after_rank_buffers "
                     "min_free_mib=%llu\n",
                     (unsigned long long)opt.nccl_min_free_mib);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }

    SharedTpRuntime shared_tp_runtime;
    if (opt.share_tp_runtime && open_shared_tp_runtime(opt, &shared_tp_runtime) != 0) {
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 8;
    }
    if (shared_tp_runtime.initialized) {
        std::printf("tp_ep_all_layer_tp_runtime_shared\tdevices\t%d\tslots\t%d\tctx\t262144\t"
                    "kv_bytes_per_gpu\t%llu\tcomp_state_bytes_per_gpu\t%llu\t"
                    "scratch_bytes_per_gpu\t%llu\ttotal_bytes_per_gpu\t%llu\tPASS\n",
                    kGpus, opt.slots,
                    (unsigned long long)shared_tp_runtime.report.gpu[0].kv_bytes,
                    (unsigned long long)shared_tp_runtime.report.gpu[0].comp_state_bytes,
                    (unsigned long long)shared_tp_runtime.report.gpu[0].scratch_bytes,
                    (unsigned long long)shared_tp_runtime.report.gpu[0].total_bytes);
    } else {
        std::printf("tp_ep_all_layer_tp_runtime_shared\tdevices\t%d\tslots\t%d\tctx\t262144\t"
                    "mode\tlocal_per_layer\tPASS\n", kGpus, opt.slots);
    }
    if (report_vram_checkpoint(opt, "after_tp_runtime") != 0) {
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }

    SharedExpertBindings shared_expert_bindings;
    if (opt.share_expert_bindings &&
        open_shared_expert_bindings(opt, &shared_expert_bindings) != 0) {
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 9;
    }
    if (shared_expert_bindings.initialized) {
        std::printf("tp_ep_all_layer_expert_bindings_shared\tlayers\t43\tdevices\t%d\t"
                    "bytes\t%llu\tbytes_per_gpu\t%llu\tPASS\n",
                    kGpus,
                    (unsigned long long)shared_expert_bindings.bytes,
                    (unsigned long long)(shared_expert_bindings.bytes / kGpus));
    } else {
        std::printf("tp_ep_all_layer_expert_bindings_shared\tlayers\t43\tdevices\t%d\t"
                    "mode\tlocal_per_layer\tPASS\n", kGpus);
    }

    SharedDenseOps shared_dense_ops;
    if (opt.share_dense_ops && open_shared_dense_ops(opt, shared_dense_f16_cache,
                                                     &shared_dense_ops) != 0) {
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 10;
    }
    if (shared_dense_ops.initialized) {
        std::printf("tp_ep_all_layer_dense_ops_shared\tlayers\t43\tdevices\t%d\t"
                    "loaded_bytes\t%llu\tPASS\n",
                    kGpus, (unsigned long long)shared_dense_ops.loaded_bytes);
    } else {
        std::printf("tp_ep_all_layer_dense_ops_shared\tlayers\t43\tdevices\t%d\t"
                    "mode\tlocal_per_layer\tPASS\n", kGpus);
    }
    if (report_vram_checkpoint(opt, "after_dense_ops") != 0) {
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }

    std::vector<ContractRow> resident_rows[43];
    LayerStats resident_stats[43];
    const bool resident_serving_loop =
        opt.serving_bench && opt.token_major_all_layers &&
        shared_tp_runtime.initialized && shared_expert_bindings.initialized &&
        shared_dense_f16_cache != nullptr;
    if (resident_serving_loop) {
        for (int layer = 0; layer < 43; ++layer) {
            if (parse_contract(opt.contract_path, layer, &resident_rows[layer],
                               &resident_stats[layer]) != 0 ||
                resident_stats[layer].bad_rows != 0) {
                std::fprintf(stderr, "resident serving contract parse failed layer=%d bad_rows=%llu\n",
                             layer, (unsigned long long)resident_stats[layer].bad_rows);
                free_shared_dense_ops(&shared_dense_ops, opt);
                close_shared_expert_bindings(&shared_expert_bindings);
                close_shared_tp_runtime(&shared_tp_runtime);
                close_shared_rank_buffers(&shared_rank_buffers);
                close_shared_api(&shared_api);
                if (shared_dense_f16_cache) {
                    free_dense_f16_cache(all_layer_dense_f16_cache, opt);
                }
                return 11;
            }
        }
        std::printf("tp_ep_resident_serving_loop\tlayers\t43\tmode\tdirect_decode\tPASS\n");
    }

    SharedHcControls shared_hc_controls;
    if (opt.tp_hc_final_expand_gate) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0 ||
            open_shared_hc_controls(opt, all_rows, &shared_hc_controls) != 0) {
            std::fprintf(stderr, "tp_ep HC final-expand controls open failed\n");
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 12;
        }
        std::printf("tp_ep_hc_final_expand_shared\tlayers\t43\tslots\t%d\t"
                    "control_bytes\t%llu\tPASS\n",
                    opt.slots, (unsigned long long)shared_hc_controls.control_bytes);
        if (report_vram_checkpoint(opt, "after_hc_controls") != 0) {
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
    }
    SharedHcControls *shared_hc_controls_arg =
        shared_hc_controls.initialized ? &shared_hc_controls : nullptr;

    SharedOutputHead shared_output_head;
    if (opt.diagnostic_output_head && !opt.diagnostic_output_head_lazy_gate) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0 ||
            open_shared_output_head(opt, all_rows, &shared_output_head) != 0) {
            std::fprintf(stderr, "tp_ep diagnostic output-head open failed\n");
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 12;
        }
        std::printf("tp_ep_diagnostic_output_head_shared\tslots\t%d\tvocab\t%d\t"
                    "rows_per_gpu\t%d\toutput_weight_bytes\t%llu\t"
                    "logits_bytes\t%llu\tproxy_hc\t%d\tPASS\n",
                    opt.slots,
                    shared_output_head.vocab,
                    shared_output_head.rows_per_gpu,
                    (unsigned long long)shared_output_head.output_weight_bytes,
                    (unsigned long long)shared_output_head.logits_bytes,
                    opt.tp_hc_final_expand_gate ? 0 : 1);
        if (report_vram_checkpoint(opt, "after_output_head") != 0) {
            close_shared_output_head(opt, &shared_output_head);
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
        if (report_nccl_vram_checkpoint(opt, "nccl_after_output_head") != 0) {
            std::fprintf(stderr,
                         "tp_ep_nccl_vram_admission_failed label=nccl_after_output_head "
                         "min_free_mib=%llu\n",
                         (unsigned long long)opt.nccl_min_free_mib);
            close_shared_output_head(opt, &shared_output_head);
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
    }
    SharedOutputHead *shared_output_head_arg =
        shared_output_head.initialized ? &shared_output_head : nullptr;

    SharedTokenEmbedding shared_token_embedding;
    if (opt.serve_http) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0 ||
            open_shared_token_embedding(opt, all_rows, &shared_token_embedding) != 0) {
            std::fprintf(stderr, "tp_ep token embedding open failed\n");
            close_shared_output_head(opt, &shared_output_head);
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 13;
        }
        std::printf("tp_ep_token_embedding_shared\tslots\t%d\tvocab\t%d\t"
                    "rows_per_gpu\t%d\tweight_bytes\t%llu\tdevice\t%d\tPASS\n",
                    opt.slots,
                    shared_token_embedding.vocab,
                    shared_token_embedding.rows_per_gpu,
                    (unsigned long long)shared_token_embedding.weight_bytes,
                    opt.devices[0]);
        if (report_vram_checkpoint(opt, "after_token_embedding") != 0) {
            close_shared_token_embedding(opt, &shared_token_embedding);
            close_shared_output_head(opt, &shared_output_head);
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
    }
    SharedTokenEmbedding *shared_token_embedding_arg =
        shared_token_embedding.initialized ? &shared_token_embedding : nullptr;

    if (opt.serve_http) {
        int rc = 0;
        if (!resident_serving_loop || !shared_dense_ops.initialized) {
            std::fprintf(stderr, "tp_ep_http requires resident serving loop and shared dense ops\n");
            rc = 13;
        } else {
            if (opt.decode_steps <= 0) opt.decode_steps = 32;
            rc = run_tp_ep_http_server(opt,
                                       shared_dense_f16_cache,
                                       &shared_api,
                                       &shared_rank_buffers,
                                       &shared_tp_runtime,
                                       &shared_expert_bindings,
                                       &shared_dense_ops,
                                       shared_output_head_arg,
                                       shared_hc_controls_arg,
                                       shared_token_embedding_arg,
                                       resident_rows,
                                       resident_stats);
        }
        close_shared_token_embedding(opt, &shared_token_embedding);
        close_shared_output_head(opt, &shared_output_head);
        close_shared_hc_controls(opt, &shared_hc_controls);
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return rc;
    }

    if (opt.token_major_all_layers) {
        const int rc = run_token_major_serving_loop(opt,
                                                    shared_dense_f16_cache,
                                                    &shared_api,
                                                    &shared_rank_buffers,
                                                    &shared_tp_runtime,
                                                    &shared_expert_bindings,
                                                    &shared_dense_ops,
                                                    shared_output_head_arg,
                                                    shared_hc_controls_arg,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    resident_rows,
                                                    resident_stats,
                                                    resident_serving_loop,
                                                    nullptr);
        close_shared_output_head(opt, &shared_output_head);
        close_shared_hc_controls(opt, &shared_hc_controls);
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return rc;
    }

    int pass_layers = 0;
    double sum_decode_ms = 0.0;
    double sum_ep_ms = 0.0;
    double sum_dense_ms = 0.0;
    double sum_compose_ms = 0.0;
    double sum_compose_reduce_ms = 0.0;
    double sum_compose_copy_ms = 0.0;
    double sum_compose_final_ms = 0.0;
    double sum_hc_current_input_ms = 0.0;
    uint64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int layer = 0; layer < 43; ++layer) {
        Options layer_opt = opt;
        layer_opt.layer = layer;
        LayerRunSummary s;
        SharedTpRuntime *tp_runtime_arg =
            shared_tp_runtime.initialized ? &shared_tp_runtime : nullptr;
        const SharedExpertBindings *expert_arg =
            shared_expert_bindings.initialized ? &shared_expert_bindings : nullptr;
        const SharedDenseOps *dense_ops_arg =
            shared_dense_ops.initialized ? &shared_dense_ops : nullptr;
        const int rc = run_layer(layer_opt, &s, shared_dense_f16_cache, &shared_api,
                                 &shared_rank_buffers, tp_runtime_arg, expert_arg,
                                 dense_ops_arg, shared_hc_controls_arg);
        std::printf("tp_ep_all_layer_item\tlayer\t%d\tratio\t%d\t"
                    "total_rows\t%llu\tdense_rows\t%llu\tcontrol_rows\t%llu\t"
                    "expert_rows\t%llu\tkv_rows\t%llu\tcomp_rows\t%llu\t"
                    "decode_ms_per_step\t%.6f\tdecode_slot_step_tok_s\t%.6f\t"
                    "decode_ep_ms_per_step\t%.6f\tdecode_dense_ms_per_step\t%.6f\t"
                    "decode_compose_ms_per_step\t%.6f\t"
                    "decode_compose_reduce_ms_per_step\t%.6f\t"
                    "decode_compose_copy_ms_per_step\t%.6f\t"
                    "decode_compose_final_ms_per_step\t%.6f\t"
                    "decode_hc_current_input_ms_per_step\t%.6f\t"
                    "decode_checksum\t%llu\tdecode_finite_bad\t%d\trc\t%d\t%s\n",
                    s.layer, s.ratio,
                    (unsigned long long)s.total_rows,
                    (unsigned long long)s.dense_rows,
                    (unsigned long long)s.control_rows,
                    (unsigned long long)s.expert_rows,
                    (unsigned long long)s.kv_rows,
                    (unsigned long long)s.comp_rows,
                    s.decode_ms_per_step,
                    s.decode_slot_step_tok_s,
                    s.decode_ep_ms_per_step,
                    s.decode_dense_ms_per_step,
                    s.decode_compose_ms_per_step,
                    s.decode_compose_reduce_ms_per_step,
                    s.decode_compose_copy_ms_per_step,
                    s.decode_compose_final_ms_per_step,
                    s.decode_hc_current_input_ms_per_step,
                    (unsigned long long)s.decode_checksum,
                    s.decode_finite_bad,
                    rc,
                    (rc == 0 && s.pass) ? "PASS" : "FAIL");
        if (rc == 0 && s.pass) {
            pass_layers++;
            sum_decode_ms += s.decode_ms_per_step;
            sum_ep_ms += s.decode_ep_ms_per_step;
            sum_dense_ms += s.decode_dense_ms_per_step;
            sum_compose_ms += s.decode_compose_ms_per_step;
            sum_compose_reduce_ms += s.decode_compose_reduce_ms_per_step;
            sum_compose_copy_ms += s.decode_compose_copy_ms_per_step;
            sum_compose_final_ms += s.decode_compose_final_ms_per_step;
            sum_hc_current_input_ms += s.decode_hc_current_input_ms_per_step;
            checksum ^= s.decode_checksum + (uint64_t)(layer + 1) * 104729ull;
        } else {
            const auto stop = std::chrono::steady_clock::now();
            const double wall_ms =
                std::chrono::duration<double, std::milli>(stop - start).count();
            std::printf("tp_ep_all_layer_scaffold\tlayers\t43\tpass_layers\t%d\t"
                        "failed_layer\t%d\tdescriptor_checks\t%d\tpredecode_probes\t%d\t"
                        "shared_api\t%d\tshared_rank_buffers\t%d\tshared_tp_runtime\t%d\t"
                        "shared_expert_bindings\t%d\t"
                        "shared_dense_ops\t%d\t"
                        "overlap_ep_dense\t%d\tdirect_remote_compose\t%d\t"
                        "source_copy_schedule\t%d\tskip_self_compose_copy\t%d\t"
                        "multi_copy_streams\t%d\t"
                        "wall_ms\t%.6f\tFAIL\n",
                        pass_layers, layer, opt.skip_descriptor_checks ? 0 : 1,
                        opt.skip_predecode_probes ? 0 : 1, shared_api.initialized ? 1 : 0,
                        shared_rank_buffers.initialized ? 1 : 0,
                        shared_tp_runtime.initialized ? 1 : 0,
                        shared_expert_bindings.initialized ? 1 : 0,
                        shared_dense_ops.initialized ? 1 : 0,
                        opt.overlap_ep_dense ? 1 : 0,
                        opt.direct_remote_compose ? 1 : 0,
                        opt.source_copy_schedule ? 1 : 0,
                        opt.skip_self_compose_copy ? 1 : 0,
                        opt.multi_copy_streams ? 1 : 0,
                        wall_ms);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return rc == 0 ? 1 : rc;
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    const double slot_step_tok_s = sum_decode_ms > 0.0
        ? (double)opt.slots * 1000.0 / sum_decode_ms
        : 0.0;
    std::printf("tp_ep_all_layer_scaffold\tlayers\t43\tpass_layers\t%d\t"
                "slots\t%d\tctx\t262144\tdecode_steps_per_layer\t%d\t"
                "descriptor_checks\t%d\tpredecode_probes\t%d\tshared_api\t%d\t"
                "shared_rank_buffers\t%d\tshared_tp_runtime\t%d\t"
                "shared_expert_bindings\t%d\t"
                "shared_dense_ops\t%d\t"
                "overlap_ep_dense\t%d\tdirect_remote_compose\t%d\t"
                "source_copy_schedule\t%d\tskip_self_compose_copy\t%d\t"
                "multi_copy_streams\t%d\t"
                "sum_decode_ms_per_token\t%.6f\tprojected_slot_step_tok_s\t%.6f\t"
                "sum_ep_ms\t%.6f\tsum_dense_ms\t%.6f\tsum_compose_ms\t%.6f\t"
                "sum_compose_reduce_ms\t%.6f\tsum_compose_copy_ms\t%.6f\t"
                "sum_compose_final_ms\t%.6f\t"
                "tp_hc_current_input_gate\t%d\t"
                "tp_hc_current_input_peer_gather\t%d\t"
                "tp_hc_current_input_nccl_allgather\t%d\t"
                "tp_hc_current_input_stream_sync\t%d\t"
                "sum_hc_current_input_ms\t%.6f\t"
                "wall_ms\t%.6f\tchecksum\t%llu\tPASS\n",
                pass_layers, opt.slots, opt.decode_steps,
                opt.skip_descriptor_checks ? 0 : 1,
                opt.skip_predecode_probes ? 0 : 1,
                shared_api.initialized ? 1 : 0,
                shared_rank_buffers.initialized ? 1 : 0,
                shared_tp_runtime.initialized ? 1 : 0,
                shared_expert_bindings.initialized ? 1 : 0,
                shared_dense_ops.initialized ? 1 : 0,
                opt.overlap_ep_dense ? 1 : 0,
                opt.direct_remote_compose ? 1 : 0,
                opt.source_copy_schedule ? 1 : 0,
                opt.skip_self_compose_copy ? 1 : 0,
                opt.multi_copy_streams ? 1 : 0,
                sum_decode_ms, slot_step_tok_s, sum_ep_ms, sum_dense_ms,
                sum_compose_ms, sum_compose_reduce_ms, sum_compose_copy_ms,
                sum_compose_final_ms,
                opt.tp_hc_current_input_gate ? 1 : 0,
                opt.tp_hc_current_input_peer_gather_gate ? 1 : 0,
                opt.tp_hc_current_input_nccl_allgather_gate ? 1 : 0,
                opt.tp_hc_current_input_stream_sync_gate ? 1 : 0,
                sum_hc_current_input_ms,
                wall_ms, (unsigned long long)checksum);
    close_shared_hc_controls(opt, &shared_hc_controls);
    free_shared_dense_ops(&shared_dense_ops, opt);
    close_shared_expert_bindings(&shared_expert_bindings);
    close_shared_tp_runtime(&shared_tp_runtime);
    close_shared_rank_buffers(&shared_rank_buffers);
    close_shared_api(&shared_api);
    if (shared_dense_f16_cache) {
        free_dense_f16_cache(all_layer_dense_f16_cache, opt);
    }
    return 0;
}
