#ifndef DS4_V100_CONTEXT_H
#define DS4_V100_CONTEXT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_V100_EXPECTED_GPUS 8
#define DS4_V100_N_LAYERS 43
#define DS4_V100_HC_ROWS 4
#define DS4_V100_HC_COLS 4096
#define DS4_V100_RELAY_BUFFERS 2
#define DS4_V100_SWA_ROWS 128
#define DS4_V100_HEAD_DIM 512
#define DS4_V100_INDEXER_HEAD_DIM 128
#define DS4_V100_MIN_VRAM_BYTES (31ull * 1024ull * 1024ull * 1024ull)
#define DS4_V100_DEFAULT_RESERVE_BYTES (2ull * 1024ull * 1024ull * 1024ull)

typedef struct ds4_v100_context ds4_v100_context;

typedef enum {
    DS4_V100_INIT_PROBE_ONLY = 0,
    DS4_V100_INIT_USE_EXISTING_ARENAS = 1,
    DS4_V100_INIT_FULL_RESIDENT = 2,
} ds4_v100_init_mode;

typedef enum {
    DS4_V100_SOURCE_UNKNOWN = 0,
    DS4_V100_SOURCE_BF16,
    DS4_V100_SOURCE_F32,
    DS4_V100_SOURCE_I32,
    DS4_V100_SOURCE_F8_E4M3_B128,
    DS4_V100_SOURCE_MXFP4,
    DS4_V100_SOURCE_FP4,
} ds4_v100_source_dtype;

typedef enum {
    DS4_V100_FAMILY_UNKNOWN = 0,
    DS4_V100_FAMILY_BF16_GLOBAL,
    DS4_V100_FAMILY_F32_CONTROL,
    DS4_V100_FAMILY_FP8_DENSE,
    DS4_V100_FAMILY_MXFP4_EXPERT,
    DS4_V100_FAMILY_HC_CONTROL,
    DS4_V100_FAMILY_KV_CACHE,
} ds4_v100_tensor_family;

typedef enum {
    DS4_V100_EXEC_F32_CONTROL = 0,
    DS4_V100_EXEC_F16_HMMA,
    DS4_V100_EXEC_LOWBIT_KERNEL,
    DS4_V100_EXEC_DIAGNOSTIC_ONLY,
    DS4_V100_EXEC_UNSUPPORTED,
    DS4_V100_EXEC_COUNT,
} ds4_v100_exec_kind;

typedef enum {
    DS4_V100_LAYER_SWA_ONLY = 0,
    DS4_V100_LAYER_RATIO_4 = 1,
    DS4_V100_LAYER_RATIO_128 = 2,
} ds4_v100_layer_class;

typedef struct {
    uint64_t raw_swa_bytes;
    uint64_t compressed_attn_bytes;
    uint64_t indexer_kv_bytes;
    uint64_t compression_state_bytes;
    uint64_t total_bytes;
} ds4_v100_kv_budget;

typedef struct {
    uint64_t raw_swa_offset;
    uint64_t raw_swa_bytes;
    uint64_t compressed_attn_offset;
    uint64_t compressed_attn_bytes;
    uint64_t indexer_kv_offset;
    uint64_t indexer_kv_bytes;
    uint64_t compression_state_offset;
    uint64_t compression_state_bytes;
    uint64_t total_bytes;
} ds4_v100_kv_arena_plan;

typedef struct {
    ds4_v100_source_dtype source_dtype;
    ds4_v100_tensor_family family;
    ds4_v100_exec_kind exec_kind;
    const char *conversion_stub;
    const char *forbidden_claim;
} ds4_v100_policy;

typedef struct {
    int visible_id;
    int cc_major;
    int cc_minor;
    uint64_t total_global_mem;
    char pci_bus_id[32];
    char uuid[64];
    bool peer_access[DS4_V100_EXPECTED_GPUS];
} ds4_v100_device_fact;

typedef struct {
    int stage_id;
    int gpu;
    int layer_begin;
    int layer_end;
    bool owns_token_embedding;
    bool owns_output_head;
    uint64_t tensor_count;
    uint64_t arena_bytes;
    uint64_t scratch_bytes;
    uint64_t relay_f16_bytes;
    uint64_t relay_f32_debug_bytes;
    uint64_t planned_kv_bytes;
    ds4_v100_kv_arena_plan kv_arena;
    uint64_t kv_raw_swa_bytes;
    uint64_t kv_compressed_attn_bytes;
    uint64_t kv_indexer_bytes;
    uint64_t kv_compression_state_bytes;
    uint64_t output_head_reserve_bytes;
    uint64_t mtp_reserve_bytes;
    uint64_t reserve_bytes;
    uint64_t device_total_bytes;
} ds4_v100_stage_info;

typedef struct {
    int layer_id;
    int stage_id;
    ds4_v100_layer_class layer_class;
    ds4_v100_kv_budget kv_budget;
    uint64_t tensor_count;
    bool has_f32_control;
    bool has_fp8_dense;
    bool has_mxfp4_expert;
    bool has_hc_control;
} ds4_v100_layer_info;

typedef struct {
    const char *pack_index_path;
    int expected_gpus;
    ds4_v100_init_mode mode;
    bool require_production_topology;
    bool enable_f32_debug_relay;
    uint64_t scratch_bytes_per_gpu;
    uint64_t relay_max_active_slots;
    uint64_t reserve_bytes_per_gpu;
    uint64_t planned_kv_bytes_per_gpu;
    uint64_t kv_ctx_tokens;
    uint64_t kv_active_slots;
    uint64_t output_head_reserve_bytes;
    uint64_t mtp_reserve_bytes;
    const ds4_v100_device_fact *device_facts;
    int n_device_facts;
} ds4_v100_context_options;

void ds4_v100_context_options_init(ds4_v100_context_options *opts);

ds4_v100_source_dtype ds4_v100_source_dtype_parse(const char *s);
const char *ds4_v100_source_dtype_name(ds4_v100_source_dtype dtype);
ds4_v100_tensor_family ds4_v100_tensor_family_infer(const char *source_dtype,
                                                    const char *runtime_layout,
                                                    const char *kernel_family);
const char *ds4_v100_tensor_family_name(ds4_v100_tensor_family family);
const char *ds4_v100_exec_kind_name(ds4_v100_exec_kind kind);
ds4_v100_layer_class ds4_v100_layer_class_for_layer(int layer_id);
const char *ds4_v100_layer_class_name(ds4_v100_layer_class layer_class);
ds4_v100_kv_budget ds4_v100_kv_budget_for_layer(int layer_id,
                                                uint64_t ctx_tokens,
                                                uint64_t active_slots);

int ds4_v100_classify_or_die(const char *source_dtype,
                             const char *runtime_layout,
                             const char *kernel_family,
                             ds4_v100_policy *out,
                             char *err,
                             size_t errlen);

int ds4_v100_stage_for_layer(int layer_id);
int ds4_v100_context_open(ds4_v100_context **out,
                          const ds4_v100_context_options *opts,
                          char *err,
                          size_t errlen);
void ds4_v100_context_close(ds4_v100_context *ctx);

int ds4_v100_context_stage_count(const ds4_v100_context *ctx);
const ds4_v100_stage_info *ds4_v100_context_stage(const ds4_v100_context *ctx,
                                                  int stage_id);
const ds4_v100_layer_info *ds4_v100_context_layer(const ds4_v100_context *ctx,
                                                  int layer_id);
uint64_t ds4_v100_context_tensor_count(const ds4_v100_context *ctx);
uint64_t ds4_v100_context_exec_count(const ds4_v100_context *ctx,
                                     ds4_v100_exec_kind kind);
bool ds4_v100_context_has_token_embedding(const ds4_v100_context *ctx);
int ds4_v100_context_validate_layer_skeleton(const ds4_v100_context *ctx,
                                             FILE *report,
                                             char *err,
                                             size_t errlen);
void ds4_v100_context_print_report(const ds4_v100_context *ctx, FILE *fp);

typedef struct ds4_v100_cuda_context ds4_v100_cuda_context;

typedef enum {
    DS4_V100_RELAY_F16 = 0,
    DS4_V100_RELAY_F32_DEBUG = 1,
} ds4_v100_relay_dtype;

int ds4_v100_cuda_collect_device_facts(ds4_v100_device_fact *facts,
                                       int fact_cap,
                                       int *out_count,
                                       char *err,
                                       size_t errlen);
int ds4_v100_cuda_context_open(ds4_v100_cuda_context **out,
                               const ds4_v100_context_options *opts,
                               char *err,
                               size_t errlen);
void ds4_v100_cuda_context_close(ds4_v100_cuda_context *ctx);
int ds4_v100_cuda_context_relay_smoke(ds4_v100_cuda_context *ctx,
                                      int src_stage,
                                      int dst_stage,
                                      ds4_v100_relay_dtype dtype,
                                      uint64_t active_slots,
                                      char *err,
                                      size_t errlen);

#ifdef __cplusplus
}
#endif

#endif
