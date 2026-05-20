#ifndef DS4_V100_SCHEDULER_H
#define DS4_V100_SCHEDULER_H

#include "ds4_v100_context.h"
#include "ds4_v100_layer_execute.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_v100_stage_scheduler ds4_v100_stage_scheduler;
typedef struct ds4_v100_stage_scheduler_snapshot ds4_v100_stage_scheduler_snapshot;

enum {
    DS4_V100_SCHED_MAX_SLOTS = 8,
};

typedef struct {
    const char *pack_index_path;
    const void *model_map;
    uint64_t model_size;
    int stage_id;
    uint32_t raw_cap;
    uint32_t raw_window;
    uint32_t attn_comp_cap;
    uint32_t index_comp_cap;
    uint32_t indexer_top_k;
    uint64_t kv_ctx_tokens;
    uint64_t kv_active_slots;
    bool fp8_kv_cache;
    bool suppress_router_readback;
} ds4_v100_stage_scheduler_options;

typedef struct {
    int stage_id;
    int gpu;
    int first_layer;
    int last_layer;
    uint32_t layers_executed;
    uint32_t position;
    uint32_t token;
    uint64_t arena_bytes;
    uint64_t uploaded_tensors;
    uint64_t uploaded_bytes;
    ds4_v100_layer_execute_report last_layer_report;
    double timing_hc_attn_ms;
    double timing_attention_ms;
    double timing_hc_ffn_ms;
    double timing_ffn_ms;
    double timing_hc_final_ms;
    double timing_total_ms;
} ds4_v100_stage_scheduler_report;

typedef struct {
    int stage_id;
    int gpu;
    int layer;
    int kind;
    uint32_t position;
    uint32_t token;
    const ds4_gpu_tensor *hc;
    uint64_t hc_bytes;
    ds4_v100_layer_execute_report layer_report;
} ds4_v100_stage_scheduler_checkpoint;

typedef int (*ds4_v100_stage_scheduler_checkpoint_fn)(
    const ds4_v100_stage_scheduler_checkpoint *checkpoint,
    void *user,
    char *err,
    size_t errlen);

void ds4_v100_stage_scheduler_options_init(ds4_v100_stage_scheduler_options *opts);

int ds4_v100_stage_scheduler_open(ds4_v100_stage_scheduler **out,
                                  const ds4_v100_stage_scheduler_options *opts,
                                  char *err,
                                  size_t errlen);

void ds4_v100_stage_scheduler_close(ds4_v100_stage_scheduler *sched);

int ds4_v100_stage_scheduler_reset(ds4_v100_stage_scheduler *sched,
                                   char *err,
                                   size_t errlen);

int ds4_v100_stage_scheduler_snapshot_create(
    const ds4_v100_stage_scheduler *sched,
    ds4_v100_stage_scheduler_snapshot **out,
    char *err,
    size_t errlen);

int ds4_v100_stage_scheduler_snapshot_restore(
    ds4_v100_stage_scheduler *sched,
    const ds4_v100_stage_scheduler_snapshot *snapshot,
    char *err,
    size_t errlen);

uint64_t ds4_v100_stage_scheduler_snapshot_bytes(
    const ds4_v100_stage_scheduler_snapshot *snapshot);

void ds4_v100_stage_scheduler_snapshot_free(
    ds4_v100_stage_scheduler_snapshot *snapshot);

int ds4_v100_stage_scheduler_decode_token(ds4_v100_stage_scheduler *sched,
                                          uint32_t token,
                                          uint32_t position,
                                          ds4_v100_stage_scheduler_report *report,
                                          char *err,
                                          size_t errlen);

int ds4_v100_stage_scheduler_decode_token_batch(
    ds4_v100_stage_scheduler *sched,
    const uint32_t *tokens,
    const uint32_t *positions,
    uint32_t n_slots,
    ds4_v100_stage_scheduler_report *reports,
    char *err,
    size_t errlen);

int ds4_v100_stage_scheduler_decode_token_slot_span(
    ds4_v100_stage_scheduler *sched,
    uint32_t slot_start,
    const uint32_t *tokens,
    const uint32_t *positions,
    uint32_t n_slots,
    ds4_v100_stage_scheduler_report *reports,
    char *err,
    size_t errlen);

int ds4_v100_stage_scheduler_decode_token_checkpoints(
    ds4_v100_stage_scheduler *sched,
    uint32_t token,
    uint32_t position,
    ds4_v100_stage_scheduler_report *report,
    ds4_v100_stage_scheduler_checkpoint_fn checkpoint_fn,
    void *checkpoint_user,
    char *err,
    size_t errlen);

int ds4_v100_stage_scheduler_handoff(ds4_v100_stage_scheduler *dst,
                                     const ds4_v100_stage_scheduler *src,
                                     char *err,
                                     size_t errlen);

int ds4_v100_stage_scheduler_handoff_batch(ds4_v100_stage_scheduler *dst,
                                           const ds4_v100_stage_scheduler *src,
                                           uint32_t n_slots,
                                           char *err,
                                           size_t errlen);

int ds4_v100_stage_scheduler_handoff_slot_span(ds4_v100_stage_scheduler *dst,
                                               const ds4_v100_stage_scheduler *src,
                                               uint32_t slot_start,
                                               uint32_t n_slots,
                                               char *err,
                                               size_t errlen);

int ds4_v100_stage_scheduler_handoff_slot_span_async(ds4_v100_stage_scheduler *dst,
                                                     const ds4_v100_stage_scheduler *src,
                                                     uint32_t slot_start,
                                                     uint32_t n_slots,
                                                     char *err,
                                                     size_t errlen);

int ds4_v100_stage_scheduler_decode_hc(ds4_v100_stage_scheduler *sched,
                                       uint32_t token,
                                       uint32_t position,
                                       ds4_v100_stage_scheduler_report *report,
                                       char *err,
                                       size_t errlen);

int ds4_v100_stage_scheduler_decode_hc_batch(
    ds4_v100_stage_scheduler *sched,
    const uint32_t *tokens,
    const uint32_t *positions,
    uint32_t n_slots,
    ds4_v100_stage_scheduler_report *reports,
    char *err,
    size_t errlen);

int ds4_v100_stage_scheduler_decode_hc_slot_span(
    ds4_v100_stage_scheduler *sched,
    uint32_t slot_start,
    const uint32_t *tokens,
    const uint32_t *positions,
    uint32_t n_slots,
    ds4_v100_stage_scheduler_report *reports,
    char *err,
    size_t errlen);

int ds4_v100_stage_scheduler_decode_hc_checkpoints(
    ds4_v100_stage_scheduler *sched,
    uint32_t token,
    uint32_t position,
    ds4_v100_stage_scheduler_report *report,
    ds4_v100_stage_scheduler_checkpoint_fn checkpoint_fn,
    void *checkpoint_user,
    char *err,
    size_t errlen);

int ds4_v100_stage_scheduler_read_hc(const ds4_v100_stage_scheduler *sched,
                                     void *dst,
                                     uint64_t bytes);

int ds4_v100_stage_scheduler_read_hc_slot(const ds4_v100_stage_scheduler *sched,
                                          uint32_t slot,
                                          void *dst,
                                          uint64_t bytes);

int ds4_v100_stage_scheduler_read_token_embedding_f32(
    const ds4_v100_stage_scheduler *sched,
    uint32_t token,
    float *dst,
    uint64_t dst_values,
    char *err,
    size_t errlen);

int ds4_v100_stage_scheduler_write_hc(ds4_v100_stage_scheduler *sched,
                                      const void *src,
                                      uint64_t bytes);

int ds4_v100_stage_scheduler_select_topk(ds4_v100_stage_scheduler *sched,
                                         uint32_t *tokens,
                                         float *logits,
                                         uint32_t k,
                                         char *err,
                                         size_t errlen);

int ds4_v100_stage_scheduler_select_topk_slot(ds4_v100_stage_scheduler *sched,
                                              uint32_t slot,
                                              uint32_t *tokens,
                                              float *logits,
                                              uint32_t k,
                                              char *err,
                                              size_t errlen);

int ds4_v100_stage_scheduler_select_token(ds4_v100_stage_scheduler *sched,
                                          uint32_t *token,
                                          float *logit,
                                          char *err,
                                          size_t errlen);

int ds4_v100_stage_scheduler_select_token_slot(ds4_v100_stage_scheduler *sched,
                                               uint32_t slot,
                                               uint32_t *token,
                                               float *logit,
                                               char *err,
                                               size_t errlen);

#ifdef __cplusplus
}
#endif

#endif
