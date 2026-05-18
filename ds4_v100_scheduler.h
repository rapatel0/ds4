#ifndef DS4_V100_SCHEDULER_H
#define DS4_V100_SCHEDULER_H

#include "ds4_v100_context.h"
#include "ds4_v100_layer_execute.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_v100_stage_scheduler ds4_v100_stage_scheduler;

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
} ds4_v100_stage_scheduler_report;

void ds4_v100_stage_scheduler_options_init(ds4_v100_stage_scheduler_options *opts);

int ds4_v100_stage_scheduler_open(ds4_v100_stage_scheduler **out,
                                  const ds4_v100_stage_scheduler_options *opts,
                                  char *err,
                                  size_t errlen);

void ds4_v100_stage_scheduler_close(ds4_v100_stage_scheduler *sched);

int ds4_v100_stage_scheduler_decode_token(ds4_v100_stage_scheduler *sched,
                                          uint32_t token,
                                          uint32_t position,
                                          ds4_v100_stage_scheduler_report *report,
                                          char *err,
                                          size_t errlen);

int ds4_v100_stage_scheduler_handoff(ds4_v100_stage_scheduler *dst,
                                     const ds4_v100_stage_scheduler *src,
                                     char *err,
                                     size_t errlen);

int ds4_v100_stage_scheduler_decode_hc(ds4_v100_stage_scheduler *sched,
                                       uint32_t token,
                                       uint32_t position,
                                       ds4_v100_stage_scheduler_report *report,
                                       char *err,
                                       size_t errlen);

int ds4_v100_stage_scheduler_read_hc(const ds4_v100_stage_scheduler *sched,
                                     void *dst,
                                     uint64_t bytes);

int ds4_v100_stage_scheduler_select_token(ds4_v100_stage_scheduler *sched,
                                          uint32_t *token,
                                          float *logit,
                                          char *err,
                                          size_t errlen);

#ifdef __cplusplus
}
#endif

#endif
