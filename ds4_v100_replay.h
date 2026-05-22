#ifndef DS4_V100_REPLAY_H
#define DS4_V100_REPLAY_H

#include "ds4.h"
#include "ds4_v100_scheduler.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_v100_replay ds4_v100_replay;

typedef enum {
    DS4_V100_REPLAY_ASYNC_PIPELINE_OFF = 0,
    DS4_V100_REPLAY_ASYNC_PIPELINE_PERSISTENT = 1,
    DS4_V100_REPLAY_ASYNC_PIPELINE_PER_STEP = 2,
    DS4_V100_REPLAY_ASYNC_PIPELINE_MAILBOX = 3,
} ds4_v100_replay_async_pipeline_mode;

typedef struct {
    const char *model_path;
    const char *pack_index_path;
    const char *turbomind_pack_index_path;
    const char *shard_dir;
    uint64_t kv_ctx_tokens;
    uint32_t attn_comp_cap;
    uint32_t index_comp_cap;
    uint32_t indexer_top_k;
    uint64_t kv_active_slots;
    bool fp8_kv_cache;
    bool serial_open;
    bool wavefront_decode;
    bool async_pipeline_decode;
    bool async_handoff;
    bool async_event_handoff;
    ds4_v100_replay_async_pipeline_mode async_pipeline_mode;
    bool suppress_router_readback;
} ds4_v100_replay_options;

typedef struct {
    uint32_t token;
    float logit;
    char *text;
    size_t text_len;
} ds4_v100_replay_output;

typedef struct {
    uint32_t prompt_tokens;
    uint32_t generated_tokens;
    uint32_t total_input_tokens;
    uint32_t layers_executed;
    uint64_t uploaded_tensors;
    uint64_t uploaded_bytes;
    uint64_t arena_bytes[DS4_V100_EXPECTED_GPUS];
    double open_ms[DS4_V100_EXPECTED_GPUS];
    double open_total_ms;
    double prompt_replay_ms;
    double continuation_decode_ms;
    double stage_decode_ms[DS4_V100_EXPECTED_GPUS];
    double stage_hc_attn_ms[DS4_V100_EXPECTED_GPUS];
    double stage_attention_ms[DS4_V100_EXPECTED_GPUS];
    double stage_hc_ffn_ms[DS4_V100_EXPECTED_GPUS];
    double stage_ffn_ms[DS4_V100_EXPECTED_GPUS];
    double stage_hc_final_ms[DS4_V100_EXPECTED_GPUS];
    double stage_profile_total_ms[DS4_V100_EXPECTED_GPUS];
    double handoff_ms[DS4_V100_EXPECTED_GPUS - 1];
    uint64_t async_pipeline_dispatches;
    double async_pipeline_total_ms;
    double async_pipeline_setup_ms;
    double async_pipeline_host_wait_ms;
    double async_pipeline_complete_ms;
    double async_pipeline_worker_wait_ms[DS4_V100_EXPECTED_GPUS];
    double async_pipeline_sync_ms[DS4_V100_EXPECTED_GPUS];
    double output_head_ms;
    double token_text_ms;
    double total_ms;
} ds4_v100_replay_counters;

void ds4_v100_replay_options_init(ds4_v100_replay_options *opts);

int ds4_v100_replay_open(ds4_v100_replay **out,
                         const ds4_v100_replay_options *opts,
                         char *err,
                         size_t errlen);

void ds4_v100_replay_open_counters(const ds4_v100_replay *rt,
                                   ds4_v100_replay_counters *out);

const void *ds4_v100_replay_model_map(const ds4_v100_replay *rt);

uint64_t ds4_v100_replay_model_size(const ds4_v100_replay *rt);

void ds4_v100_replay_close(ds4_v100_replay *rt);

int ds4_v100_replay_reset(ds4_v100_replay *rt, char *err, size_t errlen);

void ds4_v100_replay_encode_prompt(ds4_v100_replay *rt,
                                   const char *system,
                                   const char *prompt,
                                   ds4_think_mode think_mode,
                                   ds4_tokens *out);

int ds4_v100_replay_generate(ds4_v100_replay *rt,
                             const ds4_tokens *prompt,
                             uint32_t max_tokens,
                             ds4_v100_replay_output *outputs,
                             uint32_t output_cap,
                             uint32_t *out_count,
                             ds4_v100_replay_counters *counters,
                             char *err,
                             size_t errlen);

int ds4_v100_replay_generate_batch(ds4_v100_replay *rt,
                                   const ds4_tokens *prompts,
                                   uint32_t n_prompts,
                                   uint32_t max_tokens,
                                   ds4_v100_replay_output *outputs,
                                   uint32_t output_stride,
                                   uint32_t *out_counts,
                                   ds4_v100_replay_counters *counters,
                                   char *err,
                                   size_t errlen);

int ds4_v100_replay_generate_first_token_batch(
    ds4_v100_replay *rt,
    const ds4_tokens *prompts,
    uint32_t n_prompts,
    ds4_v100_replay_output *outputs,
    ds4_v100_replay_counters *counters,
    char *err,
    size_t errlen);

int ds4_v100_replay_begin_generation(ds4_v100_replay *rt,
                                      uint32_t prompt_tokens,
                                      ds4_v100_replay_counters *counters,
                                      char *err,
                                      size_t errlen);

int ds4_v100_replay_feed_token_at_position(ds4_v100_replay *rt,
                                            uint32_t token,
                                            uint32_t position,
                                            ds4_v100_replay_counters *counters,
                                            double *bucket_ms,
                                            char *err,
                                            size_t errlen);

int ds4_v100_replay_select_current_token(ds4_v100_replay *rt,
                                          ds4_v100_replay_output *out,
                                          ds4_v100_replay_counters *counters,
                                          char *err,
                                          size_t errlen);

void ds4_v100_replay_finish_generation(ds4_v100_replay *rt,
                                        uint32_t generated_tokens,
                                        double total_ms,
                                        ds4_v100_replay_counters *counters);

int ds4_v100_replay_read_token_embedding_f32(ds4_v100_replay *rt,
                                             uint32_t token,
                                             float *dst,
                                             uint64_t dst_values,
                                             char *err,
                                             size_t errlen);

int ds4_v100_replay_read_output_hc(ds4_v100_replay *rt,
                                    float *dst,
                                    uint64_t bytes,
                                    char *err,
                                    size_t errlen);

int ds4_v100_replay_read_output_hc_slot(ds4_v100_replay *rt,
                                        uint32_t slot,
                                        float *dst,
                                        uint64_t bytes,
                                        char *err,
                                        size_t errlen);

void ds4_v100_replay_output_free(ds4_v100_replay_output *out);

#ifdef __cplusplus
}
#endif

#endif
