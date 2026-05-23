#ifndef DS4_V100_MTP_FORWARD_COMMON_H
#define DS4_V100_MTP_FORWARD_COMMON_H

#include "ds4_gpu.h"
#include "ds4_v100_context.h"
#include "ds4_v100_mtp.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    DS4_V100_MTP_FORWARD_MAX_TOPK = 16,
    DS4_V100_MTP_FORWARD_N_EMBD = 4096,
    DS4_V100_MTP_FORWARD_N_HC = 4,
    DS4_V100_MTP_FORWARD_HC_VALUES =
        DS4_V100_MTP_FORWARD_N_EMBD * DS4_V100_MTP_FORWARD_N_HC,
    DS4_V100_MTP_FORWARD_RAW_CAP = 128,
};

typedef struct ds4_v100_mtp_forward ds4_v100_mtp_forward;

typedef struct {
    uint32_t raw_row;
    uint32_t n_raw;
    uint32_t output_vocab;
    uint64_t output_weight_bytes;
    uint64_t free_after_output_upload_bytes;
    uint64_t scratch_device_bytes;
    uint64_t scratch_host_bytes;
    uint64_t run_count;
} ds4_v100_mtp_forward_report;

int ds4_v100_mtp_forward_open(ds4_v100_mtp_forward **out,
                              ds4_v100_mtp_sidecar *sidecar,
                              const void *base_model,
                              uint64_t base_model_size,
                              const ds4_v100_tensor_binding *output_weight,
                              int gpu,
                              char *err,
                              size_t errlen);

int ds4_v100_mtp_forward_run_host(ds4_v100_mtp_forward *fwd,
                                  const float *embed,
                                  const float *prev_hc,
                                  uint32_t position,
                                  uint32_t top_k,
                                  uint32_t *tokens,
                                  float *logits,
                                  ds4_v100_mtp_forward_report *report,
                                  char *err,
                                  size_t errlen);

int ds4_v100_mtp_forward_run_host_next_hc(ds4_v100_mtp_forward *fwd,
                                          const float *embed,
                                          const float *prev_hc,
                                          uint32_t position,
                                          uint32_t top_k,
                                          uint32_t *tokens,
                                          float *logits,
                                          float *next_hc,
                                          uint64_t next_hc_values,
                                          ds4_v100_mtp_forward_report *report,
                                          char *err,
                                          size_t errlen);

void ds4_v100_mtp_forward_close(ds4_v100_mtp_forward *fwd);

#ifdef __cplusplus
}
#endif

#endif
