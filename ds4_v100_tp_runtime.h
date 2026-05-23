#ifndef DS4_V100_TP_RUNTIME_H
#define DS4_V100_TP_RUNTIME_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_V100_TP_MAX_GPUS 8

typedef enum {
    DS4_V100_TP_KV_F16 = 0,
    DS4_V100_TP_KV_F8_E4M3_B128 = 1,
    DS4_V100_TP_KV_Q8_0 = 2,
} ds4_v100_tp_kv_dtype;

typedef struct {
    int devices[DS4_V100_TP_MAX_GPUS];
    uint32_t slots;
    uint64_t ctx;
    uint32_t hidden;
    ds4_v100_tp_kv_dtype kv_dtype;
    uint64_t scratch_bytes;
} ds4_v100_tp_runtime_config;

typedef struct {
    uint64_t hidden_bytes;
    uint64_t kv_bytes;
    uint64_t comp_state_bytes;
    uint64_t scratch_bytes;
    uint64_t total_bytes;
} ds4_v100_tp_gpu_report;

typedef struct {
    ds4_v100_tp_gpu_report gpu[DS4_V100_TP_MAX_GPUS];
} ds4_v100_tp_runtime_report;

typedef struct ds4_v100_tp_runtime ds4_v100_tp_runtime;

void ds4_v100_tp_runtime_default_config(ds4_v100_tp_runtime_config *cfg);

int ds4_v100_tp_runtime_open(ds4_v100_tp_runtime **out,
                             const ds4_v100_tp_runtime_config *cfg,
                             char *err,
                             size_t err_len);

int ds4_v100_tp_runtime_fixture(ds4_v100_tp_runtime *rt,
                                double *max_abs,
                                char *err,
                                size_t err_len);

void ds4_v100_tp_runtime_get_report(const ds4_v100_tp_runtime *rt,
                                    ds4_v100_tp_runtime_report *report);

void ds4_v100_tp_runtime_close(ds4_v100_tp_runtime *rt);

#ifdef __cplusplus
}
#endif

#endif
