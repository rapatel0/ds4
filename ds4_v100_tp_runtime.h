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
    DS4_V100_TP_KV_F8_E5M2_B128 = 3,
} ds4_v100_tp_kv_dtype;

typedef enum {
    DS4_V100_TP_KV_ROW_ATTN = 0,
    DS4_V100_TP_KV_ROW_INDEXER = 1,
    DS4_V100_TP_KV_ROW_ATTN_RAW = 2,
} ds4_v100_tp_kv_row_kind;

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

typedef struct {
    int layer;
    int ratio;
    uint32_t slot;
    uint64_t position;
    ds4_v100_tp_kv_row_kind kind;
    uint32_t logical_cols;
    uint64_t logical_row_bytes;
    uint64_t physical_row;
    uint64_t offset[DS4_V100_TP_MAX_GPUS];
    uint64_t row_bytes[DS4_V100_TP_MAX_GPUS];
} ds4_v100_tp_kv_row_view;

typedef struct {
    ds4_v100_tp_kv_row_view view;
    double max_abs;
    double mean_abs;
    uint32_t bad_values;
    uint32_t byte_mismatches;
    uint32_t first_bad_index;
    uint8_t first_bad_got;
    uint8_t first_bad_expected;
    uint64_t checksum;
} ds4_v100_tp_kv_row_roundtrip_result;

typedef struct {
    ds4_v100_tp_kv_row_view view;
    double max_abs;
    double mean_abs;
    uint32_t bad_values;
    uint64_t checksum;
} ds4_v100_tp_kv_device_roundtrip_result;

typedef struct {
    int layer;
    int ratio;
    uint32_t slot;
    uint64_t position;
    uint64_t attn_row;
    uint64_t attn_offset[DS4_V100_TP_MAX_GPUS];
    uint64_t attn_row_bytes[DS4_V100_TP_MAX_GPUS];
    uint64_t indexer_row;
    uint64_t indexer_offset[DS4_V100_TP_MAX_GPUS];
    uint64_t indexer_row_bytes[DS4_V100_TP_MAX_GPUS];
    double max_abs;
} ds4_v100_tp_dense_kv_result;

void ds4_v100_tp_runtime_default_config(ds4_v100_tp_runtime_config *cfg);

int ds4_v100_tp_runtime_open(ds4_v100_tp_runtime **out,
                             const ds4_v100_tp_runtime_config *cfg,
                             char *err,
                             size_t err_len);

int ds4_v100_tp_runtime_fixture(ds4_v100_tp_runtime *rt,
                                double *max_abs,
                                char *err,
                                size_t err_len);

int ds4_v100_tp_runtime_dense_kv_slice(ds4_v100_tp_runtime *rt,
                                       int layer,
                                       uint32_t slot,
                                       uint64_t position,
                                       int write_indexer,
                                       ds4_v100_tp_dense_kv_result *result,
                                       char *err,
                                       size_t err_len);

int ds4_v100_tp_runtime_kv_row_view(ds4_v100_tp_runtime *rt,
                                    int layer,
                                    uint32_t slot,
                                    uint64_t position,
                                    ds4_v100_tp_kv_row_kind kind,
                                    ds4_v100_tp_kv_row_view *view,
                                    char *err,
                                    size_t err_len);

int ds4_v100_tp_runtime_kv_row_roundtrip_f32(ds4_v100_tp_runtime *rt,
                                             int layer,
                                             uint32_t slot,
                                             uint64_t position,
                                             ds4_v100_tp_kv_row_kind kind,
                                             ds4_v100_tp_kv_row_roundtrip_result *result,
                                             char *err,
                                             size_t err_len);

int ds4_v100_tp_runtime_kv_row_store_f32_device(
    ds4_v100_tp_runtime *rt,
    int layer,
    uint32_t slot,
    uint64_t position,
    ds4_v100_tp_kv_row_kind kind,
    const void *src_by_gpu[DS4_V100_TP_MAX_GPUS],
    char *err,
    size_t err_len);

int ds4_v100_tp_runtime_kv_row_load_f32_device(
    ds4_v100_tp_runtime *rt,
    int layer,
    uint32_t slot,
    uint64_t position,
    ds4_v100_tp_kv_row_kind kind,
    void *dst_by_gpu[DS4_V100_TP_MAX_GPUS],
    char *err,
    size_t err_len);

int ds4_v100_tp_runtime_kv_rows_store_f32_device(
    ds4_v100_tp_runtime *rt,
    int layer,
    uint32_t first_slot,
    uint32_t slot_count,
    uint64_t position,
    ds4_v100_tp_kv_row_kind kind,
    const void *src_by_gpu[DS4_V100_TP_MAX_GPUS],
    uint64_t src_stride_floats,
    char *err,
    size_t err_len);

int ds4_v100_tp_runtime_kv_rows_load_f32_device(
    ds4_v100_tp_runtime *rt,
    int layer,
    uint32_t first_slot,
    uint32_t slot_count,
    uint64_t position,
    ds4_v100_tp_kv_row_kind kind,
    void *dst_by_gpu[DS4_V100_TP_MAX_GPUS],
    uint64_t dst_stride_floats,
    char *err,
    size_t err_len);

int ds4_v100_tp_runtime_kv_row_device_roundtrip_f32(
    ds4_v100_tp_runtime *rt,
    int layer,
    uint32_t slot,
    uint64_t position,
    ds4_v100_tp_kv_row_kind kind,
    ds4_v100_tp_kv_device_roundtrip_result *result,
    char *err,
    size_t err_len);

void ds4_v100_tp_runtime_get_report(const ds4_v100_tp_runtime *rt,
                                    ds4_v100_tp_runtime_report *report);

void ds4_v100_tp_runtime_close(ds4_v100_tp_runtime *rt);

#ifdef __cplusplus
}
#endif

#endif
