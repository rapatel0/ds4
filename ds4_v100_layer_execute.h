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
#define DS4_V100_OUT_GROUPS 8u
#define DS4_V100_OUT_GROUP_DIM 4096u
#define DS4_V100_OUT_GROUP_RANK 1024u

typedef struct {
    const void *model_map;
    uint64_t model_size;
    ds4_gpu_arena *arena;

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
} ds4_v100_layer_execute_config;

typedef struct {
    int32_t selected_experts[6];
    float route_weights[6];
    uint32_t routes;
} ds4_v100_layer_execute_report;

int ds4_v100_layer_execute_decode(
        const ds4_v100_layer_state          *state,
        const ds4_v100_layer_execute_config *cfg,
        const ds4_gpu_tensor                *hidden,
        ds4_gpu_tensor                      *next_hidden,
        ds4_v100_layer_execute_report       *report,
        char                                *err,
        size_t                               errlen);

#ifdef __cplusplus
}
#endif

#endif
