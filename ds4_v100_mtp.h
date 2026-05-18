#ifndef DS4_V100_MTP_H
#define DS4_V100_MTP_H

#include "ds4.h"
#include "ds4_gpu.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_v100_mtp_sidecar ds4_v100_mtp_sidecar;

typedef struct {
    const char *mtp_path;
    int gpu;
    uint64_t upload_chunk_bytes;
    bool require_device_arena;
} ds4_v100_mtp_sidecar_options;

void ds4_v100_mtp_sidecar_options_init(ds4_v100_mtp_sidecar_options *opts);

int ds4_v100_mtp_sidecar_open(ds4_v100_mtp_sidecar **out,
                              const ds4_v100_mtp_sidecar_options *opts,
                              FILE *report,
                              char *err,
                              size_t errlen);

void ds4_v100_mtp_sidecar_close(ds4_v100_mtp_sidecar *sidecar);

const ds4_mtp_sidecar_info *ds4_v100_mtp_sidecar_info(
        const ds4_v100_mtp_sidecar *sidecar);

const ds4_mtp_sidecar_tensor_info *ds4_v100_mtp_sidecar_tensor(
        const ds4_v100_mtp_sidecar *sidecar,
        const char *name);

ds4_gpu_arena *ds4_v100_mtp_sidecar_arena(ds4_v100_mtp_sidecar *sidecar);
uint64_t ds4_v100_mtp_sidecar_uploaded_bytes(const ds4_v100_mtp_sidecar *sidecar);
uint64_t ds4_v100_mtp_sidecar_spot_checks(const ds4_v100_mtp_sidecar *sidecar);
int ds4_v100_mtp_sidecar_gpu(const ds4_v100_mtp_sidecar *sidecar);

#ifdef __cplusplus
}
#endif

#endif
