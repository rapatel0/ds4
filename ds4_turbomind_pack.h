#ifndef DS4_TURBOMIND_PACK_H
#define DS4_TURBOMIND_PACK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_tm_pack ds4_tm_pack;

typedef struct {
    const char *semantic_tensor_id;
    const char *source_name;
    const char *source_dtype;
    const char *source_shape;
    const char *runtime_layout;
    const char *kernel_family;
    const char *sidecar_file;
    const char *source_shard_file;
    const char *source_checksum;
    int owning_gpu;
    int layer_id;
    uint32_t n;
    uint32_t k;
    uint32_t experts_packed;
    uint32_t experts_total;
    uint64_t weight_bytes_per_expert;
    uint64_t scale_bytes_per_expert;
    int k_pack;
    int weight_stride;
    int scale_stride;
    uint64_t weight_offset;
    uint64_t scale_offset;
    uint64_t source_shard_offset;
    uint64_t source_byte_length;
    int tm_abi_version;
} ds4_tm_pack_entry;

int ds4_tm_pack_open(ds4_tm_pack **out, const char *path, char *err, size_t errlen);
void ds4_tm_pack_close(ds4_tm_pack *pack);

uint64_t ds4_tm_pack_count(const ds4_tm_pack *pack);
int ds4_tm_pack_max_gpu(const ds4_tm_pack *pack);

int ds4_tm_pack_lookup(const ds4_tm_pack *pack,
                       const char *semantic_tensor_id,
                       ds4_tm_pack_entry *out);
int ds4_tm_pack_for_each(const ds4_tm_pack *pack,
                         int (*cb)(const ds4_tm_pack_entry *entry, void *ud),
                         void *ud);
int ds4_tm_pack_sidecar_bytes(const ds4_tm_pack *pack,
                              const char *sidecar_file,
                              uint64_t *out);

#ifdef __cplusplus
}
#endif

#endif
