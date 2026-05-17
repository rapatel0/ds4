#ifndef DS4_PACK_H
#define DS4_PACK_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#define DS4_PACK_MAX_GPUS 16

typedef struct ds4_pack ds4_pack;

typedef struct {
    const char *semantic_tensor_id;
    const char *source_name;
    const char *source_dtype;
    const char *source_shape;
    const char *runtime_layout;
    const char *kernel_family;
    const char *shard_file;
    int owning_gpu;
    int layer_id;
    int64_t scale_offset;
    uint64_t source_offset;
    uint64_t byte_length;
    uint64_t shard_offset;
} ds4_pack_entry;

typedef struct {
    const char *name;
    size_t name_len;
    const char *source_dtype;
    const char *source_shape;
    uint64_t source_offset;
    uint64_t byte_length;
} ds4_pack_source_tensor;

typedef struct {
    uint64_t source_tensors;
    uint64_t pack_rows;
    uint64_t ok_rows;
    uint64_t failed_rows;
    uint64_t extra_pack_rows;
} ds4_pack_reconcile_summary;

int ds4_pack_open(ds4_pack **out, const char *path, char *err, size_t errlen);
void ds4_pack_close(ds4_pack *pack);

uint64_t ds4_pack_count(const ds4_pack *pack);
int ds4_pack_max_gpu(const ds4_pack *pack);
uint64_t ds4_pack_payload_bytes(const ds4_pack *pack, int gpu);
uint64_t ds4_pack_arena_bytes(const ds4_pack *pack, int gpu);
uint64_t ds4_pack_tensor_count(const ds4_pack *pack, int gpu);

int ds4_pack_lookup(const ds4_pack *pack,
                    const char *semantic_tensor_id,
                    ds4_pack_entry *out);
int ds4_pack_for_each(const ds4_pack *pack,
                      int (*cb)(const ds4_pack_entry *entry, void *ud),
                      void *ud);

int ds4_pack_reconcile(const ds4_pack *pack,
                       const ds4_pack_source_tensor *source,
                       size_t n_source,
                       uint64_t source_file_size,
                       int n_gpus,
                       FILE *report,
                       ds4_pack_reconcile_summary *summary,
                       char *err,
                       size_t errlen);

int ds4_pack_validate_shards(const ds4_pack *pack,
                             const char *shard_dir,
                             FILE *report,
                             char *err,
                             size_t errlen);

#endif
