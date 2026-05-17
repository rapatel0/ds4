#include "ds4_gpu.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ds4_gpu_arena {
    unsigned char *ptr;
    uint64_t bytes;
    uint64_t used;
    int gpu;
    int valid;
};

static int arena_range_ok(const ds4_gpu_arena *a, uint64_t offset, uint64_t bytes) {
    return a && a->valid && offset <= a->bytes && bytes <= a->bytes - offset;
}

int ds4_gpu_device_count(void) {
    return 0;
}

int ds4_gpu_arena_open(ds4_gpu_arena **out, int gpu, uint64_t bytes) {
    if (!out || gpu < 0) return 1;
    ds4_gpu_arena *a = (ds4_gpu_arena *)calloc(1, sizeof(*a));
    if (!a) return 1;
    a->bytes = bytes;
    a->gpu = gpu;
    a->valid = 1;
    if (bytes) {
        a->ptr = (unsigned char *)malloc((size_t)bytes);
        if (!a->ptr) {
            free(a);
            return 1;
        }
    }
    *out = a;
    return 0;
}

void ds4_gpu_arena_close(ds4_gpu_arena *arena) {
    if (!arena) return;
    free(arena->ptr);
    free(arena);
}

int ds4_gpu_arena_upload(ds4_gpu_arena *arena,
                         uint64_t offset,
                         const void *host_src,
                         uint64_t bytes) {
    if (!arena_range_ok(arena, offset, bytes) || (bytes && !host_src)) {
        if (arena) arena->valid = 0;
        return 1;
    }
    if (bytes) memcpy(arena->ptr + offset, host_src, (size_t)bytes);
    if (offset + bytes > arena->used) arena->used = offset + bytes;
    return 0;
}

int ds4_gpu_arena_read(const ds4_gpu_arena *arena,
                       uint64_t offset,
                       void *dst,
                       uint64_t bytes) {
    if (!arena_range_ok(arena, offset, bytes) || (bytes && !dst)) return 1;
    if (bytes) memcpy(dst, arena->ptr + offset, (size_t)bytes);
    return 0;
}

uint64_t ds4_gpu_arena_bytes(const ds4_gpu_arena *arena) {
    return arena ? arena->bytes : 0;
}

uint64_t ds4_gpu_arena_used(const ds4_gpu_arena *arena) {
    return arena ? arena->used : 0;
}

uint64_t ds4_gpu_arena_free_after_upload_bytes(const ds4_gpu_arena *arena) {
    (void)arena;
    return 0;
}

int ds4_gpu_arena_gpu(const ds4_gpu_arena *arena) {
    return arena ? arena->gpu : -1;
}

const char *ds4_gpu_arena_memory_kind(const ds4_gpu_arena *arena) {
    (void)arena;
    return "host-stub";
}

int ds4_gpu_arena_is_device_memory(const ds4_gpu_arena *arena) {
    (void)arena;
    return 0;
}

void ds4_gpu_arena_print_memory_report(FILE *fp,
                                       ds4_gpu_arena * const *arenas,
                                       int n_arenas) {
    if (!fp) fp = stderr;
    fprintf(fp, "gpu\tarena_bytes\tused_bytes\tmemory_kind\tvalid\n");
    for (int i = 0; i < n_arenas; i++) {
        const ds4_gpu_arena *a = arenas ? arenas[i] : NULL;
        if (!a) continue;
        fprintf(fp, "%d\t%" PRIu64 "\t%" PRIu64 "\t%s\t%d\n",
                a->gpu, a->bytes, a->used, ds4_gpu_arena_memory_kind(a), a->valid);
    }
}

void ds4_gpu_print_topology_report(FILE *fp) {
    if (!fp) fp = stderr;
    fprintf(fp, "gpu_topology\tstub\tdevice_count\t0\n");
}
