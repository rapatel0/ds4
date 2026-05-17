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

static float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static int checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return 1;
    *out = a * b;
    return 0;
}

static int bf16_view_range_ok(const ds4_gpu_arena *arena,
                              const ds4_gpu_bf16_matrix_view *view,
                              const uint32_t *row_ids,
                              uint32_t n_rows,
                              const float *out_f32,
                              uint64_t out_bytes,
                              uint64_t *out_values) {
    if (!arena || !view || !row_ids || !out_f32 || !arena->valid) return 0;
    if (n_rows == 0 || view->rows == 0 || view->cols == 0) return 0;
    if (view->row_stride_elements < view->cols) return 0;
    if ((view->arena_offset & 1ull) != 0 || (view->byte_length & 1ull) != 0) return 0;
    if (!arena_range_ok(arena, view->arena_offset, view->byte_length)) return 0;

    uint64_t values = 0;
    uint64_t output_bytes = 0;
    if (checked_mul_u64((uint64_t)n_rows, (uint64_t)view->cols, &values)) return 0;
    if (checked_mul_u64(values, sizeof(float), &output_bytes)) return 0;
    if (out_bytes < output_bytes) return 0;

    uint64_t total_elements = view->byte_length / sizeof(uint16_t);
    uint64_t last_row = (uint64_t)view->rows - 1u;
    uint64_t last_start = 0;
    if (checked_mul_u64(last_row, (uint64_t)view->row_stride_elements, &last_start)) return 0;
    if ((uint64_t)view->cols > total_elements ||
        last_start > total_elements - (uint64_t)view->cols) {
        return 0;
    }

    for (uint32_t i = 0; i < n_rows; i++) {
        if (row_ids[i] >= view->rows) return 0;
    }

    if (out_values) *out_values = values;
    return 1;
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

int ds4_gpu_arena_bf16_row_gather_f32(
        const ds4_gpu_arena            *arena,
        const ds4_gpu_bf16_matrix_view *view,
        const uint32_t                 *row_ids,
        uint32_t                        n_rows,
        float                          *out_f32,
        uint64_t                        out_bytes) {
    uint64_t values = 0;
    if (!bf16_view_range_ok(arena, view, row_ids, n_rows, out_f32, out_bytes, &values)) {
        return 1;
    }
    (void)values;

    const uint16_t *base = (const uint16_t *)(const void *)(arena->ptr + view->arena_offset);
    uint64_t out_i = 0;
    for (uint32_t r = 0; r < n_rows; r++) {
        const uint16_t *row = base + (uint64_t)row_ids[r] * view->row_stride_elements;
        for (uint32_t c = 0; c < view->cols; c++) {
            out_f32[out_i++] = bf16_to_f32(row[c]);
        }
    }
    return 0;
}
