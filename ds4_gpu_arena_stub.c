#include "ds4_gpu.h"

#include <inttypes.h>
#include <math.h>
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

static float f32_from_bits(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float e8m0_to_f32(uint8_t e) {
    return f32_from_bits(e == 0 ? 0x00400000u : ((uint32_t)e << 23));
}

static float e4m3fn_to_f32(uint8_t x) {
    const uint8_t ax = x & 0x7fu;
    const int sign = (x & 0x80u) != 0;
    if (ax == 0) return f32_from_bits(sign ? 0x80000000u : 0u);
    if (ax == 0x7f) return f32_from_bits(0x7fc00000u);
    const int exp = (x >> 3) & 0x0f;
    const int man = x & 0x07;
    float v = exp == 0 ? ldexpf((float)man, -9)
                       : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    return sign ? -v : v;
}

static int checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return 1;
    *out = a * b;
    return 0;
}

static int f8_e4m3_b128_row_bytes(uint32_t cols, uint64_t *out) {
    if (cols == 0 || cols % 128u) return 1;
    uint64_t blocks = (uint64_t)cols / 128ull;
    if (blocks > UINT64_MAX / 129ull) return 1;
    *out = blocks * 129ull;
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

static int f8_view_range_ok(const ds4_gpu_arena *arena,
                            const ds4_gpu_source_row_view *view,
                            const uint32_t *row_ids,
                            uint32_t n_rows,
                            const float *out_f32,
                            uint64_t out_bytes,
                            uint64_t *out_values) {
    if (!arena || !view || !row_ids || !out_f32 || !arena->valid) return 0;
    if (n_rows == 0 || view->rows == 0 || view->cols == 0 || view->row_stride_bytes == 0) return 0;
    if (!arena_range_ok(arena, view->arena_offset, view->byte_length)) return 0;

    uint64_t row_bytes = 0;
    if (f8_e4m3_b128_row_bytes(view->cols, &row_bytes)) return 0;
    if ((uint64_t)view->row_stride_bytes < row_bytes) return 0;

    uint64_t values = 0;
    uint64_t output_bytes = 0;
    if (checked_mul_u64((uint64_t)n_rows, (uint64_t)view->cols, &values)) return 0;
    if (checked_mul_u64(values, sizeof(float), &output_bytes)) return 0;
    if (out_bytes < output_bytes) return 0;

    uint64_t last_row = (uint64_t)view->rows - 1u;
    uint64_t last_start = 0;
    if (checked_mul_u64(last_row, (uint64_t)view->row_stride_bytes, &last_start)) return 0;
    if (last_start > view->byte_length || row_bytes > view->byte_length - last_start) return 0;

    for (uint32_t i = 0; i < n_rows; i++) {
        if (row_ids[i] >= view->rows) return 0;
    }

    if (out_values) *out_values = values;
    return 1;
}

int ds4_gpu_device_count(void) {
    return 0;
}

int ds4_gpu_set_device(int gpu) {
    return gpu >= 0;
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

int ds4_gpu_arena_bf16_matmul_f32(
        const ds4_gpu_arena            *arena,
        const ds4_gpu_bf16_matrix_view *view,
        const ds4_gpu_tensor           *x_f32,
        ds4_gpu_tensor                 *out_f32) {
    (void)arena;
    (void)view;
    (void)x_f32;
    (void)out_f32;
    return 1;
}

int ds4_gpu_arena_bf16_matmul_f32_rows(
        const ds4_gpu_arena            *arena,
        const ds4_gpu_bf16_matrix_view *view,
        const ds4_gpu_tensor           *x_f32,
        uint32_t                        n_rows,
        ds4_gpu_tensor                 *out_f32) {
    (void)arena;
    (void)view;
    (void)x_f32;
    (void)n_rows;
    (void)out_f32;
    return 1;
}

int ds4_gpu_top1_f32_rows_tensor(const ds4_gpu_tensor *logits,
                                 uint32_t n_rows,
                                 uint32_t n_logits,
                                 uint32_t *tokens,
                                 float *logits_out) {
    (void)logits;
    (void)n_rows;
    (void)n_logits;
    (void)tokens;
    (void)logits_out;
    return 0;
}

int ds4_gpu_arena_f32_matmul_f32(
        const ds4_gpu_arena           *arena,
        const ds4_gpu_source_row_view *view,
        const ds4_gpu_tensor          *x_f32,
        ds4_gpu_tensor                *out_f32) {
    (void)arena;
    (void)view;
    (void)x_f32;
    (void)out_f32;
    return 1;
}

int ds4_gpu_arena_f8_e4m3_b128_row_decode_f32(
        const ds4_gpu_arena           *arena,
        const ds4_gpu_source_row_view *view,
        const uint32_t                *row_ids,
        uint32_t                       n_rows,
        float                         *out_f32,
        uint64_t                       out_bytes) {
    uint64_t values = 0;
    if (!f8_view_range_ok(arena, view, row_ids, n_rows, out_f32, out_bytes, &values)) {
        return 1;
    }
    (void)values;

    uint64_t out_i = 0;
    const uint8_t *base = arena->ptr + view->arena_offset;
    for (uint32_t r = 0; r < n_rows; r++) {
        const uint8_t *row = base + (uint64_t)row_ids[r] * view->row_stride_bytes;
        for (uint32_t c = 0; c < view->cols; c++) {
            const uint8_t *block = row + (uint64_t)(c / 128u) * 129ull;
            out_f32[out_i++] = e4m3fn_to_f32(block[1u + (c % 128u)]) * e8m0_to_f32(block[0]);
        }
    }
    return 0;
}

int ds4_gpu_arena_f8_e4m3_b128_matmul_batch_f32(
        const ds4_gpu_arena           *arena,
        const ds4_gpu_source_row_view *view,
        const ds4_gpu_tensor          *x_f32,
        uint32_t                       n_tokens,
        ds4_gpu_tensor                *out_f32) {
    (void)arena;
    (void)view;
    (void)x_f32;
    (void)n_tokens;
    (void)out_f32;
    return 1;
}

int ds4_gpu_arena_f8_e4m3_b128_pair_swiglu_batch_ptrs_f32(
        const ds4_gpu_arena           *arena,
        const ds4_gpu_source_row_view *gate,
        const ds4_gpu_source_row_view *up,
        ds4_gpu_tensor                *x_row_ptrs,
        const ds4_gpu_tensor *const   *x_rows_f32,
        uint32_t                       n_tokens,
        ds4_gpu_tensor                *out_f32,
        float                          clamp,
        float                          weight) {
    (void)arena;
    (void)gate;
    (void)up;
    (void)x_row_ptrs;
    (void)x_rows_f32;
    (void)n_tokens;
    (void)out_f32;
    (void)clamp;
    (void)weight;
    return 1;
}

int ds4_gpu_arena_f8_e4m3_b128_pair_swiglu_batch_ptr_table_f32(
        const ds4_gpu_arena           *arena,
        const ds4_gpu_source_row_view *gate,
        const ds4_gpu_source_row_view *up,
        const ds4_gpu_tensor          *x_row_ptrs,
        uint32_t                       n_tokens,
        ds4_gpu_tensor                *out_f32,
        float                          clamp,
        float                          weight) {
    (void)arena;
    (void)gate;
    (void)up;
    (void)x_row_ptrs;
    (void)n_tokens;
    (void)out_f32;
    (void)clamp;
    (void)weight;
    return 1;
}

int ds4_gpu_arena_mxfp4_matmul_f32(
        const ds4_gpu_arena           *arena,
        const ds4_gpu_source_row_view *view,
        const ds4_gpu_tensor          *x_f32,
        ds4_gpu_tensor                *out_f32) {
    (void)arena;
    (void)view;
    (void)x_f32;
    (void)out_f32;
    return 1;
}

int ds4_gpu_arena_q8_0_matmul_f32(
        const ds4_gpu_arena           *arena,
        const ds4_gpu_source_row_view *view,
        const ds4_gpu_tensor          *x_f32,
        ds4_gpu_tensor                *out_f32,
        uint64_t                       n_tok) {
    (void)arena;
    (void)view;
    (void)x_f32;
    (void)out_f32;
    (void)n_tok;
    return 1;
}

int ds4_gpu_arena_f32_rms_norm_f32(
        const ds4_gpu_arena           *arena,
        const ds4_gpu_source_row_view *weight,
        const ds4_gpu_tensor          *x_f32,
        ds4_gpu_tensor                *out_f32,
        uint32_t                       n,
        uint32_t                       rows,
        float                          eps) {
    (void)arena;
    (void)weight;
    (void)x_f32;
    (void)out_f32;
    (void)n;
    (void)rows;
    (void)eps;
    return 1;
}

int ds4_gpu_arena_q4_k_routed_moe_one_f32(
        const ds4_gpu_arena             *arena,
        const ds4_gpu_q4_k_expert_view  *gate,
        const ds4_gpu_q4_k_expert_view  *up,
        const ds4_gpu_q4_k_expert_view  *down_w,
        ds4_gpu_tensor                  *out_f32,
        ds4_gpu_tensor                  *gate_tmp_f32,
        ds4_gpu_tensor                  *up_tmp_f32,
        ds4_gpu_tensor                  *mid_tmp_f32,
        ds4_gpu_tensor                  *down_tmp_f32,
        const ds4_gpu_tensor            *selected_i32,
        const ds4_gpu_tensor            *weights_f32,
        const ds4_gpu_tensor            *x_f32,
        uint32_t                         n_expert,
        float                            clamp) {
    (void)arena;
    (void)gate;
    (void)up;
    (void)down_w;
    (void)out_f32;
    (void)gate_tmp_f32;
    (void)up_tmp_f32;
    (void)mid_tmp_f32;
    (void)down_tmp_f32;
    (void)selected_i32;
    (void)weights_f32;
    (void)x_f32;
    (void)n_expert;
    (void)clamp;
    return 1;
}

int ds4_gpu_arena_router_select_bias_tensor(
        const ds4_gpu_arena           *arena,
        const ds4_gpu_source_row_view *bias,
        ds4_gpu_tensor                *selected_i32,
        ds4_gpu_tensor                *weights_f32,
        ds4_gpu_tensor                *probs_f32,
        const ds4_gpu_tensor          *logits_f32) {
    (void)arena;
    (void)bias;
    (void)selected_i32;
    (void)weights_f32;
    (void)probs_f32;
    (void)logits_f32;
    return 1;
}

int ds4_gpu_arena_hc_split_weighted_sum_tensor(
        const ds4_gpu_arena           *arena,
        const ds4_gpu_source_row_view *scale,
        const ds4_gpu_source_row_view *base,
        ds4_gpu_tensor                *out,
        ds4_gpu_tensor                *split,
        const ds4_gpu_tensor          *mix,
        const ds4_gpu_tensor          *residual_hc,
        uint32_t                       n_embd,
        uint32_t                       n_hc,
        uint32_t                       sinkhorn_iters,
        float                          eps) {
    (void)arena;
    (void)scale;
    (void)base;
    (void)out;
    (void)split;
    (void)mix;
    (void)residual_hc;
    (void)n_embd;
    (void)n_hc;
    (void)sinkhorn_iters;
    (void)eps;
    return 1;
}

int ds4_gpu_arena_output_hc_weights_tensor(
        const ds4_gpu_arena           *arena,
        const ds4_gpu_source_row_view *scale,
        const ds4_gpu_source_row_view *base,
        ds4_gpu_tensor                *out,
        const ds4_gpu_tensor          *pre,
        uint32_t                       n_hc,
        float                          eps) {
    (void)arena;
    (void)scale;
    (void)base;
    (void)out;
    (void)pre;
    (void)n_hc;
    (void)eps;
    return 1;
}

int ds4_gpu_arena_attention_decode_heads_tensor(
        const ds4_gpu_arena           *arena,
        const ds4_gpu_source_row_view *sinks,
        ds4_gpu_tensor                *heads,
        const ds4_gpu_tensor          *q,
        const ds4_gpu_tensor          *raw_kv,
        uint32_t                       n_raw,
        uint32_t                       raw_cap,
        uint32_t                       raw_start,
        const ds4_gpu_tensor          *comp_kv,
        uint32_t                       n_comp,
        const ds4_gpu_tensor          *comp_mask,
        uint32_t                       use_mask,
        uint32_t                       n_head,
        uint32_t                       head_dim) {
    (void)arena;
    (void)sinks;
    (void)heads;
    (void)q;
    (void)raw_kv;
    (void)n_raw;
    (void)raw_cap;
    (void)raw_start;
    (void)comp_kv;
    (void)n_comp;
    (void)comp_mask;
    (void)use_mask;
    (void)n_head;
    (void)head_dim;
    return 1;
}
