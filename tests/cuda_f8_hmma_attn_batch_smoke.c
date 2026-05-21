#include "ds4_gpu.h"
#include "ds4_source_formats.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_f8_hmma_attn_batch_smoke: %s\n", msg);
        failures++;
    }
}

static void fill_f8_row(uint8_t *row, uint32_t cols, uint32_t r, uint32_t salt) {
    static const uint8_t codes[] = {
        0x00, 0x80, 0x30, 0x38, 0x40, 0xb0, 0xb8, 0xc0,
    };
    const uint64_t row_bytes = ds4_src_f8_e4m3_b128_row_bytes(cols);
    memset(row, 0, (size_t)row_bytes);
    for (uint32_t b = 0; b < cols / DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS; b++) {
        uint8_t *block = row + (uint64_t)b * DS4_SRC_F8_E4M3_B128_BLOCK_BYTES;
        block[0] = (uint8_t)(118u + ((r + b + salt) & 7u));
        for (uint32_t i = 0; i < DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS; i++) {
            block[1u + i] = codes[(r + b + i + salt) & 7u];
        }
    }
}

static void fill_x(float *x, uint32_t tokens, uint32_t cols, uint32_t salt) {
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t c = 0; c < cols; c++) {
            x[(uint64_t)t * cols + c] =
                (float)((int)((c + 17u * t + salt) % 23u) - 11) * 0.001953125f;
        }
    }
}

static int compare_outputs(const float *scalar,
                           const float *hmma,
                           uint32_t tokens,
                           uint32_t rows,
                           const char *label) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    uint64_t max_i = 0;
    for (uint64_t i = 0; i < (uint64_t)tokens * rows; i++) {
        const float diff = fabsf(hmma[i] - scalar[i]);
        const float denom = fmaxf(fabsf(scalar[i]), 1.0e-6f);
        const float rel = diff / denom;
        if (diff > max_abs) {
            max_abs = diff;
            max_rel = rel;
            max_i = i;
        }
    }
    if (max_abs > 0.20f && max_rel > 0.20f) {
        fprintf(stderr,
                "cuda_f8_hmma_attn_batch_smoke: %s max diff %.8g rel %.8g at token %llu row %llu scalar %.8g hmma %.8g\n",
                label,
                max_abs,
                max_rel,
                (unsigned long long)(max_i / rows),
                (unsigned long long)(max_i % rows),
                scalar[max_i],
                hmma[max_i]);
        return 0;
    }
    return 1;
}

static void run_ptr_case(uint32_t rows, uint32_t cols, const char *label, uint32_t salt) {
    enum { TOKENS = 8 };
    const uint64_t row_bytes = ds4_src_f8_e4m3_b128_row_bytes(cols);
    const uint64_t matrix_bytes = (uint64_t)rows * row_bytes;
    uint8_t *payload = (uint8_t *)malloc((size_t)matrix_bytes);
    float *x = (float *)malloc((uint64_t)TOKENS * cols * sizeof(float));
    float *scalar = (float *)malloc((uint64_t)TOKENS * rows * sizeof(float));
    float *hmma = (float *)malloc((uint64_t)TOKENS * rows * sizeof(float));
    check(payload && x && scalar && hmma, "ptr host buffers allocate");
    if (!payload || !x || !scalar || !hmma) goto done_host;

    for (uint32_t r = 0; r < rows; r++) {
        fill_f8_row(payload + (uint64_t)r * row_bytes, cols, r, salt);
    }
    fill_x(x, TOKENS, cols, salt);

    ds4_gpu_arena *arena = NULL;
    ds4_gpu_tensor *x_ptrs_t = NULL;
    ds4_gpu_tensor *scalar_t = NULL;
    ds4_gpu_tensor *hmma_t = NULL;
    ds4_gpu_tensor *x_rows[TOKENS] = {0};

    check(ds4_gpu_arena_open(&arena, 0, matrix_bytes) == 0, "ptr arena open");
    if (!arena) goto done_gpu;
    check(ds4_gpu_arena_upload(arena, 0, payload, matrix_bytes) == 0, "ptr arena upload");

    ds4_gpu_source_row_view view = {
        .arena_offset = 0,
        .byte_length = matrix_bytes,
        .rows = rows,
        .cols = cols,
        .row_stride_bytes = (uint32_t)row_bytes,
    };

    x_ptrs_t = ds4_gpu_tensor_alloc(TOKENS * sizeof(void *));
    scalar_t = ds4_gpu_tensor_alloc((uint64_t)TOKENS * rows * sizeof(float));
    hmma_t = ds4_gpu_tensor_alloc((uint64_t)TOKENS * rows * sizeof(float));
    check(x_ptrs_t && scalar_t && hmma_t, "ptr device tensors allocate");
    for (uint32_t t = 0; t < TOKENS; t++) {
        x_rows[t] = ds4_gpu_tensor_alloc(cols * sizeof(float));
        check(x_rows[t] != NULL, "ptr x row allocate");
        if (x_rows[t]) {
            check(ds4_gpu_tensor_write(x_rows[t],
                                       0,
                                       x + (uint64_t)t * cols,
                                       cols * sizeof(float)),
                  "ptr x row upload");
        }
    }
    if (x_ptrs_t && scalar_t && hmma_t) {
        unsetenv("DS4_CUDA_F8_F16_CACHE");
        setenv("DS4_CUDA_F8_ROWPAIR", "1", 1);
        setenv("DS4_CUDA_F8_HMMA_ATTN_BATCH", "0", 1);
        check(ds4_gpu_tensor_write_f32_row_ptrs(x_ptrs_t,
                                                (const ds4_gpu_tensor *const *)x_rows,
                                                TOKENS,
                                                cols * sizeof(float)),
              "ptr table write");
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_batch_ptr_table_f32(arena,
                                                                    &view,
                                                                    x_ptrs_t,
                                                                    TOKENS,
                                                                    scalar_t) == 0,
              "ptr scalar batch");
        setenv("DS4_CUDA_F8_HMMA_ATTN_BATCH", "1", 1);
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_batch_ptr_table_f32(arena,
                                                                    &view,
                                                                    x_ptrs_t,
                                                                    TOKENS,
                                                                    hmma_t) == 0,
              "ptr hmma batch");
        check(ds4_gpu_tensor_read(scalar_t,
                                  0,
                                  scalar,
                                  (uint64_t)TOKENS * rows * sizeof(float)),
              "ptr scalar read");
        check(ds4_gpu_tensor_read(hmma_t,
                                  0,
                                  hmma,
                                  (uint64_t)TOKENS * rows * sizeof(float)),
              "ptr hmma read");
        check(compare_outputs(scalar, hmma, TOKENS, rows, label), label);
    }

done_gpu:
    for (uint32_t t = 0; t < TOKENS; t++) ds4_gpu_tensor_free(x_rows[t]);
    ds4_gpu_tensor_free(hmma_t);
    ds4_gpu_tensor_free(scalar_t);
    ds4_gpu_tensor_free(x_ptrs_t);
    ds4_gpu_arena_close(arena);
done_host:
    free(hmma);
    free(scalar);
    free(x);
    free(payload);
}

static void run_contiguous_case(uint32_t rows, uint32_t cols, const char *label, uint32_t salt) {
    enum { TOKENS = 8 };
    const uint64_t row_bytes = ds4_src_f8_e4m3_b128_row_bytes(cols);
    const uint64_t matrix_bytes = (uint64_t)rows * row_bytes;
    uint8_t *payload = (uint8_t *)malloc((size_t)matrix_bytes);
    float *x = (float *)malloc((uint64_t)TOKENS * cols * sizeof(float));
    float *scalar = (float *)malloc((uint64_t)TOKENS * rows * sizeof(float));
    float *hmma = (float *)malloc((uint64_t)TOKENS * rows * sizeof(float));
    check(payload && x && scalar && hmma, "contiguous host buffers allocate");
    if (!payload || !x || !scalar || !hmma) goto done_host;

    for (uint32_t r = 0; r < rows; r++) {
        fill_f8_row(payload + (uint64_t)r * row_bytes, cols, r, salt);
    }
    fill_x(x, TOKENS, cols, salt);

    ds4_gpu_arena *arena = NULL;
    ds4_gpu_tensor *x_t = NULL;
    ds4_gpu_tensor *scalar_t = NULL;
    ds4_gpu_tensor *hmma_t = NULL;

    check(ds4_gpu_arena_open(&arena, 0, matrix_bytes) == 0, "contiguous arena open");
    if (!arena) goto done_gpu;
    check(ds4_gpu_arena_upload(arena, 0, payload, matrix_bytes) == 0, "contiguous arena upload");

    ds4_gpu_source_row_view view = {
        .arena_offset = 0,
        .byte_length = matrix_bytes,
        .rows = rows,
        .cols = cols,
        .row_stride_bytes = (uint32_t)row_bytes,
    };

    x_t = ds4_gpu_tensor_alloc((uint64_t)TOKENS * cols * sizeof(float));
    scalar_t = ds4_gpu_tensor_alloc((uint64_t)TOKENS * rows * sizeof(float));
    hmma_t = ds4_gpu_tensor_alloc((uint64_t)TOKENS * rows * sizeof(float));
    check(x_t && scalar_t && hmma_t, "contiguous device tensors allocate");
    if (x_t && scalar_t && hmma_t) {
        check(ds4_gpu_tensor_write(x_t, 0, x, (uint64_t)TOKENS * cols * sizeof(float)),
              "contiguous x upload");
        unsetenv("DS4_CUDA_F8_F16_CACHE");
        setenv("DS4_CUDA_F8_ROWPAIR", "1", 1);
        setenv("DS4_CUDA_F8_HMMA_ATTN_BATCH", "0", 1);
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_batch_f32(arena,
                                                          &view,
                                                          x_t,
                                                          TOKENS,
                                                          scalar_t) == 0,
              "contiguous scalar batch");
        setenv("DS4_CUDA_F8_HMMA_ATTN_BATCH", "1", 1);
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_batch_f32(arena,
                                                          &view,
                                                          x_t,
                                                          TOKENS,
                                                          hmma_t) == 0,
              "contiguous hmma batch");
        check(ds4_gpu_tensor_read(scalar_t,
                                  0,
                                  scalar,
                                  (uint64_t)TOKENS * rows * sizeof(float)),
              "contiguous scalar read");
        check(ds4_gpu_tensor_read(hmma_t,
                                  0,
                                  hmma,
                                  (uint64_t)TOKENS * rows * sizeof(float)),
              "contiguous hmma read");
        check(compare_outputs(scalar, hmma, TOKENS, rows, label), label);
    }

done_gpu:
    ds4_gpu_tensor_free(hmma_t);
    ds4_gpu_tensor_free(scalar_t);
    ds4_gpu_tensor_free(x_t);
    ds4_gpu_arena_close(arena);
done_host:
    free(hmma);
    free(scalar);
    free(x);
    free(payload);
}

int main(void) {
    if (ds4_gpu_device_count() < 1) {
        fprintf(stderr, "cuda_f8_hmma_attn_batch_smoke: no CUDA devices visible\n");
        return 1;
    }

    run_ptr_case(1024, 4096, "q_a ptr-table", 1);
    run_ptr_case(512, 4096, "kv ptr-table", 3);
    run_contiguous_case(32768, 1024, "q_b contiguous", 5);

    if (failures) return 1;
    puts("cuda_f8_hmma_attn_batch_smoke: ok");
    return 0;
}
