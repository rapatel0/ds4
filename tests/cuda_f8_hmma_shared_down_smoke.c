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
        fprintf(stderr, "cuda_f8_hmma_shared_down_smoke: %s\n", msg);
        failures++;
    }
}

static void fill_f8_row(uint8_t *row, uint32_t cols, uint32_t r) {
    static const uint8_t codes[] = {
        0x00, 0x80, 0x30, 0x38, 0x40, 0xb0, 0xb8, 0xc0,
    };
    const uint64_t row_bytes = ds4_src_f8_e4m3_b128_row_bytes(cols);
    memset(row, 0, (size_t)row_bytes);
    for (uint32_t b = 0; b < cols / DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS; b++) {
        uint8_t *block = row + (uint64_t)b * DS4_SRC_F8_E4M3_B128_BLOCK_BYTES;
        block[0] = (uint8_t)(120u + ((r + b) & 3u));
        for (uint32_t i = 0; i < DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS; i++) {
            block[1u + i] = codes[(r + b + i) & 7u];
        }
    }
}

int main(void) {
    enum {
        ROWS = 4096,
        COLS = 2048,
        TOKENS = 8,
    };

    if (ds4_gpu_device_count() < 1) {
        fprintf(stderr, "cuda_f8_hmma_shared_down_smoke: no CUDA devices visible\n");
        return 1;
    }

    const uint64_t row_bytes = ds4_src_f8_e4m3_b128_row_bytes(COLS);
    const uint64_t payload_bytes = (uint64_t)ROWS * row_bytes;
    uint8_t *payload = (uint8_t *)malloc((size_t)payload_bytes);
    float *x = (float *)malloc((uint64_t)TOKENS * COLS * sizeof(float));
    float *scalar = (float *)malloc((uint64_t)TOKENS * ROWS * sizeof(float));
    float *hmma = (float *)malloc((uint64_t)TOKENS * ROWS * sizeof(float));
    check(payload && x && scalar && hmma, "host buffers allocate");
    if (!payload || !x || !scalar || !hmma) return 1;

    for (uint32_t r = 0; r < ROWS; r++) {
        fill_f8_row(payload + (uint64_t)r * row_bytes, COLS, r);
    }
    for (uint32_t t = 0; t < TOKENS; t++) {
        for (uint32_t c = 0; c < COLS; c++) {
            x[(uint64_t)t * COLS + c] =
                (float)((int)((c + 13u * t) % 17u) - 8) * 0.00390625f;
        }
    }

    ds4_gpu_arena *arena = NULL;
    check(ds4_gpu_arena_open(&arena, 0, payload_bytes) == 0, "arena open");
    if (!arena) return 1;
    check(ds4_gpu_arena_upload(arena, 0, payload, payload_bytes) == 0,
          "arena upload");

    ds4_gpu_source_row_view view = {
        .arena_offset = 0,
        .byte_length = payload_bytes,
        .rows = ROWS,
        .cols = COLS,
        .row_stride_bytes = (uint32_t)row_bytes,
    };

    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc((uint64_t)TOKENS * COLS * sizeof(float));
    ds4_gpu_tensor *scalar_t = ds4_gpu_tensor_alloc((uint64_t)TOKENS * ROWS * sizeof(float));
    ds4_gpu_tensor *hmma_t = ds4_gpu_tensor_alloc((uint64_t)TOKENS * ROWS * sizeof(float));
    check(x_t && scalar_t && hmma_t, "device tensors allocate");
    if (x_t && scalar_t && hmma_t) {
        check(ds4_gpu_tensor_write(x_t, 0, x, (uint64_t)TOKENS * COLS * sizeof(float)),
              "x upload");
        unsetenv("DS4_CUDA_F8_F16_CACHE");
        setenv("DS4_CUDA_F8_ROWPAIR", "1", 1);
        setenv("DS4_CUDA_F8_HMMA_SHARED_DOWN", "0", 1);
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_batch_f32(arena,
                                                           &view,
                                                           x_t,
                                                           TOKENS,
                                                           scalar_t) == 0,
              "scalar f8 batch matmul");
        setenv("DS4_CUDA_F8_HMMA_SHARED_DOWN", "1", 1);
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_batch_f32(arena,
                                                           &view,
                                                           x_t,
                                                           TOKENS,
                                                           hmma_t) == 0,
              "hmma f8 batch matmul");
        check(ds4_gpu_tensor_read(scalar_t,
                                  0,
                                  scalar,
                                  (uint64_t)TOKENS * ROWS * sizeof(float)),
              "scalar output read");
        check(ds4_gpu_tensor_read(hmma_t,
                                  0,
                                  hmma,
                                  (uint64_t)TOKENS * ROWS * sizeof(float)),
              "hmma output read");

        float max_abs = 0.0f;
        float max_rel = 0.0f;
        uint64_t max_i = 0;
        for (uint64_t i = 0; i < (uint64_t)TOKENS * ROWS; i++) {
            const float diff = fabsf(hmma[i] - scalar[i]);
            const float denom = fmaxf(fabsf(scalar[i]), 1.0e-6f);
            const float rel = diff / denom;
            if (diff > max_abs) {
                max_abs = diff;
                max_rel = rel;
                max_i = i;
            }
        }
        if (max_abs > 0.05f && max_rel > 0.05f) {
            fprintf(stderr,
                    "cuda_f8_hmma_shared_down_smoke: max diff %.8g rel %.8g at token %llu row %llu scalar %.8g hmma %.8g\n",
                    max_abs,
                    max_rel,
                    (unsigned long long)(max_i / ROWS),
                    (unsigned long long)(max_i % ROWS),
                    scalar[max_i],
                    hmma[max_i]);
            failures++;
        }
    }

    ds4_gpu_tensor_free(hmma_t);
    ds4_gpu_tensor_free(scalar_t);
    ds4_gpu_tensor_free(x_t);
    ds4_gpu_arena_close(arena);
    free(hmma);
    free(scalar);
    free(x);
    free(payload);

    if (failures) return 1;
    puts("cuda_f8_hmma_shared_down_smoke: ok");
    return 0;
}
