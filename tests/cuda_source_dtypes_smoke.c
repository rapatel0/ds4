#include "ds4_gpu.h"
#include "ds4_source_formats.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_source_dtypes_smoke: %s\n", msg);
        failures++;
    }
}

static uint32_t f32_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

static void expect_same_f32(float got, float want, const char *label) {
    if (f32_bits(got) != f32_bits(want)) {
        fprintf(stderr,
                "cuda_source_dtypes_smoke: %s got 0x%08x expected 0x%08x\n",
                label,
                f32_bits(got),
                f32_bits(want));
        failures++;
    }
}

static void expect_close_f32(float got, float want, float tol, const char *label) {
    if (fabsf(got - want) > tol) {
        fprintf(stderr,
                "cuda_source_dtypes_smoke: %s got %.8g expected %.8g\n",
                label,
                got,
                want);
        failures++;
    }
}

static void fill_f8_row(uint8_t *row, uint32_t cols, uint8_t scale0) {
    static const uint8_t codes[] = {
        0x00, 0x80, 0x01, 0x38, 0x3c, 0x40, 0xb8, 0xc0,
    };
    const uint64_t row_bytes = ds4_src_f8_e4m3_b128_row_bytes(cols);
    memset(row, 0xa5, (size_t)row_bytes);
    for (uint32_t b = 0; b < cols / DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS; b++) {
        uint8_t *block = row + (uint64_t)b * DS4_SRC_F8_E4M3_B128_BLOCK_BYTES;
        block[0] = (uint8_t)(scale0 + b);
        for (uint32_t i = 0; i < DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS; i++) {
            block[1 + i] = codes[(i + b) % (sizeof(codes) / sizeof(codes[0]))];
        }
    }
}

int main(void) {
    enum {
        ROWS = 3,
        COLS = 256,
        OFFSET = 5,
    };
    const uint64_t row_bytes = ds4_src_f8_e4m3_b128_row_bytes(COLS);
    const uint32_t stride = (uint32_t)(row_bytes + 7);
    const uint64_t span = (uint64_t)ROWS * stride;

    int devices = ds4_gpu_device_count();
    if (devices < 1) {
        fprintf(stderr, "cuda_source_dtypes_smoke: no CUDA devices visible\n");
        return 1;
    }

    uint8_t payload[OFFSET + ROWS * (int)(DS4_SRC_F8_E4M3_B128_BLOCK_BYTES * 2 + 7)];
    memset(payload, 0x5c, sizeof(payload));
    for (uint32_t r = 0; r < ROWS; r++) {
        fill_f8_row(payload + OFFSET + (uint64_t)r * stride, COLS, (uint8_t)(126 + r));
    }

    ds4_gpu_arena *arena = NULL;
    check(ds4_gpu_arena_open(&arena, 0, sizeof(payload)) == 0, "arena open");
    if (!arena) return 1;
    check(ds4_gpu_arena_is_device_memory(arena), "arena is device memory");
    check(ds4_gpu_arena_upload(arena, 0, payload, sizeof(payload)) == 0,
          "arena upload");

    ds4_gpu_source_row_view view = {
        .arena_offset = OFFSET,
        .byte_length = span,
        .rows = ROWS,
        .cols = COLS,
        .row_stride_bytes = stride,
    };
    uint32_t rows[] = {2, 0};
    float got[(sizeof(rows) / sizeof(rows[0])) * COLS];
    float ref[(sizeof(rows) / sizeof(rows[0])) * COLS];
    memset(got, 0, sizeof(got));
    memset(ref, 0, sizeof(ref));

    for (uint32_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) {
        char err[128] = {0};
        const uint8_t *row = payload + OFFSET + (uint64_t)rows[i] * stride;
        check(ds4_src_f8_e4m3_b128_validate_row_span(COLS, row_bytes, err, sizeof(err)) == 0,
              "reference f8 span");
        check(ds4_src_f8_e4m3_b128_row_to_f32(ref + (uint64_t)i * COLS,
                                              row,
                                              COLS,
                                              err,
                                              sizeof(err)) == 0,
              "reference f8 decode");
    }

    check(ds4_gpu_arena_f8_e4m3_b128_row_decode_f32(arena, &view, rows, 2,
                                                     got, sizeof(got)) == 0,
          "cuda f8 row decode");
    for (uint32_t i = 0; i < sizeof(got) / sizeof(got[0]); i++) {
        expect_same_f32(got[i], ref[i], "f8 decode value");
    }

    float x[COLS];
    for (uint32_t i = 0; i < COLS; i++) {
        x[i] = (float)((int)(i % 23) - 11) * 0.03125f;
    }
    float ref_dot[ROWS];
    for (uint32_t r = 0; r < ROWS; r++) {
        char err[128] = {0};
        const uint8_t *row = payload + OFFSET + (uint64_t)r * stride;
        check(ds4_src_f8_e4m3_b128_row_dot(&ref_dot[r], row, x, COLS,
                                           err, sizeof(err)) == 0,
              "reference f8 dot");
    }
    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(sizeof(ref_dot));
    check(x_t && out_t, "f8 matmul tensors allocate");
    if (x_t && out_t) {
        float out_dot[ROWS];
        memset(out_dot, 0, sizeof(out_dot));
        check(ds4_gpu_tensor_write(x_t, 0, x, sizeof(x)), "f8 matmul x upload");
        check(ds4_gpu_arena_f8_e4m3_b128_matmul_f32(arena, &view, x_t, out_t) == 0,
              "cuda f8 matmul");
        check(ds4_gpu_tensor_read(out_t, 0, out_dot, sizeof(out_dot)),
              "f8 matmul output read");
        for (uint32_t r = 0; r < ROWS; r++) {
            expect_close_f32(out_dot[r], ref_dot[r], 1e-5f, "f8 matmul row");
        }
    }
    uint32_t bad_row[] = {ROWS};
    check(ds4_gpu_arena_f8_e4m3_b128_row_decode_f32(arena, &view, bad_row, 1,
                                                     got, sizeof(got)) != 0,
          "accepted invalid f8 row id");
    check(ds4_gpu_arena_f8_e4m3_b128_row_decode_f32(arena, &view, rows, 2,
                                                     got, sizeof(float) * (COLS - 1)) != 0,
          "accepted undersized f8 output");

    ds4_gpu_source_row_view bad = view;
    bad.cols = 129;
    check(ds4_gpu_arena_f8_e4m3_b128_row_decode_f32(arena, &bad, rows, 1,
                                                     got, sizeof(got)) != 0,
          "accepted misaligned f8 column count");

    bad = view;
    bad.row_stride_bytes = (uint32_t)(row_bytes - 1);
    check(ds4_gpu_arena_f8_e4m3_b128_row_decode_f32(arena, &bad, rows, 1,
                                                     got, sizeof(got)) != 0,
          "accepted undersized f8 row stride");

    bad = view;
    bad.byte_length = (uint64_t)(ROWS - 1) * stride + row_bytes - 1;
    check(ds4_gpu_arena_f8_e4m3_b128_row_decode_f32(arena, &bad, rows, 1,
                                                     got, sizeof(got)) != 0,
          "accepted truncated f8 view span");
    check(ds4_gpu_arena_f8_e4m3_b128_matmul_f32(arena, &bad, x_t, out_t) != 0,
          "accepted truncated f8 matmul view");

    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(x_t);
    ds4_gpu_arena_close(arena);
    if (failures) return 1;
    puts("cuda_source_dtypes_smoke: ok");
    return 0;
}
