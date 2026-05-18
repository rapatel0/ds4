#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_bf16_probe: %s\n", msg);
        failures++;
    }
}

static uint32_t f32_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

static uint32_t bf16_expected_bits(uint16_t bf16) {
    return (uint32_t)bf16 << 16;
}

static void expect_bits(float f, uint16_t bf16, const char *label) {
    if (f32_bits(f) != bf16_expected_bits(bf16)) {
        fprintf(stderr,
                "cuda_bf16_probe: %s got 0x%08x expected 0x%08x\n",
                label,
                f32_bits(f),
                bf16_expected_bits(bf16));
        failures++;
    }
}

int main(void) {
    enum {
        ROWS = 3,
        COLS = 4,
        STRIDE = 5,
        OFFSET = 2
    };

    int devices = ds4_gpu_device_count();
    if (devices < 1) {
        fprintf(stderr, "cuda_bf16_probe: no CUDA devices visible\n");
        return 1;
    }

    const uint16_t matrix[ROWS * STRIDE] = {
        0x0000, 0x8000, 0x3f80, 0xbf80, 0x1111,
        0x4000, 0x7f80, 0xff80, 0x7fc0, 0x2222,
        0x4120, 0xc120, 0x3f80, 0x0001, 0x3333,
    };
    unsigned char payload[OFFSET + sizeof(matrix)];
    memset(payload, 0xa5, sizeof(payload));
    memcpy(payload + OFFSET, matrix, sizeof(matrix));

    ds4_gpu_arena *arena = NULL;
    check(ds4_gpu_arena_open(&arena, 0, sizeof(payload)) == 0, "arena open");
    if (!arena) return 1;
    check(ds4_gpu_arena_is_device_memory(arena), "arena is device memory");
    check(ds4_gpu_arena_upload(arena, 0, payload, sizeof(payload)) == 0,
          "arena upload");

    ds4_gpu_bf16_matrix_view view = {
        .arena_offset = OFFSET,
        .byte_length = sizeof(matrix),
        .rows = ROWS,
        .cols = COLS,
        .row_stride_elements = STRIDE,
    };
    uint32_t rows[] = {0, 2, 1};
    float out[sizeof(rows) / sizeof(rows[0]) * COLS];
    memset(out, 0, sizeof(out));

    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &view, rows, 3,
                                            out, sizeof(out)) == 0,
          "bf16 gather");
    expect_bits(out[0], 0x0000, "row0 col0");
    expect_bits(out[1], 0x8000, "negative zero");
    expect_bits(out[2], 0x3f80, "one");
    expect_bits(out[3], 0xbf80, "minus one");
    expect_bits(out[4], 0x4120, "row2 col0");
    expect_bits(out[5], 0xc120, "row2 col1");
    expect_bits(out[6], 0x3f80, "row2 one");
    expect_bits(out[7], 0x0001, "row2 subnormal");
    expect_bits(out[8], 0x4000, "row1 two");
    expect_bits(out[9], 0x7f80, "row1 inf");
    expect_bits(out[10], 0xff80, "row1 minus inf");
    expect_bits(out[11], 0x7fc0, "row1 nan");

    uint32_t bad_row[] = {ROWS};
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &view, bad_row, 1,
                                            out, sizeof(out)) != 0,
          "accepted invalid row id");
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &view, rows, 1,
                                            out, sizeof(float) * (COLS - 1)) != 0,
          "accepted undersized output");

    ds4_gpu_bf16_matrix_view bad = view;
    bad.arena_offset = 1;
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &bad, rows, 1,
                                            out, sizeof(out)) != 0,
          "accepted odd offset");

    ds4_gpu_arena_close(arena);
    if (failures) return 1;
    puts("cuda_bf16_probe: ok");
    return 0;
}
