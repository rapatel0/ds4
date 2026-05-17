#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "bf16_probe_smoke: %s\n", msg);
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
                "bf16_probe_smoke: %s got 0x%08x expected 0x%08x\n",
                label,
                f32_bits(f),
                bf16_expected_bits(bf16));
        failures++;
    }
}

static void upload_or_die(ds4_gpu_arena *arena, uint64_t offset,
                          const void *src, uint64_t bytes) {
    if (ds4_gpu_arena_upload(arena, offset, src, bytes) != 0) {
        fprintf(stderr, "bf16_probe_smoke: arena upload failed\n");
        failures++;
    }
}

int main(void) {
    enum {
        ROWS = 4,
        COLS = 4,
        STRIDE = 5,
        OFFSET = 2
    };

    const uint16_t matrix[ROWS * STRIDE] = {
        0x0000, 0x8000, 0x3f80, 0xbf80, 0xdead,
        0x4000, 0x7f80, 0xff80, 0x7fc0, 0xbeef,
        0x3c23, 0x3dcd, 0x3e4d, 0x3f00, 0xabcd,
        0x4120, 0xc120, 0x3f80, 0x0001, 0x1234,
    };
    const uint64_t matrix_bytes = sizeof(matrix);
    unsigned char padding[OFFSET + sizeof(matrix)];
    memset(padding, 0xa5, sizeof(padding));
    memcpy(padding + OFFSET, matrix, sizeof(matrix));

    ds4_gpu_arena *arena = NULL;
    check(ds4_gpu_arena_open(&arena, 0, sizeof(padding)) == 0, "arena open");
    if (!arena) return 1;
    upload_or_die(arena, 0, padding, sizeof(padding));

    ds4_gpu_bf16_matrix_view view = {
        .arena_offset = OFFSET,
        .byte_length = matrix_bytes,
        .rows = ROWS,
        .cols = COLS,
        .row_stride_elements = STRIDE,
    };

    uint32_t rows[] = {0, 3, 1, 1};
    float out[sizeof(rows) / sizeof(rows[0]) * COLS];
    memset(out, 0, sizeof(out));

    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &view, rows, 4,
                                            out, sizeof(out)) == 0,
          "row gather failed");
    expect_bits(out[0], 0x0000, "row0 col0");
    expect_bits(out[1], 0x8000, "negative zero");
    expect_bits(out[2], 0x3f80, "one");
    expect_bits(out[3], 0xbf80, "minus one");
    expect_bits(out[4], 0x4120, "last row first col");
    expect_bits(out[5], 0xc120, "last row second col");
    expect_bits(out[6], 0x3f80, "last row repeated one");
    expect_bits(out[7], 0x0001, "subnormal");
    expect_bits(out[8], 0x4000, "two");
    expect_bits(out[9], 0x7f80, "positive infinity");
    expect_bits(out[10], 0xff80, "negative infinity");
    expect_bits(out[11], 0x7fc0, "nan payload");
    expect_bits(out[12], 0x4000, "repeated row");
    expect_bits(out[13], 0x7f80, "repeated row inf");

    if (f32_bits(out[2]) == 0x3f800000u) {
        /* IEEE F16 0x3f80 is not 1.0f; this catches F16 reinterpretation. */
        check(1, "bf16-vs-f16 divergence");
    } else {
        check(0, "bf16-vs-f16 divergence");
    }

    uint32_t bad_row[] = {ROWS};
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &view, bad_row, 1,
                                            out, sizeof(out)) != 0,
          "accepted invalid row id");
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &view, rows, 0,
                                            out, sizeof(out)) != 0,
          "accepted zero rows");
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &view, rows, 1,
                                            out, sizeof(float) * (COLS - 1)) != 0,
          "accepted undersized output");

    ds4_gpu_bf16_matrix_view bad = view;
    bad.arena_offset = 1;
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &bad, rows, 1,
                                            out, sizeof(out)) != 0,
          "accepted odd arena offset");
    bad = view;
    bad.byte_length = matrix_bytes - 1;
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &bad, rows, 1,
                                            out, sizeof(out)) != 0,
          "accepted odd byte length");
    bad = view;
    bad.row_stride_elements = COLS - 1;
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &bad, rows, 1,
                                            out, sizeof(out)) != 0,
          "accepted short stride");
    bad = view;
    bad.cols = 0;
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &bad, rows, 1,
                                            out, sizeof(out)) != 0,
          "accepted zero columns");
    bad = view;
    bad.byte_length = matrix_bytes - 4;
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &bad, rows, 1,
                                            out, sizeof(out)) != 0,
          "accepted truncated view");

    check(ds4_gpu_arena_bf16_row_gather_f32(NULL, &view, rows, 1,
                                            out, sizeof(out)) != 0,
          "accepted null arena");
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, NULL, rows, 1,
                                            out, sizeof(out)) != 0,
          "accepted null view");
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &view, NULL, 1,
                                            out, sizeof(out)) != 0,
          "accepted null rows");
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &view, rows, 1,
                                            NULL, sizeof(out)) != 0,
          "accepted null output");

    unsigned char tiny[2] = {0, 0};
    check(ds4_gpu_arena_upload(arena, sizeof(padding) - 1, tiny, sizeof(tiny)) != 0,
          "out-of-range upload unexpectedly succeeded");
    check(ds4_gpu_arena_bf16_row_gather_f32(arena, &view, rows, 1,
                                            out, sizeof(out)) != 0,
          "accepted invalid arena");

    ds4_gpu_arena_close(arena);
    if (failures) return 1;
    puts("bf16_probe_smoke: ok");
    return 0;
}
