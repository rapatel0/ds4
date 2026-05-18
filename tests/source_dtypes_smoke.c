#include "ds4_source_formats.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures;

static uint32_t f32_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "source_dtypes_smoke: %s\n", msg);
        failures++;
    }
}

static void expect_bits(float got, uint32_t want, const char *label) {
    if (f32_bits(got) != want) {
        fprintf(stderr,
                "source_dtypes_smoke: %s got 0x%08x expected 0x%08x\n",
                label,
                f32_bits(got),
                want);
        failures++;
    }
}

static void expect_close(float got, float want, const char *label) {
    const float diff = fabsf(got - want);
    if (diff > 1e-6f) {
        fprintf(stderr,
                "source_dtypes_smoke: %s got %.9g expected %.9g\n",
                label,
                got,
                want);
        failures++;
    }
}

static void test_scalar_formats(void) {
    expect_bits(ds4_src_bf16_to_f32(0x3f80), 0x3f800000u, "bf16 one");
    expect_bits(ds4_src_bf16_to_f32(0x8000), 0x80000000u, "bf16 negative zero");
    expect_bits(ds4_src_bf16_to_f32(0x7fc0), 0x7fc00000u, "bf16 nan payload");

    expect_bits(ds4_src_e8m0_to_f32(0), 0x00400000u, "e8m0 zero exponent sentinel");
    expect_bits(ds4_src_e8m0_to_f32(126), 0x3f000000u, "e8m0 half");
    expect_bits(ds4_src_e8m0_to_f32(127), 0x3f800000u, "e8m0 one");
    expect_bits(ds4_src_e8m0_to_f32(128), 0x40000000u, "e8m0 two");

    expect_bits(ds4_src_e4m3fn_to_f32(0x00), 0x00000000u, "e4m3 positive zero");
    expect_bits(ds4_src_e4m3fn_to_f32(0x80), 0x80000000u, "e4m3 negative zero");
    expect_close(ds4_src_e4m3fn_to_f32(0x01), 0x1.0p-9f, "e4m3 subnormal");
    expect_close(ds4_src_e4m3fn_to_f32(0x38), 1.0f, "e4m3 one");
    expect_close(ds4_src_e4m3fn_to_f32(0x3c), 1.5f, "e4m3 one point five");
    expect_close(ds4_src_e4m3fn_to_f32(0x40), 2.0f, "e4m3 two");
    expect_close(ds4_src_e4m3fn_to_f32(0xb8), -1.0f, "e4m3 negative one");
    expect_bits(ds4_src_e4m3fn_to_f32(0x7f), 0x00000000u, "e4m3 fn nan maps zero");

    expect_close(ds4_src_mxfp4_nibble_to_f32(0x0), 0.0f, "mxfp4 zero");
    expect_close(ds4_src_mxfp4_nibble_to_f32(0x1), 0.5f, "mxfp4 half");
    expect_close(ds4_src_mxfp4_nibble_to_f32(0x7), 6.0f, "mxfp4 max");
    expect_close(ds4_src_mxfp4_nibble_to_f32(0x8), 0.0f, "mxfp4 negative zero code maps zero");
    expect_close(ds4_src_mxfp4_nibble_to_f32(0xf), -6.0f, "mxfp4 negative max");
}

static void test_bf16_row(void) {
    const uint16_t src[] = {0x3f80, 0xbf80, 0x4000, 0x0001};
    float dst[4] = {0};
    check(ds4_src_bf16_row_to_f32(dst, src, 4) == 0, "bf16 row decode");
    expect_bits(dst[0], 0x3f800000u, "bf16 row one");
    expect_bits(dst[1], 0xbf800000u, "bf16 row negative one");
    expect_bits(dst[2], 0x40000000u, "bf16 row two");
    expect_bits(dst[3], 0x00010000u, "bf16 row subnormal");
    check(ds4_src_bf16_row_to_f32(dst, src, 0) != 0, "bf16 accepted zero length");
}

static void test_f8_row(void) {
    uint8_t row[DS4_SRC_F8_E4M3_B128_BLOCK_BYTES * 2];
    memset(row, 0, sizeof(row));
    row[0] = 127;
    row[1] = 0x38;
    row[2] = 0x3c;
    row[3] = 0xb8;
    row[DS4_SRC_F8_E4M3_B128_BLOCK_BYTES] = 128;
    row[DS4_SRC_F8_E4M3_B128_BLOCK_BYTES + 1] = 0x38;
    row[DS4_SRC_F8_E4M3_B128_BLOCK_BYTES + 2] = 0x40;
    row[DS4_SRC_F8_E4M3_B128_BLOCK_BYTES + 3] = 0x01;

    float dst[DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS * 2];
    char err[128] = {0};
    check(ds4_src_f8_e4m3_b128_row_bytes(256) == sizeof(row), "f8 row bytes");
    check(ds4_src_f8_e4m3_b128_row_to_f32(dst, row, 256, err, sizeof(err)) == 0,
          "f8 row decode");
    expect_close(dst[0], 1.0f, "f8 block0 one");
    expect_close(dst[1], 1.5f, "f8 block0 one point five");
    expect_close(dst[2], -1.0f, "f8 block0 negative one");
    expect_close(dst[128], 2.0f, "f8 block1 scaled one");
    expect_close(dst[129], 4.0f, "f8 block1 scaled two");
    expect_close(dst[130], 0x1.0p-8f, "f8 block1 scaled subnormal");
    float x[DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS * 2];
    memset(x, 0, sizeof(x));
    x[0] = 2.0f;
    x[1] = 4.0f;
    x[128] = -1.0f;
    float dot = 0.0f;
    check(ds4_src_f8_e4m3_b128_row_dot(&dot, row, x, 256, err, sizeof(err)) == 0,
          "f8 row dot");
    expect_close(dot, 6.0f, "f8 dot value");
    check(ds4_src_f8_e4m3_b128_row_to_f32(dst, row, 129, err, sizeof(err)) != 0,
          "f8 accepted misaligned row");
    check(err[0] != '\0', "f8 error message");
}

static void test_mxfp4_row(void) {
    uint8_t row[DS4_SRC_MXFP4_BLOCK_BYTES * 2];
    memset(row, 0, sizeof(row));
    row[0] = 127;
    row[1] = 0x21;
    row[2] = 0xf7;
    row[DS4_SRC_MXFP4_BLOCK_BYTES] = 128;
    row[DS4_SRC_MXFP4_BLOCK_BYTES + 1] = 0x43;
    row[DS4_SRC_MXFP4_BLOCK_BYTES + 2] = 0xa8;

    float dst[DS4_SRC_MXFP4_BLOCK_ELEMS * 2];
    char err[128] = {0};
    check(ds4_src_mxfp4_row_bytes(64) == sizeof(row), "mxfp4 row bytes");
    check(ds4_src_mxfp4_row_to_f32(dst, row, 64, err, sizeof(err)) == 0,
          "mxfp4 row decode");
    expect_close(dst[0], 0.5f, "mxfp4 low nibble first");
    expect_close(dst[1], 1.0f, "mxfp4 high nibble second");
    expect_close(dst[2], 6.0f, "mxfp4 positive max");
    expect_close(dst[3], -6.0f, "mxfp4 negative max");
    expect_close(dst[32], 3.0f, "mxfp4 block1 scaled low");
    expect_close(dst[33], 4.0f, "mxfp4 block1 scaled high");
    expect_close(dst[34], 0.0f, "mxfp4 block1 zero code");
    expect_close(dst[35], -2.0f, "mxfp4 block1 negative one scaled");
    float x[DS4_SRC_MXFP4_BLOCK_ELEMS * 2];
    memset(x, 0, sizeof(x));
    x[0] = 2.0f;
    x[1] = 3.0f;
    x[32] = -1.0f;
    float dot = 0.0f;
    check(ds4_src_mxfp4_row_dot(&dot, row, x, 64, err, sizeof(err)) == 0,
          "mxfp4 row dot");
    expect_close(dot, 1.0f + 3.0f - 3.0f, "mxfp4 dot value");
    check(ds4_src_mxfp4_row_to_f32(dst, row, 33, err, sizeof(err)) != 0,
          "mxfp4 accepted misaligned row");
    check(err[0] != '\0', "mxfp4 error message");
}

static void test_i32_row(void) {
    const int32_t src[] = {0, 7, 255};
    uint32_t dst[3] = {0};
    char err[128] = {0};
    check(ds4_src_i32_row_to_u32(dst, src, 3, err, sizeof(err)) == 0, "i32 row");
    check(dst[0] == 0 && dst[1] == 7 && dst[2] == 255, "i32 values");

    const int32_t bad[] = {-1};
    check(ds4_src_i32_row_to_u32(dst, bad, 1, err, sizeof(err)) != 0,
          "i32 accepted negative routing id");
    check(err[0] != '\0', "i32 error message");
}

int main(void) {
    test_scalar_formats();
    test_bf16_row();
    test_f8_row();
    test_mxfp4_row();
    test_i32_row();

    if (failures) {
        fprintf(stderr, "source_dtypes_smoke: %d failures\n", failures);
        return 1;
    }
    printf("source_dtypes_smoke: ok\n");
    return 0;
}
