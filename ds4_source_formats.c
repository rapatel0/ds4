#include "ds4_source_formats.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static float f32_from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void set_err(char *err, size_t err_size, const char *msg) {
    if (!err || err_size == 0) return;
    snprintf(err, err_size, "%s", msg);
}

float ds4_src_bf16_to_f32(uint16_t bits) {
    return f32_from_bits((uint32_t)bits << 16);
}

float ds4_src_e8m0_to_f32(uint8_t e) {
    const uint32_t bits = e == 0 ? 0x00400000u : ((uint32_t)e << 23);
    return f32_from_bits(bits);
}

float ds4_src_e4m3fn_to_f32(uint8_t x) {
    const uint8_t abs = x & 0x7f;
    const bool sign = (x & 0x80) != 0;
    if (abs == 0) return f32_from_bits(sign ? 0x80000000u : 0u);
    if (abs == 0x7f) return 0.0f;

    const int exp = (x >> 3) & 0x0f;
    const int man = x & 0x07;
    float value = exp == 0 ? ldexpf((float)man, -9)
                           : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    return sign ? -value : value;
}

float ds4_src_mxfp4_nibble_to_f32(uint8_t q) {
    static const float fp4_table[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return fp4_table[q & 0x0f];
}

uint64_t ds4_src_f8_e4m3_b128_row_bytes(uint64_t ncols) {
    if (ncols % DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS) return 0;
    return (ncols / DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS) *
           DS4_SRC_F8_E4M3_B128_BLOCK_BYTES;
}

uint64_t ds4_src_mxfp4_row_bytes(uint64_t ncols) {
    if (ncols % DS4_SRC_MXFP4_BLOCK_ELEMS) return 0;
    return (ncols / DS4_SRC_MXFP4_BLOCK_ELEMS) * DS4_SRC_MXFP4_BLOCK_BYTES;
}

int ds4_src_bf16_row_to_f32(float *dst, const uint16_t *src, uint64_t n) {
    if (!dst || !src || n == 0) return -1;
    for (uint64_t i = 0; i < n; i++) dst[i] = ds4_src_bf16_to_f32(src[i]);
    return 0;
}

int ds4_src_f8_e4m3_b128_row_to_f32(float *dst, const uint8_t *src,
                                    uint64_t ncols, char *err, size_t err_size) {
    if (!dst || !src) {
        set_err(err, err_size, "null f8 row buffer");
        return -1;
    }
    if (ncols == 0 || ncols % DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS) {
        set_err(err, err_size, "f8 row length is not a nonzero multiple of 128");
        return -1;
    }

    const uint64_t nblocks = ncols / DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS;
    for (uint64_t b = 0; b < nblocks; b++) {
        const uint8_t *block = src + b * DS4_SRC_F8_E4M3_B128_BLOCK_BYTES;
        const float scale = ds4_src_e8m0_to_f32(block[0]);
        const uint8_t *qs = block + 1;
        float *out = dst + b * DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS;
        for (uint64_t i = 0; i < DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS; i++) {
            out[i] = ds4_src_e4m3fn_to_f32(qs[i]) * scale;
        }
    }
    return 0;
}

int ds4_src_mxfp4_row_to_f32(float *dst, const uint8_t *src, uint64_t ncols,
                             char *err, size_t err_size) {
    if (!dst || !src) {
        set_err(err, err_size, "null mxfp4 row buffer");
        return -1;
    }
    if (ncols == 0 || ncols % DS4_SRC_MXFP4_BLOCK_ELEMS) {
        set_err(err, err_size, "mxfp4 row length is not a nonzero multiple of 32");
        return -1;
    }

    const uint64_t nblocks = ncols / DS4_SRC_MXFP4_BLOCK_ELEMS;
    for (uint64_t b = 0; b < nblocks; b++) {
        const uint8_t *block = src + b * DS4_SRC_MXFP4_BLOCK_BYTES;
        const float scale = ds4_src_e8m0_to_f32(block[0]);
        const uint8_t *qs = block + 1;
        float *out = dst + b * DS4_SRC_MXFP4_BLOCK_ELEMS;
        for (uint64_t j = 0; j < DS4_SRC_MXFP4_BLOCK_ELEMS / 2; j++) {
            const uint8_t q = qs[j];
            out[2 * j + 0] = ds4_src_mxfp4_nibble_to_f32(q & 0x0f) * scale;
            out[2 * j + 1] = ds4_src_mxfp4_nibble_to_f32((q >> 4) & 0x0f) * scale;
        }
    }
    return 0;
}

int ds4_src_i32_row_to_u32(uint32_t *dst, const int32_t *src, uint64_t n,
                           char *err, size_t err_size) {
    if (!dst || !src) {
        set_err(err, err_size, "null i32 row buffer");
        return -1;
    }
    if (n == 0) {
        set_err(err, err_size, "i32 row length is zero");
        return -1;
    }
    for (uint64_t i = 0; i < n; i++) {
        if (src[i] < 0) {
            set_err(err, err_size, "negative i32 routing id");
            return -1;
        }
        dst[i] = (uint32_t)src[i];
    }
    return 0;
}
