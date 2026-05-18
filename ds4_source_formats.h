#ifndef DS4_SOURCE_FORMATS_H
#define DS4_SOURCE_FORMATS_H

#include <stddef.h>
#include <stdint.h>

enum {
    DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS = 128,
    DS4_SRC_F8_E4M3_B128_BLOCK_BYTES = 129,
    DS4_SRC_MXFP4_BLOCK_ELEMS = 32,
    DS4_SRC_MXFP4_BLOCK_BYTES = 17,
};

float ds4_src_bf16_to_f32(uint16_t bits);
float ds4_src_e8m0_to_f32(uint8_t e);
float ds4_src_e4m3fn_to_f32(uint8_t x);
float ds4_src_mxfp4_nibble_to_f32(uint8_t q);

int ds4_src_bf16_row_to_f32(float *dst, const uint16_t *src, uint64_t n);
int ds4_src_f8_e4m3_b128_row_to_f32(float *dst, const uint8_t *src,
                                    uint64_t ncols, char *err, size_t err_size);
int ds4_src_mxfp4_row_to_f32(float *dst, const uint8_t *src, uint64_t ncols,
                             char *err, size_t err_size);
int ds4_src_i32_row_to_u32(uint32_t *dst, const int32_t *src, uint64_t n,
                           char *err, size_t err_size);

uint64_t ds4_src_f8_e4m3_b128_row_bytes(uint64_t ncols);
uint64_t ds4_src_mxfp4_row_bytes(uint64_t ncols);

#endif
