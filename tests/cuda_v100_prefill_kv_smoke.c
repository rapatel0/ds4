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
        fprintf(stderr, "cuda_v100_prefill_kv_smoke: %s\n", msg);
        failures++;
    }
}

static uint32_t f32_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

static float f32_from_bits(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t f32_to_f16_bits(float f) {
    const uint32_t x = f32_bits(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x007fffffu;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t out = mant >> shift;
        if ((mant >> (shift - 1u)) & 1u) out++;
        return (uint16_t)(sign | out);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t out = (uint32_t)exp << 10;
    mant += 0x00001000u;
    if (mant & 0x00800000u) {
        mant = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
        out = (uint32_t)exp << 10;
    }
    return (uint16_t)(sign | out | (mant >> 13));
}

static float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) return f32_from_bits(sign);
        exp = 1;
        while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x03ffu;
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return f32_from_bits(bits);
}

static void fill_f8_row(uint8_t *row, uint32_t cols, uint8_t scale0) {
    static const uint8_t codes[] = {
        0x00, 0x01, 0x30, 0x38, 0x3c, 0x40, 0xb8, 0xc0,
    };
    const uint64_t row_bytes = ds4_src_f8_e4m3_b128_row_bytes(cols);
    memset(row, 0, (size_t)row_bytes);
    for (uint32_t b = 0; b < cols / DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS; b++) {
        uint8_t *block = row + (uint64_t)b * DS4_SRC_F8_E4M3_B128_BLOCK_BYTES;
        block[0] = (uint8_t)(scale0 + b);
        for (uint32_t i = 0; i < DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS; i++) {
            block[1u + i] = codes[(i + b) % (sizeof(codes) / sizeof(codes[0]))];
        }
    }
}

static void expect_f16_row(const char *label,
                           const uint16_t *got,
                           const float *want,
                           uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const uint16_t want_bits = f32_to_f16_bits(want[i]);
        if (got[i] != want_bits) {
            fprintf(stderr,
                    "cuda_v100_prefill_kv_smoke: %s[%u] got 0x%04x %.8g expected 0x%04x %.8g\n",
                    label,
                    i,
                    got[i],
                    f16_bits_to_f32(got[i]),
                    want_bits,
                    f16_bits_to_f32(want_bits));
            failures++;
            return;
        }
    }
}

static void expect_attn_state(const char *label,
                              const float *got,
                              const float *row,
                              uint32_t values,
                              uint32_t head_dim,
                              uint32_t ratio) {
    for (uint32_t i = 0; i < values; i++) {
        const uint32_t lane = i % head_dim;
        const uint32_t state_row = i / head_dim;
        const float want = row[lane] + (float)state_row * 0.125f + (float)ratio * 0.001f;
        if (fabsf(got[i] - want) > 1e-5f) {
            fprintf(stderr,
                    "cuda_v100_prefill_kv_smoke: %s[%u] got %.8g expected %.8g\n",
                    label,
                    i,
                    got[i],
                    want);
            failures++;
            return;
        }
    }
}

static void expect_indexer_state(const char *label,
                                 const float *got,
                                 const float *row,
                                 uint32_t values,
                                 uint32_t dim,
                                 uint32_t ratio) {
    for (uint32_t i = 0; i < values; i++) {
        const uint32_t lane = i % dim;
        const uint32_t state_row = i / dim;
        const float want = row[lane] - (float)state_row * 0.0625f - (float)ratio * 0.001f;
        if (fabsf(got[i] - want) > 1e-5f) {
            fprintf(stderr,
                    "cuda_v100_prefill_kv_smoke: %s[%u] got %.8g expected %.8g\n",
                    label,
                    i,
                    got[i],
                    want);
            failures++;
            return;
        }
    }
}

static void read_f16_row(ds4_gpu_tensor *tensor,
                         uint64_t row,
                         uint32_t dim,
                         uint16_t *out) {
    const uint64_t off = row * dim * sizeof(uint16_t);
    check(ds4_gpu_tensor_read(tensor, off, out, (uint64_t)dim * sizeof(uint16_t)),
          "read f16 row");
}

int main(void) {
    enum {
        SLOTS = 2,
        RAW_ROWS = 128,
        COMP_ROWS = 16,
        HEAD_DIM = 512,
        INDEX_DIM = 128,
        RATIO4_ATTN_STATE = 8 * 1024,
        RATIO4_INDEX_STATE = 8 * 256,
        RATIO128_ATTN_STATE = 128 * 512,
        SOURCE_ROWS = 2,
        SOURCE_COLS = HEAD_DIM,
        SOURCE_OFFSET = 3,
    };

    if (ds4_gpu_device_count() < 1) {
        fprintf(stderr, "cuda_v100_prefill_kv_smoke: no CUDA devices visible\n");
        return 1;
    }

    const uint64_t source_row_bytes = ds4_src_f8_e4m3_b128_row_bytes(SOURCE_COLS);
    const uint32_t source_stride = (uint32_t)(source_row_bytes + 5u);
    uint8_t source_payload[SOURCE_OFFSET + SOURCE_ROWS *
                           (DS4_SRC_F8_E4M3_B128_BLOCK_BYTES * 4 + 5)];
    memset(source_payload, 0x6b, sizeof(source_payload));
    for (uint32_t r = 0; r < SOURCE_ROWS; r++) {
        fill_f8_row(source_payload + SOURCE_OFFSET + (uint64_t)r * source_stride,
                    SOURCE_COLS,
                    (uint8_t)(127u + r));
    }

    ds4_gpu_arena *source_arena = NULL;
    check(ds4_gpu_arena_open(&source_arena, 0, sizeof(source_payload)) == 0,
          "source arena open");
    check(source_arena != NULL, "source arena exists");
    if (!source_arena) return 1;
    check(ds4_gpu_arena_upload(source_arena, 0, source_payload, sizeof(source_payload)) == 0,
          "source arena upload");

    ds4_gpu_source_row_view source_view = {
        .arena_offset = SOURCE_OFFSET,
        .byte_length = (uint64_t)SOURCE_ROWS * source_stride,
        .rows = SOURCE_ROWS,
        .cols = SOURCE_COLS,
        .row_stride_bytes = source_stride,
    };
    uint32_t row_id[] = {1};
    float attn_row[HEAD_DIM];
    memset(attn_row, 0, sizeof(attn_row));
    check(ds4_gpu_arena_f8_e4m3_b128_row_decode_f32(source_arena, &source_view,
                                                     row_id, 1, attn_row,
                                                     sizeof(attn_row)) == 0,
          "source f8 decode for kv input");

    float ref_row[HEAD_DIM];
    char err[128] = {0};
    check(ds4_src_f8_e4m3_b128_row_to_f32(
              ref_row,
              source_payload + SOURCE_OFFSET + (uint64_t)row_id[0] * source_stride,
              SOURCE_COLS,
              err,
              sizeof(err)) == 0,
          "source f8 reference decode");
    for (uint32_t i = 0; i < HEAD_DIM; i++) {
        if (f32_bits(attn_row[i]) != f32_bits(ref_row[i])) {
            check(0, "source f8 bridge mismatch");
            break;
        }
    }

    float indexer_row[INDEX_DIM];
    for (uint32_t i = 0; i < INDEX_DIM; i++) indexer_row[i] = attn_row[i] * 0.5f;

    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc((uint64_t)SLOTS * RAW_ROWS * HEAD_DIM * sizeof(uint16_t));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc((uint64_t)SLOTS * COMP_ROWS * HEAD_DIM * sizeof(uint16_t));
    ds4_gpu_tensor *indexer = ds4_gpu_tensor_alloc((uint64_t)SLOTS * COMP_ROWS * INDEX_DIM * sizeof(uint16_t));
    ds4_gpu_tensor *attn_state = ds4_gpu_tensor_alloc((uint64_t)SLOTS * RATIO128_ATTN_STATE * sizeof(float));
    ds4_gpu_tensor *index_state = ds4_gpu_tensor_alloc((uint64_t)SLOTS * RATIO4_INDEX_STATE * sizeof(float));
    check(raw && comp && indexer && attn_state && index_state, "kv tensors allocate");

    ds4_gpu_v100_prefill_kv_update ratio128 = {
        .ratio = 128,
        .slot = 0,
        .slots = SLOTS,
        .raw_rows = RAW_ROWS,
        .raw_row = 7,
        .comp_rows = COMP_ROWS,
        .comp_row = 3,
        .head_dim = HEAD_DIM,
        .indexer_head_dim = 0,
        .attn_state_values = RATIO128_ATTN_STATE,
        .indexer_state_values = 0,
    };
    check(ds4_gpu_v100_prefill_kv_update_f16_tensor(raw, comp, NULL, attn_state,
                                                     NULL, attn_row, NULL,
                                                     &ratio128),
          "ratio128 kv update");

    uint16_t half_row[HEAD_DIM];
    read_f16_row(raw, (uint64_t)ratio128.slot * RAW_ROWS + ratio128.raw_row,
                 HEAD_DIM, half_row);
    expect_f16_row("ratio128 raw", half_row, attn_row, HEAD_DIM);
    read_f16_row(comp, (uint64_t)ratio128.slot * COMP_ROWS + ratio128.comp_row,
                 HEAD_DIM, half_row);
    expect_f16_row("ratio128 comp", half_row, attn_row, HEAD_DIM);

    float *state_read = (float *)malloc((size_t)RATIO128_ATTN_STATE * sizeof(float));
    check(state_read != NULL, "state read alloc");
    if (state_read) {
        check(ds4_gpu_tensor_read(attn_state,
                                  (uint64_t)ratio128.slot * RATIO128_ATTN_STATE * sizeof(float),
                                  state_read,
                                  (uint64_t)RATIO128_ATTN_STATE * sizeof(float)),
              "ratio128 attn state read");
        expect_attn_state("ratio128 attn state", state_read, attn_row,
                          RATIO128_ATTN_STATE, HEAD_DIM, ratio128.ratio);
    }

    ds4_gpu_v100_prefill_kv_update ratio4 = {
        .ratio = 4,
        .slot = 1,
        .slots = SLOTS,
        .raw_rows = RAW_ROWS,
        .raw_row = 9,
        .comp_rows = COMP_ROWS,
        .comp_row = 5,
        .head_dim = HEAD_DIM,
        .indexer_head_dim = INDEX_DIM,
        .attn_state_values = RATIO4_ATTN_STATE,
        .indexer_state_values = RATIO4_INDEX_STATE,
    };
    check(ds4_gpu_v100_prefill_kv_update_f16_tensor(raw, comp, indexer,
                                                     attn_state, index_state,
                                                     attn_row, indexer_row,
                                                     &ratio4),
          "ratio4 kv update");
    read_f16_row(raw, (uint64_t)ratio4.slot * RAW_ROWS + ratio4.raw_row,
                 HEAD_DIM, half_row);
    expect_f16_row("ratio4 raw", half_row, attn_row, HEAD_DIM);
    read_f16_row(comp, (uint64_t)ratio4.slot * COMP_ROWS + ratio4.comp_row,
                 HEAD_DIM, half_row);
    expect_f16_row("ratio4 comp", half_row, attn_row, HEAD_DIM);

    uint16_t index_half[INDEX_DIM];
    read_f16_row(indexer, (uint64_t)ratio4.slot * COMP_ROWS + ratio4.comp_row,
                 INDEX_DIM, index_half);
    expect_f16_row("ratio4 indexer", index_half, indexer_row, INDEX_DIM);
    if (state_read) {
        check(ds4_gpu_tensor_read(attn_state,
                                  (uint64_t)ratio4.slot * RATIO4_ATTN_STATE * sizeof(float),
                                  state_read,
                                  (uint64_t)RATIO4_ATTN_STATE * sizeof(float)),
              "ratio4 attn state read");
        expect_attn_state("ratio4 attn state", state_read, attn_row,
                          RATIO4_ATTN_STATE, HEAD_DIM, ratio4.ratio);
        check(ds4_gpu_tensor_read(index_state,
                                  (uint64_t)ratio4.slot * RATIO4_INDEX_STATE * sizeof(float),
                                  state_read,
                                  (uint64_t)RATIO4_INDEX_STATE * sizeof(float)),
              "ratio4 index state read");
        expect_indexer_state("ratio4 index state", state_read, indexer_row,
                             RATIO4_INDEX_STATE, INDEX_DIM, ratio4.ratio);
    }

    ds4_gpu_v100_prefill_kv_update bad = ratio4;
    bad.ratio = 7;
    check(!ds4_gpu_v100_prefill_kv_update_f16_tensor(raw, comp, indexer,
                                                      attn_state, index_state,
                                                      attn_row, indexer_row,
                                                      &bad),
          "accepted invalid ratio class");
    bad = ratio4;
    bad.slot = SLOTS;
    check(!ds4_gpu_v100_prefill_kv_update_f16_tensor(raw, comp, indexer,
                                                      attn_state, index_state,
                                                      attn_row, indexer_row,
                                                      &bad),
          "accepted invalid slot");
    bad = ratio4;
    bad.raw_row = RAW_ROWS;
    check(!ds4_gpu_v100_prefill_kv_update_f16_tensor(raw, comp, indexer,
                                                      attn_state, index_state,
                                                      attn_row, indexer_row,
                                                      &bad),
          "accepted invalid raw row");
    bad = ratio4;
    bad.comp_row = COMP_ROWS;
    check(!ds4_gpu_v100_prefill_kv_update_f16_tensor(raw, comp, indexer,
                                                      attn_state, index_state,
                                                      attn_row, indexer_row,
                                                      &bad),
          "accepted invalid compressed row");
    check(!ds4_gpu_v100_prefill_kv_update_f16_tensor(raw, comp, NULL,
                                                      attn_state, index_state,
                                                      attn_row, indexer_row,
                                                      &ratio4),
          "accepted missing ratio4 indexer tensor");

    free(state_read);
    ds4_gpu_tensor_free(index_state);
    ds4_gpu_tensor_free(attn_state);
    ds4_gpu_tensor_free(indexer);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_arena_close(source_arena);

    if (failures) return 1;
    puts("cuda_v100_prefill_kv_smoke: ok");
    return 0;
}
