#include "engine/tp_runtime.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int kGpus = 8;
constexpr int kLayers = 43;
constexpr int kSwa = 128;
constexpr int kHeadDim = 512;
constexpr int kIndexerHeadDim = 128;

struct gpu_state {
    int device = 0;
    half *hidden_in = nullptr;
    half *hidden_out = nullptr;
    void *kv = nullptr;
    void *comp_state = nullptr;
    void *scratch = nullptr;
    ds4_tp_gpu_report report = {};
};

struct layer_kv_layout {
    int ratio = 0;
    uint64_t attn_rows = 0;
    uint64_t attn_base = 0;
    uint64_t attn_row_bytes = 0;
    uint64_t indexer_rows = 0;
    uint64_t indexer_base = 0;
    uint64_t indexer_row_bytes = 0;
};

} // namespace

struct ds4_tp_runtime {
    ds4_tp_runtime_config cfg = {};
    gpu_state gpu[kGpus];
    layer_kv_layout kv_layout[kLayers] = {};
    uint64_t kv_slot_stride = 0;
};

namespace {

static void set_err(char *err, size_t err_len, const char *msg) {
    if (err && err_len) {
        std::snprintf(err, err_len, "%s", msg);
    }
}

static int fail_cuda(char *err, size_t err_len, const char *what, cudaError_t rc) {
    if (err && err_len) {
        std::snprintf(err, err_len, "%s: %s", what, cudaGetErrorString(rc));
    }
    return -1;
}

static uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a) {
        std::fprintf(stderr, "ds4_tp_runtime: integer overflow\n");
        std::abort();
    }
    return a * b;
}

static uint64_t ceil_div(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

static uint64_t bytes_blocks(uint64_t elems, uint64_t block_elems, uint64_t block_bytes) {
    return checked_mul((elems + block_elems - 1) / block_elems, block_bytes);
}

static uint64_t kv_values_bytes(uint64_t values, ds4_tp_kv_dtype kv) {
    switch (kv) {
    case DS4_V100_TP_KV_F16:
        return checked_mul(values, 2);
    case DS4_V100_TP_KV_F8_E4M3_B128:
    case DS4_V100_TP_KV_F8_E5M2_B128:
        return bytes_blocks(values, 128, 129);
    case DS4_V100_TP_KV_Q8_0:
        return bytes_blocks(values, 32, 34);
    }
    return checked_mul(values, 2);
}

static bool is_f8_kv(ds4_tp_kv_dtype kv) {
    return kv == DS4_V100_TP_KV_F8_E4M3_B128 ||
           kv == DS4_V100_TP_KV_F8_E5M2_B128;
}

static int layer_ratio(int layer) {
    if (layer < 2) return 0;
    return (layer % 2) == 0 ? 4 : 128;
}

static uint64_t row_values_bytes(uint64_t values, ds4_tp_kv_dtype kv) {
    return kv_values_bytes(values, kv);
}

static uint64_t layer_comp_state_bytes(int layer, uint64_t ctx) {
    const int ratio = layer_ratio(layer);
    if (!ratio) return 0;
    const uint64_t attn = checked_mul(ctx / (uint64_t)ratio, kHeadDim);
    if (ratio == 4) {
        const uint64_t indexer = checked_mul(ctx / 4u, kIndexerHeadDim);
        return checked_mul((attn + indexer) / 8u, 4u);
    }
    return checked_mul(attn / 8u, 4u);
}

static void planned_kv_bytes(const ds4_tp_runtime_config *cfg,
                             uint64_t *kv_per_gpu,
                             uint64_t *comp_per_gpu) {
    uint64_t comp = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        comp += layer_comp_state_bytes(layer, cfg->ctx);
    }
    uint64_t slot_stride = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        const int ratio = layer_ratio(layer);
        const uint64_t attn_rows = (uint64_t)kSwa + (ratio ? cfg->ctx / (uint64_t)ratio : 0);
        const uint64_t attn_row_bytes =
            ceil_div(row_values_bytes(kHeadDim, cfg->kv_dtype), kGpus);
        slot_stride += checked_mul(attn_rows, attn_row_bytes);
        if (ratio == 4) {
            const uint64_t indexer_rows = cfg->ctx / 4u;
            const uint64_t indexer_row_bytes =
                ceil_div(row_values_bytes(kIndexerHeadDim, cfg->kv_dtype), kGpus);
            slot_stride += checked_mul(indexer_rows, indexer_row_bytes);
        }
    }
    *kv_per_gpu = checked_mul(slot_stride, cfg->slots);
    *comp_per_gpu = checked_mul(ceil_div(comp, kGpus), cfg->slots);
}

static void build_kv_layout(ds4_tp_runtime *rt) {
    uint64_t cursor = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        layer_kv_layout *l = &rt->kv_layout[layer];
        l->ratio = layer_ratio(layer);
        l->attn_rows = (uint64_t)kSwa + (l->ratio ? rt->cfg.ctx / (uint64_t)l->ratio : 0);
        l->attn_row_bytes =
            ceil_div(row_values_bytes(kHeadDim, rt->cfg.kv_dtype), kGpus);
        l->attn_base = cursor;
        cursor += checked_mul(l->attn_rows, l->attn_row_bytes);
        if (l->ratio == 4) {
            l->indexer_rows = rt->cfg.ctx / 4u;
            l->indexer_row_bytes =
                ceil_div(row_values_bytes(kIndexerHeadDim, rt->cfg.kv_dtype), kGpus);
            l->indexer_base = cursor;
            cursor += checked_mul(l->indexer_rows, l->indexer_row_bytes);
        }
    }
    rt->kv_slot_stride = cursor;
}

__global__ void fixture_kernel(const half *in, half *out, half *scratch,
                               size_t hidden_elems, size_t scratch_elems,
                               float value) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_elems) {
        const float v = __half2float(in[i]) + value;
        out[i] = __float2half(v);
    }
    if (i < scratch_elems) {
        scratch[i] = __float2half(value);
    }
}

__global__ void init_hidden_kernel(half *hidden, uint64_t hidden_elems, int gpu) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_elems) return;
    const float v = (float)(gpu + 1) + (float)(i % 251u) * 0.0009765625f;
    hidden[i] = __float2half(v);
}

__device__ unsigned char expected_kv_byte(int gpu, int layer, uint32_t slot,
                                          uint64_t position, size_t byte_index,
                                          float hidden_term, int indexer) {
    const unsigned int h = (unsigned int)(hidden_term * 16.0f);
    const unsigned int v = (unsigned int)(gpu * 19 + layer * 17 + slot * 13 +
                                          (unsigned int)(position * 7) +
                                          (unsigned int)byte_index + h +
                                          (indexer ? 113u : 0u));
    return (unsigned char)(v & 0xffu);
}

__device__ uint8_t e8m0_encode_pow2_scale(float scale) {
    if (!isfinite(scale) || scale <= 0.0f) return 0;
    int exp = (int)ceilf(log2f(scale)) + 127;
    if (exp < 1) exp = 1;
    if (exp > 254) exp = 254;
    return (uint8_t)exp;
}

__device__ float e8m0_to_f32_dev(uint8_t e) {
    return __uint_as_float(e == 0 ? 0x00400000u : ((uint32_t)e << 23));
}

__device__ float e4m3fn_to_f32_dev(uint8_t x) {
    const uint32_t sign = ((uint32_t)x & 0x80u) << 24;
    const uint32_t ax = (uint32_t)x & 0x7fu;
    if (ax == 0) return __uint_as_float(sign ? 0x80000000u : 0u);
    if (ax == 0x7f) return __uint_as_float(0x7fc00000u);
    const uint32_t exp = ax >> 3;
    const uint32_t man = ax & 0x07u;
    if (exp != 0) {
        return __uint_as_float(sign | ((exp + 120u) << 23) | (man << 20));
    }
    const uint32_t hi = man >= 4u ? 2u : (man >= 2u ? 1u : 0u);
    const uint32_t mant = (man << (23u - hi)) & 0x007fffffu;
    return __uint_as_float(sign | ((118u + hi) << 23) | mant);
}

__device__ uint8_t e4m3fn_quant_byte_dev(float x) {
    const uint8_t sign = x < 0.0f ? 0x80u : 0u;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (e4m3fn_to_f32_dev((uint8_t)mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - e4m3fn_to_f32_dev((uint8_t)best));
        const float next_diff = fabsf(ax - e4m3fn_to_f32_dev((uint8_t)(best + 1)));
        if (next_diff < best_diff ||
            (next_diff == best_diff && (((best + 1) & 1) == 0) && ((best & 1) != 0))) {
            best++;
        }
    }
    return (uint8_t)(sign | (uint8_t)best);
}

__device__ float e5m2_to_f32_dev(uint8_t x) {
    const uint32_t sign = ((uint32_t)x & 0x80u) << 24;
    const uint32_t ax = (uint32_t)x & 0x7fu;
    if (ax == 0) return __uint_as_float(sign ? 0x80000000u : 0u);
    const uint32_t exp = ax >> 2;
    const uint32_t man = ax & 0x03u;
    if (exp == 31u) {
        return man == 0u ? __uint_as_float(sign | 0x7f800000u)
                         : __uint_as_float(0x7fc00000u);
    }
    if (exp != 0u) {
        return __uint_as_float(sign | ((exp + 112u) << 23) | (man << 21));
    }
    const uint32_t hi = man >= 2u ? 1u : 0u;
    const uint32_t mant = (man << (23u - hi)) & 0x007fffffu;
    return __uint_as_float(sign | ((113u + hi) << 23) | mant);
}

__device__ uint8_t e5m2_quant_byte_dev(float x) {
    const uint8_t sign = x < 0.0f ? 0x80u : 0u;
    const float ax = fminf(fabsf(x), 57344.0f);
    int lo = 0;
    int hi = 123;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (e5m2_to_f32_dev((uint8_t)mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    int best = lo;
    if (best < 123) {
        const float best_diff = fabsf(ax - e5m2_to_f32_dev((uint8_t)best));
        const float next_diff = fabsf(ax - e5m2_to_f32_dev((uint8_t)(best + 1)));
        if (next_diff < best_diff ||
            (next_diff == best_diff && (((best + 1) & 1) == 0) && ((best & 1) != 0))) {
            best++;
        }
    }
    return (uint8_t)(sign | (uint8_t)best);
}

__device__ float f8_kv_max_dev(int kv_dtype) {
    return kv_dtype == DS4_V100_TP_KV_F8_E5M2_B128 ? 57344.0f : 448.0f;
}

__device__ uint8_t f8_kv_quant_byte_dev(float x, int kv_dtype) {
    return kv_dtype == DS4_V100_TP_KV_F8_E5M2_B128
        ? e5m2_quant_byte_dev(x)
        : e4m3fn_quant_byte_dev(x);
}

__device__ float f8_kv_to_f32_dev(uint8_t x, int kv_dtype) {
    return kv_dtype == DS4_V100_TP_KV_F8_E5M2_B128
        ? e5m2_to_f32_dev(x)
        : e4m3fn_to_f32_dev(x);
}

__device__ float typed_row_source_value(int layer, uint32_t slot, uint64_t position,
                                        uint32_t col, int indexer) {
    const float base =
        (float)((layer + 1) * 0.125f) +
        (float)((int)(slot % 17u) - 8) * 0.03125f +
        (float)((int)(position % 257u) - 128) * 0.0009765625f +
        (float)((int)(col % 97u) - 48) * 0.0078125f;
    return indexer ? base * 0.75f - 0.125f : base;
}

__global__ void typed_kv_store_f8_row_kernel(unsigned char *kv,
                                             uint64_t offset,
                                             uint64_t shard_row_bytes,
                                             int gpu,
                                             int layer,
                                             uint32_t slot,
                                             uint64_t position,
                                             uint32_t logical_cols,
                                             uint64_t logical_row_bytes,
                                             int indexer,
                                             int kv_dtype) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= shard_row_bytes) return;
    const uint64_t global_byte = (uint64_t)gpu * shard_row_bytes + i;
    if (global_byte >= logical_row_bytes) return;
    const uint32_t block = (uint32_t)(global_byte / 129ull);
    const uint32_t in_block = (uint32_t)(global_byte - (uint64_t)block * 129ull);
    const uint32_t col0 = block * 128u;
    if (col0 >= logical_cols) return;

    float amax = 0.0f;
    for (uint32_t c = 0; c < 128u && col0 + c < logical_cols; ++c) {
        amax = fmaxf(amax, fabsf(typed_row_source_value(layer, slot, position,
                                                        col0 + c, indexer)));
    }
    if (amax < 1.0e-8f) amax = 1.0e-8f;
    const uint8_t scale_byte = e8m0_encode_pow2_scale(amax / f8_kv_max_dev(kv_dtype));
    const float decoded_scale = e8m0_to_f32_dev(scale_byte);

    if (in_block == 0u) {
        kv[offset + i] = scale_byte;
        return;
    }
    const uint32_t col = col0 + in_block - 1u;
    if (col >= logical_cols) {
        kv[offset + i] = 0;
        return;
    }
    const float v = typed_row_source_value(layer, slot, position, col, indexer);
    kv[offset + i] = f8_kv_quant_byte_dev(v / decoded_scale, kv_dtype);
}

__global__ void fill_typed_row_source_kernel(float *dst,
                                             uint32_t logical_cols,
                                             int layer,
                                             uint32_t slot,
                                             uint64_t position,
                                             int indexer) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= logical_cols) return;
    dst[col] = typed_row_source_value(layer, slot, position, col, indexer);
}

__global__ void store_f32_device_to_f8_kv_row_kernel(unsigned char *kv,
                                                     uint64_t offset,
                                                     uint64_t shard_row_bytes,
                                                     int gpu,
                                                     const float *src,
                                                     uint32_t logical_cols,
                                                     uint64_t logical_row_bytes,
                                                     int kv_dtype) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= shard_row_bytes) return;
    const uint64_t global_byte = (uint64_t)gpu * shard_row_bytes + i;
    if (global_byte >= logical_row_bytes) return;
    const uint32_t block = (uint32_t)(global_byte / 129ull);
    const uint32_t in_block = (uint32_t)(global_byte - (uint64_t)block * 129ull);
    const uint32_t col0 = block * 128u;
    if (col0 >= logical_cols) return;

    float amax = 0.0f;
    for (uint32_t c = 0; c < 128u && col0 + c < logical_cols; ++c) {
        amax = fmaxf(amax, fabsf(src[col0 + c]));
    }
    if (amax < 1.0e-8f) amax = 1.0e-8f;
    const uint8_t scale_byte = e8m0_encode_pow2_scale(amax / f8_kv_max_dev(kv_dtype));
    const float decoded_scale = e8m0_to_f32_dev(scale_byte);

    if (in_block == 0u) {
        kv[offset + i] = scale_byte;
        return;
    }
    const uint32_t col = col0 + in_block - 1u;
    if (col >= logical_cols) {
        kv[offset + i] = 0;
        return;
    }
    kv[offset + i] = f8_kv_quant_byte_dev(src[col] / decoded_scale, kv_dtype);
}

__global__ void store_f32_device_to_f8_kv_rows_kernel(unsigned char *kv,
                                                      uint64_t first_offset,
                                                      uint64_t kv_slot_stride,
                                                      uint64_t shard_row_bytes,
                                                      int gpu,
                                                      const float *src,
                                                      uint64_t src_stride_floats,
                                                      uint32_t logical_cols,
                                                      uint64_t logical_row_bytes,
                                                      uint32_t slot_count,
                                                      int kv_dtype) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)slot_count * shard_row_bytes;
    if (idx >= total) return;
    const uint32_t slot = (uint32_t)(idx / shard_row_bytes);
    const uint64_t i = idx - (uint64_t)slot * shard_row_bytes;
    const uint64_t global_byte = (uint64_t)gpu * shard_row_bytes + i;
    if (global_byte >= logical_row_bytes) return;
    const uint32_t block = (uint32_t)(global_byte / 129ull);
    const uint32_t in_block = (uint32_t)(global_byte - (uint64_t)block * 129ull);
    const uint32_t col0 = block * 128u;
    if (col0 >= logical_cols) return;

    const float *slot_src = src + (uint64_t)slot * src_stride_floats;
    float amax = 0.0f;
    for (uint32_t c = 0; c < 128u && col0 + c < logical_cols; ++c) {
        amax = fmaxf(amax, fabsf(slot_src[col0 + c]));
    }
    if (amax < 1.0e-8f) amax = 1.0e-8f;
    const uint8_t scale_byte = e8m0_encode_pow2_scale(amax / f8_kv_max_dev(kv_dtype));
    const float decoded_scale = e8m0_to_f32_dev(scale_byte);

    const uint64_t out = first_offset + (uint64_t)slot * kv_slot_stride + i;
    if (in_block == 0u) {
        kv[out] = scale_byte;
        return;
    }
    const uint32_t col = col0 + in_block - 1u;
    if (col >= logical_cols) {
        kv[out] = 0;
        return;
    }
    kv[out] = f8_kv_quant_byte_dev(slot_src[col] / decoded_scale, kv_dtype);
}

__device__ uint64_t typed_kv_physical_row_dev(uint64_t position,
                                              int ratio,
                                              int kind) {
    if (kind == DS4_V100_TP_KV_ROW_ATTN_RAW) {
        return position % (uint64_t)kSwa;
    }
    if (kind == DS4_V100_TP_KV_ROW_INDEXER) {
        return position / 4ull;
    }
    return ratio == 0 ? position % (uint64_t)kSwa
                      : (uint64_t)kSwa + position / (uint64_t)ratio;
}

__global__ void store_f32_device_to_f8_kv_rows_at_position_kernel(
    unsigned char *kv,
    uint64_t first_base_offset,
    uint64_t kv_slot_stride,
    uint64_t rows,
    uint64_t shard_row_bytes,
    int gpu,
    int ratio,
    int kind,
    const uint64_t *decode_position,
    const float *src,
    uint64_t src_stride_floats,
    uint32_t logical_cols,
    uint64_t logical_row_bytes,
    uint32_t slot_count,
    int kv_dtype) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)slot_count * shard_row_bytes;
    if (idx >= total) return;
    const uint64_t position = decode_position ? *decode_position : 0ull;
    const uint64_t physical_row = typed_kv_physical_row_dev(position, ratio, kind);
    if (physical_row >= rows) return;
    const uint32_t slot = (uint32_t)(idx / shard_row_bytes);
    const uint64_t i = idx - (uint64_t)slot * shard_row_bytes;
    const uint64_t global_byte = (uint64_t)gpu * shard_row_bytes + i;
    if (global_byte >= logical_row_bytes) return;
    const uint32_t block = (uint32_t)(global_byte / 129ull);
    const uint32_t in_block = (uint32_t)(global_byte - (uint64_t)block * 129ull);
    const uint32_t col0 = block * 128u;
    if (col0 >= logical_cols) return;

    const float *slot_src = src + (uint64_t)slot * src_stride_floats;
    float amax = 0.0f;
    for (uint32_t c = 0; c < 128u && col0 + c < logical_cols; ++c) {
        amax = fmaxf(amax, fabsf(slot_src[col0 + c]));
    }
    if (amax < 1.0e-8f) amax = 1.0e-8f;
    const uint8_t scale_byte = e8m0_encode_pow2_scale(amax / f8_kv_max_dev(kv_dtype));
    const float decoded_scale = e8m0_to_f32_dev(scale_byte);

    const uint64_t out = first_base_offset + (uint64_t)slot * kv_slot_stride +
                         physical_row * shard_row_bytes + i;
    if (in_block == 0u) {
        kv[out] = scale_byte;
        return;
    }
    const uint32_t col = col0 + in_block - 1u;
    if (col >= logical_cols) {
        kv[out] = 0;
        return;
    }
    kv[out] = f8_kv_quant_byte_dev(slot_src[col] / decoded_scale, kv_dtype);
}

__device__ uint8_t get_sharded_row_byte(uint64_t global_byte,
                                        uint64_t row_bytes,
                                        const unsigned char *p0,
                                        const unsigned char *p1,
                                        const unsigned char *p2,
                                        const unsigned char *p3,
                                        const unsigned char *p4,
                                        const unsigned char *p5,
                                        const unsigned char *p6,
                                        const unsigned char *p7) {
    const uint64_t owner = global_byte / row_bytes;
    const uint64_t off = global_byte - owner * row_bytes;
    switch ((int)owner) {
    case 0: return p0[off];
    case 1: return p1[off];
    case 2: return p2[off];
    case 3: return p3[off];
    case 4: return p4[off];
    case 5: return p5[off];
    case 6: return p6[off];
    default: return p7[off];
    }
}

__global__ void load_f8_kv_row_to_f32_device_kernel(float *dst,
                                                    uint32_t logical_cols,
                                                    uint64_t row_bytes,
                                                    const unsigned char *p0,
                                                    const unsigned char *p1,
                                                    const unsigned char *p2,
                                                    const unsigned char *p3,
                                                    const unsigned char *p4,
                                                    const unsigned char *p5,
                                                    const unsigned char *p6,
                                                    const unsigned char *p7,
                                                    int kv_dtype) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= logical_cols) return;
    const uint64_t block = (uint64_t)(col / 128u);
    const uint64_t block_off = block * 129ull;
    const uint64_t q_off = block_off + 1ull + (uint64_t)(col & 127u);
    const uint8_t scale_byte =
        get_sharded_row_byte(block_off, row_bytes, p0, p1, p2, p3, p4, p5, p6, p7);
    const uint8_t q =
        get_sharded_row_byte(q_off, row_bytes, p0, p1, p2, p3, p4, p5, p6, p7);
    dst[col] = f8_kv_to_f32_dev(q, kv_dtype) * e8m0_to_f32_dev(scale_byte);
}

__global__ void load_f8_kv_rows_to_f32_device_kernel(float *dst,
                                                     uint64_t dst_stride_floats,
                                                     uint32_t logical_cols,
                                                     uint64_t row_bytes,
                                                     uint64_t kv_slot_stride,
                                                     uint32_t slot_count,
                                                     const unsigned char *p0,
                                                     const unsigned char *p1,
                                                     const unsigned char *p2,
                                                     const unsigned char *p3,
                                                     const unsigned char *p4,
                                                     const unsigned char *p5,
                                                     const unsigned char *p6,
                                                     const unsigned char *p7,
                                                     int kv_dtype) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)slot_count * (uint64_t)logical_cols;
    if (idx >= total) return;
    const uint32_t slot = (uint32_t)(idx / (uint64_t)logical_cols);
    const uint32_t col = (uint32_t)(idx - (uint64_t)slot * (uint64_t)logical_cols);
    const uint64_t block = (uint64_t)(col / 128u);
    const uint64_t block_off = block * 129ull;
    const uint64_t q_off = block_off + 1ull + (uint64_t)(col & 127u);
    const unsigned char *s0 = p0 + (uint64_t)slot * kv_slot_stride;
    const unsigned char *s1 = p1 + (uint64_t)slot * kv_slot_stride;
    const unsigned char *s2 = p2 + (uint64_t)slot * kv_slot_stride;
    const unsigned char *s3 = p3 + (uint64_t)slot * kv_slot_stride;
    const unsigned char *s4 = p4 + (uint64_t)slot * kv_slot_stride;
    const unsigned char *s5 = p5 + (uint64_t)slot * kv_slot_stride;
    const unsigned char *s6 = p6 + (uint64_t)slot * kv_slot_stride;
    const unsigned char *s7 = p7 + (uint64_t)slot * kv_slot_stride;
    const uint8_t scale_byte =
        get_sharded_row_byte(block_off, row_bytes, s0, s1, s2, s3, s4, s5, s6, s7);
    const uint8_t q =
        get_sharded_row_byte(q_off, row_bytes, s0, s1, s2, s3, s4, s5, s6, s7);
    dst[(uint64_t)slot * dst_stride_floats + col] =
        f8_kv_to_f32_dev(q, kv_dtype) * e8m0_to_f32_dev(scale_byte);
}

__global__ void load_f8_kv_rows_at_position_to_f32_device_kernel(
    float *dst,
    uint64_t dst_stride_floats,
    uint32_t logical_cols,
    uint64_t rows,
    uint64_t row_bytes,
    uint64_t kv_slot_stride,
    uint32_t slot_count,
    int ratio,
    int kind,
    const uint64_t *decode_position,
    const unsigned char *p0,
    const unsigned char *p1,
    const unsigned char *p2,
    const unsigned char *p3,
    const unsigned char *p4,
    const unsigned char *p5,
    const unsigned char *p6,
    const unsigned char *p7,
    int kv_dtype) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)slot_count * (uint64_t)logical_cols;
    if (idx >= total) return;
    const uint64_t position = decode_position ? *decode_position : 0ull;
    const uint64_t physical_row = typed_kv_physical_row_dev(position, ratio, kind);
    if (physical_row >= rows) return;
    const uint32_t slot = (uint32_t)(idx / (uint64_t)logical_cols);
    const uint32_t col = (uint32_t)(idx - (uint64_t)slot * (uint64_t)logical_cols);
    const uint64_t block = (uint64_t)(col / 128u);
    const uint64_t block_off = block * 129ull;
    const uint64_t q_off = block_off + 1ull + (uint64_t)(col & 127u);
    const uint64_t row_off = physical_row * row_bytes;
    const unsigned char *s0 = p0 + (uint64_t)slot * kv_slot_stride + row_off;
    const unsigned char *s1 = p1 + (uint64_t)slot * kv_slot_stride + row_off;
    const unsigned char *s2 = p2 + (uint64_t)slot * kv_slot_stride + row_off;
    const unsigned char *s3 = p3 + (uint64_t)slot * kv_slot_stride + row_off;
    const unsigned char *s4 = p4 + (uint64_t)slot * kv_slot_stride + row_off;
    const unsigned char *s5 = p5 + (uint64_t)slot * kv_slot_stride + row_off;
    const unsigned char *s6 = p6 + (uint64_t)slot * kv_slot_stride + row_off;
    const unsigned char *s7 = p7 + (uint64_t)slot * kv_slot_stride + row_off;
    const uint8_t scale_byte =
        get_sharded_row_byte(block_off, row_bytes, s0, s1, s2, s3, s4, s5, s6, s7);
    const uint8_t q =
        get_sharded_row_byte(q_off, row_bytes, s0, s1, s2, s3, s4, s5, s6, s7);
    const uint64_t dst_row_offset =
        kind == DS4_V100_TP_KV_ROW_ATTN_RAW
            ? physical_row * (uint64_t)logical_cols
            : 0ull;
    dst[(uint64_t)slot * dst_stride_floats + dst_row_offset + col] =
        f8_kv_to_f32_dev(q, kv_dtype) * e8m0_to_f32_dev(scale_byte);
}

__global__ void dense_kv_slice_kernel(half *hidden, unsigned char *kv,
                                      uint64_t hidden_elems,
                                      uint64_t attn_offset,
                                      uint64_t attn_row_bytes,
                                      uint64_t indexer_offset,
                                      uint64_t indexer_row_bytes,
                                      int gpu,
                                      int layer,
                                      uint32_t slot,
                                      uint64_t position) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < attn_row_bytes) {
        const float hidden_term = __half2float(hidden[(i * 37u) % hidden_elems]);
        kv[attn_offset + i] = expected_kv_byte(gpu, layer, slot, position, i,
                                               hidden_term, 0);
    }
    if (indexer_offset != UINT64_MAX && i < indexer_row_bytes) {
        const float hidden_term = __half2float(hidden[(i * 53u) % hidden_elems]);
        kv[indexer_offset + i] = expected_kv_byte(gpu, layer, slot, position, i,
                                                  hidden_term, 1);
    }
}

static unsigned char expected_kv_byte_host(int gpu, int layer, uint32_t slot,
                                           uint64_t position, size_t byte_index,
                                           int multiplier, int indexer) {
    const float hidden_value =
        (float)(gpu + 1) + (float)((byte_index * (size_t)multiplier) % 251u) *
        0.0009765625f;
    const float hidden_term = __half2float(__float2half(hidden_value));
    const unsigned int h = (unsigned int)(hidden_term * 16.0f);
    const unsigned int v = (unsigned int)(gpu * 19 + layer * 17 + slot * 13 +
                                          (unsigned int)(position * 7) +
                                          (unsigned int)byte_index + h +
                                          (indexer ? 113u : 0u));
    return (unsigned char)(v & 0xffu);
}

static float f32_from_bits_host(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static float e8m0_to_f32_host(uint8_t e) {
    return f32_from_bits_host(e == 0 ? 0x00400000u : ((uint32_t)e << 23));
}

static uint8_t e8m0_encode_pow2_scale_host(float scale) {
    if (!std::isfinite(scale) || scale <= 0.0f) return 0;
    int exp = (int)std::ceil(std::log2(scale)) + 127;
    if (exp < 1) exp = 1;
    if (exp > 254) exp = 254;
    return (uint8_t)exp;
}

static float e4m3fn_to_f32_host(uint8_t x) {
    const uint8_t ax = x & 0x7f;
    const bool sign = (x & 0x80) != 0;
    if (ax == 0) return f32_from_bits_host(sign ? 0x80000000u : 0u);
    if (ax == 0x7f) return f32_from_bits_host(0x7fc00000u);
    const int exp = (ax >> 3) & 0x0f;
    const int man = ax & 0x07;
    const float value = exp == 0 ? std::ldexp((float)man, -9)
                                 : std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
    return sign ? -value : value;
}

static uint8_t e4m3fn_quant_byte_host(float x) {
    const uint8_t sign = x < 0.0f ? 0x80u : 0u;
    const float ax = std::fmin(std::fabs(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (e4m3fn_to_f32_host((uint8_t)mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = std::fabs(ax - e4m3fn_to_f32_host((uint8_t)best));
        const float next_diff = std::fabs(ax - e4m3fn_to_f32_host((uint8_t)(best + 1)));
        if (next_diff < best_diff ||
            (next_diff == best_diff && (((best + 1) & 1) == 0) && ((best & 1) != 0))) {
            best++;
        }
    }
    return (uint8_t)(sign | (uint8_t)best);
}

static float e5m2_to_f32_host(uint8_t x) {
    const uint8_t ax = x & 0x7f;
    const bool sign = (x & 0x80) != 0;
    if (ax == 0) return f32_from_bits_host(sign ? 0x80000000u : 0u);
    const int exp = (ax >> 2) & 0x1f;
    const int man = ax & 0x03;
    float value;
    if (exp == 31) {
        value = man == 0 ? std::numeric_limits<float>::infinity()
                         : std::numeric_limits<float>::quiet_NaN();
    } else if (exp == 0) {
        value = std::ldexp((float)man, -16);
    } else {
        value = std::ldexp(1.0f + (float)man / 4.0f, exp - 15);
    }
    return sign ? -value : value;
}

static uint8_t e5m2_quant_byte_host(float x) {
    const uint8_t sign = x < 0.0f ? 0x80u : 0u;
    const float ax = std::fmin(std::fabs(x), 57344.0f);
    int lo = 0;
    int hi = 123;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (e5m2_to_f32_host((uint8_t)mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    int best = lo;
    if (best < 123) {
        const float best_diff = std::fabs(ax - e5m2_to_f32_host((uint8_t)best));
        const float next_diff = std::fabs(ax - e5m2_to_f32_host((uint8_t)(best + 1)));
        if (next_diff < best_diff ||
            (next_diff == best_diff && (((best + 1) & 1) == 0) && ((best & 1) != 0))) {
            best++;
        }
    }
    return (uint8_t)(sign | (uint8_t)best);
}

static float f8_kv_max_host(ds4_tp_kv_dtype kv_dtype) {
    return kv_dtype == DS4_V100_TP_KV_F8_E5M2_B128 ? 57344.0f : 448.0f;
}

static uint8_t f8_kv_quant_byte_host(float x, ds4_tp_kv_dtype kv_dtype) {
    return kv_dtype == DS4_V100_TP_KV_F8_E5M2_B128
        ? e5m2_quant_byte_host(x)
        : e4m3fn_quant_byte_host(x);
}

static float f8_kv_to_f32_host(uint8_t x, ds4_tp_kv_dtype kv_dtype) {
    return kv_dtype == DS4_V100_TP_KV_F8_E5M2_B128
        ? e5m2_to_f32_host(x)
        : e4m3fn_to_f32_host(x);
}

static float typed_row_source_value_host(int layer, uint32_t slot, uint64_t position,
                                         uint32_t col, int indexer) {
    const float base =
        (float)((layer + 1) * 0.125f) +
        (float)((int)(slot % 17u) - 8) * 0.03125f +
        (float)((int)(position % 257u) - 128) * 0.0009765625f +
        (float)((int)(col % 97u) - 48) * 0.0078125f;
    return indexer ? base * 0.75f - 0.125f : base;
}

static void fill_expected_f8_row(std::vector<unsigned char> *packed,
                                 std::vector<float> *decoded,
                                 int layer,
                                 uint32_t slot,
                                 uint64_t position,
                                 uint32_t logical_cols,
                                 int indexer,
                                 ds4_tp_kv_dtype kv_dtype) {
    const uint64_t row_bytes = kv_values_bytes(logical_cols, kv_dtype);
    packed->assign((size_t)row_bytes, 0);
    decoded->assign((size_t)logical_cols, 0.0f);
    const uint32_t blocks = logical_cols / 128u;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint32_t col0 = b * 128u;
        float amax = 0.0f;
        for (uint32_t c = 0; c < 128u; ++c) {
            amax = std::fmax(amax, std::fabs(typed_row_source_value_host(
                                            layer, slot, position, col0 + c, indexer)));
        }
        if (amax < 1.0e-8f) amax = 1.0e-8f;
        const uint8_t scale_byte = e8m0_encode_pow2_scale_host(amax / f8_kv_max_host(kv_dtype));
        const float decoded_scale = e8m0_to_f32_host(scale_byte);
        const uint64_t block_off = (uint64_t)b * 129ull;
        (*packed)[(size_t)block_off] = scale_byte;
        for (uint32_t c = 0; c < 128u; ++c) {
            const float v = typed_row_source_value_host(layer, slot, position,
                                                        col0 + c, indexer);
            const uint8_t q = f8_kv_quant_byte_host(v / decoded_scale, kv_dtype);
            (*packed)[(size_t)(block_off + 1u + c)] = q;
            (*decoded)[(size_t)(col0 + c)] = f8_kv_to_f32_host(q, kv_dtype) * decoded_scale;
        }
    }
}

} // namespace

extern "C" void ds4_tp_runtime_default_config(ds4_tp_runtime_config *cfg) {
    std::memset(cfg, 0, sizeof(*cfg));
    for (int i = 0; i < kGpus; ++i) cfg->devices[i] = i;
    cfg->slots = 32;
    cfg->ctx = 262144;
    cfg->hidden = 4096;
    cfg->kv_dtype = DS4_V100_TP_KV_F8_E4M3_B128;
    cfg->scratch_bytes = 1536ull * 1024ull * 1024ull;
    cfg->allocate_comp_state = 1;
}

extern "C" int ds4_tp_runtime_open(ds4_tp_runtime **out,
                                         const ds4_tp_runtime_config *cfg,
                                         char *err,
                                         size_t err_len) {
    if (!out || !cfg) {
        set_err(err, err_len, "invalid argument");
        return -1;
    }
    *out = nullptr;

    int device_count = 0;
    cudaError_t rc = cudaGetDeviceCount(&device_count);
    if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaGetDeviceCount", rc);
    for (int i = 0; i < kGpus; ++i) {
        if (cfg->devices[i] < 0 || cfg->devices[i] >= device_count) {
            set_err(err, err_len, "configured device is outside visible device count");
            return -1;
        }
    }

    ds4_tp_runtime *rt = new ds4_tp_runtime();
    rt->cfg = *cfg;
    build_kv_layout(rt);

    uint64_t kv_per_gpu = 0;
    uint64_t comp_per_gpu = 0;
    planned_kv_bytes(cfg, &kv_per_gpu, &comp_per_gpu);
    const uint64_t hidden_bytes =
        checked_mul(checked_mul((uint64_t)cfg->slots, cfg->hidden), sizeof(half));

    for (int i = 0; i < kGpus; ++i) {
        rc = cudaSetDevice(cfg->devices[i]);
        if (rc != cudaSuccess) {
            ds4_tp_runtime_close(rt);
            return fail_cuda(err, err_len, "cudaSetDevice", rc);
        }
        for (int j = 0; j < kGpus; ++j) {
            if (i == j) continue;
            int can_access = 0;
            rc = cudaDeviceCanAccessPeer(&can_access, cfg->devices[i], cfg->devices[j]);
            if (rc != cudaSuccess) {
                ds4_tp_runtime_close(rt);
                return fail_cuda(err, err_len, "cudaDeviceCanAccessPeer", rc);
            }
            if (!can_access) {
                ds4_tp_runtime_close(rt);
                set_err(err, err_len, "peer access unavailable");
                return -1;
            }
            rc = cudaDeviceEnablePeerAccess(cfg->devices[j], 0);
            if (rc == cudaErrorPeerAccessAlreadyEnabled) {
                (void)cudaGetLastError();
            } else if (rc != cudaSuccess) {
                ds4_tp_runtime_close(rt);
                return fail_cuda(err, err_len, "cudaDeviceEnablePeerAccess", rc);
            }
        }
    }

    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        g->device = cfg->devices[i];
        rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) {
            ds4_tp_runtime_close(rt);
            return fail_cuda(err, err_len, "cudaSetDevice", rc);
        }
        const uint64_t allocated_comp_per_gpu =
            cfg->allocate_comp_state ? comp_per_gpu : 0;
        if ((rc = cudaMalloc(&g->hidden_in, hidden_bytes)) != cudaSuccess ||
            (rc = cudaMalloc(&g->hidden_out, hidden_bytes)) != cudaSuccess ||
            (rc = cudaMalloc(&g->kv, kv_per_gpu)) != cudaSuccess ||
            (allocated_comp_per_gpu != 0 &&
             (rc = cudaMalloc(&g->comp_state, allocated_comp_per_gpu)) != cudaSuccess) ||
            (rc = cudaMalloc(&g->scratch, cfg->scratch_bytes)) != cudaSuccess) {
            ds4_tp_runtime_close(rt);
            return fail_cuda(err, err_len, "cudaMalloc", rc);
        }
        rc = cudaMemset(g->hidden_in, 0, hidden_bytes);
        if (rc != cudaSuccess) {
            ds4_tp_runtime_close(rt);
            return fail_cuda(err, err_len, "cudaMemset", rc);
        }
        g->report.hidden_bytes = 2 * hidden_bytes;
        g->report.kv_bytes = kv_per_gpu;
        g->report.comp_state_bytes = allocated_comp_per_gpu;
        g->report.scratch_bytes = cfg->scratch_bytes;
        g->report.total_bytes =
            g->report.hidden_bytes + g->report.kv_bytes +
            g->report.comp_state_bytes + g->report.scratch_bytes;
    }

    *out = rt;
    return 0;
}

extern "C" int ds4_tp_runtime_fixture(ds4_tp_runtime *rt,
                                            double *max_abs,
                                            char *err,
                                            size_t err_len) {
    if (!rt) {
        set_err(err, err_len, "runtime is null");
        return -1;
    }
    const size_t hidden_elems = (size_t)rt->cfg.slots * (size_t)rt->cfg.hidden;
    const size_t scratch_elems =
        std::min((size_t)(rt->cfg.scratch_bytes / sizeof(half)), hidden_elems);
    const int block = 256;
    const int grid = (int)((std::max(hidden_elems, scratch_elems) + block - 1) / block);
    double worst = 0.0;

    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);
        const float value = (float)(i + 1);
        fixture_kernel<<<grid, block>>>(g->hidden_in, g->hidden_out, (half *)g->scratch,
                                        hidden_elems, scratch_elems, value);
        rc = cudaGetLastError();
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "fixture_kernel", rc);
    }
    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);
        rc = cudaDeviceSynchronize();
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaDeviceSynchronize", rc);

        std::vector<half> host(hidden_elems);
        rc = cudaMemcpy(host.data(), g->hidden_out, hidden_elems * sizeof(half),
                        cudaMemcpyDeviceToHost);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaMemcpy", rc);
        const float expected = (float)(i + 1);
        for (size_t j = 0; j < hidden_elems; ++j) {
            const float got = __half2float(host[j]);
            if (!std::isfinite(got)) {
                set_err(err, err_len, "non-finite fixture output");
                return -1;
            }
            worst = std::max(worst, (double)std::fabs(got - expected));
        }
    }
    if (max_abs) *max_abs = worst;
    return 0;
}

extern "C" int ds4_tp_runtime_dense_kv_slice(ds4_tp_runtime *rt,
                                                   int layer,
                                                   uint32_t slot,
                                                   uint64_t position,
                                                   int write_indexer,
                                                   ds4_tp_dense_kv_result *result,
                                                   char *err,
                                                   size_t err_len) {
    if (!rt || !result) {
        set_err(err, err_len, "invalid argument");
        return -1;
    }
    if (layer < 0 || layer >= kLayers) {
        set_err(err, err_len, "layer is outside DS4 range");
        return -1;
    }
    if (slot >= rt->cfg.slots) {
        set_err(err, err_len, "slot is outside configured slot count");
        return -1;
    }
    if (position >= rt->cfg.ctx) {
        set_err(err, err_len, "position is outside configured context");
        return -1;
    }

    const layer_kv_layout *layout = &rt->kv_layout[layer];
    if (write_indexer && layout->ratio != 4) {
        set_err(err, err_len, "indexer KV is only present on ratio-4 layers");
        return -1;
    }

    std::memset(result, 0, sizeof(*result));
    result->layer = layer;
    result->ratio = layout->ratio;
    result->slot = slot;
    result->position = position;
    result->indexer_row = std::numeric_limits<uint64_t>::max();
    for (int i = 0; i < kGpus; ++i) {
        result->indexer_offset[i] = std::numeric_limits<uint64_t>::max();
    }

    if (layout->ratio == 0) {
        result->attn_row = position % (uint64_t)kSwa;
    } else {
        result->attn_row = (uint64_t)kSwa + position / (uint64_t)layout->ratio;
    }
    if (result->attn_row >= layout->attn_rows) {
        set_err(err, err_len, "computed attn KV row is outside layout");
        return -1;
    }
    if (write_indexer) {
        result->indexer_row = position / 4u;
        if (result->indexer_row >= layout->indexer_rows) {
            set_err(err, err_len, "computed indexer KV row is outside layout");
            return -1;
        }
    }

    const uint64_t hidden_elems =
        checked_mul((uint64_t)rt->cfg.slots, (uint64_t)rt->cfg.hidden);
    const int block = 256;
    const int hidden_grid = (int)((hidden_elems + block - 1) / block);
    const uint64_t max_row_bytes =
        std::max(layout->attn_row_bytes,
                 write_indexer ? layout->indexer_row_bytes : 0u);
    const int kv_grid = (int)((max_row_bytes + block - 1) / block);
    double worst = 0.0;

    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);

        const uint64_t slot_base = checked_mul((uint64_t)slot, rt->kv_slot_stride);
        const uint64_t attn_offset =
            slot_base + layout->attn_base +
            checked_mul(result->attn_row, layout->attn_row_bytes);
        const uint64_t indexer_offset =
            write_indexer
                ? slot_base + layout->indexer_base +
                      checked_mul(result->indexer_row, layout->indexer_row_bytes)
                : std::numeric_limits<uint64_t>::max();

        result->attn_offset[i] = attn_offset;
        result->attn_row_bytes[i] = layout->attn_row_bytes;
        result->indexer_offset[i] = indexer_offset;
        result->indexer_row_bytes[i] = write_indexer ? layout->indexer_row_bytes : 0;

        init_hidden_kernel<<<hidden_grid, block>>>(g->hidden_in, hidden_elems, i);
        rc = cudaGetLastError();
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "init_hidden_kernel", rc);
        dense_kv_slice_kernel<<<kv_grid, block>>>(
            g->hidden_in, (unsigned char *)g->kv, hidden_elems, attn_offset,
            layout->attn_row_bytes, indexer_offset,
            write_indexer ? layout->indexer_row_bytes : 0, i, layer, slot, position);
        rc = cudaGetLastError();
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "dense_kv_slice_kernel", rc);
    }

    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);
        rc = cudaDeviceSynchronize();
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaDeviceSynchronize", rc);

        std::vector<unsigned char> row((size_t)layout->attn_row_bytes);
        rc = cudaMemcpy(row.data(), (unsigned char *)g->kv + result->attn_offset[i],
                        (size_t)layout->attn_row_bytes, cudaMemcpyDeviceToHost);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaMemcpy attn KV", rc);
        for (size_t j = 0; j < row.size(); ++j) {
            const unsigned char expected =
                expected_kv_byte_host(i, layer, slot, position, j, 37, 0);
            worst = std::max(worst, std::fabs((double)row[j] - (double)expected));
        }

        if (write_indexer) {
            std::vector<unsigned char> indexer_row((size_t)layout->indexer_row_bytes);
            rc = cudaMemcpy(indexer_row.data(),
                            (unsigned char *)g->kv + result->indexer_offset[i],
                            (size_t)layout->indexer_row_bytes,
                            cudaMemcpyDeviceToHost);
            if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaMemcpy indexer KV", rc);
            for (size_t j = 0; j < indexer_row.size(); ++j) {
                const unsigned char expected =
                    expected_kv_byte_host(i, layer, slot, position, j, 53, 1);
                worst = std::max(worst,
                                 std::fabs((double)indexer_row[j] - (double)expected));
            }
        }
    }

    result->max_abs = worst;
    return 0;
}

extern "C" int ds4_tp_runtime_kv_row_view(ds4_tp_runtime *rt,
                                                int layer,
                                                uint32_t slot,
                                                uint64_t position,
                                                ds4_tp_kv_row_kind kind,
                                                ds4_tp_kv_row_view *view,
                                                char *err,
                                                size_t err_len) {
    if (!rt || !view) {
        set_err(err, err_len, "invalid argument");
        return -1;
    }
    if (layer < 0 || layer >= kLayers) {
        set_err(err, err_len, "layer is outside DS4 range");
        return -1;
    }
    if (slot >= rt->cfg.slots) {
        set_err(err, err_len, "slot is outside configured slot count");
        return -1;
    }
    if (position >= rt->cfg.ctx) {
        set_err(err, err_len, "position is outside configured context");
        return -1;
    }

    const layer_kv_layout *layout = &rt->kv_layout[layer];
    std::memset(view, 0, sizeof(*view));
    view->layer = layer;
    view->ratio = layout->ratio;
    view->slot = slot;
    view->position = position;
    view->kind = kind;

    uint64_t base = 0;
    uint64_t rows = 0;
    uint64_t row_bytes = 0;
    if (kind == DS4_V100_TP_KV_ROW_ATTN) {
        view->logical_cols = kHeadDim;
        view->logical_row_bytes = kv_values_bytes(kHeadDim, rt->cfg.kv_dtype);
        view->physical_row = layout->ratio == 0
            ? position % (uint64_t)kSwa
            : (uint64_t)kSwa + position / (uint64_t)layout->ratio;
        base = layout->attn_base;
        rows = layout->attn_rows;
        row_bytes = layout->attn_row_bytes;
    } else if (kind == DS4_V100_TP_KV_ROW_ATTN_RAW) {
        view->logical_cols = kHeadDim;
        view->logical_row_bytes = kv_values_bytes(kHeadDim, rt->cfg.kv_dtype);
        view->physical_row = position % (uint64_t)kSwa;
        base = layout->attn_base;
        rows = layout->attn_rows;
        row_bytes = layout->attn_row_bytes;
    } else if (kind == DS4_V100_TP_KV_ROW_INDEXER) {
        if (layout->ratio != 4) {
            set_err(err, err_len, "indexer KV is only present on ratio-4 layers");
            return -1;
        }
        view->logical_cols = kIndexerHeadDim;
        view->logical_row_bytes = kv_values_bytes(kIndexerHeadDim, rt->cfg.kv_dtype);
        view->physical_row = position / 4u;
        base = layout->indexer_base;
        rows = layout->indexer_rows;
        row_bytes = layout->indexer_row_bytes;
    } else {
        set_err(err, err_len, "unknown KV row kind");
        return -1;
    }

    if (view->physical_row >= rows) {
        set_err(err, err_len, "computed KV row is outside layout");
        return -1;
    }

    const uint64_t slot_base = checked_mul((uint64_t)slot, rt->kv_slot_stride);
    for (int i = 0; i < kGpus; ++i) {
        view->offset[i] = slot_base + base + checked_mul(view->physical_row, row_bytes);
        view->row_bytes[i] = row_bytes;
    }
    return 0;
}

extern "C" int ds4_tp_runtime_kv_row_roundtrip_f32(
    ds4_tp_runtime *rt,
    int layer,
    uint32_t slot,
    uint64_t position,
    ds4_tp_kv_row_kind kind,
    ds4_tp_kv_row_roundtrip_result *result,
    char *err,
    size_t err_len) {
    if (!rt || !result) {
        set_err(err, err_len, "invalid argument");
        return -1;
    }
    if (!is_f8_kv(rt->cfg.kv_dtype)) {
        set_err(err, err_len, "typed row roundtrip currently supports F8 KV only");
        return -1;
    }

    ds4_tp_kv_row_view view;
    if (ds4_tp_runtime_kv_row_view(rt, layer, slot, position, kind, &view,
                                        err, err_len) != 0) {
        return -1;
    }

    const int indexer = kind == DS4_V100_TP_KV_ROW_INDEXER ? 1 : 0;
    const int block = 256;
    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);
        const int grid = (int)((view.row_bytes[i] + block - 1) / block);
        typed_kv_store_f8_row_kernel<<<grid, block>>>(
            (unsigned char *)g->kv, view.offset[i], view.row_bytes[i], i, layer,
            slot, position, view.logical_cols, view.logical_row_bytes, indexer,
            (int)rt->cfg.kv_dtype);
        rc = cudaGetLastError();
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "typed_kv_store_f8_row_kernel", rc);
    }
    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);
        rc = cudaDeviceSynchronize();
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaDeviceSynchronize", rc);
    }

    std::vector<unsigned char> packed((size_t)view.logical_row_bytes, 0);
    for (int i = 0; i < kGpus; ++i) {
        const uint64_t dst = (uint64_t)i * view.row_bytes[i];
        if (dst >= view.logical_row_bytes) continue;
        const uint64_t copy_bytes =
            std::min(view.row_bytes[i], view.logical_row_bytes - dst);
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);
        rc = cudaMemcpy(packed.data() + dst, (unsigned char *)g->kv + view.offset[i],
                        (size_t)copy_bytes, cudaMemcpyDeviceToHost);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaMemcpy KV row shard", rc);
    }

    std::vector<unsigned char> expected_packed;
    std::vector<float> expected_decoded;
    fill_expected_f8_row(&expected_packed, &expected_decoded, layer, slot, position,
                         view.logical_cols, indexer, rt->cfg.kv_dtype);

    std::vector<float> decoded((size_t)view.logical_cols, 0.0f);
    uint32_t byte_bad = 0;
    uint32_t first_bad_index = 0;
    uint8_t first_bad_got = 0;
    uint8_t first_bad_expected = 0;
    uint64_t checksum = 1469598103934665603ull;
    for (size_t i = 0; i < packed.size(); ++i) {
        checksum ^= (uint64_t)packed[i];
        checksum *= 1099511628211ull;
        if (packed[i] != expected_packed[i]) {
            if (byte_bad == 0) {
                first_bad_index = (uint32_t)i;
                first_bad_got = packed[i];
                first_bad_expected = expected_packed[i];
            }
            byte_bad++;
        }
    }

    const uint32_t blocks = view.logical_cols / 128u;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint64_t block_off = (uint64_t)b * 129ull;
        const float scale = e8m0_to_f32_host(packed[(size_t)block_off]);
        for (uint32_t c = 0; c < 128u; ++c) {
            const uint8_t q = packed[(size_t)(block_off + 1u + c)];
            decoded[(size_t)(b * 128u + c)] =
                f8_kv_to_f32_host(q, rt->cfg.kv_dtype) * scale;
        }
    }

    double max_abs = 0.0;
    double sum_abs = 0.0;
    uint32_t decoded_bad = 0;
    for (uint32_t i = 0; i < view.logical_cols; ++i) {
        const double diff = std::fabs((double)decoded[i] - (double)expected_decoded[i]);
        max_abs = std::max(max_abs, diff);
        sum_abs += diff;
        if (diff != 0.0) decoded_bad++;
    }

    std::memset(result, 0, sizeof(*result));
    result->view = view;
    result->max_abs = max_abs;
    result->mean_abs = sum_abs / (double)view.logical_cols;
    result->bad_values = decoded_bad;
    result->byte_mismatches = byte_bad;
    result->first_bad_index = first_bad_index;
    result->first_bad_got = first_bad_got;
    result->first_bad_expected = first_bad_expected;
    result->checksum = checksum;
    return 0;
}

extern "C" int ds4_tp_runtime_kv_row_store_f32_device(
    ds4_tp_runtime *rt,
    int layer,
    uint32_t slot,
    uint64_t position,
    ds4_tp_kv_row_kind kind,
    const void *src_by_gpu[kGpus],
    char *err,
    size_t err_len) {
    if (!rt || !src_by_gpu) {
        set_err(err, err_len, "invalid argument");
        return -1;
    }
    if (!is_f8_kv(rt->cfg.kv_dtype)) {
        set_err(err, err_len, "device KV store currently supports F8 KV only");
        return -1;
    }

    ds4_tp_kv_row_view view;
    if (ds4_tp_runtime_kv_row_view(rt, layer, slot, position, kind, &view,
                                        err, err_len) != 0) {
        return -1;
    }

    const int block = 256;
    for (int i = 0; i < kGpus; ++i) {
        if (!src_by_gpu[i]) {
            set_err(err, err_len, "null source row pointer");
            return -1;
        }
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);
        const int grid = (int)((view.row_bytes[i] + block - 1) / block);
        store_f32_device_to_f8_kv_row_kernel<<<grid, block>>>(
            (unsigned char *)g->kv, view.offset[i], view.row_bytes[i], i,
            (const float *)src_by_gpu[i], view.logical_cols, view.logical_row_bytes,
            (int)rt->cfg.kv_dtype);
        rc = cudaGetLastError();
        if (rc != cudaSuccess) {
            return fail_cuda(err, err_len, "store_f32_device_to_f8_kv_row_kernel", rc);
        }
    }
    return 0;
}

extern "C" int ds4_tp_runtime_kv_row_load_f32_device(
    ds4_tp_runtime *rt,
    int layer,
    uint32_t slot,
    uint64_t position,
    ds4_tp_kv_row_kind kind,
    void *dst_by_gpu[kGpus],
    char *err,
    size_t err_len) {
    if (!rt || !dst_by_gpu) {
        set_err(err, err_len, "invalid argument");
        return -1;
    }
    if (!is_f8_kv(rt->cfg.kv_dtype)) {
        set_err(err, err_len, "device KV load currently supports F8 KV only");
        return -1;
    }

    ds4_tp_kv_row_view view;
    if (ds4_tp_runtime_kv_row_view(rt, layer, slot, position, kind, &view,
                                        err, err_len) != 0) {
        return -1;
    }

    const unsigned char *p[kGpus] = {};
    for (int i = 0; i < kGpus; ++i) {
        if (!dst_by_gpu[i]) {
            set_err(err, err_len, "null destination row pointer");
            return -1;
        }
        p[i] = (const unsigned char *)rt->gpu[i].kv + view.offset[i];
    }

    const int block = 256;
    const int grid = (int)((view.logical_cols + block - 1) / block);
    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);
        load_f8_kv_row_to_f32_device_kernel<<<grid, block>>>(
            (float *)dst_by_gpu[i], view.logical_cols, view.row_bytes[0],
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
            (int)rt->cfg.kv_dtype);
        rc = cudaGetLastError();
        if (rc != cudaSuccess) {
            return fail_cuda(err, err_len, "load_f8_kv_row_to_f32_device_kernel", rc);
        }
    }
    return 0;
}

static int kv_rows_store_f32_device_impl(
    ds4_tp_runtime *rt,
    int layer,
    uint32_t first_slot,
    uint32_t slot_count,
    uint64_t position,
    ds4_tp_kv_row_kind kind,
    const void *src_by_gpu[kGpus],
    uint64_t src_stride_floats,
    void *const stream_by_gpu[kGpus],
    char *err,
    size_t err_len) {
    if (!rt || !src_by_gpu || slot_count == 0 || src_stride_floats == 0) {
        set_err(err, err_len, "invalid argument");
        return -1;
    }
    if (first_slot >= rt->cfg.slots || slot_count > rt->cfg.slots - first_slot) {
        set_err(err, err_len, "slot range is outside configured slot count");
        return -1;
    }
    if (!is_f8_kv(rt->cfg.kv_dtype)) {
        set_err(err, err_len, "device KV rows store currently supports F8 KV only");
        return -1;
    }

    ds4_tp_kv_row_view view;
    if (ds4_tp_runtime_kv_row_view(rt, layer, first_slot, position, kind,
                                        &view, err, err_len) != 0) {
        return -1;
    }

    const int block = 256;
    for (int i = 0; i < kGpus; ++i) {
        if (!src_by_gpu[i]) {
            set_err(err, err_len, "null source row pointer");
            return -1;
        }
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);
        const uint64_t total = (uint64_t)slot_count * view.row_bytes[i];
        const int grid = (int)((total + block - 1) / block);
        cudaStream_t stream = stream_by_gpu ? (cudaStream_t)stream_by_gpu[i] : (cudaStream_t)0;
        store_f32_device_to_f8_kv_rows_kernel<<<grid, block, 0, stream>>>(
            (unsigned char *)g->kv, view.offset[i], rt->kv_slot_stride,
            view.row_bytes[i], i, (const float *)src_by_gpu[i],
            src_stride_floats, view.logical_cols, view.logical_row_bytes,
            slot_count, (int)rt->cfg.kv_dtype);
        rc = cudaGetLastError();
        if (rc != cudaSuccess) {
            return fail_cuda(err, err_len, "store_f32_device_to_f8_kv_rows_kernel", rc);
        }
    }
    return 0;
}

static int kv_kind_layout(const ds4_tp_runtime *rt,
                          int layer,
                          ds4_tp_kv_row_kind kind,
                          uint32_t *logical_cols,
                          uint64_t *logical_row_bytes,
                          uint64_t *base,
                          uint64_t *rows,
                          uint64_t *row_bytes,
                          char *err,
                          size_t err_len) {
    if (!rt || layer < 0 || layer >= kLayers) {
        set_err(err, err_len, "invalid runtime or layer");
        return -1;
    }
    const layer_kv_layout *layout = &rt->kv_layout[layer];
    if (kind == DS4_V100_TP_KV_ROW_ATTN ||
        kind == DS4_V100_TP_KV_ROW_ATTN_RAW) {
        *logical_cols = kHeadDim;
        *logical_row_bytes = kv_values_bytes(kHeadDim, rt->cfg.kv_dtype);
        *base = layout->attn_base;
        *rows = layout->attn_rows;
        *row_bytes = layout->attn_row_bytes;
        return 0;
    }
    if (kind == DS4_V100_TP_KV_ROW_INDEXER) {
        if (layout->ratio != 4) {
            set_err(err, err_len, "indexer KV is only present on ratio-4 layers");
            return -1;
        }
        *logical_cols = kIndexerHeadDim;
        *logical_row_bytes = kv_values_bytes(kIndexerHeadDim, rt->cfg.kv_dtype);
        *base = layout->indexer_base;
        *rows = layout->indexer_rows;
        *row_bytes = layout->indexer_row_bytes;
        return 0;
    }
    set_err(err, err_len, "unknown KV row kind");
    return -1;
}

static int kv_rows_store_f32_device_at_position_impl(
    ds4_tp_runtime *rt,
    int layer,
    uint32_t first_slot,
    uint32_t slot_count,
    ds4_tp_kv_row_kind kind,
    const void *src_by_gpu[kGpus],
    uint64_t src_stride_floats,
    void *const stream_by_gpu[kGpus],
    const void *const position_by_gpu[kGpus],
    char *err,
    size_t err_len) {
    if (!rt || !src_by_gpu || !position_by_gpu || slot_count == 0 ||
        src_stride_floats == 0) {
        set_err(err, err_len, "invalid argument");
        return -1;
    }
    if (first_slot >= rt->cfg.slots || slot_count > rt->cfg.slots - first_slot) {
        set_err(err, err_len, "slot range is outside configured slot count");
        return -1;
    }
    if (!is_f8_kv(rt->cfg.kv_dtype)) {
        set_err(err, err_len, "device KV rows store currently supports F8 KV only");
        return -1;
    }

    uint32_t logical_cols = 0;
    uint64_t logical_row_bytes = 0;
    uint64_t base = 0;
    uint64_t rows = 0;
    uint64_t row_bytes = 0;
    if (kv_kind_layout(rt, layer, kind, &logical_cols, &logical_row_bytes,
                       &base, &rows, &row_bytes, err, err_len) != 0) {
        return -1;
    }

    const uint64_t slot_base = checked_mul((uint64_t)first_slot, rt->kv_slot_stride);
    const uint64_t first_base_offset = slot_base + base;
    const int ratio = rt->kv_layout[layer].ratio;
    const int block = 256;
    for (int i = 0; i < kGpus; ++i) {
        if (!src_by_gpu[i] || !position_by_gpu[i]) {
            set_err(err, err_len, "null source or position pointer");
            return -1;
        }
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);
        const uint64_t total = (uint64_t)slot_count * row_bytes;
        const int grid = (int)((total + block - 1) / block);
        cudaStream_t stream = stream_by_gpu ? (cudaStream_t)stream_by_gpu[i] : (cudaStream_t)0;
        store_f32_device_to_f8_kv_rows_at_position_kernel<<<grid, block, 0, stream>>>(
            (unsigned char *)g->kv, first_base_offset, rt->kv_slot_stride,
            rows, row_bytes, i, ratio, (int)kind,
            (const uint64_t *)position_by_gpu[i], (const float *)src_by_gpu[i],
            src_stride_floats, logical_cols, logical_row_bytes, slot_count,
            (int)rt->cfg.kv_dtype);
        rc = cudaGetLastError();
        if (rc != cudaSuccess) {
            return fail_cuda(err, err_len,
                             "store_f32_device_to_f8_kv_rows_at_position_kernel", rc);
        }
    }
    return 0;
}

extern "C" int ds4_tp_runtime_kv_rows_store_f32_device(
    ds4_tp_runtime *rt,
    int layer,
    uint32_t first_slot,
    uint32_t slot_count,
    uint64_t position,
    ds4_tp_kv_row_kind kind,
    const void *src_by_gpu[kGpus],
    uint64_t src_stride_floats,
    char *err,
    size_t err_len) {
    return kv_rows_store_f32_device_impl(
        rt, layer, first_slot, slot_count, position, kind, src_by_gpu,
        src_stride_floats, nullptr, err, err_len);
}

extern "C" int ds4_tp_runtime_kv_rows_store_f32_device_streams(
    ds4_tp_runtime *rt,
    int layer,
    uint32_t first_slot,
    uint32_t slot_count,
    uint64_t position,
    ds4_tp_kv_row_kind kind,
    const void *src_by_gpu[kGpus],
    uint64_t src_stride_floats,
    void *const stream_by_gpu[kGpus],
    char *err,
    size_t err_len) {
    return kv_rows_store_f32_device_impl(
        rt, layer, first_slot, slot_count, position, kind, src_by_gpu,
        src_stride_floats, stream_by_gpu, err, err_len);
}

extern "C" int ds4_tp_runtime_kv_rows_store_f32_device_streams_at_position(
    ds4_tp_runtime *rt,
    int layer,
    uint32_t first_slot,
    uint32_t slot_count,
    ds4_tp_kv_row_kind kind,
    const void *src_by_gpu[kGpus],
    uint64_t src_stride_floats,
    void *const stream_by_gpu[kGpus],
    const void *const position_by_gpu[kGpus],
    char *err,
    size_t err_len) {
    return kv_rows_store_f32_device_at_position_impl(
        rt, layer, first_slot, slot_count, kind, src_by_gpu,
        src_stride_floats, stream_by_gpu, position_by_gpu, err, err_len);
}

static int kv_rows_load_f32_device_impl(
    ds4_tp_runtime *rt,
    int layer,
    uint32_t first_slot,
    uint32_t slot_count,
    uint64_t position,
    ds4_tp_kv_row_kind kind,
    void *dst_by_gpu[kGpus],
    uint64_t dst_stride_floats,
    void *const stream_by_gpu[kGpus],
    char *err,
    size_t err_len) {
    if (!rt || !dst_by_gpu || slot_count == 0 || dst_stride_floats == 0) {
        set_err(err, err_len, "invalid argument");
        return -1;
    }
    if (first_slot >= rt->cfg.slots || slot_count > rt->cfg.slots - first_slot) {
        set_err(err, err_len, "slot range is outside configured slot count");
        return -1;
    }
    if (!is_f8_kv(rt->cfg.kv_dtype)) {
        set_err(err, err_len, "device KV rows load currently supports F8 KV only");
        return -1;
    }

    ds4_tp_kv_row_view view;
    if (ds4_tp_runtime_kv_row_view(rt, layer, first_slot, position, kind,
                                        &view, err, err_len) != 0) {
        return -1;
    }

    const unsigned char *p[kGpus] = {};
    for (int i = 0; i < kGpus; ++i) {
        if (!dst_by_gpu[i]) {
            set_err(err, err_len, "null destination row pointer");
            return -1;
        }
        p[i] = (const unsigned char *)rt->gpu[i].kv + view.offset[i];
    }

    const int block = 256;
    const uint64_t total = (uint64_t)slot_count * (uint64_t)view.logical_cols;
    const int grid = (int)((total + block - 1) / block);
    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);
        cudaStream_t stream = stream_by_gpu ? (cudaStream_t)stream_by_gpu[i] : (cudaStream_t)0;
        load_f8_kv_rows_to_f32_device_kernel<<<grid, block, 0, stream>>>(
            (float *)dst_by_gpu[i], dst_stride_floats, view.logical_cols,
            view.row_bytes[0], rt->kv_slot_stride, slot_count,
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
            (int)rt->cfg.kv_dtype);
        rc = cudaGetLastError();
        if (rc != cudaSuccess) {
            return fail_cuda(err, err_len, "load_f8_kv_rows_to_f32_device_kernel", rc);
        }
    }
    return 0;
}

static int kv_rows_load_f32_device_at_position_impl(
    ds4_tp_runtime *rt,
    int layer,
    uint32_t first_slot,
    uint32_t slot_count,
    ds4_tp_kv_row_kind kind,
    void *dst_by_gpu[kGpus],
    uint64_t dst_stride_floats,
    void *const stream_by_gpu[kGpus],
    const void *const position_by_gpu[kGpus],
    char *err,
    size_t err_len) {
    if (!rt || !dst_by_gpu || !position_by_gpu || slot_count == 0 ||
        dst_stride_floats == 0) {
        set_err(err, err_len, "invalid argument");
        return -1;
    }
    if (first_slot >= rt->cfg.slots || slot_count > rt->cfg.slots - first_slot) {
        set_err(err, err_len, "slot range is outside configured slot count");
        return -1;
    }
    if (!is_f8_kv(rt->cfg.kv_dtype)) {
        set_err(err, err_len, "device KV rows load currently supports F8 KV only");
        return -1;
    }

    uint32_t logical_cols = 0;
    uint64_t logical_row_bytes = 0;
    uint64_t base = 0;
    uint64_t rows = 0;
    uint64_t row_bytes = 0;
    if (kv_kind_layout(rt, layer, kind, &logical_cols, &logical_row_bytes,
                       &base, &rows, &row_bytes, err, err_len) != 0) {
        return -1;
    }
    (void)logical_row_bytes;

    const uint64_t slot_base = checked_mul((uint64_t)first_slot, rt->kv_slot_stride);
    const uint64_t first_base_offset = slot_base + base;
    const unsigned char *p[kGpus] = {};
    for (int i = 0; i < kGpus; ++i) {
        if (!dst_by_gpu[i] || !position_by_gpu[i]) {
            set_err(err, err_len, "null destination or position pointer");
            return -1;
        }
        p[i] = (const unsigned char *)rt->gpu[i].kv + first_base_offset;
    }

    const int ratio = rt->kv_layout[layer].ratio;
    const int block = 256;
    const uint64_t total = (uint64_t)slot_count * (uint64_t)logical_cols;
    const int grid = (int)((total + block - 1) / block);
    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);
        cudaStream_t stream = stream_by_gpu ? (cudaStream_t)stream_by_gpu[i] : (cudaStream_t)0;
        load_f8_kv_rows_at_position_to_f32_device_kernel<<<grid, block, 0, stream>>>(
            (float *)dst_by_gpu[i], dst_stride_floats, logical_cols,
            rows, row_bytes, rt->kv_slot_stride, slot_count, ratio, (int)kind,
            (const uint64_t *)position_by_gpu[i],
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
            (int)rt->cfg.kv_dtype);
        rc = cudaGetLastError();
        if (rc != cudaSuccess) {
            return fail_cuda(err, err_len,
                             "load_f8_kv_rows_at_position_to_f32_device_kernel", rc);
        }
    }
    return 0;
}

extern "C" int ds4_tp_runtime_kv_rows_load_f32_device(
    ds4_tp_runtime *rt,
    int layer,
    uint32_t first_slot,
    uint32_t slot_count,
    uint64_t position,
    ds4_tp_kv_row_kind kind,
    void *dst_by_gpu[kGpus],
    uint64_t dst_stride_floats,
    char *err,
    size_t err_len) {
    return kv_rows_load_f32_device_impl(
        rt, layer, first_slot, slot_count, position, kind, dst_by_gpu,
        dst_stride_floats, nullptr, err, err_len);
}

extern "C" int ds4_tp_runtime_kv_rows_load_f32_device_streams(
    ds4_tp_runtime *rt,
    int layer,
    uint32_t first_slot,
    uint32_t slot_count,
    uint64_t position,
    ds4_tp_kv_row_kind kind,
    void *dst_by_gpu[kGpus],
    uint64_t dst_stride_floats,
    void *const stream_by_gpu[kGpus],
    char *err,
    size_t err_len) {
    return kv_rows_load_f32_device_impl(
        rt, layer, first_slot, slot_count, position, kind, dst_by_gpu,
        dst_stride_floats, stream_by_gpu, err, err_len);
}

extern "C" int ds4_tp_runtime_kv_rows_load_f32_device_streams_at_position(
    ds4_tp_runtime *rt,
    int layer,
    uint32_t first_slot,
    uint32_t slot_count,
    ds4_tp_kv_row_kind kind,
    void *dst_by_gpu[kGpus],
    uint64_t dst_stride_floats,
    void *const stream_by_gpu[kGpus],
    const void *const position_by_gpu[kGpus],
    char *err,
    size_t err_len) {
    return kv_rows_load_f32_device_at_position_impl(
        rt, layer, first_slot, slot_count, kind, dst_by_gpu,
        dst_stride_floats, stream_by_gpu, position_by_gpu, err, err_len);
}

extern "C" int ds4_tp_runtime_kv_row_device_roundtrip_f32(
    ds4_tp_runtime *rt,
    int layer,
    uint32_t slot,
    uint64_t position,
    ds4_tp_kv_row_kind kind,
    ds4_tp_kv_device_roundtrip_result *result,
    char *err,
    size_t err_len) {
    if (!rt || !result) {
        set_err(err, err_len, "invalid argument");
        return -1;
    }

    ds4_tp_kv_row_view view;
    if (ds4_tp_runtime_kv_row_view(rt, layer, slot, position, kind, &view,
                                        err, err_len) != 0) {
        return -1;
    }
    if (!is_f8_kv(rt->cfg.kv_dtype)) {
        set_err(err, err_len, "device row roundtrip currently supports F8 KV only");
        return -1;
    }

    const int indexer = kind == DS4_V100_TP_KV_ROW_INDEXER ? 1 : 0;
    const uint64_t row_bytes = checked_mul(view.logical_cols, sizeof(float));
    void *src[kGpus] = {};
    void *dst[kGpus] = {};
    cudaError_t rc = cudaSuccess;

    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) goto fail;
        rc = cudaMalloc(&src[i], row_bytes);
        if (rc != cudaSuccess) goto fail;
        rc = cudaMalloc(&dst[i], row_bytes);
        if (rc != cudaSuccess) goto fail;
        const int block = 256;
        const int grid = (int)((view.logical_cols + block - 1) / block);
        fill_typed_row_source_kernel<<<grid, block>>>(
            (float *)src[i], view.logical_cols, layer, slot, position, indexer);
        rc = cudaGetLastError();
        if (rc != cudaSuccess) goto fail;
        rc = cudaMemset(dst[i], 0, row_bytes);
        if (rc != cudaSuccess) goto fail;
    }

    if (ds4_tp_runtime_kv_row_store_f32_device(
            rt, layer, slot, position, kind, (const void **)src, err, err_len) != 0) {
        goto fail_with_err;
    }
    for (int i = 0; i < kGpus; ++i) {
        rc = cudaSetDevice(rt->gpu[i].device);
        if (rc != cudaSuccess) goto fail;
        rc = cudaDeviceSynchronize();
        if (rc != cudaSuccess) goto fail;
    }
    if (ds4_tp_runtime_kv_row_load_f32_device(
            rt, layer, slot, position, kind, dst, err, err_len) != 0) {
        goto fail_with_err;
    }
    for (int i = 0; i < kGpus; ++i) {
        rc = cudaSetDevice(rt->gpu[i].device);
        if (rc != cudaSuccess) goto fail;
        rc = cudaDeviceSynchronize();
        if (rc != cudaSuccess) goto fail;
    }

    {
        std::vector<unsigned char> expected_packed;
        std::vector<float> expected_decoded;
        fill_expected_f8_row(&expected_packed, &expected_decoded, layer, slot, position,
                             view.logical_cols, indexer, rt->cfg.kv_dtype);
        double max_abs = 0.0;
        double sum_abs = 0.0;
        uint32_t bad = 0;
        uint64_t checksum = 1469598103934665603ull;
        std::vector<float> host((size_t)view.logical_cols);
        for (int i = 0; i < kGpus; ++i) {
            rc = cudaSetDevice(rt->gpu[i].device);
            if (rc != cudaSuccess) goto fail;
            rc = cudaMemcpy(host.data(), dst[i], row_bytes, cudaMemcpyDeviceToHost);
            if (rc != cudaSuccess) goto fail;
            for (uint32_t col = 0; col < view.logical_cols; ++col) {
                uint32_t bits = 0;
                std::memcpy(&bits, &host[(size_t)col], sizeof(bits));
                checksum ^= (uint64_t)bits + (uint64_t)(i + 1) * 1099511628211ull;
                checksum *= 1099511628211ull;
                const double diff =
                    std::fabs((double)host[(size_t)col] -
                              (double)expected_decoded[(size_t)col]);
                max_abs = std::max(max_abs, diff);
                sum_abs += diff;
                if (diff != 0.0) bad++;
            }
        }

        std::memset(result, 0, sizeof(*result));
        result->view = view;
        result->max_abs = max_abs;
        result->mean_abs =
            sum_abs / (double)(view.logical_cols * (uint32_t)kGpus);
        result->bad_values = bad;
        result->checksum = checksum;
    }

    for (int i = 0; i < kGpus; ++i) {
        cudaSetDevice(rt->gpu[i].device);
        cudaFree(src[i]);
        cudaFree(dst[i]);
    }
    return 0;

fail:
    fail_cuda(err, err_len, "device row roundtrip CUDA", rc);
fail_with_err:
    for (int i = 0; i < kGpus; ++i) {
        cudaSetDevice(rt->gpu[i].device);
        cudaFree(src[i]);
        cudaFree(dst[i]);
    }
    return -1;
}

extern "C" void ds4_tp_runtime_get_report(const ds4_tp_runtime *rt,
                                                ds4_tp_runtime_report *report) {
    std::memset(report, 0, sizeof(*report));
    if (!rt) return;
    for (int i = 0; i < kGpus; ++i) {
        report->gpu[i] = rt->gpu[i].report;
    }
}

extern "C" void ds4_tp_runtime_close(ds4_tp_runtime *rt) {
    if (!rt) return;
    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        cudaSetDevice(g->device);
        cudaFree(g->hidden_in);
        cudaFree(g->hidden_out);
        cudaFree(g->kv);
        cudaFree(g->comp_state);
        cudaFree(g->scratch);
    }
    delete rt;
}
