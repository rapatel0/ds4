#define _FILE_OFFSET_BITS 64

#include "ds4_v100_tp_runtime.h"
#include "kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <mma.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <sys/types.h>
#include <vector>

namespace {

constexpr int kGpus = 8;
constexpr int kHidden = 4096;
constexpr int kMid = 2048;
constexpr int kFusedN = 2 * kMid;
constexpr int kGlobalExperts = 256;
constexpr int kLocalExperts = kGlobalExperts / kGpus;
constexpr int kActiveLocalExperts = 6;
constexpr int kGroupSize = 32;
constexpr int kDType = GGML_TM_DTYPE_MXFP4;

#define CHECK_CUDA(expr)                                                              \
    do {                                                                              \
        cudaError_t err__ = (expr);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            std::fprintf(stderr, "cuda error %s:%d: %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(err__));                                  \
            std::exit(2);                                                             \
        }                                                                             \
    } while (0)

typedef int (*pfn_init)(int);
typedef void (*pfn_shutdown)(void);
typedef int (*pfn_mmgt)(const void *, const int *, const int *, int, int,
                        const void * const *, const void * const *, int, int, int, int, int,
                        void *, void *);
typedef int (*pfn_mmgs)(const void *, const int *, const int *, int, int,
                        const void * const *, const void * const *, int, int, int, int, int,
                        void *, void *);

struct alignas(16) StridedPtrH {
    void *p;
    int stride;
};
static_assert(sizeof(StridedPtrH) == 16, "StridedPtrH must match TurboMind ABI");

struct Api {
    pfn_init init = nullptr;
    pfn_shutdown shutdown = nullptr;
    pfn_mmgt mmgt = nullptr;
    pfn_mmgs mmgs = nullptr;
};

struct ContractRow {
    std::string record_type;
    std::string tensor_id;
    std::string family;
    std::string source_dtype;
    std::string source_shape;
    std::string runtime_layout;
    int layer = -1;
    int owning_gpu = -1;
    int tp_rank = -1;
    int ep_rank = -1;
    int shard_index = -1;
    int shard_count = -1;
    int expert_first = -1;
    int expert_count = 0;
    int kv_ratio = -1;
    uint64_t kv_rows_per_slot = 0;
    uint64_t bytes_estimate = 0;
    std::string source_pack_file;
    uint64_t source_shard_offset = 0;
    uint64_t source_byte_length = 0;
    std::string kernel_family;
};

struct TmIndexEntry {
    std::string semantic_tensor_id;
    std::string runtime_layout;
    std::string sidecar_file;
    int layer_id = -1;
    int n = 0;
    int k = 0;
    int experts_packed = 0;
    int experts_total = 0;
    size_t weight_bytes_per_expert = 0;
    size_t scale_bytes_per_expert = 0;
    int k_pack = 0;
    int weight_stride = 0;
    int scale_stride = 0;
    uint64_t weight_offset = 0;
    uint64_t scale_offset = 0;
};

struct DescriptorBindings {
    TmIndexEntry gated;
    TmIndexEntry down;
    bool have_gated = false;
    bool have_down = false;
};

struct PackedExperts {
    std::vector<void *> d_w_active;
    std::vector<void *> d_s_active;
    void *d_w_table = nullptr;
    void *d_s_table = nullptr;
    int k_pack = 0;
};

struct RankState {
    int rank = 0;
    int device = 0;
    int routes = 0;
    int active_experts = 0;
    int max_routes_per_expert = 0;
    cudaStream_t stream = nullptr;
    int *d_offsets = nullptr;
    int *d_route_slots = nullptr;
    __half *d_a = nullptr;
    __half *d_gated = nullptr;
    __half *d_down = nullptr;
    float *d_ep_contrib_all = nullptr;
    __half *d_ep_contrib_half_all = nullptr;
    float *d_ep_remote[kGpus] = {};
    __half *d_ep_remote_half[kGpus] = {};
    float *d_ep_sum = nullptr;
    float *d_next_hidden = nullptr;
    PackedExperts gated;
    PackedExperts down;
    cudaEvent_t start = nullptr;
    cudaEvent_t mid = nullptr;
    cudaEvent_t stop = nullptr;
};

struct GpuFamilyStats {
    uint64_t dense_rows = 0;
    uint64_t control_rows = 0;
    uint64_t expert_rows = 0;
    uint64_t kv_rows = 0;
    uint64_t comp_rows = 0;
    uint64_t dense_bytes = 0;
    uint64_t control_bytes = 0;
    uint64_t expert_descriptor_bytes = 0;
    uint64_t ep_loaded_bytes = 0;
    uint64_t checksum = 0;
};

struct LayerStats {
    uint64_t total_rows = 0;
    uint64_t dense_rows = 0;
    uint64_t control_rows = 0;
    uint64_t expert_rows = 0;
    uint64_t kv_rows = 0;
    uint64_t comp_rows = 0;
    uint64_t bad_rows = 0;
    uint64_t dense_loaded_bytes = 0;
    uint64_t control_loaded_bytes = 0;
    uint64_t ep_loaded_bytes = 0;
    uint64_t checksum = 0;
    GpuFamilyStats gpu[kGpus];
};

struct DenseComputeStats {
    bool enabled = false;
    bool pass = true;
    std::string tensor_id;
    int rows_per_gpu = 0;
    int cols = 0;
    int slots = 0;
    uint64_t loaded_bytes = 0;
    double compute_ms = 0.0;
    double repeat_max_abs = 0.0;
    double oracle_max_abs = 0.0;
    int repeat_bad = 0;
    int repeat_nan = 0;
    int oracle_bad = 0;
};

struct DeviceDenseOutputs {
    std::vector<float *> d_out;
    int rows_per_gpu = 0;
    int cols = 0;
    int slots = 0;
    uint64_t loaded_bytes = 0;
    double compute_ms = 0.0;
};

struct ResidentF8Dense {
    std::vector<uint8_t *> d_w;
    std::vector<float *> d_x;
    std::vector<__half *> d_w_half;
    std::vector<bool> owns_w_half;
    std::vector<__half *> d_x_half;
    std::vector<float *> d_out;
    std::vector<cublasHandle_t> cublas;
    int rows_per_gpu = 0;
    int cols = 0;
    int slots = 0;
    uint64_t row_bytes = 0;
    uint64_t loaded_bytes = 0;
};

struct DenseF16CacheEntry {
    std::string tensor_id;
    int gpu = -1;
    int cols = 0;
    int rows_per_gpu = 0;
    uint64_t offset = 0;
    uint64_t source_bytes = 0;
    uint64_t cache_bytes = 0;
};

struct DenseF16Cache {
    bool enabled = false;
    std::vector<uint8_t *> arena;
    std::vector<uint8_t *> temp;
    std::vector<DenseF16CacheEntry> entries;
    uint64_t rows = 0;
    uint64_t source_bytes = 0;
    uint64_t cache_bytes = 0;
    uint64_t cache_aligned_bytes = 0;
    uint64_t max_temp_bytes = 0;
};

struct ComposeStats {
    bool enabled = false;
    bool pass = true;
    uint64_t ep_contribution_bytes = 0;
    uint64_t ep_return_bytes = 0;
    double attn_dense_ms = 0.0;
    double shared_dense_ms = 0.0;
    double compose_ms = 0.0;
    double repeat_max_abs = 0.0;
    int finite_bad = 0;
    int repeat_bad = 0;
    uint64_t checksum = 0;
    bool ep_return_fp16 = false;
    bool fused_compose_sum = false;
    bool dense_hmma_compose = false;
    bool dense_f16_cublas_compose = false;
    bool dense_f16_cache_compose = false;
};

struct LayerRunSummary {
    int layer = -1;
    int ratio = 0;
    bool pass = false;
    uint64_t total_rows = 0;
    uint64_t dense_rows = 0;
    uint64_t control_rows = 0;
    uint64_t expert_rows = 0;
    uint64_t kv_rows = 0;
    uint64_t comp_rows = 0;
    double decode_ms_per_step = 0.0;
    double decode_slot_step_tok_s = 0.0;
    double decode_ep_ms_per_step = 0.0;
    double decode_dense_ms_per_step = 0.0;
    double decode_compose_ms_per_step = 0.0;
    uint64_t decode_checksum = 0;
};

struct SharedApi {
    void *lib = nullptr;
    Api api = {};
    bool initialized = false;
};

struct SharedRankBuffers {
    RankState ranks[kGpus];
    bool initialized = false;
    uint64_t core_bytes = 0;
};

struct SharedTpRuntime {
    ds4_v100_tp_runtime *rt = nullptr;
    ds4_v100_tp_runtime_report report = {};
    bool initialized = false;
};

struct DecodeLoopStats {
    bool enabled = false;
    bool pass = true;
    int steps = 0;
    int slots = 0;
    uint64_t slot_steps = 0;
    uint64_t dense_loaded_bytes = 0;
    uint64_t ep_contribution_bytes = 0;
    uint64_t ep_return_bytes = 0;
    double total_ms = 0.0;
    double ms_per_step = 0.0;
    double tok_s = 0.0;
    double ep_ms_per_step = 0.0;
    double dense_ms_per_step = 0.0;
    double compose_ms_per_step = 0.0;
    int finite_bad = 0;
    uint64_t checksum = 0;
    bool ep_return_fp16 = false;
    bool fused_compose_sum = false;
    bool dense_hmma_compose = false;
    bool dense_f16_cublas_compose = false;
    bool dense_f16_cache_compose = false;
};

struct Options {
    const char *lib_path = "./build/turbomind-v100/libggml-turbomind.so";
    const char *pack_dir = nullptr;
    const char *contract_path = nullptr;
    const char *tm_index_path = nullptr;
    int devices[kGpus] = {0, 1, 2, 3, 4, 5, 6, 7};
    int slots = 32;
    int top_k = 6;
    int layer = 2;
    uint32_t kv_slot = 7;
    uint64_t position = 1024;
    int warmup = 5;
    int iters = 30;
    const char *dense_compute_tensor = nullptr;
    bool dense_compute_all_f8 = false;
    bool dense_compute_all_bf16 = false;
    bool compose_next_hidden = false;
    int decode_steps = 0;
    bool ep_return_fp16 = false;
    bool fuse_compose_sum = false;
    bool dense_hmma_compose = false;
    bool dense_f16_cublas_compose = false;
    bool dense_f16_cache_compose = false;
    bool all_layers = false;
    bool skip_descriptor_checks = false;
    bool skip_predecode_probes = false;
};

__global__ void checksum_bytes_kernel(const unsigned char *data, uint64_t n,
                                      unsigned long long *out) {
    unsigned long long local = 0;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (uint64_t)blockDim.x * gridDim.x) {
        local += (unsigned long long)data[i] * (unsigned long long)((i % 251u) + 1u);
    }
    atomicAdd(out, local);
}

__device__ float f8_e8m0_to_f32_dev(uint8_t e) {
    return __uint_as_float(e == 0 ? 0x00400000u : ((uint32_t)e << 23));
}

__device__ float f8_e4m3fn_to_f32_dev(uint8_t x) {
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

__device__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ float block_sum_256_f32(float v) {
    __shared__ float warp_sums[8];
    v = warp_sum_f32(v);
    if ((threadIdx.x & 31u) == 0u) warp_sums[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < 8u ? warp_sums[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32u) v = warp_sum_f32(v);
    return v;
}

__global__ void f8_b128_dense_kernel(float *out,
                                     const uint8_t *weights,
                                     const float *x,
                                     uint32_t rows,
                                     uint32_t cols,
                                     uint32_t row_stride_bytes,
                                     uint32_t slots) {
    const uint32_t row = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (row >= rows || slot >= slots) return;
    const uint8_t *wrow = weights + (uint64_t)row * row_stride_bytes;
    const float *xrow = x + (uint64_t)slot * cols;
    float acc = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const uint8_t *block = wrow + (uint64_t)(c / 128u) * 129ull;
        const float scale = f8_e8m0_to_f32_dev(block[0]);
        const float w = f8_e4m3fn_to_f32_dev(block[1u + (c % 128u)]) * scale;
        acc += w * xrow[c];
    }
    acc = block_sum_256_f32(acc);
    if (threadIdx.x == 0u) out[(uint64_t)slot * rows + row] = acc;
}

__global__ void f8_b128_dense_hmma_m16_kernel(float *out,
                                              const uint8_t *weights,
                                              const float *x,
                                              uint32_t rows,
                                              uint32_t cols,
                                              uint32_t row_stride_bytes,
                                              uint32_t slots) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    enum {
        WARPS_PER_BLOCK = 4,
        TILE_M = 16,
        TILE_N = 16,
        TILE_K = 16,
        ROWS_PER_BLOCK = WARPS_PER_BLOCK * TILE_N,
    };

    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (warp >= WARPS_PER_BLOCK) return;

    const uint32_t row_block = blockIdx.x * ROWS_PER_BLOCK;
    const uint32_t token_block = blockIdx.y * TILE_M;

    __shared__ __half a_sh[TILE_M * TILE_K];
    __shared__ __half b_sh[WARPS_PER_BLOCK * TILE_K * TILE_N];
    __shared__ float c_sh[WARPS_PER_BLOCK * TILE_M * TILE_N];

    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (uint32_t k0 = 0; k0 < cols; k0 += TILE_K) {
        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            const uint32_t token = i >> 4u;
            const uint32_t k = i & 15u;
            const uint32_t global_token = token_block + token;
            float v = 0.0f;
            if (global_token < slots) {
                v = x[(uint64_t)global_token * cols + k0 + k];
            }
            a_sh[i] = __float2half_rn(v);
        }

        for (uint32_t i = tid; i < WARPS_PER_BLOCK * TILE_K * TILE_N; i += blockDim.x) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t out_col = local >> 4u;
            const uint32_t k = local & 15u;
            const uint32_t row = row_block + wtile * TILE_N + out_col;
            float w = 0.0f;
            if (row < rows) {
                const uint32_t col = k0 + k;
                const uint8_t *row_base = weights + (uint64_t)row * row_stride_bytes;
                const uint8_t *block = row_base + (uint64_t)(col >> 7u) * 129ull;
                w = f8_e4m3fn_to_f32_dev(block[1u + (col & 127u)]) *
                    f8_e8m0_to_f32_dev(block[0]);
            }
            b_sh[i] = __float2half_rn(w);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, a_sh, TILE_K);
        wmma::load_matrix_sync(b_frag, b_sh + warp * TILE_K * TILE_N, TILE_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(c_sh + warp * TILE_M * TILE_N,
                            c_frag,
                            TILE_N,
                            wmma::mem_row_major);
    __syncthreads();

    for (uint32_t i = tid; i < WARPS_PER_BLOCK * TILE_M * TILE_N; i += blockDim.x) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t token = local >> 4u;
        const uint32_t out_col = local & 15u;
        const uint32_t global_token = token_block + token;
        const uint32_t row = row_block + wtile * TILE_N + out_col;
        if (global_token < slots && row < rows) {
            out[(uint64_t)global_token * rows + row] =
                c_sh[wtile * TILE_M * TILE_N + local];
        }
    }
#else
    (void)out;
    (void)weights;
    (void)x;
    (void)rows;
    (void)cols;
    (void)row_stride_bytes;
    (void)slots;
#endif
}

__global__ void f8_b128_to_half_kernel(__half *out,
                                       const uint8_t *weights,
                                       uint32_t rows,
                                       uint32_t cols,
                                       uint32_t row_stride_bytes) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)rows * cols;
    if (i >= n) return;
    const uint32_t row = (uint32_t)(i / cols);
    const uint32_t col = (uint32_t)(i - (uint64_t)row * cols);
    const uint8_t *row_base = weights + (uint64_t)row * row_stride_bytes;
    const uint8_t *block = row_base + (uint64_t)(col >> 7u) * 129ull;
    const float w = f8_e4m3fn_to_f32_dev(block[1u + (col & 127u)]) *
                    f8_e8m0_to_f32_dev(block[0]);
    out[i] = __float2half_rn(w);
}

__device__ float bf16_to_f32_dev(uint16_t v) {
    return __uint_as_float((uint32_t)v << 16);
}

__global__ void bf16_to_half_kernel(__half *out, const uint16_t *in, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(bf16_to_f32_dev(in[i]));
}

__global__ void bf16_dense_kernel(float *out,
                                  const uint16_t *weights,
                                  const float *x,
                                  uint32_t rows,
                                  uint32_t cols,
                                  uint32_t row_stride_elements,
                                  uint32_t slots) {
    const uint32_t row = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (row >= rows || slot >= slots) return;
    const uint16_t *wrow = weights + (uint64_t)row * row_stride_elements;
    const float *xrow = x + (uint64_t)slot * cols;
    float acc = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        acc += bf16_to_f32_dev(wrow[c]) * xrow[c];
    }
    acc = block_sum_256_f32(acc);
    if (threadIdx.x == 0u) out[(uint64_t)slot * rows + row] = acc;
}

__global__ void zero_f32_kernel(float *dst, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = 0.0f;
}

__global__ void ep_reduce_all_dest_shards_kernel(float *contrib,
                                                 const __half *route_hidden,
                                                 const int *route_slots,
                                                 int routes,
                                                 int slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)routes * kHidden;
    if (i >= total) return;
    const int route = (int)(i / kHidden);
    const int h = (int)(i % kHidden);
    const int slot = route_slots[route];
    if (slot < 0 || slot >= slots) return;
    const int dest = h / (kHidden / kGpus);
    const int local_h = h % (kHidden / kGpus);
    const uint64_t out_idx =
        ((uint64_t)dest * slots + (uint64_t)slot) * (kHidden / kGpus) + local_h;
    atomicAdd(contrib + out_idx, __half2float(route_hidden[i]));
}

__global__ void add_f32_kernel(float *dst, const float *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void cast_f32_to_half_kernel(__half *dst, const float *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void add_half_to_f32_kernel(float *dst, const __half *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += __half2float(src[i]);
}

__global__ void compose_next_hidden_kernel(float *next,
                                           const float *attn,
                                           const float *shared,
                                           const float *ep_sum,
                                           int rank,
                                           int slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t elems = (uint64_t)slots * (kHidden / kGpus);
    if (i >= elems) return;
    const int slot = (int)(i / (kHidden / kGpus));
    const int local_h = (int)(i % (kHidden / kGpus));
    const float residual =
        ((float)(rank + 1) * 0.01f) + ((float)slot * 0.001f) +
        ((float)local_h * 0.00001f);
    next[i] = residual + attn[i] + shared[i] + ep_sum[i] * 0.125f;
}

__global__ void compose_next_hidden_sum8_kernel(float *next,
                                                const float *attn,
                                                const float *shared,
                                                const float *r0,
                                                const float *r1,
                                                const float *r2,
                                                const float *r3,
                                                const float *r4,
                                                const float *r5,
                                                const float *r6,
                                                const float *r7,
                                                int rank,
                                                int slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t elems = (uint64_t)slots * (kHidden / kGpus);
    if (i >= elems) return;
    const int slot = (int)(i / (kHidden / kGpus));
    const int local_h = (int)(i % (kHidden / kGpus));
    const float residual =
        ((float)(rank + 1) * 0.01f) + ((float)slot * 0.001f) +
        ((float)local_h * 0.00001f);
    const float ep =
        r0[i] + r1[i] + r2[i] + r3[i] + r4[i] + r5[i] + r6[i] + r7[i];
    next[i] = residual + attn[i] + shared[i] + ep * 0.125f;
}

bool parse_int(const char *text, int *out) {
    if (!text || !*text) return false;
    char *end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = (int)v;
    return true;
}

bool parse_u64(const char *text, uint64_t *out) {
    if (!text || !*text) return false;
    char *end = nullptr;
    const unsigned long long v = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') return false;
    *out = (uint64_t)v;
    return true;
}

bool parse_size(const char *text, size_t *out) {
    uint64_t v = 0;
    if (!parse_u64(text, &v)) return false;
    if (v > (uint64_t)std::numeric_limits<size_t>::max()) return false;
    *out = (size_t)v;
    return true;
}

std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (start <= line.size()) {
        const size_t tab = line.find('\t', start);
        if (tab == std::string::npos) {
            fields.emplace_back(line.substr(start));
            break;
        }
        fields.emplace_back(line.substr(start, tab - start));
        start = tab + 1;
    }
    return fields;
}

bool safe_sidecar_name(const std::string &name) {
    return !name.empty() &&
           name.find('/') == std::string::npos &&
           name.find('\\') == std::string::npos &&
           name.find("..") == std::string::npos;
}

std::string path_join(const char *dir, const std::string &base) {
    std::string out(dir ? dir : "");
    if (!out.empty() && out.back() != '/') out.push_back('/');
    out += base;
    return out;
}

int read_exact_at(const std::string &path, uint64_t offset, void *dst, size_t bytes) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "cannot open sidecar %s: %s\n", path.c_str(), std::strerror(errno));
        return 1;
    }
    if (fseeko(fp, (off_t)offset, SEEK_SET) != 0) {
        std::fprintf(stderr, "cannot seek sidecar %s offset %llu: %s\n",
                     path.c_str(), (unsigned long long)offset, std::strerror(errno));
        std::fclose(fp);
        return 2;
    }
    const size_t got = std::fread(dst, 1, bytes, fp);
    if (got != bytes) {
        std::fprintf(stderr, "short read sidecar %s offset %llu bytes %zu got %zu\n",
                     path.c_str(), (unsigned long long)offset, bytes, got);
        std::fclose(fp);
        return 3;
    }
    std::fclose(fp);
    return 0;
}

bool parse_devices(const char *text, int devices[kGpus]) {
    std::vector<int> parsed;
    const char *cur = text;
    while (cur && *cur) {
        const char *comma = std::strchr(cur, ',');
        std::string piece;
        if (comma) {
            piece.assign(cur, comma - cur);
            cur = comma + 1;
        } else {
            piece.assign(cur);
            cur = nullptr;
        }
        int dev = 0;
        if (!parse_int(piece.c_str(), &dev) || dev < 0) return false;
        parsed.push_back(dev);
    }
    if ((int)parsed.size() != kGpus) return false;
    for (int i = 0; i < kGpus; ++i) {
        for (int j = i + 1; j < kGpus; ++j) {
            if (parsed[i] == parsed[j]) return false;
        }
        devices[i] = parsed[i];
    }
    return true;
}

void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s --pack-dir DIR --contract FILE --tm-index FILE [options]\n"
                 "       [--lib PATH] [--devices 0,1,2,3,4,5,6,7]\n"
                 "       [--slots N] [--top-k N] [--layer N] [--kv-slot N]\n"
                 "       [--position N] [--warmup N] [--iters N]\n"
                 "       [--dense-compute-tensor NAME] [--dense-compute-all-f8]\n"
                 "       [--dense-compute-all-bf16] [--dense-compute-all]\n"
                 "       [--compose-next-hidden] [--decode-steps N]\n"
                 "       [--ep-return-fp16] [--fuse-compose-sum]\n"
                 "       [--dense-hmma-compose] [--dense-f16-cublas-compose]\n"
                 "       [--dense-f16-cache-compose] [--all-layers]\n"
                 "       [--skip-descriptor-checks] [--skip-predecode-probes]\n",
                 argv0);
}

bool parse_args(int argc, char **argv, Options *opt) {
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        const char *val = (i + 1 < argc) ? argv[i + 1] : nullptr;
        if (std::strcmp(arg, "--lib") == 0) {
            if (!val) return false;
            opt->lib_path = val;
            ++i;
        } else if (std::strcmp(arg, "--pack-dir") == 0) {
            if (!val) return false;
            opt->pack_dir = val;
            ++i;
        } else if (std::strcmp(arg, "--contract") == 0) {
            if (!val) return false;
            opt->contract_path = val;
            ++i;
        } else if (std::strcmp(arg, "--tm-index") == 0) {
            if (!val) return false;
            opt->tm_index_path = val;
            ++i;
        } else if (std::strcmp(arg, "--devices") == 0) {
            if (!val || !parse_devices(val, opt->devices)) return false;
            ++i;
        } else if (std::strcmp(arg, "--slots") == 0) {
            if (!val || !parse_int(val, &opt->slots) || opt->slots <= 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--top-k") == 0) {
            if (!val || !parse_int(val, &opt->top_k) || opt->top_k <= 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--layer") == 0) {
            if (!val || !parse_int(val, &opt->layer)) return false;
            ++i;
        } else if (std::strcmp(arg, "--kv-slot") == 0) {
            int slot = 0;
            if (!val || !parse_int(val, &slot) || slot < 0) return false;
            opt->kv_slot = (uint32_t)slot;
            ++i;
        } else if (std::strcmp(arg, "--position") == 0) {
            if (!val || !parse_u64(val, &opt->position)) return false;
            ++i;
        } else if (std::strcmp(arg, "--warmup") == 0) {
            if (!val || !parse_int(val, &opt->warmup) || opt->warmup < 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--iters") == 0) {
            if (!val || !parse_int(val, &opt->iters) || opt->iters <= 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--dense-compute-tensor") == 0) {
            if (!val) return false;
            opt->dense_compute_tensor = val;
            ++i;
        } else if (std::strcmp(arg, "--dense-compute-all-f8") == 0) {
            opt->dense_compute_all_f8 = true;
        } else if (std::strcmp(arg, "--dense-compute-all-bf16") == 0) {
            opt->dense_compute_all_bf16 = true;
        } else if (std::strcmp(arg, "--dense-compute-all") == 0) {
            opt->dense_compute_all_f8 = true;
            opt->dense_compute_all_bf16 = true;
        } else if (std::strcmp(arg, "--compose-next-hidden") == 0) {
            opt->compose_next_hidden = true;
        } else if (std::strcmp(arg, "--decode-steps") == 0) {
            if (!val || !parse_int(val, &opt->decode_steps) || opt->decode_steps < 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--ep-return-fp16") == 0) {
            opt->ep_return_fp16 = true;
        } else if (std::strcmp(arg, "--fuse-compose-sum") == 0) {
            opt->fuse_compose_sum = true;
        } else if (std::strcmp(arg, "--dense-hmma-compose") == 0) {
            opt->dense_hmma_compose = true;
        } else if (std::strcmp(arg, "--dense-f16-cublas-compose") == 0) {
            opt->dense_f16_cublas_compose = true;
        } else if (std::strcmp(arg, "--dense-f16-cache-compose") == 0) {
            opt->dense_f16_cache_compose = true;
        } else if (std::strcmp(arg, "--all-layers") == 0) {
            opt->all_layers = true;
        } else if (std::strcmp(arg, "--skip-descriptor-checks") == 0) {
            opt->skip_descriptor_checks = true;
        } else if (std::strcmp(arg, "--skip-predecode-probes") == 0) {
            opt->skip_predecode_probes = true;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }
    return opt->pack_dir && opt->contract_path && opt->tm_index_path &&
           opt->top_k <= kActiveLocalExperts && opt->layer >= 0 &&
           !(opt->dense_hmma_compose && opt->dense_f16_cublas_compose) &&
           (!opt->dense_f16_cache_compose || opt->dense_f16_cublas_compose) &&
           !(opt->dense_compute_tensor &&
             (opt->dense_compute_all_f8 || opt->dense_compute_all_bf16));
}

bool parse_contract_row(const std::vector<std::string> &f, ContractRow *out) {
    if (f.size() < 23) return false;
    ContractRow r;
    r.record_type = f[0];
    r.tensor_id = f[1];
    if (!parse_int(f[3].c_str(), &r.layer)) return false;
    r.family = f[4];
    r.source_dtype = f[5];
    r.source_shape = f[6];
    r.runtime_layout = f[7];
    if (!parse_int(f[8].c_str(), &r.owning_gpu)) return false;
    if (!parse_int(f[9].c_str(), &r.tp_rank)) return false;
    if (!parse_int(f[10].c_str(), &r.ep_rank)) return false;
    if (!parse_int(f[12].c_str(), &r.shard_index)) return false;
    if (!parse_int(f[13].c_str(), &r.shard_count)) return false;
    if (!parse_int(f[14].c_str(), &r.expert_first)) return false;
    if (!parse_int(f[15].c_str(), &r.expert_count)) return false;
    if (!parse_int(f[16].c_str(), &r.kv_ratio)) return false;
    if (!parse_u64(f[17].c_str(), &r.kv_rows_per_slot)) return false;
    if (!parse_u64(f[18].c_str(), &r.bytes_estimate)) return false;
    r.source_pack_file = f[19];
    if (!parse_u64(f[20].c_str(), &r.source_shard_offset)) return false;
    if (!parse_u64(f[21].c_str(), &r.source_byte_length)) return false;
    r.kernel_family = f[22];
    if (!safe_sidecar_name(r.source_pack_file) && r.source_pack_file != "-") return false;
    *out = r;
    return true;
}

int parse_contract(const char *path, int layer, std::vector<ContractRow> *rows,
                   LayerStats *stats) {
    FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "cannot open contract %s: %s\n", path, std::strerror(errno));
        return 1;
    }
    char buf[8192];
    bool first = true;
    while (std::fgets(buf, sizeof(buf), fp)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (first) {
            first = false;
            continue;
        }
        if (line.empty()) continue;
        std::vector<std::string> f = split_tabs(line);
        ContractRow r;
        if (!parse_contract_row(f, &r)) {
            stats->bad_rows++;
            continue;
        }
        if (layer >= 0 && r.layer != layer) continue;
        if (r.owning_gpu < 0 || r.owning_gpu >= kGpus) {
            stats->bad_rows++;
            continue;
        }
        rows->push_back(r);
        stats->total_rows++;
        GpuFamilyStats &g = stats->gpu[r.owning_gpu];
        if (r.record_type == "dense_tp") {
            stats->dense_rows++;
            g.dense_rows++;
            g.dense_bytes += r.bytes_estimate;
        } else if (r.record_type == "replicated_control") {
            stats->control_rows++;
            g.control_rows++;
            g.control_bytes += r.bytes_estimate;
        } else if (r.record_type == "ep_expert") {
            stats->expert_rows++;
            g.expert_rows++;
            g.expert_descriptor_bytes += r.bytes_estimate;
        } else if (r.record_type == "kv_shard") {
            stats->kv_rows++;
            g.kv_rows++;
        } else if (r.record_type == "kv_comp_state") {
            stats->comp_rows++;
            g.comp_rows++;
        }
    }
    std::fclose(fp);
    return rows->empty() ? 2 : 0;
}

uint64_t physical_row_offset(const ContractRow &r) {
    if (r.record_type == "dense_tp" && r.shard_index >= 0 && r.shard_count > 1 &&
        r.source_byte_length >= r.bytes_estimate * (uint64_t)r.shard_count) {
        return r.source_shard_offset + (uint64_t)r.shard_index * r.bytes_estimate;
    }
    return r.source_shard_offset;
}

bool parse_shape2(const std::string &shape, int *cols, int *rows) {
    if (shape.size() < 5 || shape.front() != '[' || shape.back() != ']') return false;
    const size_t x = shape.find('x');
    if (x == std::string::npos) return false;
    std::string a = shape.substr(1, x - 1);
    std::string b = shape.substr(x + 1, shape.size() - x - 2);
    return parse_int(a.c_str(), cols) && parse_int(b.c_str(), rows) &&
           *cols > 0 && *rows > 0;
}

std::string layer_tensor_name(int layer, const char *suffix) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "blk.%d.%s", layer, suffix);
    return std::string(buf);
}

int ds4_layer_ratio(int layer) {
    if (layer < 2) return 0;
    return (layer % 2) == 0 ? 4 : 128;
}

uint64_t f8_row_bytes(int cols) {
    return (uint64_t)(cols / 128) * 129ull;
}

float e8m0_to_f32_host(uint8_t e) {
    uint32_t bits = e == 0 ? 0x00400000u : ((uint32_t)e << 23);
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

float e4m3fn_to_f32_host(uint8_t x) {
    const uint8_t ax = x & 0x7fu;
    const bool sign = (x & 0x80u) != 0;
    if (ax == 0) return sign ? -0.0f : 0.0f;
    if (ax == 0x7f) return std::numeric_limits<float>::quiet_NaN();
    const int exp = (x >> 3) & 0x0f;
    const int man = x & 0x07;
    const float value = exp == 0 ? std::ldexp((float)man, -9)
                                 : std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
    return sign ? -value : value;
}

float cpu_f8_dot(const uint8_t *row, const float *x, int cols) {
    double acc = 0.0;
    const int blocks = cols / 128;
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *block = row + (uint64_t)b * 129ull;
        const float scale = e8m0_to_f32_host(block[0]);
        for (int c = 0; c < 128; ++c) {
            acc += (double)(e4m3fn_to_f32_host(block[1 + c]) * scale) *
                   (double)x[b * 128 + c];
        }
    }
    return (float)acc;
}

float bf16_to_f32_host(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float v = 0.0f;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

float cpu_bf16_dot(const uint16_t *row, const float *x, int cols) {
    double acc = 0.0;
    for (int c = 0; c < cols; ++c) {
        acc += (double)bf16_to_f32_host(row[c]) * (double)x[c];
    }
    return (float)acc;
}

int device_checksum_row(int device, const char *pack_dir, const ContractRow &r,
                        uint64_t *checksum) {
    if (r.bytes_estimate == 0 || r.source_pack_file == "-") return 0;
    CHECK_CUDA(cudaSetDevice(device));
    const uint64_t offset = physical_row_offset(r);
    if (offset + r.bytes_estimate > r.source_shard_offset + r.source_byte_length &&
        r.record_type == "dense_tp") {
        std::fprintf(stderr, "dense shard exceeds source span for %s\n", r.tensor_id.c_str());
        return 1;
    }
    std::vector<unsigned char> host((size_t)r.bytes_estimate);
    const std::string path = path_join(pack_dir, r.source_pack_file);
    if (read_exact_at(path, offset, host.data(), host.size()) != 0) return 2;

    unsigned char *d = nullptr;
    unsigned long long *d_sum = nullptr;
    CHECK_CUDA(cudaMalloc(&d, host.size()));
    CHECK_CUDA(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemcpy(d, host.data(), host.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    const int block = 256;
    const int grid = (int)std::min<uint64_t>(4096, (r.bytes_estimate + block - 1) / block);
    checksum_bytes_kernel<<<std::max(grid, 1), block>>>(d, r.bytes_estimate, d_sum);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    unsigned long long h_sum = 0;
    CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(h_sum), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d));
    CHECK_CUDA(cudaFree(d_sum));
    *checksum = (uint64_t)h_sum;
    return 0;
}

bool select_dense_rows(const std::vector<ContractRow> &rows,
                       const char *tensor,
                       std::vector<ContractRow> *selected,
                       int *cols,
                       int *total_rows) {
    selected->clear();
    for (const ContractRow &r : rows) {
        if (r.record_type == "dense_tp" && r.tensor_id == tensor) selected->push_back(r);
    }
    if ((int)selected->size() != kGpus) return false;
    std::sort(selected->begin(), selected->end(),
              [](const ContractRow &a, const ContractRow &b) {
                  return a.owning_gpu < b.owning_gpu;
              });
    int parsed_cols = 0;
    int parsed_rows = 0;
    if (!parse_shape2((*selected)[0].source_shape, &parsed_cols, &parsed_rows)) return false;
    for (int i = 0; i < kGpus; ++i) {
        const ContractRow &r = (*selected)[i];
        if (r.owning_gpu != i ||
            r.tp_rank != i ||
            r.shard_index != i ||
            r.shard_count != kGpus ||
            r.source_dtype != "f8_e4m3_b128" ||
            r.source_shape != (*selected)[0].source_shape) {
            return false;
        }
    }
    if (parsed_cols % 128 != 0 || parsed_rows % kGpus != 0) return false;
    const uint64_t row_bytes = f8_row_bytes(parsed_cols);
    const uint64_t rows_per_gpu = (uint64_t)parsed_rows / kGpus;
    for (const ContractRow &r : *selected) {
        if (r.bytes_estimate != row_bytes * rows_per_gpu) return false;
    }
    *cols = parsed_cols;
    *total_rows = parsed_rows;
    return true;
}

std::vector<std::string> discover_f8_dense_tensors(const std::vector<ContractRow> &rows) {
    std::vector<std::string> out;
    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" || r.source_dtype != "f8_e4m3_b128") continue;
        if (std::find(out.begin(), out.end(), r.tensor_id) == out.end()) {
            out.push_back(r.tensor_id);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool select_bf16_dense_rows(const std::vector<ContractRow> &rows,
                            const char *tensor,
                            std::vector<ContractRow> *selected,
                            int *cols,
                            int *total_rows) {
    selected->clear();
    for (const ContractRow &r : rows) {
        if (r.record_type == "dense_tp" && r.tensor_id == tensor) selected->push_back(r);
    }
    if ((int)selected->size() != kGpus) return false;
    std::sort(selected->begin(), selected->end(),
              [](const ContractRow &a, const ContractRow &b) {
                  return a.owning_gpu < b.owning_gpu;
              });
    int parsed_cols = 0;
    int parsed_rows = 0;
    if (!parse_shape2((*selected)[0].source_shape, &parsed_cols, &parsed_rows)) return false;
    for (int i = 0; i < kGpus; ++i) {
        const ContractRow &r = (*selected)[i];
        if (r.owning_gpu != i ||
            r.tp_rank != i ||
            r.shard_index != i ||
            r.shard_count != kGpus ||
            r.source_dtype != "bf16" ||
            r.source_shape != (*selected)[0].source_shape) {
            return false;
        }
    }
    if (parsed_rows % kGpus != 0) return false;
    const uint64_t rows_per_gpu = (uint64_t)parsed_rows / kGpus;
    const uint64_t shard_bytes = rows_per_gpu * (uint64_t)parsed_cols * sizeof(uint16_t);
    for (const ContractRow &r : *selected) {
        if (r.bytes_estimate != shard_bytes) return false;
    }
    *cols = parsed_cols;
    *total_rows = parsed_rows;
    return true;
}

std::vector<std::string> discover_bf16_dense_tensors(const std::vector<ContractRow> &rows) {
    std::vector<std::string> out;
    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" || r.source_dtype != "bf16") continue;
        if (std::find(out.begin(), out.end(), r.tensor_id) == out.end()) {
            out.push_back(r.tensor_id);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

int run_dense_compute_gate(const Options &opt,
                           const std::vector<ContractRow> &rows,
                           const char *tensor,
                           DenseComputeStats *stats) {
    if (!tensor) return 0;
    stats->enabled = true;
    stats->tensor_id = tensor;
    stats->slots = opt.slots;

    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    if (!select_dense_rows(rows, tensor, &selected, &cols, &total_rows)) {
        std::fprintf(stderr, "dense compute tensor validation failed for %s\n",
                     tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    const uint64_t row_bytes = f8_row_bytes(cols);
    const uint64_t shard_bytes = row_bytes * (uint64_t)rows_per_gpu;
    stats->rows_per_gpu = rows_per_gpu;
    stats->cols = cols;

    std::vector<float> h_x((size_t)opt.slots * cols);
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int c = 0; c < cols; ++c) {
            const int m = (slot * 17 + c * 13) % 257;
            h_x[(size_t)slot * cols + c] = ((float)m - 128.0f) * 0.00025f;
        }
    }

    double worst_ms = 0.0;
    std::vector<std::vector<uint8_t>> host_weights((size_t)kGpus);
    std::vector<std::vector<float>> host_outputs((size_t)kGpus);

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = selected[(size_t)gpu];
        host_weights[(size_t)gpu].resize((size_t)shard_bytes);
        host_outputs[(size_t)gpu].resize((size_t)opt.slots * rows_per_gpu);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r),
                          host_weights[(size_t)gpu].data(), (size_t)shard_bytes) != 0) {
            return 2;
        }
        stats->loaded_bytes += shard_bytes;

        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        uint8_t *d_w = nullptr;
        float *d_x = nullptr;
        float *d_out1 = nullptr;
        float *d_out2 = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaMalloc(&d_w, (size_t)shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out1, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out2, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_w, host_weights[(size_t)gpu].data(), (size_t)shard_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        const dim3 grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1);
        for (int i = 0; i < opt.warmup; ++i) {
            f8_b128_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                                cols, (uint32_t)row_bytes, opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < opt.iters; ++i) {
            f8_b128_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                                cols, (uint32_t)row_bytes, opt.slots);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        worst_ms = std::max(worst_ms, (double)ms / opt.iters);

        f8_b128_dense_kernel<<<grid, 256>>>(d_out2, d_w, d_x, rows_per_gpu,
                                            cols, (uint32_t)row_bytes, opt.slots);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> second(host_outputs[(size_t)gpu].size());
        CHECK_CUDA(cudaMemcpy(host_outputs[(size_t)gpu].data(), d_out1,
                              host_outputs[(size_t)gpu].size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(second.data(), d_out2,
                              second.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < second.size(); ++i) {
            const float a = host_outputs[(size_t)gpu][i];
            const float b = second[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                stats->repeat_nan++;
                stats->pass = false;
                continue;
            }
            const double diff = std::fabs((double)a - (double)b);
            stats->repeat_max_abs = std::max(stats->repeat_max_abs, diff);
            if (diff > 0.0) {
                stats->repeat_bad++;
                stats->pass = false;
            }
        }

        const int sample_slots = std::min(opt.slots, 2);
        const int sample_rows = std::min(rows_per_gpu, 4);
        for (int slot = 0; slot < sample_slots; ++slot) {
            for (int row = 0; row < sample_rows; ++row) {
                const float expected =
                    cpu_f8_dot(host_weights[(size_t)gpu].data() + (uint64_t)row * row_bytes,
                               h_x.data() + (size_t)slot * cols, cols);
                const float got = host_outputs[(size_t)gpu][(size_t)slot * rows_per_gpu + row];
                const double diff = std::fabs((double)expected - (double)got);
                stats->oracle_max_abs = std::max(stats->oracle_max_abs, diff);
                const double tol = 1.0e-4 + std::fabs((double)expected) * 1.0e-4;
                if (!std::isfinite(got) || diff > tol) {
                    stats->oracle_bad++;
                    stats->pass = false;
                }
            }
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_out1));
        CHECK_CUDA(cudaFree(d_out2));
    }

    stats->compute_ms = worst_ms;
    return stats->pass ? 0 : 3;
}

int run_bf16_dense_compute_gate(const Options &opt,
                                const std::vector<ContractRow> &rows,
                                const char *tensor,
                                DenseComputeStats *stats) {
    if (!tensor) return 0;
    stats->enabled = true;
    stats->tensor_id = tensor;
    stats->slots = opt.slots;

    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    if (!select_bf16_dense_rows(rows, tensor, &selected, &cols, &total_rows)) {
        std::fprintf(stderr, "bf16 dense compute tensor validation failed for %s\n",
                     tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    const uint64_t shard_bytes = (uint64_t)rows_per_gpu * cols * sizeof(uint16_t);
    stats->rows_per_gpu = rows_per_gpu;
    stats->cols = cols;

    std::vector<float> h_x((size_t)opt.slots * cols);
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int c = 0; c < cols; ++c) {
            const int m = (slot * 19 + c * 11) % 263;
            h_x[(size_t)slot * cols + c] = ((float)m - 131.0f) * 0.00025f;
        }
    }

    double worst_ms = 0.0;
    std::vector<std::vector<uint16_t>> host_weights((size_t)kGpus);
    std::vector<std::vector<float>> host_outputs((size_t)kGpus);

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = selected[(size_t)gpu];
        host_weights[(size_t)gpu].resize((size_t)rows_per_gpu * cols);
        host_outputs[(size_t)gpu].resize((size_t)opt.slots * rows_per_gpu);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r),
                          host_weights[(size_t)gpu].data(), (size_t)shard_bytes) != 0) {
            return 2;
        }
        stats->loaded_bytes += shard_bytes;

        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        uint16_t *d_w = nullptr;
        float *d_x = nullptr;
        float *d_out1 = nullptr;
        float *d_out2 = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaMalloc(&d_w, (size_t)shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out1, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out2, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_w, host_weights[(size_t)gpu].data(), (size_t)shard_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        const dim3 grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1);
        for (int i = 0; i < opt.warmup; ++i) {
            bf16_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                             cols, cols, opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < opt.iters; ++i) {
            bf16_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                             cols, cols, opt.slots);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        worst_ms = std::max(worst_ms, (double)ms / opt.iters);

        bf16_dense_kernel<<<grid, 256>>>(d_out2, d_w, d_x, rows_per_gpu,
                                         cols, cols, opt.slots);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> second(host_outputs[(size_t)gpu].size());
        CHECK_CUDA(cudaMemcpy(host_outputs[(size_t)gpu].data(), d_out1,
                              host_outputs[(size_t)gpu].size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(second.data(), d_out2,
                              second.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < second.size(); ++i) {
            const float a = host_outputs[(size_t)gpu][i];
            const float b = second[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                stats->repeat_nan++;
                stats->pass = false;
                continue;
            }
            const double diff = std::fabs((double)a - (double)b);
            stats->repeat_max_abs = std::max(stats->repeat_max_abs, diff);
            if (diff > 0.0) {
                stats->repeat_bad++;
                stats->pass = false;
            }
        }

        const int sample_slots = std::min(opt.slots, 2);
        const int sample_rows = std::min(rows_per_gpu, 4);
        for (int slot = 0; slot < sample_slots; ++slot) {
            for (int row = 0; row < sample_rows; ++row) {
                const float expected =
                    cpu_bf16_dot(host_weights[(size_t)gpu].data() + (uint64_t)row * cols,
                                 h_x.data() + (size_t)slot * cols, cols);
                const float got = host_outputs[(size_t)gpu][(size_t)slot * rows_per_gpu + row];
                const double diff = std::fabs((double)expected - (double)got);
                stats->oracle_max_abs = std::max(stats->oracle_max_abs, diff);
                const double tol = 1.0e-4 + std::fabs((double)expected) * 1.0e-4;
                if (!std::isfinite(got) || diff > tol) {
                    stats->oracle_bad++;
                    stats->pass = false;
                }
            }
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_out1));
        CHECK_CUDA(cudaFree(d_out2));
    }

    stats->compute_ms = worst_ms;
    return stats->pass ? 0 : 3;
}

void free_device_dense_outputs(DeviceDenseOutputs &out, const Options &opt) {
    for (int gpu = 0; gpu < (int)out.d_out.size(); ++gpu) {
        if (!out.d_out[(size_t)gpu]) continue;
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaFree(out.d_out[(size_t)gpu]));
    }
    out = DeviceDenseOutputs{};
}

void free_resident_f8_dense(ResidentF8Dense &op, const Options &opt) {
    for (int gpu = 0; gpu < (int)op.d_w.size(); ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (op.d_w[(size_t)gpu]) CHECK_CUDA(cudaFree(op.d_w[(size_t)gpu]));
        if (op.d_x[(size_t)gpu]) CHECK_CUDA(cudaFree(op.d_x[(size_t)gpu]));
        if (gpu < (int)op.d_w_half.size() && op.d_w_half[(size_t)gpu]) {
            const bool owns = gpu >= (int)op.owns_w_half.size() || op.owns_w_half[(size_t)gpu];
            if (owns) CHECK_CUDA(cudaFree(op.d_w_half[(size_t)gpu]));
        }
        if (gpu < (int)op.d_x_half.size() && op.d_x_half[(size_t)gpu]) {
            CHECK_CUDA(cudaFree(op.d_x_half[(size_t)gpu]));
        }
        if (op.d_out[(size_t)gpu]) CHECK_CUDA(cudaFree(op.d_out[(size_t)gpu]));
        if (gpu < (int)op.cublas.size() && op.cublas[(size_t)gpu]) {
            (void)cublasDestroy(op.cublas[(size_t)gpu]);
        }
    }
    op = ResidentF8Dense{};
}

uint64_t align_up_u64(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

void free_dense_f16_cache(DenseF16Cache &cache, const Options &opt) {
    for (int gpu = 0; gpu < (int)cache.arena.size(); ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (cache.arena[(size_t)gpu]) CHECK_CUDA(cudaFree(cache.arena[(size_t)gpu]));
        if (gpu < (int)cache.temp.size() && cache.temp[(size_t)gpu]) {
            CHECK_CUDA(cudaFree(cache.temp[(size_t)gpu]));
        }
    }
    cache = DenseF16Cache{};
}

const DenseF16CacheEntry *find_dense_f16_cache_entry(const DenseF16Cache &cache,
                                                     const char *tensor,
                                                     int gpu) {
    if (!cache.enabled) return nullptr;
    for (const DenseF16CacheEntry &e : cache.entries) {
        if (e.gpu == gpu && e.tensor_id == tensor) return &e;
    }
    return nullptr;
}

int prepare_dense_f16_cache(const Options &opt,
                            const std::vector<ContractRow> &rows,
                            DenseF16Cache *cache) {
    if (!opt.dense_f16_cache_compose) return 0;
    cache->enabled = true;
    cache->arena.assign((size_t)kGpus, nullptr);
    cache->temp.assign((size_t)kGpus, nullptr);
    uint64_t gpu_offsets[kGpus] = {};
    uint64_t gpu_temp[kGpus] = {};

    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" ||
            (r.source_dtype != "f8_e4m3_b128" && r.source_dtype != "bf16")) {
            continue;
        }
        int cols = 0;
        int total_rows = 0;
        if (!parse_shape2(r.source_shape, &cols, &total_rows)) continue;
        uint64_t rows_per_gpu = 0;
        if (r.source_dtype == "f8_e4m3_b128") {
            if (cols % 128 != 0) continue;
            const uint64_t rb = f8_row_bytes(cols);
            if (rb == 0 || r.bytes_estimate % rb != 0) continue;
            rows_per_gpu = r.bytes_estimate / rb;
        } else {
            const uint64_t rb = (uint64_t)cols * sizeof(uint16_t);
            if (rb == 0 || r.bytes_estimate % rb != 0) continue;
            rows_per_gpu = r.bytes_estimate / rb;
        }
        DenseF16CacheEntry e;
        e.tensor_id = r.tensor_id;
        e.gpu = r.owning_gpu;
        e.cols = cols;
        e.rows_per_gpu = (int)rows_per_gpu;
        e.offset = gpu_offsets[r.owning_gpu];
        e.source_bytes = r.bytes_estimate;
        e.cache_bytes = rows_per_gpu * (uint64_t)cols * sizeof(__half);
        cache->entries.push_back(e);
        cache->rows++;
        cache->source_bytes += e.source_bytes;
        cache->cache_bytes += e.cache_bytes;
        const uint64_t aligned = align_up_u64(e.cache_bytes, 256);
        gpu_offsets[r.owning_gpu] += aligned;
        cache->cache_aligned_bytes += aligned;
        gpu_temp[r.owning_gpu] = std::max(gpu_temp[r.owning_gpu], e.source_bytes);
        cache->max_temp_bytes = std::max(cache->max_temp_bytes, e.source_bytes);
    }

    if (cache->entries.empty()) return 1;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (gpu_offsets[gpu]) CHECK_CUDA(cudaMalloc(&cache->arena[(size_t)gpu],
                                                    (size_t)gpu_offsets[gpu]));
        if (gpu_temp[gpu]) CHECK_CUDA(cudaMalloc(&cache->temp[(size_t)gpu],
                                                 (size_t)gpu_temp[gpu]));
    }

    std::vector<uint8_t> host;
    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" ||
            (r.source_dtype != "f8_e4m3_b128" && r.source_dtype != "bf16")) {
            continue;
        }
        const DenseF16CacheEntry *e =
            find_dense_f16_cache_entry(*cache, r.tensor_id.c_str(), r.owning_gpu);
        if (!e || e->source_bytes != r.bytes_estimate) continue;
        host.resize((size_t)r.bytes_estimate);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), host.data(), host.size()) != 0) {
            free_dense_f16_cache(*cache, opt);
            return 2;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[r.owning_gpu]));
        CHECK_CUDA(cudaMemcpy(cache->temp[(size_t)r.owning_gpu], host.data(), host.size(),
                              cudaMemcpyHostToDevice));
        __half *dst =
            reinterpret_cast<__half *>(cache->arena[(size_t)r.owning_gpu] + e->offset);
        const uint64_t elems = e->cache_bytes / sizeof(__half);
        const unsigned int grid = (unsigned int)((elems + 255) / 256);
        if (r.source_dtype == "f8_e4m3_b128") {
            f8_b128_to_half_kernel<<<grid, 256>>>(
                dst, cache->temp[(size_t)r.owning_gpu], e->rows_per_gpu,
                e->cols, (uint32_t)f8_row_bytes(e->cols));
        } else {
            bf16_to_half_kernel<<<grid, 256>>>(
                dst, reinterpret_cast<const uint16_t *>(cache->temp[(size_t)r.owning_gpu]),
                elems);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (cache->temp[(size_t)gpu]) {
            CHECK_CUDA(cudaFree(cache->temp[(size_t)gpu]));
            cache->temp[(size_t)gpu] = nullptr;
        }
    }
    return 0;
}

int prepare_resident_f8_dense(const Options &opt,
                              const std::vector<ContractRow> &rows,
                              const char *tensor,
                              int seed,
                              const DenseF16Cache *cache,
                              ResidentF8Dense *op) {
    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    if (!select_dense_rows(rows, tensor, &selected, &cols, &total_rows)) {
        std::fprintf(stderr, "resident dense tensor validation failed for %s\n", tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    if (rows_per_gpu != kHidden / kGpus) {
        std::fprintf(stderr, "resident dense tensor %s rows_per_gpu=%d expected=%d\n",
                     tensor, rows_per_gpu, kHidden / kGpus);
        return 2;
    }
    const uint64_t row_bytes = f8_row_bytes(cols);
    const uint64_t shard_bytes = row_bytes * (uint64_t)rows_per_gpu;
    op->d_w.assign((size_t)kGpus, nullptr);
    op->d_x.assign((size_t)kGpus, nullptr);
    op->d_w_half.assign((size_t)kGpus, nullptr);
    op->owns_w_half.assign((size_t)kGpus, true);
    op->d_x_half.assign((size_t)kGpus, nullptr);
    op->d_out.assign((size_t)kGpus, nullptr);
    op->cublas.assign((size_t)kGpus, nullptr);
    op->rows_per_gpu = rows_per_gpu;
    op->cols = cols;
    op->slots = opt.slots;
    op->row_bytes = row_bytes;

    std::vector<float> h_x((size_t)opt.slots * cols);
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int c = 0; c < cols; ++c) {
            const int m = (slot * (17 + seed) + c * (13 + seed * 3)) % 269;
            h_x[(size_t)slot * cols + c] = ((float)m - 134.0f) * 0.0002f;
        }
    }

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = selected[(size_t)gpu];
        const DenseF16CacheEntry *cache_entry =
            opt.dense_f16_cache_compose && opt.dense_f16_cublas_compose && cache
                ? find_dense_f16_cache_entry(*cache, tensor, gpu)
                : nullptr;
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (!cache_entry) {
            std::vector<uint8_t> h_w((size_t)shard_bytes);
            const std::string path = path_join(opt.pack_dir, r.source_pack_file);
            if (read_exact_at(path, physical_row_offset(r), h_w.data(), h_w.size()) != 0) {
                free_resident_f8_dense(*op, opt);
                return 3;
            }
            op->loaded_bytes += shard_bytes;
            CHECK_CUDA(cudaMalloc(&op->d_w[(size_t)gpu], (size_t)shard_bytes));
            CHECK_CUDA(cudaMemcpy(op->d_w[(size_t)gpu], h_w.data(), (size_t)shard_bytes,
                                  cudaMemcpyHostToDevice));
        } else {
            op->loaded_bytes += cache_entry->source_bytes;
        }
        CHECK_CUDA(cudaMalloc(&op->d_x[(size_t)gpu], h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&op->d_out[(size_t)gpu],
                              (size_t)opt.slots * rows_per_gpu * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(op->d_x[(size_t)gpu], h_x.data(),
                              h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
        if (opt.dense_f16_cublas_compose) {
            (void)cudaGetLastError();
            if (cache_entry) {
                if (cache_entry->cols != cols || cache_entry->rows_per_gpu != rows_per_gpu) {
                    free_resident_f8_dense(*op, opt);
                    return 4;
                }
                op->d_w_half[(size_t)gpu] =
                    reinterpret_cast<__half *>(cache->arena[(size_t)gpu] + cache_entry->offset);
                op->owns_w_half[(size_t)gpu] = false;
            } else {
                CHECK_CUDA(cudaMalloc(&op->d_w_half[(size_t)gpu],
                                      (size_t)rows_per_gpu * cols * sizeof(__half)));
                op->owns_w_half[(size_t)gpu] = true;
                const uint64_t w_elems = (uint64_t)rows_per_gpu * cols;
                f8_b128_to_half_kernel<<<(unsigned int)((w_elems + 255) / 256), 256>>>(
                    op->d_w_half[(size_t)gpu], op->d_w[(size_t)gpu],
                    rows_per_gpu, cols, (uint32_t)row_bytes);
                CHECK_CUDA(cudaGetLastError());
            }
            CHECK_CUDA(cudaMalloc(&op->d_x_half[(size_t)gpu],
                                  h_x.size() * sizeof(__half)));
            const uint64_t x_elems = (uint64_t)opt.slots * cols;
            cast_f32_to_half_kernel<<<(unsigned int)((x_elems + 255) / 256), 256>>>(
                op->d_x_half[(size_t)gpu], op->d_x[(size_t)gpu], x_elems);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            cublasStatus_t st = cublasCreate(&op->cublas[(size_t)gpu]);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "cublasCreate failed on gpu %d: %d\n", gpu, (int)st);
                free_resident_f8_dense(*op, opt);
                return 4;
            }
            (void)cublasSetMathMode(op->cublas[(size_t)gpu], CUBLAS_TENSOR_OP_MATH);
        }
    }
    return 0;
}

int launch_resident_f8_dense(const Options &opt,
                             const ResidentF8Dense &op,
                             RankState ranks[kGpus]) {
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        (void)cudaGetLastError();
        if (opt.dense_f16_cublas_compose) {
            if (!op.cublas[(size_t)gpu] ||
                !op.d_w_half[(size_t)gpu] ||
                !op.d_x_half[(size_t)gpu]) {
                return 1;
            }
            cublasStatus_t st = cublasSetStream(op.cublas[(size_t)gpu], ranks[gpu].stream);
            if (st != CUBLAS_STATUS_SUCCESS) return 2;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            st = cublasGemmEx(op.cublas[(size_t)gpu],
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              op.rows_per_gpu,
                              op.slots,
                              op.cols,
                              &alpha,
                              op.d_w_half[(size_t)gpu],
                              CUDA_R_16F,
                              op.cols,
                              op.d_x_half[(size_t)gpu],
                              CUDA_R_16F,
                              op.cols,
                              &beta,
                              op.d_out[(size_t)gpu],
                              CUDA_R_32F,
                              op.rows_per_gpu,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "cublasGemmEx failed gpu=%d status=%d\n", gpu, (int)st);
                return 3;
            }
        } else if (opt.dense_hmma_compose) {
            const dim3 grid((unsigned int)((op.rows_per_gpu + 63) / 64),
                            (unsigned int)((op.slots + 15) / 16),
                            1);
            f8_b128_dense_hmma_m16_kernel<<<grid, 128, 0, ranks[gpu].stream>>>(
                op.d_out[(size_t)gpu], op.d_w[(size_t)gpu], op.d_x[(size_t)gpu],
                op.rows_per_gpu, op.cols, (uint32_t)op.row_bytes, op.slots);
        } else {
            const dim3 grid((unsigned int)op.rows_per_gpu, (unsigned int)op.slots, 1);
            f8_b128_dense_kernel<<<grid, 256, 0, ranks[gpu].stream>>>(
                op.d_out[(size_t)gpu], op.d_w[(size_t)gpu], op.d_x[(size_t)gpu],
                op.rows_per_gpu, op.cols, (uint32_t)op.row_bytes, op.slots);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    return 0;
}

int run_f8_dense_to_device(const Options &opt,
                           const std::vector<ContractRow> &rows,
                           const char *tensor,
                           int seed,
                           DeviceDenseOutputs *out) {
    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    if (!select_dense_rows(rows, tensor, &selected, &cols, &total_rows)) {
        std::fprintf(stderr, "device dense tensor validation failed for %s\n", tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    if (rows_per_gpu != kHidden / kGpus) {
        std::fprintf(stderr, "device dense tensor %s rows_per_gpu=%d expected=%d\n",
                     tensor, rows_per_gpu, kHidden / kGpus);
        return 2;
    }
    const uint64_t row_bytes = f8_row_bytes(cols);
    const uint64_t shard_bytes = row_bytes * (uint64_t)rows_per_gpu;
    out->d_out.assign((size_t)kGpus, nullptr);
    out->rows_per_gpu = rows_per_gpu;
    out->cols = cols;
    out->slots = opt.slots;

    std::vector<float> h_x((size_t)opt.slots * cols);
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int c = 0; c < cols; ++c) {
            const int m = (slot * (17 + seed) + c * (13 + seed * 3)) % 269;
            h_x[(size_t)slot * cols + c] = ((float)m - 134.0f) * 0.0002f;
        }
    }

    double worst_ms = 0.0;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = selected[(size_t)gpu];
        std::vector<uint8_t> h_w((size_t)shard_bytes);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), h_w.data(), h_w.size()) != 0) {
            free_device_dense_outputs(*out, opt);
            return 3;
        }
        out->loaded_bytes += shard_bytes;

        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        (void)cudaGetLastError();
        uint8_t *d_w = nullptr;
        float *d_x = nullptr;
        __half *d_w_half = nullptr;
        __half *d_x_half = nullptr;
        cublasHandle_t blas = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaMalloc(&d_w, (size_t)shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_out[(size_t)gpu],
                              (size_t)opt.slots * rows_per_gpu * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), (size_t)shard_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (opt.dense_f16_cublas_compose) {
            CHECK_CUDA(cudaMalloc(&d_w_half, (size_t)rows_per_gpu * cols * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&d_x_half, h_x.size() * sizeof(__half)));
            const uint64_t w_elems = (uint64_t)rows_per_gpu * cols;
            f8_b128_to_half_kernel<<<(unsigned int)((w_elems + 255) / 256), 256>>>(
                d_w_half, d_w, rows_per_gpu, cols, (uint32_t)row_bytes);
            CHECK_CUDA(cudaGetLastError());
            const uint64_t x_elems = (uint64_t)opt.slots * cols;
            cast_f32_to_half_kernel<<<(unsigned int)((x_elems + 255) / 256), 256>>>(
                d_x_half, d_x, x_elems);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            cublasStatus_t st = cublasCreate(&blas);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "cublasCreate failed on gpu %d: %d\n", gpu, (int)st);
                return 4;
            }
            (void)cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH);
        }
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        const dim3 scalar_grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1);
        const dim3 hmma_grid((unsigned int)((rows_per_gpu + 63) / 64),
                             (unsigned int)((opt.slots + 15) / 16),
                             1);
        for (int i = 0; i < opt.warmup; ++i) {
            if (opt.dense_f16_cublas_compose) {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                cublasStatus_t st = cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                                  rows_per_gpu, opt.slots, cols,
                                                  &alpha, d_w_half, CUDA_R_16F, cols,
                                                  d_x_half, CUDA_R_16F, cols,
                                                  &beta, out->d_out[(size_t)gpu],
                                                  CUDA_R_32F, rows_per_gpu,
                                                  CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS) return 5;
            } else if (opt.dense_hmma_compose) {
                f8_b128_dense_hmma_m16_kernel<<<hmma_grid, 128>>>(
                    out->d_out[(size_t)gpu], d_w, d_x, rows_per_gpu, cols,
                    (uint32_t)row_bytes, opt.slots);
            } else {
                f8_b128_dense_kernel<<<scalar_grid, 256>>>(
                    out->d_out[(size_t)gpu], d_w, d_x, rows_per_gpu, cols,
                    (uint32_t)row_bytes, opt.slots);
            }
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < opt.iters; ++i) {
            if (opt.dense_f16_cublas_compose) {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                cublasStatus_t st = cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                                  rows_per_gpu, opt.slots, cols,
                                                  &alpha, d_w_half, CUDA_R_16F, cols,
                                                  d_x_half, CUDA_R_16F, cols,
                                                  &beta, out->d_out[(size_t)gpu],
                                                  CUDA_R_32F, rows_per_gpu,
                                                  CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS) return 6;
            } else if (opt.dense_hmma_compose) {
                f8_b128_dense_hmma_m16_kernel<<<hmma_grid, 128>>>(
                    out->d_out[(size_t)gpu], d_w, d_x, rows_per_gpu, cols,
                    (uint32_t)row_bytes, opt.slots);
            } else {
                f8_b128_dense_kernel<<<scalar_grid, 256>>>(
                    out->d_out[(size_t)gpu], d_w, d_x, rows_per_gpu, cols,
                    (uint32_t)row_bytes, opt.slots);
            }
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        worst_ms = std::max(worst_ms, (double)ms / opt.iters);
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_x));
        if (d_w_half) CHECK_CUDA(cudaFree(d_w_half));
        if (d_x_half) CHECK_CUDA(cudaFree(d_x_half));
        if (blas) (void)cublasDestroy(blas);
    }
    out->compute_ms = worst_ms;
    return 0;
}

bool parse_tm_entry(const std::vector<std::string> &f, TmIndexEntry *out) {
    if (f.size() < 25) return false;
    TmIndexEntry e;
    e.semantic_tensor_id = f[0];
    e.runtime_layout = f[4];
    if (!parse_int(f[6].c_str(), &e.layer_id)) return false;
    if (!parse_int(f[8].c_str(), &e.n)) return false;
    if (!parse_int(f[9].c_str(), &e.k)) return false;
    if (!parse_int(f[10].c_str(), &e.experts_packed)) return false;
    if (!parse_int(f[11].c_str(), &e.experts_total)) return false;
    if (!parse_size(f[12].c_str(), &e.weight_bytes_per_expert)) return false;
    if (!parse_size(f[13].c_str(), &e.scale_bytes_per_expert)) return false;
    if (!parse_int(f[14].c_str(), &e.k_pack)) return false;
    if (!parse_int(f[15].c_str(), &e.weight_stride)) return false;
    if (!parse_int(f[16].c_str(), &e.scale_stride)) return false;
    e.sidecar_file = f[17];
    if (!parse_u64(f[18].c_str(), &e.weight_offset)) return false;
    if (!parse_u64(f[19].c_str(), &e.scale_offset)) return false;
    if (!safe_sidecar_name(e.sidecar_file)) return false;
    *out = e;
    return true;
}

bool valid_tm_entry(const TmIndexEntry &e, int n, int k, const char *layout) {
    return e.n == n &&
           e.k == k &&
           e.experts_total == kGlobalExperts &&
           e.experts_packed >= kGlobalExperts &&
           e.weight_bytes_per_expert > 0 &&
           e.scale_bytes_per_expert > 0 &&
           e.k_pack > 0 &&
           e.weight_stride > 0 &&
           e.scale_stride > 0 &&
           e.runtime_layout == layout;
}

int parse_tm_index(const char *path, int layer, DescriptorBindings *out) {
    FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "cannot open tm index %s: %s\n", path, std::strerror(errno));
        return 1;
    }
    char gated_name[128];
    char down_name[128];
    std::snprintf(gated_name, sizeof(gated_name), "blk.%d.ffn_gate_up_exps.weight", layer);
    std::snprintf(down_name, sizeof(down_name), "blk.%d.ffn_down_exps.weight", layer);
    char buf[8192];
    bool first = true;
    while (std::fgets(buf, sizeof(buf), fp)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (first) {
            first = false;
            continue;
        }
        if (line.empty()) continue;
        std::vector<std::string> f = split_tabs(line);
        TmIndexEntry e;
        if (!parse_tm_entry(f, &e)) {
            std::fclose(fp);
            return 2;
        }
        if (e.layer_id != layer) continue;
        if (e.semantic_tensor_id == gated_name) {
            if (!valid_tm_entry(e, kFusedN, kHidden,
                                "turbomind_mxfp4_grouped_gate_up_interleaved")) {
                std::fclose(fp);
                return 3;
            }
            out->gated = e;
            out->have_gated = true;
        } else if (e.semantic_tensor_id == down_name) {
            if (!valid_tm_entry(e, kHidden, kMid, "turbomind_mxfp4_grouped")) {
                std::fclose(fp);
                return 4;
            }
            out->down = e;
            out->have_down = true;
        }
    }
    std::fclose(fp);
    return out->have_gated && out->have_down ? 0 : 5;
}

void load_api(void *lib, Api *api) {
    api->init = (pfn_init)dlsym(lib, "ggml_turbomind_init");
    api->shutdown = (pfn_shutdown)dlsym(lib, "ggml_turbomind_shutdown");
    api->mmgt = (pfn_mmgt)dlsym(lib, "ggml_turbomind_mul_mat_grouped_total_tokens");
    api->mmgs = (pfn_mmgs)dlsym(lib, "ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens");
    if (!api->init || !api->shutdown || !api->mmgt || !api->mmgs) {
        std::fprintf(stderr, "dlsym failed for required TurboMind ABI\n");
        std::exit(2);
    }
}

int open_shared_api(const Options &opt, SharedApi *shared) {
    shared->lib = dlopen(opt.lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!shared->lib) {
        std::fprintf(stderr, "dlopen failed for %s: %s\n", opt.lib_path, dlerror());
        return 1;
    }
    load_api(shared->lib, &shared->api);
    for (int p = 0; p < kGpus; ++p) {
        if (shared->api.init(opt.devices[p]) != 0) {
            std::fprintf(stderr, "ggml_turbomind_init failed on device %d\n", opt.devices[p]);
            if (shared->api.shutdown) shared->api.shutdown();
            dlclose(shared->lib);
            *shared = SharedApi{};
            return 2;
        }
    }
    shared->initialized = true;
    return 0;
}

void close_shared_api(SharedApi *shared) {
    if (!shared || !shared->lib) return;
    if (shared->initialized && shared->api.shutdown) shared->api.shutdown();
    dlclose(shared->lib);
    *shared = SharedApi{};
}

void free_packed(PackedExperts &p) {
    for (void *v : p.d_w_active) {
        if (v) CHECK_CUDA(cudaFree(v));
    }
    for (void *v : p.d_s_active) {
        if (v) CHECK_CUDA(cudaFree(v));
    }
    if (p.d_w_table) CHECK_CUDA(cudaFree(p.d_w_table));
    if (p.d_s_table) CHECK_CUDA(cudaFree(p.d_s_table));
    p = PackedExperts{};
}

int pack_descriptor_set(int device, const TmIndexEntry &entry, int rank,
                        const std::vector<int> &active, const char *pack_dir,
                        PackedExperts *out, uint64_t *host_bytes_read) {
    CHECK_CUDA(cudaSetDevice(device));
    const std::string sidecar_path = path_join(pack_dir, entry.sidecar_file);
    out->d_w_active.assign(active.size(), nullptr);
    out->d_s_active.assign(active.size(), nullptr);
    out->k_pack = entry.k_pack;

    std::vector<uint8_t> h_weight(entry.weight_bytes_per_expert);
    std::vector<uint8_t> h_scale(entry.scale_bytes_per_expert);
    for (size_t i = 0; i < active.size(); ++i) {
        const int global_expert = rank * kLocalExperts + active[i];
        const uint64_t w_off = entry.weight_offset +
                               (uint64_t)global_expert * entry.weight_bytes_per_expert;
        const uint64_t s_off = entry.scale_offset +
                               (uint64_t)global_expert * entry.scale_bytes_per_expert;
        if (read_exact_at(sidecar_path, w_off, h_weight.data(), h_weight.size()) != 0 ||
            read_exact_at(sidecar_path, s_off, h_scale.data(), h_scale.size()) != 0) {
            return 1;
        }
        CHECK_CUDA(cudaMalloc(&out->d_w_active[i], h_weight.size()));
        CHECK_CUDA(cudaMalloc(&out->d_s_active[i], h_scale.size()));
        CHECK_CUDA(cudaMemcpy(out->d_w_active[i], h_weight.data(), h_weight.size(),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_s_active[i], h_scale.data(), h_scale.size(),
                              cudaMemcpyHostToDevice));
        *host_bytes_read += (uint64_t)h_weight.size() + (uint64_t)h_scale.size();
    }

    std::vector<StridedPtrH> w_table((size_t)kLocalExperts);
    std::vector<StridedPtrH> s_table((size_t)kLocalExperts);
    for (int e = 0; e < kLocalExperts; ++e) {
        w_table[(size_t)e] = StridedPtrH{out->d_w_active[0], entry.weight_stride};
        s_table[(size_t)e] = StridedPtrH{out->d_s_active[0], entry.scale_stride};
    }
    for (size_t i = 0; i < active.size(); ++i) {
        w_table[(size_t)active[i]] = StridedPtrH{out->d_w_active[i], entry.weight_stride};
        s_table[(size_t)active[i]] = StridedPtrH{out->d_s_active[i], entry.scale_stride};
    }
    CHECK_CUDA(cudaMalloc(&out->d_w_table, w_table.size() * sizeof(StridedPtrH)));
    CHECK_CUDA(cudaMemcpy(out->d_w_table, w_table.data(),
                          w_table.size() * sizeof(StridedPtrH), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&out->d_s_table, s_table.size() * sizeof(StridedPtrH)));
    CHECK_CUDA(cudaMemcpy(out->d_s_table, s_table.data(),
                          s_table.size() * sizeof(StridedPtrH), cudaMemcpyHostToDevice));
    return 0;
}

int run_gate(RankState &rank, const Api &api) {
    return api.mmgs(rank.d_a, nullptr, rank.d_offsets, kLocalExperts, rank.routes,
                    (const void * const *)rank.gated.d_w_table,
                    (const void * const *)rank.gated.d_s_table,
                    kDType, kFusedN, kHidden, kGroupSize, rank.gated.k_pack,
                    rank.d_gated, rank.stream);
}

int run_down(RankState &rank, const Api &api) {
    return api.mmgt(rank.d_gated, nullptr, rank.d_offsets, kLocalExperts, rank.routes,
                    (const void * const *)rank.down.d_w_table,
                    (const void * const *)rank.down.d_s_table,
                    kDType, kHidden, kMid, kGroupSize, rank.down.k_pack,
                    rank.d_down, rank.stream);
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

int check_repeat(RankState &rank, const Api &api, double *max_abs, int *bad, int *nan) {
    const size_t elems = (size_t)rank.routes * kHidden;
    std::vector<__half> first(elems);
    std::vector<__half> second(elems);
    CHECK_CUDA(cudaSetDevice(rank.device));
    if (run_gate(rank, api) != 0 || run_down(rank, api) != 0) return 1;
    CHECK_CUDA(cudaStreamSynchronize(rank.stream));
    CHECK_CUDA(cudaMemcpy(first.data(), rank.d_down, elems * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    if (run_gate(rank, api) != 0 || run_down(rank, api) != 0) return 1;
    CHECK_CUDA(cudaStreamSynchronize(rank.stream));
    CHECK_CUDA(cudaMemcpy(second.data(), rank.d_down, elems * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elems; ++i) {
        const float a = __half2float(first[i]);
        const float b = __half2float(second[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++*nan;
            continue;
        }
        const double diff = std::fabs((double)a - (double)b);
        *max_abs = std::max(*max_abs, diff);
        if (diff > 0.0) ++*bad;
    }
    return 0;
}

void build_offsets_for_rank(int rank, int slots, int top_k,
                            std::vector<int> *offsets,
                            std::vector<int> *route_slots,
                            int *routes,
                            int *active_experts,
                            int *max_routes_per_expert) {
    std::vector<int> counts((size_t)kLocalExperts, 0);
    for (int slot = 0; slot < slots; ++slot) {
        for (int k = 0; k < top_k; ++k) {
            const int dst_rank = (slot * top_k + k) % kGpus;
            if (dst_rank != rank) continue;
            const int local = (slot + k * 7 + rank) % kActiveLocalExperts;
            counts[(size_t)local]++;
        }
    }
    offsets->assign((size_t)kLocalExperts + 1, 0);
    int running = 0;
    int active = 0;
    int max_routes = 0;
    for (int e = 0; e < kLocalExperts; ++e) {
        (*offsets)[(size_t)e] = running;
        running += counts[(size_t)e];
        if (counts[(size_t)e] > 0) ++active;
        max_routes = std::max(max_routes, counts[(size_t)e]);
    }
    (*offsets)[(size_t)kLocalExperts] = running;
    if (route_slots) {
        route_slots->assign((size_t)running, -1);
        std::vector<int> cursor = *offsets;
        for (int slot = 0; slot < slots; ++slot) {
            for (int k = 0; k < top_k; ++k) {
                const int dst_rank = (slot * top_k + k) % kGpus;
                if (dst_rank != rank) continue;
                const int local = (slot + k * 7 + rank) % kActiveLocalExperts;
                const int idx = cursor[(size_t)local]++;
                (*route_slots)[(size_t)idx] = slot;
            }
        }
    }
    *routes = running;
    *active_experts = active;
    *max_routes_per_expert = max_routes;
}

int open_shared_rank_buffers(const Options &opt, SharedRankBuffers *shared) {
    shared->core_bytes = 0;
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = shared->ranks[p];
        r.rank = p;
        r.device = opt.devices[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaStreamCreate(&r.stream));
        CHECK_CUDA(cudaEventCreate(&r.start));
        CHECK_CUDA(cudaEventCreate(&r.mid));
        CHECK_CUDA(cudaEventCreate(&r.stop));

        std::vector<int> offsets;
        std::vector<int> route_slots;
        build_offsets_for_rank(p, opt.slots, opt.top_k, &offsets, &route_slots, &r.routes,
                               &r.active_experts, &r.max_routes_per_expert);

        const size_t a_elems = (size_t)r.routes * kHidden;
        CHECK_CUDA(cudaMalloc(&r.d_offsets, offsets.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(r.d_offsets, offsets.data(), offsets.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_route_slots, route_slots.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(r.d_route_slots, route_slots.data(),
                              route_slots.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_a, a_elems * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&r.d_gated, (size_t)r.routes * kMid * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&r.d_down, a_elems * sizeof(__half)));

        std::mt19937 rng(0xE2350000u + (uint32_t)p * 97u);
        std::uniform_real_distribution<float> dist(-0.003f, 0.003f);
        std::vector<__half> h_a(a_elems);
        for (__half &v : h_a) v = __float2half(dist(rng));
        CHECK_CUDA(cudaMemcpy(r.d_a, h_a.data(), a_elems * sizeof(__half),
                              cudaMemcpyHostToDevice));

        shared->core_bytes += offsets.size() * sizeof(int);
        shared->core_bytes += route_slots.size() * sizeof(int);
        shared->core_bytes += a_elems * sizeof(__half);
        shared->core_bytes += (size_t)r.routes * kMid * sizeof(__half);
        shared->core_bytes += a_elems * sizeof(__half);
    }
    shared->initialized = true;
    return 0;
}

void close_shared_rank_buffers(SharedRankBuffers *shared) {
    if (!shared || !shared->initialized) return;
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = shared->ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        free_packed(r.gated);
        free_packed(r.down);
        if (r.d_offsets) CHECK_CUDA(cudaFree(r.d_offsets));
        if (r.d_route_slots) CHECK_CUDA(cudaFree(r.d_route_slots));
        if (r.d_a) CHECK_CUDA(cudaFree(r.d_a));
        if (r.d_gated) CHECK_CUDA(cudaFree(r.d_gated));
        if (r.d_down) CHECK_CUDA(cudaFree(r.d_down));
        if (r.d_ep_contrib_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_all));
        if (r.d_ep_contrib_half_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_half_all));
        for (int src = 0; src < kGpus; ++src) {
            if (r.d_ep_remote[src]) CHECK_CUDA(cudaFree(r.d_ep_remote[src]));
            if (r.d_ep_remote_half[src]) CHECK_CUDA(cudaFree(r.d_ep_remote_half[src]));
        }
        if (r.d_ep_sum) CHECK_CUDA(cudaFree(r.d_ep_sum));
        if (r.d_next_hidden) CHECK_CUDA(cudaFree(r.d_next_hidden));
        if (r.start) CHECK_CUDA(cudaEventDestroy(r.start));
        if (r.mid) CHECK_CUDA(cudaEventDestroy(r.mid));
        if (r.stop) CHECK_CUDA(cudaEventDestroy(r.stop));
        if (r.stream) CHECK_CUDA(cudaStreamDestroy(r.stream));
        r = RankState{};
    }
    *shared = SharedRankBuffers{};
}

void fill_tp_runtime_config(const Options &opt, ds4_v100_tp_runtime_config *cfg) {
    ds4_v100_tp_runtime_default_config(cfg);
    cfg->slots = (uint32_t)opt.slots;
    cfg->ctx = 262144;
    cfg->kv_dtype = DS4_V100_TP_KV_F8_E4M3_B128;
    cfg->scratch_bytes = 1536ull * 1024ull * 1024ull;
    for (int i = 0; i < kGpus; ++i) cfg->devices[i] = opt.devices[i];
}

int open_shared_tp_runtime(const Options &opt, SharedTpRuntime *shared) {
    ds4_v100_tp_runtime_config cfg;
    fill_tp_runtime_config(opt, &cfg);
    char err[512] = {0};
    if (ds4_v100_tp_runtime_open(&shared->rt, &cfg, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "tp_runtime_open_failed\t%s\n", err);
        *shared = SharedTpRuntime{};
        return 1;
    }
    ds4_v100_tp_runtime_get_report(shared->rt, &shared->report);
    shared->initialized = true;
    return 0;
}

void close_shared_tp_runtime(SharedTpRuntime *shared) {
    if (!shared || !shared->rt) return;
    ds4_v100_tp_runtime_close(shared->rt);
    *shared = SharedTpRuntime{};
}

int ensure_compose_buffers(const Options &opt, RankState ranks[kGpus]) {
    const uint64_t shard_elems = (uint64_t)opt.slots * (kHidden / kGpus);
    const uint64_t shard_bytes = shard_elems * sizeof(float);
    const uint64_t all_contrib_elems = (uint64_t)kGpus * shard_elems;
    const uint64_t all_contrib_bytes = all_contrib_elems * sizeof(float);
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_ep_contrib_all) CHECK_CUDA(cudaMalloc(&r.d_ep_contrib_all,
                                                        (size_t)all_contrib_bytes));
        if (opt.ep_return_fp16 && !r.d_ep_contrib_half_all) {
            CHECK_CUDA(cudaMalloc(&r.d_ep_contrib_half_all,
                                  (size_t)(all_contrib_elems * sizeof(__half))));
        }
        if (!r.d_ep_sum) CHECK_CUDA(cudaMalloc(&r.d_ep_sum, (size_t)shard_bytes));
        if (!r.d_next_hidden) CHECK_CUDA(cudaMalloc(&r.d_next_hidden, (size_t)shard_bytes));
        for (int src = 0; src < kGpus; ++src) {
            if (!r.d_ep_remote[src]) CHECK_CUDA(cudaMalloc(&r.d_ep_remote[src],
                                                           (size_t)shard_bytes));
            if (opt.ep_return_fp16 && !r.d_ep_remote_half[src]) {
                CHECK_CUDA(cudaMalloc(&r.d_ep_remote_half[src],
                                      (size_t)(shard_elems * sizeof(__half))));
            }
        }
    }
    return 0;
}

int run_next_hidden_compose(const Options &opt,
                            const std::vector<ContractRow> &rows,
                            RankState ranks[kGpus],
                            ComposeStats *stats) {
    if (!opt.compose_next_hidden) return 0;
    stats->enabled = true;
    stats->ep_return_fp16 = opt.ep_return_fp16;
    stats->fused_compose_sum = opt.fuse_compose_sum && !opt.ep_return_fp16;
    stats->dense_hmma_compose = opt.dense_hmma_compose;
    stats->dense_f16_cublas_compose = opt.dense_f16_cublas_compose;

    DeviceDenseOutputs attn;
    DeviceDenseOutputs shared;
    const std::string attn_tensor = layer_tensor_name(opt.layer, "attn_output_b.weight");
    const std::string shared_tensor = layer_tensor_name(opt.layer, "ffn_down_shexp.weight");
    if (run_f8_dense_to_device(opt, rows, attn_tensor.c_str(), 1, &attn) != 0 ||
        run_f8_dense_to_device(opt, rows, shared_tensor.c_str(), 2, &shared) != 0) {
        free_device_dense_outputs(attn, opt);
        free_device_dense_outputs(shared, opt);
        return 1;
    }
    stats->attn_dense_ms = attn.compute_ms;
    stats->shared_dense_ms = shared.compute_ms;

    const uint64_t shard_elems = (uint64_t)opt.slots * (kHidden / kGpus);
    const uint64_t shard_bytes = shard_elems * sizeof(float);
    const uint64_t return_shard_bytes =
        shard_elems * (opt.ep_return_fp16 ? sizeof(__half) : sizeof(float));
    const uint64_t all_contrib_elems = (uint64_t)kGpus * shard_elems;
    const uint64_t all_contrib_bytes = all_contrib_elems * sizeof(float);
    stats->ep_contribution_bytes = all_contrib_bytes * kGpus;
    stats->ep_return_bytes = return_shard_bytes * kGpus * kGpus;
    if (ensure_compose_buffers(opt, ranks) != 0) {
        free_device_dense_outputs(attn, opt);
        free_device_dense_outputs(shared, opt);
        return 2;
    }

    const auto compose_start = std::chrono::steady_clock::now();

    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        const int block = 256;
        int grid = (int)((all_contrib_elems + block - 1) / block);
        zero_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_contrib_all,
                                                      all_contrib_elems);
        CHECK_CUDA(cudaGetLastError());
        const uint64_t route_hidden_elems = (uint64_t)r.routes * kHidden;
        grid = (int)((route_hidden_elems + block - 1) / block);
        ep_reduce_all_dest_shards_kernel<<<grid, block, 0, r.stream>>>(
            r.d_ep_contrib_all, r.d_down, r.d_route_slots, r.routes, opt.slots);
        CHECK_CUDA(cudaGetLastError());
        if (opt.ep_return_fp16) {
            grid = (int)((all_contrib_elems + block - 1) / block);
            cast_f32_to_half_kernel<<<grid, block, 0, r.stream>>>(
                r.d_ep_contrib_half_all, r.d_ep_contrib_all, all_contrib_elems);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    for (int p = 0; p < kGpus; ++p) {
        CHECK_CUDA(cudaSetDevice(ranks[p].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[p].stream));
    }

    for (int dst = 0; dst < kGpus; ++dst) {
        CHECK_CUDA(cudaSetDevice(ranks[dst].device));
        for (int src = 0; src < kGpus; ++src) {
            if (opt.ep_return_fp16) {
                const __half *src_ptr =
                    ranks[src].d_ep_contrib_half_all + (uint64_t)dst * shard_elems;
                CHECK_CUDA(cudaMemcpyPeerAsync(ranks[dst].d_ep_remote_half[src],
                                               ranks[dst].device,
                                               src_ptr,
                                               ranks[src].device,
                                               (size_t)return_shard_bytes,
                                               ranks[dst].stream));
            } else {
                const float *src_ptr = ranks[src].d_ep_contrib_all +
                                       (uint64_t)dst * shard_elems;
                CHECK_CUDA(cudaMemcpyPeerAsync(ranks[dst].d_ep_remote[src],
                                               ranks[dst].device,
                                               src_ptr,
                                               ranks[src].device,
                                               (size_t)return_shard_bytes,
                                               ranks[dst].stream));
            }
        }
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        CHECK_CUDA(cudaSetDevice(ranks[dst].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[dst].stream));
    }

    std::vector<std::vector<float>> first((size_t)kGpus);
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            const int block = 256;
            int grid = (int)((shard_elems + block - 1) / block);
            if (stats->fused_compose_sum) {
                compose_next_hidden_sum8_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, attn.d_out[(size_t)dst], shared.d_out[(size_t)dst],
                    r.d_ep_remote[0], r.d_ep_remote[1], r.d_ep_remote[2],
                    r.d_ep_remote[3], r.d_ep_remote[4], r.d_ep_remote[5],
                    r.d_ep_remote[6], r.d_ep_remote[7], dst, opt.slots);
            } else {
                zero_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_sum, shard_elems);
                CHECK_CUDA(cudaGetLastError());
                for (int src = 0; src < kGpus; ++src) {
                    if (opt.ep_return_fp16) {
                        add_half_to_f32_kernel<<<grid, block, 0, r.stream>>>(
                            r.d_ep_sum, r.d_ep_remote_half[src], shard_elems);
                    } else {
                        add_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_sum,
                                                                     r.d_ep_remote[src],
                                                                     shard_elems);
                    }
                }
                CHECK_CUDA(cudaGetLastError());
                compose_next_hidden_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, attn.d_out[(size_t)dst], shared.d_out[(size_t)dst],
                    r.d_ep_sum, dst, opt.slots);
            }
            CHECK_CUDA(cudaGetLastError());
        }
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_CUDA(cudaStreamSynchronize(r.stream));
            std::vector<float> host((size_t)shard_elems);
            CHECK_CUDA(cudaMemcpy(host.data(), r.d_next_hidden, (size_t)shard_bytes,
                                  cudaMemcpyDeviceToHost));
            if (repeat == 0) {
                first[(size_t)dst] = host;
                for (uint64_t i = 0; i < shard_elems; ++i) {
                    if (!std::isfinite(host[(size_t)i])) {
                        stats->finite_bad++;
                        stats->pass = false;
                    }
                    uint32_t bits = 0;
                    std::memcpy(&bits, &host[(size_t)i], sizeof(bits));
                    stats->checksum ^=
                        (uint64_t)bits + (uint64_t)(dst + 1) * 1000003ull + i * 9176ull;
                }
            } else {
                for (uint64_t i = 0; i < shard_elems; ++i) {
                    const double diff =
                        std::fabs((double)host[(size_t)i] -
                                  (double)first[(size_t)dst][(size_t)i]);
                    stats->repeat_max_abs = std::max(stats->repeat_max_abs, diff);
                    if (diff > 0.0) {
                        stats->repeat_bad++;
                        stats->pass = false;
                    }
                }
            }
        }
    }

    const auto compose_stop = std::chrono::steady_clock::now();
    stats->compose_ms =
        std::chrono::duration<double, std::milli>(compose_stop - compose_start).count();
    if (stats->checksum == 0 || stats->finite_bad != 0 || stats->repeat_bad != 0) {
        stats->pass = false;
    }

    free_device_dense_outputs(attn, opt);
    free_device_dense_outputs(shared, opt);
    return stats->pass ? 0 : 2;
}

int run_decode_loop(const Options &opt,
                    const std::vector<ContractRow> &rows,
                    RankState ranks[kGpus],
                    const Api &api,
                    const DenseF16Cache *cache,
                    DecodeLoopStats *stats) {
    if (opt.decode_steps <= 0) return 0;
    stats->enabled = true;
    stats->ep_return_fp16 = opt.ep_return_fp16;
    stats->fused_compose_sum = opt.fuse_compose_sum && !opt.ep_return_fp16;
    stats->dense_hmma_compose = opt.dense_hmma_compose;
    stats->dense_f16_cublas_compose = opt.dense_f16_cublas_compose;
    stats->dense_f16_cache_compose = opt.dense_f16_cache_compose;
    stats->steps = opt.decode_steps;
    stats->slots = opt.slots;
    stats->slot_steps = (uint64_t)opt.decode_steps * (uint64_t)opt.slots;

    ResidentF8Dense attn;
    ResidentF8Dense shared;
    const std::string attn_tensor = layer_tensor_name(opt.layer, "attn_output_b.weight");
    const std::string shared_tensor = layer_tensor_name(opt.layer, "ffn_down_shexp.weight");
    if (prepare_resident_f8_dense(opt, rows, attn_tensor.c_str(), 1, cache, &attn) != 0 ||
        prepare_resident_f8_dense(opt, rows, shared_tensor.c_str(), 2, cache, &shared) != 0) {
        free_resident_f8_dense(attn, opt);
        free_resident_f8_dense(shared, opt);
        return 1;
    }
    stats->dense_loaded_bytes = attn.loaded_bytes + shared.loaded_bytes;

    const uint64_t shard_elems = (uint64_t)opt.slots * (kHidden / kGpus);
    const uint64_t shard_bytes = shard_elems * sizeof(float);
    const uint64_t return_shard_bytes =
        shard_elems * (opt.ep_return_fp16 ? sizeof(__half) : sizeof(float));
    const uint64_t all_contrib_elems = (uint64_t)kGpus * shard_elems;
    const uint64_t all_contrib_bytes = all_contrib_elems * sizeof(float);
    stats->ep_contribution_bytes = all_contrib_bytes * kGpus;
    stats->ep_return_bytes = return_shard_bytes * kGpus * kGpus;
    if (ensure_compose_buffers(opt, ranks) != 0) {
        free_resident_f8_dense(attn, opt);
        free_resident_f8_dense(shared, opt);
        return 2;
    }

    auto sync_all = [&]() {
        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[p].stream));
        }
    };

    auto run_one_step = [&](double *ep_ms, double *dense_ms, double *compose_ms) -> int {
        auto t0 = std::chrono::steady_clock::now();
        for (int p = 0; p < kGpus; ++p) {
            if (run_gate(ranks[p], api) != 0 || run_down(ranks[p], api) != 0) return 1;
        }
        sync_all();
        auto t1 = std::chrono::steady_clock::now();
        if (launch_resident_f8_dense(opt, attn, ranks) != 0 ||
            launch_resident_f8_dense(opt, shared, ranks) != 0) {
            return 2;
        }
        sync_all();
        auto t2 = std::chrono::steady_clock::now();

        const int block = 256;
        for (int p = 0; p < kGpus; ++p) {
            RankState &r = ranks[p];
            CHECK_CUDA(cudaSetDevice(r.device));
            int grid = (int)((all_contrib_elems + block - 1) / block);
            zero_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_contrib_all,
                                                          all_contrib_elems);
            CHECK_CUDA(cudaGetLastError());
            const uint64_t route_hidden_elems = (uint64_t)r.routes * kHidden;
            grid = (int)((route_hidden_elems + block - 1) / block);
            ep_reduce_all_dest_shards_kernel<<<grid, block, 0, r.stream>>>(
                r.d_ep_contrib_all, r.d_down, r.d_route_slots, r.routes, opt.slots);
            CHECK_CUDA(cudaGetLastError());
            if (opt.ep_return_fp16) {
                grid = (int)((all_contrib_elems + block - 1) / block);
                cast_f32_to_half_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_ep_contrib_half_all, r.d_ep_contrib_all, all_contrib_elems);
                CHECK_CUDA(cudaGetLastError());
            }
        }
        sync_all();

        for (int dst = 0; dst < kGpus; ++dst) {
            CHECK_CUDA(cudaSetDevice(ranks[dst].device));
            for (int src = 0; src < kGpus; ++src) {
                if (opt.ep_return_fp16) {
                    const __half *src_ptr =
                        ranks[src].d_ep_contrib_half_all + (uint64_t)dst * shard_elems;
                    CHECK_CUDA(cudaMemcpyPeerAsync(ranks[dst].d_ep_remote_half[src],
                                                   ranks[dst].device,
                                                   src_ptr,
                                                   ranks[src].device,
                                                   (size_t)return_shard_bytes,
                                                   ranks[dst].stream));
                } else {
                    const float *src_ptr = ranks[src].d_ep_contrib_all +
                                           (uint64_t)dst * shard_elems;
                    CHECK_CUDA(cudaMemcpyPeerAsync(ranks[dst].d_ep_remote[src],
                                                   ranks[dst].device,
                                                   src_ptr,
                                                   ranks[src].device,
                                                   (size_t)return_shard_bytes,
                                                   ranks[dst].stream));
                }
            }
        }
        sync_all();

        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            int grid = (int)((shard_elems + block - 1) / block);
            if (stats->fused_compose_sum) {
                compose_next_hidden_sum8_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, attn.d_out[(size_t)dst], shared.d_out[(size_t)dst],
                    r.d_ep_remote[0], r.d_ep_remote[1], r.d_ep_remote[2],
                    r.d_ep_remote[3], r.d_ep_remote[4], r.d_ep_remote[5],
                    r.d_ep_remote[6], r.d_ep_remote[7], dst, opt.slots);
            } else {
                zero_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_sum, shard_elems);
                CHECK_CUDA(cudaGetLastError());
                for (int src = 0; src < kGpus; ++src) {
                    if (opt.ep_return_fp16) {
                        add_half_to_f32_kernel<<<grid, block, 0, r.stream>>>(
                            r.d_ep_sum, r.d_ep_remote_half[src], shard_elems);
                    } else {
                        add_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_sum,
                                                                     r.d_ep_remote[src],
                                                                     shard_elems);
                    }
                }
                CHECK_CUDA(cudaGetLastError());
                compose_next_hidden_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, attn.d_out[(size_t)dst], shared.d_out[(size_t)dst],
                    r.d_ep_sum, dst, opt.slots);
            }
            CHECK_CUDA(cudaGetLastError());
        }
        sync_all();
        auto t3 = std::chrono::steady_clock::now();
        *ep_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        *dense_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        *compose_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
        return 0;
    };

    double warm_ep = 0.0;
    double warm_dense = 0.0;
    double warm_compose = 0.0;
    for (int i = 0; i < opt.warmup; ++i) {
        if (run_one_step(&warm_ep, &warm_dense, &warm_compose) != 0) {
            free_resident_f8_dense(attn, opt);
            free_resident_f8_dense(shared, opt);
            return 3;
        }
    }

    double ep_ms = 0.0;
    double dense_ms = 0.0;
    double compose_ms = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < opt.decode_steps; ++i) {
        if (run_one_step(&ep_ms, &dense_ms, &compose_ms) != 0) {
            free_resident_f8_dense(attn, opt);
            free_resident_f8_dense(shared, opt);
            return 4;
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    stats->total_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    stats->ms_per_step = stats->total_ms / (double)opt.decode_steps;
    stats->tok_s = stats->total_ms > 0.0
        ? (double)stats->slot_steps * 1000.0 / stats->total_ms
        : 0.0;
    stats->ep_ms_per_step = ep_ms / (double)opt.decode_steps;
    stats->dense_ms_per_step = dense_ms / (double)opt.decode_steps;
    stats->compose_ms_per_step = compose_ms / (double)opt.decode_steps;

    for (int dst = 0; dst < kGpus; ++dst) {
        RankState &r = ranks[dst];
        CHECK_CUDA(cudaSetDevice(r.device));
        std::vector<float> host((size_t)shard_elems);
        CHECK_CUDA(cudaMemcpy(host.data(), r.d_next_hidden, (size_t)shard_bytes,
                              cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < shard_elems; ++i) {
            const float v = host[(size_t)i];
            if (!std::isfinite(v)) {
                stats->finite_bad++;
                stats->pass = false;
            }
            uint32_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            stats->checksum ^=
                (uint64_t)bits + (uint64_t)(dst + 1) * 2000003ull + i * 7907ull;
        }
    }
    if (stats->checksum == 0 || stats->finite_bad != 0) stats->pass = false;

    free_resident_f8_dense(attn, opt);
    free_resident_f8_dense(shared, opt);
    return stats->pass ? 0 : 5;
}

} // namespace

int run_layer(const Options &opt,
              LayerRunSummary *summary,
              const DenseF16Cache *shared_dense_f16_cache,
              const SharedApi *shared_api,
              SharedRankBuffers *shared_rank_buffers,
              SharedTpRuntime *shared_tp_runtime) {
    std::vector<ContractRow> rows;
    LayerStats layer_stats;
    if (parse_contract(opt.contract_path, opt.layer, &rows, &layer_stats) != 0 ||
        layer_stats.bad_rows != 0) {
        std::fprintf(stderr, "contract parse failed bad_rows=%llu\n",
                     (unsigned long long)layer_stats.bad_rows);
        return 2;
    }
    DescriptorBindings bindings;
    if (parse_tm_index(opt.tm_index_path, opt.layer, &bindings) != 0) {
        std::fprintf(stderr, "tm index parse failed for layer %d\n", opt.layer);
        return 2;
    }

    const auto descriptor_start = std::chrono::steady_clock::now();
    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" && r.record_type != "replicated_control") continue;
        if (!opt.skip_descriptor_checks) {
            uint64_t checksum = 0;
            if (device_checksum_row(opt.devices[r.owning_gpu], opt.pack_dir, r, &checksum) != 0) {
                return 3;
            }
            layer_stats.gpu[r.owning_gpu].checksum ^=
                checksum + (uint64_t)(r.owning_gpu + 1) * 131u;
            layer_stats.checksum ^= checksum + (uint64_t)(r.owning_gpu + 1) * 257u;
        }
        if (r.record_type == "dense_tp") layer_stats.dense_loaded_bytes += r.bytes_estimate;
        else layer_stats.control_loaded_bytes += r.bytes_estimate;
    }
    const auto descriptor_stop = std::chrono::steady_clock::now();
    const double descriptor_ms =
        std::chrono::duration<double, std::milli>(descriptor_stop - descriptor_start).count();

    DenseComputeStats dense_compute;
    DenseComputeStats bf16_compute;
    std::vector<DenseComputeStats> dense_compute_results;
    std::vector<DenseComputeStats> bf16_compute_results;
    std::vector<std::string> dense_tensors;
    if (opt.dense_compute_all_f8) {
        dense_tensors = discover_f8_dense_tensors(rows);
    } else if (opt.dense_compute_tensor) {
        dense_tensors.emplace_back(opt.dense_compute_tensor);
    }
    for (const std::string &tensor : dense_tensors) {
        DenseComputeStats one;
        if (run_dense_compute_gate(opt, rows, tensor.c_str(), &one) != 0) {
            std::fprintf(stderr, "dense compute gate failed for %s\n", tensor.c_str());
            return 3;
        }
        std::printf("dense_compute_tensor\ttensor\t%s\trows_per_gpu\t%d\tcols\t%d\t"
                    "slots\t%d\tloaded_bytes\t%llu\tcompute_ms\t%.6f\t"
                    "repeat_max_abs\t%.9f\trepeat_bad\t%d\trepeat_nan\t%d\t"
                    "oracle_max_abs\t%.9f\toracle_bad\t%d\t%s\n",
                    one.tensor_id.c_str(), one.rows_per_gpu, one.cols, one.slots,
                    (unsigned long long)one.loaded_bytes, one.compute_ms,
                    one.repeat_max_abs, one.repeat_bad, one.repeat_nan,
                    one.oracle_max_abs, one.oracle_bad, one.pass ? "PASS" : "FAIL");
        dense_compute_results.push_back(one);
        dense_compute.enabled = true;
        dense_compute.tensor_id = opt.dense_compute_all_f8 ? "all_f8" : one.tensor_id;
        dense_compute.rows_per_gpu = std::max(dense_compute.rows_per_gpu, one.rows_per_gpu);
        dense_compute.cols = std::max(dense_compute.cols, one.cols);
        dense_compute.slots = one.slots;
        dense_compute.loaded_bytes += one.loaded_bytes;
        dense_compute.compute_ms = std::max(dense_compute.compute_ms, one.compute_ms);
        dense_compute.repeat_max_abs =
            std::max(dense_compute.repeat_max_abs, one.repeat_max_abs);
        dense_compute.oracle_max_abs =
            std::max(dense_compute.oracle_max_abs, one.oracle_max_abs);
        dense_compute.repeat_bad += one.repeat_bad;
        dense_compute.repeat_nan += one.repeat_nan;
        dense_compute.oracle_bad += one.oracle_bad;
        dense_compute.pass = dense_compute.pass && one.pass;
    }
    std::vector<std::string> bf16_tensors;
    if (opt.dense_compute_all_bf16) {
        bf16_tensors = discover_bf16_dense_tensors(rows);
    }
    for (const std::string &tensor : bf16_tensors) {
        DenseComputeStats one;
        if (run_bf16_dense_compute_gate(opt, rows, tensor.c_str(), &one) != 0) {
            std::fprintf(stderr, "bf16 dense compute gate failed for %s\n", tensor.c_str());
            return 3;
        }
        std::printf("bf16_dense_compute_tensor\ttensor\t%s\trows_per_gpu\t%d\tcols\t%d\t"
                    "slots\t%d\tloaded_bytes\t%llu\tcompute_ms\t%.6f\t"
                    "repeat_max_abs\t%.9f\trepeat_bad\t%d\trepeat_nan\t%d\t"
                    "oracle_max_abs\t%.9f\toracle_bad\t%d\t%s\n",
                    one.tensor_id.c_str(), one.rows_per_gpu, one.cols, one.slots,
                    (unsigned long long)one.loaded_bytes, one.compute_ms,
                    one.repeat_max_abs, one.repeat_bad, one.repeat_nan,
                    one.oracle_max_abs, one.oracle_bad, one.pass ? "PASS" : "FAIL");
        bf16_compute_results.push_back(one);
        bf16_compute.enabled = true;
        bf16_compute.tensor_id = "all_bf16";
        bf16_compute.rows_per_gpu = std::max(bf16_compute.rows_per_gpu, one.rows_per_gpu);
        bf16_compute.cols = std::max(bf16_compute.cols, one.cols);
        bf16_compute.slots = one.slots;
        bf16_compute.loaded_bytes += one.loaded_bytes;
        bf16_compute.compute_ms = std::max(bf16_compute.compute_ms, one.compute_ms);
        bf16_compute.repeat_max_abs =
            std::max(bf16_compute.repeat_max_abs, one.repeat_max_abs);
        bf16_compute.oracle_max_abs =
            std::max(bf16_compute.oracle_max_abs, one.oracle_max_abs);
        bf16_compute.repeat_bad += one.repeat_bad;
        bf16_compute.repeat_nan += one.repeat_nan;
        bf16_compute.oracle_bad += one.oracle_bad;
        bf16_compute.pass = bf16_compute.pass && one.pass;
    }

    DenseF16Cache local_dense_f16_cache;
    const DenseF16Cache *dense_f16_cache = shared_dense_f16_cache;
    if (!dense_f16_cache) {
        if (prepare_dense_f16_cache(opt, rows, &local_dense_f16_cache) != 0) {
            std::fprintf(stderr, "dense f16 cache prepare failed\n");
            return 4;
        }
        dense_f16_cache = &local_dense_f16_cache;
    }
    if (!shared_dense_f16_cache && dense_f16_cache->enabled) {
        std::printf("tp_ep_dense_f16_cache\tlayer\t%d\trows\t%llu\t"
                    "source_bytes\t%llu\tcache_bytes\t%llu\t"
                    "cache_aligned_bytes\t%llu\tmax_temp_bytes\t%llu\tPASS\n",
                    opt.layer,
                    (unsigned long long)dense_f16_cache->rows,
                    (unsigned long long)dense_f16_cache->source_bytes,
                    (unsigned long long)dense_f16_cache->cache_bytes,
                    (unsigned long long)dense_f16_cache->cache_aligned_bytes,
                    (unsigned long long)dense_f16_cache->max_temp_bytes);
    }

    ds4_v100_tp_runtime_config cfg;
    fill_tp_runtime_config(opt, &cfg);

    char err[512] = {0};
    ds4_v100_tp_runtime *rt = nullptr;
    ds4_v100_tp_runtime_report runtime_report;
    if (shared_tp_runtime) {
        rt = shared_tp_runtime->rt;
        runtime_report = shared_tp_runtime->report;
    } else {
        if (ds4_v100_tp_runtime_open(&rt, &cfg, err, sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_open_failed\t%s\n", err);
            return 4;
        }
        ds4_v100_tp_runtime_get_report(rt, &runtime_report);
    }
    auto close_local_runtime = [&]() {
        if (!shared_tp_runtime && rt) ds4_v100_tp_runtime_close(rt);
    };

    ds4_v100_tp_dense_kv_result kv_result;
    const auto kv_start = std::chrono::steady_clock::now();
    const int write_indexer = ds4_layer_ratio(opt.layer) == 4 ? 1 : 0;
    if (ds4_v100_tp_runtime_dense_kv_slice(rt, opt.layer, opt.kv_slot, opt.position,
                                           write_indexer, &kv_result, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "tp_runtime_dense_kv_slice_failed\t%s\n", err);
        close_local_runtime();
        return 5;
    }
    const auto kv_stop = std::chrono::steady_clock::now();
    const double dense_kv_ms =
        std::chrono::duration<double, std::milli>(kv_stop - kv_start).count();

    void *lib = nullptr;
    Api local_api;
    const Api *api = nullptr;
    if (shared_api) {
        api = &shared_api->api;
    } else {
        lib = dlopen(opt.lib_path, RTLD_LAZY | RTLD_LOCAL);
        if (!lib) {
            std::fprintf(stderr, "dlopen failed for %s: %s\n", opt.lib_path, dlerror());
            close_local_runtime();
            return 6;
        }
        load_api(lib, &local_api);
        api = &local_api;
    }

    RankState local_ranks[kGpus];
    RankState *ranks = shared_rank_buffers ? shared_rank_buffers->ranks : local_ranks;
    int aggregate_routes = 0;
    int min_routes = std::numeric_limits<int>::max();
    int max_routes = 0;
    uint64_t ep_loaded_bytes = 0;

    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        r.rank = p;
        r.device = opt.devices[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!shared_api && api->init(r.device) != 0) {
            std::fprintf(stderr, "ggml_turbomind_init failed on device %d\n", r.device);
            if (!shared_api) {
                api->shutdown();
                dlclose(lib);
            }
            close_local_runtime();
            return 7;
        }
        if (!shared_rank_buffers) {
            CHECK_CUDA(cudaStreamCreate(&r.stream));
            CHECK_CUDA(cudaEventCreate(&r.start));
            CHECK_CUDA(cudaEventCreate(&r.mid));
            CHECK_CUDA(cudaEventCreate(&r.stop));

            std::vector<int> offsets;
            std::vector<int> route_slots;
            build_offsets_for_rank(p, opt.slots, opt.top_k, &offsets, &route_slots, &r.routes,
                                   &r.active_experts, &r.max_routes_per_expert);

            const size_t a_elems = (size_t)r.routes * kHidden;
            CHECK_CUDA(cudaMalloc(&r.d_offsets, offsets.size() * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(r.d_offsets, offsets.data(), offsets.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&r.d_route_slots, route_slots.size() * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(r.d_route_slots, route_slots.data(),
                                  route_slots.size() * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&r.d_a, a_elems * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&r.d_gated, (size_t)r.routes * kMid * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&r.d_down, a_elems * sizeof(__half)));

            std::mt19937 rng(0xE2350000u + (uint32_t)p * 97u);
            std::uniform_real_distribution<float> dist(-0.003f, 0.003f);
            std::vector<__half> h_a(a_elems);
            for (__half &v : h_a) v = __float2half(dist(rng));
            CHECK_CUDA(cudaMemcpy(r.d_a, h_a.data(), a_elems * sizeof(__half),
                                  cudaMemcpyHostToDevice));
        }
        aggregate_routes += r.routes;
        min_routes = std::min(min_routes, r.routes);
        max_routes = std::max(max_routes, r.routes);

        std::vector<int> active;
        for (int e = 0; e < kActiveLocalExperts; ++e) active.push_back(e);
        if (pack_descriptor_set(r.device, bindings.gated, p, active, opt.pack_dir,
                                &r.gated, &ep_loaded_bytes) != 0 ||
            pack_descriptor_set(r.device, bindings.down, p, active, opt.pack_dir,
                               &r.down, &ep_loaded_bytes) != 0) {
            close_local_runtime();
            return 8;
        }
        layer_stats.gpu[p].ep_loaded_bytes = ep_loaded_bytes;
    }
    layer_stats.ep_loaded_bytes = ep_loaded_bytes;

    if (!opt.skip_predecode_probes) {
        for (int i = 0; i < opt.warmup; ++i) {
            for (int p = 0; p < kGpus; ++p) {
                if (run_gate(ranks[p], *api) != 0 || run_down(ranks[p], *api) != 0) {
                    close_local_runtime();
                    return 9;
                }
            }
            for (int p = 0; p < kGpus; ++p) {
                CHECK_CUDA(cudaSetDevice(ranks[p].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[p].stream));
            }
        }

        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_CUDA(cudaEventRecord(ranks[p].start, ranks[p].stream));
        }
        for (int i = 0; i < opt.iters; ++i) {
            for (int p = 0; p < kGpus; ++p) {
                if (run_gate(ranks[p], *api) != 0) return 10;
            }
        }
        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_CUDA(cudaEventRecord(ranks[p].mid, ranks[p].stream));
        }
        for (int i = 0; i < opt.iters; ++i) {
            for (int p = 0; p < kGpus; ++p) {
                if (run_down(ranks[p], *api) != 0) return 11;
            }
        }
        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_CUDA(cudaEventRecord(ranks[p].stop, ranks[p].stream));
        }
    }

    double worst_gate_ms = 0.0;
    double worst_down_ms = 0.0;
    double worst_ep_ms = 0.0;
    for (int p = 0; p < kGpus; ++p) {
        CHECK_CUDA(cudaSetDevice(ranks[p].device));
        double gate_ms = 0.0;
        double down_ms = 0.0;
        if (!opt.skip_predecode_probes) {
            CHECK_CUDA(cudaEventSynchronize(ranks[p].stop));
            gate_ms = (double)elapsed_ms(ranks[p].start, ranks[p].mid) / opt.iters;
            down_ms = (double)elapsed_ms(ranks[p].mid, ranks[p].stop) / opt.iters;
        }
        worst_gate_ms = std::max(worst_gate_ms, gate_ms);
        worst_down_ms = std::max(worst_down_ms, down_ms);
        worst_ep_ms = std::max(worst_ep_ms, gate_ms + down_ms);
        std::printf("rank\t%d\tdevice\t%d\troutes\t%d\tactive_local_experts\t%d\t"
                    "max_routes_per_expert\t%d\tgate_ms\t%.6f\tdown_ms\t%.6f\t"
                    "ep_ms\t%.6f\tdense_rows\t%llu\tcontrol_rows\t%llu\t"
                    "expert_rows\t%llu\tkv_rows\t%llu\tcomp_rows\t%llu\t"
                    "checksum\t%llu\n",
                    p, ranks[p].device, ranks[p].routes, ranks[p].active_experts,
                    ranks[p].max_routes_per_expert, gate_ms, down_ms, gate_ms + down_ms,
                    (unsigned long long)layer_stats.gpu[p].dense_rows,
                    (unsigned long long)layer_stats.gpu[p].control_rows,
                    (unsigned long long)layer_stats.gpu[p].expert_rows,
                    (unsigned long long)layer_stats.gpu[p].kv_rows,
                    (unsigned long long)layer_stats.gpu[p].comp_rows,
                    (unsigned long long)layer_stats.gpu[p].checksum);
    }

    double repeat_max_abs = 0.0;
    int repeat_bad = 0;
    int repeat_nan = 0;
    if (!opt.skip_predecode_probes) {
        for (int p = 0; p < kGpus; ++p) {
            if (check_repeat(ranks[p], *api, &repeat_max_abs, &repeat_bad, &repeat_nan) != 0) {
                close_local_runtime();
                return 12;
            }
        }
    }

    ComposeStats compose;
    const int compose_rc = run_next_hidden_compose(opt, rows, ranks, &compose);
    if (compose.enabled) {
        std::printf("tp_ep_next_hidden_compose\tslots\t%d\tctx\t%llu\t"
                    "hidden_shard\t%d\tep_contribution_bytes\t%llu\t"
                    "ep_return_dtype\t%s\tep_return_bytes\t%llu\tdense_hmma\t%d\t"
                    "dense_f16_cublas\t%d\t"
                    "attn_dense_ms\t%.6f\t"
                    "shared_dense_ms\t%.6f\tfused_compose_sum\t%d\tcompose_ms\t%.6f\t"
                    "checksum\t%llu\tfinite_bad\t%d\trepeat_max_abs\t%.9f\t"
                    "repeat_bad\t%d\t%s\n",
                    opt.slots, (unsigned long long)cfg.ctx, kHidden / kGpus,
                    (unsigned long long)compose.ep_contribution_bytes,
                    compose.ep_return_fp16 ? "fp16" : "fp32",
                    (unsigned long long)compose.ep_return_bytes,
                    compose.dense_hmma_compose ? 1 : 0,
                    compose.dense_f16_cublas_compose ? 1 : 0,
                    compose.attn_dense_ms, compose.shared_dense_ms,
                    compose.fused_compose_sum ? 1 : 0,
                    compose.compose_ms, (unsigned long long)compose.checksum,
                    compose.finite_bad, compose.repeat_max_abs,
                    compose.repeat_bad, compose.pass ? "PASS" : "FAIL");
    }
    if (compose_rc != 0) {
        close_local_runtime();
        return 13;
    }

    DecodeLoopStats decode_loop;
    const int decode_rc = run_decode_loop(opt, rows, ranks, *api, dense_f16_cache, &decode_loop);
    if (decode_loop.enabled) {
        std::printf("tp_ep_decode_loop\tsteps\t%d\tslots\t%d\tslot_steps\t%llu\t"
                    "total_ms\t%.6f\tms_per_step\t%.6f\tslot_step_tok_s\t%.6f\t"
                    "dense_hmma\t%d\tdense_f16_cublas\t%d\tdense_f16_cache\t%d\t"
                    "ep_ms_per_step\t%.6f\tdense_ms_per_step\t%.6f\t"
                    "fused_compose_sum\t%d\tcompose_ms_per_step\t%.6f\t"
                    "dense_loaded_bytes\t%llu\t"
                    "ep_contribution_bytes\t%llu\tep_return_dtype\t%s\t"
                    "ep_return_bytes\t%llu\t"
                    "checksum\t%llu\tfinite_bad\t%d\t%s\n",
                    decode_loop.steps, decode_loop.slots,
                    (unsigned long long)decode_loop.slot_steps,
                    decode_loop.total_ms, decode_loop.ms_per_step,
                    decode_loop.tok_s,
                    decode_loop.dense_hmma_compose ? 1 : 0,
                    decode_loop.dense_f16_cublas_compose ? 1 : 0,
                    decode_loop.dense_f16_cache_compose ? 1 : 0,
                    decode_loop.ep_ms_per_step,
                    decode_loop.dense_ms_per_step,
                    decode_loop.fused_compose_sum ? 1 : 0,
                    decode_loop.compose_ms_per_step,
                    (unsigned long long)decode_loop.dense_loaded_bytes,
                    (unsigned long long)decode_loop.ep_contribution_bytes,
                    decode_loop.ep_return_fp16 ? "fp16" : "fp32",
                    (unsigned long long)decode_loop.ep_return_bytes,
                    (unsigned long long)decode_loop.checksum,
                    decode_loop.finite_bad,
                    decode_loop.pass ? "PASS" : "FAIL");
    }
    if (decode_rc != 0) {
        close_local_runtime();
        return 14;
    }

    const uint64_t dispatch_bytes = (uint64_t)aggregate_routes * kHidden * sizeof(__half);
    const uint64_t return_bytes = dispatch_bytes;
    const double imbalance = min_routes > 0 ? (double)max_routes / (double)min_routes : 0.0;
    const double scaffold_ms = descriptor_ms + dense_kv_ms + worst_ep_ms;
    const bool comp_rows_expected = ds4_layer_ratio(opt.layer) != 0;
    const bool pass = layer_stats.dense_rows > 0 &&
                      layer_stats.control_rows > 0 &&
                      layer_stats.expert_rows > 0 &&
                      layer_stats.kv_rows > 0 &&
                      (!comp_rows_expected || layer_stats.comp_rows > 0) &&
                      (opt.skip_descriptor_checks || layer_stats.checksum != 0) &&
                      kv_result.max_abs == 0.0 &&
                      repeat_bad == 0 &&
                      repeat_nan == 0 &&
                      (!dense_compute.enabled || dense_compute.pass) &&
                      (!bf16_compute.enabled || bf16_compute.pass) &&
                      (!compose.enabled || compose.pass) &&
                      (!decode_loop.enabled || decode_loop.pass);

    std::printf("runtime_bytes_per_gpu\thidden\t%llu\tkv\t%llu\tcomp_state\t%llu\t"
                "scratch\t%llu\ttotal\t%llu\n",
                (unsigned long long)runtime_report.gpu[0].hidden_bytes,
                (unsigned long long)runtime_report.gpu[0].kv_bytes,
                (unsigned long long)runtime_report.gpu[0].comp_state_bytes,
                (unsigned long long)runtime_report.gpu[0].scratch_bytes,
                (unsigned long long)runtime_report.gpu[0].total_bytes);
    std::printf("dense_kv_slice\tlayer\t%d\tratio\t%d\tslot\t%u\tposition\t%llu\t"
                "attn_row\t%llu\tindexer_row\t%llu\tattn_row_bytes\t%llu\t"
                "indexer_row_bytes\t%llu\tmax_abs\t%.9f\tdense_kv_ms\t%.6f\n",
                kv_result.layer, kv_result.ratio, kv_result.slot,
                (unsigned long long)kv_result.position,
                (unsigned long long)kv_result.attn_row,
                (unsigned long long)kv_result.indexer_row,
                (unsigned long long)kv_result.attn_row_bytes[0],
                (unsigned long long)kv_result.indexer_row_bytes[0],
                kv_result.max_abs, dense_kv_ms);
    std::printf("tp_ep_full_layer_scaffold\tslots\t%d\tctx\t%llu\ttop_k\t%d\t"
                "layer\t%d\ttotal_rows\t%llu\tdense_rows\t%llu\tcontrol_rows\t%llu\t"
                "expert_rows\t%llu\tkv_rows\t%llu\tcomp_rows\t%llu\t"
                "dense_loaded_bytes\t%llu\tcontrol_loaded_bytes\t%llu\t"
                "ep_loaded_bytes\t%llu\tdescriptor_checksum\t%llu\t"
                "dense_compute_tensor\t%s\tdense_compute_rows_per_gpu\t%d\t"
                "dense_compute_cols\t%d\tdense_compute_slots\t%d\t"
                "dense_compute_loaded_bytes\t%llu\tdense_compute_ms\t%.6f\t"
                "dense_compute_repeat_max_abs\t%.9f\tdense_compute_repeat_bad\t%d\t"
                "dense_compute_repeat_nan\t%d\tdense_compute_oracle_max_abs\t%.9f\t"
                "dense_compute_oracle_bad\t%d\tdense_compute_pass\t%d\t"
                "bf16_compute_tensor\t%s\tbf16_compute_rows_per_gpu\t%d\t"
                "bf16_compute_cols\t%d\tbf16_compute_slots\t%d\t"
                "bf16_compute_loaded_bytes\t%llu\tbf16_compute_ms\t%.6f\t"
                "bf16_compute_repeat_max_abs\t%.9f\tbf16_compute_repeat_bad\t%d\t"
                "bf16_compute_repeat_nan\t%d\tbf16_compute_oracle_max_abs\t%.9f\t"
                "bf16_compute_oracle_bad\t%d\tbf16_compute_pass\t%d\t"
                "compose_next_hidden\t%d\tcompose_ep_contribution_bytes\t%llu\t"
                "compose_ep_return_dtype\t%s\tcompose_ep_return_bytes\t%llu\t"
                "compose_dense_hmma\t%d\tcompose_dense_f16_cublas\t%d\t"
                "compose_attn_dense_ms\t%.6f\t"
                "compose_shared_dense_ms\t%.6f\tcompose_fused_sum\t%d\t"
                "compose_ms\t%.6f\t"
                "compose_checksum\t%llu\tcompose_finite_bad\t%d\t"
                "compose_repeat_max_abs\t%.9f\tcompose_repeat_bad\t%d\t"
                "compose_pass\t%d\t"
                "decode_steps\t%d\tdecode_slot_steps\t%llu\tdecode_total_ms\t%.6f\t"
                "decode_ms_per_step\t%.6f\tdecode_slot_step_tok_s\t%.6f\t"
                "decode_dense_hmma\t%d\tdecode_dense_f16_cublas\t%d\t"
                "decode_dense_f16_cache\t%d\t"
                "decode_ep_ms_per_step\t%.6f\tdecode_dense_ms_per_step\t%.6f\t"
                "decode_fused_compose_sum\t%d\tdecode_compose_ms_per_step\t%.6f\t"
                "decode_ep_return_dtype\t%s\t"
                "decode_ep_return_bytes\t%llu\tdecode_checksum\t%llu\t"
                "decode_finite_bad\t%d\tdecode_pass\t%d\t"
                "aggregate_routes\t%d\tdispatch_bytes\t%llu\treturn_bytes\t%llu\t"
                "route_imbalance\t%.6f\tdescriptor_ms\t%.6f\tdense_kv_ms\t%.6f\t"
                "worst_gate_ms\t%.6f\tworst_down_ms\t%.6f\tworst_ep_ms\t%.6f\t"
                "scaffold_ms\t%.6f\trepeat_max_abs\t%.9f\trepeat_bad\t%d\t"
                "repeat_nan\t%d\t%s\n",
                opt.slots, (unsigned long long)cfg.ctx, opt.top_k, opt.layer,
                (unsigned long long)layer_stats.total_rows,
                (unsigned long long)layer_stats.dense_rows,
                (unsigned long long)layer_stats.control_rows,
                (unsigned long long)layer_stats.expert_rows,
                (unsigned long long)layer_stats.kv_rows,
                (unsigned long long)layer_stats.comp_rows,
                (unsigned long long)layer_stats.dense_loaded_bytes,
                (unsigned long long)layer_stats.control_loaded_bytes,
                (unsigned long long)layer_stats.ep_loaded_bytes,
                (unsigned long long)layer_stats.checksum,
                dense_compute.enabled ? dense_compute.tensor_id.c_str() : "disabled",
                dense_compute.rows_per_gpu,
                dense_compute.cols,
                dense_compute.slots,
                (unsigned long long)dense_compute.loaded_bytes,
                dense_compute.compute_ms,
                dense_compute.repeat_max_abs,
                dense_compute.repeat_bad,
                dense_compute.repeat_nan,
                dense_compute.oracle_max_abs,
                dense_compute.oracle_bad,
                dense_compute.enabled && dense_compute.pass ? 1 : 0,
                bf16_compute.enabled ? bf16_compute.tensor_id.c_str() : "disabled",
                bf16_compute.rows_per_gpu,
                bf16_compute.cols,
                bf16_compute.slots,
                (unsigned long long)bf16_compute.loaded_bytes,
                bf16_compute.compute_ms,
                bf16_compute.repeat_max_abs,
                bf16_compute.repeat_bad,
                bf16_compute.repeat_nan,
                bf16_compute.oracle_max_abs,
                bf16_compute.oracle_bad,
                bf16_compute.enabled && bf16_compute.pass ? 1 : 0,
                compose.enabled ? 1 : 0,
                (unsigned long long)compose.ep_contribution_bytes,
                compose.ep_return_fp16 ? "fp16" : "fp32",
                (unsigned long long)compose.ep_return_bytes,
                compose.dense_hmma_compose ? 1 : 0,
                compose.dense_f16_cublas_compose ? 1 : 0,
                compose.attn_dense_ms,
                compose.shared_dense_ms,
                compose.fused_compose_sum ? 1 : 0,
                compose.compose_ms,
                (unsigned long long)compose.checksum,
                compose.finite_bad,
                compose.repeat_max_abs,
                compose.repeat_bad,
                compose.enabled && compose.pass ? 1 : 0,
                decode_loop.steps,
                (unsigned long long)decode_loop.slot_steps,
                decode_loop.total_ms,
                decode_loop.ms_per_step,
                decode_loop.tok_s,
                decode_loop.dense_hmma_compose ? 1 : 0,
                decode_loop.dense_f16_cublas_compose ? 1 : 0,
                decode_loop.dense_f16_cache_compose ? 1 : 0,
                decode_loop.ep_ms_per_step,
                decode_loop.dense_ms_per_step,
                decode_loop.fused_compose_sum ? 1 : 0,
                decode_loop.compose_ms_per_step,
                decode_loop.ep_return_fp16 ? "fp16" : "fp32",
                (unsigned long long)decode_loop.ep_return_bytes,
                (unsigned long long)decode_loop.checksum,
                decode_loop.finite_bad,
                decode_loop.enabled && decode_loop.pass ? 1 : 0,
                aggregate_routes,
                (unsigned long long)dispatch_bytes,
                (unsigned long long)return_bytes,
                imbalance, descriptor_ms, dense_kv_ms, worst_gate_ms, worst_down_ms,
                worst_ep_ms, scaffold_ms, repeat_max_abs, repeat_bad, repeat_nan,
                pass ? "PASS" : "FAIL");

    if (summary) {
        summary->layer = opt.layer;
        summary->ratio = ds4_layer_ratio(opt.layer);
        summary->pass = pass;
        summary->total_rows = layer_stats.total_rows;
        summary->dense_rows = layer_stats.dense_rows;
        summary->control_rows = layer_stats.control_rows;
        summary->expert_rows = layer_stats.expert_rows;
        summary->kv_rows = layer_stats.kv_rows;
        summary->comp_rows = layer_stats.comp_rows;
        summary->decode_ms_per_step = decode_loop.ms_per_step;
        summary->decode_slot_step_tok_s = decode_loop.tok_s;
        summary->decode_ep_ms_per_step = decode_loop.ep_ms_per_step;
        summary->decode_dense_ms_per_step = decode_loop.dense_ms_per_step;
        summary->decode_compose_ms_per_step = decode_loop.compose_ms_per_step;
        summary->decode_checksum = decode_loop.checksum;
    }

    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        free_packed(r.gated);
        r.gated = PackedExperts{};
        free_packed(r.down);
        r.down = PackedExperts{};
        if (!shared_rank_buffers) {
            CHECK_CUDA(cudaFree(r.d_offsets));
            CHECK_CUDA(cudaFree(r.d_route_slots));
            CHECK_CUDA(cudaFree(r.d_a));
            CHECK_CUDA(cudaFree(r.d_gated));
            CHECK_CUDA(cudaFree(r.d_down));
            if (r.d_ep_contrib_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_all));
            if (r.d_ep_contrib_half_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_half_all));
            for (int src = 0; src < kGpus; ++src) {
                if (r.d_ep_remote[src]) CHECK_CUDA(cudaFree(r.d_ep_remote[src]));
                if (r.d_ep_remote_half[src]) CHECK_CUDA(cudaFree(r.d_ep_remote_half[src]));
            }
            if (r.d_ep_sum) CHECK_CUDA(cudaFree(r.d_ep_sum));
            if (r.d_next_hidden) CHECK_CUDA(cudaFree(r.d_next_hidden));
            CHECK_CUDA(cudaEventDestroy(r.start));
            CHECK_CUDA(cudaEventDestroy(r.mid));
            CHECK_CUDA(cudaEventDestroy(r.stop));
            CHECK_CUDA(cudaStreamDestroy(r.stream));
        }
    }
    if (!shared_api) {
        api->shutdown();
        dlclose(lib);
    }
    close_local_runtime();
    if (!shared_dense_f16_cache) free_dense_f16_cache(local_dense_f16_cache, opt);
    return pass ? 0 : 1;
}

int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        usage(argv[0]);
        return 2;
    }

    if (!opt.all_layers) {
        return run_layer(opt, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    DenseF16Cache all_layer_dense_f16_cache;
    DenseF16Cache *shared_dense_f16_cache = nullptr;
    if (opt.dense_f16_cache_compose) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0) {
            std::fprintf(stderr, "all-layer contract parse failed bad_rows=%llu\n",
                         (unsigned long long)all_stats.bad_rows);
            return 2;
        }
        const auto cache_start = std::chrono::steady_clock::now();
        if (prepare_dense_f16_cache(opt, all_rows, &all_layer_dense_f16_cache) != 0) {
            std::fprintf(stderr, "all-layer dense f16 cache prepare failed\n");
            return 4;
        }
        const auto cache_stop = std::chrono::steady_clock::now();
        const double cache_ms =
            std::chrono::duration<double, std::milli>(cache_stop - cache_start).count();
        shared_dense_f16_cache = &all_layer_dense_f16_cache;
        std::printf("tp_ep_all_layer_dense_f16_cache\trows\t%llu\t"
                    "source_bytes\t%llu\tcache_bytes\t%llu\t"
                    "cache_aligned_bytes\t%llu\tmax_temp_bytes\t%llu\t"
                    "cache_ms\t%.6f\tPASS\n",
                    (unsigned long long)all_layer_dense_f16_cache.rows,
                    (unsigned long long)all_layer_dense_f16_cache.source_bytes,
                    (unsigned long long)all_layer_dense_f16_cache.cache_bytes,
                    (unsigned long long)all_layer_dense_f16_cache.cache_aligned_bytes,
                    (unsigned long long)all_layer_dense_f16_cache.max_temp_bytes,
                    cache_ms);
    }

    SharedApi shared_api;
    if (open_shared_api(opt, &shared_api) != 0) {
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 6;
    }
    std::printf("tp_ep_all_layer_turbomind_api_shared\tdevices\t%d\tPASS\n", kGpus);

    SharedRankBuffers shared_rank_buffers;
    if (open_shared_rank_buffers(opt, &shared_rank_buffers) != 0) {
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 7;
    }
    std::printf("tp_ep_all_layer_rank_buffers_shared\tdevices\t%d\tcore_bytes\t%llu\tPASS\n",
                kGpus, (unsigned long long)shared_rank_buffers.core_bytes);

    SharedTpRuntime shared_tp_runtime;
    if (open_shared_tp_runtime(opt, &shared_tp_runtime) != 0) {
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 8;
    }
    std::printf("tp_ep_all_layer_tp_runtime_shared\tdevices\t%d\tslots\t%d\tctx\t262144\t"
                "kv_bytes_per_gpu\t%llu\tcomp_state_bytes_per_gpu\t%llu\t"
                "scratch_bytes_per_gpu\t%llu\ttotal_bytes_per_gpu\t%llu\tPASS\n",
                kGpus, opt.slots,
                (unsigned long long)shared_tp_runtime.report.gpu[0].kv_bytes,
                (unsigned long long)shared_tp_runtime.report.gpu[0].comp_state_bytes,
                (unsigned long long)shared_tp_runtime.report.gpu[0].scratch_bytes,
                (unsigned long long)shared_tp_runtime.report.gpu[0].total_bytes);

    int pass_layers = 0;
    double sum_decode_ms = 0.0;
    double sum_ep_ms = 0.0;
    double sum_dense_ms = 0.0;
    double sum_compose_ms = 0.0;
    uint64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int layer = 0; layer < 43; ++layer) {
        Options layer_opt = opt;
        layer_opt.layer = layer;
        LayerRunSummary s;
        const int rc = run_layer(layer_opt, &s, shared_dense_f16_cache, &shared_api,
                                 &shared_rank_buffers, &shared_tp_runtime);
        std::printf("tp_ep_all_layer_item\tlayer\t%d\tratio\t%d\t"
                    "total_rows\t%llu\tdense_rows\t%llu\tcontrol_rows\t%llu\t"
                    "expert_rows\t%llu\tkv_rows\t%llu\tcomp_rows\t%llu\t"
                    "decode_ms_per_step\t%.6f\tdecode_slot_step_tok_s\t%.6f\t"
                    "decode_ep_ms_per_step\t%.6f\tdecode_dense_ms_per_step\t%.6f\t"
                    "decode_compose_ms_per_step\t%.6f\tdecode_checksum\t%llu\t%s\n",
                    s.layer, s.ratio,
                    (unsigned long long)s.total_rows,
                    (unsigned long long)s.dense_rows,
                    (unsigned long long)s.control_rows,
                    (unsigned long long)s.expert_rows,
                    (unsigned long long)s.kv_rows,
                    (unsigned long long)s.comp_rows,
                    s.decode_ms_per_step,
                    s.decode_slot_step_tok_s,
                    s.decode_ep_ms_per_step,
                    s.decode_dense_ms_per_step,
                    s.decode_compose_ms_per_step,
                    (unsigned long long)s.decode_checksum,
                    (rc == 0 && s.pass) ? "PASS" : "FAIL");
        if (rc == 0 && s.pass) {
            pass_layers++;
            sum_decode_ms += s.decode_ms_per_step;
            sum_ep_ms += s.decode_ep_ms_per_step;
            sum_dense_ms += s.decode_dense_ms_per_step;
            sum_compose_ms += s.decode_compose_ms_per_step;
            checksum ^= s.decode_checksum + (uint64_t)(layer + 1) * 104729ull;
        } else {
            const auto stop = std::chrono::steady_clock::now();
            const double wall_ms =
                std::chrono::duration<double, std::milli>(stop - start).count();
            std::printf("tp_ep_all_layer_scaffold\tlayers\t43\tpass_layers\t%d\t"
                        "failed_layer\t%d\tdescriptor_checks\t%d\tpredecode_probes\t%d\t"
                        "shared_api\t%d\tshared_rank_buffers\t%d\tshared_tp_runtime\t%d\t"
                        "wall_ms\t%.6f\tFAIL\n",
                        pass_layers, layer, opt.skip_descriptor_checks ? 0 : 1,
                        opt.skip_predecode_probes ? 0 : 1, shared_api.initialized ? 1 : 0,
                        shared_rank_buffers.initialized ? 1 : 0,
                        shared_tp_runtime.initialized ? 1 : 0, wall_ms);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return rc == 0 ? 1 : rc;
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    const double slot_step_tok_s = sum_decode_ms > 0.0
        ? (double)opt.slots * 1000.0 / sum_decode_ms
        : 0.0;
    std::printf("tp_ep_all_layer_scaffold\tlayers\t43\tpass_layers\t%d\t"
                "slots\t%d\tctx\t262144\tdecode_steps_per_layer\t%d\t"
                "descriptor_checks\t%d\tpredecode_probes\t%d\tshared_api\t%d\t"
                "shared_rank_buffers\t%d\tshared_tp_runtime\t%d\t"
                "sum_decode_ms_per_token\t%.6f\tprojected_slot_step_tok_s\t%.6f\t"
                "sum_ep_ms\t%.6f\tsum_dense_ms\t%.6f\tsum_compose_ms\t%.6f\t"
                "wall_ms\t%.6f\tchecksum\t%llu\tPASS\n",
                pass_layers, opt.slots, opt.decode_steps,
                opt.skip_descriptor_checks ? 0 : 1,
                opt.skip_predecode_probes ? 0 : 1,
                shared_api.initialized ? 1 : 0,
                shared_rank_buffers.initialized ? 1 : 0,
                shared_tp_runtime.initialized ? 1 : 0,
                sum_decode_ms, slot_step_tok_s, sum_ep_ms, sum_dense_ms,
                sum_compose_ms, wall_ms, (unsigned long long)checksum);
    close_shared_tp_runtime(&shared_tp_runtime);
    close_shared_rank_buffers(&shared_rank_buffers);
    close_shared_api(&shared_api);
    if (shared_dense_f16_cache) {
        free_dense_f16_cache(all_layer_dense_f16_cache, opt);
    }
    return 0;
}
