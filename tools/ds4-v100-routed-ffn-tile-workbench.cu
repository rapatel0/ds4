// Sprint 214 routed-FFN workbench.
//
// This is intentionally standalone. It compares the proven TurboMind V100
// six-route FFN sequence against a tile-local down+route-reduce candidate for
// the exact production decode shape:
//   routes=6, hidden=4096, middle=2048, active experts=6.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h"

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
} } while (0)

typedef int (*pfn_init)(int);
typedef void (*pfn_shutdown)(void);
typedef int (*pfn_packed_bytes)(int, int, int, int, size_t *, size_t *);
typedef int (*pfn_pack_weight)(const void *, int, int, int, int, void *, void *, int *, void *);
typedef int (*pfn_mul_mat_grouped_gated_silu_total_tokens)(
    const void *, const int *, const int *, int, int,
    const void * const *, const void * const *, int, int, int, int, int, void *, void *);
typedef int (*pfn_mul_mat_grouped_total_tokens)(
    const void *, const int *, const int *, int, int,
    const void * const *, const void * const *, int, int, int, int, int, void *, void *);
typedef int (*pfn_ds4_mxfp4_gated_silu_6)(
    const void *, const int *, int, int, const void * const *, const void * const *,
    int, void *, void *);
typedef int (*pfn_ds4_mxfp4_down_reduce)(
    const void *, const int *, int, int, const void * const *, const void * const *,
    int, float *, const int *, const float *, int, void *);
typedef int (*pfn_ds4_reduce6_half_to_float)(
    const void *, float *, const int *, const float *, int, int, void *);

struct block_mxfp4 {
    uint8_t e;
    uint8_t qs[16];
};

struct alignas(16) StridedPtrH {
    void *p;
    int stride;
};
static_assert(sizeof(StridedPtrH) == 16, "StridedPtrH ABI mismatch");

struct PackedExperts {
    std::vector<void *> d_w_active;
    std::vector<void *> d_s_active;
    void *d_w_table = nullptr;
    void *d_s_table = nullptr;
    int k_pack = 0;
};

struct Api {
    void *lib = nullptr;
    pfn_init init = nullptr;
    pfn_shutdown shutdown = nullptr;
    pfn_packed_bytes packed_bytes = nullptr;
    pfn_pack_weight pack_weight = nullptr;
    pfn_mul_mat_grouped_gated_silu_total_tokens mmgs = nullptr;
    pfn_mul_mat_grouped_total_tokens mmgt = nullptr;
    pfn_ds4_mxfp4_gated_silu_6 gated6 = nullptr;
    pfn_ds4_mxfp4_down_reduce down_reduce6 = nullptr;
    pfn_ds4_reduce6_half_to_float reduce6_half = nullptr;
};

static int arg_int(int argc, char **argv, const char *name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return std::atoi(argv[i + 1]);
        }
    }
    return fallback;
}

static const char *arg_str(int argc, char **argv, const char *name, const char *fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return fallback;
}

static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static void make_mxfp4_fixture(std::vector<block_mxfp4> &blocks, int n, int k, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> e_dist(116, 122);
    std::uniform_int_distribution<int> q_dist(0, 255);
    blocks.resize((size_t)n * (k / 32));
    for (block_mxfp4 &b : blocks) {
        b.e = (uint8_t)e_dist(rng);
        for (uint8_t &q : b.qs) {
            q = (uint8_t)q_dist(rng);
        }
    }
}

static void make_fused_interleaved_fixture(std::vector<block_mxfp4> &fused,
                                           const std::vector<block_mxfp4> &gate,
                                           const std::vector<block_mxfp4> &up,
                                           int n,
                                           int k) {
    const int blocks_per_row = k / 32;
    fused.resize((size_t)2 * n * blocks_per_row);
    for (int row = 0; row < n; ++row) {
        const size_t src = (size_t)row * blocks_per_row;
        const size_t gate_dst = (size_t)(2 * row) * blocks_per_row;
        const size_t up_dst = (size_t)(2 * row + 1) * blocks_per_row;
        std::copy(gate.begin() + src, gate.begin() + src + blocks_per_row,
                  fused.begin() + gate_dst);
        std::copy(up.begin() + src, up.begin() + src + blocks_per_row,
                  fused.begin() + up_dst);
    }
}

static void free_packed(PackedExperts &p) {
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

static int pack_fixture_set(Api &api,
                            int ggml_type,
                            int n,
                            int k,
                            int group_size,
                            int num_experts,
                            const std::vector<int> &active,
                            const std::vector<std::vector<block_mxfp4>> &fixtures,
                            PackedExperts &out) {
    size_t wb = 0;
    size_t sb = 0;
    int rc = api.packed_bytes(ggml_type, n, k, group_size, &wb, &sb);
    if (rc != 0) {
        std::fprintf(stderr, "[tile_workbench] packed_bytes n=%d k=%d rc=%d\n", n, k, rc);
        return 1;
    }

    out.d_w_active.assign(active.size(), nullptr);
    out.d_s_active.assign(active.size(), nullptr);
    for (size_t i = 0; i < active.size(); ++i) {
        void *d_src = nullptr;
        CHECK_CUDA(cudaMalloc(&d_src, fixtures[i].size() * sizeof(block_mxfp4)));
        CHECK_CUDA(cudaMemcpy(d_src, fixtures[i].data(),
                              fixtures[i].size() * sizeof(block_mxfp4),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&out.d_w_active[i], wb));
        CHECK_CUDA(cudaMalloc(&out.d_s_active[i], sb));
        int this_pack = 0;
        rc = api.pack_weight(d_src, ggml_type, n, k, group_size,
                             out.d_w_active[i], out.d_s_active[i], &this_pack, nullptr);
        CHECK_CUDA(cudaFree(d_src));
        if (rc != 0) {
            std::fprintf(stderr, "[tile_workbench] pack expert=%d n=%d k=%d rc=%d\n",
                         active[i], n, k, rc);
            return 2;
        }
        if (i == 0) {
            out.k_pack = this_pack;
        } else if (this_pack != out.k_pack) {
            std::fprintf(stderr, "[tile_workbench] inconsistent k_pack 0x%x vs 0x%x\n",
                         out.k_pack, this_pack);
            return 3;
        }
    }

    std::vector<StridedPtrH> h_w(num_experts);
    std::vector<StridedPtrH> h_s(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        h_w[e] = StridedPtrH{out.d_w_active[0], k * 32};
        h_s[e] = StridedPtrH{out.d_s_active[0], n};
    }
    for (size_t i = 0; i < active.size(); ++i) {
        h_w[active[i]] = StridedPtrH{out.d_w_active[i], k * 32};
        h_s[active[i]] = StridedPtrH{out.d_s_active[i], n};
    }
    CHECK_CUDA(cudaMalloc(&out.d_w_table, h_w.size() * sizeof(StridedPtrH)));
    CHECK_CUDA(cudaMalloc(&out.d_s_table, h_s.size() * sizeof(StridedPtrH)));
    CHECK_CUDA(cudaMemcpy(out.d_w_table, h_w.data(), h_w.size() * sizeof(StridedPtrH),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(out.d_s_table, h_s.data(), h_s.size() * sizeof(StridedPtrH),
                          cudaMemcpyHostToDevice));
    return 0;
}

__device__ static inline float wb_e8m0_to_f32(uint8_t e) {
    if (e == 0) return 0.0f;
    if (e == 255) return __int_as_float(0x7f800000);
    return ldexpf(1.0f, (int)e - 127);
}

__device__ static inline float wb_fp4_e2m1_to_f32(uint8_t q) {
    static const float tbl[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = tbl[q & 7u];
    return (q & 8u) ? -v : v;
}

__global__ static void raw_mxfp4_down_reduce6_kernel(const half *gated,
                                                     const block_mxfp4 *down,
                                                     const float *route_weights,
                                                     float *out) {
    constexpr int kRoutes = 6;
    constexpr int kHidden = 4096;
    constexpr int kMid = 2048;
    constexpr int kBlocks = kMid / 32;
    const int col = (int)blockIdx.x;
    if (col >= kHidden) return;

    float acc = 0.0f;
    for (int route = 0; route < kRoutes; ++route) {
        const float route_weight = route_weights[route];
        const half *gated_route = gated + (size_t)route * kMid;
        const block_mxfp4 *row =
            down + ((size_t)route * kHidden + (size_t)col) * kBlocks;
        for (int b = threadIdx.x; b < kBlocks; b += blockDim.x) {
            const block_mxfp4 blk = row[b];
            const float scale = wb_e8m0_to_f32(blk.e);
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint8_t packed = blk.qs[j];
                const int mid0 = b * 32 + j;
                const int mid1 = mid0 + 16;
                const float w0 = wb_fp4_e2m1_to_f32(packed & 0x0f) * scale;
                const float w1 = wb_fp4_e2m1_to_f32((packed >> 4) & 0x0f) * scale;
                acc += route_weight *
                       (__half2float(gated_route[mid0]) * w0 +
                        __half2float(gated_route[mid1]) * w1);
            }
        }
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[col] = partial[0];
    }
}

static Api load_api(const char *path) {
    Api api;
    api.lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!api.lib) {
        std::fprintf(stderr, "[tile_workbench] dlopen %s failed: %s\n", path, dlerror());
        std::exit(2);
    }
    api.init = (pfn_init)dlsym(api.lib, "ggml_turbomind_init");
    api.shutdown = (pfn_shutdown)dlsym(api.lib, "ggml_turbomind_shutdown");
    api.packed_bytes = (pfn_packed_bytes)dlsym(api.lib, "ggml_turbomind_packed_bytes");
    api.pack_weight = (pfn_pack_weight)dlsym(api.lib, "ggml_turbomind_pack_weight_expert");
    api.mmgs = (pfn_mul_mat_grouped_gated_silu_total_tokens)
        dlsym(api.lib, "ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens");
    api.mmgt = (pfn_mul_mat_grouped_total_tokens)
        dlsym(api.lib, "ggml_turbomind_mul_mat_grouped_total_tokens");
    api.gated6 = (pfn_ds4_mxfp4_gated_silu_6)
        dlsym(api.lib, "ggml_turbomind_ds4_mxfp4_gated_silu_6");
    api.down_reduce6 = (pfn_ds4_mxfp4_down_reduce)
        dlsym(api.lib, "ggml_turbomind_ds4_mxfp4_down_6_m16_reduce");
    api.reduce6_half = (pfn_ds4_reduce6_half_to_float)
        dlsym(api.lib, "ggml_turbomind_ds4_reduce6_half_to_float");
    if (!api.init || !api.shutdown || !api.packed_bytes || !api.pack_weight ||
        !api.mmgs || !api.mmgt || !api.down_reduce6 || !api.reduce6_half) {
        std::fprintf(stderr, "[tile_workbench] required symbols missing\n");
        std::exit(2);
    }
    return api;
}

static float time_loop(int warmup, int iters, void (*fn)(void *), void *ctx) {
    for (int i = 0; i < warmup; ++i) {
        fn(ctx);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        fn(ctx);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    const float ms = elapsed_ms(start, stop) / (float)iters;
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

struct BenchCtx {
    Api *api;
    int *d_offsets;
    half *d_a;
    half *d_gated;
    half *d_down_routes;
    float *d_atomic;
    float *d_split;
    float *d_candidate;
    int *d_sorted_pairs;
    float *d_route_weights;
    PackedExperts *gated_packed;
    PackedExperts *down_packed;
    block_mxfp4 *d_down_raw;
    int rc = 0;
};

static void atomic_sequence(void *opaque) {
    BenchCtx *c = (BenchCtx *)opaque;
    c->rc = c->api->gated6
        ? c->api->gated6(c->d_a, c->d_offsets, 6, 6,
                         (const void * const *)c->gated_packed->d_w_table,
                         (const void * const *)c->gated_packed->d_s_table,
                         c->gated_packed->k_pack, c->d_gated, nullptr)
        : c->api->mmgs(c->d_a, nullptr, c->d_offsets, 6, 6,
                       (const void * const *)c->gated_packed->d_w_table,
                       (const void * const *)c->gated_packed->d_s_table,
                       GGML_TM_DTYPE_MXFP4, 4096, 4096, 32,
                       c->gated_packed->k_pack, c->d_gated, nullptr);
    if (c->rc) return;
    CHECK_CUDA(cudaMemset(c->d_atomic, 0, 4096 * sizeof(float)));
    c->rc = c->api->down_reduce6(
        c->d_gated, c->d_offsets, 6, 6,
        (const void * const *)c->down_packed->d_w_table,
        (const void * const *)c->down_packed->d_s_table,
        c->down_packed->k_pack, c->d_atomic, c->d_sorted_pairs, c->d_route_weights, 6, nullptr);
}

static void split_sequence(void *opaque) {
    BenchCtx *c = (BenchCtx *)opaque;
    c->rc = c->api->gated6
        ? c->api->gated6(c->d_a, c->d_offsets, 6, 6,
                         (const void * const *)c->gated_packed->d_w_table,
                         (const void * const *)c->gated_packed->d_s_table,
                         c->gated_packed->k_pack, c->d_gated, nullptr)
        : c->api->mmgs(c->d_a, nullptr, c->d_offsets, 6, 6,
                       (const void * const *)c->gated_packed->d_w_table,
                       (const void * const *)c->gated_packed->d_s_table,
                       GGML_TM_DTYPE_MXFP4, 4096, 4096, 32,
                       c->gated_packed->k_pack, c->d_gated, nullptr);
    if (c->rc) return;
    c->rc = c->api->mmgt(
        c->d_gated, nullptr, c->d_offsets, 6, 6,
        (const void * const *)c->down_packed->d_w_table,
        (const void * const *)c->down_packed->d_s_table,
        GGML_TM_DTYPE_MXFP4, 4096, 2048, 32,
        c->down_packed->k_pack, c->d_down_routes, nullptr);
    if (c->rc) return;
    CHECK_CUDA(cudaMemset(c->d_split, 0, 4096 * sizeof(float)));
    c->rc = c->api->reduce6_half(c->d_down_routes, c->d_split,
                                 c->d_sorted_pairs, c->d_route_weights, 6, 4096, nullptr);
}

static void candidate_sequence(void *opaque) {
    BenchCtx *c = (BenchCtx *)opaque;
    c->rc = c->api->gated6
        ? c->api->gated6(c->d_a, c->d_offsets, 6, 6,
                         (const void * const *)c->gated_packed->d_w_table,
                         (const void * const *)c->gated_packed->d_s_table,
                         c->gated_packed->k_pack, c->d_gated, nullptr)
        : c->api->mmgs(c->d_a, nullptr, c->d_offsets, 6, 6,
                       (const void * const *)c->gated_packed->d_w_table,
                       (const void * const *)c->gated_packed->d_s_table,
                       GGML_TM_DTYPE_MXFP4, 4096, 4096, 32,
                       c->gated_packed->k_pack, c->d_gated, nullptr);
    if (c->rc) return;
    raw_mxfp4_down_reduce6_kernel<<<4096, 256>>>(
        c->d_gated, c->d_down_raw, c->d_route_weights, c->d_candidate);
    CHECK_CUDA(cudaGetLastError());
}

static void candidate_down_only(void *opaque) {
    BenchCtx *c = (BenchCtx *)opaque;
    raw_mxfp4_down_reduce6_kernel<<<4096, 256>>>(
        c->d_gated, c->d_down_raw, c->d_route_weights, c->d_candidate);
    CHECK_CUDA(cudaGetLastError());
}

int main(int argc, char **argv) {
    const char *lib_path = arg_str(argc, argv, "--lib", "./build/turbomind-v100/libggml-turbomind.so");
    const int warmup = arg_int(argc, argv, "--warmup", 20);
    const int iters = arg_int(argc, argv, "--iters", 200);

    constexpr int kRoutes = 6;
    constexpr int kHidden = 4096;
    constexpr int kMid = 2048;
    constexpr int kGroup = 32;
    constexpr int kFusedN = kMid * 2;
    const std::vector<int> active{0, 1, 2, 3, 4, 5};

    Api api = load_api(lib_path);
    if (api.init(0) != 0) {
        std::fprintf(stderr, "[tile_workbench] ggml_turbomind_init failed\n");
        return 2;
    }

    std::vector<std::vector<block_mxfp4>> gate(kRoutes);
    std::vector<std::vector<block_mxfp4>> up(kRoutes);
    std::vector<std::vector<block_mxfp4>> fused(kRoutes);
    std::vector<std::vector<block_mxfp4>> down(kRoutes);
    for (int i = 0; i < kRoutes; ++i) {
        make_mxfp4_fixture(gate[i], kMid, kHidden, 0x47000000u + (uint32_t)i * 101u);
        make_mxfp4_fixture(up[i], kMid, kHidden, 0x55000000u + (uint32_t)i * 131u);
        make_fused_interleaved_fixture(fused[i], gate[i], up[i], kMid, kHidden);
        make_mxfp4_fixture(down[i], kHidden, kMid, 0x63000000u + (uint32_t)i * 137u);
    }

    PackedExperts gated_packed;
    PackedExperts down_packed;
    if (pack_fixture_set(api, GGML_TM_DTYPE_MXFP4, kFusedN, kHidden, kGroup,
                         kRoutes, active, fused, gated_packed) != 0 ||
        pack_fixture_set(api, GGML_TM_DTYPE_MXFP4, kHidden, kMid, kGroup,
                         kRoutes, active, down, down_packed) != 0) {
        api.shutdown();
        return 3;
    }

    std::vector<int> h_offsets{0, 1, 2, 3, 4, 5, 6};
    std::vector<int> h_sorted_pairs{0, 1, 2, 3, 4, 5};
    std::vector<float> h_route_weights(kRoutes, 1.0f);
    std::mt19937 rng(0x21400000u);
    std::uniform_real_distribution<float> ad(-0.1f, 0.1f);
    std::vector<half> h_a((size_t)kRoutes * kHidden);
    for (half &v : h_a) {
        v = __float2half(ad(rng));
    }

    std::vector<block_mxfp4> h_down_raw((size_t)kRoutes * kHidden * (kMid / 32));
    for (int route = 0; route < kRoutes; ++route) {
        std::copy(down[route].begin(), down[route].end(),
                  h_down_raw.begin() + (size_t)route * kHidden * (kMid / 32));
    }

    int *d_offsets = nullptr;
    int *d_sorted_pairs = nullptr;
    float *d_route_weights = nullptr;
    half *d_a = nullptr;
    half *d_gated = nullptr;
    half *d_down_routes = nullptr;
    float *d_atomic = nullptr;
    float *d_split = nullptr;
    float *d_candidate = nullptr;
    block_mxfp4 *d_down_raw = nullptr;
    CHECK_CUDA(cudaMalloc(&d_offsets, h_offsets.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sorted_pairs, h_sorted_pairs.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_route_weights, h_route_weights.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_a, h_a.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_gated, (size_t)kRoutes * kMid * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_down_routes, (size_t)kRoutes * kHidden * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_atomic, kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_split, kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_candidate, kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_down_raw, h_down_raw.size() * sizeof(block_mxfp4)));
    CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sorted_pairs, h_sorted_pairs.data(), h_sorted_pairs.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_route_weights, h_route_weights.data(), h_route_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_down_raw, h_down_raw.data(), h_down_raw.size() * sizeof(block_mxfp4), cudaMemcpyHostToDevice));

    BenchCtx ctx;
    ctx.api = &api;
    ctx.d_offsets = d_offsets;
    ctx.d_a = d_a;
    ctx.d_gated = d_gated;
    ctx.d_down_routes = d_down_routes;
    ctx.d_atomic = d_atomic;
    ctx.d_split = d_split;
    ctx.d_candidate = d_candidate;
    ctx.d_sorted_pairs = d_sorted_pairs;
    ctx.d_route_weights = d_route_weights;
    ctx.gated_packed = &gated_packed;
    ctx.down_packed = &down_packed;
    ctx.d_down_raw = d_down_raw;

    atomic_sequence(&ctx);
    split_sequence(&ctx);
    candidate_sequence(&ctx);
    if (ctx.rc != 0) {
        std::fprintf(stderr, "[tile_workbench] baseline call failed rc=%d\n", ctx.rc);
        return 4;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_atomic(kHidden);
    std::vector<float> h_split(kHidden);
    std::vector<float> h_candidate(kHidden);
    CHECK_CUDA(cudaMemcpy(h_atomic.data(), d_atomic, kHidden * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_split.data(), d_split, kHidden * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_candidate.data(), d_candidate, kHidden * sizeof(float), cudaMemcpyDeviceToHost));

    auto compare = [](const std::vector<float> &ref, const std::vector<float> &got,
                      float &max_abs, float &rel, int &bad) {
        double sum_abs = 0.0;
        double sum_ref = 0.0;
        max_abs = 0.0f;
        bad = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const float d = std::fabs(got[i] - ref[i]);
            max_abs = std::max(max_abs, d);
            sum_abs += d;
            sum_ref += std::fabs(ref[i]);
            if (!std::isfinite(got[i]) || d > 256.0f) {
                bad++;
            }
        }
        rel = sum_ref > 0.0 ? (float)(sum_abs / sum_ref) : 0.0f;
    };

    float split_max_abs = 0.0f;
    float split_rel = 0.0f;
    int split_bad = 0;
    float candidate_max_abs = 0.0f;
    float candidate_rel = 0.0f;
    int candidate_bad = 0;
    compare(h_atomic, h_split, split_max_abs, split_rel, split_bad);
    compare(h_atomic, h_candidate, candidate_max_abs, candidate_rel, candidate_bad);

    const float atomic_ms = time_loop(warmup, iters, atomic_sequence, &ctx);
    if (ctx.rc != 0) {
        std::fprintf(stderr, "[tile_workbench] atomic timing rc=%d\n", ctx.rc);
        return 5;
    }
    const float split_ms = time_loop(warmup, iters, split_sequence, &ctx);
    if (ctx.rc != 0) {
        std::fprintf(stderr, "[tile_workbench] split timing rc=%d\n", ctx.rc);
        return 6;
    }
    atomic_sequence(&ctx);
    CHECK_CUDA(cudaDeviceSynchronize());
    const float candidate_down_ms = time_loop(warmup, iters, candidate_down_only, &ctx);
    const float candidate_ms = time_loop(warmup, iters, candidate_sequence, &ctx);
    CHECK_CUDA(cudaDeviceSynchronize());

    const float best_ms = std::min(atomic_ms, split_ms);
    const float candidate_vs_best = best_ms > 0.0f ? best_ms / candidate_ms : 0.0f;
    std::fprintf(stderr,
                 "[tile_workbench] shape routes=6 hidden=4096 mid=2048 warmup=%d iters=%d "
                 "gated6=%s atomic_sequence_ms=%.6f split_sequence_ms=%.6f "
                 "candidate_sequence_ms=%.6f candidate_down_only_ms=%.6f "
                 "candidate_vs_best=%.3fx split_max_abs=%.4e split_rel=%.4e split_bad=%d/4096 "
                 "candidate_max_abs=%.4e candidate_rel=%.4e candidate_bad=%d/4096 "
                 "decision_gate_ms=0.116100\n",
                 warmup, iters, api.gated6 ? "yes" : "no",
                 atomic_ms, split_ms, candidate_ms, candidate_down_ms,
                 candidate_vs_best,
                 split_max_abs, split_rel, split_bad,
                 candidate_max_abs, candidate_rel, candidate_bad);

    const int ok = split_bad == 0 && split_rel <= 1e-3f &&
                   candidate_bad == 0 && candidate_rel <= 1e-3f;
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_sorted_pairs));
    CHECK_CUDA(cudaFree(d_route_weights));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_gated));
    CHECK_CUDA(cudaFree(d_down_routes));
    CHECK_CUDA(cudaFree(d_atomic));
    CHECK_CUDA(cudaFree(d_split));
    CHECK_CUDA(cudaFree(d_candidate));
    CHECK_CUDA(cudaFree(d_down_raw));
    free_packed(gated_packed);
    free_packed(down_packed);
    api.shutdown();
    dlclose(api.lib);
    return ok ? 0 : 7;
}
