// Two-GPU tensor-parallel routed-FFN proxy for DS4/V100.
//
// This benchmark keeps the production scheduler untouched. It answers one
// bounded question: if we split the DS4 routed FFN middle dimension across two
// V100s, do concurrent half-FFNs plus peer payload movement beat the current
// layer-owned single-GPU FFN shape?

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml-turbomind-api.h"

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
} } while (0)

#define CHECK_NCCL(x) do { ncclResult_t e = (x); if (e != ncclSuccess) { \
    fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(e)); \
    std::exit(1); \
} } while (0)

typedef int  (*pfn_init)(int);
typedef void (*pfn_shutdown)(void);
typedef int  (*pfn_packed_bytes)(int, int, int, int, size_t *, size_t *);
typedef int  (*pfn_pack_weight)(const void *, int, int, int, int, void *, void *, int *, void *);
typedef int  (*pfn_mul_mat_grouped_total_tokens)(const void *, const int *, const int *, int, int,
                                                 const void * const *, const void * const *,
                                                 int, int, int, int, int, void *, void *);
typedef int  (*pfn_mul_mat_grouped_gated_silu_total_tokens)(const void *, const int *, const int *, int, int,
                                                            const void * const *, const void * const *,
                                                            int, int, int, int, int, void *, void *);

struct block_mxfp4 {
    uint8_t e;
    uint8_t qs[16];
};

struct alignas(16) StridedPtrH {
    void * p;
    int    stride;
};
static_assert(sizeof(StridedPtrH) == 16, "StridedPtrH must match turbomind StridedPtr");

struct Case {
    int tokens_per_active;
};

struct PackedExperts {
    std::vector<void *> d_w_active;
    std::vector<void *> d_s_active;
    void * d_w_table = nullptr;
    void * d_s_table = nullptr;
    int k_pack = 0;
};

struct DeviceSide {
    int device = 0;
    PackedExperts gated;
    PackedExperts down;
    int * d_offsets = nullptr;
    __half * d_A = nullptr;
    __half * d_gated = nullptr;
    __half * d_down = nullptr;
    cudaStream_t stream = nullptr;
};

enum class SplitTransport {
    Nccl,
    Peer,
};

static int env_int(const char *name, int fallback, int lo, int hi) {
    const char *v = std::getenv(name);
    if (!v || !v[0]) return fallback;
    char *end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (!end || *end != '\0' || parsed < lo || parsed > hi) {
        fprintf(stderr, "[tp_split_2gpu] ignoring invalid %s=%s\n", name, v);
        return fallback;
    }
    return (int)parsed;
}

static bool manual_peer_baseline_allowed() {
    const char *v = std::getenv("DS4_ALLOW_MANUAL_PEER_BASELINE");
    return v && (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
                 std::strcmp(v, "yes") == 0);
}

static SplitTransport split_transport_from_env() {
    const char *v = std::getenv("DS4_TP_SPLIT_TRANSPORT");
    if (!v || !v[0] || std::strcmp(v, "nccl") == 0) return SplitTransport::Nccl;
    if (std::strcmp(v, "peer") == 0) return SplitTransport::Peer;
    fprintf(stderr, "[tp_split_2gpu] ignoring invalid DS4_TP_SPLIT_TRANSPORT=%s\n", v);
    return SplitTransport::Nccl;
}

static const char * split_transport_name(SplitTransport transport) {
    return transport == SplitTransport::Nccl ? "nccl" : "peer";
}

static std::vector<Case> parse_cases_from_env() {
    const char *v = std::getenv("DS4_TP_SPLIT_CASES");
    if (!v || !v[0]) {
        return {{128}, {256}};
    }

    std::vector<Case> out;
    const char *p = v;
    while (*p) {
        char *end = nullptr;
        long parsed = std::strtol(p, &end, 10);
        if (end == p || parsed < 1 || parsed > 256) {
            fprintf(stderr, "[tp_split_2gpu] invalid DS4_TP_SPLIT_CASES=%s\n", v);
            std::exit(2);
        }
        out.push_back(Case{(int)parsed});
        p = end;
        if (*p == ',') {
            ++p;
        } else if (*p != '\0') {
            fprintf(stderr, "[tp_split_2gpu] invalid DS4_TP_SPLIT_CASES=%s\n", v);
            std::exit(2);
        }
    }
    return out;
}

static void make_mxfp4_fixture(std::vector<block_mxfp4> & blocks, int N, int K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> e_dist(116, 122);
    std::uniform_int_distribution<int> q_dist(0, 255);
    blocks.resize((size_t) N * (K / 32));
    for (block_mxfp4 & b : blocks) {
        b.e = (uint8_t) e_dist(rng);
        for (uint8_t & q : b.qs) {
            q = (uint8_t) q_dist(rng);
        }
    }
}

static void make_fused_interleaved_fixture(std::vector<block_mxfp4> & fused,
                                           const std::vector<block_mxfp4> & gate,
                                           const std::vector<block_mxfp4> & up,
                                           int N,
                                           int K) {
    const int blocks_per_row = K / 32;
    fused.resize((size_t) 2 * N * blocks_per_row);
    for (int row = 0; row < N; ++row) {
        const size_t src = (size_t) row * blocks_per_row;
        const size_t gate_dst = (size_t) (2 * row) * blocks_per_row;
        const size_t up_dst = (size_t) (2 * row + 1) * blocks_per_row;
        std::copy(gate.begin() + src, gate.begin() + src + blocks_per_row, fused.begin() + gate_dst);
        std::copy(up.begin()   + src, up.begin()   + src + blocks_per_row, fused.begin() + up_dst);
    }
}

static void slice_rows_fixture(std::vector<block_mxfp4> & out,
                               const std::vector<block_mxfp4> & src,
                               int K,
                               int row_begin,
                               int row_count) {
    const int blocks_per_row = K / 32;
    out.resize((size_t) row_count * blocks_per_row);
    for (int row = 0; row < row_count; ++row) {
        const size_t src_off = (size_t) (row_begin + row) * blocks_per_row;
        const size_t dst_off = (size_t) row * blocks_per_row;
        std::copy(src.begin() + src_off,
                  src.begin() + src_off + blocks_per_row,
                  out.begin() + dst_off);
    }
}

static void slice_cols_fixture(std::vector<block_mxfp4> & out,
                               const std::vector<block_mxfp4> & src,
                               int N,
                               int src_K,
                               int col_begin,
                               int col_count) {
    const int src_blocks_per_row = src_K / 32;
    const int dst_blocks_per_row = col_count / 32;
    const int col_block_begin = col_begin / 32;
    out.resize((size_t) N * dst_blocks_per_row);
    for (int row = 0; row < N; ++row) {
        const size_t src_off = (size_t) row * src_blocks_per_row + col_block_begin;
        const size_t dst_off = (size_t) row * dst_blocks_per_row;
        std::copy(src.begin() + src_off,
                  src.begin() + src_off + dst_blocks_per_row,
                  out.begin() + dst_off);
    }
}

static void free_packed(PackedExperts & p) {
    for (void * v : p.d_w_active) {
        if (v) CHECK_CUDA(cudaFree(v));
    }
    for (void * v : p.d_s_active) {
        if (v) CHECK_CUDA(cudaFree(v));
    }
    if (p.d_w_table) CHECK_CUDA(cudaFree(p.d_w_table));
    if (p.d_s_table) CHECK_CUDA(cudaFree(p.d_s_table));
    p = PackedExperts{};
}

static int pack_fixture_set(int device,
                            pfn_packed_bytes pb,
                            pfn_pack_weight pw,
                            int ggml_type,
                            int N,
                            int K,
                            int group_size,
                            int num_experts,
                            const std::vector<int> & active,
                            const std::vector<std::vector<block_mxfp4>> & fixtures,
                            PackedExperts & out) {
    CHECK_CUDA(cudaSetDevice(device));

    size_t wb = 0;
    size_t sb = 0;
    int rc = pb(ggml_type, N, K, group_size, &wb, &sb);
    if (rc != 0) {
        fprintf(stderr, "[tp_split_2gpu] packed_bytes N=%d K=%d rc=%d\n", N, K, rc);
        return 1;
    }

    out.d_w_active.assign(active.size(), nullptr);
    out.d_s_active.assign(active.size(), nullptr);

    for (size_t i = 0; i < active.size(); ++i) {
        void * d_src = nullptr;
        CHECK_CUDA(cudaMalloc(&d_src, fixtures[i].size() * sizeof(block_mxfp4)));
        CHECK_CUDA(cudaMemcpy(d_src, fixtures[i].data(),
                              fixtures[i].size() * sizeof(block_mxfp4),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&out.d_w_active[i], wb));
        CHECK_CUDA(cudaMalloc(&out.d_s_active[i], sb));

        int this_pack = 0;
        rc = pw(d_src, ggml_type, N, K, group_size,
                out.d_w_active[i], out.d_s_active[i], &this_pack, nullptr);
        CHECK_CUDA(cudaFree(d_src));
        if (rc != 0) {
            fprintf(stderr, "[tp_split_2gpu] pack expert=%d N=%d K=%d rc=%d\n",
                    active[i], N, K, rc);
            return 2;
        }
        if (i == 0) {
            out.k_pack = this_pack;
        } else if (this_pack != out.k_pack) {
            fprintf(stderr, "[tp_split_2gpu] inconsistent k_pack 0x%x vs 0x%x\n",
                    this_pack, out.k_pack);
            return 3;
        }
    }

    std::vector<StridedPtrH> h_w(num_experts);
    std::vector<StridedPtrH> h_s(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        h_w[e] = StridedPtrH{out.d_w_active[0], K * 32};
        h_s[e] = StridedPtrH{out.d_s_active[0], N};
    }
    for (size_t i = 0; i < active.size(); ++i) {
        h_w[active[i]] = StridedPtrH{out.d_w_active[i], K * 32};
        h_s[active[i]] = StridedPtrH{out.d_s_active[i], N};
    }

    CHECK_CUDA(cudaMalloc(&out.d_w_table, h_w.size() * sizeof(StridedPtrH)));
    CHECK_CUDA(cudaMalloc(&out.d_s_table, h_s.size() * sizeof(StridedPtrH)));
    CHECK_CUDA(cudaMemcpy(out.d_w_table, h_w.data(), h_w.size() * sizeof(StridedPtrH),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(out.d_s_table, h_s.data(), h_s.size() * sizeof(StridedPtrH),
                          cudaMemcpyHostToDevice));
    return 0;
}

static void enable_peer_pair(int a, int b) {
    int can_ab = 0;
    int can_ba = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can_ab, a, b));
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can_ba, b, a));
    if (!can_ab || !can_ba) {
        fprintf(stderr, "[tp_split_2gpu] peer access unavailable %d<->%d\n", a, b);
        std::exit(2);
    }
    CHECK_CUDA(cudaSetDevice(a));
    cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
    } else {
        CHECK_CUDA(e);
    }
    CHECK_CUDA(cudaSetDevice(b));
    e = cudaDeviceEnablePeerAccess(a, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
    } else {
        CHECK_CUDA(e);
    }
}

static int run_side(DeviceSide & side,
                    pfn_mul_mat_grouped_gated_silu_total_tokens mmgs,
                    pfn_mul_mat_grouped_total_tokens mmgt,
                    int total_tokens,
                    int num_experts,
                    int fused_N,
                    int hidden,
                    int mid_half,
                    int group_size) {
    CHECK_CUDA(cudaSetDevice(side.device));
    int rc = mmgs(side.d_A, nullptr, side.d_offsets, num_experts, total_tokens,
                  (const void * const *) side.gated.d_w_table,
                  (const void * const *) side.gated.d_s_table,
                  GGML_TM_DTYPE_MXFP4, fused_N, hidden, group_size,
                  side.gated.k_pack, side.d_gated, side.stream);
    if (rc != 0) return rc;
    rc = mmgt(side.d_gated, nullptr, side.d_offsets, num_experts, total_tokens,
              (const void * const *) side.down.d_w_table,
              (const void * const *) side.down.d_s_table,
              GGML_TM_DTYPE_MXFP4, hidden, mid_half, group_size,
              side.down.k_pack, side.d_down, side.stream);
    return rc;
}

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return (double)ms;
}

static void nccl_broadcast_input_2(ncclComm_t comms[2],
                                   const int devices[2],
                                   __half *send0,
                                   __half *recv0,
                                   __half *recv1,
                                   size_t bytes,
                                   cudaStream_t stream0,
                                   cudaStream_t stream1) {
    CHECK_NCCL(ncclGroupStart());
    CHECK_CUDA(cudaSetDevice(devices[0]));
    CHECK_NCCL(ncclBroadcast(send0, recv0, bytes, ncclChar, 0, comms[0], stream0));
    CHECK_CUDA(cudaSetDevice(devices[1]));
    CHECK_NCCL(ncclBroadcast(recv1, recv1, bytes, ncclChar, 0, comms[1], stream1));
    CHECK_NCCL(ncclGroupEnd());
    CHECK_CUDA(cudaSetDevice(devices[0]));
    CHECK_CUDA(cudaStreamSynchronize(stream0));
    CHECK_CUDA(cudaSetDevice(devices[1]));
    CHECK_CUDA(cudaStreamSynchronize(stream1));
}

static void nccl_gather_output_2(ncclComm_t comms[2],
                                 const int devices[2],
                                 __half *dst0,
                                 __half *src1,
                                 size_t bytes,
                                 cudaStream_t stream0,
                                 cudaStream_t stream1) {
    CHECK_NCCL(ncclGroupStart());
    CHECK_CUDA(cudaSetDevice(devices[0]));
    CHECK_NCCL(ncclRecv(dst0, bytes, ncclChar, 1, comms[0], stream0));
    CHECK_CUDA(cudaSetDevice(devices[1]));
    CHECK_NCCL(ncclSend(src1, bytes, ncclChar, 0, comms[1], stream1));
    CHECK_NCCL(ncclGroupEnd());
    CHECK_CUDA(cudaSetDevice(devices[0]));
    CHECK_CUDA(cudaStreamSynchronize(stream0));
    CHECK_CUDA(cudaSetDevice(devices[1]));
    CHECK_CUDA(cudaStreamSynchronize(stream1));
}

static int compare_tp_sum(const std::vector<__half> & full,
                          const std::vector<__half> & half0,
                          const std::vector<__half> & half1,
                          int tokens_per_active,
                          int total_tokens) {
    if (full.size() != half0.size() || full.size() != half1.size()) {
        fprintf(stderr,
                "[tp_split_2gpu tpa=%d] correctness FAIL size mismatch full=%zu half0=%zu half1=%zu\n",
                tokens_per_active, full.size(), half0.size(), half1.size());
        return 1;
    }

    double sum_abs = 0.0;
    double sum_ref = 0.0;
    float max_abs = 0.0f;
    int bad = 0;
    int nan = 0;
    constexpr float abs_tol = 16.0f;
    constexpr float rel_elem_tol = 0.05f;
    constexpr double rel_tol = 0.01;
    constexpr double bad_frac_tol = 0.001;

    for (size_t i = 0; i < full.size(); ++i) {
        const float ref = __half2float(full[i]);
        const float got = __half2float(half0[i]) + __half2float(half1[i]);
        if (!std::isfinite(ref) || !std::isfinite(got)) {
            nan++;
            bad++;
            continue;
        }
        const float diff = fabsf(got - ref);
        const float elem_tol = std::max(abs_tol, rel_elem_tol * fabsf(ref));
        max_abs = std::max(max_abs, diff);
        sum_abs += (double)diff;
        sum_ref += (double)fabsf(ref);
        if (diff > elem_tol) {
            bad++;
        }
    }

    const double rel = sum_ref > 0.0 ? sum_abs / sum_ref : 0.0;
    const double bad_frac = full.empty() ? 0.0 : (double)bad / (double)full.size();
    const bool fail = nan != 0 || rel > rel_tol || bad_frac > bad_frac_tol;
    fprintf(stderr,
            "[tp_split_2gpu tpa=%d] correctness total_routes=%d values=%zu max_abs=%.4e rel=%.4e bad=%d bad_frac=%.4e nan=%d status=%s\n",
            tokens_per_active,
            total_tokens,
            full.size(),
            max_abs,
            rel,
            bad,
            bad_frac,
            nan,
            fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}

static int run_case(void * lib, const Case & c) {
    auto in   = (pfn_init)      dlsym(lib, "ggml_turbomind_init");
    auto sh   = (pfn_shutdown)  dlsym(lib, "ggml_turbomind_shutdown");
    auto pb   = (pfn_packed_bytes) dlsym(lib, "ggml_turbomind_packed_bytes");
    auto pw   = (pfn_pack_weight)  dlsym(lib, "ggml_turbomind_pack_weight_expert");
    auto mmgt = (pfn_mul_mat_grouped_total_tokens)
        dlsym(lib, "ggml_turbomind_mul_mat_grouped_total_tokens");
    auto mmgs = (pfn_mul_mat_grouped_gated_silu_total_tokens)
        dlsym(lib, "ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens");
    if (!in || !sh || !pb || !pw || !mmgt || !mmgs) {
        fprintf(stderr, "[tp_split_2gpu] dlsym failed\n");
        return 1;
    }

    const int dev0 = env_int("DS4_TP_SPLIT_GPU0", 0, 0, 31);
    const int dev1 = env_int("DS4_TP_SPLIT_GPU1", 3, 0, 31);
    const int warmup_iters = env_int("DS4_TP_SPLIT_WARMUP_ITERS", 5, 0, 1000);
    const int bench_iters = env_int("DS4_TP_SPLIT_BENCH_ITERS", 50, 1, 10000);
    const SplitTransport transport = split_transport_from_env();
    if (transport == SplitTransport::Peer && !manual_peer_baseline_allowed()) {
        fprintf(stderr,
                "[tp_split_2gpu] manual peer-copy baseline transport requires "
                "DS4_ALLOW_MANUAL_PEER_BASELINE=1; default or set "
                "DS4_TP_SPLIT_TRANSPORT=nccl for the NCCL path\n");
        return 2;
    }

    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    if (dev0 >= dev_count || dev1 >= dev_count || dev0 == dev1) {
        fprintf(stderr, "[tp_split_2gpu] invalid device pair %d,%d visible=%d\n", dev0, dev1, dev_count);
        return 2;
    }

    CHECK_CUDA(cudaSetDevice(dev0));
    if (in(dev0) != 0) return 3;
    CHECK_CUDA(cudaSetDevice(dev1));
    if (in(dev1) != 0) return 3;
    if (transport == SplitTransport::Peer) {
        enable_peer_pair(dev0, dev1);
    }
    const int nccl_devices[2] = {dev0, dev1};
    ncclComm_t comms[2] = {nullptr, nullptr};
    if (transport == SplitTransport::Nccl) {
        CHECK_NCCL(ncclCommInitAll(comms, 2, nccl_devices));
    }

    constexpr int ggml_type = GGML_TM_DTYPE_MXFP4;
    constexpr int group_size = 32;
    constexpr int hidden = 4096;
    constexpr int mid = 2048;
    constexpr int mid_half = mid / 2;
    constexpr int fused_N = 2 * mid;
    constexpr int fused_half_N = 2 * mid_half;
    constexpr int num_experts = 6;
    const std::vector<int> active{0, 1, 2, 3, 4, 5};
    const int total_tokens = (int) active.size() * c.tokens_per_active;

    std::vector<std::vector<block_mxfp4>> gate(active.size());
    std::vector<std::vector<block_mxfp4>> up(active.size());
    std::vector<std::vector<block_mxfp4>> down(active.size());
    std::vector<std::vector<block_mxfp4>> gated_full(active.size());
    std::vector<std::vector<block_mxfp4>> gated_half0(active.size());
    std::vector<std::vector<block_mxfp4>> gated_half1(active.size());
    std::vector<std::vector<block_mxfp4>> down_half0(active.size());
    std::vector<std::vector<block_mxfp4>> down_half1(active.size());
    for (size_t i = 0; i < active.size(); ++i) {
        make_mxfp4_fixture(gate[i], mid, hidden, 0x47000000u + (uint32_t)i * 101u);
        make_mxfp4_fixture(up[i],   mid, hidden, 0x55000000u + (uint32_t)i * 131u);
        make_mxfp4_fixture(down[i], hidden, mid, 0x63000000u + (uint32_t)i * 137u);
        make_fused_interleaved_fixture(gated_full[i], gate[i], up[i], mid, hidden);

        std::vector<block_mxfp4> gate0;
        std::vector<block_mxfp4> gate1;
        std::vector<block_mxfp4> up0;
        std::vector<block_mxfp4> up1;
        slice_rows_fixture(gate0, gate[i], hidden, 0, mid_half);
        slice_rows_fixture(gate1, gate[i], hidden, mid_half, mid_half);
        slice_rows_fixture(up0, up[i], hidden, 0, mid_half);
        slice_rows_fixture(up1, up[i], hidden, mid_half, mid_half);
        make_fused_interleaved_fixture(gated_half0[i], gate0, up0, mid_half, hidden);
        make_fused_interleaved_fixture(gated_half1[i], gate1, up1, mid_half, hidden);
        slice_cols_fixture(down_half0[i], down[i], hidden, mid, 0, mid_half);
        slice_cols_fixture(down_half1[i], down[i], hidden, mid, mid_half, mid_half);
    }

    std::vector<int> h_offsets(num_experts + 1, 0);
    int running = 0;
    for (int e = 0; e < num_experts; ++e) {
        h_offsets[e] = running;
        running += c.tokens_per_active;
    }
    h_offsets[num_experts] = running;

    std::mt19937 rng(0xB5010000u + (uint32_t)c.tokens_per_active);
    std::uniform_real_distribution<float> ad(-0.1f, 0.1f);
    std::vector<__half> h_A((size_t) total_tokens * hidden);
    for (__half & v : h_A) {
        v = __float2half(ad(rng));
    }

    DeviceSide full{};
    full.device = dev0;
    full.gated = PackedExperts{};
    full.down = PackedExperts{};
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaStreamCreate(&full.stream));
    CHECK_CUDA(cudaMalloc(&full.d_offsets, h_offsets.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(full.d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&full.d_A, h_A.size() * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(full.d_A, h_A.data(), h_A.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&full.d_gated, (size_t) total_tokens * mid * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&full.d_down, (size_t) total_tokens * hidden * sizeof(__half)));
    if (pack_fixture_set(dev0, pb, pw, ggml_type, fused_N, hidden, group_size,
                         num_experts, active, gated_full, full.gated) != 0 ||
        pack_fixture_set(dev0, pb, pw, ggml_type, hidden, mid, group_size,
                         num_experts, active, down, full.down) != 0) {
        fprintf(stderr, "[tp_split_2gpu] full pack failed\n");
        return 4;
    }

    DeviceSide s0{};
    DeviceSide s1{};
    s0.device = dev0;
    s1.device = dev1;
    for (DeviceSide *side : {&s0, &s1}) {
        CHECK_CUDA(cudaSetDevice(side->device));
        CHECK_CUDA(cudaStreamCreate(&side->stream));
        CHECK_CUDA(cudaMalloc(&side->d_offsets, h_offsets.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(side->d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&side->d_A, h_A.size() * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&side->d_gated, (size_t) total_tokens * mid_half * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&side->d_down, (size_t) total_tokens * hidden * sizeof(__half)));
    }
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaMemcpy(s0.d_A, h_A.data(), h_A.size() * sizeof(__half), cudaMemcpyHostToDevice));
    if (transport == SplitTransport::Nccl) {
        nccl_broadcast_input_2(comms, nccl_devices, full.d_A, s0.d_A, s1.d_A,
                               h_A.size() * sizeof(__half), s0.stream, s1.stream);
    } else {
        CHECK_CUDA(cudaMemcpyPeer(s1.d_A, dev1, full.d_A, dev0, h_A.size() * sizeof(__half)));
    }

    if (pack_fixture_set(dev0, pb, pw, ggml_type, fused_half_N, hidden, group_size,
                         num_experts, active, gated_half0, s0.gated) != 0 ||
        pack_fixture_set(dev0, pb, pw, ggml_type, hidden, mid_half, group_size,
                         num_experts, active, down_half0, s0.down) != 0 ||
        pack_fixture_set(dev1, pb, pw, ggml_type, fused_half_N, hidden, group_size,
                         num_experts, active, gated_half1, s1.gated) != 0 ||
        pack_fixture_set(dev1, pb, pw, ggml_type, hidden, mid_half, group_size,
                         num_experts, active, down_half1, s1.down) != 0) {
        fprintf(stderr, "[tp_split_2gpu] half pack failed\n");
        return 5;
    }

    __half * d_recv = nullptr;
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaMalloc(&d_recv, (size_t) total_tokens * hidden * sizeof(__half)));

    for (int i = 0; i < warmup_iters; ++i) {
        int rc = run_side(full, mmgs, mmgt, total_tokens, num_experts, fused_N, hidden, mid, group_size);
        if (rc != 0) return 6;
        rc = run_side(s0, mmgs, mmgt, total_tokens, num_experts, fused_half_N, hidden, mid_half, group_size);
        if (rc != 0) return 6;
        rc = run_side(s1, mmgs, mmgt, total_tokens, num_experts, fused_half_N, hidden, mid_half, group_size);
        if (rc != 0) return 6;
    }
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t full_start = nullptr;
    cudaEvent_t full_stop = nullptr;
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaEventCreate(&full_start));
    CHECK_CUDA(cudaEventCreate(&full_stop));
    CHECK_CUDA(cudaEventRecord(full_start, full.stream));
    for (int i = 0; i < bench_iters; ++i) {
        int rc = run_side(full, mmgs, mmgt, total_tokens, num_experts, fused_N, hidden, mid, group_size);
        if (rc != 0) return 7;
    }
    CHECK_CUDA(cudaEventRecord(full_stop, full.stream));
    CHECK_CUDA(cudaEventSynchronize(full_stop));
    const double full_ms = elapsed_ms(full_start, full_stop) / (double)bench_iters;

    cudaEvent_t s0_start = nullptr;
    cudaEvent_t s0_stop = nullptr;
    cudaEvent_t s1_start = nullptr;
    cudaEvent_t s1_stop = nullptr;
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaEventCreate(&s0_start));
    CHECK_CUDA(cudaEventCreate(&s0_stop));
    CHECK_CUDA(cudaEventRecord(s0_start, s0.stream));
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaEventCreate(&s1_start));
    CHECK_CUDA(cudaEventCreate(&s1_stop));
    CHECK_CUDA(cudaEventRecord(s1_start, s1.stream));
    for (int i = 0; i < bench_iters; ++i) {
        int rc = run_side(s0, mmgs, mmgt, total_tokens, num_experts, fused_half_N, hidden, mid_half, group_size);
        if (rc != 0) return 8;
        rc = run_side(s1, mmgs, mmgt, total_tokens, num_experts, fused_half_N, hidden, mid_half, group_size);
        if (rc != 0) return 8;
    }
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaEventRecord(s0_stop, s0.stream));
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaEventRecord(s1_stop, s1.stream));
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaEventSynchronize(s0_stop));
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaEventSynchronize(s1_stop));
    const double s0_ms = elapsed_ms(s0_start, s0_stop) / (double)bench_iters;
    const double s1_ms = elapsed_ms(s1_start, s1_stop) / (double)bench_iters;
    const double concurrent_compute_ms = std::max(s0_ms, s1_ms);

    const size_t a_bytes = h_A.size() * sizeof(__half);
    const size_t out_bytes = (size_t) total_tokens * hidden * sizeof(__half);
    const auto wall_start = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        if (transport == SplitTransport::Nccl) {
            nccl_broadcast_input_2(comms, nccl_devices, full.d_A, s0.d_A, s1.d_A,
                                   a_bytes, s0.stream, s1.stream);
        } else {
            CHECK_CUDA(cudaSetDevice(dev1));
            CHECK_CUDA(cudaMemcpyPeer(s1.d_A, dev1, full.d_A, dev0, a_bytes));
        }
        int rc = run_side(s0, mmgs, mmgt, total_tokens, num_experts, fused_half_N, hidden, mid_half, group_size);
        if (rc != 0) return 9;
        rc = run_side(s1, mmgs, mmgt, total_tokens, num_experts, fused_half_N, hidden, mid_half, group_size);
        if (rc != 0) return 9;
        CHECK_CUDA(cudaSetDevice(dev0));
        CHECK_CUDA(cudaStreamSynchronize(s0.stream));
        CHECK_CUDA(cudaSetDevice(dev1));
        CHECK_CUDA(cudaStreamSynchronize(s1.stream));
        if (transport == SplitTransport::Nccl) {
            nccl_gather_output_2(comms, nccl_devices, d_recv, s1.d_down,
                                 out_bytes, s0.stream, s1.stream);
        } else {
            CHECK_CUDA(cudaSetDevice(dev0));
            CHECK_CUDA(cudaMemcpyPeer(d_recv, dev0, s1.d_down, dev1, out_bytes));
        }
    }
    const auto wall_stop = std::chrono::steady_clock::now();
    const double total_copy_ms =
        std::chrono::duration<double, std::milli>(wall_stop - wall_start).count() / (double)bench_iters;

    const double a_mib = (double)a_bytes / (1024.0 * 1024.0);
    const double out_mib = (double)out_bytes / (1024.0 * 1024.0);
    fprintf(stderr,
            "[tp_split_2gpu tpa=%d] gpu_pair=%d,%d routes=%d transport=%s full_ms=%.4f half0_ms=%.4f half1_ms=%.4f concurrent_compute_ms=%.4f compute_speedup=%.3fx total_with_copy_ms=%.4f total_with_copy_speedup=%.3fx input_payload_mib=%.2f output_payload_mib=%.2f\n",
            c.tokens_per_active,
            dev0,
            dev1,
            total_tokens,
            split_transport_name(transport),
            full_ms,
            s0_ms,
            s1_ms,
            concurrent_compute_ms,
            full_ms / concurrent_compute_ms,
            total_copy_ms,
            full_ms / total_copy_ms,
            a_mib,
            out_mib);

    std::vector<__half> h_full((size_t) total_tokens * hidden);
    std::vector<__half> h_half0((size_t) total_tokens * hidden);
    std::vector<__half> h_half1((size_t) total_tokens * hidden);
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaMemcpy(h_full.data(), full.d_down, h_full.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_half0.data(), s0.d_down, h_half0.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_half1.data(), d_recv, h_half1.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    const int correctness_rc = compare_tp_sum(h_full, h_half0, h_half1, c.tokens_per_active, total_tokens);

    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaEventDestroy(full_start));
    CHECK_CUDA(cudaEventDestroy(full_stop));
    CHECK_CUDA(cudaEventDestroy(s0_start));
    CHECK_CUDA(cudaEventDestroy(s0_stop));
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaEventDestroy(s1_start));
    CHECK_CUDA(cudaEventDestroy(s1_stop));
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaFree(d_recv));

    for (DeviceSide *side : {&full, &s0, &s1}) {
        CHECK_CUDA(cudaSetDevice(side->device));
        free_packed(side->gated);
        free_packed(side->down);
        if (side->d_offsets) CHECK_CUDA(cudaFree(side->d_offsets));
        if (side->d_A) CHECK_CUDA(cudaFree(side->d_A));
        if (side->d_gated) CHECK_CUDA(cudaFree(side->d_gated));
        if (side->d_down) CHECK_CUDA(cudaFree(side->d_down));
        if (side->stream) CHECK_CUDA(cudaStreamDestroy(side->stream));
    }
    if (transport == SplitTransport::Nccl) {
        CHECK_CUDA(cudaSetDevice(dev0));
        CHECK_NCCL(ncclCommDestroy(comms[0]));
        CHECK_CUDA(cudaSetDevice(dev1));
        CHECK_NCCL(ncclCommDestroy(comms[1]));
    }

    sh();
    return correctness_rc;
}

int main(int argc, char ** argv) {
    const char * lib_path = argc > 1 ? argv[1] : "./libggml-turbomind.so";
    void * lib = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    int failures = 0;
    for (const Case & c : parse_cases_from_env()) {
        failures += run_case(lib, c) != 0;
    }
    dlclose(lib);
    return failures ? 1 : 0;
}
