// Four-GPU tensor-parallel routed-FFN proxy for DS4/V100.
//
// This benchmark is a compute-envelope gate for a future full-layer TP4/EP
// topology. It does not model a production routed-only overlay; it measures
// whether splitting the routed FFN middle dimension four ways gives enough
// real TurboMind MXFP4 compute speedup to justify the TP4 boundary measured by
// the layer proxy.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <nccl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml-turbomind-api.h"

#define CHECK_CUDA(x) do { cudaError_t err__ = (x); if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    std::exit(1); \
} } while (0)

#define CHECK_NCCL_SPLIT(x) do { ncclResult_t err__ = (x); if (err__ != ncclSuccess) { \
    fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(err__)); \
    std::exit(1); \
} } while (0)

namespace {

constexpr int kParts = 4;
constexpr int kHidden = 4096;
constexpr int kMid = 2048;
constexpr int kMidPart = kMid / kParts;
constexpr int kFusedN = 2 * kMid;
constexpr int kFusedPartN = 2 * kMidPart;
constexpr int kNumExperts = 6;
constexpr int kGroupSize = 32;
constexpr int kGgmlType = GGML_TM_DTYPE_MXFP4;

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

struct Api {
    pfn_init init = nullptr;
    pfn_shutdown shutdown = nullptr;
    pfn_packed_bytes packed_bytes = nullptr;
    pfn_pack_weight pack_weight = nullptr;
    pfn_mul_mat_grouped_total_tokens mmgt = nullptr;
    pfn_mul_mat_grouped_gated_silu_total_tokens mmgs = nullptr;
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
        fprintf(stderr, "[tp_split_4gpu] ignoring invalid %s=%s\n", name, v);
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
    fprintf(stderr, "[tp_split_4gpu] ignoring invalid DS4_TP_SPLIT_TRANSPORT=%s\n", v);
    return SplitTransport::Nccl;
}

static const char * split_transport_name(SplitTransport transport) {
    return transport == SplitTransport::Nccl ? "nccl" : "peer";
}

static std::vector<Case> parse_cases_from_env() {
    const char *v = std::getenv("DS4_TP_SPLIT_CASES");
    if (!v || !v[0]) {
        return {{1}, {16}, {128}};
    }

    std::vector<Case> out;
    const char *p = v;
    while (*p) {
        char *end = nullptr;
        long parsed = std::strtol(p, &end, 10);
        if (end == p || parsed < 1 || parsed > 512) {
            fprintf(stderr, "[tp_split_4gpu] invalid DS4_TP_SPLIT_CASES=%s\n", v);
            std::exit(2);
        }
        out.push_back(Case{(int)parsed});
        p = end;
        if (*p == ',') {
            ++p;
        } else if (*p != '\0') {
            fprintf(stderr, "[tp_split_4gpu] invalid DS4_TP_SPLIT_CASES=%s\n", v);
            std::exit(2);
        }
    }
    return out;
}

static std::array<int, kParts> parse_devices_from_env() {
    std::array<int, kParts> devices = {
        env_int("DS4_TP_SPLIT_GPU0", 0, 0, 31),
        env_int("DS4_TP_SPLIT_GPU1", 1, 0, 31),
        env_int("DS4_TP_SPLIT_GPU2", 2, 0, 31),
        env_int("DS4_TP_SPLIT_GPU3", 3, 0, 31),
    };
    const char *v = std::getenv("DS4_TP_SPLIT4_GPUS");
    if (v && v[0]) {
        const char *p = v;
        for (int i = 0; i < kParts; ++i) {
            char *end = nullptr;
            long parsed = std::strtol(p, &end, 10);
            if (end == p || parsed < 0 || parsed > 31) {
                fprintf(stderr, "[tp_split_4gpu] invalid DS4_TP_SPLIT4_GPUS=%s\n", v);
                std::exit(2);
            }
            devices[i] = (int)parsed;
            p = end;
            if (i + 1 < kParts) {
                if (*p != ',') {
                    fprintf(stderr, "[tp_split_4gpu] invalid DS4_TP_SPLIT4_GPUS=%s\n", v);
                    std::exit(2);
                }
                ++p;
            } else if (*p != '\0') {
                fprintf(stderr, "[tp_split_4gpu] invalid DS4_TP_SPLIT4_GPUS=%s\n", v);
                std::exit(2);
            }
        }
    }
    for (int i = 0; i < kParts; ++i) {
        for (int j = i + 1; j < kParts; ++j) {
            if (devices[i] == devices[j]) {
                fprintf(stderr, "[tp_split_4gpu] duplicate device %d\n", devices[i]);
                std::exit(2);
            }
        }
    }
    return devices;
}

static void nccl_broadcast_input_4(ncclComm_t comms[kParts],
                                   const std::array<int, kParts> & devices,
                                   __half *send0,
                                   std::array<DeviceSide, kParts> & sides,
                                   size_t bytes) {
    CHECK_NCCL_SPLIT(ncclGroupStart());
    for (int p = 0; p < kParts; ++p) {
        CHECK_CUDA(cudaSetDevice(devices[p]));
        CHECK_NCCL_SPLIT(ncclBroadcast(p == 0 ? send0 : sides[p].d_A,
                                       sides[p].d_A,
                                       bytes,
                                       ncclChar,
                                       0,
                                       comms[p],
                                       sides[p].stream));
    }
    CHECK_NCCL_SPLIT(ncclGroupEnd());
    for (int p = 0; p < kParts; ++p) {
        CHECK_CUDA(cudaSetDevice(devices[p]));
        CHECK_CUDA(cudaStreamSynchronize(sides[p].stream));
    }
}

static void nccl_gather_outputs_4(ncclComm_t comms[kParts],
                                  const std::array<int, kParts> & devices,
                                  std::array<DeviceSide, kParts> & sides,
                                  std::array<__half *, kParts> & d_recv,
                                  size_t bytes) {
    CHECK_NCCL_SPLIT(ncclGroupStart());
    CHECK_CUDA(cudaSetDevice(devices[0]));
    for (int p = 1; p < kParts; ++p) {
        CHECK_NCCL_SPLIT(ncclRecv(d_recv[p], bytes, ncclChar, p,
                                  comms[0], sides[0].stream));
    }
    for (int p = 1; p < kParts; ++p) {
        CHECK_CUDA(cudaSetDevice(devices[p]));
        CHECK_NCCL_SPLIT(ncclSend(sides[p].d_down, bytes, ncclChar, 0,
                                  comms[p], sides[p].stream));
    }
    CHECK_NCCL_SPLIT(ncclGroupEnd());
    for (int p = 0; p < kParts; ++p) {
        CHECK_CUDA(cudaSetDevice(devices[p]));
        CHECK_CUDA(cudaStreamSynchronize(sides[p].stream));
    }
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
                            const Api & api,
                            int N,
                            int K,
                            const std::vector<int> & active,
                            const std::vector<std::vector<block_mxfp4>> & fixtures,
                            PackedExperts & out) {
    CHECK_CUDA(cudaSetDevice(device));

    size_t wb = 0;
    size_t sb = 0;
    int rc = api.packed_bytes(kGgmlType, N, K, kGroupSize, &wb, &sb);
    if (rc != 0) {
        fprintf(stderr, "[tp_split_4gpu] packed_bytes N=%d K=%d rc=%d\n", N, K, rc);
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
        rc = api.pack_weight(d_src, kGgmlType, N, K, kGroupSize,
                             out.d_w_active[i], out.d_s_active[i], &this_pack, nullptr);
        CHECK_CUDA(cudaFree(d_src));
        if (rc != 0) {
            fprintf(stderr, "[tp_split_4gpu] pack expert=%d N=%d K=%d rc=%d\n",
                    active[i], N, K, rc);
            return 2;
        }
        if (i == 0) {
            out.k_pack = this_pack;
        } else if (this_pack != out.k_pack) {
            fprintf(stderr, "[tp_split_4gpu] inconsistent k_pack 0x%x vs 0x%x\n",
                    this_pack, out.k_pack);
            return 3;
        }
    }

    std::vector<StridedPtrH> h_w(kNumExperts);
    std::vector<StridedPtrH> h_s(kNumExperts);
    for (int e = 0; e < kNumExperts; ++e) {
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

static void enable_peer_all(const std::array<int, kParts> & devices) {
    for (int i = 0; i < kParts; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        for (int j = 0; j < kParts; ++j) {
            if (i == j) continue;
            int can = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&can, devices[i], devices[j]));
            if (!can) {
                fprintf(stderr, "[tp_split_4gpu] peer access unavailable %d->%d\n",
                        devices[i], devices[j]);
                std::exit(2);
            }
            cudaError_t e = cudaDeviceEnablePeerAccess(devices[j], 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) {
                (void)cudaGetLastError();
            } else {
                CHECK_CUDA(e);
            }
        }
    }
}

static int run_side(DeviceSide & side,
                    const Api & api,
                    int total_tokens,
                    int fused_N,
                    int mid_width) {
    CHECK_CUDA(cudaSetDevice(side.device));
    int rc = api.mmgs(side.d_A, nullptr, side.d_offsets, kNumExperts, total_tokens,
                      (const void * const *) side.gated.d_w_table,
                      (const void * const *) side.gated.d_s_table,
                      kGgmlType, fused_N, kHidden, kGroupSize,
                      side.gated.k_pack, side.d_gated, side.stream);
    if (rc != 0) return rc;
    rc = api.mmgt(side.d_gated, nullptr, side.d_offsets, kNumExperts, total_tokens,
                  (const void * const *) side.down.d_w_table,
                  (const void * const *) side.down.d_s_table,
                  kGgmlType, kHidden, mid_width, kGroupSize,
                  side.down.k_pack, side.d_down, side.stream);
    return rc;
}

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return (double)ms;
}

static int compare_tp_sum(const std::vector<__half> & full,
                          const std::array<std::vector<__half>, kParts> & parts,
                          int tokens_per_active,
                          int total_tokens) {
    double sum_abs = 0.0;
    double sum_ref = 0.0;
    float max_abs = 0.0f;
    int bad = 0;
    int nan = 0;
    constexpr float abs_tol = 16.0f;
    constexpr float rel_elem_tol = 0.05f;
    constexpr double rel_tol = 0.01;
    constexpr double bad_frac_tol = 0.001;

    for (const auto & part : parts) {
        if (part.size() != full.size()) {
            fprintf(stderr,
                    "[tp_split_4gpu tpa=%d] correctness FAIL size mismatch full=%zu part=%zu\n",
                    tokens_per_active, full.size(), part.size());
            return 1;
        }
    }

    for (size_t i = 0; i < full.size(); ++i) {
        const float ref = __half2float(full[i]);
        float got = 0.0f;
        for (int p = 0; p < kParts; ++p) {
            got += __half2float(parts[p][i]);
        }
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
            "[tp_split_4gpu tpa=%d] correctness total_routes=%d values=%zu max_abs=%.4e rel=%.4e bad=%d bad_frac=%.4e nan=%d status=%s\n",
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

static Api load_api(void * lib) {
    Api api;
    api.init = (pfn_init)dlsym(lib, "ggml_turbomind_init");
    api.shutdown = (pfn_shutdown)dlsym(lib, "ggml_turbomind_shutdown");
    api.packed_bytes = (pfn_packed_bytes)dlsym(lib, "ggml_turbomind_packed_bytes");
    api.pack_weight = (pfn_pack_weight)dlsym(lib, "ggml_turbomind_pack_weight_expert");
    api.mmgt = (pfn_mul_mat_grouped_total_tokens)
        dlsym(lib, "ggml_turbomind_mul_mat_grouped_total_tokens");
    api.mmgs = (pfn_mul_mat_grouped_gated_silu_total_tokens)
        dlsym(lib, "ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens");
    if (!api.init || !api.shutdown || !api.packed_bytes || !api.pack_weight ||
        !api.mmgt || !api.mmgs) {
        fprintf(stderr, "[tp_split_4gpu] dlsym failed\n");
        std::exit(1);
    }
    return api;
}

static int run_case(void * lib, const Case & c) {
    const Api api = load_api(lib);
    const std::array<int, kParts> devices = parse_devices_from_env();
    const int warmup_iters = env_int("DS4_TP_SPLIT_WARMUP_ITERS", 5, 0, 1000);
    const int bench_iters = env_int("DS4_TP_SPLIT_BENCH_ITERS", 50, 1, 10000);
    const bool verbose = env_int("DS4_TP_SPLIT_VERBOSE", 0, 0, 1) != 0;
    const SplitTransport transport = split_transport_from_env();
    if (transport == SplitTransport::Peer && !manual_peer_baseline_allowed()) {
        fprintf(stderr,
                "[tp_split_4gpu] manual peer-copy baseline transport requires "
                "DS4_ALLOW_MANUAL_PEER_BASELINE=1; default or set "
                "DS4_TP_SPLIT_TRANSPORT=nccl for the NCCL path\n");
        return 2;
    }

    if (verbose) {
        fprintf(stderr,
                "[tp_split_4gpu] start tpa=%d gpus=%d,%d,%d,%d transport=%s warmup=%d iters=%d\n",
                c.tokens_per_active,
                devices[0], devices[1], devices[2], devices[3],
                split_transport_name(transport),
                warmup_iters, bench_iters);
        fflush(stderr);
    }

    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    for (int p = 0; p < kParts; ++p) {
        if (devices[p] >= dev_count) {
            fprintf(stderr, "[tp_split_4gpu] invalid device %d visible=%d\n",
                    devices[p], dev_count);
            return 2;
        }
        CHECK_CUDA(cudaSetDevice(devices[p]));
        if (api.init(devices[p]) != 0) return 3;
    }
    if (transport == SplitTransport::Peer) {
        enable_peer_all(devices);
    }
    ncclComm_t comms[kParts] = {};
    if (transport == SplitTransport::Nccl) {
        CHECK_NCCL_SPLIT(ncclCommInitAll(comms, kParts, devices.data()));
    }
    if (verbose) {
        fprintf(stderr, "[tp_split_4gpu] initialized tpa=%d\n", c.tokens_per_active);
        fflush(stderr);
    }

    const std::vector<int> active{0, 1, 2, 3, 4, 5};
    const int total_tokens = (int)active.size() * c.tokens_per_active;

    std::vector<std::vector<block_mxfp4>> gate(active.size());
    std::vector<std::vector<block_mxfp4>> up(active.size());
    std::vector<std::vector<block_mxfp4>> down(active.size());
    std::vector<std::vector<block_mxfp4>> gated_full(active.size());
    std::array<std::vector<std::vector<block_mxfp4>>, kParts> gated_part;
    std::array<std::vector<std::vector<block_mxfp4>>, kParts> down_part;
    for (int p = 0; p < kParts; ++p) {
        gated_part[p].resize(active.size());
        down_part[p].resize(active.size());
    }
    for (size_t i = 0; i < active.size(); ++i) {
        make_mxfp4_fixture(gate[i], kMid, kHidden, 0x47000000u + (uint32_t)i * 101u);
        make_mxfp4_fixture(up[i],   kMid, kHidden, 0x55000000u + (uint32_t)i * 131u);
        make_mxfp4_fixture(down[i], kHidden, kMid, 0x63000000u + (uint32_t)i * 137u);
        make_fused_interleaved_fixture(gated_full[i], gate[i], up[i], kMid, kHidden);
        for (int p = 0; p < kParts; ++p) {
            const int begin = p * kMidPart;
            std::vector<block_mxfp4> gate_slice;
            std::vector<block_mxfp4> up_slice;
            slice_rows_fixture(gate_slice, gate[i], kHidden, begin, kMidPart);
            slice_rows_fixture(up_slice, up[i], kHidden, begin, kMidPart);
            make_fused_interleaved_fixture(gated_part[p][i], gate_slice, up_slice,
                                           kMidPart, kHidden);
            slice_cols_fixture(down_part[p][i], down[i], kHidden, kMid,
                               begin, kMidPart);
        }
    }
    if (verbose) {
        fprintf(stderr, "[tp_split_4gpu] fixtures ready tpa=%d\n", c.tokens_per_active);
        fflush(stderr);
    }

    std::vector<int> h_offsets(kNumExperts + 1, 0);
    int running = 0;
    for (int e = 0; e < kNumExperts; ++e) {
        h_offsets[e] = running;
        running += c.tokens_per_active;
    }
    h_offsets[kNumExperts] = running;

    std::mt19937 rng(0xB5040000u + (uint32_t)c.tokens_per_active);
    std::uniform_real_distribution<float> ad(-0.1f, 0.1f);
    std::vector<__half> h_A((size_t) total_tokens * kHidden);
    for (__half & v : h_A) {
        v = __float2half(ad(rng));
    }

    DeviceSide full{};
    full.device = devices[0];
    CHECK_CUDA(cudaSetDevice(full.device));
    CHECK_CUDA(cudaStreamCreate(&full.stream));
    CHECK_CUDA(cudaMalloc(&full.d_offsets, h_offsets.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(full.d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&full.d_A, h_A.size() * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(full.d_A, h_A.data(), h_A.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&full.d_gated, (size_t) total_tokens * kMid * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&full.d_down, (size_t) total_tokens * kHidden * sizeof(__half)));
    if (pack_fixture_set(full.device, api, kFusedN, kHidden, active, gated_full, full.gated) != 0 ||
        pack_fixture_set(full.device, api, kHidden, kMid, active, down, full.down) != 0) {
        fprintf(stderr, "[tp_split_4gpu] full pack failed\n");
        return 4;
    }
    if (verbose) {
        fprintf(stderr, "[tp_split_4gpu] full pack ready tpa=%d\n", c.tokens_per_active);
        fflush(stderr);
    }

    std::array<DeviceSide, kParts> sides;
    for (int p = 0; p < kParts; ++p) {
        sides[p].device = devices[p];
        CHECK_CUDA(cudaSetDevice(sides[p].device));
        CHECK_CUDA(cudaStreamCreate(&sides[p].stream));
        CHECK_CUDA(cudaMalloc(&sides[p].d_offsets, h_offsets.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(sides[p].d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&sides[p].d_A, h_A.size() * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&sides[p].d_gated, (size_t) total_tokens * kMidPart * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&sides[p].d_down, (size_t) total_tokens * kHidden * sizeof(__half)));
        if (p == 0) {
            CHECK_CUDA(cudaMemcpy(sides[p].d_A, h_A.data(), h_A.size() * sizeof(__half), cudaMemcpyHostToDevice));
        } else if (transport == SplitTransport::Peer) {
            CHECK_CUDA(cudaMemcpyPeer(sides[p].d_A, sides[p].device, full.d_A, full.device,
                                      h_A.size() * sizeof(__half)));
        }
        if (p + 1 == kParts && transport == SplitTransport::Nccl) {
            nccl_broadcast_input_4(comms, devices, full.d_A, sides,
                                   h_A.size() * sizeof(__half));
        }
        if (pack_fixture_set(sides[p].device, api, kFusedPartN, kHidden,
                             active, gated_part[p], sides[p].gated) != 0 ||
            pack_fixture_set(sides[p].device, api, kHidden, kMidPart,
                             active, down_part[p], sides[p].down) != 0) {
            fprintf(stderr, "[tp_split_4gpu] part %d pack failed\n", p);
            return 5;
        }
    }
    if (verbose) {
        fprintf(stderr, "[tp_split_4gpu] shard packs ready tpa=%d\n", c.tokens_per_active);
        fflush(stderr);
    }

    std::array<__half *, kParts> d_recv{};
    for (int p = 1; p < kParts; ++p) {
        CHECK_CUDA(cudaSetDevice(full.device));
        CHECK_CUDA(cudaMalloc(&d_recv[p], (size_t) total_tokens * kHidden * sizeof(__half)));
    }

    for (int i = 0; i < warmup_iters; ++i) {
        if (run_side(full, api, total_tokens, kFusedN, kMid) != 0) return 6;
    }
    CHECK_CUDA(cudaSetDevice(full.device));
    CHECK_CUDA(cudaStreamSynchronize(full.stream));
    if (verbose) {
        fprintf(stderr, "[tp_split_4gpu] full warmup done tpa=%d\n", c.tokens_per_active);
        fflush(stderr);
    }
    for (int i = 0; i < warmup_iters; ++i) {
        for (int p = 0; p < kParts; ++p) {
            if (run_side(sides[p], api, total_tokens, kFusedPartN, kMidPart) != 0) return 6;
        }
    }
    for (int p = 0; p < kParts; ++p) {
        CHECK_CUDA(cudaSetDevice(devices[p]));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    if (verbose) {
        fprintf(stderr, "[tp_split_4gpu] warmup done tpa=%d\n", c.tokens_per_active);
        fflush(stderr);
    }

    cudaEvent_t full_start = nullptr;
    cudaEvent_t full_stop = nullptr;
    CHECK_CUDA(cudaSetDevice(full.device));
    CHECK_CUDA(cudaEventCreate(&full_start));
    CHECK_CUDA(cudaEventCreate(&full_stop));
    CHECK_CUDA(cudaEventRecord(full_start, full.stream));
    for (int i = 0; i < bench_iters; ++i) {
        if (run_side(full, api, total_tokens, kFusedN, kMid) != 0) return 7;
    }
    CHECK_CUDA(cudaEventRecord(full_stop, full.stream));
    CHECK_CUDA(cudaEventSynchronize(full_stop));
    const double full_ms = elapsed_ms(full_start, full_stop) / (double)bench_iters;
    if (verbose) {
        fprintf(stderr, "[tp_split_4gpu] full bench done tpa=%d full_ms=%.4f\n",
                c.tokens_per_active, full_ms);
        fflush(stderr);
    }

    std::array<cudaEvent_t, kParts> start{};
    std::array<cudaEvent_t, kParts> stop{};
    for (int p = 0; p < kParts; ++p) {
        CHECK_CUDA(cudaSetDevice(sides[p].device));
        CHECK_CUDA(cudaEventCreate(&start[p]));
        CHECK_CUDA(cudaEventCreate(&stop[p]));
        CHECK_CUDA(cudaEventRecord(start[p], sides[p].stream));
    }
    for (int i = 0; i < bench_iters; ++i) {
        for (int p = 0; p < kParts; ++p) {
            if (run_side(sides[p], api, total_tokens, kFusedPartN, kMidPart) != 0) return 8;
        }
    }
    std::array<double, kParts> part_ms{};
    double concurrent_compute_ms = 0.0;
    for (int p = 0; p < kParts; ++p) {
        CHECK_CUDA(cudaSetDevice(sides[p].device));
        CHECK_CUDA(cudaEventRecord(stop[p], sides[p].stream));
        CHECK_CUDA(cudaEventSynchronize(stop[p]));
        part_ms[p] = elapsed_ms(start[p], stop[p]) / (double)bench_iters;
        concurrent_compute_ms = std::max(concurrent_compute_ms, part_ms[p]);
    }
    if (verbose) {
        fprintf(stderr, "[tp_split_4gpu] shard bench done tpa=%d concurrent_ms=%.4f\n",
                c.tokens_per_active, concurrent_compute_ms);
        fflush(stderr);
    }

    const size_t a_bytes = h_A.size() * sizeof(__half);
    const size_t out_bytes = (size_t) total_tokens * kHidden * sizeof(__half);
    const auto wall_start = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        if (transport == SplitTransport::Nccl) {
            nccl_broadcast_input_4(comms, devices, full.d_A, sides, a_bytes);
        } else {
            for (int p = 1; p < kParts; ++p) {
                CHECK_CUDA(cudaMemcpyPeer(sides[p].d_A, sides[p].device, full.d_A, full.device,
                                          a_bytes));
            }
        }
        for (int p = 0; p < kParts; ++p) {
            if (run_side(sides[p], api, total_tokens, kFusedPartN, kMidPart) != 0) return 9;
        }
        for (int p = 0; p < kParts; ++p) {
            CHECK_CUDA(cudaSetDevice(sides[p].device));
            CHECK_CUDA(cudaStreamSynchronize(sides[p].stream));
        }
        if (transport == SplitTransport::Nccl) {
            nccl_gather_outputs_4(comms, devices, sides, d_recv, out_bytes);
        } else {
            for (int p = 1; p < kParts; ++p) {
                CHECK_CUDA(cudaMemcpyPeer(d_recv[p], full.device, sides[p].d_down,
                                          sides[p].device, out_bytes));
            }
        }
    }
    const auto wall_stop = std::chrono::steady_clock::now();
    const double total_copy_ms =
        std::chrono::duration<double, std::milli>(wall_stop - wall_start).count() / (double)bench_iters;
    if (verbose) {
        fprintf(stderr, "[tp_split_4gpu] copy-inclusive bench done tpa=%d total_ms=%.4f\n",
                c.tokens_per_active, total_copy_ms);
        fflush(stderr);
    }

    const double a_mib = (double)a_bytes * (double)(kParts - 1) / (1024.0 * 1024.0);
    const double out_mib = (double)out_bytes * (double)(kParts - 1) / (1024.0 * 1024.0);
    fprintf(stderr,
            "[tp_split_4gpu tpa=%d] gpus=%d,%d,%d,%d routes=%d transport=%s full_ms=%.4f part_ms=%.4f,%.4f,%.4f,%.4f concurrent_compute_ms=%.4f compute_speedup=%.3fx total_with_copy_ms=%.4f total_with_copy_speedup=%.3fx input_payload_mib=%.2f output_payload_mib=%.2f\n",
            c.tokens_per_active,
            devices[0], devices[1], devices[2], devices[3],
            total_tokens,
            split_transport_name(transport),
            full_ms,
            part_ms[0], part_ms[1], part_ms[2], part_ms[3],
            concurrent_compute_ms,
            full_ms / concurrent_compute_ms,
            total_copy_ms,
            full_ms / total_copy_ms,
            a_mib,
            out_mib);

    std::vector<__half> h_full((size_t) total_tokens * kHidden);
    std::array<std::vector<__half>, kParts> h_parts;
    for (int p = 0; p < kParts; ++p) {
        h_parts[p].resize((size_t) total_tokens * kHidden);
    }
    CHECK_CUDA(cudaSetDevice(full.device));
    CHECK_CUDA(cudaMemcpy(h_full.data(), full.d_down, h_full.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_parts[0].data(), sides[0].d_down, h_parts[0].size() * sizeof(__half), cudaMemcpyDeviceToHost));
    for (int p = 1; p < kParts; ++p) {
        CHECK_CUDA(cudaMemcpy(h_parts[p].data(), d_recv[p], h_parts[p].size() * sizeof(__half), cudaMemcpyDeviceToHost));
    }
    const int correctness_rc = compare_tp_sum(h_full, h_parts, c.tokens_per_active, total_tokens);

    CHECK_CUDA(cudaSetDevice(full.device));
    CHECK_CUDA(cudaEventDestroy(full_start));
    CHECK_CUDA(cudaEventDestroy(full_stop));
    for (int p = 0; p < kParts; ++p) {
        CHECK_CUDA(cudaSetDevice(sides[p].device));
        CHECK_CUDA(cudaEventDestroy(start[p]));
        CHECK_CUDA(cudaEventDestroy(stop[p]));
    }
    for (int p = 1; p < kParts; ++p) {
        CHECK_CUDA(cudaSetDevice(full.device));
        CHECK_CUDA(cudaFree(d_recv[p]));
    }

    CHECK_CUDA(cudaSetDevice(full.device));
    free_packed(full.gated);
    free_packed(full.down);
    if (full.d_offsets) CHECK_CUDA(cudaFree(full.d_offsets));
    if (full.d_A) CHECK_CUDA(cudaFree(full.d_A));
    if (full.d_gated) CHECK_CUDA(cudaFree(full.d_gated));
    if (full.d_down) CHECK_CUDA(cudaFree(full.d_down));
    if (full.stream) CHECK_CUDA(cudaStreamDestroy(full.stream));

    for (DeviceSide & side : sides) {
        CHECK_CUDA(cudaSetDevice(side.device));
        free_packed(side.gated);
        free_packed(side.down);
        if (side.d_offsets) CHECK_CUDA(cudaFree(side.d_offsets));
        if (side.d_A) CHECK_CUDA(cudaFree(side.d_A));
        if (side.d_gated) CHECK_CUDA(cudaFree(side.d_gated));
        if (side.d_down) CHECK_CUDA(cudaFree(side.d_down));
        if (side.stream) CHECK_CUDA(cudaStreamDestroy(side.stream));
    }
    if (transport == SplitTransport::Nccl) {
        for (int p = 0; p < kParts; ++p) {
            CHECK_CUDA(cudaSetDevice(devices[p]));
            CHECK_NCCL_SPLIT(ncclCommDestroy(comms[p]));
        }
    }

    api.shutdown();
    return correctness_rc;
}

} // namespace

#ifndef DS4_TP_SPLIT_4GPU_NO_MAIN
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
#endif
