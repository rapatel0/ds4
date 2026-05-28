#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h"

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
typedef int (*pfn_packed_bytes)(int, int, int, int, size_t *, size_t *);
typedef int (*pfn_pack_weight)(const void *, int, int, int, int, void *, void *, int *, void *);
typedef int (*pfn_mmgt)(const void *, const int *, const int *, int, int,
                        const void * const *, const void * const *, int, int, int, int, int,
                        void *, void *);
typedef int (*pfn_mmgs)(const void *, const int *, const int *, int, int,
                        const void * const *, const void * const *, int, int, int, int, int,
                        void *, void *);

struct block_mxfp4 {
    uint8_t e;
    uint8_t qs[16];
};

struct alignas(16) StridedPtrH {
    void *p;
    int stride;
};
static_assert(sizeof(StridedPtrH) == 16, "StridedPtrH must match TurboMind ABI");

struct Api {
    pfn_init init = nullptr;
    pfn_shutdown shutdown = nullptr;
    pfn_packed_bytes packed_bytes = nullptr;
    pfn_pack_weight pack_weight = nullptr;
    pfn_mmgt mmgt = nullptr;
    pfn_mmgs mmgs = nullptr;
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
    __half *d_a = nullptr;
    __half *d_gated = nullptr;
    __half *d_down = nullptr;
    PackedExperts gated;
    PackedExperts down;
    cudaEvent_t start = nullptr;
    cudaEvent_t mid = nullptr;
    cudaEvent_t stop = nullptr;
};

struct Options {
    const char *lib_path = "./libggml-turbomind.so";
    int devices[kGpus] = {0, 1, 2, 3, 4, 5, 6, 7};
    int slots = 32;
    int top_k = 6;
    int warmup = 5;
    int iters = 30;
};

bool parse_int(const char *text, int *out) {
    if (!text || !*text) return false;
    char *end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || v < 0 || v > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = (int)v;
    return true;
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
        if (!parse_int(piece.c_str(), &dev)) return false;
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
                 "usage: %s [--lib PATH] [--devices 0,1,2,3,4,5,6,7]\n"
                 "       [--slots N] [--top-k N] [--warmup N] [--iters N]\n",
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
        } else if (std::strcmp(arg, "--devices") == 0) {
            if (!val || !parse_devices(val, opt->devices)) return false;
            ++i;
        } else if (std::strcmp(arg, "--slots") == 0) {
            if (!val || !parse_int(val, &opt->slots) || opt->slots <= 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--top-k") == 0) {
            if (!val || !parse_int(val, &opt->top_k) || opt->top_k <= 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--warmup") == 0) {
            if (!val || !parse_int(val, &opt->warmup)) return false;
            ++i;
        } else if (std::strcmp(arg, "--iters") == 0) {
            if (!val || !parse_int(val, &opt->iters) || opt->iters <= 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }
    return opt->top_k <= kActiveLocalExperts;
}

void load_api(void *lib, Api *api) {
    api->init = (pfn_init)dlsym(lib, "ggml_turbomind_init");
    api->shutdown = (pfn_shutdown)dlsym(lib, "ggml_turbomind_shutdown");
    api->packed_bytes = (pfn_packed_bytes)dlsym(lib, "ggml_turbomind_packed_bytes");
    api->pack_weight = (pfn_pack_weight)dlsym(lib, "ggml_turbomind_pack_weight_expert");
    api->mmgt = (pfn_mmgt)dlsym(lib, "ggml_turbomind_mul_mat_grouped_total_tokens");
    api->mmgs = (pfn_mmgs)dlsym(lib, "ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens");
    if (!api->init || !api->shutdown || !api->packed_bytes || !api->pack_weight ||
        !api->mmgt || !api->mmgs) {
        std::fprintf(stderr, "dlsym failed for required TurboMind ABI\n");
        std::exit(2);
    }
}

void enable_peer_access(const Options &opt) {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    for (int i = 0; i < kGpus; ++i) {
        if (opt.devices[i] >= device_count) {
            std::fprintf(stderr, "device %d outside visible count %d\n", opt.devices[i],
                         device_count);
            std::exit(2);
        }
    }
    for (int i = 0; i < kGpus; ++i) {
        CHECK_CUDA(cudaSetDevice(opt.devices[i]));
        for (int j = 0; j < kGpus; ++j) {
            if (i == j) continue;
            int can = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&can, opt.devices[i], opt.devices[j]));
            if (!can) {
                std::fprintf(stderr, "device %d cannot peer access %d\n", opt.devices[i],
                             opt.devices[j]);
                std::exit(2);
            }
            cudaError_t err = cudaDeviceEnablePeerAccess(opt.devices[j], 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                (void)cudaGetLastError();
            } else if (err != cudaSuccess) {
                std::fprintf(stderr, "peer enable %d -> %d failed: %s\n", opt.devices[i],
                             opt.devices[j], cudaGetErrorString(err));
                std::exit(2);
            }
        }
    }
}

void make_mxfp4_fixture(std::vector<block_mxfp4> &blocks, int n, int k, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> e_dist(116, 120);
    std::uniform_int_distribution<int> q_dist(0, 255);
    blocks.resize((size_t)n * (k / 32));
    for (block_mxfp4 &b : blocks) {
        b.e = (uint8_t)e_dist(rng);
        for (uint8_t &q : b.qs) q = (uint8_t)q_dist(rng);
    }
}

void make_fused_interleaved_fixture(std::vector<block_mxfp4> &fused,
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

int pack_fixture_set(int device, const Api &api, int n, int k,
                     const std::vector<int> &active,
                     const std::vector<std::vector<block_mxfp4>> &fixtures,
                     PackedExperts *out) {
    CHECK_CUDA(cudaSetDevice(device));
    size_t wb = 0;
    size_t sb = 0;
    int rc = api.packed_bytes(kDType, n, k, kGroupSize, &wb, &sb);
    if (rc != 0) {
        std::fprintf(stderr, "packed_bytes failed device=%d N=%d K=%d rc=%d\n",
                     device, n, k, rc);
        return 1;
    }

    out->d_w_active.assign(active.size(), nullptr);
    out->d_s_active.assign(active.size(), nullptr);
    for (size_t i = 0; i < active.size(); ++i) {
        void *d_src = nullptr;
        CHECK_CUDA(cudaMalloc(&d_src, fixtures[i].size() * sizeof(block_mxfp4)));
        CHECK_CUDA(cudaMemcpy(d_src, fixtures[i].data(),
                              fixtures[i].size() * sizeof(block_mxfp4),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&out->d_w_active[i], wb));
        CHECK_CUDA(cudaMalloc(&out->d_s_active[i], sb));
        int k_pack = 0;
        rc = api.pack_weight(d_src, kDType, n, k, kGroupSize, out->d_w_active[i],
                             out->d_s_active[i], &k_pack, nullptr);
        CHECK_CUDA(cudaFree(d_src));
        if (rc != 0) {
            std::fprintf(stderr, "pack_weight failed device=%d expert=%zu N=%d K=%d rc=%d\n",
                         device, i, n, k, rc);
            return 2;
        }
        if (i == 0) out->k_pack = k_pack;
        else if (out->k_pack != k_pack) return 3;
    }

    std::vector<StridedPtrH> w_table((size_t)kLocalExperts);
    std::vector<StridedPtrH> s_table((size_t)kLocalExperts);
    for (int e = 0; e < kLocalExperts; ++e) {
        w_table[(size_t)e] = StridedPtrH{out->d_w_active[0], k * 32};
        s_table[(size_t)e] = StridedPtrH{out->d_s_active[0], n};
    }
    for (size_t i = 0; i < active.size(); ++i) {
        w_table[(size_t)active[i]] = StridedPtrH{out->d_w_active[i], k * 32};
        s_table[(size_t)active[i]] = StridedPtrH{out->d_s_active[i], n};
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
    *routes = running;
    *active_experts = active;
    *max_routes_per_expert = max_routes;
}

} // namespace

int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        usage(argv[0]);
        return 2;
    }

    void *lib = dlopen(opt.lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        std::fprintf(stderr, "dlopen failed for %s: %s\n", opt.lib_path, dlerror());
        return 2;
    }
    Api api;
    load_api(lib, &api);
    enable_peer_access(opt);

    RankState ranks[kGpus];
    int aggregate_routes = 0;
    int min_routes = std::numeric_limits<int>::max();
    int max_routes = 0;

    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        r.rank = p;
        r.device = opt.devices[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (api.init(r.device) != 0) {
            std::fprintf(stderr, "ggml_turbomind_init failed on device %d\n", r.device);
            return 3;
        }
        CHECK_CUDA(cudaStreamCreate(&r.stream));
        CHECK_CUDA(cudaEventCreate(&r.start));
        CHECK_CUDA(cudaEventCreate(&r.mid));
        CHECK_CUDA(cudaEventCreate(&r.stop));

        std::vector<int> offsets;
        build_offsets_for_rank(p, opt.slots, opt.top_k, &offsets, &r.routes,
                               &r.active_experts, &r.max_routes_per_expert);
        aggregate_routes += r.routes;
        min_routes = std::min(min_routes, r.routes);
        max_routes = std::max(max_routes, r.routes);

        const size_t a_elems = (size_t)r.routes * kHidden;
        CHECK_CUDA(cudaMalloc(&r.d_offsets, offsets.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(r.d_offsets, offsets.data(), offsets.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_a, a_elems * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&r.d_gated, (size_t)r.routes * kMid * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&r.d_down, a_elems * sizeof(__half)));

        std::mt19937 rng(0xE0310000u + (uint32_t)p * 97u);
        std::uniform_real_distribution<float> dist(-0.003f, 0.003f);
        std::vector<__half> h_a(a_elems);
        for (__half &v : h_a) v = __float2half(dist(rng));
        CHECK_CUDA(cudaMemcpy(r.d_a, h_a.data(), a_elems * sizeof(__half),
                              cudaMemcpyHostToDevice));

        std::vector<int> active;
        for (int e = 0; e < kActiveLocalExperts; ++e) active.push_back(e);
        std::vector<std::vector<block_mxfp4>> gated(active.size());
        std::vector<std::vector<block_mxfp4>> down(active.size());
        for (size_t i = 0; i < active.size(); ++i) {
            std::vector<block_mxfp4> gate;
            std::vector<block_mxfp4> up;
            make_mxfp4_fixture(gate, kMid, kHidden,
                               0x51000000u + (uint32_t)p * 1009u + (uint32_t)i * 37u);
            make_mxfp4_fixture(up, kMid, kHidden,
                               0x52000000u + (uint32_t)p * 1009u + (uint32_t)i * 41u);
            make_fused_interleaved_fixture(gated[i], gate, up, kMid, kHidden);
            make_mxfp4_fixture(down[i], kHidden, kMid,
                               0x53000000u + (uint32_t)p * 1009u + (uint32_t)i * 43u);
        }
        if (pack_fixture_set(r.device, api, kFusedN, kHidden, active, gated, &r.gated) != 0 ||
            pack_fixture_set(r.device, api, kHidden, kMid, active, down, &r.down) != 0) {
            return 4;
        }
    }

    for (int i = 0; i < opt.warmup; ++i) {
        for (int p = 0; p < kGpus; ++p) {
            if (run_gate(ranks[p], api) != 0 || run_down(ranks[p], api) != 0) return 5;
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
            if (run_gate(ranks[p], api) != 0) return 6;
        }
    }
    for (int p = 0; p < kGpus; ++p) {
        CHECK_CUDA(cudaSetDevice(ranks[p].device));
        CHECK_CUDA(cudaEventRecord(ranks[p].mid, ranks[p].stream));
    }
    for (int i = 0; i < opt.iters; ++i) {
        for (int p = 0; p < kGpus; ++p) {
            if (run_down(ranks[p], api) != 0) return 7;
        }
    }
    for (int p = 0; p < kGpus; ++p) {
        CHECK_CUDA(cudaSetDevice(ranks[p].device));
        CHECK_CUDA(cudaEventRecord(ranks[p].stop, ranks[p].stream));
    }

    double worst_gate_ms = 0.0;
    double worst_down_ms = 0.0;
    double worst_total_ms = 0.0;
    for (int p = 0; p < kGpus; ++p) {
        CHECK_CUDA(cudaSetDevice(ranks[p].device));
        CHECK_CUDA(cudaEventSynchronize(ranks[p].stop));
        const double gate_ms = (double)elapsed_ms(ranks[p].start, ranks[p].mid) / opt.iters;
        const double down_ms = (double)elapsed_ms(ranks[p].mid, ranks[p].stop) / opt.iters;
        worst_gate_ms = std::max(worst_gate_ms, gate_ms);
        worst_down_ms = std::max(worst_down_ms, down_ms);
        worst_total_ms = std::max(worst_total_ms, gate_ms + down_ms);
        std::printf("rank\t%d\tdevice\t%d\troutes\t%d\tactive_local_experts\t%d\t"
                    "max_routes_per_expert\t%d\tgate_ms\t%.6f\tdown_ms\t%.6f\t"
                    "total_ms\t%.6f\n",
                    p, ranks[p].device, ranks[p].routes, ranks[p].active_experts,
                    ranks[p].max_routes_per_expert, gate_ms, down_ms, gate_ms + down_ms);
    }

    double repeat_max_abs = 0.0;
    int repeat_bad = 0;
    int repeat_nan = 0;
    for (int p = 0; p < kGpus; ++p) {
        if (check_repeat(ranks[p], api, &repeat_max_abs, &repeat_bad, &repeat_nan) != 0) {
            return 8;
        }
    }

    const uint64_t dispatch_bytes = (uint64_t)aggregate_routes * kHidden * sizeof(__half);
    const uint64_t return_bytes = dispatch_bytes;
    const double imbalance = min_routes > 0 ? (double)max_routes / (double)min_routes : 0.0;
    std::printf("tp_ep_expert_smoke\tslots\t%d\ttop_k\t%d\taggregate_routes\t%d\t"
                "global_experts\t%d\tlocal_experts\t%d\tactive_local_experts\t%d\t"
                "dispatch_bytes\t%llu\treturn_bytes\t%llu\troute_imbalance\t%.6f\t"
                "worst_gate_ms\t%.6f\tworst_down_ms\t%.6f\tworst_total_ms\t%.6f\t"
                "repeat_max_abs\t%.9f\trepeat_bad\t%d\trepeat_nan\t%d\t%s\n",
                opt.slots, opt.top_k, aggregate_routes, kGlobalExperts, kLocalExperts,
                kActiveLocalExperts, (unsigned long long)dispatch_bytes,
                (unsigned long long)return_bytes, imbalance, worst_gate_ms, worst_down_ms,
                worst_total_ms, repeat_max_abs, repeat_bad, repeat_nan,
                (repeat_bad == 0 && repeat_nan == 0) ? "PASS" : "FAIL");

    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        free_packed(r.gated);
        free_packed(r.down);
        CHECK_CUDA(cudaFree(r.d_offsets));
        CHECK_CUDA(cudaFree(r.d_a));
        CHECK_CUDA(cudaFree(r.d_gated));
        CHECK_CUDA(cudaFree(r.d_down));
        CHECK_CUDA(cudaEventDestroy(r.start));
        CHECK_CUDA(cudaEventDestroy(r.mid));
        CHECK_CUDA(cudaEventDestroy(r.stop));
        CHECK_CUDA(cudaStreamDestroy(r.stream));
    }
    api.shutdown();
    dlclose(lib);
    return (repeat_bad == 0 && repeat_nan == 0) ? 0 : 1;
}
