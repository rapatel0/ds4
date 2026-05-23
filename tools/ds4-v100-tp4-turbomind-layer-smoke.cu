#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
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

constexpr int kParticipants = 4;
constexpr int kHidden = 4096;
constexpr int kFullMid = 2048;
constexpr int kMidShard = kFullMid / kParticipants;
constexpr int kFusedFullN = 2 * kFullMid;
constexpr int kFusedShardN = 2 * kMidShard;
constexpr int kExperts = 6;
constexpr int kGroupSize = 32;
constexpr int kDType = GGML_TM_DTYPE_MXFP4;

#define CHECK_CUDA(expr)                                                                          \
    do {                                                                                          \
        cudaError_t err__ = (expr);                                                               \
        if (err__ != cudaSuccess) {                                                               \
            std::fprintf(stderr, "cuda error %s:%d: %s\n", __FILE__, __LINE__,                  \
                         cudaGetErrorString(err__));                                              \
            std::exit(2);                                                                         \
        }                                                                                         \
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
    void * p;
    int stride;
};
static_assert(sizeof(StridedPtrH) == 16, "StridedPtrH must match TurboMind ABI");

struct PackedExperts {
    std::vector<void *> d_w_active;
    std::vector<void *> d_s_active;
    void * d_w_table = nullptr;
    void * d_s_table = nullptr;
    int k_pack = 0;
};

struct DeviceSide {
    int device = 0;
    cudaStream_t stream = nullptr;
    int * d_offsets = nullptr;
    __half * d_a = nullptr;
    __half * d_gated = nullptr;
    __half * d_down = nullptr;
    PackedExperts gated;
    PackedExperts down;
};

struct Options {
    const char * lib_path = "./libggml-turbomind.so";
    int devices[kParticipants] = {0, 1, 2, 3};
    int tokens_per_active = 32;
    int warmup = 5;
    int iters = 50;
};

struct Api {
    pfn_init init = nullptr;
    pfn_shutdown shutdown = nullptr;
    pfn_packed_bytes packed_bytes = nullptr;
    pfn_pack_weight pack_weight = nullptr;
    pfn_mmgt mmgt = nullptr;
    pfn_mmgs mmgs = nullptr;
};

__global__ void copy_half_to_float_kernel(float * dst, const __half * src, size_t elems) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elems) {
        dst[i] = __half2float(src[i]);
    }
}

__global__ void add_half_to_float_kernel(float * dst, const __half * src, size_t elems) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elems) {
        dst[i] += __half2float(src[i]);
    }
}

bool parse_int(const char * text, int * out) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char * end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || v < 0 || v > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = (int) v;
    return true;
}

bool parse_devices(const char * text, int devices[kParticipants]) {
    std::vector<int> parsed;
    const char * cur = text;
    while (cur && *cur) {
        const char * comma = std::strchr(cur, ',');
        std::string piece;
        if (comma) {
            piece.assign(cur, comma - cur);
            cur = comma + 1;
        } else {
            piece.assign(cur);
            cur = nullptr;
        }
        int dev = 0;
        if (!parse_int(piece.c_str(), &dev)) {
            return false;
        }
        parsed.push_back(dev);
    }
    if ((int) parsed.size() != kParticipants) {
        return false;
    }
    for (int i = 0; i < kParticipants; ++i) {
        for (int j = i + 1; j < kParticipants; ++j) {
            if (parsed[i] == parsed[j]) {
                return false;
            }
        }
        devices[i] = parsed[i];
    }
    return true;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s [--lib PATH] [--devices 0,1,2,3]\n"
                 "       [--tokens-per-active N] [--warmup N] [--iters N]\n",
                 argv0);
}

bool parse_args(int argc, char ** argv, Options * opt) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        const char * val = (i + 1 < argc) ? argv[i + 1] : nullptr;
        if (std::strcmp(arg, "--lib") == 0) {
            if (!val) {
                return false;
            }
            opt->lib_path = val;
            ++i;
        } else if (std::strcmp(arg, "--devices") == 0) {
            if (!val || !parse_devices(val, opt->devices)) {
                std::fprintf(stderr, "invalid --devices; expected four unique ids\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--tokens-per-active") == 0) {
            if (!val || !parse_int(val, &opt->tokens_per_active) ||
                opt->tokens_per_active <= 0) {
                std::fprintf(stderr, "invalid --tokens-per-active\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--warmup") == 0) {
            if (!val || !parse_int(val, &opt->warmup)) {
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--iters") == 0) {
            if (!val || !parse_int(val, &opt->iters) || opt->iters <= 0) {
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg);
            return false;
        }
    }
    return true;
}

void make_mxfp4_fixture(std::vector<block_mxfp4> & blocks, int n, int k, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> e_dist(116, 122);
    std::uniform_int_distribution<int> q_dist(0, 255);
    blocks.resize((size_t) n * (k / 32));
    for (block_mxfp4 & b : blocks) {
        b.e = (uint8_t) e_dist(rng);
        for (uint8_t & q : b.qs) {
            q = (uint8_t) q_dist(rng);
        }
    }
}

void make_fused_interleaved_fixture(std::vector<block_mxfp4> & fused,
                                    const std::vector<block_mxfp4> & gate,
                                    const std::vector<block_mxfp4> & up,
                                    int n, int k) {
    const int blocks_per_row = k / 32;
    fused.resize((size_t) 2 * n * blocks_per_row);
    for (int row = 0; row < n; ++row) {
        const size_t src = (size_t) row * blocks_per_row;
        const size_t gate_dst = (size_t) (2 * row) * blocks_per_row;
        const size_t up_dst = (size_t) (2 * row + 1) * blocks_per_row;
        std::copy(gate.begin() + src, gate.begin() + src + blocks_per_row,
                  fused.begin() + gate_dst);
        std::copy(up.begin() + src, up.begin() + src + blocks_per_row,
                  fused.begin() + up_dst);
    }
}

void slice_rows_fixture(std::vector<block_mxfp4> & out,
                        const std::vector<block_mxfp4> & src,
                        int k, int row_begin, int row_count) {
    const int blocks_per_row = k / 32;
    out.resize((size_t) row_count * blocks_per_row);
    for (int row = 0; row < row_count; ++row) {
        const size_t src_off = (size_t) (row_begin + row) * blocks_per_row;
        const size_t dst_off = (size_t) row * blocks_per_row;
        std::copy(src.begin() + src_off, src.begin() + src_off + blocks_per_row,
                  out.begin() + dst_off);
    }
}

void slice_cols_fixture(std::vector<block_mxfp4> & out,
                        const std::vector<block_mxfp4> & src,
                        int n, int src_k, int col_begin, int col_count) {
    const int src_blocks_per_row = src_k / 32;
    const int dst_blocks_per_row = col_count / 32;
    const int col_block_begin = col_begin / 32;
    out.resize((size_t) n * dst_blocks_per_row);
    for (int row = 0; row < n; ++row) {
        const size_t src_off = (size_t) row * src_blocks_per_row + col_block_begin;
        const size_t dst_off = (size_t) row * dst_blocks_per_row;
        std::copy(src.begin() + src_off, src.begin() + src_off + dst_blocks_per_row,
                  out.begin() + dst_off);
    }
}

void free_packed(PackedExperts & p) {
    for (void * v : p.d_w_active) {
        if (v) {
            CHECK_CUDA(cudaFree(v));
        }
    }
    for (void * v : p.d_s_active) {
        if (v) {
            CHECK_CUDA(cudaFree(v));
        }
    }
    if (p.d_w_table) {
        CHECK_CUDA(cudaFree(p.d_w_table));
    }
    if (p.d_s_table) {
        CHECK_CUDA(cudaFree(p.d_s_table));
    }
    p = PackedExperts{};
}

int pack_fixture_set(int device, const Api & api, int n, int k,
                     const std::vector<int> & active,
                     const std::vector<std::vector<block_mxfp4>> & fixtures,
                     PackedExperts * out) {
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
        void * d_src = nullptr;
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
        if (i == 0) {
            out->k_pack = k_pack;
        } else if (out->k_pack != k_pack) {
            std::fprintf(stderr, "inconsistent k_pack %d vs %d\n", out->k_pack, k_pack);
            return 3;
        }
    }

    std::vector<StridedPtrH> w_table((size_t) kExperts);
    std::vector<StridedPtrH> s_table((size_t) kExperts);
    for (int e = 0; e < kExperts; ++e) {
        w_table[(size_t) e] = StridedPtrH{out->d_w_active[0], k * 32};
        s_table[(size_t) e] = StridedPtrH{out->d_s_active[0], n};
    }
    for (size_t i = 0; i < active.size(); ++i) {
        w_table[(size_t) active[i]] = StridedPtrH{out->d_w_active[i], k * 32};
        s_table[(size_t) active[i]] = StridedPtrH{out->d_s_active[i], n};
    }
    CHECK_CUDA(cudaMalloc(&out->d_w_table, w_table.size() * sizeof(StridedPtrH)));
    CHECK_CUDA(cudaMemcpy(out->d_w_table, w_table.data(),
                          w_table.size() * sizeof(StridedPtrH), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&out->d_s_table, s_table.size() * sizeof(StridedPtrH)));
    CHECK_CUDA(cudaMemcpy(out->d_s_table, s_table.data(),
                          s_table.size() * sizeof(StridedPtrH), cudaMemcpyHostToDevice));
    return 0;
}

int run_side(DeviceSide & side, const Api & api, int total_routes,
             int fused_n, int mid_width) {
    CHECK_CUDA(cudaSetDevice(side.device));
    int rc = api.mmgs(side.d_a, nullptr, side.d_offsets, kExperts, total_routes,
                      (const void * const *) side.gated.d_w_table,
                      (const void * const *) side.gated.d_s_table,
                      kDType, fused_n, kHidden, kGroupSize, side.gated.k_pack,
                      side.d_gated, side.stream);
    if (rc != 0) {
        return rc;
    }
    rc = api.mmgt(side.d_gated, nullptr, side.d_offsets, kExperts, total_routes,
                  (const void * const *) side.down.d_w_table,
                  (const void * const *) side.down.d_s_table,
                  kDType, kHidden, mid_width, kGroupSize, side.down.k_pack,
                  side.d_down, side.stream);
    return rc;
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

void enable_peer_access(const Options & opt) {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    for (int i = 0; i < kParticipants; ++i) {
        if (opt.devices[i] >= device_count) {
            std::fprintf(stderr, "device %d outside visible count %d\n", opt.devices[i],
                         device_count);
            std::exit(2);
        }
    }
    for (int i = 0; i < kParticipants; ++i) {
        CHECK_CUDA(cudaSetDevice(opt.devices[i]));
        for (int j = 0; j < kParticipants; ++j) {
            if (i == j) {
                continue;
            }
            int can = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&can, opt.devices[i], opt.devices[j]));
            if (!can) {
                std::fprintf(stderr, "device %d cannot peer access %d\n", opt.devices[i],
                             opt.devices[j]);
                std::exit(2);
            }
            cudaError_t err = cudaDeviceEnablePeerAccess(opt.devices[j], 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                (void) cudaGetLastError();
            } else if (err != cudaSuccess) {
                std::fprintf(stderr, "peer enable %d -> %d failed: %s\n", opt.devices[i],
                             opt.devices[j], cudaGetErrorString(err));
                std::exit(2);
            }
        }
    }
}

void sync_device_or_die(int device, const char * label) {
    CHECK_CUDA(cudaSetDevice(device));
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cuda sync failed after %s on device %d: %s\n",
                     label, device, cudaGetErrorString(err));
        std::exit(2);
    }
}

double run_reduce_to_root(const Options & opt, DeviceSide sides[kParticipants],
                          float * d_sum, __half ** d_recv, int total_routes,
                          bool keep_result) {
    const size_t elems = (size_t) total_routes * kHidden;
    const size_t bytes = elems * sizeof(__half);
    const int root = opt.devices[0];
    const int block = 256;
    const int grid = (int) ((elems + block - 1) / block);
    const auto start = std::chrono::steady_clock::now();

    CHECK_CUDA(cudaSetDevice(root));
    (void) cudaGetLastError();
    copy_half_to_float_kernel<<<grid, block, 0, 0>>>(d_sum, sides[0].d_down, elems);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int p = 1; p < kParticipants; ++p) {
        CHECK_CUDA(cudaMemcpyPeer(d_recv[p], root, sides[p].d_down, opt.devices[p], bytes));
        add_half_to_float_kernel<<<grid, block, 0, 0>>>(d_sum, d_recv[p], elems);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    if (!keep_result) {
        CHECK_CUDA(cudaMemset(d_sum, 0, elems * sizeof(float)));
    }
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

int compare_tp_sum_host(const Options & opt, const DeviceSide & full,
                        DeviceSide sides[kParticipants], int total_routes) {
    const size_t elems = (size_t) total_routes * kHidden;
    std::vector<__half> h_full(elems);
    std::vector<std::vector<__half>> parts(kParticipants);
    for (int p = 0; p < kParticipants; ++p) {
        parts[p].resize(elems);
    }

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaMemcpy(h_full.data(), full.d_down, elems * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    for (int p = 0; p < kParticipants; ++p) {
        CHECK_CUDA(cudaSetDevice(opt.devices[p]));
        CHECK_CUDA(cudaMemcpy(parts[p].data(), sides[p].d_down, elems * sizeof(__half),
                              cudaMemcpyDeviceToHost));
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

    for (size_t i = 0; i < elems; ++i) {
        const float ref = __half2float(h_full[i]);
        float got = 0.0f;
        for (int p = 0; p < kParticipants; ++p) {
            got += __half2float(parts[p][i]);
        }
        if (!std::isfinite(ref) || !std::isfinite(got)) {
            ++nan;
            ++bad;
            continue;
        }
        const float diff = std::fabs(got - ref);
        const float elem_tol = std::max(abs_tol, rel_elem_tol * std::fabs(ref));
        max_abs = std::max(max_abs, diff);
        sum_abs += (double) diff;
        sum_ref += (double) std::fabs(ref);
        if (diff > elem_tol) {
            ++bad;
        }
    }

    const double rel = sum_ref > 0.0 ? sum_abs / sum_ref : 0.0;
    const double bad_frac = elems ? (double) bad / (double) elems : 0.0;
    const bool fail = nan != 0 || rel > rel_tol || bad_frac > bad_frac_tol;
    std::printf("correctness_host_sum routes=%d values=%zu max_abs=%.6g rel=%.6g "
                "bad=%d bad_frac=%.6g nan=%d %s\n",
                total_routes, elems, max_abs, rel, bad, bad_frac, nan,
                fail ? "FAIL" : "ok");
    return fail ? 1 : 0;
}

void load_api(void * lib, Api * api) {
    api->init = (pfn_init) dlsym(lib, "ggml_turbomind_init");
    api->shutdown = (pfn_shutdown) dlsym(lib, "ggml_turbomind_shutdown");
    api->packed_bytes = (pfn_packed_bytes) dlsym(lib, "ggml_turbomind_packed_bytes");
    api->pack_weight = (pfn_pack_weight) dlsym(lib, "ggml_turbomind_pack_weight_expert");
    api->mmgt = (pfn_mmgt) dlsym(lib, "ggml_turbomind_mul_mat_grouped_total_tokens");
    api->mmgs =
        (pfn_mmgs) dlsym(lib, "ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens");
    if (!api->init || !api->shutdown || !api->packed_bytes || !api->pack_weight ||
        !api->mmgt || !api->mmgs) {
        std::fprintf(stderr, "dlsym failed for required TurboMind ABI\n");
        std::exit(2);
    }
}

} // namespace

int main(int argc, char ** argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        usage(argv[0]);
        return 2;
    }

    void * lib = dlopen(opt.lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        std::fprintf(stderr, "dlopen failed for %s: %s\n", opt.lib_path, dlerror());
        return 2;
    }
    Api api;
    load_api(lib, &api);
    enable_peer_access(opt);

    for (int p = 0; p < kParticipants; ++p) {
        CHECK_CUDA(cudaSetDevice(opt.devices[p]));
        if (api.init(opt.devices[p]) != 0) {
            std::fprintf(stderr, "ggml_turbomind_init failed on device %d\n", opt.devices[p]);
            return 3;
        }
    }

    const int total_routes = kExperts * opt.tokens_per_active;
    const size_t a_elems = (size_t) total_routes * kHidden;
    const size_t out_elems = (size_t) total_routes * kHidden;
    const size_t a_bytes = a_elems * sizeof(__half);
    const size_t out_bytes = out_elems * sizeof(__half);
    const std::vector<int> active{0, 1, 2, 3, 4, 5};

    std::vector<std::vector<block_mxfp4>> gate(active.size());
    std::vector<std::vector<block_mxfp4>> up(active.size());
    std::vector<std::vector<block_mxfp4>> down(active.size());
    std::vector<std::vector<block_mxfp4>> gated_full(active.size());
    std::vector<std::vector<block_mxfp4>> gated_shard[kParticipants];
    std::vector<std::vector<block_mxfp4>> down_shard[kParticipants];
    for (int p = 0; p < kParticipants; ++p) {
        gated_shard[p].resize(active.size());
        down_shard[p].resize(active.size());
    }

    for (size_t i = 0; i < active.size(); ++i) {
        make_mxfp4_fixture(gate[i], kFullMid, kHidden, 0x47000000u + (uint32_t) i * 101u);
        make_mxfp4_fixture(up[i], kFullMid, kHidden, 0x55000000u + (uint32_t) i * 131u);
        make_mxfp4_fixture(down[i], kHidden, kFullMid, 0x63000000u + (uint32_t) i * 137u);
        make_fused_interleaved_fixture(gated_full[i], gate[i], up[i], kFullMid, kHidden);
        for (int p = 0; p < kParticipants; ++p) {
            std::vector<block_mxfp4> gate_part;
            std::vector<block_mxfp4> up_part;
            slice_rows_fixture(gate_part, gate[i], kHidden, p * kMidShard, kMidShard);
            slice_rows_fixture(up_part, up[i], kHidden, p * kMidShard, kMidShard);
            make_fused_interleaved_fixture(gated_shard[p][i], gate_part, up_part,
                                           kMidShard, kHidden);
            slice_cols_fixture(down_shard[p][i], down[i], kHidden, kFullMid,
                               p * kMidShard, kMidShard);
        }
    }

    std::vector<int> h_offsets((size_t) kExperts + 1, 0);
    int running = 0;
    for (int e = 0; e < kExperts; ++e) {
        h_offsets[(size_t) e] = running;
        running += opt.tokens_per_active;
    }
    h_offsets[(size_t) kExperts] = running;

    std::mt19937 rng(0xB5010000u + (uint32_t) opt.tokens_per_active);
    std::uniform_real_distribution<float> ad(-0.1f, 0.1f);
    std::vector<__half> h_a(a_elems);
    for (__half & v : h_a) {
        v = __float2half(ad(rng));
    }

    DeviceSide full;
    full.device = opt.devices[0];
    CHECK_CUDA(cudaSetDevice(full.device));
    CHECK_CUDA(cudaStreamCreate(&full.stream));
    CHECK_CUDA(cudaMalloc(&full.d_offsets, h_offsets.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(full.d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&full.d_a, a_bytes));
    CHECK_CUDA(cudaMemcpy(full.d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&full.d_gated, (size_t) total_routes * kFullMid * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&full.d_down, out_bytes));
    if (pack_fixture_set(full.device, api, kFusedFullN, kHidden, active, gated_full,
                         &full.gated) != 0 ||
        pack_fixture_set(full.device, api, kHidden, kFullMid, active, down, &full.down) != 0) {
        return 4;
    }

    for (int i = 0; i < opt.warmup; ++i) {
        if (run_side(full, api, total_routes, kFusedFullN, kFullMid) != 0) {
            return 6;
        }
    }
    sync_device_or_die(full.device, "full reference warmup");

    cudaEvent_t full_start = nullptr;
    cudaEvent_t full_stop = nullptr;
    CHECK_CUDA(cudaSetDevice(full.device));
    CHECK_CUDA(cudaEventCreate(&full_start));
    CHECK_CUDA(cudaEventCreate(&full_stop));
    CHECK_CUDA(cudaEventRecord(full_start, full.stream));
    for (int i = 0; i < opt.iters; ++i) {
        if (run_side(full, api, total_routes, kFusedFullN, kFullMid) != 0) {
            return 8;
        }
    }
    CHECK_CUDA(cudaEventRecord(full_stop, full.stream));
    CHECK_CUDA(cudaEventSynchronize(full_stop));
    const double full_ms = elapsed_ms(full_start, full_stop) / (double) opt.iters;

    DeviceSide sides[kParticipants];
    for (int p = 0; p < kParticipants; ++p) {
        sides[p].device = opt.devices[p];
        CHECK_CUDA(cudaSetDevice(sides[p].device));
        CHECK_CUDA(cudaStreamCreate(&sides[p].stream));
        CHECK_CUDA(cudaMalloc(&sides[p].d_offsets, h_offsets.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(sides[p].d_offsets, h_offsets.data(),
                              h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&sides[p].d_a, a_bytes));
        CHECK_CUDA(cudaMemcpy(sides[p].d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&sides[p].d_gated,
                              (size_t) total_routes * kMidShard * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&sides[p].d_down, out_bytes));
        if (pack_fixture_set(sides[p].device, api, kFusedShardN, kHidden, active,
                             gated_shard[p], &sides[p].gated) != 0 ||
            pack_fixture_set(sides[p].device, api, kHidden, kMidShard, active,
                             down_shard[p], &sides[p].down) != 0) {
            return 5;
        }
    }

    float * d_sum = nullptr;
    __half * d_recv[kParticipants] = {};
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaMalloc(&d_sum, out_elems * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_sum, 0, out_elems * sizeof(float)));
    for (int p = 1; p < kParticipants; ++p) {
        CHECK_CUDA(cudaMalloc(&d_recv[p], out_bytes));
    }

    for (int i = 0; i < opt.warmup; ++i) {
        for (int p = 0; p < kParticipants; ++p) {
            if (run_side(sides[p], api, total_routes, kFusedShardN, kMidShard) != 0) {
                return 7;
            }
            char label[64];
            std::snprintf(label, sizeof(label), "tp4 shard warmup p=%d", p);
            sync_device_or_die(sides[p].device, label);
        }
        (void) run_reduce_to_root(opt, sides, d_sum, d_recv, total_routes, false);
    }

    cudaEvent_t tp_start[kParticipants] = {};
    cudaEvent_t tp_stop[kParticipants] = {};
    for (int p = 0; p < kParticipants; ++p) {
        CHECK_CUDA(cudaSetDevice(sides[p].device));
        CHECK_CUDA(cudaEventCreate(&tp_start[p]));
        CHECK_CUDA(cudaEventCreate(&tp_stop[p]));
        CHECK_CUDA(cudaEventRecord(tp_start[p], sides[p].stream));
    }
    for (int i = 0; i < opt.iters; ++i) {
        for (int p = 0; p < kParticipants; ++p) {
            if (run_side(sides[p], api, total_routes, kFusedShardN, kMidShard) != 0) {
                return 9;
            }
        }
    }
    for (int p = 0; p < kParticipants; ++p) {
        CHECK_CUDA(cudaSetDevice(sides[p].device));
        CHECK_CUDA(cudaEventRecord(tp_stop[p], sides[p].stream));
    }
    double tp_compute_ms = 0.0;
    for (int p = 0; p < kParticipants; ++p) {
        CHECK_CUDA(cudaSetDevice(sides[p].device));
        CHECK_CUDA(cudaEventSynchronize(tp_stop[p]));
        tp_compute_ms = std::max(tp_compute_ms,
                                 (double) elapsed_ms(tp_start[p], tp_stop[p]) /
                                     (double) opt.iters);
    }

    double reduce_ms = 0.0;
    for (int i = 0; i < opt.iters; ++i) {
        reduce_ms += run_reduce_to_root(opt, sides, d_sum, d_recv, total_routes, false);
    }
    reduce_ms /= (double) opt.iters;
    const int correctness = compare_tp_sum_host(opt, full, sides, total_routes);
    const double total_tp_ms = tp_compute_ms + reduce_ms;
    const double input_mib = (double) a_bytes / (1024.0 * 1024.0);
    const double output_mib = (double) out_bytes / (1024.0 * 1024.0);
    std::printf("ds4-v100-tp4-turbomind-layer-smoke devices=");
    for (int p = 0; p < kParticipants; ++p) {
        std::printf("%s%d", p ? "," : "", opt.devices[p]);
    }
    std::printf(" tokens_per_active=%d routes=%d experts=%d hidden=%d full_mid=%d "
                "mid_shard=%d dtype=mxfp4 group_size=%d warmup=%d iters=%d\n",
                opt.tokens_per_active, total_routes, kExperts, kHidden, kFullMid,
                kMidShard, kGroupSize, opt.warmup, opt.iters);
    std::printf("latency_ms full=%.6f tp4_compute=%.6f tp4_reduce=%.6f tp4_total=%.6f "
                "compute_speedup=%.3f total_speedup=%.3f input_mib=%.3f output_mib=%.3f\n",
                full_ms, tp_compute_ms, reduce_ms, total_tp_ms,
                full_ms / tp_compute_ms, full_ms / total_tp_ms, input_mib, output_mib);

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaFree(d_sum));
    for (int p = 1; p < kParticipants; ++p) {
        CHECK_CUDA(cudaFree(d_recv[p]));
    }
    CHECK_CUDA(cudaEventDestroy(full_start));
    CHECK_CUDA(cudaEventDestroy(full_stop));
    for (int p = 0; p < kParticipants; ++p) {
        CHECK_CUDA(cudaSetDevice(sides[p].device));
        CHECK_CUDA(cudaEventDestroy(tp_start[p]));
        CHECK_CUDA(cudaEventDestroy(tp_stop[p]));
    }

    free_packed(full.gated);
    free_packed(full.down);
    CHECK_CUDA(cudaSetDevice(full.device));
    CHECK_CUDA(cudaFree(full.d_offsets));
    CHECK_CUDA(cudaFree(full.d_a));
    CHECK_CUDA(cudaFree(full.d_gated));
    CHECK_CUDA(cudaFree(full.d_down));
    CHECK_CUDA(cudaStreamDestroy(full.stream));
    for (int p = 0; p < kParticipants; ++p) {
        CHECK_CUDA(cudaSetDevice(sides[p].device));
        free_packed(sides[p].gated);
        free_packed(sides[p].down);
        CHECK_CUDA(cudaFree(sides[p].d_offsets));
        CHECK_CUDA(cudaFree(sides[p].d_a));
        CHECK_CUDA(cudaFree(sides[p].d_gated));
        CHECK_CUDA(cudaFree(sides[p].d_down));
        CHECK_CUDA(cudaStreamDestroy(sides[p].stream));
    }
    api.shutdown();
    dlclose(lib);
    return correctness;
}
