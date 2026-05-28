#define _FILE_OFFSET_BITS 64

#include "engine/tp_runtime.h"
#include "kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

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

struct TmIndexEntry {
    std::string semantic_tensor_id;
    std::string runtime_layout;
    std::string sidecar_file;
    int owning_gpu = -1;
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
    const char *lib_path = "./build/turbomind-v100/libggml-turbomind.so";
    int devices[kGpus] = {0, 1, 2, 3, 4, 5, 6, 7};
    int slots = 32;
    int top_k = 6;
    int layer = 2;
    uint32_t kv_slot = 7;
    uint64_t position = 1024;
    int warmup = 5;
    int iters = 30;
    const char *pack_dir = nullptr;
    const char *tm_index_path = nullptr;
    bool descriptor_backed_experts = false;
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

bool parse_tm_entry(const std::vector<std::string> &f, TmIndexEntry *out) {
    if (f.size() < 25) return false;
    TmIndexEntry e;
    e.semantic_tensor_id = f[0];
    e.runtime_layout = f[4];
    if (!parse_int(f[5].c_str(), &e.owning_gpu)) return false;
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
            std::fprintf(stderr, "invalid tm index row for %s\n", path);
            std::fclose(fp);
            return 2;
        }
        if (e.layer_id != layer) continue;
        if (e.semantic_tensor_id == gated_name) {
            if (!valid_tm_entry(e, kFusedN, kHidden,
                                "turbomind_mxfp4_grouped_gate_up_interleaved")) {
                std::fprintf(stderr, "invalid gated descriptor for %s\n", gated_name);
                std::fclose(fp);
                return 3;
            }
            out->gated = e;
            out->have_gated = true;
        } else if (e.semantic_tensor_id == down_name) {
            if (!valid_tm_entry(e, kHidden, kMid, "turbomind_mxfp4_grouped")) {
                std::fprintf(stderr, "invalid down descriptor for %s\n", down_name);
                std::fclose(fp);
                return 4;
            }
            out->down = e;
            out->have_down = true;
        }
    }
    std::fclose(fp);
    if (!out->have_gated || !out->have_down) {
        std::fprintf(stderr, "missing layer %d gated/down descriptors in %s\n", layer, path);
        return 5;
    }
    return 0;
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
                 "       [--slots N] [--top-k N] [--layer N] [--kv-slot N]\n"
                 "       [--position N] [--warmup N] [--iters N]\n"
                 "       [--descriptor-backed-experts --pack-dir DIR --tm-index PATH]\n",
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
        } else if (std::strcmp(arg, "--layer") == 0) {
            if (!val || !parse_int(val, &opt->layer)) return false;
            ++i;
        } else if (std::strcmp(arg, "--kv-slot") == 0) {
            int slot = 0;
            if (!val || !parse_int(val, &slot)) return false;
            opt->kv_slot = (uint32_t)slot;
            ++i;
        } else if (std::strcmp(arg, "--position") == 0) {
            if (!val || !parse_u64(val, &opt->position)) return false;
            ++i;
        } else if (std::strcmp(arg, "--warmup") == 0) {
            if (!val || !parse_int(val, &opt->warmup)) return false;
            ++i;
        } else if (std::strcmp(arg, "--iters") == 0) {
            if (!val || !parse_int(val, &opt->iters) || opt->iters <= 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--pack-dir") == 0) {
            if (!val) return false;
            opt->pack_dir = val;
            ++i;
        } else if (std::strcmp(arg, "--tm-index") == 0) {
            if (!val) return false;
            opt->tm_index_path = val;
            ++i;
        } else if (std::strcmp(arg, "--descriptor-backed-experts") == 0) {
            opt->descriptor_backed_experts = true;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }
    if (opt->descriptor_backed_experts && (!opt->pack_dir || !opt->tm_index_path)) return false;
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

    ds4_tp_runtime_config cfg;
    ds4_tp_runtime_default_config(&cfg);
    cfg.slots = (uint32_t)opt.slots;
    cfg.ctx = 262144;
    cfg.kv_dtype = DS4_V100_TP_KV_F8_E4M3_B128;
    cfg.scratch_bytes = 1536ull * 1024ull * 1024ull;
    for (int i = 0; i < kGpus; ++i) cfg.devices[i] = opt.devices[i];

    char err[512] = {0};
    ds4_tp_runtime *rt = nullptr;
    if (ds4_tp_runtime_open(&rt, &cfg, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "tp_runtime_open_failed\t%s\n", err);
        return 1;
    }

    ds4_tp_runtime_report runtime_report;
    ds4_tp_runtime_get_report(rt, &runtime_report);

    ds4_tp_dense_kv_result kv_result;
    const auto kv_start = std::chrono::steady_clock::now();
    if (ds4_tp_runtime_dense_kv_slice(rt, opt.layer, opt.kv_slot, opt.position,
                                           1, &kv_result, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "tp_runtime_dense_kv_slice_failed\t%s\n", err);
        ds4_tp_runtime_close(rt);
        return 1;
    }
    const auto kv_stop = std::chrono::steady_clock::now();
    const double dense_kv_ms =
        std::chrono::duration<double, std::milli>(kv_stop - kv_start).count();

    void *lib = dlopen(opt.lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        std::fprintf(stderr, "dlopen failed for %s: %s\n", opt.lib_path, dlerror());
        ds4_tp_runtime_close(rt);
        return 2;
    }
    Api api;
    load_api(lib, &api);

    DescriptorBindings bindings;
    uint64_t descriptor_bytes_read = 0;
    if (opt.descriptor_backed_experts) {
        const int rc = parse_tm_index(opt.tm_index_path, opt.layer, &bindings);
        if (rc != 0) {
            ds4_tp_runtime_close(rt);
            return 2;
        }
    }

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
            ds4_tp_runtime_close(rt);
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

        std::mt19937 rng(0xE2320000u + (uint32_t)p * 97u);
        std::uniform_real_distribution<float> dist(-0.003f, 0.003f);
        std::vector<__half> h_a(a_elems);
        for (__half &v : h_a) v = __float2half(dist(rng));
        CHECK_CUDA(cudaMemcpy(r.d_a, h_a.data(), a_elems * sizeof(__half),
                              cudaMemcpyHostToDevice));

        std::vector<int> active;
        for (int e = 0; e < kActiveLocalExperts; ++e) active.push_back(e);
        if (opt.descriptor_backed_experts) {
            if (pack_descriptor_set(r.device, bindings.gated, p, active, opt.pack_dir,
                                    &r.gated, &descriptor_bytes_read) != 0 ||
                pack_descriptor_set(r.device, bindings.down, p, active, opt.pack_dir,
                                    &r.down, &descriptor_bytes_read) != 0) {
                ds4_tp_runtime_close(rt);
                return 4;
            }
        } else {
            std::vector<std::vector<block_mxfp4>> gated(active.size());
            std::vector<std::vector<block_mxfp4>> down(active.size());
            for (size_t i = 0; i < active.size(); ++i) {
                std::vector<block_mxfp4> gate;
                std::vector<block_mxfp4> up;
                make_mxfp4_fixture(gate, kMid, kHidden,
                                   0x61000000u + (uint32_t)p * 1009u + (uint32_t)i * 37u);
                make_mxfp4_fixture(up, kMid, kHidden,
                                   0x62000000u + (uint32_t)p * 1009u + (uint32_t)i * 41u);
                make_fused_interleaved_fixture(gated[i], gate, up, kMid, kHidden);
                make_mxfp4_fixture(down[i], kHidden, kMid,
                                   0x63000000u + (uint32_t)p * 1009u + (uint32_t)i * 43u);
            }
            if (pack_fixture_set(r.device, api, kFusedN, kHidden, active, gated, &r.gated) != 0 ||
                pack_fixture_set(r.device, api, kHidden, kMid, active, down, &r.down) != 0) {
                ds4_tp_runtime_close(rt);
                return 4;
            }
        }
    }

    for (int i = 0; i < opt.warmup; ++i) {
        for (int p = 0; p < kGpus; ++p) {
            if (run_gate(ranks[p], api) != 0 || run_down(ranks[p], api) != 0) {
                ds4_tp_runtime_close(rt);
                return 5;
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
    double worst_ep_ms = 0.0;
    for (int p = 0; p < kGpus; ++p) {
        CHECK_CUDA(cudaSetDevice(ranks[p].device));
        CHECK_CUDA(cudaEventSynchronize(ranks[p].stop));
        const double gate_ms = (double)elapsed_ms(ranks[p].start, ranks[p].mid) / opt.iters;
        const double down_ms = (double)elapsed_ms(ranks[p].mid, ranks[p].stop) / opt.iters;
        worst_gate_ms = std::max(worst_gate_ms, gate_ms);
        worst_down_ms = std::max(worst_down_ms, down_ms);
        worst_ep_ms = std::max(worst_ep_ms, gate_ms + down_ms);
        std::printf("rank\t%d\tdevice\t%d\troutes\t%d\tactive_local_experts\t%d\t"
                    "max_routes_per_expert\t%d\tgate_ms\t%.6f\tdown_ms\t%.6f\t"
                    "ep_ms\t%.6f\n",
                    p, ranks[p].device, ranks[p].routes, ranks[p].active_experts,
                    ranks[p].max_routes_per_expert, gate_ms, down_ms, gate_ms + down_ms);
    }

    double repeat_max_abs = 0.0;
    int repeat_bad = 0;
    int repeat_nan = 0;
    for (int p = 0; p < kGpus; ++p) {
        if (check_repeat(ranks[p], api, &repeat_max_abs, &repeat_bad, &repeat_nan) != 0) {
            ds4_tp_runtime_close(rt);
            return 8;
        }
    }

    const uint64_t dispatch_bytes = (uint64_t)aggregate_routes * kHidden * sizeof(__half);
    const uint64_t return_bytes = dispatch_bytes;
    const double imbalance = min_routes > 0 ? (double)max_routes / (double)min_routes : 0.0;
    const double one_layer_ms = dense_kv_ms + worst_ep_ms;

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
    std::printf("tp_ep_layer_smoke\tslots\t%d\tctx\t%llu\ttop_k\t%d\t"
                "aggregate_routes\t%d\tglobal_experts\t%d\tlocal_experts\t%d\t"
                "active_local_experts\t%d\tdispatch_bytes\t%llu\treturn_bytes\t%llu\t"
                "expert_source\t%s\tdescriptor_bytes_read\t%llu\t"
                "route_imbalance\t%.6f\tworst_gate_ms\t%.6f\tworst_down_ms\t%.6f\t"
                "worst_ep_ms\t%.6f\tone_layer_ms\t%.6f\trepeat_max_abs\t%.9f\t"
                "repeat_bad\t%d\trepeat_nan\t%d\t%s\n",
                opt.slots, (unsigned long long)cfg.ctx, opt.top_k, aggregate_routes,
                kGlobalExperts, kLocalExperts, kActiveLocalExperts,
                (unsigned long long)dispatch_bytes, (unsigned long long)return_bytes,
                opt.descriptor_backed_experts ? "descriptor" : "synthetic",
                (unsigned long long)descriptor_bytes_read,
                imbalance, worst_gate_ms, worst_down_ms, worst_ep_ms, one_layer_ms,
                repeat_max_abs, repeat_bad, repeat_nan,
                (kv_result.max_abs == 0.0 && repeat_bad == 0 && repeat_nan == 0) ? "PASS" : "FAIL");

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
    ds4_tp_runtime_close(rt);
    return (kv_result.max_abs == 0.0 && repeat_bad == 0 && repeat_nan == 0) ? 0 : 1;
}
