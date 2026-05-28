#define _FILE_OFFSET_BITS 64

#include "ds4_v100_tp_runtime.h"
#include "kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h"
extern "C" {
#include "ds4.h"
}

#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <mma.h>
#if __has_include(<nccl.h>)
#include <nccl.h>
#else
#include "third_party/nccl_compat/nccl.h"
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kGpus = 8;
constexpr int kHidden = 4096;
constexpr int kMid = 2048;
constexpr int kHeadDim = 512;
constexpr int kHeadCount = 64;
constexpr int kLocalHeads = kHeadCount / kGpus;
constexpr int kAttentionOutputAInput = kLocalHeads * kHeadDim;
constexpr int kAttentionOutputAFull = 8192;
constexpr int kRawSwaRows = 128;
constexpr int kRotaryDim = 64;
constexpr int kIndexerHeadDim = 128;
constexpr int kIndexerHead = 64;
constexpr int kIndexerTopK = 512;
constexpr int kCompWidthMax = 2 * kHeadDim;
constexpr int kBoundedCompRows = 8;
constexpr int kIndexCompWidth = 2 * kIndexerHeadDim;
constexpr int kIndexCompStateRows = 8;
constexpr uint32_t kRopeOrigCtx = 65536;
constexpr float kRopeFreqBase = 10000.0f;
constexpr float kCompressRopeFreqBase = 160000.0f;
constexpr float kRopeScaleFactor = 16.0f;
constexpr float kRopeYarnBetaFast = 32.0f;
constexpr float kRopeYarnBetaSlow = 1.0f;
constexpr int kFusedN = 2 * kMid;
constexpr int kGlobalExperts = 256;
constexpr int kLocalExperts = kGlobalExperts / kGpus;
constexpr int kPackedLocalExperts = kLocalExperts;
constexpr int kRouterHashRows = 129280;
constexpr int kGroupSize = 32;
constexpr int kDType = GGML_TM_DTYPE_MXFP4;
constexpr int kHcRows = 4;
constexpr int kHcMix = 24;
constexpr int kModelTopK = 6;
constexpr int kGraphOrderEventSlots = 1024;
constexpr float kSyntheticRouteWeight = 0.125f;
constexpr float kRoutedSwigluClamp = 10.0f;
constexpr float kReferenceRouteInputTargetAbs = 32.0f;
constexpr float kReferenceHcStateTargetAbs = 32.0f;
constexpr float kFp16Max = 65504.0f;

enum class PeerLinkClass {
    Self = 0,
    Nv1 = 1,
    Nv2 = 2,
    Sys = 3,
    Unknown = 4,
};

struct PeerCopySnapshot {
    uint64_t ops = 0;
    uint64_t bytes = 0;
    uint64_t self_ops = 0;
    uint64_t self_bytes = 0;
    uint64_t nv1_ops = 0;
    uint64_t nv1_bytes = 0;
    uint64_t nv2_ops = 0;
    uint64_t nv2_bytes = 0;
    uint64_t sys_ops = 0;
    uint64_t sys_bytes = 0;
    uint64_t unknown_ops = 0;
    uint64_t unknown_bytes = 0;
    int first_sys_src = -1;
    int first_sys_dst = -1;
    uint64_t first_sys_bytes = 0;
    int first_sys_line = 0;
    const char *first_sys_site = nullptr;
    uint64_t top_sys_site_ops = 0;
    uint64_t top_sys_site_bytes = 0;
    uint64_t top_sys_site_total_ops = 0;
    uint64_t top_sys_site_total_bytes = 0;
    int top_sys_site_line = 0;
    const char *top_sys_site = nullptr;
};

struct PeerCopySiteBucket {
    std::atomic<int> line{0};
    const char *site = nullptr;
    std::atomic<uint64_t> ops{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> sys_ops{0};
    std::atomic<uint64_t> sys_bytes{0};
    std::atomic<uint64_t> nv1_ops{0};
    std::atomic<uint64_t> nv1_bytes{0};
    std::atomic<uint64_t> nv2_ops{0};
    std::atomic<uint64_t> nv2_bytes{0};
    std::atomic<uint64_t> self_ops{0};
    std::atomic<uint64_t> self_bytes{0};
    std::atomic<uint64_t> unknown_ops{0};
    std::atomic<uint64_t> unknown_bytes{0};
};

struct PeerCopyAccounting {
    std::atomic<int> enabled{0};
    std::atomic<int> reject_sys{0};
    std::atomic<uint64_t> ops{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> self_ops{0};
    std::atomic<uint64_t> self_bytes{0};
    std::atomic<uint64_t> nv1_ops{0};
    std::atomic<uint64_t> nv1_bytes{0};
    std::atomic<uint64_t> nv2_ops{0};
    std::atomic<uint64_t> nv2_bytes{0};
    std::atomic<uint64_t> sys_ops{0};
    std::atomic<uint64_t> sys_bytes{0};
    std::atomic<uint64_t> unknown_ops{0};
    std::atomic<uint64_t> unknown_bytes{0};
    std::atomic<int> first_sys_src{-1};
    std::atomic<int> first_sys_dst{-1};
    std::atomic<uint64_t> first_sys_bytes{0};
    std::atomic<int> first_sys_line{0};
    const char *first_sys_site = nullptr;
    PeerCopySiteBucket sites[256];
};

PeerCopyAccounting g_peer_copy_accounting;

int visible_to_physical_device(int visible) {
    static int initialized = 0;
    static int map[64];
    if (!initialized) {
        for (int i = 0; i < 64; ++i) map[i] = i;
        const char *env = std::getenv("CUDA_VISIBLE_DEVICES");
        if (env && *env) {
            const char *p = env;
            int idx = 0;
            while (*p && idx < 64) {
                char *end = nullptr;
                const long value = std::strtol(p, &end, 10);
                if (end == p) break;
                map[idx++] = (int)value;
                p = end;
                while (*p == ',' || *p == ' ' || *p == '\t') ++p;
            }
        }
        initialized = 1;
    }
    return (visible >= 0 && visible < 64) ? map[visible] : visible;
}

int v100_nvlink_count(int a, int b) {
    if (a == b) return 0;
    const int lo = std::min(a, b);
    const int hi = std::max(a, b);
    if ((lo == 0 && hi == 1) || (lo == 0 && hi == 2) ||
        (lo == 1 && hi == 3) || (lo == 2 && hi == 6) ||
        (lo == 3 && hi == 7) || (lo == 4 && hi == 5) ||
        (lo == 4 && hi == 6) || (lo == 5 && hi == 7)) {
        return 1;
    }
    if ((lo == 0 && hi == 3) || (lo == 0 && hi == 4) ||
        (lo == 1 && hi == 2) || (lo == 1 && hi == 5) ||
        (lo == 2 && hi == 3) || (lo == 4 && hi == 7) ||
        (lo == 5 && hi == 6) || (lo == 6 && hi == 7)) {
        return 2;
    }
    return 0;
}

PeerLinkClass peer_link_class(int dst_device, int src_device) {
    if (dst_device == src_device) return PeerLinkClass::Self;
    const int dst = visible_to_physical_device(dst_device);
    const int src = visible_to_physical_device(src_device);
    if (dst < 0 || src < 0) return PeerLinkClass::Unknown;
    const int links = v100_nvlink_count(dst, src);
    if (links == 1) return PeerLinkClass::Nv1;
    if (links >= 2) return PeerLinkClass::Nv2;
    return PeerLinkClass::Sys;
}

void reset_peer_copy_accounting(bool enabled, bool reject_sys) {
    g_peer_copy_accounting.enabled.store(enabled ? 1 : 0, std::memory_order_relaxed);
    g_peer_copy_accounting.reject_sys.store(reject_sys ? 1 : 0, std::memory_order_relaxed);
    g_peer_copy_accounting.ops.store(0, std::memory_order_relaxed);
    g_peer_copy_accounting.bytes.store(0, std::memory_order_relaxed);
    g_peer_copy_accounting.self_ops.store(0, std::memory_order_relaxed);
    g_peer_copy_accounting.self_bytes.store(0, std::memory_order_relaxed);
    g_peer_copy_accounting.nv1_ops.store(0, std::memory_order_relaxed);
    g_peer_copy_accounting.nv1_bytes.store(0, std::memory_order_relaxed);
    g_peer_copy_accounting.nv2_ops.store(0, std::memory_order_relaxed);
    g_peer_copy_accounting.nv2_bytes.store(0, std::memory_order_relaxed);
    g_peer_copy_accounting.sys_ops.store(0, std::memory_order_relaxed);
    g_peer_copy_accounting.sys_bytes.store(0, std::memory_order_relaxed);
    g_peer_copy_accounting.unknown_ops.store(0, std::memory_order_relaxed);
    g_peer_copy_accounting.unknown_bytes.store(0, std::memory_order_relaxed);
    g_peer_copy_accounting.first_sys_src.store(-1, std::memory_order_relaxed);
    g_peer_copy_accounting.first_sys_dst.store(-1, std::memory_order_relaxed);
    g_peer_copy_accounting.first_sys_bytes.store(0, std::memory_order_relaxed);
    g_peer_copy_accounting.first_sys_line.store(0, std::memory_order_relaxed);
    g_peer_copy_accounting.first_sys_site = nullptr;
    for (PeerCopySiteBucket &bucket : g_peer_copy_accounting.sites) {
        bucket.line.store(0, std::memory_order_relaxed);
        bucket.site = nullptr;
        bucket.ops.store(0, std::memory_order_relaxed);
        bucket.bytes.store(0, std::memory_order_relaxed);
        bucket.sys_ops.store(0, std::memory_order_relaxed);
        bucket.sys_bytes.store(0, std::memory_order_relaxed);
        bucket.nv1_ops.store(0, std::memory_order_relaxed);
        bucket.nv1_bytes.store(0, std::memory_order_relaxed);
        bucket.nv2_ops.store(0, std::memory_order_relaxed);
        bucket.nv2_bytes.store(0, std::memory_order_relaxed);
        bucket.self_ops.store(0, std::memory_order_relaxed);
        bucket.self_bytes.store(0, std::memory_order_relaxed);
        bucket.unknown_ops.store(0, std::memory_order_relaxed);
        bucket.unknown_bytes.store(0, std::memory_order_relaxed);
    }
}

PeerCopySiteBucket *peer_copy_site_bucket(const char *site, int line) {
    if (line <= 0) return nullptr;
    for (PeerCopySiteBucket &bucket : g_peer_copy_accounting.sites) {
        const int existing = bucket.line.load(std::memory_order_relaxed);
        if (existing == line) return &bucket;
        if (existing == 0) {
            int expected = 0;
            if (bucket.line.compare_exchange_strong(expected, line,
                                                    std::memory_order_relaxed)) {
                bucket.site = site;
                return &bucket;
            }
            if (expected == line) return &bucket;
        }
    }
    return nullptr;
}

[[maybe_unused]] void record_peer_copy(int dst_device, int src_device,
                                       size_t bytes, const char *site,
                                       int line) {
    if (!g_peer_copy_accounting.enabled.load(std::memory_order_relaxed) &&
        !g_peer_copy_accounting.reject_sys.load(std::memory_order_relaxed)) {
        return;
    }
    const PeerLinkClass cls = peer_link_class(dst_device, src_device);
    g_peer_copy_accounting.ops.fetch_add(1, std::memory_order_relaxed);
    g_peer_copy_accounting.bytes.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
    PeerCopySiteBucket *bucket = peer_copy_site_bucket(site, line);
    if (bucket) {
        bucket->ops.fetch_add(1, std::memory_order_relaxed);
        bucket->bytes.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
    }
    if (cls == PeerLinkClass::Self) {
        g_peer_copy_accounting.self_ops.fetch_add(1, std::memory_order_relaxed);
        g_peer_copy_accounting.self_bytes.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
        if (bucket) {
            bucket->self_ops.fetch_add(1, std::memory_order_relaxed);
            bucket->self_bytes.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
        }
    } else if (cls == PeerLinkClass::Nv1) {
        g_peer_copy_accounting.nv1_ops.fetch_add(1, std::memory_order_relaxed);
        g_peer_copy_accounting.nv1_bytes.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
        if (bucket) {
            bucket->nv1_ops.fetch_add(1, std::memory_order_relaxed);
            bucket->nv1_bytes.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
        }
    } else if (cls == PeerLinkClass::Nv2) {
        g_peer_copy_accounting.nv2_ops.fetch_add(1, std::memory_order_relaxed);
        g_peer_copy_accounting.nv2_bytes.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
        if (bucket) {
            bucket->nv2_ops.fetch_add(1, std::memory_order_relaxed);
            bucket->nv2_bytes.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
        }
    } else if (cls == PeerLinkClass::Sys) {
        g_peer_copy_accounting.sys_ops.fetch_add(1, std::memory_order_relaxed);
        g_peer_copy_accounting.sys_bytes.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
        if (bucket) {
            bucket->sys_ops.fetch_add(1, std::memory_order_relaxed);
            bucket->sys_bytes.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
        }
        int expected = -1;
        if (g_peer_copy_accounting.first_sys_src.compare_exchange_strong(
                expected, src_device, std::memory_order_relaxed)) {
            g_peer_copy_accounting.first_sys_dst.store(dst_device, std::memory_order_relaxed);
            g_peer_copy_accounting.first_sys_bytes.store((uint64_t)bytes, std::memory_order_relaxed);
            g_peer_copy_accounting.first_sys_line.store(line, std::memory_order_relaxed);
            g_peer_copy_accounting.first_sys_site = site;
        }
    } else {
        g_peer_copy_accounting.unknown_ops.fetch_add(1, std::memory_order_relaxed);
        g_peer_copy_accounting.unknown_bytes.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
        if (bucket) {
            bucket->unknown_ops.fetch_add(1, std::memory_order_relaxed);
            bucket->unknown_bytes.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
        }
    }
}

PeerCopySnapshot peer_copy_snapshot() {
    PeerCopySnapshot s;
    s.ops = g_peer_copy_accounting.ops.load(std::memory_order_relaxed);
    s.bytes = g_peer_copy_accounting.bytes.load(std::memory_order_relaxed);
    s.self_ops = g_peer_copy_accounting.self_ops.load(std::memory_order_relaxed);
    s.self_bytes = g_peer_copy_accounting.self_bytes.load(std::memory_order_relaxed);
    s.nv1_ops = g_peer_copy_accounting.nv1_ops.load(std::memory_order_relaxed);
    s.nv1_bytes = g_peer_copy_accounting.nv1_bytes.load(std::memory_order_relaxed);
    s.nv2_ops = g_peer_copy_accounting.nv2_ops.load(std::memory_order_relaxed);
    s.nv2_bytes = g_peer_copy_accounting.nv2_bytes.load(std::memory_order_relaxed);
    s.sys_ops = g_peer_copy_accounting.sys_ops.load(std::memory_order_relaxed);
    s.sys_bytes = g_peer_copy_accounting.sys_bytes.load(std::memory_order_relaxed);
    s.unknown_ops = g_peer_copy_accounting.unknown_ops.load(std::memory_order_relaxed);
    s.unknown_bytes = g_peer_copy_accounting.unknown_bytes.load(std::memory_order_relaxed);
    s.first_sys_src = g_peer_copy_accounting.first_sys_src.load(std::memory_order_relaxed);
    s.first_sys_dst = g_peer_copy_accounting.first_sys_dst.load(std::memory_order_relaxed);
    s.first_sys_bytes = g_peer_copy_accounting.first_sys_bytes.load(std::memory_order_relaxed);
    s.first_sys_line = g_peer_copy_accounting.first_sys_line.load(std::memory_order_relaxed);
    s.first_sys_site = g_peer_copy_accounting.first_sys_site;
    for (const PeerCopySiteBucket &bucket : g_peer_copy_accounting.sites) {
        const int line = bucket.line.load(std::memory_order_relaxed);
        if (line == 0) continue;
        const uint64_t sys_bytes = bucket.sys_bytes.load(std::memory_order_relaxed);
        if (sys_bytes > s.top_sys_site_bytes) {
            s.top_sys_site_bytes = sys_bytes;
            s.top_sys_site_ops = bucket.sys_ops.load(std::memory_order_relaxed);
            s.top_sys_site_total_ops = bucket.ops.load(std::memory_order_relaxed);
            s.top_sys_site_total_bytes = bucket.bytes.load(std::memory_order_relaxed);
            s.top_sys_site_line = line;
            s.top_sys_site = bucket.site;
        }
    }
    return s;
}

void print_peer_copy_summary(const char *label) {
    const PeerCopySnapshot s = peer_copy_snapshot();
    std::printf("tp_ep_peer_copy_summary\tlabel\t%s\taccounting\t%d\treject_sys\t%d\t"
                "ops\t%llu\tbytes\t%llu\tself_ops\t%llu\tself_bytes\t%llu\t"
                "nv1_ops\t%llu\tnv1_bytes\t%llu\tnv2_ops\t%llu\tnv2_bytes\t%llu\t"
                "sys_ops\t%llu\tsys_bytes\t%llu\tunknown_ops\t%llu\tunknown_bytes\t%llu\t"
                "first_sys_src\t%d\tfirst_sys_dst\t%d\tfirst_sys_bytes\t%llu\t"
                "first_sys_site\t%s\tfirst_sys_line\t%d\t%s\n",
                label ? label : "default",
                g_peer_copy_accounting.enabled.load(std::memory_order_relaxed),
                g_peer_copy_accounting.reject_sys.load(std::memory_order_relaxed),
                (unsigned long long)s.ops, (unsigned long long)s.bytes,
                (unsigned long long)s.self_ops, (unsigned long long)s.self_bytes,
                (unsigned long long)s.nv1_ops, (unsigned long long)s.nv1_bytes,
                (unsigned long long)s.nv2_ops, (unsigned long long)s.nv2_bytes,
                (unsigned long long)s.sys_ops, (unsigned long long)s.sys_bytes,
                (unsigned long long)s.unknown_ops, (unsigned long long)s.unknown_bytes,
                s.first_sys_src, s.first_sys_dst,
                (unsigned long long)s.first_sys_bytes,
                s.first_sys_site ? s.first_sys_site : "-",
                s.first_sys_line,
                s.sys_ops == 0 ? "PASS" : "FAIL");
    for (const PeerCopySiteBucket &bucket : g_peer_copy_accounting.sites) {
        const int line = bucket.line.load(std::memory_order_relaxed);
        const uint64_t sys_ops = bucket.sys_ops.load(std::memory_order_relaxed);
        if (line == 0 || sys_ops == 0) continue;
        std::printf("tp_ep_peer_copy_site\tlabel\t%s\tsite\t%s\tline\t%d\t"
                    "ops\t%llu\tbytes\t%llu\tself_ops\t%llu\tself_bytes\t%llu\t"
                    "nv1_ops\t%llu\tnv1_bytes\t%llu\tnv2_ops\t%llu\tnv2_bytes\t%llu\t"
                    "sys_ops\t%llu\tsys_bytes\t%llu\tunknown_ops\t%llu\tunknown_bytes\t%llu\t%s\n",
                    label ? label : "default",
                    bucket.site ? bucket.site : "-",
                    line,
                    (unsigned long long)bucket.ops.load(std::memory_order_relaxed),
                    (unsigned long long)bucket.bytes.load(std::memory_order_relaxed),
                    (unsigned long long)bucket.self_ops.load(std::memory_order_relaxed),
                    (unsigned long long)bucket.self_bytes.load(std::memory_order_relaxed),
                    (unsigned long long)bucket.nv1_ops.load(std::memory_order_relaxed),
                    (unsigned long long)bucket.nv1_bytes.load(std::memory_order_relaxed),
                    (unsigned long long)bucket.nv2_ops.load(std::memory_order_relaxed),
                    (unsigned long long)bucket.nv2_bytes.load(std::memory_order_relaxed),
                    (unsigned long long)sys_ops,
                    (unsigned long long)bucket.sys_bytes.load(std::memory_order_relaxed),
                    (unsigned long long)bucket.unknown_ops.load(std::memory_order_relaxed),
                    (unsigned long long)bucket.unknown_bytes.load(std::memory_order_relaxed),
                    bucket.sys_bytes.load(std::memory_order_relaxed) == 0 ? "PASS" : "FAIL");
    }
}

#define CHECK_CUDA(expr)                                                              \
    do {                                                                              \
        cudaError_t err__ = (expr);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            std::fprintf(stderr, "cuda error %s:%d: %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(err__));                                  \
            std::exit(2);                                                             \
        }                                                                             \
    } while (0)

#define CHECK_NCCL(expr)                                                              \
    do {                                                                              \
        ncclResult_t err__ = (expr);                                                   \
        if (err__ != ncclSuccess) {                                                   \
            std::fprintf(stderr, "nccl error %s:%d: %s\n", __FILE__, __LINE__,       \
                         ncclGetErrorString(err__));                                  \
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
    pfn_mmgs mmgs_clamped = nullptr;
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
    void *d_w_contiguous = nullptr;
    void *d_s_contiguous = nullptr;
    void *d_w_table = nullptr;
    void *d_s_table = nullptr;
    int k_pack = 0;
};

struct RankState {
    int rank = 0;
    int device = 0;
    int routes = 0;
    int route_capacity = 0;
    int active_experts = 0;
    int max_routes_per_expert = 0;
    cudaStream_t stream = nullptr;
    cudaStream_t dense_stream = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaStream_t copy_streams[kGpus] = {};
    cudaEvent_t copy_done[kGpus] = {};
    cudaEvent_t stream_done = nullptr;
    cudaEvent_t dense_done = nullptr;
    cudaEvent_t graph_stream_done[kGraphOrderEventSlots] = {};
    cudaEvent_t graph_dense_done[kGraphOrderEventSlots] = {};
    int graph_event_cursor = 0;
    int *d_route_index_by_slot[kGpus] = {};
    int *d_route_indices_by_slot[kGpus] = {};
    int *d_route_count_by_slot[kGpus] = {};
    int *d_route_compact_plan = nullptr;
    size_t route_compact_plan_ints = 0;
    int *d_router_selected_plan = nullptr;
    float *d_router_weights_plan = nullptr;
    int *d_route_offsets_all = nullptr;
    int *d_route_totals = nullptr;
    int *d_offsets = nullptr;
    int *d_route_slots = nullptr;
    float *d_route_weights = nullptr;
    float *d_route_inv_scale = nullptr;
    __half *d_a = nullptr;
    __half *d_gate_up = nullptr;
    __half *d_gated = nullptr;
    __half *d_down = nullptr;
    float *d_ep_contrib_all = nullptr;
    __half *d_ep_contrib_half_all = nullptr;
    float *d_ep_contrib_bcast_all = nullptr;
    __half *d_ep_contrib_half_bcast_all = nullptr;
    float *d_ep_remote[kGpus] = {};
    __half *d_ep_remote_half[kGpus] = {};
    float *d_ep_sum = nullptr;
    float *d_next_hidden = nullptr;
    float *d_current_shard = nullptr;
    float *d_current_full = nullptr;
    float *d_current_full_normed = nullptr;
    float *d_current_full_rank_major = nullptr;
    float *d_post_attn_full_rank_major = nullptr;
    float *d_rank_major_norm_scale = nullptr;
    float *d_router_logits_shard = nullptr;
    float *d_router_logits_rank_major = nullptr;
    unsigned long long *d_half_diff_counts = nullptr;
    unsigned int *d_half_diff_max_bits = nullptr;
    int *d_half_diff_first = nullptr;
    unsigned long long *d_post_attn_route_audit = nullptr;
    float *d_final_hc_shard = nullptr;
    float *d_hc_scratch_shard = nullptr;
    float *d_hc_split = nullptr;
    float *d_hc_reduce_max = nullptr;
    float *d_hc_reduce_sumsq = nullptr;
    float *d_hc_reduce_mix = nullptr;
    float *d_attn_kv_full = nullptr;
    float *d_attn_raw_swa = nullptr;
    float *d_attn_raw_swa_layers[43] = {};
    float *d_attn_sinks = nullptr;
    float *d_attn_heads = nullptr;
    float *d_attn_output_a_full = nullptr;
    float *d_post_attn_shard = nullptr;
    float *d_attn_comp_kv_cur = nullptr;
    float *d_attn_comp_score_cur = nullptr;
    float *d_attn_comp_state_kv = nullptr;
    float *d_attn_comp_state_score = nullptr;
    float *d_attn_comp_rows = nullptr;
    float *d_attn_comp_state_kv_layers[43] = {};
    float *d_attn_comp_state_score_layers[43] = {};
    float *d_attn_comp_rows_layers[43] = {};
    uint32_t attn_comp_rows_written_layers[43] = {};
    uint64_t attn_comp_row_position_layers[43][kBoundedCompRows] = {};
    uint64_t attn_comp_row_loaded_position_layers[43][kBoundedCompRows] = {};
    bool attn_comp_row_loaded_layers[43][kBoundedCompRows] = {};
    float *d_index_comp_kv_cur = nullptr;
    float *d_index_comp_score_cur = nullptr;
    float *d_index_comp_state_kv = nullptr;
    float *d_index_comp_state_score = nullptr;
    float *d_index_comp_rows = nullptr;
    float *d_index_comp_state_kv_layers[43] = {};
    float *d_index_comp_state_score_layers[43] = {};
    float *d_index_comp_rows_layers[43] = {};
    uint32_t index_comp_rows_written_layers[43] = {};
    uint64_t index_comp_row_position_layers[43][kBoundedCompRows] = {};
    uint64_t index_comp_row_loaded_position_layers[43][kBoundedCompRows] = {};
    bool index_comp_row_loaded_layers[43][kBoundedCompRows] = {};
    float *d_indexer_scores = nullptr;
    uint32_t *d_indexer_topk = nullptr;
    bool hc_initialized = false;
    PackedExperts gated;
    PackedExperts down;
    ncclComm_t compose_nccl = nullptr;
    bool compose_nccl_initialized = false;
    cudaEvent_t dense_wait = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t mid = nullptr;
    cudaEvent_t stop = nullptr;
};

struct RoutePlanHostWorkspace {
    bool initialized = false;
    bool uploads_pending = false;
    int slots = 0;
    int top_k = 0;
    int devices[kGpus] = {};
    size_t route_capacity = 0;
    size_t compact_plan_ints = 0;
    int *h_selected = nullptr;
    float *h_weights = nullptr;
    int *h_offsets[kGpus] = {};
    int *h_route_slots[kGpus] = {};
    float *h_route_weights[kGpus] = {};
    int *h_route_index_by_slot[kGpus] = {};
    int *h_route_indices_by_slot[kGpus] = {};
    int *h_route_count_by_slot[kGpus] = {};
    int *h_compact_plan = nullptr;
    cudaEvent_t upload_done[kGpus] = {};
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

struct LayerDenseOps {
    ResidentF8Dense attn_q_a;
    ResidentF8Dense attn_q_b;
    ResidentF8Dense attn_kv_latent;
    ResidentF8Dense attn_output_a;
    ResidentF8Dense attn_compress_kv;
    ResidentF8Dense attn_compress_gate;
    ResidentF8Dense indexer_attn_q_b;
    ResidentF8Dense indexer_proj;
    ResidentF8Dense indexer_compress_kv;
    ResidentF8Dense indexer_compress_gate;
    ResidentF8Dense attn;
    ResidentF8Dense shared;
    ResidentF8Dense shared_gate;
    ResidentF8Dense shared_up;
    bool initialized = false;
};

struct SharedDenseOps {
    LayerDenseOps layers[43];
    uint64_t loaded_bytes = 0;
    bool initialized = false;
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
    uint64_t gpu_cache_aligned_bytes[kGpus] = {};
    uint64_t gpu_temp_bytes[kGpus] = {};
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
    bool nccl_reduce_scatter_compose = false;
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
    double decode_compose_reduce_ms_per_step = 0.0;
    double decode_compose_copy_ms_per_step = 0.0;
    double decode_compose_final_ms_per_step = 0.0;
    double decode_hc_current_input_ms_per_step = 0.0;
    double decode_hc_current_seed_ms_per_step = 0.0;
    double decode_hc_current_attn_mix_ms_per_step = 0.0;
    double decode_hc_current_split_ms_per_step = 0.0;
    double decode_hc_current_gather_ms_per_step = 0.0;
    double decode_hc_current_ffn_router_ms_per_step = 0.0;
    double decode_hc_current_ffn_norm_ms_per_step = 0.0;
    double decode_hc_current_router_select_ms_per_step = 0.0;
    double decode_hc_current_router_d2h_ms_per_step = 0.0;
    double decode_hc_current_route_upload_ms_per_step = 0.0;
    double decode_hc_current_fill_pack_ms_per_step = 0.0;
    double decode_pre_ep_hc_current_ms_per_step = 0.0;
    double decode_pre_ep_attention_projection_ms_per_step = 0.0;
    double decode_pre_ep_compressed_kv_ms_per_step = 0.0;
    double decode_pre_ep_attention_state_ms_per_step = 0.0;
    double decode_pre_ep_typed_history_ms_per_step = 0.0;
    double decode_pre_ep_raw_read_ms_per_step = 0.0;
    double decode_pre_ep_attention_output_ms_per_step = 0.0;
    double decode_pre_ep_post_attention_ffn_input_ms_per_step = 0.0;
    double decode_final_hc_ms_per_step = 0.0;
    int decode_cudagraph_sync_all_calls = 0;
    int decode_cudagraph_event_barrier_calls = 0;
    int decode_cudagraph_rank_stream_syncs = 0;
    int decode_cudagraph_dense_stream_syncs = 0;
    int decode_cudagraph_copy_stream_syncs = 0;
    int decode_cudagraph_capture_attempted = 0;
    int decode_cudagraph_capture_succeeded = 0;
    int decode_cudagraph_capture_error = 0;
    size_t decode_cudagraph_capture_nodes = 0;
    int decode_cudagraph_replay_attempted = 0;
    int decode_cudagraph_replay_succeeded = 0;
    int decode_cudagraph_replay_error = 0;
    int decode_cudagraph_persistent_cache_hits = 0;
    int decode_cudagraph_persistent_cache_misses = 0;
    int decode_cudagraph_persistent_invalidations = 0;
    int decode_cudagraph_persistent_invalidate_layer = 0;
    int decode_cudagraph_persistent_invalidate_slots = 0;
    int decode_cudagraph_persistent_invalidate_position = 0;
    int decode_cudagraph_persistent_invalidate_root_device = 0;
    int decode_cudagraph_persistent_invalidate_root_stream = 0;
    double decode_cudagraph_instantiate_ms = 0.0;
    double decode_cudagraph_replay_ms = 0.0;
    uint64_t decode_checksum = 0;
    int decode_finite_bad = 0;
    int rc = 0;
};

struct ServingBenchResult {
    uint64_t prompt_tokens = 0;
    uint64_t generated_tokens = 0;
    uint64_t continuation_tokens = 0;
    double first_token_decode_ms = 0.0;
    double continuation_decode_ms = 0.0;
    double first_token_wall_ms = 0.0;
    double continuation_wall_ms = 0.0;
    double total_decode_ms = 0.0;
    double total_wall_ms = 0.0;
    double total_ep_ms = 0.0;
    double total_dense_ms = 0.0;
    double total_compose_ms = 0.0;
    double total_compose_reduce_ms = 0.0;
    double total_compose_copy_ms = 0.0;
    double total_compose_final_ms = 0.0;
    double total_hc_current_input_ms = 0.0;
    double aggregate_generated_tok_s_decode = 0.0;
    double aggregate_generated_tok_s_wall = 0.0;
    double aggregate_continuation_tok_s_decode = 0.0;
    double aggregate_continuation_tok_s_wall = 0.0;
    bool diagnostic_output_head = false;
    bool diagnostic_output_head_proxy_hc = false;
    double output_head_ms = 0.0;
    double output_head_gather_ms = 0.0;
    double output_head_prep_ms = 0.0;
    double output_head_broadcast_ms = 0.0;
    double output_head_projection_ms = 0.0;
    double output_head_top1_ms = 0.0;
    bool token_input_seed = false;
    uint32_t first_input_token = UINT32_MAX;
    std::vector<uint32_t> selected_tokens;
    std::vector<float> selected_logits;
    std::vector<uint64_t> step_checksums;
    uint64_t checksum = 0;
};

struct SharedApi {
    void *lib = nullptr;
    Api api = {};
    bool initialized = false;
};

struct TpCudaGraphLayerExec {
    bool initialized = false;
    int layer = -1;
    int slots = 0;
    uint64_t position = 0;
    int root_device = -1;
    cudaStream_t root_stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    size_t nodes = 0;
    int captures = 0;
    int instantiates = 0;
    int replays = 0;
    int failures = 0;
    double instantiate_ms = 0.0;
    double replay_ms = 0.0;
};

struct TpCudaGraphCache {
    TpCudaGraphLayerExec layers[43];
};

struct SharedRankBuffers {
    RankState ranks[kGpus];
    TpCudaGraphCache graph_cache;
    bool initialized = false;
    uint64_t core_bytes = 0;
};

struct SharedTpRuntime {
    ds4_v100_tp_runtime *rt = nullptr;
    ds4_v100_tp_runtime_report report = {};
    bool initialized = false;
};

struct LayerExpertCache {
    DescriptorBindings bindings;
    PackedExperts gated[kGpus];
    PackedExperts down[kGpus];
    uint64_t bytes = 0;
    bool initialized = false;
};

struct SharedExpertBindings {
    LayerExpertCache layers[43];
    uint64_t bytes = 0;
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
    double compose_reduce_ms_per_step = 0.0;
    double compose_copy_ms_per_step = 0.0;
    double compose_final_ms_per_step = 0.0;
    double hc_current_input_ms_per_step = 0.0;
    double hc_current_seed_ms_per_step = 0.0;
    double hc_current_attn_mix_ms_per_step = 0.0;
    double hc_current_split_ms_per_step = 0.0;
    double hc_current_gather_ms_per_step = 0.0;
    double hc_current_ffn_router_ms_per_step = 0.0;
    double hc_current_ffn_norm_ms_per_step = 0.0;
    double hc_current_router_select_ms_per_step = 0.0;
    double hc_current_router_d2h_ms_per_step = 0.0;
    double hc_current_route_upload_ms_per_step = 0.0;
    double hc_current_fill_pack_ms_per_step = 0.0;
    double pre_ep_hc_current_ms_per_step = 0.0;
    double pre_ep_attention_projection_ms_per_step = 0.0;
    double pre_ep_compressed_kv_ms_per_step = 0.0;
    double pre_ep_attention_state_ms_per_step = 0.0;
    double pre_ep_typed_history_ms_per_step = 0.0;
    double pre_ep_raw_read_ms_per_step = 0.0;
    double pre_ep_attention_output_ms_per_step = 0.0;
    double pre_ep_post_attention_ffn_input_ms_per_step = 0.0;
    double final_hc_ms_per_step = 0.0;
    int cudagraph_sync_all_calls = 0;
    int cudagraph_event_barrier_calls = 0;
    int cudagraph_rank_stream_syncs = 0;
    int cudagraph_dense_stream_syncs = 0;
    int cudagraph_copy_stream_syncs = 0;
    int cudagraph_capture_attempted = 0;
    int cudagraph_capture_succeeded = 0;
    int cudagraph_capture_error = 0;
    size_t cudagraph_capture_nodes = 0;
    int cudagraph_replay_attempted = 0;
    int cudagraph_replay_succeeded = 0;
    int cudagraph_replay_error = 0;
    int cudagraph_persistent_cache_hits = 0;
    int cudagraph_persistent_cache_misses = 0;
    int cudagraph_persistent_invalidations = 0;
    int cudagraph_persistent_invalidate_layer = 0;
    int cudagraph_persistent_invalidate_slots = 0;
    int cudagraph_persistent_invalidate_position = 0;
    int cudagraph_persistent_invalidate_root_device = 0;
    int cudagraph_persistent_invalidate_root_stream = 0;
    double cudagraph_instantiate_ms = 0.0;
    double cudagraph_replay_ms = 0.0;
    int finite_bad = 0;
    uint64_t checksum = 0;
    bool ep_return_fp16 = false;
    bool fused_compose_sum = false;
    bool dense_hmma_compose = false;
    bool dense_f16_cublas_compose = false;
    bool dense_f16_cache_compose = false;
    bool nccl_reduce_scatter_compose = false;
};

struct HcCurrentInputBreakdown {
    double seed_ms = 0.0;
    double attn_mix_ms = 0.0;
    double split_ms = 0.0;
    double gather_ms = 0.0;
    double ffn_router_ms = 0.0;
    double ffn_norm_ms = 0.0;
    double router_select_ms = 0.0;
    double router_d2h_ms = 0.0;
    double route_upload_ms = 0.0;
    double fill_pack_ms = 0.0;
};

struct PreEpPrefixBreakdown {
    double hc_current_ms = 0.0;
    double attention_projection_ms = 0.0;
    double compressed_kv_ms = 0.0;
    double attention_state_ms = 0.0;
    double typed_history_ms = 0.0;
    double raw_read_ms = 0.0;
    double attention_output_ms = 0.0;
    double post_attention_ffn_input_ms = 0.0;
};

struct Options {
    const char *lib_path = "./build/turbomind-v100/libggml-turbomind.so";
    const char *pack_dir = nullptr;
    const char *contract_path = nullptr;
    const char *tm_index_path = nullptr;
    const char *tokenizer_model_path = nullptr;
    int devices[kGpus] = {0, 1, 2, 3, 4, 5, 6, 7};
    int slots = 32;
    int top_k = 6;
    int layer = 2;
    int resident_profile_layer = -1;
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
    bool share_tp_runtime = false;
    bool tp_runtime_explicit = false;
    bool tp_runtime_skip_unused_comp_state = false;
    uint64_t tp_runtime_scratch_mib = 1536;
    bool share_expert_bindings = true;
    bool parallel_expert_load_gate = false;
    bool overlap_ep_dense = true;
    bool direct_remote_compose = false;
    bool source_copy_schedule = true;
    bool copy_event_compose = false;
    bool compact_route_compose = false;
    bool token_major_all_layers = false;
    bool share_dense_ops = false;
    bool skip_self_compose_copy = true;
    bool multi_copy_streams = false;
    bool nccl_reduce_scatter_compose_gate = false;
    bool defer_nccl_init_gate = false;
    bool serving_bench = false;
    bool skip_decode_checksum = false;
    bool serve_http = false;
    const char *host = "127.0.0.1";
    int port = 18082;
    int max_requests = 0;
    int microbatch_wait_us = 5000;
    bool output_head_gate = false;
    bool output_head_resident_gate = false;
    bool decode_cudagraph_gate = false;
    bool decode_cudagraph_replay_probe_gate = false;
    bool decode_cudagraph_persistent_replay_gate = false;
    bool decode_cudagraph_output_sync_gate = false;
    bool decode_cudagraph_hc_current_sync_gate = false;
    const char *decode_cudagraph_stage_sync = nullptr;
    const char *decode_cudagraph_suffix_stage = nullptr;
    bool decode_stage_checksum_gate = false;
    bool compact_moe_decode_gate = false;
    bool fused_gated_silu_gate = false;
    bool final_hc_carry_gate = false;
    bool diagnostic_output_head = false;
    bool diagnostic_output_head_lazy_gate = false;
    bool tp_hc_final_expand_gate = false;
    bool tp_hc_current_input_gate = false;
    bool tp_hc_current_input_peer_gather_gate = false;
    bool tp_hc_current_input_nccl_allgather_gate = false;
    bool tp_hc_current_allreduce_gate = false;
    bool tp_hc_current_input_stream_sync_gate = false;
    bool tp_hc_current_input_fused_fill_pack_gate = false;
    bool tp_hc_current_full_parity_gate = false;
    bool tp_hc_persist_state_gate = false;
    bool tp_peer_accounting_gate = false;
    bool tp_peer_reject_sys_gate = false;
    bool model_router_routes = false;
    bool router_cublas_gate = false;
    bool router_hash_fast_gate = false;
    bool gpu_route_plan_gate = false;
    bool route_plan_async_upload_gate = false;
    bool routed_ffn_norm_input_gate = false;
    bool routed_ffn_rank_major_input_gate = false;
    bool routed_ffn_rank_major_shared_input_gate = false;
    bool routed_ffn_rank_major_route_input_gate = false;
    bool routed_ffn_rank_major_input_parity_gate = false;
    bool post_attention_route_reuse_audit_gate = false;
    bool post_attention_fixed_capacity_route_plan_gate = false;
    bool post_attention_device_actual_route_sync_gate = false;
    int post_attention_static_rank_route_cap = 0;
    int post_attention_static_executor_route_cap = 0;
    int post_attention_static_compose_route_cap = 0;
    bool post_attention_masked_compact_copy_gate = false;
    bool post_attention_slot_major_ffn_norm_gate = false;
    bool post_attention_skip_slot_major_ffn_norm_gate = false;
    bool model_router_rank_major_logits_gate = false;
    bool model_router_allreduce_logits_gate = false;
    bool true_shared_ffn_gate = false;
    bool tp_kv_all_slots_gate = false;
    bool reference_hc_reduce_gate = false;
    bool reference_hc_state_guard_gate = false;
    bool true_ds4_attention_residency_gate = false;
    bool true_ds4_attention_projection_gate = false;
    bool true_ds4_attention_projection_direct_input_fill_gate = false;
    bool true_ds4_attention_projection_rank_local_input_gate = false;
    bool true_ds4_attention_projection_rank_major_input_gate = false;
    bool true_ds4_attention_projection_input_parity_gate = false;
    bool true_ds4_attention_state_gate = false;
    bool true_ds4_attention_rope_gate = false;
    bool true_ds4_attention_saturation_audit_gate = false;
    bool true_ds4_attention_kv_norm_reference_gate = false;
    bool true_ds4_attention_raw_read_gate = false;
    bool true_ds4_attention_raw_window_gate = false;
    bool true_ds4_attention_typed_kv_raw_gate = false;
    bool true_ds4_attention_typed_kv_compressed_gate = false;
    bool true_ds4_attention_typed_kv_indexer_gate = false;
    bool true_ds4_attention_typed_kv_history_gate = false;
    bool true_ds4_attention_typed_kv_skip_current_load_gate = false;
    bool true_ds4_attention_typed_kv_skip_raw_store_gate = false;
    bool true_ds4_attention_typed_kv_skip_compressed_store_gate = false;
    bool true_ds4_attention_typed_kv_skip_indexer_store_gate = false;
    bool true_ds4_attention_typed_kv_quiet_gate = false;
    bool true_ds4_attention_typed_kv_batch_rows_gate = false;
    bool true_ds4_attention_typed_kv_stream_sync_gate = false;
    bool fp8_e5m2_kv_gate = false;
    bool true_ds4_attention_output_gate = false;
    bool true_ds4_attention_output_nccl_allgather_gate = false;
    bool true_ds4_post_attention_ffn_input_gate = false;
    bool true_ds4_semantic_skip_stats_gate = false;
    bool true_ds4_compressed_kv_gate = false;
    bool true_ds4_indexer_attention_gate = false;
    bool true_ds4_compressed_kv_direct_input_fill_gate = false;
    bool true_ds4_compressed_kv_dense_event_wait_gate = false;
    bool true_ds4_compressed_kv_skip_dense_stats_gate = false;
    bool true_ds4_compressed_kv_fused_attn_input_fill_gate = false;
    bool true_ds4_compressed_kv_fused_input_fill_gate = false;
    bool true_ds4_compressed_kv_fused_rope_round_gate = false;
    bool true_ds4_compressed_kv_fused_pool_norm_gate = false;
    bool true_ds4_compressed_kv_fused_pool_norm_rope_round_gate = false;
    bool true_ds4_compressed_reference_diff_gate = false;
    bool cuda_profiler_window = false;
    bool cuda_profiler_all_devices = false;
    int cuda_profiler_device = -1;
    uint32_t true_ds4_attention_raw_valid_rows = 1;
    uint64_t vram_min_free_mib = 0;
    uint64_t nccl_min_free_mib = 0;
    bool vram_report = false;
};

void log_hc_current_full_rank_parity(const Options &opt,
                                     RankState ranks[kGpus],
                                     int layer,
                                     size_t elems);
int nccl_broadcast_f32_from_device0_to_current_full(
    const Options &opt,
    RankState ranks[kGpus],
    const float *src_device0,
    uint64_t elems,
    const char *label);

bool tp_ep_profiler_start_if_requested(const Options &opt) {
    if (!opt.cuda_profiler_window) return false;
    if (!opt.cuda_profiler_all_devices) {
        if (opt.cuda_profiler_device >= 0) {
            const cudaError_t set_err = cudaSetDevice(opt.cuda_profiler_device);
            if (set_err != cudaSuccess) {
                std::fprintf(stderr,
                             "tp_ep_cuda_profiler_start_failed\tdevice\t%d\tphase\tset_device\terr\t%s\n",
                             opt.cuda_profiler_device, cudaGetErrorString(set_err));
                return false;
            }
        }
        const cudaError_t err = cudaProfilerStart();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "tp_ep_cuda_profiler_start_failed\terr\t%s\n",
                         cudaGetErrorString(err));
            return false;
        }
        std::fprintf(stderr, "tp_ep_cuda_profiler_window\tstate\tstart\tdevice\t%d\n",
                     opt.cuda_profiler_device);
        return true;
    }
    bool active = false;
    for (int rank = 0; rank < kGpus; ++rank) {
        cudaError_t err = cudaSetDevice(opt.devices[rank]);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "tp_ep_cuda_profiler_start_failed\tdevice\t%d\tphase\tset_device\terr\t%s\n",
                         opt.devices[rank], cudaGetErrorString(err));
            continue;
        }
        err = cudaProfilerStart();
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "tp_ep_cuda_profiler_start_failed\tdevice\t%d\tphase\tstart\terr\t%s\n",
                         opt.devices[rank], cudaGetErrorString(err));
            continue;
        }
        active = true;
    }
    std::fprintf(stderr, "tp_ep_cuda_profiler_window\tstate\tstart\tdevices\t%d\n",
                 active ? kGpus : 0);
    return active;
}

int tp_ep_profiler_stop_if_active(const Options &opt, bool *active) {
    if (!active || !*active) return 0;
    if (!opt.cuda_profiler_all_devices) {
        if (opt.cuda_profiler_device >= 0) {
            const cudaError_t set_err = cudaSetDevice(opt.cuda_profiler_device);
            if (set_err != cudaSuccess) {
                std::fprintf(stderr,
                             "tp_ep_cuda_profiler_stop_failed\tdevice\t%d\tphase\tset_device\terr\t%s\n",
                             opt.cuda_profiler_device, cudaGetErrorString(set_err));
                *active = false;
                return 1;
            }
        }
        const cudaError_t err = cudaProfilerStop();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "tp_ep_cuda_profiler_stop_failed\terr\t%s\n",
                         cudaGetErrorString(err));
            *active = false;
            return 1;
        }
        std::fprintf(stderr, "tp_ep_cuda_profiler_window\tstate\tstop\n");
        *active = false;
        return 0;
    }
    int failures = 0;
    for (int rank = 0; rank < kGpus; ++rank) {
        cudaError_t err = cudaSetDevice(opt.devices[rank]);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "tp_ep_cuda_profiler_stop_failed\tdevice\t%d\tphase\tset_device\terr\t%s\n",
                         opt.devices[rank], cudaGetErrorString(err));
            failures++;
            continue;
        }
        err = cudaProfilerStop();
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "tp_ep_cuda_profiler_stop_failed\tdevice\t%d\tphase\tstop\terr\t%s\n",
                         opt.devices[rank], cudaGetErrorString(err));
            failures++;
        }
    }
    std::fprintf(stderr, "tp_ep_cuda_profiler_window\tstate\tstop\tfailures\t%d\n",
                 failures);
    *active = false;
    return failures == 0 ? 0 : 1;
}

struct TpEpProfilerWindowGuard {
    bool active = false;
    const Options &opt;

    explicit TpEpProfilerWindowGuard(const Options &opt)
        : active(tp_ep_profiler_start_if_requested(opt)), opt(opt) {}

    ~TpEpProfilerWindowGuard() {
        (void)tp_ep_profiler_stop_if_active(opt, &active);
    }
};

int enqueue_cross_gpu_stream_barrier(RankState ranks[kGpus],
                                     bool include_copy_streams);

void sync_typed_kv_boundary(const Options &opt, RankState ranks[kGpus]) {
    if (opt.decode_cudagraph_gate) {
        const int rc = enqueue_cross_gpu_stream_barrier(ranks, false);
        if (rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_typed_kv_graph_boundary_failed\trc\t%d\n",
                         rc);
            std::abort();
        }
        return;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        if (opt.true_ds4_attention_typed_kv_stream_sync_gate) {
            CHECK_CUDA(cudaStreamSynchronize(0));
        } else {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }
}

struct TensorF32Stats {
    int finite_bad = 0;
    size_t first_bad = (size_t)-1;
    float max_abs = 0.0f;
};

struct TensorF32DiffStats {
    int bad = 0;
    size_t first_bad = (size_t)-1;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
};

struct HalfInputDiffStats {
    unsigned long long compared = 0;
    unsigned long long mismatches = 0;
    int first_mismatch = -1;
    float max_abs = 0.0f;
};

TensorF32Stats collect_tensor_f32_stats(const float *ptr, size_t elems,
                                        cudaStream_t stream);
TensorF32Stats collect_raw_swa_row_stats(const float *ptr, uint32_t slots,
                                         uint32_t raw_rows, uint32_t raw_row,
                                         uint32_t head_dim,
                                         cudaStream_t stream);
TensorF32DiffStats collect_tensor_f32_diff_stats(const float *a, const float *b,
                                                 size_t elems,
                                                 cudaStream_t stream);
void merge_tensor_stats(TensorF32Stats *dst, const TensorF32Stats &src);
void log_tensor_f32_stats(const char *tag, int layer, int rank_id,
                          const float *ptr, size_t elems, cudaStream_t stream);
bool should_log_routed_semantic_stats(const Options &opt);
bool should_log_reference_hc_window(const Options &opt);

#include "kernels/v100/common.cuh"
#include "kernels/v100/dense.cuh"
#include "kernels/v100/hc_mix.cuh"
#include "kernels/v100/hc_shards.cuh"
#include "kernels/v100/norm.cuh"
#include "kernels/v100/compose.cuh"
#include "kernels/v100/router.cuh"
#include "kernels/v100/diagnostics.cuh"
#include "kernels/v100/fill_pack.cuh"
#include "kernels/v100/attention.cuh"

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

constexpr uint64_t kMiB = 1024ull * 1024ull;

bool should_report_vram(const Options &opt) {
    return opt.vram_report || opt.vram_min_free_mib > 0;
}

bool nccl_gate_active(const Options &opt) {
    return opt.nccl_reduce_scatter_compose_gate ||
           opt.tp_hc_current_input_nccl_allgather_gate ||
           opt.tp_hc_current_allreduce_gate ||
           opt.true_ds4_attention_output_nccl_allgather_gate;
}

int report_vram_checkpoint_min_free(const Options &opt,
                                    const char *label,
                                    uint64_t min_free_mib_threshold) {
    const uint64_t min_free_bytes = min_free_mib_threshold * kMiB;
    uint64_t min_free_mib = UINT64_MAX;
    uint64_t max_used_mib = 0;
    int failures = 0;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        size_t free_b = 0;
        size_t total_b = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
        const uint64_t used_b = (uint64_t)total_b - (uint64_t)free_b;
        const uint64_t free_mib = (uint64_t)free_b / kMiB;
        const uint64_t used_mib = used_b / kMiB;
        const uint64_t total_mib = (uint64_t)total_b / kMiB;
        min_free_mib = std::min(min_free_mib, free_mib);
        max_used_mib = std::max(max_used_mib, used_mib);
        const bool pass =
            min_free_mib_threshold == 0 || (uint64_t)free_b >= min_free_bytes;
        if (!pass) failures++;
        std::printf("tp_ep_vram\tlabel\t%s\tgpu\t%d\tfree_mib\t%llu\t"
                    "used_mib\t%llu\ttotal_mib\t%llu\tmin_free_mib\t%llu\t%s\n",
                    label, gpu,
                    (unsigned long long)free_mib,
                    (unsigned long long)used_mib,
                    (unsigned long long)total_mib,
                    (unsigned long long)min_free_mib_threshold,
                    pass ? "PASS" : "FAIL");
    }
    if (min_free_mib == UINT64_MAX) min_free_mib = 0;
    std::printf("tp_ep_vram_summary\tlabel\t%s\tmin_free_mib\t%llu\t"
                "max_used_mib\t%llu\tthreshold_mib\t%llu\tfailures\t%d\t%s\n",
                label,
                (unsigned long long)min_free_mib,
                (unsigned long long)max_used_mib,
                (unsigned long long)min_free_mib_threshold,
                failures,
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

int report_vram_checkpoint(const Options &opt, const char *label) {
    if (!should_report_vram(opt)) return 0;
    return report_vram_checkpoint_min_free(opt, label, opt.vram_min_free_mib);
}

int report_nccl_vram_checkpoint(const Options &opt, const char *label) {
    if (!nccl_gate_active(opt) || opt.nccl_min_free_mib == 0) return 0;
    return report_vram_checkpoint_min_free(opt, label, opt.nccl_min_free_mib);
}

int check_planned_vram_allocation(const Options &opt,
                                  const char *label,
                                  const uint64_t planned_bytes[kGpus]) {
    if (!should_report_vram(opt)) return 0;
    const uint64_t min_free_bytes = opt.vram_min_free_mib * kMiB;
    int failures = 0;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        size_t free_b = 0;
        size_t total_b = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
        const uint64_t required_b = planned_bytes[gpu] + min_free_bytes;
        const bool pass = (uint64_t)free_b >= required_b;
        if (!pass) failures++;
        std::printf("tp_ep_vram_plan\tlabel\t%s\tgpu\t%d\tfree_mib\t%llu\t"
                    "planned_mib\t%llu\tthreshold_mib\t%llu\ttotal_mib\t%llu\t%s\n",
                    label, gpu,
                    (unsigned long long)((uint64_t)free_b / kMiB),
                    (unsigned long long)(planned_bytes[gpu] / kMiB),
                    (unsigned long long)opt.vram_min_free_mib,
                    (unsigned long long)((uint64_t)total_b / kMiB),
                    pass ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}

void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s --pack-dir DIR --contract FILE --tm-index FILE [options]\n"
                 "       [--lib PATH] [--tokenizer-model PATH]\n"
                 "       [--devices 0,1,2,3,4,5,6,7]\n"
                 "       [--slots N] [--top-k N] [--layer N] [--resident-profile-layer N]\n"
                 "       [--kv-slot N]\n"
                 "       [--position N] [--warmup N] [--iters N]\n"
                 "       [--dense-compute-tensor NAME] [--dense-compute-all-f8]\n"
                 "       [--dense-compute-all-bf16] [--dense-compute-all]\n"
                 "       [--compose-next-hidden] [--decode-steps N]\n"
                 "       [--ep-return-fp16] [--fuse-compose-sum]\n"
                 "       [--dense-hmma-compose] [--dense-f16-cublas-compose]\n"
                 "       [--dense-f16-cache-compose] [--all-layers]\n"
                 "       [--skip-descriptor-checks] [--skip-predecode-probes]\n"
                 "       [--share-tp-runtime] [--local-tp-runtime]\n"
                 "       [--tp-runtime-scratch-mib N]\n"
                 "       [--shared-expert-bindings] [--local-expert-bindings]\n"
                 "       [--parallel-expert-load-gate]\n"
                 "       [--overlap-ep-dense] [--serial-ep-dense]\n"
                 "       [--direct-remote-compose]\n"
                 "       [--source-copy-schedule] [--dest-copy-schedule]\n"
                 "       [--copy-event-compose]\n"
                 "       [--compact-route-compose] [--compact-moe-decode-gate]\n"
                 "       [--fused-gated-silu-gate]\n"
                 "       [--fp8-e5m2-kv-gate]\n"
                 "       [--token-major-all-layers] [--shared-dense-ops]\n"
                 "       [--skip-self-compose-copy] [--copy-self-compose]\n"
                 "       [--multi-copy-streams]\n"
                 "       [--nccl-reduce-scatter-compose-gate] [--serving-bench]\n"
                 "       [--defer-nccl-init-gate]\n"
                 "       [--skip-decode-checksum]\n"
                 "       [--serve-http] [--host ADDR] [--port N] [--max-requests N]\n"
                 "       [--microbatch-wait-us N]\n"
                 "       [--vram-report] [--vram-min-free-mib N]\n"
                 "       [--nccl-min-free-mib N]\n"
                 "       [--output-head-gate] [--output-head-resident-gate]\n"
                 "       [--diagnostic-output-head-lazy-gate]\n"
                 "       [--decode-cudagraph-output-sync-gate]\n"
                 "       [--decode-cudagraph-hc-current-sync-gate]\n"
                 "       [--decode-cudagraph-stage-sync-gate STAGES]\n"
                 "       [--decode-cudagraph-suffix-stage-gate STAGE]\n"
                 "       [--decode-stage-checksum-gate]\n"
                 "       [--final-hc-carry-gate] [--tp-hc-final-expand-gate]\n"
                 "       [--tp-hc-current-input-gate]\n"
                 "       [--tp-hc-current-input-peer-gather-gate]\n"
                 "       [--tp-hc-current-input-nccl-allgather-gate]\n"
                 "       [--tp-hc-current-allreduce-gate]\n"
                 "       [--tp-hc-current-input-stream-sync-gate]\n"
                 "       [--tp-hc-current-input-fused-fill-pack-gate]\n"
                 "       [--tp-hc-current-full-parity-gate]\n"
                 "       [--tp-hc-persist-state-gate] [--tp-kv-all-slots-gate]\n"
                 "       [--tp-peer-accounting-gate] [--tp-peer-reject-sys-gate]\n"
                 "       [--model-router-routes]\n"
                 "       [--routed-ffn-norm-input-gate]\n"
                 "       [--routed-ffn-rank-major-input-gate]\n"
                 "       [--routed-ffn-rank-major-shared-input-gate]\n"
                 "       [--routed-ffn-rank-major-route-input-gate]\n"
                 "       [--routed-ffn-rank-major-input-parity-gate]\n"
                 "       [--post-attention-route-reuse-audit-gate]\n"
                 "       [--post-attention-fixed-capacity-route-plan-gate]\n"
                 "       [--post-attention-static-rank-route-cap N]\n"
                 "       [--post-attention-static-executor-route-cap N]\n"
                 "       [--post-attention-static-compose-route-cap N]\n"
                 "       [--post-attention-masked-compact-copy-gate]\n"
                 "       [--post-attention-skip-slot-major-ffn-norm-gate]\n"
                 "       [--model-router-rank-major-logits-gate]\n"
                 "       [--model-router-allreduce-logits-gate]\n"
                 "       [--true-shared-ffn-gate]\n"
                 "       [--true-ds4-attention-residency-gate]\n"
                 "       [--true-ds4-attention-projection-gate]\n"
                 "       [--true-ds4-attention-projection-rank-local-input-gate]\n"
                 "       [--true-ds4-attention-projection-rank-major-input-gate]\n"
                 "       [--true-ds4-attention-projection-input-parity-gate]\n"
                 "       [--true-ds4-attention-state-gate]\n"
                 "       [--true-ds4-attention-rope-gate]\n"
                 "       [--true-ds4-attention-saturation-audit-gate]\n"
                 "       [--true-ds4-attention-kv-norm-reference-gate]\n"
                 "       [--true-ds4-attention-raw-read-gate]\n"
                 "       [--true-ds4-attention-raw-window-gate]\n"
                 "       [--true-ds4-attention-typed-kv-raw-gate]\n"
                 "       [--true-ds4-attention-typed-kv-compressed-gate]\n"
                 "       [--true-ds4-attention-typed-kv-indexer-gate]\n"
                 "       [--true-ds4-attention-typed-kv-history-gate]\n"
                 "       [--true-ds4-attention-typed-kv-skip-current-load-gate]\n"
                 "       [--true-ds4-attention-typed-kv-skip-raw-store-gate]\n"
                 "       [--true-ds4-attention-typed-kv-skip-compressed-store-gate]\n"
                 "       [--true-ds4-attention-typed-kv-skip-indexer-store-gate]\n"
                 "       [--true-ds4-attention-typed-kv-quiet-gate]\n"
                 "       [--true-ds4-attention-typed-kv-batch-rows-gate]\n"
                 "       [--true-ds4-attention-typed-kv-stream-sync-gate]\n"
                 "       [--true-ds4-attention-output-gate]\n"
                 "       [--true-ds4-attention-output-nccl-allgather-gate]\n"
                 "       [--true-ds4-post-attention-ffn-input-gate]\n"
                 "       [--true-ds4-compressed-kv-gate]\n"
                 "       [--true-ds4-indexer-attention-gate]\n"
                 "       [--true-ds4-compressed-kv-dense-event-wait-gate]\n"
                 "       [--true-ds4-compressed-kv-fused-input-fill-gate]\n"
                 "       [--true-ds4-compressed-kv-fused-rope-round-gate]\n"
                 "       [--true-ds4-compressed-kv-fused-pool-norm-gate]\n"
                 "       [--true-ds4-compressed-reference-diff-gate]\n"
                 "       [--reference-hc-reduce-gate]\n"
                 "       [--reference-hc-state-guard-gate]\n"
                 "       [--cuda-profiler-window] [--cuda-profiler-device N]\n"
                 "       [--cuda-profiler-all-devices]\n"
                 "       [--decode-cudagraph-gate]\n"
                 "       [--decode-cudagraph-replay-probe-gate]\n"
                 "       [--decode-cudagraph-persistent-replay-gate]\n"
                 "       [--router-cublas-gate]\n"
                 "       [--router-hash-fast-gate]\n"
                 "       [--gpu-route-plan-gate]\n"
                 "       [--route-plan-async-upload-gate]\n"
                 "       [--diagnostic-output-head]\n"
                 "       [--diagnostic-output-head-lazy-gate]\n",
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
        } else if (std::strcmp(arg, "--tokenizer-model") == 0) {
            if (!val) return false;
            opt->tokenizer_model_path = val;
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
        } else if (std::strcmp(arg, "--resident-profile-layer") == 0) {
            if (!val || !parse_int(val, &opt->resident_profile_layer) ||
                opt->resident_profile_layer < 0 || opt->resident_profile_layer >= 43) {
                return false;
            }
            opt->layer = opt->resident_profile_layer;
            opt->all_layers = true;
            opt->share_tp_runtime = true;
            opt->tp_runtime_explicit = true;
            opt->share_expert_bindings = true;
            opt->share_dense_ops = true;
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
        } else if (std::strcmp(arg, "--share-tp-runtime") == 0) {
            opt->share_tp_runtime = true;
            opt->tp_runtime_explicit = true;
        } else if (std::strcmp(arg, "--local-tp-runtime") == 0) {
            opt->share_tp_runtime = false;
            opt->tp_runtime_explicit = true;
        } else if (std::strcmp(arg, "--tp-runtime-skip-unused-comp-state-gate") == 0) {
            opt->tp_runtime_skip_unused_comp_state = true;
        } else if (std::strcmp(arg, "--tp-runtime-scratch-mib") == 0) {
            if (!val || !parse_u64(val, &opt->tp_runtime_scratch_mib) ||
                opt->tp_runtime_scratch_mib < 64 || opt->tp_runtime_scratch_mib > 4096) {
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--shared-expert-bindings") == 0) {
            opt->share_expert_bindings = true;
        } else if (std::strcmp(arg, "--local-expert-bindings") == 0) {
            opt->share_expert_bindings = false;
        } else if (std::strcmp(arg, "--parallel-expert-load-gate") == 0) {
            opt->parallel_expert_load_gate = true;
        } else if (std::strcmp(arg, "--overlap-ep-dense") == 0) {
            opt->overlap_ep_dense = true;
        } else if (std::strcmp(arg, "--serial-ep-dense") == 0) {
            opt->overlap_ep_dense = false;
        } else if (std::strcmp(arg, "--direct-remote-compose") == 0) {
            opt->direct_remote_compose = true;
        } else if (std::strcmp(arg, "--source-copy-schedule") == 0) {
            opt->source_copy_schedule = true;
        } else if (std::strcmp(arg, "--dest-copy-schedule") == 0) {
            opt->source_copy_schedule = false;
        } else if (std::strcmp(arg, "--copy-event-compose") == 0) {
            opt->copy_event_compose = true;
        } else if (std::strcmp(arg, "--compact-route-compose") == 0) {
            opt->compact_route_compose = true;
        } else if (std::strcmp(arg, "--compact-moe-decode-gate") == 0) {
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
        } else if (std::strcmp(arg, "--fused-gated-silu-gate") == 0) {
            opt->fused_gated_silu_gate = true;
        } else if (std::strcmp(arg, "--token-major-all-layers") == 0) {
            opt->token_major_all_layers = true;
        } else if (std::strcmp(arg, "--shared-dense-ops") == 0) {
            opt->share_dense_ops = true;
        } else if (std::strcmp(arg, "--skip-self-compose-copy") == 0) {
            opt->skip_self_compose_copy = true;
        } else if (std::strcmp(arg, "--copy-self-compose") == 0) {
            opt->skip_self_compose_copy = false;
        } else if (std::strcmp(arg, "--multi-copy-streams") == 0) {
            opt->multi_copy_streams = true;
        } else if (std::strcmp(arg, "--nccl-reduce-scatter-compose-gate") == 0) {
            opt->nccl_reduce_scatter_compose_gate = true;
        } else if (std::strcmp(arg, "--defer-nccl-init-gate") == 0) {
            opt->defer_nccl_init_gate = true;
        } else if (std::strcmp(arg, "--serving-bench") == 0) {
            opt->serving_bench = true;
        } else if (std::strcmp(arg, "--skip-decode-checksum") == 0) {
            opt->skip_decode_checksum = true;
        } else if (std::strcmp(arg, "--serve-http") == 0) {
            opt->serve_http = true;
            opt->serving_bench = true;
            opt->token_major_all_layers = true;
            opt->all_layers = true;
            opt->share_tp_runtime = true;
            opt->tp_runtime_explicit = true;
            opt->skip_decode_checksum = true;
        } else if (std::strcmp(arg, "--host") == 0) {
            if (!val) return false;
            opt->host = val;
            ++i;
        } else if (std::strcmp(arg, "--port") == 0) {
            if (!val || !parse_int(val, &opt->port) || opt->port <= 0 || opt->port > 65535) return false;
            ++i;
        } else if (std::strcmp(arg, "--max-requests") == 0) {
            if (!val || !parse_int(val, &opt->max_requests) || opt->max_requests < 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--microbatch-wait-us") == 0) {
            if (!val || !parse_int(val, &opt->microbatch_wait_us) ||
                opt->microbatch_wait_us < 0 || opt->microbatch_wait_us > 1000000) return false;
            ++i;
        } else if (std::strcmp(arg, "--vram-report") == 0) {
            opt->vram_report = true;
        } else if (std::strcmp(arg, "--vram-min-free-mib") == 0) {
            if (!val || !parse_u64(val, &opt->vram_min_free_mib)) return false;
            ++i;
        } else if (std::strcmp(arg, "--nccl-min-free-mib") == 0) {
            if (!val || !parse_u64(val, &opt->nccl_min_free_mib)) return false;
            ++i;
        } else if (std::strcmp(arg, "--output-head-gate") == 0) {
            opt->output_head_gate = true;
        } else if (std::strcmp(arg, "--output-head-resident-gate") == 0) {
            opt->output_head_resident_gate = true;
        } else if (std::strcmp(arg, "--decode-cudagraph-gate") == 0) {
            opt->decode_cudagraph_gate = true;
        } else if (std::strcmp(arg, "--decode-cudagraph-replay-probe-gate") == 0) {
            opt->decode_cudagraph_gate = true;
            opt->decode_cudagraph_replay_probe_gate = true;
        } else if (std::strcmp(arg, "--decode-cudagraph-persistent-replay-gate") == 0) {
            opt->decode_cudagraph_gate = true;
            opt->decode_cudagraph_replay_probe_gate = true;
            opt->decode_cudagraph_persistent_replay_gate = true;
        } else if (std::strcmp(arg, "--decode-cudagraph-output-sync-gate") == 0) {
            opt->decode_cudagraph_gate = true;
            opt->decode_cudagraph_output_sync_gate = true;
        } else if (std::strcmp(arg, "--decode-cudagraph-hc-current-sync-gate") == 0) {
            opt->decode_cudagraph_gate = true;
            opt->decode_cudagraph_hc_current_sync_gate = true;
        } else if (std::strcmp(arg, "--decode-cudagraph-stage-sync-gate") == 0) {
            if (i + 1 >= argc) return false;
            opt->decode_cudagraph_gate = true;
            opt->decode_cudagraph_stage_sync = argv[++i];
        } else if (std::strcmp(arg, "--decode-cudagraph-suffix-stage-gate") == 0) {
            if (i + 1 >= argc) return false;
            const char *stage = argv[++i];
            if (std::strcmp(stage, "routed_ffn") != 0 &&
                std::strcmp(stage, "dense") != 0 &&
                std::strcmp(stage, "compose") != 0 &&
                std::strcmp(stage, "final_hc") != 0 &&
                std::strcmp(stage, "compose_eager_final_hc") != 0) {
                return false;
            }
            opt->decode_cudagraph_suffix_stage = stage;
        } else if (std::strcmp(arg, "--decode-stage-checksum-gate") == 0) {
            opt->decode_stage_checksum_gate = true;
        } else if (std::strcmp(arg, "--final-hc-carry-gate") == 0) {
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-final-expand-gate") == 0) {
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-gate") == 0) {
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-peer-gather-gate") == 0) {
            opt->tp_hc_current_input_peer_gather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-nccl-allgather-gate") == 0) {
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-allreduce-gate") == 0) {
            opt->tp_hc_current_allreduce_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-stream-sync-gate") == 0) {
            opt->tp_hc_current_input_stream_sync_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-fused-fill-pack-gate") == 0) {
            opt->tp_hc_current_input_fused_fill_pack_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-full-parity-gate") == 0) {
            opt->tp_hc_current_full_parity_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-peer-accounting-gate") == 0) {
            opt->tp_peer_accounting_gate = true;
        } else if (std::strcmp(arg, "--tp-peer-reject-sys-gate") == 0) {
            opt->tp_peer_reject_sys_gate = true;
            opt->tp_peer_accounting_gate = true;
        } else if (std::strcmp(arg, "--model-router-routes") == 0) {
            opt->model_router_routes = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--router-cublas-gate") == 0) {
            opt->router_cublas_gate = true;
            opt->model_router_routes = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--router-hash-fast-gate") == 0) {
            opt->router_hash_fast_gate = true;
            opt->model_router_routes = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--gpu-route-plan-gate") == 0) {
            opt->gpu_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--route-plan-async-upload-gate") == 0) {
            opt->route_plan_async_upload_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--routed-ffn-norm-input-gate") == 0) {
            opt->routed_ffn_norm_input_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--routed-ffn-rank-major-input-gate") == 0) {
            opt->routed_ffn_rank_major_input_gate = true;
            opt->routed_ffn_rank_major_shared_input_gate = true;
            opt->routed_ffn_rank_major_route_input_gate = true;
            opt->routed_ffn_norm_input_gate = true;
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--routed-ffn-rank-major-shared-input-gate") == 0) {
            opt->routed_ffn_rank_major_shared_input_gate = true;
            opt->routed_ffn_norm_input_gate = true;
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--routed-ffn-rank-major-route-input-gate") == 0) {
            opt->routed_ffn_rank_major_route_input_gate = true;
            opt->routed_ffn_norm_input_gate = true;
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--routed-ffn-rank-major-input-parity-gate") == 0) {
            opt->routed_ffn_rank_major_input_parity_gate = true;
            opt->routed_ffn_norm_input_gate = true;
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-route-reuse-audit-gate") == 0) {
            opt->post_attention_route_reuse_audit_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-fixed-capacity-route-plan-gate") == 0) {
            opt->post_attention_fixed_capacity_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-device-actual-route-sync-gate") == 0) {
            opt->post_attention_device_actual_route_sync_gate = true;
            opt->post_attention_fixed_capacity_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-static-rank-route-cap") == 0) {
            if (!val || !parse_int(val, &opt->post_attention_static_rank_route_cap) ||
                opt->post_attention_static_rank_route_cap <= 0) {
                return false;
            }
            ++i;
            opt->post_attention_fixed_capacity_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-static-executor-route-cap") == 0) {
            if (!val || !parse_int(val, &opt->post_attention_static_executor_route_cap) ||
                opt->post_attention_static_executor_route_cap <= 0) {
                return false;
            }
            ++i;
            opt->post_attention_fixed_capacity_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-static-compose-route-cap") == 0) {
            if (!val || !parse_int(val, &opt->post_attention_static_compose_route_cap) ||
                opt->post_attention_static_compose_route_cap <= 0) {
                return false;
            }
            ++i;
            opt->post_attention_fixed_capacity_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-masked-compact-copy-gate") == 0) {
            opt->post_attention_masked_compact_copy_gate = true;
            opt->post_attention_fixed_capacity_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-slot-major-ffn-norm-gate") == 0) {
            opt->post_attention_slot_major_ffn_norm_gate = true;
        } else if (std::strcmp(arg, "--post-attention-skip-slot-major-ffn-norm-gate") == 0) {
            opt->post_attention_skip_slot_major_ffn_norm_gate = true;
        } else if (std::strcmp(arg, "--model-router-rank-major-logits-gate") == 0) {
            opt->model_router_rank_major_logits_gate = true;
            opt->model_router_routes = true;
            opt->routed_ffn_rank_major_input_gate = true;
            opt->routed_ffn_rank_major_shared_input_gate = true;
            opt->routed_ffn_rank_major_route_input_gate = true;
            opt->routed_ffn_norm_input_gate = true;
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--model-router-allreduce-logits-gate") == 0) {
            opt->model_router_allreduce_logits_gate = true;
            opt->model_router_routes = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-shared-ffn-gate") == 0) {
            opt->true_shared_ffn_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-residency-gate") == 0) {
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-projection-gate") == 0) {
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-projection-direct-input-fill-gate") == 0) {
            opt->true_ds4_attention_projection_direct_input_fill_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-projection-rank-local-input-gate") == 0) {
            opt->true_ds4_attention_projection_rank_local_input_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-projection-rank-major-input-gate") == 0) {
            opt->true_ds4_attention_projection_rank_major_input_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-projection-input-parity-gate") == 0) {
            opt->true_ds4_attention_projection_input_parity_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-state-gate") == 0) {
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-rope-gate") == 0) {
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-saturation-audit-gate") == 0) {
            opt->true_ds4_attention_saturation_audit_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-kv-norm-reference-gate") == 0) {
            opt->true_ds4_attention_kv_norm_reference_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-raw-read-gate") == 0) {
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-raw-window-gate") == 0) {
            opt->true_ds4_attention_raw_window_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-raw-gate") == 0) {
            opt->true_ds4_attention_typed_kv_raw_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-compressed-gate") == 0) {
            opt->true_ds4_attention_typed_kv_compressed_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-indexer-gate") == 0) {
            opt->true_ds4_attention_typed_kv_indexer_gate = true;
            opt->true_ds4_indexer_attention_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-history-gate") == 0) {
            opt->true_ds4_attention_typed_kv_history_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-skip-current-load-gate") == 0) {
            opt->true_ds4_attention_typed_kv_skip_current_load_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-skip-raw-store-gate") == 0) {
            opt->true_ds4_attention_typed_kv_skip_raw_store_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-skip-compressed-store-gate") == 0) {
            opt->true_ds4_attention_typed_kv_skip_compressed_store_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-skip-indexer-store-gate") == 0) {
            opt->true_ds4_attention_typed_kv_skip_indexer_store_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-quiet-gate") == 0) {
            opt->true_ds4_attention_typed_kv_quiet_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-batch-rows-gate") == 0) {
            opt->true_ds4_attention_typed_kv_batch_rows_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-stream-sync-gate") == 0) {
            opt->true_ds4_attention_typed_kv_stream_sync_gate = true;
        } else if (std::strcmp(arg, "--fp8-e5m2-kv-gate") == 0) {
            opt->fp8_e5m2_kv_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-output-gate") == 0) {
            opt->true_ds4_attention_output_gate = true;
            opt->true_ds4_attention_raw_window_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-output-nccl-allgather-gate") == 0) {
            opt->true_ds4_attention_output_nccl_allgather_gate = true;
            opt->true_ds4_attention_output_gate = true;
            opt->true_ds4_attention_raw_window_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-post-attention-ffn-input-gate") == 0) {
            opt->true_ds4_post_attention_ffn_input_gate = true;
            opt->true_ds4_attention_output_gate = true;
            opt->true_ds4_attention_raw_window_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->true_shared_ffn_gate = true;
            opt->routed_ffn_norm_input_gate = true;
            opt->model_router_routes = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-semantic-skip-stats-gate") == 0) {
            opt->true_ds4_semantic_skip_stats_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-gate") == 0) {
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-indexer-attention-gate") == 0) {
            opt->true_ds4_indexer_attention_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-direct-input-fill-gate") == 0) {
            opt->true_ds4_compressed_kv_direct_input_fill_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-dense-event-wait-gate") == 0) {
            opt->true_ds4_compressed_kv_dense_event_wait_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-skip-dense-stats-gate") == 0) {
            opt->true_ds4_compressed_kv_skip_dense_stats_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-attn-input-fill-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_attn_input_fill_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-input-fill-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_input_fill_gate = true;
            opt->true_ds4_indexer_attention_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-rope-round-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_rope_round_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-pool-norm-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_pool_norm_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-pool-norm-rope-round-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_pool_norm_rope_round_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-reference-diff-gate") == 0) {
            opt->true_ds4_compressed_reference_diff_gate = true;
            opt->true_ds4_indexer_attention_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--reference-hc-reduce-gate") == 0) {
            opt->reference_hc_reduce_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--reference-hc-state-guard-gate") == 0) {
            opt->reference_hc_state_guard_gate = true;
            opt->reference_hc_reduce_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-persist-state-gate") == 0) {
            opt->tp_hc_persist_state_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-kv-all-slots-gate") == 0) {
            opt->tp_kv_all_slots_gate = true;
        } else if (std::strcmp(arg, "--cuda-profiler-window") == 0) {
            opt->cuda_profiler_window = true;
        } else if (std::strcmp(arg, "--cuda-profiler-device") == 0) {
            if (!val || !parse_int(val, &opt->cuda_profiler_device) ||
                opt->cuda_profiler_device < 0 || opt->cuda_profiler_device >= kGpus) {
                return false;
            }
            opt->cuda_profiler_window = true;
            ++i;
        } else if (std::strcmp(arg, "--cuda-profiler-all-devices") == 0) {
            opt->cuda_profiler_window = true;
            opt->cuda_profiler_all_devices = true;
        } else if (std::strcmp(arg, "--diagnostic-output-head") == 0) {
            opt->diagnostic_output_head = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--diagnostic-output-head-lazy-gate") == 0) {
            opt->diagnostic_output_head = true;
            opt->diagnostic_output_head_lazy_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }
    return opt->pack_dir && opt->contract_path && opt->tm_index_path &&
           opt->top_k <= kPackedLocalExperts && opt->layer >= 0 &&
           (!opt->model_router_routes || opt->top_k == kModelTopK) &&
           (!opt->gpu_route_plan_gate || opt->compact_moe_decode_gate) &&
           (!opt->nccl_reduce_scatter_compose_gate ||
            !opt->decode_cudagraph_gate) &&
           (!opt->tp_hc_current_input_nccl_allgather_gate ||
            opt->tp_hc_current_input_gate) &&
           (!opt->tp_hc_current_allreduce_gate ||
            opt->tp_hc_current_input_gate) &&
           !(opt->model_router_routes && opt->compact_route_compose &&
             !opt->compact_moe_decode_gate) &&
           !(opt->dense_hmma_compose && opt->dense_f16_cublas_compose) &&
           (!opt->dense_f16_cache_compose || opt->dense_f16_cublas_compose) &&
           (!opt->true_ds4_attention_residency_gate ||
            (opt->share_dense_ops && opt->dense_f16_cache_compose &&
             opt->dense_f16_cublas_compose)) &&
           (!opt->true_ds4_attention_projection_gate ||
            opt->true_ds4_attention_residency_gate) &&
           (!opt->true_ds4_attention_projection_direct_input_fill_gate ||
            opt->true_ds4_attention_projection_gate) &&
           (!opt->true_ds4_attention_projection_rank_local_input_gate ||
            opt->true_ds4_attention_projection_gate) &&
           (!opt->true_ds4_attention_projection_rank_major_input_gate ||
            (opt->true_ds4_attention_projection_gate &&
             opt->tp_hc_current_input_nccl_allgather_gate)) &&
           (!opt->true_ds4_attention_state_gate ||
            opt->true_ds4_attention_projection_gate) &&
           (!opt->true_ds4_attention_rope_gate ||
            opt->true_ds4_attention_state_gate) &&
           (!opt->true_ds4_attention_saturation_audit_gate ||
            opt->true_ds4_attention_rope_gate) &&
           (!opt->true_ds4_attention_kv_norm_reference_gate ||
            opt->true_ds4_attention_projection_gate) &&
           (!opt->true_ds4_attention_raw_read_gate ||
            opt->true_ds4_attention_state_gate) &&
           (!opt->true_ds4_attention_raw_window_gate ||
            opt->true_ds4_attention_raw_read_gate) &&
           (!opt->true_ds4_attention_typed_kv_raw_gate ||
            opt->true_ds4_attention_state_gate) &&
           (!opt->true_ds4_attention_typed_kv_compressed_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_attention_typed_kv_indexer_gate ||
            opt->true_ds4_indexer_attention_gate) &&
           (!opt->true_ds4_attention_typed_kv_history_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_attention_typed_kv_skip_current_load_gate ||
            (opt->true_ds4_attention_typed_kv_raw_gate ||
             opt->true_ds4_attention_typed_kv_compressed_gate ||
             opt->true_ds4_attention_typed_kv_indexer_gate)) &&
           (!opt->true_ds4_attention_typed_kv_skip_raw_store_gate ||
            opt->true_ds4_attention_typed_kv_raw_gate) &&
           (!opt->true_ds4_attention_typed_kv_skip_compressed_store_gate ||
            opt->true_ds4_attention_typed_kv_compressed_gate) &&
           (!opt->true_ds4_attention_typed_kv_skip_indexer_store_gate ||
            opt->true_ds4_attention_typed_kv_indexer_gate) &&
           (!opt->true_ds4_attention_typed_kv_quiet_gate ||
            (opt->true_ds4_attention_typed_kv_raw_gate ||
             opt->true_ds4_attention_typed_kv_compressed_gate ||
             opt->true_ds4_attention_typed_kv_indexer_gate ||
             opt->true_ds4_attention_typed_kv_history_gate)) &&
           (!opt->true_ds4_attention_typed_kv_batch_rows_gate ||
            (opt->true_ds4_attention_typed_kv_raw_gate ||
             opt->true_ds4_attention_typed_kv_compressed_gate ||
             opt->true_ds4_attention_typed_kv_indexer_gate ||
             opt->true_ds4_attention_typed_kv_history_gate)) &&
           (!opt->true_ds4_attention_typed_kv_stream_sync_gate ||
            (opt->true_ds4_attention_typed_kv_raw_gate ||
             opt->true_ds4_attention_typed_kv_compressed_gate ||
             opt->true_ds4_attention_typed_kv_indexer_gate ||
             opt->true_ds4_attention_typed_kv_history_gate)) &&
           (!opt->true_ds4_attention_output_gate ||
            opt->true_ds4_attention_raw_window_gate) &&
           (!opt->true_ds4_attention_output_nccl_allgather_gate ||
            opt->true_ds4_attention_output_gate) &&
           (!opt->true_ds4_post_attention_ffn_input_gate ||
            (opt->true_ds4_attention_output_gate && opt->true_shared_ffn_gate &&
             opt->model_router_routes && opt->routed_ffn_norm_input_gate)) &&
           (!(opt->routed_ffn_rank_major_input_gate ||
              opt->routed_ffn_rank_major_shared_input_gate ||
              opt->routed_ffn_rank_major_route_input_gate ||
              opt->routed_ffn_rank_major_input_parity_gate) ||
            (opt->true_ds4_post_attention_ffn_input_gate &&
             opt->tp_hc_current_input_nccl_allgather_gate)) &&
           (!opt->model_router_rank_major_logits_gate ||
            opt->routed_ffn_rank_major_input_gate) &&
           !(opt->model_router_rank_major_logits_gate &&
             opt->model_router_allreduce_logits_gate) &&
           (!opt->true_ds4_semantic_skip_stats_gate ||
            (opt->true_ds4_attention_output_gate ||
             opt->true_ds4_post_attention_ffn_input_gate)) &&
           (!opt->true_ds4_compressed_kv_gate ||
            opt->true_ds4_attention_projection_gate) &&
           (!opt->true_ds4_indexer_attention_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_direct_input_fill_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_dense_event_wait_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_skip_dense_stats_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_fused_attn_input_fill_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_fused_input_fill_gate ||
            opt->true_ds4_indexer_attention_gate) &&
           (!opt->true_ds4_compressed_kv_fused_rope_round_gate ||
            (opt->true_ds4_compressed_kv_gate &&
             opt->true_ds4_attention_rope_gate)) &&
           (!opt->true_ds4_compressed_kv_fused_pool_norm_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_fused_pool_norm_rope_round_gate ||
            (opt->true_ds4_compressed_kv_gate &&
             opt->true_ds4_attention_rope_gate)) &&
           (!opt->true_ds4_compressed_reference_diff_gate ||
            opt->true_ds4_indexer_attention_gate) &&
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

void enqueue_graph_f32_copy_between_devices(const Options &opt,
                                            int dst_device,
                                            int src_device,
                                            float *dst,
                                            const float *src,
                                            uint64_t elems,
                                            cudaStream_t stream,
                                            int block) {
    (void)dst_device;
    (void)src_device;
    (void)opt;
    copy_f32_kernel<<<(unsigned int)((elems + (uint64_t)block - 1) /
                                     (uint64_t)block),
                      block, 0, stream>>>(dst, src, elems);
    CHECK_CUDA(cudaGetLastError());
}

void enqueue_graph_f32_copy_from_device0(const Options &opt,
                                         RankState &rank_state,
                                         int /*rank*/,
                                         float *dst,
                                         const float *src,
                                         uint64_t elems,
                                         cudaStream_t stream,
                                         int block) {
    enqueue_graph_f32_copy_between_devices(opt, rank_state.device, opt.devices[0],
                                           dst, src, elems, stream, block);
}

void enqueue_graph_i32_copy_from_device0(const Options &opt,
                                         RankState &rank_state,
                                         int /*rank*/,
                                         int *dst,
                                       const int *src,
                                       uint64_t elems,
                                       cudaStream_t stream,
                                       int block) {
    (void)opt;
    (void)rank_state;
    copy_i32_kernel<<<(unsigned int)((elems + (uint64_t)block - 1) /
                                     (uint64_t)block),
                      block, 0, stream>>>(dst, src, elems);
    CHECK_CUDA(cudaGetLastError());
}

int nccl_broadcast_bytes_from_rank(RankState ranks[kGpus],
                                   int root,
                                   const void *src_root,
                                   void *dst_by_rank[kGpus],
                                   size_t bytes,
                                   const char *label) {
    if (root < 0 || root >= kGpus || !src_root || !dst_by_rank || bytes == 0) {
        return 1;
    }
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.compose_nccl_initialized || !r.compose_nccl ||
            !dst_by_rank[rank]) {
            std::fprintf(stderr,
                         "tp_ep_nccl_broadcast_missing\tlabel\t%s\t"
                         "rank\t%d\tcompose\t%d\tdst\t%d\n",
                         label ? label : "-", rank,
                         (r.compose_nccl_initialized && r.compose_nccl) ? 1 : 0,
                         dst_by_rank[rank] ? 1 : 0);
            return 2;
        }
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        const void *send = rank == root ? src_root : dst_by_rank[rank];
        CHECK_NCCL(ncclBroadcast(send, dst_by_rank[rank], bytes, ncclChar, root,
                                 r.compose_nccl, r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    CHECK_CUDA(cudaSetDevice(prior_device));
    return 0;
}

int nccl_broadcast_bytes_from_rank0(RankState ranks[kGpus],
                                    const void *src_rank0,
                                    void *dst_by_rank[kGpus],
                                    size_t bytes,
                                    const char *label) {
    return nccl_broadcast_bytes_from_rank(ranks, 0, src_rank0, dst_by_rank,
                                          bytes, label);
}

int broadcast_ep_return_slices(RankState ranks[kGpus],
                               bool fp16,
                               bool skip_self_copy,
                               uint64_t src_stride_elems,
                               const uint64_t copy_elems_by_src[kGpus],
                               const char *label) {
    if (!copy_elems_by_src || src_stride_elems == 0) return 1;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int src = 0; src < kGpus; ++src) {
        const uint64_t copy_elems = copy_elems_by_src[src];
        const uint64_t bcast_elems = (uint64_t)kGpus * src_stride_elems;
        const size_t elem_bytes = fp16 ? sizeof(__half) : sizeof(float);
        const size_t bcast_bytes = (size_t)(bcast_elems * elem_bytes);
        void *scratch_by_rank[kGpus] = {};
        for (int rank = 0; rank < kGpus; ++rank) {
            scratch_by_rank[rank] = fp16
                ? (void *)ranks[rank].d_ep_contrib_half_bcast_all
                : (void *)ranks[rank].d_ep_contrib_bcast_all;
            if (!scratch_by_rank[rank]) return 2;
        }
        const void *src_all = fp16
            ? (const void *)ranks[src].d_ep_contrib_half_all
            : (const void *)ranks[src].d_ep_contrib_all;
        if (!src_all) return 3;
        if (nccl_broadcast_bytes_from_rank(
                ranks, src, src_all, scratch_by_rank, bcast_bytes,
                label ? label : "ep_return_broadcast") != 0) {
            return 4;
        }
        if (copy_elems == 0) continue;
        const size_t copy_bytes = (size_t)(copy_elems * elem_bytes);
        for (int dst = 0; dst < kGpus; ++dst) {
            if (skip_self_copy && src == dst) continue;
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            const uint64_t offset_elems = (uint64_t)dst * src_stride_elems;
            if (fp16) {
                if (!r.d_ep_remote_half[src]) return 5;
                const __half *src_ptr =
                    r.d_ep_contrib_half_bcast_all + offset_elems;
                CHECK_CUDA(cudaMemcpyAsync(r.d_ep_remote_half[src], src_ptr,
                                           copy_bytes, cudaMemcpyDeviceToDevice,
                                           r.stream));
            } else {
                if (!r.d_ep_remote[src]) return 5;
                const float *src_ptr = r.d_ep_contrib_bcast_all + offset_elems;
                CHECK_CUDA(cudaMemcpyAsync(r.d_ep_remote[src], src_ptr,
                                           copy_bytes, cudaMemcpyDeviceToDevice,
                                           r.stream));
            }
        }
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
    return 0;
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

int attn_comp_state_rows_for_ratio(int ratio) {
    if (ratio == 4) return 2 * ratio;
    return ratio > 0 ? ratio : 0;
}

int attn_comp_state_width_for_ratio(int ratio) {
    if (ratio == 4) return 2 * kHeadDim;
    return ratio > 0 ? kHeadDim : 0;
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

struct SharedHcControls {
    bool initialized = false;
    int slots = 0;
    int devices[kGpus] = {};
    float *d_hc = nullptr;
    float *d_hc_norm = nullptr;
    float *d_mix = nullptr;
    float *d_split = nullptr;
    float *d_current_full = nullptr;
    float *d_attn_normed = nullptr;
    float *d_q_a_full = nullptr;
    float *d_q_a_normed = nullptr;
    float *d_kv_full = nullptr;
    float *d_kv_normed = nullptr;
    float *d_ffn_normed = nullptr;
    float *d_attn_comp_kv_full = nullptr;
    float *d_attn_comp_score_full = nullptr;
    float *d_index_comp_kv_full = nullptr;
    float *d_index_comp_score_full = nullptr;
    float *d_indexer_q_full = nullptr;
    float *d_indexer_w_full = nullptr;
    float *d_attn_norm_weight[43] = {};
    float *d_attn_norm_weight_rank[43][kGpus] = {};
    float *d_q_a_norm_weight[43] = {};
    float *d_kv_a_norm_weight[43] = {};
    float *d_attn_compress_ape[43] = {};
    float *d_attn_compress_norm[43] = {};
    float *d_indexer_compress_ape[43] = {};
    float *d_indexer_compress_norm[43] = {};
    float *d_attn_sinks[43] = {};
    float *d_attn_fn[43] = {};
    float *d_attn_fn_rank[43][kGpus] = {};
    float *d_attn_base[43] = {};
    float *d_attn_base_rank[43][kGpus] = {};
    float *d_attn_scale[43] = {};
    float *d_attn_scale_rank[43][kGpus] = {};
    float *d_ffn_fn[43] = {};
    float *d_ffn_fn_rank[43][kGpus] = {};
    float *d_ffn_base[43] = {};
    float *d_ffn_base_rank[43][kGpus] = {};
    float *d_ffn_scale[43] = {};
    float *d_ffn_scale_rank[43][kGpus] = {};
    float *d_ffn_norm_weight[43] = {};
    float *d_ffn_norm_weight_rank[43][kGpus] = {};
    float *d_router_w[43] = {};
    float *d_router_w_ep[43][kGpus] = {};
    float *d_router_w_shard[43][kGpus] = {};
    float *d_router_bias[43] = {};
    int *d_router_hash[43] = {};
    uint32_t router_hash_rows[43] = {};
    float *d_router_logits = nullptr;
    int *d_router_selected = nullptr;
    float *d_router_weights = nullptr;
    uint32_t *d_router_tokens = nullptr;
    unsigned char *d_router_active = nullptr;
    cublasHandle_t router_blas = nullptr;
    RoutePlanHostWorkspace route_plan_ws;
    uint64_t control_bytes = 0;
};

int init_route_plan_host_workspace(const Options &opt,
                                   RoutePlanHostWorkspace *ws) {
    if (!ws) return 1;
    if (ws->initialized) return 0;
    ws->slots = opt.slots;
    ws->top_k = opt.top_k;
    for (int rank = 0; rank < kGpus; ++rank) {
        ws->devices[rank] = opt.devices[rank];
    }
    ws->route_capacity = (size_t)opt.slots * (size_t)opt.top_k;
    ws->compact_plan_ints =
        (size_t)kGpus * ((size_t)opt.slots * (size_t)opt.top_k +
                         (size_t)opt.slots);
    CHECK_CUDA(cudaHostAlloc(&ws->h_selected,
                             ws->route_capacity * sizeof(int),
                             cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&ws->h_weights,
                             ws->route_capacity * sizeof(float),
                             cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&ws->h_compact_plan,
                             ws->compact_plan_ints * sizeof(int),
                             cudaHostAllocDefault));
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaHostAlloc(&ws->h_offsets[rank],
                                 (size_t)(kLocalExperts + 1) * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_slots[rank],
                                 ws->route_capacity * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_weights[rank],
                                 ws->route_capacity * sizeof(float),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_index_by_slot[rank],
                                 (size_t)opt.slots * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_indices_by_slot[rank],
                                 ws->route_capacity * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_count_by_slot[rank],
                                 (size_t)opt.slots * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
        CHECK_CUDA(cudaEventCreateWithFlags(&ws->upload_done[rank],
                                            cudaEventDisableTiming));
    }
    ws->initialized = true;
    return 0;
}

void close_route_plan_host_workspace(RoutePlanHostWorkspace *ws) {
    if (!ws || !ws->initialized) return;
    if (ws->uploads_pending) {
        for (int rank = 0; rank < kGpus; ++rank) {
            if (ws->upload_done[rank]) {
                CHECK_CUDA(cudaSetDevice(ws->devices[rank]));
                CHECK_CUDA(cudaEventSynchronize(ws->upload_done[rank]));
            }
        }
        ws->uploads_pending = false;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ws->devices[rank]));
        if (ws->upload_done[rank]) CHECK_CUDA(cudaEventDestroy(ws->upload_done[rank]));
        if (ws->h_route_count_by_slot[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_count_by_slot[rank]));
        if (ws->h_route_indices_by_slot[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_indices_by_slot[rank]));
        if (ws->h_route_index_by_slot[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_index_by_slot[rank]));
        if (ws->h_route_weights[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_weights[rank]));
        if (ws->h_route_slots[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_slots[rank]));
        if (ws->h_offsets[rank]) CHECK_CUDA(cudaFreeHost(ws->h_offsets[rank]));
    }
    if (ws->h_compact_plan) CHECK_CUDA(cudaFreeHost(ws->h_compact_plan));
    if (ws->h_weights) CHECK_CUDA(cudaFreeHost(ws->h_weights));
    if (ws->h_selected) CHECK_CUDA(cudaFreeHost(ws->h_selected));
    *ws = RoutePlanHostWorkspace{};
}

bool find_replicated_control_row(const std::vector<ContractRow> &rows,
                                 const char *tensor,
                                 ContractRow *out) {
    for (const ContractRow &r : rows) {
        if (r.record_type == "replicated_control" && r.tensor_id == tensor) {
            *out = r;
            return true;
        }
    }
    return false;
}

int load_control_f32(const Options &opt,
                     const std::vector<ContractRow> &rows,
                     const char *tensor,
                     size_t elems,
                     std::vector<float> *out) {
    ContractRow r;
    if (!find_replicated_control_row(rows, tensor, &r)) {
        std::fprintf(stderr, "missing replicated control tensor %s\n", tensor);
        return 1;
    }
    if (r.source_dtype != "f32" || r.bytes_estimate != elems * sizeof(float)) {
        std::fprintf(stderr, "bad replicated control tensor %s dtype=%s bytes=%llu expected=%zu\n",
                     tensor, r.source_dtype.c_str(),
                     (unsigned long long)r.bytes_estimate, elems * sizeof(float));
        return 2;
    }
    out->resize(elems);
    const std::string path = path_join(opt.pack_dir, r.source_pack_file);
    if (read_exact_at(path, physical_row_offset(r), out->data(), elems * sizeof(float)) != 0) {
        return 3;
    }
    return 0;
}

int load_optional_control_f32(const Options &opt,
                              const std::vector<ContractRow> &rows,
                              const char *tensor,
                              size_t elems,
                              std::vector<float> *out,
                              bool *found) {
    ContractRow r;
    if (!find_replicated_control_row(rows, tensor, &r)) {
        out->clear();
        if (found) *found = false;
        return 0;
    }
    if (found) *found = true;
    return load_control_f32(opt, rows, tensor, elems, out);
}

int load_optional_control_i32(const Options &opt,
                              const std::vector<ContractRow> &rows,
                              const char *tensor,
                              size_t elems,
                              std::vector<int> *out,
                              bool *found) {
    ContractRow r;
    if (!find_replicated_control_row(rows, tensor, &r)) {
        out->clear();
        if (found) *found = false;
        return 0;
    }
    if (found) *found = true;
    if (r.source_dtype != "i32" || r.bytes_estimate != elems * sizeof(int)) {
        std::fprintf(stderr, "bad replicated control tensor %s dtype=%s bytes=%llu expected=%zu\n",
                     tensor, r.source_dtype.c_str(),
                     (unsigned long long)r.bytes_estimate, elems * sizeof(int));
        return 2;
    }
    out->resize(elems);
    const std::string path = path_join(opt.pack_dir, r.source_pack_file);
    if (read_exact_at(path, physical_row_offset(r), out->data(), elems * sizeof(int)) != 0) {
        return 3;
    }
    return 0;
}

int open_shared_hc_controls(const Options &opt,
                            const std::vector<ContractRow> &rows,
                            SharedHcControls *out) {
    out->slots = opt.slots;
    for (int rank = 0; rank < kGpus; ++rank) out->devices[rank] = opt.devices[rank];
    const uint64_t hc_elems = (uint64_t)opt.slots * kHcRows * (uint64_t)kHidden;
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    cublasStatus_t blas_status = cublasCreate(&out->router_blas);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "router cublasCreate failed status=%d\n",
                     (int)blas_status);
        return 1;
    }
    CHECK_CUDA(cudaMalloc(&out->d_hc, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_hc_norm, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_mix, (size_t)opt.slots * kHcMix * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_split, (size_t)opt.slots * kHcMix * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_current_full,
                          (size_t)opt.slots * kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_attn_normed,
                          (size_t)opt.slots * kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_q_a_full,
                          (size_t)opt.slots * 1024u * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_q_a_normed,
                          (size_t)opt.slots * 1024u * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_kv_full,
                          (size_t)opt.slots * kHeadDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_kv_normed,
                          (size_t)opt.slots * kHeadDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_ffn_normed,
                          (size_t)opt.slots * kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_attn_comp_kv_full,
                          (size_t)opt.slots * kCompWidthMax * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_attn_comp_score_full,
                          (size_t)opt.slots * kCompWidthMax * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_index_comp_kv_full,
                          (size_t)opt.slots * kIndexCompWidth * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_index_comp_score_full,
                          (size_t)opt.slots * kIndexCompWidth * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_indexer_q_full,
                          (size_t)opt.slots * kIndexerHead *
                              (size_t)kIndexerHeadDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_indexer_w_full,
                          (size_t)opt.slots * kIndexerHead * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_router_logits,
                          (size_t)opt.slots * kGlobalExperts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_router_selected,
                          (size_t)opt.slots * kModelTopK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&out->d_router_weights,
                          (size_t)opt.slots * kModelTopK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_router_tokens,
                          (size_t)opt.slots * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&out->d_router_active,
                          (size_t)opt.slots * sizeof(unsigned char)));
    CHECK_CUDA(cudaMemset(out->d_router_tokens, 0,
                          (size_t)opt.slots * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(out->d_router_active, 1,
                          (size_t)opt.slots * sizeof(unsigned char)));
    if (opt.route_plan_async_upload_gate &&
        init_route_plan_host_workspace(opt, &out->route_plan_ws) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));

    for (int layer = 0; layer < 43; ++layer) {
        std::vector<float> attn_fn;
        std::vector<float> attn_base;
        std::vector<float> attn_scale;
        std::vector<float> fn;
        std::vector<float> base;
        std::vector<float> scale;
        std::vector<float> ffn_norm_weight;
        std::vector<float> attn_norm_weight;
        std::vector<float> q_a_norm_weight;
        std::vector<float> kv_a_norm_weight;
        std::vector<float> attn_sinks;
        std::vector<float> attn_compress_ape;
        std::vector<float> attn_compress_norm;
        std::vector<float> indexer_compress_ape;
        std::vector<float> indexer_compress_norm;
        std::vector<float> router_w;
        std::vector<float> router_bias;
        std::vector<int> router_hash;
        const std::string attn_norm_name = layer_tensor_name(layer, "attn_norm.weight");
        const std::string q_a_norm_name = layer_tensor_name(layer, "attn_q_a_norm.weight");
        const std::string kv_a_norm_name = layer_tensor_name(layer, "attn_kv_a_norm.weight");
        const std::string attn_sinks_name = layer_tensor_name(layer, "attn_sinks");
        const std::string attn_compress_ape_name =
            layer_tensor_name(layer, "attn_compress_ape");
        const std::string attn_compress_norm_name =
            layer_tensor_name(layer, "attn_compress_norm.weight");
        const std::string indexer_compress_ape_name =
            layer_tensor_name(layer, "indexer.compress_ape");
        const std::string indexer_compress_norm_name =
            layer_tensor_name(layer, "indexer.compress_norm.weight");
        const std::string attn_fn_name = layer_tensor_name(layer, "hc_attn_fn");
        const std::string attn_base_name = layer_tensor_name(layer, "hc_attn_base");
        const std::string attn_scale_name = layer_tensor_name(layer, "hc_attn_scale");
        const std::string fn_name = layer_tensor_name(layer, "hc_ffn_fn");
        const std::string base_name = layer_tensor_name(layer, "hc_ffn_base");
        const std::string scale_name = layer_tensor_name(layer, "hc_ffn_scale");
        const std::string ffn_norm_name = layer_tensor_name(layer, "ffn_norm.weight");
        const std::string router_name = layer_tensor_name(layer, "ffn_gate_inp.weight");
        const std::string bias_name = layer_tensor_name(layer, "exp_probs_b");
        const std::string hash_name = layer_tensor_name(layer, "ffn_gate_tid2eid");
        const int ratio = ds4_layer_ratio(layer);
        bool have_attn_compress_ape = false;
        bool have_attn_compress_norm = false;
        bool have_indexer_compress_ape = false;
        bool have_indexer_compress_norm = false;
        bool have_bias = false;
        bool have_hash = false;
        if (load_control_f32(opt, rows, attn_fn_name.c_str(),
                             (size_t)kHcRows * (size_t)kHidden * kHcMix, &attn_fn) ||
            load_control_f32(opt, rows, attn_base_name.c_str(), kHcMix, &attn_base) ||
            load_control_f32(opt, rows, attn_scale_name.c_str(), 3, &attn_scale) ||
            load_control_f32(opt, rows, attn_norm_name.c_str(),
                             kHidden, &attn_norm_weight) ||
            load_control_f32(opt, rows, q_a_norm_name.c_str(),
                             1024, &q_a_norm_weight) ||
            load_control_f32(opt, rows, kv_a_norm_name.c_str(),
                             kHeadDim, &kv_a_norm_weight) ||
            load_control_f32(opt, rows, attn_sinks_name.c_str(),
                             kHeadCount, &attn_sinks) ||
            (ratio != 0 &&
             (load_optional_control_f32(opt, rows, attn_compress_ape_name.c_str(),
                                        (size_t)ratio *
                                            (size_t)(ratio == 4 ? kCompWidthMax
                                                               : kHeadDim),
                                        &attn_compress_ape,
                                        &have_attn_compress_ape) ||
              load_optional_control_f32(opt, rows, attn_compress_norm_name.c_str(),
                                        kHeadDim, &attn_compress_norm,
                                        &have_attn_compress_norm))) ||
            (ratio == 4 &&
             (load_optional_control_f32(opt, rows, indexer_compress_ape_name.c_str(),
                                        (size_t)ratio * (size_t)kIndexCompWidth,
                                        &indexer_compress_ape,
                                        &have_indexer_compress_ape) ||
              load_optional_control_f32(opt, rows, indexer_compress_norm_name.c_str(),
                                        kIndexerHeadDim,
                                        &indexer_compress_norm,
                                        &have_indexer_compress_norm))) ||
            load_control_f32(opt, rows, fn_name.c_str(),
                             (size_t)kHcRows * (size_t)kHidden * kHcMix, &fn) ||
            load_control_f32(opt, rows, base_name.c_str(), kHcMix, &base) ||
            load_control_f32(opt, rows, scale_name.c_str(), 3, &scale) ||
            load_control_f32(opt, rows, ffn_norm_name.c_str(),
                             kHidden, &ffn_norm_weight) ||
            load_control_f32(opt, rows, router_name.c_str(),
                             (size_t)kHidden * kGlobalExperts, &router_w) ||
            load_optional_control_f32(opt, rows, bias_name.c_str(),
                                      kGlobalExperts, &router_bias, &have_bias) ||
            load_optional_control_i32(opt, rows, hash_name.c_str(),
                                      (size_t)kRouterHashRows * kModelTopK,
                                      &router_hash, &have_hash)) {
            return 1;
        }
        CHECK_CUDA(cudaMalloc(&out->d_attn_fn[layer], attn_fn.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_base[layer], attn_base.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_scale[layer], attn_scale.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_fn[layer], fn.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_base[layer], base.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_scale[layer], scale.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_norm_weight[layer],
                              ffn_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_norm_weight[layer],
                              attn_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_q_a_norm_weight[layer],
                              q_a_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_kv_a_norm_weight[layer],
                              kv_a_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_sinks[layer],
                              attn_sinks.size() * sizeof(float)));
        if (have_attn_compress_ape && have_attn_compress_norm) {
            CHECK_CUDA(cudaMalloc(&out->d_attn_compress_ape[layer],
                                  attn_compress_ape.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&out->d_attn_compress_norm[layer],
                                  attn_compress_norm.size() * sizeof(float)));
        }
        if (have_indexer_compress_ape && have_indexer_compress_norm) {
            CHECK_CUDA(cudaMalloc(&out->d_indexer_compress_ape[layer],
                                  indexer_compress_ape.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&out->d_indexer_compress_norm[layer],
                                  indexer_compress_norm.size() * sizeof(float)));
        }
        if (!opt.model_router_rank_major_logits_gate &&
            !opt.model_router_allreduce_logits_gate) {
            CHECK_CUDA(cudaMalloc(&out->d_router_w[layer],
                                  router_w.size() * sizeof(float)));
        }
        if (opt.tp_hc_current_allreduce_gate) {
            const int shard_cols = kHidden / kGpus;
            const size_t local_cols = (size_t)kHcRows * (size_t)shard_cols;
            std::vector<float> fn_rank(local_cols * (size_t)kHcMix);
            std::vector<float> attn_fn_rank(local_cols * (size_t)kHcMix);
            for (int rank = 0; rank < kGpus; ++rank) {
                for (int row = 0; row < kHcRows; ++row) {
                    for (int local_h = 0; local_h < shard_cols; ++local_h) {
                        const size_t local_c =
                            (size_t)row * (size_t)shard_cols + (size_t)local_h;
                        const size_t global_c =
                            (size_t)row * (size_t)kHidden +
                            (size_t)rank * (size_t)shard_cols +
                            (size_t)local_h;
                        for (int mix = 0; mix < kHcMix; ++mix) {
                            attn_fn_rank[local_c * (size_t)kHcMix + (size_t)mix] =
                                attn_fn[global_c * (size_t)kHcMix + (size_t)mix];
                            fn_rank[local_c * (size_t)kHcMix + (size_t)mix] =
                                fn[global_c * (size_t)kHcMix + (size_t)mix];
                        }
                    }
                }
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_attn_fn_rank[layer][rank],
                                      attn_fn_rank.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_attn_fn_rank[layer][rank],
                                      attn_fn_rank.data(),
                                      attn_fn_rank.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_attn_base_rank[layer][rank],
                                      attn_base.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_attn_base_rank[layer][rank],
                                      attn_base.data(),
                                      attn_base.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_attn_scale_rank[layer][rank],
                                      attn_scale.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_attn_scale_rank[layer][rank],
                                      attn_scale.data(),
                                      attn_scale.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_ffn_fn_rank[layer][rank],
                                      fn_rank.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_ffn_fn_rank[layer][rank],
                                      fn_rank.data(),
                                      fn_rank.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_ffn_base_rank[layer][rank],
                                      base.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_ffn_base_rank[layer][rank],
                                      base.data(),
                                      base.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_ffn_scale_rank[layer][rank],
                                      scale.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_ffn_scale_rank[layer][rank],
                                      scale.data(),
                                      scale.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        if (opt.model_router_rank_major_logits_gate) {
            std::vector<float> router_w_ep((size_t)kHidden * (size_t)kLocalExperts);
            for (int rank = 0; rank < kGpus; ++rank) {
                for (int h = 0; h < kHidden; ++h) {
                    for (int e = 0; e < kLocalExperts; ++e) {
                        router_w_ep[(size_t)h * (size_t)kLocalExperts + (size_t)e] =
                            router_w[(size_t)h * (size_t)kGlobalExperts +
                                     (size_t)(rank * kLocalExperts + e)];
                    }
                }
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_router_w_ep[layer][rank],
                                      router_w_ep.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_router_w_ep[layer][rank],
                                      router_w_ep.data(),
                                      router_w_ep.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        if (opt.model_router_allreduce_logits_gate) {
            const int shard_cols = kHidden / kGpus;
            std::vector<float> router_w_shard(
                (size_t)shard_cols * (size_t)kGlobalExperts);
            for (int rank = 0; rank < kGpus; ++rank) {
                for (int local_h = 0; local_h < shard_cols; ++local_h) {
                    const int global_h = rank * shard_cols + local_h;
                    for (int expert = 0; expert < kGlobalExperts; ++expert) {
                        router_w_shard[(size_t)local_h *
                                           (size_t)kGlobalExperts +
                                       (size_t)expert] =
                            router_w[(size_t)global_h *
                                         (size_t)kGlobalExperts +
                                     (size_t)expert];
                    }
                }
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_router_w_shard[layer][rank],
                                      router_w_shard.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_router_w_shard[layer][rank],
                                      router_w_shard.data(),
                                      router_w_shard.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        if (have_bias) {
            CHECK_CUDA(cudaMalloc(&out->d_router_bias[layer],
                                  router_bias.size() * sizeof(float)));
        }
        if (have_hash) {
            CHECK_CUDA(cudaMalloc(&out->d_router_hash[layer],
                                  router_hash.size() * sizeof(int)));
            out->router_hash_rows[layer] = kRouterHashRows;
        }
        CHECK_CUDA(cudaMemcpy(out->d_attn_fn[layer], attn_fn.data(),
                              attn_fn.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_attn_base[layer], attn_base.data(),
                              attn_base.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_attn_scale[layer], attn_scale.data(),
                              attn_scale.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_fn[layer], fn.data(),
                              fn.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_base[layer], base.data(),
                              base.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_scale[layer], scale.data(),
                              scale.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_norm_weight[layer], ffn_norm_weight.data(),
                              ffn_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        out->d_ffn_norm_weight_rank[layer][0] = out->d_ffn_norm_weight[layer];
        if (opt.model_router_allreduce_logits_gate ||
            opt.routed_ffn_rank_major_input_gate ||
            opt.routed_ffn_rank_major_shared_input_gate ||
            opt.routed_ffn_rank_major_route_input_gate ||
            opt.routed_ffn_rank_major_input_parity_gate) {
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_ffn_norm_weight_rank[layer][rank],
                                      ffn_norm_weight.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_ffn_norm_weight_rank[layer][rank],
                                      ffn_norm_weight.data(),
                                      ffn_norm_weight.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        CHECK_CUDA(cudaMemcpy(out->d_attn_norm_weight[layer], attn_norm_weight.data(),
                              attn_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        out->d_attn_norm_weight_rank[layer][0] = out->d_attn_norm_weight[layer];
        if (opt.true_ds4_attention_projection_rank_local_input_gate ||
            opt.true_ds4_attention_projection_rank_major_input_gate) {
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_attn_norm_weight_rank[layer][rank],
                                      attn_norm_weight.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_attn_norm_weight_rank[layer][rank],
                                      attn_norm_weight.data(),
                                      attn_norm_weight.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        CHECK_CUDA(cudaMemcpy(out->d_q_a_norm_weight[layer], q_a_norm_weight.data(),
                              q_a_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_kv_a_norm_weight[layer], kv_a_norm_weight.data(),
                              kv_a_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_attn_sinks[layer], attn_sinks.data(),
                              attn_sinks.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (out->d_attn_compress_ape[layer]) {
            CHECK_CUDA(cudaMemcpy(out->d_attn_compress_ape[layer],
                                  attn_compress_ape.data(),
                                  attn_compress_ape.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(out->d_attn_compress_norm[layer],
                                  attn_compress_norm.data(),
                                  attn_compress_norm.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (out->d_indexer_compress_ape[layer]) {
            CHECK_CUDA(cudaMemcpy(out->d_indexer_compress_ape[layer],
                                  indexer_compress_ape.data(),
                                  indexer_compress_ape.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(out->d_indexer_compress_norm[layer],
                                  indexer_compress_norm.data(),
                                  indexer_compress_norm.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (out->d_router_w[layer]) {
            CHECK_CUDA(cudaMemcpy(out->d_router_w[layer], router_w.data(),
                                  router_w.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (have_bias) {
            CHECK_CUDA(cudaMemcpy(out->d_router_bias[layer], router_bias.data(),
                                  router_bias.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (have_hash) {
            CHECK_CUDA(cudaMemcpy(out->d_router_hash[layer], router_hash.data(),
                                  router_hash.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }
        out->control_bytes +=
            (attn_fn.size() + attn_base.size() + attn_scale.size() +
             attn_norm_weight.size() + q_a_norm_weight.size() +
             kv_a_norm_weight.size() + attn_sinks.size() +
             attn_compress_ape.size() + attn_compress_norm.size() +
             indexer_compress_ape.size() + indexer_compress_norm.size() +
             fn.size() + base.size() + scale.size() +
             ffn_norm_weight.size() + router_w.size() + router_bias.size()) *
                sizeof(float) +
            router_hash.size() * sizeof(int);
    }
    out->initialized = true;
    return 0;
}

void close_shared_hc_controls(const Options &opt, SharedHcControls *out) {
    if (!out || !out->initialized) return;
    close_route_plan_host_workspace(&out->route_plan_ws);
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    if (out->router_blas) {
        cublasStatus_t st = cublasDestroy(out->router_blas);
        if (st != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "router cublasDestroy failed status=%d\n", (int)st);
        }
    }
    for (int layer = 0; layer < 43; ++layer) {
        for (int rank = 1; rank < kGpus; ++rank) {
            if (out->d_attn_norm_weight_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_attn_norm_weight_rank[layer][rank]));
            }
            if (out->d_ffn_norm_weight_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_ffn_norm_weight_rank[layer][rank]));
            }
        }
        for (int rank = 0; rank < kGpus; ++rank) {
            if (out->d_attn_fn_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_attn_fn_rank[layer][rank]));
            }
            if (out->d_attn_base_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_attn_base_rank[layer][rank]));
            }
            if (out->d_attn_scale_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_attn_scale_rank[layer][rank]));
            }
            if (out->d_ffn_fn_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_ffn_fn_rank[layer][rank]));
            }
            if (out->d_ffn_base_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_ffn_base_rank[layer][rank]));
            }
            if (out->d_ffn_scale_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_ffn_scale_rank[layer][rank]));
            }
        }
        if (out->d_router_hash[layer]) CHECK_CUDA(cudaFree(out->d_router_hash[layer]));
        if (out->d_router_bias[layer]) CHECK_CUDA(cudaFree(out->d_router_bias[layer]));
        for (int rank = 0; rank < kGpus; ++rank) {
            if (out->d_router_w_ep[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_router_w_ep[layer][rank]));
            }
            if (out->d_router_w_shard[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_router_w_shard[layer][rank]));
            }
        }
        CHECK_CUDA(cudaSetDevice(out->devices[0]));
        if (out->d_router_w[layer]) CHECK_CUDA(cudaFree(out->d_router_w[layer]));
        if (out->d_indexer_compress_norm[layer]) CHECK_CUDA(cudaFree(out->d_indexer_compress_norm[layer]));
        if (out->d_indexer_compress_ape[layer]) CHECK_CUDA(cudaFree(out->d_indexer_compress_ape[layer]));
        if (out->d_attn_compress_norm[layer]) CHECK_CUDA(cudaFree(out->d_attn_compress_norm[layer]));
        if (out->d_attn_compress_ape[layer]) CHECK_CUDA(cudaFree(out->d_attn_compress_ape[layer]));
        if (out->d_attn_sinks[layer]) CHECK_CUDA(cudaFree(out->d_attn_sinks[layer]));
        if (out->d_kv_a_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_kv_a_norm_weight[layer]));
        if (out->d_q_a_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_q_a_norm_weight[layer]));
        if (out->d_attn_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_attn_norm_weight[layer]));
        if (out->d_ffn_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_ffn_norm_weight[layer]));
        if (out->d_ffn_scale[layer]) CHECK_CUDA(cudaFree(out->d_ffn_scale[layer]));
        if (out->d_ffn_base[layer]) CHECK_CUDA(cudaFree(out->d_ffn_base[layer]));
        if (out->d_ffn_fn[layer]) CHECK_CUDA(cudaFree(out->d_ffn_fn[layer]));
        if (out->d_attn_scale[layer]) CHECK_CUDA(cudaFree(out->d_attn_scale[layer]));
        if (out->d_attn_base[layer]) CHECK_CUDA(cudaFree(out->d_attn_base[layer]));
        if (out->d_attn_fn[layer]) CHECK_CUDA(cudaFree(out->d_attn_fn[layer]));
    }
    if (out->d_router_weights) CHECK_CUDA(cudaFree(out->d_router_weights));
    if (out->d_router_selected) CHECK_CUDA(cudaFree(out->d_router_selected));
    if (out->d_router_logits) CHECK_CUDA(cudaFree(out->d_router_logits));
    if (out->d_router_tokens) CHECK_CUDA(cudaFree(out->d_router_tokens));
    if (out->d_router_active) CHECK_CUDA(cudaFree(out->d_router_active));
    if (out->d_index_comp_score_full) CHECK_CUDA(cudaFree(out->d_index_comp_score_full));
    if (out->d_index_comp_kv_full) CHECK_CUDA(cudaFree(out->d_index_comp_kv_full));
    if (out->d_indexer_w_full) CHECK_CUDA(cudaFree(out->d_indexer_w_full));
    if (out->d_indexer_q_full) CHECK_CUDA(cudaFree(out->d_indexer_q_full));
    if (out->d_attn_comp_score_full) CHECK_CUDA(cudaFree(out->d_attn_comp_score_full));
    if (out->d_attn_comp_kv_full) CHECK_CUDA(cudaFree(out->d_attn_comp_kv_full));
    if (out->d_ffn_normed) CHECK_CUDA(cudaFree(out->d_ffn_normed));
    if (out->d_kv_normed) CHECK_CUDA(cudaFree(out->d_kv_normed));
    if (out->d_kv_full) CHECK_CUDA(cudaFree(out->d_kv_full));
    if (out->d_q_a_normed) CHECK_CUDA(cudaFree(out->d_q_a_normed));
    if (out->d_q_a_full) CHECK_CUDA(cudaFree(out->d_q_a_full));
    if (out->d_attn_normed) CHECK_CUDA(cudaFree(out->d_attn_normed));
    if (out->d_current_full) CHECK_CUDA(cudaFree(out->d_current_full));
    if (out->d_split) CHECK_CUDA(cudaFree(out->d_split));
    if (out->d_mix) CHECK_CUDA(cudaFree(out->d_mix));
    if (out->d_hc_norm) CHECK_CUDA(cudaFree(out->d_hc_norm));
    if (out->d_hc) CHECK_CUDA(cudaFree(out->d_hc));
    *out = SharedHcControls{};
}

int upload_model_router_route_plan(const Options &opt,
                                   RankState ranks[kGpus],
                                   const std::vector<int> &selected,
                                   const std::vector<float> &weights);
int upload_model_router_route_plan_async(const Options &opt,
                                         RankState ranks[kGpus],
                                         const int *selected,
                                         const float *weights,
                                         RoutePlanHostWorkspace *ws);
int upload_model_router_route_plan_gpu(const Options &opt,
                                       SharedHcControls *hc,
                                       RankState ranks[kGpus]);
int enqueue_dense_wait_after_rank_stream(RankState ranks[kGpus]);
int enqueue_control_wait_after_rank_streams(const Options &opt,
                                            RankState ranks[kGpus],
                                            cudaStream_t control_stream);
int enqueue_control_wait_after_dense_streams(const Options &opt,
                                             RankState ranks[kGpus],
                                             cudaStream_t control_stream);

int next_graph_order_event_slot(RankState ranks[kGpus]) {
    const int slot = ranks[0].graph_event_cursor % kGraphOrderEventSlots;
    ranks[0].graph_event_cursor =
        (ranks[0].graph_event_cursor + 1) % kGraphOrderEventSlots;
    return slot;
}

cudaEvent_t graph_stream_done_event(RankState &r, int slot) {
    cudaEvent_t ev = r.graph_stream_done[slot % kGraphOrderEventSlots];
    return ev ? ev : r.stream_done;
}

cudaEvent_t graph_dense_done_event(RankState &r, int slot) {
    cudaEvent_t ev = r.graph_dense_done[slot % kGraphOrderEventSlots];
    return ev ? ev : r.dense_done;
}

int run_model_router_dense_logits(const Options &opt,
                                  SharedHcControls *hc,
                                  int layer,
                                  cudaStream_t stream) {
    if (!hc || !hc->d_router_w[layer] || !hc->d_router_logits ||
        !hc->d_ffn_normed) {
        return 1;
    }
    if (!opt.router_cublas_gate) {
        const dim3 router_grid((unsigned int)kGlobalExperts,
                               (unsigned int)opt.slots, 1u);
        f32_dense_colmajor_kernel<<<router_grid, 256, 0, stream>>>(
            hc->d_router_logits, hc->d_router_w[layer], hc->d_ffn_normed,
            (uint32_t)kGlobalExperts, (uint32_t)kHidden, (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
        return 0;
    }
    if (!hc->router_blas) return 2;
    cublasStatus_t st = cublasSetStream(hc->router_blas, stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "router cublasSetStream failed status=%d\n", (int)st);
        return 3;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    st = cublasSgemm(hc->router_blas,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     kGlobalExperts, opt.slots, kHidden,
                     &alpha,
                     hc->d_router_w[layer], kGlobalExperts,
                     hc->d_ffn_normed, kHidden,
                     &beta,
                     hc->d_router_logits, kGlobalExperts);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "router cublasSgemm failed layer=%d status=%d\n",
                     layer, (int)st);
        return 4;
    }
    return 0;
}

int run_model_router_rank_major_logits(const Options &opt,
                                       SharedHcControls *hc,
                                       RankState ranks[kGpus],
                                       int layer,
                                       cudaStream_t control_stream,
                                       bool post_attention_input) {
    if (!opt.model_router_rank_major_logits_gate) return 0;
    if (!hc || !hc->d_router_logits || layer < 0 || layer >= 43) return 1;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.compose_nccl_initialized || !r.compose_nccl ||
            !(post_attention_input ? r.d_post_attn_full_rank_major
                                   : r.d_current_full_rank_major) ||
            !r.d_rank_major_norm_scale ||
            !r.d_router_logits_shard || !r.d_router_logits_rank_major ||
            !hc->d_ffn_norm_weight_rank[layer][rank] ||
            !hc->d_router_w_ep[layer][rank]) {
            return 2;
        }
    }
    const uint32_t shard_cols = (uint32_t)(kHidden / kGpus);
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        const float *rank_major = post_attention_input
            ? r.d_post_attn_full_rank_major
            : r.d_current_full_rank_major;
        CHECK_CUDA(cudaSetDevice(r.device));
        rank_major_norm_scale_kernel<<<
            (unsigned int)opt.slots, 256, 0, r.stream>>>(
            r.d_rank_major_norm_scale, rank_major,
            shard_cols, (uint32_t)kGpus, (uint32_t)opt.slots, 1.0e-6f);
        const dim3 grid((unsigned int)kLocalExperts, (unsigned int)opt.slots, 1u);
        router_logits_ep_from_rank_major_kernel<<<grid, 256, 0, r.stream>>>(
            r.d_router_logits_shard, rank_major,
            hc->d_ffn_norm_weight_rank[layer][rank],
            r.d_rank_major_norm_scale,
            hc->d_router_w_ep[layer][rank],
            shard_cols, (uint32_t)kGpus, (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclAllGather(r.d_router_logits_shard,
                                 r.d_router_logits_rank_major,
                                 (size_t)opt.slots * kLocalExperts,
                                 ncclFloat,
                                 r.compose_nccl,
                                 r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    if (opt.decode_cudagraph_gate) {
        if (enqueue_control_wait_after_rank_streams(
                opt, ranks, control_stream) != 0) {
            return 3;
        }
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    const uint64_t elems = (uint64_t)opt.slots * (uint64_t)kGlobalExperts;
    router_logits_rank_major_to_slot_major_kernel<<<
        (unsigned int)((elems + 255ull) / 256ull), 256, 0, control_stream>>>(
        hc->d_router_logits, ranks[0].d_router_logits_rank_major,
        (uint32_t)opt.slots);
    CHECK_CUDA(cudaGetLastError());
    return 0;
}

int run_model_router_allreduce_logits(const Options &opt,
                                      SharedHcControls *hc,
                                      RankState ranks[kGpus],
                                      int layer,
                                      cudaStream_t control_stream,
                                      bool post_attention_input) {
    if (!opt.model_router_allreduce_logits_gate) return 0;
    if (opt.decode_cudagraph_gate) return 11;
    if (!hc || !hc->d_router_logits || layer < 0 || layer >= 43) return 1;
    const uint32_t shard_cols = (uint32_t)(kHidden / kGpus);
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.compose_nccl_initialized || !r.compose_nccl ||
            !(post_attention_input ? r.d_post_attn_shard : r.d_current_shard) ||
            !r.d_hc_reduce_max ||
            !r.d_hc_reduce_sumsq || !r.d_router_logits_rank_major ||
            !hc->d_ffn_norm_weight_rank[layer][rank] ||
            !hc->d_router_w_shard[layer][rank]) {
            return 2;
        }
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        const float *input_shard = post_attention_input
            ? r.d_post_attn_shard
            : r.d_current_shard;
        CHECK_CUDA(cudaSetDevice(r.device));
        current_shard_max_kernel<<<
            (unsigned int)opt.slots, 256, 0, r.stream>>>(
            r.d_hc_reduce_max, input_shard, shard_cols,
            (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclAllReduce(r.d_hc_reduce_max, r.d_hc_reduce_max,
                                 (size_t)opt.slots, ncclFloat, ncclMax,
                                 r.compose_nccl, r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        const float *input_shard = post_attention_input
            ? r.d_post_attn_shard
            : r.d_current_shard;
        CHECK_CUDA(cudaSetDevice(r.device));
        current_shard_stable_sumsq_kernel<<<
            (unsigned int)opt.slots, 256, 0, r.stream>>>(
            r.d_hc_reduce_sumsq, input_shard, r.d_hc_reduce_max,
            shard_cols, (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclAllReduce(r.d_hc_reduce_sumsq, r.d_hc_reduce_sumsq,
                                 (size_t)opt.slots, ncclFloat, ncclSum,
                                 r.compose_nccl, r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        const float *input_shard = post_attention_input
            ? r.d_post_attn_shard
            : r.d_current_shard;
        CHECK_CUDA(cudaSetDevice(r.device));
        const dim3 grid((unsigned int)kGlobalExperts,
                        (unsigned int)opt.slots, 1u);
        router_logits_allreduce_partial_kernel<<<grid, 256, 0, r.stream>>>(
            r.d_router_logits_rank_major, input_shard,
            hc->d_ffn_norm_weight_rank[layer][rank], r.d_hc_reduce_max,
            r.d_hc_reduce_sumsq, hc->d_router_w_shard[layer][rank],
            (uint32_t)rank, shard_cols, (uint32_t)opt.slots, 1.0e-6f);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclAllReduce(r.d_router_logits_rank_major,
                                 r.d_router_logits_rank_major,
                                 (size_t)opt.slots * kGlobalExperts,
                                 ncclFloat, ncclSum, r.compose_nccl,
                                 r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaMemcpyAsync(hc->d_router_logits,
                               ranks[0].d_router_logits_rank_major,
                               (size_t)opt.slots * kGlobalExperts *
                                   sizeof(float),
                               cudaMemcpyDeviceToDevice,
                               control_stream));
    CHECK_CUDA(cudaStreamSynchronize(control_stream));
    return 0;
}

int run_shared_hc_final_expand(const Options &opt,
                               SharedHcControls *hc,
                               RankState ranks[kGpus],
                               int layer) {
    if (!hc || !hc->initialized || hc->slots != opt.slots ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    const uint64_t shard_elems =
        (uint64_t)opt.slots * (uint64_t)(kHidden / kGpus);
    const uint64_t hc_shard_elems = shard_elems * kHcRows;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    cudaStream_t control_stream = graph_event_order ? ranks[0].stream : (cudaStream_t)0;
    auto control_wait_on_rank_streams = [&]() -> int {
        if (!graph_event_order) return 0;
        const int slot = next_graph_order_event_slot(ranks);
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            cudaEvent_t ev = graph_stream_done_event(r, slot);
            if (!ev) return 1;
            CHECK_CUDA(cudaEventRecord(ev, r.stream));
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaStreamWaitEvent(control_stream,
                                           graph_stream_done_event(ranks[rank],
                                                                   slot),
                                           0));
        }
        return 0;
    };
    auto rank_streams_wait_on_control = [&]() -> int {
        if (!graph_event_order) return 0;
        const int slot = next_graph_order_event_slot(ranks);
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        cudaEvent_t ev = graph_stream_done_event(ranks[0], slot);
        if (!ev) return 1;
        CHECK_CUDA(cudaEventRecord(ev, control_stream));
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_CUDA(cudaStreamWaitEvent(r.stream, ev, 0));
        }
        return 0;
    };
    if (graph_event_order) {
        if (control_wait_on_rank_streams() != 0) return 4;
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        if (!ranks[rank].d_final_hc_shard) return 2;
        gather_hc_shard_to_full_kernel<<<
            (unsigned int)((hc_shard_elems + 255) / 256), 256, 0,
            control_stream>>>(
            hc->d_hc, ranks[rank].d_final_hc_shard, rank, (uint32_t)opt.slots);
    }
    CHECK_CUDA(cudaGetLastError());
    if (!graph_event_order) {
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    rms_norm_plain_rows_stable_kernel<<<
        (unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_hc_norm, hc->d_hc, kHcRows * (uint32_t)kHidden,
        (uint32_t)opt.slots, 1.0e-6f);
    const dim3 mix_grid((unsigned int)kHcMix, (unsigned int)opt.slots, 1u);
    f32_dense_colmajor_kernel<<<mix_grid, 256, 0, control_stream>>>(
        hc->d_mix, hc->d_ffn_fn[layer], hc->d_hc_norm,
        (uint32_t)kHcMix, kHcRows * (uint32_t)kHidden, (uint32_t)opt.slots);
    hc_split_rows_kernel<<<
        (unsigned int)(((uint64_t)opt.slots + 255) / 256), 256, 0,
        control_stream>>>(
        hc->d_split, hc->d_mix, hc->d_ffn_scale[layer], hc->d_ffn_base[layer],
        (uint32_t)opt.slots, opt.reference_hc_reduce_gate ? 20u : 4u);
    CHECK_CUDA(cudaGetLastError());
    if (graph_event_order) {
        if (rank_streams_wait_on_control() != 0) return 5;
    } else {
        CHECK_CUDA(cudaDeviceSynchronize());
        void *dsts[kGpus] = {};
        for (int rank = 0; rank < kGpus; ++rank) {
            dsts[rank] = ranks[rank].d_hc_split;
        }
        if (nccl_broadcast_bytes_from_rank0(
                ranks, hc->d_split, dsts,
                (size_t)opt.slots * kHcMix * sizeof(float),
                "hc_final_split") != 0) {
            return 6;
        }
    }

    const int block = 256;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_hc_scratch_shard || !r.d_hc_split) return 3;
        if (graph_event_order) {
            enqueue_graph_f32_copy_from_device0(
                opt, r, rank, r.d_hc_split, hc->d_split,
                (uint64_t)opt.slots * kHcMix, r.stream, block);
        }
        const int grid = (int)((hc_shard_elems + block - 1) / block);
        hc_expand_shard_kernel<<<grid, block, 0, r.stream>>>(
            r.d_hc_scratch_shard, r.d_next_hidden, r.d_final_hc_shard,
            r.d_hc_split, (uint32_t)opt.slots);
        if (opt.reference_hc_state_guard_gate) {
            clamp_f32_abs_kernel<<<grid, block, 0, r.stream>>>(
                r.d_hc_scratch_shard, hc_shard_elems,
                kReferenceHcStateTargetAbs);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    if (graph_event_order) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            std::swap(r.d_final_hc_shard, r.d_hc_scratch_shard);
            r.hc_initialized = true;
        }
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_CUDA(cudaStreamSynchronize(r.stream));
            std::swap(r.d_final_hc_shard, r.d_hc_scratch_shard);
            r.hc_initialized = true;
        }
    }
    return 0;
}

int run_shared_hc_current_input(const Options &opt,
                                SharedHcControls *hc,
                                RankState ranks[kGpus],
                                const ResidentF8Dense &attn_op,
                                const ResidentF8Dense &shared_op,
                                int layer,
                                bool reuse_model_router_route_plan,
                                HcCurrentInputBreakdown *breakdown) {
    if (!hc || !hc->initialized || hc->slots != opt.slots ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    if (attn_op.cols <= 0 || shared_op.cols <= 0) return 2;
    const uint64_t shard_elems =
        (uint64_t)opt.slots * (uint64_t)(kHidden / kGpus);
    const uint64_t hc_shard_elems = shard_elems * kHcRows;
    const uint64_t full_elems = (uint64_t)opt.slots * kHidden;
    const int block = 256;
    const auto t_start = std::chrono::steady_clock::now();
    const bool graph_event_order = opt.decode_cudagraph_gate;
    cudaStream_t control_stream =
        (opt.tp_hc_current_input_stream_sync_gate || graph_event_order)
            ? ranks[0].stream
            : (cudaStream_t)0;
    auto control_wait_on_rank_streams = [&]() -> int {
        if (!graph_event_order) return 0;
        const int slot = next_graph_order_event_slot(ranks);
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            cudaEvent_t ev = graph_stream_done_event(r, slot);
            if (!ev) return 1;
            CHECK_CUDA(cudaEventRecord(ev, r.stream));
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaStreamWaitEvent(control_stream,
                                           graph_stream_done_event(ranks[rank],
                                                                   slot),
                                           0));
        }
        return 0;
    };
    auto rank_streams_wait_on_control = [&]() -> int {
        if (!graph_event_order) return 0;
        const int slot = next_graph_order_event_slot(ranks);
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        cudaEvent_t ev = graph_stream_done_event(ranks[0], slot);
        if (!ev) return 1;
        CHECK_CUDA(cudaEventRecord(ev, control_stream));
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_CUDA(cudaStreamWaitEvent(r.stream, ev, 0));
        }
        return 0;
    };
    auto sync_control_device = [&]() {
        if (graph_event_order) return;
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        if (opt.tp_hc_current_input_stream_sync_gate) {
            CHECK_CUDA(cudaStreamSynchronize(control_stream));
        } else {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    };

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_final_hc_shard || !r.d_current_shard || !r.d_current_full ||
            !r.d_hc_split) {
            return 3;
        }
        if (!r.hc_initialized) {
            seed_initial_hc_shard_kernel<<<
                (unsigned int)((hc_shard_elems + block - 1) / block), block,
                0, r.stream>>>(r.d_final_hc_shard, rank, opt.slots);
            CHECK_CUDA(cudaGetLastError());
            r.hc_initialized = true;
        }
    }
    if (graph_event_order) {
        if (control_wait_on_rank_streams() != 0) return 6;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    auto t_seed_done = std::chrono::steady_clock::now();
    if (should_log_reference_hc_window(opt)) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("hc_current_shard", layer, rank,
                                 ranks[rank].d_current_shard,
                                 (size_t)shard_elems, ranks[rank].stream);
        }
    }

    if (opt.tp_hc_current_allreduce_gate) {
        if (graph_event_order) return 11;
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            if (!r.compose_nccl_initialized || !r.compose_nccl ||
                !r.d_hc_reduce_max || !r.d_hc_reduce_sumsq ||
                !r.d_hc_reduce_mix || !hc->d_attn_fn_rank[layer][rank] ||
                !hc->d_attn_scale_rank[layer][rank] ||
                !hc->d_attn_base_rank[layer][rank]) {
                return 12;
            }
            CHECK_CUDA(cudaSetDevice(r.device));
            const dim3 partial_grid((unsigned int)(kHcMix + 1),
                                    (unsigned int)opt.slots, 1u);
            hc_local_max_mix_partial_kernel<<<partial_grid, 256, 0, r.stream>>>(
                r.d_hc_reduce_max, r.d_hc_reduce_mix, r.d_final_hc_shard,
                hc->d_attn_fn_rank[layer][rank], (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_NCCL(ncclGroupStart());
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_NCCL(ncclAllReduce(r.d_hc_reduce_max, r.d_hc_reduce_max,
                                     (size_t)opt.slots, ncclFloat, ncclMax,
                                     r.compose_nccl, r.stream));
            CHECK_NCCL(ncclAllReduce(r.d_hc_reduce_mix, r.d_hc_reduce_mix,
                                     (size_t)opt.slots * kHcMix, ncclFloat,
                                     ncclSum, r.compose_nccl, r.stream));
        }
        CHECK_NCCL(ncclGroupEnd());
        bool have_ref_mix_for_full_parity = false;
        if (opt.tp_hc_current_full_parity_gate) {
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
            for (int rank = 0; rank < kGpus; ++rank) {
                gather_hc_shard_to_full_kernel<<<
                    (unsigned int)((hc_shard_elems + block - 1) / block),
                    block, 0, control_stream>>>(
                    hc->d_hc, ranks[rank].d_final_hc_shard, rank,
                    (uint32_t)opt.slots);
            }
            CHECK_CUDA(cudaGetLastError());
            sync_control_device();
            rms_norm_plain_rows_stable_kernel<<<
                (unsigned int)opt.slots, 256, 0, control_stream>>>(
                hc->d_hc_norm, hc->d_hc, kHcRows * (uint32_t)kHidden,
                (uint32_t)opt.slots, 1.0e-6f);
            const dim3 ref_mix_grid((unsigned int)kHcMix,
                                    (unsigned int)opt.slots, 1u);
            f32_dense_colmajor_kernel<<<ref_mix_grid, 256, 0, control_stream>>>(
                hc->d_split, hc->d_attn_fn[layer], hc->d_hc_norm,
                (uint32_t)kHcMix, kHcRows * (uint32_t)kHidden,
                (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
            sync_control_device();
            have_ref_mix_for_full_parity = true;
        }
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            hc_local_stable_sumsq_kernel<<<
                (unsigned int)opt.slots, 256, 0, r.stream>>>(
                r.d_hc_reduce_sumsq, r.d_final_hc_shard, r.d_hc_reduce_max,
                (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_NCCL(ncclGroupStart());
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_NCCL(ncclAllReduce(r.d_hc_reduce_sumsq, r.d_hc_reduce_sumsq,
                                     (size_t)opt.slots, ncclFloat, ncclSum,
                                     r.compose_nccl, r.stream));
        }
        CHECK_NCCL(ncclGroupEnd());
        if (have_ref_mix_for_full_parity) {
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
            hc_scale_reduced_mix_kernel<<<
                (unsigned int)(((uint64_t)opt.slots * kHcMix + 255) / 256),
                256, 0, control_stream>>>(
                hc->d_mix, ranks[0].d_hc_reduce_max,
                ranks[0].d_hc_reduce_sumsq, ranks[0].d_hc_reduce_mix,
                (uint32_t)opt.slots, 1.0e-6f);
            CHECK_CUDA(cudaGetLastError());
            sync_control_device();
            const TensorF32DiffStats diff = collect_tensor_f32_diff_stats(
                hc->d_mix, hc->d_split, (size_t)opt.slots * kHcMix,
                control_stream);
            std::printf("tp_ep_hc_current_allreduce_mix_diff\tlayer\t%d\t"
                        "slots\t%d\tmax_abs_diff\t%.9g\tmax_rel_diff\t%.9g\t"
                        "diff_bad\t%d\tfirst_bad\t%zu\t%s\n",
                        layer, opt.slots, diff.max_abs, diff.max_rel,
                        diff.bad, diff.first_bad,
                        (diff.max_abs <= 1.0e-4f ||
                         diff.max_rel <= 1.0e-4f) ? "PASS" : "WARN");
        }
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            hc_apply_reduced_mix_split_kernel<<<
                (unsigned int)(((uint64_t)opt.slots + 255) / 256), 256, 0,
                r.stream>>>(
                r.d_hc_split, r.d_hc_reduce_max, r.d_hc_reduce_sumsq,
                r.d_hc_reduce_mix, hc->d_attn_scale_rank[layer][rank],
                hc->d_attn_base_rank[layer][rank], (uint32_t)opt.slots,
                opt.reference_hc_reduce_gate ? 20u : 4u, 1.0e-6f);
            CHECK_CUDA(cudaGetLastError());
        }
    } else {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            gather_hc_shard_to_full_kernel<<<
                (unsigned int)((hc_shard_elems + block - 1) / block), block,
                0, control_stream>>>(
                hc->d_hc, ranks[rank].d_final_hc_shard, rank, (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();

        rms_norm_plain_rows_stable_kernel<<<(unsigned int)opt.slots, 256, 0, control_stream>>>(
            hc->d_hc_norm, hc->d_hc, kHcRows * (uint32_t)kHidden,
            (uint32_t)opt.slots, 1.0e-6f);
        const dim3 mix_grid((unsigned int)kHcMix, (unsigned int)opt.slots, 1u);
        f32_dense_colmajor_kernel<<<mix_grid, 256, 0, control_stream>>>(
            hc->d_mix, hc->d_attn_fn[layer], hc->d_hc_norm,
            (uint32_t)kHcMix, kHcRows * (uint32_t)kHidden, (uint32_t)opt.slots);
        hc_split_rows_kernel<<<
            (unsigned int)(((uint64_t)opt.slots + 255) / 256), 256, 0,
            control_stream>>>(
            hc->d_split, hc->d_mix, hc->d_attn_scale[layer], hc->d_attn_base[layer],
            (uint32_t)opt.slots, opt.reference_hc_reduce_gate ? 20u : 4u);
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();
        if (graph_event_order) {
            if (rank_streams_wait_on_control() != 0) return 7;
        }
    }

    if (!opt.tp_hc_current_allreduce_gate && !graph_event_order) {
        void *dsts[kGpus] = {};
        for (int rank = 0; rank < kGpus; ++rank) {
            dsts[rank] = ranks[rank].d_hc_split;
        }
        if (nccl_broadcast_bytes_from_rank0(
                ranks, hc->d_split, dsts,
                (size_t)opt.slots * kHcMix * sizeof(float),
                "hc_current_split") != 0) {
            return 10;
        }
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (opt.tp_hc_current_allreduce_gate) {
            // Split is already resident on this rank from the NCCL all-reduce path.
        } else if (graph_event_order) {
            enqueue_graph_f32_copy_from_device0(
                opt, r, rank, r.d_hc_split, hc->d_split,
                (uint64_t)opt.slots * kHcMix, r.stream, block);
        }
        hc_weighted_sum_shard_kernel<<<
            (unsigned int)((shard_elems + block - 1) / block), block,
            0, r.stream>>>(r.d_current_shard, r.d_final_hc_shard,
                           r.d_hc_split, (uint32_t)opt.slots,
                           opt.reference_hc_reduce_gate ? 1 : 0);
        CHECK_CUDA(cudaGetLastError());
    }
    auto t_split_done = std::chrono::steady_clock::now();
    if (graph_event_order) {
        if (control_wait_on_rank_streams() != 0) return 8;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    auto t_weighted_done = std::chrono::steady_clock::now();

    float *control_current_full = hc->d_current_full;
    const bool peer_gather_current = opt.tp_hc_current_input_peer_gather_gate;
    const bool nccl_gather_current =
        opt.tp_hc_current_input_nccl_allgather_gate;
    const bool rank_local_current_full =
        peer_gather_current || nccl_gather_current;
    if (nccl_gather_current) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            if (!r.compose_nccl_initialized || !r.compose_nccl ||
                !r.d_current_full_rank_major) {
                return 9;
            }
        }
        CHECK_NCCL(ncclGroupStart());
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_NCCL(ncclAllGather(r.d_current_shard,
                                     r.d_current_full_rank_major,
                                     (size_t)shard_elems,
                                     ncclFloat,
                                     r.compose_nccl,
                                     r.stream));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            rank_major_current_shards_to_slot_major_kernel<<<
                (unsigned int)((full_elems + block - 1) / block), block, 0,
                r.stream>>>(
                r.d_current_full, r.d_current_full_rank_major,
                (uint32_t)(kHidden / kGpus), (uint32_t)kGpus,
                (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        if (graph_event_order) {
            if (control_wait_on_rank_streams() != 0) return 9;
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
        control_current_full = ranks[0].d_current_full;
    } else if (peer_gather_current) {
        const uint64_t full_grid_elems = full_elems;
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            gather_current_shards_to_full8_kernel<<<
                (unsigned int)((full_grid_elems + block - 1) / block), block,
                0, r.stream>>>(r.d_current_full,
                               ranks[0].d_current_shard,
                               ranks[1].d_current_shard,
                               ranks[2].d_current_shard,
                               ranks[3].d_current_shard,
                               ranks[4].d_current_shard,
                               ranks[5].d_current_shard,
                               ranks[6].d_current_shard,
                               ranks[7].d_current_shard,
                               (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        if (graph_event_order) {
            if (control_wait_on_rank_streams() != 0) return 9;
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
        control_current_full = ranks[0].d_current_full;
    } else {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            gather_current_shard_to_full_kernel<<<
                (unsigned int)((shard_elems + block - 1) / block), block,
                0, control_stream>>>(
                hc->d_current_full, ranks[rank].d_current_shard, rank,
                (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();
    }
    auto t_gather_done = std::chrono::steady_clock::now();
    if (opt.tp_hc_current_full_parity_gate && rank_local_current_full) {
        log_hc_current_full_rank_parity(opt, ranks, layer, (size_t)full_elems);
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    if (should_log_reference_hc_window(opt)) {
        log_tensor_f32_stats("hc_current_full", layer, 0, control_current_full,
                             (size_t)full_elems, nullptr);
    }

    if (!hc->d_ffn_normed || !hc->d_ffn_norm_weight[layer]) return 4;
    rms_norm_weight_rows_stable_kernel<<<(unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_ffn_normed, control_current_full, hc->d_ffn_norm_weight[layer],
        (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());
    sync_control_device();
    auto t_norm_done = std::chrono::steady_clock::now();
    if (graph_event_order) {
        if (rank_streams_wait_on_control() != 0) return 10;
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    }
    if (should_log_reference_hc_window(opt)) {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        log_tensor_f32_stats("hc_ffn_normed", layer, 0, hc->d_ffn_normed,
                             (size_t)full_elems, nullptr);
    }

    auto t_router_select_done = t_norm_done;
    auto t_router_d2h_done = t_norm_done;
    auto t_route_upload_done = t_norm_done;
    if (opt.model_router_routes && reuse_model_router_route_plan) {
        int total_routes = 0;
        for (int rank = 0; rank < kGpus; ++rank) total_routes += ranks[rank].routes;
        if (total_routes <= 0) return 5;
        t_router_select_done = t_norm_done;
        t_router_d2h_done = t_norm_done;
        t_route_upload_done = t_norm_done;
    } else if (opt.model_router_routes) {
        if ((!opt.model_router_rank_major_logits_gate &&
             !opt.model_router_allreduce_logits_gate &&
             !hc->d_router_w[layer]) ||
            !hc->d_router_logits ||
            !hc->d_router_selected || !hc->d_router_weights) {
            return 4;
        }
        const int router_dense_rc = opt.model_router_allreduce_logits_gate
            ? run_model_router_allreduce_logits(opt, hc, ranks, layer,
                                                control_stream, false)
            : (opt.model_router_rank_major_logits_gate
                   ? run_model_router_rank_major_logits(opt, hc, ranks, layer,
                                                        control_stream, false)
                   : run_model_router_dense_logits(opt, hc, layer,
                                                   control_stream));
        if (router_dense_rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_model_router_dense_failed\tlayer\t%d\trc\t%d\n",
                         layer, router_dense_rc);
            return 4;
        }
        if (opt.router_hash_fast_gate && hc->d_router_hash[layer] &&
            hc->d_router_tokens && hc->router_hash_rows[layer] > 0u) {
            router_select_hash_fast_rows_kernel<<<
                (unsigned int)opt.slots, 1, 0, control_stream>>>(
                hc->d_router_selected, hc->d_router_weights,
                hc->d_router_logits, hc->d_router_hash[layer],
                hc->d_router_tokens, hc->d_router_active,
                hc->router_hash_rows[layer], (uint32_t)opt.slots);
        } else {
            router_select_topk_rows_kernel<<<
                (unsigned int)opt.slots, 1, 0, control_stream>>>(
                hc->d_router_selected, hc->d_router_weights,
                hc->d_router_logits, hc->d_router_bias[layer],
                hc->d_router_hash[layer], hc->d_router_tokens,
                hc->d_router_active, hc->router_hash_rows[layer],
                (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();
        t_router_select_done = std::chrono::steady_clock::now();
        int route_rc = 0;
        if (opt.gpu_route_plan_gate) {
            t_router_d2h_done = t_router_select_done;
            route_rc = upload_model_router_route_plan_gpu(opt, hc, ranks);
        } else if (opt.route_plan_async_upload_gate) {
            RoutePlanHostWorkspace *ws = &hc->route_plan_ws;
            if (!ws->initialized) return 5;
            const size_t route_elems = (size_t)opt.slots * (size_t)opt.top_k;
            CHECK_CUDA(cudaMemcpyAsync(ws->h_selected, hc->d_router_selected,
                                       route_elems * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       control_stream));
            CHECK_CUDA(cudaMemcpyAsync(ws->h_weights, hc->d_router_weights,
                                       route_elems * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       control_stream));
            CHECK_CUDA(cudaStreamSynchronize(control_stream));
            t_router_d2h_done = std::chrono::steady_clock::now();
            route_rc = upload_model_router_route_plan_async(
                opt, ranks, ws->h_selected, ws->h_weights, ws);
        } else {
            std::vector<int> selected((size_t)opt.slots * (size_t)opt.top_k);
            std::vector<float> weights((size_t)opt.slots * (size_t)opt.top_k);
            CHECK_CUDA(cudaMemcpy(selected.data(), hc->d_router_selected,
                                  selected.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(weights.data(), hc->d_router_weights,
                                  weights.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            t_router_d2h_done = std::chrono::steady_clock::now();
            route_rc = upload_model_router_route_plan(opt, ranks,
                                                      selected, weights);
        }
        if (route_rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_model_router_route_plan_failed\tlayer\t%d\trc\t%d\n",
                         layer, route_rc);
            return 5;
        }
        t_route_upload_done = std::chrono::steady_clock::now();
    } else {
        t_router_select_done = t_norm_done;
        t_router_d2h_done = t_norm_done;
        t_route_upload_done = t_norm_done;
    }
    auto t_router_done = t_route_upload_done;

    const bool fused_fill_pack =
        opt.tp_hc_current_input_fused_fill_pack_gate &&
        !rank_local_current_full && !graph_event_order &&
        !opt.reference_hc_reduce_gate &&
        (!opt.routed_ffn_norm_input_gate || hc->d_ffn_normed);
    if (!fused_fill_pack && !rank_local_current_full) {
        const int bcast_rc = nccl_broadcast_f32_from_device0_to_current_full(
            opt, ranks, hc->d_current_full, full_elems,
            "hc_current_full_input");
        if (bcast_rc != 0) return 12;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        const uint64_t attn_elems = (uint64_t)opt.slots * (uint64_t)attn_op.cols;
        const uint64_t shared_elems = (uint64_t)opt.slots * (uint64_t)shared_op.cols;
        const uint64_t route_elems = (uint64_t)r.routes * kHidden;
        if (fused_fill_pack) {
            const float *state_src =
                (opt.routed_ffn_norm_input_gate && route_elems > 0)
                    ? hc->d_ffn_normed
                    : hc->d_current_full;
            const float *route_src =
                opt.routed_ffn_norm_input_gate ? hc->d_ffn_normed : hc->d_current_full;
            const uint64_t total = std::max(
                std::max(full_elems, attn_elems),
                std::max(shared_elems, route_elems));
            hc_current_fused_fill_pack_kernel<<<
                (unsigned int)((total + block - 1) / block), block,
                0, r.stream>>>(
                r.d_current_full, state_src, hc->d_current_full, route_src,
                attn_op.d_x[(size_t)rank], (uint32_t)attn_op.cols,
                shared_op.d_x[(size_t)rank], (uint32_t)shared_op.cols,
                attn_op.d_x_half[(size_t)rank],
                shared_op.d_x_half[(size_t)rank],
                route_elems > 0 ? r.d_a : nullptr,
                route_elems > 0 ? r.d_route_slots : nullptr,
                r.routes, (uint32_t)opt.slots, total);
            CHECK_CUDA(cudaGetLastError());
        } else {
            if (!rank_local_current_full) {
                // A4a: current-full transport is handled once above via NCCL.
            }
            if (attn_op.d_x[(size_t)rank]) {
                fill_dense_input_from_current_kernel<<<
                    (unsigned int)((attn_elems + block - 1) / block), block,
                    0, r.stream>>>(attn_op.d_x[(size_t)rank], r.d_current_full,
                                   (uint32_t)attn_op.cols, (uint32_t)opt.slots);
            }
            if (shared_op.d_x[(size_t)rank]) {
                fill_dense_input_from_current_kernel<<<
                    (unsigned int)((shared_elems + block - 1) / block), block,
                    0, r.stream>>>(shared_op.d_x[(size_t)rank], r.d_current_full,
                                   (uint32_t)shared_op.cols, (uint32_t)opt.slots);
            }
            if (attn_op.d_x_half[(size_t)rank]) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((attn_elems + block - 1) / block), block,
                    0, r.stream>>>(attn_op.d_x_half[(size_t)rank], r.d_current_full,
                                   (uint32_t)attn_op.cols, (uint32_t)opt.slots);
            }
            if (shared_op.d_x_half[(size_t)rank]) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((shared_elems + block - 1) / block), block,
                    0, r.stream>>>(shared_op.d_x_half[(size_t)rank],
                                   r.d_current_full, (uint32_t)shared_op.cols,
                                   (uint32_t)opt.slots);
            }
            if (opt.routed_ffn_norm_input_gate && route_elems > 0) {
                // Packed route input is emitted after the loop, once ffn_normed
                // has been broadcast to every rank by NCCL.
            }
            if (route_elems > 0 && !opt.routed_ffn_norm_input_gate) {
                if (opt.reference_hc_reduce_gate) {
                    pack_current_full_to_routes_scaled_kernel<<<
                        (unsigned int)r.routes, 256, 0, r.stream>>>(
                            r.d_a, r.d_route_inv_scale, r.d_current_full,
                            r.d_route_slots, r.routes, kReferenceRouteInputTargetAbs);
                } else {
                    pack_current_full_to_routes_kernel<<<
                        (unsigned int)((route_elems + block - 1) / block), block,
                        0, r.stream>>>(r.d_a, r.d_current_full, r.d_route_slots, r.routes);
                }
                CHECK_CUDA(cudaGetLastError());
            }
        }
        if (should_log_reference_hc_window(opt) && r.d_route_inv_scale && r.routes > 0) {
            log_tensor_f32_stats("route_inv_scale", layer, rank,
                                 r.d_route_inv_scale, (size_t)r.routes,
                                 r.stream);
        }
    }
    if (opt.routed_ffn_norm_input_gate) {
        const int bcast_rc = nccl_broadcast_f32_from_device0_to_current_full(
            opt, ranks, hc->d_ffn_normed, full_elems,
            "hc_current_ffn_normed_route_input");
        if (bcast_rc != 0) return 13;
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            const uint64_t route_elems = (uint64_t)r.routes * kHidden;
            if (route_elems == 0) continue;
            if (opt.reference_hc_reduce_gate) {
                pack_current_full_to_routes_scaled_kernel<<<
                    (unsigned int)r.routes, 256, 0, r.stream>>>(
                        r.d_a, r.d_route_inv_scale, r.d_current_full,
                        r.d_route_slots, r.routes, kReferenceRouteInputTargetAbs);
            } else {
                pack_current_full_to_routes_kernel<<<
                    (unsigned int)((route_elems + block - 1) / block), block,
                    0, r.stream>>>(r.d_a, r.d_current_full, r.d_route_slots, r.routes);
            }
            CHECK_CUDA(cudaGetLastError());
        }
    }
    if (graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 11;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    auto t_fill_done = std::chrono::steady_clock::now();
    if (breakdown) {
        breakdown->seed_ms +=
            std::chrono::duration<double, std::milli>(t_seed_done - t_start).count();
        breakdown->attn_mix_ms +=
            std::chrono::duration<double, std::milli>(t_split_done - t_seed_done).count();
        breakdown->split_ms +=
            std::chrono::duration<double, std::milli>(t_weighted_done - t_split_done).count();
        breakdown->gather_ms +=
            std::chrono::duration<double, std::milli>(t_gather_done - t_weighted_done).count();
        breakdown->ffn_router_ms +=
            std::chrono::duration<double, std::milli>(t_router_done - t_gather_done).count();
        breakdown->ffn_norm_ms +=
            std::chrono::duration<double, std::milli>(t_norm_done - t_gather_done).count();
        breakdown->router_select_ms +=
            std::chrono::duration<double, std::milli>(t_router_select_done - t_norm_done).count();
        breakdown->router_d2h_ms +=
            std::chrono::duration<double, std::milli>(t_router_d2h_done - t_router_select_done).count();
        breakdown->route_upload_ms +=
            std::chrono::duration<double, std::milli>(t_route_upload_done - t_router_d2h_done).count();
        breakdown->fill_pack_ms +=
            std::chrono::duration<double, std::milli>(t_fill_done - t_router_done).count();
    }
    return 0;
}

struct OutputHeadGateStats {
    bool pass = true;
    int slots = 0;
    int vocab = 0;
    int rows_per_gpu = 0;
    uint64_t output_weight_bytes = 0;
    uint64_t logits_bytes = 0;
    double total_ms = 0.0;
    double projection_ms = 0.0;
    double projection_kernel_worst_ms = 0.0;
    double host_reduce_ms = 0.0;
    uint32_t first_token = UINT32_MAX;
    float first_logit = 0.0f;
    uint64_t checksum = 0;
    int finite_bad = 0;
};

struct OutputHeadResidentGateStats {
    bool pass = true;
    int slots = 0;
    int vocab = 0;
    int rows_per_gpu = 0;
    int warmup = 0;
    int iters = 0;
    uint64_t output_weight_bytes = 0;
    uint64_t logits_bytes = 0;
    double load_ms = 0.0;
    double avg_total_ms = 0.0;
    double avg_hc_prep_ms = 0.0;
    double avg_broadcast_ms = 0.0;
    double avg_projection_wall_ms = 0.0;
    double avg_projection_kernel_worst_ms = 0.0;
    double avg_readback_reduce_ms = 0.0;
    double output_head_tok_s = 0.0;
    uint32_t first_token = UINT32_MAX;
    float first_logit = 0.0f;
    uint64_t checksum = 0;
    int finite_bad = 0;
};

struct SharedOutputHead {
    bool initialized = false;
    int slots = 0;
    int vocab = 0;
    int rows_per_gpu = 0;
    ContractRow output_rows[kGpus];
    float *d_hc = nullptr;
    float *d_hc_norm = nullptr;
    float *d_head_pre = nullptr;
    float *d_head_weights = nullptr;
    float *d_embd = nullptr;
    float *d_embd_norm = nullptr;
    float *d_head_fn = nullptr;
    float *d_head_base = nullptr;
    float *d_head_scale = nullptr;
    float *d_output_norm = nullptr;
    uint16_t *d_w[kGpus] = {};
    float *d_x[kGpus] = {};
    float *d_logits[kGpus] = {};
    uint32_t *d_best_token[kGpus] = {};
    float *d_best_logit[kGpus] = {};
    cudaEvent_t projection_start[kGpus] = {};
    cudaEvent_t projection_stop[kGpus] = {};
    cudaStream_t stream[kGpus] = {};
    cudaEvent_t prep_ready = {};
    cudaEvent_t broadcast_ready[kGpus] = {};
    cudaEvent_t top1_done[kGpus] = {};
    uint32_t *h_best_token[kGpus] = {};
    float *h_best_logit[kGpus] = {};
    uint64_t output_weight_bytes = 0;
    uint64_t logits_bytes = 0;
};

struct OutputHeadRunResult {
    bool pass = true;
    double total_ms = 0.0;
    double gather_ms = 0.0;
    double prep_ms = 0.0;
    double broadcast_ms = 0.0;
    double projection_ms = 0.0;
    double projection_kernel_worst_ms = 0.0;
    double top1_ms = 0.0;
    std::vector<uint32_t> tokens;
    std::vector<float> logits;
    uint64_t checksum = 0;
    int finite_bad = 0;
    int device_sync_count = 0;
    int stream_sync_count = 0;
    int event_sync_count = 0;
};

struct SharedTokenEmbedding {
    bool initialized = false;
    int slots = 0;
    int vocab = 0;
    int rows_per_gpu = 0;
    std::vector<uint16_t> h_w_full;
    uint16_t *d_slot_rows[kGpus] = {};
    uint64_t weight_bytes = 0;
};

int open_shared_token_embedding(const Options &opt,
                                const std::vector<ContractRow> &rows,
                                SharedTokenEmbedding *out) {
    out->slots = opt.slots;
    std::vector<ContractRow> emb_rows;
    int cols = 0;
    int vocab = 0;
    if (!select_bf16_dense_rows(rows, "token_embd.weight", &emb_rows, &cols, &vocab)) {
        std::fprintf(stderr, "shared token embedding failed to select token_embd.weight shards\n");
        return 1;
    }
    if (cols != kHidden || vocab <= 0 || vocab % kGpus != 0) {
        std::fprintf(stderr, "shared token embedding invalid shape cols=%d vocab=%d\n",
                     cols, vocab);
        return 2;
    }
    out->vocab = vocab;
    out->rows_per_gpu = vocab / kGpus;
    const uint64_t shard_elems = (uint64_t)out->rows_per_gpu * (uint64_t)kHidden;
    const uint64_t shard_bytes = shard_elems * sizeof(uint16_t);
    const uint64_t full_elems = shard_elems * kGpus;

    out->h_w_full.assign((size_t)full_elems, 0);
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
        CHECK_CUDA(cudaMalloc(&out->d_slot_rows[(size_t)rank],
                              (size_t)opt.slots * kHidden * sizeof(uint16_t)));
    }

    std::vector<uint16_t> host((size_t)shard_elems);
    for (int shard = 0; shard < kGpus; ++shard) {
        const ContractRow &r = emb_rows[(size_t)shard];
        const int shard_index = r.shard_index >= 0 ? r.shard_index : shard;
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), host.data(),
                          (size_t)shard_bytes) != 0) {
            return 3;
        }
        std::memcpy(out->h_w_full.data() + (uint64_t)shard_index * shard_elems,
                    host.data(), (size_t)shard_bytes);
        out->weight_bytes += shard_bytes;
    }
    out->initialized = true;
    return 0;
}

void close_shared_token_embedding(const Options &opt, SharedTokenEmbedding *out) {
    if (!out || !out->initialized) return;
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
        if (out->d_slot_rows[(size_t)rank]) {
            CHECK_CUDA(cudaFree(out->d_slot_rows[(size_t)rank]));
        }
    }
    *out = SharedTokenEmbedding{};
}

int seed_rank_hc_from_input_tokens(const Options &opt,
                                   SharedTokenEmbedding *embedding,
                                   RankState ranks[kGpus],
                                   const std::vector<uint32_t> &tokens) {
    if (!embedding || !embedding->initialized ||
        (int)tokens.size() < opt.slots ||
        embedding->h_w_full.empty()) {
        return 1;
    }
    const uint64_t shard_elems =
        (uint64_t)opt.slots * 4ull * (uint64_t)(kHidden / kGpus);
    std::vector<uint16_t> slot_rows((size_t)opt.slots * kHidden);
    for (int slot = 0; slot < opt.slots; ++slot) {
        uint32_t token = tokens[(size_t)slot];
        if (token >= (uint32_t)embedding->vocab) token = 0;
        std::memcpy(slot_rows.data() + (size_t)slot * kHidden,
                    embedding->h_w_full.data() + (uint64_t)token * kHidden,
                    (size_t)kHidden * sizeof(uint16_t));
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.d_final_hc_shard || !embedding->d_slot_rows[(size_t)rank]) return 2;
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaMemcpyAsync(embedding->d_slot_rows[(size_t)rank],
                                   slot_rows.data(),
                                   slot_rows.size() * sizeof(uint16_t),
                                   cudaMemcpyHostToDevice, r.stream));
        seed_hc_shard_from_token_embedding_kernel<<<
            (unsigned int)((shard_elems + 255) / 256), 256, 0, r.stream>>>(
            r.d_final_hc_shard,
            embedding->d_slot_rows[(size_t)rank],
            (uint32_t)opt.slots,
            rank);
        CHECK_CUDA(cudaGetLastError());
        r.hc_initialized = true;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }
    return 0;
}

int run_output_head_gate(const Options &opt,
                         const std::vector<ContractRow> &rows,
                         OutputHeadGateStats *stats) {
    stats->slots = opt.slots;

    std::vector<ContractRow> output_rows;
    int output_cols = 0;
    int vocab = 0;
    if (!select_bf16_dense_rows(rows, "output.weight", &output_rows, &output_cols, &vocab)) {
        std::fprintf(stderr, "output-head gate failed to select output.weight shards\n");
        return 1;
    }
    if (output_cols != kHidden || vocab <= 0 || vocab % kGpus != 0) {
        std::fprintf(stderr, "output-head gate invalid output.weight shape cols=%d vocab=%d\n",
                     output_cols, vocab);
        return 2;
    }
    const int rows_per_gpu = vocab / kGpus;
    stats->vocab = vocab;
    stats->rows_per_gpu = rows_per_gpu;

    std::vector<float> hc_head_fn;
    std::vector<float> hc_head_base;
    std::vector<float> hc_head_scale;
    std::vector<float> output_norm;
    if (load_control_f32(opt, rows, "hc_head_fn", (size_t)4 * 4 * kHidden, &hc_head_fn) ||
        load_control_f32(opt, rows, "hc_head_base", 4, &hc_head_base) ||
        load_control_f32(opt, rows, "hc_head_scale", 1, &hc_head_scale) ||
        load_control_f32(opt, rows, "output_norm.weight", kHidden, &output_norm)) {
        return 3;
    }

    const uint64_t hc_elems = (uint64_t)opt.slots * 4ull * (uint64_t)kHidden;
    const uint64_t embd_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const uint64_t logits_elems = (uint64_t)opt.slots * (uint64_t)rows_per_gpu;
    const uint64_t output_shard_bytes = (uint64_t)rows_per_gpu * (uint64_t)kHidden *
                                        sizeof(uint16_t);
    const auto total_start = std::chrono::steady_clock::now();

    float *d_hc = nullptr;
    float *d_hc_norm = nullptr;
    float *d_head_pre = nullptr;
    float *d_head_weights = nullptr;
    float *d_embd = nullptr;
    float *d_embd_norm = nullptr;
    float *d_head_fn = nullptr;
    float *d_head_base = nullptr;
    float *d_head_scale = nullptr;
    float *d_output_norm = nullptr;
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaMalloc(&d_hc, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hc_norm, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_pre, (size_t)opt.slots * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_weights, (size_t)opt.slots * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_embd, (size_t)embd_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_embd_norm, (size_t)embd_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_fn, hc_head_fn.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_base, hc_head_base.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_scale, hc_head_scale.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_norm, output_norm.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_head_fn, hc_head_fn.data(), hc_head_fn.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_head_base, hc_head_base.data(), hc_head_base.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_head_scale, hc_head_scale.data(), hc_head_scale.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_output_norm, output_norm.data(), output_norm.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    synthetic_hc_kernel<<<(unsigned int)((hc_elems + 255) / 256), 256>>>(d_hc, opt.slots);
    rms_norm_plain_rows_stable_kernel<<<(unsigned int)opt.slots, 256>>>(
        d_hc_norm, d_hc, 4u * (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
    const dim3 head_grid(4u, (unsigned int)opt.slots, 1u);
    f32_dense_kernel<<<head_grid, 256>>>(d_head_pre, d_head_fn, d_hc_norm,
                                         4u, 4u * (uint32_t)kHidden,
                                         (uint32_t)opt.slots);
    output_hc_weights_rows_kernel<<<(unsigned int)(((uint64_t)opt.slots * 4ull + 255) / 256), 256>>>(
        d_head_weights, d_head_pre, d_head_scale, d_head_base, (uint32_t)opt.slots);
    hc_weighted_sum_rows_kernel<<<(unsigned int)((embd_elems + 255) / 256), 256>>>(
        d_embd, d_hc, d_head_weights, (uint32_t)opt.slots);
    rms_norm_weight_rows_stable_kernel<<<(unsigned int)opt.slots, 256>>>(
        d_embd_norm, d_embd, d_output_norm, (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_embd_norm((size_t)embd_elems);
    CHECK_CUDA(cudaMemcpy(h_embd_norm.data(), d_embd_norm,
                          h_embd_norm.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<std::vector<float>> host_logits((size_t)kGpus);
    std::vector<uint16_t> host_w;
    std::vector<uint32_t> best_token((size_t)opt.slots, UINT32_MAX);
    std::vector<float> best_logit((size_t)opt.slots, -std::numeric_limits<float>::max());

    const auto projection_start = std::chrono::steady_clock::now();
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = output_rows[(size_t)gpu];
        host_w.resize((size_t)rows_per_gpu * (size_t)kHidden);
        host_logits[(size_t)gpu].resize((size_t)logits_elems);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), host_w.data(),
                          (size_t)output_shard_bytes) != 0) {
            stats->pass = false;
            return 4;
        }
        stats->output_weight_bytes += output_shard_bytes;
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        uint16_t *d_w = nullptr;
        __half *d_w_half = nullptr;
        float *d_x = nullptr;
        __half *d_x_half = nullptr;
        float *d_logits = nullptr;
        cublasHandle_t blas = nullptr;
        cudaEvent_t kernel_start = nullptr;
        cudaEvent_t kernel_stop = nullptr;
        CHECK_CUDA(cudaMalloc(&d_w, (size_t)output_shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x, h_embd_norm.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_logits, (size_t)logits_elems * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_w, host_w.data(), (size_t)output_shard_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, h_embd_norm.data(), h_embd_norm.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (opt.dense_f16_cublas_compose) {
            const uint64_t w_elems = (uint64_t)rows_per_gpu * (uint64_t)kHidden;
            const uint64_t x_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
            CHECK_CUDA(cudaMalloc(&d_w_half, (size_t)w_elems * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&d_x_half, (size_t)x_elems * sizeof(__half)));
            bf16_to_half_kernel<<<(unsigned int)((w_elems + 255) / 256), 256>>>(
                d_w_half, d_w, w_elems);
            cast_f32_to_half_kernel<<<(unsigned int)((x_elems + 255) / 256), 256>>>(
                d_x_half, d_x, x_elems);
            CHECK_CUDA(cudaGetLastError());
            cublasStatus_t st = cublasCreate(&blas);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "output-head cublasCreate failed gpu=%d status=%d\n",
                             gpu, (int)st);
                return 5;
            }
            (void)cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH);
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CHECK_CUDA(cudaEventCreate(&kernel_start));
            CHECK_CUDA(cudaEventCreate(&kernel_stop));
            CHECK_CUDA(cudaEventRecord(kernel_start));
            st = cublasGemmEx(blas,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              rows_per_gpu,
                              opt.slots,
                              kHidden,
                              &alpha,
                              d_w_half,
                              CUDA_R_16F,
                              kHidden,
                              d_x_half,
                              CUDA_R_16F,
                              kHidden,
                              &beta,
                              d_logits,
                              CUDA_R_32F,
                              rows_per_gpu,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "output-head cublasGemmEx failed gpu=%d status=%d\n",
                             gpu, (int)st);
                return 6;
            }
            CHECK_CUDA(cudaEventRecord(kernel_stop));
        } else {
            const dim3 grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1u);
            CHECK_CUDA(cudaEventCreate(&kernel_start));
            CHECK_CUDA(cudaEventCreate(&kernel_stop));
            CHECK_CUDA(cudaEventRecord(kernel_start));
            bf16_dense_kernel<<<grid, 256>>>(d_logits, d_w, d_x,
                                             (uint32_t)rows_per_gpu,
                                             (uint32_t)kHidden,
                                             (uint32_t)kHidden,
                                             (uint32_t)opt.slots);
            CHECK_CUDA(cudaEventRecord(kernel_stop));
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        float kernel_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
        stats->projection_kernel_worst_ms =
            std::max(stats->projection_kernel_worst_ms, (double)kernel_ms);
        CHECK_CUDA(cudaMemcpy(host_logits[(size_t)gpu].data(), d_logits,
                              (size_t)logits_elems * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventDestroy(kernel_stop));
        CHECK_CUDA(cudaEventDestroy(kernel_start));
        if (blas) (void)cublasDestroy(blas);
        CHECK_CUDA(cudaFree(d_logits));
        if (d_x_half) CHECK_CUDA(cudaFree(d_x_half));
        CHECK_CUDA(cudaFree(d_x));
        if (d_w_half) CHECK_CUDA(cudaFree(d_w_half));
        CHECK_CUDA(cudaFree(d_w));
    }
    const auto projection_stop = std::chrono::steady_clock::now();
    stats->projection_ms =
        std::chrono::duration<double, std::milli>(projection_stop - projection_start).count();
    stats->logits_bytes = logits_elems * sizeof(float) * kGpus;

    const auto reduce_start = std::chrono::steady_clock::now();
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const int shard_index = output_rows[(size_t)gpu].shard_index >= 0
            ? output_rows[(size_t)gpu].shard_index
            : gpu;
        for (int slot = 0; slot < opt.slots; ++slot) {
            const float *row = host_logits[(size_t)gpu].data() +
                               (uint64_t)slot * (uint64_t)rows_per_gpu;
            for (int v = 0; v < rows_per_gpu; ++v) {
                const float logit = row[v];
                if (!std::isfinite(logit)) {
                    stats->finite_bad++;
                    stats->pass = false;
                    continue;
                }
                if (logit > best_logit[(size_t)slot]) {
                    best_logit[(size_t)slot] = logit;
                    best_token[(size_t)slot] =
                        (uint32_t)(shard_index * rows_per_gpu + v);
                }
            }
        }
    }
    const auto reduce_stop = std::chrono::steady_clock::now();
    stats->host_reduce_ms =
        std::chrono::duration<double, std::milli>(reduce_stop - reduce_start).count();

    for (int slot = 0; slot < opt.slots; ++slot) {
        if (best_token[(size_t)slot] >= (uint32_t)vocab ||
            !std::isfinite(best_logit[(size_t)slot])) {
            stats->pass = false;
        }
        uint32_t bits = 0;
        std::memcpy(&bits, &best_logit[(size_t)slot], sizeof(bits));
        stats->checksum ^= (uint64_t)best_token[(size_t)slot] * 1000003ull +
                           (uint64_t)bits + (uint64_t)(slot + 1) * 7907ull;
    }
    stats->first_token = best_token.empty() ? UINT32_MAX : best_token[0];
    stats->first_logit = best_logit.empty() ? 0.0f : best_logit[0];
    const auto total_stop = std::chrono::steady_clock::now();
    stats->total_ms =
        std::chrono::duration<double, std::milli>(total_stop - total_start).count();
    if (stats->checksum == 0 || stats->finite_bad != 0) stats->pass = false;

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaFree(d_output_norm));
    CHECK_CUDA(cudaFree(d_head_scale));
    CHECK_CUDA(cudaFree(d_head_base));
    CHECK_CUDA(cudaFree(d_head_fn));
    CHECK_CUDA(cudaFree(d_embd_norm));
    CHECK_CUDA(cudaFree(d_embd));
    CHECK_CUDA(cudaFree(d_head_weights));
    CHECK_CUDA(cudaFree(d_head_pre));
    CHECK_CUDA(cudaFree(d_hc_norm));
    CHECK_CUDA(cudaFree(d_hc));

    std::printf("tp_ep_output_head_gate\tslots\t%d\tvocab\t%d\trows_per_gpu\t%d\t"
                "projection_kernel\t%s\t"
                "output_weight_bytes\t%llu\tlogits_bytes\t%llu\t"
                "projection_ms\t%.6f\tprojection_kernel_worst_ms\t%.6f\t"
                "host_reduce_ms\t%.6f\ttotal_ms\t%.6f\t"
                "first_token\t%u\tfirst_logit\t%.9f\tfinite_bad\t%d\t"
                "checksum\t%llu\t%s\n",
                stats->slots, stats->vocab, stats->rows_per_gpu,
                opt.dense_f16_cublas_compose ? "bf16_to_fp16_cublas" : "bf16_scalar",
                (unsigned long long)stats->output_weight_bytes,
                (unsigned long long)stats->logits_bytes,
                stats->projection_ms, stats->projection_kernel_worst_ms,
                stats->host_reduce_ms, stats->total_ms,
                stats->first_token, stats->first_logit, stats->finite_bad,
                (unsigned long long)stats->checksum,
                stats->pass ? "PASS" : "FAIL");
    return stats->pass ? 0 : 5;
}

int run_output_head_resident_gate(const Options &opt,
                                  const std::vector<ContractRow> &rows,
                                  OutputHeadResidentGateStats *stats) {
    if (opt.dense_f16_cublas_compose) {
        std::fprintf(stderr, "resident output-head gate currently supports bf16_scalar only\n");
        return 2;
    }
    stats->slots = opt.slots;
    stats->warmup = opt.warmup;
    stats->iters = opt.iters;

    std::vector<ContractRow> output_rows;
    int output_cols = 0;
    int vocab = 0;
    if (!select_bf16_dense_rows(rows, "output.weight", &output_rows, &output_cols, &vocab)) {
        std::fprintf(stderr, "resident output-head gate failed to select output.weight shards\n");
        return 1;
    }
    if (output_cols != kHidden || vocab <= 0 || vocab % kGpus != 0) {
        std::fprintf(stderr, "resident output-head gate invalid output.weight shape cols=%d vocab=%d\n",
                     output_cols, vocab);
        return 2;
    }
    const int rows_per_gpu = vocab / kGpus;
    stats->vocab = vocab;
    stats->rows_per_gpu = rows_per_gpu;

    std::vector<float> hc_head_fn;
    std::vector<float> hc_head_base;
    std::vector<float> hc_head_scale;
    std::vector<float> output_norm;
    if (load_control_f32(opt, rows, "hc_head_fn", (size_t)4 * 4 * kHidden, &hc_head_fn) ||
        load_control_f32(opt, rows, "hc_head_base", 4, &hc_head_base) ||
        load_control_f32(opt, rows, "hc_head_scale", 1, &hc_head_scale) ||
        load_control_f32(opt, rows, "output_norm.weight", kHidden, &output_norm)) {
        return 3;
    }

    const uint64_t hc_elems = (uint64_t)opt.slots * 4ull * (uint64_t)kHidden;
    const uint64_t embd_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const uint64_t logits_elems = (uint64_t)opt.slots * (uint64_t)rows_per_gpu;
    const uint64_t output_shard_bytes = (uint64_t)rows_per_gpu * (uint64_t)kHidden *
                                        sizeof(uint16_t);

    float *d_hc = nullptr;
    float *d_hc_norm = nullptr;
    float *d_head_pre = nullptr;
    float *d_head_weights = nullptr;
    float *d_embd = nullptr;
    float *d_embd_norm = nullptr;
    float *d_head_fn = nullptr;
    float *d_head_base = nullptr;
    float *d_head_scale = nullptr;
    float *d_output_norm = nullptr;
    uint16_t *d_w[kGpus] = {};
    float *d_x[kGpus] = {};
    float *d_logits[kGpus] = {};
    uint32_t *d_best_token[kGpus] = {};
    float *d_best_logit[kGpus] = {};
    cudaEvent_t projection_start[kGpus] = {};
    cudaEvent_t projection_stop[kGpus] = {};
    ncclComm_t broadcast_nccl[kGpus] = {};

    const auto load_start = std::chrono::steady_clock::now();
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaMalloc(&d_hc, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hc_norm, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_pre, (size_t)opt.slots * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_weights, (size_t)opt.slots * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_embd, (size_t)embd_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_embd_norm, (size_t)embd_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_fn, hc_head_fn.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_base, hc_head_base.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_head_scale, hc_head_scale.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_norm, output_norm.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_head_fn, hc_head_fn.data(), hc_head_fn.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_head_base, hc_head_base.data(), hc_head_base.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_head_scale, hc_head_scale.data(), hc_head_scale.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_output_norm, output_norm.data(), output_norm.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::vector<uint16_t> host_w((size_t)rows_per_gpu * (size_t)kHidden);
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = output_rows[(size_t)gpu];
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), host_w.data(),
                          (size_t)output_shard_bytes) != 0) {
            return 4;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaMalloc(&d_w[gpu], (size_t)output_shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x[gpu], (size_t)embd_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_logits[gpu], (size_t)logits_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_best_token[gpu], (size_t)opt.slots * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&d_best_logit[gpu], (size_t)opt.slots * sizeof(float)));
        CHECK_CUDA(cudaEventCreate(&projection_start[gpu]));
        CHECK_CUDA(cudaEventCreate(&projection_stop[gpu]));
        CHECK_CUDA(cudaMemcpy(d_w[gpu], host_w.data(), (size_t)output_shard_bytes,
                              cudaMemcpyHostToDevice));
        stats->output_weight_bytes += output_shard_bytes;
    }
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_NCCL(ncclCommInitAll(broadcast_nccl, kGpus, opt.devices));
    const auto load_stop = std::chrono::steady_clock::now();
    stats->load_ms =
        std::chrono::duration<double, std::milli>(load_stop - load_start).count();
    stats->logits_bytes = logits_elems * sizeof(float) * kGpus;

    const int total_iters = opt.warmup + opt.iters;
    std::vector<std::vector<uint32_t>> host_best_token((size_t)kGpus);
    std::vector<std::vector<float>> host_best_logit((size_t)kGpus);
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        host_best_token[(size_t)gpu].resize((size_t)opt.slots);
        host_best_logit[(size_t)gpu].resize((size_t)opt.slots);
    }
    std::vector<uint32_t> best_token((size_t)opt.slots, UINT32_MAX);
    std::vector<float> best_logit((size_t)opt.slots, -std::numeric_limits<float>::max());

    for (int iter = 0; iter < total_iters; ++iter) {
        const bool measure = iter >= opt.warmup;
        const auto iter_start = std::chrono::steady_clock::now();

        const auto prep_start = std::chrono::steady_clock::now();
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        synthetic_hc_kernel<<<(unsigned int)((hc_elems + 255) / 256), 256>>>(d_hc, opt.slots);
        rms_norm_plain_rows_stable_kernel<<<(unsigned int)opt.slots, 256>>>(
            d_hc_norm, d_hc, 4u * (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
        const dim3 head_grid(4u, (unsigned int)opt.slots, 1u);
        f32_dense_kernel<<<head_grid, 256>>>(d_head_pre, d_head_fn, d_hc_norm,
                                             4u, 4u * (uint32_t)kHidden,
                                             (uint32_t)opt.slots);
        output_hc_weights_rows_kernel<<<(unsigned int)(((uint64_t)opt.slots * 4ull + 255) / 256), 256>>>(
            d_head_weights, d_head_pre, d_head_scale, d_head_base, (uint32_t)opt.slots);
        hc_weighted_sum_rows_kernel<<<(unsigned int)((embd_elems + 255) / 256), 256>>>(
            d_embd, d_hc, d_head_weights, (uint32_t)opt.slots);
        rms_norm_weight_rows_stable_kernel<<<(unsigned int)opt.slots, 256>>>(
            d_embd_norm, d_embd, d_output_norm, (uint32_t)kHidden,
            (uint32_t)opt.slots, 1.0e-6f);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        const auto prep_stop = std::chrono::steady_clock::now();

        const auto broadcast_start = std::chrono::steady_clock::now();
        CHECK_NCCL(ncclGroupStart());
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            const void *send = gpu == 0 ? (const void *)d_embd_norm
                                        : (const void *)d_x[gpu];
            CHECK_NCCL(ncclBroadcast(send, d_x[gpu],
                                     (size_t)embd_elems * sizeof(float),
                                     ncclChar, 0, broadcast_nccl[gpu],
                                     (cudaStream_t)0));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        const auto broadcast_stop = std::chrono::steady_clock::now();

        const auto projection_start_wall = std::chrono::steady_clock::now();
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            const dim3 grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1u);
            CHECK_CUDA(cudaEventRecord(projection_start[gpu]));
            bf16_dense_kernel<<<grid, 256>>>(d_logits[gpu], d_w[gpu], d_x[gpu],
                                             (uint32_t)rows_per_gpu,
                                             (uint32_t)kHidden,
                                             (uint32_t)kHidden,
                                             (uint32_t)opt.slots);
            CHECK_CUDA(cudaEventRecord(projection_stop[gpu]));
            CHECK_CUDA(cudaGetLastError());
        }
        double iter_kernel_worst_ms = 0.0;
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            CHECK_CUDA(cudaDeviceSynchronize());
            float kernel_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&kernel_ms,
                                            projection_start[gpu],
                                            projection_stop[gpu]));
            iter_kernel_worst_ms = std::max(iter_kernel_worst_ms, (double)kernel_ms);
        }
        const auto projection_stop_wall = std::chrono::steady_clock::now();

        const auto reduce_start = std::chrono::steady_clock::now();
        std::fill(best_token.begin(), best_token.end(), UINT32_MAX);
        std::fill(best_logit.begin(), best_logit.end(),
                  -std::numeric_limits<float>::max());
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            const int shard_index = output_rows[(size_t)gpu].shard_index >= 0
                ? output_rows[(size_t)gpu].shard_index
                : gpu;
            shard_top1_kernel<<<(unsigned int)opt.slots, 256>>>(
                d_best_token[gpu],
                d_best_logit[gpu],
                d_logits[gpu],
                (uint32_t)rows_per_gpu,
                (uint32_t)(shard_index * rows_per_gpu),
                (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(host_best_token[(size_t)gpu].data(), d_best_token[gpu],
                                  (size_t)opt.slots * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(host_best_logit[(size_t)gpu].data(), d_best_logit[gpu],
                                  (size_t)opt.slots * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            for (int slot = 0; slot < opt.slots; ++slot) {
                const float logit = host_best_logit[(size_t)gpu][(size_t)slot];
                if (!std::isfinite(logit)) {
                    if (measure) stats->finite_bad++;
                    stats->pass = false;
                    continue;
                }
                if (logit > best_logit[(size_t)slot]) {
                    best_logit[(size_t)slot] = logit;
                    best_token[(size_t)slot] =
                        host_best_token[(size_t)gpu][(size_t)slot];
                }
            }
        }
        const auto reduce_stop = std::chrono::steady_clock::now();
        const auto iter_stop = std::chrono::steady_clock::now();

        if (measure) {
            stats->avg_hc_prep_ms +=
                std::chrono::duration<double, std::milli>(prep_stop - prep_start).count();
            stats->avg_broadcast_ms +=
                std::chrono::duration<double, std::milli>(broadcast_stop - broadcast_start).count();
            stats->avg_projection_wall_ms +=
                std::chrono::duration<double, std::milli>(projection_stop_wall - projection_start_wall).count();
            stats->avg_projection_kernel_worst_ms += iter_kernel_worst_ms;
            stats->avg_readback_reduce_ms +=
                std::chrono::duration<double, std::milli>(reduce_stop - reduce_start).count();
            stats->avg_total_ms +=
                std::chrono::duration<double, std::milli>(iter_stop - iter_start).count();
            for (int slot = 0; slot < opt.slots; ++slot) {
                if (best_token[(size_t)slot] >= (uint32_t)vocab ||
                    !std::isfinite(best_logit[(size_t)slot])) {
                    stats->pass = false;
                }
                uint32_t bits = 0;
                std::memcpy(&bits, &best_logit[(size_t)slot], sizeof(bits));
                stats->checksum ^=
                    (uint64_t)best_token[(size_t)slot] * 1000003ull +
                    (uint64_t)bits +
                    (uint64_t)(slot + 1) * 7907ull +
                    (uint64_t)(iter + 1) * 104729ull;
            }
            stats->first_token = best_token.empty() ? UINT32_MAX : best_token[0];
            stats->first_logit = best_logit.empty() ? 0.0f : best_logit[0];
        }
    }

    if (opt.iters > 0) {
        stats->avg_hc_prep_ms /= (double)opt.iters;
        stats->avg_broadcast_ms /= (double)opt.iters;
        stats->avg_projection_wall_ms /= (double)opt.iters;
        stats->avg_projection_kernel_worst_ms /= (double)opt.iters;
        stats->avg_readback_reduce_ms /= (double)opt.iters;
        stats->avg_total_ms /= (double)opt.iters;
        stats->output_head_tok_s = stats->avg_total_ms > 0.0
            ? (double)opt.slots * 1000.0 / stats->avg_total_ms
            : 0.0;
    }
    if (stats->checksum == 0 || stats->finite_bad != 0) stats->pass = false;

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (broadcast_nccl[gpu]) CHECK_NCCL(ncclCommDestroy(broadcast_nccl[gpu]));
        if (projection_stop[gpu]) CHECK_CUDA(cudaEventDestroy(projection_stop[gpu]));
        if (projection_start[gpu]) CHECK_CUDA(cudaEventDestroy(projection_start[gpu]));
        if (d_best_logit[gpu]) CHECK_CUDA(cudaFree(d_best_logit[gpu]));
        if (d_best_token[gpu]) CHECK_CUDA(cudaFree(d_best_token[gpu]));
        if (d_logits[gpu]) CHECK_CUDA(cudaFree(d_logits[gpu]));
        if (d_x[gpu]) CHECK_CUDA(cudaFree(d_x[gpu]));
        if (d_w[gpu]) CHECK_CUDA(cudaFree(d_w[gpu]));
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaFree(d_output_norm));
    CHECK_CUDA(cudaFree(d_head_scale));
    CHECK_CUDA(cudaFree(d_head_base));
    CHECK_CUDA(cudaFree(d_head_fn));
    CHECK_CUDA(cudaFree(d_embd_norm));
    CHECK_CUDA(cudaFree(d_embd));
    CHECK_CUDA(cudaFree(d_head_weights));
    CHECK_CUDA(cudaFree(d_head_pre));
    CHECK_CUDA(cudaFree(d_hc_norm));
    CHECK_CUDA(cudaFree(d_hc));

    std::printf("tp_ep_output_head_resident_gate\tslots\t%d\tvocab\t%d\t"
                "rows_per_gpu\t%d\twarmup\t%d\titers\t%d\t"
                "projection_kernel\tbf16_scalar\t"
                "output_weight_bytes\t%llu\tlogits_bytes\t%llu\t"
                "load_ms\t%.6f\tavg_total_ms\t%.6f\t"
                "avg_hc_prep_ms\t%.6f\tavg_broadcast_ms\t%.6f\t"
                "avg_projection_wall_ms\t%.6f\t"
                "avg_projection_kernel_worst_ms\t%.6f\t"
                "avg_device_top1_readback_ms\t%.6f\t"
                "output_head_tok_s\t%.6f\t"
                "first_token\t%u\tfirst_logit\t%.9f\tfinite_bad\t%d\t"
                "checksum\t%llu\t%s\n",
                stats->slots, stats->vocab, stats->rows_per_gpu,
                stats->warmup, stats->iters,
                (unsigned long long)stats->output_weight_bytes,
                (unsigned long long)stats->logits_bytes,
                stats->load_ms, stats->avg_total_ms,
                stats->avg_hc_prep_ms, stats->avg_broadcast_ms,
                stats->avg_projection_wall_ms,
                stats->avg_projection_kernel_worst_ms,
                stats->avg_readback_reduce_ms,
                stats->output_head_tok_s,
                stats->first_token, stats->first_logit, stats->finite_bad,
                (unsigned long long)stats->checksum,
                stats->pass ? "PASS" : "FAIL");
    return stats->pass ? 0 : 5;
}

int open_shared_output_head(const Options &opt,
                            const std::vector<ContractRow> &rows,
                            SharedOutputHead *out) {
    out->slots = opt.slots;
    std::vector<ContractRow> output_rows;
    int output_cols = 0;
    int vocab = 0;
    if (!select_bf16_dense_rows(rows, "output.weight", &output_rows,
                                &output_cols, &vocab)) {
        std::fprintf(stderr, "shared output-head failed to select output.weight shards\n");
        return 1;
    }
    if (output_cols != kHidden || vocab <= 0 || vocab % kGpus != 0) {
        std::fprintf(stderr, "shared output-head invalid output.weight shape cols=%d vocab=%d\n",
                     output_cols, vocab);
        return 2;
    }
    out->vocab = vocab;
    out->rows_per_gpu = vocab / kGpus;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        out->output_rows[gpu] = output_rows[(size_t)gpu];
    }

    std::vector<float> hc_head_fn;
    std::vector<float> hc_head_base;
    std::vector<float> hc_head_scale;
    std::vector<float> output_norm;
    if (load_control_f32(opt, rows, "hc_head_fn", (size_t)4 * 4 * kHidden,
                         &hc_head_fn) ||
        load_control_f32(opt, rows, "hc_head_base", 4, &hc_head_base) ||
        load_control_f32(opt, rows, "hc_head_scale", 1, &hc_head_scale) ||
        load_control_f32(opt, rows, "output_norm.weight", kHidden, &output_norm)) {
        return 3;
    }

    const uint64_t hc_elems = (uint64_t)opt.slots * 4ull * (uint64_t)kHidden;
    const uint64_t embd_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const uint64_t logits_elems = (uint64_t)opt.slots * (uint64_t)out->rows_per_gpu;
    const uint64_t output_shard_bytes =
        (uint64_t)out->rows_per_gpu * (uint64_t)kHidden * sizeof(uint16_t);

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaMalloc(&out->d_hc, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_hc_norm, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_head_pre, (size_t)opt.slots * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_head_weights, (size_t)opt.slots * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_embd, (size_t)embd_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_embd_norm, (size_t)embd_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_head_fn, hc_head_fn.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_head_base, hc_head_base.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_head_scale, hc_head_scale.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_output_norm, output_norm.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(out->d_head_fn, hc_head_fn.data(),
                          hc_head_fn.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(out->d_head_base, hc_head_base.data(),
                          hc_head_base.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(out->d_head_scale, hc_head_scale.data(),
                          hc_head_scale.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(out->d_output_norm, output_norm.data(),
                          output_norm.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaStreamCreateWithFlags(&out->stream[0], cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreateWithFlags(&out->prep_ready, cudaEventDisableTiming));

    std::vector<uint16_t> host_w((size_t)out->rows_per_gpu * (size_t)kHidden);
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = out->output_rows[gpu];
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), host_w.data(),
                          (size_t)output_shard_bytes) != 0) {
            return 4;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaMalloc(&out->d_w[gpu], (size_t)output_shard_bytes));
        CHECK_CUDA(cudaMalloc(&out->d_x[gpu], (size_t)embd_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_logits[gpu],
                              (size_t)logits_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_best_token[gpu],
                              (size_t)opt.slots * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&out->d_best_logit[gpu],
                              (size_t)opt.slots * sizeof(float)));
        CHECK_CUDA(cudaEventCreate(&out->projection_start[gpu]));
        CHECK_CUDA(cudaEventCreate(&out->projection_stop[gpu]));
        if (gpu != 0) {
            CHECK_CUDA(cudaStreamCreateWithFlags(&out->stream[gpu],
                                                 cudaStreamNonBlocking));
        }
        CHECK_CUDA(cudaEventCreateWithFlags(&out->broadcast_ready[gpu],
                                            cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&out->top1_done[gpu],
                                            cudaEventDisableTiming));
        CHECK_CUDA(cudaMallocHost(&out->h_best_token[gpu],
                                  (size_t)opt.slots * sizeof(uint32_t)));
        CHECK_CUDA(cudaMallocHost(&out->h_best_logit[gpu],
                                  (size_t)opt.slots * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(out->d_w[gpu], host_w.data(),
                              (size_t)output_shard_bytes, cudaMemcpyHostToDevice));
        out->output_weight_bytes += output_shard_bytes;
    }
    out->logits_bytes = logits_elems * sizeof(float) * kGpus;
    out->initialized = true;
    return 0;
}

void close_shared_output_head(const Options &opt, SharedOutputHead *out) {
    if (!out || !out->initialized) return;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (out->h_best_logit[gpu]) CHECK_CUDA(cudaFreeHost(out->h_best_logit[gpu]));
        if (out->h_best_token[gpu]) CHECK_CUDA(cudaFreeHost(out->h_best_token[gpu]));
        if (out->top1_done[gpu]) CHECK_CUDA(cudaEventDestroy(out->top1_done[gpu]));
        if (out->broadcast_ready[gpu]) CHECK_CUDA(cudaEventDestroy(out->broadcast_ready[gpu]));
        if (out->projection_stop[gpu]) CHECK_CUDA(cudaEventDestroy(out->projection_stop[gpu]));
        if (out->projection_start[gpu]) CHECK_CUDA(cudaEventDestroy(out->projection_start[gpu]));
        if (out->stream[gpu]) CHECK_CUDA(cudaStreamDestroy(out->stream[gpu]));
        if (out->d_best_logit[gpu]) CHECK_CUDA(cudaFree(out->d_best_logit[gpu]));
        if (out->d_best_token[gpu]) CHECK_CUDA(cudaFree(out->d_best_token[gpu]));
        if (out->d_logits[gpu]) CHECK_CUDA(cudaFree(out->d_logits[gpu]));
        if (out->d_x[gpu]) CHECK_CUDA(cudaFree(out->d_x[gpu]));
        if (out->d_w[gpu]) CHECK_CUDA(cudaFree(out->d_w[gpu]));
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    if (out->prep_ready) CHECK_CUDA(cudaEventDestroy(out->prep_ready));
    if (out->d_output_norm) CHECK_CUDA(cudaFree(out->d_output_norm));
    if (out->d_head_scale) CHECK_CUDA(cudaFree(out->d_head_scale));
    if (out->d_head_base) CHECK_CUDA(cudaFree(out->d_head_base));
    if (out->d_head_fn) CHECK_CUDA(cudaFree(out->d_head_fn));
    if (out->d_embd_norm) CHECK_CUDA(cudaFree(out->d_embd_norm));
    if (out->d_embd) CHECK_CUDA(cudaFree(out->d_embd));
    if (out->d_head_weights) CHECK_CUDA(cudaFree(out->d_head_weights));
    if (out->d_head_pre) CHECK_CUDA(cudaFree(out->d_head_pre));
    if (out->d_hc_norm) CHECK_CUDA(cudaFree(out->d_hc_norm));
    if (out->d_hc) CHECK_CUDA(cudaFree(out->d_hc));
    *out = SharedOutputHead{};
}

int run_shared_output_head_from_rank_hc(const Options &opt,
                                        SharedOutputHead *head,
                                        RankState ranks[kGpus],
                                        OutputHeadRunResult *result) {
    if (!head || !head->initialized || head->slots != opt.slots) return 1;
    const auto total_start = std::chrono::steady_clock::now();
    const uint64_t hc_shard_elems =
        (uint64_t)opt.slots * 4ull * (uint64_t)(kHidden / kGpus);
    const uint64_t hc_elems = (uint64_t)opt.slots * 4ull * (uint64_t)kHidden;
    const uint64_t embd_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const uint64_t logits_elems =
        (uint64_t)opt.slots * (uint64_t)head->rows_per_gpu;

    if (opt.decode_cudagraph_gate) {
        if (opt.decode_cudagraph_output_sync_gate) {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaDeviceSynchronize());
                result->device_sync_count++;
            }
        } else {
            const int wait_rc =
                enqueue_control_wait_after_rank_streams(opt, ranks, (cudaStream_t)0);
            if (wait_rc != 0) return wait_rc;
            const int dense_wait_rc =
                enqueue_control_wait_after_dense_streams(opt, ranks, (cudaStream_t)0);
            if (dense_wait_rc != 0) return dense_wait_rc;
        }
    }

    const auto gather_start = std::chrono::steady_clock::now();
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        if (!ranks[rank].d_final_hc_shard) {
            std::fprintf(stderr, "diagnostic output-head missing final HC shard rank=%d\n",
                         rank);
            return 2;
        }
        gather_hc_shard_to_full_kernel<<<(unsigned int)((hc_shard_elems + 255) / 256), 256>>>(
            head->d_hc, ranks[rank].d_final_hc_shard, rank, (uint32_t)opt.slots);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    result->device_sync_count++;
    const auto gather_stop = std::chrono::steady_clock::now();

    const auto prep_start = std::chrono::steady_clock::now();
    rms_norm_plain_rows_stable_kernel<<<(unsigned int)opt.slots, 256>>>(
        head->d_hc_norm, head->d_hc, 4u * (uint32_t)kHidden,
        (uint32_t)opt.slots, 1.0e-6f);
    const dim3 head_grid(4u, (unsigned int)opt.slots, 1u);
    f32_dense_kernel<<<head_grid, 256>>>(head->d_head_pre, head->d_head_fn,
                                         head->d_hc_norm, 4u,
                                         4u * (uint32_t)kHidden,
                                         (uint32_t)opt.slots);
    output_hc_weights_rows_kernel<<<(unsigned int)(((uint64_t)opt.slots * 4ull + 255) / 256), 256>>>(
        head->d_head_weights, head->d_head_pre, head->d_head_scale,
        head->d_head_base, (uint32_t)opt.slots);
    hc_weighted_sum_rows_kernel<<<(unsigned int)((embd_elems + 255) / 256), 256>>>(
        head->d_embd, head->d_hc, head->d_head_weights, (uint32_t)opt.slots);
    rms_norm_weight_rows_stable_kernel<<<(unsigned int)opt.slots, 256>>>(
        head->d_embd_norm, head->d_embd, head->d_output_norm,
        (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    result->device_sync_count++;
    const auto prep_stop = std::chrono::steady_clock::now();

    const auto broadcast_start = std::chrono::steady_clock::now();
    void *x_dsts[kGpus] = {};
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        x_dsts[gpu] = head->d_x[gpu];
    }
    if (nccl_broadcast_bytes_from_rank0(
            ranks, head->d_embd_norm, x_dsts,
            (size_t)embd_elems * sizeof(float),
            "shared_output_head_x") != 0) {
        return 6;
    }
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(ranks[gpu].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[gpu].stream));
        result->device_sync_count++;
    }
    const auto broadcast_stop = std::chrono::steady_clock::now();

    const auto projection_start_wall = std::chrono::steady_clock::now();
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        const dim3 grid((unsigned int)head->rows_per_gpu, (unsigned int)opt.slots, 1u);
        CHECK_CUDA(cudaEventRecord(head->projection_start[gpu]));
        bf16_dense_kernel<<<grid, 256>>>(head->d_logits[gpu], head->d_w[gpu],
                                         head->d_x[gpu],
                                         (uint32_t)head->rows_per_gpu,
                                         (uint32_t)kHidden,
                                         (uint32_t)kHidden,
                                         (uint32_t)opt.slots);
        CHECK_CUDA(cudaEventRecord(head->projection_stop[gpu]));
        CHECK_CUDA(cudaGetLastError());
    }
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaDeviceSynchronize());
        result->device_sync_count++;
        float kernel_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&kernel_ms,
                                        head->projection_start[gpu],
                                        head->projection_stop[gpu]));
        result->projection_kernel_worst_ms =
            std::max(result->projection_kernel_worst_ms, (double)kernel_ms);
    }
    const auto projection_stop_wall = std::chrono::steady_clock::now();

    const auto top1_start = std::chrono::steady_clock::now();
    std::vector<std::vector<uint32_t>> host_tokens((size_t)kGpus);
    std::vector<std::vector<float>> host_logits((size_t)kGpus);
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        host_tokens[(size_t)gpu].resize((size_t)opt.slots);
        host_logits[(size_t)gpu].resize((size_t)opt.slots);
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        const int shard_index = head->output_rows[gpu].shard_index >= 0
            ? head->output_rows[gpu].shard_index
            : gpu;
        shard_top1_kernel<<<(unsigned int)opt.slots, 256>>>(
            head->d_best_token[gpu], head->d_best_logit[gpu],
            head->d_logits[gpu], (uint32_t)head->rows_per_gpu,
            (uint32_t)(shard_index * head->rows_per_gpu), (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }

    result->tokens.assign((size_t)opt.slots, UINT32_MAX);
    result->logits.assign((size_t)opt.slots, -std::numeric_limits<float>::max());
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaDeviceSynchronize());
        result->device_sync_count++;
        CHECK_CUDA(cudaMemcpy(host_tokens[(size_t)gpu].data(), head->d_best_token[gpu],
                              (size_t)opt.slots * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_logits[(size_t)gpu].data(), head->d_best_logit[gpu],
                              (size_t)opt.slots * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int slot = 0; slot < opt.slots; ++slot) {
            const float logit = host_logits[(size_t)gpu][(size_t)slot];
            if (!std::isfinite(logit)) {
                result->finite_bad++;
                result->pass = false;
                continue;
            }
            if (logit > result->logits[(size_t)slot]) {
                result->logits[(size_t)slot] = logit;
                result->tokens[(size_t)slot] = host_tokens[(size_t)gpu][(size_t)slot];
            }
        }
    }
    const auto top1_stop = std::chrono::steady_clock::now();
    const auto total_stop = std::chrono::steady_clock::now();

    for (int slot = 0; slot < opt.slots; ++slot) {
        if (result->tokens[(size_t)slot] >= (uint32_t)head->vocab ||
            !std::isfinite(result->logits[(size_t)slot])) {
            result->pass = false;
        }
        uint32_t bits = 0;
        std::memcpy(&bits, &result->logits[(size_t)slot], sizeof(bits));
        result->checksum ^= (uint64_t)result->tokens[(size_t)slot] * 1000003ull +
                            (uint64_t)bits + (uint64_t)(slot + 1) * 7907ull;
    }
    if (result->checksum == 0 || result->finite_bad != 0) result->pass = false;

    result->gather_ms =
        std::chrono::duration<double, std::milli>(gather_stop - gather_start).count();
    result->prep_ms =
        std::chrono::duration<double, std::milli>(prep_stop - prep_start).count();
    result->broadcast_ms =
        std::chrono::duration<double, std::milli>(broadcast_stop - broadcast_start).count();
    result->projection_ms =
        std::chrono::duration<double, std::milli>(projection_stop_wall - projection_start_wall).count();
    result->top1_ms =
        std::chrono::duration<double, std::milli>(top1_stop - top1_start).count();
    result->total_ms =
        std::chrono::duration<double, std::milli>(total_stop - total_start).count();
    (void)hc_elems;
    (void)logits_elems;
    return result->pass ? 0 : 5;
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
    uint64_t planned_bytes[kGpus] = {};
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        cache->gpu_cache_aligned_bytes[gpu] = gpu_offsets[gpu];
        cache->gpu_temp_bytes[gpu] = gpu_temp[gpu];
        planned_bytes[gpu] = gpu_offsets[gpu] + gpu_temp[gpu];
    }
    if (check_planned_vram_allocation(opt, "dense_f16_cache_prealloc", planned_bytes) != 0) {
        std::fprintf(stderr,
                     "dense_f16_cache_vram_admission_failed min_free_mib=%llu\n",
                     (unsigned long long)opt.vram_min_free_mib);
        return 3;
    }
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
                              ResidentF8Dense *op,
                              int expected_rows_per_gpu = kHidden / kGpus,
                              bool keep_packed_f8 = false,
                              bool keep_float_input = false) {
    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    bool source_is_f8 = select_dense_rows(rows, tensor, &selected, &cols, &total_rows);
    bool source_is_bf16 = false;
    if (!source_is_f8) {
        source_is_bf16 = select_bf16_dense_rows(rows, tensor, &selected, &cols, &total_rows);
    }
    if (!source_is_f8 && !source_is_bf16) {
        std::fprintf(stderr, "resident dense tensor validation failed for %s\n", tensor);
        return 1;
    }
    if (source_is_bf16 && keep_packed_f8) {
        std::fprintf(stderr, "resident dense tensor %s requested packed f8 retention for bf16 source\n",
                     tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    if (rows_per_gpu != expected_rows_per_gpu) {
        std::fprintf(stderr, "resident dense tensor %s rows_per_gpu=%d expected=%d\n",
                     tensor, rows_per_gpu, expected_rows_per_gpu);
        return 2;
    }
    const uint64_t row_bytes =
        source_is_f8 ? f8_row_bytes(cols) : (uint64_t)cols * sizeof(uint16_t);
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
        if (source_is_bf16 && !cache_entry) {
            std::fprintf(stderr,
                         "resident bf16 dense tensor %s requires dense f16 cache on gpu %d\n",
                         tensor, gpu);
            free_resident_f8_dense(*op, opt);
            return 3;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (!cache_entry || keep_packed_f8) {
            std::vector<uint8_t> h_w((size_t)shard_bytes);
            const std::string path = path_join(opt.pack_dir, r.source_pack_file);
            if (read_exact_at(path, physical_row_offset(r), h_w.data(), h_w.size()) != 0) {
                free_resident_f8_dense(*op, opt);
                return 3;
            }
            CHECK_CUDA(cudaMalloc(&op->d_w[(size_t)gpu], (size_t)shard_bytes));
            CHECK_CUDA(cudaMemcpy(op->d_w[(size_t)gpu], h_w.data(), (size_t)shard_bytes,
                                  cudaMemcpyHostToDevice));
        }
        op->loaded_bytes += shard_bytes;
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
                if (!source_is_f8) {
                    free_resident_f8_dense(*op, opt);
                    return 5;
                }
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
            if (!keep_float_input) {
                CHECK_CUDA(cudaFree(op->d_x[(size_t)gpu]));
                op->d_x[(size_t)gpu] = nullptr;
            }
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

void free_shared_dense_ops(SharedDenseOps *ops, const Options &opt) {
    if (!ops) return;
    for (int layer = 0; layer < 43; ++layer) {
        free_resident_f8_dense(ops->layers[layer].attn_q_a, opt);
        free_resident_f8_dense(ops->layers[layer].attn_q_b, opt);
        free_resident_f8_dense(ops->layers[layer].attn_kv_latent, opt);
        free_resident_f8_dense(ops->layers[layer].attn_output_a, opt);
        free_resident_f8_dense(ops->layers[layer].attn_compress_kv, opt);
        free_resident_f8_dense(ops->layers[layer].attn_compress_gate, opt);
        free_resident_f8_dense(ops->layers[layer].indexer_attn_q_b, opt);
        free_resident_f8_dense(ops->layers[layer].indexer_proj, opt);
        free_resident_f8_dense(ops->layers[layer].indexer_compress_kv, opt);
        free_resident_f8_dense(ops->layers[layer].indexer_compress_gate, opt);
        free_resident_f8_dense(ops->layers[layer].attn, opt);
        free_resident_f8_dense(ops->layers[layer].shared, opt);
        free_resident_f8_dense(ops->layers[layer].shared_gate, opt);
        free_resident_f8_dense(ops->layers[layer].shared_up, opt);
        ops->layers[layer] = LayerDenseOps{};
    }
    *ops = SharedDenseOps{};
}

int open_shared_dense_ops(const Options &opt,
                          const DenseF16Cache *cache,
                          SharedDenseOps *ops) {
    if (!opt.dense_f16_cublas_compose || !opt.dense_f16_cache_compose || !cache) {
        return 1;
    }
    for (int layer = 0; layer < 43; ++layer) {
        std::vector<ContractRow> rows;
        LayerStats stats;
        if (parse_contract(opt.contract_path, layer, &rows, &stats) != 0 ||
            stats.bad_rows != 0) {
            free_shared_dense_ops(ops, opt);
            return 2;
        }
        Options layer_opt = opt;
        layer_opt.layer = layer;
        LayerDenseOps &d = ops->layers[layer];
        const std::string attn_q_a_tensor = layer_tensor_name(layer, "attn_q_a.weight");
        const std::string attn_q_b_tensor = layer_tensor_name(layer, "attn_q_b.weight");
        const std::string attn_kv_tensor = layer_tensor_name(layer, "attn_kv_latent.weight");
        const std::string attn_output_a_tensor = layer_tensor_name(layer, "attn_output_a.weight");
        const std::string attn_compress_kv_tensor = layer_tensor_name(layer, "attn_compress_kv.weight");
        const std::string attn_compress_gate_tensor = layer_tensor_name(layer, "attn_compress_gate.weight");
        const std::string indexer_attn_q_b_tensor = layer_tensor_name(layer, "indexer.attn_q_b.weight");
        const std::string indexer_proj_tensor = layer_tensor_name(layer, "indexer.proj.weight");
        const std::string indexer_compress_kv_tensor = layer_tensor_name(layer, "indexer.compress_kv.weight");
        const std::string indexer_compress_gate_tensor = layer_tensor_name(layer, "indexer.compress_gate.weight");
        const std::string attn_tensor = layer_tensor_name(layer, "attn_output_b.weight");
        const std::string shared_tensor = layer_tensor_name(layer, "ffn_down_shexp.weight");
        const std::string shared_gate_tensor = layer_tensor_name(layer, "ffn_gate_shexp.weight");
        const std::string shared_up_tensor = layer_tensor_name(layer, "ffn_up_shexp.weight");
        if (opt.true_ds4_attention_residency_gate) {
            if (prepare_resident_f8_dense(layer_opt, rows, attn_q_a_tensor.c_str(), 11,
                                          cache, &d.attn_q_a, 1024 / kGpus) != 0 ||
                prepare_resident_f8_dense(layer_opt, rows, attn_q_b_tensor.c_str(), 12,
                                          cache, &d.attn_q_b, 32768 / kGpus) != 0 ||
                prepare_resident_f8_dense(layer_opt, rows, attn_kv_tensor.c_str(), 13,
                                          cache, &d.attn_kv_latent, kHeadDim / kGpus) != 0 ||
                prepare_resident_f8_dense(layer_opt, rows, attn_output_a_tensor.c_str(), 14,
                                          cache, &d.attn_output_a, 8192 / kGpus) != 0) {
                free_shared_dense_ops(ops, opt);
                return 5;
            }
            ops->loaded_bytes += d.attn_q_a.loaded_bytes + d.attn_q_b.loaded_bytes +
                                 d.attn_kv_latent.loaded_bytes +
                                 d.attn_output_a.loaded_bytes;
        }
        if (opt.true_ds4_compressed_kv_gate) {
            const int ratio = ds4_layer_ratio(layer);
            if (ratio != 0) {
                const int comp_width = ratio == 4 ? 2 * kHeadDim : kHeadDim;
                if (prepare_resident_f8_dense(layer_opt, rows, attn_compress_kv_tensor.c_str(),
                                              15, cache, &d.attn_compress_kv,
                                              comp_width / kGpus) != 0 ||
                    prepare_resident_f8_dense(layer_opt, rows, attn_compress_gate_tensor.c_str(),
                                              16, cache, &d.attn_compress_gate,
                                              comp_width / kGpus) != 0) {
                    free_shared_dense_ops(ops, opt);
                    return 6;
                }
                ops->loaded_bytes += d.attn_compress_kv.loaded_bytes +
                                     d.attn_compress_gate.loaded_bytes;
            }
            if (opt.true_ds4_indexer_attention_gate && ratio == 4) {
                if (prepare_resident_f8_dense(layer_opt, rows, indexer_attn_q_b_tensor.c_str(),
                                              17, cache, &d.indexer_attn_q_b,
                                              (kIndexerHead * kIndexerHeadDim) / kGpus) != 0 ||
                    prepare_resident_f8_dense(layer_opt, rows, indexer_proj_tensor.c_str(),
                                              18, cache, &d.indexer_proj,
                                              kIndexerHead / kGpus) != 0 ||
                    prepare_resident_f8_dense(layer_opt, rows, indexer_compress_kv_tensor.c_str(),
                                              19, cache, &d.indexer_compress_kv,
                                              (2 * kIndexerHeadDim) / kGpus) != 0 ||
                    prepare_resident_f8_dense(layer_opt, rows, indexer_compress_gate_tensor.c_str(),
                                              20, cache, &d.indexer_compress_gate,
                                              (2 * kIndexerHeadDim) / kGpus) != 0) {
                    free_shared_dense_ops(ops, opt);
                    return 7;
                }
                ops->loaded_bytes += d.indexer_attn_q_b.loaded_bytes +
                                     d.indexer_proj.loaded_bytes +
                                     d.indexer_compress_kv.loaded_bytes +
                                     d.indexer_compress_gate.loaded_bytes;
            }
        }
        if (prepare_resident_f8_dense(layer_opt, rows, attn_tensor.c_str(), 1, cache,
                                      &d.attn) != 0 ||
            prepare_resident_f8_dense(layer_opt, rows, shared_tensor.c_str(), 2, cache,
                                      &d.shared, kHidden / kGpus,
                                      opt.true_shared_ffn_gate,
                                      opt.true_shared_ffn_gate) != 0) {
            free_shared_dense_ops(ops, opt);
            return 3;
        }
        if (opt.true_shared_ffn_gate) {
            if (prepare_resident_f8_dense(layer_opt, rows, shared_gate_tensor.c_str(), 3,
                                          cache, &d.shared_gate, kMid / kGpus) != 0 ||
                prepare_resident_f8_dense(layer_opt, rows, shared_up_tensor.c_str(), 4,
                                          cache, &d.shared_up, kMid / kGpus) != 0) {
                free_shared_dense_ops(ops, opt);
                return 4;
            }
            ops->loaded_bytes += d.shared_gate.loaded_bytes + d.shared_up.loaded_bytes;
        }
        d.initialized = true;
        ops->loaded_bytes += d.attn.loaded_bytes + d.shared.loaded_bytes;
    }
    ops->initialized = true;
    return 0;
}

int launch_resident_f8_dense(const Options &opt,
                             const ResidentF8Dense &op,
                             RankState ranks[kGpus]) {
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        (void)cudaGetLastError();
        cudaStream_t stream = ranks[gpu].dense_stream ? ranks[gpu].dense_stream
                                                       : ranks[gpu].stream;
        if (opt.dense_f16_cublas_compose) {
            if (!op.cublas[(size_t)gpu] ||
                !op.d_w_half[(size_t)gpu] ||
                !op.d_x_half[(size_t)gpu]) {
                return 1;
            }
            cublasStatus_t st = cublasSetStream(op.cublas[(size_t)gpu], stream);
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
            f8_b128_dense_hmma_m16_kernel<<<grid, 128, 0, stream>>>(
                op.d_out[(size_t)gpu], op.d_w[(size_t)gpu], op.d_x[(size_t)gpu],
                op.rows_per_gpu, op.cols, (uint32_t)op.row_bytes, op.slots);
        } else {
            const dim3 grid((unsigned int)op.rows_per_gpu, (unsigned int)op.slots, 1);
            f8_b128_dense_kernel<<<grid, 256, 0, stream>>>(
                op.d_out[(size_t)gpu], op.d_w[(size_t)gpu], op.d_x[(size_t)gpu],
                op.rows_per_gpu, op.cols, (uint32_t)op.row_bytes, op.slots);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    return 0;
}

int launch_resident_f8_dense_f32_input(const Options &opt,
                                       const ResidentF8Dense &op,
                                       RankState ranks[kGpus]) {
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (!op.d_w[(size_t)gpu] || !op.d_x[(size_t)gpu]) return 1;
        cudaStream_t stream = ranks[gpu].dense_stream ? ranks[gpu].dense_stream
                                                       : ranks[gpu].stream;
        const dim3 grid((unsigned int)op.rows_per_gpu, (unsigned int)op.slots, 1);
        f8_b128_dense_kernel<<<grid, 256, 0, stream>>>(
            op.d_out[(size_t)gpu], op.d_w[(size_t)gpu], op.d_x[(size_t)gpu],
            op.rows_per_gpu, op.cols, (uint32_t)op.row_bytes, op.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    return 0;
}

int enqueue_dense_wait_after_rank_stream(RankState ranks[kGpus]) {
    const int slot = next_graph_order_event_slot(ranks);
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        RankState &r = ranks[gpu];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.dense_stream || r.dense_stream == r.stream) continue;
        cudaEvent_t ev = graph_stream_done_event(r, slot);
        if (!ev) return 1;
        CHECK_CUDA(cudaEventRecord(ev, r.stream));
        CHECK_CUDA(cudaStreamWaitEvent(r.dense_stream, ev, 0));
    }
    return 0;
}

int enqueue_rank_streams_wait_after_dense_streams(RankState ranks[kGpus]) {
    const int slot = next_graph_order_event_slot(ranks);
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        RankState &r = ranks[gpu];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.dense_stream || r.dense_stream == r.stream) continue;
        cudaEvent_t ev = graph_dense_done_event(r, slot);
        if (!ev) return 1;
        CHECK_CUDA(cudaEventRecord(ev, r.dense_stream));
        CHECK_CUDA(cudaStreamWaitEvent(r.stream, ev, 0));
    }
    return 0;
}

int enqueue_cross_gpu_stream_barrier(RankState ranks[kGpus],
                                     bool include_copy_streams) {
    const int slot = next_graph_order_event_slot(ranks);
    for (int src = 0; src < kGpus; ++src) {
        RankState &r = ranks[src];
        CHECK_CUDA(cudaSetDevice(r.device));
        cudaEvent_t stream_ev = graph_stream_done_event(r, slot);
        cudaEvent_t dense_ev = graph_dense_done_event(r, slot);
        if (!stream_ev || !dense_ev) return 1;
        CHECK_CUDA(cudaEventRecord(stream_ev, r.stream));
        CHECK_CUDA(cudaEventRecord(dense_ev,
                                   r.dense_stream ? r.dense_stream : r.stream));
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        RankState &r = ranks[dst];
        CHECK_CUDA(cudaSetDevice(r.device));
        for (int src = 0; src < kGpus; ++src) {
            CHECK_CUDA(cudaStreamWaitEvent(r.stream,
                                           graph_stream_done_event(ranks[src],
                                                                   slot),
                                           0));
            CHECK_CUDA(cudaStreamWaitEvent(r.stream,
                                           graph_dense_done_event(ranks[src],
                                                                  slot),
                                           0));
            if (r.dense_stream) {
                CHECK_CUDA(cudaStreamWaitEvent(r.dense_stream,
                                               graph_stream_done_event(ranks[src],
                                                                       slot),
                                               0));
                CHECK_CUDA(cudaStreamWaitEvent(r.dense_stream,
                                               graph_dense_done_event(ranks[src],
                                                                      slot),
                                               0));
            }
            if (include_copy_streams) {
                for (int q = 0; q < kGpus; ++q) {
                    cudaStream_t copy_stream = r.copy_streams[q]
                        ? r.copy_streams[q]
                        : r.copy_stream ? r.copy_stream : r.stream;
                    CHECK_CUDA(cudaStreamWaitEvent(copy_stream,
                                                   graph_stream_done_event(
                                                       ranks[src], slot),
                                                   0));
                    CHECK_CUDA(cudaStreamWaitEvent(copy_stream,
                                                   graph_dense_done_event(
                                                       ranks[src], slot),
                                                   0));
                }
            }
        }
    }
    return 0;
}

int enqueue_control_wait_after_rank_streams(const Options &opt,
                                            RankState ranks[kGpus],
                                            cudaStream_t control_stream) {
    const int slot = next_graph_order_event_slot(ranks);
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        cudaEvent_t ev = graph_stream_done_event(r, slot);
        if (!ev) return 1;
        CHECK_CUDA(cudaEventRecord(ev, r.stream));
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaStreamWaitEvent(control_stream,
                                       graph_stream_done_event(ranks[rank],
                                                               slot),
                                       0));
    }
    return 0;
}

int enqueue_control_wait_after_dense_streams(const Options &opt,
                                             RankState ranks[kGpus],
                                             cudaStream_t control_stream) {
    const int slot = next_graph_order_event_slot(ranks);
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        cudaEvent_t ev = graph_dense_done_event(r, slot);
        if (!ev) return 1;
        CHECK_CUDA(cudaEventRecord(ev,
                                   r.dense_stream ? r.dense_stream : r.stream));
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaStreamWaitEvent(control_stream,
                                       graph_dense_done_event(ranks[rank],
                                                              slot),
                                       0));
    }
    return 0;
}

int enqueue_rank_streams_wait_after_control(const Options &opt,
                                            RankState ranks[kGpus],
                                            cudaStream_t control_stream) {
    const int slot = next_graph_order_event_slot(ranks);
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    cudaEvent_t ev = graph_stream_done_event(ranks[0], slot);
    if (!ev) return 1;
    CHECK_CUDA(cudaEventRecord(ev, control_stream));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaStreamWaitEvent(r.stream, ev, 0));
    }
    return 0;
}

int nccl_broadcast_f32_from_device0_to_current_full(
    const Options &opt,
    RankState ranks[kGpus],
    const float *src_device0,
    uint64_t elems,
    const char *label);

int fill_shared_ffn_inputs_from_normed(const Options &opt,
                                       const SharedHcControls *hc,
                                       const ResidentF8Dense &gate,
                                       const ResidentF8Dense &up,
                                       RankState ranks[kGpus]) {
    if (!hc || !hc->d_ffn_normed) return 1;
    if (gate.cols != kHidden || up.cols != kHidden ||
        gate.rows_per_gpu != kMid / kGpus ||
        up.rows_per_gpu != kMid / kGpus) {
        return 2;
    }
    const uint64_t full_elems = (uint64_t)opt.slots * kHidden;
    const uint64_t x_elems = (uint64_t)opt.slots * kHidden;
    const int block = 256;
    if (nccl_broadcast_f32_from_device0_to_current_full(
            opt, ranks, hc->d_ffn_normed, full_elems,
            "shared_ffn_normed_input") != 0) {
        return 4;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_current_full) return 3;
        if (gate.d_x_half[(size_t)rank]) {
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((x_elems + block - 1) / block), block, 0,
                r.stream>>>(gate.d_x_half[(size_t)rank], r.d_current_full,
                             (uint32_t)gate.cols, (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        if (up.d_x_half[(size_t)rank]) {
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((x_elems + block - 1) / block), block, 0,
                r.stream>>>(up.d_x_half[(size_t)rank], r.d_current_full,
                             (uint32_t)up.cols, (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    return 0;
}

int materialize_shared_swiglu_down_input(const Options &opt,
                                         const ResidentF8Dense &gate,
                                         const ResidentF8Dense &up,
                                         const ResidentF8Dense &down,
                                         RankState ranks[kGpus]) {
    if (gate.rows_per_gpu != kMid / kGpus ||
        up.rows_per_gpu != kMid / kGpus ||
        down.cols != kMid) {
        return 1;
    }
    const uint32_t rows = (uint32_t)gate.rows_per_gpu;
    const int block = 256;
    const uint64_t shard_elems = (uint64_t)opt.slots * rows;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    for (int src = 0; src < kGpus; ++src) {
        CHECK_CUDA(cudaSetDevice(ranks[src].device));
        if (!down.d_x[(size_t)src] ||
            !gate.d_out[(size_t)src] ||
            !up.d_out[(size_t)src]) {
            return 2;
        }
        shared_swiglu_shard_to_float_kernel<<<
            (unsigned int)((shard_elems + block - 1) / block), block, 0,
            ranks[src].stream>>>(down.d_x[(size_t)src],
                                 gate.d_out[(size_t)src],
                                 up.d_out[(size_t)src],
                                 (uint32_t)src, rows, (uint32_t)opt.slots,
                                 kRoutedSwigluClamp);
        CHECK_CUDA(cudaGetLastError());
    }
    if (graph_event_order) {
        if (enqueue_cross_gpu_stream_barrier(ranks, false) != 0) return 3;
    } else {
        for (int src = 0; src < kGpus; ++src) {
            CHECK_CUDA(cudaSetDevice(ranks[src].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[src].stream));
        }
    }
    const size_t width = (size_t)rows * sizeof(float);
    if (graph_event_order) {
        for (int dst = 0; dst < kGpus; ++dst) {
            CHECK_CUDA(cudaSetDevice(ranks[dst].device));
            cudaStream_t stream = ranks[dst].stream;
            for (int src = 0; src < kGpus; ++src) {
                if (src == dst) continue;
                for (int slot = 0; slot < opt.slots; ++slot) {
                    float *dst_ptr = down.d_x[(size_t)dst] +
                                     (size_t)slot * kMid + (size_t)src * rows;
                    const float *src_ptr = down.d_x[(size_t)src] +
                                           (size_t)slot * kMid + (size_t)src * rows;
                    enqueue_graph_f32_copy_between_devices(
                        opt, ranks[dst].device, ranks[src].device,
                        dst_ptr, src_ptr, (uint64_t)rows, stream, block);
                }
            }
        }
    } else {
        for (int src = 0; src < kGpus; ++src) {
            for (int slot = 0; slot < opt.slots; ++slot) {
                void *dsts[kGpus] = {};
                for (int dst = 0; dst < kGpus; ++dst) {
                    dsts[dst] = down.d_x[(size_t)dst] +
                                (size_t)slot * kMid + (size_t)src * rows;
                }
                const float *src_ptr = down.d_x[(size_t)src] +
                                       (size_t)slot * kMid + (size_t)src * rows;
                if (nccl_broadcast_bytes_from_rank(
                        ranks, src, src_ptr, dsts, width,
                        "shared_swiglu_down_input") != 0) {
                    return 5;
                }
            }
        }
    }
    if (graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 4;
    } else {
        for (int dst = 0; dst < kGpus; ++dst) {
            CHECK_CUDA(cudaSetDevice(ranks[dst].device));
            if (ranks[dst].copy_stream) {
                CHECK_CUDA(cudaStreamSynchronize(ranks[dst].copy_stream));
            }
            CHECK_CUDA(cudaStreamSynchronize(ranks[dst].stream));
        }
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
    api->mmgs_clamped =
        (pfn_mmgs)dlsym(lib, "ggml_turbomind_mul_mat_grouped_gated_silu_clamped_total_tokens");
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
    if (p.d_w_contiguous) {
        CHECK_CUDA(cudaFree(p.d_w_contiguous));
    } else {
        for (void *v : p.d_w_active) {
            if (v) CHECK_CUDA(cudaFree(v));
        }
    }
    if (p.d_s_contiguous) {
        CHECK_CUDA(cudaFree(p.d_s_contiguous));
    } else {
        for (void *v : p.d_s_active) {
            if (v) CHECK_CUDA(cudaFree(v));
        }
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

    if (active.empty()) return 1;
    const size_t total_weight_bytes = entry.weight_bytes_per_expert * active.size();
    const size_t total_scale_bytes = entry.scale_bytes_per_expert * active.size();
    cudaError_t alloc_rc = cudaMalloc(&out->d_w_contiguous, total_weight_bytes);
    if (alloc_rc == cudaSuccess) {
        alloc_rc = cudaMalloc(&out->d_s_contiguous, total_scale_bytes);
        if (alloc_rc != cudaSuccess) {
            CHECK_CUDA(cudaFree(out->d_w_contiguous));
            out->d_w_contiguous = nullptr;
        }
    }
    if (alloc_rc != cudaSuccess) {
        (void)cudaGetLastError();
        out->d_w_contiguous = nullptr;
        out->d_s_contiguous = nullptr;
    }

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
        if (out->d_w_contiguous && out->d_s_contiguous) {
            out->d_w_active[i] = static_cast<uint8_t *>(out->d_w_contiguous) +
                                 i * entry.weight_bytes_per_expert;
            out->d_s_active[i] = static_cast<uint8_t *>(out->d_s_contiguous) +
                                 i * entry.scale_bytes_per_expert;
        } else {
            CHECK_CUDA(cudaMalloc(&out->d_w_active[i], entry.weight_bytes_per_expert));
            CHECK_CUDA(cudaMalloc(&out->d_s_active[i], entry.scale_bytes_per_expert));
        }
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

void free_layer_expert_cache(LayerExpertCache *layer) {
    if (!layer) return;
    for (int p = 0; p < kGpus; ++p) {
        free_packed(layer->gated[p]);
        free_packed(layer->down[p]);
    }
    *layer = LayerExpertCache{};
}

void close_shared_expert_bindings(SharedExpertBindings *shared);

int open_shared_expert_bindings(const Options &opt, SharedExpertBindings *shared) {
    std::vector<int> active;
    for (int e = 0; e < kPackedLocalExperts; ++e) active.push_back(e);
    const auto start = std::chrono::steady_clock::now();

    for (int layer = 0; layer < 43; ++layer) {
        if (opt.resident_profile_layer >= 0 && layer != opt.resident_profile_layer) {
            continue;
        }
        LayerExpertCache &cache = shared->layers[layer];
        if (parse_tm_index(opt.tm_index_path, layer, &cache.bindings) != 0) {
            std::fprintf(stderr, "tm index parse failed for layer %d\n", layer);
            close_shared_expert_bindings(shared);
            return 1;
        }
        uint64_t layer_bytes_by_gpu[kGpus] = {};
        if (opt.parallel_expert_load_gate) {
            int rc[kGpus] = {};
            std::thread workers[kGpus];
            for (int p = 0; p < kGpus; ++p) {
                workers[p] = std::thread([&, p]() {
                    uint64_t layer_bytes = 0;
                    int local_rc = pack_descriptor_set(
                        opt.devices[p], cache.bindings.gated, p, active,
                        opt.pack_dir, &cache.gated[p], &layer_bytes);
                    if (local_rc == 0) {
                        local_rc = pack_descriptor_set(
                            opt.devices[p], cache.bindings.down, p, active,
                            opt.pack_dir, &cache.down[p], &layer_bytes);
                    }
                    layer_bytes_by_gpu[p] = layer_bytes;
                    rc[p] = local_rc;
                });
            }
            for (int p = 0; p < kGpus; ++p) workers[p].join();
            for (int p = 0; p < kGpus; ++p) {
                if (rc[p] != 0) {
                    close_shared_expert_bindings(shared);
                    return 2;
                }
                cache.bytes += layer_bytes_by_gpu[p];
                shared->bytes += layer_bytes_by_gpu[p];
            }
        } else {
            for (int p = 0; p < kGpus; ++p) {
                uint64_t layer_bytes = 0;
                if (pack_descriptor_set(opt.devices[p], cache.bindings.gated, p, active,
                                        opt.pack_dir, &cache.gated[p], &layer_bytes) != 0 ||
                    pack_descriptor_set(opt.devices[p], cache.bindings.down, p, active,
                                        opt.pack_dir, &cache.down[p], &layer_bytes) != 0) {
                    close_shared_expert_bindings(shared);
                    return 2;
                }
                cache.bytes += layer_bytes;
                shared->bytes += layer_bytes;
            }
        }
        if (opt.parallel_expert_load_gate) {
            std::printf("tp_ep_parallel_expert_load_layer\tlayer\t%d\tbytes\t%llu\tPASS\n",
                        layer, (unsigned long long)cache.bytes);
            std::fflush(stdout);
        }
        cache.initialized = true;
    }
    shared->initialized = true;
    const auto stop = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    std::printf("tp_ep_shared_expert_bindings_load\tlayers\t43\tparallel\t%d\t"
                "bytes\t%llu\tload_ms\t%.6f\tPASS\n",
                opt.parallel_expert_load_gate ? 1 : 0,
                (unsigned long long)shared->bytes,
                ms);
    std::fflush(stdout);
    return 0;
}

void close_shared_expert_bindings(SharedExpertBindings *shared) {
    if (!shared) return;
    for (int layer = 0; layer < 43; ++layer) {
        free_layer_expert_cache(&shared->layers[layer]);
    }
    *shared = SharedExpertBindings{};
}

int routed_executor_rows(const RankState &rank, const Options &opt) {
    int rows = rank.routes;
    if (opt.post_attention_static_executor_route_cap > 0) {
        rows = std::min(rows, opt.post_attention_static_executor_route_cap);
    }
    return rows;
}

int routed_compose_rows(const RankState &rank, const Options &opt) {
    int rows = rank.routes;
    if (opt.post_attention_static_compose_route_cap > 0) {
        rows = std::min(rows, opt.post_attention_static_compose_route_cap);
    }
    return rows;
}

int run_gate(RankState &rank, const Api &api, int executor_rows) {
    if (executor_rows <= 0) return 0;
    return api.mmgs(rank.d_a, nullptr, rank.d_offsets, kLocalExperts, executor_rows,
                    (const void * const *)rank.gated.d_w_table,
                    (const void * const *)rank.gated.d_s_table,
                    kDType, kFusedN, kHidden, kGroupSize, rank.gated.k_pack,
                    rank.d_gated, rank.stream);
}

int run_gate_clamped(RankState &rank, const Api &api, bool apply_route_scale,
                     int executor_rows) {
    if (executor_rows <= 0) return 0;
    if (!rank.d_gate_up) return 1;
    CHECK_CUDA(cudaSetDevice(rank.device));
    const int rc = api.mmgt(rank.d_a, nullptr, rank.d_offsets, kLocalExperts, executor_rows,
                            (const void * const *)rank.gated.d_w_table,
                            (const void * const *)rank.gated.d_s_table,
                            kDType, kFusedN, kHidden, kGroupSize, rank.gated.k_pack,
                            rank.d_gate_up, rank.stream);
    if (rc != 0) return rc;
    const uint64_t elems = (uint64_t)executor_rows * kMid;
    routed_fused_gate_up_swiglu_clamp_kernel<<<
        (unsigned int)((elems + 255) / 256), 256, 0, rank.stream>>>(
            rank.d_gated, rank.d_gate_up,
            apply_route_scale ? rank.d_route_inv_scale : nullptr,
            (uint64_t)executor_rows, kRoutedSwigluClamp);
    CHECK_CUDA(cudaGetLastError());
    return 0;
}

int run_gate_selected(RankState &rank, const Api &api, const Options &opt) {
    const int executor_rows = routed_executor_rows(rank, opt);
    if (!opt.routed_ffn_norm_input_gate) {
        return run_gate(rank, api, executor_rows);
    }
    if (opt.fused_gated_silu_gate && !opt.reference_hc_reduce_gate &&
        api.mmgs_clamped) {
        return api.mmgs_clamped(
            rank.d_a, nullptr, rank.d_offsets, kLocalExperts, executor_rows,
            (const void * const *)rank.gated.d_w_table,
            (const void * const *)rank.gated.d_s_table,
            kDType, kFusedN, kHidden, kGroupSize, rank.gated.k_pack,
            rank.d_gated, rank.stream);
    }
    return run_gate_clamped(rank, api, opt.reference_hc_reduce_gate,
                            executor_rows);
}

int run_down(RankState &rank, const Api &api, const Options &opt) {
    const int executor_rows = routed_executor_rows(rank, opt);
    if (executor_rows <= 0) return 0;
    return api.mmgt(rank.d_gated, nullptr, rank.d_offsets, kLocalExperts, executor_rows,
                    (const void * const *)rank.down.d_w_table,
                    (const void * const *)rank.down.d_s_table,
                    kDType, kHidden, kMid, kGroupSize, rank.down.k_pack,
                    rank.d_down, rank.stream);
}

void log_route_half_stats(const char *tag, int layer, int rank_id,
                          const __half *ptr, size_t elems, cudaStream_t stream) {
    if (!ptr || elems == 0) return;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<__half> host(elems);
    CHECK_CUDA(cudaMemcpy(host.data(), ptr, elems * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    int finite_bad = 0;
    size_t first_bad = (size_t)-1;
    float max_abs = 0.0f;
    for (size_t i = 0; i < elems; ++i) {
        const float v = __half2float(host[i]);
        if (!std::isfinite(v)) {
            if (finite_bad == 0) first_bad = i;
            ++finite_bad;
        } else {
            max_abs = fmaxf(max_abs, fabsf(v));
        }
    }
    std::fprintf(stderr,
                 "tp_ep_route_tensor_stats\ttag\t%s\tlayer\t%d\trank\t%d\telems\t%zu\tfinite_bad\t%d\tfirst_bad\t%zu\tmax_abs\t%.9g\n",
                 tag, layer, rank_id, elems, finite_bad, first_bad, max_abs);
}

void merge_tensor_stats(TensorF32Stats *dst, const TensorF32Stats &src) {
    if (!dst) return;
    if (src.finite_bad != 0 && dst->finite_bad == 0) {
        dst->first_bad = src.first_bad;
    }
    dst->finite_bad += src.finite_bad;
    dst->max_abs = fmaxf(dst->max_abs, src.max_abs);
}

TensorF32Stats collect_tensor_f32_stats(const float *ptr, size_t elems,
                                        cudaStream_t stream) {
    TensorF32Stats stats;
    if (!ptr || elems == 0) return stats;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> host(elems);
    CHECK_CUDA(cudaMemcpy(host.data(), ptr, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elems; ++i) {
        const float v = host[i];
        if (!std::isfinite(v)) {
            if (stats.finite_bad == 0) stats.first_bad = i;
            ++stats.finite_bad;
        } else {
            stats.max_abs = fmaxf(stats.max_abs, fabsf(v));
        }
    }
    return stats;
}

TensorF32Stats collect_raw_swa_row_stats(const float *ptr, uint32_t slots,
                                         uint32_t raw_rows, uint32_t raw_row,
                                         uint32_t head_dim,
                                         cudaStream_t stream) {
    TensorF32Stats stats;
    if (!ptr || slots == 0 || raw_rows == 0 || raw_row >= raw_rows ||
        head_dim == 0) {
        return stats;
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> host((size_t)slots * (size_t)head_dim);
    const float *src = ptr + (uint64_t)raw_row * (uint64_t)head_dim;
    CHECK_CUDA(cudaMemcpy2D(host.data(), (size_t)head_dim * sizeof(float),
                            src,
                            (size_t)raw_rows * (size_t)head_dim * sizeof(float),
                            (size_t)head_dim * sizeof(float), (size_t)slots,
                            cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < host.size(); ++i) {
        const float v = host[i];
        if (!std::isfinite(v)) {
            if (stats.finite_bad == 0) stats.first_bad = i;
            ++stats.finite_bad;
        } else {
            stats.max_abs = fmaxf(stats.max_abs, fabsf(v));
        }
    }
    return stats;
}

TensorF32DiffStats collect_tensor_f32_diff_stats(const float *a, const float *b,
                                                 size_t elems,
                                                 cudaStream_t stream) {
    TensorF32DiffStats stats;
    if (!a || !b || elems == 0) return stats;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> ha(elems);
    std::vector<float> hb(elems);
    CHECK_CUDA(cudaMemcpy(ha.data(), a, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hb.data(), b, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elems; ++i) {
        const float av = ha[i];
        const float bv = hb[i];
        if (!std::isfinite(av) || !std::isfinite(bv)) {
            if (stats.bad == 0) stats.first_bad = i;
            ++stats.bad;
            continue;
        }
        const float diff = fabsf(av - bv);
        const float denom = fmaxf(fabsf(bv), 1.0e-12f);
        stats.max_abs = fmaxf(stats.max_abs, diff);
        stats.max_rel = fmaxf(stats.max_rel, diff / denom);
    }
    return stats;
}

void log_tensor_f32_diff_summary(const char *tag, int layer,
                                 const float *got, const float *ref,
                                 size_t elems, cudaStream_t stream) {
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> hg(elems);
    std::vector<float> hr(elems);
    CHECK_CUDA(cudaMemcpy(hg.data(), got, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hr.data(), ref, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double sq = 0.0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float got_max = 0.0f;
    float ref_max = 0.0f;
    int finite_bad = 0;
    size_t first_bad = (size_t)-1;
    for (size_t i = 0; i < elems; ++i) {
        const float g = hg[i];
        const float r = hr[i];
        if (!std::isfinite(g) || !std::isfinite(r)) {
            if (finite_bad == 0) first_bad = i;
            ++finite_bad;
            continue;
        }
        got_max = fmaxf(got_max, fabsf(g));
        ref_max = fmaxf(ref_max, fabsf(r));
        const float diff = fabsf(g - r);
        const float rel = diff / fmaxf(fabsf(r), 1.0e-12f);
        max_abs = fmaxf(max_abs, diff);
        max_rel = fmaxf(max_rel, rel);
        sq += (double)diff * (double)diff;
    }
    const double rms = elems ? std::sqrt(sq / (double)elems) : 0.0;
    const char *status = (finite_bad == 0 && max_abs <= 1.0e-5f) ? "PASS" : "DIFF";
    std::printf("tp_ep_compressed_reference_diff\tlayer\t%d\ttensor\t%s\t"
                "elems\t%zu\tmax_abs\t%.9g\trms\t%.9g\tmax_rel\t%.9g\t"
                "finite_bad\t%d\tfirst_bad\t%zu\tgot_max\t%.9g\t"
                "reference_max\t%.9g\t%s\n",
                layer, tag, elems, max_abs, rms, max_rel, finite_bad,
                first_bad, got_max, ref_max, status);
}

void log_tensor_f32_stats(const char *tag, int layer, int rank_id,
                          const float *ptr, size_t elems, cudaStream_t stream) {
    const TensorF32Stats stats = collect_tensor_f32_stats(ptr, elems, stream);
    std::fprintf(stderr,
                 "tp_ep_tensor_stats\ttag\t%s\tlayer\t%d\trank\t%d\telems\t%zu\tfinite_bad\t%d\tfirst_bad\t%zu\tmax_abs\t%.9g\n",
                 tag, layer, rank_id, elems, stats.finite_bad, stats.first_bad,
                 stats.max_abs);
}

void log_hc_current_full_rank_parity(const Options &opt,
                                     RankState ranks[kGpus],
                                     int layer,
                                     size_t elems) {
    if (elems == 0 || !ranks[0].d_current_full) return;
    CHECK_CUDA(cudaSetDevice(ranks[0].device));
    if (ranks[0].stream) CHECK_CUDA(cudaStreamSynchronize(ranks[0].stream));
    std::vector<float> ref(elems);
    std::vector<float> got(elems);
    CHECK_CUDA(cudaMemcpy(ref.data(), ranks[0].d_current_full,
                          elems * sizeof(float), cudaMemcpyDeviceToHost));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.d_current_full) continue;
        CHECK_CUDA(cudaSetDevice(r.device));
        if (r.stream) CHECK_CUDA(cudaStreamSynchronize(r.stream));
        CHECK_CUDA(cudaMemcpy(got.data(), r.d_current_full,
                              elems * sizeof(float), cudaMemcpyDeviceToHost));
        unsigned long long mismatches = 0;
        size_t first_mismatch = (size_t)-1;
        float max_abs = 0.0f;
        int finite_bad = 0;
        for (size_t i = 0; i < elems; ++i) {
            const float a = got[i];
            const float b = ref[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                if (first_mismatch == (size_t)-1) first_mismatch = i;
                ++mismatches;
                ++finite_bad;
                continue;
            }
            const float diff = fabsf(a - b);
            if (diff > 0.0f) {
                if (first_mismatch == (size_t)-1) first_mismatch = i;
                ++mismatches;
                max_abs = fmaxf(max_abs, diff);
            }
        }
        std::printf("tp_ep_hc_current_full_rank_diff\tlayer\t%d\trank\t%d\t"
                    "elems\t%zu\tmismatches\t%llu\tfirst_mismatch\t%zu\t"
                    "max_abs\t%.9g\tfinite_bad\t%d\t%s\n",
                    layer, rank, elems,
                    (unsigned long long)mismatches,
                    first_mismatch, max_abs, finite_bad,
                    mismatches == 0ull ? "PASS" : "DIFF");
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
}

HalfInputDiffStats collect_shared_half_input_diff(RankState &r,
                                                  const __half *actual,
                                                  const float *current_full,
                                                  uint32_t cols,
                                                  uint32_t slots,
                                                  cudaStream_t stream) {
    HalfInputDiffStats stats;
    if (!actual || !current_full || !r.d_half_diff_counts ||
        !r.d_half_diff_max_bits || !r.d_half_diff_first ||
        cols == 0 || slots == 0) {
        return stats;
    }
    compare_shared_half_input_with_current_kernel<<<1, 256, 0, stream>>>(
        r.d_half_diff_counts, r.d_half_diff_max_bits, r.d_half_diff_first,
        actual, current_full, cols, slots);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    unsigned long long counts[2] = {};
    unsigned int max_bits = 0u;
    CHECK_CUDA(cudaMemcpy(counts, r.d_half_diff_counts, sizeof(counts),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&max_bits, r.d_half_diff_max_bits, sizeof(max_bits),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&stats.first_mismatch, r.d_half_diff_first,
                          sizeof(stats.first_mismatch),
                          cudaMemcpyDeviceToHost));
    stats.compared = counts[0];
    stats.mismatches = counts[1];
    std::memcpy(&stats.max_abs, &max_bits, sizeof(stats.max_abs));
    return stats;
}

HalfInputDiffStats collect_half_input_tensor_diff(RankState &r,
                                                  const __half *actual,
                                                  const float *expected_f32,
                                                  uint32_t cols,
                                                  uint32_t slots,
                                                  cudaStream_t stream) {
    HalfInputDiffStats stats;
    if (!actual || !expected_f32 || !r.d_half_diff_counts ||
        !r.d_half_diff_max_bits || !r.d_half_diff_first ||
        cols == 0 || slots == 0) {
        return stats;
    }
    compare_half_input_with_f32_tensor_kernel<<<1, 256, 0, stream>>>(
        r.d_half_diff_counts, r.d_half_diff_max_bits, r.d_half_diff_first,
        actual, expected_f32, cols, slots);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    unsigned long long counts[2] = {};
    unsigned int max_bits = 0u;
    CHECK_CUDA(cudaMemcpy(counts, r.d_half_diff_counts, sizeof(counts),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&max_bits, r.d_half_diff_max_bits, sizeof(max_bits),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&stats.first_mismatch, r.d_half_diff_first,
                          sizeof(stats.first_mismatch),
                          cudaMemcpyDeviceToHost));
    stats.compared = counts[0];
    stats.mismatches = counts[1];
    std::memcpy(&stats.max_abs, &max_bits, sizeof(stats.max_abs));
    return stats;
}

HalfInputDiffStats collect_route_half_input_diff(RankState &r,
                                                 const __half *actual,
                                                 const float *current_full,
                                                 const int *route_slots,
                                                 int routes,
                                                 cudaStream_t stream) {
    HalfInputDiffStats stats;
    if (!actual || !current_full || !route_slots || !r.d_half_diff_counts ||
        !r.d_half_diff_max_bits || !r.d_half_diff_first || routes <= 0) {
        return stats;
    }
    compare_route_half_input_with_current_kernel<<<1, 256, 0, stream>>>(
        r.d_half_diff_counts, r.d_half_diff_max_bits, r.d_half_diff_first,
        actual, current_full, route_slots, routes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    unsigned long long counts[2] = {};
    unsigned int max_bits = 0u;
    CHECK_CUDA(cudaMemcpy(counts, r.d_half_diff_counts, sizeof(counts),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&max_bits, r.d_half_diff_max_bits, sizeof(max_bits),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&stats.first_mismatch, r.d_half_diff_first,
                          sizeof(stats.first_mismatch),
                          cudaMemcpyDeviceToHost));
    stats.compared = counts[0];
    stats.mismatches = counts[1];
    std::memcpy(&stats.max_abs, &max_bits, sizeof(stats.max_abs));
    return stats;
}

HalfInputDiffStats collect_route_half_input_diff_limited(
    RankState &r,
    const __half *actual,
    const float *current_full,
    const int *route_slots,
    const int *route_totals,
    int routes,
    int rank,
    cudaStream_t stream) {
    HalfInputDiffStats stats;
    if (!actual || !current_full || !route_slots || !r.d_half_diff_counts ||
        !r.d_half_diff_max_bits || !r.d_half_diff_first || routes <= 0) {
        return stats;
    }
    compare_route_half_input_with_current_limited_kernel<<<1, 256, 0, stream>>>(
        r.d_half_diff_counts, r.d_half_diff_max_bits, r.d_half_diff_first,
        actual, current_full, route_slots, route_totals, routes, rank);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    unsigned long long counts[2] = {};
    unsigned int max_bits = 0u;
    CHECK_CUDA(cudaMemcpy(counts, r.d_half_diff_counts, sizeof(counts),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&max_bits, r.d_half_diff_max_bits, sizeof(max_bits),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&stats.first_mismatch, r.d_half_diff_first,
                          sizeof(stats.first_mismatch),
                          cudaMemcpyDeviceToHost));
    stats.compared = counts[0];
    stats.mismatches = counts[1];
    std::memcpy(&stats.max_abs, &max_bits, sizeof(stats.max_abs));
    return stats;
}

void log_half_input_diff(const char *family,
                         int layer,
                         int rank,
                         const HalfInputDiffStats &stats) {
    const char *status = stats.mismatches == 0ull ? "PASS" : "DIFF";
    std::printf("tp_ep_rank_major_half_input_diff\tlayer\t%d\trank\t%d\t"
                "family\t%s\tcompared\t%llu\tmismatches\t%llu\t"
                "first_mismatch\t%d\tmax_abs\t%.9g\t%s\n",
                layer, rank, family,
                (unsigned long long)stats.compared,
                (unsigned long long)stats.mismatches,
                stats.first_mismatch, stats.max_abs, status);
}

void log_attention_projection_input_diff(const char *family,
                                         int layer,
                                         int rank,
                                         const HalfInputDiffStats &stats) {
    const char *status = stats.mismatches == 0ull ? "PASS" : "DIFF";
    std::printf("tp_ep_attention_projection_input_diff\tlayer\t%d\trank\t%d\t"
                "family\t%s\tcompared\t%llu\tmismatches\t%llu\t"
                "first_mismatch\t%d\tmax_abs\t%.9g\t%s\n",
                layer, rank, family,
                (unsigned long long)stats.compared,
                (unsigned long long)stats.mismatches,
                stats.first_mismatch, stats.max_abs, status);
}

unsigned short f32_to_half_raw_host(float v) {
    if (!std::isfinite(v)) v = 0.0f;
    v = std::fmin(kFp16Max, std::fmax(-kFp16Max, v));
    const __half h = __float2half(v);
    unsigned short raw = 0u;
    std::memcpy(&raw, &h, sizeof(raw));
    return raw;
}

float rank_major_debug_scale(const std::vector<float> &src,
                             uint32_t slot,
                             uint32_t shard_cols,
                             uint32_t ranks,
                             uint32_t slots,
                             float eps) {
    const uint32_t cols = shard_cols * ranks;
    float max_abs = 0.0f;
    for (uint32_t col = 0; col < cols; ++col) {
        const uint32_t src_rank = col / shard_cols;
        const uint32_t local_col = col - src_rank * shard_cols;
        const uint64_t src_i =
            ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot) *
                (uint64_t)shard_cols +
            (uint64_t)local_col;
        const float v = src[src_i];
        if (std::isfinite(v)) max_abs = std::fmax(max_abs, std::fabs(v));
    }
    float sum = 0.0f;
    if (max_abs > 0.0f && std::isfinite(max_abs)) {
        for (uint32_t col = 0; col < cols; ++col) {
            const uint32_t src_rank = col / shard_cols;
            const uint32_t local_col = col - src_rank * shard_cols;
            const uint64_t src_i =
                ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot) *
                    (uint64_t)shard_cols +
                (uint64_t)local_col;
            const float v = src[src_i];
            if (std::isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    if (!(max_abs > 0.0f) || !std::isfinite(max_abs)) {
        return 1.0f / std::sqrt(eps);
    }
    return 1.0f / std::sqrt(sum / (float)cols + eps / (max_abs * max_abs)) /
           max_abs;
}

float slot_major_debug_scale(const std::vector<float> &src,
                             uint32_t slot,
                             uint32_t cols,
                             float eps) {
    float max_abs = 0.0f;
    const uint64_t base = (uint64_t)slot * (uint64_t)cols;
    for (uint32_t col = 0; col < cols; ++col) {
        const float v = src[base + col];
        if (std::isfinite(v)) max_abs = std::fmax(max_abs, std::fabs(v));
    }
    float sum = 0.0f;
    if (max_abs > 0.0f && std::isfinite(max_abs)) {
        for (uint32_t col = 0; col < cols; ++col) {
            const float v = src[base + col];
            if (std::isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    if (!(max_abs > 0.0f) || !std::isfinite(max_abs)) {
        return 1.0f / std::sqrt(eps);
    }
    return 1.0f / std::sqrt(sum / (float)cols + eps / (max_abs * max_abs)) /
           max_abs;
}

void log_attention_rank_major_input_debug(
    const char *family,
    int layer,
    RankState &r,
    const __half *actual,
    const float *expected_f32,
    const float *slot_major,
    const float *rank_major,
    const float *weight,
    uint32_t slots,
    cudaStream_t stream) {
    if (!actual || !expected_f32 || !slot_major || !rank_major || !weight ||
        slots == 0) {
        return;
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const uint32_t shard_cols = kHidden / kGpus;
    const size_t elems = (size_t)slots * (size_t)kHidden;
    std::vector<__half> h_actual(elems);
    std::vector<float> h_expected(elems);
    std::vector<float> h_slot(elems);
    std::vector<float> h_rank_major(elems);
    std::vector<float> h_weight(kHidden);
    CHECK_CUDA(cudaMemcpy(h_actual.data(), actual,
                          elems * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_expected.data(), expected_f32,
                          elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_slot.data(), slot_major,
                          elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_rank_major.data(), rank_major,
                          elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_weight.data(), weight,
                          kHidden * sizeof(float), cudaMemcpyDeviceToHost));

    unsigned long long raw_mismatches = 0ull;
    size_t raw_first = (size_t)-1;
    float raw_max_abs = 0.0f;
    for (uint32_t slot = 0; slot < slots; ++slot) {
        for (uint32_t col = 0; col < (uint32_t)kHidden; ++col) {
            const uint32_t src_rank = col / shard_cols;
            const uint32_t local_col = col - src_rank * shard_cols;
            const uint64_t src_i =
                ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot) *
                    (uint64_t)shard_cols +
                (uint64_t)local_col;
            const uint64_t slot_i =
                (uint64_t)slot * (uint64_t)kHidden + (uint64_t)col;
            const float a = h_rank_major[src_i];
            const float b = h_slot[slot_i];
            const float diff = std::fabs(a - b);
            if (diff > 0.0f || !std::isfinite(a) || !std::isfinite(b)) {
                if (raw_first == (size_t)-1) raw_first = (size_t)slot_i;
                ++raw_mismatches;
                raw_max_abs = std::fmax(raw_max_abs, diff);
            }
        }
    }

    int first_half = -1;
    unsigned short got_raw = 0u;
    unsigned short exp_raw = 0u;
    float got = 0.0f;
    float expected = 0.0f;
    for (size_t i = 0; i < elems; ++i) {
        std::memcpy(&got_raw, &h_actual[i], sizeof(got_raw));
        exp_raw = f32_to_half_raw_host(h_expected[i]);
        if (got_raw != exp_raw) {
            first_half = (int)i;
            got = __half2float(h_actual[i]);
            expected = __half2float(__float2half(h_expected[i]));
            break;
        }
    }

    uint32_t slot = 0u;
    uint32_t col = 0u;
    uint64_t src_i = 0u;
    float rank_major_value = 0.0f;
    float slot_major_value = 0.0f;
    float norm_weight = 0.0f;
    float slot_scale = 0.0f;
    float rank_major_scale = 0.0f;
    if (first_half >= 0) {
        slot = (uint32_t)((uint32_t)first_half / (uint32_t)kHidden);
        col = (uint32_t)((uint32_t)first_half % (uint32_t)kHidden);
        const uint32_t src_rank = col / shard_cols;
        const uint32_t local_col = col - src_rank * shard_cols;
        src_i = ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot) *
                    (uint64_t)shard_cols +
                (uint64_t)local_col;
        rank_major_value = h_rank_major[src_i];
        slot_major_value = h_slot[(uint64_t)slot * (uint64_t)kHidden + col];
        norm_weight = h_weight[col];
        slot_scale = slot_major_debug_scale(h_slot, slot, (uint32_t)kHidden,
                                            1.0e-6f);
        rank_major_scale = rank_major_debug_scale(
            h_rank_major, slot, shard_cols, (uint32_t)kGpus, slots, 1.0e-6f);
    }

    std::printf("tp_ep_attention_rank_major_input_debug\tlayer\t%d\t"
                "family\t%s\traw_mismatches\t%llu\traw_first\t%zu\t"
                "raw_max_abs\t%.9g\tfirst_half_mismatch\t%d\tslot\t%u\t"
                "col\t%u\tsrc_index\t%llu\trank_major_value\t%.9g\t"
                "slot_major_value\t%.9g\tweight\t%.9g\tgot_half\t%.9g\t"
                "expected_half\t%.9g\tslot_scale\t%.9g\t"
                "rank_major_scale\t%.9g\t%s\n",
                layer, family, (unsigned long long)raw_mismatches, raw_first,
                raw_max_abs, first_half, slot, col,
                (unsigned long long)src_i, rank_major_value, slot_major_value,
                norm_weight, got, expected, slot_scale, rank_major_scale,
                (raw_mismatches == 0ull && first_half < 0) ? "PASS" : "DIFF");
}

bool should_log_routed_semantic_stats(const Options &opt) {
    if (opt.decode_cudagraph_gate || opt.true_ds4_semantic_skip_stats_gate) {
        return false;
    }
    if (!opt.routed_ffn_norm_input_gate) return false;
    if (opt.layer <= 2) return true;
    return opt.reference_hc_reduce_gate && opt.layer >= 30 && opt.layer <= 32;
}

bool should_log_reference_hc_window(const Options &opt) {
    return opt.reference_hc_reduce_gate && opt.layer >= 30 && opt.layer <= 32;
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

int check_repeat(RankState &rank, const Api &api, double *max_abs, int *bad, int *nan) {
    const Options no_executor_cap{};
    const size_t elems = (size_t)rank.routes * kHidden;
    std::vector<__half> first(elems);
    std::vector<__half> second(elems);
    CHECK_CUDA(cudaSetDevice(rank.device));
    if (run_gate(rank, api, rank.routes) != 0 ||
        run_down(rank, api, no_executor_cap) != 0) return 1;
    CHECK_CUDA(cudaStreamSynchronize(rank.stream));
    CHECK_CUDA(cudaMemcpy(first.data(), rank.d_down, elems * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    if (run_gate(rank, api, rank.routes) != 0 ||
        run_down(rank, api, no_executor_cap) != 0) return 1;
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
                            std::vector<float> *route_weights,
                            int *routes,
                            int *active_experts,
                            int *max_routes_per_expert) {
    std::vector<int> counts((size_t)kLocalExperts, 0);
    for (int slot = 0; slot < slots; ++slot) {
        for (int k = 0; k < top_k; ++k) {
            const int dst_rank = (slot * top_k + k) % kGpus;
            if (dst_rank != rank) continue;
            const int local = (slot + k * 7 + rank) % kPackedLocalExperts;
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
        if (route_weights) route_weights->assign((size_t)running, kSyntheticRouteWeight);
        std::vector<int> cursor = *offsets;
        for (int slot = 0; slot < slots; ++slot) {
            for (int k = 0; k < top_k; ++k) {
                const int dst_rank = (slot * top_k + k) % kGpus;
                if (dst_rank != rank) continue;
                const int local = (slot + k * 7 + rank) % kPackedLocalExperts;
                const int idx = cursor[(size_t)local]++;
                (*route_slots)[(size_t)idx] = slot;
                if (route_weights) (*route_weights)[(size_t)idx] = kSyntheticRouteWeight;
            }
        }
    }
    *routes = running;
    *active_experts = active;
    *max_routes_per_expert = max_routes;
}

void build_route_index_by_slot_for_rank(int rank, int slots, int top_k,
                                        std::vector<int> *route_index_by_slot) {
    std::vector<int> offsets;
    std::vector<int> route_slots;
    int routes = 0;
    int active_experts = 0;
    int max_routes_per_expert = 0;
    build_offsets_for_rank(rank, slots, top_k, &offsets, &route_slots, nullptr, &routes,
                           &active_experts, &max_routes_per_expert);
    route_index_by_slot->assign((size_t)slots, -1);
    for (int route = 0; route < routes; ++route) {
        const int slot = route_slots[(size_t)route];
        if (slot >= 0 && slot < slots) {
            (*route_index_by_slot)[(size_t)slot] = route;
        }
    }
}

void build_route_indices_by_slot_for_rank(int rank, int slots, int top_k,
                                          std::vector<int> *route_indices,
                                          std::vector<int> *route_counts) {
    std::vector<int> offsets;
    std::vector<int> route_slots;
    int routes = 0;
    int active_experts = 0;
    int max_routes_per_expert = 0;
    build_offsets_for_rank(rank, slots, top_k, &offsets, &route_slots, nullptr, &routes,
                           &active_experts, &max_routes_per_expert);
    route_indices->assign((size_t)slots * (size_t)top_k, -1);
    route_counts->assign((size_t)slots, 0);
    for (int route = 0; route < routes; ++route) {
        const int slot = route_slots[(size_t)route];
        if (slot < 0 || slot >= slots) continue;
        int &count = (*route_counts)[(size_t)slot];
        if (count < top_k) {
            (*route_indices)[(size_t)slot * (size_t)top_k + (size_t)count] = route;
            count++;
        }
    }
}

size_t compact_route_plan_ints(const Options &opt) {
    const size_t indices = (size_t)opt.slots * (size_t)opt.top_k;
    const size_t counts = (size_t)opt.slots;
    return (size_t)kGpus * (indices + counts);
}

void bind_compact_route_plan(RankState *r, const Options &opt) {
    const size_t indices = (size_t)opt.slots * (size_t)opt.top_k;
    const size_t counts = (size_t)opt.slots;
    int *base = r->d_route_compact_plan;
    for (int src = 0; src < kGpus; ++src) {
        r->d_route_indices_by_slot[src] = base + (size_t)src * indices;
        r->d_route_count_by_slot[src] =
            base + (size_t)kGpus * indices + (size_t)src * counts;
    }
}

int upload_model_router_route_plan_gpu(const Options &opt,
                                       SharedHcControls *hc,
                                       RankState ranks[kGpus]) {
    if (!opt.compact_moe_decode_gate || !hc || !hc->d_router_selected ||
        !hc->d_router_weights) {
        return 1;
    }
    const size_t selected_bytes =
        (size_t)opt.slots * (size_t)opt.top_k * sizeof(int);
    const size_t weights_bytes =
        (size_t)opt.slots * (size_t)opt.top_k * sizeof(float);
    const size_t offsets_all_bytes =
        (size_t)kGpus * (size_t)(kLocalExperts + 1) * sizeof(int);
    const int block = 256;
    const uint32_t route_entries = (uint32_t)(opt.slots * opt.top_k);
    void *selected_dsts[kGpus] = {};
    void *weights_dsts[kGpus] = {};
    for (int rank = 0; rank < kGpus; ++rank) {
        selected_dsts[rank] = ranks[rank].d_router_selected_plan;
        weights_dsts[rank] = ranks[rank].d_router_weights_plan;
    }
    if (nccl_broadcast_bytes_from_rank0(
            ranks, hc->d_router_selected, selected_dsts, selected_bytes,
            "router_plan_selected") != 0 ||
        nccl_broadcast_bytes_from_rank0(
            ranks, hc->d_router_weights, weights_dsts, weights_bytes,
            "router_plan_weights") != 0) {
        return 4;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.d_router_selected_plan || !r.d_router_weights_plan ||
            !r.d_route_offsets_all || !r.d_route_totals ||
            !r.d_route_compact_plan) {
            return 2;
        }
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaMemsetAsync(r.d_route_offsets_all, 0,
                                   offsets_all_bytes, r.stream));
        CHECK_CUDA(cudaMemsetAsync(r.d_route_totals, 0,
                                   (size_t)kGpus * sizeof(int), r.stream));
        gpu_route_count_all_kernel<<<
            (unsigned int)((route_entries + block - 1) / block), block,
            0, r.stream>>>(
            r.d_router_selected_plan, r.d_route_offsets_all,
            (uint32_t)opt.slots, (uint32_t)opt.top_k);
        gpu_route_prefix_all_kernel<<<1, kGpus, 0, r.stream>>>(
            r.d_route_offsets_all, r.d_route_totals);
        gpu_route_init_compact_plan_kernel<<<
            (unsigned int)((compact_route_plan_ints(opt) + block - 1) / block),
            block, 0, r.stream>>>(
            r.d_route_compact_plan, (uint32_t)opt.slots, (uint32_t)opt.top_k);
        gpu_route_copy_own_offsets_kernel<<<1, kLocalExperts + 1, 0, r.stream>>>(
            r.d_offsets, r.d_route_offsets_all, (uint32_t)rank);
        gpu_route_fill_all_kernel<<<
            (unsigned int)((route_entries + block - 1) / block), block,
            0, r.stream>>>(
            r.d_router_selected_plan, r.d_router_weights_plan,
            r.d_route_offsets_all, rank, r.d_route_slots, r.d_route_weights,
            r.d_route_compact_plan, (uint32_t)opt.slots, (uint32_t)opt.top_k);
        CHECK_CUDA(cudaGetLastError());
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }
    std::vector<int> totals((size_t)kGpus, 0);
    std::vector<int> offsets_all((size_t)kGpus * (size_t)(kLocalExperts + 1), 0);
    CHECK_CUDA(cudaSetDevice(ranks[0].device));
    CHECK_CUDA(cudaMemcpy(totals.data(), ranks[0].d_route_totals,
                          totals.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(offsets_all.data(), ranks[0].d_route_offsets_all,
                          offsets_all.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    for (int rank = 0; rank < kGpus; ++rank) {
        if (totals[(size_t)rank] > ranks[rank].route_capacity) return 3;
        ranks[rank].routes = totals[(size_t)rank];
        int active = 0;
        int max_routes = 0;
        const int *off = offsets_all.data() + (size_t)rank * (kLocalExperts + 1);
        for (int local = 0; local < kLocalExperts; ++local) {
            const int count = off[local + 1] - off[local];
            if (count > 0) ++active;
            max_routes = std::max(max_routes, count);
        }
        ranks[rank].active_experts = active;
        ranks[rank].max_routes_per_expert = max_routes;
    }
    static bool route_stats_emitted[43] = {};
    if (opt.model_router_routes && opt.layer >= 0 && opt.layer <= 5 &&
        !route_stats_emitted[opt.layer]) {
        route_stats_emitted[opt.layer] = true;
        std::fprintf(stderr,
                     "tp_ep_model_router_route_stats\tlayer\t%d\troutes\t%d,%d,%d,%d,%d,%d,%d,%d\tmax_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                     opt.layer,
                     ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                     ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                     ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                     ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                     ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                     ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
    }
    int duplicate_slots = 0;
    int max_same_rank_routes = 0;
    std::vector<int> compact_counts((size_t)kGpus * (size_t)opt.slots, 0);
    CHECK_CUDA(cudaSetDevice(ranks[0].device));
    CHECK_CUDA(cudaMemcpy(
        compact_counts.data(),
        ranks[0].d_route_compact_plan +
            (size_t)kGpus * (size_t)opt.slots * (size_t)opt.top_k,
        compact_counts.size() * sizeof(int),
        cudaMemcpyDeviceToHost));
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int rank = 0; rank < kGpus; ++rank) {
            const int c = compact_counts[(size_t)rank * (size_t)opt.slots + slot];
            max_same_rank_routes = std::max(max_same_rank_routes, c);
            if (c > 1) duplicate_slots++;
        }
    }
    static bool compact_stats_emitted[43] = {};
    if (opt.layer >= 0 && opt.layer < 43 && !compact_stats_emitted[opt.layer]) {
        compact_stats_emitted[opt.layer] = true;
        const uint64_t all_dest_bytes =
            (uint64_t)kGpus * (uint64_t)kGpus * (uint64_t)opt.slots *
            (uint64_t)(kHidden / kGpus) * sizeof(float);
        const uint64_t total_routes =
            (uint64_t)ranks[0].routes + (uint64_t)ranks[1].routes +
            (uint64_t)ranks[2].routes + (uint64_t)ranks[3].routes +
            (uint64_t)ranks[4].routes + (uint64_t)ranks[5].routes +
            (uint64_t)ranks[6].routes + (uint64_t)ranks[7].routes;
        const uint64_t compact_bytes =
            (uint64_t)kGpus * total_routes * (uint64_t)(kHidden / kGpus) *
            sizeof(float);
        std::printf("tp_ep_compact_moe_route_stats\tlayer\t%d\t"
                    "duplicate_slots\t%d\tmax_same_rank_routes\t%d\t"
                    "all_dest_bytes\t%llu\tcompact_bytes\t%llu\t"
                    "routes\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                    "active_experts\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                    "max_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                    opt.layer, duplicate_slots, max_same_rank_routes,
                    (unsigned long long)all_dest_bytes,
                    (unsigned long long)compact_bytes,
                    ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                    ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                    ranks[0].active_experts, ranks[1].active_experts,
                    ranks[2].active_experts, ranks[3].active_experts,
                    ranks[4].active_experts, ranks[5].active_experts,
                    ranks[6].active_experts, ranks[7].active_experts,
                    ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                    ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                    ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                    ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
    }
    return 0;
}

int upload_post_attention_fixed_capacity_route_plan_gpu(
    const Options &opt,
    SharedHcControls *hc,
    RankState ranks[kGpus],
    cudaStream_t control_stream,
    bool graph_event_order) {
    if (!opt.compact_moe_decode_gate || !hc || !hc->d_router_selected ||
        !hc->d_router_weights) {
        return 1;
    }
    if (opt.post_attention_device_actual_route_sync_gate && graph_event_order) {
        return 7;
    }
    const int block = 256;
    const uint32_t route_entries = (uint32_t)(opt.slots * opt.top_k);
    const size_t selected_bytes = (size_t)route_entries * sizeof(int);
    const size_t weights_bytes = (size_t)route_entries * sizeof(float);
    const size_t offsets_all_bytes =
        (size_t)kGpus * (size_t)(kLocalExperts + 1) * sizeof(int);
    if (!graph_event_order) {
        void *selected_dsts[kGpus] = {};
        void *weights_dsts[kGpus] = {};
        for (int rank = 0; rank < kGpus; ++rank) {
            selected_dsts[rank] = ranks[rank].d_router_selected_plan;
            weights_dsts[rank] = ranks[rank].d_router_weights_plan;
        }
        if (nccl_broadcast_bytes_from_rank0(
                ranks, hc->d_router_selected, selected_dsts, selected_bytes,
                "post_attention_route_selected") != 0 ||
            nccl_broadcast_bytes_from_rank0(
                ranks, hc->d_router_weights, weights_dsts, weights_bytes,
                "post_attention_route_weights") != 0) {
            return 8;
        }
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.d_router_selected_plan || !r.d_router_weights_plan ||
            !r.d_route_offsets_all || !r.d_route_totals ||
            !r.d_route_compact_plan || !r.d_offsets ||
            !r.d_route_slots || !r.d_route_weights) {
            return 2;
        }
        r.routes = r.route_capacity;
        if (opt.post_attention_static_rank_route_cap > 0) {
            r.routes = std::min(r.routes,
                                opt.post_attention_static_rank_route_cap);
        }
        r.active_experts = kLocalExperts;
        r.max_routes_per_expert = r.routes;
        CHECK_CUDA(cudaSetDevice(r.device));
        if (graph_event_order) {
            enqueue_graph_i32_copy_from_device0(
                opt, r, rank, r.d_router_selected_plan,
                hc->d_router_selected, route_entries, r.stream, block);
            enqueue_graph_f32_copy_from_device0(
                opt, r, rank, r.d_router_weights_plan,
                hc->d_router_weights, route_entries, r.stream, block);
        }
        CHECK_CUDA(cudaMemsetAsync(r.d_route_offsets_all, 0,
                                   offsets_all_bytes, r.stream));
        CHECK_CUDA(cudaMemsetAsync(r.d_route_totals, 0,
                                   (size_t)kGpus * sizeof(int), r.stream));
        CHECK_CUDA(cudaMemsetAsync(r.d_route_slots, 0,
                                   (size_t)r.route_capacity * sizeof(int),
                                   r.stream));
        CHECK_CUDA(cudaMemsetAsync(r.d_route_weights, 0,
                                   (size_t)r.route_capacity * sizeof(float),
                                   r.stream));
        gpu_route_count_all_kernel<<<
            (unsigned int)((route_entries + block - 1) / block), block,
            0, r.stream>>>(
            r.d_router_selected_plan, r.d_route_offsets_all,
            (uint32_t)opt.slots, (uint32_t)opt.top_k);
        gpu_route_prefix_all_kernel<<<1, kGpus, 0, r.stream>>>(
            r.d_route_offsets_all, r.d_route_totals);
        gpu_route_init_compact_plan_kernel<<<
            (unsigned int)((compact_route_plan_ints(opt) + block - 1) / block),
            block, 0, r.stream>>>(
            r.d_route_compact_plan, (uint32_t)opt.slots, (uint32_t)opt.top_k);
        gpu_route_copy_own_offsets_kernel<<<1, kLocalExperts + 1, 0, r.stream>>>(
            r.d_offsets, r.d_route_offsets_all, (uint32_t)rank);
        gpu_route_fill_all_kernel<<<
            (unsigned int)((route_entries + block - 1) / block), block,
            0, r.stream>>>(
            r.d_router_selected_plan, r.d_router_weights_plan,
            r.d_route_offsets_all, rank, r.d_route_slots, r.d_route_weights,
            r.d_route_compact_plan, (uint32_t)opt.slots, (uint32_t)opt.top_k);
        if (opt.post_attention_route_reuse_audit_gate &&
            r.d_post_attn_route_audit) {
            CHECK_CUDA(cudaMemsetAsync(r.d_post_attn_route_audit, 0,
                                       4u * sizeof(unsigned long long),
                                       r.stream));
            post_attention_route_plan_audit_kernel<<<
                (unsigned int)kLocalExperts, 128, 0, r.stream>>>(
                r.d_post_attn_route_audit, r.d_offsets, r.d_route_slots,
                r.d_route_weights, r.d_router_selected_plan,
                r.d_router_weights_plan, (uint32_t)rank,
                (uint32_t)opt.slots, (uint32_t)opt.top_k);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    if (opt.post_attention_device_actual_route_sync_gate) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
        std::vector<int> totals((size_t)kGpus, 0);
        std::vector<int> offsets_all((size_t)kGpus * (size_t)(kLocalExperts + 1), 0);
        CHECK_CUDA(cudaSetDevice(ranks[0].device));
        CHECK_CUDA(cudaMemcpy(totals.data(), ranks[0].d_route_totals,
                              totals.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(offsets_all.data(), ranks[0].d_route_offsets_all,
                              offsets_all.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            if (totals[(size_t)rank] > r.route_capacity) return 8;
            r.routes = totals[(size_t)rank];
            int active = 0;
            int max_routes = 0;
            const int *off = offsets_all.data() + (size_t)rank * (kLocalExperts + 1);
            for (int local = 0; local < kLocalExperts; ++local) {
                const int count = off[local + 1] - off[local];
                if (count > 0) ++active;
                max_routes = std::max(max_routes, count);
            }
            r.active_experts = active;
            r.max_routes_per_expert = max_routes;
        }
    }
    (void)control_stream;
    return 0;
}

int upload_model_router_route_plan(const Options &opt,
                                   RankState ranks[kGpus],
                                   const std::vector<int> &selected,
                                   const std::vector<float> &weights) {
    if ((int)selected.size() < opt.slots * opt.top_k ||
        (int)weights.size() < opt.slots * opt.top_k) {
        return 1;
    }
    std::vector<int> offsets[kGpus];
    std::vector<int> route_slots[kGpus];
    std::vector<float> route_weights[kGpus];
    std::vector<int> route_index_by_slot[kGpus];
    std::vector<int> route_indices_by_slot[kGpus];
    std::vector<int> route_count_by_slot[kGpus];
    std::vector<int> counts[kGpus];
    const bool needs_single_route_index = !opt.compact_moe_decode_gate;
    const bool needs_packed_compact_plan = opt.compact_moe_decode_gate;
    for (int rank = 0; rank < kGpus; ++rank) {
        counts[rank].assign((size_t)kLocalExperts, 0);
        if (needs_single_route_index) {
            route_index_by_slot[rank].assign((size_t)opt.slots, -1);
        }
        route_indices_by_slot[rank].assign((size_t)opt.slots * (size_t)opt.top_k,
                                           -1);
        route_count_by_slot[rank].assign((size_t)opt.slots, 0);
    }
    bool compact_duplicate = false;
    for (int slot = 0; slot < opt.slots; ++slot) {
        bool seen_rank[kGpus] = {};
        for (int k = 0; k < opt.top_k; ++k) {
            const int expert = selected[(size_t)slot * opt.top_k + (size_t)k];
            if (expert < 0) continue;
            if (expert < 0 || expert >= kGlobalExperts) return 2;
            const int rank = expert / kLocalExperts;
            const int local = expert % kLocalExperts;
            counts[rank][(size_t)local]++;
            if (seen_rank[rank]) compact_duplicate = true;
            seen_rank[rank] = true;
        }
    }
    if (opt.compact_route_compose && compact_duplicate &&
        !opt.compact_moe_decode_gate) {
        return 3;
    }
    if (opt.routed_ffn_norm_input_gate && opt.layer >= 0 && opt.layer <= 2) {
        for (int slot = 0; slot < opt.slots; ++slot) {
            for (int k = 0; k < opt.top_k; ++k) {
                const int expert = selected[(size_t)slot * opt.top_k + (size_t)k];
                if (expert < 0) continue;
                const int rank = expert / kLocalExperts;
                const int local = expert % kLocalExperts;
                const float w = weights[(size_t)slot * opt.top_k + (size_t)k];
                StridedPtrH gw = {};
                StridedPtrH gs = {};
                StridedPtrH dw = {};
                StridedPtrH ds = {};
                if (rank >= 0 && rank < kGpus) {
                    CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                    if (ranks[rank].gated.d_w_table) {
                        CHECK_CUDA(cudaMemcpy(&gw,
                                              (const StridedPtrH *)ranks[rank].gated.d_w_table + local,
                                              sizeof(gw), cudaMemcpyDeviceToHost));
                    }
                    if (ranks[rank].gated.d_s_table) {
                        CHECK_CUDA(cudaMemcpy(&gs,
                                              (const StridedPtrH *)ranks[rank].gated.d_s_table + local,
                                              sizeof(gs), cudaMemcpyDeviceToHost));
                    }
                    if (ranks[rank].down.d_w_table) {
                        CHECK_CUDA(cudaMemcpy(&dw,
                                              (const StridedPtrH *)ranks[rank].down.d_w_table + local,
                                              sizeof(dw), cudaMemcpyDeviceToHost));
                    }
                    if (ranks[rank].down.d_s_table) {
                        CHECK_CUDA(cudaMemcpy(&ds,
                                              (const StridedPtrH *)ranks[rank].down.d_s_table + local,
                                              sizeof(ds), cudaMemcpyDeviceToHost));
                    }
                }
                std::fprintf(stderr,
                             "tp_ep_model_router_route_id\tlayer\t%d\tslot\t%d\tk\t%d\texpert\t%d\trank\t%d\tlocal\t%d\tweight\t%.9g\tgated_w\t%p\tgated_ws\t%d\tgated_s\t%p\tgated_ss\t%d\tdown_w\t%p\tdown_ws\t%d\tdown_s\t%p\tdown_ss\t%d\n",
                             opt.layer, slot, k, expert, rank, local, w,
                             gw.p, gw.stride, gs.p, gs.stride,
                             dw.p, dw.stride, ds.p, ds.stride);
            }
        }
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        offsets[rank].assign((size_t)kLocalExperts + 1, 0);
        int running = 0;
        int active = 0;
        int max_routes = 0;
        for (int e = 0; e < kLocalExperts; ++e) {
            offsets[rank][(size_t)e] = running;
            running += counts[rank][(size_t)e];
            if (counts[rank][(size_t)e] > 0) ++active;
            max_routes = std::max(max_routes, counts[rank][(size_t)e]);
        }
        offsets[rank][(size_t)kLocalExperts] = running;
        if (running > ranks[rank].route_capacity) return 4;
        route_slots[rank].assign((size_t)running, -1);
        route_weights[rank].assign((size_t)running, 0.0f);
        std::vector<int> cursor = offsets[rank];
        for (int slot = 0; slot < opt.slots; ++slot) {
            for (int k = 0; k < opt.top_k; ++k) {
                const int expert = selected[(size_t)slot * opt.top_k + (size_t)k];
                if (expert < 0) continue;
                const int dst_rank = expert / kLocalExperts;
                if (dst_rank != rank) continue;
                const int local = expert % kLocalExperts;
                const int idx = cursor[(size_t)local]++;
                route_slots[rank][(size_t)idx] = slot;
                const float w = weights[(size_t)slot * opt.top_k + (size_t)k];
                if (!std::isfinite(w)) {
                    std::fprintf(stderr,
                                 "tp_ep_model_router_nonfinite_weight\trank\t%d\tslot\t%d\texpert\t%d\tk\t%d\n",
                                 rank, slot, expert, k);
                    return 5;
                }
                route_weights[rank][(size_t)idx] = w;
                if (needs_single_route_index &&
                    route_index_by_slot[rank][(size_t)slot] < 0) {
                    route_index_by_slot[rank][(size_t)slot] = idx;
                }
                int &route_count = route_count_by_slot[rank][(size_t)slot];
                if (route_count >= opt.top_k) return 6;
                route_indices_by_slot[rank][(size_t)slot * (size_t)opt.top_k +
                                            (size_t)route_count] = idx;
                route_count++;
            }
        }
        RankState &r = ranks[rank];
        r.routes = running;
        r.active_experts = active;
        r.max_routes_per_expert = max_routes;
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaMemcpy(r.d_offsets, offsets[rank].data(),
                              offsets[rank].size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(r.d_route_slots, route_slots[rank].data(),
                              route_slots[rank].size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(r.d_route_weights, route_weights[rank].data(),
                              route_weights[rank].size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    static bool route_stats_emitted[43] = {};
    if (opt.model_router_routes && opt.layer >= 0 && opt.layer <= 5 &&
        !route_stats_emitted[opt.layer]) {
        route_stats_emitted[opt.layer] = true;
        std::fprintf(stderr,
                     "tp_ep_model_router_route_stats\tlayer\t%d\troutes\t%d,%d,%d,%d,%d,%d,%d,%d\tmax_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                     opt.layer,
                     ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                     ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                     ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                     ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                     ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                     ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
    }
    if (opt.compact_moe_decode_gate && opt.model_router_routes) {
        int duplicate_slots = 0;
        int max_same_rank_routes = 0;
        for (int slot = 0; slot < opt.slots; ++slot) {
            for (int rank = 0; rank < kGpus; ++rank) {
                const int c = route_count_by_slot[rank][(size_t)slot];
                max_same_rank_routes = std::max(max_same_rank_routes, c);
                if (c > 1) duplicate_slots++;
            }
        }
        static bool compact_stats_emitted[43] = {};
        if (opt.layer >= 0 && opt.layer < 43 && !compact_stats_emitted[opt.layer]) {
            compact_stats_emitted[opt.layer] = true;
            const uint64_t all_dest_bytes =
                (uint64_t)kGpus * (uint64_t)kGpus * (uint64_t)opt.slots *
                (uint64_t)(kHidden / kGpus) * sizeof(float);
            const uint64_t total_routes =
                (uint64_t)ranks[0].routes + (uint64_t)ranks[1].routes +
                (uint64_t)ranks[2].routes + (uint64_t)ranks[3].routes +
                (uint64_t)ranks[4].routes + (uint64_t)ranks[5].routes +
                (uint64_t)ranks[6].routes + (uint64_t)ranks[7].routes;
            const uint64_t compact_bytes =
                (uint64_t)kGpus * total_routes * (uint64_t)(kHidden / kGpus) *
                sizeof(float);
            std::printf("tp_ep_compact_moe_route_stats\tlayer\t%d\t"
                        "duplicate_slots\t%d\tmax_same_rank_routes\t%d\t"
                        "all_dest_bytes\t%llu\tcompact_bytes\t%llu\t"
                        "routes\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                        "active_experts\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                        "max_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                        opt.layer, duplicate_slots, max_same_rank_routes,
                        (unsigned long long)all_dest_bytes,
                        (unsigned long long)compact_bytes,
                        ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                        ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                        ranks[0].active_experts, ranks[1].active_experts,
                        ranks[2].active_experts, ranks[3].active_experts,
                        ranks[4].active_experts, ranks[5].active_experts,
                        ranks[6].active_experts, ranks[7].active_experts,
                        ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                        ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                        ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                        ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
        }
    }
    std::vector<int> compact_plan;
    if (needs_packed_compact_plan) {
        compact_plan.assign(compact_route_plan_ints(opt), -1);
        const size_t compact_indices = (size_t)opt.slots * (size_t)opt.top_k;
        const size_t compact_counts = (size_t)opt.slots;
        for (int src = 0; src < kGpus; ++src) {
            std::copy(route_indices_by_slot[src].begin(),
                      route_indices_by_slot[src].end(),
                      compact_plan.begin() + (size_t)src * compact_indices);
            std::copy(route_count_by_slot[src].begin(),
                      route_count_by_slot[src].end(),
                      compact_plan.begin() + (size_t)kGpus * compact_indices +
                          (size_t)src * compact_counts);
        }
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        CHECK_CUDA(cudaSetDevice(ranks[dst].device));
        for (int src = 0; src < kGpus; ++src) {
            if (needs_single_route_index) {
                CHECK_CUDA(cudaMemcpy(ranks[dst].d_route_index_by_slot[src],
                                      route_index_by_slot[src].data(),
                                      route_index_by_slot[src].size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
            }
            if (!needs_packed_compact_plan) {
                CHECK_CUDA(cudaMemcpy(ranks[dst].d_route_indices_by_slot[src],
                                      route_indices_by_slot[src].data(),
                                      route_indices_by_slot[src].size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(ranks[dst].d_route_count_by_slot[src],
                                      route_count_by_slot[src].data(),
                                      route_count_by_slot[src].size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
            }
        }
        if (needs_packed_compact_plan) {
            if (!ranks[dst].d_route_compact_plan ||
                ranks[dst].route_compact_plan_ints < compact_plan.size()) {
                return 7;
            }
            CHECK_CUDA(cudaMemcpy(ranks[dst].d_route_compact_plan,
                                  compact_plan.data(),
                                  compact_plan.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }
    }
    return 0;
}

int upload_model_router_route_plan_async(const Options &opt,
                                         RankState ranks[kGpus],
                                         const int *selected,
                                         const float *weights,
                                         RoutePlanHostWorkspace *ws) {
    if (!selected || !weights || !ws || !ws->initialized ||
        ws->slots != opt.slots || ws->top_k != opt.top_k ||
        ws->route_capacity < (size_t)opt.slots * (size_t)opt.top_k) {
        return 1;
    }
    if (opt.routed_ffn_norm_input_gate) {
        return 8;
    }
    if (ws->uploads_pending) {
        for (int rank = 0; rank < kGpus; ++rank) {
            if (ws->upload_done[rank]) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaEventSynchronize(ws->upload_done[rank]));
            }
        }
        ws->uploads_pending = false;
    }

    const bool needs_single_route_index = !opt.compact_moe_decode_gate;
    const bool needs_packed_compact_plan = opt.compact_moe_decode_gate;
    std::vector<int> counts[kGpus];
    std::vector<int> cursor[kGpus];
    for (int rank = 0; rank < kGpus; ++rank) {
        counts[rank].assign((size_t)kLocalExperts, 0);
        std::fill(ws->h_route_indices_by_slot[rank],
                  ws->h_route_indices_by_slot[rank] +
                      (size_t)opt.slots * (size_t)opt.top_k,
                  -1);
        std::fill(ws->h_route_count_by_slot[rank],
                  ws->h_route_count_by_slot[rank] + (size_t)opt.slots,
                  0);
        if (needs_single_route_index) {
            std::fill(ws->h_route_index_by_slot[rank],
                      ws->h_route_index_by_slot[rank] + (size_t)opt.slots,
                      -1);
        }
    }

    bool compact_duplicate = false;
    for (int slot = 0; slot < opt.slots; ++slot) {
        bool seen_rank[kGpus] = {};
        for (int k = 0; k < opt.top_k; ++k) {
            const int expert = selected[(size_t)slot * (size_t)opt.top_k + (size_t)k];
            if (expert < 0) continue;
            if (expert >= kGlobalExperts) return 2;
            const int rank = expert / kLocalExperts;
            const int local = expert % kLocalExperts;
            counts[rank][(size_t)local]++;
            if (seen_rank[rank]) compact_duplicate = true;
            seen_rank[rank] = true;
        }
    }
    if (opt.compact_route_compose && compact_duplicate &&
        !opt.compact_moe_decode_gate) {
        return 3;
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        int running = 0;
        int active = 0;
        int max_routes = 0;
        for (int e = 0; e < kLocalExperts; ++e) {
            ws->h_offsets[rank][e] = running;
            running += counts[rank][(size_t)e];
            if (counts[rank][(size_t)e] > 0) ++active;
            max_routes = std::max(max_routes, counts[rank][(size_t)e]);
        }
        ws->h_offsets[rank][kLocalExperts] = running;
        if (running > ranks[rank].route_capacity ||
            (size_t)running > ws->route_capacity) {
            return 4;
        }
        std::fill(ws->h_route_slots[rank],
                  ws->h_route_slots[rank] + (size_t)running, -1);
        std::fill(ws->h_route_weights[rank],
                  ws->h_route_weights[rank] + (size_t)running, 0.0f);
        cursor[rank].assign(ws->h_offsets[rank],
                            ws->h_offsets[rank] + kLocalExperts + 1);
    }

    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int k = 0; k < opt.top_k; ++k) {
            const size_t route_key = (size_t)slot * (size_t)opt.top_k + (size_t)k;
            const int expert = selected[route_key];
            if (expert < 0) continue;
            const int rank = expert / kLocalExperts;
            const int local = expert % kLocalExperts;
            const int idx = cursor[rank][(size_t)local]++;
            ws->h_route_slots[rank][idx] = slot;
            const float w = weights[route_key];
            if (!std::isfinite(w)) {
                std::fprintf(stderr,
                             "tp_ep_model_router_nonfinite_weight\trank\t%d\tslot\t%d\texpert\t%d\tk\t%d\n",
                             rank, slot, expert, k);
                return 5;
            }
            ws->h_route_weights[rank][idx] = w;
            if (needs_single_route_index &&
                ws->h_route_index_by_slot[rank][slot] < 0) {
                ws->h_route_index_by_slot[rank][slot] = idx;
            }
            int &route_count = ws->h_route_count_by_slot[rank][slot];
            if (route_count >= opt.top_k) return 6;
            ws->h_route_indices_by_slot[rank]
                [(size_t)slot * (size_t)opt.top_k + (size_t)route_count] = idx;
            route_count++;
        }
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        const int running = ws->h_offsets[rank][kLocalExperts];
        int active = 0;
        int max_routes = 0;
        for (int e = 0; e < kLocalExperts; ++e) {
            if (counts[rank][(size_t)e] > 0) ++active;
            max_routes = std::max(max_routes, counts[rank][(size_t)e]);
        }
        r.routes = running;
        r.active_experts = active;
        r.max_routes_per_expert = max_routes;
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaMemcpyAsync(r.d_offsets, ws->h_offsets[rank],
                                   (size_t)(kLocalExperts + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, r.stream));
        if (running > 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_route_slots, ws->h_route_slots[rank],
                                       (size_t)running * sizeof(int),
                                       cudaMemcpyHostToDevice, r.stream));
            CHECK_CUDA(cudaMemcpyAsync(r.d_route_weights, ws->h_route_weights[rank],
                                       (size_t)running * sizeof(float),
                                       cudaMemcpyHostToDevice, r.stream));
        }
    }

    static bool route_stats_emitted[43] = {};
    if (opt.model_router_routes && opt.layer >= 0 && opt.layer <= 5 &&
        !route_stats_emitted[opt.layer]) {
        route_stats_emitted[opt.layer] = true;
        std::fprintf(stderr,
                     "tp_ep_model_router_route_stats_async\tlayer\t%d\troutes\t%d,%d,%d,%d,%d,%d,%d,%d\tmax_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                     opt.layer,
                     ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                     ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                     ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                     ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                     ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                     ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
    }
    if (opt.compact_moe_decode_gate && opt.model_router_routes) {
        int duplicate_slots = 0;
        int max_same_rank_routes = 0;
        for (int slot = 0; slot < opt.slots; ++slot) {
            for (int rank = 0; rank < kGpus; ++rank) {
                const int c = ws->h_route_count_by_slot[rank][slot];
                max_same_rank_routes = std::max(max_same_rank_routes, c);
                if (c > 1) duplicate_slots++;
            }
        }
        static bool compact_stats_emitted[43] = {};
        if (opt.layer >= 0 && opt.layer < 43 && !compact_stats_emitted[opt.layer]) {
            compact_stats_emitted[opt.layer] = true;
            const uint64_t all_dest_bytes =
                (uint64_t)kGpus * (uint64_t)kGpus * (uint64_t)opt.slots *
                (uint64_t)(kHidden / kGpus) * sizeof(float);
            const uint64_t total_routes =
                (uint64_t)ranks[0].routes + (uint64_t)ranks[1].routes +
                (uint64_t)ranks[2].routes + (uint64_t)ranks[3].routes +
                (uint64_t)ranks[4].routes + (uint64_t)ranks[5].routes +
                (uint64_t)ranks[6].routes + (uint64_t)ranks[7].routes;
            const uint64_t compact_bytes =
                (uint64_t)kGpus * total_routes * (uint64_t)(kHidden / kGpus) *
                sizeof(float);
            std::printf("tp_ep_compact_moe_route_stats_async\tlayer\t%d\t"
                        "duplicate_slots\t%d\tmax_same_rank_routes\t%d\t"
                        "all_dest_bytes\t%llu\tcompact_bytes\t%llu\t"
                        "routes\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                        "active_experts\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                        "max_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                        opt.layer, duplicate_slots, max_same_rank_routes,
                        (unsigned long long)all_dest_bytes,
                        (unsigned long long)compact_bytes,
                        ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                        ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                        ranks[0].active_experts, ranks[1].active_experts,
                        ranks[2].active_experts, ranks[3].active_experts,
                        ranks[4].active_experts, ranks[5].active_experts,
                        ranks[6].active_experts, ranks[7].active_experts,
                        ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                        ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                        ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                        ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
        }
    }

    if (needs_packed_compact_plan) {
        if (ws->compact_plan_ints <
            (size_t)kGpus * ((size_t)opt.slots * (size_t)opt.top_k +
                             (size_t)opt.slots)) {
            return 7;
        }
        std::fill(ws->h_compact_plan,
                  ws->h_compact_plan + ws->compact_plan_ints, -1);
        const size_t compact_indices = (size_t)opt.slots * (size_t)opt.top_k;
        const size_t compact_counts = (size_t)opt.slots;
        for (int src = 0; src < kGpus; ++src) {
            std::memcpy(ws->h_compact_plan + (size_t)src * compact_indices,
                        ws->h_route_indices_by_slot[src],
                        compact_indices * sizeof(int));
            std::memcpy(ws->h_compact_plan + (size_t)kGpus * compact_indices +
                            (size_t)src * compact_counts,
                        ws->h_route_count_by_slot[src],
                        compact_counts * sizeof(int));
        }
    }

    for (int dst = 0; dst < kGpus; ++dst) {
        CHECK_CUDA(cudaSetDevice(ranks[dst].device));
        for (int src = 0; src < kGpus; ++src) {
            if (needs_single_route_index) {
                CHECK_CUDA(cudaMemcpyAsync(ranks[dst].d_route_index_by_slot[src],
                                           ws->h_route_index_by_slot[src],
                                           (size_t)opt.slots * sizeof(int),
                                           cudaMemcpyHostToDevice,
                                           ranks[dst].stream));
            }
            if (!needs_packed_compact_plan) {
                CHECK_CUDA(cudaMemcpyAsync(ranks[dst].d_route_indices_by_slot[src],
                                           ws->h_route_indices_by_slot[src],
                                           (size_t)opt.slots *
                                               (size_t)opt.top_k * sizeof(int),
                                           cudaMemcpyHostToDevice,
                                           ranks[dst].stream));
                CHECK_CUDA(cudaMemcpyAsync(ranks[dst].d_route_count_by_slot[src],
                                           ws->h_route_count_by_slot[src],
                                           (size_t)opt.slots * sizeof(int),
                                           cudaMemcpyHostToDevice,
                                           ranks[dst].stream));
            }
        }
        if (needs_packed_compact_plan) {
            if (!ranks[dst].d_route_compact_plan ||
                ranks[dst].route_compact_plan_ints < ws->compact_plan_ints) {
                return 7;
            }
            CHECK_CUDA(cudaMemcpyAsync(ranks[dst].d_route_compact_plan,
                                       ws->h_compact_plan,
                                       ws->compact_plan_ints * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       ranks[dst].stream));
        }
        CHECK_CUDA(cudaEventRecord(ws->upload_done[dst], ranks[dst].stream));
    }
    ws->uploads_pending = true;
    return 0;
}

void print_post_attention_route_reuse_audit(const Options &opt,
                                            RankState ranks[kGpus],
                                            const char *label) {
    if (!opt.post_attention_route_reuse_audit_gate) return;
    unsigned long long total[4] = {};
    int cap_overflow = 0;
    int cap_max_total = 0;
    int compose_cap_overflow = 0;
    int compose_cap_max_total = 0;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.d_post_attn_route_audit) continue;
        unsigned long long h[4] = {};
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaMemcpy(h, r.d_post_attn_route_audit,
                              sizeof(h), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 4; ++i) total[i] += h[i];
        std::printf("tp_ep_post_attention_route_reuse_audit\tlabel\t%s\t"
                    "layer\t%d\trank\t%d\troutes_checked\t%llu\t"
                    "missing_selected\t%llu\tweight_mismatch\t%llu\t"
                    "invalid_slot\t%llu\n",
                    label ? label : "unknown", opt.layer, rank,
                    h[0], h[1], h[2], h[3]);
        if (opt.post_attention_static_rank_route_cap > 0 &&
            r.d_route_totals) {
            int route_total = 0;
            CHECK_CUDA(cudaMemcpy(&route_total, r.d_route_totals + rank,
                                  sizeof(route_total),
                                  cudaMemcpyDeviceToHost));
            cap_max_total = std::max(cap_max_total, route_total);
            if (route_total > opt.post_attention_static_rank_route_cap) {
                ++cap_overflow;
            }
            std::printf("tp_ep_static_route_cap_audit\tlabel\t%s\t"
                        "layer\t%d\trank\t%d\tcap\t%d\tactual\t%d\t%s\n",
                        label ? label : "unknown", opt.layer, rank,
                        opt.post_attention_static_rank_route_cap, route_total,
                        route_total <= opt.post_attention_static_rank_route_cap
                            ? "PASS" : "OVERFLOW");
        }
        if (opt.post_attention_static_compose_route_cap > 0 &&
            r.d_route_totals) {
            int route_total = 0;
            CHECK_CUDA(cudaMemcpy(&route_total, r.d_route_totals + rank,
                                  sizeof(route_total),
                                  cudaMemcpyDeviceToHost));
            compose_cap_max_total = std::max(compose_cap_max_total, route_total);
            if (route_total > opt.post_attention_static_compose_route_cap) {
                ++compose_cap_overflow;
            }
            std::printf("tp_ep_static_compose_route_cap_audit\tlabel\t%s\t"
                        "layer\t%d\trank\t%d\tcap\t%d\tactual\t%d\t%s\n",
                        label ? label : "unknown", opt.layer, rank,
                        opt.post_attention_static_compose_route_cap,
                        route_total,
                        route_total <= opt.post_attention_static_compose_route_cap
                            ? "PASS" : "OVERFLOW");
        }
    }
    std::printf("tp_ep_post_attention_route_reuse_audit_total\tlabel\t%s\t"
                "layer\t%d\troutes_checked\t%llu\tmissing_selected\t%llu\t"
                "weight_mismatch\t%llu\tinvalid_slot\t%llu\n",
                label ? label : "unknown", opt.layer,
                total[0], total[1], total[2], total[3]);
    if (opt.post_attention_static_rank_route_cap > 0) {
        std::printf("tp_ep_static_route_cap_audit_total\tlabel\t%s\t"
                    "layer\t%d\tcap\t%d\tmax_actual\t%d\toverflow_ranks\t%d\t%s\n",
                    label ? label : "unknown", opt.layer,
                    opt.post_attention_static_rank_route_cap, cap_max_total,
                    cap_overflow, cap_overflow == 0 ? "PASS" : "OVERFLOW");
    }
    if (opt.post_attention_static_compose_route_cap > 0) {
        std::printf("tp_ep_static_compose_route_cap_audit_total\tlabel\t%s\t"
                    "layer\t%d\tcap\t%d\tmax_actual\t%d\toverflow_ranks\t%d\t%s\n",
                    label ? label : "unknown", opt.layer,
                    opt.post_attention_static_compose_route_cap,
                    compose_cap_max_total, compose_cap_overflow,
                    compose_cap_overflow == 0 ? "PASS" : "OVERFLOW");
    }
}

int open_compose_nccl(const Options &opt, RankState ranks[kGpus]);
void close_compose_nccl(RankState ranks[kGpus]);

int open_shared_rank_buffers(const Options &opt, SharedRankBuffers *shared) {
    shared->core_bytes = 0;
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = shared->ranks[p];
        r.rank = p;
        r.device = opt.devices[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaStreamCreate(&r.stream));
        CHECK_CUDA(cudaStreamCreate(&r.dense_stream));
        CHECK_CUDA(cudaStreamCreate(&r.copy_stream));
        for (int q = 0; q < kGpus; ++q) {
            CHECK_CUDA(cudaStreamCreate(&r.copy_streams[q]));
            CHECK_CUDA(cudaEventCreateWithFlags(&r.copy_done[q], cudaEventDisableTiming));
        }
        CHECK_CUDA(cudaEventCreateWithFlags(&r.stream_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&r.dense_done, cudaEventDisableTiming));
        for (int e = 0; e < kGraphOrderEventSlots; ++e) {
            CHECK_CUDA(cudaEventCreateWithFlags(&r.graph_stream_done[e],
                                                cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&r.graph_dense_done[e],
                                                cudaEventDisableTiming));
        }
        CHECK_CUDA(cudaEventCreateWithFlags(&r.dense_wait, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreate(&r.start));
        CHECK_CUDA(cudaEventCreate(&r.mid));
        CHECK_CUDA(cudaEventCreate(&r.stop));
        r.route_compact_plan_ints = compact_route_plan_ints(opt);
        CHECK_CUDA(cudaMalloc(&r.d_route_compact_plan,
                              r.route_compact_plan_ints * sizeof(int)));
        bind_compact_route_plan(&r, opt);
        CHECK_CUDA(cudaMalloc(&r.d_router_selected_plan,
                              (size_t)opt.slots * (size_t)opt.top_k * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&r.d_router_weights_plan,
                              (size_t)opt.slots * (size_t)opt.top_k * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&r.d_route_offsets_all,
                              (size_t)kGpus * (size_t)(kLocalExperts + 1) *
                                  sizeof(int)));
        CHECK_CUDA(cudaMalloc(&r.d_route_totals,
                              (size_t)kGpus * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&r.d_post_attn_route_audit,
                              4u * sizeof(unsigned long long)));
        CHECK_CUDA(cudaMemset(r.d_post_attn_route_audit, 0,
                              4u * sizeof(unsigned long long)));
        std::vector<int> compact_plan(r.route_compact_plan_ints, -1);
        const size_t compact_indices = (size_t)opt.slots * (size_t)opt.top_k;
        const size_t compact_counts = (size_t)opt.slots;
        for (int src = 0; src < kGpus; ++src) {
            std::vector<int> route_index_by_slot;
            build_route_index_by_slot_for_rank(src, opt.slots, opt.top_k,
                                               &route_index_by_slot);
            CHECK_CUDA(cudaMalloc(&r.d_route_index_by_slot[src],
                                  route_index_by_slot.size() * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(r.d_route_index_by_slot[src],
                                  route_index_by_slot.data(),
                                  route_index_by_slot.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            std::vector<int> route_indices_by_slot;
            std::vector<int> route_count_by_slot;
            build_route_indices_by_slot_for_rank(src, opt.slots, opt.top_k,
                                                 &route_indices_by_slot,
                                                 &route_count_by_slot);
            std::copy(route_indices_by_slot.begin(), route_indices_by_slot.end(),
                      compact_plan.begin() + (size_t)src * compact_indices);
            std::copy(route_count_by_slot.begin(), route_count_by_slot.end(),
                      compact_plan.begin() + (size_t)kGpus * compact_indices +
                          (size_t)src * compact_counts);
            shared->core_bytes += route_index_by_slot.size() * sizeof(int);
        }
        CHECK_CUDA(cudaMemcpy(r.d_route_compact_plan, compact_plan.data(),
                              compact_plan.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        shared->core_bytes += compact_plan.size() * sizeof(int);
        shared->core_bytes += 4u * sizeof(unsigned long long);

        std::vector<int> offsets;
        std::vector<int> route_slots;
        std::vector<float> route_weights;
        build_offsets_for_rank(p, opt.slots, opt.top_k, &offsets, &route_slots,
                               &route_weights, &r.routes, &r.active_experts,
                               &r.max_routes_per_expert);

        r.route_capacity = opt.slots * opt.top_k;
        const size_t route_capacity_elems = (size_t)r.route_capacity * kHidden;
        CHECK_CUDA(cudaMalloc(&r.d_offsets, offsets.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(r.d_offsets, offsets.data(), offsets.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_route_slots,
                              (size_t)r.route_capacity * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(r.d_route_slots, route_slots.data(),
                              route_slots.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_route_weights,
                              (size_t)r.route_capacity * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(r.d_route_weights, route_weights.data(),
                              route_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_route_inv_scale,
                              (size_t)r.route_capacity * sizeof(float)));
        std::vector<float> route_inv_scale((size_t)r.route_capacity, 1.0f);
        CHECK_CUDA(cudaMemcpy(r.d_route_inv_scale, route_inv_scale.data(),
                              route_inv_scale.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_a, route_capacity_elems * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&r.d_gate_up,
                              (size_t)r.route_capacity * kFusedN * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&r.d_gated,
                              (size_t)r.route_capacity * kMid * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&r.d_down, route_capacity_elems * sizeof(__half)));
        if (opt.model_router_rank_major_logits_gate ||
            opt.model_router_allreduce_logits_gate) {
            if (opt.model_router_rank_major_logits_gate) {
                CHECK_CUDA(cudaMalloc(&r.d_rank_major_norm_scale,
                                      (size_t)opt.slots * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_router_logits_shard,
                                      (size_t)opt.slots * kLocalExperts *
                                          sizeof(float)));
            }
            CHECK_CUDA(cudaMalloc(&r.d_router_logits_rank_major,
                                  (size_t)opt.slots * kGlobalExperts * sizeof(float)));
        }

        std::mt19937 rng(0xE2350000u + (uint32_t)p * 97u);
        std::uniform_real_distribution<float> dist(-0.003f, 0.003f);
        std::vector<__half> h_a(route_capacity_elems);
        for (__half &v : h_a) v = __float2half(dist(rng));
        CHECK_CUDA(cudaMemcpy(r.d_a, h_a.data(),
                              route_capacity_elems * sizeof(__half),
                              cudaMemcpyHostToDevice));

        shared->core_bytes += offsets.size() * sizeof(int);
        shared->core_bytes += (size_t)r.route_capacity * sizeof(int);
        shared->core_bytes += (size_t)r.route_capacity * sizeof(float);
        shared->core_bytes += (size_t)r.route_capacity * sizeof(float);
        shared->core_bytes += route_capacity_elems * sizeof(__half);
        shared->core_bytes += (size_t)r.route_capacity * kFusedN * sizeof(__half);
        shared->core_bytes += (size_t)r.route_capacity * kMid * sizeof(__half);
        shared->core_bytes += route_capacity_elems * sizeof(__half);
        if (opt.model_router_rank_major_logits_gate) {
            shared->core_bytes += (size_t)opt.slots * sizeof(float);
            shared->core_bytes += (size_t)opt.slots * kLocalExperts * sizeof(float);
        }
        if (opt.model_router_rank_major_logits_gate ||
            opt.model_router_allreduce_logits_gate) {
            shared->core_bytes += (size_t)opt.slots * kGlobalExperts * sizeof(float);
        }
    }
    if (!opt.defer_nccl_init_gate && open_compose_nccl(opt, shared->ranks) != 0) {
        return 1;
    }
    shared->initialized = true;
    return 0;
}

int open_compose_nccl(const Options &opt, RankState ranks[kGpus]) {
    const bool need_compose =
        opt.nccl_reduce_scatter_compose_gate &&
        !opt.compact_route_compose && !opt.ep_return_fp16;
    const bool need_attention_output =
        opt.true_ds4_attention_output_nccl_allgather_gate;
    const bool need_hc_current =
        opt.tp_hc_current_input_nccl_allgather_gate ||
        opt.tp_hc_current_allreduce_gate ||
        opt.model_router_allreduce_logits_gate;
    const bool need_full_current_broadcast =
        opt.tp_hc_current_input_gate ||
        opt.true_shared_ffn_gate ||
        opt.true_ds4_attention_projection_gate ||
        opt.true_ds4_compressed_kv_gate ||
        opt.true_ds4_post_attention_ffn_input_gate;
    const bool need_transport_sweep =
        opt.model_router_routes ||
        opt.compact_moe_decode_gate ||
        opt.true_ds4_attention_raw_read_gate ||
        opt.true_ds4_attention_raw_window_gate ||
        opt.true_ds4_attention_typed_kv_compressed_gate;
    if (!need_compose && !need_attention_output && !need_hc_current &&
        !need_full_current_broadcast && !need_transport_sweep) {
        return 0;
    }
    int devices[kGpus] = {};
    ncclComm_t comms[kGpus] = {};
    for (int p = 0; p < kGpus; ++p) devices[p] = ranks[p].device;
    CHECK_NCCL(ncclCommInitAll(comms, kGpus, devices));
    for (int p = 0; p < kGpus; ++p) {
        ranks[p].compose_nccl = comms[p];
        ranks[p].compose_nccl_initialized = true;
    }
    std::printf("tp_ep_nccl\tdevices\t%d\tcompose_reduce_scatter\t%d\t"
                "attention_output_allgather\t%d\t"
                "hc_current_nccl\t%d\tfull_current_broadcast\t%d\t"
                "transport_sweep\t%d\tPASS\n",
                kGpus, need_compose ? 1 : 0, need_attention_output ? 1 : 0,
                need_hc_current ? 1 : 0,
                need_full_current_broadcast ? 1 : 0,
                need_transport_sweep ? 1 : 0);
    return 0;
}

int nccl_broadcast_f32_from_device0_to_current_full(
    const Options &opt,
    RankState ranks[kGpus],
    const float *src_device0,
    uint64_t elems,
    const char *label) {
    if (!src_device0 || elems == 0) return 1;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.compose_nccl_initialized || !r.compose_nccl ||
            !r.d_current_full) {
            std::fprintf(stderr,
                         "tp_ep_full_current_nccl_broadcast_missing\tlabel\t%s\t"
                         "rank\t%d\tcompose\t%d\tbuffer\t%d\n",
                         label ? label : "-", rank,
                         (r.compose_nccl_initialized && r.compose_nccl) ? 1 : 0,
                         r.d_current_full ? 1 : 0);
            return 2;
        }
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        const float *send = rank == 0 ? src_device0 : r.d_current_full;
        CHECK_NCCL(ncclBroadcast(send, r.d_current_full, (size_t)elems,
                                 ncclFloat, 0, r.compose_nccl, r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    return 0;
}

void close_compose_nccl(RankState ranks[kGpus]) {
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        if (!r.compose_nccl_initialized || !r.compose_nccl) continue;
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclCommDestroy(r.compose_nccl));
        r.compose_nccl = nullptr;
        r.compose_nccl_initialized = false;
    }
}

void close_tp_cuda_graph_layer_exec(TpCudaGraphLayerExec *entry) {
    if (!entry) return;
    if (entry->root_device >= 0) {
        CHECK_CUDA(cudaSetDevice(entry->root_device));
    }
    if (entry->exec) CHECK_CUDA(cudaGraphExecDestroy(entry->exec));
    if (entry->graph) CHECK_CUDA(cudaGraphDestroy(entry->graph));
    *entry = TpCudaGraphLayerExec{};
}

void close_tp_cuda_graph_cache(TpCudaGraphCache *cache) {
    if (!cache) return;
    for (int layer = 0; layer < 43; ++layer) {
        close_tp_cuda_graph_layer_exec(&cache->layers[layer]);
    }
}

void close_shared_rank_buffers(SharedRankBuffers *shared) {
    if (!shared || !shared->initialized) return;
    close_tp_cuda_graph_cache(&shared->graph_cache);
    close_compose_nccl(shared->ranks);
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = shared->ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        free_packed(r.gated);
        free_packed(r.down);
        if (r.d_offsets) CHECK_CUDA(cudaFree(r.d_offsets));
        if (r.d_route_slots) CHECK_CUDA(cudaFree(r.d_route_slots));
        if (r.d_route_weights) CHECK_CUDA(cudaFree(r.d_route_weights));
        if (r.d_route_inv_scale) CHECK_CUDA(cudaFree(r.d_route_inv_scale));
        if (r.d_a) CHECK_CUDA(cudaFree(r.d_a));
        if (r.d_gate_up) CHECK_CUDA(cudaFree(r.d_gate_up));
        if (r.d_gated) CHECK_CUDA(cudaFree(r.d_gated));
        if (r.d_down) CHECK_CUDA(cudaFree(r.d_down));
        if (r.d_ep_contrib_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_all));
        if (r.d_ep_contrib_half_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_half_all));
        if (r.d_ep_contrib_bcast_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_bcast_all));
        if (r.d_ep_contrib_half_bcast_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_half_bcast_all));
        if (r.d_route_totals) CHECK_CUDA(cudaFree(r.d_route_totals));
        if (r.d_route_offsets_all) CHECK_CUDA(cudaFree(r.d_route_offsets_all));
        if (r.d_router_weights_plan) CHECK_CUDA(cudaFree(r.d_router_weights_plan));
        if (r.d_router_selected_plan) CHECK_CUDA(cudaFree(r.d_router_selected_plan));
        const bool has_route_compact_plan = r.d_route_compact_plan != nullptr;
        if (r.d_route_compact_plan) CHECK_CUDA(cudaFree(r.d_route_compact_plan));
        for (int src = 0; src < kGpus; ++src) {
            if (r.d_route_index_by_slot[src]) CHECK_CUDA(cudaFree(r.d_route_index_by_slot[src]));
            if (!has_route_compact_plan && r.d_route_indices_by_slot[src]) {
                CHECK_CUDA(cudaFree(r.d_route_indices_by_slot[src]));
            }
            if (!has_route_compact_plan && r.d_route_count_by_slot[src]) {
                CHECK_CUDA(cudaFree(r.d_route_count_by_slot[src]));
            }
            if (r.d_ep_remote[src]) CHECK_CUDA(cudaFree(r.d_ep_remote[src]));
            if (r.d_ep_remote_half[src]) CHECK_CUDA(cudaFree(r.d_ep_remote_half[src]));
        }
        if (r.d_ep_sum) CHECK_CUDA(cudaFree(r.d_ep_sum));
        if (r.d_next_hidden) CHECK_CUDA(cudaFree(r.d_next_hidden));
        if (r.d_current_shard) CHECK_CUDA(cudaFree(r.d_current_shard));
        if (r.d_current_full) CHECK_CUDA(cudaFree(r.d_current_full));
        if (r.d_current_full_normed) CHECK_CUDA(cudaFree(r.d_current_full_normed));
        if (r.d_current_full_rank_major) CHECK_CUDA(cudaFree(r.d_current_full_rank_major));
        if (r.d_post_attn_full_rank_major) CHECK_CUDA(cudaFree(r.d_post_attn_full_rank_major));
        if (r.d_rank_major_norm_scale) CHECK_CUDA(cudaFree(r.d_rank_major_norm_scale));
        if (r.d_router_logits_shard) CHECK_CUDA(cudaFree(r.d_router_logits_shard));
        if (r.d_router_logits_rank_major) CHECK_CUDA(cudaFree(r.d_router_logits_rank_major));
        if (r.d_half_diff_counts) CHECK_CUDA(cudaFree(r.d_half_diff_counts));
        if (r.d_half_diff_max_bits) CHECK_CUDA(cudaFree(r.d_half_diff_max_bits));
        if (r.d_half_diff_first) CHECK_CUDA(cudaFree(r.d_half_diff_first));
        if (r.d_post_attn_route_audit) CHECK_CUDA(cudaFree(r.d_post_attn_route_audit));
        if (r.d_final_hc_shard) CHECK_CUDA(cudaFree(r.d_final_hc_shard));
        if (r.d_hc_scratch_shard) CHECK_CUDA(cudaFree(r.d_hc_scratch_shard));
        if (r.d_hc_split) CHECK_CUDA(cudaFree(r.d_hc_split));
        if (r.d_hc_reduce_max) CHECK_CUDA(cudaFree(r.d_hc_reduce_max));
        if (r.d_hc_reduce_sumsq) CHECK_CUDA(cudaFree(r.d_hc_reduce_sumsq));
        if (r.d_hc_reduce_mix) CHECK_CUDA(cudaFree(r.d_hc_reduce_mix));
        for (int layer = 0; layer < 43; ++layer) {
            if (r.d_attn_raw_swa_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_attn_raw_swa_layers[layer]));
            }
        }
        if (r.d_attn_kv_full) CHECK_CUDA(cudaFree(r.d_attn_kv_full));
        if (r.d_attn_heads) CHECK_CUDA(cudaFree(r.d_attn_heads));
        if (r.d_attn_output_a_full) CHECK_CUDA(cudaFree(r.d_attn_output_a_full));
        if (r.d_post_attn_shard) CHECK_CUDA(cudaFree(r.d_post_attn_shard));
        if (r.d_attn_sinks) CHECK_CUDA(cudaFree(r.d_attn_sinks));
        if (r.d_indexer_topk) CHECK_CUDA(cudaFree(r.d_indexer_topk));
        if (r.d_indexer_scores) CHECK_CUDA(cudaFree(r.d_indexer_scores));
        for (int layer = 0; layer < 43; ++layer) {
            if (r.d_index_comp_rows_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_index_comp_rows_layers[layer]));
            }
            if (r.d_index_comp_state_score_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_index_comp_state_score_layers[layer]));
            }
            if (r.d_index_comp_state_kv_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_index_comp_state_kv_layers[layer]));
            }
        }
        if (r.d_index_comp_score_cur) CHECK_CUDA(cudaFree(r.d_index_comp_score_cur));
        if (r.d_index_comp_kv_cur) CHECK_CUDA(cudaFree(r.d_index_comp_kv_cur));
        for (int layer = 0; layer < 43; ++layer) {
            if (r.d_attn_comp_rows_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_attn_comp_rows_layers[layer]));
            }
            if (r.d_attn_comp_state_score_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_attn_comp_state_score_layers[layer]));
            }
            if (r.d_attn_comp_state_kv_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_attn_comp_state_kv_layers[layer]));
            }
        }
        if (r.d_attn_comp_score_cur) CHECK_CUDA(cudaFree(r.d_attn_comp_score_cur));
        if (r.d_attn_comp_kv_cur) CHECK_CUDA(cudaFree(r.d_attn_comp_kv_cur));
        if (r.dense_wait) CHECK_CUDA(cudaEventDestroy(r.dense_wait));
        if (r.start) CHECK_CUDA(cudaEventDestroy(r.start));
        if (r.mid) CHECK_CUDA(cudaEventDestroy(r.mid));
        if (r.stop) CHECK_CUDA(cudaEventDestroy(r.stop));
        for (int q = 0; q < kGpus; ++q) {
            if (r.copy_done[q]) CHECK_CUDA(cudaEventDestroy(r.copy_done[q]));
            if (r.copy_streams[q]) CHECK_CUDA(cudaStreamDestroy(r.copy_streams[q]));
        }
        for (int e = 0; e < kGraphOrderEventSlots; ++e) {
            if (r.graph_stream_done[e]) {
                CHECK_CUDA(cudaEventDestroy(r.graph_stream_done[e]));
            }
            if (r.graph_dense_done[e]) {
                CHECK_CUDA(cudaEventDestroy(r.graph_dense_done[e]));
            }
        }
        if (r.dense_done) CHECK_CUDA(cudaEventDestroy(r.dense_done));
        if (r.stream_done) CHECK_CUDA(cudaEventDestroy(r.stream_done));
        if (r.copy_stream) CHECK_CUDA(cudaStreamDestroy(r.copy_stream));
        if (r.dense_stream) CHECK_CUDA(cudaStreamDestroy(r.dense_stream));
        if (r.stream) CHECK_CUDA(cudaStreamDestroy(r.stream));
        r = RankState{};
    }
    *shared = SharedRankBuffers{};
}

void fill_tp_runtime_config(const Options &opt, ds4_v100_tp_runtime_config *cfg) {
    ds4_v100_tp_runtime_default_config(cfg);
    cfg->slots = (uint32_t)opt.slots;
    cfg->ctx = 262144;
    cfg->kv_dtype = opt.fp8_e5m2_kv_gate
        ? DS4_V100_TP_KV_F8_E5M2_B128
        : DS4_V100_TP_KV_F8_E4M3_B128;
    cfg->scratch_bytes = opt.tp_runtime_scratch_mib * 1024ull * 1024ull;
    cfg->allocate_comp_state = opt.tp_runtime_skip_unused_comp_state ? 0u : 1u;
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
    const uint64_t compact_segment_routes =
        opt.compact_moe_decode_gate ? (uint64_t)opt.slots * (uint64_t)opt.top_k
                                    : (uint64_t)opt.slots;
    const uint64_t shard_bytes = shard_elems * sizeof(float);
    const uint64_t remote_float_elems =
        opt.compact_route_compose && !opt.ep_return_fp16
            ? compact_segment_routes * (uint64_t)(kHidden / kGpus)
            : shard_elems;
    const uint64_t remote_float_bytes = remote_float_elems * sizeof(float);
    const uint64_t all_contrib_elems =
        (uint64_t)kGpus * compact_segment_routes * (uint64_t)(kHidden / kGpus);
    const uint64_t all_contrib_bytes = all_contrib_elems * sizeof(float);
    const int layer = opt.layer;
    if ((opt.true_ds4_attention_state_gate || opt.true_ds4_compressed_kv_gate ||
         opt.true_ds4_indexer_attention_gate) &&
        (layer < 0 || layer >= 43)) {
        return 20;
    }
    const int ratio = (layer >= 0 && layer < 43) ? ds4_layer_ratio(layer) : 0;
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_ep_contrib_all) CHECK_CUDA(cudaMalloc(&r.d_ep_contrib_all,
                                                        (size_t)all_contrib_bytes));
        if (!r.d_ep_contrib_bcast_all) {
            CHECK_CUDA(cudaMalloc(&r.d_ep_contrib_bcast_all,
                                  (size_t)all_contrib_bytes));
        }
        if (opt.ep_return_fp16 && !r.d_ep_contrib_half_all) {
            CHECK_CUDA(cudaMalloc(&r.d_ep_contrib_half_all,
                                  (size_t)(all_contrib_elems * sizeof(__half))));
        }
        if (opt.ep_return_fp16 && !r.d_ep_contrib_half_bcast_all) {
            CHECK_CUDA(cudaMalloc(&r.d_ep_contrib_half_bcast_all,
                                  (size_t)(all_contrib_elems * sizeof(__half))));
        }
        if (!r.d_ep_sum) CHECK_CUDA(cudaMalloc(&r.d_ep_sum, (size_t)shard_bytes));
        if (!r.d_next_hidden) CHECK_CUDA(cudaMalloc(&r.d_next_hidden, (size_t)shard_bytes));
        if (opt.tp_hc_current_input_gate && !r.d_current_shard) {
            CHECK_CUDA(cudaMalloc(&r.d_current_shard, (size_t)shard_bytes));
        }
        if (opt.tp_hc_current_input_gate && !r.d_current_full) {
            CHECK_CUDA(cudaMalloc(&r.d_current_full,
                                  (size_t)opt.slots * kHidden * sizeof(float)));
        }
        if (opt.true_ds4_attention_projection_rank_local_input_gate &&
            !r.d_current_full_normed) {
            CHECK_CUDA(cudaMalloc(&r.d_current_full_normed,
                                  (size_t)opt.slots * kHidden * sizeof(float)));
        }
        if (opt.tp_hc_current_input_nccl_allgather_gate &&
            !r.d_current_full_rank_major) {
            CHECK_CUDA(cudaMalloc(&r.d_current_full_rank_major,
                                  (size_t)opt.slots * kHidden * sizeof(float)));
        }
        if ((opt.routed_ffn_rank_major_input_gate ||
             opt.routed_ffn_rank_major_shared_input_gate ||
             opt.routed_ffn_rank_major_route_input_gate ||
             opt.routed_ffn_rank_major_input_parity_gate) &&
            !r.d_post_attn_full_rank_major) {
            CHECK_CUDA(cudaMalloc(&r.d_post_attn_full_rank_major,
                                  (size_t)opt.slots * kHidden * sizeof(float)));
        }
        if ((opt.routed_ffn_rank_major_input_parity_gate ||
             opt.true_ds4_attention_projection_input_parity_gate) &&
            !r.d_half_diff_counts) {
            CHECK_CUDA(cudaMalloc(&r.d_half_diff_counts,
                                  2 * sizeof(unsigned long long)));
            CHECK_CUDA(cudaMalloc(&r.d_half_diff_max_bits,
                                  sizeof(unsigned int)));
            CHECK_CUDA(cudaMalloc(&r.d_half_diff_first, sizeof(int)));
        }
        if (opt.final_hc_carry_gate && !r.d_final_hc_shard) {
            CHECK_CUDA(cudaMalloc(&r.d_final_hc_shard, (size_t)(4ull * shard_bytes)));
        }
        if (opt.tp_hc_final_expand_gate && !r.d_hc_scratch_shard) {
            CHECK_CUDA(cudaMalloc(&r.d_hc_scratch_shard, (size_t)(4ull * shard_bytes)));
        }
        if (opt.tp_hc_final_expand_gate && !r.d_hc_split) {
            CHECK_CUDA(cudaMalloc(&r.d_hc_split, (size_t)opt.slots * kHcMix * sizeof(float)));
        }
        if (opt.tp_hc_current_allreduce_gate ||
            opt.model_router_allreduce_logits_gate) {
            if (!r.d_hc_reduce_max) {
                CHECK_CUDA(cudaMalloc(&r.d_hc_reduce_max,
                                      (size_t)opt.slots * sizeof(float)));
            }
            if (!r.d_hc_reduce_sumsq) {
                CHECK_CUDA(cudaMalloc(&r.d_hc_reduce_sumsq,
                                      (size_t)opt.slots * sizeof(float)));
            }
        }
        if (opt.tp_hc_current_allreduce_gate) {
            if (!r.d_hc_reduce_mix) {
                CHECK_CUDA(cudaMalloc(&r.d_hc_reduce_mix,
                                      (size_t)opt.slots * kHcMix * sizeof(float)));
            }
        }
        if (opt.true_ds4_attention_state_gate && !r.d_attn_kv_full) {
            CHECK_CUDA(cudaMalloc(&r.d_attn_kv_full,
                                  (size_t)opt.slots * kHeadDim * sizeof(float)));
        }
        if (opt.true_ds4_attention_state_gate) {
            if (!r.d_attn_raw_swa_layers[layer]) {
                CHECK_CUDA(cudaMalloc(&r.d_attn_raw_swa_layers[layer],
                                      (size_t)opt.slots * kRawSwaRows *
                                          (size_t)kHeadDim * sizeof(float)));
                CHECK_CUDA(cudaMemsetAsync(r.d_attn_raw_swa_layers[layer], 0,
                                           (size_t)opt.slots * kRawSwaRows *
                                               (size_t)kHeadDim * sizeof(float),
                                           r.stream));
            }
            r.d_attn_raw_swa = r.d_attn_raw_swa_layers[layer];
        }
        if (opt.true_ds4_attention_raw_read_gate && !r.d_attn_sinks) {
            CHECK_CUDA(cudaMalloc(&r.d_attn_sinks,
                                  (size_t)kLocalHeads * sizeof(float)));
        }
        if (opt.true_ds4_attention_raw_read_gate && !r.d_attn_heads) {
            CHECK_CUDA(cudaMalloc(&r.d_attn_heads,
                                  (size_t)opt.slots * kLocalHeads *
                                      (size_t)kHeadDim * sizeof(float)));
        }
        if (opt.true_ds4_attention_output_gate && !r.d_attn_output_a_full) {
            CHECK_CUDA(cudaMalloc(&r.d_attn_output_a_full,
                                  (size_t)opt.slots *
                                      (size_t)kAttentionOutputAFull * sizeof(float)));
        }
        if (opt.true_ds4_post_attention_ffn_input_gate && !r.d_post_attn_shard) {
            CHECK_CUDA(cudaMalloc(&r.d_post_attn_shard, (size_t)shard_bytes));
        }
        if (opt.true_ds4_compressed_kv_gate && ratio != 0) {
            const int comp_state_rows = attn_comp_state_rows_for_ratio(ratio);
            const int comp_state_width = attn_comp_state_width_for_ratio(ratio);
            if (!r.d_attn_comp_kv_cur) {
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_kv_cur,
                                      (size_t)opt.slots * kCompWidthMax * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_score_cur,
                                      (size_t)opt.slots * kCompWidthMax * sizeof(float)));
            }
            if (!r.d_attn_comp_state_kv_layers[layer]) {
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_state_kv_layers[layer],
                                      (size_t)opt.slots * (size_t)comp_state_rows *
                                          (size_t)comp_state_width * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_state_score_layers[layer],
                                      (size_t)opt.slots * (size_t)comp_state_rows *
                                          (size_t)comp_state_width * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_rows_layers[layer],
                                      (size_t)opt.slots * kBoundedCompRows *
                                          (size_t)kHeadDim * sizeof(float)));
                CHECK_CUDA(cudaMemsetAsync(r.d_attn_comp_state_kv_layers[layer], 0,
                                           (size_t)opt.slots * (size_t)comp_state_rows *
                                               (size_t)comp_state_width * sizeof(float),
                                           r.stream));
                CHECK_CUDA(cudaMemsetAsync(r.d_attn_comp_state_score_layers[layer], 0,
                                           (size_t)opt.slots * (size_t)comp_state_rows *
                                               (size_t)comp_state_width * sizeof(float),
                                           r.stream));
                CHECK_CUDA(cudaMemsetAsync(r.d_attn_comp_rows_layers[layer], 0,
                                           (size_t)opt.slots * kBoundedCompRows *
                                               (size_t)kHeadDim * sizeof(float),
                                           r.stream));
            }
            r.d_attn_comp_state_kv = r.d_attn_comp_state_kv_layers[layer];
            r.d_attn_comp_state_score = r.d_attn_comp_state_score_layers[layer];
            r.d_attn_comp_rows = r.d_attn_comp_rows_layers[layer];
        }
        if (opt.true_ds4_indexer_attention_gate && ratio == 4) {
            if (!r.d_index_comp_kv_cur) {
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_kv_cur,
                                      (size_t)opt.slots * kIndexCompWidth * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_score_cur,
                                      (size_t)opt.slots * kIndexCompWidth * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_indexer_scores,
                                      (size_t)opt.slots * kIndexerTopK * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_indexer_topk,
                                      (size_t)opt.slots * kIndexerTopK * sizeof(uint32_t)));
            }
            if (!r.d_index_comp_state_kv_layers[layer]) {
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_state_kv_layers[layer],
                                      (size_t)opt.slots * kIndexCompStateRows *
                                          (size_t)kIndexCompWidth * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_state_score_layers[layer],
                                      (size_t)opt.slots * kIndexCompStateRows *
                                          (size_t)kIndexCompWidth * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_rows_layers[layer],
                                      (size_t)opt.slots * kBoundedCompRows *
                                          (size_t)kIndexerHeadDim * sizeof(float)));
                CHECK_CUDA(cudaMemsetAsync(r.d_index_comp_state_kv_layers[layer], 0,
                                           (size_t)opt.slots * kIndexCompStateRows *
                                               (size_t)kIndexCompWidth * sizeof(float),
                                           r.stream));
                CHECK_CUDA(cudaMemsetAsync(r.d_index_comp_state_score_layers[layer], 0,
                                           (size_t)opt.slots * kIndexCompStateRows *
                                               (size_t)kIndexCompWidth * sizeof(float),
                                           r.stream));
                CHECK_CUDA(cudaMemsetAsync(r.d_index_comp_rows_layers[layer], 0,
                                           (size_t)opt.slots * kBoundedCompRows *
                                               (size_t)kIndexerHeadDim * sizeof(float),
                                           r.stream));
            }
            r.d_index_comp_state_kv = r.d_index_comp_state_kv_layers[layer];
            r.d_index_comp_state_score = r.d_index_comp_state_score_layers[layer];
            r.d_index_comp_rows = r.d_index_comp_rows_layers[layer];
        }
        for (int src = 0; src < kGpus; ++src) {
            if (!r.d_ep_remote[src]) CHECK_CUDA(cudaMalloc(&r.d_ep_remote[src],
                                                           (size_t)remote_float_bytes));
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
    stats->fused_compose_sum =
        opt.fuse_compose_sum && !opt.ep_return_fp16 && !opt.compact_route_compose;
    stats->dense_hmma_compose = opt.dense_hmma_compose;
    stats->dense_f16_cublas_compose = opt.dense_f16_cublas_compose;
    stats->nccl_reduce_scatter_compose =
        opt.nccl_reduce_scatter_compose_gate &&
        !opt.compact_route_compose && !opt.ep_return_fp16;

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
    const bool skip_self_copy = opt.skip_self_compose_copy && !opt.ep_return_fp16;
    const bool nccl_reduce_scatter = stats->nccl_reduce_scatter_compose;
    stats->ep_contribution_bytes = all_contrib_bytes * kGpus;
    if (opt.compact_route_compose && !opt.ep_return_fp16) {
        uint64_t compact_return_bytes = 0;
        for (int src = 0; src < kGpus; ++src) {
            const int src_compose_routes = routed_compose_rows(ranks[src], opt);
            compact_return_bytes +=
                (uint64_t)src_compose_routes * (kHidden / kGpus) * sizeof(float) *
                (skip_self_copy ? (kGpus - 1) : kGpus);
        }
        stats->ep_return_bytes = compact_return_bytes;
    } else {
        stats->ep_return_bytes = return_shard_bytes *
                                 (skip_self_copy ? (kGpus * kGpus - kGpus)
                                                 : (kGpus * kGpus));
    }
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
        if (route_hidden_elems > 0) {
            ep_reduce_all_dest_shards_kernel<<<grid, block, 0, r.stream>>>(
                r.d_ep_contrib_all, r.d_down, r.d_route_slots, r.d_route_weights,
                nullptr, r.routes, opt.slots, p);
            CHECK_CUDA(cudaGetLastError());
        }
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

    if (nccl_reduce_scatter) {
        for (int p = 0; p < kGpus; ++p) {
            if (!ranks[p].compose_nccl_initialized || !ranks[p].compose_nccl) {
                return 3;
            }
        }
        CHECK_NCCL(ncclGroupStart());
        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_NCCL(ncclReduceScatter(ranks[p].d_ep_contrib_all,
                                         ranks[p].d_ep_sum,
                                         (size_t)shard_elems,
                                         ncclFloat,
                                         ncclSum,
                                         ranks[p].compose_nccl,
                                         ranks[p].stream));
        }
        CHECK_NCCL(ncclGroupEnd());
    } else {
        uint64_t copy_elems_by_src[kGpus] = {};
        for (int src = 0; src < kGpus; ++src) {
            copy_elems_by_src[src] = shard_elems;
        }
        if (broadcast_ep_return_slices(
                ranks, opt.ep_return_fp16, skip_self_copy, shard_elems,
                copy_elems_by_src,
                opt.ep_return_fp16 ? "ep_compose_half_bcast"
                                   : "ep_compose_float_bcast") != 0) {
            return 4;
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
            if (nccl_reduce_scatter) {
                compose_next_hidden_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn.d_out[(size_t)dst],
                    shared.d_out[(size_t)dst], r.d_ep_sum, dst, opt.slots);
            } else if (stats->fused_compose_sum) {
                const float *r0 = skip_self_copy && dst == 0
                    ? ranks[0].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[0];
                const float *r1 = skip_self_copy && dst == 1
                    ? ranks[1].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[1];
                const float *r2 = skip_self_copy && dst == 2
                    ? ranks[2].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[2];
                const float *r3 = skip_self_copy && dst == 3
                    ? ranks[3].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[3];
                const float *r4 = skip_self_copy && dst == 4
                    ? ranks[4].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[4];
                const float *r5 = skip_self_copy && dst == 5
                    ? ranks[5].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[5];
                const float *r6 = skip_self_copy && dst == 6
                    ? ranks[6].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[6];
                const float *r7 = skip_self_copy && dst == 7
                    ? ranks[7].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[7];
                compose_next_hidden_sum8_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn.d_out[(size_t)dst],
                    shared.d_out[(size_t)dst], r0, r1, r2, r3, r4, r5, r6, r7,
                    dst, opt.slots);
            } else {
                zero_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_sum, shard_elems);
                CHECK_CUDA(cudaGetLastError());
                for (int src = 0; src < kGpus; ++src) {
                    if (opt.ep_return_fp16) {
                        add_half_to_f32_kernel<<<grid, block, 0, r.stream>>>(
                            r.d_ep_sum, r.d_ep_remote_half[src], shard_elems);
                    } else {
                        const float *src_contrib = skip_self_copy && src == dst
                            ? ranks[src].d_ep_contrib_all + (uint64_t)dst * shard_elems
                            : r.d_ep_remote[src];
                        add_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_sum,
                                                                     src_contrib,
                                                                     shard_elems);
                    }
                }
                CHECK_CUDA(cudaGetLastError());
                compose_next_hidden_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn.d_out[(size_t)dst],
                    shared.d_out[(size_t)dst], r.d_ep_sum, dst, opt.slots);
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

int run_true_ds4_attention_projection_prefix(const Options &opt,
                                             SharedHcControls *hc,
                                             const LayerDenseOps *ops,
                                             RankState ranks[kGpus],
                                             int layer) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    if (!hc->d_current_full || !hc->d_attn_normed ||
        !hc->d_q_a_full || !hc->d_q_a_normed ||
        !hc->d_kv_full || !hc->d_kv_normed ||
        !hc->d_attn_norm_weight[layer] ||
        !hc->d_q_a_norm_weight[layer] ||
        !hc->d_kv_a_norm_weight[layer]) {
        return 2;
    }
    if (ops->attn_q_a.cols != kHidden || ops->attn_q_a.rows_per_gpu != 1024 / kGpus ||
        ops->attn_q_b.cols != 1024 || ops->attn_q_b.rows_per_gpu != 32768 / kGpus ||
        ops->attn_kv_latent.cols != kHidden ||
        ops->attn_kv_latent.rows_per_gpu != kHeadDim / kGpus) {
        std::fprintf(stderr,
                     "tp_ep_true_attention_projection_bad_shape\tlayer\t%d\t"
                     "q_a_cols\t%d\tq_a_rows_per_gpu\t%d\t"
                     "q_b_cols\t%d\tq_b_rows_per_gpu\t%d\t"
                     "kv_cols\t%d\tkv_rows_per_gpu\t%d\n",
                     layer,
                     ops->attn_q_a.cols, ops->attn_q_a.rows_per_gpu,
                     ops->attn_q_b.cols, ops->attn_q_b.rows_per_gpu,
                     ops->attn_kv_latent.cols, ops->attn_kv_latent.rows_per_gpu);
        return 3;
    }

    const auto start = std::chrono::steady_clock::now();
    const int block = 256;
    const uint64_t hidden_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const uint64_t q_a_elems = (uint64_t)opt.slots * 1024ull;
    const uint64_t kv_elems = (uint64_t)opt.slots * (uint64_t)kHeadDim;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    const bool direct_input_fill =
        opt.true_ds4_attention_projection_direct_input_fill_gate;
    const bool rank_major_input =
        opt.true_ds4_attention_projection_rank_major_input_gate &&
        opt.tp_hc_current_input_nccl_allgather_gate;
    const bool rank_local_input =
        opt.true_ds4_attention_projection_rank_local_input_gate ||
        rank_major_input;
    const bool refresh_rank_major_from_slot_major =
        rank_major_input && opt.routed_ffn_norm_input_gate;
    const bool gathered_current_full =
        opt.tp_hc_current_input_peer_gather_gate ||
        opt.tp_hc_current_input_nccl_allgather_gate;
    const bool broadcast_normed_current = !direct_input_fill && !rank_major_input;
    float *attention_current_full = hc->d_current_full;
    if (gathered_current_full && ranks[0].d_current_full) {
        attention_current_full = ranks[0].d_current_full;
    }
    cudaStream_t control_stream = graph_event_order ? ranks[0].stream : (cudaStream_t)0;

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    rms_norm_weight_rows_stable_kernel<<<
        (unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_attn_normed, attention_current_full, hc->d_attn_norm_weight[layer],
        (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());
    if (graph_event_order) {
        if (enqueue_rank_streams_wait_after_control(
                opt, ranks, control_stream) != 0) return 8;
    } else {
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    if (broadcast_normed_current) {
        const int bcast_rc = nccl_broadcast_f32_from_device0_to_current_full(
            opt, ranks, hc->d_attn_normed, hidden_elems,
            "attention_projection_normed_current");
        if (bcast_rc != 0) return 14;
        if (graph_event_order) {
            if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 15;
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_current_full ||
            !ops->attn_q_a.d_x_half[(size_t)rank] ||
            !ops->attn_kv_latent.d_x_half[(size_t)rank]) {
            return 4;
        }
        if (rank_local_input) {
            float *rank_weight = hc->d_attn_norm_weight_rank[layer][rank];
            if (!rank_weight) return 12;
            if (broadcast_normed_current) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->attn_q_a.d_x_half[(size_t)rank],
                                 r.d_current_full, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->attn_kv_latent.d_x_half[(size_t)rank],
                                 r.d_current_full, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
            } else if (rank_major_input && r.d_current_full_rank_major) {
                if (refresh_rank_major_from_slot_major) {
                    slot_major_current_to_rank_major_kernel<<<
                        (unsigned int)((hidden_elems + block - 1) / block),
                        block, 0, r.stream>>>(
                        r.d_current_full_rank_major, r.d_current_full,
                        (uint32_t)(kHidden / kGpus), (uint32_t)kGpus,
                        (uint32_t)opt.slots);
                    CHECK_CUDA(cudaGetLastError());
                }
                fill_two_hidden_inputs_half_from_rank_major_norm_kernel<<<
                    (unsigned int)opt.slots, 256, 0, r.stream>>>(
                    ops->attn_q_a.d_x_half[(size_t)rank],
                    ops->attn_kv_latent.d_x_half[(size_t)rank],
                    r.d_current_full_rank_major, rank_weight,
                    (uint32_t)(kHidden / kGpus), (uint32_t)kGpus,
                    (uint32_t)opt.slots, 1.0e-6f);
            } else {
                if (!r.d_current_full_normed) return 13;
                rms_norm_weight_rows_stable_kernel<<<
                    (unsigned int)opt.slots, 256, 0, r.stream>>>(
                    r.d_current_full_normed, r.d_current_full, rank_weight,
                    (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->attn_q_a.d_x_half[(size_t)rank],
                                 r.d_current_full_normed, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->attn_kv_latent.d_x_half[(size_t)rank],
                                 r.d_current_full_normed, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
            }
        } else if (direct_input_fill) {
            fill_two_hidden_inputs_half_from_current_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->attn_q_a.d_x_half[(size_t)rank],
                             ops->attn_kv_latent.d_x_half[(size_t)rank],
                             hc->d_attn_normed, (uint32_t)opt.slots);
        } else {
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->attn_q_a.d_x_half[(size_t)rank],
                             r.d_current_full, (uint32_t)kHidden,
                             (uint32_t)opt.slots);
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->attn_kv_latent.d_x_half[(size_t)rank],
                             r.d_current_full, (uint32_t)kHidden,
                             (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    if (graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 9;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }

    if (!graph_event_order && opt.true_ds4_attention_projection_input_parity_gate) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            cudaStream_t stream = r.stream ? r.stream : (cudaStream_t)0;
            const HalfInputDiffStats q_a_diff = collect_half_input_tensor_diff(
                r, ops->attn_q_a.d_x_half[(size_t)rank], hc->d_attn_normed,
                (uint32_t)kHidden, (uint32_t)opt.slots, stream);
            log_attention_projection_input_diff("attn_q_a_input", layer, rank,
                                                q_a_diff);
            if (rank_major_input && rank == 0 && layer <= 1) {
                log_attention_rank_major_input_debug(
                    "attn_q_a_input", layer, r,
                    ops->attn_q_a.d_x_half[(size_t)rank], hc->d_attn_normed,
                    r.d_current_full, r.d_current_full_rank_major,
                    hc->d_attn_norm_weight[layer], (uint32_t)opt.slots,
                    stream);
            }
            const HalfInputDiffStats kv_diff = collect_half_input_tensor_diff(
                r, ops->attn_kv_latent.d_x_half[(size_t)rank], hc->d_attn_normed,
                (uint32_t)kHidden, (uint32_t)opt.slots, stream);
            log_attention_projection_input_diff("attn_kv_latent_input", layer,
                                                rank, kv_diff);
        }
    }

    if (launch_resident_f8_dense(opt, ops->attn_q_a, ranks) != 0 ||
        launch_resident_f8_dense(opt, ops->attn_kv_latent, ranks) != 0) {
        return 5;
    }
    if (graph_event_order) {
        if (enqueue_control_wait_after_dense_streams(
                opt, ranks, control_stream) != 0) return 10;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            if (ranks[rank].dense_stream) {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].dense_stream));
            } else {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        gather_dense_shard_to_full_kernel<<<
            (unsigned int)(((uint64_t)opt.slots * (1024u / kGpus) + block - 1) / block),
            block, 0, control_stream>>>(
            hc->d_q_a_full, ops->attn_q_a.d_out[(size_t)rank], rank,
            1024u / kGpus, 1024u, (uint32_t)opt.slots);
        gather_dense_shard_to_full_kernel<<<
            (unsigned int)(((uint64_t)opt.slots * (kHeadDim / kGpus) + block - 1) / block),
            block, 0, control_stream>>>(
            hc->d_kv_full, ops->attn_kv_latent.d_out[(size_t)rank], rank,
            kHeadDim / kGpus, kHeadDim, (uint32_t)opt.slots);
    }
    CHECK_CUDA(cudaGetLastError());
    if (!graph_event_order) {
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    rms_norm_weight_rows_stable_kernel<<<
        (unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_q_a_normed, hc->d_q_a_full, hc->d_q_a_norm_weight[layer],
        1024u, (uint32_t)opt.slots, 1.0e-6f);
    rms_norm_weight_rows_stable_kernel<<<
        (unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_kv_normed, hc->d_kv_full, hc->d_kv_a_norm_weight[layer],
        (uint32_t)kHeadDim, (uint32_t)opt.slots, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());
    if (graph_event_order) {
        if (enqueue_rank_streams_wait_after_control(
                opt, ranks, control_stream) != 0) return 11;
    } else {
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    if (opt.true_ds4_attention_kv_norm_reference_gate) {
        float *d_kv_ref = nullptr;
        CHECK_CUDA(cudaMalloc(&d_kv_ref, (size_t)kv_elems * sizeof(float)));
        rms_norm_weight_rows_kernel<<<(unsigned int)opt.slots, 256>>>(
            d_kv_ref, hc->d_kv_full, hc->d_kv_a_norm_weight[layer],
            (uint32_t)kHeadDim, (uint32_t)opt.slots, 1.0e-6f);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        const TensorF32Stats kv_in =
            collect_tensor_f32_stats(hc->d_kv_full, (size_t)kv_elems, nullptr);
        const TensorF32Stats kv_stable =
            collect_tensor_f32_stats(hc->d_kv_normed, (size_t)kv_elems,
                                     nullptr);
        const TensorF32Stats kv_ref =
            collect_tensor_f32_stats(d_kv_ref, (size_t)kv_elems, nullptr);
        const TensorF32Stats kv_w =
            collect_tensor_f32_stats(hc->d_kv_a_norm_weight[layer],
                                     (size_t)kHeadDim, nullptr);
        const TensorF32DiffStats diff = collect_tensor_f32_diff_stats(
            hc->d_kv_normed, d_kv_ref, (size_t)kv_elems, nullptr);
        std::printf("tp_ep_true_attention_kv_norm_reference\tlayer\t%d\t"
                    "slots\t%d\tkv_in_max\t%.9g\tkv_in_bad\t%d\t"
                    "kv_weight_max\t%.9g\tkv_weight_bad\t%d\t"
                    "stable_max\t%.9g\tstable_bad\t%d\t"
                    "reference_max\t%.9g\treference_bad\t%d\t"
                    "max_abs_diff\t%.9g\tmax_rel_diff\t%.9g\tdiff_bad\t%d\t"
                    "first_bad\t%zu\tPASS\n",
                    layer, opt.slots, kv_in.max_abs, kv_in.finite_bad,
                    kv_w.max_abs, kv_w.finite_bad, kv_stable.max_abs,
                    kv_stable.finite_bad, kv_ref.max_abs, kv_ref.finite_bad,
                    diff.max_abs, diff.max_rel, diff.bad, diff.first_bad);
        CHECK_CUDA(cudaFree(d_kv_ref));
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!ops->attn_q_b.d_x_half[(size_t)rank]) return 6;
        fill_dense_input_half_from_tensor_kernel<<<
            (unsigned int)((q_a_elems + block - 1) / block), block, 0,
            r.stream>>>(ops->attn_q_b.d_x_half[(size_t)rank],
                         hc->d_q_a_normed, 1024u, (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    if (graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 12;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }

    if (launch_resident_f8_dense(opt, ops->attn_q_b, ranks) != 0) {
        return 7;
    }
    if (!graph_event_order) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            if (ranks[rank].dense_stream) {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].dense_stream));
            } else {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }

    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    if (!graph_event_order && layer <= 2) {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        log_tensor_f32_stats("true_attn_q_a_full", layer, 0, hc->d_q_a_full,
                             (size_t)q_a_elems, nullptr);
        log_tensor_f32_stats("true_attn_kv_normed", layer, 0, hc->d_kv_normed,
                             (size_t)kv_elems, nullptr);
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("true_attn_q_b_shard", layer, rank,
                                 ops->attn_q_b.d_out[(size_t)rank],
                                 (size_t)opt.slots * ops->attn_q_b.rows_per_gpu,
                                 ranks[rank].dense_stream ? ranks[rank].dense_stream
                                                          : ranks[rank].stream);
        }
    }
    if (!graph_event_order && opt.true_ds4_attention_saturation_audit_gate) {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        const TensorF32Stats current =
            collect_tensor_f32_stats(attention_current_full, (size_t)hidden_elems,
                                     nullptr);
        const TensorF32Stats attn_normed =
            collect_tensor_f32_stats(hc->d_attn_normed, (size_t)hidden_elems,
                                     nullptr);
        const TensorF32Stats q_a =
            collect_tensor_f32_stats(hc->d_q_a_full, (size_t)q_a_elems,
                                     nullptr);
        const TensorF32Stats q_a_normed =
            collect_tensor_f32_stats(hc->d_q_a_normed, (size_t)q_a_elems,
                                     nullptr);
        const TensorF32Stats kv =
            collect_tensor_f32_stats(hc->d_kv_full, (size_t)kv_elems,
                                     nullptr);
        const TensorF32Stats kv_normed =
            collect_tensor_f32_stats(hc->d_kv_normed, (size_t)kv_elems,
                                     nullptr);
        TensorF32Stats q_b;
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            const TensorF32Stats shard = collect_tensor_f32_stats(
                ops->attn_q_b.d_out[(size_t)rank],
                (size_t)opt.slots * ops->attn_q_b.rows_per_gpu,
                ranks[rank].dense_stream ? ranks[rank].dense_stream
                                         : ranks[rank].stream);
            merge_tensor_stats(&q_b, shard);
        }
        std::printf("tp_ep_true_attention_saturation_projection\tlayer\t%d\t"
                    "slots\t%d\tcurrent_max\t%.9g\tcurrent_bad\t%d\t"
                    "attn_normed_max\t%.9g\tattn_normed_bad\t%d\t"
                    "q_a_max\t%.9g\tq_a_bad\t%d\t"
                    "q_a_normed_max\t%.9g\tq_a_normed_bad\t%d\t"
                    "kv_max\t%.9g\tkv_bad\t%d\t"
                    "kv_normed_max\t%.9g\tkv_normed_bad\t%d\t"
                    "q_b_pre_head_max\t%.9g\tq_b_pre_head_bad\t%d\tPASS\n",
                    layer, opt.slots, current.max_abs, current.finite_bad,
                    attn_normed.max_abs, attn_normed.finite_bad, q_a.max_abs,
                    q_a.finite_bad, q_a_normed.max_abs, q_a_normed.finite_bad,
                    kv.max_abs, kv.finite_bad, kv_normed.max_abs,
                    kv_normed.finite_bad, q_b.max_abs, q_b.finite_bad);
    }
    std::printf("tp_ep_true_attention_projection_prefix\tlayer\t%d\tslots\t%d\t"
                "q_a_cols\t1024\tkv_cols\t%d\tq_width\t32768\t"
                "direct_input_fill\t%d\trank_local_input\t%d\t"
                "rank_major_input\t%d\tcurrent_source\t%s\tms\t%.6f\tPASS\n",
                layer, opt.slots, kHeadDim, direct_input_fill ? 1 : 0,
                rank_local_input ? 1 : 0, rank_major_input ? 1 : 0,
                attention_current_full == hc->d_current_full ? "shared_hc" : "rank0",
                ms);
    return 0;
}

int run_true_ds4_compressed_reference_diff_gate(const Options &opt,
                                                SharedHcControls *hc,
                                                RankState ranks[kGpus],
                                                int layer,
                                                int ratio,
                                                int comp_width,
                                                uint32_t emitted,
                                                uint32_t comp_row,
                                                uint32_t visible_rows) {
    if (!opt.true_ds4_compressed_reference_diff_gate) return 0;
    if (!hc || !hc->initialized || layer < 0 || layer >= 43) return 1;
    if (ratio != 4 || !emitted) {
        std::printf("tp_ep_compressed_reference_diff\tlayer\t%d\tratio\t%d\t"
                    "emitted\t%u\tSKIP\n",
                    layer, ratio, emitted);
        return 0;
    }
    RankState &r0 = ranks[0];
    CHECK_CUDA(cudaSetDevice(r0.device));
    if (!hc->d_attn_comp_kv_full || !hc->d_attn_comp_score_full ||
        !hc->d_attn_compress_ape[layer] || !hc->d_attn_compress_norm[layer] ||
        !r0.d_attn_comp_kv_cur || !r0.d_attn_comp_score_cur ||
        !r0.d_attn_comp_rows || !r0.d_index_comp_rows ||
        !r0.d_indexer_scores || !r0.d_indexer_topk ||
        !hc->d_index_comp_kv_full || !hc->d_index_comp_score_full ||
        !hc->d_indexer_compress_ape[layer] ||
        !hc->d_indexer_compress_norm[layer] ||
        !hc->d_indexer_q_full || !hc->d_indexer_w_full) {
        return 2;
    }

    const int block = 256;
    const uint32_t state_rows =
        (uint32_t)attn_comp_state_rows_for_ratio(ratio);
    const uint32_t state_width =
        (uint32_t)attn_comp_state_width_for_ratio(ratio);
    const float comp_freq_scale = 1.0f / kRopeScaleFactor;
    const float comp_ext_factor = 1.0f;
    float comp_attn_factor = 1.0f;
    comp_attn_factor /= 1.0f + 0.1f * logf(1.0f / comp_freq_scale);
    const size_t attn_state_elems =
        (size_t)opt.slots * state_rows * (size_t)state_width;
    const size_t attn_row_elems = (size_t)opt.slots * kHeadDim;
    const size_t index_state_elems =
        (size_t)opt.slots * kIndexCompStateRows * (size_t)kIndexCompWidth;
    const size_t index_row_elems = (size_t)opt.slots * kIndexerHeadDim;

    float *d_attn_state_kv = nullptr;
    float *d_attn_state_score = nullptr;
    float *d_attn_row_ref = nullptr;
    float *d_attn_row_tp = nullptr;
    float *d_index_state_kv = nullptr;
    float *d_index_state_score = nullptr;
    float *d_index_row_ref = nullptr;
    float *d_index_row_tp = nullptr;
    float *d_index_score_ref = nullptr;
    float *d_index_score_ref_compact = nullptr;
    float *d_index_score_tp = nullptr;
    uint32_t *d_index_topk_ref = nullptr;

    CHECK_CUDA(cudaMalloc(&d_attn_state_kv, attn_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_state_score, attn_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_row_ref, attn_row_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_row_tp, attn_row_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_state_kv, index_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_state_score, index_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_row_ref, index_row_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_row_tp, index_row_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_score_ref,
                          (size_t)opt.slots * kIndexerTopK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_score_ref_compact,
                          (size_t)opt.slots * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_score_tp, (size_t)opt.slots * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_topk_ref,
                          (size_t)opt.slots * kIndexerTopK * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_attn_state_kv, 0, attn_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_attn_state_score, 0, attn_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_index_state_kv, 0, index_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_index_state_score, 0, index_state_elems * sizeof(float)));

    log_tensor_f32_diff_summary("attn_comp_kv_current_peer_copy", layer,
                                r0.d_attn_comp_kv_cur,
                                hc->d_attn_comp_kv_full,
                                (size_t)opt.slots * (size_t)comp_width,
                                r0.stream);
    log_tensor_f32_diff_summary("attn_comp_score_current_peer_copy", layer,
                                r0.d_attn_comp_score_cur,
                                hc->d_attn_comp_score_full,
                                (size_t)opt.slots * (size_t)comp_width,
                                r0.stream);

    compressor_pool_emit_slots_kernel<<<
        dim3((unsigned int)((kHeadDim + block - 1) / block),
             (unsigned int)opt.slots, 1u),
        block>>>(d_attn_row_ref, r0.d_attn_comp_state_kv,
                 r0.d_attn_comp_state_score,
                 (uint32_t)opt.slots, (uint32_t)kHeadDim, (uint32_t)ratio,
                 0u, 1u, state_rows, state_width);
    compressor_norm_emit_slots_kernel<<<(unsigned int)opt.slots, 256>>>(
        d_attn_row_ref, hc->d_attn_compress_norm[layer], (uint32_t)opt.slots,
        (uint32_t)kHeadDim, 0u, 1u, 1.0e-6f);
    if (opt.true_ds4_attention_rope_gate) {
        rope_tail_comp_emit_slots_kernel<<<(unsigned int)opt.slots, 64>>>(
            d_attn_row_ref, (uint32_t)opt.slots, (uint32_t)kHeadDim,
            (uint32_t)kRotaryDim, 0u, 1u,
            (uint32_t)(opt.position + 1ull - (uint64_t)ratio),
            kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
            comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
            kRopeYarnBetaSlow);
    }
    round_comp_emit_slots_kernel<<<
        (unsigned int)(((uint64_t)opt.slots * kHeadDim + block - 1) / block),
        block>>>(d_attn_row_ref, (uint32_t)opt.slots, (uint32_t)kHeadDim,
                 0u, 1u);
    pack_comp_row_kernel<<<
        (unsigned int)(((uint64_t)opt.slots * kHeadDim + block - 1) / block),
        block>>>(d_attn_row_tp, r0.d_attn_comp_rows, (uint32_t)opt.slots,
                 (uint32_t)kHeadDim, comp_row, (uint32_t)kBoundedCompRows);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    log_tensor_f32_diff_summary("attn_comp_row_compact_reference", layer,
                                d_attn_row_tp, d_attn_row_ref, attn_row_elems,
                                nullptr);

    compressor_pool_emit_slots_kernel<<<
        dim3((unsigned int)((kIndexerHeadDim + block - 1) / block),
             (unsigned int)opt.slots, 1u),
        block>>>(d_index_row_ref, r0.d_index_comp_state_kv,
                 r0.d_index_comp_state_score,
                 (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, 4u,
                 0u, 1u, (uint32_t)kIndexCompStateRows,
                 (uint32_t)kIndexCompWidth);
    compressor_norm_emit_slots_kernel<<<(unsigned int)opt.slots, 256>>>(
        d_index_row_ref, hc->d_indexer_compress_norm[layer],
        (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, 0u, 1u,
        1.0e-6f);
    if (opt.true_ds4_attention_rope_gate) {
        rope_tail_comp_emit_slots_kernel<<<(unsigned int)opt.slots, 64>>>(
            d_index_row_ref, (uint32_t)opt.slots,
            (uint32_t)kIndexerHeadDim, (uint32_t)kRotaryDim, 0u, 1u,
            (uint32_t)(opt.position + 1ull - 4ull),
            kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
            comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
            kRopeYarnBetaSlow);
    }
    round_comp_emit_slots_kernel<<<
        (unsigned int)(((uint64_t)opt.slots * kIndexerHeadDim + block - 1) /
                       block),
        block>>>(d_index_row_ref, (uint32_t)opt.slots,
                 (uint32_t)kIndexerHeadDim, 0u, 1u);
    pack_comp_row_kernel<<<
        (unsigned int)(((uint64_t)opt.slots * kIndexerHeadDim + block - 1) /
                       block),
        block>>>(d_index_row_tp, r0.d_index_comp_rows, (uint32_t)opt.slots,
                 (uint32_t)kIndexerHeadDim, comp_row,
                 (uint32_t)kBoundedCompRows);
    indexer_score_bounded_rows_slots_kernel<<<(unsigned int)opt.slots, 256>>>(
        d_index_score_ref, d_index_topk_ref, hc->d_indexer_q_full,
        hc->d_indexer_w_full, d_index_row_ref, (uint32_t)opt.slots,
        1u, 1u, (uint32_t)kIndexerTopK,
        1.0f / sqrtf((float)(kIndexerHead * kIndexerHeadDim)));
    pack_indexer_score_column_kernel<<<
        (unsigned int)((opt.slots + block - 1) / block), block>>>(
        d_index_score_tp, r0.d_indexer_scores, (uint32_t)opt.slots,
        (uint32_t)kIndexerTopK, comp_row);
    pack_indexer_score_column_kernel<<<
        (unsigned int)((opt.slots + block - 1) / block), block>>>(
        d_index_score_ref_compact, d_index_score_ref, (uint32_t)opt.slots,
        (uint32_t)kIndexerTopK, 0u);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    log_tensor_f32_diff_summary("index_comp_row_compact_reference", layer,
                                d_index_row_tp, d_index_row_ref,
                                index_row_elems, nullptr);
    log_tensor_f32_diff_summary("indexer_score_row_compact_reference", layer,
                                d_index_score_tp, d_index_score_ref_compact,
                                (size_t)opt.slots, nullptr);
    std::printf("tp_ep_compressed_reference_diff_summary\tlayer\t%d\t"
                "ratio\t%d\temitted\t%u\tcomp_row\t%u\t"
                "visible_compressed_rows\t%u\tPASS\n",
                layer, ratio, emitted, comp_row, visible_rows);

    CHECK_CUDA(cudaFree(d_index_topk_ref));
    CHECK_CUDA(cudaFree(d_index_score_tp));
    CHECK_CUDA(cudaFree(d_index_score_ref_compact));
    CHECK_CUDA(cudaFree(d_index_score_ref));
    CHECK_CUDA(cudaFree(d_index_row_tp));
    CHECK_CUDA(cudaFree(d_index_row_ref));
    CHECK_CUDA(cudaFree(d_index_state_score));
    CHECK_CUDA(cudaFree(d_index_state_kv));
    CHECK_CUDA(cudaFree(d_attn_row_tp));
    CHECK_CUDA(cudaFree(d_attn_row_ref));
    CHECK_CUDA(cudaFree(d_attn_state_score));
    CHECK_CUDA(cudaFree(d_attn_state_kv));
    return 0;
}

int run_true_ds4_compressed_kv_projection_gate(const Options &opt,
                                               SharedHcControls *hc,
                                               const LayerDenseOps *ops,
                                               RankState ranks[kGpus],
                                               ds4_v100_tp_runtime *rt,
                                               int layer) {
    if (!opt.true_ds4_compressed_kv_gate) return 0;
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 43 || !hc->d_attn_normed || !hc->d_q_a_normed) {
        return 1;
    }
    const int ratio = ds4_layer_ratio(layer);
    const uint32_t emitted =
        ratio != 0 && (((opt.position + 1ull) % (uint64_t)ratio) == 0ull) ? 1u : 0u;
    const uint32_t indexer_topk =
        opt.true_ds4_indexer_attention_gate && ratio == 4 ? kIndexerTopK : 0u;
    if (ratio == 0) {
        std::printf("tp_ep_compressed_kv_projection\tlayer\t%d\tslots\t%d\t"
                    "ratio\t0\temitted_compressed_rows\t0\t"
                    "visible_compressed_rows\t0\tindexer_topk_count\t0\t"
                    "attn_input_fill_ms\t0.000000\tattn_dense_ms\t0.000000\t"
                    "attn_gather_ms\t0.000000\tattn_state_emit_ms\t0.000000\t"
                    "attn_typed_ms\t0.000000\tindexer_input_fill_ms\t0.000000\t"
                    "indexer_dense_ms\t0.000000\tindexer_gather_rope_ms\t0.000000\t"
                    "indexer_state_emit_ms\t0.000000\tindexer_typed_score_ms\t0.000000\t"
                    "reference_diff_ms\t0.000000\tratio_shift_ms\t0.000000\t"
                    "ms\t0.000000\tPASS\n",
                    layer, opt.slots);
        return 0;
    }

    const int comp_width = ratio == 4 ? 2 * kHeadDim : kHeadDim;
    const int comp_state_rows = attn_comp_state_rows_for_ratio(ratio);
    const int comp_state_width = attn_comp_state_width_for_ratio(ratio);
    if (ops->attn_compress_kv.cols != kHidden ||
        ops->attn_compress_gate.cols != kHidden ||
        ops->attn_compress_kv.rows_per_gpu != comp_width / kGpus ||
        ops->attn_compress_gate.rows_per_gpu != comp_width / kGpus) {
        std::fprintf(stderr,
                     "tp_ep_compressed_kv_bad_shape\tlayer\t%d\t"
                     "ratio\t%d\tkv_cols\t%d\tkv_rows_per_gpu\t%d\t"
                     "gate_cols\t%d\tgate_rows_per_gpu\t%d\n",
                     layer, ratio, ops->attn_compress_kv.cols,
                     ops->attn_compress_kv.rows_per_gpu,
                     ops->attn_compress_gate.cols,
                     ops->attn_compress_gate.rows_per_gpu);
        return 2;
    }

    const auto start = std::chrono::steady_clock::now();
    auto t_stage = start;
    auto elapsed_ms = [](std::chrono::steady_clock::time_point a,
                         std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    double attn_input_fill_ms = 0.0;
    double attn_dense_ms = 0.0;
    double attn_gather_ms = 0.0;
    double attn_state_emit_ms = 0.0;
    double attn_typed_ms = 0.0;
    double indexer_input_fill_ms = 0.0;
    double indexer_dense_ms = 0.0;
    double indexer_gather_rope_ms = 0.0;
    double indexer_state_emit_ms = 0.0;
    double indexer_typed_score_ms = 0.0;
    double reference_diff_ms = 0.0;
    double ratio_shift_ms = 0.0;
    const int block = 256;
    const uint64_t hidden_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const bool direct_current_input_fill =
        opt.true_ds4_compressed_kv_direct_input_fill_gate;
    const bool dense_event_wait =
        opt.true_ds4_compressed_kv_dense_event_wait_gate;
    const bool skip_dense_stats =
        opt.true_ds4_compressed_kv_skip_dense_stats_gate;
    const bool fused_attn_current_fill =
        opt.true_ds4_compressed_kv_fused_attn_input_fill_gate;
    const bool fused_ratio4_current_fill =
        opt.true_ds4_compressed_kv_fused_input_fill_gate &&
        opt.true_ds4_indexer_attention_gate && ratio == 4;
    const bool fused_rope_round =
        opt.true_ds4_compressed_kv_fused_rope_round_gate &&
        opt.true_ds4_attention_rope_gate && emitted;
    const bool fused_pool_norm =
        opt.true_ds4_compressed_kv_fused_pool_norm_gate && emitted;
    const bool fused_pool_norm_rope_round =
        opt.true_ds4_compressed_kv_fused_pool_norm_rope_round_gate &&
        opt.true_ds4_attention_rope_gate && emitted;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    cudaStream_t control_stream = graph_event_order ? ranks[0].stream : (cudaStream_t)0;
    if (!direct_current_input_fill) {
        const int bcast_rc = nccl_broadcast_f32_from_device0_to_current_full(
            opt, ranks, hc->d_attn_normed, hidden_elems,
            "compressed_kv_normed_current");
        if (bcast_rc != 0) return 23;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_current_full ||
            !ops->attn_compress_kv.d_x_half[(size_t)rank] ||
            !ops->attn_compress_gate.d_x_half[(size_t)rank] ||
            (fused_ratio4_current_fill &&
             (!ops->indexer_proj.d_x_half[(size_t)rank] ||
              !ops->indexer_compress_kv.d_x_half[(size_t)rank] ||
              !ops->indexer_compress_gate.d_x_half[(size_t)rank]))) {
            return 3;
        }
        const float *current_src = hc->d_attn_normed;
        if (!direct_current_input_fill) {
            current_src = r.d_current_full;
        }
        if (fused_ratio4_current_fill) {
            fill_ratio4_compressed_indexer_inputs_half_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(
                ops->attn_compress_kv.d_x_half[(size_t)rank],
                ops->attn_compress_gate.d_x_half[(size_t)rank],
                ops->indexer_proj.d_x_half[(size_t)rank],
                ops->indexer_compress_kv.d_x_half[(size_t)rank],
                ops->indexer_compress_gate.d_x_half[(size_t)rank],
                current_src, (uint32_t)opt.slots);
        } else if (fused_attn_current_fill) {
            fill_attn_compressed_inputs_half_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(
                ops->attn_compress_kv.d_x_half[(size_t)rank],
                ops->attn_compress_gate.d_x_half[(size_t)rank],
                current_src, (uint32_t)opt.slots);
        } else {
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->attn_compress_kv.d_x_half[(size_t)rank],
                             current_src, (uint32_t)kHidden,
                             (uint32_t)opt.slots);
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->attn_compress_gate.d_x_half[(size_t)rank],
                             current_src, (uint32_t)kHidden,
                             (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    if (dense_event_wait || graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 22;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_input_fill_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }
    if (launch_resident_f8_dense(opt, ops->attn_compress_kv, ranks) != 0 ||
        launch_resident_f8_dense(opt, ops->attn_compress_gate, ranks) != 0) {
        return 4;
    }

    TensorF32Stats attn_kv_stats;
    TensorF32Stats attn_gate_stats;
    if (graph_event_order) {
        if (enqueue_control_wait_after_dense_streams(
                opt, ranks, control_stream) != 0) return 24;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            cudaStream_t stream = ranks[rank].dense_stream ? ranks[rank].dense_stream
                                                           : ranks[rank].stream;
            CHECK_CUDA(cudaStreamSynchronize(stream));
            if (!skip_dense_stats) {
            const size_t comp_elems =
                (size_t)opt.slots * (size_t)ops->attn_compress_kv.rows_per_gpu;
                merge_tensor_stats(&attn_kv_stats,
                                   collect_tensor_f32_stats(
                                       ops->attn_compress_kv.d_out[(size_t)rank],
                                       comp_elems, stream));
                merge_tensor_stats(&attn_gate_stats,
                                   collect_tensor_f32_stats(
                                       ops->attn_compress_gate.d_out[(size_t)rank],
                                       comp_elems, stream));
            }
        }
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_dense_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }

    if (!hc->d_attn_comp_kv_full || !hc->d_attn_comp_score_full ||
        !hc->d_attn_compress_ape[layer] || !hc->d_attn_compress_norm[layer]) {
        return 9;
    }
    uint32_t emitted_comp_row = 0u;
    uint32_t visible = 0u;
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        gather_dense_shard_to_full_kernel<<<
            (unsigned int)(((uint64_t)opt.slots *
                                (uint64_t)ops->attn_compress_kv.rows_per_gpu +
                            block - 1) /
                           block),
            block, 0, control_stream>>>(
            hc->d_attn_comp_kv_full,
            ops->attn_compress_kv.d_out[(size_t)rank], rank,
            (uint32_t)ops->attn_compress_kv.rows_per_gpu,
            (uint32_t)comp_width, (uint32_t)opt.slots);
        gather_dense_shard_to_full_kernel<<<
            (unsigned int)(((uint64_t)opt.slots *
                                (uint64_t)ops->attn_compress_gate.rows_per_gpu +
                            block - 1) /
                           block),
            block, 0, control_stream>>>(
            hc->d_attn_comp_score_full,
            ops->attn_compress_gate.d_out[(size_t)rank], rank,
            (uint32_t)ops->attn_compress_gate.rows_per_gpu,
            (uint32_t)comp_width, (uint32_t)opt.slots);
    }
    CHECK_CUDA(cudaGetLastError());
    if (graph_event_order) {
        if (enqueue_rank_streams_wait_after_control(
                opt, ranks, control_stream) != 0) return 25;
    } else {
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_gather_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }

    const float comp_freq_scale = 1.0f / kRopeScaleFactor;
    const float comp_ext_factor = 1.0f;
    float comp_attn_factor = 1.0f;
    comp_attn_factor /= 1.0f + 0.1f * logf(1.0f / comp_freq_scale);
    const size_t comp_bytes = (size_t)opt.slots * comp_width * sizeof(float);
    const uint64_t comp_elems = (uint64_t)opt.slots * (uint64_t)comp_width;
    if (!graph_event_order) {
        void *kv_dsts[kGpus] = {};
        void *score_dsts[kGpus] = {};
        for (int rank = 0; rank < kGpus; ++rank) {
            kv_dsts[rank] = ranks[rank].d_attn_comp_kv_cur;
            score_dsts[rank] = ranks[rank].d_attn_comp_score_cur;
        }
        if (nccl_broadcast_bytes_from_rank0(
                ranks, hc->d_attn_comp_kv_full, kv_dsts, comp_bytes,
                "attn_comp_kv_cur") != 0 ||
            nccl_broadcast_bytes_from_rank0(
                ranks, hc->d_attn_comp_score_full, score_dsts, comp_bytes,
                "attn_comp_score_cur") != 0) {
            return 28;
        }
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_comp_kv_cur || !r.d_attn_comp_score_cur ||
            !r.d_attn_comp_state_kv || !r.d_attn_comp_state_score ||
            !r.d_attn_comp_rows) {
            return 10;
        }
        if (graph_event_order) {
            enqueue_graph_f32_copy_from_device0(
                opt, r, rank, r.d_attn_comp_kv_cur,
                hc->d_attn_comp_kv_full, comp_elems, r.stream, block);
            enqueue_graph_f32_copy_from_device0(
                opt, r, rank, r.d_attn_comp_score_cur,
                hc->d_attn_comp_score_full, comp_elems, r.stream, block);
        }
        compressor_store_slots_kernel<<<
            (unsigned int)(((uint64_t)opt.slots * (uint64_t)comp_width +
                            block - 1) /
                           block),
            block, 0, r.stream>>>(
            r.d_attn_comp_kv_cur, r.d_attn_comp_score_cur,
            r.d_attn_comp_state_kv, r.d_attn_comp_state_score,
            hc->d_attn_compress_ape[layer], (uint32_t)opt.slots,
            (uint32_t)kHeadDim, (uint32_t)ratio, (uint32_t)opt.position,
            (uint32_t)comp_state_rows, (uint32_t)comp_state_width);
        if (emitted) {
            const uint32_t comp_row =
                r.attn_comp_rows_written_layers[layer] %
                (uint32_t)kBoundedCompRows;
            if (rank == 0) emitted_comp_row = comp_row;
            r.attn_comp_row_position_layers[layer][comp_row] = opt.position;
            r.attn_comp_row_loaded_layers[layer][comp_row] = false;
            if (fused_pool_norm_rope_round) {
                compressor_pool_norm_rope_round_emit_slots_kernel<<<
                    (unsigned int)opt.slots, 256, 0, r.stream>>>(
                    r.d_attn_comp_rows, r.d_attn_comp_state_kv,
                    r.d_attn_comp_state_score, hc->d_attn_compress_norm[layer],
                    (uint32_t)opt.slots, (uint32_t)kHeadDim, (uint32_t)ratio,
                    comp_row, (uint32_t)kBoundedCompRows,
                    (uint32_t)comp_state_rows, (uint32_t)comp_state_width,
                    1.0e-6f, (uint32_t)kRotaryDim,
                    (uint32_t)(opt.position + 1ull - (uint64_t)ratio),
                    kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                    comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                    kRopeYarnBetaSlow);
            } else if (fused_pool_norm) {
                compressor_pool_norm_emit_slots_kernel<<<
                    (unsigned int)opt.slots, 256, 0, r.stream>>>(
                    r.d_attn_comp_rows, r.d_attn_comp_state_kv,
                    r.d_attn_comp_state_score, hc->d_attn_compress_norm[layer],
                    (uint32_t)opt.slots, (uint32_t)kHeadDim, (uint32_t)ratio,
                    comp_row, (uint32_t)kBoundedCompRows,
                    (uint32_t)comp_state_rows, (uint32_t)comp_state_width,
                    1.0e-6f);
            } else {
                compressor_pool_emit_slots_kernel<<<
                    dim3((unsigned int)((kHeadDim + block - 1) / block),
                         (unsigned int)opt.slots, 1u),
                    block, 0, r.stream>>>(
                    r.d_attn_comp_rows, r.d_attn_comp_state_kv,
                    r.d_attn_comp_state_score, (uint32_t)opt.slots,
                    (uint32_t)kHeadDim, (uint32_t)ratio, comp_row,
                    (uint32_t)kBoundedCompRows, (uint32_t)comp_state_rows,
                    (uint32_t)comp_state_width);
                compressor_norm_emit_slots_kernel<<<(unsigned int)opt.slots, 256,
                                                    0, r.stream>>>(
                    r.d_attn_comp_rows, hc->d_attn_compress_norm[layer],
                    (uint32_t)opt.slots, (uint32_t)kHeadDim, comp_row,
                    (uint32_t)kBoundedCompRows, 1.0e-6f);
            }
            if (fused_pool_norm_rope_round) {
                // RoPE and F16 rounding were already applied by the fused emit.
            } else if (fused_rope_round) {
                rope_tail_round_comp_emit_slots_kernel<<<
                    (unsigned int)opt.slots, 256, 0, r.stream>>>(
                    r.d_attn_comp_rows, (uint32_t)opt.slots,
                    (uint32_t)kHeadDim, (uint32_t)kRotaryDim, comp_row,
                    (uint32_t)kBoundedCompRows,
                    (uint32_t)(opt.position + 1ull - (uint64_t)ratio),
                    kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                    comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                    kRopeYarnBetaSlow);
            } else {
                if (opt.true_ds4_attention_rope_gate) {
                    rope_tail_comp_emit_slots_kernel<<<
                        (unsigned int)opt.slots, 64, 0, r.stream>>>(
                        r.d_attn_comp_rows, (uint32_t)opt.slots,
                        (uint32_t)kHeadDim, (uint32_t)kRotaryDim, comp_row,
                        (uint32_t)kBoundedCompRows,
                        (uint32_t)(opt.position + 1ull - (uint64_t)ratio),
                        kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                        comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                        kRopeYarnBetaSlow);
                }
                round_comp_emit_slots_kernel<<<
                    (unsigned int)(((uint64_t)opt.slots * kHeadDim + block - 1) /
                                   block),
                    block, 0, r.stream>>>(
                    r.d_attn_comp_rows, (uint32_t)opt.slots, (uint32_t)kHeadDim,
                    comp_row, (uint32_t)kBoundedCompRows);
            }
            r.attn_comp_rows_written_layers[layer]++;
        }
        visible = std::max(
            visible,
            std::min(r.attn_comp_rows_written_layers[layer],
                     (uint32_t)kBoundedCompRows));
        CHECK_CUDA(cudaGetLastError());
    }
    if (!graph_event_order) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_state_emit_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }
    if (opt.true_ds4_attention_typed_kv_compressed_gate && emitted) {
        if (!rt) {
            std::fprintf(stderr,
                         "tp_ep_true_attention_typed_kv_compressed_failed\t"
                         "layer\t%d\treason\tmissing_tp_runtime\n",
                         layer);
            return 14;
        }
        char err[512] = {0};
        ds4_v100_tp_kv_row_view view;
        if (ds4_v100_tp_runtime_kv_row_view(
                rt, layer, 0, opt.position, DS4_V100_TP_KV_ROW_ATTN, &view, err,
                sizeof(err)) != 0) {
            std::fprintf(stderr,
                         "tp_ep_true_attention_typed_kv_compressed_view_failed\t"
                         "layer\t%d\t%s\n",
                         layer, err);
            return 15;
        }
        int current_store = 0;
        if (!opt.true_ds4_attention_typed_kv_skip_compressed_store_gate) {
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                const void *src[kGpus] = {};
                void *streams[kGpus] = {};
                const size_t row_offset =
                    (size_t)emitted_comp_row * (size_t)kHeadDim;
                for (int rank = 0; rank < kGpus; ++rank) {
                    src[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                    streams[rank] = opt.decode_cudagraph_gate
                        ? (void *)ranks[rank].stream
                        : nullptr;
                }
                const int store_rc = opt.decode_cudagraph_gate
                    ? ds4_v100_tp_runtime_kv_rows_store_f32_device_streams(
                          rt, layer, 0, (uint32_t)opt.slots, opt.position,
                          DS4_V100_TP_KV_ROW_ATTN, src,
                          (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                          streams, err, sizeof(err))
                    : ds4_v100_tp_runtime_kv_rows_store_f32_device(
                          rt, layer, 0, (uint32_t)opt.slots, opt.position,
                          DS4_V100_TP_KV_ROW_ATTN, src,
                          (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                          err, sizeof(err));
                if (store_rc != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_compressed_store_failed\t"
                                 "layer\t%d\tmode\tbatched\t%s\n",
                                 layer, err);
                    return 16;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    const void *src[kGpus] = {};
                    const size_t row_offset =
                        ((size_t)slot * (size_t)kBoundedCompRows +
                         (size_t)emitted_comp_row) *
                        (size_t)kHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        src[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                    }
                    if (ds4_v100_tp_runtime_kv_row_store_f32_device(
                            rt, layer, slot, opt.position, DS4_V100_TP_KV_ROW_ATTN,
                            src, err, sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_compressed_store_failed\t"
                                     "layer\t%d\tslot\t%u\t%s\n",
                                     layer, slot, err);
                        return 16;
                    }
                }
            }
            current_store = 1;
        }
        sync_typed_kv_boundary(opt, ranks);
        int current_load = 0;
        if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
            current_store) {
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                void *dst[kGpus] = {};
                void *streams[kGpus] = {};
                const size_t row_offset =
                    (size_t)emitted_comp_row * (size_t)kHeadDim;
                for (int rank = 0; rank < kGpus; ++rank) {
                    dst[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                    streams[rank] = opt.decode_cudagraph_gate
                        ? (void *)ranks[rank].stream
                        : nullptr;
                }
                const int load_rc = opt.decode_cudagraph_gate
                    ? ds4_v100_tp_runtime_kv_rows_load_f32_device_streams(
                          rt, layer, 0, (uint32_t)opt.slots, opt.position,
                          DS4_V100_TP_KV_ROW_ATTN, dst,
                          (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                          streams, err, sizeof(err))
                    : ds4_v100_tp_runtime_kv_rows_load_f32_device(
                          rt, layer, 0, (uint32_t)opt.slots, opt.position,
                          DS4_V100_TP_KV_ROW_ATTN, dst,
                          (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                          err, sizeof(err));
                if (load_rc != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_compressed_load_failed\t"
                                 "layer\t%d\tmode\tbatched\t%s\n",
                                 layer, err);
                    return 17;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    void *dst[kGpus] = {};
                    const size_t row_offset =
                        ((size_t)slot * (size_t)kBoundedCompRows +
                         (size_t)emitted_comp_row) *
                        (size_t)kHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        dst[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                    }
                    if (ds4_v100_tp_runtime_kv_row_load_f32_device(
                            rt, layer, slot, opt.position, DS4_V100_TP_KV_ROW_ATTN,
                            dst, err, sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_compressed_load_failed\t"
                                     "layer\t%d\tslot\t%u\t%s\n",
                                     layer, slot, err);
                        return 17;
                    }
                }
            }
            current_load = 1;
        }
        sync_typed_kv_boundary(opt, ranks);
        if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate ||
            !current_store) {
            for (int rank = 0; rank < kGpus; ++rank) {
                ranks[rank].attn_comp_row_loaded_layers[layer][emitted_comp_row] = true;
                ranks[rank].attn_comp_row_loaded_position_layers[layer][emitted_comp_row] =
                    opt.position;
            }
        }
        if (!opt.true_ds4_attention_typed_kv_quiet_gate) {
            std::printf("tp_ep_true_attention_typed_kv_compressed\tlayer\t%d\t"
                        "slots\t%d\tratio\t%d\tposition\t%llu\t"
                        "bounded_row\t%u\tphysical_row\t%llu\tlogical_cols\t%u\t"
                        "logical_row_bytes\t%llu\trow_bytes_per_gpu\t%llu\t"
                        "current_store\t%d\tcurrent_load\t%d\tPASS\n",
                        layer, opt.slots, ratio, (unsigned long long)opt.position,
                        emitted_comp_row, (unsigned long long)view.physical_row,
                        view.logical_cols, (unsigned long long)view.logical_row_bytes,
                        (unsigned long long)view.row_bytes[0], current_store,
                        current_load);
        }
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_typed_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }

    TensorF32Stats index_q_stats;
    TensorF32Stats index_w_stats;
    TensorF32Stats index_kv_stats;
    TensorF32Stats index_gate_stats;
    if (opt.true_ds4_indexer_attention_gate && ratio == 4) {
        t_stage = std::chrono::steady_clock::now();
        if (ops->indexer_attn_q_b.cols != 1024 ||
            ops->indexer_attn_q_b.rows_per_gpu != (kIndexerHead * kIndexerHeadDim) / kGpus ||
            ops->indexer_proj.cols != kHidden ||
            ops->indexer_proj.rows_per_gpu != kIndexerHead / kGpus ||
            ops->indexer_compress_kv.cols != kHidden ||
            ops->indexer_compress_kv.rows_per_gpu != (2 * kIndexerHeadDim) / kGpus ||
            ops->indexer_compress_gate.cols != kHidden ||
            ops->indexer_compress_gate.rows_per_gpu != (2 * kIndexerHeadDim) / kGpus) {
            return 5;
        }
        const uint64_t q_a_elems = (uint64_t)opt.slots * 1024ull;
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            const float *current_src =
                direct_current_input_fill ? hc->d_attn_normed : r.d_current_full;
            if (!ops->indexer_attn_q_b.d_x_half[(size_t)rank] ||
                !ops->indexer_proj.d_x_half[(size_t)rank] ||
                !ops->indexer_compress_kv.d_x_half[(size_t)rank] ||
                !ops->indexer_compress_gate.d_x_half[(size_t)rank]) {
                return 6;
            }
            fill_dense_input_half_from_tensor_kernel<<<
                (unsigned int)((q_a_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->indexer_attn_q_b.d_x_half[(size_t)rank],
                             hc->d_q_a_normed, 1024u, (uint32_t)opt.slots);
            if (!fused_ratio4_current_fill) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->indexer_proj.d_x_half[(size_t)rank],
                                 current_src, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->indexer_compress_kv.d_x_half[(size_t)rank],
                                 current_src, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->indexer_compress_gate.d_x_half[(size_t)rank],
                                 current_src, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
            }
            CHECK_CUDA(cudaGetLastError());
        }
        if (dense_event_wait || graph_event_order) {
            if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 23;
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_input_fill_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
        if (launch_resident_f8_dense(opt, ops->indexer_attn_q_b, ranks) != 0 ||
            launch_resident_f8_dense(opt, ops->indexer_proj, ranks) != 0 ||
            launch_resident_f8_dense(opt, ops->indexer_compress_kv, ranks) != 0 ||
            launch_resident_f8_dense(opt, ops->indexer_compress_gate, ranks) != 0) {
            return 7;
        }
        if (graph_event_order) {
            if (enqueue_control_wait_after_dense_streams(
                    opt, ranks, control_stream) != 0) return 26;
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                cudaStream_t stream = ranks[rank].dense_stream ? ranks[rank].dense_stream
                                                               : ranks[rank].stream;
                CHECK_CUDA(cudaStreamSynchronize(stream));
                if (!skip_dense_stats) {
                merge_tensor_stats(&index_q_stats,
                                   collect_tensor_f32_stats(
                                       ops->indexer_attn_q_b.d_out[(size_t)rank],
                                       (size_t)opt.slots *
                                           (size_t)ops->indexer_attn_q_b.rows_per_gpu,
                                       stream));
                merge_tensor_stats(&index_w_stats,
                                   collect_tensor_f32_stats(
                                       ops->indexer_proj.d_out[(size_t)rank],
                                       (size_t)opt.slots *
                                           (size_t)ops->indexer_proj.rows_per_gpu,
                                       stream));
                merge_tensor_stats(&index_kv_stats,
                                   collect_tensor_f32_stats(
                                       ops->indexer_compress_kv.d_out[(size_t)rank],
                                       (size_t)opt.slots *
                                           (size_t)ops->indexer_compress_kv.rows_per_gpu,
                                       stream));
                merge_tensor_stats(&index_gate_stats,
                                   collect_tensor_f32_stats(
                                       ops->indexer_compress_gate.d_out[(size_t)rank],
                                       (size_t)opt.slots *
                                           (size_t)ops->indexer_compress_gate.rows_per_gpu,
                                       stream));
                }
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_dense_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
        if (!hc->d_indexer_q_full || !hc->d_indexer_w_full) return 13;
        if (!hc->d_index_comp_kv_full || !hc->d_index_comp_score_full ||
            !hc->d_indexer_compress_ape[layer] ||
            !hc->d_indexer_compress_norm[layer]) {
            return 11;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            gather_dense_shard_to_full_kernel<<<
                (unsigned int)(((uint64_t)opt.slots *
                                    (uint64_t)ops->indexer_attn_q_b.rows_per_gpu +
                                block - 1) /
                               block),
                block, 0, control_stream>>>(
                hc->d_indexer_q_full,
                ops->indexer_attn_q_b.d_out[(size_t)rank], rank,
                (uint32_t)ops->indexer_attn_q_b.rows_per_gpu,
                (uint32_t)(kIndexerHead * kIndexerHeadDim),
                (uint32_t)opt.slots);
            gather_dense_shard_to_full_kernel<<<
                (unsigned int)(((uint64_t)opt.slots *
                                    (uint64_t)ops->indexer_proj.rows_per_gpu +
                                block - 1) /
                               block),
                block, 0, control_stream>>>(
                hc->d_indexer_w_full,
                ops->indexer_proj.d_out[(size_t)rank], rank,
                (uint32_t)ops->indexer_proj.rows_per_gpu,
                (uint32_t)kIndexerHead, (uint32_t)opt.slots);
            gather_dense_shard_to_full_kernel<<<
                (unsigned int)(((uint64_t)opt.slots *
                                    (uint64_t)ops->indexer_compress_kv.rows_per_gpu +
                                block - 1) /
                               block),
                block, 0, control_stream>>>(
                hc->d_index_comp_kv_full,
                ops->indexer_compress_kv.d_out[(size_t)rank], rank,
                (uint32_t)ops->indexer_compress_kv.rows_per_gpu,
                (uint32_t)kIndexCompWidth, (uint32_t)opt.slots);
            gather_dense_shard_to_full_kernel<<<
                (unsigned int)(((uint64_t)opt.slots *
                                    (uint64_t)ops->indexer_compress_gate.rows_per_gpu +
                                block - 1) /
                               block),
                block, 0, control_stream>>>(
                hc->d_index_comp_score_full,
                ops->indexer_compress_gate.d_out[(size_t)rank], rank,
                (uint32_t)ops->indexer_compress_gate.rows_per_gpu,
                (uint32_t)kIndexCompWidth, (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        if (!graph_event_order) {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        if (opt.true_ds4_attention_rope_gate) {
            rope_tail_rows_kernel<<<
                (unsigned int)(opt.slots * kIndexerHead), 64, 0,
                control_stream>>>(
                hc->d_indexer_q_full, (uint32_t)(opt.slots * kIndexerHead),
                (uint32_t)kIndexerHeadDim, (uint32_t)kRotaryDim,
                (uint32_t)opt.position, kRopeOrigCtx, 0, kCompressRopeFreqBase,
                comp_freq_scale, comp_ext_factor, comp_attn_factor,
                kRopeYarnBetaFast, kRopeYarnBetaSlow);
            CHECK_CUDA(cudaGetLastError());
            if (!graph_event_order) {
                CHECK_CUDA(cudaDeviceSynchronize());
            }
        }
        if (graph_event_order) {
            if (enqueue_rank_streams_wait_after_control(
                    opt, ranks, control_stream) != 0) return 27;
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_gather_rope_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
        const size_t index_bytes =
            (size_t)opt.slots * kIndexCompWidth * sizeof(float);
        const uint64_t index_elems =
            (uint64_t)opt.slots * (uint64_t)kIndexCompWidth;
        if (!graph_event_order) {
            void *kv_dsts[kGpus] = {};
            void *score_dsts[kGpus] = {};
            for (int rank = 0; rank < kGpus; ++rank) {
                kv_dsts[rank] = ranks[rank].d_index_comp_kv_cur;
                score_dsts[rank] = ranks[rank].d_index_comp_score_cur;
            }
            if (nccl_broadcast_bytes_from_rank0(
                    ranks, hc->d_index_comp_kv_full, kv_dsts, index_bytes,
                    "index_comp_kv_cur") != 0 ||
                nccl_broadcast_bytes_from_rank0(
                    ranks, hc->d_index_comp_score_full, score_dsts,
                    index_bytes, "index_comp_score_cur") != 0) {
                return 28;
            }
        }
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            if (!r.d_index_comp_kv_cur || !r.d_index_comp_score_cur ||
                !r.d_index_comp_state_kv || !r.d_index_comp_state_score ||
                !r.d_index_comp_rows || !r.d_indexer_scores ||
                !r.d_indexer_topk) {
                return 12;
            }
            if (graph_event_order) {
                enqueue_graph_f32_copy_from_device0(
                    opt, r, rank, r.d_index_comp_kv_cur,
                    hc->d_index_comp_kv_full, index_elems, r.stream, block);
                enqueue_graph_f32_copy_from_device0(
                    opt, r, rank, r.d_index_comp_score_cur,
                    hc->d_index_comp_score_full, index_elems, r.stream, block);
            }
            compressor_store_slots_kernel<<<
                (unsigned int)(((uint64_t)opt.slots * kIndexCompWidth +
                                block - 1) /
                               block),
                block, 0, r.stream>>>(
                r.d_index_comp_kv_cur, r.d_index_comp_score_cur,
                r.d_index_comp_state_kv, r.d_index_comp_state_score,
                hc->d_indexer_compress_ape[layer], (uint32_t)opt.slots,
                (uint32_t)kIndexerHeadDim, 4u, (uint32_t)opt.position,
                (uint32_t)kIndexCompStateRows, (uint32_t)kIndexCompWidth);
            if (emitted) {
                const uint32_t comp_row =
                    r.index_comp_rows_written_layers[layer] %
                    (uint32_t)kBoundedCompRows;
                r.index_comp_row_position_layers[layer][comp_row] = opt.position;
                r.index_comp_row_loaded_layers[layer][comp_row] = false;
                const uint32_t visible_after =
                    std::min(r.index_comp_rows_written_layers[layer] + 1u,
                             (uint32_t)kBoundedCompRows);
                if (fused_pool_norm_rope_round) {
                    compressor_pool_norm_rope_round_emit_slots_kernel<<<
                        (unsigned int)opt.slots, 256, 0, r.stream>>>(
                        r.d_index_comp_rows, r.d_index_comp_state_kv,
                        r.d_index_comp_state_score,
                        hc->d_indexer_compress_norm[layer],
                        (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, 4u,
                        comp_row, (uint32_t)kBoundedCompRows,
                        (uint32_t)kIndexCompStateRows,
                        (uint32_t)kIndexCompWidth, 1.0e-6f,
                        (uint32_t)kRotaryDim,
                        (uint32_t)(opt.position + 1ull - 4ull),
                        kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                        comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                        kRopeYarnBetaSlow);
                } else if (fused_pool_norm) {
                    compressor_pool_norm_emit_slots_kernel<<<
                        (unsigned int)opt.slots, 256, 0, r.stream>>>(
                        r.d_index_comp_rows, r.d_index_comp_state_kv,
                        r.d_index_comp_state_score,
                        hc->d_indexer_compress_norm[layer],
                        (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, 4u,
                        comp_row, (uint32_t)kBoundedCompRows,
                        (uint32_t)kIndexCompStateRows,
                        (uint32_t)kIndexCompWidth, 1.0e-6f);
                } else {
                    compressor_pool_emit_slots_kernel<<<
                        dim3((unsigned int)((kIndexerHeadDim + block - 1) / block),
                             (unsigned int)opt.slots, 1u),
                        block, 0, r.stream>>>(
                        r.d_index_comp_rows, r.d_index_comp_state_kv,
                        r.d_index_comp_state_score, (uint32_t)opt.slots,
                        (uint32_t)kIndexerHeadDim, 4u, comp_row,
                        (uint32_t)kBoundedCompRows,
                        (uint32_t)kIndexCompStateRows,
                        (uint32_t)kIndexCompWidth);
                    compressor_norm_emit_slots_kernel<<<(unsigned int)opt.slots,
                                                        256, 0, r.stream>>>(
                        r.d_index_comp_rows, hc->d_indexer_compress_norm[layer],
                        (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, comp_row,
                        (uint32_t)kBoundedCompRows, 1.0e-6f);
                }
                if (fused_pool_norm_rope_round) {
                    // RoPE and F16 rounding were already applied by the fused emit.
                } else if (fused_rope_round) {
                    rope_tail_round_comp_emit_slots_kernel<<<
                        (unsigned int)opt.slots, 256, 0, r.stream>>>(
                        r.d_index_comp_rows, (uint32_t)opt.slots,
                        (uint32_t)kIndexerHeadDim, (uint32_t)kRotaryDim,
                        comp_row, (uint32_t)kBoundedCompRows,
                        (uint32_t)(opt.position + 1ull - 4ull),
                        kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                        comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                        kRopeYarnBetaSlow);
                } else {
                    if (opt.true_ds4_attention_rope_gate) {
                        rope_tail_comp_emit_slots_kernel<<<
                            (unsigned int)opt.slots, 64, 0, r.stream>>>(
                            r.d_index_comp_rows, (uint32_t)opt.slots,
                            (uint32_t)kIndexerHeadDim, (uint32_t)kRotaryDim,
                            comp_row, (uint32_t)kBoundedCompRows,
                            (uint32_t)(opt.position + 1ull - 4ull),
                            kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                            comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                            kRopeYarnBetaSlow);
                    }
                    round_comp_emit_slots_kernel<<<
                        (unsigned int)(((uint64_t)opt.slots *
                                            kIndexerHeadDim +
                                        block - 1) /
                                       block),
                        block, 0, r.stream>>>(
                        r.d_index_comp_rows, (uint32_t)opt.slots,
                        (uint32_t)kIndexerHeadDim, comp_row,
                        (uint32_t)kBoundedCompRows);
                }
                if (rank == 0 && !opt.true_ds4_attention_typed_kv_indexer_gate) {
                    indexer_score_bounded_rows_slots_kernel<<<
                        (unsigned int)opt.slots, 256, 0, r.stream>>>(
                        r.d_indexer_scores, r.d_indexer_topk,
                        hc->d_indexer_q_full, hc->d_indexer_w_full,
                        r.d_index_comp_rows, (uint32_t)opt.slots,
                        visible_after, (uint32_t)kBoundedCompRows,
                        (uint32_t)kIndexerTopK,
                        1.0f / sqrtf((float)(kIndexerHead * kIndexerHeadDim)));
                } else if (!opt.true_ds4_attention_typed_kv_indexer_gate) {
                    seed_single_topk_kernel<<<(unsigned int)opt.slots, 256, 0,
                                               r.stream>>>(
                        r.d_indexer_scores, r.d_indexer_topk,
                        (uint32_t)opt.slots, (uint32_t)kIndexerTopK);
                }
                r.index_comp_rows_written_layers[layer]++;
            }
            CHECK_CUDA(cudaGetLastError());
        }
        if (!graph_event_order) {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_state_emit_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
        if (opt.true_ds4_attention_typed_kv_indexer_gate && emitted) {
            if (!rt) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_typed_kv_indexer_failed\t"
                             "layer\t%d\treason\tmissing_tp_runtime\n",
                             layer);
                return 18;
            }
            char err[512] = {0};
            ds4_v100_tp_kv_row_view view;
            if (ds4_v100_tp_runtime_kv_row_view(
                    rt, layer, 0, opt.position, DS4_V100_TP_KV_ROW_INDEXER,
                    &view, err, sizeof(err)) != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_typed_kv_indexer_view_failed\t"
                             "layer\t%d\t%s\n",
                             layer, err);
                return 19;
            }
            const uint32_t bounded_row =
                (ranks[0].index_comp_rows_written_layers[layer] +
                 (uint32_t)kBoundedCompRows - 1u) %
                (uint32_t)kBoundedCompRows;
            const uint32_t visible_after =
                std::min(ranks[0].index_comp_rows_written_layers[layer],
                         (uint32_t)kBoundedCompRows);
            int current_store = 0;
            if (!opt.true_ds4_attention_typed_kv_skip_indexer_store_gate) {
                if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                    const void *src[kGpus] = {};
                    void *streams[kGpus] = {};
                    const size_t row_offset =
                        (size_t)bounded_row * (size_t)kIndexerHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        src[rank] = ranks[rank].d_index_comp_rows + row_offset;
                        streams[rank] = opt.decode_cudagraph_gate
                            ? (void *)ranks[rank].stream
                            : nullptr;
                    }
                    const int store_rc = opt.decode_cudagraph_gate
                        ? ds4_v100_tp_runtime_kv_rows_store_f32_device_streams(
                              rt, layer, 0, (uint32_t)opt.slots, opt.position,
                              DS4_V100_TP_KV_ROW_INDEXER, src,
                              (uint64_t)kBoundedCompRows * (uint64_t)kIndexerHeadDim,
                              streams, err, sizeof(err))
                        : ds4_v100_tp_runtime_kv_rows_store_f32_device(
                              rt, layer, 0, (uint32_t)opt.slots, opt.position,
                              DS4_V100_TP_KV_ROW_INDEXER, src,
                              (uint64_t)kBoundedCompRows * (uint64_t)kIndexerHeadDim,
                              err, sizeof(err));
                    if (store_rc != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_indexer_store_failed\t"
                                     "layer\t%d\tmode\tbatched\t%s\n",
                                     layer, err);
                        return 20;
                    }
                } else {
                    for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                        const void *src[kGpus] = {};
                        const size_t row_offset =
                            ((size_t)slot * (size_t)kBoundedCompRows +
                             (size_t)bounded_row) *
                            (size_t)kIndexerHeadDim;
                        for (int rank = 0; rank < kGpus; ++rank) {
                            src[rank] = ranks[rank].d_index_comp_rows + row_offset;
                        }
                        if (ds4_v100_tp_runtime_kv_row_store_f32_device(
                                rt, layer, slot, opt.position,
                                DS4_V100_TP_KV_ROW_INDEXER, src, err,
                                sizeof(err)) != 0) {
                            std::fprintf(stderr,
                                         "tp_ep_true_attention_typed_kv_indexer_store_failed\t"
                                         "layer\t%d\tslot\t%u\t%s\n",
                                         layer, slot, err);
                            return 20;
                        }
                    }
                }
                current_store = 1;
            }
            sync_typed_kv_boundary(opt, ranks);
            int current_load = 0;
            if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
                current_store) {
                if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                    void *dst[kGpus] = {};
                    void *streams[kGpus] = {};
                    const size_t row_offset =
                        (size_t)bounded_row * (size_t)kIndexerHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        dst[rank] = ranks[rank].d_index_comp_rows + row_offset;
                        streams[rank] = opt.decode_cudagraph_gate
                            ? (void *)ranks[rank].stream
                            : nullptr;
                    }
                    const int load_rc = opt.decode_cudagraph_gate
                        ? ds4_v100_tp_runtime_kv_rows_load_f32_device_streams(
                              rt, layer, 0, (uint32_t)opt.slots, opt.position,
                              DS4_V100_TP_KV_ROW_INDEXER, dst,
                              (uint64_t)kBoundedCompRows * (uint64_t)kIndexerHeadDim,
                              streams, err, sizeof(err))
                        : ds4_v100_tp_runtime_kv_rows_load_f32_device(
                              rt, layer, 0, (uint32_t)opt.slots, opt.position,
                              DS4_V100_TP_KV_ROW_INDEXER, dst,
                              (uint64_t)kBoundedCompRows * (uint64_t)kIndexerHeadDim,
                              err, sizeof(err));
                    if (load_rc != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_indexer_load_failed\t"
                                     "layer\t%d\tmode\tbatched\t%s\n",
                                     layer, err);
                        return 21;
                    }
                } else {
                    for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                        void *dst[kGpus] = {};
                        const size_t row_offset =
                            ((size_t)slot * (size_t)kBoundedCompRows +
                             (size_t)bounded_row) *
                            (size_t)kIndexerHeadDim;
                        for (int rank = 0; rank < kGpus; ++rank) {
                            dst[rank] = ranks[rank].d_index_comp_rows + row_offset;
                        }
                        if (ds4_v100_tp_runtime_kv_row_load_f32_device(
                                rt, layer, slot, opt.position,
                                DS4_V100_TP_KV_ROW_INDEXER, dst, err,
                                sizeof(err)) != 0) {
                            std::fprintf(stderr,
                                         "tp_ep_true_attention_typed_kv_indexer_load_failed\t"
                                         "layer\t%d\tslot\t%u\t%s\n",
                                         layer, slot, err);
                            return 21;
                        }
                    }
                }
                current_load = 1;
            }
            sync_typed_kv_boundary(opt, ranks);
            if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate ||
                !current_store) {
                for (int rank = 0; rank < kGpus; ++rank) {
                    ranks[rank].index_comp_row_loaded_layers[layer][bounded_row] = true;
                    ranks[rank].index_comp_row_loaded_position_layers[layer][bounded_row] =
                        opt.position;
                }
            }
            CHECK_CUDA(cudaSetDevice(ranks[0].device));
            indexer_score_bounded_rows_slots_kernel<<<
                (unsigned int)opt.slots, 256, 0, ranks[0].stream>>>(
                ranks[0].d_indexer_scores, ranks[0].d_indexer_topk,
                hc->d_indexer_q_full, hc->d_indexer_w_full,
                ranks[0].d_index_comp_rows, (uint32_t)opt.slots, visible_after,
                (uint32_t)kBoundedCompRows, (uint32_t)kIndexerTopK,
                1.0f / sqrtf((float)(kIndexerHead * kIndexerHeadDim)));
            CHECK_CUDA(cudaGetLastError());
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                seed_single_topk_kernel<<<(unsigned int)opt.slots, 256, 0,
                                           ranks[rank].stream>>>(
                    ranks[rank].d_indexer_scores, ranks[rank].d_indexer_topk,
                    (uint32_t)opt.slots, (uint32_t)kIndexerTopK);
                CHECK_CUDA(cudaGetLastError());
            }
            if (graph_event_order) {
                const int slot = next_graph_order_event_slot(ranks);
                CHECK_CUDA(cudaSetDevice(ranks[0].device));
                cudaEvent_t ev = graph_stream_done_event(ranks[0], slot);
                if (!ev) return 28;
                CHECK_CUDA(cudaEventRecord(ev, ranks[0].stream));
                for (int rank = 1; rank < kGpus; ++rank) {
                    CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                    CHECK_CUDA(cudaStreamWaitEvent(ranks[rank].stream,
                                                   ev, 0));
                }
            } else {
                for (int rank = 0; rank < kGpus; ++rank) {
                    CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                    CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
                }
            }
            if (!opt.true_ds4_attention_typed_kv_quiet_gate) {
                std::printf("tp_ep_true_attention_typed_kv_indexer\tlayer\t%d\t"
                            "slots\t%d\tratio\t%d\tposition\t%llu\t"
                            "bounded_row\t%u\tvisible_rows\t%u\tphysical_row\t%llu\t"
                            "logical_cols\t%u\tlogical_row_bytes\t%llu\t"
                            "row_bytes_per_gpu\t%llu\tcurrent_store\t%d\t"
                            "current_load\t%d\tPASS\n",
                            layer, opt.slots, ratio,
                            (unsigned long long)opt.position, bounded_row,
                            visible_after, (unsigned long long)view.physical_row,
                            view.logical_cols,
                            (unsigned long long)view.logical_row_bytes,
                            (unsigned long long)view.row_bytes[0], current_store,
                            current_load);
            }
        }
        if (emitted && ranks[0].d_indexer_topk) {
            const uint64_t topk_elems =
                (uint64_t)opt.slots * (uint64_t)kIndexerTopK;
            const size_t topk_bytes = (size_t)topk_elems * sizeof(uint32_t);
            if (!graph_event_order) {
                void *topk_dsts[kGpus] = {};
                for (int rank = 0; rank < kGpus; ++rank) {
                    topk_dsts[rank] = ranks[rank].d_indexer_topk;
                }
                if (nccl_broadcast_bytes_from_rank0(
                        ranks, ranks[0].d_indexer_topk, topk_dsts,
                        topk_bytes, "indexer_topk_emit") != 0) {
                    return 29;
                }
            }
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                if (graph_event_order) {
                    copy_u32_kernel<<<
                        (unsigned int)((topk_elems + block - 1) / block),
                        block, 0, ranks[rank].stream>>>(
                        ranks[rank].d_indexer_topk, ranks[0].d_indexer_topk,
                        topk_elems);
                    CHECK_CUDA(cudaGetLastError());
                }
            }
            if (!graph_event_order) {
                for (int rank = 1; rank < kGpus; ++rank) {
                    CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                    CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
                }
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_typed_score_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
    }

    const auto diff_start = std::chrono::steady_clock::now();
    const int diff_rc = run_true_ds4_compressed_reference_diff_gate(
        opt, hc, ranks, layer, ratio, comp_width, emitted, emitted_comp_row,
        visible);
    if (diff_rc != 0) return diff_rc;
    reference_diff_ms =
        elapsed_ms(diff_start, std::chrono::steady_clock::now());
    const auto shift_start = std::chrono::steady_clock::now();
    if (emitted && ratio == 4) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            compressor_shift_ratio4_slots_kernel<<<
                (unsigned int)(((uint64_t)opt.slots * 4ull *
                                    (uint64_t)comp_width +
                                block - 1) /
                               block),
                block, 0, r.stream>>>(
                r.d_attn_comp_state_kv, r.d_attn_comp_state_score,
                (uint32_t)opt.slots, (uint32_t)comp_width,
                (uint32_t)comp_state_rows, (uint32_t)comp_state_width);
            if (opt.true_ds4_indexer_attention_gate && r.d_index_comp_state_kv &&
                r.d_index_comp_state_score) {
                compressor_shift_ratio4_slots_kernel<<<
                    (unsigned int)(((uint64_t)opt.slots * 4ull *
                                        (uint64_t)kIndexCompWidth +
                                    block - 1) /
                                   block),
                    block, 0, r.stream>>>(
                    r.d_index_comp_state_kv, r.d_index_comp_state_score,
                    (uint32_t)opt.slots, (uint32_t)kIndexCompWidth,
                    (uint32_t)kIndexCompStateRows,
                    (uint32_t)kIndexCompWidth);
            }
            CHECK_CUDA(cudaGetLastError());
        }
        if (!graph_event_order) {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }
    ratio_shift_ms =
        elapsed_ms(shift_start, std::chrono::steady_clock::now());

    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    std::printf("tp_ep_compressed_kv_projection\tlayer\t%d\tslots\t%d\t"
                "ratio\t%d\temitted_compressed_rows\t%u\t"
                "visible_compressed_rows\t%u\tindexer_topk_count\t%u\t"
                "attn_comp_width\t%d\tattn_kv_max\t%.9g\tattn_kv_bad\t%d\t"
                "attn_gate_max\t%.9g\tattn_gate_bad\t%d\t"
                "index_q_max\t%.9g\tindex_q_bad\t%d\t"
                "index_w_max\t%.9g\tindex_w_bad\t%d\t"
                "index_kv_max\t%.9g\tindex_kv_bad\t%d\t"
                "index_gate_max\t%.9g\tindex_gate_bad\t%d\t"
                "attn_input_fill_ms\t%.6f\tattn_dense_ms\t%.6f\t"
                "attn_gather_ms\t%.6f\tattn_state_emit_ms\t%.6f\t"
                "attn_typed_ms\t%.6f\tindexer_input_fill_ms\t%.6f\t"
                "indexer_dense_ms\t%.6f\tindexer_gather_rope_ms\t%.6f\t"
                "indexer_state_emit_ms\t%.6f\tindexer_typed_score_ms\t%.6f\t"
                "reference_diff_ms\t%.6f\tratio_shift_ms\t%.6f\t"
                "direct_input_fill\t%d\tdense_event_wait\t%d\t"
                "skip_dense_stats\t%d\t"
                "fused_attn_input_fill\t%d\t"
                "fused_input_fill\t%d\tfused_rope_round\t%d\t"
                "fused_pool_norm\t%d\tfused_pool_norm_rope_round\t%d\t"
                "ms\t%.6f\tPASS\n",
                layer, opt.slots, ratio, emitted, visible, indexer_topk,
                comp_width, attn_kv_stats.max_abs, attn_kv_stats.finite_bad,
                attn_gate_stats.max_abs, attn_gate_stats.finite_bad,
                index_q_stats.max_abs, index_q_stats.finite_bad,
                index_w_stats.max_abs, index_w_stats.finite_bad,
                index_kv_stats.max_abs, index_kv_stats.finite_bad,
                index_gate_stats.max_abs, index_gate_stats.finite_bad,
                attn_input_fill_ms, attn_dense_ms, attn_gather_ms,
                attn_state_emit_ms, attn_typed_ms, indexer_input_fill_ms,
                indexer_dense_ms, indexer_gather_rope_ms,
                indexer_state_emit_ms, indexer_typed_score_ms,
                reference_diff_ms, ratio_shift_ms,
                direct_current_input_fill ? 1 : 0,
                dense_event_wait ? 1 : 0,
                skip_dense_stats ? 1 : 0,
                fused_attn_current_fill ? 1 : 0,
                fused_ratio4_current_fill ? 1 : 0,
                fused_rope_round ? 1 : 0,
                fused_pool_norm ? 1 : 0,
                fused_pool_norm_rope_round ? 1 : 0, ms);
    return (!skip_dense_stats &&
            (attn_kv_stats.finite_bad || attn_gate_stats.finite_bad ||
            index_q_stats.finite_bad || index_w_stats.finite_bad ||
             index_kv_stats.finite_bad || index_gate_stats.finite_bad)) ? 8 : 0;
}

int run_true_ds4_attention_state_update(const Options &opt,
                                        SharedHcControls *hc,
                                        const LayerDenseOps *ops,
                                        RankState ranks[kGpus],
                                        ds4_v100_tp_runtime *rt,
                                        int layer) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    if (!hc->d_kv_normed ||
        ops->attn_q_b.rows_per_gpu != kLocalHeads * kHeadDim) {
        return 2;
    }

    const auto start = std::chrono::steady_clock::now();
    const int block = 256;
    const uint32_t raw_row = (uint32_t)(opt.position % kRawSwaRows);
    const uint64_t kv_elems = (uint64_t)opt.slots * (uint64_t)kHeadDim;
    const uint64_t raw_elems =
        (uint64_t)opt.slots * (uint64_t)kRawSwaRows * (uint64_t)kHeadDim;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    const int ratio = ds4_layer_ratio(layer);
    const bool compressed = ratio != 0;
    const float freq_base =
        compressed ? kCompressRopeFreqBase : kRopeFreqBase;
    const float freq_scale =
        compressed && kRopeScaleFactor > 0.0f ? 1.0f / kRopeScaleFactor : 1.0f;
    const float ext_factor =
        compressed && kRopeScaleFactor > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    if (!graph_event_order) {
        void *kv_dsts[kGpus] = {};
        for (int rank = 0; rank < kGpus; ++rank) {
            if (!ranks[rank].d_attn_kv_full) return 3;
            kv_dsts[rank] = ranks[rank].d_attn_kv_full;
        }
        if (nccl_broadcast_bytes_from_rank0(
                ranks, hc->d_kv_normed, kv_dsts,
                (size_t)kv_elems * sizeof(float), "attention_state_kv_full") != 0) {
            return 9;
        }
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_kv_full || !r.d_attn_raw_swa ||
            !ops->attn_q_b.d_out[(size_t)rank]) {
            return 3;
        }
        head_rms_norm_local_heads_kernel<<<
            (unsigned int)(opt.slots * kLocalHeads), 256, 0,
            r.dense_stream ? r.dense_stream : r.stream>>>(
            ops->attn_q_b.d_out[(size_t)rank], (uint32_t)opt.slots,
            (uint32_t)kLocalHeads, (uint32_t)kHeadDim, 1.0e-6f);
        CHECK_CUDA(cudaGetLastError());
        if (opt.true_ds4_attention_rope_gate) {
            rope_tail_rows_kernel<<<
                (unsigned int)(opt.slots * kLocalHeads), 64, 0,
                r.dense_stream ? r.dense_stream : r.stream>>>(
                ops->attn_q_b.d_out[(size_t)rank],
                (uint32_t)(opt.slots * kLocalHeads), (uint32_t)kHeadDim,
                (uint32_t)kRotaryDim, (uint32_t)opt.position,
                compressed ? kRopeOrigCtx : 0u, 0, freq_base, freq_scale,
                ext_factor, attn_factor, kRopeYarnBetaFast,
                kRopeYarnBetaSlow);
            CHECK_CUDA(cudaGetLastError());
        }
        if (graph_event_order) {
            enqueue_graph_f32_copy_from_device0(
                opt, r, rank, r.d_attn_kv_full, hc->d_kv_normed, kv_elems,
                r.stream, block);
        }
        if (opt.true_ds4_attention_rope_gate) {
            rope_tail_rows_kernel<<<
                (unsigned int)opt.slots, 64, 0, r.stream>>>(
                r.d_attn_kv_full, (uint32_t)opt.slots, (uint32_t)kHeadDim,
                (uint32_t)kRotaryDim, (uint32_t)opt.position,
                compressed ? kRopeOrigCtx : 0u, 0, freq_base, freq_scale,
                ext_factor, attn_factor, kRopeYarnBetaFast,
                kRopeYarnBetaSlow);
            CHECK_CUDA(cudaGetLastError());
        }
        if (!opt.true_ds4_attention_typed_kv_raw_gate ||
            opt.true_ds4_attention_typed_kv_skip_current_load_gate) {
            kv_fp8_round_store_raw_swa_kernel<<<
                (unsigned int)((kv_elems + block - 1) / block), block, 0,
                r.stream>>>(
                r.d_attn_raw_swa, r.d_attn_kv_full, (uint32_t)opt.slots,
                (uint32_t)kRawSwaRows, raw_row, (uint32_t)kHeadDim,
                (uint32_t)kRotaryDim);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    if (opt.decode_cudagraph_gate) {
        if (enqueue_rank_streams_wait_after_dense_streams(ranks) != 0) {
            return 8;
        }
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            if (ranks[rank].dense_stream) {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].dense_stream));
            }
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    if (opt.true_ds4_attention_typed_kv_raw_gate) {
        if (!rt) {
            std::fprintf(stderr,
                         "tp_ep_true_attention_typed_kv_raw_failed\tlayer\t%d\t"
                         "reason\tmissing_tp_runtime\n",
                         layer);
            return 4;
        }
        char err[512] = {0};
        ds4_v100_tp_kv_row_view view;
        if (ds4_v100_tp_runtime_kv_row_view(
                rt, layer, 0, opt.position, DS4_V100_TP_KV_ROW_ATTN_RAW, &view,
                err, sizeof(err)) != 0) {
            std::fprintf(stderr,
                         "tp_ep_true_attention_typed_kv_raw_view_failed\tlayer\t%d\t%s\n",
                         layer, err);
            return 5;
        }
        int current_store = 0;
        if (!opt.true_ds4_attention_typed_kv_skip_raw_store_gate) {
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                const void *src[kGpus] = {};
                void *streams[kGpus] = {};
                for (int rank = 0; rank < kGpus; ++rank) {
                    src[rank] = ranks[rank].d_attn_kv_full;
                    streams[rank] = opt.decode_cudagraph_gate
                        ? (void *)ranks[rank].stream
                        : nullptr;
                }
                const int store_rc = opt.decode_cudagraph_gate
                    ? ds4_v100_tp_runtime_kv_rows_store_f32_device_streams(
                          rt, layer, 0, (uint32_t)opt.slots, opt.position,
                          DS4_V100_TP_KV_ROW_ATTN_RAW, src,
                          (uint64_t)kHeadDim, streams, err, sizeof(err))
                    : ds4_v100_tp_runtime_kv_rows_store_f32_device(
                          rt, layer, 0, (uint32_t)opt.slots, opt.position,
                          DS4_V100_TP_KV_ROW_ATTN_RAW, src,
                          (uint64_t)kHeadDim, err, sizeof(err));
                if (store_rc != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_raw_store_failed\t"
                                 "layer\t%d\tmode\tbatched\t%s\n",
                                 layer, err);
                    return 6;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    const void *src[kGpus] = {};
                    for (int rank = 0; rank < kGpus; ++rank) {
                        src[rank] = ranks[rank].d_attn_kv_full +
                                    (size_t)slot * (size_t)kHeadDim;
                    }
                    if (ds4_v100_tp_runtime_kv_row_store_f32_device(
                            rt, layer, slot, opt.position,
                            DS4_V100_TP_KV_ROW_ATTN_RAW, src, err,
                            sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_raw_store_failed\t"
                                     "layer\t%d\tslot\t%u\t%s\n",
                                     layer, slot, err);
                        return 6;
                    }
                }
            }
            current_store = 1;
        }
        sync_typed_kv_boundary(opt, ranks);
        int current_load = 0;
        if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
            current_store) {
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                void *dst[kGpus] = {};
                void *streams[kGpus] = {};
                const size_t row_offset =
                    (size_t)raw_row * (size_t)kHeadDim;
                for (int rank = 0; rank < kGpus; ++rank) {
                    dst[rank] = ranks[rank].d_attn_raw_swa + row_offset;
                    streams[rank] = opt.decode_cudagraph_gate
                        ? (void *)ranks[rank].stream
                        : nullptr;
                }
                const int load_rc = opt.decode_cudagraph_gate
                    ? ds4_v100_tp_runtime_kv_rows_load_f32_device_streams(
                          rt, layer, 0, (uint32_t)opt.slots, opt.position,
                          DS4_V100_TP_KV_ROW_ATTN_RAW, dst,
                          (uint64_t)kRawSwaRows * (uint64_t)kHeadDim,
                          streams, err, sizeof(err))
                    : ds4_v100_tp_runtime_kv_rows_load_f32_device(
                          rt, layer, 0, (uint32_t)opt.slots, opt.position,
                          DS4_V100_TP_KV_ROW_ATTN_RAW, dst,
                          (uint64_t)kRawSwaRows * (uint64_t)kHeadDim,
                          err, sizeof(err));
                if (load_rc != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_raw_load_failed\t"
                                 "layer\t%d\tmode\tbatched\t%s\n",
                                 layer, err);
                    return 7;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    void *dst[kGpus] = {};
                    const size_t row_offset =
                        ((size_t)slot * (size_t)kRawSwaRows + (size_t)raw_row) *
                        (size_t)kHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        dst[rank] = ranks[rank].d_attn_raw_swa + row_offset;
                    }
                    if (ds4_v100_tp_runtime_kv_row_load_f32_device(
                            rt, layer, slot, opt.position,
                            DS4_V100_TP_KV_ROW_ATTN_RAW, dst, err,
                            sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_raw_load_failed\t"
                                     "layer\t%d\tslot\t%u\t%s\n",
                                     layer, slot, err);
                        return 7;
                    }
                }
            }
            current_load = 1;
        }
        sync_typed_kv_boundary(opt, ranks);
        if (!opt.true_ds4_attention_typed_kv_quiet_gate) {
            std::printf("tp_ep_true_attention_typed_kv_raw\tlayer\t%d\tslots\t%d\t"
                        "position\t%llu\tphysical_row\t%llu\traw_row\t%u\tlogical_cols\t%u\t"
                        "logical_row_bytes\t%llu\trow_bytes_per_gpu\t%llu\t"
                        "current_store\t%d\tcurrent_load\t%d\tPASS\n",
                        layer, opt.slots, (unsigned long long)opt.position,
                        (unsigned long long)view.physical_row, raw_row,
                        view.logical_cols, (unsigned long long)view.logical_row_bytes,
                        (unsigned long long)view.row_bytes[0], current_store,
                        current_load);
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    if (!opt.decode_cudagraph_gate && layer <= 2) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("true_attn_q_heads_normed_shard", layer, rank,
                                 ops->attn_q_b.d_out[(size_t)rank],
                                 (size_t)opt.slots * ops->attn_q_b.rows_per_gpu,
                                 ranks[rank].dense_stream ? ranks[rank].dense_stream
                                                          : ranks[rank].stream);
        }
        CHECK_CUDA(cudaSetDevice(ranks[0].device));
        log_tensor_f32_stats("true_attn_raw_swa_rank0", layer, 0,
                             ranks[0].d_attn_raw_swa, (size_t)raw_elems,
                             ranks[0].stream);
    }
    if (opt.true_ds4_attention_saturation_audit_gate) {
        TensorF32Stats q_heads;
        TensorF32Stats kv_rope;
        TensorF32Stats raw_row_stats;
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            const cudaStream_t q_stream =
                ranks[rank].dense_stream ? ranks[rank].dense_stream
                                         : ranks[rank].stream;
            merge_tensor_stats(
                &q_heads,
                collect_tensor_f32_stats(
                    ops->attn_q_b.d_out[(size_t)rank],
                    (size_t)opt.slots * ops->attn_q_b.rows_per_gpu,
                    q_stream));
            merge_tensor_stats(
                &kv_rope,
                collect_tensor_f32_stats(ranks[rank].d_attn_kv_full,
                                         (size_t)kv_elems,
                                         ranks[rank].stream));
            merge_tensor_stats(
                &raw_row_stats,
                collect_raw_swa_row_stats(ranks[rank].d_attn_raw_swa,
                                          (uint32_t)opt.slots,
                                          (uint32_t)kRawSwaRows, raw_row,
                                          (uint32_t)kHeadDim,
                                          ranks[rank].stream));
        }
        std::printf("tp_ep_true_attention_saturation_state\tlayer\t%d\t"
                    "slots\t%d\traw_row\t%u\tq_heads_post_rope_max\t%.9g\t"
                    "q_heads_post_rope_bad\t%d\tkv_post_rope_max\t%.9g\t"
                    "kv_post_rope_bad\t%d\traw_swa_row_max\t%.9g\t"
                    "raw_swa_row_bad\t%d\tPASS\n",
                    layer, opt.slots, raw_row, q_heads.max_abs,
                    q_heads.finite_bad, kv_rope.max_abs, kv_rope.finite_bad,
                    raw_row_stats.max_abs, raw_row_stats.finite_bad);
    }
    if (opt.true_ds4_attention_rope_gate) {
        std::printf("tp_ep_true_attention_rope\tlayer\t%d\tslots\t%d\t"
                    "local_heads\t%d\thead_dim\t%d\trotary_dim\t%d\t"
                    "freq_base\t%.1f\tfreq_scale\t%.9f\tposition\t%llu\tPASS\n",
                    layer, opt.slots, kLocalHeads, kHeadDim, kRotaryDim,
                    freq_base, freq_scale, (unsigned long long)opt.position);
    }
    std::printf("tp_ep_true_attention_state_update\tlayer\t%d\tslots\t%d\t"
                "local_heads\t%d\thead_dim\t%d\traw_rows\t%d\traw_row\t%u\t"
                "kv_width\t%d\tms\t%.6f\tPASS\n",
                layer, opt.slots, kLocalHeads, kHeadDim, kRawSwaRows, raw_row,
                kHeadDim, ms);
    return 0;
}

int run_true_ds4_attention_typed_kv_history_load(const Options &opt,
                                                 SharedHcControls *hc,
                                                 RankState ranks[kGpus],
                                                 ds4_v100_tp_runtime *rt,
                                                 int layer) {
    if (!opt.true_ds4_attention_typed_kv_history_gate) return 0;
    if (!rt || layer < 0 || layer >= 43) return 1;
    const int ratio = ds4_layer_ratio(layer);
    if (ratio == 0) return 0;

    const uint32_t visible_attn =
        std::min(ranks[0].attn_comp_rows_written_layers[layer],
                 (uint32_t)kBoundedCompRows);
    char err[512] = {0};
    int loaded_attn = 0;
    int reloaded_attn = 0;
    for (uint32_t row = 0; row < visible_attn; ++row) {
        const uint64_t pos = ranks[0].attn_comp_row_position_layers[layer][row];
        if (opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
            pos == opt.position) {
            loaded_attn++;
            continue;
        }
        if (ranks[0].attn_comp_row_loaded_layers[layer][row] &&
            ranks[0].attn_comp_row_loaded_position_layers[layer][row] == pos) {
            loaded_attn++;
            continue;
        }
        if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
            void *dst[kGpus] = {};
            void *streams[kGpus] = {};
            const size_t row_offset =
                (size_t)row * (size_t)kHeadDim;
            for (int rank = 0; rank < kGpus; ++rank) {
                dst[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                streams[rank] = opt.decode_cudagraph_gate
                    ? (void *)ranks[rank].stream
                    : nullptr;
            }
            const int load_rc = opt.decode_cudagraph_gate
                ? ds4_v100_tp_runtime_kv_rows_load_f32_device_streams(
                      rt, layer, 0, (uint32_t)opt.slots, pos,
                      DS4_V100_TP_KV_ROW_ATTN, dst,
                      (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                      streams, err, sizeof(err))
                : ds4_v100_tp_runtime_kv_rows_load_f32_device(
                      rt, layer, 0, (uint32_t)opt.slots, pos,
                      DS4_V100_TP_KV_ROW_ATTN, dst,
                      (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                      err, sizeof(err));
            if (load_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_typed_kv_history_attn_load_failed\t"
                             "layer\t%d\trow\t%u\tmode\tbatched\tposition\t%llu\t%s\n",
                             layer, row, (unsigned long long)pos, err);
                return 2;
            }
        } else {
            for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                void *dst[kGpus] = {};
                const size_t row_offset =
                    ((size_t)slot * (size_t)kBoundedCompRows + (size_t)row) *
                    (size_t)kHeadDim;
                for (int rank = 0; rank < kGpus; ++rank) {
                    dst[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                }
                if (ds4_v100_tp_runtime_kv_row_load_f32_device(
                        rt, layer, slot, pos, DS4_V100_TP_KV_ROW_ATTN, dst, err,
                        sizeof(err)) != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_history_attn_load_failed\t"
                                 "layer\t%d\trow\t%u\tslot\t%u\tposition\t%llu\t%s\n",
                                 layer, row, slot, (unsigned long long)pos, err);
                    return 2;
                }
            }
        }
        loaded_attn++;
        reloaded_attn++;
        for (int rank = 0; rank < kGpus; ++rank) {
            ranks[rank].attn_comp_row_loaded_layers[layer][row] = true;
            ranks[rank].attn_comp_row_loaded_position_layers[layer][row] = pos;
        }
    }
    sync_typed_kv_boundary(opt, ranks);

    int loaded_indexer = 0;
    int reloaded_indexer = 0;
    if (opt.true_ds4_indexer_attention_gate && ratio == 4 && visible_attn > 0) {
        if (!hc || !hc->initialized || !hc->d_indexer_q_full ||
            !hc->d_indexer_w_full || !ranks[0].d_indexer_scores ||
            !ranks[0].d_indexer_topk) {
            return 3;
        }
        const uint32_t visible_index =
            std::min(ranks[0].index_comp_rows_written_layers[layer],
                     (uint32_t)kBoundedCompRows);
        for (uint32_t row = 0; row < visible_index; ++row) {
            const uint64_t pos = ranks[0].index_comp_row_position_layers[layer][row];
            if (opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
                pos == opt.position) {
                loaded_indexer++;
                continue;
            }
            if (ranks[0].index_comp_row_loaded_layers[layer][row] &&
                ranks[0].index_comp_row_loaded_position_layers[layer][row] == pos) {
                loaded_indexer++;
                continue;
            }
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                void *dst[kGpus] = {};
                void *streams[kGpus] = {};
                const size_t row_offset =
                    (size_t)row * (size_t)kIndexerHeadDim;
                for (int rank = 0; rank < kGpus; ++rank) {
                    dst[rank] = ranks[rank].d_index_comp_rows + row_offset;
                    streams[rank] = opt.decode_cudagraph_gate
                        ? (void *)ranks[rank].stream
                        : nullptr;
                }
                const int load_rc = opt.decode_cudagraph_gate
                    ? ds4_v100_tp_runtime_kv_rows_load_f32_device_streams(
                          rt, layer, 0, (uint32_t)opt.slots, pos,
                          DS4_V100_TP_KV_ROW_INDEXER, dst,
                          (uint64_t)kBoundedCompRows * (uint64_t)kIndexerHeadDim,
                          streams, err, sizeof(err))
                    : ds4_v100_tp_runtime_kv_rows_load_f32_device(
                          rt, layer, 0, (uint32_t)opt.slots, pos,
                          DS4_V100_TP_KV_ROW_INDEXER, dst,
                          (uint64_t)kBoundedCompRows * (uint64_t)kIndexerHeadDim,
                          err, sizeof(err));
                if (load_rc != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_history_indexer_load_failed\t"
                                 "layer\t%d\trow\t%u\tmode\tbatched\tposition\t%llu\t%s\n",
                                 layer, row, (unsigned long long)pos, err);
                    return 4;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    void *dst[kGpus] = {};
                    const size_t row_offset =
                        ((size_t)slot * (size_t)kBoundedCompRows + (size_t)row) *
                        (size_t)kIndexerHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        dst[rank] = ranks[rank].d_index_comp_rows + row_offset;
                    }
                    if (ds4_v100_tp_runtime_kv_row_load_f32_device(
                            rt, layer, slot, pos, DS4_V100_TP_KV_ROW_INDEXER, dst,
                            err, sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_history_indexer_load_failed\t"
                                     "layer\t%d\trow\t%u\tslot\t%u\tposition\t%llu\t%s\n",
                                     layer, row, slot, (unsigned long long)pos, err);
                        return 4;
                    }
                }
            }
            loaded_indexer++;
            reloaded_indexer++;
            for (int rank = 0; rank < kGpus; ++rank) {
                ranks[rank].index_comp_row_loaded_layers[layer][row] = true;
                ranks[rank].index_comp_row_loaded_position_layers[layer][row] = pos;
            }
        }
        sync_typed_kv_boundary(opt, ranks);
        CHECK_CUDA(cudaSetDevice(ranks[0].device));
        indexer_score_bounded_rows_slots_kernel<<<
            (unsigned int)opt.slots, 256, 0, ranks[0].stream>>>(
            ranks[0].d_indexer_scores, ranks[0].d_indexer_topk,
            hc->d_indexer_q_full, hc->d_indexer_w_full,
            ranks[0].d_index_comp_rows, (uint32_t)opt.slots, visible_index,
            (uint32_t)kBoundedCompRows, (uint32_t)kIndexerTopK,
            1.0f / sqrtf((float)(kIndexerHead * kIndexerHeadDim)));
        CHECK_CUDA(cudaGetLastError());
        for (int rank = 1; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            seed_single_topk_kernel<<<(unsigned int)opt.slots, 256, 0,
                                       ranks[rank].stream>>>(
                ranks[rank].d_indexer_scores, ranks[rank].d_indexer_topk,
                (uint32_t)opt.slots, (uint32_t)kIndexerTopK);
            CHECK_CUDA(cudaGetLastError());
        }
        if (opt.decode_cudagraph_gate) {
            const int slot = next_graph_order_event_slot(ranks);
            CHECK_CUDA(cudaSetDevice(ranks[0].device));
            cudaEvent_t ev = graph_stream_done_event(ranks[0], slot);
            if (!ev) return 5;
            CHECK_CUDA(cudaEventRecord(ev, ranks[0].stream));
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamWaitEvent(ranks[rank].stream,
                                               ev, 0));
            }
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
        const size_t topk_bytes = (size_t)opt.slots * kIndexerTopK * sizeof(uint32_t);
        const uint64_t topk_elems = (uint64_t)opt.slots * (uint64_t)kIndexerTopK;
        const int block = 256;
        if (!opt.decode_cudagraph_gate) {
            void *topk_dsts[kGpus] = {};
            for (int rank = 0; rank < kGpus; ++rank) {
                topk_dsts[rank] = ranks[rank].d_indexer_topk;
            }
            if (nccl_broadcast_bytes_from_rank0(
                    ranks, ranks[0].d_indexer_topk, topk_dsts, topk_bytes,
                    "indexer_topk_history") != 0) {
                return 9;
            }
        }
        for (int rank = 1; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            if (opt.decode_cudagraph_gate) {
                copy_u32_kernel<<<(unsigned int)((topk_elems + block - 1) / block),
                                  block, 0, ranks[rank].stream>>>(
                    ranks[rank].d_indexer_topk, ranks[0].d_indexer_topk,
                    topk_elems);
                CHECK_CUDA(cudaGetLastError());
            }
        }
        if (!opt.decode_cudagraph_gate) {
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }
    sync_typed_kv_boundary(opt, ranks);

    if (!opt.true_ds4_attention_typed_kv_quiet_gate) {
        std::printf("tp_ep_true_attention_typed_kv_history\tlayer\t%d\tslots\t%d\t"
                    "ratio\t%d\tvisible_attn_rows\t%u\tloaded_attn_rows\t%d\t"
                    "loaded_indexer_rows\t%d\treloaded_attn_rows\t%d\t"
                    "reloaded_indexer_rows\t%d\tPASS\n",
                    layer, opt.slots, ratio, visible_attn, loaded_attn,
                    loaded_indexer, reloaded_attn, reloaded_indexer);
    }
    return 0;
}

int run_true_ds4_attention_raw_read(const Options &opt,
                                    SharedHcControls *hc,
                                    const LayerDenseOps *ops,
                                    RankState ranks[kGpus],
                                    int layer) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    if (!hc->d_attn_sinks[layer] ||
        ops->attn_q_b.rows_per_gpu != kLocalHeads * kHeadDim) {
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    const uint32_t raw_row = (uint32_t)(opt.position % kRawSwaRows);
    const uint64_t heads_elems =
        (uint64_t)opt.slots * (uint64_t)kLocalHeads * (uint64_t)kHeadDim;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_raw_swa || !r.d_attn_sinks || !r.d_attn_heads ||
            !ops->attn_q_b.d_out[(size_t)rank]) {
            return 3;
        }
        const size_t sinks_offset = (size_t)rank * (size_t)kLocalHeads;
        if (rank == 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_attn_sinks,
                                       hc->d_attn_sinks[layer] + sinks_offset,
                                       (size_t)kLocalHeads * sizeof(float),
                                       cudaMemcpyDeviceToDevice, r.stream));
        } else {
            void *sinks_dsts[kGpus] = {};
            for (int dst_rank = 0; dst_rank < kGpus; ++dst_rank) {
                sinks_dsts[dst_rank] = ranks[dst_rank].d_attn_sinks;
            }
            if (nccl_broadcast_bytes_from_rank0(
                    ranks, hc->d_attn_sinks[layer] + sinks_offset, sinks_dsts,
                    (size_t)kLocalHeads * sizeof(float),
                    "attention_raw_sinks") != 0) {
                return 4;
            }
        }
        attention_raw_swa_one_row_kernel<<<
            (unsigned int)(opt.slots * kLocalHeads), 256, 0, r.stream>>>(
            r.d_attn_heads, ops->attn_q_b.d_out[(size_t)rank], r.d_attn_raw_swa,
            r.d_attn_sinks, (uint32_t)opt.slots, (uint32_t)kLocalHeads,
            (uint32_t)kHeadDim, (uint32_t)kRawSwaRows, raw_row);
        CHECK_CUDA(cudaGetLastError());
    }
    if (!opt.decode_cudagraph_gate) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    if (!opt.decode_cudagraph_gate && layer <= 2) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("true_attn_raw_read_heads", layer, rank,
                                 ranks[rank].d_attn_heads, (size_t)heads_elems,
                                 ranks[rank].stream);
        }
    }
    std::printf("tp_ep_true_attention_raw_read\tlayer\t%d\tslots\t%d\t"
                "local_heads\t%d\thead_dim\t%d\traw_rows\t%d\traw_row\t%u\t"
                "ms\t%.6f\tPASS\n",
                layer, opt.slots, kLocalHeads, kHeadDim, kRawSwaRows, raw_row, ms);
    return 0;
}

int run_true_ds4_attention_raw_window(const Options &opt,
                                      SharedHcControls *hc,
                                      const LayerDenseOps *ops,
                                      RankState ranks[kGpus],
                                      int layer) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    if (!hc->d_attn_sinks[layer] ||
        ops->attn_q_b.rows_per_gpu != kLocalHeads * kHeadDim) {
        return 2;
    }
    const uint32_t valid_rows =
        std::max(1u, std::min(opt.true_ds4_attention_raw_valid_rows,
                              (uint32_t)kRawSwaRows));
    const int ratio = ds4_layer_ratio(layer);
    const auto start = std::chrono::steady_clock::now();
    const uint32_t raw_row = (uint32_t)(opt.position % kRawSwaRows);
    const uint64_t heads_elems =
        (uint64_t)opt.slots * (uint64_t)kLocalHeads * (uint64_t)kHeadDim;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_raw_swa || !r.d_attn_sinks || !r.d_attn_heads ||
            !ops->attn_q_b.d_out[(size_t)rank]) {
            return 3;
        }
        const size_t sinks_offset = (size_t)rank * (size_t)kLocalHeads;
        if (graph_event_order) {
            enqueue_graph_f32_copy_from_device0(
                opt, r, rank, r.d_attn_sinks,
                hc->d_attn_sinks[layer] + sinks_offset, (uint64_t)kLocalHeads,
                r.stream, 32);
        } else if (rank == 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_attn_sinks,
                                       hc->d_attn_sinks[layer] + sinks_offset,
                                       (size_t)kLocalHeads * sizeof(float),
                                       cudaMemcpyDeviceToDevice, r.stream));
        } else {
            void *sinks_dsts[kGpus] = {};
            for (int dst_rank = 0; dst_rank < kGpus; ++dst_rank) {
                sinks_dsts[dst_rank] = ranks[dst_rank].d_attn_sinks;
            }
            if (nccl_broadcast_bytes_from_rank0(
                    ranks, hc->d_attn_sinks[layer] + sinks_offset, sinks_dsts,
                    (size_t)kLocalHeads * sizeof(float),
                    "attention_raw_window_sinks") != 0) {
                return 4;
            }
        }
        const uint32_t visible_comp_rows =
            opt.true_ds4_compressed_kv_gate && ratio != 0
                ? std::min(r.attn_comp_rows_written_layers[layer],
                           (uint32_t)kBoundedCompRows)
                : 0u;
        const uint32_t selected_comp_rows =
            visible_comp_rows == 0u
                ? 0u
                : (ratio == 4 && opt.true_ds4_indexer_attention_gate
                       ? std::min(visible_comp_rows, (uint32_t)kBoundedCompRows)
                       : visible_comp_rows);
        if (selected_comp_rows > 0u) {
            if (!r.d_attn_comp_rows ||
                (ratio == 4 && opt.true_ds4_indexer_attention_gate &&
                 !r.d_indexer_topk)) {
                return 4;
            }
            attention_raw_compressed_window_kernel<<<
                (unsigned int)(opt.slots * kLocalHeads), 256, 0, r.stream>>>(
                r.d_attn_heads, ops->attn_q_b.d_out[(size_t)rank],
                r.d_attn_raw_swa, r.d_attn_comp_rows,
                ratio == 4 && opt.true_ds4_indexer_attention_gate
                    ? r.d_indexer_topk
                    : nullptr,
                r.d_attn_sinks, (uint32_t)opt.slots, (uint32_t)kLocalHeads,
                (uint32_t)kHeadDim, (uint32_t)kRawSwaRows, raw_row,
                valid_rows, visible_comp_rows, selected_comp_rows,
                (uint32_t)kBoundedCompRows, (uint32_t)kIndexerTopK);
        } else {
            attention_raw_swa_window_kernel<<<
                (unsigned int)(opt.slots * kLocalHeads), 256, 0, r.stream>>>(
                r.d_attn_heads, ops->attn_q_b.d_out[(size_t)rank],
                r.d_attn_raw_swa, r.d_attn_sinks, (uint32_t)opt.slots,
                (uint32_t)kLocalHeads, (uint32_t)kHeadDim,
                (uint32_t)kRawSwaRows, raw_row, valid_rows);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    if (!opt.decode_cudagraph_gate) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    if (!opt.decode_cudagraph_gate && layer <= 2) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("true_attn_raw_window_heads", layer, rank,
                                 ranks[rank].d_attn_heads, (size_t)heads_elems,
                                 ranks[rank].stream);
        }
    }
    std::printf("tp_ep_true_attention_raw_window\tlayer\t%d\tslots\t%d\t"
                "local_heads\t%d\thead_dim\t%d\traw_rows\t%d\traw_row\t%u\t"
                "valid_rows\t%u\tvisible_compressed_rows\t%u\t"
                "selected_compressed_rows\t%u\tms\t%.6f\tPASS\n",
                layer, opt.slots, kLocalHeads, kHeadDim, kRawSwaRows, raw_row,
                valid_rows,
                opt.true_ds4_compressed_kv_gate && ratio != 0
                    ? std::min(ranks[0].attn_comp_rows_written_layers[layer],
                               (uint32_t)kBoundedCompRows)
                    : 0u,
                opt.true_ds4_compressed_kv_gate && ratio != 0
                    ? std::min(ranks[0].attn_comp_rows_written_layers[layer],
                               (uint32_t)kBoundedCompRows)
                    : 0u,
                ms);
    return 0;
}

int run_true_ds4_attention_output_projection(const Options &opt,
                                             const LayerDenseOps *ops,
                                             RankState ranks[kGpus],
                                             int layer) {
    if (!ops || !ops->initialized || layer < 0 || layer >= 43) {
        return 1;
    }
    if (ops->attn_output_a.cols != kAttentionOutputAInput ||
        ops->attn_output_a.rows_per_gpu != kAttentionOutputAFull / kGpus ||
        ops->attn.cols != kAttentionOutputAFull ||
        ops->attn.rows_per_gpu != kHidden / kGpus) {
        std::fprintf(stderr,
                     "tp_ep_true_attention_output_bad_shape\tlayer\t%d\t"
                     "out_a_cols\t%d\tout_a_rows_per_gpu\t%d\t"
                     "out_b_cols\t%d\tout_b_rows_per_gpu\t%d\n",
                     layer, ops->attn_output_a.cols,
                     ops->attn_output_a.rows_per_gpu, ops->attn.cols,
                     ops->attn.rows_per_gpu);
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    const int block = 256;
    const size_t out_a_shard_cols = (size_t)ops->attn_output_a.rows_per_gpu;
    const size_t out_a_shard_row_bytes = out_a_shard_cols * sizeof(float);
    const size_t out_a_full_row_bytes = (size_t)kAttentionOutputAFull * sizeof(float);
    const uint64_t head_input_elems =
        (uint64_t)opt.slots * (uint64_t)kAttentionOutputAInput;
    const uint64_t out_a_full_elems =
        (uint64_t)opt.slots * (uint64_t)kAttentionOutputAFull;
    const bool graph_event_order = opt.decode_cudagraph_gate;

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_heads || !r.d_attn_output_a_full ||
            !ops->attn_output_a.d_x_half[(size_t)rank] ||
            !ops->attn.d_x_half[(size_t)rank]) {
            return 3;
        }
        fill_dense_input_half_from_tensor_kernel<<<
            (unsigned int)((head_input_elems + block - 1) / block), block, 0,
            r.stream>>>(ops->attn_output_a.d_x_half[(size_t)rank],
                          r.d_attn_heads,
                          (uint32_t)kAttentionOutputAInput,
                          (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    if (graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 4;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }

    if (launch_resident_f8_dense(opt, ops->attn_output_a, ranks) != 0) {
        return 5;
    }
    if (graph_event_order) {
        if (enqueue_rank_streams_wait_after_dense_streams(ranks) != 0) return 5;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            if (ranks[rank].dense_stream) {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].dense_stream));
            } else {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }

    const bool use_nccl_allgather =
        opt.true_ds4_attention_output_nccl_allgather_gate;
    if (use_nccl_allgather) {
        for (int rank = 0; rank < kGpus; ++rank) {
            if (!ranks[rank].compose_nccl_initialized ||
                !ranks[rank].compose_nccl ||
                !ops->attn_output_a.d_out[(size_t)rank]) {
                return 6;
            }
        }
        CHECK_NCCL(ncclGroupStart());
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_NCCL(ncclAllGather(ops->attn_output_a.d_out[(size_t)rank],
                                     r.d_attn_output_a_full,
                                     (size_t)opt.slots * out_a_shard_cols,
                                     ncclFloat,
                                     r.compose_nccl,
                                     r.stream));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            fill_dense_input_half_from_rank_major_shards_kernel<<<
                (unsigned int)((out_a_full_elems + block - 1) / block),
                block, 0, r.stream>>>(
                ops->attn.d_x_half[(size_t)rank], r.d_attn_output_a_full,
                (uint32_t)out_a_shard_cols, (uint32_t)kGpus,
                (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
    } else {
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &dr = ranks[dst];
            CHECK_CUDA(cudaSetDevice(dr.device));
            for (int src = 0; src < kGpus; ++src) {
                const float *src_shard = ops->attn_output_a.d_out[(size_t)src];
                if (!src_shard) return 6;
                CHECK_CUDA(cudaMemcpy2DAsync(
                    dr.d_attn_output_a_full + (size_t)src * out_a_shard_cols,
                    out_a_full_row_bytes, src_shard, out_a_shard_row_bytes,
                    out_a_shard_row_bytes, (size_t)opt.slots, cudaMemcpyDefault,
                    dr.stream));
            }
            fill_dense_input_half_from_tensor_kernel<<<
                (unsigned int)((out_a_full_elems + block - 1) / block), block, 0,
                dr.stream>>>(ops->attn.d_x_half[(size_t)dst],
                              dr.d_attn_output_a_full,
                              (uint32_t)kAttentionOutputAFull,
                              (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    if (graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 6;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }

    if (launch_resident_f8_dense(opt, ops->attn, ranks) != 0) {
        return 7;
    }
    if (graph_event_order) {
        if (enqueue_rank_streams_wait_after_dense_streams(ranks) != 0) return 7;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            if (ranks[rank].dense_stream) {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].dense_stream));
            } else {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }

    TensorF32Stats head_stats;
    TensorF32Stats out_a_stats;
    TensorF32Stats out_b_stats;
    if (!opt.true_ds4_semantic_skip_stats_gate) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            merge_tensor_stats(
                &head_stats,
                collect_tensor_f32_stats(r.d_attn_heads,
                                         (size_t)head_input_elems, r.stream));
            merge_tensor_stats(
                &out_a_stats,
                collect_tensor_f32_stats(r.d_attn_output_a_full,
                                         (size_t)out_a_full_elems, r.stream));
            merge_tensor_stats(
                &out_b_stats,
                collect_tensor_f32_stats(
                    ops->attn.d_out[(size_t)rank],
                    (size_t)opt.slots * (size_t)ops->attn.rows_per_gpu,
                    r.dense_stream ? r.dense_stream : r.stream));
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    std::printf("tp_ep_true_attention_output_projection\tlayer\t%d\tslots\t%d\t"
                "head_input_cols\t%d\tout_a_cols\t%d\tout_b_shard_cols\t%d\t"
                "nccl_allgather\t%d\t"
                "stats_skipped\t%d\t"
                "heads_max\t%.9g\theads_bad\t%d\t"
                "out_a_max\t%.9g\tout_a_bad\t%d\t"
                "out_b_max\t%.9g\tout_b_bad\t%d\tms\t%.6f\tPASS\n",
                layer, opt.slots, kAttentionOutputAInput, kAttentionOutputAFull,
                ops->attn.rows_per_gpu, use_nccl_allgather ? 1 : 0,
                opt.true_ds4_semantic_skip_stats_gate ? 1 : 0,
                head_stats.max_abs,
                head_stats.finite_bad, out_a_stats.max_abs,
                out_a_stats.finite_bad, out_b_stats.max_abs,
                out_b_stats.finite_bad, ms);
    return 0;
}

int run_true_ds4_post_attention_ffn_input(const Options &opt,
                                          SharedHcControls *hc,
                                          const LayerDenseOps *ops,
                                          RankState ranks[kGpus],
                                          int layer,
                                          bool reuse_model_router_route_plan) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        hc->slots != opt.slots || layer < 0 || layer >= 43) {
        return 1;
    }
    if (ops->attn.rows_per_gpu != kHidden / kGpus ||
        ops->shared_gate.cols != kHidden ||
        ops->shared_up.cols != kHidden ||
        ops->shared_gate.rows_per_gpu != kMid / kGpus ||
        ops->shared_up.rows_per_gpu != kMid / kGpus) {
        return 2;
    }
    if (!hc->d_current_full || !hc->d_ffn_normed ||
        !hc->d_ffn_norm_weight[layer]) {
        return 3;
    }

    const auto start = std::chrono::steady_clock::now();
    const int block = 256;
    const uint64_t shard_elems =
        (uint64_t)opt.slots * (uint64_t)(kHidden / kGpus);
    const uint64_t full_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    const bool rank_major_shared_input =
        (opt.routed_ffn_rank_major_input_gate ||
         opt.routed_ffn_rank_major_shared_input_gate) &&
        opt.tp_hc_current_input_nccl_allgather_gate;
    const bool rank_major_route_input =
        (opt.routed_ffn_rank_major_input_gate ||
         opt.routed_ffn_rank_major_route_input_gate) &&
        opt.tp_hc_current_input_nccl_allgather_gate;
    const bool rank_major_input =
        rank_major_shared_input || rank_major_route_input;
    const bool post_attention_route_reuse_audit =
        opt.post_attention_route_reuse_audit_gate &&
        opt.model_router_routes &&
        reuse_model_router_route_plan;
    const bool post_attention_fixed_capacity_route_plan =
        opt.post_attention_fixed_capacity_route_plan_gate &&
        opt.model_router_routes &&
        reuse_model_router_route_plan;
    cudaStream_t control_stream =
        graph_event_order ? ranks[0].stream : (cudaStream_t)0;
    auto sync_control_device = [&]() {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        if (graph_event_order && reuse_model_router_route_plan) return;
        if (graph_event_order) {
            CHECK_CUDA(cudaStreamSynchronize(control_stream));
        } else {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    };

    TensorF32Stats post_shard_stats;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_current_shard || !r.d_post_attn_shard ||
            !ops->attn.d_out[(size_t)rank]) {
            return 4;
        }
        add_current_attention_shard_kernel<<<
            (unsigned int)((shard_elems + block - 1) / block), block, 0,
            r.stream>>>(r.d_post_attn_shard, r.d_current_shard,
                         ops->attn.d_out[(size_t)rank], shard_elems);
        CHECK_CUDA(cudaGetLastError());
    }
    if (graph_event_order) {
        if (enqueue_control_wait_after_rank_streams(opt, ranks,
                                                    control_stream) != 0) {
            return 9;
        }
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            if (!opt.true_ds4_semantic_skip_stats_gate) {
                merge_tensor_stats(
                    &post_shard_stats,
                    collect_tensor_f32_stats(ranks[rank].d_post_attn_shard,
                                             (size_t)shard_elems,
                                             ranks[rank].stream));
            }
        }
    }
    if (graph_event_order && !opt.true_ds4_semantic_skip_stats_gate &&
        !reuse_model_router_route_plan) {
        for (int rank = 0; rank < kGpus; ++rank) {
            merge_tensor_stats(
                &post_shard_stats,
                collect_tensor_f32_stats(ranks[rank].d_post_attn_shard,
                                         (size_t)shard_elems,
                                         ranks[rank].stream));
        }
    }

    if (rank_major_input) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            if (!r.compose_nccl_initialized || !r.compose_nccl ||
                !r.d_post_attn_full_rank_major ||
                !hc->d_ffn_norm_weight_rank[layer][rank]) {
                return 10;
            }
        }
        CHECK_NCCL(ncclGroupStart());
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_NCCL(ncclAllGather(r.d_post_attn_shard,
                                     r.d_post_attn_full_rank_major,
                                     (size_t)shard_elems,
                                     ncclFloat,
                                     r.compose_nccl,
                                     r.stream));
        }
        CHECK_NCCL(ncclGroupEnd());
        if (graph_event_order) {
            if (enqueue_control_wait_after_rank_streams(opt, ranks,
                                                        control_stream) != 0) {
                return 9;
            }
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }
    const bool needs_slot_major_ffn_norm =
        !rank_major_input ||
        !rank_major_shared_input ||
        !rank_major_route_input ||
        !(opt.model_router_rank_major_logits_gate ||
          opt.model_router_allreduce_logits_gate) ||
        !opt.post_attention_skip_slot_major_ffn_norm_gate ||
        opt.post_attention_slot_major_ffn_norm_gate ||
        opt.routed_ffn_rank_major_input_parity_gate ||
        !opt.true_ds4_semantic_skip_stats_gate;
    if (needs_slot_major_ffn_norm) {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            gather_current_shard_to_full_kernel<<<
                (unsigned int)((shard_elems + block - 1) / block), block, 0,
                control_stream>>>(
                hc->d_current_full, ranks[rank].d_post_attn_shard, rank,
                (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();

        rms_norm_weight_rows_stable_kernel<<<
            (unsigned int)opt.slots, 256, 0, control_stream>>>(
            hc->d_ffn_normed, hc->d_current_full, hc->d_ffn_norm_weight[layer],
            (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();
    }
    TensorF32Stats ffn_norm_stats;
    if (needs_slot_major_ffn_norm &&
        !opt.true_ds4_semantic_skip_stats_gate &&
        !(graph_event_order && reuse_model_router_route_plan)) {
        ffn_norm_stats =
            collect_tensor_f32_stats(hc->d_ffn_normed, (size_t)full_elems,
                                     control_stream);
    }

    if (opt.model_router_routes && reuse_model_router_route_plan &&
        !post_attention_route_reuse_audit &&
        !post_attention_fixed_capacity_route_plan) {
        int total_routes = 0;
        for (int rank = 0; rank < kGpus; ++rank) {
            total_routes += ranks[rank].routes;
        }
        if (total_routes <= 0) return 5;
    } else if (opt.model_router_routes) {
        if ((!opt.model_router_rank_major_logits_gate &&
             !opt.model_router_allreduce_logits_gate &&
             !hc->d_router_w[layer]) ||
            !hc->d_router_logits ||
            !hc->d_router_selected || !hc->d_router_weights) {
            return 5;
        }
        const int router_dense_rc = opt.model_router_allreduce_logits_gate
            ? run_model_router_allreduce_logits(opt, hc, ranks, layer,
                                                control_stream, true)
            : (opt.model_router_rank_major_logits_gate
                   ? run_model_router_rank_major_logits(opt, hc, ranks, layer,
                                                        control_stream, true)
                   : run_model_router_dense_logits(opt, hc, layer,
                                                   control_stream));
        if (router_dense_rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_post_attention_router_dense_failed\tlayer\t%d\trc\t%d\n",
                         layer, router_dense_rc);
            return 5;
        }
        if (opt.router_hash_fast_gate && hc->d_router_hash[layer] &&
            hc->d_router_tokens && hc->router_hash_rows[layer] > 0u) {
            router_select_hash_fast_rows_kernel<<<
                (unsigned int)opt.slots, 1, 0, control_stream>>>(
                hc->d_router_selected, hc->d_router_weights,
                hc->d_router_logits, hc->d_router_hash[layer],
                hc->d_router_tokens, hc->d_router_active,
                hc->router_hash_rows[layer], (uint32_t)opt.slots);
        } else {
            router_select_topk_rows_kernel<<<
                (unsigned int)opt.slots, 1, 0, control_stream>>>(
                hc->d_router_selected, hc->d_router_weights,
                hc->d_router_logits, hc->d_router_bias[layer],
                hc->d_router_hash[layer], hc->d_router_tokens,
                hc->d_router_active, hc->router_hash_rows[layer],
                (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();
        int route_rc = 0;
        if (post_attention_fixed_capacity_route_plan) {
            if (graph_event_order) {
                if (enqueue_rank_streams_wait_after_control(opt, ranks,
                                                            control_stream) != 0) {
                    return 9;
                }
            }
            route_rc = upload_post_attention_fixed_capacity_route_plan_gpu(
                opt, hc, ranks, control_stream, graph_event_order);
        } else if (post_attention_route_reuse_audit) {
            if (graph_event_order) {
                if (enqueue_rank_streams_wait_after_control(opt, ranks,
                                                            control_stream) != 0) {
                    return 9;
                }
            }
            const size_t selected_bytes =
                (size_t)opt.slots * (size_t)opt.top_k * sizeof(int);
            const size_t weights_bytes =
                (size_t)opt.slots * (size_t)opt.top_k * sizeof(float);
            if (!graph_event_order) {
                void *selected_dsts[kGpus] = {};
                void *weights_dsts[kGpus] = {};
                for (int rank = 0; rank < kGpus; ++rank) {
                    selected_dsts[rank] = ranks[rank].d_router_selected_plan;
                    weights_dsts[rank] = ranks[rank].d_router_weights_plan;
                }
                if (nccl_broadcast_bytes_from_rank0(
                        ranks, hc->d_router_selected, selected_dsts,
                        selected_bytes,
                        "post_attention_audit_selected") != 0 ||
                    nccl_broadcast_bytes_from_rank0(
                        ranks, hc->d_router_weights, weights_dsts,
                        weights_bytes,
                        "post_attention_audit_weights") != 0) {
                    return 8;
                }
            }
            for (int rank = 0; rank < kGpus; ++rank) {
                RankState &r = ranks[rank];
                if (!r.d_post_attn_route_audit ||
                    !r.d_router_selected_plan ||
                    !r.d_router_weights_plan ||
                    !r.d_offsets || !r.d_route_slots || !r.d_route_weights) {
                    return 6;
                }
                CHECK_CUDA(cudaSetDevice(r.device));
                if (graph_event_order) {
                    enqueue_graph_i32_copy_from_device0(
                        opt, r, rank, r.d_router_selected_plan,
                        hc->d_router_selected,
                        (uint64_t)opt.slots * (uint64_t)opt.top_k,
                        r.stream, block);
                    enqueue_graph_f32_copy_from_device0(
                        opt, r, rank, r.d_router_weights_plan,
                        hc->d_router_weights,
                        (uint64_t)opt.slots * (uint64_t)opt.top_k,
                        r.stream, block);
                }
                CHECK_CUDA(cudaMemsetAsync(r.d_post_attn_route_audit, 0,
                                           4u * sizeof(unsigned long long),
                                           r.stream));
                post_attention_route_plan_audit_kernel<<<
                    (unsigned int)kLocalExperts, 128, 0, r.stream>>>(
                    r.d_post_attn_route_audit, r.d_offsets, r.d_route_slots,
                    r.d_route_weights, r.d_router_selected_plan,
                    r.d_router_weights_plan, (uint32_t)rank,
                    (uint32_t)opt.slots, (uint32_t)opt.top_k);
                CHECK_CUDA(cudaGetLastError());
            }
        } else if (opt.gpu_route_plan_gate) {
            route_rc = upload_model_router_route_plan_gpu(opt, hc, ranks);
        } else if (opt.route_plan_async_upload_gate) {
            RoutePlanHostWorkspace *ws = &hc->route_plan_ws;
            if (!ws->initialized) return 6;
            const size_t route_elems = (size_t)opt.slots * (size_t)opt.top_k;
            CHECK_CUDA(cudaMemcpyAsync(ws->h_selected, hc->d_router_selected,
                                       route_elems * sizeof(int),
                                       cudaMemcpyDeviceToHost, control_stream));
            CHECK_CUDA(cudaMemcpyAsync(ws->h_weights, hc->d_router_weights,
                                       route_elems * sizeof(float),
                                       cudaMemcpyDeviceToHost, control_stream));
            CHECK_CUDA(cudaStreamSynchronize(control_stream));
            route_rc = upload_model_router_route_plan_async(
                opt, ranks, ws->h_selected, ws->h_weights, ws);
        } else {
            if (graph_event_order) {
                CHECK_CUDA(cudaSetDevice(opt.devices[0]));
                CHECK_CUDA(cudaStreamSynchronize(control_stream));
            }
            std::vector<int> selected((size_t)opt.slots * (size_t)opt.top_k);
            std::vector<float> weights((size_t)opt.slots * (size_t)opt.top_k);
            CHECK_CUDA(cudaMemcpy(selected.data(), hc->d_router_selected,
                                  selected.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(weights.data(), hc->d_router_weights,
                                  weights.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            route_rc = upload_model_router_route_plan(opt, ranks,
                                                      selected, weights);
        }
        if (route_rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_post_attention_route_plan_failed\tlayer\t%d\trc\t%d\n",
                         layer, route_rc);
            return 6;
        }
    }

    const uint64_t x_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    if (graph_event_order) {
        if (enqueue_rank_streams_wait_after_control(opt, ranks,
                                                    control_stream) != 0) {
            return 9;
        }
    }
    const bool post_ffn_slot_major_broadcast =
        (!rank_major_input) ||
        (rank_major_input &&
         (opt.routed_ffn_rank_major_input_parity_gate ||
          !rank_major_shared_input || !rank_major_route_input));
    if (post_ffn_slot_major_broadcast) {
        const int bcast_rc = nccl_broadcast_f32_from_device0_to_current_full(
            opt, ranks, hc->d_ffn_normed, full_elems,
            "post_attention_ffn_normed_current");
        if (bcast_rc != 0) return 10;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (rank_major_input) {
            if (!r.d_post_attn_full_rank_major ||
                !hc->d_ffn_norm_weight_rank[layer][rank]) {
                return 7;
            }
            const bool needs_slot_major_copy =
                opt.routed_ffn_rank_major_input_parity_gate ||
                !rank_major_shared_input || !rank_major_route_input;
            if (needs_slot_major_copy) {
                if (!r.d_current_full) return 7;
            }
            if (ops->shared_gate.d_x_half[(size_t)rank] &&
                ops->shared_up.d_x_half[(size_t)rank]) {
                if (rank_major_shared_input) {
                    fill_two_hidden_inputs_half_from_rank_major_norm_kernel<<<
                        (unsigned int)opt.slots, 256, 0, r.stream>>>(
                        ops->shared_gate.d_x_half[(size_t)rank],
                        ops->shared_up.d_x_half[(size_t)rank],
                        r.d_post_attn_full_rank_major,
                        hc->d_ffn_norm_weight_rank[layer][rank],
                        (uint32_t)(kHidden / kGpus), (uint32_t)kGpus,
                        (uint32_t)opt.slots, 1.0e-6f);
                } else {
                    fill_dense_input_half_from_current_kernel<<<
                        (unsigned int)((x_elems + block - 1) / block), block,
                        0, r.stream>>>(
                        ops->shared_gate.d_x_half[(size_t)rank],
                        r.d_current_full, (uint32_t)ops->shared_gate.cols,
                        (uint32_t)opt.slots);
                    fill_dense_input_half_from_current_kernel<<<
                        (unsigned int)((x_elems + block - 1) / block), block,
                        0, r.stream>>>(
                        ops->shared_up.d_x_half[(size_t)rank],
                        r.d_current_full, (uint32_t)ops->shared_up.cols,
                        (uint32_t)opt.slots);
                }
                CHECK_CUDA(cudaGetLastError());
                if (opt.routed_ffn_rank_major_input_parity_gate &&
                    !reuse_model_router_route_plan &&
                    rank_major_shared_input) {
                    HalfInputDiffStats gate_diff =
                        collect_shared_half_input_diff(
                            r, ops->shared_gate.d_x_half[(size_t)rank],
                            r.d_current_full, (uint32_t)ops->shared_gate.cols,
                            (uint32_t)opt.slots, r.stream);
                    log_half_input_diff("shared_gate", layer, rank, gate_diff);
                    HalfInputDiffStats up_diff =
                        collect_shared_half_input_diff(
                            r, ops->shared_up.d_x_half[(size_t)rank],
                            r.d_current_full, (uint32_t)ops->shared_up.cols,
                            (uint32_t)opt.slots, r.stream);
                    log_half_input_diff("shared_up", layer, rank, up_diff);
                }
            }
            if (r.routes > 0) {
                if (rank_major_route_input) {
                    const int *route_total_limit =
                        opt.post_attention_fixed_capacity_route_plan_gate
                            ? r.d_route_totals
                            : nullptr;
                    if (opt.reference_hc_reduce_gate) {
                        pack_rank_major_norm_current_to_routes_scaled_kernel<<<
                            (unsigned int)r.routes, 256, 0, r.stream>>>(
                                r.d_a, r.d_route_inv_scale,
                                r.d_post_attn_full_rank_major,
                                hc->d_ffn_norm_weight_rank[layer][rank],
                                r.d_route_slots, route_total_limit, r.routes,
                                (uint32_t)rank,
                                (uint32_t)(kHidden / kGpus), (uint32_t)kGpus,
                                (uint32_t)opt.slots, 1.0e-6f,
                                kReferenceRouteInputTargetAbs);
                    } else {
                        pack_rank_major_norm_current_to_routes_kernel<<<
                            (unsigned int)r.routes, 256, 0, r.stream>>>(
                                r.d_a, r.d_post_attn_full_rank_major,
                                hc->d_ffn_norm_weight_rank[layer][rank],
                                r.d_route_slots, route_total_limit, r.routes,
                                (uint32_t)rank,
                                (uint32_t)(kHidden / kGpus), (uint32_t)kGpus,
                                (uint32_t)opt.slots, 1.0e-6f);
                    }
                } else {
                    const uint64_t route_elems = (uint64_t)r.routes * kHidden;
                    if (opt.reference_hc_reduce_gate) {
                        pack_current_full_to_routes_scaled_kernel<<<
                            (unsigned int)r.routes, 256, 0, r.stream>>>(
                                r.d_a, r.d_route_inv_scale, r.d_current_full,
                                r.d_route_slots, r.routes,
                                kReferenceRouteInputTargetAbs);
                    } else {
                        pack_current_full_to_routes_kernel<<<
                            (unsigned int)((route_elems + block - 1) / block),
                            block, 0, r.stream>>>(
                                r.d_a, r.d_current_full, r.d_route_slots,
                                r.routes);
                    }
                }
                CHECK_CUDA(cudaGetLastError());
                if (opt.routed_ffn_rank_major_input_parity_gate &&
                    !reuse_model_router_route_plan &&
                    rank_major_route_input &&
                    !opt.reference_hc_reduce_gate) {
                    HalfInputDiffStats route_diff =
                        collect_route_half_input_diff(
                            r, r.d_a, r.d_current_full, r.d_route_slots,
                            r.routes, r.stream);
                    log_half_input_diff("route_a", layer, rank, route_diff);
                }
            }
        } else {
            if (!r.d_current_full) return 7;
            if (ops->shared_gate.d_x_half[(size_t)rank]) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((x_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->shared_gate.d_x_half[(size_t)rank],
                                 r.d_current_full,
                                 (uint32_t)ops->shared_gate.cols,
                                 (uint32_t)opt.slots);
                CHECK_CUDA(cudaGetLastError());
            }
            if (ops->shared_up.d_x_half[(size_t)rank]) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((x_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->shared_up.d_x_half[(size_t)rank],
                                 r.d_current_full,
                                 (uint32_t)ops->shared_up.cols,
                                 (uint32_t)opt.slots);
                CHECK_CUDA(cudaGetLastError());
            }
            const uint64_t route_elems = (uint64_t)r.routes * kHidden;
            if (route_elems > 0) {
                if (opt.reference_hc_reduce_gate) {
                    pack_current_full_to_routes_scaled_kernel<<<
                        (unsigned int)r.routes, 256, 0, r.stream>>>(
                            r.d_a, r.d_route_inv_scale, r.d_current_full,
                            r.d_route_slots, r.routes, kReferenceRouteInputTargetAbs);
                } else {
                    pack_current_full_to_routes_kernel<<<
                        (unsigned int)((route_elems + block - 1) / block), block,
                        0, r.stream>>>(r.d_a, r.d_current_full, r.d_route_slots,
                                       r.routes);
                }
                CHECK_CUDA(cudaGetLastError());
            }
        }
    }
    if (graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 9;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }

    TensorF32Stats route_inv_scale_stats;
    int total_routes = 0;
    for (int rank = 0; rank < kGpus; ++rank) {
        total_routes += ranks[rank].routes;
        if (!opt.true_ds4_semantic_skip_stats_gate &&
            !(graph_event_order && reuse_model_router_route_plan) &&
            ranks[rank].d_route_inv_scale && ranks[rank].routes > 0) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            merge_tensor_stats(
                &route_inv_scale_stats,
                collect_tensor_f32_stats(ranks[rank].d_route_inv_scale,
                                         (size_t)ranks[rank].routes,
                                         ranks[rank].stream));
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    std::printf("tp_ep_post_attention_ffn_input\tlayer\t%d\tslots\t%d\t"
                "total_routes\t%d\tstats_skipped\t%d\tpost_max\t%.9g\tpost_bad\t%d\t"
                "ffn_norm_max\t%.9g\tffn_norm_bad\t%d\t"
                "route_inv_scale_max\t%.9g\troute_inv_scale_bad\t%d\t"
                "rank_major_input\t%d\trank_major_shared_input\t%d\t"
                "rank_major_route_input\t%d\tslot_major_ffn_norm\t%d\t"
                "ms\t%.6f\tPASS\n",
                layer, opt.slots, total_routes,
                opt.true_ds4_semantic_skip_stats_gate ? 1 : 0,
                post_shard_stats.max_abs,
                post_shard_stats.finite_bad, ffn_norm_stats.max_abs,
                ffn_norm_stats.finite_bad, route_inv_scale_stats.max_abs,
                route_inv_scale_stats.finite_bad, rank_major_input ? 1 : 0,
                rank_major_shared_input ? 1 : 0,
                rank_major_route_input ? 1 : 0,
                needs_slot_major_ffn_norm ? 1 : 0, ms);
    return (post_shard_stats.finite_bad || ffn_norm_stats.finite_bad ||
            route_inv_scale_stats.finite_bad) ? 8 : 0;
}

#include "engine/decode_loop.cu"

} // namespace

int run_resident_layer_decode(const Options &opt,
                              const std::vector<ContractRow> &rows,
                              const LayerStats &layer_stats,
                              RankState ranks[kGpus],
                              const Api &api,
                              ds4_v100_tp_runtime *rt,
                              const LayerExpertCache *layer_expert_cache,
                              const DenseF16Cache *dense_f16_cache,
                              const LayerDenseOps *layer_dense_ops,
                              SharedHcControls *shared_hc_controls,
                              TpCudaGraphLayerExec *persistent_graph,
                              LayerRunSummary *summary) {
    if (!rt || !layer_expert_cache || !dense_f16_cache) return 2;

    char err[512] = {0};
    ds4_v100_tp_dense_kv_result kv_result;
    const int write_indexer = ds4_layer_ratio(opt.layer) == 4 ? 1 : 0;
    const uint32_t kv_first_slot = opt.tp_kv_all_slots_gate ? 0u : opt.kv_slot;
    const uint32_t kv_end_slot = opt.tp_kv_all_slots_gate ? (uint32_t)opt.slots : opt.kv_slot + 1u;
    for (uint32_t slot = kv_first_slot; slot < kv_end_slot; ++slot) {
        if (ds4_v100_tp_runtime_dense_kv_slice(rt, opt.layer, slot, opt.position,
                                               write_indexer, &kv_result, err,
                                               sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_dense_kv_slice_failed\tslot\t%u\t%s\n",
                         slot, err);
            return 3;
        }
        if (kv_result.max_abs != 0.0) return 4;
    }

    for (int p = 0; p < kGpus; ++p) {
        ranks[p].gated = layer_expert_cache->gated[p];
        ranks[p].down = layer_expert_cache->down[p];
    }

    DecodeLoopStats decode_loop;
    const int rc = run_decode_loop(opt, rows, ranks, api, rt, dense_f16_cache,
                                   layer_dense_ops, shared_hc_controls,
                                   persistent_graph,
                                   &decode_loop);

    for (int p = 0; p < kGpus; ++p) {
        ranks[p].gated = PackedExperts{};
        ranks[p].down = PackedExperts{};
    }

    if (summary) {
        summary->layer = opt.layer;
        summary->ratio = ds4_layer_ratio(opt.layer);
        summary->pass = rc == 0 && decode_loop.pass;
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
        summary->decode_compose_reduce_ms_per_step =
            decode_loop.compose_reduce_ms_per_step;
        summary->decode_compose_copy_ms_per_step =
            decode_loop.compose_copy_ms_per_step;
        summary->decode_compose_final_ms_per_step =
            decode_loop.compose_final_ms_per_step;
        summary->decode_hc_current_input_ms_per_step =
            decode_loop.hc_current_input_ms_per_step;
        summary->decode_hc_current_seed_ms_per_step =
            decode_loop.hc_current_seed_ms_per_step;
        summary->decode_hc_current_attn_mix_ms_per_step =
            decode_loop.hc_current_attn_mix_ms_per_step;
        summary->decode_hc_current_split_ms_per_step =
            decode_loop.hc_current_split_ms_per_step;
        summary->decode_hc_current_gather_ms_per_step =
            decode_loop.hc_current_gather_ms_per_step;
        summary->decode_hc_current_ffn_router_ms_per_step =
            decode_loop.hc_current_ffn_router_ms_per_step;
        summary->decode_hc_current_ffn_norm_ms_per_step =
            decode_loop.hc_current_ffn_norm_ms_per_step;
        summary->decode_hc_current_router_select_ms_per_step =
            decode_loop.hc_current_router_select_ms_per_step;
        summary->decode_hc_current_router_d2h_ms_per_step =
            decode_loop.hc_current_router_d2h_ms_per_step;
        summary->decode_hc_current_route_upload_ms_per_step =
            decode_loop.hc_current_route_upload_ms_per_step;
        summary->decode_hc_current_fill_pack_ms_per_step =
            decode_loop.hc_current_fill_pack_ms_per_step;
        summary->decode_pre_ep_hc_current_ms_per_step =
            decode_loop.pre_ep_hc_current_ms_per_step;
        summary->decode_pre_ep_attention_projection_ms_per_step =
            decode_loop.pre_ep_attention_projection_ms_per_step;
        summary->decode_pre_ep_compressed_kv_ms_per_step =
            decode_loop.pre_ep_compressed_kv_ms_per_step;
        summary->decode_pre_ep_attention_state_ms_per_step =
            decode_loop.pre_ep_attention_state_ms_per_step;
        summary->decode_pre_ep_typed_history_ms_per_step =
            decode_loop.pre_ep_typed_history_ms_per_step;
        summary->decode_pre_ep_raw_read_ms_per_step =
            decode_loop.pre_ep_raw_read_ms_per_step;
        summary->decode_pre_ep_attention_output_ms_per_step =
            decode_loop.pre_ep_attention_output_ms_per_step;
        summary->decode_pre_ep_post_attention_ffn_input_ms_per_step =
            decode_loop.pre_ep_post_attention_ffn_input_ms_per_step;
        summary->decode_final_hc_ms_per_step = decode_loop.final_hc_ms_per_step;
        summary->decode_cudagraph_sync_all_calls =
            decode_loop.cudagraph_sync_all_calls;
        summary->decode_cudagraph_event_barrier_calls =
            decode_loop.cudagraph_event_barrier_calls;
        summary->decode_cudagraph_rank_stream_syncs =
            decode_loop.cudagraph_rank_stream_syncs;
        summary->decode_cudagraph_dense_stream_syncs =
            decode_loop.cudagraph_dense_stream_syncs;
        summary->decode_cudagraph_copy_stream_syncs =
            decode_loop.cudagraph_copy_stream_syncs;
        summary->decode_cudagraph_capture_attempted =
            decode_loop.cudagraph_capture_attempted;
        summary->decode_cudagraph_capture_succeeded =
            decode_loop.cudagraph_capture_succeeded;
        summary->decode_cudagraph_capture_error =
            decode_loop.cudagraph_capture_error;
        summary->decode_cudagraph_capture_nodes =
            decode_loop.cudagraph_capture_nodes;
        summary->decode_cudagraph_replay_attempted =
            decode_loop.cudagraph_replay_attempted;
        summary->decode_cudagraph_replay_succeeded =
            decode_loop.cudagraph_replay_succeeded;
        summary->decode_cudagraph_replay_error =
            decode_loop.cudagraph_replay_error;
        summary->decode_cudagraph_persistent_cache_hits =
            decode_loop.cudagraph_persistent_cache_hits;
        summary->decode_cudagraph_persistent_cache_misses =
            decode_loop.cudagraph_persistent_cache_misses;
        summary->decode_cudagraph_persistent_invalidations =
            decode_loop.cudagraph_persistent_invalidations;
        summary->decode_cudagraph_persistent_invalidate_layer =
            decode_loop.cudagraph_persistent_invalidate_layer;
        summary->decode_cudagraph_persistent_invalidate_slots =
            decode_loop.cudagraph_persistent_invalidate_slots;
        summary->decode_cudagraph_persistent_invalidate_position =
            decode_loop.cudagraph_persistent_invalidate_position;
        summary->decode_cudagraph_persistent_invalidate_root_device =
            decode_loop.cudagraph_persistent_invalidate_root_device;
        summary->decode_cudagraph_persistent_invalidate_root_stream =
            decode_loop.cudagraph_persistent_invalidate_root_stream;
        summary->decode_cudagraph_instantiate_ms =
            decode_loop.cudagraph_instantiate_ms;
        summary->decode_cudagraph_replay_ms =
            decode_loop.cudagraph_replay_ms;
        summary->decode_checksum = decode_loop.checksum;
        summary->decode_finite_bad = decode_loop.finite_bad;
        summary->rc = rc;
    }
    return rc;
}

int run_layer(const Options &opt,
              LayerRunSummary *summary,
              const DenseF16Cache *shared_dense_f16_cache,
              const SharedApi *shared_api,
              SharedRankBuffers *shared_rank_buffers,
              SharedTpRuntime *shared_tp_runtime,
              const SharedExpertBindings *shared_expert_bindings,
              const SharedDenseOps *shared_dense_ops,
              SharedHcControls *shared_hc_controls) {
    std::vector<ContractRow> rows;
    LayerStats layer_stats;
    if (parse_contract(opt.contract_path, opt.layer, &rows, &layer_stats) != 0 ||
        layer_stats.bad_rows != 0) {
        std::fprintf(stderr, "contract parse failed bad_rows=%llu\n",
                     (unsigned long long)layer_stats.bad_rows);
        return 2;
    }
    DescriptorBindings bindings;
    const LayerExpertCache *layer_expert_cache = nullptr;
    if (shared_expert_bindings) {
        layer_expert_cache = &shared_expert_bindings->layers[opt.layer];
        bindings = layer_expert_cache->bindings;
    } else {
        if (parse_tm_index(opt.tm_index_path, opt.layer, &bindings) != 0) {
            std::fprintf(stderr, "tm index parse failed for layer %d\n", opt.layer);
            return 2;
        }
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
            CHECK_CUDA(cudaStreamCreate(&r.dense_stream));
            CHECK_CUDA(cudaStreamCreate(&r.copy_stream));
            for (int q = 0; q < kGpus; ++q) {
                CHECK_CUDA(cudaStreamCreate(&r.copy_streams[q]));
                CHECK_CUDA(cudaEventCreateWithFlags(&r.copy_done[q], cudaEventDisableTiming));
            }
            CHECK_CUDA(cudaEventCreateWithFlags(&r.stream_done, cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&r.dense_done, cudaEventDisableTiming));
            for (int e = 0; e < kGraphOrderEventSlots; ++e) {
                CHECK_CUDA(cudaEventCreateWithFlags(&r.graph_stream_done[e],
                                                    cudaEventDisableTiming));
                CHECK_CUDA(cudaEventCreateWithFlags(&r.graph_dense_done[e],
                                                    cudaEventDisableTiming));
            }
            CHECK_CUDA(cudaEventCreateWithFlags(&r.dense_wait, cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreate(&r.start));
            CHECK_CUDA(cudaEventCreate(&r.mid));
            CHECK_CUDA(cudaEventCreate(&r.stop));
            r.route_compact_plan_ints = compact_route_plan_ints(opt);
            CHECK_CUDA(cudaMalloc(&r.d_route_compact_plan,
                                  r.route_compact_plan_ints * sizeof(int)));
            bind_compact_route_plan(&r, opt);
            CHECK_CUDA(cudaMalloc(&r.d_router_selected_plan,
                                  (size_t)opt.slots * (size_t)opt.top_k * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&r.d_router_weights_plan,
                                  (size_t)opt.slots * (size_t)opt.top_k * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&r.d_route_offsets_all,
                                  (size_t)kGpus * (size_t)(kLocalExperts + 1) *
                                      sizeof(int)));
            CHECK_CUDA(cudaMalloc(&r.d_route_totals,
                                  (size_t)kGpus * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&r.d_post_attn_route_audit,
                                  4u * sizeof(unsigned long long)));
            CHECK_CUDA(cudaMemset(r.d_post_attn_route_audit, 0,
                                  4u * sizeof(unsigned long long)));
            std::vector<int> compact_plan(r.route_compact_plan_ints, -1);
            const size_t compact_indices = (size_t)opt.slots * (size_t)opt.top_k;
            const size_t compact_counts = (size_t)opt.slots;
            for (int src = 0; src < kGpus; ++src) {
                std::vector<int> route_index_by_slot;
                build_route_index_by_slot_for_rank(src, opt.slots, opt.top_k,
                                                   &route_index_by_slot);
                CHECK_CUDA(cudaMalloc(&r.d_route_index_by_slot[src],
                                      route_index_by_slot.size() * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(r.d_route_index_by_slot[src],
                                      route_index_by_slot.data(),
                                      route_index_by_slot.size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
                std::vector<int> route_indices_by_slot;
                std::vector<int> route_count_by_slot;
                build_route_indices_by_slot_for_rank(src, opt.slots, opt.top_k,
                                                     &route_indices_by_slot,
                                                     &route_count_by_slot);
                std::copy(route_indices_by_slot.begin(), route_indices_by_slot.end(),
                          compact_plan.begin() + (size_t)src * compact_indices);
                std::copy(route_count_by_slot.begin(), route_count_by_slot.end(),
                          compact_plan.begin() + (size_t)kGpus * compact_indices +
                              (size_t)src * compact_counts);
            }
            CHECK_CUDA(cudaMemcpy(r.d_route_compact_plan, compact_plan.data(),
                                  compact_plan.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));

            std::vector<int> offsets;
            std::vector<int> route_slots;
            std::vector<float> route_weights;
            build_offsets_for_rank(p, opt.slots, opt.top_k, &offsets, &route_slots,
                                   &route_weights, &r.routes, &r.active_experts,
                                   &r.max_routes_per_expert);

            r.route_capacity = opt.slots * opt.top_k;
            const size_t route_capacity_elems = (size_t)r.route_capacity * kHidden;
            CHECK_CUDA(cudaMalloc(&r.d_offsets, offsets.size() * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(r.d_offsets, offsets.data(), offsets.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&r.d_route_slots,
                                  (size_t)r.route_capacity * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(r.d_route_slots, route_slots.data(),
                                  route_slots.size() * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&r.d_route_weights,
                                  (size_t)r.route_capacity * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(r.d_route_weights, route_weights.data(),
                                  route_weights.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&r.d_route_inv_scale,
                                  (size_t)r.route_capacity * sizeof(float)));
            std::vector<float> route_inv_scale((size_t)r.route_capacity, 1.0f);
            CHECK_CUDA(cudaMemcpy(r.d_route_inv_scale, route_inv_scale.data(),
                                  route_inv_scale.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&r.d_a, route_capacity_elems * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&r.d_gate_up,
                                  (size_t)r.route_capacity * kFusedN * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&r.d_gated,
                                  (size_t)r.route_capacity * kMid * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&r.d_down, route_capacity_elems * sizeof(__half)));

            std::mt19937 rng(0xE2350000u + (uint32_t)p * 97u);
            std::uniform_real_distribution<float> dist(-0.003f, 0.003f);
            std::vector<__half> h_a(route_capacity_elems);
            for (__half &v : h_a) v = __float2half(dist(rng));
            CHECK_CUDA(cudaMemcpy(r.d_a, h_a.data(),
                                  route_capacity_elems * sizeof(__half),
                                  cudaMemcpyHostToDevice));
        }
        aggregate_routes += r.routes;
        min_routes = std::min(min_routes, r.routes);
        max_routes = std::max(max_routes, r.routes);

        if (layer_expert_cache) {
            r.gated = layer_expert_cache->gated[p];
            r.down = layer_expert_cache->down[p];
            ep_loaded_bytes += layer_expert_cache->gated[p].d_w_active.size()
                ? layer_expert_cache->bytes / kGpus
                : 0;
        } else {
            std::vector<int> active;
            for (int e = 0; e < kPackedLocalExperts; ++e) active.push_back(e);
            if (pack_descriptor_set(r.device, bindings.gated, p, active, opt.pack_dir,
                                    &r.gated, &ep_loaded_bytes) != 0 ||
                pack_descriptor_set(r.device, bindings.down, p, active, opt.pack_dir,
                                   &r.down, &ep_loaded_bytes) != 0) {
                close_local_runtime();
                return 8;
            }
        }
        layer_stats.gpu[p].ep_loaded_bytes = ep_loaded_bytes;
    }
    layer_stats.ep_loaded_bytes = ep_loaded_bytes;

    if (!shared_rank_buffers && open_compose_nccl(opt, ranks) != 0) {
        close_local_runtime();
        return 8;
    }

    if (!opt.skip_predecode_probes) {
        for (int i = 0; i < opt.warmup; ++i) {
            for (int p = 0; p < kGpus; ++p) {
                const int gate_rc = run_gate_selected(ranks[p], *api, opt);
                if (gate_rc != 0 || run_down(ranks[p], *api, opt) != 0) {
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
                const int gate_rc = run_gate_selected(ranks[p], *api, opt);
                if (gate_rc != 0) return 10;
            }
        }
        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_CUDA(cudaEventRecord(ranks[p].mid, ranks[p].stream));
        }
        for (int i = 0; i < opt.iters; ++i) {
            for (int p = 0; p < kGpus; ++p) {
                if (run_down(ranks[p], *api, opt) != 0) return 11;
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
        std::printf("rank\t%d\tdevice\t%d\troutes\t%d\troute_capacity\t%d\t"
                    "active_local_experts\t%d\t"
                    "max_routes_per_expert\t%d\tgate_ms\t%.6f\tdown_ms\t%.6f\t"
                    "ep_ms\t%.6f\tdense_rows\t%llu\tcontrol_rows\t%llu\t"
                    "expert_rows\t%llu\tkv_rows\t%llu\tcomp_rows\t%llu\t"
                    "checksum\t%llu\n",
                    p, ranks[p].device, ranks[p].routes, ranks[p].route_capacity,
                    ranks[p].active_experts,
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
                    "shared_dense_ms\t%.6f\tfused_compose_sum\t%d\t"
                    "nccl_reduce_scatter\t%d\tcompose_ms\t%.6f\t"
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
                    compose.nccl_reduce_scatter_compose ? 1 : 0,
                    compose.compose_ms, (unsigned long long)compose.checksum,
                    compose.finite_bad, compose.repeat_max_abs,
                    compose.repeat_bad, compose.pass ? "PASS" : "FAIL");
    }
    if (compose_rc != 0) {
        close_local_runtime();
        return 13;
    }

    DecodeLoopStats decode_loop;
    const LayerDenseOps *layer_dense_ops =
        shared_dense_ops && shared_dense_ops->initialized
            ? &shared_dense_ops->layers[opt.layer]
            : nullptr;
    const int decode_rc = run_decode_loop(opt, rows, ranks, *api, rt, dense_f16_cache,
                                          layer_dense_ops, shared_hc_controls,
                                          shared_rank_buffers
                                              ? &shared_rank_buffers->graph_cache.layers[opt.layer]
                                              : nullptr,
                                          &decode_loop);
    if (decode_loop.enabled) {
        std::printf("tp_ep_decode_loop\tsteps\t%d\tslots\t%d\tslot_steps\t%llu\t"
                    "total_ms\t%.6f\tms_per_step\t%.6f\tslot_step_tok_s\t%.6f\t"
                    "dense_hmma\t%d\tdense_f16_cublas\t%d\tdense_f16_cache\t%d\t"
                    "overlap_ep_dense\t%d\tdirect_remote_compose\t%d\t"
                    "source_copy_schedule\t%d\tskip_self_compose_copy\t%d\t"
                    "multi_copy_streams\t%d\t"
                    "decode_cudagraph_gate\t%d\t"
                    "decode_cudagraph_replay_probe_gate\t%d\t"
                    "ep_ms_per_step\t%.6f\tdense_ms_per_step\t%.6f\t"
                    "fused_compose_sum\t%d\tnccl_reduce_scatter\t%d\t"
                    "compose_ms_per_step\t%.6f\t"
                    "compose_reduce_ms_per_step\t%.6f\t"
                    "compose_copy_ms_per_step\t%.6f\t"
                    "compose_final_ms_per_step\t%.6f\t"
                    "hc_current_input_gate\t%d\t"
                    "hc_current_input_peer_gather\t%d\t"
                    "hc_current_input_nccl_allgather\t%d\t"
                    "hc_current_allreduce\t%d\t"
                    "hc_current_input_stream_sync\t%d\t"
                    "hc_current_input_ms_per_step\t%.6f\t"
                    "final_hc_carry_gate\t%d\tfinal_hc_ms_per_step\t%.6f\t"
                    "dense_loaded_bytes\t%llu\t"
                    "ep_contribution_bytes\t%llu\tep_return_dtype\t%s\t"
                    "ep_return_bytes\t%llu\t"
                    "cudagraph_replay_attempted\t%d\t"
                    "cudagraph_replay_succeeded\t%d\t"
                    "cudagraph_instantiate_ms\t%.6f\t"
                    "cudagraph_replay_ms\t%.6f\t"
                    "checksum\t%llu\tfinite_bad\t%d\t%s\n",
                    decode_loop.steps, decode_loop.slots,
                    (unsigned long long)decode_loop.slot_steps,
                    decode_loop.total_ms, decode_loop.ms_per_step,
                    decode_loop.tok_s,
                    decode_loop.dense_hmma_compose ? 1 : 0,
                    decode_loop.dense_f16_cublas_compose ? 1 : 0,
                    decode_loop.dense_f16_cache_compose ? 1 : 0,
                    opt.overlap_ep_dense ? 1 : 0,
                    opt.direct_remote_compose ? 1 : 0,
                    opt.source_copy_schedule ? 1 : 0,
                    opt.skip_self_compose_copy ? 1 : 0,
                    opt.multi_copy_streams ? 1 : 0,
                    opt.decode_cudagraph_gate ? 1 : 0,
                    opt.decode_cudagraph_replay_probe_gate ? 1 : 0,
                    decode_loop.ep_ms_per_step,
                    decode_loop.dense_ms_per_step,
                    decode_loop.fused_compose_sum ? 1 : 0,
                    decode_loop.nccl_reduce_scatter_compose ? 1 : 0,
                    decode_loop.compose_ms_per_step,
                    decode_loop.compose_reduce_ms_per_step,
                    decode_loop.compose_copy_ms_per_step,
                    decode_loop.compose_final_ms_per_step,
                    opt.tp_hc_current_input_gate ? 1 : 0,
                    opt.tp_hc_current_input_peer_gather_gate ? 1 : 0,
                    opt.tp_hc_current_input_nccl_allgather_gate ? 1 : 0,
                    opt.tp_hc_current_allreduce_gate ? 1 : 0,
                    opt.tp_hc_current_input_stream_sync_gate ? 1 : 0,
                    decode_loop.hc_current_input_ms_per_step,
                    opt.final_hc_carry_gate ? 1 : 0,
                    decode_loop.final_hc_ms_per_step,
                    (unsigned long long)decode_loop.dense_loaded_bytes,
                    (unsigned long long)decode_loop.ep_contribution_bytes,
                    decode_loop.ep_return_fp16 ? "fp16" : "fp32",
                    (unsigned long long)decode_loop.ep_return_bytes,
                    decode_loop.cudagraph_replay_attempted,
                    decode_loop.cudagraph_replay_succeeded,
                    decode_loop.cudagraph_instantiate_ms,
                    decode_loop.cudagraph_replay_ms,
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
                "compose_nccl_reduce_scatter\t%d\t"
                "compose_ms\t%.6f\t"
                "compose_checksum\t%llu\tcompose_finite_bad\t%d\t"
                "compose_repeat_max_abs\t%.9f\tcompose_repeat_bad\t%d\t"
                "compose_pass\t%d\t"
                "decode_steps\t%d\tdecode_slot_steps\t%llu\tdecode_total_ms\t%.6f\t"
                "decode_ms_per_step\t%.6f\tdecode_slot_step_tok_s\t%.6f\t"
                "decode_dense_hmma\t%d\tdecode_dense_f16_cublas\t%d\t"
                "decode_dense_f16_cache\t%d\t"
                "decode_overlap_ep_dense\t%d\tdecode_direct_remote_compose\t%d\t"
                "decode_source_copy_schedule\t%d\t"
                "decode_ep_ms_per_step\t%.6f\tdecode_dense_ms_per_step\t%.6f\t"
                "decode_fused_compose_sum\t%d\tdecode_nccl_reduce_scatter\t%d\t"
                "decode_compose_ms_per_step\t%.6f\t"
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
                compose.nccl_reduce_scatter_compose ? 1 : 0,
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
                opt.overlap_ep_dense ? 1 : 0,
                opt.direct_remote_compose ? 1 : 0,
                opt.source_copy_schedule ? 1 : 0,
                decode_loop.ep_ms_per_step,
                decode_loop.dense_ms_per_step,
                decode_loop.fused_compose_sum ? 1 : 0,
                decode_loop.nccl_reduce_scatter_compose ? 1 : 0,
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
        summary->decode_compose_reduce_ms_per_step =
            decode_loop.compose_reduce_ms_per_step;
        summary->decode_compose_copy_ms_per_step =
            decode_loop.compose_copy_ms_per_step;
        summary->decode_compose_final_ms_per_step =
            decode_loop.compose_final_ms_per_step;
        summary->decode_hc_current_input_ms_per_step =
            decode_loop.hc_current_input_ms_per_step;
        summary->decode_hc_current_seed_ms_per_step =
            decode_loop.hc_current_seed_ms_per_step;
        summary->decode_hc_current_attn_mix_ms_per_step =
            decode_loop.hc_current_attn_mix_ms_per_step;
        summary->decode_hc_current_split_ms_per_step =
            decode_loop.hc_current_split_ms_per_step;
        summary->decode_hc_current_gather_ms_per_step =
            decode_loop.hc_current_gather_ms_per_step;
        summary->decode_hc_current_ffn_router_ms_per_step =
            decode_loop.hc_current_ffn_router_ms_per_step;
        summary->decode_hc_current_ffn_norm_ms_per_step =
            decode_loop.hc_current_ffn_norm_ms_per_step;
        summary->decode_hc_current_router_select_ms_per_step =
            decode_loop.hc_current_router_select_ms_per_step;
        summary->decode_hc_current_router_d2h_ms_per_step =
            decode_loop.hc_current_router_d2h_ms_per_step;
        summary->decode_hc_current_route_upload_ms_per_step =
            decode_loop.hc_current_route_upload_ms_per_step;
        summary->decode_hc_current_fill_pack_ms_per_step =
            decode_loop.hc_current_fill_pack_ms_per_step;
        summary->decode_pre_ep_hc_current_ms_per_step =
            decode_loop.pre_ep_hc_current_ms_per_step;
        summary->decode_pre_ep_attention_projection_ms_per_step =
            decode_loop.pre_ep_attention_projection_ms_per_step;
        summary->decode_pre_ep_compressed_kv_ms_per_step =
            decode_loop.pre_ep_compressed_kv_ms_per_step;
        summary->decode_pre_ep_attention_state_ms_per_step =
            decode_loop.pre_ep_attention_state_ms_per_step;
        summary->decode_pre_ep_typed_history_ms_per_step =
            decode_loop.pre_ep_typed_history_ms_per_step;
        summary->decode_pre_ep_raw_read_ms_per_step =
            decode_loop.pre_ep_raw_read_ms_per_step;
        summary->decode_pre_ep_attention_output_ms_per_step =
            decode_loop.pre_ep_attention_output_ms_per_step;
        summary->decode_pre_ep_post_attention_ffn_input_ms_per_step =
            decode_loop.pre_ep_post_attention_ffn_input_ms_per_step;
        summary->decode_final_hc_ms_per_step = decode_loop.final_hc_ms_per_step;
        summary->decode_checksum = decode_loop.checksum;
    }

    if (!shared_rank_buffers) {
        close_compose_nccl(ranks);
    }
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!layer_expert_cache) free_packed(r.gated);
        r.gated = PackedExperts{};
        if (!layer_expert_cache) free_packed(r.down);
        r.down = PackedExperts{};
        if (!shared_rank_buffers) {
            CHECK_CUDA(cudaFree(r.d_offsets));
            CHECK_CUDA(cudaFree(r.d_route_slots));
            CHECK_CUDA(cudaFree(r.d_route_weights));
            CHECK_CUDA(cudaFree(r.d_route_inv_scale));
            CHECK_CUDA(cudaFree(r.d_a));
            CHECK_CUDA(cudaFree(r.d_gate_up));
            CHECK_CUDA(cudaFree(r.d_gated));
            CHECK_CUDA(cudaFree(r.d_down));
            if (r.d_ep_contrib_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_all));
            if (r.d_ep_contrib_half_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_half_all));
            if (r.d_ep_contrib_bcast_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_bcast_all));
            if (r.d_ep_contrib_half_bcast_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_half_bcast_all));
            for (int src = 0; src < kGpus; ++src) {
                if (r.d_ep_remote[src]) CHECK_CUDA(cudaFree(r.d_ep_remote[src]));
                if (r.d_ep_remote_half[src]) CHECK_CUDA(cudaFree(r.d_ep_remote_half[src]));
            }
            if (r.d_ep_sum) CHECK_CUDA(cudaFree(r.d_ep_sum));
            if (r.d_next_hidden) CHECK_CUDA(cudaFree(r.d_next_hidden));
            if (r.d_current_shard) CHECK_CUDA(cudaFree(r.d_current_shard));
            if (r.d_current_full) CHECK_CUDA(cudaFree(r.d_current_full));
            if (r.d_current_full_normed) CHECK_CUDA(cudaFree(r.d_current_full_normed));
            if (r.d_current_full_rank_major) CHECK_CUDA(cudaFree(r.d_current_full_rank_major));
            if (r.d_post_attn_full_rank_major) CHECK_CUDA(cudaFree(r.d_post_attn_full_rank_major));
            if (r.d_rank_major_norm_scale) CHECK_CUDA(cudaFree(r.d_rank_major_norm_scale));
            if (r.d_router_logits_shard) CHECK_CUDA(cudaFree(r.d_router_logits_shard));
            if (r.d_router_logits_rank_major) CHECK_CUDA(cudaFree(r.d_router_logits_rank_major));
            if (r.d_half_diff_counts) CHECK_CUDA(cudaFree(r.d_half_diff_counts));
            if (r.d_half_diff_max_bits) CHECK_CUDA(cudaFree(r.d_half_diff_max_bits));
            if (r.d_half_diff_first) CHECK_CUDA(cudaFree(r.d_half_diff_first));
            if (r.d_post_attn_route_audit) CHECK_CUDA(cudaFree(r.d_post_attn_route_audit));
            if (r.d_final_hc_shard) CHECK_CUDA(cudaFree(r.d_final_hc_shard));
            if (r.d_hc_scratch_shard) CHECK_CUDA(cudaFree(r.d_hc_scratch_shard));
            if (r.d_hc_split) CHECK_CUDA(cudaFree(r.d_hc_split));
            for (int layer = 0; layer < 43; ++layer) {
                if (r.d_attn_raw_swa_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_attn_raw_swa_layers[layer]));
                }
            }
            if (r.d_attn_kv_full) CHECK_CUDA(cudaFree(r.d_attn_kv_full));
            if (r.d_attn_heads) CHECK_CUDA(cudaFree(r.d_attn_heads));
            if (r.d_attn_output_a_full) CHECK_CUDA(cudaFree(r.d_attn_output_a_full));
            if (r.d_post_attn_shard) CHECK_CUDA(cudaFree(r.d_post_attn_shard));
            if (r.d_attn_sinks) CHECK_CUDA(cudaFree(r.d_attn_sinks));
            if (r.d_indexer_topk) CHECK_CUDA(cudaFree(r.d_indexer_topk));
            if (r.d_indexer_scores) CHECK_CUDA(cudaFree(r.d_indexer_scores));
            for (int layer = 0; layer < 43; ++layer) {
                if (r.d_index_comp_rows_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_index_comp_rows_layers[layer]));
                }
                if (r.d_index_comp_state_score_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_index_comp_state_score_layers[layer]));
                }
                if (r.d_index_comp_state_kv_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_index_comp_state_kv_layers[layer]));
                }
            }
            if (r.d_index_comp_score_cur) CHECK_CUDA(cudaFree(r.d_index_comp_score_cur));
            if (r.d_index_comp_kv_cur) CHECK_CUDA(cudaFree(r.d_index_comp_kv_cur));
            for (int layer = 0; layer < 43; ++layer) {
                if (r.d_attn_comp_rows_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_attn_comp_rows_layers[layer]));
                }
                if (r.d_attn_comp_state_score_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_attn_comp_state_score_layers[layer]));
                }
                if (r.d_attn_comp_state_kv_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_attn_comp_state_kv_layers[layer]));
                }
            }
            if (r.d_attn_comp_score_cur) CHECK_CUDA(cudaFree(r.d_attn_comp_score_cur));
            if (r.d_attn_comp_kv_cur) CHECK_CUDA(cudaFree(r.d_attn_comp_kv_cur));
            if (r.dense_wait) CHECK_CUDA(cudaEventDestroy(r.dense_wait));
            if (r.start) CHECK_CUDA(cudaEventDestroy(r.start));
            if (r.mid) CHECK_CUDA(cudaEventDestroy(r.mid));
            if (r.stop) CHECK_CUDA(cudaEventDestroy(r.stop));
            if (r.d_route_totals) CHECK_CUDA(cudaFree(r.d_route_totals));
            if (r.d_route_offsets_all) CHECK_CUDA(cudaFree(r.d_route_offsets_all));
            if (r.d_router_weights_plan) CHECK_CUDA(cudaFree(r.d_router_weights_plan));
            if (r.d_router_selected_plan) CHECK_CUDA(cudaFree(r.d_router_selected_plan));
            const bool has_route_compact_plan = r.d_route_compact_plan != nullptr;
            if (r.d_route_compact_plan) CHECK_CUDA(cudaFree(r.d_route_compact_plan));
            for (int src = 0; src < kGpus; ++src) {
                if (r.d_route_index_by_slot[src]) CHECK_CUDA(cudaFree(r.d_route_index_by_slot[src]));
                if (!has_route_compact_plan && r.d_route_indices_by_slot[src]) {
                    CHECK_CUDA(cudaFree(r.d_route_indices_by_slot[src]));
                }
                if (!has_route_compact_plan && r.d_route_count_by_slot[src]) {
                    CHECK_CUDA(cudaFree(r.d_route_count_by_slot[src]));
                }
            }
            for (int q = 0; q < kGpus; ++q) {
                if (r.copy_done[q]) CHECK_CUDA(cudaEventDestroy(r.copy_done[q]));
                if (r.copy_streams[q]) CHECK_CUDA(cudaStreamDestroy(r.copy_streams[q]));
            }
            for (int e = 0; e < kGraphOrderEventSlots; ++e) {
                if (r.graph_stream_done[e]) {
                    CHECK_CUDA(cudaEventDestroy(r.graph_stream_done[e]));
                }
                if (r.graph_dense_done[e]) {
                    CHECK_CUDA(cudaEventDestroy(r.graph_dense_done[e]));
                }
            }
            if (r.dense_done) CHECK_CUDA(cudaEventDestroy(r.dense_done));
            if (r.stream_done) CHECK_CUDA(cudaEventDestroy(r.stream_done));
            CHECK_CUDA(cudaStreamDestroy(r.copy_stream));
            CHECK_CUDA(cudaStreamDestroy(r.dense_stream));
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

int run_token_major_serving_loop(const Options &opt,
                                 const DenseF16Cache *shared_dense_f16_cache,
                                 const SharedApi *shared_api,
                                 SharedRankBuffers *shared_rank_buffers,
                                 SharedTpRuntime *shared_tp_runtime,
                                 const SharedExpertBindings *shared_expert_bindings,
                                 const SharedDenseOps *shared_dense_ops,
                                 SharedOutputHead *shared_output_head,
                                 SharedHcControls *shared_hc_controls,
                                 SharedTokenEmbedding *shared_token_embedding,
                                 const std::vector<uint32_t> *decode_input_tokens,
                                 const std::vector<unsigned char> *decode_active_slots,
                                 std::vector<ContractRow> resident_rows[43],
                                 LayerStats resident_stats[43],
                                 bool resident_serving_loop,
                                 ServingBenchResult *serving_result) {
    int pass_invocations = 0;
    double sum_decode_ms = 0.0;
    double sum_ep_ms = 0.0;
    double sum_dense_ms = 0.0;
    double sum_compose_ms = 0.0;
    double sum_compose_reduce_ms = 0.0;
    double sum_compose_copy_ms = 0.0;
    double sum_compose_final_ms = 0.0;
    double sum_hc_current_input_ms = 0.0;
    double sum_hc_current_seed_ms = 0.0;
    double sum_hc_current_attn_mix_ms = 0.0;
    double sum_hc_current_split_ms = 0.0;
    double sum_hc_current_gather_ms = 0.0;
    double sum_hc_current_ffn_router_ms = 0.0;
    double sum_hc_current_ffn_norm_ms = 0.0;
    double sum_hc_current_router_select_ms = 0.0;
    double sum_hc_current_router_d2h_ms = 0.0;
    double sum_hc_current_route_upload_ms = 0.0;
    double sum_hc_current_fill_pack_ms = 0.0;
    double sum_pre_ep_hc_current_ms = 0.0;
    double sum_pre_ep_attention_projection_ms = 0.0;
    double sum_pre_ep_compressed_kv_ms = 0.0;
    double sum_pre_ep_attention_state_ms = 0.0;
    double sum_pre_ep_typed_history_ms = 0.0;
    double sum_pre_ep_raw_read_ms = 0.0;
    double sum_pre_ep_attention_output_ms = 0.0;
    double sum_pre_ep_post_attention_ffn_input_ms = 0.0;
    double sum_final_hc_ms = 0.0;
    int sum_cudagraph_sync_all_calls = 0;
    int sum_cudagraph_event_barrier_calls = 0;
    int sum_cudagraph_rank_stream_syncs = 0;
    int sum_cudagraph_dense_stream_syncs = 0;
    int sum_cudagraph_copy_stream_syncs = 0;
    int sum_cudagraph_capture_attempted = 0;
    int sum_cudagraph_capture_succeeded = 0;
    int sum_cudagraph_capture_error = 0;
    size_t sum_cudagraph_capture_nodes = 0;
    int sum_cudagraph_replay_attempted = 0;
    int sum_cudagraph_replay_succeeded = 0;
    int sum_cudagraph_replay_error = 0;
    int sum_cudagraph_persistent_cache_hits = 0;
    int sum_cudagraph_persistent_cache_misses = 0;
    int sum_cudagraph_persistent_invalidations = 0;
    int sum_cudagraph_persistent_invalidate_layer = 0;
    int sum_cudagraph_persistent_invalidate_slots = 0;
    int sum_cudagraph_persistent_invalidate_position = 0;
    int sum_cudagraph_persistent_invalidate_root_device = 0;
    int sum_cudagraph_persistent_invalidate_root_stream = 0;
    double sum_cudagraph_instantiate_ms = 0.0;
    double sum_cudagraph_replay_ms = 0.0;
    double first_token_decode_ms = 0.0;
    double continuation_decode_ms = 0.0;
    double first_token_wall_ms = 0.0;
    double continuation_wall_ms = 0.0;
    uint64_t checksum = 0;
    if (opt.final_hc_carry_gate && !opt.tp_hc_persist_state_gate &&
        shared_rank_buffers && shared_rank_buffers->initialized) {
        for (int rank = 0; rank < kGpus; ++rank) {
            shared_rank_buffers->ranks[rank].hc_initialized = false;
        }
    }
    TpEpProfilerWindowGuard profiler_guard(opt);
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < opt.decode_steps; ++step) {
        const auto step_start = std::chrono::steady_clock::now();
        double step_decode_ms = 0.0;
        if (step == 0 && shared_token_embedding && decode_input_tokens &&
            !decode_input_tokens->empty()) {
            if (!shared_rank_buffers || !shared_rank_buffers->initialized ||
                ensure_compose_buffers(opt, shared_rank_buffers->ranks) != 0) {
                std::fprintf(stderr, "tp_ep_token_embedding_seed_failed\treason\tmissing_rank_buffers\n");
                return 15;
            }
            const int seed_rc = seed_rank_hc_from_input_tokens(
                opt, shared_token_embedding, shared_rank_buffers->ranks,
                *decode_input_tokens);
            if (seed_rc != 0) {
                std::fprintf(stderr, "tp_ep_token_embedding_seed_failed\trc\t%d\n",
                             seed_rc);
                return 15;
            }
        }
        if (shared_hc_controls && shared_hc_controls->initialized &&
            shared_hc_controls->d_router_tokens &&
            decode_input_tokens && decode_input_tokens->size() >= (size_t)opt.slots) {
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
            CHECK_CUDA(cudaMemcpy(shared_hc_controls->d_router_tokens,
                                  decode_input_tokens->data(),
                                  (size_t)opt.slots * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
        }
        if (shared_hc_controls && shared_hc_controls->initialized &&
            shared_hc_controls->d_router_active &&
            decode_active_slots && decode_active_slots->size() >= (size_t)opt.slots) {
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
            CHECK_CUDA(cudaMemcpy(shared_hc_controls->d_router_active,
                                  decode_active_slots->data(),
                                  (size_t)opt.slots * sizeof(unsigned char),
                                  cudaMemcpyHostToDevice));
        }
        for (int layer = 0; layer < 43; ++layer) {
            Options layer_opt = opt;
            layer_opt.layer = layer;
            layer_opt.position = opt.position + (uint64_t)step;
            layer_opt.decode_steps = 1;
            layer_opt.true_ds4_attention_raw_valid_rows =
                std::max(1u, std::min((uint32_t)(step + 1), (uint32_t)kRawSwaRows));
            layer_opt.warmup = 0;
            LayerRunSummary s;
            SharedTpRuntime *tp_runtime_arg =
                shared_tp_runtime && shared_tp_runtime->initialized ? shared_tp_runtime : nullptr;
            const SharedExpertBindings *expert_arg =
                shared_expert_bindings && shared_expert_bindings->initialized
                    ? shared_expert_bindings
                    : nullptr;
            const SharedDenseOps *dense_ops_arg =
                shared_dense_ops && shared_dense_ops->initialized ? shared_dense_ops : nullptr;
            int rc = 0;
            if (resident_serving_loop) {
                if (!shared_api || !shared_api->initialized ||
                    !shared_rank_buffers || !shared_rank_buffers->initialized ||
                    !shared_tp_runtime || !shared_tp_runtime->initialized ||
                    !shared_expert_bindings || !shared_expert_bindings->initialized ||
                    !shared_dense_f16_cache || !shared_dense_f16_cache->enabled) {
                    std::fprintf(stderr, "resident serving loop missing shared state\n");
                    rc = 2;
                    s.pass = false;
                } else {
                    const LayerDenseOps *layer_dense_ops =
                        dense_ops_arg ? &dense_ops_arg->layers[layer] : nullptr;
                    TpCudaGraphLayerExec *persistent_graph =
                        opt.decode_cudagraph_persistent_replay_gate
                            ? &shared_rank_buffers->graph_cache.layers[layer]
                            : nullptr;
                    rc = run_resident_layer_decode(layer_opt,
                                                   resident_rows[layer],
                                                   resident_stats[layer],
                                                   shared_rank_buffers->ranks,
                                                   shared_api->api,
                                                   shared_tp_runtime->rt,
                                                   &shared_expert_bindings->layers[layer],
                                                   shared_dense_f16_cache,
                                                   layer_dense_ops,
                                                   shared_hc_controls,
                                                   persistent_graph,
                                                   &s);
                }
            } else {
                rc = run_layer(layer_opt, &s, shared_dense_f16_cache, shared_api,
                               shared_rank_buffers, tp_runtime_arg, expert_arg,
                               dense_ops_arg, shared_hc_controls);
            }
            std::printf("tp_ep_token_major_item\tstep\t%d\tlayer\t%d\tratio\t%d\t"
                        "position\t%llu\t"
                        "decode_ms_per_step\t%.6f\tdecode_slot_step_tok_s\t%.6f\t"
                        "decode_ep_ms_per_step\t%.6f\tdecode_dense_ms_per_step\t%.6f\t"
                        "decode_compose_ms_per_step\t%.6f\t"
                        "decode_compose_reduce_ms_per_step\t%.6f\t"
                        "decode_compose_copy_ms_per_step\t%.6f\t"
                        "decode_compose_final_ms_per_step\t%.6f\t"
                        "decode_hc_current_input_ms_per_step\t%.6f\t"
                        "decode_hc_current_seed_ms_per_step\t%.6f\t"
                        "decode_hc_current_attn_mix_ms_per_step\t%.6f\t"
                        "decode_hc_current_split_ms_per_step\t%.6f\t"
                        "decode_hc_current_gather_ms_per_step\t%.6f\t"
                        "decode_hc_current_ffn_router_ms_per_step\t%.6f\t"
                        "decode_hc_current_ffn_norm_ms_per_step\t%.6f\t"
                        "decode_hc_current_router_select_ms_per_step\t%.6f\t"
                        "decode_hc_current_router_d2h_ms_per_step\t%.6f\t"
                        "decode_hc_current_route_upload_ms_per_step\t%.6f\t"
                        "decode_hc_current_fill_pack_ms_per_step\t%.6f\t"
                        "decode_pre_ep_hc_current_ms_per_step\t%.6f\t"
                        "decode_pre_ep_attention_projection_ms_per_step\t%.6f\t"
                        "decode_pre_ep_compressed_kv_ms_per_step\t%.6f\t"
                        "decode_pre_ep_attention_state_ms_per_step\t%.6f\t"
                        "decode_pre_ep_typed_history_ms_per_step\t%.6f\t"
                        "decode_pre_ep_raw_read_ms_per_step\t%.6f\t"
                        "decode_pre_ep_attention_output_ms_per_step\t%.6f\t"
                        "decode_pre_ep_post_attention_ffn_input_ms_per_step\t%.6f\t"
                        "decode_final_hc_ms_per_step\t%.6f\t"
                        "decode_cudagraph_replay_attempted\t%d\t"
                        "decode_cudagraph_replay_succeeded\t%d\t"
                        "decode_cudagraph_persistent_cache_hits\t%d\t"
                        "decode_cudagraph_persistent_cache_misses\t%d\t"
                        "decode_cudagraph_persistent_invalidations\t%d\t"
                        "decode_cudagraph_persistent_invalidate_position\t%d\t"
                        "decode_cudagraph_instantiate_ms\t%.6f\t"
                        "decode_cudagraph_replay_ms\t%.6f\t"
                        "decode_checksum\t%llu\tdecode_finite_bad\t%d\trc\t%d\t%s\n",
                        step, s.layer, s.ratio,
                        (unsigned long long)layer_opt.position,
                        s.decode_ms_per_step,
                        s.decode_slot_step_tok_s,
                        s.decode_ep_ms_per_step,
                        s.decode_dense_ms_per_step,
                        s.decode_compose_ms_per_step,
                        s.decode_compose_reduce_ms_per_step,
                        s.decode_compose_copy_ms_per_step,
                        s.decode_compose_final_ms_per_step,
                        s.decode_hc_current_input_ms_per_step,
                        s.decode_hc_current_seed_ms_per_step,
                        s.decode_hc_current_attn_mix_ms_per_step,
                        s.decode_hc_current_split_ms_per_step,
                        s.decode_hc_current_gather_ms_per_step,
                        s.decode_hc_current_ffn_router_ms_per_step,
                        s.decode_hc_current_ffn_norm_ms_per_step,
                        s.decode_hc_current_router_select_ms_per_step,
                        s.decode_hc_current_router_d2h_ms_per_step,
                        s.decode_hc_current_route_upload_ms_per_step,
                        s.decode_hc_current_fill_pack_ms_per_step,
                        s.decode_pre_ep_hc_current_ms_per_step,
                        s.decode_pre_ep_attention_projection_ms_per_step,
                        s.decode_pre_ep_compressed_kv_ms_per_step,
                        s.decode_pre_ep_attention_state_ms_per_step,
                        s.decode_pre_ep_typed_history_ms_per_step,
                        s.decode_pre_ep_raw_read_ms_per_step,
                        s.decode_pre_ep_attention_output_ms_per_step,
                        s.decode_pre_ep_post_attention_ffn_input_ms_per_step,
                        s.decode_final_hc_ms_per_step,
                        s.decode_cudagraph_replay_attempted,
                        s.decode_cudagraph_replay_succeeded,
                        s.decode_cudagraph_persistent_cache_hits,
                        s.decode_cudagraph_persistent_cache_misses,
                        s.decode_cudagraph_persistent_invalidations,
                        s.decode_cudagraph_persistent_invalidate_position,
                        s.decode_cudagraph_instantiate_ms,
                        s.decode_cudagraph_replay_ms,
                        (unsigned long long)s.decode_checksum,
                        s.decode_finite_bad,
                        rc,
                        (rc == 0 && s.pass) ? "PASS" : "FAIL");
            if (rc == 0 && s.pass) {
                pass_invocations++;
                sum_decode_ms += s.decode_ms_per_step;
                step_decode_ms += s.decode_ms_per_step;
                sum_ep_ms += s.decode_ep_ms_per_step;
                sum_dense_ms += s.decode_dense_ms_per_step;
                sum_compose_ms += s.decode_compose_ms_per_step;
                sum_compose_reduce_ms += s.decode_compose_reduce_ms_per_step;
                sum_compose_copy_ms += s.decode_compose_copy_ms_per_step;
                sum_compose_final_ms += s.decode_compose_final_ms_per_step;
                sum_hc_current_input_ms += s.decode_hc_current_input_ms_per_step;
                sum_hc_current_seed_ms += s.decode_hc_current_seed_ms_per_step;
                sum_hc_current_attn_mix_ms += s.decode_hc_current_attn_mix_ms_per_step;
                sum_hc_current_split_ms += s.decode_hc_current_split_ms_per_step;
                sum_hc_current_gather_ms += s.decode_hc_current_gather_ms_per_step;
                sum_hc_current_ffn_router_ms += s.decode_hc_current_ffn_router_ms_per_step;
                sum_hc_current_ffn_norm_ms += s.decode_hc_current_ffn_norm_ms_per_step;
                sum_hc_current_router_select_ms +=
                    s.decode_hc_current_router_select_ms_per_step;
                sum_hc_current_router_d2h_ms +=
                    s.decode_hc_current_router_d2h_ms_per_step;
                sum_hc_current_route_upload_ms +=
                    s.decode_hc_current_route_upload_ms_per_step;
                sum_hc_current_fill_pack_ms += s.decode_hc_current_fill_pack_ms_per_step;
                sum_pre_ep_hc_current_ms += s.decode_pre_ep_hc_current_ms_per_step;
                sum_pre_ep_attention_projection_ms +=
                    s.decode_pre_ep_attention_projection_ms_per_step;
                sum_pre_ep_compressed_kv_ms += s.decode_pre_ep_compressed_kv_ms_per_step;
                sum_pre_ep_attention_state_ms +=
                    s.decode_pre_ep_attention_state_ms_per_step;
                sum_pre_ep_typed_history_ms += s.decode_pre_ep_typed_history_ms_per_step;
                sum_pre_ep_raw_read_ms += s.decode_pre_ep_raw_read_ms_per_step;
                sum_pre_ep_attention_output_ms +=
                    s.decode_pre_ep_attention_output_ms_per_step;
                sum_pre_ep_post_attention_ffn_input_ms +=
                    s.decode_pre_ep_post_attention_ffn_input_ms_per_step;
                sum_final_hc_ms += s.decode_final_hc_ms_per_step;
                sum_cudagraph_sync_all_calls +=
                    s.decode_cudagraph_sync_all_calls;
                sum_cudagraph_event_barrier_calls +=
                    s.decode_cudagraph_event_barrier_calls;
                sum_cudagraph_rank_stream_syncs +=
                    s.decode_cudagraph_rank_stream_syncs;
                sum_cudagraph_dense_stream_syncs +=
                    s.decode_cudagraph_dense_stream_syncs;
                sum_cudagraph_copy_stream_syncs +=
                    s.decode_cudagraph_copy_stream_syncs;
                sum_cudagraph_capture_attempted +=
                    s.decode_cudagraph_capture_attempted;
                sum_cudagraph_capture_succeeded +=
                    s.decode_cudagraph_capture_succeeded;
                if (sum_cudagraph_capture_error == 0 &&
                    s.decode_cudagraph_capture_error != 0) {
                    sum_cudagraph_capture_error =
                        s.decode_cudagraph_capture_error;
                }
                sum_cudagraph_capture_nodes +=
                    s.decode_cudagraph_capture_nodes;
                sum_cudagraph_replay_attempted +=
                    s.decode_cudagraph_replay_attempted;
                sum_cudagraph_replay_succeeded +=
                    s.decode_cudagraph_replay_succeeded;
                if (sum_cudagraph_replay_error == 0 &&
                    s.decode_cudagraph_replay_error != 0) {
                    sum_cudagraph_replay_error =
                        s.decode_cudagraph_replay_error;
                }
                sum_cudagraph_persistent_cache_hits +=
                    s.decode_cudagraph_persistent_cache_hits;
                sum_cudagraph_persistent_cache_misses +=
                    s.decode_cudagraph_persistent_cache_misses;
                sum_cudagraph_persistent_invalidations +=
                    s.decode_cudagraph_persistent_invalidations;
                sum_cudagraph_persistent_invalidate_layer +=
                    s.decode_cudagraph_persistent_invalidate_layer;
                sum_cudagraph_persistent_invalidate_slots +=
                    s.decode_cudagraph_persistent_invalidate_slots;
                sum_cudagraph_persistent_invalidate_position +=
                    s.decode_cudagraph_persistent_invalidate_position;
                sum_cudagraph_persistent_invalidate_root_device +=
                    s.decode_cudagraph_persistent_invalidate_root_device;
                sum_cudagraph_persistent_invalidate_root_stream +=
                    s.decode_cudagraph_persistent_invalidate_root_stream;
                sum_cudagraph_instantiate_ms +=
                    s.decode_cudagraph_instantiate_ms;
                sum_cudagraph_replay_ms +=
                    s.decode_cudagraph_replay_ms;
                checksum ^= s.decode_checksum +
                            (uint64_t)(step + 1) * 1000003ull +
                            (uint64_t)(layer + 1) * 104729ull;
            } else {
                const auto stop = std::chrono::steady_clock::now();
                const double wall_ms =
                    std::chrono::duration<double, std::milli>(stop - start).count();
                std::printf("tp_ep_token_major_scaffold\tsteps\t%d\tlayers\t43\t"
                            "pass_invocations\t%d\tfailed_step\t%d\tfailed_layer\t%d\t"
                            "slots\t%d\tctx\t262144\twall_ms\t%.6f\tFAIL\n",
                            opt.decode_steps, pass_invocations, step, layer,
                            opt.slots, wall_ms);
                std::fflush(stdout);
                return rc == 0 ? 1 : rc;
            }
        }
        const auto step_stop = std::chrono::steady_clock::now();
        const double step_wall_ms =
            std::chrono::duration<double, std::milli>(step_stop - step_start).count();
        if (step == 0) {
            first_token_decode_ms += step_decode_ms;
            first_token_wall_ms += step_wall_ms;
        } else {
            continuation_decode_ms += step_decode_ms;
            continuation_wall_ms += step_wall_ms;
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    const double ms_per_token = opt.decode_steps > 0
        ? sum_decode_ms / (double)opt.decode_steps
        : 0.0;
    const double slot_step_tok_s = sum_decode_ms > 0.0
        ? (double)opt.slots * (double)opt.decode_steps * 1000.0 / sum_decode_ms
        : 0.0;
    std::printf("tp_ep_token_major_scaffold\tsteps\t%d\tlayers\t43\t"
                "pass_invocations\t%d\tslots\t%d\tctx\t262144\t"
                "shared_api\t%d\tshared_rank_buffers\t%d\tshared_tp_runtime\t%d\t"
                "shared_expert_bindings\t%d\toverlap_ep_dense\t%d\t"
                "shared_dense_ops\t%d\t"
                "skip_decode_checksum\t%d\t"
                "direct_remote_compose\t%d\tsource_copy_schedule\t%d\t"
                "skip_self_compose_copy\t%d\t"
                "multi_copy_streams\t%d\t"
                "compact_moe_decode_gate\t%d\t"
                "router_cublas_gate\t%d\t"
                "router_hash_fast_gate\t%d\t"
                "gpu_route_plan_gate\t%d\t"
                "route_plan_async_upload_gate\t%d\t"
                "fused_gated_silu_gate\t%d\t"
                "routed_ffn_norm_input_gate\t%d\t"
                "routed_ffn_rank_major_input_gate\t%d\t"
                "routed_ffn_rank_major_shared_input_gate\t%d\t"
                "routed_ffn_rank_major_route_input_gate\t%d\t"
                "routed_ffn_rank_major_input_parity_gate\t%d\t"
                "post_attention_route_reuse_audit_gate\t%d\t"
                "post_attention_fixed_capacity_route_plan_gate\t%d\t"
                "post_attention_device_actual_route_sync_gate\t%d\t"
                "post_attention_static_rank_route_cap\t%d\t"
                "post_attention_static_executor_route_cap\t%d\t"
                "post_attention_static_compose_route_cap\t%d\t"
                "post_attention_masked_compact_copy_gate\t%d\t"
                "post_attention_slot_major_ffn_norm_gate\t%d\t"
                "post_attention_skip_slot_major_ffn_norm_gate\t%d\t"
                "model_router_rank_major_logits_gate\t%d\t"
                "model_router_allreduce_logits_gate\t%d\t"
                "routed_gate_standalone_swiglu\t%d\t"
                "sum_decode_ms\t%.6f\tms_per_token\t%.6f\t"
                "projected_slot_step_tok_s\t%.6f\t"
                "sum_ep_ms\t%.6f\tsum_dense_ms\t%.6f\tsum_compose_ms\t%.6f\t"
                "sum_compose_reduce_ms\t%.6f\tsum_compose_copy_ms\t%.6f\t"
                "sum_compose_final_ms\t%.6f\t"
                "tp_hc_current_input_gate\t%d\t"
                "tp_hc_current_input_peer_gather\t%d\t"
                "tp_hc_current_input_nccl_allgather\t%d\t"
                "tp_hc_current_allreduce\t%d\t"
                "tp_hc_current_input_stream_sync\t%d\t"
                "sum_hc_current_input_ms\t%.6f\t"
                "sum_hc_current_seed_ms\t%.6f\t"
                "sum_hc_current_attn_mix_ms\t%.6f\t"
                "sum_hc_current_split_ms\t%.6f\t"
                "sum_hc_current_gather_ms\t%.6f\t"
                "sum_hc_current_ffn_router_ms\t%.6f\t"
                "sum_hc_current_ffn_norm_ms\t%.6f\t"
                "sum_hc_current_router_select_ms\t%.6f\t"
                "sum_hc_current_router_d2h_ms\t%.6f\t"
                "sum_hc_current_route_upload_ms\t%.6f\t"
                "sum_hc_current_fill_pack_ms\t%.6f\t"
                "sum_pre_ep_hc_current_ms\t%.6f\t"
                "attention_projection_rank_local_input_gate\t%d\t"
                "attention_projection_rank_major_input_gate\t%d\t"
                "sum_pre_ep_attention_projection_ms\t%.6f\t"
                "sum_pre_ep_compressed_kv_ms\t%.6f\t"
                "sum_pre_ep_attention_state_ms\t%.6f\t"
                "sum_pre_ep_typed_history_ms\t%.6f\t"
                "sum_pre_ep_raw_read_ms\t%.6f\t"
                "sum_pre_ep_attention_output_ms\t%.6f\t"
                "sum_pre_ep_post_attention_ffn_input_ms\t%.6f\t"
                "final_hc_carry_gate\t%d\tsum_final_hc_ms\t%.6f\t"
                "decode_cudagraph_capture_attempted\t%d\t"
                "decode_cudagraph_capture_succeeded\t%d\t"
                "decode_cudagraph_replay_attempted\t%d\t"
                "decode_cudagraph_replay_succeeded\t%d\t"
                "decode_cudagraph_persistent_cache_hits\t%d\t"
                "decode_cudagraph_persistent_cache_misses\t%d\t"
                "decode_cudagraph_persistent_invalidations\t%d\t"
                "decode_cudagraph_persistent_invalidate_layer\t%d\t"
                "decode_cudagraph_persistent_invalidate_slots\t%d\t"
                "decode_cudagraph_persistent_invalidate_position\t%d\t"
                "decode_cudagraph_persistent_invalidate_root_device\t%d\t"
                "decode_cudagraph_persistent_invalidate_root_stream\t%d\t"
                "decode_cudagraph_instantiate_ms\t%.6f\t"
                "decode_cudagraph_replay_ms\t%.6f\t"
                "wall_ms\t%.6f\tchecksum\t%llu\tPASS\n",
                opt.decode_steps, pass_invocations, opt.slots,
                shared_api && shared_api->initialized ? 1 : 0,
                shared_rank_buffers && shared_rank_buffers->initialized ? 1 : 0,
                shared_tp_runtime && shared_tp_runtime->initialized ? 1 : 0,
                shared_expert_bindings && shared_expert_bindings->initialized ? 1 : 0,
                opt.overlap_ep_dense ? 1 : 0,
                shared_dense_ops && shared_dense_ops->initialized ? 1 : 0,
                opt.skip_decode_checksum ? 1 : 0,
                opt.direct_remote_compose ? 1 : 0,
                opt.source_copy_schedule ? 1 : 0,
                opt.skip_self_compose_copy ? 1 : 0,
                opt.multi_copy_streams ? 1 : 0,
                opt.compact_moe_decode_gate ? 1 : 0,
                opt.router_cublas_gate ? 1 : 0,
                opt.router_hash_fast_gate ? 1 : 0,
                opt.gpu_route_plan_gate ? 1 : 0,
                opt.route_plan_async_upload_gate ? 1 : 0,
                opt.fused_gated_silu_gate ? 1 : 0,
                opt.routed_ffn_norm_input_gate ? 1 : 0,
                opt.routed_ffn_rank_major_input_gate ? 1 : 0,
                opt.routed_ffn_rank_major_shared_input_gate ? 1 : 0,
                opt.routed_ffn_rank_major_route_input_gate ? 1 : 0,
                opt.routed_ffn_rank_major_input_parity_gate ? 1 : 0,
                opt.post_attention_route_reuse_audit_gate ? 1 : 0,
                opt.post_attention_fixed_capacity_route_plan_gate ? 1 : 0,
                opt.post_attention_device_actual_route_sync_gate ? 1 : 0,
                opt.post_attention_static_rank_route_cap,
                opt.post_attention_static_executor_route_cap,
                opt.post_attention_static_compose_route_cap,
                opt.post_attention_masked_compact_copy_gate ? 1 : 0,
                opt.post_attention_slot_major_ffn_norm_gate ? 1 : 0,
                opt.post_attention_skip_slot_major_ffn_norm_gate ? 1 : 0,
                opt.model_router_rank_major_logits_gate ? 1 : 0,
                opt.model_router_allreduce_logits_gate ? 1 : 0,
                (opt.routed_ffn_norm_input_gate &&
                 !(opt.fused_gated_silu_gate && !opt.reference_hc_reduce_gate)) ? 1 : 0,
                sum_decode_ms, ms_per_token, slot_step_tok_s,
                sum_ep_ms, sum_dense_ms, sum_compose_ms,
                sum_compose_reduce_ms, sum_compose_copy_ms,
                sum_compose_final_ms,
                opt.tp_hc_current_input_gate ? 1 : 0,
                opt.tp_hc_current_input_peer_gather_gate ? 1 : 0,
                opt.tp_hc_current_input_nccl_allgather_gate ? 1 : 0,
                opt.tp_hc_current_allreduce_gate ? 1 : 0,
                opt.tp_hc_current_input_stream_sync_gate ? 1 : 0,
                sum_hc_current_input_ms,
                sum_hc_current_seed_ms,
                sum_hc_current_attn_mix_ms,
                sum_hc_current_split_ms,
                sum_hc_current_gather_ms,
                sum_hc_current_ffn_router_ms,
                sum_hc_current_ffn_norm_ms,
                sum_hc_current_router_select_ms,
                sum_hc_current_router_d2h_ms,
                sum_hc_current_route_upload_ms,
                sum_hc_current_fill_pack_ms,
                sum_pre_ep_hc_current_ms,
                opt.true_ds4_attention_projection_rank_local_input_gate ? 1 : 0,
                opt.true_ds4_attention_projection_rank_major_input_gate ? 1 : 0,
                sum_pre_ep_attention_projection_ms,
                sum_pre_ep_compressed_kv_ms,
                sum_pre_ep_attention_state_ms,
                sum_pre_ep_typed_history_ms,
                sum_pre_ep_raw_read_ms,
                sum_pre_ep_attention_output_ms,
                sum_pre_ep_post_attention_ffn_input_ms,
                opt.final_hc_carry_gate ? 1 : 0, sum_final_hc_ms,
                sum_cudagraph_capture_attempted,
                sum_cudagraph_capture_succeeded,
                sum_cudagraph_replay_attempted,
                sum_cudagraph_replay_succeeded,
                sum_cudagraph_persistent_cache_hits,
                sum_cudagraph_persistent_cache_misses,
                sum_cudagraph_persistent_invalidations,
                sum_cudagraph_persistent_invalidate_layer,
                sum_cudagraph_persistent_invalidate_slots,
                sum_cudagraph_persistent_invalidate_position,
                sum_cudagraph_persistent_invalidate_root_device,
                sum_cudagraph_persistent_invalidate_root_stream,
                sum_cudagraph_instantiate_ms,
                sum_cudagraph_replay_ms,
                wall_ms, (unsigned long long)checksum);
    if (opt.serving_bench || serving_result) {
        const uint64_t prompt_tokens = (uint64_t)opt.slots;
        const uint64_t generated_tokens = (uint64_t)opt.slots *
                                          (uint64_t)opt.decode_steps;
        const uint64_t continuation_tokens = opt.decode_steps > 1
            ? (uint64_t)opt.slots * (uint64_t)(opt.decode_steps - 1)
            : 0ull;
        const double generated_tok_s_decode = sum_decode_ms > 0.0
            ? (double)generated_tokens * 1000.0 / sum_decode_ms
            : 0.0;
        const double generated_tok_s_wall = wall_ms > 0.0
            ? (double)generated_tokens * 1000.0 / wall_ms
            : 0.0;
        const double continuation_tok_s_decode = continuation_decode_ms > 0.0
            ? (double)continuation_tokens * 1000.0 / continuation_decode_ms
            : 0.0;
        const double continuation_tok_s_wall = continuation_wall_ms > 0.0
            ? (double)continuation_tokens * 1000.0 / continuation_wall_ms
            : 0.0;
        if (serving_result) {
            serving_result->prompt_tokens = prompt_tokens;
            serving_result->generated_tokens = generated_tokens;
            serving_result->continuation_tokens = continuation_tokens;
            serving_result->first_token_decode_ms = first_token_decode_ms;
            serving_result->continuation_decode_ms = continuation_decode_ms;
            serving_result->first_token_wall_ms = first_token_wall_ms;
            serving_result->continuation_wall_ms = continuation_wall_ms;
            serving_result->total_decode_ms = sum_decode_ms;
            serving_result->total_wall_ms = wall_ms;
            serving_result->total_ep_ms = sum_ep_ms;
            serving_result->total_dense_ms = sum_dense_ms;
            serving_result->total_compose_ms = sum_compose_ms;
            serving_result->total_compose_reduce_ms = sum_compose_reduce_ms;
            serving_result->total_compose_copy_ms = sum_compose_copy_ms;
            serving_result->total_compose_final_ms = sum_compose_final_ms;
            serving_result->total_hc_current_input_ms = sum_hc_current_input_ms;
            serving_result->token_input_seed =
                shared_token_embedding && decode_input_tokens &&
                !decode_input_tokens->empty();
            serving_result->first_input_token =
                decode_input_tokens && !decode_input_tokens->empty()
                    ? (*decode_input_tokens)[0]
                    : UINT32_MAX;
            serving_result->aggregate_generated_tok_s_decode = generated_tok_s_decode;
            serving_result->aggregate_generated_tok_s_wall = generated_tok_s_wall;
            serving_result->aggregate_continuation_tok_s_decode = continuation_tok_s_decode;
            serving_result->aggregate_continuation_tok_s_wall = continuation_tok_s_wall;
            serving_result->checksum = checksum;
        }
        SharedOutputHead lazy_output_head;
        SharedOutputHead *output_head_for_step = shared_output_head;
        const bool use_lazy_output_head =
            opt.diagnostic_output_head && opt.diagnostic_output_head_lazy_gate &&
            (opt.serving_bench || serving_result) &&
            (!output_head_for_step || !output_head_for_step->initialized) &&
            shared_rank_buffers && shared_rank_buffers->initialized;
        if (use_lazy_output_head) {
            std::vector<ContractRow> all_rows;
            LayerStats all_stats;
            if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
                all_stats.bad_rows != 0 ||
                open_shared_output_head(opt, all_rows, &lazy_output_head) != 0) {
                std::fprintf(stderr, "tp_ep lazy diagnostic output-head open failed\n");
                close_shared_output_head(opt, &lazy_output_head);
                return 12;
            }
            std::printf("tp_ep_diagnostic_output_head_lazy_shared\tslots\t%d\t"
                        "vocab\t%d\trows_per_gpu\t%d\toutput_weight_bytes\t%llu\t"
                        "logits_bytes\t%llu\tproxy_hc\t%d\tPASS\n",
                        opt.slots,
                        lazy_output_head.vocab,
                        lazy_output_head.rows_per_gpu,
                        (unsigned long long)lazy_output_head.output_weight_bytes,
                        (unsigned long long)lazy_output_head.logits_bytes,
                        opt.tp_hc_final_expand_gate ? 0 : 1);
            if (report_vram_checkpoint(opt, "after_lazy_output_head") != 0) {
                close_shared_output_head(opt, &lazy_output_head);
                return 14;
            }
            if (nccl_gate_active(opt) && opt.nccl_min_free_mib != 0) {
                (void)report_vram_checkpoint_min_free(
                    opt, "nccl_after_lazy_output_head", opt.nccl_min_free_mib);
            }
            output_head_for_step = &lazy_output_head;
        }
        if (output_head_for_step && output_head_for_step->initialized &&
            shared_rank_buffers && shared_rank_buffers->initialized) {
            OutputHeadRunResult head_result;
            const int head_rc = run_shared_output_head_from_rank_hc(
                opt, output_head_for_step, shared_rank_buffers->ranks, &head_result);
            std::printf("tp_ep_diagnostic_output_head\tsteps\t%d\tslots\t%d\t"
                        "proxy_hc\t%d\ttotal_ms\t%.6f\tgather_ms\t%.6f\t"
                        "prep_ms\t%.6f\tbroadcast_ms\t%.6f\tprojection_ms\t%.6f\t"
                        "projection_kernel_worst_ms\t%.6f\ttop1_ms\t%.6f\t"
                        "device_sync_count\t%d\t"
                        "stream_sync_count\t%d\tevent_sync_count\t%d\t"
                        "first_token\t%u\tfirst_logit\t%.9f\tfinite_bad\t%d\t"
                        "checksum\t%llu\t%s\n",
                        opt.decode_steps, opt.slots,
                        opt.tp_hc_final_expand_gate ? 0 : 1,
                        head_result.total_ms,
                        head_result.gather_ms, head_result.prep_ms,
                        head_result.broadcast_ms, head_result.projection_ms,
                        head_result.projection_kernel_worst_ms, head_result.top1_ms,
                        head_result.device_sync_count,
                        head_result.stream_sync_count,
                        head_result.event_sync_count,
                        head_result.tokens.empty() ? UINT32_MAX : head_result.tokens[0],
                        head_result.logits.empty() ? 0.0f : head_result.logits[0],
                        head_result.finite_bad,
                        (unsigned long long)head_result.checksum,
                        head_rc == 0 && head_result.pass ? "PASS" : "FAIL");
            if (head_rc != 0 || !head_result.pass) {
                if (lazy_output_head.initialized) {
                    close_shared_output_head(opt, &lazy_output_head);
                }
                return head_rc == 0 ? 14 : head_rc;
            }
            if (serving_result) {
                serving_result->diagnostic_output_head = true;
                serving_result->diagnostic_output_head_proxy_hc =
                    !opt.tp_hc_final_expand_gate;
                serving_result->output_head_ms = head_result.total_ms;
                serving_result->output_head_gather_ms = head_result.gather_ms;
                serving_result->output_head_prep_ms = head_result.prep_ms;
                serving_result->output_head_broadcast_ms = head_result.broadcast_ms;
                serving_result->output_head_projection_ms = head_result.projection_ms;
                serving_result->output_head_top1_ms = head_result.top1_ms;
                serving_result->selected_tokens = head_result.tokens;
                serving_result->selected_logits = head_result.logits;
                serving_result->checksum ^= head_result.checksum + 0x0A17EADull;
            }
        }
        if (lazy_output_head.initialized) {
            close_shared_output_head(opt, &lazy_output_head);
            if (report_vram_checkpoint(opt, "after_lazy_output_head_close") != 0) {
                return 14;
            }
            if (nccl_gate_active(opt) && opt.nccl_min_free_mib != 0) {
                (void)report_vram_checkpoint_min_free(
                    opt, "nccl_after_lazy_output_head_close",
                    opt.nccl_min_free_mib);
            }
        }
        if (opt.serving_bench) {
            std::printf("tp_ep_serving_bench\tschema\tds4_v100_tp_ep_serving_bench.v1\t"
                        "requests\t%d\tslots\t%d\tctx\t262144\tgenerated_per_request\t%d\t"
                        "prompt_tokens\t%llu\tgenerated_tokens\t%llu\t"
                        "continuation_tokens\t%llu\t"
                        "first_token_decode_ms\t%.6f\tcontinuation_decode_ms\t%.6f\t"
                        "first_token_wall_ms\t%.6f\tcontinuation_wall_ms\t%.6f\t"
                        "total_decode_ms\t%.6f\ttotal_wall_ms\t%.6f\t"
                        "aggregate_generated_tok_s_decode\t%.6f\t"
                        "aggregate_generated_tok_s_wall\t%.6f\t"
                        "aggregate_continuation_tok_s_decode\t%.6f\t"
                        "aggregate_continuation_tok_s_wall\t%.6f\t"
                        "checksum\t%llu\tPASS\n",
                        opt.slots, opt.slots, opt.decode_steps,
                        (unsigned long long)prompt_tokens,
                        (unsigned long long)generated_tokens,
                        (unsigned long long)continuation_tokens,
                        first_token_decode_ms, continuation_decode_ms,
                        first_token_wall_ms, continuation_wall_ms,
                        sum_decode_ms, wall_ms,
                        generated_tok_s_decode, generated_tok_s_wall,
                        continuation_tok_s_decode, continuation_tok_s_wall,
                        (unsigned long long)checksum);
        }
    }
    if (opt.decode_cudagraph_gate) {
        const int graph_audit_steps = opt.warmup + opt.decode_steps;
        const int total_stream_syncs = sum_cudagraph_rank_stream_syncs +
                                      sum_cudagraph_dense_stream_syncs +
                                      sum_cudagraph_copy_stream_syncs;
        const bool output_head_outside_step =
            shared_output_head && shared_output_head->initialized &&
            shared_rank_buffers && shared_rank_buffers->initialized;
        const bool host_token_dependency =
            output_head_outside_step && serving_result &&
            serving_result->diagnostic_output_head;
        const int helper_host_sync_blocker_classes =
            (opt.tp_hc_current_input_gate && !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_projection_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_compressed_kv_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_state_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_typed_kv_history_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_raw_read_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_output_gate ? 1 : 0) +
            (opt.true_ds4_post_attention_ffn_input_gate ? 1 : 0) +
            (opt.final_hc_carry_gate && opt.tp_hc_final_expand_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0);
        const bool capture_replay_validated =
            sum_cudagraph_capture_attempted > 0 &&
            sum_cudagraph_capture_attempted == sum_cudagraph_capture_succeeded &&
            (!opt.decode_cudagraph_replay_probe_gate ||
             (sum_cudagraph_replay_attempted > 0 &&
              sum_cudagraph_replay_attempted == sum_cudagraph_replay_succeeded));
        const bool capture_eligible =
            capture_replay_validated ||
            (total_stream_syncs == 0 && sum_cudagraph_sync_all_calls == 0 &&
             helper_host_sync_blocker_classes == 0);
        const char *blocker = capture_eligible
            ? "none"
            : (total_stream_syncs != 0 || sum_cudagraph_sync_all_calls != 0
                   ? "host_stream_synchronization"
                   : "helper_host_synchronization");
        std::printf("tp_ep_decode_cudagraph_audit\tsteps\t%d\t"
                    "sync_all_calls\t%d\tevent_barrier_calls\t%d\t"
                    "stream_sync_count\t%d\t"
                    "rank_stream_sync_count\t%d\tdense_stream_sync_count\t%d\t"
                    "copy_stream_sync_count\t%d\toutput_head_outside_step\t%d\t"
                    "host_selected_token_dependency\t%d\t"
                    "helper_host_sync_blocker_classes\t%d\t"
                    "capture_attempted\t%d\tcapture_succeeded\t%d\t"
                    "capture_error_code\t%d\tcapture_error_name\t%s\t"
                    "capture_nodes\t%zu\t"
                    "replay_attempted\t%d\treplay_succeeded\t%d\t"
                    "replay_error_code\t%d\treplay_error_name\t%s\t"
                    "persistent_cache_hits\t%d\t"
                    "persistent_cache_misses\t%d\t"
                    "persistent_invalidations\t%d\t"
                    "persistent_invalidate_layer\t%d\t"
                    "persistent_invalidate_slots\t%d\t"
                    "persistent_invalidate_position\t%d\t"
                    "persistent_invalidate_root_device\t%d\t"
                    "persistent_invalidate_root_stream\t%d\t"
                    "sum_instantiate_ms\t%.6f\tsum_replay_ms\t%.6f\t"
                    "capture_eligible\t%d\tblocker\t%s\n",
                    graph_audit_steps,
                    sum_cudagraph_sync_all_calls,
                    sum_cudagraph_event_barrier_calls,
                    total_stream_syncs,
                    sum_cudagraph_rank_stream_syncs,
                    sum_cudagraph_dense_stream_syncs,
                    sum_cudagraph_copy_stream_syncs,
                    output_head_outside_step ? 1 : 0,
                    host_token_dependency ? 1 : 0,
                    helper_host_sync_blocker_classes,
                    sum_cudagraph_capture_attempted,
                    sum_cudagraph_capture_succeeded,
                    sum_cudagraph_capture_error,
                    cudaGetErrorName((cudaError_t)sum_cudagraph_capture_error),
                    sum_cudagraph_capture_nodes,
                    sum_cudagraph_replay_attempted,
                    sum_cudagraph_replay_succeeded,
                    sum_cudagraph_replay_error,
                    cudaGetErrorName((cudaError_t)sum_cudagraph_replay_error),
                    sum_cudagraph_persistent_cache_hits,
                    sum_cudagraph_persistent_cache_misses,
                    sum_cudagraph_persistent_invalidations,
                    sum_cudagraph_persistent_invalidate_layer,
                    sum_cudagraph_persistent_invalidate_slots,
                    sum_cudagraph_persistent_invalidate_position,
                    sum_cudagraph_persistent_invalidate_root_device,
                    sum_cudagraph_persistent_invalidate_root_stream,
                    sum_cudagraph_instantiate_ms,
                    sum_cudagraph_replay_ms,
                    capture_eligible ? 1 : 0,
                    blocker);
    }
    return 0;
}

static int http_write_json(int fd, int status, const char *body) {
    const char *reason = status == 200 ? "OK" : (status == 404 ? "Not Found" : "Error");
    const int n = dprintf(fd,
                          "HTTP/1.1 %d %s\r\n"
                          "Connection: close\r\n"
                          "Content-Type: application/json\r\n"
                          "Content-Length: %zu\r\n\r\n"
                          "%s",
                          status, reason, std::strlen(body), body);
    return n < 0 ? -1 : 0;
}

static int http_write_text(int fd, const char *body) {
    const int n = dprintf(fd,
                          "HTTP/1.1 200 OK\r\n"
                          "Connection: close\r\n"
                          "Content-Type: text/plain; version=0.0.4\r\n"
                          "Content-Length: %zu\r\n\r\n"
                          "%s",
                          std::strlen(body), body);
    return n < 0 ? -1 : 0;
}

static int json_find_int(const char *body, const char *key, int fallback) {
    if (!body || !key) return fallback;
    const char *p = std::strstr(body, key);
    if (!p) return fallback;
    p += std::strlen(key);
    while (*p && (*p == '"' || *p == '\'' || *p == ' ' || *p == '\t' || *p == ':')) ++p;
    char *end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p || v < 0 || v > 1000000) return fallback;
    return (int)v;
}

struct HttpParsedRequest {
    int fd = -1;
    std::string method;
    std::string path;
    std::string body;
    int requested_tokens = 0;
    std::string cache_key;
    bool cache_key_explicit = false;
    bool prompt_fingerprint_present = false;
    uint64_t prompt_fingerprint = 0;
    std::vector<uint32_t> prompt_token_ids;
    uint64_t cache_position = 0;
    int cache_slot = -1;
    bool cache_hit = false;
    bool cache_prompt_match = true;
    bool cache_evicted = false;
    std::string evicted_key;
    uint32_t decode_input_token = UINT32_MAX;
    std::vector<uint32_t> generated_token_ids;
    uint64_t prompt_prefill_tokens = 0;
};

static int http_content_length(const char *req) {
    const char *p = std::strstr(req, "Content-Length:");
    if (!p) return 0;
    p += std::strlen("Content-Length:");
    while (*p == ' ' || *p == '\t') ++p;
    char *end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p || v < 0 || v > 4096) return 0;
    return (int)v;
}

static std::string json_find_string(const char *body, const char *key) {
    if (!body || !key) return "";
    const char *p = std::strstr(body, key);
    if (!p) return "";
    p += std::strlen(key);
    while (*p && (*p == ' ' || *p == '\t' || *p == ':')) ++p;
    if (*p != '"' && *p != '\'') {
        while (*p && *p != '"' && *p != '\'') ++p;
    }
    if (!*p) return "";
    const char quote = *p++;
    std::string out;
    while (*p && *p != quote && out.size() < 256) {
        if (*p == '\\' && p[1]) {
            ++p;
        }
        out.push_back(*p++);
    }
    return out;
}

static uint64_t fnv1a64(const std::string &s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= (uint64_t)c;
        h *= 1099511628211ull;
    }
    return h;
}

static std::string http_json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out.push_back(c);
                }
                break;
        }
    }
    return out;
}

static std::string http_json_uint_array(const std::vector<uint32_t> &values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out += ",";
        out += std::to_string((unsigned long long)values[i]);
    }
    out += "]";
    return out;
}

static std::string http_json_u64_array(const std::vector<uint64_t> &values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out += ",";
        out += std::to_string((unsigned long long)values[i]);
    }
    out += "]";
    return out;
}

static bool http_is_chat_completion_post(const HttpParsedRequest &req);

struct TokenizerRuntime {
    ds4_engine *engine = nullptr;
    bool initialized = false;
};

static bool open_tokenizer_runtime(const char *model_path, TokenizerRuntime *out) {
    if (!model_path || !model_path[0] || !out) return false;
    ds4_engine_options opt;
    std::memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = DS4_BACKEND_CPU;
    opt.inspect_only = true;
    opt.n_threads = 1;
    if (ds4_engine_open(&out->engine, &opt) != 0 || !out->engine) {
        out->engine = nullptr;
        out->initialized = false;
        return false;
    }
    out->initialized = true;
    return true;
}

static void close_tokenizer_runtime(TokenizerRuntime *rt) {
    if (!rt) return;
    if (rt->engine) ds4_engine_close(rt->engine);
    rt->engine = nullptr;
    rt->initialized = false;
}

static std::string decode_token_text(ds4_engine *engine,
                                     const std::vector<uint32_t> &tokens) {
    if (!engine) return "";
    std::string out;
    for (uint32_t token : tokens) {
        size_t len = 0;
        char *piece = ds4_token_text(engine, (int)token, &len);
        if (piece && len > 0) out.append(piece, len);
        std::free(piece);
    }
    return out;
}

static bool materialize_prompt_tokens(ds4_engine *engine,
                                      HttpParsedRequest *req,
                                      std::string *err) {
    if (!req || !req->prompt_token_ids.empty()) return true;

    const std::string prompt = json_find_string(req->body.c_str(), "\"prompt\"");
    const std::string content = json_find_string(req->body.c_str(), "\"content\"");
    const bool has_text = !prompt.empty() || !content.empty();
    if (!has_text) return true;
    if (!engine) {
        if (err) *err = "text_prompt_requires_tokenizer";
        return false;
    }

    ds4_tokens toks;
    std::memset(&toks, 0, sizeof(toks));
    if (!content.empty() && http_is_chat_completion_post(*req)) {
        ds4_encode_chat_prompt(engine, "", content.c_str(), DS4_THINK_NONE, &toks);
    } else {
        ds4_tokenize_text(engine, prompt.c_str(), &toks);
    }
    req->prompt_token_ids.clear();
    req->prompt_token_ids.reserve((size_t)toks.len);
    for (int i = 0; i < toks.len; ++i) {
        if (toks.v[i] >= 0) req->prompt_token_ids.push_back((uint32_t)toks.v[i]);
    }
    ds4_tokens_free(&toks);
    if (req->prompt_token_ids.empty()) {
        if (err) *err = "tokenizer_produced_no_tokens";
        return false;
    }
    return true;
}

static bool http_read_request(int fd, HttpParsedRequest *out) {
    char req[8192];
    size_t used = 0;
    for (;;) {
        if (used + 1 >= sizeof(req)) return false;
        const ssize_t nr = read(fd, req + used, sizeof(req) - 1 - used);
        if (nr <= 0) return false;
        used += (size_t)nr;
        req[used] = '\0';
        const char *body = std::strstr(req, "\r\n\r\n");
        if (body) {
            const int content_length = http_content_length(req);
            const size_t header_bytes = (size_t)(body + 4 - req);
            if (used >= header_bytes + (size_t)content_length) break;
        }
    }
    char method[16] = {};
    char path[256] = {};
    if (std::sscanf(req, "%15s %255s", method, path) != 2) return false;
    const char *body = std::strstr(req, "\r\n\r\n");
    out->fd = fd;
    out->method = method;
    out->path = path;
    out->body = body ? body + 4 : "";
    return true;
}

static bool http_is_generation_post(const HttpParsedRequest &req) {
    return req.method == "POST" &&
           (req.path == "/v100/selected-token" ||
            req.path == "/v1/v100/selected-token" ||
            req.path == "/v1/completions" ||
            req.path == "/v1/chat/completions" ||
            req.path == "/v100/diagnostic-completions");
}

static bool http_is_completion_post(const HttpParsedRequest &req) {
    return req.method == "POST" &&
           (req.path == "/v1/completions" ||
            req.path == "/v100/diagnostic-completions");
}

static bool http_is_chat_completion_post(const HttpParsedRequest &req) {
    return req.method == "POST" && req.path == "/v1/chat/completions";
}

static bool http_wait_for_connection(int listen_fd, int wait_us) {
    if (wait_us <= 0) return false;
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(listen_fd, &rfds);
    timeval tv = {};
    tv.tv_sec = wait_us / 1000000;
    tv.tv_usec = wait_us % 1000000;
    const int rc = select(listen_fd + 1, &rfds, nullptr, nullptr, &tv);
    return rc > 0 && FD_ISSET(listen_fd, &rfds);
}

static int http_requested_tokens(const HttpParsedRequest &req, int fallback) {
    int out = json_find_int(req.body.c_str(), "max_tokens", fallback);
    if (out <= 0) out = fallback;
    if (out <= 0) out = 1;
    return out;
}

static std::string http_request_cache_key(const HttpParsedRequest &req,
                                          uint64_t request_serial,
                                          bool *explicit_key) {
    std::string key = json_find_string(req.body.c_str(), "\"session_id\"");
    if (key.empty()) key = json_find_string(req.body.c_str(), "\"cache_key\"");
    if (key.empty()) key = json_find_string(req.body.c_str(), "\"conversation_id\"");
    if (!key.empty()) {
        if (explicit_key) *explicit_key = true;
        return key;
    }

    const std::string prompt = json_find_string(req.body.c_str(), "\"prompt\"");
    if (!prompt.empty()) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "prompt:%016llx",
                      (unsigned long long)fnv1a64(prompt));
        if (explicit_key) *explicit_key = false;
        return buf;
    }

    char buf[64];
    std::snprintf(buf, sizeof(buf), "ephemeral:%llu",
                  (unsigned long long)request_serial);
    if (explicit_key) *explicit_key = false;
    return buf;
}

static bool json_find_uint_array(const char *body,
                                 const char *key,
                                 std::vector<uint32_t> *out,
                                 size_t limit) {
    out->clear();
    if (!body || !key) return false;
    const char *p = std::strstr(body, key);
    if (!p) return false;
    p += std::strlen(key);
    while (*p && *p != '[' && *p != '{' && *p != '"' && *p != '\'') ++p;
    if (*p != '[') return false;
    ++p;
    while (*p && *p != ']') {
        while (*p == ' ' || *p == '\t' || *p == '\n' ||
               *p == '\r' || *p == ',') ++p;
        if (*p == ']') break;
        char *end = nullptr;
        unsigned long v = std::strtoul(p, &end, 10);
        if (end == p || v > UINT32_MAX || out->size() >= limit) {
            out->clear();
            return false;
        }
        out->push_back((uint32_t)v);
        p = end;
        while (*p == ' ' || *p == '\t' || *p == '\n' ||
               *p == '\r') ++p;
        if (*p && *p != ',' && *p != ']') {
            out->clear();
            return false;
        }
    }
    return *p == ']' && !out->empty();
}

static uint64_t fnv1a64_u32(const std::vector<uint32_t> &tokens) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t token : tokens) {
        for (int i = 0; i < 4; ++i) {
            h ^= (uint64_t)((token >> (8 * i)) & 0xffu);
            h *= 1099511628211ull;
        }
    }
    return h;
}

static void http_request_prompt_fingerprint(HttpParsedRequest *req) {
    if (!req->prompt_token_ids.empty()) {
        req->prompt_fingerprint_present = true;
        req->prompt_fingerprint = fnv1a64_u32(req->prompt_token_ids);
        return;
    }
    if (json_find_uint_array(req->body.c_str(), "\"prompt_tokens\"",
                             &req->prompt_token_ids, 262144) ||
        json_find_uint_array(req->body.c_str(), "\"prompt\"",
                             &req->prompt_token_ids, 262144)) {
        req->prompt_fingerprint_present = true;
        req->prompt_fingerprint = fnv1a64_u32(req->prompt_token_ids);
        return;
    }

    const std::string prompt = json_find_string(req->body.c_str(), "\"prompt\"");
    if (prompt.empty()) {
        req->prompt_fingerprint_present = false;
        req->prompt_fingerprint = 0;
        req->prompt_token_ids.clear();
        return;
    }
    req->prompt_fingerprint_present = true;
    req->prompt_fingerprint = fnv1a64(prompt);
}

struct TpEpHttpSessionSlot {
    int id = -1;
    bool occupied = false;
    bool kv_valid = false;
    bool hc_valid = false;
    bool prompt_fingerprint_known = false;
    std::string key;
    uint64_t prompt_fingerprint = 0;
    std::vector<uint32_t> prompt_token_ids;
    std::vector<uint32_t> generated_token_ids;
    uint64_t pos = 0;
    uint64_t prompt_tokens = 0;
    uint64_t generated_tokens = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t last_used = 0;
    uint32_t last_selected_token = UINT32_MAX;
};

struct TpEpHttpSessionAssignment {
    int slot = -1;
    bool hit = false;
    bool prompt_match = true;
    bool evicted = false;
    std::string evicted_key;
    uint64_t pos_in = 0;
    uint64_t pos_out = 0;
};

struct TpEpHttpContextAdmission {
    bool ok = false;
    bool cache_hit = false;
    uint64_t start_position = 0;
    uint64_t prompt_prefill_steps = 0;
    uint64_t requested_decode_steps = 0;
    uint64_t final_position = 0;
    uint64_t ctx = 262144ull;
};

struct TpEpHttpSessionTable {
    std::vector<TpEpHttpSessionSlot> slots;
    uint64_t clock = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;

    void init(int n_slots) {
        slots.resize((size_t)n_slots);
        for (int i = 0; i < n_slots; ++i) {
            slots[(size_t)i].id = i;
        }
    }

    int find(const std::string &key) const {
        for (const auto &slot : slots) {
            if (slot.occupied && slot.key == key) return slot.id;
        }
        return -1;
    }

    bool slot_prompt_matches(const TpEpHttpSessionSlot &slot,
                             bool prompt_present,
                             uint64_t prompt_fingerprint) const {
        if (!prompt_present) return true;
        return slot.prompt_fingerprint_known &&
               slot.prompt_fingerprint == prompt_fingerprint;
    }

    uint64_t preview_position(const std::string &key,
                              bool prompt_present,
                              uint64_t prompt_fingerprint,
                              uint64_t base_pos) const {
        const int idx = find(key);
        if (idx >= 0 &&
            slots[(size_t)idx].kv_valid &&
            slots[(size_t)idx].hc_valid &&
            slot_prompt_matches(slots[(size_t)idx],
                                prompt_present,
                                prompt_fingerprint)) {
            return slots[(size_t)idx].pos;
        }
        return base_pos;
    }

    bool preview_hit(const std::string &key,
                     bool prompt_present,
                     uint64_t prompt_fingerprint,
                     uint64_t *pos_out) const {
        const int idx = find(key);
        if (idx >= 0 &&
            slots[(size_t)idx].kv_valid &&
            slots[(size_t)idx].hc_valid &&
            slot_prompt_matches(slots[(size_t)idx],
                                prompt_present,
                                prompt_fingerprint)) {
            if (pos_out) *pos_out = slots[(size_t)idx].pos;
            return true;
        }
        return false;
    }

    TpEpHttpSessionAssignment assign(const std::string &key,
                                     bool prompt_present,
                                     uint64_t prompt_fingerprint,
                                     const std::vector<uint32_t> &prompt_tokens,
                                     uint64_t base_pos,
                                     const std::vector<bool> &protected_slots) {
        TpEpHttpSessionAssignment a;
        ++clock;

        int idx = find(key);
        if (idx >= 0) {
            auto &slot = slots[(size_t)idx];
            const bool prompt_match =
                slot_prompt_matches(slot, prompt_present, prompt_fingerprint);
            a.slot = idx;
            a.prompt_match = prompt_match;
            a.hit = slot.kv_valid && slot.hc_valid && prompt_match;
            a.pos_in = a.hit ? slot.pos : base_pos;
            slot.last_used = clock;
            if (a.hit) {
                hits++;
                slot.hits++;
            } else {
                misses++;
                slot.misses++;
                slot.kv_valid = false;
                slot.hc_valid = false;
                slot.pos = base_pos;
                if (prompt_present) {
                    slot.prompt_fingerprint_known = true;
                    slot.prompt_fingerprint = prompt_fingerprint;
                }
                slot.prompt_token_ids = prompt_tokens;
                slot.generated_token_ids.clear();
                slot.prompt_tokens = 0;
                slot.generated_tokens = 0;
                slot.last_selected_token = UINT32_MAX;
            }
            return a;
        }

        for (auto &slot : slots) {
            if (!slot.occupied) {
                idx = slot.id;
                break;
            }
        }
        if (idx < 0) {
            uint64_t best_last = UINT64_MAX;
            for (const auto &slot : slots) {
                if (slot.id >= 0 && slot.id < (int)protected_slots.size() &&
                    protected_slots[(size_t)slot.id]) {
                    continue;
                }
                if (slot.last_used < best_last) {
                    best_last = slot.last_used;
                    idx = slot.id;
                }
            }
        }
        if (idx < 0) return a;

        auto &slot = slots[(size_t)idx];
        if (slot.occupied) {
            a.evicted = true;
            a.evicted_key = slot.key;
            evictions++;
        }
        slot.occupied = true;
        slot.kv_valid = false;
        slot.hc_valid = false;
        slot.prompt_fingerprint_known = prompt_present;
        slot.key = key;
        slot.prompt_fingerprint = prompt_present ? prompt_fingerprint : 0;
        slot.prompt_token_ids = prompt_tokens;
        slot.generated_token_ids.clear();
        slot.pos = base_pos;
        slot.prompt_tokens = 0;
        slot.generated_tokens = 0;
        slot.hits = 0;
        slot.misses = 1;
        slot.last_used = clock;
        slot.last_selected_token = UINT32_MAX;
        misses++;

        a.slot = idx;
        a.hit = false;
        a.prompt_match = true;
        a.pos_in = base_pos;
        return a;
    }

    void commit(const TpEpHttpSessionAssignment &a,
                uint64_t prompt_tokens,
                uint64_t generated_tokens,
                uint64_t position_advance,
                const std::vector<uint32_t> &selected_tokens) {
        if (a.slot < 0 || a.slot >= (int)slots.size()) return;
        auto &slot = slots[(size_t)a.slot];
        slot.kv_valid = true;
        slot.hc_valid = true;
        slot.pos = a.pos_in + position_advance;
        slot.prompt_tokens += prompt_tokens;
        slot.generated_tokens += generated_tokens;
        for (uint32_t selected_token : selected_tokens) {
            if (selected_token != UINT32_MAX) {
                slot.generated_token_ids.push_back(selected_token);
                slot.last_selected_token = selected_token;
            }
        }
        slot.last_used = ++clock;
    }

    int used() const {
        int n = 0;
        for (const auto &slot : slots) n += slot.occupied ? 1 : 0;
        return n;
    }

    void slots_json(char *out, size_t out_size) const {
        size_t used_bytes = 0;
        int n = std::snprintf(out, out_size,
                              "{\"slots_total\":%zu,\"slots_used\":%d,"
                              "\"cache_hits\":%llu,\"cache_misses\":%llu,"
                              "\"cache_evictions\":%llu,\"slots\":[",
                              slots.size(), used(),
                              (unsigned long long)hits,
                              (unsigned long long)misses,
                              (unsigned long long)evictions);
        if (n < 0) return;
        used_bytes = (size_t)std::min(n, (int)out_size);
        for (size_t i = 0; i < slots.size() && used_bytes + 256 < out_size; ++i) {
            const auto &slot = slots[i];
            const std::string key = http_json_escape(slot.key);
            n = std::snprintf(out + used_bytes, out_size - used_bytes,
                              "%s{\"id\":%d,\"occupied\":%d,\"key\":\"%s\","
                              "\"pos\":%llu,\"kv_valid\":%d,\"hc_valid\":%d,"
                              "\"prompt_fingerprint_known\":%d,"
                              "\"prompt_fingerprint\":%llu,"
                              "\"prompt_tokens\":%llu,\"generated_tokens\":%llu,"
                              "\"prompt_token_ids\":%zu,"
                              "\"generated_token_ids\":%zu,"
                              "\"last_selected_token\":%u,"
                              "\"hits\":%llu,\"misses\":%llu}",
                              i == 0 ? "" : ",",
                              slot.id, slot.occupied ? 1 : 0, key.c_str(),
                              (unsigned long long)slot.pos,
                              slot.kv_valid ? 1 : 0,
                              slot.hc_valid ? 1 : 0,
                              slot.prompt_fingerprint_known ? 1 : 0,
                              (unsigned long long)slot.prompt_fingerprint,
                              (unsigned long long)slot.prompt_tokens,
                              (unsigned long long)slot.generated_tokens,
                              slot.prompt_token_ids.size(),
                              slot.generated_token_ids.size(),
                              slot.last_selected_token,
                              (unsigned long long)slot.hits,
                              (unsigned long long)slot.misses);
            if (n < 0) break;
            used_bytes += (size_t)n;
        }
        if (used_bytes + 4 < out_size) {
            std::snprintf(out + used_bytes, out_size - used_bytes, "]}\n");
        }
    }
};

static TpEpHttpContextAdmission tp_ep_http_context_admission(
    const TpEpHttpSessionTable &sessions,
    const HttpParsedRequest &req,
    uint64_t base_position,
    uint64_t ctx) {
    TpEpHttpContextAdmission out;
    out.ctx = ctx;
    out.requested_decode_steps =
        req.requested_tokens > 0 ? (uint64_t)req.requested_tokens : 0ull;
    uint64_t hit_pos = 0;
    out.cache_hit = sessions.preview_hit(req.cache_key,
                                         req.prompt_fingerprint_present,
                                         req.prompt_fingerprint,
                                         &hit_pos);
    out.start_position = out.cache_hit ? hit_pos : base_position;
    if (!out.cache_hit && req.prompt_token_ids.size() > 1) {
        out.prompt_prefill_steps =
            (uint64_t)req.prompt_token_ids.size() - 1ull;
    }
    out.final_position = out.start_position + out.prompt_prefill_steps +
                         out.requested_decode_steps;
    out.ok = out.final_position <= out.ctx;
    return out;
}

static std::string tp_ep_http_context_error_json(
    const TpEpHttpContextAdmission &admission) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "{\"error\":\"context_window_exceeded\","
                  "\"ctx\":%llu,"
                  "\"start_position\":%llu,"
                  "\"prompt_prefill_steps\":%llu,"
                  "\"requested_decode_steps\":%llu,"
                  "\"final_position\":%llu,"
                  "\"cache_hit\":%d}\n",
                  (unsigned long long)admission.ctx,
                  (unsigned long long)admission.start_position,
                  (unsigned long long)admission.prompt_prefill_steps,
                  (unsigned long long)admission.requested_decode_steps,
                  (unsigned long long)admission.final_position,
                  admission.cache_hit ? 1 : 0);
    return std::string(buf);
}

static unsigned long long http_epoch_seconds() {
    using namespace std::chrono;
    return (unsigned long long)duration_cast<seconds>(
        system_clock::now().time_since_epoch()).count();
}

static void http_drain_matching_pending(std::deque<HttpParsedRequest> *pending,
                                        int requested_tokens,
                                        uint64_t cache_position,
                                        int max_batch,
                                        std::vector<HttpParsedRequest> *batch) {
    for (auto it = pending->begin();
         it != pending->end() && (int)batch->size() < max_batch;) {
        bool duplicate_key = false;
        for (const auto &req : *batch) {
            if (req.cache_key == it->cache_key) {
                duplicate_key = true;
                break;
            }
        }
        if (it->requested_tokens == requested_tokens &&
            it->cache_position == cache_position &&
            !duplicate_key) {
            batch->push_back(std::move(*it));
            it = pending->erase(it);
        } else {
            ++it;
        }
    }
}

int run_tp_ep_http_server(const Options &base_opt,
                          const DenseF16Cache *shared_dense_f16_cache,
                          const SharedApi *shared_api,
                          SharedRankBuffers *shared_rank_buffers,
                          SharedTpRuntime *shared_tp_runtime,
                          const SharedExpertBindings *shared_expert_bindings,
                          const SharedDenseOps *shared_dense_ops,
                          SharedOutputHead *shared_output_head,
                          SharedHcControls *shared_hc_controls,
                          SharedTokenEmbedding *shared_token_embedding,
                          std::vector<ContractRow> resident_rows[43],
                          LayerStats resident_stats[43]) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::perror("tp_ep_http_socket");
        return 30;
    }
    int yes = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)base_opt.port);
    if (inet_pton(AF_INET, base_opt.host, &addr.sin_addr) != 1) {
        std::fprintf(stderr, "tp_ep_http_bad_host\t%s\n", base_opt.host);
        close(listen_fd);
        return 31;
    }
    if (bind(listen_fd, (sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(listen_fd, 16) != 0) {
        std::perror("tp_ep_http_bind_listen");
        close(listen_fd);
        return 32;
    }

    uint64_t served = 0;
    uint64_t generation_requests = 0;
    uint64_t generation_batches = 0;
    uint64_t coalesced_requests = 0;
    uint64_t bucketed_requests = 0;
    uint64_t rejected = 0;
    TpEpHttpSessionTable sessions;
    sessions.init(base_opt.slots);
    std::deque<HttpParsedRequest> pending_generation;
    uint64_t next_position = base_opt.position;
    uint64_t total_prompt_tokens = 0;
    uint64_t total_generated_tokens = 0;
    uint64_t total_continuation_tokens = 0;
    double total_decode_ms = 0.0;
    double total_wall_ms = 0.0;
    double total_continuation_decode_ms = 0.0;
    double total_continuation_wall_ms = 0.0;
    double total_ep_ms = 0.0;
    double total_dense_ms = 0.0;
    double total_compose_ms = 0.0;
    double total_compose_reduce_ms = 0.0;
    double total_compose_copy_ms = 0.0;
    double total_compose_final_ms = 0.0;
    ServingBenchResult last = {};
    TokenizerRuntime tokenizer;
    if (base_opt.tokenizer_model_path && base_opt.tokenizer_model_path[0]) {
        if (!open_tokenizer_runtime(base_opt.tokenizer_model_path, &tokenizer)) {
            std::fprintf(stderr, "tp_ep_http tokenizer open failed: %s\n",
                         base_opt.tokenizer_model_path);
            close(listen_fd);
            return 31;
        }
        std::printf("tp_ep_http_tokenizer\tmodel\t%s\tPASS\n",
                    base_opt.tokenizer_model_path);
    }
    std::printf("tp_ep_http_serving\thttp://%s:%d/v100/selected-token\tPASS\n",
                base_opt.host, base_opt.port);
    std::printf("tp_ep_http_completions\thttp://%s:%d/v1/completions\tDIAGNOSTIC\n",
                base_opt.host, base_opt.port);
    std::printf("tp_ep_http_chat_completions\thttp://%s:%d/v1/chat/completions\tDIAGNOSTIC\n",
                base_opt.host, base_opt.port);
    std::fflush(stdout);

    while (base_opt.max_requests == 0 || (int)served < base_opt.max_requests ||
           !pending_generation.empty()) {
        HttpParsedRequest first_req;
        int fd = -1;
        if (!pending_generation.empty()) {
            first_req = std::move(pending_generation.front());
            pending_generation.pop_front();
            fd = first_req.fd;
        } else {
            fd = accept(listen_fd, nullptr, nullptr);
            if (fd < 0) {
                if (errno == EINTR) continue;
                std::perror("tp_ep_http_accept");
                break;
            }
            served++;
            if (!http_read_request(fd, &first_req)) {
                close(fd);
                continue;
            }
        }

        if (first_req.method == "GET" && first_req.path == "/health") {
            http_write_json(fd, 200, "{\"status\":\"ok\",\"backend\":\"tp_ep_resident\"}\n");
        } else if (first_req.method == "GET" &&
                   (first_req.path == "/status" || first_req.path == "/v100/status")) {
            const double cumulative_generated_tok_s_wall = total_wall_ms > 0.0
                ? (double)total_generated_tokens * 1000.0 / total_wall_ms
                : 0.0;
            const double cumulative_generated_tok_s_decode = total_decode_ms > 0.0
                ? (double)total_generated_tokens * 1000.0 / total_decode_ms
                : 0.0;
            const double cumulative_continuation_tok_s_wall = total_continuation_wall_ms > 0.0
                ? (double)total_continuation_tokens * 1000.0 / total_continuation_wall_ms
                : 0.0;
            const double cumulative_continuation_tok_s_decode = total_continuation_decode_ms > 0.0
                ? (double)total_continuation_tokens * 1000.0 / total_continuation_decode_ms
                : 0.0;
            const PeerCopySnapshot peer = peer_copy_snapshot();
            char out[8192];
            std::snprintf(out, sizeof(out),
                          "{\"status\":\"ok\",\"backend\":\"tp_ep_resident\","
                          "\"tp\":8,\"ep\":8,\"pp\":1,\"ctx\":262144,"
                          "\"slots\":%d,\"served_requests\":%llu,"
                          "\"generation_requests\":%llu,\"generation_batches\":%llu,"
                          "\"coalesced_requests\":%llu,\"bucketed_requests\":%llu,"
                          "\"pending_generation_requests\":%zu,"
                          "\"microbatch_wait_us\":%d,"
                          "\"kv_runtime_resident\":%d,"
                          "\"kv_all_slots_gate\":%d,"
                          "\"hc_persist_state_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_raw_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_compressed_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_indexer_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_history_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_skip_current_load_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_skip_raw_store_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_skip_compressed_store_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_skip_indexer_store_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_quiet_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_batch_rows_gate\":%d,"
                          "\"true_ds4_attention_typed_kv_stream_sync_gate\":%d,"
                          "\"fp8_e5m2_kv_gate\":%d,"
                          "\"router_hash_fast_gate\":%d,"
                          "\"route_plan_async_upload_gate\":%d,"
                          "\"cache_slots_total\":%zu,"
                          "\"cache_slots_used\":%d,"
                          "\"cache_hits\":%llu,"
                          "\"cache_misses\":%llu,"
                          "\"cache_evictions\":%llu,"
                          "\"peer_copy_accounting\":%d,"
                          "\"peer_copy_reject_sys\":%d,"
                          "\"peer_copy_ops\":%llu,"
                          "\"peer_copy_bytes\":%llu,"
                          "\"peer_copy_nv1_ops\":%llu,"
                          "\"peer_copy_nv1_bytes\":%llu,"
                          "\"peer_copy_nv2_ops\":%llu,"
                          "\"peer_copy_nv2_bytes\":%llu,"
                          "\"peer_copy_sys_ops\":%llu,"
                          "\"peer_copy_sys_bytes\":%llu,"
                          "\"peer_copy_unknown_ops\":%llu,"
                          "\"peer_copy_unknown_bytes\":%llu,"
                          "\"peer_copy_first_sys_src\":%d,"
                          "\"peer_copy_first_sys_dst\":%d,"
                          "\"peer_copy_first_sys_bytes\":%llu,"
                          "\"peer_copy_first_sys_site\":\"%s\","
                          "\"peer_copy_first_sys_line\":%d,"
                          "\"peer_copy_top_sys_site\":\"%s\","
                          "\"peer_copy_top_sys_site_line\":%d,"
                          "\"peer_copy_top_sys_site_ops\":%llu,"
                          "\"peer_copy_top_sys_site_bytes\":%llu,"
                          "\"peer_copy_top_sys_site_total_ops\":%llu,"
                          "\"peer_copy_top_sys_site_total_bytes\":%llu,"
                          "\"rejected_requests\":%llu,"
                          "\"total_prompt_tokens\":%llu,"
                          "\"total_generated_tokens\":%llu,"
                          "\"total_continuation_tokens\":%llu,"
                          "\"next_position\":%llu,"
                          "\"warmed_ready\":true,\"resident_ready\":true,"
                          "\"last_generated_tok_s_wall\":%.6f,"
                          "\"last_continuation_tok_s_wall\":%.6f,"
                          "\"last_compose_copy_ms\":%.6f,"
                          "\"cumulative_generated_tok_s_wall\":%.6f,"
                          "\"cumulative_continuation_tok_s_wall\":%.6f,"
                          "\"cumulative_generated_tok_s_decode\":%.6f,"
                          "\"cumulative_continuation_tok_s_decode\":%.6f,"
                          "\"cumulative_ep_ms\":%.6f,"
                          "\"cumulative_dense_ms\":%.6f,"
                          "\"cumulative_compose_ms\":%.6f,"
                          "\"cumulative_compose_reduce_ms\":%.6f,"
                          "\"cumulative_compose_copy_ms\":%.6f,"
                          "\"cumulative_compose_final_ms\":%.6f}\n",
                          base_opt.slots,
                          (unsigned long long)served,
                          (unsigned long long)generation_requests,
                          (unsigned long long)generation_batches,
                          (unsigned long long)coalesced_requests,
                          (unsigned long long)bucketed_requests,
                          pending_generation.size(),
                          base_opt.microbatch_wait_us,
                          shared_tp_runtime && shared_tp_runtime->initialized ? 1 : 0,
                          base_opt.tp_kv_all_slots_gate ? 1 : 0,
                          base_opt.tp_hc_persist_state_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_raw_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_compressed_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_indexer_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_history_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_current_load_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_raw_store_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_compressed_store_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_indexer_store_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_quiet_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_batch_rows_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_stream_sync_gate ? 1 : 0,
                          base_opt.fp8_e5m2_kv_gate ? 1 : 0,
                          base_opt.router_hash_fast_gate ? 1 : 0,
                          base_opt.route_plan_async_upload_gate ? 1 : 0,
                          sessions.slots.size(),
                          sessions.used(),
                          (unsigned long long)sessions.hits,
                          (unsigned long long)sessions.misses,
                          (unsigned long long)sessions.evictions,
                          g_peer_copy_accounting.enabled.load(std::memory_order_relaxed),
                          g_peer_copy_accounting.reject_sys.load(std::memory_order_relaxed),
                          (unsigned long long)peer.ops,
                          (unsigned long long)peer.bytes,
                          (unsigned long long)peer.nv1_ops,
                          (unsigned long long)peer.nv1_bytes,
                          (unsigned long long)peer.nv2_ops,
                          (unsigned long long)peer.nv2_bytes,
                          (unsigned long long)peer.sys_ops,
                          (unsigned long long)peer.sys_bytes,
                          (unsigned long long)peer.unknown_ops,
                          (unsigned long long)peer.unknown_bytes,
                          peer.first_sys_src,
                          peer.first_sys_dst,
                          (unsigned long long)peer.first_sys_bytes,
                          peer.first_sys_site ? peer.first_sys_site : "-",
                          peer.first_sys_line,
                          peer.top_sys_site ? peer.top_sys_site : "-",
                          peer.top_sys_site_line,
                          (unsigned long long)peer.top_sys_site_ops,
                          (unsigned long long)peer.top_sys_site_bytes,
                          (unsigned long long)peer.top_sys_site_total_ops,
                          (unsigned long long)peer.top_sys_site_total_bytes,
                          (unsigned long long)rejected,
                          (unsigned long long)total_prompt_tokens,
                          (unsigned long long)total_generated_tokens,
                          (unsigned long long)total_continuation_tokens,
                          (unsigned long long)next_position,
                          last.aggregate_generated_tok_s_wall,
                          last.aggregate_continuation_tok_s_wall,
                          last.total_compose_copy_ms,
                          cumulative_generated_tok_s_wall,
                          cumulative_continuation_tok_s_wall,
                          cumulative_generated_tok_s_decode,
                          cumulative_continuation_tok_s_decode,
                          total_ep_ms,
                          total_dense_ms,
                          total_compose_ms,
                          total_compose_reduce_ms,
                          total_compose_copy_ms,
                          total_compose_final_ms);
            http_write_json(fd, 200, out);
        } else if (first_req.method == "GET" && first_req.path == "/v100/slots") {
            char out[16384];
            sessions.slots_json(out, sizeof(out));
            http_write_json(fd, 200, out);
        } else if (first_req.method == "GET" && first_req.path == "/metrics") {
            const double cumulative_generated_tok_s_wall = total_wall_ms > 0.0
                ? (double)total_generated_tokens * 1000.0 / total_wall_ms
                : 0.0;
            const double cumulative_generated_tok_s_decode = total_decode_ms > 0.0
                ? (double)total_generated_tokens * 1000.0 / total_decode_ms
                : 0.0;
            const double cumulative_continuation_tok_s_wall = total_continuation_wall_ms > 0.0
                ? (double)total_continuation_tokens * 1000.0 / total_continuation_wall_ms
                : 0.0;
            const double cumulative_continuation_tok_s_decode = total_continuation_decode_ms > 0.0
                ? (double)total_continuation_tokens * 1000.0 / total_continuation_decode_ms
                : 0.0;
            const PeerCopySnapshot peer = peer_copy_snapshot();
            char out[8192];
            std::snprintf(out, sizeof(out),
                          "ds4_v100_tp_ep_resident_ready 1\n"
                          "ds4_v100_tp_ep_slots %d\n"
                          "ds4_v100_tp_ep_served_requests %llu\n"
                          "ds4_v100_tp_ep_generation_requests %llu\n"
                          "ds4_v100_tp_ep_generation_batches %llu\n"
                          "ds4_v100_tp_ep_coalesced_requests %llu\n"
                          "ds4_v100_tp_ep_bucketed_requests %llu\n"
                          "ds4_v100_tp_ep_pending_generation_requests %zu\n"
                          "ds4_v100_tp_ep_microbatch_wait_us %d\n"
                          "ds4_v100_tp_ep_kv_runtime_resident %d\n"
                          "ds4_v100_tp_ep_kv_all_slots_gate %d\n"
                          "ds4_v100_tp_ep_hc_persist_state_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_raw_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_compressed_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_indexer_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_history_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_skip_current_load_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_skip_raw_store_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_skip_compressed_store_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_skip_indexer_store_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_quiet_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_batch_rows_gate %d\n"
                          "ds4_v100_tp_ep_true_ds4_attention_typed_kv_stream_sync_gate %d\n"
                          "ds4_v100_tp_ep_fp8_e5m2_kv_gate %d\n"
                          "ds4_v100_tp_ep_router_hash_fast_gate %d\n"
                          "ds4_v100_tp_ep_route_plan_async_upload_gate %d\n"
                          "ds4_v100_tp_ep_cache_slots_total %zu\n"
                          "ds4_v100_tp_ep_cache_slots_used %d\n"
                          "ds4_v100_tp_ep_cache_hits %llu\n"
                          "ds4_v100_tp_ep_cache_misses %llu\n"
                          "ds4_v100_tp_ep_cache_evictions %llu\n"
                          "ds4_v100_tp_ep_peer_copy_accounting %d\n"
                          "ds4_v100_tp_ep_peer_copy_reject_sys %d\n"
                          "ds4_v100_tp_ep_peer_copy_ops %llu\n"
                          "ds4_v100_tp_ep_peer_copy_bytes %llu\n"
                          "ds4_v100_tp_ep_peer_copy_nv1_ops %llu\n"
                          "ds4_v100_tp_ep_peer_copy_nv1_bytes %llu\n"
                          "ds4_v100_tp_ep_peer_copy_nv2_ops %llu\n"
                          "ds4_v100_tp_ep_peer_copy_nv2_bytes %llu\n"
                          "ds4_v100_tp_ep_peer_copy_sys_ops %llu\n"
                          "ds4_v100_tp_ep_peer_copy_sys_bytes %llu\n"
                          "ds4_v100_tp_ep_peer_copy_unknown_ops %llu\n"
                          "ds4_v100_tp_ep_peer_copy_unknown_bytes %llu\n"
                          "ds4_v100_tp_ep_peer_copy_first_sys_src %d\n"
                          "ds4_v100_tp_ep_peer_copy_first_sys_dst %d\n"
                          "ds4_v100_tp_ep_peer_copy_first_sys_bytes %llu\n"
                          "ds4_v100_tp_ep_peer_copy_first_sys_line %d\n"
                          "ds4_v100_tp_ep_peer_copy_top_sys_site_line %d\n"
                          "ds4_v100_tp_ep_peer_copy_top_sys_site_ops %llu\n"
                          "ds4_v100_tp_ep_peer_copy_top_sys_site_bytes %llu\n"
                          "ds4_v100_tp_ep_peer_copy_top_sys_site_total_ops %llu\n"
                          "ds4_v100_tp_ep_peer_copy_top_sys_site_total_bytes %llu\n"
                          "ds4_v100_tp_ep_rejected_requests %llu\n"
                          "ds4_v100_tp_ep_total_prompt_tokens %llu\n"
                          "ds4_v100_tp_ep_total_generated_tokens %llu\n"
                          "ds4_v100_tp_ep_total_continuation_tokens %llu\n"
                          "ds4_v100_tp_ep_next_position %llu\n"
                          "ds4_v100_tp_ep_generated_tok_s_wall %.6f\n"
                          "ds4_v100_tp_ep_continuation_tok_s_wall %.6f\n"
                          "ds4_v100_tp_ep_last_compose_copy_ms %.6f\n"
                          "ds4_v100_tp_ep_cumulative_generated_tok_s_wall %.6f\n"
                          "ds4_v100_tp_ep_cumulative_continuation_tok_s_wall %.6f\n"
                          "ds4_v100_tp_ep_cumulative_generated_tok_s_decode %.6f\n"
                          "ds4_v100_tp_ep_cumulative_continuation_tok_s_decode %.6f\n"
                          "ds4_v100_tp_ep_cumulative_ep_ms %.6f\n"
                          "ds4_v100_tp_ep_cumulative_dense_ms %.6f\n"
                          "ds4_v100_tp_ep_cumulative_compose_ms %.6f\n"
                          "ds4_v100_tp_ep_cumulative_compose_reduce_ms %.6f\n"
                          "ds4_v100_tp_ep_cumulative_compose_copy_ms %.6f\n"
                          "ds4_v100_tp_ep_cumulative_compose_final_ms %.6f\n",
                          base_opt.slots,
                          (unsigned long long)served,
                          (unsigned long long)generation_requests,
                          (unsigned long long)generation_batches,
                          (unsigned long long)coalesced_requests,
                          (unsigned long long)bucketed_requests,
                          pending_generation.size(),
                          base_opt.microbatch_wait_us,
                          shared_tp_runtime && shared_tp_runtime->initialized ? 1 : 0,
                          base_opt.tp_kv_all_slots_gate ? 1 : 0,
                          base_opt.tp_hc_persist_state_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_raw_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_compressed_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_indexer_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_history_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_current_load_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_raw_store_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_compressed_store_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_skip_indexer_store_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_quiet_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_batch_rows_gate ? 1 : 0,
                          base_opt.true_ds4_attention_typed_kv_stream_sync_gate ? 1 : 0,
                          base_opt.fp8_e5m2_kv_gate ? 1 : 0,
                          base_opt.router_hash_fast_gate ? 1 : 0,
                          base_opt.route_plan_async_upload_gate ? 1 : 0,
                          sessions.slots.size(),
                          sessions.used(),
                          (unsigned long long)sessions.hits,
                          (unsigned long long)sessions.misses,
                          (unsigned long long)sessions.evictions,
                          g_peer_copy_accounting.enabled.load(std::memory_order_relaxed),
                          g_peer_copy_accounting.reject_sys.load(std::memory_order_relaxed),
                          (unsigned long long)peer.ops,
                          (unsigned long long)peer.bytes,
                          (unsigned long long)peer.nv1_ops,
                          (unsigned long long)peer.nv1_bytes,
                          (unsigned long long)peer.nv2_ops,
                          (unsigned long long)peer.nv2_bytes,
                          (unsigned long long)peer.sys_ops,
                          (unsigned long long)peer.sys_bytes,
                          (unsigned long long)peer.unknown_ops,
                          (unsigned long long)peer.unknown_bytes,
                          peer.first_sys_src,
                          peer.first_sys_dst,
                          (unsigned long long)peer.first_sys_bytes,
                          peer.first_sys_line,
                          peer.top_sys_site_line,
                          (unsigned long long)peer.top_sys_site_ops,
                          (unsigned long long)peer.top_sys_site_bytes,
                          (unsigned long long)peer.top_sys_site_total_ops,
                          (unsigned long long)peer.top_sys_site_total_bytes,
                          (unsigned long long)rejected,
                          (unsigned long long)total_prompt_tokens,
                          (unsigned long long)total_generated_tokens,
                          (unsigned long long)total_continuation_tokens,
                          (unsigned long long)next_position,
                          last.aggregate_generated_tok_s_wall,
                          last.aggregate_continuation_tok_s_wall,
                          last.total_compose_copy_ms,
                          cumulative_generated_tok_s_wall,
                          cumulative_continuation_tok_s_wall,
                          cumulative_generated_tok_s_decode,
                          cumulative_continuation_tok_s_decode,
                          total_ep_ms,
                          total_dense_ms,
                          total_compose_ms,
                          total_compose_reduce_ms,
                          total_compose_copy_ms,
                          total_compose_final_ms);
            http_write_text(fd, out);
        } else if (http_is_generation_post(first_req)) {
            std::string prompt_error;
            if (!materialize_prompt_tokens(tokenizer.engine, &first_req, &prompt_error)) {
                std::string body = "{\"error\":\"" + http_json_escape(prompt_error) + "\"}\n";
                http_write_json(first_req.fd, 400, body.c_str());
                close(first_req.fd);
                rejected++;
                continue;
            }
            first_req.requested_tokens = http_requested_tokens(first_req, base_opt.decode_steps);
            first_req.cache_key = http_request_cache_key(first_req,
                                                         served + pending_generation.size(),
                                                         &first_req.cache_key_explicit);
            http_request_prompt_fingerprint(&first_req);
            first_req.cache_position =
                sessions.preview_position(first_req.cache_key,
                                          first_req.prompt_fingerprint_present,
                                          first_req.prompt_fingerprint,
                                          base_opt.position);
            {
                const TpEpHttpContextAdmission admission =
                    tp_ep_http_context_admission(sessions, first_req,
                                                 base_opt.position, 262144ull);
                if (!admission.ok) {
                    std::fprintf(stderr,
                                 "tp_ep_http_context_rejected\tstart_position\t%llu\t"
                                 "prompt_prefill_steps\t%llu\trequested_steps\t%llu\t"
                                 "final_position\t%llu\tctx\t%llu\tcache_hit\t%d\n",
                                 (unsigned long long)admission.start_position,
                                 (unsigned long long)admission.prompt_prefill_steps,
                                 (unsigned long long)admission.requested_decode_steps,
                                 (unsigned long long)admission.final_position,
                                 (unsigned long long)admission.ctx,
                                 admission.cache_hit ? 1 : 0);
                    const std::string body =
                        tp_ep_http_context_error_json(admission);
                    http_write_json(first_req.fd, 400, body.c_str());
                    close(first_req.fd);
                    rejected++;
                    continue;
                }
            }

            std::vector<HttpParsedRequest> batch;
            batch.push_back(first_req);
            http_drain_matching_pending(&pending_generation,
                                        first_req.requested_tokens,
                                        first_req.cache_position,
                                        base_opt.slots,
                                        &batch);
            while ((int)batch.size() < base_opt.slots &&
                   http_wait_for_connection(listen_fd, base_opt.microbatch_wait_us)) {
                int extra_fd = accept(listen_fd, nullptr, nullptr);
                if (extra_fd < 0) {
                    if (errno == EINTR) continue;
                    break;
                }
                served++;
                HttpParsedRequest extra_req;
                if (!http_read_request(extra_fd, &extra_req)) {
                    close(extra_fd);
                    continue;
                }
                if (!http_is_generation_post(extra_req)) {
                    rejected++;
                    http_write_json(extra_fd, 404, "{\"error\":\"not_found_during_coalesce\"}\n");
                    close(extra_fd);
                    continue;
                }
                std::string extra_prompt_error;
                if (!materialize_prompt_tokens(tokenizer.engine, &extra_req, &extra_prompt_error)) {
                    std::string body = "{\"error\":\"" + http_json_escape(extra_prompt_error) + "\"}\n";
                    http_write_json(extra_fd, 400, body.c_str());
                    close(extra_fd);
                    rejected++;
                    continue;
                }
                extra_req.requested_tokens = http_requested_tokens(extra_req, first_req.requested_tokens);
                extra_req.cache_key = http_request_cache_key(extra_req,
                                                            served + pending_generation.size(),
                                                            &extra_req.cache_key_explicit);
                http_request_prompt_fingerprint(&extra_req);
                extra_req.cache_position =
                    sessions.preview_position(extra_req.cache_key,
                                              extra_req.prompt_fingerprint_present,
                                              extra_req.prompt_fingerprint,
                                              base_opt.position);
                {
                    const TpEpHttpContextAdmission admission =
                        tp_ep_http_context_admission(sessions, extra_req,
                                                     base_opt.position, 262144ull);
                    if (!admission.ok) {
                        std::fprintf(stderr,
                                     "tp_ep_http_context_rejected\tstart_position\t%llu\t"
                                     "prompt_prefill_steps\t%llu\trequested_steps\t%llu\t"
                                     "final_position\t%llu\tctx\t%llu\tcache_hit\t%d\n",
                                     (unsigned long long)admission.start_position,
                                     (unsigned long long)admission.prompt_prefill_steps,
                                     (unsigned long long)admission.requested_decode_steps,
                                     (unsigned long long)admission.final_position,
                                     (unsigned long long)admission.ctx,
                                     admission.cache_hit ? 1 : 0);
                        const std::string body =
                            tp_ep_http_context_error_json(admission);
                        http_write_json(extra_fd, 400, body.c_str());
                        close(extra_fd);
                        rejected++;
                        continue;
                    }
                }
                if (extra_req.requested_tokens != first_req.requested_tokens) {
                    bucketed_requests++;
                    pending_generation.push_back(std::move(extra_req));
                    continue;
                }
                if (extra_req.cache_position != first_req.cache_position) {
                    bucketed_requests++;
                    pending_generation.push_back(std::move(extra_req));
                    continue;
                }
                bool duplicate_key = false;
                for (const auto &req : batch) {
                    if (req.cache_key == extra_req.cache_key) {
                        duplicate_key = true;
                        break;
                    }
                }
                if (duplicate_key) {
                    bucketed_requests++;
                    pending_generation.push_back(std::move(extra_req));
                    continue;
                }
                batch.push_back(extra_req);
            }

            std::vector<TpEpHttpSessionAssignment> assignments(batch.size());
            std::vector<bool> protected_slots((size_t)base_opt.slots, false);
            bool assignment_failed = false;
            for (size_t i = 0; i < batch.size(); ++i) {
                assignments[i] = sessions.assign(batch[i].cache_key,
                                                 batch[i].prompt_fingerprint_present,
                                                 batch[i].prompt_fingerprint,
                                                 batch[i].prompt_token_ids,
                                                 base_opt.position,
                                                 protected_slots);
                if (assignments[i].slot < 0) {
                    assignment_failed = true;
                    break;
                }
                batch[i].cache_slot = assignments[i].slot;
                batch[i].cache_hit = assignments[i].hit;
                batch[i].cache_prompt_match = assignments[i].prompt_match;
                batch[i].cache_evicted = assignments[i].evicted;
                batch[i].evicted_key = assignments[i].evicted_key;
                batch[i].cache_position = assignments[i].pos_in;
                protected_slots[(size_t)assignments[i].slot] = true;
            }
            if (assignment_failed) {
                rejected += (uint64_t)batch.size();
                for (HttpParsedRequest &queued : batch) {
                    http_write_json(queued.fd, 503, "{\"error\":\"no_cache_slot_available\"}\n");
                    close(queued.fd);
                }
                continue;
            }

            Options req_opt = base_opt;
            const int requested_decode_steps = first_req.requested_tokens;
            req_opt.decode_steps = 1;
            req_opt.slots = base_opt.slots;
            req_opt.position = first_req.cache_position;
            req_opt.serving_bench = false;
            std::vector<uint32_t> decode_input_tokens((size_t)req_opt.slots, 0u);
            std::vector<unsigned char> decode_active_slots((size_t)req_opt.slots, 0u);
            for (size_t i = 0; i < batch.size(); ++i) {
                uint32_t input_token = 0;
                if (assignments[i].slot >= 0 &&
                    assignments[i].slot < (int)sessions.slots.size()) {
                    const TpEpHttpSessionSlot &slot =
                        sessions.slots[(size_t)assignments[i].slot];
                    if (assignments[i].hit &&
                        slot.last_selected_token != UINT32_MAX) {
                        input_token = slot.last_selected_token;
                    } else if (!batch[i].prompt_token_ids.empty()) {
                        input_token = batch[i].prompt_token_ids.back();
                    } else if (!slot.prompt_token_ids.empty()) {
                        input_token = slot.prompt_token_ids.back();
                    }
                } else if (!batch[i].prompt_token_ids.empty()) {
                    input_token = batch[i].prompt_token_ids.back();
                }
                batch[i].decode_input_token = input_token;
                if (batch[i].cache_slot >= 0 &&
                    batch[i].cache_slot < req_opt.slots) {
                    decode_input_tokens[(size_t)batch[i].cache_slot] = input_token;
                    decode_active_slots[(size_t)batch[i].cache_slot] = 1u;
                }
            }
            int max_prompt_prefill_steps = 0;
            int rc = 0;
            for (size_t i = 0; i < batch.size(); ++i) {
                if (!assignments[i].hit && batch[i].prompt_token_ids.size() > 1) {
                    max_prompt_prefill_steps = std::max(
                        max_prompt_prefill_steps,
                        (int)batch[i].prompt_token_ids.size() - 1);
                }
            }
            for (int prefill_step = 0; prefill_step < max_prompt_prefill_steps; ++prefill_step) {
                bool any_prefill = false;
                std::vector<uint32_t> prefill_input_tokens((size_t)req_opt.slots, 0u);
                for (size_t i = 0; i < batch.size(); ++i) {
                    const int slot = batch[i].cache_slot;
                    if (assignments[i].hit || slot < 0 || slot >= req_opt.slots) continue;
                    if ((size_t)prefill_step + 1u >= batch[i].prompt_token_ids.size()) continue;
                    const uint32_t tok = batch[i].prompt_token_ids[(size_t)prefill_step];
                    prefill_input_tokens[(size_t)slot] = tok;
                    batch[i].prompt_prefill_tokens++;
                    any_prefill = true;
                }
                if (!any_prefill) continue;
                Options prefill_opt = req_opt;
                prefill_opt.position = first_req.cache_position + (uint64_t)prefill_step;
                prefill_opt.diagnostic_output_head = false;
                prefill_opt.diagnostic_output_head_lazy_gate = false;
                std::vector<unsigned char> prefill_active_slots((size_t)req_opt.slots, 0u);
                for (size_t i = 0; i < batch.size(); ++i) {
                    const int slot = batch[i].cache_slot;
                    if (assignments[i].hit || slot < 0 || slot >= req_opt.slots) continue;
                    if ((size_t)prefill_step + 1u >= batch[i].prompt_token_ids.size()) continue;
                    prefill_active_slots[(size_t)slot] = 1u;
                }
                ServingBenchResult prefill_result;
                rc = run_token_major_serving_loop(prefill_opt,
                                                  shared_dense_f16_cache,
                                                  shared_api,
                                                  shared_rank_buffers,
                                                  shared_tp_runtime,
                                                  shared_expert_bindings,
                                                  shared_dense_ops,
                                                  nullptr,
                                                  shared_hc_controls,
                                                  shared_token_embedding,
                                                  &prefill_input_tokens,
                                                  &prefill_active_slots,
                                                  resident_rows,
                                                  resident_stats,
                                                  true,
                                                  &prefill_result);
                if (rc != 0) break;
                total_decode_ms += prefill_result.total_decode_ms;
                total_wall_ms += prefill_result.total_wall_ms;
                total_ep_ms += prefill_result.total_ep_ms;
                total_dense_ms += prefill_result.total_dense_ms;
                total_compose_ms += prefill_result.total_compose_ms;
                total_compose_reduce_ms += prefill_result.total_compose_reduce_ms;
                total_compose_copy_ms += prefill_result.total_compose_copy_ms;
                total_compose_final_ms += prefill_result.total_compose_final_ms;
            }
            ServingBenchResult result;
            bool missing_output_head = false;
            for (int step = 0; rc == 0 && step < requested_decode_steps; ++step) {
                req_opt.position = first_req.cache_position +
                                   (uint64_t)max_prompt_prefill_steps +
                                   (uint64_t)step;
                ServingBenchResult step_result;
                rc = run_token_major_serving_loop(req_opt,
                                                  shared_dense_f16_cache,
                                                  shared_api,
                                                  shared_rank_buffers,
                                                  shared_tp_runtime,
                                                  shared_expert_bindings,
                                                  shared_dense_ops,
                                                  shared_output_head,
                                                  shared_hc_controls,
                                                  shared_token_embedding,
                                                  &decode_input_tokens,
                                                  &decode_active_slots,
                                                  resident_rows,
                                                  resident_stats,
                                                  true,
                                                  &step_result);
                if (rc != 0) break;
                if (!step_result.diagnostic_output_head ||
                    step_result.selected_tokens.size() < (size_t)req_opt.slots) {
                    missing_output_head = true;
                    break;
                }
                result.prompt_tokens = step == 0 ? step_result.prompt_tokens : result.prompt_tokens;
                result.generated_tokens += (uint64_t)req_opt.slots;
                result.continuation_tokens = requested_decode_steps > 1
                    ? (uint64_t)req_opt.slots * (uint64_t)(requested_decode_steps - 1)
                    : 0ull;
                if (step == 0) {
                    result.first_token_decode_ms += step_result.first_token_decode_ms;
                    result.first_token_wall_ms += step_result.first_token_wall_ms;
                } else {
                    result.continuation_decode_ms += step_result.first_token_decode_ms;
                    result.continuation_wall_ms += step_result.first_token_wall_ms;
                }
                result.total_decode_ms += step_result.total_decode_ms;
                result.total_wall_ms += step_result.total_wall_ms;
                result.total_ep_ms += step_result.total_ep_ms;
                result.total_dense_ms += step_result.total_dense_ms;
                result.total_compose_ms += step_result.total_compose_ms;
                result.total_compose_reduce_ms += step_result.total_compose_reduce_ms;
                result.total_compose_copy_ms += step_result.total_compose_copy_ms;
                result.total_compose_final_ms += step_result.total_compose_final_ms;
                result.total_hc_current_input_ms += step_result.total_hc_current_input_ms;
                result.diagnostic_output_head = step_result.diagnostic_output_head;
                result.diagnostic_output_head_proxy_hc =
                    step_result.diagnostic_output_head_proxy_hc;
                result.output_head_ms += step_result.output_head_ms;
                result.output_head_gather_ms += step_result.output_head_gather_ms;
                result.output_head_prep_ms += step_result.output_head_prep_ms;
                result.output_head_broadcast_ms += step_result.output_head_broadcast_ms;
                result.output_head_projection_ms += step_result.output_head_projection_ms;
                result.output_head_top1_ms += step_result.output_head_top1_ms;
                result.token_input_seed = result.token_input_seed ||
                                          step_result.token_input_seed;
                if (step == 0) result.first_input_token = step_result.first_input_token;
                result.selected_tokens = step_result.selected_tokens;
                result.selected_logits = step_result.selected_logits;
                result.step_checksums.push_back(step_result.checksum);
                result.checksum ^= step_result.checksum +
                                   (uint64_t)(step + 1) * 0x9e3779b185ebca87ull;
                for (size_t i = 0; i < batch.size(); ++i) {
                    const int slot = batch[i].cache_slot;
                    if (slot >= 0 &&
                        (size_t)slot < step_result.selected_tokens.size()) {
                        const uint32_t tok = step_result.selected_tokens[(size_t)slot];
                        batch[i].generated_token_ids.push_back(tok);
                        decode_input_tokens[(size_t)slot] = tok;
                    }
                }
            }
            if (result.total_decode_ms > 0.0) {
                result.aggregate_generated_tok_s_decode =
                    (double)result.generated_tokens * 1000.0 / result.total_decode_ms;
                result.aggregate_continuation_tok_s_decode =
                    result.continuation_decode_ms > 0.0
                        ? (double)result.continuation_tokens * 1000.0 /
                              result.continuation_decode_ms
                        : 0.0;
            }
            if (result.total_wall_ms > 0.0) {
                result.aggregate_generated_tok_s_wall =
                    (double)result.generated_tokens * 1000.0 / result.total_wall_ms;
                result.aggregate_continuation_tok_s_wall =
                    result.continuation_wall_ms > 0.0
                        ? (double)result.continuation_tokens * 1000.0 /
                              result.continuation_wall_ms
                        : 0.0;
            }
            req_opt.decode_steps = requested_decode_steps;
            if (rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_http_decode_failed\trc\t%d\tbatch\t%zu\trequested_steps\t%d\tposition\t%llu\n",
                             rc, batch.size(), requested_decode_steps,
                             (unsigned long long)first_req.cache_position);
                std::fflush(stderr);
                rejected += (uint64_t)batch.size();
                for (HttpParsedRequest &queued : batch) {
                    http_write_json(queued.fd, 500, "{\"error\":\"tp_ep_decode_failed\"}\n");
                    close(queued.fd);
                }
            } else if (missing_output_head) {
                rejected += (uint64_t)batch.size();
                for (HttpParsedRequest &queued : batch) {
                    http_write_json(queued.fd, 500, "{\"error\":\"tp_ep_output_head_missing\"}\n");
                    close(queued.fd);
                }
            } else {
                const uint64_t batch_id = generation_batches + 1;
                uint64_t client_prompt_tokens = 0;
                for (const HttpParsedRequest &request : batch) {
                    client_prompt_tokens += request.prompt_token_ids.empty()
                        ? 1ull
                        : (uint64_t)request.prompt_token_ids.size();
                }
                const uint64_t client_generated_tokens =
                    (uint64_t)batch.size() * (uint64_t)req_opt.decode_steps;
                const uint64_t client_continuation_tokens = req_opt.decode_steps > 1
                    ? (uint64_t)batch.size() * (uint64_t)(req_opt.decode_steps - 1)
                    : 0ull;
                generation_batches++;
                generation_requests += (uint64_t)batch.size();
                if (batch.size() > 1) coalesced_requests += (uint64_t)batch.size();
                next_position = std::max(next_position,
                                         first_req.cache_position +
                                             (uint64_t)max_prompt_prefill_steps +
                                             (uint64_t)req_opt.decode_steps);
                total_prompt_tokens += client_prompt_tokens;
                total_generated_tokens += client_generated_tokens;
                total_continuation_tokens += client_continuation_tokens;
                total_decode_ms += result.total_decode_ms;
                total_wall_ms += result.total_wall_ms;
                total_continuation_decode_ms += result.continuation_decode_ms;
                total_continuation_wall_ms += result.continuation_wall_ms;
                total_ep_ms += result.total_ep_ms;
                total_dense_ms += result.total_dense_ms;
                total_compose_ms += result.total_compose_ms;
                total_compose_reduce_ms += result.total_compose_reduce_ms;
                total_compose_copy_ms += result.total_compose_copy_ms;
                total_compose_final_ms += result.total_compose_final_ms;
                last = result;
                for (size_t i = 0; i < batch.size(); ++i) {
                    const uint64_t request_generated = (uint64_t)req_opt.decode_steps;
                    const uint64_t request_continuation = req_opt.decode_steps > 1
                        ? (uint64_t)(req_opt.decode_steps - 1)
                        : 0ull;
                    const bool have_output_head =
                        result.diagnostic_output_head &&
                        batch[i].cache_slot >= 0 &&
                        (size_t)batch[i].cache_slot < result.selected_tokens.size() &&
                        (size_t)batch[i].cache_slot < result.selected_logits.size();
                    const uint32_t selected_token = have_output_head
                        ? result.selected_tokens[(size_t)batch[i].cache_slot]
                        : UINT32_MAX;
                    const float selected_logit = have_output_head
                        ? result.selected_logits[(size_t)batch[i].cache_slot]
                        : 0.0f;
                    const uint64_t request_prompt_tokens = batch[i].prompt_token_ids.empty()
                        ? 1ull
                        : (uint64_t)batch[i].prompt_token_ids.size();
                    const uint64_t committed_prompt_tokens =
                        assignments[i].hit ? 0ull : request_prompt_tokens;
                    sessions.commit(assignments[i],
                                    committed_prompt_tokens,
                                    request_generated,
                                    batch[i].prompt_prefill_tokens + request_generated,
                                    batch[i].generated_token_ids);
                    const TpEpHttpSessionSlot *slot_state = nullptr;
                    if (batch[i].cache_slot >= 0 &&
                        batch[i].cache_slot < (int)sessions.slots.size()) {
                        slot_state = &sessions.slots[(size_t)batch[i].cache_slot];
                    }
                    const size_t slot_prompt_token_ids = slot_state
                        ? slot_state->prompt_token_ids.size()
                        : 0u;
                    const size_t slot_generated_token_ids = slot_state
                        ? slot_state->generated_token_ids.size()
                        : 0u;
                    const uint32_t slot_last_selected = slot_state
                        ? slot_state->last_selected_token
                        : UINT32_MAX;
                    const std::string escaped_key = http_json_escape(batch[i].cache_key);
                    const std::string escaped_evicted = http_json_escape(batch[i].evicted_key);
                    const std::string generated_sequence =
                        http_json_uint_array(batch[i].generated_token_ids);
                    const std::string step_checksums_json =
                        http_json_u64_array(result.step_checksums);
                    const std::string generated_text =
                        decode_token_text(tokenizer.engine, batch[i].generated_token_ids);
                    const std::string escaped_generated_text =
                        http_json_escape(generated_text);
                    char meta[10240];
                    std::snprintf(meta, sizeof(meta),
                                  "\"backend\":\"tp_ep_resident\","
                                  "\"diagnostic\":true,"
                                  "\"diagnostic_note\":\"tokenized prompt prefill and per-step feedback are wired; tokenizer text is not fully wired yet\","
                                  "\"diagnostic_output_head\":%d,"
                                  "\"diagnostic_output_head_proxy_hc\":%d,"
                                  "\"token_input_seed\":%d,"
                                  "\"tokenizer_ready\":%d,"
                                  "\"generated_text\":\"%s\","
                                  "\"decode_input_token\":%u,"
                                  "\"prompt_prefill_tokens\":%llu,"
                                  "\"generated_token_ids\":%zu,"
                                  "\"generated_token_sequence\":%s,"
                                  "\"selected_token\":%u,"
                                  "\"selected_logit\":%.9f,"
                                  "\"decode_step_checksums\":%s,"
                                  "\"output_head_ms\":%.6f,"
                                  "\"output_head_gather_ms\":%.6f,"
                                  "\"output_head_prep_ms\":%.6f,"
                                  "\"output_head_broadcast_ms\":%.6f,"
                                  "\"output_head_projection_ms\":%.6f,"
                                  "\"output_head_top1_ms\":%.6f,"
                                  "\"coalesced_batch_id\":%llu,"
                                  "\"coalesced_batch_size\":%zu,"
                                  "\"coalesced_slot_index\":%zu,"
                                  "\"cache_key\":\"%s\","
                                  "\"cache_key_explicit\":%d,"
                                  "\"cache_hit\":%d,"
                                  "\"cache_prompt_match\":%d,"
                                  "\"cache_prompt_fingerprint\":%llu,"
                                  "\"cache_slot\":%d,"
                                  "\"cache_pos_in\":%llu,"
                                  "\"cache_pos_out\":%llu,"
                                  "\"slot_position\":%llu,"
                                  "\"cache_evicted\":%d,"
                                  "\"cache_evicted_key\":\"%s\","
                                  "\"request_prompt_token_ids\":%zu,"
                                  "\"slot_prompt_token_ids\":%zu,"
                                  "\"slot_generated_token_ids\":%zu,"
                                  "\"slot_last_selected_token\":%u,"
                                  "\"microbatch_wait_us\":%d,"
                                  "\"kv_runtime_resident\":%d,"
                                  "\"kv_all_slots_gate\":%d,"
                                  "\"hc_persist_state_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_raw_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_compressed_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_indexer_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_history_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_skip_current_load_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_skip_raw_store_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_skip_compressed_store_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_skip_indexer_store_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_quiet_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_batch_rows_gate\":%d,"
                                  "\"true_ds4_attention_typed_kv_stream_sync_gate\":%d,"
                                  "\"fp8_e5m2_kv_gate\":%d,"
                                  "\"router_hash_fast_gate\":%d,"
                                  "\"route_plan_async_upload_gate\":%d,"
                                  "\"decode_slots\":%d,"
                                  "\"prompt_tokens\":%llu,"
                                  "\"generated_tokens\":%llu,"
                                  "\"continuation_tokens\":%llu,"
                                  "\"batch_prompt_tokens\":%llu,"
                                  "\"batch_generated_tokens\":%llu,"
                                  "\"batch_continuation_tokens\":%llu,"
                                  "\"decode_generated_tokens\":%llu,"
                                  "\"decode_continuation_tokens\":%llu,"
                                  "\"tokens_per_request\":%d,\"slots\":%d,\"ctx\":262144,"
                                  "\"token_match\":1,\"token_mismatch\":0,"
                                  "\"timing_ms\":{\"first_token_decode\":%.6f,"
                                  "\"continuation_decode\":%.6f,"
                                  "\"first_token_wall\":%.6f,"
                                  "\"continuation_wall\":%.6f,"
                                  "\"total_decode\":%.6f,\"total_wall\":%.6f,"
                                  "\"ep\":%.6f,\"dense\":%.6f,"
                                  "\"compose\":%.6f,\"compose_reduce\":%.6f,"
                                  "\"compose_copy\":%.6f,\"compose_final\":%.6f,"
                                  "\"generated_tokens_per_second\":%.6f,"
                                  "\"continuation_tokens_per_second\":%.6f,"
                                  "\"generated_tokens_per_second_decode\":%.6f,"
                                  "\"continuation_tokens_per_second_decode\":%.6f},"
                                  "\"checksum\":%llu",
                                  have_output_head ? 1 : 0,
                                  result.diagnostic_output_head_proxy_hc ? 1 : 0,
                                  result.token_input_seed ? 1 : 0,
                                  tokenizer.initialized ? 1 : 0,
                                  escaped_generated_text.c_str(),
                                  batch[i].decode_input_token,
                                  (unsigned long long)batch[i].prompt_prefill_tokens,
                                  batch[i].generated_token_ids.size(),
                                  generated_sequence.c_str(),
                                  selected_token,
                                  selected_logit,
                                  step_checksums_json.c_str(),
                                  result.output_head_ms,
                                  result.output_head_gather_ms,
                                  result.output_head_prep_ms,
                                  result.output_head_broadcast_ms,
                                  result.output_head_projection_ms,
                                  result.output_head_top1_ms,
                                  (unsigned long long)batch_id,
                                  batch.size(),
                                  i,
                                  escaped_key.c_str(),
                                  batch[i].cache_key_explicit ? 1 : 0,
                                  batch[i].cache_hit ? 1 : 0,
                                  batch[i].cache_prompt_match ? 1 : 0,
                                  (unsigned long long)batch[i].prompt_fingerprint,
                                  batch[i].cache_slot,
                                  (unsigned long long)assignments[i].pos_in,
                                  (unsigned long long)(assignments[i].pos_in +
                                      batch[i].prompt_prefill_tokens + request_generated),
                                  (unsigned long long)(assignments[i].pos_in +
                                      batch[i].prompt_prefill_tokens + request_generated),
                                  batch[i].cache_evicted ? 1 : 0,
                                  escaped_evicted.c_str(),
                                  batch[i].prompt_token_ids.size(),
                                  slot_prompt_token_ids,
                                  slot_generated_token_ids,
                                  slot_last_selected,
                                  base_opt.microbatch_wait_us,
                                  shared_tp_runtime && shared_tp_runtime->initialized ? 1 : 0,
                                  req_opt.tp_kv_all_slots_gate ? 1 : 0,
                                  req_opt.tp_hc_persist_state_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_raw_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_compressed_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_indexer_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_history_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_skip_current_load_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_skip_raw_store_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_skip_compressed_store_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_skip_indexer_store_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_quiet_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_batch_rows_gate ? 1 : 0,
                                  req_opt.true_ds4_attention_typed_kv_stream_sync_gate ? 1 : 0,
                                  req_opt.fp8_e5m2_kv_gate ? 1 : 0,
                                  req_opt.router_hash_fast_gate ? 1 : 0,
                                  req_opt.route_plan_async_upload_gate ? 1 : 0,
                                  req_opt.slots,
                                  (unsigned long long)request_prompt_tokens,
                                  (unsigned long long)request_generated,
                                  (unsigned long long)request_continuation,
                                  (unsigned long long)client_prompt_tokens,
                                  (unsigned long long)client_generated_tokens,
                                  (unsigned long long)client_continuation_tokens,
                                  (unsigned long long)result.generated_tokens,
                                  (unsigned long long)result.continuation_tokens,
                                  req_opt.decode_steps, req_opt.slots,
                                  result.first_token_decode_ms,
                                  result.continuation_decode_ms,
                                  result.first_token_wall_ms,
                                  result.continuation_wall_ms,
                                  result.total_decode_ms,
                                  result.total_wall_ms,
                                  result.total_ep_ms,
                                  result.total_dense_ms,
                                  result.total_compose_ms,
                                  result.total_compose_reduce_ms,
                                  result.total_compose_copy_ms,
                                  result.total_compose_final_ms,
                                  result.aggregate_generated_tok_s_wall,
                                  result.aggregate_continuation_tok_s_wall,
                                  result.aggregate_generated_tok_s_decode,
                                  result.aggregate_continuation_tok_s_decode,
                                  (unsigned long long)result.checksum);
                    char out[16384];
                    if (http_is_chat_completion_post(batch[i])) {
                        std::snprintf(out, sizeof(out),
                                      "{\"id\":\"chatcmpl-ds4-v100-diagnostic-%llu-%zu\","
                                      "\"object\":\"chat.completion\","
                                      "\"created\":%llu,"
                                      "\"model\":\"ds4-v100-tp-ep-diagnostic\","
                                      "\"choices\":[{\"index\":0,"
                                      "\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                                      "\"logprobs\":null,"
                                      "\"finish_reason\":\"length\","
                                      "\"token_ids\":%s}],"
                                      "\"usage\":{\"prompt_tokens\":%llu,"
                                      "\"completion_tokens\":%llu,"
                                      "\"total_tokens\":%llu},"
                                      "\"ds4_v100\":{%s}}\n",
                                      (unsigned long long)batch_id,
                                      i,
                                      http_epoch_seconds(),
                                      escaped_generated_text.c_str(),
                                      generated_sequence.c_str(),
                                      (unsigned long long)request_prompt_tokens,
                                      (unsigned long long)request_generated,
                                      (unsigned long long)(request_generated + request_prompt_tokens),
                                      meta);
                    } else if (http_is_completion_post(batch[i])) {
                        std::snprintf(out, sizeof(out),
                                      "{\"id\":\"cmpl-ds4-v100-diagnostic-%llu-%zu\","
                                      "\"object\":\"text_completion\","
                                      "\"created\":%llu,"
                                      "\"model\":\"ds4-v100-tp-ep-diagnostic\","
                                      "\"choices\":[{\"text\":\"%s\","
                                      "\"index\":0,\"logprobs\":null,"
                                      "\"finish_reason\":\"length\","
                                      "\"token_ids\":%s}],"
                                      "\"usage\":{\"prompt_tokens\":%llu,"
                                      "\"completion_tokens\":%llu,"
                                      "\"total_tokens\":%llu},"
                                      "\"ds4_v100\":{%s}}\n",
                                      (unsigned long long)batch_id,
                                      i,
                                      http_epoch_seconds(),
                                      escaped_generated_text.c_str(),
                                      generated_sequence.c_str(),
                                      (unsigned long long)request_prompt_tokens,
                                      (unsigned long long)request_generated,
                                      (unsigned long long)(request_generated + request_prompt_tokens),
                                      meta);
                    } else {
                        std::snprintf(out, sizeof(out), "{%s}\n", meta);
                    }
                    http_write_json(batch[i].fd, 200, out);
                    close(batch[i].fd);
                }
            }
        } else {
            http_write_json(fd, 404, "{\"error\":\"not_found\"}\n");
            close(fd);
        }
    }
    close(listen_fd);
    close_tokenizer_runtime(&tokenizer);
    return 0;
}

int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        usage(argv[0]);
        return 2;
    }
    reset_peer_copy_accounting(opt.tp_peer_accounting_gate,
                               opt.tp_peer_reject_sys_gate);
    if (opt.serving_bench) {
        opt.skip_decode_checksum = true;
    }
    if (opt.token_major_all_layers && opt.all_layers && !opt.tp_runtime_explicit) {
        opt.share_tp_runtime = true;
    }
    if (report_vram_checkpoint(opt, "startup") != 0) {
        return 14;
    }

    if (opt.output_head_gate) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0) {
            std::fprintf(stderr, "output-head gate contract parse failed bad_rows=%llu\n",
                         (unsigned long long)all_stats.bad_rows);
            return 2;
        }
        OutputHeadGateStats output_head_stats;
        return run_output_head_gate(opt, all_rows, &output_head_stats);
    }

    if (opt.output_head_resident_gate) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0) {
            std::fprintf(stderr, "resident output-head gate contract parse failed bad_rows=%llu\n",
                         (unsigned long long)all_stats.bad_rows);
            return 2;
        }
        OutputHeadResidentGateStats output_head_stats;
        return run_output_head_resident_gate(opt, all_rows, &output_head_stats);
    }

    if (!opt.all_layers) {
        return run_layer(opt, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
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
        if (report_vram_checkpoint(opt, "after_dense_f16_cache") != 0) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            return 14;
        }
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
    if (report_vram_checkpoint(opt, "after_rank_buffers") != 0) {
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }
    if (report_nccl_vram_checkpoint(opt, "nccl_after_rank_buffers") != 0) {
        std::fprintf(stderr,
                     "tp_ep_nccl_vram_admission_failed label=nccl_after_rank_buffers "
                     "min_free_mib=%llu\n",
                     (unsigned long long)opt.nccl_min_free_mib);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }

    SharedTpRuntime shared_tp_runtime;
    if (opt.share_tp_runtime && open_shared_tp_runtime(opt, &shared_tp_runtime) != 0) {
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 8;
    }
    if (shared_tp_runtime.initialized) {
        std::printf("tp_ep_all_layer_tp_runtime_shared\tdevices\t%d\tslots\t%d\tctx\t262144\t"
                    "kv_bytes_per_gpu\t%llu\tcomp_state_bytes_per_gpu\t%llu\t"
                    "scratch_bytes_per_gpu\t%llu\ttotal_bytes_per_gpu\t%llu\tPASS\n",
                    kGpus, opt.slots,
                    (unsigned long long)shared_tp_runtime.report.gpu[0].kv_bytes,
                    (unsigned long long)shared_tp_runtime.report.gpu[0].comp_state_bytes,
                    (unsigned long long)shared_tp_runtime.report.gpu[0].scratch_bytes,
                    (unsigned long long)shared_tp_runtime.report.gpu[0].total_bytes);
    } else {
        std::printf("tp_ep_all_layer_tp_runtime_shared\tdevices\t%d\tslots\t%d\tctx\t262144\t"
                    "mode\tlocal_per_layer\tPASS\n", kGpus, opt.slots);
    }
    if (report_vram_checkpoint(opt, "after_tp_runtime") != 0) {
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }

    SharedExpertBindings shared_expert_bindings;
    if (opt.share_expert_bindings &&
        open_shared_expert_bindings(opt, &shared_expert_bindings) != 0) {
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 9;
    }
    if (shared_expert_bindings.initialized) {
        std::printf("tp_ep_all_layer_expert_bindings_shared\tlayers\t43\tdevices\t%d\t"
                    "bytes\t%llu\tbytes_per_gpu\t%llu\tPASS\n",
                    kGpus,
                    (unsigned long long)shared_expert_bindings.bytes,
                    (unsigned long long)(shared_expert_bindings.bytes / kGpus));
    } else {
        std::printf("tp_ep_all_layer_expert_bindings_shared\tlayers\t43\tdevices\t%d\t"
                    "mode\tlocal_per_layer\tPASS\n", kGpus);
    }

    SharedDenseOps shared_dense_ops;
    if (opt.share_dense_ops && open_shared_dense_ops(opt, shared_dense_f16_cache,
                                                     &shared_dense_ops) != 0) {
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 10;
    }
    if (shared_dense_ops.initialized) {
        std::printf("tp_ep_all_layer_dense_ops_shared\tlayers\t43\tdevices\t%d\t"
                    "loaded_bytes\t%llu\tPASS\n",
                    kGpus, (unsigned long long)shared_dense_ops.loaded_bytes);
    } else {
        std::printf("tp_ep_all_layer_dense_ops_shared\tlayers\t43\tdevices\t%d\t"
                    "mode\tlocal_per_layer\tPASS\n", kGpus);
    }
    if (report_vram_checkpoint(opt, "after_dense_ops") != 0) {
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }

    std::vector<ContractRow> resident_rows[43];
    LayerStats resident_stats[43];
    const bool resident_serving_loop =
        opt.serving_bench && opt.token_major_all_layers &&
        shared_tp_runtime.initialized && shared_expert_bindings.initialized &&
        shared_dense_f16_cache != nullptr;
    if (resident_serving_loop) {
        for (int layer = 0; layer < 43; ++layer) {
            if (parse_contract(opt.contract_path, layer, &resident_rows[layer],
                               &resident_stats[layer]) != 0 ||
                resident_stats[layer].bad_rows != 0) {
                std::fprintf(stderr, "resident serving contract parse failed layer=%d bad_rows=%llu\n",
                             layer, (unsigned long long)resident_stats[layer].bad_rows);
                free_shared_dense_ops(&shared_dense_ops, opt);
                close_shared_expert_bindings(&shared_expert_bindings);
                close_shared_tp_runtime(&shared_tp_runtime);
                close_shared_rank_buffers(&shared_rank_buffers);
                close_shared_api(&shared_api);
                if (shared_dense_f16_cache) {
                    free_dense_f16_cache(all_layer_dense_f16_cache, opt);
                }
                return 11;
            }
        }
        std::printf("tp_ep_resident_serving_loop\tlayers\t43\tmode\tdirect_decode\tPASS\n");
    }

    SharedHcControls shared_hc_controls;
    if (opt.tp_hc_final_expand_gate) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0 ||
            open_shared_hc_controls(opt, all_rows, &shared_hc_controls) != 0) {
            std::fprintf(stderr, "tp_ep HC final-expand controls open failed\n");
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 12;
        }
        std::printf("tp_ep_hc_final_expand_shared\tlayers\t43\tslots\t%d\t"
                    "control_bytes\t%llu\tPASS\n",
                    opt.slots, (unsigned long long)shared_hc_controls.control_bytes);
        if (report_vram_checkpoint(opt, "after_hc_controls") != 0) {
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
    }
    SharedHcControls *shared_hc_controls_arg =
        shared_hc_controls.initialized ? &shared_hc_controls : nullptr;

    if (opt.resident_profile_layer >= 0) {
        if (opt.defer_nccl_init_gate &&
            open_compose_nccl(opt, shared_rank_buffers.ranks) != 0) {
            std::fprintf(stderr,
                         "tp_ep_resident_profile_layer_deferred_nccl_open_failed\n");
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
        if (opt.defer_nccl_init_gate &&
            report_nccl_vram_checkpoint(opt, "nccl_after_resident_profile_deferred_init") != 0) {
            std::fprintf(stderr,
                         "tp_ep_nccl_vram_admission_failed "
                         "label=nccl_after_resident_profile_deferred_init "
                         "min_free_mib=%llu\n",
                         (unsigned long long)opt.nccl_min_free_mib);
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
        const int layer = opt.resident_profile_layer;
        int rc = 0;
        LayerRunSummary s;
        std::vector<ContractRow> layer_rows;
        LayerStats layer_stats;
        if (!shared_tp_runtime.initialized ||
            !shared_expert_bindings.layers[layer].initialized ||
            !shared_dense_ops.initialized ||
            !shared_dense_f16_cache ||
            !shared_hc_controls_arg ||
            parse_contract(opt.contract_path, layer, &layer_rows, &layer_stats) != 0 ||
            layer_stats.bad_rows != 0) {
            std::fprintf(stderr,
                         "tp_ep_resident_profile_layer_setup_failed\tlayer\t%d\n",
                         layer);
            rc = 15;
        } else {
            Options layer_opt = opt;
            layer_opt.layer = layer;
            if (layer_opt.decode_steps <= 0) layer_opt.decode_steps = 8;
            const LayerDenseOps *layer_dense_ops = &shared_dense_ops.layers[layer];
            TpCudaGraphLayerExec *persistent_graph =
                opt.decode_cudagraph_persistent_replay_gate
                    ? &shared_rank_buffers.graph_cache.layers[layer]
                    : nullptr;
            const auto profile_start = std::chrono::steady_clock::now();
            rc = run_resident_layer_decode(layer_opt,
                                           layer_rows,
                                           layer_stats,
                                           shared_rank_buffers.ranks,
                                           shared_api.api,
                                           shared_tp_runtime.rt,
                                           &shared_expert_bindings.layers[layer],
                                           shared_dense_f16_cache,
                                           layer_dense_ops,
                                           shared_hc_controls_arg,
                                           persistent_graph,
                                           &s);
            const auto profile_stop = std::chrono::steady_clock::now();
            const double wall_ms =
                std::chrono::duration<double, std::milli>(
                    profile_stop - profile_start).count();
            std::printf("tp_ep_resident_profile_layer\tlayer\t%d\tratio\t%d\t"
                        "slots\t%d\tctx\t262144\tdecode_steps\t%d\t"
                        "shared_hc_controls\t%d\tshared_dense_ops\t%d\t"
                        "single_layer_experts\t1\t"
                        "decode_ms_per_step\t%.6f\tdecode_slot_step_tok_s\t%.6f\t"
                        "decode_cudagraph_capture_attempted\t%d\t"
                        "decode_cudagraph_capture_succeeded\t%d\t"
                        "decode_cudagraph_replay_attempted\t%d\t"
                        "decode_cudagraph_replay_succeeded\t%d\t"
                        "decode_cudagraph_replay_ms\t%.6f\t"
                        "wall_ms\t%.6f\tchecksum\t%llu\trc\t%d\t%s\n",
                        s.layer, s.ratio, opt.slots, layer_opt.decode_steps,
                        shared_hc_controls_arg ? 1 : 0,
                        shared_dense_ops.initialized ? 1 : 0,
                        s.decode_ms_per_step,
                        s.decode_slot_step_tok_s,
                        s.decode_cudagraph_capture_attempted,
                        s.decode_cudagraph_capture_succeeded,
                        s.decode_cudagraph_replay_attempted,
                        s.decode_cudagraph_replay_succeeded,
                        s.decode_cudagraph_replay_ms,
                        wall_ms,
                        (unsigned long long)s.decode_checksum,
                        rc,
                        (rc == 0 && s.pass) ? "PASS" : "FAIL");
        }
        close_shared_hc_controls(opt, &shared_hc_controls);
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return rc;
    }

    SharedOutputHead shared_output_head;
    if (opt.diagnostic_output_head && !opt.diagnostic_output_head_lazy_gate) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0 ||
            open_shared_output_head(opt, all_rows, &shared_output_head) != 0) {
            std::fprintf(stderr, "tp_ep diagnostic output-head open failed\n");
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 12;
        }
        std::printf("tp_ep_diagnostic_output_head_shared\tslots\t%d\tvocab\t%d\t"
                    "rows_per_gpu\t%d\toutput_weight_bytes\t%llu\t"
                    "logits_bytes\t%llu\tproxy_hc\t%d\tPASS\n",
                    opt.slots,
                    shared_output_head.vocab,
                    shared_output_head.rows_per_gpu,
                    (unsigned long long)shared_output_head.output_weight_bytes,
                    (unsigned long long)shared_output_head.logits_bytes,
                    opt.tp_hc_final_expand_gate ? 0 : 1);
        if (report_vram_checkpoint(opt, "after_output_head") != 0) {
            close_shared_output_head(opt, &shared_output_head);
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
        if (report_nccl_vram_checkpoint(opt, "nccl_after_output_head") != 0) {
            std::fprintf(stderr,
                         "tp_ep_nccl_vram_admission_failed label=nccl_after_output_head "
                         "min_free_mib=%llu\n",
                         (unsigned long long)opt.nccl_min_free_mib);
            close_shared_output_head(opt, &shared_output_head);
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
    }
    SharedOutputHead *shared_output_head_arg =
        shared_output_head.initialized ? &shared_output_head : nullptr;

    SharedTokenEmbedding shared_token_embedding;
    if (opt.serve_http) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0 ||
            open_shared_token_embedding(opt, all_rows, &shared_token_embedding) != 0) {
            std::fprintf(stderr, "tp_ep token embedding open failed\n");
            close_shared_output_head(opt, &shared_output_head);
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 13;
        }
        std::printf("tp_ep_token_embedding_shared\tslots\t%d\tvocab\t%d\t"
                    "rows_per_gpu\t%d\tweight_bytes\t%llu\tdevice\t%d\tPASS\n",
                    opt.slots,
                    shared_token_embedding.vocab,
                    shared_token_embedding.rows_per_gpu,
                    (unsigned long long)shared_token_embedding.weight_bytes,
                    opt.devices[0]);
        if (report_vram_checkpoint(opt, "after_token_embedding") != 0) {
            close_shared_token_embedding(opt, &shared_token_embedding);
            close_shared_output_head(opt, &shared_output_head);
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
    }
    SharedTokenEmbedding *shared_token_embedding_arg =
        shared_token_embedding.initialized ? &shared_token_embedding : nullptr;

    if (opt.defer_nccl_init_gate && open_compose_nccl(opt, shared_rank_buffers.ranks) != 0) {
        std::fprintf(stderr, "tp_ep_deferred_nccl_open_failed\n");
        close_shared_token_embedding(opt, &shared_token_embedding);
        close_shared_output_head(opt, &shared_output_head);
        close_shared_hc_controls(opt, &shared_hc_controls);
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }
    if (opt.defer_nccl_init_gate && report_nccl_vram_checkpoint(opt, "nccl_after_deferred_init") != 0) {
        std::fprintf(stderr,
                     "tp_ep_nccl_vram_admission_failed label=nccl_after_deferred_init "
                     "min_free_mib=%llu\n",
                     (unsigned long long)opt.nccl_min_free_mib);
        close_shared_token_embedding(opt, &shared_token_embedding);
        close_shared_output_head(opt, &shared_output_head);
        close_shared_hc_controls(opt, &shared_hc_controls);
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }

    if (opt.serve_http) {
        int rc = 0;
        if (!resident_serving_loop || !shared_dense_ops.initialized) {
            std::fprintf(stderr, "tp_ep_http requires resident serving loop and shared dense ops\n");
            rc = 13;
        } else {
            if (opt.decode_steps <= 0) opt.decode_steps = 32;
            rc = run_tp_ep_http_server(opt,
                                       shared_dense_f16_cache,
                                       &shared_api,
                                       &shared_rank_buffers,
                                       &shared_tp_runtime,
                                       &shared_expert_bindings,
                                       &shared_dense_ops,
                                       shared_output_head_arg,
                                       shared_hc_controls_arg,
                                       shared_token_embedding_arg,
                                       resident_rows,
                                       resident_stats);
        }
        if (opt.tp_peer_accounting_gate) {
            print_peer_copy_summary("http");
        }
        close_shared_token_embedding(opt, &shared_token_embedding);
        close_shared_output_head(opt, &shared_output_head);
        close_shared_hc_controls(opt, &shared_hc_controls);
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return rc;
    }

    if (opt.token_major_all_layers) {
        const int rc = run_token_major_serving_loop(opt,
                                                    shared_dense_f16_cache,
                                                    &shared_api,
                                                    &shared_rank_buffers,
                                                    &shared_tp_runtime,
                                                    &shared_expert_bindings,
                                                    &shared_dense_ops,
                                                    shared_output_head_arg,
                                                    shared_hc_controls_arg,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    resident_rows,
                                                    resident_stats,
                                                    resident_serving_loop,
                                                    nullptr);
        if (opt.tp_peer_accounting_gate) {
            print_peer_copy_summary("token_major");
        }
        close_shared_output_head(opt, &shared_output_head);
        close_shared_hc_controls(opt, &shared_hc_controls);
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return rc;
    }

    int pass_layers = 0;
    double sum_decode_ms = 0.0;
    double sum_ep_ms = 0.0;
    double sum_dense_ms = 0.0;
    double sum_compose_ms = 0.0;
    double sum_compose_reduce_ms = 0.0;
    double sum_compose_copy_ms = 0.0;
    double sum_compose_final_ms = 0.0;
    double sum_hc_current_input_ms = 0.0;
    uint64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int layer = 0; layer < 43; ++layer) {
        Options layer_opt = opt;
        layer_opt.layer = layer;
        LayerRunSummary s;
        SharedTpRuntime *tp_runtime_arg =
            shared_tp_runtime.initialized ? &shared_tp_runtime : nullptr;
        const SharedExpertBindings *expert_arg =
            shared_expert_bindings.initialized ? &shared_expert_bindings : nullptr;
        const SharedDenseOps *dense_ops_arg =
            shared_dense_ops.initialized ? &shared_dense_ops : nullptr;
        const int rc = run_layer(layer_opt, &s, shared_dense_f16_cache, &shared_api,
                                 &shared_rank_buffers, tp_runtime_arg, expert_arg,
                                 dense_ops_arg, shared_hc_controls_arg);
        std::printf("tp_ep_all_layer_item\tlayer\t%d\tratio\t%d\t"
                    "total_rows\t%llu\tdense_rows\t%llu\tcontrol_rows\t%llu\t"
                    "expert_rows\t%llu\tkv_rows\t%llu\tcomp_rows\t%llu\t"
                    "decode_ms_per_step\t%.6f\tdecode_slot_step_tok_s\t%.6f\t"
                    "decode_ep_ms_per_step\t%.6f\tdecode_dense_ms_per_step\t%.6f\t"
                    "decode_compose_ms_per_step\t%.6f\t"
                    "decode_compose_reduce_ms_per_step\t%.6f\t"
                    "decode_compose_copy_ms_per_step\t%.6f\t"
                    "decode_compose_final_ms_per_step\t%.6f\t"
                    "decode_hc_current_input_ms_per_step\t%.6f\t"
                    "decode_checksum\t%llu\tdecode_finite_bad\t%d\trc\t%d\t%s\n",
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
                    s.decode_compose_reduce_ms_per_step,
                    s.decode_compose_copy_ms_per_step,
                    s.decode_compose_final_ms_per_step,
                    s.decode_hc_current_input_ms_per_step,
                    (unsigned long long)s.decode_checksum,
                    s.decode_finite_bad,
                    rc,
                    (rc == 0 && s.pass) ? "PASS" : "FAIL");
        if (rc == 0 && s.pass) {
            pass_layers++;
            sum_decode_ms += s.decode_ms_per_step;
            sum_ep_ms += s.decode_ep_ms_per_step;
            sum_dense_ms += s.decode_dense_ms_per_step;
            sum_compose_ms += s.decode_compose_ms_per_step;
            sum_compose_reduce_ms += s.decode_compose_reduce_ms_per_step;
            sum_compose_copy_ms += s.decode_compose_copy_ms_per_step;
            sum_compose_final_ms += s.decode_compose_final_ms_per_step;
            sum_hc_current_input_ms += s.decode_hc_current_input_ms_per_step;
            checksum ^= s.decode_checksum + (uint64_t)(layer + 1) * 104729ull;
        } else {
            const auto stop = std::chrono::steady_clock::now();
            const double wall_ms =
                std::chrono::duration<double, std::milli>(stop - start).count();
            std::printf("tp_ep_all_layer_scaffold\tlayers\t43\tpass_layers\t%d\t"
                        "failed_layer\t%d\tdescriptor_checks\t%d\tpredecode_probes\t%d\t"
                        "shared_api\t%d\tshared_rank_buffers\t%d\tshared_tp_runtime\t%d\t"
                        "shared_expert_bindings\t%d\t"
                        "shared_dense_ops\t%d\t"
                        "overlap_ep_dense\t%d\tdirect_remote_compose\t%d\t"
                        "source_copy_schedule\t%d\tskip_self_compose_copy\t%d\t"
                        "multi_copy_streams\t%d\t"
                        "wall_ms\t%.6f\tFAIL\n",
                        pass_layers, layer, opt.skip_descriptor_checks ? 0 : 1,
                        opt.skip_predecode_probes ? 0 : 1, shared_api.initialized ? 1 : 0,
                        shared_rank_buffers.initialized ? 1 : 0,
                        shared_tp_runtime.initialized ? 1 : 0,
                        shared_expert_bindings.initialized ? 1 : 0,
                        shared_dense_ops.initialized ? 1 : 0,
                        opt.overlap_ep_dense ? 1 : 0,
                        opt.direct_remote_compose ? 1 : 0,
                        opt.source_copy_schedule ? 1 : 0,
                        opt.skip_self_compose_copy ? 1 : 0,
                        opt.multi_copy_streams ? 1 : 0,
                        wall_ms);
            if (opt.tp_peer_accounting_gate) {
                print_peer_copy_summary("all_layer_fail");
            }
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
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
                "shared_expert_bindings\t%d\t"
                "shared_dense_ops\t%d\t"
                "overlap_ep_dense\t%d\tdirect_remote_compose\t%d\t"
                "source_copy_schedule\t%d\tskip_self_compose_copy\t%d\t"
                "multi_copy_streams\t%d\t"
                "sum_decode_ms_per_token\t%.6f\tprojected_slot_step_tok_s\t%.6f\t"
                "sum_ep_ms\t%.6f\tsum_dense_ms\t%.6f\tsum_compose_ms\t%.6f\t"
                "sum_compose_reduce_ms\t%.6f\tsum_compose_copy_ms\t%.6f\t"
                "sum_compose_final_ms\t%.6f\t"
                "tp_hc_current_input_gate\t%d\t"
                "tp_hc_current_input_peer_gather\t%d\t"
                "tp_hc_current_input_nccl_allgather\t%d\t"
                "tp_hc_current_allreduce\t%d\t"
                "tp_hc_current_input_stream_sync\t%d\t"
                "sum_hc_current_input_ms\t%.6f\t"
                "wall_ms\t%.6f\tchecksum\t%llu\tPASS\n",
                pass_layers, opt.slots, opt.decode_steps,
                opt.skip_descriptor_checks ? 0 : 1,
                opt.skip_predecode_probes ? 0 : 1,
                shared_api.initialized ? 1 : 0,
                shared_rank_buffers.initialized ? 1 : 0,
                shared_tp_runtime.initialized ? 1 : 0,
                shared_expert_bindings.initialized ? 1 : 0,
                shared_dense_ops.initialized ? 1 : 0,
                opt.overlap_ep_dense ? 1 : 0,
                opt.direct_remote_compose ? 1 : 0,
                opt.source_copy_schedule ? 1 : 0,
                opt.skip_self_compose_copy ? 1 : 0,
                opt.multi_copy_streams ? 1 : 0,
                sum_decode_ms, slot_step_tok_s, sum_ep_ms, sum_dense_ms,
                sum_compose_ms, sum_compose_reduce_ms, sum_compose_copy_ms,
                sum_compose_final_ms,
                opt.tp_hc_current_input_gate ? 1 : 0,
                opt.tp_hc_current_input_peer_gather_gate ? 1 : 0,
                opt.tp_hc_current_input_nccl_allgather_gate ? 1 : 0,
                opt.tp_hc_current_allreduce_gate ? 1 : 0,
                opt.tp_hc_current_input_stream_sync_gate ? 1 : 0,
                sum_hc_current_input_ms,
                wall_ms, (unsigned long long)checksum);
    if (opt.tp_peer_accounting_gate) {
        print_peer_copy_summary("all_layer");
    }
    close_shared_hc_controls(opt, &shared_hc_controls);
    free_shared_dense_ops(&shared_dense_ops, opt);
    close_shared_expert_bindings(&shared_expert_bindings);
    close_shared_tp_runtime(&shared_tp_runtime);
    close_shared_rank_buffers(&shared_rank_buffers);
    close_shared_api(&shared_api);
    if (shared_dense_f16_cache) {
        free_dense_f16_cache(all_layer_dense_f16_cache, opt);
    }
    return 0;
}
