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

#include "appliance/options.cu"

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

#include "engine/router_step.cu"
#include "engine/hc_final.cu"
#include "engine/hc_current.cu"

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

#include "engine/ep_dense.cu"
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

#include "engine/ep_executor.cu"
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

#include "engine/router_plan.cu"
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

#include "engine/ep_compose.cu"
#include "engine/attention_projection.cu"
#include "engine/compressed_kv_step.cu"
#include "engine/attention_read.cu"
#include "engine/attention_output.cu"
#include "engine/post_attention_ffn.cu"
#include "engine/decode_loop.cu"

} // namespace

#include "engine/layer_decode.cu"
#include "engine/layer_runner.cu"
#include "engine/token_major_loop.cu"
#include "appliance/http_server.cu"
#include "appliance/entrypoint.cu"
