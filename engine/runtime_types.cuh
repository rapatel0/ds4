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
