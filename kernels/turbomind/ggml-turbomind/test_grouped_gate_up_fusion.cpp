// Benchmark the DS4 expert gate/up fusion shape on the TurboMind MXFP4 path.
//
// The current appliance calls grouped GEMM twice for routed gate and up. This
// test compares that against one grouped GEMM with N doubled and packed rows
// laid out as [gate rows][up rows] per expert.

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
#include <vector>

#include "ggml-turbomind-api.h"

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
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
typedef int  (*pfn_ds4_mxfp4_gated_silu_96)(const void *, const int *, int, int,
                                            const void * const *, const void * const *,
                                            int, void *, void *);

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

static void make_mxfp4_fixture(std::vector<block_mxfp4> & blocks, int N, int K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> e_dist(123, 130);
    std::uniform_int_distribution<int> q_dist(0, 255);
    blocks.resize((size_t) N * (K / 32));
    for (block_mxfp4 & b : blocks) {
        b.e = (uint8_t) e_dist(rng);
        for (uint8_t & q : b.qs) {
            q = (uint8_t) q_dist(rng);
        }
    }
}

static void make_fused_fixture(std::vector<block_mxfp4> & fused,
                               const std::vector<block_mxfp4> & gate,
                               const std::vector<block_mxfp4> & up,
                               int N,
                               int K) {
    const int blocks_per_row = K / 32;
    fused.resize((size_t) 2 * N * blocks_per_row);
    for (int row = 0; row < N; ++row) {
        const size_t src = (size_t) row * blocks_per_row;
        const size_t gate_dst = (size_t) row * blocks_per_row;
        const size_t up_dst = (size_t) (row + N) * blocks_per_row;
        std::copy(gate.begin() + src, gate.begin() + src + blocks_per_row, fused.begin() + gate_dst);
        std::copy(up.begin()   + src, up.begin()   + src + blocks_per_row, fused.begin() + up_dst);
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

static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static int env_int(const char *name, int fallback, int lo, int hi) {
    const char *v = std::getenv(name);
    if (!v || !v[0]) return fallback;
    char *end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (!end || *end != '\0' || parsed < lo || parsed > hi) {
        fprintf(stderr,
                "[gate_up_fusion] ignoring invalid %s=%s, expected integer [%d,%d]\n",
                name, v, lo, hi);
        return fallback;
    }
    return (int)parsed;
}

static bool env_flag(const char *name, bool fallback) {
    const char *v = std::getenv(name);
    if (!v || !v[0]) return fallback;
    if (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "on") == 0) {
        return true;
    }
    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0) {
        return false;
    }
    fprintf(stderr,
            "[gate_up_fusion] ignoring invalid %s=%s, expected 0/1/true/false/on/off\n",
            name, v);
    return fallback;
}

static std::vector<Case> parse_cases_from_env() {
    const char *v = std::getenv("DS4_TURBOMIND_GATE_UP_CASES");
    if (!v || !v[0]) {
        return {{1}, {4}, {8}, {16}, {32}, {64}, {128}};
    }

    std::vector<Case> out;
    const char *p = v;
    while (*p) {
        char *end = nullptr;
        long parsed = std::strtol(p, &end, 10);
        if (end == p || parsed < 1 || parsed > 128) {
            fprintf(stderr,
                    "[gate_up_fusion] invalid DS4_TURBOMIND_GATE_UP_CASES=%s; "
                    "expected comma-separated integers in [1,128]\n",
                    v);
            std::exit(2);
        }
        out.push_back(Case{(int)parsed});
        p = end;
        if (*p == ',') {
            ++p;
        } else if (*p != '\0') {
            fprintf(stderr,
                    "[gate_up_fusion] invalid DS4_TURBOMIND_GATE_UP_CASES=%s; "
                    "unexpected character '%c'\n",
                    v, *p);
            std::exit(2);
        }
    }
    if (out.empty()) {
        fprintf(stderr, "[gate_up_fusion] no benchmark cases selected\n");
        std::exit(2);
    }
    return out;
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

static int pack_fixture_set(pfn_packed_bytes pb,
                            pfn_pack_weight pw,
                            int ggml_type,
                            int N,
                            int K,
                            int group_size,
                            int num_experts,
                            const std::vector<int> & active,
                            const std::vector<std::vector<block_mxfp4>> & fixtures,
                            PackedExperts & out) {
    size_t wb = 0;
    size_t sb = 0;
    int rc = pb(ggml_type, N, K, group_size, &wb, &sb);
    if (rc != 0) {
        fprintf(stderr, "[gate_up_fusion] packed_bytes N=%d K=%d rc=%d\n", N, K, rc);
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
            fprintf(stderr, "[gate_up_fusion] pack expert=%d N=%d K=%d rc=%d\n",
                    active[i], N, K, rc);
            return 2;
        }
        if (i == 0) {
            out.k_pack = this_pack;
        } else if (this_pack != out.k_pack) {
            fprintf(stderr, "[gate_up_fusion] inconsistent k_pack 0x%x vs 0x%x\n",
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

static int run_case(void * lib, const Case & c) {
    auto in   = (pfn_init)                         dlsym(lib, "ggml_turbomind_init");
    auto sh   = (pfn_shutdown)                     dlsym(lib, "ggml_turbomind_shutdown");
    auto pb   = (pfn_packed_bytes)                 dlsym(lib, "ggml_turbomind_packed_bytes");
    auto pw   = (pfn_pack_weight)                  dlsym(lib, "ggml_turbomind_pack_weight_expert");
    auto mmgt = (pfn_mul_mat_grouped_total_tokens) dlsym(lib, "ggml_turbomind_mul_mat_grouped_total_tokens");
    auto mmgs = (pfn_mul_mat_grouped_gated_silu_total_tokens)
        dlsym(lib, "ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens");
    auto probe = (pfn_ds4_mxfp4_gated_silu_96)
        dlsym(lib, "ggml_turbomind_ds4_mxfp4_gated_silu_96");
    if (!in || !sh || !pb || !pw || !mmgt || !mmgs) {
        fprintf(stderr, "[gate_up_fusion] dlsym failed\n");
        return 1;
    }
    if (in(0) != 0) {
        fprintf(stderr, "[gate_up_fusion] init failed\n");
        return 2;
    }

    constexpr int ggml_type = GGML_TM_DTYPE_MXFP4;
    constexpr int group_size = 32;
    const bool compact_groups = env_flag("DS4_TURBOMIND_GATE_UP_COMPACT_GROUPS", false);
    const int num_experts = compact_groups ? 6 : 256;
    constexpr int K = 4096;
    constexpr int N = 2048;
    constexpr int fused_N = 2 * N;
    const std::vector<int> active = compact_groups
        ? std::vector<int>{0, 1, 2, 3, 4, 5}
        : std::vector<int>{0, 17, 42, 87, 173, 255};
    const int total_tokens = (int) active.size() * c.tokens_per_active;
    const bool run_probe = probe && compact_groups && total_tokens == 96;

    std::vector<std::vector<block_mxfp4>> gate(active.size());
    std::vector<std::vector<block_mxfp4>> up(active.size());
    std::vector<std::vector<block_mxfp4>> fused(active.size());
    std::vector<std::vector<block_mxfp4>> fused_interleaved(active.size());
    for (size_t i = 0; i < active.size(); ++i) {
        make_mxfp4_fixture(gate[i], N, K, 0x47000000u + (uint32_t)i * 101u);
        make_mxfp4_fixture(up[i],   N, K, 0x55000000u + (uint32_t)i * 131u);
        make_fused_fixture(fused[i], gate[i], up[i], N, K);
        make_fused_interleaved_fixture(fused_interleaved[i], gate[i], up[i], N, K);
    }

    PackedExperts gate_packed;
    PackedExperts up_packed;
    PackedExperts fused_packed;
    PackedExperts gated_packed;
    if (pack_fixture_set(pb, pw, ggml_type, N, K, group_size, num_experts, active, gate, gate_packed) != 0 ||
        pack_fixture_set(pb, pw, ggml_type, N, K, group_size, num_experts, active, up, up_packed) != 0 ||
        pack_fixture_set(pb, pw, ggml_type, fused_N, K, group_size, num_experts, active, fused, fused_packed) != 0 ||
        pack_fixture_set(pb, pw, ggml_type, fused_N, K, group_size, num_experts, active, fused_interleaved, gated_packed) != 0) {
        free_packed(gate_packed);
        free_packed(up_packed);
        free_packed(fused_packed);
        free_packed(gated_packed);
        sh();
        return 3;
    }

    std::vector<int> h_offsets(num_experts + 1, 0);
    int running = 0;
    size_t active_pos = 0;
    for (int e = 0; e < num_experts; ++e) {
        h_offsets[e] = running;
        if (active_pos < active.size() && e == active[active_pos]) {
            running += c.tokens_per_active;
            active_pos++;
        }
    }
    h_offsets[num_experts] = running;
    if (running != total_tokens) {
        fprintf(stderr, "[gate_up_fusion] bad offsets total %d expected %d\n", running, total_tokens);
        free_packed(gate_packed);
        free_packed(up_packed);
        free_packed(fused_packed);
        free_packed(gated_packed);
        sh();
        return 4;
    }

    int * d_offsets = nullptr;
    CHECK_CUDA(cudaMalloc(&d_offsets, h_offsets.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    std::mt19937 rng(0xA5010000u + (uint32_t)c.tokens_per_active);
    std::uniform_real_distribution<float> ad(-0.1f, 0.1f);
    std::vector<__half> h_A((size_t) total_tokens * K);
    for (__half & v : h_A) {
        v = __float2half(ad(rng));
    }

    __half * d_A = nullptr;
    __half * d_gate = nullptr;
    __half * d_up = nullptr;
    __half * d_fused = nullptr;
    __half * d_gated = nullptr;
    __half * d_probe = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_gate, (size_t) total_tokens * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_up, (size_t) total_tokens * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_fused, (size_t) total_tokens * fused_N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_gated, (size_t) total_tokens * N * sizeof(__half)));
    if (run_probe) {
        CHECK_CUDA(cudaMalloc(&d_probe, (size_t) total_tokens * N * sizeof(__half)));
    }
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_gate, 0, (size_t) total_tokens * N * sizeof(__half)));
    CHECK_CUDA(cudaMemset(d_up, 0, (size_t) total_tokens * N * sizeof(__half)));
    CHECK_CUDA(cudaMemset(d_fused, 0, (size_t) total_tokens * fused_N * sizeof(__half)));
    CHECK_CUDA(cudaMemset(d_gated, 0, (size_t) total_tokens * N * sizeof(__half)));
    if (run_probe) {
        CHECK_CUDA(cudaMemset(d_probe, 0, (size_t) total_tokens * N * sizeof(__half)));
    }

    const int warmup_iters = env_int("DS4_TURBOMIND_GATE_UP_WARMUP_ITERS", 3, 0, 1000);
    const int bench_iters = env_int("DS4_TURBOMIND_GATE_UP_BENCH_ITERS", 30, 1, 10000);
    fprintf(stderr,
            "[gate_up_fusion tpa=%d] shape group_mode=%s groups=%d active=%zu total_routes=%d max_routes_per_expert=%d warmup_iters=%d bench_iters=%d\n",
            c.tokens_per_active, compact_groups ? "compact" : "sparse256",
            num_experts, active.size(), total_tokens, c.tokens_per_active,
            warmup_iters, bench_iters);
    int rc = 0;

    for (int iter = 0; iter < warmup_iters; ++iter) {
        rc = mmgt(d_A, nullptr, d_offsets, num_experts, total_tokens,
                  (const void * const *) gate_packed.d_w_table,
                  (const void * const *) gate_packed.d_s_table,
                  ggml_type, N, K, group_size, gate_packed.k_pack, d_gate, nullptr);
        if (rc != 0) {
            fprintf(stderr, "[gate_up_fusion] gate warmup rc=%d\n", rc);
            return 5;
        }
        rc = mmgt(d_A, nullptr, d_offsets, num_experts, total_tokens,
                  (const void * const *) up_packed.d_w_table,
                  (const void * const *) up_packed.d_s_table,
                  ggml_type, N, K, group_size, up_packed.k_pack, d_up, nullptr);
        if (rc != 0) {
            fprintf(stderr, "[gate_up_fusion] up warmup rc=%d\n", rc);
            return 6;
        }
        rc = mmgt(d_A, nullptr, d_offsets, num_experts, total_tokens,
                  (const void * const *) fused_packed.d_w_table,
                  (const void * const *) fused_packed.d_s_table,
                  ggml_type, fused_N, K, group_size, fused_packed.k_pack, d_fused, nullptr);
        if (rc != 0) {
            fprintf(stderr, "[gate_up_fusion] fused warmup rc=%d\n", rc);
            return 7;
        }
        rc = mmgs(d_A, nullptr, d_offsets, num_experts, total_tokens,
                  (const void * const *) gated_packed.d_w_table,
                  (const void * const *) gated_packed.d_s_table,
                  ggml_type, fused_N, K, group_size, gated_packed.k_pack, d_gated, nullptr);
        if (rc != 0) {
            fprintf(stderr, "[gate_up_fusion] gated warmup rc=%d\n", rc);
            return 7;
        }
        if (run_probe) {
            rc = probe(d_A, d_offsets, num_experts, total_tokens,
                       (const void * const *) gated_packed.d_w_table,
                       (const void * const *) gated_packed.d_s_table,
                       gated_packed.k_pack, d_probe, nullptr);
            if (rc != 0) {
                fprintf(stderr, "[gate_up_fusion] ds4 probe warmup rc=%d\n", rc);
                return 7;
            }
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t sep_start = nullptr;
    cudaEvent_t sep_stop = nullptr;
    cudaEvent_t fused_start = nullptr;
    cudaEvent_t fused_stop = nullptr;
    cudaEvent_t gated_start = nullptr;
    cudaEvent_t gated_stop = nullptr;
    cudaEvent_t probe_start = nullptr;
    cudaEvent_t probe_stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&sep_start));
    CHECK_CUDA(cudaEventCreate(&sep_stop));
    CHECK_CUDA(cudaEventCreate(&fused_start));
    CHECK_CUDA(cudaEventCreate(&fused_stop));
    CHECK_CUDA(cudaEventCreate(&gated_start));
    CHECK_CUDA(cudaEventCreate(&gated_stop));
    if (run_probe) {
        CHECK_CUDA(cudaEventCreate(&probe_start));
        CHECK_CUDA(cudaEventCreate(&probe_stop));
    }

    CHECK_CUDA(cudaEventRecord(sep_start, nullptr));
    for (int iter = 0; iter < bench_iters; ++iter) {
        rc = mmgt(d_A, nullptr, d_offsets, num_experts, total_tokens,
                  (const void * const *) gate_packed.d_w_table,
                  (const void * const *) gate_packed.d_s_table,
                  ggml_type, N, K, group_size, gate_packed.k_pack, d_gate, nullptr);
        if (rc != 0) {
            fprintf(stderr, "[gate_up_fusion] gate rc=%d\n", rc);
            return 8;
        }
        rc = mmgt(d_A, nullptr, d_offsets, num_experts, total_tokens,
                  (const void * const *) up_packed.d_w_table,
                  (const void * const *) up_packed.d_s_table,
                  ggml_type, N, K, group_size, up_packed.k_pack, d_up, nullptr);
        if (rc != 0) {
            fprintf(stderr, "[gate_up_fusion] up rc=%d\n", rc);
            return 9;
        }
    }
    CHECK_CUDA(cudaEventRecord(sep_stop, nullptr));
    CHECK_CUDA(cudaEventSynchronize(sep_stop));

    CHECK_CUDA(cudaEventRecord(fused_start, nullptr));
    for (int iter = 0; iter < bench_iters; ++iter) {
        rc = mmgt(d_A, nullptr, d_offsets, num_experts, total_tokens,
                  (const void * const *) fused_packed.d_w_table,
                  (const void * const *) fused_packed.d_s_table,
                  ggml_type, fused_N, K, group_size, fused_packed.k_pack, d_fused, nullptr);
        if (rc != 0) {
            fprintf(stderr, "[gate_up_fusion] fused rc=%d\n", rc);
            return 10;
        }
    }
    CHECK_CUDA(cudaEventRecord(fused_stop, nullptr));
    CHECK_CUDA(cudaEventSynchronize(fused_stop));

    CHECK_CUDA(cudaEventRecord(gated_start, nullptr));
    for (int iter = 0; iter < bench_iters; ++iter) {
        rc = mmgs(d_A, nullptr, d_offsets, num_experts, total_tokens,
                  (const void * const *) gated_packed.d_w_table,
                  (const void * const *) gated_packed.d_s_table,
                  ggml_type, fused_N, K, group_size, gated_packed.k_pack, d_gated, nullptr);
        if (rc != 0) {
            fprintf(stderr, "[gate_up_fusion] gated rc=%d\n", rc);
            return 10;
        }
    }
    CHECK_CUDA(cudaEventRecord(gated_stop, nullptr));
    CHECK_CUDA(cudaEventSynchronize(gated_stop));

    if (run_probe) {
        CHECK_CUDA(cudaEventRecord(probe_start, nullptr));
        for (int iter = 0; iter < bench_iters; ++iter) {
            rc = probe(d_A, d_offsets, num_experts, total_tokens,
                       (const void * const *) gated_packed.d_w_table,
                       (const void * const *) gated_packed.d_s_table,
                       gated_packed.k_pack, d_probe, nullptr);
            if (rc != 0) {
                fprintf(stderr, "[gate_up_fusion] ds4 probe rc=%d\n", rc);
                return 10;
            }
        }
        CHECK_CUDA(cudaEventRecord(probe_stop, nullptr));
        CHECK_CUDA(cudaEventSynchronize(probe_stop));
    }

    const float separate_ms = elapsed_ms(sep_start, sep_stop) / (float) bench_iters;
    const float fused_ms = elapsed_ms(fused_start, fused_stop) / (float) bench_iters;
    const float gated_ms = elapsed_ms(gated_start, gated_stop) / (float) bench_iters;
    const float probe_ms = run_probe ? elapsed_ms(probe_start, probe_stop) / (float) bench_iters : 0.0f;
    CHECK_CUDA(cudaEventDestroy(sep_start));
    CHECK_CUDA(cudaEventDestroy(sep_stop));
    CHECK_CUDA(cudaEventDestroy(fused_start));
    CHECK_CUDA(cudaEventDestroy(fused_stop));
    CHECK_CUDA(cudaEventDestroy(gated_start));
    CHECK_CUDA(cudaEventDestroy(gated_stop));
    if (run_probe) {
        CHECK_CUDA(cudaEventDestroy(probe_start));
        CHECK_CUDA(cudaEventDestroy(probe_stop));
    }

    std::vector<__half> h_gate((size_t) total_tokens * N);
    std::vector<__half> h_up((size_t) total_tokens * N);
    std::vector<__half> h_fused((size_t) total_tokens * fused_N);
    std::vector<__half> h_gated((size_t) total_tokens * N);
    std::vector<__half> h_probe(run_probe ? (size_t) total_tokens * N : 0);
    CHECK_CUDA(cudaMemcpy(h_gate.data(), d_gate, h_gate.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_up.data(), d_up, h_up.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_fused.data(), d_fused, h_fused.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_gated.data(), d_gated, h_gated.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    if (run_probe) {
        CHECK_CUDA(cudaMemcpy(h_probe.data(), d_probe, h_probe.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    }

    float max_abs_gate = 0.0f;
    float max_abs_up = 0.0f;
    float sum_abs = 0.0f;
    float sum_ref = 0.0f;
    float gated_max_abs = 0.0f;
    float gated_sum_abs = 0.0f;
    float gated_sum_ref = 0.0f;
    float probe_max_abs = 0.0f;
    float probe_sum_abs = 0.0f;
    float probe_sum_ref = 0.0f;
    int bad = 0;
    int gated_bad = 0;
    int probe_bad = 0;
    constexpr float gated_abs_tol = 16.0f;
    const size_t values_per_half = (size_t) total_tokens * N;
    for (int row = 0; row < total_tokens; ++row) {
        for (int col = 0; col < N; ++col) {
            const size_t half_idx = (size_t) row * N + col;
            const size_t fused_gate_idx = (size_t) row * fused_N + col;
            const size_t fused_up_idx = (size_t) row * fused_N + N + col;
            const float gate_ref = __half2float(h_gate[half_idx]);
            const float up_ref = __half2float(h_up[half_idx]);
            const float gate_f = __half2float(h_fused[fused_gate_idx]);
            const float up_f = __half2float(h_fused[fused_up_idx]);
            const float gated = __half2float(h_gated[half_idx]);
            const float probe_v = run_probe ? __half2float(h_probe[half_idx]) : 0.0f;
            const float silu = gate_ref / (1.0f + expf(-gate_ref));
            const float gated_ref = silu * up_ref;
            const float dg = fabsf(gate_f - gate_ref);
            const float du = fabsf(up_f - up_ref);
            const float dgs = fabsf(gated - gated_ref);
            const float dp = run_probe ? fabsf(probe_v - gated) : 0.0f;
            max_abs_gate = std::max(max_abs_gate, dg);
            max_abs_up = std::max(max_abs_up, du);
            gated_max_abs = std::max(gated_max_abs, dgs);
            probe_max_abs = std::max(probe_max_abs, dp);
            sum_abs += dg + du;
            sum_ref += fabsf(gate_ref) + fabsf(up_ref);
            gated_sum_abs += dgs;
            gated_sum_ref += fabsf(gated_ref);
            if (run_probe) {
                probe_sum_abs += dp;
                probe_sum_ref += fabsf(gated);
            }
            if (!std::isfinite(gate_f) || !std::isfinite(up_f) || dg > 0.25f || du > 0.25f) {
                bad++;
            }
            if (!std::isfinite(gated) || dgs > gated_abs_tol) {
                gated_bad++;
            }
            if (run_probe && (!std::isfinite(probe_v) || dp > 0.25f)) {
                probe_bad++;
            }
        }
    }
    const float rel = sum_ref > 0 ? sum_abs / sum_ref : 0.0f;
    const float gated_rel = gated_sum_ref > 0 ? gated_sum_abs / gated_sum_ref : 0.0f;
    const float probe_rel = probe_sum_ref > 0 ? probe_sum_abs / probe_sum_ref : 0.0f;
    fprintf(stderr,
            "[gate_up_fusion tpa=%d] total_routes=%d active=%zu separate_ms=%.4f fused_ms=%.4f gated_ms=%.4f probe_ms=%.4f fused_speedup=%.3fx gated_speedup=%.3fx probe_vs_gated=%.3fx max_abs_gate=%.4e max_abs_up=%.4e rel=%.4e bad=%d/%zu gated_max_abs=%.4e gated_abs_tol=%.1f gated_rel=%.4e gated_bad=%d/%zu probe_max_abs=%.4e probe_rel=%.4e probe_bad=%d/%zu k_pack_sep=0x%x k_pack_fused=0x%x k_pack_gated=0x%x\n",
            c.tokens_per_active, total_tokens, active.size(),
            separate_ms, fused_ms, gated_ms, probe_ms,
            separate_ms / fused_ms, separate_ms / gated_ms, run_probe ? gated_ms / probe_ms : 0.0f,
            max_abs_gate, max_abs_up, rel, bad, 2 * values_per_half,
            gated_max_abs, gated_abs_tol, gated_rel, gated_bad, values_per_half,
            probe_max_abs, probe_rel, probe_bad, values_per_half,
            gate_packed.k_pack, fused_packed.k_pack, gated_packed.k_pack);

    free_packed(gate_packed);
    free_packed(up_packed);
    free_packed(fused_packed);
    free_packed(gated_packed);
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_gate));
    CHECK_CUDA(cudaFree(d_up));
    CHECK_CUDA(cudaFree(d_fused));
    CHECK_CUDA(cudaFree(d_gated));
    if (d_probe) CHECK_CUDA(cudaFree(d_probe));
    sh();

    if (bad != 0 || gated_bad != 0 || probe_bad != 0 ||
        rel > 1e-3f || gated_rel > 1e-3f ||
        (run_probe && probe_rel > 1e-3f) ||
        max_abs_gate > 0.25f || max_abs_up > 0.25f || gated_max_abs > gated_abs_tol) {
        fprintf(stderr, "[gate_up_fusion tpa=%d] FAIL\n", c.tokens_per_active);
        return 11;
    }
    fprintf(stderr, "[gate_up_fusion tpa=%d] PASS\n", c.tokens_per_active);
    return 0;
}

int main(int argc, char ** argv) {
    const char * lib_path = argc > 1 ? argv[1] : "./libggml-turbomind.so";
    void * lib = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    const std::vector<Case> cases = parse_cases_from_env();

    int failures = 0;
    for (const Case & c : cases) {
        failures += run_case(lib, c) != 0;
    }
    dlclose(lib);
    return failures ? 1 : 0;
}
