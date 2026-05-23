// Four-GPU resident TP4 layer-slice proxy for DS4/V100.
//
// This benchmark composes the real TurboMind MXFP4 routed-FFN TP4 split with a
// resident per-layer hidden all-reduce. It is intentionally still a proxy: it
// does not implement DS4 attention, norms, or scheduler integration. Its job is
// to answer whether TP4 remains plausible when the routed FFN and the layer
// boundary live in the same device-resident loop instead of copying hidden
// state into and out of a routed-only overlay.

#define DS4_TP_SPLIT_4GPU_NO_MAIN
#include "test_tp_split_4gpu.cpp"

namespace {

__global__ void sum4_half_kernel(const __half * a,
                                 const __half * b,
                                 const __half * c,
                                 const __half * d,
                                 __half * out,
                                 size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __half2float(a[i]) + __half2float(b[i]) +
                    __half2float(c[i]) + __half2float(d[i]);
    out[i] = __float2half(v);
}

__global__ void add_half_inplace_kernel(__half * dst, const __half * src, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
}

enum class ResidentReduceAlgo {
    Root,
    Doubling,
};

static int resident_env_int(const char *name, int fallback, int lo, int hi) {
    const char *v = std::getenv(name);
    if (!v || !v[0]) return fallback;
    char *end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (!end || *end != '\0' || parsed < lo || parsed > hi) {
        fprintf(stderr, "[tp4_resident] ignoring invalid %s=%s\n", name, v);
        return fallback;
    }
    return (int) parsed;
}

static ResidentReduceAlgo resident_reduce_algo_from_env() {
    const char *v = std::getenv("DS4_TP4_RESIDENT_ALGO");
    if (!v || !v[0] || std::strcmp(v, "root") == 0) return ResidentReduceAlgo::Root;
    if (std::strcmp(v, "doubling") == 0) return ResidentReduceAlgo::Doubling;
    fprintf(stderr, "[tp4_resident] ignoring invalid DS4_TP4_RESIDENT_ALGO=%s\n", v);
    return ResidentReduceAlgo::Root;
}

static const char * resident_reduce_algo_name(ResidentReduceAlgo algo) {
    return algo == ResidentReduceAlgo::Doubling ? "doubling" : "root";
}

static void sync_devices(const std::array<int, kParts> & devices) {
    for (int p = 0; p < kParts; ++p) {
        CHECK_CUDA(cudaSetDevice(devices[p]));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

static int run_full_layers(DeviceSide & full,
                           const Api & api,
                           int total_tokens,
                           int layers) {
    for (int layer = 0; layer < layers; ++layer) {
        const int rc = run_side(full, api, total_tokens, kFusedN, kMid);
        if (rc != 0) return rc;
        CHECK_CUDA(cudaSetDevice(full.device));
        CHECK_CUDA(cudaStreamSynchronize(full.stream));
        std::swap(full.d_A, full.d_down);
    }
    return 0;
}

static int run_tp4_resident_layers(std::array<DeviceSide, kParts> & sides,
                                   const Api & api,
                                   const std::array<int, kParts> & devices,
                                   int total_tokens,
                                   int layers,
                                   std::array<__half *, kParts> & reduce_recv,
                                   ResidentReduceAlgo algo) {
    const size_t elems = (size_t) total_tokens * kHidden;
    const size_t bytes = elems * sizeof(__half);
    constexpr int threads = 256;
    const int blocks = (int) ((elems + threads - 1) / threads);

    for (int layer = 0; layer < layers; ++layer) {
        for (int p = 0; p < kParts; ++p) {
            const int rc = run_side(sides[p], api, total_tokens, kFusedPartN, kMidPart);
            if (rc != 0) return rc;
        }
        for (int p = 0; p < kParts; ++p) {
            CHECK_CUDA(cudaSetDevice(devices[p]));
            CHECK_CUDA(cudaStreamSynchronize(sides[p].stream));
        }

        if (algo == ResidentReduceAlgo::Root) {
            for (int p = 1; p < kParts; ++p) {
                CHECK_CUDA(cudaMemcpyPeer(reduce_recv[p], devices[0],
                                          sides[p].d_down, devices[p],
                                          bytes));
            }
            CHECK_CUDA(cudaSetDevice(devices[0]));
            sum4_half_kernel<<<blocks, threads, 0, sides[0].stream>>>(
                sides[0].d_down,
                reduce_recv[1],
                reduce_recv[2],
                reduce_recv[3],
                sides[0].d_A,
                elems);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(sides[0].stream));

            for (int p = 1; p < kParts; ++p) {
                CHECK_CUDA(cudaMemcpyPeer(sides[p].d_A, devices[p],
                                          sides[0].d_A, devices[0],
                                          bytes));
            }
        } else {
            const int round1_peer[kParts] = {1, 0, 3, 2};
            const int round2_peer[kParts] = {2, 3, 0, 1};

            for (int p = 0; p < kParts; ++p) {
                const int peer = round1_peer[p];
                CHECK_CUDA(cudaMemcpyPeer(reduce_recv[p], devices[p],
                                          sides[peer].d_down, devices[peer],
                                          bytes));
            }
            for (int p = 0; p < kParts; ++p) {
                CHECK_CUDA(cudaSetDevice(devices[p]));
                add_half_inplace_kernel<<<blocks, threads, 0, sides[p].stream>>>(
                    sides[p].d_down, reduce_recv[p], elems);
                CHECK_CUDA(cudaGetLastError());
            }
            for (int p = 0; p < kParts; ++p) {
                CHECK_CUDA(cudaSetDevice(devices[p]));
                CHECK_CUDA(cudaStreamSynchronize(sides[p].stream));
            }

            for (int p = 0; p < kParts; ++p) {
                const int peer = round2_peer[p];
                CHECK_CUDA(cudaMemcpyPeer(reduce_recv[p], devices[p],
                                          sides[peer].d_down, devices[peer],
                                          bytes));
            }
            for (int p = 0; p < kParts; ++p) {
                CHECK_CUDA(cudaSetDevice(devices[p]));
                add_half_inplace_kernel<<<blocks, threads, 0, sides[p].stream>>>(
                    sides[p].d_down, reduce_recv[p], elems);
                CHECK_CUDA(cudaGetLastError());
            }
            for (int p = 0; p < kParts; ++p) {
                CHECK_CUDA(cudaSetDevice(devices[p]));
                CHECK_CUDA(cudaStreamSynchronize(sides[p].stream));
                std::swap(sides[p].d_A, sides[p].d_down);
            }
        }
    }
    return 0;
}

static int compare_hidden(const std::vector<__half> & full,
                          const std::vector<__half> & tp,
                          int tokens_per_active,
                          int total_tokens,
                          int layers) {
    if (full.size() != tp.size()) {
        fprintf(stderr, "[tp4_resident tpa=%d] correctness FAIL size mismatch full=%zu tp=%zu\n",
                tokens_per_active, full.size(), tp.size());
        return 1;
    }

    double sum_abs = 0.0;
    double sum_ref = 0.0;
    float max_abs = 0.0f;
    int bad = 0;
    int nan = 0;
    constexpr float abs_tol = 64.0f;
    constexpr float rel_elem_tol = 0.10f;
    constexpr double rel_tol = 0.02;
    constexpr double bad_frac_tol = 0.005;

    for (size_t i = 0; i < full.size(); ++i) {
        const float ref = __half2float(full[i]);
        const float got = __half2float(tp[i]);
        if (!std::isfinite(ref) || !std::isfinite(got)) {
            nan++;
            bad++;
            continue;
        }
        const float diff = fabsf(got - ref);
        const float elem_tol = std::max(abs_tol, rel_elem_tol * fabsf(ref));
        max_abs = std::max(max_abs, diff);
        sum_abs += (double) diff;
        sum_ref += (double) fabsf(ref);
        if (diff > elem_tol) {
            bad++;
        }
    }

    const double rel = sum_ref > 0.0 ? sum_abs / sum_ref : 0.0;
    const double bad_frac = full.empty() ? 0.0 : (double) bad / (double) full.size();
    const bool rel_meaningful = sum_ref > 1.0e-3;
    const bool fail = nan != 0 || (rel_meaningful && rel > rel_tol) ||
                      bad_frac > bad_frac_tol;
    fprintf(stderr,
            "[tp4_resident tpa=%d layers=%d] correctness total_routes=%d values=%zu max_abs=%.4e rel=%.4e bad=%d bad_frac=%.4e nan=%d status=%s\n",
            tokens_per_active,
            layers,
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

static int run_resident_case(void * lib, const Case & c) {
    const Api api = load_api(lib);
    const std::array<int, kParts> devices = parse_devices_from_env();
    const int layers = resident_env_int("DS4_TP4_RESIDENT_LAYERS", 4, 1, 43);
    const int warmup_iters = resident_env_int("DS4_TP4_RESIDENT_WARMUP_ITERS", 2, 0, 1000);
    const int bench_iters = resident_env_int("DS4_TP4_RESIDENT_BENCH_ITERS", 10, 1, 10000);
    const bool verbose = resident_env_int("DS4_TP4_RESIDENT_VERBOSE", 0, 0, 1) != 0;
    const ResidentReduceAlgo algo = resident_reduce_algo_from_env();

    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    for (int p = 0; p < kParts; ++p) {
        if (devices[p] >= dev_count) {
            fprintf(stderr, "[tp4_resident] invalid device %d visible=%d\n",
                    devices[p], dev_count);
            return 2;
        }
        CHECK_CUDA(cudaSetDevice(devices[p]));
        if (api.init(devices[p]) != 0) return 3;
    }
    enable_peer_all(devices);

    const std::vector<int> active{0, 1, 2, 3, 4, 5};
    const int total_tokens = (int) active.size() * c.tokens_per_active;

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
        make_mxfp4_fixture(gate[i], kMid, kHidden, 0x71000000u + (uint32_t) i * 101u);
        make_mxfp4_fixture(up[i],   kMid, kHidden, 0x72000000u + (uint32_t) i * 131u);
        make_mxfp4_fixture(down[i], kHidden, kMid, 0x73000000u + (uint32_t) i * 137u);
        make_fused_interleaved_fixture(gated_full[i], gate[i], up[i], kMid, kHidden);
        for (int p = 0; p < kParts; ++p) {
            const int begin = p * kMidPart;
            std::vector<block_mxfp4> gate_slice;
            std::vector<block_mxfp4> up_slice;
            slice_rows_fixture(gate_slice, gate[i], kHidden, begin, kMidPart);
            slice_rows_fixture(up_slice, up[i], kHidden, begin, kMidPart);
            make_fused_interleaved_fixture(gated_part[p][i], gate_slice, up_slice,
                                           kMidPart, kHidden);
            slice_cols_fixture(down_part[p][i], down[i], kHidden, kMid, begin, kMidPart);
        }
    }

    std::vector<int> h_offsets(kNumExperts + 1, 0);
    int running = 0;
    for (int e = 0; e < kNumExperts; ++e) {
        h_offsets[e] = running;
        running += c.tokens_per_active;
    }
    h_offsets[kNumExperts] = running;

    std::mt19937 rng(0xC2030000u + (uint32_t) c.tokens_per_active);
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
    CHECK_CUDA(cudaMemcpy(full.d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&full.d_A, h_A.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&full.d_gated, (size_t) total_tokens * kMid * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&full.d_down, (size_t) total_tokens * kHidden * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(full.d_A, h_A.data(), h_A.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
    if (pack_fixture_set(full.device, api, kFusedN, kHidden, active, gated_full, full.gated) != 0 ||
        pack_fixture_set(full.device, api, kHidden, kMid, active, down, full.down) != 0) {
        fprintf(stderr, "[tp4_resident] full pack failed\n");
        return 4;
    }

    std::array<DeviceSide, kParts> sides;
    for (int p = 0; p < kParts; ++p) {
        sides[p].device = devices[p];
        CHECK_CUDA(cudaSetDevice(sides[p].device));
        CHECK_CUDA(cudaStreamCreate(&sides[p].stream));
        CHECK_CUDA(cudaMalloc(&sides[p].d_offsets, h_offsets.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(sides[p].d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&sides[p].d_A, h_A.size() * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&sides[p].d_gated, (size_t) total_tokens * kMidPart * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&sides[p].d_down, (size_t) total_tokens * kHidden * sizeof(__half)));
        CHECK_CUDA(cudaMemcpy(sides[p].d_A, h_A.data(), h_A.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
        if (pack_fixture_set(sides[p].device, api, kFusedPartN, kHidden,
                             active, gated_part[p], sides[p].gated) != 0 ||
            pack_fixture_set(sides[p].device, api, kHidden, kMidPart,
                             active, down_part[p], sides[p].down) != 0) {
            fprintf(stderr, "[tp4_resident] part %d pack failed\n", p);
            return 5;
        }
    }

    std::array<__half *, kParts> reduce_recv{};
    for (int p = 0; p < kParts; ++p) {
        if (algo == ResidentReduceAlgo::Root && p == 0) continue;
        CHECK_CUDA(cudaSetDevice(algo == ResidentReduceAlgo::Root ? devices[0] : devices[p]));
        CHECK_CUDA(cudaMalloc(&reduce_recv[p], h_A.size() * sizeof(__half)));
    }

    if (verbose) {
        fprintf(stderr,
                "[tp4_resident] start tpa=%d routes=%d layers=%d algo=%s gpus=%d,%d,%d,%d warmup=%d iters=%d\n",
                c.tokens_per_active, total_tokens, layers,
                resident_reduce_algo_name(algo),
                devices[0], devices[1], devices[2], devices[3],
                warmup_iters, bench_iters);
        fflush(stderr);
    }

    for (int i = 0; i < warmup_iters; ++i) {
        CHECK_CUDA(cudaSetDevice(full.device));
        CHECK_CUDA(cudaMemcpy(full.d_A, h_A.data(), h_A.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
        if (run_full_layers(full, api, total_tokens, layers) != 0) return 6;
        for (int p = 0; p < kParts; ++p) {
            CHECK_CUDA(cudaSetDevice(sides[p].device));
            CHECK_CUDA(cudaMemcpy(sides[p].d_A, h_A.data(), h_A.size() * sizeof(__half),
                                  cudaMemcpyHostToDevice));
        }
        if (run_tp4_resident_layers(sides, api, devices, total_tokens, layers,
                                    reduce_recv, algo) != 0) {
            return 7;
        }
    }
    sync_devices(devices);

    double full_total_ms = 0.0;
    for (int i = 0; i < bench_iters; ++i) {
        CHECK_CUDA(cudaSetDevice(full.device));
        CHECK_CUDA(cudaMemcpy(full.d_A, h_A.data(), h_A.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
        const auto start = std::chrono::steady_clock::now();
        if (run_full_layers(full, api, total_tokens, layers) != 0) return 8;
        CHECK_CUDA(cudaSetDevice(full.device));
        CHECK_CUDA(cudaStreamSynchronize(full.stream));
        const auto stop = std::chrono::steady_clock::now();
        full_total_ms += std::chrono::duration<double, std::milli>(stop - start).count();
    }
    full_total_ms /= (double) bench_iters;

    double tp_total_ms = 0.0;
    for (int i = 0; i < bench_iters; ++i) {
        for (int p = 0; p < kParts; ++p) {
            CHECK_CUDA(cudaSetDevice(sides[p].device));
            CHECK_CUDA(cudaMemcpy(sides[p].d_A, h_A.data(), h_A.size() * sizeof(__half),
                                  cudaMemcpyHostToDevice));
        }
        sync_devices(devices);
        const auto start = std::chrono::steady_clock::now();
        if (run_tp4_resident_layers(sides, api, devices, total_tokens, layers,
                                    reduce_recv, algo) != 0) {
            return 9;
        }
        sync_devices(devices);
        const auto stop = std::chrono::steady_clock::now();
        tp_total_ms += std::chrono::duration<double, std::milli>(stop - start).count();
    }
    tp_total_ms /= (double) bench_iters;

    CHECK_CUDA(cudaSetDevice(full.device));
    CHECK_CUDA(cudaMemcpy(full.d_A, h_A.data(), h_A.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
    if (run_full_layers(full, api, total_tokens, layers) != 0) return 10;
    for (int p = 0; p < kParts; ++p) {
        CHECK_CUDA(cudaSetDevice(sides[p].device));
        CHECK_CUDA(cudaMemcpy(sides[p].d_A, h_A.data(), h_A.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
    }
    if (run_tp4_resident_layers(sides, api, devices, total_tokens, layers,
                                reduce_recv, algo) != 0) {
        return 11;
    }

    std::vector<__half> h_full(h_A.size());
    std::vector<__half> h_tp(h_A.size());
    CHECK_CUDA(cudaSetDevice(full.device));
    CHECK_CUDA(cudaMemcpy(h_full.data(), full.d_A, h_full.size() * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaSetDevice(devices[0]));
    CHECK_CUDA(cudaMemcpy(h_tp.data(), sides[0].d_A, h_tp.size() * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    const int correctness_rc = compare_hidden(h_full, h_tp, c.tokens_per_active,
                                              total_tokens, layers);

    const size_t hidden_bytes = h_A.size() * sizeof(__half);
    const double boundary_mib =
        (double) hidden_bytes * 6.0 * (double) layers / (1024.0 * 1024.0);
    fprintf(stderr,
            "[tp4_resident tpa=%d] algo=%s gpus=%d,%d,%d,%d routes=%d layers=%d full_total_ms=%.4f tp4_resident_ms=%.4f full_ms_per_layer=%.4f tp4_ms_per_layer=%.4f speedup=%.3fx boundary_mib_per_iter=%.2f\n",
            c.tokens_per_active,
            resident_reduce_algo_name(algo),
            devices[0], devices[1], devices[2], devices[3],
            total_tokens,
            layers,
            full_total_ms,
            tp_total_ms,
            full_total_ms / (double) layers,
            tp_total_ms / (double) layers,
            full_total_ms / tp_total_ms,
            boundary_mib);

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
    for (int p = 0; p < kParts; ++p) {
        if (!reduce_recv[p]) continue;
        CHECK_CUDA(cudaSetDevice(algo == ResidentReduceAlgo::Root ? devices[0] : devices[p]));
        CHECK_CUDA(cudaFree(reduce_recv[p]));
    }

    api.shutdown();
    return correctness_rc;
}

} // namespace

int main(int argc, char ** argv) {
    const char * lib_path = argc > 1 ? argv[1] : "./libggml-turbomind.so";
    void * lib = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    int failures = 0;
    for (const Case & c : parse_cases_from_env()) {
        failures += run_resident_case(lib, c) != 0;
    }
    dlclose(lib);
    return failures ? 1 : 0;
}
