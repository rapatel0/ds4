// Measure CUDA peer-copy payload timing for DS4 TP/EP planning.
//
// This is a communication proxy, not a correctness test. Tensor-parallel FFN
// split needs to exchange/reduce one hidden vector per routed token after the
// down projection. The copy payload here gives a lower-bound timing for that
// movement on the actual V100 topology.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
} } while (0)

static int env_int(const char *name, int fallback, int lo, int hi) {
    const char *v = std::getenv(name);
    if (!v || !v[0]) return fallback;
    char *end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (!end || *end != '\0' || parsed < lo || parsed > hi) {
        fprintf(stderr, "[p2p_reduce_proxy] ignoring invalid %s=%s\n", name, v);
        return fallback;
    }
    return (int)parsed;
}

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return (double)ms;
}

static void enable_peer_if_possible(int src, int dst) {
    int can = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can, src, dst));
    if (!can) {
        fprintf(stderr, "[p2p_reduce_proxy] peer access unavailable src=%d dst=%d\n", src, dst);
        return;
    }

    CHECK_CUDA(cudaSetDevice(src));
    cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
        CHECK_CUDA(e);
    } else if (e == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
    }
}

static double measure_copy_ms(int src, int dst, size_t bytes, int warmup, int iters) {
    enable_peer_if_possible(src, dst);
    enable_peer_if_possible(dst, src);

    void *d_src = nullptr;
    void *d_dst = nullptr;
    CHECK_CUDA(cudaSetDevice(src));
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMemset(d_src, 0x5a, bytes));
    CHECK_CUDA(cudaSetDevice(dst));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));
    CHECK_CUDA(cudaMemset(d_dst, 0, bytes));

    CHECK_CUDA(cudaSetDevice(src));
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        CHECK_CUDA(cudaMemcpyPeerAsync(d_dst, dst, d_src, src, bytes, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaMemcpyPeerAsync(d_dst, dst, d_src, src, bytes, stream));
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    const double ms = elapsed_ms(start, stop) / (double)iters;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaSetDevice(src));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaSetDevice(dst));
    CHECK_CUDA(cudaFree(d_dst));
    return ms;
}

int main() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        fprintf(stderr, "[p2p_reduce_proxy] need at least 2 GPUs, found %d\n", device_count);
        return 1;
    }

    const int routes = env_int("DS4_P2P_ROUTES", 768, 1, 4096);
    const int hidden = env_int("DS4_P2P_HIDDEN", 4096, 1, 65536);
    const int bytes_per_value = env_int("DS4_P2P_BYTES_PER_VALUE", 2, 1, 4);
    const int warmup = env_int("DS4_P2P_WARMUP_ITERS", 10, 0, 1000);
    const int iters = env_int("DS4_P2P_BENCH_ITERS", 100, 1, 100000);
    const size_t bytes = (size_t)routes * (size_t)hidden * (size_t)bytes_per_value;
    const double mib = (double)bytes / (1024.0 * 1024.0);

    std::vector<std::pair<int, int>> pairs;
    pairs.push_back({0, 1});
    pairs.push_back({0, 3});
    pairs.push_back({0, 4});
    pairs.push_back({0, 5});
    pairs.push_back({4, 5});
    pairs.push_back({4, 7});
    pairs.push_back({3, 7});

    fprintf(stderr,
            "[p2p_reduce_proxy] devices=%d routes=%d hidden=%d bytes_per_value=%d payload_mib=%.2f warmup=%d iters=%d\n",
            device_count, routes, hidden, bytes_per_value, mib, warmup, iters);

    int failures = 0;
    for (const auto &p : pairs) {
        const int src = p.first;
        const int dst = p.second;
        if (src >= device_count || dst >= device_count) {
            continue;
        }
        int can = 0;
        CHECK_CUDA(cudaDeviceCanAccessPeer(&can, src, dst));
        if (!can) {
            fprintf(stderr, "[p2p_reduce_proxy] src=%d dst=%d peer_access=0 skipped\n", src, dst);
            failures++;
            continue;
        }
        const double ms = measure_copy_ms(src, dst, bytes, warmup, iters);
        const double gbps = ((double)bytes / 1.0e9) / (ms / 1000.0);
        fprintf(stderr,
                "[p2p_reduce_proxy] src=%d dst=%d peer_access=1 copy_ms=%.4f payload_gbps=%.2f\n",
                src, dst, ms, gbps);
    }

    return failures ? 1 : 0;
}
