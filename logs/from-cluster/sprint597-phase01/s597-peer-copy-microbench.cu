// Sprint 597 Phase 1: standalone peer-copy microbench.
//
// Times copy_f32_kernel-style UVA remote loads for all 56 directed (dst, src)
// GPU pairs plus the 8 same-device controls, at EP-return payload sizes.
// Mirrors the promoted EP-return transport semantics:
//   - kernel launched on the dst device's stream (engine/runtime_pack.cu:176-190,
//     enqueue_graph_f32_copy_between_devices ignores device ids; UVA remote load)
//   - block size 256 (engine/decode_loop.cu:1092)
//   - peer access enabled the way engine/tp_runtime.cu:1110-1131 does it
//     (cudaDeviceCanAccessPeer check then cudaDeviceEnablePeerAccess, tolerating
//     cudaErrorPeerAccessAlreadyEnabled)
//
// This is a NEW standalone tool. It does not touch the serving hot path.
//
// Build (pod):
//   nvcc -O3 -arch=sm_70 -o /workspace/s597-phase01-artifacts/s597-peer-copy-microbench \
//        /workspace/ds4/tools/s597-peer-copy-microbench.cu
// Run:
//   ./s597-peer-copy-microbench > peer-copy-microbench.tsv

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t rc__ = (call);                                            \
        if (rc__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",              \
                         cudaGetErrorName(rc__), __FILE__, __LINE__,          \
                         cudaGetErrorString(rc__));                           \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// Same body as kernels/v100/common.cuh:33 copy_f32_kernel.
__global__ void s597_copy_f32_kernel(float *dst, const float *src, uint64_t n) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (uint64_t)blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

int main() {
    int ngpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&ngpus));
    if (ngpus < 8) {
        std::fprintf(stderr, "need 8 GPUs, got %d\n", ngpus);
        return 1;
    }
    const int kGpus = 8;
    const int block = 256;          // engine/decode_loop.cu:1092
    const int warmup_iters = 20;
    const int burst_iters = 200;    // pipelined throughput window
    const int isolated_iters = 50;  // per-launch latency samples

    // Payload sizes (bytes): the EP-return band, plus the exact promoted
    // fixed-capacity payload 192 routes x (4096/8) f32 = 384 KiB.
    const uint64_t sizes[] = {8ull << 10, 64ull << 10, 192ull << 10,
                              384ull << 10, 512ull << 10};
    const int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    const uint64_t max_bytes = sizes[nsizes - 1];
    const uint64_t max_elems = max_bytes / sizeof(float);

    // Peer-access matrix + enable, mirroring engine/tp_runtime.cu.
    int can[kGpus][kGpus];
    for (int i = 0; i < kGpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        for (int j = 0; j < kGpus; ++j) {
            if (i == j) { can[i][j] = 1; continue; }
            int c = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&c, i, j));
            can[i][j] = c;
            if (!c) {
                std::fprintf(stderr, "peer access unavailable %d -> %d\n", i, j);
                return 1;
            }
            cudaError_t rc = cudaDeviceEnablePeerAccess(j, 0);
            if (rc == cudaErrorPeerAccessAlreadyEnabled) {
                (void)cudaGetLastError();
            } else if (rc != cudaSuccess) {
                std::fprintf(stderr, "enable peer %d -> %d failed: %s\n", i, j,
                             cudaGetErrorString(rc));
                return 1;
            }
        }
    }

    float *d_src[kGpus];
    float *d_dst[kGpus];
    cudaStream_t stream[kGpus];
    cudaEvent_t ev_a[kGpus], ev_b[kGpus];
    for (int i = 0; i < kGpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&d_src[i], max_bytes));
        CHECK_CUDA(cudaMalloc(&d_dst[i], max_bytes));
        CHECK_CUDA(cudaMemset(d_src[i], 0x3c, max_bytes));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
        CHECK_CUDA(cudaEventCreate(&ev_a[i]));
        CHECK_CUDA(cudaEventCreate(&ev_b[i]));
    }
    for (int i = 0; i < kGpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    std::printf("schema\ts597_peer_copy_microbench.v1\n");
    std::printf("bytes\tdst\tsrc\tsame_device\tgrid_blocks\tblock\t"
                "burst_iters\tburst_us_per_copy\tburst_gbps\t"
                "isolated_iters\tisolated_us_per_copy\tisolated_gbps\n");

    for (int s = 0; s < nsizes; ++s) {
        const uint64_t bytes = sizes[s];
        const uint64_t elems = bytes / sizeof(float);
        const unsigned int grid =
            (unsigned int)((elems + (uint64_t)block - 1) / (uint64_t)block);
        for (int dst = 0; dst < kGpus; ++dst) {
            CHECK_CUDA(cudaSetDevice(dst));
            for (int src = 0; src < kGpus; ++src) {
                // Warmup.
                for (int it = 0; it < warmup_iters; ++it) {
                    s597_copy_f32_kernel<<<grid, block, 0, stream[dst]>>>(
                        d_dst[dst], d_src[src], elems);
                }
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaStreamSynchronize(stream[dst]));

                // Burst (pipelined) window.
                CHECK_CUDA(cudaEventRecord(ev_a[dst], stream[dst]));
                for (int it = 0; it < burst_iters; ++it) {
                    s597_copy_f32_kernel<<<grid, block, 0, stream[dst]>>>(
                        d_dst[dst], d_src[src], elems);
                }
                CHECK_CUDA(cudaEventRecord(ev_b[dst], stream[dst]));
                CHECK_CUDA(cudaEventSynchronize(ev_b[dst]));
                float burst_ms = 0.0f;
                CHECK_CUDA(cudaEventElapsedTime(&burst_ms, ev_a[dst], ev_b[dst]));
                const double burst_us = burst_ms * 1000.0 / burst_iters;
                const double burst_gbps =
                    (double)bytes / (burst_us * 1e-6) / 1e9;

                // Isolated launches (kernel-duration latency).
                double iso_total_ms = 0.0;
                for (int it = 0; it < isolated_iters; ++it) {
                    CHECK_CUDA(cudaEventRecord(ev_a[dst], stream[dst]));
                    s597_copy_f32_kernel<<<grid, block, 0, stream[dst]>>>(
                        d_dst[dst], d_src[src], elems);
                    CHECK_CUDA(cudaEventRecord(ev_b[dst], stream[dst]));
                    CHECK_CUDA(cudaEventSynchronize(ev_b[dst]));
                    float ms = 0.0f;
                    CHECK_CUDA(cudaEventElapsedTime(&ms, ev_a[dst], ev_b[dst]));
                    iso_total_ms += (double)ms;
                }
                const double iso_us = iso_total_ms * 1000.0 / isolated_iters;
                const double iso_gbps = (double)bytes / (iso_us * 1e-6) / 1e9;

                std::printf("%llu\t%d\t%d\t%d\t%u\t%d\t%d\t%.3f\t%.3f\t%d\t%.3f\t%.3f\n",
                            (unsigned long long)bytes, dst, src,
                            dst == src ? 1 : 0, grid, block,
                            burst_iters, burst_us, burst_gbps,
                            isolated_iters, iso_us, iso_gbps);
                std::fflush(stdout);
            }
        }
    }

    for (int i = 0; i < kGpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(d_src[i]));
        CHECK_CUDA(cudaFree(d_dst[i]));
        CHECK_CUDA(cudaStreamDestroy(stream[i]));
        CHECK_CUDA(cudaEventDestroy(ev_a[i]));
        CHECK_CUDA(cudaEventDestroy(ev_b[i]));
    }
    return 0;
}
