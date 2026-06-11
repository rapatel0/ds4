// Sprint 598 C1 capture probe (standalone, no engine dependencies).
//
// Proves (or precisely refutes) that the grouped per-source ncclBroadcast
// pattern used by broadcast_ep_return_slices (engine/runtime_pack.cu:267)
// can be stream-captured into a single 8-rank single-process CUDA graph and
// replayed stably with this NCCL build, using the same fork/join pattern as
// engine/decode_loop.cu's capture probe (origin stream on rank 0, seed event
// fan-out, join events back to the origin).
//
// Payload mirrors the promoted EP-return shape: per src, 8 segments x
// 192 routes x 512 f32 (3 MiB broadcast per src, 384 KiB slice per dst).
//
// Build (pod):
//   nvcc -O3 -arch=sm_70 -o /workspace/s598-artifacts/s598-nccl-capture-probe \
//        /workspace/ds4/tools/s598-nccl-capture-probe.cu -lnccl
// Run:
//   NCCL_P2P_LEVEL=NVL ./s598-nccl-capture-probe

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>

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

#define CHECK_NCCL(call)                                                      \
    do {                                                                      \
        ncclResult_t rc__ = (call);                                           \
        if (rc__ != ncclSuccess) {                                            \
            std::fprintf(stderr, "NCCL error %d (%s) at %s:%d\n", (int)rc__,  \
                         ncclGetErrorString(rc__), __FILE__, __LINE__);       \
            std::printf("s598_nccl_capture_probe\tFAIL\tnccl_error\t%s\n",    \
                        ncclGetErrorString(rc__));                            \
            std::exit(2);                                                     \
        }                                                                     \
    } while (0)

static const int kGpus = 8;
static const int kRoutes = 192;
static const int kSlice = 512; // kHidden / kGpus
static const size_t kSegElems = (size_t)kRoutes * kSlice;       // 98304
static const size_t kAllElems = (size_t)kGpus * kSegElems;      // 786432

__global__ void fill_pattern_kernel(float *dst, size_t n, float base) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)blockDim.x * gridDim.x) {
        dst[i] = base + (float)(i % 1024);
    }
}

__global__ void check_pattern_kernel(const float *src, size_t n, float base,
                                     int *bad) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)blockDim.x * gridDim.x) {
        if (src[i] != base + (float)(i % 1024)) atomicAdd(bad, 1);
    }
}

int main() {
    int ndev = 0;
    CHECK_CUDA(cudaGetDeviceCount(&ndev));
    if (ndev < kGpus) {
        std::fprintf(stderr, "need 8 GPUs, got %d\n", ndev);
        return 1;
    }
    int nccl_version = 0;
    CHECK_NCCL(ncclGetVersion(&nccl_version));

    int devs[kGpus];
    for (int i = 0; i < kGpus; ++i) devs[i] = i;
    ncclComm_t comms[kGpus];
    CHECK_NCCL(ncclCommInitAll(comms, kGpus, devs));

    cudaStream_t stream[kGpus];
    float *d_contrib[kGpus];   // src-side: 8 segments
    float *d_bcast[kGpus];     // bcast scratch: 8 segments
    float *d_remote[kGpus][kGpus]; // dst-side per-src slice
    int *d_bad[kGpus];
    for (int r = 0; r < kGpus; ++r) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream[r], cudaStreamNonBlocking));
        CHECK_CUDA(cudaMalloc(&d_contrib[r], kAllElems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_bcast[r], kAllElems * sizeof(float)));
        for (int s = 0; s < kGpus; ++s) {
            CHECK_CUDA(cudaMalloc(&d_remote[r][s], kSegElems * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_remote[r][s], 0, kSegElems * sizeof(float)));
        }
        CHECK_CUDA(cudaMalloc(&d_bad[r], sizeof(int)));
    }

    auto fill_contribs = [&](float epoch) {
        for (int r = 0; r < kGpus; ++r) {
            CHECK_CUDA(cudaSetDevice(r));
            fill_pattern_kernel<<<256, 256, 0, stream[r]>>>(
                d_contrib[r], kAllElems, epoch + 10000.0f * (float)r);
            CHECK_CUDA(cudaStreamSynchronize(stream[r]));
        }
    };

    // Enqueue one EP-return broadcast round: for each src, a grouped
    // 8-rank ncclBroadcast of src's full contrib into every rank's bcast
    // scratch, then each dst slices its segment into d_remote[dst][src].
    auto enqueue_round = [&]() {
        for (int src = 0; src < kGpus; ++src) {
            CHECK_NCCL(ncclGroupStart());
            for (int r = 0; r < kGpus; ++r) {
                CHECK_CUDA(cudaSetDevice(r));
                const void *send = r == src ? (const void *)d_contrib[src]
                                            : (const void *)d_bcast[r];
                CHECK_NCCL(ncclBroadcast(send, d_bcast[r],
                                         kAllElems * sizeof(float), ncclChar,
                                         src, comms[r], stream[r]));
            }
            CHECK_NCCL(ncclGroupEnd());
            for (int dst = 0; dst < kGpus; ++dst) {
                CHECK_CUDA(cudaSetDevice(dst));
                CHECK_CUDA(cudaMemcpyAsync(
                    d_remote[dst][src], d_bcast[dst] + (size_t)dst * kSegElems,
                    kSegElems * sizeof(float), cudaMemcpyDeviceToDevice,
                    stream[dst]));
            }
        }
    };

    auto verify = [&](float epoch, const char *label) -> bool {
        int total_bad = 0;
        for (int dst = 0; dst < kGpus; ++dst) {
            CHECK_CUDA(cudaSetDevice(dst));
            CHECK_CUDA(cudaMemset(d_bad[dst], 0, sizeof(int)));
            for (int src = 0; src < kGpus; ++src) {
                // d_remote[dst][src] should hold contrib[src] segment dst:
                // pattern base + offset (dst*kSegElems) % 1024 folded in by
                // checking against a shifted base is awkward; instead check
                // element-wise vs the expected generator with global index.
                // Expected value at slice i = epoch + 10000*src +
                // ((dst*kSegElems + i) % 1024).
                // Re-use check kernel by passing base and recomputing via a
                // device-side closure is not possible; do an offset check:
                // since kSegElems % 1024 == 0, (dst*kSegElems + i) % 1024 ==
                // i % 1024, so base is epoch + 10000*src.
                check_pattern_kernel<<<256, 256, 0, stream[dst]>>>(
                    d_remote[dst][src], kSegElems,
                    epoch + 10000.0f * (float)src, d_bad[dst]);
            }
            int bad = 0;
            CHECK_CUDA(cudaStreamSynchronize(stream[dst]));
            CHECK_CUDA(cudaMemcpy(&bad, d_bad[dst], sizeof(int),
                                  cudaMemcpyDeviceToHost));
            total_bad += bad;
        }
        std::printf("s598_probe_verify\t%s\tbad_elems\t%d\t%s\n", label,
                    total_bad, total_bad == 0 ? "PASS" : "FAIL");
        return total_bad == 0;
    };

    // ---- Eager control round -------------------------------------------
    fill_contribs(1.0f);
    enqueue_round();
    for (int r = 0; r < kGpus; ++r) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaStreamSynchronize(stream[r]));
    }
    if (!verify(1.0f, "eager")) {
        std::printf("s598_nccl_capture_probe\tFAIL\teager_parity\n");
        return 3;
    }

    // ---- Capture (fork/join from rank-0 origin, engine pattern) ---------
    fill_contribs(2.0f);
    cudaEvent_t seed, join[kGpus];
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventCreateWithFlags(&seed, cudaEventDisableTiming));
    for (int r = 0; r < kGpus; ++r) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaEventCreateWithFlags(&join[r], cudaEventDisableTiming));
    }
    CHECK_CUDA(cudaSetDevice(0));
    cudaError_t rc = cudaStreamBeginCapture(stream[0],
                                            cudaStreamCaptureModeGlobal);
    if (rc != cudaSuccess) {
        std::printf("s598_nccl_capture_probe\tFAIL\tbegin_capture\t%s\n",
                    cudaGetErrorName(rc));
        return 4;
    }
    CHECK_CUDA(cudaEventRecord(seed, stream[0]));
    for (int r = 1; r < kGpus; ++r) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaStreamWaitEvent(stream[r], seed, 0));
    }
    enqueue_round();
    for (int r = 1; r < kGpus; ++r) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaEventRecord(join[r], stream[r]));
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaStreamWaitEvent(stream[0], join[r], 0));
    }
    CHECK_CUDA(cudaSetDevice(0));
    cudaGraph_t graph = nullptr;
    rc = cudaStreamEndCapture(stream[0], &graph);
    if (rc != cudaSuccess || !graph) {
        std::printf("s598_nccl_capture_probe\tFAIL\tend_capture\t%s\n",
                    cudaGetErrorName(rc));
        return 5;
    }
    size_t nodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(graph, nullptr, &nodes));
    cudaGraphExec_t exec = nullptr;
    rc = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (rc != cudaSuccess || !exec) {
        std::printf("s598_nccl_capture_probe\tFAIL\tinstantiate\t%s\n",
                    cudaGetErrorName(rc));
        return 6;
    }
    std::printf("s598_probe_capture\tnodes\t%zu\tPASS\n", nodes);

    // ---- Replay 1: verify the captured round moves epoch-2 data ---------
    CHECK_CUDA(cudaGraphLaunch(exec, stream[0]));
    CHECK_CUDA(cudaStreamSynchronize(stream[0]));
    if (!verify(2.0f, "first_replay")) {
        std::printf("s598_nccl_capture_probe\tFAIL\tfirst_replay_parity\n");
        return 7;
    }

    // ---- Replay with fresh data: graph must re-read updated contribs ----
    fill_contribs(3.0f);
    CHECK_CUDA(cudaGraphLaunch(exec, stream[0]));
    CHECK_CUDA(cudaStreamSynchronize(stream[0]));
    if (!verify(3.0f, "fresh_data_replay")) {
        std::printf("s598_nccl_capture_probe\tFAIL\tfresh_data_parity\n");
        return 8;
    }

    // ---- Timed replay loop (50x) ----------------------------------------
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    const int iters = 50;
    CHECK_CUDA(cudaEventRecord(t0, stream[0]));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaGraphLaunch(exec, stream[0]));
    }
    CHECK_CUDA(cudaEventRecord(t1, stream[0]));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    std::printf("s598_probe_replay_timing\titers\t%d\tms_per_replay\t%.4f\t"
                "(8-src grouped broadcast round, 3 MiB per src)\n",
                iters, ms / iters);
    if (!verify(3.0f, "post_timing")) {
        std::printf("s598_nccl_capture_probe\tFAIL\tpost_timing_parity\n");
        return 9;
    }

    std::printf("s598_nccl_capture_probe\tnccl_version\t%d\tnodes\t%zu\t"
                "ms_per_replay\t%.4f\tPASS\n",
                nccl_version, nodes, ms / iters);
    for (int r = 0; r < kGpus; ++r) ncclCommDestroy(comms[r]);
    return 0;
}
