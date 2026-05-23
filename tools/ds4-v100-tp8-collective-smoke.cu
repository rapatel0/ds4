#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kParticipants = 8;

#define CUDA_CHECK(expr)                                                                          \
    do {                                                                                          \
        cudaError_t err__ = (expr);                                                               \
        if (err__ != cudaSuccess) {                                                               \
            std::fprintf(stderr, "cuda error %s:%d: %s\n", __FILE__, __LINE__,                  \
                         cudaGetErrorString(err__));                                              \
            std::exit(2);                                                                         \
        }                                                                                         \
    } while (0)

__global__ void reduce_sum_half_kernel(half * dst, const half * staging, int participants,
                                       size_t elems) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elems) {
        return;
    }

    float sum = 0.0f;
    for (int p = 0; p < participants; ++p) {
        sum += __half2float(staging[(size_t) p * elems + i]);
    }
    dst[i] = __float2half(sum);
}

__global__ void add_inplace_half_kernel(half * dst, const half * src, size_t elems) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elems) {
        return;
    }
    dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
}

enum class Algorithm {
    Root,
    Doubling,
};

struct Options {
    int devices[kParticipants] = {0, 1, 2, 3, 4, 5, 6, 7};
    int tokens = 32;
    int hidden = 4096;
    int warmup = 10;
    int iters = 100;
    int root_index = 0;
    Algorithm algo = Algorithm::Doubling;
};

bool parse_int(const char * text, int * out) {
    if (text == nullptr || *text == '\0') {
        return false;
    }

    errno = 0;
    char * end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || v < 0 ||
        v > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = (int) v;
    return true;
}

bool parse_devices(const char * text, int devices[kParticipants]) {
    std::vector<int> parsed;
    const char * cur = text;
    while (cur != nullptr && *cur != '\0') {
        const char * comma = std::strchr(cur, ',');
        std::string piece;
        if (comma != nullptr) {
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
                 "usage: %s [--devices 0,1,2,3,4,5,6,7] [--tokens N] [--hidden N]\n"
                 "       [--warmup N] [--iters N] [--root-index 0..7]\n"
                 "       [--algo root|doubling]\n",
                 argv0);
}

bool parse_args(int argc, char ** argv, Options * opt) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        const char * val = (i + 1 < argc) ? argv[i + 1] : nullptr;
        if (std::strcmp(arg, "--devices") == 0) {
            if (val == nullptr || !parse_devices(val, opt->devices)) {
                std::fprintf(stderr, "invalid --devices value; expected eight unique ids\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--tokens") == 0) {
            if (val == nullptr || !parse_int(val, &opt->tokens) || opt->tokens <= 0) {
                std::fprintf(stderr, "invalid --tokens value\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--hidden") == 0) {
            if (val == nullptr || !parse_int(val, &opt->hidden) || opt->hidden <= 0) {
                std::fprintf(stderr, "invalid --hidden value\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--warmup") == 0) {
            if (val == nullptr || !parse_int(val, &opt->warmup)) {
                std::fprintf(stderr, "invalid --warmup value\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--iters") == 0) {
            if (val == nullptr || !parse_int(val, &opt->iters) || opt->iters <= 0) {
                std::fprintf(stderr, "invalid --iters value\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--root-index") == 0) {
            if (val == nullptr || !parse_int(val, &opt->root_index) || opt->root_index < 0 ||
                opt->root_index >= kParticipants) {
                std::fprintf(stderr, "invalid --root-index value\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--algo") == 0) {
            if (val == nullptr) {
                std::fprintf(stderr, "invalid --algo value\n");
                return false;
            }
            if (std::strcmp(val, "root") == 0) {
                opt->algo = Algorithm::Root;
            } else if (std::strcmp(val, "doubling") == 0) {
                opt->algo = Algorithm::Doubling;
            } else {
                std::fprintf(stderr, "invalid --algo value; expected root or doubling\n");
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

void enable_peer_access_or_die(const Options & opt) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    for (int i = 0; i < kParticipants; ++i) {
        if (opt.devices[i] < 0 || opt.devices[i] >= device_count) {
            std::fprintf(stderr, "device %d is outside visible device count %d\n",
                         opt.devices[i], device_count);
            std::exit(2);
        }
    }

    for (int i = 0; i < kParticipants; ++i) {
        CUDA_CHECK(cudaSetDevice(opt.devices[i]));
        for (int j = 0; j < kParticipants; ++j) {
            if (i == j) {
                continue;
            }
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, opt.devices[i], opt.devices[j]));
            if (!can_access) {
                std::fprintf(stderr, "device %d cannot peer-access device %d\n",
                             opt.devices[i], opt.devices[j]);
                std::exit(2);
            }
            cudaError_t err = cudaDeviceEnablePeerAccess(opt.devices[j], 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                (void) cudaGetLastError();
            } else if (err != cudaSuccess) {
                std::fprintf(stderr, "failed to enable peer access %d -> %d: %s\n",
                             opt.devices[i], opt.devices[j], cudaGetErrorString(err));
                std::exit(2);
            }
        }
    }
}

const char * algo_name(Algorithm algo) {
    switch (algo) {
    case Algorithm::Root: return "root";
    case Algorithm::Doubling: return "doubling";
    }
    return "unknown";
}

void fill_host_input(std::vector<half> * h, int participant) {
    const half value = __float2half((float) (participant + 1));
    std::fill(h->begin(), h->end(), value);
}

float expected_value() {
    return (float) (kParticipants * (kParticipants + 1) / 2);
}

void sync_streams(const Options & opt, cudaStream_t streams[kParticipants]) {
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaStreamSynchronize(streams[p]));
    }
}

void run_root_collective(const Options & opt, half ** inputs, half ** outputs, half * staging,
                         cudaStream_t root_stream, size_t elems, size_t bytes) {
    const int root_dev = opt.devices[opt.root_index];

    CUDA_CHECK(cudaSetDevice(root_dev));
    for (int p = 0; p < kParticipants; ++p) {
        half * dst = staging + (size_t) p * elems;
        if (p == opt.root_index) {
            CUDA_CHECK(cudaMemcpyAsync(dst, inputs[p], bytes, cudaMemcpyDeviceToDevice,
                                       root_stream));
        } else {
            CUDA_CHECK(cudaMemcpyPeerAsync(dst, root_dev, inputs[p], opt.devices[p], bytes,
                                           root_stream));
        }
    }

    const int block = 256;
    const int grid = (int) ((elems + block - 1) / block);
    reduce_sum_half_kernel<<<grid, block, 0, root_stream>>>(outputs[opt.root_index], staging,
                                                            kParticipants, elems);
    CUDA_CHECK(cudaGetLastError());

    for (int p = 0; p < kParticipants; ++p) {
        if (p == opt.root_index) {
            continue;
        }
        CUDA_CHECK(cudaMemcpyPeerAsync(outputs[p], opt.devices[p], outputs[opt.root_index],
                                       root_dev, bytes, root_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(root_stream));
}

void run_doubling_collective(const Options & opt, half ** inputs, half ** outputs,
                             half ** recv, cudaStream_t streams[kParticipants],
                             size_t elems, size_t bytes) {
    const int block = 256;
    const int grid = (int) ((elems + block - 1) / block);

    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaMemcpyAsync(outputs[p], inputs[p], bytes, cudaMemcpyDeviceToDevice,
                                   streams[p]));
    }
    sync_streams(opt, streams);

    for (int step = 1; step < kParticipants; step <<= 1) {
        for (int p = 0; p < kParticipants; ++p) {
            const int peer = p ^ step;
            CUDA_CHECK(cudaSetDevice(opt.devices[peer]));
            CUDA_CHECK(cudaMemcpyPeerAsync(recv[peer], opt.devices[peer], outputs[p],
                                           opt.devices[p], bytes, streams[peer]));
        }
        sync_streams(opt, streams);

        for (int p = 0; p < kParticipants; ++p) {
            CUDA_CHECK(cudaSetDevice(opt.devices[p]));
            add_inplace_half_kernel<<<grid, block, 0, streams[p]>>>(outputs[p], recv[p], elems);
            CUDA_CHECK(cudaGetLastError());
        }
        sync_streams(opt, streams);
    }
}

void run_selected_collective(const Options & opt, half ** inputs, half ** outputs,
                             half * staging, half ** recv, cudaStream_t root_stream,
                             cudaStream_t streams[kParticipants], size_t elems,
                             size_t bytes) {
    if (opt.algo == Algorithm::Doubling) {
        run_doubling_collective(opt, inputs, outputs, recv, streams, elems, bytes);
    } else {
        run_root_collective(opt, inputs, outputs, staging, root_stream, elems, bytes);
    }
}

float verify_outputs(const Options & opt, half ** outputs, size_t elems, size_t bytes) {
    std::vector<half> h(elems);
    const float expected = expected_value();
    float max_abs = 0.0f;

    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaMemcpy(h.data(), outputs[p], bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < elems; ++i) {
            const float diff = std::fabs(__half2float(h[i]) - expected);
            max_abs = std::max(max_abs, diff);
        }
    }

    return max_abs;
}

} // namespace

int main(int argc, char ** argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        usage(argv[0]);
        return 2;
    }

    enable_peer_access_or_die(opt);

    const size_t elems = (size_t) opt.tokens * (size_t) opt.hidden;
    const size_t bytes = elems * sizeof(half);
    const int root_dev = opt.devices[opt.root_index];

    half * inputs[kParticipants] = {};
    half * outputs[kParticipants] = {};
    half * recv[kParticipants] = {};
    half * staging = nullptr;

    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaMalloc(&inputs[p], bytes));
        CUDA_CHECK(cudaMalloc(&outputs[p], bytes));
        CUDA_CHECK(cudaMalloc(&recv[p], bytes));

        std::vector<half> h(elems);
        fill_host_input(&h, p);
        CUDA_CHECK(cudaMemcpy(inputs[p], h.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(outputs[p], 0, bytes));
        CUDA_CHECK(cudaMemset(recv[p], 0, bytes));
    }

    CUDA_CHECK(cudaSetDevice(root_dev));
    CUDA_CHECK(cudaMalloc(&staging, bytes * kParticipants));
    cudaStream_t root_stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&root_stream));
    cudaStream_t streams[kParticipants] = {};
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaStreamCreate(&streams[p]));
    }

    for (int i = 0; i < opt.warmup; ++i) {
        run_selected_collective(opt, inputs, outputs, staging, recv, root_stream, streams,
                                elems, bytes);
    }

    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;

    for (int i = 0; i < opt.iters; ++i) {
        auto start = std::chrono::steady_clock::now();
        run_selected_collective(opt, inputs, outputs, staging, recv, root_stream, streams,
                                elems, bytes);
        auto stop = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }

    const float max_abs = verify_outputs(opt, outputs, elems, bytes);
    const double avg_ms = total_ms / (double) opt.iters;
    const double wire_factor = opt.algo == Algorithm::Doubling
                                   ? (double) kParticipants * 2.0
                                   : (double) (kParticipants - 1) * 2.0;
    const double wire_bytes = (double) bytes * wire_factor;
    const double effective_gbps = wire_bytes / (avg_ms / 1000.0) / 1.0e9;

    std::printf("ds4-v100-tp8-collective-smoke algo=%s devices=", algo_name(opt.algo));
    for (int p = 0; p < kParticipants; ++p) {
        std::printf("%s%d", p ? "," : "", opt.devices[p]);
    }
    std::printf(" root=%d tokens=%d hidden=%d dtype=f16 bytes_per_tensor=%zu "
                "warmup=%d iters=%d\n",
                root_dev, opt.tokens, opt.hidden, bytes, opt.warmup, opt.iters);
    std::printf("latency_ms avg=%.6f min=%.6f max=%.6f effective_wire_gbps=%.3f\n",
                avg_ms, min_ms, max_ms, effective_gbps);
    std::printf("verify max_abs=%.9f %s\n", max_abs, max_abs <= 1.0e-5f ? "ok" : "FAIL");

    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaStreamDestroy(streams[p]));
    }
    CUDA_CHECK(cudaStreamDestroy(root_stream));
    CUDA_CHECK(cudaFree(staging));
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaFree(inputs[p]));
        CUDA_CHECK(cudaFree(outputs[p]));
        CUDA_CHECK(cudaFree(recv[p]));
    }

    return max_abs <= 1.0e-5f ? 0 : 1;
}
