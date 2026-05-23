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
constexpr int kTopK = 6;

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
    if (i >= elems) return;

    float sum = 0.0f;
    for (int p = 0; p < participants; ++p) {
        sum += __half2float(staging[(size_t) p * elems + i]);
    }
    dst[i] = __float2half(sum);
}

__global__ void reduce_sum_half_offset_kernel(half * dst, const half * staging,
                                              int participants, size_t full_elems,
                                              size_t offset, size_t chunk_elems) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chunk_elems) return;

    float sum = 0.0f;
    for (int p = 0; p < participants; ++p) {
        sum += __half2float(staging[(size_t) p * full_elems + offset + i]);
    }
    dst[offset + i] = __float2half(sum);
}

__global__ void add_inplace_half_kernel(half * dst, const half * src, size_t elems) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elems) return;
    dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
}

enum class Mode {
    AllReduce,
    ReduceScatter,
    AllGather,
    ReduceScatterAllGather,
    ExpertReduce,
};

enum class Algorithm {
    Root,
    Doubling,
};

struct Options {
    int devices[kParticipants] = {0, 1, 2, 3, 4, 5, 6, 7};
    int tokens = 32;
    int hidden = 4096;
    int layers = 43;
    int collectives_per_layer = 1;
    int warmup = 3;
    int iters = 20;
    int root_index = 0;
    Mode mode = Mode::AllReduce;
    Algorithm algo = Algorithm::Doubling;
};

bool parse_int(const char * text, int * out) {
    if (text == nullptr || *text == '\0') return false;
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
        if (!parse_int(piece.c_str(), &dev)) return false;
        parsed.push_back(dev);
    }

    if ((int) parsed.size() != kParticipants) return false;
    for (int i = 0; i < kParticipants; ++i) {
        for (int j = i + 1; j < kParticipants; ++j) {
            if (parsed[i] == parsed[j]) return false;
        }
        devices[i] = parsed[i];
    }
    return true;
}

const char * mode_name(Mode mode) {
    switch (mode) {
    case Mode::AllReduce: return "allreduce";
    case Mode::ReduceScatter: return "reduce-scatter";
    case Mode::AllGather: return "allgather";
    case Mode::ReduceScatterAllGather: return "rs-ag";
    case Mode::ExpertReduce: return "ep-reduce";
    }
    return "unknown";
}

const char * algo_name(Algorithm algo) {
    switch (algo) {
    case Algorithm::Root: return "root";
    case Algorithm::Doubling: return "doubling";
    }
    return "unknown";
}

bool parse_mode(const char * text, Mode * out) {
    if (std::strcmp(text, "allreduce") == 0) *out = Mode::AllReduce;
    else if (std::strcmp(text, "reduce-scatter") == 0) *out = Mode::ReduceScatter;
    else if (std::strcmp(text, "allgather") == 0) *out = Mode::AllGather;
    else if (std::strcmp(text, "rs-ag") == 0) *out = Mode::ReduceScatterAllGather;
    else if (std::strcmp(text, "ep-reduce") == 0) *out = Mode::ExpertReduce;
    else return false;
    return true;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s [--mode allreduce|reduce-scatter|allgather|rs-ag|ep-reduce]\n"
                 "       [--algo root|doubling] [--devices 0,1,2,3,4,5,6,7]\n"
                 "       [--tokens N] [--hidden N] [--layers N]\n"
                 "       [--collectives-per-layer N] [--warmup N] [--iters N]\n",
                 argv0);
}

bool parse_args(int argc, char ** argv, Options * opt) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        const char * val = (i + 1 < argc) ? argv[i + 1] : nullptr;
        if (std::strcmp(arg, "--mode") == 0) {
            if (val == nullptr || !parse_mode(val, &opt->mode)) {
                std::fprintf(stderr, "invalid --mode value\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--algo") == 0) {
            if (val == nullptr) return false;
            if (std::strcmp(val, "root") == 0) opt->algo = Algorithm::Root;
            else if (std::strcmp(val, "doubling") == 0) opt->algo = Algorithm::Doubling;
            else {
                std::fprintf(stderr, "invalid --algo value\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--devices") == 0) {
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
        } else if (std::strcmp(arg, "--layers") == 0) {
            if (val == nullptr || !parse_int(val, &opt->layers) || opt->layers <= 0) {
                std::fprintf(stderr, "invalid --layers value\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--collectives-per-layer") == 0) {
            if (val == nullptr || !parse_int(val, &opt->collectives_per_layer) ||
                opt->collectives_per_layer <= 0) {
                std::fprintf(stderr, "invalid --collectives-per-layer value\n");
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
            if (val == nullptr || !parse_int(val, &opt->root_index) ||
                opt->root_index < 0 || opt->root_index >= kParticipants) {
                std::fprintf(stderr, "invalid --root-index value\n");
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
            if (i == j) continue;
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

void sync_streams(const Options & opt, cudaStream_t streams[kParticipants]) {
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaStreamSynchronize(streams[p]));
    }
}

void fill_full_input(std::vector<half> * h, int participant) {
    std::fill(h->begin(), h->end(), __float2half((float) (participant + 1)));
}

void fill_chunk_input(std::vector<half> * h, int participant) {
    std::fill(h->begin(), h->end(), __float2half((float) (participant + 1)));
}

float reduce_expected() {
    return (float) (kParticipants * (kParticipants + 1) / 2);
}

void run_root_allreduce(const Options & opt, half ** inputs, half ** outputs, half * staging,
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
        if (p == opt.root_index) continue;
        CUDA_CHECK(cudaMemcpyPeerAsync(outputs[p], opt.devices[p], outputs[opt.root_index],
                                       root_dev, bytes, root_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(root_stream));
}

void run_doubling_allreduce(const Options & opt, half ** inputs, half ** outputs, half ** recv,
                            cudaStream_t streams[kParticipants], size_t elems, size_t bytes) {
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

void run_allreduce(const Options & opt, half ** inputs, half ** outputs, half ** recv,
                   half * staging, cudaStream_t root_stream,
                   cudaStream_t streams[kParticipants], size_t elems, size_t bytes) {
    if (opt.algo == Algorithm::Doubling) {
        run_doubling_allreduce(opt, inputs, outputs, recv, streams, elems, bytes);
    } else {
        run_root_allreduce(opt, inputs, outputs, staging, root_stream, elems, bytes);
    }
}

void run_reduce_scatter_root(const Options & opt, half ** full_inputs, half ** chunk_outputs,
                             half * staging, half * root_full, cudaStream_t root_stream,
                             size_t full_elems, size_t chunk_elems, size_t full_bytes,
                             size_t chunk_bytes) {
    const int root_dev = opt.devices[opt.root_index];
    CUDA_CHECK(cudaSetDevice(root_dev));
    for (int p = 0; p < kParticipants; ++p) {
        half * dst = staging + (size_t) p * full_elems;
        if (p == opt.root_index) {
            CUDA_CHECK(cudaMemcpyAsync(dst, full_inputs[p], full_bytes,
                                       cudaMemcpyDeviceToDevice, root_stream));
        } else {
            CUDA_CHECK(cudaMemcpyPeerAsync(dst, root_dev, full_inputs[p], opt.devices[p],
                                           full_bytes, root_stream));
        }
    }

    const int block = 256;
    const int grid = (int) ((chunk_elems + block - 1) / block);
    for (int p = 0; p < kParticipants; ++p) {
        reduce_sum_half_offset_kernel<<<grid, block, 0, root_stream>>>(
            root_full, staging, kParticipants, full_elems, (size_t) p * chunk_elems,
            chunk_elems);
        CUDA_CHECK(cudaGetLastError());
    }

    for (int p = 0; p < kParticipants; ++p) {
        half * src = root_full + (size_t) p * chunk_elems;
        if (p == opt.root_index) {
            CUDA_CHECK(cudaMemcpyAsync(chunk_outputs[p], src, chunk_bytes,
                                       cudaMemcpyDeviceToDevice, root_stream));
        } else {
            CUDA_CHECK(cudaMemcpyPeerAsync(chunk_outputs[p], opt.devices[p], src, root_dev,
                                           chunk_bytes, root_stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(root_stream));
}

void run_allgather_direct(const Options & opt, half ** chunk_inputs, half ** full_outputs,
                          cudaStream_t streams[kParticipants], size_t chunk_elems,
                          size_t chunk_bytes) {
    for (int dst = 0; dst < kParticipants; ++dst) {
        CUDA_CHECK(cudaSetDevice(opt.devices[dst]));
        for (int src = 0; src < kParticipants; ++src) {
            half * dst_ptr = full_outputs[dst] + (size_t) src * chunk_elems;
            if (dst == src) {
                CUDA_CHECK(cudaMemcpyAsync(dst_ptr, chunk_inputs[src], chunk_bytes,
                                           cudaMemcpyDeviceToDevice, streams[dst]));
            } else {
                CUDA_CHECK(cudaMemcpyPeerAsync(dst_ptr, opt.devices[dst], chunk_inputs[src],
                                               opt.devices[src], chunk_bytes, streams[dst]));
            }
        }
    }
    sync_streams(opt, streams);
}

void run_once(const Options & opt, half ** full_inputs, half ** full_outputs, half ** recv,
              half ** chunk_inputs, half ** chunk_outputs, half * staging, half * root_full,
              cudaStream_t root_stream, cudaStream_t streams[kParticipants],
              size_t full_elems, size_t chunk_elems, size_t full_bytes, size_t chunk_bytes) {
    switch (opt.mode) {
    case Mode::AllReduce:
    case Mode::ExpertReduce:
        run_allreduce(opt, full_inputs, full_outputs, recv, staging, root_stream, streams,
                      full_elems, full_bytes);
        break;
    case Mode::ReduceScatter:
        run_reduce_scatter_root(opt, full_inputs, chunk_outputs, staging, root_full,
                                root_stream, full_elems, chunk_elems, full_bytes,
                                chunk_bytes);
        break;
    case Mode::AllGather:
        run_allgather_direct(opt, chunk_inputs, full_outputs, streams, chunk_elems,
                             chunk_bytes);
        break;
    case Mode::ReduceScatterAllGather:
        run_reduce_scatter_root(opt, full_inputs, chunk_outputs, staging, root_full,
                                root_stream, full_elems, chunk_elems, full_bytes,
                                chunk_bytes);
        run_allgather_direct(opt, chunk_outputs, full_outputs, streams, chunk_elems,
                             chunk_bytes);
        break;
    }
}

float verify_full_reduce(const Options & opt, half ** outputs, size_t elems, size_t bytes) {
    std::vector<half> h(elems);
    const float expected = reduce_expected();
    float max_abs = 0.0f;
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaMemcpy(h.data(), outputs[p], bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < elems; ++i) {
            const float got = __half2float(h[i]);
            if (!std::isfinite(got)) return std::numeric_limits<float>::infinity();
            max_abs = std::max(max_abs, std::fabs(got - expected));
        }
    }
    return max_abs;
}

float verify_reduce_scatter(const Options & opt, half ** outputs, size_t elems, size_t bytes) {
    std::vector<half> h(elems);
    const float expected = reduce_expected();
    float max_abs = 0.0f;
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaMemcpy(h.data(), outputs[p], bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < elems; ++i) {
            const float got = __half2float(h[i]);
            if (!std::isfinite(got)) return std::numeric_limits<float>::infinity();
            max_abs = std::max(max_abs, std::fabs(got - expected));
        }
    }
    return max_abs;
}

float verify_allgather(const Options & opt, half ** outputs, size_t chunk_elems,
                       size_t full_bytes) {
    const size_t full_elems = chunk_elems * kParticipants;
    std::vector<half> h(full_elems);
    float max_abs = 0.0f;
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaMemcpy(h.data(), outputs[p], full_bytes, cudaMemcpyDeviceToHost));
        for (int src = 0; src < kParticipants; ++src) {
            const float expected = (float) (src + 1);
            for (size_t i = 0; i < chunk_elems; ++i) {
                const float got = __half2float(h[(size_t) src * chunk_elems + i]);
                if (!std::isfinite(got)) return std::numeric_limits<float>::infinity();
                max_abs = std::max(max_abs, std::fabs(got - expected));
            }
        }
    }
    return max_abs;
}

float verify_outputs(const Options & opt, half ** full_outputs, half ** chunk_outputs,
                     size_t full_elems, size_t chunk_elems, size_t full_bytes,
                     size_t chunk_bytes) {
    switch (opt.mode) {
    case Mode::AllReduce:
    case Mode::ExpertReduce:
    case Mode::ReduceScatterAllGather:
        return verify_full_reduce(opt, full_outputs, full_elems, full_bytes);
    case Mode::ReduceScatter:
        return verify_reduce_scatter(opt, chunk_outputs, chunk_elems, chunk_bytes);
    case Mode::AllGather:
        return verify_allgather(opt, full_outputs, chunk_elems, full_bytes);
    }
    return std::numeric_limits<float>::infinity();
}

double wire_bytes_per_collective(const Options & opt, size_t full_bytes, size_t chunk_bytes) {
    switch (opt.mode) {
    case Mode::AllReduce:
    case Mode::ExpertReduce:
        if (opt.algo == Algorithm::Doubling) return (double) full_bytes * kParticipants * 2.0;
        return (double) full_bytes * (kParticipants - 1) * 2.0;
    case Mode::ReduceScatter:
        return (double) full_bytes * (kParticipants - 1) +
               (double) chunk_bytes * (kParticipants - 1);
    case Mode::AllGather:
        return (double) full_bytes * (kParticipants - 1);
    case Mode::ReduceScatterAllGather:
        return (double) full_bytes * (kParticipants - 1) +
               (double) chunk_bytes * (kParticipants - 1) +
               (double) full_bytes * (kParticipants - 1);
    }
    return 0.0;
}

} // namespace

int main(int argc, char ** argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        usage(argv[0]);
        return 2;
    }

    enable_peer_access_or_die(opt);

    const size_t full_elems = (size_t) opt.tokens * (size_t) opt.hidden;
    if (full_elems % kParticipants != 0) {
        std::fprintf(stderr, "tokens*hidden must be divisible by %d for shard modes\n",
                     kParticipants);
        return 2;
    }
    const size_t chunk_elems = full_elems / kParticipants;
    const size_t full_bytes = full_elems * sizeof(half);
    const size_t chunk_bytes = chunk_elems * sizeof(half);
    const int root_dev = opt.devices[opt.root_index];

    half * full_inputs[kParticipants] = {};
    half * full_outputs[kParticipants] = {};
    half * recv[kParticipants] = {};
    half * chunk_inputs[kParticipants] = {};
    half * chunk_outputs[kParticipants] = {};
    half * staging = nullptr;
    half * root_full = nullptr;

    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaMalloc(&full_inputs[p], full_bytes));
        CUDA_CHECK(cudaMalloc(&full_outputs[p], full_bytes));
        CUDA_CHECK(cudaMalloc(&recv[p], full_bytes));
        CUDA_CHECK(cudaMalloc(&chunk_inputs[p], chunk_bytes));
        CUDA_CHECK(cudaMalloc(&chunk_outputs[p], chunk_bytes));

        std::vector<half> full(full_elems);
        std::vector<half> chunk(chunk_elems);
        fill_full_input(&full, p);
        fill_chunk_input(&chunk, p);
        CUDA_CHECK(cudaMemcpy(full_inputs[p], full.data(), full_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(chunk_inputs[p], chunk.data(), chunk_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(full_outputs[p], 0, full_bytes));
        CUDA_CHECK(cudaMemset(recv[p], 0, full_bytes));
        CUDA_CHECK(cudaMemset(chunk_outputs[p], 0, chunk_bytes));
    }

    CUDA_CHECK(cudaSetDevice(root_dev));
    CUDA_CHECK(cudaMalloc(&staging, full_bytes * kParticipants));
    CUDA_CHECK(cudaMalloc(&root_full, full_bytes));
    cudaStream_t root_stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&root_stream));
    cudaStream_t streams[kParticipants] = {};
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaStreamCreate(&streams[p]));
    }

    const int ops_per_iter = opt.layers * opt.collectives_per_layer;
    for (int i = 0; i < opt.warmup; ++i) {
        for (int op = 0; op < ops_per_iter; ++op) {
            run_once(opt, full_inputs, full_outputs, recv, chunk_inputs, chunk_outputs,
                     staging, root_full, root_stream, streams, full_elems, chunk_elems,
                     full_bytes, chunk_bytes);
        }
    }

    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    for (int i = 0; i < opt.iters; ++i) {
        auto start = std::chrono::steady_clock::now();
        for (int op = 0; op < ops_per_iter; ++op) {
            run_once(opt, full_inputs, full_outputs, recv, chunk_inputs, chunk_outputs,
                     staging, root_full, root_stream, streams, full_elems, chunk_elems,
                     full_bytes, chunk_bytes);
        }
        auto stop = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }

    const float max_abs = verify_outputs(opt, full_outputs, chunk_outputs, full_elems,
                                         chunk_elems, full_bytes, chunk_bytes);
    const double avg_ms = total_ms / (double) opt.iters;
    const double per_layer_ms = avg_ms / (double) opt.layers;
    const double per_collective_ms = avg_ms / (double) ops_per_iter;
    const double wire_bytes = wire_bytes_per_collective(opt, full_bytes, chunk_bytes) *
                              (double) ops_per_iter;
    const double effective_gbps = wire_bytes / (avg_ms / 1000.0) / 1.0e9;
    const double overhead_only_tok_s = (double) opt.tokens / (avg_ms / 1000.0);
    const double routes = (double) opt.tokens * (double) kTopK;
    const double routes_per_gpu = routes / (double) kParticipants;

    std::printf("ds4-v100-tp8-collective-workbench mode=%s algo=%s devices=",
                mode_name(opt.mode), algo_name(opt.algo));
    for (int p = 0; p < kParticipants; ++p) {
        std::printf("%s%d", p ? "," : "", opt.devices[p]);
    }
    std::printf(" root=%d tokens=%d hidden=%d dtype=f16 full_bytes=%zu chunk_bytes=%zu "
                "layers=%d collectives_per_layer=%d ops_per_iter=%d warmup=%d iters=%d\n",
                root_dev, opt.tokens, opt.hidden, full_bytes, chunk_bytes, opt.layers,
                opt.collectives_per_layer, ops_per_iter, opt.warmup, opt.iters);
    std::printf("latency_ms avg=%.6f min=%.6f max=%.6f per_layer=%.6f "
                "per_collective=%.6f effective_wire_gbps=%.3f "
                "overhead_only_tok_s=%.3f\n",
                avg_ms, min_ms, max_ms, per_layer_ms, per_collective_ms,
                effective_gbps, overhead_only_tok_s);
    std::printf("ep_shape top_k=%d active_routes=%.0f routes_per_gpu=%.3f\n",
                kTopK, routes, routes_per_gpu);
    std::printf("verify max_abs=%.9f %s\n", max_abs,
                max_abs <= 1.0e-5f ? "ok" : "FAIL");

    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaStreamDestroy(streams[p]));
    }
    CUDA_CHECK(cudaStreamDestroy(root_stream));
    CUDA_CHECK(cudaFree(staging));
    CUDA_CHECK(cudaFree(root_full));
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaFree(full_inputs[p]));
        CUDA_CHECK(cudaFree(full_outputs[p]));
        CUDA_CHECK(cudaFree(recv[p]));
        CUDA_CHECK(cudaFree(chunk_inputs[p]));
        CUDA_CHECK(cudaFree(chunk_outputs[p]));
    }

    return max_abs <= 1.0e-5f ? 0 : 1;
}
