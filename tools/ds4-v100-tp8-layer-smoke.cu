#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

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
constexpr int kSwaRows = 128;
constexpr int kHeadDim = 512;
constexpr int kIndexerHeadDim = 128;

#define CUDA_CHECK(expr)                                                                          \
    do {                                                                                          \
        cudaError_t err__ = (expr);                                                               \
        if (err__ != cudaSuccess) {                                                               \
            std::fprintf(stderr, "cuda error %s:%d: %s\n", __FILE__, __LINE__,                  \
                         cudaGetErrorString(err__));                                              \
            std::exit(2);                                                                         \
        }                                                                                         \
    } while (0)

#define NCCL_CHECK(expr)                                                                          \
    do {                                                                                          \
        ncclResult_t err__ = (expr);                                                              \
        if (err__ != ncclSuccess) {                                                               \
            std::fprintf(stderr, "nccl error %s:%d: %s\n", __FILE__, __LINE__,                  \
                         ncclGetErrorString(err__));                                              \
            std::exit(2);                                                                         \
        }                                                                                         \
    } while (0)

enum class Algorithm {
    Root,
    Doubling,
    Nccl,
};

enum class KvDType {
    F16,
    F8E4M3B128,
    Q8_0,
};

struct Options {
    int devices[kParticipants] = {0, 1, 2, 3, 4, 5, 6, 7};
    int tokens = 32;
    int hidden = 4096;
    int ctx = 262144;
    int slots = 32;
    int ratio = 4;
    int compute_repeats = 64;
    int warmup = 3;
    int iters = 20;
    int root_index = 0;
    Algorithm algo = Algorithm::Nccl;
    bool allow_manual_peer_baseline = false;
    KvDType kv_dtype = KvDType::F8E4M3B128;
};

struct KvPlan {
    size_t rows;
    size_t attn_values;
    size_t indexer_values;
    size_t per_slot_bytes;
    size_t logical_bytes;
    size_t shard_bytes;
};

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

__global__ void layer_like_compute_kernel(half * dst, const half * src,
                                          const unsigned char * kv_shard,
                                          size_t kv_shard_bytes, size_t elems,
                                          int repeats, float participant_bias) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elems) {
        return;
    }

    const volatile unsigned char * kv = kv_shard;
    const size_t kv_i = kv_shard_bytes ? (i * 1315423911ULL) % kv_shard_bytes : 0;
    const float kv_term = kv_shard_bytes ? ((float) (kv[kv_i] & 7)) * 0.00001f : 0.0f;
    float v = __half2float(src[i]) * 0.125f + participant_bias + kv_term;
    for (int r = 0; r < repeats; ++r) {
        v = fmaf(v, 1.0000001f, 0.0000001f);
    }
    dst[i] = __float2half(v);
}

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

size_t ceil_div_size(size_t a, size_t b) {
    return (a + b - 1) / b;
}

size_t checked_mul_size(size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        std::fprintf(stderr, "size overflow\n");
        std::exit(2);
    }
    return a * b;
}

size_t bytes_blocks(size_t elems, size_t block_elems, size_t block_bytes) {
    return checked_mul_size(ceil_div_size(elems, block_elems), block_bytes);
}

size_t values_bytes(size_t values, KvDType dtype) {
    switch (dtype) {
    case KvDType::F16:
        return checked_mul_size(values, 2);
    case KvDType::F8E4M3B128:
        return bytes_blocks(values, 128, 129);
    case KvDType::Q8_0:
        return bytes_blocks(values, 32, 34);
    }
    return 0;
}

KvPlan make_kv_plan(const Options & opt) {
    if (opt.ratio != 4 && opt.ratio != 128) {
        std::fprintf(stderr, "--ratio must be 4 or 128\n");
        std::exit(2);
    }

    KvPlan plan = {};
    plan.rows = (size_t) kSwaRows + (size_t) opt.ctx / (size_t) opt.ratio;
    plan.attn_values = checked_mul_size(plan.rows, kHeadDim);
    plan.indexer_values =
        opt.ratio == 4 ? checked_mul_size((size_t) opt.ctx / 4u, kIndexerHeadDim) : 0;
    plan.per_slot_bytes =
        values_bytes(plan.attn_values, opt.kv_dtype) + values_bytes(plan.indexer_values,
                                                                    opt.kv_dtype);
    plan.logical_bytes = checked_mul_size(plan.per_slot_bytes, (size_t) opt.slots);
    plan.shard_bytes = ceil_div_size(plan.logical_bytes, kParticipants);
    return plan;
}

const char * algo_name(Algorithm algo) {
    switch (algo) {
    case Algorithm::Root: return "root";
    case Algorithm::Doubling: return "doubling";
    case Algorithm::Nccl: return "nccl";
    }
    return "unknown";
}

const char * kv_dtype_name(KvDType dtype) {
    switch (dtype) {
    case KvDType::F16: return "f16";
    case KvDType::F8E4M3B128: return "f8_e4m3_b128";
    case KvDType::Q8_0: return "q8_0";
    }
    return "unknown";
}

void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s [--devices 0,1,2,3,4,5,6,7] [--tokens N]\n"
                 "       [--hidden N] [--ctx N] [--slots N] [--ratio 4|128]\n"
                 "       [--kv-dtype f8|q8|f16] [--compute-repeats N]\n"
                 "       [--warmup N] [--iters N] [--root-index 0..7]\n"
                 "       [--algo root|doubling|nccl] [--allow-manual-peer-baseline]\n",
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
        } else if (std::strcmp(arg, "--ctx") == 0) {
            if (val == nullptr || !parse_int(val, &opt->ctx) || opt->ctx <= 0) {
                std::fprintf(stderr, "invalid --ctx value\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--slots") == 0) {
            if (val == nullptr || !parse_int(val, &opt->slots) || opt->slots <= 0) {
                std::fprintf(stderr, "invalid --slots value\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--ratio") == 0) {
            if (val == nullptr || !parse_int(val, &opt->ratio) ||
                (opt->ratio != 4 && opt->ratio != 128)) {
                std::fprintf(stderr, "invalid --ratio value; expected 4 or 128\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--compute-repeats") == 0) {
            if (val == nullptr || !parse_int(val, &opt->compute_repeats)) {
                std::fprintf(stderr, "invalid --compute-repeats value\n");
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
            } else if (std::strcmp(val, "nccl") == 0) {
                opt->algo = Algorithm::Nccl;
            } else {
                std::fprintf(stderr,
                             "invalid --algo value; expected root, doubling, or nccl\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--allow-manual-peer-baseline") == 0) {
            opt->allow_manual_peer_baseline = true;
        } else if (std::strcmp(arg, "--kv-dtype") == 0) {
            if (val == nullptr) {
                std::fprintf(stderr, "invalid --kv-dtype value\n");
                return false;
            }
            if (std::strcmp(val, "f8") == 0 || std::strcmp(val, "f8_e4m3_b128") == 0) {
                opt->kv_dtype = KvDType::F8E4M3B128;
            } else if (std::strcmp(val, "q8") == 0 || std::strcmp(val, "q8_0") == 0) {
                opt->kv_dtype = KvDType::Q8_0;
            } else if (std::strcmp(val, "f16") == 0) {
                opt->kv_dtype = KvDType::F16;
            } else {
                std::fprintf(stderr, "invalid --kv-dtype value; expected f8, q8, or f16\n");
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

void fill_host_input(std::vector<half> * h, int participant) {
    const half value = __float2half((float) (participant + 1));
    std::fill(h->begin(), h->end(), value);
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

void run_nccl_allreduce(const Options & opt, half ** inputs, half ** outputs,
                        cudaStream_t streams[kParticipants], ncclComm_t comms[kParticipants],
                        size_t elems) {
    NCCL_CHECK(ncclGroupStart());
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        NCCL_CHECK(ncclAllReduce(inputs[p], outputs[p], elems, ncclHalf, ncclSum,
                                 comms[p], streams[p]));
    }
    NCCL_CHECK(ncclGroupEnd());
    sync_streams(opt, streams);
}

void run_selected_collective(const Options & opt, half ** inputs, half ** outputs,
                             half * staging, half ** recv, cudaStream_t root_stream,
                             cudaStream_t streams[kParticipants], size_t elems,
                             size_t bytes, ncclComm_t comms[kParticipants]) {
    if (opt.algo == Algorithm::Nccl) {
        run_nccl_allreduce(opt, inputs, outputs, streams, comms, elems);
    } else if (opt.algo == Algorithm::Doubling) {
        run_doubling_collective(opt, inputs, outputs, recv, streams, elems, bytes);
    } else {
        run_root_collective(opt, inputs, outputs, staging, root_stream, elems, bytes);
    }
}

void run_compute(const Options & opt, half ** inputs, half ** outputs,
                 unsigned char ** kv_shards, const KvPlan & kv_plan,
                 cudaStream_t streams[kParticipants], size_t elems) {
    const int block = 256;
    const int grid = (int) ((elems + block - 1) / block);
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        const float bias = (float) (p + 1) * 0.0005f;
        layer_like_compute_kernel<<<grid, block, 0, streams[p]>>>(
            outputs[p], inputs[p], kv_shards[p], kv_plan.shard_bytes, elems,
            opt.compute_repeats, bias);
        CUDA_CHECK(cudaGetLastError());
    }
    sync_streams(opt, streams);
}

struct Timings {
    double compute_ms;
    double reduce_ms;
    double total_ms;
    half ** final_buf;
};

Timings run_one_layer(const Options & opt, half ** buf0, half ** buf1, half ** recv,
                      unsigned char ** kv_shards, const KvPlan & kv_plan, half * staging,
                      cudaStream_t root_stream, cudaStream_t streams[kParticipants],
                      ncclComm_t comms[kParticipants], size_t elems, size_t bytes,
                      bool timed) {
    Timings t = {};
    half ** cur = buf0;
    half ** next = buf1;

    const auto total_start = std::chrono::steady_clock::now();
    for (int phase = 0; phase < 2; ++phase) {
        const auto compute_start = std::chrono::steady_clock::now();
        run_compute(opt, cur, next, kv_shards, kv_plan, streams, elems);
        const auto compute_stop = std::chrono::steady_clock::now();

        const auto reduce_start = std::chrono::steady_clock::now();
        run_selected_collective(opt, next, cur, staging, recv, root_stream, streams, elems,
                                bytes, comms);
        const auto reduce_stop = std::chrono::steady_clock::now();

        if (timed) {
            t.compute_ms +=
                std::chrono::duration<double, std::milli>(compute_stop - compute_start).count();
            t.reduce_ms +=
                std::chrono::duration<double, std::milli>(reduce_stop - reduce_start).count();
        }
    }
    const auto total_stop = std::chrono::steady_clock::now();
    if (timed) {
        t.total_ms = std::chrono::duration<double, std::milli>(total_stop - total_start).count();
    }
    t.final_buf = cur;
    return t;
}

float verify_outputs(const Options & opt, half ** outputs, size_t elems, size_t bytes) {
    std::vector<half> ref(elems);
    std::vector<half> h(elems);
    float max_abs = 0.0f;

    CUDA_CHECK(cudaSetDevice(opt.devices[0]));
    CUDA_CHECK(cudaMemcpy(ref.data(), outputs[0], bytes, cudaMemcpyDeviceToHost));
    for (int p = 1; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaMemcpy(h.data(), outputs[p], bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < elems; ++i) {
            const float a = __half2float(ref[i]);
            const float b = __half2float(h[i]);
            if (!std::isfinite(a) || !std::isfinite(b)) {
                return std::numeric_limits<float>::infinity();
            }
            max_abs = std::max(max_abs, std::fabs(a - b));
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
    if (opt.algo != Algorithm::Nccl && !opt.allow_manual_peer_baseline) {
        std::fprintf(stderr,
                     "manual peer-copy baseline algorithms require "
                     "--allow-manual-peer-baseline; default or use --algo nccl for "
                     "promotion evidence\n");
        return 2;
    }

    enable_peer_access_or_die(opt);
    const KvPlan kv_plan = make_kv_plan(opt);

    const size_t elems = (size_t) opt.tokens * (size_t) opt.hidden;
    const size_t bytes = elems * sizeof(half);
    const int root_dev = opt.devices[opt.root_index];

    half * buf0[kParticipants] = {};
    half * buf1[kParticipants] = {};
    half * recv[kParticipants] = {};
    unsigned char * kv_shards[kParticipants] = {};
    half * staging = nullptr;

    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaMalloc(&buf0[p], bytes));
        CUDA_CHECK(cudaMalloc(&buf1[p], bytes));
        CUDA_CHECK(cudaMalloc(&recv[p], bytes));
        CUDA_CHECK(cudaMalloc(&kv_shards[p], kv_plan.shard_bytes));

        std::vector<half> h(elems);
        fill_host_input(&h, p);
        CUDA_CHECK(cudaMemcpy(buf0[p], h.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(buf1[p], 0, bytes));
        CUDA_CHECK(cudaMemset(recv[p], 0, bytes));
        CUDA_CHECK(cudaMemset(kv_shards[p], 17 + p, kv_plan.shard_bytes));
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
    ncclComm_t comms[kParticipants] = {};
    if (opt.algo == Algorithm::Nccl) {
        NCCL_CHECK(ncclCommInitAll(comms, kParticipants, opt.devices));
    }

    half ** final_buf = nullptr;
    for (int i = 0; i < opt.warmup; ++i) {
        Timings t = run_one_layer(opt, buf0, buf1, recv, kv_shards, kv_plan, staging,
                                  root_stream, streams, comms, elems, bytes, false);
        final_buf = t.final_buf;
    }

    double total_ms = 0.0;
    double compute_ms = 0.0;
    double reduce_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;

    for (int i = 0; i < opt.iters; ++i) {
        Timings t = run_one_layer(opt, buf0, buf1, recv, kv_shards, kv_plan, staging,
                                  root_stream, streams, comms, elems, bytes, true);
        final_buf = t.final_buf;
        total_ms += t.total_ms;
        compute_ms += t.compute_ms;
        reduce_ms += t.reduce_ms;
        min_ms = std::min(min_ms, t.total_ms);
        max_ms = std::max(max_ms, t.total_ms);
    }

    const float max_abs = verify_outputs(opt, final_buf, elems, bytes);
    const double avg_ms = total_ms / (double) opt.iters;
    const double avg_compute_ms = compute_ms / (double) opt.iters;
    const double avg_reduce_ms = reduce_ms / (double) opt.iters;
    const double per_reduction_ms = avg_reduce_ms / 2.0;
    const double doubling_steps = 3.0;
    const double wire_factor_per_reduction =
        opt.algo == Algorithm::Doubling ? (double) kParticipants * doubling_steps
                                        : (double) (kParticipants - 1) * 2.0;
    const double wire_bytes = (double) bytes * wire_factor_per_reduction * 2.0;
    const double effective_gbps = wire_bytes / (avg_reduce_ms / 1000.0) / 1.0e9;
    const double tok_s = (double) opt.tokens / (avg_ms / 1000.0);

    std::printf("ds4-v100-tp8-layer-smoke algo=%s devices=", algo_name(opt.algo));
    for (int p = 0; p < kParticipants; ++p) {
        std::printf("%s%d", p ? "," : "", opt.devices[p]);
    }
    std::printf(" root=%d tokens=%d hidden=%d dtype=f16 ctx=%d slots=%d ratio=%d "
                "kv_dtype=%s kv_per_slot_bytes=%zu kv_logical_bytes=%zu "
                "kv_shard_bytes=%zu compute_repeats=%d warmup=%d iters=%d\n",
                root_dev, opt.tokens, opt.hidden, opt.ctx, opt.slots, opt.ratio,
                kv_dtype_name(opt.kv_dtype), kv_plan.per_slot_bytes, kv_plan.logical_bytes,
                kv_plan.shard_bytes, opt.compute_repeats, opt.warmup, opt.iters);
    std::printf("latency_ms total_avg=%.6f min=%.6f max=%.6f compute_avg=%.6f "
                "reduce_avg=%.6f per_reduction=%.6f effective_wire_gbps=%.3f tok_s=%.3f\n",
                avg_ms, min_ms, max_ms, avg_compute_ms, avg_reduce_ms,
                per_reduction_ms, effective_gbps, tok_s);
    std::printf("verify cross_device_max_abs=%.9f %s\n", max_abs,
                max_abs <= 1.0e-5f ? "ok" : "FAIL");

    if (opt.algo == Algorithm::Nccl) {
        for (int p = 0; p < kParticipants; ++p) {
            CUDA_CHECK(cudaSetDevice(opt.devices[p]));
            NCCL_CHECK(ncclCommDestroy(comms[p]));
        }
    }
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaStreamDestroy(streams[p]));
    }
    CUDA_CHECK(cudaStreamDestroy(root_stream));
    CUDA_CHECK(cudaFree(staging));
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaFree(buf0[p]));
        CUDA_CHECK(cudaFree(buf1[p]));
        CUDA_CHECK(cudaFree(recv[p]));
        CUDA_CHECK(cudaFree(kv_shards[p]));
    }

    return max_abs <= 1.0e-5f ? 0 : 1;
}
