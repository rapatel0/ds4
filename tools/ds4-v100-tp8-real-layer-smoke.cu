#include <cublas_v2.h>
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

#define CUBLAS_CHECK(expr)                                                                        \
    do {                                                                                          \
        cublasStatus_t st__ = (expr);                                                             \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                                      \
            std::fprintf(stderr, "cublas error %s:%d: status %d\n", __FILE__, __LINE__,          \
                         (int) st__);                                                             \
            std::exit(2);                                                                         \
        }                                                                                         \
    } while (0)

enum class KvDType {
    F16,
    F8E4M3B128,
    Q8_0,
};

struct Options {
    int devices[kParticipants] = {0, 1, 2, 3, 4, 5, 6, 7};
    int tokens = 32;
    int hidden = 4096;
    int mid_shard = 1024;
    int ctx = 262144;
    int slots = 32;
    int ratio = 4;
    int warmup = 3;
    int iters = 20;
    KvDType kv_dtype = KvDType::F8E4M3B128;
};

struct KvPlan {
    size_t rows;
    size_t per_slot_bytes;
    size_t logical_bytes;
    size_t shard_bytes;
};

struct Timings {
    double gate_up_ms;
    double act_ms;
    double down_ms;
    double reduce_ms;
    double total_ms;
};

__global__ void fill_half_pattern_kernel(half * dst, size_t elems, float scale, float bias) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elems) {
        return;
    }
    const int lane = (int) (i % 251u);
    const float v = ((float) lane - 125.0f) * scale + bias;
    dst[i] = __float2half(v);
}

__global__ void gated_silu_kernel(half * mid, const half * gate, const half * up, size_t elems) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elems) {
        return;
    }
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    const float sig = 1.0f / (1.0f + expf(-g));
    mid[i] = __float2half((g * sig) * u);
}

__global__ void add_inplace_half_kernel(half * dst, const half * src, size_t elems) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elems) {
        return;
    }
    dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
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
    const size_t attn_values = checked_mul_size(plan.rows, kHeadDim);
    const size_t indexer_values =
        opt.ratio == 4 ? checked_mul_size((size_t) opt.ctx / 4u, kIndexerHeadDim) : 0;
    plan.per_slot_bytes =
        values_bytes(attn_values, opt.kv_dtype) + values_bytes(indexer_values, opt.kv_dtype);
    plan.logical_bytes = checked_mul_size(plan.per_slot_bytes, (size_t) opt.slots);
    plan.shard_bytes = ceil_div_size(plan.logical_bytes, kParticipants);
    return plan;
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
                 "       [--hidden N] [--mid-shard N] [--ctx N] [--slots N]\n"
                 "       [--ratio 4|128] [--kv-dtype f8|q8|f16]\n"
                 "       [--warmup N] [--iters N]\n",
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
        } else if (std::strcmp(arg, "--mid-shard") == 0) {
            if (val == nullptr || !parse_int(val, &opt->mid_shard) || opt->mid_shard <= 0) {
                std::fprintf(stderr, "invalid --mid-shard value\n");
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

void sync_streams(const Options & opt, cudaStream_t streams[kParticipants]) {
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaStreamSynchronize(streams[p]));
    }
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

void init_tensor(const Options & opt, int p, half * ptr, size_t elems, float scale, float bias,
                 cudaStream_t stream) {
    const int block = 256;
    const int grid = (int) ((elems + block - 1) / block);
    CUDA_CHECK(cudaSetDevice(opt.devices[p]));
    fill_half_pattern_kernel<<<grid, block, 0, stream>>>(ptr, elems, scale, bias);
    CUDA_CHECK(cudaGetLastError());
}

void run_gemm(cublasHandle_t handle, int m, int n, int k, const half * a, const half * b,
              half * c) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                              a, CUDA_R_16F, m, b, CUDA_R_16F, k, &beta, c,
                              CUDA_R_16F, m, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

Timings run_one_layer(const Options & opt, cublasHandle_t handles[kParticipants],
                      cudaStream_t streams[kParticipants], half ** x, half ** gate_w,
                      half ** up_w, half ** down_w, half ** gate, half ** up,
                      half ** mid, half ** partial, half ** reduced, half ** recv,
                      size_t hidden_elems, size_t mid_elems, size_t hidden_bytes) {
    Timings t = {};
    const auto total_start = std::chrono::steady_clock::now();

    const auto gu_start = std::chrono::steady_clock::now();
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        run_gemm(handles[p], opt.mid_shard, opt.tokens, opt.hidden, gate_w[p], x[p], gate[p]);
        run_gemm(handles[p], opt.mid_shard, opt.tokens, opt.hidden, up_w[p], x[p], up[p]);
    }
    sync_streams(opt, streams);
    const auto gu_stop = std::chrono::steady_clock::now();

    const auto act_start = std::chrono::steady_clock::now();
    const int block = 256;
    const int mid_grid = (int) ((mid_elems + block - 1) / block);
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        gated_silu_kernel<<<mid_grid, block, 0, streams[p]>>>(mid[p], gate[p], up[p],
                                                              mid_elems);
        CUDA_CHECK(cudaGetLastError());
    }
    sync_streams(opt, streams);
    const auto act_stop = std::chrono::steady_clock::now();

    const auto down_start = std::chrono::steady_clock::now();
    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        run_gemm(handles[p], opt.hidden, opt.tokens, opt.mid_shard, down_w[p], mid[p],
                 partial[p]);
    }
    sync_streams(opt, streams);
    const auto down_stop = std::chrono::steady_clock::now();

    const auto reduce_start = std::chrono::steady_clock::now();
    run_doubling_collective(opt, partial, reduced, recv, streams, hidden_elems, hidden_bytes);
    const auto reduce_stop = std::chrono::steady_clock::now();

    const auto total_stop = std::chrono::steady_clock::now();
    t.gate_up_ms = std::chrono::duration<double, std::milli>(gu_stop - gu_start).count();
    t.act_ms = std::chrono::duration<double, std::milli>(act_stop - act_start).count();
    t.down_ms = std::chrono::duration<double, std::milli>(down_stop - down_start).count();
    t.reduce_ms = std::chrono::duration<double, std::milli>(reduce_stop - reduce_start).count();
    t.total_ms = std::chrono::duration<double, std::milli>(total_stop - total_start).count();
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

    enable_peer_access_or_die(opt);
    const KvPlan kv_plan = make_kv_plan(opt);

    const size_t hidden_elems = (size_t) opt.hidden * (size_t) opt.tokens;
    const size_t mid_elems = (size_t) opt.mid_shard * (size_t) opt.tokens;
    const size_t hidden_bytes = hidden_elems * sizeof(half);
    const size_t mid_bytes = mid_elems * sizeof(half);
    const size_t gate_weight_elems = (size_t) opt.mid_shard * (size_t) opt.hidden;
    const size_t down_weight_elems = (size_t) opt.hidden * (size_t) opt.mid_shard;

    half * x[kParticipants] = {};
    half * gate_w[kParticipants] = {};
    half * up_w[kParticipants] = {};
    half * down_w[kParticipants] = {};
    half * gate[kParticipants] = {};
    half * up[kParticipants] = {};
    half * mid[kParticipants] = {};
    half * partial[kParticipants] = {};
    half * reduced[kParticipants] = {};
    half * recv[kParticipants] = {};
    unsigned char * kv_shards[kParticipants] = {};
    cudaStream_t streams[kParticipants] = {};
    cublasHandle_t handles[kParticipants] = {};

    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUDA_CHECK(cudaStreamCreate(&streams[p]));
        CUBLAS_CHECK(cublasCreate(&handles[p]));
        CUBLAS_CHECK(cublasSetStream(handles[p], streams[p]));
        CUBLAS_CHECK(cublasSetMathMode(handles[p], CUBLAS_TENSOR_OP_MATH));

        CUDA_CHECK(cudaMalloc(&x[p], hidden_bytes));
        CUDA_CHECK(cudaMalloc(&gate_w[p], gate_weight_elems * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&up_w[p], gate_weight_elems * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&down_w[p], down_weight_elems * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&gate[p], mid_bytes));
        CUDA_CHECK(cudaMalloc(&up[p], mid_bytes));
        CUDA_CHECK(cudaMalloc(&mid[p], mid_bytes));
        CUDA_CHECK(cudaMalloc(&partial[p], hidden_bytes));
        CUDA_CHECK(cudaMalloc(&reduced[p], hidden_bytes));
        CUDA_CHECK(cudaMalloc(&recv[p], hidden_bytes));
        CUDA_CHECK(cudaMalloc(&kv_shards[p], kv_plan.shard_bytes));

        init_tensor(opt, p, x[p], hidden_elems, 0.0002f, 0.001f * (float) (p + 1),
                    streams[p]);
        init_tensor(opt, p, gate_w[p], gate_weight_elems, 0.00003f,
                    0.00001f * (float) (p + 1), streams[p]);
        init_tensor(opt, p, up_w[p], gate_weight_elems, 0.00002f,
                    0.00002f * (float) (p + 1), streams[p]);
        init_tensor(opt, p, down_w[p], down_weight_elems, 0.000025f,
                    0.000015f * (float) (p + 1), streams[p]);
        CUDA_CHECK(cudaMemsetAsync(gate[p], 0, mid_bytes, streams[p]));
        CUDA_CHECK(cudaMemsetAsync(up[p], 0, mid_bytes, streams[p]));
        CUDA_CHECK(cudaMemsetAsync(mid[p], 0, mid_bytes, streams[p]));
        CUDA_CHECK(cudaMemsetAsync(partial[p], 0, hidden_bytes, streams[p]));
        CUDA_CHECK(cudaMemsetAsync(reduced[p], 0, hidden_bytes, streams[p]));
        CUDA_CHECK(cudaMemsetAsync(recv[p], 0, hidden_bytes, streams[p]));
        CUDA_CHECK(cudaMemsetAsync(kv_shards[p], 29 + p, kv_plan.shard_bytes, streams[p]));
    }
    sync_streams(opt, streams);

    for (int i = 0; i < opt.warmup; ++i) {
        (void) run_one_layer(opt, handles, streams, x, gate_w, up_w, down_w, gate, up, mid,
                             partial, reduced, recv, hidden_elems, mid_elems, hidden_bytes);
    }

    Timings sum = {};
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    for (int i = 0; i < opt.iters; ++i) {
        Timings t = run_one_layer(opt, handles, streams, x, gate_w, up_w, down_w, gate, up,
                                  mid, partial, reduced, recv, hidden_elems, mid_elems,
                                  hidden_bytes);
        sum.gate_up_ms += t.gate_up_ms;
        sum.act_ms += t.act_ms;
        sum.down_ms += t.down_ms;
        sum.reduce_ms += t.reduce_ms;
        sum.total_ms += t.total_ms;
        min_ms = std::min(min_ms, t.total_ms);
        max_ms = std::max(max_ms, t.total_ms);
    }

    const float max_abs = verify_outputs(opt, reduced, hidden_elems, hidden_bytes);
    const double inv_iters = 1.0 / (double) opt.iters;
    const double avg_total = sum.total_ms * inv_iters;
    const double avg_gate_up = sum.gate_up_ms * inv_iters;
    const double avg_act = sum.act_ms * inv_iters;
    const double avg_down = sum.down_ms * inv_iters;
    const double avg_reduce = sum.reduce_ms * inv_iters;
    const double tok_s = (double) opt.tokens / (avg_total / 1000.0);
    const double flops_per_gpu =
        2.0 * (double) opt.tokens * (double) opt.hidden * (double) opt.mid_shard * 3.0;
    const double aggregate_tflops =
        (flops_per_gpu * (double) kParticipants) / (avg_total / 1000.0) / 1.0e12;
    const double wire_bytes = (double) hidden_bytes * (double) kParticipants * 3.0;
    const double effective_wire_gbps = wire_bytes / (avg_reduce / 1000.0) / 1.0e9;

    std::printf("ds4-v100-tp8-real-layer-smoke devices=");
    for (int p = 0; p < kParticipants; ++p) {
        std::printf("%s%d", p ? "," : "", opt.devices[p]);
    }
    std::printf(" tokens=%d hidden=%d mid_shard=%d full_mid=%d ctx=%d slots=%d ratio=%d "
                "kv_dtype=%s kv_per_slot_bytes=%zu kv_logical_bytes=%zu kv_shard_bytes=%zu "
                "warmup=%d iters=%d\n",
                opt.tokens, opt.hidden, opt.mid_shard, opt.mid_shard * kParticipants,
                opt.ctx, opt.slots, opt.ratio, kv_dtype_name(opt.kv_dtype),
                kv_plan.per_slot_bytes, kv_plan.logical_bytes, kv_plan.shard_bytes,
                opt.warmup, opt.iters);
    std::printf("latency_ms total_avg=%.6f min=%.6f max=%.6f gate_up=%.6f "
                "activation=%.6f down=%.6f reduce=%.6f effective_wire_gbps=%.3f "
                "fixture_tflops=%.3f tok_s=%.3f\n",
                avg_total, min_ms, max_ms, avg_gate_up, avg_act, avg_down, avg_reduce,
                effective_wire_gbps, aggregate_tflops, tok_s);
    std::printf("verify cross_device_max_abs=%.9f %s\n", max_abs,
                max_abs <= 1.0e-5f ? "ok" : "FAIL");

    for (int p = 0; p < kParticipants; ++p) {
        CUDA_CHECK(cudaSetDevice(opt.devices[p]));
        CUBLAS_CHECK(cublasDestroy(handles[p]));
        CUDA_CHECK(cudaStreamDestroy(streams[p]));
        CUDA_CHECK(cudaFree(x[p]));
        CUDA_CHECK(cudaFree(gate_w[p]));
        CUDA_CHECK(cudaFree(up_w[p]));
        CUDA_CHECK(cudaFree(down_w[p]));
        CUDA_CHECK(cudaFree(gate[p]));
        CUDA_CHECK(cudaFree(up[p]));
        CUDA_CHECK(cudaFree(mid[p]));
        CUDA_CHECK(cudaFree(partial[p]));
        CUDA_CHECK(cudaFree(reduced[p]));
        CUDA_CHECK(cudaFree(recv[p]));
        CUDA_CHECK(cudaFree(kv_shards[p]));
    }

    return max_abs <= 1.0e-5f ? 0 : 1;
}
