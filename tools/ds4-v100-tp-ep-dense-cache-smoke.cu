#define _FILE_OFFSET_BITS 64

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/types.h>
#include <vector>

namespace {

constexpr int kGpus = 8;
constexpr uint64_t KiB = 1024ull;
constexpr uint64_t MiB = 1024ull * KiB;
constexpr uint64_t GiB = 1024ull * MiB;

#define CHECK_CUDA(expr)                                                              \
    do {                                                                              \
        cudaError_t err__ = (expr);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            std::fprintf(stderr, "cuda error %s:%d: %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(err__));                                  \
            std::exit(2);                                                             \
        }                                                                             \
    } while (0)

struct Options {
    const char *pack_dir = nullptr;
    const char *contract_path = nullptr;
    int devices[kGpus] = {0, 1, 2, 3, 4, 5, 6, 7};
    int layer = -999;
    int slots = 32;
    int warmup = 1;
    int iters = 5;
    bool execute_table = false;
};

struct DenseRow {
    std::string tensor_id;
    std::string dtype;
    std::string shape;
    int layer = -1;
    int gpu = -1;
    int shard_index = -1;
    int shard_count = -1;
    int cols = 0;
    int rows_per_gpu = 0;
    uint64_t source_bytes = 0;
    std::string pack_file;
    uint64_t source_shard_offset = 0;
    uint64_t source_byte_length = 0;
    uint64_t physical_offset = 0;
    uint64_t cache_offset = 0;
    uint64_t cache_bytes = 0;
    uint64_t cache_aligned_bytes = 0;
};

struct GpuPlan {
    uint64_t rows = 0;
    uint64_t f8_rows = 0;
    uint64_t bf16_rows = 0;
    uint64_t source_bytes = 0;
    uint64_t f8_source_bytes = 0;
    uint64_t bf16_source_bytes = 0;
    uint64_t cache_bytes = 0;
    uint64_t cache_aligned_bytes = 0;
    uint64_t max_temp_bytes = 0;
    size_t free_before = 0;
    size_t total_before = 0;
    size_t free_after_alloc = 0;
    size_t free_after_temp_free = 0;
    double host_read_ms = 0.0;
    double h2d_convert_ms = 0.0;
    unsigned long long checksum = 0;
    unsigned long long nonfinite = 0;
};

struct DenseGroup {
    std::string tensor_id;
    int layer = -1;
    int cols = 0;
    int rows_per_gpu = 0;
    uint64_t cache_offset[kGpus] = {};
    bool have[kGpus] = {};
};

struct ExecuteStats {
    bool enabled = false;
    bool pass = true;
    uint64_t groups = 0;
    uint64_t layer_groups = 0;
    uint64_t gemms_per_iter = 0;
    uint64_t total_gemms = 0;
    uint64_t max_cols = 0;
    uint64_t max_rows_per_gpu = 0;
    uint64_t flops_per_iter = 0;
    double total_ms = 0.0;
    double ms_per_iter = 0.0;
    double dense_table_tflops = 0.0;
    unsigned long long checksum = 0;
    unsigned long long nonfinite = 0;
};

static uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

static double as_gib(uint64_t b) {
    return (double)b / (double)GiB;
}

static std::vector<std::string> split_tabs(const std::string &s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        const size_t tab = s.find('\t', start);
        if (tab == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, tab - start));
        start = tab + 1;
    }
    return out;
}

static bool parse_int(const char *s, int *out) {
    if (!s || !*s) return false;
    char *end = nullptr;
    errno = 0;
    const long v = std::strtol(s, &end, 10);
    if (errno || !end || *end) return false;
    *out = (int)v;
    return true;
}

static bool parse_u64(const char *s, uint64_t *out) {
    if (!s || !*s) return false;
    char *end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (errno || !end || *end) return false;
    *out = (uint64_t)v;
    return true;
}

static bool parse_shape2(const std::string &shape, int *cols, int *rows) {
    if (shape.size() < 5 || shape.front() != '[' || shape.back() != ']') return false;
    const size_t x = shape.find('x');
    if (x == std::string::npos) return false;
    if (!parse_int(shape.substr(1, x - 1).c_str(), cols)) return false;
    if (!parse_int(shape.substr(x + 1, shape.size() - x - 2).c_str(), rows)) return false;
    return *cols > 0 && *rows > 0;
}

static bool parse_devices(const char *s, int devices[kGpus]) {
    std::string v(s ? s : "");
    size_t start = 0;
    for (int i = 0; i < kGpus; ++i) {
        const size_t comma = v.find(',', start);
        const std::string part = comma == std::string::npos
            ? v.substr(start)
            : v.substr(start, comma - start);
        if (!parse_int(part.c_str(), &devices[i])) return false;
        if (comma == std::string::npos) return i == kGpus - 1;
        start = comma + 1;
    }
    return start >= v.size();
}

static std::string path_join(const char *a, const std::string &b) {
    std::string out(a ? a : "");
    if (!out.empty() && out.back() != '/') out.push_back('/');
    out += b;
    return out;
}

static int read_exact_at(const std::string &path, uint64_t off, void *dst, size_t n) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "open failed %s: %s\n", path.c_str(), std::strerror(errno));
        return 1;
    }
    if (fseeko(fp, (off_t)off, SEEK_SET) != 0) {
        std::fprintf(stderr, "seek failed %s: %s\n", path.c_str(), std::strerror(errno));
        std::fclose(fp);
        return 2;
    }
    const size_t got = std::fread(dst, 1, n, fp);
    std::fclose(fp);
    if (got != n) {
        std::fprintf(stderr, "short read %s got=%zu want=%zu\n", path.c_str(), got, n);
        return 3;
    }
    return 0;
}

static uint64_t f8_row_bytes(int cols) {
    return (uint64_t)(cols / 128) * 129ull;
}

static bool parse_dense_contract_row(const std::vector<std::string> &f,
                                     DenseRow *out,
                                     int layer_filter) {
    if (f.size() < 23 || f[0] != "dense_tp") return false;
    DenseRow r;
    r.tensor_id = f[1];
    if (!parse_int(f[3].c_str(), &r.layer)) return false;
    if (layer_filter != -999 && r.layer != layer_filter) return false;
    r.dtype = f[5];
    if (r.dtype != "f8_e4m3_b128" && r.dtype != "bf16") return false;
    r.shape = f[6];
    if (!parse_int(f[8].c_str(), &r.gpu) || r.gpu < 0 || r.gpu >= kGpus) return false;
    if (!parse_int(f[12].c_str(), &r.shard_index)) return false;
    if (!parse_int(f[13].c_str(), &r.shard_count) || r.shard_count <= 0) return false;
    if (!parse_u64(f[18].c_str(), &r.source_bytes)) return false;
    r.pack_file = f[19];
    if (!parse_u64(f[20].c_str(), &r.source_shard_offset)) return false;
    if (!parse_u64(f[21].c_str(), &r.source_byte_length)) return false;
    int total_rows = 0;
    if (!parse_shape2(r.shape, &r.cols, &total_rows)) return false;
    if (r.dtype == "f8_e4m3_b128") {
        if (r.cols % 128 != 0) return false;
        const uint64_t rb = f8_row_bytes(r.cols);
        if (rb == 0 || r.source_bytes % rb != 0) return false;
        r.rows_per_gpu = (int)(r.source_bytes / rb);
    } else {
        const uint64_t rb = (uint64_t)r.cols * sizeof(uint16_t);
        if (rb == 0 || r.source_bytes % rb != 0) return false;
        r.rows_per_gpu = (int)(r.source_bytes / rb);
    }
    const uint64_t span = r.source_bytes * (uint64_t)r.shard_count;
    r.physical_offset = r.source_shard_offset;
    if (r.shard_index >= 0 && r.source_byte_length >= span) {
        r.physical_offset += (uint64_t)r.shard_index * r.source_bytes;
    }
    r.cache_bytes = (uint64_t)r.rows_per_gpu * (uint64_t)r.cols * sizeof(__half);
    r.cache_aligned_bytes = align_up(r.cache_bytes, 256);
    *out = r;
    return true;
}

static int parse_contract(const Options &opt, std::vector<DenseRow> *rows) {
    FILE *fp = std::fopen(opt.contract_path, "rb");
    if (!fp) {
        std::fprintf(stderr, "cannot open contract %s: %s\n",
                     opt.contract_path, std::strerror(errno));
        return 1;
    }
    char buf[8192];
    bool first = true;
    while (std::fgets(buf, sizeof(buf), fp)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (first) {
            first = false;
            continue;
        }
        DenseRow r;
        if (parse_dense_contract_row(split_tabs(line), &r, opt.layer)) rows->push_back(r);
    }
    std::fclose(fp);
    return rows->empty() ? 2 : 0;
}

__device__ float f8_e8m0_to_f32_dev(uint8_t e) {
    return __uint_as_float(e == 0 ? 0x00400000u : ((uint32_t)e << 23));
}

__device__ float f8_e4m3fn_to_f32_dev(uint8_t x) {
    const uint32_t sign = ((uint32_t)x & 0x80u) << 24;
    const uint32_t ax = (uint32_t)x & 0x7fu;
    if (ax == 0) return __uint_as_float(sign ? 0x80000000u : 0u);
    if (ax == 0x7f) return __uint_as_float(0x7fc00000u);
    const uint32_t exp = ax >> 3;
    const uint32_t man = ax & 0x07u;
    if (exp != 0) return __uint_as_float(sign | ((exp + 120u) << 23) | (man << 20));
    const uint32_t hi = man >= 4u ? 2u : (man >= 2u ? 1u : 0u);
    const uint32_t mant = (man << (23u - hi)) & 0x007fffffu;
    return __uint_as_float(sign | ((118u + hi) << 23) | mant);
}

__global__ void f8_b128_to_half_kernel(__half *out,
                                       const uint8_t *weights,
                                       uint64_t elems,
                                       uint32_t cols,
                                       uint32_t row_stride_bytes) {
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < elems;
         idx += (uint64_t)blockDim.x * gridDim.x) {
        const uint64_t row = idx / cols;
        const uint32_t col = (uint32_t)(idx - row * cols);
        const uint8_t *wrow = weights + row * row_stride_bytes;
        const uint8_t *block = wrow + (uint64_t)(col / 128u) * 129ull;
        const float v = f8_e4m3fn_to_f32_dev(block[1u + (col % 128u)]) *
                        f8_e8m0_to_f32_dev(block[0]);
        out[idx] = __float2half(v);
    }
}

__global__ void bf16_bits_to_half_kernel(__half *out, const uint16_t *in, uint64_t elems) {
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < elems;
         idx += (uint64_t)blockDim.x * gridDim.x) {
        out[idx] = __float2half(__uint_as_float((uint32_t)in[idx] << 16));
    }
}

__global__ void checksum_half_kernel(const __half *data,
                                     uint64_t elems,
                                     unsigned long long *checksum,
                                     unsigned long long *nonfinite) {
    unsigned long long local = 0;
    unsigned long long bad = 0;
    const uint16_t *bits = reinterpret_cast<const uint16_t *>(data);
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < elems;
         idx += (uint64_t)blockDim.x * gridDim.x) {
        const uint16_t v = bits[idx];
        local += (unsigned long long)v * (unsigned long long)((idx % 251u) + 1u);
        if ((v & 0x7c00u) == 0x7c00u) bad++;
    }
    atomicAdd(checksum, local);
    atomicAdd(nonfinite, bad);
}

__global__ void fill_half_kernel(__half *out, uint64_t elems, uint32_t seed) {
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < elems;
         idx += (uint64_t)blockDim.x * gridDim.x) {
        const uint32_t m = (uint32_t)((idx * 17ull + (uint64_t)seed * 97ull) % 4093ull);
        out[idx] = __float2half(((float)m - 2048.0f) * 0.00005f);
    }
}

__global__ void checksum_float_kernel(const float *data,
                                      uint64_t elems,
                                      unsigned long long *checksum,
                                      unsigned long long *nonfinite) {
    unsigned long long local = 0;
    unsigned long long bad = 0;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < elems;
         idx += (uint64_t)blockDim.x * gridDim.x) {
        const float v = data[idx];
        uint32_t bits = 0;
        memcpy(&bits, &v, sizeof(bits));
        local += (unsigned long long)bits * (unsigned long long)((idx % 251u) + 1u);
        if (!isfinite(v)) bad++;
    }
    atomicAdd(checksum, local);
    atomicAdd(nonfinite, bad);
}

static DenseGroup *find_or_add_group(std::vector<DenseGroup> *groups, const DenseRow &r) {
    for (DenseGroup &g : *groups) {
        if (g.layer == r.layer && g.tensor_id == r.tensor_id) return &g;
    }
    groups->push_back(DenseGroup{});
    DenseGroup &g = groups->back();
    g.tensor_id = r.tensor_id;
    g.layer = r.layer;
    g.cols = r.cols;
    g.rows_per_gpu = r.rows_per_gpu;
    return &g;
}

static std::vector<DenseGroup> build_dense_groups(const std::vector<DenseRow> &rows) {
    std::vector<DenseGroup> groups;
    for (const DenseRow &r : rows) {
        if (r.layer < 0) continue;
        DenseGroup *g = find_or_add_group(&groups, r);
        if (g->cols != r.cols || g->rows_per_gpu != r.rows_per_gpu) continue;
        g->cache_offset[r.gpu] = r.cache_offset;
        g->have[r.gpu] = true;
    }
    std::vector<DenseGroup> complete;
    for (const DenseGroup &g : groups) {
        bool ok = true;
        for (int gpu = 0; gpu < kGpus; ++gpu) ok = ok && g.have[gpu];
        if (ok) complete.push_back(g);
    }
    std::sort(complete.begin(), complete.end(),
              [](const DenseGroup &a, const DenseGroup &b) {
                  if (a.layer != b.layer) return a.layer < b.layer;
                  return a.tensor_id < b.tensor_id;
              });
    return complete;
}

static int execute_dense_table(const Options &opt,
                               const std::vector<DenseRow> &rows,
                               uint8_t *d_cache[kGpus],
                               ExecuteStats *stats) {
    if (!opt.execute_table) return 0;
    stats->enabled = true;
    const std::vector<DenseGroup> groups = build_dense_groups(rows);
    stats->groups = groups.size();
    for (const DenseGroup &g : groups) {
        stats->max_cols = std::max<uint64_t>(stats->max_cols, (uint64_t)g.cols);
        stats->max_rows_per_gpu =
            std::max<uint64_t>(stats->max_rows_per_gpu, (uint64_t)g.rows_per_gpu);
        stats->flops_per_iter +=
            2ull * (uint64_t)opt.slots * (uint64_t)g.cols *
            (uint64_t)g.rows_per_gpu * (uint64_t)kGpus;
    }
    stats->layer_groups = groups.size();
    stats->gemms_per_iter = groups.size() * (uint64_t)kGpus;
    stats->total_gemms = stats->gemms_per_iter * (uint64_t)opt.iters;
    if (groups.empty()) return 1;

    cublasHandle_t blas[kGpus] = {};
    cudaStream_t streams[kGpus] = {};
    __half *d_x[kGpus] = {};
    float *d_out[kGpus] = {};
    unsigned long long *d_checksum[kGpus] = {};
    unsigned long long *d_nonfinite[kGpus] = {};
    const uint64_t max_x_elems = (uint64_t)opt.slots * stats->max_cols;
    const uint64_t max_out_elems = (uint64_t)opt.slots * stats->max_rows_per_gpu;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaStreamCreate(&streams[gpu]));
        CHECK_CUDA(cudaMalloc(&d_x[gpu], (size_t)max_x_elems * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_out[gpu], (size_t)max_out_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_checksum[gpu], sizeof(unsigned long long)));
        CHECK_CUDA(cudaMalloc(&d_nonfinite[gpu], sizeof(unsigned long long)));
        CHECK_CUDA(cudaMemset(d_checksum[gpu], 0, sizeof(unsigned long long)));
        CHECK_CUDA(cudaMemset(d_nonfinite[gpu], 0, sizeof(unsigned long long)));
        cublasStatus_t st = cublasCreate(&blas[gpu]);
        if (st != CUBLAS_STATUS_SUCCESS) return 2;
        (void)cublasSetMathMode(blas[gpu], CUBLAS_TENSOR_OP_MATH);
        (void)cublasSetStream(blas[gpu], streams[gpu]);
    }

    auto run_group = [&](const DenseGroup &g, bool checksum) -> int {
        const uint64_t x_elems = (uint64_t)opt.slots * (uint64_t)g.cols;
        const uint64_t out_elems = (uint64_t)opt.slots * (uint64_t)g.rows_per_gpu;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            const unsigned int x_grid =
                (unsigned int)std::min<uint64_t>(65535, (x_elems + 255) / 256);
            fill_half_kernel<<<x_grid, 256, 0, streams[gpu]>>>(
                d_x[gpu], x_elems, (uint32_t)(g.layer * 131 + gpu * 17 + g.cols));
            CHECK_CUDA(cudaGetLastError());
            const __half *w =
                reinterpret_cast<const __half *>(d_cache[gpu] + g.cache_offset[gpu]);
            cublasStatus_t st = cublasGemmEx(blas[gpu],
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             g.rows_per_gpu,
                                             opt.slots,
                                             g.cols,
                                             &alpha,
                                             w,
                                             CUDA_R_16F,
                                             g.cols,
                                             d_x[gpu],
                                             CUDA_R_16F,
                                             g.cols,
                                             &beta,
                                             d_out[gpu],
                                             CUDA_R_32F,
                                             g.rows_per_gpu,
                                             CUDA_R_32F,
                                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) return 3;
            if (checksum) {
                const unsigned int out_grid =
                    (unsigned int)std::min<uint64_t>(65535, (out_elems + 255) / 256);
                checksum_float_kernel<<<out_grid, 256, 0, streams[gpu]>>>(
                    d_out[gpu], out_elems, d_checksum[gpu], d_nonfinite[gpu]);
                CHECK_CUDA(cudaGetLastError());
            }
        }
        for (int gpu = 0; gpu < kGpus; ++gpu) {
            CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
            CHECK_CUDA(cudaStreamSynchronize(streams[gpu]));
        }
        return 0;
    };

    for (int i = 0; i < opt.warmup; ++i) {
        for (const DenseGroup &g : groups) {
            const int rc = run_group(g, false);
            if (rc != 0) return rc;
        }
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < opt.iters; ++i) {
        for (const DenseGroup &g : groups) {
            const int rc = run_group(g, false);
            if (rc != 0) return rc;
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    stats->total_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    stats->ms_per_iter = stats->total_ms / (double)opt.iters;
    stats->dense_table_tflops =
        stats->ms_per_iter > 0.0
            ? (double)stats->flops_per_iter / (stats->ms_per_iter * 1.0e9)
            : 0.0;

    for (const DenseGroup &g : groups) {
        const int rc = run_group(g, true);
        if (rc != 0) return rc;
    }
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        unsigned long long h_checksum = 0;
        unsigned long long h_bad = 0;
        CHECK_CUDA(cudaMemcpy(&h_checksum, d_checksum[gpu], sizeof(h_checksum),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_bad, d_nonfinite[gpu], sizeof(h_bad),
                              cudaMemcpyDeviceToHost));
        stats->checksum ^= h_checksum + (unsigned long long)(gpu + 1) * 1000003ull;
        stats->nonfinite += h_bad;
    }
    if (stats->checksum == 0 || stats->nonfinite != 0) stats->pass = false;

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (blas[gpu]) (void)cublasDestroy(blas[gpu]);
        if (d_x[gpu]) CHECK_CUDA(cudaFree(d_x[gpu]));
        if (d_out[gpu]) CHECK_CUDA(cudaFree(d_out[gpu]));
        if (d_checksum[gpu]) CHECK_CUDA(cudaFree(d_checksum[gpu]));
        if (d_nonfinite[gpu]) CHECK_CUDA(cudaFree(d_nonfinite[gpu]));
        if (streams[gpu]) CHECK_CUDA(cudaStreamDestroy(streams[gpu]));
    }
    return stats->pass ? 0 : 4;
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s --pack-dir DIR --contract FILE [options]\n"
                 "Options:\n"
                 "  --devices 0,1,2,3,4,5,6,7  CUDA devices\n"
                 "  --layer N                    Filter to one layer. Default: all dense rows\n"
                 "  --execute-table              Run cache-backed cuBLAS over dense layer groups\n"
                 "  --slots N                    Active tokens for table execution. Default: 32\n"
                 "  --warmup N                   Warmup table iterations. Default: 1\n"
                 "  --iters N                    Timed table iterations. Default: 5\n",
                 argv0);
}

static bool parse_args(int argc, char **argv, Options *opt) {
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : nullptr;
        if (!std::strcmp(a, "--pack-dir") && v) {
            opt->pack_dir = v;
            ++i;
        } else if (!std::strcmp(a, "--contract") && v) {
            opt->contract_path = v;
            ++i;
        } else if (!std::strcmp(a, "--devices") && v) {
            if (!parse_devices(v, opt->devices)) return false;
            ++i;
        } else if (!std::strcmp(a, "--layer") && v) {
            if (!parse_int(v, &opt->layer)) return false;
            ++i;
        } else if (!std::strcmp(a, "--slots") && v) {
            if (!parse_int(v, &opt->slots) || opt->slots <= 0) return false;
            ++i;
        } else if (!std::strcmp(a, "--warmup") && v) {
            if (!parse_int(v, &opt->warmup) || opt->warmup < 0) return false;
            ++i;
        } else if (!std::strcmp(a, "--iters") && v) {
            if (!parse_int(v, &opt->iters) || opt->iters <= 0) return false;
            ++i;
        } else if (!std::strcmp(a, "--execute-table")) {
            opt->execute_table = true;
        } else if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) {
            usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }
    return opt->pack_dir && opt->contract_path;
}

} // namespace

int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        usage(argv[0]);
        return 2;
    }

    std::vector<DenseRow> rows;
    if (parse_contract(opt, &rows) != 0) {
        std::fprintf(stderr, "no compatible dense rows found\n");
        return 2;
    }

    GpuPlan plan[kGpus];
    for (DenseRow &r : rows) {
        GpuPlan &g = plan[r.gpu];
        r.cache_offset = g.cache_aligned_bytes;
        g.rows++;
        g.source_bytes += r.source_bytes;
        g.cache_bytes += r.cache_bytes;
        g.cache_aligned_bytes += r.cache_aligned_bytes;
        g.max_temp_bytes = std::max(g.max_temp_bytes, r.source_bytes);
        if (r.dtype == "f8_e4m3_b128") {
            g.f8_rows++;
            g.f8_source_bytes += r.source_bytes;
        } else {
            g.bf16_rows++;
            g.bf16_source_bytes += r.source_bytes;
        }
    }

    uint8_t *d_cache[kGpus] = {};
    uint8_t *d_temp[kGpus] = {};
    unsigned long long *d_checksum[kGpus] = {};
    unsigned long long *d_nonfinite[kGpus] = {};
    cudaEvent_t start[kGpus] = {};
    cudaEvent_t stop[kGpus] = {};

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaMemGetInfo(&plan[gpu].free_before, &plan[gpu].total_before));
        if (plan[gpu].cache_aligned_bytes) {
            CHECK_CUDA(cudaMalloc(&d_cache[gpu], (size_t)plan[gpu].cache_aligned_bytes));
        }
        if (plan[gpu].max_temp_bytes) {
            CHECK_CUDA(cudaMalloc(&d_temp[gpu], (size_t)plan[gpu].max_temp_bytes));
        }
        CHECK_CUDA(cudaMalloc(&d_checksum[gpu], sizeof(unsigned long long)));
        CHECK_CUDA(cudaMalloc(&d_nonfinite[gpu], sizeof(unsigned long long)));
        CHECK_CUDA(cudaMemset(d_checksum[gpu], 0, sizeof(unsigned long long)));
        CHECK_CUDA(cudaMemset(d_nonfinite[gpu], 0, sizeof(unsigned long long)));
        CHECK_CUDA(cudaEventCreate(&start[gpu]));
        CHECK_CUDA(cudaEventCreate(&stop[gpu]));
        CHECK_CUDA(cudaMemGetInfo(&plan[gpu].free_after_alloc, &plan[gpu].total_before));
    }

    std::vector<uint8_t> host;
    for (const DenseRow &r : rows) {
        GpuPlan &g = plan[r.gpu];
        host.resize((size_t)r.source_bytes);
        const std::string path = path_join(opt.pack_dir, r.pack_file);
        const auto h0 = std::chrono::steady_clock::now();
        if (read_exact_at(path, r.physical_offset, host.data(), host.size()) != 0) {
            return 3;
        }
        const auto h1 = std::chrono::steady_clock::now();
        g.host_read_ms += std::chrono::duration<double, std::milli>(h1 - h0).count();

        CHECK_CUDA(cudaSetDevice(opt.devices[r.gpu]));
        CHECK_CUDA(cudaEventRecord(start[r.gpu]));
        CHECK_CUDA(cudaMemcpy(d_temp[r.gpu], host.data(), host.size(), cudaMemcpyHostToDevice));
        __half *dst = reinterpret_cast<__half *>(d_cache[r.gpu] + r.cache_offset);
        const uint64_t elems = r.cache_bytes / sizeof(__half);
        const unsigned int block = 256;
        const unsigned int grid = (unsigned int)std::min<uint64_t>(65535, (elems + block - 1) / block);
        if (r.dtype == "f8_e4m3_b128") {
            f8_b128_to_half_kernel<<<grid, block>>>(
                dst, d_temp[r.gpu], elems, (uint32_t)r.cols, (uint32_t)f8_row_bytes(r.cols));
        } else {
            bf16_bits_to_half_kernel<<<grid, block>>>(
                dst, reinterpret_cast<const uint16_t *>(d_temp[r.gpu]), elems);
        }
        CHECK_CUDA(cudaGetLastError());
        checksum_half_kernel<<<grid, block>>>(dst, elems, d_checksum[r.gpu], d_nonfinite[r.gpu]);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop[r.gpu]));
        CHECK_CUDA(cudaEventSynchronize(stop[r.gpu]));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start[r.gpu], stop[r.gpu]));
        g.h2d_convert_ms += (double)ms;
    }

    bool pass = true;
    uint64_t total_rows = 0;
    uint64_t total_cache = 0;
    uint64_t total_source = 0;
    ExecuteStats execute;
    const int execute_rc = execute_dense_table(opt, rows, d_cache, &execute);
    if (execute_rc != 0) pass = false;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (d_temp[gpu]) CHECK_CUDA(cudaFree(d_temp[gpu]));
        CHECK_CUDA(cudaMemGetInfo(&plan[gpu].free_after_temp_free, &plan[gpu].total_before));
        CHECK_CUDA(cudaMemcpy(&plan[gpu].checksum, d_checksum[gpu],
                              sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&plan[gpu].nonfinite, d_nonfinite[gpu],
                              sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        total_rows += plan[gpu].rows;
        total_cache += plan[gpu].cache_aligned_bytes;
        total_source += plan[gpu].source_bytes;
        if (plan[gpu].rows == 0 || plan[gpu].checksum == 0 || plan[gpu].nonfinite != 0) {
            pass = false;
        }
        std::printf("gpu_dense_cache\tgpu\t%d\tdevice\t%d\trows\t%llu\tf8_rows\t%llu\t"
                    "bf16_rows\t%llu\tsource_gib\t%.6f\tf8_source_gib\t%.6f\t"
                    "bf16_source_gib\t%.6f\tcache_gib\t%.6f\tcache_aligned_gib\t%.6f\t"
                    "max_temp_mib\t%.3f\tfree_before_gib\t%.3f\tfree_after_alloc_gib\t%.3f\t"
                    "free_after_temp_free_gib\t%.3f\thost_read_ms\t%.3f\t"
                    "h2d_convert_ms\t%.3f\tchecksum\t%llu\tnonfinite\t%llu\n",
                    gpu, opt.devices[gpu],
                    (unsigned long long)plan[gpu].rows,
                    (unsigned long long)plan[gpu].f8_rows,
                    (unsigned long long)plan[gpu].bf16_rows,
                    as_gib(plan[gpu].source_bytes),
                    as_gib(plan[gpu].f8_source_bytes),
                    as_gib(plan[gpu].bf16_source_bytes),
                    as_gib(plan[gpu].cache_bytes),
                    as_gib(plan[gpu].cache_aligned_bytes),
                    (double)plan[gpu].max_temp_bytes / (double)MiB,
                    as_gib((uint64_t)plan[gpu].free_before),
                    as_gib((uint64_t)plan[gpu].free_after_alloc),
                    as_gib((uint64_t)plan[gpu].free_after_temp_free),
                    plan[gpu].host_read_ms,
                    plan[gpu].h2d_convert_ms,
                    plan[gpu].checksum,
                    plan[gpu].nonfinite);
    }

    if (execute.enabled) {
        std::printf("tp_ep_dense_table_execute\tlayer\t%s\tslots\t%d\t"
                    "groups\t%llu\tgemms_per_iter\t%llu\titers\t%d\t"
                    "total_gemms\t%llu\tmax_cols\t%llu\tmax_rows_per_gpu\t%llu\t"
                    "flops_per_iter\t%llu\ttotal_ms\t%.6f\tms_per_iter\t%.6f\t"
                    "dense_table_tflops\t%.6f\tchecksum\t%llu\tnonfinite\t%llu\t%s\n",
                    opt.layer == -999 ? "all" : std::to_string(opt.layer).c_str(),
                    opt.slots,
                    (unsigned long long)execute.groups,
                    (unsigned long long)execute.gemms_per_iter,
                    opt.iters,
                    (unsigned long long)execute.total_gemms,
                    (unsigned long long)execute.max_cols,
                    (unsigned long long)execute.max_rows_per_gpu,
                    (unsigned long long)execute.flops_per_iter,
                    execute.total_ms,
                    execute.ms_per_iter,
                    execute.dense_table_tflops,
                    execute.checksum,
                    execute.nonfinite,
                    execute.pass ? "PASS" : "FAIL");
    }

    std::printf("tp_ep_dense_cache_smoke\tlayer\t%s\trows\t%llu\tsource_gib\t%.6f\t"
                "cache_aligned_gib\t%.6f\t%s\n",
                opt.layer == -999 ? "all" : std::to_string(opt.layer).c_str(),
                (unsigned long long)total_rows,
                as_gib(total_source),
                as_gib(total_cache),
                pass ? "PASS" : "FAIL");

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (d_cache[gpu]) CHECK_CUDA(cudaFree(d_cache[gpu]));
        if (d_checksum[gpu]) CHECK_CUDA(cudaFree(d_checksum[gpu]));
        if (d_nonfinite[gpu]) CHECK_CUDA(cudaFree(d_nonfinite[gpu]));
        if (start[gpu]) CHECK_CUDA(cudaEventDestroy(start[gpu]));
        if (stop[gpu]) CHECK_CUDA(cudaEventDestroy(stop[gpu]));
    }

    return pass ? 0 : 1;
}
