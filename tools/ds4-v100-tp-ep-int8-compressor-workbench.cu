#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "v12_kernels.cuh"
#include "v13_kernels.cuh"

#define DS4_CUDA_CHECK(call)                                                     \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                         cudaGetErrorString(err__), __FILE__, __LINE__, #call);  \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

#define DS4_CUBLAS_CHECK(call)                                                   \
    do {                                                                         \
        cublasStatus_t st__ = (call);                                            \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d: %s\n",              \
                         (int) st__, __FILE__, __LINE__, #call);                 \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

struct Options {
    int warmup = 20;
    int iters = 200;
    std::string out_tsv = "int8-compressor-workbench.tsv";
    std::string report = "INT8_COMPRESSOR_WORKBENCH.md";
};

struct Shape {
    int m;
    int n;
    int k;
};

struct Result {
    int m = 0;
    int n = 0;
    int k = 0;
    const char * label = "";
    double ms = 0.0;
    double tflops = 0.0;
    double gbps = 0.0;
    float max_abs = 0.0f;
    float p99_abs = 0.0f;
    double mean_abs = 0.0;
    bool ok = false;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s [--warmup N] [--iters N] [--out-tsv PATH] [--report PATH]\n",
                 argv0);
}

static bool parse_int(const char * s, int * out) {
    char * end = nullptr;
    errno = 0;
    long v = std::strtol(s, &end, 10);
    if (errno || end == s || *end != '\0' || v <= 0 || v > 100000000) {
        return false;
    }
    *out = (int) v;
    return true;
}

static Options parse_options(int argc, char ** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (arg == "--warmup" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opt.warmup)) {
                usage(argv[0]);
                std::exit(2);
            }
        } else if (arg == "--iters" && i + 1 < argc) {
            if (!parse_int(argv[++i], &opt.iters)) {
                usage(argv[0]);
                std::exit(2);
            }
        } else if (arg == "--out-tsv" && i + 1 < argc) {
            opt.out_tsv = argv[++i];
        } else if (arg == "--report" && i + 1 < argc) {
            opt.report = argv[++i];
        } else {
            usage(argv[0]);
            std::exit(2);
        }
    }
    return opt;
}

static float deterministic_a(size_t i) {
    const int v = (int) ((i * 17u + 3u) % 67u) - 33;
    return (float) v * (1.0f / 64.0f);
}

static float deterministic_w(size_t i) {
    const int v = (int) ((i * 13u + 7u) % 31u) - 15;
    return (float) v * (1.0f / 32.0f);
}

static void init_shape(const Shape & s,
                       std::vector<float> * a_f32,
                       std::vector<__half> * a_f16,
                       std::vector<float> * w_f32,
                       std::vector<__half> * w_f16,
                       std::vector<int8_t> * w_qs,
                       std::vector<__half> * w_scales) {
    const int qk = 32;
    a_f32->resize((size_t) s.m * (size_t) s.k);
    a_f16->resize(a_f32->size());
    w_f32->resize((size_t) s.n * (size_t) s.k);
    w_f16->resize(w_f32->size());
    w_qs->resize(w_f32->size());
    w_scales->resize((size_t) s.n * (size_t) (s.k / qk));

    for (size_t i = 0; i < a_f32->size(); ++i) {
        const float v = deterministic_a(i);
        (*a_f32)[i] = v;
        (*a_f16)[i] = __float2half(v);
    }
    for (size_t i = 0; i < w_f32->size(); ++i) {
        (*w_f32)[i] = deterministic_w(i);
    }
    for (int row = 0; row < s.n; ++row) {
        for (int block = 0; block < s.k / qk; ++block) {
            float max_abs = 0.0f;
            for (int kk = 0; kk < qk; ++kk) {
                const int k_idx = block * qk + kk;
                max_abs = std::max(max_abs, std::fabs((*w_f32)[(size_t) row * s.k + k_idx]));
            }
            const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            (*w_scales)[(size_t) row * (s.k / qk) + block] = __float2half(scale);
            for (int kk = 0; kk < qk; ++kk) {
                const int k_idx = block * qk + kk;
                int q = (int) std::lrintf((*w_f32)[(size_t) row * s.k + k_idx] / scale);
                q = std::max(-127, std::min(127, q));
                (*w_qs)[(size_t) row * s.k + k_idx] = (int8_t) q;
                const float deq = (float) q * scale;
                (*w_f16)[(size_t) row * s.k + k_idx] = __float2half(deq);
            }
        }
    }
}

static void reference_int8(const Shape & s,
                           const std::vector<float> & a,
                           const std::vector<int8_t> & w_qs,
                           const std::vector<__half> & w_scales,
                           std::vector<float> * ref) {
    ref->assign((size_t) s.m * (size_t) s.n, 0.0f);
    const int scale_cols = s.k / 32;
    for (int row = 0; row < s.m; ++row) {
        for (int col = 0; col < s.n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < s.k; ++kk) {
                const float scale = __half2float(w_scales[(size_t) col * scale_cols + (kk / 32)]);
                const float w = (float) w_qs[(size_t) col * s.k + kk] * scale;
                acc += a[(size_t) row * s.k + kk] * w;
            }
            (*ref)[(size_t) row * s.n + col] = acc;
        }
    }
}

static void error_stats(const std::vector<float> & got,
                        const std::vector<float> & ref,
                        float * max_abs,
                        float * p99_abs,
                        double * mean_abs) {
    std::vector<float> errs(got.size());
    *max_abs = 0.0f;
    *mean_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float e = std::fabs(got[i] - ref[i]);
        errs[i] = e;
        *max_abs = std::max(*max_abs, e);
        *mean_abs += e;
    }
    *mean_abs /= (double) got.size();
    std::sort(errs.begin(), errs.end());
    size_t p99 = (size_t) ((double) errs.size() * 0.99);
    if (p99 >= errs.size()) {
        p99 = errs.size() - 1;
    }
    *p99_abs = errs[p99];
}

static void launch_v12s(const int8_t * d_w_qs,
                        const __half * d_w_scales,
                        const float * d_a,
                        float * d_c,
                        const Shape & s,
                        cudaStream_t stream) {
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int WARPS = 4;
    constexpr int ATOMS_M = 8;
    constexpr int ATOMS_N = 1;
    constexpr int KS = 8;
    const dim3 block(WARPS * 32);
    const dim3 grid((unsigned) ((s.n + BN - 1) / BN), (unsigned) ((s.m + BM - 1) / BM), KS);
    const size_t smem_bytes = 28672;
    tc_grid::kernels::int8_v12::mm_int8_lut_v12s<BM, BN, BK, WARPS, ATOMS_M, ATOMS_N, KS>
        <<<grid, block, smem_bytes, stream>>>(d_w_qs, d_w_scales, d_a, d_c, s.m, s.n, s.k);
}

static void launch_v13(const int8_t * d_w_qs,
                       const __half * d_w_scales,
                       const float * d_a,
                       float * d_c,
                       const Shape & s,
                       cudaStream_t stream) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    const dim3 block(128);
    const dim3 grid((unsigned) ((s.n + BN - 1) / BN), (unsigned) ((s.m + BM - 1) / BM));
    const size_t smem_bytes = 34816;
    tc_grid::kernels::int8_v13::mm_int8_lut_v13_rf_v6<BM, BN, BK, 16>
        <<<grid, block, smem_bytes, stream>>>(d_w_qs, d_w_scales, d_a, d_c, s.m, s.n, s.k);
}

static Result run_int8_case(const Shape & s,
                            const char * label,
                            bool needs_zero,
                            void (*launcher)(const int8_t *, const __half *, const float *, float *, const Shape &, cudaStream_t),
                            const std::vector<float> & a,
                            const std::vector<int8_t> & w_qs,
                            const std::vector<__half> & w_scales,
                            const std::vector<float> & ref,
                            const Options & opt) {
    Result r;
    r.m = s.m;
    r.n = s.n;
    r.k = s.k;
    r.label = label;

    float * d_a = nullptr;
    int8_t * d_w_qs = nullptr;
    __half * d_w_scales = nullptr;
    float * d_c = nullptr;
    const size_t a_bytes = a.size() * sizeof(float);
    const size_t w_bytes = w_qs.size() * sizeof(int8_t);
    const size_t scale_bytes = w_scales.size() * sizeof(__half);
    const size_t c_bytes = (size_t) s.m * s.n * sizeof(float);
    DS4_CUDA_CHECK(cudaMalloc((void **) &d_a, a_bytes));
    DS4_CUDA_CHECK(cudaMalloc((void **) &d_w_qs, w_bytes));
    DS4_CUDA_CHECK(cudaMalloc((void **) &d_w_scales, scale_bytes));
    DS4_CUDA_CHECK(cudaMalloc((void **) &d_c, c_bytes));
    DS4_CUDA_CHECK(cudaMemcpy(d_a, a.data(), a_bytes, cudaMemcpyHostToDevice));
    DS4_CUDA_CHECK(cudaMemcpy(d_w_qs, w_qs.data(), w_bytes, cudaMemcpyHostToDevice));
    DS4_CUDA_CHECK(cudaMemcpy(d_w_scales, w_scales.data(), scale_bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < opt.warmup; ++i) {
        if (needs_zero) {
            DS4_CUDA_CHECK(cudaMemsetAsync(d_c, 0, c_bytes, nullptr));
        }
        launcher(d_w_qs, d_w_scales, d_a, d_c, s, nullptr);
    }
    DS4_CUDA_CHECK(cudaGetLastError());
    DS4_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    DS4_CUDA_CHECK(cudaEventCreate(&start));
    DS4_CUDA_CHECK(cudaEventCreate(&stop));
    DS4_CUDA_CHECK(cudaEventRecord(start, nullptr));
    for (int i = 0; i < opt.iters; ++i) {
        if (needs_zero) {
            DS4_CUDA_CHECK(cudaMemsetAsync(d_c, 0, c_bytes, nullptr));
        }
        launcher(d_w_qs, d_w_scales, d_a, d_c, s, nullptr);
    }
    DS4_CUDA_CHECK(cudaEventRecord(stop, nullptr));
    DS4_CUDA_CHECK(cudaEventSynchronize(stop));
    DS4_CUDA_CHECK(cudaGetLastError());
    float elapsed_ms = 0.0f;
    DS4_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    DS4_CUDA_CHECK(cudaEventDestroy(start));
    DS4_CUDA_CHECK(cudaEventDestroy(stop));

    std::vector<float> got((size_t) s.m * s.n);
    DS4_CUDA_CHECK(cudaMemcpy(got.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));
    error_stats(got, ref, &r.max_abs, &r.p99_abs, &r.mean_abs);
    r.ok = r.max_abs <= 0.02f && r.p99_abs <= 0.02f;
    r.ms = (double) elapsed_ms / (double) opt.iters;
    const double flops = 2.0 * (double) s.m * s.n * s.k;
    r.tflops = flops / (r.ms * 1.0e9);
    const double bytes = (double) a_bytes + (double) w_bytes + (double) scale_bytes + (double) c_bytes;
    r.gbps = bytes / (r.ms * 1.0e6);

    DS4_CUDA_CHECK(cudaFree(d_a));
    DS4_CUDA_CHECK(cudaFree(d_w_qs));
    DS4_CUDA_CHECK(cudaFree(d_w_scales));
    DS4_CUDA_CHECK(cudaFree(d_c));
    return r;
}

static Result run_cublas_f16_case(const Shape & s,
                                  const std::vector<__half> & a,
                                  const std::vector<__half> & w,
                                  const std::vector<float> & ref,
                                  const Options & opt) {
    Result r;
    r.m = s.m;
    r.n = s.n;
    r.k = s.k;
    r.label = "cublas-f16-tensorop";

    __half * d_a = nullptr;
    __half * d_w = nullptr;
    float * d_c = nullptr;
    const size_t a_bytes = a.size() * sizeof(__half);
    const size_t w_bytes = w.size() * sizeof(__half);
    const size_t c_bytes = (size_t) s.m * s.n * sizeof(float);
    DS4_CUDA_CHECK(cudaMalloc((void **) &d_a, a_bytes));
    DS4_CUDA_CHECK(cudaMalloc((void **) &d_w, w_bytes));
    DS4_CUDA_CHECK(cudaMalloc((void **) &d_c, c_bytes));
    DS4_CUDA_CHECK(cudaMemcpy(d_a, a.data(), a_bytes, cudaMemcpyHostToDevice));
    DS4_CUDA_CHECK(cudaMemcpy(d_w, w.data(), w_bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle = nullptr;
    DS4_CUBLAS_CHECK(cublasCreate(&handle));
    DS4_CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto gemm = [&]() {
        // Row-major C[M,N] = A[M,K] * W[N,K]^T is column-major
        // C_col[N,M] = W_col[K,N]^T * A_col[K,M].
        DS4_CUBLAS_CHECK(cublasGemmEx(handle,
                                      CUBLAS_OP_T,
                                      CUBLAS_OP_N,
                                      s.n,
                                      s.m,
                                      s.k,
                                      &alpha,
                                      d_w,
                                      CUDA_R_16F,
                                      s.k,
                                      d_a,
                                      CUDA_R_16F,
                                      s.k,
                                      &beta,
                                      d_c,
                                      CUDA_R_32F,
                                      s.n,
                                      CUDA_R_32F,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    for (int i = 0; i < opt.warmup; ++i) {
        gemm();
    }
    DS4_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    DS4_CUDA_CHECK(cudaEventCreate(&start));
    DS4_CUDA_CHECK(cudaEventCreate(&stop));
    DS4_CUDA_CHECK(cudaEventRecord(start, nullptr));
    for (int i = 0; i < opt.iters; ++i) {
        gemm();
    }
    DS4_CUDA_CHECK(cudaEventRecord(stop, nullptr));
    DS4_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    DS4_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    DS4_CUDA_CHECK(cudaEventDestroy(start));
    DS4_CUDA_CHECK(cudaEventDestroy(stop));

    std::vector<float> got((size_t) s.m * s.n);
    DS4_CUDA_CHECK(cudaMemcpy(got.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));
    error_stats(got, ref, &r.max_abs, &r.p99_abs, &r.mean_abs);
    r.ok = r.max_abs <= 0.08f && r.p99_abs <= 0.03f;
    r.ms = (double) elapsed_ms / (double) opt.iters;
    const double flops = 2.0 * (double) s.m * s.n * s.k;
    r.tflops = flops / (r.ms * 1.0e9);
    const double bytes = (double) a_bytes + (double) w_bytes + (double) c_bytes;
    r.gbps = bytes / (r.ms * 1.0e6);

    DS4_CUBLAS_CHECK(cublasDestroy(handle));
    DS4_CUDA_CHECK(cudaFree(d_a));
    DS4_CUDA_CHECK(cudaFree(d_w));
    DS4_CUDA_CHECK(cudaFree(d_c));
    return r;
}

static void write_outputs(const Options & opt, const std::vector<Result> & results) {
    FILE * tsv = std::fopen(opt.out_tsv.c_str(), "w");
    if (!tsv) {
        std::fprintf(stderr, "failed to open TSV %s: %s\n", opt.out_tsv.c_str(), std::strerror(errno));
        std::exit(1);
    }
    std::fprintf(tsv, "M\tN\tK\tlabel\tms\ttflops\tgbps\tmax_abs\tp99_abs\tmean_abs\tok\n");
    for (const Result & r : results) {
        std::fprintf(tsv, "%d\t%d\t%d\t%s\t%.9f\t%.6f\t%.6f\t%.9g\t%.9g\t%.9g\t%d\n",
                     r.m, r.n, r.k, r.label, r.ms, r.tflops, r.gbps,
                     r.max_abs, r.p99_abs, r.mean_abs, r.ok ? 1 : 0);
    }
    std::fclose(tsv);

    FILE * md = std::fopen(opt.report.c_str(), "w");
    if (!md) {
        std::fprintf(stderr, "failed to open report %s: %s\n", opt.report.c_str(), std::strerror(errno));
        std::exit(1);
    }
    std::fprintf(md, "# INT8 Compressor Workbench\n\n");
    std::fprintf(md, "Target shapes: `M=32,N=128/64,K=4096`.\n\n");
    std::fprintf(md, "| M | N | K | Kernel | ms | TFLOP/s | GB/s | max abs | p99 abs | mean abs | OK |\n");
    std::fprintf(md, "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for (const Result & r : results) {
        std::fprintf(md, "| %d | %d | %d | `%s` | %.6f | %.3f | %.3f | %.6g | %.6g | %.6g | %d |\n",
                     r.m, r.n, r.k, r.label, r.ms, r.tflops, r.gbps,
                     r.max_abs, r.p99_abs, r.mean_abs, r.ok ? 1 : 0);
    }
    std::fprintf(md, "\nNotes:\n\n");
    std::fprintf(md, "- `tc-grid-v12s-ks8+zero` includes the required output zeroing for split-K atomic accumulation.\n");
    std::fprintf(md, "- The cuBLAS baseline uses FP16 tensor-op inputs and FP32 output as the BF16-on-V100 proxy.\n");
    std::fprintf(md, "- INT8 inputs use the tc-grid contract: FP32 activations, INT8 weights, FP16 per-row/per-32K scales.\n");
    std::fclose(md);
}

int main(int argc, char ** argv) {
    const Options opt = parse_options(argc, argv);
    std::vector<Result> results;
    const Shape shapes[] = {{32, 128, 4096}, {32, 64, 4096}};

    int device = 0;
    DS4_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    DS4_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("device=%d name=%s sm=%d%d warmup=%d iters=%d\n",
                device, prop.name, prop.major, prop.minor, opt.warmup, opt.iters);

    for (const Shape & s : shapes) {
        std::vector<float> a_f32;
        std::vector<__half> a_f16;
        std::vector<float> w_f32;
        std::vector<__half> w_f16;
        std::vector<int8_t> w_qs;
        std::vector<__half> w_scales;
        std::vector<float> ref;
        init_shape(s, &a_f32, &a_f16, &w_f32, &w_f16, &w_qs, &w_scales);
        reference_int8(s, a_f32, w_qs, w_scales, &ref);

        results.push_back(run_cublas_f16_case(s, a_f16, w_f16, ref, opt));
        results.push_back(run_int8_case(s, "tc-grid-v12s-ks8+zero", true, launch_v12s,
                                        a_f32, w_qs, w_scales, ref, opt));
        results.push_back(run_int8_case(s, "tc-grid-v13-rf-v6", false, launch_v13,
                                        a_f32, w_qs, w_scales, ref, opt));
        for (size_t i = results.size() - 3; i < results.size(); ++i) {
            const Result & r = results[i];
            std::printf("shape=%dx%dx%d label=%s ms=%.6f tflops=%.3f gbps=%.3f max_abs=%.6g p99_abs=%.6g ok=%d\n",
                        r.m, r.n, r.k, r.label, r.ms, r.tflops, r.gbps,
                        r.max_abs, r.p99_abs, r.ok ? 1 : 0);
        }
    }

    write_outputs(opt, results);
    return std::all_of(results.begin(), results.end(), [](const Result & r) { return r.ok; }) ? 0 : 1;
}
