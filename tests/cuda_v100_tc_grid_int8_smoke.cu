#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "dispatch.h"
#include "v13_kernels.cuh"

#define DS4_TCGRID_CHECK(call)                                                \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",             \
                         cudaGetErrorString(err__), __FILE__, __LINE__,       \
                         #call);                                             \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

static void init_inputs(int m, int n, int k,
                        std::vector<float> * a,
                        std::vector<int8_t> * w_qs,
                        std::vector<__half> * w_scales) {
    a->resize((size_t) m * (size_t) k);
    w_qs->resize((size_t) n * (size_t) k);
    w_scales->resize((size_t) n * (size_t) (k / 32));

    for (size_t i = 0; i < a->size(); ++i) {
        const int v = (int) ((i * 17u + 3u) % 31u) - 15;
        (*a)[i] = (float) v * (1.0f / 256.0f);
    }
    for (size_t i = 0; i < w_qs->size(); ++i) {
        const int v = (int) ((i * 13u + 7u) % 9u) - 4;
        (*w_qs)[i] = (int8_t) v;
    }
    for (size_t i = 0; i < w_scales->size(); ++i) {
        (*w_scales)[i] = __float2half(1.0f / 32.0f);
    }
}

static void reference_int8_gemm(int m, int n, int k,
                                const std::vector<float> & a,
                                const std::vector<int8_t> & w_qs,
                                const std::vector<__half> & w_scales,
                                std::vector<float> * ref) {
    ref->assign((size_t) m * (size_t) n, 0.0f);
    const int scale_cols = k / 32;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                const float scale =
                    __half2float(w_scales[(size_t) col * (size_t) scale_cols + (size_t) (kk / 32)]);
                const float w = (float) w_qs[(size_t) col * (size_t) k + (size_t) kk] * scale;
                acc += a[(size_t) row * (size_t) k + (size_t) kk] * w;
            }
            (*ref)[(size_t) row * (size_t) n + (size_t) col] = acc;
        }
    }
}

static void launch_v13_rf_v6(const int8_t * d_w_qs,
                             const __half * d_w_scales,
                             const float * d_a,
                             float * d_c,
                             int m, int n, int k,
                             cudaStream_t stream) {
    const dim3 block(128);
    const dim3 grid((unsigned) ((n + 127) / 128), (unsigned) ((m + 127) / 128));
    const size_t smem_bytes = 34816;
    tc_grid::kernels::int8_v13::mm_int8_lut_v13_rf_v6<128, 128, 16, 16>
        <<<grid, block, smem_bytes, stream>>>(d_w_qs, d_w_scales, d_a, d_c, m, n, k);
}

static void allocate_and_upload(int m, int n, int k,
                                float ** d_a,
                                int8_t ** d_w_qs,
                                __half ** d_w_scales,
                                float ** d_c) {
    std::vector<float> a;
    std::vector<int8_t> w_qs;
    std::vector<__half> w_scales;
    init_inputs(m, n, k, &a, &w_qs, &w_scales);

    DS4_TCGRID_CHECK(cudaMalloc((void **) d_a, a.size() * sizeof(float)));
    DS4_TCGRID_CHECK(cudaMalloc((void **) d_w_qs, w_qs.size() * sizeof(int8_t)));
    DS4_TCGRID_CHECK(cudaMalloc((void **) d_w_scales, w_scales.size() * sizeof(__half)));
    DS4_TCGRID_CHECK(cudaMalloc((void **) d_c, (size_t) m * (size_t) n * sizeof(float)));
    DS4_TCGRID_CHECK(cudaMemcpy(*d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    DS4_TCGRID_CHECK(cudaMemcpy(*d_w_qs, w_qs.data(), w_qs.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    DS4_TCGRID_CHECK(cudaMemcpy(*d_w_scales, w_scales.data(), w_scales.size() * sizeof(__half), cudaMemcpyHostToDevice));
    DS4_TCGRID_CHECK(cudaMemset(*d_c, 0, (size_t) m * (size_t) n * sizeof(float)));
}

static int run_correctness() {
    const int m = 128;
    const int n = 128;
    const int k = 128;
    std::vector<float> a;
    std::vector<int8_t> w_qs;
    std::vector<__half> w_scales;
    std::vector<float> ref;
    init_inputs(m, n, k, &a, &w_qs, &w_scales);
    reference_int8_gemm(m, n, k, a, w_qs, w_scales, &ref);

    float * d_a = nullptr;
    int8_t * d_w_qs = nullptr;
    __half * d_w_scales = nullptr;
    float * d_c = nullptr;
    DS4_TCGRID_CHECK(cudaMalloc((void **) &d_a, a.size() * sizeof(float)));
    DS4_TCGRID_CHECK(cudaMalloc((void **) &d_w_qs, w_qs.size() * sizeof(int8_t)));
    DS4_TCGRID_CHECK(cudaMalloc((void **) &d_w_scales, w_scales.size() * sizeof(__half)));
    DS4_TCGRID_CHECK(cudaMalloc((void **) &d_c, (size_t) m * (size_t) n * sizeof(float)));
    DS4_TCGRID_CHECK(cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    DS4_TCGRID_CHECK(cudaMemcpy(d_w_qs, w_qs.data(), w_qs.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    DS4_TCGRID_CHECK(cudaMemcpy(d_w_scales, w_scales.data(), w_scales.size() * sizeof(__half), cudaMemcpyHostToDevice));
    DS4_TCGRID_CHECK(cudaMemset(d_c, 0, (size_t) m * (size_t) n * sizeof(float)));

    launch_v13_rf_v6(d_w_qs, d_w_scales, d_a, d_c, m, n, k, nullptr);
    DS4_TCGRID_CHECK(cudaGetLastError());
    DS4_TCGRID_CHECK(cudaDeviceSynchronize());

    std::vector<float> got((size_t) m * (size_t) n);
    DS4_TCGRID_CHECK(cudaMemcpy(got.data(), d_c, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    DS4_TCGRID_CHECK(cudaFree(d_a));
    DS4_TCGRID_CHECK(cudaFree(d_w_qs));
    DS4_TCGRID_CHECK(cudaFree(d_w_scales));
    DS4_TCGRID_CHECK(cudaFree(d_c));

    std::vector<float> abs_err(got.size());
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        abs_err[i] = std::fabs(got[i] - ref[i]);
        max_abs = std::max(max_abs, abs_err[i]);
        mean_abs += abs_err[i];
    }
    mean_abs /= (double) got.size();
    std::sort(abs_err.begin(), abs_err.end());
    size_t p99_idx = (size_t) ((double) abs_err.size() * 0.99);
    if (p99_idx >= abs_err.size()) {
        p99_idx = abs_err.size() - 1;
    }
    const float p99_abs = abs_err[p99_idx];
    const int ok = (max_abs <= 0.08f && p99_abs <= 0.03f);
    std::printf("tc_grid_int8_correctness M=%d N=%d K=%d max_abs=%.8g p99_abs=%.8g mean_abs=%.8g %s\n",
                m, n, k, max_abs, p99_abs, mean_abs, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

static int run_benchmark(int m, int n, int k, int iters) {
    const auto spec = tc_grid::dispatch::choose_kernel(m, n, k);
    if (spec.version != 88 || spec.BM != 128 || spec.BN != 128 || spec.BK != 16) {
        std::fprintf(stderr, "unexpected tc-grid dispatch label=%s version=%d BM=%d BN=%d BK=%d\n",
                     spec.label, spec.version, spec.BM, spec.BN, spec.BK);
        return 1;
    }

    float * d_a = nullptr;
    int8_t * d_w_qs = nullptr;
    __half * d_w_scales = nullptr;
    float * d_c = nullptr;
    allocate_and_upload(m, n, k, &d_a, &d_w_qs, &d_w_scales, &d_c);

    for (int i = 0; i < 10; ++i) {
        launch_v13_rf_v6(d_w_qs, d_w_scales, d_a, d_c, m, n, k, nullptr);
    }
    DS4_TCGRID_CHECK(cudaGetLastError());
    DS4_TCGRID_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    DS4_TCGRID_CHECK(cudaEventCreate(&start));
    DS4_TCGRID_CHECK(cudaEventCreate(&stop));
    DS4_TCGRID_CHECK(cudaEventRecord(start, nullptr));
    for (int i = 0; i < iters; ++i) {
        launch_v13_rf_v6(d_w_qs, d_w_scales, d_a, d_c, m, n, k, nullptr);
    }
    DS4_TCGRID_CHECK(cudaEventRecord(stop, nullptr));
    DS4_TCGRID_CHECK(cudaEventSynchronize(stop));
    DS4_TCGRID_CHECK(cudaGetLastError());
    float elapsed_ms = 0.0f;
    DS4_TCGRID_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    DS4_TCGRID_CHECK(cudaEventDestroy(start));
    DS4_TCGRID_CHECK(cudaEventDestroy(stop));

    const double ms_mean = (double) elapsed_ms / (double) iters;
    const double flops = 2.0 * (double) m * (double) n * (double) k;
    const double tflops = flops / (ms_mean * 1.0e9);
    const double bytes =
        (double) m * (double) k * sizeof(float) +
        (double) n * (double) k * sizeof(int8_t) +
        (double) n * (double) (k / 32) * sizeof(__half) +
        (double) m * (double) n * sizeof(float);
    const double gbps = bytes / (ms_mean * 1.0e6);
    std::printf("tc_grid_int8_bench label=%s M=%d N=%d K=%d iters=%d ms=%.6f tflops=%.3f gbps=%.3f ok\n",
                spec.label, m, n, k, iters, ms_mean, tflops, gbps);

    DS4_TCGRID_CHECK(cudaFree(d_a));
    DS4_TCGRID_CHECK(cudaFree(d_w_qs));
    DS4_TCGRID_CHECK(cudaFree(d_w_scales));
    DS4_TCGRID_CHECK(cudaFree(d_c));
    return 0;
}

int main(int argc, char ** argv) {
    int m = 128;
    int n = 2048;
    int k = 4096;
    int iters = 100;
    if (argc >= 5) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
        iters = std::atoi(argv[4]);
    }
    if (m <= 0 || n <= 0 || k <= 0 || iters <= 0 || (k % 32) != 0) {
        std::fprintf(stderr, "usage: %s [M N K iters], with positive values and K %% 32 == 0\n", argv[0]);
        return 2;
    }
    int failures = run_correctness();
    failures += run_benchmark(m, n, k, iters);
    return failures == 0 ? 0 : 1;
}
