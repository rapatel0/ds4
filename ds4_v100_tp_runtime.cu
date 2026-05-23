#include "ds4_v100_tp_runtime.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kGpus = 8;
constexpr int kLayers = 43;
constexpr int kSwa = 128;
constexpr int kHeadDim = 512;
constexpr int kIndexerHeadDim = 128;

struct gpu_state {
    int device = 0;
    half *hidden_in = nullptr;
    half *hidden_out = nullptr;
    void *kv = nullptr;
    void *comp_state = nullptr;
    void *scratch = nullptr;
    ds4_v100_tp_gpu_report report = {};
};

} // namespace

struct ds4_v100_tp_runtime {
    ds4_v100_tp_runtime_config cfg = {};
    gpu_state gpu[kGpus];
};

namespace {

static void set_err(char *err, size_t err_len, const char *msg) {
    if (err && err_len) {
        std::snprintf(err, err_len, "%s", msg);
    }
}

static int fail_cuda(char *err, size_t err_len, const char *what, cudaError_t rc) {
    if (err && err_len) {
        std::snprintf(err, err_len, "%s: %s", what, cudaGetErrorString(rc));
    }
    return -1;
}

static uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a) {
        std::fprintf(stderr, "ds4_v100_tp_runtime: integer overflow\n");
        std::abort();
    }
    return a * b;
}

static uint64_t ceil_div(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

static uint64_t bytes_blocks(uint64_t elems, uint64_t block_elems, uint64_t block_bytes) {
    return checked_mul((elems + block_elems - 1) / block_elems, block_bytes);
}

static uint64_t kv_values_bytes(uint64_t values, ds4_v100_tp_kv_dtype kv) {
    switch (kv) {
    case DS4_V100_TP_KV_F16:
        return checked_mul(values, 2);
    case DS4_V100_TP_KV_F8_E4M3_B128:
        return bytes_blocks(values, 128, 129);
    case DS4_V100_TP_KV_Q8_0:
        return bytes_blocks(values, 32, 34);
    }
    return checked_mul(values, 2);
}

static int layer_ratio(int layer) {
    if (layer < 2) return 0;
    return (layer % 2) == 0 ? 4 : 128;
}

static uint64_t layer_attn_kv_bytes(int layer, uint64_t ctx, ds4_v100_tp_kv_dtype kv) {
    const int ratio = layer_ratio(layer);
    const uint64_t rows = (uint64_t)kSwa + (ratio ? ctx / (uint64_t)ratio : 0);
    return kv_values_bytes(checked_mul(rows, kHeadDim), kv);
}

static uint64_t layer_indexer_kv_bytes(int layer, uint64_t ctx, ds4_v100_tp_kv_dtype kv) {
    if (layer_ratio(layer) != 4) return 0;
    return kv_values_bytes(checked_mul(ctx / 4u, kIndexerHeadDim), kv);
}

static uint64_t layer_comp_state_bytes(int layer, uint64_t ctx) {
    const int ratio = layer_ratio(layer);
    if (!ratio) return 0;
    const uint64_t attn = checked_mul(ctx / (uint64_t)ratio, kHeadDim);
    if (ratio == 4) {
        const uint64_t indexer = checked_mul(ctx / 4u, kIndexerHeadDim);
        return checked_mul((attn + indexer) / 8u, 4u);
    }
    return checked_mul(attn / 8u, 4u);
}

static void planned_kv_bytes(const ds4_v100_tp_runtime_config *cfg,
                             uint64_t *kv_per_gpu,
                             uint64_t *comp_per_gpu) {
    uint64_t kv = 0;
    uint64_t comp = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        kv += layer_attn_kv_bytes(layer, cfg->ctx, cfg->kv_dtype);
        kv += layer_indexer_kv_bytes(layer, cfg->ctx, cfg->kv_dtype);
        comp += layer_comp_state_bytes(layer, cfg->ctx);
    }
    *kv_per_gpu = checked_mul(ceil_div(kv, kGpus), cfg->slots);
    *comp_per_gpu = checked_mul(ceil_div(comp, kGpus), cfg->slots);
}

__global__ void fixture_kernel(const half *in, half *out, half *scratch,
                               size_t hidden_elems, size_t scratch_elems,
                               float value) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_elems) {
        const float v = __half2float(in[i]) + value;
        out[i] = __float2half(v);
    }
    if (i < scratch_elems) {
        scratch[i] = __float2half(value);
    }
}

} // namespace

extern "C" void ds4_v100_tp_runtime_default_config(ds4_v100_tp_runtime_config *cfg) {
    std::memset(cfg, 0, sizeof(*cfg));
    for (int i = 0; i < kGpus; ++i) cfg->devices[i] = i;
    cfg->slots = 32;
    cfg->ctx = 262144;
    cfg->hidden = 4096;
    cfg->kv_dtype = DS4_V100_TP_KV_F8_E4M3_B128;
    cfg->scratch_bytes = 1536ull * 1024ull * 1024ull;
}

extern "C" int ds4_v100_tp_runtime_open(ds4_v100_tp_runtime **out,
                                         const ds4_v100_tp_runtime_config *cfg,
                                         char *err,
                                         size_t err_len) {
    if (!out || !cfg) {
        set_err(err, err_len, "invalid argument");
        return -1;
    }
    *out = nullptr;

    int device_count = 0;
    cudaError_t rc = cudaGetDeviceCount(&device_count);
    if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaGetDeviceCount", rc);
    for (int i = 0; i < kGpus; ++i) {
        if (cfg->devices[i] < 0 || cfg->devices[i] >= device_count) {
            set_err(err, err_len, "configured device is outside visible device count");
            return -1;
        }
    }

    ds4_v100_tp_runtime *rt = new ds4_v100_tp_runtime();
    rt->cfg = *cfg;

    uint64_t kv_per_gpu = 0;
    uint64_t comp_per_gpu = 0;
    planned_kv_bytes(cfg, &kv_per_gpu, &comp_per_gpu);
    const uint64_t hidden_bytes =
        checked_mul(checked_mul((uint64_t)cfg->slots, cfg->hidden), sizeof(half));

    for (int i = 0; i < kGpus; ++i) {
        rc = cudaSetDevice(cfg->devices[i]);
        if (rc != cudaSuccess) {
            ds4_v100_tp_runtime_close(rt);
            return fail_cuda(err, err_len, "cudaSetDevice", rc);
        }
        for (int j = 0; j < kGpus; ++j) {
            if (i == j) continue;
            int can_access = 0;
            rc = cudaDeviceCanAccessPeer(&can_access, cfg->devices[i], cfg->devices[j]);
            if (rc != cudaSuccess) {
                ds4_v100_tp_runtime_close(rt);
                return fail_cuda(err, err_len, "cudaDeviceCanAccessPeer", rc);
            }
            if (!can_access) {
                ds4_v100_tp_runtime_close(rt);
                set_err(err, err_len, "peer access unavailable");
                return -1;
            }
            rc = cudaDeviceEnablePeerAccess(cfg->devices[j], 0);
            if (rc == cudaErrorPeerAccessAlreadyEnabled) {
                (void)cudaGetLastError();
            } else if (rc != cudaSuccess) {
                ds4_v100_tp_runtime_close(rt);
                return fail_cuda(err, err_len, "cudaDeviceEnablePeerAccess", rc);
            }
        }
    }

    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        g->device = cfg->devices[i];
        rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) {
            ds4_v100_tp_runtime_close(rt);
            return fail_cuda(err, err_len, "cudaSetDevice", rc);
        }
        if ((rc = cudaMalloc(&g->hidden_in, hidden_bytes)) != cudaSuccess ||
            (rc = cudaMalloc(&g->hidden_out, hidden_bytes)) != cudaSuccess ||
            (rc = cudaMalloc(&g->kv, kv_per_gpu)) != cudaSuccess ||
            (rc = cudaMalloc(&g->comp_state, comp_per_gpu)) != cudaSuccess ||
            (rc = cudaMalloc(&g->scratch, cfg->scratch_bytes)) != cudaSuccess) {
            ds4_v100_tp_runtime_close(rt);
            return fail_cuda(err, err_len, "cudaMalloc", rc);
        }
        rc = cudaMemset(g->hidden_in, 0, hidden_bytes);
        if (rc != cudaSuccess) {
            ds4_v100_tp_runtime_close(rt);
            return fail_cuda(err, err_len, "cudaMemset", rc);
        }
        g->report.hidden_bytes = 2 * hidden_bytes;
        g->report.kv_bytes = kv_per_gpu;
        g->report.comp_state_bytes = comp_per_gpu;
        g->report.scratch_bytes = cfg->scratch_bytes;
        g->report.total_bytes =
            g->report.hidden_bytes + g->report.kv_bytes +
            g->report.comp_state_bytes + g->report.scratch_bytes;
    }

    *out = rt;
    return 0;
}

extern "C" int ds4_v100_tp_runtime_fixture(ds4_v100_tp_runtime *rt,
                                            double *max_abs,
                                            char *err,
                                            size_t err_len) {
    if (!rt) {
        set_err(err, err_len, "runtime is null");
        return -1;
    }
    const size_t hidden_elems = (size_t)rt->cfg.slots * (size_t)rt->cfg.hidden;
    const size_t scratch_elems =
        std::min((size_t)(rt->cfg.scratch_bytes / sizeof(half)), hidden_elems);
    const int block = 256;
    const int grid = (int)((std::max(hidden_elems, scratch_elems) + block - 1) / block);
    double worst = 0.0;

    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);
        const float value = (float)(i + 1);
        fixture_kernel<<<grid, block>>>(g->hidden_in, g->hidden_out, (half *)g->scratch,
                                        hidden_elems, scratch_elems, value);
        rc = cudaGetLastError();
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "fixture_kernel", rc);
    }
    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        cudaError_t rc = cudaSetDevice(g->device);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaSetDevice", rc);
        rc = cudaDeviceSynchronize();
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaDeviceSynchronize", rc);

        std::vector<half> host(hidden_elems);
        rc = cudaMemcpy(host.data(), g->hidden_out, hidden_elems * sizeof(half),
                        cudaMemcpyDeviceToHost);
        if (rc != cudaSuccess) return fail_cuda(err, err_len, "cudaMemcpy", rc);
        const float expected = (float)(i + 1);
        for (size_t j = 0; j < hidden_elems; ++j) {
            const float got = __half2float(host[j]);
            if (!std::isfinite(got)) {
                set_err(err, err_len, "non-finite fixture output");
                return -1;
            }
            worst = std::max(worst, (double)std::fabs(got - expected));
        }
    }
    if (max_abs) *max_abs = worst;
    return 0;
}

extern "C" void ds4_v100_tp_runtime_get_report(const ds4_v100_tp_runtime *rt,
                                                ds4_v100_tp_runtime_report *report) {
    std::memset(report, 0, sizeof(*report));
    if (!rt) return;
    for (int i = 0; i < kGpus; ++i) {
        report->gpu[i] = rt->gpu[i].report;
    }
}

extern "C" void ds4_v100_tp_runtime_close(ds4_v100_tp_runtime *rt) {
    if (!rt) return;
    for (int i = 0; i < kGpus; ++i) {
        gpu_state *g = &rt->gpu[i];
        cudaSetDevice(g->device);
        cudaFree(g->hidden_in);
        cudaFree(g->hidden_out);
        cudaFree(g->kv);
        cudaFree(g->comp_state);
        cudaFree(g->scratch);
    }
    delete rt;
}
