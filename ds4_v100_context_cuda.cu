#include "ds4_v100_context.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int gpu;
    cudaStream_t stream;
    cudaStream_t relay_stream;
    cublasHandle_t cublas;
    void *scratch;
    uint64_t scratch_bytes;
    void *relay_f16_in;
    void *relay_f16_out;
    uint64_t relay_f16_bytes;
    void *relay_f32_in;
    void *relay_f32_out;
    uint64_t relay_f32_bytes;
} ds4_v100_cuda_stage;

struct ds4_v100_cuda_context {
    ds4_v100_context *host;
    int n_stages;
    uint64_t relay_max_active_slots;
    ds4_v100_cuda_stage stages[DS4_V100_EXPECTED_GPUS];
    bool peer_access[DS4_V100_EXPECTED_GPUS][DS4_V100_EXPECTED_GPUS];
};

static int cuda_v100_error(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return 1;
}

static int cuda_v100_ok(cudaError_t rc, const char *what, char *err, size_t errlen) {
    if (rc == cudaSuccess) return 0;
    return cuda_v100_error(err, errlen, "%s: %s", what, cudaGetErrorString(rc));
}

static int cublas_v100_ok(cublasStatus_t rc, const char *what, char *err, size_t errlen) {
    if (rc == CUBLAS_STATUS_SUCCESS) return 0;
    return cuda_v100_error(err, errlen, "%s: cublas status %d", what, (int)rc);
}

int ds4_v100_cuda_collect_device_facts(ds4_v100_device_fact *facts,
                                       int fact_cap,
                                       int *out_count,
                                       char *err,
                                       size_t errlen) {
    if (!facts || fact_cap <= 0 || !out_count) {
        return cuda_v100_error(err, errlen, "missing device fact output");
    }
    int n = 0;
    cudaError_t rc = cudaGetDeviceCount(&n);
    if (rc != cudaSuccess) {
        return cuda_v100_ok(rc, "cudaGetDeviceCount", err, errlen);
    }
    if (n > fact_cap) n = fact_cap;
    memset(facts, 0, sizeof(facts[0]) * (size_t)n);
    for (int i = 0; i < n; i++) {
        cudaDeviceProp prop;
        rc = cudaGetDeviceProperties(&prop, i);
        if (rc != cudaSuccess) return cuda_v100_ok(rc, "cudaGetDeviceProperties", err, errlen);
        facts[i].visible_id = i;
        facts[i].cc_major = prop.major;
        facts[i].cc_minor = prop.minor;
        facts[i].total_global_mem = (uint64_t)prop.totalGlobalMem;
        char bus[32];
        rc = cudaDeviceGetPCIBusId(bus, sizeof(bus), i);
        if (rc == cudaSuccess) {
            snprintf(facts[i].pci_bus_id, sizeof(facts[i].pci_bus_id), "%s", bus);
        } else {
            (void)cudaGetLastError();
            snprintf(facts[i].pci_bus_id, sizeof(facts[i].pci_bus_id),
                     "%04x:%02x:%02x", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
        }
        facts[i].uuid[0] = '\0';
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                facts[i].peer_access[j] = true;
                continue;
            }
            int can = 0;
            rc = cudaDeviceCanAccessPeer(&can, i, j);
            if (rc != cudaSuccess) return cuda_v100_ok(rc, "cudaDeviceCanAccessPeer", err, errlen);
            facts[i].peer_access[j] = can != 0;
        }
    }
    *out_count = n;
    return 0;
}

static void stage_free(ds4_v100_cuda_stage *s) {
    if (!s) return;
    if (s->gpu >= 0) (void)cudaSetDevice(s->gpu);
    if (s->relay_f32_out) (void)cudaFree(s->relay_f32_out);
    if (s->relay_f32_in) (void)cudaFree(s->relay_f32_in);
    if (s->relay_f16_out) (void)cudaFree(s->relay_f16_out);
    if (s->relay_f16_in) (void)cudaFree(s->relay_f16_in);
    if (s->scratch) (void)cudaFree(s->scratch);
    if (s->cublas) (void)cublasDestroy(s->cublas);
    if (s->relay_stream) (void)cudaStreamDestroy(s->relay_stream);
    if (s->stream) (void)cudaStreamDestroy(s->stream);
    memset(s, 0, sizeof(*s));
    s->gpu = -1;
}

static int stage_alloc(ds4_v100_cuda_stage *s,
                       const ds4_v100_stage_info *info,
                       bool enable_f32_debug,
                       char *err,
                       size_t errlen) {
    memset(s, 0, sizeof(*s));
    s->gpu = info->gpu;
    s->scratch_bytes = info->scratch_bytes;
    s->relay_f16_bytes = info->relay_f16_bytes;
    s->relay_f32_bytes = enable_f32_debug ? info->relay_f32_debug_bytes : 0;
    if (cuda_v100_ok(cudaSetDevice(s->gpu), "cudaSetDevice", err, errlen)) return 1;
    if (cuda_v100_ok(cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking),
                     "cudaStreamCreate", err, errlen)) return 1;
    if (cuda_v100_ok(cudaStreamCreateWithFlags(&s->relay_stream, cudaStreamNonBlocking),
                     "cudaStreamCreate relay", err, errlen)) return 1;
    if (cublas_v100_ok(cublasCreate(&s->cublas), "cublasCreate", err, errlen)) return 1;
    if (cublas_v100_ok(cublasSetStream(s->cublas, s->stream), "cublasSetStream", err, errlen)) return 1;
    if (s->scratch_bytes &&
        cuda_v100_ok(cudaMalloc(&s->scratch, (size_t)s->scratch_bytes),
                     "cudaMalloc scratch", err, errlen)) {
        return 1;
    }
    if (s->relay_f16_bytes) {
        if (cuda_v100_ok(cudaMalloc(&s->relay_f16_in, (size_t)s->relay_f16_bytes),
                         "cudaMalloc relay f16 in", err, errlen)) return 1;
        if (cuda_v100_ok(cudaMalloc(&s->relay_f16_out, (size_t)s->relay_f16_bytes),
                         "cudaMalloc relay f16 out", err, errlen)) return 1;
    }
    if (s->relay_f32_bytes) {
        if (cuda_v100_ok(cudaMalloc(&s->relay_f32_in, (size_t)s->relay_f32_bytes),
                         "cudaMalloc relay f32 in", err, errlen)) return 1;
        if (cuda_v100_ok(cudaMalloc(&s->relay_f32_out, (size_t)s->relay_f32_bytes),
                         "cudaMalloc relay f32 out", err, errlen)) return 1;
    }
    return 0;
}

int ds4_v100_cuda_context_open(ds4_v100_cuda_context **out,
                               const ds4_v100_context_options *opts,
                               char *err,
                               size_t errlen) {
    if (!out) return cuda_v100_error(err, errlen, "missing CUDA context output");
    *out = NULL;

    ds4_v100_device_fact facts[DS4_V100_EXPECTED_GPUS];
    int n_facts = 0;
    if (ds4_v100_cuda_collect_device_facts(facts, DS4_V100_EXPECTED_GPUS,
                                           &n_facts, err, errlen)) {
        return 1;
    }
    if (n_facts <= 0) return cuda_v100_error(err, errlen, "no CUDA devices visible");

    ds4_v100_context_options local;
    if (opts) local = *opts;
    else ds4_v100_context_options_init(&local);
    if (local.expected_gpus <= 0) local.expected_gpus = n_facts;
    if (local.expected_gpus > n_facts) {
        return cuda_v100_error(err, errlen, "requested %d GPUs but only %d visible",
                               local.expected_gpus, n_facts);
    }
    local.device_facts = facts;
    local.n_device_facts = n_facts;

    ds4_v100_cuda_context *ctx =
        (ds4_v100_cuda_context *)calloc(1, sizeof(*ctx));
    if (!ctx) return cuda_v100_error(err, errlen, "out of memory allocating CUDA context");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) ctx->stages[i].gpu = -1;
    ctx->n_stages = local.expected_gpus;
    ctx->relay_max_active_slots = local.relay_max_active_slots;

    if (ds4_v100_context_open(&ctx->host, &local, err, errlen)) {
        ds4_v100_cuda_context_close(ctx);
        return 1;
    }
    for (int i = 0; i < n_facts && i < DS4_V100_EXPECTED_GPUS; i++) {
        for (int j = 0; j < n_facts && j < DS4_V100_EXPECTED_GPUS; j++) {
            ctx->peer_access[i][j] = facts[i].peer_access[j];
        }
    }
    for (int i = 0; i < ctx->n_stages - 1; i++) {
        if (!ctx->peer_access[i][i + 1]) continue;
        (void)cudaSetDevice(i);
        cudaError_t rc = cudaDeviceEnablePeerAccess(i + 1, 0);
        if (rc != cudaSuccess && rc != cudaErrorPeerAccessAlreadyEnabled) {
            return cuda_v100_ok(rc, "cudaDeviceEnablePeerAccess", err, errlen);
        }
        (void)cudaSetDevice(i + 1);
        rc = cudaDeviceEnablePeerAccess(i, 0);
        if (rc != cudaSuccess && rc != cudaErrorPeerAccessAlreadyEnabled) {
            return cuda_v100_ok(rc, "cudaDeviceEnablePeerAccess reverse", err, errlen);
        }
    }
    for (int i = 0; i < ctx->n_stages; i++) {
        const ds4_v100_stage_info *info = ds4_v100_context_stage(ctx->host, i);
        if (!info || stage_alloc(&ctx->stages[i], info, local.enable_f32_debug_relay,
                                 err, errlen)) {
            ds4_v100_cuda_context_close(ctx);
            return 1;
        }
    }
    *out = ctx;
    return 0;
}

void ds4_v100_cuda_context_close(ds4_v100_cuda_context *ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->n_stages; i++) stage_free(&ctx->stages[i]);
    ds4_v100_context_close(ctx->host);
    free(ctx);
}

static uint64_t relay_bytes(ds4_v100_relay_dtype dtype, uint64_t active_slots) {
    const uint64_t elem = dtype == DS4_V100_RELAY_F16 ? 2ull : 4ull;
    return active_slots * DS4_V100_HC_ROWS * DS4_V100_HC_COLS * elem;
}

int ds4_v100_cuda_context_relay_smoke(ds4_v100_cuda_context *ctx,
                                      int src_stage,
                                      int dst_stage,
                                      ds4_v100_relay_dtype dtype,
                                      uint64_t active_slots,
                                      char *err,
                                      size_t errlen) {
    if (!ctx) return cuda_v100_error(err, errlen, "missing CUDA context");
    if (src_stage < 0 || src_stage >= ctx->n_stages ||
        dst_stage < 0 || dst_stage >= ctx->n_stages) {
        return cuda_v100_error(err, errlen, "invalid relay stage");
    }
    if (active_slots == 0 || active_slots > ctx->relay_max_active_slots) {
        return cuda_v100_error(err, errlen, "active_slots outside relay capacity");
    }
    ds4_v100_cuda_stage *src = &ctx->stages[src_stage];
    ds4_v100_cuda_stage *dst = &ctx->stages[dst_stage];
    uint64_t bytes = relay_bytes(dtype, active_slots);
    void *src_ptr = NULL;
    void *dst_ptr = NULL;
    uint64_t src_cap = 0;
    uint64_t dst_cap = 0;
    if (dtype == DS4_V100_RELAY_F16) {
        src_ptr = src->relay_f16_out;
        dst_ptr = dst->relay_f16_in;
        src_cap = src->relay_f16_bytes;
        dst_cap = dst->relay_f16_bytes;
    } else {
        src_ptr = src->relay_f32_out;
        dst_ptr = dst->relay_f32_in;
        src_cap = src->relay_f32_bytes;
        dst_cap = dst->relay_f32_bytes;
    }
    if (!src_ptr || !dst_ptr || bytes > src_cap || bytes > dst_cap) {
        return cuda_v100_error(err, errlen, "relay buffers are not allocated or too small");
    }
    if (src_stage != dst_stage && !ctx->peer_access[src_stage][dst_stage]) {
        return cuda_v100_error(err, errlen, "relay requires device-to-device peer access");
    }

    unsigned char *host_in = (unsigned char *)malloc((size_t)bytes);
    unsigned char *host_out = (unsigned char *)malloc((size_t)bytes);
    if (!host_in || !host_out) {
        free(host_in);
        free(host_out);
        return cuda_v100_error(err, errlen, "out of memory allocating relay verify buffers");
    }
    for (uint64_t i = 0; i < bytes; i++) host_in[i] = (unsigned char)((i * 131u + 17u) & 0xffu);
    memset(host_out, 0, (size_t)bytes);

    if (cuda_v100_ok(cudaSetDevice(src->gpu), "cudaSetDevice relay source", err, errlen) ||
        cuda_v100_ok(cudaMemcpyAsync(src_ptr, host_in, (size_t)bytes,
                                     cudaMemcpyHostToDevice, src->relay_stream),
                     "relay source upload", err, errlen) ||
        cuda_v100_ok(cudaStreamSynchronize(src->relay_stream),
                     "relay source sync", err, errlen)) {
        free(host_in);
        free(host_out);
        return 1;
    }

    cudaError_t rc;
    if (src_stage == dst_stage) {
        rc = cudaMemcpyAsync(dst_ptr, src_ptr, (size_t)bytes,
                             cudaMemcpyDeviceToDevice, src->relay_stream);
    } else {
        rc = cudaMemcpyPeerAsync(dst_ptr, dst->gpu, src_ptr, src->gpu,
                                 (size_t)bytes, src->relay_stream);
    }
    if (cuda_v100_ok(rc, "relay device copy", err, errlen) ||
        cuda_v100_ok(cudaStreamSynchronize(src->relay_stream),
                     "relay copy sync", err, errlen) ||
        cuda_v100_ok(cudaSetDevice(dst->gpu), "cudaSetDevice relay dest", err, errlen) ||
        cuda_v100_ok(cudaMemcpyAsync(host_out, dst_ptr, (size_t)bytes,
                                     cudaMemcpyDeviceToHost, dst->relay_stream),
                     "relay readback", err, errlen) ||
        cuda_v100_ok(cudaStreamSynchronize(dst->relay_stream),
                     "relay readback sync", err, errlen)) {
        free(host_in);
        free(host_out);
        return 1;
    }
    int cmp = memcmp(host_in, host_out, (size_t)bytes);
    free(host_in);
    free(host_out);
    if (cmp != 0) return cuda_v100_error(err, errlen, "relay byte verification failed");
    return 0;
}
