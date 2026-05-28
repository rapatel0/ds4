#include "engine/context.h"

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
    void *kv_arena;
    uint64_t kv_arena_bytes;
} ds4_cuda_stage;

struct ds4_cuda_context {
    ds4_context *host;
    int n_stages;
    uint64_t relay_max_active_slots;
    ds4_cuda_stage stages[DS4_V100_EXPECTED_GPUS];
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

int ds4_cuda_collect_device_facts(ds4_device_fact *facts,
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

static void stage_free(ds4_cuda_stage *s) {
    if (!s) return;
    if (s->gpu >= 0) (void)cudaSetDevice(s->gpu);
    if (s->kv_arena) (void)cudaFree(s->kv_arena);
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

static int stage_alloc(ds4_cuda_stage *s,
                       const ds4_stage_info *info,
                       bool enable_f32_debug,
                       char *err,
                       size_t errlen) {
    memset(s, 0, sizeof(*s));
    s->gpu = info->gpu;
    s->scratch_bytes = info->scratch_bytes;
    s->relay_f16_bytes = info->relay_f16_bytes;
    s->relay_f32_bytes = enable_f32_debug ? info->relay_f32_debug_bytes : 0;
    s->kv_arena_bytes = info->kv_arena.total_bytes;
    if (s->kv_arena_bytes > (uint64_t)((size_t)-1)) {
        return cuda_v100_error(err, errlen, "kv arena is too large for cudaMalloc");
    }
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
    if (s->kv_arena_bytes &&
        cuda_v100_ok(cudaMalloc(&s->kv_arena, (size_t)s->kv_arena_bytes),
                     "cudaMalloc kv arena", err, errlen)) {
        return 1;
    }
    return 0;
}

int ds4_cuda_context_open(ds4_cuda_context **out,
                               const ds4_context_options *opts,
                               char *err,
                               size_t errlen) {
    if (!out) return cuda_v100_error(err, errlen, "missing CUDA context output");
    *out = NULL;

    ds4_device_fact facts[DS4_V100_EXPECTED_GPUS];
    int n_facts = 0;
    if (ds4_cuda_collect_device_facts(facts, DS4_V100_EXPECTED_GPUS,
                                           &n_facts, err, errlen)) {
        return 1;
    }
    if (n_facts <= 0) return cuda_v100_error(err, errlen, "no CUDA devices visible");

    ds4_context_options local;
    if (opts) local = *opts;
    else ds4_context_options_init(&local);
    if (local.expected_gpus <= 0) local.expected_gpus = n_facts;
    if (local.expected_gpus > n_facts) {
        return cuda_v100_error(err, errlen, "requested %d GPUs but only %d visible",
                               local.expected_gpus, n_facts);
    }
    local.device_facts = facts;
    local.n_device_facts = n_facts;

    ds4_cuda_context *ctx =
        (ds4_cuda_context *)calloc(1, sizeof(*ctx));
    if (!ctx) return cuda_v100_error(err, errlen, "out of memory allocating CUDA context");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) ctx->stages[i].gpu = -1;
    ctx->n_stages = local.expected_gpus;
    ctx->relay_max_active_slots = local.relay_max_active_slots;

    if (ds4_context_open(&ctx->host, &local, err, errlen)) {
        ds4_cuda_context_close(ctx);
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
        const ds4_stage_info *info = ds4_context_stage(ctx->host, i);
        if (!info || stage_alloc(&ctx->stages[i], info, local.enable_f32_debug_relay,
                                 err, errlen)) {
            ds4_cuda_context_close(ctx);
            return 1;
        }
    }
    *out = ctx;
    return 0;
}

void ds4_cuda_context_close(ds4_cuda_context *ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->n_stages; i++) stage_free(&ctx->stages[i]);
    ds4_context_close(ctx->host);
    free(ctx);
}

int ds4_cuda_context_layer_kv_view(ds4_cuda_context *ctx,
                                        int layer_id,
                                        ds4_cuda_layer_kv_view *out,
                                        char *err,
                                        size_t errlen) {
    if (!ctx || !out) return cuda_v100_error(err, errlen, "missing CUDA KV view output");
    const ds4_layer_info *layer = ds4_context_layer(ctx->host, layer_id);
    if (!layer) return cuda_v100_error(err, errlen, "invalid layer id %d", layer_id);
    if (layer->stage_id < 0 || layer->stage_id >= ctx->n_stages) {
        return cuda_v100_error(err, errlen, "layer %d has invalid stage %d",
                               layer_id, layer->stage_id);
    }
    ds4_cuda_stage *stage = &ctx->stages[layer->stage_id];
    if (!stage->kv_arena || stage->kv_arena_bytes == 0) {
        return cuda_v100_error(err, errlen, "stage %d has no allocated KV arena",
                               layer->stage_id);
    }
    if (layer->kv_view.total_bytes == 0) {
        return cuda_v100_error(err, errlen, "layer %d has no derived KV view", layer_id);
    }
    memset(out, 0, sizeof(*out));
    out->layer_id = layer_id;
    out->stage_id = layer->stage_id;
    out->gpu = stage->gpu;
    out->kv_arena_base = stage->kv_arena;
    out->kv_arena_bytes = stage->kv_arena_bytes;
    out->view = layer->kv_view;
    return 0;
}

static uint64_t max_u64(uint64_t a, uint64_t b) {
    return a > b ? a : b;
}

__global__ static void v100_context_prefill_kv_update_kernel(
        unsigned char *arena,
        ds4_layer_kv_view view,
        uint32_t ratio,
        uint32_t slot,
        uint32_t slots,
        uint32_t raw_row,
        uint32_t comp_row,
        uint32_t comp_rows,
        const float *attn_row,
        const float *indexer_row) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < DS4_V100_HEAD_DIM) {
        const __half v = __float2half_rn(attn_row[tid]);
        __half *raw = (__half *)(arena + view.raw_swa_offset);
        __half *comp = (__half *)(arena + view.compressed_attn_offset);
        const uint64_t raw_idx =
            ((uint64_t)slot * DS4_V100_SWA_ROWS + raw_row) *
            DS4_V100_HEAD_DIM + tid;
        const uint64_t comp_idx =
            ((uint64_t)slot * comp_rows + comp_row) * DS4_V100_HEAD_DIM + tid;
        raw[raw_idx] = v;
        comp[comp_idx] = v;
    }

    if (ratio == 4u && indexer_row && tid < DS4_V100_INDEXER_HEAD_DIM) {
        __half *indexer = (__half *)(arena + view.indexer_kv_offset);
        const uint64_t index_idx =
            ((uint64_t)slot * comp_rows + comp_row) *
            DS4_V100_INDEXER_HEAD_DIM + tid;
        indexer[index_idx] = __float2half_rn(indexer_row[tid]);
    }

    const uint64_t attn_kv_values = view.attn_state_kv_bytes / sizeof(float);
    const uint64_t attn_score_values = view.attn_state_score_bytes / sizeof(float);
    if (tid < attn_kv_values) {
        const uint32_t lane = (uint32_t)(tid % DS4_V100_HEAD_DIM);
        const uint32_t row = (uint32_t)(tid / DS4_V100_HEAD_DIM);
        float *state = (float *)(arena + view.attn_state_kv_offset);
        state[tid] = attn_row[lane] + (float)row * 0.125f + (float)ratio * 0.001f;
    }
    if (tid < attn_score_values) {
        const uint32_t lane = (uint32_t)(tid % DS4_V100_HEAD_DIM);
        const uint32_t row = (uint32_t)(tid / DS4_V100_HEAD_DIM);
        float *state = (float *)(arena + view.attn_state_score_offset);
        state[tid] = attn_row[lane] - (float)row * 0.03125f - (float)ratio * 0.002f;
    }

    if (ratio == 4u && indexer_row) {
        const uint64_t idx_kv_values = view.indexer_state_kv_bytes / sizeof(float);
        const uint64_t idx_score_values = view.indexer_state_score_bytes / sizeof(float);
        if (tid < idx_kv_values) {
            const uint32_t lane = (uint32_t)(tid % DS4_V100_INDEXER_HEAD_DIM);
            const uint32_t row = (uint32_t)(tid / DS4_V100_INDEXER_HEAD_DIM);
            float *state = (float *)(arena + view.indexer_state_kv_offset);
            state[tid] = indexer_row[lane] + (float)row * 0.0625f +
                         (float)ratio * 0.003f;
        }
        if (tid < idx_score_values) {
            const uint32_t lane = (uint32_t)(tid % DS4_V100_INDEXER_HEAD_DIM);
            const uint32_t row = (uint32_t)(tid / DS4_V100_INDEXER_HEAD_DIM);
            float *state = (float *)(arena + view.indexer_state_score_offset);
            state[tid] = indexer_row[lane] - (float)row * 0.015625f -
                         (float)ratio * 0.004f;
        }
    }

    (void)slots;
}

static int infer_layer_slots_and_comp_rows(const ds4_layer_kv_view *view,
                                           uint64_t *out_slots,
                                           uint64_t *out_comp_rows) {
    const uint64_t raw_per_slot =
        (uint64_t)DS4_V100_SWA_ROWS * DS4_V100_HEAD_DIM * sizeof(__half);
    if (!view || raw_per_slot == 0 ||
        view->raw_swa_bytes == 0 ||
        view->raw_swa_bytes % raw_per_slot != 0) {
        return 1;
    }
    const uint64_t slots = view->raw_swa_bytes / raw_per_slot;
    if (slots == 0) return 1;

    const uint64_t comp_per_row = (uint64_t)DS4_V100_HEAD_DIM * sizeof(__half);
    if (view->compressed_attn_bytes == 0 ||
        view->compressed_attn_bytes % (slots * comp_per_row) != 0) {
        return 1;
    }
    const uint64_t comp_rows = view->compressed_attn_bytes / (slots * comp_per_row);
    if (comp_rows == 0) return 1;
    *out_slots = slots;
    *out_comp_rows = comp_rows;
    return 0;
}

static int require_same_device_f32_row(const void *ptr,
                                       uint64_t min_bytes,
                                       int gpu,
                                       const char *what,
                                       char *err,
                                       size_t errlen) {
    if (!ptr) return cuda_v100_error(err, errlen, "missing %s", what);
    cudaPointerAttributes attr;
    memset(&attr, 0, sizeof(attr));
    cudaError_t rc = cudaPointerGetAttributes(&attr, ptr);
    if (rc != cudaSuccess) {
        (void)cudaGetLastError();
        return cuda_v100_error(err, errlen, "%s is not a CUDA allocation", what);
    }
    if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
        return cuda_v100_error(err, errlen, "%s is not device-resident", what);
    }
    if (attr.type == cudaMemoryTypeDevice && attr.device != gpu) {
        return cuda_v100_error(err, errlen, "%s is on gpu %d, expected gpu %d",
                               what, attr.device, gpu);
    }
    (void)min_bytes;
    return 0;
}

int ds4_cuda_context_prefill_kv_update_f16(
        ds4_cuda_context                    *ctx,
        int                                      layer_id,
        const ds4_cuda_prefill_kv_update   *update,
        char                                    *err,
        size_t                                   errlen) {
    if (!ctx || !update || !update->attn_row_f32) {
        return cuda_v100_error(err, errlen, "missing prefill KV update input");
    }

    const ds4_layer_info *layer = ds4_context_layer(ctx->host, layer_id);
    if (!layer) return cuda_v100_error(err, errlen, "invalid layer id %d", layer_id);
    if (layer->layer_class != DS4_V100_LAYER_RATIO_4 &&
        layer->layer_class != DS4_V100_LAYER_RATIO_128) {
        return cuda_v100_error(err, errlen, "layer %d has no compressed KV path", layer_id);
    }
    if (layer->stage_id < 0 || layer->stage_id >= ctx->n_stages) {
        return cuda_v100_error(err, errlen, "layer %d has invalid stage %d",
                               layer_id, layer->stage_id);
    }
    ds4_cuda_stage *stage = &ctx->stages[layer->stage_id];
    if (!stage->kv_arena || stage->kv_arena_bytes == 0) {
        return cuda_v100_error(err, errlen, "stage %d has no allocated KV arena",
                               layer->stage_id);
    }

    const ds4_layer_kv_view *view = &layer->kv_view;
    uint64_t slots = 0;
    uint64_t comp_rows = 0;
    if (infer_layer_slots_and_comp_rows(view, &slots, &comp_rows)) {
        return cuda_v100_error(err, errlen, "layer %d has invalid KV view geometry",
                               layer_id);
    }
    if (slots > UINT32_MAX || comp_rows > UINT32_MAX) {
        return cuda_v100_error(err, errlen, "layer %d KV view geometry exceeds diagnostic limits",
                               layer_id);
    }
    if (update->slot >= slots || update->raw_row >= DS4_V100_SWA_ROWS ||
        update->comp_row >= comp_rows) {
        return cuda_v100_error(err, errlen, "KV update row or slot is out of bounds");
    }

    const uint32_t ratio =
        layer->layer_class == DS4_V100_LAYER_RATIO_4 ? 4u : 128u;
    if (ratio == 4u) {
        if (!update->indexer_row_f32 || view->indexer_kv_bytes == 0 ||
            view->indexer_state_kv_bytes == 0 ||
            view->indexer_state_score_bytes == 0) {
            return cuda_v100_error(err, errlen, "ratio-4 KV update requires indexer inputs");
        }
        const uint64_t index_per_row =
            (uint64_t)DS4_V100_INDEXER_HEAD_DIM * sizeof(__half);
        if (view->indexer_kv_bytes % (slots * index_per_row) != 0 ||
            view->indexer_kv_bytes / (slots * index_per_row) < comp_rows) {
            return cuda_v100_error(err, errlen, "ratio-4 indexer KV view is too small");
        }
    }

    float *dev_attn = NULL;
    float *dev_indexer = NULL;
    int ok = 1;
    if (cuda_v100_ok(cudaSetDevice(stage->gpu), "cudaSetDevice prefill KV",
                     err, errlen)) {
        return 1;
    }
    const uint64_t attn_bytes = (uint64_t)DS4_V100_HEAD_DIM * sizeof(float);
    if (cuda_v100_ok(cudaMalloc(&dev_attn, (size_t)attn_bytes),
                     "cudaMalloc prefill KV attn row", err, errlen) ||
        cuda_v100_ok(cudaMemcpy(dev_attn, update->attn_row_f32, (size_t)attn_bytes,
                                cudaMemcpyHostToDevice),
                     "prefill KV attn row upload", err, errlen)) {
        ok = 0;
    }
    if (ok && ratio == 4u) {
        const uint64_t indexer_bytes =
            (uint64_t)DS4_V100_INDEXER_HEAD_DIM * sizeof(float);
        if (cuda_v100_ok(cudaMalloc(&dev_indexer, (size_t)indexer_bytes),
                         "cudaMalloc prefill KV indexer row", err, errlen) ||
            cuda_v100_ok(cudaMemcpy(dev_indexer, update->indexer_row_f32,
                                    (size_t)indexer_bytes, cudaMemcpyHostToDevice),
                         "prefill KV indexer row upload", err, errlen)) {
            ok = 0;
        }
    }

    if (ok) {
        uint64_t n = DS4_V100_HEAD_DIM;
        n = max_u64(n, view->attn_state_kv_bytes / sizeof(float));
        n = max_u64(n, view->attn_state_score_bytes / sizeof(float));
        if (ratio == 4u) {
            n = max_u64(n, DS4_V100_INDEXER_HEAD_DIM);
            n = max_u64(n, view->indexer_state_kv_bytes / sizeof(float));
            n = max_u64(n, view->indexer_state_score_bytes / sizeof(float));
        }
        v100_context_prefill_kv_update_kernel<<<(unsigned int)((n + 255u) / 256u), 256>>>(
            (unsigned char *)stage->kv_arena,
            *view,
            ratio,
            update->slot,
            (uint32_t)slots,
            update->raw_row,
            update->comp_row,
            (uint32_t)comp_rows,
            dev_attn,
            dev_indexer);
        ok = !cuda_v100_ok(cudaGetLastError(), "prefill KV update launch", err, errlen) &&
             !cuda_v100_ok(cudaDeviceSynchronize(), "prefill KV update sync", err, errlen);
    }

    if (dev_indexer) (void)cudaFree(dev_indexer);
    if (dev_attn) (void)cudaFree(dev_attn);
    return ok ? 0 : 1;
}

int ds4_cuda_context_prefill_kv_update_f16_device(
        ds4_cuda_context                           *ctx,
        int                                             layer_id,
        const ds4_cuda_prefill_kv_update_device   *update,
        char                                           *err,
        size_t                                          errlen) {
    if (!ctx || !update || !update->attn_row_device_f32) {
        return cuda_v100_error(err, errlen, "missing device prefill KV update input");
    }

    const ds4_layer_info *layer = ds4_context_layer(ctx->host, layer_id);
    if (!layer) return cuda_v100_error(err, errlen, "invalid layer id %d", layer_id);
    if (layer->layer_class != DS4_V100_LAYER_RATIO_4 &&
        layer->layer_class != DS4_V100_LAYER_RATIO_128) {
        return cuda_v100_error(err, errlen, "layer %d has no compressed KV path", layer_id);
    }
    if (layer->stage_id < 0 || layer->stage_id >= ctx->n_stages) {
        return cuda_v100_error(err, errlen, "layer %d has invalid stage %d",
                               layer_id, layer->stage_id);
    }
    ds4_cuda_stage *stage = &ctx->stages[layer->stage_id];
    if (!stage->kv_arena || stage->kv_arena_bytes == 0) {
        return cuda_v100_error(err, errlen, "stage %d has no allocated KV arena",
                               layer->stage_id);
    }

    const ds4_layer_kv_view *view = &layer->kv_view;
    uint64_t slots = 0;
    uint64_t comp_rows = 0;
    if (infer_layer_slots_and_comp_rows(view, &slots, &comp_rows)) {
        return cuda_v100_error(err, errlen, "layer %d has invalid KV view geometry",
                               layer_id);
    }
    if (slots > UINT32_MAX || comp_rows > UINT32_MAX) {
        return cuda_v100_error(err, errlen, "layer %d KV view geometry exceeds diagnostic limits",
                               layer_id);
    }
    if (update->slot >= slots || update->raw_row >= DS4_V100_SWA_ROWS ||
        update->comp_row >= comp_rows) {
        return cuda_v100_error(err, errlen, "KV update row or slot is out of bounds");
    }

    const uint32_t ratio =
        layer->layer_class == DS4_V100_LAYER_RATIO_4 ? 4u : 128u;
    if (cuda_v100_ok(cudaSetDevice(stage->gpu), "cudaSetDevice device prefill KV",
                     err, errlen)) {
        return 1;
    }
    if (require_same_device_f32_row(update->attn_row_device_f32,
                                    (uint64_t)DS4_V100_HEAD_DIM * sizeof(float),
                                    stage->gpu,
                                    "device attention row",
                                    err,
                                    errlen)) {
        return 1;
    }
    if (ratio == 4u) {
        if (!update->indexer_row_device_f32 || view->indexer_kv_bytes == 0 ||
            view->indexer_state_kv_bytes == 0 ||
            view->indexer_state_score_bytes == 0) {
            return cuda_v100_error(err, errlen, "ratio-4 KV update requires device indexer inputs");
        }
        const uint64_t index_per_row =
            (uint64_t)DS4_V100_INDEXER_HEAD_DIM * sizeof(__half);
        if (view->indexer_kv_bytes % (slots * index_per_row) != 0 ||
            view->indexer_kv_bytes / (slots * index_per_row) < comp_rows) {
            return cuda_v100_error(err, errlen, "ratio-4 indexer KV view is too small");
        }
        if (require_same_device_f32_row(update->indexer_row_device_f32,
                                        (uint64_t)DS4_V100_INDEXER_HEAD_DIM * sizeof(float),
                                        stage->gpu,
                                        "device indexer row",
                                        err,
                                        errlen)) {
            return 1;
        }
    }

    uint64_t n = DS4_V100_HEAD_DIM;
    n = max_u64(n, view->attn_state_kv_bytes / sizeof(float));
    n = max_u64(n, view->attn_state_score_bytes / sizeof(float));
    if (ratio == 4u) {
        n = max_u64(n, DS4_V100_INDEXER_HEAD_DIM);
        n = max_u64(n, view->indexer_state_kv_bytes / sizeof(float));
        n = max_u64(n, view->indexer_state_score_bytes / sizeof(float));
    }
    v100_context_prefill_kv_update_kernel<<<(unsigned int)((n + 255u) / 256u), 256>>>(
        (unsigned char *)stage->kv_arena,
        *view,
        ratio,
        update->slot,
        (uint32_t)slots,
        update->raw_row,
        update->comp_row,
        (uint32_t)comp_rows,
        (const float *)update->attn_row_device_f32,
        (const float *)update->indexer_row_device_f32);
    return cuda_v100_ok(cudaGetLastError(), "device prefill KV update launch", err, errlen) ||
           cuda_v100_ok(cudaDeviceSynchronize(), "device prefill KV update sync", err, errlen);
}

int ds4_cuda_context_read_kv_arena(ds4_cuda_context *ctx,
                                        int stage_id,
                                        uint64_t offset,
                                        void *dst,
                                        uint64_t bytes,
                                        char *err,
                                        size_t errlen) {
    if (!ctx || !dst) return cuda_v100_error(err, errlen, "missing KV arena read output");
    if (stage_id < 0 || stage_id >= ctx->n_stages) {
        return cuda_v100_error(err, errlen, "invalid stage id %d", stage_id);
    }
    ds4_cuda_stage *stage = &ctx->stages[stage_id];
    if (!stage->kv_arena || offset > stage->kv_arena_bytes ||
        bytes > stage->kv_arena_bytes - offset) {
        return cuda_v100_error(err, errlen, "KV arena read range is out of bounds");
    }
    if (cuda_v100_ok(cudaSetDevice(stage->gpu), "cudaSetDevice KV read",
                     err, errlen) ||
        cuda_v100_ok(cudaMemcpy(dst, (unsigned char *)stage->kv_arena + offset,
                                (size_t)bytes, cudaMemcpyDeviceToHost),
                     "KV arena readback", err, errlen)) {
        return 1;
    }
    return 0;
}

static uint64_t relay_bytes(ds4_relay_dtype dtype, uint64_t active_slots) {
    const uint64_t elem = dtype == DS4_V100_RELAY_F16 ? 2ull : 4ull;
    return active_slots * DS4_V100_HC_ROWS * DS4_V100_HC_COLS * elem;
}

int ds4_cuda_context_relay_smoke(ds4_cuda_context *ctx,
                                      int src_stage,
                                      int dst_stage,
                                      ds4_relay_dtype dtype,
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
    ds4_cuda_stage *src = &ctx->stages[src_stage];
    ds4_cuda_stage *dst = &ctx->stages[dst_stage];
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
