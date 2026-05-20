extern "C" {
#include "ds4_gpu.h"
#include "ds4_source_formats.h"
}

#include "ggml-turbomind-api.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef int  (*pfn_init)(int);
typedef void (*pfn_shutdown)(void);
typedef int  (*pfn_packed_bytes)(int, int, int, int, size_t *, size_t *);
typedef int  (*pfn_pack_weight)(const void *, int, int, int, int, void *, void *, int *, void *);
typedef int  (*pfn_mul_mat_grouped)(const void *, const int *, const int *, int,
                                    const void * const *, const void * const *,
                                    int, int, int, int, int, void *, void *);

struct alignas(16) StridedPtrH {
    void *p;
    int stride;
};
static_assert(sizeof(StridedPtrH) == 16, "StridedPtrH must match TurboMind StridedPtr");

static int failures;

static void check(bool cond, const char *msg) {
    if (!cond) {
        std::fprintf(stderr, "cuda_v100_turbomind_adapter_smoke: %s\n", msg);
        failures++;
    }
}

static bool cuda_check(cudaError_t rc, const char *msg) {
    if (rc != cudaSuccess) {
        std::fprintf(stderr,
                     "cuda_v100_turbomind_adapter_smoke: %s: %s\n",
                     msg,
                     cudaGetErrorString(rc));
        failures++;
        return false;
    }
    return true;
}

static uint64_t expert_bytes(uint32_t rows, uint32_t cols) {
    return (uint64_t)rows * ds4_src_mxfp4_row_bytes(cols);
}

static void fill_mxfp4_row(uint8_t *row, uint32_t cols, uint32_t seed) {
    static const uint8_t codes[] = {
        0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x1,
        0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x1, 0x2,
    };
    const uint64_t row_bytes = ds4_src_mxfp4_row_bytes(cols);
    std::memset(row, 0, (size_t)row_bytes);
    for (uint32_t b = 0; b < cols / DS4_SRC_MXFP4_BLOCK_ELEMS; b++) {
        uint8_t *block = row + (uint64_t)b * DS4_SRC_MXFP4_BLOCK_BYTES;
        block[0] = 119u;
        for (uint32_t j = 0; j < DS4_SRC_MXFP4_BLOCK_ELEMS / 2; j++) {
            const uint8_t lo = codes[(seed + b * 7u + j * 3u) & 15u];
            const uint8_t hi = codes[(seed + b * 5u + j * 11u + 1u) & 15u];
            block[1u + j] = (uint8_t)(lo | (hi << 4));
        }
    }
}

static void fill_expert_matrix(std::vector<uint8_t> &payload,
                               uint64_t offset,
                               uint32_t experts,
                               uint32_t rows,
                               uint32_t cols,
                               uint64_t expert_stride,
                               uint32_t seed) {
    const uint64_t row_bytes = ds4_src_mxfp4_row_bytes(cols);
    for (uint32_t expert = 0; expert < experts; expert++) {
        uint8_t *base = payload.data() + offset + (uint64_t)expert * expert_stride;
        for (uint32_t row = 0; row < rows; row++) {
            fill_mxfp4_row(base + (uint64_t)row * row_bytes,
                           cols,
                           seed + expert * 131u + row * 17u);
        }
    }
}

__global__ static void swiglu_half_kernel(__half *out,
                                          const __half *gate,
                                          const __half *up,
                                          const float *weights,
                                          uint32_t n_rows,
                                          uint32_t cols,
                                          float clamp) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_rows * cols;
    if (idx >= n) return;
    const uint32_t row = (uint32_t)(idx / cols);
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    const float s = g / (1.0f + expf(-g));
    out[idx] = __float2half_rn(s * u * weights[row]);
}

__global__ static void sum_half_routes_to_f32_kernel(float *__restrict__ out,
                                                     const __half *__restrict__ routes,
                                                     uint32_t n_routes,
                                                     uint32_t hidden) {
    const uint32_t h = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (h >= hidden) return;
    float acc = 0.0f;
    for (uint32_t route = 0; route < n_routes; route++) {
        acc += __half2float(routes[(uint64_t)route * hidden + h]);
    }
    out[h] = acc;
}

static bool load_api(const char *lib_path,
                     void **handle,
                     pfn_init *init,
                     pfn_shutdown *shutdown,
                     pfn_packed_bytes *packed_bytes,
                     pfn_pack_weight *pack_weight,
                     pfn_mul_mat_grouped *mul_mat_grouped) {
    *handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!*handle) {
        std::fprintf(stderr,
                     "cuda_v100_turbomind_adapter_smoke: failed to open %s: %s\n",
                     lib_path,
                     dlerror());
        return false;
    }
    *init = (pfn_init)dlsym(*handle, "ggml_turbomind_init");
    *shutdown = (pfn_shutdown)dlsym(*handle, "ggml_turbomind_shutdown");
    *packed_bytes = (pfn_packed_bytes)dlsym(*handle, "ggml_turbomind_packed_bytes");
    *pack_weight = (pfn_pack_weight)dlsym(*handle, "ggml_turbomind_pack_weight_expert");
    *mul_mat_grouped = (pfn_mul_mat_grouped)dlsym(*handle, "ggml_turbomind_mul_mat_grouped");
    if (!*init || !*shutdown || !*packed_bytes || !*pack_weight || !*mul_mat_grouped) {
        std::fprintf(stderr,
                     "cuda_v100_turbomind_adapter_smoke: missing TurboMind C ABI symbol\n");
        return false;
    }
    return true;
}

struct PackedExperts {
    std::vector<void *> weights;
    std::vector<void *> scales;
    StridedPtrH *d_weights = nullptr;
    StridedPtrH *d_scales = nullptr;
    int k_pack = 0;
    int n = 0;
    int k = 0;
};

static bool pack_experts(PackedExperts *out,
                         pfn_packed_bytes packed_bytes,
                         pfn_pack_weight pack_weight,
                         const uint8_t *d_src,
                         uint32_t experts,
                         uint64_t expert_stride,
                         int n,
                         int k) {
    size_t weight_bytes = 0;
    size_t scale_bytes = 0;
    if (packed_bytes(GGML_TM_DTYPE_MXFP4,
                     n,
                     k,
                     DS4_SRC_MXFP4_BLOCK_ELEMS,
                     &weight_bytes,
                     &scale_bytes) != 0) {
        std::fprintf(stderr, "cuda_v100_turbomind_adapter_smoke: packed_bytes failed\n");
        return false;
    }
    out->weights.assign(experts, nullptr);
    out->scales.assign(experts, nullptr);
    out->n = n;
    out->k = k;
    for (uint32_t expert = 0; expert < experts; expert++) {
        if (!cuda_check(cudaMalloc(&out->weights[expert], weight_bytes), "weight pack malloc") ||
            !cuda_check(cudaMalloc(&out->scales[expert], scale_bytes), "scale pack malloc")) {
            return false;
        }
        int this_pack = 0;
        const uint8_t *src = d_src + (uint64_t)expert * expert_stride;
        if (pack_weight(src,
                        GGML_TM_DTYPE_MXFP4,
                        n,
                        k,
                        DS4_SRC_MXFP4_BLOCK_ELEMS,
                        out->weights[expert],
                        out->scales[expert],
                        &this_pack,
                        nullptr) != 0) {
            std::fprintf(stderr,
                         "cuda_v100_turbomind_adapter_smoke: pack expert %u failed\n",
                         expert);
            return false;
        }
        if (expert == 0) {
            out->k_pack = this_pack;
        } else if (out->k_pack != this_pack) {
            std::fprintf(stderr,
                         "cuda_v100_turbomind_adapter_smoke: inconsistent k_pack 0x%x vs 0x%x\n",
                         out->k_pack,
                         this_pack);
            return false;
        }
    }
    std::vector<StridedPtrH> h_weights(experts);
    std::vector<StridedPtrH> h_scales(experts);
    for (uint32_t expert = 0; expert < experts; expert++) {
        h_weights[expert] = StridedPtrH{out->weights[expert], k * 32};
        h_scales[expert] = StridedPtrH{out->scales[expert], n};
    }
    if (!cuda_check(cudaMalloc(&out->d_weights, experts * sizeof(StridedPtrH)), "weight table malloc") ||
        !cuda_check(cudaMalloc(&out->d_scales, experts * sizeof(StridedPtrH)), "scale table malloc") ||
        !cuda_check(cudaMemcpy(out->d_weights,
                               h_weights.data(),
                               experts * sizeof(StridedPtrH),
                               cudaMemcpyHostToDevice),
                    "weight table upload") ||
        !cuda_check(cudaMemcpy(out->d_scales,
                               h_scales.data(),
                               experts * sizeof(StridedPtrH),
                               cudaMemcpyHostToDevice),
                    "scale table upload")) {
        return false;
    }
    return cuda_check(cudaDeviceSynchronize(), "pack synchronize");
}

static void free_packed(PackedExperts *p) {
    for (void *ptr : p->weights) (void)cudaFree(ptr);
    for (void *ptr : p->scales) (void)cudaFree(ptr);
    (void)cudaFree(p->d_weights);
    (void)cudaFree(p->d_scales);
    *p = PackedExperts();
}

int main(int argc, char **argv) {
    enum {
        EXPERTS = 8,
        ROUTES = 6,
        HIDDEN = 4096,
        MID = 2048,
    };

    const char *lib_path = argc > 1 ? argv[1] : getenv("DS4_TURBOMIND_LIB");
    if (!lib_path || !lib_path[0]) {
        lib_path = "./build/turbomind-v100/libggml-turbomind.so";
    }

    void *lib = nullptr;
    pfn_init tm_init = nullptr;
    pfn_shutdown tm_shutdown = nullptr;
    pfn_packed_bytes tm_packed_bytes = nullptr;
    pfn_pack_weight tm_pack_weight = nullptr;
    pfn_mul_mat_grouped tm_mul_mat_grouped = nullptr;
    if (!load_api(lib_path,
                  &lib,
                  &tm_init,
                  &tm_shutdown,
                  &tm_packed_bytes,
                  &tm_pack_weight,
                  &tm_mul_mat_grouped)) {
        return 2;
    }

    check(ds4_gpu_device_count() > 0, "no CUDA devices visible");
    check(ds4_gpu_init(), "ds4_gpu_init failed");
    check(ds4_gpu_set_device(0), "ds4_gpu_set_device failed");
    check(tm_init(0) == 0, "ggml_turbomind_init failed");
    if (failures) return 3;

    const uint64_t hidden_row_bytes = ds4_src_mxfp4_row_bytes(HIDDEN);
    const uint64_t mid_row_bytes = ds4_src_mxfp4_row_bytes(MID);
    const uint64_t gate_expert_bytes = expert_bytes(MID, HIDDEN);
    const uint64_t down_expert_bytes = expert_bytes(HIDDEN, MID);
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = gate_offset + (uint64_t)EXPERTS * gate_expert_bytes;
    const uint64_t down_offset = up_offset + (uint64_t)EXPERTS * gate_expert_bytes;
    const uint64_t payload_bytes = down_offset + (uint64_t)EXPERTS * down_expert_bytes;

    std::vector<uint8_t> payload((size_t)payload_bytes);
    fill_expert_matrix(payload, gate_offset, EXPERTS, MID, HIDDEN, gate_expert_bytes, 0x100u);
    fill_expert_matrix(payload, up_offset, EXPERTS, MID, HIDDEN, gate_expert_bytes, 0x300u);
    fill_expert_matrix(payload, down_offset, EXPERTS, HIDDEN, MID, down_expert_bytes, 0x700u);

    ds4_gpu_arena *arena = nullptr;
    check(ds4_gpu_arena_open(&arena, 0, payload_bytes) == 0, "arena open failed");
    check(ds4_gpu_arena_upload(arena, 0, payload.data(), payload_bytes) == 0, "arena upload failed");
    if (failures) return 4;

    uint8_t *d_gate = nullptr;
    uint8_t *d_up = nullptr;
    uint8_t *d_down = nullptr;
    cuda_check(cudaMalloc(&d_gate, (size_t)EXPERTS * gate_expert_bytes), "gate source malloc");
    cuda_check(cudaMalloc(&d_up, (size_t)EXPERTS * gate_expert_bytes), "up source malloc");
    cuda_check(cudaMalloc(&d_down, (size_t)EXPERTS * down_expert_bytes), "down source malloc");
    cuda_check(cudaMemcpy(d_gate,
                          payload.data() + gate_offset,
                          (size_t)EXPERTS * gate_expert_bytes,
                          cudaMemcpyHostToDevice),
               "gate source upload");
    cuda_check(cudaMemcpy(d_up,
                          payload.data() + up_offset,
                          (size_t)EXPERTS * gate_expert_bytes,
                          cudaMemcpyHostToDevice),
               "up source upload");
    cuda_check(cudaMemcpy(d_down,
                          payload.data() + down_offset,
                          (size_t)EXPERTS * down_expert_bytes,
                          cudaMemcpyHostToDevice),
               "down source upload");

    PackedExperts gate_pack;
    PackedExperts up_pack;
    PackedExperts down_pack;
    check(pack_experts(&gate_pack, tm_packed_bytes, tm_pack_weight,
                       d_gate, EXPERTS, gate_expert_bytes, MID, HIDDEN),
          "gate pack failed");
    check(pack_experts(&up_pack, tm_packed_bytes, tm_pack_weight,
                       d_up, EXPERTS, gate_expert_bytes, MID, HIDDEN),
          "up pack failed");
    check(pack_experts(&down_pack, tm_packed_bytes, tm_pack_weight,
                       d_down, EXPERTS, down_expert_bytes, HIDDEN, MID),
          "down pack failed");
    if (failures) return 5;

    const int32_t selected[ROUTES] = {5, 1, 6, 0, 3, 2};
    const float route_weights[ROUTES] = {0.31f, 0.27f, 0.23f, 0.19f, 0.17f, 0.13f};
    std::vector<float> hidden(HIDDEN);
    for (uint32_t i = 0; i < HIDDEN; i++) {
        hidden[i] = 0.015f +
                    0.003f * sinf((float)i * 0.013f) +
                    0.001f * cosf((float)i * 0.031f);
    }

    ds4_gpu_tensor *hidden_t = ds4_gpu_tensor_alloc((uint64_t)HIDDEN * sizeof(float));
    ds4_gpu_tensor *selected_t = ds4_gpu_tensor_alloc((uint64_t)ROUTES * sizeof(int32_t));
    ds4_gpu_tensor *weights_t = ds4_gpu_tensor_alloc((uint64_t)ROUTES * sizeof(float));
    ds4_gpu_tensor *mid_ref_t = ds4_gpu_tensor_alloc((uint64_t)ROUTES * MID * sizeof(float));
    ds4_gpu_tensor *ref_out_t = ds4_gpu_tensor_alloc((uint64_t)HIDDEN * sizeof(float));
    check(hidden_t && selected_t && weights_t && mid_ref_t && ref_out_t, "reference tensor allocation failed");
    check(ds4_gpu_tensor_write(hidden_t, 0, hidden.data(), (uint64_t)HIDDEN * sizeof(float)),
          "hidden upload failed");
    check(ds4_gpu_tensor_write(selected_t, 0, selected, (uint64_t)ROUTES * sizeof(int32_t)),
          "selected upload failed");
    check(ds4_gpu_tensor_write(weights_t, 0, route_weights, (uint64_t)ROUTES * sizeof(float)),
          "weights upload failed");
    check(ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_f32(
              arena,
              gate_offset,
              (uint64_t)EXPERTS * gate_expert_bytes,
              up_offset,
              (uint64_t)EXPERTS * gate_expert_bytes,
              down_offset,
              (uint64_t)EXPERTS * down_expert_bytes,
              gate_expert_bytes,
              (uint32_t)hidden_row_bytes,
              down_expert_bytes,
              (uint32_t)mid_row_bytes,
              HIDDEN,
              MID,
              EXPERTS,
              selected_t,
              weights_t,
              ROUTES,
              hidden_t,
              mid_ref_t,
              ref_out_t) == 0,
          "DS4 arena reference failed");

    std::vector<int> counts(EXPERTS, 0);
    for (uint32_t route = 0; route < ROUTES; route++) counts[(uint32_t)selected[route]]++;
    std::vector<int> offsets(EXPERTS + 1, 0);
    for (uint32_t expert = 0; expert < EXPERTS; expert++) {
        offsets[expert + 1] = offsets[expert] + counts[expert];
    }
    std::vector<int> cursor = offsets;
    std::vector<__half> a_routes(ROUTES * HIDDEN);
    std::vector<float> route_weights_sorted(ROUTES);
    for (uint32_t route = 0; route < ROUTES; route++) {
        const uint32_t expert = (uint32_t)selected[route];
        const uint32_t row = (uint32_t)cursor[expert]++;
        route_weights_sorted[row] = route_weights[route];
        for (uint32_t c = 0; c < HIDDEN; c++) {
            a_routes[(uint64_t)row * HIDDEN + c] = __float2half_rn(hidden[c]);
        }
    }

    __half *d_a = nullptr;
    __half *d_gate_out = nullptr;
    __half *d_up_out = nullptr;
    __half *d_mid = nullptr;
    __half *d_down_routes = nullptr;
    float *d_route_weights = nullptr;
    float *d_tm_out = nullptr;
    int *d_offsets = nullptr;
    cuda_check(cudaMalloc(&d_a, (uint64_t)ROUTES * HIDDEN * sizeof(__half)), "A malloc");
    cuda_check(cudaMalloc(&d_gate_out, (uint64_t)ROUTES * MID * sizeof(__half)), "gate out malloc");
    cuda_check(cudaMalloc(&d_up_out, (uint64_t)ROUTES * MID * sizeof(__half)), "up out malloc");
    cuda_check(cudaMalloc(&d_mid, (uint64_t)ROUTES * MID * sizeof(__half)), "mid malloc");
    cuda_check(cudaMalloc(&d_down_routes, (uint64_t)ROUTES * HIDDEN * sizeof(__half)), "down routes malloc");
    cuda_check(cudaMalloc(&d_route_weights, (uint64_t)ROUTES * sizeof(float)), "route weights malloc");
    cuda_check(cudaMalloc(&d_tm_out, (uint64_t)HIDDEN * sizeof(float)), "tm out malloc");
    cuda_check(cudaMalloc(&d_offsets, (uint64_t)(EXPERTS + 1) * sizeof(int)), "offsets malloc");
    cuda_check(cudaMemcpy(d_a, a_routes.data(), a_routes.size() * sizeof(__half), cudaMemcpyHostToDevice),
               "A upload");
    cuda_check(cudaMemcpy(d_route_weights,
                          route_weights_sorted.data(),
                          route_weights_sorted.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "route weights upload");
    cuda_check(cudaMemcpy(d_offsets,
                          offsets.data(),
                          offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "offsets upload");

    check(tm_mul_mat_grouped(d_a,
                             nullptr,
                             d_offsets,
                             EXPERTS,
                             (const void * const *)gate_pack.d_weights,
                             (const void * const *)gate_pack.d_scales,
                             GGML_TM_DTYPE_MXFP4,
                             MID,
                             HIDDEN,
                             DS4_SRC_MXFP4_BLOCK_ELEMS,
                             gate_pack.k_pack,
                             d_gate_out,
                             nullptr) == 0,
          "TurboMind grouped gate failed");
    check(tm_mul_mat_grouped(d_a,
                             nullptr,
                             d_offsets,
                             EXPERTS,
                             (const void * const *)up_pack.d_weights,
                             (const void * const *)up_pack.d_scales,
                             GGML_TM_DTYPE_MXFP4,
                             MID,
                             HIDDEN,
                             DS4_SRC_MXFP4_BLOCK_ELEMS,
                             up_pack.k_pack,
                             d_up_out,
                             nullptr) == 0,
          "TurboMind grouped up failed");
    swiglu_half_kernel<<<((uint64_t)ROUTES * MID + 255u) / 256u, 256>>>(
        d_mid, d_gate_out, d_up_out, d_route_weights, ROUTES, MID, 10.0f);
    cuda_check(cudaGetLastError(), "SwiGLU launch");
    check(tm_mul_mat_grouped(d_mid,
                             nullptr,
                             d_offsets,
                             EXPERTS,
                             (const void * const *)down_pack.d_weights,
                             (const void * const *)down_pack.d_scales,
                             GGML_TM_DTYPE_MXFP4,
                             HIDDEN,
                             MID,
                             DS4_SRC_MXFP4_BLOCK_ELEMS,
                             down_pack.k_pack,
                             d_down_routes,
                             nullptr) == 0,
          "TurboMind grouped down failed");
    sum_half_routes_to_f32_kernel<<<(HIDDEN + 255u) / 256u, 256>>>(
        d_tm_out, d_down_routes, ROUTES, HIDDEN);
    cuda_check(cudaGetLastError(), "route sum launch");
    cuda_check(cudaDeviceSynchronize(), "adapter synchronize");

    std::vector<float> ref(HIDDEN);
    std::vector<float> got(HIDDEN);
    check(ds4_gpu_tensor_read(ref_out_t, 0, ref.data(), (uint64_t)HIDDEN * sizeof(float)),
          "reference read failed");
    cuda_check(cudaMemcpy(got.data(), d_tm_out, (uint64_t)HIDDEN * sizeof(float), cudaMemcpyDeviceToHost),
               "adapter read failed");

    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    float sum_ref = 0.0f;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < HIDDEN; i++) {
        const float d = fabsf(got[i] - ref[i]);
        max_abs = fmaxf(max_abs, d);
        sum_abs += d;
        sum_ref += fabsf(ref[i]);
        if (!std::isfinite(got[i]) || d > 2.0f) bad++;
    }
    const float rel = sum_ref > 0.0f ? sum_abs / sum_ref : 0.0f;
    const bool rel_ok = sum_ref < 1.0e-5f || rel < 0.08f;
    std::fprintf(stderr,
                 "cuda_v100_turbomind_adapter_smoke: experts=%u routes=%u "
                 "gate_kpack=0x%x down_kpack=0x%x max_abs=%.6g rel=%.6g bad=%u\n",
                 (unsigned)EXPERTS,
                 (unsigned)ROUTES,
                 gate_pack.k_pack,
                 down_pack.k_pack,
                 max_abs,
                 rel,
                 bad);
    check(bad == 0 && rel_ok && max_abs < 2.0f, "adapter output outside tolerance");

    (void)cudaFree(d_offsets);
    (void)cudaFree(d_tm_out);
    (void)cudaFree(d_route_weights);
    (void)cudaFree(d_down_routes);
    (void)cudaFree(d_mid);
    (void)cudaFree(d_up_out);
    (void)cudaFree(d_gate_out);
    (void)cudaFree(d_a);
    ds4_gpu_tensor_free(ref_out_t);
    ds4_gpu_tensor_free(mid_ref_t);
    ds4_gpu_tensor_free(weights_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(hidden_t);
    free_packed(&down_pack);
    free_packed(&up_pack);
    free_packed(&gate_pack);
    (void)cudaFree(d_down);
    (void)cudaFree(d_up);
    (void)cudaFree(d_gate);
    ds4_gpu_arena_close(arena);
    tm_shutdown();
    dlclose(lib);
    ds4_gpu_cleanup();

    if (failures) {
        std::fprintf(stderr, "cuda_v100_turbomind_adapter_smoke: FAIL\n");
        return 1;
    }
    std::fprintf(stderr, "cuda_v100_turbomind_adapter_smoke: PASS\n");
    return 0;
}
