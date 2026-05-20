extern "C" {
#include "ds4_gpu.h"
#include "ds4_pack.h"
#include "ds4_source_formats.h"
#include "ds4_turbomind_pack.h"
}

#include "ggml-turbomind-api.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef int  (*pfn_init)(int);
typedef void (*pfn_shutdown)(void);
typedef int  (*pfn_mul_mat_grouped)(const void *, const int *, const int *, int,
                                    const void * const *, const void * const *,
                                    int, int, int, int, int, void *, void *);

struct alignas(16) StridedPtrH {
    void *p;
    int stride;
};
static_assert(sizeof(StridedPtrH) == 16, "StridedPtrH must match TurboMind StridedPtr");

struct options {
    const char *lib_path = "./build/turbomind-v100/libggml-turbomind.so";
    const char *tm_index_path = "/tmp/ds4-sprint085-tm-pack/turbomind-pack-index.tsv";
    const char *tm_dir = "/tmp/ds4-sprint085-tm-pack";
    const char *source_index_path = "docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv";
    const char *source_path = "/models/DSv4-Flash-256e-fixed.gguf";
    int layer = 0;
};

struct MatrixView {
    ds4_tm_pack_entry e;
    StridedPtrH *d_weights = nullptr;
    StridedPtrH *d_scales = nullptr;
};

static int failures;

static void check(bool cond, const char *msg) {
    if (!cond) {
        std::fprintf(stderr, "cuda_v100_turbomind_sidecar_smoke: %s\n", msg);
        failures++;
    }
}

static bool cuda_check(cudaError_t rc, const char *msg) {
    if (rc != cudaSuccess) {
        std::fprintf(stderr,
                     "cuda_v100_turbomind_sidecar_smoke: %s: %s\n",
                     msg,
                     cudaGetErrorString(rc));
        failures++;
        return false;
    }
    return true;
}

static void usage(FILE *fp) {
    std::fprintf(fp,
                 "Usage: cuda_v100_turbomind_sidecar_smoke [options]\n"
                 "\n"
                 "Options:\n"
                 "  --lib FILE            libggml-turbomind.so path\n"
                 "  --tm-index FILE       turbomind-pack-index.tsv path\n"
                 "  --tm-dir DIR          directory containing gpuN.turbomind\n"
                 "  --source-index FILE   normal DS4 V100 pack-index.tsv path\n"
                 "  --source FILE         source DS4 GGUF path\n"
                 "  --layer N             layer id to validate. Default: 0\n");
}

static int parse_int(const char *s, const char *name) {
    char *end = nullptr;
    errno = 0;
    long v = std::strtol(s, &end, 10);
    if (errno || !end || *end || v < 0 || v > INT32_MAX) {
        std::fprintf(stderr, "cuda_v100_turbomind_sidecar_smoke: invalid %s: %s\n", name, s);
        std::exit(2);
    }
    return (int)v;
}

static void parse_args(int argc, char **argv, options *opt) {
    if (const char *v = std::getenv("DS4_TURBOMIND_LIB")) opt->lib_path = v;
    if (const char *v = std::getenv("DS4_TURBOMIND_INDEX")) opt->tm_index_path = v;
    if (const char *v = std::getenv("DS4_TURBOMIND_DIR")) opt->tm_dir = v;
    if (const char *v = std::getenv("DS4_SOURCE_INDEX")) opt->source_index_path = v;
    if (const char *v = std::getenv("DS4_SOURCE_GGUF")) opt->source_path = v;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        auto need = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "cuda_v100_turbomind_sidecar_smoke: %s requires a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(a, "--lib")) {
            opt->lib_path = need(a);
        } else if (!std::strcmp(a, "--tm-index")) {
            opt->tm_index_path = need(a);
        } else if (!std::strcmp(a, "--tm-dir")) {
            opt->tm_dir = need(a);
        } else if (!std::strcmp(a, "--source-index")) {
            opt->source_index_path = need(a);
        } else if (!std::strcmp(a, "--source")) {
            opt->source_path = need(a);
        } else if (!std::strcmp(a, "--layer")) {
            opt->layer = parse_int(need(a), a);
        } else if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) {
            usage(stdout);
            std::exit(0);
        } else {
            std::fprintf(stderr, "cuda_v100_turbomind_sidecar_smoke: unknown option %s\n", a);
            usage(stderr);
            std::exit(2);
        }
    }
}

static bool load_api(const char *lib_path,
                     void **handle,
                     pfn_init *init,
                     pfn_shutdown *shutdown,
                     pfn_mul_mat_grouped *mul_mat_grouped) {
    *handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!*handle) {
        std::fprintf(stderr,
                     "cuda_v100_turbomind_sidecar_smoke: failed to open %s: %s\n",
                     lib_path,
                     dlerror());
        return false;
    }
    *init = (pfn_init)dlsym(*handle, "ggml_turbomind_init");
    *shutdown = (pfn_shutdown)dlsym(*handle, "ggml_turbomind_shutdown");
    *mul_mat_grouped = (pfn_mul_mat_grouped)dlsym(*handle, "ggml_turbomind_mul_mat_grouped");
    if (!*init || !*shutdown || !*mul_mat_grouped) {
        std::fprintf(stderr,
                     "cuda_v100_turbomind_sidecar_smoke: missing TurboMind C ABI symbol\n");
        return false;
    }
    return true;
}

static std::string path_join(const char *dir, const char *base) {
    std::string out(dir);
    if (!out.empty() && out.back() != '/') out += '/';
    out += base;
    return out;
}

static void read_exact(FILE *fp, uint64_t off, void *dst, size_t bytes, const char *label) {
    if (std::fseek(fp, (long)off, SEEK_SET) != 0) {
        std::fprintf(stderr,
                     "cuda_v100_turbomind_sidecar_smoke: seek %s failed: %s\n",
                     label,
                     std::strerror(errno));
        failures++;
        return;
    }
    if (std::fread(dst, 1, bytes, fp) != bytes) {
        std::fprintf(stderr,
                     "cuda_v100_turbomind_sidecar_smoke: short read for %s\n",
                     label);
        failures++;
    }
}

static bool read_file_prefix(const char *path, uint64_t bytes, std::vector<uint8_t> *out) {
    FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr,
                     "cuda_v100_turbomind_sidecar_smoke: cannot open %s: %s\n",
                     path,
                     std::strerror(errno));
        return false;
    }
    out->assign((size_t)bytes, 0);
    const size_t got = std::fread(out->data(), 1, out->size(), fp);
    std::fclose(fp);
    if (got != out->size()) {
        std::fprintf(stderr,
                     "cuda_v100_turbomind_sidecar_smoke: short read from %s: got %zu need %zu\n",
                     path,
                     got,
                     out->size());
        return false;
    }
    return true;
}

static bool lookup_tm(ds4_tm_pack *pack,
                      int layer,
                      const char *suffix,
                      ds4_tm_pack_entry *out) {
    char semantic[128];
    std::snprintf(semantic, sizeof(semantic), "blk.%d.%s", layer, suffix);
    if (ds4_tm_pack_lookup(pack, semantic, out)) {
        std::fprintf(stderr,
                     "cuda_v100_turbomind_sidecar_smoke: missing TurboMind entry %s\n",
                     semantic);
        return false;
    }
    return true;
}

static bool lookup_source(ds4_pack *pack,
                          int layer,
                          const char *suffix,
                          ds4_pack_entry *out) {
    char semantic[128];
    std::snprintf(semantic, sizeof(semantic), "blk.%d.%s", layer, suffix);
    if (ds4_pack_lookup(pack, semantic, out)) {
        std::fprintf(stderr,
                     "cuda_v100_turbomind_sidecar_smoke: missing source entry %s\n",
                     semantic);
        return false;
    }
    return true;
}

static uint64_t matrix_expert_stride(uint32_t rows, uint32_t cols) {
    return (uint64_t)rows * ds4_src_mxfp4_row_bytes(cols);
}

static bool read_source_experts(FILE *source,
                                const ds4_pack_entry &entry,
                                uint32_t rows,
                                uint32_t cols,
                                uint32_t experts,
                                std::vector<uint8_t> *payload,
                                uint64_t payload_offset) {
    const uint64_t stride = matrix_expert_stride(rows, cols);
    if (entry.byte_length / stride < experts) {
        std::fprintf(stderr,
                     "cuda_v100_turbomind_sidecar_smoke: source entry %s has too few experts\n",
                     entry.semantic_tensor_id);
        return false;
    }
    for (uint32_t expert = 0; expert < experts; expert++) {
        read_exact(source,
                   entry.source_offset + (uint64_t)expert * stride,
                   payload->data() + payload_offset + (uint64_t)expert * stride,
                   (size_t)stride,
                   entry.semantic_tensor_id);
    }
    return failures == 0;
}

static bool build_matrix_view(MatrixView *view, const ds4_tm_pack_entry &e, uint8_t *d_sidecar) {
    view->e = e;
    std::vector<StridedPtrH> h_weights(e.experts_packed);
    std::vector<StridedPtrH> h_scales(e.experts_packed);
    for (uint32_t expert = 0; expert < e.experts_packed; expert++) {
        h_weights[expert] = StridedPtrH{
            d_sidecar + e.weight_offset + (uint64_t)expert * e.weight_bytes_per_expert,
            e.weight_stride};
        h_scales[expert] = StridedPtrH{
            d_sidecar + e.scale_offset + (uint64_t)expert * e.scale_bytes_per_expert,
            e.scale_stride};
    }
    if (!cuda_check(cudaMalloc(&view->d_weights, h_weights.size() * sizeof(StridedPtrH)),
                    "weight table malloc") ||
        !cuda_check(cudaMalloc(&view->d_scales, h_scales.size() * sizeof(StridedPtrH)),
                    "scale table malloc") ||
        !cuda_check(cudaMemcpy(view->d_weights,
                               h_weights.data(),
                               h_weights.size() * sizeof(StridedPtrH),
                               cudaMemcpyHostToDevice),
                    "weight table upload") ||
        !cuda_check(cudaMemcpy(view->d_scales,
                               h_scales.data(),
                               h_scales.size() * sizeof(StridedPtrH),
                               cudaMemcpyHostToDevice),
                    "scale table upload")) {
        return false;
    }
    return true;
}

static ds4_gpu_turbomind_mxfp4_matrix_view gpu_tm_view(const ds4_tm_pack_entry &e) {
    ds4_gpu_turbomind_mxfp4_matrix_view out;
    std::memset(&out, 0, sizeof(out));
    out.weight_offset = e.weight_offset;
    out.scale_offset = e.scale_offset;
    out.weight_bytes_per_expert = e.weight_bytes_per_expert;
    out.scale_bytes_per_expert = e.scale_bytes_per_expert;
    out.n = e.n;
    out.k = e.k;
    out.experts_packed = e.experts_packed;
    out.experts_total = e.experts_total;
    out.k_pack = e.k_pack;
    out.weight_stride = e.weight_stride;
    out.scale_stride = e.scale_stride;
    return out;
}

static void free_matrix_view(MatrixView *view) {
    (void)cudaFree(view->d_weights);
    (void)cudaFree(view->d_scales);
    *view = MatrixView();
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

int main(int argc, char **argv) {
    options opt;
    parse_args(argc, argv, &opt);

    void *lib = nullptr;
    pfn_init tm_init = nullptr;
    pfn_shutdown tm_shutdown = nullptr;
    pfn_mul_mat_grouped tm_mul_mat_grouped = nullptr;
    if (!load_api(opt.lib_path, &lib, &tm_init, &tm_shutdown, &tm_mul_mat_grouped)) {
        return 2;
    }

    check(ds4_gpu_device_count() > 0, "no CUDA devices visible");
    check(ds4_gpu_init(), "ds4_gpu_init failed");
    check(ds4_gpu_set_device(0), "ds4_gpu_set_device failed");
    check(tm_init(0) == 0, "ggml_turbomind_init failed");
    if (failures) return 3;

    char err[512] = {0};
    ds4_tm_pack *tm_pack = nullptr;
    if (ds4_tm_pack_open(&tm_pack, opt.tm_index_path, err, sizeof(err))) {
        std::fprintf(stderr, "cuda_v100_turbomind_sidecar_smoke: %s\n", err);
        return 4;
    }
    ds4_pack *source_pack = nullptr;
    if (ds4_pack_open(&source_pack, opt.source_index_path, err, sizeof(err))) {
        std::fprintf(stderr, "cuda_v100_turbomind_sidecar_smoke: %s\n", err);
        return 4;
    }

    ds4_tm_pack_entry gate_tm;
    ds4_tm_pack_entry up_tm;
    ds4_tm_pack_entry down_tm;
    check(lookup_tm(tm_pack, opt.layer, "ffn_gate_exps.weight", &gate_tm), "gate sidecar lookup failed");
    check(lookup_tm(tm_pack, opt.layer, "ffn_up_exps.weight", &up_tm), "up sidecar lookup failed");
    check(lookup_tm(tm_pack, opt.layer, "ffn_down_exps.weight", &down_tm), "down sidecar lookup failed");
    check(gate_tm.experts_packed == up_tm.experts_packed &&
              gate_tm.experts_packed == down_tm.experts_packed,
          "sidecar expert counts differ");
    check(gate_tm.experts_packed >= 2, "sidecar must contain at least two experts");
    check(gate_tm.k == up_tm.k && gate_tm.n == up_tm.n, "gate/up dimensions differ");
    check(down_tm.n == gate_tm.k && down_tm.k == gate_tm.n, "down dimensions do not invert gate/up");
    check(!std::strcmp(gate_tm.sidecar_file, up_tm.sidecar_file) &&
              !std::strcmp(gate_tm.sidecar_file, down_tm.sidecar_file),
          "selected tensors span multiple sidecar files");
    if (failures) return 5;

    const uint32_t experts = gate_tm.experts_packed;
    const uint32_t hidden = gate_tm.k;
    const uint32_t mid = gate_tm.n;
    const uint32_t routes = std::min<uint32_t>(6, std::max<uint32_t>(2, experts * 2));

    uint64_t required_sidecar_bytes = 0;
    check(ds4_tm_pack_sidecar_bytes(tm_pack, gate_tm.sidecar_file, &required_sidecar_bytes) == 0,
          "sidecar byte accounting failed");
    std::string sidecar_path = path_join(opt.tm_dir, gate_tm.sidecar_file);
    std::vector<uint8_t> sidecar;
    check(read_file_prefix(sidecar_path.c_str(), required_sidecar_bytes, &sidecar),
          "sidecar read failed");
    uint8_t *d_sidecar = nullptr;
    check(cuda_check(cudaMalloc(&d_sidecar, sidecar.size()), "sidecar device alloc"),
          "sidecar alloc failed");
    check(cuda_check(cudaMemcpy(d_sidecar, sidecar.data(), sidecar.size(), cudaMemcpyHostToDevice),
                     "sidecar upload"),
          "sidecar upload failed");
    if (failures) return 6;

    MatrixView gate_view;
    MatrixView up_view;
    MatrixView down_view;
    check(build_matrix_view(&gate_view, gate_tm, d_sidecar), "gate table build failed");
    check(build_matrix_view(&up_view, up_tm, d_sidecar), "up table build failed");
    check(build_matrix_view(&down_view, down_tm, d_sidecar), "down table build failed");
    if (failures) return 7;

    ds4_pack_entry gate_src;
    ds4_pack_entry up_src;
    ds4_pack_entry down_src;
    check(lookup_source(source_pack, opt.layer, "ffn_gate_exps.weight", &gate_src),
          "gate source lookup failed");
    check(lookup_source(source_pack, opt.layer, "ffn_up_exps.weight", &up_src),
          "up source lookup failed");
    check(lookup_source(source_pack, opt.layer, "ffn_down_exps.weight", &down_src),
          "down source lookup failed");
    FILE *source = std::fopen(opt.source_path, "rb");
    check(source != nullptr, "source GGUF open failed");
    if (failures) return 8;

    const uint64_t gate_expert_bytes = matrix_expert_stride(mid, hidden);
    const uint64_t down_expert_bytes = matrix_expert_stride(hidden, mid);
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = gate_offset + (uint64_t)experts * gate_expert_bytes;
    const uint64_t down_offset = up_offset + (uint64_t)experts * gate_expert_bytes;
    const uint64_t payload_bytes = down_offset + (uint64_t)experts * down_expert_bytes;
    std::vector<uint8_t> payload((size_t)payload_bytes);
    check(read_source_experts(source, gate_src, mid, hidden, experts, &payload, gate_offset),
          "gate source read failed");
    check(read_source_experts(source, up_src, mid, hidden, experts, &payload, up_offset),
          "up source read failed");
    check(read_source_experts(source, down_src, hidden, mid, experts, &payload, down_offset),
          "down source read failed");
    std::fclose(source);
    if (failures) return 9;

    ds4_gpu_arena *arena = nullptr;
    check(ds4_gpu_arena_open(&arena, 0, payload_bytes) == 0, "arena open failed");
    check(ds4_gpu_arena_upload(arena, 0, payload.data(), payload_bytes) == 0, "arena upload failed");
    if (failures) return 10;

    std::vector<int32_t> selected(routes);
    std::vector<float> route_weights(routes);
    for (uint32_t route = 0; route < routes; route++) {
        selected[route] = (int32_t)((route * 3u + 1u) % experts);
        route_weights[route] = 0.31f / (1.0f + (float)route * 0.17f);
    }
    std::vector<float> hidden_row(hidden);
    for (uint32_t i = 0; i < hidden; i++) {
        hidden_row[i] = 0.015f +
                        0.003f * sinf((float)i * 0.013f) +
                        0.001f * cosf((float)i * 0.031f);
    }

    ds4_gpu_tensor *hidden_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *selected_t = ds4_gpu_tensor_alloc((uint64_t)routes * sizeof(int32_t));
    ds4_gpu_tensor *weights_t = ds4_gpu_tensor_alloc((uint64_t)routes * sizeof(float));
    ds4_gpu_tensor *mid_ref_t = ds4_gpu_tensor_alloc((uint64_t)routes * mid * sizeof(float));
    ds4_gpu_tensor *ref_out_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    check(hidden_t && selected_t && weights_t && mid_ref_t && ref_out_t,
          "reference tensor allocation failed");
    check(ds4_gpu_tensor_write(hidden_t, 0, hidden_row.data(), (uint64_t)hidden * sizeof(float)),
          "hidden upload failed");
    check(ds4_gpu_tensor_write(selected_t, 0, selected.data(), (uint64_t)routes * sizeof(int32_t)),
          "selected upload failed");
    check(ds4_gpu_tensor_write(weights_t, 0, route_weights.data(), (uint64_t)routes * sizeof(float)),
          "weights upload failed");
    unsetenv("DS4_V100_TURBOMIND_ROUTED_FFN");
    unsetenv("DS4_V100_TURBOMIND_STRICT");
    check(ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_f32(
              arena,
              gate_offset,
              (uint64_t)experts * gate_expert_bytes,
              up_offset,
              (uint64_t)experts * gate_expert_bytes,
              down_offset,
              (uint64_t)experts * down_expert_bytes,
              gate_expert_bytes,
              (uint32_t)ds4_src_mxfp4_row_bytes(hidden),
              down_expert_bytes,
              (uint32_t)ds4_src_mxfp4_row_bytes(mid),
              hidden,
              mid,
              experts,
              selected_t,
              weights_t,
              routes,
              hidden_t,
              mid_ref_t,
              ref_out_t) == 0,
          "source arena reference failed");
    if (failures) return 11;

    std::vector<int> counts(experts, 0);
    for (uint32_t route = 0; route < routes; route++) counts[(uint32_t)selected[route]]++;
    std::vector<int> offsets(experts + 1, 0);
    for (uint32_t expert = 0; expert < experts; expert++) {
        offsets[expert + 1] = offsets[expert] + counts[expert];
    }
    std::vector<int> cursor = offsets;
    std::vector<__half> a_routes((uint64_t)routes * hidden);
    std::vector<float> weights_sorted(routes);
    for (uint32_t route = 0; route < routes; route++) {
        const uint32_t expert = (uint32_t)selected[route];
        const uint32_t row = (uint32_t)cursor[expert]++;
        weights_sorted[row] = route_weights[route];
        for (uint32_t c = 0; c < hidden; c++) {
            a_routes[(uint64_t)row * hidden + c] = __float2half_rn(hidden_row[c]);
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
    cuda_check(cudaMalloc(&d_a, (uint64_t)routes * hidden * sizeof(__half)), "A malloc");
    cuda_check(cudaMalloc(&d_gate_out, (uint64_t)routes * mid * sizeof(__half)), "gate out malloc");
    cuda_check(cudaMalloc(&d_up_out, (uint64_t)routes * mid * sizeof(__half)), "up out malloc");
    cuda_check(cudaMalloc(&d_mid, (uint64_t)routes * mid * sizeof(__half)), "mid malloc");
    cuda_check(cudaMalloc(&d_down_routes, (uint64_t)routes * hidden * sizeof(__half)), "down routes malloc");
    cuda_check(cudaMalloc(&d_route_weights, (uint64_t)routes * sizeof(float)), "route weights malloc");
    cuda_check(cudaMalloc(&d_tm_out, (uint64_t)hidden * sizeof(float)), "tm out malloc");
    cuda_check(cudaMalloc(&d_offsets, (uint64_t)(experts + 1) * sizeof(int)), "offsets malloc");
    cuda_check(cudaMemcpy(d_a, a_routes.data(), a_routes.size() * sizeof(__half), cudaMemcpyHostToDevice),
               "A upload");
    cuda_check(cudaMemcpy(d_route_weights,
                          weights_sorted.data(),
                          weights_sorted.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "route weights upload");
    cuda_check(cudaMemcpy(d_offsets,
                          offsets.data(),
                          offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "offsets upload");
    if (failures) return 12;

    const auto start = std::chrono::steady_clock::now();
    check(tm_mul_mat_grouped(d_a,
                             nullptr,
                             d_offsets,
                             experts,
                             (const void * const *)gate_view.d_weights,
                             (const void * const *)gate_view.d_scales,
                             GGML_TM_DTYPE_MXFP4,
                             mid,
                             hidden,
                             DS4_SRC_MXFP4_BLOCK_ELEMS,
                             gate_view.e.k_pack,
                             d_gate_out,
                             nullptr) == 0,
          "TurboMind sidecar gate failed");
    check(tm_mul_mat_grouped(d_a,
                             nullptr,
                             d_offsets,
                             experts,
                             (const void * const *)up_view.d_weights,
                             (const void * const *)up_view.d_scales,
                             GGML_TM_DTYPE_MXFP4,
                             mid,
                             hidden,
                             DS4_SRC_MXFP4_BLOCK_ELEMS,
                             up_view.e.k_pack,
                             d_up_out,
                             nullptr) == 0,
          "TurboMind sidecar up failed");
    swiglu_half_kernel<<<((uint64_t)routes * mid + 255u) / 256u, 256>>>(
        d_mid, d_gate_out, d_up_out, d_route_weights, routes, mid, 10.0f);
    cuda_check(cudaGetLastError(), "SwiGLU launch");
    check(tm_mul_mat_grouped(d_mid,
                             nullptr,
                             d_offsets,
                             experts,
                             (const void * const *)down_view.d_weights,
                             (const void * const *)down_view.d_scales,
                             GGML_TM_DTYPE_MXFP4,
                             hidden,
                             mid,
                             DS4_SRC_MXFP4_BLOCK_ELEMS,
                             down_view.e.k_pack,
                             d_down_routes,
                             nullptr) == 0,
          "TurboMind sidecar down failed");
    sum_half_routes_to_f32_kernel<<<(hidden + 255u) / 256u, 256>>>(
        d_tm_out, d_down_routes, routes, hidden);
    cuda_check(cudaGetLastError(), "route sum launch");
    cuda_check(cudaDeviceSynchronize(), "sidecar synchronize");
    const auto end = std::chrono::steady_clock::now();

    std::vector<float> ref(hidden);
    std::vector<float> got(hidden);
    check(ds4_gpu_tensor_read(ref_out_t, 0, ref.data(), (uint64_t)hidden * sizeof(float)),
          "reference read failed");
    cuda_check(cudaMemcpy(got.data(), d_tm_out, (uint64_t)hidden * sizeof(float), cudaMemcpyDeviceToHost),
               "sidecar read failed");

    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    float sum_ref = 0.0f;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < hidden; i++) {
        const float d = fabsf(got[i] - ref[i]);
        max_abs = fmaxf(max_abs, d);
        sum_abs += d;
        sum_ref += fabsf(ref[i]);
        if (!std::isfinite(got[i]) || d > 2.0f) bad++;
    }
    const float rel = sum_ref > 0.0f ? sum_abs / sum_ref : 0.0f;
    const bool rel_ok = sum_ref < 1.0e-5f || rel < 0.08f;
    const double host_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::fprintf(stderr,
                 "cuda_v100_turbomind_sidecar_smoke: layer=%d experts=%u routes=%u "
                 "sidecar_bytes=%" PRIu64 " max_abs=%.6g rel=%.6g bad=%u host_ms=%.3f\n",
                 opt.layer,
                 experts,
                 routes,
                 required_sidecar_bytes,
                 max_abs,
                 rel,
                 bad,
                 host_ms);
    check(bad == 0 && rel_ok && max_abs < 2.0f, "sidecar output outside tolerance");

    ds4_gpu_arena *tm_arena = nullptr;
    check(ds4_gpu_arena_open(&tm_arena, 0, sidecar.size()) == 0,
          "TurboMind appliance arena open failed");
    check(ds4_gpu_arena_upload(tm_arena, 0, sidecar.data(), sidecar.size()) == 0,
          "TurboMind appliance arena upload failed");
    ds4_gpu_tensor *api_out_t = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    check(api_out_t != nullptr, "TurboMind packed API output allocation failed");
    ds4_gpu_turbomind_mxfp4_matrix_view gate_gpu = gpu_tm_view(gate_tm);
    ds4_gpu_turbomind_mxfp4_matrix_view up_gpu = gpu_tm_view(up_tm);
    ds4_gpu_turbomind_mxfp4_matrix_view down_gpu = gpu_tm_view(down_tm);
    check(ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_f32(
              tm_arena,
              &gate_gpu,
              &up_gpu,
              &down_gpu,
              hidden,
              mid,
              experts,
              selected_t,
              weights_t,
              routes,
              hidden_t,
              1,
              api_out_t) == 0,
          "DS4 packed TurboMind arena API failed");
    std::vector<float> api_got(hidden);
    check(ds4_gpu_tensor_read(api_out_t, 0, api_got.data(), (uint64_t)hidden * sizeof(float)),
          "packed API read failed");
    float api_max_abs = 0.0f;
    float api_sum_abs = 0.0f;
    float api_sum_ref = 0.0f;
    uint32_t api_bad = 0;
    for (uint32_t i = 0; i < hidden; i++) {
        const float d = fabsf(api_got[i] - ref[i]);
        api_max_abs = fmaxf(api_max_abs, d);
        api_sum_abs += d;
        api_sum_ref += fabsf(ref[i]);
        if (!std::isfinite(api_got[i]) || d > 2.0f) api_bad++;
    }
    const float api_rel = api_sum_ref > 0.0f ? api_sum_abs / api_sum_ref : 0.0f;
    const bool api_rel_ok = api_sum_ref < 1.0e-5f || api_rel < 0.08f;
    std::fprintf(stderr,
                 "cuda_v100_turbomind_sidecar_smoke: packed_api max_abs=%.6g "
                 "rel=%.6g bad=%u\n",
                 api_max_abs,
                 api_rel,
                 api_bad);
    check(api_bad == 0 && api_rel_ok && api_max_abs < 2.0f,
          "packed TurboMind arena API output outside tolerance");

    (void)cudaFree(d_offsets);
    (void)cudaFree(d_tm_out);
    (void)cudaFree(d_route_weights);
    (void)cudaFree(d_down_routes);
    (void)cudaFree(d_mid);
    (void)cudaFree(d_up_out);
    (void)cudaFree(d_gate_out);
    (void)cudaFree(d_a);
    ds4_gpu_tensor_free(ref_out_t);
    ds4_gpu_tensor_free(api_out_t);
    ds4_gpu_tensor_free(mid_ref_t);
    ds4_gpu_tensor_free(weights_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(hidden_t);
    ds4_gpu_arena_close(arena);
    ds4_gpu_arena_close(tm_arena);
    free_matrix_view(&down_view);
    free_matrix_view(&up_view);
    free_matrix_view(&gate_view);
    (void)cudaFree(d_sidecar);
    ds4_pack_close(source_pack);
    ds4_tm_pack_close(tm_pack);
    tm_shutdown();
    dlclose(lib);
    ds4_gpu_cleanup();

    if (failures) {
        std::fprintf(stderr, "cuda_v100_turbomind_sidecar_smoke: FAIL\n");
        return 1;
    }
    std::fprintf(stderr, "cuda_v100_turbomind_sidecar_smoke: PASS\n");
    return 0;
}
