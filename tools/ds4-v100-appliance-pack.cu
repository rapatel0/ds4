#define _FILE_OFFSET_BITS 64

extern "C" {
#include "ds4_pack.h"
#include "ds4_source_formats.h"
}

#include "ggml-turbomind-api.h"

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cerrno>
#include <climits>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

enum {
    MAX_GPUS = 8,
    COPY_BUFFER_BYTES = 8 * 1024 * 1024,
};

typedef int  (*pfn_api_version)(void);
typedef int  (*pfn_init)(int);
typedef void (*pfn_shutdown)(void);
typedef int  (*pfn_packed_bytes)(int, int, int, int, size_t *, size_t *);
typedef int  (*pfn_pack_weight)(const void *, int, int, int, int, void *, void *, int *, void *);

struct tm_api {
    void *handle = nullptr;
    pfn_api_version api_version = nullptr;
    pfn_init init = nullptr;
    pfn_shutdown shutdown = nullptr;
    pfn_packed_bytes packed_bytes = nullptr;
    pfn_pack_weight pack_weight = nullptr;
};

struct options {
    const char *index_path = nullptr;
    const char *source_path = nullptr;
    const char *out_dir = nullptr;
    const char *lib_path = "./build/turbomind-v100/libggml-turbomind.so";
    uint32_t gpus = 8;
    uint64_t alignment = 256;
    int pack_gpu = 0;
    int only_gpu = -1;
    int layer_filter = -1;
    uint32_t expert_limit = 0;
    bool skip_non_experts = false;
    bool fuse_gate_up = false;
    bool keep_separate_gate_up = false;
};

struct shape3 {
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t experts = 0;
};

struct pack_state {
    options opt;
    tm_api api;
    FILE *source = nullptr;
    ds4_pack *pack = nullptr;
    FILE *pack_index = nullptr;
    FILE *tm_index = nullptr;
    FILE *gpu_files[MAX_GPUS] = {nullptr};
    uint64_t cursor[MAX_GPUS] = {0};
    uint64_t source_rows = 0;
    uint64_t tm_rows = 0;
    uint64_t skipped_rows = 0;
    uint64_t source_bytes = 0;
    uint64_t tm_weight_bytes = 0;
    uint64_t tm_scale_bytes = 0;
};

static void die(const char *msg) {
    std::fprintf(stderr, "ds4-v100-appliance-pack: %s\n", msg);
    std::exit(1);
}

static void die_errno(const char *what, const char *path) {
    std::fprintf(stderr,
                 "ds4-v100-appliance-pack: %s %s: %s\n",
                 what,
                 path,
                 std::strerror(errno));
    std::exit(1);
}

static bool cuda_ok(cudaError_t rc, const char *what) {
    if (rc == cudaSuccess) return true;
    std::fprintf(stderr,
                 "ds4-v100-appliance-pack: %s: %s\n",
                 what,
                 cudaGetErrorString(rc));
    return false;
}

static uint64_t parse_u64(const char *s, const char *name) {
    char *end = nullptr;
    errno = 0;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (errno || !end || *end) {
        std::fprintf(stderr, "ds4-v100-appliance-pack: invalid %s: %s\n", name, s);
        std::exit(2);
    }
    return (uint64_t)v;
}

static int parse_i32(const char *s, const char *name) {
    char *end = nullptr;
    errno = 0;
    long v = std::strtol(s, &end, 10);
    if (errno || !end || *end || v < 0 || v > INT_MAX) {
        std::fprintf(stderr, "ds4-v100-appliance-pack: invalid %s: %s\n", name, s);
        std::exit(2);
    }
    return (int)v;
}

static void usage(FILE *fp) {
    std::fprintf(fp,
                 "Usage: ds4-v100-appliance-pack --index FILE --source GGUF --out-dir DIR [options]\n"
                 "\n"
                 "Options:\n"
                 "  --gpus N                 Number of GPU shard files. Default: 8\n"
                 "  --align N                Shard alignment in bytes. Default: 256\n"
                 "  --lib FILE               libggml-turbomind.so path\n"
                 "  --pack-gpu N             CUDA device used for offline packing. Default: 0\n"
                 "  --only-gpu N             Bounded validation: emit rows only for owning GPU N\n"
                 "  --layer N                Bounded validation: TurboMind-pack routed experts only for layer N\n"
                 "  --expert-limit N         Bounded validation: pack first N experts per routed tensor\n"
                 "  --fuse-gate-up           Emit fused gate_up routed expert tensors\n"
                 "  --keep-separate-gate-up  With --fuse-gate-up, also emit separate gate/up tensors\n"
                 "  --skip-non-experts       Bounded validation: do not copy non-selected tensors\n"
                 "\n"
                 "Without bounded options this emits one production-shaped appliance directory:\n"
                 "gpuN.weights plus pack-index.tsv and turbomind-pack-index.tsv.\n");
}

static void parse_args(int argc, char **argv, options *opt) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        auto need = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "ds4-v100-appliance-pack: %s requires a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(a, "--index")) {
            opt->index_path = need(a);
        } else if (!std::strcmp(a, "--source")) {
            opt->source_path = need(a);
        } else if (!std::strcmp(a, "--out-dir")) {
            opt->out_dir = need(a);
        } else if (!std::strcmp(a, "--gpus")) {
            opt->gpus = (uint32_t)parse_u64(need(a), a);
        } else if (!std::strcmp(a, "--align")) {
            opt->alignment = parse_u64(need(a), a);
        } else if (!std::strcmp(a, "--lib")) {
            opt->lib_path = need(a);
        } else if (!std::strcmp(a, "--pack-gpu")) {
            opt->pack_gpu = parse_i32(need(a), a);
        } else if (!std::strcmp(a, "--only-gpu")) {
            opt->only_gpu = parse_i32(need(a), a);
        } else if (!std::strcmp(a, "--layer")) {
            opt->layer_filter = parse_i32(need(a), a);
        } else if (!std::strcmp(a, "--expert-limit")) {
            opt->expert_limit = (uint32_t)parse_u64(need(a), a);
        } else if (!std::strcmp(a, "--fuse-gate-up")) {
            opt->fuse_gate_up = true;
        } else if (!std::strcmp(a, "--keep-separate-gate-up")) {
            opt->keep_separate_gate_up = true;
        } else if (!std::strcmp(a, "--skip-non-experts")) {
            opt->skip_non_experts = true;
        } else if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) {
            usage(stdout);
            std::exit(0);
        } else {
            std::fprintf(stderr, "ds4-v100-appliance-pack: unknown option %s\n", a);
            usage(stderr);
            std::exit(2);
        }
    }
    if (!opt->index_path || !opt->source_path || !opt->out_dir) {
        usage(stderr);
        std::exit(2);
    }
    if (opt->gpus == 0 || opt->gpus > MAX_GPUS) die("--gpus must be in 1..8");
    if (opt->only_gpu >= (int)opt->gpus) die("--only-gpu must be less than --gpus");
    if (opt->alignment == 0) die("--align must be positive");
    if (opt->keep_separate_gate_up && !opt->fuse_gate_up) {
        die("--keep-separate-gate-up requires --fuse-gate-up");
    }
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    const uint64_t rem = value % alignment;
    if (!rem) return value;
    const uint64_t delta = alignment - rem;
    if (value > UINT64_MAX - delta) die("offset overflow during alignment");
    return value + delta;
}

static void mkdir_if_needed(const char *path) {
    if (mkdir(path, 0775) == 0) return;
    if (errno == EEXIST) return;
    die_errno("cannot create output directory", path);
}

static std::string path_join(const char *dir, const char *base) {
    std::string out(dir);
    if (!out.empty() && out.back() != '/') out += '/';
    out += base;
    return out;
}

static bool load_api(const char *path, tm_api *api) {
    api->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!api->handle) {
        std::fprintf(stderr,
                     "ds4-v100-appliance-pack: failed to open %s: %s\n",
                     path,
                     dlerror());
        return false;
    }
    api->api_version = (pfn_api_version)dlsym(api->handle, "ggml_turbomind_api_version");
    api->init = (pfn_init)dlsym(api->handle, "ggml_turbomind_init");
    api->shutdown = (pfn_shutdown)dlsym(api->handle, "ggml_turbomind_shutdown");
    api->packed_bytes = (pfn_packed_bytes)dlsym(api->handle, "ggml_turbomind_packed_bytes");
    api->pack_weight = (pfn_pack_weight)dlsym(api->handle, "ggml_turbomind_pack_weight_expert");
    if (!api->api_version || !api->init || !api->shutdown ||
        !api->packed_bytes || !api->pack_weight) {
        std::fprintf(stderr, "ds4-v100-appliance-pack: missing TurboMind C ABI symbol\n");
        return false;
    }
    if (api->api_version() != GGML_TURBOMIND_API_VERSION) {
        std::fprintf(stderr,
                     "ds4-v100-appliance-pack: TurboMind ABI mismatch got %d expected %d\n",
                     api->api_version(),
                     GGML_TURBOMIND_API_VERSION);
        return false;
    }
    return true;
}

static bool parse_shape3(const char *s, shape3 *out) {
    unsigned k = 0;
    unsigned n = 0;
    unsigned experts = 0;
    char tail = 0;
    if (!s || !out || std::sscanf(s, "[%ux%ux%u]%c", &k, &n, &experts, &tail) != 3) {
        return false;
    }
    out->k = k;
    out->n = n;
    out->experts = experts;
    return out->k && out->n && out->experts;
}

static bool is_routed_expert(const ds4_pack_entry *e) {
    if (!e || !e->semantic_tensor_id) return false;
    return std::strstr(e->semantic_tensor_id, ".ffn_gate_exps.weight") ||
           std::strstr(e->semantic_tensor_id, ".ffn_up_exps.weight") ||
           std::strstr(e->semantic_tensor_id, ".ffn_down_exps.weight");
}

static bool is_gate_expert(const ds4_pack_entry *e) {
    return e && e->semantic_tensor_id &&
           std::strstr(e->semantic_tensor_id, ".ffn_gate_exps.weight");
}

static bool is_up_expert(const ds4_pack_entry *e) {
    return e && e->semantic_tensor_id &&
           std::strstr(e->semantic_tensor_id, ".ffn_up_exps.weight");
}

static void read_exact(FILE *fp, uint64_t off, void *dst, size_t bytes, const char *label) {
    if (fseeko(fp, (off_t)off, SEEK_SET) != 0) die_errno("cannot seek source", label);
    if (std::fread(dst, 1, bytes, fp) != bytes) die_errno("cannot read source", label);
}

static void write_exact(FILE *fp, uint64_t off, const void *src, size_t bytes, const char *label) {
    if (fseeko(fp, (off_t)off, SEEK_SET) != 0) die_errno("cannot seek output", label);
    if (bytes && std::fwrite(src, 1, bytes, fp) != bytes) die_errno("cannot write output", label);
}

static void copy_source_payload(pack_state *st,
                                const ds4_pack_entry *e,
                                uint64_t shard_offset) {
    std::vector<uint8_t> buf(COPY_BUFFER_BYTES);
    uint64_t done = 0;
    while (done < e->byte_length) {
        uint64_t n = e->byte_length - done;
        if (n > COPY_BUFFER_BYTES) n = COPY_BUFFER_BYTES;
        read_exact(st->source, e->source_offset + done, buf.data(), (size_t)n, e->source_name);
        write_exact(st->gpu_files[e->owning_gpu],
                    shard_offset + done,
                    buf.data(),
                    (size_t)n,
                    e->semantic_tensor_id);
        done += n;
    }
}

static void write_source_index_row(pack_state *st,
                                   const ds4_pack_entry *e,
                                   uint64_t shard_offset) {
    std::fprintf(st->pack_index,
                 "%s\t%s\t%s\t%s\t%s\t%d\t%d\t%s\t%" PRIu64 "\t%" PRIu64
                 "\tgpu%d.weights\t%" PRIu64 "\t%" PRId64 "\tpending\n",
                 e->semantic_tensor_id,
                 e->source_name,
                 e->source_dtype,
                 e->source_shape,
                 e->runtime_layout,
                 e->owning_gpu,
                 e->layer_id,
                 e->kernel_family,
                 e->source_offset,
                 e->byte_length,
                 e->owning_gpu,
                 shard_offset,
                 e->scale_offset);
}

static void emit_source_tensor(pack_state *st, const ds4_pack_entry *e) {
    const uint32_t gpu = (uint32_t)e->owning_gpu;
    st->cursor[gpu] = align_up(st->cursor[gpu], st->opt.alignment);
    const uint64_t shard_offset = st->cursor[gpu];
    if (e->byte_length > UINT64_MAX - st->cursor[gpu]) die("source shard offset overflow");
    st->cursor[gpu] += e->byte_length;
    copy_source_payload(st, e, shard_offset);
    write_source_index_row(st, e, shard_offset);
    st->source_rows++;
    st->source_bytes += e->byte_length;
}

static void emit_turbomind_tensor(pack_state *st, const ds4_pack_entry *e) {
    if (std::strcmp(e->source_dtype, "mxfp4") != 0) {
        die("routed expert tensor is not mxfp4");
    }
    shape3 shape;
    if (!parse_shape3(e->source_shape, &shape)) {
        std::fprintf(stderr,
                     "ds4-v100-appliance-pack: cannot parse shape %s for %s\n",
                     e->source_shape,
                     e->semantic_tensor_id);
        std::exit(1);
    }
    const uint64_t row_bytes = ds4_src_mxfp4_row_bytes(shape.k);
    const uint64_t expert_stride = row_bytes * shape.n;
    if (expert_stride == 0 || e->byte_length / expert_stride < shape.experts) {
        die("MXFP4 expert stride does not match pack-index byte length");
    }
    const uint32_t experts_to_pack =
        st->opt.expert_limit && st->opt.expert_limit < shape.experts ?
        st->opt.expert_limit : shape.experts;

    size_t weight_bytes = 0;
    size_t scale_bytes = 0;
    if (st->api.packed_bytes(GGML_TM_DTYPE_MXFP4,
                             (int)shape.n,
                             (int)shape.k,
                             DS4_SRC_MXFP4_BLOCK_ELEMS,
                             &weight_bytes,
                             &scale_bytes) != 0 ||
        !weight_bytes || !scale_bytes) {
        die("TurboMind packed_bytes failed");
    }

    uint8_t *d_src = nullptr;
    uint8_t *d_weight = nullptr;
    uint8_t *d_scale = nullptr;
    if (!cuda_ok(cudaMalloc(&d_src, (size_t)expert_stride), "source device alloc") ||
        !cuda_ok(cudaMalloc(&d_weight, weight_bytes), "weight device alloc") ||
        !cuda_ok(cudaMalloc(&d_scale, scale_bytes), "scale device alloc")) {
        std::exit(1);
    }

    std::vector<uint8_t> host_src((size_t)expert_stride);
    std::vector<uint8_t> host_weight(weight_bytes);
    std::vector<uint8_t> host_scale(scale_bytes);

    const uint32_t gpu = (uint32_t)e->owning_gpu;
    st->cursor[gpu] = align_up(st->cursor[gpu], st->opt.alignment);
    const uint64_t weight_offset = st->cursor[gpu];
    const uint64_t weight_total = (uint64_t)experts_to_pack * weight_bytes;
    if (weight_total > UINT64_MAX - st->cursor[gpu]) die("TurboMind weight offset overflow");
    st->cursor[gpu] += weight_total;
    st->cursor[gpu] = align_up(st->cursor[gpu], st->opt.alignment);
    const uint64_t scale_offset = st->cursor[gpu];
    const uint64_t scale_total = (uint64_t)experts_to_pack * scale_bytes;
    if (scale_total > UINT64_MAX - st->cursor[gpu]) die("TurboMind scale offset overflow");
    st->cursor[gpu] += scale_total;

    int expected_k_pack = 0;
    for (uint32_t expert = 0; expert < experts_to_pack; expert++) {
        read_exact(st->source,
                   e->source_offset + (uint64_t)expert * expert_stride,
                   host_src.data(),
                   host_src.size(),
                   e->source_name);
        if (!cuda_ok(cudaMemcpy(d_src,
                                host_src.data(),
                                host_src.size(),
                                cudaMemcpyHostToDevice),
                     "source upload")) {
            std::exit(1);
        }
        int k_pack = 0;
        if (st->api.pack_weight(d_src,
                                GGML_TM_DTYPE_MXFP4,
                                (int)shape.n,
                                (int)shape.k,
                                DS4_SRC_MXFP4_BLOCK_ELEMS,
                                d_weight,
                                d_scale,
                                &k_pack,
                                nullptr) != 0) {
            die("TurboMind pack_weight_expert failed");
        }
        if (expert == 0) {
            expected_k_pack = k_pack;
        } else if (k_pack != expected_k_pack) {
            die("TurboMind k_pack changed across experts");
        }
        if (!cuda_ok(cudaMemcpy(host_weight.data(),
                                d_weight,
                                weight_bytes,
                                cudaMemcpyDeviceToHost),
                     "packed weight download") ||
            !cuda_ok(cudaMemcpy(host_scale.data(),
                                d_scale,
                                scale_bytes,
                                cudaMemcpyDeviceToHost),
                     "packed scale download")) {
            std::exit(1);
        }
        write_exact(st->gpu_files[gpu],
                    weight_offset + (uint64_t)expert * weight_bytes,
                    host_weight.data(),
                    host_weight.size(),
                    e->semantic_tensor_id);
        write_exact(st->gpu_files[gpu],
                    scale_offset + (uint64_t)expert * scale_bytes,
                    host_scale.data(),
                    host_scale.size(),
                    e->semantic_tensor_id);
    }

    std::fprintf(st->tm_index,
                 "%s\t%s\tmxfp4\t%s\tturbomind_mxfp4_grouped\t%d\t%d\t"
                 "turbomind_mxfp4_grouped_sm70\t%u\t%u\t%u\t%u\t%zu\t%zu\t%d\t%d\t%d\t"
                 "gpu%d.weights\t%" PRIu64 "\t%" PRIu64 "\t%s\t%" PRIu64 "\t%" PRIu64
                 "\tpending\t%d\n",
                 e->semantic_tensor_id,
                 e->source_name,
                 e->source_shape,
                 e->owning_gpu,
                 e->layer_id,
                 shape.n,
                 shape.k,
                 experts_to_pack,
                 shape.experts,
                 weight_bytes,
                 scale_bytes,
                 expected_k_pack,
                 (int)shape.k * 32,
                 (int)shape.n,
                 e->owning_gpu,
                 weight_offset,
                 scale_offset,
                 e->shard_file,
                 e->shard_offset,
                 e->byte_length,
                 GGML_TURBOMIND_API_VERSION);

    st->tm_rows++;
    st->tm_weight_bytes += weight_total;
    st->tm_scale_bytes += scale_total;
    std::fprintf(stderr,
                 "appliance packed %s experts=%u/%u gpu=%u weight_offset=%" PRIu64
                 " scale_offset=%" PRIu64 " k_pack=0x%x\n",
                 e->semantic_tensor_id,
                 experts_to_pack,
                 shape.experts,
                 gpu,
                 weight_offset,
                 scale_offset,
                 expected_k_pack);

    (void)cudaFree(d_scale);
    (void)cudaFree(d_weight);
    (void)cudaFree(d_src);
}

static void emit_fused_gate_up_turbomind_tensor(pack_state *st, const ds4_pack_entry *gate) {
    if (!st || !st->pack || !gate || !is_gate_expert(gate)) {
        die("invalid fused gate_up request");
    }
    if (std::strcmp(gate->source_dtype, "mxfp4") != 0) {
        die("gate routed expert tensor is not mxfp4");
    }

    std::string up_semantic(gate->semantic_tensor_id);
    const char *needle = "ffn_gate_exps.weight";
    const size_t pos = up_semantic.find(needle);
    if (pos == std::string::npos) die("cannot derive up tensor semantic id");
    up_semantic.replace(pos, std::strlen(needle), "ffn_up_exps.weight");

    ds4_pack_entry up;
    if (ds4_pack_lookup(st->pack, up_semantic.c_str(), &up)) {
        std::fprintf(stderr,
                     "ds4-v100-appliance-pack: missing paired up tensor %s\n",
                     up_semantic.c_str());
        std::exit(1);
    }
    if (std::strcmp(up.source_dtype, "mxfp4") != 0) {
        die("up routed expert tensor is not mxfp4");
    }
    if (gate->owning_gpu != up.owning_gpu ||
        gate->layer_id != up.layer_id ||
        std::strcmp(gate->source_shape, up.source_shape) != 0) {
        die("gate/up routed expert tensors are not pair-compatible");
    }

    shape3 shape;
    if (!parse_shape3(gate->source_shape, &shape)) {
        std::fprintf(stderr,
                     "ds4-v100-appliance-pack: cannot parse shape %s for %s\n",
                     gate->source_shape,
                     gate->semantic_tensor_id);
        std::exit(1);
    }
    const uint64_t row_bytes = ds4_src_mxfp4_row_bytes(shape.k);
    const uint64_t expert_stride = row_bytes * shape.n;
    if (shape.n > UINT32_MAX / 2u) die("fused gate_up N overflow");
    const uint32_t fused_n = shape.n * 2u;
    const uint64_t fused_expert_stride = expert_stride * 2u;
    if (expert_stride == 0 ||
        gate->byte_length / expert_stride < shape.experts ||
        up.byte_length / expert_stride < shape.experts) {
        die("MXFP4 gate/up expert stride does not match pack-index byte length");
    }
    const uint32_t experts_to_pack =
        st->opt.expert_limit && st->opt.expert_limit < shape.experts ?
        st->opt.expert_limit : shape.experts;

    size_t weight_bytes = 0;
    size_t scale_bytes = 0;
    if (st->api.packed_bytes(GGML_TM_DTYPE_MXFP4,
                             (int)fused_n,
                             (int)shape.k,
                             DS4_SRC_MXFP4_BLOCK_ELEMS,
                             &weight_bytes,
                             &scale_bytes) != 0 ||
        !weight_bytes || !scale_bytes) {
        die("TurboMind fused packed_bytes failed");
    }

    uint8_t *d_src = nullptr;
    uint8_t *d_weight = nullptr;
    uint8_t *d_scale = nullptr;
    if (!cuda_ok(cudaMalloc(&d_src, (size_t)fused_expert_stride), "fused source device alloc") ||
        !cuda_ok(cudaMalloc(&d_weight, weight_bytes), "fused weight device alloc") ||
        !cuda_ok(cudaMalloc(&d_scale, scale_bytes), "fused scale device alloc")) {
        std::exit(1);
    }

    std::vector<uint8_t> host_gate((size_t)expert_stride);
    std::vector<uint8_t> host_up((size_t)expert_stride);
    std::vector<uint8_t> host_fused((size_t)fused_expert_stride);
    std::vector<uint8_t> host_weight(weight_bytes);
    std::vector<uint8_t> host_scale(scale_bytes);

    const uint32_t gpu = (uint32_t)gate->owning_gpu;
    st->cursor[gpu] = align_up(st->cursor[gpu], st->opt.alignment);
    const uint64_t weight_offset = st->cursor[gpu];
    const uint64_t weight_total = (uint64_t)experts_to_pack * weight_bytes;
    if (weight_total > UINT64_MAX - st->cursor[gpu]) die("fused TurboMind weight offset overflow");
    st->cursor[gpu] += weight_total;
    st->cursor[gpu] = align_up(st->cursor[gpu], st->opt.alignment);
    const uint64_t scale_offset = st->cursor[gpu];
    const uint64_t scale_total = (uint64_t)experts_to_pack * scale_bytes;
    if (scale_total > UINT64_MAX - st->cursor[gpu]) die("fused TurboMind scale offset overflow");
    st->cursor[gpu] += scale_total;

    int expected_k_pack = 0;
    for (uint32_t expert = 0; expert < experts_to_pack; expert++) {
        read_exact(st->source,
                   gate->source_offset + (uint64_t)expert * expert_stride,
                   host_gate.data(),
                   host_gate.size(),
                   gate->source_name);
        read_exact(st->source,
                   up.source_offset + (uint64_t)expert * expert_stride,
                   host_up.data(),
                   host_up.size(),
                   up.source_name);
        std::memcpy(host_fused.data(), host_gate.data(), host_gate.size());
        std::memcpy(host_fused.data() + host_gate.size(), host_up.data(), host_up.size());
        if (!cuda_ok(cudaMemcpy(d_src,
                                host_fused.data(),
                                host_fused.size(),
                                cudaMemcpyHostToDevice),
                     "fused source upload")) {
            std::exit(1);
        }
        int k_pack = 0;
        if (st->api.pack_weight(d_src,
                                GGML_TM_DTYPE_MXFP4,
                                (int)fused_n,
                                (int)shape.k,
                                DS4_SRC_MXFP4_BLOCK_ELEMS,
                                d_weight,
                                d_scale,
                                &k_pack,
                                nullptr) != 0) {
            die("TurboMind fused pack_weight_expert failed");
        }
        if (expert == 0) {
            expected_k_pack = k_pack;
        } else if (k_pack != expected_k_pack) {
            die("TurboMind fused k_pack changed across experts");
        }
        if (!cuda_ok(cudaMemcpy(host_weight.data(),
                                d_weight,
                                weight_bytes,
                                cudaMemcpyDeviceToHost),
                     "fused packed weight download") ||
            !cuda_ok(cudaMemcpy(host_scale.data(),
                                d_scale,
                                scale_bytes,
                                cudaMemcpyDeviceToHost),
                     "fused packed scale download")) {
            std::exit(1);
        }
        write_exact(st->gpu_files[gpu],
                    weight_offset + (uint64_t)expert * weight_bytes,
                    host_weight.data(),
                    host_weight.size(),
                    gate->semantic_tensor_id);
        write_exact(st->gpu_files[gpu],
                    scale_offset + (uint64_t)expert * scale_bytes,
                    host_scale.data(),
                    host_scale.size(),
                    gate->semantic_tensor_id);
    }

    std::string fused_semantic(gate->semantic_tensor_id);
    fused_semantic.replace(pos, std::strlen(needle), "ffn_gate_up_exps.weight");
    char fused_shape[64];
    std::snprintf(fused_shape, sizeof(fused_shape), "[%ux%ux%u]", shape.k, fused_n, shape.experts);
    std::string fused_source_name(gate->source_name);
    fused_source_name += "+";
    fused_source_name += up.source_name;
    if (gate->byte_length > UINT64_MAX - up.byte_length) die("fused source byte length overflow");
    const uint64_t source_bytes = gate->byte_length + up.byte_length;

    std::fprintf(st->tm_index,
                 "%s\t%s\tmxfp4\t%s\tturbomind_mxfp4_grouped\t%d\t%d\t"
                 "turbomind_mxfp4_grouped_sm70\t%u\t%u\t%u\t%u\t%zu\t%zu\t%d\t%d\t%d\t"
                 "gpu%d.weights\t%" PRIu64 "\t%" PRIu64 "\t%s\t%" PRIu64 "\t%" PRIu64
                 "\tpending\t%d\n",
                 fused_semantic.c_str(),
                 fused_source_name.c_str(),
                 fused_shape,
                 gate->owning_gpu,
                 gate->layer_id,
                 fused_n,
                 shape.k,
                 experts_to_pack,
                 shape.experts,
                 weight_bytes,
                 scale_bytes,
                 expected_k_pack,
                 (int)shape.k * 32,
                 (int)fused_n,
                 gate->owning_gpu,
                 weight_offset,
                 scale_offset,
                 gate->shard_file,
                 gate->shard_offset,
                 source_bytes,
                 GGML_TURBOMIND_API_VERSION);

    st->tm_rows++;
    st->tm_weight_bytes += weight_total;
    st->tm_scale_bytes += scale_total;
    std::fprintf(stderr,
                 "appliance packed %s experts=%u/%u gpu=%u fused_N=%u weight_offset=%" PRIu64
                 " scale_offset=%" PRIu64 " k_pack=0x%x\n",
                 fused_semantic.c_str(),
                 experts_to_pack,
                 shape.experts,
                 gpu,
                 fused_n,
                 weight_offset,
                 scale_offset,
                 expected_k_pack);

    (void)cudaFree(d_scale);
    (void)cudaFree(d_weight);
    (void)cudaFree(d_src);
}

static int emit_entry_cb(const ds4_pack_entry *e, void *ud) {
    pack_state *st = (pack_state *)ud;
    if (e->owning_gpu < 0 || e->owning_gpu >= (int)st->opt.gpus) {
        std::fprintf(stderr,
                     "ds4-v100-appliance-pack: %s has invalid owning_gpu=%d\n",
                     e->semantic_tensor_id,
                     e->owning_gpu);
        return 1;
    }
    if (st->opt.only_gpu >= 0 && e->owning_gpu != st->opt.only_gpu) {
        st->skipped_rows++;
        return 0;
    }
    if (is_routed_expert(e)) {
        if (st->opt.layer_filter >= 0 && e->layer_id != st->opt.layer_filter) {
            if (st->opt.skip_non_experts) {
                st->skipped_rows++;
            } else {
                emit_source_tensor(st, e);
            }
            return 0;
        }
        if (st->opt.fuse_gate_up && is_gate_expert(e)) {
            emit_fused_gate_up_turbomind_tensor(st, e);
            if (!st->opt.keep_separate_gate_up) return 0;
        }
        if (st->opt.fuse_gate_up && is_up_expert(e) && !st->opt.keep_separate_gate_up) {
            st->skipped_rows++;
            return 0;
        }
        emit_turbomind_tensor(st, e);
        return 0;
    }
    if (st->opt.skip_non_experts) {
        st->skipped_rows++;
        return 0;
    }
    emit_source_tensor(st, e);
    return 0;
}

static void open_outputs(pack_state *st) {
    mkdir_if_needed(st->opt.out_dir);
    std::string pack_index = path_join(st->opt.out_dir, "pack-index.tsv");
    std::string tm_index = path_join(st->opt.out_dir, "turbomind-pack-index.tsv");
    st->pack_index = std::fopen(pack_index.c_str(), "wb");
    if (!st->pack_index) die_errno("cannot create", pack_index.c_str());
    st->tm_index = std::fopen(tm_index.c_str(), "wb");
    if (!st->tm_index) die_errno("cannot create", tm_index.c_str());
    std::fprintf(st->pack_index,
                 "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\t"
                 "runtime_layout\towning_gpu\tlayer_id\tkernel_family\t"
                 "source_offset\tbyte_length\tshard_file\tshard_offset\t"
                 "scale_offset\tchecksum\n");
    std::fprintf(st->tm_index,
                 "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\t"
                 "runtime_layout\towning_gpu\tlayer_id\tkernel_family\t"
                 "n\tk\texperts_packed\texperts_total\tweight_bytes_per_expert\t"
                 "scale_bytes_per_expert\tk_pack\tweight_stride\tscale_stride\t"
                 "sidecar_file\tweight_offset\tscale_offset\tsource_shard_file\t"
                 "source_shard_offset\tsource_byte_length\tsource_checksum\t"
                 "tm_abi_version\n");
    for (uint32_t gpu = 0; gpu < st->opt.gpus; gpu++) {
        char base[64];
        std::snprintf(base, sizeof(base), "gpu%u.weights", gpu);
        std::string path = path_join(st->opt.out_dir, base);
        st->gpu_files[gpu] = std::fopen(path.c_str(), "wb");
        if (!st->gpu_files[gpu]) die_errno("cannot create", path.c_str());
    }
}

static void close_outputs(pack_state *st) {
    if (st->pack_index && std::fclose(st->pack_index) != 0) die("cannot close pack-index.tsv");
    if (st->tm_index && std::fclose(st->tm_index) != 0) die("cannot close turbomind-pack-index.tsv");
    for (uint32_t gpu = 0; gpu < st->opt.gpus; gpu++) {
        FILE *fp = st->gpu_files[gpu];
        if (!fp) continue;
        if (fflush(fp) != 0) die("cannot flush GPU shard");
        if (ftruncate(fileno(fp), (off_t)st->cursor[gpu]) != 0) die("cannot truncate GPU shard");
        if (std::fclose(fp) != 0) die("cannot close GPU shard");
    }
}

int main(int argc, char **argv) {
    pack_state st;
    parse_args(argc, argv, &st.opt);

    if (!cuda_ok(cudaSetDevice(st.opt.pack_gpu), "set CUDA device")) return 1;
    if (!load_api(st.opt.lib_path, &st.api)) return 1;
    if (st.api.init(st.opt.pack_gpu) != 0) die("ggml_turbomind_init failed");

    char err[512] = {0};
    ds4_pack *pack = nullptr;
    if (ds4_pack_open(&pack, st.opt.index_path, err, sizeof(err))) {
        std::fprintf(stderr, "ds4-v100-appliance-pack: %s\n", err);
        return 1;
    }
    st.pack = pack;

    st.source = std::fopen(st.opt.source_path, "rb");
    if (!st.source) die_errno("cannot open source", st.opt.source_path);
    open_outputs(&st);

    if (ds4_pack_for_each(pack, emit_entry_cb, &st) != 0) return 1;
    close_outputs(&st);
    if (std::fclose(st.source) != 0) die_errno("cannot close source", st.opt.source_path);
    ds4_pack_close(pack);
    st.pack = nullptr;
    st.api.shutdown();
    dlclose(st.api.handle);

    std::fprintf(stderr,
                 "ds4-v100-appliance-pack: wrote %s source_rows=%" PRIu64
                 " tm_rows=%" PRIu64 " skipped_rows=%" PRIu64
                 " source_bytes=%" PRIu64 " tm_weight_bytes=%" PRIu64
                 " tm_scale_bytes=%" PRIu64 "\n",
                 st.opt.out_dir,
                 st.source_rows,
                 st.tm_rows,
                 st.skipped_rows,
                 st.source_bytes,
                 st.tm_weight_bytes,
                 st.tm_scale_bytes);
    for (uint32_t gpu = 0; gpu < st.opt.gpus; gpu++) {
        std::fprintf(stderr,
                     "ds4-v100-appliance-pack: gpu%u.weights bytes=%" PRIu64 "\n",
                     gpu,
                     st.cursor[gpu]);
    }
    return 0;
}
