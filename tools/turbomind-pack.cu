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
#include <vector>

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
    int gpu = 0;
    int layer = -1;
    std::string kind = "all";
    uint32_t expert_limit = 0;
};

struct shape3 {
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t experts = 0;
};

static void die(const char *msg) {
    std::fprintf(stderr, "ds4-v100-turbomind-pack: %s\n", msg);
    std::exit(1);
}

static void die_errno(const char *what, const char *path) {
    std::fprintf(stderr,
                 "ds4-v100-turbomind-pack: %s %s: %s\n",
                 what,
                 path,
                 std::strerror(errno));
    std::exit(1);
}

static bool cuda_ok(cudaError_t rc, const char *what) {
    if (rc == cudaSuccess) return true;
    std::fprintf(stderr,
                 "ds4-v100-turbomind-pack: %s: %s\n",
                 what,
                 cudaGetErrorString(rc));
    return false;
}

static void usage(FILE *fp) {
    std::fprintf(fp,
                 "Usage: ds4-v100-turbomind-pack --index FILE --source GGUF --out-dir DIR --layer N [options]\n"
                 "\n"
                 "Options:\n"
                 "  --kind gate|up|down|all       Tensor family to pack. Default: all\n"
                 "  --expert-limit N              Pack first N experts for a bounded smoke. Default: all\n"
                 "  --gpu N                       CUDA device for packing. Default: 0\n"
                 "  --lib FILE                    libggml-turbomind.so path\n");
}

static uint32_t parse_u32(const char *s, const char *name) {
    char *end = nullptr;
    errno = 0;
    unsigned long v = std::strtoul(s, &end, 10);
    if (errno || !end || *end || v > UINT32_MAX) {
        std::fprintf(stderr, "ds4-v100-turbomind-pack: invalid %s: %s\n", name, s);
        std::exit(2);
    }
    return (uint32_t)v;
}

static void parse_args(int argc, char **argv, options *opt) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        auto need = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "ds4-v100-turbomind-pack: %s requires a value\n", name);
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
        } else if (!std::strcmp(a, "--layer")) {
            opt->layer = (int)parse_u32(need(a), a);
        } else if (!std::strcmp(a, "--kind")) {
            opt->kind = need(a);
        } else if (!std::strcmp(a, "--expert-limit")) {
            opt->expert_limit = parse_u32(need(a), a);
        } else if (!std::strcmp(a, "--gpu")) {
            opt->gpu = (int)parse_u32(need(a), a);
        } else if (!std::strcmp(a, "--lib")) {
            opt->lib_path = need(a);
        } else if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) {
            usage(stdout);
            std::exit(0);
        } else {
            std::fprintf(stderr, "ds4-v100-turbomind-pack: unknown option %s\n", a);
            usage(stderr);
            std::exit(2);
        }
    }
    if (!opt->index_path || !opt->source_path || !opt->out_dir || opt->layer < 0) {
        usage(stderr);
        std::exit(2);
    }
    if (opt->kind != "gate" && opt->kind != "up" &&
        opt->kind != "down" && opt->kind != "all") {
        die("--kind must be gate, up, down, or all");
    }
}

static bool parse_shape3(const char *s, shape3 *out) {
    if (!s || !out) return false;
    unsigned k = 0;
    unsigned n = 0;
    unsigned experts = 0;
    char tail = 0;
    if (std::sscanf(s, "[%ux%ux%u]%c", &k, &n, &experts, &tail) != 3) return false;
    out->k = k;
    out->n = n;
    out->experts = experts;
    return out->k && out->n && out->experts;
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
                     "ds4-v100-turbomind-pack: failed to open %s: %s\n",
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
        std::fprintf(stderr, "ds4-v100-turbomind-pack: missing TurboMind C ABI symbol\n");
        return false;
    }
    if (api->api_version() != GGML_TURBOMIND_API_VERSION) {
        std::fprintf(stderr,
                     "ds4-v100-turbomind-pack: TurboMind ABI mismatch got %d expected %d\n",
                     api->api_version(),
                     GGML_TURBOMIND_API_VERSION);
        return false;
    }
    return true;
}

static void read_exact(FILE *fp, uint64_t off, void *dst, size_t bytes, const char *label) {
    if (fseeko(fp, (off_t)off, SEEK_SET) != 0) die_errno("cannot seek source", label);
    if (std::fread(dst, 1, bytes, fp) != bytes) die_errno("cannot read source", label);
}

static uint64_t tell_or_die(FILE *fp, const char *label) {
    off_t off = ftello(fp);
    if (off < 0) die_errno("cannot tell", label);
    return (uint64_t)off;
}

static void write_exact(FILE *fp, const void *src, size_t bytes, const char *label) {
    if (bytes && std::fwrite(src, 1, bytes, fp) != bytes) die_errno("cannot write", label);
}

static void pack_tensor(const options &opt,
                        const ds4_pack_entry &entry,
                        const shape3 &shape,
                        FILE *source,
                        FILE *sidecar,
                        FILE *index,
                        const tm_api &api) {
    const uint64_t row_bytes = ds4_src_mxfp4_row_bytes(shape.k);
    const uint64_t expert_stride = row_bytes * shape.n;
    if (expert_stride == 0 || entry.byte_length / expert_stride < shape.experts) {
        die("MXFP4 expert stride does not match pack-index byte length");
    }
    const uint32_t experts_to_pack =
        opt.expert_limit && opt.expert_limit < shape.experts ? opt.expert_limit : shape.experts;

    size_t weight_bytes = 0;
    size_t scale_bytes = 0;
    if (api.packed_bytes(GGML_TM_DTYPE_MXFP4,
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
    std::vector<uint8_t> all_scales(scale_bytes * (size_t)experts_to_pack);

    const uint64_t weight_offset = tell_or_die(sidecar, "sidecar");
    int expected_k_pack = 0;
    for (uint32_t expert = 0; expert < experts_to_pack; expert++) {
        read_exact(source,
                   entry.source_offset + (uint64_t)expert * expert_stride,
                   host_src.data(),
                   host_src.size(),
                   entry.source_name);
        if (!cuda_ok(cudaMemcpy(d_src,
                                host_src.data(),
                                host_src.size(),
                                cudaMemcpyHostToDevice),
                     "source upload")) {
            std::exit(1);
        }
        int k_pack = 0;
        if (api.pack_weight(d_src,
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
        write_exact(sidecar, host_weight.data(), host_weight.size(), "sidecar weight");
        std::memcpy(all_scales.data() + (size_t)expert * scale_bytes,
                    host_scale.data(),
                    scale_bytes);
    }
    const uint64_t scale_offset = tell_or_die(sidecar, "sidecar");
    write_exact(sidecar, all_scales.data(), all_scales.size(), "sidecar scale");

    std::fprintf(index,
                 "%s\t%s\tmxfp4\t%s\tturbomind_mxfp4_grouped\t%d\t%d\t"
                 "turbomind_mxfp4_grouped_sm70\t%u\t%u\t%u\t%u\t%zu\t%zu\t%d\t%d\t%d\t"
                 "gpu%d.turbomind\t%" PRIu64 "\t%" PRIu64 "\t%s\t%" PRIu64 "\t%" PRIu64
                 "\t%s\t%d\n",
                 entry.semantic_tensor_id,
                 entry.source_name,
                 entry.source_shape,
                 entry.owning_gpu,
                 entry.layer_id,
                 shape.n,
                 shape.k,
                 experts_to_pack,
                 shape.experts,
                 weight_bytes,
                 scale_bytes,
                 expected_k_pack,
                 (int)shape.k * 32,
                 (int)shape.n,
                 entry.owning_gpu,
                 weight_offset,
                 scale_offset,
                 entry.shard_file,
                 entry.shard_offset,
                 entry.byte_length,
                 "pending",
                 GGML_TURBOMIND_API_VERSION);

    std::fprintf(stderr,
                 "packed %s experts=%u/%u N=%u K=%u weight=%zu scale=%zu k_pack=0x%x\n",
                 entry.semantic_tensor_id,
                 experts_to_pack,
                 shape.experts,
                 shape.n,
                 shape.k,
                 weight_bytes,
                 scale_bytes,
                 expected_k_pack);

    (void)cudaFree(d_scale);
    (void)cudaFree(d_weight);
    (void)cudaFree(d_src);
}

static bool want_kind(const std::string &requested, const char *kind) {
    return requested == "all" || requested == kind;
}

int main(int argc, char **argv) {
    options opt;
    parse_args(argc, argv, &opt);
    mkdir_if_needed(opt.out_dir);

    if (!cuda_ok(cudaSetDevice(opt.gpu), "set device")) return 1;
    tm_api api;
    if (!load_api(opt.lib_path, &api)) return 1;
    if (api.init(opt.gpu) != 0) die("ggml_turbomind_init failed");

    char err[512] = {0};
    ds4_pack *pack = nullptr;
    if (ds4_pack_open(&pack, opt.index_path, err, sizeof(err))) {
        std::fprintf(stderr, "ds4-v100-turbomind-pack: %s\n", err);
        return 1;
    }

    FILE *source = std::fopen(opt.source_path, "rb");
    if (!source) die_errno("cannot open source", opt.source_path);

    std::string sidecar_name = "gpu0.turbomind";
    std::string index_name = path_join(opt.out_dir, "turbomind-pack-index.tsv");
    std::string sidecar_path;
    FILE *sidecar = nullptr;
    FILE *index = std::fopen(index_name.c_str(), "wb");
    if (!index) die_errno("cannot create index", index_name.c_str());
    std::fprintf(index,
                 "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\t"
                 "runtime_layout\towning_gpu\tlayer_id\tkernel_family\t"
                 "n\tk\texperts_packed\texperts_total\tweight_bytes_per_expert\t"
                 "scale_bytes_per_expert\tk_pack\tweight_stride\tscale_stride\t"
                 "sidecar_file\tweight_offset\tscale_offset\tsource_shard_file\t"
                 "source_shard_offset\tsource_byte_length\tsource_checksum\t"
                 "tm_abi_version\n");

    const char *kinds[] = {"gate", "up", "down"};
    const char *suffixes[] = {
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    };

    uint32_t packed = 0;
    for (uint32_t i = 0; i < 3; i++) {
        if (!want_kind(opt.kind, kinds[i])) continue;
        char semantic[128];
        std::snprintf(semantic, sizeof(semantic), "blk.%d.%s", opt.layer, suffixes[i]);
        ds4_pack_entry entry;
        if (ds4_pack_lookup(pack, semantic, &entry)) {
            std::fprintf(stderr, "ds4-v100-turbomind-pack: missing %s\n", semantic);
            return 1;
        }
        if (std::strcmp(entry.source_dtype, "mxfp4") != 0) {
            std::fprintf(stderr, "ds4-v100-turbomind-pack: %s is not mxfp4\n", semantic);
            return 1;
        }
        shape3 shape;
        if (!parse_shape3(entry.source_shape, &shape)) {
            std::fprintf(stderr,
                         "ds4-v100-turbomind-pack: cannot parse shape %s for %s\n",
                         entry.source_shape,
                         semantic);
            return 1;
        }
        if (!sidecar) {
            char base[64];
            std::snprintf(base, sizeof(base), "gpu%d.turbomind", entry.owning_gpu);
            sidecar_name = base;
            sidecar_path = path_join(opt.out_dir, sidecar_name.c_str());
            sidecar = std::fopen(sidecar_path.c_str(), "wb");
            if (!sidecar) die_errno("cannot create sidecar", sidecar_path.c_str());
        } else {
            char expect[64];
            std::snprintf(expect, sizeof(expect), "gpu%d.turbomind", entry.owning_gpu);
            if (sidecar_name != expect) die("selected tensors span multiple GPUs");
        }
        pack_tensor(opt, entry, shape, source, sidecar, index, api);
        packed++;
    }

    if (!packed) die("no tensors selected");
    if (sidecar && std::fclose(sidecar) != 0) die_errno("cannot close sidecar", sidecar_path.c_str());
    if (std::fclose(index) != 0) die_errno("cannot close index", index_name.c_str());
    if (std::fclose(source) != 0) die_errno("cannot close source", opt.source_path);
    ds4_pack_close(pack);
    api.shutdown();
    dlclose(api.handle);

    std::fprintf(stderr, "wrote %s\n", index_name.c_str());
    if (!sidecar_path.empty()) std::fprintf(stderr, "wrote %s\n", sidecar_path.c_str());
    return 0;
}
