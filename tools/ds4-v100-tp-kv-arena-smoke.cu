#include <cuda_runtime.h>

#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define KiB (1024ULL)
#define MiB (1024ULL * KiB)
#define GiB (1024ULL * MiB)

enum {
    DS4_N_LAYER             = 43,
    DS4_N_EMBD              = 4096,
    DS4_N_VOCAB             = 129280,
    DS4_N_HEAD_DIM          = 512,
    DS4_N_INDEXER_HEAD_DIM  = 128,
    DS4_N_SWA               = 128,
    DS4_N_GPU               = 8,
    DS4_N_TP                = 8,
    DS4_MAX_ARENAS_PER_GPU  = 6,
};

typedef enum {
    KV_F16,
    KV_F8_E4M3_B128,
    KV_Q8_0,
} kv_dtype;

typedef struct {
    uint64_t ctx;
    uint32_t slots;
    uint32_t gpus;
    uint64_t weight_total_bytes;
    double reserve_gib;
    double scratch_gib;
    kv_dtype kv;
    bool touch;
    bool alloc_weights;
    const char *pack_dir;
} options;

typedef struct {
    uint64_t weights;
    uint64_t kv;
    uint64_t kv_ideal;
    uint64_t kv_row_shard_overhead;
    uint64_t comp_state;
    uint64_t scratch;
    uint64_t collectives;
    uint64_t globals;
    uint64_t reserve;
} gpu_plan;

typedef struct {
    const char *name;
    uint64_t bytes;
    unsigned char *ptr;
} arena;

typedef struct {
    uint32_t dev;
    uint64_t total;
    uint64_t free_before;
    uint64_t free_after;
    arena arenas[DS4_MAX_ARENAS_PER_GPU];
} device_alloc;

__global__ static void touch_arena_kernel(unsigned char *ptr, uint64_t bytes) {
    const uint64_t stride = 2ULL * MiB;
    const uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    const uint64_t off = idx * stride;
    if (off < bytes) ptr[off] = (unsigned char)(idx & 0xffu);
    if (idx == 0 && bytes > 0) ptr[bytes - 1] = 0xa5u;
}

static void die(const char *msg) {
    fprintf(stderr, "ds4-v100-tp-kv-arena-smoke: %s\n", msg);
    exit(1);
}

static void cuda_die(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4-v100-tp-kv-arena-smoke: %s: %s\n",
                what, cudaGetErrorString(err));
        exit(1);
    }
}

static uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a) die("integer overflow");
    return a * b;
}

static uint64_t ceil_div_u64(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

static uint64_t bytes_blocks(uint64_t elems, uint64_t block_elems, uint64_t block_bytes) {
    return checked_mul(ceil_div_u64(elems, block_elems), block_bytes);
}

static uint64_t bytes_f16(uint64_t elems) { return checked_mul(elems, 2); }
static uint64_t bytes_f32(uint64_t elems) { return checked_mul(elems, 4); }
static uint64_t bytes_f8_e4m3_b128(uint64_t elems) { return bytes_blocks(elems, 128, 129); }
static uint64_t bytes_q8_0(uint64_t elems) { return bytes_blocks(elems, 32, 34); }

static uint64_t from_gib(double gib) {
    return (uint64_t)(gib * (double)GiB);
}

static double as_gib(uint64_t bytes) {
    return (double)bytes / (double)GiB;
}

static uint64_t parse_u64_arg(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    const unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(2);
    }
    return (uint64_t)v;
}

static double parse_double_arg(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    const double v = strtod(s, &end);
    if (errno || !end || *end) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(2);
    }
    return v;
}

static const char *kv_name(kv_dtype kv) {
    switch (kv) {
    case KV_F16: return "f16";
    case KV_F8_E4M3_B128: return "f8_e4m3_b128";
    case KV_Q8_0: return "q8_0";
    default: return "unknown";
    }
}

static kv_dtype parse_kv(const char *s) {
    if (!strcmp(s, "f16")) return KV_F16;
    if (!strcmp(s, "f8") || !strcmp(s, "f8_e4m3_b128")) return KV_F8_E4M3_B128;
    if (!strcmp(s, "q8") || !strcmp(s, "q8_0")) return KV_Q8_0;
    die("--kv-dtype must be f16, f8, or q8_0");
    return KV_F8_E4M3_B128;
}

static uint64_t kv_values_bytes(uint64_t values, kv_dtype kv) {
    switch (kv) {
    case KV_F16: return bytes_f16(values);
    case KV_F8_E4M3_B128: return bytes_f8_e4m3_b128(values);
    case KV_Q8_0: return bytes_q8_0(values);
    default: return bytes_f16(values);
    }
}

static int layer_ratio(uint32_t il) {
    if (il < 2) return 0;
    return (il % 2) == 0 ? 4 : 128;
}

static uint64_t layer_attn_kv_bytes(uint32_t il, uint64_t ctx, kv_dtype kv) {
    const int ratio = layer_ratio(il);
    const uint64_t rows = (uint64_t)DS4_N_SWA + (ratio ? ctx / (uint64_t)ratio : 0);
    return kv_values_bytes(checked_mul(rows, DS4_N_HEAD_DIM), kv);
}

static uint64_t layer_indexer_kv_bytes(uint32_t il, uint64_t ctx, kv_dtype kv) {
    if (layer_ratio(il) != 4) return 0;
    return kv_values_bytes(checked_mul(ctx / 4u, DS4_N_INDEXER_HEAD_DIM), kv);
}

static uint64_t shard_bytes(uint64_t bytes);

static uint64_t layer_attn_kv_physical_bytes_per_gpu(uint32_t il, uint64_t ctx,
                                                     kv_dtype kv) {
    const int ratio = layer_ratio(il);
    const uint64_t rows = (uint64_t)DS4_N_SWA + (ratio ? ctx / (uint64_t)ratio : 0);
    const uint64_t row_bytes = shard_bytes(kv_values_bytes(DS4_N_HEAD_DIM, kv));
    return checked_mul(rows, row_bytes);
}

static uint64_t layer_indexer_kv_physical_bytes_per_gpu(uint32_t il, uint64_t ctx,
                                                        kv_dtype kv) {
    if (layer_ratio(il) != 4) return 0;
    const uint64_t rows = ctx / 4u;
    const uint64_t row_bytes = shard_bytes(kv_values_bytes(DS4_N_INDEXER_HEAD_DIM, kv));
    return checked_mul(rows, row_bytes);
}

static uint64_t layer_comp_state_bytes(uint32_t il, uint64_t ctx) {
    const int ratio = layer_ratio(il);
    if (!ratio) return 0;

    const uint64_t comp_rows = ctx / (uint64_t)ratio;
    const uint64_t attn = checked_mul(comp_rows, DS4_N_HEAD_DIM);
    if (ratio == 4) {
        const uint64_t indexer = checked_mul(ctx / 4u, DS4_N_INDEXER_HEAD_DIM);
        return bytes_f32((attn + indexer) / 8u);
    }
    return bytes_f32(attn / 8u);
}

static uint64_t shard_bytes(uint64_t bytes) {
    return ceil_div_u64(bytes, DS4_N_TP);
}

static uint64_t persistent_kv_aggregate_bytes(uint64_t ctx, uint32_t slots, kv_dtype kv) {
    uint64_t total = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        total += layer_attn_kv_bytes(il, ctx, kv);
        total += layer_indexer_kv_bytes(il, ctx, kv);
    }
    return checked_mul(total, slots);
}

static uint64_t persistent_kv_physical_bytes_per_gpu(uint64_t ctx, uint32_t slots,
                                                     kv_dtype kv) {
    uint64_t total = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        total += layer_attn_kv_physical_bytes_per_gpu(il, ctx, kv);
        total += layer_indexer_kv_physical_bytes_per_gpu(il, ctx, kv);
    }
    return checked_mul(total, slots);
}

static uint64_t comp_state_aggregate_bytes(uint64_t ctx, uint32_t slots) {
    uint64_t total = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        total += layer_comp_state_bytes(il, ctx);
    }
    return checked_mul(total, slots);
}

static uint64_t global_bytes_per_gpu(void) {
    const uint64_t embed = bytes_f16(checked_mul(DS4_N_EMBD, DS4_N_VOCAB));
    const uint64_t output = bytes_f16(checked_mul(DS4_N_EMBD, DS4_N_VOCAB));
    const uint64_t output_control =
        bytes_f32(checked_mul((uint64_t)DS4_N_EMBD * 4u, 4u)) +
        bytes_f32(4u) +
        bytes_f32(1u) +
        bytes_f32(DS4_N_EMBD);
    return shard_bytes(embed) + shard_bytes(output) + output_control;
}

static uint64_t collective_workspace_bytes(uint32_t slots) {
    return checked_mul(2u * (uint64_t)slots, DS4_N_EMBD * 2ull);
}

static uint64_t gpu_no_reserve(const gpu_plan *p) {
    return p->weights + p->kv + p->comp_state + p->scratch +
           p->collectives + p->globals;
}

static uint64_t gpu_total(const gpu_plan *p) {
    return gpu_no_reserve(p) + p->reserve;
}

static gpu_plan plan_gpu(const options *opt) {
    gpu_plan p = {0};
    p.weights = opt->alloc_weights ? shard_bytes(opt->weight_total_bytes) : 0;
    p.kv_ideal = shard_bytes(persistent_kv_aggregate_bytes(opt->ctx, opt->slots, opt->kv));
    p.kv = persistent_kv_physical_bytes_per_gpu(opt->ctx, opt->slots, opt->kv);
    p.kv_row_shard_overhead = p.kv > p.kv_ideal ? p.kv - p.kv_ideal : 0;
    p.comp_state = shard_bytes(comp_state_aggregate_bytes(opt->ctx, opt->slots));
    p.scratch = from_gib(opt->scratch_gib);
    p.collectives = collective_workspace_bytes(opt->slots);
    p.globals = global_bytes_per_gpu();
    p.reserve = from_gib(opt->reserve_gib);
    return p;
}

static bool is_gpu_weight_file(const char *name) {
    return !strncmp(name, "gpu", 3) && strstr(name, ".weights") != NULL;
}

static uint64_t weight_total_from_pack_dir(const char *path) {
    DIR *dir = opendir(path);
    if (!dir) {
        fprintf(stderr, "failed to open --pack-dir %s: %s\n", path, strerror(errno));
        exit(2);
    }

    uint64_t total = 0;
    uint32_t matched = 0;
    struct dirent *ent = NULL;
    while ((ent = readdir(dir)) != NULL) {
        if (!is_gpu_weight_file(ent->d_name)) continue;
        char full[4096];
        const int n = snprintf(full, sizeof(full), "%s/%s", path, ent->d_name);
        if (n < 0 || (size_t)n >= sizeof(full)) die("--pack-dir path too long");

        struct stat st;
        if (stat(full, &st) != 0) {
            fprintf(stderr, "failed to stat %s: %s\n", full, strerror(errno));
            closedir(dir);
            exit(2);
        }
        if (st.st_size <= 0) continue;
        total += (uint64_t)st.st_size;
        matched++;
    }
    closedir(dir);

    if (matched == 0) die("--pack-dir did not contain gpu*.weights files");
    return total;
}

static void set_arena(arena *a, const char *name, uint64_t bytes) {
    a->name = name;
    a->bytes = bytes;
    a->ptr = NULL;
}

static void free_device_alloc(device_alloc *da) {
    cudaSetDevice((int)da->dev);
    for (uint32_t i = 0; i < DS4_MAX_ARENAS_PER_GPU; i++) {
        if (da->arenas[i].ptr) {
            cudaFree(da->arenas[i].ptr);
            da->arenas[i].ptr = NULL;
        }
    }
}

static void touch_arena(arena *a) {
    if (a->bytes == 0) return;
    const uint64_t stride = 2ULL * MiB;
    const uint64_t touches = ceil_div_u64(a->bytes, stride);
    const uint32_t threads = 256;
    const uint32_t blocks = (uint32_t)ceil_div_u64(touches, threads);
    touch_arena_kernel<<<blocks, threads>>>(a->ptr, a->bytes);
    cuda_die(cudaGetLastError(), "touch launch");
}

static void print_plan(const options *opt, const gpu_plan *p) {
    printf("DS4 V100 TP/EP KV arena allocation smoke\n");
    printf("topology: PP=1 TP=8 EP=8 requested_gpus=%u\n", opt->gpus);
    printf("configured: slots=%u ctx=%" PRIu64 " kv=%s reserve=%.2f GiB scratch=%.2f GiB touch=%s weights=%s\n",
           opt->slots, opt->ctx, kv_name(opt->kv), opt->reserve_gib,
           opt->scratch_gib, opt->touch ? "on" : "off",
           opt->alloc_weights ? "allocated" : "skipped");
    printf("weights_total=%.2f GiB weight_per_gpu=%.2f GiB%s\n",
           as_gib(opt->weight_total_bytes), as_gib(p->weights),
           opt->pack_dir ? " (from --pack-dir)" : " (static/default)");
    printf("\n## Per-GPU budget\n");
    printf("| Component | Bytes | GiB |\n");
    printf("|---|---:|---:|\n");
    printf("| weights | %" PRIu64 " | %.3f |\n", p->weights, as_gib(p->weights));
    printf("| persistent_kv_physical_row_sharded | %" PRIu64 " | %.3f |\n", p->kv, as_gib(p->kv));
    printf("| persistent_kv_ideal_aggregate_shard | %" PRIu64 " | %.3f |\n", p->kv_ideal, as_gib(p->kv_ideal));
    printf("| persistent_kv_row_shard_overhead | %" PRIu64 " | %.3f |\n",
           p->kv_row_shard_overhead, as_gib(p->kv_row_shard_overhead));
    printf("| comp_state | %" PRIu64 " | %.3f |\n", p->comp_state, as_gib(p->comp_state));
    printf("| scratch | %" PRIu64 " | %.3f |\n", p->scratch, as_gib(p->scratch));
    printf("| collectives | %" PRIu64 " | %.3f |\n", p->collectives, as_gib(p->collectives));
    printf("| globals | %" PRIu64 " | %.3f |\n", p->globals, as_gib(p->globals));
    printf("| no_reserve_total | %" PRIu64 " | %.3f |\n", gpu_no_reserve(p), as_gib(gpu_no_reserve(p)));
    printf("| reserve_required | %" PRIu64 " | %.3f |\n", p->reserve, as_gib(p->reserve));
    printf("| total_budget | %" PRIu64 " | %.3f |\n", gpu_total(p), as_gib(gpu_total(p)));
}

static bool allocate_all(const options *opt, const gpu_plan *p, device_alloc *devs) {
    bool ok = true;
    for (uint32_t d = 0; d < opt->gpus; d++) {
        device_alloc *da = &devs[d];
        da->dev = d;
        set_arena(&da->arenas[0], "weights", p->weights);
        set_arena(&da->arenas[1], "persistent_kv", p->kv);
        set_arena(&da->arenas[2], "comp_state", p->comp_state);
        set_arena(&da->arenas[3], "scratch", p->scratch);
        set_arena(&da->arenas[4], "collectives", p->collectives);
        set_arena(&da->arenas[5], "globals", p->globals);

        cudaError_t err = cudaSetDevice((int)d);
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu%u set-device failed: %s\n", d, cudaGetErrorString(err));
            ok = false;
            break;
        }
        size_t free_b = 0;
        size_t total_b = 0;
        err = cudaMemGetInfo(&free_b, &total_b);
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu%u mem-get-info failed: %s\n", d, cudaGetErrorString(err));
            ok = false;
            break;
        }
        da->free_before = (uint64_t)free_b;
        da->total = (uint64_t)total_b;

        for (uint32_t i = 0; i < DS4_MAX_ARENAS_PER_GPU; i++) {
            arena *a = &da->arenas[i];
            if (a->bytes == 0) continue;
            err = cudaMalloc((void **)&a->ptr, (size_t)a->bytes);
            if (err != cudaSuccess) {
                fprintf(stderr, "gpu%u cudaMalloc(%s %.3f GiB) failed: %s\n",
                        d, a->name, as_gib(a->bytes), cudaGetErrorString(err));
                ok = false;
                break;
            }
            if (opt->touch) touch_arena(a);
        }
        if (!ok) break;

        if (opt->touch) cuda_die(cudaDeviceSynchronize(), "touch synchronize");
        err = cudaMemGetInfo(&free_b, &total_b);
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu%u post mem-get-info failed: %s\n", d, cudaGetErrorString(err));
            ok = false;
            break;
        }
        da->free_after = (uint64_t)free_b;
        if (da->free_after < p->reserve) {
            fprintf(stderr, "gpu%u free_after %.3f GiB below reserve %.3f GiB\n",
                    d, as_gib(da->free_after), as_gib(p->reserve));
            ok = false;
            break;
        }
    }
    return ok;
}

static void print_allocs(const options *opt, const gpu_plan *p, const device_alloc *devs) {
    printf("\n## Allocation result\n");
    printf("| GPU | Total | Free before | Planned alloc | Free after | Reserve required | Status |\n");
    printf("|---:|---:|---:|---:|---:|---:|---|\n");
    for (uint32_t d = 0; d < opt->gpus; d++) {
        const device_alloc *da = &devs[d];
        const bool populated = da->total != 0;
        const bool passed = populated && da->free_after >= p->reserve;
        printf("| gpu%u | %.3f GiB | %.3f GiB | %.3f GiB | %.3f GiB | %.3f GiB | %s |\n",
               d, as_gib(da->total), as_gib(da->free_before),
               as_gib(gpu_no_reserve(p)), as_gib(da->free_after),
               as_gib(p->reserve), passed ? "PASS" : "FAIL");
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-v100-tp-kv-arena-smoke [options]\n"
        "\n"
        "CUDA allocation smoke for the DS4 V100 TP8/EP8 production KV arena.\n"
        "This tool intentionally exposes no PP/layer-split topology modes.\n"
        "\n"
        "Options:\n"
        "  --ctx N                    Context tokens. Default: 262144\n"
        "  --slots N                  Configured slots. Default: 32\n"
        "  --gpus N                   GPUs to allocate. Default: 8\n"
        "  --kv-dtype f16|f8|q8_0      KV planning dtype. Default: f8\n"
        "  --weight-total-bytes N      Total resident weight bytes\n"
        "  --weight-total-gib F        Total resident weight GiB. Default: 145.45\n"
        "  --pack-dir PATH             Sum gpu*.weights files and use as weight total\n"
        "  --reserve-gib F             Required free memory after allocation. Default: 2.0\n"
        "  --scratch-gib F             Scratch per GPU. Default: 1.5\n"
        "  --no-touch                  Do not touch allocated pages\n"
        "  --no-weight-alloc           Skip dummy weight allocation\n");
}

int main(int argc, char **argv) {
    options opt = {
        .ctx = 262144,
        .slots = 32,
        .gpus = DS4_N_GPU,
        .weight_total_bytes = (uint64_t)(145.45 * (double)GiB),
        .reserve_gib = 2.0,
        .scratch_gib = 1.5,
        .kv = KV_F8_E4M3_B128,
        .touch = true,
        .alloc_weights = true,
        .pack_dir = NULL,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(arg, "--ctx") && i + 1 < argc) {
            opt.ctx = parse_u64_arg(argv[++i], "--ctx");
        } else if (!strcmp(arg, "--slots") && i + 1 < argc) {
            opt.slots = (uint32_t)parse_u64_arg(argv[++i], "--slots");
        } else if (!strcmp(arg, "--gpus") && i + 1 < argc) {
            opt.gpus = (uint32_t)parse_u64_arg(argv[++i], "--gpus");
        } else if (!strcmp(arg, "--kv-dtype") && i + 1 < argc) {
            opt.kv = parse_kv(argv[++i]);
        } else if (!strcmp(arg, "--weight-total-bytes") && i + 1 < argc) {
            opt.weight_total_bytes = parse_u64_arg(argv[++i], "--weight-total-bytes");
        } else if (!strcmp(arg, "--weight-total-gib") && i + 1 < argc) {
            opt.weight_total_bytes = from_gib(parse_double_arg(argv[++i], "--weight-total-gib"));
        } else if (!strcmp(arg, "--pack-dir") && i + 1 < argc) {
            opt.pack_dir = argv[++i];
        } else if (!strcmp(arg, "--reserve-gib") && i + 1 < argc) {
            opt.reserve_gib = parse_double_arg(argv[++i], "--reserve-gib");
        } else if (!strcmp(arg, "--scratch-gib") && i + 1 < argc) {
            opt.scratch_gib = parse_double_arg(argv[++i], "--scratch-gib");
        } else if (!strcmp(arg, "--no-touch")) {
            opt.touch = false;
        } else if (!strcmp(arg, "--no-weight-alloc")) {
            opt.alloc_weights = false;
        } else if (!strcmp(arg, "--topology") || !strcmp(arg, "--pp")) {
            die("PP/layer-split options are intentionally unsupported");
        } else {
            usage(stderr);
            return 2;
        }
    }

    if (opt.slots == 0) die("--slots must be positive");
    if (opt.ctx == 0) die("--ctx must be positive");
    if (opt.gpus == 0 || opt.gpus > DS4_N_GPU) die("--gpus must be between 1 and 8");
    if (opt.reserve_gib < 0.0 || opt.scratch_gib < 0.0) die("reserve/scratch must be non-negative");
    if (opt.pack_dir) opt.weight_total_bytes = weight_total_from_pack_dir(opt.pack_dir);

    int visible = 0;
    cuda_die(cudaGetDeviceCount(&visible), "get device count");
    if (visible < (int)opt.gpus) {
        fprintf(stderr, "requested %u GPUs but only %d visible\n", opt.gpus, visible);
        return 1;
    }

    const gpu_plan p = plan_gpu(&opt);
    print_plan(&opt, &p);

    device_alloc devs[DS4_N_GPU] = {};
    const bool ok = allocate_all(&opt, &p, devs);
    print_allocs(&opt, &p, devs);

    for (int d = (int)opt.gpus - 1; d >= 0; d--) {
        free_device_alloc(&devs[d]);
    }

    printf("\nverdict: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
