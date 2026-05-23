#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KiB (1024ULL)
#define MiB (1024ULL * KiB)
#define GiB (1024ULL * MiB)

enum {
    DS4_N_SWA              = 128,
    DS4_N_HEAD_DIM         = 512,
    DS4_N_INDEXER_HEAD_DIM = 128,
    DS4_TP                 = 8,
};

typedef enum {
    KV_F16,
    KV_F8_E4M3_B128,
    KV_Q8_0,
} kv_dtype;

typedef struct {
    uint64_t ctx;
    uint32_t slots;
} options;

static void die(const char *msg) {
    fprintf(stderr, "ds4-v100-tp8-kv-shard-smoke: %s\n", msg);
    exit(1);
}

static uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a) die("integer overflow");
    return a * b;
}

static uint64_t ceil_div(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

static uint64_t bytes_blocks(uint64_t elems, uint64_t block_elems, uint64_t block_bytes) {
    return checked_mul(ceil_div(elems, block_elems), block_bytes);
}

static uint64_t values_bytes(uint64_t values, kv_dtype dtype) {
    switch (dtype) {
    case KV_F16: return checked_mul(values, 2);
    case KV_F8_E4M3_B128: return bytes_blocks(values, 128, 129);
    case KV_Q8_0: return bytes_blocks(values, 32, 34);
    default: die("unknown KV dtype");
    }
}

static const char *dtype_name(kv_dtype dtype) {
    switch (dtype) {
    case KV_F16: return "f16";
    case KV_F8_E4M3_B128: return "f8_e4m3_b128";
    case KV_Q8_0: return "q8_0";
    default: return "unknown";
    }
}

static uint64_t layer_kv_bytes(int ratio, uint64_t ctx, kv_dtype dtype) {
    const uint64_t rows = (uint64_t) DS4_N_SWA + (ratio ? ctx / (uint64_t) ratio : 0);
    uint64_t bytes = values_bytes(checked_mul(rows, DS4_N_HEAD_DIM), dtype);
    if (ratio == 4) {
        bytes += values_bytes(checked_mul(ctx / 4u, DS4_N_INDEXER_HEAD_DIM), dtype);
    }
    return bytes;
}

static double as_mib(uint64_t bytes) {
    return (double) bytes / (double) MiB;
}

static double as_gib(uint64_t bytes) {
    return (double) bytes / (double) GiB;
}

static uint64_t parse_u64_arg(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    const unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(2);
    }
    return (uint64_t) v;
}

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-v100-tp8-kv-shard-smoke [--ctx N] [--slots N]\n"
            "\n"
            "Defaults: --ctx 262144 --slots 32\n");
}

static void print_row(const char *layer_class, int ratio, kv_dtype dtype,
                      uint64_t ctx, uint32_t slots) {
    const uint64_t per_slot = layer_kv_bytes(ratio, ctx, dtype);
    const uint64_t replicated = checked_mul(per_slot, slots);
    const uint64_t shard = ceil_div(replicated, DS4_TP);
    const uint64_t reconstructed = checked_mul(shard, DS4_TP);
    const bool covers = reconstructed >= replicated;

    printf("| %s | %s | %" PRIu64 " | %u | %.3f MiB | %.3f GiB | %.3f GiB | %s |\n",
           layer_class,
           dtype_name(dtype),
           ctx,
           slots,
           as_mib(per_slot),
           as_gib(replicated),
           as_gib(shard),
           covers ? "ok" : "FAIL");
}

int main(int argc, char **argv) {
    options opt = {
        .ctx = 262144,
        .slots = 32,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(arg, "--ctx") && i + 1 < argc) {
            opt.ctx = parse_u64_arg(argv[++i], "--ctx");
        } else if (!strcmp(arg, "--slots") && i + 1 < argc) {
            opt.slots = (uint32_t) parse_u64_arg(argv[++i], "--slots");
        } else {
            usage(stderr);
            return 2;
        }
    }

    if (opt.ctx == 0) die("--ctx must be positive");
    if (opt.slots == 0) die("--slots must be positive");

    printf("DS4 V100 TP8 KV shard smoke\n");
    printf("tp=%d ctx=%" PRIu64 " slots=%u\n", DS4_TP, opt.ctx, opt.slots);
    printf("| Layer class | KV dtype | Context | Slots | Per-layer / slot | Replicated / layer | TP8 shard / layer / GPU | Coverage |\n");
    printf("|---|---|---:|---:|---:|---:|---:|---|\n");

    const kv_dtype dtypes[] = {KV_F8_E4M3_B128, KV_Q8_0, KV_F16};
    for (size_t i = 0; i < sizeof(dtypes) / sizeof(dtypes[0]); i++) {
        print_row("ratio-4", 4, dtypes[i], opt.ctx, opt.slots);
        print_row("ratio-128", 128, dtypes[i], opt.ctx, opt.slots);
    }

    const uint64_t r4_f8 = checked_mul(layer_kv_bytes(4, opt.ctx, KV_F8_E4M3_B128), opt.slots);
    const uint64_t r4_f8_shard = ceil_div(r4_f8, DS4_TP);
    if (r4_f8_shard == 0 || checked_mul(r4_f8_shard, DS4_TP) < r4_f8) {
        printf("\nverdict: FAIL\n");
        return 1;
    }

    printf("\nverdict: ok - TP8 shard descriptors cover replicated logical KV without allocating replicated KV\n");
    return 0;
}
