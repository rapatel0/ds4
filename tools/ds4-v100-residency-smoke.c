#include "ds4_gpu.h"
#include "ds4_pack.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    PROVIDER_GGUF,
    PROVIDER_SHARD,
    CHUNK_BYTES = 8 * 1024 * 1024,
    SPOT_BYTES = 4096
};

typedef struct {
    const char *model_path;
    const char *index_path;
    const char *shard_dir;
    const char *report_path;
    int provider;
    int require_gpus;
    int reserve_mib;
    bool crosscheck;
} smoke_options;

typedef struct {
    ds4_pack *pack;
    FILE *report;
    int provider;
    int model_fd;
    const unsigned char *model_map;
    uint64_t model_size;
    int shard_fds[DS4_PACK_MAX_GPUS];
    ds4_gpu_arena *arenas[DS4_PACK_MAX_GPUS];
    unsigned char *chunk;
    unsigned char *expect;
    unsigned char *actual;
    uint64_t uploaded_tensors;
    uint64_t uploaded_bytes;
    uint64_t spot_checks;
} smoke_ctx;

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-v100-residency-smoke --model FILE --index FILE [options]\n"
        "\n"
        "Options:\n"
        "  --provider gguf|shard      Source bytes from GGUF offsets or gpuN.weights. Default: gguf\n"
        "  --shard-dir DIR            Directory containing gpuN.weights for provider=shard\n"
        "  --reserve-mib N            Reserve reported for the run. Default: 3072\n"
        "  --require-gpus N           Require at least N visible CUDA devices for real runs\n"
        "  --report FILE              Write the smoke report to FILE\n"
        "  --crosscheck               Compare one deterministic tensor between providers\n");
}

static const char *need_arg(int *i, int argc, char **argv, const char *arg) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-v100-residency-smoke: %s requires an argument\n", arg);
        exit(2);
    }
    return argv[++*i];
}

static int parse_int_arg(const char *s, const char *arg) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || !end || *end || v < 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-v100-residency-smoke: bad integer for %s: %s\n", arg, s);
        exit(2);
    }
    return (int)v;
}

static smoke_options parse_options(int argc, char **argv) {
    smoke_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.provider = PROVIDER_GGUF;
    opt.reserve_mib = 3072;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "--model")) {
            opt.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--index")) {
            opt.index_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--shard-dir")) {
            opt.shard_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--report")) {
            opt.report_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--reserve-mib")) {
            opt.reserve_mib = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--require-gpus")) {
            opt.require_gpus = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--provider")) {
            const char *p = need_arg(&i, argc, argv, arg);
            if (!strcmp(p, "gguf")) opt.provider = PROVIDER_GGUF;
            else if (!strcmp(p, "shard")) opt.provider = PROVIDER_SHARD;
            else {
                fprintf(stderr, "ds4-v100-residency-smoke: bad provider: %s\n", p);
                exit(2);
            }
        } else if (!strcmp(arg, "--crosscheck")) {
            opt.crosscheck = true;
        } else {
            fprintf(stderr, "ds4-v100-residency-smoke: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }
    if (!opt.model_path || !opt.index_path) {
        usage(stderr);
        exit(2);
    }
    if ((opt.provider == PROVIDER_SHARD || opt.crosscheck) &&
        (!opt.shard_dir || !opt.shard_dir[0])) {
        fprintf(stderr, "ds4-v100-residency-smoke: --shard-dir is required for shard reads\n");
        exit(2);
    }
    return opt;
}

static uint64_t file_size_or_die(int fd, const char *path) {
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "ds4-v100-residency-smoke: cannot stat %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (st.st_size < 0) {
        fprintf(stderr, "ds4-v100-residency-smoke: negative size: %s\n", path);
        exit(1);
    }
    return (uint64_t)st.st_size;
}

static int join_path(char *out, size_t outlen, const char *dir, const char *name) {
    size_t dlen = strlen(dir);
    while (dlen && dir[dlen - 1] == '/') dlen--;
    int n = snprintf(out, outlen, "%.*s/%s", (int)dlen, dir, name);
    return n < 0 || (size_t)n >= outlen;
}

static void open_model(smoke_ctx *ctx, const char *path) {
    ctx->model_fd = open(path, O_RDONLY);
    if (ctx->model_fd < 0) {
        fprintf(stderr, "ds4-v100-residency-smoke: cannot open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    ctx->model_size = file_size_or_die(ctx->model_fd, path);
    ctx->model_map = (const unsigned char *)mmap(NULL, (size_t)ctx->model_size,
                                                 PROT_READ, MAP_PRIVATE,
                                                 ctx->model_fd, 0);
    if (ctx->model_map == MAP_FAILED) {
        fprintf(stderr, "ds4-v100-residency-smoke: cannot mmap %s: %s\n", path, strerror(errno));
        exit(1);
    }
}

static int open_shard(smoke_ctx *ctx, const char *shard_dir, int gpu) {
    if (ctx->shard_fds[gpu] >= 0) return ctx->shard_fds[gpu];
    char name[32];
    char path[4096];
    snprintf(name, sizeof(name), "gpu%d.weights", gpu);
    if (join_path(path, sizeof(path), shard_dir, name)) {
        fprintf(stderr, "ds4-v100-residency-smoke: shard path too long\n");
        return -1;
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ds4-v100-residency-smoke: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    ctx->shard_fds[gpu] = fd;
    return fd;
}

static int read_shard_exact(smoke_ctx *ctx, const smoke_options *opt,
                            const ds4_pack_entry *e,
                            uint64_t rel,
                            void *dst,
                            uint64_t bytes) {
    int fd = open_shard(ctx, opt->shard_dir, e->owning_gpu);
    if (fd < 0) return 1;
    uint64_t off = e->shard_offset + rel;
    unsigned char *p = (unsigned char *)dst;
    while (bytes) {
        size_t n = bytes > (uint64_t)SSIZE_MAX ? (size_t)SSIZE_MAX : (size_t)bytes;
        ssize_t got = pread(fd, p, n, (off_t)off);
        if (got <= 0) return 1;
        p += got;
        off += (uint64_t)got;
        bytes -= (uint64_t)got;
    }
    return 0;
}

static int read_provider_exact(smoke_ctx *ctx, const smoke_options *opt,
                               const ds4_pack_entry *e,
                               int provider,
                               uint64_t rel,
                               void *dst,
                               uint64_t bytes) {
    if (provider == PROVIDER_GGUF) {
        if (e->source_offset + rel > ctx->model_size ||
            bytes > ctx->model_size - e->source_offset - rel) {
            return 1;
        }
        memcpy(dst, ctx->model_map + e->source_offset + rel, (size_t)bytes);
        return 0;
    }
    return read_shard_exact(ctx, opt, e, rel, dst, bytes);
}

typedef struct {
    smoke_ctx *ctx;
    const smoke_options *opt;
} upload_ud;

static int upload_entry(const ds4_pack_entry *e, void *ud_ptr) {
    upload_ud *ud = (upload_ud *)ud_ptr;
    smoke_ctx *ctx = ud->ctx;
    const smoke_options *opt = ud->opt;
    ds4_gpu_arena *arena = ctx->arenas[e->owning_gpu];
    if (!arena) return 1;

    uint64_t done = 0;
    while (done < e->byte_length) {
        uint64_t n = e->byte_length - done;
        if (n > CHUNK_BYTES) n = CHUNK_BYTES;
        const void *src = NULL;
        if (opt->provider == PROVIDER_GGUF) {
            if (e->source_offset + done > ctx->model_size ||
                n > ctx->model_size - e->source_offset - done) return 1;
            src = ctx->model_map + e->source_offset + done;
        } else {
            if (read_shard_exact(ctx, opt, e, done, ctx->chunk, n) != 0) return 1;
            src = ctx->chunk;
        }
        if (ds4_gpu_arena_upload(arena, e->shard_offset + done, src, n) != 0) return 1;
        done += n;
    }

    uint64_t spots[2] = {0, 0};
    int n_spots = 1;
    uint64_t spot_len = e->byte_length < SPOT_BYTES ? e->byte_length : SPOT_BYTES;
    if (e->byte_length > SPOT_BYTES) {
        spots[1] = e->byte_length - spot_len;
        n_spots = 2;
    }
    for (int i = 0; i < n_spots; i++) {
        if (read_provider_exact(ctx, opt, e, opt->provider, spots[i], ctx->expect, spot_len) != 0) return 1;
        if (ds4_gpu_arena_read(arena, e->shard_offset + spots[i], ctx->actual, spot_len) != 0) return 1;
        if (memcmp(ctx->expect, ctx->actual, (size_t)spot_len) != 0) return 1;
        ctx->spot_checks++;
    }

    ctx->uploaded_tensors++;
    ctx->uploaded_bytes += e->byte_length;
    return 0;
}

static int compare_entry_between_providers(const ds4_pack_entry *e, void *ud_ptr) {
    upload_ud *ud = (upload_ud *)ud_ptr;
    smoke_ctx *ctx = ud->ctx;
    const smoke_options *opt = ud->opt;
    if (e->byte_length > 64ull * 1024ull * 1024ull) return 0;
    uint64_t done = 0;
    while (done < e->byte_length) {
        uint64_t n = e->byte_length - done;
        if (n > CHUNK_BYTES / 2) n = CHUNK_BYTES / 2;
        if (read_provider_exact(ctx, opt, e, PROVIDER_GGUF, done, ctx->chunk, n) != 0) return 2;
        if (read_provider_exact(ctx, opt, e, PROVIDER_SHARD, done, ctx->expect, n) != 0) return 2;
        if (memcmp(ctx->chunk, ctx->expect, (size_t)n) != 0) return 2;
        done += n;
    }
    fprintf(ctx->report, "crosscheck\t%s\t%" PRIu64 "\tOK\n", e->semantic_tensor_id, e->byte_length);
    return 1;
}

static void cleanup(smoke_ctx *ctx) {
    for (int i = 0; i < DS4_PACK_MAX_GPUS; i++) {
        ds4_gpu_arena_close(ctx->arenas[i]);
        if (ctx->shard_fds[i] >= 0) close(ctx->shard_fds[i]);
    }
    if (ctx->model_map && ctx->model_map != MAP_FAILED) munmap((void *)ctx->model_map, (size_t)ctx->model_size);
    if (ctx->model_fd >= 0) close(ctx->model_fd);
    free(ctx->chunk);
    free(ctx->expect);
    free(ctx->actual);
    ds4_pack_close(ctx->pack);
}

int main(int argc, char **argv) {
    smoke_options opt = parse_options(argc, argv);
    smoke_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.model_fd = -1;
    ctx.provider = opt.provider;
    for (int i = 0; i < DS4_PACK_MAX_GPUS; i++) ctx.shard_fds[i] = -1;

    FILE *report = stdout;
    if (opt.report_path) {
        report = fopen(opt.report_path, "w");
        if (!report) {
            fprintf(stderr, "ds4-v100-residency-smoke: cannot open report %s: %s\n",
                    opt.report_path, strerror(errno));
            return 1;
        }
    }
    ctx.report = report;

    char err[512];
    if (ds4_pack_open(&ctx.pack, opt.index_path, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-v100-residency-smoke: %s\n", err);
        return 1;
    }
    int max_gpu = ds4_pack_max_gpu(ctx.pack);
    if (max_gpu < 0 || max_gpu >= DS4_PACK_MAX_GPUS) {
        fprintf(stderr, "ds4-v100-residency-smoke: bad max gpu %d\n", max_gpu);
        return 1;
    }
    if (opt.provider == PROVIDER_SHARD &&
        ds4_pack_validate_shards(ctx.pack, opt.shard_dir, report, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-v100-residency-smoke: %s\n", err);
        return 1;
    }

    open_model(&ctx, opt.model_path);
    int device_count = ds4_gpu_device_count();
    int needed = max_gpu + 1;
    fprintf(report, "provider\t%s\n", opt.provider == PROVIDER_GGUF ? "gguf" : "shard");
    fprintf(report, "pack_rows\t%" PRIu64 "\n", ds4_pack_count(ctx.pack));
    fprintf(report, "model_size\t%" PRIu64 "\n", ctx.model_size);
    fprintf(report, "visible_devices\t%d\n", device_count);
    fprintf(report, "required_devices\t%d\n", needed);
    fprintf(report, "reserve_mib\t%d\n", opt.reserve_mib);
    ds4_gpu_print_topology_report(report);
    if (opt.require_gpus > 0 && device_count < opt.require_gpus) {
        fprintf(stderr, "ds4-v100-residency-smoke: visible devices %d < required %d\n",
                device_count, opt.require_gpus);
        return 1;
    }
    if (device_count > 0 && device_count < needed) {
        fprintf(stderr, "ds4-v100-residency-smoke: pack needs %d devices, visible %d\n",
                needed, device_count);
        return 1;
    }

    ctx.chunk = (unsigned char *)malloc(CHUNK_BYTES);
    ctx.expect = (unsigned char *)malloc(SPOT_BYTES > CHUNK_BYTES / 2 ? SPOT_BYTES : CHUNK_BYTES / 2);
    ctx.actual = (unsigned char *)malloc(SPOT_BYTES);
    if (!ctx.chunk || !ctx.expect || !ctx.actual) {
        fprintf(stderr, "ds4-v100-residency-smoke: out of memory\n");
        return 1;
    }

    fprintf(report, "gpu\tlogical_payload_bytes\tarena_bytes\ttensors\n");
    for (int gpu = 0; gpu <= max_gpu; gpu++) {
        uint64_t arena_bytes = ds4_pack_arena_bytes(ctx.pack, gpu);
        if (!arena_bytes) continue;
        fprintf(report, "%d\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\n",
                gpu,
                ds4_pack_payload_bytes(ctx.pack, gpu),
                arena_bytes,
                ds4_pack_tensor_count(ctx.pack, gpu));
        if (ds4_gpu_arena_open(&ctx.arenas[gpu], gpu, arena_bytes) != 0) {
            fprintf(stderr, "ds4-v100-residency-smoke: arena open failed on gpu %d\n", gpu);
            cleanup(&ctx);
            return 1;
        }
    }

    upload_ud ud = {&ctx, &opt};
    if (ds4_pack_for_each(ctx.pack, upload_entry, &ud) != 0) {
        fprintf(stderr, "ds4-v100-residency-smoke: upload or spot-check failed\n");
        cleanup(&ctx);
        return 1;
    }
    fprintf(report, "uploaded_tensors\t%" PRIu64 "\n", ctx.uploaded_tensors);
    fprintf(report, "uploaded_bytes\t%" PRIu64 "\n", ctx.uploaded_bytes);
    fprintf(report, "spot_checks\t%" PRIu64 "\n", ctx.spot_checks);
    ds4_gpu_arena_print_memory_report(report, ctx.arenas, max_gpu + 1);
    if (device_count > 0) {
        uint64_t reserve_bytes = (uint64_t)opt.reserve_mib * 1024ull * 1024ull;
        for (int gpu = 0; gpu <= max_gpu; gpu++) {
            if (!ctx.arenas[gpu]) continue;
            uint64_t free_after = ds4_gpu_arena_free_after_upload_bytes(ctx.arenas[gpu]);
            if (free_after < reserve_bytes) {
                fprintf(stderr,
                        "ds4-v100-residency-smoke: gpu %d free bytes %" PRIu64
                        " below reserve %" PRIu64 "\n",
                        gpu, free_after, reserve_bytes);
                cleanup(&ctx);
                return 1;
            }
        }
    }

    if (opt.crosscheck) {
        int rc = ds4_pack_for_each(ctx.pack, compare_entry_between_providers, &ud);
        if (rc == 0) {
            fprintf(stderr, "ds4-v100-residency-smoke: no suitable crosscheck tensor found\n");
            cleanup(&ctx);
            return 1;
        }
        if (rc != 1) {
            fprintf(stderr, "ds4-v100-residency-smoke: provider crosscheck failed\n");
            cleanup(&ctx);
            return 1;
        }
    }

    fprintf(report, "result\tOK\n");
    if (opt.report_path) fclose(report);
    cleanup(&ctx);
    return 0;
}
