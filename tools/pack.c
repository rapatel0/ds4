#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

/*
 * DS4 V100 manifest packer.
 *
 * This tool consumes the manifest emitted by ds4-v100-plan and turns it into a
 * deterministic per-GPU shard layout.  By default it is a dry-run planner.  It
 * only copies tensor payloads when --emit-shards is explicitly passed.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define KiB (1024ULL)
#define MiB (1024ULL * KiB)
#define GiB (1024ULL * MiB)

enum {
    DS4_PACK_MAX_GPUS = 8,
    DS4_COPY_BUFFER = 8 * 1024 * 1024,
};

typedef struct {
    char *semantic_tensor_id;
    char *source_name;
    char *source_dtype;
    char *source_shape;
    char *runtime_layout;
    int owning_gpu;
    int layer_id;
    char *kernel_family;
    uint64_t source_offset;
    uint64_t byte_length;
    char *scale_offset;
    char *checksum;
    char *byte_offset_basis;
    uint64_t shard_offset;
} manifest_row;

typedef struct {
    manifest_row *v;
    uint64_t len;
    uint64_t cap;
} manifest_rows;

typedef struct {
    uint64_t tensors;
    uint64_t payload_bytes;
    uint64_t padded_bytes;
} gpu_bucket;

typedef struct {
    const char *manifest_path;
    const char *source_path;
    const char *out_dir;
    uint32_t gpus;
    uint64_t alignment;
    bool write_index;
    bool emit_shards;
} options;

static void die(const char *msg) {
    fprintf(stderr, "ds4-v100-pack: %s\n", msg);
    exit(1);
}

static void die_errno(const char *prefix, const char *path) {
    fprintf(stderr, "ds4-v100-pack: %s: %s: %s\n", prefix, path, strerror(errno));
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n);
    if (!q) die("out of memory");
    return q;
}

static char *xstrdup(const char *s) {
    char *p = strdup(s);
    if (!p) die("out of memory");
    return p;
}

static uint64_t parse_u64(const char *s, const char *field) {
    char *end = NULL;
    errno = 0;
    const unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "ds4-v100-pack: invalid %s: %s\n", field, s);
        exit(2);
    }
    return (uint64_t)v;
}

static int parse_i32(const char *s, const char *field) {
    char *end = NULL;
    errno = 0;
    const long v = strtol(s, &end, 10);
    if (errno || !end || *end || v < INT32_MIN || v > INT32_MAX) {
        fprintf(stderr, "ds4-v100-pack: invalid %s: %s\n", field, s);
        exit(2);
    }
    return (int)v;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    if (alignment == 0) return value;
    const uint64_t rem = value % alignment;
    if (rem == 0) return value;
    const uint64_t delta = alignment - rem;
    if (value > UINT64_MAX - delta) die("integer overflow while aligning shard offset");
    return value + delta;
}

static double as_gib(uint64_t bytes) {
    return (double)bytes / (double)GiB;
}

static void rows_push(manifest_rows *rows, manifest_row row) {
    if (rows->len == rows->cap) {
        uint64_t cap = rows->cap ? rows->cap * 2 : 1024;
        if (cap > (uint64_t)SIZE_MAX / sizeof(rows->v[0])) die("manifest is too large");
        rows->v = xrealloc(rows->v, (size_t)cap * sizeof(rows->v[0]));
        rows->cap = cap;
    }
    rows->v[rows->len++] = row;
}

static void free_rows(manifest_rows *rows) {
    for (uint64_t i = 0; i < rows->len; i++) {
        manifest_row *r = &rows->v[i];
        free(r->semantic_tensor_id);
        free(r->source_name);
        free(r->source_dtype);
        free(r->source_shape);
        free(r->runtime_layout);
        free(r->kernel_family);
        free(r->scale_offset);
        free(r->checksum);
        free(r->byte_offset_basis);
    }
    free(rows->v);
    memset(rows, 0, sizeof(*rows));
}

static char *rstrip_newline(char *s) {
    size_t n = strlen(s);
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = '\0';
    return s;
}

static int split_tabs(char *line, char **fields, int cap) {
    int n = 0;
    fields[n++] = line;
    for (char *p = line; *p; p++) {
        if (*p != '\t') continue;
        *p = '\0';
        if (n >= cap) return n + 1;
        fields[n++] = p + 1;
    }
    return n;
}

static void require_manifest_header(char *line) {
    static const char expected[] =
        "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\t"
        "runtime_layout\towning_gpu\tlayer_id\tkernel_family\t"
        "byte_offset\tbyte_length\tscale_offset\tchecksum\tbyte_offset_basis";
    if (strcmp(rstrip_newline(line), expected) != 0) {
        die("manifest header does not match Sprint 002 TSV schema");
    }
}

static void read_manifest(const char *path, manifest_rows *rows, uint32_t gpus) {
    FILE *fp = fopen(path, "r");
    if (!fp) die_errno("cannot open manifest", path);

    char *line = NULL;
    size_t cap = 0;
    ssize_t nread = getline(&line, &cap, fp);
    if (nread < 0) die("manifest is empty");
    require_manifest_header(line);

    uint64_t line_no = 1;
    while ((nread = getline(&line, &cap, fp)) >= 0) {
        line_no++;
        rstrip_newline(line);
        if (!line[0]) continue;

        char *fields[13];
        const int n = split_tabs(line, fields, 13);
        if (n != 13) {
            fprintf(stderr,
                    "ds4-v100-pack: manifest line %" PRIu64 " has %d fields, expected 13\n",
                    line_no, n);
            exit(1);
        }

        manifest_row row = {
            .semantic_tensor_id = xstrdup(fields[0]),
            .source_name = xstrdup(fields[1]),
            .source_dtype = xstrdup(fields[2]),
            .source_shape = xstrdup(fields[3]),
            .runtime_layout = xstrdup(fields[4]),
            .owning_gpu = parse_i32(fields[5], "owning_gpu"),
            .layer_id = parse_i32(fields[6], "layer_id"),
            .kernel_family = xstrdup(fields[7]),
            .source_offset = parse_u64(fields[8], "byte_offset"),
            .byte_length = parse_u64(fields[9], "byte_length"),
            .scale_offset = xstrdup(fields[10]),
            .checksum = xstrdup(fields[11]),
            .byte_offset_basis = xstrdup(fields[12]),
        };

        if (strcmp(row.byte_offset_basis, "absolute_gguf_file") != 0) {
            fprintf(stderr,
                    "ds4-v100-pack: tensor %s has unsupported byte_offset_basis=%s\n",
                    row.source_name,
                    row.byte_offset_basis);
            exit(1);
        }
        if (row.owning_gpu < 0 || row.owning_gpu >= (int)gpus) {
            fprintf(stderr,
                    "ds4-v100-pack: tensor %s has invalid owning_gpu=%d for --gpus %u\n",
                    row.source_name,
                    row.owning_gpu,
                    gpus);
            exit(1);
        }
        if (row.source_offset > UINT64_MAX - row.byte_length) {
            fprintf(stderr, "ds4-v100-pack: tensor %s source range overflows\n", row.source_name);
            exit(1);
        }
        rows_push(rows, row);
    }

    free(line);
    if (ferror(fp)) die_errno("cannot read manifest", path);
    if (fclose(fp) != 0) die_errno("cannot close manifest", path);
}

static uint64_t source_file_size(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) die_errno("cannot stat source GGUF", path);
    if (st.st_size < 0) die("source GGUF size is negative");
    return (uint64_t)st.st_size;
}

static void validate_source_ranges(const manifest_rows *rows, uint64_t source_size) {
    for (uint64_t i = 0; i < rows->len; i++) {
        const manifest_row *r = &rows->v[i];
        if (r->source_offset > source_size || r->byte_length > source_size - r->source_offset) {
            fprintf(stderr,
                    "ds4-v100-pack: tensor %s range [%" PRIu64 ", %" PRIu64 ") exceeds source size %" PRIu64 "\n",
                    r->source_name,
                    r->source_offset,
                    r->source_offset + r->byte_length,
                    source_size);
            exit(1);
        }
    }
}

static void assign_shard_offsets(manifest_rows *rows, gpu_bucket *gpu, uint32_t gpus, uint64_t alignment) {
    uint64_t cursor[DS4_PACK_MAX_GPUS] = {0};
    memset(gpu, 0, (size_t)gpus * sizeof(gpu[0]));
    for (uint64_t i = 0; i < rows->len; i++) {
        manifest_row *r = &rows->v[i];
        const uint32_t g = (uint32_t)r->owning_gpu;
        cursor[g] = align_up(cursor[g], alignment);
        r->shard_offset = cursor[g];
        if (r->byte_length > UINT64_MAX - cursor[g]) die("shard byte offset overflow");
        cursor[g] += r->byte_length;
        gpu[g].tensors++;
        gpu[g].payload_bytes += r->byte_length;
    }
    for (uint32_t g = 0; g < gpus; g++) gpu[g].padded_bytes = cursor[g];
}

static void print_summary(const manifest_rows *rows, const gpu_bucket *gpu, const options *opt) {
    printf("DS4 V100 packer\n");
    printf("manifest: %s\n", opt->manifest_path);
    printf("mode: %s\n", opt->emit_shards ? "emit-shards" : "dry-run");
    printf("gpus: %u\n", opt->gpus);
    printf("alignment: %" PRIu64 " bytes\n", opt->alignment);
    printf("tensors: %" PRIu64 "\n", rows->len);
    if (opt->source_path) printf("source: %s\n", opt->source_path);
    if (opt->out_dir) printf("out_dir: %s\n", opt->out_dir);

    printf("\nShard plan\n");
    printf("| GPU | Tensors | Payload | Padded shard size | Padding |\n");
    printf("|---:|---:|---:|---:|---:|\n");
    uint64_t total_payload = 0;
    uint64_t total_padded = 0;
    for (uint32_t g = 0; g < opt->gpus; g++) {
        const uint64_t padding = gpu[g].padded_bytes - gpu[g].payload_bytes;
        total_payload += gpu[g].payload_bytes;
        total_padded += gpu[g].padded_bytes;
        printf("| gpu%u | %" PRIu64 " | %.2f GiB | %.2f GiB | %.3f MiB |\n",
               g,
               gpu[g].tensors,
               as_gib(gpu[g].payload_bytes),
               as_gib(gpu[g].padded_bytes),
               (double)padding / (double)MiB);
    }
    printf("| total | %" PRIu64 " | %.2f GiB | %.2f GiB | %.3f MiB |\n",
           rows->len,
           as_gib(total_payload),
           as_gib(total_padded),
           (double)(total_padded - total_payload) / (double)MiB);
}

static void mkdir_if_needed(const char *path) {
    if (mkdir(path, 0775) == 0) return;
    if (errno == EEXIST) return;
    die_errno("cannot create output directory", path);
}

static void path_join(char *dst, size_t dstlen, const char *dir, const char *base) {
    const int n = snprintf(dst, dstlen, "%s/%s", dir, base);
    if (n < 0 || (size_t)n >= dstlen) die("output path is too long");
}

static void write_index(const manifest_rows *rows, const options *opt) {
    if (!opt->out_dir) die("--write-index requires --out-dir");
    mkdir_if_needed(opt->out_dir);

    char path[4096];
    path_join(path, sizeof(path), opt->out_dir, "pack-index.tsv");
    FILE *fp = fopen(path, "w");
    if (!fp) die_errno("cannot write pack index", path);

    fprintf(fp,
            "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\t"
            "runtime_layout\towning_gpu\tlayer_id\tkernel_family\t"
            "source_offset\tbyte_length\tshard_file\tshard_offset\t"
            "scale_offset\tchecksum\n");
    for (uint64_t i = 0; i < rows->len; i++) {
        const manifest_row *r = &rows->v[i];
        fprintf(fp,
                "%s\t%s\t%s\t%s\t%s\t%d\t%d\t%s\t%" PRIu64 "\t%" PRIu64
                "\tgpu%d.weights\t%" PRIu64 "\t%s\t%s\n",
                r->semantic_tensor_id,
                r->source_name,
                r->source_dtype,
                r->source_shape,
                r->runtime_layout,
                r->owning_gpu,
                r->layer_id,
                r->kernel_family,
                r->source_offset,
                r->byte_length,
                r->owning_gpu,
                r->shard_offset,
                r->scale_offset,
                r->checksum);
    }

    if (fclose(fp) != 0) die_errno("cannot close pack index", path);
    printf("\nWrote pack index: %s\n", path);
}

static void copy_exact(FILE *src, FILE *dst, uint8_t *buf, uint64_t n, const char *name) {
    uint64_t left = n;
    while (left) {
        const size_t chunk = left > DS4_COPY_BUFFER ? DS4_COPY_BUFFER : (size_t)left;
        if (fread(buf, 1, chunk, src) != chunk) {
            fprintf(stderr, "ds4-v100-pack: short read while copying %s\n", name);
            exit(1);
        }
        if (fwrite(buf, 1, chunk, dst) != chunk) {
            fprintf(stderr, "ds4-v100-pack: short write while copying %s\n", name);
            exit(1);
        }
        left -= chunk;
    }
}

static void emit_shards(const manifest_rows *rows, const gpu_bucket *gpu, const options *opt) {
    if (!opt->source_path) die("--emit-shards requires --source");
    if (!opt->out_dir) die("--emit-shards requires --out-dir");
    mkdir_if_needed(opt->out_dir);

    FILE *src = fopen(opt->source_path, "rb");
    if (!src) die_errno("cannot open source GGUF", opt->source_path);

    FILE *dst[DS4_PACK_MAX_GPUS] = {0};
    for (uint32_t g = 0; g < opt->gpus; g++) {
        char path[4096];
        char base[64];
        snprintf(base, sizeof(base), "gpu%u.weights", g);
        path_join(path, sizeof(path), opt->out_dir, base);
        dst[g] = fopen(path, "wb");
        if (!dst[g]) die_errno("cannot create shard", path);
    }

    uint8_t *buf = xmalloc(DS4_COPY_BUFFER);
    for (uint64_t i = 0; i < rows->len; i++) {
        const manifest_row *r = &rows->v[i];
        FILE *out = dst[r->owning_gpu];
        if (fseeko(src, (off_t)r->source_offset, SEEK_SET) != 0) {
            die_errno("cannot seek source GGUF", opt->source_path);
        }
        if (fseeko(out, (off_t)r->shard_offset, SEEK_SET) != 0) {
            die("cannot seek shard output");
        }
        copy_exact(src, out, buf, r->byte_length, r->source_name);
    }
    free(buf);

    for (uint32_t g = 0; g < opt->gpus; g++) {
        if (fflush(dst[g]) != 0) die("cannot flush shard");
        if (ftruncate(fileno(dst[g]), (off_t)gpu[g].padded_bytes) != 0) {
            die("cannot truncate shard to planned size");
        }
        if (fclose(dst[g]) != 0) die("cannot close shard");
    }
    if (fclose(src) != 0) die_errno("cannot close source GGUF", opt->source_path);
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-v100-pack --manifest FILE [options]\n"
        "\n"
        "Options:\n"
        "  --manifest FILE       Sprint 002 pack manifest TSV\n"
        "  --source FILE         Source GGUF; validates source ranges, required for --emit-shards\n"
        "  --out-dir DIR         Output directory for pack-index.tsv and optional shards\n"
        "  --gpus N              Number of GPUs. Default: 8\n"
        "  --align N             Shard tensor alignment in bytes. Default: 256\n"
        "  --write-index         Write pack-index.tsv under --out-dir\n"
        "  --emit-shards         Copy tensor payloads into gpuN.weights files\n"
        "\n"
        "Default behavior is dry-run only. --emit-shards is the only mode that copies model bytes.\n");
}

int main(int argc, char **argv) {
    options opt = {
        .gpus = 8,
        .alignment = 256,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(arg, "--manifest") && i + 1 < argc) {
            opt.manifest_path = argv[++i];
        } else if (!strcmp(arg, "--source") && i + 1 < argc) {
            opt.source_path = argv[++i];
        } else if (!strcmp(arg, "--out-dir") && i + 1 < argc) {
            opt.out_dir = argv[++i];
        } else if (!strcmp(arg, "--gpus") && i + 1 < argc) {
            opt.gpus = (uint32_t)parse_u64(argv[++i], "--gpus");
        } else if (!strcmp(arg, "--align") && i + 1 < argc) {
            opt.alignment = parse_u64(argv[++i], "--align");
        } else if (!strcmp(arg, "--write-index")) {
            opt.write_index = true;
        } else if (!strcmp(arg, "--emit-shards")) {
            opt.emit_shards = true;
        } else {
            usage(stderr);
            return 2;
        }
    }

    if (!opt.manifest_path) die("--manifest is required");
    if (opt.gpus == 0 || opt.gpus > DS4_PACK_MAX_GPUS) die("--gpus must be in 1..8");
    if (opt.alignment == 0) die("--align must be positive");
    if (opt.write_index && !opt.out_dir) die("--write-index requires --out-dir");
    if (opt.emit_shards) {
        if (!opt.source_path) die("--emit-shards requires --source");
        if (!opt.out_dir) die("--emit-shards requires --out-dir");
        opt.write_index = true;
    }

    manifest_rows rows = {0};
    read_manifest(opt.manifest_path, &rows, opt.gpus);
    if (opt.source_path) validate_source_ranges(&rows, source_file_size(opt.source_path));

    gpu_bucket gpu[DS4_PACK_MAX_GPUS] = {{0}};
    assign_shard_offsets(&rows, gpu, opt.gpus, opt.alignment);
    print_summary(&rows, gpu, &opt);

    if (opt.write_index) write_index(&rows, &opt);
    if (opt.emit_shards) {
        emit_shards(&rows, gpu, &opt);
        printf("Wrote %u GPU shard files under %s\n", opt.gpus, opt.out_dir);
    }

    free_rows(&rows);
    return 0;
}
