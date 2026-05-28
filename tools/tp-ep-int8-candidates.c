#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

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
    CONTRACT_COLS = 23,
    MAX_GPUS = 16,
};

typedef struct {
    const char *contract_path;
    const char *out_tsv_path;
    const char *report_path;
    uint64_t slots;
    uint64_t qk;
} options;

typedef struct {
    const char *name;
    uint64_t rows;
    uint64_t source_bytes;
    uint64_t int8_data_bytes;
    uint64_t int8_scale_bytes;
    uint64_t int8_total_bytes;
} family_summary;

typedef struct {
    const char *family;
    const char *dtype;
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t rows;
    uint64_t source_bytes;
    uint64_t int8_total_bytes;
} shape_summary;

typedef struct {
    family_summary families[8];
    size_t n_families;
    shape_summary shapes[64];
    size_t n_shapes;
    uint64_t rows;
    uint64_t source_bytes;
    uint64_t int8_data_bytes;
    uint64_t int8_scale_bytes;
    uint64_t int8_total_bytes;
    uint64_t gpu_source_bytes[MAX_GPUS];
    uint64_t gpu_int8_total_bytes[MAX_GPUS];
} summary;

static void die(const char *msg) {
    fprintf(stderr, "ds4-v100-tp-ep-int8-candidates: %s\n", msg);
    exit(1);
}

static void die_errno(const char *what, const char *path) {
    fprintf(stderr, "ds4-v100-tp-ep-int8-candidates: %s %s: %s\n",
            what, path, strerror(errno));
    exit(1);
}

static uint64_t parse_u64(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    const unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(2);
    }
    return (uint64_t)v;
}

static int parse_i32_field(const char *s) {
    if (!s || !*s) return -1;
    char *end = NULL;
    errno = 0;
    const long v = strtol(s, &end, 10);
    if (errno || !end || *end) return -1;
    if (v < -2147483647L - 1L || v > 2147483647L) return -1;
    return (int)v;
}

static uint64_t parse_u64_field(const char *s) {
    if (!s || !*s) return 0;
    char *end = NULL;
    errno = 0;
    const unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) return 0;
    return (uint64_t)v;
}

static bool parse_shape2(const char *shape, uint64_t *cols, uint64_t *rows) {
    if (!shape || shape[0] != '[') return false;
    char *end = NULL;
    errno = 0;
    const unsigned long long c = strtoull(shape + 1, &end, 10);
    if (errno || !end || end == shape + 1 || (*end != 'x' && *end != 'X')) return false;
    errno = 0;
    const unsigned long long r = strtoull(end + 1, &end, 10);
    if (errno || !end || *end != ']') return false;
    if (end[1] != '\0' || c == 0 || r == 0) return false;
    *cols = (uint64_t)c;
    *rows = (uint64_t)r;
    return true;
}

static uint64_t ceil_div_u64(uint64_t a, uint64_t b) {
    return (a + b - 1u) / b;
}

static uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a && b > UINT64_MAX / a) die("integer overflow");
    return a * b;
}

static uint64_t f8_e4m3_b128_row_bytes(uint64_t cols) {
    return checked_mul(ceil_div_u64(cols, 128), 129);
}

static bool ends_with(const char *s, const char *suffix) {
    const size_t ns = strlen(s);
    const size_t nx = strlen(suffix);
    return ns >= nx && memcmp(s + ns - nx, suffix, nx) == 0;
}

static const char *candidate_family(const char *source_name, const char *source_dtype) {
    (void)source_dtype;
    if (ends_with(source_name, ".attn_compress_kv.weight") ||
        ends_with(source_name, ".attn_compress_gate.weight")) {
        return "attn_compressor_bf16";
    }
    if (ends_with(source_name, ".indexer.compress_kv.weight") ||
        ends_with(source_name, ".indexer.compress_gate.weight")) {
        return "indexer_compressor_bf16";
    }
    if (ends_with(source_name, ".indexer.proj.weight")) {
        return "indexer_proj_tiny";
    }
    if (ends_with(source_name, ".indexer.attn_q_b.weight")) {
        return "indexer_q_f8";
    }
    return NULL;
}

static const char *dtype_key(const char *dtype) {
    if (!strcmp(dtype, "bf16")) return "bf16";
    if (!strcmp(dtype, "f8_e4m3_b128")) return "f8_e4m3_b128";
    if (!strcmp(dtype, "f32")) return "f32";
    return "other";
}

static const char *decision_note(const char *family,
                                 const char *dtype,
                                 uint64_t n,
                                 uint64_t k,
                                 uint64_t int8_total,
                                 uint64_t source_bytes) {
    if (n < 16) {
        return "tiny_N; prefer fusion or keep current path before standalone INT8 GEMM";
    }
    if (strcmp(dtype, "f8_e4m3_b128") == 0) {
        return int8_total > source_bytes
            ? "compute-only candidate; INT8+scale is larger than F8 block-128"
            : "compute candidate; benchmark against existing F8 LUT path";
    }
    if (strcmp(family, "attn_compressor_bf16") == 0 && k == 4096 && n >= 128) {
        return "primary candidate; tc-grid INT8 v13_rf/v12_ms3 shape family";
    }
    if (strcmp(family, "attn_compressor_bf16") == 0 && k == 4096 && n >= 64) {
        return "secondary candidate; benchmark N64 INT8 and BF16 cuBLAS";
    }
    if (strcmp(family, "indexer_compressor_bf16") == 0 && k == 4096 && n >= 32) {
        return "candidate if fused with indexer state path; standalone GEMM may underfill";
    }
    return "candidate requires custom shape benchmark";
}

static const char *kernel_hint(const char *family,
                               const char *dtype,
                               uint64_t m,
                               uint64_t n,
                               uint64_t k) {
    (void)m;
    if (n < 16) return "fused_tiny_n";
    if (strcmp(dtype, "f8_e4m3_b128") == 0) return "compare_f8_lut_vs_int8_v13";
    if (k == 4096 && n >= 128) return "tc_grid_int8_v13_rf_v5_or_v12_ms3";
    if (k == 4096 && n >= 64) return "tc_grid_int8_v12s_n64";
    if (strcmp(family, "indexer_compressor_bf16") == 0) return "fused_indexer_int8";
    return "custom_int8_workbench";
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

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-v100-tp-ep-int8-candidates --contract PATH [options]\n"
            "\n"
            "Options:\n"
            "  --slots N       Active/configured slots for M estimate. Default: 32\n"
            "  --qk N          INT8 scale block size. Default: 32\n"
            "  --out-tsv PATH  Write per-candidate TSV\n"
            "  --report PATH   Write markdown summary\n");
}

static void require_contract_header(char *line) {
    static const char expected[] =
        "record_type\ttensor_id\tsource_name\tlayer_id\tfamily\tsource_dtype\t"
        "source_shape\truntime_layout\towning_gpu\ttp_rank\tep_rank\t"
        "split_axis\tshard_index\tshard_count\texpert_first\texpert_count\t"
        "kv_ratio\tkv_rows_per_slot\tbytes_estimate\tsource_pack_file\t"
        "source_shard_offset\tsource_byte_length\tkernel_family";
    if (strcmp(rstrip_newline(line), expected) != 0) {
        die("contract header does not match TP/EP pack-contract schema");
    }
}

static void parse_args(int argc, char **argv, options *opt) {
    *opt = (options){
        .contract_path = NULL,
        .out_tsv_path = NULL,
        .report_path = NULL,
        .slots = 32,
        .qk = 32,
    };
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--contract") && v) {
            opt->contract_path = v;
            i++;
        } else if (!strcmp(a, "--slots") && v) {
            opt->slots = parse_u64(v, a);
            i++;
        } else if (!strcmp(a, "--qk") && v) {
            opt->qk = parse_u64(v, a);
            i++;
        } else if (!strcmp(a, "--out-tsv") && v) {
            opt->out_tsv_path = v;
            i++;
        } else if (!strcmp(a, "--report") && v) {
            opt->report_path = v;
            i++;
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            exit(0);
        } else {
            usage(stderr);
            exit(2);
        }
    }
    if (!opt->contract_path) die("--contract is required");
    if (opt->slots == 0) die("--slots must be positive");
    if (opt->qk == 0) die("--qk must be positive");
}

static family_summary *summary_family(summary *s, const char *name) {
    for (size_t i = 0; i < s->n_families; i++) {
        if (!strcmp(s->families[i].name, name)) return &s->families[i];
    }
    if (s->n_families >= sizeof(s->families) / sizeof(s->families[0])) {
        die("too many candidate families");
    }
    family_summary *f = &s->families[s->n_families++];
    memset(f, 0, sizeof(*f));
    f->name = name;
    return f;
}

static shape_summary *summary_shape(summary *s,
                                    const char *family,
                                    const char *dtype,
                                    uint64_t m,
                                    uint64_t n,
                                    uint64_t k) {
    for (size_t i = 0; i < s->n_shapes; i++) {
        shape_summary *sh = &s->shapes[i];
        if (!strcmp(sh->family, family) && !strcmp(sh->dtype, dtype) &&
            sh->m == m && sh->n == n && sh->k == k) {
            return sh;
        }
    }
    if (s->n_shapes >= sizeof(s->shapes) / sizeof(s->shapes[0])) {
        die("too many candidate shapes");
    }
    shape_summary *sh = &s->shapes[s->n_shapes++];
    memset(sh, 0, sizeof(*sh));
    sh->family = family;
    sh->dtype = dtype;
    sh->m = m;
    sh->n = n;
    sh->k = k;
    return sh;
}

static void add_summary(summary *s,
                        const char *family,
                        const char *dtype,
                        int gpu,
                        uint64_t m,
                        uint64_t n,
                        uint64_t k,
                        uint64_t source_bytes,
                        uint64_t data_bytes,
                        uint64_t scale_bytes,
                        uint64_t total_bytes) {
    s->rows++;
    s->source_bytes += source_bytes;
    s->int8_data_bytes += data_bytes;
    s->int8_scale_bytes += scale_bytes;
    s->int8_total_bytes += total_bytes;
    if (gpu >= 0 && gpu < MAX_GPUS) {
        s->gpu_source_bytes[gpu] += source_bytes;
        s->gpu_int8_total_bytes[gpu] += total_bytes;
    }

    family_summary *f = summary_family(s, family);
    f->rows++;
    f->source_bytes += source_bytes;
    f->int8_data_bytes += data_bytes;
    f->int8_scale_bytes += scale_bytes;
    f->int8_total_bytes += total_bytes;

    shape_summary *sh = summary_shape(s, family, dtype, m, n, k);
    sh->rows++;
    sh->source_bytes += source_bytes;
    sh->int8_total_bytes += total_bytes;
}

static double as_mib(uint64_t bytes) {
    return (double)bytes / (double)MiB;
}

static double as_gib(uint64_t bytes) {
    return (double)bytes / (double)GiB;
}

static void write_report(FILE *fp, const options *opt, const summary *s) {
    fprintf(fp, "# DS4 TP/EP INT8 Candidate Audit\n\n");
    fprintf(fp, "- Contract: `%s`\n", opt->contract_path);
    fprintf(fp, "- Slots/M estimate: `%" PRIu64 "`\n", opt->slots);
    fprintf(fp, "- INT8 scale block: `%" PRIu64 "`\n\n", opt->qk);

    fprintf(fp, "## Topline\n\n");
    fprintf(fp, "| Rows | Source bytes | INT8+scale bytes | Delta | Source GiB | INT8 GiB |\n");
    fprintf(fp, "|---:|---:|---:|---:|---:|---:|\n");
    fprintf(fp, "| %" PRIu64 " | %" PRIu64 " | %" PRIu64 " | %" PRId64 " | %.3f | %.3f |\n\n",
            s->rows, s->source_bytes, s->int8_total_bytes,
            (int64_t)(s->int8_total_bytes - s->source_bytes),
            as_gib(s->source_bytes), as_gib(s->int8_total_bytes));

    fprintf(fp, "## By Family\n\n");
    fprintf(fp, "| Family | Rows | Source MiB | INT8 data MiB | Scale MiB | INT8 total MiB | Delta MiB |\n");
    fprintf(fp, "|---|---:|---:|---:|---:|---:|---:|\n");
    for (size_t i = 0; i < s->n_families; i++) {
        const family_summary *f = &s->families[i];
        fprintf(fp, "| %s | %" PRIu64 " | %.3f | %.3f | %.3f | %.3f | %.3f |\n",
                f->name, f->rows, as_mib(f->source_bytes),
                as_mib(f->int8_data_bytes), as_mib(f->int8_scale_bytes),
                as_mib(f->int8_total_bytes),
                as_mib(f->int8_total_bytes) - as_mib(f->source_bytes));
    }

    fprintf(fp, "\n## By Shape\n\n");
    fprintf(fp, "| Family | DType | M | N | K | Rows | Source MiB | INT8 total MiB |\n");
    fprintf(fp, "|---|---|---:|---:|---:|---:|---:|---:|\n");
    for (size_t i = 0; i < s->n_shapes; i++) {
        const shape_summary *sh = &s->shapes[i];
        fprintf(fp, "| %s | %s | %" PRIu64 " | %" PRIu64 " | %" PRIu64
                    " | %" PRIu64 " | %.3f | %.3f |\n",
                sh->family, sh->dtype, sh->m, sh->n, sh->k, sh->rows,
                as_mib(sh->source_bytes), as_mib(sh->int8_total_bytes));
    }

    fprintf(fp, "\n## By GPU\n\n");
    fprintf(fp, "| GPU | Source MiB | INT8 total MiB | Delta MiB |\n");
    fprintf(fp, "|---:|---:|---:|---:|\n");
    for (int gpu = 0; gpu < MAX_GPUS; gpu++) {
        if (!s->gpu_source_bytes[gpu] && !s->gpu_int8_total_bytes[gpu]) continue;
        fprintf(fp, "| %d | %.3f | %.3f | %.3f |\n",
                gpu, as_mib(s->gpu_source_bytes[gpu]),
                as_mib(s->gpu_int8_total_bytes[gpu]),
                as_mib(s->gpu_int8_total_bytes[gpu]) -
                    as_mib(s->gpu_source_bytes[gpu]));
    }
}

static int process_contract(const options *opt, FILE *tsv, summary *sum) {
    FILE *fp = fopen(opt->contract_path, "r");
    if (!fp) die_errno("cannot open contract", opt->contract_path);

    char *line = NULL;
    size_t cap = 0;
    ssize_t nread = getline(&line, &cap, fp);
    if (nread < 0) die("empty contract");
    require_contract_header(line);

    if (tsv) {
        fprintf(tsv,
                "tensor_id\tsource_name\tlayer_id\tcandidate_family\tsource_dtype\t"
                "source_shape\truntime_layout\towning_gpu\ttp_rank\tshard_index\t"
                "shard_count\tM\tN\tK\tsource_bytes\tint8_qk\tint8_data_bytes\t"
                "int8_scale_bytes\tint8_total_bytes\tbyte_delta\tkernel_hint\tdecision\n");
    }

    uint64_t line_no = 1;
    while ((nread = getline(&line, &cap, fp)) >= 0) {
        line_no++;
        rstrip_newline(line);
        if (!line[0]) continue;

        char *fields[CONTRACT_COLS];
        const int nf = split_tabs(line, fields, CONTRACT_COLS);
        if (nf != CONTRACT_COLS) {
            fprintf(stderr, "contract line %" PRIu64 " has %d columns, expected %d\n",
                    line_no, nf, CONTRACT_COLS);
            exit(1);
        }

        const char *record_type = fields[0];
        const char *tensor_id = fields[1];
        const char *source_name = fields[2];
        const int layer_id = parse_i32_field(fields[3]);
        const char *source_dtype = fields[5];
        const char *source_shape = fields[6];
        const char *runtime_layout = fields[7];
        const int owning_gpu = parse_i32_field(fields[8]);
        const int tp_rank = parse_i32_field(fields[9]);
        const int shard_index = parse_i32_field(fields[12]);
        const int shard_count = parse_i32_field(fields[13]);
        const uint64_t source_bytes = parse_u64_field(fields[18]);

        if (strcmp(record_type, "dense_tp") != 0) continue;
        const char *fam = candidate_family(source_name, source_dtype);
        if (!fam) continue;

        uint64_t k = 0;
        uint64_t total_rows = 0;
        if (!parse_shape2(source_shape, &k, &total_rows)) {
            fprintf(stderr, "cannot parse source_shape on line %" PRIu64 ": %s\n",
                    line_no, source_shape);
            exit(1);
        }

        uint64_t n = 0;
        if (!strcmp(source_dtype, "bf16")) {
            n = source_bytes / (2u * k);
        } else if (!strcmp(source_dtype, "f8_e4m3_b128")) {
            const uint64_t row_bytes = f8_e4m3_b128_row_bytes(k);
            n = row_bytes ? source_bytes / row_bytes : 0;
        } else {
            n = shard_count > 0 ? ceil_div_u64(total_rows, (uint64_t)shard_count) : total_rows;
        }
        if (n == 0) {
            fprintf(stderr, "cannot infer shard rows on line %" PRIu64 "\n", line_no);
            exit(1);
        }

        const uint64_t data_bytes = checked_mul(n, k);
        const uint64_t scale_bytes = checked_mul(checked_mul(n, ceil_div_u64(k, opt->qk)), 2u);
        const uint64_t total_bytes = data_bytes + scale_bytes;
        const int64_t delta = (int64_t)(total_bytes - source_bytes);
        const char *hint = kernel_hint(fam, source_dtype, opt->slots, n, k);
        const char *note = decision_note(fam, source_dtype, n, k, total_bytes, source_bytes);

        const char *stable_dtype = dtype_key(source_dtype);
        add_summary(sum, fam, stable_dtype, owning_gpu, opt->slots, n, k,
                    source_bytes, data_bytes, scale_bytes, total_bytes);

        if (tsv) {
            fprintf(tsv,
                    "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t"
                    "%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64
                    "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64
                    "\t%" PRId64 "\t%s\t%s\n",
                    tensor_id, source_name, layer_id, fam, source_dtype,
                    source_shape, runtime_layout, owning_gpu, tp_rank,
                    shard_index, shard_count, opt->slots, n, k, source_bytes,
                    opt->qk, data_bytes, scale_bytes, total_bytes, delta,
                    hint, note);
        }
    }

    free(line);
    if (ferror(fp)) die_errno("cannot read contract", opt->contract_path);
    if (fclose(fp) != 0) die_errno("cannot close contract", opt->contract_path);
    return 0;
}

int main(int argc, char **argv) {
    options opt;
    parse_args(argc, argv, &opt);

    FILE *tsv = NULL;
    if (opt.out_tsv_path) {
        tsv = fopen(opt.out_tsv_path, "w");
        if (!tsv) die_errno("cannot open output TSV", opt.out_tsv_path);
    }

    summary sum;
    memset(&sum, 0, sizeof(sum));
    process_contract(&opt, tsv, &sum);
    if (tsv && fclose(tsv) != 0) die_errno("cannot close output TSV", opt.out_tsv_path);

    FILE *report = stdout;
    if (opt.report_path) {
        report = fopen(opt.report_path, "w");
        if (!report) die_errno("cannot open report", opt.report_path);
    }
    write_report(report, &opt, &sum);
    if (opt.report_path && fclose(report) != 0) {
        die_errno("cannot close report", opt.report_path);
    }
    return 0;
}
