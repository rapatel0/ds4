#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    DS4_TP_EP_GPUS = 8,
    DS4_TP_EP_EXPERTS = 256,
    DS4_TP_EP_LOCAL_EXPERTS = DS4_TP_EP_EXPERTS / DS4_TP_EP_GPUS,
};

typedef struct {
    const char *contract_path;
    int layer;
    int samples;
} options;

typedef struct {
    uint64_t rows;
    uint64_t bytes;
    uint64_t dense_rows;
    uint64_t control_rows;
    uint64_t expert_rows;
    uint64_t kv_rows;
    uint64_t comp_rows;
    uint64_t expert_span_rows;
    uint64_t expert_first_mismatch;
    uint64_t expert_count_mismatch;
    uint64_t tp_rank_mismatch;
    uint64_t ep_rank_mismatch;
    uint64_t shard_count_mismatch;
    uint64_t replicated_kv_rows;
    uint64_t bad_kv_ratio_rows;
} gpu_summary;

typedef struct {
    uint64_t total_rows;
    uint64_t dense_rows;
    uint64_t control_rows;
    uint64_t expert_rows;
    uint64_t kv_rows;
    uint64_t comp_rows;
    uint64_t bad_rows;
    uint64_t sample_printed;
    gpu_summary gpu[DS4_TP_EP_GPUS];
} layer_summary;

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-v100-tp-ep-layer-descriptor-smoke "
            "--contract FILE [--layer N] [--samples N]\n");
}

static int parse_i32(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    const long v = strtol(s, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(2);
    }
    return (int)v;
}

static uint64_t parse_u64_field(const char *s) {
    if (!s || !*s) return 0;
    return (uint64_t)strtoull(s, NULL, 10);
}

static int parse_i32_field(const char *s) {
    if (!s || !*s) return -1;
    return (int)strtol(s, NULL, 10);
}

static void parse_args(int argc, char **argv, options *opt) {
    *opt = (options){
        .contract_path = NULL,
        .layer = 2,
        .samples = 8,
    };
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--contract") && v) {
            opt->contract_path = v;
            ++i;
        } else if (!strcmp(a, "--layer") && v) {
            opt->layer = parse_i32(v, a);
            ++i;
        } else if (!strcmp(a, "--samples") && v) {
            opt->samples = parse_i32(v, a);
            ++i;
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            exit(0);
        } else {
            usage(stderr);
            exit(2);
        }
    }
    if (!opt->contract_path) {
        usage(stderr);
        exit(2);
    }
    if (opt->layer < 0 || opt->samples < 0) {
        usage(stderr);
        exit(2);
    }
}

static int split_tsv(char *line, char **cols, int max_cols) {
    int n = 0;
    char *p = line;
    while (n < max_cols) {
        cols[n++] = p;
        char *tab = strchr(p, '\t');
        if (!tab) break;
        *tab = '\0';
        p = tab + 1;
    }
    if (n > 0) {
        char *last = cols[n - 1];
        const size_t len = strlen(last);
        if (len && last[len - 1] == '\n') last[len - 1] = '\0';
        const size_t len2 = strlen(last);
        if (len2 && last[len2 - 1] == '\r') last[len2 - 1] = '\0';
    }
    return n;
}

static bool streq(const char *a, const char *b) {
    return strcmp(a ? a : "", b ? b : "") == 0;
}

static void print_sample(const char **cols, int ncols) {
    if (ncols < 23) return;
    printf("sample\trecord_type\t%s\ttensor_id\t%s\tfamily\t%s\towning_gpu\t%s\t"
           "tp_rank\t%s\tep_rank\t%s\tsplit_axis\t%s\texpert_first\t%s\t"
           "expert_count\t%s\tkv_ratio\t%s\tbytes\t%s\tkernel\t%s\n",
           cols[0], cols[1], cols[4], cols[8], cols[9], cols[10], cols[11],
           cols[14], cols[15], cols[16], cols[18], cols[22]);
}

static void inspect_row(const options *opt, char **cols, int ncols, layer_summary *sum) {
    if (ncols < 23) {
        sum->bad_rows++;
        return;
    }

    const int layer = parse_i32_field(cols[3]);
    if (layer != opt->layer) return;

    const char *record = cols[0];
    const char *split_axis = cols[11];
    const int owning_gpu = parse_i32_field(cols[8]);
    const int tp_rank = parse_i32_field(cols[9]);
    const int ep_rank = parse_i32_field(cols[10]);
    const int shard_count = parse_i32_field(cols[13]);
    const int expert_first = parse_i32_field(cols[14]);
    const int expert_count = parse_i32_field(cols[15]);
    const int kv_ratio = parse_i32_field(cols[16]);
    const uint64_t bytes = parse_u64_field(cols[18]);

    if (owning_gpu < 0 || owning_gpu >= DS4_TP_EP_GPUS) {
        sum->bad_rows++;
        return;
    }

    gpu_summary *g = &sum->gpu[owning_gpu];
    sum->total_rows++;
    g->rows++;
    g->bytes += bytes;

    if (sum->sample_printed < (uint64_t)opt->samples) {
        print_sample((const char **)cols, ncols);
        sum->sample_printed++;
    }

    if (streq(record, "dense_tp")) {
        sum->dense_rows++;
        g->dense_rows++;
        if (tp_rank != owning_gpu) {
            g->tp_rank_mismatch++;
            sum->bad_rows++;
        }
        if (shard_count != DS4_TP_EP_GPUS) {
            g->shard_count_mismatch++;
            sum->bad_rows++;
        }
    } else if (streq(record, "replicated_control")) {
        sum->control_rows++;
        g->control_rows++;
        if (!streq(split_axis, "replicate")) {
            sum->bad_rows++;
        }
        if (tp_rank != owning_gpu) {
            g->tp_rank_mismatch++;
            sum->bad_rows++;
        }
        if (shard_count != DS4_TP_EP_GPUS) {
            g->shard_count_mismatch++;
            sum->bad_rows++;
        }
    } else if (streq(record, "ep_expert")) {
        sum->expert_rows++;
        g->expert_rows++;
        if (ep_rank != owning_gpu) {
            g->ep_rank_mismatch++;
            sum->bad_rows++;
        }
        if (expert_first != owning_gpu * DS4_TP_EP_LOCAL_EXPERTS) {
            g->expert_first_mismatch++;
            sum->bad_rows++;
        }
        if (expert_count != DS4_TP_EP_LOCAL_EXPERTS) {
            g->expert_count_mismatch++;
            sum->bad_rows++;
        }
        g->expert_span_rows++;
    } else if (streq(record, "kv_shard")) {
        sum->kv_rows++;
        g->kv_rows++;
        if (shard_count != DS4_TP_EP_GPUS || !streq(split_axis, "kv_dim")) {
            g->replicated_kv_rows++;
            sum->bad_rows++;
        }
        if (kv_ratio != 4) {
            g->bad_kv_ratio_rows++;
            sum->bad_rows++;
        }
    } else if (streq(record, "kv_comp_state")) {
        sum->comp_rows++;
        g->comp_rows++;
        if (shard_count != DS4_TP_EP_GPUS) {
            g->shard_count_mismatch++;
            sum->bad_rows++;
        }
        if (kv_ratio != 4) {
            g->bad_kv_ratio_rows++;
            sum->bad_rows++;
        }
    }
}

static bool all_gpus_have(uint64_t (*field)(const gpu_summary *), const layer_summary *sum) {
    for (int gpu = 0; gpu < DS4_TP_EP_GPUS; ++gpu) {
        if (field(&sum->gpu[gpu]) == 0) return false;
    }
    return true;
}

static uint64_t dense_field(const gpu_summary *g) { return g->dense_rows; }
static uint64_t control_field(const gpu_summary *g) { return g->control_rows; }
static uint64_t expert_field(const gpu_summary *g) { return g->expert_rows; }
static uint64_t kv_field(const gpu_summary *g) { return g->kv_rows; }
static uint64_t comp_field(const gpu_summary *g) { return g->comp_rows; }

int main(int argc, char **argv) {
    options opt;
    parse_args(argc, argv, &opt);

    FILE *fp = fopen(opt.contract_path, "r");
    if (!fp) {
        fprintf(stderr, "ds4-v100-tp-ep-layer-descriptor-smoke: open %s: %s\n",
                opt.contract_path, strerror(errno));
        return 1;
    }

    layer_summary sum;
    memset(&sum, 0, sizeof(sum));

    char line[16384];
    uint64_t line_no = 0;
    while (fgets(line, sizeof(line), fp)) {
        line_no++;
        if (line_no == 1 && !strncmp(line, "record_type\t", 12)) continue;
        char *cols[32] = {0};
        const int ncols = split_tsv(line, cols, 32);
        inspect_row(&opt, cols, ncols, &sum);
    }
    fclose(fp);

    printf("layer_descriptor_summary\tlayer\t%d\ttotal_rows\t%" PRIu64
           "\tdense_rows\t%" PRIu64 "\tcontrol_rows\t%" PRIu64
           "\texpert_rows\t%" PRIu64 "\tkv_rows\t%" PRIu64
           "\tcomp_rows\t%" PRIu64 "\tbad_rows\t%" PRIu64 "\n",
           opt.layer, sum.total_rows, sum.dense_rows, sum.control_rows,
           sum.expert_rows, sum.kv_rows, sum.comp_rows, sum.bad_rows);

    for (int gpu = 0; gpu < DS4_TP_EP_GPUS; ++gpu) {
        const gpu_summary *g = &sum.gpu[gpu];
        printf("gpu\t%d\trows\t%" PRIu64 "\tbytes\t%" PRIu64
               "\tdense_rows\t%" PRIu64 "\tcontrol_rows\t%" PRIu64
               "\texpert_rows\t%" PRIu64 "\tkv_rows\t%" PRIu64
               "\tcomp_rows\t%" PRIu64 "\texpert_first\t%d\texpert_count\t%d"
               "\tmismatches\t%" PRIu64 "\n",
               gpu, g->rows, g->bytes, g->dense_rows, g->control_rows,
               g->expert_rows, g->kv_rows, g->comp_rows,
               gpu * DS4_TP_EP_LOCAL_EXPERTS, DS4_TP_EP_LOCAL_EXPERTS,
               g->expert_first_mismatch + g->expert_count_mismatch +
                   g->tp_rank_mismatch + g->ep_rank_mismatch +
                   g->shard_count_mismatch + g->replicated_kv_rows +
                   g->bad_kv_ratio_rows);
    }

    bool ok = true;
    ok = ok && sum.total_rows > 0;
    ok = ok && all_gpus_have(dense_field, &sum);
    ok = ok && all_gpus_have(control_field, &sum);
    ok = ok && all_gpus_have(expert_field, &sum);
    ok = ok && all_gpus_have(kv_field, &sum);
    ok = ok && all_gpus_have(comp_field, &sum);
    ok = ok && sum.bad_rows == 0;

    printf("tp_ep_layer_descriptor_smoke\tlayer\t%d\tpp\t1\ttp\t8\tep\t8"
           "\tglobal_experts\t%d\tlocal_experts\t%d\tkv\tsharded\t%s\n",
           opt.layer, DS4_TP_EP_EXPERTS, DS4_TP_EP_LOCAL_EXPERTS,
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
