#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <ctype.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define KiB (1024ULL)
#define MiB (1024ULL * KiB)
#define GiB (1024ULL * MiB)

enum {
    DS4_N_GPU              = 8,
    DS4_N_TP               = 8,
    DS4_N_EP               = 8,
    DS4_N_LAYER            = 43,
    DS4_N_EXPERT           = 256,
    DS4_N_EXPERT_USED      = 6,
    DS4_N_SWA              = 128,
    DS4_N_HEAD_DIM         = 512,
    DS4_N_INDEXER_HEAD_DIM = 128,
};

typedef enum {
    KV_F16,
    KV_F8_E4M3_B128,
    KV_Q8_0,
} kv_dtype;

typedef struct {
    const char *pack_dir;
    const char *out_dir;
    uint64_t ctx;
    uint32_t slots;
    kv_dtype kv;
    double reserve_gib;
    double scratch_gib;
} options;

typedef struct {
    uint64_t dense_tp;
    uint64_t dense_f8_cacheable_packed;
    uint64_t dense_bf16_shadowable_packed;
    uint64_t dense_f16_cache;
    uint64_t dense_bf16_f16_shadow;
    uint64_t replicated_control;
    uint64_t ep_expert;
    uint64_t kv;
    uint64_t comp_state;
    uint64_t scratch;
    uint64_t reserve;
    uint64_t rows;
} gpu_summary;

typedef struct {
    FILE *contract;
    gpu_summary gpu[DS4_N_GPU];
    uint64_t pack_rows;
    uint64_t dense_rows;
    uint64_t dense_f8_cacheable_rows;
    uint64_t dense_bf16_shadowable_rows;
    uint64_t control_rows;
    uint64_t expert_rows;
    uint64_t kv_rows;
} emit_state;

static void die(const char *msg) {
    fprintf(stderr, "ds4-v100-tp-ep-pack-contract: %s\n", msg);
    exit(1);
}

static void die_errno(const char *what, const char *path) {
    fprintf(stderr, "ds4-v100-tp-ep-pack-contract: %s %s: %s\n",
            what, path, strerror(errno));
    exit(1);
}

static uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a) die("integer overflow");
    return a * b;
}

static uint64_t ceil_div_u64(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

static uint64_t bytes_blocks(uint64_t elems, uint64_t block_elems, uint64_t block_bytes) {
    return checked_mul((elems + block_elems - 1) / block_elems, block_bytes);
}

static uint64_t bytes_f16(uint64_t elems) { return checked_mul(elems, 2); }
static uint64_t bytes_f32(uint64_t elems) { return checked_mul(elems, 4); }
static uint64_t bytes_f8(uint64_t elems) { return bytes_blocks(elems, 128, 129); }
static uint64_t bytes_q8_0(uint64_t elems) { return bytes_blocks(elems, 32, 34); }

static uint64_t from_gib(double gib) {
    return (uint64_t)(gib * (double)GiB);
}

static double as_gib(uint64_t bytes) {
    return (double)bytes / (double)GiB;
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
    return (int)strtol(s, NULL, 10);
}

static uint64_t parse_u64_field(const char *s) {
    if (!s || !*s) return 0;
    return (uint64_t)strtoull(s, NULL, 10);
}

static bool parse_shape2_u64(const char *shape, uint64_t *cols, uint64_t *rows) {
    if (!shape) return false;
    const char *p = shape;
    while (isspace((unsigned char)*p)) p++;
    if (*p++ != '[') return false;
    while (isspace((unsigned char)*p)) p++;

    char *end = NULL;
    errno = 0;
    const unsigned long long c = strtoull(p, &end, 10);
    if (errno || end == p || c == 0) return false;
    p = end;
    while (isspace((unsigned char)*p)) p++;
    if (*p != 'x' && *p != 'X') return false;
    p++;
    while (isspace((unsigned char)*p)) p++;

    errno = 0;
    const unsigned long long r = strtoull(p, &end, 10);
    if (errno || end == p || r == 0) return false;
    p = end;
    while (isspace((unsigned char)*p)) p++;
    if (*p++ != ']') return false;
    while (isspace((unsigned char)*p)) p++;
    if (*p) return false;

    *cols = (uint64_t)c;
    *rows = (uint64_t)r;
    return true;
}

static double parse_double(const char *s, const char *name) {
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

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-v100-tp-ep-pack-contract --pack-dir DIR --out-dir DIR [options]\n"
            "\n"
            "Options:\n"
            "  --ctx N               Context tokens. Default: 262144\n"
            "  --slots N             Configured slots. Default: 32\n"
            "  --kv-dtype f16|f8|q8_0 KV cache dtype. Default: f8\n"
            "  --reserve-gib F       Reserve per GPU. Default: 2.0\n"
            "  --scratch-gib F       Scratch per GPU. Default: 1.5\n");
}

static void parse_args(int argc, char **argv, options *opt) {
    *opt = (options){
        .pack_dir = NULL,
        .out_dir = NULL,
        .ctx = 262144,
        .slots = 32,
        .kv = KV_F8_E4M3_B128,
        .reserve_gib = 2.0,
        .scratch_gib = 1.5,
    };
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--pack-dir") && v) {
            opt->pack_dir = v;
            i++;
        } else if (!strcmp(a, "--out-dir") && v) {
            opt->out_dir = v;
            i++;
        } else if (!strcmp(a, "--ctx") && v) {
            opt->ctx = parse_u64(v, a);
            i++;
        } else if (!strcmp(a, "--slots") && v) {
            opt->slots = (uint32_t)parse_u64(v, a);
            i++;
        } else if (!strcmp(a, "--kv-dtype") && v) {
            opt->kv = parse_kv(v);
            i++;
        } else if (!strcmp(a, "--reserve-gib") && v) {
            opt->reserve_gib = parse_double(v, a);
            i++;
        } else if (!strcmp(a, "--scratch-gib") && v) {
            opt->scratch_gib = parse_double(v, a);
            i++;
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            exit(0);
        } else {
            usage(stderr);
            exit(2);
        }
    }
    if (!opt->pack_dir || !opt->out_dir) {
        usage(stderr);
        exit(2);
    }
    if (opt->slots == 0) die("--slots must be positive");
}

static void join_path(char *dst, size_t dst_len, const char *a, const char *b) {
    const int n = snprintf(dst, dst_len, "%s/%s", a, b);
    if (n < 0 || (size_t)n >= dst_len) die("path too long");
}

static FILE *open_read_joined(const char *dir, const char *name) {
    char path[4096];
    join_path(path, sizeof(path), dir, name);
    FILE *fp = fopen(path, "r");
    if (!fp) die_errno("open", path);
    return fp;
}

static FILE *open_write_joined(const char *dir, const char *name) {
    char path[4096];
    join_path(path, sizeof(path), dir, name);
    FILE *fp = fopen(path, "w");
    if (!fp) die_errno("open", path);
    return fp;
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
        last[strcspn(last, "\r\n")] = '\0';
    }
    return n;
}

static bool contains(const char *s, const char *needle) {
    return s && strstr(s, needle) != NULL;
}

static bool is_expert_source_tensor(const char *semantic, const char *source) {
    return (contains(semantic, "_exps.weight") || contains(source, "_exps.weight") ||
            contains(semantic, "ffn_gate_up_exps.weight"));
}

static bool is_replicated_control(const char *dtype, const char *layout,
                                  const char *kernel_family) {
    if (!strcmp(dtype, "f32") || !strcmp(dtype, "i32")) return true;
    if (contains(layout, "control")) return true;
    if (contains(kernel_family, "control") || contains(kernel_family, "router")) return true;
    return false;
}

static const char *dense_split_axis(const char *semantic, const char *kernel_family) {
    if (contains(semantic, "token_embd") || contains(semantic, "output")) return "vocab";
    if (contains(kernel_family, "attention")) return "attention_projection";
    if (contains(kernel_family, "shared")) return "ffn_shared_projection";
    return "tensor_dim";
}

static int layer_ratio(int layer) {
    if (layer < 2) return 0;
    return (layer % 2) == 0 ? 4 : 128;
}

static uint64_t kv_values_bytes(uint64_t values, kv_dtype kv) {
    switch (kv) {
    case KV_F16: return bytes_f16(values);
    case KV_F8_E4M3_B128: return bytes_f8(values);
    case KV_Q8_0: return bytes_q8_0(values);
    default: return bytes_f16(values);
    }
}

static uint64_t layer_attn_kv_bytes(int layer, uint64_t ctx, kv_dtype kv) {
    const int ratio = layer_ratio(layer);
    const uint64_t rows = (uint64_t)DS4_N_SWA + (ratio ? ctx / (uint64_t)ratio : 0);
    return kv_values_bytes(checked_mul(rows, DS4_N_HEAD_DIM), kv);
}

static uint64_t layer_indexer_kv_bytes(int layer, uint64_t ctx, kv_dtype kv) {
    if (layer_ratio(layer) != 4) return 0;
    return kv_values_bytes(checked_mul(ctx / 4u, DS4_N_INDEXER_HEAD_DIM), kv);
}

static uint64_t layer_comp_state_bytes(int layer, uint64_t ctx) {
    const int ratio = layer_ratio(layer);
    if (!ratio) return 0;
    const uint64_t attn = checked_mul(ctx / (uint64_t)ratio, DS4_N_HEAD_DIM);
    if (ratio == 4) {
        const uint64_t indexer = checked_mul(ctx / 4u, DS4_N_INDEXER_HEAD_DIM);
        return bytes_f32((attn + indexer) / 8u);
    }
    return bytes_f32(attn / 8u);
}

static void emit_contract_header(FILE *fp) {
    fprintf(fp,
            "record_type\ttensor_id\tsource_name\tlayer_id\tfamily\tsource_dtype\t"
            "source_shape\truntime_layout\towning_gpu\ttp_rank\tep_rank\tsplit_axis\t"
            "shard_index\tshard_count\texpert_first\texpert_count\tkv_ratio\t"
            "kv_rows_per_slot\tbytes_estimate\tsource_pack_file\tsource_shard_offset\t"
            "source_byte_length\tkernel_family\n");
}

static void emit_row(emit_state *st,
                     const char *record_type,
                     const char *tensor_id,
                     const char *source_name,
                     int layer_id,
                     const char *family,
                     const char *source_dtype,
                     const char *source_shape,
                     const char *runtime_layout,
                     int owning_gpu,
                     int tp_rank,
                     int ep_rank,
                     const char *split_axis,
                     int shard_index,
                     int shard_count,
                     int expert_first,
                     int expert_count,
                     int kv_ratio,
                     uint64_t kv_rows_per_slot,
                     uint64_t bytes_estimate,
                     const char *source_pack_file,
                     uint64_t source_shard_offset,
                     uint64_t source_byte_length,
                     const char *kernel_family) {
    fprintf(st->contract,
            "%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t"
            "%" PRIu64 "\t%" PRIu64 "\t%s\t%" PRIu64 "\t%" PRIu64 "\t%s\n",
            record_type, tensor_id, source_name, layer_id, family, source_dtype,
            source_shape, runtime_layout, owning_gpu, tp_rank, ep_rank, split_axis,
            shard_index, shard_count, expert_first, expert_count, kv_ratio,
            kv_rows_per_slot, bytes_estimate, source_pack_file, source_shard_offset,
            source_byte_length, kernel_family);
    st->gpu[owning_gpu].rows++;
}

static void emit_pack_row(emit_state *st, char **c, int n) {
    if (n < 14) return;
    const char *semantic = c[0];
    const char *source = c[1];
    const char *dtype = c[2];
    const char *shape = c[3];
    const char *layout = c[4];
    const int layer = parse_i32_field(c[6]);
    const char *kernel = c[7];
    const uint64_t bytes = parse_u64_field(c[9]);
    const char *shard_file = c[10];
    const uint64_t shard_offset = parse_u64_field(c[11]);

    st->pack_rows++;
    if (is_expert_source_tensor(semantic, source)) return;

    if (is_replicated_control(dtype, layout, kernel)) {
        for (int gpu = 0; gpu < DS4_N_GPU; gpu++) {
            emit_row(st, "replicated_control", semantic, source, layer,
                     "control_or_router", dtype, shape, layout, gpu, gpu, -1,
                     "replicate", gpu, DS4_N_GPU, -1, 0, -1, 0, bytes,
                     shard_file, shard_offset, bytes, kernel);
            st->gpu[gpu].replicated_control += bytes;
            st->control_rows++;
        }
        return;
    }

    const uint64_t shard_bytes = ceil_div_u64(bytes, DS4_N_TP);
    uint64_t expanded_f16_shard_bytes = 0;
    uint64_t shape_cols = 0;
    uint64_t shape_rows = 0;
    if (parse_shape2_u64(shape, &shape_cols, &shape_rows)) {
        const uint64_t rows_per_gpu = ceil_div_u64(shape_rows, DS4_N_TP);
        expanded_f16_shard_bytes = checked_mul(checked_mul(shape_cols, rows_per_gpu), 2);
    }
    for (int gpu = 0; gpu < DS4_N_GPU; gpu++) {
        emit_row(st, "dense_tp", semantic, source, layer, "dense_or_embedding",
                 dtype, shape, layout, gpu, gpu, -1, dense_split_axis(semantic, kernel),
                 gpu, DS4_N_TP, -1, 0, -1, 0, shard_bytes, shard_file, shard_offset,
                 bytes, kernel);
        st->gpu[gpu].dense_tp += shard_bytes;
        if (expanded_f16_shard_bytes && !strcmp(dtype, "f8_e4m3_b128")) {
            st->gpu[gpu].dense_f8_cacheable_packed += shard_bytes;
            st->gpu[gpu].dense_f16_cache += expanded_f16_shard_bytes;
            st->dense_f8_cacheable_rows++;
        } else if (expanded_f16_shard_bytes && !strcmp(dtype, "bf16")) {
            st->gpu[gpu].dense_bf16_shadowable_packed += shard_bytes;
            st->gpu[gpu].dense_bf16_f16_shadow += expanded_f16_shard_bytes;
            st->dense_bf16_shadowable_rows++;
        }
        st->dense_rows++;
    }
}

static void process_pack_index(const options *opt, emit_state *st) {
    FILE *fp = open_read_joined(opt->pack_dir, "pack-index.tsv");
    char *line = NULL;
    size_t cap = 0;
    bool first = true;
    while (getline(&line, &cap, fp) >= 0) {
        if (first) {
            first = false;
            continue;
        }
        char *cols[32] = {0};
        const int n = split_tsv(line, cols, 32);
        emit_pack_row(st, cols, n);
    }
    free(line);
    fclose(fp);
}

static void emit_tm_row(emit_state *st, char **c, int n) {
    if (n < 24) return;
    const char *semantic = c[0];
    const char *source = c[1];
    const char *dtype = c[2];
    const char *shape = c[3];
    const char *layout = c[4];
    const int layer = parse_i32_field(c[6]);
    const char *kernel = c[7];
    const uint64_t experts_total = parse_u64_field(c[11]);
    const uint64_t weight_per = parse_u64_field(c[12]);
    const uint64_t scale_per = parse_u64_field(c[13]);
    const char *shard_file = c[17];
    const uint64_t weight_offset = parse_u64_field(c[18]);
    const uint64_t source_len = parse_u64_field(c[22]);
    const uint64_t experts_per_gpu = ceil_div_u64(experts_total, DS4_N_EP);

    for (int gpu = 0; gpu < DS4_N_GPU; gpu++) {
        const uint64_t first = (uint64_t)gpu * experts_per_gpu;
        uint64_t count = 0;
        if (first < experts_total) {
            const uint64_t remaining = experts_total - first;
            count = remaining < experts_per_gpu ? remaining : experts_per_gpu;
        }
        const uint64_t bytes = checked_mul(count, weight_per + scale_per);
        emit_row(st, "ep_expert", semantic, source, layer, "routed_expert",
                 dtype, shape, layout, gpu, -1, gpu, "expert_id", gpu, DS4_N_EP,
                 (int)first, (int)count, -1, 0, bytes, shard_file, weight_offset,
                 source_len, kernel);
        st->gpu[gpu].ep_expert += bytes;
        st->expert_rows++;
    }
}

static void process_tm_index(const options *opt, emit_state *st) {
    FILE *fp = open_read_joined(opt->pack_dir, "turbomind-pack-index.tsv");
    char *line = NULL;
    size_t cap = 0;
    bool first = true;
    while (getline(&line, &cap, fp) >= 0) {
        if (first) {
            first = false;
            continue;
        }
        char *cols[40] = {0};
        const int n = split_tsv(line, cols, 40);
        emit_tm_row(st, cols, n);
    }
    free(line);
    fclose(fp);
}

static void emit_kv_records(const options *opt, emit_state *st) {
    for (int layer = 0; layer < DS4_N_LAYER; layer++) {
        const int ratio = layer_ratio(layer);
        const uint64_t attn_rows = (uint64_t)DS4_N_SWA + (ratio ? opt->ctx / (uint64_t)ratio : 0);
        const uint64_t attn_bytes = layer_attn_kv_bytes(layer, opt->ctx, opt->kv);
        const uint64_t attn_gpu_bytes = checked_mul(ceil_div_u64(attn_bytes, DS4_N_TP), opt->slots);
        char tensor_id[128];
        snprintf(tensor_id, sizeof(tensor_id), "kv.attn.blk.%d", layer);
        for (int gpu = 0; gpu < DS4_N_GPU; gpu++) {
            emit_row(st, "kv_shard", tensor_id, "-", layer, "attn_kv", kv_name(opt->kv),
                     "[rows_per_slot x 512]", "tp_sharded_kv", gpu, gpu, -1,
                     "kv_dim", gpu, DS4_N_TP, -1, 0, ratio, attn_rows,
                     attn_gpu_bytes, "-", 0, attn_bytes, "ds4_tp_sharded_kv");
            st->gpu[gpu].kv += attn_gpu_bytes;
            st->kv_rows++;
        }

        if (ratio == 4) {
            const uint64_t index_rows = opt->ctx / 4u;
            const uint64_t index_bytes = layer_indexer_kv_bytes(layer, opt->ctx, opt->kv);
            const uint64_t index_gpu_bytes =
                checked_mul(ceil_div_u64(index_bytes, DS4_N_TP), opt->slots);
            snprintf(tensor_id, sizeof(tensor_id), "kv.indexer.blk.%d", layer);
            for (int gpu = 0; gpu < DS4_N_GPU; gpu++) {
                emit_row(st, "kv_shard", tensor_id, "-", layer, "indexer_kv",
                         kv_name(opt->kv), "[rows_per_slot x 128]", "tp_sharded_kv",
                         gpu, gpu, -1, "kv_dim", gpu, DS4_N_TP, -1, 0, ratio,
                         index_rows, index_gpu_bytes, "-", 0, index_bytes,
                         "ds4_tp_sharded_indexer_kv");
                st->gpu[gpu].kv += index_gpu_bytes;
                st->kv_rows++;
            }
        }

        if (ratio != 0) {
            const uint64_t comp_bytes = layer_comp_state_bytes(layer, opt->ctx);
            const uint64_t comp_gpu_bytes =
                checked_mul(ceil_div_u64(comp_bytes, DS4_N_TP), opt->slots);
            snprintf(tensor_id, sizeof(tensor_id), "kv.comp_state.blk.%d", layer);
            for (int gpu = 0; gpu < DS4_N_GPU; gpu++) {
                emit_row(st, "kv_comp_state", tensor_id, "-", layer, "compression_state",
                         "f32", "[state]", "tp_sharded_state", gpu, gpu, -1,
                         "state_dim", gpu, DS4_N_TP, -1, 0, ratio, 0,
                         comp_gpu_bytes, "-", 0, comp_bytes,
                         "ds4_tp_sharded_compression_state");
                st->gpu[gpu].comp_state += comp_gpu_bytes;
                st->kv_rows++;
            }
        }
    }
}

static uint64_t gpu_total(const gpu_summary *g) {
    return g->dense_tp + g->replicated_control + g->ep_expert + g->kv +
           g->comp_state + g->scratch + g->reserve;
}

static uint64_t gpu_total_with_dense_f16_keep(const gpu_summary *g) {
    return gpu_total(g) + g->dense_f16_cache + g->dense_bf16_f16_shadow;
}

static uint64_t gpu_total_with_dense_f16_replace(const gpu_summary *g) {
    return gpu_total(g) - g->dense_f8_cacheable_packed - g->dense_bf16_shadowable_packed +
           g->dense_f16_cache + g->dense_bf16_f16_shadow;
}

static void write_summary(const options *opt, const emit_state *st) {
    FILE *fp = open_write_joined(opt->out_dir, "tp-ep-memory-summary.tsv");
    fprintf(fp,
            "gpu\tdense_tp_bytes\treplicated_control_bytes\tep_expert_bytes\t"
            "kv_bytes\tcomp_state_bytes\tscratch_bytes\treserve_bytes\tbase_total_bytes\t"
            "base_total_gib\tdense_f8_cacheable_packed_bytes\tdense_f16_cache_bytes\t"
            "dense_bf16_shadowable_packed_bytes\tdense_bf16_f16_shadow_bytes\t"
            "dense_f16_keep_total_bytes\tdense_f16_keep_total_gib\t"
            "dense_f16_replace_total_bytes\tdense_f16_replace_total_gib\t"
            "headroom_replace_gib\trows\n");
    for (int gpu = 0; gpu < DS4_N_GPU; gpu++) {
        const gpu_summary *g = &st->gpu[gpu];
        const uint64_t base_total = gpu_total(g);
        const uint64_t keep_total = gpu_total_with_dense_f16_keep(g);
        const uint64_t replace_total = gpu_total_with_dense_f16_replace(g);
        const double headroom_replace_gib = 32.0 - as_gib(replace_total);
        fprintf(fp,
                "%d\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64
                "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%.3f\t%" PRIu64
                "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%.3f"
                "\t%" PRIu64 "\t%.3f\t%.3f\t%" PRIu64 "\n",
                gpu, g->dense_tp, g->replicated_control, g->ep_expert, g->kv,
                g->comp_state, g->scratch, g->reserve, base_total,
                as_gib(base_total), g->dense_f8_cacheable_packed,
                g->dense_f16_cache, g->dense_bf16_shadowable_packed,
                g->dense_bf16_f16_shadow, keep_total, as_gib(keep_total),
                replace_total, as_gib(replace_total), headroom_replace_gib,
                g->rows);
    }
    fclose(fp);
}

static void write_markdown(const options *opt, const emit_state *st) {
    FILE *fp = open_write_joined(opt->out_dir, "tp-ep-pack-contract.md");
    fprintf(fp, "# DS4 V100 TP/EP Pack Contract\n\n");
    fprintf(fp, "Generated from `%s`.\n\n", opt->pack_dir);
    fprintf(fp, "Topology: `PP=1`, `TP=8`, `EP=8`, KV sharded, MTP off.\n\n");
    fprintf(fp, "Config: slots `%u`, ctx `%" PRIu64 "`, KV `%s`.\n\n",
            opt->slots, opt->ctx, kv_name(opt->kv));
    fprintf(fp, "## Record Counts\n\n");
    fprintf(fp, "- source pack rows read: `%" PRIu64 "`\n", st->pack_rows);
    fprintf(fp, "- dense TP contract rows: `%" PRIu64 "`\n", st->dense_rows);
    fprintf(fp, "- F8 dense rows eligible for FP16 cache: `%" PRIu64 "`\n",
            st->dense_f8_cacheable_rows);
    fprintf(fp, "- BF16 dense rows eligible for FP16 shadow: `%" PRIu64 "`\n",
            st->dense_bf16_shadowable_rows);
    fprintf(fp, "- replicated control rows: `%" PRIu64 "`\n", st->control_rows);
    fprintf(fp, "- EP expert rows: `%" PRIu64 "`\n", st->expert_rows);
    fprintf(fp, "- KV/state rows: `%" PRIu64 "`\n\n", st->kv_rows);
    fprintf(fp, "## Memory Summary\n\n");
    fprintf(fp, "| GPU | Dense TP | Control | EP expert | KV | Comp | Scratch | Reserve | Total |\n");
    fprintf(fp, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (int gpu = 0; gpu < DS4_N_GPU; gpu++) {
        const gpu_summary *g = &st->gpu[gpu];
        fprintf(fp, "| %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |\n",
                gpu, as_gib(g->dense_tp), as_gib(g->replicated_control),
                as_gib(g->ep_expert), as_gib(g->kv), as_gib(g->comp_state),
                as_gib(g->scratch), as_gib(g->reserve), as_gib(gpu_total(g)));
    }
    fprintf(fp, "\n## Dense FP16 Runtime Cache Admission\n\n");
    fprintf(fp, "The base plan keeps source packed dense weights resident. Sprint 244 showed a\n");
    fprintf(fp, "resident FP16/cuBLAS dense path is much faster for the representative TP/EP\n");
    fprintf(fp, "layer loop, so this section estimates the memory cost of materializing dense\n");
    fprintf(fp, "runtime FP16 weights on each TP rank.\n\n");
    fprintf(fp, "| GPU | F8 packed eligible | F8->FP16 cache | BF16 packed shadowable | BF16->FP16 shadow | Keep-packed total | Replace-source total | Replace headroom vs 32 GiB |\n");
    fprintf(fp, "|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (int gpu = 0; gpu < DS4_N_GPU; gpu++) {
        const gpu_summary *g = &st->gpu[gpu];
        const uint64_t replace_total = gpu_total_with_dense_f16_replace(g);
        fprintf(fp, "| %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |\n",
                gpu, as_gib(g->dense_f8_cacheable_packed),
                as_gib(g->dense_f16_cache),
                as_gib(g->dense_bf16_shadowable_packed),
                as_gib(g->dense_bf16_f16_shadow),
                as_gib(gpu_total_with_dense_f16_keep(g)),
                as_gib(replace_total),
                32.0 - as_gib(replace_total));
    }
    fprintf(fp, "\n## Contract Rules\n\n");
    fprintf(fp, "- Dense low-bit tensors are TP8 sharded; PP/layer ownership is rejected.\n");
    fprintf(fp, "- Small F32/I32 control and router tensors are replicated across TP ranks.\n");
    fprintf(fp, "- Routed experts are EP8 sharded by expert id, `32` experts per GPU.\n");
    fprintf(fp, "- KV/cache records use the corrected DS4 compression schedule.\n");
    fprintf(fp, "- Dense FP16 cache admission is a runtime option, not a source-format change.\n");
    fprintf(fp, "  The practical serving target replaces cacheable dense source tensors in\n");
    fprintf(fp, "  VRAM with the FP16 runtime cache instead of keeping both copies.\n");
    fclose(fp);
}

int main(int argc, char **argv) {
    options opt;
    parse_args(argc, argv, &opt);

    if (mkdir(opt.out_dir, 0775) != 0 && errno != EEXIST) {
        die_errno("mkdir", opt.out_dir);
    }

    emit_state st;
    memset(&st, 0, sizeof(st));
    for (int gpu = 0; gpu < DS4_N_GPU; gpu++) {
        st.gpu[gpu].scratch = from_gib(opt.scratch_gib);
        st.gpu[gpu].reserve = from_gib(opt.reserve_gib);
    }

    st.contract = open_write_joined(opt.out_dir, "tp-ep-pack-contract.tsv");
    emit_contract_header(st.contract);
    process_pack_index(&opt, &st);
    process_tm_index(&opt, &st);
    emit_kv_records(&opt, &st);
    fclose(st.contract);

    write_summary(&opt, &st);
    write_markdown(&opt, &st);

    printf("tp_ep_pack_contract\tout_dir=%s\tpack_rows=%" PRIu64
           "\tdense_rows=%" PRIu64 "\tcontrol_rows=%" PRIu64
           "\texpert_rows=%" PRIu64 "\tkv_rows=%" PRIu64 "\n",
           opt.out_dir, st.pack_rows, st.dense_rows, st.control_rows,
           st.expert_rows, st.kv_rows);
    for (int gpu = 0; gpu < DS4_N_GPU; gpu++) {
        printf("gpu\t%d\ttotal_gib\t%.3f\tdense_gib\t%.3f\tcontrol_gib\t%.3f\t"
               "expert_gib\t%.3f\tkv_gib\t%.3f\tcomp_gib\t%.3f\t"
               "dense_f16_replace_total_gib\t%.3f\tdense_f16_replace_headroom_gib\t%.3f\n",
               gpu, as_gib(gpu_total(&st.gpu[gpu])),
               as_gib(st.gpu[gpu].dense_tp),
               as_gib(st.gpu[gpu].replicated_control),
               as_gib(st.gpu[gpu].ep_expert),
               as_gib(st.gpu[gpu].kv),
               as_gib(st.gpu[gpu].comp_state),
               as_gib(gpu_total_with_dense_f16_replace(&st.gpu[gpu])),
               32.0 - as_gib(gpu_total_with_dense_f16_replace(&st.gpu[gpu])));
    }
    return 0;
}
