#define _FILE_OFFSET_BITS 64

#include "ds4_pack.h"
#include "ds4_turbomind_pack.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *source_index_path;
    const char *tm_index_path;
    double vram_gib;
    double reserve_gib;
    double kv_gib;
    double scratch_gib;
    int gpus;
} options;

typedef struct {
    uint64_t expert_payload[DS4_PACK_MAX_GPUS];
} expert_sum;

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-v100-turbomind-admit --source-index FILE --tm-index FILE [options]\n"
            "\n"
            "Options:\n"
            "  --gpus N          Number of GPUs to report. Default: inferred\n"
            "  --vram-gib N      VRAM per GPU. Default: 32\n"
            "  --reserve-gib N   Required free reserve per GPU. Default: 4\n"
            "  --kv-gib N        KV budget per GPU to include. Default: 0\n"
            "  --scratch-gib N   Scratch budget per GPU to include. Default: 0\n");
}

static double parse_double(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (errno || !end || *end || !isfinite(v) || v < 0.0) {
        fprintf(stderr, "ds4-v100-turbomind-admit: invalid %s: %s\n", name, s);
        exit(2);
    }
    return v;
}

static int parse_int(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end || v <= 0 || v > DS4_PACK_MAX_GPUS) {
        fprintf(stderr, "ds4-v100-turbomind-admit: invalid %s: %s\n", name, s);
        exit(2);
    }
    return (int)v;
}

static void parse_args(int argc, char **argv, options *opt) {
    memset(opt, 0, sizeof(*opt));
    opt->vram_gib = 32.0;
    opt->reserve_gib = 4.0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = NULL;
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            exit(0);
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "ds4-v100-turbomind-admit: %s requires a value\n", a);
            exit(2);
        }
        v = argv[++i];
        if (!strcmp(a, "--source-index")) {
            opt->source_index_path = v;
        } else if (!strcmp(a, "--tm-index")) {
            opt->tm_index_path = v;
        } else if (!strcmp(a, "--gpus")) {
            opt->gpus = parse_int(v, a);
        } else if (!strcmp(a, "--vram-gib")) {
            opt->vram_gib = parse_double(v, a);
        } else if (!strcmp(a, "--reserve-gib")) {
            opt->reserve_gib = parse_double(v, a);
        } else if (!strcmp(a, "--kv-gib")) {
            opt->kv_gib = parse_double(v, a);
        } else if (!strcmp(a, "--scratch-gib")) {
            opt->scratch_gib = parse_double(v, a);
        } else {
            fprintf(stderr, "ds4-v100-turbomind-admit: unknown option %s\n", a);
            usage(stderr);
            exit(2);
        }
    }
    if (!opt->source_index_path || !opt->tm_index_path) {
        usage(stderr);
        exit(2);
    }
}

static uint64_t gib_to_bytes(double gib) {
    const double bytes = gib * 1024.0 * 1024.0 * 1024.0;
    if (bytes >= (double)UINT64_MAX) return UINT64_MAX;
    return (uint64_t)(bytes + 0.5);
}

static double bytes_to_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static int is_routed_expert(const char *id) {
    return strstr(id, ".ffn_gate_exps.weight") ||
           strstr(id, ".ffn_up_exps.weight") ||
           strstr(id, ".ffn_down_exps.weight");
}

static int sum_expert_cb(const ds4_pack_entry *entry, void *ud) {
    expert_sum *sum = (expert_sum *)ud;
    if (!entry || entry->owning_gpu < 0 || entry->owning_gpu >= DS4_PACK_MAX_GPUS) return 0;
    if (is_routed_expert(entry->semantic_tensor_id)) {
        sum->expert_payload[entry->owning_gpu] += entry->byte_length;
    }
    return 0;
}

static uint64_t checked_add(uint64_t a, uint64_t b) {
    if (UINT64_MAX - a < b) return UINT64_MAX;
    return a + b;
}

int main(int argc, char **argv) {
    options opt;
    parse_args(argc, argv, &opt);

    char err[512] = {0};
    ds4_pack *source = NULL;
    if (ds4_pack_open(&source, opt.source_index_path, err, sizeof(err))) {
        fprintf(stderr, "ds4-v100-turbomind-admit: %s\n", err);
        return 1;
    }
    ds4_tm_pack *tm = NULL;
    if (ds4_tm_pack_open(&tm, opt.tm_index_path, err, sizeof(err))) {
        fprintf(stderr, "ds4-v100-turbomind-admit: %s\n", err);
        ds4_pack_close(source);
        return 1;
    }

    expert_sum expert;
    memset(&expert, 0, sizeof(expert));
    ds4_pack_for_each(source, sum_expert_cb, &expert);

    int gpus = opt.gpus;
    if (!gpus) {
        int source_max = ds4_pack_max_gpu(source);
        int tm_max = ds4_tm_pack_max_gpu(tm);
        int max_gpu = source_max > tm_max ? source_max : tm_max;
        gpus = max_gpu + 1;
        if (gpus <= 0) gpus = 1;
        if (gpus > DS4_PACK_MAX_GPUS) gpus = DS4_PACK_MAX_GPUS;
    }

    const uint64_t vram = gib_to_bytes(opt.vram_gib);
    const uint64_t reserve = gib_to_bytes(opt.reserve_gib);
    const uint64_t kv = gib_to_bytes(opt.kv_gib);
    const uint64_t scratch = gib_to_bytes(opt.scratch_gib);
    uint64_t fixed = checked_add(checked_add(reserve, kv), scratch);

    printf("gpu\tsource_arena_gib\tsource_payload_gib\tsource_expert_payload_gib\t"
           "tm_sidecar_gib\tduplicate_total_gib\treplacement_payload_total_gib\t"
           "duplicate_free_gib\treplacement_free_gib\tduplicate_status\treplacement_status\n");

    int duplicate_ok = 1;
    int replacement_ok = 1;
    for (int gpu = 0; gpu < gpus; gpu++) {
        uint64_t tm_bytes = 0;
        char sidecar[32];
        snprintf(sidecar, sizeof(sidecar), "gpu%d.turbomind", gpu);
        if (ds4_tm_pack_sidecar_bytes(tm, sidecar, &tm_bytes)) tm_bytes = 0;

        const uint64_t source_arena = ds4_pack_arena_bytes(source, gpu);
        const uint64_t source_payload = ds4_pack_payload_bytes(source, gpu);
        const uint64_t expert_payload = expert.expert_payload[gpu];
        uint64_t non_expert_payload =
            source_payload > expert_payload ? source_payload - expert_payload : 0;

        uint64_t duplicate_total = checked_add(checked_add(source_arena, tm_bytes), fixed);
        uint64_t replacement_total = checked_add(checked_add(non_expert_payload, tm_bytes), fixed);
        uint64_t duplicate_free = duplicate_total <= vram ? vram - duplicate_total : 0;
        uint64_t replacement_free = replacement_total <= vram ? vram - replacement_total : 0;
        const char *duplicate_status = duplicate_total <= vram ? "OK" : "OVER";
        const char *replacement_status = replacement_total <= vram ? "OK" : "OVER";
        if (duplicate_total > vram) duplicate_ok = 0;
        if (replacement_total > vram) replacement_ok = 0;

        printf("%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\n",
               gpu,
               bytes_to_gib(source_arena),
               bytes_to_gib(source_payload),
               bytes_to_gib(expert_payload),
               bytes_to_gib(tm_bytes),
               bytes_to_gib(duplicate_total),
               bytes_to_gib(replacement_total),
               bytes_to_gib(duplicate_free),
               bytes_to_gib(replacement_free),
               duplicate_status,
               replacement_status);
    }

    printf("summary\tduplicate_fit=%s\treplacement_fit=%s\tvram_gib=%.3f\treserve_gib=%.3f\tkv_gib=%.3f\tscratch_gib=%.3f\n",
           duplicate_ok ? "yes" : "no",
           replacement_ok ? "yes" : "no",
           opt.vram_gib,
           opt.reserve_gib,
           opt.kv_gib,
           opt.scratch_gib);

    ds4_tm_pack_close(tm);
    ds4_pack_close(source);
    return duplicate_ok || replacement_ok ? 0 : 3;
}
