#include "ds4_v100_context.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_u64(const char *s, unsigned long long *out) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (!end || *end != '\0') return 1;
    *out = v;
    return 0;
}

int main(int argc, char **argv) {
    int production = 0;
    int requested_stages = 0;
    const char *pack_index = NULL;
    unsigned long long planned_kv_mib = 0;
    unsigned long long reserve_mib = 2048;
    unsigned long long output_head_mib = 0;
    unsigned long long mtp_mib = 0;
    unsigned long long kv_ctx = 0;
    unsigned long long kv_slots = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--production")) {
            production = 1;
        } else if (!strcmp(argv[i], "--stages") && i + 1 < argc) {
            requested_stages = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--pack-index") && i + 1 < argc) {
            pack_index = argv[++i];
        } else if (!strcmp(argv[i], "--planned-kv-mib") && i + 1 < argc) {
            if (parse_u64(argv[++i], &planned_kv_mib)) return 2;
        } else if (!strcmp(argv[i], "--kv-ctx") && i + 1 < argc) {
            if (parse_u64(argv[++i], &kv_ctx)) return 2;
        } else if (!strcmp(argv[i], "--kv-slots") && i + 1 < argc) {
            if (parse_u64(argv[++i], &kv_slots) || kv_slots == 0) return 2;
        } else if (!strcmp(argv[i], "--reserve-mib") && i + 1 < argc) {
            if (parse_u64(argv[++i], &reserve_mib)) return 2;
        } else if (!strcmp(argv[i], "--output-head-mib") && i + 1 < argc) {
            if (parse_u64(argv[++i], &output_head_mib)) return 2;
        } else if (!strcmp(argv[i], "--mtp-mib") && i + 1 < argc) {
            if (parse_u64(argv[++i], &mtp_mib)) return 2;
        } else {
            fprintf(stderr,
                    "usage: tests/cuda_v100_context_smoke [--production] [--stages N]\n"
                    "                                    [--pack-index PATH] [--planned-kv-mib N]\n"
                    "                                    [--kv-ctx N] [--kv-slots N]\n"
                    "                                    [--reserve-mib N] [--output-head-mib N] [--mtp-mib N]\n");
            return 2;
        }
    }

    char err[512];
    ds4_v100_device_fact facts[DS4_V100_EXPECTED_GPUS];
    int n = 0;
    if (ds4_v100_cuda_collect_device_facts(facts, DS4_V100_EXPECTED_GPUS,
                                           &n, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_context_smoke: %s\n", err);
        return 1;
    }
    if (n < 1) {
        fprintf(stderr, "cuda_v100_context_smoke: no CUDA devices visible\n");
        return 1;
    }
    int stages = requested_stages ? requested_stages : (production ? DS4_V100_EXPECTED_GPUS : (n >= 2 ? 2 : 1));
    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    opts.expected_gpus = stages;
    opts.pack_index_path = pack_index;
    opts.relay_max_active_slots = 1;
    opts.scratch_bytes_per_gpu = 1024 * 1024;
    opts.enable_f32_debug_relay = true;
    opts.require_production_topology = production != 0;
    opts.planned_kv_bytes_per_gpu = planned_kv_mib * 1048576ull;
    opts.kv_ctx_tokens = kv_ctx;
    opts.kv_active_slots = kv_slots;
    opts.reserve_bytes_per_gpu = reserve_mib * 1048576ull;
    opts.output_head_reserve_bytes = output_head_mib * 1048576ull;
    opts.mtp_reserve_bytes = mtp_mib * 1048576ull;

    ds4_v100_cuda_context *ctx = NULL;
    if (ds4_v100_cuda_context_open(&ctx, &opts, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_context_smoke: %s\n", err);
        return 1;
    }
    printf("cuda_v100_context_smoke: devices=%d stages=%d production=%d\n", n, stages, production);
    for (int i = 0; i < n; i++) {
        printf("device\t%d\tcc\t%d.%d\tmem\t%llu\tpci\t%s\n",
               i, facts[i].cc_major, facts[i].cc_minor,
               (unsigned long long)facts[i].total_global_mem,
               facts[i].pci_bus_id);
    }
    printf("p2p_from\\to");
    for (int j = 0; j < n; j++) printf("\t%d", j);
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("%d", i);
        for (int j = 0; j < n; j++) printf("\t%d", facts[i].peer_access[j] ? 1 : 0);
        printf("\n");
    }
    ds4_v100_cuda_context_close(ctx);
    puts("cuda_v100_context_smoke: ok");
    return 0;
}
