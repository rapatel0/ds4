#include "engine/context.h"

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tools/ds4-v100-context-smoke [options]\n"
            "\n"
            "Options:\n"
            "  --pack-index PATH          Pack index TSV to bind descriptors\n"
            "  --tm-index PATH            TurboMind pack index TSV to bind descriptors\n"
            "  --allow-partial            Skip full layer-skeleton validation for bounded packs\n"
            "  --slots N                  Relay active-slot capacity (default 1)\n"
            "  --scratch-bytes N          Scratch budget per GPU\n"
            "  --reserve-mib N            Reserve floor per GPU (default 2048)\n"
            "  --planned-kv-mib N         Planned KV reserve per GPU\n"
            "  --kv-ctx N                 Derive F16 KV bytes for this per-slot context\n"
            "  --kv-slots N               Active slots for derived F16 KV budget\n"
            "  --output-head-mib N        gpu7 output-head reserve placeholder\n"
            "  --mtp-mib N                gpu7 MTP reserve placeholder\n"
            "  --f32-debug-relay          Include FP32 debug relay buffers\n"
            "  --production-topology      Require injected/production topology facts\n"
            "  --mode probe-only|use-existing-arenas|full-resident\n"
            "  --help                     Show this help\n");
}

static int parse_u64(const char *s, uint64_t *out) {
    if (!s || !*s) return 1;
    errno = 0;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end != '\0') return 1;
    *out = (uint64_t)v;
    return 0;
}

static int next_arg(int *i, int argc, char **argv, const char **out) {
    if (*i + 1 >= argc) return 1;
    *out = argv[++(*i)];
    return 0;
}

int main(int argc, char **argv) {
    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    bool allow_partial = false;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        const char *val = NULL;
        uint64_t n = 0;
        if (!strcmp(arg, "--help")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(arg, "--pack-index")) {
            if (next_arg(&i, argc, argv, &val)) {
                fprintf(stderr, "missing value for %s\n", arg);
                return 2;
            }
            opts.pack_index_path = val;
        } else if (!strcmp(arg, "--tm-index")) {
            if (next_arg(&i, argc, argv, &val)) {
                fprintf(stderr, "missing value for %s\n", arg);
                return 2;
            }
            opts.turbomind_pack_index_path = val;
        } else if (!strcmp(arg, "--allow-partial")) {
            allow_partial = true;
        } else if (!strcmp(arg, "--slots")) {
            if (next_arg(&i, argc, argv, &val) || parse_u64(val, &n) || n == 0) {
                fprintf(stderr, "bad value for %s\n", arg);
                return 2;
            }
            opts.relay_max_active_slots = n;
        } else if (!strcmp(arg, "--scratch-bytes")) {
            if (next_arg(&i, argc, argv, &val) || parse_u64(val, &n)) {
                fprintf(stderr, "bad value for %s\n", arg);
                return 2;
            }
            opts.scratch_bytes_per_gpu = n;
        } else if (!strcmp(arg, "--reserve-mib")) {
            if (next_arg(&i, argc, argv, &val) || parse_u64(val, &n)) {
                fprintf(stderr, "bad value for %s\n", arg);
                return 2;
            }
            opts.reserve_bytes_per_gpu = n * 1048576ull;
        } else if (!strcmp(arg, "--planned-kv-mib")) {
            if (next_arg(&i, argc, argv, &val) || parse_u64(val, &n)) {
                fprintf(stderr, "bad value for %s\n", arg);
                return 2;
            }
            opts.planned_kv_bytes_per_gpu = n * 1048576ull;
        } else if (!strcmp(arg, "--kv-ctx")) {
            if (next_arg(&i, argc, argv, &val) || parse_u64(val, &n) || n == 0) {
                fprintf(stderr, "bad value for %s\n", arg);
                return 2;
            }
            opts.kv_ctx_tokens = n;
        } else if (!strcmp(arg, "--kv-slots")) {
            if (next_arg(&i, argc, argv, &val) || parse_u64(val, &n) || n == 0) {
                fprintf(stderr, "bad value for %s\n", arg);
                return 2;
            }
            opts.kv_active_slots = n;
        } else if (!strcmp(arg, "--output-head-mib")) {
            if (next_arg(&i, argc, argv, &val) || parse_u64(val, &n)) {
                fprintf(stderr, "bad value for %s\n", arg);
                return 2;
            }
            opts.output_head_reserve_bytes = n * 1048576ull;
        } else if (!strcmp(arg, "--mtp-mib")) {
            if (next_arg(&i, argc, argv, &val) || parse_u64(val, &n)) {
                fprintf(stderr, "bad value for %s\n", arg);
                return 2;
            }
            opts.mtp_reserve_bytes = n * 1048576ull;
        } else if (!strcmp(arg, "--f32-debug-relay")) {
            opts.enable_f32_debug_relay = true;
        } else if (!strcmp(arg, "--production-topology")) {
            opts.require_production_topology = true;
        } else if (!strcmp(arg, "--mode")) {
            if (next_arg(&i, argc, argv, &val)) {
                fprintf(stderr, "missing value for %s\n", arg);
                return 2;
            }
            if (!strcmp(val, "probe-only")) opts.mode = DS4_V100_INIT_PROBE_ONLY;
            else if (!strcmp(val, "use-existing-arenas")) opts.mode = DS4_V100_INIT_USE_EXISTING_ARENAS;
            else if (!strcmp(val, "full-resident")) opts.mode = DS4_V100_INIT_FULL_RESIDENT;
            else {
                fprintf(stderr, "bad mode: %s\n", val);
                return 2;
            }
        } else {
            fprintf(stderr, "unknown option: %s\n", arg);
            usage(stderr);
            return 2;
        }
    }

    char err[512];
    ds4_v100_context *ctx = NULL;
    if (ds4_v100_context_open(&ctx, &opts, err, sizeof(err))) {
        fprintf(stderr, "context_open\tFAIL\t%s\n", err);
        return 1;
    }
    ds4_v100_context_print_report(ctx, stdout);
    if (allow_partial) {
        fprintf(stdout, "layer_skeleton_result\tSKIPPED_PARTIAL\n");
    } else {
        fprintf(stdout, "layer_skeleton_begin\n");
        if (ds4_v100_context_validate_layer_skeleton(ctx, stdout, err, sizeof(err))) {
            fprintf(stderr, "layer_skeleton\tFAIL\t%s\n", err);
            ds4_v100_context_close(ctx);
            return 1;
        }
        fprintf(stdout, "layer_skeleton_result\tOK\n");
    }
    fprintf(stdout, "context_smoke_result\tOK\n");
    ds4_v100_context_close(ctx);
    return 0;
}
