#include "ds4_v100_tp_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static unsigned long long parse_u64(const char *s) {
    return strtoull(s, nullptr, 10);
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s [--ctx N] [--slots N] [--scratch-mib N] "
                 "[--kv-dtype f16|f8|q8_0]\n",
                 argv0);
}

static bool parse_kv(const char *s, ds4_v100_tp_kv_dtype *out) {
    if (std::strcmp(s, "f16") == 0) *out = DS4_V100_TP_KV_F16;
    else if (std::strcmp(s, "f8") == 0 || std::strcmp(s, "f8_e4m3_b128") == 0)
        *out = DS4_V100_TP_KV_F8_E4M3_B128;
    else if (std::strcmp(s, "q8") == 0 || std::strcmp(s, "q8_0") == 0)
        *out = DS4_V100_TP_KV_Q8_0;
    else return false;
    return true;
}

int main(int argc, char **argv) {
    ds4_v100_tp_runtime_config cfg;
    ds4_v100_tp_runtime_default_config(&cfg);

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        const char *val = i + 1 < argc ? argv[i + 1] : nullptr;
        if (std::strcmp(arg, "--ctx") == 0 && val) {
            cfg.ctx = parse_u64(val);
            ++i;
        } else if (std::strcmp(arg, "--slots") == 0 && val) {
            cfg.slots = (uint32_t)parse_u64(val);
            ++i;
        } else if (std::strcmp(arg, "--scratch-mib") == 0 && val) {
            cfg.scratch_bytes = parse_u64(val) * 1024ull * 1024ull;
            ++i;
        } else if (std::strcmp(arg, "--kv-dtype") == 0 && val) {
            if (!parse_kv(val, &cfg.kv_dtype)) {
                usage(argv[0]);
                return 2;
            }
            ++i;
        } else if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    char err[512] = {0};
    ds4_v100_tp_runtime *rt = nullptr;
    if (ds4_v100_tp_runtime_open(&rt, &cfg, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "tp_runtime_open_failed\t%s\n", err);
        return 1;
    }

    double max_abs = 0.0;
    if (ds4_v100_tp_runtime_fixture(rt, &max_abs, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "tp_runtime_fixture_failed\t%s\n", err);
        ds4_v100_tp_runtime_close(rt);
        return 1;
    }

    ds4_v100_tp_runtime_report report;
    ds4_v100_tp_runtime_get_report(rt, &report);
    std::printf("tp_runtime_smoke\tctx=%llu\tslots=%u\thidden=%u\t"
                "scratch_bytes=%llu\tfixture_max_abs=%.9f\n",
                (unsigned long long)cfg.ctx, cfg.slots, cfg.hidden,
                (unsigned long long)cfg.scratch_bytes, max_abs);
    for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
        const ds4_v100_tp_gpu_report *g = &report.gpu[gpu];
        std::printf("gpu\t%d\thidden_bytes\t%llu\tkv_bytes\t%llu\t"
                    "comp_state_bytes\t%llu\tscratch_bytes\t%llu\ttotal_bytes\t%llu\n",
                    gpu,
                    (unsigned long long)g->hidden_bytes,
                    (unsigned long long)g->kv_bytes,
                    (unsigned long long)g->comp_state_bytes,
                    (unsigned long long)g->scratch_bytes,
                    (unsigned long long)g->total_bytes);
    }
    ds4_v100_tp_runtime_close(rt);
    return max_abs <= 1.0e-5 ? 0 : 1;
}
