#include "engine/tp_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static unsigned long long parse_u64(const char *s) {
    return strtoull(s, nullptr, 10);
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s [--ctx N] [--slots N] [--scratch-mib N] "
                 "[--kv-dtype f16|f8|f8_e4m3_b128|f8_e5m2_b128|q8_0] [--dense-kv-slice] "
                 "[--typed-kv-row] [--device-kv-row] [--kind attn|attn_raw|indexer] "
                 "[--layer N] [--slot N] [--position N] [--indexer on|off]\n",
                 argv0);
}

static bool parse_kv(const char *s, ds4_v100_tp_kv_dtype *out) {
    if (std::strcmp(s, "f16") == 0) *out = DS4_V100_TP_KV_F16;
    else if (std::strcmp(s, "f8") == 0 || std::strcmp(s, "f8_e4m3_b128") == 0)
        *out = DS4_V100_TP_KV_F8_E4M3_B128;
    else if (std::strcmp(s, "f8_e5m2_b128") == 0 || std::strcmp(s, "e5m2") == 0)
        *out = DS4_V100_TP_KV_F8_E5M2_B128;
    else if (std::strcmp(s, "q8") == 0 || std::strcmp(s, "q8_0") == 0)
        *out = DS4_V100_TP_KV_Q8_0;
    else return false;
    return true;
}

int main(int argc, char **argv) {
    ds4_v100_tp_runtime_config cfg;
    ds4_v100_tp_runtime_default_config(&cfg);
    bool dense_kv_slice = false;
    bool typed_kv_row = false;
    bool device_kv_row = false;
    ds4_v100_tp_kv_row_kind row_kind = DS4_V100_TP_KV_ROW_ATTN;
    int layer = 2;
    uint32_t slot = 0;
    unsigned long long position = 0;
    int indexer = 0;

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
        } else if (std::strcmp(arg, "--dense-kv-slice") == 0) {
            dense_kv_slice = true;
        } else if (std::strcmp(arg, "--typed-kv-row") == 0) {
            typed_kv_row = true;
        } else if (std::strcmp(arg, "--device-kv-row") == 0) {
            device_kv_row = true;
        } else if (std::strcmp(arg, "--kind") == 0 && val) {
            if (std::strcmp(val, "attn") == 0 || std::strcmp(val, "attention") == 0) {
                row_kind = DS4_V100_TP_KV_ROW_ATTN;
            } else if (std::strcmp(val, "attn_raw") == 0 ||
                       std::strcmp(val, "attention_raw") == 0 ||
                       std::strcmp(val, "raw") == 0) {
                row_kind = DS4_V100_TP_KV_ROW_ATTN_RAW;
            } else if (std::strcmp(val, "indexer") == 0) {
                row_kind = DS4_V100_TP_KV_ROW_INDEXER;
            } else {
                usage(argv[0]);
                return 2;
            }
            ++i;
        } else if (std::strcmp(arg, "--layer") == 0 && val) {
            layer = (int)parse_u64(val);
            ++i;
        } else if (std::strcmp(arg, "--slot") == 0 && val) {
            slot = (uint32_t)parse_u64(val);
            ++i;
        } else if (std::strcmp(arg, "--position") == 0 && val) {
            position = parse_u64(val);
            ++i;
        } else if (std::strcmp(arg, "--indexer") == 0 && val) {
            if (std::strcmp(val, "on") == 0 || std::strcmp(val, "true") == 0 ||
                std::strcmp(val, "1") == 0) {
                indexer = 1;
            } else if (std::strcmp(val, "off") == 0 || std::strcmp(val, "false") == 0 ||
                       std::strcmp(val, "0") == 0) {
                indexer = 0;
            } else {
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

    if (dense_kv_slice) {
        ds4_v100_tp_dense_kv_result result;
        if (ds4_v100_tp_runtime_dense_kv_slice(rt, layer, slot, position, indexer,
                                               &result, err, sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_dense_kv_slice_failed\t%s\n", err);
            ds4_v100_tp_runtime_close(rt);
            return 1;
        }
        std::printf("tp_dense_kv_slice\tctx=%llu\tslots=%u\thidden=%u\t"
                    "layer=%d\tratio=%d\tslot=%u\tposition=%llu\t"
                    "attn_row=%llu\tindexer_row=%llu\tmax_abs=%.9f\n",
                    (unsigned long long)cfg.ctx, cfg.slots, cfg.hidden,
                    result.layer, result.ratio, result.slot,
                    (unsigned long long)result.position,
                    (unsigned long long)result.attn_row,
                    (unsigned long long)result.indexer_row,
                    result.max_abs);
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            std::printf("gpu\t%d\tattn_offset\t%llu\tattn_row_bytes\t%llu\t"
                        "indexer_offset\t%llu\tindexer_row_bytes\t%llu\n",
                        gpu,
                        (unsigned long long)result.attn_offset[gpu],
                        (unsigned long long)result.attn_row_bytes[gpu],
                        (unsigned long long)result.indexer_offset[gpu],
                        (unsigned long long)result.indexer_row_bytes[gpu]);
        }
        ds4_v100_tp_runtime_close(rt);
        return result.max_abs <= 0.0 ? 0 : 1;
    }

    if (typed_kv_row) {
        ds4_v100_tp_kv_row_roundtrip_result result;
        if (ds4_v100_tp_runtime_kv_row_roundtrip_f32(rt, layer, slot, position,
                                                     row_kind, &result, err,
                                                     sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_typed_kv_row_failed\t%s\n", err);
            ds4_v100_tp_runtime_close(rt);
            return 1;
        }
        const char *kind_name =
            result.view.kind == DS4_V100_TP_KV_ROW_INDEXER
                ? "indexer"
                : (result.view.kind == DS4_V100_TP_KV_ROW_ATTN_RAW ? "attn_raw"
                                                                    : "attn");
        std::printf("tp_typed_kv_row\tctx=%llu\tslots=%u\thidden=%u\t"
                    "layer=%d\tratio=%d\tslot=%u\tposition=%llu\tkind=%s\t"
                    "physical_row=%llu\tlogical_cols=%u\tlogical_row_bytes=%llu\t"
                    "row_bytes_per_gpu=%llu\tbad_values=%u\tbyte_mismatches=%u\tfirst_bad_index=%u\t"
                    "first_bad_got=%u\tfirst_bad_expected=%u\tmax_abs=%.9f\t"
                    "mean_abs=%.9f\tchecksum=%llu\n",
                    (unsigned long long)cfg.ctx, cfg.slots, cfg.hidden,
                    result.view.layer, result.view.ratio, result.view.slot,
                    (unsigned long long)result.view.position, kind_name,
                    (unsigned long long)result.view.physical_row,
                    result.view.logical_cols,
                    (unsigned long long)result.view.logical_row_bytes,
                    (unsigned long long)result.view.row_bytes[0],
                    result.bad_values, result.byte_mismatches, result.first_bad_index,
                    (unsigned int)result.first_bad_got,
                    (unsigned int)result.first_bad_expected,
                    result.max_abs, result.mean_abs,
                    (unsigned long long)result.checksum);
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            std::printf("gpu\t%d\toffset\t%llu\trow_bytes\t%llu\n",
                        gpu,
                        (unsigned long long)result.view.offset[gpu],
                        (unsigned long long)result.view.row_bytes[gpu]);
        }
        ds4_v100_tp_runtime_close(rt);
        return result.bad_values == 0 && result.max_abs == 0.0 ? 0 : 1;
    }

    if (device_kv_row) {
        ds4_v100_tp_kv_device_roundtrip_result result;
        if (ds4_v100_tp_runtime_kv_row_device_roundtrip_f32(
                rt, layer, slot, position, row_kind, &result, err, sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_device_kv_row_failed\t%s\n", err);
            ds4_v100_tp_runtime_close(rt);
            return 1;
        }
        const char *kind_name =
            result.view.kind == DS4_V100_TP_KV_ROW_INDEXER
                ? "indexer"
                : (result.view.kind == DS4_V100_TP_KV_ROW_ATTN_RAW ? "attn_raw"
                                                                    : "attn");
        std::printf("tp_device_kv_row\tctx=%llu\tslots=%u\thidden=%u\t"
                    "layer=%d\tratio=%d\tslot=%u\tposition=%llu\tkind=%s\t"
                    "physical_row=%llu\tlogical_cols=%u\tlogical_row_bytes=%llu\t"
                    "row_bytes_per_gpu=%llu\tbad_values=%u\tmax_abs=%.9f\t"
                    "mean_abs=%.9f\tchecksum=%llu\n",
                    (unsigned long long)cfg.ctx, cfg.slots, cfg.hidden,
                    result.view.layer, result.view.ratio, result.view.slot,
                    (unsigned long long)result.view.position, kind_name,
                    (unsigned long long)result.view.physical_row,
                    result.view.logical_cols,
                    (unsigned long long)result.view.logical_row_bytes,
                    (unsigned long long)result.view.row_bytes[0],
                    result.bad_values, result.max_abs, result.mean_abs,
                    (unsigned long long)result.checksum);
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            std::printf("gpu\t%d\toffset\t%llu\trow_bytes\t%llu\n",
                        gpu,
                        (unsigned long long)result.view.offset[gpu],
                        (unsigned long long)result.view.row_bytes[gpu]);
        }
        ds4_v100_tp_runtime_close(rt);
        return result.bad_values == 0 && result.max_abs == 0.0 ? 0 : 1;
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
