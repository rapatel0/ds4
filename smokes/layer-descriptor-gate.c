#include "ds4_pack.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *suffix;
    const char *source_dtype;
    const char *runtime_layout;
    const char *kernel_family;
} expected_layer_tensor;

typedef struct {
    const char *name;
    const char *source_dtype;
    const char *runtime_layout;
    const char *kernel_family;
    int owning_gpu;
    int layer_id;
} expected_global_tensor;

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tools/ds4-v100-layer-descriptor-gate --index FILE [options]\n"
            "\n"
            "Options:\n"
            "  --index FILE      pack-index.tsv path\n"
            "  --layer N         layer to validate (default 2)\n"
            "  --gpus N          GPU count for ownership checks (default 8)\n"
            "  --help            show this help\n");
}

static int parse_int_arg(const char *s, const char *name) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || !*s || !end || *end != '\0' || v < 0 || v > 1024) {
        fprintf(stderr, "ds4-v100-layer-descriptor-gate: invalid %s: %s\n", name, s ? s : "(null)");
        exit(2);
    }
    return (int)v;
}

static int layer_ratio(int layer) {
    if (layer < 2) return 0;
    return (layer % 2) == 0 ? 4 : 128;
}

static int layer_device_8gpu(int layer) {
    if (layer < 0 || layer > 42) return -1;
    if (layer <= 5) return 0;
    if (layer <= 11) return 1;
    if (layer <= 17) return 2;
    if (layer <= 23) return 3;
    if (layer <= 29) return 4;
    if (layer <= 34) return 5;
    if (layer <= 39) return 6;
    return 7;
}

static int layer_device(int layer, int gpus) {
    if (gpus == 8) return layer_device_8gpu(layer);
    if (gpus <= 0 || layer < 0 || layer > 42) return -1;
    return (layer * gpus) / 43;
}

static int same_str(const char *a, const char *b) {
    return a && b && strcmp(a, b) == 0;
}

static int validate_entry(const ds4_pack_entry *e,
                          const char *id,
                          const char *dtype,
                          const char *layout,
                          const char *family,
                          int layer,
                          int gpu) {
    int failures = 0;
    if (e->layer_id != layer) failures++;
    if (e->owning_gpu != gpu) failures++;
    if (!same_str(e->source_dtype, dtype)) failures++;
    if (!same_str(e->runtime_layout, layout)) failures++;
    if (!same_str(e->kernel_family, family)) failures++;
    if (e->byte_length == 0) failures++;
    char shard[32];
    snprintf(shard, sizeof(shard), "gpu%d.weights", e->owning_gpu);
    if (!same_str(e->shard_file, shard)) failures++;
    printf("descriptor\t%s\t%s\tgpu=%d\tlayer=%d\tdtype=%s\tlayout=%s\tfamily=%s\tbytes=%" PRIu64 "\tshard=%s\toffset=%" PRIu64 "\n",
           failures ? "FAIL" : "PASS",
           id,
           e->owning_gpu,
           e->layer_id,
           e->source_dtype,
           e->runtime_layout,
           e->kernel_family,
           e->byte_length,
           e->shard_file,
           e->shard_offset);
    return failures;
}

static int require_layer_tensor(const ds4_pack *pack,
                                int layer,
                                int gpu,
                                const expected_layer_tensor *want,
                                uint64_t *bytes) {
    char id[160];
    snprintf(id, sizeof(id), "blk.%d.%s", layer, want->suffix);
    ds4_pack_entry e;
    if (ds4_pack_lookup(pack, id, &e)) {
        printf("descriptor\tFAIL\t%s\tmissing\n", id);
        return 1;
    }
    if (bytes) *bytes += e.byte_length;
    return validate_entry(&e,
                          id,
                          want->source_dtype,
                          want->runtime_layout,
                          want->kernel_family,
                          layer,
                          gpu);
}

static int require_global_tensor(const ds4_pack *pack,
                                 const expected_global_tensor *want,
                                 uint64_t *bytes) {
    ds4_pack_entry e;
    if (ds4_pack_lookup(pack, want->name, &e)) {
        printf("descriptor\tFAIL\t%s\tmissing\n", want->name);
        return 1;
    }
    if (bytes) *bytes += e.byte_length;
    return validate_entry(&e,
                          want->name,
                          want->source_dtype,
                          want->runtime_layout,
                          want->kernel_family,
                          want->layer_id,
                          want->owning_gpu);
}

int main(int argc, char **argv) {
    const char *index_path = NULL;
    int layer = 2;
    int gpus = 8;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(argv[i], "--index") && i + 1 < argc) {
            index_path = argv[++i];
        } else if (!strcmp(argv[i], "--layer") && i + 1 < argc) {
            layer = parse_int_arg(argv[++i], "--layer");
        } else if (!strcmp(argv[i], "--gpus") && i + 1 < argc) {
            gpus = parse_int_arg(argv[++i], "--gpus");
        } else {
            usage(stderr);
            return 2;
        }
    }
    if (!index_path) {
        usage(stderr);
        return 2;
    }
    if (layer < 0 || layer > 42 || gpus <= 0 || gpus > DS4_PACK_MAX_GPUS) {
        fprintf(stderr, "ds4-v100-layer-descriptor-gate: bad layer/gpus\n");
        return 2;
    }

    char err[512] = {0};
    ds4_pack *pack = NULL;
    if (ds4_pack_open(&pack, index_path, err, sizeof(err))) {
        fprintf(stderr, "ds4-v100-layer-descriptor-gate: %s\n", err);
        return 1;
    }

    static const expected_layer_tensor common[] = {
        {"attn_sinks", "f32", "source_f32_control", "ds4_attention_control"},
        {"attn_kv_a_norm.weight", "f32", "source_f32_control", "ds4_attention_control"},
        {"attn_q_a_norm.weight", "f32", "source_f32_control", "ds4_attention_control"},
        {"attn_kv_latent.weight", "f8_e4m3_b128", "source_f8_e4m3_b128_blocked", "v100_fp8_dequant_f16_hmma_pending"},
        {"attn_output_a.weight", "f8_e4m3_b128", "source_f8_e4m3_b128_blocked", "v100_fp8_dequant_f16_hmma_pending"},
        {"attn_output_b.weight", "f8_e4m3_b128", "source_f8_e4m3_b128_blocked", "v100_fp8_dequant_f16_hmma_pending"},
        {"attn_q_a.weight", "f8_e4m3_b128", "source_f8_e4m3_b128_blocked", "v100_fp8_dequant_f16_hmma_pending"},
        {"attn_q_b.weight", "f8_e4m3_b128", "source_f8_e4m3_b128_blocked", "v100_fp8_dequant_f16_hmma_pending"},
        {"attn_norm.weight", "f32", "source_f32_control", "ds4_attention_control"},
        {"ffn_down_exps.weight", "mxfp4", "source_mxfp4_grouped", "v100_grouped_mxfp4_pending"},
        {"ffn_gate_exps.weight", "mxfp4", "source_mxfp4_grouped", "v100_grouped_mxfp4_pending"},
        {"ffn_up_exps.weight", "mxfp4", "source_mxfp4_grouped", "v100_grouped_mxfp4_pending"},
        {"ffn_gate_inp.weight", "f32", "source_f32", "ds4_router_f32_i32"},
        {"ffn_gate_shexp.weight", "f8_e4m3_b128", "source_f8_e4m3_b128_blocked", "v100_shared_fp8_dense_pending"},
        {"ffn_down_shexp.weight", "f8_e4m3_b128", "source_f8_e4m3_b128_blocked", "v100_shared_fp8_dense_pending"},
        {"ffn_up_shexp.weight", "f8_e4m3_b128", "source_f8_e4m3_b128_blocked", "v100_shared_fp8_dense_pending"},
        {"ffn_norm.weight", "f32", "source_f32", "ds4_control_f32"},
        {"hc_attn_base", "f32", "source_f32", "ds4_hc_control_f32"},
        {"hc_attn_fn", "f32", "source_f32", "ds4_hc_control_f32"},
        {"hc_attn_scale", "f32", "source_f32", "ds4_hc_control_f32"},
        {"hc_ffn_base", "f32", "source_f32", "ds4_hc_control_f32"},
        {"hc_ffn_fn", "f32", "source_f32", "ds4_hc_control_f32"},
        {"hc_ffn_scale", "f32", "source_f32", "ds4_hc_control_f32"},
    };
    static const expected_layer_tensor ratio_compress[] = {
        {"attn_compress_ape", "f32", "source_f32", "ds4_compressor_bf16_f32_pending"},
        {"attn_compress_norm.weight", "f32", "source_f32", "ds4_compressor_bf16_f32_pending"},
        {"attn_compress_gate.weight", "bf16", "source_bf16", "ds4_compressor_bf16_f32_pending"},
        {"attn_compress_kv.weight", "bf16", "source_bf16", "ds4_compressor_bf16_f32_pending"},
    };
    static const expected_layer_tensor indexer[] = {
        {"indexer.compress_ape", "f32", "source_f32", "ds4_indexer_mixed_pending"},
        {"indexer.compress_norm.weight", "f32", "source_f32", "ds4_indexer_mixed_pending"},
        {"indexer.compress_gate.weight", "bf16", "source_bf16", "ds4_indexer_mixed_pending"},
        {"indexer.compress_kv.weight", "bf16", "source_bf16", "ds4_indexer_mixed_pending"},
        {"indexer.proj.weight", "bf16", "source_bf16", "ds4_indexer_mixed_pending"},
        {"indexer.attn_q_b.weight", "f8_e4m3_b128", "source_f8_e4m3_b128_blocked", "ds4_indexer_mixed_pending"},
    };
    static const expected_layer_tensor hash_router[] = {
        {"ffn_gate_tid2eid", "i32", "source_i32", "ds4_router_f32_i32"},
    };
    static const expected_layer_tensor bias_router[] = {
        {"exp_probs_b", "f32", "source_f32", "ds4_router_f32_i32"},
    };
    static const expected_global_tensor globals[] = {
        {"output.weight", "bf16", "source_bf16", "v100_bf16_output_pending", 7, -1},
    };

    const int ratio = layer_ratio(layer);
    const int gpu = layer_device(layer, gpus);
    int failures = 0;
    uint64_t expected = 0;
    uint64_t bytes = 0;
    printf("descriptor_gate_begin\tlayer=%d\tratio=%d\tgpu=%d\tindex=%s\n",
           layer,
           ratio,
           gpu,
           index_path);

#define REQUIRE_LIST(list) do { \
    for (uint32_t i = 0; i < sizeof(list) / sizeof((list)[0]); i++) { \
        failures += require_layer_tensor(pack, layer, gpu, &(list)[i], &bytes); \
        expected++; \
    } \
} while (0)

    REQUIRE_LIST(common);
    if (ratio != 0) REQUIRE_LIST(ratio_compress);
    if (ratio == 4) REQUIRE_LIST(indexer);
    if (layer <= 2) REQUIRE_LIST(hash_router);
    else REQUIRE_LIST(bias_router);

#undef REQUIRE_LIST

    for (uint32_t i = 0; i < sizeof(globals) / sizeof(globals[0]); i++) {
        failures += require_global_tensor(pack, &globals[i], &bytes);
        expected++;
    }

    printf("descriptor_summary\t%s\tlayer=%d\texpected=%" PRIu64 "\tfailures=%d\tbytes=%" PRIu64 "\n",
           failures ? "FAIL" : "PASS",
           layer,
           expected,
           failures,
           bytes);
    ds4_pack_close(pack);
    return failures ? 1 : 0;
}
