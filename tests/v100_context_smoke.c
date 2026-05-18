#include "ds4_v100_context.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void die(const char *msg) {
    fprintf(stderr, "v100_context_smoke: %s\n", msg);
    exit(1);
}

static void require_true(int cond, const char *msg) {
    if (!cond) die(msg);
}

static void write_file(const char *path, const char *body) {
    FILE *fp = fopen(path, "wb");
    if (!fp) die("cannot create temp pack index");
    if (fputs(body, fp) < 0) die("cannot write temp pack index");
    fclose(fp);
}

static void fill_good_facts(ds4_v100_device_fact facts[DS4_V100_EXPECTED_GPUS]) {
    memset(facts, 0, sizeof(ds4_v100_device_fact) * DS4_V100_EXPECTED_GPUS);
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        facts[i].visible_id = i;
        facts[i].cc_major = 7;
        facts[i].cc_minor = 0;
        facts[i].total_global_mem = 34ull * 1000ull * 1000ull * 1000ull;
        for (int j = 0; j < DS4_V100_EXPECTED_GPUS; j++) facts[i].peer_access[j] = true;
    }
}

static void test_classification(void) {
    char err[256];
    ds4_v100_policy p;
    require_true(ds4_v100_classify_or_die("bf16", "source_bf16",
                                          "ds4_embedding_bf16", &p,
                                          err, sizeof(err)) == 0,
                 "bf16 embedding should classify");
    require_true(p.exec_kind == DS4_V100_EXEC_DIAGNOSTIC_ONLY,
                 "bf16 embedding must be diagnostic only");

    require_true(ds4_v100_classify_or_die("f8_e4m3_b128",
                                          "source_f8_e4m3_b128_blocked",
                                          "v100_fp8_dequant_f16_hmma_pending",
                                          &p, err, sizeof(err)) == 0,
                 "fp8 dense should classify");
    require_true(p.exec_kind == DS4_V100_EXEC_F16_HMMA &&
                 p.conversion_stub != NULL,
                 "fp8 dense must require conversion stub to f16 hmma");

    require_true(ds4_v100_classify_or_die("mxfp4", "source_mxfp4_grouped",
                                          "v100_grouped_mxfp4_pending",
                                          &p, err, sizeof(err)) == 0,
                 "mxfp4 expert should classify");
    require_true(p.exec_kind == DS4_V100_EXEC_LOWBIT_KERNEL,
                 "mxfp4 expert must be lowbit kernel");

    require_true(ds4_v100_classify_or_die("bf16", "source_bf16",
                                          "v100_fp8_dequant_f16_hmma_pending",
                                          &p, err, sizeof(err)) != 0,
                 "bf16 must not classify as fp8 dense");
}

static void test_layer_map(void) {
    require_true(ds4_v100_stage_for_layer(-1) == -1, "negative layer invalid");
    require_true(ds4_v100_stage_for_layer(0) == 0, "layer 0 gpu0");
    require_true(ds4_v100_stage_for_layer(5) == 0, "layer 5 gpu0");
    require_true(ds4_v100_stage_for_layer(6) == 1, "layer 6 gpu1");
    require_true(ds4_v100_stage_for_layer(34) == 5, "layer 34 gpu5");
    require_true(ds4_v100_stage_for_layer(42) == 7, "layer 42 gpu7");
    require_true(ds4_v100_stage_for_layer(43) == -1, "layer 43 invalid");
}

static void test_topology_fail_closed(void) {
    char err[256];
    ds4_v100_device_fact facts[DS4_V100_EXPECTED_GPUS];
    fill_good_facts(facts);

    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    opts.require_production_topology = true;
    opts.device_facts = facts;
    opts.n_device_facts = DS4_V100_EXPECTED_GPUS - 1;
    ds4_v100_context *ctx = NULL;
    require_true(ds4_v100_context_open(&ctx, &opts, err, sizeof(err)) != 0,
                 "production mode must require eight facts");

    opts.n_device_facts = DS4_V100_EXPECTED_GPUS;
    facts[3].cc_major = 8;
    require_true(ds4_v100_context_open(&ctx, &opts, err, sizeof(err)) != 0,
                 "production mode must reject non-v100 cc");

    facts[3].cc_major = 7;
    facts[4].peer_access[5] = false;
    require_true(ds4_v100_context_open(&ctx, &opts, err, sizeof(err)) != 0,
                 "production mode must reject missing peer edge");
}

static char *make_pack_path(void) {
    char tmpl[] = "/tmp/ds4-v100-pack-XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) die("mkstemp failed");
    close(fd);
    char *out = (char *)malloc(strlen(tmpl) + 1);
    if (!out) die("malloc path failed");
    strcpy(out, tmpl);
    return out;
}

static const char *pack_header =
    "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\t"
    "runtime_layout\towning_gpu\tlayer_id\tkernel_family\tsource_offset\t"
    "byte_length\tshard_file\tshard_offset\tscale_offset\tchecksum\n";

static void test_pack_binding(void) {
    char *path = make_pack_path();
    char body[4096];
    snprintf(body, sizeof(body),
             "%s"
             "token_embd.weight\ttoken_embd.weight\tbf16\t[2x4]\tsource_bf16\t0\t-1\tds4_embedding_bf16\t0\t16\tgpu0.weights\t0\t-1\tpending\n"
             "blk.0.attn_norm.weight\tblk.0.attn_norm.weight\tf32\t[4]\tsource_f32_control\t0\t0\tds4_attention_control\t16\t16\tgpu0.weights\t16\t-1\tpending\n"
             "blk.0.attn_q.weight\tblk.0.attn_q.weight\tf8_e4m3_b128\t[128]\tsource_f8_e4m3_b128_blocked\t0\t0\tv100_fp8_dequant_f16_hmma_pending\t32\t129\tgpu0.weights\t32\t-1\tpending\n"
             "blk.0.ffn_exps.weight\tblk.0.ffn_exps.weight\tmxfp4\t[32]\tsource_mxfp4_grouped\t0\t0\tv100_grouped_mxfp4_pending\t161\t17\tgpu0.weights\t161\t-1\tpending\n"
             "blk.0.hc_attn_fn\tblk.0.hc_attn_fn\tf32\t[4x4]\tsource_f32\t0\t0\tds4_hc_control_f32\t178\t64\tgpu0.weights\t178\t-1\tpending\n",
             pack_header);
    write_file(path, body);

    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    opts.pack_index_path = path;
    opts.relay_max_active_slots = 2;
    opts.scratch_bytes_per_gpu = 1024;
    opts.enable_f32_debug_relay = true;

    char err[256];
    ds4_v100_context *ctx = NULL;
    require_true(ds4_v100_context_open(&ctx, &opts, err, sizeof(err)) == 0,
                 "context should open from synthetic pack");
    require_true(ds4_v100_context_tensor_count(ctx) == 5, "descriptor count");
    require_true(ds4_v100_context_has_token_embedding(ctx), "token embedding descriptor");
    require_true(ds4_v100_context_exec_count(ctx, DS4_V100_EXEC_DIAGNOSTIC_ONLY) == 1,
                 "diagnostic exec count");
    require_true(ds4_v100_context_exec_count(ctx, DS4_V100_EXEC_F32_CONTROL) == 2,
                 "f32 control exec count");
    require_true(ds4_v100_context_exec_count(ctx, DS4_V100_EXEC_F16_HMMA) == 1,
                 "f16 hmma exec count");
    require_true(ds4_v100_context_exec_count(ctx, DS4_V100_EXEC_LOWBIT_KERNEL) == 1,
                 "lowbit exec count");
    const ds4_v100_stage_info *s0 = ds4_v100_context_stage(ctx, 0);
    require_true(s0 && s0->relay_f16_bytes == 2ull * 2ull * 4ull * 4096ull * 2ull,
                 "stage 0 relay f16 bytes");
    require_true(s0->relay_f32_debug_bytes == 2ull * 2ull * 4ull * 4096ull * 4ull,
                 "stage 0 relay f32 bytes");
    ds4_v100_context_close(ctx);

    unlink(path);
    free(path);
}

static void test_bad_pack_binding(void) {
    char *path = make_pack_path();
    char body[2048];
    snprintf(body, sizeof(body),
             "%s"
             "blk.0.attn_q.weight\tblk.0.attn_q.weight\tf8_e4m3_b128\t[128]\tsource_f8_e4m3_b128_blocked\t0\t0\tv100_fp8_dequant_f16_hmma_pending\t0\t128\tgpu0.weights\t0\t-1\tpending\n",
             pack_header);
    write_file(path, body);

    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    opts.pack_index_path = path;
    char err[256];
    ds4_v100_context *ctx = NULL;
    require_true(ds4_v100_context_open(&ctx, &opts, err, sizeof(err)) != 0,
                 "bad f8 byte length must fail");
    unlink(path);
    free(path);
}

static void test_memory_reserve_fail_closed(void) {
    ds4_v100_device_fact facts[DS4_V100_EXPECTED_GPUS];
    fill_good_facts(facts);
    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    opts.require_production_topology = true;
    opts.device_facts = facts;
    opts.n_device_facts = DS4_V100_EXPECTED_GPUS;
    opts.scratch_bytes_per_gpu = facts[0].total_global_mem;
    char err[256];
    ds4_v100_context *ctx = NULL;
    require_true(ds4_v100_context_open(&ctx, &opts, err, sizeof(err)) != 0,
                 "memory reserve failure must fail closed");
}

int main(void) {
    test_classification();
    test_layer_map();
    test_topology_fail_closed();
    test_pack_binding();
    test_bad_pack_binding();
    test_memory_reserve_fail_closed();
    printf("v100_context_smoke: ok\n");
    return 0;
}
