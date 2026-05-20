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

static void require_kv_arena_plan(const ds4_v100_stage_info *s, const char *label) {
    char msg[128];
    uint64_t off = 0;

    snprintf(msg, sizeof(msg), "%s raw kv offset", label);
    require_true(s->kv_arena.raw_swa_offset == off, msg);
    snprintf(msg, sizeof(msg), "%s raw kv arena bytes", label);
    require_true(s->kv_arena.raw_swa_bytes == s->kv_raw_swa_bytes, msg);
    off += s->kv_raw_swa_bytes;

    snprintf(msg, sizeof(msg), "%s compressed kv offset", label);
    require_true(s->kv_arena.compressed_attn_offset == off, msg);
    snprintf(msg, sizeof(msg), "%s compressed kv arena bytes", label);
    require_true(s->kv_arena.compressed_attn_bytes == s->kv_compressed_attn_bytes, msg);
    off += s->kv_compressed_attn_bytes;

    snprintf(msg, sizeof(msg), "%s indexer kv offset", label);
    require_true(s->kv_arena.indexer_kv_offset == off, msg);
    snprintf(msg, sizeof(msg), "%s indexer kv arena bytes", label);
    require_true(s->kv_arena.indexer_kv_bytes == s->kv_indexer_bytes, msg);
    off += s->kv_indexer_bytes;

    snprintf(msg, sizeof(msg), "%s compression state kv offset", label);
    require_true(s->kv_arena.compression_state_offset == off, msg);
    snprintf(msg, sizeof(msg), "%s compression state arena bytes", label);
    require_true(s->kv_arena.compression_state_bytes == s->kv_compression_state_bytes, msg);
    off += s->kv_compression_state_bytes;

    snprintf(msg, sizeof(msg), "%s kv arena total", label);
    require_true(s->kv_arena.total_bytes == off, msg);
    snprintf(msg, sizeof(msg), "%s planned kv total", label);
    require_true(s->planned_kv_bytes == s->kv_arena.total_bytes, msg);
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
    require_true(p.conversion_stub &&
                 !strcmp(p.conversion_stub, "bf16_source_to_fp16_or_f32_boundary"),
                 "bf16 must declare conversion boundary");
    require_true(p.forbidden_claim &&
                 !strcmp(p.forbidden_claim, "native_bf16_tensor_core_execution"),
                 "bf16 must forbid native v100 bf16 claim");

    require_true(ds4_v100_classify_or_die("bf16", "source_bf16",
                                          "native_bf16_tensor_core_execution",
                                          &p, err, sizeof(err)) == 0,
                 "native bf16 claim should remain diagnostic policy");
    require_true(p.exec_kind == DS4_V100_EXEC_DIAGNOSTIC_ONLY &&
                 p.forbidden_claim &&
                 !strcmp(p.forbidden_claim, "native_bf16_tensor_core_execution"),
                 "native bf16 claim must not become executable");

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
    require_true(ds4_v100_classify_or_die("f32", "source_f32",
                                          "v100_fp32_gemm",
                                          &p, err, sizeof(err)) != 0,
                 "f32 model gemm fallback must fail policy");
    require_true(ds4_v100_classify_or_die("f32", "source_f32_matmul",
                                          "ds4_control",
                                          &p, err, sizeof(err)) != 0,
                 "f32 matmul layout must fail policy");
}

static void test_layer_map(void) {
    require_true(ds4_v100_stage_for_layer(-1) == -1, "negative layer invalid");
    require_true(ds4_v100_stage_for_layer(0) == 0, "layer 0 gpu0");
    require_true(ds4_v100_stage_for_layer(5) == 0, "layer 5 gpu0");
    require_true(ds4_v100_stage_for_layer(6) == 1, "layer 6 gpu1");
    require_true(ds4_v100_stage_for_layer(34) == 5, "layer 34 gpu5");
    require_true(ds4_v100_stage_for_layer(42) == 7, "layer 42 gpu7");
    require_true(ds4_v100_stage_for_layer(43) == -1, "layer 43 invalid");
    require_true(ds4_v100_layer_class_for_layer(0) == DS4_V100_LAYER_SWA_ONLY,
                 "layer 0 swa-only class");
    require_true(ds4_v100_layer_class_for_layer(1) == DS4_V100_LAYER_SWA_ONLY,
                 "layer 1 swa-only class");
    require_true(ds4_v100_layer_class_for_layer(2) == DS4_V100_LAYER_RATIO_4,
                 "layer 2 ratio-4 class");
    require_true(ds4_v100_layer_class_for_layer(3) == DS4_V100_LAYER_RATIO_128,
                 "layer 3 ratio-128 class");
    require_true(!strcmp(ds4_v100_layer_class_name(DS4_V100_LAYER_RATIO_4), "ratio_4"),
                 "ratio-4 class name");
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

static const char *tm_pack_header =
    "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\t"
    "runtime_layout\towning_gpu\tlayer_id\tkernel_family\t"
    "n\tk\texperts_packed\texperts_total\tweight_bytes_per_expert\t"
    "scale_bytes_per_expert\tk_pack\tweight_stride\tscale_stride\t"
    "sidecar_file\tweight_offset\tscale_offset\tsource_shard_file\t"
    "source_shard_offset\tsource_byte_length\tsource_checksum\t"
    "tm_abi_version\n";

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
    const ds4_v100_layer_info *l0 = ds4_v100_context_layer(ctx, 0);
    require_true(l0 && l0->stage_id == 0, "layer 0 info");
    require_true(l0->has_f32_control, "layer 0 f32 control family");
    require_true(l0->has_fp8_dense, "layer 0 fp8 dense family");
    require_true(l0->has_mxfp4_expert, "layer 0 mxfp4 family");
    require_true(l0->has_hc_control, "layer 0 hc family");
    ds4_v100_context_close(ctx);

    unlink(path);
    free(path);
}

static void test_turbomind_pack_binding(void) {
    char *pack_path = make_pack_path();
    char *tm_path = make_pack_path();
    char pack_body[4096];
    snprintf(pack_body, sizeof(pack_body),
             "%s"
             "token_embd.weight\ttoken_embd.weight\tbf16\t[2x4]\tsource_bf16\t0\t-1\tds4_embedding_bf16\t0\t16\tgpu0.weights\t0\t-1\tpending\n"
             "blk.0.attn_norm.weight\tblk.0.attn_norm.weight\tf32\t[4]\tsource_f32_control\t0\t0\tds4_attention_control\t16\t16\tgpu0.weights\t16\t-1\tpending\n"
             "blk.0.attn_q.weight\tblk.0.attn_q.weight\tf8_e4m3_b128\t[128]\tsource_f8_e4m3_b128_blocked\t0\t0\tv100_fp8_dequant_f16_hmma_pending\t32\t129\tgpu0.weights\t32\t-1\tpending\n"
             "blk.0.hc_attn_fn\tblk.0.hc_attn_fn\tf32\t[4x4]\tsource_f32\t0\t0\tds4_hc_control_f32\t178\t64\tgpu0.weights\t178\t-1\tpending\n",
             pack_header);
    write_file(pack_path, pack_body);

    char tm_body[2048];
    snprintf(tm_body, sizeof(tm_body),
             "%s"
             "blk.0.ffn_gate_exps.weight\tblk.0.ffn_gate_exps.weight\tmxfp4\t[4096x2048x256]\tturbomind_mxfp4_grouped\t0\t0\tturbomind_mxfp4_grouped_sm70\t2048\t4096\t256\t256\t1024\t256\t3413217\t131072\t2048\tgpu0.weights\t100000\t200000\tgpu0.weights\t1000\t178257920\tpending\t1\n",
             tm_pack_header);
    write_file(tm_path, tm_body);

    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    opts.pack_index_path = pack_path;
    opts.turbomind_pack_index_path = tm_path;

    char err[256];
    ds4_v100_context *ctx = NULL;
    require_true(ds4_v100_context_open(&ctx, &opts, err, sizeof(err)) == 0,
                 "context should open with TurboMind pack index");
    require_true(ds4_v100_context_tensor_count(ctx) == 5,
                 "TurboMind descriptor contributes to tensor count");
    require_true(ds4_v100_context_exec_count(ctx, DS4_V100_EXEC_LOWBIT_KERNEL) == 1,
                 "TurboMind descriptor contributes lowbit exec count");
    const ds4_v100_layer_info *l0 = ds4_v100_context_layer(ctx, 0);
    require_true(l0 && l0->has_mxfp4_expert,
                 "TurboMind descriptor marks layer mxfp4 expert present");
    const ds4_v100_stage_info *s0 = ds4_v100_context_stage(ctx, 0);
    require_true(s0 && s0->arena_bytes >= 265536,
                 "TurboMind descriptor extends stage arena bytes");
    ds4_v100_turbomind_binding tm;
    require_true(ds4_v100_context_require_layer_turbomind_binding(
                     ctx, 0, "ffn_gate_exps.weight", &tm, err, sizeof(err)) == 0,
                 "TurboMind layer binding lookup");
    require_true(tm.k == 4096 && tm.n == 2048 && tm.experts_total == 256,
                 "TurboMind binding dimensions");
    require_true(!strcmp(tm.shard_file, "gpu0.weights"),
                 "TurboMind binding points into appliance shard");
    ds4_v100_context_close(ctx);

    unlink(tm_path);
    unlink(pack_path);
    free(tm_path);
    free(pack_path);
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

static void test_kv_budget_math(void) {
    const uint64_t ctx = 1048576ull;
    const uint64_t raw = 128ull * 512ull * 2ull;
    const uint64_t ratio4_comp = (ctx / 4ull) * 512ull * 2ull;
    const uint64_t ratio4_indexer = (ctx / 4ull) * 128ull * 2ull;
    const uint64_t ratio4_state =
        2ull * (2ull * 512ull) * (2ull * 4ull) * 4ull +
        2ull * (2ull * 128ull) * (2ull * 4ull) * 4ull;
    const uint64_t ratio128_comp = (ctx / 128ull) * 512ull * 2ull;
    const uint64_t ratio128_state = 2ull * 512ull * 128ull * 4ull;

    ds4_v100_kv_budget b = ds4_v100_kv_budget_for_layer(0, ctx, 1);
    require_true(b.raw_swa_bytes == raw, "swa raw bytes");
    require_true(b.compressed_attn_bytes == 0, "swa compressed bytes");
    require_true(b.indexer_kv_bytes == 0, "swa indexer bytes");
    require_true(b.compression_state_bytes == 0, "swa compression state bytes");
    require_true(b.total_bytes == raw, "swa total bytes");

    b = ds4_v100_kv_budget_for_layer(2, ctx, 1);
    require_true(b.raw_swa_bytes == raw, "ratio4 raw bytes");
    require_true(b.compressed_attn_bytes == ratio4_comp, "ratio4 compressed bytes");
    require_true(b.indexer_kv_bytes == ratio4_indexer, "ratio4 indexer bytes");
    require_true(b.compression_state_bytes == ratio4_state, "ratio4 compression state bytes");
    require_true(b.total_bytes == raw + ratio4_comp + ratio4_indexer + ratio4_state,
                 "ratio4 total bytes");

    b = ds4_v100_kv_budget_for_layer(3, ctx, 2);
    require_true(b.raw_swa_bytes == raw * 2ull, "ratio128 raw bytes slots");
    require_true(b.compressed_attn_bytes == ratio128_comp * 2ull,
                 "ratio128 compressed bytes slots");
    require_true(b.indexer_kv_bytes == 0, "ratio128 indexer bytes");
    require_true(b.compression_state_bytes == ratio128_state,
                 "ratio128 compression state bytes");
}

static void test_kv_stage_admission(void) {
    const uint64_t ctx_tokens = 1048576ull;
    const uint64_t raw = 128ull * 512ull * 2ull;
    const uint64_t ratio4_comp = (ctx_tokens / 4ull) * 512ull * 2ull;
    const uint64_t ratio4_indexer = (ctx_tokens / 4ull) * 128ull * 2ull;
    const uint64_t ratio4_attn_state = (2ull * 512ull) * (2ull * 4ull) * 4ull;
    const uint64_t ratio4_index_state = (2ull * 128ull) * (2ull * 4ull) * 4ull;
    const uint64_t ratio4_state =
        2ull * ratio4_attn_state + 2ull * ratio4_index_state;
    const uint64_t ratio128_comp = (ctx_tokens / 128ull) * 512ull * 2ull;
    const uint64_t ratio128_attn_state = 512ull * 128ull * 4ull;
    const uint64_t ratio128_state = 2ull * ratio128_attn_state;

    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    opts.kv_ctx_tokens = ctx_tokens;
    opts.kv_active_slots = 1;

    char err[256];
    ds4_v100_context *ctx = NULL;
    require_true(ds4_v100_context_open(&ctx, &opts, err, sizeof(err)) == 0,
                 "context should open with derived kv plan");

    int counts[3] = {0, 0, 0};
    for (int layer = 0; layer < DS4_V100_N_LAYERS; layer++) {
        const ds4_v100_layer_info *li = ds4_v100_context_layer(ctx, layer);
        require_true(li != NULL, "layer info for kv count");
        counts[li->layer_class]++;
    }
    require_true(counts[DS4_V100_LAYER_SWA_ONLY] == 2, "swa-only layer count");
    require_true(counts[DS4_V100_LAYER_RATIO_4] == 21, "ratio4 layer count");
    require_true(counts[DS4_V100_LAYER_RATIO_128] == 20, "ratio128 layer count");

    const ds4_v100_stage_info *s0 = ds4_v100_context_stage(ctx, 0);
    require_true(s0 != NULL, "stage 0 kv info");
    require_true(s0->kv_raw_swa_bytes == 6ull * raw, "stage0 raw kv bytes");
    require_true(s0->kv_compressed_attn_bytes ==
                 2ull * ratio4_comp + 2ull * ratio128_comp,
                 "stage0 compressed kv bytes");
    require_true(s0->kv_indexer_bytes == 2ull * ratio4_indexer,
                 "stage0 indexer kv bytes");
    require_true(s0->kv_compression_state_bytes ==
                 2ull * ratio4_state + 2ull * ratio128_state,
                 "stage0 compression state bytes");
    require_kv_arena_plan(s0, "stage0");

    const ds4_v100_layer_info *l0 = ds4_v100_context_layer(ctx, 0);
    const ds4_v100_layer_info *l2 = ds4_v100_context_layer(ctx, 2);
    const ds4_v100_layer_info *l3 = ds4_v100_context_layer(ctx, 3);
    require_true(l0 && l2 && l3, "layer kv views exist");
    require_true(l0->kv_view.raw_swa_offset == 0, "layer0 raw view offset");
    require_true(l0->kv_view.raw_swa_bytes == raw, "layer0 raw view bytes");
    require_true(l2->kv_view.raw_swa_offset == 2ull * raw, "layer2 raw view offset");
    require_true(l2->kv_view.compressed_attn_offset ==
                 s0->kv_arena.compressed_attn_offset,
                 "layer2 compressed view offset");
    require_true(l2->kv_view.compressed_attn_bytes == ratio4_comp,
                 "layer2 compressed view bytes");
    require_true(l2->kv_view.indexer_kv_offset == s0->kv_arena.indexer_kv_offset,
                 "layer2 indexer view offset");
    require_true(l2->kv_view.indexer_kv_bytes == ratio4_indexer,
                 "layer2 indexer view bytes");
    require_true(l2->kv_view.attn_state_kv_offset ==
                 s0->kv_arena.compression_state_offset,
                 "layer2 attn state kv offset");
    require_true(l2->kv_view.attn_state_kv_bytes == ratio4_attn_state,
                 "layer2 attn state kv bytes");
    require_true(l2->kv_view.attn_state_score_offset ==
                 s0->kv_arena.compression_state_offset + ratio4_attn_state,
                 "layer2 attn state score offset");
    require_true(l2->kv_view.indexer_state_kv_offset ==
                 s0->kv_arena.compression_state_offset + 2ull * ratio4_attn_state,
                 "layer2 indexer state kv offset");
    require_true(l2->kv_view.indexer_state_kv_bytes == ratio4_index_state,
                 "layer2 indexer state kv bytes");
    require_true(l2->kv_view.indexer_state_score_offset ==
                 s0->kv_arena.compression_state_offset + 2ull * ratio4_attn_state +
                 ratio4_index_state,
                 "layer2 indexer state score offset");
    require_true(l2->kv_view.total_bytes == l2->kv_budget.total_bytes,
                 "layer2 view total");
    require_true(l3->kv_view.compressed_attn_offset ==
                 s0->kv_arena.compressed_attn_offset + ratio4_comp,
                 "layer3 compressed view offset");
    require_true(l3->kv_view.attn_state_kv_offset ==
                 s0->kv_arena.compression_state_offset + ratio4_state,
                 "layer3 attn state kv offset");
    require_true(l3->kv_view.attn_state_kv_bytes == ratio128_attn_state,
                 "layer3 attn state kv bytes");
    require_true(l3->kv_view.indexer_kv_bytes == 0 &&
                 l3->kv_view.indexer_state_kv_bytes == 0,
                 "layer3 has no indexer views");

    const ds4_v100_stage_info *s1 = ds4_v100_context_stage(ctx, 1);
    require_true(s1 != NULL, "stage 1 kv info");
    require_true(s1->kv_raw_swa_bytes == 6ull * raw, "stage1 raw kv bytes");
    require_true(s1->kv_compressed_attn_bytes ==
                 3ull * ratio4_comp + 3ull * ratio128_comp,
                 "stage1 compressed kv bytes");
    require_true(s1->kv_indexer_bytes == 3ull * ratio4_indexer,
                 "stage1 indexer kv bytes");
    require_true(s1->kv_compression_state_bytes ==
                 3ull * ratio4_state + 3ull * ratio128_state,
                 "stage1 compression state bytes");
    require_kv_arena_plan(s1, "stage1");
    ds4_v100_context_close(ctx);
}

static void test_kv_context_tiers(void) {
    const uint64_t tiers[] = { 131072ull, 262144ull, 524288ull, 1048576ull };
    uint64_t prev_stage0 = 0;

    for (size_t i = 0; i < sizeof(tiers) / sizeof(tiers[0]); i++) {
        ds4_v100_context_options opts;
        ds4_v100_context_options_init(&opts);
        opts.kv_ctx_tokens = tiers[i];
        opts.kv_active_slots = 1;

        char err[256];
        ds4_v100_context *ctx = NULL;
        require_true(ds4_v100_context_open(&ctx, &opts, err, sizeof(err)) == 0,
                     "context tier should open with derived kv plan");
        const ds4_v100_stage_info *s0 = ds4_v100_context_stage(ctx, 0);
        require_true(s0 != NULL, "stage 0 tier kv info");
        require_true(s0->planned_kv_bytes > prev_stage0,
                     "stage0 kv budget must increase with context tier");
        prev_stage0 = s0->planned_kv_bytes;
        ds4_v100_context_close(ctx);
    }
}

static void test_kv_reserve_fail_closed(void) {
    ds4_v100_device_fact facts[DS4_V100_EXPECTED_GPUS];
    fill_good_facts(facts);
    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    opts.require_production_topology = true;
    opts.device_facts = facts;
    opts.n_device_facts = DS4_V100_EXPECTED_GPUS;
    opts.kv_ctx_tokens = 1048576ull;
    opts.kv_active_slots = 64;
    char err[256];
    ds4_v100_context *ctx = NULL;
    require_true(ds4_v100_context_open(&ctx, &opts, err, sizeof(err)) != 0,
                 "oversized derived kv plan must fail closed");
}

static void test_kv_plan_mode_fail_closed(void) {
    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    opts.kv_ctx_tokens = 262144ull;
    opts.planned_kv_bytes_per_gpu = 512ull * 1024ull * 1024ull;

    char err[256];
    ds4_v100_context *ctx = NULL;
    require_true(ds4_v100_context_open(&ctx, &opts, err, sizeof(err)) != 0,
                 "coarse and derived kv plans must not be combined");
}

int main(void) {
    test_classification();
    test_layer_map();
    test_topology_fail_closed();
    test_pack_binding();
    test_turbomind_pack_binding();
    test_bad_pack_binding();
    test_memory_reserve_fail_closed();
    test_kv_budget_math();
    test_kv_stage_admission();
    test_kv_context_tiers();
    test_kv_reserve_fail_closed();
    test_kv_plan_mode_fail_closed();
    printf("v100_context_smoke: ok\n");
    return 0;
}
