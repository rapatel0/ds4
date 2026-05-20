#include "ds4_v100_context.h"
#include "ds4_pack.h"
#include "ds4_turbomind_pack.h"

#include <ctype.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const ds4_pack_entry *entry;
    ds4_v100_policy policy;
} ds4_v100_tensor_desc;

typedef struct {
    const ds4_tm_pack_entry *entry;
    ds4_v100_policy policy;
} ds4_v100_tm_tensor_desc;

struct ds4_v100_context {
    ds4_v100_context_options opts;
    ds4_pack *pack;
    ds4_tm_pack *tm_pack;
    ds4_v100_stage_info stages[DS4_V100_EXPECTED_GPUS];
    ds4_v100_tensor_desc *descs;
    ds4_v100_tm_tensor_desc *tm_descs;
    uint64_t n_descs;
    uint64_t cap_descs;
    uint64_t n_tm_descs;
    uint64_t cap_tm_descs;
    uint64_t exec_counts[DS4_V100_EXEC_COUNT];
    bool has_token_embedding;
    ds4_v100_layer_info layers[DS4_V100_N_LAYERS];
};

typedef struct {
    ds4_v100_context *ctx;
    char *err;
    size_t errlen;
} ds4_v100_bind_state;

static int v100_error(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return 1;
}

static bool str_eq_ci(const char *a, const char *b) {
    if (!a || !b) return false;
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return false;
        a++;
        b++;
    }
    return *a == '\0' && *b == '\0';
}

static bool contains_ci(const char *s, const char *needle) {
    if (!s || !needle || !*needle) return false;
    size_t n = strlen(needle);
    for (const char *p = s; *p; p++) {
        size_t i = 0;
        while (i < n && p[i] &&
               tolower((unsigned char)p[i]) == tolower((unsigned char)needle[i])) {
            i++;
        }
        if (i == n) return true;
    }
    return false;
}

void ds4_v100_context_options_init(ds4_v100_context_options *opts) {
    if (!opts) return;
    memset(opts, 0, sizeof(*opts));
    opts->expected_gpus = DS4_V100_EXPECTED_GPUS;
    opts->mode = DS4_V100_INIT_PROBE_ONLY;
    opts->relay_max_active_slots = 1;
    opts->reserve_bytes_per_gpu = DS4_V100_DEFAULT_RESERVE_BYTES;
    opts->kv_active_slots = 1;
}

ds4_v100_source_dtype ds4_v100_source_dtype_parse(const char *s) {
    if (str_eq_ci(s, "bf16")) return DS4_V100_SOURCE_BF16;
    if (str_eq_ci(s, "f32")) return DS4_V100_SOURCE_F32;
    if (str_eq_ci(s, "i32")) return DS4_V100_SOURCE_I32;
    if (str_eq_ci(s, "f8_e4m3_b128")) return DS4_V100_SOURCE_F8_E4M3_B128;
    if (str_eq_ci(s, "mxfp4")) return DS4_V100_SOURCE_MXFP4;
    if (str_eq_ci(s, "fp4")) return DS4_V100_SOURCE_FP4;
    return DS4_V100_SOURCE_UNKNOWN;
}

const char *ds4_v100_source_dtype_name(ds4_v100_source_dtype dtype) {
    switch (dtype) {
    case DS4_V100_SOURCE_BF16: return "bf16";
    case DS4_V100_SOURCE_F32: return "f32";
    case DS4_V100_SOURCE_I32: return "i32";
    case DS4_V100_SOURCE_F8_E4M3_B128: return "f8_e4m3_b128";
    case DS4_V100_SOURCE_MXFP4: return "mxfp4";
    case DS4_V100_SOURCE_FP4: return "fp4";
    case DS4_V100_SOURCE_UNKNOWN:
    default: return "unknown";
    }
}

ds4_v100_tensor_family ds4_v100_tensor_family_infer(const char *source_dtype,
                                                    const char *runtime_layout,
                                                    const char *kernel_family) {
    ds4_v100_source_dtype dtype = ds4_v100_source_dtype_parse(source_dtype);
    if (contains_ci(kernel_family, "hc_control")) return DS4_V100_FAMILY_HC_CONTROL;
    if (contains_ci(kernel_family, "kv")) return DS4_V100_FAMILY_KV_CACHE;
    if (dtype == DS4_V100_SOURCE_F8_E4M3_B128 ||
        contains_ci(kernel_family, "fp8") ||
        contains_ci(runtime_layout, "f8_e4m3_b128")) {
        return DS4_V100_FAMILY_FP8_DENSE;
    }
    if (dtype == DS4_V100_SOURCE_MXFP4 || dtype == DS4_V100_SOURCE_FP4 ||
        contains_ci(kernel_family, "mxfp4") ||
        contains_ci(runtime_layout, "mxfp4")) {
        return DS4_V100_FAMILY_MXFP4_EXPERT;
    }
    if (dtype == DS4_V100_SOURCE_BF16) return DS4_V100_FAMILY_BF16_GLOBAL;
    if (dtype == DS4_V100_SOURCE_F32 || dtype == DS4_V100_SOURCE_I32) return DS4_V100_FAMILY_F32_CONTROL;
    return DS4_V100_FAMILY_UNKNOWN;
}

const char *ds4_v100_tensor_family_name(ds4_v100_tensor_family family) {
    switch (family) {
    case DS4_V100_FAMILY_BF16_GLOBAL: return "bf16_global";
    case DS4_V100_FAMILY_F32_CONTROL: return "f32_control";
    case DS4_V100_FAMILY_FP8_DENSE: return "fp8_dense";
    case DS4_V100_FAMILY_MXFP4_EXPERT: return "mxfp4_expert";
    case DS4_V100_FAMILY_HC_CONTROL: return "hc_control";
    case DS4_V100_FAMILY_KV_CACHE: return "kv_cache";
    case DS4_V100_FAMILY_UNKNOWN:
    default: return "unknown";
    }
}

const char *ds4_v100_exec_kind_name(ds4_v100_exec_kind kind) {
    switch (kind) {
    case DS4_V100_EXEC_F32_CONTROL: return "f32_control";
    case DS4_V100_EXEC_F16_HMMA: return "f16_hmma_after_convert";
    case DS4_V100_EXEC_LOWBIT_KERNEL: return "lowbit_kernel";
    case DS4_V100_EXEC_DIAGNOSTIC_ONLY: return "diagnostic_only";
    case DS4_V100_EXEC_UNSUPPORTED: return "unsupported";
    case DS4_V100_EXEC_COUNT:
    default: return "invalid";
    }
}

ds4_v100_layer_class ds4_v100_layer_class_for_layer(int layer_id) {
    if (layer_id < 0 || layer_id >= DS4_V100_N_LAYERS) return DS4_V100_LAYER_SWA_ONLY;
    if (layer_id <= 1) return DS4_V100_LAYER_SWA_ONLY;
    return (layer_id % 2) == 0 ? DS4_V100_LAYER_RATIO_4 : DS4_V100_LAYER_RATIO_128;
}

const char *ds4_v100_layer_class_name(ds4_v100_layer_class layer_class) {
    switch (layer_class) {
    case DS4_V100_LAYER_SWA_ONLY: return "swa_only";
    case DS4_V100_LAYER_RATIO_4: return "ratio_4";
    case DS4_V100_LAYER_RATIO_128: return "ratio_128";
    default: return "unknown";
    }
}

static uint64_t sat_mul_u64(uint64_t a, uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a) return UINT64_MAX;
    return a * b;
}

static uint64_t sat_add_u64(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) return UINT64_MAX;
    return a + b;
}

static uint64_t align_up_u64(uint64_t v, uint64_t align) {
    if (align == 0) return v;
    uint64_t mask = align - 1u;
    if ((align & mask) != 0) return v;
    if (v > UINT64_MAX - mask) return UINT64_MAX;
    return (v + mask) & ~mask;
}

ds4_v100_kv_budget ds4_v100_kv_budget_for_layer(int layer_id,
                                                uint64_t ctx_tokens,
                                                uint64_t active_slots) {
    ds4_v100_kv_budget b;
    memset(&b, 0, sizeof(b));
    if (layer_id < 0 || layer_id >= DS4_V100_N_LAYERS || ctx_tokens == 0 || active_slots == 0) {
        return b;
    }

    const ds4_v100_layer_class layer_class = ds4_v100_layer_class_for_layer(layer_id);
    const uint64_t elem_bytes = 2;
    const uint64_t raw_per_slot =
        (uint64_t)DS4_V100_SWA_ROWS * DS4_V100_HEAD_DIM * elem_bytes;
    b.raw_swa_bytes = sat_mul_u64(raw_per_slot, active_slots);

    if (layer_class == DS4_V100_LAYER_RATIO_4) {
        const uint64_t comp_slots = ctx_tokens / 4;
        const uint64_t comp_per_slot = sat_mul_u64(comp_slots, DS4_V100_HEAD_DIM * elem_bytes);
        const uint64_t index_per_slot =
            sat_mul_u64(comp_slots, DS4_V100_INDEXER_HEAD_DIM * elem_bytes);
        b.compressed_attn_bytes = sat_mul_u64(comp_per_slot, active_slots);
        b.indexer_kv_bytes = sat_mul_u64(index_per_slot, active_slots);
        /* Mirrors llama_memory_deepseek4: two F32 state tensors per compressor,
         * shaped as ape_ne0 by comp_slots*ratio for the fixed DS4 compressor. */
        b.compression_state_bytes =
            2ull * (2ull * DS4_V100_HEAD_DIM) * (2ull * 4ull) * sizeof(float);
        b.compression_state_bytes = sat_add_u64(
            b.compression_state_bytes,
            2ull * (2ull * DS4_V100_INDEXER_HEAD_DIM) * (2ull * 4ull) * sizeof(float));
    } else if (layer_class == DS4_V100_LAYER_RATIO_128) {
        const uint64_t comp_slots = ctx_tokens / 128;
        const uint64_t comp_per_slot = sat_mul_u64(comp_slots, DS4_V100_HEAD_DIM * elem_bytes);
        b.compressed_attn_bytes = sat_mul_u64(comp_per_slot, active_slots);
        b.compression_state_bytes =
            2ull * (uint64_t)DS4_V100_HEAD_DIM * 128ull * sizeof(float);
    }

    b.total_bytes = sat_add_u64(b.raw_swa_bytes, b.compressed_attn_bytes);
    b.total_bytes = sat_add_u64(b.total_bytes, b.indexer_kv_bytes);
    b.total_bytes = sat_add_u64(b.total_bytes, b.compression_state_bytes);
    return b;
}

static void kv_state_split_for_layer(int layer_id,
                                     uint64_t *attn_kv,
                                     uint64_t *attn_score,
                                     uint64_t *indexer_kv,
                                     uint64_t *indexer_score) {
    if (attn_kv) *attn_kv = 0;
    if (attn_score) *attn_score = 0;
    if (indexer_kv) *indexer_kv = 0;
    if (indexer_score) *indexer_score = 0;

    const ds4_v100_layer_class layer_class = ds4_v100_layer_class_for_layer(layer_id);
    if (layer_class == DS4_V100_LAYER_RATIO_4) {
        const uint64_t attn =
            (2ull * DS4_V100_HEAD_DIM) * (2ull * 4ull) * sizeof(float);
        const uint64_t idx =
            (2ull * DS4_V100_INDEXER_HEAD_DIM) * (2ull * 4ull) * sizeof(float);
        if (attn_kv) *attn_kv = attn;
        if (attn_score) *attn_score = attn;
        if (indexer_kv) *indexer_kv = idx;
        if (indexer_score) *indexer_score = idx;
    } else if (layer_class == DS4_V100_LAYER_RATIO_128) {
        const uint64_t attn = (uint64_t)DS4_V100_HEAD_DIM * 128ull * sizeof(float);
        if (attn_kv) *attn_kv = attn;
        if (attn_score) *attn_score = attn;
    }
}

int ds4_v100_classify_or_die(const char *source_dtype,
                             const char *runtime_layout,
                             const char *kernel_family,
                             ds4_v100_policy *out,
                             char *err,
                             size_t errlen) {
    ds4_v100_source_dtype dtype = ds4_v100_source_dtype_parse(source_dtype);
    ds4_v100_tensor_family family =
        ds4_v100_tensor_family_infer(source_dtype, runtime_layout, kernel_family);

    ds4_v100_policy p;
    memset(&p, 0, sizeof(p));
    p.source_dtype = dtype;
    p.family = family;
    p.exec_kind = DS4_V100_EXEC_UNSUPPORTED;

    switch (family) {
    case DS4_V100_FAMILY_BF16_GLOBAL:
        if (dtype != DS4_V100_SOURCE_BF16) break;
        p.exec_kind = DS4_V100_EXEC_DIAGNOSTIC_ONLY;
        p.conversion_stub = "bf16_source_to_fp16_or_f32_boundary";
        p.forbidden_claim = "native_bf16_tensor_core_execution";
        break;
    case DS4_V100_FAMILY_HC_CONTROL:
    case DS4_V100_FAMILY_F32_CONTROL:
        if (dtype != DS4_V100_SOURCE_F32 && dtype != DS4_V100_SOURCE_I32) break;
        if (contains_ci(kernel_family, "gemm") ||
            contains_ci(kernel_family, "matmul") ||
            contains_ci(runtime_layout, "gemm") ||
            contains_ci(runtime_layout, "matmul")) {
            break;
        }
        p.exec_kind = DS4_V100_EXEC_F32_CONTROL;
        p.forbidden_claim = "decode_completion";
        break;
    case DS4_V100_FAMILY_FP8_DENSE:
        if (dtype != DS4_V100_SOURCE_F8_E4M3_B128) break;
        p.exec_kind = DS4_V100_EXEC_F16_HMMA;
        p.conversion_stub = "fp8_e4m3_b128_unpack_to_fp16_hmma_pending";
        p.forbidden_claim = "native_fp8_tensor_core_execution";
        break;
    case DS4_V100_FAMILY_MXFP4_EXPERT:
        if (dtype != DS4_V100_SOURCE_MXFP4 && dtype != DS4_V100_SOURCE_FP4) break;
        p.exec_kind = DS4_V100_EXEC_LOWBIT_KERNEL;
        p.conversion_stub = "mxfp4_or_fp4_lowbit_kernel_pending";
        p.forbidden_claim = "native_fp4_tensor_core_execution";
        break;
    case DS4_V100_FAMILY_KV_CACHE:
        p.exec_kind = DS4_V100_EXEC_UNSUPPORTED;
        p.forbidden_claim = "kv_population_in_sprint_006";
        break;
    case DS4_V100_FAMILY_UNKNOWN:
    default:
        break;
    }

    if (p.exec_kind == DS4_V100_EXEC_UNSUPPORTED) {
        return v100_error(err, errlen,
                          "unsupported V100 execution policy: dtype=%s layout=%s kernel=%s",
                          source_dtype ? source_dtype : "(null)",
                          runtime_layout ? runtime_layout : "(null)",
                          kernel_family ? kernel_family : "(null)");
    }
    if (out) *out = p;
    return 0;
}

int ds4_v100_stage_for_layer(int layer_id) {
    if (layer_id < 0 || layer_id >= DS4_V100_N_LAYERS) return -1;
    if (layer_id <= 5) return 0;
    if (layer_id <= 11) return 1;
    if (layer_id <= 17) return 2;
    if (layer_id <= 23) return 3;
    if (layer_id <= 29) return 4;
    if (layer_id <= 34) return 5;
    if (layer_id <= 39) return 6;
    return 7;
}

static void init_stage_map(ds4_v100_context *ctx) {
    static const int begins[DS4_V100_EXPECTED_GPUS] = {0, 6, 12, 18, 24, 30, 35, 40};
    static const int ends[DS4_V100_EXPECTED_GPUS] = {5, 11, 17, 23, 29, 34, 39, 42};
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        ds4_v100_stage_info *s = &ctx->stages[i];
        s->stage_id = i;
        s->gpu = i;
        s->layer_begin = begins[i];
        s->layer_end = ends[i];
        s->owns_token_embedding = i == 0;
        s->owns_output_head = i == 7;
        s->scratch_bytes = ctx->opts.scratch_bytes_per_gpu;
        s->planned_kv_bytes = ctx->opts.planned_kv_bytes_per_gpu;
        s->reserve_bytes = ctx->opts.reserve_bytes_per_gpu;
        s->relay_f16_bytes =
            DS4_V100_RELAY_BUFFERS * ctx->opts.relay_max_active_slots *
            DS4_V100_HC_ROWS * DS4_V100_HC_COLS * 2ull;
        s->relay_f32_debug_bytes = ctx->opts.enable_f32_debug_relay ?
            DS4_V100_RELAY_BUFFERS * ctx->opts.relay_max_active_slots *
            DS4_V100_HC_ROWS * DS4_V100_HC_COLS * 4ull : 0;
        if (i == 7) {
            s->output_head_reserve_bytes = ctx->opts.output_head_reserve_bytes;
            s->mtp_reserve_bytes = ctx->opts.mtp_reserve_bytes;
        }
    }
    for (int layer = 0; layer < DS4_V100_N_LAYERS; layer++) {
        ctx->layers[layer].layer_id = layer;
        ctx->layers[layer].stage_id = ds4_v100_stage_for_layer(layer);
        ctx->layers[layer].layer_class = ds4_v100_layer_class_for_layer(layer);
    }
}

static void apply_derived_kv_plan(ds4_v100_context *ctx) {
    if (!ctx || ctx->opts.kv_ctx_tokens == 0) return;
    const uint64_t slots = ctx->opts.kv_active_slots;
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        ctx->stages[i].planned_kv_bytes = 0;
        memset(&ctx->stages[i].kv_arena, 0, sizeof(ctx->stages[i].kv_arena));
        ctx->stages[i].kv_raw_swa_bytes = 0;
        ctx->stages[i].kv_compressed_attn_bytes = 0;
        ctx->stages[i].kv_indexer_bytes = 0;
        ctx->stages[i].kv_compression_state_bytes = 0;
    }
    for (int layer = 0; layer < DS4_V100_N_LAYERS; layer++) {
        ds4_v100_layer_info *li = &ctx->layers[layer];
        li->kv_budget = ds4_v100_kv_budget_for_layer(layer, ctx->opts.kv_ctx_tokens, slots);
        ds4_v100_stage_info *stage = &ctx->stages[li->stage_id];
        stage->kv_raw_swa_bytes =
            sat_add_u64(stage->kv_raw_swa_bytes, li->kv_budget.raw_swa_bytes);
        stage->kv_compressed_attn_bytes =
            sat_add_u64(stage->kv_compressed_attn_bytes, li->kv_budget.compressed_attn_bytes);
        stage->kv_indexer_bytes =
            sat_add_u64(stage->kv_indexer_bytes, li->kv_budget.indexer_kv_bytes);
        stage->kv_compression_state_bytes =
            sat_add_u64(stage->kv_compression_state_bytes, li->kv_budget.compression_state_bytes);
        stage->planned_kv_bytes =
            sat_add_u64(stage->planned_kv_bytes, li->kv_budget.total_bytes);
    }
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        ds4_v100_stage_info *stage = &ctx->stages[i];
        ds4_v100_kv_arena_plan *arena = &stage->kv_arena;
        uint64_t off = 0;

        arena->raw_swa_offset = off;
        arena->raw_swa_bytes = stage->kv_raw_swa_bytes;
        off = align_up_u64(sat_add_u64(off, arena->raw_swa_bytes), 256ull);

        arena->compressed_attn_offset = off;
        arena->compressed_attn_bytes = stage->kv_compressed_attn_bytes;
        off = align_up_u64(sat_add_u64(off, arena->compressed_attn_bytes), 256ull);

        arena->indexer_kv_offset = off;
        arena->indexer_kv_bytes = stage->kv_indexer_bytes;
        off = align_up_u64(sat_add_u64(off, arena->indexer_kv_bytes), 256ull);

        arena->compression_state_offset = off;
        arena->compression_state_bytes = stage->kv_compression_state_bytes;
        off = align_up_u64(sat_add_u64(off, arena->compression_state_bytes), 256ull);

        arena->total_bytes = off;
        stage->planned_kv_bytes = arena->total_bytes;
    }
    uint64_t raw_cursor[DS4_V100_EXPECTED_GPUS] = {0};
    uint64_t comp_cursor[DS4_V100_EXPECTED_GPUS] = {0};
    uint64_t index_cursor[DS4_V100_EXPECTED_GPUS] = {0};
    uint64_t state_cursor[DS4_V100_EXPECTED_GPUS] = {0};

    for (int layer = 0; layer < DS4_V100_N_LAYERS; layer++) {
        ds4_v100_layer_info *li = &ctx->layers[layer];
        ds4_v100_stage_info *stage = &ctx->stages[li->stage_id];
        ds4_v100_layer_kv_view *view = &li->kv_view;
        memset(view, 0, sizeof(*view));

        view->raw_swa_offset = sat_add_u64(stage->kv_arena.raw_swa_offset,
                                           raw_cursor[li->stage_id]);
        view->raw_swa_bytes = li->kv_budget.raw_swa_bytes;
        raw_cursor[li->stage_id] =
            sat_add_u64(raw_cursor[li->stage_id], view->raw_swa_bytes);

        view->compressed_attn_offset =
            sat_add_u64(stage->kv_arena.compressed_attn_offset,
                        comp_cursor[li->stage_id]);
        view->compressed_attn_bytes = li->kv_budget.compressed_attn_bytes;
        comp_cursor[li->stage_id] =
            sat_add_u64(comp_cursor[li->stage_id], view->compressed_attn_bytes);

        view->indexer_kv_offset =
            sat_add_u64(stage->kv_arena.indexer_kv_offset,
                        index_cursor[li->stage_id]);
        view->indexer_kv_bytes = li->kv_budget.indexer_kv_bytes;
        index_cursor[li->stage_id] =
            sat_add_u64(index_cursor[li->stage_id], view->indexer_kv_bytes);

        uint64_t attn_kv = 0;
        uint64_t attn_score = 0;
        uint64_t idx_kv = 0;
        uint64_t idx_score = 0;
        kv_state_split_for_layer(layer, &attn_kv, &attn_score, &idx_kv, &idx_score);
        const uint64_t state_base = stage->kv_arena.compression_state_offset;

        view->attn_state_kv_offset =
            sat_add_u64(state_base, state_cursor[li->stage_id]);
        view->attn_state_kv_bytes = attn_kv;
        state_cursor[li->stage_id] =
            sat_add_u64(state_cursor[li->stage_id], view->attn_state_kv_bytes);

        view->attn_state_score_offset =
            sat_add_u64(state_base, state_cursor[li->stage_id]);
        view->attn_state_score_bytes = attn_score;
        state_cursor[li->stage_id] =
            sat_add_u64(state_cursor[li->stage_id], view->attn_state_score_bytes);

        view->indexer_state_kv_offset =
            sat_add_u64(state_base, state_cursor[li->stage_id]);
        view->indexer_state_kv_bytes = idx_kv;
        state_cursor[li->stage_id] =
            sat_add_u64(state_cursor[li->stage_id], view->indexer_state_kv_bytes);

        view->indexer_state_score_offset =
            sat_add_u64(state_base, state_cursor[li->stage_id]);
        view->indexer_state_score_bytes = idx_score;
        state_cursor[li->stage_id] =
            sat_add_u64(state_cursor[li->stage_id], view->indexer_state_score_bytes);

        view->total_bytes = li->kv_budget.total_bytes;
    }
}

static uint64_t checked_stage_used(const ds4_v100_stage_info *s) {
    return s->arena_bytes + s->scratch_bytes + s->relay_f16_bytes +
           s->relay_f32_debug_bytes + s->planned_kv_bytes +
           s->output_head_reserve_bytes + s->mtp_reserve_bytes;
}

static int validate_topology(ds4_v100_context *ctx, char *err, size_t errlen) {
    if (!ctx->opts.require_production_topology) return 0;
    if (ctx->opts.expected_gpus != DS4_V100_EXPECTED_GPUS) {
        return v100_error(err, errlen, "production topology requires %d GPUs",
                          DS4_V100_EXPECTED_GPUS);
    }
    if (!ctx->opts.device_facts ||
        ctx->opts.n_device_facts != DS4_V100_EXPECTED_GPUS) {
        return v100_error(err, errlen, "production topology requires %d device facts",
                          DS4_V100_EXPECTED_GPUS);
    }
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        const ds4_v100_device_fact *f = &ctx->opts.device_facts[i];
        if (f->visible_id != i) {
            return v100_error(err, errlen, "device fact %d has visible id %d", i, f->visible_id);
        }
        if (f->cc_major != 7) {
            return v100_error(err, errlen, "device %d is not V100-class compute capability 7.x", i);
        }
        if (f->total_global_mem < DS4_V100_MIN_VRAM_BYTES) {
            return v100_error(err, errlen, "device %d has insufficient VRAM: %" PRIu64,
                              i, f->total_global_mem);
        }
        ctx->stages[i].device_total_bytes = f->total_global_mem;
    }
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS - 1; i++) {
        if (!ctx->opts.device_facts[i].peer_access[i + 1] ||
            !ctx->opts.device_facts[i + 1].peer_access[i]) {
            return v100_error(err, errlen, "missing peer edge between stage %d and %d", i, i + 1);
        }
    }
    return 0;
}

static int parse_shape_dims(const char *shape,
                            uint64_t *dims,
                            uint32_t cap,
                            uint32_t *out_n,
                            uint64_t *out_elements) {
    if (!shape || shape[0] != '[') return 1;
    const char *p = shape + 1;
    uint64_t product = 1;
    uint32_t n = 0;
    while (*p && *p != ']') {
        if (!isdigit((unsigned char)*p)) return 1;
        uint64_t dim = 0;
        while (isdigit((unsigned char)*p)) {
            unsigned d = (unsigned)(*p - '0');
            if (dim > (UINT64_MAX - d) / 10) return 1;
            dim = dim * 10 + d;
            p++;
        }
        if (dim == 0 || product > UINT64_MAX / dim) return 1;
        if (n >= cap) return 1;
        if (dims) dims[n] = dim;
        n++;
        product *= dim;
        if (*p == 'x') {
            p++;
            continue;
        }
        if (*p != ']') return 1;
    }
    if (*p != ']' || p[1] != '\0' || n == 0) return 1;
    if (out_n) *out_n = n;
    if (out_elements) *out_elements = product;
    return 0;
}

static int parse_shape_elements(const char *shape, uint64_t *out) {
    return parse_shape_dims(shape, NULL, DS4_V100_MAX_SHAPE_DIMS, NULL, out);
}

static int expected_bytes_for_entry(const ds4_pack_entry *e, uint64_t *out) {
    uint64_t elements = 0;
    if (parse_shape_elements(e->source_shape, &elements)) return 1;
    ds4_v100_source_dtype dtype = ds4_v100_source_dtype_parse(e->source_dtype);
    switch (dtype) {
    case DS4_V100_SOURCE_BF16:
        if (elements > UINT64_MAX / 2) return 1;
        *out = elements * 2;
        return 0;
    case DS4_V100_SOURCE_F32:
    case DS4_V100_SOURCE_I32:
        if (elements > UINT64_MAX / 4) return 1;
        *out = elements * 4;
        return 0;
    case DS4_V100_SOURCE_F8_E4M3_B128:
        if (elements > UINT64_MAX - ((elements + 127) / 128)) return 1;
        *out = elements + ((elements + 127) / 128);
        return 0;
    case DS4_V100_SOURCE_MXFP4:
    case DS4_V100_SOURCE_FP4:
        if (elements % 32) return 1;
        if (elements > UINT64_MAX / 17) return 1;
        *out = (elements * 17) / 32;
        return 0;
    case DS4_V100_SOURCE_UNKNOWN:
    default:
        return 1;
    }
}

static int append_desc(ds4_v100_context *ctx, const ds4_pack_entry *entry,
                       const ds4_v100_policy *policy,
                       char *err, size_t errlen) {
    if (ctx->n_descs == ctx->cap_descs) {
        uint64_t next = ctx->cap_descs ? ctx->cap_descs * 2 : 256;
        if (next < ctx->cap_descs || next > SIZE_MAX / sizeof(ctx->descs[0])) {
            return v100_error(err, errlen, "too many V100 descriptors");
        }
        ds4_v100_tensor_desc *p =
            (ds4_v100_tensor_desc *)realloc(ctx->descs, (size_t)next * sizeof(ctx->descs[0]));
        if (!p) return v100_error(err, errlen, "out of memory growing V100 descriptors");
        ctx->descs = p;
        ctx->cap_descs = next;
    }
    ctx->descs[ctx->n_descs].entry = entry;
    ctx->descs[ctx->n_descs].policy = *policy;
    ctx->n_descs++;
    ctx->exec_counts[policy->exec_kind]++;
    return 0;
}

static int append_tm_desc(ds4_v100_context *ctx,
                          const ds4_tm_pack_entry *entry,
                          const ds4_v100_policy *policy,
                          char *err,
                          size_t errlen) {
    if (ctx->n_tm_descs == ctx->cap_tm_descs) {
        uint64_t next = ctx->cap_tm_descs ? ctx->cap_tm_descs * 2 : 128;
        if (next < ctx->cap_tm_descs || next > SIZE_MAX / sizeof(ctx->tm_descs[0])) {
            return v100_error(err, errlen, "too many V100 TurboMind descriptors");
        }
        ds4_v100_tm_tensor_desc *p =
            (ds4_v100_tm_tensor_desc *)realloc(ctx->tm_descs,
                                               (size_t)next * sizeof(ctx->tm_descs[0]));
        if (!p) return v100_error(err, errlen, "out of memory growing V100 TurboMind descriptors");
        ctx->tm_descs = p;
        ctx->cap_tm_descs = next;
    }
    ctx->tm_descs[ctx->n_tm_descs].entry = entry;
    ctx->tm_descs[ctx->n_tm_descs].policy = *policy;
    ctx->n_tm_descs++;
    ctx->exec_counts[policy->exec_kind]++;
    return 0;
}

static int bind_pack_entry(const ds4_pack_entry *e, void *ud) {
    ds4_v100_bind_state *state = (ds4_v100_bind_state *)ud;
    ds4_v100_context *ctx = state->ctx;
    int max_gpu = ctx->opts.expected_gpus > 0 ? ctx->opts.expected_gpus : DS4_V100_EXPECTED_GPUS;
    if (e->owning_gpu < 0 || e->owning_gpu >= max_gpu) {
        return v100_error(state->err, state->errlen, "%s has invalid owning GPU %d",
                          e->semantic_tensor_id, e->owning_gpu);
    }
    if (e->layer_id >= DS4_V100_N_LAYERS) {
        return v100_error(state->err, state->errlen, "%s has invalid layer id %d",
                          e->semantic_tensor_id, e->layer_id);
    }
    if (e->layer_id >= 0 && ds4_v100_stage_for_layer(e->layer_id) != e->owning_gpu) {
        return v100_error(state->err, state->errlen,
                          "%s owner gpu %d does not match layer %d stage %d",
                          e->semantic_tensor_id, e->owning_gpu, e->layer_id,
                          ds4_v100_stage_for_layer(e->layer_id));
    }

    uint64_t expected = 0;
    if (expected_bytes_for_entry(e, &expected)) {
        return v100_error(state->err, state->errlen,
                          "%s has unsupported source shape/dtype %s %s",
                          e->semantic_tensor_id, e->source_dtype, e->source_shape);
    }
    if (expected != e->byte_length) {
        return v100_error(state->err, state->errlen,
                          "%s byte length mismatch: expected %" PRIu64 " got %" PRIu64,
                          e->semantic_tensor_id, expected, e->byte_length);
    }

    ds4_v100_policy policy;
    if (ds4_v100_classify_or_die(e->source_dtype, e->runtime_layout,
                                 e->kernel_family, &policy, state->err, state->errlen)) {
        return 1;
    }
    ds4_v100_stage_info *stage = &ctx->stages[e->owning_gpu];
    uint64_t arena_bytes = ds4_pack_arena_bytes(ctx->pack, e->owning_gpu);
    if (e->shard_offset > arena_bytes || e->byte_length > arena_bytes - e->shard_offset) {
        return v100_error(state->err, state->errlen,
                          "%s arena span overflows gpu%d arena",
                          e->semantic_tensor_id, e->owning_gpu);
    }
    stage->tensor_count++;
    if (e->layer_id >= 0) {
        ds4_v100_layer_info *li = &ctx->layers[e->layer_id];
        li->tensor_count++;
        switch (policy.family) {
        case DS4_V100_FAMILY_F32_CONTROL: li->has_f32_control = true; break;
        case DS4_V100_FAMILY_FP8_DENSE: li->has_fp8_dense = true; break;
        case DS4_V100_FAMILY_MXFP4_EXPERT: li->has_mxfp4_expert = true; break;
        case DS4_V100_FAMILY_HC_CONTROL: li->has_hc_control = true; break;
        case DS4_V100_FAMILY_BF16_GLOBAL:
        case DS4_V100_FAMILY_KV_CACHE:
        case DS4_V100_FAMILY_UNKNOWN:
        default:
            break;
        }
    }
    if (!strcmp(e->semantic_tensor_id, "token_embd.weight")) ctx->has_token_embedding = true;
    return append_desc(ctx, e, &policy, state->err, state->errlen);
}

static int bind_pack(ds4_v100_context *ctx, char *err, size_t errlen) {
    if (!ctx->opts.pack_index_path || !ctx->opts.pack_index_path[0]) return 0;
    if (ds4_pack_open(&ctx->pack, ctx->opts.pack_index_path, err, errlen)) return 1;
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        ctx->stages[i].arena_bytes = ds4_pack_arena_bytes(ctx->pack, i);
    }
    ds4_v100_bind_state state = { ctx, err, errlen };
    return ds4_pack_for_each(ctx->pack, bind_pack_entry, &state);
}

static bool is_routed_expert_id(const char *id) {
    return id &&
           (strstr(id, ".ffn_gate_exps.weight") ||
            strstr(id, ".ffn_up_exps.weight") ||
            strstr(id, ".ffn_gate_up_exps.weight") ||
            strstr(id, ".ffn_down_exps.weight"));
}

static int bind_tm_pack_entry(const ds4_tm_pack_entry *e, void *ud) {
    ds4_v100_bind_state *state = (ds4_v100_bind_state *)ud;
    ds4_v100_context *ctx = state->ctx;
    int max_gpu = ctx->opts.expected_gpus > 0 ? ctx->opts.expected_gpus : DS4_V100_EXPECTED_GPUS;
    if (e->owning_gpu < 0 || e->owning_gpu >= max_gpu) {
        return v100_error(state->err, state->errlen, "%s has invalid TurboMind owning GPU %d",
                          e->semantic_tensor_id, e->owning_gpu);
    }
    if (e->layer_id < 0 || e->layer_id >= DS4_V100_N_LAYERS) {
        return v100_error(state->err, state->errlen, "%s has invalid TurboMind layer id %d",
                          e->semantic_tensor_id, e->layer_id);
    }
    if (ds4_v100_stage_for_layer(e->layer_id) != e->owning_gpu) {
        return v100_error(state->err, state->errlen,
                          "%s TurboMind owner gpu %d does not match layer %d stage %d",
                          e->semantic_tensor_id, e->owning_gpu, e->layer_id,
                          ds4_v100_stage_for_layer(e->layer_id));
    }
    if (!is_routed_expert_id(e->semantic_tensor_id)) {
        return v100_error(state->err, state->errlen,
                          "%s TurboMind binding is not a routed expert tensor",
                          e->semantic_tensor_id);
    }
    if (!str_eq_ci(e->source_dtype, "mxfp4")) {
        return v100_error(state->err, state->errlen,
                          "%s TurboMind source dtype must be mxfp4",
                          e->semantic_tensor_id);
    }
    uint64_t dims[DS4_V100_MAX_SHAPE_DIMS] = {0};
    uint32_t n_dims = 0;
    if (parse_shape_dims(e->source_shape, dims, DS4_V100_MAX_SHAPE_DIMS, &n_dims, NULL) ||
        n_dims != 3 ||
        dims[0] != e->k ||
        dims[1] != e->n ||
        dims[2] != e->experts_total) {
        return v100_error(state->err, state->errlen,
                          "%s TurboMind shape does not match n/k/expert metadata",
                          e->semantic_tensor_id);
    }

    ds4_v100_policy policy;
    if (ds4_v100_classify_or_die(e->source_dtype, e->runtime_layout,
                                 e->kernel_family, &policy, state->err, state->errlen)) {
        return 1;
    }
    ds4_v100_stage_info *stage = &ctx->stages[e->owning_gpu];
    uint64_t weight_end =
        e->weight_offset + (uint64_t)e->experts_packed * e->weight_bytes_per_expert;
    uint64_t scale_end =
        e->scale_offset + (uint64_t)e->experts_packed * e->scale_bytes_per_expert;
    if (weight_end < e->weight_offset || scale_end < e->scale_offset) {
        return v100_error(state->err, state->errlen,
                          "%s TurboMind arena span overflows",
                          e->semantic_tensor_id);
    }
    if (weight_end > stage->arena_bytes) stage->arena_bytes = weight_end;
    if (scale_end > stage->arena_bytes) stage->arena_bytes = scale_end;
    stage->tensor_count++;

    ds4_v100_layer_info *li = &ctx->layers[e->layer_id];
    li->tensor_count++;
    li->has_mxfp4_expert = true;
    return append_tm_desc(ctx, e, &policy, state->err, state->errlen);
}

static int bind_tm_pack(ds4_v100_context *ctx, char *err, size_t errlen) {
    if (!ctx->opts.turbomind_pack_index_path ||
        !ctx->opts.turbomind_pack_index_path[0]) {
        return 0;
    }
    if (ds4_tm_pack_open(&ctx->tm_pack, ctx->opts.turbomind_pack_index_path, err, errlen)) {
        return 1;
    }
    ds4_v100_bind_state state = { ctx, err, errlen };
    return ds4_tm_pack_for_each(ctx->tm_pack, bind_tm_pack_entry, &state);
}

static int validate_memory_budget(ds4_v100_context *ctx, char *err, size_t errlen) {
    if (!ctx->opts.require_production_topology) return 0;
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        const ds4_v100_stage_info *s = &ctx->stages[i];
        uint64_t used = checked_stage_used(s);
        if (used > s->device_total_bytes ||
            s->reserve_bytes > s->device_total_bytes - used) {
            return v100_error(err, errlen,
                              "stage %d falls below reserve: used=%" PRIu64 " total=%" PRIu64 " reserve=%" PRIu64,
                              i, used, s->device_total_bytes, s->reserve_bytes);
        }
    }
    return 0;
}

int ds4_v100_context_open(ds4_v100_context **out,
                          const ds4_v100_context_options *opts,
                          char *err,
                          size_t errlen) {
    if (!out) return v100_error(err, errlen, "missing output context pointer");
    *out = NULL;
    ds4_v100_context_options local;
    if (opts) local = *opts;
    else ds4_v100_context_options_init(&local);
    if (local.expected_gpus <= 0 || local.expected_gpus > DS4_V100_EXPECTED_GPUS) {
        return v100_error(err, errlen, "expected_gpus must be 1..%d", DS4_V100_EXPECTED_GPUS);
    }
    if (local.relay_max_active_slots == 0) {
        return v100_error(err, errlen, "relay_max_active_slots must be nonzero");
    }
    if (local.kv_ctx_tokens != 0 && local.kv_active_slots == 0) {
        return v100_error(err, errlen, "kv_active_slots must be nonzero when kv_ctx_tokens is set");
    }
    if (local.kv_ctx_tokens != 0 && local.planned_kv_bytes_per_gpu != 0) {
        return v100_error(err, errlen,
                          "planned_kv_bytes_per_gpu cannot be combined with derived kv_ctx_tokens");
    }

    ds4_v100_context *ctx = (ds4_v100_context *)calloc(1, sizeof(*ctx));
    if (!ctx) return v100_error(err, errlen, "out of memory allocating V100 context");
    ctx->opts = local;
    init_stage_map(ctx);
    apply_derived_kv_plan(ctx);
    if (validate_topology(ctx, err, errlen) ||
        bind_pack(ctx, err, errlen) ||
        bind_tm_pack(ctx, err, errlen) ||
        validate_memory_budget(ctx, err, errlen)) {
        ds4_v100_context_close(ctx);
        return 1;
    }
    *out = ctx;
    return 0;
}

void ds4_v100_context_close(ds4_v100_context *ctx) {
    if (!ctx) return;
    ds4_tm_pack_close(ctx->tm_pack);
    ds4_pack_close(ctx->pack);
    free(ctx->tm_descs);
    free(ctx->descs);
    free(ctx);
}

int ds4_v100_context_stage_count(const ds4_v100_context *ctx) {
    return ctx ? ctx->opts.expected_gpus : 0;
}

const ds4_v100_stage_info *ds4_v100_context_stage(const ds4_v100_context *ctx,
                                                  int stage_id) {
    if (!ctx || stage_id < 0 || stage_id >= ctx->opts.expected_gpus) return NULL;
    return &ctx->stages[stage_id];
}

const ds4_v100_layer_info *ds4_v100_context_layer(const ds4_v100_context *ctx,
                                                  int layer_id) {
    if (!ctx || layer_id < 0 || layer_id >= DS4_V100_N_LAYERS) return NULL;
    return &ctx->layers[layer_id];
}

uint64_t ds4_v100_context_tensor_count(const ds4_v100_context *ctx) {
    return ctx ? ctx->n_descs + ctx->n_tm_descs : 0;
}

uint64_t ds4_v100_context_exec_count(const ds4_v100_context *ctx,
                                     ds4_v100_exec_kind kind) {
    if (!ctx || kind < 0 || kind >= DS4_V100_EXEC_COUNT) return 0;
    return ctx->exec_counts[kind];
}

bool ds4_v100_context_has_token_embedding(const ds4_v100_context *ctx) {
    return ctx && ctx->has_token_embedding;
}

static int fill_binding(const ds4_v100_tensor_desc *desc,
                        ds4_v100_tensor_binding *out,
                        char *err,
                        size_t errlen) {
    if (!desc || !desc->entry || !out) {
        return v100_error(err, errlen, "missing descriptor binding output");
    }
    const ds4_pack_entry *e = desc->entry;
    memset(out, 0, sizeof(*out));
    out->semantic_tensor_id = e->semantic_tensor_id;
    out->source_name = e->source_name;
    out->source_dtype = e->source_dtype;
    out->source_shape = e->source_shape;
    out->runtime_layout = e->runtime_layout;
    out->kernel_family = e->kernel_family;
    out->shard_file = e->shard_file;
    out->owning_gpu = e->owning_gpu;
    out->layer_id = e->layer_id;
    out->scale_offset = e->scale_offset;
    out->source_offset = e->source_offset;
    out->byte_length = e->byte_length;
    out->shard_offset = e->shard_offset;
    out->policy = desc->policy;
    if (parse_shape_dims(e->source_shape,
                         out->shape,
                         DS4_V100_MAX_SHAPE_DIMS,
                         &out->n_shape_dims,
                         NULL)) {
        return v100_error(err, errlen, "%s has invalid source shape %s",
                          e->semantic_tensor_id, e->source_shape);
    }
    return 0;
}

static int fill_tm_binding(const ds4_v100_tm_tensor_desc *desc,
                           ds4_v100_turbomind_binding *out,
                           char *err,
                           size_t errlen) {
    if (!desc || !desc->entry || !out) {
        return v100_error(err, errlen, "missing TurboMind descriptor binding output");
    }
    const ds4_tm_pack_entry *e = desc->entry;
    memset(out, 0, sizeof(*out));
    out->semantic_tensor_id = e->semantic_tensor_id;
    out->source_name = e->source_name;
    out->source_dtype = e->source_dtype;
    out->source_shape = e->source_shape;
    out->runtime_layout = e->runtime_layout;
    out->kernel_family = e->kernel_family;
    out->shard_file = e->sidecar_file;
    out->source_shard_file = e->source_shard_file;
    out->owning_gpu = e->owning_gpu;
    out->layer_id = e->layer_id;
    out->n = e->n;
    out->k = e->k;
    out->experts_packed = e->experts_packed;
    out->experts_total = e->experts_total;
    out->weight_bytes_per_expert = e->weight_bytes_per_expert;
    out->scale_bytes_per_expert = e->scale_bytes_per_expert;
    out->k_pack = e->k_pack;
    out->weight_stride = e->weight_stride;
    out->scale_stride = e->scale_stride;
    out->weight_offset = e->weight_offset;
    out->scale_offset = e->scale_offset;
    out->source_shard_offset = e->source_shard_offset;
    out->source_byte_length = e->source_byte_length;
    out->tm_abi_version = e->tm_abi_version;
    out->policy = desc->policy;
    if (parse_shape_dims(e->source_shape,
                         out->shape,
                         DS4_V100_MAX_SHAPE_DIMS,
                         &out->n_shape_dims,
                         NULL)) {
        return v100_error(err, errlen, "%s has invalid TurboMind source shape %s",
                          e->semantic_tensor_id, e->source_shape);
    }
    return 0;
}

int ds4_v100_context_lookup_tensor_binding(const ds4_v100_context *ctx,
                                           const char *semantic_tensor_id,
                                           ds4_v100_tensor_binding *out,
                                           char *err,
                                           size_t errlen) {
    if (!ctx) return v100_error(err, errlen, "missing V100 context");
    if (!semantic_tensor_id || !semantic_tensor_id[0]) {
        return v100_error(err, errlen, "missing semantic tensor id");
    }
    if (!out) return v100_error(err, errlen, "missing tensor binding output");
    for (uint64_t i = 0; i < ctx->n_descs; i++) {
        const ds4_v100_tensor_desc *desc = &ctx->descs[i];
        if (desc->entry && !strcmp(desc->entry->semantic_tensor_id, semantic_tensor_id)) {
            return fill_binding(desc, out, err, errlen);
        }
    }
    return v100_error(err, errlen, "missing tensor descriptor %s", semantic_tensor_id);
}

int ds4_v100_context_lookup_turbomind_binding(const ds4_v100_context *ctx,
                                              const char *semantic_tensor_id,
                                              ds4_v100_turbomind_binding *out,
                                              char *err,
                                              size_t errlen) {
    if (!ctx) return v100_error(err, errlen, "missing V100 context");
    if (!semantic_tensor_id || !semantic_tensor_id[0]) {
        return v100_error(err, errlen, "missing semantic tensor id");
    }
    if (!out) return v100_error(err, errlen, "missing TurboMind binding output");
    for (uint64_t i = 0; i < ctx->n_tm_descs; i++) {
        const ds4_v100_tm_tensor_desc *desc = &ctx->tm_descs[i];
        if (desc->entry && !strcmp(desc->entry->semantic_tensor_id, semantic_tensor_id)) {
            return fill_tm_binding(desc, out, err, errlen);
        }
    }
    return v100_error(err, errlen, "missing TurboMind tensor descriptor %s",
                      semantic_tensor_id);
}

int ds4_v100_context_require_layer_tensor_binding(const ds4_v100_context *ctx,
                                                  int layer_id,
                                                  const char *tensor_suffix,
                                                  ds4_v100_tensor_binding *out,
                                                  char *err,
                                                  size_t errlen) {
    if (layer_id < 0 || layer_id >= DS4_V100_N_LAYERS) {
        return v100_error(err, errlen, "bad layer id %d", layer_id);
    }
    if (!tensor_suffix || !tensor_suffix[0]) {
        return v100_error(err, errlen, "missing layer tensor suffix");
    }
    char id[192];
    int n = snprintf(id, sizeof(id), "blk.%d.%s", layer_id, tensor_suffix);
    if (n < 0 || (size_t)n >= sizeof(id)) {
        return v100_error(err, errlen, "layer tensor id too long");
    }
    return ds4_v100_context_lookup_tensor_binding(ctx, id, out, err, errlen);
}

int ds4_v100_context_require_layer_turbomind_binding(const ds4_v100_context *ctx,
                                                     int layer_id,
                                                     const char *tensor_suffix,
                                                     ds4_v100_turbomind_binding *out,
                                                     char *err,
                                                     size_t errlen) {
    if (layer_id < 0 || layer_id >= DS4_V100_N_LAYERS) {
        return v100_error(err, errlen, "bad layer id %d", layer_id);
    }
    if (!tensor_suffix || !tensor_suffix[0]) {
        return v100_error(err, errlen, "missing layer tensor suffix");
    }
    char id[192];
    int n = snprintf(id, sizeof(id), "blk.%d.%s", layer_id, tensor_suffix);
    if (n < 0 || (size_t)n >= sizeof(id)) {
        return v100_error(err, errlen, "layer tensor id too long");
    }
    return ds4_v100_context_lookup_turbomind_binding(ctx, id, out, err, errlen);
}

int ds4_v100_context_output_head_binding(const ds4_v100_context *ctx,
                                         ds4_v100_tensor_binding *out,
                                         char *err,
                                         size_t errlen) {
    return ds4_v100_context_lookup_tensor_binding(ctx, "output.weight", out, err, errlen);
}

int ds4_v100_context_validate_layer_skeleton(const ds4_v100_context *ctx,
                                             FILE *report,
                                             char *err,
                                             size_t errlen) {
    if (!ctx) return v100_error(err, errlen, "missing V100 context");
    const uint64_t total_descs = ds4_v100_context_tensor_count(ctx);
    if (report) {
        fprintf(report, "layer\tstage\tclass\tkv_bytes\ttensors\tf32_control\tfp8_dense\tmxfp4_expert\thc_control\tstatus\n");
    }
    for (int layer = 0; layer < DS4_V100_N_LAYERS; layer++) {
        const ds4_v100_layer_info *li = &ctx->layers[layer];
        int expect_stage = ds4_v100_stage_for_layer(layer);
        const char *status = "OK";
        if (li->stage_id != expect_stage) status = "BAD_STAGE";
        else if (total_descs > 0 && li->tensor_count == 0) status = "MISSING_LAYER_DESCRIPTORS";
        else if (total_descs > 0 && !li->has_f32_control) status = "MISSING_F32_CONTROL";
        else if (total_descs > 0 && !li->has_fp8_dense) status = "MISSING_FP8_DENSE";
        else if (total_descs > 0 && !li->has_mxfp4_expert) status = "MISSING_MXFP4_EXPERT";
        else if (total_descs > 0 && !li->has_hc_control) status = "MISSING_HC_CONTROL";
        if (report) {
            fprintf(report, "%d\t%d\t%s\t%" PRIu64 "\t%" PRIu64 "\t%d\t%d\t%d\t%d\t%s\n",
                    layer, li->stage_id,
                    ds4_v100_layer_class_name(li->layer_class),
                    li->kv_budget.total_bytes,
                    li->tensor_count,
                    li->has_f32_control ? 1 : 0,
                    li->has_fp8_dense ? 1 : 0,
                    li->has_mxfp4_expert ? 1 : 0,
                    li->has_hc_control ? 1 : 0,
                    status);
        }
        if (strcmp(status, "OK")) {
            return v100_error(err, errlen, "layer %d skeleton validation failed: %s",
                              layer, status);
        }
    }
    if (total_descs > 0 && !ctx->has_token_embedding) {
        return v100_error(err, errlen, "missing token embedding descriptor");
    }
    return 0;
}

void ds4_v100_context_print_report(const ds4_v100_context *ctx, FILE *fp) {
    if (!ctx || !fp) return;
    fprintf(fp, "ds4_v100_context_report\n");
    fprintf(fp, "expected_gpus\t%d\n", ctx->opts.expected_gpus);
    fprintf(fp, "mode\t%d\n", (int)ctx->opts.mode);
    fprintf(fp, "policy\tbf16_fp8_fp4_are_not_native_v100_tensor_core_formats\n");
    fprintf(fp, "policy\tproduction_dense_gemm_target_fp16_hmma_with_fp32_accumulation\n");
    fprintf(fp, "kv_ctx_tokens\t%" PRIu64 "\n", ctx->opts.kv_ctx_tokens);
    fprintf(fp, "kv_active_slots\t%" PRIu64 "\n", ctx->opts.kv_active_slots);
    fprintf(fp, "tensor_count\t%" PRIu64 "\n", ds4_v100_context_tensor_count(ctx));
    fprintf(fp, "pack_tensor_count\t%" PRIu64 "\n", ctx->n_descs);
    fprintf(fp, "turbomind_tensor_count\t%" PRIu64 "\n", ctx->n_tm_descs);
    for (int k = 0; k < DS4_V100_EXEC_COUNT; k++) {
        fprintf(fp, "exec_count\t%s\t%" PRIu64 "\n",
                ds4_v100_exec_kind_name((ds4_v100_exec_kind)k),
                ctx->exec_counts[k]);
    }
    int class_counts[3] = {0, 0, 0};
    for (int layer = 0; layer < DS4_V100_N_LAYERS; layer++) {
        class_counts[ctx->layers[layer].layer_class]++;
    }
    fprintf(fp, "layer_class_count\t%s\t%d\n",
            ds4_v100_layer_class_name(DS4_V100_LAYER_SWA_ONLY),
            class_counts[DS4_V100_LAYER_SWA_ONLY]);
    fprintf(fp, "layer_class_count\t%s\t%d\n",
            ds4_v100_layer_class_name(DS4_V100_LAYER_RATIO_4),
            class_counts[DS4_V100_LAYER_RATIO_4]);
    fprintf(fp, "layer_class_count\t%s\t%d\n",
            ds4_v100_layer_class_name(DS4_V100_LAYER_RATIO_128),
            class_counts[DS4_V100_LAYER_RATIO_128]);
    fprintf(fp, "stage\tgpu\tlayers\tarena_bytes\tscratch_bytes\trelay_f16_bytes\trelay_f32_debug_bytes\tkv_arena_bytes\tkv_raw_swa_offset\tkv_raw_swa_bytes\tkv_compressed_attn_offset\tkv_compressed_attn_bytes\tkv_indexer_offset\tkv_indexer_bytes\tkv_compression_state_offset\tkv_compression_state_bytes\tplanned_kv_bytes\treserve_bytes\tdevice_total_bytes\n");
    for (int i = 0; i < ctx->opts.expected_gpus; i++) {
        const ds4_v100_stage_info *s = &ctx->stages[i];
        fprintf(fp, "%d\t%d\t%d-%d\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\n",
                s->stage_id, s->gpu, s->layer_begin, s->layer_end,
                s->arena_bytes, s->scratch_bytes, s->relay_f16_bytes,
                s->relay_f32_debug_bytes, s->kv_arena.total_bytes,
                s->kv_arena.raw_swa_offset,
                s->kv_arena.raw_swa_bytes, s->kv_arena.compressed_attn_offset,
                s->kv_arena.compressed_attn_bytes, s->kv_arena.indexer_kv_offset,
                s->kv_arena.indexer_kv_bytes, s->kv_arena.compression_state_offset,
                s->kv_arena.compression_state_bytes, s->planned_kv_bytes,
                s->reserve_bytes, s->device_total_bytes);
    }
    fprintf(fp, "layer_kv_view\tlayer\tstage\tclass\traw_swa_offset\traw_swa_bytes\tcompressed_attn_offset\tcompressed_attn_bytes\tindexer_kv_offset\tindexer_kv_bytes\tattn_state_kv_offset\tattn_state_kv_bytes\tattn_state_score_offset\tattn_state_score_bytes\tindexer_state_kv_offset\tindexer_state_kv_bytes\tindexer_state_score_offset\tindexer_state_score_bytes\ttotal_bytes\n");
    for (int layer = 0; layer < DS4_V100_N_LAYERS; layer++) {
        const ds4_v100_layer_info *li = &ctx->layers[layer];
        const ds4_v100_layer_kv_view *v = &li->kv_view;
        fprintf(fp, "layer_kv_view\t%d\t%d\t%s\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\n",
                layer, li->stage_id, ds4_v100_layer_class_name(li->layer_class),
                v->raw_swa_offset, v->raw_swa_bytes,
                v->compressed_attn_offset, v->compressed_attn_bytes,
                v->indexer_kv_offset, v->indexer_kv_bytes,
                v->attn_state_kv_offset, v->attn_state_kv_bytes,
                v->attn_state_score_offset, v->attn_state_score_bytes,
                v->indexer_state_kv_offset, v->indexer_state_kv_bytes,
                v->indexer_state_score_offset, v->indexer_state_score_bytes,
                v->total_bytes);
    }
}
