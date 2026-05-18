#include "ds4_v100_context.h"
#include "ds4_pack.h"

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

struct ds4_v100_context {
    ds4_v100_context_options opts;
    ds4_pack *pack;
    ds4_v100_stage_info stages[DS4_V100_EXPECTED_GPUS];
    ds4_v100_tensor_desc *descs;
    uint64_t n_descs;
    uint64_t cap_descs;
    uint64_t exec_counts[DS4_V100_EXEC_COUNT];
    bool has_token_embedding;
    bool layer_seen[DS4_V100_N_LAYERS];
};

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
        p.forbidden_claim = "native_bf16_tensor_core_execution";
        break;
    case DS4_V100_FAMILY_HC_CONTROL:
    case DS4_V100_FAMILY_F32_CONTROL:
        if (dtype != DS4_V100_SOURCE_F32 && dtype != DS4_V100_SOURCE_I32) break;
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

static int parse_shape_elements(const char *shape, uint64_t *out) {
    if (!shape || shape[0] != '[') return 1;
    const char *p = shape + 1;
    uint64_t product = 1;
    bool saw_dim = false;
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
        product *= dim;
        saw_dim = true;
        if (*p == 'x') {
            p++;
            continue;
        }
        if (*p != ']') return 1;
    }
    if (*p != ']' || p[1] != '\0' || !saw_dim) return 1;
    *out = product;
    return 0;
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

static int bind_pack_entry(const ds4_pack_entry *e, void *ud) {
    ds4_v100_context *ctx = (ds4_v100_context *)ud;
    char *err = NULL;
    size_t errlen = 0;
    (void)err;
    (void)errlen;
    int max_gpu = ctx->opts.expected_gpus > 0 ? ctx->opts.expected_gpus : DS4_V100_EXPECTED_GPUS;
    if (e->owning_gpu < 0 || e->owning_gpu >= max_gpu) return 1;
    if (e->layer_id >= DS4_V100_N_LAYERS) return 1;
    if (e->layer_id >= 0 && ds4_v100_stage_for_layer(e->layer_id) != e->owning_gpu) return 1;

    uint64_t expected = 0;
    if (expected_bytes_for_entry(e, &expected) || expected != e->byte_length) return 1;

    ds4_v100_policy policy;
    if (ds4_v100_classify_or_die(e->source_dtype, e->runtime_layout,
                                 e->kernel_family, &policy, NULL, 0)) {
        return 1;
    }
    ds4_v100_stage_info *stage = &ctx->stages[e->owning_gpu];
    uint64_t arena_bytes = ds4_pack_arena_bytes(ctx->pack, e->owning_gpu);
    if (e->shard_offset > arena_bytes || e->byte_length > arena_bytes - e->shard_offset) return 1;
    stage->tensor_count++;
    if (e->layer_id >= 0) ctx->layer_seen[e->layer_id] = true;
    if (!strcmp(e->semantic_tensor_id, "token_embd.weight")) ctx->has_token_embedding = true;
    return append_desc(ctx, e, &policy, NULL, 0);
}

static int bind_pack(ds4_v100_context *ctx, char *err, size_t errlen) {
    if (!ctx->opts.pack_index_path || !ctx->opts.pack_index_path[0]) return 0;
    if (ds4_pack_open(&ctx->pack, ctx->opts.pack_index_path, err, errlen)) return 1;
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        ctx->stages[i].arena_bytes = ds4_pack_arena_bytes(ctx->pack, i);
    }
    int rc = ds4_pack_for_each(ctx->pack, bind_pack_entry, ctx);
    if (rc) {
        return v100_error(err, errlen, "V100 descriptor binding failed");
    }
    return 0;
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

    ds4_v100_context *ctx = (ds4_v100_context *)calloc(1, sizeof(*ctx));
    if (!ctx) return v100_error(err, errlen, "out of memory allocating V100 context");
    ctx->opts = local;
    init_stage_map(ctx);
    if (validate_topology(ctx, err, errlen) ||
        bind_pack(ctx, err, errlen) ||
        validate_memory_budget(ctx, err, errlen)) {
        ds4_v100_context_close(ctx);
        return 1;
    }
    *out = ctx;
    return 0;
}

void ds4_v100_context_close(ds4_v100_context *ctx) {
    if (!ctx) return;
    ds4_pack_close(ctx->pack);
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

uint64_t ds4_v100_context_tensor_count(const ds4_v100_context *ctx) {
    return ctx ? ctx->n_descs : 0;
}

uint64_t ds4_v100_context_exec_count(const ds4_v100_context *ctx,
                                     ds4_v100_exec_kind kind) {
    if (!ctx || kind < 0 || kind >= DS4_V100_EXEC_COUNT) return 0;
    return ctx->exec_counts[kind];
}

bool ds4_v100_context_has_token_embedding(const ds4_v100_context *ctx) {
    return ctx && ctx->has_token_embedding;
}

void ds4_v100_context_print_report(const ds4_v100_context *ctx, FILE *fp) {
    if (!ctx || !fp) return;
    fprintf(fp, "ds4_v100_context_report\n");
    fprintf(fp, "expected_gpus\t%d\n", ctx->opts.expected_gpus);
    fprintf(fp, "mode\t%d\n", (int)ctx->opts.mode);
    fprintf(fp, "policy\tbf16_fp8_fp4_are_not_native_v100_tensor_core_formats\n");
    fprintf(fp, "policy\tproduction_dense_gemm_target_fp16_hmma_with_fp32_accumulation\n");
    fprintf(fp, "tensor_count\t%" PRIu64 "\n", ctx->n_descs);
    for (int k = 0; k < DS4_V100_EXEC_COUNT; k++) {
        fprintf(fp, "exec_count\t%s\t%" PRIu64 "\n",
                ds4_v100_exec_kind_name((ds4_v100_exec_kind)k),
                ctx->exec_counts[k]);
    }
    fprintf(fp, "stage\tgpu\tlayers\tarena_bytes\tscratch_bytes\trelay_f16_bytes\trelay_f32_debug_bytes\tplanned_kv_bytes\treserve_bytes\tdevice_total_bytes\n");
    for (int i = 0; i < ctx->opts.expected_gpus; i++) {
        const ds4_v100_stage_info *s = &ctx->stages[i];
        fprintf(fp, "%d\t%d\t%d-%d\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\n",
                s->stage_id, s->gpu, s->layer_begin, s->layer_end,
                s->arena_bytes, s->scratch_bytes, s->relay_f16_bytes,
                s->relay_f32_debug_bytes, s->planned_kv_bytes,
                s->reserve_bytes, s->device_total_bytes);
    }
}
