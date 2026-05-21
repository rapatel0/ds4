#define _POSIX_C_SOURCE 200809L

/*
 * DS4 V100 appliance planning and inventory helper.
 *
 * This is deliberately not part of the runtime.  Sprint 001 needs an exact,
 * reproducible contract for source tensor inventory, static layer ownership,
 * KV admission, and first kernel-family choices before the CUDA graph is
 * refactored.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define DS4_GGUF_MAGIC 0x46554747u
#define DS4_MAX_DIMS 4

#define KiB (1024ULL)
#define MiB (1024ULL * KiB)
#define GiB (1024ULL * MiB)

enum {
    DS4_N_LAYER            = 43,
    DS4_N_EMBD             = 4096,
    DS4_N_VOCAB            = 129280,
    DS4_N_HEAD             = 64,
    DS4_N_HEAD_DIM         = 512,
    DS4_N_OUT_GROUP        = 8,
    DS4_N_LORA_Q           = 1024,
    DS4_N_LORA_O           = 1024,
    DS4_N_EXPERT           = 256,
    DS4_N_EXPERT_USED      = 6,
    DS4_N_FF_EXP           = 2048,
    DS4_N_HASH_LAYER       = 3,
    DS4_N_SWA              = 128,
    DS4_N_INDEXER_HEAD     = 64,
    DS4_N_INDEXER_HEAD_DIM = 128,
    DS4_N_HC               = 4,
    DS4_PLAN_MAX_SLOTS     = 16,
};

typedef struct {
    const char *name;
    uint64_t block_elems;
    uint64_t block_bytes;
} ggml_type_info;

static const ggml_type_info ggml_types[43] = {
    [0]  = {"f32",             1,   4},
    [1]  = {"f16",             1,   2},
    [2]  = {"q4_0",           32,  18},
    [3]  = {"q4_1",           32,  20},
    [6]  = {"q5_0",           32,  22},
    [7]  = {"q5_1",           32,  24},
    [8]  = {"q8_0",           32,  34},
    [9]  = {"q8_1",           32,  40},
    [10] = {"q2_k",          256,  84},
    [11] = {"q3_k",          256, 110},
    [12] = {"q4_k",          256, 144},
    [13] = {"q5_k",          256, 176},
    [14] = {"q6_k",          256, 210},
    [15] = {"q8_k",          256, 292},
    [16] = {"iq2_xxs",       256,  66},
    [17] = {"iq2_xs",        256,  74},
    [18] = {"iq3_xxs",       256,  98},
    [19] = {"iq1_s",         256, 110},
    [20] = {"iq4_nl",        256,  50},
    [21] = {"iq3_s",         256, 110},
    [22] = {"iq2_s",         256,  82},
    [23] = {"iq4_xs",        256, 136},
    [24] = {"i8",              1,   1},
    [25] = {"i16",             1,   2},
    [26] = {"i32",             1,   4},
    [27] = {"i64",             1,   8},
    [28] = {"f64",             1,   8},
    [29] = {"iq1_m",         256,  56},
    [30] = {"bf16",            1,   2},
    [39] = {"mxfp4",          32,  17},
    [42] = {"f8_e4m3_b128",  128, 129},
};

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

typedef struct {
    FILE *fp;
    uint64_t pos;
    char error[256];
} reader;

typedef struct {
    char *name;
    uint32_t ndim;
    uint64_t dim[DS4_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} tensor_desc;

typedef struct {
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t file_size;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    tensor_desc *tensors;
} gguf_inventory;

typedef enum {
    FAMILY_GLOBAL,
    FAMILY_CONTROL,
    FAMILY_HC,
    FAMILY_ATTENTION,
    FAMILY_COMPRESSOR,
    FAMILY_INDEXER,
    FAMILY_ROUTER,
    FAMILY_ROUTED_EXPERT,
    FAMILY_SHARED_EXPERT,
    FAMILY_OUTPUT_HEAD,
    FAMILY_MTP,
    FAMILY_UNKNOWN,
    FAMILY_COUNT,
} tensor_family;

typedef struct {
    uint64_t count;
    uint64_t bytes;
    uint64_t elements;
} bucket;

typedef struct {
    uint64_t weights;
    uint64_t kv;
    uint64_t comp_state;
    uint64_t scratch;
    uint64_t relay;
    uint64_t globals;
    uint64_t mtp;
    uint64_t reserve;
} gpu_plan;

typedef struct {
    uint64_t ctx;
    uint32_t slots;
    uint32_t gpus;
    bool mtp;
    bool json;
    uint64_t device_total_bytes;
    double reserve_gib;
    double scratch_gib;
    const char *inventory_path;
    const char *inventory_tsv;
    const char *manifest_path;
} options;

static void die(const char *msg) {
    fprintf(stderr, "ds4-v100-plan: %s\n", msg);
    exit(1);
}

static void die_errno(const char *prefix, const char *path) {
    fprintf(stderr, "ds4-v100-plan: %s: %s: %s\n", prefix, path, strerror(errno));
    exit(1);
}

static const char *type_name(uint32_t type) {
    if (type < (uint32_t)(sizeof(ggml_types) / sizeof(ggml_types[0])) &&
        ggml_types[type].name) return ggml_types[type].name;
    return "unknown";
}

static bool type_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    if (type >= (uint32_t)(sizeof(ggml_types) / sizeof(ggml_types[0]))) return false;
    const ggml_type_info *info = &ggml_types[type];
    if (!info->name || info->block_elems == 0) return false;
    const uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

static uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a) die("integer overflow in byte planner");
    return a * b;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    if (alignment == 0) return value;
    const uint64_t rem = value % alignment;
    if (rem == 0) return value;
    const uint64_t delta = alignment - rem;
    if (value > UINT64_MAX - delta) die("integer overflow while aligning GGUF tensor data");
    return value + delta;
}

static uint64_t elems2(uint64_t a, uint64_t b) {
    return checked_mul(a, b);
}

static uint64_t elems3(uint64_t a, uint64_t b, uint64_t c) {
    return checked_mul(checked_mul(a, b), c);
}

static uint64_t bytes_blocks(uint64_t elems, uint64_t block_elems, uint64_t block_bytes) {
    const uint64_t blocks = (elems + block_elems - 1) / block_elems;
    return checked_mul(blocks, block_bytes);
}

static uint64_t bytes_f16(uint64_t elems) { return checked_mul(elems, 2); }
static uint64_t bytes_f32(uint64_t elems) { return checked_mul(elems, 4); }
static uint64_t bytes_i32(uint64_t elems) { return checked_mul(elems, 4); }
static uint64_t bytes_i8(uint64_t elems)  { return elems; }
static uint64_t bytes_mxfp4(uint64_t elems) { return bytes_blocks(elems, 32, 17); }
static uint64_t bytes_f8_e4m3_b128(uint64_t elems) { return bytes_blocks(elems, 128, 129); }

static double as_gib(uint64_t bytes) {
    return (double)bytes / (double)GiB;
}

static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool read_exact(reader *r, void *dst, size_t n) {
    if (fread(dst, 1, n, r->fp) != n) {
        snprintf(r->error, sizeof(r->error), "short read at byte %" PRIu64, r->pos);
        return false;
    }
    r->pos += (uint64_t)n;
    return true;
}

static bool read_u32(reader *r, uint32_t *out) {
    uint8_t b[4];
    if (!read_exact(r, b, sizeof(b))) return false;
    *out = (uint32_t)b[0] |
           ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) |
           ((uint32_t)b[3] << 24);
    return true;
}

static bool read_u64(reader *r, uint64_t *out) {
    uint8_t b[8];
    if (!read_exact(r, b, sizeof(b))) return false;
    *out = (uint64_t)b[0] |
           ((uint64_t)b[1] << 8) |
           ((uint64_t)b[2] << 16) |
           ((uint64_t)b[3] << 24) |
           ((uint64_t)b[4] << 32) |
           ((uint64_t)b[5] << 40) |
           ((uint64_t)b[6] << 48) |
           ((uint64_t)b[7] << 56);
    return true;
}

static bool skip_bytes(reader *r, uint64_t n) {
    if (n > (uint64_t)INT64_MAX) {
        snprintf(r->error, sizeof(r->error), "skip too large");
        return false;
    }
    if (fseeko(r->fp, (off_t)n, SEEK_CUR) != 0) {
        snprintf(r->error, sizeof(r->error), "seek failed at byte %" PRIu64, r->pos);
        return false;
    }
    r->pos += n;
    return true;
}

static char *read_string(reader *r) {
    uint64_t len = 0;
    if (!read_u64(r, &len)) return NULL;
    if (len > (uint64_t)SIZE_MAX - 1) {
        snprintf(r->error, sizeof(r->error), "string too large");
        return NULL;
    }
    char *s = malloc((size_t)len + 1);
    if (!s) die("out of memory");
    if (!read_exact(r, s, (size_t)len)) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

static bool skip_value(reader *r, uint32_t type, int depth) {
    if (depth > 8) {
        snprintf(r->error, sizeof(r->error), "metadata array nesting too deep");
        return false;
    }
    const uint64_t scalar = scalar_value_size(type);
    if (scalar != 0) return skip_bytes(r, scalar);
    if (type == GGUF_VALUE_STRING) {
        char *s = read_string(r);
        if (!s) return false;
        free(s);
        return true;
    }
    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type = 0;
        uint64_t len = 0;
        if (!read_u32(r, &item_type)) return false;
        if (!read_u64(r, &len)) return false;
        const uint64_t item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                snprintf(r->error, sizeof(r->error), "metadata array too large");
                return false;
            }
            return skip_bytes(r, len * item_size);
        }
        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(r, item_type, depth + 1)) return false;
        }
        return true;
    }
    snprintf(r->error, sizeof(r->error), "unknown GGUF metadata type %u", type);
    return false;
}

static int layer_ratio(uint32_t il) {
    if (il < 2) return 0;
    return (il % 2) == 0 ? 4 : 128;
}

static int layer_device(uint32_t il, uint32_t gpus) {
    static const uint32_t starts[8] = {0, 6, 12, 18, 24, 30, 35, 40};
    static const uint32_t ends[8]   = {6, 12, 18, 24, 30, 35, 40, 43};
    if (gpus == 8) {
        for (uint32_t g = 0; g < 8; g++) {
            if (il >= starts[g] && il < ends[g]) return (int)g;
        }
        return 7;
    }
    const uint32_t per = (DS4_N_LAYER + gpus - 1) / gpus;
    uint32_t g = il / per;
    if (g >= gpus) g = gpus - 1;
    return (int)g;
}

static uint64_t layer_weight_bytes(uint32_t il, bool int8_experts) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const int ratio = layer_ratio(il);
    uint64_t b = 0;

    b += bytes_f32(elems2(hc_dim, hc_mix_dim)) + bytes_f32(3) + bytes_f32(hc_mix_dim);
    b += bytes_f32(DS4_N_EMBD);
    b += bytes_f8_e4m3_b128(elems2(DS4_N_EMBD, DS4_N_LORA_Q));
    b += bytes_f32(DS4_N_LORA_Q);
    b += bytes_f8_e4m3_b128(elems2(DS4_N_LORA_Q, q_dim));
    b += bytes_f8_e4m3_b128(elems2(DS4_N_EMBD, DS4_N_HEAD_DIM));
    b += bytes_f32(DS4_N_HEAD_DIM);
    b += bytes_f32(DS4_N_HEAD);
    b += bytes_f8_e4m3_b128(elems2(DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low_dim));
    b += bytes_f8_e4m3_b128(elems2(out_low_dim, DS4_N_EMBD));

    if (ratio != 0) {
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t comp_width = (uint64_t)coff * DS4_N_HEAD_DIM;
        b += bytes_f32(elems2(comp_width, (uint64_t)ratio));
        b += bytes_f16(elems2(DS4_N_EMBD, comp_width));
        b += bytes_f16(elems2(DS4_N_EMBD, comp_width));
        b += bytes_f32(DS4_N_HEAD_DIM);
    }
    if (ratio == 4) {
        const uint64_t index_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
        const uint64_t index_width = 2u * DS4_N_INDEXER_HEAD_DIM;
        b += bytes_f8_e4m3_b128(elems2(DS4_N_LORA_Q, index_q_dim));
        b += bytes_f16(elems2(DS4_N_EMBD, DS4_N_INDEXER_HEAD));
        b += bytes_f32(elems2(index_width, (uint64_t)ratio));
        b += bytes_f16(elems2(DS4_N_EMBD, index_width));
        b += bytes_f16(elems2(DS4_N_EMBD, index_width));
        b += bytes_f32(DS4_N_INDEXER_HEAD_DIM);
    }

    b += bytes_f32(elems2(hc_dim, hc_mix_dim)) + bytes_f32(3) + bytes_f32(hc_mix_dim);
    b += bytes_f32(DS4_N_EMBD);
    b += bytes_f32(elems2(DS4_N_EMBD, DS4_N_EXPERT));
    b += bytes_f32(DS4_N_EXPERT);

    const uint64_t gate_or_up = elems3(DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    const uint64_t down = elems3(DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
    if (int8_experts) {
        b += bytes_i8(gate_or_up) + bytes_i8(gate_or_up) + bytes_i8(down);
    } else {
        b += bytes_mxfp4(gate_or_up) + bytes_mxfp4(gate_or_up) + bytes_mxfp4(down);
    }

    b += bytes_f8_e4m3_b128(elems2(DS4_N_EMBD, DS4_N_FF_EXP));
    b += bytes_f8_e4m3_b128(elems2(DS4_N_EMBD, DS4_N_FF_EXP));
    b += bytes_f8_e4m3_b128(elems2(DS4_N_FF_EXP, DS4_N_EMBD));

    if (il < DS4_N_HASH_LAYER) {
        b += bytes_i32(elems2(DS4_N_EXPERT_USED, DS4_N_VOCAB));
    }
    return b;
}

static uint64_t global_bytes_for_gpu(uint32_t gpu, uint32_t gpus) {
    uint64_t b = 0;
    if (gpu == 0) {
        b += bytes_f16(elems2(DS4_N_EMBD, DS4_N_VOCAB));
    }
    if (gpu + 1 == gpus) {
        const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
        b += bytes_f32(elems2(hc_dim, DS4_N_HC));
        b += bytes_f32(DS4_N_HC);
        b += bytes_f32(1);
        b += bytes_f32(DS4_N_EMBD);
        b += bytes_f16(elems2(DS4_N_EMBD, DS4_N_VOCAB));
    }
    return b;
}

static uint64_t layer_kv_bytes(uint32_t il, uint64_t ctx, uint64_t elem_bytes) {
    const int ratio = layer_ratio(il);
    const uint64_t rows = (uint64_t)DS4_N_SWA + (ratio ? ctx / (uint64_t)ratio : 0);
    uint64_t b = checked_mul(checked_mul(rows, DS4_N_HEAD_DIM), elem_bytes);
    if (ratio == 4) {
        const uint64_t index_rows = ctx / 4u;
        b += checked_mul(checked_mul(index_rows, DS4_N_INDEXER_HEAD_DIM), elem_bytes);
    }
    return b;
}

static uint64_t layer_comp_state_bytes(uint32_t il) {
    const int ratio = layer_ratio(il);
    if (ratio == 4) {
        return 2ull * (2ull * DS4_N_HEAD_DIM) * (2ull * 4ull) * sizeof(float) +
               2ull * (2ull * DS4_N_INDEXER_HEAD_DIM) * (2ull * 4ull) * sizeof(float);
    }
    if (ratio == 128) {
        return 2ull * DS4_N_HEAD_DIM * 128ull * sizeof(float);
    }
    return 0;
}

static uint64_t comp_state_bytes_for_gpu(uint32_t gpu, uint32_t gpus) {
    uint64_t b = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (layer_device(il, gpus) == (int)gpu) b += layer_comp_state_bytes(il);
    }
    return b;
}

static uint64_t relay_bytes(uint32_t slots, bool debug_f32) {
    const uint64_t elem = debug_f32 ? 4 : 2;
    return checked_mul(checked_mul(checked_mul(2, slots), DS4_N_HC * DS4_N_EMBD), elem);
}

static uint64_t plan_gpu(gpu_plan *p, uint32_t gpu, const options *opt, uint64_t ctx, uint32_t slots) {
    memset(p, 0, sizeof(*p));
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (layer_device(il, opt->gpus) != (int)gpu) continue;
        p->weights += layer_weight_bytes(il, false);
        p->kv += checked_mul(layer_kv_bytes(il, ctx, 2), slots);
    }
    p->comp_state = comp_state_bytes_for_gpu(gpu, opt->gpus);
    p->scratch = (uint64_t)(opt->scratch_gib * (double)GiB);
    p->relay = relay_bytes(slots, false);
    p->globals = global_bytes_for_gpu(gpu, opt->gpus);
    p->reserve = (uint64_t)(opt->reserve_gib * (double)GiB);
    if (opt->mtp && gpu + 1 == opt->gpus) {
        p->mtp = (uint64_t)(3.6 * (double)GiB);
    }
    return p->weights + p->kv + p->comp_state + p->scratch + p->relay +
           p->globals + p->mtp + p->reserve;
}

static uint64_t plan_total_no_reserve(const gpu_plan *p) {
    return p->weights + p->kv + p->comp_state + p->scratch + p->relay + p->globals + p->mtp;
}

static uint32_t admitted_slots_for_ctx(const options *opt, uint64_t ctx, uint64_t *worst_total_out) {
    uint32_t admitted = 0;
    uint64_t last_worst = 0;
    for (uint32_t slots = 1; slots <= DS4_PLAN_MAX_SLOTS; slots++) {
        uint64_t worst = 0;
        bool ok = true;
        for (uint32_t gpu = 0; gpu < opt->gpus; gpu++) {
            gpu_plan p;
            const uint64_t total = plan_gpu(&p, gpu, opt, ctx, slots);
            if (total > opt->device_total_bytes) ok = false;
            if (total > worst) worst = total;
        }
        if (!ok) break;
        admitted = slots;
        last_worst = worst;
    }
    if (worst_total_out) *worst_total_out = last_worst;
    return admitted;
}

static uint64_t planned_headroom_after_reserve(uint64_t device_total_bytes,
                                               uint64_t no_reserve,
                                               uint64_t reserve) {
    if (no_reserve > device_total_bytes || reserve > device_total_bytes - no_reserve) return 0;
    return device_total_bytes - no_reserve - reserve;
}

static void compute_plan_worst(const options *opt,
                               uint64_t ctx,
                               uint32_t slots,
                               uint64_t *worst_total,
                               uint32_t *worst_gpu,
                               bool *fits) {
    uint64_t worst = 0;
    uint32_t gpu_at_worst = 0;
    bool ok = true;
    for (uint32_t gpu = 0; gpu < opt->gpus; gpu++) {
        gpu_plan p;
        const uint64_t total = plan_gpu(&p, gpu, opt, ctx, slots);
        if (total > opt->device_total_bytes) ok = false;
        if (total > worst) {
            worst = total;
            gpu_at_worst = gpu;
        }
    }
    if (worst_total) *worst_total = worst;
    if (worst_gpu) *worst_gpu = gpu_at_worst;
    if (fits) *fits = ok;
}

static void print_planner_json(const options *opt) {
    static const uint64_t tiers[] = { 131072ULL, 262144ULL, 524288ULL, 1048576ULL };
    static const uint32_t target_slots[] = { 1, 2, 4, 8, 16 };

    uint64_t worst = 0;
    uint32_t worst_gpu = 0;
    bool fits = false;
    compute_plan_worst(opt, opt->ctx, opt->slots, &worst, &worst_gpu, &fits);

    printf("{");
    printf("\"schema\":\"ds4_v100_planner_envelope.v1\",");
    printf("\"architecture\":\"docs/architecture/DS4-V100-LAYOUT.md\",");
    printf("\"source_stance\":\"dense_fp8_routed_mxfp4_bf16_embedding_output_f16_kv_first\",");
    printf("\"vram_bytes_per_gpu\":%" PRIu64 ",", opt->device_total_bytes);
    printf("\"configured\":{\"ctx_tokens\":%" PRIu64 ",\"slots\":%u,\"gpus\":%u,"
           "\"mtp\":%s,\"reserve_gib\":%.3f,\"scratch_gib\":%.3f,"
           "\"fits\":%s,\"worst_gpu\":%u,\"worst_total_bytes\":%" PRIu64 "},",
           opt->ctx,
           opt->slots,
           opt->gpus,
           opt->mtp ? "true" : "false",
           opt->reserve_gib,
           opt->scratch_gib,
           fits ? "true" : "false",
           worst_gpu,
           worst);

    printf("\"configured_gpus\":[");
    for (uint32_t gpu = 0; gpu < opt->gpus; gpu++) {
        gpu_plan p;
        const uint64_t total = plan_gpu(&p, gpu, opt, opt->ctx, opt->slots);
        const uint64_t no_reserve = plan_total_no_reserve(&p);
        const uint64_t headroom =
            planned_headroom_after_reserve(opt->device_total_bytes, no_reserve, p.reserve);
        if (gpu) printf(",");
        printf("{\"gpu\":%u,\"weights_bytes\":%" PRIu64 ",\"kv_bytes\":%" PRIu64
               ",\"comp_state_bytes\":%" PRIu64 ",\"scratch_bytes\":%" PRIu64
               ",\"relay_bytes\":%" PRIu64 ",\"globals_bytes\":%" PRIu64
               ",\"mtp_bytes\":%" PRIu64 ",\"reserve_bytes\":%" PRIu64
               ",\"planned_total_bytes\":%" PRIu64 ",\"headroom_after_reserve_bytes\":%" PRIu64
               ",\"fits\":%s}",
               gpu,
               p.weights,
               p.kv,
               p.comp_state,
               p.scratch,
               p.relay,
               p.globals,
               p.mtp,
               p.reserve,
               total,
               headroom,
               total <= opt->device_total_bytes ? "true" : "false");
    }
    printf("],");

    printf("\"admission_tiers\":[");
    for (uint32_t i = 0; i < sizeof(tiers) / sizeof(tiers[0]); i++) {
        uint64_t tier_worst = 0;
        const uint32_t admitted = admitted_slots_for_ctx(opt, tiers[i], &tier_worst);
        if (i) printf(",");
        printf("{\"ctx_tokens\":%" PRIu64 ",\"max_admitted_slots\":%u,"
               "\"worst_total_bytes_at_max\":%" PRIu64 "}",
               tiers[i],
               admitted,
               tier_worst);
    }
    printf("],");

    printf("\"target_matrix\":[");
    bool first = true;
    for (uint32_t i = 0; i < sizeof(tiers) / sizeof(tiers[0]); i++) {
        uint64_t tier_worst = 0;
        const uint32_t admitted = admitted_slots_for_ctx(opt, tiers[i], &tier_worst);
        (void)tier_worst;
        for (uint32_t j = 0; j < sizeof(target_slots) / sizeof(target_slots[0]); j++) {
            uint64_t matrix_worst = 0;
            uint32_t matrix_gpu = 0;
            bool matrix_fits = false;
            compute_plan_worst(opt,
                               tiers[i],
                               target_slots[j],
                               &matrix_worst,
                               &matrix_gpu,
                               &matrix_fits);
            if (!first) printf(",");
            first = false;
            printf("{\"ctx_tokens\":%" PRIu64 ",\"slots\":%u,\"fits\":%s,"
                   "\"admitted_by_tier\":%s,\"worst_gpu\":%u,"
                   "\"worst_total_bytes\":%" PRIu64 "}",
                   tiers[i],
                   target_slots[j],
                   matrix_fits ? "true" : "false",
                   target_slots[j] <= admitted ? "true" : "false",
                   matrix_gpu,
                   matrix_worst);
        }
    }
    printf("]}");
    printf("\n");
}

static void print_layer_map(const options *opt) {
    printf("\nLayer map\n");
    printf("| GPU | Layers | Layer mix | Est. weights |\n");
    printf("|---:|---|---|---:|\n");
    for (uint32_t gpu = 0; gpu < opt->gpus; gpu++) {
        int first = -1, last = -1;
        uint32_t swa = 0, r4 = 0, r128 = 0;
        uint64_t weights = 0;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (layer_device(il, opt->gpus) != (int)gpu) continue;
            if (first < 0) first = (int)il;
            last = (int)il;
            const int ratio = layer_ratio(il);
            if (ratio == 0) swa++;
            else if (ratio == 4) r4++;
            else r128++;
            weights += layer_weight_bytes(il, false);
        }
        printf("| gpu%u | ", gpu);
        if (first < 0) printf("- | ");
        else printf("%d-%d | ", first, last);
        printf("%u SWA, %u ratio-4, %u ratio-128 | %.2f GiB |\n",
               swa, r4, r128, as_gib(weights));
    }
}

static void print_kernel_policy(void) {
    printf("\nTensor-family source/runtime/kernel policy\n");
    printf("| Tensor family | Source dtype assumption | Runtime layout | First kernel family | INT8 stance |\n");
    printf("|---|---|---|---|---|\n");
    printf("| dense attention Q/KV/output | F8_E4M3_B128 | source FP8 blocked pack | FP8 dequant + FP16 HMMA dense | candidate only after scale/quality gate |\n");
    printf("| routed experts | MXFP4 / FP4 | source MXFP4 grouped pack | TurboMind sm70 grouped MXFP4 or owned grouped low-bit | candidate but may exceed VRAM |\n");
    printf("| shared expert | F8_E4M3_B128 | source FP8 dense pack | safe dense/shared-expert kernel | candidate only, unsafe single dense stays off |\n");
    printf("| router/norms/HC/control | F32/BF16/I32 | source-faithful small tensors | DS4 control kernels | only if exact enough and useful |\n");
    printf("| output head | BF16 | BF16 source-faithful on final GPU | BF16/F16 output projection first, vocab TP later | INT8/FP8 candidate only after quality gate |\n");
    printf("| KV cache | cache state | F16 first | DS4 compressed KV/attention kernels | F8 later, not INT8 default |\n");
}

static void print_manifest_schema(void) {
    printf("\nPack manifest fields for next sprint\n");
    printf("semantic_tensor_id, source_name, source_dtype, source_shape, runtime_layout,\n");
    printf("owning_gpu, layer_id, kernel_family, byte_offset, byte_length, scale_offset, checksum\n");
}

static void run_planner(const options *opt) {
    printf("DS4 V100 planner\n");
    printf("architecture: docs/architecture/DS4-V100-LAYOUT.md\n");
    printf("configured: ctx=%" PRIu64 " slots=%u gpus=%u mtp=%s device=%.2f GiB reserve=%.2f GiB scratch=%.2f GiB/GPU\n",
           opt->ctx, opt->slots, opt->gpus, opt->mtp ? "on" : "off",
           as_gib(opt->device_total_bytes), opt->reserve_gib, opt->scratch_gib);
    printf("source stance: dense FP8, routed MXFP4, BF16 embedding/output, F16 KV first\n");

    print_layer_map(opt);

    printf("\nConfigured memory plan\n");
    printf("| GPU | Weights | KV | Comp state | Scratch | Relay | Globals | MTP | Reserve | Planned total | Headroom after reserve |\n");
    printf("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    bool fits = true;
    uint64_t worst = 0;
    for (uint32_t gpu = 0; gpu < opt->gpus; gpu++) {
        gpu_plan p;
        const uint64_t total = plan_gpu(&p, gpu, opt, opt->ctx, opt->slots);
        if (total > opt->device_total_bytes) fits = false;
        if (total > worst) worst = total;
        const uint64_t no_reserve = plan_total_no_reserve(&p);
        const uint64_t headroom =
            planned_headroom_after_reserve(opt->device_total_bytes, no_reserve, p.reserve);
        printf("| gpu%u | %.2f | %.2f | %.2f | %.2f | %.3f | %.2f | %.2f | %.2f | %.2f | %.2f |\n",
               gpu,
               as_gib(p.weights), as_gib(p.kv), as_gib(p.comp_state),
               as_gib(p.scratch), as_gib(p.relay), as_gib(p.globals),
               as_gib(p.mtp), as_gib(p.reserve), as_gib(total),
               total <= opt->device_total_bytes
                   ? as_gib(headroom)
                   : -as_gib(total - opt->device_total_bytes));
    }

    printf("\nAdmission by context tier, F16 KV\n");
    printf("| Context | Max admitted slots | Worst-GPU planned total at max |\n");
    printf("|---:|---:|---:|\n");
    const uint64_t tiers[] = { 131072ULL, 262144ULL, 524288ULL, 1048576ULL };
    for (uint32_t i = 0; i < sizeof(tiers) / sizeof(tiers[0]); i++) {
        uint64_t tier_worst = 0;
        const uint32_t slots = admitted_slots_for_ctx(opt, tiers[i], &tier_worst);
        printf("| %" PRIu64 " | %u | %.2f GiB |\n", tiers[i], slots, as_gib(tier_worst));
    }

    uint64_t int8_worst = 0;
    bool int8_fits = true;
    for (uint32_t gpu = 0; gpu < opt->gpus; gpu++) {
        uint64_t int8_weights = 0;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (layer_device(il, opt->gpus) == (int)gpu) int8_weights += layer_weight_bytes(il, true);
        }
        gpu_plan p;
        const uint64_t total = plan_gpu(&p, gpu, opt, opt->ctx, opt->slots);
        const uint64_t expanded = total - p.weights + int8_weights;
        if (expanded > opt->device_total_bytes) int8_fits = false;
        if (expanded > int8_worst) int8_worst = expanded;
    }
    printf("\nINT8-expanded routed-expert warning: worst configured GPU would be %.2f GiB; %s.\n",
           as_gib(int8_worst),
           int8_fits ? "fits this configured plan, but still requires quality gates"
                     : "does not fit with current reserve and should not be a default");

    print_kernel_policy();
    print_manifest_schema();

    printf("\nVerdict: %s\n", fits ? "SHIP-ready planner baseline" : "STOP for configured overfill");
    printf("Worst configured GPU total including reserve: %.2f GiB / %.2f GiB\n",
           as_gib(worst),
           as_gib(opt->device_total_bytes));
}

static const char *family_name(tensor_family family) {
    switch (family) {
    case FAMILY_GLOBAL: return "global";
    case FAMILY_CONTROL: return "control";
    case FAMILY_HC: return "hc";
    case FAMILY_ATTENTION: return "attention";
    case FAMILY_COMPRESSOR: return "compressor";
    case FAMILY_INDEXER: return "indexer";
    case FAMILY_ROUTER: return "router";
    case FAMILY_ROUTED_EXPERT: return "routed_expert";
    case FAMILY_SHARED_EXPERT: return "shared_expert";
    case FAMILY_OUTPUT_HEAD: return "output_head";
    case FAMILY_MTP: return "mtp";
    case FAMILY_UNKNOWN: return "unknown";
    default: return "unknown";
    }
}

static tensor_family classify_tensor(const char *name) {
    if (strstr(name, "mtp") || strstr(name, "draft")) return FAMILY_MTP;
    if (strcmp(name, "output.weight") == 0) return FAMILY_OUTPUT_HEAD;
    if (strncmp(name, "output_", 7) == 0 ||
        strcmp(name, "token_embd.weight") == 0 ||
        strcmp(name, "output_norm.weight") == 0) return FAMILY_GLOBAL;
    if (strstr(name, "hc_attn") || strstr(name, "hc_ffn") || strstr(name, "hc_")) return FAMILY_HC;
    if (strstr(name, "attn_compress")) return FAMILY_COMPRESSOR;
    if (strstr(name, "ffn_gate_exps") ||
        strstr(name, "ffn_up_exps") ||
        strstr(name, "ffn_down_exps")) return FAMILY_ROUTED_EXPERT;
    if (strstr(name, "ffn_gate_shexp") ||
        strstr(name, "ffn_up_shexp") ||
        strstr(name, "ffn_down_shexp")) return FAMILY_SHARED_EXPERT;
    if (strstr(name, "ffn_norm")) return FAMILY_CONTROL;
    if (strstr(name, "ffn_gate_inp") ||
        strstr(name, "exp_probs") ||
        strstr(name, "tid2eid")) return FAMILY_ROUTER;
    if (strstr(name, "indexer")) return FAMILY_INDEXER;
    if (strstr(name, "compressor")) return FAMILY_COMPRESSOR;
    if (strstr(name, "attn_")) return FAMILY_ATTENTION;
    return FAMILY_UNKNOWN;
}

static int tensor_layer(const char *name) {
    int layer = -1;
    if (sscanf(name, "blk.%d.", &layer) == 1) return layer;
    return -1;
}

static void dims_string(const tensor_desc *t, char *buf, size_t len) {
    size_t off = 0;
    int n = snprintf(buf + off, len - off, "[");
    if (n < 0) return;
    off += (size_t)n;
    for (uint32_t i = 0; i < t->ndim; i++) {
        n = snprintf(buf + off, len - off, "%s%" PRIu64, i ? "x" : "", t->dim[i]);
        if (n < 0) return;
        off += (size_t)n;
        if (off >= len) break;
    }
    if (off < len) snprintf(buf + off, len - off, "]");
}

static void parse_gguf(const char *path, gguf_inventory *inv) {
    memset(inv, 0, sizeof(*inv));
    inv->alignment = 32;
    struct stat st;
    if (stat(path, &st) != 0) die_errno("cannot stat inventory model", path);
    inv->file_size = (uint64_t)st.st_size;

    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("cannot open inventory model", path);
    reader r = {.fp = fp};

    uint32_t magic = 0;
    if (!read_u32(&r, &magic)) die(r.error);
    if (magic != DS4_GGUF_MAGIC) die("inventory model is not a GGUF file");
    if (!read_u32(&r, &inv->version)) die(r.error);
    if (!read_u64(&r, &inv->n_tensors)) die(r.error);
    if (!read_u64(&r, &inv->n_kv)) die(r.error);
    if (inv->version != 3) die("only GGUF v3 inventory is supported");

    for (uint64_t i = 0; i < inv->n_kv; i++) {
        char *key = read_string(&r);
        if (!key) die(r.error);
        uint32_t type = 0;
        if (!read_u32(&r, &type)) die(r.error);
        if (!strcmp(key, "general.alignment") && type == GGUF_VALUE_UINT32) {
            uint32_t alignment = 0;
            if (!read_u32(&r, &alignment)) die(r.error);
            if (alignment != 0) inv->alignment = alignment;
            free(key);
            continue;
        }
        free(key);
        if (!skip_value(&r, type, 0)) die(r.error);
    }

    inv->tensors = calloc((size_t)inv->n_tensors, sizeof(inv->tensors[0]));
    if (!inv->tensors) die("out of memory allocating inventory tensor table");
    for (uint64_t i = 0; i < inv->n_tensors; i++) {
        tensor_desc *t = &inv->tensors[i];
        t->name = read_string(&r);
        if (!t->name) die(r.error);
        if (!read_u32(&r, &t->ndim)) die(r.error);
        if (t->ndim == 0 || t->ndim > DS4_MAX_DIMS) die("unsupported tensor rank in inventory");
        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!read_u64(&r, &t->dim[d])) die(r.error);
            t->elements = checked_mul(t->elements, t->dim[d]);
        }
        if (!read_u32(&r, &t->type)) die(r.error);
        if (!read_u64(&r, &t->rel_offset)) die(r.error);
        if (!type_nbytes(t->type, t->elements, &t->bytes)) {
            t->bytes = 0;
        }
    }
    inv->tensor_data_pos = align_up(r.pos, inv->alignment);
    for (uint64_t i = 0; i < inv->n_tensors; i++) {
        tensor_desc *t = &inv->tensors[i];
        if (t->rel_offset > UINT64_MAX - inv->tensor_data_pos) die("GGUF tensor offset overflow");
        t->abs_offset = inv->tensor_data_pos + t->rel_offset;
        if (t->bytes != 0 &&
            (t->abs_offset > inv->file_size || t->bytes > inv->file_size - t->abs_offset)) {
            die("GGUF tensor points outside file");
        }
    }
    fclose(fp);
}

static void free_inventory(gguf_inventory *inv) {
    if (!inv->tensors) return;
    for (uint64_t i = 0; i < inv->n_tensors; i++) free(inv->tensors[i].name);
    free(inv->tensors);
    inv->tensors = NULL;
}

static void write_inventory_tsv(const gguf_inventory *inv, const char *path, uint32_t gpus) {
    FILE *fp = fopen(path, "w");
    if (!fp) die_errno("cannot write inventory TSV", path);
    fprintf(fp, "name\tlayer\tgpu\tfamily\tdims\telements\tggml_type_id\tggml_type\tbytes\n");
    for (uint64_t i = 0; i < inv->n_tensors; i++) {
        const tensor_desc *t = &inv->tensors[i];
        const int layer = tensor_layer(t->name);
        const int gpu = layer >= 0 ? layer_device((uint32_t)layer, gpus) : -1;
        char dims[128];
        dims_string(t, dims, sizeof(dims));
        fprintf(fp, "%s\t%d\t%d\t%s\t%s\t%" PRIu64 "\t%u\t%s\t%" PRIu64 "\n",
                t->name, layer, gpu, family_name(classify_tensor(t->name)), dims,
                t->elements, t->type, type_name(t->type), t->bytes);
    }
    if (fclose(fp) != 0) die_errno("cannot close inventory TSV", path);
}

static int owning_gpu_for_tensor(const char *name, uint32_t gpus) {
    const int layer = tensor_layer(name);
    if (layer >= 0) return layer_device((uint32_t)layer, gpus);
    if (!strcmp(name, "token_embd.weight")) return 0;
    if (!strcmp(name, "output.weight") ||
        !strcmp(name, "output_norm.weight") ||
        !strcmp(name, "hc_head_base") ||
        !strcmp(name, "hc_head_fn") ||
        !strcmp(name, "hc_head_scale")) {
        return (int)gpus - 1;
    }
    return -1;
}

static const char *runtime_layout_for_tensor(const tensor_desc *t) {
    const tensor_family family = classify_tensor(t->name);
    switch (family) {
    case FAMILY_ROUTED_EXPERT:
        return t->type == 39 ? "source_mxfp4_grouped" : "unsupported_routed_source";
    case FAMILY_ATTENTION:
    case FAMILY_SHARED_EXPERT:
        return t->type == 42 ? "source_f8_e4m3_b128_blocked" : "source_f32_control";
    case FAMILY_COMPRESSOR:
    case FAMILY_INDEXER:
        if (t->type == 42) return "source_f8_e4m3_b128_blocked";
        if (t->type == 30) return "source_bf16";
        if (t->type == 0) return "source_f32";
        return "source_mixed_unknown";
    case FAMILY_OUTPUT_HEAD:
        return t->type == 30 ? "source_bf16" : "source_output_unknown";
    case FAMILY_HC:
    case FAMILY_CONTROL:
    case FAMILY_ROUTER:
    case FAMILY_GLOBAL:
        if (t->type == 30) return "source_bf16";
        if (t->type == 26) return "source_i32";
        if (t->type == 0) return "source_f32";
        return "source_control_unknown";
    case FAMILY_MTP:
        return "mtp_deferred";
    case FAMILY_UNKNOWN:
    case FAMILY_COUNT:
    default:
        return "unknown";
    }
}

static const char *kernel_family_for_tensor(const tensor_desc *t) {
    const tensor_family family = classify_tensor(t->name);
    switch (family) {
    case FAMILY_ROUTED_EXPERT:
        return "v100_grouped_mxfp4_pending";
    case FAMILY_ATTENTION:
        return t->type == 42 ? "v100_fp8_dequant_f16_hmma_pending" : "ds4_attention_control";
    case FAMILY_COMPRESSOR:
        return "ds4_compressor_bf16_f32_pending";
    case FAMILY_INDEXER:
        return "ds4_indexer_mixed_pending";
    case FAMILY_SHARED_EXPERT:
        return "v100_shared_fp8_dense_pending";
    case FAMILY_OUTPUT_HEAD:
        return "v100_bf16_output_pending";
    case FAMILY_HC:
        return "ds4_hc_control_f32";
    case FAMILY_ROUTER:
        return "ds4_router_f32_i32";
    case FAMILY_CONTROL:
        return "ds4_control_f32";
    case FAMILY_GLOBAL:
        if (!strcmp(t->name, "token_embd.weight")) return "ds4_embedding_bf16";
        return "ds4_global_control";
    case FAMILY_MTP:
        return "mtp_deferred";
    case FAMILY_UNKNOWN:
    case FAMILY_COUNT:
    default:
        return "unknown";
    }
}

static void write_manifest_tsv(const gguf_inventory *inv, const char *path, uint32_t gpus) {
    FILE *fp = fopen(path, "w");
    if (!fp) die_errno("cannot write manifest", path);
    fprintf(fp,
            "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\t"
            "runtime_layout\towning_gpu\tlayer_id\tkernel_family\t"
            "byte_offset\tbyte_length\tscale_offset\tchecksum\tbyte_offset_basis\n");
    for (uint64_t i = 0; i < inv->n_tensors; i++) {
        const tensor_desc *t = &inv->tensors[i];
        char dims[128];
        dims_string(t, dims, sizeof(dims));
        const int layer = tensor_layer(t->name);
        const int gpu = owning_gpu_for_tensor(t->name, gpus);
        fprintf(fp,
                "%s\t%s\t%s\t%s\t%s\t%d\t%d\t%s\t%" PRIu64 "\t%" PRIu64 "\t-1\tpending\tabsolute_gguf_file\n",
                t->name,
                t->name,
                type_name(t->type),
                dims,
                runtime_layout_for_tensor(t),
                gpu,
                layer,
                kernel_family_for_tensor(t),
                t->abs_offset,
                t->bytes);
    }
    if (fclose(fp) != 0) die_errno("cannot close manifest", path);
}

static void run_inventory(const char *path, const char *tsv_path, const char *manifest_path, uint32_t gpus) {
    gguf_inventory inv;
    parse_gguf(path, &inv);

    bucket type_buckets[43] = {{0}};
    bucket family_buckets[FAMILY_COUNT] = {{0}};
    uint64_t described = 0;
    uint64_t unknown_type_count = 0;
    for (uint64_t i = 0; i < inv.n_tensors; i++) {
        const tensor_desc *t = &inv.tensors[i];
        if (t->type < 43 && ggml_types[t->type].name) {
            type_buckets[t->type].count++;
            type_buckets[t->type].bytes += t->bytes;
            type_buckets[t->type].elements += t->elements;
        } else {
            unknown_type_count++;
        }
        const tensor_family family = classify_tensor(t->name);
        family_buckets[family].count++;
        family_buckets[family].bytes += t->bytes;
        family_buckets[family].elements += t->elements;
        described += t->bytes;
    }

    printf("\nGGUF inventory\n");
    printf("path: %s\n", path);
    printf("file size: %.2f GiB (%" PRIu64 " bytes)\n", as_gib(inv.file_size), inv.file_size);
    printf("gguf: v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors\n",
           inv.version, inv.n_kv, inv.n_tensors);
    printf("tensor bytes described: %.2f GiB\n", as_gib(described));
    if (unknown_type_count) printf("unknown tensor types: %" PRIu64 "\n", unknown_type_count);

    printf("\nTensor types\n");
    printf("| Type id | Type | Count | Bytes |\n");
    printf("|---:|---|---:|---:|\n");
    for (uint32_t type = 0; type < 43; type++) {
        if (!type_buckets[type].count) continue;
        printf("| %u | %s | %" PRIu64 " | %.2f GiB |\n",
               type, type_name(type), type_buckets[type].count, as_gib(type_buckets[type].bytes));
    }

    printf("\nTensor families\n");
    printf("| Family | Count | Bytes |\n");
    printf("|---|---:|---:|\n");
    for (uint32_t family = 0; family < FAMILY_COUNT; family++) {
        if (!family_buckets[family].count) continue;
        printf("| %s | %" PRIu64 " | %.2f GiB |\n",
               family_name((tensor_family)family),
               family_buckets[family].count,
               as_gib(family_buckets[family].bytes));
    }

    printf("\nArchitecture reconciliation hints\n");
    printf("- Type 39 mxfp4 count: %" PRIu64 "\n", type_buckets[39].count);
    printf("- Type 42 f8_e4m3_b128 count: %" PRIu64 "\n", type_buckets[42].count);
    printf("- Routed expert family bytes: %.2f GiB\n", as_gib(family_buckets[FAMILY_ROUTED_EXPERT].bytes));
    printf("- Dense attention family bytes: %.2f GiB\n", as_gib(family_buckets[FAMILY_ATTENTION].bytes));
    printf("- Unknown family count: %" PRIu64 "\n", family_buckets[FAMILY_UNKNOWN].count);

    if (tsv_path) {
        write_inventory_tsv(&inv, tsv_path, gpus);
        printf("\nWrote tensor TSV: %s\n", tsv_path);
    }
    if (manifest_path) {
        write_manifest_tsv(&inv, manifest_path, gpus);
        printf("Wrote pack manifest TSV: %s\n", manifest_path);
    }

    free_inventory(&inv);
}

static uint64_t parse_u64_arg(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    const unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(2);
    }
    return (uint64_t)v;
}

static double parse_double_arg(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    const double v = strtod(s, &end);
    if (errno || !end || *end) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(2);
    }
    return v;
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-v100-plan [options]\n"
        "\n"
        "Planner options:\n"
        "  --ctx N                 Context tokens for configured plan. Default: 262144\n"
        "  --slots N               Configured slots for plan. Default: 4\n"
        "  --gpus N                Visible GPUs. Default: 8\n"
        "  --mtp on|off            Include rough MTP bytes on final GPU. Default: off\n"
        "  --device-total-bytes N   Per-GPU memory capacity. Default: 32 GiB\n"
        "  --reserve-gib F         Reserve per GPU after planned allocations. Default: 4.0\n"
        "  --scratch-gib F         Scratch per GPU. Default: 1.0\n"
        "  --json                  Emit machine-readable slot/context envelope JSON\n"
        "\n"
        "Inventory options:\n"
        "  --inventory FILE        Parse GGUF tensor directory and print dtype/family summary\n"
        "  --inventory-tsv FILE    Write full tensor inventory TSV\n"
        "  --manifest FILE         Write first-pass pack manifest TSV\n"
        "\n"
        "Examples:\n"
        "  ds4-v100-plan --ctx 262144 --slots 4 --gpus 8 --mtp off\n"
        "  ds4-v100-plan --inventory /models/DSv4-Flash-256e-fixed.gguf --manifest /tmp/ds4-manifest.tsv\n");
}

int main(int argc, char **argv) {
    options opt = {
        .ctx = 262144,
        .slots = 4,
        .gpus = 8,
        .mtp = false,
        .device_total_bytes = 32ULL * GiB,
        .reserve_gib = 4.0,
        .scratch_gib = 1.0,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(arg, "--ctx") && i + 1 < argc) {
            opt.ctx = parse_u64_arg(argv[++i], "--ctx");
        } else if (!strcmp(arg, "--slots") && i + 1 < argc) {
            opt.slots = (uint32_t)parse_u64_arg(argv[++i], "--slots");
        } else if (!strcmp(arg, "--gpus") && i + 1 < argc) {
            opt.gpus = (uint32_t)parse_u64_arg(argv[++i], "--gpus");
        } else if (!strcmp(arg, "--mtp") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "on")) opt.mtp = true;
            else if (!strcmp(v, "off")) opt.mtp = false;
            else die("--mtp must be on or off");
        } else if (!strcmp(arg, "--device-total-bytes") && i + 1 < argc) {
            opt.device_total_bytes = parse_u64_arg(argv[++i], "--device-total-bytes");
        } else if (!strcmp(arg, "--reserve-gib") && i + 1 < argc) {
            opt.reserve_gib = parse_double_arg(argv[++i], "--reserve-gib");
        } else if (!strcmp(arg, "--scratch-gib") && i + 1 < argc) {
            opt.scratch_gib = parse_double_arg(argv[++i], "--scratch-gib");
        } else if (!strcmp(arg, "--json")) {
            opt.json = true;
        } else if (!strcmp(arg, "--inventory") && i + 1 < argc) {
            opt.inventory_path = argv[++i];
        } else if (!strcmp(arg, "--inventory-tsv") && i + 1 < argc) {
            opt.inventory_tsv = argv[++i];
        } else if (!strcmp(arg, "--manifest") && i + 1 < argc) {
            opt.manifest_path = argv[++i];
        } else {
            usage(stderr);
            return 2;
        }
    }

    if (opt.gpus == 0 || opt.gpus > 8) die("--gpus must be in 1..8");
    if (opt.slots == 0) die("--slots must be positive");
    if (opt.device_total_bytes == 0) die("--device-total-bytes must be positive");
    if (opt.reserve_gib < 0.0 || opt.scratch_gib < 0.0) die("reserve/scratch must be non-negative");
    if ((opt.inventory_tsv || opt.manifest_path) && !opt.inventory_path) {
        die("--inventory-tsv and --manifest require --inventory");
    }
    if (opt.json && opt.inventory_path) {
        die("--json cannot be combined with --inventory");
    }

    if (opt.json) {
        print_planner_json(&opt);
    } else {
        run_planner(&opt);
    }
    if (opt.inventory_path) {
        run_inventory(opt.inventory_path, opt.inventory_tsv, opt.manifest_path, opt.gpus);
    }
    return 0;
}
