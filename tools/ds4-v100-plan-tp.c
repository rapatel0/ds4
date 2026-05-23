#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    DS4_PLAN_MAX_SLOTS     = 256,
};

typedef enum {
    TOPO_PP8_TP1,
    TOPO_PP4_TP2,
    TOPO_PP2_TP4,
    TOPO_PP1_TP8,
    TOPO_COUNT,
} topology_id;

typedef enum {
    KV_F16,
    KV_F8_E4M3_B128,
    KV_Q8_0,
} kv_dtype;

typedef struct {
    uint64_t ctx;
    uint32_t slots;
    uint32_t gpus;
    uint64_t device_total_bytes;
    double reserve_gib;
    double scratch_gib;
    bool mtp;
    bool kv_sharded;
    kv_dtype kv;
    int topology_filter;
} options;

typedef struct {
    uint64_t weights;
    uint64_t kv;
    uint64_t comp_state;
    uint64_t scratch;
    uint64_t relay;
    uint64_t globals;
    uint64_t collectives;
    uint64_t mtp;
    uint64_t reserve;
} gpu_plan;

static void die(const char *msg) {
    fprintf(stderr, "ds4-v100-plan-tp: %s\n", msg);
    exit(1);
}

static uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a) die("integer overflow in byte planner");
    return a * b;
}

static uint64_t ceil_div_u64(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

static uint64_t elems2(uint64_t a, uint64_t b) {
    return checked_mul(a, b);
}

static uint64_t elems3(uint64_t a, uint64_t b, uint64_t c) {
    return checked_mul(checked_mul(a, b), c);
}

static uint64_t bytes_blocks(uint64_t elems, uint64_t block_elems, uint64_t block_bytes) {
    return checked_mul((elems + block_elems - 1) / block_elems, block_bytes);
}

static uint64_t bytes_f16(uint64_t elems) { return checked_mul(elems, 2); }
static uint64_t bytes_f32(uint64_t elems) { return checked_mul(elems, 4); }
static uint64_t bytes_i32(uint64_t elems) { return checked_mul(elems, 4); }
static uint64_t bytes_mxfp4(uint64_t elems) { return bytes_blocks(elems, 32, 17); }
static uint64_t bytes_f8_e4m3_b128(uint64_t elems) { return bytes_blocks(elems, 128, 129); }

static double as_gib(uint64_t bytes) {
    return (double)bytes / (double)GiB;
}

static double as_mib(uint64_t bytes) {
    return (double)bytes / (double)MiB;
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

static int layer_ratio(uint32_t il) {
    if (il < 2) return 0;
    return (il % 2) == 0 ? 4 : 128;
}

static const char *topology_name(topology_id topo) {
    switch (topo) {
    case TOPO_PP8_TP1: return "pp8_tp1_layer";
    case TOPO_PP4_TP2: return "pp4_tp2";
    case TOPO_PP2_TP4: return "pp2_tp4";
    case TOPO_PP1_TP8: return "pp1_tp8";
    default: return "unknown";
    }
}

static const char *kv_name(kv_dtype kv) {
    switch (kv) {
    case KV_F16: return "f16";
    case KV_F8_E4M3_B128: return "f8_e4m3_b128";
    case KV_Q8_0: return "q8_0";
    default: return "unknown";
    }
}

static uint32_t topo_tp(topology_id topo) {
    switch (topo) {
    case TOPO_PP8_TP1: return 1;
    case TOPO_PP4_TP2: return 2;
    case TOPO_PP2_TP4: return 4;
    case TOPO_PP1_TP8: return 8;
    default: return 1;
    }
}

static uint32_t topo_pp(topology_id topo) {
    switch (topo) {
    case TOPO_PP8_TP1: return 8;
    case TOPO_PP4_TP2: return 4;
    case TOPO_PP2_TP4: return 2;
    case TOPO_PP1_TP8: return 1;
    default: return 8;
    }
}

static bool stage_layer_range(topology_id topo, uint32_t stage, uint32_t *first, uint32_t *end) {
    if (topo == TOPO_PP8_TP1) {
        static const uint32_t starts[8] = {0, 6, 12, 18, 24, 30, 35, 40};
        static const uint32_t ends[8]   = {6, 12, 18, 24, 30, 35, 40, 43};
        if (stage >= 8) return false;
        *first = starts[stage];
        *end = ends[stage];
        return true;
    }
    if (topo == TOPO_PP4_TP2) {
        static const uint32_t starts[4] = {0, 11, 22, 33};
        static const uint32_t ends[4]   = {11, 22, 33, 43};
        if (stage >= 4) return false;
        *first = starts[stage];
        *end = ends[stage];
        return true;
    }
    if (topo == TOPO_PP2_TP4) {
        static const uint32_t starts[2] = {0, 22};
        static const uint32_t ends[2]   = {22, 43};
        if (stage >= 2) return false;
        *first = starts[stage];
        *end = ends[stage];
        return true;
    }
    if (topo == TOPO_PP1_TP8) {
        if (stage != 0) return false;
        *first = 0;
        *end = DS4_N_LAYER;
        return true;
    }
    return false;
}

static int layer_stage(topology_id topo, uint32_t il) {
    const uint32_t pp = topo_pp(topo);
    for (uint32_t stage = 0; stage < pp; stage++) {
        uint32_t first = 0, end = 0;
        if (!stage_layer_range(topo, stage, &first, &end)) continue;
        if (il >= first && il < end) return (int)stage;
    }
    return -1;
}

static bool gpu_in_stage(topology_id topo, uint32_t gpu, uint32_t stage) {
    const uint32_t tp = topo_tp(topo);
    const uint32_t first_gpu = stage * tp;
    return gpu >= first_gpu && gpu < first_gpu + tp;
}

static bool gpu_owns_layer(topology_id topo, uint32_t gpu, uint32_t il) {
    const int stage = layer_stage(topo, il);
    return stage >= 0 && gpu_in_stage(topo, gpu, (uint32_t)stage);
}

static uint64_t layer_total_weight_bytes(uint32_t il) {
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

    b += bytes_mxfp4(elems3(DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT));
    b += bytes_mxfp4(elems3(DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT));
    b += bytes_mxfp4(elems3(DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT));

    b += bytes_f8_e4m3_b128(elems2(DS4_N_EMBD, DS4_N_FF_EXP));
    b += bytes_f8_e4m3_b128(elems2(DS4_N_EMBD, DS4_N_FF_EXP));
    b += bytes_f8_e4m3_b128(elems2(DS4_N_FF_EXP, DS4_N_EMBD));

    if (il < DS4_N_HASH_LAYER) b += bytes_i32(elems2(DS4_N_EXPERT_USED, DS4_N_VOCAB));
    return b;
}

static uint64_t layer_replicated_weight_bytes(uint32_t il) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    uint64_t b = 0;

    b += 2u * (bytes_f32(elems2(hc_dim, hc_mix_dim)) + bytes_f32(3) + bytes_f32(hc_mix_dim));
    b += 2u * bytes_f32(DS4_N_EMBD);
    b += bytes_f32(elems2(DS4_N_EMBD, DS4_N_EXPERT));
    b += bytes_f32(DS4_N_EXPERT);
    if (il < DS4_N_HASH_LAYER) b += bytes_i32(elems2(DS4_N_EXPERT_USED, DS4_N_VOCAB));
    return b;
}

static uint64_t layer_gpu_weight_bytes(topology_id topo, uint32_t il) {
    const uint64_t total = layer_total_weight_bytes(il);
    const uint32_t tp = topo_tp(topo);
    if (tp == 1) return total;
    uint64_t replicated = layer_replicated_weight_bytes(il);
    if (replicated > total) replicated = total;
    return replicated + ceil_div_u64(total - replicated, tp);
}

static uint64_t kv_values_bytes(uint64_t values, kv_dtype kv) {
    switch (kv) {
    case KV_F16: return bytes_f16(values);
    case KV_F8_E4M3_B128: return bytes_f8_e4m3_b128(values);
    case KV_Q8_0: return bytes_blocks(values, 32, 34);
    default: return bytes_f16(values);
    }
}

static uint64_t layer_kv_bytes(uint32_t il, uint64_t ctx, kv_dtype kv) {
    const int ratio = layer_ratio(il);
    const uint64_t rows = (uint64_t)DS4_N_SWA + (ratio ? ctx / (uint64_t)ratio : 0);
    uint64_t b = kv_values_bytes(elems2(rows, DS4_N_HEAD_DIM), kv);
    if (ratio == 4) {
        b += kv_values_bytes(elems2(ctx / 4u, DS4_N_INDEXER_HEAD_DIM), kv);
    }
    return b;
}

static uint64_t layer_comp_state_bytes(uint32_t il) {
    const int ratio = layer_ratio(il);
    if (ratio == 4) {
        return 2ull * (2ull * DS4_N_HEAD_DIM) * (2ull * 4ull) * sizeof(float) +
               2ull * (2ull * DS4_N_INDEXER_HEAD_DIM) * (2ull * 4ull) * sizeof(float);
    }
    if (ratio == 128) return 2ull * DS4_N_HEAD_DIM * 128ull * sizeof(float);
    return 0;
}

static uint64_t global_bytes_for_gpu(topology_id topo, uint32_t gpu) {
    const uint64_t embed = bytes_f16(elems2(DS4_N_EMBD, DS4_N_VOCAB));
    const uint64_t output = bytes_f16(elems2(DS4_N_EMBD, DS4_N_VOCAB));
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t output_control =
        bytes_f32(elems2(hc_dim, DS4_N_HC)) +
        bytes_f32(DS4_N_HC) +
        bytes_f32(1) +
        bytes_f32(DS4_N_EMBD);
    const uint32_t tp = topo_tp(topo);

    uint64_t b = 0;
    if (topo == TOPO_PP8_TP1) {
        if (gpu == 0) b += embed;
        if (gpu == 7) b += output_control + output;
        return b;
    }
    if (gpu_in_stage(topo, gpu, 0)) b += ceil_div_u64(embed, tp);
    if (gpu_in_stage(topo, gpu, topo_pp(topo) - 1)) b += output_control + ceil_div_u64(output, tp);
    return b;
}

static uint64_t relay_bytes(topology_id topo, uint32_t slots) {
    const uint32_t pp = topo_pp(topo);
    if (pp <= 1) return 0;
    return checked_mul(checked_mul(2, slots), DS4_N_HC * DS4_N_EMBD * 2ull);
}

static uint64_t collective_scratch_bytes(topology_id topo, uint32_t slots) {
    const uint32_t tp = topo_tp(topo);
    if (tp <= 1) return 0;
    return checked_mul(checked_mul(2, slots), DS4_N_EMBD * 2ull);
}

static uint64_t collective_wire_bytes_per_step(topology_id topo, uint32_t slots) {
    const uint32_t tp = topo_tp(topo);
    if (tp <= 1) return 0;
    const uint64_t payload = checked_mul(slots, DS4_N_EMBD * 2ull);
    const uint64_t reductions_per_layer = 2;
    const uint64_t ring_factor_num = 2ull * (tp - 1ull);
    return checked_mul(checked_mul(payload, reductions_per_layer), DS4_N_LAYER) *
           ring_factor_num / tp;
}

static uint64_t stage_relay_wire_bytes_per_step(topology_id topo, uint32_t slots) {
    const uint32_t pp = topo_pp(topo);
    if (pp <= 1) return 0;
    const uint64_t payload = checked_mul(slots, DS4_N_HC * DS4_N_EMBD * 2ull);
    return checked_mul(payload, pp - 1u);
}

static uint64_t plan_gpu(gpu_plan *p, const options *opt, topology_id topo, uint32_t gpu,
                         uint64_t ctx, uint32_t slots) {
    memset(p, 0, sizeof(*p));
    const uint32_t tp = topo_tp(topo);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!gpu_owns_layer(topo, gpu, il)) continue;
        p->weights += layer_gpu_weight_bytes(topo, il);
        uint64_t kv = layer_kv_bytes(il, ctx, opt->kv);
        uint64_t comp = layer_comp_state_bytes(il);
        if (tp > 1 && opt->kv_sharded) {
            kv = ceil_div_u64(kv, tp);
            comp = ceil_div_u64(comp, tp);
        }
        p->kv += checked_mul(kv, slots);
        p->comp_state += comp;
    }
    p->scratch = (uint64_t)(opt->scratch_gib * (double)GiB);
    p->relay = relay_bytes(topo, slots);
    p->globals = global_bytes_for_gpu(topo, gpu);
    p->collectives = collective_scratch_bytes(topo, slots);
    p->reserve = (uint64_t)(opt->reserve_gib * (double)GiB);
    if (opt->mtp && topo_pp(topo) > 0 && gpu_in_stage(topo, gpu, topo_pp(topo) - 1)) {
        p->mtp = ceil_div_u64((uint64_t)(3.6 * (double)GiB), tp);
    }
    return p->weights + p->kv + p->comp_state + p->scratch + p->relay +
           p->globals + p->collectives + p->mtp + p->reserve;
}

static uint64_t no_reserve_total(const gpu_plan *p) {
    return p->weights + p->kv + p->comp_state + p->scratch + p->relay +
           p->globals + p->collectives + p->mtp;
}

static void compute_worst(const options *opt, topology_id topo, uint64_t ctx, uint32_t slots,
                          uint64_t *worst_total, uint32_t *worst_gpu, bool *fits) {
    uint64_t worst = 0;
    uint32_t gpu_at_worst = 0;
    bool ok = true;
    for (uint32_t gpu = 0; gpu < opt->gpus; gpu++) {
        gpu_plan p;
        const uint64_t total = plan_gpu(&p, opt, topo, gpu, ctx, slots);
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

static uint32_t admitted_slots(const options *opt, topology_id topo, uint64_t ctx,
                               uint64_t *worst_total_at_max) {
    uint32_t admitted = 0;
    uint64_t last_worst = 0;
    for (uint32_t slots = 1; slots <= DS4_PLAN_MAX_SLOTS; slots++) {
        uint64_t worst = 0;
        bool fits = false;
        compute_worst(opt, topo, ctx, slots, &worst, NULL, &fits);
        if (!fits) break;
        admitted = slots;
        last_worst = worst;
    }
    if (worst_total_at_max) *worst_total_at_max = last_worst;
    return admitted;
}

static void print_stage_map(topology_id topo) {
    printf("  stage map:");
    for (uint32_t stage = 0; stage < topo_pp(topo); stage++) {
        uint32_t first = 0, end = 0;
        stage_layer_range(topo, stage, &first, &end);
        printf(" s%u[gpu%u", stage, stage * topo_tp(topo));
        if (topo_tp(topo) > 1) printf("-gpu%u", stage * topo_tp(topo) + topo_tp(topo) - 1);
        printf(":L%u-%u]", first, end - 1);
    }
    printf("\n");
}

static void print_topology_plan(const options *opt, topology_id topo) {
    uint64_t worst = 0;
    uint32_t worst_gpu = 0;
    bool fits = false;
    compute_worst(opt, topo, opt->ctx, opt->slots, &worst, &worst_gpu, &fits);

    printf("\n## %s\n", topology_name(topo));
    print_stage_map(topo);
    printf("  tp=%u pp=%u kv=%s kv_%s slots=%u ctx=%" PRIu64 "\n",
           topo_tp(topo), topo_pp(topo), kv_name(opt->kv),
           opt->kv_sharded ? "sharded" : "replicated", opt->slots, opt->ctx);
    printf("  configured verdict: %s; worst gpu%u %.2f / %.2f GiB\n",
           fits ? "fits" : "over budget", worst_gpu, as_gib(worst), as_gib(opt->device_total_bytes));
    printf("  estimated TP collective wire per decode step: %.2f MiB\n",
           as_mib(collective_wire_bytes_per_step(topo, opt->slots)));
    printf("  estimated pipeline relay wire per decode step: %.2f MiB\n",
           as_mib(stage_relay_wire_bytes_per_step(topo, opt->slots)));

    printf("\n| GPU | Weights | KV | Comp | Scratch | Relay | Globals | Coll. scratch | MTP | Reserve | Total | Headroom |\n");
    printf("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (uint32_t gpu = 0; gpu < opt->gpus; gpu++) {
        gpu_plan p;
        const uint64_t total = plan_gpu(&p, opt, topo, gpu, opt->ctx, opt->slots);
        const uint64_t no_reserve = no_reserve_total(&p);
        const double headroom = no_reserve + p.reserve <= opt->device_total_bytes
            ? as_gib(opt->device_total_bytes - no_reserve - p.reserve)
            : -as_gib(total - opt->device_total_bytes);
        printf("| gpu%u | %.2f | %.2f | %.3f | %.2f | %.3f | %.2f | %.3f | %.2f | %.2f | %.2f | %.2f |\n",
               gpu, as_gib(p.weights), as_gib(p.kv), as_gib(p.comp_state),
               as_gib(p.scratch), as_gib(p.relay), as_gib(p.globals),
               as_gib(p.collectives), as_gib(p.mtp), as_gib(p.reserve),
               as_gib(total), headroom);
    }
}

static void print_summary(const options *opt) {
    static const uint64_t contexts[] = { 131072ULL, 262144ULL, 524288ULL, 1048576ULL };

    printf("DS4 V100 TP planner envelope\n");
    printf("assumptions: MXFP4 routed experts, FP8 dense weights, FP16 activations, no persistent dequantized weights\n");
    printf("configured: slots=%u ctx=%" PRIu64 " kv=%s kv_%s reserve=%.2f GiB scratch=%.2f GiB mtp=%s\n",
           opt->slots, opt->ctx, kv_name(opt->kv), opt->kv_sharded ? "sharded" : "replicated",
           opt->reserve_gib, opt->scratch_gib, opt->mtp ? "on" : "off");
    printf("note: TP collective bytes are an execution-risk estimate, not resident memory.\n");

    printf("\nTopology admission summary\n");
    printf("| Topology | TP | PP | Config fits | Worst total | Max slots @128K | Max slots @256K | Max slots @512K | Max slots @1M | TP wire/step @configured |\n");
    printf("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|\n");
    for (topology_id topo = 0; topo < TOPO_COUNT; topo++) {
        if (opt->topology_filter >= 0 && opt->topology_filter != (int)topo) continue;
        uint64_t worst = 0;
        bool fits = false;
        compute_worst(opt, topo, opt->ctx, opt->slots, &worst, NULL, &fits);
        uint32_t admitted[4];
        for (uint32_t i = 0; i < 4; i++) admitted[i] = admitted_slots(opt, topo, contexts[i], NULL);
        printf("| %s | %u | %u | %s | %.2f GiB | %u | %u | %u | %u | %.2f MiB |\n",
               topology_name(topo), topo_tp(topo), topo_pp(topo), fits ? "yes" : "no",
               as_gib(worst), admitted[0], admitted[1], admitted[2], admitted[3],
               as_mib(collective_wire_bytes_per_step(topo, opt->slots)));
    }

    for (topology_id topo = 0; topo < TOPO_COUNT; topo++) {
        if (opt->topology_filter >= 0 && opt->topology_filter != (int)topo) continue;
        print_topology_plan(opt, topo);
    }
}

static int parse_topology(const char *s) {
    for (topology_id topo = 0; topo < TOPO_COUNT; topo++) {
        if (!strcmp(s, topology_name(topo))) return (int)topo;
    }
    if (!strcmp(s, "all")) return -1;
    if (!strcmp(s, "layer") || !strcmp(s, "layer8")) return TOPO_PP8_TP1;
    if (!strcmp(s, "tp2")) return TOPO_PP4_TP2;
    if (!strcmp(s, "tp4")) return TOPO_PP2_TP4;
    if (!strcmp(s, "tp8")) return TOPO_PP1_TP8;
    die("--topology must be all, layer8, tp2, tp4, tp8, or a full topology name");
    return -1;
}

static kv_dtype parse_kv(const char *s) {
    if (!strcmp(s, "f16")) return KV_F16;
    if (!strcmp(s, "f8") || !strcmp(s, "f8_e4m3_b128")) return KV_F8_E4M3_B128;
    if (!strcmp(s, "q8") || !strcmp(s, "q8_0")) return KV_Q8_0;
    die("--kv-dtype must be f16, f8, or q8_0");
    return KV_F8_E4M3_B128;
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-v100-plan-tp [options]\n"
        "\n"
        "Options:\n"
        "  --ctx N                    Context tokens. Default: 262144\n"
        "  --slots N                  Configured slots. Default: 32\n"
        "  --topology NAME            all, layer8, tp2, tp4, tp8. Default: all\n"
        "  --kv-dtype f16|f8|q8_0      KV cache planning dtype. Default: f8\n"
        "  --kv-sharding on|off        Shard KV inside TP groups. Default: on\n"
        "  --mtp on|off                Include MTP block on output stage. Default: off\n"
        "  --reserve-gib F            Reserve per GPU. Default: 4.0\n"
        "  --scratch-gib F            Scratch per GPU. Default: 1.0\n"
        "  --device-total-bytes N      Per-GPU VRAM. Default: 32 GiB\n"
        "\n"
        "Examples:\n"
        "  ds4-v100-plan-tp --slots 32 --ctx 262144 --kv-dtype f8\n"
        "  ds4-v100-plan-tp --topology tp4 --slots 32 --ctx 262144 --kv-dtype f8\n");
}

int main(int argc, char **argv) {
    options opt = {
        .ctx = 262144,
        .slots = 32,
        .gpus = 8,
        .device_total_bytes = 32ULL * GiB,
        .reserve_gib = 4.0,
        .scratch_gib = 1.0,
        .mtp = false,
        .kv_sharded = true,
        .kv = KV_F8_E4M3_B128,
        .topology_filter = -1,
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
        } else if (!strcmp(arg, "--topology") && i + 1 < argc) {
            opt.topology_filter = parse_topology(argv[++i]);
        } else if (!strcmp(arg, "--kv-dtype") && i + 1 < argc) {
            opt.kv = parse_kv(argv[++i]);
        } else if (!strcmp(arg, "--kv-sharding") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "on")) opt.kv_sharded = true;
            else if (!strcmp(v, "off")) opt.kv_sharded = false;
            else die("--kv-sharding must be on or off");
        } else if (!strcmp(arg, "--mtp") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "on")) opt.mtp = true;
            else if (!strcmp(v, "off")) opt.mtp = false;
            else die("--mtp must be on or off");
        } else if (!strcmp(arg, "--reserve-gib") && i + 1 < argc) {
            opt.reserve_gib = parse_double_arg(argv[++i], "--reserve-gib");
        } else if (!strcmp(arg, "--scratch-gib") && i + 1 < argc) {
            opt.scratch_gib = parse_double_arg(argv[++i], "--scratch-gib");
        } else if (!strcmp(arg, "--device-total-bytes") && i + 1 < argc) {
            opt.device_total_bytes = parse_u64_arg(argv[++i], "--device-total-bytes");
        } else {
            usage(stderr);
            return 2;
        }
    }

    if (opt.gpus != 8) die("this TP planner variant currently assumes exactly 8 GPUs");
    if (opt.slots == 0) die("--slots must be positive");
    if (opt.reserve_gib < 0.0 || opt.scratch_gib < 0.0) die("reserve/scratch must be non-negative");
    print_summary(&opt);
    return 0;
}
