#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define KiB (1024ULL)
#define MiB (1024ULL * KiB)
#define GiB (1024ULL * MiB)

enum {
    DS4_N_LAYER            = 43,
    DS4_N_EMBD             = 4096,
    DS4_N_VOCAB            = 129280,
    DS4_N_HEAD_DIM         = 512,
    DS4_N_INDEXER_HEAD_DIM = 128,
    DS4_N_EXPERT           = 256,
    DS4_N_EXPERT_USED      = 6,
    DS4_N_SWA              = 128,
    DS4_N_GPU              = 8,
    DS4_N_TP               = 8,
    DS4_N_EP               = 8,
    DS4_MAX_SLOTS          = 256,
};

typedef enum {
    KV_F16,
    KV_F8_E4M3_B128,
    KV_Q8_0,
} kv_dtype;

typedef struct {
    uint64_t ctx;
    uint32_t slots;
    uint64_t device_total_bytes;
    uint64_t weight_total_bytes;
    double reserve_gib;
    double scratch_gib;
    kv_dtype kv;
    bool json;
    const char *pack_dir;
} options;

typedef struct {
    uint64_t weights;
    uint64_t kv;
    uint64_t comp_state;
    uint64_t scratch;
    uint64_t collectives;
    uint64_t globals;
    uint64_t reserve;
} gpu_plan;

typedef struct {
    uint64_t attn_kv;
    uint64_t indexer_kv;
    uint64_t comp_state;
} kv_plan;

static void die(const char *msg) {
    fprintf(stderr, "ds4-v100-plan-tp: %s\n", msg);
    exit(1);
}

static uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a) die("integer overflow in planner");
    return a * b;
}

static uint64_t ceil_div_u64(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

static uint64_t bytes_blocks(uint64_t elems, uint64_t block_elems, uint64_t block_bytes) {
    return checked_mul((elems + block_elems - 1) / block_elems, block_bytes);
}

static uint64_t bytes_f16(uint64_t elems) { return checked_mul(elems, 2); }
static uint64_t bytes_f32(uint64_t elems) { return checked_mul(elems, 4); }
static uint64_t bytes_f8_e4m3_b128(uint64_t elems) { return bytes_blocks(elems, 128, 129); }
static uint64_t bytes_q8_0(uint64_t elems) { return bytes_blocks(elems, 32, 34); }

static double as_gib(uint64_t bytes) {
    return (double)bytes / (double)GiB;
}

static double as_mib(uint64_t bytes) {
    return (double)bytes / (double)MiB;
}

static uint64_t from_gib(double gib) {
    return (uint64_t)(gib * (double)GiB);
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

static const char *kv_name(kv_dtype kv) {
    switch (kv) {
    case KV_F16: return "f16";
    case KV_F8_E4M3_B128: return "f8_e4m3_b128";
    case KV_Q8_0: return "q8_0";
    default: return "unknown";
    }
}

static kv_dtype parse_kv(const char *s) {
    if (!strcmp(s, "f16")) return KV_F16;
    if (!strcmp(s, "f8") || !strcmp(s, "f8_e4m3_b128")) return KV_F8_E4M3_B128;
    if (!strcmp(s, "q8") || !strcmp(s, "q8_0")) return KV_Q8_0;
    die("--kv-dtype must be f16, f8, or q8_0");
    return KV_F8_E4M3_B128;
}

static int layer_ratio(uint32_t il) {
    if (il < 2) return 0;
    return (il % 2) == 0 ? 4 : 128;
}

static uint64_t kv_values_bytes(uint64_t values, kv_dtype kv) {
    switch (kv) {
    case KV_F16: return bytes_f16(values);
    case KV_F8_E4M3_B128: return bytes_f8_e4m3_b128(values);
    case KV_Q8_0: return bytes_q8_0(values);
    default: return bytes_f16(values);
    }
}

static uint64_t layer_attn_kv_bytes(uint32_t il, uint64_t ctx, kv_dtype kv) {
    const int ratio = layer_ratio(il);
    const uint64_t rows = (uint64_t)DS4_N_SWA + (ratio ? ctx / (uint64_t)ratio : 0);
    return kv_values_bytes(checked_mul(rows, DS4_N_HEAD_DIM), kv);
}

static uint64_t layer_indexer_kv_bytes(uint32_t il, uint64_t ctx, kv_dtype kv) {
    if (layer_ratio(il) != 4) return 0;
    return kv_values_bytes(checked_mul(ctx / 4u, DS4_N_INDEXER_HEAD_DIM), kv);
}

static uint64_t layer_comp_state_bytes(uint32_t il, uint64_t ctx) {
    const int ratio = layer_ratio(il);
    if (!ratio) return 0;

    /*
     * DS4 compression state is implementation-owned metadata rather than
     * persistent model weight. The current runtime scales with compressed rows
     * and ratio. Use an intentionally visible envelope so admission decisions
     * do not silently ignore it.
     */
    const uint64_t comp_rows = ctx / (uint64_t)ratio;
    const uint64_t attn = checked_mul(comp_rows, DS4_N_HEAD_DIM);
    if (ratio == 4) {
        const uint64_t indexer = checked_mul(ctx / 4u, DS4_N_INDEXER_HEAD_DIM);
        return bytes_f32((attn + indexer) / 8u);
    }
    return bytes_f32(attn / 8u);
}

static kv_plan aggregate_kv(uint64_t ctx, uint32_t slots, kv_dtype kv) {
    kv_plan p = {0, 0, 0};
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        p.attn_kv += layer_attn_kv_bytes(il, ctx, kv);
        p.indexer_kv += layer_indexer_kv_bytes(il, ctx, kv);
        p.comp_state += layer_comp_state_bytes(il, ctx);
    }
    p.attn_kv = checked_mul(p.attn_kv, slots);
    p.indexer_kv = checked_mul(p.indexer_kv, slots);
    p.comp_state = checked_mul(p.comp_state, slots);
    return p;
}

static bool is_gpu_weight_file(const char *name) {
    return !strncmp(name, "gpu", 3) && strstr(name, ".weights") != NULL;
}

static uint64_t weight_total_from_pack_dir(const char *path) {
    DIR *dir = opendir(path);
    if (!dir) {
        fprintf(stderr, "failed to open --pack-dir %s: %s\n", path, strerror(errno));
        exit(2);
    }

    uint64_t total = 0;
    uint32_t matched = 0;
    struct dirent *ent = NULL;
    while ((ent = readdir(dir)) != NULL) {
        if (!is_gpu_weight_file(ent->d_name)) continue;
        char full[4096];
        const int n = snprintf(full, sizeof(full), "%s/%s", path, ent->d_name);
        if (n < 0 || (size_t)n >= sizeof(full)) die("--pack-dir path too long");

        struct stat st;
        if (stat(full, &st) != 0) {
            fprintf(stderr, "failed to stat %s: %s\n", full, strerror(errno));
            closedir(dir);
            exit(2);
        }
        if (st.st_size <= 0) continue;
        total += (uint64_t)st.st_size;
        matched++;
    }
    closedir(dir);

    if (matched == 0) die("--pack-dir did not contain gpu*.weights files");
    return total;
}

static uint64_t shard_bytes(uint64_t bytes) {
    return ceil_div_u64(bytes, DS4_N_TP);
}

static uint64_t global_bytes_per_gpu(void) {
    const uint64_t embed = bytes_f16(checked_mul(DS4_N_EMBD, DS4_N_VOCAB));
    const uint64_t output = bytes_f16(checked_mul(DS4_N_EMBD, DS4_N_VOCAB));
    const uint64_t output_control =
        bytes_f32(checked_mul((uint64_t)DS4_N_EMBD * 4u, 4u)) +
        bytes_f32(4u) +
        bytes_f32(1u) +
        bytes_f32(DS4_N_EMBD);
    return shard_bytes(embed) + shard_bytes(output) + output_control;
}

static uint64_t collective_workspace_bytes(uint32_t slots) {
    return checked_mul(2u * (uint64_t)slots, DS4_N_EMBD * 2ull);
}

static uint64_t hidden_payload_bytes(uint32_t slots) {
    return checked_mul(slots, DS4_N_EMBD * 2ull);
}

static uint64_t ring_allreduce_wire_per_collective(uint32_t slots) {
    const uint64_t payload = hidden_payload_bytes(slots);
    return payload * (2ull * (DS4_N_TP - 1ull)) / DS4_N_TP;
}

static uint64_t hidden_collective_wire_per_step(uint32_t slots) {
    const uint64_t reductions_per_layer = 2;
    return checked_mul(checked_mul(ring_allreduce_wire_per_collective(slots),
                                   reductions_per_layer),
                       DS4_N_LAYER);
}

static uint64_t ep_dispatch_payload_per_step(uint32_t slots) {
    const uint64_t routes = checked_mul(slots, DS4_N_EXPERT_USED);
    return checked_mul(routes, DS4_N_EMBD * 2ull);
}

static uint64_t gpu_total(const gpu_plan *p) {
    return p->weights + p->kv + p->comp_state + p->scratch +
           p->collectives + p->globals + p->reserve;
}

static uint64_t gpu_no_reserve(const gpu_plan *p) {
    return p->weights + p->kv + p->comp_state + p->scratch +
           p->collectives + p->globals;
}

static gpu_plan plan_gpu(const options *opt) {
    const kv_plan kv = aggregate_kv(opt->ctx, opt->slots, opt->kv);
    gpu_plan p = {0};
    p.weights = shard_bytes(opt->weight_total_bytes);
    p.kv = shard_bytes(kv.attn_kv + kv.indexer_kv);
    p.comp_state = shard_bytes(kv.comp_state);
    p.scratch = from_gib(opt->scratch_gib);
    p.collectives = collective_workspace_bytes(opt->slots);
    p.globals = global_bytes_per_gpu();
    p.reserve = from_gib(opt->reserve_gib);
    return p;
}

static bool config_fits(const options *opt, uint64_t *total_out) {
    const gpu_plan p = plan_gpu(opt);
    const uint64_t total = gpu_total(&p);
    if (total_out) *total_out = total;
    return total <= opt->device_total_bytes;
}

static uint32_t admitted_slots_at_ctx(const options *base, uint64_t ctx, uint64_t *total_at_max) {
    options opt = *base;
    opt.ctx = ctx;
    uint32_t admitted = 0;
    uint64_t last_total = 0;
    for (uint32_t slots = 1; slots <= DS4_MAX_SLOTS; slots++) {
        opt.slots = slots;
        uint64_t total = 0;
        if (!config_fits(&opt, &total)) break;
        admitted = slots;
        last_total = total;
    }
    if (total_at_max) *total_at_max = last_total;
    return admitted;
}

static double routes_per_gpu(uint32_t slots) {
    return ((double)slots * (double)DS4_N_EXPERT_USED) / (double)DS4_N_EP;
}

static double routes_per_expert(uint32_t slots) {
    return ((double)slots * (double)DS4_N_EXPERT_USED) / (double)DS4_N_EXPERT;
}

static void print_json(const options *opt) {
    static const uint64_t contexts[] = {131072ULL, 262144ULL, 524288ULL, 1048576ULL};
    const kv_plan kv = aggregate_kv(opt->ctx, opt->slots, opt->kv);
    const gpu_plan gpu = plan_gpu(opt);
    const uint64_t total = gpu_total(&gpu);
    const uint64_t no_reserve = gpu_no_reserve(&gpu);

    printf("{\n");
    printf("  \"topology\":\"tp8_ep8\",\n");
    printf("  \"pp\":1,\n");
    printf("  \"tp\":8,\n");
    printf("  \"ep\":8,\n");
    printf("  \"slots\":%u,\n", opt->slots);
    printf("  \"ctx\":%" PRIu64 ",\n", opt->ctx);
    printf("  \"kv_dtype\":\"%s\",\n", kv_name(opt->kv));
    printf("  \"weight_total_bytes\":%" PRIu64 ",\n", opt->weight_total_bytes);
    printf("  \"kv_aggregate_bytes\":%" PRIu64 ",\n", kv.attn_kv + kv.indexer_kv);
    printf("  \"comp_state_aggregate_bytes\":%" PRIu64 ",\n", kv.comp_state);
    printf("  \"hidden_collective_wire_per_step_bytes\":%" PRIu64 ",\n",
           hidden_collective_wire_per_step(opt->slots));
    printf("  \"ep_dispatch_return_wire_per_step_bytes\":%" PRIu64 ",\n",
           (uint64_t)(2ull * ep_dispatch_payload_per_step(opt->slots)));
    printf("  \"routes_per_gpu_avg\":%.3f,\n", routes_per_gpu(opt->slots));
    printf("  \"routes_per_expert_avg\":%.3f,\n", routes_per_expert(opt->slots));
    printf("  \"gpu\":{\n");
    printf("    \"weights\":%" PRIu64 ",\n", gpu.weights);
    printf("    \"kv\":%" PRIu64 ",\n", gpu.kv);
    printf("    \"comp_state\":%" PRIu64 ",\n", gpu.comp_state);
    printf("    \"scratch\":%" PRIu64 ",\n", gpu.scratch);
    printf("    \"collectives\":%" PRIu64 ",\n", gpu.collectives);
    printf("    \"globals\":%" PRIu64 ",\n", gpu.globals);
    printf("    \"reserve\":%" PRIu64 ",\n", gpu.reserve);
    printf("    \"total\":%" PRIu64 ",\n", total);
    printf("    \"headroom_after_reserve\":%" PRIu64 "\n",
           total <= opt->device_total_bytes ? opt->device_total_bytes - total : 0);
    printf("  },\n");
    printf("  \"admission\":[\n");
    for (uint32_t i = 0; i < 4; i++) {
        uint64_t tier_total = 0;
        const uint32_t admitted = admitted_slots_at_ctx(opt, contexts[i], &tier_total);
        printf("    {\"ctx\":%" PRIu64 ",\"max_slots\":%u,\"gpu_total_at_max_bytes\":%" PRIu64 "}%s\n",
               contexts[i], admitted, tier_total, i == 3 ? "" : ",");
    }
    printf("  ],\n");
    printf("  \"fits\":%s,\n", total <= opt->device_total_bytes ? "true" : "false");
    printf("  \"no_reserve_bytes\":%" PRIu64 "\n", no_reserve);
    printf("}\n");
}

static void print_human(const options *opt) {
    static const uint64_t contexts[] = {131072ULL, 262144ULL, 524288ULL, 1048576ULL};
    const kv_plan kv = aggregate_kv(opt->ctx, opt->slots, opt->kv);
    const gpu_plan p = plan_gpu(opt);
    const uint64_t total = gpu_total(&p);
    const uint64_t no_reserve = gpu_no_reserve(&p);
    const bool fits = total <= opt->device_total_bytes;

    printf("DS4 V100 TP/EP planner contract\n");
    printf("topology: PP=1(no pipeline) TP=8 EP=8 KV=sharded\n");
    printf("configured: slots=%u ctx=%" PRIu64 " kv=%s reserve=%.2f GiB scratch=%.2f GiB\n",
           opt->slots, opt->ctx, kv_name(opt->kv), opt->reserve_gib, opt->scratch_gib);
    printf("weights: total %.2f GiB, per TP rank %.2f GiB%s\n",
           as_gib(opt->weight_total_bytes), as_gib(p.weights),
           opt->pack_dir ? " (from --pack-dir)" : " (static estimate)");
    printf("verdict: %s; per-GPU total %.2f / %.2f GiB; headroom after reserve %.2f GiB\n",
           fits ? "fits" : "over budget", as_gib(total), as_gib(opt->device_total_bytes),
           fits ? as_gib(opt->device_total_bytes - total) : -as_gib(total - opt->device_total_bytes));

    printf("\n## KV aggregate before TP sharding\n");
    printf("| Component | Aggregate | Per GPU |\n");
    printf("|---|---:|---:|\n");
    printf("| attn_kv | %.2f GiB | %.2f GiB |\n", as_gib(kv.attn_kv), as_gib(shard_bytes(kv.attn_kv)));
    printf("| indexer_kv | %.2f GiB | %.2f GiB |\n", as_gib(kv.indexer_kv), as_gib(shard_bytes(kv.indexer_kv)));
    printf("| comp_state envelope | %.2f GiB | %.2f GiB |\n", as_gib(kv.comp_state), as_gib(shard_bytes(kv.comp_state)));

    printf("\n## Per-GPU resident budget\n");
    printf("| GPU | Weights | KV | Comp | Scratch | Collectives | Globals | Reserve | Total | Headroom |\n");
    printf("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (uint32_t gpu = 0; gpu < DS4_N_GPU; gpu++) {
        (void)gpu;
        const double headroom = fits
            ? as_gib(opt->device_total_bytes - total)
            : -as_gib(total - opt->device_total_bytes);
        printf("| gpu%u | %.2f | %.2f | %.2f | %.2f | %.3f | %.2f | %.2f | %.2f | %.2f |\n",
               gpu, as_gib(p.weights), as_gib(p.kv), as_gib(p.comp_state),
               as_gib(p.scratch), as_gib(p.collectives), as_gib(p.globals),
               as_gib(p.reserve), as_gib(total), headroom);
    }

    printf("\n## Admission tiers\n");
    printf("| Context | Max slots | Per-GPU total at max |\n");
    printf("|---:|---:|---:|\n");
    for (uint32_t i = 0; i < 4; i++) {
        uint64_t tier_total = 0;
        const uint32_t admitted = admitted_slots_at_ctx(opt, contexts[i], &tier_total);
        printf("| %" PRIu64 " | %u | %.2f GiB |\n", contexts[i], admitted, as_gib(tier_total));
    }

    printf("\n## Decode-shape traffic estimates\n");
    printf("| Path | Per decode step |\n");
    printf("|---|---:|\n");
    printf("| hidden payload per rank | %.3f MiB |\n", as_mib(hidden_payload_bytes(opt->slots)));
    printf("| one ring all-reduce per rank | %.3f MiB |\n", as_mib(ring_allreduce_wire_per_collective(opt->slots)));
    printf("| hidden collectives, 2/layer x 43 | %.3f MiB |\n",
           as_mib(hidden_collective_wire_per_step(opt->slots)));
    printf("| EP dispatch + return aggregate | %.3f MiB |\n",
           as_mib(2ull * ep_dispatch_payload_per_step(opt->slots)));

    printf("\n## Expert ownership and density\n");
    printf("| Metric | Value |\n");
    printf("|---|---:|\n");
    printf("| experts per GPU | %u |\n", DS4_N_EXPERT / DS4_N_EP);
    printf("| active routes per decode step | %u |\n", opt->slots * DS4_N_EXPERT_USED);
    printf("| average routes per GPU | %.2f |\n", routes_per_gpu(opt->slots));
    printf("| average routes per expert | %.3f |\n", routes_per_expert(opt->slots));

    printf("\nnotes:\n");
    printf("- Planner intentionally exposes no PP/layer-split topology modes.\n");
    printf("- KV is always TP-sharded here; replicated KV is not a production target for 32-slot/256K.\n");
    printf("- Weight bytes are a residency estimate until the TP/EP pack contract lands.\n");
    printf("- No-reserve per-GPU total is %.2f GiB.\n", as_gib(no_reserve));
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-v100-plan-tp [options]\n"
        "\n"
        "TP8/EP8-only planner for the DS4 V100 appliance. This tool does not\n"
        "expose PP/layer-split topology variants.\n"
        "\n"
        "Options:\n"
        "  --ctx N                    Context tokens. Default: 262144\n"
        "  --slots N                  Configured slots. Default: 32\n"
        "  --kv-dtype f16|f8|q8_0      KV cache planning dtype. Default: f8\n"
        "  --weight-total-gib F        Total resident weight estimate. Default: 145.45\n"
        "  --pack-dir PATH             Sum gpu*.weights files and use as weight total\n"
        "  --reserve-gib F             Reserve per GPU. Default: 2.0\n"
        "  --scratch-gib F             Scratch per GPU. Default: 1.5\n"
        "  --device-total-bytes N      Per-GPU VRAM. Default: 32 GiB\n"
        "  --json                      Emit machine-readable JSON\n"
        "\n"
        "Examples:\n"
        "  ds4-v100-plan-tp --slots 32 --ctx 262144 --kv-dtype f8\n"
        "  ds4-v100-plan-tp --pack-dir /workspace/packs/ds4-appliance-full-tm-gated-s181\n");
}

int main(int argc, char **argv) {
    options opt = {
        .ctx = 262144,
        .slots = 32,
        .device_total_bytes = 32ULL * GiB,
        .weight_total_bytes = (uint64_t)(145.45 * (double)GiB),
        .reserve_gib = 2.0,
        .scratch_gib = 1.5,
        .kv = KV_F8_E4M3_B128,
        .json = false,
        .pack_dir = NULL,
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
        } else if (!strcmp(arg, "--kv-dtype") && i + 1 < argc) {
            opt.kv = parse_kv(argv[++i]);
        } else if (!strcmp(arg, "--weight-total-gib") && i + 1 < argc) {
            opt.weight_total_bytes = from_gib(parse_double_arg(argv[++i], "--weight-total-gib"));
        } else if (!strcmp(arg, "--pack-dir") && i + 1 < argc) {
            opt.pack_dir = argv[++i];
        } else if (!strcmp(arg, "--reserve-gib") && i + 1 < argc) {
            opt.reserve_gib = parse_double_arg(argv[++i], "--reserve-gib");
        } else if (!strcmp(arg, "--scratch-gib") && i + 1 < argc) {
            opt.scratch_gib = parse_double_arg(argv[++i], "--scratch-gib");
        } else if (!strcmp(arg, "--device-total-bytes") && i + 1 < argc) {
            opt.device_total_bytes = parse_u64_arg(argv[++i], "--device-total-bytes");
        } else if (!strcmp(arg, "--json")) {
            opt.json = true;
        } else if (!strcmp(arg, "--topology")) {
            die("--topology was removed; this planner is TP8/EP8-only");
        } else if (!strcmp(arg, "--kv-sharding")) {
            die("--kv-sharding was removed; TP/EP planner always uses sharded KV");
        } else if (!strcmp(arg, "--mtp")) {
            die("--mtp is not part of the Sprint 226 TP/EP planner contract");
        } else {
            usage(stderr);
            return 2;
        }
    }

    if (opt.slots == 0) die("--slots must be positive");
    if (opt.slots > DS4_MAX_SLOTS) die("--slots exceeds planner maximum");
    if (opt.reserve_gib < 0.0 || opt.scratch_gib < 0.0) die("reserve/scratch must be non-negative");
    if (opt.pack_dir) opt.weight_total_bytes = weight_total_from_pack_dir(opt.pack_dir);

    if (opt.json) print_json(&opt);
    else print_human(&opt);
    return 0;
}
