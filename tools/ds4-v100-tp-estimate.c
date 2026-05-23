#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KiB (1024.0)
#define MiB (1024.0 * KiB)
#define GiB (1024.0 * MiB)

enum {
    DS4_LAYERS = 43,
    DS4_GPUS = 8,
    DS4_HIDDEN = 4096,
    DS4_HC = 4,
    DS4_ROUTES = 6,
};

typedef struct {
    uint32_t slots;
    uint32_t active_microbatch;
    uint64_t ctx;
    double nvlink_gbps;
    uint32_t collectives_per_layer;
    bool json;
} options;

typedef struct {
    const char *name;
    const char *description;
    uint32_t tp;
    uint32_t pp;
    double interstage_bytes;
    double routed_overlay_bytes;
    double tp_allreduce_bytes;
    double ep_dispatch_bytes;
    double total_wire_bytes;
    double min_transfer_ms;
    const char *decision;
    const char *next_step;
} estimate;

static void die(const char *msg) {
    fprintf(stderr, "ds4-v100-tp-estimate: %s\n", msg);
    exit(1);
}

static uint64_t parse_u64(const char *s, const char *name) {
    if (!s || !*s) die("missing numeric argument");
    errno = 0;
    char *end = NULL;
    const unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "ds4-v100-tp-estimate: invalid %s: %s\n", name, s);
        exit(1);
    }
    return (uint64_t)v;
}

static double parse_f64(const char *s, const char *name) {
    if (!s || !*s) die("missing numeric argument");
    errno = 0;
    char *end = NULL;
    const double v = strtod(s, &end);
    if (errno || !end || *end || !(v > 0.0)) {
        fprintf(stderr, "ds4-v100-tp-estimate: invalid %s: %s\n", name, s);
        exit(1);
    }
    return v;
}

static void options_init(options *opt) {
    memset(opt, 0, sizeof(*opt));
    opt->slots = 16;
    opt->active_microbatch = 16;
    opt->ctx = 262144;
    opt->nvlink_gbps = 150.0;
    opt->collectives_per_layer = 4;
}

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-v100-tp-estimate [options]\n"
            "\n"
            "Options:\n"
            "  --slots N                    configured slots, default 16\n"
            "  --active-microbatch N        active decode slots per step, default 16\n"
            "  --ctx N                      context tokens, default 262144\n"
            "  --nvlink-gbps F              effective one-way GB/s budget, default 150\n"
            "  --collectives-per-layer N    full-TP hidden collectives per layer, default 4\n"
            "  --json                       print machine-readable JSON\n"
            "  --help                       show this help\n");
}

static void parse_options(options *opt, int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--slots")) {
            if (++i >= argc) die("--slots requires a value");
            opt->slots = (uint32_t)parse_u64(argv[i], "--slots");
        } else if (!strcmp(arg, "--active-microbatch")) {
            if (++i >= argc) die("--active-microbatch requires a value");
            opt->active_microbatch = (uint32_t)parse_u64(argv[i], "--active-microbatch");
        } else if (!strcmp(arg, "--ctx")) {
            if (++i >= argc) die("--ctx requires a value");
            opt->ctx = parse_u64(argv[i], "--ctx");
        } else if (!strcmp(arg, "--nvlink-gbps")) {
            if (++i >= argc) die("--nvlink-gbps requires a value");
            opt->nvlink_gbps = parse_f64(argv[i], "--nvlink-gbps");
        } else if (!strcmp(arg, "--collectives-per-layer")) {
            if (++i >= argc) die("--collectives-per-layer requires a value");
            opt->collectives_per_layer =
                (uint32_t)parse_u64(argv[i], "--collectives-per-layer");
        } else if (!strcmp(arg, "--json")) {
            opt->json = true;
        } else if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            usage(stdout);
            exit(0);
        } else {
            fprintf(stderr, "ds4-v100-tp-estimate: unknown option: %s\n", arg);
            usage(stderr);
            exit(1);
        }
    }
    if (opt->slots == 0 || opt->active_microbatch == 0) {
        die("--slots and --active-microbatch must be positive");
    }
    if (opt->active_microbatch > opt->slots) {
        die("--active-microbatch must be <= --slots");
    }
    if (opt->collectives_per_layer == 0) {
        die("--collectives-per-layer must be positive");
    }
}

static double ring_factor(uint32_t participants) {
    if (participants <= 1) return 0.0;
    return 2.0 * (double)(participants - 1u) / (double)participants;
}

static double min_ms(double bytes, double gbps) {
    return bytes / (gbps * 1000000000.0) * 1000.0;
}

static double kv_estimate_bytes(uint64_t ctx, uint32_t slots) {
    const double single_slot_at_1m = 8.5 * GiB;
    return single_slot_at_1m * ((double)ctx / 1048576.0) * (double)slots;
}

static estimate make_layer_split(const options *opt) {
    const double hc_bytes = (double)DS4_HC * DS4_HIDDEN * sizeof(float);
    const double interstage = (double)(DS4_GPUS - 1u) * hc_bytes *
                              (double)opt->active_microbatch;
    estimate e = {
        .name = "layer8",
        .description = "current contiguous 8-stage layer split",
        .tp = 1,
        .pp = 8,
        .interstage_bytes = interstage,
        .decision = "baseline",
        .next_step = "keep as fit-first baseline while TP/EP is proven",
    };
    e.total_wire_bytes = e.interstage_bytes;
    e.min_transfer_ms = min_ms(e.total_wire_bytes, opt->nvlink_gbps);
    return e;
}

static estimate make_routed_tp2_overlay(const options *opt) {
    const double hidden_bytes = (double)DS4_HIDDEN * sizeof(float);
    const double route_bytes = (double)DS4_ROUTES * (sizeof(int32_t) + sizeof(float));
    const double per_layer = (2.0 * hidden_bytes + route_bytes) *
                             (double)opt->active_microbatch;
    estimate e = {
        .name = "routed_tp2_overlay",
        .description = "existing routed-only TP2 copy-in/copy-out overlay",
        .tp = 2,
        .pp = 8,
        .routed_overlay_bytes = per_layer * DS4_LAYERS,
        .decision = "rejected",
        .next_step = "do not expand; it preserves the wrong F32 per-layer boundary",
    };
    e.total_wire_bytes = e.routed_overlay_bytes;
    e.min_transfer_ms = min_ms(e.total_wire_bytes, opt->nvlink_gbps);
    return e;
}

static estimate make_full_tp(const options *opt, uint32_t tp, uint32_t pp, const char *name) {
    const double hidden_bytes = (double)DS4_HIDDEN * sizeof(float);
    const double active = (double)opt->active_microbatch;
    const double layers_per_token = (double)DS4_LAYERS;
    const double allreduce = layers_per_token *
        (double)opt->collectives_per_layer *
        ring_factor(tp) *
        hidden_bytes *
        active;
    const double ep_dispatch = layers_per_token *
        2.0 *
        ((double)(tp - 1u) / (double)tp) *
        (double)DS4_ROUTES *
        (double)DS4_HIDDEN *
        sizeof(uint16_t) *
        active;
    const double hc_bytes = (double)DS4_HC * DS4_HIDDEN * sizeof(float);
    const double interstage = pp > 1u
        ? (double)(pp - 1u) * hc_bytes * active
        : 0.0;

    estimate e = {
        .name = name,
        .description = "candidate full-layer TP/EP ownership envelope",
        .tp = tp,
        .pp = pp,
        .interstage_bytes = interstage,
        .tp_allreduce_bytes = allreduce,
        .ep_dispatch_bytes = ep_dispatch,
    };
    e.total_wire_bytes = e.interstage_bytes + e.tp_allreduce_bytes + e.ep_dispatch_bytes;
    e.min_transfer_ms = min_ms(e.total_wire_bytes, opt->nvlink_gbps);

    if (tp == 2u) {
        e.decision = "possible_probe";
        e.next_step = "prototype only if attention/shared dense are TP-owned too";
    } else if (tp == 4u && pp == 2u) {
        e.decision = "fallback_candidate";
        e.next_step = "use only if TP8 memory/collective pressure fails";
    } else if (tp == 4u) {
        e.decision = "strong_candidate";
        e.next_step = "best next bounded prototype: TP4 dense plus EP4/8 experts";
    } else {
        e.decision = "high_risk_candidate";
        e.next_step = "TP8 needs real collective implementation before runtime work";
    }
    return e;
}

static void print_bytes(double bytes) {
    if (bytes >= GiB) {
        printf("%.3f GiB", bytes / GiB);
    } else if (bytes >= MiB) {
        printf("%.3f MiB", bytes / MiB);
    } else {
        printf("%.3f KiB", bytes / KiB);
    }
}

static void print_text(const options *opt, const estimate *est, size_t n) {
    const double hidden_bytes = (double)DS4_HIDDEN * sizeof(float);
    const double hc_bytes = (double)DS4_HC * DS4_HIDDEN * sizeof(float);
    printf("DS4 V100 TP/EP topology estimate\n");
    printf("ctx=%" PRIu64 " slots=%u active_microbatch=%u nvlink_gbps=%.1f collectives_per_layer=%u\n",
           opt->ctx,
           opt->slots,
           opt->active_microbatch,
           opt->nvlink_gbps,
           opt->collectives_per_layer);
    printf("hidden_payload=%.1f KiB hc_payload=%.1f KiB routed_f16_dispatch_per_slot=%.1f KiB\n",
           hidden_bytes / KiB,
           hc_bytes / KiB,
           (double)DS4_ROUTES * DS4_HIDDEN * sizeof(uint16_t) / KiB);
    printf("kv_envelope_f16_estimate=");
    print_bytes(kv_estimate_bytes(opt->ctx, opt->slots));
    printf(" aggregate\n\n");

    printf("| Topology | TP | PP | Interstage | Routed overlay | TP allreduce | EP dispatch | Total wire/token | Min xfer ms | Decision |\n");
    printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for (size_t i = 0; i < n; i++) {
        printf("| %s | %u | %u | ",
               est[i].name, est[i].tp, est[i].pp);
        print_bytes(est[i].interstage_bytes);
        printf(" | ");
        print_bytes(est[i].routed_overlay_bytes);
        printf(" | ");
        print_bytes(est[i].tp_allreduce_bytes);
        printf(" | ");
        print_bytes(est[i].ep_dispatch_bytes);
        printf(" | ");
        print_bytes(est[i].total_wire_bytes);
        printf(" | %.3f | %s |\n", est[i].min_transfer_ms, est[i].decision);
    }

    printf("\nImplementation guidance\n");
    for (size_t i = 0; i < n; i++) {
        printf("- %s: %s\n", est[i].name, est[i].next_step);
    }
}

static void print_json_string(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\') {
            putchar('\\');
            putchar(*p);
        } else if (*p >= 0x20 && *p < 0x7f) {
            putchar(*p);
        } else {
            printf("\\u%04x", (unsigned)*p);
        }
    }
    putchar('"');
}

static void print_json(const options *opt, const estimate *est, size_t n) {
    printf("{\"schema\":\"ds4_v100_tp_ep_estimate.v1\",");
    printf("\"ctx\":%" PRIu64 ",\"slots\":%u,\"active_microbatch\":%u,"
           "\"nvlink_gbps\":%.3f,\"collectives_per_layer\":%u,",
           opt->ctx,
           opt->slots,
           opt->active_microbatch,
           opt->nvlink_gbps,
           opt->collectives_per_layer);
    printf("\"hidden_payload_bytes\":%u,\"hc_payload_bytes\":%u,"
           "\"kv_envelope_f16_bytes\":%.0f,",
           DS4_HIDDEN * (unsigned)sizeof(float),
           DS4_HC * DS4_HIDDEN * (unsigned)sizeof(float),
           kv_estimate_bytes(opt->ctx, opt->slots));
    printf("\"topologies\":[");
    for (size_t i = 0; i < n; i++) {
        if (i) putchar(',');
        printf("{\"name\":");
        print_json_string(est[i].name);
        printf(",\"description\":");
        print_json_string(est[i].description);
        printf(",\"tp\":%u,\"pp\":%u,"
               "\"interstage_bytes\":%.0f,"
               "\"routed_overlay_bytes\":%.0f,"
               "\"tp_allreduce_bytes\":%.0f,"
               "\"ep_dispatch_bytes\":%.0f,"
               "\"total_wire_bytes\":%.0f,"
               "\"min_transfer_ms\":%.6f,"
               "\"decision\":",
               est[i].tp,
               est[i].pp,
               est[i].interstage_bytes,
               est[i].routed_overlay_bytes,
               est[i].tp_allreduce_bytes,
               est[i].ep_dispatch_bytes,
               est[i].total_wire_bytes,
               est[i].min_transfer_ms);
        print_json_string(est[i].decision);
        printf(",\"next_step\":");
        print_json_string(est[i].next_step);
        putchar('}');
    }
    printf("]}\n");
}

int main(int argc, char **argv) {
    options opt;
    options_init(&opt);
    parse_options(&opt, argc, argv);

    estimate est[6];
    size_t n = 0;
    est[n++] = make_layer_split(&opt);
    est[n++] = make_routed_tp2_overlay(&opt);
    est[n++] = make_full_tp(&opt, 2, 1, "tp2_pp1_full");
    est[n++] = make_full_tp(&opt, 4, 1, "tp4_pp1_full");
    est[n++] = make_full_tp(&opt, 8, 1, "tp8_pp1_full");
    est[n++] = make_full_tp(&opt, 4, 2, "tp4_pp2_hybrid");

    if (opt.json) {
        print_json(&opt, est, n);
    } else {
        print_text(&opt, est, n);
    }
    return 0;
}

