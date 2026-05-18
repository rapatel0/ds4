#include "ds4_v100_layer_state.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "v100_layer_state_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp, "usage: tests/v100_layer_state_smoke --index FILE [--layer N]\n");
}

static int parse_int_arg(const char *s, const char *name, int max_v) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || !*s || !end || *end != '\0' || v < 0 || v > max_v) {
        fprintf(stderr, "v100_layer_state_smoke: invalid %s: %s\n", name, s ? s : "(null)");
        exit(2);
    }
    return (int)v;
}

int main(int argc, char **argv) {
    const char *index = NULL;
    int layer = 2;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(argv[i], "--index") && i + 1 < argc) {
            index = argv[++i];
        } else if (!strcmp(argv[i], "--layer") && i + 1 < argc) {
            layer = parse_int_arg(argv[++i], "--layer", 42);
        } else {
            usage(stderr);
            return 2;
        }
    }
    if (!index) {
        usage(stderr);
        return 2;
    }

    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    opts.pack_index_path = index;
    opts.kv_ctx_tokens = 1048576;
    opts.kv_active_slots = 1;

    char err[512] = {0};
    ds4_v100_context *ctx = NULL;
    if (ds4_v100_context_open(&ctx, &opts, err, sizeof(err))) {
        fprintf(stderr, "v100_layer_state_smoke: %s\n", err);
        return 1;
    }

    ds4_v100_layer_state state;
    if (ds4_v100_layer_state_init(&state, ctx, layer, err, sizeof(err))) {
        fprintf(stderr, "v100_layer_state_smoke: %s\n", err);
        ds4_v100_context_close(ctx);
        return 1;
    }

    const ds4_v100_layer_info *li = ds4_v100_context_layer(ctx, layer);
    const ds4_v100_stage_info *stage = ds4_v100_context_stage(ctx, state.stage_id);
    check(li != NULL, "layer info exists");
    check(stage != NULL, "stage info exists");
    check(state.stage_id == ds4_v100_stage_for_layer(layer), "stage matches layer map");
    check(state.owning_gpu == state.stage_id, "owning GPU matches stage");
    check(state.layer_class == ds4_v100_layer_class_for_layer(layer), "layer class matches schedule");
    check(state.hidden_size == 4096, "hidden size");
    check(state.q_lora_rank == 1024, "q lora rank");
    check(state.q_width == 32768, "q width");
    check(state.kv_latent_width == 512, "kv latent width");
    check(state.attention_output_rank == 8192, "attention output rank");
    check(state.intermediate_size == 2048, "intermediate size");
    check(state.routed_experts == 256, "routed expert count");
    check(state.routes_per_token == 6, "routes per token");
    if (layer <= 2) {
        check(state.router_kind == DS4_V100_ROUTER_HASH, "hash router kind");
        check(state.has_hash_router, "hash router metadata present");
        check(state.router_token_capacity == 129280, "hash router token capacity");
    }

    const int32_t selected[6] = {84, 17, 31, 63, 127, 255};
    ds4_v100_route_matrices route;
    check(ds4_v100_layer_state_route_matrices(&state,
                                              (uint32_t)selected[0],
                                              &route,
                                              err,
                                              sizeof(err)) == 0,
          "route matrix views");
    check(route.gate.cols == state.hidden_size && route.gate.rows == state.intermediate_size,
          "route gate dimensions");
    check(route.up.cols == state.hidden_size && route.up.rows == state.intermediate_size,
          "route up dimensions");
    check(route.down.cols == state.intermediate_size && route.down.rows == state.hidden_size,
          "route down dimensions");

    ds4_gpu_source_row_view view;
    check(ds4_v100_bound_matrix_source_view(&route.gate, &view, err, sizeof(err)) == 0,
          "route source row view");
    check(view.cols == state.hidden_size && view.rows == state.intermediate_size,
          "source row view dimensions");

    uint64_t span = 0;
    check(ds4_v100_layer_state_ffn_arena_span(&state,
                                              selected,
                                              6,
                                              &span,
                                              err,
                                              sizeof(err)) == 0,
          "ffn arena span");
    check(stage && span <= stage->arena_bytes, "ffn arena span fits owning stage arena");
    check(state.kv_view.total_bytes == li->kv_view.total_bytes, "kv view snapshot");

    uint64_t attn_span = 0;
    check(ds4_v100_layer_state_attention_arena_span(&state,
                                                    &attn_span,
                                                    err,
                                                    sizeof(err)) == 0,
          "attention arena span");
    check(stage && attn_span <= stage->arena_bytes, "attention arena span fits owning stage arena");

    printf("v100_layer_state_smoke: layer=%d stage=%d gpu=%d class=%s router=%s hidden=%u q=%u kv=%u mid=%u experts=%u ffn_span=%" PRIu64 " attn_span=%" PRIu64 " %s\n",
           state.layer_id,
           state.stage_id,
           state.owning_gpu,
           ds4_v100_layer_class_name(state.layer_class),
           ds4_v100_router_kind_name(state.router_kind),
           state.hidden_size,
           state.q_width,
           state.kv_latent_width,
           state.intermediate_size,
           state.routed_experts,
           span,
           attn_span,
           failures ? "FAIL" : "ok");

    ds4_v100_context_close(ctx);
    return failures ? 1 : 0;
}
