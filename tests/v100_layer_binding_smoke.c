#include "ds4_source_formats.h"
#include "engine/context.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "v100_layer_binding_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/v100_layer_binding_smoke --index FILE [--layer N]\n");
}

static int parse_int(const char *s, const char *name) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || !*s || !end || *end != '\0' || v < 0 || v > 42) {
        fprintf(stderr, "v100_layer_binding_smoke: invalid %s: %s\n", name, s ? s : "(null)");
        exit(2);
    }
    return (int)v;
}

static void expect_shape(const ds4_tensor_binding *b,
                         uint32_t n,
                         uint64_t a,
                         uint64_t c,
                         uint64_t d,
                         const char *label) {
    check(b->n_shape_dims == n, label);
    if (n > 0) check(b->shape[0] == a, label);
    if (n > 1) check(b->shape[1] == c, label);
    if (n > 2) check(b->shape[2] == d, label);
}

static void expect_binding(const ds4_tensor_binding *b,
                           const char *dtype,
                           ds4_exec_kind exec_kind,
                           int layer,
                           int gpu,
                           const char *label) {
    check(b->semantic_tensor_id && b->semantic_tensor_id[0], label);
    check(!strcmp(b->source_dtype, dtype), label);
    check(b->policy.exec_kind == exec_kind, label);
    check(b->layer_id == layer, label);
    check(b->owning_gpu == gpu, label);
    check(b->byte_length != 0, label);
    check(b->shard_file && !strcmp(b->shard_file, gpu == 7 ? "gpu7.weights" : "gpu0.weights"), label);
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
            layer = parse_int(argv[++i], "--layer");
        } else {
            usage(stderr);
            return 2;
        }
    }
    if (!index) {
        usage(stderr);
        return 2;
    }

    ds4_context_options opts;
    ds4_context_options_init(&opts);
    opts.pack_index_path = index;
    opts.kv_ctx_tokens = 1048576;
    opts.kv_active_slots = 1;

    char err[512] = {0};
    ds4_context *ctx = NULL;
    check(ds4_context_open(&ctx, &opts, err, sizeof(err)) == 0, "context open");
    if (!ctx) {
        fprintf(stderr, "v100_layer_binding_smoke: %s\n", err);
        return 1;
    }

    const int gpu = ds4_stage_for_layer(layer);
    ds4_tensor_binding gate;
    ds4_tensor_binding up;
    ds4_tensor_binding down;
    ds4_tensor_binding shared_gate;
    ds4_tensor_binding shared_up;
    ds4_tensor_binding shared_down;
    ds4_tensor_binding router;
    ds4_tensor_binding hc;
    ds4_tensor_binding head;

    check(ds4_context_require_layer_tensor_binding(ctx, layer, "ffn_gate_exps.weight", &gate, err, sizeof(err)) == 0,
          "gate expert binding");
    check(ds4_context_require_layer_tensor_binding(ctx, layer, "ffn_up_exps.weight", &up, err, sizeof(err)) == 0,
          "up expert binding");
    check(ds4_context_require_layer_tensor_binding(ctx, layer, "ffn_down_exps.weight", &down, err, sizeof(err)) == 0,
          "down expert binding");
    check(ds4_context_require_layer_tensor_binding(ctx, layer, "ffn_gate_shexp.weight", &shared_gate, err, sizeof(err)) == 0,
          "shared gate binding");
    check(ds4_context_require_layer_tensor_binding(ctx, layer, "ffn_up_shexp.weight", &shared_up, err, sizeof(err)) == 0,
          "shared up binding");
    check(ds4_context_require_layer_tensor_binding(ctx, layer, "ffn_down_shexp.weight", &shared_down, err, sizeof(err)) == 0,
          "shared down binding");
    check(ds4_context_require_layer_tensor_binding(ctx, layer, layer <= 2 ? "ffn_gate_tid2eid" : "exp_probs_b", &router, err, sizeof(err)) == 0,
          "router binding");
    check(ds4_context_require_layer_tensor_binding(ctx, layer, "hc_ffn_fn", &hc, err, sizeof(err)) == 0,
          "hc binding");
    check(ds4_context_output_head_binding(ctx, &head, err, sizeof(err)) == 0,
          "output head binding");

    expect_binding(&gate, "mxfp4", DS4_V100_EXEC_LOWBIT_KERNEL, layer, gpu, "gate expert policy");
    expect_binding(&up, "mxfp4", DS4_V100_EXEC_LOWBIT_KERNEL, layer, gpu, "up expert policy");
    expect_binding(&down, "mxfp4", DS4_V100_EXEC_LOWBIT_KERNEL, layer, gpu, "down expert policy");
    expect_binding(&shared_gate, "f8_e4m3_b128", DS4_V100_EXEC_F16_HMMA, layer, gpu, "shared gate policy");
    expect_binding(&shared_up, "f8_e4m3_b128", DS4_V100_EXEC_F16_HMMA, layer, gpu, "shared up policy");
    expect_binding(&shared_down, "f8_e4m3_b128", DS4_V100_EXEC_F16_HMMA, layer, gpu, "shared down policy");
    check(router.policy.exec_kind == DS4_V100_EXEC_F32_CONTROL, "router policy");
    check(hc.policy.family == DS4_V100_FAMILY_HC_CONTROL, "hc family");
    expect_binding(&head, "bf16", DS4_V100_EXEC_DIAGNOSTIC_ONLY, -1, 7, "output head policy");

    expect_shape(&gate, 3, 4096, 2048, 256, "gate expert shape");
    expect_shape(&up, 3, 4096, 2048, 256, "up expert shape");
    expect_shape(&down, 3, 2048, 4096, 256, "down expert shape");
    expect_shape(&shared_gate, 2, 4096, 2048, 0, "shared gate shape");
    expect_shape(&shared_up, 2, 4096, 2048, 0, "shared up shape");
    expect_shape(&shared_down, 2, 2048, 4096, 0, "shared down shape");
    expect_shape(&head, 2, 4096, 129280, 0, "output head shape");

    const uint64_t routed_row = ds4_src_mxfp4_row_bytes(gate.shape[0]);
    const uint64_t routed_expert = gate.shape[1] * routed_row;
    const uint64_t shared_row = ds4_src_f8_e4m3_b128_row_bytes(shared_gate.shape[0]);
    check(routed_expert * gate.shape[2] == gate.byte_length, "routed expert byte length");
    check(shared_row * shared_gate.shape[1] == shared_gate.byte_length, "shared gate byte length");
    check(gate.shard_offset != up.shard_offset &&
          gate.shard_offset != down.shard_offset &&
          up.shard_offset != down.shard_offset,
          "routed descriptor spans are distinct");

    ds4_tensor_binding missing;
    check(ds4_context_require_layer_tensor_binding(ctx, layer, "missing.weight", &missing, err, sizeof(err)) != 0,
          "missing tensor should fail");
    check(ds4_context_require_layer_tensor_binding(ctx, -1, "ffn_gate_exps.weight", &missing, err, sizeof(err)) != 0,
          "bad layer should fail");

    printf("v100_layer_binding_smoke: layer=%d gpu=%d routed_expert_bytes=%" PRIu64 " shared_row_bytes=%" PRIu64 " ok\n",
           layer,
           gpu,
           routed_expert,
           shared_row);
    ds4_context_close(ctx);
    return failures ? 1 : 0;
}
