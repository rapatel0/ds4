#include "ds4_gpu.h"
#include "ds4_v100_layer_state.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef struct {
    const char *index_path;
    const char *tm_index_path;
    const char *tm_dir;
    int layer;
    int owner_gpu;
    int peer_gpu;
    uint32_t tokens;
    uint32_t iters;
} options;

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_v100_tp_routed_ffn_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/cuda_v100_tp_routed_ffn_smoke --index FILE --tm-index FILE --tm-dir DIR [options]\n"
            "\n"
            "Options:\n"
            "  --layer N       Layer to test. Default: 3\n"
            "  --owner-gpu N   Owner GPU. Default: 0\n"
            "  --peer-gpu N    TP peer GPU. Default: 3\n"
            "  --tokens N      Token rows to batch. Default: 16\n"
            "  --iters N       Timing iterations. Default: 20\n");
}

static int parse_int_arg(const char *s, const char *name, int max_v) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !s || !*s || !end || *end != '\0' || v < 0 || v > max_v) {
        fprintf(stderr, "cuda_v100_tp_routed_ffn_smoke: invalid %s: %s\n",
                name, s ? s : "(null)");
        exit(2);
    }
    return (int)v;
}

static void parse_args(int argc, char **argv, options *opt) {
    memset(opt, 0, sizeof(*opt));
    opt->index_path = getenv("DS4_V100_PACK_INDEX");
    opt->tm_index_path = getenv("DS4_V100_TURBOMIND_INDEX");
    opt->tm_dir = getenv("DS4_V100_TURBOMIND_DIR");
    opt->layer = 3;
    opt->owner_gpu = 0;
    opt->peer_gpu = 3;
    opt->tokens = 16;
    opt->iters = 20;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = NULL;
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            exit(0);
        }
        if (i + 1 < argc) v = argv[i + 1];
        if (!strcmp(a, "--index") && v) {
            opt->index_path = argv[++i];
        } else if (!strcmp(a, "--tm-index") && v) {
            opt->tm_index_path = argv[++i];
        } else if (!strcmp(a, "--tm-dir") && v) {
            opt->tm_dir = argv[++i];
        } else if (!strcmp(a, "--layer") && v) {
            opt->layer = parse_int_arg(argv[++i], a, 42);
        } else if (!strcmp(a, "--owner-gpu") && v) {
            opt->owner_gpu = parse_int_arg(argv[++i], a, 15);
        } else if (!strcmp(a, "--peer-gpu") && v) {
            opt->peer_gpu = parse_int_arg(argv[++i], a, 15);
        } else if (!strcmp(a, "--tokens") && v) {
            opt->tokens = (uint32_t)parse_int_arg(argv[++i], a, 1024);
        } else if (!strcmp(a, "--iters") && v) {
            opt->iters = (uint32_t)parse_int_arg(argv[++i], a, 10000);
        } else {
            usage(stderr);
            exit(2);
        }
    }
    if (!opt->index_path || !opt->tm_index_path || !opt->tm_dir) {
        usage(stderr);
        exit(2);
    }
}

static char *path_join(const char *dir, const char *base) {
    const size_t nd = strlen(dir);
    const size_t nb = strlen(base);
    const int slash = nd > 0 && dir[nd - 1] != '/';
    char *out = (char *)malloc(nd + (size_t)slash + nb + 1u);
    if (!out) return NULL;
    memcpy(out, dir, nd);
    if (slash) out[nd] = '/';
    memcpy(out + nd + (size_t)slash, base, nb);
    out[nd + (size_t)slash + nb] = '\0';
    return out;
}

static uint8_t *read_file(const char *path, uint64_t *out_bytes) {
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size < 0) {
        fprintf(stderr, "cuda_v100_tp_routed_ffn_smoke: cannot stat %s: %s\n",
                path, strerror(errno));
        return NULL;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "cuda_v100_tp_routed_ffn_smoke: cannot open %s: %s\n",
                path, strerror(errno));
        return NULL;
    }
    uint64_t bytes = (uint64_t)st.st_size;
    uint8_t *buf = (uint8_t *)malloc((size_t)(bytes ? bytes : 1));
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)bytes, fp);
    fclose(fp);
    if (got != (size_t)bytes) {
        fprintf(stderr,
                "cuda_v100_tp_routed_ffn_smoke: short read %s got=%zu need=%" PRIu64 "\n",
                path, got, bytes);
        free(buf);
        return NULL;
    }
    *out_bytes = bytes;
    return buf;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static float prng_f32(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    const uint32_t bits = (*state >> 9) | 0x3f800000u;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return (v - 1.5f) * 0.125f;
}

static int run_ref(const ds4_v100_layer_state *state,
                   const ds4_gpu_arena *arena,
                   const ds4_gpu_tensor *selected,
                   const ds4_gpu_tensor *weights,
                   const ds4_gpu_tensor *x,
                   uint32_t tokens,
                   ds4_gpu_tensor *out) {
    return ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_f32(
        arena,
        &state->turbomind_gate_up_view,
        &state->turbomind_down_view,
        state->hidden_size,
        state->intermediate_size,
        state->routed_experts,
        selected,
        weights,
        state->routes_per_token,
        x,
        tokens,
        out);
}

static int run_tp_half(const ds4_v100_layer_state *state,
                       const ds4_gpu_arena *arena,
                       uint32_t half,
                       const ds4_gpu_tensor *selected,
                       const ds4_gpu_tensor *weights,
                       const ds4_gpu_tensor *x,
                       uint32_t tokens,
                       ds4_gpu_tensor *out) {
    return ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_f32(
        arena,
        &state->turbomind_tp2_gate_up_view[half],
        &state->turbomind_tp2_down_view[half],
        state->hidden_size,
        state->intermediate_size / 2u,
        state->routed_experts,
        selected,
        weights,
        state->routes_per_token,
        x,
        tokens,
        out);
}

int main(int argc, char **argv) {
    options opt;
    parse_args(argc, argv, &opt);

    ds4_v100_context_options ctx_opts;
    ds4_v100_context_options_init(&ctx_opts);
    ctx_opts.pack_index_path = opt.index_path;
    ctx_opts.turbomind_pack_index_path = opt.tm_index_path;
    ctx_opts.kv_ctx_tokens = 262144;
    ctx_opts.kv_active_slots = opt.tokens;

    char err[512] = {0};
    ds4_v100_context *ctx = NULL;
    if (ds4_v100_context_open(&ctx, &ctx_opts, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_tp_routed_ffn_smoke: %s\n", err);
        return 1;
    }
    ds4_v100_layer_state state;
    if (ds4_v100_layer_state_init(&state, ctx, opt.layer, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_tp_routed_ffn_smoke: %s\n", err);
        ds4_v100_context_close(ctx);
        return 1;
    }
    check(state.has_turbomind_routed, "normal TurboMind routed binding missing");
    check(state.has_turbomind_fused_gate_up, "normal fused gate_up binding missing");
    check(state.has_turbomind_tp2_routed, "TP2 routed binding missing");
    check(state.owning_gpu == opt.owner_gpu, "layer owner GPU does not match --owner-gpu");
    check(state.turbomind_tp2_peer_gpu == opt.peer_gpu, "TP peer GPU does not match --peer-gpu");
    check(state.routes_per_token == 6, "expected six routed experts per token");
    check(state.routed_experts == state.turbomind_tp2_gate_up_view[0].experts_packed,
          "TP2 gate/up tp0 must pack all experts for this smoke");
    check(state.routed_experts == state.turbomind_tp2_gate_up_view[1].experts_packed,
          "TP2 gate/up tp1 must pack all experts for this smoke");
    check(state.routed_experts == state.turbomind_tp2_down_view[0].experts_packed,
          "TP2 down tp0 must pack all experts for this smoke");
    check(state.routed_experts == state.turbomind_tp2_down_view[1].experts_packed,
          "TP2 down tp1 must pack all experts for this smoke");
    if (failures) {
        ds4_v100_context_close(ctx);
        return 1;
    }

    char *owner_path = path_join(opt.tm_dir, state.turbomind_gate_up_binding.shard_file);
    char *peer_path = path_join(opt.tm_dir, state.turbomind_tp2_gate_up_binding[1].shard_file);
    if (!owner_path || !peer_path) {
        fprintf(stderr, "cuda_v100_tp_routed_ffn_smoke: path allocation failed\n");
        ds4_v100_context_close(ctx);
        return 1;
    }
    uint64_t owner_bytes = 0;
    uint64_t peer_bytes = 0;
    uint8_t *owner_buf = read_file(owner_path, &owner_bytes);
    uint8_t *peer_buf = read_file(peer_path, &peer_bytes);
    if (!owner_buf || !peer_buf) {
        free(owner_path);
        free(peer_path);
        free(owner_buf);
        free(peer_buf);
        ds4_v100_context_close(ctx);
        return 1;
    }

    if (!ds4_gpu_init()) {
        fprintf(stderr, "cuda_v100_tp_routed_ffn_smoke: ds4_gpu_init failed\n");
        free(owner_path);
        free(peer_path);
        free(owner_buf);
        free(peer_buf);
        ds4_v100_context_close(ctx);
        return 1;
    }
    check(ds4_gpu_enable_peer_access(opt.owner_gpu, opt.peer_gpu),
          "peer access enable failed");

    ds4_gpu_arena *owner_arena = NULL;
    ds4_gpu_arena *peer_arena = NULL;
    check(ds4_gpu_arena_open(&owner_arena, opt.owner_gpu, owner_bytes) == 0,
          "owner arena open failed");
    check(ds4_gpu_arena_open(&peer_arena, opt.peer_gpu, peer_bytes) == 0,
          "peer arena open failed");
    check(owner_arena && ds4_gpu_arena_upload(owner_arena, 0, owner_buf, owner_bytes) == 0,
          "owner arena upload failed");
    check(peer_arena && ds4_gpu_arena_upload(peer_arena, 0, peer_buf, peer_bytes) == 0,
          "peer arena upload failed");
    free(owner_buf);
    free(peer_buf);
    if (failures) goto done;

    const uint32_t hidden = state.hidden_size;
    const uint32_t routes = state.routes_per_token;
    const uint64_t hidden_values = (uint64_t)opt.tokens * hidden;
    const uint64_t route_values = (uint64_t)opt.tokens * routes;
    float *x_host = (float *)malloc((size_t)hidden_values * sizeof(float));
    int32_t *selected_host = (int32_t *)malloc((size_t)route_values * sizeof(int32_t));
    float *weights_host = (float *)malloc((size_t)route_values * sizeof(float));
    float *ref_host = (float *)malloc((size_t)hidden_values * sizeof(float));
    float *tp_host = (float *)malloc((size_t)hidden_values * sizeof(float));
    check(x_host && selected_host && weights_host && ref_host && tp_host,
          "host allocation failed");
    if (failures) goto done;

    uint32_t rng = 0xD504163u + (uint32_t)opt.layer * 97u + opt.tokens;
    for (uint64_t i = 0; i < hidden_values; i++) x_host[i] = prng_f32(&rng);
    for (uint32_t t = 0; t < opt.tokens; t++) {
        float sum = 0.0f;
        for (uint32_t r = 0; r < routes; r++) {
            const uint64_t idx = (uint64_t)t * routes + r;
            selected_host[idx] = (int32_t)((t * 13u + r * 17u) % state.routed_experts);
            weights_host[idx] = 0.05f + 0.01f * (float)(r + 1u);
            sum += weights_host[idx];
        }
        for (uint32_t r = 0; r < routes; r++) {
            weights_host[(uint64_t)t * routes + r] /= sum;
        }
    }

    ds4_gpu_set_device(opt.owner_gpu);
    ds4_gpu_tensor *x_owner = ds4_gpu_tensor_alloc(hidden_values * sizeof(float));
    ds4_gpu_tensor *selected_owner = ds4_gpu_tensor_alloc(route_values * sizeof(int32_t));
    ds4_gpu_tensor *weights_owner = ds4_gpu_tensor_alloc(route_values * sizeof(float));
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(hidden_values * sizeof(float));
    ds4_gpu_tensor *owner_out = ds4_gpu_tensor_alloc(hidden_values * sizeof(float));
    ds4_gpu_tensor *peer_recv = ds4_gpu_tensor_alloc(hidden_values * sizeof(float));
    ds4_gpu_tensor *tp_sum = ds4_gpu_tensor_alloc(hidden_values * sizeof(float));
    ds4_gpu_set_device(opt.peer_gpu);
    ds4_gpu_tensor *x_peer = ds4_gpu_tensor_alloc(hidden_values * sizeof(float));
    ds4_gpu_tensor *selected_peer = ds4_gpu_tensor_alloc(route_values * sizeof(int32_t));
    ds4_gpu_tensor *weights_peer = ds4_gpu_tensor_alloc(route_values * sizeof(float));
    ds4_gpu_tensor *peer_out = ds4_gpu_tensor_alloc(hidden_values * sizeof(float));
    check(x_owner && selected_owner && weights_owner && ref_out &&
          owner_out && peer_recv && tp_sum && x_peer && selected_peer &&
          weights_peer && peer_out,
          "device tensor allocation failed");
    if (failures) goto done_tensors;

    check(ds4_gpu_tensor_write(x_owner, 0, x_host, hidden_values * sizeof(float)),
          "owner x upload failed");
    check(ds4_gpu_tensor_write(selected_owner, 0, selected_host, route_values * sizeof(int32_t)),
          "owner selected upload failed");
    check(ds4_gpu_tensor_write(weights_owner, 0, weights_host, route_values * sizeof(float)),
          "owner weights upload failed");
    check(ds4_gpu_tensor_copy(x_peer, 0, x_owner, 0, hidden_values * sizeof(float)),
          "peer x copy failed");
    check(ds4_gpu_tensor_copy(selected_peer, 0, selected_owner, 0, route_values * sizeof(int32_t)),
          "peer selected copy failed");
    check(ds4_gpu_tensor_copy(weights_peer, 0, weights_owner, 0, route_values * sizeof(float)),
          "peer weights copy failed");
    if (failures) goto done_tensors;

    check(run_ref(&state, owner_arena, selected_owner, weights_owner, x_owner,
                  opt.tokens, ref_out) == 0,
          "reference single-GPU routed FFN failed");
    check(run_tp_half(&state, owner_arena, 0, selected_owner, weights_owner,
                      x_owner, opt.tokens, owner_out) == 0,
          "TP owner half failed");
    check(run_tp_half(&state, peer_arena, 1, selected_peer, weights_peer,
                      x_peer, opt.tokens, peer_out) == 0,
          "TP peer half failed");
    check(ds4_gpu_tensor_copy(peer_recv, 0, peer_out, 0, hidden_values * sizeof(float)),
          "peer output copy failed");
    ds4_gpu_set_device(opt.owner_gpu);
    check(ds4_gpu_add_tensor(tp_sum, owner_out, peer_recv, (uint32_t)hidden_values),
          "TP partial sum failed");
    check(ds4_gpu_synchronize(), "initial synchronize failed");
    check(ds4_gpu_tensor_read(ref_out, 0, ref_host, hidden_values * sizeof(float)),
          "reference output read failed");
    check(ds4_gpu_tensor_read(tp_sum, 0, tp_host, hidden_values * sizeof(float)),
          "TP output read failed");
    if (failures) goto done_tensors;

    double sum_abs = 0.0;
    double sum_ref = 0.0;
    float max_abs = 0.0f;
    uint64_t bad = 0;
    uint64_t nan = 0;
    for (uint64_t i = 0; i < hidden_values; i++) {
        const float ref = ref_host[i];
        const float got = tp_host[i];
        if (!isfinite(ref) || !isfinite(got)) {
            nan++;
            bad++;
            continue;
        }
        const float d = fabsf(ref - got);
        const float tol = fmaxf(16.0f, 0.05f * fabsf(ref));
        if (d > max_abs) max_abs = d;
        sum_abs += (double)d;
        sum_ref += (double)fabsf(ref);
        if (d > tol) bad++;
    }
    const double rel = sum_ref > 0.0 ? sum_abs / sum_ref : 0.0;
    const double bad_frac = hidden_values ? (double)bad / (double)hidden_values : 0.0;
    check(nan == 0, "non-finite TP comparison output");
    check(rel < 0.02, "TP relative error too high");
    check(bad_frac < 0.001, "TP element error fraction too high");

    const uint32_t iters = opt.iters ? opt.iters : 1u;
    double t0 = now_ms();
    for (uint32_t i = 0; i < iters; i++) {
        if (run_ref(&state, owner_arena, selected_owner, weights_owner, x_owner,
                    opt.tokens, ref_out) != 0) {
            check(0, "reference timing iteration failed");
            break;
        }
    }
    ds4_gpu_set_device(opt.owner_gpu);
    check(ds4_gpu_synchronize(), "reference timing synchronize failed");
    const double ref_ms = (now_ms() - t0) / (double)iters;

    t0 = now_ms();
    for (uint32_t i = 0; i < iters; i++) {
        if (run_tp_half(&state, owner_arena, 0, selected_owner, weights_owner,
                        x_owner, opt.tokens, owner_out) != 0) {
            check(0, "owner timing iteration failed");
            break;
        }
    }
    ds4_gpu_set_device(opt.owner_gpu);
    check(ds4_gpu_synchronize(), "owner timing synchronize failed");
    const double owner_ms = (now_ms() - t0) / (double)iters;

    t0 = now_ms();
    for (uint32_t i = 0; i < iters; i++) {
        if (run_tp_half(&state, peer_arena, 1, selected_peer, weights_peer,
                        x_peer, opt.tokens, peer_out) != 0) {
            check(0, "peer timing iteration failed");
            break;
        }
    }
    ds4_gpu_set_device(opt.peer_gpu);
    check(ds4_gpu_synchronize(), "peer timing synchronize failed");
    const double peer_ms = (now_ms() - t0) / (double)iters;

    t0 = now_ms();
    for (uint32_t i = 0; i < iters; i++) {
        if (!ds4_gpu_tensor_copy(x_peer, 0, x_owner, 0, hidden_values * sizeof(float)) ||
            !ds4_gpu_tensor_copy(selected_peer, 0, selected_owner, 0, route_values * sizeof(int32_t)) ||
            !ds4_gpu_tensor_copy(weights_peer, 0, weights_owner, 0, route_values * sizeof(float))) {
            check(0, "TP input copy timing iteration failed");
            break;
        }
    }
    const double copy_in_ms = (now_ms() - t0) / (double)iters;

    t0 = now_ms();
    for (uint32_t i = 0; i < iters; i++) {
        if (!ds4_gpu_tensor_copy(peer_recv, 0, peer_out, 0, hidden_values * sizeof(float))) {
            check(0, "TP output copy timing iteration failed");
            break;
        }
    }
    const double copy_out_ms = (now_ms() - t0) / (double)iters;

    ds4_gpu_set_device(opt.owner_gpu);
    t0 = now_ms();
    for (uint32_t i = 0; i < iters; i++) {
        if (!ds4_gpu_add_tensor(tp_sum, owner_out, peer_recv, (uint32_t)hidden_values)) {
            check(0, "TP sum timing iteration failed");
            break;
        }
    }
    check(ds4_gpu_synchronize(), "TP sum timing synchronize failed");
    const double sum_ms = (now_ms() - t0) / (double)iters;

    t0 = now_ms();
    for (uint32_t i = 0; i < iters; i++) {
        if (!ds4_gpu_tensor_copy(x_peer, 0, x_owner, 0, hidden_values * sizeof(float)) ||
            !ds4_gpu_tensor_copy(selected_peer, 0, selected_owner, 0, route_values * sizeof(int32_t)) ||
            !ds4_gpu_tensor_copy(weights_peer, 0, weights_owner, 0, route_values * sizeof(float)) ||
            run_tp_half(&state, owner_arena, 0, selected_owner, weights_owner,
                        x_owner, opt.tokens, owner_out) != 0 ||
            run_tp_half(&state, peer_arena, 1, selected_peer, weights_peer,
                        x_peer, opt.tokens, peer_out) != 0 ||
            !ds4_gpu_tensor_copy(peer_recv, 0, peer_out, 0, hidden_values * sizeof(float))) {
            check(0, "TP total timing iteration failed");
            break;
        }
        ds4_gpu_set_device(opt.owner_gpu);
        if (!ds4_gpu_add_tensor(tp_sum, owner_out, peer_recv, (uint32_t)hidden_values)) {
            check(0, "TP total add failed");
            break;
        }
    }
    ds4_gpu_set_device(opt.owner_gpu);
    check(ds4_gpu_synchronize(), "TP total timing owner synchronize failed");
    ds4_gpu_set_device(opt.peer_gpu);
    check(ds4_gpu_synchronize(), "TP total timing peer synchronize failed");
    const double total_ms = (now_ms() - t0) / (double)iters;

    fprintf(stderr,
            "cuda_v100_tp_routed_ffn_smoke: layer=%d pair=%d,%d tokens=%u routes=%u "
            "values=%" PRIu64 " max_abs=%.6g rel=%.6g bad=%" PRIu64
            " bad_frac=%.6g ref_ms=%.4f owner_ms=%.4f peer_ms=%.4f "
            "copy_in_ms=%.4f copy_out_ms=%.4f sum_ms=%.4f total_ms=%.4f speedup=%.3fx "
            "owner_bytes=%" PRIu64 " peer_bytes=%" PRIu64 "\n",
            opt.layer,
            opt.owner_gpu,
            opt.peer_gpu,
            opt.tokens,
            opt.tokens * routes,
            hidden_values,
            max_abs,
            rel,
            bad,
            bad_frac,
            ref_ms,
            owner_ms,
            peer_ms,
            copy_in_ms,
            copy_out_ms,
            sum_ms,
            total_ms,
            total_ms > 0.0 ? ref_ms / total_ms : 0.0,
            owner_bytes,
            peer_bytes);

done_tensors:
    ds4_gpu_tensor_free(peer_out);
    ds4_gpu_tensor_free(weights_peer);
    ds4_gpu_tensor_free(selected_peer);
    ds4_gpu_tensor_free(x_peer);
    ds4_gpu_tensor_free(tp_sum);
    ds4_gpu_tensor_free(peer_recv);
    ds4_gpu_tensor_free(owner_out);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(weights_owner);
    ds4_gpu_tensor_free(selected_owner);
    ds4_gpu_tensor_free(x_owner);
    free(tp_host);
    free(ref_host);
    free(weights_host);
    free(selected_host);
    free(x_host);
done:
    ds4_gpu_arena_close(peer_arena);
    ds4_gpu_arena_close(owner_arena);
    ds4_gpu_cleanup();
    free(owner_path);
    free(peer_path);
    ds4_v100_context_close(ctx);
    if (failures) {
        fprintf(stderr, "cuda_v100_tp_routed_ffn_smoke: FAIL\n");
        return 1;
    }
    fprintf(stderr, "cuda_v100_tp_routed_ffn_smoke: PASS\n");
    return 0;
}
