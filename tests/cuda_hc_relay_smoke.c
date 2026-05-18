#include "ds4_v100_context.h"

#include <stdio.h>

int main(void) {
    char err[512];
    ds4_v100_device_fact facts[DS4_V100_EXPECTED_GPUS];
    int n = 0;
    if (ds4_v100_cuda_collect_device_facts(facts, DS4_V100_EXPECTED_GPUS,
                                           &n, err, sizeof(err))) {
        fprintf(stderr, "cuda_hc_relay_smoke: %s\n", err);
        return 1;
    }
    if (n < 1) {
        fprintf(stderr, "cuda_hc_relay_smoke: no CUDA devices visible\n");
        return 1;
    }

    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    opts.expected_gpus = n >= 2 ? 2 : 1;
    opts.relay_max_active_slots = 2;
    opts.scratch_bytes_per_gpu = 1024 * 1024;
    opts.enable_f32_debug_relay = true;

    ds4_v100_cuda_context *ctx = NULL;
    if (ds4_v100_cuda_context_open(&ctx, &opts, err, sizeof(err))) {
        fprintf(stderr, "cuda_hc_relay_smoke: %s\n", err);
        return 1;
    }
    if (ds4_v100_cuda_context_relay_smoke(ctx, 0, 0, DS4_V100_RELAY_F16,
                                          1, err, sizeof(err))) {
        fprintf(stderr, "cuda_hc_relay_smoke: f16 loopback: %s\n", err);
        ds4_v100_cuda_context_close(ctx);
        return 1;
    }
    if (ds4_v100_cuda_context_relay_smoke(ctx, 0, 0, DS4_V100_RELAY_F32_DEBUG,
                                          1, err, sizeof(err))) {
        fprintf(stderr, "cuda_hc_relay_smoke: f32 loopback: %s\n", err);
        ds4_v100_cuda_context_close(ctx);
        return 1;
    }
    if (opts.expected_gpus >= 2) {
        if (!facts[0].peer_access[1]) {
            fprintf(stderr, "cuda_hc_relay_smoke: device 0 cannot access peer 1\n");
            ds4_v100_cuda_context_close(ctx);
            return 1;
        }
        if (ds4_v100_cuda_context_relay_smoke(ctx, 0, 1, DS4_V100_RELAY_F16,
                                              1, err, sizeof(err))) {
            fprintf(stderr, "cuda_hc_relay_smoke: f16 peer: %s\n", err);
            ds4_v100_cuda_context_close(ctx);
            return 1;
        }
    }
    ds4_v100_cuda_context_close(ctx);
    puts("cuda_hc_relay_smoke: ok");
    return 0;
}
