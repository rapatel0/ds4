#include "ds4_v100_context.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t f32_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

static float f32_from_bits(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t f32_to_f16_bits(float f) {
    const uint32_t x = f32_bits(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x007fffffu;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t out = mant >> shift;
        if ((mant >> (shift - 1u)) & 1u) out++;
        return (uint16_t)(sign | out);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t out = (uint32_t)exp << 10;
    mant += 0x00001000u;
    if (mant & 0x00800000u) {
        mant = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
        out = (uint32_t)exp << 10;
    }
    return (uint16_t)(sign | out | (mant >> 13));
}

static float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) return f32_from_bits(sign);
        exp = 1;
        while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x03ffu;
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return f32_from_bits(bits);
}

static int parse_u64(const char *s, unsigned long long *out) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (!end || *end != '\0') return 1;
    *out = v;
    return 0;
}

static int expect_f16_row(const char *label,
                          const uint16_t *got,
                          const float *want,
                          uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const uint16_t want_bits = f32_to_f16_bits(want[i]);
        if (got[i] != want_bits) {
            fprintf(stderr,
                    "cuda_v100_context_smoke: %s[%u] got 0x%04x %.8g expected 0x%04x %.8g\n",
                    label,
                    i,
                    got[i],
                    f16_bits_to_f32(got[i]),
                    want_bits,
                    f16_bits_to_f32(want_bits));
            return 1;
        }
    }
    return 0;
}

static int expect_state_sample(const char *label,
                               const float *got,
                               uint64_t values,
                               const float *row,
                               uint32_t dim,
                               float row_scale,
                               float ratio_scale,
                               uint32_t ratio) {
    const uint64_t probes[] = {0, 1, 127, 128, 511, 512, values / 2, values - 1};
    for (uint32_t p = 0; p < sizeof(probes) / sizeof(probes[0]); p++) {
        const uint64_t i = probes[p];
        if (i >= values) continue;
        const uint32_t lane = (uint32_t)(i % dim);
        const uint32_t state_row = (uint32_t)(i / dim);
        const float want = row[lane] + (float)state_row * row_scale +
                           (float)ratio * ratio_scale;
        if (fabsf(got[i] - want) > 1e-5f) {
            fprintf(stderr,
                    "cuda_v100_context_smoke: %s[%llu] got %.8g expected %.8g\n",
                    label,
                    (unsigned long long)i,
                    got[i],
                    want);
            return 1;
        }
    }
    return 0;
}

static int read_f16_row(ds4_v100_cuda_context *ctx,
                        const ds4_v100_cuda_layer_kv_view *view,
                        uint64_t base_offset,
                        uint64_t row,
                        uint32_t dim,
                        uint16_t *out,
                        char *err,
                        size_t errlen) {
    const uint64_t off = base_offset + row * dim * sizeof(uint16_t);
    return ds4_v100_cuda_context_read_kv_arena(ctx, view->stage_id, off, out,
                                               (uint64_t)dim * sizeof(uint16_t),
                                               err, errlen);
}

int main(int argc, char **argv) {
    int production = 0;
    int requested_stages = 0;
    const char *pack_index = NULL;
    unsigned long long planned_kv_mib = 0;
    unsigned long long reserve_mib = 2048;
    unsigned long long output_head_mib = 0;
    unsigned long long mtp_mib = 0;
    unsigned long long kv_ctx = 0;
    unsigned long long kv_slots = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--production")) {
            production = 1;
        } else if (!strcmp(argv[i], "--stages") && i + 1 < argc) {
            requested_stages = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--pack-index") && i + 1 < argc) {
            pack_index = argv[++i];
        } else if (!strcmp(argv[i], "--planned-kv-mib") && i + 1 < argc) {
            if (parse_u64(argv[++i], &planned_kv_mib)) return 2;
        } else if (!strcmp(argv[i], "--kv-ctx") && i + 1 < argc) {
            if (parse_u64(argv[++i], &kv_ctx)) return 2;
        } else if (!strcmp(argv[i], "--kv-slots") && i + 1 < argc) {
            if (parse_u64(argv[++i], &kv_slots) || kv_slots == 0) return 2;
        } else if (!strcmp(argv[i], "--reserve-mib") && i + 1 < argc) {
            if (parse_u64(argv[++i], &reserve_mib)) return 2;
        } else if (!strcmp(argv[i], "--output-head-mib") && i + 1 < argc) {
            if (parse_u64(argv[++i], &output_head_mib)) return 2;
        } else if (!strcmp(argv[i], "--mtp-mib") && i + 1 < argc) {
            if (parse_u64(argv[++i], &mtp_mib)) return 2;
        } else {
            fprintf(stderr,
                    "usage: tests/cuda_v100_context_smoke [--production] [--stages N]\n"
                    "                                    [--pack-index PATH] [--planned-kv-mib N]\n"
                    "                                    [--kv-ctx N] [--kv-slots N]\n"
                    "                                    [--reserve-mib N] [--output-head-mib N] [--mtp-mib N]\n");
            return 2;
        }
    }

    char err[512];
    ds4_v100_device_fact facts[DS4_V100_EXPECTED_GPUS];
    int n = 0;
    if (ds4_v100_cuda_collect_device_facts(facts, DS4_V100_EXPECTED_GPUS,
                                           &n, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_context_smoke: %s\n", err);
        return 1;
    }
    if (n < 1) {
        fprintf(stderr, "cuda_v100_context_smoke: no CUDA devices visible\n");
        return 1;
    }
    int stages = requested_stages ? requested_stages : (production ? DS4_V100_EXPECTED_GPUS : (n >= 2 ? 2 : 1));
    ds4_v100_context_options opts;
    ds4_v100_context_options_init(&opts);
    opts.expected_gpus = stages;
    opts.pack_index_path = pack_index;
    opts.relay_max_active_slots = 1;
    opts.scratch_bytes_per_gpu = 1024 * 1024;
    opts.enable_f32_debug_relay = true;
    opts.require_production_topology = production != 0;
    opts.planned_kv_bytes_per_gpu = planned_kv_mib * 1048576ull;
    opts.kv_ctx_tokens = kv_ctx;
    opts.kv_active_slots = kv_slots;
    opts.reserve_bytes_per_gpu = reserve_mib * 1048576ull;
    opts.output_head_reserve_bytes = output_head_mib * 1048576ull;
    opts.mtp_reserve_bytes = mtp_mib * 1048576ull;

    ds4_v100_cuda_context *ctx = NULL;
    if (ds4_v100_cuda_context_open(&ctx, &opts, err, sizeof(err))) {
        fprintf(stderr, "cuda_v100_context_smoke: %s\n", err);
        return 1;
    }
    if (kv_ctx) {
        ds4_v100_cuda_layer_kv_view view;
        if (ds4_v100_cuda_context_layer_kv_view(ctx, 2, &view, err, sizeof(err))) {
            fprintf(stderr, "cuda_v100_context_smoke: %s\n", err);
            ds4_v100_cuda_context_close(ctx);
            return 1;
        }
        printf("kv_view\tlayer\t%d\tstage\t%d\tgpu\t%d\tarena_bytes\t%llu\tview_total\t%llu\traw_offset\t%llu\tcomp_offset\t%llu\tstate_offset\t%llu\n",
               view.layer_id,
               view.stage_id,
               view.gpu,
               (unsigned long long)view.kv_arena_bytes,
               (unsigned long long)view.view.total_bytes,
               (unsigned long long)view.view.raw_swa_offset,
               (unsigned long long)view.view.compressed_attn_offset,
               (unsigned long long)view.view.attn_state_kv_offset);
        float attn_row[DS4_V100_HEAD_DIM];
        float indexer_row[DS4_V100_INDEXER_HEAD_DIM];
        for (uint32_t i = 0; i < DS4_V100_HEAD_DIM; i++) {
            attn_row[i] = (float)((int)(i % 37) - 18) * 0.03125f;
            if (i < DS4_V100_INDEXER_HEAD_DIM) {
                indexer_row[i] = attn_row[i] * 0.5f + 0.125f;
            }
        }

        ds4_v100_cuda_prefill_kv_update update = {
            .slot = 0,
            .raw_row = 9,
            .comp_row = 5,
            .attn_row_f32 = attn_row,
            .indexer_row_f32 = indexer_row,
        };
        if (ds4_v100_cuda_context_prefill_kv_update_f16(ctx, 2, &update,
                                                        err, sizeof(err))) {
            fprintf(stderr, "cuda_v100_context_smoke: %s\n", err);
            ds4_v100_cuda_context_close(ctx);
            return 1;
        }
        const uint64_t ratio4_comp_rows =
            view.view.compressed_attn_bytes /
            ((uint64_t)kv_slots * DS4_V100_HEAD_DIM * sizeof(uint16_t));
        uint16_t half_row[DS4_V100_HEAD_DIM];
        if (read_f16_row(ctx, &view, view.view.raw_swa_offset, update.raw_row,
                         DS4_V100_HEAD_DIM, half_row, err, sizeof(err)) ||
            expect_f16_row("stage ratio4 raw", half_row, attn_row,
                           DS4_V100_HEAD_DIM) ||
            read_f16_row(ctx, &view, view.view.compressed_attn_offset,
                         update.comp_row, DS4_V100_HEAD_DIM, half_row,
                         err, sizeof(err)) ||
            expect_f16_row("stage ratio4 comp", half_row, attn_row,
                           DS4_V100_HEAD_DIM)) {
            if (err[0]) fprintf(stderr, "cuda_v100_context_smoke: %s\n", err);
            ds4_v100_cuda_context_close(ctx);
            return 1;
        }
        (void)ratio4_comp_rows;

        uint16_t index_half[DS4_V100_INDEXER_HEAD_DIM];
        if (read_f16_row(ctx, &view, view.view.indexer_kv_offset,
                         update.comp_row, DS4_V100_INDEXER_HEAD_DIM,
                         index_half, err, sizeof(err)) ||
            expect_f16_row("stage ratio4 indexer", index_half, indexer_row,
                           DS4_V100_INDEXER_HEAD_DIM)) {
            if (err[0]) fprintf(stderr, "cuda_v100_context_smoke: %s\n", err);
            ds4_v100_cuda_context_close(ctx);
            return 1;
        }

        const uint64_t state_values =
            view.view.attn_state_kv_bytes / sizeof(float);
        float *state = (float *)malloc((size_t)view.view.attn_state_kv_bytes);
        if (!state) {
            fprintf(stderr, "cuda_v100_context_smoke: state alloc failed\n");
            ds4_v100_cuda_context_close(ctx);
            return 1;
        }
        if (ds4_v100_cuda_context_read_kv_arena(ctx, view.stage_id,
                                                view.view.attn_state_kv_offset,
                                                state,
                                                view.view.attn_state_kv_bytes,
                                                err, sizeof(err)) ||
            expect_state_sample("stage ratio4 attn state kv", state, state_values,
                                attn_row, DS4_V100_HEAD_DIM, 0.125f, 0.001f, 4)) {
            if (err[0]) fprintf(stderr, "cuda_v100_context_smoke: %s\n", err);
            free(state);
            ds4_v100_cuda_context_close(ctx);
            return 1;
        }
        free(state);

        ds4_v100_cuda_layer_kv_view ratio128_view;
        if (ds4_v100_cuda_context_layer_kv_view(ctx, 3, &ratio128_view,
                                                err, sizeof(err))) {
            fprintf(stderr, "cuda_v100_context_smoke: %s\n", err);
            ds4_v100_cuda_context_close(ctx);
            return 1;
        }
        update.raw_row = 11;
        update.comp_row = 2;
        update.indexer_row_f32 = NULL;
        if (ds4_v100_cuda_context_prefill_kv_update_f16(ctx, 3, &update,
                                                        err, sizeof(err))) {
            fprintf(stderr, "cuda_v100_context_smoke: %s\n", err);
            ds4_v100_cuda_context_close(ctx);
            return 1;
        }
        if (read_f16_row(ctx, &ratio128_view, ratio128_view.view.raw_swa_offset,
                         update.raw_row, DS4_V100_HEAD_DIM, half_row,
                         err, sizeof(err)) ||
            expect_f16_row("stage ratio128 raw", half_row, attn_row,
                           DS4_V100_HEAD_DIM) ||
            read_f16_row(ctx, &ratio128_view,
                         ratio128_view.view.compressed_attn_offset,
                         update.comp_row, DS4_V100_HEAD_DIM, half_row,
                         err, sizeof(err)) ||
            expect_f16_row("stage ratio128 comp", half_row, attn_row,
                           DS4_V100_HEAD_DIM)) {
            if (err[0]) fprintf(stderr, "cuda_v100_context_smoke: %s\n", err);
            ds4_v100_cuda_context_close(ctx);
            return 1;
        }

        update.indexer_row_f32 = NULL;
        if (!ds4_v100_cuda_context_prefill_kv_update_f16(ctx, 2, &update,
                                                         err, sizeof(err))) {
            fprintf(stderr, "cuda_v100_context_smoke: accepted missing ratio4 indexer row\n");
            ds4_v100_cuda_context_close(ctx);
            return 1;
        }
        printf("kv_update\tlayers\t2,3\tstatus\tok\n");
    }
    printf("cuda_v100_context_smoke: devices=%d stages=%d production=%d\n", n, stages, production);
    for (int i = 0; i < n; i++) {
        printf("device\t%d\tcc\t%d.%d\tmem\t%llu\tpci\t%s\n",
               i, facts[i].cc_major, facts[i].cc_minor,
               (unsigned long long)facts[i].total_global_mem,
               facts[i].pci_bus_id);
    }
    printf("p2p_from\\to");
    for (int j = 0; j < n; j++) printf("\t%d", j);
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("%d", i);
        for (int j = 0; j < n; j++) printf("\t%d", facts[i].peer_access[j] ? 1 : 0);
        printf("\n");
    }
    ds4_v100_cuda_context_close(ctx);
    puts("cuda_v100_context_smoke: ok");
    return 0;
}
