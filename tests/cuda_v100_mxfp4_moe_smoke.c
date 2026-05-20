#include "ds4_gpu.h"
#include "ds4_source_formats.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_v100_mxfp4_moe_smoke: %s\n", msg);
        failures++;
    }
}

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

static uint16_t f32_to_bf16_trunc(float f) {
    return (uint16_t)(f32_bits(f) >> 16);
}

static float bf16_to_f32(uint16_t bf16) {
    return f32_from_bits((uint32_t)bf16 << 16);
}

static float softplus_ref(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static float swiglu_ref(float gate, float up, float clamp, float weight) {
    if (clamp > 1.0e-6f) {
        if (gate > clamp) gate = clamp;
        if (up > clamp) up = clamp;
        if (up < -clamp) up = -clamp;
    }
    return (gate / (1.0f + expf(-gate))) * up * weight;
}

static void topk1(const float *x, uint32_t n, uint32_t *out) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++) {
        if (x[i] > x[best]) best = i;
    }
    *out = best;
}

static void expect_close(float got, float want, float tol, const char *label) {
    if (fabsf(got - want) > tol) {
        fprintf(stderr,
                "cuda_v100_mxfp4_moe_smoke: %s got %.8g expected %.8g\n",
                label,
                got,
                want);
        failures++;
    }
}

static void fill_mxfp4_row(uint8_t *row, uint32_t cols, uint32_t seed) {
    static const uint8_t codes[] = {
        0x0, 0x1, 0x2, 0x3, 0x4, 0x9, 0xa, 0xb,
        0xc, 0x5, 0x6, 0xd, 0x7, 0xe, 0xf, 0x8,
    };
    const uint64_t row_bytes = ds4_src_mxfp4_row_bytes(cols);
    memset(row, 0xa5, (size_t)row_bytes);
    for (uint32_t b = 0; b < cols / DS4_SRC_MXFP4_BLOCK_ELEMS; b++) {
        uint8_t *block = row + (uint64_t)b * DS4_SRC_MXFP4_BLOCK_BYTES;
        block[0] = 127u;
        for (uint32_t j = 0; j < DS4_SRC_MXFP4_BLOCK_ELEMS / 2; j++) {
            const uint8_t lo = codes[(seed + b * 3u + j) % 16u];
            const uint8_t hi = codes[(seed + b * 5u + j + 7u) % 16u];
            block[1u + j] = (uint8_t)(lo | (hi << 4));
        }
    }
}

static int cpu_router_ref(const float *logits,
                          int32_t selected[6],
                          float weights[6],
                          float probs[256]) {
    for (uint32_t e = 0; e < 256; e++) {
        probs[e] = sqrtf(softplus_ref(logits[e]));
    }
    for (uint32_t k = 0; k < 6; k++) selected[k] = -1;
    for (uint32_t e = 0; e < 256; e++) {
        const float score = probs[e];
        for (uint32_t k = 0; k < 6; k++) {
            if (selected[k] < 0 || score > probs[(uint32_t)selected[k]]) {
                for (uint32_t j = 5; j > k; j--) selected[j] = selected[j - 1u];
                selected[k] = (int32_t)e;
                break;
            }
        }
    }
    float sum = 0.0f;
    for (uint32_t k = 0; k < 6; k++) {
        weights[k] = probs[(uint32_t)selected[k]];
        sum += weights[k];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t k = 0; k < 6; k++) weights[k] = weights[k] / sum * 1.5f;
    return 0;
}

static void cpu_moe_ref(const unsigned char *payload,
                        uint64_t gate_offset,
                        uint64_t up_offset,
                        uint64_t down_offset,
                        uint64_t gate_expert_bytes,
                        uint64_t down_expert_bytes,
                        uint64_t hidden_row_bytes,
                        uint64_t mid_row_bytes,
                        const float *hidden,
                        const int32_t selected[6],
                        const float weights[6],
                        uint32_t hidden_dim,
                        uint32_t mid_dim,
                        float *out) {
    float gate[64];
    float up[64];
    float mid[64];
    char err[128];
    memset(out, 0, (size_t)hidden_dim * sizeof(float));
    for (uint32_t route = 0; route < 6; route++) {
        const uint32_t expert = (uint32_t)selected[route];
        const uint8_t *gate_base = payload + gate_offset +
            (uint64_t)expert * gate_expert_bytes;
        const uint8_t *up_base = payload + up_offset +
            (uint64_t)expert * gate_expert_bytes;
        const uint8_t *down_base = payload + down_offset +
            (uint64_t)expert * down_expert_bytes;
        for (uint32_t m = 0; m < mid_dim; m++) {
            err[0] = '\0';
            check(ds4_src_mxfp4_row_dot(&gate[m],
                                        gate_base + (uint64_t)m * hidden_row_bytes,
                                        hidden,
                                        hidden_dim,
                                        err,
                                        sizeof(err)) == 0,
                  "cpu gate mxfp4 dot");
            check(ds4_src_mxfp4_row_dot(&up[m],
                                        up_base + (uint64_t)m * hidden_row_bytes,
                                        hidden,
                                        hidden_dim,
                                        err,
                                        sizeof(err)) == 0,
                  "cpu up mxfp4 dot");
            mid[m] = swiglu_ref(gate[m], up[m], 10.0f, weights[route]);
        }
        for (uint32_t h = 0; h < hidden_dim; h++) {
            float v = 0.0f;
            err[0] = '\0';
            check(ds4_src_mxfp4_row_dot(&v,
                                        down_base + (uint64_t)h * mid_row_bytes,
                                        mid,
                                        mid_dim,
                                        err,
                                        sizeof(err)) == 0,
                  "cpu down mxfp4 dot");
            out[h] += v;
        }
    }
}

int main(void) {
    enum {
        EXPERTS = 256,
        ROUTES = 6,
        HIDDEN = 128,
        MID = 64,
        VOCAB = 32,
    };
    const uint32_t strong[ROUTES] = {13, 29, 41, 67, 101, 149};

    int devices = ds4_gpu_device_count();
    if (devices < 1) {
        fprintf(stderr, "cuda_v100_mxfp4_moe_smoke: no CUDA devices visible\n");
        return 1;
    }

    const uint64_t hidden_row_bytes = ds4_src_mxfp4_row_bytes(HIDDEN);
    const uint64_t mid_row_bytes = ds4_src_mxfp4_row_bytes(MID);
    const uint64_t gate_expert_bytes = (uint64_t)MID * hidden_row_bytes;
    const uint64_t down_expert_bytes = (uint64_t)HIDDEN * mid_row_bytes;
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = gate_offset + (uint64_t)EXPERTS * gate_expert_bytes;
    const uint64_t down_offset = up_offset + (uint64_t)EXPERTS * gate_expert_bytes;
    const uint64_t head_offset = down_offset + (uint64_t)EXPERTS * down_expert_bytes;
    const uint64_t head_bytes = (uint64_t)VOCAB * HIDDEN * sizeof(uint16_t);
    const uint64_t payload_bytes = head_offset + head_bytes;

    unsigned char *payload = (unsigned char *)malloc((size_t)payload_bytes);
    float hidden[HIDDEN];
    float router_logits[256];
    float cpu_probs[256];
    int32_t cpu_selected[ROUTES];
    float cpu_weights[ROUTES];
    float cpu_hidden[HIDDEN];
    float gpu_hidden[HIDDEN];
    float mid_ref[MID];
    float mid_fused[MID];
    float next_ref[HIDDEN];
    float next_fused[HIDDEN];
    float batch_hidden[2 * HIDDEN];
    float batch_out[2 * HIDDEN];
    float cpu_logits[VOCAB];
    float gpu_logits[VOCAB];
    if (!payload) {
        fprintf(stderr, "cuda_v100_mxfp4_moe_smoke: out of memory\n");
        return 1;
    }
    memset(payload, 0, (size_t)payload_bytes);

    for (uint32_t h = 0; h < HIDDEN; h++) {
        hidden[h] = (float)((int)(h % 31u) - 15) * 0.00390625f;
    }
    for (uint32_t e = 0; e < 256; e++) router_logits[e] = -5.0f;
    for (uint32_t k = 0; k < ROUTES; k++) {
        router_logits[strong[k]] = 6.0f - (float)k * 0.25f;
    }
    (void)cpu_router_ref(router_logits, cpu_selected, cpu_weights, cpu_probs);

    for (uint32_t e = 0; e < EXPERTS; e++) {
        uint8_t *gate_base = payload + gate_offset + (uint64_t)e * gate_expert_bytes;
        uint8_t *up_base = payload + up_offset + (uint64_t)e * gate_expert_bytes;
        uint8_t *down_base = payload + down_offset + (uint64_t)e * down_expert_bytes;
        for (uint32_t m = 0; m < MID; m++) {
            fill_mxfp4_row(gate_base + (uint64_t)m * hidden_row_bytes,
                           HIDDEN,
                           e + m * 3u);
            fill_mxfp4_row(up_base + (uint64_t)m * hidden_row_bytes,
                           HIDDEN,
                           e * 5u + m);
        }
        for (uint32_t h = 0; h < HIDDEN; h++) {
            fill_mxfp4_row(down_base + (uint64_t)h * mid_row_bytes,
                           MID,
                           e * 7u + h);
        }
    }

    uint16_t *head = (uint16_t *)(void *)(payload + head_offset);
    for (uint32_t v = 0; v < VOCAB; v++) {
        const float coeff = v == 7 ? 0.003f : (v == 19 ? 0.002f : 0.0001f * (float)((int)(v % 5u) - 2));
        for (uint32_t h = 0; h < HIDDEN; h++) {
            float w = coeff * (float)((int)(h % 11u) - 5) +
                      (float)((int)((v * 13u + h * 3u) % 17u) - 8) * 0.00002f;
            head[(uint64_t)v * HIDDEN + h] = f32_to_bf16_trunc(w);
        }
    }

    cpu_moe_ref(payload,
                gate_offset,
                up_offset,
                down_offset,
                gate_expert_bytes,
                down_expert_bytes,
                hidden_row_bytes,
                mid_row_bytes,
                hidden,
                cpu_selected,
                cpu_weights,
                HIDDEN,
                MID,
                cpu_hidden);
    for (uint32_t v = 0; v < VOCAB; v++) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < HIDDEN; h++) {
            acc += bf16_to_f32(head[(uint64_t)v * HIDDEN + h]) * cpu_hidden[h];
        }
        cpu_logits[v] = acc;
    }

    ds4_gpu_arena *arena = NULL;
    check(ds4_gpu_arena_open(&arena, 0, payload_bytes) == 0, "arena open");
    if (!arena) {
        free(payload);
        return 1;
    }
    check(ds4_gpu_arena_is_device_memory(arena), "arena is device memory");
    check(ds4_gpu_arena_upload(arena, 0, payload, payload_bytes) == 0, "arena upload");

    ds4_gpu_tensor *hidden_t = ds4_gpu_tensor_alloc(HIDDEN * sizeof(float));
    ds4_gpu_tensor *hidden_alt_t = ds4_gpu_tensor_alloc(HIDDEN * sizeof(float));
    ds4_gpu_tensor *gate_t = ds4_gpu_tensor_alloc(MID * sizeof(float));
    ds4_gpu_tensor *up_t = ds4_gpu_tensor_alloc(MID * sizeof(float));
    ds4_gpu_tensor *mid_t = ds4_gpu_tensor_alloc(MID * sizeof(float));
    ds4_gpu_tensor *fused_mid_t = ds4_gpu_tensor_alloc(MID * sizeof(float));
    ds4_gpu_tensor *group_mid_t = ds4_gpu_tensor_alloc((uint64_t)ROUTES * MID * sizeof(float));
    ds4_gpu_tensor *group_out_t = ds4_gpu_tensor_alloc(HIDDEN * sizeof(float));
    ds4_gpu_tensor *batch_hidden_t = ds4_gpu_tensor_alloc(2ull * HIDDEN * sizeof(float));
    ds4_gpu_tensor *batch_selected_t = ds4_gpu_tensor_alloc(2ull * ROUTES * sizeof(int32_t));
    ds4_gpu_tensor *batch_weights_t = ds4_gpu_tensor_alloc(2ull * ROUTES * sizeof(float));
    ds4_gpu_tensor *batch_ptrs_t = ds4_gpu_tensor_alloc(2ull * sizeof(void *));
    ds4_gpu_tensor *batch_mid_t = ds4_gpu_tensor_alloc(2ull * ROUTES * MID * sizeof(float));
    ds4_gpu_tensor *batch_out_t = ds4_gpu_tensor_alloc(2ull * HIDDEN * sizeof(float));
    ds4_gpu_tensor *route_t = ds4_gpu_tensor_alloc(HIDDEN * sizeof(float));
    ds4_gpu_tensor *fused_next_t = ds4_gpu_tensor_alloc(HIDDEN * sizeof(float));
    ds4_gpu_tensor *accum_a = ds4_gpu_tensor_alloc(HIDDEN * sizeof(float));
    ds4_gpu_tensor *accum_b = ds4_gpu_tensor_alloc(HIDDEN * sizeof(float));
    ds4_gpu_tensor *router_t = ds4_gpu_tensor_alloc(256 * sizeof(float));
    ds4_gpu_tensor *probs_t = ds4_gpu_tensor_alloc(256 * sizeof(float));
    ds4_gpu_tensor *selected_t = ds4_gpu_tensor_alloc(ROUTES * sizeof(int32_t));
    ds4_gpu_tensor *weights_t = ds4_gpu_tensor_alloc(ROUTES * sizeof(float));
    ds4_gpu_tensor *logits_t = ds4_gpu_tensor_alloc(VOCAB * sizeof(float));
    check(hidden_t && hidden_alt_t &&
              gate_t && up_t && mid_t && fused_mid_t && group_mid_t && group_out_t &&
              batch_hidden_t && batch_selected_t && batch_weights_t && batch_ptrs_t &&
              batch_mid_t && batch_out_t &&
              route_t && fused_next_t &&
              accum_a && accum_b &&
              router_t && probs_t && selected_t && weights_t && logits_t,
          "tensor allocate");

    if (hidden_t && hidden_alt_t &&
        gate_t && up_t && mid_t && fused_mid_t && group_mid_t && group_out_t &&
        batch_hidden_t && batch_selected_t && batch_weights_t && batch_ptrs_t &&
        batch_mid_t && batch_out_t &&
        route_t && fused_next_t &&
        accum_a && accum_b &&
        router_t && probs_t && selected_t && weights_t && logits_t) {
        int dummy_model = 0;
        int32_t gpu_selected[ROUTES];
        float gpu_weights[ROUTES];
        check(ds4_gpu_tensor_write(hidden_t, 0, hidden, sizeof(hidden)), "hidden upload");
        check(ds4_gpu_tensor_write(hidden_alt_t, 0, hidden, sizeof(hidden)), "hidden alt upload");
        check(ds4_gpu_tensor_write(router_t, 0, router_logits, sizeof(router_logits)),
              "router upload");
        check(ds4_gpu_router_select_tensor(selected_t,
                                           weights_t,
                                           probs_t,
                                           &dummy_model,
                                           sizeof(dummy_model),
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           router_t),
              "router select");
        check(ds4_gpu_tensor_read(selected_t, 0, gpu_selected, sizeof(gpu_selected)),
              "selected read");
        check(ds4_gpu_tensor_read(weights_t, 0, gpu_weights, sizeof(gpu_weights)),
              "weights read");
        for (uint32_t k = 0; k < ROUTES; k++) {
            if (gpu_selected[k] != cpu_selected[k]) {
                fprintf(stderr,
                        "cuda_v100_mxfp4_moe_smoke: route %u got expert %d expected %d\n",
                        k,
                        gpu_selected[k],
                        cpu_selected[k]);
                failures++;
            }
            expect_close(gpu_weights[k], cpu_weights[k], 2e-5f, "router weight");
        }

        check(ds4_gpu_tensor_fill_f32(accum_a, 0.0f, HIDDEN), "accum zero");
        ds4_gpu_tensor *accum = accum_a;
        ds4_gpu_tensor *next = accum_b;
        for (uint32_t k = 0; k < ROUTES; k++) {
            const uint32_t expert = (uint32_t)gpu_selected[k];
            ds4_gpu_source_row_view gate_view = {
                .arena_offset = gate_offset + (uint64_t)expert * gate_expert_bytes,
                .byte_length = gate_expert_bytes,
                .rows = MID,
                .cols = HIDDEN,
                .row_stride_bytes = (uint32_t)hidden_row_bytes,
            };
            ds4_gpu_source_row_view up_view = {
                .arena_offset = up_offset + (uint64_t)expert * gate_expert_bytes,
                .byte_length = gate_expert_bytes,
                .rows = MID,
                .cols = HIDDEN,
                .row_stride_bytes = (uint32_t)hidden_row_bytes,
            };
            ds4_gpu_source_row_view down_view = {
                .arena_offset = down_offset + (uint64_t)expert * down_expert_bytes,
                .byte_length = down_expert_bytes,
                .rows = HIDDEN,
                .cols = MID,
                .row_stride_bytes = (uint32_t)mid_row_bytes,
            };
            check(ds4_gpu_arena_mxfp4_matmul_f32(arena, &gate_view, hidden_t, gate_t) == 0,
                  "gate mxfp4 matmul");
            check(ds4_gpu_arena_mxfp4_matmul_f32(arena, &up_view, hidden_t, up_t) == 0,
                  "up mxfp4 matmul");
            check(ds4_gpu_swiglu_tensor(mid_t, gate_t, up_t, MID, 10.0f, gpu_weights[k]),
                  "swiglu");
            check(ds4_gpu_arena_mxfp4_pair_swiglu_f32(arena,
                                                       &gate_view,
                                                       &up_view,
                                                       hidden_t,
                                                       fused_mid_t,
                                                       10.0f,
                                                       gpu_weights[k]) == 0,
                  "fused gate/up swiglu");
            check(ds4_gpu_tensor_read(mid_t, 0, mid_ref, sizeof(mid_ref)),
                  "mid ref read");
            check(ds4_gpu_tensor_read(fused_mid_t, 0, mid_fused, sizeof(mid_fused)),
                  "mid fused read");
            for (uint32_t m = 0; m < MID; m++) {
                expect_close(mid_fused[m], mid_ref[m], 1e-4f, "fused mid");
            }
            check(ds4_gpu_arena_mxfp4_matmul_f32(arena, &down_view, fused_mid_t, route_t) == 0,
                  "down mxfp4 matmul");
            check(ds4_gpu_add_tensor(next, accum, route_t, HIDDEN), "route accumulate");
            check(ds4_gpu_arena_mxfp4_matmul_add_f32(arena,
                                                      &down_view,
                                                      fused_mid_t,
                                                      accum,
                                                      fused_next_t) == 0,
                  "fused down accumulate");
            check(ds4_gpu_tensor_read(next, 0, next_ref, sizeof(next_ref)),
                  "next ref read");
            check(ds4_gpu_tensor_read(fused_next_t, 0, next_fused, sizeof(next_fused)),
                  "next fused read");
            for (uint32_t h = 0; h < HIDDEN; h++) {
                expect_close(next_fused[h], next_ref[h], 1e-4f, "fused next");
            }
            ds4_gpu_tensor *tmp = accum;
            accum = next;
            next = tmp;
        }

        check(ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_f32(
                  arena,
                  gate_offset,
                  gate_expert_bytes * EXPERTS,
                  up_offset,
                  gate_expert_bytes * EXPERTS,
                  down_offset,
                  down_expert_bytes * EXPERTS,
                  gate_expert_bytes,
                  (uint32_t)hidden_row_bytes,
                  down_expert_bytes,
                  (uint32_t)mid_row_bytes,
                  HIDDEN,
                  MID,
                  EXPERTS,
                  selected_t,
                  weights_t,
                  ROUTES,
                  hidden_t,
                  group_mid_t,
                  group_out_t) == 0,
              "grouped routed mxfp4");
        check(ds4_gpu_tensor_read(group_out_t, 0, next_fused, sizeof(next_fused)),
              "grouped routed read");
        check(ds4_gpu_tensor_read(accum, 0, next_ref, sizeof(next_ref)),
              "per-route routed read");
        for (uint32_t h = 0; h < HIDDEN; h++) {
            expect_close(next_fused[h], next_ref[h], 1e-4f, "grouped routed");
        }
        for (uint32_t h = 0; h < HIDDEN; h++) {
            batch_hidden[h] = hidden[h];
            batch_hidden[HIDDEN + h] = hidden[h];
        }
        int32_t batch_selected[2 * ROUTES];
        float batch_weights[2 * ROUTES];
        for (uint32_t k = 0; k < ROUTES; k++) {
            batch_selected[k] = gpu_selected[k];
            batch_selected[ROUTES + k] = gpu_selected[k];
            batch_weights[k] = gpu_weights[k];
            batch_weights[ROUTES + k] = gpu_weights[k];
        }
        check(ds4_gpu_tensor_write(batch_hidden_t, 0, batch_hidden, sizeof(batch_hidden)),
              "batch hidden upload");
        check(ds4_gpu_tensor_write(batch_selected_t, 0, batch_selected, sizeof(batch_selected)),
              "batch selected upload");
        check(ds4_gpu_tensor_write(batch_weights_t, 0, batch_weights, sizeof(batch_weights)),
              "batch weights upload");
        check(ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_batch_f32(
                  arena,
                  gate_offset,
                  gate_expert_bytes * EXPERTS,
                  up_offset,
                  gate_expert_bytes * EXPERTS,
                  down_offset,
                  down_expert_bytes * EXPERTS,
                  gate_expert_bytes,
                  (uint32_t)hidden_row_bytes,
                  down_expert_bytes,
                  (uint32_t)mid_row_bytes,
                  HIDDEN,
                  MID,
                  EXPERTS,
                  batch_selected_t,
                  batch_weights_t,
                  ROUTES,
                  batch_hidden_t,
                  2,
                  batch_mid_t,
                  batch_out_t) == 0,
              "batched grouped routed mxfp4");
        check(ds4_gpu_tensor_read(batch_out_t, 0, batch_out, sizeof(batch_out)),
              "batched grouped routed read");
        for (uint32_t h = 0; h < HIDDEN; h++) {
            expect_close(batch_out[h], next_ref[h], 1e-4f, "batched grouped routed slot0");
            expect_close(batch_out[HIDDEN + h], next_ref[h], 1e-4f, "batched grouped routed slot1");
        }
        const ds4_gpu_tensor *batch_inputs[2] = {hidden_t, hidden_alt_t};
        check(ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_batch_ptrs_f32(
                  arena,
                  gate_offset,
                  gate_expert_bytes * EXPERTS,
                  up_offset,
                  gate_expert_bytes * EXPERTS,
                  down_offset,
                  down_expert_bytes * EXPERTS,
                  gate_expert_bytes,
                  (uint32_t)hidden_row_bytes,
                  down_expert_bytes,
                  (uint32_t)mid_row_bytes,
                  HIDDEN,
                  MID,
                  EXPERTS,
                  batch_selected_t,
                  batch_weights_t,
                  ROUTES,
                  batch_ptrs_t,
                  batch_inputs,
                  2,
                  batch_mid_t,
                  batch_out_t) == 0,
              "batched grouped routed ptrs mxfp4");
        check(ds4_gpu_tensor_read(batch_out_t, 0, batch_out, sizeof(batch_out)),
              "batched grouped routed ptrs read");
        for (uint32_t h = 0; h < HIDDEN; h++) {
            expect_close(batch_out[h], next_ref[h], 1e-4f, "batched grouped routed ptrs slot0");
            expect_close(batch_out[HIDDEN + h], next_ref[h], 1e-4f, "batched grouped routed ptrs slot1");
        }

        setenv("DS4_CUDA_MXFP4_ROUTE_ROWS2", "1", 1);
        check(ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_batch_ptrs_f32(
                  arena,
                  gate_offset,
                  gate_expert_bytes * EXPERTS,
                  up_offset,
                  gate_expert_bytes * EXPERTS,
                  down_offset,
                  down_expert_bytes * EXPERTS,
                  gate_expert_bytes,
                  (uint32_t)hidden_row_bytes,
                  down_expert_bytes,
                  (uint32_t)mid_row_bytes,
                  HIDDEN,
                  MID,
                  EXPERTS,
                  batch_selected_t,
                  batch_weights_t,
                  ROUTES,
                  batch_ptrs_t,
                  batch_inputs,
                  2,
                  batch_mid_t,
                  batch_out_t) == 0,
              "batched grouped routed ptrs rows2 mxfp4");
        unsetenv("DS4_CUDA_MXFP4_ROUTE_ROWS2");
        check(ds4_gpu_tensor_read(batch_out_t, 0, batch_out, sizeof(batch_out)),
              "batched grouped routed ptrs rows2 read");
        for (uint32_t h = 0; h < HIDDEN; h++) {
            expect_close(batch_out[h],
                         next_ref[h],
                         1e-4f,
                         "batched grouped routed ptrs rows2 slot0");
            expect_close(batch_out[HIDDEN + h],
                         next_ref[h],
                         1e-4f,
                         "batched grouped routed ptrs rows2 slot1");
        }

        ds4_gpu_bf16_matrix_view head_view = {
            .arena_offset = head_offset,
            .byte_length = head_bytes,
            .rows = VOCAB,
            .cols = HIDDEN,
            .row_stride_elements = HIDDEN,
        };
        check(ds4_gpu_arena_bf16_matmul_f32(arena, &head_view, accum, logits_t) == 0,
              "bf16 output-head matmul");
        check(ds4_gpu_tensor_read(accum, 0, gpu_hidden, sizeof(gpu_hidden)),
              "moe hidden read");
        check(ds4_gpu_tensor_read(logits_t, 0, gpu_logits, sizeof(gpu_logits)),
              "logits read");
        for (uint32_t h = 0; h < HIDDEN; h++) {
            expect_close(gpu_hidden[h], cpu_hidden[h], 2e-2f, "moe hidden");
        }
        for (uint32_t v = 0; v < VOCAB; v++) {
            expect_close(gpu_logits[v], cpu_logits[v], 1e-2f, "selected-token logit");
        }
        uint32_t cpu_tok = 0;
        uint32_t gpu_tok = 0;
        topk1(cpu_logits, VOCAB, &cpu_tok);
        topk1(gpu_logits, VOCAB, &gpu_tok);
        if (gpu_tok != cpu_tok) {
            fprintf(stderr,
                    "cuda_v100_mxfp4_moe_smoke: selected token got %u expected %u\n",
                    gpu_tok,
                    cpu_tok);
            failures++;
        }

        ds4_gpu_source_row_view bad = {
            .arena_offset = gate_offset,
            .byte_length = gate_expert_bytes,
            .rows = MID,
            .cols = HIDDEN + 1,
            .row_stride_bytes = (uint32_t)hidden_row_bytes,
        };
        check(ds4_gpu_arena_mxfp4_matmul_f32(arena, &bad, hidden_t, gate_t) != 0,
              "accepted misaligned mxfp4 cols");
        bad.cols = HIDDEN;
        bad.row_stride_bytes = (uint32_t)(hidden_row_bytes - 1);
        check(ds4_gpu_arena_mxfp4_matmul_f32(arena, &bad, hidden_t, gate_t) != 0,
              "accepted undersized mxfp4 stride");
        bad.row_stride_bytes = (uint32_t)hidden_row_bytes;
        bad.byte_length = gate_expert_bytes - 1;
        check(ds4_gpu_arena_mxfp4_matmul_f32(arena, &bad, hidden_t, gate_t) != 0,
              "accepted truncated mxfp4 view");
    }

    ds4_gpu_tensor_free(logits_t);
    ds4_gpu_tensor_free(weights_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(probs_t);
    ds4_gpu_tensor_free(router_t);
    ds4_gpu_tensor_free(accum_b);
    ds4_gpu_tensor_free(accum_a);
    ds4_gpu_tensor_free(fused_next_t);
    ds4_gpu_tensor_free(route_t);
    ds4_gpu_tensor_free(batch_out_t);
    ds4_gpu_tensor_free(batch_mid_t);
    ds4_gpu_tensor_free(batch_ptrs_t);
    ds4_gpu_tensor_free(batch_weights_t);
    ds4_gpu_tensor_free(batch_selected_t);
    ds4_gpu_tensor_free(batch_hidden_t);
    ds4_gpu_tensor_free(group_out_t);
    ds4_gpu_tensor_free(group_mid_t);
    ds4_gpu_tensor_free(fused_mid_t);
    ds4_gpu_tensor_free(mid_t);
    ds4_gpu_tensor_free(up_t);
    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(hidden_alt_t);
    ds4_gpu_tensor_free(hidden_t);
    ds4_gpu_arena_close(arena);
    free(payload);

    if (failures) return 1;
    puts("cuda_v100_mxfp4_moe_smoke: ok");
    return 0;
}
