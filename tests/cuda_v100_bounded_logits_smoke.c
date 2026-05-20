#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_v100_bounded_logits_smoke: %s\n", msg);
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

static void topk3(const float *x, uint32_t n, uint32_t out[3]) {
    out[0] = out[1] = out[2] = UINT32_MAX;
    for (uint32_t i = 0; i < n; i++) {
        if (out[0] == UINT32_MAX || x[i] > x[out[0]]) {
            out[2] = out[1];
            out[1] = out[0];
            out[0] = i;
        } else if (out[1] == UINT32_MAX || x[i] > x[out[1]]) {
            out[2] = out[1];
            out[1] = i;
        } else if (out[2] == UINT32_MAX || x[i] > x[out[2]]) {
            out[2] = i;
        }
    }
}

static void expect_close(float got, float want, float tol, const char *label) {
    if (fabsf(got - want) > tol) {
        fprintf(stderr,
                "cuda_v100_bounded_logits_smoke: %s got %.8g expected %.8g\n",
                label,
                got,
                want);
        failures++;
    }
}

int main(void) {
    enum {
        ROWS = 64,
        COLS = 512,
        STRIDE = 520,
        OFFSET = 8,
        TOPK = 3,
    };

    int devices = ds4_gpu_device_count();
    if (devices < 1) {
        fprintf(stderr, "cuda_v100_bounded_logits_smoke: no CUDA devices visible\n");
        return 1;
    }

    float hidden[COLS];
    for (uint32_t c = 0; c < COLS; c++) {
        hidden[c] = (float)((int)(c % 29u) - 14) * 0.01171875f;
    }

    const uint64_t matrix_bytes = (uint64_t)ROWS * STRIDE * sizeof(uint16_t);
    const uint64_t payload_bytes = OFFSET + matrix_bytes;
    unsigned char *payload = (unsigned char *)malloc((size_t)payload_bytes);
    float *ref = (float *)calloc(ROWS, sizeof(float));
    float *got = (float *)calloc(ROWS, sizeof(float));
    if (!payload || !ref || !got) {
        fprintf(stderr, "cuda_v100_bounded_logits_smoke: out of memory\n");
        free(got);
        free(ref);
        free(payload);
        return 1;
    }
    memset(payload, 0x7b, (size_t)payload_bytes);
    uint16_t *matrix = (uint16_t *)(void *)(payload + OFFSET);

    for (uint32_t r = 0; r < ROWS; r++) {
        float coeff = 0.0f;
        if (r == 13) coeff = 3.0f;
        else if (r == 29) coeff = 2.0f;
        else if (r == 41) coeff = 1.25f;
        else coeff = (float)((int)(r % 7u) - 3) * 0.004f;
        for (uint32_t c = 0; c < STRIDE; c++) {
            float w = 0.0f;
            if (c < COLS) {
                const float noise =
                    (float)((int)((r * 17u + c * 5u) % 23u) - 11) * 0.000244140625f;
                w = coeff * hidden[c] + noise;
            }
            matrix[(uint64_t)r * STRIDE + c] = f32_to_bf16_trunc(w);
        }
    }

    for (uint32_t r = 0; r < ROWS; r++) {
        float acc = 0.0f;
        for (uint32_t c = 0; c < COLS; c++) {
            acc += bf16_to_f32(matrix[(uint64_t)r * STRIDE + c]) * hidden[c];
        }
        ref[r] = acc;
    }

    ds4_gpu_arena *arena = NULL;
    check(ds4_gpu_arena_open(&arena, 0, payload_bytes) == 0, "arena open");
    if (!arena) {
        free(got);
        free(ref);
        free(payload);
        return 1;
    }
    check(ds4_gpu_arena_is_device_memory(arena), "arena is device memory");
    check(ds4_gpu_arena_upload(arena, 0, payload, payload_bytes) == 0,
          "arena upload");

    ds4_gpu_bf16_matrix_view view = {
        .arena_offset = OFFSET,
        .byte_length = matrix_bytes,
        .rows = ROWS,
        .cols = COLS,
        .row_stride_elements = STRIDE,
    };
    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc(sizeof(hidden));
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(ROWS * sizeof(float));
    check(x_t && out_t, "logit tensors allocate");
    if (x_t && out_t) {
        check(ds4_gpu_tensor_write(x_t, 0, hidden, sizeof(hidden)),
              "hidden upload");
        check(ds4_gpu_arena_bf16_matmul_f32(arena, &view, x_t, out_t) == 0,
              "bf16 output-head matmul");
        check(ds4_gpu_tensor_read(out_t, 0, got, ROWS * sizeof(float)),
              "logits read");
        for (uint32_t r = 0; r < ROWS; r++) {
            expect_close(got[r], ref[r], 5e-4f, "bounded logit");
        }

        uint32_t ref_top[TOPK];
        uint32_t got_top[TOPK];
        topk3(ref, ROWS, ref_top);
        topk3(got, ROWS, got_top);
        for (uint32_t i = 0; i < TOPK; i++) {
            if (got_top[i] != ref_top[i]) {
                fprintf(stderr,
                        "cuda_v100_bounded_logits_smoke: top%u got %u expected %u\n",
                        i + 1,
                        got_top[i],
                        ref_top[i]);
                failures++;
            }
        }
        uint32_t device_top1 = UINT32_MAX;
        float device_logit = 0.0f;
        check(ds4_gpu_top1_f32_tensor(out_t, ROWS, &device_top1, &device_logit),
              "device top1");
        if (device_top1 != got_top[0] || fabsf(device_logit - got[got_top[0]]) > 5e-4f) {
            fprintf(stderr,
                    "cuda_v100_bounded_logits_smoke: device top1 got %u %.8g expected gpu %u %.8g\n",
                    device_top1,
                    device_logit,
                    got_top[0],
                    got[got_top[0]]);
            failures++;
        }
    }

    ds4_gpu_bf16_matrix_view bad = view;
    bad.byte_length = matrix_bytes - (uint64_t)(STRIDE - COLS + 1) * sizeof(uint16_t);
    check(ds4_gpu_arena_bf16_matmul_f32(arena, &bad, x_t, out_t) != 0,
          "accepted truncated bf16 logits view");
    bad = view;
    bad.row_stride_elements = COLS - 1;
    check(ds4_gpu_arena_bf16_matmul_f32(arena, &bad, x_t, out_t) != 0,
          "accepted undersized bf16 logits stride");
    bad = view;
    bad.arena_offset = OFFSET + 1;
    check(ds4_gpu_arena_bf16_matmul_f32(arena, &bad, x_t, out_t) != 0,
          "accepted odd bf16 logits offset");

    ds4_gpu_tensor *short_x = ds4_gpu_tensor_alloc((COLS - 1) * sizeof(float));
    ds4_gpu_tensor *short_out = ds4_gpu_tensor_alloc((ROWS - 1) * sizeof(float));
    if (short_x) {
        check(ds4_gpu_arena_bf16_matmul_f32(arena, &view, short_x, out_t) != 0,
              "accepted undersized hidden tensor");
    }
    if (short_out) {
        check(ds4_gpu_arena_bf16_matmul_f32(arena, &view, x_t, short_out) != 0,
              "accepted undersized output tensor");
    }

    ds4_gpu_tensor_free(short_out);
    ds4_gpu_tensor_free(short_x);
    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(x_t);
    ds4_gpu_arena_close(arena);
    free(got);
    free(ref);
    free(payload);

    if (failures) return 1;
    puts("cuda_v100_bounded_logits_smoke: ok");
    return 0;
}
