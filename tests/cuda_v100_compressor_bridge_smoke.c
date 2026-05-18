#include "ds4_gpu.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_v100_compressor_bridge_smoke: %s\n", msg);
        failures++;
    }
}

static void fill_inputs(float *kv,
                        float *sc,
                        uint32_t n_tokens,
                        uint32_t width,
                        float seed) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t j = 0; j < width; j++) {
            const int kv_i = (int)((t * 17u + j * 3u) % 41u) - 20;
            const int sc_i = (int)((t * 11u + j * 5u) % 29u) - 14;
            kv[(uint64_t)t * width + j] = seed + (float)kv_i * 0.015625f;
            sc[(uint64_t)t * width + j] = (float)sc_i * 0.03125f;
        }
    }
}

static void rms_norm(float *row, uint32_t n, float eps) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) ss += (double)row[i] * (double)row[i];
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint32_t i = 0; i < n; i++) row[i] *= scale;
}

static void cpu_ref_comp_row(float *out,
                             const float *kv,
                             const float *sc,
                             uint32_t ratio,
                             uint32_t head_dim,
                             uint32_t comp_row,
                             float rms_eps) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    for (uint32_t d = 0; d < head_dim; d++) {
        float vals[128];
        float scores[128];
        uint32_t n_cand = 0;
        float max_s = -FLT_MAX;

        if (ratio == 4u) {
            if (comp_row > 0) {
                const uint32_t base = (comp_row - 1u) * ratio;
                for (uint32_t r = 0; r < ratio; r++) {
                    const uint32_t t = base + r;
                    vals[n_cand] = kv[(uint64_t)t * width + d];
                    scores[n_cand] = sc[(uint64_t)t * width + d];
                    max_s = fmaxf(max_s, scores[n_cand]);
                    n_cand++;
                }
            }
            const uint32_t base = comp_row * ratio;
            for (uint32_t r = 0; r < ratio; r++) {
                const uint32_t t = base + r;
                vals[n_cand] = kv[(uint64_t)t * width + head_dim + d];
                scores[n_cand] = sc[(uint64_t)t * width + head_dim + d];
                max_s = fmaxf(max_s, scores[n_cand]);
                n_cand++;
            }
        } else {
            const uint32_t base = comp_row * ratio;
            for (uint32_t r = 0; r < ratio; r++) {
                const uint32_t t = base + r;
                vals[n_cand] = kv[(uint64_t)t * width + d];
                scores[n_cand] = sc[(uint64_t)t * width + d];
                max_s = fmaxf(max_s, scores[n_cand]);
                n_cand++;
            }
        }

        float den = 0.0f;
        float acc = 0.0f;
        for (uint32_t i = 0; i < n_cand; i++) {
            const float w = expf(scores[i] - max_s);
            den += w;
            acc += vals[i] * w;
        }
        out[d] = den != 0.0f ? acc / den : 0.0f;
    }
    rms_norm(out, head_dim, rms_eps);
}

static int compare_row(const char *label,
                       const float *got,
                       const float *want,
                       uint32_t n,
                       float tol) {
    for (uint32_t i = 0; i < n; i++) {
        if (fabsf(got[i] - want[i]) > tol) {
            fprintf(stderr,
                    "cuda_v100_compressor_bridge_smoke: %s[%u] got %.8g expected %.8g\n",
                    label,
                    i,
                    got[i],
                    want[i]);
            return 1;
        }
    }
    return 0;
}

static int compare_state_row(const char *label,
                             const float *got_kv,
                             const float *got_score,
                             const float *kv,
                             const float *sc,
                             uint32_t src_row,
                             uint32_t dst_row,
                             uint32_t width,
                             float tol) {
    const float *want_kv = kv + (uint64_t)src_row * width;
    const float *want_sc = sc + (uint64_t)src_row * width;
    const float *got_kv_row = got_kv + (uint64_t)dst_row * width;
    const float *got_sc_row = got_score + (uint64_t)dst_row * width;
    for (uint32_t i = 0; i < width; i++) {
        if (fabsf(got_kv_row[i] - want_kv[i]) > tol ||
            fabsf(got_sc_row[i] - want_sc[i]) > tol) {
            fprintf(stderr,
                    "cuda_v100_compressor_bridge_smoke: %s[%u] got kv %.8g score %.8g expected kv %.8g score %.8g\n",
                    label,
                    i,
                    got_kv_row[i],
                    got_sc_row[i],
                    want_kv[i],
                    want_sc[i]);
            return 1;
        }
    }
    return 0;
}

static void run_case(const char *label,
                     uint32_t ratio,
                     uint32_t head_dim,
                     uint32_t n_tokens,
                     float seed) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const float rms_eps = 1e-6f;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)ratio * width * sizeof(float);
    const uint64_t norm_offset = ape_bytes;
    const uint64_t model_bytes = ape_bytes + (uint64_t)head_dim * sizeof(float);

    float *kv = (float *)malloc((size_t)kv_bytes);
    float *sc = (float *)malloc((size_t)kv_bytes);
    float *model = (float *)calloc(1, (size_t)model_bytes);
    float *got = (float *)malloc((size_t)comp_bytes);
    float *want = (float *)malloc((size_t)head_dim * sizeof(float));
    float *state_kv = (float *)malloc((size_t)state_bytes);
    float *state_score = (float *)malloc((size_t)state_bytes);
    check(kv && sc && model && got && want && state_kv && state_score,
          "host allocations");
    if (!kv || !sc || !model || !got || !want || !state_kv || !state_score) {
        free(kv);
        free(sc);
        free(model);
        free(got);
        free(want);
        free(state_kv);
        free(state_score);
        return;
    }

    fill_inputs(kv, sc, n_tokens, width, seed);
    for (uint32_t i = 0; i < head_dim; i++) {
        model[(norm_offset / sizeof(float)) + i] = 1.0f;
    }
    check(ds4_gpu_set_model_map(model, model_bytes), "model map");

    ds4_gpu_tensor *kv_t = ds4_gpu_tensor_alloc(kv_bytes);
    ds4_gpu_tensor *sc_t = ds4_gpu_tensor_alloc(kv_bytes);
    ds4_gpu_tensor *comp_t = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *state_kv_t = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *state_score_t = ds4_gpu_tensor_alloc(state_bytes);
    check(kv_t && sc_t && comp_t && state_kv_t && state_score_t,
          "device allocations");
    if (kv_t && sc_t && comp_t && state_kv_t && state_score_t) {
        check(ds4_gpu_tensor_write(kv_t, 0, kv, kv_bytes), "kv upload");
        check(ds4_gpu_tensor_write(sc_t, 0, sc, kv_bytes), "score upload");
        check(ds4_gpu_compressor_prefill_tensor(comp_t,
                                                state_kv_t,
                                                state_score_t,
                                                kv_t,
                                                sc_t,
                                                model,
                                                model_bytes,
                                                0,
                                                0,
                                                norm_offset,
                                                0,
                                                head_dim,
                                                ratio,
                                                0,
                                                n_tokens,
                                                0,
                                                0,
                                                false,
                                                1000000.0f,
                                                1.0f,
                                                0.0f,
                                                1.0f,
                                                32.0f,
                                                1.0f,
                                                rms_eps),
              "compressor prefill");
        check(ds4_gpu_tensor_read(comp_t, 0, got, comp_bytes), "comp read");
        check(ds4_gpu_tensor_read(state_kv_t, 0, state_kv, state_bytes),
              "state kv read");
        check(ds4_gpu_tensor_read(state_score_t, 0, state_score, state_bytes),
              "state score read");

        for (uint32_t c = 0; c < n_comp; c++) {
            cpu_ref_comp_row(want, kv, sc, ratio, head_dim, c, rms_eps);
            if (compare_row(label, got + (uint64_t)c * head_dim, want,
                            head_dim, 2e-4f)) {
                failures++;
                break;
            }
        }

        if (ratio == 4u) {
            if (compare_state_row(label, state_kv, state_score, kv, sc,
                                  0, 0, width, 1e-6f) ||
                compare_state_row(label, state_kv, state_score, kv, sc,
                                  n_tokens - 1u, 4u, width, 1e-6f)) {
                failures++;
            }
        } else if ((n_tokens % ratio) != 0u) {
            if (compare_state_row(label, state_kv, state_score, kv, sc,
                                  n_tokens - 1u, 0, width, 1e-6f)) {
                failures++;
            }
        }
    }

    ds4_gpu_tensor_free(state_score_t);
    ds4_gpu_tensor_free(state_kv_t);
    ds4_gpu_tensor_free(comp_t);
    ds4_gpu_tensor_free(sc_t);
    ds4_gpu_tensor_free(kv_t);
    free(state_score);
    free(state_kv);
    free(want);
    free(got);
    free(model);
    free(sc);
    free(kv);
}

int main(void) {
    setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    if (ds4_gpu_device_count() < 1) {
        fprintf(stderr, "cuda_v100_compressor_bridge_smoke: no CUDA devices visible\n");
        return 1;
    }

    run_case("ratio128-attn", 128, 512, 129, 0.25f);
    run_case("ratio4-attn", 4, 512, 5, -0.125f);
    run_case("ratio4-indexer", 4, 128, 5, 0.0625f);

    ds4_gpu_cleanup();
    if (failures) return 1;
    puts("cuda_v100_compressor_bridge_smoke: ok");
    return 0;
}
