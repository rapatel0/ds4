#include "ds4_gpu.h"
#include "ds4_source_formats.h"
#include "ds4_v100_context.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    IN_DIM = 128,
    HEAD_DIM = DS4_V100_HEAD_DIM,
    INDEX_DIM = DS4_V100_INDEXER_HEAD_DIM,
};

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_v100_projection_attention_smoke: %s\n", msg);
        failures++;
    }
}

static float f32_from_bits(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
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

static void fill_x(float *x, uint32_t token, uint32_t ratio) {
    for (uint32_t i = 0; i < IN_DIM; i++) {
        const int v = (int)((i * 7u + token * 13u + ratio) % 31u) - 15;
        x[i] = (float)v * 0.00390625f;
    }
}

static void fill_f8_row(uint8_t *row, uint32_t cols, uint32_t seed) {
    static const uint8_t codes[] = {
        0x00, 0x01, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30,
        0x80, 0x81, 0x88, 0x90, 0x98, 0xa0, 0xa8, 0xb0,
    };
    const uint64_t row_bytes = ds4_src_f8_e4m3_b128_row_bytes(cols);
    memset(row, 0, (size_t)row_bytes);
    for (uint32_t b = 0; b < cols / DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS; b++) {
        uint8_t *block = row + (uint64_t)b * DS4_SRC_F8_E4M3_B128_BLOCK_BYTES;
        block[0] = 124u;
        for (uint32_t i = 0; i < DS4_SRC_F8_E4M3_B128_BLOCK_ELEMS; i++) {
            block[1u + i] = codes[(seed + i * 5u + b * 3u) %
                                  (sizeof(codes) / sizeof(codes[0]))];
        }
    }
}

static void fill_f8_matrix(uint8_t *payload,
                           uint64_t offset,
                           uint32_t rows,
                           uint32_t stride,
                           uint32_t cols,
                           uint32_t seed) {
    for (uint32_t r = 0; r < rows; r++) {
        fill_f8_row(payload + offset + (uint64_t)r * stride,
                    cols,
                    seed + r * 17u);
    }
}

static int host_project(float *out,
                        const uint8_t *payload,
                        const ds4_gpu_source_row_view *view,
                        const float *x) {
    char err[128] = {0};
    for (uint32_t r = 0; r < view->rows; r++) {
        const uint8_t *row = payload + view->arena_offset +
                             (uint64_t)r * view->row_stride_bytes;
        if (ds4_src_f8_e4m3_b128_row_dot(&out[r], row, x, view->cols,
                                         err, sizeof(err)) != 0) {
            fprintf(stderr, "cuda_v100_projection_attention_smoke: %s\n", err);
            return 0;
        }
    }
    return 1;
}

static int device_project_copy(ds4_gpu_arena *arena,
                               const ds4_gpu_source_row_view *view,
                               ds4_gpu_tensor *x_t,
                               ds4_gpu_tensor *tmp_t,
                               ds4_gpu_tensor *dst_t,
                               uint64_t dst_offset,
                               const float *x) {
    const uint64_t bytes = (uint64_t)view->rows * sizeof(float);
    return ds4_gpu_tensor_write(x_t, 0, x, IN_DIM * sizeof(float)) &&
           ds4_gpu_arena_f8_e4m3_b128_matmul_f32(arena, view, x_t, tmp_t) == 0 &&
           (!dst_t || ds4_gpu_tensor_copy(dst_t, dst_offset, tmp_t, 0, bytes));
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

static void cpu_ref_attention(float *out,
                              const float *q,
                              const float *raw,
                              const float *comp,
                              float sink,
                              uint32_t ratio,
                              uint32_t n_tokens,
                              uint32_t n_comp) {
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t raw_count = t + 1u;
        uint32_t visible_comp = (t + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
        const uint32_t n_score = raw_count + visible_comp;
        float scores[256];
        float max_s = sink;
        const float *qt = q + (uint64_t)t * HEAD_DIM;
        for (uint32_t r = 0; r < raw_count; r++) {
            const float *kv = raw + (uint64_t)r * HEAD_DIM;
            double dot = 0.0;
            for (uint32_t d = 0; d < HEAD_DIM; d++) dot += (double)qt[d] * (double)kv[d];
            scores[r] = (float)dot * scale;
            max_s = fmaxf(max_s, scores[r]);
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            const float *kv = comp + (uint64_t)c * HEAD_DIM;
            double dot = 0.0;
            for (uint32_t d = 0; d < HEAD_DIM; d++) dot += (double)qt[d] * (double)kv[d];
            scores[raw_count + c] = (float)dot * scale;
            max_s = fmaxf(max_s, scores[raw_count + c]);
        }
        float den = expf(sink - max_s);
        for (uint32_t i = 0; i < n_score; i++) {
            scores[i] = expf(scores[i] - max_s);
            den += scores[i];
        }
        float *ot = out + (uint64_t)t * HEAD_DIM;
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) {
                acc += raw[(uint64_t)r * HEAD_DIM + d] * scores[r];
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                acc += comp[(uint64_t)c * HEAD_DIM + d] * scores[raw_count + c];
            }
            ot[d] = acc / den;
        }
    }
}

static int compare_f32(const char *label,
                       const float *got,
                       const float *want,
                       uint64_t n,
                       float tol) {
    for (uint64_t i = 0; i < n; i++) {
        if (fabsf(got[i] - want[i]) > tol) {
            fprintf(stderr,
                    "cuda_v100_projection_attention_smoke: %s[%llu] got %.8g expected %.8g\n",
                    label,
                    (unsigned long long)i,
                    got[i],
                    want[i]);
            return 0;
        }
    }
    return 1;
}

static int compare_f16_row_to_f32(const char *label,
                                  const uint16_t *got,
                                  const float *want,
                                  uint32_t n,
                                  float tol) {
    for (uint32_t i = 0; i < n; i++) {
        const float got_f = f16_bits_to_f32(got[i]);
        if (fabsf(got_f - want[i]) > tol) {
            fprintf(stderr,
                    "cuda_v100_projection_attention_smoke: %s[%u] got %.8g expected %.8g\n",
                    label,
                    i,
                    got_f,
                    want[i]);
            return 0;
        }
    }
    return 1;
}

static int read_stage_f16_row(ds4_v100_cuda_context *ctx,
                              const ds4_v100_cuda_layer_kv_view *view,
                              uint64_t base_offset,
                              uint64_t row,
                              uint32_t dim,
                              uint16_t *out,
                              char *err,
                              size_t errlen) {
    const uint64_t off = base_offset + row * dim * sizeof(uint16_t);
    return ds4_v100_cuda_context_read_kv_arena(ctx,
                                               view->stage_id,
                                               off,
                                               out,
                                               (uint64_t)dim * sizeof(uint16_t),
                                               err,
                                               errlen);
}

static void run_case(uint32_t ratio) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * HEAD_DIM;
    const uint32_t n_tokens = ratio == 4u ? 8u : 128u;
    const uint32_t n_comp = n_tokens / ratio;
    const float rms_eps = 1e-6f;
    const uint64_t row_bytes = ds4_src_f8_e4m3_b128_row_bytes(IN_DIM);
    const uint32_t stride = (uint32_t)row_bytes;

    const uint64_t q_off = 0;
    const uint64_t kv_off = q_off + (uint64_t)HEAD_DIM * stride;
    const uint64_t sc_off = kv_off + (uint64_t)width * stride;
    const uint64_t idx_off = sc_off + (uint64_t)width * stride;
    const uint64_t idx_rows = ratio == 4u ? INDEX_DIM : 0u;
    const uint64_t payload_bytes = idx_off + idx_rows * stride;
    uint8_t *payload = (uint8_t *)malloc((size_t)payload_bytes);
    check(payload != NULL, "source payload alloc");
    if (!payload) return;
    memset(payload, 0, (size_t)payload_bytes);
    fill_f8_matrix(payload, q_off, HEAD_DIM, stride, IN_DIM, 11u + ratio);
    fill_f8_matrix(payload, kv_off, width, stride, IN_DIM, 29u + ratio);
    fill_f8_matrix(payload, sc_off, width, stride, IN_DIM, 47u + ratio);
    if (ratio == 4u) {
        fill_f8_matrix(payload, idx_off, INDEX_DIM, stride, IN_DIM, 71u + ratio);
    }

    ds4_gpu_source_row_view q_view = {
        .arena_offset = q_off,
        .byte_length = (uint64_t)HEAD_DIM * stride,
        .rows = HEAD_DIM,
        .cols = IN_DIM,
        .row_stride_bytes = stride,
    };
    ds4_gpu_source_row_view kv_view = {
        .arena_offset = kv_off,
        .byte_length = (uint64_t)width * stride,
        .rows = width,
        .cols = IN_DIM,
        .row_stride_bytes = stride,
    };
    ds4_gpu_source_row_view sc_view = {
        .arena_offset = sc_off,
        .byte_length = (uint64_t)width * stride,
        .rows = width,
        .cols = IN_DIM,
        .row_stride_bytes = stride,
    };
    ds4_gpu_source_row_view idx_view = {
        .arena_offset = idx_off,
        .byte_length = idx_rows * stride,
        .rows = (uint32_t)idx_rows,
        .cols = IN_DIM,
        .row_stride_bytes = stride,
    };

    float *q_host = (float *)calloc((size_t)n_tokens * HEAD_DIM, sizeof(float));
    float *raw_host = (float *)calloc((size_t)n_tokens * HEAD_DIM, sizeof(float));
    float *kv_host = (float *)calloc((size_t)n_tokens * width, sizeof(float));
    float *sc_host = (float *)calloc((size_t)n_tokens * width, sizeof(float));
    float *comp_host = (float *)calloc((size_t)n_comp * HEAD_DIM, sizeof(float));
    float *comp_want = (float *)calloc(HEAD_DIM, sizeof(float));
    float *heads_host = (float *)calloc((size_t)n_tokens * HEAD_DIM, sizeof(float));
    float *heads_want = (float *)calloc((size_t)n_tokens * HEAD_DIM, sizeof(float));
    float *stage_index_host = ratio == 4u ? (float *)calloc(INDEX_DIM, sizeof(float)) : NULL;
    float *x = (float *)calloc(IN_DIM, sizeof(float));
    check(q_host && raw_host && kv_host && sc_host && comp_host && comp_want &&
          heads_host && heads_want && x && (ratio != 4u || stage_index_host),
          "host buffers allocate");
    if (!q_host || !raw_host || !kv_host || !sc_host || !comp_host ||
        !comp_want || !heads_host || !heads_want || !x ||
        (ratio == 4u && !stage_index_host)) {
        free(x);
        free(stage_index_host);
        free(heads_want);
        free(heads_host);
        free(comp_want);
        free(comp_host);
        free(sc_host);
        free(kv_host);
        free(raw_host);
        free(q_host);
        free(payload);
        return;
    }

    ds4_gpu_arena *arena = NULL;
    check(ds4_gpu_arena_open(&arena, 0, payload_bytes) == 0, "source arena open");
    check(arena && ds4_gpu_arena_upload(arena, 0, payload, payload_bytes) == 0,
          "source arena upload");
    if (!arena) {
        free(x);
        free(stage_index_host);
        free(heads_want);
        free(heads_host);
        free(comp_want);
        free(comp_host);
        free(sc_host);
        free(kv_host);
        free(raw_host);
        free(q_host);
        free(payload);
        return;
    }

    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc(IN_DIM * sizeof(float));
    ds4_gpu_tensor *tmp_t = ds4_gpu_tensor_alloc((uint64_t)width * sizeof(float));
    ds4_gpu_tensor *idx_t = ratio == 4u ? ds4_gpu_tensor_alloc(INDEX_DIM * sizeof(float)) : NULL;
    ds4_gpu_tensor *q_t = ds4_gpu_tensor_alloc((uint64_t)n_tokens * HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *raw_t = ds4_gpu_tensor_alloc((uint64_t)n_tokens * HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *kv_t = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *sc_t = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *comp_t = ds4_gpu_tensor_alloc((uint64_t)n_comp * HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *state_kv_t = ds4_gpu_tensor_alloc((uint64_t)(ratio == 4u ? 8u : 128u) *
                                                      width * sizeof(float));
    ds4_gpu_tensor *state_sc_t = ds4_gpu_tensor_alloc((uint64_t)(ratio == 4u ? 8u : 128u) *
                                                      width * sizeof(float));
    ds4_gpu_tensor *heads_t = ds4_gpu_tensor_alloc((uint64_t)n_tokens * HEAD_DIM * sizeof(float));
    check(x_t && tmp_t && q_t && raw_t && kv_t && sc_t && comp_t &&
          state_kv_t && state_sc_t && heads_t && (ratio != 4u || idx_t),
          "device buffers allocate");
    if (x_t && tmp_t && q_t && raw_t && kv_t && sc_t && comp_t &&
        state_kv_t && state_sc_t && heads_t && (ratio != 4u || idx_t)) {
        for (uint32_t t = 0; t < n_tokens; t++) {
            fill_x(x, t, ratio);
            check(host_project(q_host + (uint64_t)t * HEAD_DIM, payload, &q_view, x),
                  "host q projection");
            check(device_project_copy(arena, &q_view, x_t, tmp_t, q_t,
                                      (uint64_t)t * HEAD_DIM * sizeof(float), x),
                  "device q projection");

            check(host_project(kv_host + (uint64_t)t * width, payload, &kv_view, x),
                  "host kv projection");
            memcpy(raw_host + (uint64_t)t * HEAD_DIM,
                   kv_host + (uint64_t)t * width,
                   HEAD_DIM * sizeof(float));
            check(device_project_copy(arena, &kv_view, x_t, tmp_t, kv_t,
                                      (uint64_t)t * width * sizeof(float), x),
                  "device kv projection");
            check(ds4_gpu_tensor_copy(raw_t,
                                      (uint64_t)t * HEAD_DIM * sizeof(float),
                                      tmp_t,
                                      0,
                                      HEAD_DIM * sizeof(float)),
                  "device raw projection copy");

            check(host_project(sc_host + (uint64_t)t * width, payload, &sc_view, x),
                  "host score projection");
            check(device_project_copy(arena, &sc_view, x_t, tmp_t, sc_t,
                                      (uint64_t)t * width * sizeof(float), x),
                  "device score projection");
        }

        uint64_t ape_bytes = (uint64_t)ratio * width * sizeof(float);
        uint64_t norm_offset = ape_bytes;
        uint64_t sink_offset = norm_offset + HEAD_DIM * sizeof(float);
        uint64_t model_bytes = sink_offset + sizeof(float);
        float *model = (float *)calloc(1, (size_t)model_bytes);
        check(model != NULL, "model scratch allocate");
        if (model) {
            for (uint32_t i = 0; i < HEAD_DIM; i++) {
                model[(norm_offset / sizeof(float)) + i] = 1.0f;
            }
            check(ds4_gpu_set_model_map(model, model_bytes), "model map");
            check(ds4_gpu_compressor_prefill_tensor(comp_t,
                                                    state_kv_t,
                                                    state_sc_t,
                                                    kv_t,
                                                    sc_t,
                                                    model,
                                                    model_bytes,
                                                    0,
                                                    0,
                                                    norm_offset,
                                                    0,
                                                    HEAD_DIM,
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
            check(ds4_gpu_tensor_read(comp_t,
                                      0,
                                      comp_host,
                                      (uint64_t)n_comp * HEAD_DIM * sizeof(float)),
                  "compressed rows read");
            for (uint32_t c = 0; c < n_comp; c++) {
                cpu_ref_comp_row(comp_want, kv_host, sc_host, ratio, HEAD_DIM, c, rms_eps);
                if (!compare_f32("compressed row",
                                 comp_host + (uint64_t)c * HEAD_DIM,
                                 comp_want,
                                 HEAD_DIM,
                                 2e-4f)) {
                    failures++;
                    break;
                }
            }

            check(ds4_gpu_attention_prefill_static_mixed_heads_tensor(heads_t,
                                                                      model,
                                                                      model_bytes,
                                                                      sink_offset,
                                                                      q_t,
                                                                      raw_t,
                                                                      comp_t,
                                                                      n_tokens,
                                                                      n_comp,
                                                                      0,
                                                                      ratio,
                                                                      1,
                                                                      HEAD_DIM),
                  "attention prefill");
            check(ds4_gpu_tensor_read(heads_t,
                                      0,
                                      heads_host,
                                      (uint64_t)n_tokens * HEAD_DIM * sizeof(float)),
                  "attention heads read");
            cpu_ref_attention(heads_want, q_host, raw_host, comp_host, 0.0f,
                              ratio, n_tokens, n_comp);
            check(compare_f32("attention heads",
                              heads_host,
                              heads_want,
                              (uint64_t)n_tokens * HEAD_DIM,
                              8e-4f),
                  "attention reference compare");
            free(model);
        }

        fill_x(x, n_tokens - 1u, ratio);
        check(device_project_copy(arena, &kv_view, x_t, tmp_t, NULL, 0, x),
              "device stage attention projection");
        if (ratio == 4u) {
            check(host_project(stage_index_host, payload, &idx_view, x),
                  "host indexer projection");
            check(device_project_copy(arena, &idx_view, x_t, idx_t, NULL, 0, x),
                  "device indexer projection");
        }

        ds4_v100_context_options opts;
        ds4_v100_context_options_init(&opts);
        opts.expected_gpus = 1;
        opts.kv_ctx_tokens = 4096;
        opts.kv_active_slots = 1;
        opts.scratch_bytes_per_gpu = 1024 * 1024;
        opts.reserve_bytes_per_gpu = 0;
        char err[512] = {0};
        ds4_v100_cuda_context *ctx = NULL;
        check(ds4_v100_cuda_context_open(&ctx, &opts, err, sizeof(err)) == 0,
              err[0] ? err : "context open");
        if (ctx) {
            const int layer = ratio == 4u ? 2 : 3;
            ds4_v100_cuda_layer_kv_view layer_view;
            check(ds4_v100_cuda_context_layer_kv_view(ctx, layer, &layer_view,
                                                       err, sizeof(err)) == 0,
                  err[0] ? err : "layer kv view");
            ds4_v100_cuda_prefill_kv_update_device update = {
                .slot = 0,
                .raw_row = ratio == 4u ? 9u : 11u,
                .comp_row = ratio == 4u ? 5u : 2u,
                .attn_row_device_f32 = ds4_gpu_tensor_contents(tmp_t),
                .indexer_row_device_f32 = ratio == 4u ? ds4_gpu_tensor_contents(idx_t) : NULL,
            };
            check(ds4_v100_cuda_context_prefill_kv_update_f16_device(ctx,
                                                                     layer,
                                                                     &update,
                                                                     err,
                                                                     sizeof(err)) == 0,
                  err[0] ? err : "device kv update");
            uint16_t half_row[HEAD_DIM];
            const float *stage_attn = kv_host + (uint64_t)(n_tokens - 1u) * width;
            check(read_stage_f16_row(ctx,
                                     &layer_view,
                                     layer_view.view.raw_swa_offset,
                                     update.raw_row,
                                     HEAD_DIM,
                                     half_row,
                                     err,
                                     sizeof(err)) == 0,
                  err[0] ? err : "stage raw read");
            check(compare_f16_row_to_f32("stage raw", half_row, stage_attn,
                                         HEAD_DIM, 2e-3f),
                  "stage raw compare");
            check(read_stage_f16_row(ctx,
                                     &layer_view,
                                     layer_view.view.compressed_attn_offset,
                                     update.comp_row,
                                     HEAD_DIM,
                                     half_row,
                                     err,
                                     sizeof(err)) == 0,
                  err[0] ? err : "stage comp read");
            check(compare_f16_row_to_f32("stage comp", half_row, stage_attn,
                                         HEAD_DIM, 2e-3f),
                  "stage comp compare");
            if (ratio == 4u) {
                uint16_t index_half[INDEX_DIM];
                check(read_stage_f16_row(ctx,
                                         &layer_view,
                                         layer_view.view.indexer_kv_offset,
                                         update.comp_row,
                                         INDEX_DIM,
                                         index_half,
                                         err,
                                         sizeof(err)) == 0,
                      err[0] ? err : "stage indexer read");
                check(compare_f16_row_to_f32("stage indexer",
                                             index_half,
                                             stage_index_host,
                                             INDEX_DIM,
                                             2e-3f),
                      "stage indexer compare");
            }
            ds4_v100_cuda_context_close(ctx);
        }
    }

    ds4_gpu_tensor_free(heads_t);
    ds4_gpu_tensor_free(state_sc_t);
    ds4_gpu_tensor_free(state_kv_t);
    ds4_gpu_tensor_free(comp_t);
    ds4_gpu_tensor_free(sc_t);
    ds4_gpu_tensor_free(kv_t);
    ds4_gpu_tensor_free(raw_t);
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(idx_t);
    ds4_gpu_tensor_free(tmp_t);
    ds4_gpu_tensor_free(x_t);
    ds4_gpu_arena_close(arena);
    free(x);
    free(stage_index_host);
    free(heads_want);
    free(heads_host);
    free(comp_want);
    free(comp_host);
    free(sc_host);
    free(kv_host);
    free(raw_host);
    free(q_host);
    free(payload);
}

int main(void) {
    setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    setenv("DS4_CUDA_NO_CUBLAS_ATTENTION", "1", 1);
    setenv("DS4_CUDA_NO_WINDOW_ATTENTION", "1", 1);
    if (!ds4_gpu_init()) return 1;
    if (ds4_gpu_device_count() < 1) {
        fprintf(stderr, "cuda_v100_projection_attention_smoke: no CUDA devices visible\n");
        return 1;
    }

    run_case(128);
    run_case(4);

    ds4_gpu_cleanup();
    if (failures) return 1;
    puts("cuda_v100_projection_attention_smoke: ok");
    return 0;
}
