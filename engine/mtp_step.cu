#include "engine/mtp_step.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    MTPF_N_EMBD = DS4_MTP_STEP_N_EMBD,
    MTPF_N_HC = DS4_MTP_STEP_N_HC,
    MTPF_HC_DIM = DS4_MTP_STEP_HC_VALUES,
    MTPF_HC_MIX = 2 * MTPF_N_HC + MTPF_N_HC * MTPF_N_HC,
    MTPF_N_HEAD = 64,
    MTPF_HEAD_DIM = 512,
    MTPF_N_ROT = 64,
    MTPF_RAW_CAP = DS4_MTP_STEP_RAW_CAP,
    MTPF_Q_LORA = 1024,
    MTPF_OUT_GROUPS = 8,
    MTPF_OUT_GROUP_DIM = 4096,
    MTPF_OUT_GROUP_RANK = 1024,
    MTPF_OUT_LOW_DIM = MTPF_OUT_GROUPS * MTPF_OUT_GROUP_RANK,
    MTPF_HC_SINKHORN_ITERS = 20,
    MTPF_N_EXPERT = 256,
    MTPF_N_ROUTE = 6,
    MTPF_N_FF_EXP = 2048,
};

#define MTPF_RMS_EPS 1.0e-6f
#define MTPF_HC_EPS  1.0e-6f
#define MTPF_ROUTED_SWIGLU_CLAMP 10.0f

typedef struct {
    ds4_gpu_source_row_view enorm;
    ds4_gpu_source_row_view hnorm;
    ds4_gpu_source_row_view e_proj;
    ds4_gpu_source_row_view h_proj;
    ds4_gpu_source_row_view hc_attn_fn;
    ds4_gpu_source_row_view hc_attn_scale;
    ds4_gpu_source_row_view hc_attn_base;
    ds4_gpu_source_row_view attn_norm;
    ds4_gpu_source_row_view attn_q_a;
    ds4_gpu_source_row_view attn_q_a_norm;
    ds4_gpu_source_row_view attn_q_b;
    ds4_gpu_source_row_view attn_kv;
    ds4_gpu_source_row_view attn_kv_norm;
    ds4_gpu_source_row_view attn_sinks;
    ds4_gpu_source_row_view attn_output_a;
    ds4_gpu_source_row_view attn_output_b;
    ds4_gpu_source_row_view hc_ffn_fn;
    ds4_gpu_source_row_view hc_ffn_scale;
    ds4_gpu_source_row_view hc_ffn_base;
    ds4_gpu_source_row_view ffn_norm;
    ds4_gpu_source_row_view ffn_gate_inp;
    ds4_gpu_source_row_view exp_probs_b;
    ds4_gpu_source_row_view ffn_gate_shexp;
    ds4_gpu_source_row_view ffn_up_shexp;
    ds4_gpu_source_row_view ffn_down_shexp;
    ds4_gpu_q4_k_expert_view ffn_gate_exps;
    ds4_gpu_q4_k_expert_view ffn_up_exps;
    ds4_gpu_q4_k_expert_view ffn_down_exps;
    ds4_gpu_source_row_view hc_head_fn;
    ds4_gpu_source_row_view hc_head_scale;
    ds4_gpu_source_row_view hc_head_base;
    ds4_gpu_source_row_view output_norm;
} mtpf_views;

typedef struct {
    ds4_gpu_tensor *embed_t;
    ds4_gpu_tensor *prev_hc_t;
    ds4_gpu_tensor *prefix_t;
    ds4_gpu_tensor *attn_next_t;
    ds4_gpu_tensor *ffn_next_t;
    ds4_gpu_tensor *row0;
    ds4_gpu_tensor *row1;
    ds4_gpu_tensor *hc0;
    ds4_gpu_tensor *hc1;
    ds4_gpu_tensor *mix_t;
    ds4_gpu_tensor *split_t;
    ds4_gpu_tensor *q_t;
    ds4_gpu_tensor *q_norm_t;
    ds4_gpu_tensor *heads_t;
    ds4_gpu_tensor *attn_heads_t;
    ds4_gpu_tensor *kv_t;
    ds4_gpu_tensor *raw_t;
    ds4_gpu_tensor *low_t;
    ds4_gpu_tensor *router_logits_t;
    ds4_gpu_tensor *router_probs_t;
    ds4_gpu_tensor *selected_t;
    ds4_gpu_tensor *weights_t;
    ds4_gpu_tensor *routed_t;
    ds4_gpu_tensor *q4_gate_tmp_t;
    ds4_gpu_tensor *q4_up_tmp_t;
    ds4_gpu_tensor *q4_mid_tmp_t;
    ds4_gpu_tensor *q4_down_tmp_t;
    ds4_gpu_tensor *shared_gate_t;
    ds4_gpu_tensor *shared_up_t;
    ds4_gpu_tensor *shared_mid_t;
    ds4_gpu_tensor *shared_t;
    ds4_gpu_tensor *ffn_t;
    ds4_gpu_tensor *head_pre_t;
    ds4_gpu_tensor *head_weights_t;
    ds4_gpu_tensor *logits_t;
    float *all_logits;
    uint64_t device_bytes;
    uint64_t host_bytes;
} mtpf_scratch;

struct ds4_mtp_forward {
    ds4_mtp_sidecar *sidecar;
    ds4_gpu_arena *output_arena;
    ds4_gpu_bf16_matrix_view output_view;
    mtpf_views views;
    mtpf_scratch scratch;
    int gpu;
    uint64_t output_weight_bytes;
    uint64_t free_after_output_upload_bytes;
    uint64_t run_count;
};

static int mtpf_error(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg ? msg : "MTP forward error");
    return 1;
}

static int source_view_rows(const ds4_gpu_source_row_view *src,
                            uint32_t row0,
                            uint32_t rows,
                            ds4_gpu_source_row_view *out) {
    if (!src || !out || rows == 0 || row0 > src->rows ||
        rows > src->rows - row0 || src->row_stride_bytes == 0) {
        return 1;
    }
    const uint64_t skip = (uint64_t)row0 * src->row_stride_bytes;
    const uint64_t byte_length = (uint64_t)rows * src->row_stride_bytes;
    if (skip > src->byte_length || byte_length > src->byte_length - skip) {
        return 1;
    }
    *out = *src;
    out->arena_offset += skip;
    out->byte_length = byte_length;
    out->rows = rows;
    return 0;
}

static void insert_topk(uint32_t *tokens,
                        float *logits,
                        uint32_t k,
                        uint32_t token,
                        float logit) {
    for (uint32_t i = 0; i < k; i++) {
        if (tokens[i] == UINT32_MAX || logit > logits[i]) {
            for (uint32_t j = k - 1; j > i; j--) {
                tokens[j] = tokens[j - 1];
                logits[j] = logits[j - 1];
            }
            tokens[i] = token;
            logits[i] = logit;
            return;
        }
    }
}

static void topk_from_logits(const float *all_logits,
                             uint32_t vocab,
                             uint32_t top_k,
                             uint32_t *tokens,
                             float *logits) {
    for (uint32_t i = 0; i < top_k; i++) {
        tokens[i] = UINT32_MAX;
        logits[i] = -FLT_MAX;
    }
    for (uint32_t i = 0; i < vocab; i++) {
        if (isfinite(all_logits[i])) insert_topk(tokens, logits, top_k, i, all_logits[i]);
    }
}

static int output_bf16_view_from_binding(const ds4_tensor_binding *b,
                                         ds4_gpu_bf16_matrix_view *out,
                                         char *err,
                                         size_t errlen) {
    if (!b || !out) return mtpf_error(err, errlen, "missing output binding");
    if (!b->source_dtype || strcmp(b->source_dtype, "bf16") != 0 ||
        b->n_shape_dims != 2 ||
        b->shape[0] != MTPF_N_EMBD ||
        b->shape[1] == 0 ||
        b->shape[1] > UINT32_MAX ||
        b->byte_length != b->shape[0] * b->shape[1] * sizeof(uint16_t)) {
        return mtpf_error(err, errlen, "invalid output.weight bf16 binding");
    }
    memset(out, 0, sizeof(*out));
    out->arena_offset = 0;
    out->byte_length = b->byte_length;
    out->rows = (uint32_t)b->shape[1];
    out->cols = (uint32_t)b->shape[0];
    out->row_stride_elements = (uint32_t)b->shape[0];
    return 0;
}

static int bind_views(ds4_mtp_sidecar *sidecar,
                      mtpf_views *v,
                      char *err,
                      size_t errlen) {
#define BIND_F32_VEC(name, field) \
    do { if (ds4_mtp_sidecar_f32_vector_view(sidecar, name, &v->field, err, errlen) != 0) return 1; } while (0)
#define BIND_F32_MAT(name, field) \
    do { if (ds4_mtp_sidecar_f32_matrix_view(sidecar, name, &v->field, err, errlen) != 0) return 1; } while (0)
#define BIND_Q8(name, field) \
    do { if (ds4_mtp_sidecar_q8_0_view(sidecar, name, &v->field, err, errlen) != 0) return 1; } while (0)
#define BIND_Q4(name, field) \
    do { if (ds4_mtp_sidecar_q4_k_expert_view(sidecar, name, &v->field, err, errlen) != 0) return 1; } while (0)
    BIND_F32_VEC("mtp.0.enorm.weight", enorm);
    BIND_F32_VEC("mtp.0.hnorm.weight", hnorm);
    BIND_Q8("mtp.0.e_proj.weight", e_proj);
    BIND_Q8("mtp.0.h_proj.weight", h_proj);
    BIND_F32_MAT("mtp.0.hc_attn_fn.weight", hc_attn_fn);
    BIND_F32_VEC("mtp.0.hc_attn_scale.weight", hc_attn_scale);
    BIND_F32_VEC("mtp.0.hc_attn_base.weight", hc_attn_base);
    BIND_F32_VEC("mtp.0.attn_norm.weight", attn_norm);
    BIND_Q8("mtp.0.attn_q_a.weight", attn_q_a);
    BIND_F32_VEC("mtp.0.attn_q_a_norm.weight", attn_q_a_norm);
    BIND_Q8("mtp.0.attn_q_b.weight", attn_q_b);
    BIND_Q8("mtp.0.attn_kv.weight", attn_kv);
    BIND_F32_VEC("mtp.0.attn_kv_a_norm.weight", attn_kv_norm);
    BIND_F32_VEC("mtp.0.attn_sinks.weight", attn_sinks);
    BIND_Q8("mtp.0.attn_output_a.weight", attn_output_a);
    BIND_Q8("mtp.0.attn_output_b.weight", attn_output_b);
    BIND_F32_MAT("mtp.0.hc_ffn_fn.weight", hc_ffn_fn);
    BIND_F32_VEC("mtp.0.hc_ffn_scale.weight", hc_ffn_scale);
    BIND_F32_VEC("mtp.0.hc_ffn_base.weight", hc_ffn_base);
    BIND_F32_VEC("mtp.0.ffn_norm.weight", ffn_norm);
    BIND_F32_MAT("mtp.0.ffn_gate_inp.weight", ffn_gate_inp);
    BIND_F32_VEC("mtp.0.exp_probs_b.bias", exp_probs_b);
    BIND_Q8("mtp.0.ffn_gate_shexp.weight", ffn_gate_shexp);
    BIND_Q8("mtp.0.ffn_up_shexp.weight", ffn_up_shexp);
    BIND_Q8("mtp.0.ffn_down_shexp.weight", ffn_down_shexp);
    BIND_Q4("mtp.0.ffn_gate_exps.weight", ffn_gate_exps);
    BIND_Q4("mtp.0.ffn_up_exps.weight", ffn_up_exps);
    BIND_Q4("mtp.0.ffn_down_exps.weight", ffn_down_exps);
    BIND_F32_MAT("mtp.0.hc_head_fn.weight", hc_head_fn);
    BIND_F32_VEC("mtp.0.hc_head_scale.weight", hc_head_scale);
    BIND_F32_VEC("mtp.0.hc_head_base.weight", hc_head_base);
    BIND_F32_VEC("mtp.0.norm.weight", output_norm);
#undef BIND_Q4
#undef BIND_Q8
#undef BIND_F32_MAT
#undef BIND_F32_VEC

    if (v->enorm.cols != MTPF_N_EMBD ||
        v->hnorm.cols != MTPF_N_EMBD ||
        v->e_proj.rows != MTPF_N_EMBD ||
        v->e_proj.cols != MTPF_N_EMBD ||
        v->h_proj.rows != MTPF_N_EMBD ||
        v->h_proj.cols != MTPF_N_EMBD ||
        v->hc_attn_fn.rows != MTPF_HC_MIX ||
        v->hc_attn_fn.cols != MTPF_HC_DIM ||
        v->hc_attn_scale.cols != 3u ||
        v->hc_attn_base.cols != MTPF_HC_MIX ||
        v->attn_norm.cols != MTPF_N_EMBD ||
        v->attn_q_a.rows != MTPF_Q_LORA ||
        v->attn_q_a.cols != MTPF_N_EMBD ||
        v->attn_q_a_norm.cols != MTPF_Q_LORA ||
        v->attn_q_b.rows != MTPF_N_HEAD * MTPF_HEAD_DIM ||
        v->attn_q_b.cols != MTPF_Q_LORA ||
        v->attn_kv.rows != MTPF_HEAD_DIM ||
        v->attn_kv.cols != MTPF_N_EMBD ||
        v->attn_kv_norm.cols != MTPF_HEAD_DIM ||
        v->attn_sinks.cols != MTPF_N_HEAD ||
        v->attn_output_a.rows != MTPF_OUT_LOW_DIM ||
        v->attn_output_a.cols != MTPF_OUT_GROUP_DIM ||
        v->attn_output_b.rows != MTPF_N_EMBD ||
        v->attn_output_b.cols != MTPF_OUT_LOW_DIM ||
        v->hc_ffn_fn.rows != MTPF_HC_MIX ||
        v->hc_ffn_fn.cols != MTPF_HC_DIM ||
        v->hc_ffn_scale.cols != 3u ||
        v->hc_ffn_base.cols != MTPF_HC_MIX ||
        v->ffn_norm.cols != MTPF_N_EMBD ||
        v->ffn_gate_inp.rows != MTPF_N_EXPERT ||
        v->ffn_gate_inp.cols != MTPF_N_EMBD ||
        v->exp_probs_b.cols != MTPF_N_EXPERT ||
        v->ffn_gate_shexp.rows != MTPF_N_FF_EXP ||
        v->ffn_gate_shexp.cols != MTPF_N_EMBD ||
        v->ffn_up_shexp.rows != MTPF_N_FF_EXP ||
        v->ffn_up_shexp.cols != MTPF_N_EMBD ||
        v->ffn_down_shexp.rows != MTPF_N_EMBD ||
        v->ffn_down_shexp.cols != MTPF_N_FF_EXP ||
        v->ffn_gate_exps.experts != MTPF_N_EXPERT ||
        v->ffn_up_exps.experts != MTPF_N_EXPERT ||
        v->ffn_down_exps.experts != MTPF_N_EXPERT ||
        v->hc_head_fn.rows != MTPF_N_HC ||
        v->hc_head_fn.cols != MTPF_HC_DIM ||
        v->hc_head_scale.cols != 1u ||
        v->hc_head_base.cols != MTPF_N_HC ||
        v->output_norm.cols != MTPF_N_EMBD) {
        return mtpf_error(err, errlen, "unexpected MTP forward tensor layout");
    }
    return 0;
}

static int grouped_output_arena(ds4_mtp_forward *fwd,
                                const ds4_gpu_tensor *heads,
                                ds4_gpu_tensor *low,
                                ds4_gpu_tensor *out) {
    ds4_gpu_arena *arena = ds4_mtp_sidecar_arena(fwd->sidecar);
    if (!arena || !heads || !low || !out) return 1;
    if (!ds4_gpu_tensor_fill_f32(low, 0.0f, MTPF_OUT_LOW_DIM)) return 1;
    for (uint32_t g = 0; g < MTPF_OUT_GROUPS; g++) {
        ds4_gpu_source_row_view group_view;
        if (source_view_rows(&fwd->views.attn_output_a,
                             g * MTPF_OUT_GROUP_RANK,
                             MTPF_OUT_GROUP_RANK,
                             &group_view) != 0) {
            return 1;
        }
        ds4_gpu_tensor *head_view = ds4_gpu_tensor_view(
                heads,
                (uint64_t)g * MTPF_OUT_GROUP_DIM * sizeof(float),
                (uint64_t)MTPF_OUT_GROUP_DIM * sizeof(float));
        ds4_gpu_tensor *low_view = ds4_gpu_tensor_view(
                low,
                (uint64_t)g * MTPF_OUT_GROUP_RANK * sizeof(float),
                (uint64_t)MTPF_OUT_GROUP_RANK * sizeof(float));
        if (!head_view || !low_view) {
            ds4_gpu_tensor_free(low_view);
            ds4_gpu_tensor_free(head_view);
            return 1;
        }
        const int failed = ds4_gpu_arena_q8_0_matmul_f32(
                arena,
                &group_view,
                head_view,
                low_view,
                1);
        ds4_gpu_tensor_free(low_view);
        ds4_gpu_tensor_free(head_view);
        if (failed) return 1;
    }
    return ds4_gpu_arena_q8_0_matmul_f32(
            arena,
            &fwd->views.attn_output_b,
            low,
            out,
            1);
}

static ds4_gpu_tensor *mtpf_scratch_tensor(mtpf_scratch *s, uint64_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    if (t) s->device_bytes += bytes;
    return t;
}

static void mtpf_scratch_free(mtpf_scratch *s) {
    if (!s) return;
    free(s->all_logits);
    ds4_gpu_tensor_free(s->logits_t);
    ds4_gpu_tensor_free(s->head_weights_t);
    ds4_gpu_tensor_free(s->head_pre_t);
    ds4_gpu_tensor_free(s->ffn_t);
    ds4_gpu_tensor_free(s->shared_t);
    ds4_gpu_tensor_free(s->shared_mid_t);
    ds4_gpu_tensor_free(s->shared_up_t);
    ds4_gpu_tensor_free(s->shared_gate_t);
    ds4_gpu_tensor_free(s->q4_down_tmp_t);
    ds4_gpu_tensor_free(s->q4_mid_tmp_t);
    ds4_gpu_tensor_free(s->q4_up_tmp_t);
    ds4_gpu_tensor_free(s->q4_gate_tmp_t);
    ds4_gpu_tensor_free(s->routed_t);
    ds4_gpu_tensor_free(s->weights_t);
    ds4_gpu_tensor_free(s->selected_t);
    ds4_gpu_tensor_free(s->router_probs_t);
    ds4_gpu_tensor_free(s->router_logits_t);
    ds4_gpu_tensor_free(s->low_t);
    ds4_gpu_tensor_free(s->raw_t);
    ds4_gpu_tensor_free(s->kv_t);
    ds4_gpu_tensor_free(s->attn_heads_t);
    ds4_gpu_tensor_free(s->heads_t);
    ds4_gpu_tensor_free(s->q_norm_t);
    ds4_gpu_tensor_free(s->q_t);
    ds4_gpu_tensor_free(s->split_t);
    ds4_gpu_tensor_free(s->mix_t);
    ds4_gpu_tensor_free(s->hc1);
    ds4_gpu_tensor_free(s->hc0);
    ds4_gpu_tensor_free(s->row1);
    ds4_gpu_tensor_free(s->row0);
    ds4_gpu_tensor_free(s->ffn_next_t);
    ds4_gpu_tensor_free(s->attn_next_t);
    ds4_gpu_tensor_free(s->prefix_t);
    ds4_gpu_tensor_free(s->prev_hc_t);
    ds4_gpu_tensor_free(s->embed_t);
    memset(s, 0, sizeof(*s));
}

static int mtpf_scratch_alloc(ds4_mtp_forward *fwd,
                              char *err,
                              size_t errlen) {
    if (!fwd) return mtpf_error(err, errlen, "missing MTP forward scratch owner");
    if (!ds4_gpu_set_device(fwd->gpu)) {
        return mtpf_error(err, errlen, "failed to set MTP scratch device");
    }
    mtpf_scratch *s = &fwd->scratch;
    mtpf_scratch_free(s);

    const uint64_t embd_bytes = (uint64_t)MTPF_N_EMBD * sizeof(float);
    const uint64_t hc_bytes = (uint64_t)MTPF_HC_DIM * sizeof(float);
    const uint64_t mix_bytes = (uint64_t)MTPF_HC_MIX * sizeof(float);
    const uint64_t q_lora_bytes = (uint64_t)MTPF_Q_LORA * sizeof(float);
    const uint64_t heads_bytes = (uint64_t)MTPF_N_HEAD * MTPF_HEAD_DIM * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)MTPF_HEAD_DIM * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)MTPF_RAW_CAP * MTPF_HEAD_DIM * sizeof(float);
    const uint64_t low_bytes = (uint64_t)MTPF_OUT_LOW_DIM * sizeof(float);
    const uint64_t mid_bytes = (uint64_t)MTPF_N_FF_EXP * sizeof(float);
    const uint64_t route_i32_bytes = (uint64_t)MTPF_N_ROUTE * sizeof(int32_t);
    const uint64_t route_f32_bytes = (uint64_t)MTPF_N_ROUTE * sizeof(float);
    const uint64_t probs_bytes = (uint64_t)MTPF_N_EXPERT * sizeof(float);
    const uint64_t q4_mid_values = (uint64_t)MTPF_N_ROUTE * MTPF_N_FF_EXP;
    const uint64_t q4_down_values = (uint64_t)MTPF_N_ROUTE * MTPF_N_EMBD;
    const uint64_t logits_bytes = (uint64_t)fwd->output_view.rows * sizeof(float);

#define ALLOC_T(field, bytes_) \
    do { s->field = mtpf_scratch_tensor(s, (bytes_)); } while (0)
    ALLOC_T(embed_t, embd_bytes);
    ALLOC_T(prev_hc_t, hc_bytes);
    ALLOC_T(prefix_t, hc_bytes);
    ALLOC_T(attn_next_t, hc_bytes);
    ALLOC_T(ffn_next_t, hc_bytes);
    ALLOC_T(row0, embd_bytes);
    ALLOC_T(row1, embd_bytes);
    ALLOC_T(hc0, hc_bytes);
    ALLOC_T(hc1, hc_bytes);
    ALLOC_T(mix_t, mix_bytes);
    ALLOC_T(split_t, mix_bytes);
    ALLOC_T(q_t, q_lora_bytes);
    ALLOC_T(q_norm_t, q_lora_bytes);
    ALLOC_T(heads_t, heads_bytes);
    ALLOC_T(attn_heads_t, heads_bytes);
    ALLOC_T(kv_t, kv_bytes);
    ALLOC_T(raw_t, raw_bytes);
    ALLOC_T(low_t, low_bytes);
    ALLOC_T(router_logits_t, probs_bytes);
    ALLOC_T(router_probs_t, probs_bytes);
    ALLOC_T(selected_t, route_i32_bytes);
    ALLOC_T(weights_t, route_f32_bytes);
    ALLOC_T(routed_t, embd_bytes);
    ALLOC_T(q4_gate_tmp_t, q4_mid_values * sizeof(float));
    ALLOC_T(q4_up_tmp_t, q4_mid_values * sizeof(float));
    ALLOC_T(q4_mid_tmp_t, q4_mid_values * sizeof(float));
    ALLOC_T(q4_down_tmp_t, q4_down_values * sizeof(float));
    ALLOC_T(shared_gate_t, mid_bytes);
    ALLOC_T(shared_up_t, mid_bytes);
    ALLOC_T(shared_mid_t, mid_bytes);
    ALLOC_T(shared_t, embd_bytes);
    ALLOC_T(ffn_t, embd_bytes);
    ALLOC_T(head_pre_t, (uint64_t)MTPF_N_HC * sizeof(float));
    ALLOC_T(head_weights_t, (uint64_t)MTPF_N_HC * sizeof(float));
    ALLOC_T(logits_t, logits_bytes);
#undef ALLOC_T

    s->all_logits = (float *)malloc((size_t)logits_bytes);
    if (s->all_logits) s->host_bytes = logits_bytes;

    if (!s->embed_t || !s->prev_hc_t || !s->prefix_t || !s->attn_next_t ||
        !s->ffn_next_t || !s->row0 || !s->row1 || !s->hc0 || !s->hc1 ||
        !s->mix_t || !s->split_t || !s->q_t || !s->q_norm_t || !s->heads_t ||
        !s->attn_heads_t || !s->kv_t || !s->raw_t || !s->low_t ||
        !s->router_logits_t || !s->router_probs_t || !s->selected_t ||
        !s->weights_t || !s->routed_t || !s->q4_gate_tmp_t ||
        !s->q4_up_tmp_t || !s->q4_mid_tmp_t || !s->q4_down_tmp_t ||
        !s->shared_gate_t || !s->shared_up_t || !s->shared_mid_t ||
        !s->shared_t || !s->ffn_t || !s->head_pre_t || !s->head_weights_t ||
        !s->logits_t || !s->all_logits) {
        mtpf_scratch_free(s);
        return mtpf_error(err, errlen, "MTP forward persistent scratch allocation failed");
    }
    return 0;
}

int ds4_mtp_forward_open(ds4_mtp_forward **out,
                              ds4_mtp_sidecar *sidecar,
                              const void *base_model,
                              uint64_t base_model_size,
                              const ds4_tensor_binding *output_weight,
                              int gpu,
                              char *err,
                              size_t errlen) {
    if (!out) return mtpf_error(err, errlen, "missing MTP forward output");
    *out = NULL;
    if (!sidecar || !base_model || !output_weight) {
        return mtpf_error(err, errlen, "missing MTP forward input");
    }
    if (output_weight->source_offset > base_model_size ||
        output_weight->byte_length > base_model_size - output_weight->source_offset) {
        return mtpf_error(err, errlen, "output.weight outside model map");
    }
    ds4_mtp_forward *fwd =
        (ds4_mtp_forward *)calloc(1, sizeof(*fwd));
    if (!fwd) return mtpf_error(err, errlen, "failed to allocate MTP forward");
    fwd->sidecar = sidecar;
    fwd->gpu = gpu;
    if (output_bf16_view_from_binding(output_weight, &fwd->output_view, err, errlen) ||
        bind_views(sidecar, &fwd->views, err, errlen)) {
        ds4_mtp_forward_close(fwd);
        return 1;
    }
    if (ds4_gpu_arena_open(&fwd->output_arena, gpu, output_weight->byte_length) != 0 ||
        !ds4_gpu_arena_is_device_memory(fwd->output_arena) ||
        ds4_gpu_arena_upload(fwd->output_arena,
                             0,
                             (const unsigned char *)base_model + output_weight->source_offset,
                             output_weight->byte_length) != 0) {
        ds4_mtp_forward_close(fwd);
        return mtpf_error(err, errlen, "output.weight upload failed");
    }
    fwd->output_weight_bytes = output_weight->byte_length;
    fwd->free_after_output_upload_bytes =
        ds4_gpu_arena_free_after_upload_bytes(fwd->output_arena);
    if (mtpf_scratch_alloc(fwd, err, errlen)) {
        ds4_mtp_forward_close(fwd);
        return 1;
    }
    *out = fwd;
    return 0;
}

int ds4_mtp_forward_run_host_next_hc(ds4_mtp_forward *fwd,
                                          const float *embed,
                                          const float *prev_hc,
                                          uint32_t position,
                                          uint32_t top_k,
                                          uint32_t *tokens,
                                          float *out_logits,
                                          float *next_hc,
                                          uint64_t next_hc_values,
                                          ds4_mtp_forward_report *report,
                                          char *err,
                                          size_t errlen) {
    if (!fwd || !embed || !prev_hc || !tokens || !out_logits ||
        top_k == 0 || top_k > DS4_MTP_STEP_MAX_TOPK) {
        return mtpf_error(err, errlen, "invalid MTP forward run input");
    }
    if (next_hc && next_hc_values < MTPF_HC_DIM) {
        return mtpf_error(err, errlen, "MTP forward next_hc buffer is too small");
    }
    if (!ds4_gpu_set_device(fwd->gpu)) {
        return mtpf_error(err, errlen, "failed to set MTP forward device");
    }
    ds4_gpu_arena *arena = ds4_mtp_sidecar_arena(fwd->sidecar);
    if (!arena) return mtpf_error(err, errlen, "missing MTP sidecar arena");

    const uint64_t embd_bytes = (uint64_t)MTPF_N_EMBD * sizeof(float);
    const uint64_t hc_bytes = (uint64_t)MTPF_HC_DIM * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)fwd->output_view.rows * sizeof(float);

    mtpf_scratch *s = &fwd->scratch;
    ds4_gpu_tensor *embed_t = s->embed_t;
    ds4_gpu_tensor *prev_hc_t = s->prev_hc_t;
    ds4_gpu_tensor *prefix_t = s->prefix_t;
    ds4_gpu_tensor *attn_next_t = s->attn_next_t;
    ds4_gpu_tensor *ffn_next_t = s->ffn_next_t;
    ds4_gpu_tensor *row0 = s->row0;
    ds4_gpu_tensor *row1 = s->row1;
    ds4_gpu_tensor *hc0 = s->hc0;
    ds4_gpu_tensor *hc1 = s->hc1;
    ds4_gpu_tensor *mix_t = s->mix_t;
    ds4_gpu_tensor *split_t = s->split_t;
    ds4_gpu_tensor *q_t = s->q_t;
    ds4_gpu_tensor *q_norm_t = s->q_norm_t;
    ds4_gpu_tensor *heads_t = s->heads_t;
    ds4_gpu_tensor *attn_heads_t = s->attn_heads_t;
    ds4_gpu_tensor *kv_t = s->kv_t;
    ds4_gpu_tensor *raw_t = s->raw_t;
    ds4_gpu_tensor *low_t = s->low_t;
    ds4_gpu_tensor *router_logits_t = s->router_logits_t;
    ds4_gpu_tensor *router_probs_t = s->router_probs_t;
    ds4_gpu_tensor *selected_t = s->selected_t;
    ds4_gpu_tensor *weights_t = s->weights_t;
    ds4_gpu_tensor *routed_t = s->routed_t;
    ds4_gpu_tensor *q4_gate_tmp_t = s->q4_gate_tmp_t;
    ds4_gpu_tensor *q4_up_tmp_t = s->q4_up_tmp_t;
    ds4_gpu_tensor *q4_mid_tmp_t = s->q4_mid_tmp_t;
    ds4_gpu_tensor *q4_down_tmp_t = s->q4_down_tmp_t;
    ds4_gpu_tensor *shared_gate_t = s->shared_gate_t;
    ds4_gpu_tensor *shared_up_t = s->shared_up_t;
    ds4_gpu_tensor *shared_mid_t = s->shared_mid_t;
    ds4_gpu_tensor *shared_t = s->shared_t;
    ds4_gpu_tensor *ffn_t = s->ffn_t;
    ds4_gpu_tensor *head_pre_t = s->head_pre_t;
    ds4_gpu_tensor *head_weights_t = s->head_weights_t;
    ds4_gpu_tensor *logits_t = s->logits_t;
    float *all_logits = s->all_logits;
    int rc = 1;

    if (!embed_t || !prev_hc_t || !prefix_t || !attn_next_t || !ffn_next_t ||
        !row0 || !row1 || !hc0 || !hc1 || !mix_t || !split_t || !q_t ||
        !q_norm_t || !heads_t || !attn_heads_t || !kv_t || !raw_t || !low_t ||
        !router_logits_t || !router_probs_t || !selected_t || !weights_t ||
        !routed_t || !q4_gate_tmp_t || !q4_up_tmp_t || !q4_mid_tmp_t ||
        !q4_down_tmp_t || !shared_gate_t || !shared_up_t || !shared_mid_t ||
        !shared_t || !ffn_t || !head_pre_t || !head_weights_t || !logits_t ||
        !all_logits) {
        mtpf_error(err, errlen, "MTP forward scratch is not initialized");
        goto done;
    }

    const uint32_t raw_row = position % MTPF_RAW_CAP;
    if (!ds4_gpu_tensor_write(embed_t, 0, embed, embd_bytes) ||
        !ds4_gpu_tensor_write(prev_hc_t, 0, prev_hc, hc_bytes) ||
        !ds4_gpu_tensor_fill_f32(raw_t, 0.0f, MTPF_RAW_CAP * MTPF_HEAD_DIM) ||
        ds4_gpu_arena_f32_rms_norm_f32(arena, &fwd->views.enorm, embed_t, row0, MTPF_N_EMBD, 1, MTPF_RMS_EPS) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &fwd->views.e_proj, row0, row1, 1) != 0 ||
        !ds4_gpu_repeat_hc_tensor(hc0, row1, MTPF_N_EMBD, MTPF_N_HC) ||
        ds4_gpu_arena_f32_rms_norm_f32(arena, &fwd->views.hnorm, prev_hc_t, hc1, MTPF_N_EMBD, MTPF_N_HC, MTPF_RMS_EPS) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &fwd->views.h_proj, hc1, prefix_t, MTPF_N_HC) != 0 ||
        !ds4_gpu_add_tensor(prefix_t, hc0, prefix_t, MTPF_HC_DIM) ||
        !ds4_gpu_rms_norm_plain_tensor(hc0, prefix_t, MTPF_HC_DIM, MTPF_RMS_EPS) ||
        ds4_gpu_arena_f32_matmul_f32(arena, &fwd->views.hc_attn_fn, hc0, mix_t) != 0 ||
        ds4_gpu_arena_hc_split_weighted_sum_tensor(arena, &fwd->views.hc_attn_scale, &fwd->views.hc_attn_base, row0, split_t, mix_t, prefix_t, MTPF_N_EMBD, MTPF_N_HC, MTPF_HC_SINKHORN_ITERS, MTPF_HC_EPS) != 0 ||
        ds4_gpu_arena_f32_rms_norm_f32(arena, &fwd->views.attn_norm, row0, row1, MTPF_N_EMBD, 1, MTPF_RMS_EPS) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &fwd->views.attn_q_a, row1, q_t, 1) != 0 ||
        ds4_gpu_arena_f32_rms_norm_f32(arena, &fwd->views.attn_q_a_norm, q_t, q_norm_t, MTPF_Q_LORA, 1, MTPF_RMS_EPS) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &fwd->views.attn_q_b, q_norm_t, heads_t, 1) != 0 ||
        !ds4_gpu_head_rms_norm_tensor(heads_t, 1, MTPF_N_HEAD, MTPF_HEAD_DIM, MTPF_RMS_EPS) ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &fwd->views.attn_kv, row1, kv_t, 1) != 0 ||
        ds4_gpu_arena_f32_rms_norm_f32(arena, &fwd->views.attn_kv_norm, kv_t, kv_t, MTPF_HEAD_DIM, 1, MTPF_RMS_EPS) != 0 ||
        !ds4_gpu_kv_fp8_store_raw_tensor(kv_t, raw_t, MTPF_RAW_CAP, raw_row, MTPF_HEAD_DIM, MTPF_N_ROT) ||
        ds4_gpu_arena_attention_decode_heads_tensor(arena, &fwd->views.attn_sinks, attn_heads_t, heads_t, raw_t, 1, MTPF_RAW_CAP, raw_row, NULL, 0, NULL, 0, MTPF_N_HEAD, MTPF_HEAD_DIM) != 0 ||
        grouped_output_arena(fwd, attn_heads_t, low_t, row0) != 0 ||
        !ds4_gpu_hc_expand_split_tensor(attn_next_t, row0, prefix_t, split_t, MTPF_N_EMBD, MTPF_N_HC) ||
        !ds4_gpu_rms_norm_plain_tensor(hc0, attn_next_t, MTPF_HC_DIM, MTPF_RMS_EPS) ||
        ds4_gpu_arena_f32_matmul_f32(arena, &fwd->views.hc_ffn_fn, hc0, mix_t) != 0 ||
        ds4_gpu_arena_hc_split_weighted_sum_tensor(arena, &fwd->views.hc_ffn_scale, &fwd->views.hc_ffn_base, row0, split_t, mix_t, attn_next_t, MTPF_N_EMBD, MTPF_N_HC, MTPF_HC_SINKHORN_ITERS, MTPF_HC_EPS) != 0 ||
        ds4_gpu_arena_f32_rms_norm_f32(arena, &fwd->views.ffn_norm, row0, row1, MTPF_N_EMBD, 1, MTPF_RMS_EPS) != 0 ||
        ds4_gpu_arena_f32_matmul_f32(arena, &fwd->views.ffn_gate_inp, row1, router_logits_t) != 0 ||
        ds4_gpu_arena_router_select_bias_tensor(arena, &fwd->views.exp_probs_b, selected_t, weights_t, router_probs_t, router_logits_t) != 0 ||
        ds4_gpu_arena_q4_k_routed_moe_one_f32(arena, &fwd->views.ffn_gate_exps, &fwd->views.ffn_up_exps, &fwd->views.ffn_down_exps, routed_t, q4_gate_tmp_t, q4_up_tmp_t, q4_mid_tmp_t, q4_down_tmp_t, selected_t, weights_t, row1, MTPF_N_ROUTE, MTPF_ROUTED_SWIGLU_CLAMP) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &fwd->views.ffn_gate_shexp, row1, shared_gate_t, 1) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &fwd->views.ffn_up_shexp, row1, shared_up_t, 1) != 0 ||
        !ds4_gpu_swiglu_tensor(shared_mid_t, shared_gate_t, shared_up_t, MTPF_N_FF_EXP, 0.0f, 1.0f) ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &fwd->views.ffn_down_shexp, shared_mid_t, shared_t, 1) != 0 ||
        !ds4_gpu_add_tensor(ffn_t, shared_t, routed_t, MTPF_N_EMBD) ||
        !ds4_gpu_hc_expand_add_split_tensor(ffn_next_t, shared_t, routed_t, attn_next_t, split_t, MTPF_N_EMBD, MTPF_N_HC) ||
        !ds4_gpu_rms_norm_plain_tensor(hc0, ffn_next_t, MTPF_HC_DIM, MTPF_RMS_EPS) ||
        ds4_gpu_arena_f32_matmul_f32(arena, &fwd->views.hc_head_fn, hc0, head_pre_t) != 0 ||
        ds4_gpu_arena_output_hc_weights_tensor(arena, &fwd->views.hc_head_scale, &fwd->views.hc_head_base, head_weights_t, head_pre_t, MTPF_N_HC, MTPF_HC_EPS) != 0 ||
        !ds4_gpu_hc_weighted_sum_tensor(row0, ffn_next_t, head_weights_t, MTPF_N_EMBD, MTPF_N_HC) ||
        ds4_gpu_arena_f32_rms_norm_f32(arena, &fwd->views.output_norm, row0, row1, MTPF_N_EMBD, 1, MTPF_RMS_EPS) != 0 ||
        ds4_gpu_arena_bf16_matmul_f32(fwd->output_arena, &fwd->output_view, row1, logits_t) != 0 ||
        !ds4_gpu_synchronize() ||
        (next_hc && !ds4_gpu_tensor_read(ffn_next_t, 0, next_hc, hc_bytes)) ||
        !ds4_gpu_tensor_read(logits_t, 0, all_logits, logits_bytes)) {
        mtpf_error(err, errlen, "MTP forward GPU sequence failed");
        goto done;
    }

    topk_from_logits(all_logits, fwd->output_view.rows, top_k, tokens, out_logits);
    fwd->run_count++;
    if (report) {
        memset(report, 0, sizeof(*report));
        report->raw_row = raw_row;
        report->n_raw = 1;
        report->output_vocab = fwd->output_view.rows;
        report->output_weight_bytes = fwd->output_weight_bytes;
        report->free_after_output_upload_bytes = fwd->free_after_output_upload_bytes;
        report->scratch_device_bytes = s->device_bytes;
        report->scratch_host_bytes = s->host_bytes;
        report->run_count = fwd->run_count;
    }
    rc = 0;

done:
    return rc;
}

int ds4_mtp_forward_run_host(ds4_mtp_forward *fwd,
                                  const float *embed,
                                  const float *prev_hc,
                                  uint32_t position,
                                  uint32_t top_k,
                                  uint32_t *tokens,
                                  float *out_logits,
                                  ds4_mtp_forward_report *report,
                                  char *err,
                                  size_t errlen) {
    return ds4_mtp_forward_run_host_next_hc(fwd,
                                                 embed,
                                                 prev_hc,
                                                 position,
                                                 top_k,
                                                 tokens,
                                                 out_logits,
                                                 NULL,
                                                 0,
                                                 report,
                                                 err,
                                                 errlen);
}

void ds4_mtp_forward_close(ds4_mtp_forward *fwd) {
    if (!fwd) return;
    mtpf_scratch_free(&fwd->scratch);
    ds4_gpu_arena_close(fwd->output_arena);
    free(fwd);
}
