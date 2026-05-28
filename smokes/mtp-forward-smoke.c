#define main mtp_attn_smoke_unused_main
#define options mtp_attn_options
#define usage mtp_attn_usage
#define need_arg mtp_attn_need_arg
#define parse_options mtp_attn_parse_options
#define parse_int mtp_attn_parse_int
#define parse_double mtp_attn_parse_double
#include "mtp-attn-smoke.c"
#undef parse_double
#undef parse_int
#undef parse_options
#undef need_arg
#undef usage
#undef options
#undef main

#define main mtp_ffn_smoke_unused_main
#define options mtp_ffn_options
#define usage mtp_ffn_usage
#define need_arg mtp_ffn_need_arg
#define parse_options mtp_ffn_parse_options
#define parse_int mtp_ffn_parse_int
#define parse_double mtp_ffn_parse_double
#define now_ms mtp_ffn_now_ms
#define fill_hc_state mtp_ffn_fill_hc_state
#define f16_to_f32_host mtp_ffn_f16_to_f32_host
#define sigmoid_host mtp_ffn_sigmoid_host
#define rms_norm_plain_host mtp_ffn_rms_norm_plain_host
#define rms_norm_weight_host mtp_ffn_rms_norm_weight_host
#define matmul_f32_host mtp_ffn_matmul_f32_host
#define hc_split_sinkhorn_host mtp_ffn_hc_split_sinkhorn_host
#define hc_weighted_sum_host mtp_ffn_hc_weighted_sum_host
#define quantize_q8_0_activation_host mtp_ffn_quantize_q8_0_activation_host
#define dot_i8_host mtp_ffn_dot_i8_host
#define matmul_q8_0_host mtp_ffn_matmul_q8_0_host
#define compare_outputs mtp_ffn_compare_outputs
#define tensor_bytes mtp_ffn_tensor_bytes
#include "mtp-ffn-smoke.c"
#undef tensor_bytes
#undef compare_outputs
#undef matmul_q8_0_host
#undef dot_i8_host
#undef quantize_q8_0_activation_host
#undef hc_weighted_sum_host
#undef hc_split_sinkhorn_host
#undef matmul_f32_host
#undef rms_norm_weight_host
#undef rms_norm_plain_host
#undef sigmoid_host
#undef f16_to_f32_host
#undef fill_hc_state
#undef now_ms
#undef parse_double
#undef parse_int
#undef parse_options
#undef need_arg
#undef usage
#undef options
#undef main

#define main mtp_logits_smoke_unused_main
#define options mtp_logits_options
#define model_map mtp_logits_model_map
#define usage mtp_logits_usage
#define need_arg mtp_logits_need_arg
#define parse_options mtp_logits_parse_options
#define parse_int mtp_logits_parse_int
#define parse_double mtp_logits_parse_double
#define map_model_file mtp_logits_map_model_file
#define unmap_model_file mtp_logits_unmap_model_file
#define sigmoid_stable mtp_logits_sigmoid_stable
#define insert_topk mtp_logits_insert_topk
#define deterministic_hc mtp_logits_deterministic_hc
#define cpu_rms_norm_plain mtp_logits_cpu_rms_norm_plain
#define cpu_rms_norm_weight mtp_logits_cpu_rms_norm_weight
#define cpu_f32_matvec mtp_logits_cpu_f32_matvec
#define cpu_hc_weighted_sum mtp_logits_cpu_hc_weighted_sum
#define tensor_bytes mtp_logits_tensor_bytes
#define output_bf16_view_from_binding mtp_logits_output_bf16_view_from_binding
#define cpu_mtp_logits_topk mtp_logits_cpu_mtp_logits_topk
#define arena_upload_chunks mtp_logits_arena_upload_chunks
#define topk_from_logits mtp_logits_topk_from_logits
#include "mtp-logits-smoke.c"
#undef topk_from_logits
#undef arena_upload_chunks
#undef cpu_mtp_logits_topk
#undef output_bf16_view_from_binding
#undef tensor_bytes
#undef cpu_hc_weighted_sum
#undef cpu_f32_matvec
#undef cpu_rms_norm_weight
#undef cpu_rms_norm_plain
#undef deterministic_hc
#undef insert_topk
#undef sigmoid_stable
#undef unmap_model_file
#undef map_model_file
#undef parse_double
#undef parse_int
#undef parse_options
#undef need_arg
#undef usage
#undef model_map
#undef options
#undef main

enum {
    MTP_FORWARD_MAX_TOPK = 16,
};

typedef struct {
    const char *model;
    const char *mtp_model;
    const char *pack_index;
    const char *report_path;
    int gpu;
    int require_gpus;
    int reserve_mib;
    uint32_t top_k;
    double prefix_tol;
    double attn_tol;
    double ffn_tol;
    double logit_tol;
} fwd_options;

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
} fwd_views;

static void fwd_usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-v100-mtp-forward-smoke --model FILE --mtp-model FILE --pack-index FILE [options]\n"
            "\n"
            "Options:\n"
            "  --index FILE            Alias for --pack-index\n"
            "  --gpu N                 Upload and execute on CUDA device N. Default: 7\n"
            "  --require-gpus N        Require at least N visible CUDA devices\n"
            "  --reserve-mib N         Require this much free memory after upload. Default: 4096\n"
            "  --top-k N               Number of draft candidates to compare. Default: 5\n"
            "  --prefix-tol F          Prefix HC tolerance. Default: 0.05\n"
            "  --attn-tol F            Attention HC tolerance. Default: 0.75\n"
            "  --ffn-tol F             FFN HC tolerance. Default: 1.25\n"
            "  --logit-tol F           Selected-logit tolerance. Default: 0.10\n"
            "  --report FILE           Write detailed report to FILE instead of stdout\n");
}

static const char *fwd_need_arg(int *i, int argc, char **argv, const char *arg) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: %s requires an argument\n", arg);
        exit(2);
    }
    return argv[++*i];
}

static int fwd_parse_int(const char *s, const char *arg) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || !end || *end || v < 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: bad integer for %s: %s\n", arg, s);
        exit(2);
    }
    return (int)v;
}

static double fwd_parse_double(const char *s, const char *arg) {
    errno = 0;
    char *end = NULL;
    double v = strtod(s, &end);
    if (errno || !s[0] || !end || *end || !(v >= 0.0)) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: bad float for %s: %s\n", arg, s);
        exit(2);
    }
    return v;
}

static fwd_options fwd_parse_options(int argc, char **argv) {
    fwd_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.gpu = 7;
    opt.reserve_mib = 4096;
    opt.top_k = 5;
    opt.prefix_tol = 0.05;
    opt.attn_tol = 0.75;
    opt.ffn_tol = 1.25;
    opt.logit_tol = 0.10;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            fwd_usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "--model")) {
            opt.model = fwd_need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-model")) {
            opt.mtp_model = fwd_need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--pack-index") || !strcmp(arg, "--index")) {
            opt.pack_index = fwd_need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--report")) {
            opt.report_path = fwd_need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--gpu")) {
            opt.gpu = fwd_parse_int(fwd_need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--require-gpus")) {
            opt.require_gpus = fwd_parse_int(fwd_need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--reserve-mib")) {
            opt.reserve_mib = fwd_parse_int(fwd_need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--top-k")) {
            opt.top_k = (uint32_t)fwd_parse_int(fwd_need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--prefix-tol")) {
            opt.prefix_tol = fwd_parse_double(fwd_need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--attn-tol")) {
            opt.attn_tol = fwd_parse_double(fwd_need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ffn-tol")) {
            opt.ffn_tol = fwd_parse_double(fwd_need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--logit-tol")) {
            opt.logit_tol = fwd_parse_double(fwd_need_arg(&i, argc, argv, arg), arg);
        } else {
            fprintf(stderr, "ds4-v100-mtp-forward-smoke: unknown option: %s\n", arg);
            fwd_usage(stderr);
            exit(2);
        }
    }
    if (!opt.model || !opt.model[0] ||
        !opt.mtp_model || !opt.mtp_model[0] ||
        !opt.pack_index || !opt.pack_index[0] ||
        opt.top_k == 0 || opt.top_k > MTP_FORWARD_MAX_TOPK) {
        fwd_usage(stderr);
        exit(2);
    }
    return opt;
}

static void fwd_fill_embed(float *x) {
    for (uint32_t i = 0; i < MTP_ATTN_N_EMBD; i++) {
        int v = (int)((i * 193u + i / 11u + 7u) % 509u) - 254;
        x[i] = (float)v * 0.001953125f;
    }
}

static void fwd_rms_norm_weight_rows_host(float *out,
                                          const float *x,
                                          const float *weight,
                                          uint32_t n,
                                          uint32_t rows,
                                          float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        rms_norm_weight_host(out + (uint64_t)r * n,
                             x + (uint64_t)r * n,
                             weight,
                             n,
                             eps);
    }
}

static void fwd_topk_from_logits(const float *all_logits,
                                 uint32_t vocab,
                                 uint32_t top_k,
                                 uint32_t *tokens,
                                 float *logits) {
    for (uint32_t i = 0; i < top_k; i++) {
        tokens[i] = UINT32_MAX;
        logits[i] = -FLT_MAX;
    }
    for (uint32_t i = 0; i < vocab; i++) {
        if (isfinite(all_logits[i])) {
            mtp_logits_insert_topk(tokens, logits, top_k, i, all_logits[i]);
        }
    }
}

static int fwd_compare(const char *label,
                       const float *got,
                       const float *ref,
                       uint64_t n,
                       double tol,
                       FILE *report,
                       double *max_abs_out) {
    double max_abs = 0.0;
    double max_rel = 0.0;
    uint64_t max_i = 0;
    for (uint64_t i = 0; i < n; i++) {
        const double a = (double)got[i];
        const double b = (double)ref[i];
        const double d = fabs(a - b);
        const double denom = fmax(fabs(b), 1.0e-12);
        const double rel = d / denom;
        if (d > max_abs) {
            max_abs = d;
            max_i = i;
        }
        if (rel > max_rel) max_rel = rel;
    }
    if (max_abs_out) *max_abs_out = max_abs;
    fprintf(report,
            "mtp_forward_compare\t%s\tmax_abs=%.9g\tmax_rel=%.9g\tmax_i=%" PRIu64
            "\ttol=%.9g\t%s\n",
            label,
            max_abs,
            max_rel,
            max_i,
            tol,
            max_abs <= tol ? "PASS" : "FAIL");
    return max_abs <= tol ? 0 : 1;
}

static int fwd_bind_views(ds4_mtp_sidecar *sidecar,
                          fwd_views *v,
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

    if (v->enorm.cols != MTP_ATTN_N_EMBD ||
        v->hnorm.cols != MTP_ATTN_N_EMBD ||
        v->e_proj.rows != MTP_ATTN_N_EMBD ||
        v->e_proj.cols != MTP_ATTN_N_EMBD ||
        v->h_proj.rows != MTP_ATTN_N_EMBD ||
        v->h_proj.cols != MTP_ATTN_N_EMBD ||
        v->hc_attn_fn.rows != MTP_ATTN_HC_MIX ||
        v->hc_attn_fn.cols != MTP_ATTN_HC_DIM ||
        v->hc_attn_scale.cols != 3u ||
        v->hc_attn_base.cols != MTP_ATTN_HC_MIX ||
        v->attn_norm.cols != MTP_ATTN_N_EMBD ||
        v->attn_q_a.rows != MTP_ATTN_Q_LORA ||
        v->attn_q_a.cols != MTP_ATTN_N_EMBD ||
        v->attn_q_a_norm.cols != MTP_ATTN_Q_LORA ||
        v->attn_q_b.rows != MTP_ATTN_N_HEAD * MTP_ATTN_HEAD_DIM ||
        v->attn_q_b.cols != MTP_ATTN_Q_LORA ||
        v->attn_kv.rows != MTP_ATTN_HEAD_DIM ||
        v->attn_kv.cols != MTP_ATTN_N_EMBD ||
        v->attn_kv_norm.cols != MTP_ATTN_HEAD_DIM ||
        v->attn_sinks.cols != MTP_ATTN_N_HEAD ||
        v->attn_output_a.rows != MTP_ATTN_OUT_LOW_DIM ||
        v->attn_output_a.cols != MTP_ATTN_OUT_GROUP_DIM ||
        v->attn_output_b.rows != MTP_ATTN_N_EMBD ||
        v->attn_output_b.cols != MTP_ATTN_OUT_LOW_DIM ||
        v->hc_ffn_fn.rows != MTP_FFN_HC_MIX ||
        v->hc_ffn_fn.cols != MTP_FFN_HC_DIM ||
        v->hc_ffn_scale.cols != 3u ||
        v->hc_ffn_base.cols != MTP_FFN_HC_MIX ||
        v->ffn_norm.cols != MTP_FFN_N_EMBD ||
        v->ffn_gate_inp.rows != MTP_FFN_N_EXPERT ||
        v->ffn_gate_inp.cols != MTP_FFN_N_EMBD ||
        v->exp_probs_b.cols != MTP_FFN_N_EXPERT ||
        v->ffn_gate_shexp.rows != MTP_FFN_N_FF_EXP ||
        v->ffn_gate_shexp.cols != MTP_FFN_N_EMBD ||
        v->ffn_up_shexp.rows != MTP_FFN_N_FF_EXP ||
        v->ffn_up_shexp.cols != MTP_FFN_N_EMBD ||
        v->ffn_down_shexp.rows != MTP_FFN_N_EMBD ||
        v->ffn_down_shexp.cols != MTP_FFN_N_FF_EXP ||
        v->ffn_gate_exps.experts != MTP_FFN_N_EXPERT ||
        v->ffn_up_exps.experts != MTP_FFN_N_EXPERT ||
        v->ffn_down_exps.experts != MTP_FFN_N_EXPERT ||
        v->hc_head_fn.rows != MTP_ATTN_N_HC ||
        v->hc_head_fn.cols != MTP_ATTN_HC_DIM ||
        v->hc_head_scale.cols != 1u ||
        v->hc_head_base.cols != MTP_ATTN_N_HC ||
        v->output_norm.cols != MTP_ATTN_N_EMBD) {
        snprintf(err, errlen, "unexpected MTP forward tensor layout");
        return 1;
    }
    return 0;
}

static int fwd_cpu_reference(const mtp_logits_model_map *base,
                             ds4_mtp_sidecar *sidecar,
                             const ds4_tensor_binding *output_weight,
                             const fwd_views *v,
                             const float *embed,
                             const float *prev_hc,
                             float *ref_prefix,
                             float *ref_attn_next,
                             float *ref_ffn_next,
                             uint32_t top_k,
                             uint32_t *tokens,
                             float *logits) {
    const float *enorm = (const float *)tensor_bytes(sidecar, "mtp.0.enorm.weight");
    const float *hnorm = (const float *)tensor_bytes(sidecar, "mtp.0.hnorm.weight");
    const unsigned char *e_proj = tensor_bytes(sidecar, "mtp.0.e_proj.weight");
    const unsigned char *h_proj = tensor_bytes(sidecar, "mtp.0.h_proj.weight");
    const float *hc_attn_fn = (const float *)tensor_bytes(sidecar, "mtp.0.hc_attn_fn.weight");
    const float *hc_attn_scale = (const float *)tensor_bytes(sidecar, "mtp.0.hc_attn_scale.weight");
    const float *hc_attn_base = (const float *)tensor_bytes(sidecar, "mtp.0.hc_attn_base.weight");
    const float *attn_norm = (const float *)tensor_bytes(sidecar, "mtp.0.attn_norm.weight");
    const unsigned char *attn_q_a = tensor_bytes(sidecar, "mtp.0.attn_q_a.weight");
    const float *attn_q_a_norm = (const float *)tensor_bytes(sidecar, "mtp.0.attn_q_a_norm.weight");
    const unsigned char *attn_q_b = tensor_bytes(sidecar, "mtp.0.attn_q_b.weight");
    const unsigned char *attn_kv = tensor_bytes(sidecar, "mtp.0.attn_kv.weight");
    const float *attn_kv_norm = (const float *)tensor_bytes(sidecar, "mtp.0.attn_kv_a_norm.weight");
    const float *attn_sinks = (const float *)tensor_bytes(sidecar, "mtp.0.attn_sinks.weight");
    const unsigned char *attn_output_a = tensor_bytes(sidecar, "mtp.0.attn_output_a.weight");
    const unsigned char *attn_output_b = tensor_bytes(sidecar, "mtp.0.attn_output_b.weight");
    const float *hc_ffn_fn = (const float *)mtp_ffn_tensor_bytes(sidecar, "mtp.0.hc_ffn_fn.weight", NULL);
    const float *hc_ffn_scale = (const float *)mtp_ffn_tensor_bytes(sidecar, "mtp.0.hc_ffn_scale.weight", NULL);
    const float *hc_ffn_base = (const float *)mtp_ffn_tensor_bytes(sidecar, "mtp.0.hc_ffn_base.weight", NULL);
    const float *ffn_norm_w = (const float *)mtp_ffn_tensor_bytes(sidecar, "mtp.0.ffn_norm.weight", NULL);
    const float *router_w = (const float *)mtp_ffn_tensor_bytes(sidecar, "mtp.0.ffn_gate_inp.weight", NULL);
    const float *router_bias = (const float *)mtp_ffn_tensor_bytes(sidecar, "mtp.0.exp_probs_b.bias", NULL);
    const unsigned char *shared_gate = mtp_ffn_tensor_bytes(sidecar, "mtp.0.ffn_gate_shexp.weight", NULL);
    const unsigned char *shared_up = mtp_ffn_tensor_bytes(sidecar, "mtp.0.ffn_up_shexp.weight", NULL);
    const unsigned char *shared_down = mtp_ffn_tensor_bytes(sidecar, "mtp.0.ffn_down_shexp.weight", NULL);
    const ds4_mtp_sidecar_tensor_info *q4_gate_tensor = NULL;
    const ds4_mtp_sidecar_tensor_info *q4_up_tensor = NULL;
    const ds4_mtp_sidecar_tensor_info *q4_down_tensor = NULL;
    (void)mtp_ffn_tensor_bytes(sidecar, "mtp.0.ffn_gate_exps.weight", &q4_gate_tensor);
    (void)mtp_ffn_tensor_bytes(sidecar, "mtp.0.ffn_up_exps.weight", &q4_up_tensor);
    (void)mtp_ffn_tensor_bytes(sidecar, "mtp.0.ffn_down_exps.weight", &q4_down_tensor);
    if (!enorm || !hnorm || !e_proj || !h_proj || !hc_attn_fn ||
        !hc_attn_scale || !hc_attn_base || !attn_norm || !attn_q_a ||
        !attn_q_a_norm || !attn_q_b || !attn_kv || !attn_kv_norm ||
        !attn_sinks || !attn_output_a || !attn_output_b || !hc_ffn_fn ||
        !hc_ffn_scale || !hc_ffn_base || !ffn_norm_w || !router_w ||
        !router_bias || !shared_gate || !shared_up || !shared_down ||
        !q4_gate_tensor || !q4_up_tensor || !q4_down_tensor) {
        return 1;
    }

    const uint64_t embd_bytes = (uint64_t)MTP_ATTN_N_EMBD * sizeof(float);
    const uint64_t hc_bytes = (uint64_t)MTP_ATTN_HC_DIM * sizeof(float);
    const uint64_t mix_bytes = (uint64_t)MTP_ATTN_HC_MIX * sizeof(float);
    const uint64_t q_lora_bytes = (uint64_t)MTP_ATTN_Q_LORA * sizeof(float);
    const uint64_t heads_values = (uint64_t)MTP_ATTN_N_HEAD * MTP_ATTN_HEAD_DIM;
    const uint64_t heads_bytes = heads_values * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)MTP_ATTN_HEAD_DIM * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)MTP_ATTN_RAW_CAP * MTP_ATTN_HEAD_DIM * sizeof(float);
    const uint64_t low_bytes = (uint64_t)MTP_ATTN_OUT_LOW_DIM * sizeof(float);
    const uint64_t mid_bytes = (uint64_t)MTP_FFN_N_FF_EXP * sizeof(float);
    const uint64_t probs_bytes = (uint64_t)MTP_FFN_N_EXPERT * sizeof(float);

    float *tmp0 = (float *)malloc((size_t)hc_bytes);
    float *tmp1 = (float *)malloc((size_t)hc_bytes);
    float *row0 = (float *)malloc((size_t)embd_bytes);
    float *row1 = (float *)malloc((size_t)embd_bytes);
    float *mix = (float *)malloc((size_t)mix_bytes);
    float *split = (float *)malloc((size_t)mix_bytes);
    float *q = (float *)malloc((size_t)q_lora_bytes);
    float *q_norm = (float *)malloc((size_t)q_lora_bytes);
    float *heads = (float *)malloc((size_t)heads_bytes);
    float *attn_heads = (float *)malloc((size_t)heads_bytes);
    float *kv = (float *)malloc((size_t)kv_bytes);
    float *raw = (float *)calloc(1, (size_t)raw_bytes);
    float *low = (float *)malloc((size_t)low_bytes);
    float *ffn_router_logits = (float *)malloc((size_t)probs_bytes);
    float *ffn_router_probs = (float *)malloc((size_t)probs_bytes);
    float *routed = (float *)malloc((size_t)embd_bytes);
    float *shared_gate_out = (float *)malloc((size_t)mid_bytes);
    float *shared_up_out = (float *)malloc((size_t)mid_bytes);
    float *shared_mid = (float *)malloc((size_t)mid_bytes);
    float *shared = (float *)malloc((size_t)embd_bytes);
    int32_t selected[MTP_FFN_N_ROUTE];
    float weights[MTP_FFN_N_ROUTE];
    int rc = 1;
    if (!tmp0 || !tmp1 || !row0 || !row1 || !mix || !split || !q || !q_norm ||
        !heads || !attn_heads || !kv || !raw || !low || !ffn_router_logits || !ffn_router_probs ||
        !routed || !shared_gate_out || !shared_up_out || !shared_mid || !shared) {
        goto done;
    }

    rms_norm_weight_host(row0, embed, enorm, MTP_ATTN_N_EMBD, MTP_ATTN_RMS_EPS);
    if (matmul_q8_0_host(row1, e_proj, MTP_ATTN_N_EMBD, MTP_ATTN_N_EMBD, row0, 1) != 0) goto done;
    for (uint32_t h = 0; h < MTP_ATTN_N_HC; h++) {
        memcpy(tmp0 + (uint64_t)h * MTP_ATTN_N_EMBD, row1, (size_t)embd_bytes);
    }
    fwd_rms_norm_weight_rows_host(tmp1, prev_hc, hnorm, MTP_ATTN_N_EMBD, MTP_ATTN_N_HC, MTP_ATTN_RMS_EPS);
    if (matmul_q8_0_host(ref_prefix, h_proj, MTP_ATTN_N_EMBD, MTP_ATTN_N_EMBD, tmp1, MTP_ATTN_N_HC) != 0) goto done;
    for (uint64_t i = 0; i < MTP_ATTN_HC_DIM; i++) ref_prefix[i] += tmp0[i];

    rms_norm_plain_host(tmp0, ref_prefix, MTP_ATTN_HC_DIM, MTP_ATTN_RMS_EPS);
    matmul_f32_host(mix, hc_attn_fn, MTP_ATTN_HC_MIX, MTP_ATTN_HC_DIM, tmp0);
    hc_split_sinkhorn_host(split, mix, hc_attn_scale, hc_attn_base, MTP_ATTN_N_HC, MTP_ATTN_HC_SINKHORN_ITERS, MTP_ATTN_HC_EPS);
    hc_weighted_sum_host(row0, ref_prefix, split, MTP_ATTN_N_EMBD, MTP_ATTN_N_HC);
    rms_norm_weight_host(row1, row0, attn_norm, MTP_ATTN_N_EMBD, MTP_ATTN_RMS_EPS);
    if (matmul_q8_0_host(q, attn_q_a, MTP_ATTN_N_EMBD, MTP_ATTN_Q_LORA, row1, 1) != 0) goto done;
    rms_norm_weight_host(q_norm, q, attn_q_a_norm, MTP_ATTN_Q_LORA, MTP_ATTN_RMS_EPS);
    if (matmul_q8_0_host(heads, attn_q_b, MTP_ATTN_Q_LORA, MTP_ATTN_N_HEAD * MTP_ATTN_HEAD_DIM, q_norm, 1) != 0) goto done;
    head_rms_norm_host(heads, MTP_ATTN_N_HEAD, MTP_ATTN_HEAD_DIM, MTP_ATTN_RMS_EPS);
    if (matmul_q8_0_host(kv, attn_kv, MTP_ATTN_N_EMBD, MTP_ATTN_HEAD_DIM, row1, 1) != 0) goto done;
    rms_norm_weight_host(kv, kv, attn_kv_norm, MTP_ATTN_HEAD_DIM, MTP_ATTN_RMS_EPS);
    dsv4_fp8_kv_quantize_row_inplace_host(kv, MTP_ATTN_HEAD_DIM, MTP_ATTN_N_ROT);
    f16_round_inplace_host(kv, MTP_ATTN_HEAD_DIM);
    memcpy(raw, kv, (size_t)kv_bytes);
    attention_ref(attn_heads, heads, raw, attn_sinks, 1, 0);
    for (uint32_t g = 0; g < MTP_ATTN_OUT_GROUPS; g++) {
        if (matmul_q8_0_host(low + (uint64_t)g * MTP_ATTN_OUT_GROUP_RANK,
                             attn_output_a + (uint64_t)g * MTP_ATTN_OUT_GROUP_RANK * v->attn_output_a.row_stride_bytes,
                             MTP_ATTN_OUT_GROUP_DIM,
                             MTP_ATTN_OUT_GROUP_RANK,
                             attn_heads + (uint64_t)g * MTP_ATTN_OUT_GROUP_DIM,
                             1) != 0) goto done;
    }
    if (matmul_q8_0_host(row0, attn_output_b, MTP_ATTN_OUT_LOW_DIM, MTP_ATTN_N_EMBD, low, 1) != 0) goto done;
    hc_expand_split_host(ref_attn_next, row0, ref_prefix, split, MTP_ATTN_N_EMBD, MTP_ATTN_N_HC);

    mtp_ffn_rms_norm_plain_host(tmp0, ref_attn_next, MTP_FFN_HC_DIM, MTP_FFN_RMS_EPS);
    mtp_ffn_matmul_f32_host(mix, hc_ffn_fn, MTP_FFN_HC_MIX, MTP_FFN_HC_DIM, tmp0);
    mtp_ffn_hc_split_sinkhorn_host(split, mix, hc_ffn_scale, hc_ffn_base, MTP_FFN_N_HC, MTP_FFN_HC_SINKHORN_ITERS, MTP_FFN_HC_EPS);
    mtp_ffn_hc_weighted_sum_host(row0, ref_attn_next, split, MTP_FFN_N_EMBD, MTP_FFN_N_HC);
    mtp_ffn_rms_norm_weight_host(row1, row0, ffn_norm_w, MTP_FFN_N_EMBD, MTP_FFN_RMS_EPS);
    mtp_ffn_matmul_f32_host(ffn_router_logits, router_w, MTP_FFN_N_EXPERT, MTP_FFN_N_EMBD, row1);
    router_select_host(selected, weights, ffn_router_probs, ffn_router_logits, router_bias);
    if (q4k_reference(routed,
                              (const unsigned char *)ds4_mtp_sidecar_map(sidecar),
                              q4_gate_tensor,
                              q4_up_tensor,
                              q4_down_tensor,
                              &v->ffn_gate_exps,
                              &v->ffn_up_exps,
                              &v->ffn_down_exps,
                              selected,
                              weights,
                              row1,
                              MTP_FFN_ROUTED_SWIGLU_CLAMP) != 0) goto done;
    if (mtp_ffn_matmul_q8_0_host(shared_gate_out, shared_gate, MTP_FFN_N_EMBD, MTP_FFN_N_FF_EXP, row1, 1) != 0 ||
        mtp_ffn_matmul_q8_0_host(shared_up_out, shared_up, MTP_FFN_N_EMBD, MTP_FFN_N_FF_EXP, row1, 1) != 0) goto done;
    swiglu_host(shared_mid, shared_gate_out, shared_up_out, MTP_FFN_N_FF_EXP, 0.0f, 1.0f);
    if (mtp_ffn_matmul_q8_0_host(shared, shared_down, MTP_FFN_N_FF_EXP, MTP_FFN_N_EMBD, shared_mid, 1) != 0) goto done;
    hc_expand_add_host(ref_ffn_next, shared, routed, ref_attn_next, split, MTP_FFN_N_EMBD, MTP_FFN_N_HC);

    if (mtp_logits_cpu_mtp_logits_topk(base, sidecar, output_weight, ref_ffn_next, top_k, tokens, logits) != 0) goto done;
    rc = 0;

done:
    free(shared);
    free(shared_mid);
    free(shared_up_out);
    free(shared_gate_out);
    free(routed);
    free(ffn_router_probs);
    free(ffn_router_logits);
    free(low);
    free(raw);
    free(kv);
    free(attn_heads);
    free(heads);
    free(q_norm);
    free(q);
    free(split);
    free(mix);
    free(row1);
    free(row0);
    free(tmp1);
    free(tmp0);
    return rc;
}

#ifndef DS4_MTP_FORWARD_NO_MAIN
int main(int argc, char **argv) {
    fwd_options opt = fwd_parse_options(argc, argv);
    FILE *report = stdout;
    if (opt.report_path) {
        report = fopen(opt.report_path, "w");
        if (!report) {
            fprintf(stderr,
                    "ds4-v100-mtp-forward-smoke: cannot open report %s: %s\n",
                    opt.report_path,
                    strerror(errno));
            return 1;
        }
    }

    int rc = 1;
    char err[512] = {0};
    mtp_logits_model_map base_map;
    memset(&base_map, 0, sizeof(base_map));
    base_map.fd = -1;
    ds4_context *ctx = NULL;
    ds4_mtp_sidecar *sidecar = NULL;
    ds4_gpu_arena *output_arena = NULL;
    ds4_gpu_tensor *embed_t = NULL;
    ds4_gpu_tensor *prev_hc_t = NULL;
    ds4_gpu_tensor *prefix_t = NULL;
    ds4_gpu_tensor *attn_next_t = NULL;
    ds4_gpu_tensor *ffn_next_t = NULL;
    float *embed = NULL;
    float *prev_hc = NULL;
    float *ref_prefix = NULL;
    float *ref_attn_next = NULL;
    float *ref_ffn_next = NULL;
    float *got_prefix = NULL;
    float *got_attn_next = NULL;
    float *got_ffn_next = NULL;
    float *gpu_all_logits = NULL;
    uint32_t cpu_tokens[MTP_FORWARD_MAX_TOPK];
    uint32_t gpu_tokens[MTP_FORWARD_MAX_TOPK];
    float cpu_logits[MTP_FORWARD_MAX_TOPK];
    float gpu_logits[MTP_FORWARD_MAX_TOPK];

    fprintf(report, "model\t%s\n", opt.model);
    fprintf(report, "mtp_model\t%s\n", opt.mtp_model);
    fprintf(report, "pack_index\t%s\n", opt.pack_index);
    fprintf(report, "gpu\t%d\n", opt.gpu);
    fprintf(report, "require_gpus\t%d\n", opt.require_gpus);
    fprintf(report, "reserve_mib\t%d\n", opt.reserve_mib);
    fprintf(report, "top_k\t%" PRIu32 "\n", opt.top_k);

    if (mtp_logits_map_model_file(opt.model, &base_map) != 0) goto done;

    ds4_context_options ctx_opts;
    ds4_context_options_init(&ctx_opts);
    ctx_opts.pack_index_path = opt.pack_index;
    if (ds4_context_open(&ctx, &ctx_opts, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: %s\n", err);
        goto done;
    }

    ds4_tensor_binding output_weight;
    if (ds4_context_output_head_binding(ctx, &output_weight, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: %s\n", err);
        goto done;
    }
    ds4_gpu_bf16_matrix_view output_view;
    if (mtp_logits_output_bf16_view_from_binding(&output_weight, &output_view, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: %s\n", err);
        goto done;
    }
    if (output_weight.source_offset > base_map.size ||
        output_weight.byte_length > base_map.size - output_weight.source_offset) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: output.weight outside model map\n");
        goto done;
    }

    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: ds4_gpu_init failed\n");
        goto done;
    }
    int n_dev = ds4_gpu_device_count();
    fprintf(report, "visible_cuda_devices\t%d\n", n_dev);
    if (opt.require_gpus > 0 && n_dev < opt.require_gpus) {
        fprintf(stderr,
                "ds4-v100-mtp-forward-smoke: need %d CUDA devices, got %d\n",
                opt.require_gpus,
                n_dev);
        goto done;
    }
    if (!ds4_gpu_set_device(opt.gpu)) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: failed to set gpu%d\n", opt.gpu);
        goto done;
    }

    ds4_mtp_sidecar_options mtp_opts;
    ds4_mtp_sidecar_options_init(&mtp_opts);
    mtp_opts.mtp_path = opt.mtp_model;
    mtp_opts.gpu = opt.gpu;
    mtp_opts.require_device_arena = true;
    if (ds4_mtp_sidecar_open(&sidecar, &mtp_opts, report, err, sizeof(err)) != 0) {
        fprintf(stderr,
                "ds4-v100-mtp-forward-smoke: %s\n",
                err[0] ? err : "failed to open MTP sidecar");
        goto done;
    }

    fwd_views views;
    memset(&views, 0, sizeof(views));
    if (fwd_bind_views(sidecar, &views, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: %s\n", err);
        goto done;
    }

    if (ds4_gpu_arena_open(&output_arena, opt.gpu, output_weight.byte_length) != 0 ||
        !ds4_gpu_arena_is_device_memory(output_arena) ||
        mtp_logits_arena_upload_chunks(output_arena,
                                       0,
                                       base_map.ptr + output_weight.source_offset,
                                       output_weight.byte_length) != 0) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: output.weight upload failed\n");
        goto done;
    }
    const uint64_t reserve_bytes = (uint64_t)opt.reserve_mib * 1024ull * 1024ull;
    const uint64_t free_after = ds4_gpu_arena_free_after_upload_bytes(output_arena);
    fprintf(report, "output_weight_bytes\t%" PRIu64 "\n", output_weight.byte_length);
    fprintf(report, "output_vocab\t%" PRIu32 "\n", output_view.rows);
    fprintf(report, "reserve_bytes\t%" PRIu64 "\n", reserve_bytes);
    fprintf(report, "free_after_output_upload_bytes\t%" PRIu64 "\n", free_after);
    if (free_after < reserve_bytes) {
        fprintf(stderr,
                "ds4-v100-mtp-forward-smoke: free_after_output_upload %" PRIu64
                " below reserve %" PRIu64 "\n",
                free_after,
                reserve_bytes);
        goto done;
    }

    const uint64_t embd_bytes = (uint64_t)MTP_ATTN_N_EMBD * sizeof(float);
    const uint64_t hc_bytes = (uint64_t)MTP_ATTN_HC_DIM * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)output_view.rows * sizeof(float);
    embed = (float *)malloc((size_t)embd_bytes);
    prev_hc = (float *)malloc((size_t)hc_bytes);
    ref_prefix = (float *)malloc((size_t)hc_bytes);
    ref_attn_next = (float *)malloc((size_t)hc_bytes);
    ref_ffn_next = (float *)malloc((size_t)hc_bytes);
    got_prefix = (float *)malloc((size_t)hc_bytes);
    got_attn_next = (float *)malloc((size_t)hc_bytes);
    got_ffn_next = (float *)malloc((size_t)hc_bytes);
    gpu_all_logits = (float *)malloc((size_t)logits_bytes);
    if (!embed || !prev_hc || !ref_prefix || !ref_attn_next || !ref_ffn_next ||
        !got_prefix || !got_attn_next || !got_ffn_next || !gpu_all_logits) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: host allocation failed\n");
        goto done;
    }
    fwd_fill_embed(embed);
    fill_hc_state(prev_hc);

    if (fwd_cpu_reference(&base_map,
                          sidecar,
                          &output_weight,
                          &views,
                          embed,
                          prev_hc,
                          ref_prefix,
                          ref_attn_next,
                          ref_ffn_next,
                          opt.top_k,
                          cpu_tokens,
                          cpu_logits) != 0) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: CPU forward reference failed\n");
        goto done;
    }

    const uint64_t mix_bytes = (uint64_t)MTP_ATTN_HC_MIX * sizeof(float);
    const uint64_t q_lora_bytes = (uint64_t)MTP_ATTN_Q_LORA * sizeof(float);
    const uint64_t heads_bytes = (uint64_t)MTP_ATTN_N_HEAD * MTP_ATTN_HEAD_DIM * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)MTP_ATTN_HEAD_DIM * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)MTP_ATTN_RAW_CAP * MTP_ATTN_HEAD_DIM * sizeof(float);
    const uint64_t low_bytes = (uint64_t)MTP_ATTN_OUT_LOW_DIM * sizeof(float);
    const uint64_t mid_bytes = (uint64_t)MTP_FFN_N_FF_EXP * sizeof(float);
    const uint64_t route_i32_bytes = (uint64_t)MTP_FFN_N_ROUTE * sizeof(int32_t);
    const uint64_t route_f32_bytes = (uint64_t)MTP_FFN_N_ROUTE * sizeof(float);
    const uint64_t probs_bytes = (uint64_t)MTP_FFN_N_EXPERT * sizeof(float);
    const uint64_t q4_mid_values = (uint64_t)MTP_FFN_N_ROUTE * MTP_FFN_N_FF_EXP;
    const uint64_t q4_down_values = (uint64_t)MTP_FFN_N_ROUTE * MTP_FFN_N_EMBD;

    embed_t = ds4_gpu_tensor_alloc(embd_bytes);
    prev_hc_t = ds4_gpu_tensor_alloc(hc_bytes);
    prefix_t = ds4_gpu_tensor_alloc(hc_bytes);
    attn_next_t = ds4_gpu_tensor_alloc(hc_bytes);
    ffn_next_t = ds4_gpu_tensor_alloc(hc_bytes);
    ds4_gpu_tensor *row0 = ds4_gpu_tensor_alloc(embd_bytes);
    ds4_gpu_tensor *row1 = ds4_gpu_tensor_alloc(embd_bytes);
    ds4_gpu_tensor *hc0 = ds4_gpu_tensor_alloc(hc_bytes);
    ds4_gpu_tensor *hc1 = ds4_gpu_tensor_alloc(hc_bytes);
    ds4_gpu_tensor *mix_t = ds4_gpu_tensor_alloc(mix_bytes);
    ds4_gpu_tensor *split_t = ds4_gpu_tensor_alloc(mix_bytes);
    ds4_gpu_tensor *q_t = ds4_gpu_tensor_alloc(q_lora_bytes);
    ds4_gpu_tensor *q_norm_t = ds4_gpu_tensor_alloc(q_lora_bytes);
    ds4_gpu_tensor *heads_t = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *attn_heads_t = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *kv_t = ds4_gpu_tensor_alloc(kv_bytes);
    ds4_gpu_tensor *raw_t = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *low_t = ds4_gpu_tensor_alloc(low_bytes);
    ds4_gpu_tensor *router_logits_t = ds4_gpu_tensor_alloc(probs_bytes);
    ds4_gpu_tensor *router_probs_t = ds4_gpu_tensor_alloc(probs_bytes);
    ds4_gpu_tensor *selected_t = ds4_gpu_tensor_alloc(route_i32_bytes);
    ds4_gpu_tensor *weights_t = ds4_gpu_tensor_alloc(route_f32_bytes);
    ds4_gpu_tensor *routed_t = ds4_gpu_tensor_alloc(embd_bytes);
    ds4_gpu_tensor *q4_gate_tmp_t = ds4_gpu_tensor_alloc(q4_mid_values * sizeof(float));
    ds4_gpu_tensor *q4_up_tmp_t = ds4_gpu_tensor_alloc(q4_mid_values * sizeof(float));
    ds4_gpu_tensor *q4_mid_tmp_t = ds4_gpu_tensor_alloc(q4_mid_values * sizeof(float));
    ds4_gpu_tensor *q4_down_tmp_t = ds4_gpu_tensor_alloc(q4_down_values * sizeof(float));
    ds4_gpu_tensor *shared_gate_t = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *shared_up_t = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *shared_mid_t = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *shared_t = ds4_gpu_tensor_alloc(embd_bytes);
    ds4_gpu_tensor *ffn_t = ds4_gpu_tensor_alloc(embd_bytes);
    ds4_gpu_tensor *head_pre_t = ds4_gpu_tensor_alloc((uint64_t)MTP_ATTN_N_HC * sizeof(float));
    ds4_gpu_tensor *head_weights_t = ds4_gpu_tensor_alloc((uint64_t)MTP_ATTN_N_HC * sizeof(float));
    ds4_gpu_tensor *logits_t = ds4_gpu_tensor_alloc(logits_bytes);

    if (!embed_t || !prev_hc_t || !prefix_t || !attn_next_t || !ffn_next_t ||
        !row0 || !row1 || !hc0 || !hc1 || !mix_t || !split_t || !q_t ||
        !q_norm_t || !heads_t || !attn_heads_t ||
        !kv_t || !raw_t || !low_t || !router_logits_t || !router_probs_t ||
        !selected_t || !weights_t || !routed_t || !q4_gate_tmp_t ||
        !q4_up_tmp_t || !q4_mid_tmp_t || !q4_down_tmp_t || !shared_gate_t ||
        !shared_up_t || !shared_mid_t || !shared_t || !ffn_t || !head_pre_t ||
        !head_weights_t || !logits_t) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: device allocation failed\n");
        goto done_tensors;
    }

    ds4_gpu_arena *arena = ds4_mtp_sidecar_arena(sidecar);
    const double t0 = now_ms();
    if (!ds4_gpu_tensor_write(embed_t, 0, embed, embd_bytes) ||
        !ds4_gpu_tensor_write(prev_hc_t, 0, prev_hc, hc_bytes) ||
        !ds4_gpu_tensor_fill_f32(raw_t, 0.0f, MTP_ATTN_RAW_CAP * MTP_ATTN_HEAD_DIM) ||
        ds4_gpu_arena_f32_rms_norm_f32(arena, &views.enorm, embed_t, row0, MTP_ATTN_N_EMBD, 1, MTP_ATTN_RMS_EPS) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &views.e_proj, row0, row1, 1) != 0 ||
        !ds4_gpu_repeat_hc_tensor(hc0, row1, MTP_ATTN_N_EMBD, MTP_ATTN_N_HC) ||
        ds4_gpu_arena_f32_rms_norm_f32(arena, &views.hnorm, prev_hc_t, hc1, MTP_ATTN_N_EMBD, MTP_ATTN_N_HC, MTP_ATTN_RMS_EPS) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &views.h_proj, hc1, prefix_t, MTP_ATTN_N_HC) != 0 ||
        !ds4_gpu_add_tensor(prefix_t, hc0, prefix_t, MTP_ATTN_HC_DIM) ||
        !ds4_gpu_rms_norm_plain_tensor(hc0, prefix_t, MTP_ATTN_HC_DIM, MTP_ATTN_RMS_EPS) ||
        ds4_gpu_arena_f32_matmul_f32(arena, &views.hc_attn_fn, hc0, mix_t) != 0 ||
        ds4_gpu_arena_hc_split_weighted_sum_tensor(arena, &views.hc_attn_scale, &views.hc_attn_base, row0, split_t, mix_t, prefix_t, MTP_ATTN_N_EMBD, MTP_ATTN_N_HC, MTP_ATTN_HC_SINKHORN_ITERS, MTP_ATTN_HC_EPS) != 0 ||
        ds4_gpu_arena_f32_rms_norm_f32(arena, &views.attn_norm, row0, row1, MTP_ATTN_N_EMBD, 1, MTP_ATTN_RMS_EPS) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &views.attn_q_a, row1, q_t, 1) != 0 ||
        ds4_gpu_arena_f32_rms_norm_f32(arena, &views.attn_q_a_norm, q_t, q_norm_t, MTP_ATTN_Q_LORA, 1, MTP_ATTN_RMS_EPS) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &views.attn_q_b, q_norm_t, heads_t, 1) != 0 ||
        !ds4_gpu_head_rms_norm_tensor(heads_t, 1, MTP_ATTN_N_HEAD, MTP_ATTN_HEAD_DIM, MTP_ATTN_RMS_EPS) ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &views.attn_kv, row1, kv_t, 1) != 0 ||
        ds4_gpu_arena_f32_rms_norm_f32(arena, &views.attn_kv_norm, kv_t, kv_t, MTP_ATTN_HEAD_DIM, 1, MTP_ATTN_RMS_EPS) != 0 ||
        !ds4_gpu_kv_fp8_store_raw_tensor(kv_t, raw_t, MTP_ATTN_RAW_CAP, 0, MTP_ATTN_HEAD_DIM, MTP_ATTN_N_ROT) ||
        ds4_gpu_arena_attention_decode_heads_tensor(arena, &views.attn_sinks, attn_heads_t, heads_t, raw_t, 1, MTP_ATTN_RAW_CAP, 0, NULL, 0, NULL, 0, MTP_ATTN_N_HEAD, MTP_ATTN_HEAD_DIM) != 0 ||
        grouped_output_arena(sidecar, &views.attn_output_a, &views.attn_output_b, attn_heads_t, low_t, row0) != 0 ||
        !ds4_gpu_hc_expand_split_tensor(attn_next_t, row0, prefix_t, split_t, MTP_ATTN_N_EMBD, MTP_ATTN_N_HC) ||
        !ds4_gpu_rms_norm_plain_tensor(hc0, attn_next_t, MTP_FFN_HC_DIM, MTP_FFN_RMS_EPS) ||
        ds4_gpu_arena_f32_matmul_f32(arena, &views.hc_ffn_fn, hc0, mix_t) != 0 ||
        ds4_gpu_arena_hc_split_weighted_sum_tensor(arena, &views.hc_ffn_scale, &views.hc_ffn_base, row0, split_t, mix_t, attn_next_t, MTP_FFN_N_EMBD, MTP_FFN_N_HC, MTP_FFN_HC_SINKHORN_ITERS, MTP_FFN_HC_EPS) != 0 ||
        ds4_gpu_arena_f32_rms_norm_f32(arena, &views.ffn_norm, row0, row1, MTP_FFN_N_EMBD, 1, MTP_FFN_RMS_EPS) != 0 ||
        ds4_gpu_arena_f32_matmul_f32(arena, &views.ffn_gate_inp, row1, router_logits_t) != 0 ||
        ds4_gpu_arena_router_select_bias_tensor(arena, &views.exp_probs_b, selected_t, weights_t, router_probs_t, router_logits_t) != 0 ||
        ds4_gpu_arena_q4_k_routed_moe_one_f32(arena, &views.ffn_gate_exps, &views.ffn_up_exps, &views.ffn_down_exps, routed_t, q4_gate_tmp_t, q4_up_tmp_t, q4_mid_tmp_t, q4_down_tmp_t, selected_t, weights_t, row1, MTP_FFN_N_ROUTE, MTP_FFN_ROUTED_SWIGLU_CLAMP) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &views.ffn_gate_shexp, row1, shared_gate_t, 1) != 0 ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &views.ffn_up_shexp, row1, shared_up_t, 1) != 0 ||
        !ds4_gpu_swiglu_tensor(shared_mid_t, shared_gate_t, shared_up_t, MTP_FFN_N_FF_EXP, 0.0f, 1.0f) ||
        ds4_gpu_arena_q8_0_matmul_f32(arena, &views.ffn_down_shexp, shared_mid_t, shared_t, 1) != 0 ||
        !ds4_gpu_add_tensor(ffn_t, shared_t, routed_t, MTP_FFN_N_EMBD) ||
        !ds4_gpu_hc_expand_add_split_tensor(ffn_next_t, shared_t, routed_t, attn_next_t, split_t, MTP_FFN_N_EMBD, MTP_FFN_N_HC) ||
        !ds4_gpu_rms_norm_plain_tensor(hc0, ffn_next_t, MTP_ATTN_HC_DIM, MTP_ATTN_RMS_EPS) ||
        ds4_gpu_arena_f32_matmul_f32(arena, &views.hc_head_fn, hc0, head_pre_t) != 0 ||
        ds4_gpu_arena_output_hc_weights_tensor(arena, &views.hc_head_scale, &views.hc_head_base, head_weights_t, head_pre_t, MTP_ATTN_N_HC, MTP_ATTN_HC_EPS) != 0 ||
        !ds4_gpu_hc_weighted_sum_tensor(row0, ffn_next_t, head_weights_t, MTP_ATTN_N_EMBD, MTP_ATTN_N_HC) ||
        ds4_gpu_arena_f32_rms_norm_f32(arena, &views.output_norm, row0, row1, MTP_ATTN_N_EMBD, 1, MTP_ATTN_RMS_EPS) != 0 ||
        ds4_gpu_arena_bf16_matmul_f32(output_arena, &output_view, row1, logits_t) != 0 ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(prefix_t, 0, got_prefix, hc_bytes) ||
        !ds4_gpu_tensor_read(attn_next_t, 0, got_attn_next, hc_bytes) ||
        !ds4_gpu_tensor_read(ffn_next_t, 0, got_ffn_next, hc_bytes) ||
        !ds4_gpu_tensor_read(logits_t, 0, gpu_all_logits, logits_bytes)) {
        fprintf(stderr, "ds4-v100-mtp-forward-smoke: GPU forward sequence failed\n");
        goto done_tensors;
    }
    const double t1 = now_ms();

    fwd_topk_from_logits(gpu_all_logits, output_view.rows, opt.top_k, gpu_tokens, gpu_logits);

    int failures = 0;
    double max_abs = 0.0;
    if (fwd_compare("prefix_hc", got_prefix, ref_prefix, MTP_ATTN_HC_DIM, opt.prefix_tol, report, &max_abs) != 0) failures++;
    double global_max_abs = max_abs;
    if (fwd_compare("attn_next_hc", got_attn_next, ref_attn_next, MTP_ATTN_HC_DIM, opt.attn_tol, report, &max_abs) != 0) failures++;
    if (max_abs > global_max_abs) global_max_abs = max_abs;
    if (fwd_compare("ffn_next_hc", got_ffn_next, ref_ffn_next, MTP_ATTN_HC_DIM, opt.ffn_tol, report, &max_abs) != 0) failures++;
    if (max_abs > global_max_abs) global_max_abs = max_abs;

    double logit_max_abs = 0.0;
    for (uint32_t i = 0; i < opt.top_k; i++) {
        double delta = fabs((double)cpu_logits[i] - (double)gpu_logits[i]);
        if (delta > logit_max_abs) logit_max_abs = delta;
        fprintf(report,
                "mtp_forward_topk\trank=%" PRIu32
                "\tcpu_token=%" PRIu32 "\tgpu_token=%" PRIu32
                "\tcpu_logit=%.9g\tgpu_logit=%.9g\tdelta=%.9g\n",
                i + 1,
                cpu_tokens[i],
                gpu_tokens[i],
                cpu_logits[i],
                gpu_logits[i],
                delta);
        if (cpu_tokens[i] != gpu_tokens[i] || delta > opt.logit_tol) failures++;
    }

    fprintf(report,
            "mtp_forward_summary\tgpu_ms=%.3f\ttop1_cpu=%" PRIu32
            "\ttop1_gpu=%" PRIu32 "\tboundary_max_abs=%.9g\tlogit_max_abs=%.9g\t%s\n",
            t1 - t0,
            cpu_tokens[0],
            gpu_tokens[0],
            global_max_abs,
            logit_max_abs,
            failures ? "FAIL" : "PASS");
    printf("mtp_forward_smoke: cpu_top1=%" PRIu32
           " gpu_top1=%" PRIu32 " boundary_max_abs=%.9g logit_max_abs=%.9g %s\n",
           cpu_tokens[0],
           gpu_tokens[0],
           global_max_abs,
           logit_max_abs,
           failures ? "FAIL" : "PASS");
    if (failures == 0) rc = 0;

done_tensors:
    ds4_gpu_tensor_free(logits_t);
    ds4_gpu_tensor_free(head_weights_t);
    ds4_gpu_tensor_free(head_pre_t);
    ds4_gpu_tensor_free(ffn_t);
    ds4_gpu_tensor_free(shared_t);
    ds4_gpu_tensor_free(shared_mid_t);
    ds4_gpu_tensor_free(shared_up_t);
    ds4_gpu_tensor_free(shared_gate_t);
    ds4_gpu_tensor_free(q4_down_tmp_t);
    ds4_gpu_tensor_free(q4_mid_tmp_t);
    ds4_gpu_tensor_free(q4_up_tmp_t);
    ds4_gpu_tensor_free(q4_gate_tmp_t);
    ds4_gpu_tensor_free(routed_t);
    ds4_gpu_tensor_free(weights_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(router_probs_t);
    ds4_gpu_tensor_free(router_logits_t);
    ds4_gpu_tensor_free(low_t);
    ds4_gpu_tensor_free(raw_t);
    ds4_gpu_tensor_free(kv_t);
    ds4_gpu_tensor_free(attn_heads_t);
    ds4_gpu_tensor_free(heads_t);
    ds4_gpu_tensor_free(q_norm_t);
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(split_t);
    ds4_gpu_tensor_free(mix_t);
    ds4_gpu_tensor_free(hc1);
    ds4_gpu_tensor_free(hc0);
    ds4_gpu_tensor_free(row1);
    ds4_gpu_tensor_free(row0);
done:
    ds4_gpu_tensor_free(ffn_next_t);
    ds4_gpu_tensor_free(attn_next_t);
    ds4_gpu_tensor_free(prefix_t);
    ds4_gpu_tensor_free(prev_hc_t);
    ds4_gpu_tensor_free(embed_t);
    ds4_gpu_arena_close(output_arena);
    ds4_mtp_sidecar_close(sidecar);
    ds4_context_close(ctx);
    mtp_logits_unmap_model_file(&base_map);
    free(gpu_all_logits);
    free(got_ffn_next);
    free(got_attn_next);
    free(got_prefix);
    free(ref_ffn_next);
    free(ref_attn_next);
    free(ref_prefix);
    free(prev_hc);
    free(embed);
    ds4_gpu_cleanup();
    if (report && report != stdout) fclose(report);
    return rc;
}
#endif
