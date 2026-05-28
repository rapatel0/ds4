#include "ds4.h"
#include "engine/scheduler.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    MAX_CHECKPOINTS = 128,
    ROUTES_PER_TOKEN = 6,
};

typedef struct {
    const unsigned char *ptr;
    uint64_t size;
    int fd;
} model_map;

typedef struct {
    const int *layers;
    int n_layers;
    float *gpu_hc;
    ds4_layer_execute_report *gpu_reports;
    uint8_t seen[MAX_CHECKPOINTS];
} checkpoint_capture;

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_v100_scheduler_checkpoint_parity_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/cuda_v100_scheduler_checkpoint_parity_smoke "
            "--index FILE --model FILE [--prompt-file FILE] [--layers CSV] "
            "[--ctx N] [--prompt-tokens N] [--max-abs X] [--max-rms X]\n");
    fprintf(fp, "       layer specs: -1 for seed, N for layer final, aN for after-attention\n");
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr,
                "cuda_v100_scheduler_checkpoint_parity_smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *buf = (char *)malloc((size_t)len + 1u);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)len, fp);
    fclose(fp);
    if (got != (size_t)len) {
        free(buf);
        return NULL;
    }
    buf[len] = '\0';
    return buf;
}

static int map_model_file(const char *path, model_map *out) {
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr,
                "cuda_v100_scheduler_checkpoint_parity_smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr,
                "cuda_v100_scheduler_checkpoint_parity_smoke: cannot stat %s\n",
                path);
        close(fd);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr,
                "cuda_v100_scheduler_checkpoint_parity_smoke: cannot mmap %s: %s\n",
                path,
                strerror(errno));
        close(fd);
        return 1;
    }
    out->ptr = (const unsigned char *)p;
    out->size = (uint64_t)st.st_size;
    out->fd = fd;
    return 0;
}

static void unmap_model_file(model_map *m) {
    if (!m) return;
    if (m->ptr) munmap((void *)m->ptr, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

static int parse_int_arg(const char *s, const char *name, int min_v, int max_v) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || !*s || !end || *end != '\0' || v < min_v || v > max_v) {
        fprintf(stderr,
                "cuda_v100_scheduler_checkpoint_parity_smoke: invalid %s: %s\n",
                name,
                s ? s : "(null)");
        exit(2);
    }
    return (int)v;
}

static float parse_float_arg(const char *s, const char *name) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (!s || !*s || !end || *end != '\0' || !isfinite(v) || v < 0.0f) {
        fprintf(stderr,
                "cuda_v100_scheduler_checkpoint_parity_smoke: invalid %s: %s\n",
                name,
                s ? s : "(null)");
        exit(2);
    }
    return v;
}

static int checkpoint_actual_layer(int spec) {
    if (spec >= DS4_HC_CHECKPOINT_AFTER_ATTN_BASE) {
        return spec - DS4_HC_CHECKPOINT_AFTER_ATTN_BASE;
    }
    return spec;
}

static int checkpoint_kind(int spec) {
    if (spec == -1) return DS4_V100_HC_CHECKPOINT_SEED;
    if (spec >= DS4_HC_CHECKPOINT_AFTER_ATTN_BASE) return DS4_V100_HC_CHECKPOINT_AFTER_ATTN;
    return DS4_V100_HC_CHECKPOINT_LAYER_FINAL;
}

static const char *checkpoint_kind_name(int kind) {
    switch (kind) {
    case DS4_V100_HC_CHECKPOINT_SEED: return "seed";
    case DS4_V100_HC_CHECKPOINT_AFTER_ATTN: return "after_attn";
    case DS4_V100_HC_CHECKPOINT_LAYER_FINAL: return "layer_final";
    default: return "unknown";
    }
}

static int parse_layer_spec(const char *tok) {
    if (tok && tok[0] == 'a') {
        int layer = parse_int_arg(tok + 1, "--layers", 0, DS4_V100_N_LAYERS - 1);
        return DS4_HC_CHECKPOINT_AFTER_ATTN_BASE + layer;
    }
    int spec = parse_int_arg(tok, "--layers", -1, DS4_HC_CHECKPOINT_AFTER_ATTN_BASE + DS4_V100_N_LAYERS - 1);
    int layer = checkpoint_actual_layer(spec);
    if (layer < -1 || layer >= DS4_V100_N_LAYERS ||
        (spec >= DS4_V100_N_LAYERS && spec < DS4_HC_CHECKPOINT_AFTER_ATTN_BASE)) {
        fprintf(stderr, "cuda_v100_scheduler_checkpoint_parity_smoke: invalid --layers: %s\n", tok);
        exit(2);
    }
    return spec;
}

static int parse_layers(const char *csv, int *layers, int cap) {
    char *copy = strdup(csv);
    if (!copy) return -1;
    int n = 0;
    char *save = NULL;
    for (char *tok = strtok_r(copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        if (n >= cap) {
            free(copy);
            return -1;
        }
        layers[n++] = parse_layer_spec(tok);
    }
    free(copy);
    return n;
}

static void stats_diff(const float *a,
                       const float *b,
                       uint64_t n,
                       float *max_abs,
                       float *rms_abs,
                       float *a_rms,
                       float *b_rms,
                       int *nonfinite) {
    double ss = 0.0;
    double ass = 0.0;
    double bss = 0.0;
    float max_d = 0.0f;
    int bad = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) bad = 1;
        const double d = (double)a[i] - (double)b[i];
        const float ad = fabsf((float)d);
        if (ad > max_d) max_d = ad;
        ss += d * d;
        ass += (double)a[i] * (double)a[i];
        bss += (double)b[i] * (double)b[i];
    }
    *max_abs = max_d;
    *rms_abs = n ? (float)sqrt(ss / (double)n) : 0.0f;
    *a_rms = n ? (float)sqrt(ass / (double)n) : 0.0f;
    *b_rms = n ? (float)sqrt(bss / (double)n) : 0.0f;
    *nonfinite = bad;
}

static int capture_v100_checkpoint(const ds4_stage_scheduler_checkpoint *cp,
                                   void *user,
                                   char *err,
                                   size_t errlen) {
    checkpoint_capture *cap = (checkpoint_capture *)user;
    if (!cp || !cap || !cp->hc) {
        if (err && errlen) snprintf(err, errlen, "missing checkpoint capture input");
        return 1;
    }
    const uint64_t hc_bytes = (uint64_t)DS4_HC_CHECKPOINT_VALUES * sizeof(float);
    if (cp->hc_bytes != hc_bytes) {
        if (err && errlen) snprintf(err, errlen, "unexpected V100 checkpoint HC size");
        return 1;
    }
    for (int i = 0; i < cap->n_layers; i++) {
        if (checkpoint_actual_layer(cap->layers[i]) != cp->layer ||
            checkpoint_kind(cap->layers[i]) != cp->kind) {
            continue;
        }
        if (!ds4_gpu_set_device(cp->gpu) ||
            !ds4_gpu_tensor_read(cp->hc,
                                 0,
                                 cap->gpu_hc + (uint64_t)i * DS4_HC_CHECKPOINT_VALUES,
                                 hc_bytes)) {
            if (err && errlen) snprintf(err, errlen, "V100 checkpoint read failed at layer %d", cp->layer);
            return 1;
        }
        cap->seen[i] = 1;
        cap->gpu_reports[i] = cp->layer_report;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *index = NULL;
    const char *model_path = NULL;
    const char *prompt_file = "tests/test-vectors/prompts/short_reasoning_plain.txt";
    const char *layers_csv = "-1,0,1,2,3,a4";
    int ctx_size = 4096;
    int prompt_token_limit = 0;
    float max_abs_limit = 1.0e-2f;
    float max_rms_limit = 1.0e-3f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--index") && i + 1 < argc) {
            index = argv[++i];
        } else if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) {
            prompt_file = argv[++i];
        } else if (!strcmp(argv[i], "--layers") && i + 1 < argc) {
            layers_csv = argv[++i];
        } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            ctx_size = parse_int_arg(argv[++i], "--ctx", 1, 2000000);
        } else if (!strcmp(argv[i], "--prompt-tokens") && i + 1 < argc) {
            prompt_token_limit = parse_int_arg(argv[++i], "--prompt-tokens", 1, 2000000);
        } else if (!strcmp(argv[i], "--max-abs") && i + 1 < argc) {
            max_abs_limit = parse_float_arg(argv[++i], "--max-abs");
        } else if (!strcmp(argv[i], "--max-rms") && i + 1 < argc) {
            max_rms_limit = parse_float_arg(argv[++i], "--max-rms");
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(stdout);
            return 0;
        } else {
            usage(stderr);
            return 2;
        }
    }
    if (!index || !model_path) {
        usage(stderr);
        return 2;
    }

    int layers[MAX_CHECKPOINTS];
    int n_layers = parse_layers(layers_csv, layers, MAX_CHECKPOINTS);
    if (n_layers <= 0) {
        fprintf(stderr, "cuda_v100_scheduler_checkpoint_parity_smoke: invalid --layers\n");
        return 2;
    }
    int required_stages = 0;
    for (int i = 0; i < n_layers; i++) {
        const int layer = checkpoint_actual_layer(layers[i]);
        const int stage = layer < 0 ? 0 : ds4_stage_for_layer(layer);
        if (stage + 1 > required_stages) required_stages = stage + 1;
    }
    if (required_stages <= 0 || required_stages > DS4_V100_EXPECTED_GPUS) {
        fprintf(stderr, "cuda_v100_scheduler_checkpoint_parity_smoke: invalid required stage count\n");
        return 2;
    }

    ds4_engine_options eopts;
    memset(&eopts, 0, sizeof(eopts));
    eopts.model_path = model_path;
    eopts.backend = DS4_BACKEND_CPU;
    eopts.n_threads = 1;
    eopts.source_layout_oracle = true;
    eopts.source_layout_oracle_sessions = true;
    ds4_engine *cpu_engine = NULL;
    if (ds4_engine_open(&cpu_engine, &eopts) != 0) {
        fprintf(stderr, "cuda_v100_scheduler_checkpoint_parity_smoke: CPU oracle engine open failed\n");
        return 1;
    }

    char *prompt_text = read_file(prompt_file);
    if (!prompt_text) {
        ds4_engine_close(cpu_engine);
        return 1;
    }
    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(cpu_engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    free(prompt_text);
    check(prompt.len > 0, "empty prompt tokenization");
    const int full_prompt_len = prompt.len;
    if (prompt_token_limit > 0 && prompt_token_limit < prompt.len) {
        prompt.len = prompt_token_limit;
    }
    if (prompt.len > ctx_size) {
        fprintf(stderr,
                "cuda_v100_scheduler_checkpoint_parity_smoke: prompt tokens %d exceed ctx %d\n",
                prompt.len,
                ctx_size);
        failures++;
    }

    const uint64_t hc_values = DS4_HC_CHECKPOINT_VALUES;
    float *cpu_hc = (float *)calloc((size_t)n_layers * (size_t)hc_values, sizeof(float));
    float *gpu_hc = (float *)calloc((size_t)n_layers * (size_t)hc_values, sizeof(float));
    int32_t *cpu_selected = (int32_t *)calloc((size_t)n_layers * ROUTES_PER_TOKEN, sizeof(int32_t));
    float *cpu_weights = (float *)calloc((size_t)n_layers * ROUTES_PER_TOKEN, sizeof(float));
    ds4_layer_execute_report *gpu_reports =
        (ds4_layer_execute_report *)calloc((size_t)n_layers, sizeof(*gpu_reports));
    check(cpu_hc && gpu_hc && cpu_selected && cpu_weights && gpu_reports, "checkpoint allocation");
    checkpoint_capture capture = {
        .layers = layers,
        .n_layers = n_layers,
        .gpu_hc = gpu_hc,
        .gpu_reports = gpu_reports,
    };

    char err[512] = {0};
    if (failures == 0) {
        check(ds4_engine_cpu_hc_checkpoints(cpu_engine,
                                            &prompt,
                                            layers,
                                            n_layers,
                                            ctx_size,
                                            cpu_hc,
                                            (uint64_t)n_layers * hc_values,
                                            err,
                                            sizeof(err)) == 0,
              err[0] ? err : "CPU HC checkpoints");
        err[0] = '\0';
        check(ds4_engine_cpu_route_checkpoints(cpu_engine,
                                               &prompt,
                                               layers,
                                               n_layers,
                                               ctx_size,
                                               cpu_selected,
                                               cpu_weights,
                                               (uint64_t)n_layers * ROUTES_PER_TOKEN,
                                               err,
                                               sizeof(err)) == 0,
              err[0] ? err : "CPU route checkpoints");
    }

    int devices = ds4_gpu_device_count();
    if (devices < required_stages) {
        fprintf(stderr,
                "cuda_v100_scheduler_checkpoint_parity_smoke: need %d CUDA devices, got %d\n",
                required_stages,
                devices);
        failures++;
    }

    model_map model;
    memset(&model, 0, sizeof(model));
    model.fd = -1;
    if (failures == 0 && map_model_file(model_path, &model)) failures++;
    if (failures == 0) check(ds4_gpu_set_model_fd(model.fd), "model fd");

    ds4_stage_scheduler *scheds[DS4_V100_EXPECTED_GPUS];
    memset(scheds, 0, sizeof(scheds));
    ds4_stage_scheduler_options opts;
    ds4_stage_scheduler_options_init(&opts);
    opts.pack_index_path = index;
    opts.model_map = model.ptr;
    opts.model_size = model.size;
    opts.kv_ctx_tokens = (uint64_t)ctx_size;
    uint32_t comp_cap = (uint32_t)prompt.len / 4u + 8u;
    if (comp_cap < 64u) comp_cap = 64u;
    opts.attn_comp_cap = comp_cap;
    opts.index_comp_cap = comp_cap;
    opts.indexer_top_k = 512;

    if (failures == 0) {
        for (int i = 0; i < required_stages; i++) {
            opts.stage_id = i;
            err[0] = '\0';
            if (ds4_stage_scheduler_open(&scheds[i], &opts, err, sizeof(err))) {
                fprintf(stderr,
                        "cuda_v100_scheduler_checkpoint_parity_smoke: open stage %d failed: %s\n",
                        i,
                        err[0] ? err : "scheduler open");
                failures++;
                break;
            }
        }
    }

    for (int pos = 0; pos < prompt.len && failures == 0; pos++) {
        if (prompt.v[pos] < 0) {
            check(0, "negative prompt token");
            break;
        }
        ds4_stage_scheduler_report report;
        err[0] = '\0';
        const bool final_token = pos == prompt.len - 1;
        check(ds4_stage_scheduler_decode_token_checkpoints(
                  scheds[0],
                  (uint32_t)prompt.v[pos],
                  (uint32_t)pos,
                  &report,
                  final_token ? capture_v100_checkpoint : NULL,
                  final_token ? &capture : NULL,
                  err,
                  sizeof(err)) == 0,
              err[0] ? err : "stage 0 prompt decode");
        for (int stage = 1; stage < required_stages && failures == 0; stage++) {
            err[0] = '\0';
            check(ds4_stage_scheduler_handoff(scheds[stage],
                                                   scheds[stage - 1],
                                                   err,
                                                   sizeof(err)) == 0,
                  err[0] ? err : "stage handoff");
            err[0] = '\0';
            check(ds4_stage_scheduler_decode_hc_checkpoints(
                      scheds[stage],
                      (uint32_t)prompt.v[pos],
                      (uint32_t)pos,
                      &report,
                      final_token ? capture_v100_checkpoint : NULL,
                      final_token ? &capture : NULL,
                      err,
                      sizeof(err)) == 0,
                  err[0] ? err : "stage prompt decode");
        }
    }

    if (failures == 0) {
        for (int i = 0; i < n_layers; i++) {
            if (!capture.seen[i]) {
                fprintf(stderr,
                        "cuda_v100_scheduler_checkpoint_parity_smoke: missing V100 checkpoint for layer %d\n",
                        layers[i]);
                failures++;
            }
        }
    }

    int first_bad = 0;
    int have_first_bad = 0;
    if (failures == 0) {
        for (int i = 0; i < n_layers; i++) {
            float max_abs = 0.0f;
            float rms_abs = 0.0f;
            float cpu_rms = 0.0f;
            float gpu_rms = 0.0f;
            int nonfinite = 0;
            stats_diff(cpu_hc + (uint64_t)i * hc_values,
                       gpu_hc + (uint64_t)i * hc_values,
                       hc_values,
                       &max_abs,
                       &rms_abs,
                       &cpu_rms,
                       &gpu_rms,
                       &nonfinite);
            const int layer = checkpoint_actual_layer(layers[i]);
            const int kind = checkpoint_kind(layers[i]);
            bool route_bad = false;
            if (layer >= 0 && kind == DS4_V100_HC_CHECKPOINT_LAYER_FINAL) {
                ds4_layer_execute_report *gr = &gpu_reports[i];
                if (gr->routes != ROUTES_PER_TOKEN) {
                    route_bad = true;
                }
                for (uint32_t r = 0; r < gr->routes && r < ROUTES_PER_TOKEN; r++) {
                    const float cw = cpu_weights[(uint64_t)i * ROUTES_PER_TOKEN + r];
                    const float gw = gr->route_weights[r];
                    const float wtol = 2.0e-5f + 2.0e-5f * fabsf(cw);
                    if (gr->selected_experts[r] != cpu_selected[(uint64_t)i * ROUTES_PER_TOKEN + r] ||
                        fabsf(gw - cw) > wtol) {
                        route_bad = true;
                    }
                }
            }
            const bool ok = !nonfinite &&
                            !route_bad &&
                            max_abs <= max_abs_limit &&
                            rms_abs <= max_rms_limit;
            if (!ok && !have_first_bad) {
                first_bad = layers[i];
                have_first_bad = 1;
            }
            printf("checkpoint\tlayer=%d\tkind=%s\tstage=%d\tmax_abs=%.9g\trms_abs=%.9g\tcpu_rms=%.9g\tgpu_rms=%.9g\troute0=%d/%.7g:%d/%.7g\t%s\n",
                   layer,
                   checkpoint_kind_name(kind),
                   layer < 0 ? 0 : ds4_stage_for_layer(layer),
                   max_abs,
                   rms_abs,
                   cpu_rms,
                   gpu_rms,
                   layer >= 0 ? cpu_selected[(uint64_t)i * ROUTES_PER_TOKEN] : -1,
                   layer >= 0 ? cpu_weights[(uint64_t)i * ROUTES_PER_TOKEN] : 0.0f,
                   layer >= 0 ? gpu_reports[i].selected_experts[0] : -1,
                   layer >= 0 ? gpu_reports[i].route_weights[0] : 0.0f,
                   ok ? "PASS" : "DIFF");
        }
    }
    if (have_first_bad) failures++;

    printf("cuda_v100_scheduler_checkpoint_parity_smoke: prompt_tokens=%d checkpoints=%d stages=%d first_divergent_layer=%d first_divergent_kind=%s limits=max_abs:%g,rms:%g %s\n",
           prompt.len,
           n_layers,
           required_stages,
           have_first_bad ? checkpoint_actual_layer(first_bad) : -1,
           have_first_bad ? checkpoint_kind_name(checkpoint_kind(first_bad)) : "none",
           max_abs_limit,
           max_rms_limit,
           failures ? "FAIL" : "ok");
    if (prompt.len != full_prompt_len) {
        printf("cuda_v100_scheduler_checkpoint_parity_smoke: prompt_tokens_limited=%d full_prompt_tokens=%d\n",
               prompt.len,
               full_prompt_len);
    }

    for (int i = required_stages - 1; i >= 0; i--) {
        ds4_stage_scheduler_close(scheds[i]);
    }
    unmap_model_file(&model);
    free(gpu_reports);
    free(cpu_weights);
    free(cpu_selected);
    free(gpu_hc);
    free(cpu_hc);
    ds4_tokens_free(&prompt);
    ds4_engine_close(cpu_engine);
    return failures ? 1 : 0;
}
