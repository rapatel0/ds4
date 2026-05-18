#include "ds4.h"
#include "ds4_v100_scheduler.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    const unsigned char *ptr;
    uint64_t size;
    int fd;
} model_map;

static int failures;

static void check(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "cuda_v100_selected_token_smoke: %s\n", msg);
        failures++;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/cuda_v100_selected_token_smoke --index FILE --model FILE "
            "[--prompt-file FILE] [--expected-token-hex HEX]\n");
}

static int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static int parse_hex_bytes(const char *hex, unsigned char **out, size_t *out_len) {
    *out = NULL;
    *out_len = 0;
    if (!hex) return 0;
    size_t n = strlen(hex);
    if (n == 0 || (n & 1u)) return 1;
    unsigned char *buf = (unsigned char *)malloc(n / 2u);
    if (!buf) return 1;
    for (size_t i = 0; i < n; i += 2) {
        int hi = hex_value(hex[i]);
        int lo = hex_value(hex[i + 1]);
        if (hi < 0 || lo < 0) {
            free(buf);
            return 1;
        }
        buf[i / 2u] = (unsigned char)((hi << 4) | lo);
    }
    *out = buf;
    *out_len = n / 2u;
    return 0;
}

static void print_hex(FILE *fp, const unsigned char *p, size_t n) {
    static const char h[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        fputc(h[p[i] >> 4], fp);
        fputc(h[p[i] & 15], fp);
    }
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "cuda_v100_selected_token_smoke: cannot open %s: %s\n",
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
        fprintf(stderr, "cuda_v100_selected_token_smoke: cannot open %s: %s\n",
                path,
                strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "cuda_v100_selected_token_smoke: cannot stat %s\n", path);
        close(fd);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "cuda_v100_selected_token_smoke: cannot mmap %s: %s\n",
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

int main(int argc, char **argv) {
    const char *index = NULL;
    const char *model_path = NULL;
    const char *prompt_file = "tests/test-vectors/prompts/short_reasoning_plain.txt";
    const char *expected_hex = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--index") && i + 1 < argc) {
            index = argv[++i];
        } else if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) {
            prompt_file = argv[++i];
        } else if (!strcmp(argv[i], "--expected-token-hex") && i + 1 < argc) {
            expected_hex = argv[++i];
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

    int devices = ds4_gpu_device_count();
    if (devices < DS4_V100_EXPECTED_GPUS) {
        fprintf(stderr,
                "cuda_v100_selected_token_smoke: need 8 CUDA devices, got %d\n",
                devices);
        return 1;
    }

    unsigned char *expected = NULL;
    size_t expected_len = 0;
    if (parse_hex_bytes(expected_hex, &expected, &expected_len)) {
        fprintf(stderr, "cuda_v100_selected_token_smoke: invalid expected token hex\n");
        return 2;
    }

    ds4_engine_options eopts;
    memset(&eopts, 0, sizeof(eopts));
    eopts.model_path = model_path;
    eopts.backend = DS4_BACKEND_CPU;
    eopts.inspect_only = true;
    eopts.n_threads = 1;
    ds4_engine *tok_engine = NULL;
    if (ds4_engine_open(&tok_engine, &eopts) != 0) {
        fprintf(stderr, "cuda_v100_selected_token_smoke: tokenizer engine open failed\n");
        free(expected);
        return 1;
    }

    char *prompt_text = read_file(prompt_file);
    if (!prompt_text) {
        ds4_engine_close(tok_engine);
        free(expected);
        return 1;
    }
    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(tok_engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    free(prompt_text);
    check(prompt.len > 0, "empty prompt tokenization");

    model_map model;
    if (map_model_file(model_path, &model)) {
        ds4_tokens_free(&prompt);
        ds4_engine_close(tok_engine);
        free(expected);
        return 1;
    }
    check(ds4_gpu_set_model_fd(model.fd), "model fd");

    ds4_v100_stage_scheduler *scheds[DS4_V100_EXPECTED_GPUS];
    memset(scheds, 0, sizeof(scheds));
    uint32_t selected = UINT32_MAX;
    float selected_logit = 0.0f;

    ds4_v100_stage_scheduler_options opts;
    ds4_v100_stage_scheduler_options_init(&opts);
    opts.pack_index_path = index;
    opts.model_map = model.ptr;
    opts.model_size = model.size;
    opts.attn_comp_cap = 64;
    opts.index_comp_cap = 64;
    opts.indexer_top_k = 512;

    char err[512] = {0};
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        opts.stage_id = i;
        if (ds4_v100_stage_scheduler_open(&scheds[i], &opts, err, sizeof(err))) {
            fprintf(stderr,
                    "cuda_v100_selected_token_smoke: open stage %d failed: %s\n",
                    i,
                    err[0] ? err : "scheduler open");
            failures++;
            goto cleanup;
        }
    }

    for (int pos = 0; pos < prompt.len && failures == 0; pos++) {
        if (prompt.v[pos] < 0) {
            check(0, "negative prompt token");
            break;
        }
        ds4_v100_stage_scheduler_report report;
        err[0] = '\0';
        check(ds4_v100_stage_scheduler_decode_token(scheds[0],
                                                    (uint32_t)prompt.v[pos],
                                                    (uint32_t)pos,
                                                    &report,
                                                    err,
                                                    sizeof(err)) == 0,
              err[0] ? err : "stage 0 prompt decode");
        for (int stage = 1; stage < DS4_V100_EXPECTED_GPUS && failures == 0; stage++) {
            err[0] = '\0';
            check(ds4_v100_stage_scheduler_handoff(scheds[stage],
                                                   scheds[stage - 1],
                                                   err,
                                                   sizeof(err)) == 0,
                  err[0] ? err : "stage handoff");
            err[0] = '\0';
            check(ds4_v100_stage_scheduler_decode_hc(scheds[stage],
                                                     (uint32_t)prompt.v[pos],
                                                     (uint32_t)pos,
                                                     &report,
                                                     err,
                                                     sizeof(err)) == 0,
                  err[0] ? err : "stage prompt decode");
        }
    }

    if (failures == 0) {
        err[0] = '\0';
        check(ds4_v100_stage_scheduler_select_token(scheds[DS4_V100_EXPECTED_GPUS - 1],
                                                    &selected,
                                                    &selected_logit,
                                                    err,
                                                    sizeof(err)) == 0,
              err[0] ? err : "selected token");
    }
    if (failures == 0 && expected) {
        size_t got_len = 0;
        char *got = ds4_token_text(tok_engine, (int)selected, &got_len);
        bool ok = got && got_len == expected_len &&
                  memcmp(got, expected, expected_len) == 0;
        if (!ok) {
            fprintf(stderr,
                    "cuda_v100_selected_token_smoke: selected token mismatch expected=");
            print_hex(stderr, expected, expected_len);
            fprintf(stderr, " got=");
            if (got) print_hex(stderr, (const unsigned char *)got, got_len);
            fprintf(stderr, " token=%" PRIu32 " logit=%.8g\n",
                    selected,
                    selected_logit);
            failures++;
        }
        free(got);
    }

cleanup:
    printf("cuda_v100_selected_token_smoke: prompt_tokens=%d selected=%" PRIu32
           " logit=%.8g expected=",
           prompt.len,
           selected,
           selected_logit);
    if (expected) print_hex(stdout, expected, expected_len);
    else printf("none");
    printf(" %s\n", failures ? "FAIL" : "ok");

    for (int i = DS4_V100_EXPECTED_GPUS - 1; i >= 0; i--) {
        ds4_v100_stage_scheduler_close(scheds[i]);
    }
    unmap_model_file(&model);
    ds4_tokens_free(&prompt);
    ds4_engine_close(tok_engine);
    free(expected);
    return failures ? 1 : 0;
}
