#include "ds4_v100_replay.h"

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    DS4_V100_REPLAY_MAX_TOKENS = 64,
};

typedef struct {
    const char *model_path;
    const char *index_path;
    const char *prompt;
    const char *prompt_file;
    const char *system;
    const char *expected_hex;
    uint64_t ctx;
    uint32_t tokens;
    bool json;
} replay_cli_options;

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tools/ds4-v100-replay --model FILE --index FILE [options]\n"
            "\n"
            "Options:\n"
            "  --model FILE              source-layout GGUF model\n"
            "  --index FILE              V100 pack-index.tsv\n"
            "  --prompt TEXT             prompt text\n"
            "  --prompt-file FILE        prompt file\n"
            "  --system TEXT             system prompt, default empty\n"
            "  --tokens N                greedy tokens to generate, default 1\n"
            "  --ctx N                   KV context tokens, default 1048576\n"
            "  --expected-token-hex HEX  require first generated token bytes\n"
            "  --json                    emit JSON\n"
            "  --help                    show this help\n");
}

static const char *need_arg(int *i, int argc, char **argv, const char *arg) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-v100-replay: %s requires an argument\n", arg);
        exit(2);
    }
    return argv[++*i];
}

static uint64_t parse_u64_arg(const char *s, const char *arg) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (!s || !*s || !end || *end || v == 0) {
        fprintf(stderr, "ds4-v100-replay: invalid %s: %s\n", arg, s ? s : "(null)");
        exit(2);
    }
    return (uint64_t)v;
}

static replay_cli_options parse_options(int argc, char **argv) {
    replay_cli_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.ctx = 1048576;
    opt.tokens = 1;
    opt.system = "";
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "--model")) {
            opt.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--index")) {
            opt.index_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt")) {
            opt.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            opt.prompt_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--system")) {
            opt.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--tokens")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v > DS4_V100_REPLAY_MAX_TOKENS) {
                fprintf(stderr,
                        "ds4-v100-replay: --tokens must be <= %d\n",
                        DS4_V100_REPLAY_MAX_TOKENS);
                exit(2);
            }
            opt.tokens = (uint32_t)v;
        } else if (!strcmp(arg, "--ctx")) {
            opt.ctx = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--expected-token-hex")) {
            opt.expected_hex = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--json")) {
            opt.json = true;
        } else {
            fprintf(stderr, "ds4-v100-replay: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }
    if (!opt.model_path || !opt.index_path || (!opt.prompt && !opt.prompt_file)) {
        usage(stderr);
        exit(2);
    }
    if (opt.prompt && opt.prompt_file) {
        fprintf(stderr, "ds4-v100-replay: use --prompt or --prompt-file, not both\n");
        exit(2);
    }
    return opt;
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-v100-replay: cannot open %s: %s\n", path, strerror(errno));
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
    const size_t n = strlen(hex);
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

static void json_escape(FILE *fp, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"': fputs("\\\"", fp); break;
        case '\\': fputs("\\\\", fp); break;
        case '\b': fputs("\\b", fp); break;
        case '\f': fputs("\\f", fp); break;
        case '\n': fputs("\\n", fp); break;
        case '\r': fputs("\\r", fp); break;
        case '\t': fputs("\\t", fp); break;
        default:
            if (c < 0x20) fprintf(fp, "\\u%04x", c);
            else fputc((char)c, fp);
            break;
        }
    }
}

static void print_json(const ds4_v100_replay_output *outputs,
                       uint32_t n_outputs,
                       const ds4_v100_replay_counters *c) {
    printf("{");
    printf("\"prompt_tokens\":%" PRIu32 ",", c->prompt_tokens);
    printf("\"generated_tokens\":%" PRIu32 ",", n_outputs);
    printf("\"total_input_tokens\":%" PRIu32 ",", c->total_input_tokens);
    printf("\"tokens\":[");
    for (uint32_t i = 0; i < n_outputs; i++) {
        if (i) printf(",");
        printf("{\"id\":%" PRIu32 ",\"logit\":%.9g,\"text\":\"",
               outputs[i].token,
               outputs[i].logit);
        json_escape(stdout, outputs[i].text ? outputs[i].text : "", outputs[i].text_len);
        printf("\",\"text_hex\":\"");
        if (outputs[i].text) print_hex(stdout, (const unsigned char *)outputs[i].text, outputs[i].text_len);
        printf("\"}");
    }
    printf("],\"timing_ms\":{");
    printf("\"open_total\":%.3f,", c->open_total_ms);
    printf("\"prompt_replay\":%.3f,", c->prompt_replay_ms);
    printf("\"continuation_decode\":%.3f,", c->continuation_decode_ms);
    printf("\"output_head\":%.3f,", c->output_head_ms);
    printf("\"token_text\":%.3f,", c->token_text_ms);
    printf("\"total\":%.3f,", c->total_ms);
    printf("\"prompt_tokens_per_second\":%.6f,",
           c->prompt_replay_ms > 0.0 ? (double)c->prompt_tokens * 1000.0 / c->prompt_replay_ms : 0.0);
    printf("\"continuation_tokens_per_second\":%.6f,",
           c->continuation_decode_ms > 0.0 && c->generated_tokens > 1
               ? (double)(c->generated_tokens - 1) * 1000.0 / c->continuation_decode_ms
               : 0.0);
    printf("\"generated_tokens_per_second\":%.6f,",
           c->total_ms > 0.0 ? (double)c->generated_tokens * 1000.0 / c->total_ms : 0.0);
    printf("\"stage_decode\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) printf(",");
        printf("%.3f", c->stage_decode_ms[i]);
    }
    printf("],\"handoff\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS - 1; i++) {
        if (i) printf(",");
        printf("%.3f", c->handoff_ms[i]);
    }
    printf("],\"open_stage\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) printf(",");
        printf("%.3f", c->open_ms[i]);
    }
    printf("]},\"memory\":{");
    printf("\"uploaded_tensors\":%" PRIu64 ",", c->uploaded_tensors);
    printf("\"uploaded_bytes\":%" PRIu64 ",", c->uploaded_bytes);
    printf("\"arena_bytes\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) printf(",");
        printf("%" PRIu64, c->arena_bytes[i]);
    }
    printf("]}}");
    printf("\n");
}

int main(int argc, char **argv) {
    replay_cli_options opt = parse_options(argc, argv);
    char *prompt_owned = opt.prompt_file ? read_file(opt.prompt_file) : NULL;
    const char *prompt_text = opt.prompt_file ? prompt_owned : opt.prompt;
    if (!prompt_text) return 1;

    unsigned char *expected = NULL;
    size_t expected_len = 0;
    if (parse_hex_bytes(opt.expected_hex, &expected, &expected_len)) {
        fprintf(stderr, "ds4-v100-replay: invalid --expected-token-hex\n");
        free(prompt_owned);
        return 2;
    }

    ds4_v100_replay_options ropts;
    ds4_v100_replay_options_init(&ropts);
    ropts.model_path = opt.model_path;
    ropts.pack_index_path = opt.index_path;
    ropts.kv_ctx_tokens = opt.ctx;

    char err[512] = {0};
    ds4_v100_replay *rt = NULL;
    if (ds4_v100_replay_open(&rt, &ropts, err, sizeof(err))) {
        fprintf(stderr, "ds4-v100-replay: %s\n", err[0] ? err : "open failed");
        free(expected);
        free(prompt_owned);
        return 1;
    }

    ds4_tokens prompt = {0};
    ds4_v100_replay_encode_prompt(rt, opt.system, prompt_text, DS4_THINK_NONE, &prompt);
    ds4_v100_replay_output outputs[DS4_V100_REPLAY_MAX_TOKENS];
    memset(outputs, 0, sizeof(outputs));
    ds4_v100_replay_counters counters;
    memset(&counters, 0, sizeof(counters));
    uint32_t n_outputs = 0;
    int rc = 0;
    if (ds4_v100_replay_generate(rt,
                                 &prompt,
                                 opt.tokens,
                                 outputs,
                                 opt.tokens,
                                 &n_outputs,
                                 &counters,
                                 err,
                                 sizeof(err))) {
        fprintf(stderr, "ds4-v100-replay: %s\n", err[0] ? err : "generation failed");
        rc = 1;
    }

    if (rc == 0 && expected && n_outputs > 0) {
        const bool ok = outputs[0].text &&
                        outputs[0].text_len == expected_len &&
                        memcmp(outputs[0].text, expected, expected_len) == 0;
        if (!ok) {
            fprintf(stderr, "ds4-v100-replay: selected token mismatch expected=");
            print_hex(stderr, expected, expected_len);
            fprintf(stderr, " got=");
            if (outputs[0].text) {
                print_hex(stderr, (const unsigned char *)outputs[0].text, outputs[0].text_len);
            }
            fprintf(stderr, " token=%" PRIu32 " logit=%.8g\n",
                    outputs[0].token,
                    outputs[0].logit);
            rc = 1;
        }
    }

    if (rc == 0) {
        if (opt.json) {
            print_json(outputs, n_outputs, &counters);
        } else {
            printf("ds4-v100-replay: prompt_tokens=%" PRIu32
                   " generated=%" PRIu32
                   " first_token=%" PRIu32
                   " first_logit=%.8g"
                   " first_hex=",
                   counters.prompt_tokens,
                   n_outputs,
                   n_outputs ? outputs[0].token : UINT32_MAX,
                   n_outputs ? outputs[0].logit : 0.0f);
            if (n_outputs && outputs[0].text) {
                print_hex(stdout, (const unsigned char *)outputs[0].text, outputs[0].text_len);
            } else {
                printf("none");
            }
            printf(" prompt_ms=%.3f continuation_ms=%.3f output_ms=%.3f total_ms=%.3f ok\n",
                   counters.prompt_replay_ms,
                   counters.continuation_decode_ms,
                   counters.output_head_ms,
                   counters.total_ms);
        }
    }

    for (uint32_t i = 0; i < n_outputs; i++) ds4_v100_replay_output_free(&outputs[i]);
    ds4_tokens_free(&prompt);
    ds4_v100_replay_close(rt);
    free(expected);
    free(prompt_owned);
    return rc;
}
