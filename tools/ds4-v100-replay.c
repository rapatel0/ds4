#include "ds4_v100_replay.h"

#include <errno.h>
#include <arpa/inet.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

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
    const char *host;
    uint64_t ctx;
    uint32_t tokens;
    uint32_t max_requests;
    int port;
    bool json;
    bool serve;
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
            "  --serve                   run a minimal HTTP endpoint\n"
            "  --host ADDR               server bind address, default 127.0.0.1\n"
            "  --port N                  server port, default 8000\n"
            "  --max-requests N          server requests before exit, default unlimited\n"
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
    opt.host = "127.0.0.1";
    opt.port = 8000;
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
        } else if (!strcmp(arg, "--serve")) {
            opt.serve = true;
        } else if (!strcmp(arg, "--host")) {
            opt.host = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--port")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v > 65535) {
                fprintf(stderr, "ds4-v100-replay: invalid --port\n");
                exit(2);
            }
            opt.port = (int)v;
        } else if (!strcmp(arg, "--max-requests")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v > UINT32_MAX) {
                fprintf(stderr, "ds4-v100-replay: invalid --max-requests\n");
                exit(2);
            }
            opt.max_requests = (uint32_t)v;
        } else {
            fprintf(stderr, "ds4-v100-replay: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }
    if (!opt.model_path || !opt.index_path || (!opt.serve && !opt.prompt && !opt.prompt_file)) {
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

static void print_json_fp(FILE *fp,
                          const ds4_v100_replay_output *outputs,
                          uint32_t n_outputs,
                          const ds4_v100_replay_counters *c) {
    fprintf(fp, "{");
    fprintf(fp, "\"prompt_tokens\":%" PRIu32 ",", c->prompt_tokens);
    fprintf(fp, "\"generated_tokens\":%" PRIu32 ",", n_outputs);
    fprintf(fp, "\"total_input_tokens\":%" PRIu32 ",", c->total_input_tokens);
    fprintf(fp, "\"tokens\":[");
    for (uint32_t i = 0; i < n_outputs; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp,
                "{\"id\":%" PRIu32 ",\"logit\":%.9g,\"text\":\"",
                outputs[i].token,
                outputs[i].logit);
        json_escape(fp, outputs[i].text ? outputs[i].text : "", outputs[i].text_len);
        fprintf(fp, "\",\"text_hex\":\"");
        if (outputs[i].text) print_hex(fp, (const unsigned char *)outputs[i].text, outputs[i].text_len);
        fprintf(fp, "\"}");
    }
    fprintf(fp, "],\"timing_ms\":{");
    fprintf(fp, "\"open_total\":%.3f,", c->open_total_ms);
    fprintf(fp, "\"prompt_replay\":%.3f,", c->prompt_replay_ms);
    fprintf(fp, "\"continuation_decode\":%.3f,", c->continuation_decode_ms);
    fprintf(fp, "\"output_head\":%.3f,", c->output_head_ms);
    fprintf(fp, "\"token_text\":%.3f,", c->token_text_ms);
    fprintf(fp, "\"total\":%.3f,", c->total_ms);
    fprintf(fp,
            "\"prompt_tokens_per_second\":%.6f,",
            c->prompt_replay_ms > 0.0 ? (double)c->prompt_tokens * 1000.0 / c->prompt_replay_ms : 0.0);
    fprintf(fp,
            "\"continuation_tokens_per_second\":%.6f,",
            c->continuation_decode_ms > 0.0 && c->generated_tokens > 1
                ? (double)(c->generated_tokens - 1) * 1000.0 / c->continuation_decode_ms
                : 0.0);
    fprintf(fp,
            "\"generated_tokens_per_second\":%.6f,",
            c->total_ms > 0.0 ? (double)c->generated_tokens * 1000.0 / c->total_ms : 0.0);
    fprintf(fp, "\"stage_decode\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->stage_decode_ms[i]);
    }
    fprintf(fp, "],\"handoff\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS - 1; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->handoff_ms[i]);
    }
    fprintf(fp, "],\"open_stage\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->open_ms[i]);
    }
    fprintf(fp, "]},\"memory\":{");
    fprintf(fp, "\"uploaded_tensors\":%" PRIu64 ",", c->uploaded_tensors);
    fprintf(fp, "\"uploaded_bytes\":%" PRIu64 ",", c->uploaded_bytes);
    fprintf(fp, "\"arena_bytes\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%" PRIu64, c->arena_bytes[i]);
    }
    fprintf(fp, "]}}\n");
}

static void print_json(const ds4_v100_replay_output *outputs,
                       uint32_t n_outputs,
                       const ds4_v100_replay_counters *c) {
    print_json_fp(stdout, outputs, n_outputs, c);
}

static int hex4(const char *s, uint32_t *out) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        int h = hex_value(s[i]);
        if (h < 0) return 0;
        v = (v << 4) | (uint32_t)h;
    }
    *out = v;
    return 1;
}

static void append_utf8(char **buf, size_t *len, size_t *cap, uint32_t cp) {
    char tmp[4];
    size_t n = 0;
    if (cp <= 0x7f) {
        tmp[n++] = (char)cp;
    } else if (cp <= 0x7ff) {
        tmp[n++] = (char)(0xc0 | (cp >> 6));
        tmp[n++] = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        tmp[n++] = (char)(0xe0 | (cp >> 12));
        tmp[n++] = (char)(0x80 | ((cp >> 6) & 0x3f));
        tmp[n++] = (char)(0x80 | (cp & 0x3f));
    } else {
        tmp[n++] = (char)(0xf0 | (cp >> 18));
        tmp[n++] = (char)(0x80 | ((cp >> 12) & 0x3f));
        tmp[n++] = (char)(0x80 | ((cp >> 6) & 0x3f));
        tmp[n++] = (char)(0x80 | (cp & 0x3f));
    }
    if (*len + n + 1 > *cap) {
        size_t next = *cap ? *cap * 2 : 128;
        while (next < *len + n + 1) next *= 2;
        char *p = (char *)realloc(*buf, next);
        if (!p) return;
        *buf = p;
        *cap = next;
    }
    memcpy(*buf + *len, tmp, n);
    *len += n;
    (*buf)[*len] = '\0';
}

static char *json_get_string(const char *body, const char *key) {
    char pattern[96];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(body, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p != ':') return NULL;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p != '"') return NULL;
    p++;
    char *out = NULL;
    size_t len = 0;
    size_t cap = 0;
    while (*p && *p != '"') {
        unsigned char c = (unsigned char)*p++;
        if (c == '\\') {
            c = (unsigned char)*p++;
            switch (c) {
            case '"': append_utf8(&out, &len, &cap, '"'); break;
            case '\\': append_utf8(&out, &len, &cap, '\\'); break;
            case '/': append_utf8(&out, &len, &cap, '/'); break;
            case 'b': append_utf8(&out, &len, &cap, '\b'); break;
            case 'f': append_utf8(&out, &len, &cap, '\f'); break;
            case 'n': append_utf8(&out, &len, &cap, '\n'); break;
            case 'r': append_utf8(&out, &len, &cap, '\r'); break;
            case 't': append_utf8(&out, &len, &cap, '\t'); break;
            case 'u': {
                uint32_t cp = 0;
                if (!hex4(p, &cp)) {
                    free(out);
                    return NULL;
                }
                p += 4;
                append_utf8(&out, &len, &cap, cp);
                break;
            }
            default:
                free(out);
                return NULL;
            }
        } else {
            append_utf8(&out, &len, &cap, c);
        }
    }
    if (*p != '"') {
        free(out);
        return NULL;
    }
    if (!out) {
        out = (char *)malloc(1);
        if (out) out[0] = '\0';
    }
    return out;
}

static bool json_get_u32(const char *body, const char *key, uint32_t *out) {
    char pattern[96];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(body, pattern);
    if (!p) return false;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p != ':') return false;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    char *end = NULL;
    unsigned long v = strtoul(p, &end, 10);
    if (end == p || v > UINT32_MAX) return false;
    *out = (uint32_t)v;
    return true;
}

static int http_content_length(const char *req) {
    const char *p = req;
    while (*p) {
        const char *line_end = strstr(p, "\r\n");
        if (!line_end) break;
        if (line_end == p) break;
        if ((size_t)(line_end - p) > 15 &&
            strncasecmp(p, "Content-Length:", 15) == 0) {
            p += 15;
            while (*p == ' ' || *p == '\t') p++;
            return (int)strtol(p, NULL, 10);
        }
        p = line_end + 2;
    }
    return 0;
}

static char *read_http_request(int fd, size_t *out_len) {
    size_t cap = 8192;
    size_t len = 0;
    char *buf = (char *)malloc(cap + 1);
    if (!buf) return NULL;
    size_t want = 0;
    for (;;) {
        if (len == cap) {
            if (cap >= 1024 * 1024) {
                free(buf);
                return NULL;
            }
            cap *= 2;
            char *p = (char *)realloc(buf, cap + 1);
            if (!p) {
                free(buf);
                return NULL;
            }
            buf = p;
        }
        ssize_t n = recv(fd, buf + len, cap - len, 0);
        if (n <= 0) {
            free(buf);
            return NULL;
        }
        len += (size_t)n;
        buf[len] = '\0';
        char *body = strstr(buf, "\r\n\r\n");
        if (body && want == 0) {
            body += 4;
            int clen = http_content_length(buf);
            if (clen < 0 || clen > 1024 * 1024) {
                free(buf);
                return NULL;
            }
            want = (size_t)(body - buf) + (size_t)clen;
        }
        if (want && len >= want) {
            *out_len = len;
            return buf;
        }
    }
}

static void http_error(int fd, int status, const char *msg) {
    dprintf(fd,
            "HTTP/1.1 %d %s\r\nConnection: close\r\nContent-Type: application/json\r\n\r\n"
            "{\"error\":\"%s\"}\n",
            status,
            msg,
            msg);
}

static int handle_http_request(int fd, ds4_v100_replay *rt, const replay_cli_options *opt) {
    size_t req_len = 0;
    char *req = read_http_request(fd, &req_len);
    (void)req_len;
    if (!req) {
        http_error(fd, 400, "bad_request");
        return 1;
    }
    char method[8] = {0};
    char path[128] = {0};
    if (sscanf(req, "%7s %127s", method, path) != 2) {
        http_error(fd, 400, "bad_request");
        free(req);
        return 1;
    }
    if (!strcmp(method, "GET") && !strcmp(path, "/health")) {
        dprintf(fd,
                "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: application/json\r\n\r\n"
                "{\"status\":\"ok\"}\n");
        free(req);
        return 0;
    }
    if (strcmp(method, "POST") ||
        (strcmp(path, "/v100/selected-token") && strcmp(path, "/v1/v100/selected-token"))) {
        http_error(fd, 404, "not_found");
        free(req);
        return 1;
    }
    char *body = strstr(req, "\r\n\r\n");
    if (!body) {
        http_error(fd, 400, "missing_body");
        free(req);
        return 1;
    }
    body += 4;
    char *prompt = json_get_string(body, "prompt");
    if (!prompt) {
        http_error(fd, 400, "missing_prompt");
        free(req);
        return 1;
    }
    uint32_t tokens = opt->tokens;
    (void)json_get_u32(body, "tokens", &tokens);
    if (tokens == 0 || tokens > DS4_V100_REPLAY_MAX_TOKENS) {
        free(prompt);
        http_error(fd, 400, "bad_tokens");
        free(req);
        return 1;
    }

    char err[512] = {0};
    if (ds4_v100_replay_reset(rt, err, sizeof(err))) {
        free(prompt);
        http_error(fd, 500, err[0] ? err : "reset_failed");
        free(req);
        return 1;
    }
    ds4_tokens prompt_tokens = {0};
    ds4_v100_replay_encode_prompt(rt, opt->system, prompt, DS4_THINK_NONE, &prompt_tokens);
    ds4_v100_replay_output outputs[DS4_V100_REPLAY_MAX_TOKENS];
    memset(outputs, 0, sizeof(outputs));
    ds4_v100_replay_counters counters;
    memset(&counters, 0, sizeof(counters));
    uint32_t n_outputs = 0;
    int rc = ds4_v100_replay_generate(rt,
                                      &prompt_tokens,
                                      tokens,
                                      outputs,
                                      tokens,
                                      &n_outputs,
                                      &counters,
                                      err,
                                      sizeof(err));
    if (rc) {
        http_error(fd, 500, err[0] ? err : "generation_failed");
    } else {
        dprintf(fd,
                "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: application/json\r\n\r\n");
        FILE *fp = fdopen(dup(fd), "w");
        if (fp) {
            print_json_fp(fp, outputs, n_outputs, &counters);
            fclose(fp);
        }
    }
    for (uint32_t i = 0; i < n_outputs; i++) ds4_v100_replay_output_free(&outputs[i]);
    ds4_tokens_free(&prompt_tokens);
    free(prompt);
    free(req);
    return rc;
}

static int run_server(const replay_cli_options *opt, ds4_v100_replay *rt) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("ds4-v100-replay: socket");
        return 1;
    }
    int yes = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)opt->port);
    if (inet_pton(AF_INET, opt->host, &addr.sin_addr) != 1) {
        fprintf(stderr, "ds4-v100-replay: invalid --host: %s\n", opt->host);
        close(listen_fd);
        return 2;
    }
    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(listen_fd, 8) != 0) {
        perror("ds4-v100-replay: bind/listen");
        close(listen_fd);
        return 1;
    }
    fprintf(stderr,
            "ds4-v100-replay: serving http://%s:%d/v100/selected-token\n",
            opt->host,
            opt->port);
    uint32_t served = 0;
    while (opt->max_requests == 0 || served < opt->max_requests) {
        int fd = accept(listen_fd, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            perror("ds4-v100-replay: accept");
            close(listen_fd);
            return 1;
        }
        (void)handle_http_request(fd, rt, opt);
        close(fd);
        served++;
    }
    close(listen_fd);
    return 0;
}

int main(int argc, char **argv) {
    replay_cli_options opt = parse_options(argc, argv);
    char *prompt_owned = NULL;
    const char *prompt_text = NULL;
    if (!opt.serve) {
        prompt_owned = opt.prompt_file ? read_file(opt.prompt_file) : NULL;
        prompt_text = opt.prompt_file ? prompt_owned : opt.prompt;
        if (!prompt_text) return 1;
    }

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

    if (opt.serve) {
        int rc = run_server(&opt, rt);
        ds4_v100_replay_close(rt);
        free(expected);
        return rc;
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
