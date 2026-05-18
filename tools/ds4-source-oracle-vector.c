#include "ds4.h"

#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ORACLE_MAX_CASES 128
#define ORACLE_MAX_TOKEN_BYTES 128

typedef struct {
    char id[96];
    char prompt_path[512];
    int ctx;
    unsigned char selected[ORACLE_MAX_TOKEN_BYTES];
    int selected_len;
} oracle_case;

typedef struct {
    const char *model_path;
    const char *vector_path;
    const char *only_id;
    int threads;
    bool all_cases;
    bool dry_parse;
    bool guard_checks;
    bool guards_only;
} oracle_config;

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tools/ds4-source-oracle-vector [options]\n"
            "\n"
            "Options:\n"
            "  --model FILE        Source-layout GGUF model path\n"
            "  --vectors FILE      Vector fixture (default tests/test-vectors/official.vec)\n"
            "  --only ID           Run one case (default short_reasoning_plain)\n"
            "  --all               Run all vector cases\n"
            "  --threads N         CPU worker threads\n"
            "  --guard-checks      Also run source-layout guard checks\n"
            "  --guards-only       Run source-layout guard checks without vectors\n"
            "  --dry-parse         Parse vectors without opening a model\n"
            "  --help              Show this help\n");
}

static int set_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg);
    return 1;
}

static int parse_int_arg(const char *s, int *out) {
    if (!s || !*s) return 1;
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end != '\0' || v <= 0 || v > 1024) return 1;
    *out = (int)v;
    return 0;
}

static char *trim_line(char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    size_t n = strlen(line);
    while (n && isspace((unsigned char)line[n - 1])) line[--n] = '\0';
    return line;
}

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static bool hex_to_bytes(const char *hex, unsigned char *out, int cap, int *len) {
    int n = 0;
    while (*hex && !isspace((unsigned char)*hex)) {
        if (!hex[1]) return false;
        int hi = hex_digit(hex[0]);
        int lo = hex_digit(hex[1]);
        if (hi < 0 || lo < 0 || n >= cap) return false;
        out[n++] = (unsigned char)((hi << 4) | lo);
        hex += 2;
    }
    *len = n;
    return true;
}

static void print_hex(FILE *fp, const unsigned char *p, size_t n) {
    static const char hexdigits[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        fputc(hexdigits[p[i] >> 4], fp);
        fputc(hexdigits[p[i] & 0xf], fp);
    }
}

static char *read_file(const char *path, char *err, size_t errlen) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        if (err && errlen) snprintf(err, errlen, "cannot open %s", path);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        if (err && errlen) snprintf(err, errlen, "cannot seek %s", path);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        if (err && errlen) snprintf(err, errlen, "cannot size %s", path);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) {
        fclose(fp);
        set_err(err, errlen, "out of memory reading prompt");
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)len, fp);
    fclose(fp);
    if (got != (size_t)len) {
        free(buf);
        if (err && errlen) snprintf(err, errlen, "short read from %s", path);
        return NULL;
    }
    buf[len] = '\0';
    return buf;
}

static int read_vector_cases(const char *path,
                             const oracle_config *cfg,
                             oracle_case *cases,
                             int *out_count,
                             char *err,
                             size_t errlen) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        if (err && errlen) snprintf(err, errlen, "cannot open vector file %s", path);
        return 1;
    }

    char line[2048];
    oracle_case current;
    bool in_case = false;
    bool selected_step0 = false;
    int count = 0;
    int line_no = 0;

    memset(&current, 0, sizeof(current));
    while (fgets(line, sizeof(line), fp)) {
        line_no++;
        char *p = trim_line(line);
        if (!p[0] || p[0] == '#') continue;

        if (!strncmp(p, "case ", 5)) {
            if (in_case) {
                fclose(fp);
                if (err && errlen) snprintf(err, errlen, "line %d: nested vector case", line_no);
                return 1;
            }
            int steps = 0;
            memset(&current, 0, sizeof(current));
            if (sscanf(p, "case %95s %d %d %511s",
                       current.id, &current.ctx, &steps, current.prompt_path) != 4 ||
                current.ctx <= 0 || steps <= 0) {
                fclose(fp);
                if (err && errlen) snprintf(err, errlen, "line %d: bad vector case", line_no);
                return 1;
            }
            in_case = true;
            selected_step0 = false;
            continue;
        }

        if (!strcmp(p, "end")) {
            if (!in_case || !selected_step0) {
                fclose(fp);
                if (err && errlen) snprintf(err, errlen, "line %d: incomplete vector case", line_no);
                return 1;
            }
            const bool selected = cfg->all_cases ||
                (cfg->only_id && !strcmp(cfg->only_id, current.id));
            if (selected) {
                if (count >= ORACLE_MAX_CASES) {
                    fclose(fp);
                    return set_err(err, errlen, "too many selected vector cases");
                }
                cases[count++] = current;
            }
            in_case = false;
            continue;
        }

        if (!in_case) {
            fclose(fp);
            if (err && errlen) snprintf(err, errlen, "line %d: unexpected vector line", line_no);
            return 1;
        }

        if (!strncmp(p, "step ", 5)) {
            int step = -1;
            int ntop = 0;
            char hex[ORACLE_MAX_TOKEN_BYTES * 2 + 2];
            if (sscanf(p, "step %d %257s %d", &step, hex, &ntop) != 3) {
                fclose(fp);
                if (err && errlen) snprintf(err, errlen, "line %d: bad vector step", line_no);
                return 1;
            }
            if (step == 0) {
                if (!hex_to_bytes(hex, current.selected, ORACLE_MAX_TOKEN_BYTES,
                                  &current.selected_len)) {
                    fclose(fp);
                    if (err && errlen) snprintf(err, errlen, "line %d: bad selected token hex", line_no);
                    return 1;
                }
                selected_step0 = true;
            }
            continue;
        }

        if (!strncmp(p, "top ", 4)) continue;

        fclose(fp);
        if (err && errlen) snprintf(err, errlen, "line %d: unexpected vector line", line_no);
        return 1;
    }
    fclose(fp);

    if (in_case) return set_err(err, errlen, "unterminated vector case");
    if (count == 0) return set_err(err, errlen, "no matching vector cases");
    *out_count = count;
    return 0;
}

static bool token_bytes_equal(ds4_engine *engine,
                              int token,
                              const unsigned char *want,
                              int want_len,
                              unsigned char *got_buf,
                              size_t *got_len) {
    size_t len = 0;
    char *got = ds4_token_text(engine, token, &len);
    bool ok = got && len == (size_t)want_len && memcmp(got, want, (size_t)want_len) == 0;
    if (got && got_buf && got_len) {
        size_t copy = len < ORACLE_MAX_TOKEN_BYTES ? len : ORACLE_MAX_TOKEN_BYTES;
        memcpy(got_buf, got, copy);
        *got_len = copy;
    }
    free(got);
    return ok;
}

static int run_vector_case(ds4_engine *engine, const oracle_case *vc) {
    char err[256];
    char *prompt_text = read_file(vc->prompt_path, err, sizeof(err));
    if (!prompt_text) {
        fprintf(stderr, "ds4-source-oracle-vector: %s\n", err);
        return 1;
    }

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    free(prompt_text);

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, vc->ctx) != 0) {
        fprintf(stderr, "ds4-source-oracle-vector: %s session create failed\n", vc->id);
        ds4_tokens_free(&prompt);
        return 1;
    }
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-source-oracle-vector: %s sync failed: %s\n", vc->id, err);
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        return 1;
    }

    unsigned char got[ORACLE_MAX_TOKEN_BYTES];
    size_t got_len = 0;
    int token = ds4_session_argmax(session);
    bool ok = token_bytes_equal(engine, token, vc->selected, vc->selected_len, got, &got_len);
    if (!ok) {
        fprintf(stderr, "ds4-source-oracle-vector: %s selected token mismatch expected=", vc->id);
        print_hex(stderr, vc->selected, (size_t)vc->selected_len);
        fprintf(stderr, " got=");
        print_hex(stderr, got, got_len);
        fprintf(stderr, " token=%d\n", token);
    } else {
        printf("vector\t%s\tOK\tselected=", vc->id);
        print_hex(stdout, vc->selected, (size_t)vc->selected_len);
        printf("\ttoken=%d\n", token);
    }

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    return ok ? 0 : 1;
}

static int expect_open_failure(const char *name, const ds4_engine_options *opt) {
    ds4_engine *engine = NULL;
    int rc = ds4_engine_open(&engine, opt);
    if (rc == 0) {
        fprintf(stderr, "guard\t%s\tFAIL\tunexpected engine open\n", name);
        ds4_engine_close(engine);
        return 1;
    }
    printf("guard\t%s\tOK\n", name);
    fflush(stdout);
    return 0;
}

static int run_guard_checks(const oracle_config *cfg) {
    int failures = 0;
    ds4_engine_options opt;

    memset(&opt, 0, sizeof(opt));
    opt.model_path = cfg->model_path;
    opt.backend = DS4_BACKEND_CPU;
    opt.n_threads = cfg->threads;
    failures += expect_open_failure("normal_source_layout_rejection", &opt);

    memset(&opt, 0, sizeof(opt));
    opt.model_path = cfg->model_path;
    opt.backend = DS4_BACKEND_METAL;
    opt.n_threads = cfg->threads;
    opt.source_layout_oracle = true;
    failures += expect_open_failure("non_cpu_oracle_rejection", &opt);

    memset(&opt, 0, sizeof(opt));
    opt.model_path = cfg->model_path;
    opt.backend = DS4_BACKEND_CPU;
    opt.n_threads = cfg->threads;
    opt.source_layout_oracle = true;
    opt.mtp_path = "diagnostic-mtp-sidecar-not-opened.gguf";
    failures += expect_open_failure("mtp_oracle_rejection", &opt);

    memset(&opt, 0, sizeof(opt));
    opt.model_path = cfg->model_path;
    opt.backend = DS4_BACKEND_CPU;
    opt.n_threads = cfg->threads;
    opt.source_layout_oracle = true;
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "guard\tmissing_session_unlock\tFAIL\tengine open failed\n");
        failures++;
    } else {
        ds4_session *session = NULL;
        if (ds4_session_create(&session, engine, 4096) == 0) {
            fprintf(stderr, "guard\tmissing_session_unlock\tFAIL\tunexpected session create\n");
            ds4_session_free(session);
            failures++;
        } else {
            printf("guard\tmissing_session_unlock\tOK\n");
            fflush(stdout);
        }
        ds4_engine_close(engine);
    }

    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    oracle_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.vector_path = "tests/test-vectors/official.vec";
    cfg.only_id = "short_reasoning_plain";

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(arg, "--model") && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if (!strcmp(arg, "--vectors") && i + 1 < argc) {
            cfg.vector_path = argv[++i];
        } else if (!strcmp(arg, "--only") && i + 1 < argc) {
            cfg.only_id = argv[++i];
            cfg.all_cases = false;
        } else if (!strcmp(arg, "--all")) {
            cfg.all_cases = true;
        } else if (!strcmp(arg, "--threads") && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &cfg.threads)) {
                fprintf(stderr, "ds4-source-oracle-vector: bad --threads value\n");
                return 2;
            }
        } else if (!strcmp(arg, "--dry-parse")) {
            cfg.dry_parse = true;
        } else if (!strcmp(arg, "--guard-checks")) {
            cfg.guard_checks = true;
        } else if (!strcmp(arg, "--guards-only")) {
            cfg.guard_checks = true;
            cfg.guards_only = true;
        } else {
            fprintf(stderr, "ds4-source-oracle-vector: unknown or incomplete option: %s\n", arg);
            usage(stderr);
            return 2;
        }
    }

    oracle_case cases[ORACLE_MAX_CASES];
    int n_cases = 0;
    char err[256];
    if (read_vector_cases(cfg.vector_path, &cfg, cases, &n_cases, err, sizeof(err))) {
        fprintf(stderr, "ds4-source-oracle-vector: %s\n", err);
        return 1;
    }

    if (cfg.dry_parse) {
        for (int i = 0; i < n_cases; i++) {
            printf("vector\t%s\tPARSED\tctx=%d\tselected=",
                   cases[i].id, cases[i].ctx);
            print_hex(stdout, cases[i].selected, (size_t)cases[i].selected_len);
            printf("\tprompt=%s\n", cases[i].prompt_path);
        }
        printf("ds4-source-oracle-vector: dry-parse ok\n");
        return 0;
    }

    if (!cfg.model_path || !cfg.model_path[0]) {
        fprintf(stderr, "ds4-source-oracle-vector: --model is required unless --dry-parse is set\n");
        return 2;
    }

    int rc = 0;
    if (cfg.guard_checks) {
        rc |= run_guard_checks(&cfg);
        if (cfg.guards_only) return rc ? 1 : 0;
    }

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = cfg.model_path;
    opt.backend = DS4_BACKEND_CPU;
    opt.n_threads = cfg.threads;
    opt.source_layout_oracle = true;
    opt.source_layout_oracle_sessions = true;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "ds4-source-oracle-vector: source oracle engine open failed\n");
        return 1;
    }

    for (int i = 0; i < n_cases; i++) {
        rc |= run_vector_case(engine, &cases[i]);
    }

    ds4_engine_close(engine);
    if (rc == 0) printf("ds4-source-oracle-vector: ok\n");
    return rc ? 1 : 0;
}
