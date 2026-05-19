#include "ds4_v100_replay.h"
#include "ds4_v100_context.h"
#include "ds4_v100_mtp.h"
#include "ds4-v100-mtp-forward-common.h"

#include <errno.h>
#include <arpa/inet.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

enum {
    DS4_V100_REPLAY_MAX_TOKENS = 64,
};

typedef enum {
    REPLAY_MTP_SERVING_OFF = 0,
    REPLAY_MTP_SERVING_VERIFY = 1,
} replay_mtp_serving_mode;

typedef enum {
    REPLAY_QUEUE_REJECT_BUSY = 0,
    REPLAY_QUEUE_SEQUENTIAL = 1,
} replay_queue_policy;

typedef struct {
    const char *model_path;
    const char *mtp_model_path;
    const char *index_path;
    const char *prompt;
    const char *prompt_file;
    const char *system;
    const char *expected_hex;
    const char *host;
    uint64_t ctx;
    uint32_t tokens;
    uint32_t max_requests;
    uint32_t slots;
    uint32_t active_microbatch;
    uint32_t mtp_top_k;
    int mtp_gpu;
    int mtp_reserve_mib;
    int port;
    replay_mtp_serving_mode mtp_serving;
    replay_queue_policy queue_policy;
    bool json;
    bool serve;
    bool open_only;
    bool serial_open;
} replay_cli_options;

typedef struct {
    bool enabled;
    bool attempted;
    bool accepted;
    bool skipped;
    uint32_t committed_token;
    uint32_t committed_pos;
    uint32_t target_token;
    uint32_t draft_token;
    uint32_t top_k;
    uint32_t raw_row;
    uint32_t n_raw;
    uint32_t output_vocab;
    uint32_t draft_tokens[DS4_V100_MTP_FORWARD_MAX_TOPK];
    float draft_logits[DS4_V100_MTP_FORWARD_MAX_TOPK];
    float target_logit;
    float draft_logit;
    double draft_ms;
    uint64_t sidecar_uploaded_bytes;
    uint64_t sidecar_arena_bytes;
    uint64_t output_weight_bytes;
    uint64_t free_after_output_upload_bytes;
    char reason[96];
} replay_mtp_result;

typedef struct {
    bool enabled;
    uint32_t top_k;
    int gpu;
    int reserve_mib;
    uint64_t requests;
    uint64_t drafts;
    uint64_t accepted;
    uint64_t rejected;
    uint64_t skipped;
    ds4_v100_mtp_sidecar *sidecar;
    ds4_v100_mtp_forward *forward;
    uint64_t sidecar_uploaded_bytes;
    uint64_t sidecar_arena_bytes;
    uint64_t output_weight_bytes;
} replay_mtp_service;

typedef struct {
    uint64_t accepted_connections;
    uint64_t generation_requests;
    uint64_t rejected_requests;
    uint64_t rejected_busy;
    uint64_t rejected_context;
    uint64_t rejected_bad_request;
    uint64_t tensor_batched_groups;
    uint64_t tensor_batched_requests;
    uint64_t tensor_batched_tokens;
} replay_server_stats;

typedef struct replay_pending_generation replay_pending_generation;

typedef struct {
    const replay_cli_options *opt;
    ds4_v100_replay *rt;
    replay_mtp_service *mtp;
    pthread_mutex_t generation_mu;
    pthread_mutex_t queue_mu;
    pthread_cond_t queue_done;
    pthread_mutex_t stats_mu;
    pthread_mutex_t handlers_mu;
    pthread_cond_t handlers_done;
    pthread_mutex_t pending_mu;
    pthread_cond_t pending_cv;
    replay_pending_generation *pending_head;
    replay_pending_generation *pending_tail;
    uint32_t pending_count;
    replay_server_stats stats;
    uint32_t active_handlers;
    uint32_t active_generation_slots;
} replay_server_state;

typedef struct {
    replay_server_state *state;
    int fd;
} replay_request_worker;

struct replay_pending_generation {
    replay_pending_generation *next;
    ds4_tokens prompt_tokens;
    uint32_t tokens;
    ds4_v100_replay_output outputs[DS4_V100_REPLAY_MAX_TOKENS];
    uint32_t n_outputs;
    ds4_v100_replay_counters counters;
    int rc;
    char err[512];
    bool done;
};

typedef struct {
    replay_pending_generation *items[DS4_V100_SCHED_MAX_SLOTS];
    uint32_t count;
} replay_pending_batch;

static const char *queue_policy_name(replay_queue_policy p) {
    switch (p) {
    case REPLAY_QUEUE_SEQUENTIAL: return "sequential";
    case REPLAY_QUEUE_REJECT_BUSY:
    default: return "reject-busy";
    }
}

static void format_mode(char *dst, size_t dst_len, bool mtp_enabled, const replay_cli_options *opt) {
    if (!dst || dst_len == 0 || !opt) {
        return;
    }
    if (mtp_enabled) {
        if (opt->slots <= 1) {
            snprintf(dst, dst_len, "mtp_verify_one_slot");
            return;
        }
        snprintf(dst, dst_len, "mtp_verify_slots_%u", opt->slots);
        return;
    }
    if (opt->slots <= 1) {
        snprintf(dst, dst_len, "base_one_slot");
        return;
    }
    snprintf(dst, dst_len, "base_slots_%u", opt->slots);
}

static void server_stats_snapshot(replay_server_state *state, replay_server_stats *out) {
    if (!state || !out) return;
    pthread_mutex_lock(&state->stats_mu);
    *out = state->stats;
    pthread_mutex_unlock(&state->stats_mu);
}

static void server_stats_add(replay_server_state *state,
                             uint64_t accepted_connections,
                             uint64_t generation_requests,
                             uint64_t rejected_requests,
                             uint64_t rejected_busy,
                             uint64_t rejected_context,
                             uint64_t rejected_bad_request) {
    if (!state) return;
    pthread_mutex_lock(&state->stats_mu);
    state->stats.accepted_connections += accepted_connections;
    state->stats.generation_requests += generation_requests;
    state->stats.rejected_requests += rejected_requests;
    state->stats.rejected_busy += rejected_busy;
    state->stats.rejected_context += rejected_context;
    state->stats.rejected_bad_request += rejected_bad_request;
    pthread_mutex_unlock(&state->stats_mu);
}

static void server_stats_add_tensor_batch(replay_server_state *state,
                                          uint64_t groups,
                                          uint64_t requests,
                                          uint64_t tokens) {
    if (!state) return;
    pthread_mutex_lock(&state->stats_mu);
    state->stats.tensor_batched_groups += groups;
    state->stats.tensor_batched_requests += requests;
    state->stats.tensor_batched_tokens += tokens;
    pthread_mutex_unlock(&state->stats_mu);
}

static void pending_enqueue(replay_server_state *state, replay_pending_generation *req) {
    if (!state || !req) return;
    req->next = NULL;
    req->done = false;
    pthread_mutex_lock(&state->pending_mu);
    if (state->pending_tail) state->pending_tail->next = req;
    else state->pending_head = req;
    state->pending_tail = req;
    state->pending_count++;
    pthread_cond_broadcast(&state->pending_cv);
    pthread_mutex_unlock(&state->pending_mu);
}

static void pending_wait_for_microbatch(replay_server_state *state,
                                        uint32_t cap,
                                        bool mtp_enabled) {
    if (!state || cap <= 1 || mtp_enabled) return;

    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct timespec deadline;
    deadline.tv_sec = tv.tv_sec;
    deadline.tv_nsec = (long)tv.tv_usec * 1000L + 5000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec += deadline.tv_nsec / 1000000000L;
        deadline.tv_nsec %= 1000000000L;
    }

    pthread_mutex_lock(&state->pending_mu);
    while (state->pending_count > 0 && state->pending_count < cap) {
        const int rc = pthread_cond_timedwait(&state->pending_cv,
                                              &state->pending_mu,
                                              &deadline);
        if (rc == ETIMEDOUT) break;
        if (rc != 0) break;
    }
    pthread_mutex_unlock(&state->pending_mu);
}

static void pending_remove(replay_server_state *state, replay_pending_generation *req) {
    if (!state || !req) return;
    pthread_mutex_lock(&state->pending_mu);
    replay_pending_generation *prev = NULL;
    replay_pending_generation *cur = state->pending_head;
    while (cur) {
        if (cur == req) {
            if (prev) prev->next = cur->next;
            else state->pending_head = cur->next;
            if (state->pending_tail == cur) state->pending_tail = prev;
            if (state->pending_count > 0) state->pending_count--;
            break;
        }
        prev = cur;
        cur = cur->next;
    }
    pthread_mutex_unlock(&state->pending_mu);
}

static void pending_collect_batch(replay_server_state *state,
                                  uint32_t cap,
                                  replay_pending_batch *batch) {
    if (!state || !batch || cap == 0) return;
    memset(batch, 0, sizeof(*batch));
    pthread_mutex_lock(&state->pending_mu);
    replay_pending_generation *cur = state->pending_head;
    while (cur && batch->count < cap) {
        if (!cur->done) batch->items[batch->count++] = cur;
        cur = cur->next;
    }
    pthread_mutex_unlock(&state->pending_mu);
}

static void pending_mark_done(replay_pending_generation *req, int rc, const char *err) {
    if (!req) return;
    req->rc = rc;
    req->done = true;
    if (rc) {
        snprintf(req->err, sizeof(req->err), "%s", (err && err[0]) ? err : "generation_failed");
    }
}

static int process_pending_generation_batch(replay_server_state *state) {
    if (!state || !state->opt || !state->rt) return 1;
    uint32_t cap = state->opt->active_microbatch ? state->opt->active_microbatch : 1;
    if (cap > DS4_V100_SCHED_MAX_SLOTS) cap = DS4_V100_SCHED_MAX_SLOTS;
    const bool mtp_enabled = state->mtp && state->mtp->enabled;
    pending_wait_for_microbatch(state, cap, mtp_enabled);
    replay_pending_batch batch;
    pending_collect_batch(state, cap, &batch);
    if (batch.count == 0) return 0;

    const uint32_t batch_tokens = batch.items[0] ? batch.items[0]->tokens : 0;
    bool can_batch = !mtp_enabled && batch.count > 1 && batch_tokens > 0 &&
                     batch_tokens <= DS4_V100_REPLAY_MAX_TOKENS;
    for (uint32_t i = 0; i < batch.count && can_batch; i++) {
        if (!batch.items[i] || batch.items[i]->tokens != batch_tokens) can_batch = false;
    }

    if (can_batch) {
        ds4_tokens prompts[DS4_V100_SCHED_MAX_SLOTS];
        ds4_v100_replay_output batch_outputs[DS4_V100_SCHED_MAX_SLOTS *
                                             DS4_V100_REPLAY_MAX_TOKENS];
        uint32_t batch_counts[DS4_V100_SCHED_MAX_SLOTS];
        memset(prompts, 0, sizeof(prompts));
        memset(batch_outputs, 0, sizeof(batch_outputs));
        memset(batch_counts, 0, sizeof(batch_counts));
        for (uint32_t i = 0; i < batch.count; i++) prompts[i] = batch.items[i]->prompt_tokens;

        char err[512] = {0};
        ds4_v100_replay_counters counters;
        memset(&counters, 0, sizeof(counters));
        int rc = ds4_v100_replay_reset(state->rt, err, sizeof(err));
        if (!rc) {
            rc = ds4_v100_replay_generate_batch(state->rt,
                                                prompts,
                                                batch.count,
                                                batch_tokens,
                                                batch_outputs,
                                                DS4_V100_REPLAY_MAX_TOKENS,
                                                batch_counts,
                                                &counters,
                                                err,
                                                sizeof(err));
        }
        if (!rc) {
            server_stats_add_tensor_batch(state,
                                          1,
                                          batch.count,
                                          (uint64_t)batch.count * batch_tokens);
        }
        for (uint32_t i = 0; i < batch.count; i++) {
            replay_pending_generation *req = batch.items[i];
            if (!req) continue;
            memset(&req->counters, 0, sizeof(req->counters));
            req->counters = counters;
            req->counters.prompt_tokens = (uint32_t)req->prompt_tokens.len;
            if (rc) {
                pending_mark_done(req, 1, err);
            } else {
                const uint32_t n_out = batch_counts[i];
                for (uint32_t j = 0; j < n_out; j++) {
                    req->outputs[j] =
                        batch_outputs[(uint64_t)i * DS4_V100_REPLAY_MAX_TOKENS + j];
                }
                req->n_outputs = n_out;
                req->counters.generated_tokens = n_out;
                req->counters.total_input_tokens =
                    (uint32_t)req->prompt_tokens.len + (n_out > 0 ? n_out - 1u : 0u);
                pending_mark_done(req, 0, NULL);
            }
            pending_remove(state, req);
        }
        if (rc) {
            for (uint32_t i = 0; i < batch.count; i++) {
                for (uint32_t j = 0; j < batch_tokens; j++) {
                    ds4_v100_replay_output_free(
                        &batch_outputs[(uint64_t)i * DS4_V100_REPLAY_MAX_TOKENS + j]);
                }
            }
        }
        return 0;
    }

    for (uint32_t i = 0; i < batch.count; i++) {
        replay_pending_generation *req = batch.items[i];
        if (!req) continue;
        char err[512] = {0};
        ds4_v100_replay_counters counters;
        memset(&counters, 0, sizeof(counters));
        int rc = ds4_v100_replay_reset(state->rt, err, sizeof(err));
        if (!rc) {
            rc = ds4_v100_replay_generate(state->rt,
                                          &req->prompt_tokens,
                                          req->tokens,
                                          req->outputs,
                                          req->tokens,
                                          &req->n_outputs,
                                          &counters,
                                          err,
                                          sizeof(err));
        }
        req->counters = counters;
        pending_mark_done(req, rc, err);
        pending_remove(state, req);
    }
    return 0;
}

static int acquire_generation_slot(replay_server_state *state) {
    if (!state || !state->opt) return 1;
    const uint32_t cap = state->opt->active_microbatch ? state->opt->active_microbatch : 1;
    pthread_mutex_lock(&state->queue_mu);
    while (state->active_generation_slots >= cap) {
        if (state->opt->queue_policy == REPLAY_QUEUE_REJECT_BUSY) {
            pthread_mutex_unlock(&state->queue_mu);
            return 1;
        }
        if (pthread_cond_wait(&state->queue_done, &state->queue_mu) != 0) {
            pthread_mutex_unlock(&state->queue_mu);
            return 1;
        }
    }
    state->active_generation_slots++;
    pthread_mutex_unlock(&state->queue_mu);
    return 0;
}

static void release_generation_slot(replay_server_state *state) {
    if (!state) return;
    pthread_mutex_lock(&state->queue_mu);
    if (state->active_generation_slots > 0) state->active_generation_slots--;
    pthread_cond_signal(&state->queue_done);
    pthread_mutex_unlock(&state->queue_mu);
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tools/ds4-v100-replay --model FILE --index FILE [options]\n"
            "\n"
            "Options:\n"
            "  --model FILE              source-layout GGUF model\n"
            "  --mtp-model FILE          DeepSeek-V4 Flash MTP sidecar GGUF\n"
            "  --index FILE              V100 pack-index.tsv\n"
            "  --prompt TEXT             prompt text\n"
            "  --prompt-file FILE        prompt file\n"
            "  --system TEXT             system prompt, default empty\n"
            "  --tokens N                greedy tokens to generate, default 1\n"
            "  --ctx N                   KV context tokens, default 1048576\n"
            "  --slots N                 configured admission slots, default 1\n"
            "  --active-microbatch N     active decode request slots, default 1\n"
            "  --queue-policy MODE       reject-busy or sequential, default reject-busy\n"
            "  --expected-token-hex HEX  require first generated token bytes\n"
            "  --json                    emit JSON\n"
            "  --open-only               open resident stages, print timing, and exit\n"
            "  --serial-open             open resident stages serially for benchmarking\n"
            "  --mtp-serving MODE        off or verify, default off\n"
            "  --mtp-top-k N             MTP draft top-k to report, default 5\n"
            "  --mtp-gpu N               MTP sidecar GPU, default 7\n"
            "  --mtp-reserve-mib N       MTP free-memory reserve, default 4096\n"
            "  --serve                   run a minimal HTTP endpoint\n"
            "                            GET /health, GET /v100/status, GET /metrics,\n"
            "                            POST /v100/selected-token\n"
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

static uint64_t parse_u64_arg_allow_zero(const char *s, const char *arg) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (!s || !*s || !end || *end) {
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
    opt.slots = 1;
    opt.active_microbatch = 1;
    opt.system = "";
    opt.host = "127.0.0.1";
    opt.port = 8000;
    opt.queue_policy = REPLAY_QUEUE_REJECT_BUSY;
    opt.mtp_top_k = 5;
    opt.mtp_gpu = 7;
    opt.mtp_reserve_mib = 4096;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "--model")) {
            opt.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-model")) {
            opt.mtp_model_path = need_arg(&i, argc, argv, arg);
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
        } else if (!strcmp(arg, "--slots")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v > 8) {
                fprintf(stderr, "ds4-v100-replay: --slots must be in [1,8]\n");
                exit(2);
            }
            opt.slots = (uint32_t)v;
        } else if (!strcmp(arg, "--active-microbatch")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v > 8) {
                fprintf(stderr, "ds4-v100-replay: --active-microbatch must be in [1,8]\n");
                exit(2);
            }
            opt.active_microbatch = (uint32_t)v;
        } else if (!strcmp(arg, "--queue-policy")) {
            const char *v = need_arg(&i, argc, argv, arg);
            if (!strcmp(v, "reject-busy") || !strcmp(v, "reject") || !strcmp(v, "busy")) {
                opt.queue_policy = REPLAY_QUEUE_REJECT_BUSY;
            } else if (!strcmp(v, "sequential") || !strcmp(v, "queue")) {
                opt.queue_policy = REPLAY_QUEUE_SEQUENTIAL;
            } else {
                fprintf(stderr, "ds4-v100-replay: --queue-policy must be reject-busy or sequential\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--expected-token-hex")) {
            opt.expected_hex = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--json")) {
            opt.json = true;
        } else if (!strcmp(arg, "--open-only")) {
            opt.open_only = true;
        } else if (!strcmp(arg, "--serial-open")) {
            opt.serial_open = true;
        } else if (!strcmp(arg, "--mtp-serving")) {
            const char *v = need_arg(&i, argc, argv, arg);
            if (!strcmp(v, "off") || !strcmp(v, "false") || !strcmp(v, "0")) {
                opt.mtp_serving = REPLAY_MTP_SERVING_OFF;
            } else if (!strcmp(v, "verify")) {
                opt.mtp_serving = REPLAY_MTP_SERVING_VERIFY;
            } else {
                fprintf(stderr, "ds4-v100-replay: --mtp-serving must be off or verify\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--mtp-top-k")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v < 2 || v > DS4_V100_MTP_FORWARD_MAX_TOPK) {
                fprintf(stderr,
                        "ds4-v100-replay: --mtp-top-k must be in [2,%d]\n",
                        DS4_V100_MTP_FORWARD_MAX_TOPK);
                exit(2);
            }
            opt.mtp_top_k = (uint32_t)v;
        } else if (!strcmp(arg, "--mtp-gpu")) {
            uint64_t v = parse_u64_arg_allow_zero(need_arg(&i, argc, argv, arg), arg);
            if (v > INT32_MAX) {
                fprintf(stderr, "ds4-v100-replay: invalid --mtp-gpu\n");
                exit(2);
            }
            opt.mtp_gpu = (int)v;
        } else if (!strcmp(arg, "--mtp-reserve-mib")) {
            uint64_t v = parse_u64_arg_allow_zero(need_arg(&i, argc, argv, arg), arg);
            if (v > INT32_MAX) {
                fprintf(stderr, "ds4-v100-replay: invalid --mtp-reserve-mib\n");
                exit(2);
            }
            opt.mtp_reserve_mib = (int)v;
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
    if (!opt.model_path || !opt.index_path ||
        (!opt.serve && !opt.open_only && !opt.prompt && !opt.prompt_file)) {
        usage(stderr);
        exit(2);
    }
    if (opt.open_only && opt.serve) {
        fprintf(stderr, "ds4-v100-replay: use --open-only or --serve, not both\n");
        exit(2);
    }
    if (opt.active_microbatch == 0 || opt.active_microbatch > opt.slots) {
        fprintf(stderr, "ds4-v100-replay: --active-microbatch must be in [1,slots]\n");
        exit(2);
    }
    if (opt.open_only && opt.mtp_serving != REPLAY_MTP_SERVING_OFF) {
        fprintf(stderr, "ds4-v100-replay: use --open-only or --mtp-serving, not both\n");
        exit(2);
    }
    if (opt.mtp_serving != REPLAY_MTP_SERVING_OFF && (!opt.mtp_model_path || !opt.mtp_model_path[0])) {
        fprintf(stderr, "ds4-v100-replay: --mtp-serving requires --mtp-model\n");
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

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static void mtp_result_init(replay_mtp_result *r) {
    if (!r) return;
    memset(r, 0, sizeof(*r));
    r->committed_token = UINT32_MAX;
    r->target_token = UINT32_MAX;
    r->draft_token = UINT32_MAX;
    for (uint32_t i = 0; i < DS4_V100_MTP_FORWARD_MAX_TOPK; i++) {
        r->draft_tokens[i] = UINT32_MAX;
    }
}

static void replay_mtp_service_close(replay_mtp_service *svc) {
    if (!svc) return;
    ds4_v100_mtp_forward_close(svc->forward);
    ds4_v100_mtp_sidecar_close(svc->sidecar);
    free(svc);
}

static int replay_mtp_service_open(replay_mtp_service **out,
                                   const replay_cli_options *opt,
                                   ds4_v100_replay *rt,
                                   char *err,
                                   size_t errlen) {
    if (out) *out = NULL;
    if (!out || !opt || !rt) return 1;
    if (opt->mtp_serving == REPLAY_MTP_SERVING_OFF) return 0;

    replay_mtp_service *svc = (replay_mtp_service *)calloc(1, sizeof(*svc));
    if (!svc) {
        snprintf(err, errlen, "failed to allocate MTP service");
        return 1;
    }
    svc->enabled = true;
    svc->top_k = opt->mtp_top_k;
    svc->gpu = opt->mtp_gpu;
    svc->reserve_mib = opt->mtp_reserve_mib;

    ds4_v100_mtp_sidecar_options mtp_opts;
    ds4_v100_mtp_sidecar_options_init(&mtp_opts);
    mtp_opts.mtp_path = opt->mtp_model_path;
    mtp_opts.gpu = opt->mtp_gpu;
    mtp_opts.require_device_arena = true;
    if (ds4_v100_mtp_sidecar_open(&svc->sidecar, &mtp_opts, NULL, err, errlen) != 0) {
        replay_mtp_service_close(svc);
        return 1;
    }
    ds4_gpu_arena *arena = ds4_v100_mtp_sidecar_arena(svc->sidecar);
    svc->sidecar_uploaded_bytes = ds4_v100_mtp_sidecar_uploaded_bytes(svc->sidecar);
    svc->sidecar_arena_bytes = ds4_gpu_arena_bytes(arena);
    const uint64_t reserve_bytes = (uint64_t)opt->mtp_reserve_mib * 1024ull * 1024ull;
    const uint64_t free_after_upload = ds4_gpu_arena_free_after_upload_bytes(arena);
    if (free_after_upload < reserve_bytes) {
        snprintf(err,
                 errlen,
                 "MTP sidecar free_after_upload %" PRIu64 " below reserve %" PRIu64,
                 free_after_upload,
                 reserve_bytes);
        replay_mtp_service_close(svc);
        return 1;
    }

    ds4_v100_context *ctx = NULL;
    ds4_v100_context_options ctx_opts;
    ds4_v100_context_options_init(&ctx_opts);
    ctx_opts.pack_index_path = opt->index_path;
    if (ds4_v100_context_open(&ctx, &ctx_opts, err, errlen) != 0) {
        replay_mtp_service_close(svc);
        return 1;
    }
    ds4_v100_tensor_binding output_weight;
    int rc = ds4_v100_context_output_head_binding(ctx, &output_weight, err, errlen);
    if (rc != 0) {
        ds4_v100_context_close(ctx);
        replay_mtp_service_close(svc);
        return 1;
    }
    svc->output_weight_bytes = output_weight.byte_length;
    if (ds4_v100_mtp_forward_open(&svc->forward,
                                  svc->sidecar,
                                  ds4_v100_replay_model_map(rt),
                                  ds4_v100_replay_model_size(rt),
                                  &output_weight,
                                  opt->mtp_gpu,
                                  err,
                                  errlen) != 0) {
        ds4_v100_context_close(ctx);
        replay_mtp_service_close(svc);
        return 1;
    }
    ds4_v100_context_close(ctx);
    *out = svc;
    return 0;
}

static int replay_mtp_service_run(replay_mtp_service *svc,
                                  ds4_v100_replay *rt,
                                  const ds4_v100_replay_output *outputs,
                                  uint32_t n_outputs,
                                  const ds4_v100_replay_counters *counters,
                                  replay_mtp_result *result,
                                  char *err,
                                  size_t errlen) {
    mtp_result_init(result);
    if (!svc || !svc->enabled || !result) return 0;
    result->enabled = true;
    result->top_k = svc->top_k;
    result->sidecar_uploaded_bytes = svc->sidecar_uploaded_bytes;
    result->sidecar_arena_bytes = svc->sidecar_arena_bytes;
    result->output_weight_bytes = svc->output_weight_bytes;
    svc->requests++;
    if (!rt || !outputs || !counters || n_outputs < 2) {
        result->skipped = true;
        snprintf(result->reason, sizeof(result->reason), "need_at_least_two_tokens");
        svc->skipped++;
        return 0;
    }
    const uint32_t committed_idx = n_outputs - 2u;
    const uint32_t target_idx = n_outputs - 1u;
    result->committed_token = outputs[committed_idx].token;
    result->committed_pos = counters->prompt_tokens + committed_idx;
    result->target_token = outputs[target_idx].token;
    result->target_logit = outputs[target_idx].logit;

    float embed[DS4_V100_MTP_FORWARD_N_EMBD];
    float hc[DS4_V100_MTP_FORWARD_HC_VALUES];
    if (ds4_v100_replay_read_token_embedding_f32(rt,
                                                 result->committed_token,
                                                 embed,
                                                 DS4_V100_MTP_FORWARD_N_EMBD,
                                                 err,
                                                 errlen) != 0 ||
        ds4_v100_replay_read_output_hc(rt,
                                       hc,
                                       sizeof(hc),
                                       err,
                                       errlen) != 0) {
        return 1;
    }
    ds4_v100_mtp_forward_report report;
    memset(&report, 0, sizeof(report));
    result->attempted = true;
    const double t0 = now_ms();
    if (ds4_v100_mtp_forward_run_host(svc->forward,
                                      embed,
                                      hc,
                                      result->committed_pos,
                                      svc->top_k,
                                      result->draft_tokens,
                                      result->draft_logits,
                                      &report,
                                      err,
                                      errlen) != 0) {
        return 1;
    }
    result->draft_ms = now_ms() - t0;
    result->draft_token = result->draft_tokens[0];
    result->draft_logit = result->draft_logits[0];
    result->raw_row = report.raw_row;
    result->n_raw = report.n_raw;
    result->output_vocab = report.output_vocab;
    result->output_weight_bytes = report.output_weight_bytes;
    result->free_after_output_upload_bytes = report.free_after_output_upload_bytes;
    if (result->free_after_output_upload_bytes) {
        const uint64_t reserve_bytes = (uint64_t)svc->reserve_mib * 1024ull * 1024ull;
        if (result->free_after_output_upload_bytes < reserve_bytes) {
            snprintf(err,
                     errlen,
                     "MTP output free_after_upload %" PRIu64 " below reserve %" PRIu64,
                     result->free_after_output_upload_bytes,
                     reserve_bytes);
            return 1;
        }
    }
    result->accepted = result->draft_token == result->target_token;
    svc->drafts++;
    if (result->accepted) svc->accepted++;
    else svc->rejected++;
    return 0;
}

static void print_mtp_json(FILE *fp, const replay_mtp_result *mtp) {
    if (!mtp) return;
    fprintf(fp, ",\"mtp\":{");
    fprintf(fp, "\"enabled\":%s,", mtp->enabled ? "true" : "false");
    fprintf(fp, "\"attempted\":%s,", mtp->attempted ? "true" : "false");
    fprintf(fp, "\"skipped\":%s,", mtp->skipped ? "true" : "false");
    fprintf(fp, "\"accepted\":%s,", mtp->accepted ? "true" : "false");
    fprintf(fp, "\"committed_token\":%" PRIu32 ",", mtp->committed_token);
    fprintf(fp, "\"committed_pos\":%" PRIu32 ",", mtp->committed_pos);
    fprintf(fp, "\"target_token\":%" PRIu32 ",", mtp->target_token);
    fprintf(fp, "\"draft_token\":%" PRIu32 ",", mtp->draft_token);
    fprintf(fp, "\"target_logit\":%.9g,", mtp->target_logit);
    fprintf(fp, "\"draft_logit\":%.9g,", mtp->draft_logit);
    fprintf(fp, "\"top_k\":%" PRIu32 ",", mtp->top_k);
    fprintf(fp, "\"draft_ms\":%.3f,", mtp->draft_ms);
    fprintf(fp, "\"raw_row\":%" PRIu32 ",", mtp->raw_row);
    fprintf(fp, "\"n_raw\":%" PRIu32 ",", mtp->n_raw);
    fprintf(fp, "\"output_vocab\":%" PRIu32 ",", mtp->output_vocab);
    fprintf(fp, "\"sidecar_uploaded_bytes\":%" PRIu64 ",", mtp->sidecar_uploaded_bytes);
    fprintf(fp, "\"sidecar_arena_bytes\":%" PRIu64 ",", mtp->sidecar_arena_bytes);
    fprintf(fp, "\"output_weight_bytes\":%" PRIu64 ",", mtp->output_weight_bytes);
    fprintf(fp, "\"free_after_output_upload_bytes\":%" PRIu64 ",", mtp->free_after_output_upload_bytes);
    fprintf(fp, "\"reason\":\"");
    json_escape(fp, mtp->reason, strlen(mtp->reason));
    fprintf(fp, "\",\"draft_topk\":[");
    const uint32_t n_top = mtp->attempted ? mtp->top_k : 0;
    for (uint32_t i = 0; i < n_top && i < DS4_V100_MTP_FORWARD_MAX_TOPK; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp,
                "{\"id\":%" PRIu32 ",\"logit\":%.9g}",
                mtp->draft_tokens[i],
                mtp->draft_logits[i]);
    }
    fprintf(fp, "]}");
}

static void print_json_fp(FILE *fp,
                          const ds4_v100_replay_output *outputs,
                          uint32_t n_outputs,
                          const ds4_v100_replay_counters *c,
                          const replay_mtp_result *mtp) {
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
    fprintf(fp, "]}");
    print_mtp_json(fp, mtp);
    fprintf(fp, "}\n");
}

static void print_open_json(const ds4_v100_replay_counters *c, bool serial_open) {
    printf("{\"open_only\":true,\"open_mode\":\"%s\",\"timing_ms\":{",
           serial_open ? "serial" : "parallel");
    printf("\"open_total\":%.3f,\"open_stage\":[", c->open_total_ms);
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) printf(",");
        printf("%.3f", c->open_ms[i]);
    }
    printf("]}}\n");
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

static void http_ok_json_begin(int fd) {
    dprintf(fd,
            "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: application/json\r\n\r\n");
}

static void write_status_json(FILE *fp,
                              const replay_cli_options *opt,
                              const replay_mtp_service *mtp,
                              const replay_server_stats *stats) {
    const bool mtp_enabled = mtp && mtp->enabled;
    const uint64_t served = stats ? stats->accepted_connections : 0;
    char mode[64] = "base";
    format_mode(mode, sizeof(mode), mtp_enabled, opt);
    fprintf(fp, "{");
    fprintf(fp, "\"service\":\"ds4-v100-replay\",");
    fprintf(fp, "\"status\":\"ok\",");
    fprintf(fp, "\"mode\":\"%s\",", mode);
    fprintf(fp, "\"readiness_level\":%d,", mtp_enabled ? 3 : 2);
    fprintf(fp, "\"mtp_enabled\":%s,", mtp_enabled ? "true" : "false");
    fprintf(fp, "\"model_path\":\"");
    json_escape(fp, opt->model_path ? opt->model_path : "", strlen(opt->model_path ? opt->model_path : ""));
    fprintf(fp, "\",\"pack_index_path\":\"");
    json_escape(fp, opt->index_path ? opt->index_path : "", strlen(opt->index_path ? opt->index_path : ""));
    fprintf(fp, "\",\"ctx_tokens\":%" PRIu64 ",", opt->ctx);
    fprintf(fp, "\"default_tokens\":%" PRIu32 ",", opt->tokens);
    fprintf(fp, "\"max_tokens\":%u,", DS4_V100_REPLAY_MAX_TOKENS);
    fprintf(fp,
            "\"limits\":{\"slots\":%" PRIu32 ",\"configured_slots\":%" PRIu32
            ",\"active_slots\":%" PRIu32 ",\"active_microbatch\":%" PRIu32
            ",\"concurrent_requests\":%" PRIu32
            ",\"queue_policy\":\"%s\",\"scheduler_slots_ready\":true,"
            "\"tensor_batched_slots\":%s,\"sequential_requests\":%s,"
            "\"streaming\":false,\"external_exposure\":false,"
            "\"speculative_serving\":%s},",
            opt->slots,
            opt->slots,
            opt->active_microbatch,
            opt->active_microbatch,
            opt->active_microbatch,
            queue_policy_name(opt->queue_policy),
            opt->active_microbatch > 1 ? "true" : "false",
            opt->queue_policy == REPLAY_QUEUE_SEQUENTIAL ? "true" : "false",
            mtp_enabled ? "true" : "false");
    fprintf(fp, "\"served_requests\":%" PRIu64 ",", served);
    fprintf(fp, "\"generation_requests\":%" PRIu64 ",", stats ? stats->generation_requests : 0);
    fprintf(fp,
            "\"tensor_batched_groups\":%" PRIu64
            ",\"tensor_batched_requests\":%" PRIu64
            ",\"tensor_batched_tokens\":%" PRIu64 ",",
            stats ? stats->tensor_batched_groups : 0,
            stats ? stats->tensor_batched_requests : 0,
            stats ? stats->tensor_batched_tokens : 0);
    fprintf(fp, "\"rejected_requests\":%" PRIu64, stats ? stats->rejected_requests : 0);
    if (mtp_enabled) {
        fprintf(fp, ",\"mtp\":{");
        fprintf(fp, "\"serving_mode\":\"verify\",");
        fprintf(fp, "\"top_k\":%" PRIu32 ",", mtp->top_k);
        fprintf(fp, "\"gpu\":%d,", mtp->gpu);
        fprintf(fp, "\"reserve_mib\":%d,", mtp->reserve_mib);
        fprintf(fp, "\"sidecar_uploaded_bytes\":%" PRIu64 ",", mtp->sidecar_uploaded_bytes);
        fprintf(fp, "\"sidecar_arena_bytes\":%" PRIu64 ",", mtp->sidecar_arena_bytes);
        fprintf(fp, "\"output_weight_bytes\":%" PRIu64 ",", mtp->output_weight_bytes);
        fprintf(fp, "\"requests\":%" PRIu64 ",", mtp->requests);
        fprintf(fp, "\"drafts\":%" PRIu64 ",", mtp->drafts);
        fprintf(fp, "\"accepted\":%" PRIu64 ",", mtp->accepted);
        fprintf(fp, "\"rejected\":%" PRIu64 ",", mtp->rejected);
        fprintf(fp, "\"skipped\":%" PRIu64, mtp->skipped);
        fprintf(fp, "}");
    }
    fprintf(fp, "}\n");
}

static void write_metrics_text(FILE *fp,
                               const replay_cli_options *opt,
                               const replay_mtp_service *mtp,
                               const replay_server_stats *stats) {
    const bool mtp_enabled = mtp && mtp->enabled;
    const uint64_t served = stats ? stats->accepted_connections : 0;
    fprintf(fp, "# HELP ds4_v100_readiness_level Deployment readiness level exposed by the replay service.\n");
    fprintf(fp, "# TYPE ds4_v100_readiness_level gauge\n");
    fprintf(fp, "ds4_v100_readiness_level %d\n", mtp_enabled ? 3 : 2);
    fprintf(fp, "# HELP ds4_v100_served_requests HTTP requests accepted by this process.\n");
    fprintf(fp, "# TYPE ds4_v100_served_requests counter\n");
    fprintf(fp, "ds4_v100_served_requests %" PRIu64 "\n", served);
    fprintf(fp, "# HELP ds4_v100_generation_requests_total Generation requests accepted by the scheduler.\n");
    fprintf(fp, "# TYPE ds4_v100_generation_requests_total counter\n");
    fprintf(fp, "ds4_v100_generation_requests_total %" PRIu64 "\n", stats ? stats->generation_requests : 0);
    fprintf(fp, "# HELP ds4_v100_tensor_batched_groups_total Same-token-count tensor batch groups executed by the scheduler.\n");
    fprintf(fp, "# TYPE ds4_v100_tensor_batched_groups_total counter\n");
    fprintf(fp, "ds4_v100_tensor_batched_groups_total %" PRIu64 "\n", stats ? stats->tensor_batched_groups : 0);
    fprintf(fp, "# HELP ds4_v100_tensor_batched_requests_total Requests served through tensor batch groups.\n");
    fprintf(fp, "# TYPE ds4_v100_tensor_batched_requests_total counter\n");
    fprintf(fp, "ds4_v100_tensor_batched_requests_total %" PRIu64 "\n", stats ? stats->tensor_batched_requests : 0);
    fprintf(fp, "# HELP ds4_v100_tensor_batched_tokens_total Generated tokens served through tensor batch groups.\n");
    fprintf(fp, "# TYPE ds4_v100_tensor_batched_tokens_total counter\n");
    fprintf(fp, "ds4_v100_tensor_batched_tokens_total %" PRIu64 "\n", stats ? stats->tensor_batched_tokens : 0);
    fprintf(fp, "# HELP ds4_v100_rejected_requests_total HTTP generation requests rejected by admission policy.\n");
    fprintf(fp, "# TYPE ds4_v100_rejected_requests_total counter\n");
    fprintf(fp, "ds4_v100_rejected_requests_total %" PRIu64 "\n", stats ? stats->rejected_requests : 0);
    fprintf(fp, "# HELP ds4_v100_rejected_busy_total HTTP generation requests rejected because the scheduler was busy.\n");
    fprintf(fp, "# TYPE ds4_v100_rejected_busy_total counter\n");
    fprintf(fp, "ds4_v100_rejected_busy_total %" PRIu64 "\n", stats ? stats->rejected_busy : 0);
    fprintf(fp, "# HELP ds4_v100_rejected_context_total HTTP generation requests rejected for exceeding context.\n");
    fprintf(fp, "# TYPE ds4_v100_rejected_context_total counter\n");
    fprintf(fp, "ds4_v100_rejected_context_total %" PRIu64 "\n", stats ? stats->rejected_context : 0);
    fprintf(fp, "# HELP ds4_v100_rejected_bad_request_total HTTP generation requests rejected for malformed input.\n");
    fprintf(fp, "# TYPE ds4_v100_rejected_bad_request_total counter\n");
    fprintf(fp, "ds4_v100_rejected_bad_request_total %" PRIu64 "\n", stats ? stats->rejected_bad_request : 0);
    fprintf(fp, "# HELP ds4_v100_ctx_tokens Configured KV context tokens per slot.\n");
    fprintf(fp, "# TYPE ds4_v100_ctx_tokens gauge\n");
    fprintf(fp, "ds4_v100_ctx_tokens %" PRIu64 "\n", opt->ctx);
    fprintf(fp, "# HELP ds4_v100_default_tokens Default generated tokens per request.\n");
    fprintf(fp, "# TYPE ds4_v100_default_tokens gauge\n");
    fprintf(fp, "ds4_v100_default_tokens %" PRIu32 "\n", opt->tokens);
    fprintf(fp, "# HELP ds4_v100_max_tokens Maximum generated tokens accepted by the appliance endpoint.\n");
    fprintf(fp, "# TYPE ds4_v100_max_tokens gauge\n");
    fprintf(fp, "ds4_v100_max_tokens %u\n", DS4_V100_REPLAY_MAX_TOKENS);
    fprintf(fp, "# HELP ds4_v100_configured_slots Configured admission slots.\n");
    fprintf(fp, "# TYPE ds4_v100_configured_slots gauge\n");
    fprintf(fp, "ds4_v100_configured_slots %" PRIu32 "\n", opt->slots);
    fprintf(fp, "# HELP ds4_v100_active_microbatch Active decode requests supported by this process.\n");
    fprintf(fp, "# TYPE ds4_v100_active_microbatch gauge\n");
    fprintf(fp, "ds4_v100_active_microbatch %" PRIu32 "\n", opt->active_microbatch);
    fprintf(fp, "# HELP ds4_v100_active_slots Active slots scheduled concurrently by this process.\n");
    fprintf(fp, "# TYPE ds4_v100_active_slots gauge\n");
    fprintf(fp, "ds4_v100_active_slots %" PRIu32 "\n", opt->active_microbatch);
    fprintf(fp, "# HELP ds4_v100_concurrent_request_capacity Concurrent generation request capacity.\n");
    fprintf(fp, "# TYPE ds4_v100_concurrent_request_capacity gauge\n");
    fprintf(fp, "ds4_v100_concurrent_request_capacity %" PRIu32 "\n", opt->active_microbatch);
    fprintf(fp, "# HELP ds4_v100_scheduler_slots_ready Whether true device-resident multi-slot scheduling is implemented.\n");
    fprintf(fp, "# TYPE ds4_v100_scheduler_slots_ready gauge\n");
    fprintf(fp, "ds4_v100_scheduler_slots_ready 1\n");
    fprintf(fp, "# HELP ds4_v100_mtp_enabled Whether speculative MTP serving is enabled.\n");
    fprintf(fp, "# TYPE ds4_v100_mtp_enabled gauge\n");
    fprintf(fp, "ds4_v100_mtp_enabled %d\n", mtp_enabled ? 1 : 0);
    fprintf(fp, "# HELP ds4_v100_mtp_requests_total Requests evaluated by the MTP verify service.\n");
    fprintf(fp, "# TYPE ds4_v100_mtp_requests_total counter\n");
    fprintf(fp, "ds4_v100_mtp_requests_total %" PRIu64 "\n", mtp_enabled ? mtp->requests : 0);
    fprintf(fp, "# HELP ds4_v100_mtp_drafts_total MTP draft attempts completed.\n");
    fprintf(fp, "# TYPE ds4_v100_mtp_drafts_total counter\n");
    fprintf(fp, "ds4_v100_mtp_drafts_total %" PRIu64 "\n", mtp_enabled ? mtp->drafts : 0);
    fprintf(fp, "# HELP ds4_v100_mtp_accepted_total MTP drafts matching the base target token.\n");
    fprintf(fp, "# TYPE ds4_v100_mtp_accepted_total counter\n");
    fprintf(fp, "ds4_v100_mtp_accepted_total %" PRIu64 "\n", mtp_enabled ? mtp->accepted : 0);
    fprintf(fp, "# HELP ds4_v100_mtp_rejected_total MTP drafts rejected by exact target-token comparison.\n");
    fprintf(fp, "# TYPE ds4_v100_mtp_rejected_total counter\n");
    fprintf(fp, "ds4_v100_mtp_rejected_total %" PRIu64 "\n", mtp_enabled ? mtp->rejected : 0);
    fprintf(fp, "# HELP ds4_v100_mtp_skipped_total Requests where MTP verify was skipped.\n");
    fprintf(fp, "# TYPE ds4_v100_mtp_skipped_total counter\n");
    fprintf(fp, "ds4_v100_mtp_skipped_total %" PRIu64 "\n", mtp_enabled ? mtp->skipped : 0);
}

static int handle_http_request(int fd, replay_server_state *state) {
    if (!state || !state->opt || !state->rt) {
        http_error(fd, 500, "server_state_missing");
        return 1;
    }
    ds4_v100_replay *rt = state->rt;
    const replay_cli_options *opt = state->opt;
    replay_mtp_service *mtp = state->mtp;
    size_t req_len = 0;
    char *req = read_http_request(fd, &req_len);
    (void)req_len;
    if (!req) {
        server_stats_add(state, 0, 0, 1, 0, 0, 1);
        http_error(fd, 400, "bad_request");
        return 1;
    }
    char method[8] = {0};
    char path[128] = {0};
    if (sscanf(req, "%7s %127s", method, path) != 2) {
        server_stats_add(state, 0, 0, 1, 0, 0, 1);
        http_error(fd, 400, "bad_request");
        free(req);
        return 1;
    }
    if (!strcmp(method, "GET") && !strcmp(path, "/health")) {
        http_ok_json_begin(fd);
        dprintf(fd, "{\"status\":\"ok\"}\n");
        free(req);
        return 0;
    }
    if (!strcmp(method, "GET") &&
        (!strcmp(path, "/status") || !strcmp(path, "/v100/status"))) {
        http_ok_json_begin(fd);
        FILE *fp = fdopen(dup(fd), "w");
        if (fp) {
            replay_server_stats stats;
            server_stats_snapshot(state, &stats);
            write_status_json(fp, opt, mtp, &stats);
            fclose(fp);
        }
        free(req);
        return 0;
    }
    if (!strcmp(method, "GET") && !strcmp(path, "/metrics")) {
        dprintf(fd, "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: text/plain; version=0.0.4\r\n\r\n");
        FILE *fp = fdopen(dup(fd), "w");
        if (fp) {
            replay_server_stats stats;
            server_stats_snapshot(state, &stats);
            write_metrics_text(fp, opt, mtp, &stats);
            fclose(fp);
        }
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
        server_stats_add(state, 0, 0, 1, 0, 0, 1);
        http_error(fd, 400, "missing_body");
        free(req);
        return 1;
    }
    body += 4;
    char *prompt = json_get_string(body, "prompt");
    if (!prompt) {
        server_stats_add(state, 0, 0, 1, 0, 0, 1);
        http_error(fd, 400, "missing_prompt");
        free(req);
        return 1;
    }
    uint32_t tokens = opt->tokens;
    (void)json_get_u32(body, "tokens", &tokens);
    if (tokens == 0 || tokens > DS4_V100_REPLAY_MAX_TOKENS) {
        free(prompt);
        server_stats_add(state, 0, 0, 1, 0, 0, 1);
        http_error(fd, 400, "bad_tokens");
        free(req);
        return 1;
    }

    if (acquire_generation_slot(state) != 0) {
        free(prompt);
        server_stats_add(state, 0, 0, 1, 1, 0, 0);
        http_error(fd, 429, "busy");
        free(req);
        return 1;
    }

    replay_pending_generation pending;
    memset(&pending, 0, sizeof(pending));
    ds4_v100_replay_encode_prompt(rt, opt->system, prompt, DS4_THINK_NONE, &pending.prompt_tokens);
    if (opt->ctx && (uint64_t)pending.prompt_tokens.len + tokens > opt->ctx) {
        ds4_tokens_free(&pending.prompt_tokens);
        free(prompt);
        server_stats_add(state, 0, 0, 1, 0, 1, 0);
        http_error(fd, 413, "context_exceeded");
        free(req);
        release_generation_slot(state);
        return 1;
    }
    pending.tokens = tokens;
    pending_enqueue(state, &pending);

    bool pending_done = false;
    for (uint32_t pass = 0; pass < 2; pass++) {
        pthread_mutex_lock(&state->generation_mu);
        if (!pending.done) {
            if (process_pending_generation_batch(state) != 0) {
                pending_mark_done(&pending, 1, "generation_batch_failed");
                pending_remove(state, &pending);
            }
        }
        pending_done = pending.done;
        pthread_mutex_unlock(&state->generation_mu);
        if (pending_done) break;
    }
    if (!pending_done) {
        pending_mark_done(&pending, 1, "pending_generation_not_completed");
        pending_remove(state, &pending);
    }

    char err[512] = {0};
    if (pending.err[0]) snprintf(err, sizeof(err), "%s", pending.err);
    int rc = pending.rc;
    if (!rc) server_stats_add(state, 0, 1, 0, 0, 0, 0);
    replay_mtp_result mtp_result;
    const replay_mtp_result *mtp_json = NULL;
    if (rc) {
        http_error(fd, 500, err[0] ? err : "generation_failed");
    } else {
        if (mtp && mtp->enabled) {
            if (replay_mtp_service_run(mtp,
                                       rt,
                                       pending.outputs,
                                       pending.n_outputs,
                                       &pending.counters,
                                       &mtp_result,
                                       err,
                                       sizeof(err)) != 0) {
                http_error(fd, 500, err[0] ? err : "mtp_verify_failed");
                rc = 1;
            } else {
                mtp_json = &mtp_result;
            }
        }
    }
    if (!rc) {
        http_ok_json_begin(fd);
        FILE *fp = fdopen(dup(fd), "w");
        if (fp) {
            print_json_fp(fp, pending.outputs, pending.n_outputs, &pending.counters, mtp_json);
            fclose(fp);
        }
    }
    for (uint32_t i = 0; i < pending.n_outputs; i++) ds4_v100_replay_output_free(&pending.outputs[i]);
    ds4_tokens_free(&pending.prompt_tokens);
    free(prompt);
    free(req);
    release_generation_slot(state);
    return rc;
}

static void *request_worker_main(void *arg) {
    replay_request_worker *w = (replay_request_worker *)arg;
    if (!w) return NULL;
    replay_server_state *state = w->state;
    const int fd = w->fd;
    free(w);
    (void)handle_http_request(fd, state);
    close(fd);
    pthread_mutex_lock(&state->handlers_mu);
    if (state->active_handlers > 0) state->active_handlers--;
    pthread_cond_signal(&state->handlers_done);
    pthread_mutex_unlock(&state->handlers_mu);
    return NULL;
}

static int run_server(const replay_cli_options *opt,
                      ds4_v100_replay *rt,
                      replay_mtp_service *mtp) {
    replay_server_state state;
    memset(&state, 0, sizeof(state));
    state.opt = opt;
    state.rt = rt;
    state.mtp = mtp;
    pthread_mutex_init(&state.generation_mu, NULL);
    pthread_mutex_init(&state.queue_mu, NULL);
    pthread_cond_init(&state.queue_done, NULL);
    pthread_mutex_init(&state.stats_mu, NULL);
    pthread_mutex_init(&state.handlers_mu, NULL);
    pthread_cond_init(&state.handlers_done, NULL);
    pthread_mutex_init(&state.pending_mu, NULL);
    pthread_cond_init(&state.pending_cv, NULL);

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("ds4-v100-replay: socket");
        pthread_mutex_destroy(&state.queue_mu);
        pthread_cond_destroy(&state.queue_done);
        pthread_mutex_destroy(&state.generation_mu);
        pthread_mutex_destroy(&state.stats_mu);
        pthread_mutex_destroy(&state.handlers_mu);
        pthread_cond_destroy(&state.handlers_done);
        pthread_mutex_destroy(&state.pending_mu);
        pthread_cond_destroy(&state.pending_cv);
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
        pthread_mutex_destroy(&state.queue_mu);
        pthread_cond_destroy(&state.queue_done);
        pthread_mutex_destroy(&state.generation_mu);
        pthread_mutex_destroy(&state.stats_mu);
        pthread_mutex_destroy(&state.handlers_mu);
        pthread_cond_destroy(&state.handlers_done);
        pthread_mutex_destroy(&state.pending_mu);
        pthread_cond_destroy(&state.pending_cv);
        return 2;
    }
    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(listen_fd, 8) != 0) {
        perror("ds4-v100-replay: bind/listen");
        close(listen_fd);
        pthread_mutex_destroy(&state.queue_mu);
        pthread_cond_destroy(&state.queue_done);
        pthread_mutex_destroy(&state.generation_mu);
        pthread_mutex_destroy(&state.stats_mu);
        pthread_mutex_destroy(&state.handlers_mu);
        pthread_cond_destroy(&state.handlers_done);
        pthread_mutex_destroy(&state.pending_mu);
        pthread_cond_destroy(&state.pending_cv);
        return 1;
    }
    fprintf(stderr,
            "ds4-v100-replay: serving http://%s:%d/v100/selected-token\n",
            opt->host,
            opt->port);
    uint32_t accepted = 0;
    while (opt->max_requests == 0 || accepted < opt->max_requests) {
        int fd = accept(listen_fd, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            perror("ds4-v100-replay: accept");
            close(listen_fd);
            pthread_mutex_destroy(&state.queue_mu);
            pthread_cond_destroy(&state.queue_done);
            pthread_mutex_destroy(&state.generation_mu);
            pthread_mutex_destroy(&state.stats_mu);
            pthread_mutex_destroy(&state.handlers_mu);
            pthread_cond_destroy(&state.handlers_done);
            pthread_mutex_destroy(&state.pending_mu);
            pthread_cond_destroy(&state.pending_cv);
            return 1;
        }
        server_stats_add(&state, 1, 0, 0, 0, 0, 0);
        accepted++;

        replay_request_worker *worker = (replay_request_worker *)calloc(1, sizeof(*worker));
        if (!worker) {
            (void)handle_http_request(fd, &state);
            close(fd);
            continue;
        }
        worker->state = &state;
        worker->fd = fd;
        pthread_t thread;
        pthread_mutex_lock(&state.handlers_mu);
        state.active_handlers++;
        pthread_mutex_unlock(&state.handlers_mu);
        if (pthread_create(&thread, NULL, request_worker_main, worker) != 0) {
            pthread_mutex_lock(&state.handlers_mu);
            if (state.active_handlers > 0) state.active_handlers--;
            pthread_cond_signal(&state.handlers_done);
            pthread_mutex_unlock(&state.handlers_mu);
            free(worker);
            (void)handle_http_request(fd, &state);
            close(fd);
            continue;
        }
        pthread_detach(thread);
    }
    close(listen_fd);
    pthread_mutex_lock(&state.handlers_mu);
    while (state.active_handlers != 0) {
        pthread_cond_wait(&state.handlers_done, &state.handlers_mu);
    }
    pthread_mutex_unlock(&state.handlers_mu);

    pthread_mutex_destroy(&state.queue_mu);
    pthread_cond_destroy(&state.queue_done);
    pthread_mutex_destroy(&state.generation_mu);
    pthread_mutex_destroy(&state.stats_mu);
    pthread_mutex_destroy(&state.handlers_mu);
    pthread_cond_destroy(&state.handlers_done);
    pthread_mutex_destroy(&state.pending_mu);
    pthread_cond_destroy(&state.pending_cv);
    return 0;
}

int main(int argc, char **argv) {
    replay_cli_options opt = parse_options(argc, argv);
    char *prompt_owned = NULL;
    const char *prompt_text = NULL;
    if (!opt.serve && !opt.open_only) {
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
    ropts.kv_active_slots = opt.slots;
    ropts.serial_open = opt.serial_open;

    char err[512] = {0};
    ds4_v100_replay *rt = NULL;
    if (ds4_v100_replay_open(&rt, &ropts, err, sizeof(err))) {
        fprintf(stderr, "ds4-v100-replay: %s\n", err[0] ? err : "open failed");
        free(expected);
        free(prompt_owned);
        return 1;
    }

    if (opt.open_only) {
        ds4_v100_replay_counters counters;
        ds4_v100_replay_open_counters(rt, &counters);
        if (opt.json) {
            print_open_json(&counters, opt.serial_open);
        } else {
            printf("open_mode\t%s\n", opt.serial_open ? "serial" : "parallel");
            printf("open_total_ms\t%.3f\n", counters.open_total_ms);
            for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
                printf("open_stage_ms\t%d\t%.3f\n", i, counters.open_ms[i]);
            }
        }
        ds4_v100_replay_close(rt);
        free(expected);
        return 0;
    }

    replay_mtp_service *mtp = NULL;
    if (replay_mtp_service_open(&mtp, &opt, rt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-v100-replay: %s\n", err[0] ? err : "MTP service open failed");
        ds4_v100_replay_close(rt);
        free(expected);
        free(prompt_owned);
        return 1;
    }

    if (opt.serve) {
        int rc = run_server(&opt, rt, mtp);
        replay_mtp_service_close(mtp);
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
    replay_mtp_result mtp_result;
    const replay_mtp_result *mtp_json = NULL;
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
    if (rc == 0 && mtp && mtp->enabled) {
        if (replay_mtp_service_run(mtp,
                                   rt,
                                   outputs,
                                   n_outputs,
                                   &counters,
                                   &mtp_result,
                                   err,
                                   sizeof(err)) != 0) {
            fprintf(stderr, "ds4-v100-replay: %s\n", err[0] ? err : "MTP verify failed");
            rc = 1;
        } else {
            mtp_json = &mtp_result;
        }
    }

    if (rc == 0) {
        if (opt.json) {
            print_json_fp(stdout, outputs, n_outputs, &counters, mtp_json);
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
            if (mtp_json) {
                printf("ds4-v100-replay: mtp_attempted=%s mtp_accepted=%s committed=%" PRIu32
                       " target=%" PRIu32 " draft=%" PRIu32 " draft_ms=%.3f\n",
                       mtp_json->attempted ? "true" : "false",
                       mtp_json->accepted ? "true" : "false",
                       mtp_json->committed_token,
                       mtp_json->target_token,
                       mtp_json->draft_token,
                       mtp_json->draft_ms);
            }
        }
    }

    for (uint32_t i = 0; i < n_outputs; i++) ds4_v100_replay_output_free(&outputs[i]);
    ds4_tokens_free(&prompt);
    replay_mtp_service_close(mtp);
    ds4_v100_replay_close(rt);
    free(expected);
    free(prompt_owned);
    return rc;
}
