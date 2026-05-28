#include "engine/replay.h"
#include "engine/context.h"
#include "engine/mtp.h"
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
    REPLAY_MTP_SERVING_COMMIT = 2,
} replay_mtp_serving_mode;

typedef enum {
    REPLAY_QUEUE_REJECT_BUSY = 0,
    REPLAY_QUEUE_SEQUENTIAL = 1,
} replay_queue_policy;

typedef struct {
    const char *model_path;
    const char *mtp_model_path;
    const char *index_path;
    const char *turbomind_index_path;
    const char *shard_dir;
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
    uint32_t microbatch_wait_us;
    uint32_t synthetic_prompt_token;
    uint32_t synthetic_prompt_len;
    uint32_t prompt_token_limit;
    uint32_t target_block_smoke;
    uint32_t mtp_draft_block_smoke;
    uint32_t mtp_block2_commit_smoke;
    uint32_t reset_parity_smoke;
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
    bool profile_decode;
    bool wavefront_decode;
    bool async_pipeline_decode;
    bool async_handoff;
    bool async_event_handoff;
    bool startup_warmup;
    bool cuda_profiler_window;
    bool synthetic_prompt_token_set;
    ds4_replay_async_pipeline_mode async_pipeline_mode;
} replay_cli_options;

typedef struct {
    bool enabled;
    bool commit_mode;
    bool attempted;
    bool accepted;
    bool skipped;
    bool commit_applied;
    uint32_t committed_token;
    uint32_t committed_pos;
    uint32_t target_token;
    uint32_t draft_token;
    uint32_t top_k;
    uint32_t attempts;
    uint32_t accepted_count;
    uint32_t rejected_count;
    uint32_t commit_count;
    uint32_t draft_tokens_proposed;
    uint32_t draft_tokens_accepted;
    uint32_t accepted_prefix_len;
    uint32_t target_tokens_verified;
    uint32_t target_forwards;
    uint32_t effective_output_tokens;
    uint32_t speculative_saves;
    uint32_t raw_row;
    uint32_t n_raw;
    uint32_t output_vocab;
    uint32_t draft_tokens[DS4_V100_MTP_FORWARD_MAX_TOPK];
    float draft_logits[DS4_V100_MTP_FORWARD_MAX_TOPK];
    float target_logit;
    float draft_logit;
    double draft_ms;
    double draft_total_ms;
    uint64_t sidecar_uploaded_bytes;
    uint64_t sidecar_arena_bytes;
    uint64_t output_weight_bytes;
    uint64_t free_after_output_upload_bytes;
    uint64_t scratch_device_bytes;
    uint64_t scratch_host_bytes;
    uint64_t forward_run_count;
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
    uint64_t committed;
    ds4_mtp_sidecar *sidecar;
    ds4_mtp_forward *forward;
    uint64_t sidecar_uploaded_bytes;
    uint64_t sidecar_arena_bytes;
    uint64_t output_weight_bytes;
    pthread_mutex_t mu;
    bool mu_ready;
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
    ds4_replay *rt;
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
    ds4_replay_output outputs[DS4_V100_REPLAY_MAX_TOKENS];
    uint32_t n_outputs;
    ds4_replay_counters counters;
    replay_mtp_result mtp_result;
    int rc;
    char err[512];
    bool mtp_result_ready;
    bool done;
};

typedef struct {
    replay_pending_generation *items[DS4_V100_SCHED_MAX_SLOTS];
    uint32_t count;
} replay_pending_batch;

static void mtp_result_init(replay_mtp_result *r);
static int replay_generate_mtp_commit_one_slot(replay_mtp_service *svc,
                                               ds4_replay *rt,
                                               const ds4_tokens *prompt,
                                               uint32_t max_tokens,
                                               ds4_replay_output *outputs,
                                               uint32_t output_cap,
                                               uint32_t *out_count,
                                               ds4_replay_counters *counters,
                                               replay_mtp_result *mtp_result,
                                               char *err,
                                               size_t errlen);
static int replay_mtp_service_run_slot(replay_mtp_service *svc,
                                       ds4_replay *rt,
                                       uint32_t hc_slot,
                                       const ds4_replay_output *outputs,
                                       uint32_t n_outputs,
                                       const ds4_replay_counters *counters,
                                       replay_mtp_result *result,
                                       char *err,
                                       size_t errlen);
static int replay_output_copy(ds4_replay_output *dst,
                              const ds4_replay_output *src,
                              char *err,
                              size_t errlen);

static const char *queue_policy_name(replay_queue_policy p) {
    switch (p) {
    case REPLAY_QUEUE_SEQUENTIAL: return "sequential";
    case REPLAY_QUEUE_REJECT_BUSY:
    default: return "reject-busy";
    }
}

static const char *mtp_serving_mode_name(replay_mtp_serving_mode mode) {
    switch (mode) {
    case REPLAY_MTP_SERVING_VERIFY: return "verify";
    case REPLAY_MTP_SERVING_COMMIT: return "commit";
    case REPLAY_MTP_SERVING_OFF:
    default: return "off";
    }
}

static const char *async_pipeline_mode_name(ds4_replay_async_pipeline_mode mode) {
    switch (mode) {
    case DS4_V100_REPLAY_ASYNC_PIPELINE_PERSISTENT:
        return "persistent";
    case DS4_V100_REPLAY_ASYNC_PIPELINE_PER_STEP:
        return "per-step";
    case DS4_V100_REPLAY_ASYNC_PIPELINE_MAILBOX:
        return "mailbox";
    case DS4_V100_REPLAY_ASYNC_PIPELINE_OFF:
    default:
        return "off";
    }
}

static void format_mode(char *dst, size_t dst_len, bool mtp_enabled, const replay_cli_options *opt) {
    if (!dst || dst_len == 0 || !opt) {
        return;
    }
    if (mtp_enabled) {
        if (opt->slots <= 1) {
            snprintf(dst,
                     dst_len,
                     opt->mtp_serving == REPLAY_MTP_SERVING_COMMIT
                         ? "mtp_commit_one_slot"
                         : "mtp_verify_one_slot");
            return;
        }
        snprintf(dst,
                 dst_len,
                 opt->mtp_serving == REPLAY_MTP_SERVING_COMMIT
                     ? "mtp_commit_slots_%u"
                     : "mtp_verify_slots_%u",
                 opt->slots);
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
                                        bool force_single) {
    if (!state || cap <= 1 || force_single) return;
    const uint32_t wait_us = state->opt ? state->opt->microbatch_wait_us : 0;
    if (wait_us == 0) return;

    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct timespec deadline;
    deadline.tv_sec = tv.tv_sec;
    deadline.tv_sec += (time_t)(wait_us / 1000000u);
    deadline.tv_nsec =
        (long)tv.tv_usec * 1000L + (long)(wait_us % 1000000u) * 1000L;
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

static bool server_profiler_start_if_requested(replay_server_state *state) {
    if (!state || !state->opt || !state->opt->cuda_profiler_window) return false;
    if (!ds4_gpu_profiler_start()) {
        fprintf(stderr, "ds4-v100-replay: cuda profiler start failed\n");
        return false;
    }
    return true;
}

static void server_profiler_stop_if_active(bool *active) {
    if (!active || !*active) return;
    if (!ds4_gpu_profiler_stop()) {
        fprintf(stderr, "ds4-v100-replay: cuda profiler stop failed\n");
    }
    *active = false;
}

static int process_pending_generation_batch(replay_server_state *state) {
    if (!state || !state->opt || !state->rt) return 1;
    uint32_t cap = state->opt->active_microbatch ? state->opt->active_microbatch : 1;
    if (cap > DS4_V100_SCHED_MAX_SLOTS) cap = DS4_V100_SCHED_MAX_SLOTS;
    const bool mtp_enabled = state->mtp && state->mtp->enabled;
    const bool mtp_commit_enabled =
        mtp_enabled && state->opt->mtp_serving == REPLAY_MTP_SERVING_COMMIT;
    const bool mtp_verify_enabled =
        mtp_enabled && state->opt->mtp_serving == REPLAY_MTP_SERVING_VERIFY;
    pending_wait_for_microbatch(state, cap, mtp_commit_enabled);
    replay_pending_batch batch;
    pending_collect_batch(state, cap, &batch);
    if (batch.count == 0) return 0;

    bool profiler_active = server_profiler_start_if_requested(state);
    const uint32_t batch_tokens = batch.items[0] ? batch.items[0]->tokens : 0;
    const int first_prompt_len = batch.items[0] ? batch.items[0]->prompt_tokens.len : 0;
    bool can_batch = !mtp_commit_enabled && batch.count > 1 && batch_tokens > 0 &&
                     batch_tokens <= DS4_V100_REPLAY_MAX_TOKENS;
    for (uint32_t i = 0; i < batch.count && can_batch; i++) {
        if (!batch.items[i] || batch.items[i]->tokens != batch_tokens) {
            can_batch = false;
        } else if (mtp_verify_enabled && batch.items[i]->prompt_tokens.len != first_prompt_len) {
            can_batch = false;
        }
    }

    if (can_batch) {
        ds4_tokens prompts[DS4_V100_SCHED_MAX_SLOTS];
        ds4_replay_output batch_outputs[DS4_V100_SCHED_MAX_SLOTS *
                                             DS4_V100_REPLAY_MAX_TOKENS];
        uint32_t batch_counts[DS4_V100_SCHED_MAX_SLOTS];
        memset(prompts, 0, sizeof(prompts));
        memset(batch_outputs, 0, sizeof(batch_outputs));
        memset(batch_counts, 0, sizeof(batch_counts));
        for (uint32_t i = 0; i < batch.count; i++) prompts[i] = batch.items[i]->prompt_tokens;

        char err[512] = {0};
        ds4_replay_counters counters;
        memset(&counters, 0, sizeof(counters));
        int rc = ds4_replay_reset(state->rt, err, sizeof(err));
        if (!rc) {
            rc = ds4_replay_generate_batch(state->rt,
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
                if (mtp_verify_enabled) {
                    if (replay_mtp_service_run_slot(state->mtp,
                                                    state->rt,
                                                    i,
                                                    req->outputs,
                                                    req->n_outputs,
                                                    &req->counters,
                                                    &req->mtp_result,
                                                    err,
                                                    sizeof(err)) != 0) {
                        pending_mark_done(req, 1, err);
                    } else {
                        req->mtp_result_ready = true;
                        pending_mark_done(req, 0, NULL);
                    }
                } else {
                    pending_mark_done(req, 0, NULL);
                }
            }
            pending_remove(state, req);
        }
        if (rc) {
            for (uint32_t i = 0; i < batch.count; i++) {
                for (uint32_t j = 0; j < batch_tokens; j++) {
                    ds4_replay_output_free(
                        &batch_outputs[(uint64_t)i * DS4_V100_REPLAY_MAX_TOKENS + j]);
                }
            }
        }
        server_profiler_stop_if_active(&profiler_active);
        return 0;
    }

    for (uint32_t i = 0; i < batch.count; i++) {
        replay_pending_generation *req = batch.items[i];
        if (!req) continue;
        char err[512] = {0};
        ds4_replay_counters counters;
        memset(&counters, 0, sizeof(counters));
        int rc = ds4_replay_reset(state->rt, err, sizeof(err));
        if (!rc) {
            if (mtp_enabled && state->opt->mtp_serving == REPLAY_MTP_SERVING_COMMIT) {
                rc = replay_generate_mtp_commit_one_slot(state->mtp,
                                                         state->rt,
                                                         &req->prompt_tokens,
                                                         req->tokens,
                                                         req->outputs,
                                                         req->tokens,
                                                         &req->n_outputs,
                                                         &counters,
                                                         &req->mtp_result,
                                                         err,
                                                         sizeof(err));
                req->mtp_result_ready = rc == 0;
            } else {
                rc = ds4_replay_generate(state->rt,
                                              &req->prompt_tokens,
                                              req->tokens,
                                              req->outputs,
                                              req->tokens,
                                              &req->n_outputs,
                                              &counters,
                                              err,
                                              sizeof(err));
            }
        }
        req->counters = counters;
        pending_mark_done(req, rc, err);
        pending_remove(state, req);
    }
    server_profiler_stop_if_active(&profiler_active);
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
            "  --tm-index FILE           TurboMind appliance pack-index.tsv\n"
            "  --shard-dir DIR           directory containing gpuN.weights shards\n"
            "  --appliance-dir DIR       shorthand for DIR/pack-index.tsv, DIR/turbomind-pack-index.tsv, and shards\n"
            "  --prompt TEXT             prompt text\n"
            "  --prompt-file FILE        prompt file\n"
            "  --synthetic-prompt-token ID\n"
            "                            build prompt from repeated token ID\n"
            "  --synthetic-prompt-len N  synthetic repeated-token prompt length\n"
            "  --prompt-token-limit N    truncate encoded prompt to N tokens for diagnostics\n"
            "  --system TEXT             system prompt, default empty\n"
            "  --tokens N                greedy tokens to generate, default 1\n"
            "  --reset-parity-smoke N    generate, reset, regenerate, and compare N tokens\n"
            "  --target-block-smoke N    verify/restore a forced one-slot target block\n"
            "  --mtp-draft-block-smoke N chain MTP drafts and verify the block\n"
            "  --mtp-block2-commit-smoke N\n"
            "                            exact block-2 MTP commit diagnostic for N tokens\n"
            "  --ctx N                   KV context tokens, default 1048576\n"
            "  --slots N                 configured admission slots, default 1\n"
            "  --active-microbatch N     active decode request slots, default 1\n"
            "  --microbatch-wait-us N    max wait to coalesce active requests, default 5000\n"
            "  --queue-policy MODE       reject-busy or sequential, default reject-busy\n"
            "  --expected-token-hex HEX  require first generated token bytes\n"
            "  --json                    emit JSON\n"
            "  --open-only               open resident stages, print timing, and exit\n"
            "  --serial-open             open resident stages serially for benchmarking\n"
            "  --profile-decode          enable synchronized per-stage decode profiling\n"
            "  --wavefront-decode        enable opt-in stage-wavefront batch decode\n"
            "  --async-pipeline-decode   enable preferred opt-in async pipeline decode\n"
            "  --async-pipeline-mode M   off, persistent, per-step, or mailbox\n"
            "  --async-pipeline-per-step enable diagnostic per-token-step async workers\n"
            "  --async-handoff          queue HC peer handoff copies on the destination stream\n"
            "  --async-event-handoff    use CUDA events for per-step stage handoff ordering\n"
            "  --startup-warmup        run one internal generation before listening\n"
            "  --cuda-profiler-window  call cudaProfilerStart/Stop around generation\n"
            "  --mtp-serving MODE        off, verify, or commit, default off\n"
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

static char *join_path_alloc(const char *dir, const char *base) {
    if (!dir || !base) return NULL;
    size_t nd = strlen(dir);
    size_t nb = strlen(base);
    char *out = (char *)malloc(nd + 1u + nb + 1u);
    if (!out) {
        fprintf(stderr, "ds4-v100-replay: out of memory while joining path\n");
        exit(2);
    }
    memcpy(out, dir, nd);
    out[nd] = '/';
    memcpy(out + nd + 1u, base, nb + 1u);
    return out;
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
    opt.microbatch_wait_us = 5000;
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
        } else if (!strcmp(arg, "--tm-index")) {
            opt.turbomind_index_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--shard-dir")) {
            opt.shard_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--appliance-dir")) {
            opt.shard_dir = need_arg(&i, argc, argv, arg);
            opt.index_path = join_path_alloc(opt.shard_dir, "pack-index.tsv");
            opt.turbomind_index_path =
                join_path_alloc(opt.shard_dir, "turbomind-pack-index.tsv");
        } else if (!strcmp(arg, "--prompt")) {
            opt.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            opt.prompt_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--synthetic-prompt-token")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v > INT32_MAX) {
                fprintf(stderr, "ds4-v100-replay: invalid --synthetic-prompt-token\n");
                exit(2);
            }
            opt.synthetic_prompt_token = (uint32_t)v;
            opt.synthetic_prompt_token_set = true;
        } else if (!strcmp(arg, "--synthetic-prompt-len")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v > INT32_MAX) {
                fprintf(stderr, "ds4-v100-replay: invalid --synthetic-prompt-len\n");
                exit(2);
            }
            opt.synthetic_prompt_len = (uint32_t)v;
        } else if (!strcmp(arg, "--prompt-token-limit")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v == 0 || v > INT32_MAX) {
                fprintf(stderr, "ds4-v100-replay: invalid --prompt-token-limit\n");
                exit(2);
            }
            opt.prompt_token_limit = (uint32_t)v;
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
        } else if (!strcmp(arg, "--reset-parity-smoke")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v == 0 || v > DS4_V100_REPLAY_MAX_TOKENS) {
                fprintf(stderr,
                        "ds4-v100-replay: --reset-parity-smoke must be in [1,%d]\n",
                        DS4_V100_REPLAY_MAX_TOKENS);
                exit(2);
            }
            opt.reset_parity_smoke = (uint32_t)v;
        } else if (!strcmp(arg, "--target-block-smoke")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v >= DS4_V100_REPLAY_MAX_TOKENS) {
                fprintf(stderr,
                        "ds4-v100-replay: --target-block-smoke must be in [1,%d]\n",
                        DS4_V100_REPLAY_MAX_TOKENS - 1);
                exit(2);
            }
            opt.target_block_smoke = (uint32_t)v;
        } else if (!strcmp(arg, "--mtp-draft-block-smoke")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v >= DS4_V100_REPLAY_MAX_TOKENS) {
                fprintf(stderr,
                        "ds4-v100-replay: --mtp-draft-block-smoke must be in [1,%d]\n",
                        DS4_V100_REPLAY_MAX_TOKENS - 1);
                exit(2);
            }
            opt.mtp_draft_block_smoke = (uint32_t)v;
        } else if (!strcmp(arg, "--mtp-block2-commit-smoke")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v > DS4_V100_REPLAY_MAX_TOKENS) {
                fprintf(stderr,
                        "ds4-v100-replay: --mtp-block2-commit-smoke must be in [1,%d]\n",
                        DS4_V100_REPLAY_MAX_TOKENS);
                exit(2);
            }
            opt.mtp_block2_commit_smoke = (uint32_t)v;
        } else if (!strcmp(arg, "--ctx")) {
            opt.ctx = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--slots")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v > DS4_V100_SCHED_MAX_SLOTS) {
                fprintf(stderr, "ds4-v100-replay: --slots must be in [1,%u]\n",
                        DS4_V100_SCHED_MAX_SLOTS);
                exit(2);
            }
            opt.slots = (uint32_t)v;
        } else if (!strcmp(arg, "--active-microbatch")) {
            uint64_t v = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
            if (v > DS4_V100_SCHED_MAX_SLOTS) {
                fprintf(stderr, "ds4-v100-replay: --active-microbatch must be in [1,%u]\n",
                        DS4_V100_SCHED_MAX_SLOTS);
                exit(2);
            }
            opt.active_microbatch = (uint32_t)v;
        } else if (!strcmp(arg, "--microbatch-wait-us")) {
            uint64_t v = parse_u64_arg_allow_zero(need_arg(&i, argc, argv, arg), arg);
            if (v > 1000000ull) {
                fprintf(stderr, "ds4-v100-replay: --microbatch-wait-us must be <= 1000000\n");
                exit(2);
            }
            opt.microbatch_wait_us = (uint32_t)v;
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
        } else if (!strcmp(arg, "--profile-decode")) {
            opt.profile_decode = true;
        } else if (!strcmp(arg, "--wavefront-decode")) {
            opt.wavefront_decode = true;
        } else if (!strcmp(arg, "--async-pipeline-decode")) {
            opt.async_pipeline_decode = true;
            opt.async_pipeline_mode = DS4_V100_REPLAY_ASYNC_PIPELINE_PER_STEP;
        } else if (!strcmp(arg, "--async-pipeline-per-step")) {
            opt.async_pipeline_decode = true;
            opt.async_pipeline_mode = DS4_V100_REPLAY_ASYNC_PIPELINE_PER_STEP;
        } else if (!strcmp(arg, "--async-handoff")) {
            opt.async_handoff = true;
        } else if (!strcmp(arg, "--async-event-handoff")) {
            opt.async_event_handoff = true;
            opt.async_pipeline_decode = true;
            opt.async_pipeline_mode = DS4_V100_REPLAY_ASYNC_PIPELINE_PER_STEP;
        } else if (!strcmp(arg, "--startup-warmup")) {
            opt.startup_warmup = true;
        } else if (!strcmp(arg, "--cuda-profiler-window")) {
            opt.cuda_profiler_window = true;
        } else if (!strcmp(arg, "--async-pipeline-mode")) {
            const char *v = need_arg(&i, argc, argv, arg);
            if (!strcmp(v, "off") || !strcmp(v, "false") || !strcmp(v, "0")) {
                opt.async_pipeline_decode = false;
                opt.async_pipeline_mode = DS4_V100_REPLAY_ASYNC_PIPELINE_OFF;
            } else if (!strcmp(v, "persistent") || !strcmp(v, "on") ||
                       !strcmp(v, "true") || !strcmp(v, "1")) {
                opt.async_pipeline_decode = true;
                opt.async_pipeline_mode = DS4_V100_REPLAY_ASYNC_PIPELINE_PERSISTENT;
            } else if (!strcmp(v, "per-step") || !strcmp(v, "per_step") ||
                       !strcmp(v, "step")) {
                opt.async_pipeline_decode = true;
                opt.async_pipeline_mode = DS4_V100_REPLAY_ASYNC_PIPELINE_PER_STEP;
            } else if (!strcmp(v, "mailbox") || !strcmp(v, "mbox")) {
                opt.async_pipeline_decode = true;
                opt.async_pipeline_mode = DS4_V100_REPLAY_ASYNC_PIPELINE_MAILBOX;
            } else {
                fprintf(stderr,
                        "ds4-v100-replay: --async-pipeline-mode must be off, persistent, per-step, or mailbox\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--mtp-serving")) {
            const char *v = need_arg(&i, argc, argv, arg);
            if (!strcmp(v, "off") || !strcmp(v, "false") || !strcmp(v, "0")) {
                opt.mtp_serving = REPLAY_MTP_SERVING_OFF;
            } else if (!strcmp(v, "verify")) {
                opt.mtp_serving = REPLAY_MTP_SERVING_VERIFY;
            } else if (!strcmp(v, "commit")) {
                opt.mtp_serving = REPLAY_MTP_SERVING_COMMIT;
            } else {
                fprintf(stderr, "ds4-v100-replay: --mtp-serving must be off, verify, or commit\n");
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
    bool synthetic_prompt = opt.synthetic_prompt_len != 0;
    if (opt.reset_parity_smoke && opt.tokens < opt.reset_parity_smoke) {
        opt.tokens = opt.reset_parity_smoke;
    }
    if (opt.mtp_block2_commit_smoke && opt.tokens < opt.mtp_block2_commit_smoke) {
        opt.tokens = opt.mtp_block2_commit_smoke;
    }
    if (!opt.model_path || !opt.index_path ||
        (!opt.serve && !opt.open_only && !opt.prompt && !opt.prompt_file &&
         !synthetic_prompt)) {
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
    if (opt.mtp_serving == REPLAY_MTP_SERVING_COMMIT && opt.active_microbatch != 1) {
        fprintf(stderr, "ds4-v100-replay: --mtp-serving commit currently requires --active-microbatch 1\n");
        exit(2);
    }
    if (opt.reset_parity_smoke) {
        if (opt.slots != 1 || opt.active_microbatch != 1) {
            fprintf(stderr, "ds4-v100-replay: --reset-parity-smoke currently requires --slots 1 --active-microbatch 1\n");
            exit(2);
        }
        if (opt.serve || opt.open_only) {
            fprintf(stderr, "ds4-v100-replay: use --reset-parity-smoke without --serve or --open-only\n");
            exit(2);
        }
        if (opt.mtp_serving != REPLAY_MTP_SERVING_OFF) {
            fprintf(stderr, "ds4-v100-replay: --reset-parity-smoke validates the target path; do not combine it with --mtp-serving\n");
            exit(2);
        }
        if (opt.target_block_smoke || opt.mtp_draft_block_smoke || opt.mtp_block2_commit_smoke) {
            fprintf(stderr, "ds4-v100-replay: --reset-parity-smoke cannot be combined with other smoke modes\n");
            exit(2);
        }
    }
    if (opt.target_block_smoke) {
        if (opt.slots != 1 || opt.active_microbatch != 1) {
            fprintf(stderr, "ds4-v100-replay: --target-block-smoke currently requires --slots 1 --active-microbatch 1\n");
            exit(2);
        }
        if (opt.serve || opt.open_only) {
            fprintf(stderr, "ds4-v100-replay: use --target-block-smoke without --serve or --open-only\n");
            exit(2);
        }
        if (opt.mtp_serving != REPLAY_MTP_SERVING_OFF) {
            fprintf(stderr, "ds4-v100-replay: --target-block-smoke validates the target path; do not combine it with --mtp-serving\n");
            exit(2);
        }
        if (opt.tokens <= opt.target_block_smoke) {
            fprintf(stderr, "ds4-v100-replay: --tokens must be greater than --target-block-smoke\n");
            exit(2);
        }
    }
    if (opt.mtp_draft_block_smoke) {
        if (!opt.mtp_model_path || !opt.mtp_model_path[0]) {
            fprintf(stderr, "ds4-v100-replay: --mtp-draft-block-smoke requires --mtp-model\n");
            exit(2);
        }
        if (opt.slots != 1 || opt.active_microbatch != 1) {
            fprintf(stderr, "ds4-v100-replay: --mtp-draft-block-smoke currently requires --slots 1 --active-microbatch 1\n");
            exit(2);
        }
        if (opt.serve || opt.open_only) {
            fprintf(stderr, "ds4-v100-replay: use --mtp-draft-block-smoke without --serve or --open-only\n");
            exit(2);
        }
        if (opt.mtp_serving != REPLAY_MTP_SERVING_OFF) {
            fprintf(stderr, "ds4-v100-replay: --mtp-draft-block-smoke is a diagnostic; do not combine it with --mtp-serving\n");
            exit(2);
        }
        if (opt.tokens <= opt.mtp_draft_block_smoke) {
            fprintf(stderr, "ds4-v100-replay: --tokens must be greater than --mtp-draft-block-smoke\n");
            exit(2);
        }
    }
    if (opt.mtp_block2_commit_smoke) {
        if (!opt.mtp_model_path || !opt.mtp_model_path[0]) {
            fprintf(stderr, "ds4-v100-replay: --mtp-block2-commit-smoke requires --mtp-model\n");
            exit(2);
        }
        if (opt.slots != 1 || opt.active_microbatch != 1) {
            fprintf(stderr, "ds4-v100-replay: --mtp-block2-commit-smoke currently requires --slots 1 --active-microbatch 1\n");
            exit(2);
        }
        if (opt.serve || opt.open_only) {
            fprintf(stderr, "ds4-v100-replay: use --mtp-block2-commit-smoke without --serve or --open-only\n");
            exit(2);
        }
        if (opt.mtp_serving != REPLAY_MTP_SERVING_OFF) {
            fprintf(stderr, "ds4-v100-replay: --mtp-block2-commit-smoke is a diagnostic; do not combine it with --mtp-serving\n");
            exit(2);
        }
        if (opt.mtp_block2_commit_smoke < 2) {
            fprintf(stderr, "ds4-v100-replay: --mtp-block2-commit-smoke must be >= 2\n");
            exit(2);
        }
    }
    if (opt.async_event_handoff &&
        opt.async_pipeline_mode != DS4_V100_REPLAY_ASYNC_PIPELINE_PER_STEP) {
        fprintf(stderr, "ds4-v100-replay: --async-event-handoff requires --async-pipeline-mode per-step\n");
        exit(2);
    }
    if (opt.prompt && opt.prompt_file) {
        fprintf(stderr, "ds4-v100-replay: use --prompt or --prompt-file, not both\n");
        exit(2);
    }
    if (synthetic_prompt && (opt.prompt || opt.prompt_file)) {
        fprintf(stderr, "ds4-v100-replay: synthetic prompt mode cannot be combined with --prompt or --prompt-file\n");
        exit(2);
    }
    if (!synthetic_prompt && opt.synthetic_prompt_token_set) {
        fprintf(stderr, "ds4-v100-replay: --synthetic-prompt-token requires --synthetic-prompt-len\n");
        exit(2);
    }
    if (synthetic_prompt && !opt.synthetic_prompt_token_set) {
        fprintf(stderr, "ds4-v100-replay: --synthetic-prompt-len requires --synthetic-prompt-token\n");
        exit(2);
    }
    if (synthetic_prompt && opt.system && opt.system[0]) {
        fprintf(stderr, "ds4-v100-replay: synthetic prompt mode cannot be combined with --system\n");
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

static void mtp_result_set_service(replay_mtp_service *svc, replay_mtp_result *result) {
    if (!svc || !result) return;
    result->enabled = true;
    result->top_k = svc->top_k;
    result->sidecar_uploaded_bytes = svc->sidecar_uploaded_bytes;
    result->sidecar_arena_bytes = svc->sidecar_arena_bytes;
    result->output_weight_bytes = svc->output_weight_bytes;
}

static int replay_output_copy(ds4_replay_output *dst,
                              const ds4_replay_output *src,
                              char *err,
                              size_t errlen) {
    if (!dst || !src) {
        snprintf(err, errlen, "missing replay output copy input");
        return 1;
    }
    memset(dst, 0, sizeof(*dst));
    dst->token = src->token;
    dst->logit = src->logit;
    dst->text_len = src->text_len;
    if (src->text && src->text_len) {
        dst->text = (char *)malloc(src->text_len + 1u);
        if (!dst->text) {
            snprintf(err, errlen, "failed to copy replay output text");
            return 1;
        }
        memcpy(dst->text, src->text, src->text_len);
        dst->text[src->text_len] = '\0';
    }
    return 0;
}

static void replay_mtp_service_close(replay_mtp_service *svc) {
    if (!svc) return;
    ds4_mtp_forward_close(svc->forward);
    ds4_mtp_sidecar_close(svc->sidecar);
    if (svc->mu_ready) pthread_mutex_destroy(&svc->mu);
    free(svc);
}

static int replay_mtp_service_open(replay_mtp_service **out,
                                   const replay_cli_options *opt,
                                   ds4_replay *rt,
                                   char *err,
                                   size_t errlen) {
    if (out) *out = NULL;
    if (!out || !opt || !rt) return 1;
    if (opt->mtp_serving == REPLAY_MTP_SERVING_OFF &&
        opt->mtp_draft_block_smoke == 0 &&
        opt->mtp_block2_commit_smoke == 0) {
        return 0;
    }

    replay_mtp_service *svc = (replay_mtp_service *)calloc(1, sizeof(*svc));
    if (!svc) {
        snprintf(err, errlen, "failed to allocate MTP service");
        return 1;
    }
    svc->enabled = true;
    svc->top_k = opt->mtp_top_k;
    svc->gpu = opt->mtp_gpu;
    svc->reserve_mib = opt->mtp_reserve_mib;
    if (pthread_mutex_init(&svc->mu, NULL) != 0) {
        snprintf(err, errlen, "failed to initialize MTP service mutex");
        replay_mtp_service_close(svc);
        return 1;
    }
    svc->mu_ready = true;

    ds4_mtp_sidecar_options mtp_opts;
    ds4_mtp_sidecar_options_init(&mtp_opts);
    mtp_opts.mtp_path = opt->mtp_model_path;
    mtp_opts.gpu = opt->mtp_gpu;
    mtp_opts.require_device_arena = true;
    if (ds4_mtp_sidecar_open(&svc->sidecar, &mtp_opts, NULL, err, errlen) != 0) {
        replay_mtp_service_close(svc);
        return 1;
    }
    ds4_gpu_arena *arena = ds4_mtp_sidecar_arena(svc->sidecar);
    svc->sidecar_uploaded_bytes = ds4_mtp_sidecar_uploaded_bytes(svc->sidecar);
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

    ds4_context *ctx = NULL;
    ds4_context_options ctx_opts;
    ds4_context_options_init(&ctx_opts);
    ctx_opts.pack_index_path = opt->index_path;
    if (ds4_context_open(&ctx, &ctx_opts, err, errlen) != 0) {
        replay_mtp_service_close(svc);
        return 1;
    }
    ds4_tensor_binding output_weight;
    int rc = ds4_context_output_head_binding(ctx, &output_weight, err, errlen);
    if (rc != 0) {
        ds4_context_close(ctx);
        replay_mtp_service_close(svc);
        return 1;
    }
    svc->output_weight_bytes = output_weight.byte_length;
    if (ds4_mtp_forward_open(&svc->forward,
                                  svc->sidecar,
                                  ds4_replay_model_map(rt),
                                  ds4_replay_model_size(rt),
                                  &output_weight,
                                  opt->mtp_gpu,
                                  err,
                                  errlen) != 0) {
        ds4_context_close(ctx);
        replay_mtp_service_close(svc);
        return 1;
    }
    ds4_context_close(ctx);
    *out = svc;
    return 0;
}

static int replay_mtp_service_draft(replay_mtp_service *svc,
                                    ds4_replay *rt,
                                    uint32_t hc_slot,
                                    uint32_t committed_token,
                                    uint32_t committed_pos,
                                    uint32_t target_token,
                                    float target_logit,
                                    replay_mtp_result *result,
                                    char *err,
                                    size_t errlen) {
    if (!svc || !svc->enabled || !rt || !result) return 0;
    mtp_result_set_service(svc, result);
    result->committed_token = committed_token;
    result->committed_pos = committed_pos;
    result->target_token = target_token;
    result->target_logit = target_logit;
    float embed[DS4_V100_MTP_FORWARD_N_EMBD];
    float hc[DS4_V100_MTP_FORWARD_HC_VALUES];
    if (ds4_replay_read_token_embedding_f32(rt,
                                                 committed_token,
                                                 embed,
                                                 DS4_V100_MTP_FORWARD_N_EMBD,
                                                 err,
                                                 errlen) != 0 ||
        ds4_replay_read_output_hc_slot(rt,
                                            hc_slot,
                                            hc,
                                            sizeof(hc),
                                            err,
                                            errlen) != 0) {
        return 1;
    }
    ds4_mtp_forward_report report;
    memset(&report, 0, sizeof(report));
    result->attempted = true;
    result->attempts++;
    result->draft_tokens_proposed++;
    result->target_tokens_verified++;
    const double t0 = now_ms();
    pthread_mutex_lock(&svc->mu);
    if (ds4_mtp_forward_run_host(svc->forward,
                                      embed,
                                      hc,
                                      committed_pos,
                                      svc->top_k,
                                      result->draft_tokens,
                                      result->draft_logits,
                                      &report,
                                      err,
                                      errlen) != 0) {
        pthread_mutex_unlock(&svc->mu);
        return 1;
    }
    pthread_mutex_unlock(&svc->mu);
    result->draft_ms = now_ms() - t0;
    result->draft_total_ms += result->draft_ms;
    result->draft_token = result->draft_tokens[0];
    result->draft_logit = result->draft_logits[0];
    result->raw_row = report.raw_row;
    result->n_raw = report.n_raw;
    result->output_vocab = report.output_vocab;
    result->output_weight_bytes = report.output_weight_bytes;
    result->free_after_output_upload_bytes = report.free_after_output_upload_bytes;
    result->scratch_device_bytes = report.scratch_device_bytes;
    result->scratch_host_bytes = report.scratch_host_bytes;
    result->forward_run_count = report.run_count;
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
    if (result->accepted) {
        result->accepted_count++;
        result->draft_tokens_accepted++;
        if (result->accepted_prefix_len < 1u) result->accepted_prefix_len = 1u;
    } else {
        result->rejected_count++;
    }
    svc->drafts++;
    if (result->accepted) svc->accepted++;
    else svc->rejected++;
    return 0;
}

static int replay_mtp_service_run(replay_mtp_service *svc,
                                  ds4_replay *rt,
                                  const ds4_replay_output *outputs,
                                  uint32_t n_outputs,
                                  const ds4_replay_counters *counters,
                                  replay_mtp_result *result,
                                  char *err,
                                  size_t errlen) {
    return replay_mtp_service_run_slot(svc,
                                       rt,
                                       0,
                                       outputs,
                                       n_outputs,
                                       counters,
                                       result,
                                       err,
                                       errlen);
}

static int replay_mtp_service_run_slot(replay_mtp_service *svc,
                                       ds4_replay *rt,
                                       uint32_t hc_slot,
                                       const ds4_replay_output *outputs,
                                       uint32_t n_outputs,
                                       const ds4_replay_counters *counters,
                                       replay_mtp_result *result,
                                       char *err,
                                       size_t errlen) {
    mtp_result_init(result);
    if (!svc || !svc->enabled || !result) return 0;
    mtp_result_set_service(svc, result);
    svc->requests++;
    if (!rt || !outputs || !counters || n_outputs < 2) {
        result->skipped = true;
        snprintf(result->reason, sizeof(result->reason), "need_at_least_two_tokens");
        svc->skipped++;
        return 0;
    }
    const uint32_t committed_idx = n_outputs - 2u;
    const uint32_t target_idx = n_outputs - 1u;
    int rc = replay_mtp_service_draft(svc,
                                      rt,
                                      hc_slot,
                                      outputs[committed_idx].token,
                                      counters->prompt_tokens + committed_idx,
                                      outputs[target_idx].token,
                                      outputs[target_idx].logit,
                                      result,
                                      err,
                                      errlen);
    if (rc == 0) {
        result->target_forwards = n_outputs;
        result->effective_output_tokens = n_outputs;
        result->speculative_saves = 0;
        if (result->reason[0] == '\0') {
            snprintf(result->reason,
                     sizeof(result->reason),
                     "posthoc_verify_no_target_forward_savings");
        }
    }
    return rc;
}

static int replay_generate_mtp_commit_one_slot(replay_mtp_service *svc,
                                               ds4_replay *rt,
                                               const ds4_tokens *prompt,
                                               uint32_t max_tokens,
                                               ds4_replay_output *outputs,
                                               uint32_t output_cap,
                                               uint32_t *out_count,
                                               ds4_replay_counters *counters,
                                               replay_mtp_result *mtp_result,
                                               char *err,
                                               size_t errlen) {
    if (out_count) *out_count = 0;
    mtp_result_init(mtp_result);
    if (!svc || !svc->enabled || !rt || !prompt || prompt->len <= 0 ||
        !outputs || !mtp_result || max_tokens == 0 || output_cap < max_tokens) {
        snprintf(err, errlen, "missing MTP commit generation input");
        return 1;
    }
    mtp_result->commit_mode = true;
    mtp_result_set_service(svc, mtp_result);
    svc->requests++;

    ds4_replay_counters local;
    ds4_replay_counters *c = counters ? counters : &local;
    if (ds4_replay_begin_generation(rt,
                                         (uint32_t)prompt->len,
                                         c,
                                         err,
                                         errlen) != 0) {
        return 1;
    }

    const double total0 = now_ms();
    for (int pos = 0; pos < prompt->len; pos++) {
        if (prompt->v[pos] < 0) {
            snprintf(err, errlen, "negative prompt token");
            return 1;
        }
        if (ds4_replay_feed_token_at_position(rt,
                                                   (uint32_t)prompt->v[pos],
                                                   (uint32_t)pos,
                                                   c,
                                                   &c->prompt_replay_ms,
                                                   err,
                                                   errlen) != 0) {
            return 1;
        }
        c->total_input_tokens++;
    }

    uint32_t n_out = 0;
    if (ds4_replay_select_current_token(rt,
                                             &outputs[n_out],
                                             c,
                                             err,
                                             errlen) != 0) {
        return 1;
    }
    n_out++;
    mtp_result->target_forwards = n_out;
    mtp_result->effective_output_tokens = n_out;

    if (max_tokens < 2) {
        mtp_result->skipped = true;
        snprintf(mtp_result->reason, sizeof(mtp_result->reason), "need_at_least_two_tokens");
        svc->skipped++;
        ds4_replay_finish_generation(rt, n_out, now_ms() - total0, c);
        if (out_count) *out_count = n_out;
        return 0;
    }

    for (uint32_t step = 1; step < max_tokens; step++) {
        const uint32_t committed_token = outputs[n_out - 1].token;
        const uint32_t committed_pos = (uint32_t)prompt->len + step - 1u;
        if (ds4_replay_feed_token_at_position(rt,
                                                   committed_token,
                                                   committed_pos,
                                                   c,
                                                   &c->continuation_decode_ms,
                                                   err,
                                                   errlen) != 0) {
            return 1;
        }
        c->total_input_tokens++;

        ds4_replay_output target;
        memset(&target, 0, sizeof(target));
        if (ds4_replay_select_current_token(rt, &target, c, err, errlen) != 0) {
            return 1;
        }
        if (replay_mtp_service_draft(svc,
                                     rt,
                                     0,
                                     committed_token,
                                     committed_pos,
                                     target.token,
                                     target.logit,
                                     mtp_result,
                                     err,
                                     errlen) != 0) {
            ds4_replay_output_free(&target);
            return 1;
        }
        if (mtp_result->accepted) {
            mtp_result->commit_applied = true;
            mtp_result->commit_count++;
            svc->committed++;
        }
        outputs[n_out++] = target;
        mtp_result->target_forwards = n_out;
        mtp_result->effective_output_tokens = n_out;
        mtp_result->speculative_saves = 0;
    }

    if (mtp_result->reason[0] == '\0') {
        snprintf(mtp_result->reason,
                 sizeof(mtp_result->reason),
                 "serial_commit_no_target_forward_savings");
    }
    ds4_replay_finish_generation(rt, n_out, now_ms() - total0, c);
    if (out_count) *out_count = n_out;
    return 0;
}

static void print_mtp_json(FILE *fp, const replay_mtp_result *mtp) {
    if (!mtp) return;
    fprintf(fp, ",\"mtp\":{");
    fprintf(fp, "\"enabled\":%s,", mtp->enabled ? "true" : "false");
    fprintf(fp, "\"commit_mode\":%s,", mtp->commit_mode ? "true" : "false");
    fprintf(fp, "\"attempted\":%s,", mtp->attempted ? "true" : "false");
    fprintf(fp, "\"skipped\":%s,", mtp->skipped ? "true" : "false");
    fprintf(fp, "\"accepted\":%s,", mtp->accepted ? "true" : "false");
    fprintf(fp, "\"commit_applied\":%s,", mtp->commit_applied ? "true" : "false");
    fprintf(fp, "\"committed_token\":%" PRIu32 ",", mtp->committed_token);
    fprintf(fp, "\"committed_pos\":%" PRIu32 ",", mtp->committed_pos);
    fprintf(fp, "\"target_token\":%" PRIu32 ",", mtp->target_token);
    fprintf(fp, "\"draft_token\":%" PRIu32 ",", mtp->draft_token);
    fprintf(fp, "\"attempts\":%" PRIu32 ",", mtp->attempts);
    fprintf(fp, "\"accepted_count\":%" PRIu32 ",", mtp->accepted_count);
    fprintf(fp, "\"rejected_count\":%" PRIu32 ",", mtp->rejected_count);
    fprintf(fp, "\"commit_count\":%" PRIu32 ",", mtp->commit_count);
    fprintf(fp, "\"draft_tokens_proposed\":%" PRIu32 ",", mtp->draft_tokens_proposed);
    fprintf(fp, "\"draft_tokens_accepted\":%" PRIu32 ",", mtp->draft_tokens_accepted);
    fprintf(fp, "\"accepted_prefix_len\":%" PRIu32 ",", mtp->accepted_prefix_len);
    fprintf(fp, "\"target_tokens_verified\":%" PRIu32 ",", mtp->target_tokens_verified);
    fprintf(fp, "\"target_forwards\":%" PRIu32 ",", mtp->target_forwards);
    fprintf(fp, "\"effective_output_tokens\":%" PRIu32 ",", mtp->effective_output_tokens);
    fprintf(fp, "\"speculative_saves\":%" PRIu32 ",", mtp->speculative_saves);
    fprintf(fp, "\"target_logit\":%.9g,", mtp->target_logit);
    fprintf(fp, "\"draft_logit\":%.9g,", mtp->draft_logit);
    fprintf(fp, "\"top_k\":%" PRIu32 ",", mtp->top_k);
    fprintf(fp, "\"draft_ms\":%.3f,", mtp->draft_ms);
    fprintf(fp, "\"draft_total_ms\":%.3f,", mtp->draft_total_ms);
    fprintf(fp, "\"raw_row\":%" PRIu32 ",", mtp->raw_row);
    fprintf(fp, "\"n_raw\":%" PRIu32 ",", mtp->n_raw);
    fprintf(fp, "\"output_vocab\":%" PRIu32 ",", mtp->output_vocab);
    fprintf(fp, "\"sidecar_uploaded_bytes\":%" PRIu64 ",", mtp->sidecar_uploaded_bytes);
    fprintf(fp, "\"sidecar_arena_bytes\":%" PRIu64 ",", mtp->sidecar_arena_bytes);
    fprintf(fp, "\"output_weight_bytes\":%" PRIu64 ",", mtp->output_weight_bytes);
    fprintf(fp, "\"free_after_output_upload_bytes\":%" PRIu64 ",", mtp->free_after_output_upload_bytes);
    fprintf(fp, "\"scratch_device_bytes\":%" PRIu64 ",", mtp->scratch_device_bytes);
    fprintf(fp, "\"scratch_host_bytes\":%" PRIu64 ",", mtp->scratch_host_bytes);
    fprintf(fp, "\"forward_run_count\":%" PRIu64 ",", mtp->forward_run_count);
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
                          const ds4_replay_output *outputs,
                          uint32_t n_outputs,
                          const ds4_replay_counters *c,
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
    fprintf(fp, "],\"stage_profile\":{\"hc_attn\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->stage_hc_attn_ms[i]);
    }
    fprintf(fp, "],\"attention\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->stage_attention_ms[i]);
    }
    fprintf(fp, "],\"attn_proj\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->stage_attn_proj_ms[i]);
    }
    fprintf(fp, "],\"attn_cache\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->stage_attn_cache_ms[i]);
    }
    fprintf(fp, "],\"attn_softmax\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->stage_attn_softmax_ms[i]);
    }
    fprintf(fp, "],\"attn_inverse_rope\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->stage_attn_inverse_rope_ms[i]);
    }
    fprintf(fp, "],\"attn_output\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->stage_attn_output_ms[i]);
    }
    fprintf(fp, "],\"hc_ffn\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->stage_hc_ffn_ms[i]);
    }
    fprintf(fp, "],\"ffn\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->stage_ffn_ms[i]);
    }
    fprintf(fp, "],\"hc_final\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->stage_hc_final_ms[i]);
    }
    fprintf(fp, "],\"total\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->stage_profile_total_ms[i]);
    }
    fprintf(fp, "]},\"handoff\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS - 1; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->handoff_ms[i]);
    }
    fprintf(fp, "],\"open_stage\":[");
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) fprintf(fp, ",");
        fprintf(fp, "%.3f", c->open_ms[i]);
    }
    if (c->async_pipeline_dispatches > 0) {
        fprintf(fp, "],\"async_pipeline\":{");
        fprintf(fp, "\"dispatches\":%" PRIu64 ",", c->async_pipeline_dispatches);
        fprintf(fp, "\"total\":%.3f,", c->async_pipeline_total_ms);
        fprintf(fp, "\"setup\":%.3f,", c->async_pipeline_setup_ms);
        fprintf(fp, "\"host_wait\":%.3f,", c->async_pipeline_host_wait_ms);
        fprintf(fp, "\"complete\":%.3f,", c->async_pipeline_complete_ms);
        fprintf(fp, "\"wait_prev\":[");
        for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
            if (i) fprintf(fp, ",");
            fprintf(fp, "%.3f", c->async_pipeline_worker_wait_ms[i]);
        }
        fprintf(fp, "],\"handoff\":[");
        for (int i = 0; i < DS4_V100_EXPECTED_GPUS - 1; i++) {
            if (i) fprintf(fp, ",");
            fprintf(fp, "%.3f", c->handoff_ms[i]);
        }
        fprintf(fp, "],\"device_sync\":[");
        for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
            if (i) fprintf(fp, ",");
            fprintf(fp, "%.3f", c->async_pipeline_sync_ms[i]);
        }
        fprintf(fp, "]}");
    } else {
        fprintf(fp, "]");
    }
    fprintf(fp, "},\"memory\":{");
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

static void print_open_json(const ds4_replay_counters *c, bool serial_open) {
    printf("{\"open_only\":true,\"open_mode\":\"%s\",\"timing_ms\":{",
           serial_open ? "serial" : "parallel");
    printf("\"open_total\":%.3f,\"open_stage\":[", c->open_total_ms);
    for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
        if (i) printf(",");
        printf("%.3f", c->open_ms[i]);
    }
    printf("]}}\n");
}

static int replay_output_token_matches(const char *label,
                                       uint32_t i,
                                       const ds4_replay_output *got,
                                       const ds4_replay_output *want) {
    if (!got || !want || got->token != want->token) {
        fprintf(stderr,
                "ds4-v100-replay: target-block %s[%u] token mismatch got=%" PRIu32
                " want=%" PRIu32 "\n",
                label,
                i,
                got ? got->token : UINT32_MAX,
                want ? want->token : UINT32_MAX);
        return 1;
    }
    return 0;
}

static int run_reset_parity_smoke(ds4_replay *rt,
                                  const ds4_tokens *prompt,
                                  uint32_t max_tokens,
                                  const unsigned char *expected,
                                  size_t expected_len,
                                  bool json,
                                  char *err,
                                  size_t errlen) {
    if (!rt || !prompt || prompt->len <= 0 || max_tokens == 0 ||
        max_tokens > DS4_V100_REPLAY_MAX_TOKENS) {
        snprintf(err, errlen, "invalid reset parity smoke input");
        return 1;
    }

    ds4_replay_output first[DS4_V100_REPLAY_MAX_TOKENS];
    ds4_replay_output second[DS4_V100_REPLAY_MAX_TOKENS];
    memset(first, 0, sizeof(first));
    memset(second, 0, sizeof(second));

    ds4_replay_counters first_counters;
    ds4_replay_counters second_counters;
    memset(&first_counters, 0, sizeof(first_counters));
    memset(&second_counters, 0, sizeof(second_counters));

    uint32_t first_count = 0;
    uint32_t second_count = 0;
    uint32_t mismatch_index = UINT32_MAX;
    int rc = 1;

    if (ds4_replay_generate(rt,
                                 prompt,
                                 max_tokens,
                                 first,
                                 DS4_V100_REPLAY_MAX_TOKENS,
                                 &first_count,
                                 &first_counters,
                                 err,
                                 errlen) ||
        first_count != max_tokens) {
        if (err[0] == '\0') snprintf(err, errlen, "reset parity first generation failed");
        goto done;
    }
    if (expected) {
        if (!first[0].text ||
            first[0].text_len != expected_len ||
            memcmp(first[0].text, expected, expected_len) != 0) {
            fprintf(stderr, "ds4-v100-replay: reset-parity selected token mismatch expected=");
            print_hex(stderr, expected, expected_len);
            fprintf(stderr, " got=");
            if (first[0].text) {
                print_hex(stderr, (const unsigned char *)first[0].text, first[0].text_len);
            } else {
                fprintf(stderr, "none");
            }
            fprintf(stderr, " token=%" PRIu32 " logit=%.8g\n",
                    first[0].token,
                    first[0].logit);
            snprintf(err, errlen, "reset parity expected first token mismatch");
            goto done;
        }
    }

    if (ds4_replay_reset(rt, err, errlen)) goto done;

    if (ds4_replay_generate(rt,
                                 prompt,
                                 max_tokens,
                                 second,
                                 DS4_V100_REPLAY_MAX_TOKENS,
                                 &second_count,
                                 &second_counters,
                                 err,
                                 errlen) ||
        second_count != max_tokens) {
        if (err[0] == '\0') snprintf(err, errlen, "reset parity second generation failed");
        goto done;
    }

    for (uint32_t i = 0; i < max_tokens; i++) {
        if (first[i].token != second[i].token) {
            mismatch_index = i;
            break;
        }
    }
    const bool match = mismatch_index == UINT32_MAX;

    if (json) {
        printf("{\"reset_parity_smoke\":true,"
               "\"prompt_tokens\":%d,"
               "\"generated_tokens\":%" PRIu32 ","
               "\"match\":%s,"
               "\"mismatch_index\":",
               prompt->len,
               max_tokens,
               match ? "true" : "false");
        if (match) printf("null");
        else printf("%" PRIu32, mismatch_index);
        printf(",\"first_token\":%" PRIu32
               ",\"second_token\":%" PRIu32
               ",\"first_hex\":\"",
               first_count ? first[0].token : UINT32_MAX,
               second_count ? second[0].token : UINT32_MAX);
        if (first_count && first[0].text) {
            print_hex(stdout, (const unsigned char *)first[0].text, first[0].text_len);
        }
        printf("\",\"second_hex\":\"");
        if (second_count && second[0].text) {
            print_hex(stdout, (const unsigned char *)second[0].text, second[0].text_len);
        }
        printf("\",\"first_total_ms\":%.3f,"
               "\"second_total_ms\":%.3f,"
               "\"first_prompt_ms\":%.3f,"
               "\"second_prompt_ms\":%.3f}\n",
               first_counters.total_ms,
               second_counters.total_ms,
               first_counters.prompt_replay_ms,
               second_counters.prompt_replay_ms);
    } else {
        printf("ds4-v100-replay: reset_parity_smoke prompt_tokens=%d"
               " generated_tokens=%" PRIu32
               " match=%s mismatch_index=",
               prompt->len,
               max_tokens,
               match ? "true" : "false");
        if (match) printf("none");
        else printf("%" PRIu32, mismatch_index);
        printf(" first_token=%" PRIu32
               " second_token=%" PRIu32
               " first_total_ms=%.3f second_total_ms=%.3f"
               " first_prompt_ms=%.3f second_prompt_ms=%.3f %s\n",
               first_count ? first[0].token : UINT32_MAX,
               second_count ? second[0].token : UINT32_MAX,
               first_counters.total_ms,
               second_counters.total_ms,
               first_counters.prompt_replay_ms,
               second_counters.prompt_replay_ms,
               match ? "ok" : "mismatch");
    }

    if (!match) {
        snprintf(err,
                 errlen,
                 "reset parity mismatch at token %" PRIu32 ": first=%" PRIu32 " second=%" PRIu32,
                 mismatch_index,
                 first[mismatch_index].token,
                 second[mismatch_index].token);
        goto done;
    }
    rc = 0;

done:
    for (uint32_t i = 0; i < first_count && i < DS4_V100_REPLAY_MAX_TOKENS; i++) {
        ds4_replay_output_free(&first[i]);
    }
    for (uint32_t i = 0; i < second_count && i < DS4_V100_REPLAY_MAX_TOKENS; i++) {
        ds4_replay_output_free(&second[i]);
    }
    return rc;
}

static int run_target_block_smoke(ds4_replay *rt,
                                  const ds4_tokens *prompt,
                                  uint32_t max_tokens,
                                  uint32_t block_n,
                                  const unsigned char *expected,
                                  size_t expected_len,
                                  bool json,
                                  char *err,
                                  size_t errlen) {
    if (!rt || !prompt || prompt->len <= 0 || block_n == 0 || max_tokens <= block_n ||
        max_tokens > DS4_V100_REPLAY_MAX_TOKENS) {
        snprintf(err, errlen, "invalid target block smoke input");
        return 1;
    }

    ds4_replay_output baseline[DS4_V100_REPLAY_MAX_TOKENS];
    ds4_replay_output prompt_first;
    ds4_replay_output block_first[DS4_V100_REPLAY_MAX_TOKENS];
    ds4_replay_output block_second[DS4_V100_REPLAY_MAX_TOKENS];
    memset(baseline, 0, sizeof(baseline));
    memset(&prompt_first, 0, sizeof(prompt_first));
    memset(block_first, 0, sizeof(block_first));
    memset(block_second, 0, sizeof(block_second));

    ds4_replay_counters baseline_counters;
    ds4_replay_counters smoke_counters;
    memset(&baseline_counters, 0, sizeof(baseline_counters));
    memset(&smoke_counters, 0, sizeof(smoke_counters));

    uint32_t baseline_count = 0;
    int rc = ds4_replay_generate(rt,
                                      prompt,
                                      max_tokens,
                                      baseline,
                                      DS4_V100_REPLAY_MAX_TOKENS,
                                      &baseline_count,
                                      &baseline_counters,
                                      err,
                                      errlen);
    if (rc || baseline_count <= block_n) {
        if (!rc) snprintf(err, errlen, "target block baseline produced too few tokens");
        rc = 1;
        goto done;
    }
    if (expected) {
        if (!baseline[0].text ||
            baseline[0].text_len != expected_len ||
            memcmp(baseline[0].text, expected, expected_len) != 0) {
            fprintf(stderr, "ds4-v100-replay: target-block selected token mismatch expected=");
            print_hex(stderr, expected, expected_len);
            fprintf(stderr, " got=");
            if (baseline[0].text) {
                print_hex(stderr,
                          (const unsigned char *)baseline[0].text,
                          baseline[0].text_len);
            } else {
                fprintf(stderr, "none");
            }
            fprintf(stderr, " token=%" PRIu32 " logit=%.8g\n",
                    baseline[0].token,
                    baseline[0].logit);
            snprintf(err, errlen, "target block expected first token mismatch");
            rc = 1;
            goto done;
        }
    }

    if (ds4_replay_reset(rt, err, errlen) ||
        ds4_replay_begin_generation(rt,
                                         (uint32_t)prompt->len,
                                         &smoke_counters,
                                         err,
                                         errlen)) {
        rc = 1;
        goto done;
    }

    const double smoke0 = now_ms();
    for (int pos = 0; pos < prompt->len; pos++) {
        if (prompt->v[pos] < 0) {
            snprintf(err, errlen, "negative prompt token at position %d", pos);
            rc = 1;
            goto done;
        }
        if (ds4_replay_feed_token_at_position(rt,
                                                   (uint32_t)prompt->v[pos],
                                                   (uint32_t)pos,
                                                   &smoke_counters,
                                                   &smoke_counters.prompt_replay_ms,
                                                   err,
                                                   errlen)) {
            rc = 1;
            goto done;
        }
        smoke_counters.total_input_tokens++;
    }
    if (ds4_replay_select_current_token(rt,
                                             &prompt_first,
                                             &smoke_counters,
                                             err,
                                             errlen)) {
        rc = 1;
        goto done;
    }
    if (replay_output_token_matches("prompt", 0, &prompt_first, &baseline[0])) {
        snprintf(err, errlen, "target block prompt first token mismatch");
        rc = 1;
        goto done;
    }

    ds4_replay_snapshot *snapshot = NULL;
    if (ds4_replay_snapshot_create(rt, &snapshot, err, errlen)) {
        rc = 1;
        goto done;
    }
    const uint64_t snapshot_bytes = ds4_replay_snapshot_bytes(snapshot);

    uint32_t forced_tokens[DS4_V100_REPLAY_MAX_TOKENS];
    uint32_t positions[DS4_V100_REPLAY_MAX_TOKENS];
    for (uint32_t i = 0; i < block_n; i++) {
        forced_tokens[i] = baseline[i].token;
        positions[i] = (uint32_t)prompt->len + i;
    }

    ds4_replay_target_block_report first_report;
    ds4_replay_target_block_report second_report;
    memset(&first_report, 0, sizeof(first_report));
    memset(&second_report, 0, sizeof(second_report));
    if (ds4_replay_verify_token_block(rt,
                                           forced_tokens,
                                           positions,
                                           block_n,
                                           block_n,
                                           block_first,
                                           DS4_V100_REPLAY_MAX_TOKENS,
                                           &first_report,
                                           &smoke_counters,
                                           err,
                                           errlen)) {
        ds4_replay_snapshot_free(snapshot);
        rc = 1;
        goto done;
    }
    first_report.snapshot_bytes = snapshot_bytes;
    for (uint32_t i = 0; i < block_n; i++) {
        if (replay_output_token_matches("first", i, &block_first[i], &baseline[i + 1u])) {
            snprintf(err, errlen, "target block first pass mismatch");
            ds4_replay_snapshot_free(snapshot);
            rc = 1;
            goto done;
        }
    }

    if (ds4_replay_snapshot_restore(rt, snapshot, err, errlen)) {
        ds4_replay_snapshot_free(snapshot);
        rc = 1;
        goto done;
    }
    if (ds4_replay_verify_token_block(rt,
                                           forced_tokens,
                                           positions,
                                           block_n,
                                           block_n,
                                           block_second,
                                           DS4_V100_REPLAY_MAX_TOKENS,
                                           &second_report,
                                           &smoke_counters,
                                           err,
                                           errlen)) {
        ds4_replay_snapshot_free(snapshot);
        rc = 1;
        goto done;
    }
    second_report.snapshot_bytes = snapshot_bytes;
    for (uint32_t i = 0; i < block_n; i++) {
        if (replay_output_token_matches("second", i, &block_second[i], &baseline[i + 1u]) ||
            replay_output_token_matches("restore", i, &block_second[i], &block_first[i])) {
            snprintf(err, errlen, "target block restore pass mismatch");
            ds4_replay_snapshot_free(snapshot);
            rc = 1;
            goto done;
        }
    }

    smoke_counters.generated_tokens = block_n + 1u;
    smoke_counters.total_ms = now_ms() - smoke0;
    ds4_replay_finish_generation(rt,
                                      smoke_counters.generated_tokens,
                                      smoke_counters.total_ms,
                                      &smoke_counters);

    if (json) {
        printf("{\"target_block_smoke\":true,"
               "\"prompt_tokens\":%d,"
               "\"baseline_tokens\":%" PRIu32 ","
               "\"block_tokens\":%" PRIu32 ","
               "\"first_token\":%" PRIu32 ","
               "\"first_hex\":\"",
               prompt->len,
               baseline_count,
               block_n,
               baseline[0].token);
        if (baseline[0].text) {
            print_hex(stdout,
                      (const unsigned char *)baseline[0].text,
                      baseline[0].text_len);
        }
        printf("\","
               "\"snapshot_bytes\":%" PRIu64 ","
               "\"target_forwards\":%" PRIu32 ","
               "\"accepted_prefix_len\":%" PRIu32 ","
               "\"target_tokens_verified\":%" PRIu32 ","
               "\"effective_output_tokens\":%" PRIu32 ","
               "\"speculative_saves\":%" PRIu32 ","
               "\"first_verify_ms\":%.3f,"
               "\"second_verify_ms\":%.3f}\n",
               snapshot_bytes,
               first_report.target_forwards,
               first_report.accepted_prefix_len,
               first_report.target_tokens_verified,
               first_report.effective_output_tokens,
               first_report.speculative_saves,
               first_report.verify_ms,
               second_report.verify_ms);
    } else {
        printf("ds4-v100-replay: target_block_smoke block_tokens=%" PRIu32
               " baseline_tokens=%" PRIu32
               " first_token=%" PRIu32
               " first_hex=",
               block_n,
               baseline_count,
               baseline[0].token);
        if (baseline[0].text) {
            print_hex(stdout,
                      (const unsigned char *)baseline[0].text,
                      baseline[0].text_len);
        } else {
            printf("none");
        }
        printf(" snapshot_bytes=%" PRIu64
               " target_forwards=%" PRIu32
               " accepted_prefix_len=%" PRIu32
               " target_tokens_verified=%" PRIu32
               " effective_output_tokens=%" PRIu32
               " speculative_saves=%" PRIu32
               " first_verify_ms=%.3f second_verify_ms=%.3f ok\n",
               snapshot_bytes,
               first_report.target_forwards,
               first_report.accepted_prefix_len,
               first_report.target_tokens_verified,
               first_report.effective_output_tokens,
               first_report.speculative_saves,
               first_report.verify_ms,
               second_report.verify_ms);
    }

    ds4_replay_snapshot_free(snapshot);

done:
    for (uint32_t i = 0; i < baseline_count && i < DS4_V100_REPLAY_MAX_TOKENS; i++) {
        ds4_replay_output_free(&baseline[i]);
    }
    ds4_replay_output_free(&prompt_first);
    for (uint32_t i = 0; i < block_n && i < DS4_V100_REPLAY_MAX_TOKENS; i++) {
        ds4_replay_output_free(&block_first[i]);
        ds4_replay_output_free(&block_second[i]);
    }
    return rc;
}

static int run_mtp_draft_block_smoke(replay_mtp_service *svc,
                                     ds4_replay *rt,
                                     const ds4_tokens *prompt,
                                     uint32_t max_tokens,
                                     uint32_t block_n,
                                     const unsigned char *expected,
                                     size_t expected_len,
                                     bool json,
                                     char *err,
                                     size_t errlen) {
    if (!svc || !svc->enabled || !rt || !prompt || prompt->len <= 0 ||
        block_n == 0 || max_tokens <= block_n ||
        block_n >= DS4_V100_REPLAY_MAX_TOKENS) {
        snprintf(err, errlen, "invalid MTP draft block smoke input");
        return 1;
    }

    ds4_replay_counters counters;
    memset(&counters, 0, sizeof(counters));
    ds4_replay_output first_target;
    ds4_replay_output target_outputs[DS4_V100_REPLAY_MAX_TOKENS];
    memset(&first_target, 0, sizeof(first_target));
    memset(target_outputs, 0, sizeof(target_outputs));

    uint32_t draft_tokens[DS4_V100_REPLAY_MAX_TOKENS];
    uint32_t forced_tokens[DS4_V100_REPLAY_MAX_TOKENS];
    uint32_t positions[DS4_V100_REPLAY_MAX_TOKENS];
    float draft_logits[DS4_V100_REPLAY_MAX_TOKENS];
    uint32_t top_tokens[DS4_V100_MTP_FORWARD_MAX_TOPK];
    float top_logits[DS4_V100_MTP_FORWARD_MAX_TOPK];
    memset(draft_tokens, 0xff, sizeof(draft_tokens));
    memset(forced_tokens, 0, sizeof(forced_tokens));
    memset(positions, 0, sizeof(positions));
    memset(draft_logits, 0, sizeof(draft_logits));

    float embed[DS4_V100_MTP_FORWARD_N_EMBD];
    float hc_a[DS4_V100_MTP_FORWARD_HC_VALUES];
    float hc_b[DS4_V100_MTP_FORWARD_HC_VALUES];
    memset(embed, 0, sizeof(embed));
    memset(hc_a, 0, sizeof(hc_a));
    memset(hc_b, 0, sizeof(hc_b));

    int rc = 1;
    ds4_replay_snapshot *snapshot = NULL;
    ds4_replay_target_block_report verify_report;
    ds4_mtp_forward_report last_mtp_report;
    memset(&verify_report, 0, sizeof(verify_report));
    memset(&last_mtp_report, 0, sizeof(last_mtp_report));

    if (ds4_replay_reset(rt, err, errlen) ||
        ds4_replay_begin_generation(rt,
                                         (uint32_t)prompt->len,
                                         &counters,
                                         err,
                                         errlen)) {
        goto done;
    }

    const double total0 = now_ms();
    for (int pos = 0; pos < prompt->len; pos++) {
        if (prompt->v[pos] < 0) {
            snprintf(err, errlen, "negative prompt token at position %d", pos);
            goto done;
        }
        if (ds4_replay_feed_token_at_position(rt,
                                                   (uint32_t)prompt->v[pos],
                                                   (uint32_t)pos,
                                                   &counters,
                                                   &counters.prompt_replay_ms,
                                                   err,
                                                   errlen)) {
            goto done;
        }
        counters.total_input_tokens++;
    }
    if (ds4_replay_select_current_token(rt,
                                             &first_target,
                                             &counters,
                                             err,
                                             errlen)) {
        goto done;
    }
    if (expected) {
        if (!first_target.text ||
            first_target.text_len != expected_len ||
            memcmp(first_target.text, expected, expected_len) != 0) {
            fprintf(stderr, "ds4-v100-replay: mtp-draft-block selected token mismatch expected=");
            print_hex(stderr, expected, expected_len);
            fprintf(stderr, " got=");
            if (first_target.text) {
                print_hex(stderr,
                          (const unsigned char *)first_target.text,
                          first_target.text_len);
            } else {
                fprintf(stderr, "none");
            }
            fprintf(stderr, " token=%" PRIu32 " logit=%.8g\n",
                    first_target.token,
                    first_target.logit);
            snprintf(err, errlen, "MTP draft block expected first token mismatch");
            goto done;
        }
    }

    if (ds4_replay_snapshot_create(rt, &snapshot, err, errlen)) goto done;
    const uint64_t snapshot_bytes = ds4_replay_snapshot_bytes(snapshot);

    const uint32_t first_pos = (uint32_t)prompt->len;
    if (ds4_replay_feed_token_at_position(rt,
                                               first_target.token,
                                               first_pos,
                                               &counters,
                                               &counters.continuation_decode_ms,
                                               err,
                                               errlen)) {
        goto done;
    }
    counters.total_input_tokens++;
    if (ds4_replay_read_output_hc_slot(rt,
                                            0,
                                            hc_a,
                                            sizeof(hc_a),
                                            err,
                                            errlen)) {
        goto done;
    }

    uint32_t current_token = first_target.token;
    float *current_hc = hc_a;
    float *next_hc = hc_b;
    double mtp_ms = 0.0;
    for (uint32_t i = 0; i < block_n; i++) {
        if (ds4_replay_read_token_embedding_f32(rt,
                                                     current_token,
                                                     embed,
                                                     DS4_V100_MTP_FORWARD_N_EMBD,
                                                     err,
                                                     errlen)) {
            goto done;
        }
        memset(top_tokens, 0xff, sizeof(top_tokens));
        memset(top_logits, 0, sizeof(top_logits));
        ds4_mtp_forward_report report;
        memset(&report, 0, sizeof(report));
        const double mtp0 = now_ms();
        pthread_mutex_lock(&svc->mu);
        int mtp_rc = ds4_mtp_forward_run_host_next_hc(
            svc->forward,
            embed,
            current_hc,
            first_pos + i,
            svc->top_k,
            top_tokens,
            top_logits,
            next_hc,
            DS4_V100_MTP_FORWARD_HC_VALUES,
            &report,
            err,
            errlen);
        pthread_mutex_unlock(&svc->mu);
        if (mtp_rc) goto done;
        mtp_ms += now_ms() - mtp0;
        last_mtp_report = report;
        draft_tokens[i] = top_tokens[0];
        draft_logits[i] = top_logits[0];
        current_token = draft_tokens[i];
        float *tmp = current_hc;
        current_hc = next_hc;
        next_hc = tmp;
    }

    if (ds4_replay_snapshot_restore(rt, snapshot, err, errlen)) goto done;

    forced_tokens[0] = first_target.token;
    positions[0] = first_pos;
    for (uint32_t i = 1; i < block_n; i++) {
        forced_tokens[i] = draft_tokens[i - 1u];
        positions[i] = first_pos + i;
    }

    if (ds4_replay_verify_token_block(rt,
                                           forced_tokens,
                                           positions,
                                           block_n,
                                           0,
                                           target_outputs,
                                           DS4_V100_REPLAY_MAX_TOKENS,
                                           &verify_report,
                                           &counters,
                                           err,
                                           errlen)) {
        goto done;
    }
    verify_report.snapshot_bytes = snapshot_bytes;

    uint32_t accepted_prefix = 0;
    for (uint32_t i = 0; i < block_n; i++) {
        if (target_outputs[i].token == draft_tokens[i]) accepted_prefix++;
        else break;
    }
    const uint32_t effective_output_tokens = 1u + accepted_prefix;
    const uint32_t speculative_saves =
        effective_output_tokens > verify_report.target_forwards
            ? effective_output_tokens - verify_report.target_forwards
            : 0u;

    counters.generated_tokens = effective_output_tokens;
    counters.total_ms = now_ms() - total0;
    ds4_replay_finish_generation(rt,
                                      counters.generated_tokens,
                                      counters.total_ms,
                                      &counters);

    if (json) {
        printf("{\"mtp_draft_block_smoke\":true,"
               "\"prompt_tokens\":%d,"
               "\"block_tokens\":%" PRIu32 ","
               "\"first_token\":%" PRIu32 ","
               "\"first_hex\":\"",
               prompt->len,
               block_n,
               first_target.token);
        if (first_target.text) {
            print_hex(stdout,
                      (const unsigned char *)first_target.text,
                      first_target.text_len);
        }
        printf("\",\"draft_tokens\":[");
        for (uint32_t i = 0; i < block_n; i++) {
            if (i) printf(",");
            printf("%" PRIu32, draft_tokens[i]);
        }
        printf("],\"target_tokens\":[");
        for (uint32_t i = 0; i < block_n; i++) {
            if (i) printf(",");
            printf("%" PRIu32, target_outputs[i].token);
        }
        printf("],\"accepted_prefix_len\":%" PRIu32
               ",\"target_forwards\":%" PRIu32
               ",\"target_tokens_verified\":%" PRIu32
               ",\"effective_output_tokens\":%" PRIu32
               ",\"speculative_saves\":%" PRIu32
               ",\"snapshot_bytes\":%" PRIu64
               ",\"mtp_ms\":%.3f"
               ",\"verify_ms\":%.3f"
               ",\"mtp_raw_row\":%" PRIu32
               ",\"mtp_n_raw\":%" PRIu32 "}\n",
               accepted_prefix,
               verify_report.target_forwards,
               verify_report.target_tokens_verified,
               effective_output_tokens,
               speculative_saves,
               snapshot_bytes,
               mtp_ms,
               verify_report.verify_ms,
               last_mtp_report.raw_row,
               last_mtp_report.n_raw);
    } else {
        printf("ds4-v100-replay: mtp_draft_block_smoke block_tokens=%" PRIu32
               " first_token=%" PRIu32
               " first_hex=",
               block_n,
               first_target.token);
        if (first_target.text) {
            print_hex(stdout,
                      (const unsigned char *)first_target.text,
                      first_target.text_len);
        } else {
            printf("none");
        }
        printf(" draft_tokens=");
        for (uint32_t i = 0; i < block_n; i++) {
            if (i) printf(",");
            printf("%" PRIu32, draft_tokens[i]);
        }
        printf(" target_tokens=");
        for (uint32_t i = 0; i < block_n; i++) {
            if (i) printf(",");
            printf("%" PRIu32, target_outputs[i].token);
        }
        printf(" accepted_prefix_len=%" PRIu32
               " target_forwards=%" PRIu32
               " target_tokens_verified=%" PRIu32
               " effective_output_tokens=%" PRIu32
               " speculative_saves=%" PRIu32
               " snapshot_bytes=%" PRIu64
               " mtp_ms=%.3f"
               " verify_ms=%.3f"
               " mtp_raw_row=%" PRIu32
               " mtp_n_raw=%" PRIu32
               " ok\n",
               accepted_prefix,
               verify_report.target_forwards,
               verify_report.target_tokens_verified,
               effective_output_tokens,
               speculative_saves,
               snapshot_bytes,
               mtp_ms,
               verify_report.verify_ms,
               last_mtp_report.raw_row,
               last_mtp_report.n_raw);
    }
    rc = 0;

done:
    ds4_replay_snapshot_free(snapshot);
    ds4_replay_output_free(&first_target);
    for (uint32_t i = 0; i < block_n && i < DS4_V100_REPLAY_MAX_TOKENS; i++) {
        ds4_replay_output_free(&target_outputs[i]);
    }
    return rc;
}

static int run_mtp_block2_commit_smoke(replay_mtp_service *svc,
                                       ds4_replay *rt,
                                       const ds4_tokens *prompt,
                                       uint32_t max_tokens,
                                       const unsigned char *expected,
                                       size_t expected_len,
                                       bool json,
                                       char *err,
                                       size_t errlen) {
    if (!svc || !svc->enabled || !rt || !prompt || prompt->len <= 0 ||
        max_tokens < 2 || max_tokens > DS4_V100_REPLAY_MAX_TOKENS) {
        snprintf(err, errlen, "invalid MTP block2 commit smoke input");
        return 1;
    }

    ds4_replay_output baseline[DS4_V100_REPLAY_MAX_TOKENS];
    ds4_replay_output outputs[DS4_V100_REPLAY_MAX_TOKENS];
    ds4_replay_output current;
    memset(baseline, 0, sizeof(baseline));
    memset(outputs, 0, sizeof(outputs));
    memset(&current, 0, sizeof(current));
    ds4_replay_counters baseline_counters;
    ds4_replay_counters counters;
    memset(&baseline_counters, 0, sizeof(baseline_counters));
    memset(&counters, 0, sizeof(counters));
    uint32_t baseline_count = 0;
    uint32_t n_out = 0;
    int rc = 1;

    if (ds4_replay_generate(rt,
                                 prompt,
                                 max_tokens,
                                 baseline,
                                 DS4_V100_REPLAY_MAX_TOKENS,
                                 &baseline_count,
                                 &baseline_counters,
                                 err,
                                 errlen) ||
        baseline_count != max_tokens) {
        if (err[0] == '\0') snprintf(err, errlen, "block2 baseline generation failed");
        goto done;
    }
    if (expected) {
        if (!baseline[0].text ||
            baseline[0].text_len != expected_len ||
            memcmp(baseline[0].text, expected, expected_len) != 0) {
            fprintf(stderr, "ds4-v100-replay: mtp-block2 selected token mismatch expected=");
            print_hex(stderr, expected, expected_len);
            fprintf(stderr, " got=");
            if (baseline[0].text) {
                print_hex(stderr,
                          (const unsigned char *)baseline[0].text,
                          baseline[0].text_len);
            } else {
                fprintf(stderr, "none");
            }
            fprintf(stderr, " token=%" PRIu32 " logit=%.8g\n",
                    baseline[0].token,
                    baseline[0].logit);
            snprintf(err, errlen, "MTP block2 expected first token mismatch");
            goto done;
        }
    }

    if (ds4_replay_reset(rt, err, errlen) ||
        ds4_replay_begin_generation(rt,
                                         (uint32_t)prompt->len,
                                         &counters,
                                         err,
                                         errlen)) {
        goto done;
    }

    const double total0 = now_ms();
    for (int pos = 0; pos < prompt->len; pos++) {
        if (prompt->v[pos] < 0) {
            snprintf(err, errlen, "negative prompt token at position %d", pos);
            goto done;
        }
        if (ds4_replay_feed_token_at_position(rt,
                                                   (uint32_t)prompt->v[pos],
                                                   (uint32_t)pos,
                                                   &counters,
                                                   &counters.prompt_replay_ms,
                                                   err,
                                                   errlen)) {
            goto done;
        }
        counters.total_input_tokens++;
    }

    if (ds4_replay_select_current_token(rt,
                                             &current,
                                             &counters,
                                             err,
                                             errlen)) {
        goto done;
    }

    uint32_t target_forwards = 0;
    uint32_t target_tokens_verified = 0;
    uint32_t accepted_prefix_total = 0;
    uint32_t draft_tokens_proposed = 0;
    uint32_t draft_tokens_accepted = 0;
    uint32_t blocks = 0;
    uint32_t block2_full_accepts = 0;
    uint32_t block2_partial_accepts = 0;
    uint32_t block2_rejects = 0;
    double mtp_ms = 0.0;
    double verify_ms = 0.0;
    uint32_t last_raw_row = 0;
    uint32_t last_n_raw = 0;

    if (replay_output_copy(&outputs[n_out], &current, err, errlen)) goto done;
    n_out++;

    while (n_out < max_tokens) {
        const uint32_t remaining = max_tokens - n_out;
        if (remaining < 2) {
            const uint32_t pos = (uint32_t)prompt->len + n_out - 1u;
            const double t0 = now_ms();
            if (ds4_replay_feed_token_at_position(rt,
                                                       current.token,
                                                       pos,
                                                       &counters,
                                                       &counters.continuation_decode_ms,
                                                       err,
                                                       errlen)) {
                goto done;
            }
            counters.total_input_tokens++;
            ds4_replay_output next;
            memset(&next, 0, sizeof(next));
            if (ds4_replay_select_current_token(rt,
                                                     &next,
                                                     &counters,
                                                     err,
                                                     errlen)) {
                ds4_replay_output_free(&next);
                goto done;
            }
            verify_ms += now_ms() - t0;
            target_forwards++;
            ds4_replay_output_free(&current);
            current = next;
            if (replay_output_copy(&outputs[n_out], &current, err, errlen)) goto done;
            n_out++;
            continue;
        }

        blocks++;
        const uint32_t pos = (uint32_t)prompt->len + n_out - 1u;

        ds4_replay_output target0;
        ds4_replay_output target1;
        memset(&target0, 0, sizeof(target0));
        memset(&target1, 0, sizeof(target1));
        uint32_t draft_tokens[2] = { UINT32_MAX, UINT32_MAX };
        float draft_logits[2] = { 0.0f, 0.0f };
        uint32_t top_tokens[DS4_V100_MTP_FORWARD_MAX_TOPK];
        float top_logits[DS4_V100_MTP_FORWARD_MAX_TOPK];
        float embed[DS4_V100_MTP_FORWARD_N_EMBD];
        float hc_a[DS4_V100_MTP_FORWARD_HC_VALUES];
        float hc_b[DS4_V100_MTP_FORWARD_HC_VALUES];
        memset(embed, 0, sizeof(embed));
        memset(hc_a, 0, sizeof(hc_a));
        memset(hc_b, 0, sizeof(hc_b));

        const double verify0 = now_ms();
        if (ds4_replay_feed_token_at_position(rt,
                                                   current.token,
                                                   pos,
                                                   &counters,
                                                   &counters.continuation_decode_ms,
                                                   err,
                                                   errlen)) {
            ds4_replay_output_free(&target0);
            ds4_replay_output_free(&target1);
            goto done;
        }
        counters.total_input_tokens++;
        if (ds4_replay_read_output_hc_slot(rt,
                                                0,
                                                hc_a,
                                                sizeof(hc_a),
                                                err,
                                                errlen) ||
            ds4_replay_select_current_token(rt,
                                                 &target0,
                                                 &counters,
                                                 err,
                                                 errlen)) {
            ds4_replay_output_free(&target0);
            ds4_replay_output_free(&target1);
            goto done;
        }
        verify_ms += now_ms() - verify0;
        target_forwards++;
        target_tokens_verified++;

        uint32_t current_token = current.token;
        float *current_hc = hc_a;
        float *next_hc = hc_b;
        for (uint32_t i = 0; i < 2; i++) {
            if (ds4_replay_read_token_embedding_f32(rt,
                                                         current_token,
                                                         embed,
                                                         DS4_V100_MTP_FORWARD_N_EMBD,
                                                         err,
                                                         errlen)) {
                ds4_replay_output_free(&target0);
                ds4_replay_output_free(&target1);
                goto done;
            }
            memset(top_tokens, 0xff, sizeof(top_tokens));
            memset(top_logits, 0, sizeof(top_logits));
            ds4_mtp_forward_report report;
            memset(&report, 0, sizeof(report));
            const double mtp0 = now_ms();
            pthread_mutex_lock(&svc->mu);
            int mtp_rc = ds4_mtp_forward_run_host_next_hc(
                svc->forward,
                embed,
                current_hc,
                pos + i,
                svc->top_k,
                top_tokens,
                top_logits,
                next_hc,
                DS4_V100_MTP_FORWARD_HC_VALUES,
                &report,
                err,
                errlen);
            pthread_mutex_unlock(&svc->mu);
            if (mtp_rc) {
                ds4_replay_output_free(&target0);
                ds4_replay_output_free(&target1);
                goto done;
            }
            mtp_ms += now_ms() - mtp0;
            draft_tokens[i] = top_tokens[0];
            draft_logits[i] = top_logits[0];
            current_token = draft_tokens[i];
            float *tmp = current_hc;
            current_hc = next_hc;
            next_hc = tmp;
            last_raw_row = report.raw_row;
            last_n_raw = report.n_raw;
            draft_tokens_proposed++;
        }

        uint32_t accepted_prefix = 0;
        if (target0.token == draft_tokens[0]) {
            accepted_prefix = 1;
            const double verify1 = now_ms();
            if (ds4_replay_feed_token_at_position(rt,
                                                       draft_tokens[0],
                                                       pos + 1u,
                                                       &counters,
                                                       &counters.continuation_decode_ms,
                                                       err,
                                                       errlen)) {
                ds4_replay_output_free(&target0);
                ds4_replay_output_free(&target1);
                goto done;
            }
            counters.total_input_tokens++;
            if (ds4_replay_select_current_token(rt,
                                                     &target1,
                                                     &counters,
                                                     err,
                                                     errlen)) {
                ds4_replay_output_free(&target0);
                ds4_replay_output_free(&target1);
                goto done;
            }
            verify_ms += now_ms() - verify1;
            target_forwards++;
            target_tokens_verified++;
            if (target1.token == draft_tokens[1]) accepted_prefix = 2;
        }

        accepted_prefix_total += accepted_prefix;
        draft_tokens_accepted += accepted_prefix;
        if (accepted_prefix == 2) block2_full_accepts++;
        else if (accepted_prefix == 1) block2_partial_accepts++;
        else block2_rejects++;

        if (replay_output_copy(&outputs[n_out], &target0, err, errlen)) {
            ds4_replay_output_free(&target0);
            ds4_replay_output_free(&target1);
            goto done;
        }
        n_out++;
        ds4_replay_output_free(&current);

        if (accepted_prefix >= 1 && n_out < max_tokens) {
            if (replay_output_copy(&outputs[n_out], &target1, err, errlen)) {
                ds4_replay_output_free(&target0);
                ds4_replay_output_free(&target1);
                goto done;
            }
            n_out++;
            if (replay_output_copy(&current, &target1, err, errlen)) {
                ds4_replay_output_free(&target0);
                ds4_replay_output_free(&target1);
                goto done;
            }
        } else {
            if (replay_output_copy(&current, &target0, err, errlen)) {
                ds4_replay_output_free(&target0);
                ds4_replay_output_free(&target1);
                goto done;
            }
        }
        ds4_replay_output_free(&target0);
        ds4_replay_output_free(&target1);
        (void)draft_logits;
    }

    counters.generated_tokens = n_out;
    counters.total_ms = now_ms() - total0;
    ds4_replay_finish_generation(rt, counters.generated_tokens, counters.total_ms, &counters);

    if (n_out != baseline_count) {
        snprintf(err, errlen, "MTP block2 output count mismatch baseline=%" PRIu32 " got=%" PRIu32,
                 baseline_count, n_out);
        goto done;
    }
    for (uint32_t i = 0; i < n_out; i++) {
        if (outputs[i].token != baseline[i].token) {
            snprintf(err,
                     errlen,
                     "MTP block2 token mismatch at %" PRIu32 ": baseline=%" PRIu32 " got=%" PRIu32,
                     i,
                     baseline[i].token,
                     outputs[i].token);
            goto done;
        }
    }

    const uint32_t speculative_saves =
        n_out > target_forwards ? n_out - target_forwards : 0u;
    const double block2_tps =
        counters.total_ms > 0.0 ? (double)counters.generated_tokens * 1000.0 / counters.total_ms : 0.0;
    const double baseline_tps =
        baseline_counters.total_ms > 0.0
            ? (double)baseline_counters.generated_tokens * 1000.0 / baseline_counters.total_ms
            : 0.0;

    if (json) {
        printf("{\"mtp_block2_commit_smoke\":true,"
               "\"prompt_tokens\":%d,"
               "\"generated_tokens\":%" PRIu32 ","
               "\"baseline_tokens\":%" PRIu32 ","
               "\"token_match\":true,"
               "\"blocks\":%" PRIu32 ","
               "\"block2_full_accepts\":%" PRIu32 ","
               "\"block2_partial_accepts\":%" PRIu32 ","
               "\"block2_rejects\":%" PRIu32 ","
               "\"accepted_prefix_total\":%" PRIu32 ","
               "\"draft_tokens_proposed\":%" PRIu32 ","
               "\"draft_tokens_accepted\":%" PRIu32 ","
               "\"target_forwards\":%" PRIu32 ","
               "\"target_tokens_verified\":%" PRIu32 ","
               "\"effective_output_tokens\":%" PRIu32 ","
               "\"speculative_saves\":%" PRIu32 ","
               "\"mtp_ms\":%.3f,"
               "\"verify_ms\":%.3f,"
               "\"block2_total_ms\":%.3f,"
               "\"baseline_total_ms\":%.3f,"
               "\"block2_generated_tps\":%.6f,"
               "\"baseline_generated_tps\":%.6f,"
               "\"last_mtp_raw_row\":%" PRIu32 ","
               "\"last_mtp_n_raw\":%" PRIu32 "}\n",
               prompt->len,
               n_out,
               baseline_count,
               blocks,
               block2_full_accepts,
               block2_partial_accepts,
               block2_rejects,
               accepted_prefix_total,
               draft_tokens_proposed,
               draft_tokens_accepted,
               target_forwards,
               target_tokens_verified,
               n_out,
               speculative_saves,
               mtp_ms,
               verify_ms,
               counters.total_ms,
               baseline_counters.total_ms,
               block2_tps,
               baseline_tps,
               last_raw_row,
               last_n_raw);
    } else {
        printf("ds4-v100-replay: mtp_block2_commit_smoke generated_tokens=%" PRIu32
               " baseline_tokens=%" PRIu32
               " token_match=true blocks=%" PRIu32
               " full_accepts=%" PRIu32
               " partial_accepts=%" PRIu32
               " rejects=%" PRIu32
               " accepted_prefix_total=%" PRIu32
               " draft_tokens_proposed=%" PRIu32
               " draft_tokens_accepted=%" PRIu32
               " target_forwards=%" PRIu32
               " target_tokens_verified=%" PRIu32
               " effective_output_tokens=%" PRIu32
               " speculative_saves=%" PRIu32
               " mtp_ms=%.3f verify_ms=%.3f"
               " block2_total_ms=%.3f baseline_total_ms=%.3f"
               " block2_generated_tps=%.6f baseline_generated_tps=%.6f ok\n",
               n_out,
               baseline_count,
               blocks,
               block2_full_accepts,
               block2_partial_accepts,
               block2_rejects,
               accepted_prefix_total,
               draft_tokens_proposed,
               draft_tokens_accepted,
               target_forwards,
               target_tokens_verified,
               n_out,
               speculative_saves,
               mtp_ms,
               verify_ms,
               counters.total_ms,
               baseline_counters.total_ms,
               block2_tps,
               baseline_tps);
    }
    rc = 0;

done:
    for (uint32_t i = 0; i < baseline_count && i < DS4_V100_REPLAY_MAX_TOKENS; i++) {
        ds4_replay_output_free(&baseline[i]);
    }
    for (uint32_t i = 0; i < n_out && i < DS4_V100_REPLAY_MAX_TOKENS; i++) {
        ds4_replay_output_free(&outputs[i]);
    }
    ds4_replay_output_free(&current);
    return rc;
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
    const bool warmup_required =
        opt && opt->ctx == 262144 && opt->active_microbatch > 16;
    const bool warmed_ready = !warmup_required || (opt && opt->startup_warmup);
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
    fprintf(fp, "\"decode_profile\":%s,", opt->profile_decode ? "true" : "false");
    fprintf(fp, "\"wavefront_decode\":%s,", opt->wavefront_decode ? "true" : "false");
    fprintf(fp, "\"async_pipeline_decode\":%s,", opt->async_pipeline_decode ? "true" : "false");
    fprintf(fp,
            "\"async_pipeline_mode\":\"%s\",",
            async_pipeline_mode_name(opt->async_pipeline_mode));
    fprintf(fp, "\"async_handoff\":%s,", opt->async_handoff ? "true" : "false");
    fprintf(fp, "\"async_event_handoff\":%s,", opt->async_event_handoff ? "true" : "false");
    fprintf(fp, "\"startup_warmup\":%s,", opt->startup_warmup ? "true" : "false");
    fprintf(fp, "\"warmup_required\":%s,", warmup_required ? "true" : "false");
    fprintf(fp, "\"warmed_ready\":%s,", warmed_ready ? "true" : "false");
    fprintf(fp, "\"microbatch_wait_us\":%" PRIu32 ",", opt->microbatch_wait_us);
    fprintf(fp,
            "\"limits\":{\"slots\":%" PRIu32 ",\"configured_slots\":%" PRIu32
            ",\"active_slots\":%" PRIu32 ",\"active_microbatch\":%" PRIu32
            ",\"microbatch_wait_us\":%" PRIu32
            ",\"concurrent_requests\":%" PRIu32
            ",\"queue_policy\":\"%s\",\"scheduler_slots_ready\":true,"
            "\"tensor_batched_slots\":%s,\"sequential_requests\":%s,"
            "\"streaming\":false,\"external_exposure\":false,"
            "\"speculative_serving\":%s},",
            opt->slots,
            opt->slots,
            opt->active_microbatch,
            opt->active_microbatch,
            opt->microbatch_wait_us,
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
        fprintf(fp, "\"serving_mode\":\"%s\",", mtp_serving_mode_name(opt->mtp_serving));
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
        fprintf(fp, "\"skipped\":%" PRIu64 ",", mtp->skipped);
        fprintf(fp, "\"committed\":%" PRIu64, mtp->committed);
        fprintf(fp, "}");
    }
    fprintf(fp, "}\n");
}

static void write_metrics_text(FILE *fp,
                               const replay_cli_options *opt,
                               const replay_mtp_service *mtp,
                               const replay_server_stats *stats) {
    const bool mtp_enabled = mtp && mtp->enabled;
    const bool warmup_required =
        opt && opt->ctx == 262144 && opt->active_microbatch > 16;
    const bool warmed_ready = !warmup_required || (opt && opt->startup_warmup);
    const uint64_t served = stats ? stats->accepted_connections : 0;
    fprintf(fp, "# HELP ds4_readiness_level Deployment readiness level exposed by the replay service.\n");
    fprintf(fp, "# TYPE ds4_readiness_level gauge\n");
    fprintf(fp, "ds4_readiness_level %d\n", mtp_enabled ? 3 : 2);
    fprintf(fp, "# HELP ds4_served_requests HTTP requests accepted by this process.\n");
    fprintf(fp, "# TYPE ds4_served_requests counter\n");
    fprintf(fp, "ds4_served_requests %" PRIu64 "\n", served);
    fprintf(fp, "# HELP ds4_generation_requests_total Generation requests accepted by the scheduler.\n");
    fprintf(fp, "# TYPE ds4_generation_requests_total counter\n");
    fprintf(fp, "ds4_generation_requests_total %" PRIu64 "\n", stats ? stats->generation_requests : 0);
    fprintf(fp, "# HELP ds4_tensor_batched_groups_total Same-token-count tensor batch groups executed by the scheduler.\n");
    fprintf(fp, "# TYPE ds4_tensor_batched_groups_total counter\n");
    fprintf(fp, "ds4_tensor_batched_groups_total %" PRIu64 "\n", stats ? stats->tensor_batched_groups : 0);
    fprintf(fp, "# HELP ds4_tensor_batched_requests_total Requests served through tensor batch groups.\n");
    fprintf(fp, "# TYPE ds4_tensor_batched_requests_total counter\n");
    fprintf(fp, "ds4_tensor_batched_requests_total %" PRIu64 "\n", stats ? stats->tensor_batched_requests : 0);
    fprintf(fp, "# HELP ds4_tensor_batched_tokens_total Generated tokens served through tensor batch groups.\n");
    fprintf(fp, "# TYPE ds4_tensor_batched_tokens_total counter\n");
    fprintf(fp, "ds4_tensor_batched_tokens_total %" PRIu64 "\n", stats ? stats->tensor_batched_tokens : 0);
    fprintf(fp, "# HELP ds4_rejected_requests_total HTTP generation requests rejected by admission policy.\n");
    fprintf(fp, "# TYPE ds4_rejected_requests_total counter\n");
    fprintf(fp, "ds4_rejected_requests_total %" PRIu64 "\n", stats ? stats->rejected_requests : 0);
    fprintf(fp, "# HELP ds4_rejected_busy_total HTTP generation requests rejected because the scheduler was busy.\n");
    fprintf(fp, "# TYPE ds4_rejected_busy_total counter\n");
    fprintf(fp, "ds4_rejected_busy_total %" PRIu64 "\n", stats ? stats->rejected_busy : 0);
    fprintf(fp, "# HELP ds4_rejected_context_total HTTP generation requests rejected for exceeding context.\n");
    fprintf(fp, "# TYPE ds4_rejected_context_total counter\n");
    fprintf(fp, "ds4_rejected_context_total %" PRIu64 "\n", stats ? stats->rejected_context : 0);
    fprintf(fp, "# HELP ds4_rejected_bad_request_total HTTP generation requests rejected for malformed input.\n");
    fprintf(fp, "# TYPE ds4_rejected_bad_request_total counter\n");
    fprintf(fp, "ds4_rejected_bad_request_total %" PRIu64 "\n", stats ? stats->rejected_bad_request : 0);
    fprintf(fp, "# HELP ds4_ctx_tokens Configured KV context tokens per slot.\n");
    fprintf(fp, "# TYPE ds4_ctx_tokens gauge\n");
    fprintf(fp, "ds4_ctx_tokens %" PRIu64 "\n", opt->ctx);
    fprintf(fp, "# HELP ds4_default_tokens Default generated tokens per request.\n");
    fprintf(fp, "# TYPE ds4_default_tokens gauge\n");
    fprintf(fp, "ds4_default_tokens %" PRIu32 "\n", opt->tokens);
    fprintf(fp, "# HELP ds4_max_tokens Maximum generated tokens accepted by the appliance endpoint.\n");
    fprintf(fp, "# TYPE ds4_max_tokens gauge\n");
    fprintf(fp, "ds4_max_tokens %u\n", DS4_V100_REPLAY_MAX_TOKENS);
    fprintf(fp, "# HELP ds4_configured_slots Configured admission slots.\n");
    fprintf(fp, "# TYPE ds4_configured_slots gauge\n");
    fprintf(fp, "ds4_configured_slots %" PRIu32 "\n", opt->slots);
    fprintf(fp, "# HELP ds4_active_microbatch Active decode requests supported by this process.\n");
    fprintf(fp, "# TYPE ds4_active_microbatch gauge\n");
    fprintf(fp, "ds4_active_microbatch %" PRIu32 "\n", opt->active_microbatch);
    fprintf(fp, "# HELP ds4_microbatch_wait_us Max microseconds the server waits to coalesce active requests.\n");
    fprintf(fp, "# TYPE ds4_microbatch_wait_us gauge\n");
    fprintf(fp, "ds4_microbatch_wait_us %" PRIu32 "\n", opt->microbatch_wait_us);
    fprintf(fp, "# HELP ds4_active_slots Active slots scheduled concurrently by this process.\n");
    fprintf(fp, "# TYPE ds4_active_slots gauge\n");
    fprintf(fp, "ds4_active_slots %" PRIu32 "\n", opt->active_microbatch);
    fprintf(fp, "# HELP ds4_startup_warmup_enabled Whether the server warmed the appliance before accepting traffic.\n");
    fprintf(fp, "# TYPE ds4_startup_warmup_enabled gauge\n");
    fprintf(fp, "ds4_startup_warmup_enabled %d\n", opt->startup_warmup ? 1 : 0);
    fprintf(fp, "# HELP ds4_warmup_required Whether this context/slot shape requires startup warmup for production readiness.\n");
    fprintf(fp, "# TYPE ds4_warmup_required gauge\n");
    fprintf(fp, "ds4_warmup_required %d\n", warmup_required ? 1 : 0);
    fprintf(fp, "# HELP ds4_warmed_ready Whether the server satisfies the warmed-readiness contract before accepting traffic.\n");
    fprintf(fp, "# TYPE ds4_warmed_ready gauge\n");
    fprintf(fp, "ds4_warmed_ready %d\n", warmed_ready ? 1 : 0);
    fprintf(fp, "# HELP ds4_concurrent_request_capacity Concurrent generation request capacity.\n");
    fprintf(fp, "# TYPE ds4_concurrent_request_capacity gauge\n");
    fprintf(fp, "ds4_concurrent_request_capacity %" PRIu32 "\n", opt->active_microbatch);
    fprintf(fp, "# HELP ds4_scheduler_slots_ready Whether true device-resident multi-slot scheduling is implemented.\n");
    fprintf(fp, "# TYPE ds4_scheduler_slots_ready gauge\n");
    fprintf(fp, "ds4_scheduler_slots_ready 1\n");
    fprintf(fp, "# HELP ds4_mtp_enabled Whether speculative MTP serving is enabled.\n");
    fprintf(fp, "# TYPE ds4_mtp_enabled gauge\n");
    fprintf(fp, "ds4_mtp_enabled %d\n", mtp_enabled ? 1 : 0);
    fprintf(fp, "# HELP ds4_mtp_requests_total Requests evaluated by the MTP service.\n");
    fprintf(fp, "# TYPE ds4_mtp_requests_total counter\n");
    fprintf(fp, "ds4_mtp_requests_total %" PRIu64 "\n", mtp_enabled ? mtp->requests : 0);
    fprintf(fp, "# HELP ds4_mtp_drafts_total MTP draft attempts completed.\n");
    fprintf(fp, "# TYPE ds4_mtp_drafts_total counter\n");
    fprintf(fp, "ds4_mtp_drafts_total %" PRIu64 "\n", mtp_enabled ? mtp->drafts : 0);
    fprintf(fp, "# HELP ds4_mtp_accepted_total MTP drafts matching the base target token.\n");
    fprintf(fp, "# TYPE ds4_mtp_accepted_total counter\n");
    fprintf(fp, "ds4_mtp_accepted_total %" PRIu64 "\n", mtp_enabled ? mtp->accepted : 0);
    fprintf(fp, "# HELP ds4_mtp_rejected_total MTP drafts rejected by exact target-token comparison.\n");
    fprintf(fp, "# TYPE ds4_mtp_rejected_total counter\n");
    fprintf(fp, "ds4_mtp_rejected_total %" PRIu64 "\n", mtp_enabled ? mtp->rejected : 0);
    fprintf(fp, "# HELP ds4_mtp_skipped_total Requests where MTP was skipped.\n");
    fprintf(fp, "# TYPE ds4_mtp_skipped_total counter\n");
    fprintf(fp, "ds4_mtp_skipped_total %" PRIu64 "\n", mtp_enabled ? mtp->skipped : 0);
    fprintf(fp, "# HELP ds4_mtp_committed_total Accepted MTP drafts emitted by commit-mode serving.\n");
    fprintf(fp, "# TYPE ds4_mtp_committed_total counter\n");
    fprintf(fp, "ds4_mtp_committed_total %" PRIu64 "\n", mtp_enabled ? mtp->committed : 0);
}

static int handle_http_request(int fd, replay_server_state *state) {
    if (!state || !state->opt || !state->rt) {
        http_error(fd, 500, "server_state_missing");
        return 1;
    }
    ds4_replay *rt = state->rt;
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
    ds4_replay_encode_prompt(rt, opt->system, prompt, DS4_THINK_NONE, &pending.prompt_tokens);
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
            if (opt->mtp_serving == REPLAY_MTP_SERVING_COMMIT) {
                if (pending.mtp_result_ready) {
                    mtp_json = &pending.mtp_result;
                } else {
                    http_error(fd, 500, "mtp_commit_result_missing");
                    rc = 1;
                }
            } else if (pending.mtp_result_ready) {
                mtp_json = &pending.mtp_result;
            } else {
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
    }
    if (!rc) {
        http_ok_json_begin(fd);
        FILE *fp = fdopen(dup(fd), "w");
        if (fp) {
            print_json_fp(fp, pending.outputs, pending.n_outputs, &pending.counters, mtp_json);
            fclose(fp);
        }
    }
    for (uint32_t i = 0; i < pending.n_outputs; i++) ds4_replay_output_free(&pending.outputs[i]);
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

static int run_startup_warmup(const replay_cli_options *opt, ds4_replay *rt) {
    if (!opt || !rt || !opt->startup_warmup) return 0;
    static const char warmup_prompt[] = "16";
    char err[512] = {0};
    ds4_tokens prompt = {0};
    ds4_replay_output output = {0};
    ds4_replay_counters counters;
    memset(&counters, 0, sizeof(counters));

    ds4_replay_encode_prompt(rt, opt->system, warmup_prompt, DS4_THINK_NONE, &prompt);
    if (prompt.len <= 0) {
        fprintf(stderr, "ds4-v100-replay: startup warmup prompt encode failed\n");
        ds4_tokens_free(&prompt);
        return 1;
    }
    if (opt->ctx && (uint64_t)prompt.len + 1u > opt->ctx) {
        fprintf(stderr, "ds4-v100-replay: startup warmup exceeds configured context\n");
        ds4_tokens_free(&prompt);
        return 1;
    }

    uint32_t n_outputs = 0;
    const double t0 = now_ms();
    int rc = ds4_replay_generate(rt,
                                      &prompt,
                                      1,
                                      &output,
                                      1,
                                      &n_outputs,
                                      &counters,
                                      err,
                                      sizeof(err));
    if (!rc) {
        rc = ds4_replay_reset(rt, err, sizeof(err));
    }
    const double total_ms = now_ms() - t0;
    if (rc) {
        fprintf(stderr,
                "ds4-v100-replay: startup warmup failed: %s\n",
                err[0] ? err : "warmup generation failed");
    } else {
        fprintf(stderr,
                "ds4-v100-replay: startup warmup ok prompt_tokens=%d token=%" PRIu32
                " total_ms=%.3f\n",
                prompt.len,
                n_outputs ? output.token : UINT32_MAX,
                total_ms);
    }
    ds4_replay_output_free(&output);
    ds4_tokens_free(&prompt);
    return rc;
}

static int run_server(const replay_cli_options *opt,
                      ds4_replay *rt,
                      replay_mtp_service *mtp) {
    if (run_startup_warmup(opt, rt) != 0) return 1;

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
    const int backlog = opt->active_microbatch > 64u ? (int)opt->active_microbatch : 64;
    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(listen_fd, backlog) != 0) {
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
    const bool synthetic_prompt = opt.synthetic_prompt_len != 0;
    char *prompt_owned = NULL;
    const char *prompt_text = NULL;
    if (!opt.serve && !opt.open_only && !synthetic_prompt) {
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

    ds4_replay_options ropts;
    ds4_replay_options_init(&ropts);
    ropts.model_path = opt.model_path;
    ropts.pack_index_path = opt.index_path;
    ropts.turbomind_pack_index_path = opt.turbomind_index_path;
    ropts.shard_dir = opt.shard_dir;
    ropts.kv_ctx_tokens = opt.ctx;
    ropts.kv_active_slots = opt.slots;
    ropts.serial_open = opt.serial_open;
    ropts.wavefront_decode = opt.wavefront_decode;
    ropts.async_pipeline_decode = opt.async_pipeline_decode;
    ropts.async_handoff = opt.async_handoff;
    ropts.async_event_handoff = opt.async_event_handoff;
    ropts.async_pipeline_mode = opt.async_pipeline_mode;
    if (synthetic_prompt) {
        uint64_t comp_cap = ((uint64_t)opt.synthetic_prompt_len + opt.tokens + 3ull) / 4ull + 4ull;
        if (comp_cap < ropts.attn_comp_cap) comp_cap = ropts.attn_comp_cap;
        if (comp_cap > UINT32_MAX) {
            fprintf(stderr, "ds4-v100-replay: synthetic prompt compressed cache cap is too large\n");
            free(expected);
            free(prompt_owned);
            return 2;
        }
        ropts.attn_comp_cap = (uint32_t)comp_cap;
        ropts.index_comp_cap = (uint32_t)comp_cap;
    } else if (prompt_text) {
        uint64_t prompt_cap_basis = ((uint64_t)strlen(prompt_text) +
                                     (uint64_t)strlen(opt.system) + 1ull) / 2ull;
        if (opt.prompt_token_limit && opt.prompt_token_limit < prompt_cap_basis) {
            prompt_cap_basis = opt.prompt_token_limit;
        }
        uint64_t comp_cap = prompt_cap_basis + (uint64_t)opt.tokens + 64ull;
        if (comp_cap < ropts.attn_comp_cap) comp_cap = ropts.attn_comp_cap;
        if (comp_cap > UINT32_MAX) {
            fprintf(stderr, "ds4-v100-replay: prompt compressed cache cap is too large\n");
            free(expected);
            free(prompt_owned);
            return 2;
        }
        ropts.attn_comp_cap = (uint32_t)comp_cap;
        ropts.index_comp_cap = (uint32_t)comp_cap;
    }
    if (opt.profile_decode) {
        setenv("DS4_V100_PROFILE_DECODE", "1", 1);
    }

    char err[512] = {0};
    ds4_replay *rt = NULL;
    if (ds4_replay_open(&rt, &ropts, err, sizeof(err))) {
        fprintf(stderr, "ds4-v100-replay: %s\n", err[0] ? err : "open failed");
        free(expected);
        free(prompt_owned);
        return 1;
    }

    if (opt.open_only) {
        ds4_replay_counters counters;
        ds4_replay_open_counters(rt, &counters);
        if (opt.json) {
            print_open_json(&counters, opt.serial_open);
        } else {
            printf("open_mode\t%s\n", opt.serial_open ? "serial" : "parallel");
            printf("open_total_ms\t%.3f\n", counters.open_total_ms);
            for (int i = 0; i < DS4_V100_EXPECTED_GPUS; i++) {
                printf("open_stage_ms\t%d\t%.3f\n", i, counters.open_ms[i]);
            }
        }
        ds4_replay_close(rt);
        free(expected);
        return 0;
    }

    replay_mtp_service *mtp = NULL;
    if (replay_mtp_service_open(&mtp, &opt, rt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-v100-replay: %s\n", err[0] ? err : "MTP service open failed");
        ds4_replay_close(rt);
        free(expected);
        free(prompt_owned);
        return 1;
    }

    if (opt.serve) {
        int rc = run_server(&opt, rt, mtp);
        replay_mtp_service_close(mtp);
        ds4_replay_close(rt);
        free(expected);
        return rc;
    }

    ds4_tokens prompt = {0};
    if (synthetic_prompt) {
        for (uint32_t i = 0; i < opt.synthetic_prompt_len; i++) {
            ds4_tokens_push(&prompt, (int)opt.synthetic_prompt_token);
        }
    } else {
        ds4_replay_encode_prompt(rt, opt.system, prompt_text, DS4_THINK_NONE, &prompt);
    }
    if (opt.prompt_token_limit && prompt.len > (int)opt.prompt_token_limit) {
        prompt.len = (int)opt.prompt_token_limit;
    }
    if (opt.ctx && (uint64_t)prompt.len + opt.tokens > opt.ctx) {
        fprintf(stderr, "ds4-v100-replay: prompt exceeds configured context\n");
        replay_mtp_service_close(mtp);
        ds4_replay_close(rt);
        ds4_tokens_free(&prompt);
        free(expected);
        free(prompt_owned);
        return 1;
    }
    if (opt.reset_parity_smoke) {
        int smoke_rc = run_reset_parity_smoke(rt,
                                              &prompt,
                                              opt.reset_parity_smoke,
                                              expected,
                                              expected_len,
                                              opt.json,
                                              err,
                                              sizeof(err));
        if (smoke_rc) {
            fprintf(stderr,
                    "ds4-v100-replay: %s\n",
                    err[0] ? err : "reset parity smoke failed");
        }
        replay_mtp_service_close(mtp);
        ds4_replay_close(rt);
        ds4_tokens_free(&prompt);
        free(expected);
        free(prompt_owned);
        return smoke_rc;
    }
    if (opt.target_block_smoke) {
        int smoke_rc = run_target_block_smoke(rt,
                                              &prompt,
                                              opt.tokens,
                                              opt.target_block_smoke,
                                              expected,
                                              expected_len,
                                              opt.json,
                                              err,
                                              sizeof(err));
        if (smoke_rc) {
            fprintf(stderr,
                    "ds4-v100-replay: %s\n",
                    err[0] ? err : "target block smoke failed");
        }
        replay_mtp_service_close(mtp);
        ds4_replay_close(rt);
        ds4_tokens_free(&prompt);
        free(expected);
        free(prompt_owned);
        return smoke_rc;
    }
    if (opt.mtp_draft_block_smoke) {
        int smoke_rc = run_mtp_draft_block_smoke(mtp,
                                                 rt,
                                                 &prompt,
                                                 opt.tokens,
                                                 opt.mtp_draft_block_smoke,
                                                 expected,
                                                 expected_len,
                                                 opt.json,
                                                 err,
                                                 sizeof(err));
        if (smoke_rc) {
            fprintf(stderr,
                    "ds4-v100-replay: %s\n",
                    err[0] ? err : "MTP draft block smoke failed");
        }
        replay_mtp_service_close(mtp);
        ds4_replay_close(rt);
        ds4_tokens_free(&prompt);
        free(expected);
        free(prompt_owned);
        return smoke_rc;
    }
    if (opt.mtp_block2_commit_smoke) {
        int smoke_rc = run_mtp_block2_commit_smoke(mtp,
                                                   rt,
                                                   &prompt,
                                                   opt.mtp_block2_commit_smoke,
                                                   expected,
                                                   expected_len,
                                                   opt.json,
                                                   err,
                                                   sizeof(err));
        if (smoke_rc) {
            fprintf(stderr,
                    "ds4-v100-replay: %s\n",
                    err[0] ? err : "MTP block2 commit smoke failed");
        }
        replay_mtp_service_close(mtp);
        ds4_replay_close(rt);
        ds4_tokens_free(&prompt);
        free(expected);
        free(prompt_owned);
        return smoke_rc;
    }
    ds4_replay_output outputs[DS4_V100_REPLAY_MAX_TOKENS];
    memset(outputs, 0, sizeof(outputs));
    ds4_replay_counters counters;
    memset(&counters, 0, sizeof(counters));
    uint32_t n_outputs = 0;
    int rc = 0;
    replay_mtp_result mtp_result;
    const replay_mtp_result *mtp_json = NULL;
    bool profiler_active = false;
    if (opt.cuda_profiler_window) {
        if (!ds4_gpu_profiler_start()) {
            fprintf(stderr, "ds4-v100-replay: cuda profiler start failed\n");
            rc = 1;
        } else {
            profiler_active = true;
        }
    }
    if (rc == 0 && mtp && mtp->enabled && opt.mtp_serving == REPLAY_MTP_SERVING_COMMIT) {
        if (replay_generate_mtp_commit_one_slot(mtp,
                                                rt,
                                                &prompt,
                                                opt.tokens,
                                                outputs,
                                                opt.tokens,
                                                &n_outputs,
                                                &counters,
                                                &mtp_result,
                                                err,
                                                sizeof(err)) != 0) {
            fprintf(stderr, "ds4-v100-replay: %s\n", err[0] ? err : "MTP commit generation failed");
            rc = 1;
        } else {
            mtp_json = &mtp_result;
        }
    } else if (rc == 0 && ds4_replay_generate(rt,
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
    if (rc == 0 && mtp && mtp->enabled && opt.mtp_serving != REPLAY_MTP_SERVING_COMMIT) {
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
    if (profiler_active && !ds4_gpu_profiler_stop()) {
        fprintf(stderr, "ds4-v100-replay: cuda profiler stop failed\n");
        rc = 1;
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

    for (uint32_t i = 0; i < n_outputs; i++) ds4_replay_output_free(&outputs[i]);
    ds4_tokens_free(&prompt);
    replay_mtp_service_close(mtp);
    ds4_replay_close(rt);
    free(expected);
    free(prompt_owned);
    return rc;
}
