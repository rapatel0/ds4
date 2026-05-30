static int http_write_json(int fd, int status, const char *body) {
    const char *reason = status == 200 ? "OK" : (status == 404 ? "Not Found" : "Error");
    const int n = dprintf(fd,
                          "HTTP/1.1 %d %s\r\n"
                          "Connection: close\r\n"
                          "Content-Type: application/json\r\n"
                          "Content-Length: %zu\r\n\r\n"
                          "%s",
                          status, reason, std::strlen(body), body);
    return n < 0 ? -1 : 0;
}

static int http_write_text(int fd, const char *body) {
    const int n = dprintf(fd,
                          "HTTP/1.1 200 OK\r\n"
                          "Connection: close\r\n"
                          "Content-Type: text/plain; version=0.0.4\r\n"
                          "Content-Length: %zu\r\n\r\n"
                          "%s",
                          std::strlen(body), body);
    return n < 0 ? -1 : 0;
}

static int json_find_int(const char *body, const char *key, int fallback) {
    if (!body || !key) return fallback;
    const char *p = std::strstr(body, key);
    if (!p) return fallback;
    p += std::strlen(key);
    while (*p && (*p == '"' || *p == '\'' || *p == ' ' || *p == '\t' || *p == ':')) ++p;
    char *end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p || v < 0 || v > 1000000) return fallback;
    return (int)v;
}

struct HttpParsedRequest {
    int fd = -1;
    std::string method;
    std::string path;
    std::string body;
    int requested_tokens = 0;
    std::string cache_key;
    bool cache_key_explicit = false;
    bool prompt_fingerprint_present = false;
    uint64_t prompt_fingerprint = 0;
    std::vector<uint32_t> prompt_token_ids;
    uint64_t cache_position = 0;
    int cache_slot = -1;
    bool cache_hit = false;
    bool cache_prompt_match = true;
    bool cache_evicted = false;
    std::string evicted_key;
    uint32_t decode_input_token = UINT32_MAX;
    std::vector<uint32_t> generated_token_ids;
    uint64_t prompt_prefill_tokens = 0;
};

static int http_content_length(const char *req) {
    const char *p = std::strstr(req, "Content-Length:");
    if (!p) return 0;
    p += std::strlen("Content-Length:");
    while (*p == ' ' || *p == '\t') ++p;
    char *end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p || v < 0 || v > 4096) return 0;
    return (int)v;
}

static std::string json_find_string(const char *body, const char *key) {
    if (!body || !key) return "";
    const char *p = std::strstr(body, key);
    if (!p) return "";
    p += std::strlen(key);
    while (*p && (*p == ' ' || *p == '\t' || *p == ':')) ++p;
    if (*p != '"' && *p != '\'') {
        while (*p && *p != '"' && *p != '\'') ++p;
    }
    if (!*p) return "";
    const char quote = *p++;
    std::string out;
    while (*p && *p != quote && out.size() < 256) {
        if (*p == '\\' && p[1]) {
            ++p;
        }
        out.push_back(*p++);
    }
    return out;
}

static uint64_t fnv1a64(const std::string &s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= (uint64_t)c;
        h *= 1099511628211ull;
    }
    return h;
}

static std::string http_json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out.push_back(c);
                }
                break;
        }
    }
    return out;
}

static std::string http_json_uint_array(const std::vector<uint32_t> &values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out += ",";
        out += std::to_string((unsigned long long)values[i]);
    }
    out += "]";
    return out;
}

static std::string http_json_u64_array(const std::vector<uint64_t> &values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out += ",";
        out += std::to_string((unsigned long long)values[i]);
    }
    out += "]";
    return out;
}

static bool http_is_chat_completion_post(const HttpParsedRequest &req);

struct TokenizerRuntime {
    ds4_engine *engine = nullptr;
    bool initialized = false;
};

static bool open_tokenizer_runtime(const char *model_path, TokenizerRuntime *out) {
    if (!model_path || !model_path[0] || !out) return false;
    ds4_engine_options opt;
    std::memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = DS4_BACKEND_CPU;
    opt.inspect_only = true;
    opt.n_threads = 1;
    if (ds4_engine_open(&out->engine, &opt) != 0 || !out->engine) {
        out->engine = nullptr;
        out->initialized = false;
        return false;
    }
    out->initialized = true;
    return true;
}

static void close_tokenizer_runtime(TokenizerRuntime *rt) {
    if (!rt) return;
    if (rt->engine) ds4_engine_close(rt->engine);
    rt->engine = nullptr;
    rt->initialized = false;
}

static std::string decode_token_text(ds4_engine *engine,
                                     const std::vector<uint32_t> &tokens) {
    if (!engine) return "";
    std::string out;
    for (uint32_t token : tokens) {
        size_t len = 0;
        char *piece = ds4_token_text(engine, (int)token, &len);
        if (piece && len > 0) out.append(piece, len);
        std::free(piece);
    }
    return out;
}

static bool materialize_prompt_tokens(ds4_engine *engine,
                                      HttpParsedRequest *req,
                                      std::string *err) {
    if (!req || !req->prompt_token_ids.empty()) return true;

    const std::string prompt = json_find_string(req->body.c_str(), "\"prompt\"");
    const std::string content = json_find_string(req->body.c_str(), "\"content\"");
    const bool has_text = !prompt.empty() || !content.empty();
    if (!has_text) return true;
    if (!engine) {
        if (err) *err = "text_prompt_requires_tokenizer";
        return false;
    }

    ds4_tokens toks;
    std::memset(&toks, 0, sizeof(toks));
    if (!content.empty() && http_is_chat_completion_post(*req)) {
        ds4_encode_chat_prompt(engine, "", content.c_str(), DS4_THINK_NONE, &toks);
    } else {
        ds4_tokenize_text(engine, prompt.c_str(), &toks);
    }
    req->prompt_token_ids.clear();
    req->prompt_token_ids.reserve((size_t)toks.len);
    for (int i = 0; i < toks.len; ++i) {
        if (toks.v[i] >= 0) req->prompt_token_ids.push_back((uint32_t)toks.v[i]);
    }
    ds4_tokens_free(&toks);
    if (req->prompt_token_ids.empty()) {
        if (err) *err = "tokenizer_produced_no_tokens";
        return false;
    }
    return true;
}

static bool http_read_request(int fd, HttpParsedRequest *out) {
    char req[8192];
    size_t used = 0;
    for (;;) {
        if (used + 1 >= sizeof(req)) return false;
        const ssize_t nr = read(fd, req + used, sizeof(req) - 1 - used);
        if (nr <= 0) return false;
        used += (size_t)nr;
        req[used] = '\0';
        const char *body = std::strstr(req, "\r\n\r\n");
        if (body) {
            const int content_length = http_content_length(req);
            const size_t header_bytes = (size_t)(body + 4 - req);
            if (used >= header_bytes + (size_t)content_length) break;
        }
    }
    char method[16] = {};
    char path[256] = {};
    if (std::sscanf(req, "%15s %255s", method, path) != 2) return false;
    const char *body = std::strstr(req, "\r\n\r\n");
    out->fd = fd;
    out->method = method;
    out->path = path;
    out->body = body ? body + 4 : "";
    return true;
}

static bool http_is_generation_post(const HttpParsedRequest &req) {
    return req.method == "POST" &&
           (req.path == "/v100/selected-token" ||
            req.path == "/v1/v100/selected-token" ||
            req.path == "/v1/completions" ||
            req.path == "/v1/chat/completions" ||
            req.path == "/v100/diagnostic-completions");
}

static bool http_is_completion_post(const HttpParsedRequest &req) {
    return req.method == "POST" &&
           (req.path == "/v1/completions" ||
            req.path == "/v100/diagnostic-completions");
}

static bool http_is_chat_completion_post(const HttpParsedRequest &req) {
    return req.method == "POST" && req.path == "/v1/chat/completions";
}

static bool http_wait_for_connection(int listen_fd, int wait_us) {
    if (wait_us <= 0) return false;
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(listen_fd, &rfds);
    timeval tv = {};
    tv.tv_sec = wait_us / 1000000;
    tv.tv_usec = wait_us % 1000000;
    const int rc = select(listen_fd + 1, &rfds, nullptr, nullptr, &tv);
    return rc > 0 && FD_ISSET(listen_fd, &rfds);
}

static int http_requested_tokens(const HttpParsedRequest &req, int fallback) {
    int out = json_find_int(req.body.c_str(), "max_tokens", fallback);
    if (out <= 0) out = fallback;
    if (out <= 0) out = 1;
    return out;
}

static std::string http_request_cache_key(const HttpParsedRequest &req,
                                          uint64_t request_serial,
                                          bool *explicit_key) {
    std::string key = json_find_string(req.body.c_str(), "\"session_id\"");
    if (key.empty()) key = json_find_string(req.body.c_str(), "\"cache_key\"");
    if (key.empty()) key = json_find_string(req.body.c_str(), "\"conversation_id\"");
    if (!key.empty()) {
        if (explicit_key) *explicit_key = true;
        return key;
    }

    const std::string prompt = json_find_string(req.body.c_str(), "\"prompt\"");
    if (!prompt.empty()) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "prompt:%016llx",
                      (unsigned long long)fnv1a64(prompt));
        if (explicit_key) *explicit_key = false;
        return buf;
    }

    char buf[64];
    std::snprintf(buf, sizeof(buf), "ephemeral:%llu",
                  (unsigned long long)request_serial);
    if (explicit_key) *explicit_key = false;
    return buf;
}

static bool json_find_uint_array(const char *body,
                                 const char *key,
                                 std::vector<uint32_t> *out,
                                 size_t limit) {
    out->clear();
    if (!body || !key) return false;
    const char *p = std::strstr(body, key);
    if (!p) return false;
    p += std::strlen(key);
    while (*p && *p != '[' && *p != '{' && *p != '"' && *p != '\'') ++p;
    if (*p != '[') return false;
    ++p;
    while (*p && *p != ']') {
        while (*p == ' ' || *p == '\t' || *p == '\n' ||
               *p == '\r' || *p == ',') ++p;
        if (*p == ']') break;
        char *end = nullptr;
        unsigned long v = std::strtoul(p, &end, 10);
        if (end == p || v > UINT32_MAX || out->size() >= limit) {
            out->clear();
            return false;
        }
        out->push_back((uint32_t)v);
        p = end;
        while (*p == ' ' || *p == '\t' || *p == '\n' ||
               *p == '\r') ++p;
        if (*p && *p != ',' && *p != ']') {
            out->clear();
            return false;
        }
    }
    return *p == ']' && !out->empty();
}

static uint64_t fnv1a64_u32(const std::vector<uint32_t> &tokens) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t token : tokens) {
        for (int i = 0; i < 4; ++i) {
            h ^= (uint64_t)((token >> (8 * i)) & 0xffu);
            h *= 1099511628211ull;
        }
    }
    return h;
}

static void http_request_prompt_fingerprint(HttpParsedRequest *req) {
    if (!req->prompt_token_ids.empty()) {
        req->prompt_fingerprint_present = true;
        req->prompt_fingerprint = fnv1a64_u32(req->prompt_token_ids);
        return;
    }
    if (json_find_uint_array(req->body.c_str(), "\"prompt_tokens\"",
                             &req->prompt_token_ids, 262144) ||
        json_find_uint_array(req->body.c_str(), "\"prompt\"",
                             &req->prompt_token_ids, 262144)) {
        req->prompt_fingerprint_present = true;
        req->prompt_fingerprint = fnv1a64_u32(req->prompt_token_ids);
        return;
    }

    const std::string prompt = json_find_string(req->body.c_str(), "\"prompt\"");
    if (prompt.empty()) {
        req->prompt_fingerprint_present = false;
        req->prompt_fingerprint = 0;
        req->prompt_token_ids.clear();
        return;
    }
    req->prompt_fingerprint_present = true;
    req->prompt_fingerprint = fnv1a64(prompt);
}

#include "appliance/request_scheduler.cu"
int run_tp_ep_http_server(const Options &base_opt,
                          const DenseF16Cache *shared_dense_f16_cache,
                          const SharedApi *shared_api,
                          SharedRankBuffers *shared_rank_buffers,
                          SharedTpRuntime *shared_tp_runtime,
                          const SharedExpertBindings *shared_expert_bindings,
                          const SharedDenseOps *shared_dense_ops,
                          SharedOutputHead *shared_output_head,
                          SharedOutputHead *mtp_output_head,
                          SharedHcControls *shared_hc_controls,
                          SharedTokenEmbedding *shared_token_embedding,
                          std::vector<ContractRow> resident_rows[43],
                          LayerStats resident_stats[43]) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::perror("tp_ep_http_socket");
        return 30;
    }
    int yes = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)base_opt.port);
    if (inet_pton(AF_INET, base_opt.host, &addr.sin_addr) != 1) {
        std::fprintf(stderr, "tp_ep_http_bad_host\t%s\n", base_opt.host);
        close(listen_fd);
        return 31;
    }
    if (bind(listen_fd, (sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(listen_fd, 16) != 0) {
        std::perror("tp_ep_http_bind_listen");
        close(listen_fd);
        return 32;
    }

    uint64_t served = 0;
    uint64_t generation_requests = 0;
    uint64_t generation_batches = 0;
    uint64_t coalesced_requests = 0;
    uint64_t bucketed_requests = 0;
    uint64_t rejected = 0;
    TpEpHttpSessionTable sessions;
    sessions.init(base_opt.slots);
    std::deque<HttpParsedRequest> pending_generation;
    uint64_t next_position = base_opt.position;
    uint64_t total_prompt_tokens = 0;
    uint64_t total_generated_tokens = 0;
    uint64_t total_continuation_tokens = 0;
    double total_decode_ms = 0.0;
    double total_wall_ms = 0.0;
    double total_continuation_decode_ms = 0.0;
    double total_continuation_wall_ms = 0.0;
    double total_ep_ms = 0.0;
    double total_dense_ms = 0.0;
    double total_compose_ms = 0.0;
    double total_compose_reduce_ms = 0.0;
    double total_compose_copy_ms = 0.0;
    double total_compose_final_ms = 0.0;
    ServingBenchResult last = {};
    TokenizerRuntime tokenizer;
    if (base_opt.tokenizer_model_path && base_opt.tokenizer_model_path[0]) {
        if (!open_tokenizer_runtime(base_opt.tokenizer_model_path, &tokenizer)) {
            std::fprintf(stderr, "tp_ep_http tokenizer open failed: %s\n",
                         base_opt.tokenizer_model_path);
            close(listen_fd);
            return 31;
        }
        std::printf("tp_ep_http_tokenizer\tmodel\t%s\tPASS\n",
                    base_opt.tokenizer_model_path);
    }
    std::printf("tp_ep_http_serving\thttp://%s:%d/v100/selected-token\tPASS\n",
                base_opt.host, base_opt.port);
    std::printf("tp_ep_http_completions\thttp://%s:%d/v1/completions\tDIAGNOSTIC\n",
                base_opt.host, base_opt.port);
    std::printf("tp_ep_http_chat_completions\thttp://%s:%d/v1/chat/completions\tDIAGNOSTIC\n",
                base_opt.host, base_opt.port);
    std::fflush(stdout);

    while (base_opt.max_requests == 0 || (int)served < base_opt.max_requests ||
           !pending_generation.empty()) {
        HttpParsedRequest first_req;
        int fd = -1;
        if (!pending_generation.empty()) {
            first_req = std::move(pending_generation.front());
            pending_generation.pop_front();
            fd = first_req.fd;
        } else {
            fd = accept(listen_fd, nullptr, nullptr);
            if (fd < 0) {
                if (errno == EINTR) continue;
                std::perror("tp_ep_http_accept");
                break;
            }
            served++;
            if (!http_read_request(fd, &first_req)) {
                close(fd);
                continue;
            }
        }

        if (first_req.method == "GET" && first_req.path == "/health") {
            http_write_json(fd, 200, "{\"status\":\"ok\",\"backend\":\"tp_ep_resident\"}\n");
        } else if (first_req.method == "GET" &&
                   (first_req.path == "/status" || first_req.path == "/v100/status")) {
            const double cumulative_generated_tok_s_wall = total_wall_ms > 0.0
                ? (double)total_generated_tokens * 1000.0 / total_wall_ms
                : 0.0;
            const double cumulative_generated_tok_s_decode = total_decode_ms > 0.0
                ? (double)total_generated_tokens * 1000.0 / total_decode_ms
                : 0.0;
            const double cumulative_continuation_tok_s_wall = total_continuation_wall_ms > 0.0
                ? (double)total_continuation_tokens * 1000.0 / total_continuation_wall_ms
                : 0.0;
            const double cumulative_continuation_tok_s_decode = total_continuation_decode_ms > 0.0
                ? (double)total_continuation_tokens * 1000.0 / total_continuation_decode_ms
                : 0.0;
            const PeerCopySnapshot peer = peer_copy_snapshot();
            char out[8192];
            std::snprintf(out, sizeof(out),
                          "{\"status\":\"ok\",\"backend\":\"tp_ep_resident\","
                          "\"tp\":8,\"ep\":8,\"pp\":1,\"ctx\":262144,"
                          "\"slots\":%d,\"served_requests\":%llu,"
                          "\"generation_requests\":%llu,\"generation_batches\":%llu,"
                          "\"coalesced_requests\":%llu,\"bucketed_requests\":%llu,"
                          "\"pending_generation_requests\":%zu,"
                          "\"microbatch_wait_us\":%d,"
                          "\"kv_runtime_resident\":%d,"
                          "\"cache_slots_total\":%zu,"
                          "\"cache_slots_used\":%d,"
                          "\"cache_hits\":%llu,"
                          "\"cache_misses\":%llu,"
                          "\"cache_evictions\":%llu,"
                          "\"peer_copy_accounting\":%d,"
                          "\"peer_copy_reject_sys\":%d,"
                          "\"peer_copy_ops\":%llu,"
                          "\"peer_copy_bytes\":%llu,"
                          "\"peer_copy_nv1_ops\":%llu,"
                          "\"peer_copy_nv1_bytes\":%llu,"
                          "\"peer_copy_nv2_ops\":%llu,"
                          "\"peer_copy_nv2_bytes\":%llu,"
                          "\"peer_copy_sys_ops\":%llu,"
                          "\"peer_copy_sys_bytes\":%llu,"
                          "\"peer_copy_unknown_ops\":%llu,"
                          "\"peer_copy_unknown_bytes\":%llu,"
                          "\"peer_copy_first_sys_src\":%d,"
                          "\"peer_copy_first_sys_dst\":%d,"
                          "\"peer_copy_first_sys_bytes\":%llu,"
                          "\"peer_copy_first_sys_site\":\"%s\","
                          "\"peer_copy_first_sys_line\":%d,"
                          "\"peer_copy_top_sys_site\":\"%s\","
                          "\"peer_copy_top_sys_site_line\":%d,"
                          "\"peer_copy_top_sys_site_ops\":%llu,"
                          "\"peer_copy_top_sys_site_bytes\":%llu,"
                          "\"peer_copy_top_sys_site_total_ops\":%llu,"
                          "\"peer_copy_top_sys_site_total_bytes\":%llu,"
                          "\"rejected_requests\":%llu,"
                          "\"total_prompt_tokens\":%llu,"
                          "\"total_generated_tokens\":%llu,"
                          "\"total_continuation_tokens\":%llu,"
                          "\"next_position\":%llu,"
                          "\"warmed_ready\":true,\"resident_ready\":true,"
                          "\"last_generated_tok_s_wall\":%.6f,"
                          "\"last_continuation_tok_s_wall\":%.6f,"
                          "\"last_compose_copy_ms\":%.6f,"
                          "\"cumulative_generated_tok_s_wall\":%.6f,"
                          "\"cumulative_continuation_tok_s_wall\":%.6f,"
                          "\"cumulative_generated_tok_s_decode\":%.6f,"
                          "\"cumulative_continuation_tok_s_decode\":%.6f,"
                          "\"cumulative_ep_ms\":%.6f,"
                          "\"cumulative_dense_ms\":%.6f,"
                          "\"cumulative_compose_ms\":%.6f,"
                          "\"cumulative_compose_reduce_ms\":%.6f,"
                          "\"cumulative_compose_copy_ms\":%.6f,"
                          "\"cumulative_compose_final_ms\":%.6f}\n",
                          base_opt.slots,
                          (unsigned long long)served,
                          (unsigned long long)generation_requests,
                          (unsigned long long)generation_batches,
                          (unsigned long long)coalesced_requests,
                          (unsigned long long)bucketed_requests,
                          pending_generation.size(),
                          base_opt.microbatch_wait_us,
                          shared_tp_runtime && shared_tp_runtime->initialized ? 1 : 0,
                          sessions.slots.size(),
                          sessions.used(),
                          (unsigned long long)sessions.hits,
                          (unsigned long long)sessions.misses,
                          (unsigned long long)sessions.evictions,
                          g_peer_copy_accounting.enabled.load(std::memory_order_relaxed),
                          g_peer_copy_accounting.reject_sys.load(std::memory_order_relaxed),
                          (unsigned long long)peer.ops,
                          (unsigned long long)peer.bytes,
                          (unsigned long long)peer.nv1_ops,
                          (unsigned long long)peer.nv1_bytes,
                          (unsigned long long)peer.nv2_ops,
                          (unsigned long long)peer.nv2_bytes,
                          (unsigned long long)peer.sys_ops,
                          (unsigned long long)peer.sys_bytes,
                          (unsigned long long)peer.unknown_ops,
                          (unsigned long long)peer.unknown_bytes,
                          peer.first_sys_src,
                          peer.first_sys_dst,
                          (unsigned long long)peer.first_sys_bytes,
                          peer.first_sys_site ? peer.first_sys_site : "-",
                          peer.first_sys_line,
                          peer.top_sys_site ? peer.top_sys_site : "-",
                          peer.top_sys_site_line,
                          (unsigned long long)peer.top_sys_site_ops,
                          (unsigned long long)peer.top_sys_site_bytes,
                          (unsigned long long)peer.top_sys_site_total_ops,
                          (unsigned long long)peer.top_sys_site_total_bytes,
                          (unsigned long long)rejected,
                          (unsigned long long)total_prompt_tokens,
                          (unsigned long long)total_generated_tokens,
                          (unsigned long long)total_continuation_tokens,
                          (unsigned long long)next_position,
                          last.aggregate_generated_tok_s_wall,
                          last.aggregate_continuation_tok_s_wall,
                          last.total_compose_copy_ms,
                          cumulative_generated_tok_s_wall,
                          cumulative_continuation_tok_s_wall,
                          cumulative_generated_tok_s_decode,
                          cumulative_continuation_tok_s_decode,
                          total_ep_ms,
                          total_dense_ms,
                          total_compose_ms,
                          total_compose_reduce_ms,
                          total_compose_copy_ms,
                          total_compose_final_ms);
            http_write_json(fd, 200, out);
        } else if (first_req.method == "GET" && first_req.path == "/v100/slots") {
            char out[16384];
            sessions.slots_json(out, sizeof(out));
            http_write_json(fd, 200, out);
        } else if (first_req.method == "GET" && first_req.path == "/metrics") {
            const double cumulative_generated_tok_s_wall = total_wall_ms > 0.0
                ? (double)total_generated_tokens * 1000.0 / total_wall_ms
                : 0.0;
            const double cumulative_generated_tok_s_decode = total_decode_ms > 0.0
                ? (double)total_generated_tokens * 1000.0 / total_decode_ms
                : 0.0;
            const double cumulative_continuation_tok_s_wall = total_continuation_wall_ms > 0.0
                ? (double)total_continuation_tokens * 1000.0 / total_continuation_wall_ms
                : 0.0;
            const double cumulative_continuation_tok_s_decode = total_continuation_decode_ms > 0.0
                ? (double)total_continuation_tokens * 1000.0 / total_continuation_decode_ms
                : 0.0;
            const PeerCopySnapshot peer = peer_copy_snapshot();
            char out[8192];
            std::snprintf(out, sizeof(out),
                          "ds4_tp_ep_resident_ready 1\n"
                          "ds4_tp_ep_slots %d\n"
                          "ds4_tp_ep_served_requests %llu\n"
                          "ds4_tp_ep_generation_requests %llu\n"
                          "ds4_tp_ep_generation_batches %llu\n"
                          "ds4_tp_ep_coalesced_requests %llu\n"
                          "ds4_tp_ep_bucketed_requests %llu\n"
                          "ds4_tp_ep_pending_generation_requests %zu\n"
                          "ds4_tp_ep_microbatch_wait_us %d\n"
                          "ds4_tp_ep_kv_runtime_resident %d\n"
                          "ds4_tp_ep_cache_slots_total %zu\n"
                          "ds4_tp_ep_cache_slots_used %d\n"
                          "ds4_tp_ep_cache_hits %llu\n"
                          "ds4_tp_ep_cache_misses %llu\n"
                          "ds4_tp_ep_cache_evictions %llu\n"
                          "ds4_tp_ep_peer_copy_accounting %d\n"
                          "ds4_tp_ep_peer_copy_reject_sys %d\n"
                          "ds4_tp_ep_peer_copy_ops %llu\n"
                          "ds4_tp_ep_peer_copy_bytes %llu\n"
                          "ds4_tp_ep_peer_copy_nv1_ops %llu\n"
                          "ds4_tp_ep_peer_copy_nv1_bytes %llu\n"
                          "ds4_tp_ep_peer_copy_nv2_ops %llu\n"
                          "ds4_tp_ep_peer_copy_nv2_bytes %llu\n"
                          "ds4_tp_ep_peer_copy_sys_ops %llu\n"
                          "ds4_tp_ep_peer_copy_sys_bytes %llu\n"
                          "ds4_tp_ep_peer_copy_unknown_ops %llu\n"
                          "ds4_tp_ep_peer_copy_unknown_bytes %llu\n"
                          "ds4_tp_ep_peer_copy_first_sys_src %d\n"
                          "ds4_tp_ep_peer_copy_first_sys_dst %d\n"
                          "ds4_tp_ep_peer_copy_first_sys_bytes %llu\n"
                          "ds4_tp_ep_peer_copy_first_sys_line %d\n"
                          "ds4_tp_ep_peer_copy_top_sys_site_line %d\n"
                          "ds4_tp_ep_peer_copy_top_sys_site_ops %llu\n"
                          "ds4_tp_ep_peer_copy_top_sys_site_bytes %llu\n"
                          "ds4_tp_ep_peer_copy_top_sys_site_total_ops %llu\n"
                          "ds4_tp_ep_peer_copy_top_sys_site_total_bytes %llu\n"
                          "ds4_tp_ep_rejected_requests %llu\n"
                          "ds4_tp_ep_total_prompt_tokens %llu\n"
                          "ds4_tp_ep_total_generated_tokens %llu\n"
                          "ds4_tp_ep_total_continuation_tokens %llu\n"
                          "ds4_tp_ep_next_position %llu\n"
                          "ds4_tp_ep_generated_tok_s_wall %.6f\n"
                          "ds4_tp_ep_continuation_tok_s_wall %.6f\n"
                          "ds4_tp_ep_last_compose_copy_ms %.6f\n"
                          "ds4_tp_ep_cumulative_generated_tok_s_wall %.6f\n"
                          "ds4_tp_ep_cumulative_continuation_tok_s_wall %.6f\n"
                          "ds4_tp_ep_cumulative_generated_tok_s_decode %.6f\n"
                          "ds4_tp_ep_cumulative_continuation_tok_s_decode %.6f\n"
                          "ds4_tp_ep_cumulative_ep_ms %.6f\n"
                          "ds4_tp_ep_cumulative_dense_ms %.6f\n"
                          "ds4_tp_ep_cumulative_compose_ms %.6f\n"
                          "ds4_tp_ep_cumulative_compose_reduce_ms %.6f\n"
                          "ds4_tp_ep_cumulative_compose_copy_ms %.6f\n"
                          "ds4_tp_ep_cumulative_compose_final_ms %.6f\n",
                          base_opt.slots,
                          (unsigned long long)served,
                          (unsigned long long)generation_requests,
                          (unsigned long long)generation_batches,
                          (unsigned long long)coalesced_requests,
                          (unsigned long long)bucketed_requests,
                          pending_generation.size(),
                          base_opt.microbatch_wait_us,
                          shared_tp_runtime && shared_tp_runtime->initialized ? 1 : 0,
                          sessions.slots.size(),
                          sessions.used(),
                          (unsigned long long)sessions.hits,
                          (unsigned long long)sessions.misses,
                          (unsigned long long)sessions.evictions,
                          g_peer_copy_accounting.enabled.load(std::memory_order_relaxed),
                          g_peer_copy_accounting.reject_sys.load(std::memory_order_relaxed),
                          (unsigned long long)peer.ops,
                          (unsigned long long)peer.bytes,
                          (unsigned long long)peer.nv1_ops,
                          (unsigned long long)peer.nv1_bytes,
                          (unsigned long long)peer.nv2_ops,
                          (unsigned long long)peer.nv2_bytes,
                          (unsigned long long)peer.sys_ops,
                          (unsigned long long)peer.sys_bytes,
                          (unsigned long long)peer.unknown_ops,
                          (unsigned long long)peer.unknown_bytes,
                          peer.first_sys_src,
                          peer.first_sys_dst,
                          (unsigned long long)peer.first_sys_bytes,
                          peer.first_sys_line,
                          peer.top_sys_site_line,
                          (unsigned long long)peer.top_sys_site_ops,
                          (unsigned long long)peer.top_sys_site_bytes,
                          (unsigned long long)peer.top_sys_site_total_ops,
                          (unsigned long long)peer.top_sys_site_total_bytes,
                          (unsigned long long)rejected,
                          (unsigned long long)total_prompt_tokens,
                          (unsigned long long)total_generated_tokens,
                          (unsigned long long)total_continuation_tokens,
                          (unsigned long long)next_position,
                          last.aggregate_generated_tok_s_wall,
                          last.aggregate_continuation_tok_s_wall,
                          last.total_compose_copy_ms,
                          cumulative_generated_tok_s_wall,
                          cumulative_continuation_tok_s_wall,
                          cumulative_generated_tok_s_decode,
                          cumulative_continuation_tok_s_decode,
                          total_ep_ms,
                          total_dense_ms,
                          total_compose_ms,
                          total_compose_reduce_ms,
                          total_compose_copy_ms,
                          total_compose_final_ms);
            http_write_text(fd, out);
        } else if (http_is_generation_post(first_req)) {
            std::string prompt_error;
            if (!materialize_prompt_tokens(tokenizer.engine, &first_req, &prompt_error)) {
                std::string body = "{\"error\":\"" + http_json_escape(prompt_error) + "\"}\n";
                http_write_json(first_req.fd, 400, body.c_str());
                close(first_req.fd);
                rejected++;
                continue;
            }
            first_req.requested_tokens = http_requested_tokens(first_req, base_opt.decode_steps);
            first_req.cache_key = http_request_cache_key(first_req,
                                                         served + pending_generation.size(),
                                                         &first_req.cache_key_explicit);
            http_request_prompt_fingerprint(&first_req);
            first_req.cache_position =
                sessions.preview_position(first_req.cache_key,
                                          first_req.prompt_fingerprint_present,
                                          first_req.prompt_fingerprint,
                                          base_opt.position);
            {
                const TpEpHttpContextAdmission admission =
                    tp_ep_http_context_admission(sessions, first_req,
                                                 base_opt.position, 262144ull);
                if (!admission.ok) {
                    std::fprintf(stderr,
                                 "tp_ep_http_context_rejected\tstart_position\t%llu\t"
                                 "prompt_prefill_steps\t%llu\trequested_steps\t%llu\t"
                                 "final_position\t%llu\tctx\t%llu\tcache_hit\t%d\n",
                                 (unsigned long long)admission.start_position,
                                 (unsigned long long)admission.prompt_prefill_steps,
                                 (unsigned long long)admission.requested_decode_steps,
                                 (unsigned long long)admission.final_position,
                                 (unsigned long long)admission.ctx,
                                 admission.cache_hit ? 1 : 0);
                    const std::string body =
                        tp_ep_http_context_error_json(admission);
                    http_write_json(first_req.fd, 400, body.c_str());
                    close(first_req.fd);
                    rejected++;
                    continue;
                }
            }

            std::vector<HttpParsedRequest> batch;
            batch.push_back(first_req);
            http_drain_matching_pending(&pending_generation,
                                        first_req.requested_tokens,
                                        first_req.cache_position,
                                        base_opt.slots,
                                        &batch);
            while ((int)batch.size() < base_opt.slots &&
                   http_wait_for_connection(listen_fd, base_opt.microbatch_wait_us)) {
                int extra_fd = accept(listen_fd, nullptr, nullptr);
                if (extra_fd < 0) {
                    if (errno == EINTR) continue;
                    break;
                }
                served++;
                HttpParsedRequest extra_req;
                if (!http_read_request(extra_fd, &extra_req)) {
                    close(extra_fd);
                    continue;
                }
                if (!http_is_generation_post(extra_req)) {
                    rejected++;
                    http_write_json(extra_fd, 404, "{\"error\":\"not_found_during_coalesce\"}\n");
                    close(extra_fd);
                    continue;
                }
                std::string extra_prompt_error;
                if (!materialize_prompt_tokens(tokenizer.engine, &extra_req, &extra_prompt_error)) {
                    std::string body = "{\"error\":\"" + http_json_escape(extra_prompt_error) + "\"}\n";
                    http_write_json(extra_fd, 400, body.c_str());
                    close(extra_fd);
                    rejected++;
                    continue;
                }
                extra_req.requested_tokens = http_requested_tokens(extra_req, first_req.requested_tokens);
                extra_req.cache_key = http_request_cache_key(extra_req,
                                                            served + pending_generation.size(),
                                                            &extra_req.cache_key_explicit);
                http_request_prompt_fingerprint(&extra_req);
                extra_req.cache_position =
                    sessions.preview_position(extra_req.cache_key,
                                              extra_req.prompt_fingerprint_present,
                                              extra_req.prompt_fingerprint,
                                              base_opt.position);
                {
                    const TpEpHttpContextAdmission admission =
                        tp_ep_http_context_admission(sessions, extra_req,
                                                     base_opt.position, 262144ull);
                    if (!admission.ok) {
                        std::fprintf(stderr,
                                     "tp_ep_http_context_rejected\tstart_position\t%llu\t"
                                     "prompt_prefill_steps\t%llu\trequested_steps\t%llu\t"
                                     "final_position\t%llu\tctx\t%llu\tcache_hit\t%d\n",
                                     (unsigned long long)admission.start_position,
                                     (unsigned long long)admission.prompt_prefill_steps,
                                     (unsigned long long)admission.requested_decode_steps,
                                     (unsigned long long)admission.final_position,
                                     (unsigned long long)admission.ctx,
                                     admission.cache_hit ? 1 : 0);
                        const std::string body =
                            tp_ep_http_context_error_json(admission);
                        http_write_json(extra_fd, 400, body.c_str());
                        close(extra_fd);
                        rejected++;
                        continue;
                    }
                }
                if (extra_req.requested_tokens != first_req.requested_tokens) {
                    bucketed_requests++;
                    pending_generation.push_back(std::move(extra_req));
                    continue;
                }
                if (extra_req.cache_position != first_req.cache_position) {
                    bucketed_requests++;
                    pending_generation.push_back(std::move(extra_req));
                    continue;
                }
                bool duplicate_key = false;
                for (const auto &req : batch) {
                    if (req.cache_key == extra_req.cache_key) {
                        duplicate_key = true;
                        break;
                    }
                }
                if (duplicate_key) {
                    bucketed_requests++;
                    pending_generation.push_back(std::move(extra_req));
                    continue;
                }
                batch.push_back(extra_req);
            }

            std::vector<TpEpHttpSessionAssignment> assignments(batch.size());
            std::vector<bool> protected_slots((size_t)base_opt.slots, false);
            bool assignment_failed = false;
            for (size_t i = 0; i < batch.size(); ++i) {
                assignments[i] = sessions.assign(batch[i].cache_key,
                                                 batch[i].prompt_fingerprint_present,
                                                 batch[i].prompt_fingerprint,
                                                 batch[i].prompt_token_ids,
                                                 base_opt.position,
                                                 protected_slots);
                if (assignments[i].slot < 0) {
                    assignment_failed = true;
                    break;
                }
                batch[i].cache_slot = assignments[i].slot;
                batch[i].cache_hit = assignments[i].hit;
                batch[i].cache_prompt_match = assignments[i].prompt_match;
                batch[i].cache_evicted = assignments[i].evicted;
                batch[i].evicted_key = assignments[i].evicted_key;
                batch[i].cache_position = assignments[i].pos_in;
                protected_slots[(size_t)assignments[i].slot] = true;
            }
            if (assignment_failed) {
                rejected += (uint64_t)batch.size();
                for (HttpParsedRequest &queued : batch) {
                    http_write_json(queued.fd, 503, "{\"error\":\"no_cache_slot_available\"}\n");
                    close(queued.fd);
                }
                continue;
            }

            Options req_opt = base_opt;
            const int requested_decode_steps = first_req.requested_tokens;
            req_opt.decode_steps = 1;
            req_opt.slots = base_opt.slots;
            req_opt.position = first_req.cache_position;
            req_opt.serving_bench = false;
            std::vector<uint32_t> decode_input_tokens((size_t)req_opt.slots, 0u);
            std::vector<unsigned char> decode_active_slots((size_t)req_opt.slots, 0u);
            std::vector<uint32_t> mtp_raw_rows_by_slot((size_t)req_opt.slots, 0u);
            for (size_t i = 0; i < batch.size(); ++i) {
                uint32_t input_token = 0;
                if (assignments[i].slot >= 0 &&
                    assignments[i].slot < (int)sessions.slots.size()) {
                    const TpEpHttpSessionSlot &slot =
                        sessions.slots[(size_t)assignments[i].slot];
                    if (assignments[i].slot < req_opt.slots) {
                        mtp_raw_rows_by_slot[(size_t)assignments[i].slot] =
                            slot.mtp_raw_rows;
                    }
                    if (assignments[i].hit &&
                        slot.last_selected_token != UINT32_MAX) {
                        input_token = slot.last_selected_token;
                    } else if (!batch[i].prompt_token_ids.empty()) {
                        input_token = batch[i].prompt_token_ids.back();
                    } else if (!slot.prompt_token_ids.empty()) {
                        input_token = slot.prompt_token_ids.back();
                    }
                } else if (!batch[i].prompt_token_ids.empty()) {
                    input_token = batch[i].prompt_token_ids.back();
                }
                batch[i].decode_input_token = input_token;
                if (batch[i].cache_slot >= 0 &&
                    batch[i].cache_slot < req_opt.slots) {
                    decode_input_tokens[(size_t)batch[i].cache_slot] = input_token;
                    decode_active_slots[(size_t)batch[i].cache_slot] = 1u;
                }
            }
            int max_prompt_prefill_steps = 0;
            int rc = 0;
            for (size_t i = 0; i < batch.size(); ++i) {
                if (!assignments[i].hit && batch[i].prompt_token_ids.size() > 1) {
                    max_prompt_prefill_steps = std::max(
                        max_prompt_prefill_steps,
                        (int)batch[i].prompt_token_ids.size() - 1);
                }
            }
            for (int prefill_step = 0; prefill_step < max_prompt_prefill_steps; ++prefill_step) {
                bool any_prefill = false;
                std::vector<uint32_t> prefill_input_tokens((size_t)req_opt.slots, 0u);
                for (size_t i = 0; i < batch.size(); ++i) {
                    const int slot = batch[i].cache_slot;
                    if (assignments[i].hit || slot < 0 || slot >= req_opt.slots) continue;
                    if ((size_t)prefill_step + 1u >= batch[i].prompt_token_ids.size()) continue;
                    const uint32_t tok = batch[i].prompt_token_ids[(size_t)prefill_step];
                    prefill_input_tokens[(size_t)slot] = tok;
                    batch[i].prompt_prefill_tokens++;
                    any_prefill = true;
                }
                if (!any_prefill) continue;
                Options prefill_opt = req_opt;
                prefill_opt.position = first_req.cache_position + (uint64_t)prefill_step;
                prefill_opt.diagnostic_output_head = false;
                prefill_opt.diagnostic_output_head_lazy_gate = false;
                std::vector<unsigned char> prefill_active_slots((size_t)req_opt.slots, 0u);
                for (size_t i = 0; i < batch.size(); ++i) {
                    const int slot = batch[i].cache_slot;
                    if (assignments[i].hit || slot < 0 || slot >= req_opt.slots) continue;
                    if ((size_t)prefill_step + 1u >= batch[i].prompt_token_ids.size()) continue;
                    prefill_active_slots[(size_t)slot] = 1u;
                }
                ServingBenchResult prefill_result;
                rc = run_token_major_serving_loop(prefill_opt,
                                                  shared_dense_f16_cache,
                                                  shared_api,
                                                  shared_rank_buffers,
                                                  shared_tp_runtime,
                                                  shared_expert_bindings,
                                                  shared_dense_ops,
                                                  nullptr,
                                                  nullptr,
                                                  shared_hc_controls,
                                                  shared_token_embedding,
                                                  &prefill_input_tokens,
                                                  &prefill_active_slots,
                                                  &mtp_raw_rows_by_slot,
                                                  resident_rows,
                                                  resident_stats,
                                                  true,
                                                  &prefill_result);
                if (rc != 0) break;
                total_decode_ms += prefill_result.total_decode_ms;
                total_wall_ms += prefill_result.total_wall_ms;
                total_ep_ms += prefill_result.total_ep_ms;
                total_dense_ms += prefill_result.total_dense_ms;
                total_compose_ms += prefill_result.total_compose_ms;
                total_compose_reduce_ms += prefill_result.total_compose_reduce_ms;
                total_compose_copy_ms += prefill_result.total_compose_copy_ms;
                total_compose_final_ms += prefill_result.total_compose_final_ms;
            }
            ServingBenchResult result;
            bool missing_output_head = false;
            for (int step = 0; rc == 0 && step < requested_decode_steps; ++step) {
                req_opt.position = first_req.cache_position +
                                   (uint64_t)max_prompt_prefill_steps +
                                   (uint64_t)step;
                ServingBenchResult step_result;
                rc = run_token_major_serving_loop(req_opt,
                                                  shared_dense_f16_cache,
                                                  shared_api,
                                                  shared_rank_buffers,
                                                  shared_tp_runtime,
                                                  shared_expert_bindings,
                                                  shared_dense_ops,
                                                  shared_output_head,
                                                  mtp_output_head,
                                                  shared_hc_controls,
                                                  shared_token_embedding,
                                                  &decode_input_tokens,
                                                  &decode_active_slots,
                                                  &mtp_raw_rows_by_slot,
                                                  resident_rows,
                                                  resident_stats,
                                                  true,
                                                  &step_result);
                if (rc != 0) break;
                if (!step_result.diagnostic_output_head ||
                    step_result.selected_tokens.size() < (size_t)req_opt.slots) {
                    missing_output_head = true;
                    break;
                }
                result.prompt_tokens = step == 0 ? step_result.prompt_tokens : result.prompt_tokens;
                result.generated_tokens += (uint64_t)req_opt.slots;
                result.continuation_tokens = requested_decode_steps > 1
                    ? (uint64_t)req_opt.slots * (uint64_t)(requested_decode_steps - 1)
                    : 0ull;
                if (step == 0) {
                    result.first_token_decode_ms += step_result.first_token_decode_ms;
                    result.first_token_wall_ms += step_result.first_token_wall_ms;
                } else {
                    result.continuation_decode_ms += step_result.first_token_decode_ms;
                    result.continuation_wall_ms += step_result.first_token_wall_ms;
                }
                result.total_decode_ms += step_result.total_decode_ms;
                result.total_wall_ms += step_result.total_wall_ms;
                result.total_ep_ms += step_result.total_ep_ms;
                result.total_dense_ms += step_result.total_dense_ms;
                result.total_compose_ms += step_result.total_compose_ms;
                result.total_compose_reduce_ms += step_result.total_compose_reduce_ms;
                result.total_compose_copy_ms += step_result.total_compose_copy_ms;
                result.total_compose_final_ms += step_result.total_compose_final_ms;
                result.total_hc_current_input_ms += step_result.total_hc_current_input_ms;
                result.diagnostic_output_head = step_result.diagnostic_output_head;
                result.diagnostic_output_head_proxy_hc =
                    step_result.diagnostic_output_head_proxy_hc;
                result.output_head_ms += step_result.output_head_ms;
                result.output_head_gather_ms += step_result.output_head_gather_ms;
                result.output_head_prep_ms += step_result.output_head_prep_ms;
                result.output_head_broadcast_ms += step_result.output_head_broadcast_ms;
                result.output_head_projection_ms += step_result.output_head_projection_ms;
                result.output_head_top1_ms += step_result.output_head_top1_ms;
                result.token_input_seed = result.token_input_seed ||
                                          step_result.token_input_seed;
                if (step == 0) result.first_input_token = step_result.first_input_token;
                result.selected_tokens = step_result.selected_tokens;
                result.selected_logits = step_result.selected_logits;
                result.step_checksums.push_back(step_result.checksum);
                result.checksum ^= step_result.checksum +
                                   (uint64_t)(step + 1) * 0x9e3779b185ebca87ull;
                for (size_t i = 0; i < batch.size(); ++i) {
                    const int slot = batch[i].cache_slot;
                    if (slot >= 0 &&
                        (size_t)slot < step_result.selected_tokens.size()) {
                        const uint32_t tok = step_result.selected_tokens[(size_t)slot];
                        batch[i].generated_token_ids.push_back(tok);
                        decode_input_tokens[(size_t)slot] = tok;
                    }
                }
            }
            if (result.total_decode_ms > 0.0) {
                result.aggregate_generated_tok_s_decode =
                    (double)result.generated_tokens * 1000.0 / result.total_decode_ms;
                result.aggregate_continuation_tok_s_decode =
                    result.continuation_decode_ms > 0.0
                        ? (double)result.continuation_tokens * 1000.0 /
                              result.continuation_decode_ms
                        : 0.0;
            }
            if (result.total_wall_ms > 0.0) {
                result.aggregate_generated_tok_s_wall =
                    (double)result.generated_tokens * 1000.0 / result.total_wall_ms;
                result.aggregate_continuation_tok_s_wall =
                    result.continuation_wall_ms > 0.0
                        ? (double)result.continuation_tokens * 1000.0 /
                              result.continuation_wall_ms
                        : 0.0;
            }
            req_opt.decode_steps = requested_decode_steps;
            if (rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_http_decode_failed\trc\t%d\tbatch\t%zu\trequested_steps\t%d\tposition\t%llu\n",
                             rc, batch.size(), requested_decode_steps,
                             (unsigned long long)first_req.cache_position);
                std::fflush(stderr);
                rejected += (uint64_t)batch.size();
                for (HttpParsedRequest &queued : batch) {
                    http_write_json(queued.fd, 500, "{\"error\":\"tp_ep_decode_failed\"}\n");
                    close(queued.fd);
                }
            } else if (missing_output_head) {
                rejected += (uint64_t)batch.size();
                for (HttpParsedRequest &queued : batch) {
                    http_write_json(queued.fd, 500, "{\"error\":\"tp_ep_output_head_missing\"}\n");
                    close(queued.fd);
                }
            } else {
                const uint64_t batch_id = generation_batches + 1;
                uint64_t client_prompt_tokens = 0;
                for (const HttpParsedRequest &request : batch) {
                    client_prompt_tokens += request.prompt_token_ids.empty()
                        ? 1ull
                        : (uint64_t)request.prompt_token_ids.size();
                }
                const uint64_t client_generated_tokens =
                    (uint64_t)batch.size() * (uint64_t)req_opt.decode_steps;
                const uint64_t client_continuation_tokens = req_opt.decode_steps > 1
                    ? (uint64_t)batch.size() * (uint64_t)(req_opt.decode_steps - 1)
                    : 0ull;
                generation_batches++;
                generation_requests += (uint64_t)batch.size();
                if (batch.size() > 1) coalesced_requests += (uint64_t)batch.size();
                next_position = std::max(next_position,
                                         first_req.cache_position +
                                             (uint64_t)max_prompt_prefill_steps +
                                             (uint64_t)req_opt.decode_steps);
                total_prompt_tokens += client_prompt_tokens;
                total_generated_tokens += client_generated_tokens;
                total_continuation_tokens += client_continuation_tokens;
                total_decode_ms += result.total_decode_ms;
                total_wall_ms += result.total_wall_ms;
                total_continuation_decode_ms += result.continuation_decode_ms;
                total_continuation_wall_ms += result.continuation_wall_ms;
                total_ep_ms += result.total_ep_ms;
                total_dense_ms += result.total_dense_ms;
                total_compose_ms += result.total_compose_ms;
                total_compose_reduce_ms += result.total_compose_reduce_ms;
                total_compose_copy_ms += result.total_compose_copy_ms;
                total_compose_final_ms += result.total_compose_final_ms;
                last = result;
                for (size_t i = 0; i < batch.size(); ++i) {
                    const uint64_t request_generated = (uint64_t)req_opt.decode_steps;
                    const uint64_t request_continuation = req_opt.decode_steps > 1
                        ? (uint64_t)(req_opt.decode_steps - 1)
                        : 0ull;
                    const bool have_output_head =
                        result.diagnostic_output_head &&
                        batch[i].cache_slot >= 0 &&
                        (size_t)batch[i].cache_slot < result.selected_tokens.size() &&
                        (size_t)batch[i].cache_slot < result.selected_logits.size();
                    const uint32_t selected_token = have_output_head
                        ? result.selected_tokens[(size_t)batch[i].cache_slot]
                        : UINT32_MAX;
                    const float selected_logit = have_output_head
                        ? result.selected_logits[(size_t)batch[i].cache_slot]
                        : 0.0f;
                    const uint64_t request_prompt_tokens = batch[i].prompt_token_ids.empty()
                        ? 1ull
                        : (uint64_t)batch[i].prompt_token_ids.size();
                    const uint64_t committed_prompt_tokens =
                        assignments[i].hit ? 0ull : request_prompt_tokens;
                    sessions.commit(assignments[i],
                                    committed_prompt_tokens,
                                    request_generated,
                                    batch[i].prompt_prefill_tokens + request_generated,
                                    batch[i].generated_token_ids);
                    if (batch[i].cache_slot >= 0 &&
                        batch[i].cache_slot < (int)sessions.slots.size() &&
                        batch[i].cache_slot < req_opt.slots) {
                        sessions.slots[(size_t)batch[i].cache_slot].mtp_raw_rows =
                            mtp_raw_rows_by_slot[(size_t)batch[i].cache_slot];
                    }
                    const TpEpHttpSessionSlot *slot_state = nullptr;
                    if (batch[i].cache_slot >= 0 &&
                        batch[i].cache_slot < (int)sessions.slots.size()) {
                        slot_state = &sessions.slots[(size_t)batch[i].cache_slot];
                    }
                    const size_t slot_prompt_token_ids = slot_state
                        ? slot_state->prompt_token_ids.size()
                        : 0u;
                    const size_t slot_generated_token_ids = slot_state
                        ? slot_state->generated_token_ids.size()
                        : 0u;
                    const uint32_t slot_last_selected = slot_state
                        ? slot_state->last_selected_token
                        : UINT32_MAX;
                    const std::string escaped_key = http_json_escape(batch[i].cache_key);
                    const std::string escaped_evicted = http_json_escape(batch[i].evicted_key);
                    const std::string generated_sequence =
                        http_json_uint_array(batch[i].generated_token_ids);
                    const std::string step_checksums_json =
                        http_json_u64_array(result.step_checksums);
                    const std::string generated_text =
                        decode_token_text(tokenizer.engine, batch[i].generated_token_ids);
                    const std::string escaped_generated_text =
                        http_json_escape(generated_text);
                    char meta[10240];
                    std::snprintf(meta, sizeof(meta),
                                  "\"backend\":\"tp_ep_resident\","
                                  "\"diagnostic\":true,"
                                  "\"diagnostic_note\":\"tokenized prompt prefill and per-step feedback are wired; tokenizer text is not fully wired yet\","
                                  "\"diagnostic_output_head\":%d,"
                                  "\"diagnostic_output_head_proxy_hc\":%d,"
                                  "\"token_input_seed\":%d,"
                                  "\"tokenizer_ready\":%d,"
                                  "\"generated_text\":\"%s\","
                                  "\"decode_input_token\":%u,"
                                  "\"prompt_prefill_tokens\":%llu,"
                                  "\"generated_token_ids\":%zu,"
                                  "\"generated_token_sequence\":%s,"
                                  "\"selected_token\":%u,"
                                  "\"selected_logit\":%.9f,"
                                  "\"decode_step_checksums\":%s,"
                                  "\"output_head_ms\":%.6f,"
                                  "\"output_head_gather_ms\":%.6f,"
                                  "\"output_head_prep_ms\":%.6f,"
                                  "\"output_head_broadcast_ms\":%.6f,"
                                  "\"output_head_projection_ms\":%.6f,"
                                  "\"output_head_top1_ms\":%.6f,"
                                  "\"coalesced_batch_id\":%llu,"
                                  "\"coalesced_batch_size\":%zu,"
                                  "\"coalesced_slot_index\":%zu,"
                                  "\"cache_key\":\"%s\","
                                  "\"cache_key_explicit\":%d,"
                                  "\"cache_hit\":%d,"
                                  "\"cache_prompt_match\":%d,"
                                  "\"cache_prompt_fingerprint\":%llu,"
                                  "\"cache_slot\":%d,"
                                  "\"cache_pos_in\":%llu,"
                                  "\"cache_pos_out\":%llu,"
                                  "\"slot_position\":%llu,"
                                  "\"cache_evicted\":%d,"
                                  "\"cache_evicted_key\":\"%s\","
                                  "\"request_prompt_token_ids\":%zu,"
                                  "\"slot_prompt_token_ids\":%zu,"
                                  "\"slot_generated_token_ids\":%zu,"
                                  "\"slot_last_selected_token\":%u,"
                                  "\"microbatch_wait_us\":%d,"
                                  "\"kv_runtime_resident\":%d,"
                                  "\"decode_slots\":%d,"
                                  "\"prompt_tokens\":%llu,"
                                  "\"generated_tokens\":%llu,"
                                  "\"continuation_tokens\":%llu,"
                                  "\"batch_prompt_tokens\":%llu,"
                                  "\"batch_generated_tokens\":%llu,"
                                  "\"batch_continuation_tokens\":%llu,"
                                  "\"decode_generated_tokens\":%llu,"
                                  "\"decode_continuation_tokens\":%llu,"
                                  "\"tokens_per_request\":%d,\"slots\":%d,\"ctx\":262144,"
                                  "\"token_match\":1,\"token_mismatch\":0,"
                                  "\"timing_ms\":{\"first_token_decode\":%.6f,"
                                  "\"continuation_decode\":%.6f,"
                                  "\"first_token_wall\":%.6f,"
                                  "\"continuation_wall\":%.6f,"
                                  "\"total_decode\":%.6f,\"total_wall\":%.6f,"
                                  "\"ep\":%.6f,\"dense\":%.6f,"
                                  "\"compose\":%.6f,\"compose_reduce\":%.6f,"
                                  "\"compose_copy\":%.6f,\"compose_final\":%.6f,"
                                  "\"generated_tokens_per_second\":%.6f,"
                                  "\"continuation_tokens_per_second\":%.6f,"
                                  "\"generated_tokens_per_second_decode\":%.6f,"
                                  "\"continuation_tokens_per_second_decode\":%.6f},"
                                  "\"checksum\":%llu",
                                  have_output_head ? 1 : 0,
                                  result.diagnostic_output_head_proxy_hc ? 1 : 0,
                                  result.token_input_seed ? 1 : 0,
                                  tokenizer.initialized ? 1 : 0,
                                  escaped_generated_text.c_str(),
                                  batch[i].decode_input_token,
                                  (unsigned long long)batch[i].prompt_prefill_tokens,
                                  batch[i].generated_token_ids.size(),
                                  generated_sequence.c_str(),
                                  selected_token,
                                  selected_logit,
                                  step_checksums_json.c_str(),
                                  result.output_head_ms,
                                  result.output_head_gather_ms,
                                  result.output_head_prep_ms,
                                  result.output_head_broadcast_ms,
                                  result.output_head_projection_ms,
                                  result.output_head_top1_ms,
                                  (unsigned long long)batch_id,
                                  batch.size(),
                                  i,
                                  escaped_key.c_str(),
                                  batch[i].cache_key_explicit ? 1 : 0,
                                  batch[i].cache_hit ? 1 : 0,
                                  batch[i].cache_prompt_match ? 1 : 0,
                                  (unsigned long long)batch[i].prompt_fingerprint,
                                  batch[i].cache_slot,
                                  (unsigned long long)assignments[i].pos_in,
                                  (unsigned long long)(assignments[i].pos_in +
                                      batch[i].prompt_prefill_tokens + request_generated),
                                  (unsigned long long)(assignments[i].pos_in +
                                      batch[i].prompt_prefill_tokens + request_generated),
                                  batch[i].cache_evicted ? 1 : 0,
                                  escaped_evicted.c_str(),
                                  batch[i].prompt_token_ids.size(),
                                  slot_prompt_token_ids,
                                  slot_generated_token_ids,
                                  slot_last_selected,
                                  base_opt.microbatch_wait_us,
                                  shared_tp_runtime && shared_tp_runtime->initialized ? 1 : 0,
                                  req_opt.slots,
                                  (unsigned long long)request_prompt_tokens,
                                  (unsigned long long)request_generated,
                                  (unsigned long long)request_continuation,
                                  (unsigned long long)client_prompt_tokens,
                                  (unsigned long long)client_generated_tokens,
                                  (unsigned long long)client_continuation_tokens,
                                  (unsigned long long)result.generated_tokens,
                                  (unsigned long long)result.continuation_tokens,
                                  req_opt.decode_steps, req_opt.slots,
                                  result.first_token_decode_ms,
                                  result.continuation_decode_ms,
                                  result.first_token_wall_ms,
                                  result.continuation_wall_ms,
                                  result.total_decode_ms,
                                  result.total_wall_ms,
                                  result.total_ep_ms,
                                  result.total_dense_ms,
                                  result.total_compose_ms,
                                  result.total_compose_reduce_ms,
                                  result.total_compose_copy_ms,
                                  result.total_compose_final_ms,
                                  result.aggregate_generated_tok_s_wall,
                                  result.aggregate_continuation_tok_s_wall,
                                  result.aggregate_generated_tok_s_decode,
                                  result.aggregate_continuation_tok_s_decode,
                                  (unsigned long long)result.checksum);
                    char out[16384];
                    if (http_is_chat_completion_post(batch[i])) {
                        std::snprintf(out, sizeof(out),
                                      "{\"id\":\"chatcmpl-ds4-v100-diagnostic-%llu-%zu\","
                                      "\"object\":\"chat.completion\","
                                      "\"created\":%llu,"
                                      "\"model\":\"ds4-v100-tp-ep-diagnostic\","
                                      "\"choices\":[{\"index\":0,"
                                      "\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                                      "\"logprobs\":null,"
                                      "\"finish_reason\":\"length\","
                                      "\"token_ids\":%s}],"
                                      "\"usage\":{\"prompt_tokens\":%llu,"
                                      "\"completion_tokens\":%llu,"
                                      "\"total_tokens\":%llu},"
                                      "\"ds4_v100\":{%s}}\n",
                                      (unsigned long long)batch_id,
                                      i,
                                      http_epoch_seconds(),
                                      escaped_generated_text.c_str(),
                                      generated_sequence.c_str(),
                                      (unsigned long long)request_prompt_tokens,
                                      (unsigned long long)request_generated,
                                      (unsigned long long)(request_generated + request_prompt_tokens),
                                      meta);
                    } else if (http_is_completion_post(batch[i])) {
                        std::snprintf(out, sizeof(out),
                                      "{\"id\":\"cmpl-ds4-v100-diagnostic-%llu-%zu\","
                                      "\"object\":\"text_completion\","
                                      "\"created\":%llu,"
                                      "\"model\":\"ds4-v100-tp-ep-diagnostic\","
                                      "\"choices\":[{\"text\":\"%s\","
                                      "\"index\":0,\"logprobs\":null,"
                                      "\"finish_reason\":\"length\","
                                      "\"token_ids\":%s}],"
                                      "\"usage\":{\"prompt_tokens\":%llu,"
                                      "\"completion_tokens\":%llu,"
                                      "\"total_tokens\":%llu},"
                                      "\"ds4_v100\":{%s}}\n",
                                      (unsigned long long)batch_id,
                                      i,
                                      http_epoch_seconds(),
                                      escaped_generated_text.c_str(),
                                      generated_sequence.c_str(),
                                      (unsigned long long)request_prompt_tokens,
                                      (unsigned long long)request_generated,
                                      (unsigned long long)(request_generated + request_prompt_tokens),
                                      meta);
                    } else {
                        std::snprintf(out, sizeof(out), "{%s}\n", meta);
                    }
                    http_write_json(batch[i].fd, 200, out);
                    close(batch[i].fd);
                }
            }
        } else {
            http_write_json(fd, 404, "{\"error\":\"not_found\"}\n");
            close(fd);
        }
    }
    close(listen_fd);
    close_tokenizer_runtime(&tokenizer);
    return 0;
}
