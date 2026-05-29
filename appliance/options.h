bool parse_int(const char *text, int *out) {
    if (!text || !*text) return false;
    char *end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = (int)v;
    return true;
}

bool parse_u64(const char *text, uint64_t *out) {
    if (!text || !*text) return false;
    char *end = nullptr;
    const unsigned long long v = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') return false;
    *out = (uint64_t)v;
    return true;
}

bool parse_size(const char *text, size_t *out) {
    uint64_t v = 0;
    if (!parse_u64(text, &v)) return false;
    if (v > (uint64_t)std::numeric_limits<size_t>::max()) return false;
    *out = (size_t)v;
    return true;
}

bool parse_devices(const char *text, int devices[kGpus]) {
    std::vector<int> parsed;
    const char *cur = text;
    while (cur && *cur) {
        const char *comma = std::strchr(cur, ',');
        std::string piece;
        if (comma) {
            piece.assign(cur, comma - cur);
            cur = comma + 1;
        } else {
            piece.assign(cur);
            cur = nullptr;
        }
        int dev = 0;
        if (!parse_int(piece.c_str(), &dev) || dev < 0) return false;
        parsed.push_back(dev);
    }
    if ((int)parsed.size() != kGpus) return false;
    for (int i = 0; i < kGpus; ++i) {
        for (int j = i + 1; j < kGpus; ++j) {
            if (parsed[i] == parsed[j]) return false;
        }
        devices[i] = parsed[i];
    }
    return true;
}

void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s --pack-dir DIR --contract FILE --tm-index FILE [options]\n"
                 "       [--lib PATH] [--tokenizer-model PATH]\n"
                 "       [--slots N]\n"
                 "       [--position N] [--decode-steps N]\n"
                 "       [--serve-http] [--host ADDR] [--port N]\n"
                 "       [--microbatch-wait-us N]\n"
                 "       [--max-requests N]\n"
                 "       [--vram-min-free-mib N]\n"
                 "       [--nccl-min-free-mib N]\n"
                 "       [--help]\n",
                 argv0);
}
bool parse_args(int argc, char **argv, Options *opt) {
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        const char *val = (i + 1 < argc) ? argv[i + 1] : nullptr;
        if (std::strcmp(arg, "--lib") == 0) {
            if (!val) return false;
            opt->lib_path = val;
            ++i;
        } else if (std::strcmp(arg, "--pack-dir") == 0) {
            if (!val) return false;
            opt->pack_dir = val;
            ++i;
        } else if (std::strcmp(arg, "--contract") == 0) {
            if (!val) return false;
            opt->contract_path = val;
            ++i;
        } else if (std::strcmp(arg, "--tm-index") == 0) {
            if (!val) return false;
            opt->tm_index_path = val;
            ++i;
        } else if (std::strcmp(arg, "--tokenizer-model") == 0) {
            if (!val) return false;
            opt->tokenizer_model_path = val;
            ++i;
        } else if (std::strcmp(arg, "--slots") == 0) {
            if (!val || !parse_int(val, &opt->slots) || opt->slots <= 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--position") == 0) {
            if (!val || !parse_u64(val, &opt->position)) return false;
            ++i;
        } else if (std::strcmp(arg, "--decode-steps") == 0) {
            if (!val || !parse_int(val, &opt->decode_steps) || opt->decode_steps < 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--serving-bench") == 0) {
            opt->serving_bench = true;
        } else if (std::strcmp(arg, "--serve-http") == 0) {
            opt->serve_http = true;
            opt->serving_bench = true;
            opt->token_major_all_layers = true;
            opt->all_layers = true;
            opt->share_tp_runtime = true;
            opt->tp_runtime_explicit = true;
            opt->skip_decode_checksum = true;
        } else if (std::strcmp(arg, "--host") == 0) {
            if (!val) return false;
            opt->host = val;
            ++i;
        } else if (std::strcmp(arg, "--port") == 0) {
            if (!val || !parse_int(val, &opt->port) || opt->port <= 0 || opt->port > 65535) return false;
            ++i;
        } else if (std::strcmp(arg, "--microbatch-wait-us") == 0) {
            if (!val || !parse_int(val, &opt->microbatch_wait_us) ||
                opt->microbatch_wait_us < 0 || opt->microbatch_wait_us > 1000000) return false;
            ++i;
        } else if (std::strcmp(arg, "--max-requests") == 0) {
            if (!val || !parse_int(val, &opt->max_requests) ||
                opt->max_requests < 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--vram-min-free-mib") == 0) {
            if (!val || !parse_u64(val, &opt->vram_min_free_mib)) return false;
            ++i;
        } else if (std::strcmp(arg, "--nccl-min-free-mib") == 0) {
            if (!val || !parse_u64(val, &opt->nccl_min_free_mib)) return false;
            ++i;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }
    return opt->pack_dir && opt->contract_path && opt->tm_index_path &&
           opt->top_k == kModelTopK && opt->top_k <= kPackedLocalExperts &&
           opt->layer >= 0 && !(opt->dense_hmma_compose && opt->dense_f16_cublas_compose) &&
           (!opt->dense_f16_cache_compose || opt->dense_f16_cublas_compose);
}
