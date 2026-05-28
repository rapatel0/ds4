struct Options {
    const char *lib_path = "./build/turbomind-v100/libggml-turbomind.so";
    const char *pack_dir = nullptr;
    const char *contract_path = nullptr;
    const char *tm_index_path = nullptr;
    const char *tokenizer_model_path = nullptr;
    int devices[kGpus] = {0, 1, 2, 3, 4, 5, 6, 7};
    int slots = 32;
    int top_k = 6;
    int layer = 2;
    int resident_profile_layer = -1;
    uint32_t kv_slot = 7;
    uint64_t position = 1024;
    int warmup = 5;
    int iters = 30;
    const char *dense_compute_tensor = nullptr;
    bool dense_compute_all_f8 = false;
    bool dense_compute_all_bf16 = false;
    bool compose_next_hidden = false;
    int decode_steps = 0;
    bool ep_return_fp16 = false;
    bool fuse_compose_sum = true;
    bool dense_hmma_compose = false;
    bool dense_f16_cublas_compose = true;
    bool dense_f16_cache_compose = true;
    bool all_layers = true;
    bool skip_descriptor_checks = true;
    bool skip_predecode_probes = true;
    bool share_tp_runtime = true;
    bool tp_runtime_explicit = false;
    bool tp_runtime_skip_unused_comp_state = true;
    uint64_t tp_runtime_scratch_mib = 1024;
    bool share_expert_bindings = true;
    bool parallel_expert_load_gate = true;
    bool overlap_ep_dense = true;
    bool direct_remote_compose = false;
    bool source_copy_schedule = true;
    bool copy_event_compose = true;
    bool compact_route_compose = true;
    bool token_major_all_layers = true;
    bool share_dense_ops = true;
    bool skip_self_compose_copy = true;
    bool multi_copy_streams = true;
    bool nccl_reduce_scatter_compose_gate = false;
    bool defer_nccl_init_gate = true;
    bool serving_bench = false;
    bool skip_decode_checksum = false;
    bool serve_http = false;
    const char *host = "127.0.0.1";
    int port = 18082;
    int max_requests = 0;
    int microbatch_wait_us = 5000;
    bool output_head_gate = false;
    bool output_head_resident_gate = false;
    bool decode_cudagraph_gate = false;
    bool decode_cudagraph_replay_probe_gate = false;
    bool decode_cudagraph_persistent_replay_gate = false;
    bool decode_cudagraph_output_sync_gate = false;
    bool decode_cudagraph_hc_current_sync_gate = false;
    const char *decode_cudagraph_stage_sync = nullptr;
    const char *decode_cudagraph_suffix_stage = nullptr;
    bool decode_stage_checksum_gate = false;
    bool compact_moe_decode_gate = true;
    bool fused_gated_silu_gate = false;
    bool final_hc_carry_gate = true;
    bool diagnostic_output_head = true;
    bool diagnostic_output_head_lazy_gate = true;
    bool tp_hc_final_expand_gate = true;
    bool tp_hc_current_input_gate = true;
    bool tp_hc_current_input_peer_gather_gate = false;
    bool tp_hc_current_input_nccl_allgather_gate = true;
    bool tp_hc_current_allreduce_gate = true;
    bool tp_hc_current_input_stream_sync_gate = true;
    bool tp_hc_current_input_fused_fill_pack_gate = false;
    bool tp_hc_current_full_parity_gate = false;
    bool tp_hc_persist_state_gate = true;
    bool tp_peer_accounting_gate = false;
    bool tp_peer_reject_sys_gate = false;
    bool model_router_routes = true;
    bool router_cublas_gate = false;
    bool router_hash_fast_gate = false;
    bool gpu_route_plan_gate = true;
    bool route_plan_async_upload_gate = true;
    bool routed_ffn_norm_input_gate = true;
    bool routed_ffn_rank_major_input_gate = false;
    bool routed_ffn_rank_major_shared_input_gate = false;
    bool routed_ffn_rank_major_route_input_gate = false;
    bool routed_ffn_rank_major_input_parity_gate = false;
    bool post_attention_route_reuse_audit_gate = false;
    bool post_attention_fixed_capacity_route_plan_gate = true;
    bool post_attention_device_actual_route_sync_gate = false;
    int post_attention_static_rank_route_cap = 0;
    int post_attention_static_executor_route_cap = 0;
    int post_attention_static_compose_route_cap = 0;
    bool post_attention_masked_compact_copy_gate = false;
    bool post_attention_slot_major_ffn_norm_gate = false;
    bool post_attention_skip_slot_major_ffn_norm_gate = false;
    bool model_router_rank_major_logits_gate = false;
    bool model_router_allreduce_logits_gate = true;
    bool true_shared_ffn_gate = true;
    bool tp_kv_all_slots_gate = false;
    bool reference_hc_reduce_gate = false;
    bool reference_hc_state_guard_gate = false;
    bool true_ds4_attention_residency_gate = true;
    bool true_ds4_attention_projection_gate = true;
    bool true_ds4_attention_projection_direct_input_fill_gate = false;
    bool true_ds4_attention_projection_rank_local_input_gate = false;
    bool true_ds4_attention_projection_rank_major_input_gate = true;
    bool true_ds4_attention_projection_input_parity_gate = false;
    bool true_ds4_attention_state_gate = true;
    bool true_ds4_attention_rope_gate = true;
    bool true_ds4_attention_saturation_audit_gate = false;
    bool true_ds4_attention_kv_norm_reference_gate = false;
    bool true_ds4_attention_raw_read_gate = true;
    bool true_ds4_attention_raw_window_gate = true;
    bool true_ds4_attention_typed_kv_raw_gate = true;
    bool true_ds4_attention_typed_kv_compressed_gate = true;
    bool true_ds4_attention_typed_kv_indexer_gate = true;
    bool true_ds4_attention_typed_kv_history_gate = true;
    bool true_ds4_attention_typed_kv_skip_current_load_gate = true;
    bool true_ds4_attention_typed_kv_skip_raw_store_gate = false;
    bool true_ds4_attention_typed_kv_skip_compressed_store_gate = false;
    bool true_ds4_attention_typed_kv_skip_indexer_store_gate = false;
    bool true_ds4_attention_typed_kv_quiet_gate = true;
    bool true_ds4_attention_typed_kv_batch_rows_gate = true;
    bool true_ds4_attention_typed_kv_stream_sync_gate = true;
    bool fp8_e5m2_kv_gate = false;
    bool true_ds4_attention_output_gate = true;
    bool true_ds4_attention_output_nccl_allgather_gate = false;
    bool true_ds4_post_attention_ffn_input_gate = true;
    bool true_ds4_semantic_skip_stats_gate = true;
    bool true_ds4_compressed_kv_gate = false;
    bool true_ds4_indexer_attention_gate = false;
    bool true_ds4_compressed_kv_direct_input_fill_gate = false;
    bool true_ds4_compressed_kv_dense_event_wait_gate = true;
    bool true_ds4_compressed_kv_skip_dense_stats_gate = true;
    bool true_ds4_compressed_kv_fused_attn_input_fill_gate = false;
    bool true_ds4_compressed_kv_fused_input_fill_gate = false;
    bool true_ds4_compressed_kv_fused_rope_round_gate = false;
    bool true_ds4_compressed_kv_fused_pool_norm_gate = true;
    bool true_ds4_compressed_kv_fused_pool_norm_rope_round_gate = false;
    bool true_ds4_compressed_reference_diff_gate = false;
    bool cuda_profiler_window = false;
    bool cuda_profiler_all_devices = false;
    int cuda_profiler_device = -1;
    uint32_t true_ds4_attention_raw_valid_rows = 1;
    uint64_t vram_min_free_mib = 0;
    uint64_t nccl_min_free_mib = 0;
    bool vram_report = false;
};

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
                 "       [--devices 0,1,2,3,4,5,6,7]\n"
                 "       [--slots N] [--top-k N] [--kv-slot N]\n"
                 "       [--position N] [--decode-steps N]\n"
                 "       [--serve-http] [--host ADDR] [--port N] [--max-requests N]\n"
                 "       [--microbatch-wait-us N]\n"
                 "       [--vram-report] [--vram-min-free-mib N]\n"
                 "       [--nccl-min-free-mib N]\n"
                 "       [--cuda-profiler-window] [--cuda-profiler-device N]\n"
                 "       [--cuda-profiler-all-devices]\n"
                 "       [--decode-cudagraph-gate]\n"
                 "       [--decode-cudagraph-persistent-replay-gate]\n"
                 "       [--decode-cudagraph-output-sync-gate]\n"
                 "       [--decode-cudagraph-hc-current-sync-gate]\n"
                 "       [--decode-cudagraph-stage-sync-gate STAGES]\n"
                 "       [--decode-cudagraph-suffix-stage-gate STAGE]\n"
                 "       [--decode-stage-checksum-gate]\n"
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
        } else if (std::strcmp(arg, "--devices") == 0) {
            if (!val || !parse_devices(val, opt->devices)) return false;
            ++i;
        } else if (std::strcmp(arg, "--slots") == 0) {
            if (!val || !parse_int(val, &opt->slots) || opt->slots <= 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--top-k") == 0) {
            if (!val || !parse_int(val, &opt->top_k) || opt->top_k <= 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--layer") == 0) {
            if (!val || !parse_int(val, &opt->layer)) return false;
            ++i;
        } else if (std::strcmp(arg, "--resident-profile-layer") == 0) {
            if (!val || !parse_int(val, &opt->resident_profile_layer) ||
                opt->resident_profile_layer < 0 || opt->resident_profile_layer >= 43) {
                return false;
            }
            opt->layer = opt->resident_profile_layer;
            opt->all_layers = true;
            opt->share_tp_runtime = true;
            opt->tp_runtime_explicit = true;
            opt->share_expert_bindings = true;
            opt->share_dense_ops = true;
            ++i;
        } else if (std::strcmp(arg, "--kv-slot") == 0) {
            int slot = 0;
            if (!val || !parse_int(val, &slot) || slot < 0) return false;
            opt->kv_slot = (uint32_t)slot;
            ++i;
        } else if (std::strcmp(arg, "--position") == 0) {
            if (!val || !parse_u64(val, &opt->position)) return false;
            ++i;
        } else if (std::strcmp(arg, "--warmup") == 0) {
            if (!val || !parse_int(val, &opt->warmup) || opt->warmup < 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--iters") == 0) {
            if (!val || !parse_int(val, &opt->iters) || opt->iters <= 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--dense-compute-tensor") == 0) {
            if (!val) return false;
            opt->dense_compute_tensor = val;
            ++i;
        } else if (std::strcmp(arg, "--dense-compute-all-f8") == 0) {
            opt->dense_compute_all_f8 = true;
        } else if (std::strcmp(arg, "--dense-compute-all-bf16") == 0) {
            opt->dense_compute_all_bf16 = true;
        } else if (std::strcmp(arg, "--dense-compute-all") == 0) {
            opt->dense_compute_all_f8 = true;
            opt->dense_compute_all_bf16 = true;
        } else if (std::strcmp(arg, "--compose-next-hidden") == 0) {
            opt->compose_next_hidden = true;
        } else if (std::strcmp(arg, "--decode-steps") == 0) {
            if (!val || !parse_int(val, &opt->decode_steps) || opt->decode_steps < 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--ep-return-fp16") == 0) {
            opt->ep_return_fp16 = true;
        } else if (std::strcmp(arg, "--fuse-compose-sum") == 0) {
            opt->fuse_compose_sum = true;
        } else if (std::strcmp(arg, "--dense-hmma-compose") == 0) {
            opt->dense_hmma_compose = true;
        } else if (std::strcmp(arg, "--dense-f16-cublas-compose") == 0) {
            opt->dense_f16_cublas_compose = true;
        } else if (std::strcmp(arg, "--dense-f16-cache-compose") == 0) {
            opt->dense_f16_cache_compose = true;
        } else if (std::strcmp(arg, "--all-layers") == 0) {
            opt->all_layers = true;
        } else if (std::strcmp(arg, "--skip-descriptor-checks") == 0) {
            opt->skip_descriptor_checks = true;
        } else if (std::strcmp(arg, "--skip-predecode-probes") == 0) {
            opt->skip_predecode_probes = true;
        } else if (std::strcmp(arg, "--share-tp-runtime") == 0) {
            opt->share_tp_runtime = true;
            opt->tp_runtime_explicit = true;
        } else if (std::strcmp(arg, "--local-tp-runtime") == 0) {
            opt->share_tp_runtime = false;
            opt->tp_runtime_explicit = true;
        } else if (std::strcmp(arg, "--tp-runtime-skip-unused-comp-state-gate") == 0) {
            opt->tp_runtime_skip_unused_comp_state = true;
        } else if (std::strcmp(arg, "--tp-runtime-scratch-mib") == 0) {
            if (!val || !parse_u64(val, &opt->tp_runtime_scratch_mib) ||
                opt->tp_runtime_scratch_mib < 64 || opt->tp_runtime_scratch_mib > 4096) {
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--shared-expert-bindings") == 0) {
            opt->share_expert_bindings = true;
        } else if (std::strcmp(arg, "--local-expert-bindings") == 0) {
            opt->share_expert_bindings = false;
        } else if (std::strcmp(arg, "--parallel-expert-load-gate") == 0) {
            opt->parallel_expert_load_gate = true;
        } else if (std::strcmp(arg, "--overlap-ep-dense") == 0) {
            opt->overlap_ep_dense = true;
        } else if (std::strcmp(arg, "--serial-ep-dense") == 0) {
            opt->overlap_ep_dense = false;
        } else if (std::strcmp(arg, "--direct-remote-compose") == 0) {
            opt->direct_remote_compose = true;
        } else if (std::strcmp(arg, "--source-copy-schedule") == 0) {
            opt->source_copy_schedule = true;
        } else if (std::strcmp(arg, "--dest-copy-schedule") == 0) {
            opt->source_copy_schedule = false;
        } else if (std::strcmp(arg, "--copy-event-compose") == 0) {
            opt->copy_event_compose = true;
        } else if (std::strcmp(arg, "--compact-route-compose") == 0) {
            opt->compact_route_compose = true;
        } else if (std::strcmp(arg, "--compact-moe-decode-gate") == 0) {
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
        } else if (std::strcmp(arg, "--fused-gated-silu-gate") == 0) {
            opt->fused_gated_silu_gate = true;
        } else if (std::strcmp(arg, "--token-major-all-layers") == 0) {
            opt->token_major_all_layers = true;
        } else if (std::strcmp(arg, "--shared-dense-ops") == 0) {
            opt->share_dense_ops = true;
        } else if (std::strcmp(arg, "--skip-self-compose-copy") == 0) {
            opt->skip_self_compose_copy = true;
        } else if (std::strcmp(arg, "--copy-self-compose") == 0) {
            opt->skip_self_compose_copy = false;
        } else if (std::strcmp(arg, "--multi-copy-streams") == 0) {
            opt->multi_copy_streams = true;
        } else if (std::strcmp(arg, "--nccl-reduce-scatter-compose-gate") == 0) {
            opt->nccl_reduce_scatter_compose_gate = true;
        } else if (std::strcmp(arg, "--defer-nccl-init-gate") == 0) {
            opt->defer_nccl_init_gate = true;
        } else if (std::strcmp(arg, "--serving-bench") == 0) {
            opt->serving_bench = true;
        } else if (std::strcmp(arg, "--skip-decode-checksum") == 0) {
            opt->skip_decode_checksum = true;
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
        } else if (std::strcmp(arg, "--max-requests") == 0) {
            if (!val || !parse_int(val, &opt->max_requests) || opt->max_requests < 0) return false;
            ++i;
        } else if (std::strcmp(arg, "--microbatch-wait-us") == 0) {
            if (!val || !parse_int(val, &opt->microbatch_wait_us) ||
                opt->microbatch_wait_us < 0 || opt->microbatch_wait_us > 1000000) return false;
            ++i;
        } else if (std::strcmp(arg, "--vram-report") == 0) {
            opt->vram_report = true;
        } else if (std::strcmp(arg, "--vram-min-free-mib") == 0) {
            if (!val || !parse_u64(val, &opt->vram_min_free_mib)) return false;
            ++i;
        } else if (std::strcmp(arg, "--nccl-min-free-mib") == 0) {
            if (!val || !parse_u64(val, &opt->nccl_min_free_mib)) return false;
            ++i;
        } else if (std::strcmp(arg, "--output-head-gate") == 0) {
            opt->output_head_gate = true;
        } else if (std::strcmp(arg, "--output-head-resident-gate") == 0) {
            opt->output_head_resident_gate = true;
        } else if (std::strcmp(arg, "--decode-cudagraph-gate") == 0) {
            opt->decode_cudagraph_gate = true;
        } else if (std::strcmp(arg, "--decode-cudagraph-replay-probe-gate") == 0) {
            opt->decode_cudagraph_gate = true;
            opt->decode_cudagraph_replay_probe_gate = true;
        } else if (std::strcmp(arg, "--decode-cudagraph-persistent-replay-gate") == 0) {
            opt->decode_cudagraph_gate = true;
            opt->decode_cudagraph_replay_probe_gate = true;
            opt->decode_cudagraph_persistent_replay_gate = true;
        } else if (std::strcmp(arg, "--decode-cudagraph-output-sync-gate") == 0) {
            opt->decode_cudagraph_gate = true;
            opt->decode_cudagraph_output_sync_gate = true;
        } else if (std::strcmp(arg, "--decode-cudagraph-hc-current-sync-gate") == 0) {
            opt->decode_cudagraph_gate = true;
            opt->decode_cudagraph_hc_current_sync_gate = true;
        } else if (std::strcmp(arg, "--decode-cudagraph-stage-sync-gate") == 0) {
            if (i + 1 >= argc) return false;
            opt->decode_cudagraph_gate = true;
            opt->decode_cudagraph_stage_sync = argv[++i];
        } else if (std::strcmp(arg, "--decode-cudagraph-suffix-stage-gate") == 0) {
            if (i + 1 >= argc) return false;
            const char *stage = argv[++i];
            if (std::strcmp(stage, "routed_ffn") != 0 &&
                std::strcmp(stage, "dense") != 0 &&
                std::strcmp(stage, "compose") != 0 &&
                std::strcmp(stage, "final_hc") != 0 &&
                std::strcmp(stage, "compose_eager_final_hc") != 0) {
                return false;
            }
            opt->decode_cudagraph_suffix_stage = stage;
        } else if (std::strcmp(arg, "--decode-stage-checksum-gate") == 0) {
            opt->decode_stage_checksum_gate = true;
        } else if (std::strcmp(arg, "--final-hc-carry-gate") == 0) {
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-final-expand-gate") == 0) {
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-gate") == 0) {
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-peer-gather-gate") == 0) {
            opt->tp_hc_current_input_peer_gather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-nccl-allgather-gate") == 0) {
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-allreduce-gate") == 0) {
            opt->tp_hc_current_allreduce_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-stream-sync-gate") == 0) {
            opt->tp_hc_current_input_stream_sync_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-input-fused-fill-pack-gate") == 0) {
            opt->tp_hc_current_input_fused_fill_pack_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-current-full-parity-gate") == 0) {
            opt->tp_hc_current_full_parity_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-peer-accounting-gate") == 0) {
            opt->tp_peer_accounting_gate = true;
        } else if (std::strcmp(arg, "--tp-peer-reject-sys-gate") == 0) {
            opt->tp_peer_reject_sys_gate = true;
            opt->tp_peer_accounting_gate = true;
        } else if (std::strcmp(arg, "--model-router-routes") == 0) {
            opt->model_router_routes = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--router-cublas-gate") == 0) {
            opt->router_cublas_gate = true;
            opt->model_router_routes = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--router-hash-fast-gate") == 0) {
            opt->router_hash_fast_gate = true;
            opt->model_router_routes = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--gpu-route-plan-gate") == 0) {
            opt->gpu_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--route-plan-async-upload-gate") == 0) {
            opt->route_plan_async_upload_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--routed-ffn-norm-input-gate") == 0) {
            opt->routed_ffn_norm_input_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--routed-ffn-rank-major-input-gate") == 0) {
            opt->routed_ffn_rank_major_input_gate = true;
            opt->routed_ffn_rank_major_shared_input_gate = true;
            opt->routed_ffn_rank_major_route_input_gate = true;
            opt->routed_ffn_norm_input_gate = true;
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--routed-ffn-rank-major-shared-input-gate") == 0) {
            opt->routed_ffn_rank_major_shared_input_gate = true;
            opt->routed_ffn_norm_input_gate = true;
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--routed-ffn-rank-major-route-input-gate") == 0) {
            opt->routed_ffn_rank_major_route_input_gate = true;
            opt->routed_ffn_norm_input_gate = true;
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--routed-ffn-rank-major-input-parity-gate") == 0) {
            opt->routed_ffn_rank_major_input_parity_gate = true;
            opt->routed_ffn_norm_input_gate = true;
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-route-reuse-audit-gate") == 0) {
            opt->post_attention_route_reuse_audit_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-fixed-capacity-route-plan-gate") == 0) {
            opt->post_attention_fixed_capacity_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-device-actual-route-sync-gate") == 0) {
            opt->post_attention_device_actual_route_sync_gate = true;
            opt->post_attention_fixed_capacity_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-static-rank-route-cap") == 0) {
            if (!val || !parse_int(val, &opt->post_attention_static_rank_route_cap) ||
                opt->post_attention_static_rank_route_cap <= 0) {
                return false;
            }
            ++i;
            opt->post_attention_fixed_capacity_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-static-executor-route-cap") == 0) {
            if (!val || !parse_int(val, &opt->post_attention_static_executor_route_cap) ||
                opt->post_attention_static_executor_route_cap <= 0) {
                return false;
            }
            ++i;
            opt->post_attention_fixed_capacity_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-static-compose-route-cap") == 0) {
            if (!val || !parse_int(val, &opt->post_attention_static_compose_route_cap) ||
                opt->post_attention_static_compose_route_cap <= 0) {
                return false;
            }
            ++i;
            opt->post_attention_fixed_capacity_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-masked-compact-copy-gate") == 0) {
            opt->post_attention_masked_compact_copy_gate = true;
            opt->post_attention_fixed_capacity_route_plan_gate = true;
            opt->model_router_routes = true;
            opt->compact_moe_decode_gate = true;
            opt->compact_route_compose = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--post-attention-slot-major-ffn-norm-gate") == 0) {
            opt->post_attention_slot_major_ffn_norm_gate = true;
        } else if (std::strcmp(arg, "--post-attention-skip-slot-major-ffn-norm-gate") == 0) {
            opt->post_attention_skip_slot_major_ffn_norm_gate = true;
        } else if (std::strcmp(arg, "--model-router-rank-major-logits-gate") == 0) {
            opt->model_router_rank_major_logits_gate = true;
            opt->model_router_routes = true;
            opt->routed_ffn_rank_major_input_gate = true;
            opt->routed_ffn_rank_major_shared_input_gate = true;
            opt->routed_ffn_rank_major_route_input_gate = true;
            opt->routed_ffn_norm_input_gate = true;
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--model-router-allreduce-logits-gate") == 0) {
            opt->model_router_allreduce_logits_gate = true;
            opt->model_router_routes = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-shared-ffn-gate") == 0) {
            opt->true_shared_ffn_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-residency-gate") == 0) {
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-projection-gate") == 0) {
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-projection-direct-input-fill-gate") == 0) {
            opt->true_ds4_attention_projection_direct_input_fill_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-projection-rank-local-input-gate") == 0) {
            opt->true_ds4_attention_projection_rank_local_input_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-projection-rank-major-input-gate") == 0) {
            opt->true_ds4_attention_projection_rank_major_input_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_current_input_nccl_allgather_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-projection-input-parity-gate") == 0) {
            opt->true_ds4_attention_projection_input_parity_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-state-gate") == 0) {
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-rope-gate") == 0) {
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-saturation-audit-gate") == 0) {
            opt->true_ds4_attention_saturation_audit_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-kv-norm-reference-gate") == 0) {
            opt->true_ds4_attention_kv_norm_reference_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-raw-read-gate") == 0) {
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-raw-window-gate") == 0) {
            opt->true_ds4_attention_raw_window_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-raw-gate") == 0) {
            opt->true_ds4_attention_typed_kv_raw_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-compressed-gate") == 0) {
            opt->true_ds4_attention_typed_kv_compressed_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-indexer-gate") == 0) {
            opt->true_ds4_attention_typed_kv_indexer_gate = true;
            opt->true_ds4_indexer_attention_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-history-gate") == 0) {
            opt->true_ds4_attention_typed_kv_history_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-skip-current-load-gate") == 0) {
            opt->true_ds4_attention_typed_kv_skip_current_load_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-skip-raw-store-gate") == 0) {
            opt->true_ds4_attention_typed_kv_skip_raw_store_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-skip-compressed-store-gate") == 0) {
            opt->true_ds4_attention_typed_kv_skip_compressed_store_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-skip-indexer-store-gate") == 0) {
            opt->true_ds4_attention_typed_kv_skip_indexer_store_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-quiet-gate") == 0) {
            opt->true_ds4_attention_typed_kv_quiet_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-batch-rows-gate") == 0) {
            opt->true_ds4_attention_typed_kv_batch_rows_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-typed-kv-stream-sync-gate") == 0) {
            opt->true_ds4_attention_typed_kv_stream_sync_gate = true;
        } else if (std::strcmp(arg, "--fp8-e5m2-kv-gate") == 0) {
            opt->fp8_e5m2_kv_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-output-gate") == 0) {
            opt->true_ds4_attention_output_gate = true;
            opt->true_ds4_attention_raw_window_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-attention-output-nccl-allgather-gate") == 0) {
            opt->true_ds4_attention_output_nccl_allgather_gate = true;
            opt->true_ds4_attention_output_gate = true;
            opt->true_ds4_attention_raw_window_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-post-attention-ffn-input-gate") == 0) {
            opt->true_ds4_post_attention_ffn_input_gate = true;
            opt->true_ds4_attention_output_gate = true;
            opt->true_ds4_attention_raw_window_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_raw_read_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->true_shared_ffn_gate = true;
            opt->routed_ffn_norm_input_gate = true;
            opt->model_router_routes = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-semantic-skip-stats-gate") == 0) {
            opt->true_ds4_semantic_skip_stats_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-gate") == 0) {
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-indexer-attention-gate") == 0) {
            opt->true_ds4_indexer_attention_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-direct-input-fill-gate") == 0) {
            opt->true_ds4_compressed_kv_direct_input_fill_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-dense-event-wait-gate") == 0) {
            opt->true_ds4_compressed_kv_dense_event_wait_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-skip-dense-stats-gate") == 0) {
            opt->true_ds4_compressed_kv_skip_dense_stats_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-attn-input-fill-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_attn_input_fill_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-input-fill-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_input_fill_gate = true;
            opt->true_ds4_indexer_attention_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-rope-round-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_rope_round_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-pool-norm-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_pool_norm_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-kv-fused-pool-norm-rope-round-gate") == 0) {
            opt->true_ds4_compressed_kv_fused_pool_norm_rope_round_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_rope_gate = true;
            opt->true_ds4_attention_state_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--true-ds4-compressed-reference-diff-gate") == 0) {
            opt->true_ds4_compressed_reference_diff_gate = true;
            opt->true_ds4_indexer_attention_gate = true;
            opt->true_ds4_compressed_kv_gate = true;
            opt->true_ds4_attention_projection_gate = true;
            opt->true_ds4_attention_residency_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--reference-hc-reduce-gate") == 0) {
            opt->reference_hc_reduce_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--reference-hc-state-guard-gate") == 0) {
            opt->reference_hc_state_guard_gate = true;
            opt->reference_hc_reduce_gate = true;
            opt->tp_hc_current_input_gate = true;
            opt->tp_hc_final_expand_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-hc-persist-state-gate") == 0) {
            opt->tp_hc_persist_state_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--tp-kv-all-slots-gate") == 0) {
            opt->tp_kv_all_slots_gate = true;
        } else if (std::strcmp(arg, "--cuda-profiler-window") == 0) {
            opt->cuda_profiler_window = true;
        } else if (std::strcmp(arg, "--cuda-profiler-device") == 0) {
            if (!val || !parse_int(val, &opt->cuda_profiler_device) ||
                opt->cuda_profiler_device < 0 || opt->cuda_profiler_device >= kGpus) {
                return false;
            }
            opt->cuda_profiler_window = true;
            ++i;
        } else if (std::strcmp(arg, "--cuda-profiler-all-devices") == 0) {
            opt->cuda_profiler_window = true;
            opt->cuda_profiler_all_devices = true;
        } else if (std::strcmp(arg, "--diagnostic-output-head") == 0) {
            opt->diagnostic_output_head = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--diagnostic-output-head-lazy-gate") == 0) {
            opt->diagnostic_output_head = true;
            opt->diagnostic_output_head_lazy_gate = true;
            opt->final_hc_carry_gate = true;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }
    return opt->pack_dir && opt->contract_path && opt->tm_index_path &&
           opt->top_k <= kPackedLocalExperts && opt->layer >= 0 &&
           (!opt->model_router_routes || opt->top_k == kModelTopK) &&
           (!opt->gpu_route_plan_gate || opt->compact_moe_decode_gate) &&
           (!opt->nccl_reduce_scatter_compose_gate ||
            !opt->decode_cudagraph_gate) &&
           (!opt->tp_hc_current_input_nccl_allgather_gate ||
            opt->tp_hc_current_input_gate) &&
           (!opt->tp_hc_current_allreduce_gate ||
            opt->tp_hc_current_input_gate) &&
           !(opt->model_router_routes && opt->compact_route_compose &&
             !opt->compact_moe_decode_gate) &&
           !(opt->dense_hmma_compose && opt->dense_f16_cublas_compose) &&
           (!opt->dense_f16_cache_compose || opt->dense_f16_cublas_compose) &&
           (!opt->true_ds4_attention_residency_gate ||
            (opt->share_dense_ops && opt->dense_f16_cache_compose &&
             opt->dense_f16_cublas_compose)) &&
           (!opt->true_ds4_attention_projection_gate ||
            opt->true_ds4_attention_residency_gate) &&
           (!opt->true_ds4_attention_projection_direct_input_fill_gate ||
            opt->true_ds4_attention_projection_gate) &&
           (!opt->true_ds4_attention_projection_rank_local_input_gate ||
            opt->true_ds4_attention_projection_gate) &&
           (!opt->true_ds4_attention_projection_rank_major_input_gate ||
            (opt->true_ds4_attention_projection_gate &&
             opt->tp_hc_current_input_nccl_allgather_gate)) &&
           (!opt->true_ds4_attention_state_gate ||
            opt->true_ds4_attention_projection_gate) &&
           (!opt->true_ds4_attention_rope_gate ||
            opt->true_ds4_attention_state_gate) &&
           (!opt->true_ds4_attention_saturation_audit_gate ||
            opt->true_ds4_attention_rope_gate) &&
           (!opt->true_ds4_attention_kv_norm_reference_gate ||
            opt->true_ds4_attention_projection_gate) &&
           (!opt->true_ds4_attention_raw_read_gate ||
            opt->true_ds4_attention_state_gate) &&
           (!opt->true_ds4_attention_raw_window_gate ||
            opt->true_ds4_attention_raw_read_gate) &&
           (!opt->true_ds4_attention_typed_kv_raw_gate ||
            opt->true_ds4_attention_state_gate) &&
           (!opt->true_ds4_attention_typed_kv_compressed_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_attention_typed_kv_indexer_gate ||
            opt->true_ds4_indexer_attention_gate) &&
           (!opt->true_ds4_attention_typed_kv_history_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_attention_typed_kv_skip_current_load_gate ||
            (opt->true_ds4_attention_typed_kv_raw_gate ||
             opt->true_ds4_attention_typed_kv_compressed_gate ||
             opt->true_ds4_attention_typed_kv_indexer_gate)) &&
           (!opt->true_ds4_attention_typed_kv_skip_raw_store_gate ||
            opt->true_ds4_attention_typed_kv_raw_gate) &&
           (!opt->true_ds4_attention_typed_kv_skip_compressed_store_gate ||
            opt->true_ds4_attention_typed_kv_compressed_gate) &&
           (!opt->true_ds4_attention_typed_kv_skip_indexer_store_gate ||
            opt->true_ds4_attention_typed_kv_indexer_gate) &&
           (!opt->true_ds4_attention_typed_kv_quiet_gate ||
            (opt->true_ds4_attention_typed_kv_raw_gate ||
             opt->true_ds4_attention_typed_kv_compressed_gate ||
             opt->true_ds4_attention_typed_kv_indexer_gate ||
             opt->true_ds4_attention_typed_kv_history_gate)) &&
           (!opt->true_ds4_attention_typed_kv_batch_rows_gate ||
            (opt->true_ds4_attention_typed_kv_raw_gate ||
             opt->true_ds4_attention_typed_kv_compressed_gate ||
             opt->true_ds4_attention_typed_kv_indexer_gate ||
             opt->true_ds4_attention_typed_kv_history_gate)) &&
           (!opt->true_ds4_attention_typed_kv_stream_sync_gate ||
            (opt->true_ds4_attention_typed_kv_raw_gate ||
             opt->true_ds4_attention_typed_kv_compressed_gate ||
             opt->true_ds4_attention_typed_kv_indexer_gate ||
             opt->true_ds4_attention_typed_kv_history_gate)) &&
           (!opt->true_ds4_attention_output_gate ||
            opt->true_ds4_attention_raw_window_gate) &&
           (!opt->true_ds4_attention_output_nccl_allgather_gate ||
            opt->true_ds4_attention_output_gate) &&
           (!opt->true_ds4_post_attention_ffn_input_gate ||
            (opt->true_ds4_attention_output_gate && opt->true_shared_ffn_gate &&
             opt->model_router_routes && opt->routed_ffn_norm_input_gate)) &&
           (!(opt->routed_ffn_rank_major_input_gate ||
              opt->routed_ffn_rank_major_shared_input_gate ||
              opt->routed_ffn_rank_major_route_input_gate ||
              opt->routed_ffn_rank_major_input_parity_gate) ||
            (opt->true_ds4_post_attention_ffn_input_gate &&
             opt->tp_hc_current_input_nccl_allgather_gate)) &&
           (!opt->model_router_rank_major_logits_gate ||
            opt->routed_ffn_rank_major_input_gate) &&
           !(opt->model_router_rank_major_logits_gate &&
             opt->model_router_allreduce_logits_gate) &&
           (!opt->true_ds4_semantic_skip_stats_gate ||
            (opt->true_ds4_attention_output_gate ||
             opt->true_ds4_post_attention_ffn_input_gate)) &&
           (!opt->true_ds4_compressed_kv_gate ||
            opt->true_ds4_attention_projection_gate) &&
           (!opt->true_ds4_indexer_attention_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_direct_input_fill_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_dense_event_wait_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_skip_dense_stats_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_fused_attn_input_fill_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_fused_input_fill_gate ||
            opt->true_ds4_indexer_attention_gate) &&
           (!opt->true_ds4_compressed_kv_fused_rope_round_gate ||
            (opt->true_ds4_compressed_kv_gate &&
             opt->true_ds4_attention_rope_gate)) &&
           (!opt->true_ds4_compressed_kv_fused_pool_norm_gate ||
            opt->true_ds4_compressed_kv_gate) &&
           (!opt->true_ds4_compressed_kv_fused_pool_norm_rope_round_gate ||
            (opt->true_ds4_compressed_kv_gate &&
             opt->true_ds4_attention_rope_gate)) &&
           (!opt->true_ds4_compressed_reference_diff_gate ||
            opt->true_ds4_indexer_attention_gate) &&
           !(opt->dense_compute_tensor &&
             (opt->dense_compute_all_f8 || opt->dense_compute_all_bf16));
}
