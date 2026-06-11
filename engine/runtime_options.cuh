/* Sprint 597 Phase 2: EP sub-stage profiler gate. Plumbed from the launcher
 * via the DS4_V100_TP_EP_EP_STAGE_PROFILE environment variable; default off.
 * Flag-off leaves the promoted path byte-identical (every profiler entry
 * point returns immediately). */
static inline bool ds4_ep_stage_profile_env_default() {
    const char *v = std::getenv("DS4_V100_TP_EP_EP_STAGE_PROFILE");
    if (!v || !*v) return false;
    return !(v[0] == '0' && v[1] == '\0');
}

/* Sprint 598 B2-C: EP-return transport selector for the promoted
 * full-capture graph branch. DS4_V100_TP_EP_EP_RETURN_TRANSPORT=copy|nccl;
 * default copy (the s597 per-pair copy_f32 path, byte-identical). nccl
 * captures the grouped per-source NCCL broadcast return
 * (broadcast_ep_return_slices) inside the decode graph instead. */
static inline bool ds4_ep_return_transport_env_nccl() {
    const char *v = std::getenv("DS4_V100_TP_EP_EP_RETURN_TRANSPORT");
    return v && std::strcmp(v, "nccl") == 0;
}

struct Options {
    const char *lib_path = "./build/turbomind-v100/libggml-turbomind.so";
    const char *pack_dir = nullptr;
    const char *contract_path = nullptr;
    const char *tm_index_path = nullptr;
    /* MTP (layer 43) dedicated weight source -- decoupled from the main pack
     * so the appliance loads 0-42 from --pack-dir and the MTP block from its
     * own EP-format pack (mxfp4/turbomind, fused gate_up, EP-split 32/rank). */
    const char *mtp_pack_dir = nullptr;
    const char *mtp_tm_index_path = nullptr;
    const char *mtp_contract_path = nullptr;
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
    bool routed_ffn_rank_major_input_gate = true;
    bool routed_ffn_rank_major_shared_input_gate = false;
    bool routed_ffn_rank_major_route_input_gate = false;
    bool routed_ffn_rank_major_input_parity_gate = false;
    bool post_attention_route_reuse_audit_gate = false;
    bool post_attention_fixed_capacity_route_plan_gate = true;
    bool post_attention_slot_major_ffn_norm_gate = false;
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
    uint32_t true_ds4_attention_raw_valid_rows = 1;
    uint64_t vram_min_free_mib = 0;
    uint64_t nccl_min_free_mib = 0;
    /* Sprint 597 Phase 2: EP sub-stage profiler (default off). */
    bool ep_stage_profile = ds4_ep_stage_profile_env_default();
    /* Sprint 598 B2-C: EP-return transport (default copy = s597 path). */
    bool ep_return_nccl = ds4_ep_return_transport_env_nccl();
};
