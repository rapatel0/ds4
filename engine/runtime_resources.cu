int open_compose_nccl(const Options &opt, RankState ranks[kGpus]);
void close_compose_nccl(RankState ranks[kGpus]);

int open_shared_rank_buffers(const Options &opt, SharedRankBuffers *shared) {
    shared->core_bytes = 0;
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = shared->ranks[p];
        r.rank = p;
        r.device = opt.devices[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaStreamCreate(&r.stream));
        CHECK_CUDA(cudaStreamCreate(&r.dense_stream));
        CHECK_CUDA(cudaStreamCreate(&r.copy_stream));
        for (int q = 0; q < kGpus; ++q) {
            CHECK_CUDA(cudaStreamCreate(&r.copy_streams[q]));
            CHECK_CUDA(cudaEventCreateWithFlags(&r.copy_done[q], cudaEventDisableTiming));
        }
        CHECK_CUDA(cudaEventCreateWithFlags(&r.stream_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&r.dense_done, cudaEventDisableTiming));
        for (int e = 0; e < kGraphOrderEventSlots; ++e) {
            CHECK_CUDA(cudaEventCreateWithFlags(&r.graph_stream_done[e],
                                                cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&r.graph_dense_done[e],
                                                cudaEventDisableTiming));
        }
        CHECK_CUDA(cudaEventCreateWithFlags(&r.dense_wait, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreate(&r.start));
        CHECK_CUDA(cudaEventCreate(&r.mid));
        CHECK_CUDA(cudaEventCreate(&r.stop));
        r.route_compact_plan_ints = compact_route_plan_ints(opt);
        CHECK_CUDA(cudaMalloc(&r.d_route_compact_plan,
                              r.route_compact_plan_ints * sizeof(int)));
        bind_compact_route_plan(&r, opt);
        CHECK_CUDA(cudaMalloc(&r.d_router_selected_plan,
                              (size_t)opt.slots * (size_t)opt.top_k * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&r.d_router_weights_plan,
                              (size_t)opt.slots * (size_t)opt.top_k * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&r.d_route_offsets_all,
                              (size_t)kGpus * (size_t)(kLocalExperts + 1) *
                                  sizeof(int)));
        CHECK_CUDA(cudaMalloc(&r.d_route_totals,
                              (size_t)kGpus * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&r.d_post_attn_route_audit,
                              4u * sizeof(unsigned long long)));
        CHECK_CUDA(cudaMemset(r.d_post_attn_route_audit, 0,
                              4u * sizeof(unsigned long long)));
        std::vector<int> compact_plan(r.route_compact_plan_ints, -1);
        const size_t compact_indices = (size_t)opt.slots * (size_t)opt.top_k;
        const size_t compact_counts = (size_t)opt.slots;
        for (int src = 0; src < kGpus; ++src) {
            std::vector<int> route_index_by_slot;
            build_route_index_by_slot_for_rank(src, opt.slots, opt.top_k,
                                               &route_index_by_slot);
            CHECK_CUDA(cudaMalloc(&r.d_route_index_by_slot[src],
                                  route_index_by_slot.size() * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(r.d_route_index_by_slot[src],
                                  route_index_by_slot.data(),
                                  route_index_by_slot.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            std::vector<int> route_indices_by_slot;
            std::vector<int> route_count_by_slot;
            build_route_indices_by_slot_for_rank(src, opt.slots, opt.top_k,
                                                 &route_indices_by_slot,
                                                 &route_count_by_slot);
            std::copy(route_indices_by_slot.begin(), route_indices_by_slot.end(),
                      compact_plan.begin() + (size_t)src * compact_indices);
            std::copy(route_count_by_slot.begin(), route_count_by_slot.end(),
                      compact_plan.begin() + (size_t)kGpus * compact_indices +
                          (size_t)src * compact_counts);
            shared->core_bytes += route_index_by_slot.size() * sizeof(int);
        }
        CHECK_CUDA(cudaMemcpy(r.d_route_compact_plan, compact_plan.data(),
                              compact_plan.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        shared->core_bytes += compact_plan.size() * sizeof(int);
        shared->core_bytes += 4u * sizeof(unsigned long long);

        std::vector<int> offsets;
        std::vector<int> route_slots;
        std::vector<float> route_weights;
        build_offsets_for_rank(p, opt.slots, opt.top_k, &offsets, &route_slots,
                               &route_weights, &r.routes, &r.active_experts,
                               &r.max_routes_per_expert);

        r.route_capacity = opt.slots * opt.top_k;
        const size_t route_capacity_elems = (size_t)r.route_capacity * kHidden;
        CHECK_CUDA(cudaMalloc(&r.d_offsets, offsets.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(r.d_offsets, offsets.data(), offsets.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_route_slots,
                              (size_t)r.route_capacity * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(r.d_route_slots, route_slots.data(),
                              route_slots.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_route_weights,
                              (size_t)r.route_capacity * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(r.d_route_weights, route_weights.data(),
                              route_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_route_inv_scale,
                              (size_t)r.route_capacity * sizeof(float)));
        std::vector<float> route_inv_scale((size_t)r.route_capacity, 1.0f);
        CHECK_CUDA(cudaMemcpy(r.d_route_inv_scale, route_inv_scale.data(),
                              route_inv_scale.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&r.d_a, route_capacity_elems * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&r.d_gate_up,
                              (size_t)r.route_capacity * kFusedN * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&r.d_gated,
                              (size_t)r.route_capacity * kMid * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&r.d_down, route_capacity_elems * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&r.d_decode_position, sizeof(uint64_t)));
        CHECK_CUDA(cudaMemset(r.d_decode_position, 0, sizeof(uint64_t)));
        if (opt.model_router_rank_major_logits_gate ||
            opt.model_router_allreduce_logits_gate) {
            if (opt.model_router_rank_major_logits_gate) {
                CHECK_CUDA(cudaMalloc(&r.d_rank_major_norm_scale,
                                      (size_t)opt.slots * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_router_logits_shard,
                                      (size_t)opt.slots * kLocalExperts *
                                          sizeof(float)));
            }
            CHECK_CUDA(cudaMalloc(&r.d_router_logits_rank_major,
                                  (size_t)opt.slots * kGlobalExperts * sizeof(float)));
        }

        std::mt19937 rng(0xE2350000u + (uint32_t)p * 97u);
        std::uniform_real_distribution<float> dist(-0.003f, 0.003f);
        std::vector<__half> h_a(route_capacity_elems);
        for (__half &v : h_a) v = __float2half(dist(rng));
        CHECK_CUDA(cudaMemcpy(r.d_a, h_a.data(),
                              route_capacity_elems * sizeof(__half),
                              cudaMemcpyHostToDevice));

        shared->core_bytes += offsets.size() * sizeof(int);
        shared->core_bytes += (size_t)r.route_capacity * sizeof(int);
        shared->core_bytes += (size_t)r.route_capacity * sizeof(float);
        shared->core_bytes += (size_t)r.route_capacity * sizeof(float);
        shared->core_bytes += route_capacity_elems * sizeof(__half);
        shared->core_bytes += (size_t)r.route_capacity * kFusedN * sizeof(__half);
        shared->core_bytes += (size_t)r.route_capacity * kMid * sizeof(__half);
        shared->core_bytes += route_capacity_elems * sizeof(__half);
        shared->core_bytes += sizeof(uint64_t);
        if (opt.model_router_rank_major_logits_gate) {
            shared->core_bytes += (size_t)opt.slots * sizeof(float);
            shared->core_bytes += (size_t)opt.slots * kLocalExperts * sizeof(float);
        }
        if (opt.model_router_rank_major_logits_gate ||
            opt.model_router_allreduce_logits_gate) {
            shared->core_bytes += (size_t)opt.slots * kGlobalExperts * sizeof(float);
        }
    }
    if (!opt.defer_nccl_init_gate && open_compose_nccl(opt, shared->ranks) != 0) {
        return 1;
    }
    shared->initialized = true;
    return 0;
}

int open_compose_nccl(const Options &opt, RankState ranks[kGpus]) {
    const bool need_compose =
        opt.nccl_reduce_scatter_compose_gate &&
        !opt.compact_route_compose && !opt.ep_return_fp16;
    const bool need_attention_output =
        opt.true_ds4_attention_output_nccl_allgather_gate;
    const bool need_hc_current =
        opt.tp_hc_current_input_nccl_allgather_gate ||
        opt.tp_hc_current_allreduce_gate ||
        opt.model_router_allreduce_logits_gate;
    const bool need_full_current_broadcast =
        opt.tp_hc_current_input_gate ||
        opt.true_shared_ffn_gate ||
        opt.true_ds4_attention_projection_gate ||
        opt.true_ds4_compressed_kv_gate ||
        opt.true_ds4_post_attention_ffn_input_gate;
    const bool need_transport_sweep =
        opt.model_router_routes ||
        opt.compact_moe_decode_gate ||
        opt.true_ds4_attention_raw_read_gate ||
        opt.true_ds4_attention_raw_window_gate ||
        opt.true_ds4_attention_typed_kv_compressed_gate;
    if (!need_compose && !need_attention_output && !need_hc_current &&
        !need_full_current_broadcast && !need_transport_sweep) {
        return 0;
    }
    int devices[kGpus] = {};
    ncclComm_t comms[kGpus] = {};
    for (int p = 0; p < kGpus; ++p) devices[p] = ranks[p].device;
    CHECK_NCCL(ncclCommInitAll(comms, kGpus, devices));
    for (int p = 0; p < kGpus; ++p) {
        ranks[p].compose_nccl = comms[p];
        ranks[p].compose_nccl_initialized = true;
    }
    std::printf("tp_ep_nccl\tdevices\t%d\tcompose_reduce_scatter\t%d\t"
                "attention_output_allgather\t%d\t"
                "hc_current_nccl\t%d\tfull_current_broadcast\t%d\t"
                "transport_sweep\t%d\tPASS\n",
                kGpus, need_compose ? 1 : 0, need_attention_output ? 1 : 0,
                need_hc_current ? 1 : 0,
                need_full_current_broadcast ? 1 : 0,
                need_transport_sweep ? 1 : 0);
    return 0;
}

int nccl_broadcast_f32_from_device0_to_current_full(
    const Options &opt,
    RankState ranks[kGpus],
    const float *src_device0,
    uint64_t elems,
    const char *label) {
    if (!src_device0 || elems == 0) return 1;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.compose_nccl_initialized || !r.compose_nccl ||
            !r.d_current_full) {
            std::fprintf(stderr,
                         "tp_ep_full_current_nccl_broadcast_missing\tlabel\t%s\t"
                         "rank\t%d\tcompose\t%d\tbuffer\t%d\n",
                         label ? label : "-", rank,
                         (r.compose_nccl_initialized && r.compose_nccl) ? 1 : 0,
                         r.d_current_full ? 1 : 0);
            return 2;
        }
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        const float *send = rank == 0 ? src_device0 : r.d_current_full;
        CHECK_NCCL(ncclBroadcast(send, r.d_current_full, (size_t)elems,
                                 ncclFloat, 0, r.compose_nccl, r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    return 0;
}

void close_compose_nccl(RankState ranks[kGpus]) {
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        if (!r.compose_nccl_initialized || !r.compose_nccl) continue;
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclCommDestroy(r.compose_nccl));
        r.compose_nccl = nullptr;
        r.compose_nccl_initialized = false;
    }
}

void close_tp_cuda_graph_layer_exec(TpCudaGraphLayerExec *entry) {
    if (!entry) return;
    if (entry->root_device >= 0) {
        CHECK_CUDA(cudaSetDevice(entry->root_device));
    }
    if (entry->exec) CHECK_CUDA(cudaGraphExecDestroy(entry->exec));
    if (entry->graph) CHECK_CUDA(cudaGraphDestroy(entry->graph));
    *entry = TpCudaGraphLayerExec{};
}

void close_tp_cuda_graph_cache(TpCudaGraphCache *cache) {
    if (!cache) return;
    for (int layer = 0; layer < 43; ++layer) {
        close_tp_cuda_graph_layer_exec(&cache->layers[layer]);
    }
}

void close_shared_rank_buffers(SharedRankBuffers *shared) {
    if (!shared || !shared->initialized) return;
    close_tp_cuda_graph_cache(&shared->graph_cache);
    close_compose_nccl(shared->ranks);
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = shared->ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        free_packed(r.gated);
        free_packed(r.down);
        if (r.d_offsets) CHECK_CUDA(cudaFree(r.d_offsets));
        if (r.d_route_slots) CHECK_CUDA(cudaFree(r.d_route_slots));
        if (r.d_route_weights) CHECK_CUDA(cudaFree(r.d_route_weights));
        if (r.d_route_inv_scale) CHECK_CUDA(cudaFree(r.d_route_inv_scale));
        if (r.d_a) CHECK_CUDA(cudaFree(r.d_a));
        if (r.d_gate_up) CHECK_CUDA(cudaFree(r.d_gate_up));
        if (r.d_gated) CHECK_CUDA(cudaFree(r.d_gated));
        if (r.d_down) CHECK_CUDA(cudaFree(r.d_down));
        if (r.d_ep_contrib_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_all));
        if (r.d_ep_contrib_half_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_half_all));
        if (r.d_ep_contrib_bcast_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_bcast_all));
        if (r.d_ep_contrib_half_bcast_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_half_bcast_all));
        if (r.d_route_totals) CHECK_CUDA(cudaFree(r.d_route_totals));
        if (r.d_route_offsets_all) CHECK_CUDA(cudaFree(r.d_route_offsets_all));
        if (r.d_router_weights_plan) CHECK_CUDA(cudaFree(r.d_router_weights_plan));
        if (r.d_router_selected_plan) CHECK_CUDA(cudaFree(r.d_router_selected_plan));
        const bool has_route_compact_plan = r.d_route_compact_plan != nullptr;
        if (r.d_route_compact_plan) CHECK_CUDA(cudaFree(r.d_route_compact_plan));
        for (int src = 0; src < kGpus; ++src) {
            if (r.d_route_index_by_slot[src]) CHECK_CUDA(cudaFree(r.d_route_index_by_slot[src]));
            if (!has_route_compact_plan && r.d_route_indices_by_slot[src]) {
                CHECK_CUDA(cudaFree(r.d_route_indices_by_slot[src]));
            }
            if (!has_route_compact_plan && r.d_route_count_by_slot[src]) {
                CHECK_CUDA(cudaFree(r.d_route_count_by_slot[src]));
            }
            if (r.d_ep_remote[src]) CHECK_CUDA(cudaFree(r.d_ep_remote[src]));
            if (r.d_ep_remote_half[src]) CHECK_CUDA(cudaFree(r.d_ep_remote_half[src]));
        }
        if (r.d_ep_sum) CHECK_CUDA(cudaFree(r.d_ep_sum));
        if (r.d_next_hidden) CHECK_CUDA(cudaFree(r.d_next_hidden));
        if (r.d_current_shard) CHECK_CUDA(cudaFree(r.d_current_shard));
        if (r.d_current_full) CHECK_CUDA(cudaFree(r.d_current_full));
        if (r.d_current_full_normed) CHECK_CUDA(cudaFree(r.d_current_full_normed));
        if (r.d_current_full_rank_major) CHECK_CUDA(cudaFree(r.d_current_full_rank_major));
        if (r.d_post_attn_full_rank_major) CHECK_CUDA(cudaFree(r.d_post_attn_full_rank_major));
        if (r.d_rank_major_norm_scale) CHECK_CUDA(cudaFree(r.d_rank_major_norm_scale));
        if (r.d_router_logits_shard) CHECK_CUDA(cudaFree(r.d_router_logits_shard));
        if (r.d_router_logits_rank_major) CHECK_CUDA(cudaFree(r.d_router_logits_rank_major));
        if (r.d_half_diff_counts) CHECK_CUDA(cudaFree(r.d_half_diff_counts));
        if (r.d_half_diff_max_bits) CHECK_CUDA(cudaFree(r.d_half_diff_max_bits));
        if (r.d_half_diff_first) CHECK_CUDA(cudaFree(r.d_half_diff_first));
        if (r.d_post_attn_route_audit) CHECK_CUDA(cudaFree(r.d_post_attn_route_audit));
        if (r.d_final_hc_shard) CHECK_CUDA(cudaFree(r.d_final_hc_shard));
        if (r.d_hc_scratch_shard) CHECK_CUDA(cudaFree(r.d_hc_scratch_shard));
        if (r.d_hc_split) CHECK_CUDA(cudaFree(r.d_hc_split));
        if (r.d_decode_position) CHECK_CUDA(cudaFree(r.d_decode_position));
        if (r.d_hc_reduce_max) CHECK_CUDA(cudaFree(r.d_hc_reduce_max));
        if (r.d_hc_reduce_sumsq) CHECK_CUDA(cudaFree(r.d_hc_reduce_sumsq));
        if (r.d_hc_reduce_mix) CHECK_CUDA(cudaFree(r.d_hc_reduce_mix));
        for (int layer = 0; layer < 43; ++layer) {
            if (r.d_attn_raw_swa_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_attn_raw_swa_layers[layer]));
            }
        }
        if (r.d_attn_kv_full) CHECK_CUDA(cudaFree(r.d_attn_kv_full));
        if (r.d_attn_heads) CHECK_CUDA(cudaFree(r.d_attn_heads));
        if (r.d_attn_output_a_full) CHECK_CUDA(cudaFree(r.d_attn_output_a_full));
        if (r.d_post_attn_shard) CHECK_CUDA(cudaFree(r.d_post_attn_shard));
        if (r.d_attn_sinks) CHECK_CUDA(cudaFree(r.d_attn_sinks));
        if (r.d_indexer_topk) CHECK_CUDA(cudaFree(r.d_indexer_topk));
        if (r.d_indexer_scores) CHECK_CUDA(cudaFree(r.d_indexer_scores));
        for (int layer = 0; layer < 43; ++layer) {
            if (r.d_index_comp_rows_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_index_comp_rows_layers[layer]));
            }
            if (r.d_index_comp_state_score_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_index_comp_state_score_layers[layer]));
            }
            if (r.d_index_comp_state_kv_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_index_comp_state_kv_layers[layer]));
            }
        }
        if (r.d_index_comp_score_cur) CHECK_CUDA(cudaFree(r.d_index_comp_score_cur));
        if (r.d_index_comp_kv_cur) CHECK_CUDA(cudaFree(r.d_index_comp_kv_cur));
        for (int layer = 0; layer < 43; ++layer) {
            if (r.d_attn_comp_rows_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_attn_comp_rows_layers[layer]));
            }
            if (r.d_attn_comp_state_score_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_attn_comp_state_score_layers[layer]));
            }
            if (r.d_attn_comp_state_kv_layers[layer]) {
                CHECK_CUDA(cudaFree(r.d_attn_comp_state_kv_layers[layer]));
            }
        }
        if (r.d_attn_comp_score_cur) CHECK_CUDA(cudaFree(r.d_attn_comp_score_cur));
        if (r.d_attn_comp_kv_cur) CHECK_CUDA(cudaFree(r.d_attn_comp_kv_cur));
        if (r.dense_wait) CHECK_CUDA(cudaEventDestroy(r.dense_wait));
        if (r.start) CHECK_CUDA(cudaEventDestroy(r.start));
        if (r.mid) CHECK_CUDA(cudaEventDestroy(r.mid));
        if (r.stop) CHECK_CUDA(cudaEventDestroy(r.stop));
        for (int q = 0; q < kGpus; ++q) {
            if (r.copy_done[q]) CHECK_CUDA(cudaEventDestroy(r.copy_done[q]));
            if (r.copy_streams[q]) CHECK_CUDA(cudaStreamDestroy(r.copy_streams[q]));
        }
        for (int e = 0; e < kGraphOrderEventSlots; ++e) {
            if (r.graph_stream_done[e]) {
                CHECK_CUDA(cudaEventDestroy(r.graph_stream_done[e]));
            }
            if (r.graph_dense_done[e]) {
                CHECK_CUDA(cudaEventDestroy(r.graph_dense_done[e]));
            }
        }
        if (r.dense_done) CHECK_CUDA(cudaEventDestroy(r.dense_done));
        if (r.stream_done) CHECK_CUDA(cudaEventDestroy(r.stream_done));
        if (r.copy_stream) CHECK_CUDA(cudaStreamDestroy(r.copy_stream));
        if (r.dense_stream) CHECK_CUDA(cudaStreamDestroy(r.dense_stream));
        if (r.stream) CHECK_CUDA(cudaStreamDestroy(r.stream));
        r = RankState{};
    }
    *shared = SharedRankBuffers{};
}

void fill_tp_runtime_config(const Options &opt, ds4_tp_runtime_config *cfg) {
    ds4_tp_runtime_default_config(cfg);
    cfg->slots = (uint32_t)opt.slots;
    cfg->ctx = 262144;
    cfg->kv_dtype = opt.fp8_e5m2_kv_gate
        ? DS4_V100_TP_KV_F8_E5M2_B128
        : DS4_V100_TP_KV_F8_E4M3_B128;
    cfg->scratch_bytes = opt.tp_runtime_scratch_mib * 1024ull * 1024ull;
    cfg->allocate_comp_state = opt.tp_runtime_skip_unused_comp_state ? 0u : 1u;
    for (int i = 0; i < kGpus; ++i) cfg->devices[i] = opt.devices[i];
}

int open_shared_tp_runtime(const Options &opt, SharedTpRuntime *shared) {
    ds4_tp_runtime_config cfg;
    fill_tp_runtime_config(opt, &cfg);
    char err[512] = {0};
    if (ds4_tp_runtime_open(&shared->rt, &cfg, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "tp_runtime_open_failed\t%s\n", err);
        *shared = SharedTpRuntime{};
        return 1;
    }
    ds4_tp_runtime_get_report(shared->rt, &shared->report);
    shared->initialized = true;
    return 0;
}

void close_shared_tp_runtime(SharedTpRuntime *shared) {
    if (!shared || !shared->rt) return;
    ds4_tp_runtime_close(shared->rt);
    *shared = SharedTpRuntime{};
}

int ensure_compose_buffers(const Options &opt, RankState ranks[kGpus]) {
    const uint64_t shard_elems = (uint64_t)opt.slots * (kHidden / kGpus);
    const uint64_t compact_segment_routes =
        opt.compact_moe_decode_gate ? (uint64_t)opt.slots * (uint64_t)opt.top_k
                                    : (uint64_t)opt.slots;
    const uint64_t shard_bytes = shard_elems * sizeof(float);
    const uint64_t remote_float_elems =
        opt.compact_route_compose && !opt.ep_return_fp16
            ? compact_segment_routes * (uint64_t)(kHidden / kGpus)
            : shard_elems;
    const uint64_t remote_float_bytes = remote_float_elems * sizeof(float);
    const uint64_t all_contrib_elems =
        (uint64_t)kGpus * compact_segment_routes * (uint64_t)(kHidden / kGpus);
    const uint64_t all_contrib_bytes = all_contrib_elems * sizeof(float);
    const int layer = opt.layer;
    if ((opt.true_ds4_attention_state_gate || opt.true_ds4_compressed_kv_gate ||
         opt.true_ds4_indexer_attention_gate) &&
        (layer < 0 || layer >= 43)) {
        return 20;
    }
    const int ratio = (layer >= 0 && layer < 43) ? ds4_layer_ratio(layer) : 0;
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_ep_contrib_all) CHECK_CUDA(cudaMalloc(&r.d_ep_contrib_all,
                                                        (size_t)all_contrib_bytes));
        if (!r.d_ep_contrib_bcast_all) {
            CHECK_CUDA(cudaMalloc(&r.d_ep_contrib_bcast_all,
                                  (size_t)all_contrib_bytes));
        }
        if (opt.ep_return_fp16 && !r.d_ep_contrib_half_all) {
            CHECK_CUDA(cudaMalloc(&r.d_ep_contrib_half_all,
                                  (size_t)(all_contrib_elems * sizeof(__half))));
        }
        if (opt.ep_return_fp16 && !r.d_ep_contrib_half_bcast_all) {
            CHECK_CUDA(cudaMalloc(&r.d_ep_contrib_half_bcast_all,
                                  (size_t)(all_contrib_elems * sizeof(__half))));
        }
        if (!r.d_ep_sum) CHECK_CUDA(cudaMalloc(&r.d_ep_sum, (size_t)shard_bytes));
        if (!r.d_next_hidden) CHECK_CUDA(cudaMalloc(&r.d_next_hidden, (size_t)shard_bytes));
        if (opt.tp_hc_current_input_gate && !r.d_current_shard) {
            CHECK_CUDA(cudaMalloc(&r.d_current_shard, (size_t)shard_bytes));
        }
        if (opt.tp_hc_current_input_gate && !r.d_current_full) {
            CHECK_CUDA(cudaMalloc(&r.d_current_full,
                                  (size_t)opt.slots * kHidden * sizeof(float)));
        }
        if (opt.true_ds4_attention_projection_rank_local_input_gate &&
            !r.d_current_full_normed) {
            CHECK_CUDA(cudaMalloc(&r.d_current_full_normed,
                                  (size_t)opt.slots * kHidden * sizeof(float)));
        }
        if (opt.tp_hc_current_input_nccl_allgather_gate &&
            !r.d_current_full_rank_major) {
            CHECK_CUDA(cudaMalloc(&r.d_current_full_rank_major,
                                  (size_t)opt.slots * kHidden * sizeof(float)));
        }
        if ((opt.routed_ffn_rank_major_input_gate ||
             opt.routed_ffn_rank_major_shared_input_gate ||
             opt.routed_ffn_rank_major_route_input_gate ||
             opt.routed_ffn_rank_major_input_parity_gate) &&
            !r.d_post_attn_full_rank_major) {
            CHECK_CUDA(cudaMalloc(&r.d_post_attn_full_rank_major,
                                  (size_t)opt.slots * kHidden * sizeof(float)));
        }
        if ((opt.routed_ffn_rank_major_input_parity_gate ||
             opt.true_ds4_attention_projection_input_parity_gate) &&
            !r.d_half_diff_counts) {
            CHECK_CUDA(cudaMalloc(&r.d_half_diff_counts,
                                  2 * sizeof(unsigned long long)));
            CHECK_CUDA(cudaMalloc(&r.d_half_diff_max_bits,
                                  sizeof(unsigned int)));
            CHECK_CUDA(cudaMalloc(&r.d_half_diff_first, sizeof(int)));
        }
        if (opt.final_hc_carry_gate && !r.d_final_hc_shard) {
            CHECK_CUDA(cudaMalloc(&r.d_final_hc_shard, (size_t)(4ull * shard_bytes)));
        }
        if (opt.tp_hc_final_expand_gate && !r.d_hc_scratch_shard) {
            CHECK_CUDA(cudaMalloc(&r.d_hc_scratch_shard, (size_t)(4ull * shard_bytes)));
        }
        if (opt.tp_hc_final_expand_gate && !r.d_hc_split) {
            CHECK_CUDA(cudaMalloc(&r.d_hc_split, (size_t)opt.slots * kHcMix * sizeof(float)));
        }
        if (opt.tp_hc_current_allreduce_gate ||
            opt.model_router_allreduce_logits_gate) {
            if (!r.d_hc_reduce_max) {
                CHECK_CUDA(cudaMalloc(&r.d_hc_reduce_max,
                                      (size_t)opt.slots * sizeof(float)));
            }
            if (!r.d_hc_reduce_sumsq) {
                CHECK_CUDA(cudaMalloc(&r.d_hc_reduce_sumsq,
                                      (size_t)opt.slots * sizeof(float)));
            }
        }
        if (opt.tp_hc_current_allreduce_gate) {
            if (!r.d_hc_reduce_mix) {
                CHECK_CUDA(cudaMalloc(&r.d_hc_reduce_mix,
                                      (size_t)opt.slots * kHcMix * sizeof(float)));
            }
        }
        if (opt.true_ds4_attention_state_gate && !r.d_attn_kv_full) {
            CHECK_CUDA(cudaMalloc(&r.d_attn_kv_full,
                                  (size_t)opt.slots * kHeadDim * sizeof(float)));
        }
        if (opt.true_ds4_attention_state_gate) {
            if (!r.d_attn_raw_swa_layers[layer]) {
                CHECK_CUDA(cudaMalloc(&r.d_attn_raw_swa_layers[layer],
                                      (size_t)opt.slots * kRawSwaRows *
                                          (size_t)kHeadDim * sizeof(float)));
                CHECK_CUDA(cudaMemsetAsync(r.d_attn_raw_swa_layers[layer], 0,
                                           (size_t)opt.slots * kRawSwaRows *
                                               (size_t)kHeadDim * sizeof(float),
                                           r.stream));
            }
            r.d_attn_raw_swa = r.d_attn_raw_swa_layers[layer];
        }
        if (opt.true_ds4_attention_raw_read_gate && !r.d_attn_sinks) {
            CHECK_CUDA(cudaMalloc(&r.d_attn_sinks,
                                  (size_t)kLocalHeads * sizeof(float)));
        }
        if (opt.true_ds4_attention_raw_read_gate && !r.d_attn_heads) {
            CHECK_CUDA(cudaMalloc(&r.d_attn_heads,
                                  (size_t)opt.slots * kLocalHeads *
                                      (size_t)kHeadDim * sizeof(float)));
        }
        if (opt.true_ds4_attention_output_gate && !r.d_attn_output_a_full) {
            CHECK_CUDA(cudaMalloc(&r.d_attn_output_a_full,
                                  (size_t)opt.slots *
                                      (size_t)kAttentionOutputAFull * sizeof(float)));
        }
        if (opt.true_ds4_post_attention_ffn_input_gate && !r.d_post_attn_shard) {
            CHECK_CUDA(cudaMalloc(&r.d_post_attn_shard, (size_t)shard_bytes));
        }
        if (opt.true_ds4_compressed_kv_gate && ratio != 0) {
            const int comp_state_rows = attn_comp_state_rows_for_ratio(ratio);
            const int comp_state_width = attn_comp_state_width_for_ratio(ratio);
            if (!r.d_attn_comp_kv_cur) {
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_kv_cur,
                                      (size_t)opt.slots * kCompWidthMax * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_score_cur,
                                      (size_t)opt.slots * kCompWidthMax * sizeof(float)));
            }
            if (!r.d_attn_comp_state_kv_layers[layer]) {
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_state_kv_layers[layer],
                                      (size_t)opt.slots * (size_t)comp_state_rows *
                                          (size_t)comp_state_width * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_state_score_layers[layer],
                                      (size_t)opt.slots * (size_t)comp_state_rows *
                                          (size_t)comp_state_width * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_attn_comp_rows_layers[layer],
                                      (size_t)opt.slots * kBoundedCompRows *
                                          (size_t)kHeadDim * sizeof(float)));
                CHECK_CUDA(cudaMemsetAsync(r.d_attn_comp_state_kv_layers[layer], 0,
                                           (size_t)opt.slots * (size_t)comp_state_rows *
                                               (size_t)comp_state_width * sizeof(float),
                                           r.stream));
                CHECK_CUDA(cudaMemsetAsync(r.d_attn_comp_state_score_layers[layer], 0,
                                           (size_t)opt.slots * (size_t)comp_state_rows *
                                               (size_t)comp_state_width * sizeof(float),
                                           r.stream));
                CHECK_CUDA(cudaMemsetAsync(r.d_attn_comp_rows_layers[layer], 0,
                                           (size_t)opt.slots * kBoundedCompRows *
                                               (size_t)kHeadDim * sizeof(float),
                                           r.stream));
            }
            r.d_attn_comp_state_kv = r.d_attn_comp_state_kv_layers[layer];
            r.d_attn_comp_state_score = r.d_attn_comp_state_score_layers[layer];
            r.d_attn_comp_rows = r.d_attn_comp_rows_layers[layer];
        }
        if (opt.true_ds4_indexer_attention_gate && ratio == 4) {
            if (!r.d_index_comp_kv_cur) {
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_kv_cur,
                                      (size_t)opt.slots * kIndexCompWidth * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_score_cur,
                                      (size_t)opt.slots * kIndexCompWidth * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_indexer_scores,
                                      (size_t)opt.slots * kIndexerTopK * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_indexer_topk,
                                      (size_t)opt.slots * kIndexerTopK * sizeof(uint32_t)));
            }
            if (!r.d_index_comp_state_kv_layers[layer]) {
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_state_kv_layers[layer],
                                      (size_t)opt.slots * kIndexCompStateRows *
                                          (size_t)kIndexCompWidth * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_state_score_layers[layer],
                                      (size_t)opt.slots * kIndexCompStateRows *
                                          (size_t)kIndexCompWidth * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&r.d_index_comp_rows_layers[layer],
                                      (size_t)opt.slots * kBoundedCompRows *
                                          (size_t)kIndexerHeadDim * sizeof(float)));
                CHECK_CUDA(cudaMemsetAsync(r.d_index_comp_state_kv_layers[layer], 0,
                                           (size_t)opt.slots * kIndexCompStateRows *
                                               (size_t)kIndexCompWidth * sizeof(float),
                                           r.stream));
                CHECK_CUDA(cudaMemsetAsync(r.d_index_comp_state_score_layers[layer], 0,
                                           (size_t)opt.slots * kIndexCompStateRows *
                                               (size_t)kIndexCompWidth * sizeof(float),
                                           r.stream));
                CHECK_CUDA(cudaMemsetAsync(r.d_index_comp_rows_layers[layer], 0,
                                           (size_t)opt.slots * kBoundedCompRows *
                                               (size_t)kIndexerHeadDim * sizeof(float),
                                           r.stream));
            }
            r.d_index_comp_state_kv = r.d_index_comp_state_kv_layers[layer];
            r.d_index_comp_state_score = r.d_index_comp_state_score_layers[layer];
            r.d_index_comp_rows = r.d_index_comp_rows_layers[layer];
        }
        for (int src = 0; src < kGpus; ++src) {
            if (!r.d_ep_remote[src]) CHECK_CUDA(cudaMalloc(&r.d_ep_remote[src],
                                                           (size_t)remote_float_bytes));
            if (opt.ep_return_fp16 && !r.d_ep_remote_half[src]) {
                CHECK_CUDA(cudaMalloc(&r.d_ep_remote_half[src],
                                      (size_t)(shard_elems * sizeof(__half))));
            }
        }
    }
    return 0;
}
