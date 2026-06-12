int run_resident_layer_decode(const Options &opt,
                              const std::vector<ContractRow> &rows,
                              const LayerStats &layer_stats,
                              RankState ranks[kGpus],
                              const Api &api,
                              ds4_tp_runtime *rt,
                              const LayerExpertCache *layer_expert_cache,
                              const DenseF16Cache *dense_f16_cache,
                              const LayerDenseOps *layer_dense_ops,
                              SharedHcControls *shared_hc_controls,
                              TpCudaGraphLayerExec *persistent_graph,
                              LayerRunSummary *summary) {
    if (!rt || !layer_expert_cache || !dense_f16_cache) return 2;

    char err[512] = {0};
    ds4_tp_dense_kv_result kv_result;
    const int write_indexer = ds4_layer_ratio(opt.layer) == 4 ? 1 : 0;
    /* s601 Phase D: clamp the fixture KV slot into the configured slot
     * count (default 7; unchanged for slots >= 8). */
    const uint32_t kv_fixture_slot =
        opt.kv_slot < (uint32_t)opt.slots ? opt.kv_slot
                                          : (uint32_t)(opt.slots - 1);
    const uint32_t kv_first_slot = opt.tp_kv_all_slots_gate ? 0u : kv_fixture_slot;
    const uint32_t kv_end_slot = opt.tp_kv_all_slots_gate ? (uint32_t)opt.slots : kv_fixture_slot + 1u;
    for (uint32_t slot = kv_first_slot; slot < kv_end_slot; ++slot) {
        if (ds4_tp_runtime_dense_kv_slice(rt, opt.layer, slot, opt.position,
                                               write_indexer, &kv_result, err,
                                               sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_dense_kv_slice_failed\tslot\t%u\t%s\n",
                         slot, err);
            return 3;
        }
        if (kv_result.max_abs != 0.0) return 4;
    }

    for (int p = 0; p < kGpus; ++p) {
        ranks[p].gated = layer_expert_cache->gated[p];
        ranks[p].down = layer_expert_cache->down[p];
    }

    DecodeLoopStats decode_loop;
    const int rc = run_decode_loop(opt, rows, ranks, api, rt, dense_f16_cache,
                                   layer_dense_ops, shared_hc_controls,
                                   persistent_graph,
                                   &decode_loop);

    for (int p = 0; p < kGpus; ++p) {
        ranks[p].gated = PackedExperts{};
        ranks[p].down = PackedExperts{};
    }

    if (summary) {
        summary->layer = opt.layer;
        summary->ratio = ds4_layer_ratio(opt.layer);
        summary->pass = rc == 0 && decode_loop.pass;
        summary->total_rows = layer_stats.total_rows;
        summary->dense_rows = layer_stats.dense_rows;
        summary->control_rows = layer_stats.control_rows;
        summary->expert_rows = layer_stats.expert_rows;
        summary->kv_rows = layer_stats.kv_rows;
        summary->comp_rows = layer_stats.comp_rows;
        summary->decode_ms_per_step = decode_loop.ms_per_step;
        summary->decode_slot_step_tok_s = decode_loop.tok_s;
        summary->decode_ep_ms_per_step = decode_loop.ep_ms_per_step;
        summary->decode_dense_ms_per_step = decode_loop.dense_ms_per_step;
        summary->decode_compose_ms_per_step = decode_loop.compose_ms_per_step;
        summary->decode_compose_reduce_ms_per_step =
            decode_loop.compose_reduce_ms_per_step;
        summary->decode_compose_copy_ms_per_step =
            decode_loop.compose_copy_ms_per_step;
        summary->decode_compose_final_ms_per_step =
            decode_loop.compose_final_ms_per_step;
        summary->decode_hc_current_input_ms_per_step =
            decode_loop.hc_current_input_ms_per_step;
        summary->decode_hc_current_seed_ms_per_step =
            decode_loop.hc_current_seed_ms_per_step;
        summary->decode_hc_current_attn_mix_ms_per_step =
            decode_loop.hc_current_attn_mix_ms_per_step;
        summary->decode_hc_current_split_ms_per_step =
            decode_loop.hc_current_split_ms_per_step;
        summary->decode_hc_current_gather_ms_per_step =
            decode_loop.hc_current_gather_ms_per_step;
        summary->decode_hc_current_ffn_router_ms_per_step =
            decode_loop.hc_current_ffn_router_ms_per_step;
        summary->decode_hc_current_ffn_norm_ms_per_step =
            decode_loop.hc_current_ffn_norm_ms_per_step;
        summary->decode_hc_current_router_select_ms_per_step =
            decode_loop.hc_current_router_select_ms_per_step;
        summary->decode_hc_current_router_d2h_ms_per_step =
            decode_loop.hc_current_router_d2h_ms_per_step;
        summary->decode_hc_current_route_upload_ms_per_step =
            decode_loop.hc_current_route_upload_ms_per_step;
        summary->decode_hc_current_fill_pack_ms_per_step =
            decode_loop.hc_current_fill_pack_ms_per_step;
        summary->decode_pre_ep_hc_current_ms_per_step =
            decode_loop.pre_ep_hc_current_ms_per_step;
        summary->decode_pre_ep_attention_projection_ms_per_step =
            decode_loop.pre_ep_attention_projection_ms_per_step;
        summary->decode_pre_ep_compressed_kv_ms_per_step =
            decode_loop.pre_ep_compressed_kv_ms_per_step;
        summary->decode_pre_ep_attention_state_ms_per_step =
            decode_loop.pre_ep_attention_state_ms_per_step;
        summary->decode_pre_ep_typed_history_ms_per_step =
            decode_loop.pre_ep_typed_history_ms_per_step;
        summary->decode_pre_ep_raw_read_ms_per_step =
            decode_loop.pre_ep_raw_read_ms_per_step;
        summary->decode_pre_ep_attention_output_ms_per_step =
            decode_loop.pre_ep_attention_output_ms_per_step;
        summary->decode_pre_ep_post_attention_ffn_input_ms_per_step =
            decode_loop.pre_ep_post_attention_ffn_input_ms_per_step;
        summary->decode_final_hc_ms_per_step = decode_loop.final_hc_ms_per_step;
        summary->decode_cudagraph_sync_all_calls =
            decode_loop.cudagraph_sync_all_calls;
        summary->decode_cudagraph_event_barrier_calls =
            decode_loop.cudagraph_event_barrier_calls;
        summary->decode_cudagraph_rank_stream_syncs =
            decode_loop.cudagraph_rank_stream_syncs;
        summary->decode_cudagraph_dense_stream_syncs =
            decode_loop.cudagraph_dense_stream_syncs;
        summary->decode_cudagraph_copy_stream_syncs =
            decode_loop.cudagraph_copy_stream_syncs;
        summary->decode_cudagraph_capture_attempted =
            decode_loop.cudagraph_capture_attempted;
        summary->decode_cudagraph_capture_succeeded =
            decode_loop.cudagraph_capture_succeeded;
        summary->decode_cudagraph_capture_error =
            decode_loop.cudagraph_capture_error;
        summary->decode_cudagraph_capture_nodes =
            decode_loop.cudagraph_capture_nodes;
        summary->decode_cudagraph_replay_attempted =
            decode_loop.cudagraph_replay_attempted;
        summary->decode_cudagraph_replay_succeeded =
            decode_loop.cudagraph_replay_succeeded;
        summary->decode_cudagraph_replay_error =
            decode_loop.cudagraph_replay_error;
        summary->decode_cudagraph_persistent_cache_hits =
            decode_loop.cudagraph_persistent_cache_hits;
        summary->decode_cudagraph_persistent_cache_misses =
            decode_loop.cudagraph_persistent_cache_misses;
        summary->decode_cudagraph_persistent_invalidations =
            decode_loop.cudagraph_persistent_invalidations;
        summary->decode_cudagraph_persistent_invalidate_layer =
            decode_loop.cudagraph_persistent_invalidate_layer;
        summary->decode_cudagraph_persistent_invalidate_slots =
            decode_loop.cudagraph_persistent_invalidate_slots;
        summary->decode_cudagraph_persistent_invalidate_position =
            decode_loop.cudagraph_persistent_invalidate_position;
        summary->decode_cudagraph_persistent_invalidate_root_device =
            decode_loop.cudagraph_persistent_invalidate_root_device;
        summary->decode_cudagraph_persistent_invalidate_root_stream =
            decode_loop.cudagraph_persistent_invalidate_root_stream;
        summary->decode_cudagraph_instantiate_ms =
            decode_loop.cudagraph_instantiate_ms;
        summary->decode_cudagraph_replay_ms =
            decode_loop.cudagraph_replay_ms;
        summary->decode_checksum = decode_loop.checksum;
        summary->decode_finite_bad = decode_loop.finite_bad;
        summary->rc = rc;
    }
    return rc;
}

