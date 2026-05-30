int run_token_major_serving_loop(const Options &opt,
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
                                 const std::vector<uint32_t> *decode_input_tokens,
                                 const std::vector<unsigned char> *decode_active_slots,
                                 std::vector<ContractRow> resident_rows[43],
                                 LayerStats resident_stats[43],
                                 bool resident_serving_loop,
                                 ServingBenchResult *serving_result) {
    int pass_invocations = 0;
    double sum_decode_ms = 0.0;
    double sum_ep_ms = 0.0;
    double sum_dense_ms = 0.0;
    double sum_compose_ms = 0.0;
    double sum_compose_reduce_ms = 0.0;
    double sum_compose_copy_ms = 0.0;
    double sum_compose_final_ms = 0.0;
    double sum_hc_current_input_ms = 0.0;
    double sum_hc_current_seed_ms = 0.0;
    double sum_hc_current_attn_mix_ms = 0.0;
    double sum_hc_current_split_ms = 0.0;
    double sum_hc_current_gather_ms = 0.0;
    double sum_hc_current_ffn_router_ms = 0.0;
    double sum_hc_current_ffn_norm_ms = 0.0;
    double sum_hc_current_router_select_ms = 0.0;
    double sum_hc_current_router_d2h_ms = 0.0;
    double sum_hc_current_route_upload_ms = 0.0;
    double sum_hc_current_fill_pack_ms = 0.0;
    double sum_pre_ep_hc_current_ms = 0.0;
    double sum_pre_ep_attention_projection_ms = 0.0;
    double sum_pre_ep_compressed_kv_ms = 0.0;
    double sum_pre_ep_attention_state_ms = 0.0;
    double sum_pre_ep_typed_history_ms = 0.0;
    double sum_pre_ep_raw_read_ms = 0.0;
    double sum_pre_ep_attention_output_ms = 0.0;
    double sum_pre_ep_post_attention_ffn_input_ms = 0.0;
    double sum_final_hc_ms = 0.0;
    int sum_cudagraph_sync_all_calls = 0;
    int sum_cudagraph_event_barrier_calls = 0;
    int sum_cudagraph_rank_stream_syncs = 0;
    int sum_cudagraph_dense_stream_syncs = 0;
    int sum_cudagraph_copy_stream_syncs = 0;
    int sum_cudagraph_capture_attempted = 0;
    int sum_cudagraph_capture_succeeded = 0;
    int sum_cudagraph_capture_error = 0;
    size_t sum_cudagraph_capture_nodes = 0;
    int sum_cudagraph_replay_attempted = 0;
    int sum_cudagraph_replay_succeeded = 0;
    int sum_cudagraph_replay_error = 0;
    int sum_cudagraph_persistent_cache_hits = 0;
    int sum_cudagraph_persistent_cache_misses = 0;
    int sum_cudagraph_persistent_invalidations = 0;
    int sum_cudagraph_persistent_invalidate_layer = 0;
    int sum_cudagraph_persistent_invalidate_slots = 0;
    int sum_cudagraph_persistent_invalidate_position = 0;
    int sum_cudagraph_persistent_invalidate_root_device = 0;
    int sum_cudagraph_persistent_invalidate_root_stream = 0;
    double sum_cudagraph_instantiate_ms = 0.0;
    double sum_cudagraph_replay_ms = 0.0;
    double first_token_decode_ms = 0.0;
    double continuation_decode_ms = 0.0;
    double first_token_wall_ms = 0.0;
    double continuation_wall_ms = 0.0;
    uint64_t checksum = 0;
    if (opt.final_hc_carry_gate && !opt.tp_hc_persist_state_gate &&
        shared_rank_buffers && shared_rank_buffers->initialized) {
        for (int rank = 0; rank < kGpus; ++rank) {
            shared_rank_buffers->ranks[rank].hc_initialized = false;
        }
    }
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < opt.decode_steps; ++step) {
        const auto step_start = std::chrono::steady_clock::now();
        double step_decode_ms = 0.0;
        if (step == 0 && shared_token_embedding && decode_input_tokens &&
            !decode_input_tokens->empty()) {
            if (!shared_rank_buffers || !shared_rank_buffers->initialized ||
                ensure_compose_buffers(opt, shared_rank_buffers->ranks) != 0) {
                std::fprintf(stderr, "tp_ep_token_embedding_seed_failed\treason\tmissing_rank_buffers\n");
                return 15;
            }
            const int seed_rc = seed_rank_hc_from_input_tokens(
                opt, shared_token_embedding, shared_rank_buffers->ranks,
                *decode_input_tokens);
            if (seed_rc != 0) {
                std::fprintf(stderr, "tp_ep_token_embedding_seed_failed\trc\t%d\n",
                             seed_rc);
                return 15;
            }
        }
        if (shared_hc_controls && shared_hc_controls->initialized &&
            shared_hc_controls->d_router_tokens &&
            decode_input_tokens && decode_input_tokens->size() >= (size_t)opt.slots) {
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
            CHECK_CUDA(cudaMemcpy(shared_hc_controls->d_router_tokens,
                                  decode_input_tokens->data(),
                                  (size_t)opt.slots * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
        }
        if (shared_hc_controls && shared_hc_controls->initialized &&
            shared_hc_controls->d_router_active &&
            decode_active_slots && decode_active_slots->size() >= (size_t)opt.slots) {
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
            CHECK_CUDA(cudaMemcpy(shared_hc_controls->d_router_active,
                                  decode_active_slots->data(),
                                  (size_t)opt.slots * sizeof(unsigned char),
                                  cudaMemcpyHostToDevice));
        }
        for (int layer = 0; layer < 43; ++layer) {
            Options layer_opt = opt;
            layer_opt.layer = layer;
            layer_opt.position = opt.position + (uint64_t)step;
            layer_opt.decode_steps = 1;
            layer_opt.true_ds4_attention_raw_valid_rows =
                std::max(1u, std::min((uint32_t)(step + 1), (uint32_t)kRawSwaRows));
            layer_opt.warmup = 0;
            LayerRunSummary s;
            SharedTpRuntime *tp_runtime_arg =
                shared_tp_runtime && shared_tp_runtime->initialized ? shared_tp_runtime : nullptr;
            const SharedExpertBindings *expert_arg =
                shared_expert_bindings && shared_expert_bindings->initialized
                    ? shared_expert_bindings
                    : nullptr;
            const SharedDenseOps *dense_ops_arg =
                shared_dense_ops && shared_dense_ops->initialized ? shared_dense_ops : nullptr;
            int rc = 0;
            if (resident_serving_loop) {
                if (!shared_api || !shared_api->initialized ||
                    !shared_rank_buffers || !shared_rank_buffers->initialized ||
                    !shared_tp_runtime || !shared_tp_runtime->initialized ||
                    !shared_expert_bindings || !shared_expert_bindings->initialized ||
                    !shared_dense_f16_cache || !shared_dense_f16_cache->enabled) {
                    std::fprintf(stderr, "resident serving loop missing shared state\n");
                    rc = 2;
                    s.pass = false;
                } else {
                    const LayerDenseOps *layer_dense_ops =
                        dense_ops_arg ? &dense_ops_arg->layers[layer] : nullptr;
                    TpCudaGraphLayerExec *persistent_graph =
                        opt.decode_cudagraph_persistent_replay_gate
                            ? &shared_rank_buffers->graph_cache.layers[layer]
                            : nullptr;
                    rc = run_resident_layer_decode(layer_opt,
                                                   resident_rows[layer],
                                                   resident_stats[layer],
                                                   shared_rank_buffers->ranks,
                                                   shared_api->api,
                                                   shared_tp_runtime->rt,
                                                   &shared_expert_bindings->layers[layer],
                                                   shared_dense_f16_cache,
                                                   layer_dense_ops,
                                                   shared_hc_controls,
                                                   persistent_graph,
                                                   &s);
                }
            } else {
                rc = run_layer(layer_opt, &s, shared_dense_f16_cache, shared_api,
                               shared_rank_buffers, tp_runtime_arg, expert_arg,
                               dense_ops_arg, shared_hc_controls);
            }
            std::printf("tp_ep_token_major_item\tstep\t%d\tlayer\t%d\tratio\t%d\t"
                        "position\t%llu\t"
                        "decode_ms_per_step\t%.6f\tdecode_slot_step_tok_s\t%.6f\t"
                        "decode_ep_ms_per_step\t%.6f\tdecode_dense_ms_per_step\t%.6f\t"
                        "decode_compose_ms_per_step\t%.6f\t"
                        "decode_compose_reduce_ms_per_step\t%.6f\t"
                        "decode_compose_copy_ms_per_step\t%.6f\t"
                        "decode_compose_final_ms_per_step\t%.6f\t"
                        "decode_hc_current_input_ms_per_step\t%.6f\t"
                        "decode_hc_current_seed_ms_per_step\t%.6f\t"
                        "decode_hc_current_attn_mix_ms_per_step\t%.6f\t"
                        "decode_hc_current_split_ms_per_step\t%.6f\t"
                        "decode_hc_current_gather_ms_per_step\t%.6f\t"
                        "decode_hc_current_ffn_router_ms_per_step\t%.6f\t"
                        "decode_hc_current_ffn_norm_ms_per_step\t%.6f\t"
                        "decode_hc_current_router_select_ms_per_step\t%.6f\t"
                        "decode_hc_current_router_d2h_ms_per_step\t%.6f\t"
                        "decode_hc_current_route_upload_ms_per_step\t%.6f\t"
                        "decode_hc_current_fill_pack_ms_per_step\t%.6f\t"
                        "decode_pre_ep_hc_current_ms_per_step\t%.6f\t"
                        "decode_pre_ep_attention_projection_ms_per_step\t%.6f\t"
                        "decode_pre_ep_compressed_kv_ms_per_step\t%.6f\t"
                        "decode_pre_ep_attention_state_ms_per_step\t%.6f\t"
                        "decode_pre_ep_typed_history_ms_per_step\t%.6f\t"
                        "decode_pre_ep_raw_read_ms_per_step\t%.6f\t"
                        "decode_pre_ep_attention_output_ms_per_step\t%.6f\t"
                        "decode_pre_ep_post_attention_ffn_input_ms_per_step\t%.6f\t"
                        "decode_final_hc_ms_per_step\t%.6f\t"
                        "decode_cudagraph_replay_attempted\t%d\t"
                        "decode_cudagraph_replay_succeeded\t%d\t"
                        "decode_cudagraph_persistent_cache_hits\t%d\t"
                        "decode_cudagraph_persistent_cache_misses\t%d\t"
                        "decode_cudagraph_persistent_invalidations\t%d\t"
                        "decode_cudagraph_persistent_invalidate_position\t%d\t"
                        "decode_cudagraph_instantiate_ms\t%.6f\t"
                        "decode_cudagraph_replay_ms\t%.6f\t"
                        "decode_checksum\t%llu\tdecode_finite_bad\t%d\trc\t%d\t%s\n",
                        step, s.layer, s.ratio,
                        (unsigned long long)layer_opt.position,
                        s.decode_ms_per_step,
                        s.decode_slot_step_tok_s,
                        s.decode_ep_ms_per_step,
                        s.decode_dense_ms_per_step,
                        s.decode_compose_ms_per_step,
                        s.decode_compose_reduce_ms_per_step,
                        s.decode_compose_copy_ms_per_step,
                        s.decode_compose_final_ms_per_step,
                        s.decode_hc_current_input_ms_per_step,
                        s.decode_hc_current_seed_ms_per_step,
                        s.decode_hc_current_attn_mix_ms_per_step,
                        s.decode_hc_current_split_ms_per_step,
                        s.decode_hc_current_gather_ms_per_step,
                        s.decode_hc_current_ffn_router_ms_per_step,
                        s.decode_hc_current_ffn_norm_ms_per_step,
                        s.decode_hc_current_router_select_ms_per_step,
                        s.decode_hc_current_router_d2h_ms_per_step,
                        s.decode_hc_current_route_upload_ms_per_step,
                        s.decode_hc_current_fill_pack_ms_per_step,
                        s.decode_pre_ep_hc_current_ms_per_step,
                        s.decode_pre_ep_attention_projection_ms_per_step,
                        s.decode_pre_ep_compressed_kv_ms_per_step,
                        s.decode_pre_ep_attention_state_ms_per_step,
                        s.decode_pre_ep_typed_history_ms_per_step,
                        s.decode_pre_ep_raw_read_ms_per_step,
                        s.decode_pre_ep_attention_output_ms_per_step,
                        s.decode_pre_ep_post_attention_ffn_input_ms_per_step,
                        s.decode_final_hc_ms_per_step,
                        s.decode_cudagraph_replay_attempted,
                        s.decode_cudagraph_replay_succeeded,
                        s.decode_cudagraph_persistent_cache_hits,
                        s.decode_cudagraph_persistent_cache_misses,
                        s.decode_cudagraph_persistent_invalidations,
                        s.decode_cudagraph_persistent_invalidate_position,
                        s.decode_cudagraph_instantiate_ms,
                        s.decode_cudagraph_replay_ms,
                        (unsigned long long)s.decode_checksum,
                        s.decode_finite_bad,
                        rc,
                        (rc == 0 && s.pass) ? "PASS" : "FAIL");
            if (rc == 0 && s.pass) {
                pass_invocations++;
                sum_decode_ms += s.decode_ms_per_step;
                step_decode_ms += s.decode_ms_per_step;
                sum_ep_ms += s.decode_ep_ms_per_step;
                sum_dense_ms += s.decode_dense_ms_per_step;
                sum_compose_ms += s.decode_compose_ms_per_step;
                sum_compose_reduce_ms += s.decode_compose_reduce_ms_per_step;
                sum_compose_copy_ms += s.decode_compose_copy_ms_per_step;
                sum_compose_final_ms += s.decode_compose_final_ms_per_step;
                sum_hc_current_input_ms += s.decode_hc_current_input_ms_per_step;
                sum_hc_current_seed_ms += s.decode_hc_current_seed_ms_per_step;
                sum_hc_current_attn_mix_ms += s.decode_hc_current_attn_mix_ms_per_step;
                sum_hc_current_split_ms += s.decode_hc_current_split_ms_per_step;
                sum_hc_current_gather_ms += s.decode_hc_current_gather_ms_per_step;
                sum_hc_current_ffn_router_ms += s.decode_hc_current_ffn_router_ms_per_step;
                sum_hc_current_ffn_norm_ms += s.decode_hc_current_ffn_norm_ms_per_step;
                sum_hc_current_router_select_ms +=
                    s.decode_hc_current_router_select_ms_per_step;
                sum_hc_current_router_d2h_ms +=
                    s.decode_hc_current_router_d2h_ms_per_step;
                sum_hc_current_route_upload_ms +=
                    s.decode_hc_current_route_upload_ms_per_step;
                sum_hc_current_fill_pack_ms += s.decode_hc_current_fill_pack_ms_per_step;
                sum_pre_ep_hc_current_ms += s.decode_pre_ep_hc_current_ms_per_step;
                sum_pre_ep_attention_projection_ms +=
                    s.decode_pre_ep_attention_projection_ms_per_step;
                sum_pre_ep_compressed_kv_ms += s.decode_pre_ep_compressed_kv_ms_per_step;
                sum_pre_ep_attention_state_ms +=
                    s.decode_pre_ep_attention_state_ms_per_step;
                sum_pre_ep_typed_history_ms += s.decode_pre_ep_typed_history_ms_per_step;
                sum_pre_ep_raw_read_ms += s.decode_pre_ep_raw_read_ms_per_step;
                sum_pre_ep_attention_output_ms +=
                    s.decode_pre_ep_attention_output_ms_per_step;
                sum_pre_ep_post_attention_ffn_input_ms +=
                    s.decode_pre_ep_post_attention_ffn_input_ms_per_step;
                sum_final_hc_ms += s.decode_final_hc_ms_per_step;
                sum_cudagraph_sync_all_calls +=
                    s.decode_cudagraph_sync_all_calls;
                sum_cudagraph_event_barrier_calls +=
                    s.decode_cudagraph_event_barrier_calls;
                sum_cudagraph_rank_stream_syncs +=
                    s.decode_cudagraph_rank_stream_syncs;
                sum_cudagraph_dense_stream_syncs +=
                    s.decode_cudagraph_dense_stream_syncs;
                sum_cudagraph_copy_stream_syncs +=
                    s.decode_cudagraph_copy_stream_syncs;
                sum_cudagraph_capture_attempted +=
                    s.decode_cudagraph_capture_attempted;
                sum_cudagraph_capture_succeeded +=
                    s.decode_cudagraph_capture_succeeded;
                if (sum_cudagraph_capture_error == 0 &&
                    s.decode_cudagraph_capture_error != 0) {
                    sum_cudagraph_capture_error =
                        s.decode_cudagraph_capture_error;
                }
                sum_cudagraph_capture_nodes +=
                    s.decode_cudagraph_capture_nodes;
                sum_cudagraph_replay_attempted +=
                    s.decode_cudagraph_replay_attempted;
                sum_cudagraph_replay_succeeded +=
                    s.decode_cudagraph_replay_succeeded;
                if (sum_cudagraph_replay_error == 0 &&
                    s.decode_cudagraph_replay_error != 0) {
                    sum_cudagraph_replay_error =
                        s.decode_cudagraph_replay_error;
                }
                sum_cudagraph_persistent_cache_hits +=
                    s.decode_cudagraph_persistent_cache_hits;
                sum_cudagraph_persistent_cache_misses +=
                    s.decode_cudagraph_persistent_cache_misses;
                sum_cudagraph_persistent_invalidations +=
                    s.decode_cudagraph_persistent_invalidations;
                sum_cudagraph_persistent_invalidate_layer +=
                    s.decode_cudagraph_persistent_invalidate_layer;
                sum_cudagraph_persistent_invalidate_slots +=
                    s.decode_cudagraph_persistent_invalidate_slots;
                sum_cudagraph_persistent_invalidate_position +=
                    s.decode_cudagraph_persistent_invalidate_position;
                sum_cudagraph_persistent_invalidate_root_device +=
                    s.decode_cudagraph_persistent_invalidate_root_device;
                sum_cudagraph_persistent_invalidate_root_stream +=
                    s.decode_cudagraph_persistent_invalidate_root_stream;
                sum_cudagraph_instantiate_ms +=
                    s.decode_cudagraph_instantiate_ms;
                sum_cudagraph_replay_ms +=
                    s.decode_cudagraph_replay_ms;
                checksum ^= s.decode_checksum +
                            (uint64_t)(step + 1) * 1000003ull +
                            (uint64_t)(layer + 1) * 104729ull;
            } else {
                const auto stop = std::chrono::steady_clock::now();
                const double wall_ms =
                    std::chrono::duration<double, std::milli>(stop - start).count();
                std::printf("tp_ep_token_major_scaffold\tsteps\t%d\tlayers\t43\t"
                            "pass_invocations\t%d\tfailed_step\t%d\tfailed_layer\t%d\t"
                            "slots\t%d\tctx\t262144\twall_ms\t%.6f\tFAIL\n",
                            opt.decode_steps, pass_invocations, step, layer,
                            opt.slots, wall_ms);
                std::fflush(stdout);
                return rc == 0 ? 1 : rc;
            }
        }
        /* MTP draft (layer 43): run the MTP block via run_layer after the main
         * 0-42 stack, validating it executes in the active token-major path.
         * run_layer redirects to mtp_layer + mtp_contract on layer==43; ratio=0;
         * no f16 cache / dense_ops / graph cache for the MTP layer. */
        if (shared_expert_bindings && shared_expert_bindings->mtp_initialized &&
            opt.mtp_contract_path) {
            Options mtp_opt = opt;
            mtp_opt.layer = 43;
            /* Sprint 585: preserve the main model's final hidden (layer-42 output)
             * before the MTP prologue/body overwrite d_final_hc_shard, so the main
             * output head still produces the correct served token. Restored after
             * the draft head below. Without this the MTP draft clobbers the served
             * token (MTP and the main head share d_final_hc_shard). */
            static float *mtp_hc_snap[kGpus] = {nullptr};
            static int mtp_hc_snap_slots = 0;
            const uint64_t mtp_hc_elems =
                (uint64_t)opt.slots * kHcRows * (kHidden / kGpus);
            bool mtp_hc_saved = false;
            if (shared_rank_buffers && shared_rank_buffers->initialized) {
                if (mtp_hc_snap_slots != opt.slots) {
                    for (int r = 0; r < kGpus; ++r) {
                        if (mtp_hc_snap[r]) {
                            CHECK_CUDA(cudaSetDevice(opt.devices[r]));
                            CHECK_CUDA(cudaFree(mtp_hc_snap[r]));
                            mtp_hc_snap[r] = nullptr;
                        }
                    }
                    for (int r = 0; r < kGpus; ++r) {
                        CHECK_CUDA(cudaSetDevice(opt.devices[r]));
                        CHECK_CUDA(cudaMalloc(&mtp_hc_snap[r],
                                              mtp_hc_elems * sizeof(float)));
                    }
                    mtp_hc_snap_slots = opt.slots;
                }
                for (int r = 0; r < kGpus; ++r) {
                    RankState &rk = shared_rank_buffers->ranks[r];
                    if (!rk.d_final_hc_shard || !mtp_hc_snap[r]) continue;
                    CHECK_CUDA(cudaSetDevice(rk.device));
                    CHECK_CUDA(cudaMemcpyAsync(mtp_hc_snap[r], rk.d_final_hc_shard,
                                               mtp_hc_elems * sizeof(float),
                                               cudaMemcpyDeviceToDevice, rk.stream));
                }
                mtp_hc_saved = true;
            }
            /* Sprint 585 (1.2b): MTP embedding-combine prologue. Combines the
             * current token's embedding with the prev-hidden (layer-42 output)
             * already in ranks[].d_final_hc_shard, writing the MTP input back
             * into d_final_hc_shard for run_layer(43). No-op unless the MTP dense
             * ops (e_proj/h_proj/enorm/hnorm) are loaded. Runs BEFORE run_layer. */
            if (shared_dense_ops && shared_dense_ops->initialized &&
                shared_rank_buffers && shared_rank_buffers->initialized &&
                shared_token_embedding) {
                const int prc = run_mtp_prologue(
                    opt, shared_rank_buffers->ranks, &shared_dense_ops->layers[43],
                    shared_token_embedding, decode_input_tokens);
                if (prc != 0) {
                    std::fprintf(stderr, "tp_ep_mtp_prologue_failed\trc\t%d\n", prc);
                }
            }
            LayerRunSummary ms;
            SharedTpRuntime *mtp_tp =
                (shared_tp_runtime && shared_tp_runtime->initialized) ? shared_tp_runtime
                                                                      : nullptr;
            const int mrc = run_layer(mtp_opt, &ms, nullptr, shared_api,
                                      shared_rank_buffers, mtp_tp,
                                      shared_expert_bindings, shared_dense_ops,
                                      shared_hc_controls);
            std::printf("tp_ep_mtp_layer_scaffold\tstep\t%d\tlayer\t43\tratio\t%d\t"
                        "expert_rows\t%llu\tdense_rows\t%llu\tcontrol_rows\t%llu\t"
                        "decode_ms_per_step\t%.6f\tdecode_checksum\t%llu\trc\t%d\t%s\n",
                        step, ms.ratio, (unsigned long long)ms.expert_rows,
                        (unsigned long long)ms.dense_rows,
                        (unsigned long long)ms.control_rows, ms.decode_ms_per_step,
                        (unsigned long long)ms.decode_checksum, mrc,
                        (mrc == 0 && ms.pass) ? "PASS" : "FAIL");
            std::fflush(stdout);
            /* Sprint 585 Phase 1 (increment 1.1): MTP draft head. After the MTP
             * body (run_layer 43) leaves its output in ranks[].d_final_hc_shard,
             * run the dedicated MTP output head (shares the LM matmul, overrides
             * the head HC weights) to produce DRAFT logits/token. No new kernels;
             * validates the serving-config MTP head end-to-end. mtp_output_head is
             * loaded once in appliance_runtime (load_mtp_output_head) and threaded
             * here; null unless MTP is configured + the head is open, so the
             * shipped mtp=off serving path is untouched. The prologue
             * (embedding-combine) that makes this the TRUE MTP draft lands next
             * (increment 1.2); for now the head consumes the body output directly. */
            /* Gate on the MTP body having computed finite output, NOT on mrc==0:
             * run_layer(43) returns the benign rc=1 (kv_rows>0 scaffold check,
             * inapplicable to the raw-SWA MTP per Sprint 584) while still
             * producing a valid decode (decode_pass=1, finite). */
            if (ms.decode_finite_bad == 0 && ms.decode_checksum != 0 &&
                mtp_output_head && mtp_output_head->initialized &&
                shared_rank_buffers && shared_rank_buffers->initialized) {
                OutputHeadRunResult draft;
                const int drc = run_shared_output_head_from_rank_hc(
                    opt, mtp_output_head, shared_rank_buffers->ranks, &draft);
                std::printf("tp_ep_mtp_draft_head\tstep\t%d\tslots\t%d\t"
                            "draft_token\t%u\tdraft_logit\t%.9f\tfinite_bad\t%d\t"
                            "total_ms\t%.6f\tchecksum\t%llu\t%s\n",
                            step, opt.slots,
                            draft.tokens.empty() ? UINT32_MAX : draft.tokens[0],
                            draft.logits.empty() ? 0.0f : draft.logits[0],
                            draft.finite_bad, draft.total_ms,
                            (unsigned long long)draft.checksum,
                            (drc == 0 && draft.pass) ? "PASS" : "FAIL");
                std::fflush(stdout);
            }
            /* Sprint 585: restore the main model's final hidden so the main output
             * head (serving_result block below) produces the correct served token,
             * undisturbed by the MTP draft's use of d_final_hc_shard. */
            if (mtp_hc_saved) {
                for (int r = 0; r < kGpus; ++r) {
                    RankState &rk = shared_rank_buffers->ranks[r];
                    if (!rk.d_final_hc_shard || !mtp_hc_snap[r]) continue;
                    CHECK_CUDA(cudaSetDevice(rk.device));
                    CHECK_CUDA(cudaMemcpyAsync(rk.d_final_hc_shard, mtp_hc_snap[r],
                                               mtp_hc_elems * sizeof(float),
                                               cudaMemcpyDeviceToDevice, rk.stream));
                }
                for (int r = 0; r < kGpus; ++r) {
                    CHECK_CUDA(cudaSetDevice(shared_rank_buffers->ranks[r].device));
                    CHECK_CUDA(cudaStreamSynchronize(shared_rank_buffers->ranks[r].stream));
                }
            }
        }
        const auto step_stop = std::chrono::steady_clock::now();
        const double step_wall_ms =
            std::chrono::duration<double, std::milli>(step_stop - step_start).count();
        if (step == 0) {
            first_token_decode_ms += step_decode_ms;
            first_token_wall_ms += step_wall_ms;
        } else {
            continuation_decode_ms += step_decode_ms;
            continuation_wall_ms += step_wall_ms;
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    const double ms_per_token = opt.decode_steps > 0
        ? sum_decode_ms / (double)opt.decode_steps
        : 0.0;
    const double slot_step_tok_s = sum_decode_ms > 0.0
        ? (double)opt.slots * (double)opt.decode_steps * 1000.0 / sum_decode_ms
        : 0.0;
    std::printf("tp_ep_token_major_scaffold\tsteps\t%d\tlayers\t43\t"
                "pass_invocations\t%d\tslots\t%d\tctx\t262144\t"
                "shared_api\t%d\tshared_rank_buffers\t%d\tshared_tp_runtime\t%d\t"
                "shared_expert_bindings\t%d\toverlap_ep_dense\t%d\t"
                "shared_dense_ops\t%d\t"
                "skip_decode_checksum\t%d\t"
                "direct_remote_compose\t%d\tsource_copy_schedule\t%d\t"
                "skip_self_compose_copy\t%d\t"
                "multi_copy_streams\t%d\t"
                "compact_moe_decode_gate\t%d\t"
                "router_cublas_gate\t%d\t"
                "router_hash_fast_gate\t%d\t"
                "gpu_route_plan_gate\t%d\t"
                "route_plan_async_upload_gate\t%d\t"
                "fused_gated_silu_gate\t%d\t"
                "routed_ffn_norm_input_gate\t%d\t"
                "routed_ffn_rank_major_input_gate\t%d\t"
                "routed_ffn_rank_major_shared_input_gate\t%d\t"
                "routed_ffn_rank_major_route_input_gate\t%d\t"
                "routed_ffn_rank_major_input_parity_gate\t%d\t"
                "post_attention_route_reuse_audit_gate\t%d\t"
                "post_attention_fixed_capacity_route_plan_gate\t%d\t"
                "post_attention_slot_major_ffn_norm_gate\t%d\t"
                "model_router_rank_major_logits_gate\t%d\t"
                "model_router_allreduce_logits_gate\t%d\t"
                "routed_gate_standalone_swiglu\t%d\t"
                "sum_decode_ms\t%.6f\tms_per_token\t%.6f\t"
                "projected_slot_step_tok_s\t%.6f\t"
                "sum_ep_ms\t%.6f\tsum_dense_ms\t%.6f\tsum_compose_ms\t%.6f\t"
                "sum_compose_reduce_ms\t%.6f\tsum_compose_copy_ms\t%.6f\t"
                "sum_compose_final_ms\t%.6f\t"
                "tp_hc_current_input_gate\t%d\t"
                "tp_hc_current_input_peer_gather\t%d\t"
                "tp_hc_current_input_nccl_allgather\t%d\t"
                "tp_hc_current_allreduce\t%d\t"
                "tp_hc_current_input_stream_sync\t%d\t"
                "sum_hc_current_input_ms\t%.6f\t"
                "sum_hc_current_seed_ms\t%.6f\t"
                "sum_hc_current_attn_mix_ms\t%.6f\t"
                "sum_hc_current_split_ms\t%.6f\t"
                "sum_hc_current_gather_ms\t%.6f\t"
                "sum_hc_current_ffn_router_ms\t%.6f\t"
                "sum_hc_current_ffn_norm_ms\t%.6f\t"
                "sum_hc_current_router_select_ms\t%.6f\t"
                "sum_hc_current_router_d2h_ms\t%.6f\t"
                "sum_hc_current_route_upload_ms\t%.6f\t"
                "sum_hc_current_fill_pack_ms\t%.6f\t"
                "sum_pre_ep_hc_current_ms\t%.6f\t"
                "attention_projection_rank_local_input_gate\t%d\t"
                "attention_projection_rank_major_input_gate\t%d\t"
                "sum_pre_ep_attention_projection_ms\t%.6f\t"
                "sum_pre_ep_compressed_kv_ms\t%.6f\t"
                "sum_pre_ep_attention_state_ms\t%.6f\t"
                "sum_pre_ep_typed_history_ms\t%.6f\t"
                "sum_pre_ep_raw_read_ms\t%.6f\t"
                "sum_pre_ep_attention_output_ms\t%.6f\t"
                "sum_pre_ep_post_attention_ffn_input_ms\t%.6f\t"
                "final_hc_carry_gate\t%d\tsum_final_hc_ms\t%.6f\t"
                "decode_cudagraph_capture_attempted\t%d\t"
                "decode_cudagraph_capture_succeeded\t%d\t"
                "decode_cudagraph_replay_attempted\t%d\t"
                "decode_cudagraph_replay_succeeded\t%d\t"
                "decode_cudagraph_persistent_cache_hits\t%d\t"
                "decode_cudagraph_persistent_cache_misses\t%d\t"
                "decode_cudagraph_persistent_invalidations\t%d\t"
                "decode_cudagraph_persistent_invalidate_layer\t%d\t"
                "decode_cudagraph_persistent_invalidate_slots\t%d\t"
                "decode_cudagraph_persistent_invalidate_position\t%d\t"
                "decode_cudagraph_persistent_invalidate_root_device\t%d\t"
                "decode_cudagraph_persistent_invalidate_root_stream\t%d\t"
                "decode_cudagraph_instantiate_ms\t%.6f\t"
                "decode_cudagraph_replay_ms\t%.6f\t"
                "wall_ms\t%.6f\tchecksum\t%llu\tPASS\n",
                opt.decode_steps, pass_invocations, opt.slots,
                shared_api && shared_api->initialized ? 1 : 0,
                shared_rank_buffers && shared_rank_buffers->initialized ? 1 : 0,
                shared_tp_runtime && shared_tp_runtime->initialized ? 1 : 0,
                shared_expert_bindings && shared_expert_bindings->initialized ? 1 : 0,
                opt.overlap_ep_dense ? 1 : 0,
                shared_dense_ops && shared_dense_ops->initialized ? 1 : 0,
                opt.skip_decode_checksum ? 1 : 0,
                opt.direct_remote_compose ? 1 : 0,
                opt.source_copy_schedule ? 1 : 0,
                opt.skip_self_compose_copy ? 1 : 0,
                opt.multi_copy_streams ? 1 : 0,
                opt.compact_moe_decode_gate ? 1 : 0,
                opt.router_cublas_gate ? 1 : 0,
                opt.router_hash_fast_gate ? 1 : 0,
                opt.gpu_route_plan_gate ? 1 : 0,
                opt.route_plan_async_upload_gate ? 1 : 0,
                opt.fused_gated_silu_gate ? 1 : 0,
                opt.routed_ffn_norm_input_gate ? 1 : 0,
                opt.routed_ffn_rank_major_input_gate ? 1 : 0,
                opt.routed_ffn_rank_major_shared_input_gate ? 1 : 0,
                opt.routed_ffn_rank_major_route_input_gate ? 1 : 0,
                opt.routed_ffn_rank_major_input_parity_gate ? 1 : 0,
                opt.post_attention_route_reuse_audit_gate ? 1 : 0,
                opt.post_attention_fixed_capacity_route_plan_gate ? 1 : 0,
                opt.post_attention_slot_major_ffn_norm_gate ? 1 : 0,
                opt.model_router_rank_major_logits_gate ? 1 : 0,
                opt.model_router_allreduce_logits_gate ? 1 : 0,
                (opt.routed_ffn_norm_input_gate &&
                 !(opt.fused_gated_silu_gate && !opt.reference_hc_reduce_gate)) ? 1 : 0,
                sum_decode_ms, ms_per_token, slot_step_tok_s,
                sum_ep_ms, sum_dense_ms, sum_compose_ms,
                sum_compose_reduce_ms, sum_compose_copy_ms,
                sum_compose_final_ms,
                opt.tp_hc_current_input_gate ? 1 : 0,
                opt.tp_hc_current_input_peer_gather_gate ? 1 : 0,
                opt.tp_hc_current_input_nccl_allgather_gate ? 1 : 0,
                opt.tp_hc_current_allreduce_gate ? 1 : 0,
                opt.tp_hc_current_input_stream_sync_gate ? 1 : 0,
                sum_hc_current_input_ms,
                sum_hc_current_seed_ms,
                sum_hc_current_attn_mix_ms,
                sum_hc_current_split_ms,
                sum_hc_current_gather_ms,
                sum_hc_current_ffn_router_ms,
                sum_hc_current_ffn_norm_ms,
                sum_hc_current_router_select_ms,
                sum_hc_current_router_d2h_ms,
                sum_hc_current_route_upload_ms,
                sum_hc_current_fill_pack_ms,
                sum_pre_ep_hc_current_ms,
                opt.true_ds4_attention_projection_rank_local_input_gate ? 1 : 0,
                opt.true_ds4_attention_projection_rank_major_input_gate ? 1 : 0,
                sum_pre_ep_attention_projection_ms,
                sum_pre_ep_compressed_kv_ms,
                sum_pre_ep_attention_state_ms,
                sum_pre_ep_typed_history_ms,
                sum_pre_ep_raw_read_ms,
                sum_pre_ep_attention_output_ms,
                sum_pre_ep_post_attention_ffn_input_ms,
                opt.final_hc_carry_gate ? 1 : 0, sum_final_hc_ms,
                sum_cudagraph_capture_attempted,
                sum_cudagraph_capture_succeeded,
                sum_cudagraph_replay_attempted,
                sum_cudagraph_replay_succeeded,
                sum_cudagraph_persistent_cache_hits,
                sum_cudagraph_persistent_cache_misses,
                sum_cudagraph_persistent_invalidations,
                sum_cudagraph_persistent_invalidate_layer,
                sum_cudagraph_persistent_invalidate_slots,
                sum_cudagraph_persistent_invalidate_position,
                sum_cudagraph_persistent_invalidate_root_device,
                sum_cudagraph_persistent_invalidate_root_stream,
                sum_cudagraph_instantiate_ms,
                sum_cudagraph_replay_ms,
                wall_ms, (unsigned long long)checksum);
    if (opt.serving_bench || serving_result) {
        const uint64_t prompt_tokens = (uint64_t)opt.slots;
        const uint64_t generated_tokens = (uint64_t)opt.slots *
                                          (uint64_t)opt.decode_steps;
        const uint64_t continuation_tokens = opt.decode_steps > 1
            ? (uint64_t)opt.slots * (uint64_t)(opt.decode_steps - 1)
            : 0ull;
        const double generated_tok_s_decode = sum_decode_ms > 0.0
            ? (double)generated_tokens * 1000.0 / sum_decode_ms
            : 0.0;
        const double generated_tok_s_wall = wall_ms > 0.0
            ? (double)generated_tokens * 1000.0 / wall_ms
            : 0.0;
        const double continuation_tok_s_decode = continuation_decode_ms > 0.0
            ? (double)continuation_tokens * 1000.0 / continuation_decode_ms
            : 0.0;
        const double continuation_tok_s_wall = continuation_wall_ms > 0.0
            ? (double)continuation_tokens * 1000.0 / continuation_wall_ms
            : 0.0;
        if (serving_result) {
            serving_result->prompt_tokens = prompt_tokens;
            serving_result->generated_tokens = generated_tokens;
            serving_result->continuation_tokens = continuation_tokens;
            serving_result->first_token_decode_ms = first_token_decode_ms;
            serving_result->continuation_decode_ms = continuation_decode_ms;
            serving_result->first_token_wall_ms = first_token_wall_ms;
            serving_result->continuation_wall_ms = continuation_wall_ms;
            serving_result->total_decode_ms = sum_decode_ms;
            serving_result->total_wall_ms = wall_ms;
            serving_result->total_ep_ms = sum_ep_ms;
            serving_result->total_dense_ms = sum_dense_ms;
            serving_result->total_compose_ms = sum_compose_ms;
            serving_result->total_compose_reduce_ms = sum_compose_reduce_ms;
            serving_result->total_compose_copy_ms = sum_compose_copy_ms;
            serving_result->total_compose_final_ms = sum_compose_final_ms;
            serving_result->total_hc_current_input_ms = sum_hc_current_input_ms;
            serving_result->token_input_seed =
                shared_token_embedding && decode_input_tokens &&
                !decode_input_tokens->empty();
            serving_result->first_input_token =
                decode_input_tokens && !decode_input_tokens->empty()
                    ? (*decode_input_tokens)[0]
                    : UINT32_MAX;
            serving_result->aggregate_generated_tok_s_decode = generated_tok_s_decode;
            serving_result->aggregate_generated_tok_s_wall = generated_tok_s_wall;
            serving_result->aggregate_continuation_tok_s_decode = continuation_tok_s_decode;
            serving_result->aggregate_continuation_tok_s_wall = continuation_tok_s_wall;
            serving_result->checksum = checksum;
        }
        SharedOutputHead lazy_output_head;
        SharedOutputHead *output_head_for_step = shared_output_head;
        const bool use_lazy_output_head =
            opt.diagnostic_output_head && opt.diagnostic_output_head_lazy_gate &&
            (opt.serving_bench || serving_result) &&
            (!output_head_for_step || !output_head_for_step->initialized) &&
            shared_rank_buffers && shared_rank_buffers->initialized;
        if (use_lazy_output_head) {
            std::vector<ContractRow> all_rows;
            LayerStats all_stats;
            if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
                all_stats.bad_rows != 0 ||
                open_shared_output_head(opt, all_rows, &lazy_output_head) != 0) {
                std::fprintf(stderr, "tp_ep lazy diagnostic output-head open failed\n");
                close_shared_output_head(opt, &lazy_output_head);
                return 12;
            }
            std::printf("tp_ep_diagnostic_output_head_lazy_shared\tslots\t%d\t"
                        "vocab\t%d\trows_per_gpu\t%d\toutput_weight_bytes\t%llu\t"
                        "logits_bytes\t%llu\tproxy_hc\t%d\tPASS\n",
                        opt.slots,
                        lazy_output_head.vocab,
                        lazy_output_head.rows_per_gpu,
                        (unsigned long long)lazy_output_head.output_weight_bytes,
                        (unsigned long long)lazy_output_head.logits_bytes,
                        opt.tp_hc_final_expand_gate ? 0 : 1);
            if (report_vram_checkpoint(opt, "after_lazy_output_head") != 0) {
                close_shared_output_head(opt, &lazy_output_head);
                return 14;
            }
            if (nccl_gate_active(opt) && opt.nccl_min_free_mib != 0) {
                (void)report_vram_checkpoint_min_free(
                    opt, "nccl_after_lazy_output_head", opt.nccl_min_free_mib);
            }
            output_head_for_step = &lazy_output_head;
        }
        if (output_head_for_step && output_head_for_step->initialized &&
            shared_rank_buffers && shared_rank_buffers->initialized) {
            OutputHeadRunResult head_result;
            const int head_rc = run_shared_output_head_from_rank_hc(
                opt, output_head_for_step, shared_rank_buffers->ranks, &head_result);
            std::printf("tp_ep_diagnostic_output_head\tsteps\t%d\tslots\t%d\t"
                        "proxy_hc\t%d\ttotal_ms\t%.6f\tgather_ms\t%.6f\t"
                        "prep_ms\t%.6f\tbroadcast_ms\t%.6f\tprojection_ms\t%.6f\t"
                        "projection_kernel_worst_ms\t%.6f\ttop1_ms\t%.6f\t"
                        "device_sync_count\t%d\t"
                        "stream_sync_count\t%d\tevent_sync_count\t%d\t"
                        "first_token\t%u\tfirst_logit\t%.9f\tfinite_bad\t%d\t"
                        "checksum\t%llu\t%s\n",
                        opt.decode_steps, opt.slots,
                        opt.tp_hc_final_expand_gate ? 0 : 1,
                        head_result.total_ms,
                        head_result.gather_ms, head_result.prep_ms,
                        head_result.broadcast_ms, head_result.projection_ms,
                        head_result.projection_kernel_worst_ms, head_result.top1_ms,
                        head_result.device_sync_count,
                        head_result.stream_sync_count,
                        head_result.event_sync_count,
                        head_result.tokens.empty() ? UINT32_MAX : head_result.tokens[0],
                        head_result.logits.empty() ? 0.0f : head_result.logits[0],
                        head_result.finite_bad,
                        (unsigned long long)head_result.checksum,
                        head_rc == 0 && head_result.pass ? "PASS" : "FAIL");
            if (head_rc != 0 || !head_result.pass) {
                if (lazy_output_head.initialized) {
                    close_shared_output_head(opt, &lazy_output_head);
                }
                return head_rc == 0 ? 14 : head_rc;
            }
            if (serving_result) {
                serving_result->diagnostic_output_head = true;
                serving_result->diagnostic_output_head_proxy_hc =
                    !opt.tp_hc_final_expand_gate;
                serving_result->output_head_ms = head_result.total_ms;
                serving_result->output_head_gather_ms = head_result.gather_ms;
                serving_result->output_head_prep_ms = head_result.prep_ms;
                serving_result->output_head_broadcast_ms = head_result.broadcast_ms;
                serving_result->output_head_projection_ms = head_result.projection_ms;
                serving_result->output_head_top1_ms = head_result.top1_ms;
                serving_result->selected_tokens = head_result.tokens;
                serving_result->selected_logits = head_result.logits;
                serving_result->checksum ^= head_result.checksum + 0x0A17EADull;
            }
        }
        if (lazy_output_head.initialized) {
            close_shared_output_head(opt, &lazy_output_head);
            if (report_vram_checkpoint(opt, "after_lazy_output_head_close") != 0) {
                return 14;
            }
            if (nccl_gate_active(opt) && opt.nccl_min_free_mib != 0) {
                (void)report_vram_checkpoint_min_free(
                    opt, "nccl_after_lazy_output_head_close",
                    opt.nccl_min_free_mib);
            }
        }
        if (opt.serving_bench) {
            std::printf("tp_ep_serving_bench\tschema\tds4_v100_tp_ep_serving_bench.v1\t"
                        "requests\t%d\tslots\t%d\tctx\t262144\tgenerated_per_request\t%d\t"
                        "prompt_tokens\t%llu\tgenerated_tokens\t%llu\t"
                        "continuation_tokens\t%llu\t"
                        "first_token_decode_ms\t%.6f\tcontinuation_decode_ms\t%.6f\t"
                        "first_token_wall_ms\t%.6f\tcontinuation_wall_ms\t%.6f\t"
                        "total_decode_ms\t%.6f\ttotal_wall_ms\t%.6f\t"
                        "aggregate_generated_tok_s_decode\t%.6f\t"
                        "aggregate_generated_tok_s_wall\t%.6f\t"
                        "aggregate_continuation_tok_s_decode\t%.6f\t"
                        "aggregate_continuation_tok_s_wall\t%.6f\t"
                        "checksum\t%llu\tPASS\n",
                        opt.slots, opt.slots, opt.decode_steps,
                        (unsigned long long)prompt_tokens,
                        (unsigned long long)generated_tokens,
                        (unsigned long long)continuation_tokens,
                        first_token_decode_ms, continuation_decode_ms,
                        first_token_wall_ms, continuation_wall_ms,
                        sum_decode_ms, wall_ms,
                        generated_tok_s_decode, generated_tok_s_wall,
                        continuation_tok_s_decode, continuation_tok_s_wall,
                        (unsigned long long)checksum);
        }
    }
    if (opt.decode_cudagraph_gate) {
        const int graph_audit_steps = opt.warmup + opt.decode_steps;
        const int total_stream_syncs = sum_cudagraph_rank_stream_syncs +
                                      sum_cudagraph_dense_stream_syncs +
                                      sum_cudagraph_copy_stream_syncs;
        const bool output_head_outside_step =
            shared_output_head && shared_output_head->initialized &&
            shared_rank_buffers && shared_rank_buffers->initialized;
        const bool host_token_dependency =
            output_head_outside_step && serving_result &&
            serving_result->diagnostic_output_head;
        const int helper_host_sync_blocker_classes =
            (opt.tp_hc_current_input_gate && !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_projection_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_compressed_kv_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_state_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_typed_kv_history_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_raw_read_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_attention_output_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.true_ds4_post_attention_ffn_input_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0) +
            (opt.final_hc_carry_gate && opt.tp_hc_final_expand_gate &&
             !opt.decode_cudagraph_gate ? 1 : 0);
        const bool capture_replay_validated =
            sum_cudagraph_capture_attempted > 0 &&
            sum_cudagraph_capture_attempted == sum_cudagraph_capture_succeeded &&
            (!opt.decode_cudagraph_replay_probe_gate ||
             (sum_cudagraph_replay_attempted > 0 &&
              sum_cudagraph_replay_attempted == sum_cudagraph_replay_succeeded));
        const bool capture_eligible =
            capture_replay_validated ||
            (total_stream_syncs == 0 && sum_cudagraph_sync_all_calls == 0 &&
             helper_host_sync_blocker_classes == 0);
        const char *blocker = capture_eligible
            ? "none"
            : (total_stream_syncs != 0 || sum_cudagraph_sync_all_calls != 0
                   ? "host_stream_synchronization"
                   : "helper_host_synchronization");
        std::printf("tp_ep_decode_cudagraph_audit\tsteps\t%d\t"
                    "sync_all_calls\t%d\tevent_barrier_calls\t%d\t"
                    "stream_sync_count\t%d\t"
                    "rank_stream_sync_count\t%d\tdense_stream_sync_count\t%d\t"
                    "copy_stream_sync_count\t%d\toutput_head_outside_step\t%d\t"
                    "host_selected_token_dependency\t%d\t"
                    "helper_host_sync_blocker_classes\t%d\t"
                    "capture_attempted\t%d\tcapture_succeeded\t%d\t"
                    "capture_error_code\t%d\tcapture_error_name\t%s\t"
                    "capture_nodes\t%zu\t"
                    "replay_attempted\t%d\treplay_succeeded\t%d\t"
                    "replay_error_code\t%d\treplay_error_name\t%s\t"
                    "persistent_cache_hits\t%d\t"
                    "persistent_cache_misses\t%d\t"
                    "persistent_invalidations\t%d\t"
                    "persistent_invalidate_layer\t%d\t"
                    "persistent_invalidate_slots\t%d\t"
                    "persistent_invalidate_position\t%d\t"
                    "persistent_invalidate_root_device\t%d\t"
                    "persistent_invalidate_root_stream\t%d\t"
                    "sum_instantiate_ms\t%.6f\tsum_replay_ms\t%.6f\t"
                    "capture_eligible\t%d\tblocker\t%s\n",
                    graph_audit_steps,
                    sum_cudagraph_sync_all_calls,
                    sum_cudagraph_event_barrier_calls,
                    total_stream_syncs,
                    sum_cudagraph_rank_stream_syncs,
                    sum_cudagraph_dense_stream_syncs,
                    sum_cudagraph_copy_stream_syncs,
                    output_head_outside_step ? 1 : 0,
                    host_token_dependency ? 1 : 0,
                    helper_host_sync_blocker_classes,
                    sum_cudagraph_capture_attempted,
                    sum_cudagraph_capture_succeeded,
                    sum_cudagraph_capture_error,
                    cudaGetErrorName((cudaError_t)sum_cudagraph_capture_error),
                    sum_cudagraph_capture_nodes,
                    sum_cudagraph_replay_attempted,
                    sum_cudagraph_replay_succeeded,
                    sum_cudagraph_replay_error,
                    cudaGetErrorName((cudaError_t)sum_cudagraph_replay_error),
                    sum_cudagraph_persistent_cache_hits,
                    sum_cudagraph_persistent_cache_misses,
                    sum_cudagraph_persistent_invalidations,
                    sum_cudagraph_persistent_invalidate_layer,
                    sum_cudagraph_persistent_invalidate_slots,
                    sum_cudagraph_persistent_invalidate_position,
                    sum_cudagraph_persistent_invalidate_root_device,
                    sum_cudagraph_persistent_invalidate_root_stream,
                    sum_cudagraph_instantiate_ms,
                    sum_cudagraph_replay_ms,
                    capture_eligible ? 1 : 0,
                    blocker);
    }
    return 0;
}
