int run_decode_loop(const Options &opt,
                    const std::vector<ContractRow> &rows,
                    RankState ranks[kGpus],
                    const Api &api,
                    ds4_v100_tp_runtime *rt,
                    const DenseF16Cache *cache,
                    const LayerDenseOps *shared_dense_ops,
                    SharedHcControls *shared_hc_controls,
                    TpCudaGraphLayerExec *persistent_graph,
                    DecodeLoopStats *stats) {
    if (opt.decode_steps <= 0) return 0;
    stats->enabled = true;
    stats->ep_return_fp16 = opt.ep_return_fp16;
    stats->fused_compose_sum =
        opt.fuse_compose_sum && !opt.ep_return_fp16 && !opt.compact_route_compose;
    stats->dense_hmma_compose = opt.dense_hmma_compose;
    stats->dense_f16_cublas_compose = opt.dense_f16_cublas_compose;
    stats->dense_f16_cache_compose = opt.dense_f16_cache_compose;
    stats->nccl_reduce_scatter_compose =
        opt.nccl_reduce_scatter_compose_gate &&
        !opt.compact_route_compose && !opt.ep_return_fp16;
    stats->steps = opt.decode_steps;
    stats->slots = opt.slots;
    stats->slot_steps = (uint64_t)opt.decode_steps * (uint64_t)opt.slots;
    ResidentF8Dense attn;
    ResidentF8Dense shared;
    ResidentF8Dense shared_gate;
    ResidentF8Dense shared_up;
    const ResidentF8Dense *attn_op = nullptr;
    const ResidentF8Dense *shared_op = nullptr;
    const ResidentF8Dense *shared_gate_op = nullptr;
    const ResidentF8Dense *shared_up_op = nullptr;
    const std::string attn_tensor = layer_tensor_name(opt.layer, "attn_output_b.weight");
    const std::string shared_tensor = layer_tensor_name(opt.layer, "ffn_down_shexp.weight");
    if (shared_dense_ops) {
        attn_op = &shared_dense_ops->attn;
        shared_op = &shared_dense_ops->shared;
        if (opt.true_shared_ffn_gate) {
            shared_gate_op = &shared_dense_ops->shared_gate;
            shared_up_op = &shared_dense_ops->shared_up;
        }
    } else {
        if (prepare_resident_f8_dense(opt, rows, attn_tensor.c_str(), 1, cache, &attn) != 0 ||
            prepare_resident_f8_dense(opt, rows, shared_tensor.c_str(), 2, cache, &shared) != 0) {
            free_resident_f8_dense(attn, opt);
            free_resident_f8_dense(shared, opt);
            return 1;
        }
        attn_op = &attn;
        shared_op = &shared;
    }
    stats->dense_loaded_bytes = attn_op->loaded_bytes + shared_op->loaded_bytes;
    if (opt.true_shared_ffn_gate) {
        if (!shared_gate_op || !shared_up_op ||
            !shared_gate_op->d_out.size() || !shared_up_op->d_out.size()) {
            return 1;
        }
        stats->dense_loaded_bytes += shared_gate_op->loaded_bytes + shared_up_op->loaded_bytes;
    }
    const uint64_t shard_elems = (uint64_t)opt.slots * (kHidden / kGpus);
    const uint64_t shard_bytes = shard_elems * sizeof(float);
    const uint64_t return_shard_bytes =
        shard_elems * (opt.ep_return_fp16 ? sizeof(__half) : sizeof(float));
    const uint64_t all_contrib_elems = (uint64_t)kGpus * shard_elems;
    const uint64_t all_contrib_bytes = all_contrib_elems * sizeof(float);
    const bool skip_self_copy = opt.skip_self_compose_copy && !opt.ep_return_fp16;
    const bool nccl_reduce_scatter = stats->nccl_reduce_scatter_compose;
    stats->ep_contribution_bytes = all_contrib_bytes * kGpus;
    if (opt.compact_route_compose && !opt.ep_return_fp16) {
        uint64_t compact_return_bytes = 0;
        for (int src = 0; src < kGpus; ++src) {
            const int src_compose_routes = routed_compose_rows(ranks[src], opt);
            compact_return_bytes +=
                (uint64_t)src_compose_routes * (kHidden / kGpus) * sizeof(float) *
                (skip_self_copy ? (kGpus - 1) : kGpus);
        }
        stats->ep_return_bytes = compact_return_bytes;
    } else {
        stats->ep_return_bytes = return_shard_bytes *
                                 (skip_self_copy ? (kGpus * kGpus - kGpus)
                                                 : (kGpus * kGpus));
    }
    if (ensure_compose_buffers(opt, ranks) != 0) {
        if (!shared_dense_ops) {
            free_resident_f8_dense(attn, opt);
            free_resident_f8_dense(shared, opt);
        }
        return 2;
    }
    int cudagraph_audit_sync_all_calls = 0;
    int cudagraph_audit_event_barrier_calls = 0;
    int cudagraph_audit_stream_syncs = 0;
    int cudagraph_audit_dense_stream_syncs = 0;
    int cudagraph_audit_copy_stream_syncs = 0;
    int cudagraph_capture_attempted = 0;
    int cudagraph_capture_succeeded = 0;
    int cudagraph_capture_error = 0;
    size_t cudagraph_capture_nodes = 0;
    int cudagraph_replay_attempted = 0;
    int cudagraph_replay_succeeded = 0;
    int cudagraph_replay_error = 0;
    int cudagraph_persistent_cache_hits = 0;
    int cudagraph_persistent_cache_misses = 0;
    int cudagraph_persistent_invalidations = 0;
    int cudagraph_persistent_invalidate_layer = 0;
    int cudagraph_persistent_invalidate_slots = 0;
    int cudagraph_persistent_invalidate_position = 0;
    int cudagraph_persistent_invalidate_root_device = 0;
    int cudagraph_persistent_invalidate_root_stream = 0;
    double cudagraph_instantiate_ms = 0.0;
    double cudagraph_replay_ms = 0.0;
    double cudagraph_replay_prefix_ms = 0.0;
    bool capture_probe_active = false;
    bool persistent_prefix_only_active = false;
    bool persistent_suffix_only_active = false;
    int current_decode_step = -1;
    auto suffix_stage_is = [&](const char *stage) -> bool {
        return opt.decode_cudagraph_suffix_stage &&
               std::strcmp(opt.decode_cudagraph_suffix_stage, stage) == 0;
    };
    auto sync_all = [&]() {
        if (opt.decode_cudagraph_gate) {
            cudagraph_audit_event_barrier_calls++;
            if (enqueue_cross_gpu_stream_barrier(ranks, true) != 0) {
                return;
            }
            return;
        }
        cudagraph_audit_sync_all_calls++;
        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[p].stream));
            cudagraph_audit_stream_syncs++;
            if (ranks[p].dense_stream) {
                CHECK_CUDA(cudaStreamSynchronize(ranks[p].dense_stream));
                cudagraph_audit_dense_stream_syncs++;
            }
        }
    };
    auto stage_sync_selected = [&](const char *stage) -> bool {
        const char *cfg = opt.decode_cudagraph_stage_sync;
        if (!cfg || !*cfg || !stage || !*stage) return false;
        if (std::strcmp(cfg, "all") == 0) return true;
        const size_t stage_len = std::strlen(stage);
        const char *p = cfg;
        while (*p) {
            while (*p == ',' || *p == ' ' || *p == '\t') ++p;
            const char *start = p;
            while (*p && *p != ',' && *p != ' ' && *p != '\t') ++p;
            const size_t len = (size_t)(p - start);
            if (len == stage_len && std::strncmp(start, stage, stage_len) == 0) {
                return true;
            }
        }
        return false;
    };
    auto sync_after_decode_stage = [&](const char *stage) {
        if (!opt.decode_cudagraph_gate || capture_probe_active ||
            !stage_sync_selected(stage)) {
            return;
        }
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            if (r.stream) {
                CHECK_CUDA(cudaStreamSynchronize(r.stream));
                cudagraph_audit_stream_syncs++;
            }
            if (r.dense_stream && r.dense_stream != r.stream) {
                CHECK_CUDA(cudaStreamSynchronize(r.dense_stream));
                cudagraph_audit_dense_stream_syncs++;
            }
            if (r.copy_stream && r.copy_stream != r.stream) {
                CHECK_CUDA(cudaStreamSynchronize(r.copy_stream));
                cudagraph_audit_copy_stream_syncs++;
            }
            for (int peer = 0; peer < kGpus; ++peer) {
                cudaStream_t copy_stream = r.copy_streams[peer];
                if (copy_stream && copy_stream != r.stream &&
                    copy_stream != r.copy_stream) {
                    CHECK_CUDA(cudaStreamSynchronize(copy_stream));
                    cudagraph_audit_copy_stream_syncs++;
                }
            }
        }
    };
    auto checksum_device_bytes = [&](int device, const void *ptr,
                                     uint64_t bytes,
                                     cudaStream_t stream) -> uint64_t {
        if (!ptr || bytes == 0) return 0;
        unsigned long long *d_sum = nullptr;
        unsigned long long h_sum = 0;
        const int block = 256;
        const int grid = (int)std::min<uint64_t>(
            65535ull, std::max<uint64_t>(1ull, (bytes + block - 1) / block));
        CHECK_CUDA(cudaSetDevice(device));
        CHECK_CUDA(cudaMalloc(&d_sum, sizeof(unsigned long long)));
        CHECK_CUDA(cudaMemsetAsync(d_sum, 0, sizeof(unsigned long long),
                                   stream));
        checksum_bytes_kernel<<<grid, block, 0, stream>>>(
            static_cast<const unsigned char *>(ptr), bytes, d_sum);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpyAsync(&h_sum, d_sum, sizeof(h_sum),
                                   cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaFree(d_sum));
        return (uint64_t)h_sum;
    };
    auto log_stage_tensor = [&](const char *stage, const char *tensor,
                                int rank, const void *ptr, uint64_t bytes,
                                cudaStream_t stream) {
        if (!opt.decode_stage_checksum_gate || capture_probe_active ||
            current_decode_step < 0 || !ptr || bytes == 0) {
            return;
        }
        const uint64_t sum = checksum_device_bytes(ranks[rank].device, ptr,
                                                   bytes, stream);
        std::printf("tp_ep_decode_stage_checksum\tstep\t%d\tlayer\t%d\t"
                    "stage\t%s\ttensor\t%s\trank\t%d\tbytes\t%llu\t"
                    "checksum\t%llu\n",
                    current_decode_step, opt.layer, stage, tensor, rank,
                    (unsigned long long)bytes, (unsigned long long)sum);
    };
    auto log_rank_stage = [&](const char *stage) {
        if (!opt.decode_stage_checksum_gate || capture_probe_active ||
            current_decode_step < 0) {
            return;
        }
        const uint64_t shard_elems =
            (uint64_t)opt.slots * (uint64_t)(kHidden / kGpus);
        const uint64_t full_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
        const uint64_t route_rows = (uint64_t)opt.slots * (uint64_t)opt.top_k;
        const uint64_t route_hidden_elems = route_rows * (uint64_t)kHidden;
        const uint64_t hc_shard_elems = shard_elems * 4ull;
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            cudaStream_t stream = r.stream ? r.stream : (cudaStream_t)0;
            cudaStream_t dense_stream =
                r.dense_stream ? r.dense_stream : stream;
            if (std::strcmp(stage, "hc_current") == 0) {
                log_stage_tensor(stage, "current_shard", rank,
                                 r.d_current_shard,
                                 shard_elems * sizeof(float), stream);
                log_stage_tensor(stage, "current_full", rank,
                                 r.d_current_full,
                                 full_elems * sizeof(float), stream);
                log_stage_tensor(stage, "current_full_rank_major", rank,
                                 r.d_current_full_rank_major,
                                 full_elems * sizeof(float), stream);
                log_stage_tensor(stage, "final_hc_shard", rank,
                                 r.d_final_hc_shard,
                                 hc_shard_elems * sizeof(float), stream);
            } else if (std::strcmp(stage, "attention_projection") == 0) {
                log_stage_tensor(stage, "attn_output_a_full", rank,
                                 r.d_attn_output_a_full,
                                 full_elems * sizeof(float), stream);
            } else if (std::strcmp(stage, "attention_output") == 0) {
                log_stage_tensor(stage, "post_attn_shard", rank,
                                 r.d_post_attn_shard,
                                 shard_elems * sizeof(float), stream);
            } else if (std::strcmp(stage, "post_attention_ffn_input") == 0) {
                log_stage_tensor(stage, "post_attn_rank_major", rank,
                                 r.d_post_attn_full_rank_major,
                                 full_elems * sizeof(float), stream);
                log_stage_tensor(stage, "route_a", rank, r.d_a,
                                 route_hidden_elems * sizeof(__half), stream);
            } else if (std::strcmp(stage, "routed_ffn") == 0) {
                log_stage_tensor(stage, "route_down", rank, r.d_down,
                                 route_hidden_elems * sizeof(__half), stream);
            } else if (std::strcmp(stage, "shared_down") == 0 &&
                       shared_op && rank < (int)shared_op->d_out.size()) {
                log_stage_tensor(stage, "shared_dense_out", rank,
                                 shared_op->d_out[(size_t)rank],
                                 shard_elems * sizeof(float), dense_stream);
            } else if (std::strcmp(stage, "ep_pack") == 0) {
                log_stage_tensor(stage, "ep_contrib_all", rank,
                                 r.d_ep_contrib_all,
                                 (uint64_t)kGpus * shard_elems * sizeof(float),
                                 stream);
            } else if (std::strcmp(stage, "ep_copy") == 0) {
                log_stage_tensor(stage, "ep_remote0", rank, r.d_ep_remote[0],
                                 shard_elems * sizeof(float), stream);
            } else if (std::strcmp(stage, "compose") == 0) {
                log_stage_tensor(stage, "next_hidden", rank,
                                 r.d_next_hidden,
                                 shard_elems * sizeof(float), stream);
                if (attn_op && rank < (int)attn_op->d_out.size()) {
                    log_stage_tensor(stage, "attn_dense_out", rank,
                                     attn_op->d_out[(size_t)rank],
                                     shard_elems * sizeof(float), dense_stream);
                }
                if (shared_op && rank < (int)shared_op->d_out.size()) {
                    log_stage_tensor(stage, "shared_dense_out", rank,
                                     shared_op->d_out[(size_t)rank],
                                     shard_elems * sizeof(float), dense_stream);
                }
            } else if (std::strcmp(stage, "final_hc") == 0) {
                log_stage_tensor(stage, "final_hc_shard", rank,
                                 r.d_final_hc_shard,
                                 hc_shard_elems * sizeof(float), stream);
            } else if (attn_op && rank < (int)attn_op->d_out.size()) {
                log_stage_tensor(stage, "attn_dense_out", rank,
                                 attn_op->d_out[(size_t)rank],
                                 shard_elems * sizeof(float), dense_stream);
            }
        }
    };
    auto log_replay_stage_checksums = [&]() {
        if (!opt.decode_stage_checksum_gate) {
            return;
        }
        const int saved_decode_step = current_decode_step;
        if (current_decode_step < 0) {
            current_decode_step = 0;
        }
        log_rank_stage("hc_current");
        log_rank_stage("attention_projection");
        log_rank_stage("compressed_kv");
        log_rank_stage("attention_state");
        log_rank_stage("typed_history");
        log_rank_stage("raw_read");
        log_rank_stage("attention_output");
        log_rank_stage("post_attention_ffn_input");
        log_rank_stage("routed_ffn");
        log_rank_stage("shared_down");
        log_rank_stage("ep_pack");
        log_rank_stage("ep_copy");
        log_rank_stage("compose");
        log_rank_stage("final_hc");
        current_decode_step = saved_decode_step;
    };
    auto run_final_hc_carry = [&](double *final_hc_ms) -> int {
        if (!opt.final_hc_carry_gate) return 0;
        const auto t_start = std::chrono::steady_clock::now();
        const int block = 256;
        const uint64_t shard_elems =
            (uint64_t)opt.slots * (uint64_t)(kHidden / kGpus);
        const uint64_t hc_shard_elems = shard_elems * 4ull;
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            if (!opt.tp_hc_final_expand_gate || !r.hc_initialized) {
                int grid = (int)((hc_shard_elems + block - 1) / block);
                expand_hidden_to_proxy_hc_shard_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_final_hc_shard, r.d_next_hidden, dst, opt.slots);
                r.hc_initialized = true;
                CHECK_CUDA(cudaGetLastError());
            }
        }
        sync_all();
        if (opt.tp_hc_final_expand_gate) {
            if (!shared_hc_controls || !shared_hc_controls->initialized) {
                return 6;
            }
            if (run_shared_hc_final_expand(opt, shared_hc_controls, ranks,
                                           opt.layer) != 0) {
                return 7;
            }
        }
        if (should_log_reference_hc_window(opt)) {
            for (int dst = 0; dst < kGpus; ++dst) {
                RankState &r = ranks[dst];
                CHECK_CUDA(cudaSetDevice(r.device));
                log_tensor_f32_stats("final_hc_shard", opt.layer, dst,
                                     r.d_final_hc_shard,
                                     (size_t)hc_shard_elems, r.stream);
            }
        }
        log_rank_stage("final_hc");
        sync_after_decode_stage("final_hc");
        if (final_hc_ms) {
            const auto t_stop = std::chrono::steady_clock::now();
            *final_hc_ms +=
                std::chrono::duration<double, std::milli>(
                    t_stop - t_start).count();
        }
        return 0;
    };
    struct CaptureHostRankState {
        float *final_hc_shard = nullptr;
        float *hc_scratch_shard = nullptr;
        bool hc_initialized = false;
        uint32_t attn_rows_written = 0;
        uint32_t index_rows_written = 0;
        bool attn_loaded[kBoundedCompRows] = {};
        bool index_loaded[kBoundedCompRows] = {};
        uint64_t attn_position[kBoundedCompRows] = {};
        uint64_t index_position[kBoundedCompRows] = {};
        uint64_t attn_loaded_position[kBoundedCompRows] = {};
        uint64_t index_loaded_position[kBoundedCompRows] = {};
    };
    auto save_capture_host_state = [&](CaptureHostRankState saved[kGpus]) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CaptureHostRankState &s = saved[rank];
            s.final_hc_shard = r.d_final_hc_shard;
            s.hc_scratch_shard = r.d_hc_scratch_shard;
            s.hc_initialized = r.hc_initialized;
            s.attn_rows_written = r.attn_comp_rows_written_layers[opt.layer];
            s.index_rows_written = r.index_comp_rows_written_layers[opt.layer];
            for (int row = 0; row < kBoundedCompRows; ++row) {
                s.attn_loaded[row] = r.attn_comp_row_loaded_layers[opt.layer][row];
                s.index_loaded[row] = r.index_comp_row_loaded_layers[opt.layer][row];
                s.attn_position[row] = r.attn_comp_row_position_layers[opt.layer][row];
                s.index_position[row] = r.index_comp_row_position_layers[opt.layer][row];
                s.attn_loaded_position[row] =
                    r.attn_comp_row_loaded_position_layers[opt.layer][row];
                s.index_loaded_position[row] =
                    r.index_comp_row_loaded_position_layers[opt.layer][row];
            }
        }
    };
    auto restore_capture_host_state = [&](const CaptureHostRankState saved[kGpus]) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            const CaptureHostRankState &s = saved[rank];
            r.d_final_hc_shard = s.final_hc_shard;
            r.d_hc_scratch_shard = s.hc_scratch_shard;
            r.hc_initialized = s.hc_initialized;
            r.attn_comp_rows_written_layers[opt.layer] = s.attn_rows_written;
            r.index_comp_rows_written_layers[opt.layer] = s.index_rows_written;
            for (int row = 0; row < kBoundedCompRows; ++row) {
                r.attn_comp_row_loaded_layers[opt.layer][row] = s.attn_loaded[row];
                r.index_comp_row_loaded_layers[opt.layer][row] = s.index_loaded[row];
                r.attn_comp_row_position_layers[opt.layer][row] = s.attn_position[row];
                r.index_comp_row_position_layers[opt.layer][row] = s.index_position[row];
                r.attn_comp_row_loaded_position_layers[opt.layer][row] =
                    s.attn_loaded_position[row];
                r.index_comp_row_loaded_position_layers[opt.layer][row] =
                    s.index_loaded_position[row];
            }
        }
    };
    auto print_rank_major_half_input_parity_audit = [&](const char *label) {
        if (!opt.routed_ffn_rank_major_input_parity_gate) return;
        if (!(opt.routed_ffn_rank_major_shared_input_gate ||
              opt.routed_ffn_rank_major_route_input_gate ||
              opt.routed_ffn_rank_major_input_gate)) {
            return;
        }
        unsigned long long shared_gate_bad = 0;
        unsigned long long shared_up_bad = 0;
        unsigned long long route_bad = 0;
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            if (opt.routed_ffn_rank_major_shared_input_gate &&
                shared_gate_op && shared_up_op &&
                shared_gate_op->d_x_half.size() > (size_t)rank &&
                shared_up_op->d_x_half.size() > (size_t)rank &&
                shared_gate_op->d_x_half[(size_t)rank] &&
                shared_up_op->d_x_half[(size_t)rank] && r.d_current_full) {
                HalfInputDiffStats gate_diff = collect_shared_half_input_diff(
                    r, shared_gate_op->d_x_half[(size_t)rank],
                    r.d_current_full, (uint32_t)shared_gate_op->cols,
                    (uint32_t)opt.slots, r.stream);
                log_half_input_diff("graph_shared_gate", opt.layer, rank,
                                    gate_diff);
                shared_gate_bad += gate_diff.mismatches;
                HalfInputDiffStats up_diff = collect_shared_half_input_diff(
                    r, shared_up_op->d_x_half[(size_t)rank],
                    r.d_current_full, (uint32_t)shared_up_op->cols,
                    (uint32_t)opt.slots, r.stream);
                log_half_input_diff("graph_shared_up", opt.layer, rank,
                                    up_diff);
                shared_up_bad += up_diff.mismatches;
            }
            if (opt.routed_ffn_rank_major_route_input_gate &&
                !opt.reference_hc_reduce_gate && r.d_a && r.d_current_full &&
                r.d_route_slots && r.routes > 0) {
                const int compare_routes =
                    opt.post_attention_fixed_capacity_route_plan_gate
                        ? r.route_capacity
                        : r.routes;
                const int *route_total_limit =
                    opt.post_attention_fixed_capacity_route_plan_gate
                        ? r.d_route_totals
                        : nullptr;
                HalfInputDiffStats route_diff =
                    route_total_limit
                        ? collect_route_half_input_diff_limited(
                              r, r.d_a, r.d_current_full, r.d_route_slots,
                              route_total_limit, compare_routes, rank,
                              r.stream)
                        : collect_route_half_input_diff(
                              r, r.d_a, r.d_current_full, r.d_route_slots,
                              compare_routes, r.stream);
                log_half_input_diff("graph_route_a", opt.layer, rank,
                                    route_diff);
                route_bad += route_diff.mismatches;
            }
        }
        std::printf("tp_ep_rank_major_half_input_diff_total\tlabel\t%s\t"
                    "layer\t%d\tshared_gate_mismatches\t%llu\t"
                    "shared_up_mismatches\t%llu\troute_a_mismatches\t%llu\t"
                    "%s\n",
                    label ? label : "unknown", opt.layer,
                    (unsigned long long)shared_gate_bad,
                    (unsigned long long)shared_up_bad,
                    (unsigned long long)route_bad,
                    (shared_gate_bad == 0ull && shared_up_bad == 0ull &&
                     route_bad == 0ull)
                        ? "PASS"
                        : "DIFF");
    };
    auto begin_capture_stream = [](int device, cudaStream_t stream) -> cudaError_t {
        if (!stream) return cudaSuccess;
        cudaError_t rc = cudaSetDevice(device);
        if (rc != cudaSuccess) return rc;
        return cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    };
    auto end_capture_stream = [](int device, cudaStream_t stream,
                                 cudaGraph_t *graph) -> cudaError_t {
        if (!stream) return cudaSuccess;
        cudaError_t rc = cudaSetDevice(device);
        if (rc != cudaSuccess) return rc;
        return cudaStreamEndCapture(stream, graph);
    };
    auto destroy_capture_graphs = [](std::vector<cudaGraph_t> *graphs) {
        for (cudaGraph_t graph : *graphs) {
            if (graph) cudaGraphDestroy(graph);
        }
        graphs->clear();
    };
    auto run_one_step = [&](double *ep_ms,
                            double *dense_ms,
                            double *compose_ms,
                            double *compose_reduce_ms,
                            double *compose_copy_ms,
                            double *compose_final_ms,
                            double *hc_current_input_ms,
                            HcCurrentInputBreakdown *hc_current_breakdown,
                            PreEpPrefixBreakdown *pre_ep_breakdown,
                            double *final_hc_ms) -> int {
        auto t_pre = std::chrono::steady_clock::now();
        if (!persistent_suffix_only_active && opt.tp_hc_current_input_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            if (!shared_hc_controls || !shared_hc_controls->initialized) {
                std::fprintf(stderr, "tp_hc_current_input_failed\tlayer\t%d\treason\tmissing_controls\n",
                             opt.layer);
                return 8;
            }
            const int hc_rc = run_shared_hc_current_input(opt, shared_hc_controls, ranks,
                                                          *attn_op, *shared_op,
                                                          opt.layer,
                                                          capture_probe_active,
                                                          hc_current_breakdown);
            if (hc_rc != 0) {
                std::fprintf(stderr, "tp_hc_current_input_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, hc_rc);
                return 9;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->hc_current_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
            if (opt.decode_cudagraph_hc_current_sync_gate && !capture_probe_active) {
                for (int rank = 0; rank < kGpus; ++rank) {
                    RankState &r = ranks[rank];
                    CHECK_CUDA(cudaSetDevice(r.device));
                    if (r.stream) {
                        CHECK_CUDA(cudaStreamSynchronize(r.stream));
                    }
                    if (r.dense_stream && r.dense_stream != r.stream) {
                        CHECK_CUDA(cudaStreamSynchronize(r.dense_stream));
                    }
                    if (r.copy_stream && r.copy_stream != r.stream) {
                        CHECK_CUDA(cudaStreamSynchronize(r.copy_stream));
                    }
                    for (int peer = 0; peer < kGpus; ++peer) {
                        cudaStream_t copy_stream = r.copy_streams[peer];
                        if (copy_stream && copy_stream != r.stream &&
                            copy_stream != r.copy_stream) {
                            CHECK_CUDA(cudaStreamSynchronize(copy_stream));
                        }
                    }
                }
            }
            log_rank_stage("hc_current");
            sync_after_decode_stage("hc_current");
        }
        if (!persistent_suffix_only_active && opt.true_ds4_attention_projection_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            const int attn_rc = run_true_ds4_attention_projection_prefix(
                opt, shared_hc_controls, shared_dense_ops, ranks, opt.layer);
            if (attn_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_projection_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, attn_rc);
                return 14;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->attention_projection_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
            log_rank_stage("attention_projection");
            sync_after_decode_stage("attention_projection");
        }
        if (!persistent_suffix_only_active && opt.true_ds4_compressed_kv_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            const int comp_rc = run_true_ds4_compressed_kv_projection_gate(
                opt, shared_hc_controls, shared_dense_ops, ranks, rt, opt.layer);
            if (comp_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_compressed_kv_projection_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, comp_rc);
                return 19;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->compressed_kv_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
            log_rank_stage("compressed_kv");
            sync_after_decode_stage("compressed_kv");
        }
        if (!persistent_suffix_only_active && opt.true_ds4_attention_state_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            const int state_rc = run_true_ds4_attention_state_update(
                opt, shared_hc_controls, shared_dense_ops, ranks, rt, opt.layer);
            if (state_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_state_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, state_rc);
                return 15;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->attention_state_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
            log_rank_stage("attention_state");
            sync_after_decode_stage("attention_state");
        }
        if (!persistent_suffix_only_active && opt.true_ds4_attention_typed_kv_history_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            const int history_rc = run_true_ds4_attention_typed_kv_history_load(
                opt, shared_hc_controls, ranks, rt, opt.layer);
            if (history_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_typed_kv_history_failed\t"
                             "layer\t%d\trc\t%d\n",
                             opt.layer, history_rc);
                return 24;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->typed_history_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
            log_rank_stage("typed_history");
            sync_after_decode_stage("typed_history");
        }
        if (!persistent_suffix_only_active && opt.true_ds4_attention_raw_read_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            const int raw_read_rc = opt.true_ds4_attention_raw_window_gate
                ? run_true_ds4_attention_raw_window(
                      opt, shared_hc_controls, shared_dense_ops, ranks, opt.layer)
                : run_true_ds4_attention_raw_read(
                      opt, shared_hc_controls, shared_dense_ops, ranks, opt.layer);
            if (raw_read_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_raw_read_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, raw_read_rc);
                return 16;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->raw_read_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
            log_rank_stage("raw_read");
            sync_after_decode_stage("raw_read");
        }
        if (!persistent_suffix_only_active && opt.true_ds4_attention_output_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            const int output_rc = run_true_ds4_attention_output_projection(
                opt, shared_dense_ops, ranks, opt.layer);
            if (output_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_output_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, output_rc);
                return 17;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->attention_output_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
            log_rank_stage("attention_output");
            sync_after_decode_stage("attention_output");
        }
        if (!persistent_suffix_only_active && opt.true_ds4_post_attention_ffn_input_gate) {
            const auto t_stage = std::chrono::steady_clock::now();
            const int post_rc = run_true_ds4_post_attention_ffn_input(
                opt, shared_hc_controls, shared_dense_ops, ranks, opt.layer,
                capture_probe_active);
            if (post_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_post_attention_ffn_input_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, post_rc);
                return 18;
            }
            if (pre_ep_breakdown) {
                const auto t_done = std::chrono::steady_clock::now();
                pre_ep_breakdown->post_attention_ffn_input_ms +=
                    std::chrono::duration<double, std::milli>(t_done - t_stage).count();
            }
            log_rank_stage("post_attention_ffn_input");
            sync_after_decode_stage("post_attention_ffn_input");
        }
        if (persistent_prefix_only_active) {
            return 0;
        }
        auto t0 = std::chrono::steady_clock::now();
        if (opt.true_shared_ffn_gate &&
            !opt.true_ds4_post_attention_ffn_input_gate) {
            if (!shared_hc_controls || !shared_hc_controls->d_ffn_normed ||
                !shared_gate_op || !shared_up_op) {
                return 10;
            }
            const int fill_rc = fill_shared_ffn_inputs_from_normed(
                opt, shared_hc_controls, *shared_gate_op, *shared_up_op, ranks);
            if (fill_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_shared_ffn_input_failed\tlayer\t%d\trc\t%d\n",
                             opt.layer, fill_rc);
                return 10;
            }
        }
        const bool log_semantic_stats = should_log_routed_semantic_stats(opt);
        if (log_semantic_stats) {
            for (int p = 0; p < kGpus; ++p) {
                RankState &r = ranks[p];
                const size_t elems = (size_t)r.routes * kHidden;
                if (elems > 0) {
                    CHECK_CUDA(cudaSetDevice(r.device));
                    log_route_half_stats("route_input", opt.layer, p,
                                         r.d_a, elems, r.stream);
                }
            }
        }
        for (int p = 0; p < kGpus; ++p) {
            const int gate_rc = run_gate_selected(ranks[p], api, opt);
            if (gate_rc != 0 || run_down(ranks[p], api, opt) != 0) return 1;
        }
        log_rank_stage("routed_ffn");
        sync_after_decode_stage("routed_ffn");
        if (suffix_stage_is("routed_ffn")) {
            const auto t_done = std::chrono::steady_clock::now();
            *ep_ms += std::chrono::duration<double, std::milli>(t_done - t0).count();
            return 0;
        }
        double ep_stage_ms = 0.0;
        double dense_stage_ms = 0.0;
        if (opt.overlap_ep_dense) {
            if (launch_resident_f8_dense(opt, *attn_op, ranks) != 0) {
                return 2;
            }
            if (opt.true_shared_ffn_gate) {
                if (launch_resident_f8_dense(opt, *shared_gate_op, ranks) != 0 ||
                    launch_resident_f8_dense(opt, *shared_up_op, ranks) != 0) {
                    return 2;
                }
                if (!opt.decode_cudagraph_gate &&
                    (opt.layer <= 4 || should_log_reference_hc_window(opt))) {
                    for (int p = 0; p < kGpus; ++p) {
                        CHECK_CUDA(cudaSetDevice(ranks[p].device));
                        log_tensor_f32_stats("shared_gate", opt.layer, p,
                                             shared_gate_op->d_out[(size_t)p],
                                             (size_t)opt.slots * shared_gate_op->rows_per_gpu,
                                             ranks[p].dense_stream ? ranks[p].dense_stream
                                                                   : ranks[p].stream);
                        log_tensor_f32_stats("shared_up", opt.layer, p,
                                             shared_up_op->d_out[(size_t)p],
                                             (size_t)opt.slots * shared_up_op->rows_per_gpu,
                                             ranks[p].dense_stream ? ranks[p].dense_stream
                                                                   : ranks[p].stream);
                    }
                }
            } else if (launch_resident_f8_dense(opt, *shared_op, ranks) != 0) {
                return 2;
            }
            sync_all();
            if (opt.true_shared_ffn_gate) {
                const int swiglu_rc = materialize_shared_swiglu_down_input(
                    opt, *shared_gate_op, *shared_up_op, *shared_op, ranks);
                if (swiglu_rc != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_shared_ffn_swiglu_failed\tlayer\t%d\trc\t%d\n",
                                 opt.layer, swiglu_rc);
                    return 2;
                }
                if (!opt.decode_cudagraph_gate &&
                    (opt.layer <= 4 || should_log_reference_hc_window(opt))) {
                    for (int p = 0; p < kGpus; ++p) {
                        CHECK_CUDA(cudaSetDevice(ranks[p].device));
                        log_tensor_f32_stats("shared_mid", opt.layer, p,
                                             shared_op->d_x[(size_t)p],
                                             (size_t)opt.slots * kMid,
                                             ranks[p].copy_stream ? ranks[p].copy_stream
                                                                  : ranks[p].stream);
                    }
                }
                if (launch_resident_f8_dense_f32_input(opt, *shared_op, ranks) != 0) {
                    return 2;
                }
                sync_all();
                log_rank_stage("shared_down");
                sync_after_decode_stage("shared_down");
                if (!opt.decode_cudagraph_gate &&
                    (opt.layer <= 4 || should_log_reference_hc_window(opt))) {
                    for (int p = 0; p < kGpus; ++p) {
                        CHECK_CUDA(cudaSetDevice(ranks[p].device));
                        log_tensor_f32_stats("shared_down", opt.layer, p,
                                             shared_op->d_out[(size_t)p],
                                             (size_t)opt.slots * shared_op->rows_per_gpu,
                                             ranks[p].dense_stream ? ranks[p].dense_stream
                                                                   : ranks[p].stream);
                    }
                }
            }
            auto t2 = std::chrono::steady_clock::now();
            ep_stage_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
        } else {
            sync_all();
            auto t1 = std::chrono::steady_clock::now();
            if (launch_resident_f8_dense(opt, *attn_op, ranks) != 0) {
                return 2;
            }
            if (opt.true_shared_ffn_gate) {
                if (launch_resident_f8_dense(opt, *shared_gate_op, ranks) != 0 ||
                    launch_resident_f8_dense(opt, *shared_up_op, ranks) != 0) {
                    return 2;
                }
                sync_all();
                if (!opt.decode_cudagraph_gate &&
                    (opt.layer <= 4 || should_log_reference_hc_window(opt))) {
                    for (int p = 0; p < kGpus; ++p) {
                        CHECK_CUDA(cudaSetDevice(ranks[p].device));
                        log_tensor_f32_stats("shared_gate", opt.layer, p,
                                             shared_gate_op->d_out[(size_t)p],
                                             (size_t)opt.slots * shared_gate_op->rows_per_gpu,
                                             ranks[p].dense_stream ? ranks[p].dense_stream
                                                                   : ranks[p].stream);
                        log_tensor_f32_stats("shared_up", opt.layer, p,
                                             shared_up_op->d_out[(size_t)p],
                                             (size_t)opt.slots * shared_up_op->rows_per_gpu,
                                             ranks[p].dense_stream ? ranks[p].dense_stream
                                                                   : ranks[p].stream);
                    }
                }
                const int swiglu_rc = materialize_shared_swiglu_down_input(
                    opt, *shared_gate_op, *shared_up_op, *shared_op, ranks);
                if (swiglu_rc != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_shared_ffn_swiglu_failed\tlayer\t%d\trc\t%d\n",
                                 opt.layer, swiglu_rc);
                    return 2;
                }
                if (!opt.decode_cudagraph_gate &&
                    (opt.layer <= 4 || should_log_reference_hc_window(opt))) {
                    for (int p = 0; p < kGpus; ++p) {
                        CHECK_CUDA(cudaSetDevice(ranks[p].device));
                        log_tensor_f32_stats("shared_mid", opt.layer, p,
                                             shared_op->d_x[(size_t)p],
                                             (size_t)opt.slots * kMid,
                                             ranks[p].copy_stream ? ranks[p].copy_stream
                                                                  : ranks[p].stream);
                    }
                }
                if (launch_resident_f8_dense_f32_input(opt, *shared_op, ranks) != 0) {
                    return 2;
                }
                sync_all();
                log_rank_stage("shared_down");
                sync_after_decode_stage("shared_down");
                if (!opt.decode_cudagraph_gate &&
                    (opt.layer <= 4 || should_log_reference_hc_window(opt))) {
                    for (int p = 0; p < kGpus; ++p) {
                        CHECK_CUDA(cudaSetDevice(ranks[p].device));
                        log_tensor_f32_stats("shared_down", opt.layer, p,
                                             shared_op->d_out[(size_t)p],
                                             (size_t)opt.slots * shared_op->rows_per_gpu,
                                             ranks[p].dense_stream ? ranks[p].dense_stream
                                                                   : ranks[p].stream);
                    }
                }
            } else if (launch_resident_f8_dense(opt, *shared_op, ranks) != 0) {
                return 2;
            }
            sync_all();
            auto t2 = std::chrono::steady_clock::now();
            ep_stage_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            dense_stage_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        if (log_semantic_stats) {
            for (int p = 0; p < kGpus; ++p) {
                RankState &r = ranks[p];
                const size_t gate_up_elems = (size_t)r.routes * kFusedN;
                const size_t gated_elems = (size_t)r.routes * kMid;
                const size_t down_elems = (size_t)r.routes * kHidden;
                if (gate_up_elems > 0) {
                    CHECK_CUDA(cudaSetDevice(r.device));
                    log_route_half_stats("route_gate_up", opt.layer, p,
                                         r.d_gate_up, gate_up_elems, r.stream);
                    log_route_half_stats("route_gated", opt.layer, p,
                                         r.d_gated, gated_elems, r.stream);
                    log_route_half_stats("route_down", opt.layer, p,
                                         r.d_down, down_elems, r.stream);
                }
            }
        }
        auto t2 = std::chrono::steady_clock::now();
        if (suffix_stage_is("dense")) {
            *ep_ms += ep_stage_ms;
            *dense_ms += dense_stage_ms;
            *hc_current_input_ms +=
                std::chrono::duration<double, std::milli>(t0 - t_pre).count();
            return 0;
        }
        const int block = 256;
        const bool compact_route = opt.compact_route_compose &&
                                   !opt.ep_return_fp16 &&
                                   !opt.direct_remote_compose;
        const uint64_t compact_segment_routes =
            opt.compact_moe_decode_gate ? (uint64_t)opt.slots * (uint64_t)opt.top_k
                                        : (uint64_t)opt.slots;
        const uint64_t compact_segment_elems =
            compact_segment_routes * (uint64_t)(kHidden / kGpus);
        const bool use_nccl_reduce_scatter =
            nccl_reduce_scatter && !compact_route && !opt.ep_return_fp16;
        for (int p = 0; p < kGpus; ++p) {
            RankState &r = ranks[p];
            CHECK_CUDA(cudaSetDevice(r.device));
            const int compose_routes = compact_route
                ? routed_compose_rows(r, opt)
                : r.routes;
            const uint64_t route_hidden_elems = (uint64_t)compose_routes * kHidden;
            const int *route_total_limit =
                opt.post_attention_fixed_capacity_route_plan_gate
                    ? r.d_route_totals
                    : nullptr;
            int grid = (int)((route_hidden_elems + block - 1) / block);
            if (compact_route) {
                if (route_hidden_elems > 0) {
                    ep_pack_route_dest_shards_kernel<<<grid, block, 0, r.stream>>>(
                        r.d_ep_contrib_all, r.d_down, r.d_route_weights,
                        route_total_limit, compose_routes,
                        (int)compact_segment_routes, p);
                }
            } else {
                grid = (int)((all_contrib_elems + block - 1) / block);
                zero_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_contrib_all,
                                                              all_contrib_elems);
                CHECK_CUDA(cudaGetLastError());
                grid = (int)((route_hidden_elems + block - 1) / block);
                if (route_hidden_elems > 0) {
                    ep_reduce_all_dest_shards_kernel<<<grid, block, 0, r.stream>>>(
                        r.d_ep_contrib_all, r.d_down, r.d_route_slots,
                        r.d_route_weights, route_total_limit, compose_routes,
                        opt.slots, p);
                }
            }
            CHECK_CUDA(cudaGetLastError());
            if (opt.ep_return_fp16) {
                grid = (int)((all_contrib_elems + block - 1) / block);
                cast_f32_to_half_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_ep_contrib_half_all, r.d_ep_contrib_all, all_contrib_elems);
                CHECK_CUDA(cudaGetLastError());
            }
        }
        sync_all();
        log_rank_stage("ep_pack");
        sync_after_decode_stage("ep_pack");
        auto t_reduce_done = std::chrono::steady_clock::now();
        auto t_copy_done = t_reduce_done;

        bool ep_broadcast_copy_used = false;
        if (use_nccl_reduce_scatter) {
            for (int p = 0; p < kGpus; ++p) {
                if (!ranks[p].compose_nccl_initialized || !ranks[p].compose_nccl) {
                    return 12;
                }
            }
            CHECK_NCCL(ncclGroupStart());
            for (int p = 0; p < kGpus; ++p) {
                RankState &r = ranks[p];
                CHECK_CUDA(cudaSetDevice(r.device));
                CHECK_NCCL(ncclReduceScatter(r.d_ep_contrib_all,
                                             r.d_ep_sum,
                                             (size_t)shard_elems,
                                             ncclFloat,
                                             ncclSum,
                                             r.compose_nccl,
                                             r.stream));
            }
            CHECK_NCCL(ncclGroupEnd());
            sync_all();
            t_reduce_done = std::chrono::steady_clock::now();
            t_copy_done = t_reduce_done;
        } else if (!opt.direct_remote_compose || opt.ep_return_fp16) {
            if (opt.source_copy_schedule && opt.decode_cudagraph_gate) {
                if (opt.ep_return_fp16) return 13;
                for (int dst = 0; dst < kGpus; ++dst) {
                    CHECK_CUDA(cudaSetDevice(ranks[dst].device));
                    for (int src = 0; src < kGpus; ++src) {
                        if (skip_self_copy && src == dst) continue;
                        const float *src_ptr = ranks[src].d_ep_contrib_all +
                                               (uint64_t)dst *
                                                   (compact_route ? compact_segment_elems
                                                                  : shard_elems);
                        const uint64_t copy_elems = compact_route
                            ? (uint64_t)routed_compose_rows(ranks[src], opt) *
                                  (uint64_t)(kHidden / kGpus)
                            : shard_elems;
                        if (copy_elems > 0) {
                            if (compact_route &&
                                opt.post_attention_masked_compact_copy_gate) {
                                const int copy_routes =
                                    routed_compose_rows(ranks[src], opt);
                                copy_compact_active_route_shard_kernel<<<
                                    (unsigned int)((copy_elems +
                                                    (uint64_t)block - 1) /
                                                   (uint64_t)block),
                                    block, 0, ranks[dst].stream>>>(
                                    ranks[dst].d_ep_remote[src], src_ptr,
                                    ranks[dst].d_route_totals, src,
                                    copy_routes, kHidden / kGpus);
                                CHECK_CUDA(cudaGetLastError());
                            } else {
                                enqueue_graph_f32_copy_between_devices(
                                    opt, ranks[dst].device, ranks[src].device,
                                    ranks[dst].d_ep_remote[src], src_ptr,
                                    copy_elems, ranks[dst].stream, block);
                            }
                        }
                    }
                }
            } else if (opt.source_copy_schedule) {
                uint64_t copy_elems_by_src[kGpus] = {};
                for (int src = 0; src < kGpus; ++src) {
                    copy_elems_by_src[src] = compact_route
                        ? (uint64_t)routed_compose_rows(ranks[src], opt) *
                              (uint64_t)(kHidden / kGpus)
                        : shard_elems;
                }
                const uint64_t src_stride_elems =
                    compact_route ? compact_segment_elems : shard_elems;
                if (broadcast_ep_return_slices(
                        ranks, opt.ep_return_fp16, skip_self_copy,
                        src_stride_elems, copy_elems_by_src,
                        opt.ep_return_fp16 ? "serving_ep_copy_half_bcast"
                                           : "serving_ep_copy_float_bcast") != 0) {
                    return 14;
                }
                ep_broadcast_copy_used = true;
                cudagraph_audit_copy_stream_syncs += kGpus;
            } else {
                uint64_t copy_elems_by_src[kGpus] = {};
                for (int src = 0; src < kGpus; ++src) {
                    copy_elems_by_src[src] = compact_route
                        ? (uint64_t)routed_compose_rows(ranks[src], opt) *
                              (uint64_t)(kHidden / kGpus)
                        : shard_elems;
                }
                const uint64_t src_stride_elems =
                    compact_route ? compact_segment_elems : shard_elems;
                if (broadcast_ep_return_slices(
                        ranks, opt.ep_return_fp16, skip_self_copy,
                        src_stride_elems, copy_elems_by_src,
                        opt.ep_return_fp16 ? "serving_ep_copy_half_bcast"
                                           : "serving_ep_copy_float_bcast") != 0) {
                    return 14;
                }
                ep_broadcast_copy_used = true;
                cudagraph_audit_copy_stream_syncs += kGpus;
            }
            t_copy_done = std::chrono::steady_clock::now();
        }
        log_rank_stage("ep_copy");
        sync_after_decode_stage("ep_copy");

        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            if (!opt.decode_cudagraph_gate &&
                opt.copy_event_compose && opt.source_copy_schedule &&
                (!opt.direct_remote_compose || opt.ep_return_fp16) &&
                !ep_broadcast_copy_used) {
                for (int src = 0; src < kGpus; ++src) {
                    if (skip_self_copy && src == dst) continue;
                    CHECK_CUDA(cudaStreamWaitEvent(r.stream, ranks[src].copy_done[dst], 0));
                }
            }
            int grid = (int)((shard_elems + block - 1) / block);
            if (compact_route) {
                const float *r0 = skip_self_copy && dst == 0
                    ? ranks[0].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[0];
                const float *r1 = skip_self_copy && dst == 1
                    ? ranks[1].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[1];
                const float *r2 = skip_self_copy && dst == 2
                    ? ranks[2].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[2];
                const float *r3 = skip_self_copy && dst == 3
                    ? ranks[3].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[3];
                const float *r4 = skip_self_copy && dst == 4
                    ? ranks[4].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[4];
                const float *r5 = skip_self_copy && dst == 5
                    ? ranks[5].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[5];
                const float *r6 = skip_self_copy && dst == 6
                    ? ranks[6].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[6];
                const float *r7 = skip_self_copy && dst == 7
                    ? ranks[7].d_ep_contrib_all + (uint64_t)dst * compact_segment_elems
                    : r.d_ep_remote[7];
                if (opt.compact_moe_decode_gate) {
                    compose_next_hidden_compact8_multi_kernel<<<grid, block, 0, r.stream>>>(
                        r.d_next_hidden, r.d_current_shard, attn_op->d_out[(size_t)dst],
                        shared_op->d_out[(size_t)dst], r0, r1, r2, r3, r4, r5, r6, r7,
                        r.d_route_indices_by_slot[0], r.d_route_indices_by_slot[1],
                        r.d_route_indices_by_slot[2], r.d_route_indices_by_slot[3],
                        r.d_route_indices_by_slot[4], r.d_route_indices_by_slot[5],
                        r.d_route_indices_by_slot[6], r.d_route_indices_by_slot[7],
                        r.d_route_count_by_slot[0], r.d_route_count_by_slot[1],
                        r.d_route_count_by_slot[2], r.d_route_count_by_slot[3],
                        r.d_route_count_by_slot[4], r.d_route_count_by_slot[5],
                        r.d_route_count_by_slot[6], r.d_route_count_by_slot[7],
                        dst, opt.slots, opt.top_k);
                } else {
                    compose_next_hidden_compact8_kernel<<<grid, block, 0, r.stream>>>(
                        r.d_next_hidden, r.d_current_shard, attn_op->d_out[(size_t)dst],
                        shared_op->d_out[(size_t)dst], r0, r1, r2, r3, r4, r5, r6, r7,
                        r.d_route_index_by_slot[0], r.d_route_index_by_slot[1],
                        r.d_route_index_by_slot[2], r.d_route_index_by_slot[3],
                        r.d_route_index_by_slot[4], r.d_route_index_by_slot[5],
                        r.d_route_index_by_slot[6], r.d_route_index_by_slot[7],
                        dst, opt.slots);
                }
            } else if (use_nccl_reduce_scatter) {
                compose_next_hidden_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn_op->d_out[(size_t)dst],
                    shared_op->d_out[(size_t)dst], r.d_ep_sum, dst, opt.slots);
            } else if (stats->fused_compose_sum) {
                const float *r0 = skip_self_copy && dst == 0
                    ? ranks[0].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[0].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[0];
                const float *r1 = skip_self_copy && dst == 1
                    ? ranks[1].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[1].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[1];
                const float *r2 = skip_self_copy && dst == 2
                    ? ranks[2].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[2].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[2];
                const float *r3 = skip_self_copy && dst == 3
                    ? ranks[3].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[3].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[3];
                const float *r4 = skip_self_copy && dst == 4
                    ? ranks[4].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[4].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[4];
                const float *r5 = skip_self_copy && dst == 5
                    ? ranks[5].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[5].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[5];
                const float *r6 = skip_self_copy && dst == 6
                    ? ranks[6].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[6].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[6];
                const float *r7 = skip_self_copy && dst == 7
                    ? ranks[7].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : opt.direct_remote_compose && !opt.ep_return_fp16
                    ? ranks[7].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[7];
                compose_next_hidden_sum8_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn_op->d_out[(size_t)dst],
                    shared_op->d_out[(size_t)dst], r0, r1, r2, r3, r4, r5, r6, r7,
                    dst, opt.slots);
            } else {
                zero_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_sum, shard_elems);
                CHECK_CUDA(cudaGetLastError());
                for (int src = 0; src < kGpus; ++src) {
                    if (opt.ep_return_fp16) {
                        add_half_to_f32_kernel<<<grid, block, 0, r.stream>>>(
                            r.d_ep_sum, r.d_ep_remote_half[src], shard_elems);
                    } else {
                        const float *src_contrib = skip_self_copy && src == dst
                            ? ranks[src].d_ep_contrib_all + (uint64_t)dst * shard_elems
                            : r.d_ep_remote[src];
                        add_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_sum,
                                                                     src_contrib,
                                                                     shard_elems);
                    }
                }
                CHECK_CUDA(cudaGetLastError());
                compose_next_hidden_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn_op->d_out[(size_t)dst],
                    shared_op->d_out[(size_t)dst], r.d_ep_sum, dst, opt.slots);
            }
            CHECK_CUDA(cudaGetLastError());
        }
        sync_all();
        log_rank_stage("compose");
        sync_after_decode_stage("compose");
        if (should_log_reference_hc_window(opt)) {
            for (int dst = 0; dst < kGpus; ++dst) {
                RankState &r = ranks[dst];
                CHECK_CUDA(cudaSetDevice(r.device));
                log_tensor_f32_stats("compose_next_hidden", opt.layer, dst,
                                     r.d_next_hidden, (size_t)shard_elems,
                                     r.stream);
            }
        }
        auto t3 = std::chrono::steady_clock::now();
        if (suffix_stage_is("compose") ||
            (suffix_stage_is("compose_eager_final_hc") &&
             (capture_probe_active || persistent_suffix_only_active))) {
            *ep_ms += ep_stage_ms;
            *dense_ms += dense_stage_ms;
            *compose_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
            *compose_reduce_ms +=
                std::chrono::duration<double, std::milli>(t_reduce_done - t2).count();
            *compose_copy_ms +=
                std::chrono::duration<double, std::milli>(t_copy_done - t_reduce_done).count();
            *compose_final_ms +=
                std::chrono::duration<double, std::milli>(t3 - t_copy_done).count();
            *hc_current_input_ms +=
                std::chrono::duration<double, std::milli>(t0 - t_pre).count();
            return 0;
        }
        double final_stage_ms = 0.0;
        const int final_rc = run_final_hc_carry(&final_stage_ms);
        if (final_rc != 0) return final_rc;
        *ep_ms += ep_stage_ms;
        *dense_ms += dense_stage_ms;
        *compose_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
        *compose_reduce_ms +=
            std::chrono::duration<double, std::milli>(t_reduce_done - t2).count();
        *compose_copy_ms +=
            std::chrono::duration<double, std::milli>(t_copy_done - t_reduce_done).count();
        *compose_final_ms +=
            std::chrono::duration<double, std::milli>(t3 - t_copy_done).count();
        *hc_current_input_ms += std::chrono::duration<double, std::milli>(t0 - t_pre).count();
        *final_hc_ms += final_stage_ms;
        return 0;
    };

    auto attempt_capture_probe = [&](bool replay_after_capture) -> int {
        if (!opt.decode_cudagraph_gate) return 0;
        const int root_device = ranks[0].device;
        cudaStream_t root_stream = ranks[0].stream;
        const bool persistent_enabled =
            replay_after_capture &&
            opt.decode_cudagraph_persistent_replay_gate &&
            persistent_graph;
        const bool persistent_layer_mismatch =
            persistent_enabled && persistent_graph->initialized &&
            persistent_graph->layer != opt.layer;
        const bool persistent_slots_mismatch =
            persistent_enabled && persistent_graph->initialized &&
            persistent_graph->slots != opt.slots;
        const bool persistent_position_mismatch =
            persistent_enabled && persistent_graph->initialized &&
            persistent_graph->position != opt.position;
        const bool persistent_root_device_mismatch =
            persistent_enabled && persistent_graph->initialized &&
            persistent_graph->root_device != root_device;
        const bool persistent_root_stream_mismatch =
            persistent_enabled && persistent_graph->initialized &&
            persistent_graph->root_stream != root_stream;
        if (persistent_layer_mismatch || persistent_slots_mismatch ||
            persistent_position_mismatch || persistent_root_device_mismatch ||
            persistent_root_stream_mismatch) {
            cudagraph_persistent_invalidations++;
            cudagraph_persistent_invalidate_layer +=
                persistent_layer_mismatch ? 1 : 0;
            cudagraph_persistent_invalidate_slots +=
                persistent_slots_mismatch ? 1 : 0;
            cudagraph_persistent_invalidate_position +=
                persistent_position_mismatch ? 1 : 0;
            cudagraph_persistent_invalidate_root_device +=
                persistent_root_device_mismatch ? 1 : 0;
            cudagraph_persistent_invalidate_root_stream +=
                persistent_root_stream_mismatch ? 1 : 0;
            std::printf("tp_ep_decode_cudagraph_persistent_invalidate\t"
                        "layer\t%d\tcached_layer\t%d\tslots\t%d\t"
                        "cached_slots\t%d\tposition\t%llu\t"
                        "cached_position\t%llu\troot_device\t%d\t"
                        "cached_root_device\t%d\tlayer_mismatch\t%d\t"
                        "slots_mismatch\t%d\tposition_mismatch\t%d\t"
                        "root_device_mismatch\t%d\troot_stream_mismatch\t%d\n",
                        opt.layer, persistent_graph->layer, opt.slots,
                        persistent_graph->slots,
                        (unsigned long long)opt.position,
                        (unsigned long long)persistent_graph->position,
                        root_device, persistent_graph->root_device,
                        persistent_layer_mismatch ? 1 : 0,
                        persistent_slots_mismatch ? 1 : 0,
                        persistent_position_mismatch ? 1 : 0,
                        persistent_root_device_mismatch ? 1 : 0,
                        persistent_root_stream_mismatch ? 1 : 0);
            close_tp_cuda_graph_layer_exec(persistent_graph);
        }
        auto run_persistent_dynamic_prefix = [&]() -> int {
            if (!persistent_enabled) return 0;
            double prefix_ep = 0.0;
            double prefix_dense = 0.0;
            double prefix_compose = 0.0;
            double prefix_compose_reduce = 0.0;
            double prefix_compose_copy = 0.0;
            double prefix_compose_final = 0.0;
            double prefix_hc_current = 0.0;
            double prefix_final_hc = 0.0;
            HcCurrentInputBreakdown prefix_hc_breakdown;
            PreEpPrefixBreakdown prefix_pre_ep_breakdown;
            const auto prefix_start = std::chrono::steady_clock::now();
            persistent_prefix_only_active = true;
            const int rc = run_one_step(&prefix_ep, &prefix_dense,
                                        &prefix_compose,
                                        &prefix_compose_reduce,
                                        &prefix_compose_copy,
                                        &prefix_compose_final,
                                        &prefix_hc_current,
                                        &prefix_hc_breakdown,
                                        &prefix_pre_ep_breakdown,
                                        &prefix_final_hc);
            persistent_prefix_only_active = false;
            const auto prefix_stop = std::chrono::steady_clock::now();
            cudagraph_replay_prefix_ms +=
                std::chrono::duration<double, std::milli>(
                    prefix_stop - prefix_start).count();
            for (int rank = 0; rank < kGpus; ++rank) {
                RankState &r = ranks[rank];
                CHECK_CUDA(cudaSetDevice(r.device));
                if (r.stream) {
                    CHECK_CUDA(cudaStreamSynchronize(r.stream));
                }
                if (r.dense_stream && r.dense_stream != r.stream) {
                    CHECK_CUDA(cudaStreamSynchronize(r.dense_stream));
                }
                if (r.copy_stream && r.copy_stream != r.stream) {
                    CHECK_CUDA(cudaStreamSynchronize(r.copy_stream));
                }
                for (int peer = 0; peer < kGpus; ++peer) {
                    cudaStream_t copy_stream = r.copy_streams[peer];
                    if (copy_stream && copy_stream != r.stream &&
                        copy_stream != r.copy_stream) {
                        CHECK_CUDA(cudaStreamSynchronize(copy_stream));
                    }
                }
            }
            return rc;
        };
        if (persistent_enabled && persistent_graph->exec) {
            const int prefix_rc = run_persistent_dynamic_prefix();
            if (prefix_rc != 0) {
                return prefix_rc;
            }
            cudagraph_persistent_cache_hits++;
            cudagraph_replay_attempted++;
            persistent_graph->replays++;
            cudaError_t rc = cudaSuccess;
            cudaEvent_t replay_start = nullptr;
            cudaEvent_t replay_stop = nullptr;
            CHECK_CUDA(cudaSetDevice(root_device));
            CHECK_CUDA(cudaEventCreate(&replay_start));
            CHECK_CUDA(cudaEventCreate(&replay_stop));
            rc = cudaEventRecord(replay_start, root_stream);
            if (rc == cudaSuccess) rc = cudaGraphLaunch(persistent_graph->exec, root_stream);
            if (rc == cudaSuccess) rc = cudaEventRecord(replay_stop, root_stream);
            if (rc == cudaSuccess) rc = cudaEventSynchronize(replay_stop);
            if (rc == cudaSuccess) {
                float elapsed_ms = 0.0f;
                rc = cudaEventElapsedTime(&elapsed_ms, replay_start, replay_stop);
                if (rc == cudaSuccess) {
                    cudagraph_replay_ms = (double)elapsed_ms;
                    persistent_graph->replay_ms += cudagraph_replay_ms;
                    cudagraph_replay_succeeded++;
                }
            }
            CHECK_CUDA(cudaEventDestroy(replay_start));
            CHECK_CUDA(cudaEventDestroy(replay_stop));
            cudagraph_replay_error = rc == cudaSuccess ? 0 : (int)rc;
            cudagraph_capture_nodes = persistent_graph->nodes;
            if (rc != cudaSuccess) persistent_graph->failures++;
            if (rc == cudaSuccess &&
                suffix_stage_is("compose_eager_final_hc")) {
                double eager_final_hc_ms = 0.0;
                const int final_rc = run_final_hc_carry(&eager_final_hc_ms);
                cudagraph_replay_prefix_ms += eager_final_hc_ms;
                if (final_rc != 0) {
                    rc = cudaErrorUnknown;
                    persistent_graph->failures++;
                }
            }
            if (rc == cudaSuccess) {
                log_replay_stage_checksums();
                print_post_attention_route_reuse_audit(
                    opt, ranks, "persistent-cache-hit");
                print_rank_major_half_input_parity_audit(
                    "persistent-cache-hit");
            }
            std::printf("tp_ep_decode_cudagraph_persistent\tlayer\t%d\t"
                        "cache_hit\t1\tcache_miss\t0\tposition\t%llu\t"
                        "cached_position\t%llu\treplay_succeeded\t%d\terror_code\t%d\t"
                        "error_name\t%s\tnodes\t%zu\treplay_ms\t%.6f\t"
                        "captures\t%d\treplays\t%d\n",
                        opt.layer,
                        (unsigned long long)opt.position,
                        (unsigned long long)persistent_graph->position,
                        rc == cudaSuccess ? 1 : 0, (int)rc,
                        cudaGetErrorName(rc), persistent_graph->nodes,
                        cudagraph_replay_ms, persistent_graph->captures,
                        persistent_graph->replays);
            return rc == cudaSuccess ? 0 : 4;
        }
        if (persistent_enabled) {
            cudagraph_persistent_cache_misses++;
        }
        cudagraph_capture_attempted++;
        if (replay_after_capture) cudagraph_replay_attempted++;
        CaptureHostRankState saved[kGpus];
        save_capture_host_state(saved);
        const int prefix_rc = run_persistent_dynamic_prefix();
        if (prefix_rc != 0) {
            restore_capture_host_state(saved);
            return prefix_rc;
        }

        cudaError_t first_error = cudaSuccess;
        const char *phase = "begin";
        struct CaptureStream {
            int device = 0;
            cudaStream_t stream = nullptr;
        };
        std::vector<CaptureStream> streams;
        auto add_stream = [&](int device, cudaStream_t stream) {
            if (!stream) return;
            for (const CaptureStream &s : streams) {
                if (s.device == device && s.stream == stream) return;
            }
            streams.push_back(CaptureStream{device, stream});
        };
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            add_stream(r.device, r.stream);
            add_stream(r.device, r.dense_stream);
            add_stream(r.device, r.copy_stream);
            for (int q = 0; q < kGpus; ++q) {
                add_stream(r.device, r.copy_streams[q]);
            }
        }
        cudaEvent_t capture_seed = nullptr;
        CHECK_CUDA(cudaSetDevice(root_device));
        CHECK_CUDA(cudaEventCreateWithFlags(&capture_seed, cudaEventDisableTiming));
        std::vector<cudaEvent_t> capture_join_events(streams.size(), nullptr);
        for (size_t i = 0; i < streams.size(); ++i) {
            CHECK_CUDA(cudaSetDevice(streams[i].device));
            CHECK_CUDA(cudaEventCreateWithFlags(&capture_join_events[i],
                                                cudaEventDisableTiming));
        }
        cudaError_t rc = begin_capture_stream(root_device, root_stream);
        if (rc != cudaSuccess) first_error = rc;
        bool capture_begun = first_error == cudaSuccess;
        if (first_error == cudaSuccess) {
            phase = "join";
            rc = cudaEventRecord(capture_seed, root_stream);
            if (rc != cudaSuccess) first_error = rc;
            for (const CaptureStream &s : streams) {
                if (first_error != cudaSuccess) break;
                if (s.device == root_device && s.stream == root_stream) continue;
                rc = cudaSetDevice(s.device);
                if (rc != cudaSuccess) {
                    first_error = rc;
                    break;
                }
                rc = cudaStreamWaitEvent(s.stream, capture_seed, 0);
                if (rc != cudaSuccess) first_error = rc;
            }
        }

        int step_rc = 0;
        double cap_ep = 0.0;
        double cap_dense = 0.0;
        double cap_compose = 0.0;
        double cap_compose_reduce = 0.0;
        double cap_compose_copy = 0.0;
        double cap_compose_final = 0.0;
        double cap_hc_current = 0.0;
        double cap_final_hc = 0.0;
        if (first_error == cudaSuccess) {
            phase = "enqueue";
            capture_probe_active = true;
            persistent_suffix_only_active = persistent_enabled;
            step_rc = run_one_step(&cap_ep, &cap_dense, &cap_compose,
                                   &cap_compose_reduce, &cap_compose_copy,
                                   &cap_compose_final, &cap_hc_current,
                                   nullptr, nullptr, &cap_final_hc);
            persistent_suffix_only_active = false;
            capture_probe_active = false;
            if (step_rc != 0) {
                first_error = cudaErrorUnknown;
            }
        }
        if (first_error == cudaSuccess) {
            phase = "rejoin";
            for (size_t i = 0; i < streams.size(); ++i) {
                const CaptureStream &s = streams[i];
                if (s.device == root_device && s.stream == root_stream) continue;
                rc = cudaSetDevice(s.device);
                if (rc != cudaSuccess) {
                    first_error = rc;
                    break;
                }
                rc = cudaEventRecord(capture_join_events[i], s.stream);
                if (rc != cudaSuccess) {
                    first_error = rc;
                    break;
                }
                rc = cudaSetDevice(root_device);
                if (rc != cudaSuccess) {
                    first_error = rc;
                    break;
                }
                rc = cudaStreamWaitEvent(root_stream, capture_join_events[i], 0);
                if (rc != cudaSuccess) {
                    first_error = rc;
                    break;
                }
            }
        }

        phase = first_error == cudaSuccess ? "end" : phase;
        std::vector<cudaGraph_t> graphs;
        size_t node_count = 0;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        bool replay_success = false;
        if (capture_begun) {
            rc = end_capture_stream(root_device, root_stream, &graph);
            if (rc == cudaSuccess && graph) {
                size_t graph_nodes = 0;
                cudaError_t count_rc = cudaGraphGetNodes(graph, nullptr,
                                                         &graph_nodes);
                if (count_rc == cudaSuccess) node_count += graph_nodes;
                graphs.push_back(graph);
            } else if (first_error == cudaSuccess) {
                first_error = rc;
            }
        }
        if (first_error == cudaSuccess && step_rc == 0 && replay_after_capture &&
            graph) {
            phase = "instantiate";
            const auto instantiate_start = std::chrono::steady_clock::now();
            rc = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
            const auto instantiate_stop = std::chrono::steady_clock::now();
            cudagraph_instantiate_ms =
                std::chrono::duration<double, std::milli>(
                    instantiate_stop - instantiate_start).count();
            if (rc != cudaSuccess) {
                first_error = rc;
            } else if (persistent_enabled && persistent_graph) {
                persistent_graph->initialized = true;
                persistent_graph->layer = opt.layer;
                persistent_graph->slots = opt.slots;
                persistent_graph->position = opt.position;
                persistent_graph->root_device = root_device;
                persistent_graph->root_stream = root_stream;
                persistent_graph->graph = graph;
                persistent_graph->exec = graph_exec;
                persistent_graph->nodes = node_count;
                persistent_graph->captures++;
                persistent_graph->instantiates++;
                persistent_graph->instantiate_ms += cudagraph_instantiate_ms;
                graph = nullptr;
                graph_exec = nullptr;
                graphs.clear();
            }
        }
        if (first_error == cudaSuccess && step_rc == 0 && replay_after_capture &&
            (graph_exec || (persistent_enabled && persistent_graph && persistent_graph->exec))) {
            phase = "launch";
            cudaGraphExec_t launch_exec =
                graph_exec ? graph_exec : persistent_graph->exec;
            cudaEvent_t replay_start = nullptr;
            cudaEvent_t replay_stop = nullptr;
            CHECK_CUDA(cudaSetDevice(root_device));
            CHECK_CUDA(cudaEventCreate(&replay_start));
            CHECK_CUDA(cudaEventCreate(&replay_stop));
            rc = cudaEventRecord(replay_start, root_stream);
            for (int launch = 0; rc == cudaSuccess && launch < opt.decode_steps;
                 ++launch) {
                rc = cudaGraphLaunch(launch_exec, root_stream);
            }
            if (rc == cudaSuccess) rc = cudaEventRecord(replay_stop, root_stream);
            if (rc == cudaSuccess) rc = cudaEventSynchronize(replay_stop);
            if (rc == cudaSuccess) {
                float elapsed_ms = 0.0f;
                rc = cudaEventElapsedTime(&elapsed_ms, replay_start, replay_stop);
                if (rc == cudaSuccess) {
                    cudagraph_replay_ms = (double)elapsed_ms;
                    replay_success = true;
                    cudagraph_replay_succeeded++;
                    if (persistent_enabled && persistent_graph && persistent_graph->exec) {
                        persistent_graph->replays += opt.decode_steps;
                        persistent_graph->replay_ms += cudagraph_replay_ms;
                    }
                }
            }
            CHECK_CUDA(cudaEventDestroy(replay_start));
            CHECK_CUDA(cudaEventDestroy(replay_stop));
            if (rc != cudaSuccess) {
                if (persistent_enabled && persistent_graph) {
                    persistent_graph->failures++;
                }
                first_error = rc;
            } else {
                if (suffix_stage_is("compose_eager_final_hc")) {
                    double eager_final_hc_ms = 0.0;
                    const int final_rc = run_final_hc_carry(&eager_final_hc_ms);
                    cudagraph_replay_prefix_ms += eager_final_hc_ms;
                    if (final_rc != 0) {
                        first_error = cudaErrorUnknown;
                        replay_success = false;
                        if (persistent_enabled && persistent_graph) {
                            persistent_graph->failures++;
                        }
                    }
                }
            }
            if (first_error == cudaSuccess && replay_success) {
                log_replay_stage_checksums();
                print_post_attention_route_reuse_audit(
                    opt, ranks,
                    persistent_enabled ? "persistent-capture-replay"
                                       : "capture-replay");
                print_rank_major_half_input_parity_audit(
                    persistent_enabled ? "persistent-capture-replay"
                                       : "capture-replay");
            }
        }
        if (!replay_after_capture || !replay_success) {
            restore_capture_host_state(saved);
        }
        CHECK_CUDA(cudaSetDevice(root_device));
        CHECK_CUDA(cudaEventDestroy(capture_seed));
        for (size_t i = 0; i < capture_join_events.size(); ++i) {
            if (!capture_join_events[i]) continue;
            CHECK_CUDA(cudaSetDevice(streams[i].device));
            CHECK_CUDA(cudaEventDestroy(capture_join_events[i]));
        }

        cudagraph_capture_error = (int)first_error;
        cudagraph_capture_nodes = node_count;
        if (replay_after_capture) {
            cudagraph_replay_error = replay_success ? 0 : (int)first_error;
        }
        if (first_error == cudaSuccess && step_rc == 0) {
            cudagraph_capture_succeeded++;
        }
        std::printf("tp_ep_decode_cudagraph_capture\tlayer\t%d\tstreams\t%zu\t"
                    "roots\t1\tattempted\t1\tsucceeded\t%d\terror_code\t%d\t"
                    "error_name\t%s\tphase\t%s\tnodes\t%zu\tstep_rc\t%d\t"
                    "replay_probe\t%d\treplay_succeeded\t%d\t"
                    "persistent\t%d\tcache_hit\t0\tcache_miss\t%d\t"
                    "persistent_invalidations\t%d\t"
                    "invalidate_layer\t%d\tinvalidate_slots\t%d\t"
                    "invalidate_position\t%d\tinvalidate_root_device\t%d\t"
                    "invalidate_root_stream\t%d\t"
                    "replay_launches\t%d\t"
                    "instantiate_ms\t%.6f\treplay_ms\t%.6f\n",
                    opt.layer, streams.size(),
                    first_error == cudaSuccess && step_rc == 0 ? 1 : 0,
                    (int)first_error, cudaGetErrorName(first_error), phase,
                    node_count, step_rc,
                    replay_after_capture ? 1 : 0,
                    replay_success ? 1 : 0,
                    persistent_enabled ? 1 : 0,
                    persistent_enabled ? 1 : 0,
                    cudagraph_persistent_invalidations,
                    cudagraph_persistent_invalidate_layer,
                    cudagraph_persistent_invalidate_slots,
                    cudagraph_persistent_invalidate_position,
                    cudagraph_persistent_invalidate_root_device,
                    cudagraph_persistent_invalidate_root_stream,
                    replay_after_capture ? opt.decode_steps : 0,
                    cudagraph_instantiate_ms, cudagraph_replay_ms);
        if (graph_exec) {
            CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
        }
        destroy_capture_graphs(&graphs);
        return 0;
    };

    double warm_ep = 0.0;
    double warm_dense = 0.0;
    double warm_compose = 0.0;
    double warm_compose_reduce = 0.0;
    double warm_compose_copy = 0.0;
    double warm_compose_final = 0.0;
    double warm_hc_current_input = 0.0;
    double warm_final_hc = 0.0;
    for (int i = 0; i < opt.warmup; ++i) {
        current_decode_step = -1;
        if (run_one_step(&warm_ep, &warm_dense, &warm_compose,
                         &warm_compose_reduce, &warm_compose_copy,
                         &warm_compose_final, &warm_hc_current_input,
                         nullptr, nullptr,
                         &warm_final_hc) != 0) {
            if (!shared_dense_ops) {
                free_resident_f8_dense(attn, opt);
                free_resident_f8_dense(shared, opt);
            }
            return 3;
        }
    }

    double ep_ms = 0.0;
    double dense_ms = 0.0;
    double compose_ms = 0.0;
    double compose_reduce_ms = 0.0;
    double compose_copy_ms = 0.0;
    double compose_final_ms = 0.0;
    double hc_current_input_ms = 0.0;
    HcCurrentInputBreakdown hc_current_breakdown;
    PreEpPrefixBreakdown pre_ep_breakdown;
    double final_hc_ms = 0.0;
    bool used_graph_replay = false;
    const auto start = std::chrono::steady_clock::now();
    if (opt.decode_cudagraph_replay_probe_gate) {
        std::printf("tp_ep_decode_cudagraph_replay_probe_start\tlayer\t%d\t"
                    "steps\t%d\tslots\t%d\n",
                    opt.layer, opt.decode_steps, opt.slots);
        const int cap_rc = attempt_capture_probe(true);
        if (cap_rc != 0 || cudagraph_replay_succeeded == 0) {
            if (!shared_dense_ops) {
                free_resident_f8_dense(attn, opt);
                free_resident_f8_dense(shared, opt);
            }
            return 4;
        }
        used_graph_replay = true;
    } else {
        for (int i = 0; i < opt.decode_steps; ++i) {
            current_decode_step = i;
            if (run_one_step(&ep_ms, &dense_ms, &compose_ms,
                             &compose_reduce_ms, &compose_copy_ms,
                             &compose_final_ms, &hc_current_input_ms,
                             &hc_current_breakdown, &pre_ep_breakdown,
                             &final_hc_ms) != 0) {
                if (!shared_dense_ops) {
                    free_resident_f8_dense(attn, opt);
                    free_resident_f8_dense(shared, opt);
                }
                return 4;
            }
        }
        current_decode_step = -1;
    }
    const auto stop = std::chrono::steady_clock::now();
    stats->total_ms = used_graph_replay
        ? cudagraph_replay_ms + cudagraph_replay_prefix_ms
        : std::chrono::duration<double, std::milli>(stop - start).count();
    stats->ms_per_step = stats->total_ms / (double)opt.decode_steps;
    stats->tok_s = stats->total_ms > 0.0
        ? (double)stats->slot_steps * 1000.0 / stats->total_ms
        : 0.0;
    stats->ep_ms_per_step = ep_ms / (double)opt.decode_steps;
    stats->dense_ms_per_step = dense_ms / (double)opt.decode_steps;
    stats->compose_ms_per_step = compose_ms / (double)opt.decode_steps;
    stats->compose_reduce_ms_per_step = compose_reduce_ms / (double)opt.decode_steps;
    stats->compose_copy_ms_per_step = compose_copy_ms / (double)opt.decode_steps;
    stats->compose_final_ms_per_step = compose_final_ms / (double)opt.decode_steps;
    stats->hc_current_input_ms_per_step = hc_current_input_ms / (double)opt.decode_steps;
    stats->hc_current_seed_ms_per_step =
        hc_current_breakdown.seed_ms / (double)opt.decode_steps;
    stats->hc_current_attn_mix_ms_per_step =
        hc_current_breakdown.attn_mix_ms / (double)opt.decode_steps;
    stats->hc_current_split_ms_per_step =
        hc_current_breakdown.split_ms / (double)opt.decode_steps;
    stats->hc_current_gather_ms_per_step =
        hc_current_breakdown.gather_ms / (double)opt.decode_steps;
    stats->hc_current_ffn_router_ms_per_step =
        hc_current_breakdown.ffn_router_ms / (double)opt.decode_steps;
    stats->hc_current_ffn_norm_ms_per_step =
        hc_current_breakdown.ffn_norm_ms / (double)opt.decode_steps;
    stats->hc_current_router_select_ms_per_step =
        hc_current_breakdown.router_select_ms / (double)opt.decode_steps;
    stats->hc_current_router_d2h_ms_per_step =
        hc_current_breakdown.router_d2h_ms / (double)opt.decode_steps;
    stats->hc_current_route_upload_ms_per_step =
        hc_current_breakdown.route_upload_ms / (double)opt.decode_steps;
    stats->hc_current_fill_pack_ms_per_step =
        hc_current_breakdown.fill_pack_ms / (double)opt.decode_steps;
    stats->pre_ep_hc_current_ms_per_step =
        pre_ep_breakdown.hc_current_ms / (double)opt.decode_steps;
    stats->pre_ep_attention_projection_ms_per_step =
        pre_ep_breakdown.attention_projection_ms / (double)opt.decode_steps;
    stats->pre_ep_compressed_kv_ms_per_step =
        pre_ep_breakdown.compressed_kv_ms / (double)opt.decode_steps;
    stats->pre_ep_attention_state_ms_per_step =
        pre_ep_breakdown.attention_state_ms / (double)opt.decode_steps;
    stats->pre_ep_typed_history_ms_per_step =
        pre_ep_breakdown.typed_history_ms / (double)opt.decode_steps;
    stats->pre_ep_raw_read_ms_per_step =
        pre_ep_breakdown.raw_read_ms / (double)opt.decode_steps;
    stats->pre_ep_attention_output_ms_per_step =
        pre_ep_breakdown.attention_output_ms / (double)opt.decode_steps;
    stats->pre_ep_post_attention_ffn_input_ms_per_step =
        pre_ep_breakdown.post_attention_ffn_input_ms / (double)opt.decode_steps;
    stats->final_hc_ms_per_step = final_hc_ms / (double)opt.decode_steps;
    stats->cudagraph_sync_all_calls = cudagraph_audit_sync_all_calls;
    stats->cudagraph_event_barrier_calls =
        cudagraph_audit_event_barrier_calls;
    stats->cudagraph_rank_stream_syncs = cudagraph_audit_stream_syncs;
    stats->cudagraph_dense_stream_syncs = cudagraph_audit_dense_stream_syncs;
    stats->cudagraph_copy_stream_syncs = cudagraph_audit_copy_stream_syncs;

    if (opt.skip_decode_checksum) {
        stats->checksum = 0xD54D0000ull ^
                          ((uint64_t)(opt.layer + 1) * 1000003ull) ^
                          ((uint64_t)(opt.position + 1) * 9176ull) ^
                          ((uint64_t)opt.slots * 65537ull);
    } else if (suffix_stage_is("routed_ffn")) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            const uint64_t elems = (uint64_t)r.route_capacity * (uint64_t)kHidden;
            if (!r.d_down || elems == 0) {
                stats->pass = false;
                continue;
            }
            std::vector<__half> host((size_t)elems);
            CHECK_CUDA(cudaMemcpy(host.data(), r.d_down,
                                  (size_t)elems * sizeof(__half),
                                  cudaMemcpyDeviceToHost));
            for (uint64_t i = 0; i < elems; ++i) {
                const float v = __half2float(host[(size_t)i]);
                if (!std::isfinite(v)) {
                    stats->finite_bad++;
                    stats->pass = false;
                }
                uint16_t bits = 0;
                std::memcpy(&bits, &host[(size_t)i], sizeof(bits));
                stats->checksum ^=
                    (uint64_t)bits + (uint64_t)(rank + 1) * 1700003ull +
                    i * 6361ull;
            }
        }
    } else if (suffix_stage_is("dense")) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            const uint64_t attn_elems =
                (uint64_t)opt.slots * (uint64_t)attn_op->rows_per_gpu;
            const uint64_t shared_elems =
                (uint64_t)opt.slots * (uint64_t)shared_op->rows_per_gpu;
            const float *ptrs[2] = {
                attn_op->d_out[(size_t)rank],
                shared_op->d_out[(size_t)rank],
            };
            const uint64_t elems[2] = {attn_elems, shared_elems};
            for (int tensor = 0; tensor < 2; ++tensor) {
                if (!ptrs[tensor] || elems[tensor] == 0) {
                    stats->pass = false;
                    continue;
                }
                std::vector<float> host((size_t)elems[tensor]);
                CHECK_CUDA(cudaMemcpy(host.data(), ptrs[tensor],
                                      (size_t)elems[tensor] * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                for (uint64_t i = 0; i < elems[tensor]; ++i) {
                    const float v = host[(size_t)i];
                    if (!std::isfinite(v)) {
                        stats->finite_bad++;
                        stats->pass = false;
                    }
                    uint32_t bits = 0;
                    std::memcpy(&bits, &v, sizeof(bits));
                    stats->checksum ^=
                        (uint64_t)bits +
                        (uint64_t)(rank + 1) * 1900003ull +
                        (uint64_t)(tensor + 1) * 9901ull + i * 7411ull;
                }
            }
        }
    } else {
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            std::vector<float> host((size_t)shard_elems);
            CHECK_CUDA(cudaMemcpy(host.data(), r.d_next_hidden, (size_t)shard_bytes,
                                  cudaMemcpyDeviceToHost));
            for (uint64_t i = 0; i < shard_elems; ++i) {
                const float v = host[(size_t)i];
                if (!std::isfinite(v)) {
                    stats->finite_bad++;
                    stats->pass = false;
                }
                uint32_t bits = 0;
                std::memcpy(&bits, &v, sizeof(bits));
                stats->checksum ^=
                    (uint64_t)bits + (uint64_t)(dst + 1) * 2000003ull + i * 7907ull;
            }
        }
    }
    if (stats->checksum == 0 || stats->finite_bad != 0) stats->pass = false;
    const bool checksum_final_hc =
        opt.final_hc_carry_gate &&
        !suffix_stage_is("routed_ffn") &&
        !suffix_stage_is("dense") &&
        !suffix_stage_is("compose");
    if (checksum_final_hc) {
        uint64_t hc_checksum = 0;
        const uint64_t hc_shard_elems = shard_elems * 4ull;
        const uint64_t hc_shard_bytes = hc_shard_elems * sizeof(float);
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            if (!r.d_final_hc_shard) {
                stats->pass = false;
                continue;
            }
            std::vector<float> host((size_t)hc_shard_elems);
            CHECK_CUDA(cudaMemcpy(host.data(), r.d_final_hc_shard,
                                  (size_t)hc_shard_bytes,
                                  cudaMemcpyDeviceToHost));
            for (uint64_t i = 0; i < hc_shard_elems; ++i) {
                const float v = host[(size_t)i];
                if (!std::isfinite(v)) {
                    stats->finite_bad++;
                    stats->pass = false;
                }
                uint32_t bits = 0;
                std::memcpy(&bits, &v, sizeof(bits));
                hc_checksum ^=
                    (uint64_t)bits + (uint64_t)(dst + 1) * 3000017ull + i * 8191ull;
            }
        }
        if (hc_checksum == 0) stats->pass = false;
        stats->checksum ^= hc_checksum + 0xF17A1C00ull;
    }
    if (stats->checksum == 0 || stats->finite_bad != 0) stats->pass = false;

    if (opt.decode_cudagraph_gate && !opt.decode_cudagraph_replay_probe_gate &&
        stats->pass) {
        const int cap_rc = attempt_capture_probe(false);
        if (cap_rc != 0) {
            stats->pass = false;
        }
    }
    stats->cudagraph_capture_attempted = cudagraph_capture_attempted;
    stats->cudagraph_capture_succeeded = cudagraph_capture_succeeded;
    stats->cudagraph_capture_error = cudagraph_capture_error;
    stats->cudagraph_capture_nodes = cudagraph_capture_nodes;
    stats->cudagraph_replay_attempted = cudagraph_replay_attempted;
    stats->cudagraph_replay_succeeded = cudagraph_replay_succeeded;
    stats->cudagraph_replay_error = cudagraph_replay_error;
    stats->cudagraph_persistent_cache_hits = cudagraph_persistent_cache_hits;
    stats->cudagraph_persistent_cache_misses = cudagraph_persistent_cache_misses;
    stats->cudagraph_persistent_invalidations =
        cudagraph_persistent_invalidations;
    stats->cudagraph_persistent_invalidate_layer =
        cudagraph_persistent_invalidate_layer;
    stats->cudagraph_persistent_invalidate_slots =
        cudagraph_persistent_invalidate_slots;
    stats->cudagraph_persistent_invalidate_position =
        cudagraph_persistent_invalidate_position;
    stats->cudagraph_persistent_invalidate_root_device =
        cudagraph_persistent_invalidate_root_device;
    stats->cudagraph_persistent_invalidate_root_stream =
        cudagraph_persistent_invalidate_root_stream;
    stats->cudagraph_instantiate_ms = cudagraph_instantiate_ms;
    stats->cudagraph_replay_ms = cudagraph_replay_ms;

    if (!shared_dense_ops) {
        free_resident_f8_dense(attn, opt);
        free_resident_f8_dense(shared, opt);
    }
    return stats->pass ? 0 : 5;
}
