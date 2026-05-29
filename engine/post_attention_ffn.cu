int run_true_ds4_post_attention_ffn_input(const Options &opt,
                                          SharedHcControls *hc,
                                          const LayerDenseOps *ops,
                                          RankState ranks[kGpus],
                                          int layer,
                                          bool reuse_model_router_route_plan) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        hc->slots != opt.slots || layer < 0 || layer >= 43) {
        return 1;
    }
    if (ops->attn.rows_per_gpu != kHidden / kGpus ||
        ops->shared_gate.cols != kHidden ||
        ops->shared_up.cols != kHidden ||
        ops->shared_gate.rows_per_gpu != kMid / kGpus ||
        ops->shared_up.rows_per_gpu != kMid / kGpus) {
        return 2;
    }
    if (!hc->d_ffn_norm_weight[layer]) {
        return 3;
    }

    const auto start = std::chrono::steady_clock::now();
    const int block = 256;
    const uint64_t shard_elems =
        (uint64_t)opt.slots * (uint64_t)(kHidden / kGpus);
    const uint64_t full_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    const bool rank_major_shared_input =
        (opt.routed_ffn_rank_major_input_gate ||
         opt.routed_ffn_rank_major_shared_input_gate) &&
        opt.tp_hc_current_input_nccl_allgather_gate;
    const bool rank_major_route_input =
        (opt.routed_ffn_rank_major_input_gate ||
         opt.routed_ffn_rank_major_route_input_gate) &&
        opt.tp_hc_current_input_nccl_allgather_gate;
    const bool rank_major_input =
        rank_major_shared_input || rank_major_route_input;
    const bool post_attention_route_reuse_audit =
        opt.post_attention_route_reuse_audit_gate &&
        opt.model_router_routes &&
        reuse_model_router_route_plan;
    const bool post_attention_fixed_capacity_route_plan =
        opt.post_attention_fixed_capacity_route_plan_gate &&
        opt.model_router_routes &&
        reuse_model_router_route_plan;
    cudaStream_t control_stream =
        graph_event_order ? ranks[0].stream : (cudaStream_t)0;
    auto sync_control_device = [&]() {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        if (graph_event_order && reuse_model_router_route_plan) return;
        if (graph_event_order) {
            CHECK_CUDA(cudaStreamSynchronize(control_stream));
        } else {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    };

    TensorF32Stats post_shard_stats;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_current_shard || !r.d_post_attn_shard ||
            !ops->attn.d_out[(size_t)rank]) {
            return 4;
        }
        add_current_attention_shard_kernel<<<
            (unsigned int)((shard_elems + block - 1) / block), block, 0,
            r.stream>>>(r.d_post_attn_shard, r.d_current_shard,
                         ops->attn.d_out[(size_t)rank], shard_elems);
        CHECK_CUDA(cudaGetLastError());
    }
    if (graph_event_order) {
        if (enqueue_control_wait_after_rank_streams(opt, ranks,
                                                    control_stream) != 0) {
            return 9;
        }
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            if (!opt.true_ds4_semantic_skip_stats_gate) {
                merge_tensor_stats(
                    &post_shard_stats,
                    collect_tensor_f32_stats(ranks[rank].d_post_attn_shard,
                                             (size_t)shard_elems,
                                             ranks[rank].stream));
            }
        }
    }
    if (graph_event_order && !opt.true_ds4_semantic_skip_stats_gate &&
        !reuse_model_router_route_plan) {
        for (int rank = 0; rank < kGpus; ++rank) {
            merge_tensor_stats(
                &post_shard_stats,
                collect_tensor_f32_stats(ranks[rank].d_post_attn_shard,
                                         (size_t)shard_elems,
                                         ranks[rank].stream));
        }
    }

    if (rank_major_input) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            if (!r.compose_nccl_initialized || !r.compose_nccl ||
                !r.d_post_attn_full_rank_major ||
                !hc->d_ffn_norm_weight_rank[layer][rank]) {
                return 10;
            }
        }
        CHECK_NCCL(ncclGroupStart());
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_NCCL(ncclAllGather(r.d_post_attn_shard,
                                     r.d_post_attn_full_rank_major,
                                     (size_t)shard_elems,
                                     ncclFloat,
                                     r.compose_nccl,
                                     r.stream));
        }
        CHECK_NCCL(ncclGroupEnd());
        if (graph_event_order) {
            if (enqueue_control_wait_after_rank_streams(opt, ranks,
                                                        control_stream) != 0) {
                return 9;
            }
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }
    const bool needs_slot_major_ffn_norm =
        !rank_major_input ||
        !rank_major_shared_input ||
        !rank_major_route_input ||
        !(opt.model_router_rank_major_logits_gate ||
          opt.model_router_allreduce_logits_gate) ||
        opt.post_attention_slot_major_ffn_norm_gate ||
        opt.routed_ffn_rank_major_input_parity_gate ||
        !opt.true_ds4_semantic_skip_stats_gate;
    if (needs_slot_major_ffn_norm) {
        if (!hc->d_current_full || !hc->d_ffn_normed) {
            return 3;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            gather_current_shard_to_full_kernel<<<
                (unsigned int)((shard_elems + block - 1) / block), block, 0,
                control_stream>>>(
                hc->d_current_full, ranks[rank].d_post_attn_shard, rank,
                (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();

        rms_norm_weight_rows_stable_kernel<<<
            (unsigned int)opt.slots, 256, 0, control_stream>>>(
            hc->d_ffn_normed, hc->d_current_full, hc->d_ffn_norm_weight[layer],
            (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();
    }
    TensorF32Stats ffn_norm_stats;
    if (needs_slot_major_ffn_norm &&
        !opt.true_ds4_semantic_skip_stats_gate &&
        !(graph_event_order && reuse_model_router_route_plan)) {
        ffn_norm_stats =
            collect_tensor_f32_stats(hc->d_ffn_normed, (size_t)full_elems,
                                     control_stream);
    }

    if (opt.model_router_routes && reuse_model_router_route_plan &&
        !post_attention_route_reuse_audit &&
        !post_attention_fixed_capacity_route_plan) {
        int total_routes = 0;
        for (int rank = 0; rank < kGpus; ++rank) {
            total_routes += ranks[rank].routes;
        }
        if (total_routes <= 0) return 5;
    } else if (opt.model_router_routes) {
        if ((!opt.model_router_rank_major_logits_gate &&
             !opt.model_router_allreduce_logits_gate &&
             !hc->d_router_w[layer]) ||
            !hc->d_router_logits ||
            !hc->d_router_selected || !hc->d_router_weights) {
            return 5;
        }
        const int router_dense_rc = opt.model_router_allreduce_logits_gate
            ? run_model_router_allreduce_logits(opt, hc, ranks, layer,
                                                control_stream, true)
            : (opt.model_router_rank_major_logits_gate
                   ? run_model_router_rank_major_logits(opt, hc, ranks, layer,
                                                        control_stream, true)
                   : run_model_router_dense_logits(opt, hc, layer,
                                                   control_stream));
        if (router_dense_rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_post_attention_router_dense_failed\tlayer\t%d\trc\t%d\n",
                         layer, router_dense_rc);
            return 5;
        }
        if (opt.router_hash_fast_gate && hc->d_router_hash[layer] &&
            hc->d_router_tokens && hc->router_hash_rows[layer] > 0u) {
            router_select_hash_fast_rows_kernel<<<
                (unsigned int)opt.slots, 1, 0, control_stream>>>(
                hc->d_router_selected, hc->d_router_weights,
                hc->d_router_logits, hc->d_router_hash[layer],
                hc->d_router_tokens, hc->d_router_active,
                hc->router_hash_rows[layer], (uint32_t)opt.slots);
        } else {
            router_select_topk_rows_kernel<<<
                (unsigned int)opt.slots, 1, 0, control_stream>>>(
                hc->d_router_selected, hc->d_router_weights,
                hc->d_router_logits, hc->d_router_bias[layer],
                hc->d_router_hash[layer], hc->d_router_tokens,
                hc->d_router_active, hc->router_hash_rows[layer],
                (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();
        int route_rc = 0;
        if (post_attention_fixed_capacity_route_plan) {
            if (graph_event_order) {
                if (enqueue_rank_streams_wait_after_control(opt, ranks,
                                                            control_stream) != 0) {
                    return 9;
                }
            }
            route_rc = upload_post_attention_fixed_capacity_route_plan_gpu(
                opt, hc, ranks, control_stream, graph_event_order);
        } else if (post_attention_route_reuse_audit) {
            if (graph_event_order) {
                if (enqueue_rank_streams_wait_after_control(opt, ranks,
                                                            control_stream) != 0) {
                    return 9;
                }
            }
            const size_t selected_bytes =
                (size_t)opt.slots * (size_t)opt.top_k * sizeof(int);
            const size_t weights_bytes =
                (size_t)opt.slots * (size_t)opt.top_k * sizeof(float);
            if (!graph_event_order) {
                void *selected_dsts[kGpus] = {};
                void *weights_dsts[kGpus] = {};
                for (int rank = 0; rank < kGpus; ++rank) {
                    selected_dsts[rank] = ranks[rank].d_router_selected_plan;
                    weights_dsts[rank] = ranks[rank].d_router_weights_plan;
                }
                if (nccl_broadcast_bytes_from_rank0(
                        ranks, hc->d_router_selected, selected_dsts,
                        selected_bytes,
                        "post_attention_audit_selected") != 0 ||
                    nccl_broadcast_bytes_from_rank0(
                        ranks, hc->d_router_weights, weights_dsts,
                        weights_bytes,
                        "post_attention_audit_weights") != 0) {
                    return 8;
                }
            }
            for (int rank = 0; rank < kGpus; ++rank) {
                RankState &r = ranks[rank];
                if (!r.d_post_attn_route_audit ||
                    !r.d_router_selected_plan ||
                    !r.d_router_weights_plan ||
                    !r.d_offsets || !r.d_route_slots || !r.d_route_weights) {
                    return 6;
                }
                CHECK_CUDA(cudaSetDevice(r.device));
                if (graph_event_order) {
                    enqueue_graph_i32_copy_from_device0(
                        opt, r, rank, r.d_router_selected_plan,
                        hc->d_router_selected,
                        (uint64_t)opt.slots * (uint64_t)opt.top_k,
                        r.stream, block);
                    enqueue_graph_f32_copy_from_device0(
                        opt, r, rank, r.d_router_weights_plan,
                        hc->d_router_weights,
                        (uint64_t)opt.slots * (uint64_t)opt.top_k,
                        r.stream, block);
                }
                CHECK_CUDA(cudaMemsetAsync(r.d_post_attn_route_audit, 0,
                                           4u * sizeof(unsigned long long),
                                           r.stream));
                post_attention_route_plan_audit_kernel<<<
                    (unsigned int)kLocalExperts, 128, 0, r.stream>>>(
                    r.d_post_attn_route_audit, r.d_offsets, r.d_route_slots,
                    r.d_route_weights, r.d_router_selected_plan,
                    r.d_router_weights_plan, (uint32_t)rank,
                    (uint32_t)opt.slots, (uint32_t)opt.top_k);
                CHECK_CUDA(cudaGetLastError());
            }
        } else if (opt.gpu_route_plan_gate) {
            route_rc = upload_model_router_route_plan_gpu(opt, hc, ranks);
        } else if (opt.route_plan_async_upload_gate) {
            RoutePlanHostWorkspace *ws = &hc->route_plan_ws;
            if (!ws->initialized) return 6;
            const size_t route_elems = (size_t)opt.slots * (size_t)opt.top_k;
            CHECK_CUDA(cudaMemcpyAsync(ws->h_selected, hc->d_router_selected,
                                       route_elems * sizeof(int),
                                       cudaMemcpyDeviceToHost, control_stream));
            CHECK_CUDA(cudaMemcpyAsync(ws->h_weights, hc->d_router_weights,
                                       route_elems * sizeof(float),
                                       cudaMemcpyDeviceToHost, control_stream));
            CHECK_CUDA(cudaStreamSynchronize(control_stream));
            route_rc = upload_model_router_route_plan_async(
                opt, ranks, ws->h_selected, ws->h_weights, ws);
        } else {
            if (graph_event_order) {
                CHECK_CUDA(cudaSetDevice(opt.devices[0]));
                CHECK_CUDA(cudaStreamSynchronize(control_stream));
            }
            std::vector<int> selected((size_t)opt.slots * (size_t)opt.top_k);
            std::vector<float> weights((size_t)opt.slots * (size_t)opt.top_k);
            CHECK_CUDA(cudaMemcpy(selected.data(), hc->d_router_selected,
                                  selected.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(weights.data(), hc->d_router_weights,
                                  weights.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            route_rc = upload_model_router_route_plan(opt, ranks,
                                                      selected, weights);
        }
        if (route_rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_post_attention_route_plan_failed\tlayer\t%d\trc\t%d\n",
                         layer, route_rc);
            return 6;
        }
    }

    const uint64_t x_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    if (graph_event_order) {
        if (enqueue_rank_streams_wait_after_control(opt, ranks,
                                                    control_stream) != 0) {
            return 9;
        }
    }
    const bool post_ffn_slot_major_broadcast =
        (!rank_major_input) ||
        (rank_major_input &&
         (opt.routed_ffn_rank_major_input_parity_gate ||
          !rank_major_shared_input || !rank_major_route_input));
    if (post_ffn_slot_major_broadcast) {
        if (!hc->d_ffn_normed) return 10;
        const int bcast_rc = nccl_broadcast_f32_from_device0_to_current_full(
            opt, ranks, hc->d_ffn_normed, full_elems,
            "post_attention_ffn_normed_current");
        if (bcast_rc != 0) return 10;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (rank_major_input) {
            if (!r.d_post_attn_full_rank_major ||
                !hc->d_ffn_norm_weight_rank[layer][rank]) {
                return 7;
            }
            const bool needs_slot_major_copy =
                opt.routed_ffn_rank_major_input_parity_gate ||
                !rank_major_shared_input || !rank_major_route_input;
            if (needs_slot_major_copy) {
                if (!r.d_current_full) return 7;
            }
            if (ops->shared_gate.d_x_half[(size_t)rank] &&
                ops->shared_up.d_x_half[(size_t)rank]) {
                if (rank_major_shared_input) {
                    fill_two_hidden_inputs_half_from_rank_major_norm_kernel<<<
                        (unsigned int)opt.slots, 256, 0, r.stream>>>(
                        ops->shared_gate.d_x_half[(size_t)rank],
                        ops->shared_up.d_x_half[(size_t)rank],
                        r.d_post_attn_full_rank_major,
                        hc->d_ffn_norm_weight_rank[layer][rank],
                        (uint32_t)(kHidden / kGpus), (uint32_t)kGpus,
                        (uint32_t)opt.slots, 1.0e-6f);
                } else {
                    fill_dense_input_half_from_current_kernel<<<
                        (unsigned int)((x_elems + block - 1) / block), block,
                        0, r.stream>>>(
                        ops->shared_gate.d_x_half[(size_t)rank],
                        r.d_current_full, (uint32_t)ops->shared_gate.cols,
                        (uint32_t)opt.slots);
                    fill_dense_input_half_from_current_kernel<<<
                        (unsigned int)((x_elems + block - 1) / block), block,
                        0, r.stream>>>(
                        ops->shared_up.d_x_half[(size_t)rank],
                        r.d_current_full, (uint32_t)ops->shared_up.cols,
                        (uint32_t)opt.slots);
                }
                CHECK_CUDA(cudaGetLastError());
                if (opt.routed_ffn_rank_major_input_parity_gate &&
                    !reuse_model_router_route_plan &&
                    rank_major_shared_input) {
                    HalfInputDiffStats gate_diff =
                        collect_shared_half_input_diff(
                            r, ops->shared_gate.d_x_half[(size_t)rank],
                            r.d_current_full, (uint32_t)ops->shared_gate.cols,
                            (uint32_t)opt.slots, r.stream);
                    log_half_input_diff("shared_gate", layer, rank, gate_diff);
                    HalfInputDiffStats up_diff =
                        collect_shared_half_input_diff(
                            r, ops->shared_up.d_x_half[(size_t)rank],
                            r.d_current_full, (uint32_t)ops->shared_up.cols,
                            (uint32_t)opt.slots, r.stream);
                    log_half_input_diff("shared_up", layer, rank, up_diff);
                }
            }
            if (r.routes > 0) {
                if (rank_major_route_input) {
                    const int *route_total_limit =
                        opt.post_attention_fixed_capacity_route_plan_gate
                            ? r.d_route_totals
                            : nullptr;
                    if (opt.reference_hc_reduce_gate) {
                        pack_rank_major_norm_current_to_routes_scaled_kernel<<<
                            (unsigned int)r.routes, 256, 0, r.stream>>>(
                                r.d_a, r.d_route_inv_scale,
                                r.d_post_attn_full_rank_major,
                                hc->d_ffn_norm_weight_rank[layer][rank],
                                r.d_route_slots, route_total_limit, r.routes,
                                (uint32_t)rank,
                                (uint32_t)(kHidden / kGpus), (uint32_t)kGpus,
                                (uint32_t)opt.slots, 1.0e-6f,
                                kReferenceRouteInputTargetAbs);
                    } else {
                        pack_rank_major_norm_current_to_routes_kernel<<<
                            (unsigned int)r.routes, 256, 0, r.stream>>>(
                                r.d_a, r.d_post_attn_full_rank_major,
                                hc->d_ffn_norm_weight_rank[layer][rank],
                                r.d_route_slots, route_total_limit, r.routes,
                                (uint32_t)rank,
                                (uint32_t)(kHidden / kGpus), (uint32_t)kGpus,
                                (uint32_t)opt.slots, 1.0e-6f);
                    }
                } else {
                    const uint64_t route_elems = (uint64_t)r.routes * kHidden;
                    if (opt.reference_hc_reduce_gate) {
                        pack_current_full_to_routes_scaled_kernel<<<
                            (unsigned int)r.routes, 256, 0, r.stream>>>(
                                r.d_a, r.d_route_inv_scale, r.d_current_full,
                                r.d_route_slots, r.routes,
                                kReferenceRouteInputTargetAbs);
                    } else {
                        pack_current_full_to_routes_kernel<<<
                            (unsigned int)((route_elems + block - 1) / block),
                            block, 0, r.stream>>>(
                                r.d_a, r.d_current_full, r.d_route_slots,
                                r.routes);
                    }
                }
                CHECK_CUDA(cudaGetLastError());
                if (opt.routed_ffn_rank_major_input_parity_gate &&
                    !reuse_model_router_route_plan &&
                    rank_major_route_input &&
                    !opt.reference_hc_reduce_gate) {
                    HalfInputDiffStats route_diff =
                        collect_route_half_input_diff(
                            r, r.d_a, r.d_current_full, r.d_route_slots,
                            r.routes, r.stream);
                    log_half_input_diff("route_a", layer, rank, route_diff);
                }
            }
        } else {
            if (!r.d_current_full) return 7;
            if (ops->shared_gate.d_x_half[(size_t)rank]) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((x_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->shared_gate.d_x_half[(size_t)rank],
                                 r.d_current_full,
                                 (uint32_t)ops->shared_gate.cols,
                                 (uint32_t)opt.slots);
                CHECK_CUDA(cudaGetLastError());
            }
            if (ops->shared_up.d_x_half[(size_t)rank]) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((x_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->shared_up.d_x_half[(size_t)rank],
                                 r.d_current_full,
                                 (uint32_t)ops->shared_up.cols,
                                 (uint32_t)opt.slots);
                CHECK_CUDA(cudaGetLastError());
            }
            const uint64_t route_elems = (uint64_t)r.routes * kHidden;
            if (route_elems > 0) {
                if (opt.reference_hc_reduce_gate) {
                    pack_current_full_to_routes_scaled_kernel<<<
                        (unsigned int)r.routes, 256, 0, r.stream>>>(
                            r.d_a, r.d_route_inv_scale, r.d_current_full,
                            r.d_route_slots, r.routes, kReferenceRouteInputTargetAbs);
                } else {
                    pack_current_full_to_routes_kernel<<<
                        (unsigned int)((route_elems + block - 1) / block), block,
                        0, r.stream>>>(r.d_a, r.d_current_full, r.d_route_slots,
                                       r.routes);
                }
                CHECK_CUDA(cudaGetLastError());
            }
        }
    }
    if (graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 9;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }

    TensorF32Stats route_inv_scale_stats;
    int total_routes = 0;
    for (int rank = 0; rank < kGpus; ++rank) {
        total_routes += ranks[rank].routes;
        if (!opt.true_ds4_semantic_skip_stats_gate &&
            !(graph_event_order && reuse_model_router_route_plan) &&
            ranks[rank].d_route_inv_scale && ranks[rank].routes > 0) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            merge_tensor_stats(
                &route_inv_scale_stats,
                collect_tensor_f32_stats(ranks[rank].d_route_inv_scale,
                                         (size_t)ranks[rank].routes,
                                         ranks[rank].stream));
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    std::printf("tp_ep_post_attention_ffn_input\tlayer\t%d\tslots\t%d\t"
                "total_routes\t%d\tstats_skipped\t%d\tpost_max\t%.9g\tpost_bad\t%d\t"
                "ffn_norm_max\t%.9g\tffn_norm_bad\t%d\t"
                "route_inv_scale_max\t%.9g\troute_inv_scale_bad\t%d\t"
                "rank_major_input\t%d\trank_major_shared_input\t%d\t"
                "rank_major_route_input\t%d\tslot_major_ffn_norm\t%d\t"
                "ms\t%.6f\tPASS\n",
                layer, opt.slots, total_routes,
                opt.true_ds4_semantic_skip_stats_gate ? 1 : 0,
                post_shard_stats.max_abs,
                post_shard_stats.finite_bad, ffn_norm_stats.max_abs,
                ffn_norm_stats.finite_bad, route_inv_scale_stats.max_abs,
                route_inv_scale_stats.finite_bad, rank_major_input ? 1 : 0,
                rank_major_shared_input ? 1 : 0,
                rank_major_route_input ? 1 : 0,
                needs_slot_major_ffn_norm ? 1 : 0, ms);
    return (post_shard_stats.finite_bad || ffn_norm_stats.finite_bad ||
            route_inv_scale_stats.finite_bad) ? 8 : 0;
}
