int run_shared_hc_current_input(const Options &opt,
                                SharedHcControls *hc,
                                RankState ranks[kGpus],
                                const ResidentF8Dense &attn_op,
                                const ResidentF8Dense &shared_op,
                                int layer,
                                bool reuse_model_router_route_plan,
                                HcCurrentInputBreakdown *breakdown) {
    if (!hc || !hc->initialized || hc->slots != opt.slots ||
        layer < 0 || layer >= 44) {
        return 1;
    }
    if (attn_op.cols <= 0 || shared_op.cols <= 0) return 2;
    const uint64_t shard_elems =
        (uint64_t)opt.slots * (uint64_t)(kHidden / kGpus);
    const uint64_t hc_shard_elems = shard_elems * kHcRows;
    const uint64_t full_elems = (uint64_t)opt.slots * kHidden;
    const int block = 256;
    const auto t_start = std::chrono::steady_clock::now();
    const bool graph_event_order = opt.decode_cudagraph_gate;
    cudaStream_t control_stream =
        (opt.tp_hc_current_input_stream_sync_gate || graph_event_order)
            ? ranks[0].stream
            : (cudaStream_t)0;
    auto control_wait_on_rank_streams = [&]() -> int {
        if (!graph_event_order) return 0;
        const int slot = next_graph_order_event_slot(ranks);
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            cudaEvent_t ev = graph_stream_done_event(r, slot);
            if (!ev) return 1;
            CHECK_CUDA(cudaEventRecord(ev, r.stream));
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaStreamWaitEvent(control_stream,
                                           graph_stream_done_event(ranks[rank],
                                                                   slot),
                                           0));
        }
        return 0;
    };
    auto rank_streams_wait_on_control = [&]() -> int {
        if (!graph_event_order) return 0;
        const int slot = next_graph_order_event_slot(ranks);
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        cudaEvent_t ev = graph_stream_done_event(ranks[0], slot);
        if (!ev) return 1;
        CHECK_CUDA(cudaEventRecord(ev, control_stream));
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_CUDA(cudaStreamWaitEvent(r.stream, ev, 0));
        }
        return 0;
    };
    auto sync_control_device = [&]() {
        if (graph_event_order) return;
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        if (opt.tp_hc_current_input_stream_sync_gate) {
            CHECK_CUDA(cudaStreamSynchronize(control_stream));
        } else {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    };

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_final_hc_shard || !r.d_current_shard || !r.d_current_full ||
            !r.d_hc_split) {
            return 3;
        }
        if (!r.hc_initialized) {
            seed_initial_hc_shard_kernel<<<
                (unsigned int)((hc_shard_elems + block - 1) / block), block,
                0, r.stream>>>(r.d_final_hc_shard, rank, opt.slots);
            CHECK_CUDA(cudaGetLastError());
            r.hc_initialized = true;
        }
    }
    if (graph_event_order) {
        if (control_wait_on_rank_streams() != 0) return 6;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    auto t_seed_done = std::chrono::steady_clock::now();
    if (should_log_reference_hc_window(opt)) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("hc_current_shard", layer, rank,
                                 ranks[rank].d_current_shard,
                                 (size_t)shard_elems, ranks[rank].stream);
        }
    }

    if (opt.tp_hc_current_allreduce_gate) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            if (!r.compose_nccl_initialized || !r.compose_nccl ||
                !r.d_hc_reduce_max || !r.d_hc_reduce_sumsq ||
                !r.d_hc_reduce_mix || !hc->d_attn_fn_rank[layer][rank] ||
                !hc->d_attn_scale_rank[layer][rank] ||
                !hc->d_attn_base_rank[layer][rank]) {
                return 12;
            }
            CHECK_CUDA(cudaSetDevice(r.device));
            const dim3 partial_grid((unsigned int)(kHcMix + 1),
                                    (unsigned int)opt.slots, 1u);
            hc_local_max_mix_partial_kernel<<<partial_grid, 256, 0, r.stream>>>(
                r.d_hc_reduce_max, r.d_hc_reduce_mix, r.d_final_hc_shard,
                hc->d_attn_fn_rank[layer][rank], (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        /* s602: hc max+mix allreduce site (kernel transport shares one
         * barrier pair across both ops; flag-off this block is byte-
         * identical to the plain NCCL group). */
        S602ArOp hc_mm_ops[2] = {};
        hc_mm_ops[0].cls = kS602HcMax;
        hc_mm_ops[0].count = (uint64_t)opt.slots;
        hc_mm_ops[0].is_max = true;
        hc_mm_ops[0].fold_all = true;
        hc_mm_ops[1].cls = kS602HcMix;
        hc_mm_ops[1].count = (uint64_t)opt.slots * kHcMix;
        hc_mm_ops[1].is_max = false;
        hc_mm_ops[1].fold_all = true;
        for (int rank = 0; rank < kGpus; ++rank) {
            hc_mm_ops[0].in[rank] = ranks[rank].d_hc_reduce_max;
            hc_mm_ops[0].out[rank] = ranks[rank].d_s602_out_max;
            hc_mm_ops[1].in[rank] = ranks[rank].d_hc_reduce_mix;
            hc_mm_ops[1].out[rank] = ranks[rank].d_s602_out_mix;
        }
        const bool s602_k_mm = s602_use_kernel(opt, kS602HcMax);
        if (s602_allreduce_site_pre(opt, ranks, hc_mm_ops, 2) != 0) return 14;
        if (!s602_k_mm) {
            CHECK_NCCL(ncclGroupStart());
            for (int rank = 0; rank < kGpus; ++rank) {
                RankState &r = ranks[rank];
                CHECK_CUDA(cudaSetDevice(r.device));
                CHECK_NCCL(ncclAllReduce(r.d_hc_reduce_max, r.d_hc_reduce_max,
                                         (size_t)opt.slots, ncclFloat, ncclMax,
                                         ds4_comm_hc(r), r.stream));
                CHECK_NCCL(ncclAllReduce(r.d_hc_reduce_mix, r.d_hc_reduce_mix,
                                         (size_t)opt.slots * kHcMix, ncclFloat,
                                         ncclSum, ds4_comm_hc(r), r.stream));
            }
            CHECK_NCCL(ncclGroupEnd());
        }
        if (s602_allreduce_site_post(opt, ranks, hc_mm_ops, 2) != 0) return 14;
        bool have_ref_mix_for_full_parity = false;
        if (opt.tp_hc_current_full_parity_gate) {
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
            for (int rank = 0; rank < kGpus; ++rank) {
                gather_hc_shard_to_full_kernel<<<
                    (unsigned int)((hc_shard_elems + block - 1) / block),
                    block, 0, control_stream>>>(
                    hc->d_hc, ranks[rank].d_final_hc_shard, rank,
                    (uint32_t)opt.slots);
            }
            CHECK_CUDA(cudaGetLastError());
            sync_control_device();
            rms_norm_plain_rows_stable_kernel<<<
                (unsigned int)opt.slots, 256, 0, control_stream>>>(
                hc->d_hc_norm, hc->d_hc, kHcRows * (uint32_t)kHidden,
                (uint32_t)opt.slots, 1.0e-6f);
            const dim3 ref_mix_grid((unsigned int)kHcMix,
                                    (unsigned int)opt.slots, 1u);
            f32_dense_colmajor_kernel<<<ref_mix_grid, 256, 0, control_stream>>>(
                hc->d_split, hc->d_attn_fn[layer], hc->d_hc_norm,
                (uint32_t)kHcMix, kHcRows * (uint32_t)kHidden,
                (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
            sync_control_device();
            have_ref_mix_for_full_parity = true;
        }
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            hc_local_stable_sumsq_kernel<<<
                (unsigned int)opt.slots, 256, 0, r.stream>>>(
                r.d_hc_reduce_sumsq, r.d_final_hc_shard,
                s602_k_mm ? r.d_s602_out_max : r.d_hc_reduce_max,
                (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        /* s602: hc sumsq allreduce site. */
        S602ArOp hc_ss_op[1] = {};
        hc_ss_op[0].cls = kS602HcSumsq;
        hc_ss_op[0].count = (uint64_t)opt.slots;
        hc_ss_op[0].is_max = false;
        hc_ss_op[0].fold_all = true;
        for (int rank = 0; rank < kGpus; ++rank) {
            hc_ss_op[0].in[rank] = ranks[rank].d_hc_reduce_sumsq;
            hc_ss_op[0].out[rank] = ranks[rank].d_s602_out_sumsq;
        }
        const bool s602_k_ss = s602_use_kernel(opt, kS602HcSumsq);
        if (s602_allreduce_site_pre(opt, ranks, hc_ss_op, 1) != 0) return 14;
        if (!s602_k_ss) {
            CHECK_NCCL(ncclGroupStart());
            for (int rank = 0; rank < kGpus; ++rank) {
                RankState &r = ranks[rank];
                CHECK_CUDA(cudaSetDevice(r.device));
                CHECK_NCCL(ncclAllReduce(r.d_hc_reduce_sumsq,
                                         r.d_hc_reduce_sumsq,
                                         (size_t)opt.slots, ncclFloat, ncclSum,
                                         ds4_comm_hc(r), r.stream));
            }
            CHECK_NCCL(ncclGroupEnd());
        }
        if (s602_allreduce_site_post(opt, ranks, hc_ss_op, 1) != 0) return 14;
        if (have_ref_mix_for_full_parity) {
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
            hc_scale_reduced_mix_kernel<<<
                (unsigned int)(((uint64_t)opt.slots * kHcMix + 255) / 256),
                256, 0, control_stream>>>(
                hc->d_mix,
                s602_k_mm ? ranks[0].d_s602_out_max : ranks[0].d_hc_reduce_max,
                s602_k_ss ? ranks[0].d_s602_out_sumsq
                          : ranks[0].d_hc_reduce_sumsq,
                s602_k_mm ? ranks[0].d_s602_out_mix : ranks[0].d_hc_reduce_mix,
                (uint32_t)opt.slots, 1.0e-6f);
            CHECK_CUDA(cudaGetLastError());
            sync_control_device();
            const TensorF32DiffStats diff = collect_tensor_f32_diff_stats(
                hc->d_mix, hc->d_split, (size_t)opt.slots * kHcMix,
                control_stream);
            std::printf("tp_ep_hc_current_allreduce_mix_diff\tlayer\t%d\t"
                        "slots\t%d\tmax_abs_diff\t%.9g\tmax_rel_diff\t%.9g\t"
                        "diff_bad\t%d\tfirst_bad\t%zu\t%s\n",
                        layer, opt.slots, diff.max_abs, diff.max_rel,
                        diff.bad, diff.first_bad,
                        (diff.max_abs <= 1.0e-4f ||
                         diff.max_rel <= 1.0e-4f) ? "PASS" : "WARN");
        }
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            hc_apply_reduced_mix_split_kernel<<<
                (unsigned int)(((uint64_t)opt.slots + 255) / 256), 256, 0,
                r.stream>>>(
                r.d_hc_split,
                s602_k_mm ? r.d_s602_out_max : r.d_hc_reduce_max,
                s602_k_ss ? r.d_s602_out_sumsq : r.d_hc_reduce_sumsq,
                s602_k_mm ? r.d_s602_out_mix : r.d_hc_reduce_mix,
                hc->d_attn_scale_rank[layer][rank],
                hc->d_attn_base_rank[layer][rank], (uint32_t)opt.slots,
                opt.reference_hc_reduce_gate ? 20u : 4u, 1.0e-6f);
            CHECK_CUDA(cudaGetLastError());
        }
    } else {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            gather_hc_shard_to_full_kernel<<<
                (unsigned int)((hc_shard_elems + block - 1) / block), block,
                0, control_stream>>>(
                hc->d_hc, ranks[rank].d_final_hc_shard, rank, (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();

        rms_norm_plain_rows_stable_kernel<<<(unsigned int)opt.slots, 256, 0, control_stream>>>(
            hc->d_hc_norm, hc->d_hc, kHcRows * (uint32_t)kHidden,
            (uint32_t)opt.slots, 1.0e-6f);
        const dim3 mix_grid((unsigned int)kHcMix, (unsigned int)opt.slots, 1u);
        f32_dense_colmajor_kernel<<<mix_grid, 256, 0, control_stream>>>(
            hc->d_mix, hc->d_attn_fn[layer], hc->d_hc_norm,
            (uint32_t)kHcMix, kHcRows * (uint32_t)kHidden, (uint32_t)opt.slots);
        hc_split_rows_kernel<<<
            (unsigned int)(((uint64_t)opt.slots + 255) / 256), 256, 0,
            control_stream>>>(
            hc->d_split, hc->d_mix, hc->d_attn_scale[layer], hc->d_attn_base[layer],
            (uint32_t)opt.slots, opt.reference_hc_reduce_gate ? 20u : 4u);
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();
        if (graph_event_order) {
            if (rank_streams_wait_on_control() != 0) return 7;
        }
    }

    if (!opt.tp_hc_current_allreduce_gate && !graph_event_order) {
        void *dsts[kGpus] = {};
        for (int rank = 0; rank < kGpus; ++rank) {
            dsts[rank] = ranks[rank].d_hc_split;
        }
        if (nccl_broadcast_bytes_from_rank0(
                ranks, hc->d_split, dsts,
                (size_t)opt.slots * kHcMix * sizeof(float),
                "hc_current_split") != 0) {
            return 10;
        }
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (opt.tp_hc_current_allreduce_gate) {
            // Split is already resident on this rank from the NCCL all-reduce path.
        } else if (graph_event_order) {
            enqueue_graph_f32_copy_from_device0(
                opt, r, rank, r.d_hc_split, hc->d_split,
                (uint64_t)opt.slots * kHcMix, r.stream, block);
        }
        hc_weighted_sum_shard_kernel<<<
            (unsigned int)((shard_elems + block - 1) / block), block,
            0, r.stream>>>(r.d_current_shard, r.d_final_hc_shard,
                           r.d_hc_split, (uint32_t)opt.slots,
                           opt.reference_hc_reduce_gate ? 1 : 0);
        CHECK_CUDA(cudaGetLastError());
    }
    auto t_split_done = std::chrono::steady_clock::now();
    if (graph_event_order) {
        if (control_wait_on_rank_streams() != 0) return 8;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    auto t_weighted_done = std::chrono::steady_clock::now();

    float *control_current_full = hc->d_current_full;
    const bool peer_gather_current = opt.tp_hc_current_input_peer_gather_gate;
    const bool nccl_gather_current =
        opt.tp_hc_current_input_nccl_allgather_gate;
    const bool rank_local_current_full =
        peer_gather_current || nccl_gather_current;
    if (nccl_gather_current) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            if (!r.compose_nccl_initialized || !r.compose_nccl ||
                !r.d_current_full_rank_major) {
                return 9;
            }
        }
        /* s602: hc current allgather site (byte moves). */
        const bool s602_k_ag = s602_use_kernel(opt, kS602HcAg);
        if (!s602_k_ag) {
            CHECK_NCCL(ncclGroupStart());
            for (int rank = 0; rank < kGpus; ++rank) {
                RankState &r = ranks[rank];
                CHECK_CUDA(cudaSetDevice(r.device));
                CHECK_NCCL(ncclAllGather(r.d_current_shard,
                                         r.d_current_full_rank_major,
                                         (size_t)shard_elems,
                                         ncclFloat,
                                         ds4_comm_hc(r),
                                         r.stream));
            }
            CHECK_NCCL(ncclGroupEnd());
        }
        if (s602_k_ag || s602_use_verify(opt, kS602HcAg)) {
            float *ag_shards[kGpus] = {};
            float *ag_outs[kGpus] = {};
            for (int rank = 0; rank < kGpus; ++rank) {
                ag_shards[rank] = ranks[rank].d_current_shard;
                ag_outs[rank] = ranks[rank].d_current_full_rank_major;
            }
            if (s602_allgather_site(opt, ranks, kS602HcAg, ag_shards, ag_outs,
                                    shard_elems, 0) != 0) {
                return 9;
            }
        }
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            rank_major_current_shards_to_slot_major_kernel<<<
                (unsigned int)((full_elems + block - 1) / block), block, 0,
                r.stream>>>(
                r.d_current_full, r.d_current_full_rank_major,
                (uint32_t)(kHidden / kGpus), (uint32_t)kGpus,
                (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        if (graph_event_order) {
            if (control_wait_on_rank_streams() != 0) return 9;
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
        control_current_full = ranks[0].d_current_full;
    } else if (peer_gather_current) {
        const uint64_t full_grid_elems = full_elems;
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            gather_current_shards_to_full8_kernel<<<
                (unsigned int)((full_grid_elems + block - 1) / block), block,
                0, r.stream>>>(r.d_current_full,
                               ranks[0].d_current_shard,
                               ranks[1].d_current_shard,
                               ranks[2].d_current_shard,
                               ranks[3].d_current_shard,
                               ranks[4].d_current_shard,
                               ranks[5].d_current_shard,
                               ranks[6].d_current_shard,
                               ranks[7].d_current_shard,
                               (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        if (graph_event_order) {
            if (control_wait_on_rank_streams() != 0) return 9;
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
        control_current_full = ranks[0].d_current_full;
    } else {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            gather_current_shard_to_full_kernel<<<
                (unsigned int)((shard_elems + block - 1) / block), block,
                0, control_stream>>>(
                hc->d_current_full, ranks[rank].d_current_shard, rank,
                (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        sync_control_device();
    }
    auto t_gather_done = std::chrono::steady_clock::now();
    if (opt.tp_hc_current_full_parity_gate && rank_local_current_full) {
        log_hc_current_full_rank_parity(opt, ranks, layer, (size_t)full_elems);
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    if (should_log_reference_hc_window(opt)) {
        log_tensor_f32_stats("hc_current_full", layer, 0, control_current_full,
                             (size_t)full_elems, nullptr);
    }

    if (!hc->d_ffn_normed || !hc->d_ffn_norm_weight[layer]) return 4;
    rms_norm_weight_rows_stable_kernel<<<(unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_ffn_normed, control_current_full, hc->d_ffn_norm_weight[layer],
        (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());
    sync_control_device();
    auto t_norm_done = std::chrono::steady_clock::now();
    if (graph_event_order) {
        if (rank_streams_wait_on_control() != 0) return 10;
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    }
    if (should_log_reference_hc_window(opt)) {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        log_tensor_f32_stats("hc_ffn_normed", layer, 0, hc->d_ffn_normed,
                             (size_t)full_elems, nullptr);
    }

    auto t_router_select_done = t_norm_done;
    auto t_router_d2h_done = t_norm_done;
    auto t_route_upload_done = t_norm_done;
    if (opt.model_router_routes && reuse_model_router_route_plan) {
        int total_routes = 0;
        for (int rank = 0; rank < kGpus; ++rank) total_routes += ranks[rank].routes;
        if (total_routes <= 0) return 5;
        t_router_select_done = t_norm_done;
        t_router_d2h_done = t_norm_done;
        t_route_upload_done = t_norm_done;
    } else if (opt.model_router_routes) {
        if ((!opt.model_router_rank_major_logits_gate &&
             !opt.model_router_allreduce_logits_gate &&
             !hc->d_router_w[layer]) ||
            !hc->d_router_logits ||
            !hc->d_router_selected || !hc->d_router_weights) {
            return 4;
        }
        const int router_dense_rc = opt.model_router_allreduce_logits_gate
            ? run_model_router_allreduce_logits(opt, hc, ranks, layer,
                                                control_stream, false)
            : (opt.model_router_rank_major_logits_gate
                   ? run_model_router_rank_major_logits(opt, hc, ranks, layer,
                                                        control_stream, false)
                   : run_model_router_dense_logits(opt, hc, layer,
                                                   control_stream));
        if (router_dense_rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_model_router_dense_failed\tlayer\t%d\trc\t%d\n",
                         layer, router_dense_rc);
            return 4;
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
        t_router_select_done = std::chrono::steady_clock::now();
        int route_rc = 0;
        if (opt.gpu_route_plan_gate) {
            t_router_d2h_done = t_router_select_done;
            route_rc = upload_model_router_route_plan_gpu(opt, hc, ranks);
        } else if (opt.route_plan_async_upload_gate) {
            RoutePlanHostWorkspace *ws = &hc->route_plan_ws;
            if (!ws->initialized) return 5;
            const size_t route_elems = (size_t)opt.slots * (size_t)opt.top_k;
            CHECK_CUDA(cudaMemcpyAsync(ws->h_selected, hc->d_router_selected,
                                       route_elems * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       control_stream));
            CHECK_CUDA(cudaMemcpyAsync(ws->h_weights, hc->d_router_weights,
                                       route_elems * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       control_stream));
            CHECK_CUDA(cudaStreamSynchronize(control_stream));
            t_router_d2h_done = std::chrono::steady_clock::now();
            route_rc = upload_model_router_route_plan_async(
                opt, ranks, ws->h_selected, ws->h_weights, ws);
        } else {
            std::vector<int> selected((size_t)opt.slots * (size_t)opt.top_k);
            std::vector<float> weights((size_t)opt.slots * (size_t)opt.top_k);
            CHECK_CUDA(cudaMemcpy(selected.data(), hc->d_router_selected,
                                  selected.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(weights.data(), hc->d_router_weights,
                                  weights.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            t_router_d2h_done = std::chrono::steady_clock::now();
            route_rc = upload_model_router_route_plan(opt, ranks,
                                                      selected, weights);
        }
        if (route_rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_model_router_route_plan_failed\tlayer\t%d\trc\t%d\n",
                         layer, route_rc);
            return 5;
        }
        t_route_upload_done = std::chrono::steady_clock::now();
    } else {
        t_router_select_done = t_norm_done;
        t_router_d2h_done = t_norm_done;
        t_route_upload_done = t_norm_done;
    }
    auto t_router_done = t_route_upload_done;

    const bool fused_fill_pack =
        opt.tp_hc_current_input_fused_fill_pack_gate &&
        !rank_local_current_full && !graph_event_order &&
        !opt.reference_hc_reduce_gate &&
        (!opt.routed_ffn_norm_input_gate || hc->d_ffn_normed);
    if (!fused_fill_pack && !rank_local_current_full) {
        const int bcast_rc = nccl_broadcast_f32_from_device0_to_current_full(
            opt, ranks, hc->d_current_full, full_elems,
            "hc_current_full_input");
        if (bcast_rc != 0) return 12;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        const uint64_t attn_elems = (uint64_t)opt.slots * (uint64_t)attn_op.cols;
        const uint64_t shared_elems = (uint64_t)opt.slots * (uint64_t)shared_op.cols;
        const uint64_t route_elems = (uint64_t)r.routes * kHidden;
        if (fused_fill_pack) {
            const float *state_src =
                (opt.routed_ffn_norm_input_gate && route_elems > 0)
                    ? hc->d_ffn_normed
                    : hc->d_current_full;
            const float *route_src =
                opt.routed_ffn_norm_input_gate ? hc->d_ffn_normed : hc->d_current_full;
            const uint64_t total = std::max(
                std::max(full_elems, attn_elems),
                std::max(shared_elems, route_elems));
            hc_current_fused_fill_pack_kernel<<<
                (unsigned int)((total + block - 1) / block), block,
                0, r.stream>>>(
                r.d_current_full, state_src, hc->d_current_full, route_src,
                attn_op.d_x[(size_t)rank], (uint32_t)attn_op.cols,
                shared_op.d_x[(size_t)rank], (uint32_t)shared_op.cols,
                attn_op.d_x_half[(size_t)rank],
                shared_op.d_x_half[(size_t)rank],
                route_elems > 0 ? r.d_a : nullptr,
                route_elems > 0 ? r.d_route_slots : nullptr,
                r.routes, (uint32_t)opt.slots, total);
            CHECK_CUDA(cudaGetLastError());
        } else {
            if (!rank_local_current_full) {
                // A4a: current-full transport is handled once above via NCCL.
            }
            if (attn_op.d_x[(size_t)rank]) {
                fill_dense_input_from_current_kernel<<<
                    (unsigned int)((attn_elems + block - 1) / block), block,
                    0, r.stream>>>(attn_op.d_x[(size_t)rank], r.d_current_full,
                                   (uint32_t)attn_op.cols, (uint32_t)opt.slots);
            }
            if (shared_op.d_x[(size_t)rank]) {
                fill_dense_input_from_current_kernel<<<
                    (unsigned int)((shared_elems + block - 1) / block), block,
                    0, r.stream>>>(shared_op.d_x[(size_t)rank], r.d_current_full,
                                   (uint32_t)shared_op.cols, (uint32_t)opt.slots);
            }
            if (attn_op.d_x_half[(size_t)rank]) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((attn_elems + block - 1) / block), block,
                    0, r.stream>>>(attn_op.d_x_half[(size_t)rank], r.d_current_full,
                                   (uint32_t)attn_op.cols, (uint32_t)opt.slots);
            }
            if (shared_op.d_x_half[(size_t)rank]) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((shared_elems + block - 1) / block), block,
                    0, r.stream>>>(shared_op.d_x_half[(size_t)rank],
                                   r.d_current_full, (uint32_t)shared_op.cols,
                                   (uint32_t)opt.slots);
            }
            if (opt.routed_ffn_norm_input_gate && route_elems > 0) {
                // Packed route input is emitted after the loop, once ffn_normed
                // has been broadcast to every rank by NCCL.
            }
            if (route_elems > 0 && !opt.routed_ffn_norm_input_gate) {
                if (opt.reference_hc_reduce_gate) {
                    pack_current_full_to_routes_scaled_kernel<<<
                        (unsigned int)r.routes, 256, 0, r.stream>>>(
                            r.d_a, r.d_route_inv_scale, r.d_current_full,
                            r.d_route_slots, r.routes, kReferenceRouteInputTargetAbs);
                } else {
                    pack_current_full_to_routes_kernel<<<
                        (unsigned int)((route_elems + block - 1) / block), block,
                        0, r.stream>>>(r.d_a, r.d_current_full, r.d_route_slots, r.routes);
                }
                CHECK_CUDA(cudaGetLastError());
            }
        }
        if (should_log_reference_hc_window(opt) && r.d_route_inv_scale && r.routes > 0) {
            log_tensor_f32_stats("route_inv_scale", layer, rank,
                                 r.d_route_inv_scale, (size_t)r.routes,
                                 r.stream);
        }
    }
    if (opt.routed_ffn_norm_input_gate) {
        const int bcast_rc = nccl_broadcast_f32_from_device0_to_current_full(
            opt, ranks, hc->d_ffn_normed, full_elems,
            "hc_current_ffn_normed_route_input");
        if (bcast_rc != 0) return 13;
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            const uint64_t route_elems = (uint64_t)r.routes * kHidden;
            if (route_elems == 0) continue;
            if (opt.reference_hc_reduce_gate) {
                pack_current_full_to_routes_scaled_kernel<<<
                    (unsigned int)r.routes, 256, 0, r.stream>>>(
                        r.d_a, r.d_route_inv_scale, r.d_current_full,
                        r.d_route_slots, r.routes, kReferenceRouteInputTargetAbs);
            } else {
                pack_current_full_to_routes_kernel<<<
                    (unsigned int)((route_elems + block - 1) / block), block,
                    0, r.stream>>>(r.d_a, r.d_current_full, r.d_route_slots, r.routes);
            }
            CHECK_CUDA(cudaGetLastError());
        }
    }
    if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 11;
    auto t_fill_done = std::chrono::steady_clock::now();
    if (breakdown) {
        breakdown->seed_ms +=
            std::chrono::duration<double, std::milli>(t_seed_done - t_start).count();
        breakdown->attn_mix_ms +=
            std::chrono::duration<double, std::milli>(t_split_done - t_seed_done).count();
        breakdown->split_ms +=
            std::chrono::duration<double, std::milli>(t_weighted_done - t_split_done).count();
        breakdown->gather_ms +=
            std::chrono::duration<double, std::milli>(t_gather_done - t_weighted_done).count();
        breakdown->ffn_router_ms +=
            std::chrono::duration<double, std::milli>(t_router_done - t_gather_done).count();
        breakdown->ffn_norm_ms +=
            std::chrono::duration<double, std::milli>(t_norm_done - t_gather_done).count();
        breakdown->router_select_ms +=
            std::chrono::duration<double, std::milli>(t_router_select_done - t_norm_done).count();
        breakdown->router_d2h_ms +=
            std::chrono::duration<double, std::milli>(t_router_d2h_done - t_router_select_done).count();
        breakdown->route_upload_ms +=
            std::chrono::duration<double, std::milli>(t_route_upload_done - t_router_d2h_done).count();
        breakdown->fill_pack_ms +=
            std::chrono::duration<double, std::milli>(t_fill_done - t_router_done).count();
    }
    return 0;
}
