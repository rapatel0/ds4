int upload_model_router_route_plan_gpu(const Options &opt,
                                       SharedHcControls *hc,
                                       RankState ranks[kGpus]) {
    if (!opt.compact_moe_decode_gate || !hc || !hc->d_router_selected ||
        !hc->d_router_weights) {
        return 1;
    }
    const size_t selected_bytes =
        (size_t)opt.slots * (size_t)opt.top_k * sizeof(int);
    const size_t weights_bytes =
        (size_t)opt.slots * (size_t)opt.top_k * sizeof(float);
    const size_t offsets_all_bytes =
        (size_t)kGpus * (size_t)(kLocalExperts + 1) * sizeof(int);
    const int block = 256;
    const uint32_t route_entries = (uint32_t)(opt.slots * opt.top_k);
    void *selected_dsts[kGpus] = {};
    void *weights_dsts[kGpus] = {};
    for (int rank = 0; rank < kGpus; ++rank) {
        selected_dsts[rank] = ranks[rank].d_router_selected_plan;
        weights_dsts[rank] = ranks[rank].d_router_weights_plan;
    }
    if (nccl_broadcast_bytes_from_rank0(
            ranks, hc->d_router_selected, selected_dsts, selected_bytes,
            "router_plan_selected") != 0 ||
        nccl_broadcast_bytes_from_rank0(
            ranks, hc->d_router_weights, weights_dsts, weights_bytes,
            "router_plan_weights") != 0) {
        return 4;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.d_router_selected_plan || !r.d_router_weights_plan ||
            !r.d_route_offsets_all || !r.d_route_totals ||
            !r.d_route_compact_plan) {
            return 2;
        }
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaMemsetAsync(r.d_route_offsets_all, 0,
                                   offsets_all_bytes, r.stream));
        CHECK_CUDA(cudaMemsetAsync(r.d_route_totals, 0,
                                   (size_t)kGpus * sizeof(int), r.stream));
        gpu_route_count_all_kernel<<<
            (unsigned int)((route_entries + block - 1) / block), block,
            0, r.stream>>>(
            r.d_router_selected_plan, r.d_route_offsets_all,
            (uint32_t)opt.slots, (uint32_t)opt.top_k);
        gpu_route_prefix_all_kernel<<<1, kGpus, 0, r.stream>>>(
            r.d_route_offsets_all, r.d_route_totals);
        gpu_route_init_compact_plan_kernel<<<
            (unsigned int)((compact_route_plan_ints(opt) + block - 1) / block),
            block, 0, r.stream>>>(
            r.d_route_compact_plan, (uint32_t)opt.slots, (uint32_t)opt.top_k);
        gpu_route_copy_own_offsets_kernel<<<1, kLocalExperts + 1, 0, r.stream>>>(
            r.d_offsets, r.d_route_offsets_all, (uint32_t)rank);
        gpu_route_fill_all_kernel<<<
            (unsigned int)((route_entries + block - 1) / block), block,
            0, r.stream>>>(
            r.d_router_selected_plan, r.d_router_weights_plan,
            r.d_route_offsets_all, rank, r.d_route_slots, r.d_route_weights,
            r.d_route_compact_plan, (uint32_t)opt.slots, (uint32_t)opt.top_k);
        CHECK_CUDA(cudaGetLastError());
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }
    std::vector<int> totals((size_t)kGpus, 0);
    std::vector<int> offsets_all((size_t)kGpus * (size_t)(kLocalExperts + 1), 0);
    CHECK_CUDA(cudaSetDevice(ranks[0].device));
    CHECK_CUDA(cudaMemcpy(totals.data(), ranks[0].d_route_totals,
                          totals.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(offsets_all.data(), ranks[0].d_route_offsets_all,
                          offsets_all.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    for (int rank = 0; rank < kGpus; ++rank) {
        if (totals[(size_t)rank] > ranks[rank].route_capacity) return 3;
        ranks[rank].routes = totals[(size_t)rank];
        int active = 0;
        int max_routes = 0;
        const int *off = offsets_all.data() + (size_t)rank * (kLocalExperts + 1);
        for (int local = 0; local < kLocalExperts; ++local) {
            const int count = off[local + 1] - off[local];
            if (count > 0) ++active;
            max_routes = std::max(max_routes, count);
        }
        ranks[rank].active_experts = active;
        ranks[rank].max_routes_per_expert = max_routes;
    }
    static bool route_stats_emitted[43] = {};
    if (opt.model_router_routes && opt.layer >= 0 && opt.layer <= 5 &&
        !route_stats_emitted[opt.layer]) {
        route_stats_emitted[opt.layer] = true;
        std::fprintf(stderr,
                     "tp_ep_model_router_route_stats\tlayer\t%d\troutes\t%d,%d,%d,%d,%d,%d,%d,%d\tmax_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                     opt.layer,
                     ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                     ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                     ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                     ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                     ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                     ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
    }
    int duplicate_slots = 0;
    int max_same_rank_routes = 0;
    std::vector<int> compact_counts((size_t)kGpus * (size_t)opt.slots, 0);
    CHECK_CUDA(cudaSetDevice(ranks[0].device));
    CHECK_CUDA(cudaMemcpy(
        compact_counts.data(),
        ranks[0].d_route_compact_plan +
            (size_t)kGpus * (size_t)opt.slots * (size_t)opt.top_k,
        compact_counts.size() * sizeof(int),
        cudaMemcpyDeviceToHost));
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int rank = 0; rank < kGpus; ++rank) {
            const int c = compact_counts[(size_t)rank * (size_t)opt.slots + slot];
            max_same_rank_routes = std::max(max_same_rank_routes, c);
            if (c > 1) duplicate_slots++;
        }
    }
    static bool compact_stats_emitted[43] = {};
    if (opt.layer >= 0 && opt.layer < 43 && !compact_stats_emitted[opt.layer]) {
        compact_stats_emitted[opt.layer] = true;
        const uint64_t all_dest_bytes =
            (uint64_t)kGpus * (uint64_t)kGpus * (uint64_t)opt.slots *
            (uint64_t)(kHidden / kGpus) * sizeof(float);
        const uint64_t total_routes =
            (uint64_t)ranks[0].routes + (uint64_t)ranks[1].routes +
            (uint64_t)ranks[2].routes + (uint64_t)ranks[3].routes +
            (uint64_t)ranks[4].routes + (uint64_t)ranks[5].routes +
            (uint64_t)ranks[6].routes + (uint64_t)ranks[7].routes;
        const uint64_t compact_bytes =
            (uint64_t)kGpus * total_routes * (uint64_t)(kHidden / kGpus) *
            sizeof(float);
        std::printf("tp_ep_compact_moe_route_stats\tlayer\t%d\t"
                    "duplicate_slots\t%d\tmax_same_rank_routes\t%d\t"
                    "all_dest_bytes\t%llu\tcompact_bytes\t%llu\t"
                    "routes\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                    "active_experts\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                    "max_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                    opt.layer, duplicate_slots, max_same_rank_routes,
                    (unsigned long long)all_dest_bytes,
                    (unsigned long long)compact_bytes,
                    ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                    ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                    ranks[0].active_experts, ranks[1].active_experts,
                    ranks[2].active_experts, ranks[3].active_experts,
                    ranks[4].active_experts, ranks[5].active_experts,
                    ranks[6].active_experts, ranks[7].active_experts,
                    ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                    ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                    ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                    ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
    }
    return 0;
}

int upload_post_attention_fixed_capacity_route_plan_gpu(
    const Options &opt,
    SharedHcControls *hc,
    RankState ranks[kGpus],
    cudaStream_t control_stream,
    bool graph_event_order) {
    if (!opt.compact_moe_decode_gate || !hc || !hc->d_router_selected ||
        !hc->d_router_weights) {
        return 1;
    }
    if (opt.post_attention_device_actual_route_sync_gate && graph_event_order) {
        return 7;
    }
    const int block = 256;
    const uint32_t route_entries = (uint32_t)(opt.slots * opt.top_k);
    const size_t selected_bytes = (size_t)route_entries * sizeof(int);
    const size_t weights_bytes = (size_t)route_entries * sizeof(float);
    const size_t offsets_all_bytes =
        (size_t)kGpus * (size_t)(kLocalExperts + 1) * sizeof(int);
    if (!graph_event_order) {
        void *selected_dsts[kGpus] = {};
        void *weights_dsts[kGpus] = {};
        for (int rank = 0; rank < kGpus; ++rank) {
            selected_dsts[rank] = ranks[rank].d_router_selected_plan;
            weights_dsts[rank] = ranks[rank].d_router_weights_plan;
        }
        if (nccl_broadcast_bytes_from_rank0(
                ranks, hc->d_router_selected, selected_dsts, selected_bytes,
                "post_attention_route_selected") != 0 ||
            nccl_broadcast_bytes_from_rank0(
                ranks, hc->d_router_weights, weights_dsts, weights_bytes,
                "post_attention_route_weights") != 0) {
            return 8;
        }
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.d_router_selected_plan || !r.d_router_weights_plan ||
            !r.d_route_offsets_all || !r.d_route_totals ||
            !r.d_route_compact_plan || !r.d_offsets ||
            !r.d_route_slots || !r.d_route_weights) {
            return 2;
        }
        r.routes = r.route_capacity;
        if (opt.post_attention_static_rank_route_cap > 0) {
            r.routes = std::min(r.routes,
                                opt.post_attention_static_rank_route_cap);
        }
        r.active_experts = kLocalExperts;
        r.max_routes_per_expert = r.routes;
        CHECK_CUDA(cudaSetDevice(r.device));
        if (graph_event_order) {
            enqueue_graph_i32_copy_from_device0(
                opt, r, rank, r.d_router_selected_plan,
                hc->d_router_selected, route_entries, r.stream, block);
            enqueue_graph_f32_copy_from_device0(
                opt, r, rank, r.d_router_weights_plan,
                hc->d_router_weights, route_entries, r.stream, block);
        }
        CHECK_CUDA(cudaMemsetAsync(r.d_route_offsets_all, 0,
                                   offsets_all_bytes, r.stream));
        CHECK_CUDA(cudaMemsetAsync(r.d_route_totals, 0,
                                   (size_t)kGpus * sizeof(int), r.stream));
        CHECK_CUDA(cudaMemsetAsync(r.d_route_slots, 0,
                                   (size_t)r.route_capacity * sizeof(int),
                                   r.stream));
        CHECK_CUDA(cudaMemsetAsync(r.d_route_weights, 0,
                                   (size_t)r.route_capacity * sizeof(float),
                                   r.stream));
        gpu_route_count_all_kernel<<<
            (unsigned int)((route_entries + block - 1) / block), block,
            0, r.stream>>>(
            r.d_router_selected_plan, r.d_route_offsets_all,
            (uint32_t)opt.slots, (uint32_t)opt.top_k);
        gpu_route_prefix_all_kernel<<<1, kGpus, 0, r.stream>>>(
            r.d_route_offsets_all, r.d_route_totals);
        gpu_route_init_compact_plan_kernel<<<
            (unsigned int)((compact_route_plan_ints(opt) + block - 1) / block),
            block, 0, r.stream>>>(
            r.d_route_compact_plan, (uint32_t)opt.slots, (uint32_t)opt.top_k);
        gpu_route_copy_own_offsets_kernel<<<1, kLocalExperts + 1, 0, r.stream>>>(
            r.d_offsets, r.d_route_offsets_all, (uint32_t)rank);
        gpu_route_fill_all_kernel<<<
            (unsigned int)((route_entries + block - 1) / block), block,
            0, r.stream>>>(
            r.d_router_selected_plan, r.d_router_weights_plan,
            r.d_route_offsets_all, rank, r.d_route_slots, r.d_route_weights,
            r.d_route_compact_plan, (uint32_t)opt.slots, (uint32_t)opt.top_k);
        if (opt.post_attention_route_reuse_audit_gate &&
            r.d_post_attn_route_audit) {
            CHECK_CUDA(cudaMemsetAsync(r.d_post_attn_route_audit, 0,
                                       4u * sizeof(unsigned long long),
                                       r.stream));
            post_attention_route_plan_audit_kernel<<<
                (unsigned int)kLocalExperts, 128, 0, r.stream>>>(
                r.d_post_attn_route_audit, r.d_offsets, r.d_route_slots,
                r.d_route_weights, r.d_router_selected_plan,
                r.d_router_weights_plan, (uint32_t)rank,
                (uint32_t)opt.slots, (uint32_t)opt.top_k);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    if (opt.post_attention_device_actual_route_sync_gate) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
        std::vector<int> totals((size_t)kGpus, 0);
        std::vector<int> offsets_all((size_t)kGpus * (size_t)(kLocalExperts + 1), 0);
        CHECK_CUDA(cudaSetDevice(ranks[0].device));
        CHECK_CUDA(cudaMemcpy(totals.data(), ranks[0].d_route_totals,
                              totals.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(offsets_all.data(), ranks[0].d_route_offsets_all,
                              offsets_all.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            if (totals[(size_t)rank] > r.route_capacity) return 8;
            r.routes = totals[(size_t)rank];
            int active = 0;
            int max_routes = 0;
            const int *off = offsets_all.data() + (size_t)rank * (kLocalExperts + 1);
            for (int local = 0; local < kLocalExperts; ++local) {
                const int count = off[local + 1] - off[local];
                if (count > 0) ++active;
                max_routes = std::max(max_routes, count);
            }
            r.active_experts = active;
            r.max_routes_per_expert = max_routes;
        }
    }
    (void)control_stream;
    return 0;
}

int upload_model_router_route_plan(const Options &opt,
                                   RankState ranks[kGpus],
                                   const std::vector<int> &selected,
                                   const std::vector<float> &weights) {
    if ((int)selected.size() < opt.slots * opt.top_k ||
        (int)weights.size() < opt.slots * opt.top_k) {
        return 1;
    }
    std::vector<int> offsets[kGpus];
    std::vector<int> route_slots[kGpus];
    std::vector<float> route_weights[kGpus];
    std::vector<int> route_index_by_slot[kGpus];
    std::vector<int> route_indices_by_slot[kGpus];
    std::vector<int> route_count_by_slot[kGpus];
    std::vector<int> counts[kGpus];
    const bool needs_single_route_index = !opt.compact_moe_decode_gate;
    const bool needs_packed_compact_plan = opt.compact_moe_decode_gate;
    for (int rank = 0; rank < kGpus; ++rank) {
        counts[rank].assign((size_t)kLocalExperts, 0);
        if (needs_single_route_index) {
            route_index_by_slot[rank].assign((size_t)opt.slots, -1);
        }
        route_indices_by_slot[rank].assign((size_t)opt.slots * (size_t)opt.top_k,
                                           -1);
        route_count_by_slot[rank].assign((size_t)opt.slots, 0);
    }
    bool compact_duplicate = false;
    for (int slot = 0; slot < opt.slots; ++slot) {
        bool seen_rank[kGpus] = {};
        for (int k = 0; k < opt.top_k; ++k) {
            const int expert = selected[(size_t)slot * opt.top_k + (size_t)k];
            if (expert < 0) continue;
            if (expert < 0 || expert >= kGlobalExperts) return 2;
            const int rank = expert / kLocalExperts;
            const int local = expert % kLocalExperts;
            counts[rank][(size_t)local]++;
            if (seen_rank[rank]) compact_duplicate = true;
            seen_rank[rank] = true;
        }
    }
    if (opt.compact_route_compose && compact_duplicate &&
        !opt.compact_moe_decode_gate) {
        return 3;
    }
    if (opt.routed_ffn_norm_input_gate && opt.layer >= 0 && opt.layer <= 2) {
        for (int slot = 0; slot < opt.slots; ++slot) {
            for (int k = 0; k < opt.top_k; ++k) {
                const int expert = selected[(size_t)slot * opt.top_k + (size_t)k];
                if (expert < 0) continue;
                const int rank = expert / kLocalExperts;
                const int local = expert % kLocalExperts;
                const float w = weights[(size_t)slot * opt.top_k + (size_t)k];
                StridedPtrH gw = {};
                StridedPtrH gs = {};
                StridedPtrH dw = {};
                StridedPtrH ds = {};
                if (rank >= 0 && rank < kGpus) {
                    CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                    if (ranks[rank].gated.d_w_table) {
                        CHECK_CUDA(cudaMemcpy(&gw,
                                              (const StridedPtrH *)ranks[rank].gated.d_w_table + local,
                                              sizeof(gw), cudaMemcpyDeviceToHost));
                    }
                    if (ranks[rank].gated.d_s_table) {
                        CHECK_CUDA(cudaMemcpy(&gs,
                                              (const StridedPtrH *)ranks[rank].gated.d_s_table + local,
                                              sizeof(gs), cudaMemcpyDeviceToHost));
                    }
                    if (ranks[rank].down.d_w_table) {
                        CHECK_CUDA(cudaMemcpy(&dw,
                                              (const StridedPtrH *)ranks[rank].down.d_w_table + local,
                                              sizeof(dw), cudaMemcpyDeviceToHost));
                    }
                    if (ranks[rank].down.d_s_table) {
                        CHECK_CUDA(cudaMemcpy(&ds,
                                              (const StridedPtrH *)ranks[rank].down.d_s_table + local,
                                              sizeof(ds), cudaMemcpyDeviceToHost));
                    }
                }
                std::fprintf(stderr,
                             "tp_ep_model_router_route_id\tlayer\t%d\tslot\t%d\tk\t%d\texpert\t%d\trank\t%d\tlocal\t%d\tweight\t%.9g\tgated_w\t%p\tgated_ws\t%d\tgated_s\t%p\tgated_ss\t%d\tdown_w\t%p\tdown_ws\t%d\tdown_s\t%p\tdown_ss\t%d\n",
                             opt.layer, slot, k, expert, rank, local, w,
                             gw.p, gw.stride, gs.p, gs.stride,
                             dw.p, dw.stride, ds.p, ds.stride);
            }
        }
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        offsets[rank].assign((size_t)kLocalExperts + 1, 0);
        int running = 0;
        int active = 0;
        int max_routes = 0;
        for (int e = 0; e < kLocalExperts; ++e) {
            offsets[rank][(size_t)e] = running;
            running += counts[rank][(size_t)e];
            if (counts[rank][(size_t)e] > 0) ++active;
            max_routes = std::max(max_routes, counts[rank][(size_t)e]);
        }
        offsets[rank][(size_t)kLocalExperts] = running;
        if (running > ranks[rank].route_capacity) return 4;
        route_slots[rank].assign((size_t)running, -1);
        route_weights[rank].assign((size_t)running, 0.0f);
        std::vector<int> cursor = offsets[rank];
        for (int slot = 0; slot < opt.slots; ++slot) {
            for (int k = 0; k < opt.top_k; ++k) {
                const int expert = selected[(size_t)slot * opt.top_k + (size_t)k];
                if (expert < 0) continue;
                const int dst_rank = expert / kLocalExperts;
                if (dst_rank != rank) continue;
                const int local = expert % kLocalExperts;
                const int idx = cursor[(size_t)local]++;
                route_slots[rank][(size_t)idx] = slot;
                const float w = weights[(size_t)slot * opt.top_k + (size_t)k];
                if (!std::isfinite(w)) {
                    std::fprintf(stderr,
                                 "tp_ep_model_router_nonfinite_weight\trank\t%d\tslot\t%d\texpert\t%d\tk\t%d\n",
                                 rank, slot, expert, k);
                    return 5;
                }
                route_weights[rank][(size_t)idx] = w;
                if (needs_single_route_index &&
                    route_index_by_slot[rank][(size_t)slot] < 0) {
                    route_index_by_slot[rank][(size_t)slot] = idx;
                }
                int &route_count = route_count_by_slot[rank][(size_t)slot];
                if (route_count >= opt.top_k) return 6;
                route_indices_by_slot[rank][(size_t)slot * (size_t)opt.top_k +
                                            (size_t)route_count] = idx;
                route_count++;
            }
        }
        RankState &r = ranks[rank];
        r.routes = running;
        r.active_experts = active;
        r.max_routes_per_expert = max_routes;
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaMemcpy(r.d_offsets, offsets[rank].data(),
                              offsets[rank].size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(r.d_route_slots, route_slots[rank].data(),
                              route_slots[rank].size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(r.d_route_weights, route_weights[rank].data(),
                              route_weights[rank].size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    static bool route_stats_emitted[43] = {};
    if (opt.model_router_routes && opt.layer >= 0 && opt.layer <= 5 &&
        !route_stats_emitted[opt.layer]) {
        route_stats_emitted[opt.layer] = true;
        std::fprintf(stderr,
                     "tp_ep_model_router_route_stats\tlayer\t%d\troutes\t%d,%d,%d,%d,%d,%d,%d,%d\tmax_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                     opt.layer,
                     ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                     ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                     ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                     ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                     ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                     ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
    }
    if (opt.compact_moe_decode_gate && opt.model_router_routes) {
        int duplicate_slots = 0;
        int max_same_rank_routes = 0;
        for (int slot = 0; slot < opt.slots; ++slot) {
            for (int rank = 0; rank < kGpus; ++rank) {
                const int c = route_count_by_slot[rank][(size_t)slot];
                max_same_rank_routes = std::max(max_same_rank_routes, c);
                if (c > 1) duplicate_slots++;
            }
        }
        static bool compact_stats_emitted[43] = {};
        if (opt.layer >= 0 && opt.layer < 43 && !compact_stats_emitted[opt.layer]) {
            compact_stats_emitted[opt.layer] = true;
            const uint64_t all_dest_bytes =
                (uint64_t)kGpus * (uint64_t)kGpus * (uint64_t)opt.slots *
                (uint64_t)(kHidden / kGpus) * sizeof(float);
            const uint64_t total_routes =
                (uint64_t)ranks[0].routes + (uint64_t)ranks[1].routes +
                (uint64_t)ranks[2].routes + (uint64_t)ranks[3].routes +
                (uint64_t)ranks[4].routes + (uint64_t)ranks[5].routes +
                (uint64_t)ranks[6].routes + (uint64_t)ranks[7].routes;
            const uint64_t compact_bytes =
                (uint64_t)kGpus * total_routes * (uint64_t)(kHidden / kGpus) *
                sizeof(float);
            std::printf("tp_ep_compact_moe_route_stats\tlayer\t%d\t"
                        "duplicate_slots\t%d\tmax_same_rank_routes\t%d\t"
                        "all_dest_bytes\t%llu\tcompact_bytes\t%llu\t"
                        "routes\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                        "active_experts\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                        "max_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                        opt.layer, duplicate_slots, max_same_rank_routes,
                        (unsigned long long)all_dest_bytes,
                        (unsigned long long)compact_bytes,
                        ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                        ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                        ranks[0].active_experts, ranks[1].active_experts,
                        ranks[2].active_experts, ranks[3].active_experts,
                        ranks[4].active_experts, ranks[5].active_experts,
                        ranks[6].active_experts, ranks[7].active_experts,
                        ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                        ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                        ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                        ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
        }
    }
    std::vector<int> compact_plan;
    if (needs_packed_compact_plan) {
        compact_plan.assign(compact_route_plan_ints(opt), -1);
        const size_t compact_indices = (size_t)opt.slots * (size_t)opt.top_k;
        const size_t compact_counts = (size_t)opt.slots;
        for (int src = 0; src < kGpus; ++src) {
            std::copy(route_indices_by_slot[src].begin(),
                      route_indices_by_slot[src].end(),
                      compact_plan.begin() + (size_t)src * compact_indices);
            std::copy(route_count_by_slot[src].begin(),
                      route_count_by_slot[src].end(),
                      compact_plan.begin() + (size_t)kGpus * compact_indices +
                          (size_t)src * compact_counts);
        }
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        CHECK_CUDA(cudaSetDevice(ranks[dst].device));
        for (int src = 0; src < kGpus; ++src) {
            if (needs_single_route_index) {
                CHECK_CUDA(cudaMemcpy(ranks[dst].d_route_index_by_slot[src],
                                      route_index_by_slot[src].data(),
                                      route_index_by_slot[src].size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
            }
            if (!needs_packed_compact_plan) {
                CHECK_CUDA(cudaMemcpy(ranks[dst].d_route_indices_by_slot[src],
                                      route_indices_by_slot[src].data(),
                                      route_indices_by_slot[src].size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(ranks[dst].d_route_count_by_slot[src],
                                      route_count_by_slot[src].data(),
                                      route_count_by_slot[src].size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
            }
        }
        if (needs_packed_compact_plan) {
            if (!ranks[dst].d_route_compact_plan ||
                ranks[dst].route_compact_plan_ints < compact_plan.size()) {
                return 7;
            }
            CHECK_CUDA(cudaMemcpy(ranks[dst].d_route_compact_plan,
                                  compact_plan.data(),
                                  compact_plan.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }
    }
    return 0;
}

int upload_model_router_route_plan_async(const Options &opt,
                                         RankState ranks[kGpus],
                                         const int *selected,
                                         const float *weights,
                                         RoutePlanHostWorkspace *ws) {
    if (!selected || !weights || !ws || !ws->initialized ||
        ws->slots != opt.slots || ws->top_k != opt.top_k ||
        ws->route_capacity < (size_t)opt.slots * (size_t)opt.top_k) {
        return 1;
    }
    if (opt.routed_ffn_norm_input_gate) {
        return 8;
    }
    if (ws->uploads_pending) {
        for (int rank = 0; rank < kGpus; ++rank) {
            if (ws->upload_done[rank]) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaEventSynchronize(ws->upload_done[rank]));
            }
        }
        ws->uploads_pending = false;
    }

    const bool needs_single_route_index = !opt.compact_moe_decode_gate;
    const bool needs_packed_compact_plan = opt.compact_moe_decode_gate;
    std::vector<int> counts[kGpus];
    std::vector<int> cursor[kGpus];
    for (int rank = 0; rank < kGpus; ++rank) {
        counts[rank].assign((size_t)kLocalExperts, 0);
        std::fill(ws->h_route_indices_by_slot[rank],
                  ws->h_route_indices_by_slot[rank] +
                      (size_t)opt.slots * (size_t)opt.top_k,
                  -1);
        std::fill(ws->h_route_count_by_slot[rank],
                  ws->h_route_count_by_slot[rank] + (size_t)opt.slots,
                  0);
        if (needs_single_route_index) {
            std::fill(ws->h_route_index_by_slot[rank],
                      ws->h_route_index_by_slot[rank] + (size_t)opt.slots,
                      -1);
        }
    }

    bool compact_duplicate = false;
    for (int slot = 0; slot < opt.slots; ++slot) {
        bool seen_rank[kGpus] = {};
        for (int k = 0; k < opt.top_k; ++k) {
            const int expert = selected[(size_t)slot * (size_t)opt.top_k + (size_t)k];
            if (expert < 0) continue;
            if (expert >= kGlobalExperts) return 2;
            const int rank = expert / kLocalExperts;
            const int local = expert % kLocalExperts;
            counts[rank][(size_t)local]++;
            if (seen_rank[rank]) compact_duplicate = true;
            seen_rank[rank] = true;
        }
    }
    if (opt.compact_route_compose && compact_duplicate &&
        !opt.compact_moe_decode_gate) {
        return 3;
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        int running = 0;
        int active = 0;
        int max_routes = 0;
        for (int e = 0; e < kLocalExperts; ++e) {
            ws->h_offsets[rank][e] = running;
            running += counts[rank][(size_t)e];
            if (counts[rank][(size_t)e] > 0) ++active;
            max_routes = std::max(max_routes, counts[rank][(size_t)e]);
        }
        ws->h_offsets[rank][kLocalExperts] = running;
        if (running > ranks[rank].route_capacity ||
            (size_t)running > ws->route_capacity) {
            return 4;
        }
        std::fill(ws->h_route_slots[rank],
                  ws->h_route_slots[rank] + (size_t)running, -1);
        std::fill(ws->h_route_weights[rank],
                  ws->h_route_weights[rank] + (size_t)running, 0.0f);
        cursor[rank].assign(ws->h_offsets[rank],
                            ws->h_offsets[rank] + kLocalExperts + 1);
    }

    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int k = 0; k < opt.top_k; ++k) {
            const size_t route_key = (size_t)slot * (size_t)opt.top_k + (size_t)k;
            const int expert = selected[route_key];
            if (expert < 0) continue;
            const int rank = expert / kLocalExperts;
            const int local = expert % kLocalExperts;
            const int idx = cursor[rank][(size_t)local]++;
            ws->h_route_slots[rank][idx] = slot;
            const float w = weights[route_key];
            if (!std::isfinite(w)) {
                std::fprintf(stderr,
                             "tp_ep_model_router_nonfinite_weight\trank\t%d\tslot\t%d\texpert\t%d\tk\t%d\n",
                             rank, slot, expert, k);
                return 5;
            }
            ws->h_route_weights[rank][idx] = w;
            if (needs_single_route_index &&
                ws->h_route_index_by_slot[rank][slot] < 0) {
                ws->h_route_index_by_slot[rank][slot] = idx;
            }
            int &route_count = ws->h_route_count_by_slot[rank][slot];
            if (route_count >= opt.top_k) return 6;
            ws->h_route_indices_by_slot[rank]
                [(size_t)slot * (size_t)opt.top_k + (size_t)route_count] = idx;
            route_count++;
        }
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        const int running = ws->h_offsets[rank][kLocalExperts];
        int active = 0;
        int max_routes = 0;
        for (int e = 0; e < kLocalExperts; ++e) {
            if (counts[rank][(size_t)e] > 0) ++active;
            max_routes = std::max(max_routes, counts[rank][(size_t)e]);
        }
        r.routes = running;
        r.active_experts = active;
        r.max_routes_per_expert = max_routes;
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaMemcpyAsync(r.d_offsets, ws->h_offsets[rank],
                                   (size_t)(kLocalExperts + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, r.stream));
        if (running > 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_route_slots, ws->h_route_slots[rank],
                                       (size_t)running * sizeof(int),
                                       cudaMemcpyHostToDevice, r.stream));
            CHECK_CUDA(cudaMemcpyAsync(r.d_route_weights, ws->h_route_weights[rank],
                                       (size_t)running * sizeof(float),
                                       cudaMemcpyHostToDevice, r.stream));
        }
    }

    static bool route_stats_emitted[43] = {};
    if (opt.model_router_routes && opt.layer >= 0 && opt.layer <= 5 &&
        !route_stats_emitted[opt.layer]) {
        route_stats_emitted[opt.layer] = true;
        std::fprintf(stderr,
                     "tp_ep_model_router_route_stats_async\tlayer\t%d\troutes\t%d,%d,%d,%d,%d,%d,%d,%d\tmax_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                     opt.layer,
                     ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                     ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                     ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                     ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                     ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                     ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
    }
    if (opt.compact_moe_decode_gate && opt.model_router_routes) {
        int duplicate_slots = 0;
        int max_same_rank_routes = 0;
        for (int slot = 0; slot < opt.slots; ++slot) {
            for (int rank = 0; rank < kGpus; ++rank) {
                const int c = ws->h_route_count_by_slot[rank][slot];
                max_same_rank_routes = std::max(max_same_rank_routes, c);
                if (c > 1) duplicate_slots++;
            }
        }
        static bool compact_stats_emitted[43] = {};
        if (opt.layer >= 0 && opt.layer < 43 && !compact_stats_emitted[opt.layer]) {
            compact_stats_emitted[opt.layer] = true;
            const uint64_t all_dest_bytes =
                (uint64_t)kGpus * (uint64_t)kGpus * (uint64_t)opt.slots *
                (uint64_t)(kHidden / kGpus) * sizeof(float);
            const uint64_t total_routes =
                (uint64_t)ranks[0].routes + (uint64_t)ranks[1].routes +
                (uint64_t)ranks[2].routes + (uint64_t)ranks[3].routes +
                (uint64_t)ranks[4].routes + (uint64_t)ranks[5].routes +
                (uint64_t)ranks[6].routes + (uint64_t)ranks[7].routes;
            const uint64_t compact_bytes =
                (uint64_t)kGpus * total_routes * (uint64_t)(kHidden / kGpus) *
                sizeof(float);
            std::printf("tp_ep_compact_moe_route_stats_async\tlayer\t%d\t"
                        "duplicate_slots\t%d\tmax_same_rank_routes\t%d\t"
                        "all_dest_bytes\t%llu\tcompact_bytes\t%llu\t"
                        "routes\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                        "active_experts\t%d,%d,%d,%d,%d,%d,%d,%d\t"
                        "max_routes_per_expert\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                        opt.layer, duplicate_slots, max_same_rank_routes,
                        (unsigned long long)all_dest_bytes,
                        (unsigned long long)compact_bytes,
                        ranks[0].routes, ranks[1].routes, ranks[2].routes, ranks[3].routes,
                        ranks[4].routes, ranks[5].routes, ranks[6].routes, ranks[7].routes,
                        ranks[0].active_experts, ranks[1].active_experts,
                        ranks[2].active_experts, ranks[3].active_experts,
                        ranks[4].active_experts, ranks[5].active_experts,
                        ranks[6].active_experts, ranks[7].active_experts,
                        ranks[0].max_routes_per_expert, ranks[1].max_routes_per_expert,
                        ranks[2].max_routes_per_expert, ranks[3].max_routes_per_expert,
                        ranks[4].max_routes_per_expert, ranks[5].max_routes_per_expert,
                        ranks[6].max_routes_per_expert, ranks[7].max_routes_per_expert);
        }
    }

    if (needs_packed_compact_plan) {
        if (ws->compact_plan_ints <
            (size_t)kGpus * ((size_t)opt.slots * (size_t)opt.top_k +
                             (size_t)opt.slots)) {
            return 7;
        }
        std::fill(ws->h_compact_plan,
                  ws->h_compact_plan + ws->compact_plan_ints, -1);
        const size_t compact_indices = (size_t)opt.slots * (size_t)opt.top_k;
        const size_t compact_counts = (size_t)opt.slots;
        for (int src = 0; src < kGpus; ++src) {
            std::memcpy(ws->h_compact_plan + (size_t)src * compact_indices,
                        ws->h_route_indices_by_slot[src],
                        compact_indices * sizeof(int));
            std::memcpy(ws->h_compact_plan + (size_t)kGpus * compact_indices +
                            (size_t)src * compact_counts,
                        ws->h_route_count_by_slot[src],
                        compact_counts * sizeof(int));
        }
    }

    for (int dst = 0; dst < kGpus; ++dst) {
        CHECK_CUDA(cudaSetDevice(ranks[dst].device));
        for (int src = 0; src < kGpus; ++src) {
            if (needs_single_route_index) {
                CHECK_CUDA(cudaMemcpyAsync(ranks[dst].d_route_index_by_slot[src],
                                           ws->h_route_index_by_slot[src],
                                           (size_t)opt.slots * sizeof(int),
                                           cudaMemcpyHostToDevice,
                                           ranks[dst].stream));
            }
            if (!needs_packed_compact_plan) {
                CHECK_CUDA(cudaMemcpyAsync(ranks[dst].d_route_indices_by_slot[src],
                                           ws->h_route_indices_by_slot[src],
                                           (size_t)opt.slots *
                                               (size_t)opt.top_k * sizeof(int),
                                           cudaMemcpyHostToDevice,
                                           ranks[dst].stream));
                CHECK_CUDA(cudaMemcpyAsync(ranks[dst].d_route_count_by_slot[src],
                                           ws->h_route_count_by_slot[src],
                                           (size_t)opt.slots * sizeof(int),
                                           cudaMemcpyHostToDevice,
                                           ranks[dst].stream));
            }
        }
        if (needs_packed_compact_plan) {
            if (!ranks[dst].d_route_compact_plan ||
                ranks[dst].route_compact_plan_ints < ws->compact_plan_ints) {
                return 7;
            }
            CHECK_CUDA(cudaMemcpyAsync(ranks[dst].d_route_compact_plan,
                                       ws->h_compact_plan,
                                       ws->compact_plan_ints * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       ranks[dst].stream));
        }
        CHECK_CUDA(cudaEventRecord(ws->upload_done[dst], ranks[dst].stream));
    }
    ws->uploads_pending = true;
    return 0;
}

void print_post_attention_route_reuse_audit(const Options &opt,
                                            RankState ranks[kGpus],
                                            const char *label) {
    if (!opt.post_attention_route_reuse_audit_gate) return;
    unsigned long long total[4] = {};
    int cap_overflow = 0;
    int cap_max_total = 0;
    int compose_cap_overflow = 0;
    int compose_cap_max_total = 0;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.d_post_attn_route_audit) continue;
        unsigned long long h[4] = {};
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaMemcpy(h, r.d_post_attn_route_audit,
                              sizeof(h), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 4; ++i) total[i] += h[i];
        std::printf("tp_ep_post_attention_route_reuse_audit\tlabel\t%s\t"
                    "layer\t%d\trank\t%d\troutes_checked\t%llu\t"
                    "missing_selected\t%llu\tweight_mismatch\t%llu\t"
                    "invalid_slot\t%llu\n",
                    label ? label : "unknown", opt.layer, rank,
                    h[0], h[1], h[2], h[3]);
        if (opt.post_attention_static_rank_route_cap > 0 &&
            r.d_route_totals) {
            int route_total = 0;
            CHECK_CUDA(cudaMemcpy(&route_total, r.d_route_totals + rank,
                                  sizeof(route_total),
                                  cudaMemcpyDeviceToHost));
            cap_max_total = std::max(cap_max_total, route_total);
            if (route_total > opt.post_attention_static_rank_route_cap) {
                ++cap_overflow;
            }
            std::printf("tp_ep_static_route_cap_audit\tlabel\t%s\t"
                        "layer\t%d\trank\t%d\tcap\t%d\tactual\t%d\t%s\n",
                        label ? label : "unknown", opt.layer, rank,
                        opt.post_attention_static_rank_route_cap, route_total,
                        route_total <= opt.post_attention_static_rank_route_cap
                            ? "PASS" : "OVERFLOW");
        }
        if (opt.post_attention_static_compose_route_cap > 0 &&
            r.d_route_totals) {
            int route_total = 0;
            CHECK_CUDA(cudaMemcpy(&route_total, r.d_route_totals + rank,
                                  sizeof(route_total),
                                  cudaMemcpyDeviceToHost));
            compose_cap_max_total = std::max(compose_cap_max_total, route_total);
            if (route_total > opt.post_attention_static_compose_route_cap) {
                ++compose_cap_overflow;
            }
            std::printf("tp_ep_static_compose_route_cap_audit\tlabel\t%s\t"
                        "layer\t%d\trank\t%d\tcap\t%d\tactual\t%d\t%s\n",
                        label ? label : "unknown", opt.layer, rank,
                        opt.post_attention_static_compose_route_cap,
                        route_total,
                        route_total <= opt.post_attention_static_compose_route_cap
                            ? "PASS" : "OVERFLOW");
        }
    }
    std::printf("tp_ep_post_attention_route_reuse_audit_total\tlabel\t%s\t"
                "layer\t%d\troutes_checked\t%llu\tmissing_selected\t%llu\t"
                "weight_mismatch\t%llu\tinvalid_slot\t%llu\n",
                label ? label : "unknown", opt.layer,
                total[0], total[1], total[2], total[3]);
    if (opt.post_attention_static_rank_route_cap > 0) {
        std::printf("tp_ep_static_route_cap_audit_total\tlabel\t%s\t"
                    "layer\t%d\tcap\t%d\tmax_actual\t%d\toverflow_ranks\t%d\t%s\n",
                    label ? label : "unknown", opt.layer,
                    opt.post_attention_static_rank_route_cap, cap_max_total,
                    cap_overflow, cap_overflow == 0 ? "PASS" : "OVERFLOW");
    }
    if (opt.post_attention_static_compose_route_cap > 0) {
        std::printf("tp_ep_static_compose_route_cap_audit_total\tlabel\t%s\t"
                    "layer\t%d\tcap\t%d\tmax_actual\t%d\toverflow_ranks\t%d\t%s\n",
                    label ? label : "unknown", opt.layer,
                    opt.post_attention_static_compose_route_cap,
                    compose_cap_max_total, compose_cap_overflow,
                    compose_cap_overflow == 0 ? "PASS" : "OVERFLOW");
    }
}

