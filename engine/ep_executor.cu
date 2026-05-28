int run_gate(RankState &rank, const Api &api, int executor_rows) {
    if (executor_rows <= 0) return 0;
    return api.mmgs(rank.d_a, nullptr, rank.d_offsets, kLocalExperts, executor_rows,
                    (const void * const *)rank.gated.d_w_table,
                    (const void * const *)rank.gated.d_s_table,
                    kDType, kFusedN, kHidden, kGroupSize, rank.gated.k_pack,
                    rank.d_gated, rank.stream);
}

int run_gate_clamped(RankState &rank, const Api &api, bool apply_route_scale,
                     int executor_rows) {
    if (executor_rows <= 0) return 0;
    if (!rank.d_gate_up) return 1;
    CHECK_CUDA(cudaSetDevice(rank.device));
    const int rc = api.mmgt(rank.d_a, nullptr, rank.d_offsets, kLocalExperts, executor_rows,
                            (const void * const *)rank.gated.d_w_table,
                            (const void * const *)rank.gated.d_s_table,
                            kDType, kFusedN, kHidden, kGroupSize, rank.gated.k_pack,
                            rank.d_gate_up, rank.stream);
    if (rc != 0) return rc;
    const uint64_t elems = (uint64_t)executor_rows * kMid;
    routed_fused_gate_up_swiglu_clamp_kernel<<<
        (unsigned int)((elems + 255) / 256), 256, 0, rank.stream>>>(
            rank.d_gated, rank.d_gate_up,
            apply_route_scale ? rank.d_route_inv_scale : nullptr,
            (uint64_t)executor_rows, kRoutedSwigluClamp);
    CHECK_CUDA(cudaGetLastError());
    return 0;
}

int run_gate_selected(RankState &rank, const Api &api, const Options &opt) {
    const int executor_rows = routed_executor_rows(rank, opt);
    if (!opt.routed_ffn_norm_input_gate) {
        return run_gate(rank, api, executor_rows);
    }
    if (opt.fused_gated_silu_gate && !opt.reference_hc_reduce_gate &&
        api.mmgs_clamped) {
        return api.mmgs_clamped(
            rank.d_a, nullptr, rank.d_offsets, kLocalExperts, executor_rows,
            (const void * const *)rank.gated.d_w_table,
            (const void * const *)rank.gated.d_s_table,
            kDType, kFusedN, kHidden, kGroupSize, rank.gated.k_pack,
            rank.d_gated, rank.stream);
    }
    return run_gate_clamped(rank, api, opt.reference_hc_reduce_gate,
                            executor_rows);
}

int run_down(RankState &rank, const Api &api, const Options &opt) {
    const int executor_rows = routed_executor_rows(rank, opt);
    if (executor_rows <= 0) return 0;
    return api.mmgt(rank.d_gated, nullptr, rank.d_offsets, kLocalExperts, executor_rows,
                    (const void * const *)rank.down.d_w_table,
                    (const void * const *)rank.down.d_s_table,
                    kDType, kHidden, kMid, kGroupSize, rank.down.k_pack,
                    rank.d_down, rank.stream);
}

