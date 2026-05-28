int run_model_router_dense_logits(const Options &opt,
                                  SharedHcControls *hc,
                                  int layer,
                                  cudaStream_t stream) {
    if (!hc || !hc->d_router_w[layer] || !hc->d_router_logits ||
        !hc->d_ffn_normed) {
        return 1;
    }
    if (!opt.router_cublas_gate) {
        const dim3 router_grid((unsigned int)kGlobalExperts,
                               (unsigned int)opt.slots, 1u);
        f32_dense_colmajor_kernel<<<router_grid, 256, 0, stream>>>(
            hc->d_router_logits, hc->d_router_w[layer], hc->d_ffn_normed,
            (uint32_t)kGlobalExperts, (uint32_t)kHidden, (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
        return 0;
    }
    if (!hc->router_blas) return 2;
    cublasStatus_t st = cublasSetStream(hc->router_blas, stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "router cublasSetStream failed status=%d\n", (int)st);
        return 3;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    st = cublasSgemm(hc->router_blas,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     kGlobalExperts, opt.slots, kHidden,
                     &alpha,
                     hc->d_router_w[layer], kGlobalExperts,
                     hc->d_ffn_normed, kHidden,
                     &beta,
                     hc->d_router_logits, kGlobalExperts);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "router cublasSgemm failed layer=%d status=%d\n",
                     layer, (int)st);
        return 4;
    }
    return 0;
}

int run_model_router_rank_major_logits(const Options &opt,
                                       SharedHcControls *hc,
                                       RankState ranks[kGpus],
                                       int layer,
                                       cudaStream_t control_stream,
                                       bool post_attention_input) {
    if (!opt.model_router_rank_major_logits_gate) return 0;
    if (!hc || !hc->d_router_logits || layer < 0 || layer >= 43) return 1;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.compose_nccl_initialized || !r.compose_nccl ||
            !(post_attention_input ? r.d_post_attn_full_rank_major
                                   : r.d_current_full_rank_major) ||
            !r.d_rank_major_norm_scale ||
            !r.d_router_logits_shard || !r.d_router_logits_rank_major ||
            !hc->d_ffn_norm_weight_rank[layer][rank] ||
            !hc->d_router_w_ep[layer][rank]) {
            return 2;
        }
    }
    const uint32_t shard_cols = (uint32_t)(kHidden / kGpus);
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        const float *rank_major = post_attention_input
            ? r.d_post_attn_full_rank_major
            : r.d_current_full_rank_major;
        CHECK_CUDA(cudaSetDevice(r.device));
        rank_major_norm_scale_kernel<<<
            (unsigned int)opt.slots, 256, 0, r.stream>>>(
            r.d_rank_major_norm_scale, rank_major,
            shard_cols, (uint32_t)kGpus, (uint32_t)opt.slots, 1.0e-6f);
        const dim3 grid((unsigned int)kLocalExperts, (unsigned int)opt.slots, 1u);
        router_logits_ep_from_rank_major_kernel<<<grid, 256, 0, r.stream>>>(
            r.d_router_logits_shard, rank_major,
            hc->d_ffn_norm_weight_rank[layer][rank],
            r.d_rank_major_norm_scale,
            hc->d_router_w_ep[layer][rank],
            shard_cols, (uint32_t)kGpus, (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclAllGather(r.d_router_logits_shard,
                                 r.d_router_logits_rank_major,
                                 (size_t)opt.slots * kLocalExperts,
                                 ncclFloat,
                                 r.compose_nccl,
                                 r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    if (opt.decode_cudagraph_gate) {
        if (enqueue_control_wait_after_rank_streams(
                opt, ranks, control_stream) != 0) {
            return 3;
        }
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    const uint64_t elems = (uint64_t)opt.slots * (uint64_t)kGlobalExperts;
    router_logits_rank_major_to_slot_major_kernel<<<
        (unsigned int)((elems + 255ull) / 256ull), 256, 0, control_stream>>>(
        hc->d_router_logits, ranks[0].d_router_logits_rank_major,
        (uint32_t)opt.slots);
    CHECK_CUDA(cudaGetLastError());
    return 0;
}

int run_model_router_allreduce_logits(const Options &opt,
                                      SharedHcControls *hc,
                                      RankState ranks[kGpus],
                                      int layer,
                                      cudaStream_t control_stream,
                                      bool post_attention_input) {
    if (!opt.model_router_allreduce_logits_gate) return 0;
    if (opt.decode_cudagraph_gate) return 11;
    if (!hc || !hc->d_router_logits || layer < 0 || layer >= 43) return 1;
    const uint32_t shard_cols = (uint32_t)(kHidden / kGpus);
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.compose_nccl_initialized || !r.compose_nccl ||
            !(post_attention_input ? r.d_post_attn_shard : r.d_current_shard) ||
            !r.d_hc_reduce_max ||
            !r.d_hc_reduce_sumsq || !r.d_router_logits_rank_major ||
            !hc->d_ffn_norm_weight_rank[layer][rank] ||
            !hc->d_router_w_shard[layer][rank]) {
            return 2;
        }
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        const float *input_shard = post_attention_input
            ? r.d_post_attn_shard
            : r.d_current_shard;
        CHECK_CUDA(cudaSetDevice(r.device));
        current_shard_max_kernel<<<
            (unsigned int)opt.slots, 256, 0, r.stream>>>(
            r.d_hc_reduce_max, input_shard, shard_cols,
            (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclAllReduce(r.d_hc_reduce_max, r.d_hc_reduce_max,
                                 (size_t)opt.slots, ncclFloat, ncclMax,
                                 r.compose_nccl, r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        const float *input_shard = post_attention_input
            ? r.d_post_attn_shard
            : r.d_current_shard;
        CHECK_CUDA(cudaSetDevice(r.device));
        current_shard_stable_sumsq_kernel<<<
            (unsigned int)opt.slots, 256, 0, r.stream>>>(
            r.d_hc_reduce_sumsq, input_shard, r.d_hc_reduce_max,
            shard_cols, (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclAllReduce(r.d_hc_reduce_sumsq, r.d_hc_reduce_sumsq,
                                 (size_t)opt.slots, ncclFloat, ncclSum,
                                 r.compose_nccl, r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        const float *input_shard = post_attention_input
            ? r.d_post_attn_shard
            : r.d_current_shard;
        CHECK_CUDA(cudaSetDevice(r.device));
        const dim3 grid((unsigned int)kGlobalExperts,
                        (unsigned int)opt.slots, 1u);
        router_logits_allreduce_partial_kernel<<<grid, 256, 0, r.stream>>>(
            r.d_router_logits_rank_major, input_shard,
            hc->d_ffn_norm_weight_rank[layer][rank], r.d_hc_reduce_max,
            r.d_hc_reduce_sumsq, hc->d_router_w_shard[layer][rank],
            (uint32_t)rank, shard_cols, (uint32_t)opt.slots, 1.0e-6f);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclAllReduce(r.d_router_logits_rank_major,
                                 r.d_router_logits_rank_major,
                                 (size_t)opt.slots * kGlobalExperts,
                                 ncclFloat, ncclSum, r.compose_nccl,
                                 r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaMemcpyAsync(hc->d_router_logits,
                               ranks[0].d_router_logits_rank_major,
                               (size_t)opt.slots * kGlobalExperts *
                                   sizeof(float),
                               cudaMemcpyDeviceToDevice,
                               control_stream));
    CHECK_CUDA(cudaStreamSynchronize(control_stream));
    return 0;
}

