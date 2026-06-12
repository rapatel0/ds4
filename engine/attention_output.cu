int run_true_ds4_attention_output_projection(const Options &opt,
                                             const LayerDenseOps *ops,
                                             RankState ranks[kGpus],
                                             int layer) {
    if (!ops || !ops->initialized || layer < 0 || layer >= 44) {
        return 1;
    }
    if (ops->attn_output_a.cols != kAttentionOutputAInput ||
        ops->attn_output_a.rows_per_gpu != kAttentionOutputAFull / kGpus ||
        ops->attn.cols != kAttentionOutputAFull ||
        ops->attn.rows_per_gpu != kHidden / kGpus) {
        std::fprintf(stderr,
                     "tp_ep_true_attention_output_bad_shape\tlayer\t%d\t"
                     "out_a_cols\t%d\tout_a_rows_per_gpu\t%d\t"
                     "out_b_cols\t%d\tout_b_rows_per_gpu\t%d\n",
                     layer, ops->attn_output_a.cols,
                     ops->attn_output_a.rows_per_gpu, ops->attn.cols,
                     ops->attn.rows_per_gpu);
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    const int block = 256;
    const size_t out_a_shard_cols = (size_t)ops->attn_output_a.rows_per_gpu;
    const size_t out_a_shard_row_bytes = out_a_shard_cols * sizeof(float);
    const size_t out_a_full_row_bytes = (size_t)kAttentionOutputAFull * sizeof(float);
    const uint64_t head_input_elems =
        (uint64_t)opt.slots * (uint64_t)kAttentionOutputAInput;
    const uint64_t out_a_full_elems =
        (uint64_t)opt.slots * (uint64_t)kAttentionOutputAFull;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_heads || !r.d_attn_output_a_full ||
            !ops->attn_output_a.d_x_half[(size_t)rank] ||
            !ops->attn.d_x_half[(size_t)rank]) {
            return 3;
        }
        fill_dense_input_half_from_tensor_kernel<<<
            (unsigned int)((head_input_elems + block - 1) / block), block, 0,
            r.stream>>>(ops->attn_output_a.d_x_half[(size_t)rank],
                          r.d_attn_heads,
                          (uint32_t)kAttentionOutputAInput,
                          (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 4;

    if (launch_resident_f8_dense(opt, ops->attn_output_a, ranks) != 0) {
        return 5;
    }
    if (enqueue_rank_streams_wait_after_dense_streams(ranks) != 0) return 5;

    const bool use_nccl_allgather =
        opt.true_ds4_attention_output_nccl_allgather_gate;
    if (use_nccl_allgather) {
        for (int rank = 0; rank < kGpus; ++rank) {
            if (!ranks[rank].compose_nccl_initialized ||
                !ranks[rank].compose_nccl ||
                !ops->attn_output_a.d_out[(size_t)rank]) {
                return 6;
            }
        }
        CHECK_NCCL(ncclGroupStart());
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_NCCL(ncclAllGather(ops->attn_output_a.d_out[(size_t)rank],
                                     r.d_attn_output_a_full,
                                     (size_t)opt.slots * out_a_shard_cols,
                                     ncclFloat,
                                     ds4_comm_hc(r),
                                     r.stream));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            fill_dense_input_half_from_rank_major_shards_kernel<<<
                (unsigned int)((out_a_full_elems + block - 1) / block),
                block, 0, r.stream>>>(
                ops->attn.d_x_half[(size_t)rank], r.d_attn_output_a_full,
                (uint32_t)out_a_shard_cols, (uint32_t)kGpus,
                (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
    } else {
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &dr = ranks[dst];
            CHECK_CUDA(cudaSetDevice(dr.device));
            for (int src = 0; src < kGpus; ++src) {
                const float *src_shard = ops->attn_output_a.d_out[(size_t)src];
                if (!src_shard) return 6;
                CHECK_CUDA(cudaMemcpy2DAsync(
                    dr.d_attn_output_a_full + (size_t)src * out_a_shard_cols,
                    out_a_full_row_bytes, src_shard, out_a_shard_row_bytes,
                    out_a_shard_row_bytes, (size_t)opt.slots, cudaMemcpyDefault,
                    dr.stream));
            }
            fill_dense_input_half_from_tensor_kernel<<<
                (unsigned int)((out_a_full_elems + block - 1) / block), block, 0,
                dr.stream>>>(ops->attn.d_x_half[(size_t)dst],
                              dr.d_attn_output_a_full,
                              (uint32_t)kAttentionOutputAFull,
                              (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 6;

    if (launch_resident_f8_dense(opt, ops->attn, ranks) != 0) {
        return 7;
    }
    if (enqueue_rank_streams_wait_after_dense_streams(ranks) != 0) return 7;

    TensorF32Stats head_stats;
    TensorF32Stats out_a_stats;
    TensorF32Stats out_b_stats;
    if (!opt.true_ds4_semantic_skip_stats_gate) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            merge_tensor_stats(
                &head_stats,
                collect_tensor_f32_stats(r.d_attn_heads,
                                         (size_t)head_input_elems, r.stream));
            merge_tensor_stats(
                &out_a_stats,
                collect_tensor_f32_stats(r.d_attn_output_a_full,
                                         (size_t)out_a_full_elems, r.stream));
            merge_tensor_stats(
                &out_b_stats,
                collect_tensor_f32_stats(
                    ops->attn.d_out[(size_t)rank],
                    (size_t)opt.slots * (size_t)ops->attn.rows_per_gpu,
                    r.dense_stream ? r.dense_stream : r.stream));
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    std::printf("tp_ep_true_attention_output_projection\tlayer\t%d\tslots\t%d\t"
                "head_input_cols\t%d\tout_a_cols\t%d\tout_b_shard_cols\t%d\t"
                "nccl_allgather\t%d\t"
                "stats_skipped\t%d\t"
                "heads_max\t%.9g\theads_bad\t%d\t"
                "out_a_max\t%.9g\tout_a_bad\t%d\t"
                "out_b_max\t%.9g\tout_b_bad\t%d\tms\t%.6f\tPASS\n",
                layer, opt.slots, kAttentionOutputAInput, kAttentionOutputAFull,
                ops->attn.rows_per_gpu, use_nccl_allgather ? 1 : 0,
                opt.true_ds4_semantic_skip_stats_gate ? 1 : 0,
                head_stats.max_abs,
                head_stats.finite_bad, out_a_stats.max_abs,
                out_a_stats.finite_bad, out_b_stats.max_abs,
                out_b_stats.finite_bad, ms);
    return 0;
}
