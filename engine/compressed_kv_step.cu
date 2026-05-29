int run_true_ds4_compressed_reference_diff_gate(const Options &opt,
                                                SharedHcControls *hc,
                                                RankState ranks[kGpus],
                                                int layer,
                                                int ratio,
                                                int comp_width,
                                                uint32_t emitted,
                                                uint32_t comp_row,
                                                uint32_t visible_rows) {
    if (!opt.true_ds4_compressed_reference_diff_gate) return 0;
    if (!hc || !hc->initialized || layer < 0 || layer >= 43) return 1;
    if (ratio != 4 || !emitted) {
        std::printf("tp_ep_compressed_reference_diff\tlayer\t%d\tratio\t%d\t"
                    "emitted\t%u\tSKIP\n",
                    layer, ratio, emitted);
        return 0;
    }
    RankState &r0 = ranks[0];
    CHECK_CUDA(cudaSetDevice(r0.device));
    if (!hc->d_attn_comp_kv_full || !hc->d_attn_comp_score_full ||
        !hc->d_attn_compress_ape[layer] || !hc->d_attn_compress_norm[layer] ||
        !r0.d_attn_comp_kv_cur || !r0.d_attn_comp_score_cur ||
        !r0.d_attn_comp_rows || !r0.d_index_comp_rows ||
        !r0.d_indexer_scores || !r0.d_indexer_topk ||
        !hc->d_index_comp_kv_full || !hc->d_index_comp_score_full ||
        !hc->d_indexer_compress_ape[layer] ||
        !hc->d_indexer_compress_norm[layer] ||
        !hc->d_indexer_q_full || !hc->d_indexer_w_full) {
        return 2;
    }

    const int block = 256;
    const uint32_t state_rows =
        (uint32_t)attn_comp_state_rows_for_ratio(ratio);
    const uint32_t state_width =
        (uint32_t)attn_comp_state_width_for_ratio(ratio);
    const float comp_freq_scale = 1.0f / kRopeScaleFactor;
    const float comp_ext_factor = 1.0f;
    float comp_attn_factor = 1.0f;
    comp_attn_factor /= 1.0f + 0.1f * logf(1.0f / comp_freq_scale);
    const size_t attn_state_elems =
        (size_t)opt.slots * state_rows * (size_t)state_width;
    const size_t attn_row_elems = (size_t)opt.slots * kHeadDim;
    const size_t index_state_elems =
        (size_t)opt.slots * kIndexCompStateRows * (size_t)kIndexCompWidth;
    const size_t index_row_elems = (size_t)opt.slots * kIndexerHeadDim;

    float *d_attn_state_kv = nullptr;
    float *d_attn_state_score = nullptr;
    float *d_attn_row_ref = nullptr;
    float *d_attn_row_tp = nullptr;
    float *d_index_state_kv = nullptr;
    float *d_index_state_score = nullptr;
    float *d_index_row_ref = nullptr;
    float *d_index_row_tp = nullptr;
    float *d_index_score_ref = nullptr;
    float *d_index_score_ref_compact = nullptr;
    float *d_index_score_tp = nullptr;
    uint32_t *d_index_topk_ref = nullptr;

    CHECK_CUDA(cudaMalloc(&d_attn_state_kv, attn_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_state_score, attn_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_row_ref, attn_row_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_row_tp, attn_row_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_state_kv, index_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_state_score, index_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_row_ref, index_row_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_row_tp, index_row_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_score_ref,
                          (size_t)opt.slots * kIndexerTopK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_score_ref_compact,
                          (size_t)opt.slots * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_score_tp, (size_t)opt.slots * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_index_topk_ref,
                          (size_t)opt.slots * kIndexerTopK * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_attn_state_kv, 0, attn_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_attn_state_score, 0, attn_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_index_state_kv, 0, index_state_elems * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_index_state_score, 0, index_state_elems * sizeof(float)));

    log_tensor_f32_diff_summary("attn_comp_kv_current_peer_copy", layer,
                                r0.d_attn_comp_kv_cur,
                                hc->d_attn_comp_kv_full,
                                (size_t)opt.slots * (size_t)comp_width,
                                r0.stream);
    log_tensor_f32_diff_summary("attn_comp_score_current_peer_copy", layer,
                                r0.d_attn_comp_score_cur,
                                hc->d_attn_comp_score_full,
                                (size_t)opt.slots * (size_t)comp_width,
                                r0.stream);

    compressor_pool_emit_slots_kernel<<<
        dim3((unsigned int)((kHeadDim + block - 1) / block),
             (unsigned int)opt.slots, 1u),
        block>>>(d_attn_row_ref, r0.d_attn_comp_state_kv,
                 r0.d_attn_comp_state_score,
                 (uint32_t)opt.slots, (uint32_t)kHeadDim, (uint32_t)ratio,
                 0u, 1u, state_rows, state_width);
    compressor_norm_emit_slots_kernel<<<(unsigned int)opt.slots, 256>>>(
        d_attn_row_ref, hc->d_attn_compress_norm[layer], (uint32_t)opt.slots,
        (uint32_t)kHeadDim, 0u, 1u, 1.0e-6f);
    if (opt.true_ds4_attention_rope_gate) {
        rope_tail_comp_emit_slots_kernel<<<(unsigned int)opt.slots, 64>>>(
            d_attn_row_ref, (uint32_t)opt.slots, (uint32_t)kHeadDim,
            (uint32_t)kRotaryDim, 0u, 1u, r0.d_decode_position,
            1ll - (int64_t)ratio,
            kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
            comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
            kRopeYarnBetaSlow);
    }
    round_comp_emit_slots_kernel<<<
        (unsigned int)(((uint64_t)opt.slots * kHeadDim + block - 1) / block),
        block>>>(d_attn_row_ref, (uint32_t)opt.slots, (uint32_t)kHeadDim,
                 0u, 1u);
    pack_comp_row_kernel<<<
        (unsigned int)(((uint64_t)opt.slots * kHeadDim + block - 1) / block),
        block>>>(d_attn_row_tp, r0.d_attn_comp_rows, (uint32_t)opt.slots,
                 (uint32_t)kHeadDim, comp_row, (uint32_t)kBoundedCompRows);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    log_tensor_f32_diff_summary("attn_comp_row_compact_reference", layer,
                                d_attn_row_tp, d_attn_row_ref, attn_row_elems,
                                nullptr);

    compressor_pool_emit_slots_kernel<<<
        dim3((unsigned int)((kIndexerHeadDim + block - 1) / block),
             (unsigned int)opt.slots, 1u),
        block>>>(d_index_row_ref, r0.d_index_comp_state_kv,
                 r0.d_index_comp_state_score,
                 (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, 4u,
                 0u, 1u, (uint32_t)kIndexCompStateRows,
                 (uint32_t)kIndexCompWidth);
    compressor_norm_emit_slots_kernel<<<(unsigned int)opt.slots, 256>>>(
        d_index_row_ref, hc->d_indexer_compress_norm[layer],
        (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, 0u, 1u,
        1.0e-6f);
    if (opt.true_ds4_attention_rope_gate) {
        rope_tail_comp_emit_slots_kernel<<<(unsigned int)opt.slots, 64>>>(
            d_index_row_ref, (uint32_t)opt.slots,
            (uint32_t)kIndexerHeadDim, (uint32_t)kRotaryDim, 0u, 1u,
            r0.d_decode_position, -3ll,
            kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
            comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
            kRopeYarnBetaSlow);
    }
    round_comp_emit_slots_kernel<<<
        (unsigned int)(((uint64_t)opt.slots * kIndexerHeadDim + block - 1) /
                       block),
        block>>>(d_index_row_ref, (uint32_t)opt.slots,
                 (uint32_t)kIndexerHeadDim, 0u, 1u);
    pack_comp_row_kernel<<<
        (unsigned int)(((uint64_t)opt.slots * kIndexerHeadDim + block - 1) /
                       block),
        block>>>(d_index_row_tp, r0.d_index_comp_rows, (uint32_t)opt.slots,
                 (uint32_t)kIndexerHeadDim, comp_row,
                 (uint32_t)kBoundedCompRows);
    indexer_score_bounded_rows_slots_kernel<<<(unsigned int)opt.slots, 256>>>(
        d_index_score_ref, d_index_topk_ref, hc->d_indexer_q_full,
        hc->d_indexer_w_full, d_index_row_ref, (uint32_t)opt.slots,
        1u, 1u, (uint32_t)kIndexerTopK,
        1.0f / sqrtf((float)(kIndexerHead * kIndexerHeadDim)));
    pack_indexer_score_column_kernel<<<
        (unsigned int)((opt.slots + block - 1) / block), block>>>(
        d_index_score_tp, r0.d_indexer_scores, (uint32_t)opt.slots,
        (uint32_t)kIndexerTopK, comp_row);
    pack_indexer_score_column_kernel<<<
        (unsigned int)((opt.slots + block - 1) / block), block>>>(
        d_index_score_ref_compact, d_index_score_ref, (uint32_t)opt.slots,
        (uint32_t)kIndexerTopK, 0u);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    log_tensor_f32_diff_summary("index_comp_row_compact_reference", layer,
                                d_index_row_tp, d_index_row_ref,
                                index_row_elems, nullptr);
    log_tensor_f32_diff_summary("indexer_score_row_compact_reference", layer,
                                d_index_score_tp, d_index_score_ref_compact,
                                (size_t)opt.slots, nullptr);
    std::printf("tp_ep_compressed_reference_diff_summary\tlayer\t%d\t"
                "ratio\t%d\temitted\t%u\tcomp_row\t%u\t"
                "visible_compressed_rows\t%u\tPASS\n",
                layer, ratio, emitted, comp_row, visible_rows);

    CHECK_CUDA(cudaFree(d_index_topk_ref));
    CHECK_CUDA(cudaFree(d_index_score_tp));
    CHECK_CUDA(cudaFree(d_index_score_ref_compact));
    CHECK_CUDA(cudaFree(d_index_score_ref));
    CHECK_CUDA(cudaFree(d_index_row_tp));
    CHECK_CUDA(cudaFree(d_index_row_ref));
    CHECK_CUDA(cudaFree(d_index_state_score));
    CHECK_CUDA(cudaFree(d_index_state_kv));
    CHECK_CUDA(cudaFree(d_attn_row_tp));
    CHECK_CUDA(cudaFree(d_attn_row_ref));
    CHECK_CUDA(cudaFree(d_attn_state_score));
    CHECK_CUDA(cudaFree(d_attn_state_kv));
    return 0;
}

int run_true_ds4_compressed_kv_projection_gate(const Options &opt,
                                               SharedHcControls *hc,
                                               const LayerDenseOps *ops,
                                               RankState ranks[kGpus],
                                               ds4_tp_runtime *rt,
                                               int layer) {
    if (!opt.true_ds4_compressed_kv_gate) return 0;
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 43 || !hc->d_attn_normed || !hc->d_q_a_normed) {
        return 1;
    }
    const int ratio = ds4_layer_ratio(layer);
    const uint32_t emitted =
        ratio != 0 && (((opt.position + 1ull) % (uint64_t)ratio) == 0ull) ? 1u : 0u;
    const uint32_t indexer_topk =
        opt.true_ds4_indexer_attention_gate && ratio == 4 ? kIndexerTopK : 0u;
    if (ratio == 0) {
        std::printf("tp_ep_compressed_kv_projection\tlayer\t%d\tslots\t%d\t"
                    "ratio\t0\temitted_compressed_rows\t0\t"
                    "visible_compressed_rows\t0\tindexer_topk_count\t0\t"
                    "attn_input_fill_ms\t0.000000\tattn_dense_ms\t0.000000\t"
                    "attn_gather_ms\t0.000000\tattn_state_emit_ms\t0.000000\t"
                    "attn_typed_ms\t0.000000\tindexer_input_fill_ms\t0.000000\t"
                    "indexer_dense_ms\t0.000000\tindexer_gather_rope_ms\t0.000000\t"
                    "indexer_state_emit_ms\t0.000000\tindexer_typed_score_ms\t0.000000\t"
                    "reference_diff_ms\t0.000000\tratio_shift_ms\t0.000000\t"
                    "ms\t0.000000\tPASS\n",
                    layer, opt.slots);
        return 0;
    }

    const int comp_width = ratio == 4 ? 2 * kHeadDim : kHeadDim;
    const int comp_state_rows = attn_comp_state_rows_for_ratio(ratio);
    const int comp_state_width = attn_comp_state_width_for_ratio(ratio);
    if (ops->attn_compress_kv.cols != kHidden ||
        ops->attn_compress_gate.cols != kHidden ||
        ops->attn_compress_kv.rows_per_gpu != comp_width / kGpus ||
        ops->attn_compress_gate.rows_per_gpu != comp_width / kGpus) {
        std::fprintf(stderr,
                     "tp_ep_compressed_kv_bad_shape\tlayer\t%d\t"
                     "ratio\t%d\tkv_cols\t%d\tkv_rows_per_gpu\t%d\t"
                     "gate_cols\t%d\tgate_rows_per_gpu\t%d\n",
                     layer, ratio, ops->attn_compress_kv.cols,
                     ops->attn_compress_kv.rows_per_gpu,
                     ops->attn_compress_gate.cols,
                     ops->attn_compress_gate.rows_per_gpu);
        return 2;
    }

    const auto start = std::chrono::steady_clock::now();
    auto t_stage = start;
    auto elapsed_ms = [](std::chrono::steady_clock::time_point a,
                         std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    double attn_input_fill_ms = 0.0;
    double attn_dense_ms = 0.0;
    double attn_gather_ms = 0.0;
    double attn_state_emit_ms = 0.0;
    double attn_typed_ms = 0.0;
    double indexer_input_fill_ms = 0.0;
    double indexer_dense_ms = 0.0;
    double indexer_gather_rope_ms = 0.0;
    double indexer_state_emit_ms = 0.0;
    double indexer_typed_score_ms = 0.0;
    double reference_diff_ms = 0.0;
    double ratio_shift_ms = 0.0;
    const int block = 256;
    const uint64_t hidden_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const bool direct_current_input_fill =
        opt.true_ds4_compressed_kv_direct_input_fill_gate;
    const bool dense_event_wait =
        opt.true_ds4_compressed_kv_dense_event_wait_gate;
    const bool skip_dense_stats =
        opt.true_ds4_compressed_kv_skip_dense_stats_gate;
    const bool fused_attn_current_fill =
        opt.true_ds4_compressed_kv_fused_attn_input_fill_gate;
    const bool fused_ratio4_current_fill =
        opt.true_ds4_compressed_kv_fused_input_fill_gate &&
        opt.true_ds4_indexer_attention_gate && ratio == 4;
    const bool fused_rope_round =
        opt.true_ds4_compressed_kv_fused_rope_round_gate &&
        opt.true_ds4_attention_rope_gate && emitted;
    const bool fused_pool_norm =
        opt.true_ds4_compressed_kv_fused_pool_norm_gate && emitted;
    const bool fused_pool_norm_rope_round =
        opt.true_ds4_compressed_kv_fused_pool_norm_rope_round_gate &&
        opt.true_ds4_attention_rope_gate && emitted;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    cudaStream_t control_stream = graph_event_order ? ranks[0].stream : (cudaStream_t)0;
    if (!direct_current_input_fill) {
        const int bcast_rc = nccl_broadcast_f32_from_device0_to_current_full(
            opt, ranks, hc->d_attn_normed, hidden_elems,
            "compressed_kv_normed_current");
        if (bcast_rc != 0) return 23;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_current_full ||
            !ops->attn_compress_kv.d_x_half[(size_t)rank] ||
            !ops->attn_compress_gate.d_x_half[(size_t)rank] ||
            (fused_ratio4_current_fill &&
             (!ops->indexer_proj.d_x_half[(size_t)rank] ||
              !ops->indexer_compress_kv.d_x_half[(size_t)rank] ||
              !ops->indexer_compress_gate.d_x_half[(size_t)rank]))) {
            return 3;
        }
        const float *current_src = hc->d_attn_normed;
        if (!direct_current_input_fill) {
            current_src = r.d_current_full;
        }
        if (fused_ratio4_current_fill) {
            fill_ratio4_compressed_indexer_inputs_half_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(
                ops->attn_compress_kv.d_x_half[(size_t)rank],
                ops->attn_compress_gate.d_x_half[(size_t)rank],
                ops->indexer_proj.d_x_half[(size_t)rank],
                ops->indexer_compress_kv.d_x_half[(size_t)rank],
                ops->indexer_compress_gate.d_x_half[(size_t)rank],
                current_src, (uint32_t)opt.slots);
        } else if (fused_attn_current_fill) {
            fill_attn_compressed_inputs_half_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(
                ops->attn_compress_kv.d_x_half[(size_t)rank],
                ops->attn_compress_gate.d_x_half[(size_t)rank],
                current_src, (uint32_t)opt.slots);
        } else {
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->attn_compress_kv.d_x_half[(size_t)rank],
                             current_src, (uint32_t)kHidden,
                             (uint32_t)opt.slots);
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->attn_compress_gate.d_x_half[(size_t)rank],
                             current_src, (uint32_t)kHidden,
                             (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    if (dense_event_wait || graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 22;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_input_fill_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }
    if (launch_resident_f8_dense(opt, ops->attn_compress_kv, ranks) != 0 ||
        launch_resident_f8_dense(opt, ops->attn_compress_gate, ranks) != 0) {
        return 4;
    }

    TensorF32Stats attn_kv_stats;
    TensorF32Stats attn_gate_stats;
    if (graph_event_order) {
        if (enqueue_control_wait_after_dense_streams(
                opt, ranks, control_stream) != 0) return 24;
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            cudaStream_t stream = ranks[rank].dense_stream ? ranks[rank].dense_stream
                                                           : ranks[rank].stream;
            CHECK_CUDA(cudaStreamSynchronize(stream));
            if (!skip_dense_stats) {
            const size_t comp_elems =
                (size_t)opt.slots * (size_t)ops->attn_compress_kv.rows_per_gpu;
                merge_tensor_stats(&attn_kv_stats,
                                   collect_tensor_f32_stats(
                                       ops->attn_compress_kv.d_out[(size_t)rank],
                                       comp_elems, stream));
                merge_tensor_stats(&attn_gate_stats,
                                   collect_tensor_f32_stats(
                                       ops->attn_compress_gate.d_out[(size_t)rank],
                                       comp_elems, stream));
            }
        }
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_dense_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }

    if (!hc->d_attn_comp_kv_full || !hc->d_attn_comp_score_full ||
        !hc->d_attn_compress_ape[layer] || !hc->d_attn_compress_norm[layer]) {
        return 9;
    }
    uint32_t emitted_comp_row = 0u;
    uint32_t visible = 0u;
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        gather_dense_shard_to_full_kernel<<<
            (unsigned int)(((uint64_t)opt.slots *
                                (uint64_t)ops->attn_compress_kv.rows_per_gpu +
                            block - 1) /
                           block),
            block, 0, control_stream>>>(
            hc->d_attn_comp_kv_full,
            ops->attn_compress_kv.d_out[(size_t)rank], rank,
            (uint32_t)ops->attn_compress_kv.rows_per_gpu,
            (uint32_t)comp_width, (uint32_t)opt.slots);
        gather_dense_shard_to_full_kernel<<<
            (unsigned int)(((uint64_t)opt.slots *
                                (uint64_t)ops->attn_compress_gate.rows_per_gpu +
                            block - 1) /
                           block),
            block, 0, control_stream>>>(
            hc->d_attn_comp_score_full,
            ops->attn_compress_gate.d_out[(size_t)rank], rank,
            (uint32_t)ops->attn_compress_gate.rows_per_gpu,
            (uint32_t)comp_width, (uint32_t)opt.slots);
    }
    CHECK_CUDA(cudaGetLastError());
    if (graph_event_order) {
        if (enqueue_rank_streams_wait_after_control(
                opt, ranks, control_stream) != 0) return 25;
    } else {
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_gather_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }

    const float comp_freq_scale = 1.0f / kRopeScaleFactor;
    const float comp_ext_factor = 1.0f;
    float comp_attn_factor = 1.0f;
    comp_attn_factor /= 1.0f + 0.1f * logf(1.0f / comp_freq_scale);
    const size_t comp_bytes = (size_t)opt.slots * comp_width * sizeof(float);
    const uint64_t comp_elems = (uint64_t)opt.slots * (uint64_t)comp_width;
    if (!graph_event_order) {
        void *kv_dsts[kGpus] = {};
        void *score_dsts[kGpus] = {};
        for (int rank = 0; rank < kGpus; ++rank) {
            kv_dsts[rank] = ranks[rank].d_attn_comp_kv_cur;
            score_dsts[rank] = ranks[rank].d_attn_comp_score_cur;
        }
        if (nccl_broadcast_bytes_from_rank0(
                ranks, hc->d_attn_comp_kv_full, kv_dsts, comp_bytes,
                "attn_comp_kv_cur") != 0 ||
            nccl_broadcast_bytes_from_rank0(
                ranks, hc->d_attn_comp_score_full, score_dsts, comp_bytes,
                "attn_comp_score_cur") != 0) {
            return 28;
        }
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_comp_kv_cur || !r.d_attn_comp_score_cur ||
            !r.d_attn_comp_state_kv || !r.d_attn_comp_state_score ||
            !r.d_attn_comp_rows) {
            return 10;
        }
        if (graph_event_order) {
            enqueue_graph_f32_copy_from_device0(
                opt, r, rank, r.d_attn_comp_kv_cur,
                hc->d_attn_comp_kv_full, comp_elems, r.stream, block);
            enqueue_graph_f32_copy_from_device0(
                opt, r, rank, r.d_attn_comp_score_cur,
                hc->d_attn_comp_score_full, comp_elems, r.stream, block);
        }
        compressor_store_slots_kernel<<<
            (unsigned int)(((uint64_t)opt.slots * (uint64_t)comp_width +
                            block - 1) /
                           block),
            block, 0, r.stream>>>(
            r.d_attn_comp_kv_cur, r.d_attn_comp_score_cur,
            r.d_attn_comp_state_kv, r.d_attn_comp_state_score,
            hc->d_attn_compress_ape[layer], (uint32_t)opt.slots,
            (uint32_t)kHeadDim, (uint32_t)ratio, r.d_decode_position,
            (uint32_t)comp_state_rows, (uint32_t)comp_state_width);
        if (emitted) {
            const uint32_t comp_row =
                r.attn_comp_rows_written_layers[layer] %
                (uint32_t)kBoundedCompRows;
            if (rank == 0) emitted_comp_row = comp_row;
            r.attn_comp_row_position_layers[layer][comp_row] = opt.position;
            r.attn_comp_row_loaded_layers[layer][comp_row] = false;
            if (fused_pool_norm_rope_round) {
                compressor_pool_norm_rope_round_emit_slots_kernel<<<
                    (unsigned int)opt.slots, 256, 0, r.stream>>>(
                    r.d_attn_comp_rows, r.d_attn_comp_state_kv,
                    r.d_attn_comp_state_score, hc->d_attn_compress_norm[layer],
                    (uint32_t)opt.slots, (uint32_t)kHeadDim, (uint32_t)ratio,
                    comp_row, (uint32_t)kBoundedCompRows,
                    (uint32_t)comp_state_rows, (uint32_t)comp_state_width,
                    1.0e-6f, (uint32_t)kRotaryDim, r.d_decode_position,
                    1ll - (int64_t)ratio,
                    kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                    comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                    kRopeYarnBetaSlow);
            } else if (fused_pool_norm) {
                compressor_pool_norm_emit_slots_kernel<<<
                    (unsigned int)opt.slots, 256, 0, r.stream>>>(
                    r.d_attn_comp_rows, r.d_attn_comp_state_kv,
                    r.d_attn_comp_state_score, hc->d_attn_compress_norm[layer],
                    (uint32_t)opt.slots, (uint32_t)kHeadDim, (uint32_t)ratio,
                    comp_row, (uint32_t)kBoundedCompRows,
                    (uint32_t)comp_state_rows, (uint32_t)comp_state_width,
                    1.0e-6f);
            } else {
                compressor_pool_emit_slots_kernel<<<
                    dim3((unsigned int)((kHeadDim + block - 1) / block),
                         (unsigned int)opt.slots, 1u),
                    block, 0, r.stream>>>(
                    r.d_attn_comp_rows, r.d_attn_comp_state_kv,
                    r.d_attn_comp_state_score, (uint32_t)opt.slots,
                    (uint32_t)kHeadDim, (uint32_t)ratio, comp_row,
                    (uint32_t)kBoundedCompRows, (uint32_t)comp_state_rows,
                    (uint32_t)comp_state_width);
                compressor_norm_emit_slots_kernel<<<(unsigned int)opt.slots, 256,
                                                    0, r.stream>>>(
                    r.d_attn_comp_rows, hc->d_attn_compress_norm[layer],
                    (uint32_t)opt.slots, (uint32_t)kHeadDim, comp_row,
                    (uint32_t)kBoundedCompRows, 1.0e-6f);
            }
            if (fused_pool_norm_rope_round) {
                // RoPE and F16 rounding were already applied by the fused emit.
            } else if (fused_rope_round) {
                rope_tail_round_comp_emit_slots_kernel<<<
                    (unsigned int)opt.slots, 256, 0, r.stream>>>(
                    r.d_attn_comp_rows, (uint32_t)opt.slots,
                    (uint32_t)kHeadDim, (uint32_t)kRotaryDim, comp_row,
                    (uint32_t)kBoundedCompRows, r.d_decode_position,
                    1ll - (int64_t)ratio,
                    kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                    comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                    kRopeYarnBetaSlow);
            } else {
                if (opt.true_ds4_attention_rope_gate) {
                    rope_tail_comp_emit_slots_kernel<<<
                        (unsigned int)opt.slots, 64, 0, r.stream>>>(
                        r.d_attn_comp_rows, (uint32_t)opt.slots,
                        (uint32_t)kHeadDim, (uint32_t)kRotaryDim, comp_row,
                        (uint32_t)kBoundedCompRows, r.d_decode_position,
                        1ll - (int64_t)ratio,
                        kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                        comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                        kRopeYarnBetaSlow);
                }
                round_comp_emit_slots_kernel<<<
                    (unsigned int)(((uint64_t)opt.slots * kHeadDim + block - 1) /
                                   block),
                    block, 0, r.stream>>>(
                    r.d_attn_comp_rows, (uint32_t)opt.slots, (uint32_t)kHeadDim,
                    comp_row, (uint32_t)kBoundedCompRows);
            }
            r.attn_comp_rows_written_layers[layer]++;
        }
        visible = std::max(
            visible,
            std::min(r.attn_comp_rows_written_layers[layer],
                     (uint32_t)kBoundedCompRows));
        CHECK_CUDA(cudaGetLastError());
    }
    if (!graph_event_order) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_state_emit_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }
    if (opt.true_ds4_attention_typed_kv_compressed_gate && emitted) {
        if (!rt) {
            std::fprintf(stderr,
                         "tp_ep_true_attention_typed_kv_compressed_failed\t"
                         "layer\t%d\treason\tmissing_tp_runtime\n",
                         layer);
            return 14;
        }
        char err[512] = {0};
        ds4_tp_kv_row_view view;
        if (ds4_tp_runtime_kv_row_view(
                rt, layer, 0, opt.position, DS4_V100_TP_KV_ROW_ATTN, &view, err,
                sizeof(err)) != 0) {
            std::fprintf(stderr,
                         "tp_ep_true_attention_typed_kv_compressed_view_failed\t"
                         "layer\t%d\t%s\n",
                         layer, err);
            return 15;
        }
        int current_store = 0;
        if (!opt.true_ds4_attention_typed_kv_skip_compressed_store_gate) {
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                const void *src[kGpus] = {};
                void *streams[kGpus] = {};
                const size_t row_offset =
                    (size_t)emitted_comp_row * (size_t)kHeadDim;
                for (int rank = 0; rank < kGpus; ++rank) {
                    src[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                    streams[rank] = opt.decode_cudagraph_gate
                        ? (void *)ranks[rank].stream
                        : nullptr;
                }
                const int store_rc = opt.decode_cudagraph_gate
                    ? ds4_tp_runtime_kv_rows_store_f32_device_streams(
                          rt, layer, 0, (uint32_t)opt.slots, opt.position,
                          DS4_V100_TP_KV_ROW_ATTN, src,
                          (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                          streams, err, sizeof(err))
                    : ds4_tp_runtime_kv_rows_store_f32_device(
                          rt, layer, 0, (uint32_t)opt.slots, opt.position,
                          DS4_V100_TP_KV_ROW_ATTN, src,
                          (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                          err, sizeof(err));
                if (store_rc != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_compressed_store_failed\t"
                                 "layer\t%d\tmode\tbatched\t%s\n",
                                 layer, err);
                    return 16;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    const void *src[kGpus] = {};
                    const size_t row_offset =
                        ((size_t)slot * (size_t)kBoundedCompRows +
                         (size_t)emitted_comp_row) *
                        (size_t)kHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        src[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                    }
                    if (ds4_tp_runtime_kv_row_store_f32_device(
                            rt, layer, slot, opt.position, DS4_V100_TP_KV_ROW_ATTN,
                            src, err, sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_compressed_store_failed\t"
                                     "layer\t%d\tslot\t%u\t%s\n",
                                     layer, slot, err);
                        return 16;
                    }
                }
            }
            current_store = 1;
        }
        sync_typed_kv_boundary(opt, ranks);
        int current_load = 0;
        if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
            current_store) {
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                void *dst[kGpus] = {};
                void *streams[kGpus] = {};
                const size_t row_offset =
                    (size_t)emitted_comp_row * (size_t)kHeadDim;
                for (int rank = 0; rank < kGpus; ++rank) {
                    dst[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                    streams[rank] = opt.decode_cudagraph_gate
                        ? (void *)ranks[rank].stream
                        : nullptr;
                }
                const int load_rc = opt.decode_cudagraph_gate
                    ? ds4_tp_runtime_kv_rows_load_f32_device_streams(
                          rt, layer, 0, (uint32_t)opt.slots, opt.position,
                          DS4_V100_TP_KV_ROW_ATTN, dst,
                          (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                          streams, err, sizeof(err))
                    : ds4_tp_runtime_kv_rows_load_f32_device(
                          rt, layer, 0, (uint32_t)opt.slots, opt.position,
                          DS4_V100_TP_KV_ROW_ATTN, dst,
                          (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                          err, sizeof(err));
                if (load_rc != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_compressed_load_failed\t"
                                 "layer\t%d\tmode\tbatched\t%s\n",
                                 layer, err);
                    return 17;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    void *dst[kGpus] = {};
                    const size_t row_offset =
                        ((size_t)slot * (size_t)kBoundedCompRows +
                         (size_t)emitted_comp_row) *
                        (size_t)kHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        dst[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                    }
                    if (ds4_tp_runtime_kv_row_load_f32_device(
                            rt, layer, slot, opt.position, DS4_V100_TP_KV_ROW_ATTN,
                            dst, err, sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_compressed_load_failed\t"
                                     "layer\t%d\tslot\t%u\t%s\n",
                                     layer, slot, err);
                        return 17;
                    }
                }
            }
            current_load = 1;
        }
        sync_typed_kv_boundary(opt, ranks);
        if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate ||
            !current_store) {
            for (int rank = 0; rank < kGpus; ++rank) {
                ranks[rank].attn_comp_row_loaded_layers[layer][emitted_comp_row] = true;
                ranks[rank].attn_comp_row_loaded_position_layers[layer][emitted_comp_row] =
                    opt.position;
            }
        }
        if (!opt.true_ds4_attention_typed_kv_quiet_gate) {
            std::printf("tp_ep_true_attention_typed_kv_compressed\tlayer\t%d\t"
                        "slots\t%d\tratio\t%d\tposition\t%llu\t"
                        "bounded_row\t%u\tphysical_row\t%llu\tlogical_cols\t%u\t"
                        "logical_row_bytes\t%llu\trow_bytes_per_gpu\t%llu\t"
                        "current_store\t%d\tcurrent_load\t%d\tPASS\n",
                        layer, opt.slots, ratio, (unsigned long long)opt.position,
                        emitted_comp_row, (unsigned long long)view.physical_row,
                        view.logical_cols, (unsigned long long)view.logical_row_bytes,
                        (unsigned long long)view.row_bytes[0], current_store,
                        current_load);
        }
    }
    {
        const auto now = std::chrono::steady_clock::now();
        attn_typed_ms = elapsed_ms(t_stage, now);
        t_stage = now;
    }

    TensorF32Stats index_q_stats;
    TensorF32Stats index_w_stats;
    TensorF32Stats index_kv_stats;
    TensorF32Stats index_gate_stats;
    if (opt.true_ds4_indexer_attention_gate && ratio == 4) {
        t_stage = std::chrono::steady_clock::now();
        if (ops->indexer_attn_q_b.cols != 1024 ||
            ops->indexer_attn_q_b.rows_per_gpu != (kIndexerHead * kIndexerHeadDim) / kGpus ||
            ops->indexer_proj.cols != kHidden ||
            ops->indexer_proj.rows_per_gpu != kIndexerHead / kGpus ||
            ops->indexer_compress_kv.cols != kHidden ||
            ops->indexer_compress_kv.rows_per_gpu != (2 * kIndexerHeadDim) / kGpus ||
            ops->indexer_compress_gate.cols != kHidden ||
            ops->indexer_compress_gate.rows_per_gpu != (2 * kIndexerHeadDim) / kGpus) {
            return 5;
        }
        const uint64_t q_a_elems = (uint64_t)opt.slots * 1024ull;
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            const float *current_src =
                direct_current_input_fill ? hc->d_attn_normed : r.d_current_full;
            if (!ops->indexer_attn_q_b.d_x_half[(size_t)rank] ||
                !ops->indexer_proj.d_x_half[(size_t)rank] ||
                !ops->indexer_compress_kv.d_x_half[(size_t)rank] ||
                !ops->indexer_compress_gate.d_x_half[(size_t)rank]) {
                return 6;
            }
            fill_dense_input_half_from_tensor_kernel<<<
                (unsigned int)((q_a_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->indexer_attn_q_b.d_x_half[(size_t)rank],
                             hc->d_q_a_normed, 1024u, (uint32_t)opt.slots);
            if (!fused_ratio4_current_fill) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->indexer_proj.d_x_half[(size_t)rank],
                                 current_src, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->indexer_compress_kv.d_x_half[(size_t)rank],
                                 current_src, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->indexer_compress_gate.d_x_half[(size_t)rank],
                                 current_src, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
            }
            CHECK_CUDA(cudaGetLastError());
        }
        if (dense_event_wait || graph_event_order) {
            if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 23;
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_input_fill_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
        if (launch_resident_f8_dense(opt, ops->indexer_attn_q_b, ranks) != 0 ||
            launch_resident_f8_dense(opt, ops->indexer_proj, ranks) != 0 ||
            launch_resident_f8_dense(opt, ops->indexer_compress_kv, ranks) != 0 ||
            launch_resident_f8_dense(opt, ops->indexer_compress_gate, ranks) != 0) {
            return 7;
        }
        if (graph_event_order) {
            if (enqueue_control_wait_after_dense_streams(
                    opt, ranks, control_stream) != 0) return 26;
        } else {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                cudaStream_t stream = ranks[rank].dense_stream ? ranks[rank].dense_stream
                                                               : ranks[rank].stream;
                CHECK_CUDA(cudaStreamSynchronize(stream));
                if (!skip_dense_stats) {
                merge_tensor_stats(&index_q_stats,
                                   collect_tensor_f32_stats(
                                       ops->indexer_attn_q_b.d_out[(size_t)rank],
                                       (size_t)opt.slots *
                                           (size_t)ops->indexer_attn_q_b.rows_per_gpu,
                                       stream));
                merge_tensor_stats(&index_w_stats,
                                   collect_tensor_f32_stats(
                                       ops->indexer_proj.d_out[(size_t)rank],
                                       (size_t)opt.slots *
                                           (size_t)ops->indexer_proj.rows_per_gpu,
                                       stream));
                merge_tensor_stats(&index_kv_stats,
                                   collect_tensor_f32_stats(
                                       ops->indexer_compress_kv.d_out[(size_t)rank],
                                       (size_t)opt.slots *
                                           (size_t)ops->indexer_compress_kv.rows_per_gpu,
                                       stream));
                merge_tensor_stats(&index_gate_stats,
                                   collect_tensor_f32_stats(
                                       ops->indexer_compress_gate.d_out[(size_t)rank],
                                       (size_t)opt.slots *
                                           (size_t)ops->indexer_compress_gate.rows_per_gpu,
                                       stream));
                }
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_dense_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
        if (!hc->d_indexer_q_full || !hc->d_indexer_w_full) return 13;
        if (!hc->d_index_comp_kv_full || !hc->d_index_comp_score_full ||
            !hc->d_indexer_compress_ape[layer] ||
            !hc->d_indexer_compress_norm[layer]) {
            return 11;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        for (int rank = 0; rank < kGpus; ++rank) {
            gather_dense_shard_to_full_kernel<<<
                (unsigned int)(((uint64_t)opt.slots *
                                    (uint64_t)ops->indexer_attn_q_b.rows_per_gpu +
                                block - 1) /
                               block),
                block, 0, control_stream>>>(
                hc->d_indexer_q_full,
                ops->indexer_attn_q_b.d_out[(size_t)rank], rank,
                (uint32_t)ops->indexer_attn_q_b.rows_per_gpu,
                (uint32_t)(kIndexerHead * kIndexerHeadDim),
                (uint32_t)opt.slots);
            gather_dense_shard_to_full_kernel<<<
                (unsigned int)(((uint64_t)opt.slots *
                                    (uint64_t)ops->indexer_proj.rows_per_gpu +
                                block - 1) /
                               block),
                block, 0, control_stream>>>(
                hc->d_indexer_w_full,
                ops->indexer_proj.d_out[(size_t)rank], rank,
                (uint32_t)ops->indexer_proj.rows_per_gpu,
                (uint32_t)kIndexerHead, (uint32_t)opt.slots);
            gather_dense_shard_to_full_kernel<<<
                (unsigned int)(((uint64_t)opt.slots *
                                    (uint64_t)ops->indexer_compress_kv.rows_per_gpu +
                                block - 1) /
                               block),
                block, 0, control_stream>>>(
                hc->d_index_comp_kv_full,
                ops->indexer_compress_kv.d_out[(size_t)rank], rank,
                (uint32_t)ops->indexer_compress_kv.rows_per_gpu,
                (uint32_t)kIndexCompWidth, (uint32_t)opt.slots);
            gather_dense_shard_to_full_kernel<<<
                (unsigned int)(((uint64_t)opt.slots *
                                    (uint64_t)ops->indexer_compress_gate.rows_per_gpu +
                                block - 1) /
                               block),
                block, 0, control_stream>>>(
                hc->d_index_comp_score_full,
                ops->indexer_compress_gate.d_out[(size_t)rank], rank,
                (uint32_t)ops->indexer_compress_gate.rows_per_gpu,
                (uint32_t)kIndexCompWidth, (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        if (!graph_event_order) {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        if (opt.true_ds4_attention_rope_gate) {
            rope_tail_rows_kernel<<<
                (unsigned int)(opt.slots * kIndexerHead), 64, 0,
                control_stream>>>(
                hc->d_indexer_q_full, (uint32_t)(opt.slots * kIndexerHead),
                (uint32_t)kIndexerHeadDim, (uint32_t)kRotaryDim,
                ranks[0].d_decode_position, 0ll, kRopeOrigCtx, 0,
                kCompressRopeFreqBase, comp_freq_scale, comp_ext_factor, comp_attn_factor,
                kRopeYarnBetaFast, kRopeYarnBetaSlow);
            CHECK_CUDA(cudaGetLastError());
            if (!graph_event_order) {
                CHECK_CUDA(cudaDeviceSynchronize());
            }
        }
        if (graph_event_order) {
            if (enqueue_rank_streams_wait_after_control(
                    opt, ranks, control_stream) != 0) return 27;
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_gather_rope_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
        const size_t index_bytes =
            (size_t)opt.slots * kIndexCompWidth * sizeof(float);
        const uint64_t index_elems =
            (uint64_t)opt.slots * (uint64_t)kIndexCompWidth;
        if (!graph_event_order) {
            void *kv_dsts[kGpus] = {};
            void *score_dsts[kGpus] = {};
            for (int rank = 0; rank < kGpus; ++rank) {
                kv_dsts[rank] = ranks[rank].d_index_comp_kv_cur;
                score_dsts[rank] = ranks[rank].d_index_comp_score_cur;
            }
            if (nccl_broadcast_bytes_from_rank0(
                    ranks, hc->d_index_comp_kv_full, kv_dsts, index_bytes,
                    "index_comp_kv_cur") != 0 ||
                nccl_broadcast_bytes_from_rank0(
                    ranks, hc->d_index_comp_score_full, score_dsts,
                    index_bytes, "index_comp_score_cur") != 0) {
                return 28;
            }
        }
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            if (!r.d_index_comp_kv_cur || !r.d_index_comp_score_cur ||
                !r.d_index_comp_state_kv || !r.d_index_comp_state_score ||
                !r.d_index_comp_rows || !r.d_indexer_scores ||
                !r.d_indexer_topk) {
                return 12;
            }
            if (graph_event_order) {
                enqueue_graph_f32_copy_from_device0(
                    opt, r, rank, r.d_index_comp_kv_cur,
                    hc->d_index_comp_kv_full, index_elems, r.stream, block);
                enqueue_graph_f32_copy_from_device0(
                    opt, r, rank, r.d_index_comp_score_cur,
                    hc->d_index_comp_score_full, index_elems, r.stream, block);
            }
            compressor_store_slots_kernel<<<
                (unsigned int)(((uint64_t)opt.slots * kIndexCompWidth +
                                block - 1) /
                               block),
                block, 0, r.stream>>>(
                r.d_index_comp_kv_cur, r.d_index_comp_score_cur,
                r.d_index_comp_state_kv, r.d_index_comp_state_score,
                hc->d_indexer_compress_ape[layer], (uint32_t)opt.slots,
                (uint32_t)kIndexerHeadDim, 4u, r.d_decode_position,
                (uint32_t)kIndexCompStateRows, (uint32_t)kIndexCompWidth);
            if (emitted) {
                const uint32_t comp_row =
                    r.index_comp_rows_written_layers[layer] %
                    (uint32_t)kBoundedCompRows;
                r.index_comp_row_position_layers[layer][comp_row] = opt.position;
                r.index_comp_row_loaded_layers[layer][comp_row] = false;
                const uint32_t visible_after =
                    std::min(r.index_comp_rows_written_layers[layer] + 1u,
                             (uint32_t)kBoundedCompRows);
                if (fused_pool_norm_rope_round) {
                    compressor_pool_norm_rope_round_emit_slots_kernel<<<
                        (unsigned int)opt.slots, 256, 0, r.stream>>>(
                        r.d_index_comp_rows, r.d_index_comp_state_kv,
                        r.d_index_comp_state_score,
                        hc->d_indexer_compress_norm[layer],
                        (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, 4u,
                        comp_row, (uint32_t)kBoundedCompRows,
                        (uint32_t)kIndexCompStateRows,
                        (uint32_t)kIndexCompWidth, 1.0e-6f,
                        (uint32_t)kRotaryDim, r.d_decode_position, -3ll,
                        kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                        comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                        kRopeYarnBetaSlow);
                } else if (fused_pool_norm) {
                    compressor_pool_norm_emit_slots_kernel<<<
                        (unsigned int)opt.slots, 256, 0, r.stream>>>(
                        r.d_index_comp_rows, r.d_index_comp_state_kv,
                        r.d_index_comp_state_score,
                        hc->d_indexer_compress_norm[layer],
                        (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, 4u,
                        comp_row, (uint32_t)kBoundedCompRows,
                        (uint32_t)kIndexCompStateRows,
                        (uint32_t)kIndexCompWidth, 1.0e-6f);
                } else {
                    compressor_pool_emit_slots_kernel<<<
                        dim3((unsigned int)((kIndexerHeadDim + block - 1) / block),
                             (unsigned int)opt.slots, 1u),
                        block, 0, r.stream>>>(
                        r.d_index_comp_rows, r.d_index_comp_state_kv,
                        r.d_index_comp_state_score, (uint32_t)opt.slots,
                        (uint32_t)kIndexerHeadDim, 4u, comp_row,
                        (uint32_t)kBoundedCompRows,
                        (uint32_t)kIndexCompStateRows,
                        (uint32_t)kIndexCompWidth);
                    compressor_norm_emit_slots_kernel<<<(unsigned int)opt.slots,
                                                        256, 0, r.stream>>>(
                        r.d_index_comp_rows, hc->d_indexer_compress_norm[layer],
                        (uint32_t)opt.slots, (uint32_t)kIndexerHeadDim, comp_row,
                        (uint32_t)kBoundedCompRows, 1.0e-6f);
                }
                if (fused_pool_norm_rope_round) {
                    // RoPE and F16 rounding were already applied by the fused emit.
                } else if (fused_rope_round) {
                    rope_tail_round_comp_emit_slots_kernel<<<
                        (unsigned int)opt.slots, 256, 0, r.stream>>>(
                        r.d_index_comp_rows, (uint32_t)opt.slots,
                        (uint32_t)kIndexerHeadDim, (uint32_t)kRotaryDim,
                        comp_row, (uint32_t)kBoundedCompRows,
                        r.d_decode_position, -3ll,
                        kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                        comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                        kRopeYarnBetaSlow);
                } else {
                    if (opt.true_ds4_attention_rope_gate) {
                        rope_tail_comp_emit_slots_kernel<<<
                            (unsigned int)opt.slots, 64, 0, r.stream>>>(
                            r.d_index_comp_rows, (uint32_t)opt.slots,
                            (uint32_t)kIndexerHeadDim, (uint32_t)kRotaryDim,
                            comp_row, (uint32_t)kBoundedCompRows,
                            r.d_decode_position, -3ll,
                            kRopeOrigCtx, kCompressRopeFreqBase, comp_freq_scale,
                            comp_ext_factor, comp_attn_factor, kRopeYarnBetaFast,
                            kRopeYarnBetaSlow);
                    }
                    round_comp_emit_slots_kernel<<<
                        (unsigned int)(((uint64_t)opt.slots *
                                            kIndexerHeadDim +
                                        block - 1) /
                                       block),
                        block, 0, r.stream>>>(
                        r.d_index_comp_rows, (uint32_t)opt.slots,
                        (uint32_t)kIndexerHeadDim, comp_row,
                        (uint32_t)kBoundedCompRows);
                }
                if (rank == 0 && !opt.true_ds4_attention_typed_kv_indexer_gate) {
                    indexer_score_bounded_rows_slots_kernel<<<
                        (unsigned int)opt.slots, 256, 0, r.stream>>>(
                        r.d_indexer_scores, r.d_indexer_topk,
                        hc->d_indexer_q_full, hc->d_indexer_w_full,
                        r.d_index_comp_rows, (uint32_t)opt.slots,
                        visible_after, (uint32_t)kBoundedCompRows,
                        (uint32_t)kIndexerTopK,
                        1.0f / sqrtf((float)(kIndexerHead * kIndexerHeadDim)));
                } else if (!opt.true_ds4_attention_typed_kv_indexer_gate) {
                    seed_single_topk_kernel<<<(unsigned int)opt.slots, 256, 0,
                                               r.stream>>>(
                        r.d_indexer_scores, r.d_indexer_topk,
                        (uint32_t)opt.slots, (uint32_t)kIndexerTopK);
                }
                r.index_comp_rows_written_layers[layer]++;
            }
            CHECK_CUDA(cudaGetLastError());
        }
        if (!graph_event_order) {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_state_emit_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
        if (opt.true_ds4_attention_typed_kv_indexer_gate && emitted) {
            if (!rt) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_typed_kv_indexer_failed\t"
                             "layer\t%d\treason\tmissing_tp_runtime\n",
                             layer);
                return 18;
            }
            char err[512] = {0};
            ds4_tp_kv_row_view view;
            if (ds4_tp_runtime_kv_row_view(
                    rt, layer, 0, opt.position, DS4_V100_TP_KV_ROW_INDEXER,
                    &view, err, sizeof(err)) != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_typed_kv_indexer_view_failed\t"
                             "layer\t%d\t%s\n",
                             layer, err);
                return 19;
            }
            const uint32_t bounded_row =
                (ranks[0].index_comp_rows_written_layers[layer] +
                 (uint32_t)kBoundedCompRows - 1u) %
                (uint32_t)kBoundedCompRows;
            const uint32_t visible_after =
                std::min(ranks[0].index_comp_rows_written_layers[layer],
                         (uint32_t)kBoundedCompRows);
            int current_store = 0;
            if (!opt.true_ds4_attention_typed_kv_skip_indexer_store_gate) {
                if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                    const void *src[kGpus] = {};
                    void *streams[kGpus] = {};
                    const size_t row_offset =
                        (size_t)bounded_row * (size_t)kIndexerHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        src[rank] = ranks[rank].d_index_comp_rows + row_offset;
                        streams[rank] = opt.decode_cudagraph_gate
                            ? (void *)ranks[rank].stream
                            : nullptr;
                    }
                    const int store_rc = opt.decode_cudagraph_gate
                        ? ds4_tp_runtime_kv_rows_store_f32_device_streams(
                              rt, layer, 0, (uint32_t)opt.slots, opt.position,
                              DS4_V100_TP_KV_ROW_INDEXER, src,
                              (uint64_t)kBoundedCompRows * (uint64_t)kIndexerHeadDim,
                              streams, err, sizeof(err))
                        : ds4_tp_runtime_kv_rows_store_f32_device(
                              rt, layer, 0, (uint32_t)opt.slots, opt.position,
                              DS4_V100_TP_KV_ROW_INDEXER, src,
                              (uint64_t)kBoundedCompRows * (uint64_t)kIndexerHeadDim,
                              err, sizeof(err));
                    if (store_rc != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_indexer_store_failed\t"
                                     "layer\t%d\tmode\tbatched\t%s\n",
                                     layer, err);
                        return 20;
                    }
                } else {
                    for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                        const void *src[kGpus] = {};
                        const size_t row_offset =
                            ((size_t)slot * (size_t)kBoundedCompRows +
                             (size_t)bounded_row) *
                            (size_t)kIndexerHeadDim;
                        for (int rank = 0; rank < kGpus; ++rank) {
                            src[rank] = ranks[rank].d_index_comp_rows + row_offset;
                        }
                        if (ds4_tp_runtime_kv_row_store_f32_device(
                                rt, layer, slot, opt.position,
                                DS4_V100_TP_KV_ROW_INDEXER, src, err,
                                sizeof(err)) != 0) {
                            std::fprintf(stderr,
                                         "tp_ep_true_attention_typed_kv_indexer_store_failed\t"
                                         "layer\t%d\tslot\t%u\t%s\n",
                                         layer, slot, err);
                            return 20;
                        }
                    }
                }
                current_store = 1;
            }
            sync_typed_kv_boundary(opt, ranks);
            int current_load = 0;
            if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
                current_store) {
                if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                    void *dst[kGpus] = {};
                    void *streams[kGpus] = {};
                    const size_t row_offset =
                        (size_t)bounded_row * (size_t)kIndexerHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        dst[rank] = ranks[rank].d_index_comp_rows + row_offset;
                        streams[rank] = opt.decode_cudagraph_gate
                            ? (void *)ranks[rank].stream
                            : nullptr;
                    }
                    const int load_rc = opt.decode_cudagraph_gate
                        ? ds4_tp_runtime_kv_rows_load_f32_device_streams(
                              rt, layer, 0, (uint32_t)opt.slots, opt.position,
                              DS4_V100_TP_KV_ROW_INDEXER, dst,
                              (uint64_t)kBoundedCompRows * (uint64_t)kIndexerHeadDim,
                              streams, err, sizeof(err))
                        : ds4_tp_runtime_kv_rows_load_f32_device(
                              rt, layer, 0, (uint32_t)opt.slots, opt.position,
                              DS4_V100_TP_KV_ROW_INDEXER, dst,
                              (uint64_t)kBoundedCompRows * (uint64_t)kIndexerHeadDim,
                              err, sizeof(err));
                    if (load_rc != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_indexer_load_failed\t"
                                     "layer\t%d\tmode\tbatched\t%s\n",
                                     layer, err);
                        return 21;
                    }
                } else {
                    for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                        void *dst[kGpus] = {};
                        const size_t row_offset =
                            ((size_t)slot * (size_t)kBoundedCompRows +
                             (size_t)bounded_row) *
                            (size_t)kIndexerHeadDim;
                        for (int rank = 0; rank < kGpus; ++rank) {
                            dst[rank] = ranks[rank].d_index_comp_rows + row_offset;
                        }
                        if (ds4_tp_runtime_kv_row_load_f32_device(
                                rt, layer, slot, opt.position,
                                DS4_V100_TP_KV_ROW_INDEXER, dst, err,
                                sizeof(err)) != 0) {
                            std::fprintf(stderr,
                                         "tp_ep_true_attention_typed_kv_indexer_load_failed\t"
                                         "layer\t%d\tslot\t%u\t%s\n",
                                         layer, slot, err);
                            return 21;
                        }
                    }
                }
                current_load = 1;
            }
            sync_typed_kv_boundary(opt, ranks);
            if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate ||
                !current_store) {
                for (int rank = 0; rank < kGpus; ++rank) {
                    ranks[rank].index_comp_row_loaded_layers[layer][bounded_row] = true;
                    ranks[rank].index_comp_row_loaded_position_layers[layer][bounded_row] =
                        opt.position;
                }
            }
            CHECK_CUDA(cudaSetDevice(ranks[0].device));
            indexer_score_bounded_rows_slots_kernel<<<
                (unsigned int)opt.slots, 256, 0, ranks[0].stream>>>(
                ranks[0].d_indexer_scores, ranks[0].d_indexer_topk,
                hc->d_indexer_q_full, hc->d_indexer_w_full,
                ranks[0].d_index_comp_rows, (uint32_t)opt.slots, visible_after,
                (uint32_t)kBoundedCompRows, (uint32_t)kIndexerTopK,
                1.0f / sqrtf((float)(kIndexerHead * kIndexerHeadDim)));
            CHECK_CUDA(cudaGetLastError());
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                seed_single_topk_kernel<<<(unsigned int)opt.slots, 256, 0,
                                           ranks[rank].stream>>>(
                    ranks[rank].d_indexer_scores, ranks[rank].d_indexer_topk,
                    (uint32_t)opt.slots, (uint32_t)kIndexerTopK);
                CHECK_CUDA(cudaGetLastError());
            }
            if (graph_event_order) {
                const int slot = next_graph_order_event_slot(ranks);
                CHECK_CUDA(cudaSetDevice(ranks[0].device));
                cudaEvent_t ev = graph_stream_done_event(ranks[0], slot);
                if (!ev) return 28;
                CHECK_CUDA(cudaEventRecord(ev, ranks[0].stream));
                for (int rank = 1; rank < kGpus; ++rank) {
                    CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                    CHECK_CUDA(cudaStreamWaitEvent(ranks[rank].stream,
                                                   ev, 0));
                }
            } else {
                for (int rank = 0; rank < kGpus; ++rank) {
                    CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                    CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
                }
            }
            if (!opt.true_ds4_attention_typed_kv_quiet_gate) {
                std::printf("tp_ep_true_attention_typed_kv_indexer\tlayer\t%d\t"
                            "slots\t%d\tratio\t%d\tposition\t%llu\t"
                            "bounded_row\t%u\tvisible_rows\t%u\tphysical_row\t%llu\t"
                            "logical_cols\t%u\tlogical_row_bytes\t%llu\t"
                            "row_bytes_per_gpu\t%llu\tcurrent_store\t%d\t"
                            "current_load\t%d\tPASS\n",
                            layer, opt.slots, ratio,
                            (unsigned long long)opt.position, bounded_row,
                            visible_after, (unsigned long long)view.physical_row,
                            view.logical_cols,
                            (unsigned long long)view.logical_row_bytes,
                            (unsigned long long)view.row_bytes[0], current_store,
                            current_load);
            }
        }
        if (emitted && ranks[0].d_indexer_topk) {
            const uint64_t topk_elems =
                (uint64_t)opt.slots * (uint64_t)kIndexerTopK;
            const size_t topk_bytes = (size_t)topk_elems * sizeof(uint32_t);
            if (!graph_event_order) {
                void *topk_dsts[kGpus] = {};
                for (int rank = 0; rank < kGpus; ++rank) {
                    topk_dsts[rank] = ranks[rank].d_indexer_topk;
                }
                if (nccl_broadcast_bytes_from_rank0(
                        ranks, ranks[0].d_indexer_topk, topk_dsts,
                        topk_bytes, "indexer_topk_emit") != 0) {
                    return 29;
                }
            }
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                if (graph_event_order) {
                    copy_u32_kernel<<<
                        (unsigned int)((topk_elems + block - 1) / block),
                        block, 0, ranks[rank].stream>>>(
                        ranks[rank].d_indexer_topk, ranks[0].d_indexer_topk,
                        topk_elems);
                    CHECK_CUDA(cudaGetLastError());
                }
            }
            if (!graph_event_order) {
                for (int rank = 1; rank < kGpus; ++rank) {
                    CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                    CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
                }
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            indexer_typed_score_ms = elapsed_ms(t_stage, now);
            t_stage = now;
        }
    }

    const auto diff_start = std::chrono::steady_clock::now();
    const int diff_rc = run_true_ds4_compressed_reference_diff_gate(
        opt, hc, ranks, layer, ratio, comp_width, emitted, emitted_comp_row,
        visible);
    if (diff_rc != 0) return diff_rc;
    reference_diff_ms =
        elapsed_ms(diff_start, std::chrono::steady_clock::now());
    const auto shift_start = std::chrono::steady_clock::now();
    if (emitted && ratio == 4) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            compressor_shift_ratio4_slots_kernel<<<
                (unsigned int)(((uint64_t)opt.slots * 4ull *
                                    (uint64_t)comp_width +
                                block - 1) /
                               block),
                block, 0, r.stream>>>(
                r.d_attn_comp_state_kv, r.d_attn_comp_state_score,
                (uint32_t)opt.slots, (uint32_t)comp_width,
                (uint32_t)comp_state_rows, (uint32_t)comp_state_width);
            if (opt.true_ds4_indexer_attention_gate && r.d_index_comp_state_kv &&
                r.d_index_comp_state_score) {
                compressor_shift_ratio4_slots_kernel<<<
                    (unsigned int)(((uint64_t)opt.slots * 4ull *
                                        (uint64_t)kIndexCompWidth +
                                    block - 1) /
                                   block),
                    block, 0, r.stream>>>(
                    r.d_index_comp_state_kv, r.d_index_comp_state_score,
                    (uint32_t)opt.slots, (uint32_t)kIndexCompWidth,
                    (uint32_t)kIndexCompStateRows,
                    (uint32_t)kIndexCompWidth);
            }
            CHECK_CUDA(cudaGetLastError());
        }
        if (!graph_event_order) {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }
    ratio_shift_ms =
        elapsed_ms(shift_start, std::chrono::steady_clock::now());

    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    std::printf("tp_ep_compressed_kv_projection\tlayer\t%d\tslots\t%d\t"
                "ratio\t%d\temitted_compressed_rows\t%u\t"
                "visible_compressed_rows\t%u\tindexer_topk_count\t%u\t"
                "attn_comp_width\t%d\tattn_kv_max\t%.9g\tattn_kv_bad\t%d\t"
                "attn_gate_max\t%.9g\tattn_gate_bad\t%d\t"
                "index_q_max\t%.9g\tindex_q_bad\t%d\t"
                "index_w_max\t%.9g\tindex_w_bad\t%d\t"
                "index_kv_max\t%.9g\tindex_kv_bad\t%d\t"
                "index_gate_max\t%.9g\tindex_gate_bad\t%d\t"
                "attn_input_fill_ms\t%.6f\tattn_dense_ms\t%.6f\t"
                "attn_gather_ms\t%.6f\tattn_state_emit_ms\t%.6f\t"
                "attn_typed_ms\t%.6f\tindexer_input_fill_ms\t%.6f\t"
                "indexer_dense_ms\t%.6f\tindexer_gather_rope_ms\t%.6f\t"
                "indexer_state_emit_ms\t%.6f\tindexer_typed_score_ms\t%.6f\t"
                "reference_diff_ms\t%.6f\tratio_shift_ms\t%.6f\t"
                "direct_input_fill\t%d\tdense_event_wait\t%d\t"
                "skip_dense_stats\t%d\t"
                "fused_attn_input_fill\t%d\t"
                "fused_input_fill\t%d\tfused_rope_round\t%d\t"
                "fused_pool_norm\t%d\tfused_pool_norm_rope_round\t%d\t"
                "ms\t%.6f\tPASS\n",
                layer, opt.slots, ratio, emitted, visible, indexer_topk,
                comp_width, attn_kv_stats.max_abs, attn_kv_stats.finite_bad,
                attn_gate_stats.max_abs, attn_gate_stats.finite_bad,
                index_q_stats.max_abs, index_q_stats.finite_bad,
                index_w_stats.max_abs, index_w_stats.finite_bad,
                index_kv_stats.max_abs, index_kv_stats.finite_bad,
                index_gate_stats.max_abs, index_gate_stats.finite_bad,
                attn_input_fill_ms, attn_dense_ms, attn_gather_ms,
                attn_state_emit_ms, attn_typed_ms, indexer_input_fill_ms,
                indexer_dense_ms, indexer_gather_rope_ms,
                indexer_state_emit_ms, indexer_typed_score_ms,
                reference_diff_ms, ratio_shift_ms,
                direct_current_input_fill ? 1 : 0,
                dense_event_wait ? 1 : 0,
                skip_dense_stats ? 1 : 0,
                fused_attn_current_fill ? 1 : 0,
                fused_ratio4_current_fill ? 1 : 0,
                fused_rope_round ? 1 : 0,
                fused_pool_norm ? 1 : 0,
                fused_pool_norm_rope_round ? 1 : 0, ms);
    return (!skip_dense_stats &&
            (attn_kv_stats.finite_bad || attn_gate_stats.finite_bad ||
            index_q_stats.finite_bad || index_w_stats.finite_bad ||
             index_kv_stats.finite_bad || index_gate_stats.finite_bad)) ? 8 : 0;
}

int run_true_ds4_attention_state_update(const Options &opt,
                                        SharedHcControls *hc,
                                        const LayerDenseOps *ops,
                                        RankState ranks[kGpus],
                                        ds4_tp_runtime *rt,
                                        int layer) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    if (!hc->d_kv_normed ||
        ops->attn_q_b.rows_per_gpu != kLocalHeads * kHeadDim) {
        return 2;
    }

    const auto start = std::chrono::steady_clock::now();
    const int block = 256;
    const uint32_t raw_row = (uint32_t)(opt.position % kRawSwaRows);
    const uint64_t kv_elems = (uint64_t)opt.slots * (uint64_t)kHeadDim;
    const uint64_t raw_elems =
        (uint64_t)opt.slots * (uint64_t)kRawSwaRows * (uint64_t)kHeadDim;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    const int ratio = ds4_layer_ratio(layer);
    const bool compressed = ratio != 0;
    const float freq_base =
        compressed ? kCompressRopeFreqBase : kRopeFreqBase;
    const float freq_scale =
        compressed && kRopeScaleFactor > 0.0f ? 1.0f / kRopeScaleFactor : 1.0f;
    const float ext_factor =
        compressed && kRopeScaleFactor > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    if (!graph_event_order) {
        void *kv_dsts[kGpus] = {};
        for (int rank = 0; rank < kGpus; ++rank) {
            if (!ranks[rank].d_attn_kv_full) return 3;
            kv_dsts[rank] = ranks[rank].d_attn_kv_full;
        }
        if (nccl_broadcast_bytes_from_rank0(
                ranks, hc->d_kv_normed, kv_dsts,
                (size_t)kv_elems * sizeof(float), "attention_state_kv_full") != 0) {
            return 9;
        }
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_kv_full || !r.d_attn_raw_swa ||
            !ops->attn_q_b.d_out[(size_t)rank]) {
            return 3;
        }
        head_rms_norm_local_heads_kernel<<<
            (unsigned int)(opt.slots * kLocalHeads), 256, 0,
            r.dense_stream ? r.dense_stream : r.stream>>>(
            ops->attn_q_b.d_out[(size_t)rank], (uint32_t)opt.slots,
            (uint32_t)kLocalHeads, (uint32_t)kHeadDim, 1.0e-6f);
        CHECK_CUDA(cudaGetLastError());
        if (opt.true_ds4_attention_rope_gate) {
            rope_tail_rows_kernel<<<
                (unsigned int)(opt.slots * kLocalHeads), 64, 0,
                r.dense_stream ? r.dense_stream : r.stream>>>(
                ops->attn_q_b.d_out[(size_t)rank],
                (uint32_t)(opt.slots * kLocalHeads), (uint32_t)kHeadDim,
                (uint32_t)kRotaryDim, r.d_decode_position, 0ll,
                compressed ? kRopeOrigCtx : 0u, 0, freq_base, freq_scale,
                ext_factor, attn_factor, kRopeYarnBetaFast,
                kRopeYarnBetaSlow);
            CHECK_CUDA(cudaGetLastError());
        }
        if (graph_event_order) {
            enqueue_graph_f32_copy_from_device0(
                opt, r, rank, r.d_attn_kv_full, hc->d_kv_normed, kv_elems,
                r.stream, block);
        }
        if (opt.true_ds4_attention_rope_gate) {
            rope_tail_rows_kernel<<<
                (unsigned int)opt.slots, 64, 0, r.stream>>>(
                r.d_attn_kv_full, (uint32_t)opt.slots, (uint32_t)kHeadDim,
                (uint32_t)kRotaryDim, r.d_decode_position, 0ll,
                compressed ? kRopeOrigCtx : 0u, 0, freq_base, freq_scale,
                ext_factor, attn_factor, kRopeYarnBetaFast,
                kRopeYarnBetaSlow);
            CHECK_CUDA(cudaGetLastError());
        }
        if (!opt.true_ds4_attention_typed_kv_raw_gate ||
            opt.true_ds4_attention_typed_kv_skip_current_load_gate) {
            kv_fp8_round_store_raw_swa_kernel<<<
                (unsigned int)((kv_elems + block - 1) / block), block, 0,
                r.stream>>>(
                r.d_attn_raw_swa, r.d_attn_kv_full, r.d_decode_position,
                (uint32_t)opt.slots, (uint32_t)kRawSwaRows, (uint32_t)kHeadDim,
                (uint32_t)kRotaryDim);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    if (opt.decode_cudagraph_gate) {
        if (enqueue_rank_streams_wait_after_dense_streams(ranks) != 0) {
            return 8;
        }
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            if (ranks[rank].dense_stream) {
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].dense_stream));
            }
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    if (opt.true_ds4_attention_typed_kv_raw_gate) {
        if (!rt) {
            std::fprintf(stderr,
                         "tp_ep_true_attention_typed_kv_raw_failed\tlayer\t%d\t"
                         "reason\tmissing_tp_runtime\n",
                         layer);
            return 4;
        }
        char err[512] = {0};
        ds4_tp_kv_row_view view;
        if (ds4_tp_runtime_kv_row_view(
                rt, layer, 0, opt.position, DS4_V100_TP_KV_ROW_ATTN_RAW, &view,
                err, sizeof(err)) != 0) {
            std::fprintf(stderr,
                         "tp_ep_true_attention_typed_kv_raw_view_failed\tlayer\t%d\t%s\n",
                         layer, err);
            return 5;
        }
        int current_store = 0;
        if (!opt.true_ds4_attention_typed_kv_skip_raw_store_gate) {
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                const void *src[kGpus] = {};
                void *streams[kGpus] = {};
                const void *positions[kGpus] = {};
                for (int rank = 0; rank < kGpus; ++rank) {
                    src[rank] = ranks[rank].d_attn_kv_full;
                    streams[rank] = opt.decode_cudagraph_gate
                        ? (void *)ranks[rank].stream
                        : nullptr;
                    positions[rank] = ranks[rank].d_decode_position;
                }
                const int store_rc = opt.decode_cudagraph_gate
                    ? ds4_tp_runtime_kv_rows_store_f32_device_streams_at_position(
                          rt, layer, 0, (uint32_t)opt.slots,
                          DS4_V100_TP_KV_ROW_ATTN_RAW, src,
                          (uint64_t)kHeadDim, streams, positions, err,
                          sizeof(err))
                    : ds4_tp_runtime_kv_rows_store_f32_device(
                          rt, layer, 0, (uint32_t)opt.slots, opt.position,
                          DS4_V100_TP_KV_ROW_ATTN_RAW, src,
                          (uint64_t)kHeadDim, err, sizeof(err));
                if (store_rc != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_raw_store_failed\t"
                                 "layer\t%d\tmode\tbatched\t%s\n",
                                 layer, err);
                    return 6;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    const void *src[kGpus] = {};
                    for (int rank = 0; rank < kGpus; ++rank) {
                        src[rank] = ranks[rank].d_attn_kv_full +
                                    (size_t)slot * (size_t)kHeadDim;
                    }
                    if (ds4_tp_runtime_kv_row_store_f32_device(
                            rt, layer, slot, opt.position,
                            DS4_V100_TP_KV_ROW_ATTN_RAW, src, err,
                            sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_raw_store_failed\t"
                                     "layer\t%d\tslot\t%u\t%s\n",
                                     layer, slot, err);
                        return 6;
                    }
                }
            }
            current_store = 1;
        }
        sync_typed_kv_boundary(opt, ranks);
        int current_load = 0;
        if (!opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
            current_store) {
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                void *dst[kGpus] = {};
                void *streams[kGpus] = {};
                const void *positions[kGpus] = {};
                for (int rank = 0; rank < kGpus; ++rank) {
                    dst[rank] = ranks[rank].d_attn_raw_swa;
                    streams[rank] = opt.decode_cudagraph_gate
                        ? (void *)ranks[rank].stream
                        : nullptr;
                    positions[rank] = ranks[rank].d_decode_position;
                }
                const int load_rc = opt.decode_cudagraph_gate
                    ? ds4_tp_runtime_kv_rows_load_f32_device_streams_at_position(
                          rt, layer, 0, (uint32_t)opt.slots,
                          DS4_V100_TP_KV_ROW_ATTN_RAW, dst,
                          (uint64_t)kRawSwaRows * (uint64_t)kHeadDim,
                          streams, positions, err, sizeof(err))
                    : ds4_tp_runtime_kv_rows_load_f32_device(
                          rt, layer, 0, (uint32_t)opt.slots, opt.position,
                          DS4_V100_TP_KV_ROW_ATTN_RAW, dst,
                          (uint64_t)kRawSwaRows * (uint64_t)kHeadDim,
                          err, sizeof(err));
                if (load_rc != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_raw_load_failed\t"
                                 "layer\t%d\tmode\tbatched\t%s\n",
                                 layer, err);
                    return 7;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    void *dst[kGpus] = {};
                    const size_t row_offset =
                        ((size_t)slot * (size_t)kRawSwaRows + (size_t)raw_row) *
                        (size_t)kHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        dst[rank] = ranks[rank].d_attn_raw_swa + row_offset;
                    }
                    if (ds4_tp_runtime_kv_row_load_f32_device(
                            rt, layer, slot, opt.position,
                            DS4_V100_TP_KV_ROW_ATTN_RAW, dst, err,
                            sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_raw_load_failed\t"
                                     "layer\t%d\tslot\t%u\t%s\n",
                                     layer, slot, err);
                        return 7;
                    }
                }
            }
            current_load = 1;
        }
        sync_typed_kv_boundary(opt, ranks);
        if (!opt.true_ds4_attention_typed_kv_quiet_gate) {
            std::printf("tp_ep_true_attention_typed_kv_raw\tlayer\t%d\tslots\t%d\t"
                        "position\t%llu\tphysical_row\t%llu\traw_row\t%u\tlogical_cols\t%u\t"
                        "logical_row_bytes\t%llu\trow_bytes_per_gpu\t%llu\t"
                        "current_store\t%d\tcurrent_load\t%d\tPASS\n",
                        layer, opt.slots, (unsigned long long)opt.position,
                        (unsigned long long)view.physical_row, raw_row,
                        view.logical_cols, (unsigned long long)view.logical_row_bytes,
                        (unsigned long long)view.row_bytes[0], current_store,
                        current_load);
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    if (!opt.decode_cudagraph_gate && layer <= 2) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("true_attn_q_heads_normed_shard", layer, rank,
                                 ops->attn_q_b.d_out[(size_t)rank],
                                 (size_t)opt.slots * ops->attn_q_b.rows_per_gpu,
                                 ranks[rank].dense_stream ? ranks[rank].dense_stream
                                                          : ranks[rank].stream);
        }
        CHECK_CUDA(cudaSetDevice(ranks[0].device));
        log_tensor_f32_stats("true_attn_raw_swa_rank0", layer, 0,
                             ranks[0].d_attn_raw_swa, (size_t)raw_elems,
                             ranks[0].stream);
    }
    if (opt.true_ds4_attention_saturation_audit_gate) {
        TensorF32Stats q_heads;
        TensorF32Stats kv_rope;
        TensorF32Stats raw_row_stats;
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            const cudaStream_t q_stream =
                ranks[rank].dense_stream ? ranks[rank].dense_stream
                                         : ranks[rank].stream;
            merge_tensor_stats(
                &q_heads,
                collect_tensor_f32_stats(
                    ops->attn_q_b.d_out[(size_t)rank],
                    (size_t)opt.slots * ops->attn_q_b.rows_per_gpu,
                    q_stream));
            merge_tensor_stats(
                &kv_rope,
                collect_tensor_f32_stats(ranks[rank].d_attn_kv_full,
                                         (size_t)kv_elems,
                                         ranks[rank].stream));
            merge_tensor_stats(
                &raw_row_stats,
                collect_raw_swa_row_stats(ranks[rank].d_attn_raw_swa,
                                          (uint32_t)opt.slots,
                                          (uint32_t)kRawSwaRows, raw_row,
                                          (uint32_t)kHeadDim,
                                          ranks[rank].stream));
        }
        std::printf("tp_ep_true_attention_saturation_state\tlayer\t%d\t"
                    "slots\t%d\traw_row\t%u\tq_heads_post_rope_max\t%.9g\t"
                    "q_heads_post_rope_bad\t%d\tkv_post_rope_max\t%.9g\t"
                    "kv_post_rope_bad\t%d\traw_swa_row_max\t%.9g\t"
                    "raw_swa_row_bad\t%d\tPASS\n",
                    layer, opt.slots, raw_row, q_heads.max_abs,
                    q_heads.finite_bad, kv_rope.max_abs, kv_rope.finite_bad,
                    raw_row_stats.max_abs, raw_row_stats.finite_bad);
    }
    if (opt.true_ds4_attention_rope_gate) {
        std::printf("tp_ep_true_attention_rope\tlayer\t%d\tslots\t%d\t"
                    "local_heads\t%d\thead_dim\t%d\trotary_dim\t%d\t"
                    "freq_base\t%.1f\tfreq_scale\t%.9f\tposition\t%llu\tPASS\n",
                    layer, opt.slots, kLocalHeads, kHeadDim, kRotaryDim,
                    freq_base, freq_scale, (unsigned long long)opt.position);
    }
    std::printf("tp_ep_true_attention_state_update\tlayer\t%d\tslots\t%d\t"
                "local_heads\t%d\thead_dim\t%d\traw_rows\t%d\traw_row\t%u\t"
                "kv_width\t%d\tms\t%.6f\tPASS\n",
                layer, opt.slots, kLocalHeads, kHeadDim, kRawSwaRows, raw_row,
                kHeadDim, ms);
    return 0;
}
