int run_true_ds4_attention_projection_prefix(const Options &opt,
                                             SharedHcControls *hc,
                                             const LayerDenseOps *ops,
                                             RankState ranks[kGpus],
                                             int layer) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 44) {
        return 1;
    }
    if (!hc->d_current_full || !hc->d_attn_normed ||
        !hc->d_q_a_full || !hc->d_q_a_normed ||
        !hc->d_kv_full || !hc->d_kv_normed ||
        !hc->d_attn_norm_weight[layer] ||
        !hc->d_q_a_norm_weight[layer] ||
        !hc->d_kv_a_norm_weight[layer]) {
        return 2;
    }
    if (ops->attn_q_a.cols != kHidden || ops->attn_q_a.rows_per_gpu != 1024 / kGpus ||
        ops->attn_q_b.cols != 1024 || ops->attn_q_b.rows_per_gpu != 32768 / kGpus ||
        ops->attn_kv_latent.cols != kHidden ||
        ops->attn_kv_latent.rows_per_gpu != kHeadDim / kGpus) {
        std::fprintf(stderr,
                     "tp_ep_true_attention_projection_bad_shape\tlayer\t%d\t"
                     "q_a_cols\t%d\tq_a_rows_per_gpu\t%d\t"
                     "q_b_cols\t%d\tq_b_rows_per_gpu\t%d\t"
                     "kv_cols\t%d\tkv_rows_per_gpu\t%d\n",
                     layer,
                     ops->attn_q_a.cols, ops->attn_q_a.rows_per_gpu,
                     ops->attn_q_b.cols, ops->attn_q_b.rows_per_gpu,
                     ops->attn_kv_latent.cols, ops->attn_kv_latent.rows_per_gpu);
        return 3;
    }

    const auto start = std::chrono::steady_clock::now();
    const int block = 256;
    const uint64_t hidden_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const uint64_t q_a_elems = (uint64_t)opt.slots * 1024ull;
    const uint64_t kv_elems = (uint64_t)opt.slots * (uint64_t)kHeadDim;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    const bool direct_input_fill =
        opt.true_ds4_attention_projection_direct_input_fill_gate;
    const bool rank_major_input =
        opt.true_ds4_attention_projection_rank_major_input_gate &&
        opt.tp_hc_current_input_nccl_allgather_gate;
    const bool rank_local_input =
        opt.true_ds4_attention_projection_rank_local_input_gate ||
        rank_major_input;
    const bool refresh_rank_major_from_slot_major =
        rank_major_input && opt.routed_ffn_norm_input_gate;
    const bool gathered_current_full =
        opt.tp_hc_current_input_peer_gather_gate ||
        opt.tp_hc_current_input_nccl_allgather_gate;
    const bool broadcast_normed_current = !direct_input_fill && !rank_major_input;
    float *attention_current_full = hc->d_current_full;
    if (gathered_current_full && ranks[0].d_current_full) {
        attention_current_full = ranks[0].d_current_full;
    }
    cudaStream_t control_stream = graph_event_order ? ranks[0].stream : (cudaStream_t)0;

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    rms_norm_weight_rows_stable_kernel<<<
        (unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_attn_normed, attention_current_full, hc->d_attn_norm_weight[layer],
        (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());
    if (enqueue_rank_streams_wait_after_control(
            opt, ranks, control_stream) != 0) return 8;

    if (broadcast_normed_current) {
        const int bcast_rc = nccl_broadcast_f32_from_device0_to_current_full(
            opt, ranks, hc->d_attn_normed, hidden_elems,
            "attention_projection_normed_current");
        if (bcast_rc != 0) return 14;
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 15;
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_current_full ||
            !ops->attn_q_a.d_x_half[(size_t)rank] ||
            !ops->attn_kv_latent.d_x_half[(size_t)rank]) {
            return 4;
        }
        if (rank_local_input) {
            float *rank_weight = hc->d_attn_norm_weight_rank[layer][rank];
            if (!rank_weight) return 12;
            if (broadcast_normed_current) {
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->attn_q_a.d_x_half[(size_t)rank],
                                 r.d_current_full, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->attn_kv_latent.d_x_half[(size_t)rank],
                                 r.d_current_full, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
            } else if (rank_major_input && r.d_current_full_rank_major) {
                if (refresh_rank_major_from_slot_major) {
                    slot_major_current_to_rank_major_kernel<<<
                        (unsigned int)((hidden_elems + block - 1) / block),
                        block, 0, r.stream>>>(
                        r.d_current_full_rank_major, r.d_current_full,
                        (uint32_t)(kHidden / kGpus), (uint32_t)kGpus,
                        (uint32_t)opt.slots);
                    CHECK_CUDA(cudaGetLastError());
                }
                fill_two_hidden_inputs_half_from_rank_major_norm_kernel<<<
                    (unsigned int)opt.slots, 256, 0, r.stream>>>(
                    ops->attn_q_a.d_x_half[(size_t)rank],
                    ops->attn_kv_latent.d_x_half[(size_t)rank],
                    r.d_current_full_rank_major, rank_weight,
                    (uint32_t)(kHidden / kGpus), (uint32_t)kGpus,
                    (uint32_t)opt.slots, 1.0e-6f);
            } else {
                if (!r.d_current_full_normed) return 13;
                rms_norm_weight_rows_stable_kernel<<<
                    (unsigned int)opt.slots, 256, 0, r.stream>>>(
                    r.d_current_full_normed, r.d_current_full, rank_weight,
                    (uint32_t)kHidden, (uint32_t)opt.slots, 1.0e-6f);
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->attn_q_a.d_x_half[(size_t)rank],
                                 r.d_current_full_normed, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
                fill_dense_input_half_from_current_kernel<<<
                    (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                    r.stream>>>(ops->attn_kv_latent.d_x_half[(size_t)rank],
                                 r.d_current_full_normed, (uint32_t)kHidden,
                                 (uint32_t)opt.slots);
            }
        } else if (direct_input_fill) {
            fill_two_hidden_inputs_half_from_current_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->attn_q_a.d_x_half[(size_t)rank],
                             ops->attn_kv_latent.d_x_half[(size_t)rank],
                             hc->d_attn_normed, (uint32_t)opt.slots);
        } else {
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->attn_q_a.d_x_half[(size_t)rank],
                             r.d_current_full, (uint32_t)kHidden,
                             (uint32_t)opt.slots);
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((hidden_elems + block - 1) / block), block, 0,
                r.stream>>>(ops->attn_kv_latent.d_x_half[(size_t)rank],
                             r.d_current_full, (uint32_t)kHidden,
                             (uint32_t)opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 9;

    if (!graph_event_order && opt.true_ds4_attention_projection_input_parity_gate) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            cudaStream_t stream = r.stream ? r.stream : (cudaStream_t)0;
            const HalfInputDiffStats q_a_diff = collect_half_input_tensor_diff(
                r, ops->attn_q_a.d_x_half[(size_t)rank], hc->d_attn_normed,
                (uint32_t)kHidden, (uint32_t)opt.slots, stream);
            log_attention_projection_input_diff("attn_q_a_input", layer, rank,
                                                q_a_diff);
            if (rank_major_input && rank == 0 && layer <= 1) {
                log_attention_rank_major_input_debug(
                    "attn_q_a_input", layer, r,
                    ops->attn_q_a.d_x_half[(size_t)rank], hc->d_attn_normed,
                    r.d_current_full, r.d_current_full_rank_major,
                    hc->d_attn_norm_weight[layer], (uint32_t)opt.slots,
                    stream);
            }
            const HalfInputDiffStats kv_diff = collect_half_input_tensor_diff(
                r, ops->attn_kv_latent.d_x_half[(size_t)rank], hc->d_attn_normed,
                (uint32_t)kHidden, (uint32_t)opt.slots, stream);
            log_attention_projection_input_diff("attn_kv_latent_input", layer,
                                                rank, kv_diff);
        }
    }

    if (launch_resident_f8_dense(opt, ops->attn_q_a, ranks) != 0 ||
        launch_resident_f8_dense(opt, ops->attn_kv_latent, ranks) != 0) {
        return 5;
    }
    if (enqueue_control_wait_after_dense_streams(
            opt, ranks, control_stream) != 0) return 10;

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        gather_dense_shard_to_full_kernel<<<
            (unsigned int)(((uint64_t)opt.slots * (1024u / kGpus) + block - 1) / block),
            block, 0, control_stream>>>(
            hc->d_q_a_full, ops->attn_q_a.d_out[(size_t)rank], rank,
            1024u / kGpus, 1024u, (uint32_t)opt.slots);
        gather_dense_shard_to_full_kernel<<<
            (unsigned int)(((uint64_t)opt.slots * (kHeadDim / kGpus) + block - 1) / block),
            block, 0, control_stream>>>(
            hc->d_kv_full, ops->attn_kv_latent.d_out[(size_t)rank], rank,
            kHeadDim / kGpus, kHeadDim, (uint32_t)opt.slots);
    }
    CHECK_CUDA(cudaGetLastError());
    rms_norm_weight_rows_stable_kernel<<<
        (unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_q_a_normed, hc->d_q_a_full, hc->d_q_a_norm_weight[layer],
        1024u, (uint32_t)opt.slots, 1.0e-6f);
    rms_norm_weight_rows_stable_kernel<<<
        (unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_kv_normed, hc->d_kv_full, hc->d_kv_a_norm_weight[layer],
        (uint32_t)kHeadDim, (uint32_t)opt.slots, 1.0e-6f);
    CHECK_CUDA(cudaGetLastError());
    if (enqueue_rank_streams_wait_after_control(
            opt, ranks, control_stream) != 0) return 11;

    if (opt.true_ds4_attention_kv_norm_reference_gate) {
        float *d_kv_ref = nullptr;
        CHECK_CUDA(cudaMalloc(&d_kv_ref, (size_t)kv_elems * sizeof(float)));
        rms_norm_weight_rows_kernel<<<(unsigned int)opt.slots, 256>>>(
            d_kv_ref, hc->d_kv_full, hc->d_kv_a_norm_weight[layer],
            (uint32_t)kHeadDim, (uint32_t)opt.slots, 1.0e-6f);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        const TensorF32Stats kv_in =
            collect_tensor_f32_stats(hc->d_kv_full, (size_t)kv_elems, nullptr);
        const TensorF32Stats kv_stable =
            collect_tensor_f32_stats(hc->d_kv_normed, (size_t)kv_elems,
                                     nullptr);
        const TensorF32Stats kv_ref =
            collect_tensor_f32_stats(d_kv_ref, (size_t)kv_elems, nullptr);
        const TensorF32Stats kv_w =
            collect_tensor_f32_stats(hc->d_kv_a_norm_weight[layer],
                                     (size_t)kHeadDim, nullptr);
        const TensorF32DiffStats diff = collect_tensor_f32_diff_stats(
            hc->d_kv_normed, d_kv_ref, (size_t)kv_elems, nullptr);
        std::printf("tp_ep_true_attention_kv_norm_reference\tlayer\t%d\t"
                    "slots\t%d\tkv_in_max\t%.9g\tkv_in_bad\t%d\t"
                    "kv_weight_max\t%.9g\tkv_weight_bad\t%d\t"
                    "stable_max\t%.9g\tstable_bad\t%d\t"
                    "reference_max\t%.9g\treference_bad\t%d\t"
                    "max_abs_diff\t%.9g\tmax_rel_diff\t%.9g\tdiff_bad\t%d\t"
                    "first_bad\t%zu\tPASS\n",
                    layer, opt.slots, kv_in.max_abs, kv_in.finite_bad,
                    kv_w.max_abs, kv_w.finite_bad, kv_stable.max_abs,
                    kv_stable.finite_bad, kv_ref.max_abs, kv_ref.finite_bad,
                    diff.max_abs, diff.max_rel, diff.bad, diff.first_bad);
        CHECK_CUDA(cudaFree(d_kv_ref));
    }

    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!ops->attn_q_b.d_x_half[(size_t)rank]) return 6;
        fill_dense_input_half_from_tensor_kernel<<<
            (unsigned int)((q_a_elems + block - 1) / block), block, 0,
            r.stream>>>(ops->attn_q_b.d_x_half[(size_t)rank],
                         hc->d_q_a_normed, 1024u, (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 12;

    if (launch_resident_f8_dense(opt, ops->attn_q_b, ranks) != 0) {
        return 7;
    }
    if (enqueue_rank_streams_wait_after_dense_streams(ranks) != 0) return 7;

    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    if (!graph_event_order && layer <= 2) {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        log_tensor_f32_stats("true_attn_q_a_full", layer, 0, hc->d_q_a_full,
                             (size_t)q_a_elems, nullptr);
        log_tensor_f32_stats("true_attn_kv_normed", layer, 0, hc->d_kv_normed,
                             (size_t)kv_elems, nullptr);
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("true_attn_q_b_shard", layer, rank,
                                 ops->attn_q_b.d_out[(size_t)rank],
                                 (size_t)opt.slots * ops->attn_q_b.rows_per_gpu,
                                 ranks[rank].dense_stream ? ranks[rank].dense_stream
                                                          : ranks[rank].stream);
        }
    }
    if (!graph_event_order && opt.true_ds4_attention_saturation_audit_gate) {
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        const TensorF32Stats current =
            collect_tensor_f32_stats(attention_current_full, (size_t)hidden_elems,
                                     nullptr);
        const TensorF32Stats attn_normed =
            collect_tensor_f32_stats(hc->d_attn_normed, (size_t)hidden_elems,
                                     nullptr);
        const TensorF32Stats q_a =
            collect_tensor_f32_stats(hc->d_q_a_full, (size_t)q_a_elems,
                                     nullptr);
        const TensorF32Stats q_a_normed =
            collect_tensor_f32_stats(hc->d_q_a_normed, (size_t)q_a_elems,
                                     nullptr);
        const TensorF32Stats kv =
            collect_tensor_f32_stats(hc->d_kv_full, (size_t)kv_elems,
                                     nullptr);
        const TensorF32Stats kv_normed =
            collect_tensor_f32_stats(hc->d_kv_normed, (size_t)kv_elems,
                                     nullptr);
        TensorF32Stats q_b;
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            const TensorF32Stats shard = collect_tensor_f32_stats(
                ops->attn_q_b.d_out[(size_t)rank],
                (size_t)opt.slots * ops->attn_q_b.rows_per_gpu,
                ranks[rank].dense_stream ? ranks[rank].dense_stream
                                         : ranks[rank].stream);
            merge_tensor_stats(&q_b, shard);
        }
        std::printf("tp_ep_true_attention_saturation_projection\tlayer\t%d\t"
                    "slots\t%d\tcurrent_max\t%.9g\tcurrent_bad\t%d\t"
                    "attn_normed_max\t%.9g\tattn_normed_bad\t%d\t"
                    "q_a_max\t%.9g\tq_a_bad\t%d\t"
                    "q_a_normed_max\t%.9g\tq_a_normed_bad\t%d\t"
                    "kv_max\t%.9g\tkv_bad\t%d\t"
                    "kv_normed_max\t%.9g\tkv_normed_bad\t%d\t"
                    "q_b_pre_head_max\t%.9g\tq_b_pre_head_bad\t%d\tPASS\n",
                    layer, opt.slots, current.max_abs, current.finite_bad,
                    attn_normed.max_abs, attn_normed.finite_bad, q_a.max_abs,
                    q_a.finite_bad, q_a_normed.max_abs, q_a_normed.finite_bad,
                    kv.max_abs, kv.finite_bad, kv_normed.max_abs,
                    kv_normed.finite_bad, q_b.max_abs, q_b.finite_bad);
    }
    std::printf("tp_ep_true_attention_projection_prefix\tlayer\t%d\tslots\t%d\t"
                "q_a_cols\t1024\tkv_cols\t%d\tq_width\t32768\t"
                "direct_input_fill\t%d\trank_local_input\t%d\t"
                "rank_major_input\t%d\tcurrent_source\t%s\tms\t%.6f\tPASS\n",
                layer, opt.slots, kHeadDim, direct_input_fill ? 1 : 0,
                rank_local_input ? 1 : 0, rank_major_input ? 1 : 0,
                attention_current_full == hc->d_current_full ? "shared_hc" : "rank0",
                ms);
    return 0;
}
