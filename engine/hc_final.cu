int run_shared_hc_final_expand(const Options &opt,
                               SharedHcControls *hc,
                               RankState ranks[kGpus],
                               int layer) {
    if (!hc || !hc->initialized || hc->slots != opt.slots ||
        layer < 0 || layer >= 44) {
        return 1;
    }
    const uint64_t shard_elems =
        (uint64_t)opt.slots * (uint64_t)(kHidden / kGpus);
    const uint64_t hc_shard_elems = shard_elems * kHcRows;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    cudaStream_t control_stream = graph_event_order ? ranks[0].stream : (cudaStream_t)0;
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
    if (graph_event_order) {
        if (control_wait_on_rank_streams() != 0) return 4;
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        if (!ranks[rank].d_final_hc_shard) return 2;
        gather_hc_shard_to_full_kernel<<<
            (unsigned int)((hc_shard_elems + 255) / 256), 256, 0,
            control_stream>>>(
            hc->d_hc, ranks[rank].d_final_hc_shard, rank, (uint32_t)opt.slots);
    }
    CHECK_CUDA(cudaGetLastError());
    if (!graph_event_order) {
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    rms_norm_plain_rows_stable_kernel<<<
        (unsigned int)opt.slots, 256, 0, control_stream>>>(
        hc->d_hc_norm, hc->d_hc, kHcRows * (uint32_t)kHidden,
        (uint32_t)opt.slots, 1.0e-6f);
    const dim3 mix_grid((unsigned int)kHcMix, (unsigned int)opt.slots, 1u);
    f32_dense_colmajor_kernel<<<mix_grid, 256, 0, control_stream>>>(
        hc->d_mix, hc->d_ffn_fn[layer], hc->d_hc_norm,
        (uint32_t)kHcMix, kHcRows * (uint32_t)kHidden, (uint32_t)opt.slots);
    hc_split_rows_kernel<<<
        (unsigned int)(((uint64_t)opt.slots + 255) / 256), 256, 0,
        control_stream>>>(
        hc->d_split, hc->d_mix, hc->d_ffn_scale[layer], hc->d_ffn_base[layer],
        (uint32_t)opt.slots, opt.reference_hc_reduce_gate ? 20u : 4u);
    CHECK_CUDA(cudaGetLastError());
    if (graph_event_order) {
        if (rank_streams_wait_on_control() != 0) return 5;
    } else {
        CHECK_CUDA(cudaDeviceSynchronize());
        void *dsts[kGpus] = {};
        for (int rank = 0; rank < kGpus; ++rank) {
            dsts[rank] = ranks[rank].d_hc_split;
        }
        if (nccl_broadcast_bytes_from_rank0(
                ranks, hc->d_split, dsts,
                (size_t)opt.slots * kHcMix * sizeof(float),
                "hc_final_split") != 0) {
            return 6;
        }
    }

    const int block = 256;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_hc_scratch_shard || !r.d_hc_split) return 3;
        if (graph_event_order) {
            enqueue_graph_f32_copy_from_device0(
                opt, r, rank, r.d_hc_split, hc->d_split,
                (uint64_t)opt.slots * kHcMix, r.stream, block);
        }
        const int grid = (int)((hc_shard_elems + block - 1) / block);
        hc_expand_shard_kernel<<<grid, block, 0, r.stream>>>(
            r.d_hc_scratch_shard, r.d_next_hidden, r.d_final_hc_shard,
            r.d_hc_split, (uint32_t)opt.slots);
        if (opt.reference_hc_state_guard_gate) {
            clamp_f32_abs_kernel<<<grid, block, 0, r.stream>>>(
                r.d_hc_scratch_shard, hc_shard_elems,
                kReferenceHcStateTargetAbs);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    if (graph_event_order) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            std::swap(r.d_final_hc_shard, r.d_hc_scratch_shard);
            r.hc_initialized = true;
        }
    } else {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_CUDA(cudaStreamSynchronize(r.stream));
            std::swap(r.d_final_hc_shard, r.d_hc_scratch_shard);
            r.hc_initialized = true;
        }
    }
    return 0;
}

