int run_true_ds4_attention_typed_kv_history_load(const Options &opt,
                                                 SharedHcControls *hc,
                                                 RankState ranks[kGpus],
                                                 ds4_tp_runtime *rt,
                                                 int layer) {
    if (!opt.true_ds4_attention_typed_kv_history_gate) return 0;
    if (!rt || layer < 0 || layer >= 43) return 1;
    const int ratio = ds4_layer_ratio(layer);
    if (ratio == 0) return 0;

    const uint32_t visible_attn =
        std::min(ranks[0].attn_comp_rows_written_layers[layer],
                 (uint32_t)kBoundedCompRows);
    char err[512] = {0};
    int loaded_attn = 0;
    int reloaded_attn = 0;
    for (uint32_t row = 0; row < visible_attn; ++row) {
        const uint64_t pos = ranks[0].attn_comp_row_position_layers[layer][row];
        if (opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
            pos == opt.position) {
            loaded_attn++;
            continue;
        }
        if (ranks[0].attn_comp_row_loaded_layers[layer][row] &&
            ranks[0].attn_comp_row_loaded_position_layers[layer][row] == pos) {
            loaded_attn++;
            continue;
        }
        if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
            void *dst[kGpus] = {};
            void *streams[kGpus] = {};
            const void *positions[kGpus] = {};
            const size_t row_offset = (size_t)row * (size_t)kHeadDim;
            for (int rank = 0; rank < kGpus; ++rank) {
                dst[rank] = opt.decode_cudagraph_gate
                    ? ranks[rank].d_attn_comp_rows
                    : ranks[rank].d_attn_comp_rows + row_offset;
                streams[rank] = opt.decode_cudagraph_gate
                    ? (void *)ranks[rank].stream
                    : nullptr;
                positions[rank] = ranks[rank].d_decode_position;
            }
            const int load_rc = opt.decode_cudagraph_gate
                ? ds4_tp_runtime_kv_rows_load_f32_device_streams_at_history_row(
                      rt, layer, 0, (uint32_t)opt.slots,
                      DS4_V100_TP_KV_ROW_ATTN, row, dst,
                      (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                      (uint32_t)kBoundedCompRows, streams, positions, err,
                      sizeof(err))
                : ds4_tp_runtime_kv_rows_load_f32_device(
                      rt, layer, 0, (uint32_t)opt.slots, pos,
                      DS4_V100_TP_KV_ROW_ATTN, dst,
                      (uint64_t)kBoundedCompRows * (uint64_t)kHeadDim,
                      err, sizeof(err));
            if (load_rc != 0) {
                std::fprintf(stderr,
                             "tp_ep_true_attention_typed_kv_history_attn_load_failed\t"
                             "layer\t%d\trow\t%u\tmode\tbatched\tposition\t%llu\t%s\n",
                             layer, row, (unsigned long long)pos, err);
                return 2;
            }
        } else {
            for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                void *dst[kGpus] = {};
                const size_t row_offset =
                    ((size_t)slot * (size_t)kBoundedCompRows + (size_t)row) *
                    (size_t)kHeadDim;
                for (int rank = 0; rank < kGpus; ++rank) {
                    dst[rank] = ranks[rank].d_attn_comp_rows + row_offset;
                }
                if (ds4_tp_runtime_kv_row_load_f32_device(
                        rt, layer, slot, pos, DS4_V100_TP_KV_ROW_ATTN, dst, err,
                        sizeof(err)) != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_history_attn_load_failed\t"
                                 "layer\t%d\trow\t%u\tslot\t%u\tposition\t%llu\t%s\n",
                                 layer, row, slot, (unsigned long long)pos, err);
                    return 2;
                }
            }
        }
        loaded_attn++;
        reloaded_attn++;
        for (int rank = 0; rank < kGpus; ++rank) {
            ranks[rank].attn_comp_row_loaded_layers[layer][row] = true;
            ranks[rank].attn_comp_row_loaded_position_layers[layer][row] = pos;
        }
    }
    sync_typed_kv_boundary(opt, ranks);

    int loaded_indexer = 0;
    int reloaded_indexer = 0;
    if (opt.true_ds4_indexer_attention_gate && ratio == 4 && visible_attn > 0) {
        if (!hc || !hc->initialized || !hc->d_indexer_q_full ||
            !hc->d_indexer_w_full || !ranks[0].d_indexer_scores ||
            !ranks[0].d_indexer_topk) {
            return 3;
        }
        const uint32_t visible_index =
            std::min(ranks[0].index_comp_rows_written_layers[layer],
                     (uint32_t)kBoundedCompRows);
        for (uint32_t row = 0; row < visible_index; ++row) {
            const uint64_t pos = ranks[0].index_comp_row_position_layers[layer][row];
            if (opt.true_ds4_attention_typed_kv_skip_current_load_gate &&
                pos == opt.position) {
                loaded_indexer++;
                continue;
            }
            if (ranks[0].index_comp_row_loaded_layers[layer][row] &&
                ranks[0].index_comp_row_loaded_position_layers[layer][row] == pos) {
                loaded_indexer++;
                continue;
            }
            if (opt.true_ds4_attention_typed_kv_batch_rows_gate) {
                void *dst[kGpus] = {};
                void *streams[kGpus] = {};
                const void *positions[kGpus] = {};
                const size_t row_offset = (size_t)row * (size_t)kIndexerHeadDim;
                for (int rank = 0; rank < kGpus; ++rank) {
                    dst[rank] = opt.decode_cudagraph_gate
                        ? ranks[rank].d_index_comp_rows
                        : ranks[rank].d_index_comp_rows + row_offset;
                    streams[rank] = opt.decode_cudagraph_gate
                        ? (void *)ranks[rank].stream
                        : nullptr;
                    positions[rank] = ranks[rank].d_decode_position;
                }
                const int load_rc = opt.decode_cudagraph_gate
                    ? ds4_tp_runtime_kv_rows_load_f32_device_streams_at_history_row(
                          rt, layer, 0, (uint32_t)opt.slots,
                          DS4_V100_TP_KV_ROW_INDEXER, row, dst,
                          (uint64_t)kBoundedCompRows *
                              (uint64_t)kIndexerHeadDim,
                          (uint32_t)kBoundedCompRows, streams, positions, err,
                          sizeof(err))
                    : ds4_tp_runtime_kv_rows_load_f32_device(
                          rt, layer, 0, (uint32_t)opt.slots, pos,
                          DS4_V100_TP_KV_ROW_INDEXER, dst,
                          (uint64_t)kBoundedCompRows * (uint64_t)kIndexerHeadDim,
                          err, sizeof(err));
                if (load_rc != 0) {
                    std::fprintf(stderr,
                                 "tp_ep_true_attention_typed_kv_history_indexer_load_failed\t"
                                 "layer\t%d\trow\t%u\tmode\tbatched\tposition\t%llu\t%s\n",
                                 layer, row, (unsigned long long)pos, err);
                    return 4;
                }
            } else {
                for (uint32_t slot = 0; slot < (uint32_t)opt.slots; ++slot) {
                    void *dst[kGpus] = {};
                    const size_t row_offset =
                        ((size_t)slot * (size_t)kBoundedCompRows + (size_t)row) *
                        (size_t)kIndexerHeadDim;
                    for (int rank = 0; rank < kGpus; ++rank) {
                        dst[rank] = ranks[rank].d_index_comp_rows + row_offset;
                    }
                    if (ds4_tp_runtime_kv_row_load_f32_device(
                            rt, layer, slot, pos, DS4_V100_TP_KV_ROW_INDEXER, dst,
                            err, sizeof(err)) != 0) {
                        std::fprintf(stderr,
                                     "tp_ep_true_attention_typed_kv_history_indexer_load_failed\t"
                                     "layer\t%d\trow\t%u\tslot\t%u\tposition\t%llu\t%s\n",
                                     layer, row, slot, (unsigned long long)pos, err);
                        return 4;
                    }
                }
            }
            loaded_indexer++;
            reloaded_indexer++;
            for (int rank = 0; rank < kGpus; ++rank) {
                ranks[rank].index_comp_row_loaded_layers[layer][row] = true;
                ranks[rank].index_comp_row_loaded_position_layers[layer][row] = pos;
            }
        }
        sync_typed_kv_boundary(opt, ranks);
        CHECK_CUDA(cudaSetDevice(ranks[0].device));
        indexer_score_bounded_rows_slots_kernel<<<
            (unsigned int)opt.slots, 256, 0, ranks[0].stream>>>(
            ranks[0].d_indexer_scores, ranks[0].d_indexer_topk,
            hc->d_indexer_q_full, hc->d_indexer_w_full,
            ranks[0].d_index_comp_rows, (uint32_t)opt.slots, visible_index,
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
        if (opt.decode_cudagraph_gate) {
            const int slot = next_graph_order_event_slot(ranks);
            CHECK_CUDA(cudaSetDevice(ranks[0].device));
            cudaEvent_t ev = graph_stream_done_event(ranks[0], slot);
            if (!ev) return 5;
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
        const size_t topk_bytes = (size_t)opt.slots * kIndexerTopK * sizeof(uint32_t);
        const uint64_t topk_elems = (uint64_t)opt.slots * (uint64_t)kIndexerTopK;
        const int block = 256;
        if (!opt.decode_cudagraph_gate) {
            void *topk_dsts[kGpus] = {};
            for (int rank = 0; rank < kGpus; ++rank) {
                topk_dsts[rank] = ranks[rank].d_indexer_topk;
            }
            if (nccl_broadcast_bytes_from_rank0(
                    ranks, ranks[0].d_indexer_topk, topk_dsts, topk_bytes,
                    "indexer_topk_history") != 0) {
                return 9;
            }
        }
        for (int rank = 1; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            if (opt.decode_cudagraph_gate) {
                copy_u32_kernel<<<(unsigned int)((topk_elems + block - 1) / block),
                                  block, 0, ranks[rank].stream>>>(
                    ranks[rank].d_indexer_topk, ranks[0].d_indexer_topk,
                    topk_elems);
                CHECK_CUDA(cudaGetLastError());
            }
        }
        if (!opt.decode_cudagraph_gate) {
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
            }
        }
    }
    sync_typed_kv_boundary(opt, ranks);

    if (!opt.true_ds4_attention_typed_kv_quiet_gate) {
        std::printf("tp_ep_true_attention_typed_kv_history\tlayer\t%d\tslots\t%d\t"
                    "ratio\t%d\tvisible_attn_rows\t%u\tloaded_attn_rows\t%d\t"
                    "loaded_indexer_rows\t%d\treloaded_attn_rows\t%d\t"
                    "reloaded_indexer_rows\t%d\tPASS\n",
                    layer, opt.slots, ratio, visible_attn, loaded_attn,
                    loaded_indexer, reloaded_attn, reloaded_indexer);
    }
    return 0;
}

int run_true_ds4_attention_raw_read(const Options &opt,
                                    SharedHcControls *hc,
                                    const LayerDenseOps *ops,
                                    RankState ranks[kGpus],
                                    int layer) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    if (!hc->d_attn_sinks[layer] ||
        ops->attn_q_b.rows_per_gpu != kLocalHeads * kHeadDim) {
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    const uint32_t raw_row = (uint32_t)(opt.position % kRawSwaRows);
    const uint64_t heads_elems =
        (uint64_t)opt.slots * (uint64_t)kLocalHeads * (uint64_t)kHeadDim;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_raw_swa || !r.d_attn_sinks || !r.d_attn_heads ||
            !ops->attn_q_b.d_out[(size_t)rank]) {
            return 3;
        }
        const size_t sinks_offset = (size_t)rank * (size_t)kLocalHeads;
        if (rank == 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_attn_sinks,
                                       hc->d_attn_sinks[layer] + sinks_offset,
                                       (size_t)kLocalHeads * sizeof(float),
                                       cudaMemcpyDeviceToDevice, r.stream));
        } else {
            void *sinks_dsts[kGpus] = {};
            for (int dst_rank = 0; dst_rank < kGpus; ++dst_rank) {
                sinks_dsts[dst_rank] = ranks[dst_rank].d_attn_sinks;
            }
            if (nccl_broadcast_bytes_from_rank0(
                    ranks, hc->d_attn_sinks[layer] + sinks_offset, sinks_dsts,
                    (size_t)kLocalHeads * sizeof(float),
                    "attention_raw_sinks") != 0) {
                return 4;
            }
        }
        attention_raw_swa_one_row_kernel<<<
            (unsigned int)(opt.slots * kLocalHeads), 256, 0, r.stream>>>(
            r.d_attn_heads, ops->attn_q_b.d_out[(size_t)rank], r.d_attn_raw_swa,
            r.d_attn_sinks, (uint32_t)opt.slots, (uint32_t)kLocalHeads,
            (uint32_t)kHeadDim, (uint32_t)kRawSwaRows, r.d_decode_position);
        CHECK_CUDA(cudaGetLastError());
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    if (!opt.decode_cudagraph_gate && layer <= 2) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("true_attn_raw_read_heads", layer, rank,
                                 ranks[rank].d_attn_heads, (size_t)heads_elems,
                                 ranks[rank].stream);
        }
    }
    std::printf("tp_ep_true_attention_raw_read\tlayer\t%d\tslots\t%d\t"
                "local_heads\t%d\thead_dim\t%d\traw_rows\t%d\traw_row\t%u\t"
                "ms\t%.6f\tPASS\n",
                layer, opt.slots, kLocalHeads, kHeadDim, kRawSwaRows, raw_row, ms);
    return 0;
}

int run_true_ds4_attention_raw_window(const Options &opt,
                                      SharedHcControls *hc,
                                      const LayerDenseOps *ops,
                                      RankState ranks[kGpus],
                                      int layer) {
    if (!hc || !hc->initialized || !ops || !ops->initialized ||
        layer < 0 || layer >= 43) {
        return 1;
    }
    if (!hc->d_attn_sinks[layer] ||
        ops->attn_q_b.rows_per_gpu != kLocalHeads * kHeadDim) {
        return 2;
    }
    const uint32_t valid_rows =
        std::max(1u, std::min(opt.true_ds4_attention_raw_valid_rows,
                              (uint32_t)kRawSwaRows));
    const int ratio = ds4_layer_ratio(layer);
    const auto start = std::chrono::steady_clock::now();
    const uint32_t raw_row = (uint32_t)(opt.position % kRawSwaRows);
    const uint64_t heads_elems =
        (uint64_t)opt.slots * (uint64_t)kLocalHeads * (uint64_t)kHeadDim;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_attn_raw_swa || !r.d_attn_sinks || !r.d_attn_heads ||
            !ops->attn_q_b.d_out[(size_t)rank]) {
            return 3;
        }
        const size_t sinks_offset = (size_t)rank * (size_t)kLocalHeads;
        if (graph_event_order) {
            enqueue_graph_f32_copy_from_device0(
                opt, r, rank, r.d_attn_sinks,
                hc->d_attn_sinks[layer] + sinks_offset, (uint64_t)kLocalHeads,
                r.stream, 32);
        } else if (rank == 0) {
            CHECK_CUDA(cudaMemcpyAsync(r.d_attn_sinks,
                                       hc->d_attn_sinks[layer] + sinks_offset,
                                       (size_t)kLocalHeads * sizeof(float),
                                       cudaMemcpyDeviceToDevice, r.stream));
        } else {
            void *sinks_dsts[kGpus] = {};
            for (int dst_rank = 0; dst_rank < kGpus; ++dst_rank) {
                sinks_dsts[dst_rank] = ranks[dst_rank].d_attn_sinks;
            }
            if (nccl_broadcast_bytes_from_rank0(
                    ranks, hc->d_attn_sinks[layer] + sinks_offset, sinks_dsts,
                    (size_t)kLocalHeads * sizeof(float),
                    "attention_raw_window_sinks") != 0) {
                return 4;
            }
        }
        const uint32_t visible_comp_rows =
            opt.true_ds4_compressed_kv_gate && ratio != 0
                ? std::min(r.attn_comp_rows_written_layers[layer],
                           (uint32_t)kBoundedCompRows)
                : 0u;
        const uint32_t selected_comp_rows =
            visible_comp_rows == 0u
                ? 0u
                : (ratio == 4 && opt.true_ds4_indexer_attention_gate
                       ? std::min(visible_comp_rows, (uint32_t)kBoundedCompRows)
                       : visible_comp_rows);
        if (selected_comp_rows > 0u) {
            if (!r.d_attn_comp_rows ||
                (ratio == 4 && opt.true_ds4_indexer_attention_gate &&
                 !r.d_indexer_topk)) {
                return 4;
            }
            attention_raw_compressed_window_kernel<<<
                (unsigned int)(opt.slots * kLocalHeads), 256, 0, r.stream>>>(
                r.d_attn_heads, ops->attn_q_b.d_out[(size_t)rank],
                r.d_attn_raw_swa, r.d_attn_comp_rows,
                ratio == 4 && opt.true_ds4_indexer_attention_gate
                    ? r.d_indexer_topk
                    : nullptr,
                r.d_attn_sinks, (uint32_t)opt.slots, (uint32_t)kLocalHeads,
                (uint32_t)kHeadDim, (uint32_t)kRawSwaRows, r.d_decode_position,
                valid_rows, visible_comp_rows, selected_comp_rows,
                (uint32_t)kBoundedCompRows, (uint32_t)kIndexerTopK);
        } else {
            attention_raw_swa_window_kernel<<<
                (unsigned int)(opt.slots * kLocalHeads), 256, 0, r.stream>>>(
                r.d_attn_heads, ops->attn_q_b.d_out[(size_t)rank],
                r.d_attn_raw_swa, r.d_attn_sinks, (uint32_t)opt.slots,
                (uint32_t)kLocalHeads, (uint32_t)kHeadDim,
                (uint32_t)kRawSwaRows, r.d_decode_position, valid_rows);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    if (!opt.decode_cudagraph_gate && layer <= 2) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            log_tensor_f32_stats("true_attn_raw_window_heads", layer, rank,
                                 ranks[rank].d_attn_heads, (size_t)heads_elems,
                                 ranks[rank].stream);
        }
    }
    std::printf("tp_ep_true_attention_raw_window\tlayer\t%d\tslots\t%d\t"
                "local_heads\t%d\thead_dim\t%d\traw_rows\t%d\traw_row\t%u\t"
                "valid_rows\t%u\tvisible_compressed_rows\t%u\t"
                "selected_compressed_rows\t%u\tms\t%.6f\tPASS\n",
                layer, opt.slots, kLocalHeads, kHeadDim, kRawSwaRows, raw_row,
                valid_rows,
                opt.true_ds4_compressed_kv_gate && ratio != 0
                    ? std::min(ranks[0].attn_comp_rows_written_layers[layer],
                               (uint32_t)kBoundedCompRows)
                    : 0u,
                opt.true_ds4_compressed_kv_gate && ratio != 0
                    ? std::min(ranks[0].attn_comp_rows_written_layers[layer],
                               (uint32_t)kBoundedCompRows)
                    : 0u,
                ms);
    return 0;
}
