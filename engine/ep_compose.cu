int run_next_hidden_compose(const Options &opt,
                            const std::vector<ContractRow> &rows,
                            RankState ranks[kGpus],
                            ComposeStats *stats) {
    if (!opt.compose_next_hidden) return 0;
    stats->enabled = true;
    stats->ep_return_fp16 = opt.ep_return_fp16;
    stats->fused_compose_sum =
        opt.fuse_compose_sum && !opt.ep_return_fp16 && !opt.compact_route_compose;
    stats->dense_hmma_compose = opt.dense_hmma_compose;
    stats->dense_f16_cublas_compose = opt.dense_f16_cublas_compose;
    stats->nccl_reduce_scatter_compose =
        opt.nccl_reduce_scatter_compose_gate &&
        !opt.compact_route_compose && !opt.ep_return_fp16;

    DeviceDenseOutputs attn;
    DeviceDenseOutputs shared;
    const std::string attn_tensor = layer_tensor_name(opt.layer, "attn_output_b.weight");
    const std::string shared_tensor = layer_tensor_name(opt.layer, "ffn_down_shexp.weight");
    if (run_f8_dense_to_device(opt, rows, attn_tensor.c_str(), 1, &attn) != 0 ||
        run_f8_dense_to_device(opt, rows, shared_tensor.c_str(), 2, &shared) != 0) {
        free_device_dense_outputs(attn, opt);
        free_device_dense_outputs(shared, opt);
        return 1;
    }
    stats->attn_dense_ms = attn.compute_ms;
    stats->shared_dense_ms = shared.compute_ms;

    const uint64_t shard_elems = (uint64_t)opt.slots * (kHidden / kGpus);
    const uint64_t shard_bytes = shard_elems * sizeof(float);
    const uint64_t return_shard_bytes =
        shard_elems * (opt.ep_return_fp16 ? sizeof(__half) : sizeof(float));
    const uint64_t all_contrib_elems = (uint64_t)kGpus * shard_elems;
    const uint64_t all_contrib_bytes = all_contrib_elems * sizeof(float);
    const bool skip_self_copy = opt.skip_self_compose_copy && !opt.ep_return_fp16;
    const bool nccl_reduce_scatter = stats->nccl_reduce_scatter_compose;
    stats->ep_contribution_bytes = all_contrib_bytes * kGpus;
    if (opt.compact_route_compose && !opt.ep_return_fp16) {
        uint64_t compact_return_bytes = 0;
        for (int src = 0; src < kGpus; ++src) {
            const int src_compose_routes = routed_compose_rows(ranks[src], opt);
            compact_return_bytes +=
                (uint64_t)src_compose_routes * (kHidden / kGpus) * sizeof(float) *
                (skip_self_copy ? (kGpus - 1) : kGpus);
        }
        stats->ep_return_bytes = compact_return_bytes;
    } else {
        stats->ep_return_bytes = return_shard_bytes *
                                 (skip_self_copy ? (kGpus * kGpus - kGpus)
                                                 : (kGpus * kGpus));
    }
    if (ensure_compose_buffers(opt, ranks) != 0) {
        free_device_dense_outputs(attn, opt);
        free_device_dense_outputs(shared, opt);
        return 2;
    }

    const auto compose_start = std::chrono::steady_clock::now();

    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        const int block = 256;
        int grid = (int)((all_contrib_elems + block - 1) / block);
        zero_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_contrib_all,
                                                      all_contrib_elems);
        CHECK_CUDA(cudaGetLastError());
        const uint64_t route_hidden_elems = (uint64_t)r.routes * kHidden;
        grid = (int)((route_hidden_elems + block - 1) / block);
        if (route_hidden_elems > 0) {
            ep_reduce_all_dest_shards_kernel<<<grid, block, 0, r.stream>>>(
                r.d_ep_contrib_all, r.d_down, r.d_route_slots, r.d_route_weights,
                nullptr, r.routes, opt.slots, p);
            CHECK_CUDA(cudaGetLastError());
        }
        if (opt.ep_return_fp16) {
            grid = (int)((all_contrib_elems + block - 1) / block);
            cast_f32_to_half_kernel<<<grid, block, 0, r.stream>>>(
                r.d_ep_contrib_half_all, r.d_ep_contrib_all, all_contrib_elems);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    for (int p = 0; p < kGpus; ++p) {
        CHECK_CUDA(cudaSetDevice(ranks[p].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[p].stream));
    }

    if (nccl_reduce_scatter) {
        for (int p = 0; p < kGpus; ++p) {
            if (!ranks[p].compose_nccl_initialized || !ranks[p].compose_nccl) {
                return 3;
            }
        }
        CHECK_NCCL(ncclGroupStart());
        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_NCCL(ncclReduceScatter(ranks[p].d_ep_contrib_all,
                                         ranks[p].d_ep_sum,
                                         (size_t)shard_elems,
                                         ncclFloat,
                                         ncclSum,
                                         ds4_comm_epret(ranks[p]),
                                         ranks[p].stream));
        }
        CHECK_NCCL(ncclGroupEnd());
    } else {
        uint64_t copy_elems_by_src[kGpus] = {};
        for (int src = 0; src < kGpus; ++src) {
            copy_elems_by_src[src] = shard_elems;
        }
        if (broadcast_ep_return_slices(
                ranks, opt.ep_return_fp16, skip_self_copy, shard_elems,
                copy_elems_by_src,
                opt.ep_return_fp16 ? "ep_compose_half_bcast"
                                   : "ep_compose_float_bcast") != 0) {
            return 4;
        }
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        CHECK_CUDA(cudaSetDevice(ranks[dst].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[dst].stream));
    }

    std::vector<std::vector<float>> first((size_t)kGpus);
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            const int block = 256;
            int grid = (int)((shard_elems + block - 1) / block);
            if (nccl_reduce_scatter) {
                compose_next_hidden_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn.d_out[(size_t)dst],
                    shared.d_out[(size_t)dst], r.d_ep_sum, dst, opt.slots);
            } else if (stats->fused_compose_sum) {
                const float *r0 = skip_self_copy && dst == 0
                    ? ranks[0].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[0];
                const float *r1 = skip_self_copy && dst == 1
                    ? ranks[1].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[1];
                const float *r2 = skip_self_copy && dst == 2
                    ? ranks[2].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[2];
                const float *r3 = skip_self_copy && dst == 3
                    ? ranks[3].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[3];
                const float *r4 = skip_self_copy && dst == 4
                    ? ranks[4].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[4];
                const float *r5 = skip_self_copy && dst == 5
                    ? ranks[5].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[5];
                const float *r6 = skip_self_copy && dst == 6
                    ? ranks[6].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[6];
                const float *r7 = skip_self_copy && dst == 7
                    ? ranks[7].d_ep_contrib_all + (uint64_t)dst * shard_elems
                    : r.d_ep_remote[7];
                compose_next_hidden_sum8_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn.d_out[(size_t)dst],
                    shared.d_out[(size_t)dst], r0, r1, r2, r3, r4, r5, r6, r7,
                    dst, opt.slots);
            } else {
                zero_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_sum, shard_elems);
                CHECK_CUDA(cudaGetLastError());
                for (int src = 0; src < kGpus; ++src) {
                    if (opt.ep_return_fp16) {
                        add_half_to_f32_kernel<<<grid, block, 0, r.stream>>>(
                            r.d_ep_sum, r.d_ep_remote_half[src], shard_elems);
                    } else {
                        const float *src_contrib = skip_self_copy && src == dst
                            ? ranks[src].d_ep_contrib_all + (uint64_t)dst * shard_elems
                            : r.d_ep_remote[src];
                        add_f32_kernel<<<grid, block, 0, r.stream>>>(r.d_ep_sum,
                                                                     src_contrib,
                                                                     shard_elems);
                    }
                }
                CHECK_CUDA(cudaGetLastError());
                compose_next_hidden_kernel<<<grid, block, 0, r.stream>>>(
                    r.d_next_hidden, r.d_current_shard, attn.d_out[(size_t)dst],
                    shared.d_out[(size_t)dst], r.d_ep_sum, dst, opt.slots);
            }
            CHECK_CUDA(cudaGetLastError());
        }
        for (int dst = 0; dst < kGpus; ++dst) {
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            CHECK_CUDA(cudaStreamSynchronize(r.stream));
            std::vector<float> host((size_t)shard_elems);
            CHECK_CUDA(cudaMemcpy(host.data(), r.d_next_hidden, (size_t)shard_bytes,
                                  cudaMemcpyDeviceToHost));
            if (repeat == 0) {
                first[(size_t)dst] = host;
                for (uint64_t i = 0; i < shard_elems; ++i) {
                    if (!std::isfinite(host[(size_t)i])) {
                        stats->finite_bad++;
                        stats->pass = false;
                    }
                    uint32_t bits = 0;
                    std::memcpy(&bits, &host[(size_t)i], sizeof(bits));
                    stats->checksum ^=
                        (uint64_t)bits + (uint64_t)(dst + 1) * 1000003ull + i * 9176ull;
                }
            } else {
                for (uint64_t i = 0; i < shard_elems; ++i) {
                    const double diff =
                        std::fabs((double)host[(size_t)i] -
                                  (double)first[(size_t)dst][(size_t)i]);
                    stats->repeat_max_abs = std::max(stats->repeat_max_abs, diff);
                    if (diff > 0.0) {
                        stats->repeat_bad++;
                        stats->pass = false;
                    }
                }
            }
        }
    }

    const auto compose_stop = std::chrono::steady_clock::now();
    stats->compose_ms =
        std::chrono::duration<double, std::milli>(compose_stop - compose_start).count();
    if (stats->checksum == 0 || stats->finite_bad != 0 || stats->repeat_bad != 0) {
        stats->pass = false;
    }

    free_device_dense_outputs(attn, opt);
    free_device_dense_outputs(shared, opt);
    return stats->pass ? 0 : 2;
}

