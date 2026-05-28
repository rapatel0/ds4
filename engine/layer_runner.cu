int run_layer(const Options &opt,
              LayerRunSummary *summary,
              const DenseF16Cache *shared_dense_f16_cache,
              const SharedApi *shared_api,
              SharedRankBuffers *shared_rank_buffers,
              SharedTpRuntime *shared_tp_runtime,
              const SharedExpertBindings *shared_expert_bindings,
              const SharedDenseOps *shared_dense_ops,
              SharedHcControls *shared_hc_controls) {
    std::vector<ContractRow> rows;
    LayerStats layer_stats;
    if (parse_contract(opt.contract_path, opt.layer, &rows, &layer_stats) != 0 ||
        layer_stats.bad_rows != 0) {
        std::fprintf(stderr, "contract parse failed bad_rows=%llu\n",
                     (unsigned long long)layer_stats.bad_rows);
        return 2;
    }
    DescriptorBindings bindings;
    const LayerExpertCache *layer_expert_cache = nullptr;
    if (shared_expert_bindings) {
        layer_expert_cache = &shared_expert_bindings->layers[opt.layer];
        bindings = layer_expert_cache->bindings;
    } else {
        if (parse_tm_index(opt.tm_index_path, opt.layer, &bindings) != 0) {
            std::fprintf(stderr, "tm index parse failed for layer %d\n", opt.layer);
            return 2;
        }
    }

    const auto descriptor_start = std::chrono::steady_clock::now();
    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" && r.record_type != "replicated_control") continue;
        if (!opt.skip_descriptor_checks) {
            uint64_t checksum = 0;
            if (device_checksum_row(opt.devices[r.owning_gpu], opt.pack_dir, r, &checksum) != 0) {
                return 3;
            }
            layer_stats.gpu[r.owning_gpu].checksum ^=
                checksum + (uint64_t)(r.owning_gpu + 1) * 131u;
            layer_stats.checksum ^= checksum + (uint64_t)(r.owning_gpu + 1) * 257u;
        }
        if (r.record_type == "dense_tp") layer_stats.dense_loaded_bytes += r.bytes_estimate;
        else layer_stats.control_loaded_bytes += r.bytes_estimate;
    }
    const auto descriptor_stop = std::chrono::steady_clock::now();
    const double descriptor_ms =
        std::chrono::duration<double, std::milli>(descriptor_stop - descriptor_start).count();

    DenseComputeStats dense_compute;
    DenseComputeStats bf16_compute;
    std::vector<DenseComputeStats> dense_compute_results;
    std::vector<DenseComputeStats> bf16_compute_results;
    std::vector<std::string> dense_tensors;
    if (opt.dense_compute_all_f8) {
        dense_tensors = discover_f8_dense_tensors(rows);
    } else if (opt.dense_compute_tensor) {
        dense_tensors.emplace_back(opt.dense_compute_tensor);
    }
    for (const std::string &tensor : dense_tensors) {
        DenseComputeStats one;
        if (run_dense_compute_gate(opt, rows, tensor.c_str(), &one) != 0) {
            std::fprintf(stderr, "dense compute gate failed for %s\n", tensor.c_str());
            return 3;
        }
        std::printf("dense_compute_tensor\ttensor\t%s\trows_per_gpu\t%d\tcols\t%d\t"
                    "slots\t%d\tloaded_bytes\t%llu\tcompute_ms\t%.6f\t"
                    "repeat_max_abs\t%.9f\trepeat_bad\t%d\trepeat_nan\t%d\t"
                    "oracle_max_abs\t%.9f\toracle_bad\t%d\t%s\n",
                    one.tensor_id.c_str(), one.rows_per_gpu, one.cols, one.slots,
                    (unsigned long long)one.loaded_bytes, one.compute_ms,
                    one.repeat_max_abs, one.repeat_bad, one.repeat_nan,
                    one.oracle_max_abs, one.oracle_bad, one.pass ? "PASS" : "FAIL");
        dense_compute_results.push_back(one);
        dense_compute.enabled = true;
        dense_compute.tensor_id = opt.dense_compute_all_f8 ? "all_f8" : one.tensor_id;
        dense_compute.rows_per_gpu = std::max(dense_compute.rows_per_gpu, one.rows_per_gpu);
        dense_compute.cols = std::max(dense_compute.cols, one.cols);
        dense_compute.slots = one.slots;
        dense_compute.loaded_bytes += one.loaded_bytes;
        dense_compute.compute_ms = std::max(dense_compute.compute_ms, one.compute_ms);
        dense_compute.repeat_max_abs =
            std::max(dense_compute.repeat_max_abs, one.repeat_max_abs);
        dense_compute.oracle_max_abs =
            std::max(dense_compute.oracle_max_abs, one.oracle_max_abs);
        dense_compute.repeat_bad += one.repeat_bad;
        dense_compute.repeat_nan += one.repeat_nan;
        dense_compute.oracle_bad += one.oracle_bad;
        dense_compute.pass = dense_compute.pass && one.pass;
    }
    std::vector<std::string> bf16_tensors;
    if (opt.dense_compute_all_bf16) {
        bf16_tensors = discover_bf16_dense_tensors(rows);
    }
    for (const std::string &tensor : bf16_tensors) {
        DenseComputeStats one;
        if (run_bf16_dense_compute_gate(opt, rows, tensor.c_str(), &one) != 0) {
            std::fprintf(stderr, "bf16 dense compute gate failed for %s\n", tensor.c_str());
            return 3;
        }
        std::printf("bf16_dense_compute_tensor\ttensor\t%s\trows_per_gpu\t%d\tcols\t%d\t"
                    "slots\t%d\tloaded_bytes\t%llu\tcompute_ms\t%.6f\t"
                    "repeat_max_abs\t%.9f\trepeat_bad\t%d\trepeat_nan\t%d\t"
                    "oracle_max_abs\t%.9f\toracle_bad\t%d\t%s\n",
                    one.tensor_id.c_str(), one.rows_per_gpu, one.cols, one.slots,
                    (unsigned long long)one.loaded_bytes, one.compute_ms,
                    one.repeat_max_abs, one.repeat_bad, one.repeat_nan,
                    one.oracle_max_abs, one.oracle_bad, one.pass ? "PASS" : "FAIL");
        bf16_compute_results.push_back(one);
        bf16_compute.enabled = true;
        bf16_compute.tensor_id = "all_bf16";
        bf16_compute.rows_per_gpu = std::max(bf16_compute.rows_per_gpu, one.rows_per_gpu);
        bf16_compute.cols = std::max(bf16_compute.cols, one.cols);
        bf16_compute.slots = one.slots;
        bf16_compute.loaded_bytes += one.loaded_bytes;
        bf16_compute.compute_ms = std::max(bf16_compute.compute_ms, one.compute_ms);
        bf16_compute.repeat_max_abs =
            std::max(bf16_compute.repeat_max_abs, one.repeat_max_abs);
        bf16_compute.oracle_max_abs =
            std::max(bf16_compute.oracle_max_abs, one.oracle_max_abs);
        bf16_compute.repeat_bad += one.repeat_bad;
        bf16_compute.repeat_nan += one.repeat_nan;
        bf16_compute.oracle_bad += one.oracle_bad;
        bf16_compute.pass = bf16_compute.pass && one.pass;
    }

    DenseF16Cache local_dense_f16_cache;
    const DenseF16Cache *dense_f16_cache = shared_dense_f16_cache;
    if (!dense_f16_cache) {
        if (prepare_dense_f16_cache(opt, rows, &local_dense_f16_cache) != 0) {
            std::fprintf(stderr, "dense f16 cache prepare failed\n");
            return 4;
        }
        dense_f16_cache = &local_dense_f16_cache;
    }
    if (!shared_dense_f16_cache && dense_f16_cache->enabled) {
        std::printf("tp_ep_dense_f16_cache\tlayer\t%d\trows\t%llu\t"
                    "source_bytes\t%llu\tcache_bytes\t%llu\t"
                    "cache_aligned_bytes\t%llu\tmax_temp_bytes\t%llu\tPASS\n",
                    opt.layer,
                    (unsigned long long)dense_f16_cache->rows,
                    (unsigned long long)dense_f16_cache->source_bytes,
                    (unsigned long long)dense_f16_cache->cache_bytes,
                    (unsigned long long)dense_f16_cache->cache_aligned_bytes,
                    (unsigned long long)dense_f16_cache->max_temp_bytes);
    }

    ds4_tp_runtime_config cfg;
    fill_tp_runtime_config(opt, &cfg);

    char err[512] = {0};
    ds4_tp_runtime *rt = nullptr;
    ds4_tp_runtime_report runtime_report;
    if (shared_tp_runtime) {
        rt = shared_tp_runtime->rt;
        runtime_report = shared_tp_runtime->report;
    } else {
        if (ds4_tp_runtime_open(&rt, &cfg, err, sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_open_failed\t%s\n", err);
            return 4;
        }
        ds4_tp_runtime_get_report(rt, &runtime_report);
    }
    auto close_local_runtime = [&]() {
        if (!shared_tp_runtime && rt) ds4_tp_runtime_close(rt);
    };

    ds4_tp_dense_kv_result kv_result;
    const auto kv_start = std::chrono::steady_clock::now();
    const int write_indexer = ds4_layer_ratio(opt.layer) == 4 ? 1 : 0;
    if (ds4_tp_runtime_dense_kv_slice(rt, opt.layer, opt.kv_slot, opt.position,
                                           write_indexer, &kv_result, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "tp_runtime_dense_kv_slice_failed\t%s\n", err);
        close_local_runtime();
        return 5;
    }
    const auto kv_stop = std::chrono::steady_clock::now();
    const double dense_kv_ms =
        std::chrono::duration<double, std::milli>(kv_stop - kv_start).count();

    void *lib = nullptr;
    Api local_api;
    const Api *api = nullptr;
    if (shared_api) {
        api = &shared_api->api;
    } else {
        lib = dlopen(opt.lib_path, RTLD_LAZY | RTLD_LOCAL);
        if (!lib) {
            std::fprintf(stderr, "dlopen failed for %s: %s\n", opt.lib_path, dlerror());
            close_local_runtime();
            return 6;
        }
        load_api(lib, &local_api);
        api = &local_api;
    }

    RankState local_ranks[kGpus];
    RankState *ranks = shared_rank_buffers ? shared_rank_buffers->ranks : local_ranks;
    int aggregate_routes = 0;
    int min_routes = std::numeric_limits<int>::max();
    int max_routes = 0;
    uint64_t ep_loaded_bytes = 0;

    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        r.rank = p;
        r.device = opt.devices[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!shared_api && api->init(r.device) != 0) {
            std::fprintf(stderr, "ggml_turbomind_init failed on device %d\n", r.device);
            if (!shared_api) {
                api->shutdown();
                dlclose(lib);
            }
            close_local_runtime();
            return 7;
        }
        if (!shared_rank_buffers) {
            CHECK_CUDA(cudaStreamCreate(&r.stream));
            CHECK_CUDA(cudaStreamCreate(&r.dense_stream));
            CHECK_CUDA(cudaStreamCreate(&r.copy_stream));
            for (int q = 0; q < kGpus; ++q) {
                CHECK_CUDA(cudaStreamCreate(&r.copy_streams[q]));
                CHECK_CUDA(cudaEventCreateWithFlags(&r.copy_done[q], cudaEventDisableTiming));
            }
            CHECK_CUDA(cudaEventCreateWithFlags(&r.stream_done, cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&r.dense_done, cudaEventDisableTiming));
            for (int e = 0; e < kGraphOrderEventSlots; ++e) {
                CHECK_CUDA(cudaEventCreateWithFlags(&r.graph_stream_done[e],
                                                    cudaEventDisableTiming));
                CHECK_CUDA(cudaEventCreateWithFlags(&r.graph_dense_done[e],
                                                    cudaEventDisableTiming));
            }
            CHECK_CUDA(cudaEventCreateWithFlags(&r.dense_wait, cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreate(&r.start));
            CHECK_CUDA(cudaEventCreate(&r.mid));
            CHECK_CUDA(cudaEventCreate(&r.stop));
            r.route_compact_plan_ints = compact_route_plan_ints(opt);
            CHECK_CUDA(cudaMalloc(&r.d_route_compact_plan,
                                  r.route_compact_plan_ints * sizeof(int)));
            bind_compact_route_plan(&r, opt);
            CHECK_CUDA(cudaMalloc(&r.d_router_selected_plan,
                                  (size_t)opt.slots * (size_t)opt.top_k * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&r.d_router_weights_plan,
                                  (size_t)opt.slots * (size_t)opt.top_k * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&r.d_route_offsets_all,
                                  (size_t)kGpus * (size_t)(kLocalExperts + 1) *
                                      sizeof(int)));
            CHECK_CUDA(cudaMalloc(&r.d_route_totals,
                                  (size_t)kGpus * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&r.d_post_attn_route_audit,
                                  4u * sizeof(unsigned long long)));
            CHECK_CUDA(cudaMemset(r.d_post_attn_route_audit, 0,
                                  4u * sizeof(unsigned long long)));
            std::vector<int> compact_plan(r.route_compact_plan_ints, -1);
            const size_t compact_indices = (size_t)opt.slots * (size_t)opt.top_k;
            const size_t compact_counts = (size_t)opt.slots;
            for (int src = 0; src < kGpus; ++src) {
                std::vector<int> route_index_by_slot;
                build_route_index_by_slot_for_rank(src, opt.slots, opt.top_k,
                                                   &route_index_by_slot);
                CHECK_CUDA(cudaMalloc(&r.d_route_index_by_slot[src],
                                      route_index_by_slot.size() * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(r.d_route_index_by_slot[src],
                                      route_index_by_slot.data(),
                                      route_index_by_slot.size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
                std::vector<int> route_indices_by_slot;
                std::vector<int> route_count_by_slot;
                build_route_indices_by_slot_for_rank(src, opt.slots, opt.top_k,
                                                     &route_indices_by_slot,
                                                     &route_count_by_slot);
                std::copy(route_indices_by_slot.begin(), route_indices_by_slot.end(),
                          compact_plan.begin() + (size_t)src * compact_indices);
                std::copy(route_count_by_slot.begin(), route_count_by_slot.end(),
                          compact_plan.begin() + (size_t)kGpus * compact_indices +
                              (size_t)src * compact_counts);
            }
            CHECK_CUDA(cudaMemcpy(r.d_route_compact_plan, compact_plan.data(),
                                  compact_plan.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));

            std::vector<int> offsets;
            std::vector<int> route_slots;
            std::vector<float> route_weights;
            build_offsets_for_rank(p, opt.slots, opt.top_k, &offsets, &route_slots,
                                   &route_weights, &r.routes, &r.active_experts,
                                   &r.max_routes_per_expert);

            r.route_capacity = opt.slots * opt.top_k;
            const size_t route_capacity_elems = (size_t)r.route_capacity * kHidden;
            CHECK_CUDA(cudaMalloc(&r.d_offsets, offsets.size() * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(r.d_offsets, offsets.data(), offsets.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&r.d_route_slots,
                                  (size_t)r.route_capacity * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(r.d_route_slots, route_slots.data(),
                                  route_slots.size() * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&r.d_route_weights,
                                  (size_t)r.route_capacity * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(r.d_route_weights, route_weights.data(),
                                  route_weights.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&r.d_route_inv_scale,
                                  (size_t)r.route_capacity * sizeof(float)));
            std::vector<float> route_inv_scale((size_t)r.route_capacity, 1.0f);
            CHECK_CUDA(cudaMemcpy(r.d_route_inv_scale, route_inv_scale.data(),
                                  route_inv_scale.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&r.d_a, route_capacity_elems * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&r.d_gate_up,
                                  (size_t)r.route_capacity * kFusedN * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&r.d_gated,
                                  (size_t)r.route_capacity * kMid * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&r.d_down, route_capacity_elems * sizeof(__half)));

            std::mt19937 rng(0xE2350000u + (uint32_t)p * 97u);
            std::uniform_real_distribution<float> dist(-0.003f, 0.003f);
            std::vector<__half> h_a(route_capacity_elems);
            for (__half &v : h_a) v = __float2half(dist(rng));
            CHECK_CUDA(cudaMemcpy(r.d_a, h_a.data(),
                                  route_capacity_elems * sizeof(__half),
                                  cudaMemcpyHostToDevice));
        }
        aggregate_routes += r.routes;
        min_routes = std::min(min_routes, r.routes);
        max_routes = std::max(max_routes, r.routes);

        if (layer_expert_cache) {
            r.gated = layer_expert_cache->gated[p];
            r.down = layer_expert_cache->down[p];
            ep_loaded_bytes += layer_expert_cache->gated[p].d_w_active.size()
                ? layer_expert_cache->bytes / kGpus
                : 0;
        } else {
            std::vector<int> active;
            for (int e = 0; e < kPackedLocalExperts; ++e) active.push_back(e);
            if (pack_descriptor_set(r.device, bindings.gated, p, active, opt.pack_dir,
                                    &r.gated, &ep_loaded_bytes) != 0 ||
                pack_descriptor_set(r.device, bindings.down, p, active, opt.pack_dir,
                                   &r.down, &ep_loaded_bytes) != 0) {
                close_local_runtime();
                return 8;
            }
        }
        layer_stats.gpu[p].ep_loaded_bytes = ep_loaded_bytes;
    }
    layer_stats.ep_loaded_bytes = ep_loaded_bytes;

    if (!shared_rank_buffers && open_compose_nccl(opt, ranks) != 0) {
        close_local_runtime();
        return 8;
    }

    if (!opt.skip_predecode_probes) {
        for (int i = 0; i < opt.warmup; ++i) {
            for (int p = 0; p < kGpus; ++p) {
                const int gate_rc = run_gate_selected(ranks[p], *api, opt);
                if (gate_rc != 0 || run_down(ranks[p], *api, opt) != 0) {
                    close_local_runtime();
                    return 9;
                }
            }
            for (int p = 0; p < kGpus; ++p) {
                CHECK_CUDA(cudaSetDevice(ranks[p].device));
                CHECK_CUDA(cudaStreamSynchronize(ranks[p].stream));
            }
        }

        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_CUDA(cudaEventRecord(ranks[p].start, ranks[p].stream));
        }
        for (int i = 0; i < opt.iters; ++i) {
            for (int p = 0; p < kGpus; ++p) {
                const int gate_rc = run_gate_selected(ranks[p], *api, opt);
                if (gate_rc != 0) return 10;
            }
        }
        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_CUDA(cudaEventRecord(ranks[p].mid, ranks[p].stream));
        }
        for (int i = 0; i < opt.iters; ++i) {
            for (int p = 0; p < kGpus; ++p) {
                if (run_down(ranks[p], *api, opt) != 0) return 11;
            }
        }
        for (int p = 0; p < kGpus; ++p) {
            CHECK_CUDA(cudaSetDevice(ranks[p].device));
            CHECK_CUDA(cudaEventRecord(ranks[p].stop, ranks[p].stream));
        }
    }

    double worst_gate_ms = 0.0;
    double worst_down_ms = 0.0;
    double worst_ep_ms = 0.0;
    for (int p = 0; p < kGpus; ++p) {
        CHECK_CUDA(cudaSetDevice(ranks[p].device));
        double gate_ms = 0.0;
        double down_ms = 0.0;
        if (!opt.skip_predecode_probes) {
            CHECK_CUDA(cudaEventSynchronize(ranks[p].stop));
            gate_ms = (double)elapsed_ms(ranks[p].start, ranks[p].mid) / opt.iters;
            down_ms = (double)elapsed_ms(ranks[p].mid, ranks[p].stop) / opt.iters;
        }
        worst_gate_ms = std::max(worst_gate_ms, gate_ms);
        worst_down_ms = std::max(worst_down_ms, down_ms);
        worst_ep_ms = std::max(worst_ep_ms, gate_ms + down_ms);
        std::printf("rank\t%d\tdevice\t%d\troutes\t%d\troute_capacity\t%d\t"
                    "active_local_experts\t%d\t"
                    "max_routes_per_expert\t%d\tgate_ms\t%.6f\tdown_ms\t%.6f\t"
                    "ep_ms\t%.6f\tdense_rows\t%llu\tcontrol_rows\t%llu\t"
                    "expert_rows\t%llu\tkv_rows\t%llu\tcomp_rows\t%llu\t"
                    "checksum\t%llu\n",
                    p, ranks[p].device, ranks[p].routes, ranks[p].route_capacity,
                    ranks[p].active_experts,
                    ranks[p].max_routes_per_expert, gate_ms, down_ms, gate_ms + down_ms,
                    (unsigned long long)layer_stats.gpu[p].dense_rows,
                    (unsigned long long)layer_stats.gpu[p].control_rows,
                    (unsigned long long)layer_stats.gpu[p].expert_rows,
                    (unsigned long long)layer_stats.gpu[p].kv_rows,
                    (unsigned long long)layer_stats.gpu[p].comp_rows,
                    (unsigned long long)layer_stats.gpu[p].checksum);
    }

    double repeat_max_abs = 0.0;
    int repeat_bad = 0;
    int repeat_nan = 0;
    if (!opt.skip_predecode_probes) {
        for (int p = 0; p < kGpus; ++p) {
            if (check_repeat(ranks[p], *api, &repeat_max_abs, &repeat_bad, &repeat_nan) != 0) {
                close_local_runtime();
                return 12;
            }
        }
    }

    ComposeStats compose;
    const int compose_rc = run_next_hidden_compose(opt, rows, ranks, &compose);
    if (compose.enabled) {
        std::printf("tp_ep_next_hidden_compose\tslots\t%d\tctx\t%llu\t"
                    "hidden_shard\t%d\tep_contribution_bytes\t%llu\t"
                    "ep_return_dtype\t%s\tep_return_bytes\t%llu\tdense_hmma\t%d\t"
                    "dense_f16_cublas\t%d\t"
                    "attn_dense_ms\t%.6f\t"
                    "shared_dense_ms\t%.6f\tfused_compose_sum\t%d\t"
                    "nccl_reduce_scatter\t%d\tcompose_ms\t%.6f\t"
                    "checksum\t%llu\tfinite_bad\t%d\trepeat_max_abs\t%.9f\t"
                    "repeat_bad\t%d\t%s\n",
                    opt.slots, (unsigned long long)cfg.ctx, kHidden / kGpus,
                    (unsigned long long)compose.ep_contribution_bytes,
                    compose.ep_return_fp16 ? "fp16" : "fp32",
                    (unsigned long long)compose.ep_return_bytes,
                    compose.dense_hmma_compose ? 1 : 0,
                    compose.dense_f16_cublas_compose ? 1 : 0,
                    compose.attn_dense_ms, compose.shared_dense_ms,
                    compose.fused_compose_sum ? 1 : 0,
                    compose.nccl_reduce_scatter_compose ? 1 : 0,
                    compose.compose_ms, (unsigned long long)compose.checksum,
                    compose.finite_bad, compose.repeat_max_abs,
                    compose.repeat_bad, compose.pass ? "PASS" : "FAIL");
    }
    if (compose_rc != 0) {
        close_local_runtime();
        return 13;
    }

    DecodeLoopStats decode_loop;
    const LayerDenseOps *layer_dense_ops =
        shared_dense_ops && shared_dense_ops->initialized
            ? &shared_dense_ops->layers[opt.layer]
            : nullptr;
    const int decode_rc = run_decode_loop(opt, rows, ranks, *api, rt, dense_f16_cache,
                                          layer_dense_ops, shared_hc_controls,
                                          shared_rank_buffers
                                              ? &shared_rank_buffers->graph_cache.layers[opt.layer]
                                              : nullptr,
                                          &decode_loop);
    if (decode_loop.enabled) {
        std::printf("tp_ep_decode_loop\tsteps\t%d\tslots\t%d\tslot_steps\t%llu\t"
                    "total_ms\t%.6f\tms_per_step\t%.6f\tslot_step_tok_s\t%.6f\t"
                    "dense_hmma\t%d\tdense_f16_cublas\t%d\tdense_f16_cache\t%d\t"
                    "overlap_ep_dense\t%d\tdirect_remote_compose\t%d\t"
                    "source_copy_schedule\t%d\tskip_self_compose_copy\t%d\t"
                    "multi_copy_streams\t%d\t"
                    "decode_cudagraph_gate\t%d\t"
                    "decode_cudagraph_replay_probe_gate\t%d\t"
                    "ep_ms_per_step\t%.6f\tdense_ms_per_step\t%.6f\t"
                    "fused_compose_sum\t%d\tnccl_reduce_scatter\t%d\t"
                    "compose_ms_per_step\t%.6f\t"
                    "compose_reduce_ms_per_step\t%.6f\t"
                    "compose_copy_ms_per_step\t%.6f\t"
                    "compose_final_ms_per_step\t%.6f\t"
                    "hc_current_input_gate\t%d\t"
                    "hc_current_input_peer_gather\t%d\t"
                    "hc_current_input_nccl_allgather\t%d\t"
                    "hc_current_allreduce\t%d\t"
                    "hc_current_input_stream_sync\t%d\t"
                    "hc_current_input_ms_per_step\t%.6f\t"
                    "final_hc_carry_gate\t%d\tfinal_hc_ms_per_step\t%.6f\t"
                    "dense_loaded_bytes\t%llu\t"
                    "ep_contribution_bytes\t%llu\tep_return_dtype\t%s\t"
                    "ep_return_bytes\t%llu\t"
                    "cudagraph_replay_attempted\t%d\t"
                    "cudagraph_replay_succeeded\t%d\t"
                    "cudagraph_instantiate_ms\t%.6f\t"
                    "cudagraph_replay_ms\t%.6f\t"
                    "checksum\t%llu\tfinite_bad\t%d\t%s\n",
                    decode_loop.steps, decode_loop.slots,
                    (unsigned long long)decode_loop.slot_steps,
                    decode_loop.total_ms, decode_loop.ms_per_step,
                    decode_loop.tok_s,
                    decode_loop.dense_hmma_compose ? 1 : 0,
                    decode_loop.dense_f16_cublas_compose ? 1 : 0,
                    decode_loop.dense_f16_cache_compose ? 1 : 0,
                    opt.overlap_ep_dense ? 1 : 0,
                    opt.direct_remote_compose ? 1 : 0,
                    opt.source_copy_schedule ? 1 : 0,
                    opt.skip_self_compose_copy ? 1 : 0,
                    opt.multi_copy_streams ? 1 : 0,
                    opt.decode_cudagraph_gate ? 1 : 0,
                    opt.decode_cudagraph_replay_probe_gate ? 1 : 0,
                    decode_loop.ep_ms_per_step,
                    decode_loop.dense_ms_per_step,
                    decode_loop.fused_compose_sum ? 1 : 0,
                    decode_loop.nccl_reduce_scatter_compose ? 1 : 0,
                    decode_loop.compose_ms_per_step,
                    decode_loop.compose_reduce_ms_per_step,
                    decode_loop.compose_copy_ms_per_step,
                    decode_loop.compose_final_ms_per_step,
                    opt.tp_hc_current_input_gate ? 1 : 0,
                    opt.tp_hc_current_input_peer_gather_gate ? 1 : 0,
                    opt.tp_hc_current_input_nccl_allgather_gate ? 1 : 0,
                    opt.tp_hc_current_allreduce_gate ? 1 : 0,
                    opt.tp_hc_current_input_stream_sync_gate ? 1 : 0,
                    decode_loop.hc_current_input_ms_per_step,
                    opt.final_hc_carry_gate ? 1 : 0,
                    decode_loop.final_hc_ms_per_step,
                    (unsigned long long)decode_loop.dense_loaded_bytes,
                    (unsigned long long)decode_loop.ep_contribution_bytes,
                    decode_loop.ep_return_fp16 ? "fp16" : "fp32",
                    (unsigned long long)decode_loop.ep_return_bytes,
                    decode_loop.cudagraph_replay_attempted,
                    decode_loop.cudagraph_replay_succeeded,
                    decode_loop.cudagraph_instantiate_ms,
                    decode_loop.cudagraph_replay_ms,
                    (unsigned long long)decode_loop.checksum,
                    decode_loop.finite_bad,
                    decode_loop.pass ? "PASS" : "FAIL");
    }
    if (decode_rc != 0) {
        close_local_runtime();
        return 14;
    }

    const uint64_t dispatch_bytes = (uint64_t)aggregate_routes * kHidden * sizeof(__half);
    const uint64_t return_bytes = dispatch_bytes;
    const double imbalance = min_routes > 0 ? (double)max_routes / (double)min_routes : 0.0;
    const double scaffold_ms = descriptor_ms + dense_kv_ms + worst_ep_ms;
    const bool comp_rows_expected = ds4_layer_ratio(opt.layer) != 0;
    const bool pass = layer_stats.dense_rows > 0 &&
                      layer_stats.control_rows > 0 &&
                      layer_stats.expert_rows > 0 &&
                      layer_stats.kv_rows > 0 &&
                      (!comp_rows_expected || layer_stats.comp_rows > 0) &&
                      (opt.skip_descriptor_checks || layer_stats.checksum != 0) &&
                      kv_result.max_abs == 0.0 &&
                      repeat_bad == 0 &&
                      repeat_nan == 0 &&
                      (!dense_compute.enabled || dense_compute.pass) &&
                      (!bf16_compute.enabled || bf16_compute.pass) &&
                      (!compose.enabled || compose.pass) &&
                      (!decode_loop.enabled || decode_loop.pass);

    std::printf("runtime_bytes_per_gpu\thidden\t%llu\tkv\t%llu\tcomp_state\t%llu\t"
                "scratch\t%llu\ttotal\t%llu\n",
                (unsigned long long)runtime_report.gpu[0].hidden_bytes,
                (unsigned long long)runtime_report.gpu[0].kv_bytes,
                (unsigned long long)runtime_report.gpu[0].comp_state_bytes,
                (unsigned long long)runtime_report.gpu[0].scratch_bytes,
                (unsigned long long)runtime_report.gpu[0].total_bytes);
    std::printf("dense_kv_slice\tlayer\t%d\tratio\t%d\tslot\t%u\tposition\t%llu\t"
                "attn_row\t%llu\tindexer_row\t%llu\tattn_row_bytes\t%llu\t"
                "indexer_row_bytes\t%llu\tmax_abs\t%.9f\tdense_kv_ms\t%.6f\n",
                kv_result.layer, kv_result.ratio, kv_result.slot,
                (unsigned long long)kv_result.position,
                (unsigned long long)kv_result.attn_row,
                (unsigned long long)kv_result.indexer_row,
                (unsigned long long)kv_result.attn_row_bytes[0],
                (unsigned long long)kv_result.indexer_row_bytes[0],
                kv_result.max_abs, dense_kv_ms);
    std::printf("tp_ep_full_layer_scaffold\tslots\t%d\tctx\t%llu\ttop_k\t%d\t"
                "layer\t%d\ttotal_rows\t%llu\tdense_rows\t%llu\tcontrol_rows\t%llu\t"
                "expert_rows\t%llu\tkv_rows\t%llu\tcomp_rows\t%llu\t"
                "dense_loaded_bytes\t%llu\tcontrol_loaded_bytes\t%llu\t"
                "ep_loaded_bytes\t%llu\tdescriptor_checksum\t%llu\t"
                "dense_compute_tensor\t%s\tdense_compute_rows_per_gpu\t%d\t"
                "dense_compute_cols\t%d\tdense_compute_slots\t%d\t"
                "dense_compute_loaded_bytes\t%llu\tdense_compute_ms\t%.6f\t"
                "dense_compute_repeat_max_abs\t%.9f\tdense_compute_repeat_bad\t%d\t"
                "dense_compute_repeat_nan\t%d\tdense_compute_oracle_max_abs\t%.9f\t"
                "dense_compute_oracle_bad\t%d\tdense_compute_pass\t%d\t"
                "bf16_compute_tensor\t%s\tbf16_compute_rows_per_gpu\t%d\t"
                "bf16_compute_cols\t%d\tbf16_compute_slots\t%d\t"
                "bf16_compute_loaded_bytes\t%llu\tbf16_compute_ms\t%.6f\t"
                "bf16_compute_repeat_max_abs\t%.9f\tbf16_compute_repeat_bad\t%d\t"
                "bf16_compute_repeat_nan\t%d\tbf16_compute_oracle_max_abs\t%.9f\t"
                "bf16_compute_oracle_bad\t%d\tbf16_compute_pass\t%d\t"
                "compose_next_hidden\t%d\tcompose_ep_contribution_bytes\t%llu\t"
                "compose_ep_return_dtype\t%s\tcompose_ep_return_bytes\t%llu\t"
                "compose_dense_hmma\t%d\tcompose_dense_f16_cublas\t%d\t"
                "compose_attn_dense_ms\t%.6f\t"
                "compose_shared_dense_ms\t%.6f\tcompose_fused_sum\t%d\t"
                "compose_nccl_reduce_scatter\t%d\t"
                "compose_ms\t%.6f\t"
                "compose_checksum\t%llu\tcompose_finite_bad\t%d\t"
                "compose_repeat_max_abs\t%.9f\tcompose_repeat_bad\t%d\t"
                "compose_pass\t%d\t"
                "decode_steps\t%d\tdecode_slot_steps\t%llu\tdecode_total_ms\t%.6f\t"
                "decode_ms_per_step\t%.6f\tdecode_slot_step_tok_s\t%.6f\t"
                "decode_dense_hmma\t%d\tdecode_dense_f16_cublas\t%d\t"
                "decode_dense_f16_cache\t%d\t"
                "decode_overlap_ep_dense\t%d\tdecode_direct_remote_compose\t%d\t"
                "decode_source_copy_schedule\t%d\t"
                "decode_ep_ms_per_step\t%.6f\tdecode_dense_ms_per_step\t%.6f\t"
                "decode_fused_compose_sum\t%d\tdecode_nccl_reduce_scatter\t%d\t"
                "decode_compose_ms_per_step\t%.6f\t"
                "decode_ep_return_dtype\t%s\t"
                "decode_ep_return_bytes\t%llu\tdecode_checksum\t%llu\t"
                "decode_finite_bad\t%d\tdecode_pass\t%d\t"
                "aggregate_routes\t%d\tdispatch_bytes\t%llu\treturn_bytes\t%llu\t"
                "route_imbalance\t%.6f\tdescriptor_ms\t%.6f\tdense_kv_ms\t%.6f\t"
                "worst_gate_ms\t%.6f\tworst_down_ms\t%.6f\tworst_ep_ms\t%.6f\t"
                "scaffold_ms\t%.6f\trepeat_max_abs\t%.9f\trepeat_bad\t%d\t"
                "repeat_nan\t%d\t%s\n",
                opt.slots, (unsigned long long)cfg.ctx, opt.top_k, opt.layer,
                (unsigned long long)layer_stats.total_rows,
                (unsigned long long)layer_stats.dense_rows,
                (unsigned long long)layer_stats.control_rows,
                (unsigned long long)layer_stats.expert_rows,
                (unsigned long long)layer_stats.kv_rows,
                (unsigned long long)layer_stats.comp_rows,
                (unsigned long long)layer_stats.dense_loaded_bytes,
                (unsigned long long)layer_stats.control_loaded_bytes,
                (unsigned long long)layer_stats.ep_loaded_bytes,
                (unsigned long long)layer_stats.checksum,
                dense_compute.enabled ? dense_compute.tensor_id.c_str() : "disabled",
                dense_compute.rows_per_gpu,
                dense_compute.cols,
                dense_compute.slots,
                (unsigned long long)dense_compute.loaded_bytes,
                dense_compute.compute_ms,
                dense_compute.repeat_max_abs,
                dense_compute.repeat_bad,
                dense_compute.repeat_nan,
                dense_compute.oracle_max_abs,
                dense_compute.oracle_bad,
                dense_compute.enabled && dense_compute.pass ? 1 : 0,
                bf16_compute.enabled ? bf16_compute.tensor_id.c_str() : "disabled",
                bf16_compute.rows_per_gpu,
                bf16_compute.cols,
                bf16_compute.slots,
                (unsigned long long)bf16_compute.loaded_bytes,
                bf16_compute.compute_ms,
                bf16_compute.repeat_max_abs,
                bf16_compute.repeat_bad,
                bf16_compute.repeat_nan,
                bf16_compute.oracle_max_abs,
                bf16_compute.oracle_bad,
                bf16_compute.enabled && bf16_compute.pass ? 1 : 0,
                compose.enabled ? 1 : 0,
                (unsigned long long)compose.ep_contribution_bytes,
                compose.ep_return_fp16 ? "fp16" : "fp32",
                (unsigned long long)compose.ep_return_bytes,
                compose.dense_hmma_compose ? 1 : 0,
                compose.dense_f16_cublas_compose ? 1 : 0,
                compose.attn_dense_ms,
                compose.shared_dense_ms,
                compose.fused_compose_sum ? 1 : 0,
                compose.nccl_reduce_scatter_compose ? 1 : 0,
                compose.compose_ms,
                (unsigned long long)compose.checksum,
                compose.finite_bad,
                compose.repeat_max_abs,
                compose.repeat_bad,
                compose.enabled && compose.pass ? 1 : 0,
                decode_loop.steps,
                (unsigned long long)decode_loop.slot_steps,
                decode_loop.total_ms,
                decode_loop.ms_per_step,
                decode_loop.tok_s,
                decode_loop.dense_hmma_compose ? 1 : 0,
                decode_loop.dense_f16_cublas_compose ? 1 : 0,
                decode_loop.dense_f16_cache_compose ? 1 : 0,
                opt.overlap_ep_dense ? 1 : 0,
                opt.direct_remote_compose ? 1 : 0,
                opt.source_copy_schedule ? 1 : 0,
                decode_loop.ep_ms_per_step,
                decode_loop.dense_ms_per_step,
                decode_loop.fused_compose_sum ? 1 : 0,
                decode_loop.nccl_reduce_scatter_compose ? 1 : 0,
                decode_loop.compose_ms_per_step,
                decode_loop.ep_return_fp16 ? "fp16" : "fp32",
                (unsigned long long)decode_loop.ep_return_bytes,
                (unsigned long long)decode_loop.checksum,
                decode_loop.finite_bad,
                decode_loop.enabled && decode_loop.pass ? 1 : 0,
                aggregate_routes,
                (unsigned long long)dispatch_bytes,
                (unsigned long long)return_bytes,
                imbalance, descriptor_ms, dense_kv_ms, worst_gate_ms, worst_down_ms,
                worst_ep_ms, scaffold_ms, repeat_max_abs, repeat_bad, repeat_nan,
                pass ? "PASS" : "FAIL");

    if (summary) {
        summary->layer = opt.layer;
        summary->ratio = ds4_layer_ratio(opt.layer);
        summary->pass = pass;
        summary->total_rows = layer_stats.total_rows;
        summary->dense_rows = layer_stats.dense_rows;
        summary->control_rows = layer_stats.control_rows;
        summary->expert_rows = layer_stats.expert_rows;
        summary->kv_rows = layer_stats.kv_rows;
        summary->comp_rows = layer_stats.comp_rows;
        summary->decode_ms_per_step = decode_loop.ms_per_step;
        summary->decode_slot_step_tok_s = decode_loop.tok_s;
        summary->decode_ep_ms_per_step = decode_loop.ep_ms_per_step;
        summary->decode_dense_ms_per_step = decode_loop.dense_ms_per_step;
        summary->decode_compose_ms_per_step = decode_loop.compose_ms_per_step;
        summary->decode_compose_reduce_ms_per_step =
            decode_loop.compose_reduce_ms_per_step;
        summary->decode_compose_copy_ms_per_step =
            decode_loop.compose_copy_ms_per_step;
        summary->decode_compose_final_ms_per_step =
            decode_loop.compose_final_ms_per_step;
        summary->decode_hc_current_input_ms_per_step =
            decode_loop.hc_current_input_ms_per_step;
        summary->decode_hc_current_seed_ms_per_step =
            decode_loop.hc_current_seed_ms_per_step;
        summary->decode_hc_current_attn_mix_ms_per_step =
            decode_loop.hc_current_attn_mix_ms_per_step;
        summary->decode_hc_current_split_ms_per_step =
            decode_loop.hc_current_split_ms_per_step;
        summary->decode_hc_current_gather_ms_per_step =
            decode_loop.hc_current_gather_ms_per_step;
        summary->decode_hc_current_ffn_router_ms_per_step =
            decode_loop.hc_current_ffn_router_ms_per_step;
        summary->decode_hc_current_ffn_norm_ms_per_step =
            decode_loop.hc_current_ffn_norm_ms_per_step;
        summary->decode_hc_current_router_select_ms_per_step =
            decode_loop.hc_current_router_select_ms_per_step;
        summary->decode_hc_current_router_d2h_ms_per_step =
            decode_loop.hc_current_router_d2h_ms_per_step;
        summary->decode_hc_current_route_upload_ms_per_step =
            decode_loop.hc_current_route_upload_ms_per_step;
        summary->decode_hc_current_fill_pack_ms_per_step =
            decode_loop.hc_current_fill_pack_ms_per_step;
        summary->decode_pre_ep_hc_current_ms_per_step =
            decode_loop.pre_ep_hc_current_ms_per_step;
        summary->decode_pre_ep_attention_projection_ms_per_step =
            decode_loop.pre_ep_attention_projection_ms_per_step;
        summary->decode_pre_ep_compressed_kv_ms_per_step =
            decode_loop.pre_ep_compressed_kv_ms_per_step;
        summary->decode_pre_ep_attention_state_ms_per_step =
            decode_loop.pre_ep_attention_state_ms_per_step;
        summary->decode_pre_ep_typed_history_ms_per_step =
            decode_loop.pre_ep_typed_history_ms_per_step;
        summary->decode_pre_ep_raw_read_ms_per_step =
            decode_loop.pre_ep_raw_read_ms_per_step;
        summary->decode_pre_ep_attention_output_ms_per_step =
            decode_loop.pre_ep_attention_output_ms_per_step;
        summary->decode_pre_ep_post_attention_ffn_input_ms_per_step =
            decode_loop.pre_ep_post_attention_ffn_input_ms_per_step;
        summary->decode_final_hc_ms_per_step = decode_loop.final_hc_ms_per_step;
        summary->decode_checksum = decode_loop.checksum;
    }

    if (!shared_rank_buffers) {
        close_compose_nccl(ranks);
    }
    for (int p = 0; p < kGpus; ++p) {
        RankState &r = ranks[p];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!layer_expert_cache) free_packed(r.gated);
        r.gated = PackedExperts{};
        if (!layer_expert_cache) free_packed(r.down);
        r.down = PackedExperts{};
        if (!shared_rank_buffers) {
            CHECK_CUDA(cudaFree(r.d_offsets));
            CHECK_CUDA(cudaFree(r.d_route_slots));
            CHECK_CUDA(cudaFree(r.d_route_weights));
            CHECK_CUDA(cudaFree(r.d_route_inv_scale));
            CHECK_CUDA(cudaFree(r.d_a));
            CHECK_CUDA(cudaFree(r.d_gate_up));
            CHECK_CUDA(cudaFree(r.d_gated));
            CHECK_CUDA(cudaFree(r.d_down));
            if (r.d_ep_contrib_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_all));
            if (r.d_ep_contrib_half_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_half_all));
            if (r.d_ep_contrib_bcast_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_bcast_all));
            if (r.d_ep_contrib_half_bcast_all) CHECK_CUDA(cudaFree(r.d_ep_contrib_half_bcast_all));
            for (int src = 0; src < kGpus; ++src) {
                if (r.d_ep_remote[src]) CHECK_CUDA(cudaFree(r.d_ep_remote[src]));
                if (r.d_ep_remote_half[src]) CHECK_CUDA(cudaFree(r.d_ep_remote_half[src]));
            }
            if (r.d_ep_sum) CHECK_CUDA(cudaFree(r.d_ep_sum));
            if (r.d_next_hidden) CHECK_CUDA(cudaFree(r.d_next_hidden));
            if (r.d_current_shard) CHECK_CUDA(cudaFree(r.d_current_shard));
            if (r.d_current_full) CHECK_CUDA(cudaFree(r.d_current_full));
            if (r.d_current_full_normed) CHECK_CUDA(cudaFree(r.d_current_full_normed));
            if (r.d_current_full_rank_major) CHECK_CUDA(cudaFree(r.d_current_full_rank_major));
            if (r.d_post_attn_full_rank_major) CHECK_CUDA(cudaFree(r.d_post_attn_full_rank_major));
            if (r.d_rank_major_norm_scale) CHECK_CUDA(cudaFree(r.d_rank_major_norm_scale));
            if (r.d_router_logits_shard) CHECK_CUDA(cudaFree(r.d_router_logits_shard));
            if (r.d_router_logits_rank_major) CHECK_CUDA(cudaFree(r.d_router_logits_rank_major));
            if (r.d_half_diff_counts) CHECK_CUDA(cudaFree(r.d_half_diff_counts));
            if (r.d_half_diff_max_bits) CHECK_CUDA(cudaFree(r.d_half_diff_max_bits));
            if (r.d_half_diff_first) CHECK_CUDA(cudaFree(r.d_half_diff_first));
            if (r.d_post_attn_route_audit) CHECK_CUDA(cudaFree(r.d_post_attn_route_audit));
            if (r.d_final_hc_shard) CHECK_CUDA(cudaFree(r.d_final_hc_shard));
            if (r.d_hc_scratch_shard) CHECK_CUDA(cudaFree(r.d_hc_scratch_shard));
            if (r.d_hc_split) CHECK_CUDA(cudaFree(r.d_hc_split));
            for (int layer = 0; layer < 43; ++layer) {
                if (r.d_attn_raw_swa_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_attn_raw_swa_layers[layer]));
                }
            }
            if (r.d_attn_kv_full) CHECK_CUDA(cudaFree(r.d_attn_kv_full));
            if (r.d_attn_heads) CHECK_CUDA(cudaFree(r.d_attn_heads));
            if (r.d_attn_output_a_full) CHECK_CUDA(cudaFree(r.d_attn_output_a_full));
            if (r.d_post_attn_shard) CHECK_CUDA(cudaFree(r.d_post_attn_shard));
            if (r.d_attn_sinks) CHECK_CUDA(cudaFree(r.d_attn_sinks));
            if (r.d_indexer_topk) CHECK_CUDA(cudaFree(r.d_indexer_topk));
            if (r.d_indexer_scores) CHECK_CUDA(cudaFree(r.d_indexer_scores));
            for (int layer = 0; layer < 43; ++layer) {
                if (r.d_index_comp_rows_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_index_comp_rows_layers[layer]));
                }
                if (r.d_index_comp_state_score_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_index_comp_state_score_layers[layer]));
                }
                if (r.d_index_comp_state_kv_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_index_comp_state_kv_layers[layer]));
                }
            }
            if (r.d_index_comp_score_cur) CHECK_CUDA(cudaFree(r.d_index_comp_score_cur));
            if (r.d_index_comp_kv_cur) CHECK_CUDA(cudaFree(r.d_index_comp_kv_cur));
            for (int layer = 0; layer < 43; ++layer) {
                if (r.d_attn_comp_rows_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_attn_comp_rows_layers[layer]));
                }
                if (r.d_attn_comp_state_score_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_attn_comp_state_score_layers[layer]));
                }
                if (r.d_attn_comp_state_kv_layers[layer]) {
                    CHECK_CUDA(cudaFree(r.d_attn_comp_state_kv_layers[layer]));
                }
            }
            if (r.d_attn_comp_score_cur) CHECK_CUDA(cudaFree(r.d_attn_comp_score_cur));
            if (r.d_attn_comp_kv_cur) CHECK_CUDA(cudaFree(r.d_attn_comp_kv_cur));
            if (r.dense_wait) CHECK_CUDA(cudaEventDestroy(r.dense_wait));
            if (r.start) CHECK_CUDA(cudaEventDestroy(r.start));
            if (r.mid) CHECK_CUDA(cudaEventDestroy(r.mid));
            if (r.stop) CHECK_CUDA(cudaEventDestroy(r.stop));
            if (r.d_route_totals) CHECK_CUDA(cudaFree(r.d_route_totals));
            if (r.d_route_offsets_all) CHECK_CUDA(cudaFree(r.d_route_offsets_all));
            if (r.d_router_weights_plan) CHECK_CUDA(cudaFree(r.d_router_weights_plan));
            if (r.d_router_selected_plan) CHECK_CUDA(cudaFree(r.d_router_selected_plan));
            const bool has_route_compact_plan = r.d_route_compact_plan != nullptr;
            if (r.d_route_compact_plan) CHECK_CUDA(cudaFree(r.d_route_compact_plan));
            for (int src = 0; src < kGpus; ++src) {
                if (r.d_route_index_by_slot[src]) CHECK_CUDA(cudaFree(r.d_route_index_by_slot[src]));
                if (!has_route_compact_plan && r.d_route_indices_by_slot[src]) {
                    CHECK_CUDA(cudaFree(r.d_route_indices_by_slot[src]));
                }
                if (!has_route_compact_plan && r.d_route_count_by_slot[src]) {
                    CHECK_CUDA(cudaFree(r.d_route_count_by_slot[src]));
                }
            }
            for (int q = 0; q < kGpus; ++q) {
                if (r.copy_done[q]) CHECK_CUDA(cudaEventDestroy(r.copy_done[q]));
                if (r.copy_streams[q]) CHECK_CUDA(cudaStreamDestroy(r.copy_streams[q]));
            }
            for (int e = 0; e < kGraphOrderEventSlots; ++e) {
                if (r.graph_stream_done[e]) {
                    CHECK_CUDA(cudaEventDestroy(r.graph_stream_done[e]));
                }
                if (r.graph_dense_done[e]) {
                    CHECK_CUDA(cudaEventDestroy(r.graph_dense_done[e]));
                }
            }
            if (r.dense_done) CHECK_CUDA(cudaEventDestroy(r.dense_done));
            if (r.stream_done) CHECK_CUDA(cudaEventDestroy(r.stream_done));
            CHECK_CUDA(cudaStreamDestroy(r.copy_stream));
            CHECK_CUDA(cudaStreamDestroy(r.dense_stream));
            CHECK_CUDA(cudaStreamDestroy(r.stream));
        }
    }
    if (!shared_api) {
        api->shutdown();
        dlclose(lib);
    }
    close_local_runtime();
    if (!shared_dense_f16_cache) free_dense_f16_cache(local_dense_f16_cache, opt);
    return pass ? 0 : 1;
}

