int run_tp_ep_appliance(Options opt) {
    reset_peer_copy_accounting(opt.tp_peer_accounting_gate,
                               opt.tp_peer_reject_sys_gate);
    if (opt.serving_bench) {
        opt.skip_decode_checksum = true;
    }
    if (opt.token_major_all_layers && opt.all_layers && !opt.tp_runtime_explicit) {
        opt.share_tp_runtime = true;
    }
    if (report_vram_checkpoint(opt, "startup") != 0) {
        return 14;
    }

    if (!opt.all_layers) {
        return run_layer(opt, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    DenseF16Cache all_layer_dense_f16_cache;
    DenseF16Cache *shared_dense_f16_cache = nullptr;
    if (opt.dense_f16_cache_compose) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0) {
            std::fprintf(stderr, "all-layer contract parse failed bad_rows=%llu\n",
                         (unsigned long long)all_stats.bad_rows);
            return 2;
        }
        const auto cache_start = std::chrono::steady_clock::now();
        if (prepare_dense_f16_cache(opt, all_rows, &all_layer_dense_f16_cache) != 0) {
            std::fprintf(stderr, "all-layer dense f16 cache prepare failed\n");
            return 4;
        }
        const auto cache_stop = std::chrono::steady_clock::now();
        const double cache_ms =
            std::chrono::duration<double, std::milli>(cache_stop - cache_start).count();
        shared_dense_f16_cache = &all_layer_dense_f16_cache;
        std::printf("tp_ep_all_layer_dense_f16_cache\trows\t%llu\t"
                    "source_bytes\t%llu\tcache_bytes\t%llu\t"
                    "cache_aligned_bytes\t%llu\tmax_temp_bytes\t%llu\t"
                    "cache_ms\t%.6f\tPASS\n",
                    (unsigned long long)all_layer_dense_f16_cache.rows,
                    (unsigned long long)all_layer_dense_f16_cache.source_bytes,
                    (unsigned long long)all_layer_dense_f16_cache.cache_bytes,
                    (unsigned long long)all_layer_dense_f16_cache.cache_aligned_bytes,
                    (unsigned long long)all_layer_dense_f16_cache.max_temp_bytes,
                    cache_ms);
        if (report_vram_checkpoint(opt, "after_dense_f16_cache") != 0) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            return 14;
        }
    }

    SharedApi shared_api;
    if (open_shared_api(opt, &shared_api) != 0) {
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 6;
    }
    std::printf("tp_ep_all_layer_turbomind_api_shared\tdevices\t%d\tPASS\n", kGpus);

    SharedRankBuffers shared_rank_buffers;
    if (open_shared_rank_buffers(opt, &shared_rank_buffers) != 0) {
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 7;
    }
    std::printf("tp_ep_all_layer_rank_buffers_shared\tdevices\t%d\tcore_bytes\t%llu\tPASS\n",
                kGpus, (unsigned long long)shared_rank_buffers.core_bytes);
    if (report_vram_checkpoint(opt, "after_rank_buffers") != 0) {
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }
    if (report_nccl_vram_checkpoint(opt, "nccl_after_rank_buffers") != 0) {
        std::fprintf(stderr,
                     "tp_ep_nccl_vram_admission_failed label=nccl_after_rank_buffers "
                     "min_free_mib=%llu\n",
                     (unsigned long long)opt.nccl_min_free_mib);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }

    SharedTpRuntime shared_tp_runtime;
    if (opt.share_tp_runtime && open_shared_tp_runtime(opt, &shared_tp_runtime) != 0) {
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 8;
    }
    if (shared_tp_runtime.initialized) {
        std::printf("tp_ep_all_layer_tp_runtime_shared\tdevices\t%d\tslots\t%d\tctx\t262144\t"
                    "kv_bytes_per_gpu\t%llu\tcomp_state_bytes_per_gpu\t%llu\t"
                    "scratch_bytes_per_gpu\t%llu\ttotal_bytes_per_gpu\t%llu\tPASS\n",
                    kGpus, opt.slots,
                    (unsigned long long)shared_tp_runtime.report.gpu[0].kv_bytes,
                    (unsigned long long)shared_tp_runtime.report.gpu[0].comp_state_bytes,
                    (unsigned long long)shared_tp_runtime.report.gpu[0].scratch_bytes,
                    (unsigned long long)shared_tp_runtime.report.gpu[0].total_bytes);
    } else {
        std::printf("tp_ep_all_layer_tp_runtime_shared\tdevices\t%d\tslots\t%d\tctx\t262144\t"
                    "mode\tlocal_per_layer\tPASS\n", kGpus, opt.slots);
    }
    if (report_vram_checkpoint(opt, "after_tp_runtime") != 0) {
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }

    SharedExpertBindings shared_expert_bindings;
    if (opt.share_expert_bindings &&
        open_shared_expert_bindings(opt, &shared_expert_bindings) != 0) {
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 9;
    }
    /* MTP (layer 43) experts from the dedicated MTP pack dir (no-op unless
     * --mtp-pack-dir/--mtp-tm-index are set). EP-split 32/rank like 0-42. */
    if (open_mtp_expert_bindings(opt, &shared_expert_bindings) != 0) {
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 9;
    }
    /* MTP (layer 43) non-expert families (norms/hc/proj/shared) from the MTP
     * contract+pack (no-op unless --mtp-contract/--mtp-pack-dir are set). */
    MtpNonExpertWeights mtp_nonexpert;
    if (open_mtp_nonexpert_bindings(opt, &mtp_nonexpert) != 0) {
        close_mtp_nonexpert_bindings(&mtp_nonexpert);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 9;
    }
    if (shared_expert_bindings.initialized) {
        std::printf("tp_ep_all_layer_expert_bindings_shared\tlayers\t43\tdevices\t%d\t"
                    "bytes\t%llu\tbytes_per_gpu\t%llu\tPASS\n",
                    kGpus,
                    (unsigned long long)shared_expert_bindings.bytes,
                    (unsigned long long)(shared_expert_bindings.bytes / kGpus));
    } else {
        std::printf("tp_ep_all_layer_expert_bindings_shared\tlayers\t43\tdevices\t%d\t"
                    "mode\tlocal_per_layer\tPASS\n", kGpus);
    }

    SharedDenseOps shared_dense_ops;
    if (opt.share_dense_ops && open_shared_dense_ops(opt, shared_dense_f16_cache,
                                                     &shared_dense_ops) != 0) {
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 10;
    }
    if (shared_dense_ops.initialized) {
        std::printf("tp_ep_all_layer_dense_ops_shared\tlayers\t43\tdevices\t%d\t"
                    "loaded_bytes\t%llu\tPASS\n",
                    kGpus, (unsigned long long)shared_dense_ops.loaded_bytes);
    } else {
        std::printf("tp_ep_all_layer_dense_ops_shared\tlayers\t43\tdevices\t%d\t"
                    "mode\tlocal_per_layer\tPASS\n", kGpus);
    }
    if (report_vram_checkpoint(opt, "after_dense_ops") != 0) {
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }

    std::vector<ContractRow> resident_rows[43];
    LayerStats resident_stats[43];
    const bool resident_serving_loop =
        opt.serving_bench && opt.token_major_all_layers &&
        shared_tp_runtime.initialized && shared_expert_bindings.initialized &&
        shared_dense_f16_cache != nullptr;
    if (resident_serving_loop) {
        for (int layer = 0; layer < 43; ++layer) {
            if (parse_contract(opt.contract_path, layer, &resident_rows[layer],
                               &resident_stats[layer]) != 0 ||
                resident_stats[layer].bad_rows != 0) {
                std::fprintf(stderr, "resident serving contract parse failed layer=%d bad_rows=%llu\n",
                             layer, (unsigned long long)resident_stats[layer].bad_rows);
                free_shared_dense_ops(&shared_dense_ops, opt);
                close_shared_expert_bindings(&shared_expert_bindings);
                close_shared_tp_runtime(&shared_tp_runtime);
                close_shared_rank_buffers(&shared_rank_buffers);
                close_shared_api(&shared_api);
                if (shared_dense_f16_cache) {
                    free_dense_f16_cache(all_layer_dense_f16_cache, opt);
                }
                return 11;
            }
        }
        std::printf("tp_ep_resident_serving_loop\tlayers\t43\tmode\tdirect_decode\tPASS\n");
    }

    SharedHcControls shared_hc_controls;
    if (opt.tp_hc_final_expand_gate) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0 ||
            open_shared_hc_controls(opt, all_rows, &shared_hc_controls) != 0) {
            std::fprintf(stderr, "tp_ep HC final-expand controls open failed\n");
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 12;
        }
        std::printf("tp_ep_hc_final_expand_shared\tlayers\t43\tslots\t%d\t"
                    "control_bytes\t%llu\tPASS\n",
                    opt.slots, (unsigned long long)shared_hc_controls.control_bytes);
        /* MTP (layer 43) HC/norm controls into slot 43 (no-op unless MTP source set). */
        if (load_mtp_hc_layer43(opt, &shared_hc_controls) != 0) {
            std::fprintf(stderr, "tp_ep MTP HC layer-43 controls open failed\n");
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 12;
        }
        if (report_vram_checkpoint(opt, "after_hc_controls") != 0) {
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
    }
    SharedHcControls *shared_hc_controls_arg =
        shared_hc_controls.initialized ? &shared_hc_controls : nullptr;

    if (opt.resident_profile_layer >= 0) {
        if (opt.defer_nccl_init_gate &&
            open_compose_nccl(opt, shared_rank_buffers.ranks) != 0) {
            std::fprintf(stderr,
                         "tp_ep_resident_profile_layer_deferred_nccl_open_failed\n");
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
        if (opt.defer_nccl_init_gate &&
            report_nccl_vram_checkpoint(opt, "nccl_after_resident_profile_deferred_init") != 0) {
            std::fprintf(stderr,
                         "tp_ep_nccl_vram_admission_failed "
                         "label=nccl_after_resident_profile_deferred_init "
                         "min_free_mib=%llu\n",
                         (unsigned long long)opt.nccl_min_free_mib);
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
        const int layer = opt.resident_profile_layer;
        int rc = 0;
        LayerRunSummary s;
        std::vector<ContractRow> layer_rows;
        LayerStats layer_stats;
        if (!shared_tp_runtime.initialized ||
            !shared_expert_bindings.layers[layer].initialized ||
            !shared_dense_ops.initialized ||
            !shared_dense_f16_cache ||
            !shared_hc_controls_arg ||
            parse_contract(opt.contract_path, layer, &layer_rows, &layer_stats) != 0 ||
            layer_stats.bad_rows != 0) {
            std::fprintf(stderr,
                         "tp_ep_resident_profile_layer_setup_failed\tlayer\t%d\n",
                         layer);
            rc = 15;
        } else {
            Options layer_opt = opt;
            layer_opt.layer = layer;
            if (layer_opt.decode_steps <= 0) layer_opt.decode_steps = 8;
            const LayerDenseOps *layer_dense_ops = &shared_dense_ops.layers[layer];
            TpCudaGraphLayerExec *persistent_graph =
                opt.decode_cudagraph_persistent_replay_gate
                    ? &shared_rank_buffers.graph_cache.layers[layer]
                    : nullptr;
            const auto profile_start = std::chrono::steady_clock::now();
            rc = run_resident_layer_decode(layer_opt,
                                           layer_rows,
                                           layer_stats,
                                           shared_rank_buffers.ranks,
                                           shared_api.api,
                                           shared_tp_runtime.rt,
                                           &shared_expert_bindings.layers[layer],
                                           shared_dense_f16_cache,
                                           layer_dense_ops,
                                           shared_hc_controls_arg,
                                           persistent_graph,
                                           &s);
            const auto profile_stop = std::chrono::steady_clock::now();
            const double wall_ms =
                std::chrono::duration<double, std::milli>(
                    profile_stop - profile_start).count();
            std::printf("tp_ep_resident_profile_layer\tlayer\t%d\tratio\t%d\t"
                        "slots\t%d\tctx\t262144\tdecode_steps\t%d\t"
                        "shared_hc_controls\t%d\tshared_dense_ops\t%d\t"
                        "single_layer_experts\t1\t"
                        "decode_ms_per_step\t%.6f\tdecode_slot_step_tok_s\t%.6f\t"
                        "decode_cudagraph_capture_attempted\t%d\t"
                        "decode_cudagraph_capture_succeeded\t%d\t"
                        "decode_cudagraph_replay_attempted\t%d\t"
                        "decode_cudagraph_replay_succeeded\t%d\t"
                        "decode_cudagraph_replay_ms\t%.6f\t"
                        "wall_ms\t%.6f\tchecksum\t%llu\trc\t%d\t%s\n",
                        s.layer, s.ratio, opt.slots, layer_opt.decode_steps,
                        shared_hc_controls_arg ? 1 : 0,
                        shared_dense_ops.initialized ? 1 : 0,
                        s.decode_ms_per_step,
                        s.decode_slot_step_tok_s,
                        s.decode_cudagraph_capture_attempted,
                        s.decode_cudagraph_capture_succeeded,
                        s.decode_cudagraph_replay_attempted,
                        s.decode_cudagraph_replay_succeeded,
                        s.decode_cudagraph_replay_ms,
                        wall_ms,
                        (unsigned long long)s.decode_checksum,
                        rc,
                        (rc == 0 && s.pass) ? "PASS" : "FAIL");
        }
        close_shared_hc_controls(opt, &shared_hc_controls);
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return rc;
    }

    SharedOutputHead shared_output_head;
    if (opt.diagnostic_output_head && !opt.diagnostic_output_head_lazy_gate) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0 ||
            open_shared_output_head(opt, all_rows, &shared_output_head) != 0) {
            std::fprintf(stderr, "tp_ep diagnostic output-head open failed\n");
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 12;
        }
        std::printf("tp_ep_diagnostic_output_head_shared\tslots\t%d\tvocab\t%d\t"
                    "rows_per_gpu\t%d\toutput_weight_bytes\t%llu\t"
                    "logits_bytes\t%llu\tproxy_hc\t%d\tPASS\n",
                    opt.slots,
                    shared_output_head.vocab,
                    shared_output_head.rows_per_gpu,
                    (unsigned long long)shared_output_head.output_weight_bytes,
                    (unsigned long long)shared_output_head.logits_bytes,
                    opt.tp_hc_final_expand_gate ? 0 : 1);
        if (report_vram_checkpoint(opt, "after_output_head") != 0) {
            close_shared_output_head(opt, &shared_output_head);
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
        if (report_nccl_vram_checkpoint(opt, "nccl_after_output_head") != 0) {
            std::fprintf(stderr,
                         "tp_ep_nccl_vram_admission_failed label=nccl_after_output_head "
                         "min_free_mib=%llu\n",
                         (unsigned long long)opt.nccl_min_free_mib);
            close_shared_output_head(opt, &shared_output_head);
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
    }
    SharedOutputHead *shared_output_head_arg =
        shared_output_head.initialized ? &shared_output_head : nullptr;

    SharedTokenEmbedding shared_token_embedding;
    if (opt.serve_http) {
        std::vector<ContractRow> all_rows;
        LayerStats all_stats;
        if (parse_contract(opt.contract_path, -1, &all_rows, &all_stats) != 0 ||
            all_stats.bad_rows != 0 ||
            open_shared_token_embedding(opt, all_rows, &shared_token_embedding) != 0) {
            std::fprintf(stderr, "tp_ep token embedding open failed\n");
            close_shared_output_head(opt, &shared_output_head);
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 13;
        }
        std::printf("tp_ep_token_embedding_shared\tslots\t%d\tvocab\t%d\t"
                    "rows_per_gpu\t%d\tweight_bytes\t%llu\tdevice\t%d\tPASS\n",
                    opt.slots,
                    shared_token_embedding.vocab,
                    shared_token_embedding.rows_per_gpu,
                    (unsigned long long)shared_token_embedding.weight_bytes,
                    opt.devices[0]);
        if (report_vram_checkpoint(opt, "after_token_embedding") != 0) {
            close_shared_token_embedding(opt, &shared_token_embedding);
            close_shared_output_head(opt, &shared_output_head);
            close_shared_hc_controls(opt, &shared_hc_controls);
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return 14;
        }
    }
    SharedTokenEmbedding *shared_token_embedding_arg =
        shared_token_embedding.initialized ? &shared_token_embedding : nullptr;

    if (opt.defer_nccl_init_gate && open_compose_nccl(opt, shared_rank_buffers.ranks) != 0) {
        std::fprintf(stderr, "tp_ep_deferred_nccl_open_failed\n");
        close_shared_token_embedding(opt, &shared_token_embedding);
        close_shared_output_head(opt, &shared_output_head);
        close_shared_hc_controls(opt, &shared_hc_controls);
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }
    if (opt.defer_nccl_init_gate && report_nccl_vram_checkpoint(opt, "nccl_after_deferred_init") != 0) {
        std::fprintf(stderr,
                     "tp_ep_nccl_vram_admission_failed label=nccl_after_deferred_init "
                     "min_free_mib=%llu\n",
                     (unsigned long long)opt.nccl_min_free_mib);
        close_shared_token_embedding(opt, &shared_token_embedding);
        close_shared_output_head(opt, &shared_output_head);
        close_shared_hc_controls(opt, &shared_hc_controls);
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return 14;
    }

    if (opt.serve_http) {
        int rc = 0;
        if (!resident_serving_loop || !shared_dense_ops.initialized) {
            std::fprintf(stderr, "tp_ep_http requires resident serving loop and shared dense ops\n");
            rc = 13;
        } else {
            if (opt.decode_steps <= 0) opt.decode_steps = 32;
            rc = run_tp_ep_http_server(opt,
                                       shared_dense_f16_cache,
                                       &shared_api,
                                       &shared_rank_buffers,
                                       &shared_tp_runtime,
                                       &shared_expert_bindings,
                                       &shared_dense_ops,
                                       shared_output_head_arg,
                                       shared_hc_controls_arg,
                                       shared_token_embedding_arg,
                                       resident_rows,
                                       resident_stats);
        }
        if (opt.tp_peer_accounting_gate) {
            print_peer_copy_summary("http");
        }
        close_shared_token_embedding(opt, &shared_token_embedding);
        close_shared_output_head(opt, &shared_output_head);
        close_shared_hc_controls(opt, &shared_hc_controls);
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return rc;
    }

    if (opt.token_major_all_layers) {
        const int rc = run_token_major_serving_loop(opt,
                                                    shared_dense_f16_cache,
                                                    &shared_api,
                                                    &shared_rank_buffers,
                                                    &shared_tp_runtime,
                                                    &shared_expert_bindings,
                                                    &shared_dense_ops,
                                                    shared_output_head_arg,
                                                    shared_hc_controls_arg,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    resident_rows,
                                                    resident_stats,
                                                    resident_serving_loop,
                                                    nullptr);
        if (opt.tp_peer_accounting_gate) {
            print_peer_copy_summary("token_major");
        }
        close_shared_output_head(opt, &shared_output_head);
        close_shared_hc_controls(opt, &shared_hc_controls);
        free_shared_dense_ops(&shared_dense_ops, opt);
        close_shared_expert_bindings(&shared_expert_bindings);
        close_shared_tp_runtime(&shared_tp_runtime);
        close_shared_rank_buffers(&shared_rank_buffers);
        close_shared_api(&shared_api);
        if (shared_dense_f16_cache) {
            free_dense_f16_cache(all_layer_dense_f16_cache, opt);
        }
        return rc;
    }

    int pass_layers = 0;
    double sum_decode_ms = 0.0;
    double sum_ep_ms = 0.0;
    double sum_dense_ms = 0.0;
    double sum_compose_ms = 0.0;
    double sum_compose_reduce_ms = 0.0;
    double sum_compose_copy_ms = 0.0;
    double sum_compose_final_ms = 0.0;
    double sum_hc_current_input_ms = 0.0;
    uint64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int layer = 0; layer < 43; ++layer) {
        Options layer_opt = opt;
        layer_opt.layer = layer;
        LayerRunSummary s;
        SharedTpRuntime *tp_runtime_arg =
            shared_tp_runtime.initialized ? &shared_tp_runtime : nullptr;
        const SharedExpertBindings *expert_arg =
            shared_expert_bindings.initialized ? &shared_expert_bindings : nullptr;
        const SharedDenseOps *dense_ops_arg =
            shared_dense_ops.initialized ? &shared_dense_ops : nullptr;
        const int rc = run_layer(layer_opt, &s, shared_dense_f16_cache, &shared_api,
                                 &shared_rank_buffers, tp_runtime_arg, expert_arg,
                                 dense_ops_arg, shared_hc_controls_arg);
        std::printf("tp_ep_all_layer_item\tlayer\t%d\tratio\t%d\t"
                    "total_rows\t%llu\tdense_rows\t%llu\tcontrol_rows\t%llu\t"
                    "expert_rows\t%llu\tkv_rows\t%llu\tcomp_rows\t%llu\t"
                    "decode_ms_per_step\t%.6f\tdecode_slot_step_tok_s\t%.6f\t"
                    "decode_ep_ms_per_step\t%.6f\tdecode_dense_ms_per_step\t%.6f\t"
                    "decode_compose_ms_per_step\t%.6f\t"
                    "decode_compose_reduce_ms_per_step\t%.6f\t"
                    "decode_compose_copy_ms_per_step\t%.6f\t"
                    "decode_compose_final_ms_per_step\t%.6f\t"
                    "decode_hc_current_input_ms_per_step\t%.6f\t"
                    "decode_checksum\t%llu\tdecode_finite_bad\t%d\trc\t%d\t%s\n",
                    s.layer, s.ratio,
                    (unsigned long long)s.total_rows,
                    (unsigned long long)s.dense_rows,
                    (unsigned long long)s.control_rows,
                    (unsigned long long)s.expert_rows,
                    (unsigned long long)s.kv_rows,
                    (unsigned long long)s.comp_rows,
                    s.decode_ms_per_step,
                    s.decode_slot_step_tok_s,
                    s.decode_ep_ms_per_step,
                    s.decode_dense_ms_per_step,
                    s.decode_compose_ms_per_step,
                    s.decode_compose_reduce_ms_per_step,
                    s.decode_compose_copy_ms_per_step,
                    s.decode_compose_final_ms_per_step,
                    s.decode_hc_current_input_ms_per_step,
                    (unsigned long long)s.decode_checksum,
                    s.decode_finite_bad,
                    rc,
                    (rc == 0 && s.pass) ? "PASS" : "FAIL");
        if (rc == 0 && s.pass) {
            pass_layers++;
            sum_decode_ms += s.decode_ms_per_step;
            sum_ep_ms += s.decode_ep_ms_per_step;
            sum_dense_ms += s.decode_dense_ms_per_step;
            sum_compose_ms += s.decode_compose_ms_per_step;
            sum_compose_reduce_ms += s.decode_compose_reduce_ms_per_step;
            sum_compose_copy_ms += s.decode_compose_copy_ms_per_step;
            sum_compose_final_ms += s.decode_compose_final_ms_per_step;
            sum_hc_current_input_ms += s.decode_hc_current_input_ms_per_step;
            checksum ^= s.decode_checksum + (uint64_t)(layer + 1) * 104729ull;
        } else {
            const auto stop = std::chrono::steady_clock::now();
            const double wall_ms =
                std::chrono::duration<double, std::milli>(stop - start).count();
            std::printf("tp_ep_all_layer_scaffold\tlayers\t43\tpass_layers\t%d\t"
                        "failed_layer\t%d\tdescriptor_checks\t%d\tpredecode_probes\t%d\t"
                        "shared_api\t%d\tshared_rank_buffers\t%d\tshared_tp_runtime\t%d\t"
                        "shared_expert_bindings\t%d\t"
                        "shared_dense_ops\t%d\t"
                        "overlap_ep_dense\t%d\tdirect_remote_compose\t%d\t"
                        "source_copy_schedule\t%d\tskip_self_compose_copy\t%d\t"
                        "multi_copy_streams\t%d\t"
                        "wall_ms\t%.6f\tFAIL\n",
                        pass_layers, layer, opt.skip_descriptor_checks ? 0 : 1,
                        opt.skip_predecode_probes ? 0 : 1, shared_api.initialized ? 1 : 0,
                        shared_rank_buffers.initialized ? 1 : 0,
                        shared_tp_runtime.initialized ? 1 : 0,
                        shared_expert_bindings.initialized ? 1 : 0,
                        shared_dense_ops.initialized ? 1 : 0,
                        opt.overlap_ep_dense ? 1 : 0,
                        opt.direct_remote_compose ? 1 : 0,
                        opt.source_copy_schedule ? 1 : 0,
                        opt.skip_self_compose_copy ? 1 : 0,
                        opt.multi_copy_streams ? 1 : 0,
                        wall_ms);
            if (opt.tp_peer_accounting_gate) {
                print_peer_copy_summary("all_layer_fail");
            }
            free_shared_dense_ops(&shared_dense_ops, opt);
            close_shared_expert_bindings(&shared_expert_bindings);
            close_shared_tp_runtime(&shared_tp_runtime);
            close_shared_rank_buffers(&shared_rank_buffers);
            close_shared_api(&shared_api);
            if (shared_dense_f16_cache) {
                free_dense_f16_cache(all_layer_dense_f16_cache, opt);
            }
            return rc == 0 ? 1 : rc;
        }
    }
    /* MTP block (layer 43): validate the MTP transformer body runs via run_layer
     * (reuses the EP all-to-all dispatch). Experts from shared_expert_bindings
     * ->mtp_layer, contract from mtp_contract_path, ratio=0; dense/control load
     * from the MTP contract (no f16 cache / dense_ops for layer 43). */
    if (shared_expert_bindings.mtp_initialized && opt.mtp_contract_path) {
        Options mtp_opt = opt;
        mtp_opt.layer = 43;
        LayerRunSummary ms;
        SharedTpRuntime *tp_runtime_arg =
            shared_tp_runtime.initialized ? &shared_tp_runtime : nullptr;
        const int mrc = run_layer(mtp_opt, &ms, nullptr, &shared_api,
                                  &shared_rank_buffers, tp_runtime_arg,
                                  &shared_expert_bindings, nullptr,
                                  shared_hc_controls_arg);
        std::printf("tp_ep_mtp_layer_scaffold\tlayer\t43\tratio\t%d\t"
                    "expert_rows\t%llu\tdense_rows\t%llu\tcontrol_rows\t%llu\t"
                    "decode_ms_per_step\t%.6f\tdecode_checksum\t%llu\trc\t%d\t%s\n",
                    ms.ratio, (unsigned long long)ms.expert_rows,
                    (unsigned long long)ms.dense_rows,
                    (unsigned long long)ms.control_rows,
                    ms.decode_ms_per_step, (unsigned long long)ms.decode_checksum,
                    mrc, (mrc == 0 && ms.pass) ? "PASS" : "FAIL");
    }
    const auto stop = std::chrono::steady_clock::now();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    const double slot_step_tok_s = sum_decode_ms > 0.0
        ? (double)opt.slots * 1000.0 / sum_decode_ms
        : 0.0;
    std::printf("tp_ep_all_layer_scaffold\tlayers\t43\tpass_layers\t%d\t"
                "slots\t%d\tctx\t262144\tdecode_steps_per_layer\t%d\t"
                "descriptor_checks\t%d\tpredecode_probes\t%d\tshared_api\t%d\t"
                "shared_rank_buffers\t%d\tshared_tp_runtime\t%d\t"
                "shared_expert_bindings\t%d\t"
                "shared_dense_ops\t%d\t"
                "overlap_ep_dense\t%d\tdirect_remote_compose\t%d\t"
                "source_copy_schedule\t%d\tskip_self_compose_copy\t%d\t"
                "multi_copy_streams\t%d\t"
                "sum_decode_ms_per_token\t%.6f\tprojected_slot_step_tok_s\t%.6f\t"
                "sum_ep_ms\t%.6f\tsum_dense_ms\t%.6f\tsum_compose_ms\t%.6f\t"
                "sum_compose_reduce_ms\t%.6f\tsum_compose_copy_ms\t%.6f\t"
                "sum_compose_final_ms\t%.6f\t"
                "tp_hc_current_input_gate\t%d\t"
                "tp_hc_current_input_peer_gather\t%d\t"
                "tp_hc_current_input_nccl_allgather\t%d\t"
                "tp_hc_current_allreduce\t%d\t"
                "tp_hc_current_input_stream_sync\t%d\t"
                "sum_hc_current_input_ms\t%.6f\t"
                "wall_ms\t%.6f\tchecksum\t%llu\tPASS\n",
                pass_layers, opt.slots, opt.decode_steps,
                opt.skip_descriptor_checks ? 0 : 1,
                opt.skip_predecode_probes ? 0 : 1,
                shared_api.initialized ? 1 : 0,
                shared_rank_buffers.initialized ? 1 : 0,
                shared_tp_runtime.initialized ? 1 : 0,
                shared_expert_bindings.initialized ? 1 : 0,
                shared_dense_ops.initialized ? 1 : 0,
                opt.overlap_ep_dense ? 1 : 0,
                opt.direct_remote_compose ? 1 : 0,
                opt.source_copy_schedule ? 1 : 0,
                opt.skip_self_compose_copy ? 1 : 0,
                opt.multi_copy_streams ? 1 : 0,
                sum_decode_ms, slot_step_tok_s, sum_ep_ms, sum_dense_ms,
                sum_compose_ms, sum_compose_reduce_ms, sum_compose_copy_ms,
                sum_compose_final_ms,
                opt.tp_hc_current_input_gate ? 1 : 0,
                opt.tp_hc_current_input_peer_gather_gate ? 1 : 0,
                opt.tp_hc_current_input_nccl_allgather_gate ? 1 : 0,
                opt.tp_hc_current_allreduce_gate ? 1 : 0,
                opt.tp_hc_current_input_stream_sync_gate ? 1 : 0,
                sum_hc_current_input_ms,
                wall_ms, (unsigned long long)checksum);
    if (opt.tp_peer_accounting_gate) {
        print_peer_copy_summary("all_layer");
    }
    close_shared_hc_controls(opt, &shared_hc_controls);
    free_shared_dense_ops(&shared_dense_ops, opt);
    close_shared_expert_bindings(&shared_expert_bindings);
    close_shared_tp_runtime(&shared_tp_runtime);
    close_shared_rank_buffers(&shared_rank_buffers);
    close_shared_api(&shared_api);
    if (shared_dense_f16_cache) {
        free_dense_f16_cache(all_layer_dense_f16_cache, opt);
    }
    return 0;
}
