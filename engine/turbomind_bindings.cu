int parse_tm_index(const char *path, int layer, DescriptorBindings *out) {
    FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "cannot open tm index %s: %s\n", path, std::strerror(errno));
        return 1;
    }
    char gated_name[128];
    char down_name[128];
    std::snprintf(gated_name, sizeof(gated_name), "blk.%d.ffn_gate_up_exps.weight", layer);
    std::snprintf(down_name, sizeof(down_name), "blk.%d.ffn_down_exps.weight", layer);
    char buf[8192];
    bool first = true;
    while (std::fgets(buf, sizeof(buf), fp)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (first) {
            first = false;
            continue;
        }
        if (line.empty()) continue;
        std::vector<std::string> f = split_tabs(line);
        TmIndexEntry e;
        if (!parse_tm_entry(f, &e)) {
            std::fclose(fp);
            return 2;
        }
        if (e.layer_id != layer) continue;
        if (e.semantic_tensor_id == gated_name) {
            if (!valid_tm_entry(e, kFusedN, kHidden,
                                "turbomind_mxfp4_grouped_gate_up_interleaved")) {
                std::fclose(fp);
                return 3;
            }
            out->gated = e;
            out->have_gated = true;
        } else if (e.semantic_tensor_id == down_name) {
            if (!valid_tm_entry(e, kHidden, kMid, "turbomind_mxfp4_grouped")) {
                std::fclose(fp);
                return 4;
            }
            out->down = e;
            out->have_down = true;
        }
    }
    std::fclose(fp);
    return out->have_gated && out->have_down ? 0 : 5;
}

void load_api(void *lib, Api *api) {
    api->init = (pfn_init)dlsym(lib, "ggml_turbomind_init");
    api->shutdown = (pfn_shutdown)dlsym(lib, "ggml_turbomind_shutdown");
    api->mmgt = (pfn_mmgt)dlsym(lib, "ggml_turbomind_mul_mat_grouped_total_tokens");
    api->mmgs = (pfn_mmgs)dlsym(lib, "ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens");
    api->mmgs_clamped =
        (pfn_mmgs)dlsym(lib, "ggml_turbomind_mul_mat_grouped_gated_silu_clamped_total_tokens");
    if (!api->init || !api->shutdown || !api->mmgt || !api->mmgs) {
        std::fprintf(stderr, "dlsym failed for required TurboMind ABI\n");
        std::exit(2);
    }
}

int open_shared_api(const Options &opt, SharedApi *shared) {
    shared->lib = dlopen(opt.lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!shared->lib) {
        std::fprintf(stderr, "dlopen failed for %s: %s\n", opt.lib_path, dlerror());
        return 1;
    }
    load_api(shared->lib, &shared->api);
    for (int p = 0; p < kGpus; ++p) {
        if (shared->api.init(opt.devices[p]) != 0) {
            std::fprintf(stderr, "ggml_turbomind_init failed on device %d\n", opt.devices[p]);
            if (shared->api.shutdown) shared->api.shutdown();
            dlclose(shared->lib);
            *shared = SharedApi{};
            return 2;
        }
    }
    shared->initialized = true;
    return 0;
}

void close_shared_api(SharedApi *shared) {
    if (!shared || !shared->lib) return;
    if (shared->initialized && shared->api.shutdown) shared->api.shutdown();
    dlclose(shared->lib);
    *shared = SharedApi{};
}

void free_packed(PackedExperts &p) {
    if (p.d_w_contiguous) {
        CHECK_CUDA(cudaFree(p.d_w_contiguous));
    } else {
        for (void *v : p.d_w_active) {
            if (v) CHECK_CUDA(cudaFree(v));
        }
    }
    if (p.d_s_contiguous) {
        CHECK_CUDA(cudaFree(p.d_s_contiguous));
    } else {
        for (void *v : p.d_s_active) {
            if (v) CHECK_CUDA(cudaFree(v));
        }
    }
    if (p.d_w_table) CHECK_CUDA(cudaFree(p.d_w_table));
    if (p.d_s_table) CHECK_CUDA(cudaFree(p.d_s_table));
    p = PackedExperts{};
}

int pack_descriptor_set(int device, const TmIndexEntry &entry, int rank,
                        const std::vector<int> &active, const char *pack_dir,
                        PackedExperts *out, uint64_t *host_bytes_read) {
    CHECK_CUDA(cudaSetDevice(device));
    const std::string sidecar_path = path_join(pack_dir, entry.sidecar_file);
    out->d_w_active.assign(active.size(), nullptr);
    out->d_s_active.assign(active.size(), nullptr);
    out->k_pack = entry.k_pack;

    if (active.empty()) return 1;
    const size_t total_weight_bytes = entry.weight_bytes_per_expert * active.size();
    const size_t total_scale_bytes = entry.scale_bytes_per_expert * active.size();
    cudaError_t alloc_rc = cudaMalloc(&out->d_w_contiguous, total_weight_bytes);
    if (alloc_rc == cudaSuccess) {
        alloc_rc = cudaMalloc(&out->d_s_contiguous, total_scale_bytes);
        if (alloc_rc != cudaSuccess) {
            CHECK_CUDA(cudaFree(out->d_w_contiguous));
            out->d_w_contiguous = nullptr;
        }
    }
    if (alloc_rc != cudaSuccess) {
        (void)cudaGetLastError();
        out->d_w_contiguous = nullptr;
        out->d_s_contiguous = nullptr;
    }

    std::vector<uint8_t> h_weight(entry.weight_bytes_per_expert);
    std::vector<uint8_t> h_scale(entry.scale_bytes_per_expert);
    for (size_t i = 0; i < active.size(); ++i) {
        const int global_expert = rank * kLocalExperts + active[i];
        const uint64_t w_off = entry.weight_offset +
                               (uint64_t)global_expert * entry.weight_bytes_per_expert;
        const uint64_t s_off = entry.scale_offset +
                               (uint64_t)global_expert * entry.scale_bytes_per_expert;
        if (read_exact_at(sidecar_path, w_off, h_weight.data(), h_weight.size()) != 0 ||
            read_exact_at(sidecar_path, s_off, h_scale.data(), h_scale.size()) != 0) {
            return 1;
        }
        if (out->d_w_contiguous && out->d_s_contiguous) {
            out->d_w_active[i] = static_cast<uint8_t *>(out->d_w_contiguous) +
                                 i * entry.weight_bytes_per_expert;
            out->d_s_active[i] = static_cast<uint8_t *>(out->d_s_contiguous) +
                                 i * entry.scale_bytes_per_expert;
        } else {
            CHECK_CUDA(cudaMalloc(&out->d_w_active[i], entry.weight_bytes_per_expert));
            CHECK_CUDA(cudaMalloc(&out->d_s_active[i], entry.scale_bytes_per_expert));
        }
        CHECK_CUDA(cudaMemcpy(out->d_w_active[i], h_weight.data(), h_weight.size(),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_s_active[i], h_scale.data(), h_scale.size(),
                              cudaMemcpyHostToDevice));
        *host_bytes_read += (uint64_t)h_weight.size() + (uint64_t)h_scale.size();
    }

    std::vector<StridedPtrH> w_table((size_t)kLocalExperts);
    std::vector<StridedPtrH> s_table((size_t)kLocalExperts);
    for (int e = 0; e < kLocalExperts; ++e) {
        w_table[(size_t)e] = StridedPtrH{out->d_w_active[0], entry.weight_stride};
        s_table[(size_t)e] = StridedPtrH{out->d_s_active[0], entry.scale_stride};
    }
    for (size_t i = 0; i < active.size(); ++i) {
        w_table[(size_t)active[i]] = StridedPtrH{out->d_w_active[i], entry.weight_stride};
        s_table[(size_t)active[i]] = StridedPtrH{out->d_s_active[i], entry.scale_stride};
    }
    CHECK_CUDA(cudaMalloc(&out->d_w_table, w_table.size() * sizeof(StridedPtrH)));
    CHECK_CUDA(cudaMemcpy(out->d_w_table, w_table.data(),
                          w_table.size() * sizeof(StridedPtrH), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&out->d_s_table, s_table.size() * sizeof(StridedPtrH)));
    CHECK_CUDA(cudaMemcpy(out->d_s_table, s_table.data(),
                          s_table.size() * sizeof(StridedPtrH), cudaMemcpyHostToDevice));
    return 0;
}

void free_layer_expert_cache(LayerExpertCache *layer) {
    if (!layer) return;
    for (int p = 0; p < kGpus; ++p) {
        free_packed(layer->gated[p]);
        free_packed(layer->down[p]);
    }
    *layer = LayerExpertCache{};
}

void close_shared_expert_bindings(SharedExpertBindings *shared);

int open_shared_expert_bindings(const Options &opt, SharedExpertBindings *shared) {
    std::vector<int> active;
    for (int e = 0; e < kPackedLocalExperts; ++e) active.push_back(e);
    const auto start = std::chrono::steady_clock::now();

    for (int layer = 0; layer < 43; ++layer) {
        if (opt.resident_profile_layer >= 0 && layer != opt.resident_profile_layer) {
            continue;
        }
        LayerExpertCache &cache = shared->layers[layer];
        if (parse_tm_index(opt.tm_index_path, layer, &cache.bindings) != 0) {
            std::fprintf(stderr, "tm index parse failed for layer %d\n", layer);
            close_shared_expert_bindings(shared);
            return 1;
        }
        uint64_t layer_bytes_by_gpu[kGpus] = {};
        if (opt.parallel_expert_load_gate) {
            int rc[kGpus] = {};
            std::thread workers[kGpus];
            for (int p = 0; p < kGpus; ++p) {
                workers[p] = std::thread([&, p]() {
                    uint64_t layer_bytes = 0;
                    int local_rc = pack_descriptor_set(
                        opt.devices[p], cache.bindings.gated, p, active,
                        opt.pack_dir, &cache.gated[p], &layer_bytes);
                    if (local_rc == 0) {
                        local_rc = pack_descriptor_set(
                            opt.devices[p], cache.bindings.down, p, active,
                            opt.pack_dir, &cache.down[p], &layer_bytes);
                    }
                    layer_bytes_by_gpu[p] = layer_bytes;
                    rc[p] = local_rc;
                });
            }
            for (int p = 0; p < kGpus; ++p) workers[p].join();
            for (int p = 0; p < kGpus; ++p) {
                if (rc[p] != 0) {
                    close_shared_expert_bindings(shared);
                    return 2;
                }
                cache.bytes += layer_bytes_by_gpu[p];
                shared->bytes += layer_bytes_by_gpu[p];
            }
        } else {
            for (int p = 0; p < kGpus; ++p) {
                uint64_t layer_bytes = 0;
                if (pack_descriptor_set(opt.devices[p], cache.bindings.gated, p, active,
                                        opt.pack_dir, &cache.gated[p], &layer_bytes) != 0 ||
                    pack_descriptor_set(opt.devices[p], cache.bindings.down, p, active,
                                        opt.pack_dir, &cache.down[p], &layer_bytes) != 0) {
                    close_shared_expert_bindings(shared);
                    return 2;
                }
                cache.bytes += layer_bytes;
                shared->bytes += layer_bytes;
            }
        }
        if (opt.parallel_expert_load_gate) {
            std::printf("tp_ep_parallel_expert_load_layer\tlayer\t%d\tbytes\t%llu\tPASS\n",
                        layer, (unsigned long long)cache.bytes);
            std::fflush(stdout);
        }
        cache.initialized = true;
    }
    shared->initialized = true;
    const auto stop = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    std::printf("tp_ep_shared_expert_bindings_load\tlayers\t43\tparallel\t%d\t"
                "bytes\t%llu\tload_ms\t%.6f\tPASS\n",
                opt.parallel_expert_load_gate ? 1 : 0,
                (unsigned long long)shared->bytes,
                ms);
    std::fflush(stdout);
    return 0;
}

void close_shared_expert_bindings(SharedExpertBindings *shared) {
    if (!shared) return;
    for (int layer = 0; layer < 43; ++layer) {
        free_layer_expert_cache(&shared->layers[layer]);
    }
    *shared = SharedExpertBindings{};
}

int routed_executor_rows(const RankState &rank, const Options &opt) {
    int rows = rank.routes;
    if (opt.post_attention_static_executor_route_cap > 0) {
        rows = std::min(rows, opt.post_attention_static_executor_route_cap);
    }
    return rows;
}

int routed_compose_rows(const RankState &rank, const Options &opt) {
    int rows = rank.routes;
    if (opt.post_attention_static_compose_route_cap > 0) {
        rows = std::min(rows, opt.post_attention_static_compose_route_cap);
    }
    return rows;
}

