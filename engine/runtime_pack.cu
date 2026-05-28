std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (start <= line.size()) {
        const size_t tab = line.find('\t', start);
        if (tab == std::string::npos) {
            fields.emplace_back(line.substr(start));
            break;
        }
        fields.emplace_back(line.substr(start, tab - start));
        start = tab + 1;
    }
    return fields;
}

bool safe_sidecar_name(const std::string &name) {
    return !name.empty() &&
           name.find('/') == std::string::npos &&
           name.find('\\') == std::string::npos &&
           name.find("..") == std::string::npos;
}

std::string path_join(const char *dir, const std::string &base) {
    std::string out(dir ? dir : "");
    if (!out.empty() && out.back() != '/') out.push_back('/');
    out += base;
    return out;
}

int read_exact_at(const std::string &path, uint64_t offset, void *dst, size_t bytes) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "cannot open sidecar %s: %s\n", path.c_str(), std::strerror(errno));
        return 1;
    }
    if (fseeko(fp, (off_t)offset, SEEK_SET) != 0) {
        std::fprintf(stderr, "cannot seek sidecar %s offset %llu: %s\n",
                     path.c_str(), (unsigned long long)offset, std::strerror(errno));
        std::fclose(fp);
        return 2;
    }
    const size_t got = std::fread(dst, 1, bytes, fp);
    if (got != bytes) {
        std::fprintf(stderr, "short read sidecar %s offset %llu bytes %zu got %zu\n",
                     path.c_str(), (unsigned long long)offset, bytes, got);
        std::fclose(fp);
        return 3;
    }
    std::fclose(fp);
    return 0;
}


constexpr uint64_t kMiB = 1024ull * 1024ull;

bool should_report_vram(const Options &opt) {
    return opt.vram_report || opt.vram_min_free_mib > 0;
}

bool nccl_gate_active(const Options &opt) {
    return opt.nccl_reduce_scatter_compose_gate ||
           opt.tp_hc_current_input_nccl_allgather_gate ||
           opt.tp_hc_current_allreduce_gate ||
           opt.true_ds4_attention_output_nccl_allgather_gate;
}

int report_vram_checkpoint_min_free(const Options &opt,
                                    const char *label,
                                    uint64_t min_free_mib_threshold) {
    const uint64_t min_free_bytes = min_free_mib_threshold * kMiB;
    uint64_t min_free_mib = UINT64_MAX;
    uint64_t max_used_mib = 0;
    int failures = 0;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        size_t free_b = 0;
        size_t total_b = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
        const uint64_t used_b = (uint64_t)total_b - (uint64_t)free_b;
        const uint64_t free_mib = (uint64_t)free_b / kMiB;
        const uint64_t used_mib = used_b / kMiB;
        const uint64_t total_mib = (uint64_t)total_b / kMiB;
        min_free_mib = std::min(min_free_mib, free_mib);
        max_used_mib = std::max(max_used_mib, used_mib);
        const bool pass =
            min_free_mib_threshold == 0 || (uint64_t)free_b >= min_free_bytes;
        if (!pass) failures++;
        std::printf("tp_ep_vram\tlabel\t%s\tgpu\t%d\tfree_mib\t%llu\t"
                    "used_mib\t%llu\ttotal_mib\t%llu\tmin_free_mib\t%llu\t%s\n",
                    label, gpu,
                    (unsigned long long)free_mib,
                    (unsigned long long)used_mib,
                    (unsigned long long)total_mib,
                    (unsigned long long)min_free_mib_threshold,
                    pass ? "PASS" : "FAIL");
    }
    if (min_free_mib == UINT64_MAX) min_free_mib = 0;
    std::printf("tp_ep_vram_summary\tlabel\t%s\tmin_free_mib\t%llu\t"
                "max_used_mib\t%llu\tthreshold_mib\t%llu\tfailures\t%d\t%s\n",
                label,
                (unsigned long long)min_free_mib,
                (unsigned long long)max_used_mib,
                (unsigned long long)min_free_mib_threshold,
                failures,
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

int report_vram_checkpoint(const Options &opt, const char *label) {
    if (!should_report_vram(opt)) return 0;
    return report_vram_checkpoint_min_free(opt, label, opt.vram_min_free_mib);
}

int report_nccl_vram_checkpoint(const Options &opt, const char *label) {
    if (!nccl_gate_active(opt) || opt.nccl_min_free_mib == 0) return 0;
    return report_vram_checkpoint_min_free(opt, label, opt.nccl_min_free_mib);
}

int check_planned_vram_allocation(const Options &opt,
                                  const char *label,
                                  const uint64_t planned_bytes[kGpus]) {
    if (!should_report_vram(opt)) return 0;
    const uint64_t min_free_bytes = opt.vram_min_free_mib * kMiB;
    int failures = 0;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        size_t free_b = 0;
        size_t total_b = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
        const uint64_t required_b = planned_bytes[gpu] + min_free_bytes;
        const bool pass = (uint64_t)free_b >= required_b;
        if (!pass) failures++;
        std::printf("tp_ep_vram_plan\tlabel\t%s\tgpu\t%d\tfree_mib\t%llu\t"
                    "planned_mib\t%llu\tthreshold_mib\t%llu\ttotal_mib\t%llu\t%s\n",
                    label, gpu,
                    (unsigned long long)((uint64_t)free_b / kMiB),
                    (unsigned long long)(planned_bytes[gpu] / kMiB),
                    (unsigned long long)opt.vram_min_free_mib,
                    (unsigned long long)((uint64_t)total_b / kMiB),
                    pass ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}



bool parse_contract_row(const std::vector<std::string> &f, ContractRow *out) {
    if (f.size() < 23) return false;
    ContractRow r;
    r.record_type = f[0];
    r.tensor_id = f[1];
    if (!parse_int(f[3].c_str(), &r.layer)) return false;
    r.family = f[4];
    r.source_dtype = f[5];
    r.source_shape = f[6];
    r.runtime_layout = f[7];
    if (!parse_int(f[8].c_str(), &r.owning_gpu)) return false;
    if (!parse_int(f[9].c_str(), &r.tp_rank)) return false;
    if (!parse_int(f[10].c_str(), &r.ep_rank)) return false;
    if (!parse_int(f[12].c_str(), &r.shard_index)) return false;
    if (!parse_int(f[13].c_str(), &r.shard_count)) return false;
    if (!parse_int(f[14].c_str(), &r.expert_first)) return false;
    if (!parse_int(f[15].c_str(), &r.expert_count)) return false;
    if (!parse_int(f[16].c_str(), &r.kv_ratio)) return false;
    if (!parse_u64(f[17].c_str(), &r.kv_rows_per_slot)) return false;
    if (!parse_u64(f[18].c_str(), &r.bytes_estimate)) return false;
    r.source_pack_file = f[19];
    if (!parse_u64(f[20].c_str(), &r.source_shard_offset)) return false;
    if (!parse_u64(f[21].c_str(), &r.source_byte_length)) return false;
    r.kernel_family = f[22];
    if (!safe_sidecar_name(r.source_pack_file) && r.source_pack_file != "-") return false;
    *out = r;
    return true;
}

void enqueue_graph_f32_copy_between_devices(const Options &opt,
                                            int dst_device,
                                            int src_device,
                                            float *dst,
                                            const float *src,
                                            uint64_t elems,
                                            cudaStream_t stream,
                                            int block) {
    (void)dst_device;
    (void)src_device;
    (void)opt;
    copy_f32_kernel<<<(unsigned int)((elems + (uint64_t)block - 1) /
                                     (uint64_t)block),
                      block, 0, stream>>>(dst, src, elems);
    CHECK_CUDA(cudaGetLastError());
}

void enqueue_graph_f32_copy_from_device0(const Options &opt,
                                         RankState &rank_state,
                                         int /*rank*/,
                                         float *dst,
                                         const float *src,
                                         uint64_t elems,
                                         cudaStream_t stream,
                                         int block) {
    enqueue_graph_f32_copy_between_devices(opt, rank_state.device, opt.devices[0],
                                           dst, src, elems, stream, block);
}

void enqueue_graph_i32_copy_from_device0(const Options &opt,
                                         RankState &rank_state,
                                         int /*rank*/,
                                         int *dst,
                                       const int *src,
                                       uint64_t elems,
                                       cudaStream_t stream,
                                       int block) {
    (void)opt;
    (void)rank_state;
    copy_i32_kernel<<<(unsigned int)((elems + (uint64_t)block - 1) /
                                     (uint64_t)block),
                      block, 0, stream>>>(dst, src, elems);
    CHECK_CUDA(cudaGetLastError());
}

int nccl_broadcast_bytes_from_rank(RankState ranks[kGpus],
                                   int root,
                                   const void *src_root,
                                   void *dst_by_rank[kGpus],
                                   size_t bytes,
                                   const char *label) {
    if (root < 0 || root >= kGpus || !src_root || !dst_by_rank || bytes == 0) {
        return 1;
    }
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.compose_nccl_initialized || !r.compose_nccl ||
            !dst_by_rank[rank]) {
            std::fprintf(stderr,
                         "tp_ep_nccl_broadcast_missing\tlabel\t%s\t"
                         "rank\t%d\tcompose\t%d\tdst\t%d\n",
                         label ? label : "-", rank,
                         (r.compose_nccl_initialized && r.compose_nccl) ? 1 : 0,
                         dst_by_rank[rank] ? 1 : 0);
            return 2;
        }
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        const void *send = rank == root ? src_root : dst_by_rank[rank];
        CHECK_NCCL(ncclBroadcast(send, dst_by_rank[rank], bytes, ncclChar, root,
                                 r.compose_nccl, r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    CHECK_CUDA(cudaSetDevice(prior_device));
    return 0;
}

int nccl_broadcast_bytes_from_rank0(RankState ranks[kGpus],
                                    const void *src_rank0,
                                    void *dst_by_rank[kGpus],
                                    size_t bytes,
                                    const char *label) {
    return nccl_broadcast_bytes_from_rank(ranks, 0, src_rank0, dst_by_rank,
                                          bytes, label);
}

int broadcast_ep_return_slices(RankState ranks[kGpus],
                               bool fp16,
                               bool skip_self_copy,
                               uint64_t src_stride_elems,
                               const uint64_t copy_elems_by_src[kGpus],
                               const char *label) {
    if (!copy_elems_by_src || src_stride_elems == 0) return 1;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int src = 0; src < kGpus; ++src) {
        const uint64_t copy_elems = copy_elems_by_src[src];
        const uint64_t bcast_elems = (uint64_t)kGpus * src_stride_elems;
        const size_t elem_bytes = fp16 ? sizeof(__half) : sizeof(float);
        const size_t bcast_bytes = (size_t)(bcast_elems * elem_bytes);
        void *scratch_by_rank[kGpus] = {};
        for (int rank = 0; rank < kGpus; ++rank) {
            scratch_by_rank[rank] = fp16
                ? (void *)ranks[rank].d_ep_contrib_half_bcast_all
                : (void *)ranks[rank].d_ep_contrib_bcast_all;
            if (!scratch_by_rank[rank]) return 2;
        }
        const void *src_all = fp16
            ? (const void *)ranks[src].d_ep_contrib_half_all
            : (const void *)ranks[src].d_ep_contrib_all;
        if (!src_all) return 3;
        if (nccl_broadcast_bytes_from_rank(
                ranks, src, src_all, scratch_by_rank, bcast_bytes,
                label ? label : "ep_return_broadcast") != 0) {
            return 4;
        }
        if (copy_elems == 0) continue;
        const size_t copy_bytes = (size_t)(copy_elems * elem_bytes);
        for (int dst = 0; dst < kGpus; ++dst) {
            if (skip_self_copy && src == dst) continue;
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            const uint64_t offset_elems = (uint64_t)dst * src_stride_elems;
            if (fp16) {
                if (!r.d_ep_remote_half[src]) return 5;
                const __half *src_ptr =
                    r.d_ep_contrib_half_bcast_all + offset_elems;
                CHECK_CUDA(cudaMemcpyAsync(r.d_ep_remote_half[src], src_ptr,
                                           copy_bytes, cudaMemcpyDeviceToDevice,
                                           r.stream));
            } else {
                if (!r.d_ep_remote[src]) return 5;
                const float *src_ptr = r.d_ep_contrib_bcast_all + offset_elems;
                CHECK_CUDA(cudaMemcpyAsync(r.d_ep_remote[src], src_ptr,
                                           copy_bytes, cudaMemcpyDeviceToDevice,
                                           r.stream));
            }
        }
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
    return 0;
}

int parse_contract(const char *path, int layer, std::vector<ContractRow> *rows,
                   LayerStats *stats) {
    FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "cannot open contract %s: %s\n", path, std::strerror(errno));
        return 1;
    }
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
        ContractRow r;
        if (!parse_contract_row(f, &r)) {
            stats->bad_rows++;
            continue;
        }
        if (layer >= 0 && r.layer != layer) continue;
        if (r.owning_gpu < 0 || r.owning_gpu >= kGpus) {
            stats->bad_rows++;
            continue;
        }
        rows->push_back(r);
        stats->total_rows++;
        GpuFamilyStats &g = stats->gpu[r.owning_gpu];
        if (r.record_type == "dense_tp") {
            stats->dense_rows++;
            g.dense_rows++;
            g.dense_bytes += r.bytes_estimate;
        } else if (r.record_type == "replicated_control") {
            stats->control_rows++;
            g.control_rows++;
            g.control_bytes += r.bytes_estimate;
        } else if (r.record_type == "ep_expert") {
            stats->expert_rows++;
            g.expert_rows++;
            g.expert_descriptor_bytes += r.bytes_estimate;
        } else if (r.record_type == "kv_shard") {
            stats->kv_rows++;
            g.kv_rows++;
        } else if (r.record_type == "kv_comp_state") {
            stats->comp_rows++;
            g.comp_rows++;
        }
    }
    std::fclose(fp);
    return rows->empty() ? 2 : 0;
}

uint64_t physical_row_offset(const ContractRow &r) {
    if (r.record_type == "dense_tp" && r.shard_index >= 0 && r.shard_count > 1 &&
        r.source_byte_length >= r.bytes_estimate * (uint64_t)r.shard_count) {
        return r.source_shard_offset + (uint64_t)r.shard_index * r.bytes_estimate;
    }
    return r.source_shard_offset;
}

bool parse_shape2(const std::string &shape, int *cols, int *rows) {
    if (shape.size() < 5 || shape.front() != '[' || shape.back() != ']') return false;
    const size_t x = shape.find('x');
    if (x == std::string::npos) return false;
    std::string a = shape.substr(1, x - 1);
    std::string b = shape.substr(x + 1, shape.size() - x - 2);
    return parse_int(a.c_str(), cols) && parse_int(b.c_str(), rows) &&
           *cols > 0 && *rows > 0;
}

std::string layer_tensor_name(int layer, const char *suffix) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "blk.%d.%s", layer, suffix);
    return std::string(buf);
}

int ds4_layer_ratio(int layer) {
    if (layer < 2) return 0;
    return (layer % 2) == 0 ? 4 : 128;
}

int attn_comp_state_rows_for_ratio(int ratio) {
    if (ratio == 4) return 2 * ratio;
    return ratio > 0 ? ratio : 0;
}

int attn_comp_state_width_for_ratio(int ratio) {
    if (ratio == 4) return 2 * kHeadDim;
    return ratio > 0 ? kHeadDim : 0;
}

uint64_t f8_row_bytes(int cols) {
    return (uint64_t)(cols / 128) * 129ull;
}

float e8m0_to_f32_host(uint8_t e) {
    uint32_t bits = e == 0 ? 0x00400000u : ((uint32_t)e << 23);
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

float e4m3fn_to_f32_host(uint8_t x) {
    const uint8_t ax = x & 0x7fu;
    const bool sign = (x & 0x80u) != 0;
    if (ax == 0) return sign ? -0.0f : 0.0f;
    if (ax == 0x7f) return std::numeric_limits<float>::quiet_NaN();
    const int exp = (x >> 3) & 0x0f;
    const int man = x & 0x07;
    const float value = exp == 0 ? std::ldexp((float)man, -9)
                                 : std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
    return sign ? -value : value;
}

float cpu_f8_dot(const uint8_t *row, const float *x, int cols) {
    double acc = 0.0;
    const int blocks = cols / 128;
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *block = row + (uint64_t)b * 129ull;
        const float scale = e8m0_to_f32_host(block[0]);
        for (int c = 0; c < 128; ++c) {
            acc += (double)(e4m3fn_to_f32_host(block[1 + c]) * scale) *
                   (double)x[b * 128 + c];
        }
    }
    return (float)acc;
}

float bf16_to_f32_host(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float v = 0.0f;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

float cpu_bf16_dot(const uint16_t *row, const float *x, int cols) {
    double acc = 0.0;
    for (int c = 0; c < cols; ++c) {
        acc += (double)bf16_to_f32_host(row[c]) * (double)x[c];
    }
    return (float)acc;
}

int device_checksum_row(int device, const char *pack_dir, const ContractRow &r,
                        uint64_t *checksum) {
    if (r.bytes_estimate == 0 || r.source_pack_file == "-") return 0;
    CHECK_CUDA(cudaSetDevice(device));
    const uint64_t offset = physical_row_offset(r);
    if (offset + r.bytes_estimate > r.source_shard_offset + r.source_byte_length &&
        r.record_type == "dense_tp") {
        std::fprintf(stderr, "dense shard exceeds source span for %s\n", r.tensor_id.c_str());
        return 1;
    }
    std::vector<unsigned char> host((size_t)r.bytes_estimate);
    const std::string path = path_join(pack_dir, r.source_pack_file);
    if (read_exact_at(path, offset, host.data(), host.size()) != 0) return 2;

    unsigned char *d = nullptr;
    unsigned long long *d_sum = nullptr;
    CHECK_CUDA(cudaMalloc(&d, host.size()));
    CHECK_CUDA(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemcpy(d, host.data(), host.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    const int block = 256;
    const int grid = (int)std::min<uint64_t>(4096, (r.bytes_estimate + block - 1) / block);
    checksum_bytes_kernel<<<std::max(grid, 1), block>>>(d, r.bytes_estimate, d_sum);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    unsigned long long h_sum = 0;
    CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(h_sum), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d));
    CHECK_CUDA(cudaFree(d_sum));
    *checksum = (uint64_t)h_sum;
    return 0;
}

bool select_dense_rows(const std::vector<ContractRow> &rows,
                       const char *tensor,
                       std::vector<ContractRow> *selected,
                       int *cols,
                       int *total_rows) {
    selected->clear();
    for (const ContractRow &r : rows) {
        if (r.record_type == "dense_tp" && r.tensor_id == tensor) selected->push_back(r);
    }
    if ((int)selected->size() != kGpus) return false;
    std::sort(selected->begin(), selected->end(),
              [](const ContractRow &a, const ContractRow &b) {
                  return a.owning_gpu < b.owning_gpu;
              });
    int parsed_cols = 0;
    int parsed_rows = 0;
    if (!parse_shape2((*selected)[0].source_shape, &parsed_cols, &parsed_rows)) return false;
    for (int i = 0; i < kGpus; ++i) {
        const ContractRow &r = (*selected)[i];
        if (r.owning_gpu != i ||
            r.tp_rank != i ||
            r.shard_index != i ||
            r.shard_count != kGpus ||
            r.source_dtype != "f8_e4m3_b128" ||
            r.source_shape != (*selected)[0].source_shape) {
            return false;
        }
    }
    if (parsed_cols % 128 != 0 || parsed_rows % kGpus != 0) return false;
    const uint64_t row_bytes = f8_row_bytes(parsed_cols);
    const uint64_t rows_per_gpu = (uint64_t)parsed_rows / kGpus;
    for (const ContractRow &r : *selected) {
        if (r.bytes_estimate != row_bytes * rows_per_gpu) return false;
    }
    *cols = parsed_cols;
    *total_rows = parsed_rows;
    return true;
}

std::vector<std::string> discover_f8_dense_tensors(const std::vector<ContractRow> &rows) {
    std::vector<std::string> out;
    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" || r.source_dtype != "f8_e4m3_b128") continue;
        if (std::find(out.begin(), out.end(), r.tensor_id) == out.end()) {
            out.push_back(r.tensor_id);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool select_bf16_dense_rows(const std::vector<ContractRow> &rows,
                            const char *tensor,
                            std::vector<ContractRow> *selected,
                            int *cols,
                            int *total_rows) {
    selected->clear();
    for (const ContractRow &r : rows) {
        if (r.record_type == "dense_tp" && r.tensor_id == tensor) selected->push_back(r);
    }
    if ((int)selected->size() != kGpus) return false;
    std::sort(selected->begin(), selected->end(),
              [](const ContractRow &a, const ContractRow &b) {
                  return a.owning_gpu < b.owning_gpu;
              });
    int parsed_cols = 0;
    int parsed_rows = 0;
    if (!parse_shape2((*selected)[0].source_shape, &parsed_cols, &parsed_rows)) return false;
    for (int i = 0; i < kGpus; ++i) {
        const ContractRow &r = (*selected)[i];
        if (r.owning_gpu != i ||
            r.tp_rank != i ||
            r.shard_index != i ||
            r.shard_count != kGpus ||
            r.source_dtype != "bf16" ||
            r.source_shape != (*selected)[0].source_shape) {
            return false;
        }
    }
    if (parsed_rows % kGpus != 0) return false;
    const uint64_t rows_per_gpu = (uint64_t)parsed_rows / kGpus;
    const uint64_t shard_bytes = rows_per_gpu * (uint64_t)parsed_cols * sizeof(uint16_t);
    for (const ContractRow &r : *selected) {
        if (r.bytes_estimate != shard_bytes) return false;
    }
    *cols = parsed_cols;
    *total_rows = parsed_rows;
    return true;
}

std::vector<std::string> discover_bf16_dense_tensors(const std::vector<ContractRow> &rows) {
    std::vector<std::string> out;
    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" || r.source_dtype != "bf16") continue;
        if (std::find(out.begin(), out.end(), r.tensor_id) == out.end()) {
            out.push_back(r.tensor_id);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

int run_dense_compute_gate(const Options &opt,
                           const std::vector<ContractRow> &rows,
                           const char *tensor,
                           DenseComputeStats *stats) {
    if (!tensor) return 0;
    stats->enabled = true;
    stats->tensor_id = tensor;
    stats->slots = opt.slots;

    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    if (!select_dense_rows(rows, tensor, &selected, &cols, &total_rows)) {
        std::fprintf(stderr, "dense compute tensor validation failed for %s\n",
                     tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    const uint64_t row_bytes = f8_row_bytes(cols);
    const uint64_t shard_bytes = row_bytes * (uint64_t)rows_per_gpu;
    stats->rows_per_gpu = rows_per_gpu;
    stats->cols = cols;

    std::vector<float> h_x((size_t)opt.slots * cols);
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int c = 0; c < cols; ++c) {
            const int m = (slot * 17 + c * 13) % 257;
            h_x[(size_t)slot * cols + c] = ((float)m - 128.0f) * 0.00025f;
        }
    }

    double worst_ms = 0.0;
    std::vector<std::vector<uint8_t>> host_weights((size_t)kGpus);
    std::vector<std::vector<float>> host_outputs((size_t)kGpus);

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = selected[(size_t)gpu];
        host_weights[(size_t)gpu].resize((size_t)shard_bytes);
        host_outputs[(size_t)gpu].resize((size_t)opt.slots * rows_per_gpu);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r),
                          host_weights[(size_t)gpu].data(), (size_t)shard_bytes) != 0) {
            return 2;
        }
        stats->loaded_bytes += shard_bytes;

        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        uint8_t *d_w = nullptr;
        float *d_x = nullptr;
        float *d_out1 = nullptr;
        float *d_out2 = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaMalloc(&d_w, (size_t)shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out1, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out2, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_w, host_weights[(size_t)gpu].data(), (size_t)shard_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        const dim3 grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1);
        for (int i = 0; i < opt.warmup; ++i) {
            f8_b128_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                                cols, (uint32_t)row_bytes, opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < opt.iters; ++i) {
            f8_b128_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                                cols, (uint32_t)row_bytes, opt.slots);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        worst_ms = std::max(worst_ms, (double)ms / opt.iters);

        f8_b128_dense_kernel<<<grid, 256>>>(d_out2, d_w, d_x, rows_per_gpu,
                                            cols, (uint32_t)row_bytes, opt.slots);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> second(host_outputs[(size_t)gpu].size());
        CHECK_CUDA(cudaMemcpy(host_outputs[(size_t)gpu].data(), d_out1,
                              host_outputs[(size_t)gpu].size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(second.data(), d_out2,
                              second.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < second.size(); ++i) {
            const float a = host_outputs[(size_t)gpu][i];
            const float b = second[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                stats->repeat_nan++;
                stats->pass = false;
                continue;
            }
            const double diff = std::fabs((double)a - (double)b);
            stats->repeat_max_abs = std::max(stats->repeat_max_abs, diff);
            if (diff > 0.0) {
                stats->repeat_bad++;
                stats->pass = false;
            }
        }

        const int sample_slots = std::min(opt.slots, 2);
        const int sample_rows = std::min(rows_per_gpu, 4);
        for (int slot = 0; slot < sample_slots; ++slot) {
            for (int row = 0; row < sample_rows; ++row) {
                const float expected =
                    cpu_f8_dot(host_weights[(size_t)gpu].data() + (uint64_t)row * row_bytes,
                               h_x.data() + (size_t)slot * cols, cols);
                const float got = host_outputs[(size_t)gpu][(size_t)slot * rows_per_gpu + row];
                const double diff = std::fabs((double)expected - (double)got);
                stats->oracle_max_abs = std::max(stats->oracle_max_abs, diff);
                const double tol = 1.0e-4 + std::fabs((double)expected) * 1.0e-4;
                if (!std::isfinite(got) || diff > tol) {
                    stats->oracle_bad++;
                    stats->pass = false;
                }
            }
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_out1));
        CHECK_CUDA(cudaFree(d_out2));
    }

    stats->compute_ms = worst_ms;
    return stats->pass ? 0 : 3;
}

int run_bf16_dense_compute_gate(const Options &opt,
                                const std::vector<ContractRow> &rows,
                                const char *tensor,
                                DenseComputeStats *stats) {
    if (!tensor) return 0;
    stats->enabled = true;
    stats->tensor_id = tensor;
    stats->slots = opt.slots;

    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    if (!select_bf16_dense_rows(rows, tensor, &selected, &cols, &total_rows)) {
        std::fprintf(stderr, "bf16 dense compute tensor validation failed for %s\n",
                     tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    const uint64_t shard_bytes = (uint64_t)rows_per_gpu * cols * sizeof(uint16_t);
    stats->rows_per_gpu = rows_per_gpu;
    stats->cols = cols;

    std::vector<float> h_x((size_t)opt.slots * cols);
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int c = 0; c < cols; ++c) {
            const int m = (slot * 19 + c * 11) % 263;
            h_x[(size_t)slot * cols + c] = ((float)m - 131.0f) * 0.00025f;
        }
    }

    double worst_ms = 0.0;
    std::vector<std::vector<uint16_t>> host_weights((size_t)kGpus);
    std::vector<std::vector<float>> host_outputs((size_t)kGpus);

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = selected[(size_t)gpu];
        host_weights[(size_t)gpu].resize((size_t)rows_per_gpu * cols);
        host_outputs[(size_t)gpu].resize((size_t)opt.slots * rows_per_gpu);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r),
                          host_weights[(size_t)gpu].data(), (size_t)shard_bytes) != 0) {
            return 2;
        }
        stats->loaded_bytes += shard_bytes;

        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        uint16_t *d_w = nullptr;
        float *d_x = nullptr;
        float *d_out1 = nullptr;
        float *d_out2 = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaMalloc(&d_w, (size_t)shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out1, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out2, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_w, host_weights[(size_t)gpu].data(), (size_t)shard_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        const dim3 grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1);
        for (int i = 0; i < opt.warmup; ++i) {
            bf16_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                             cols, cols, opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < opt.iters; ++i) {
            bf16_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                             cols, cols, opt.slots);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        worst_ms = std::max(worst_ms, (double)ms / opt.iters);

        bf16_dense_kernel<<<grid, 256>>>(d_out2, d_w, d_x, rows_per_gpu,
                                         cols, cols, opt.slots);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> second(host_outputs[(size_t)gpu].size());
        CHECK_CUDA(cudaMemcpy(host_outputs[(size_t)gpu].data(), d_out1,
                              host_outputs[(size_t)gpu].size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(second.data(), d_out2,
                              second.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < second.size(); ++i) {
            const float a = host_outputs[(size_t)gpu][i];
            const float b = second[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                stats->repeat_nan++;
                stats->pass = false;
                continue;
            }
            const double diff = std::fabs((double)a - (double)b);
            stats->repeat_max_abs = std::max(stats->repeat_max_abs, diff);
            if (diff > 0.0) {
                stats->repeat_bad++;
                stats->pass = false;
            }
        }

        const int sample_slots = std::min(opt.slots, 2);
        const int sample_rows = std::min(rows_per_gpu, 4);
        for (int slot = 0; slot < sample_slots; ++slot) {
            for (int row = 0; row < sample_rows; ++row) {
                const float expected =
                    cpu_bf16_dot(host_weights[(size_t)gpu].data() + (uint64_t)row * cols,
                                 h_x.data() + (size_t)slot * cols, cols);
                const float got = host_outputs[(size_t)gpu][(size_t)slot * rows_per_gpu + row];
                const double diff = std::fabs((double)expected - (double)got);
                stats->oracle_max_abs = std::max(stats->oracle_max_abs, diff);
                const double tol = 1.0e-4 + std::fabs((double)expected) * 1.0e-4;
                if (!std::isfinite(got) || diff > tol) {
                    stats->oracle_bad++;
                    stats->pass = false;
                }
            }
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_out1));
        CHECK_CUDA(cudaFree(d_out2));
    }

    stats->compute_ms = worst_ms;
    return stats->pass ? 0 : 3;
}

struct SharedHcControls {
    bool initialized = false;
    int slots = 0;
    int devices[kGpus] = {};
    float *d_hc = nullptr;
    float *d_hc_norm = nullptr;
    float *d_mix = nullptr;
    float *d_split = nullptr;
    float *d_current_full = nullptr;
    float *d_attn_normed = nullptr;
    float *d_q_a_full = nullptr;
    float *d_q_a_normed = nullptr;
    float *d_kv_full = nullptr;
    float *d_kv_normed = nullptr;
    float *d_ffn_normed = nullptr;
    float *d_attn_comp_kv_full = nullptr;
    float *d_attn_comp_score_full = nullptr;
    float *d_index_comp_kv_full = nullptr;
    float *d_index_comp_score_full = nullptr;
    float *d_indexer_q_full = nullptr;
    float *d_indexer_w_full = nullptr;
    float *d_attn_norm_weight[43] = {};
    float *d_attn_norm_weight_rank[43][kGpus] = {};
    float *d_q_a_norm_weight[43] = {};
    float *d_kv_a_norm_weight[43] = {};
    float *d_attn_compress_ape[43] = {};
    float *d_attn_compress_norm[43] = {};
    float *d_indexer_compress_ape[43] = {};
    float *d_indexer_compress_norm[43] = {};
    float *d_attn_sinks[43] = {};
    float *d_attn_fn[43] = {};
    float *d_attn_fn_rank[43][kGpus] = {};
    float *d_attn_base[43] = {};
    float *d_attn_base_rank[43][kGpus] = {};
    float *d_attn_scale[43] = {};
    float *d_attn_scale_rank[43][kGpus] = {};
    float *d_ffn_fn[43] = {};
    float *d_ffn_fn_rank[43][kGpus] = {};
    float *d_ffn_base[43] = {};
    float *d_ffn_base_rank[43][kGpus] = {};
    float *d_ffn_scale[43] = {};
    float *d_ffn_scale_rank[43][kGpus] = {};
    float *d_ffn_norm_weight[43] = {};
    float *d_ffn_norm_weight_rank[43][kGpus] = {};
    float *d_router_w[43] = {};
    float *d_router_w_ep[43][kGpus] = {};
    float *d_router_w_shard[43][kGpus] = {};
    float *d_router_bias[43] = {};
    int *d_router_hash[43] = {};
    uint32_t router_hash_rows[43] = {};
    float *d_router_logits = nullptr;
    int *d_router_selected = nullptr;
    float *d_router_weights = nullptr;
    uint32_t *d_router_tokens = nullptr;
    unsigned char *d_router_active = nullptr;
    cublasHandle_t router_blas = nullptr;
    RoutePlanHostWorkspace route_plan_ws;
    uint64_t control_bytes = 0;
};

int init_route_plan_host_workspace(const Options &opt,
                                   RoutePlanHostWorkspace *ws) {
    if (!ws) return 1;
    if (ws->initialized) return 0;
    ws->slots = opt.slots;
    ws->top_k = opt.top_k;
    for (int rank = 0; rank < kGpus; ++rank) {
        ws->devices[rank] = opt.devices[rank];
    }
    ws->route_capacity = (size_t)opt.slots * (size_t)opt.top_k;
    ws->compact_plan_ints =
        (size_t)kGpus * ((size_t)opt.slots * (size_t)opt.top_k +
                         (size_t)opt.slots);
    CHECK_CUDA(cudaHostAlloc(&ws->h_selected,
                             ws->route_capacity * sizeof(int),
                             cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&ws->h_weights,
                             ws->route_capacity * sizeof(float),
                             cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&ws->h_compact_plan,
                             ws->compact_plan_ints * sizeof(int),
                             cudaHostAllocDefault));
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaHostAlloc(&ws->h_offsets[rank],
                                 (size_t)(kLocalExperts + 1) * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_slots[rank],
                                 ws->route_capacity * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_weights[rank],
                                 ws->route_capacity * sizeof(float),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_index_by_slot[rank],
                                 (size_t)opt.slots * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_indices_by_slot[rank],
                                 ws->route_capacity * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_count_by_slot[rank],
                                 (size_t)opt.slots * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
        CHECK_CUDA(cudaEventCreateWithFlags(&ws->upload_done[rank],
                                            cudaEventDisableTiming));
    }
    ws->initialized = true;
    return 0;
}

void close_route_plan_host_workspace(RoutePlanHostWorkspace *ws) {
    if (!ws || !ws->initialized) return;
    if (ws->uploads_pending) {
        for (int rank = 0; rank < kGpus; ++rank) {
            if (ws->upload_done[rank]) {
                CHECK_CUDA(cudaSetDevice(ws->devices[rank]));
                CHECK_CUDA(cudaEventSynchronize(ws->upload_done[rank]));
            }
        }
        ws->uploads_pending = false;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ws->devices[rank]));
        if (ws->upload_done[rank]) CHECK_CUDA(cudaEventDestroy(ws->upload_done[rank]));
        if (ws->h_route_count_by_slot[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_count_by_slot[rank]));
        if (ws->h_route_indices_by_slot[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_indices_by_slot[rank]));
        if (ws->h_route_index_by_slot[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_index_by_slot[rank]));
        if (ws->h_route_weights[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_weights[rank]));
        if (ws->h_route_slots[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_slots[rank]));
        if (ws->h_offsets[rank]) CHECK_CUDA(cudaFreeHost(ws->h_offsets[rank]));
    }
    if (ws->h_compact_plan) CHECK_CUDA(cudaFreeHost(ws->h_compact_plan));
    if (ws->h_weights) CHECK_CUDA(cudaFreeHost(ws->h_weights));
    if (ws->h_selected) CHECK_CUDA(cudaFreeHost(ws->h_selected));
    *ws = RoutePlanHostWorkspace{};
}

bool find_replicated_control_row(const std::vector<ContractRow> &rows,
                                 const char *tensor,
                                 ContractRow *out) {
    for (const ContractRow &r : rows) {
        if (r.record_type == "replicated_control" && r.tensor_id == tensor) {
            *out = r;
            return true;
        }
    }
    return false;
}

int load_control_f32(const Options &opt,
                     const std::vector<ContractRow> &rows,
                     const char *tensor,
                     size_t elems,
                     std::vector<float> *out) {
    ContractRow r;
    if (!find_replicated_control_row(rows, tensor, &r)) {
        std::fprintf(stderr, "missing replicated control tensor %s\n", tensor);
        return 1;
    }
    if (r.source_dtype != "f32" || r.bytes_estimate != elems * sizeof(float)) {
        std::fprintf(stderr, "bad replicated control tensor %s dtype=%s bytes=%llu expected=%zu\n",
                     tensor, r.source_dtype.c_str(),
                     (unsigned long long)r.bytes_estimate, elems * sizeof(float));
        return 2;
    }
    out->resize(elems);
    const std::string path = path_join(opt.pack_dir, r.source_pack_file);
    if (read_exact_at(path, physical_row_offset(r), out->data(), elems * sizeof(float)) != 0) {
        return 3;
    }
    return 0;
}

int load_optional_control_f32(const Options &opt,
                              const std::vector<ContractRow> &rows,
                              const char *tensor,
                              size_t elems,
                              std::vector<float> *out,
                              bool *found) {
    ContractRow r;
    if (!find_replicated_control_row(rows, tensor, &r)) {
        out->clear();
        if (found) *found = false;
        return 0;
    }
    if (found) *found = true;
    return load_control_f32(opt, rows, tensor, elems, out);
}

int load_optional_control_i32(const Options &opt,
                              const std::vector<ContractRow> &rows,
                              const char *tensor,
                              size_t elems,
                              std::vector<int> *out,
                              bool *found) {
    ContractRow r;
    if (!find_replicated_control_row(rows, tensor, &r)) {
        out->clear();
        if (found) *found = false;
        return 0;
    }
    if (found) *found = true;
    if (r.source_dtype != "i32" || r.bytes_estimate != elems * sizeof(int)) {
        std::fprintf(stderr, "bad replicated control tensor %s dtype=%s bytes=%llu expected=%zu\n",
                     tensor, r.source_dtype.c_str(),
                     (unsigned long long)r.bytes_estimate, elems * sizeof(int));
        return 2;
    }
    out->resize(elems);
    const std::string path = path_join(opt.pack_dir, r.source_pack_file);
    if (read_exact_at(path, physical_row_offset(r), out->data(), elems * sizeof(int)) != 0) {
        return 3;
    }
    return 0;
}

int open_shared_hc_controls(const Options &opt,
                            const std::vector<ContractRow> &rows,
                            SharedHcControls *out) {
    out->slots = opt.slots;
    for (int rank = 0; rank < kGpus; ++rank) out->devices[rank] = opt.devices[rank];
    const uint64_t hc_elems = (uint64_t)opt.slots * kHcRows * (uint64_t)kHidden;
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    cublasStatus_t blas_status = cublasCreate(&out->router_blas);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "router cublasCreate failed status=%d\n",
                     (int)blas_status);
        return 1;
    }
    CHECK_CUDA(cudaMalloc(&out->d_hc, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_hc_norm, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_mix, (size_t)opt.slots * kHcMix * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_split, (size_t)opt.slots * kHcMix * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_current_full,
                          (size_t)opt.slots * kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_attn_normed,
                          (size_t)opt.slots * kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_q_a_full,
                          (size_t)opt.slots * 1024u * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_q_a_normed,
                          (size_t)opt.slots * 1024u * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_kv_full,
                          (size_t)opt.slots * kHeadDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_kv_normed,
                          (size_t)opt.slots * kHeadDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_ffn_normed,
                          (size_t)opt.slots * kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_attn_comp_kv_full,
                          (size_t)opt.slots * kCompWidthMax * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_attn_comp_score_full,
                          (size_t)opt.slots * kCompWidthMax * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_index_comp_kv_full,
                          (size_t)opt.slots * kIndexCompWidth * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_index_comp_score_full,
                          (size_t)opt.slots * kIndexCompWidth * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_indexer_q_full,
                          (size_t)opt.slots * kIndexerHead *
                              (size_t)kIndexerHeadDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_indexer_w_full,
                          (size_t)opt.slots * kIndexerHead * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_router_logits,
                          (size_t)opt.slots * kGlobalExperts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_router_selected,
                          (size_t)opt.slots * kModelTopK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&out->d_router_weights,
                          (size_t)opt.slots * kModelTopK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_router_tokens,
                          (size_t)opt.slots * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&out->d_router_active,
                          (size_t)opt.slots * sizeof(unsigned char)));
    CHECK_CUDA(cudaMemset(out->d_router_tokens, 0,
                          (size_t)opt.slots * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(out->d_router_active, 1,
                          (size_t)opt.slots * sizeof(unsigned char)));
    if (opt.route_plan_async_upload_gate &&
        init_route_plan_host_workspace(opt, &out->route_plan_ws) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));

    for (int layer = 0; layer < 43; ++layer) {
        std::vector<float> attn_fn;
        std::vector<float> attn_base;
        std::vector<float> attn_scale;
        std::vector<float> fn;
        std::vector<float> base;
        std::vector<float> scale;
        std::vector<float> ffn_norm_weight;
        std::vector<float> attn_norm_weight;
        std::vector<float> q_a_norm_weight;
        std::vector<float> kv_a_norm_weight;
        std::vector<float> attn_sinks;
        std::vector<float> attn_compress_ape;
        std::vector<float> attn_compress_norm;
        std::vector<float> indexer_compress_ape;
        std::vector<float> indexer_compress_norm;
        std::vector<float> router_w;
        std::vector<float> router_bias;
        std::vector<int> router_hash;
        const std::string attn_norm_name = layer_tensor_name(layer, "attn_norm.weight");
        const std::string q_a_norm_name = layer_tensor_name(layer, "attn_q_a_norm.weight");
        const std::string kv_a_norm_name = layer_tensor_name(layer, "attn_kv_a_norm.weight");
        const std::string attn_sinks_name = layer_tensor_name(layer, "attn_sinks");
        const std::string attn_compress_ape_name =
            layer_tensor_name(layer, "attn_compress_ape");
        const std::string attn_compress_norm_name =
            layer_tensor_name(layer, "attn_compress_norm.weight");
        const std::string indexer_compress_ape_name =
            layer_tensor_name(layer, "indexer.compress_ape");
        const std::string indexer_compress_norm_name =
            layer_tensor_name(layer, "indexer.compress_norm.weight");
        const std::string attn_fn_name = layer_tensor_name(layer, "hc_attn_fn");
        const std::string attn_base_name = layer_tensor_name(layer, "hc_attn_base");
        const std::string attn_scale_name = layer_tensor_name(layer, "hc_attn_scale");
        const std::string fn_name = layer_tensor_name(layer, "hc_ffn_fn");
        const std::string base_name = layer_tensor_name(layer, "hc_ffn_base");
        const std::string scale_name = layer_tensor_name(layer, "hc_ffn_scale");
        const std::string ffn_norm_name = layer_tensor_name(layer, "ffn_norm.weight");
        const std::string router_name = layer_tensor_name(layer, "ffn_gate_inp.weight");
        const std::string bias_name = layer_tensor_name(layer, "exp_probs_b");
        const std::string hash_name = layer_tensor_name(layer, "ffn_gate_tid2eid");
        const int ratio = ds4_layer_ratio(layer);
        bool have_attn_compress_ape = false;
        bool have_attn_compress_norm = false;
        bool have_indexer_compress_ape = false;
        bool have_indexer_compress_norm = false;
        bool have_bias = false;
        bool have_hash = false;
        if (load_control_f32(opt, rows, attn_fn_name.c_str(),
                             (size_t)kHcRows * (size_t)kHidden * kHcMix, &attn_fn) ||
            load_control_f32(opt, rows, attn_base_name.c_str(), kHcMix, &attn_base) ||
            load_control_f32(opt, rows, attn_scale_name.c_str(), 3, &attn_scale) ||
            load_control_f32(opt, rows, attn_norm_name.c_str(),
                             kHidden, &attn_norm_weight) ||
            load_control_f32(opt, rows, q_a_norm_name.c_str(),
                             1024, &q_a_norm_weight) ||
            load_control_f32(opt, rows, kv_a_norm_name.c_str(),
                             kHeadDim, &kv_a_norm_weight) ||
            load_control_f32(opt, rows, attn_sinks_name.c_str(),
                             kHeadCount, &attn_sinks) ||
            (ratio != 0 &&
             (load_optional_control_f32(opt, rows, attn_compress_ape_name.c_str(),
                                        (size_t)ratio *
                                            (size_t)(ratio == 4 ? kCompWidthMax
                                                               : kHeadDim),
                                        &attn_compress_ape,
                                        &have_attn_compress_ape) ||
              load_optional_control_f32(opt, rows, attn_compress_norm_name.c_str(),
                                        kHeadDim, &attn_compress_norm,
                                        &have_attn_compress_norm))) ||
            (ratio == 4 &&
             (load_optional_control_f32(opt, rows, indexer_compress_ape_name.c_str(),
                                        (size_t)ratio * (size_t)kIndexCompWidth,
                                        &indexer_compress_ape,
                                        &have_indexer_compress_ape) ||
              load_optional_control_f32(opt, rows, indexer_compress_norm_name.c_str(),
                                        kIndexerHeadDim,
                                        &indexer_compress_norm,
                                        &have_indexer_compress_norm))) ||
            load_control_f32(opt, rows, fn_name.c_str(),
                             (size_t)kHcRows * (size_t)kHidden * kHcMix, &fn) ||
            load_control_f32(opt, rows, base_name.c_str(), kHcMix, &base) ||
            load_control_f32(opt, rows, scale_name.c_str(), 3, &scale) ||
            load_control_f32(opt, rows, ffn_norm_name.c_str(),
                             kHidden, &ffn_norm_weight) ||
            load_control_f32(opt, rows, router_name.c_str(),
                             (size_t)kHidden * kGlobalExperts, &router_w) ||
            load_optional_control_f32(opt, rows, bias_name.c_str(),
                                      kGlobalExperts, &router_bias, &have_bias) ||
            load_optional_control_i32(opt, rows, hash_name.c_str(),
                                      (size_t)kRouterHashRows * kModelTopK,
                                      &router_hash, &have_hash)) {
            return 1;
        }
        CHECK_CUDA(cudaMalloc(&out->d_attn_fn[layer], attn_fn.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_base[layer], attn_base.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_scale[layer], attn_scale.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_fn[layer], fn.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_base[layer], base.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_scale[layer], scale.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_norm_weight[layer],
                              ffn_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_norm_weight[layer],
                              attn_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_q_a_norm_weight[layer],
                              q_a_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_kv_a_norm_weight[layer],
                              kv_a_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_sinks[layer],
                              attn_sinks.size() * sizeof(float)));
        if (have_attn_compress_ape && have_attn_compress_norm) {
            CHECK_CUDA(cudaMalloc(&out->d_attn_compress_ape[layer],
                                  attn_compress_ape.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&out->d_attn_compress_norm[layer],
                                  attn_compress_norm.size() * sizeof(float)));
        }
        if (have_indexer_compress_ape && have_indexer_compress_norm) {
            CHECK_CUDA(cudaMalloc(&out->d_indexer_compress_ape[layer],
                                  indexer_compress_ape.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&out->d_indexer_compress_norm[layer],
                                  indexer_compress_norm.size() * sizeof(float)));
        }
        if (!opt.model_router_rank_major_logits_gate &&
            !opt.model_router_allreduce_logits_gate) {
            CHECK_CUDA(cudaMalloc(&out->d_router_w[layer],
                                  router_w.size() * sizeof(float)));
        }
        if (opt.tp_hc_current_allreduce_gate) {
            const int shard_cols = kHidden / kGpus;
            const size_t local_cols = (size_t)kHcRows * (size_t)shard_cols;
            std::vector<float> fn_rank(local_cols * (size_t)kHcMix);
            std::vector<float> attn_fn_rank(local_cols * (size_t)kHcMix);
            for (int rank = 0; rank < kGpus; ++rank) {
                for (int row = 0; row < kHcRows; ++row) {
                    for (int local_h = 0; local_h < shard_cols; ++local_h) {
                        const size_t local_c =
                            (size_t)row * (size_t)shard_cols + (size_t)local_h;
                        const size_t global_c =
                            (size_t)row * (size_t)kHidden +
                            (size_t)rank * (size_t)shard_cols +
                            (size_t)local_h;
                        for (int mix = 0; mix < kHcMix; ++mix) {
                            attn_fn_rank[local_c * (size_t)kHcMix + (size_t)mix] =
                                attn_fn[global_c * (size_t)kHcMix + (size_t)mix];
                            fn_rank[local_c * (size_t)kHcMix + (size_t)mix] =
                                fn[global_c * (size_t)kHcMix + (size_t)mix];
                        }
                    }
                }
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_attn_fn_rank[layer][rank],
                                      attn_fn_rank.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_attn_fn_rank[layer][rank],
                                      attn_fn_rank.data(),
                                      attn_fn_rank.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_attn_base_rank[layer][rank],
                                      attn_base.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_attn_base_rank[layer][rank],
                                      attn_base.data(),
                                      attn_base.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_attn_scale_rank[layer][rank],
                                      attn_scale.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_attn_scale_rank[layer][rank],
                                      attn_scale.data(),
                                      attn_scale.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_ffn_fn_rank[layer][rank],
                                      fn_rank.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_ffn_fn_rank[layer][rank],
                                      fn_rank.data(),
                                      fn_rank.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_ffn_base_rank[layer][rank],
                                      base.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_ffn_base_rank[layer][rank],
                                      base.data(),
                                      base.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_ffn_scale_rank[layer][rank],
                                      scale.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_ffn_scale_rank[layer][rank],
                                      scale.data(),
                                      scale.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        if (opt.model_router_rank_major_logits_gate) {
            std::vector<float> router_w_ep((size_t)kHidden * (size_t)kLocalExperts);
            for (int rank = 0; rank < kGpus; ++rank) {
                for (int h = 0; h < kHidden; ++h) {
                    for (int e = 0; e < kLocalExperts; ++e) {
                        router_w_ep[(size_t)h * (size_t)kLocalExperts + (size_t)e] =
                            router_w[(size_t)h * (size_t)kGlobalExperts +
                                     (size_t)(rank * kLocalExperts + e)];
                    }
                }
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_router_w_ep[layer][rank],
                                      router_w_ep.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_router_w_ep[layer][rank],
                                      router_w_ep.data(),
                                      router_w_ep.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        if (opt.model_router_allreduce_logits_gate) {
            const int shard_cols = kHidden / kGpus;
            std::vector<float> router_w_shard(
                (size_t)shard_cols * (size_t)kGlobalExperts);
            for (int rank = 0; rank < kGpus; ++rank) {
                for (int local_h = 0; local_h < shard_cols; ++local_h) {
                    const int global_h = rank * shard_cols + local_h;
                    for (int expert = 0; expert < kGlobalExperts; ++expert) {
                        router_w_shard[(size_t)local_h *
                                           (size_t)kGlobalExperts +
                                       (size_t)expert] =
                            router_w[(size_t)global_h *
                                         (size_t)kGlobalExperts +
                                     (size_t)expert];
                    }
                }
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_router_w_shard[layer][rank],
                                      router_w_shard.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_router_w_shard[layer][rank],
                                      router_w_shard.data(),
                                      router_w_shard.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        if (have_bias) {
            CHECK_CUDA(cudaMalloc(&out->d_router_bias[layer],
                                  router_bias.size() * sizeof(float)));
        }
        if (have_hash) {
            CHECK_CUDA(cudaMalloc(&out->d_router_hash[layer],
                                  router_hash.size() * sizeof(int)));
            out->router_hash_rows[layer] = kRouterHashRows;
        }
        CHECK_CUDA(cudaMemcpy(out->d_attn_fn[layer], attn_fn.data(),
                              attn_fn.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_attn_base[layer], attn_base.data(),
                              attn_base.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_attn_scale[layer], attn_scale.data(),
                              attn_scale.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_fn[layer], fn.data(),
                              fn.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_base[layer], base.data(),
                              base.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_scale[layer], scale.data(),
                              scale.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_norm_weight[layer], ffn_norm_weight.data(),
                              ffn_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        out->d_ffn_norm_weight_rank[layer][0] = out->d_ffn_norm_weight[layer];
        if (opt.model_router_allreduce_logits_gate ||
            opt.routed_ffn_rank_major_input_gate ||
            opt.routed_ffn_rank_major_shared_input_gate ||
            opt.routed_ffn_rank_major_route_input_gate ||
            opt.routed_ffn_rank_major_input_parity_gate) {
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_ffn_norm_weight_rank[layer][rank],
                                      ffn_norm_weight.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_ffn_norm_weight_rank[layer][rank],
                                      ffn_norm_weight.data(),
                                      ffn_norm_weight.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        CHECK_CUDA(cudaMemcpy(out->d_attn_norm_weight[layer], attn_norm_weight.data(),
                              attn_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        out->d_attn_norm_weight_rank[layer][0] = out->d_attn_norm_weight[layer];
        if (opt.true_ds4_attention_projection_rank_local_input_gate ||
            opt.true_ds4_attention_projection_rank_major_input_gate) {
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_attn_norm_weight_rank[layer][rank],
                                      attn_norm_weight.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_attn_norm_weight_rank[layer][rank],
                                      attn_norm_weight.data(),
                                      attn_norm_weight.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        CHECK_CUDA(cudaMemcpy(out->d_q_a_norm_weight[layer], q_a_norm_weight.data(),
                              q_a_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_kv_a_norm_weight[layer], kv_a_norm_weight.data(),
                              kv_a_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_attn_sinks[layer], attn_sinks.data(),
                              attn_sinks.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (out->d_attn_compress_ape[layer]) {
            CHECK_CUDA(cudaMemcpy(out->d_attn_compress_ape[layer],
                                  attn_compress_ape.data(),
                                  attn_compress_ape.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(out->d_attn_compress_norm[layer],
                                  attn_compress_norm.data(),
                                  attn_compress_norm.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (out->d_indexer_compress_ape[layer]) {
            CHECK_CUDA(cudaMemcpy(out->d_indexer_compress_ape[layer],
                                  indexer_compress_ape.data(),
                                  indexer_compress_ape.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(out->d_indexer_compress_norm[layer],
                                  indexer_compress_norm.data(),
                                  indexer_compress_norm.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (out->d_router_w[layer]) {
            CHECK_CUDA(cudaMemcpy(out->d_router_w[layer], router_w.data(),
                                  router_w.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (have_bias) {
            CHECK_CUDA(cudaMemcpy(out->d_router_bias[layer], router_bias.data(),
                                  router_bias.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (have_hash) {
            CHECK_CUDA(cudaMemcpy(out->d_router_hash[layer], router_hash.data(),
                                  router_hash.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }
        out->control_bytes +=
            (attn_fn.size() + attn_base.size() + attn_scale.size() +
             attn_norm_weight.size() + q_a_norm_weight.size() +
             kv_a_norm_weight.size() + attn_sinks.size() +
             attn_compress_ape.size() + attn_compress_norm.size() +
             indexer_compress_ape.size() + indexer_compress_norm.size() +
             fn.size() + base.size() + scale.size() +
             ffn_norm_weight.size() + router_w.size() + router_bias.size()) *
                sizeof(float) +
            router_hash.size() * sizeof(int);
    }
    out->initialized = true;
    return 0;
}

void close_shared_hc_controls(const Options &opt, SharedHcControls *out) {
    if (!out || !out->initialized) return;
    close_route_plan_host_workspace(&out->route_plan_ws);
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    if (out->router_blas) {
        cublasStatus_t st = cublasDestroy(out->router_blas);
        if (st != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "router cublasDestroy failed status=%d\n", (int)st);
        }
    }
    for (int layer = 0; layer < 43; ++layer) {
        for (int rank = 1; rank < kGpus; ++rank) {
            if (out->d_attn_norm_weight_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_attn_norm_weight_rank[layer][rank]));
            }
            if (out->d_ffn_norm_weight_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_ffn_norm_weight_rank[layer][rank]));
            }
        }
        for (int rank = 0; rank < kGpus; ++rank) {
            if (out->d_attn_fn_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_attn_fn_rank[layer][rank]));
            }
            if (out->d_attn_base_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_attn_base_rank[layer][rank]));
            }
            if (out->d_attn_scale_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_attn_scale_rank[layer][rank]));
            }
            if (out->d_ffn_fn_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_ffn_fn_rank[layer][rank]));
            }
            if (out->d_ffn_base_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_ffn_base_rank[layer][rank]));
            }
            if (out->d_ffn_scale_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_ffn_scale_rank[layer][rank]));
            }
        }
        if (out->d_router_hash[layer]) CHECK_CUDA(cudaFree(out->d_router_hash[layer]));
        if (out->d_router_bias[layer]) CHECK_CUDA(cudaFree(out->d_router_bias[layer]));
        for (int rank = 0; rank < kGpus; ++rank) {
            if (out->d_router_w_ep[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_router_w_ep[layer][rank]));
            }
            if (out->d_router_w_shard[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_router_w_shard[layer][rank]));
            }
        }
        CHECK_CUDA(cudaSetDevice(out->devices[0]));
        if (out->d_router_w[layer]) CHECK_CUDA(cudaFree(out->d_router_w[layer]));
        if (out->d_indexer_compress_norm[layer]) CHECK_CUDA(cudaFree(out->d_indexer_compress_norm[layer]));
        if (out->d_indexer_compress_ape[layer]) CHECK_CUDA(cudaFree(out->d_indexer_compress_ape[layer]));
        if (out->d_attn_compress_norm[layer]) CHECK_CUDA(cudaFree(out->d_attn_compress_norm[layer]));
        if (out->d_attn_compress_ape[layer]) CHECK_CUDA(cudaFree(out->d_attn_compress_ape[layer]));
        if (out->d_attn_sinks[layer]) CHECK_CUDA(cudaFree(out->d_attn_sinks[layer]));
        if (out->d_kv_a_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_kv_a_norm_weight[layer]));
        if (out->d_q_a_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_q_a_norm_weight[layer]));
        if (out->d_attn_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_attn_norm_weight[layer]));
        if (out->d_ffn_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_ffn_norm_weight[layer]));
        if (out->d_ffn_scale[layer]) CHECK_CUDA(cudaFree(out->d_ffn_scale[layer]));
        if (out->d_ffn_base[layer]) CHECK_CUDA(cudaFree(out->d_ffn_base[layer]));
        if (out->d_ffn_fn[layer]) CHECK_CUDA(cudaFree(out->d_ffn_fn[layer]));
        if (out->d_attn_scale[layer]) CHECK_CUDA(cudaFree(out->d_attn_scale[layer]));
        if (out->d_attn_base[layer]) CHECK_CUDA(cudaFree(out->d_attn_base[layer]));
        if (out->d_attn_fn[layer]) CHECK_CUDA(cudaFree(out->d_attn_fn[layer]));
    }
    if (out->d_router_weights) CHECK_CUDA(cudaFree(out->d_router_weights));
    if (out->d_router_selected) CHECK_CUDA(cudaFree(out->d_router_selected));
    if (out->d_router_logits) CHECK_CUDA(cudaFree(out->d_router_logits));
    if (out->d_router_tokens) CHECK_CUDA(cudaFree(out->d_router_tokens));
    if (out->d_router_active) CHECK_CUDA(cudaFree(out->d_router_active));
    if (out->d_index_comp_score_full) CHECK_CUDA(cudaFree(out->d_index_comp_score_full));
    if (out->d_index_comp_kv_full) CHECK_CUDA(cudaFree(out->d_index_comp_kv_full));
    if (out->d_indexer_w_full) CHECK_CUDA(cudaFree(out->d_indexer_w_full));
    if (out->d_indexer_q_full) CHECK_CUDA(cudaFree(out->d_indexer_q_full));
    if (out->d_attn_comp_score_full) CHECK_CUDA(cudaFree(out->d_attn_comp_score_full));
    if (out->d_attn_comp_kv_full) CHECK_CUDA(cudaFree(out->d_attn_comp_kv_full));
    if (out->d_ffn_normed) CHECK_CUDA(cudaFree(out->d_ffn_normed));
    if (out->d_kv_normed) CHECK_CUDA(cudaFree(out->d_kv_normed));
    if (out->d_kv_full) CHECK_CUDA(cudaFree(out->d_kv_full));
    if (out->d_q_a_normed) CHECK_CUDA(cudaFree(out->d_q_a_normed));
    if (out->d_q_a_full) CHECK_CUDA(cudaFree(out->d_q_a_full));
    if (out->d_attn_normed) CHECK_CUDA(cudaFree(out->d_attn_normed));
    if (out->d_current_full) CHECK_CUDA(cudaFree(out->d_current_full));
    if (out->d_split) CHECK_CUDA(cudaFree(out->d_split));
    if (out->d_mix) CHECK_CUDA(cudaFree(out->d_mix));
    if (out->d_hc_norm) CHECK_CUDA(cudaFree(out->d_hc_norm));
    if (out->d_hc) CHECK_CUDA(cudaFree(out->d_hc));
    *out = SharedHcControls{};
}

int upload_model_router_route_plan(const Options &opt,
                                   RankState ranks[kGpus],
                                   const std::vector<int> &selected,
                                   const std::vector<float> &weights);
int upload_model_router_route_plan_async(const Options &opt,
                                         RankState ranks[kGpus],
                                         const int *selected,
                                         const float *weights,
                                         RoutePlanHostWorkspace *ws);
int upload_model_router_route_plan_gpu(const Options &opt,
                                       SharedHcControls *hc,
                                       RankState ranks[kGpus]);
int enqueue_dense_wait_after_rank_stream(RankState ranks[kGpus]);
int enqueue_control_wait_after_rank_streams(const Options &opt,
                                            RankState ranks[kGpus],
                                            cudaStream_t control_stream);
int enqueue_control_wait_after_dense_streams(const Options &opt,
                                             RankState ranks[kGpus],
                                             cudaStream_t control_stream);

int next_graph_order_event_slot(RankState ranks[kGpus]) {
    const int slot = ranks[0].graph_event_cursor % kGraphOrderEventSlots;
    ranks[0].graph_event_cursor =
        (ranks[0].graph_event_cursor + 1) % kGraphOrderEventSlots;
    return slot;
}

cudaEvent_t graph_stream_done_event(RankState &r, int slot) {
    cudaEvent_t ev = r.graph_stream_done[slot % kGraphOrderEventSlots];
    return ev ? ev : r.stream_done;
}

cudaEvent_t graph_dense_done_event(RankState &r, int slot) {
    cudaEvent_t ev = r.graph_dense_done[slot % kGraphOrderEventSlots];
    return ev ? ev : r.dense_done;
}

