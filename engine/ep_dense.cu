int fill_shared_ffn_inputs_from_normed(const Options &opt,
                                       const SharedHcControls *hc,
                                       const ResidentF8Dense &gate,
                                       const ResidentF8Dense &up,
                                       RankState ranks[kGpus]) {
    if (!hc || !hc->d_ffn_normed) return 1;
    if (gate.cols != kHidden || up.cols != kHidden ||
        gate.rows_per_gpu != kMid / kGpus ||
        up.rows_per_gpu != kMid / kGpus) {
        return 2;
    }
    const uint64_t full_elems = (uint64_t)opt.slots * kHidden;
    const uint64_t x_elems = (uint64_t)opt.slots * kHidden;
    const int block = 256;
    if (nccl_broadcast_f32_from_device0_to_current_full(
            opt, ranks, hc->d_ffn_normed, full_elems,
            "shared_ffn_normed_input") != 0) {
        return 4;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.d_current_full) return 3;
        if (gate.d_x_half[(size_t)rank]) {
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((x_elems + block - 1) / block), block, 0,
                r.stream>>>(gate.d_x_half[(size_t)rank], r.d_current_full,
                             (uint32_t)gate.cols, (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
        if (up.d_x_half[(size_t)rank]) {
            fill_dense_input_half_from_current_kernel<<<
                (unsigned int)((x_elems + block - 1) / block), block, 0,
                r.stream>>>(up.d_x_half[(size_t)rank], r.d_current_full,
                             (uint32_t)up.cols, (uint32_t)opt.slots);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    return 0;
}

int materialize_shared_swiglu_down_input(const Options &opt,
                                         const ResidentF8Dense &gate,
                                         const ResidentF8Dense &up,
                                         const ResidentF8Dense &down,
                                         RankState ranks[kGpus]) {
    if (gate.rows_per_gpu != kMid / kGpus ||
        up.rows_per_gpu != kMid / kGpus ||
        down.cols != kMid) {
        return 1;
    }
    const uint32_t rows = (uint32_t)gate.rows_per_gpu;
    const int block = 256;
    const uint64_t shard_elems = (uint64_t)opt.slots * rows;
    const bool graph_event_order = opt.decode_cudagraph_gate;
    for (int src = 0; src < kGpus; ++src) {
        CHECK_CUDA(cudaSetDevice(ranks[src].device));
        if (!down.d_x[(size_t)src] ||
            !gate.d_out[(size_t)src] ||
            !up.d_out[(size_t)src]) {
            return 2;
        }
        shared_swiglu_shard_to_float_kernel<<<
            (unsigned int)((shard_elems + block - 1) / block), block, 0,
            ranks[src].stream>>>(down.d_x[(size_t)src],
                                 gate.d_out[(size_t)src],
                                 up.d_out[(size_t)src],
                                 (uint32_t)src, rows, (uint32_t)opt.slots,
                                 kRoutedSwigluClamp);
        CHECK_CUDA(cudaGetLastError());
    }
    if (graph_event_order) {
        if (enqueue_cross_gpu_stream_barrier(ranks, false) != 0) return 3;
    } else {
        for (int src = 0; src < kGpus; ++src) {
            CHECK_CUDA(cudaSetDevice(ranks[src].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[src].stream));
        }
    }
    const size_t width = (size_t)rows * sizeof(float);
    if (graph_event_order) {
        for (int dst = 0; dst < kGpus; ++dst) {
            CHECK_CUDA(cudaSetDevice(ranks[dst].device));
            cudaStream_t stream = ranks[dst].stream;
            for (int src = 0; src < kGpus; ++src) {
                if (src == dst) continue;
                for (int slot = 0; slot < opt.slots; ++slot) {
                    float *dst_ptr = down.d_x[(size_t)dst] +
                                     (size_t)slot * kMid + (size_t)src * rows;
                    const float *src_ptr = down.d_x[(size_t)src] +
                                           (size_t)slot * kMid + (size_t)src * rows;
                    enqueue_graph_f32_copy_between_devices(
                        opt, ranks[dst].device, ranks[src].device,
                        dst_ptr, src_ptr, (uint64_t)rows, stream, block);
                }
            }
        }
    } else {
        for (int src = 0; src < kGpus; ++src) {
            for (int slot = 0; slot < opt.slots; ++slot) {
                void *dsts[kGpus] = {};
                for (int dst = 0; dst < kGpus; ++dst) {
                    dsts[dst] = down.d_x[(size_t)dst] +
                                (size_t)slot * kMid + (size_t)src * rows;
                }
                const float *src_ptr = down.d_x[(size_t)src] +
                                       (size_t)slot * kMid + (size_t)src * rows;
                if (nccl_broadcast_bytes_from_rank(
                        ranks, src, src_ptr, dsts, width,
                        "shared_swiglu_down_input") != 0) {
                    return 5;
                }
            }
        }
    }
    if (graph_event_order) {
        if (enqueue_dense_wait_after_rank_stream(ranks) != 0) return 4;
    } else {
        for (int dst = 0; dst < kGpus; ++dst) {
            CHECK_CUDA(cudaSetDevice(ranks[dst].device));
            if (ranks[dst].copy_stream) {
                CHECK_CUDA(cudaStreamSynchronize(ranks[dst].copy_stream));
            }
            CHECK_CUDA(cudaStreamSynchronize(ranks[dst].stream));
        }
    }
    return 0;
}

int run_f8_dense_to_device(const Options &opt,
                           const std::vector<ContractRow> &rows,
                           const char *tensor,
                           int seed,
                           DeviceDenseOutputs *out) {
    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    if (!select_dense_rows(rows, tensor, &selected, &cols, &total_rows)) {
        std::fprintf(stderr, "device dense tensor validation failed for %s\n", tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    if (rows_per_gpu != kHidden / kGpus) {
        std::fprintf(stderr, "device dense tensor %s rows_per_gpu=%d expected=%d\n",
                     tensor, rows_per_gpu, kHidden / kGpus);
        return 2;
    }
    const uint64_t row_bytes = f8_row_bytes(cols);
    const uint64_t shard_bytes = row_bytes * (uint64_t)rows_per_gpu;
    out->d_out.assign((size_t)kGpus, nullptr);
    out->rows_per_gpu = rows_per_gpu;
    out->cols = cols;
    out->slots = opt.slots;

    std::vector<float> h_x((size_t)opt.slots * cols);
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int c = 0; c < cols; ++c) {
            const int m = (slot * (17 + seed) + c * (13 + seed * 3)) % 269;
            h_x[(size_t)slot * cols + c] = ((float)m - 134.0f) * 0.0002f;
        }
    }

    double worst_ms = 0.0;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = selected[(size_t)gpu];
        std::vector<uint8_t> h_w((size_t)shard_bytes);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), h_w.data(), h_w.size()) != 0) {
            free_device_dense_outputs(*out, opt);
            return 3;
        }
        out->loaded_bytes += shard_bytes;

        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        (void)cudaGetLastError();
        uint8_t *d_w = nullptr;
        float *d_x = nullptr;
        __half *d_w_half = nullptr;
        __half *d_x_half = nullptr;
        cublasHandle_t blas = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaMalloc(&d_w, (size_t)shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_out[(size_t)gpu],
                              (size_t)opt.slots * rows_per_gpu * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), (size_t)shard_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (opt.dense_f16_cublas_compose) {
            CHECK_CUDA(cudaMalloc(&d_w_half, (size_t)rows_per_gpu * cols * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&d_x_half, h_x.size() * sizeof(__half)));
            const uint64_t w_elems = (uint64_t)rows_per_gpu * cols;
            f8_b128_to_half_kernel<<<(unsigned int)((w_elems + 255) / 256), 256>>>(
                d_w_half, d_w, rows_per_gpu, cols, (uint32_t)row_bytes);
            CHECK_CUDA(cudaGetLastError());
            const uint64_t x_elems = (uint64_t)opt.slots * cols;
            cast_f32_to_half_kernel<<<(unsigned int)((x_elems + 255) / 256), 256>>>(
                d_x_half, d_x, x_elems);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            cublasStatus_t st = cublasCreate(&blas);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "cublasCreate failed on gpu %d: %d\n", gpu, (int)st);
                return 4;
            }
            (void)cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH);
        }
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        const dim3 scalar_grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1);
        const dim3 hmma_grid((unsigned int)((rows_per_gpu + 63) / 64),
                             (unsigned int)((opt.slots + 15) / 16),
                             1);
        for (int i = 0; i < opt.warmup; ++i) {
            if (opt.dense_f16_cublas_compose) {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                cublasStatus_t st = cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                                  rows_per_gpu, opt.slots, cols,
                                                  &alpha, d_w_half, CUDA_R_16F, cols,
                                                  d_x_half, CUDA_R_16F, cols,
                                                  &beta, out->d_out[(size_t)gpu],
                                                  CUDA_R_32F, rows_per_gpu,
                                                  CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS) return 5;
            } else if (opt.dense_hmma_compose) {
                f8_b128_dense_hmma_m16_kernel<<<hmma_grid, 128>>>(
                    out->d_out[(size_t)gpu], d_w, d_x, rows_per_gpu, cols,
                    (uint32_t)row_bytes, opt.slots);
            } else {
                f8_b128_dense_kernel<<<scalar_grid, 256>>>(
                    out->d_out[(size_t)gpu], d_w, d_x, rows_per_gpu, cols,
                    (uint32_t)row_bytes, opt.slots);
            }
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < opt.iters; ++i) {
            if (opt.dense_f16_cublas_compose) {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                cublasStatus_t st = cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                                  rows_per_gpu, opt.slots, cols,
                                                  &alpha, d_w_half, CUDA_R_16F, cols,
                                                  d_x_half, CUDA_R_16F, cols,
                                                  &beta, out->d_out[(size_t)gpu],
                                                  CUDA_R_32F, rows_per_gpu,
                                                  CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS) return 6;
            } else if (opt.dense_hmma_compose) {
                f8_b128_dense_hmma_m16_kernel<<<hmma_grid, 128>>>(
                    out->d_out[(size_t)gpu], d_w, d_x, rows_per_gpu, cols,
                    (uint32_t)row_bytes, opt.slots);
            } else {
                f8_b128_dense_kernel<<<scalar_grid, 256>>>(
                    out->d_out[(size_t)gpu], d_w, d_x, rows_per_gpu, cols,
                    (uint32_t)row_bytes, opt.slots);
            }
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        worst_ms = std::max(worst_ms, (double)ms / opt.iters);
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_x));
        if (d_w_half) CHECK_CUDA(cudaFree(d_w_half));
        if (d_x_half) CHECK_CUDA(cudaFree(d_x_half));
        if (blas) (void)cublasDestroy(blas);
    }
    out->compute_ms = worst_ms;
    return 0;
}

bool parse_tm_entry(const std::vector<std::string> &f, TmIndexEntry *out) {
    if (f.size() < 25) return false;
    TmIndexEntry e;
    e.semantic_tensor_id = f[0];
    e.runtime_layout = f[4];
    if (!parse_int(f[6].c_str(), &e.layer_id)) return false;
    if (!parse_int(f[8].c_str(), &e.n)) return false;
    if (!parse_int(f[9].c_str(), &e.k)) return false;
    if (!parse_int(f[10].c_str(), &e.experts_packed)) return false;
    if (!parse_int(f[11].c_str(), &e.experts_total)) return false;
    if (!parse_size(f[12].c_str(), &e.weight_bytes_per_expert)) return false;
    if (!parse_size(f[13].c_str(), &e.scale_bytes_per_expert)) return false;
    if (!parse_int(f[14].c_str(), &e.k_pack)) return false;
    if (!parse_int(f[15].c_str(), &e.weight_stride)) return false;
    if (!parse_int(f[16].c_str(), &e.scale_stride)) return false;
    e.sidecar_file = f[17];
    if (!parse_u64(f[18].c_str(), &e.weight_offset)) return false;
    if (!parse_u64(f[19].c_str(), &e.scale_offset)) return false;
    if (!safe_sidecar_name(e.sidecar_file)) return false;
    *out = e;
    return true;
}

bool valid_tm_entry(const TmIndexEntry &e, int n, int k, const char *layout) {
    return e.n == n &&
           e.k == k &&
           e.experts_total == kGlobalExperts &&
           e.experts_packed >= kGlobalExperts &&
           e.weight_bytes_per_expert > 0 &&
           e.scale_bytes_per_expert > 0 &&
           e.k_pack > 0 &&
           e.weight_stride > 0 &&
           e.scale_stride > 0 &&
           e.runtime_layout == layout;
}

