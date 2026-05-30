struct SharedOutputHead {
    bool initialized = false;
    int slots = 0;
    int vocab = 0;
    int rows_per_gpu = 0;
    ContractRow output_rows[kGpus];
    float *d_head_base_rank[kGpus] = {};
    float *d_head_scale_rank[kGpus] = {};
    float *d_head_fn_rank[kGpus] = {};
    float *d_output_norm_rank[kGpus] = {};
    float *d_reduce_max[kGpus] = {};
    float *d_reduce_sumsq[kGpus] = {};
    float *d_head_pre_rank[kGpus] = {};
    float *d_head_weights_rank[kGpus] = {};
    float *d_embd_shard[kGpus] = {};
    float *d_embd_norm_shard[kGpus] = {};
    float *d_embd_rank_major[kGpus] = {};
    uint16_t *d_w[kGpus] = {};
    float *d_x[kGpus] = {};
    float *d_logits[kGpus] = {};
    uint32_t *d_best_token[kGpus] = {};
    float *d_best_logit[kGpus] = {};
    cudaEvent_t projection_start[kGpus] = {};
    cudaEvent_t projection_stop[kGpus] = {};
    cudaStream_t stream[kGpus] = {};
    cudaEvent_t prep_ready = {};
    cudaEvent_t broadcast_ready[kGpus] = {};
    cudaEvent_t top1_done[kGpus] = {};
    uint32_t *h_best_token[kGpus] = {};
    float *h_best_logit[kGpus] = {};
    uint64_t output_weight_bytes = 0;
    uint64_t logits_bytes = 0;
};

__global__ void output_head_hc_local_max_mix4_partial_kernel(float *max_abs_out,
                                                             float *mix_out,
                                                             const float *hc_shard,
                                                             const float *fn_shard,
                                                             uint32_t slots) {
    const uint32_t op = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    constexpr uint32_t shard_cols = kHidden / kGpus;
    constexpr uint32_t local_cols = kHcRows * shard_cols;
    if (slot >= slots || op > 4u) return;
    const float *x = hc_shard + (uint64_t)slot * local_cols;
    if (op == 0u) {
        float local_max = 0.0f;
        for (uint32_t c = threadIdx.x; c < local_cols; c += blockDim.x) {
            const float v = x[c];
            if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
        }
        const float max_abs = block_max_256_f32(local_max);
        if (threadIdx.x == 0u) max_abs_out[slot] = max_abs;
        return;
    }
    const uint32_t mix = op - 1u;
    float acc = 0.0f;
    for (uint32_t c = threadIdx.x; c < local_cols; c += blockDim.x) {
        const float v = x[c];
        if (isfinite(v)) acc += fn_shard[(uint64_t)c * 4ull + mix] * v;
    }
    acc = block_sum_256_f32(acc);
    if (threadIdx.x == 0u) {
        mix_out[(uint64_t)slot * 4ull + mix] = isfinite(acc) ? acc : 0.0f;
    }
}

__global__ void output_head_hc_local_stable_sumsq_kernel(float *sumsq_out,
                                                         const float *hc_shard,
                                                         const float *global_max_abs,
                                                         uint32_t slots) {
    const uint32_t slot = blockIdx.x;
    constexpr uint32_t shard_cols = kHidden / kGpus;
    constexpr uint32_t local_cols = kHcRows * shard_cols;
    if (slot >= slots) return;
    const float max_abs = global_max_abs[slot];
    const float *x = hc_shard + (uint64_t)slot * local_cols;
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t c = threadIdx.x; c < local_cols; c += blockDim.x) {
            const float v = x[c];
            if (isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    sum = block_sum_256_f32(sum);
    if (threadIdx.x == 0u) sumsq_out[slot] = isfinite(sum) ? sum : 0.0f;
}

__global__ void output_head_weights_from_reduced_mix4_kernel(float *weights,
                                                             const float *global_max_abs,
                                                             const float *global_sumsq,
                                                             const float *global_mix,
                                                             const float *scale,
                                                             const float *base,
                                                             uint32_t slots,
                                                             float eps) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = slots * 4u;
    if (i >= n) return;
    const uint32_t slot = i >> 2u;
    const uint32_t h = i & 3u;
    const float max_abs = global_max_abs[slot];
    const float sum = global_sumsq[slot];
    float norm_scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        norm_scale =
            rsqrtf(sum / (float)(kHcRows * kHidden) +
                   eps / (max_abs * max_abs)) /
            max_abs;
    }
    const float pre = global_mix[i] * norm_scale;
    const float z = pre * scale[0] + base[h];
    weights[i] = 1.0f / (1.0f + expf(-z)) + 1.0e-6f;
}

__global__ void output_head_weighted_sum_shard_kernel(float *out,
                                                      const float *hc_shard,
                                                      const float *weights,
                                                      uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    constexpr uint32_t shard_cols = kHidden / kGpus;
    const uint64_t n = (uint64_t)slots * shard_cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / shard_cols);
    const uint32_t local_h = (uint32_t)(i - (uint64_t)slot * shard_cols);
    const uint64_t base = (uint64_t)slot * 4ull * shard_cols;
    const float w0 = weights[(uint64_t)slot * 4ull + 0ull];
    const float w1 = weights[(uint64_t)slot * 4ull + 1ull];
    const float w2 = weights[(uint64_t)slot * 4ull + 2ull];
    const float w3 = weights[(uint64_t)slot * 4ull + 3ull];
    const float v = hc_shard[base + local_h] * w0 +
                    hc_shard[base + (uint64_t)shard_cols + local_h] * w1 +
                    hc_shard[base + 2ull * (uint64_t)shard_cols + local_h] * w2 +
                    hc_shard[base + 3ull * (uint64_t)shard_cols + local_h] * w3;
    out[i] = isfinite(v) ? v : 0.0f;
}

__global__ void output_head_embd_local_max_kernel(float *max_abs_out,
                                                  const float *embd_shard,
                                                  uint32_t slots) {
    const uint32_t slot = blockIdx.x;
    constexpr uint32_t shard_cols = kHidden / kGpus;
    if (slot >= slots) return;
    const float *x = embd_shard + (uint64_t)slot * shard_cols;
    float local_max = 0.0f;
    for (uint32_t c = threadIdx.x; c < shard_cols; c += blockDim.x) {
        const float v = x[c];
        if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
    }
    const float max_abs = block_max_256_f32(local_max);
    if (threadIdx.x == 0u) max_abs_out[slot] = max_abs;
}

__global__ void output_head_embd_local_stable_sumsq_kernel(float *sumsq_out,
                                                           const float *embd_shard,
                                                           const float *global_max_abs,
                                                           uint32_t slots) {
    const uint32_t slot = blockIdx.x;
    constexpr uint32_t shard_cols = kHidden / kGpus;
    if (slot >= slots) return;
    const float max_abs = global_max_abs[slot];
    const float *x = embd_shard + (uint64_t)slot * shard_cols;
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t c = threadIdx.x; c < shard_cols; c += blockDim.x) {
            const float v = x[c];
            if (isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    sum = block_sum_256_f32(sum);
    if (threadIdx.x == 0u) sumsq_out[slot] = isfinite(sum) ? sum : 0.0f;
}

__global__ void output_head_apply_output_norm_shard_kernel(float *out,
                                                           const float *embd_shard,
                                                           const float *norm_weight_shard,
                                                           const float *global_max_abs,
                                                           const float *global_sumsq,
                                                           uint32_t slots,
                                                           float eps) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    constexpr uint32_t shard_cols = kHidden / kGpus;
    const uint64_t n = (uint64_t)slots * shard_cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / shard_cols);
    const uint32_t local_h = (uint32_t)(i - (uint64_t)slot * shard_cols);
    const float max_abs = global_max_abs[slot];
    const float sum = global_sumsq[slot];
    float norm_scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        norm_scale =
            rsqrtf(sum / (float)kHidden + eps / (max_abs * max_abs)) /
            max_abs;
    }
    const float v = embd_shard[i];
    const float y = isfinite(v) ? v * norm_scale * norm_weight_shard[local_h] : 0.0f;
    out[i] = isfinite(y) ? y : 0.0f;
}

__global__ void output_head_rank_major_to_slot_major_kernel(float *slot_major,
                                                            const float *rank_major,
                                                            uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    constexpr uint32_t shard_cols = kHidden / kGpus;
    const uint64_t n = (uint64_t)slots * (uint64_t)kHidden;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / kHidden);
    const uint32_t h = (uint32_t)(i - (uint64_t)slot * kHidden);
    const uint32_t rank = h / shard_cols;
    const uint32_t local_h = h - rank * shard_cols;
    const uint64_t src =
        ((uint64_t)rank * (uint64_t)slots + (uint64_t)slot) *
            (uint64_t)shard_cols +
        local_h;
    slot_major[i] = rank_major[src];
}

struct OutputHeadRunResult {
    bool pass = true;
    double total_ms = 0.0;
    double gather_ms = 0.0;
    double prep_ms = 0.0;
    double broadcast_ms = 0.0;
    double projection_ms = 0.0;
    double projection_kernel_worst_ms = 0.0;
    double top1_ms = 0.0;
    std::vector<uint32_t> tokens;
    std::vector<float> logits;
    uint64_t checksum = 0;
    int finite_bad = 0;
    int device_sync_count = 0;
    int stream_sync_count = 0;
    int event_sync_count = 0;
};

struct SharedTokenEmbedding {
    bool initialized = false;
    int slots = 0;
    int vocab = 0;
    int rows_per_gpu = 0;
    std::vector<uint16_t> h_w_full;
    uint16_t *d_slot_rows[kGpus] = {};
    uint64_t weight_bytes = 0;
};

int open_shared_token_embedding(const Options &opt,
                                const std::vector<ContractRow> &rows,
                                SharedTokenEmbedding *out) {
    out->slots = opt.slots;
    std::vector<ContractRow> emb_rows;
    int cols = 0;
    int vocab = 0;
    if (!select_bf16_dense_rows(rows, "token_embd.weight", &emb_rows, &cols, &vocab)) {
        std::fprintf(stderr, "shared token embedding failed to select token_embd.weight shards\n");
        return 1;
    }
    if (cols != kHidden || vocab <= 0 || vocab % kGpus != 0) {
        std::fprintf(stderr, "shared token embedding invalid shape cols=%d vocab=%d\n",
                     cols, vocab);
        return 2;
    }
    out->vocab = vocab;
    out->rows_per_gpu = vocab / kGpus;
    const uint64_t shard_elems = (uint64_t)out->rows_per_gpu * (uint64_t)kHidden;
    const uint64_t shard_bytes = shard_elems * sizeof(uint16_t);
    const uint64_t full_elems = shard_elems * kGpus;

    out->h_w_full.assign((size_t)full_elems, 0);
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
        CHECK_CUDA(cudaMalloc(&out->d_slot_rows[(size_t)rank],
                              (size_t)opt.slots * kHidden * sizeof(uint16_t)));
    }

    std::vector<uint16_t> host((size_t)shard_elems);
    for (int shard = 0; shard < kGpus; ++shard) {
        const ContractRow &r = emb_rows[(size_t)shard];
        const int shard_index = r.shard_index >= 0 ? r.shard_index : shard;
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), host.data(),
                          (size_t)shard_bytes) != 0) {
            return 3;
        }
        std::memcpy(out->h_w_full.data() + (uint64_t)shard_index * shard_elems,
                    host.data(), (size_t)shard_bytes);
        out->weight_bytes += shard_bytes;
    }
    out->initialized = true;
    return 0;
}

void close_shared_token_embedding(const Options &opt, SharedTokenEmbedding *out) {
    if (!out || !out->initialized) return;
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
        if (out->d_slot_rows[(size_t)rank]) {
            CHECK_CUDA(cudaFree(out->d_slot_rows[(size_t)rank]));
        }
    }
    *out = SharedTokenEmbedding{};
}

int seed_rank_hc_from_input_tokens(const Options &opt,
                                   SharedTokenEmbedding *embedding,
                                   RankState ranks[kGpus],
                                   const std::vector<uint32_t> &tokens) {
    if (!embedding || !embedding->initialized ||
        (int)tokens.size() < opt.slots ||
        embedding->h_w_full.empty()) {
        return 1;
    }
    const uint64_t shard_elems =
        (uint64_t)opt.slots * 4ull * (uint64_t)(kHidden / kGpus);
    std::vector<uint16_t> slot_rows((size_t)opt.slots * kHidden);
    for (int slot = 0; slot < opt.slots; ++slot) {
        uint32_t token = tokens[(size_t)slot];
        if (token >= (uint32_t)embedding->vocab) token = 0;
        std::memcpy(slot_rows.data() + (size_t)slot * kHidden,
                    embedding->h_w_full.data() + (uint64_t)token * kHidden,
                    (size_t)kHidden * sizeof(uint16_t));
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.d_final_hc_shard || !embedding->d_slot_rows[(size_t)rank]) return 2;
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaMemcpyAsync(embedding->d_slot_rows[(size_t)rank],
                                   slot_rows.data(),
                                   slot_rows.size() * sizeof(uint16_t),
                                   cudaMemcpyHostToDevice, r.stream));
        seed_hc_shard_from_token_embedding_kernel<<<
            (unsigned int)((shard_elems + 255) / 256), 256, 0, r.stream>>>(
            r.d_final_hc_shard,
            embedding->d_slot_rows[(size_t)rank],
            (uint32_t)opt.slots,
            rank);
        CHECK_CUDA(cudaGetLastError());
        r.hc_initialized = true;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
    }
    return 0;
}

int open_shared_output_head(const Options &opt,
                            const std::vector<ContractRow> &rows,
                            SharedOutputHead *out) {
    out->slots = opt.slots;
    std::vector<ContractRow> output_rows;
    int output_cols = 0;
    int vocab = 0;
    if (!select_bf16_dense_rows(rows, "output.weight", &output_rows,
                                &output_cols, &vocab)) {
        std::fprintf(stderr, "shared output-head failed to select output.weight shards\n");
        return 1;
    }
    if (output_cols != kHidden || vocab <= 0 || vocab % kGpus != 0) {
        std::fprintf(stderr, "shared output-head invalid output.weight shape cols=%d vocab=%d\n",
                     output_cols, vocab);
        return 2;
    }
    out->vocab = vocab;
    out->rows_per_gpu = vocab / kGpus;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        out->output_rows[gpu] = output_rows[(size_t)gpu];
    }

    std::vector<float> hc_head_fn;
    std::vector<float> hc_head_base;
    std::vector<float> hc_head_scale;
    std::vector<float> output_norm;
    if (load_control_f32(opt, rows, "hc_head_fn", (size_t)4 * 4 * kHidden,
                         &hc_head_fn) ||
        load_control_f32(opt, rows, "hc_head_base", 4, &hc_head_base) ||
        load_control_f32(opt, rows, "hc_head_scale", 1, &hc_head_scale) ||
        load_control_f32(opt, rows, "output_norm.weight", kHidden, &output_norm)) {
        return 3;
    }

    const uint64_t embd_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const uint64_t hc_shard_cols = 4ull * (uint64_t)(kHidden / kGpus);
    const uint64_t embd_shard_elems =
        (uint64_t)opt.slots * (uint64_t)(kHidden / kGpus);
    const uint64_t logits_elems = (uint64_t)opt.slots * (uint64_t)out->rows_per_gpu;
    const uint64_t output_shard_bytes =
        (uint64_t)out->rows_per_gpu * (uint64_t)kHidden * sizeof(uint16_t);

    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    CHECK_CUDA(cudaStreamCreateWithFlags(&out->stream[0], cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreateWithFlags(&out->prep_ready, cudaEventDisableTiming));

    std::vector<uint16_t> host_w((size_t)out->rows_per_gpu * (size_t)kHidden);
    std::vector<float> head_fn_rank((size_t)hc_shard_cols * 4u);
    std::vector<float> output_norm_rank((size_t)(kHidden / kGpus));
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = out->output_rows[gpu];
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), host_w.data(),
                          (size_t)output_shard_bytes) != 0) {
            return 4;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaMalloc(&out->d_w[gpu], (size_t)output_shard_bytes));
        CHECK_CUDA(cudaMalloc(&out->d_x[gpu], (size_t)embd_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_logits[gpu],
                              (size_t)logits_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_head_fn_rank[gpu],
                              head_fn_rank.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_head_base_rank[gpu],
                              hc_head_base.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_head_scale_rank[gpu],
                              hc_head_scale.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_output_norm_rank[gpu],
                              output_norm_rank.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_reduce_max[gpu],
                              (size_t)opt.slots * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_reduce_sumsq[gpu],
                              (size_t)opt.slots * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_head_pre_rank[gpu],
                              (size_t)opt.slots * 4u * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_head_weights_rank[gpu],
                              (size_t)opt.slots * 4u * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_embd_shard[gpu],
                              (size_t)embd_shard_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_embd_norm_shard[gpu],
                              (size_t)embd_shard_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_embd_rank_major[gpu],
                              (size_t)embd_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_best_token[gpu],
                              (size_t)opt.slots * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&out->d_best_logit[gpu],
                              (size_t)opt.slots * sizeof(float)));
        CHECK_CUDA(cudaEventCreate(&out->projection_start[gpu]));
        CHECK_CUDA(cudaEventCreate(&out->projection_stop[gpu]));
        if (gpu != 0) {
            CHECK_CUDA(cudaStreamCreateWithFlags(&out->stream[gpu],
                                                 cudaStreamNonBlocking));
        }
        CHECK_CUDA(cudaEventCreateWithFlags(&out->broadcast_ready[gpu],
                                            cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&out->top1_done[gpu],
                                            cudaEventDisableTiming));
        CHECK_CUDA(cudaMallocHost(&out->h_best_token[gpu],
                                  (size_t)opt.slots * sizeof(uint32_t)));
        CHECK_CUDA(cudaMallocHost(&out->h_best_logit[gpu],
                                  (size_t)opt.slots * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(out->d_w[gpu], host_w.data(),
                              (size_t)output_shard_bytes, cudaMemcpyHostToDevice));
        for (uint32_t c = 0; c < hc_shard_cols; ++c) {
            const uint64_t global_c = (uint64_t)gpu * hc_shard_cols + c;
            for (uint32_t h = 0; h < 4u; ++h) {
                head_fn_rank[(uint64_t)c * 4ull + h] =
                    hc_head_fn[(uint64_t)h * (4ull * (uint64_t)kHidden) + global_c];
            }
        }
        const uint32_t shard_cols = (uint32_t)(kHidden / kGpus);
        for (uint32_t c = 0; c < shard_cols; ++c) {
            output_norm_rank[c] = output_norm[(uint64_t)gpu * shard_cols + c];
        }
        CHECK_CUDA(cudaMemcpy(out->d_head_fn_rank[gpu], head_fn_rank.data(),
                              head_fn_rank.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_head_base_rank[gpu], hc_head_base.data(),
                              hc_head_base.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_head_scale_rank[gpu], hc_head_scale.data(),
                              hc_head_scale.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_output_norm_rank[gpu], output_norm_rank.data(),
                              output_norm_rank.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        out->output_weight_bytes += output_shard_bytes;
    }
    out->logits_bytes = logits_elems * sizeof(float) * kGpus;
    out->initialized = true;
    return 0;
}

/* MTP (layer 43) output head: shares the main head's LM matmul (output.weight)
 * + working buffers, overriding only the head HC weights (hc_head_* + output_norm)
 * with the MTP's, loaded from the MTP contract (blk.43.*). The MTP predicts in
 * the same vocab, so the LM head is shared. Produces draft logits from the MTP
 * body output via run_shared_output_head_from_rank_hc. No-op unless MTP set. */
int load_mtp_output_head(const Options &opt, const SharedOutputHead *main_head,
                         SharedOutputHead *mtp_head) {
    if (!opt.mtp_contract_path || !opt.mtp_pack_dir) return 0;
    if (!main_head || !main_head->initialized || !mtp_head) return 1;
    *mtp_head = *main_head;  /* share LM head (d_w), working buffers, vocab, streams */
    std::vector<ContractRow> rows;
    LayerStats st;
    if (parse_contract(opt.mtp_contract_path, 43, &rows, &st) != 0 || st.bad_rows != 0) {
        std::fprintf(stderr, "MTP output-head contract parse failed\n");
        return 2;
    }
    Options mopt = opt;
    mopt.pack_dir = opt.mtp_pack_dir;
    std::vector<float> hc_head_fn, hc_head_base, hc_head_scale, output_norm;
    if (load_control_f32(mopt, rows, "blk.43.hc_head_fn", (size_t)4 * 4 * kHidden, &hc_head_fn) ||
        load_control_f32(mopt, rows, "blk.43.hc_head_base", 4, &hc_head_base) ||
        load_control_f32(mopt, rows, "blk.43.hc_head_scale", 1, &hc_head_scale) ||
        load_control_f32(mopt, rows, "blk.43.norm.weight", kHidden, &output_norm)) {
        std::fprintf(stderr, "MTP output-head HC/norm load failed\n");
        return 3;
    }
    const uint64_t hc_shard_cols = 4ull * (uint64_t)(kHidden / kGpus);
    const uint32_t shard_cols = (uint32_t)(kHidden / kGpus);
    std::vector<float> head_fn_rank((size_t)hc_shard_cols * 4u);
    std::vector<float> output_norm_rank((size_t)shard_cols);
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        for (uint32_t c = 0; c < hc_shard_cols; ++c) {
            const uint64_t global_c = (uint64_t)gpu * hc_shard_cols + c;
            for (uint32_t h = 0; h < 4u; ++h)
                head_fn_rank[(uint64_t)c * 4ull + h] =
                    hc_head_fn[(uint64_t)h * (4ull * (uint64_t)kHidden) + global_c];
        }
        for (uint32_t c = 0; c < shard_cols; ++c)
            output_norm_rank[c] = output_norm[(uint64_t)gpu * shard_cols + c];
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        /* override the aliased head-HC pointers with fresh MTP-owned buffers */
        CHECK_CUDA(cudaMalloc(&mtp_head->d_head_fn_rank[gpu], head_fn_rank.size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(mtp_head->d_head_fn_rank[gpu], head_fn_rank.data(),
                              head_fn_rank.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&mtp_head->d_head_base_rank[gpu], hc_head_base.size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(mtp_head->d_head_base_rank[gpu], hc_head_base.data(),
                              hc_head_base.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&mtp_head->d_head_scale_rank[gpu], hc_head_scale.size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(mtp_head->d_head_scale_rank[gpu], hc_head_scale.data(),
                              hc_head_scale.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&mtp_head->d_output_norm_rank[gpu], output_norm_rank.size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(mtp_head->d_output_norm_rank[gpu], output_norm_rank.data(),
                              output_norm_rank.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    std::printf("tp_ep_mtp_output_head_load\tlayer\t43\tvocab\t%d\tPASS\n", mtp_head->vocab);
    std::fflush(stdout);
    return 0;
}

void close_shared_output_head(const Options &opt, SharedOutputHead *out) {
    if (!out || !out->initialized) return;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (out->h_best_logit[gpu]) CHECK_CUDA(cudaFreeHost(out->h_best_logit[gpu]));
        if (out->h_best_token[gpu]) CHECK_CUDA(cudaFreeHost(out->h_best_token[gpu]));
        if (out->top1_done[gpu]) CHECK_CUDA(cudaEventDestroy(out->top1_done[gpu]));
        if (out->broadcast_ready[gpu]) CHECK_CUDA(cudaEventDestroy(out->broadcast_ready[gpu]));
        if (out->projection_stop[gpu]) CHECK_CUDA(cudaEventDestroy(out->projection_stop[gpu]));
        if (out->projection_start[gpu]) CHECK_CUDA(cudaEventDestroy(out->projection_start[gpu]));
        if (out->stream[gpu]) CHECK_CUDA(cudaStreamDestroy(out->stream[gpu]));
        if (out->d_best_logit[gpu]) CHECK_CUDA(cudaFree(out->d_best_logit[gpu]));
        if (out->d_best_token[gpu]) CHECK_CUDA(cudaFree(out->d_best_token[gpu]));
        if (out->d_embd_rank_major[gpu]) CHECK_CUDA(cudaFree(out->d_embd_rank_major[gpu]));
        if (out->d_embd_norm_shard[gpu]) CHECK_CUDA(cudaFree(out->d_embd_norm_shard[gpu]));
        if (out->d_embd_shard[gpu]) CHECK_CUDA(cudaFree(out->d_embd_shard[gpu]));
        if (out->d_head_weights_rank[gpu]) CHECK_CUDA(cudaFree(out->d_head_weights_rank[gpu]));
        if (out->d_head_pre_rank[gpu]) CHECK_CUDA(cudaFree(out->d_head_pre_rank[gpu]));
        if (out->d_reduce_sumsq[gpu]) CHECK_CUDA(cudaFree(out->d_reduce_sumsq[gpu]));
        if (out->d_reduce_max[gpu]) CHECK_CUDA(cudaFree(out->d_reduce_max[gpu]));
        if (out->d_output_norm_rank[gpu]) CHECK_CUDA(cudaFree(out->d_output_norm_rank[gpu]));
        if (out->d_head_scale_rank[gpu]) CHECK_CUDA(cudaFree(out->d_head_scale_rank[gpu]));
        if (out->d_head_base_rank[gpu]) CHECK_CUDA(cudaFree(out->d_head_base_rank[gpu]));
        if (out->d_head_fn_rank[gpu]) CHECK_CUDA(cudaFree(out->d_head_fn_rank[gpu]));
        if (out->d_logits[gpu]) CHECK_CUDA(cudaFree(out->d_logits[gpu]));
        if (out->d_x[gpu]) CHECK_CUDA(cudaFree(out->d_x[gpu]));
        if (out->d_w[gpu]) CHECK_CUDA(cudaFree(out->d_w[gpu]));
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    if (out->prep_ready) CHECK_CUDA(cudaEventDestroy(out->prep_ready));
    *out = SharedOutputHead{};
}

int run_shared_output_head_from_rank_hc(const Options &opt,
                                        SharedOutputHead *head,
                                        RankState ranks[kGpus],
                                        OutputHeadRunResult *result) {
    if (!head || !head->initialized || head->slots != opt.slots) return 1;
    const auto total_start = std::chrono::steady_clock::now();
    const uint64_t embd_elems = (uint64_t)opt.slots * (uint64_t)kHidden;
    const uint64_t embd_shard_elems =
        (uint64_t)opt.slots * (uint64_t)(kHidden / kGpus);

    if (opt.decode_cudagraph_gate) {
        if (opt.decode_cudagraph_output_sync_gate) {
            for (int rank = 0; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(ranks[rank].device));
                CHECK_CUDA(cudaDeviceSynchronize());
                result->device_sync_count++;
            }
        } else {
            const int wait_rc =
                enqueue_control_wait_after_rank_streams(opt, ranks, (cudaStream_t)0);
            if (wait_rc != 0) return wait_rc;
            const int dense_wait_rc =
                enqueue_control_wait_after_dense_streams(opt, ranks, (cudaStream_t)0);
            if (dense_wait_rc != 0) return dense_wait_rc;
        }
    }

    const auto gather_start = std::chrono::steady_clock::now();
    for (int rank = 0; rank < kGpus; ++rank) {
        if (!ranks[rank].d_final_hc_shard) {
            std::fprintf(stderr, "diagnostic output-head missing final HC shard rank=%d\n",
                         rank);
            return 2;
        }
        if (!ranks[rank].compose_nccl_initialized || !ranks[rank].compose_nccl ||
            !head->d_head_fn_rank[rank] || !head->d_head_base_rank[rank] ||
            !head->d_head_scale_rank[rank] || !head->d_output_norm_rank[rank] ||
            !head->d_reduce_max[rank] || !head->d_reduce_sumsq[rank] ||
            !head->d_head_pre_rank[rank] || !head->d_head_weights_rank[rank] ||
            !head->d_embd_shard[rank] || !head->d_embd_norm_shard[rank] ||
            !head->d_embd_rank_major[rank] || !head->d_x[rank]) {
            return 2;
        }
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        const dim3 partial_grid(5u, (unsigned int)opt.slots, 1u);
        output_head_hc_local_max_mix4_partial_kernel<<<
            partial_grid, 256, 0, ranks[rank].stream>>>(
            head->d_reduce_max[rank],
            head->d_head_pre_rank[rank],
            ranks[rank].d_final_hc_shard,
            head->d_head_fn_rank[rank],
            (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclAllReduce(head->d_reduce_max[rank],
                                 head->d_reduce_max[rank],
                                 (size_t)opt.slots, ncclFloat, ncclMax,
                                 r.compose_nccl, r.stream));
        CHECK_NCCL(ncclAllReduce(head->d_head_pre_rank[rank],
                                 head->d_head_pre_rank[rank],
                                 (size_t)opt.slots * 4u, ncclFloat, ncclSum,
                                 r.compose_nccl, r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    const auto gather_stop = std::chrono::steady_clock::now();

    const auto prep_start = std::chrono::steady_clock::now();
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        output_head_hc_local_stable_sumsq_kernel<<<
            (unsigned int)opt.slots, 256, 0, ranks[rank].stream>>>(
            head->d_reduce_sumsq[rank],
            ranks[rank].d_final_hc_shard,
            head->d_reduce_max[rank],
            (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclAllReduce(head->d_reduce_sumsq[rank],
                                 head->d_reduce_sumsq[rank],
                                 (size_t)opt.slots, ncclFloat, ncclSum,
                                 r.compose_nccl, r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        output_head_weights_from_reduced_mix4_kernel<<<
            (unsigned int)(((uint64_t)opt.slots * 4ull + 255) / 256),
            256, 0, ranks[rank].stream>>>(
            head->d_head_weights_rank[rank],
            head->d_reduce_max[rank],
            head->d_reduce_sumsq[rank],
            head->d_head_pre_rank[rank],
            head->d_head_scale_rank[rank],
            head->d_head_base_rank[rank],
            (uint32_t)opt.slots, 1.0e-6f);
        output_head_weighted_sum_shard_kernel<<<
            (unsigned int)((embd_shard_elems + 255) / 256),
            256, 0, ranks[rank].stream>>>(
            head->d_embd_shard[rank],
            ranks[rank].d_final_hc_shard,
            head->d_head_weights_rank[rank],
            (uint32_t)opt.slots);
        output_head_embd_local_max_kernel<<<
            (unsigned int)opt.slots, 256, 0, ranks[rank].stream>>>(
            head->d_reduce_max[rank],
            head->d_embd_shard[rank],
            (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclAllReduce(head->d_reduce_max[rank],
                                 head->d_reduce_max[rank],
                                 (size_t)opt.slots, ncclFloat, ncclMax,
                                 r.compose_nccl, r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        output_head_embd_local_stable_sumsq_kernel<<<
            (unsigned int)opt.slots, 256, 0, ranks[rank].stream>>>(
            head->d_reduce_sumsq[rank],
            head->d_embd_shard[rank],
            head->d_reduce_max[rank],
            (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclAllReduce(head->d_reduce_sumsq[rank],
                                 head->d_reduce_sumsq[rank],
                                 (size_t)opt.slots, ncclFloat, ncclSum,
                                 r.compose_nccl, r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        output_head_apply_output_norm_shard_kernel<<<
            (unsigned int)((embd_shard_elems + 255) / 256),
            256, 0, ranks[rank].stream>>>(
            head->d_embd_norm_shard[rank],
            head->d_embd_shard[rank],
            head->d_output_norm_rank[rank],
            head->d_reduce_max[rank],
            head->d_reduce_sumsq[rank],
            (uint32_t)opt.slots, 1.0e-6f);
        CHECK_CUDA(cudaGetLastError());
    }
    const auto prep_stop = std::chrono::steady_clock::now();

    const auto broadcast_start = std::chrono::steady_clock::now();
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_NCCL(ncclAllGather(head->d_embd_norm_shard[rank],
                                 head->d_embd_rank_major[rank],
                                 (size_t)embd_shard_elems, ncclFloat,
                                 r.compose_nccl, r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        output_head_rank_major_to_slot_major_kernel<<<
            (unsigned int)((embd_elems + 255) / 256),
            256, 0, ranks[rank].stream>>>(
            head->d_x[rank],
            head->d_embd_rank_major[rank],
            (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    const auto broadcast_stop = std::chrono::steady_clock::now();

    const auto projection_start_wall = std::chrono::steady_clock::now();
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        const dim3 grid((unsigned int)head->rows_per_gpu, (unsigned int)opt.slots, 1u);
        CHECK_CUDA(cudaEventRecord(head->projection_start[gpu], ranks[gpu].stream));
        bf16_dense_kernel<<<grid, 256, 0, ranks[gpu].stream>>>(
            head->d_logits[gpu], head->d_w[gpu],
            head->d_x[gpu],
            (uint32_t)head->rows_per_gpu,
            (uint32_t)kHidden,
            (uint32_t)kHidden,
            (uint32_t)opt.slots);
        CHECK_CUDA(cudaEventRecord(head->projection_stop[gpu], ranks[gpu].stream));
        CHECK_CUDA(cudaGetLastError());
    }
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaEventSynchronize(head->projection_stop[gpu]));
        result->event_sync_count++;
        float kernel_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&kernel_ms,
                                        head->projection_start[gpu],
                                        head->projection_stop[gpu]));
        result->projection_kernel_worst_ms =
            std::max(result->projection_kernel_worst_ms, (double)kernel_ms);
    }
    const auto projection_stop_wall = std::chrono::steady_clock::now();

    const auto top1_start = std::chrono::steady_clock::now();
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        const int shard_index = head->output_rows[gpu].shard_index >= 0
            ? head->output_rows[gpu].shard_index
            : gpu;
        shard_top1_kernel<<<(unsigned int)opt.slots, 256, 0, ranks[gpu].stream>>>(
            head->d_best_token[gpu], head->d_best_logit[gpu],
            head->d_logits[gpu], (uint32_t)head->rows_per_gpu,
            (uint32_t)(shard_index * head->rows_per_gpu), (uint32_t)opt.slots);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpyAsync(head->h_best_token[gpu], head->d_best_token[gpu],
                                   (size_t)opt.slots * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, ranks[gpu].stream));
        CHECK_CUDA(cudaMemcpyAsync(head->h_best_logit[gpu], head->d_best_logit[gpu],
                                   (size_t)opt.slots * sizeof(float),
                                   cudaMemcpyDeviceToHost, ranks[gpu].stream));
    }

    result->tokens.assign((size_t)opt.slots, UINT32_MAX);
    result->logits.assign((size_t)opt.slots, -std::numeric_limits<float>::max());
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaStreamSynchronize(ranks[gpu].stream));
        result->stream_sync_count++;
        for (int slot = 0; slot < opt.slots; ++slot) {
            const float logit = head->h_best_logit[gpu][slot];
            if (!std::isfinite(logit)) {
                result->finite_bad++;
                result->pass = false;
                continue;
            }
            if (logit > result->logits[(size_t)slot]) {
                result->logits[(size_t)slot] = logit;
                result->tokens[(size_t)slot] = head->h_best_token[gpu][slot];
            }
        }
    }
    const auto top1_stop = std::chrono::steady_clock::now();
    const auto total_stop = std::chrono::steady_clock::now();

    if (opt.decode_stage_checksum_gate) {
        for (int slot = 0; slot < opt.slots; ++slot) {
            std::printf("tp_ep_decode_top1_logit\tposition\t%llu\tslot\t%d\t"
                        "token\t%u\tlogit\t%.9g\n",
                        (unsigned long long)opt.position, slot,
                        result->tokens[(size_t)slot],
                        (double)result->logits[(size_t)slot]);
        }
    }

    for (int slot = 0; slot < opt.slots; ++slot) {
        if (result->tokens[(size_t)slot] >= (uint32_t)head->vocab ||
            !std::isfinite(result->logits[(size_t)slot])) {
            result->pass = false;
        }
        uint32_t bits = 0;
        std::memcpy(&bits, &result->logits[(size_t)slot], sizeof(bits));
        result->checksum ^= (uint64_t)result->tokens[(size_t)slot] * 1000003ull +
                            (uint64_t)bits + (uint64_t)(slot + 1) * 7907ull;
    }
    if (result->checksum == 0 || result->finite_bad != 0) result->pass = false;

    result->gather_ms =
        std::chrono::duration<double, std::milli>(gather_stop - gather_start).count();
    result->prep_ms =
        std::chrono::duration<double, std::milli>(prep_stop - prep_start).count();
    result->broadcast_ms =
        std::chrono::duration<double, std::milli>(broadcast_stop - broadcast_start).count();
    result->projection_ms =
        std::chrono::duration<double, std::milli>(projection_stop_wall - projection_start_wall).count();
    result->top1_ms =
        std::chrono::duration<double, std::milli>(top1_stop - top1_start).count();
    result->total_ms =
        std::chrono::duration<double, std::milli>(total_stop - total_start).count();
    return result->pass ? 0 : 5;
}

void free_device_dense_outputs(DeviceDenseOutputs &out, const Options &opt) {
    for (int gpu = 0; gpu < (int)out.d_out.size(); ++gpu) {
        if (!out.d_out[(size_t)gpu]) continue;
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        CHECK_CUDA(cudaFree(out.d_out[(size_t)gpu]));
    }
    out = DeviceDenseOutputs{};
}

void free_resident_f8_dense(ResidentF8Dense &op, const Options &opt) {
    for (int gpu = 0; gpu < (int)op.d_w.size(); ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (op.d_w[(size_t)gpu]) CHECK_CUDA(cudaFree(op.d_w[(size_t)gpu]));
        if (op.d_x[(size_t)gpu]) CHECK_CUDA(cudaFree(op.d_x[(size_t)gpu]));
        if (gpu < (int)op.d_w_half.size() && op.d_w_half[(size_t)gpu]) {
            const bool owns = gpu >= (int)op.owns_w_half.size() || op.owns_w_half[(size_t)gpu];
            if (owns) CHECK_CUDA(cudaFree(op.d_w_half[(size_t)gpu]));
        }
        if (gpu < (int)op.d_x_half.size() && op.d_x_half[(size_t)gpu]) {
            CHECK_CUDA(cudaFree(op.d_x_half[(size_t)gpu]));
        }
        if (op.d_out[(size_t)gpu]) CHECK_CUDA(cudaFree(op.d_out[(size_t)gpu]));
        if (gpu < (int)op.cublas.size() && op.cublas[(size_t)gpu]) {
            (void)cublasDestroy(op.cublas[(size_t)gpu]);
        }
    }
    op = ResidentF8Dense{};
}

uint64_t align_up_u64(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

void free_dense_f16_cache(DenseF16Cache &cache, const Options &opt) {
    for (int gpu = 0; gpu < (int)cache.arena.size(); ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (cache.arena[(size_t)gpu]) CHECK_CUDA(cudaFree(cache.arena[(size_t)gpu]));
        if (gpu < (int)cache.temp.size() && cache.temp[(size_t)gpu]) {
            CHECK_CUDA(cudaFree(cache.temp[(size_t)gpu]));
        }
    }
    cache = DenseF16Cache{};
}

const DenseF16CacheEntry *find_dense_f16_cache_entry(const DenseF16Cache &cache,
                                                     const char *tensor,
                                                     int gpu) {
    if (!cache.enabled) return nullptr;
    for (const DenseF16CacheEntry &e : cache.entries) {
        if (e.gpu == gpu && e.tensor_id == tensor) return &e;
    }
    return nullptr;
}

int prepare_dense_f16_cache(const Options &opt,
                            const std::vector<ContractRow> &rows,
                            DenseF16Cache *cache) {
    if (!opt.dense_f16_cache_compose) return 0;
    cache->enabled = true;
    cache->arena.assign((size_t)kGpus, nullptr);
    cache->temp.assign((size_t)kGpus, nullptr);
    uint64_t gpu_offsets[kGpus] = {};
    uint64_t gpu_temp[kGpus] = {};

    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" ||
            (r.source_dtype != "f8_e4m3_b128" && r.source_dtype != "bf16")) {
            continue;
        }
        int cols = 0;
        int total_rows = 0;
        if (!parse_shape2(r.source_shape, &cols, &total_rows)) continue;
        uint64_t rows_per_gpu = 0;
        if (r.source_dtype == "f8_e4m3_b128") {
            if (cols % 128 != 0) continue;
            const uint64_t rb = f8_row_bytes(cols);
            if (rb == 0 || r.bytes_estimate % rb != 0) continue;
            rows_per_gpu = r.bytes_estimate / rb;
        } else {
            const uint64_t rb = (uint64_t)cols * sizeof(uint16_t);
            if (rb == 0 || r.bytes_estimate % rb != 0) continue;
            rows_per_gpu = r.bytes_estimate / rb;
        }
        DenseF16CacheEntry e;
        e.tensor_id = r.tensor_id;
        e.gpu = r.owning_gpu;
        e.cols = cols;
        e.rows_per_gpu = (int)rows_per_gpu;
        e.offset = gpu_offsets[r.owning_gpu];
        e.source_bytes = r.bytes_estimate;
        e.cache_bytes = rows_per_gpu * (uint64_t)cols * sizeof(__half);
        cache->entries.push_back(e);
        cache->rows++;
        cache->source_bytes += e.source_bytes;
        cache->cache_bytes += e.cache_bytes;
        const uint64_t aligned = align_up_u64(e.cache_bytes, 256);
        gpu_offsets[r.owning_gpu] += aligned;
        cache->cache_aligned_bytes += aligned;
        gpu_temp[r.owning_gpu] = std::max(gpu_temp[r.owning_gpu], e.source_bytes);
        cache->max_temp_bytes = std::max(cache->max_temp_bytes, e.source_bytes);
    }

    if (cache->entries.empty()) return 1;
    uint64_t planned_bytes[kGpus] = {};
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        cache->gpu_cache_aligned_bytes[gpu] = gpu_offsets[gpu];
        cache->gpu_temp_bytes[gpu] = gpu_temp[gpu];
        planned_bytes[gpu] = gpu_offsets[gpu] + gpu_temp[gpu];
    }
    if (check_planned_vram_allocation(opt, "dense_f16_cache_prealloc", planned_bytes) != 0) {
        std::fprintf(stderr,
                     "dense_f16_cache_vram_admission_failed min_free_mib=%llu\n",
                     (unsigned long long)opt.vram_min_free_mib);
        return 3;
    }
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (gpu_offsets[gpu]) CHECK_CUDA(cudaMalloc(&cache->arena[(size_t)gpu],
                                                    (size_t)gpu_offsets[gpu]));
        if (gpu_temp[gpu]) CHECK_CUDA(cudaMalloc(&cache->temp[(size_t)gpu],
                                                 (size_t)gpu_temp[gpu]));
    }

    std::vector<uint8_t> host;
    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" ||
            (r.source_dtype != "f8_e4m3_b128" && r.source_dtype != "bf16")) {
            continue;
        }
        const DenseF16CacheEntry *e =
            find_dense_f16_cache_entry(*cache, r.tensor_id.c_str(), r.owning_gpu);
        if (!e || e->source_bytes != r.bytes_estimate) continue;
        host.resize((size_t)r.bytes_estimate);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r), host.data(), host.size()) != 0) {
            free_dense_f16_cache(*cache, opt);
            return 2;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[r.owning_gpu]));
        CHECK_CUDA(cudaMemcpy(cache->temp[(size_t)r.owning_gpu], host.data(), host.size(),
                              cudaMemcpyHostToDevice));
        __half *dst =
            reinterpret_cast<__half *>(cache->arena[(size_t)r.owning_gpu] + e->offset);
        const uint64_t elems = e->cache_bytes / sizeof(__half);
        const unsigned int grid = (unsigned int)((elems + 255) / 256);
        if (r.source_dtype == "f8_e4m3_b128") {
            f8_b128_to_half_kernel<<<grid, 256>>>(
                dst, cache->temp[(size_t)r.owning_gpu], e->rows_per_gpu,
                e->cols, (uint32_t)f8_row_bytes(e->cols));
        } else {
            bf16_to_half_kernel<<<grid, 256>>>(
                dst, reinterpret_cast<const uint16_t *>(cache->temp[(size_t)r.owning_gpu]),
                elems);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (cache->temp[(size_t)gpu]) {
            CHECK_CUDA(cudaFree(cache->temp[(size_t)gpu]));
            cache->temp[(size_t)gpu] = nullptr;
        }
    }
    return 0;
}

int prepare_resident_f8_dense(const Options &opt,
                              const std::vector<ContractRow> &rows,
                              const char *tensor,
                              int seed,
                              const DenseF16Cache *cache,
                              ResidentF8Dense *op,
                              int expected_rows_per_gpu = kHidden / kGpus,
                              bool keep_packed_f8 = false,
                              bool keep_float_input = false) {
    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    bool source_is_f8 = select_dense_rows(rows, tensor, &selected, &cols, &total_rows);
    bool source_is_bf16 = false;
    if (!source_is_f8) {
        source_is_bf16 = select_bf16_dense_rows(rows, tensor, &selected, &cols, &total_rows);
    }
    if (!source_is_f8 && !source_is_bf16) {
        std::fprintf(stderr, "resident dense tensor validation failed for %s\n", tensor);
        return 1;
    }
    if (source_is_bf16 && keep_packed_f8) {
        std::fprintf(stderr, "resident dense tensor %s requested packed f8 retention for bf16 source\n",
                     tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    if (rows_per_gpu != expected_rows_per_gpu) {
        std::fprintf(stderr, "resident dense tensor %s rows_per_gpu=%d expected=%d\n",
                     tensor, rows_per_gpu, expected_rows_per_gpu);
        return 2;
    }
    const uint64_t row_bytes =
        source_is_f8 ? f8_row_bytes(cols) : (uint64_t)cols * sizeof(uint16_t);
    const uint64_t shard_bytes = row_bytes * (uint64_t)rows_per_gpu;
    op->d_w.assign((size_t)kGpus, nullptr);
    op->d_x.assign((size_t)kGpus, nullptr);
    op->d_w_half.assign((size_t)kGpus, nullptr);
    op->owns_w_half.assign((size_t)kGpus, true);
    op->d_x_half.assign((size_t)kGpus, nullptr);
    op->d_out.assign((size_t)kGpus, nullptr);
    op->cublas.assign((size_t)kGpus, nullptr);
    op->rows_per_gpu = rows_per_gpu;
    op->cols = cols;
    op->slots = opt.slots;
    op->row_bytes = row_bytes;

    std::vector<float> h_x((size_t)opt.slots * cols);
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int c = 0; c < cols; ++c) {
            const int m = (slot * (17 + seed) + c * (13 + seed * 3)) % 269;
            h_x[(size_t)slot * cols + c] = ((float)m - 134.0f) * 0.0002f;
        }
    }

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = selected[(size_t)gpu];
        const DenseF16CacheEntry *cache_entry =
            opt.dense_f16_cache_compose && opt.dense_f16_cublas_compose && cache
                ? find_dense_f16_cache_entry(*cache, tensor, gpu)
                : nullptr;
        if (source_is_bf16 && !cache_entry) {
            std::fprintf(stderr,
                         "resident bf16 dense tensor %s requires dense f16 cache on gpu %d\n",
                         tensor, gpu);
            free_resident_f8_dense(*op, opt);
            return 3;
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (!cache_entry || keep_packed_f8) {
            std::vector<uint8_t> h_w((size_t)shard_bytes);
            const std::string path = path_join(opt.pack_dir, r.source_pack_file);
            if (read_exact_at(path, physical_row_offset(r), h_w.data(), h_w.size()) != 0) {
                free_resident_f8_dense(*op, opt);
                return 3;
            }
            CHECK_CUDA(cudaMalloc(&op->d_w[(size_t)gpu], (size_t)shard_bytes));
            CHECK_CUDA(cudaMemcpy(op->d_w[(size_t)gpu], h_w.data(), (size_t)shard_bytes,
                                  cudaMemcpyHostToDevice));
        }
        op->loaded_bytes += shard_bytes;
        CHECK_CUDA(cudaMalloc(&op->d_x[(size_t)gpu], h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&op->d_out[(size_t)gpu],
                              (size_t)opt.slots * rows_per_gpu * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(op->d_x[(size_t)gpu], h_x.data(),
                              h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
        if (opt.dense_f16_cublas_compose) {
            (void)cudaGetLastError();
            if (cache_entry) {
                if (cache_entry->cols != cols || cache_entry->rows_per_gpu != rows_per_gpu) {
                    free_resident_f8_dense(*op, opt);
                    return 4;
                }
                op->d_w_half[(size_t)gpu] =
                    reinterpret_cast<__half *>(cache->arena[(size_t)gpu] + cache_entry->offset);
                op->owns_w_half[(size_t)gpu] = false;
            } else {
                CHECK_CUDA(cudaMalloc(&op->d_w_half[(size_t)gpu],
                                      (size_t)rows_per_gpu * cols * sizeof(__half)));
                op->owns_w_half[(size_t)gpu] = true;
                const uint64_t w_elems = (uint64_t)rows_per_gpu * cols;
                if (!source_is_f8) {
                    free_resident_f8_dense(*op, opt);
                    return 5;
                }
                f8_b128_to_half_kernel<<<(unsigned int)((w_elems + 255) / 256), 256>>>(
                    op->d_w_half[(size_t)gpu], op->d_w[(size_t)gpu],
                    rows_per_gpu, cols, (uint32_t)row_bytes);
                CHECK_CUDA(cudaGetLastError());
            }
            CHECK_CUDA(cudaMalloc(&op->d_x_half[(size_t)gpu],
                                  h_x.size() * sizeof(__half)));
            const uint64_t x_elems = (uint64_t)opt.slots * cols;
            cast_f32_to_half_kernel<<<(unsigned int)((x_elems + 255) / 256), 256>>>(
                op->d_x_half[(size_t)gpu], op->d_x[(size_t)gpu], x_elems);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            if (!keep_float_input) {
                CHECK_CUDA(cudaFree(op->d_x[(size_t)gpu]));
                op->d_x[(size_t)gpu] = nullptr;
            }
            cublasStatus_t st = cublasCreate(&op->cublas[(size_t)gpu]);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "cublasCreate failed on gpu %d: %d\n", gpu, (int)st);
                free_resident_f8_dense(*op, opt);
                return 4;
            }
            (void)cublasSetMathMode(op->cublas[(size_t)gpu], CUBLAS_TENSOR_OP_MATH);
        }
    }
    return 0;
}

void free_shared_dense_ops(SharedDenseOps *ops, const Options &opt) {
    if (!ops) return;
    for (int layer = 0; layer < 43; ++layer) {
        free_resident_f8_dense(ops->layers[layer].attn_q_a, opt);
        free_resident_f8_dense(ops->layers[layer].attn_q_b, opt);
        free_resident_f8_dense(ops->layers[layer].attn_kv_latent, opt);
        free_resident_f8_dense(ops->layers[layer].attn_output_a, opt);
        free_resident_f8_dense(ops->layers[layer].attn_compress_kv, opt);
        free_resident_f8_dense(ops->layers[layer].attn_compress_gate, opt);
        free_resident_f8_dense(ops->layers[layer].indexer_attn_q_b, opt);
        free_resident_f8_dense(ops->layers[layer].indexer_proj, opt);
        free_resident_f8_dense(ops->layers[layer].indexer_compress_kv, opt);
        free_resident_f8_dense(ops->layers[layer].indexer_compress_gate, opt);
        free_resident_f8_dense(ops->layers[layer].attn, opt);
        free_resident_f8_dense(ops->layers[layer].shared, opt);
        free_resident_f8_dense(ops->layers[layer].shared_gate, opt);
        free_resident_f8_dense(ops->layers[layer].shared_up, opt);
        ops->layers[layer] = LayerDenseOps{};
    }
    *ops = SharedDenseOps{};
}

int open_shared_dense_ops(const Options &opt,
                          const DenseF16Cache *cache,
                          SharedDenseOps *ops) {
    if (!opt.dense_f16_cublas_compose || !opt.dense_f16_cache_compose || !cache) {
        return 1;
    }
    for (int layer = 0; layer < 43; ++layer) {
        std::vector<ContractRow> rows;
        LayerStats stats;
        if (parse_contract(opt.contract_path, layer, &rows, &stats) != 0 ||
            stats.bad_rows != 0) {
            free_shared_dense_ops(ops, opt);
            return 2;
        }
        Options layer_opt = opt;
        layer_opt.layer = layer;
        LayerDenseOps &d = ops->layers[layer];
        const std::string attn_q_a_tensor = layer_tensor_name(layer, "attn_q_a.weight");
        const std::string attn_q_b_tensor = layer_tensor_name(layer, "attn_q_b.weight");
        const std::string attn_kv_tensor = layer_tensor_name(layer, "attn_kv_latent.weight");
        const std::string attn_output_a_tensor = layer_tensor_name(layer, "attn_output_a.weight");
        const std::string attn_compress_kv_tensor = layer_tensor_name(layer, "attn_compress_kv.weight");
        const std::string attn_compress_gate_tensor = layer_tensor_name(layer, "attn_compress_gate.weight");
        const std::string indexer_attn_q_b_tensor = layer_tensor_name(layer, "indexer.attn_q_b.weight");
        const std::string indexer_proj_tensor = layer_tensor_name(layer, "indexer.proj.weight");
        const std::string indexer_compress_kv_tensor = layer_tensor_name(layer, "indexer.compress_kv.weight");
        const std::string indexer_compress_gate_tensor = layer_tensor_name(layer, "indexer.compress_gate.weight");
        const std::string attn_tensor = layer_tensor_name(layer, "attn_output_b.weight");
        const std::string shared_tensor = layer_tensor_name(layer, "ffn_down_shexp.weight");
        const std::string shared_gate_tensor = layer_tensor_name(layer, "ffn_gate_shexp.weight");
        const std::string shared_up_tensor = layer_tensor_name(layer, "ffn_up_shexp.weight");
        if (opt.true_ds4_attention_residency_gate) {
            if (prepare_resident_f8_dense(layer_opt, rows, attn_q_a_tensor.c_str(), 11,
                                          cache, &d.attn_q_a, 1024 / kGpus) != 0 ||
                prepare_resident_f8_dense(layer_opt, rows, attn_q_b_tensor.c_str(), 12,
                                          cache, &d.attn_q_b, 32768 / kGpus) != 0 ||
                prepare_resident_f8_dense(layer_opt, rows, attn_kv_tensor.c_str(), 13,
                                          cache, &d.attn_kv_latent, kHeadDim / kGpus) != 0 ||
                prepare_resident_f8_dense(layer_opt, rows, attn_output_a_tensor.c_str(), 14,
                                          cache, &d.attn_output_a, 8192 / kGpus) != 0) {
                free_shared_dense_ops(ops, opt);
                return 5;
            }
            ops->loaded_bytes += d.attn_q_a.loaded_bytes + d.attn_q_b.loaded_bytes +
                                 d.attn_kv_latent.loaded_bytes +
                                 d.attn_output_a.loaded_bytes;
        }
        if (opt.true_ds4_compressed_kv_gate) {
            const int ratio = ds4_layer_ratio(layer);
            if (ratio != 0) {
                const int comp_width = ratio == 4 ? 2 * kHeadDim : kHeadDim;
                if (prepare_resident_f8_dense(layer_opt, rows, attn_compress_kv_tensor.c_str(),
                                              15, cache, &d.attn_compress_kv,
                                              comp_width / kGpus) != 0 ||
                    prepare_resident_f8_dense(layer_opt, rows, attn_compress_gate_tensor.c_str(),
                                              16, cache, &d.attn_compress_gate,
                                              comp_width / kGpus) != 0) {
                    free_shared_dense_ops(ops, opt);
                    return 6;
                }
                ops->loaded_bytes += d.attn_compress_kv.loaded_bytes +
                                     d.attn_compress_gate.loaded_bytes;
            }
            if (opt.true_ds4_indexer_attention_gate && ratio == 4) {
                if (prepare_resident_f8_dense(layer_opt, rows, indexer_attn_q_b_tensor.c_str(),
                                              17, cache, &d.indexer_attn_q_b,
                                              (kIndexerHead * kIndexerHeadDim) / kGpus) != 0 ||
                    prepare_resident_f8_dense(layer_opt, rows, indexer_proj_tensor.c_str(),
                                              18, cache, &d.indexer_proj,
                                              kIndexerHead / kGpus) != 0 ||
                    prepare_resident_f8_dense(layer_opt, rows, indexer_compress_kv_tensor.c_str(),
                                              19, cache, &d.indexer_compress_kv,
                                              (2 * kIndexerHeadDim) / kGpus) != 0 ||
                    prepare_resident_f8_dense(layer_opt, rows, indexer_compress_gate_tensor.c_str(),
                                              20, cache, &d.indexer_compress_gate,
                                              (2 * kIndexerHeadDim) / kGpus) != 0) {
                    free_shared_dense_ops(ops, opt);
                    return 7;
                }
                ops->loaded_bytes += d.indexer_attn_q_b.loaded_bytes +
                                     d.indexer_proj.loaded_bytes +
                                     d.indexer_compress_kv.loaded_bytes +
                                     d.indexer_compress_gate.loaded_bytes;
            }
        }
        if (prepare_resident_f8_dense(layer_opt, rows, attn_tensor.c_str(), 1, cache,
                                      &d.attn) != 0 ||
            prepare_resident_f8_dense(layer_opt, rows, shared_tensor.c_str(), 2, cache,
                                      &d.shared, kHidden / kGpus,
                                      opt.true_shared_ffn_gate,
                                      opt.true_shared_ffn_gate) != 0) {
            free_shared_dense_ops(ops, opt);
            return 3;
        }
        if (opt.true_shared_ffn_gate) {
            if (prepare_resident_f8_dense(layer_opt, rows, shared_gate_tensor.c_str(), 3,
                                          cache, &d.shared_gate, kMid / kGpus) != 0 ||
                prepare_resident_f8_dense(layer_opt, rows, shared_up_tensor.c_str(), 4,
                                          cache, &d.shared_up, kMid / kGpus) != 0) {
                free_shared_dense_ops(ops, opt);
                return 4;
            }
            ops->loaded_bytes += d.shared_gate.loaded_bytes + d.shared_up.loaded_bytes;
        }
        d.initialized = true;
        ops->loaded_bytes += d.attn.loaded_bytes + d.shared.loaded_bytes;
    }
    ops->initialized = true;
    return 0;
}

/* MTP (layer 43) dense F8 projections into SharedDenseOps slot 43, from the MTP
 * contract + pack dir. Isolated from the 0-42 loop. ratio=0 (no compress/
 * indexer). F8 sources load direct on cache-miss, so no DenseF16Cache extension
 * is needed. No-op unless the MTP source is configured. */
int load_mtp_dense_layer43(const Options &opt, const DenseF16Cache *cache,
                           SharedDenseOps *ops) {
    if (!opt.mtp_contract_path || !opt.mtp_pack_dir) return 0;
    if (!ops || !ops->initialized) return 1;
    const int L = 43;
    std::vector<ContractRow> rows;
    LayerStats st;
    if (parse_contract(opt.mtp_contract_path, L, &rows, &st) != 0 || st.bad_rows != 0) {
        std::fprintf(stderr, "MTP dense contract parse failed: %s\n", opt.mtp_contract_path);
        return 1;
    }
    Options mopt = opt;
    mopt.pack_dir = opt.mtp_pack_dir;
    mopt.layer = L;
    LayerDenseOps &d = ops->layers[L];
    const std::string attn_q_a_t = layer_tensor_name(L, "attn_q_a.weight");
    const std::string attn_q_b_t = layer_tensor_name(L, "attn_q_b.weight");
    const std::string attn_kv_t = layer_tensor_name(L, "attn_kv_latent.weight");
    const std::string attn_output_a_t = layer_tensor_name(L, "attn_output_a.weight");
    const std::string attn_output_b_t = layer_tensor_name(L, "attn_output_b.weight");
    const std::string shared_t = layer_tensor_name(L, "ffn_down_shexp.weight");
    const std::string shared_gate_t = layer_tensor_name(L, "ffn_gate_shexp.weight");
    const std::string shared_up_t = layer_tensor_name(L, "ffn_up_shexp.weight");
    if (opt.true_ds4_attention_residency_gate) {
        if (prepare_resident_f8_dense(mopt, rows, attn_q_a_t.c_str(), 11, cache,
                                      &d.attn_q_a, 1024 / kGpus) != 0 ||
            prepare_resident_f8_dense(mopt, rows, attn_q_b_t.c_str(), 12, cache,
                                      &d.attn_q_b, 32768 / kGpus) != 0 ||
            prepare_resident_f8_dense(mopt, rows, attn_kv_t.c_str(), 13, cache,
                                      &d.attn_kv_latent, kHeadDim / kGpus) != 0 ||
            prepare_resident_f8_dense(mopt, rows, attn_output_a_t.c_str(), 14, cache,
                                      &d.attn_output_a, 8192 / kGpus) != 0) {
            std::fprintf(stderr, "MTP dense attn residency load failed\n");
            return 5;
        }
    }
    if (prepare_resident_f8_dense(mopt, rows, attn_output_b_t.c_str(), 1, cache, &d.attn) != 0 ||
        prepare_resident_f8_dense(mopt, rows, shared_t.c_str(), 2, cache, &d.shared,
                                  kHidden / kGpus, opt.true_shared_ffn_gate,
                                  opt.true_shared_ffn_gate) != 0) {
        std::fprintf(stderr, "MTP dense attn_output_b/shared load failed\n");
        return 3;
    }
    if (opt.true_shared_ffn_gate) {
        if (prepare_resident_f8_dense(mopt, rows, shared_gate_t.c_str(), 3, cache,
                                      &d.shared_gate, kMid / kGpus) != 0 ||
            prepare_resident_f8_dense(mopt, rows, shared_up_t.c_str(), 4, cache,
                                      &d.shared_up, kMid / kGpus) != 0) {
            std::fprintf(stderr, "MTP dense shared gate/up load failed\n");
            return 4;
        }
    }
    d.initialized = true;
    std::printf("tp_ep_mtp_dense_layer43_load\tlayer\t43\tPASS\n");
    std::fflush(stdout);
    return 0;
}

int launch_resident_f8_dense(const Options &opt,
                             const ResidentF8Dense &op,
                             RankState ranks[kGpus]) {
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        (void)cudaGetLastError();
        cudaStream_t stream = ranks[gpu].dense_stream ? ranks[gpu].dense_stream
                                                       : ranks[gpu].stream;
        if (opt.dense_f16_cublas_compose) {
            if (!op.cublas[(size_t)gpu] ||
                !op.d_w_half[(size_t)gpu] ||
                !op.d_x_half[(size_t)gpu]) {
                return 1;
            }
            cublasStatus_t st = cublasSetStream(op.cublas[(size_t)gpu], stream);
            if (st != CUBLAS_STATUS_SUCCESS) return 2;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            st = cublasGemmEx(op.cublas[(size_t)gpu],
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              op.rows_per_gpu,
                              op.slots,
                              op.cols,
                              &alpha,
                              op.d_w_half[(size_t)gpu],
                              CUDA_R_16F,
                              op.cols,
                              op.d_x_half[(size_t)gpu],
                              CUDA_R_16F,
                              op.cols,
                              &beta,
                              op.d_out[(size_t)gpu],
                              CUDA_R_32F,
                              op.rows_per_gpu,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "cublasGemmEx failed gpu=%d status=%d\n", gpu, (int)st);
                return 3;
            }
        } else if (opt.dense_hmma_compose) {
            const dim3 grid((unsigned int)((op.rows_per_gpu + 63) / 64),
                            (unsigned int)((op.slots + 15) / 16),
                            1);
            f8_b128_dense_hmma_m16_kernel<<<grid, 128, 0, stream>>>(
                op.d_out[(size_t)gpu], op.d_w[(size_t)gpu], op.d_x[(size_t)gpu],
                op.rows_per_gpu, op.cols, (uint32_t)op.row_bytes, op.slots);
        } else {
            const dim3 grid((unsigned int)op.rows_per_gpu, (unsigned int)op.slots, 1);
            f8_b128_dense_kernel<<<grid, 256, 0, stream>>>(
                op.d_out[(size_t)gpu], op.d_w[(size_t)gpu], op.d_x[(size_t)gpu],
                op.rows_per_gpu, op.cols, (uint32_t)op.row_bytes, op.slots);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    return 0;
}

int launch_resident_f8_dense_f32_input(const Options &opt,
                                       const ResidentF8Dense &op,
                                       RankState ranks[kGpus]) {
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        if (!op.d_w[(size_t)gpu] || !op.d_x[(size_t)gpu]) return 1;
        cudaStream_t stream = ranks[gpu].dense_stream ? ranks[gpu].dense_stream
                                                       : ranks[gpu].stream;
        const dim3 grid((unsigned int)op.rows_per_gpu, (unsigned int)op.slots, 1);
        f8_b128_dense_kernel<<<grid, 256, 0, stream>>>(
            op.d_out[(size_t)gpu], op.d_w[(size_t)gpu], op.d_x[(size_t)gpu],
            op.rows_per_gpu, op.cols, (uint32_t)op.row_bytes, op.slots);
        CHECK_CUDA(cudaGetLastError());
    }
    return 0;
}

int enqueue_dense_wait_after_rank_stream(RankState ranks[kGpus]) {
    const int slot = next_graph_order_event_slot(ranks);
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        RankState &r = ranks[gpu];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.dense_stream || r.dense_stream == r.stream) continue;
        cudaEvent_t ev = graph_stream_done_event(r, slot);
        if (!ev) return 1;
        CHECK_CUDA(cudaEventRecord(ev, r.stream));
        CHECK_CUDA(cudaStreamWaitEvent(r.dense_stream, ev, 0));
    }
    return 0;
}

int enqueue_rank_streams_wait_after_dense_streams(RankState ranks[kGpus]) {
    const int slot = next_graph_order_event_slot(ranks);
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        RankState &r = ranks[gpu];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (!r.dense_stream || r.dense_stream == r.stream) continue;
        cudaEvent_t ev = graph_dense_done_event(r, slot);
        if (!ev) return 1;
        CHECK_CUDA(cudaEventRecord(ev, r.dense_stream));
        CHECK_CUDA(cudaStreamWaitEvent(r.stream, ev, 0));
        // Bidirectional barrier: also make the dense stream wait for the rank
        // stream. The eager path this substitutes for fully drains BOTH streams
        // (cudaStreamSynchronize on dense_stream and stream); a dense->rank-only
        // edge lets dense_stream lap rank_stream across graph replays, racing the
        // in-place writes to the attention dense output (attn_q_b.d_out) against
        // the prior step's readers. Recording the rank-stream event and waiting on
        // it from dense_stream restores the two-way ordering the eager sync implied.
        cudaEvent_t sev = graph_stream_done_event(r, slot);
        if (!sev) return 1;
        CHECK_CUDA(cudaEventRecord(sev, r.stream));
        CHECK_CUDA(cudaStreamWaitEvent(r.dense_stream, sev, 0));
    }
    return 0;
}

int enqueue_cross_gpu_stream_barrier(RankState ranks[kGpus],
                                     bool include_copy_streams) {
    const int slot = next_graph_order_event_slot(ranks);
    for (int src = 0; src < kGpus; ++src) {
        RankState &r = ranks[src];
        CHECK_CUDA(cudaSetDevice(r.device));
        cudaEvent_t stream_ev = graph_stream_done_event(r, slot);
        cudaEvent_t dense_ev = graph_dense_done_event(r, slot);
        if (!stream_ev || !dense_ev) return 1;
        CHECK_CUDA(cudaEventRecord(stream_ev, r.stream));
        CHECK_CUDA(cudaEventRecord(dense_ev,
                                   r.dense_stream ? r.dense_stream : r.stream));
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        RankState &r = ranks[dst];
        CHECK_CUDA(cudaSetDevice(r.device));
        for (int src = 0; src < kGpus; ++src) {
            CHECK_CUDA(cudaStreamWaitEvent(r.stream,
                                           graph_stream_done_event(ranks[src],
                                                                   slot),
                                           0));
            CHECK_CUDA(cudaStreamWaitEvent(r.stream,
                                           graph_dense_done_event(ranks[src],
                                                                  slot),
                                           0));
            if (r.dense_stream) {
                CHECK_CUDA(cudaStreamWaitEvent(r.dense_stream,
                                               graph_stream_done_event(ranks[src],
                                                                       slot),
                                               0));
                CHECK_CUDA(cudaStreamWaitEvent(r.dense_stream,
                                               graph_dense_done_event(ranks[src],
                                                                      slot),
                                               0));
            }
            if (include_copy_streams) {
                for (int q = 0; q < kGpus; ++q) {
                    cudaStream_t copy_stream = r.copy_streams[q]
                        ? r.copy_streams[q]
                        : r.copy_stream ? r.copy_stream : r.stream;
                    CHECK_CUDA(cudaStreamWaitEvent(copy_stream,
                                                   graph_stream_done_event(
                                                       ranks[src], slot),
                                                   0));
                    CHECK_CUDA(cudaStreamWaitEvent(copy_stream,
                                                   graph_dense_done_event(
                                                       ranks[src], slot),
                                                   0));
                }
            }
        }
    }
    return 0;
}

int enqueue_control_wait_after_rank_streams(const Options &opt,
                                            RankState ranks[kGpus],
                                            cudaStream_t control_stream) {
    const int slot = next_graph_order_event_slot(ranks);
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
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
}

int enqueue_control_wait_after_dense_streams(const Options &opt,
                                             RankState ranks[kGpus],
                                             cudaStream_t control_stream) {
    const int slot = next_graph_order_event_slot(ranks);
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        cudaEvent_t ev = graph_dense_done_event(r, slot);
        if (!ev) return 1;
        CHECK_CUDA(cudaEventRecord(ev,
                                   r.dense_stream ? r.dense_stream : r.stream));
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaStreamWaitEvent(control_stream,
                                       graph_dense_done_event(ranks[rank],
                                                              slot),
                                       0));
    }
    return 0;
}

int enqueue_rank_streams_wait_after_control(const Options &opt,
                                            RankState ranks[kGpus],
                                            cudaStream_t control_stream) {
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
}

int nccl_broadcast_f32_from_device0_to_current_full(
    const Options &opt,
    RankState ranks[kGpus],
    const float *src_device0,
    uint64_t elems,
    const char *label);
