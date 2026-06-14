__global__ void gather_hc_shard_to_full_kernel(float *full_hc,
                                               const float *shard_hc,
                                               int rank,
                                               uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t shard_cols = kHidden / kGpus;
    const uint64_t elems = (uint64_t)slots * 4ull * shard_cols;
    if (i >= elems) return;
    const uint32_t local_h = (uint32_t)(i % shard_cols);
    const uint32_t row = (uint32_t)((i / shard_cols) & 3ull);
    const uint32_t slot = (uint32_t)(i / (4ull * shard_cols));
    const uint64_t dst =
        ((uint64_t)slot * 4ull + (uint64_t)row) * (uint64_t)kHidden +
        (uint64_t)rank * shard_cols + local_h;
    full_hc[dst] = shard_hc[i];
}

__global__ void seed_hc_shard_from_token_embedding_kernel(float *shard_hc,
                                                          const uint16_t *slot_rows,
                                                          uint32_t slots,
                                                          int rank) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t shard_cols = kHidden / kGpus;
    const uint64_t elems = (uint64_t)slots * 4ull * shard_cols;
    if (i >= elems) return;
    const uint32_t local_h = (uint32_t)(i % shard_cols);
    const uint32_t slot = (uint32_t)(i / (4ull * shard_cols));
    const uint32_t hidden_col = (uint32_t)rank * shard_cols + local_h;
    shard_hc[i] = bf16_to_f32_dev(slot_rows[(uint64_t)slot * kHidden + hidden_col]);
}

__global__ void synthetic_hc_kernel(float *hc, uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * 4ull * (uint64_t)kHidden;
    if (i >= n) return;
    const uint32_t col = (uint32_t)(i % kHidden);
    const uint32_t row = (uint32_t)((i / kHidden) & 3ull);
    const uint32_t slot = (uint32_t)(i / (4ull * (uint64_t)kHidden));
    const int m = (int)((slot * 97u + row * 31u + col * 17u) % 257u);
    hc[i] = ((float)m - 128.0f) * 0.0025f;
}

__global__ void gather_current_shard_to_full_kernel(float *full,
                                                    const float *shard,
                                                    int rank,
                                                    uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t shard_cols = kHidden / kGpus;
    const uint64_t n = (uint64_t)slots * (uint64_t)shard_cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / shard_cols);
    const uint32_t local_h = (uint32_t)(i % shard_cols);
    full[(uint64_t)slot * kHidden + (uint64_t)rank * shard_cols + local_h] = shard[i];
}

__global__ void gather_current_shards_to_full8_kernel(float *full,
                                                      const float *shard0,
                                                      const float *shard1,
                                                      const float *shard2,
                                                      const float *shard3,
                                                      const float *shard4,
                                                      const float *shard5,
                                                      const float *shard6,
                                                      const float *shard7,
                                                      uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)kHidden;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / kHidden);
    const uint32_t col = (uint32_t)(i % kHidden);
    const uint32_t shard_cols = kHidden / kGpus;
    const uint32_t rank = col / shard_cols;
    const uint32_t local_h = col - rank * shard_cols;
    const uint64_t src_i = (uint64_t)slot * shard_cols + local_h;
    const float *src = shard0;
    if (rank == 1u) src = shard1;
    else if (rank == 2u) src = shard2;
    else if (rank == 3u) src = shard3;
    else if (rank == 4u) src = shard4;
    else if (rank == 5u) src = shard5;
    else if (rank == 6u) src = shard6;
    else if (rank == 7u) src = shard7;
    full[i] = src[src_i];
}

/* Sprint 605: single-launch gather of all 8 attention-output-A src shards into
 * one dst's full buffer (replaces the 8 per-src cudaMemcpy2DAsync). full has
 * `full_cols` columns laid out as [src * shard_cols + h]; each src shard is
 * [slots][shard_cols]. Byte-identical to the memcpy2D gather it replaces. */
__global__ void gather_attn_output_a_shards_to_full8_kernel(
    float *full,
    const float *shard0, const float *shard1, const float *shard2,
    const float *shard3, const float *shard4, const float *shard5,
    const float *shard6, const float *shard7,
    uint32_t full_cols, uint32_t shard_cols, uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)full_cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / full_cols);
    const uint32_t col = (uint32_t)(i % full_cols);
    const uint32_t src = col / shard_cols;
    const uint32_t local_h = col - src * shard_cols;
    const uint64_t src_i = (uint64_t)slot * shard_cols + local_h;
    const float *s = shard0;
    if (src == 1u) s = shard1;
    else if (src == 2u) s = shard2;
    else if (src == 3u) s = shard3;
    else if (src == 4u) s = shard4;
    else if (src == 5u) s = shard5;
    else if (src == 6u) s = shard6;
    else if (src == 7u) s = shard7;
    full[i] = s[src_i];
}

__global__ void rank_major_current_shards_to_slot_major_kernel(
    float *full,
    const float *rank_major,
    uint32_t shard_cols,
    uint32_t ranks,
    uint32_t slots) {
    const uint64_t cols = (uint64_t)shard_cols * (uint64_t)ranks;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / cols);
    const uint32_t col = (uint32_t)(i % cols);
    const uint32_t rank = col / shard_cols;
    const uint32_t local_col = col - rank * shard_cols;
    const uint64_t src_i =
        ((uint64_t)rank * (uint64_t)slots + (uint64_t)slot) *
            (uint64_t)shard_cols +
        (uint64_t)local_col;
    full[i] = rank_major[src_i];
}

__global__ void slot_major_current_to_rank_major_kernel(
    float *rank_major,
    const float *full,
    uint32_t shard_cols,
    uint32_t ranks,
    uint32_t slots) {
    const uint64_t cols = (uint64_t)shard_cols * (uint64_t)ranks;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / cols);
    const uint32_t col = (uint32_t)(i % cols);
    const uint32_t rank = col / shard_cols;
    const uint32_t local_col = col - rank * shard_cols;
    const uint64_t dst_i =
        ((uint64_t)rank * (uint64_t)slots + (uint64_t)slot) *
            (uint64_t)shard_cols +
        (uint64_t)local_col;
    rank_major[dst_i] = full[i];
}

__global__ void gather_dense_shard_to_full_kernel(float *full,
                                                  const float *shard,
                                                  int rank,
                                                  uint32_t shard_cols,
                                                  uint32_t total_cols,
                                                  uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)shard_cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / shard_cols);
    const uint32_t local_col = (uint32_t)(i % shard_cols);
    full[(uint64_t)slot * total_cols + (uint64_t)rank * shard_cols + local_col] =
        shard[i];
}

__global__ void fill_dense_input_half_from_tensor_kernel(__half *dst,
                                                         const float *src,
                                                         uint32_t cols,
                                                         uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)cols;
    if (i >= n) return;
    dst[i] = f32_to_half_saturate(src[i]);
}

__global__ void fill_dense_input_half_from_rank_major_shards_kernel(
    __half *dst,
    const float *rank_major,
    uint32_t shard_cols,
    uint32_t ranks,
    uint32_t slots) {
    const uint64_t cols = (uint64_t)shard_cols * (uint64_t)ranks;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / cols);
    const uint32_t col = (uint32_t)(i % cols);
    const uint32_t rank = col / shard_cols;
    const uint32_t local_col = col - rank * shard_cols;
    const uint64_t src_i =
        ((uint64_t)rank * (uint64_t)slots + (uint64_t)slot) *
            (uint64_t)shard_cols +
        (uint64_t)local_col;
    dst[i] = f32_to_half_saturate(rank_major[src_i]);
}

__global__ void expand_hidden_to_proxy_hc_shard_kernel(float *hc,
                                                       const float *hidden,
                                                       int rank,
                                                       int slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t shard_cols = kHidden / kGpus;
    const uint64_t elems = (uint64_t)slots * 4ull * shard_cols;
    if (i >= elems) return;
    const uint32_t local_h = (uint32_t)(i % shard_cols);
    const uint32_t row = (uint32_t)((i / shard_cols) & 3ull);
    const uint32_t slot = (uint32_t)(i / (4ull * shard_cols));
    const float v = hidden[(uint64_t)slot * shard_cols + local_h];
    const float row_scale = row == 0u ? 1.0f : (0.25f * (float)(row + 1u));
    const float row_bias =
        ((float)(rank + 1) * 0.0001f) + ((float)row * 0.00001f);
    hc[i] = v * row_scale + row_bias;
}

__global__ void seed_initial_hc_shard_kernel(float *hc, int rank, int slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t shard_cols = kHidden / kGpus;
    const uint64_t elems = (uint64_t)slots * 4ull * shard_cols;
    if (i >= elems) return;
    const uint32_t local_h = (uint32_t)(i % shard_cols);
    const uint32_t row = (uint32_t)((i / shard_cols) & 3ull);
    const uint32_t slot = (uint32_t)(i / (4ull * shard_cols));
    const uint32_t global_h = (uint32_t)rank * shard_cols + local_h;
    const int m = (int)((slot * 97u + row * 31u + global_h * 17u) % 257u);
    hc[i] = ((float)m - 128.0f) * 0.0025f;
}
