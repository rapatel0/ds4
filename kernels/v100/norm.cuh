__global__ void rms_norm_plain_rows_kernel(float *out,
                                           const float *in,
                                           uint32_t cols,
                                           uint32_t rows,
                                           float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *src = in + (uint64_t)row * cols;
    float *dst = out + (uint64_t)row * cols;
    float sum = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = src[c];
        sum += v * v;
    }
    sum = block_sum_256_f32(sum);
    const float scale = rsqrtf(sum / (float)cols + eps);
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        dst[c] = src[c] * scale;
    }
}

__global__ void rms_norm_plain_rows_stable_kernel(float *out,
                                                  const float *in,
                                                  uint32_t cols,
                                                  uint32_t rows,
                                                  float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *src = in + (uint64_t)row * cols;
    float *dst = out + (uint64_t)row * cols;
    float local_max = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = src[c];
        if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
    }
    const float max_abs = block_max_256_f32(local_max);
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
            const float v = src[c];
            if (isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    sum = block_sum_256_f32(sum);
    float scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        scale = rsqrtf(sum / (float)cols + eps / (max_abs * max_abs)) / max_abs;
    }
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = src[c];
        dst[c] = isfinite(v) ? v * scale : 0.0f;
    }
}

__global__ void head_rms_norm_local_heads_kernel(float *x,
                                                 uint32_t slots,
                                                 uint32_t local_heads,
                                                 uint32_t head_dim,
                                                 float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= slots * local_heads) return;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        xr[i] *= scale;
    }
}

__global__ void rms_norm_weight_rows_kernel(float *out,
                                            const float *in,
                                            const float *weight,
                                            uint32_t cols,
                                            uint32_t rows,
                                            float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *src = in + (uint64_t)row * cols;
    float *dst = out + (uint64_t)row * cols;
    float sum = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = src[c];
        sum += v * v;
    }
    sum = block_sum_256_f32(sum);
    const float scale = rsqrtf(sum / (float)cols + eps);
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        dst[c] = src[c] * scale * weight[c];
    }
}

__global__ void rms_norm_weight_rows_stable_kernel(float *out,
                                                   const float *in,
                                                   const float *weight,
                                                   uint32_t cols,
                                                   uint32_t rows,
                                                   float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *src = in + (uint64_t)row * cols;
    float *dst = out + (uint64_t)row * cols;
    float local_max = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = src[c];
        if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
    }
    const float max_abs = block_max_256_f32(local_max);
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
            const float v = src[c];
            if (isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    sum = block_sum_256_f32(sum);
    float scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        scale = rsqrtf(sum / (float)cols + eps / (max_abs * max_abs)) / max_abs;
    }
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = src[c];
        const float y = isfinite(v) ? v * scale * weight[c] : 0.0f;
        dst[c] = isfinite(y) ? y : 0.0f;
    }
}

__global__ void rank_major_norm_scale_kernel(float *scale,
                                             const float *rank_major,
                                             uint32_t shard_cols,
                                             uint32_t rank_count,
                                             uint32_t slots,
                                             float eps) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots) return;
    float local_max = 0.0f;
    for (uint32_t h = threadIdx.x; h < kHidden; h += blockDim.x) {
        const uint32_t src_rank = h / shard_cols;
        const uint32_t local_h = h - src_rank * shard_cols;
        float v = 0.0f;
        if (src_rank < rank_count) {
            const uint64_t src_i =
                ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot) *
                    (uint64_t)shard_cols +
                (uint64_t)local_h;
            v = rank_major[src_i];
        }
        if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
    }
    const float max_abs = block_max_256_f32(local_max);
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t h = threadIdx.x; h < kHidden; h += blockDim.x) {
            const uint32_t src_rank = h / shard_cols;
            const uint32_t local_h = h - src_rank * shard_cols;
            float v = 0.0f;
            if (src_rank < rank_count) {
                const uint64_t src_i =
                    ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot) *
                        (uint64_t)shard_cols +
                    (uint64_t)local_h;
                v = rank_major[src_i];
            }
            if (!isfinite(v)) v = 0.0f;
            const float scaled = v / max_abs;
            sum += scaled * scaled;
        }
    }
    sum = block_sum_256_f32(sum);
    float s = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        s = rsqrtf(sum / (float)kHidden + eps / (max_abs * max_abs)) / max_abs;
    }
    if (threadIdx.x == 0u) scale[slot] = isfinite(s) ? s : 0.0f;
}

__global__ void current_shard_max_kernel(float *max_out,
                                         const float *current_shard,
                                         uint32_t shard_cols,
                                         uint32_t slots) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots) return;
    const float *row = current_shard + (uint64_t)slot * shard_cols;
    float local_max = 0.0f;
    for (uint32_t h = threadIdx.x; h < shard_cols; h += blockDim.x) {
        float v = row[h];
        if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
    }
    const float max_abs = block_max_256_f32(local_max);
    if (threadIdx.x == 0u) max_out[slot] = max_abs;
}

__global__ void current_shard_stable_sumsq_kernel(float *sumsq_out,
                                                  const float *current_shard,
                                                  const float *global_max,
                                                  uint32_t shard_cols,
                                                  uint32_t slots) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots) return;
    const float max_abs = global_max[slot];
    const float *row = current_shard + (uint64_t)slot * shard_cols;
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t h = threadIdx.x; h < shard_cols; h += blockDim.x) {
            float v = row[h];
            if (!isfinite(v)) v = 0.0f;
            const float scaled = v / max_abs;
            sum += scaled * scaled;
        }
    }
    sum = block_sum_256_f32(sum);
    if (threadIdx.x == 0u) sumsq_out[slot] = sum;
}
