__device__ void hc4_split_one_dev(float *out,
                                  const float *mix,
                                  const float *scale,
                                  const float *base,
                                  uint32_t sinkhorn_iters,
                                  float epsv) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
    for (int i = 0; i < 4; ++i) {
        const float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }

    float c[16];
    for (int r = 0; r < 4; ++r) {
        float m = -INFINITY;
        for (int col = 0; col < 4; ++col) {
            const float v = mix[8 + r * 4 + col] * comb_scale +
                            base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; ++col) {
            const float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; ++col) c[r * 4 + col] = c[r * 4 + col] / s + epsv;
    }
    for (int col = 0; col < 4; ++col) {
        float s = epsv;
        for (int r = 0; r < 4; ++r) s += c[r * 4 + col];
        for (int r = 0; r < 4; ++r) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; ++iter) {
        for (int r = 0; r < 4; ++r) {
            float s = epsv;
            for (int col = 0; col < 4; ++col) s += c[r * 4 + col];
            for (int col = 0; col < 4; ++col) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; ++col) {
            float s = epsv;
            for (int r = 0; r < 4; ++r) s += c[r * 4 + col];
            for (int r = 0; r < 4; ++r) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; ++i) out[8 + i] = c[i];
}

__global__ void output_hc_weights_rows_kernel(float *out,
                                              const float *pre,
                                              const float *scale,
                                              const float *base,
                                              uint32_t rows) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = rows * 4u;
    if (i >= n) return;
    const uint32_t h = i & 3u;
    const float z = pre[i] * scale[0] + base[h];
    out[i] = 1.0f / (1.0f + expf(-z)) + 1.0e-6f;
}

__global__ void hc_weighted_sum_rows_kernel(float *out,
                                            const float *hc,
                                            const float *weights,
                                            uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)kHidden;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / kHidden);
    const uint32_t col = (uint32_t)(i % kHidden);
    const uint64_t base = (uint64_t)slot * 4ull * (uint64_t)kHidden;
    const float w0 = weights[(uint64_t)slot * 4ull + 0ull];
    const float w1 = weights[(uint64_t)slot * 4ull + 1ull];
    const float w2 = weights[(uint64_t)slot * 4ull + 2ull];
    const float w3 = weights[(uint64_t)slot * 4ull + 3ull];
    out[i] = hc[base + col] * w0 +
             hc[base + (uint64_t)kHidden + col] * w1 +
             hc[base + 2ull * (uint64_t)kHidden + col] * w2 +
             hc[base + 3ull * (uint64_t)kHidden + col] * w3;
}

__global__ void hc_weighted_sum_shard_kernel(float *out,
                                             const float *hc,
                                             const float *weights,
                                             uint32_t slots,
                                             int reference_reduce) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t shard_cols = kHidden / kGpus;
    const uint64_t n = (uint64_t)slots * (uint64_t)shard_cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / shard_cols);
    const uint32_t local_h = (uint32_t)(i % shard_cols);
    const uint64_t base = (uint64_t)slot * 4ull * (uint64_t)shard_cols;
    const float w0 = weights[(uint64_t)slot * kHcMix + 0ull];
    const float w1 = weights[(uint64_t)slot * kHcMix + 1ull];
    const float w2 = weights[(uint64_t)slot * kHcMix + 2ull];
    const float w3 = weights[(uint64_t)slot * kHcMix + 3ull];
    float v = hc[base + local_h] * w0 +
              hc[base + (uint64_t)shard_cols + local_h] * w1 +
              hc[base + 2ull * (uint64_t)shard_cols + local_h] * w2 +
              hc[base + 3ull * (uint64_t)shard_cols + local_h] * w3;
    if (!isfinite(v)) v = 0.0f;
    if (!reference_reduce) {
        v = fminf(1.0f, fmaxf(-1.0f, v * 0.125f));
    }
    out[i] = v;
}

__global__ void hc_local_max_mix_partial_kernel(float *max_abs_out,
                                                float *mix_out,
                                                const float *hc_shard,
                                                const float *fn_shard,
                                                uint32_t slots) {
    const uint32_t op = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    constexpr uint32_t shard_cols = kHidden / kGpus;
    constexpr uint32_t local_cols = kHcRows * shard_cols;
    if (slot >= slots || op > kHcMix) return;
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
        if (isfinite(v)) {
            acc += fn_shard[(uint64_t)c * kHcMix + mix] * v;
        }
    }
    acc = block_sum_256_f32(acc);
    if (threadIdx.x == 0u) {
        mix_out[(uint64_t)slot * kHcMix + mix] = isfinite(acc) ? acc : 0.0f;
    }
}

__global__ void hc_local_stable_sumsq_kernel(float *sumsq_out,
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

__global__ void hc_apply_reduced_mix_split_kernel(float *split,
                                                  const float *global_max_abs,
                                                  const float *global_sumsq,
                                                  const float *global_mix,
                                                  const float *scale_param,
                                                  const float *base,
                                                  uint32_t slots,
                                                  uint32_t sinkhorn_iters,
                                                  float eps) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots) return;
    const float max_abs = global_max_abs[slot];
    const float sum = global_sumsq[slot];
    float norm_scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        norm_scale =
            rsqrtf(sum / (float)(kHcRows * kHidden) +
                   eps / (max_abs * max_abs)) /
            max_abs;
    }
    float mix[kHcMix];
    for (int i = 0; i < kHcMix; ++i) {
        const float v = global_mix[(uint64_t)slot * kHcMix + (uint64_t)i];
        mix[i] = isfinite(v) ? v * norm_scale : 0.0f;
    }
    hc4_split_one_dev(split + (uint64_t)slot * kHcMix,
                      mix, scale_param, base, sinkhorn_iters, 1.0e-6f);
}

__global__ void hc_scale_reduced_mix_kernel(float *scaled_mix,
                                            const float *global_max_abs,
                                            const float *global_sumsq,
                                            const float *global_mix,
                                            uint32_t slots,
                                            float eps) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = slots * (uint32_t)kHcMix;
    if (i >= n) return;
    const uint32_t slot = i / (uint32_t)kHcMix;
    const float max_abs = global_max_abs[slot];
    const float sum = global_sumsq[slot];
    float norm_scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        norm_scale =
            rsqrtf(sum / (float)(kHcRows * kHidden) +
                   eps / (max_abs * max_abs)) /
            max_abs;
    }
    const float v = global_mix[i];
    scaled_mix[i] = isfinite(v) ? v * norm_scale : 0.0f;
}
