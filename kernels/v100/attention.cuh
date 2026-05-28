__device__ float e4m3fn_quant_dequant_dev(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (f8_e4m3fn_to_f32_dev((uint8_t)mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - f8_e4m3fn_to_f32_dev((uint8_t)best));
        const float next_diff = fabsf(ax - f8_e4m3fn_to_f32_dev((uint8_t)(best + 1)));
        if (next_diff < best_diff ||
            (next_diff == best_diff && (((best + 1) & 1) == 0) && ((best & 1) != 0))) {
            best++;
        }
    }
    return sign * f8_e4m3fn_to_f32_dev((uint8_t)best);
}

__global__ void kv_fp8_round_store_raw_swa_kernel(float *raw_swa,
                                                  const float *kv,
                                                  uint32_t slots,
                                                  uint32_t raw_rows,
                                                  uint32_t raw_row,
                                                  uint32_t head_dim,
                                                  uint32_t n_rot) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)head_dim;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / head_dim);
    const uint32_t col = (uint32_t)(i % head_dim);
    const uint32_t n_nope = head_dim - n_rot;
    float v = kv[i];
    if (col < n_nope) {
        const uint32_t block0 = (col / 64u) * 64u;
        float amax = 0.0f;
        for (uint32_t j = 0; j < 64u; ++j) {
            amax = fmaxf(amax, fabsf(kv[(uint64_t)slot * head_dim + block0 + j]));
        }
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        float q = v / scale;
        q = fminf(448.0f, fmaxf(-448.0f, q));
        v = e4m3fn_quant_dequant_dev(q) * scale;
    }
    v = __half2float(f32_to_half_saturate(v));
    raw_swa[((uint64_t)slot * raw_rows + raw_row) * head_dim + col] = v;
}

__device__ float rope_yarn_ramp_tp_dev(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__global__ void rope_tail_rows_kernel(float *x,
                                      uint32_t rows,
                                      uint32_t head_dim,
                                      uint32_t n_rot,
                                      uint32_t pos,
                                      uint32_t n_ctx_orig,
                                      int inverse,
                                      float freq_base,
                                      float freq_scale,
                                      float ext_factor,
                                      float attn_factor,
                                      float beta_fast,
                                      float beta_slow) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || n_rot > head_dim || (n_rot & 1u)) return;
    float *xr = x + (uint64_t)row * head_dim;
    const uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot *
                       logf((float)n_ctx_orig /
                            (beta_fast * 2.0f * (float)M_PI)) /
                       denom);
        corr1 = ceilf((float)n_rot *
                      logf((float)n_ctx_orig /
                           (beta_slow * 2.0f * (float)M_PI)) /
                      denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    float *tail = xr + n_nope;
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2u; pair += blockDim.x) {
        const uint32_t i = pair * 2u;
        const float theta_extrap =
            (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix =
                rope_yarn_ramp_tp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        if (inverse) s = -s;
        const float x0 = tail[i + 0];
        const float x1 = tail[i + 1];
        tail[i + 0] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
    }
}

__global__ void attention_raw_swa_one_row_kernel(float *out_heads,
                                                 const float *q_heads,
                                                 const float *raw_swa,
                                                 const float *sinks,
                                                 uint32_t slots,
                                                 uint32_t local_heads,
                                                 uint32_t head_dim,
                                                 uint32_t raw_rows,
                                                 uint32_t raw_row) {
    const uint32_t row = blockIdx.x;
    if (row >= slots * local_heads) return;
    const uint32_t slot = row / local_heads;
    const uint32_t local_head = row % local_heads;
    const float *q = q_heads + (uint64_t)row * head_dim;
    const float *kv =
        raw_swa + ((uint64_t)slot * raw_rows + raw_row) * (uint64_t)head_dim;
    float dot = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        dot += q[d] * kv[d];
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float score = partial[0] * rsqrtf((float)head_dim);
    const float sink = sinks[local_head];
    const float max_s = fmaxf(score, sink);
    const float row_w = expf(score - max_s);
    const float denom = row_w + expf(sink - max_s);
    const float scale = row_w / denom;
    float *out = out_heads + (uint64_t)row * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[d] = kv[d] * scale;
    }
}

__global__ void attention_raw_swa_window_kernel(float *out_heads,
                                                const float *q_heads,
                                                const float *raw_swa,
                                                const float *sinks,
                                                uint32_t slots,
                                                uint32_t local_heads,
                                                uint32_t head_dim,
                                                uint32_t raw_rows,
                                                uint32_t raw_row,
                                                uint32_t valid_rows) {
    const uint32_t row = blockIdx.x;
    if (row >= slots * local_heads || valid_rows == 0 || valid_rows > raw_rows) return;
    const uint32_t slot = row / local_heads;
    const uint32_t local_head = row % local_heads;
    const float *q = q_heads + (uint64_t)row * head_dim;
    __shared__ float partial[256];
    __shared__ float scores[128];

    float max_s = sinks[local_head];
    for (uint32_t i = 0; i < valid_rows; ++i) {
        const uint32_t history_offset = valid_rows - 1u - i;
        const uint32_t rr = (raw_row + raw_rows - history_offset) % raw_rows;
        const float *kv =
            raw_swa + ((uint64_t)slot * raw_rows + rr) * (uint64_t)head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += q[d] * kv[d];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        const float score = partial[0] * rsqrtf((float)head_dim);
        if (threadIdx.x == 0) {
            scores[i] = score;
        }
        max_s = fmaxf(max_s, score);
        __syncthreads();
    }

    float denom = expf(sinks[local_head] - max_s);
    for (uint32_t i = 0; i < valid_rows; ++i) {
        denom += expf(scores[i] - max_s);
    }
    float *out = out_heads + (uint64_t)row * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < valid_rows; ++i) {
            const uint32_t history_offset = valid_rows - 1u - i;
            const uint32_t rr = (raw_row + raw_rows - history_offset) % raw_rows;
            const float *kv =
                raw_swa + ((uint64_t)slot * raw_rows + rr) * (uint64_t)head_dim;
            const float w = expf(scores[i] - max_s) / denom;
            acc += kv[d] * w;
        }
        out[d] = acc;
    }
}

__global__ void compressor_store_slots_kernel(const float *kv,
                                              const float *score,
                                              float *state_kv,
                                              float *state_score,
                                              const float *ape,
                                              uint32_t slots,
                                              uint32_t head_dim,
                                              uint32_t ratio,
                                              uint32_t pos,
                                              uint32_t max_state_rows,
                                              uint32_t max_width) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * width;
    if (i >= n || ratio == 0u || width > max_width) return;
    const uint32_t slot = (uint32_t)(i / width);
    const uint32_t j = (uint32_t)(i - (uint64_t)slot * width);
    const uint32_t pos_mod = pos % ratio;
    const uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    if (dst_row >= max_state_rows) return;
    const uint64_t dst =
        ((uint64_t)slot * max_state_rows + dst_row) * (uint64_t)max_width + j;
    state_kv[dst] = kv[(uint64_t)slot * width + j];
    state_score[dst] = score[(uint64_t)slot * width + j] +
                       (ape ? ape[(uint64_t)pos_mod * width + j] : 0.0f);
}

__global__ void compressor_pool_emit_slots_kernel(float *rows,
                                                  const float *state_kv,
                                                  const float *state_score,
                                                  uint32_t slots,
                                                  uint32_t head_dim,
                                                  uint32_t ratio,
                                                  uint32_t comp_row,
                                                  uint32_t row_cap,
                                                  uint32_t max_state_rows,
                                                  uint32_t max_width) {
    const uint32_t slot = blockIdx.y;
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots || d >= head_dim || comp_row >= row_cap || ratio == 0u) return;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    if (width > max_width) return;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    const uint64_t slot_base = (uint64_t)slot * max_state_rows * (uint64_t)max_width;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4u; ++r) {
            vals[n_cand] = state_kv[slot_base + (uint64_t)r * max_width + d];
            scores[n_cand] = state_score[slot_base + (uint64_t)r * max_width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint64_t off =
                slot_base + (uint64_t)(ratio + r) * max_width + head_dim + d;
            vals[n_cand] = state_kv[off];
            scores[n_cand] = state_score[off];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        for (uint32_t r = 0; r < ratio && r < 128u; ++r) {
            vals[n_cand] = state_kv[slot_base + (uint64_t)r * max_width + d];
            scores[n_cand] = state_score[slot_base + (uint64_t)r * max_width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f;
    float acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; ++i) {
        const float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    rows[((uint64_t)slot * row_cap + comp_row) * (uint64_t)head_dim + d] =
        den != 0.0f && isfinite(acc) ? acc / den : 0.0f;
}

__global__ void compressor_norm_emit_slots_kernel(float *rows,
                                                  const float *weight,
                                                  uint32_t slots,
                                                  uint32_t head_dim,
                                                  uint32_t comp_row,
                                                  uint32_t row_cap,
                                                  float eps) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || comp_row >= row_cap) return;
    float *row = rows + ((uint64_t)slot * row_cap + comp_row) * (uint64_t)head_dim;
    float local_max = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = row[d];
        if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
    }
    const float max_abs = block_max_256_f32(local_max);
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const float v = row[d];
            if (isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    sum = block_sum_256_f32(sum);
    float scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        scale = rsqrtf(sum / (float)head_dim + eps / (max_abs * max_abs)) / max_abs;
    }
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = row[d];
        const float y = isfinite(v) ? v * scale * weight[d] : 0.0f;
        row[d] = isfinite(y) ? y : 0.0f;
    }
}

__device__ float compressor_pool_value_one_dim(const float *state_kv,
                                               const float *state_score,
                                               uint32_t head_dim,
                                               uint32_t ratio,
                                               uint32_t max_state_rows,
                                               uint32_t max_width,
                                               uint64_t slot_base,
                                               uint32_t d) {
    float max_s = -INFINITY;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4u; ++r) {
            max_s = fmaxf(max_s,
                          state_score[slot_base + (uint64_t)r * max_width + d]);
        }
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint64_t off =
                slot_base + (uint64_t)(ratio + r) * max_width + head_dim + d;
            max_s = fmaxf(max_s, state_score[off]);
        }
    } else {
        for (uint32_t r = 0; r < ratio && r < 128u; ++r) {
            max_s = fmaxf(max_s,
                          state_score[slot_base + (uint64_t)r * max_width + d]);
        }
    }

    float den = 0.0f;
    float acc = 0.0f;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint64_t off = slot_base + (uint64_t)r * max_width + d;
            const float w = expf(state_score[off] - max_s);
            den += w;
            acc += state_kv[off] * w;
        }
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint64_t off =
                slot_base + (uint64_t)(ratio + r) * max_width + head_dim + d;
            const float w = expf(state_score[off] - max_s);
            den += w;
            acc += state_kv[off] * w;
        }
    } else {
        for (uint32_t r = 0; r < ratio && r < 128u; ++r) {
            const uint64_t off = slot_base + (uint64_t)r * max_width + d;
            const float w = expf(state_score[off] - max_s);
            den += w;
            acc += state_kv[off] * w;
        }
    }
    return den != 0.0f && isfinite(acc) ? acc / den : 0.0f;
}

__global__ void compressor_pool_norm_emit_slots_kernel(float *rows,
                                                       const float *state_kv,
                                                       const float *state_score,
                                                       const float *weight,
                                                       uint32_t slots,
                                                       uint32_t head_dim,
                                                       uint32_t ratio,
                                                       uint32_t comp_row,
                                                       uint32_t row_cap,
                                                       uint32_t max_state_rows,
                                                       uint32_t max_width,
                                                       float eps) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || head_dim > 512u || comp_row >= row_cap || ratio == 0u) {
        return;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    if (width > max_width) return;

    __shared__ float pooled[512];
    const uint64_t slot_base =
        (uint64_t)slot * max_state_rows * (uint64_t)max_width;
    float local_max = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = compressor_pool_value_one_dim(
            state_kv, state_score, head_dim, ratio, max_state_rows, max_width,
            slot_base, d);
        pooled[d] = v;
        if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
    }
    __syncthreads();

    const float max_abs = block_max_256_f32(local_max);
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const float v = pooled[d];
            if (isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    sum = block_sum_256_f32(sum);
    float scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        scale = rsqrtf(sum / (float)head_dim + eps / (max_abs * max_abs)) / max_abs;
    }

    float *row = rows + ((uint64_t)slot * row_cap + comp_row) *
                           (uint64_t)head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float y = isfinite(pooled[d]) ? pooled[d] * scale * weight[d] : 0.0f;
        row[d] = isfinite(y) ? y : 0.0f;
    }
}

__global__ void compressor_pool_norm_rope_round_emit_slots_kernel(
    float *rows,
    const float *state_kv,
    const float *state_score,
    const float *weight,
    uint32_t slots,
    uint32_t head_dim,
    uint32_t ratio,
    uint32_t comp_row,
    uint32_t row_cap,
    uint32_t max_state_rows,
    uint32_t max_width,
    float eps,
    uint32_t n_rot,
    uint32_t pos,
    uint32_t n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || head_dim > 512u || comp_row >= row_cap ||
        ratio == 0u || n_rot > head_dim || (n_rot & 1u)) {
        return;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    if (width > max_width) return;

    __shared__ float normalized[512];
    const uint64_t slot_base =
        (uint64_t)slot * max_state_rows * (uint64_t)max_width;
    float local_max = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = compressor_pool_value_one_dim(
            state_kv, state_score, head_dim, ratio, max_state_rows, max_width,
            slot_base, d);
        normalized[d] = v;
        if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
    }
    __syncthreads();

    const float max_abs = block_max_256_f32(local_max);
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const float v = normalized[d];
            if (isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    sum = block_sum_256_f32(sum);
    float scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        scale = rsqrtf(sum / (float)head_dim + eps / (max_abs * max_abs)) / max_abs;
    }
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = normalized[d];
        const float y = isfinite(v) ? v * scale * weight[d] : 0.0f;
        normalized[d] = isfinite(y) ? y : 0.0f;
    }
    __syncthreads();

    float *row = rows + ((uint64_t)slot * row_cap + comp_row) *
                           (uint64_t)head_dim;
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t d = threadIdx.x; d < n_nope; d += blockDim.x) {
        row[d] = __half2float(f32_to_half_saturate(normalized[d]));
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot *
                       logf((float)n_ctx_orig /
                            (beta_fast * 2.0f * (float)M_PI)) /
                       denom);
        corr1 = ceilf((float)n_rot *
                      logf((float)n_ctx_orig /
                           (beta_slow * 2.0f * (float)M_PI)) /
                      denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2u; pair += blockDim.x) {
        const uint32_t i = pair * 2u;
        const float theta_extrap =
            (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix =
                rope_yarn_ramp_tp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        const float s = sinf(theta) * mscale;
        const float x0 = normalized[n_nope + i + 0];
        const float x1 = normalized[n_nope + i + 1];
        row[n_nope + i + 0] = __half2float(f32_to_half_saturate(x0 * c - x1 * s));
        row[n_nope + i + 1] = __half2float(f32_to_half_saturate(x0 * s + x1 * c));
    }
}

__global__ void rope_tail_comp_emit_slots_kernel(float *rows,
                                                 uint32_t slots,
                                                 uint32_t head_dim,
                                                 uint32_t n_rot,
                                                 uint32_t comp_row,
                                                 uint32_t row_cap,
                                                 uint32_t pos,
                                                 uint32_t n_ctx_orig,
                                                 float freq_base,
                                                 float freq_scale,
                                                 float ext_factor,
                                                 float attn_factor,
                                                 float beta_fast,
                                                 float beta_slow) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || comp_row >= row_cap || n_rot > head_dim || (n_rot & 1u)) return;
    float *xr = rows + ((uint64_t)slot * row_cap + comp_row) * (uint64_t)head_dim;
    const uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot *
                       logf((float)n_ctx_orig /
                            (beta_fast * 2.0f * (float)M_PI)) /
                       denom);
        corr1 = ceilf((float)n_rot *
                      logf((float)n_ctx_orig /
                           (beta_slow * 2.0f * (float)M_PI)) /
                      denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    float *tail = xr + n_nope;
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2u; pair += blockDim.x) {
        const uint32_t i = pair * 2u;
        const float theta_extrap =
            (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix =
                rope_yarn_ramp_tp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        const float s = sinf(theta) * mscale;
        const float x0 = tail[i + 0];
        const float x1 = tail[i + 1];
        tail[i + 0] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
    }
}

__global__ void rope_tail_round_comp_emit_slots_kernel(float *rows,
                                                       uint32_t slots,
                                                       uint32_t head_dim,
                                                       uint32_t n_rot,
                                                       uint32_t comp_row,
                                                       uint32_t row_cap,
                                                       uint32_t pos,
                                                       uint32_t n_ctx_orig,
                                                       float freq_base,
                                                       float freq_scale,
                                                       float ext_factor,
                                                       float attn_factor,
                                                       float beta_fast,
                                                       float beta_slow) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || comp_row >= row_cap || n_rot > head_dim ||
        (n_rot & 1u)) {
        return;
    }
    float *xr = rows + ((uint64_t)slot * row_cap + comp_row) *
                           (uint64_t)head_dim;
    const uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot *
                       logf((float)n_ctx_orig /
                            (beta_fast * 2.0f * (float)M_PI)) /
                       denom);
        corr1 = ceilf((float)n_rot *
                      logf((float)n_ctx_orig /
                           (beta_slow * 2.0f * (float)M_PI)) /
                      denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    for (uint32_t d = threadIdx.x; d < n_nope; d += blockDim.x) {
        xr[d] = __half2float(f32_to_half_saturate(xr[d]));
    }
    float *tail = xr + n_nope;
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2u; pair += blockDim.x) {
        const uint32_t i = pair * 2u;
        const float theta_extrap =
            (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix =
                rope_yarn_ramp_tp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        const float s = sinf(theta) * mscale;
        const float x0 = tail[i + 0];
        const float x1 = tail[i + 1];
        tail[i + 0] = __half2float(f32_to_half_saturate(x0 * c - x1 * s));
        tail[i + 1] = __half2float(f32_to_half_saturate(x0 * s + x1 * c));
    }
}

__global__ void round_comp_emit_slots_kernel(float *rows,
                                             uint32_t slots,
                                             uint32_t head_dim,
                                             uint32_t comp_row,
                                             uint32_t row_cap) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * head_dim;
    if (i >= n || comp_row >= row_cap) return;
    const uint32_t slot = (uint32_t)(i / head_dim);
    const uint32_t d = (uint32_t)(i - (uint64_t)slot * head_dim);
    float *row = rows + ((uint64_t)slot * row_cap + comp_row) * (uint64_t)head_dim;
    row[d] = __half2float(f32_to_half_saturate(row[d]));
}

__global__ void pack_comp_row_kernel(float *dst,
                                     const float *rows,
                                     uint32_t slots,
                                     uint32_t head_dim,
                                     uint32_t comp_row,
                                     uint32_t row_cap) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * head_dim;
    if (i >= n || comp_row >= row_cap) return;
    const uint32_t slot = (uint32_t)(i / head_dim);
    const uint32_t d = (uint32_t)(i - (uint64_t)slot * head_dim);
    dst[i] = rows[((uint64_t)slot * row_cap + comp_row) * (uint64_t)head_dim + d];
}

__global__ void pack_indexer_score_column_kernel(float *dst,
                                                 const float *scores,
                                                 uint32_t slots,
                                                 uint32_t top_k,
                                                 uint32_t column) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots || column >= top_k) return;
    dst[slot] = scores[(uint64_t)slot * top_k + column];
}

__global__ void compressor_shift_ratio4_slots_kernel(float *state_kv,
                                                     float *state_score,
                                                     uint32_t slots,
                                                     uint32_t width,
                                                     uint32_t max_state_rows,
                                                     uint32_t max_width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t half = 4ull * width;
    const uint64_t n = (uint64_t)slots * half;
    if (i >= n || width > max_width || max_state_rows < 8u) return;
    const uint32_t slot = (uint32_t)(i / half);
    const uint32_t j = (uint32_t)(i - (uint64_t)slot * half);
    const uint64_t base = (uint64_t)slot * max_state_rows * (uint64_t)max_width;
    const float v = state_kv[base + half + j];
    const float s = state_score[base + half + j];
    state_kv[base + j] = v;
    state_score[base + j] = s;
    state_kv[base + half + j] = v;
    state_score[base + half + j] = s;
}

__global__ void seed_single_topk_kernel(float *scores,
                                        uint32_t *topk,
                                        uint32_t slots,
                                        uint32_t top_k) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots) return;
    if (threadIdx.x == 0u) scores[slot] = 0.0f;
    for (uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
        topk[(uint64_t)slot * top_k + i] = 0u;
    }
}

__global__ void indexer_score_row0_slots_kernel(float *scores,
                                                uint32_t *topk,
                                                const float *q,
                                                const float *weights,
                                                const float *index_comp_rows,
                                                uint32_t slots,
                                                uint32_t comp_row,
                                                uint32_t row_cap,
                                                uint32_t top_k,
                                                float scale) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || comp_row >= row_cap) return;
    const float *krow =
        index_comp_rows + ((uint64_t)slot * row_cap + comp_row) *
                              (uint64_t)kIndexerHeadDim;
    __shared__ float partial[256];
    float total = 0.0f;
    for (uint32_t h = 0; h < kIndexerHead; ++h) {
        const float *qh =
            q + ((uint64_t)slot * kIndexerHead + h) * (uint64_t)kIndexerHeadDim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < kIndexerHeadDim; d += blockDim.x) {
            dot += qh[d] * krow[d];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0u) {
            total += fmaxf(partial[0], 0.0f) *
                     weights[(uint64_t)slot * kIndexerHead + h];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) scores[slot] = total * scale;
    for (uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
        topk[(uint64_t)slot * top_k + i] = 0u;
    }
}

__global__ void indexer_score_bounded_rows_slots_kernel(float *scores,
                                                        uint32_t *topk,
                                                        const float *q,
                                                        const float *weights,
                                                        const float *index_comp_rows,
                                                        uint32_t slots,
                                                        uint32_t visible_rows,
                                                        uint32_t row_cap,
                                                        uint32_t top_k,
                                                        float scale) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || visible_rows == 0u || visible_rows > row_cap) return;
    __shared__ float partial[256];
    for (uint32_t row = 0; row < visible_rows; ++row) {
        const float *krow =
            index_comp_rows + ((uint64_t)slot * row_cap + row) *
                                  (uint64_t)kIndexerHeadDim;
        float total = 0.0f;
        for (uint32_t h = 0; h < kIndexerHead; ++h) {
            const float *qh =
                q + ((uint64_t)slot * kIndexerHead + h) *
                        (uint64_t)kIndexerHeadDim;
            float dot = 0.0f;
            for (uint32_t d = threadIdx.x; d < kIndexerHeadDim; d += blockDim.x) {
                dot += qh[d] * krow[d];
            }
            partial[threadIdx.x] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0u) {
                total += fmaxf(partial[0], 0.0f) *
                         weights[(uint64_t)slot * kIndexerHead + h];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0u) {
            scores[(uint64_t)slot * top_k + row] = total * scale;
            topk[(uint64_t)slot * top_k + row] = row;
        }
    }
    for (uint32_t i = visible_rows + threadIdx.x; i < top_k; i += blockDim.x) {
        scores[(uint64_t)slot * top_k + i] = 0.0f;
        topk[(uint64_t)slot * top_k + i] = 0u;
    }
}

__global__ void attention_raw_compressed_window_kernel(float *out_heads,
                                                       const float *q_heads,
                                                       const float *raw_swa,
                                                       const float *comp_rows,
                                                       const uint32_t *topk,
                                                       const float *sinks,
                                                       uint32_t slots,
                                                       uint32_t local_heads,
                                                       uint32_t head_dim,
                                                       uint32_t raw_rows,
                                                       uint32_t raw_row,
                                                       uint32_t valid_raw_rows,
                                                       uint32_t visible_comp_rows,
                                                       uint32_t selected_comp_rows,
                                                       uint32_t comp_row_cap,
                                                       uint32_t top_k) {
    const uint32_t row = blockIdx.x;
    if (row >= slots * local_heads || valid_raw_rows == 0u ||
        valid_raw_rows > raw_rows) return;
    const uint32_t slot = row / local_heads;
    const uint32_t local_head = row % local_heads;
    const float *q = q_heads + (uint64_t)row * head_dim;
    __shared__ float partial[256];
    __shared__ float scores[kRawSwaRows + kBoundedCompRows];
    __shared__ uint32_t comp_index[kBoundedCompRows];

    uint32_t comp_count = selected_comp_rows;
    if (comp_count > visible_comp_rows) comp_count = visible_comp_rows;
    if (comp_count > comp_row_cap) comp_count = comp_row_cap;
    if (comp_count > (uint32_t)kBoundedCompRows) comp_count = (uint32_t)kBoundedCompRows;
    if (comp_count > 0u) {
        for (uint32_t i = threadIdx.x; i < comp_count; i += blockDim.x) {
            uint32_t idx = topk && i < top_k ? topk[(uint64_t)slot * top_k + i] : i;
            if (idx >= visible_comp_rows || idx >= comp_row_cap) idx = 0u;
            comp_index[i] = idx;
        }
    }
    __syncthreads();

    float max_s = sinks[local_head];
    for (uint32_t i = 0; i < valid_raw_rows; ++i) {
        const uint32_t history_offset = valid_raw_rows - 1u - i;
        const uint32_t rr = (raw_row + raw_rows - history_offset) % raw_rows;
        const float *kv =
            raw_swa + ((uint64_t)slot * raw_rows + rr) * (uint64_t)head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += q[d] * kv[d];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        const float score = partial[0] * rsqrtf((float)head_dim);
        if (threadIdx.x == 0u) scores[i] = score;
        max_s = fmaxf(max_s, score);
        __syncthreads();
    }
    for (uint32_t ci = 0; ci < comp_count; ++ci) {
        const float *kv =
            comp_rows + ((uint64_t)slot * comp_row_cap + comp_index[ci]) *
                            (uint64_t)head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += q[d] * kv[d];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        const float score = partial[0] * rsqrtf((float)head_dim);
        if (threadIdx.x == 0u) scores[valid_raw_rows + ci] = score;
        max_s = fmaxf(max_s, score);
        __syncthreads();
    }

    float denom = expf(sinks[local_head] - max_s);
    for (uint32_t i = 0; i < valid_raw_rows + comp_count; ++i) {
        denom += expf(scores[i] - max_s);
    }
    float *out = out_heads + (uint64_t)row * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < valid_raw_rows; ++i) {
            const uint32_t history_offset = valid_raw_rows - 1u - i;
            const uint32_t rr = (raw_row + raw_rows - history_offset) % raw_rows;
            const float *kv =
                raw_swa + ((uint64_t)slot * raw_rows + rr) * (uint64_t)head_dim;
            const float w = expf(scores[i] - max_s) / denom;
            acc += kv[d] * w;
        }
        for (uint32_t ci = 0; ci < comp_count; ++ci) {
            const float *kv =
                comp_rows + ((uint64_t)slot * comp_row_cap + comp_index[ci]) *
                                (uint64_t)head_dim;
            const float w = expf(scores[valid_raw_rows + ci] - max_s) / denom;
            acc += kv[d] * w;
        }
        out[d] = isfinite(acc) ? acc : 0.0f;
    }
}
