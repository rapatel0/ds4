__global__ void fill_two_hidden_inputs_half_from_rank_major_norm_kernel(
    __half *dst0,
    __half *dst1,
    const float *rank_major,
    const float *weight,
    uint32_t shard_cols,
    uint32_t ranks,
    uint32_t slots,
    float eps) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots) return;
    const uint32_t cols = shard_cols * ranks;
    float local_max = 0.0f;
    for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const uint32_t src_rank = col / shard_cols;
        const uint32_t local_col = col - src_rank * shard_cols;
        const uint64_t src_i =
            ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot) *
                (uint64_t)shard_cols +
            (uint64_t)local_col;
        const float v = rank_major[src_i];
        if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
    }
    const float max_abs = block_max_256_f32(local_max);
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
            const uint32_t src_rank = col / shard_cols;
            const uint32_t local_col = col - src_rank * shard_cols;
            const uint64_t src_i =
                ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot) *
                    (uint64_t)shard_cols +
                (uint64_t)local_col;
            const float v = rank_major[src_i];
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
    for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const uint32_t src_rank = col / shard_cols;
        const uint32_t local_col = col - src_rank * shard_cols;
        const uint64_t src_i =
            ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot) *
                (uint64_t)shard_cols +
            (uint64_t)local_col;
        const float v = rank_major[src_i];
        const float y = isfinite(v) ? v * scale * weight[col] : 0.0f;
        const __half h = f32_to_half_saturate(isfinite(y) ? y : 0.0f);
        const uint64_t dst_i = (uint64_t)slot * (uint64_t)cols + (uint64_t)col;
        dst0[dst_i] = h;
        dst1[dst_i] = h;
    }
}

__global__ void pack_current_full_to_routes_kernel(__half *routes,
                                                   const float *current_full,
                                                   const int *route_slots,
                                                   int routes_n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)routes_n * (uint64_t)kHidden;
    if (i >= n) return;
    const int route = (int)(i / kHidden);
    const int h = (int)(i % kHidden);
    const int slot = route_slots[route];
    routes[i] = f32_to_half_saturate(current_full[(uint64_t)slot * kHidden + h]);
}

__global__ void pack_current_full_to_routes_scaled_kernel(__half *routes,
                                                          float *route_inv_scale,
                                                          const float *current_full,
                                                          const int *route_slots,
                                                          int routes_n,
                                                          float target_abs) {
    const int route = (int)blockIdx.x;
    if (route >= routes_n) return;
    const int slot = route_slots[route];
    float max_abs = 0.0f;
    for (int h = (int)threadIdx.x; h < kHidden; h += (int)blockDim.x) {
        float v = current_full[(uint64_t)slot * kHidden + h];
        if (!isfinite(v)) v = 0.0f;
        max_abs = fmaxf(max_abs, fabsf(v));
    }
    __shared__ float s_max[256];
    s_max[threadIdx.x] = max_abs;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x],
                                       s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float safe_target = fmaxf(target_abs, 1.0f);
    const float scale = s_max[0] > safe_target ? safe_target / s_max[0] : 1.0f;
    if (threadIdx.x == 0u) {
        route_inv_scale[route] = scale > 0.0f ? 1.0f / scale : 1.0f;
    }
    for (int h = (int)threadIdx.x; h < kHidden; h += (int)blockDim.x) {
        float v = current_full[(uint64_t)slot * kHidden + h];
        if (!isfinite(v)) v = 0.0f;
        routes[(uint64_t)route * kHidden + h] = f32_to_half_saturate(v * scale);
    }
}

__global__ void pack_rank_major_norm_current_to_routes_kernel(
    __half *routes,
    const float *rank_major,
    const float *norm_weight,
    const int *route_slots,
    const int *route_totals,
    int routes_n,
    uint32_t rank,
    uint32_t shard_cols,
    uint32_t rank_count,
    uint32_t slots,
    float eps) {
    const int route = (int)blockIdx.x;
    if (route >= routes_n) return;
    if (route_totals && rank < (uint32_t)kGpus && route >= route_totals[rank]) {
        return;
    }
    const int slot_i = route_slots[route];
    float local_max = 0.0f;
    if (slot_i >= 0 && (uint32_t)slot_i < slots) {
        for (uint32_t h = threadIdx.x; h < kHidden; h += blockDim.x) {
            const uint32_t src_rank = h / shard_cols;
            const uint32_t local_h = h - src_rank * shard_cols;
            float v = 0.0f;
            if (src_rank < rank_count) {
                const uint64_t src_i =
                    ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot_i) *
                        (uint64_t)shard_cols +
                    (uint64_t)local_h;
                v = rank_major[src_i];
            }
            if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
        }
    }
    const float max_abs = block_max_256_f32(local_max);
    float sum = 0.0f;
    if (max_abs > 0.0f && isfinite(max_abs) &&
        slot_i >= 0 && (uint32_t)slot_i < slots) {
        for (uint32_t h = threadIdx.x; h < kHidden; h += blockDim.x) {
            const uint32_t src_rank = h / shard_cols;
            const uint32_t local_h = h - src_rank * shard_cols;
            float v = 0.0f;
            if (src_rank < rank_count) {
                const uint64_t src_i =
                    ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot_i) *
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
    float scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        scale = rsqrtf(sum / (float)kHidden + eps / (max_abs * max_abs)) /
                max_abs;
    }
    for (uint32_t h = threadIdx.x; h < kHidden; h += blockDim.x) {
        float v = 0.0f;
        if (slot_i >= 0 && (uint32_t)slot_i < slots) {
            const uint32_t src_rank = h / shard_cols;
            const uint32_t local_h = h - src_rank * shard_cols;
            if (src_rank < rank_count) {
                const uint64_t src_i =
                    ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot_i) *
                        (uint64_t)shard_cols +
                    (uint64_t)local_h;
                v = rank_major[src_i];
            }
        }
        if (!isfinite(v)) v = 0.0f;
        const float y = v * scale * norm_weight[h];
        routes[(uint64_t)route * (uint64_t)kHidden + (uint64_t)h] =
            f32_to_half_saturate(isfinite(y) ? y : 0.0f);
    }
}

__global__ void pack_rank_major_norm_current_to_routes_scaled_kernel(
    __half *routes,
    float *route_inv_scale,
    const float *rank_major,
    const float *norm_weight,
    const int *route_slots,
    const int *route_totals,
    int routes_n,
    uint32_t rank,
    uint32_t shard_cols,
    uint32_t rank_count,
    uint32_t slots,
    float eps,
    float target_abs) {
    const int route = (int)blockIdx.x;
    if (route >= routes_n) return;
    if (route_totals && rank < (uint32_t)kGpus && route >= route_totals[rank]) {
        if (threadIdx.x == 0u && route_inv_scale) route_inv_scale[route] = 1.0f;
        return;
    }
    const int slot_i = route_slots[route];
    float local_max = 0.0f;
    if (slot_i >= 0 && (uint32_t)slot_i < slots) {
        for (uint32_t h = threadIdx.x; h < kHidden; h += blockDim.x) {
            const uint32_t src_rank = h / shard_cols;
            const uint32_t local_h = h - src_rank * shard_cols;
            float v = 0.0f;
            if (src_rank < rank_count) {
                const uint64_t src_i =
                    ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot_i) *
                        (uint64_t)shard_cols +
                    (uint64_t)local_h;
                v = rank_major[src_i];
            }
            if (isfinite(v)) local_max = fmaxf(local_max, fabsf(v));
        }
    }
    const float max_input_abs = block_max_256_f32(local_max);
    float sum = 0.0f;
    if (max_input_abs > 0.0f && isfinite(max_input_abs) &&
        slot_i >= 0 && (uint32_t)slot_i < slots) {
        for (uint32_t h = threadIdx.x; h < kHidden; h += blockDim.x) {
            const uint32_t src_rank = h / shard_cols;
            const uint32_t local_h = h - src_rank * shard_cols;
            float v = 0.0f;
            if (src_rank < rank_count) {
                const uint64_t src_i =
                    ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot_i) *
                        (uint64_t)shard_cols +
                    (uint64_t)local_h;
                v = rank_major[src_i];
            }
            if (!isfinite(v)) v = 0.0f;
            const float scaled = v / max_input_abs;
            sum += scaled * scaled;
        }
    }
    __shared__ float s_max[256];
    sum = block_sum_256_f32(sum);
    float norm_scale = rsqrtf(eps);
    if (max_input_abs > 0.0f && isfinite(max_input_abs)) {
        norm_scale =
            rsqrtf(sum / (float)kHidden + eps / (max_input_abs * max_input_abs)) /
            max_input_abs;
    }
    float max_abs = 0.0f;
    if (slot_i >= 0 && (uint32_t)slot_i < slots) {
        for (uint32_t h = threadIdx.x; h < kHidden; h += blockDim.x) {
            const uint32_t src_rank = h / shard_cols;
            const uint32_t local_h = h - src_rank * shard_cols;
            float v = 0.0f;
            if (src_rank < rank_count) {
                const uint64_t src_i =
                    ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot_i) *
                        (uint64_t)shard_cols +
                    (uint64_t)local_h;
                v = rank_major[src_i];
            }
            if (!isfinite(v)) v = 0.0f;
            max_abs = fmaxf(max_abs, fabsf(v * norm_scale * norm_weight[h]));
        }
    }
    s_max[threadIdx.x] = max_abs;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x],
                                       s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float safe_target = fmaxf(target_abs, 1.0f);
    const float scale = s_max[0] > safe_target ? safe_target / s_max[0] : 1.0f;
    if (threadIdx.x == 0u) {
        route_inv_scale[route] = scale > 0.0f ? 1.0f / scale : 1.0f;
    }
    for (uint32_t h = threadIdx.x; h < kHidden; h += blockDim.x) {
        float v = 0.0f;
        if (slot_i >= 0 && (uint32_t)slot_i < slots) {
            const uint32_t src_rank = h / shard_cols;
            const uint32_t local_h = h - src_rank * shard_cols;
            if (src_rank < rank_count) {
                const uint64_t src_i =
                    ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot_i) *
                        (uint64_t)shard_cols +
                    (uint64_t)local_h;
                v = rank_major[src_i];
            }
        }
        if (!isfinite(v)) v = 0.0f;
        const float y = v * norm_scale * norm_weight[h] * scale;
        routes[(uint64_t)route * (uint64_t)kHidden + (uint64_t)h] =
            f32_to_half_saturate(isfinite(y) ? y : 0.0f);
    }
}

__global__ void shared_swiglu_shard_to_float_kernel(float *mid,
                                                    const float *gate,
                                                    const float *up,
                                                    uint32_t rank,
                                                    uint32_t rows_per_gpu,
                                                    uint32_t slots,
                                                    float clamp) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)rows_per_gpu;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / rows_per_gpu);
    const uint32_t local = (uint32_t)(i % rows_per_gpu);
    float g = gate[(uint64_t)slot * rows_per_gpu + local];
    float u = up[(uint64_t)slot * rows_per_gpu + local];
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    const float silu = g / (1.0f + expf(-g));
    mid[(uint64_t)slot * kMid + (uint64_t)rank * rows_per_gpu + local] =
        silu * u;
}

__global__ void routed_fused_gate_up_swiglu_clamp_kernel(__half *mid,
                                                         const __half *gate_up,
                                                         const float *route_inv_scale,
                                                         uint64_t routes,
                                                         float clamp) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = routes * (uint64_t)kMid;
    if (i >= n) return;
    const uint64_t row = i / kMid;
    const uint64_t col = i - row * (uint64_t)kMid;
    const uint64_t base = row * (uint64_t)kFusedN + col;
    float g = __half2float(gate_up[base]);
    float u = __half2float(gate_up[base + kMid]);
    if (route_inv_scale) {
        const float inv_scale = route_inv_scale[row];
        g *= inv_scale;
        u *= inv_scale;
    }
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    const float silu = g / (1.0f + expf(-g));
    mid[i] = __float2half(silu * u);
}

__global__ void fill_dense_input_from_current_kernel(float *dst,
                                                     const float *current_full,
                                                     uint32_t cols,
                                                     uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / cols);
    const uint32_t col = (uint32_t)(i % cols);
    dst[i] = current_full[(uint64_t)slot * kHidden + (uint32_t)(col % kHidden)];
}

__global__ void fill_dense_input_half_from_current_kernel(__half *dst,
                                                          const float *current_full,
                                                          uint32_t cols,
                                                          uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)cols;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / cols);
    const uint32_t col = (uint32_t)(i % cols);
    dst[i] = f32_to_half_saturate(current_full[(uint64_t)slot * kHidden +
                                               (uint32_t)(col % kHidden)]);
}

__global__ void fill_two_hidden_inputs_half_from_current_kernel(__half *dst_a,
                                                                __half *dst_b,
                                                                const float *current_full,
                                                                uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)kHidden;
    if (i >= n) return;
    const __half v = f32_to_half_saturate(current_full[i]);
    dst_a[i] = v;
    dst_b[i] = v;
}

__global__ void hc_current_fused_fill_pack_kernel(
    float *rank_current_full,
    const float *state_current_full,
    const float *dense_current_full,
    const float *route_current_full,
    float *attn_x,
    uint32_t attn_cols,
    float *shared_x,
    uint32_t shared_cols,
    __half *attn_x_half,
    __half *shared_x_half,
    __half *routes,
    const int *route_slots,
    int routes_n,
    uint32_t slots,
    uint64_t total) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint64_t full_elems = (uint64_t)slots * (uint64_t)kHidden;
    if (i < full_elems) {
        rank_current_full[i] = state_current_full[i];
    }
    const uint64_t attn_elems = (uint64_t)slots * (uint64_t)attn_cols;
    if (i < attn_elems) {
        const uint32_t slot = (uint32_t)(i / attn_cols);
        const uint32_t col = (uint32_t)(i % attn_cols);
        const float v =
            dense_current_full[(uint64_t)slot * kHidden + (uint32_t)(col % kHidden)];
        if (attn_x) attn_x[i] = v;
        if (attn_x_half) attn_x_half[i] = f32_to_half_saturate(v);
    }
    const uint64_t shared_elems = (uint64_t)slots * (uint64_t)shared_cols;
    if (i < shared_elems) {
        const uint32_t slot = (uint32_t)(i / shared_cols);
        const uint32_t col = (uint32_t)(i % shared_cols);
        const float v =
            dense_current_full[(uint64_t)slot * kHidden + (uint32_t)(col % kHidden)];
        if (shared_x) shared_x[i] = v;
        if (shared_x_half) shared_x_half[i] = f32_to_half_saturate(v);
    }
    const uint64_t route_elems = (uint64_t)routes_n * (uint64_t)kHidden;
    if (i < route_elems) {
        const int route = (int)(i / kHidden);
        const int h = (int)(i % kHidden);
        const int slot = route_slots[route];
        routes[i] = f32_to_half_saturate(
            route_current_full[(uint64_t)slot * kHidden + h]);
    }
}

__global__ void fill_attn_compressed_inputs_half_kernel(__half *attn_kv,
                                                        __half *attn_gate,
                                                        const float *current_full,
                                                        uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)kHidden;
    if (i >= n) return;
    const __half v = f32_to_half_saturate(current_full[i]);
    attn_kv[i] = v;
    attn_gate[i] = v;
}

__global__ void fill_ratio4_compressed_indexer_inputs_half_kernel(
    __half *attn_kv,
    __half *attn_gate,
    __half *indexer_proj,
    __half *indexer_kv,
    __half *indexer_gate,
    const float *current_full,
    uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)kHidden;
    if (i >= n) return;
    const __half v = f32_to_half_saturate(current_full[i]);
    attn_kv[i] = v;
    attn_gate[i] = v;
    indexer_proj[i] = v;
    indexer_kv[i] = v;
    indexer_gate[i] = v;
}

__global__ void hc_expand_shard_kernel(float *out_hc,
                                       const float *block_out,
                                       const float *residual_hc,
                                       const float *split,
                                       uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t shard_cols = kHidden / kGpus;
    const uint64_t elems = (uint64_t)slots * kHcRows * shard_cols;
    if (i >= elems) return;
    const uint32_t local_h = (uint32_t)(i % shard_cols);
    const uint32_t dst_hc = (uint32_t)((i / shard_cols) & 3ull);
    const uint32_t slot = (uint32_t)(i / ((uint64_t)kHcRows * shard_cols));
    const float *sp = split + (uint64_t)slot * kHcMix;
    const uint64_t slot_hc_base = (uint64_t)slot * kHcRows * shard_cols;
    float acc = block_out[(uint64_t)slot * shard_cols + local_h] * sp[4u + dst_hc];
    for (uint32_t src_hc = 0; src_hc < kHcRows; ++src_hc) {
        const float comb = sp[8u + dst_hc + src_hc * kHcRows];
        const float res = residual_hc[slot_hc_base + (uint64_t)src_hc * shard_cols + local_h];
        acc += comb * res;
    }
    out_hc[i] = acc;
}
