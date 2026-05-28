__global__ void zero_f32_kernel(float *dst, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = 0.0f;
}

__global__ void clamp_f32_abs_kernel(float *dst, uint64_t n, float limit) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = dst[i];
    if (!isfinite(v)) v = 0.0f;
    dst[i] = fminf(limit, fmaxf(-limit, v));
}

__global__ void ep_reduce_all_dest_shards_kernel(float *contrib,
                                                 const __half *route_hidden,
                                                 const int *route_slots,
                                                 const float *route_weights,
                                                 const int *route_totals,
                                                 int routes,
                                                 int slots,
                                                 int rank) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)routes * kHidden;
    if (i >= total) return;
    const int route = (int)(i / kHidden);
    if (route_totals && rank >= 0 && rank < kGpus && route >= route_totals[rank]) {
        return;
    }
    const int h = (int)(i % kHidden);
    const int slot = route_slots[route];
    if (slot < 0 || slot >= slots) return;
    const float w = route_weights ? route_weights[route] : kSyntheticRouteWeight;
    const int dest = h / (kHidden / kGpus);
    const int local_h = h % (kHidden / kGpus);
    const uint64_t out_idx =
        ((uint64_t)dest * slots + (uint64_t)slot) * (kHidden / kGpus) + local_h;
    atomicAdd(contrib + out_idx, __half2float(route_hidden[i]) * w);
}

__global__ void ep_pack_route_dest_shards_kernel(float *packed,
                                                 const __half *route_hidden,
                                                 const float *route_weights,
                                                 const int *route_totals,
                                                 int routes,
                                                 int segment_routes,
                                                 int rank) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)routes * kHidden;
    if (i >= total) return;
    const int route = (int)(i / kHidden);
    if (route_totals && rank >= 0 && rank < kGpus && route >= route_totals[rank]) {
        return;
    }
    const int h = (int)(i % kHidden);
    const float w = route_weights ? route_weights[route] : kSyntheticRouteWeight;
    const int dest = h / (kHidden / kGpus);
    const int local_h = h % (kHidden / kGpus);
    const uint64_t out_idx =
        ((uint64_t)dest * (uint64_t)segment_routes + (uint64_t)route) *
            (kHidden / kGpus) +
        local_h;
    packed[out_idx] = __half2float(route_hidden[i]) * w;
}

__global__ void add_f32_kernel(float *dst, const float *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void add_current_attention_shard_kernel(float *dst,
                                                   const float *current,
                                                   const float *attn,
                                                   uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c = current ? current[i] : 0.0f;
    const float a = attn ? attn[i] : 0.0f;
    float v = c + a;
    if (!isfinite(v)) v = 0.0f;
    dst[i] = v;
}

__global__ void cast_f32_to_half_kernel(__half *dst, const float *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = f32_to_half_saturate(src[i]);
}

__global__ void add_half_to_f32_kernel(float *dst, const __half *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += __half2float(src[i]);
}

__global__ void compose_next_hidden_kernel(float *next,
                                           const float *current,
                                           const float *attn,
                                           const float *shared,
                                           const float *ep_sum,
                                           int rank,
                                           int slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t elems = (uint64_t)slots * (kHidden / kGpus);
    if (i >= elems) return;
    const int slot = (int)(i / (kHidden / kGpus));
    const int local_h = (int)(i % (kHidden / kGpus));
    const float synthetic =
        ((float)(rank + 1) * 0.01f) + ((float)slot * 0.001f) +
        ((float)local_h * 0.00001f);
    const float residual = current ? current[i] : synthetic;
    next[i] = residual + attn[i] + shared[i] + ep_sum[i];
}

__global__ void compose_next_hidden_compact8_kernel(float *next,
                                                    const float *current,
                                                    const float *attn,
                                                    const float *shared,
                                                    const float *r0,
                                                    const float *r1,
                                                    const float *r2,
                                                    const float *r3,
                                                    const float *r4,
                                                    const float *r5,
                                                    const float *r6,
                                                    const float *r7,
                                                    const int *idx0,
                                                    const int *idx1,
                                                    const int *idx2,
                                                    const int *idx3,
                                                    const int *idx4,
                                                    const int *idx5,
                                                    const int *idx6,
                                                    const int *idx7,
                                                    int rank,
                                                    int slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t elems = (uint64_t)slots * (kHidden / kGpus);
    if (i >= elems) return;
    const int slot = (int)(i / (kHidden / kGpus));
    const int local_h = (int)(i % (kHidden / kGpus));
    const float synthetic =
        ((float)(rank + 1) * 0.01f) + ((float)slot * 0.001f) +
        ((float)local_h * 0.00001f);
    const float residual = current ? current[i] : synthetic;
    float ep = 0.0f;
    const int i0 = idx0[slot];
    const int i1 = idx1[slot];
    const int i2 = idx2[slot];
    const int i3 = idx3[slot];
    const int i4 = idx4[slot];
    const int i5 = idx5[slot];
    const int i6 = idx6[slot];
    const int i7 = idx7[slot];
    if (i0 >= 0) ep += r0[(uint64_t)i0 * (kHidden / kGpus) + local_h];
    if (i1 >= 0) ep += r1[(uint64_t)i1 * (kHidden / kGpus) + local_h];
    if (i2 >= 0) ep += r2[(uint64_t)i2 * (kHidden / kGpus) + local_h];
    if (i3 >= 0) ep += r3[(uint64_t)i3 * (kHidden / kGpus) + local_h];
    if (i4 >= 0) ep += r4[(uint64_t)i4 * (kHidden / kGpus) + local_h];
    if (i5 >= 0) ep += r5[(uint64_t)i5 * (kHidden / kGpus) + local_h];
    if (i6 >= 0) ep += r6[(uint64_t)i6 * (kHidden / kGpus) + local_h];
    if (i7 >= 0) ep += r7[(uint64_t)i7 * (kHidden / kGpus) + local_h];
    next[i] = residual + attn[i] + shared[i] + ep;
}

__device__ float compact_moe_sum_src_routes(const float *rows,
                                            const int *indices,
                                            const int *counts,
                                            int slot,
                                            int local_h,
                                            int top_k) {
    float acc = 0.0f;
    const int count = counts ? counts[slot] : 0;
    for (int k = 0; k < count && k < top_k; ++k) {
        const int idx = indices[(uint64_t)slot * (uint64_t)top_k + (uint64_t)k];
        if (idx >= 0) {
            acc += rows[(uint64_t)idx * (kHidden / kGpus) + local_h];
        }
    }
    return acc;
}

__global__ void compose_next_hidden_compact8_multi_kernel(
    float *next,
    const float *current,
    const float *attn,
    const float *shared,
    const float *r0,
    const float *r1,
    const float *r2,
    const float *r3,
    const float *r4,
    const float *r5,
    const float *r6,
    const float *r7,
    const int *idx0,
    const int *idx1,
    const int *idx2,
    const int *idx3,
    const int *idx4,
    const int *idx5,
    const int *idx6,
    const int *idx7,
    const int *cnt0,
    const int *cnt1,
    const int *cnt2,
    const int *cnt3,
    const int *cnt4,
    const int *cnt5,
    const int *cnt6,
    const int *cnt7,
    int rank,
    int slots,
    int top_k) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t elems = (uint64_t)slots * (kHidden / kGpus);
    if (i >= elems) return;
    const int slot = (int)(i / (kHidden / kGpus));
    const int local_h = (int)(i % (kHidden / kGpus));
    const float synthetic =
        ((float)(rank + 1) * 0.01f) + ((float)slot * 0.001f) +
        ((float)local_h * 0.00001f);
    const float residual = current ? current[i] : synthetic;
    float ep = 0.0f;
    ep += compact_moe_sum_src_routes(r0, idx0, cnt0, slot, local_h, top_k);
    ep += compact_moe_sum_src_routes(r1, idx1, cnt1, slot, local_h, top_k);
    ep += compact_moe_sum_src_routes(r2, idx2, cnt2, slot, local_h, top_k);
    ep += compact_moe_sum_src_routes(r3, idx3, cnt3, slot, local_h, top_k);
    ep += compact_moe_sum_src_routes(r4, idx4, cnt4, slot, local_h, top_k);
    ep += compact_moe_sum_src_routes(r5, idx5, cnt5, slot, local_h, top_k);
    ep += compact_moe_sum_src_routes(r6, idx6, cnt6, slot, local_h, top_k);
    ep += compact_moe_sum_src_routes(r7, idx7, cnt7, slot, local_h, top_k);
    next[i] = residual + attn[i] + shared[i] + ep;
}

__global__ void compose_next_hidden_sum8_kernel(float *next,
                                                const float *current,
                                                const float *attn,
                                                const float *shared,
                                                const float *r0,
                                                const float *r1,
                                                const float *r2,
                                                const float *r3,
                                                const float *r4,
                                                const float *r5,
                                                const float *r6,
                                                const float *r7,
                                                int rank,
                                                int slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t elems = (uint64_t)slots * (kHidden / kGpus);
    if (i >= elems) return;
    const int slot = (int)(i / (kHidden / kGpus));
    const int local_h = (int)(i % (kHidden / kGpus));
    const float synthetic =
        ((float)(rank + 1) * 0.01f) + ((float)slot * 0.001f) +
        ((float)local_h * 0.00001f);
    const float residual = current ? current[i] : synthetic;
    const float ep =
        r0[i] + r1[i] + r2[i] + r3[i] + r4[i] + r5[i] + r6[i] + r7[i];
    next[i] = residual + attn[i] + shared[i] + ep;
}
