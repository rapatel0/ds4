__global__ void checksum_bytes_kernel(const unsigned char *data, uint64_t n,
                                      unsigned long long *out) {
    unsigned long long local = 0;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (uint64_t)blockDim.x * gridDim.x) {
        local += (unsigned long long)data[i] * (unsigned long long)((i % 251u) + 1u);
    }
    atomicAdd(out, local);
}

__global__ void copy_f32_kernel(float *dst, const float *src, uint64_t n) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (uint64_t)blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

__global__ void copy_compact_active_route_shard_kernel(
    float *dst,
    const float *src,
    const int *route_totals,
    int src_rank,
    int routes,
    int shard_cols) {
    const uint64_t n = (uint64_t)routes * (uint64_t)shard_cols;
    int active = routes;
    if (route_totals && src_rank >= 0 && src_rank < kGpus) {
        active = route_totals[src_rank];
        if (active < 0) active = 0;
        if (active > routes) active = routes;
    }
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (uint64_t)blockDim.x * gridDim.x) {
        const int route = (int)(i / (uint64_t)shard_cols);
        dst[i] = route < active ? src[i] : 0.0f;
    }
}

__global__ void copy_i32_kernel(int *dst, const int *src, uint64_t n) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (uint64_t)blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

__global__ void copy_u32_kernel(uint32_t *dst, const uint32_t *src, uint64_t n) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (uint64_t)blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

__device__ float f8_e8m0_to_f32_dev(uint8_t e) {
    return __uint_as_float(e == 0 ? 0x00400000u : ((uint32_t)e << 23));
}

__device__ float f8_e4m3fn_to_f32_dev(uint8_t x) {
    const uint32_t sign = ((uint32_t)x & 0x80u) << 24;
    const uint32_t ax = (uint32_t)x & 0x7fu;
    if (ax == 0) return __uint_as_float(sign ? 0x80000000u : 0u);
    if (ax == 0x7f) return __uint_as_float(0x7fc00000u);
    const uint32_t exp = ax >> 3;
    const uint32_t man = ax & 0x07u;
    if (exp != 0) {
        return __uint_as_float(sign | ((exp + 120u) << 23) | (man << 20));
    }
    const uint32_t hi = man >= 4u ? 2u : (man >= 2u ? 1u : 0u);
    const uint32_t mant = (man << (23u - hi)) & 0x007fffffu;
    return __uint_as_float(sign | ((118u + hi) << 23) | mant);
}

__device__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ float block_sum_256_f32(float v) {
    __shared__ float warp_sums[8];
    __shared__ float block_sum;
    v = warp_sum_f32(v);
    if ((threadIdx.x & 31u) == 0u) warp_sums[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < 8u ? warp_sums[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32u) v = warp_sum_f32(v);
    if (threadIdx.x == 0u) block_sum = v;
    __syncthreads();
    return block_sum;
}

__device__ float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ float block_max_256_f32(float v) {
    __shared__ float warp_maxes[8];
    __shared__ float block_max;
    v = warp_max_f32(v);
    if ((threadIdx.x & 31u) == 0u) warp_maxes[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < 8u ? warp_maxes[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32u) v = warp_max_f32(v);
    if (threadIdx.x == 0u) block_max = v;
    __syncthreads();
    return block_max;
}

__device__ __half f32_to_half_saturate(float v) {
    if (!isfinite(v)) return __float2half(0.0f);
    v = fminf(kFp16Max, fmaxf(-kFp16Max, v));
    return __float2half(v);
}
