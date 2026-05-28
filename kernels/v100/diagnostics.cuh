__global__ void compare_shared_half_input_with_current_kernel(
    unsigned long long *counts,
    unsigned int *max_bits,
    int *first_mismatch,
    const __half *actual,
    const float *current_full,
    uint32_t cols,
    uint32_t slots) {
    __shared__ unsigned long long s_mismatch[256];
    __shared__ float s_max[256];
    __shared__ int s_first[256];
    if (threadIdx.x == 0u) {
        counts[0] = (unsigned long long)slots * (unsigned long long)cols;
        counts[1] = 0ull;
        *max_bits = 0u;
        *first_mismatch = 0x7fffffff;
    }
    __syncthreads();
    unsigned long long mismatches = 0ull;
    float max_diff = 0.0f;
    int first = 0x7fffffff;
    const uint64_t n = (uint64_t)slots * (uint64_t)cols;
    for (uint64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const uint32_t slot = (uint32_t)(i / cols);
        const uint32_t col = (uint32_t)(i % cols);
        const __half expected = f32_to_half_saturate(
            current_full[(uint64_t)slot * kHidden + (uint32_t)(col % kHidden)]);
        const __half got = actual[i];
        const unsigned short eraw = *reinterpret_cast<const unsigned short *>(&expected);
        const unsigned short graw = *reinterpret_cast<const unsigned short *>(&got);
        const float diff = fabsf(__half2float(got) - __half2float(expected));
        if (diff > max_diff) max_diff = diff;
        if (eraw != graw) {
            mismatches++;
            if ((int)i < first) first = (int)i;
        }
    }
    s_mismatch[threadIdx.x] = mismatches;
    s_max[threadIdx.x] = max_diff;
    s_first[threadIdx.x] = first;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_mismatch[threadIdx.x] += s_mismatch[threadIdx.x + stride];
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x],
                                       s_max[threadIdx.x + stride]);
            s_first[threadIdx.x] = min(s_first[threadIdx.x],
                                       s_first[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        counts[1] = s_mismatch[0];
        *max_bits = __float_as_uint(s_max[0]);
        *first_mismatch = s_first[0] == 0x7fffffff ? -1 : s_first[0];
    }
}

__global__ void compare_half_input_with_f32_tensor_kernel(
    unsigned long long *counts,
    unsigned int *max_bits,
    int *first_mismatch,
    const __half *actual,
    const float *expected_f32,
    uint32_t cols,
    uint32_t slots) {
    __shared__ unsigned long long s_mismatch[256];
    __shared__ float s_max[256];
    __shared__ int s_first[256];
    if (threadIdx.x == 0u) {
        counts[0] = (unsigned long long)slots * (unsigned long long)cols;
        counts[1] = 0ull;
        *max_bits = 0u;
        *first_mismatch = 0x7fffffff;
    }
    __syncthreads();
    unsigned long long mismatches = 0ull;
    float max_diff = 0.0f;
    int first = 0x7fffffff;
    const uint64_t n = (uint64_t)slots * (uint64_t)cols;
    for (uint64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const __half expected = f32_to_half_saturate(expected_f32[i]);
        const __half got = actual[i];
        const unsigned short eraw = *reinterpret_cast<const unsigned short *>(&expected);
        const unsigned short graw = *reinterpret_cast<const unsigned short *>(&got);
        const float diff = fabsf(__half2float(got) - __half2float(expected));
        if (diff > max_diff) max_diff = diff;
        if (eraw != graw) {
            mismatches++;
            if ((int)i < first) first = (int)i;
        }
    }
    s_mismatch[threadIdx.x] = mismatches;
    s_max[threadIdx.x] = max_diff;
    s_first[threadIdx.x] = first;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_mismatch[threadIdx.x] += s_mismatch[threadIdx.x + stride];
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x],
                                       s_max[threadIdx.x + stride]);
            s_first[threadIdx.x] = min(s_first[threadIdx.x],
                                       s_first[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        counts[1] = s_mismatch[0];
        *max_bits = __float_as_uint(s_max[0]);
        *first_mismatch = s_first[0] == 0x7fffffff ? -1 : s_first[0];
    }
}

__global__ void compare_route_half_input_with_current_kernel(
    unsigned long long *counts,
    unsigned int *max_bits,
    int *first_mismatch,
    const __half *actual,
    const float *current_full,
    const int *route_slots,
    int routes_n) {
    __shared__ unsigned long long s_mismatch[256];
    __shared__ float s_max[256];
    __shared__ int s_first[256];
    if (threadIdx.x == 0u) {
        counts[0] = (unsigned long long)routes_n * (unsigned long long)kHidden;
        counts[1] = 0ull;
        *max_bits = 0u;
        *first_mismatch = 0x7fffffff;
    }
    __syncthreads();
    unsigned long long mismatches = 0ull;
    float max_diff = 0.0f;
    int first = 0x7fffffff;
    const uint64_t n = (uint64_t)routes_n * (uint64_t)kHidden;
    for (uint64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const int route = (int)(i / kHidden);
        const int h = (int)(i % kHidden);
        const int slot = route_slots[route];
        float expected_f = 0.0f;
        if (slot >= 0) {
            expected_f = current_full[(uint64_t)slot * kHidden + (uint32_t)h];
        }
        const __half expected = f32_to_half_saturate(expected_f);
        const __half got = actual[i];
        const unsigned short eraw = *reinterpret_cast<const unsigned short *>(&expected);
        const unsigned short graw = *reinterpret_cast<const unsigned short *>(&got);
        const float diff = fabsf(__half2float(got) - __half2float(expected));
        if (diff > max_diff) max_diff = diff;
        if (eraw != graw) {
            mismatches++;
            if ((int)i < first) first = (int)i;
        }
    }
    s_mismatch[threadIdx.x] = mismatches;
    s_max[threadIdx.x] = max_diff;
    s_first[threadIdx.x] = first;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_mismatch[threadIdx.x] += s_mismatch[threadIdx.x + stride];
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x],
                                       s_max[threadIdx.x + stride]);
            s_first[threadIdx.x] = min(s_first[threadIdx.x],
                                       s_first[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        counts[1] = s_mismatch[0];
        *max_bits = __float_as_uint(s_max[0]);
        *first_mismatch = s_first[0] == 0x7fffffff ? -1 : s_first[0];
    }
}

__global__ void compare_route_half_input_with_current_limited_kernel(
    unsigned long long *counts,
    unsigned int *max_bits,
    int *first_mismatch,
    const __half *actual,
    const float *current_full,
    const int *route_slots,
    const int *route_totals,
    int routes_n,
    int rank) {
    __shared__ unsigned long long s_mismatch[256];
    __shared__ float s_max[256];
    __shared__ int s_first[256];
    __shared__ int s_limit;
    if (threadIdx.x == 0u) {
        int limit = routes_n;
        if (route_totals && rank >= 0 && rank < kGpus) {
            limit = route_totals[rank];
            if (limit < 0) limit = 0;
            if (limit > routes_n) limit = routes_n;
        }
        s_limit = limit;
        counts[0] = (unsigned long long)limit * (unsigned long long)kHidden;
        counts[1] = 0ull;
        *max_bits = 0u;
        *first_mismatch = 0x7fffffff;
    }
    __syncthreads();
    unsigned long long mismatches = 0ull;
    float max_diff = 0.0f;
    int first = 0x7fffffff;
    const uint64_t n = (uint64_t)s_limit * (uint64_t)kHidden;
    for (uint64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const int route = (int)(i / kHidden);
        const int h = (int)(i % kHidden);
        const int slot = route_slots[route];
        float expected_f = 0.0f;
        if (slot >= 0) {
            expected_f = current_full[(uint64_t)slot * kHidden + (uint32_t)h];
        }
        const __half expected = f32_to_half_saturate(expected_f);
        const __half got = actual[i];
        const unsigned short eraw =
            *reinterpret_cast<const unsigned short *>(&expected);
        const unsigned short graw =
            *reinterpret_cast<const unsigned short *>(&got);
        const float diff = fabsf(__half2float(got) - __half2float(expected));
        if (diff > max_diff) max_diff = diff;
        if (eraw != graw) {
            mismatches++;
            if ((int)i < first) first = (int)i;
        }
    }
    s_mismatch[threadIdx.x] = mismatches;
    s_max[threadIdx.x] = max_diff;
    s_first[threadIdx.x] = first;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_mismatch[threadIdx.x] += s_mismatch[threadIdx.x + stride];
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x],
                                       s_max[threadIdx.x + stride]);
            s_first[threadIdx.x] = min(s_first[threadIdx.x],
                                       s_first[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        counts[1] = s_mismatch[0];
        *max_bits = __float_as_uint(s_max[0]);
        *first_mismatch = s_first[0] == 0x7fffffff ? -1 : s_first[0];
    }
}
