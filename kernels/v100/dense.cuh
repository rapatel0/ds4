__global__ void f8_b128_dense_kernel(float *out,
                                     const uint8_t *weights,
                                     const float *x,
                                     uint32_t rows,
                                     uint32_t cols,
                                     uint32_t row_stride_bytes,
                                     uint32_t slots) {
    const uint32_t row = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (row >= rows || slot >= slots) return;
    const uint8_t *wrow = weights + (uint64_t)row * row_stride_bytes;
    const float *xrow = x + (uint64_t)slot * cols;
    float acc = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const uint8_t *block = wrow + (uint64_t)(c / 128u) * 129ull;
        const float scale = f8_e8m0_to_f32_dev(block[0]);
        const float w = f8_e4m3fn_to_f32_dev(block[1u + (c % 128u)]) * scale;
        acc += w * xrow[c];
    }
    acc = block_sum_256_f32(acc);
    if (threadIdx.x == 0u) out[(uint64_t)slot * rows + row] = acc;
}

__global__ void f8_b128_dense_hmma_m16_kernel(float *out,
                                              const uint8_t *weights,
                                              const float *x,
                                              uint32_t rows,
                                              uint32_t cols,
                                              uint32_t row_stride_bytes,
                                              uint32_t slots) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    enum {
        WARPS_PER_BLOCK = 4,
        TILE_M = 16,
        TILE_N = 16,
        TILE_K = 16,
        ROWS_PER_BLOCK = WARPS_PER_BLOCK * TILE_N,
    };

    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (warp >= WARPS_PER_BLOCK) return;

    const uint32_t row_block = blockIdx.x * ROWS_PER_BLOCK;
    const uint32_t token_block = blockIdx.y * TILE_M;

    __shared__ __half a_sh[TILE_M * TILE_K];
    __shared__ __half b_sh[WARPS_PER_BLOCK * TILE_K * TILE_N];
    __shared__ float c_sh[WARPS_PER_BLOCK * TILE_M * TILE_N];

    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (uint32_t k0 = 0; k0 < cols; k0 += TILE_K) {
        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            const uint32_t token = i >> 4u;
            const uint32_t k = i & 15u;
            const uint32_t global_token = token_block + token;
            float v = 0.0f;
            if (global_token < slots) {
                v = x[(uint64_t)global_token * cols + k0 + k];
            }
            a_sh[i] = __float2half_rn(v);
        }

        for (uint32_t i = tid; i < WARPS_PER_BLOCK * TILE_K * TILE_N; i += blockDim.x) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t out_col = local >> 4u;
            const uint32_t k = local & 15u;
            const uint32_t row = row_block + wtile * TILE_N + out_col;
            float w = 0.0f;
            if (row < rows) {
                const uint32_t col = k0 + k;
                const uint8_t *row_base = weights + (uint64_t)row * row_stride_bytes;
                const uint8_t *block = row_base + (uint64_t)(col >> 7u) * 129ull;
                w = f8_e4m3fn_to_f32_dev(block[1u + (col & 127u)]) *
                    f8_e8m0_to_f32_dev(block[0]);
            }
            b_sh[i] = __float2half_rn(w);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, a_sh, TILE_K);
        wmma::load_matrix_sync(b_frag, b_sh + warp * TILE_K * TILE_N, TILE_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(c_sh + warp * TILE_M * TILE_N,
                            c_frag,
                            TILE_N,
                            wmma::mem_row_major);
    __syncthreads();

    for (uint32_t i = tid; i < WARPS_PER_BLOCK * TILE_M * TILE_N; i += blockDim.x) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t token = local >> 4u;
        const uint32_t out_col = local & 15u;
        const uint32_t global_token = token_block + token;
        const uint32_t row = row_block + wtile * TILE_N + out_col;
        if (global_token < slots && row < rows) {
            out[(uint64_t)global_token * rows + row] =
                c_sh[wtile * TILE_M * TILE_N + local];
        }
    }
#else
    (void)out;
    (void)weights;
    (void)x;
    (void)rows;
    (void)cols;
    (void)row_stride_bytes;
    (void)slots;
#endif
}

__global__ void f8_b128_to_half_kernel(__half *out,
                                       const uint8_t *weights,
                                       uint32_t rows,
                                       uint32_t cols,
                                       uint32_t row_stride_bytes) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)rows * cols;
    if (i >= n) return;
    const uint32_t row = (uint32_t)(i / cols);
    const uint32_t col = (uint32_t)(i - (uint64_t)row * cols);
    const uint8_t *row_base = weights + (uint64_t)row * row_stride_bytes;
    const uint8_t *block = row_base + (uint64_t)(col >> 7u) * 129ull;
    const float w = f8_e4m3fn_to_f32_dev(block[1u + (col & 127u)]) *
                    f8_e8m0_to_f32_dev(block[0]);
    out[i] = __float2half_rn(w);
}

__device__ float bf16_to_f32_dev(uint16_t v) {
    return __uint_as_float((uint32_t)v << 16);
}

__global__ void bf16_to_half_kernel(__half *out, const uint16_t *in, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(bf16_to_f32_dev(in[i]));
}

__global__ void bf16_dense_kernel(float *out,
                                  const uint16_t *weights,
                                  const float *x,
                                  uint32_t rows,
                                  uint32_t cols,
                                  uint32_t row_stride_elements,
                                  uint32_t slots) {
    const uint32_t row = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (row >= rows || slot >= slots) return;
    const uint16_t *wrow = weights + (uint64_t)row * row_stride_elements;
    const float *xrow = x + (uint64_t)slot * cols;
    float acc = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        acc += bf16_to_f32_dev(wrow[c]) * xrow[c];
    }
    acc = block_sum_256_f32(acc);
    if (threadIdx.x == 0u) out[(uint64_t)slot * rows + row] = acc;
}

__global__ void f32_dense_kernel(float *out,
                                 const float *weights,
                                 const float *x,
                                 uint32_t rows,
                                 uint32_t cols,
                                 uint32_t slots) {
    const uint32_t row = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (row >= rows || slot >= slots) return;
    const float *wrow = weights + (uint64_t)row * cols;
    const float *xrow = x + (uint64_t)slot * cols;
    float acc = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        acc += wrow[c] * xrow[c];
    }
    acc = block_sum_256_f32(acc);
    if (threadIdx.x == 0u) out[(uint64_t)slot * rows + row] = acc;
}
