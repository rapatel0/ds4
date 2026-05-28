__global__ void shard_top1_kernel(uint32_t *out_token,
                                  float *out_logit,
                                  const float *logits,
                                  uint32_t rows,
                                  uint32_t shard_base,
                                  uint32_t slots) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots) return;
    const float *row = logits + (uint64_t)slot * rows;
    float best = -3.4028234663852886e+38F;
    uint32_t best_idx = 0;
    for (uint32_t i = threadIdx.x; i < rows; i += blockDim.x) {
        const float v = row[i];
        if (v > best) {
            best = v;
            best_idx = i;
        }
    }
    __shared__ float s_val[256];
    __shared__ uint32_t s_idx[256];
    s_val[threadIdx.x] = best;
    s_idx[threadIdx.x] = best_idx;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other = s_val[threadIdx.x + stride];
            const uint32_t other_idx = s_idx[threadIdx.x + stride];
            if (other > s_val[threadIdx.x]) {
                s_val[threadIdx.x] = other;
                s_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        out_token[slot] = shard_base + s_idx[0];
        out_logit[slot] = s_val[0];
    }
}

__global__ void f32_dense_colmajor_kernel(float *out,
                                          const float *weights,
                                          const float *x,
                                          uint32_t rows,
                                          uint32_t cols,
                                          uint32_t slots) {
    const uint32_t row = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (row >= rows || slot >= slots) return;
    const float *xrow = x + (uint64_t)slot * cols;
    float acc = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        acc += weights[(uint64_t)c * rows + row] * xrow[c];
    }
    acc = block_sum_256_f32(acc);
    if (threadIdx.x == 0u) out[(uint64_t)slot * rows + row] = acc;
}

__global__ void router_logits_ep_from_rank_major_kernel(
    float *out,
    const float *rank_major,
    const float *norm_weight,
    const float *norm_scale,
    const float *router_w_ep,
    uint32_t shard_cols,
    uint32_t rank_count,
    uint32_t slots) {
    const uint32_t local_expert = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (local_expert >= kLocalExperts || slot >= slots) return;
    const float s = norm_scale[slot];
    float acc = 0.0f;
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
        const float x = v * s * norm_weight[h];
        const float w = router_w_ep[(uint64_t)h * kLocalExperts + local_expert];
        acc += x * w;
    }
    acc = block_sum_256_f32(acc);
    if (threadIdx.x == 0u) {
        out[(uint64_t)slot * kLocalExperts + local_expert] =
            isfinite(acc) ? acc : 0.0f;
    }
}

__global__ void router_logits_allreduce_partial_kernel(
    float *partial_logits,
    const float *current_shard,
    const float *norm_weight,
    const float *global_max,
    const float *global_sumsq,
    const float *router_w_shard,
    uint32_t rank,
    uint32_t shard_cols,
    uint32_t slots,
    float eps) {
    const uint32_t expert = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (expert >= kGlobalExperts || slot >= slots) return;
    const float max_abs = global_max[slot];
    const float sumsq = global_sumsq[slot];
    float scale = rsqrtf(eps);
    if (max_abs > 0.0f && isfinite(max_abs)) {
        scale = rsqrtf(sumsq / (float)kHidden + eps / (max_abs * max_abs)) /
                max_abs;
    }
    if (!isfinite(scale)) scale = 0.0f;
    const float *row = current_shard + (uint64_t)slot * shard_cols;
    float acc = 0.0f;
    for (uint32_t local_h = threadIdx.x; local_h < shard_cols;
         local_h += blockDim.x) {
        float v = row[local_h];
        if (!isfinite(v)) v = 0.0f;
        const uint32_t global_h = rank * shard_cols + local_h;
        const float x = v * scale * norm_weight[global_h];
        const float w =
            router_w_shard[(uint64_t)local_h * kGlobalExperts + expert];
        acc += x * w;
    }
    acc = block_sum_256_f32(acc);
    if (threadIdx.x == 0u) {
        partial_logits[(uint64_t)slot * kGlobalExperts + expert] =
            isfinite(acc) ? acc : 0.0f;
    }
}

__global__ void router_logits_rank_major_to_slot_major_kernel(
    float *out,
    const float *rank_major_logits,
    uint32_t slots) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)slots * (uint64_t)kGlobalExperts;
    if (i >= n) return;
    const uint32_t slot = (uint32_t)(i / kGlobalExperts);
    const uint32_t expert = (uint32_t)(i - (uint64_t)slot * kGlobalExperts);
    const uint32_t src_rank = expert / kLocalExperts;
    const uint32_t local_expert = expert - src_rank * kLocalExperts;
    out[i] =
        rank_major_logits[((uint64_t)src_rank * (uint64_t)slots +
                           (uint64_t)slot) *
                              (uint64_t)kLocalExperts +
                          (uint64_t)local_expert];
}

__global__ void hc_split_rows_kernel(float *split,
                                     const float *mix,
                                     const float *scale,
                                     const float *base,
                                     uint32_t slots,
                                     uint32_t sinkhorn_iters) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots) return;
    hc4_split_one_dev(split + (uint64_t)slot * kHcMix,
                      mix + (uint64_t)slot * kHcMix,
                      scale, base, sinkhorn_iters, 1.0e-6f);
}

__global__ void router_select_topk_rows_kernel(int *selected,
                                               float *weights,
                                               const float *logits,
                                               const float *bias,
                                               const int *hash,
                                               const uint32_t *tokens,
                                               const unsigned char *active,
                                               uint32_t hash_rows,
                                               uint32_t slots) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || threadIdx.x != 0u) return;
    const float *row = logits + (uint64_t)slot * kGlobalExperts;
    if (active && active[slot] == 0u) {
        for (int k = 0; k < kModelTopK; ++k) {
            selected[(uint64_t)slot * kModelTopK + (uint64_t)k] = -1;
            weights[(uint64_t)slot * kModelTopK + (uint64_t)k] = 0.0f;
        }
        return;
    }
    float prob[kGlobalExperts];
    for (int e = 0; e < kGlobalExperts; ++e) {
        const float z = row[e];
        const float softplus = z > 20.0f ? z : (z < -20.0f ? expf(z) : log1pf(expf(z)));
        prob[e] = sqrtf(fmaxf(softplus, 0.0f));
    }
    if (hash && tokens && hash_rows > 0u) {
        uint32_t tok = tokens[slot];
        if (tok >= hash_rows) tok = 0u;
        const int *hrow = hash + (uint64_t)tok * kModelTopK;
        float sum = 0.0f;
        for (int k = 0; k < kModelTopK; ++k) {
            const int e = hrow[k];
            selected[(uint64_t)slot * kModelTopK + (uint64_t)k] = e;
            const float w = (e >= 0 && e < kGlobalExperts) ? prob[e] : 0.0f;
            weights[(uint64_t)slot * kModelTopK + (uint64_t)k] = w;
            sum += w;
        }
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
        for (int k = 0; k < kModelTopK; ++k) {
            weights[(uint64_t)slot * kModelTopK + (uint64_t)k] =
                weights[(uint64_t)slot * kModelTopK + (uint64_t)k] / sum * 1.5f;
        }
        return;
    }
    int best[kModelTopK];
    float best_score[kModelTopK];
    float best_prob[kModelTopK];
    for (int i = 0; i < kModelTopK; ++i) {
        best[i] = -1;
        best_score[i] = -INFINITY;
        best_prob[i] = 0.0f;
    }
    for (int e = 0; e < kGlobalExperts; ++e) {
        const float p = prob[e];
        const float score = p + (bias ? bias[e] : 0.0f);
        for (int k = 0; k < kModelTopK; ++k) {
            if (score > best_score[k]) {
                for (int m = kModelTopK - 1; m > k; --m) {
                    best[m] = best[m - 1];
                    best_score[m] = best_score[m - 1];
                    best_prob[m] = best_prob[m - 1];
                }
                best[k] = e;
                best_score[k] = score;
                best_prob[k] = p;
                break;
            }
        }
    }
    float sum = 0.0f;
    for (int k = 0; k < kModelTopK; ++k) sum += best_prob[k];
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (int k = 0; k < kModelTopK; ++k) {
        selected[(uint64_t)slot * kModelTopK + (uint64_t)k] = best[k];
        weights[(uint64_t)slot * kModelTopK + (uint64_t)k] =
            best_prob[k] / sum * 1.5f;
    }
}

__global__ void router_select_hash_fast_rows_kernel(int *selected,
                                                    float *weights,
                                                    const float *logits,
                                                    const int *hash,
                                                    const uint32_t *tokens,
                                                    const unsigned char *active,
                                                    uint32_t hash_rows,
                                                    uint32_t slots) {
    const uint32_t slot = blockIdx.x;
    if (slot >= slots || threadIdx.x != 0u) return;
    if (active && active[slot] == 0u) {
        for (int k = 0; k < kModelTopK; ++k) {
            selected[(uint64_t)slot * kModelTopK + (uint64_t)k] = -1;
            weights[(uint64_t)slot * kModelTopK + (uint64_t)k] = 0.0f;
        }
        return;
    }
    uint32_t tok = tokens[slot];
    if (tok >= hash_rows) tok = 0u;
    const int *hrow = hash + (uint64_t)tok * kModelTopK;
    const float *row = logits + (uint64_t)slot * kGlobalExperts;
    float sum = 0.0f;
    float local[kModelTopK];
    for (int k = 0; k < kModelTopK; ++k) {
        const int e = hrow[k];
        selected[(uint64_t)slot * kModelTopK + (uint64_t)k] = e;
        float w = 0.0f;
        if (e >= 0 && e < kGlobalExperts) {
            const float z = row[e];
            const float softplus =
                z > 20.0f ? z : (z < -20.0f ? expf(z) : log1pf(expf(z)));
            w = sqrtf(fmaxf(softplus, 0.0f));
        }
        local[k] = w;
        sum += w;
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (int k = 0; k < kModelTopK; ++k) {
        weights[(uint64_t)slot * kModelTopK + (uint64_t)k] =
            local[k] / sum * 1.5f;
    }
}

__global__ void gpu_route_count_all_kernel(const int *selected,
                                           int *offsets_all,
                                           uint32_t slots,
                                           uint32_t top_k) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = slots * top_k;
    if (i >= total) return;
    const int expert = selected[i];
    if (expert < 0 || expert >= kGlobalExperts) return;
    const int rank = expert / kLocalExperts;
    const int local = expert - rank * kLocalExperts;
    atomicAdd(offsets_all + (uint64_t)rank * (kLocalExperts + 1) + local + 1, 1);
}

__global__ void gpu_route_prefix_all_kernel(int *offsets_all, int *totals) {
    const int rank = (int)threadIdx.x;
    if (rank >= kGpus) return;
    int *offsets = offsets_all + (uint64_t)rank * (kLocalExperts + 1);
    int running = 0;
    for (int local = 0; local < kLocalExperts; ++local) {
        const int count = offsets[local + 1];
        offsets[local] = running;
        running += count;
    }
    offsets[kLocalExperts] = running;
    totals[rank] = running;
}

__global__ void gpu_route_init_compact_plan_kernel(int *compact_plan,
                                                   uint32_t slots,
                                                   uint32_t top_k) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t indices = (uint64_t)kGpus * slots * top_k;
    const uint64_t counts = (uint64_t)kGpus * slots;
    if (i < indices) {
        compact_plan[i] = -1;
    }
    if (i < counts) {
        compact_plan[indices + i] = 0;
    }
}

__global__ void gpu_route_copy_own_offsets_kernel(int *dst_offsets,
                                                  const int *offsets_all,
                                                  uint32_t rank) {
    const uint32_t i = threadIdx.x;
    if (i <= (uint32_t)kLocalExperts) {
        dst_offsets[i] =
            offsets_all[(uint64_t)rank * (kLocalExperts + 1) + i];
    }
}

__global__ void gpu_route_fill_all_kernel(const int *selected,
                                          const float *weights,
                                          const int *offsets_all,
                                          int local_rank,
                                          int *route_slots,
                                          float *route_weights,
                                          int *compact_plan,
                                          uint32_t slots,
                                          uint32_t top_k) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = slots * top_k;
    if (i >= total) return;
    const int expert = selected[i];
    if (expert < 0 || expert >= kGlobalExperts) return;
    const int src_rank = expert / kLocalExperts;
    const int local = expert - src_rank * kLocalExperts;
    const uint32_t slot = i / top_k;
    const uint32_t k = i - slot * top_k;
    int prior_same_expert = 0;
    for (uint32_t j = 0; j < i; ++j) {
        if (selected[j] == expert) ++prior_same_expert;
    }
    const int route_idx =
        offsets_all[(uint64_t)src_rank * (kLocalExperts + 1) + local] +
        prior_same_expert;
    int route_order = 0;
    for (uint32_t prev_k = 0; prev_k < k; ++prev_k) {
        const int prev_expert = selected[(uint64_t)slot * top_k + prev_k];
        if (prev_expert >= 0 && prev_expert < kGlobalExperts &&
            prev_expert / kLocalExperts == src_rank) {
            ++route_order;
        }
    }
    const uint64_t indices_per_src = (uint64_t)slots * top_k;
    const uint64_t counts_base = (uint64_t)kGpus * indices_per_src;
    if (route_order < (int)top_k) {
        compact_plan[(uint64_t)src_rank * indices_per_src +
                     (uint64_t)slot * top_k + route_order] = route_idx;
    }
    atomicAdd(compact_plan + counts_base + (uint64_t)src_rank * slots + slot, 1);
    if (src_rank == local_rank) {
        route_slots[route_idx] = (int)slot;
        route_weights[route_idx] = weights[i];
    }
}

__global__ void post_attention_route_plan_audit_kernel(
    unsigned long long *audit,
    const int *offsets,
    const int *route_slots,
    const float *route_weights,
    const int *selected,
    const float *weights,
    uint32_t rank,
    uint32_t slots,
    uint32_t top_k) {
    const uint32_t local = blockIdx.x;
    if (local >= (uint32_t)kLocalExperts) return;
    const int start = offsets[local];
    const int end = offsets[local + 1u];
    const int expected_expert = (int)rank * kLocalExperts + (int)local;
    for (int route = start + (int)threadIdx.x; route < end; route += (int)blockDim.x) {
        atomicAdd(audit + 0, 1ull);
        const int slot = route_slots[route];
        if (slot < 0 || slot >= (int)slots) {
            atomicAdd(audit + 3, 1ull);
            continue;
        }
        bool found = false;
        float expected_weight = 0.0f;
        for (uint32_t k = 0; k < top_k; ++k) {
            const uint64_t idx = (uint64_t)slot * top_k + k;
            if (selected[idx] == expected_expert) {
                found = true;
                expected_weight = weights[idx];
                break;
            }
        }
        if (!found) {
            atomicAdd(audit + 1, 1ull);
            continue;
        }
        const float got_weight = route_weights[route];
        if (!isfinite(got_weight) || fabsf(got_weight - expected_weight) > 1.0e-5f) {
            atomicAdd(audit + 2, 1ull);
        }
    }
}
