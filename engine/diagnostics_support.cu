void log_route_half_stats(const char *tag, int layer, int rank_id,
                          const __half *ptr, size_t elems, cudaStream_t stream) {
    if (!ptr || elems == 0) return;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<__half> host(elems);
    CHECK_CUDA(cudaMemcpy(host.data(), ptr, elems * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    int finite_bad = 0;
    size_t first_bad = (size_t)-1;
    float max_abs = 0.0f;
    for (size_t i = 0; i < elems; ++i) {
        const float v = __half2float(host[i]);
        if (!std::isfinite(v)) {
            if (finite_bad == 0) first_bad = i;
            ++finite_bad;
        } else {
            max_abs = fmaxf(max_abs, fabsf(v));
        }
    }
    std::fprintf(stderr,
                 "tp_ep_route_tensor_stats\ttag\t%s\tlayer\t%d\trank\t%d\telems\t%zu\tfinite_bad\t%d\tfirst_bad\t%zu\tmax_abs\t%.9g\n",
                 tag, layer, rank_id, elems, finite_bad, first_bad, max_abs);
}

void merge_tensor_stats(TensorF32Stats *dst, const TensorF32Stats &src) {
    if (!dst) return;
    if (src.finite_bad != 0 && dst->finite_bad == 0) {
        dst->first_bad = src.first_bad;
    }
    dst->finite_bad += src.finite_bad;
    dst->max_abs = fmaxf(dst->max_abs, src.max_abs);
}

TensorF32Stats collect_tensor_f32_stats(const float *ptr, size_t elems,
                                        cudaStream_t stream) {
    TensorF32Stats stats;
    if (!ptr || elems == 0) return stats;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> host(elems);
    CHECK_CUDA(cudaMemcpy(host.data(), ptr, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elems; ++i) {
        const float v = host[i];
        if (!std::isfinite(v)) {
            if (stats.finite_bad == 0) stats.first_bad = i;
            ++stats.finite_bad;
        } else {
            stats.max_abs = fmaxf(stats.max_abs, fabsf(v));
        }
    }
    return stats;
}

TensorF32Stats collect_raw_swa_row_stats(const float *ptr, uint32_t slots,
                                         uint32_t raw_rows, uint32_t raw_row,
                                         uint32_t head_dim,
                                         cudaStream_t stream) {
    TensorF32Stats stats;
    if (!ptr || slots == 0 || raw_rows == 0 || raw_row >= raw_rows ||
        head_dim == 0) {
        return stats;
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> host((size_t)slots * (size_t)head_dim);
    const float *src = ptr + (uint64_t)raw_row * (uint64_t)head_dim;
    CHECK_CUDA(cudaMemcpy2D(host.data(), (size_t)head_dim * sizeof(float),
                            src,
                            (size_t)raw_rows * (size_t)head_dim * sizeof(float),
                            (size_t)head_dim * sizeof(float), (size_t)slots,
                            cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < host.size(); ++i) {
        const float v = host[i];
        if (!std::isfinite(v)) {
            if (stats.finite_bad == 0) stats.first_bad = i;
            ++stats.finite_bad;
        } else {
            stats.max_abs = fmaxf(stats.max_abs, fabsf(v));
        }
    }
    return stats;
}

TensorF32DiffStats collect_tensor_f32_diff_stats(const float *a, const float *b,
                                                 size_t elems,
                                                 cudaStream_t stream) {
    TensorF32DiffStats stats;
    if (!a || !b || elems == 0) return stats;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> ha(elems);
    std::vector<float> hb(elems);
    CHECK_CUDA(cudaMemcpy(ha.data(), a, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hb.data(), b, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elems; ++i) {
        const float av = ha[i];
        const float bv = hb[i];
        if (!std::isfinite(av) || !std::isfinite(bv)) {
            if (stats.bad == 0) stats.first_bad = i;
            ++stats.bad;
            continue;
        }
        const float diff = fabsf(av - bv);
        const float denom = fmaxf(fabsf(bv), 1.0e-12f);
        stats.max_abs = fmaxf(stats.max_abs, diff);
        stats.max_rel = fmaxf(stats.max_rel, diff / denom);
    }
    return stats;
}

void log_tensor_f32_diff_summary(const char *tag, int layer,
                                 const float *got, const float *ref,
                                 size_t elems, cudaStream_t stream) {
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> hg(elems);
    std::vector<float> hr(elems);
    CHECK_CUDA(cudaMemcpy(hg.data(), got, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hr.data(), ref, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double sq = 0.0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float got_max = 0.0f;
    float ref_max = 0.0f;
    int finite_bad = 0;
    size_t first_bad = (size_t)-1;
    for (size_t i = 0; i < elems; ++i) {
        const float g = hg[i];
        const float r = hr[i];
        if (!std::isfinite(g) || !std::isfinite(r)) {
            if (finite_bad == 0) first_bad = i;
            ++finite_bad;
            continue;
        }
        got_max = fmaxf(got_max, fabsf(g));
        ref_max = fmaxf(ref_max, fabsf(r));
        const float diff = fabsf(g - r);
        const float rel = diff / fmaxf(fabsf(r), 1.0e-12f);
        max_abs = fmaxf(max_abs, diff);
        max_rel = fmaxf(max_rel, rel);
        sq += (double)diff * (double)diff;
    }
    const double rms = elems ? std::sqrt(sq / (double)elems) : 0.0;
    const char *status = (finite_bad == 0 && max_abs <= 1.0e-5f) ? "PASS" : "DIFF";
    std::printf("tp_ep_compressed_reference_diff\tlayer\t%d\ttensor\t%s\t"
                "elems\t%zu\tmax_abs\t%.9g\trms\t%.9g\tmax_rel\t%.9g\t"
                "finite_bad\t%d\tfirst_bad\t%zu\tgot_max\t%.9g\t"
                "reference_max\t%.9g\t%s\n",
                layer, tag, elems, max_abs, rms, max_rel, finite_bad,
                first_bad, got_max, ref_max, status);
}

void log_tensor_f32_stats(const char *tag, int layer, int rank_id,
                          const float *ptr, size_t elems, cudaStream_t stream) {
    const TensorF32Stats stats = collect_tensor_f32_stats(ptr, elems, stream);
    std::fprintf(stderr,
                 "tp_ep_tensor_stats\ttag\t%s\tlayer\t%d\trank\t%d\telems\t%zu\tfinite_bad\t%d\tfirst_bad\t%zu\tmax_abs\t%.9g\n",
                 tag, layer, rank_id, elems, stats.finite_bad, stats.first_bad,
                 stats.max_abs);
}

void log_hc_current_full_rank_parity(const Options &opt,
                                     RankState ranks[kGpus],
                                     int layer,
                                     size_t elems) {
    if (elems == 0 || !ranks[0].d_current_full) return;
    CHECK_CUDA(cudaSetDevice(ranks[0].device));
    if (ranks[0].stream) CHECK_CUDA(cudaStreamSynchronize(ranks[0].stream));
    std::vector<float> ref(elems);
    std::vector<float> got(elems);
    CHECK_CUDA(cudaMemcpy(ref.data(), ranks[0].d_current_full,
                          elems * sizeof(float), cudaMemcpyDeviceToHost));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.d_current_full) continue;
        CHECK_CUDA(cudaSetDevice(r.device));
        if (r.stream) CHECK_CUDA(cudaStreamSynchronize(r.stream));
        CHECK_CUDA(cudaMemcpy(got.data(), r.d_current_full,
                              elems * sizeof(float), cudaMemcpyDeviceToHost));
        unsigned long long mismatches = 0;
        size_t first_mismatch = (size_t)-1;
        float max_abs = 0.0f;
        int finite_bad = 0;
        for (size_t i = 0; i < elems; ++i) {
            const float a = got[i];
            const float b = ref[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                if (first_mismatch == (size_t)-1) first_mismatch = i;
                ++mismatches;
                ++finite_bad;
                continue;
            }
            const float diff = fabsf(a - b);
            if (diff > 0.0f) {
                if (first_mismatch == (size_t)-1) first_mismatch = i;
                ++mismatches;
                max_abs = fmaxf(max_abs, diff);
            }
        }
        std::printf("tp_ep_hc_current_full_rank_diff\tlayer\t%d\trank\t%d\t"
                    "elems\t%zu\tmismatches\t%llu\tfirst_mismatch\t%zu\t"
                    "max_abs\t%.9g\tfinite_bad\t%d\t%s\n",
                    layer, rank, elems,
                    (unsigned long long)mismatches,
                    first_mismatch, max_abs, finite_bad,
                    mismatches == 0ull ? "PASS" : "DIFF");
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
}

HalfInputDiffStats collect_shared_half_input_diff(RankState &r,
                                                  const __half *actual,
                                                  const float *current_full,
                                                  uint32_t cols,
                                                  uint32_t slots,
                                                  cudaStream_t stream) {
    HalfInputDiffStats stats;
    if (!actual || !current_full || !r.d_half_diff_counts ||
        !r.d_half_diff_max_bits || !r.d_half_diff_first ||
        cols == 0 || slots == 0) {
        return stats;
    }
    compare_shared_half_input_with_current_kernel<<<1, 256, 0, stream>>>(
        r.d_half_diff_counts, r.d_half_diff_max_bits, r.d_half_diff_first,
        actual, current_full, cols, slots);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    unsigned long long counts[2] = {};
    unsigned int max_bits = 0u;
    CHECK_CUDA(cudaMemcpy(counts, r.d_half_diff_counts, sizeof(counts),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&max_bits, r.d_half_diff_max_bits, sizeof(max_bits),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&stats.first_mismatch, r.d_half_diff_first,
                          sizeof(stats.first_mismatch),
                          cudaMemcpyDeviceToHost));
    stats.compared = counts[0];
    stats.mismatches = counts[1];
    std::memcpy(&stats.max_abs, &max_bits, sizeof(stats.max_abs));
    return stats;
}

HalfInputDiffStats collect_half_input_tensor_diff(RankState &r,
                                                  const __half *actual,
                                                  const float *expected_f32,
                                                  uint32_t cols,
                                                  uint32_t slots,
                                                  cudaStream_t stream) {
    HalfInputDiffStats stats;
    if (!actual || !expected_f32 || !r.d_half_diff_counts ||
        !r.d_half_diff_max_bits || !r.d_half_diff_first ||
        cols == 0 || slots == 0) {
        return stats;
    }
    compare_half_input_with_f32_tensor_kernel<<<1, 256, 0, stream>>>(
        r.d_half_diff_counts, r.d_half_diff_max_bits, r.d_half_diff_first,
        actual, expected_f32, cols, slots);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    unsigned long long counts[2] = {};
    unsigned int max_bits = 0u;
    CHECK_CUDA(cudaMemcpy(counts, r.d_half_diff_counts, sizeof(counts),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&max_bits, r.d_half_diff_max_bits, sizeof(max_bits),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&stats.first_mismatch, r.d_half_diff_first,
                          sizeof(stats.first_mismatch),
                          cudaMemcpyDeviceToHost));
    stats.compared = counts[0];
    stats.mismatches = counts[1];
    std::memcpy(&stats.max_abs, &max_bits, sizeof(stats.max_abs));
    return stats;
}

HalfInputDiffStats collect_route_half_input_diff(RankState &r,
                                                 const __half *actual,
                                                 const float *current_full,
                                                 const int *route_slots,
                                                 int routes,
                                                 cudaStream_t stream) {
    HalfInputDiffStats stats;
    if (!actual || !current_full || !route_slots || !r.d_half_diff_counts ||
        !r.d_half_diff_max_bits || !r.d_half_diff_first || routes <= 0) {
        return stats;
    }
    compare_route_half_input_with_current_kernel<<<1, 256, 0, stream>>>(
        r.d_half_diff_counts, r.d_half_diff_max_bits, r.d_half_diff_first,
        actual, current_full, route_slots, routes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    unsigned long long counts[2] = {};
    unsigned int max_bits = 0u;
    CHECK_CUDA(cudaMemcpy(counts, r.d_half_diff_counts, sizeof(counts),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&max_bits, r.d_half_diff_max_bits, sizeof(max_bits),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&stats.first_mismatch, r.d_half_diff_first,
                          sizeof(stats.first_mismatch),
                          cudaMemcpyDeviceToHost));
    stats.compared = counts[0];
    stats.mismatches = counts[1];
    std::memcpy(&stats.max_abs, &max_bits, sizeof(stats.max_abs));
    return stats;
}

HalfInputDiffStats collect_route_half_input_diff_limited(
    RankState &r,
    const __half *actual,
    const float *current_full,
    const int *route_slots,
    const int *route_totals,
    int routes,
    int rank,
    cudaStream_t stream) {
    HalfInputDiffStats stats;
    if (!actual || !current_full || !route_slots || !r.d_half_diff_counts ||
        !r.d_half_diff_max_bits || !r.d_half_diff_first || routes <= 0) {
        return stats;
    }
    compare_route_half_input_with_current_limited_kernel<<<1, 256, 0, stream>>>(
        r.d_half_diff_counts, r.d_half_diff_max_bits, r.d_half_diff_first,
        actual, current_full, route_slots, route_totals, routes, rank);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    unsigned long long counts[2] = {};
    unsigned int max_bits = 0u;
    CHECK_CUDA(cudaMemcpy(counts, r.d_half_diff_counts, sizeof(counts),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&max_bits, r.d_half_diff_max_bits, sizeof(max_bits),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&stats.first_mismatch, r.d_half_diff_first,
                          sizeof(stats.first_mismatch),
                          cudaMemcpyDeviceToHost));
    stats.compared = counts[0];
    stats.mismatches = counts[1];
    std::memcpy(&stats.max_abs, &max_bits, sizeof(stats.max_abs));
    return stats;
}

void log_half_input_diff(const char *family,
                         int layer,
                         int rank,
                         const HalfInputDiffStats &stats) {
    const char *status = stats.mismatches == 0ull ? "PASS" : "DIFF";
    std::printf("tp_ep_rank_major_half_input_diff\tlayer\t%d\trank\t%d\t"
                "family\t%s\tcompared\t%llu\tmismatches\t%llu\t"
                "first_mismatch\t%d\tmax_abs\t%.9g\t%s\n",
                layer, rank, family,
                (unsigned long long)stats.compared,
                (unsigned long long)stats.mismatches,
                stats.first_mismatch, stats.max_abs, status);
}

void log_attention_projection_input_diff(const char *family,
                                         int layer,
                                         int rank,
                                         const HalfInputDiffStats &stats) {
    const char *status = stats.mismatches == 0ull ? "PASS" : "DIFF";
    std::printf("tp_ep_attention_projection_input_diff\tlayer\t%d\trank\t%d\t"
                "family\t%s\tcompared\t%llu\tmismatches\t%llu\t"
                "first_mismatch\t%d\tmax_abs\t%.9g\t%s\n",
                layer, rank, family,
                (unsigned long long)stats.compared,
                (unsigned long long)stats.mismatches,
                stats.first_mismatch, stats.max_abs, status);
}

unsigned short f32_to_half_raw_host(float v) {
    if (!std::isfinite(v)) v = 0.0f;
    v = std::fmin(kFp16Max, std::fmax(-kFp16Max, v));
    const __half h = __float2half(v);
    unsigned short raw = 0u;
    std::memcpy(&raw, &h, sizeof(raw));
    return raw;
}

float rank_major_debug_scale(const std::vector<float> &src,
                             uint32_t slot,
                             uint32_t shard_cols,
                             uint32_t ranks,
                             uint32_t slots,
                             float eps) {
    const uint32_t cols = shard_cols * ranks;
    float max_abs = 0.0f;
    for (uint32_t col = 0; col < cols; ++col) {
        const uint32_t src_rank = col / shard_cols;
        const uint32_t local_col = col - src_rank * shard_cols;
        const uint64_t src_i =
            ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot) *
                (uint64_t)shard_cols +
            (uint64_t)local_col;
        const float v = src[src_i];
        if (std::isfinite(v)) max_abs = std::fmax(max_abs, std::fabs(v));
    }
    float sum = 0.0f;
    if (max_abs > 0.0f && std::isfinite(max_abs)) {
        for (uint32_t col = 0; col < cols; ++col) {
            const uint32_t src_rank = col / shard_cols;
            const uint32_t local_col = col - src_rank * shard_cols;
            const uint64_t src_i =
                ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot) *
                    (uint64_t)shard_cols +
                (uint64_t)local_col;
            const float v = src[src_i];
            if (std::isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    if (!(max_abs > 0.0f) || !std::isfinite(max_abs)) {
        return 1.0f / std::sqrt(eps);
    }
    return 1.0f / std::sqrt(sum / (float)cols + eps / (max_abs * max_abs)) /
           max_abs;
}

float slot_major_debug_scale(const std::vector<float> &src,
                             uint32_t slot,
                             uint32_t cols,
                             float eps) {
    float max_abs = 0.0f;
    const uint64_t base = (uint64_t)slot * (uint64_t)cols;
    for (uint32_t col = 0; col < cols; ++col) {
        const float v = src[base + col];
        if (std::isfinite(v)) max_abs = std::fmax(max_abs, std::fabs(v));
    }
    float sum = 0.0f;
    if (max_abs > 0.0f && std::isfinite(max_abs)) {
        for (uint32_t col = 0; col < cols; ++col) {
            const float v = src[base + col];
            if (std::isfinite(v)) {
                const float scaled = v / max_abs;
                sum += scaled * scaled;
            }
        }
    }
    if (!(max_abs > 0.0f) || !std::isfinite(max_abs)) {
        return 1.0f / std::sqrt(eps);
    }
    return 1.0f / std::sqrt(sum / (float)cols + eps / (max_abs * max_abs)) /
           max_abs;
}

void log_attention_rank_major_input_debug(
    const char *family,
    int layer,
    RankState &r,
    const __half *actual,
    const float *expected_f32,
    const float *slot_major,
    const float *rank_major,
    const float *weight,
    uint32_t slots,
    cudaStream_t stream) {
    if (!actual || !expected_f32 || !slot_major || !rank_major || !weight ||
        slots == 0) {
        return;
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const uint32_t shard_cols = kHidden / kGpus;
    const size_t elems = (size_t)slots * (size_t)kHidden;
    std::vector<__half> h_actual(elems);
    std::vector<float> h_expected(elems);
    std::vector<float> h_slot(elems);
    std::vector<float> h_rank_major(elems);
    std::vector<float> h_weight(kHidden);
    CHECK_CUDA(cudaMemcpy(h_actual.data(), actual,
                          elems * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_expected.data(), expected_f32,
                          elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_slot.data(), slot_major,
                          elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_rank_major.data(), rank_major,
                          elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_weight.data(), weight,
                          kHidden * sizeof(float), cudaMemcpyDeviceToHost));

    unsigned long long raw_mismatches = 0ull;
    size_t raw_first = (size_t)-1;
    float raw_max_abs = 0.0f;
    for (uint32_t slot = 0; slot < slots; ++slot) {
        for (uint32_t col = 0; col < (uint32_t)kHidden; ++col) {
            const uint32_t src_rank = col / shard_cols;
            const uint32_t local_col = col - src_rank * shard_cols;
            const uint64_t src_i =
                ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot) *
                    (uint64_t)shard_cols +
                (uint64_t)local_col;
            const uint64_t slot_i =
                (uint64_t)slot * (uint64_t)kHidden + (uint64_t)col;
            const float a = h_rank_major[src_i];
            const float b = h_slot[slot_i];
            const float diff = std::fabs(a - b);
            if (diff > 0.0f || !std::isfinite(a) || !std::isfinite(b)) {
                if (raw_first == (size_t)-1) raw_first = (size_t)slot_i;
                ++raw_mismatches;
                raw_max_abs = std::fmax(raw_max_abs, diff);
            }
        }
    }

    int first_half = -1;
    unsigned short got_raw = 0u;
    unsigned short exp_raw = 0u;
    float got = 0.0f;
    float expected = 0.0f;
    for (size_t i = 0; i < elems; ++i) {
        std::memcpy(&got_raw, &h_actual[i], sizeof(got_raw));
        exp_raw = f32_to_half_raw_host(h_expected[i]);
        if (got_raw != exp_raw) {
            first_half = (int)i;
            got = __half2float(h_actual[i]);
            expected = __half2float(__float2half(h_expected[i]));
            break;
        }
    }

    uint32_t slot = 0u;
    uint32_t col = 0u;
    uint64_t src_i = 0u;
    float rank_major_value = 0.0f;
    float slot_major_value = 0.0f;
    float norm_weight = 0.0f;
    float slot_scale = 0.0f;
    float rank_major_scale = 0.0f;
    if (first_half >= 0) {
        slot = (uint32_t)((uint32_t)first_half / (uint32_t)kHidden);
        col = (uint32_t)((uint32_t)first_half % (uint32_t)kHidden);
        const uint32_t src_rank = col / shard_cols;
        const uint32_t local_col = col - src_rank * shard_cols;
        src_i = ((uint64_t)src_rank * (uint64_t)slots + (uint64_t)slot) *
                    (uint64_t)shard_cols +
                (uint64_t)local_col;
        rank_major_value = h_rank_major[src_i];
        slot_major_value = h_slot[(uint64_t)slot * (uint64_t)kHidden + col];
        norm_weight = h_weight[col];
        slot_scale = slot_major_debug_scale(h_slot, slot, (uint32_t)kHidden,
                                            1.0e-6f);
        rank_major_scale = rank_major_debug_scale(
            h_rank_major, slot, shard_cols, (uint32_t)kGpus, slots, 1.0e-6f);
    }

    std::printf("tp_ep_attention_rank_major_input_debug\tlayer\t%d\t"
                "family\t%s\traw_mismatches\t%llu\traw_first\t%zu\t"
                "raw_max_abs\t%.9g\tfirst_half_mismatch\t%d\tslot\t%u\t"
                "col\t%u\tsrc_index\t%llu\trank_major_value\t%.9g\t"
                "slot_major_value\t%.9g\tweight\t%.9g\tgot_half\t%.9g\t"
                "expected_half\t%.9g\tslot_scale\t%.9g\t"
                "rank_major_scale\t%.9g\t%s\n",
                layer, family, (unsigned long long)raw_mismatches, raw_first,
                raw_max_abs, first_half, slot, col,
                (unsigned long long)src_i, rank_major_value, slot_major_value,
                norm_weight, got, expected, slot_scale, rank_major_scale,
                (raw_mismatches == 0ull && first_half < 0) ? "PASS" : "DIFF");
}

bool should_log_routed_semantic_stats(const Options &opt) {
    if (opt.decode_cudagraph_gate || opt.true_ds4_semantic_skip_stats_gate) {
        return false;
    }
    if (!opt.routed_ffn_norm_input_gate) return false;
    if (opt.layer <= 2) return true;
    return opt.reference_hc_reduce_gate && opt.layer >= 30 && opt.layer <= 32;
}

bool should_log_reference_hc_window(const Options &opt) {
    return opt.reference_hc_reduce_gate && opt.layer >= 30 && opt.layer <= 32;
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

int check_repeat(RankState &rank, const Api &api, double *max_abs, int *bad, int *nan) {
    const Options no_executor_cap{};
    const size_t elems = (size_t)rank.routes * kHidden;
    std::vector<__half> first(elems);
    std::vector<__half> second(elems);
    CHECK_CUDA(cudaSetDevice(rank.device));
    if (run_gate(rank, api, rank.routes) != 0 ||
        run_down(rank, api, no_executor_cap) != 0) return 1;
    CHECK_CUDA(cudaStreamSynchronize(rank.stream));
    CHECK_CUDA(cudaMemcpy(first.data(), rank.d_down, elems * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    if (run_gate(rank, api, rank.routes) != 0 ||
        run_down(rank, api, no_executor_cap) != 0) return 1;
    CHECK_CUDA(cudaStreamSynchronize(rank.stream));
    CHECK_CUDA(cudaMemcpy(second.data(), rank.d_down, elems * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elems; ++i) {
        const float a = __half2float(first[i]);
        const float b = __half2float(second[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++*nan;
            continue;
        }
        const double diff = std::fabs((double)a - (double)b);
        *max_abs = std::max(*max_abs, diff);
        if (diff > 0.0) ++*bad;
    }
    return 0;
}

void build_offsets_for_rank(int rank, int slots, int top_k,
                            std::vector<int> *offsets,
                            std::vector<int> *route_slots,
                            std::vector<float> *route_weights,
                            int *routes,
                            int *active_experts,
                            int *max_routes_per_expert) {
    std::vector<int> counts((size_t)kLocalExperts, 0);
    for (int slot = 0; slot < slots; ++slot) {
        for (int k = 0; k < top_k; ++k) {
            const int dst_rank = (slot * top_k + k) % kGpus;
            if (dst_rank != rank) continue;
            const int local = (slot + k * 7 + rank) % kPackedLocalExperts;
            counts[(size_t)local]++;
        }
    }
    offsets->assign((size_t)kLocalExperts + 1, 0);
    int running = 0;
    int active = 0;
    int max_routes = 0;
    for (int e = 0; e < kLocalExperts; ++e) {
        (*offsets)[(size_t)e] = running;
        running += counts[(size_t)e];
        if (counts[(size_t)e] > 0) ++active;
        max_routes = std::max(max_routes, counts[(size_t)e]);
    }
    (*offsets)[(size_t)kLocalExperts] = running;
    if (route_slots) {
        route_slots->assign((size_t)running, -1);
        if (route_weights) route_weights->assign((size_t)running, kSyntheticRouteWeight);
        std::vector<int> cursor = *offsets;
        for (int slot = 0; slot < slots; ++slot) {
            for (int k = 0; k < top_k; ++k) {
                const int dst_rank = (slot * top_k + k) % kGpus;
                if (dst_rank != rank) continue;
                const int local = (slot + k * 7 + rank) % kPackedLocalExperts;
                const int idx = cursor[(size_t)local]++;
                (*route_slots)[(size_t)idx] = slot;
                if (route_weights) (*route_weights)[(size_t)idx] = kSyntheticRouteWeight;
            }
        }
    }
    *routes = running;
    *active_experts = active;
    *max_routes_per_expert = max_routes;
}

void build_route_index_by_slot_for_rank(int rank, int slots, int top_k,
                                        std::vector<int> *route_index_by_slot) {
    std::vector<int> offsets;
    std::vector<int> route_slots;
    int routes = 0;
    int active_experts = 0;
    int max_routes_per_expert = 0;
    build_offsets_for_rank(rank, slots, top_k, &offsets, &route_slots, nullptr, &routes,
                           &active_experts, &max_routes_per_expert);
    route_index_by_slot->assign((size_t)slots, -1);
    for (int route = 0; route < routes; ++route) {
        const int slot = route_slots[(size_t)route];
        if (slot >= 0 && slot < slots) {
            (*route_index_by_slot)[(size_t)slot] = route;
        }
    }
}

void build_route_indices_by_slot_for_rank(int rank, int slots, int top_k,
                                          std::vector<int> *route_indices,
                                          std::vector<int> *route_counts) {
    std::vector<int> offsets;
    std::vector<int> route_slots;
    int routes = 0;
    int active_experts = 0;
    int max_routes_per_expert = 0;
    build_offsets_for_rank(rank, slots, top_k, &offsets, &route_slots, nullptr, &routes,
                           &active_experts, &max_routes_per_expert);
    route_indices->assign((size_t)slots * (size_t)top_k, -1);
    route_counts->assign((size_t)slots, 0);
    for (int route = 0; route < routes; ++route) {
        const int slot = route_slots[(size_t)route];
        if (slot < 0 || slot >= slots) continue;
        int &count = (*route_counts)[(size_t)slot];
        if (count < top_k) {
            (*route_indices)[(size_t)slot * (size_t)top_k + (size_t)count] = route;
            count++;
        }
    }
}

size_t compact_route_plan_ints(const Options &opt) {
    const size_t indices = (size_t)opt.slots * (size_t)opt.top_k;
    const size_t counts = (size_t)opt.slots;
    return (size_t)kGpus * (indices + counts);
}

void bind_compact_route_plan(RankState *r, const Options &opt) {
    const size_t indices = (size_t)opt.slots * (size_t)opt.top_k;
    const size_t counts = (size_t)opt.slots;
    int *base = r->d_route_compact_plan;
    for (int src = 0; src < kGpus; ++src) {
        r->d_route_indices_by_slot[src] = base + (size_t)src * indices;
        r->d_route_count_by_slot[src] =
            base + (size_t)kGpus * indices + (size_t)src * counts;
    }
}

