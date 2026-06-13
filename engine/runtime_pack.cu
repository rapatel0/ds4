std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (start <= line.size()) {
        const size_t tab = line.find('\t', start);
        if (tab == std::string::npos) {
            fields.emplace_back(line.substr(start));
            break;
        }
        fields.emplace_back(line.substr(start, tab - start));
        start = tab + 1;
    }
    return fields;
}

bool safe_sidecar_name(const std::string &name) {
    return !name.empty() &&
           name.find('/') == std::string::npos &&
           name.find('\\') == std::string::npos &&
           name.find("..") == std::string::npos;
}

std::string path_join(const char *dir, const std::string &base) {
    std::string out(dir ? dir : "");
    if (!out.empty() && out.back() != '/') out.push_back('/');
    out += base;
    return out;
}

int read_exact_at(const std::string &path, uint64_t offset, void *dst, size_t bytes) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "cannot open sidecar %s: %s\n", path.c_str(), std::strerror(errno));
        return 1;
    }
    if (fseeko(fp, (off_t)offset, SEEK_SET) != 0) {
        std::fprintf(stderr, "cannot seek sidecar %s offset %llu: %s\n",
                     path.c_str(), (unsigned long long)offset, std::strerror(errno));
        std::fclose(fp);
        return 2;
    }
    const size_t got = std::fread(dst, 1, bytes, fp);
    if (got != bytes) {
        std::fprintf(stderr, "short read sidecar %s offset %llu bytes %zu got %zu\n",
                     path.c_str(), (unsigned long long)offset, bytes, got);
        std::fclose(fp);
        return 3;
    }
    std::fclose(fp);
    return 0;
}


constexpr uint64_t kMiB = 1024ull * 1024ull;

bool should_report_vram(const Options &opt) {
    return opt.vram_min_free_mib > 0;
}

bool nccl_gate_active(const Options &opt) {
    return opt.nccl_reduce_scatter_compose_gate ||
           opt.tp_hc_current_input_nccl_allgather_gate ||
           opt.tp_hc_current_allreduce_gate ||
           opt.true_ds4_attention_output_nccl_allgather_gate;
}

int report_vram_checkpoint_min_free(const Options &opt,
                                    const char *label,
                                    uint64_t min_free_mib_threshold) {
    const uint64_t min_free_bytes = min_free_mib_threshold * kMiB;
    uint64_t min_free_mib = UINT64_MAX;
    uint64_t max_used_mib = 0;
    int failures = 0;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        size_t free_b = 0;
        size_t total_b = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
        const uint64_t used_b = (uint64_t)total_b - (uint64_t)free_b;
        const uint64_t free_mib = (uint64_t)free_b / kMiB;
        const uint64_t used_mib = used_b / kMiB;
        const uint64_t total_mib = (uint64_t)total_b / kMiB;
        min_free_mib = std::min(min_free_mib, free_mib);
        max_used_mib = std::max(max_used_mib, used_mib);
        const bool pass =
            min_free_mib_threshold == 0 || (uint64_t)free_b >= min_free_bytes;
        if (!pass) failures++;
        std::printf("tp_ep_vram\tlabel\t%s\tgpu\t%d\tfree_mib\t%llu\t"
                    "used_mib\t%llu\ttotal_mib\t%llu\tmin_free_mib\t%llu\t%s\n",
                    label, gpu,
                    (unsigned long long)free_mib,
                    (unsigned long long)used_mib,
                    (unsigned long long)total_mib,
                    (unsigned long long)min_free_mib_threshold,
                    pass ? "PASS" : "FAIL");
    }
    if (min_free_mib == UINT64_MAX) min_free_mib = 0;
    std::printf("tp_ep_vram_summary\tlabel\t%s\tmin_free_mib\t%llu\t"
                "max_used_mib\t%llu\tthreshold_mib\t%llu\tfailures\t%d\t%s\n",
                label,
                (unsigned long long)min_free_mib,
                (unsigned long long)max_used_mib,
                (unsigned long long)min_free_mib_threshold,
                failures,
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

int report_vram_checkpoint(const Options &opt, const char *label) {
    if (!should_report_vram(opt)) return 0;
    return report_vram_checkpoint_min_free(opt, label, opt.vram_min_free_mib);
}

int report_nccl_vram_checkpoint(const Options &opt, const char *label) {
    if (!nccl_gate_active(opt) || opt.nccl_min_free_mib == 0) return 0;
    return report_vram_checkpoint_min_free(opt, label, opt.nccl_min_free_mib);
}

int check_planned_vram_allocation(const Options &opt,
                                  const char *label,
                                  const uint64_t planned_bytes[kGpus]) {
    if (!should_report_vram(opt)) return 0;
    const uint64_t min_free_bytes = opt.vram_min_free_mib * kMiB;
    int failures = 0;
    for (int gpu = 0; gpu < kGpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        size_t free_b = 0;
        size_t total_b = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
        const uint64_t required_b = planned_bytes[gpu] + min_free_bytes;
        const bool pass = (uint64_t)free_b >= required_b;
        if (!pass) failures++;
        std::printf("tp_ep_vram_plan\tlabel\t%s\tgpu\t%d\tfree_mib\t%llu\t"
                    "planned_mib\t%llu\tthreshold_mib\t%llu\ttotal_mib\t%llu\t%s\n",
                    label, gpu,
                    (unsigned long long)((uint64_t)free_b / kMiB),
                    (unsigned long long)(planned_bytes[gpu] / kMiB),
                    (unsigned long long)opt.vram_min_free_mib,
                    (unsigned long long)((uint64_t)total_b / kMiB),
                    pass ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}



bool parse_contract_row(const std::vector<std::string> &f, ContractRow *out) {
    if (f.size() < 23) return false;
    ContractRow r;
    r.record_type = f[0];
    r.tensor_id = f[1];
    if (!parse_int(f[3].c_str(), &r.layer)) return false;
    r.family = f[4];
    r.source_dtype = f[5];
    r.source_shape = f[6];
    r.runtime_layout = f[7];
    if (!parse_int(f[8].c_str(), &r.owning_gpu)) return false;
    if (!parse_int(f[9].c_str(), &r.tp_rank)) return false;
    if (!parse_int(f[10].c_str(), &r.ep_rank)) return false;
    if (!parse_int(f[12].c_str(), &r.shard_index)) return false;
    if (!parse_int(f[13].c_str(), &r.shard_count)) return false;
    if (!parse_int(f[14].c_str(), &r.expert_first)) return false;
    if (!parse_int(f[15].c_str(), &r.expert_count)) return false;
    if (!parse_int(f[16].c_str(), &r.kv_ratio)) return false;
    if (!parse_u64(f[17].c_str(), &r.kv_rows_per_slot)) return false;
    if (!parse_u64(f[18].c_str(), &r.bytes_estimate)) return false;
    r.source_pack_file = f[19];
    if (!parse_u64(f[20].c_str(), &r.source_shard_offset)) return false;
    if (!parse_u64(f[21].c_str(), &r.source_byte_length)) return false;
    r.kernel_family = f[22];
    if (!safe_sidecar_name(r.source_pack_file) && r.source_pack_file != "-") return false;
    *out = r;
    return true;
}

void enqueue_graph_f32_copy_between_devices(const Options &opt,
                                            int dst_device,
                                            int src_device,
                                            float *dst,
                                            const float *src,
                                            uint64_t elems,
                                            cudaStream_t stream,
                                            int block) {
    (void)dst_device;
    (void)src_device;
    (void)opt;
    copy_f32_kernel<<<(unsigned int)((elems + (uint64_t)block - 1) /
                                     (uint64_t)block),
                      block, 0, stream>>>(dst, src, elems);
    CHECK_CUDA(cudaGetLastError());
}

void enqueue_graph_f32_copy_from_device0(const Options &opt,
                                         RankState &rank_state,
                                         int /*rank*/,
                                         float *dst,
                                         const float *src,
                                         uint64_t elems,
                                         cudaStream_t stream,
                                         int block) {
    enqueue_graph_f32_copy_between_devices(opt, rank_state.device, opt.devices[0],
                                           dst, src, elems, stream, block);
}

void enqueue_graph_i32_copy_from_device0(const Options &opt,
                                         RankState &rank_state,
                                         int /*rank*/,
                                         int *dst,
                                       const int *src,
                                       uint64_t elems,
                                       cudaStream_t stream,
                                       int block) {
    (void)opt;
    (void)rank_state;
    copy_i32_kernel<<<(unsigned int)((elems + (uint64_t)block - 1) /
                                     (uint64_t)block),
                      block, 0, stream>>>(dst, src, elems);
    CHECK_CUDA(cudaGetLastError());
}

int nccl_broadcast_bytes_from_rank(RankState ranks[kGpus],
                                   int root,
                                   const void *src_root,
                                   void *dst_by_rank[kGpus],
                                   size_t bytes,
                                   const char *label,
                                   bool epret_class = false) {
    /* s601: epret_class routes the broadcast onto the EP-return-class comm
     * (alias of the compose comm unless DS4_V100_TP_EP_COMM_SPLIT=epret). */
    if (root < 0 || root >= kGpus || !src_root || !dst_by_rank || bytes == 0) {
        return 1;
    }
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.compose_nccl_initialized || !r.compose_nccl ||
            !dst_by_rank[rank]) {
            std::fprintf(stderr,
                         "tp_ep_nccl_broadcast_missing\tlabel\t%s\t"
                         "rank\t%d\tcompose\t%d\tdst\t%d\n",
                         label ? label : "-", rank,
                         (r.compose_nccl_initialized && r.compose_nccl) ? 1 : 0,
                         dst_by_rank[rank] ? 1 : 0);
            return 2;
        }
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        const void *send = rank == root ? src_root : dst_by_rank[rank];
        CHECK_NCCL(ncclBroadcast(send, dst_by_rank[rank], bytes, ncclChar, root,
                                 epret_class ? ds4_comm_epret(r)
                                             : ds4_comm_hc(r),
                                 r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    CHECK_CUDA(cudaSetDevice(prior_device));
    return 0;
}

int nccl_broadcast_bytes_from_rank0(RankState ranks[kGpus],
                                    const void *src_rank0,
                                    void *dst_by_rank[kGpus],
                                    size_t bytes,
                                    const char *label) {
    return nccl_broadcast_bytes_from_rank(ranks, 0, src_rank0, dst_by_rank,
                                          bytes, label);
}

int broadcast_ep_return_slices(RankState ranks[kGpus],
                               bool fp16,
                               bool skip_self_copy,
                               uint64_t src_stride_elems,
                               const uint64_t copy_elems_by_src[kGpus],
                               const char *label,
                               bool skip_stream_sync = false) {
    /* s598: skip_stream_sync=true is the graph-capture-safe mode -- the
     * caller is inside stream capture, host stream syncs are illegal, and
     * downstream consumers are ordered by the per-rank stream + the
     * existing cross-rank barriers. */
    if (!copy_elems_by_src || src_stride_elems == 0) return 1;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int src = 0; src < kGpus; ++src) {
        const uint64_t copy_elems = copy_elems_by_src[src];
        if (copy_elems == 0) continue;
        if (copy_elems > src_stride_elems) return 6;
        const uint64_t bcast_elems = (uint64_t)kGpus * copy_elems;
        const size_t elem_bytes = fp16 ? sizeof(__half) : sizeof(float);
        const size_t bcast_bytes = (size_t)(bcast_elems * elem_bytes);
        void *scratch_by_rank[kGpus] = {};
        for (int rank = 0; rank < kGpus; ++rank) {
            scratch_by_rank[rank] = fp16
                ? (void *)ranks[rank].d_ep_contrib_half_bcast_all
                : (void *)ranks[rank].d_ep_contrib_bcast_all;
            if (!scratch_by_rank[rank]) return 2;
        }
        const void *src_all = fp16
            ? (const void *)ranks[src].d_ep_contrib_half_all
            : (const void *)ranks[src].d_ep_contrib_all;
        if (!src_all) return 3;
        if (copy_elems != src_stride_elems) {
            RankState &sr = ranks[src];
            CHECK_CUDA(cudaSetDevice(sr.device));
            const size_t copy_bytes = (size_t)(copy_elems * elem_bytes);
            const size_t src_pitch = (size_t)(src_stride_elems * elem_bytes);
            if (fp16) {
                CHECK_CUDA(cudaMemcpy2DAsync(
                    sr.d_ep_contrib_half_bcast_all, copy_bytes,
                    sr.d_ep_contrib_half_all, src_pitch, copy_bytes, kGpus,
                    cudaMemcpyDeviceToDevice, sr.stream));
                src_all = sr.d_ep_contrib_half_bcast_all;
            } else {
                CHECK_CUDA(cudaMemcpy2DAsync(
                    sr.d_ep_contrib_bcast_all, copy_bytes,
                    sr.d_ep_contrib_all, src_pitch, copy_bytes, kGpus,
                    cudaMemcpyDeviceToDevice, sr.stream));
                src_all = sr.d_ep_contrib_bcast_all;
            }
        }
        if (nccl_broadcast_bytes_from_rank(
                ranks, src, src_all, scratch_by_rank, bcast_bytes,
                label ? label : "ep_return_broadcast",
                /*epret_class=*/true) != 0) {
            return 4;
        }
        const size_t copy_bytes = (size_t)(copy_elems * elem_bytes);
        for (int dst = 0; dst < kGpus; ++dst) {
            if (skip_self_copy && src == dst) continue;
            RankState &r = ranks[dst];
            CHECK_CUDA(cudaSetDevice(r.device));
            const uint64_t offset_elems = (uint64_t)dst * copy_elems;
            if (fp16) {
                if (!r.d_ep_remote_half[src]) return 5;
                const __half *src_ptr =
                    r.d_ep_contrib_half_bcast_all + offset_elems;
                CHECK_CUDA(cudaMemcpyAsync(r.d_ep_remote_half[src], src_ptr,
                                           copy_bytes, cudaMemcpyDeviceToDevice,
                                           r.stream));
            } else {
                if (!r.d_ep_remote[src]) return 5;
                const float *src_ptr = r.d_ep_contrib_bcast_all + offset_elems;
                CHECK_CUDA(cudaMemcpyAsync(r.d_ep_remote[src], src_ptr,
                                           copy_bytes, cudaMemcpyDeviceToDevice,
                                           r.stream));
            }
        }
    }
    if (!skip_stream_sync) {
        for (int rank = 0; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            CHECK_CUDA(cudaStreamSynchronize(ranks[rank].stream));
        }
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
    return 0;
}

/* ===================== Sprint 601 Phase B: relay EP return =============== */
/* NCCL-free EP return: pure peer-WRITE copy kernels over NVLink with a
 * one-hop staging relay for the 12 SYS pairs (s597 Phase 1 relay table).
 * Topology fact (gpu-01, 8x V100 SXM2): GPUs {0,1,2,3} and {4,5,6,7} are
 * NVLink cliques and i <-> i^4 are NVLink partners; the only non-NVLink
 * (SYS) pairs are cross-half non-partners, and for each such directed pair
 * src->dst the partner GPU dst^4 is NVLink-adjacent to BOTH ends (it is in
 * src's half-clique and is dst's partner). Each GPU therefore relays
 * exactly 3 directed pairs - the balanced one-hop schedule from the s597
 * relay table, expressible without a table.
 * Ordering: stage W (src streams: local read -> remote write; direct for
 * NVLink dsts, staging on the relay for SYS dsts), 8x8 event barrier,
 * stage F (relay streams: local staging read -> NVLink write to dst),
 * 8x8 event barrier, then compose consumes d_ep_remote as usual. All
 * copies are byte moves - bit-exact by construction - and the event order
 * is fixed in the captured graph. */
static inline bool s601_nv_adjacent(int a, int b) {
    return ((a >> 2) == (b >> 2)) || (b == (a ^ 4));
}

int ep_return_relay_graph(const Options &opt,
                          RankState ranks[kGpus],
                          bool skip_self_copy,
                          uint64_t src_stride_elems,
                          const uint64_t copy_elems_by_src[kGpus]) {
    if (!copy_elems_by_src || src_stride_elems == 0) return 1;
    const int block = 256;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int rank = 0; rank < kGpus; ++rank) {
        if (!ranks[rank].d_ep_relay_stage || !ranks[rank].d_ep_contrib_all) {
            return 2;
        }
        for (int src = 0; src < kGpus; ++src) {
            if (!ranks[rank].d_ep_remote[src]) return 2;
        }
    }
    /* Stage W: per-src peer writes (reads local, writes remote/posted). */
    for (int src = 0; src < kGpus; ++src) {
        RankState &s = ranks[src];
        const uint64_t copy_elems = copy_elems_by_src[src];
        if (copy_elems == 0) continue;
        if (copy_elems > src_stride_elems) return 3;
        CHECK_CUDA(cudaSetDevice(s.device));
        for (int dst = 0; dst < kGpus; ++dst) {
            if (src == dst) continue;
            const float *src_ptr =
                s.d_ep_contrib_all + (uint64_t)dst * src_stride_elems;
            if (s601_nv_adjacent(src, dst)) {
                enqueue_graph_f32_copy_between_devices(
                    opt, ranks[dst].device, s.device,
                    ranks[dst].d_ep_remote[src], src_ptr, copy_elems,
                    s.stream, block);
            } else {
                const int relay = dst ^ 4;
                float *stage = ranks[relay].d_ep_relay_stage +
                               (uint64_t)src * src_stride_elems;
                enqueue_graph_f32_copy_between_devices(
                    opt, ranks[relay].device, s.device, stage, src_ptr,
                    copy_elems, s.stream, block);
            }
        }
        if (!skip_self_copy) {
            const float *src_ptr =
                s.d_ep_contrib_all + (uint64_t)src * src_stride_elems;
            enqueue_graph_f32_copy_between_devices(
                opt, s.device, s.device, s.d_ep_remote[src], src_ptr,
                copy_elems, s.stream, block);
        }
    }
    /* Relays must observe the staged slices (and dsts their direct
     * slices) before forwarding/compose: fixed-slot 8x8 event barrier. */
    if (enqueue_cross_gpu_stream_barrier(ranks, false) != 0) return 4;
    /* Stage F: each GPU forwards the 3 staged SYS slices to its partner
     * dst = relay^4 (local staging read, NVLink peer write). */
    for (int relay = 0; relay < kGpus; ++relay) {
        RankState &r = ranks[relay];
        const int dst = relay ^ 4;
        CHECK_CUDA(cudaSetDevice(r.device));
        for (int src = 0; src < kGpus; ++src) {
            if (src == dst || s601_nv_adjacent(src, dst)) continue;
            const uint64_t copy_elems = copy_elems_by_src[src];
            if (copy_elems == 0) continue;
            const float *stage =
                r.d_ep_relay_stage + (uint64_t)src * src_stride_elems;
            enqueue_graph_f32_copy_between_devices(
                opt, ranks[dst].device, r.device,
                ranks[dst].d_ep_remote[src], stage, copy_elems, r.stream,
                block);
        }
    }
    /* Dst rank streams wait for the relay forwards before compose. */
    if (enqueue_cross_gpu_stream_barrier(ranks, false) != 0) return 5;
    CHECK_CUDA(cudaSetDevice(prior_device));
    return 0;
}
/* ======================================================================== */

/* ============== Sprint 602: NCCL-free hc-class collectives ============== */
/* Replaces the remaining captured NCCL collectives (the s601-localized
 * racing set) with peer-write/kernel-reduction equivalents built on the
 * s601 relay topology (dst^4 one-hop staging for SYS pairs; fixed 8x8
 * event-barrier order; graph-capturable; no SYS traffic).
 *
 *   - allgathers + the root-0 broadcast are pure byte moves (bit-exact by
 *     construction);
 *   - the sum allreduces are ring-order-exact kernel folds reproducing
 *     NCCL's ring reduce-scatter accumulation order (chunk schedule
 *     calibrated by tools/s602-fold-probe; per-element fold = left fold
 *     along the ring starting at chunk+delta), so the s597 control anchor
 *     stays bit-valid; max allreduces are order-free bitwise.
 *
 * Everything is enqueued on the existing rank streams between two full
 * cross-GPU barriers per collective site (B0: inputs visible to relays
 * and cross-rank fold reads; B1: staged forwards visible to folds /
 * relay-written outputs visible to consumers). The promoted path
 * (DS4_V100_TP_EP_HC_TRANSPORT unset/nccl) allocates nothing and enqueues
 * nothing - byte-identical. */

enum S602Cls {
    kS602HcMax = 0, kS602HcMix, kS602HcSumsq, kS602HcAg, kS602FfnBcast,
    kS602RMax, kS602RSumsq, kS602RLogits, kS602PostAg, kS602ClsCount
};
static const char *const kS602ClsNames[kS602ClsCount] = {
    "hc_max", "hc_mix", "hc_sumsq", "hc_ag", "ffn_bcast",
    "r_max", "r_sumsq", "r_logits", "post_ag"};
/* DS4_V100_TP_EP_S602_KERNEL_MASK bit per class (hc max+mix share bit 0). */
static const int kS602ClsMaskBit[kS602ClsCount] = {0, 0, 1, 2, 3, 4, 5, 6, 7};
/* Allreduce stage layout: per-class element count = mult * slots. */
static const int kS602ClsSlotMult[kS602ClsCount] = {1, kHcMix, 1, 0, 0,
                                                    1, 1, kGlobalExperts, 0};

struct S602State {
    bool initialized = false;
    bool any = false;
    bool verify = false;
    int nrings = 0;
    int ring[kGpus][kGpus];
    int delta = 1;
    int min_chunk = 512;
    int nchannels = 1;
    uint64_t stage_slot_stride = 0;        /* floats per src slot */
    uint64_t cls_off[kS602ClsCount] = {};  /* float offset within a slot */
    /* Sprint 603 sync modes per point: 0 join, 1 peers, 2 mirror, 3 none.
     * join defaults (byte-identical to s602): E0/E1 join, AG/BC-E1 join,
     * E2 none (the next site's E0 join covers the exit closure). */
    int sync_e0 = 0;
    int sync_e1 = 0;       /* AR sites (pre-fold) */
    int sync_e1_exit = 0;  /* AG/BC sites (E1 doubles as the site exit) */
    int sync_e2 = 3;       /* AR site exit (post-fold) */
    /* Sprint 603 Phase D: 0 off, 1 bcast site, 2 all sites. */
    int dense_guard = 0;
};
static S602State g_s602x;

static inline bool s602_cls_masked(const Options &opt, int cls) {
    return (opt.s602_kernel_mask >> kS602ClsMaskBit[cls]) & 1u;
}
/* kernel transport feeds the consumers (NCCL skipped) */
static inline bool s602_use_kernel(const Options &opt, int cls) {
    return opt.hc_transport_kernel && !opt.s602_verify &&
           s602_cls_masked(opt, cls);
}
/* bring-up verifier: NCCL still feeds the consumers; the kernel transport
 * runs on shadow inputs and is bit-compared in-graph (s600 pattern) */
static inline bool s602_use_verify(const Options &opt, int cls) {
    return opt.hc_transport_kernel && opt.s602_verify &&
           s602_cls_masked(opt, cls);
}
static inline bool s602_any_active(const Options &opt) {
    return opt.hc_transport_kernel && opt.s602_kernel_mask != 0u;
}

static int s602_parse_rings(const char *spec, int ring[kGpus][kGpus]) {
    int nrings = 0;
    const char *p = spec;
    int cur[kGpus];
    int n = 0;
    while (p && nrings < kGpus) {
        if (*p == ';' || *p == '\0') {
            if (n == kGpus) {
                bool seen[kGpus] = {};
                bool ok = true;
                for (int i = 0; i < kGpus; ++i) {
                    if (cur[i] < 0 || cur[i] >= kGpus || seen[cur[i]]) {
                        ok = false;
                        break;
                    }
                    seen[cur[i]] = true;
                }
                if (!ok) return 0;
                for (int i = 0; i < kGpus; ++i) ring[nrings][i] = cur[i];
                ++nrings;
            } else if (n != 0) {
                return 0;
            }
            n = 0;
            if (*p == '\0') break;
            ++p;
            continue;
        }
        if (*p == ' ' || *p == '\t' || *p == ',') {
            ++p;
            continue;
        }
        char *end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || n >= kGpus) return 0;
        cur[n++] = (int)v;
        p = end;
    }
    return nrings;
}

/* Sprint 603: parse one sync-point override (see runtime_options.cuh). */
static int s602_parse_sync_point(const char *v, int dflt) {
    if (!v || !*v) return dflt;
    if (std::strcmp(v, "join") == 0) return 0;
    if (std::strcmp(v, "peers") == 0) return 1;
    if (std::strcmp(v, "mirror") == 0) return 2;
    if (std::strcmp(v, "none") == 0) return 3;
    return -1;
}
static const char *const kS602SyncNames[4] = {"join", "peers", "mirror",
                                              "none"};

int s602_state_init(const Options &opt, RankState ranks[kGpus]) {
    if (g_s602x.initialized) return 0;
    g_s602x.initialized = true;
    if (!s602_any_active(opt)) return 0;
    g_s602x.any = true;
    g_s602x.verify = opt.s602_verify;
    g_s602x.delta = opt.s602_fold_delta & 7;
    /* 0 = auto size rule (fold-probe run3): see s602_min_chunk_for(). */
    g_s602x.min_chunk = opt.s602_min_chunk > 0 ? opt.s602_min_chunk : 0;
    g_s602x.nchannels =
        (opt.s602_nchannels > 0 && opt.s602_nchannels <= kGpus)
            ? opt.s602_nchannels
            : 1;
    if (opt.s602_sync_edges) {
        /* edges defaults = the SPRINT-603 Phase A table; per-point env
         * overrides for bisection. */
        g_s602x.sync_e0 = s602_parse_sync_point(opt.s602_sync_e0, 1);
        g_s602x.sync_e1 = s602_parse_sync_point(opt.s602_sync_e1, 2);
        g_s602x.sync_e1_exit = s602_parse_sync_point(opt.s602_sync_e1, 1);
        g_s602x.sync_e2 = s602_parse_sync_point(opt.s602_sync_e2, 1);
        if (g_s602x.sync_e0 < 0 || g_s602x.sync_e0 > 1 ||
            g_s602x.sync_e1 < 0 || g_s602x.sync_e1 > 2 ||
            g_s602x.sync_e2 < 0 || g_s602x.sync_e2 == 2) {
            std::fprintf(stderr,
                         "tp_ep_s602_init bad sync override e0='%s' e1='%s' "
                         "e2='%s'\n",
                         opt.s602_sync_e0, opt.s602_sync_e1, opt.s602_sync_e2);
            return 1;
        }
    }
    g_s602x.dense_guard = opt.s602_dense_guard;
    g_s602x.nrings = s602_parse_rings(opt.s602_ring_spec, g_s602x.ring);
    if (g_s602x.nrings <= 0) {
        std::fprintf(stderr, "tp_ep_s602_init bad ring spec '%s'\n",
                     opt.s602_ring_spec);
        return 1;
    }
    uint64_t off = 0;
    for (int c = 0; c < kS602ClsCount; ++c) {
        g_s602x.cls_off[c] = off;
        off += (uint64_t)kS602ClsSlotMult[c] * (uint64_t)opt.slots;
    }
    g_s602x.stage_slot_stride = off;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    const size_t stage_bytes =
        (size_t)kGpus * off * sizeof(float);
    const size_t slots_b = (size_t)opt.slots * sizeof(float);
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaMalloc(&r.d_s602_stage, stage_bytes));
        CHECK_CUDA(cudaMalloc(&r.d_s602_out_max, slots_b));
        CHECK_CUDA(cudaMalloc(&r.d_s602_out_mix, slots_b * kHcMix));
        CHECK_CUDA(cudaMalloc(&r.d_s602_out_sumsq, slots_b));
        CHECK_CUDA(cudaMalloc(&r.d_s602_out_rmax, slots_b));
        CHECK_CUDA(cudaMalloc(&r.d_s602_out_rsumsq, slots_b));
        CHECK_CUDA(cudaMalloc(&r.d_s602_out_logits, slots_b * kGlobalExperts));
        if (g_s602x.verify) {
            CHECK_CUDA(cudaMalloc(&r.d_s602_ver_in, off * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&r.d_s602_ver_out,
                                  3ull * (size_t)opt.slots * kHidden *
                                      sizeof(float)));
            CHECK_CUDA(cudaMalloc(&r.d_s602_mismatch,
                                  (size_t)kS602ClsCount * 4 *
                                      sizeof(unsigned long long)));
            CHECK_CUDA(cudaMemset(r.d_s602_mismatch, 0,
                                  (size_t)kS602ClsCount * 4 *
                                      sizeof(unsigned long long)));
        }
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
    std::printf("tp_ep_s602_init\ttransport\tkernel\tmask\t0x%02x\tverify\t%d\t"
                "rings\t%d\tdelta\t%d\tmin_chunk\t%d\tnchannels\t%d\t"
                "stage_kib\t%llu\tsync\t%s\te0\t%s\te1\t%s/%s\te2\t%s\t"
                "dense_guard\t%d\tPASS\n",
                opt.s602_kernel_mask, g_s602x.verify ? 1 : 0, g_s602x.nrings,
                g_s602x.delta, g_s602x.min_chunk, g_s602x.nchannels,
                (unsigned long long)(stage_bytes / 1024ull),
                opt.s602_sync_edges ? "edges" : "join",
                kS602SyncNames[g_s602x.sync_e0],
                kS602SyncNames[g_s602x.sync_e1],
                kS602SyncNames[g_s602x.sync_e1_exit],
                kS602SyncNames[g_s602x.sync_e2], g_s602x.dense_guard);
    std::fflush(stdout);
    return 0;
}

/* ---- device kernels ---- */
struct S602Ptrs8 {
    const float *p[kGpus];
};
constexpr int kS602MaxSegs = 40;
struct S602FoldPlan {
    int nsegs;
    unsigned long long seg_begin[kS602MaxSegs];
    int seg_start[kS602MaxSegs];
    int seg_ring[kS602MaxSegs];
};
struct S602Rings {
    int r[kGpus][kGpus];
};

/* Ring-order-exact fold: out[e] = left-fold of in[ring[(start+k)&7]][e].
 * Plain float adds (no FMA contraction possible: adds only), fmaxf for the
 * max reductions (order-free bitwise on finite data). */
__global__ void s602_fold_kernel(S602Ptrs8 in, float *out,
                                 unsigned long long count, int is_max,
                                 S602FoldPlan plan, S602Rings rings) {
    for (unsigned long long i =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (unsigned long long)blockDim.x * gridDim.x) {
        int s = 0;
        while (s + 1 < plan.nsegs && i >= plan.seg_begin[s + 1]) ++s;
        const int *ring = rings.r[plan.seg_ring[s]];
        const int start = plan.seg_start[s];
        float acc = in.p[ring[start]][i];
#pragma unroll
        for (int k = 1; k < kGpus; ++k) {
            const float v = in.p[ring[(start + k) & 7]][i];
            acc = is_max ? fmaxf(acc, v) : (acc + v);
        }
        out[i] = acc;
    }
}

/* Three independent same-length copies in one launch (the per-relay
 * forward: each relay GPU serves exactly its 3 quad-mates' SYS slices). */
__global__ void s602_copy3_kernel(const float *s0, const float *s1,
                                  const float *s2, float *d0, float *d1,
                                  float *d2, unsigned long long elems) {
    const unsigned long long total = 3ull * elems;
    for (unsigned long long i =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (unsigned long long)blockDim.x * gridDim.x) {
        const int which = (int)(i / elems);
        const unsigned long long e = i % elems;
        const float v = which == 0 ? s0[e] : which == 1 ? s1[e] : s2[e];
        float *d = which == 0 ? d0 : which == 1 ? d1 : d2;
        d[e] = v;
    }
}

/* Rank-major gather of the non-null sources (self + the 4 NVLink peers;
 * the 3 SYS sources are relay-written into the same output). */
__global__ void s602_gather8_kernel(S602Ptrs8 in, float *out,
                                    unsigned long long shard_elems) {
    const unsigned long long total = (unsigned long long)kGpus * shard_elems;
    for (unsigned long long i =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (unsigned long long)blockDim.x * gridDim.x) {
        const int src = (int)(i / shard_elems);
        if (in.p[src]) out[i] = in.p[src][i - (unsigned long long)src * shard_elems];
    }
}

/* In-graph bit comparator -> per-class mismatch counters (s600 pattern). */
__global__ void s602_bitcmp_kernel(const float *a, const float *b,
                                   unsigned long long elems,
                                   unsigned long long *out) {
    for (unsigned long long i =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < elems; i += (unsigned long long)blockDim.x * gridDim.x) {
        const unsigned int ab = __float_as_uint(a[i]);
        const unsigned int bb = __float_as_uint(b[i]);
        if (ab != bb) {
            if (atomicAdd(out, 1ull) == 0ull) {
                out[1] = i;
                out[2] = (unsigned long long)ab |
                         ((unsigned long long)bb << 32);
            }
        }
    }
}

/* NCCL 2.19 LL minChunk by op size (fold-probe run3 calibration):
 * nthreads steps with size as pow2ceil(bytes/64) clamped to [96, 512];
 * minChunk = nthreads * 2 floats. Verified bit-exact on the probe for
 * every engine shape (hc_mix 384/576/768 -> 192; r_logits 256..1024 ->
 * 192, 2048 -> 256, 4096 -> 512, 6144/8192 -> 1024; sumsq <= 32 single
 * chunk under any mc >= count). */
static uint64_t s602_min_chunk_for(uint64_t count) {
    uint64_t nt = (count + 15) / 16;
    uint64_t p = 1;
    while (p < nt) p <<= 1;
    if (p < 96) p = 96;
    if (p > 512) p = 512;
    return 2 * p;
}

/* ---- host-side fold plan (mirrors the NCCL 2.19 ring AR chunk loop with
 * the calibrated parameters) ---- */
static void s602_build_plan(uint64_t count, S602FoldPlan *plan) {
    plan->nsegs = 0;
    const uint64_t loop = (uint64_t)g_s602x.nchannels * kGpus;
    const uint64_t mc = g_s602x.min_chunk > 0 ? (uint64_t)g_s602x.min_chunk
                                              : s602_min_chunk_for(count);
    uint64_t gridOffset = 0;
    while (gridOffset < count) {
        const uint64_t remaining = count - gridOffset;
        uint64_t rcs = ((remaining + loop * mc - 1) / (loop * mc)) * mc;
        if (rcs > (1ull << 20)) rcs = 1ull << 20;
        for (int bid = 0; bid < g_s602x.nchannels; ++bid) {
            for (int c = 0; c < kGpus; ++c) {
                const uint64_t begin =
                    gridOffset + ((uint64_t)bid * kGpus + (uint64_t)c) * rcs;
                if (begin >= count) break;
                const int start = (c + g_s602x.delta) & 7;
                const int ringsel = bid % g_s602x.nrings;
                if (plan->nsegs > 0 &&
                    plan->seg_start[plan->nsegs - 1] == start &&
                    plan->seg_ring[plan->nsegs - 1] == ringsel) {
                    continue; /* merge */
                }
                if (plan->nsegs >= kS602MaxSegs) {
                    /* should not happen at the engine shapes; extend the
                     * last segment rather than overflow (loud once). */
                    static bool warned = false;
                    if (!warned) {
                        std::fprintf(stderr,
                                     "tp_ep_s602_fold_plan overflow count=%llu\n",
                                     (unsigned long long)count);
                        warned = true;
                    }
                    break;
                }
                plan->seg_begin[plan->nsegs] = begin;
                plan->seg_start[plan->nsegs] = start;
                plan->seg_ring[plan->nsegs] = ringsel;
                ++plan->nsegs;
            }
        }
        gridOffset += loop * rcs;
    }
    if (plan->nsegs == 0) {
        plan->nsegs = 1;
        plan->seg_begin[0] = 0;
        plan->seg_start[0] = g_s602x.delta & 7;
        plan->seg_ring[0] = 0;
    }
}

static inline float *s602_stage_ptr(RankState &dst_rank, int src, int cls) {
    return dst_rank.d_s602_stage +
           (uint64_t)src * g_s602x.stage_slot_stride + g_s602x.cls_off[cls];
}

/* ---- site synchronization ----
 * Full 8x8 join across the RANK STREAMS ONLY at both sync points of every
 * site. Both alternatives were falsified empirically:
 *   - The s601 full barrier additionally joins the DENSE streams; 16 of
 *     those per layer destroy the layer's rank<->dense overlap (replay
 *     10.9 ms/layer vs 4.13 control; b-sb-fb confirmed bit-stable but
 *     slow). The replaced NCCL collectives only ever ordered the rank
 *     streams, so the dense joins are pure over-synchronization.
 *   - A minimal pairwise dependency set (B0 = wait 4 NVLink peers, B1 =
 *     wait mirror g^4) restores full speed but is run-to-run
 *     NONDETERMINISTIC under the Simple-stress detector (agreement
 *     0.79-0.95, b-sb/b-sb-kc), while the full barrier is bit-identical
 *     (b-sb-fb 1.0/1.0): NCCL's completion semantics join all 8 rank
 *     streams at every collective and something downstream relies on it.
 * The all-rank rank-stream join keeps NCCL's contract without the
 * dense-stream cost. DS4_V100_TP_EP_S602_FULL_BARRIER=1 restores the
 * s601 rank+dense barrier at both points (rollback/diagnosis). */
static inline bool s602_full_barrier_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_S602_FULL_BARRIER");
    return v && *v && !(v[0] == '0' && v[1] == '\0');
}
int next_graph_order_event_slot(RankState ranks[kGpus]);
cudaEvent_t graph_stream_done_event(RankState &r, int slot);
cudaEvent_t graph_dense_done_event(RankState &r, int slot);
static int s602_rank_join(RankState ranks[kGpus]) {
    if (s602_full_barrier_env()) {
        return enqueue_cross_gpu_stream_barrier(ranks, false);
    }
    const int slot = next_graph_order_event_slot(ranks);
    for (int g = 0; g < kGpus; ++g) {
        RankState &r = ranks[g];
        CHECK_CUDA(cudaSetDevice(r.device));
        cudaEvent_t ev = graph_stream_done_event(r, slot);
        if (!ev) return 1;
        CHECK_CUDA(cudaEventRecord(ev, r.stream));
    }
    for (int g = 0; g < kGpus; ++g) {
        RankState &r = ranks[g];
        CHECK_CUDA(cudaSetDevice(r.device));
        for (int p = 0; p < kGpus; ++p) {
            if (p == g) continue;
            CHECK_CUDA(cudaStreamWaitEvent(
                r.stream, graph_stream_done_event(ranks[p], slot), 0));
        }
    }
    return 0;
}
/* Sprint 603 edge points: record one pre-allocated event slot on every
 * rank stream, then each rank waits only its mirror (g^4) or its 4 NVLink
 * peers (quad-mates + mirror) - the derived dependency sets of the
 * SPRINT-603 Phase A edge table. Fixed order, graph-capturable, rank
 * streams only. */
static int s602_edge_sync(RankState ranks[kGpus], bool mirror_only) {
    const int slot = next_graph_order_event_slot(ranks);
    for (int g = 0; g < kGpus; ++g) {
        RankState &r = ranks[g];
        CHECK_CUDA(cudaSetDevice(r.device));
        cudaEvent_t ev = graph_stream_done_event(r, slot);
        if (!ev) return 1;
        CHECK_CUDA(cudaEventRecord(ev, r.stream));
    }
    for (int g = 0; g < kGpus; ++g) {
        RankState &r = ranks[g];
        CHECK_CUDA(cudaSetDevice(r.device));
        if (mirror_only) {
            CHECK_CUDA(cudaStreamWaitEvent(
                r.stream, graph_stream_done_event(ranks[g ^ 4], slot), 0));
            continue;
        }
        for (int p = 0; p < kGpus; ++p) {
            if (p == g || !s601_nv_adjacent(g, p)) continue;
            CHECK_CUDA(cudaStreamWaitEvent(
                r.stream, graph_stream_done_event(ranks[p], slot), 0));
        }
    }
    return 0;
}

/* mode: 0 join, 1 peers, 2 mirror, 3 none. none enqueues nothing and
 * consumes no event slot (the default-join capture stays byte-identical).
 * DS4_V100_TP_EP_S602_FULL_BARRIER=1 still overrides every non-none point
 * to the s601 rank+dense barrier (s602_rank_join handles it for join). */
static int s602_sync_point(RankState ranks[kGpus], int mode) {
    if (mode == 3) return 0;
    if (mode == 0) return s602_rank_join(ranks);
    if (s602_full_barrier_env()) {
        return enqueue_cross_gpu_stream_barrier(ranks, false);
    }
    return s602_edge_sync(ranks, mode == 2);
}

/* Sprint 603 Phase D fix: dense-WAR guard (see runtime_options.cuh).
 * Records one pre-allocated dense event per GPU, then each rank stream
 * waits the dense events of the GPUs whose buffers this site writes from
 * that stream (own + mirror - the relay write target). Enqueued AFTER the
 * site's E0 rank sync, BEFORE the byte moves. */
static int s602_dense_war_guard(RankState ranks[kGpus]) {
    const int slot = next_graph_order_event_slot(ranks);
    for (int g = 0; g < kGpus; ++g) {
        RankState &r = ranks[g];
        CHECK_CUDA(cudaSetDevice(r.device));
        cudaEvent_t ev = graph_dense_done_event(r, slot);
        if (!ev) return 1;
        CHECK_CUDA(cudaEventRecord(ev,
                                   r.dense_stream ? r.dense_stream : r.stream));
    }
    for (int g = 0; g < kGpus; ++g) {
        RankState &r = ranks[g];
        CHECK_CUDA(cudaSetDevice(r.device));
        CHECK_CUDA(cudaStreamWaitEvent(
            r.stream, graph_dense_done_event(ranks[g], slot), 0));
        CHECK_CUDA(cudaStreamWaitEvent(
            r.stream, graph_dense_done_event(ranks[g ^ 4], slot), 0));
    }
    return 0;
}

static int s602_sync_inputs(RankState ranks[kGpus]) {
    return s602_sync_point(ranks, g_s602x.sync_e0);
}
/* AR sites: pre-fold point. */
static int s602_sync_relayed(RankState ranks[kGpus]) {
    return s602_sync_point(ranks, g_s602x.sync_e1);
}
/* AG/BC sites: the post-byte-move point doubles as the site exit. */
static int s602_sync_relayed_exit(RankState ranks[kGpus]) {
    return s602_sync_point(ranks, g_s602x.sync_e1_exit);
}
/* AR sites: post-fold exit closure (none in join mode). */
static int s602_sync_exit(RankState ranks[kGpus]) {
    return s602_sync_point(ranks, g_s602x.sync_e2);
}

/* ---- allreduce site ---- */
struct S602ArOp {
    int cls;
    float *in[kGpus];     /* live (NCCL in-place) partial buffers */
    float *out[kGpus];    /* fold destinations */
    uint64_t count;
    bool is_max;
    bool fold_all;        /* false: only rank 0 consumes the reduction */
    const float *src[kGpus]; /* filled by pre/post: fold/relay inputs */
};

/* Verify mode: shadow the inputs on each rank's own stream BEFORE the
 * caller's NCCL group overwrites them in place. */
int s602_allreduce_site_pre(const Options &opt, RankState ranks[kGpus],
                            S602ArOp *ops, int nops) {
    const int block = 256;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int o = 0; o < nops; ++o) {
        S602ArOp &op = ops[o];
        const bool ver = s602_use_verify(opt, op.cls);
        const bool ker = s602_use_kernel(opt, op.cls);
        if (!ver && !ker) continue;
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            if (!r.d_s602_stage || !op.in[rank] || !op.out[rank]) return 1;
            if (ver) {
                if (!r.d_s602_ver_in) return 1;
                float *shadow = r.d_s602_ver_in + g_s602x.cls_off[op.cls];
                CHECK_CUDA(cudaSetDevice(r.device));
                copy_f32_kernel<<<
                    (unsigned int)((op.count + (uint64_t)block - 1) /
                                   (uint64_t)block),
                    block, 0, r.stream>>>(shadow, op.in[rank], op.count);
                CHECK_CUDA(cudaGetLastError());
                op.src[rank] = shadow;
            } else {
                op.src[rank] = op.in[rank];
            }
        }
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
    return 0;
}

/* Transport + fold (+ in-graph compare in verify mode). Shares the two
 * cross-GPU barriers across every op of the site. */
int s602_allreduce_site_post(const Options &opt, RankState ranks[kGpus],
                             S602ArOp *ops, int nops) {
    const int block = 256;
    bool any = false;
    for (int o = 0; o < nops; ++o) {
        if (s602_use_kernel(opt, ops[o].cls) ||
            s602_use_verify(opt, ops[o].cls)) {
            any = true;
        }
    }
    if (!any) return 0;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    /* B0: every rank's partial (and shadow) visible to relays + folds. */
    if (s602_sync_inputs(ranks) != 0) return 2;
    if (g_s602x.dense_guard >= 2 && s602_dense_war_guard(ranks) != 0) {
        return 2;
    }
    /* relay wave: for each folding dst, relay dst^4 forwards the 3 SYS
     * srcs' inputs (its own quad-mates - all NVLink) into dst's stage. */
    for (int o = 0; o < nops; ++o) {
        S602ArOp &op = ops[o];
        if (!s602_use_kernel(opt, op.cls) && !s602_use_verify(opt, op.cls)) {
            continue;
        }
        for (int dst = 0; dst < kGpus; ++dst) {
            if (!op.fold_all && dst != 0) continue;
            const int relay = dst ^ 4;
            RankState &rr = ranks[relay];
            const float *s[3] = {};
            float *d[3] = {};
            int n = 0;
            for (int src = 0; src < kGpus; ++src) {
                if (src == dst || s601_nv_adjacent(src, dst)) continue;
                if (n >= 3) return 3;
                s[n] = op.src[src];
                d[n] = s602_stage_ptr(ranks[dst], src, op.cls);
                ++n;
            }
            if (n != 3) return 3;
            CHECK_CUDA(cudaSetDevice(rr.device));
            s602_copy3_kernel<<<
                (unsigned int)((3ull * op.count + (uint64_t)block - 1) /
                               (uint64_t)block),
                block, 0, rr.stream>>>(s[0], s[1], s[2], d[0], d[1], d[2],
                                       op.count);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    /* B1: staged forwards visible to the folds. */
    if (s602_sync_relayed(ranks) != 0) return 4;
    S602Rings rings;
    std::memcpy(rings.r, g_s602x.ring, sizeof(rings.r));
    for (int o = 0; o < nops; ++o) {
        S602ArOp &op = ops[o];
        const bool ver = s602_use_verify(opt, op.cls);
        const bool ker = s602_use_kernel(opt, op.cls);
        if (!ver && !ker) continue;
        S602FoldPlan plan;
        s602_build_plan(op.count, &plan);
        for (int dst = 0; dst < kGpus; ++dst) {
            if (!op.fold_all && dst != 0) continue;
            RankState &r = ranks[dst];
            S602Ptrs8 in;
            for (int src = 0; src < kGpus; ++src) {
                in.p[src] = (src == dst || s601_nv_adjacent(src, dst))
                                ? op.src[src]
                                : s602_stage_ptr(r, src, op.cls);
            }
            CHECK_CUDA(cudaSetDevice(r.device));
            s602_fold_kernel<<<
                (unsigned int)((op.count + (uint64_t)block - 1) /
                               (uint64_t)block),
                block, 0, r.stream>>>(in, op.out[dst], op.count,
                                      op.is_max ? 1 : 0, plan, rings);
            CHECK_CUDA(cudaGetLastError());
            if (ver) {
                /* compare the fold against NCCL's in-place result (both
                 * resident on this rank, both produced on this stream). */
                s602_bitcmp_kernel<<<
                    (unsigned int)((op.count + (uint64_t)block - 1) /
                                   (uint64_t)block),
                    block, 0, r.stream>>>(op.out[dst], op.in[dst], op.count,
                                          r.d_s602_mismatch +
                                              (size_t)op.cls * 4);
                CHECK_CUDA(cudaGetLastError());
            }
        }
    }
    /* E2: exit closure - each rank's in-place partials are remote-read by
     * its NV peers' folds above; the site must not complete on a rank
     * before those reads do (NCCL's buffer-free contract). none in join
     * mode (the next site's E0 join covers it transitively). */
    if (s602_sync_exit(ranks) != 0) return 5;
    CHECK_CUDA(cudaSetDevice(prior_device));
    return 0;
}

/* ---- allgather site (byte moves; rank-major output) ----
 * kernel mode: writes out_by_rank directly (caller skips NCCL).
 * verify mode: caller ran NCCL; the kernel gather writes the shadow region
 * and is bit-compared against NCCL's output on every rank. */
int s602_allgather_site(const Options &opt, RankState ranks[kGpus], int cls,
                        float *const shard_by_rank[kGpus],
                        float *const out_by_rank[kGpus], uint64_t shard_elems,
                        int ver_region) {
    const bool ker = s602_use_kernel(opt, cls);
    const bool ver = s602_use_verify(opt, cls);
    if (!ker && !ver) return 0;
    const int block = 256;
    const uint64_t full_elems = (uint64_t)kGpus * shard_elems;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    float *dst_by_rank[kGpus] = {};
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!shard_by_rank[rank] || !out_by_rank[rank]) return 1;
        if (ver) {
            if (!r.d_s602_ver_out) return 1;
            dst_by_rank[rank] = r.d_s602_ver_out +
                                (uint64_t)ver_region * (uint64_t)opt.slots *
                                    kHidden;
        } else {
            dst_by_rank[rank] = out_by_rank[rank];
        }
    }
    /* B0: every rank's shard visible to peer pulls + relay forwards. */
    if (s602_sync_inputs(ranks) != 0) return 2;
    if (g_s602x.dense_guard >= 2 && s602_dense_war_guard(ranks) != 0) {
        return 2;
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        RankState &r = ranks[dst];
        S602Ptrs8 in;
        for (int src = 0; src < kGpus; ++src) {
            in.p[src] = (src == dst || s601_nv_adjacent(src, dst))
                            ? shard_by_rank[src]
                            : nullptr;
        }
        CHECK_CUDA(cudaSetDevice(r.device));
        s602_gather8_kernel<<<
            (unsigned int)((full_elems + (uint64_t)block - 1) /
                           (uint64_t)block),
            block, 0, r.stream>>>(in, dst_by_rank[dst], shard_elems);
        CHECK_CUDA(cudaGetLastError());
    }
    for (int relay = 0; relay < kGpus; ++relay) {
        RankState &rr = ranks[relay];
        const int dst = relay ^ 4;
        const float *s[3] = {};
        float *d[3] = {};
        int n = 0;
        for (int src = 0; src < kGpus; ++src) {
            if (src == dst || s601_nv_adjacent(src, dst)) continue;
            if (n >= 3) return 3;
            s[n] = shard_by_rank[src];
            d[n] = dst_by_rank[dst] + (uint64_t)src * shard_elems;
            ++n;
        }
        if (n != 3) return 3;
        CHECK_CUDA(cudaSetDevice(rr.device));
        s602_copy3_kernel<<<
            (unsigned int)((3ull * shard_elems + (uint64_t)block - 1) /
                           (uint64_t)block),
            block, 0, rr.stream>>>(s[0], s[1], s[2], d[0], d[1], d[2],
                                   shard_elems);
        CHECK_CUDA(cudaGetLastError());
    }
    /* B1: relay-written segments visible to the consumers (or comparator);
     * doubles as the site exit (all shard reads are pre-B1). */
    if (s602_sync_relayed_exit(ranks) != 0) return 4;
    if (ver) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            s602_bitcmp_kernel<<<
                (unsigned int)((full_elems + (uint64_t)block - 1) /
                               (uint64_t)block),
                block, 0, r.stream>>>(dst_by_rank[rank], out_by_rank[rank],
                                      full_elems,
                                      r.d_s602_mismatch + (size_t)cls * 4);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
    return 0;
}

/* ---- root-0 broadcast site (byte moves) ----
 * NVLink dsts 1..4 pull from the root; dsts 5,6,7 are written by relays
 * 1,2,3 (= dst^4, NVLink-adjacent to both ends); rank 0 takes a local
 * copy (the NCCL call it replaces also writes the root's recv buffer). */
int s602_broadcast0_site(const Options &opt, RankState ranks[kGpus], int cls,
                         const float *src_device0,
                         float *const out_by_rank[kGpus], uint64_t elems,
                         int ver_region) {
    const bool ker = s602_use_kernel(opt, cls);
    const bool ver = s602_use_verify(opt, cls);
    if (!ker && !ver) return 0;
    const int block = 256;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    float *dst_by_rank[kGpus] = {};
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!out_by_rank[rank]) return 1;
        if (ver) {
            if (!r.d_s602_ver_out) return 1;
            dst_by_rank[rank] = r.d_s602_ver_out +
                                (uint64_t)ver_region * (uint64_t)opt.slots *
                                    kHidden;
        } else {
            dst_by_rank[rank] = out_by_rank[rank];
        }
    }
    /* B0: root data visible to all readers; WAR on the dst buffers. */
    if (s602_sync_inputs(ranks) != 0) return 2;
    /* Dense-WAR: the previous d_current_full value is consumed on the
     * destinations' dense streams; order those reads before the writes. */
    if (g_s602x.dense_guard >= 1 && s602_dense_war_guard(ranks) != 0) {
        return 2;
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        if (!s601_nv_adjacent(0, dst) && dst != 0) continue; /* 0..4 */
        RankState &r = ranks[dst];
        CHECK_CUDA(cudaSetDevice(r.device));
        copy_f32_kernel<<<
            (unsigned int)((elems + (uint64_t)block - 1) / (uint64_t)block),
            block, 0, r.stream>>>(dst_by_rank[dst], src_device0, elems);
        CHECK_CUDA(cudaGetLastError());
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        if (dst == 0 || s601_nv_adjacent(0, dst)) continue; /* 5,6,7 */
        const int relay = dst ^ 4; /* 1,2,3 */
        RankState &rr = ranks[relay];
        CHECK_CUDA(cudaSetDevice(rr.device));
        copy_f32_kernel<<<
            (unsigned int)((elems + (uint64_t)block - 1) / (uint64_t)block),
            block, 0, rr.stream>>>(dst_by_rank[dst], src_device0, elems);
        CHECK_CUDA(cudaGetLastError());
    }
    /* B1: relay-written dsts visible to their consumers; doubles as the
     * site exit (all src reads are pre-B1). */
    if (s602_sync_relayed_exit(ranks) != 0) return 3;
    if (ver) {
        for (int rank = 0; rank < kGpus; ++rank) {
            RankState &r = ranks[rank];
            CHECK_CUDA(cudaSetDevice(r.device));
            s602_bitcmp_kernel<<<
                (unsigned int)((elems + (uint64_t)block - 1) /
                               (uint64_t)block),
                block, 0, r.stream>>>(dst_by_rank[rank], out_by_rank[rank],
                                      elems,
                                      r.d_s602_mismatch + (size_t)cls * 4);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
    return 0;
}

/* Host-side, after a fully synced replay (next to s600_collect_verify). */
void s602_collect_verify(int layer, int step, RankState ranks[kGpus]) {
    if (!g_s602x.any || !g_s602x.verify) return;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.d_s602_mismatch) continue;
        CHECK_CUDA(cudaSetDevice(r.device));
        unsigned long long h[kS602ClsCount * 4] = {};
        CHECK_CUDA(cudaMemcpy(h, r.d_s602_mismatch, sizeof(h),
                              cudaMemcpyDeviceToHost));
        bool any = false;
        for (int cls = 0; cls < kS602ClsCount; ++cls) {
            if (!h[cls * 4]) continue;
            any = true;
            std::printf("tp_ep_s602_verify_mismatch\tstep\t%d\tlayer\t%d\t"
                        "rank\t%d\tclass\t%s\tcount\t%llu\telem\t%llu\t"
                        "kernel_bits\t%08llx\tnccl_bits\t%08llx\n",
                        step, layer, rank, kS602ClsNames[cls], h[cls * 4],
                        h[cls * 4 + 1], h[cls * 4 + 2] & 0xffffffffull,
                        h[cls * 4 + 2] >> 32);
        }
        if (any) {
            std::fflush(stdout);
            CHECK_CUDA(cudaMemset(r.d_s602_mismatch, 0, sizeof(h)));
        }
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
}
/* ======================================================================== */

int parse_contract(const char *path, int layer, std::vector<ContractRow> *rows,
                   LayerStats *stats) {
    FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "cannot open contract %s: %s\n", path, std::strerror(errno));
        return 1;
    }
    char buf[8192];
    bool first = true;
    while (std::fgets(buf, sizeof(buf), fp)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (first) {
            first = false;
            continue;
        }
        if (line.empty()) continue;
        std::vector<std::string> f = split_tabs(line);
        ContractRow r;
        if (!parse_contract_row(f, &r)) {
            stats->bad_rows++;
            continue;
        }
        if (layer >= 0 && r.layer != layer) continue;
        if (r.owning_gpu < 0 || r.owning_gpu >= kGpus) {
            stats->bad_rows++;
            continue;
        }
        rows->push_back(r);
        stats->total_rows++;
        GpuFamilyStats &g = stats->gpu[r.owning_gpu];
        if (r.record_type == "dense_tp") {
            stats->dense_rows++;
            g.dense_rows++;
            g.dense_bytes += r.bytes_estimate;
        } else if (r.record_type == "replicated_control") {
            stats->control_rows++;
            g.control_rows++;
            g.control_bytes += r.bytes_estimate;
        } else if (r.record_type == "ep_expert") {
            stats->expert_rows++;
            g.expert_rows++;
            g.expert_descriptor_bytes += r.bytes_estimate;
        } else if (r.record_type == "kv_shard") {
            stats->kv_rows++;
            g.kv_rows++;
        } else if (r.record_type == "kv_comp_state") {
            stats->comp_rows++;
            g.comp_rows++;
        }
    }
    std::fclose(fp);
    return rows->empty() ? 2 : 0;
}

uint64_t physical_row_offset(const ContractRow &r) {
    if (r.record_type == "dense_tp" && r.shard_index >= 0 && r.shard_count > 1 &&
        r.source_byte_length >= r.bytes_estimate * (uint64_t)r.shard_count) {
        return r.source_shard_offset + (uint64_t)r.shard_index * r.bytes_estimate;
    }
    return r.source_shard_offset;
}

bool parse_shape2(const std::string &shape, int *cols, int *rows) {
    if (shape.size() < 5 || shape.front() != '[' || shape.back() != ']') return false;
    const size_t x = shape.find('x');
    if (x == std::string::npos) return false;
    std::string a = shape.substr(1, x - 1);
    std::string b = shape.substr(x + 1, shape.size() - x - 2);
    return parse_int(a.c_str(), cols) && parse_int(b.c_str(), rows) &&
           *cols > 0 && *rows > 0;
}

std::string layer_tensor_name(int layer, const char *suffix) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "blk.%d.%s", layer, suffix);
    return std::string(buf);
}

int ds4_layer_ratio(int layer) {
    if (layer < 2) return 0;
    if (layer == 43) return 0;  /* MTP block: simple attention, no compress/indexer */
    return (layer % 2) == 0 ? 4 : 128;
}

int attn_comp_state_rows_for_ratio(int ratio) {
    if (ratio == 4) return 2 * ratio;
    return ratio > 0 ? ratio : 0;
}

int attn_comp_state_width_for_ratio(int ratio) {
    if (ratio == 4) return 2 * kHeadDim;
    return ratio > 0 ? kHeadDim : 0;
}

uint64_t f8_row_bytes(int cols) {
    return (uint64_t)(cols / 128) * 129ull;
}

float e8m0_to_f32_host(uint8_t e) {
    uint32_t bits = e == 0 ? 0x00400000u : ((uint32_t)e << 23);
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

float e4m3fn_to_f32_host(uint8_t x) {
    const uint8_t ax = x & 0x7fu;
    const bool sign = (x & 0x80u) != 0;
    if (ax == 0) return sign ? -0.0f : 0.0f;
    if (ax == 0x7f) return std::numeric_limits<float>::quiet_NaN();
    const int exp = (x >> 3) & 0x0f;
    const int man = x & 0x07;
    const float value = exp == 0 ? std::ldexp((float)man, -9)
                                 : std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
    return sign ? -value : value;
}

float cpu_f8_dot(const uint8_t *row, const float *x, int cols) {
    double acc = 0.0;
    const int blocks = cols / 128;
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *block = row + (uint64_t)b * 129ull;
        const float scale = e8m0_to_f32_host(block[0]);
        for (int c = 0; c < 128; ++c) {
            acc += (double)(e4m3fn_to_f32_host(block[1 + c]) * scale) *
                   (double)x[b * 128 + c];
        }
    }
    return (float)acc;
}

float bf16_to_f32_host(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float v = 0.0f;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

float cpu_bf16_dot(const uint16_t *row, const float *x, int cols) {
    double acc = 0.0;
    for (int c = 0; c < cols; ++c) {
        acc += (double)bf16_to_f32_host(row[c]) * (double)x[c];
    }
    return (float)acc;
}

int device_checksum_row(int device, const char *pack_dir, const ContractRow &r,
                        uint64_t *checksum) {
    if (r.bytes_estimate == 0 || r.source_pack_file == "-") return 0;
    CHECK_CUDA(cudaSetDevice(device));
    const uint64_t offset = physical_row_offset(r);
    if (offset + r.bytes_estimate > r.source_shard_offset + r.source_byte_length &&
        r.record_type == "dense_tp") {
        std::fprintf(stderr, "dense shard exceeds source span for %s\n", r.tensor_id.c_str());
        return 1;
    }
    std::vector<unsigned char> host((size_t)r.bytes_estimate);
    const std::string path = path_join(pack_dir, r.source_pack_file);
    if (read_exact_at(path, offset, host.data(), host.size()) != 0) return 2;

    unsigned char *d = nullptr;
    unsigned long long *d_sum = nullptr;
    CHECK_CUDA(cudaMalloc(&d, host.size()));
    CHECK_CUDA(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemcpy(d, host.data(), host.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    const int block = 256;
    const int grid = (int)std::min<uint64_t>(4096, (r.bytes_estimate + block - 1) / block);
    checksum_bytes_kernel<<<std::max(grid, 1), block>>>(d, r.bytes_estimate, d_sum);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    unsigned long long h_sum = 0;
    CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(h_sum), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d));
    CHECK_CUDA(cudaFree(d_sum));
    *checksum = (uint64_t)h_sum;
    return 0;
}

bool select_dense_rows(const std::vector<ContractRow> &rows,
                       const char *tensor,
                       std::vector<ContractRow> *selected,
                       int *cols,
                       int *total_rows) {
    selected->clear();
    for (const ContractRow &r : rows) {
        if (r.record_type == "dense_tp" && r.tensor_id == tensor) selected->push_back(r);
    }
    if ((int)selected->size() != kGpus) return false;
    std::sort(selected->begin(), selected->end(),
              [](const ContractRow &a, const ContractRow &b) {
                  return a.owning_gpu < b.owning_gpu;
              });
    int parsed_cols = 0;
    int parsed_rows = 0;
    if (!parse_shape2((*selected)[0].source_shape, &parsed_cols, &parsed_rows)) return false;
    for (int i = 0; i < kGpus; ++i) {
        const ContractRow &r = (*selected)[i];
        if (r.owning_gpu != i ||
            r.tp_rank != i ||
            r.shard_index != i ||
            r.shard_count != kGpus ||
            r.source_dtype != "f8_e4m3_b128" ||
            r.source_shape != (*selected)[0].source_shape) {
            return false;
        }
    }
    if (parsed_cols % 128 != 0 || parsed_rows % kGpus != 0) return false;
    const uint64_t row_bytes = f8_row_bytes(parsed_cols);
    const uint64_t rows_per_gpu = (uint64_t)parsed_rows / kGpus;
    for (const ContractRow &r : *selected) {
        if (r.bytes_estimate != row_bytes * rows_per_gpu) return false;
    }
    *cols = parsed_cols;
    *total_rows = parsed_rows;
    return true;
}

std::vector<std::string> discover_f8_dense_tensors(const std::vector<ContractRow> &rows) {
    std::vector<std::string> out;
    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" || r.source_dtype != "f8_e4m3_b128") continue;
        if (std::find(out.begin(), out.end(), r.tensor_id) == out.end()) {
            out.push_back(r.tensor_id);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool select_bf16_dense_rows(const std::vector<ContractRow> &rows,
                            const char *tensor,
                            std::vector<ContractRow> *selected,
                            int *cols,
                            int *total_rows) {
    selected->clear();
    for (const ContractRow &r : rows) {
        if (r.record_type == "dense_tp" && r.tensor_id == tensor) selected->push_back(r);
    }
    if ((int)selected->size() != kGpus) return false;
    std::sort(selected->begin(), selected->end(),
              [](const ContractRow &a, const ContractRow &b) {
                  return a.owning_gpu < b.owning_gpu;
              });
    int parsed_cols = 0;
    int parsed_rows = 0;
    if (!parse_shape2((*selected)[0].source_shape, &parsed_cols, &parsed_rows)) return false;
    for (int i = 0; i < kGpus; ++i) {
        const ContractRow &r = (*selected)[i];
        if (r.owning_gpu != i ||
            r.tp_rank != i ||
            r.shard_index != i ||
            r.shard_count != kGpus ||
            r.source_dtype != "bf16" ||
            r.source_shape != (*selected)[0].source_shape) {
            return false;
        }
    }
    if (parsed_rows % kGpus != 0) return false;
    const uint64_t rows_per_gpu = (uint64_t)parsed_rows / kGpus;
    const uint64_t shard_bytes = rows_per_gpu * (uint64_t)parsed_cols * sizeof(uint16_t);
    for (const ContractRow &r : *selected) {
        if (r.bytes_estimate != shard_bytes) return false;
    }
    *cols = parsed_cols;
    *total_rows = parsed_rows;
    return true;
}

std::vector<std::string> discover_bf16_dense_tensors(const std::vector<ContractRow> &rows) {
    std::vector<std::string> out;
    for (const ContractRow &r : rows) {
        if (r.record_type != "dense_tp" || r.source_dtype != "bf16") continue;
        if (std::find(out.begin(), out.end(), r.tensor_id) == out.end()) {
            out.push_back(r.tensor_id);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

int run_dense_compute_gate(const Options &opt,
                           const std::vector<ContractRow> &rows,
                           const char *tensor,
                           DenseComputeStats *stats) {
    if (!tensor) return 0;
    stats->enabled = true;
    stats->tensor_id = tensor;
    stats->slots = opt.slots;

    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    if (!select_dense_rows(rows, tensor, &selected, &cols, &total_rows)) {
        std::fprintf(stderr, "dense compute tensor validation failed for %s\n",
                     tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    const uint64_t row_bytes = f8_row_bytes(cols);
    const uint64_t shard_bytes = row_bytes * (uint64_t)rows_per_gpu;
    stats->rows_per_gpu = rows_per_gpu;
    stats->cols = cols;

    std::vector<float> h_x((size_t)opt.slots * cols);
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int c = 0; c < cols; ++c) {
            const int m = (slot * 17 + c * 13) % 257;
            h_x[(size_t)slot * cols + c] = ((float)m - 128.0f) * 0.00025f;
        }
    }

    double worst_ms = 0.0;
    std::vector<std::vector<uint8_t>> host_weights((size_t)kGpus);
    std::vector<std::vector<float>> host_outputs((size_t)kGpus);

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = selected[(size_t)gpu];
        host_weights[(size_t)gpu].resize((size_t)shard_bytes);
        host_outputs[(size_t)gpu].resize((size_t)opt.slots * rows_per_gpu);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r),
                          host_weights[(size_t)gpu].data(), (size_t)shard_bytes) != 0) {
            return 2;
        }
        stats->loaded_bytes += shard_bytes;

        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        uint8_t *d_w = nullptr;
        float *d_x = nullptr;
        float *d_out1 = nullptr;
        float *d_out2 = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaMalloc(&d_w, (size_t)shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out1, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out2, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_w, host_weights[(size_t)gpu].data(), (size_t)shard_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        const dim3 grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1);
        for (int i = 0; i < opt.warmup; ++i) {
            f8_b128_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                                cols, (uint32_t)row_bytes, opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < opt.iters; ++i) {
            f8_b128_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                                cols, (uint32_t)row_bytes, opt.slots);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        worst_ms = std::max(worst_ms, (double)ms / opt.iters);

        f8_b128_dense_kernel<<<grid, 256>>>(d_out2, d_w, d_x, rows_per_gpu,
                                            cols, (uint32_t)row_bytes, opt.slots);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> second(host_outputs[(size_t)gpu].size());
        CHECK_CUDA(cudaMemcpy(host_outputs[(size_t)gpu].data(), d_out1,
                              host_outputs[(size_t)gpu].size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(second.data(), d_out2,
                              second.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < second.size(); ++i) {
            const float a = host_outputs[(size_t)gpu][i];
            const float b = second[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                stats->repeat_nan++;
                stats->pass = false;
                continue;
            }
            const double diff = std::fabs((double)a - (double)b);
            stats->repeat_max_abs = std::max(stats->repeat_max_abs, diff);
            if (diff > 0.0) {
                stats->repeat_bad++;
                stats->pass = false;
            }
        }

        const int sample_slots = std::min(opt.slots, 2);
        const int sample_rows = std::min(rows_per_gpu, 4);
        for (int slot = 0; slot < sample_slots; ++slot) {
            for (int row = 0; row < sample_rows; ++row) {
                const float expected =
                    cpu_f8_dot(host_weights[(size_t)gpu].data() + (uint64_t)row * row_bytes,
                               h_x.data() + (size_t)slot * cols, cols);
                const float got = host_outputs[(size_t)gpu][(size_t)slot * rows_per_gpu + row];
                const double diff = std::fabs((double)expected - (double)got);
                stats->oracle_max_abs = std::max(stats->oracle_max_abs, diff);
                const double tol = 1.0e-4 + std::fabs((double)expected) * 1.0e-4;
                if (!std::isfinite(got) || diff > tol) {
                    stats->oracle_bad++;
                    stats->pass = false;
                }
            }
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_out1));
        CHECK_CUDA(cudaFree(d_out2));
    }

    stats->compute_ms = worst_ms;
    return stats->pass ? 0 : 3;
}

int run_bf16_dense_compute_gate(const Options &opt,
                                const std::vector<ContractRow> &rows,
                                const char *tensor,
                                DenseComputeStats *stats) {
    if (!tensor) return 0;
    stats->enabled = true;
    stats->tensor_id = tensor;
    stats->slots = opt.slots;

    std::vector<ContractRow> selected;
    int cols = 0;
    int total_rows = 0;
    if (!select_bf16_dense_rows(rows, tensor, &selected, &cols, &total_rows)) {
        std::fprintf(stderr, "bf16 dense compute tensor validation failed for %s\n",
                     tensor);
        return 1;
    }
    const int rows_per_gpu = total_rows / kGpus;
    const uint64_t shard_bytes = (uint64_t)rows_per_gpu * cols * sizeof(uint16_t);
    stats->rows_per_gpu = rows_per_gpu;
    stats->cols = cols;

    std::vector<float> h_x((size_t)opt.slots * cols);
    for (int slot = 0; slot < opt.slots; ++slot) {
        for (int c = 0; c < cols; ++c) {
            const int m = (slot * 19 + c * 11) % 263;
            h_x[(size_t)slot * cols + c] = ((float)m - 131.0f) * 0.00025f;
        }
    }

    double worst_ms = 0.0;
    std::vector<std::vector<uint16_t>> host_weights((size_t)kGpus);
    std::vector<std::vector<float>> host_outputs((size_t)kGpus);

    for (int gpu = 0; gpu < kGpus; ++gpu) {
        const ContractRow &r = selected[(size_t)gpu];
        host_weights[(size_t)gpu].resize((size_t)rows_per_gpu * cols);
        host_outputs[(size_t)gpu].resize((size_t)opt.slots * rows_per_gpu);
        const std::string path = path_join(opt.pack_dir, r.source_pack_file);
        if (read_exact_at(path, physical_row_offset(r),
                          host_weights[(size_t)gpu].data(), (size_t)shard_bytes) != 0) {
            return 2;
        }
        stats->loaded_bytes += shard_bytes;

        CHECK_CUDA(cudaSetDevice(opt.devices[gpu]));
        uint16_t *d_w = nullptr;
        float *d_x = nullptr;
        float *d_out1 = nullptr;
        float *d_out2 = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaMalloc(&d_w, (size_t)shard_bytes));
        CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out1, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out2, host_outputs[(size_t)gpu].size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_w, host_weights[(size_t)gpu].data(), (size_t)shard_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        const dim3 grid((unsigned int)rows_per_gpu, (unsigned int)opt.slots, 1);
        for (int i = 0; i < opt.warmup; ++i) {
            bf16_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                             cols, cols, opt.slots);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < opt.iters; ++i) {
            bf16_dense_kernel<<<grid, 256>>>(d_out1, d_w, d_x, rows_per_gpu,
                                             cols, cols, opt.slots);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        worst_ms = std::max(worst_ms, (double)ms / opt.iters);

        bf16_dense_kernel<<<grid, 256>>>(d_out2, d_w, d_x, rows_per_gpu,
                                         cols, cols, opt.slots);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> second(host_outputs[(size_t)gpu].size());
        CHECK_CUDA(cudaMemcpy(host_outputs[(size_t)gpu].data(), d_out1,
                              host_outputs[(size_t)gpu].size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(second.data(), d_out2,
                              second.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < second.size(); ++i) {
            const float a = host_outputs[(size_t)gpu][i];
            const float b = second[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                stats->repeat_nan++;
                stats->pass = false;
                continue;
            }
            const double diff = std::fabs((double)a - (double)b);
            stats->repeat_max_abs = std::max(stats->repeat_max_abs, diff);
            if (diff > 0.0) {
                stats->repeat_bad++;
                stats->pass = false;
            }
        }

        const int sample_slots = std::min(opt.slots, 2);
        const int sample_rows = std::min(rows_per_gpu, 4);
        for (int slot = 0; slot < sample_slots; ++slot) {
            for (int row = 0; row < sample_rows; ++row) {
                const float expected =
                    cpu_bf16_dot(host_weights[(size_t)gpu].data() + (uint64_t)row * cols,
                                 h_x.data() + (size_t)slot * cols, cols);
                const float got = host_outputs[(size_t)gpu][(size_t)slot * rows_per_gpu + row];
                const double diff = std::fabs((double)expected - (double)got);
                stats->oracle_max_abs = std::max(stats->oracle_max_abs, diff);
                const double tol = 1.0e-4 + std::fabs((double)expected) * 1.0e-4;
                if (!std::isfinite(got) || diff > tol) {
                    stats->oracle_bad++;
                    stats->pass = false;
                }
            }
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_out1));
        CHECK_CUDA(cudaFree(d_out2));
    }

    stats->compute_ms = worst_ms;
    return stats->pass ? 0 : 3;
}

struct SharedHcControls {
    bool initialized = false;
    int slots = 0;
    int devices[kGpus] = {};
    float *d_hc = nullptr;
    float *d_hc_norm = nullptr;
    float *d_mix = nullptr;
    float *d_split = nullptr;
    float *d_current_full = nullptr;
    float *d_attn_normed = nullptr;
    float *d_q_a_full = nullptr;
    float *d_q_a_normed = nullptr;
    float *d_kv_full = nullptr;
    float *d_kv_normed = nullptr;
    float *d_ffn_normed = nullptr;
    float *d_attn_comp_kv_full = nullptr;
    float *d_attn_comp_score_full = nullptr;
    float *d_index_comp_kv_full = nullptr;
    float *d_index_comp_score_full = nullptr;
    float *d_indexer_q_full = nullptr;
    float *d_indexer_w_full = nullptr;
    /* [44] = 43 transformer layers (0-42) + the MTP block at layer 43 */
    float *d_attn_norm_weight[44] = {};
    float *d_attn_norm_weight_rank[44][kGpus] = {};
    float *d_q_a_norm_weight[44] = {};
    float *d_kv_a_norm_weight[44] = {};
    float *d_attn_compress_ape[44] = {};
    float *d_attn_compress_norm[44] = {};
    float *d_indexer_compress_ape[44] = {};
    float *d_indexer_compress_norm[44] = {};
    float *d_attn_sinks[44] = {};
    float *d_attn_fn[44] = {};
    float *d_attn_fn_rank[44][kGpus] = {};
    float *d_attn_base[44] = {};
    float *d_attn_base_rank[44][kGpus] = {};
    float *d_attn_scale[44] = {};
    float *d_attn_scale_rank[44][kGpus] = {};
    float *d_ffn_fn[44] = {};
    float *d_ffn_fn_rank[44][kGpus] = {};
    float *d_ffn_base[44] = {};
    float *d_ffn_base_rank[44][kGpus] = {};
    float *d_ffn_scale[44] = {};
    float *d_ffn_scale_rank[44][kGpus] = {};
    float *d_ffn_norm_weight[44] = {};
    float *d_ffn_norm_weight_rank[44][kGpus] = {};
    float *d_router_w[44] = {};
    float *d_router_w_ep[44][kGpus] = {};
    float *d_router_w_shard[44][kGpus] = {};
    float *d_router_bias[44] = {};
    int *d_router_hash[44] = {};
    uint32_t router_hash_rows[44] = {};
    float *d_router_logits = nullptr;
    int *d_router_selected = nullptr;
    float *d_router_weights = nullptr;
    uint32_t *d_router_tokens = nullptr;
    unsigned char *d_router_active = nullptr;
    cublasHandle_t router_blas = nullptr;
    RoutePlanHostWorkspace route_plan_ws;
    uint64_t control_bytes = 0;
};

int init_route_plan_host_workspace(const Options &opt,
                                   RoutePlanHostWorkspace *ws) {
    if (!ws) return 1;
    if (ws->initialized) return 0;
    ws->slots = opt.slots;
    ws->top_k = opt.top_k;
    for (int rank = 0; rank < kGpus; ++rank) {
        ws->devices[rank] = opt.devices[rank];
    }
    ws->route_capacity = (size_t)opt.slots * (size_t)opt.top_k;
    ws->compact_plan_ints =
        (size_t)kGpus * ((size_t)opt.slots * (size_t)opt.top_k +
                         (size_t)opt.slots);
    CHECK_CUDA(cudaHostAlloc(&ws->h_selected,
                             ws->route_capacity * sizeof(int),
                             cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&ws->h_weights,
                             ws->route_capacity * sizeof(float),
                             cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&ws->h_compact_plan,
                             ws->compact_plan_ints * sizeof(int),
                             cudaHostAllocDefault));
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaHostAlloc(&ws->h_offsets[rank],
                                 (size_t)(kLocalExperts + 1) * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_slots[rank],
                                 ws->route_capacity * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_weights[rank],
                                 ws->route_capacity * sizeof(float),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_index_by_slot[rank],
                                 (size_t)opt.slots * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_indices_by_slot[rank],
                                 ws->route_capacity * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&ws->h_route_count_by_slot[rank],
                                 (size_t)opt.slots * sizeof(int),
                                 cudaHostAllocDefault));
        CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
        CHECK_CUDA(cudaEventCreateWithFlags(&ws->upload_done[rank],
                                            cudaEventDisableTiming));
    }
    ws->initialized = true;
    return 0;
}

void close_route_plan_host_workspace(RoutePlanHostWorkspace *ws) {
    if (!ws || !ws->initialized) return;
    if (ws->uploads_pending) {
        for (int rank = 0; rank < kGpus; ++rank) {
            if (ws->upload_done[rank]) {
                CHECK_CUDA(cudaSetDevice(ws->devices[rank]));
                CHECK_CUDA(cudaEventSynchronize(ws->upload_done[rank]));
            }
        }
        ws->uploads_pending = false;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ws->devices[rank]));
        if (ws->upload_done[rank]) CHECK_CUDA(cudaEventDestroy(ws->upload_done[rank]));
        if (ws->h_route_count_by_slot[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_count_by_slot[rank]));
        if (ws->h_route_indices_by_slot[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_indices_by_slot[rank]));
        if (ws->h_route_index_by_slot[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_index_by_slot[rank]));
        if (ws->h_route_weights[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_weights[rank]));
        if (ws->h_route_slots[rank]) CHECK_CUDA(cudaFreeHost(ws->h_route_slots[rank]));
        if (ws->h_offsets[rank]) CHECK_CUDA(cudaFreeHost(ws->h_offsets[rank]));
    }
    if (ws->h_compact_plan) CHECK_CUDA(cudaFreeHost(ws->h_compact_plan));
    if (ws->h_weights) CHECK_CUDA(cudaFreeHost(ws->h_weights));
    if (ws->h_selected) CHECK_CUDA(cudaFreeHost(ws->h_selected));
    *ws = RoutePlanHostWorkspace{};
}

bool find_replicated_control_row(const std::vector<ContractRow> &rows,
                                 const char *tensor,
                                 ContractRow *out) {
    for (const ContractRow &r : rows) {
        if (r.record_type == "replicated_control" && r.tensor_id == tensor) {
            *out = r;
            return true;
        }
    }
    return false;
}

int load_control_f32(const Options &opt,
                     const std::vector<ContractRow> &rows,
                     const char *tensor,
                     size_t elems,
                     std::vector<float> *out) {
    ContractRow r;
    if (!find_replicated_control_row(rows, tensor, &r)) {
        std::fprintf(stderr, "missing replicated control tensor %s\n", tensor);
        return 1;
    }
    if (r.source_dtype != "f32" || r.bytes_estimate != elems * sizeof(float)) {
        std::fprintf(stderr, "bad replicated control tensor %s dtype=%s bytes=%llu expected=%zu\n",
                     tensor, r.source_dtype.c_str(),
                     (unsigned long long)r.bytes_estimate, elems * sizeof(float));
        return 2;
    }
    out->resize(elems);
    const std::string path = path_join(opt.pack_dir, r.source_pack_file);
    if (read_exact_at(path, physical_row_offset(r), out->data(), elems * sizeof(float)) != 0) {
        return 3;
    }
    return 0;
}

int load_optional_control_f32(const Options &opt,
                              const std::vector<ContractRow> &rows,
                              const char *tensor,
                              size_t elems,
                              std::vector<float> *out,
                              bool *found) {
    ContractRow r;
    if (!find_replicated_control_row(rows, tensor, &r)) {
        out->clear();
        if (found) *found = false;
        return 0;
    }
    if (found) *found = true;
    return load_control_f32(opt, rows, tensor, elems, out);
}

int load_optional_control_i32(const Options &opt,
                              const std::vector<ContractRow> &rows,
                              const char *tensor,
                              size_t elems,
                              std::vector<int> *out,
                              bool *found) {
    ContractRow r;
    if (!find_replicated_control_row(rows, tensor, &r)) {
        out->clear();
        if (found) *found = false;
        return 0;
    }
    if (found) *found = true;
    if (r.source_dtype != "i32" || r.bytes_estimate != elems * sizeof(int)) {
        std::fprintf(stderr, "bad replicated control tensor %s dtype=%s bytes=%llu expected=%zu\n",
                     tensor, r.source_dtype.c_str(),
                     (unsigned long long)r.bytes_estimate, elems * sizeof(int));
        return 2;
    }
    out->resize(elems);
    const std::string path = path_join(opt.pack_dir, r.source_pack_file);
    if (read_exact_at(path, physical_row_offset(r), out->data(), elems * sizeof(int)) != 0) {
        return 3;
    }
    return 0;
}

int open_shared_hc_controls(const Options &opt,
                            const std::vector<ContractRow> &rows,
                            SharedHcControls *out) {
    out->slots = opt.slots;
    for (int rank = 0; rank < kGpus; ++rank) out->devices[rank] = opt.devices[rank];
    const uint64_t hc_elems = (uint64_t)opt.slots * kHcRows * (uint64_t)kHidden;
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    cublasStatus_t blas_status = cublasCreate(&out->router_blas);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "router cublasCreate failed status=%d\n",
                     (int)blas_status);
        return 1;
    }
    CHECK_CUDA(cudaMalloc(&out->d_hc, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_hc_norm, (size_t)hc_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_mix, (size_t)opt.slots * kHcMix * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_split, (size_t)opt.slots * kHcMix * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_current_full,
                          (size_t)opt.slots * kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_attn_normed,
                          (size_t)opt.slots * kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_q_a_full,
                          (size_t)opt.slots * 1024u * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_q_a_normed,
                          (size_t)opt.slots * 1024u * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_kv_full,
                          (size_t)opt.slots * kHeadDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_kv_normed,
                          (size_t)opt.slots * kHeadDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_ffn_normed,
                          (size_t)opt.slots * kHidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_attn_comp_kv_full,
                          (size_t)opt.slots * kCompWidthMax * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_attn_comp_score_full,
                          (size_t)opt.slots * kCompWidthMax * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_index_comp_kv_full,
                          (size_t)opt.slots * kIndexCompWidth * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_index_comp_score_full,
                          (size_t)opt.slots * kIndexCompWidth * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_indexer_q_full,
                          (size_t)opt.slots * kIndexerHead *
                              (size_t)kIndexerHeadDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_indexer_w_full,
                          (size_t)opt.slots * kIndexerHead * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_router_logits,
                          (size_t)opt.slots * kGlobalExperts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_router_selected,
                          (size_t)opt.slots * kModelTopK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&out->d_router_weights,
                          (size_t)opt.slots * kModelTopK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out->d_router_tokens,
                          (size_t)opt.slots * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&out->d_router_active,
                          (size_t)opt.slots * sizeof(unsigned char)));
    CHECK_CUDA(cudaMemset(out->d_router_tokens, 0,
                          (size_t)opt.slots * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(out->d_router_active, 1,
                          (size_t)opt.slots * sizeof(unsigned char)));
    if (opt.route_plan_async_upload_gate &&
        init_route_plan_host_workspace(opt, &out->route_plan_ws) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));

    for (int layer = 0; layer < 43; ++layer) {
        std::vector<float> attn_fn;
        std::vector<float> attn_base;
        std::vector<float> attn_scale;
        std::vector<float> fn;
        std::vector<float> base;
        std::vector<float> scale;
        std::vector<float> ffn_norm_weight;
        std::vector<float> attn_norm_weight;
        std::vector<float> q_a_norm_weight;
        std::vector<float> kv_a_norm_weight;
        std::vector<float> attn_sinks;
        std::vector<float> attn_compress_ape;
        std::vector<float> attn_compress_norm;
        std::vector<float> indexer_compress_ape;
        std::vector<float> indexer_compress_norm;
        std::vector<float> router_w;
        std::vector<float> router_bias;
        std::vector<int> router_hash;
        const std::string attn_norm_name = layer_tensor_name(layer, "attn_norm.weight");
        const std::string q_a_norm_name = layer_tensor_name(layer, "attn_q_a_norm.weight");
        const std::string kv_a_norm_name = layer_tensor_name(layer, "attn_kv_a_norm.weight");
        const std::string attn_sinks_name = layer_tensor_name(layer, "attn_sinks");
        const std::string attn_compress_ape_name =
            layer_tensor_name(layer, "attn_compress_ape");
        const std::string attn_compress_norm_name =
            layer_tensor_name(layer, "attn_compress_norm.weight");
        const std::string indexer_compress_ape_name =
            layer_tensor_name(layer, "indexer.compress_ape");
        const std::string indexer_compress_norm_name =
            layer_tensor_name(layer, "indexer.compress_norm.weight");
        const std::string attn_fn_name = layer_tensor_name(layer, "hc_attn_fn");
        const std::string attn_base_name = layer_tensor_name(layer, "hc_attn_base");
        const std::string attn_scale_name = layer_tensor_name(layer, "hc_attn_scale");
        const std::string fn_name = layer_tensor_name(layer, "hc_ffn_fn");
        const std::string base_name = layer_tensor_name(layer, "hc_ffn_base");
        const std::string scale_name = layer_tensor_name(layer, "hc_ffn_scale");
        const std::string ffn_norm_name = layer_tensor_name(layer, "ffn_norm.weight");
        const std::string router_name = layer_tensor_name(layer, "ffn_gate_inp.weight");
        const std::string bias_name = layer_tensor_name(layer, "exp_probs_b");
        const std::string hash_name = layer_tensor_name(layer, "ffn_gate_tid2eid");
        const int ratio = ds4_layer_ratio(layer);
        bool have_attn_compress_ape = false;
        bool have_attn_compress_norm = false;
        bool have_indexer_compress_ape = false;
        bool have_indexer_compress_norm = false;
        bool have_bias = false;
        bool have_hash = false;
        if (load_control_f32(opt, rows, attn_fn_name.c_str(),
                             (size_t)kHcRows * (size_t)kHidden * kHcMix, &attn_fn) ||
            load_control_f32(opt, rows, attn_base_name.c_str(), kHcMix, &attn_base) ||
            load_control_f32(opt, rows, attn_scale_name.c_str(), 3, &attn_scale) ||
            load_control_f32(opt, rows, attn_norm_name.c_str(),
                             kHidden, &attn_norm_weight) ||
            load_control_f32(opt, rows, q_a_norm_name.c_str(),
                             1024, &q_a_norm_weight) ||
            load_control_f32(opt, rows, kv_a_norm_name.c_str(),
                             kHeadDim, &kv_a_norm_weight) ||
            load_control_f32(opt, rows, attn_sinks_name.c_str(),
                             kHeadCount, &attn_sinks) ||
            (ratio != 0 &&
             (load_optional_control_f32(opt, rows, attn_compress_ape_name.c_str(),
                                        (size_t)ratio *
                                            (size_t)(ratio == 4 ? kCompWidthMax
                                                               : kHeadDim),
                                        &attn_compress_ape,
                                        &have_attn_compress_ape) ||
              load_optional_control_f32(opt, rows, attn_compress_norm_name.c_str(),
                                        kHeadDim, &attn_compress_norm,
                                        &have_attn_compress_norm))) ||
            (ratio == 4 &&
             (load_optional_control_f32(opt, rows, indexer_compress_ape_name.c_str(),
                                        (size_t)ratio * (size_t)kIndexCompWidth,
                                        &indexer_compress_ape,
                                        &have_indexer_compress_ape) ||
              load_optional_control_f32(opt, rows, indexer_compress_norm_name.c_str(),
                                        kIndexerHeadDim,
                                        &indexer_compress_norm,
                                        &have_indexer_compress_norm))) ||
            load_control_f32(opt, rows, fn_name.c_str(),
                             (size_t)kHcRows * (size_t)kHidden * kHcMix, &fn) ||
            load_control_f32(opt, rows, base_name.c_str(), kHcMix, &base) ||
            load_control_f32(opt, rows, scale_name.c_str(), 3, &scale) ||
            load_control_f32(opt, rows, ffn_norm_name.c_str(),
                             kHidden, &ffn_norm_weight) ||
            load_control_f32(opt, rows, router_name.c_str(),
                             (size_t)kHidden * kGlobalExperts, &router_w) ||
            load_optional_control_f32(opt, rows, bias_name.c_str(),
                                      kGlobalExperts, &router_bias, &have_bias) ||
            load_optional_control_i32(opt, rows, hash_name.c_str(),
                                      (size_t)kRouterHashRows * kModelTopK,
                                      &router_hash, &have_hash)) {
            return 1;
        }
        CHECK_CUDA(cudaMalloc(&out->d_attn_fn[layer], attn_fn.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_base[layer], attn_base.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_scale[layer], attn_scale.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_fn[layer], fn.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_base[layer], base.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_scale[layer], scale.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_ffn_norm_weight[layer],
                              ffn_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_norm_weight[layer],
                              attn_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_q_a_norm_weight[layer],
                              q_a_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_kv_a_norm_weight[layer],
                              kv_a_norm_weight.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&out->d_attn_sinks[layer],
                              attn_sinks.size() * sizeof(float)));
        if (have_attn_compress_ape && have_attn_compress_norm) {
            CHECK_CUDA(cudaMalloc(&out->d_attn_compress_ape[layer],
                                  attn_compress_ape.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&out->d_attn_compress_norm[layer],
                                  attn_compress_norm.size() * sizeof(float)));
        }
        if (have_indexer_compress_ape && have_indexer_compress_norm) {
            CHECK_CUDA(cudaMalloc(&out->d_indexer_compress_ape[layer],
                                  indexer_compress_ape.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&out->d_indexer_compress_norm[layer],
                                  indexer_compress_norm.size() * sizeof(float)));
        }
        if (!opt.model_router_rank_major_logits_gate &&
            !opt.model_router_allreduce_logits_gate) {
            CHECK_CUDA(cudaMalloc(&out->d_router_w[layer],
                                  router_w.size() * sizeof(float)));
        }
        if (opt.tp_hc_current_allreduce_gate) {
            const int shard_cols = kHidden / kGpus;
            const size_t local_cols = (size_t)kHcRows * (size_t)shard_cols;
            std::vector<float> fn_rank(local_cols * (size_t)kHcMix);
            std::vector<float> attn_fn_rank(local_cols * (size_t)kHcMix);
            for (int rank = 0; rank < kGpus; ++rank) {
                for (int row = 0; row < kHcRows; ++row) {
                    for (int local_h = 0; local_h < shard_cols; ++local_h) {
                        const size_t local_c =
                            (size_t)row * (size_t)shard_cols + (size_t)local_h;
                        const size_t global_c =
                            (size_t)row * (size_t)kHidden +
                            (size_t)rank * (size_t)shard_cols +
                            (size_t)local_h;
                        for (int mix = 0; mix < kHcMix; ++mix) {
                            attn_fn_rank[local_c * (size_t)kHcMix + (size_t)mix] =
                                attn_fn[global_c * (size_t)kHcMix + (size_t)mix];
                            fn_rank[local_c * (size_t)kHcMix + (size_t)mix] =
                                fn[global_c * (size_t)kHcMix + (size_t)mix];
                        }
                    }
                }
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_attn_fn_rank[layer][rank],
                                      attn_fn_rank.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_attn_fn_rank[layer][rank],
                                      attn_fn_rank.data(),
                                      attn_fn_rank.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_attn_base_rank[layer][rank],
                                      attn_base.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_attn_base_rank[layer][rank],
                                      attn_base.data(),
                                      attn_base.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_attn_scale_rank[layer][rank],
                                      attn_scale.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_attn_scale_rank[layer][rank],
                                      attn_scale.data(),
                                      attn_scale.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_ffn_fn_rank[layer][rank],
                                      fn_rank.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_ffn_fn_rank[layer][rank],
                                      fn_rank.data(),
                                      fn_rank.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_ffn_base_rank[layer][rank],
                                      base.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_ffn_base_rank[layer][rank],
                                      base.data(),
                                      base.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMalloc(&out->d_ffn_scale_rank[layer][rank],
                                      scale.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_ffn_scale_rank[layer][rank],
                                      scale.data(),
                                      scale.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        if (opt.model_router_rank_major_logits_gate) {
            std::vector<float> router_w_ep((size_t)kHidden * (size_t)kLocalExperts);
            for (int rank = 0; rank < kGpus; ++rank) {
                for (int h = 0; h < kHidden; ++h) {
                    for (int e = 0; e < kLocalExperts; ++e) {
                        router_w_ep[(size_t)h * (size_t)kLocalExperts + (size_t)e] =
                            router_w[(size_t)h * (size_t)kGlobalExperts +
                                     (size_t)(rank * kLocalExperts + e)];
                    }
                }
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_router_w_ep[layer][rank],
                                      router_w_ep.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_router_w_ep[layer][rank],
                                      router_w_ep.data(),
                                      router_w_ep.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        if (opt.model_router_allreduce_logits_gate) {
            const int shard_cols = kHidden / kGpus;
            std::vector<float> router_w_shard(
                (size_t)shard_cols * (size_t)kGlobalExperts);
            for (int rank = 0; rank < kGpus; ++rank) {
                for (int local_h = 0; local_h < shard_cols; ++local_h) {
                    const int global_h = rank * shard_cols + local_h;
                    for (int expert = 0; expert < kGlobalExperts; ++expert) {
                        router_w_shard[(size_t)local_h *
                                           (size_t)kGlobalExperts +
                                       (size_t)expert] =
                            router_w[(size_t)global_h *
                                         (size_t)kGlobalExperts +
                                     (size_t)expert];
                    }
                }
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_router_w_shard[layer][rank],
                                      router_w_shard.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_router_w_shard[layer][rank],
                                      router_w_shard.data(),
                                      router_w_shard.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        if (have_bias) {
            CHECK_CUDA(cudaMalloc(&out->d_router_bias[layer],
                                  router_bias.size() * sizeof(float)));
        }
        if (have_hash) {
            CHECK_CUDA(cudaMalloc(&out->d_router_hash[layer],
                                  router_hash.size() * sizeof(int)));
            out->router_hash_rows[layer] = kRouterHashRows;
        }
        CHECK_CUDA(cudaMemcpy(out->d_attn_fn[layer], attn_fn.data(),
                              attn_fn.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_attn_base[layer], attn_base.data(),
                              attn_base.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_attn_scale[layer], attn_scale.data(),
                              attn_scale.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_fn[layer], fn.data(),
                              fn.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_base[layer], base.data(),
                              base.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_scale[layer], scale.data(),
                              scale.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_ffn_norm_weight[layer], ffn_norm_weight.data(),
                              ffn_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        out->d_ffn_norm_weight_rank[layer][0] = out->d_ffn_norm_weight[layer];
        if (opt.model_router_allreduce_logits_gate ||
            opt.routed_ffn_rank_major_input_gate ||
            opt.routed_ffn_rank_major_shared_input_gate ||
            opt.routed_ffn_rank_major_route_input_gate ||
            opt.routed_ffn_rank_major_input_parity_gate) {
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_ffn_norm_weight_rank[layer][rank],
                                      ffn_norm_weight.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_ffn_norm_weight_rank[layer][rank],
                                      ffn_norm_weight.data(),
                                      ffn_norm_weight.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        CHECK_CUDA(cudaMemcpy(out->d_attn_norm_weight[layer], attn_norm_weight.data(),
                              attn_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        out->d_attn_norm_weight_rank[layer][0] = out->d_attn_norm_weight[layer];
        if (opt.true_ds4_attention_projection_rank_local_input_gate ||
            opt.true_ds4_attention_projection_rank_major_input_gate) {
            for (int rank = 1; rank < kGpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
                CHECK_CUDA(cudaMalloc(&out->d_attn_norm_weight_rank[layer][rank],
                                      attn_norm_weight.size() * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(out->d_attn_norm_weight_rank[layer][rank],
                                      attn_norm_weight.data(),
                                      attn_norm_weight.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[0]));
        }
        CHECK_CUDA(cudaMemcpy(out->d_q_a_norm_weight[layer], q_a_norm_weight.data(),
                              q_a_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_kv_a_norm_weight[layer], kv_a_norm_weight.data(),
                              kv_a_norm_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(out->d_attn_sinks[layer], attn_sinks.data(),
                              attn_sinks.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (out->d_attn_compress_ape[layer]) {
            CHECK_CUDA(cudaMemcpy(out->d_attn_compress_ape[layer],
                                  attn_compress_ape.data(),
                                  attn_compress_ape.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(out->d_attn_compress_norm[layer],
                                  attn_compress_norm.data(),
                                  attn_compress_norm.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (out->d_indexer_compress_ape[layer]) {
            CHECK_CUDA(cudaMemcpy(out->d_indexer_compress_ape[layer],
                                  indexer_compress_ape.data(),
                                  indexer_compress_ape.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(out->d_indexer_compress_norm[layer],
                                  indexer_compress_norm.data(),
                                  indexer_compress_norm.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (out->d_router_w[layer]) {
            CHECK_CUDA(cudaMemcpy(out->d_router_w[layer], router_w.data(),
                                  router_w.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (have_bias) {
            CHECK_CUDA(cudaMemcpy(out->d_router_bias[layer], router_bias.data(),
                                  router_bias.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (have_hash) {
            CHECK_CUDA(cudaMemcpy(out->d_router_hash[layer], router_hash.data(),
                                  router_hash.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }
        out->control_bytes +=
            (attn_fn.size() + attn_base.size() + attn_scale.size() +
             attn_norm_weight.size() + q_a_norm_weight.size() +
             kv_a_norm_weight.size() + attn_sinks.size() +
             attn_compress_ape.size() + attn_compress_norm.size() +
             indexer_compress_ape.size() + indexer_compress_norm.size() +
             fn.size() + base.size() + scale.size() +
             ffn_norm_weight.size() + router_w.size() + router_bias.size()) *
                sizeof(float) +
            router_hash.size() * sizeof(int);
    }
    out->initialized = true;
    return 0;
}

/* MTP (layer 43) HC/norm controls: load the MTP block's global HC + norm +
 * router control tensors into SharedHcControls slot 43, from the MTP contract +
 * pack dir (NOT opt.pack_dir). Isolated from the 0-42 loop (no regression risk).
 * ratio=0 (no compress/indexer). No-op unless the MTP source is configured.
 * Loads the GLOBAL tensors; rank-distributed variants added if the decode path
 * requires them (surfaced incrementally by the load test). */
int load_mtp_hc_layer43(const Options &opt, SharedHcControls *out) {
    if (!opt.mtp_contract_path || !opt.mtp_pack_dir) return 0;
    if (!out || !out->initialized) return 1;
    const int L = 43;
    std::vector<ContractRow> rows;
    LayerStats st;
    if (parse_contract(opt.mtp_contract_path, L, &rows, &st) != 0 || st.bad_rows != 0) {
        std::fprintf(stderr, "MTP HC contract parse failed: %s\n", opt.mtp_contract_path);
        return 1;
    }
    Options mopt = opt;
    mopt.pack_dir = opt.mtp_pack_dir;  /* read layer-43 control bytes from the MTP pack */
    std::vector<float> attn_fn, attn_base, attn_scale, attn_norm_weight, q_a_norm_weight,
        kv_a_norm_weight, attn_sinks, fn, base, scale, ffn_norm_weight, router_w, router_bias;
    bool have_bias = false;
    const std::string attn_fn_name = layer_tensor_name(L, "hc_attn_fn");
    const std::string attn_base_name = layer_tensor_name(L, "hc_attn_base");
    const std::string attn_scale_name = layer_tensor_name(L, "hc_attn_scale");
    const std::string attn_norm_name = layer_tensor_name(L, "attn_norm.weight");
    const std::string q_a_norm_name = layer_tensor_name(L, "attn_q_a_norm.weight");
    const std::string kv_a_norm_name = layer_tensor_name(L, "attn_kv_a_norm.weight");
    const std::string attn_sinks_name = layer_tensor_name(L, "attn_sinks");
    const std::string fn_name = layer_tensor_name(L, "hc_ffn_fn");
    const std::string base_name = layer_tensor_name(L, "hc_ffn_base");
    const std::string scale_name = layer_tensor_name(L, "hc_ffn_scale");
    const std::string ffn_norm_name = layer_tensor_name(L, "ffn_norm.weight");
    const std::string router_name = layer_tensor_name(L, "ffn_gate_inp.weight");
    const std::string bias_name = layer_tensor_name(L, "exp_probs_b");
    if (load_control_f32(mopt, rows, attn_fn_name.c_str(),
                         (size_t)kHcRows * (size_t)kHidden * kHcMix, &attn_fn) ||
        load_control_f32(mopt, rows, attn_base_name.c_str(), kHcMix, &attn_base) ||
        load_control_f32(mopt, rows, attn_scale_name.c_str(), 3, &attn_scale) ||
        load_control_f32(mopt, rows, attn_norm_name.c_str(), kHidden, &attn_norm_weight) ||
        load_control_f32(mopt, rows, q_a_norm_name.c_str(), 1024, &q_a_norm_weight) ||
        load_control_f32(mopt, rows, kv_a_norm_name.c_str(), kHeadDim, &kv_a_norm_weight) ||
        load_control_f32(mopt, rows, attn_sinks_name.c_str(), kHeadCount, &attn_sinks) ||
        load_control_f32(mopt, rows, fn_name.c_str(),
                         (size_t)kHcRows * (size_t)kHidden * kHcMix, &fn) ||
        load_control_f32(mopt, rows, base_name.c_str(), kHcMix, &base) ||
        load_control_f32(mopt, rows, scale_name.c_str(), 3, &scale) ||
        load_control_f32(mopt, rows, ffn_norm_name.c_str(), kHidden, &ffn_norm_weight) ||
        load_control_f32(mopt, rows, router_name.c_str(),
                         (size_t)kHidden * kGlobalExperts, &router_w) ||
        load_optional_control_f32(mopt, rows, bias_name.c_str(), kGlobalExperts,
                                  &router_bias, &have_bias)) {
        std::fprintf(stderr, "MTP HC layer-43 control load failed\n");
        return 1;
    }
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
#define MTP_HC_UP(field, vec)                                                      \
    do {                                                                           \
        CHECK_CUDA(cudaMalloc(&out->field[L], (vec).size() * sizeof(float)));       \
        CHECK_CUDA(cudaMemcpy(out->field[L], (vec).data(),                          \
                              (vec).size() * sizeof(float), cudaMemcpyHostToDevice)); \
    } while (0)
    MTP_HC_UP(d_attn_fn, attn_fn);
    MTP_HC_UP(d_attn_base, attn_base);
    MTP_HC_UP(d_attn_scale, attn_scale);
    MTP_HC_UP(d_attn_norm_weight, attn_norm_weight);
    MTP_HC_UP(d_q_a_norm_weight, q_a_norm_weight);
    MTP_HC_UP(d_kv_a_norm_weight, kv_a_norm_weight);
    MTP_HC_UP(d_attn_sinks, attn_sinks);
    MTP_HC_UP(d_ffn_fn, fn);
    MTP_HC_UP(d_ffn_base, base);
    MTP_HC_UP(d_ffn_scale, scale);
    MTP_HC_UP(d_ffn_norm_weight, ffn_norm_weight);
    MTP_HC_UP(d_router_w, router_w);
    if (have_bias) MTP_HC_UP(d_router_bias, router_bias);
#undef MTP_HC_UP
    /* Per-rank norm weights: rank 0 aliases the global; ranks 1-7 replicate
     * (the allreduce-router + rank-major-attention paths read these). */
    out->d_ffn_norm_weight_rank[L][0] = out->d_ffn_norm_weight[L];
    if (opt.model_router_allreduce_logits_gate || opt.routed_ffn_rank_major_input_gate ||
        opt.routed_ffn_rank_major_shared_input_gate ||
        opt.routed_ffn_rank_major_route_input_gate ||
        opt.routed_ffn_rank_major_input_parity_gate) {
        for (int rank = 1; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
            CHECK_CUDA(cudaMalloc(&out->d_ffn_norm_weight_rank[L][rank],
                                  ffn_norm_weight.size() * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(out->d_ffn_norm_weight_rank[L][rank], ffn_norm_weight.data(),
                                  ffn_norm_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    }
    out->d_attn_norm_weight_rank[L][0] = out->d_attn_norm_weight[L];
    if (opt.true_ds4_attention_projection_rank_local_input_gate ||
        opt.true_ds4_attention_projection_rank_major_input_gate) {
        for (int rank = 1; rank < kGpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
            CHECK_CUDA(cudaMalloc(&out->d_attn_norm_weight_rank[L][rank],
                                  attn_norm_weight.size() * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(out->d_attn_norm_weight_rank[L][rank], attn_norm_weight.data(),
                                  attn_norm_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    }
    /* Rank-distributed HC variants (the tp_hc_current_allreduce path needs
     * d_attn_fn_rank[43][rank] etc.) -- mirror the 0-42 loop's sharding. */
    if (opt.tp_hc_current_allreduce_gate) {
        const int shard_cols = kHidden / kGpus;
        const size_t local_cols = (size_t)kHcRows * (size_t)shard_cols;
        std::vector<float> fn_rank(local_cols * (size_t)kHcMix);
        std::vector<float> attn_fn_rank(local_cols * (size_t)kHcMix);
        for (int rank = 0; rank < kGpus; ++rank) {
            for (int row = 0; row < kHcRows; ++row) {
                for (int local_h = 0; local_h < shard_cols; ++local_h) {
                    const size_t local_c =
                        (size_t)row * (size_t)shard_cols + (size_t)local_h;
                    const size_t global_c =
                        (size_t)row * (size_t)kHidden +
                        (size_t)rank * (size_t)shard_cols + (size_t)local_h;
                    for (int mix = 0; mix < kHcMix; ++mix) {
                        attn_fn_rank[local_c * (size_t)kHcMix + (size_t)mix] =
                            attn_fn[global_c * (size_t)kHcMix + (size_t)mix];
                        fn_rank[local_c * (size_t)kHcMix + (size_t)mix] =
                            fn[global_c * (size_t)kHcMix + (size_t)mix];
                    }
                }
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
#define MTP_HC_RANK(field, vec)                                                     \
    do {                                                                            \
        CHECK_CUDA(cudaMalloc(&out->field[L][rank], (vec).size() * sizeof(float)));   \
        CHECK_CUDA(cudaMemcpy(out->field[L][rank], (vec).data(),                     \
                              (vec).size() * sizeof(float), cudaMemcpyHostToDevice)); \
    } while (0)
            MTP_HC_RANK(d_attn_fn_rank, attn_fn_rank);
            MTP_HC_RANK(d_attn_base_rank, attn_base);
            MTP_HC_RANK(d_attn_scale_rank, attn_scale);
            MTP_HC_RANK(d_ffn_fn_rank, fn_rank);
            MTP_HC_RANK(d_ffn_base_rank, base);
            MTP_HC_RANK(d_ffn_scale_rank, scale);
#undef MTP_HC_RANK
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    }
    /* Router weight EP/TP shards (the dense-router-logits path needs
     * d_router_w_ep[43][rank] / d_router_w_shard[43][rank]) -- mirror the loop. */
    if (opt.model_router_rank_major_logits_gate) {
        std::vector<float> router_w_ep((size_t)kHidden * (size_t)kLocalExperts);
        for (int rank = 0; rank < kGpus; ++rank) {
            for (int h = 0; h < kHidden; ++h)
                for (int e = 0; e < kLocalExperts; ++e)
                    router_w_ep[(size_t)h * (size_t)kLocalExperts + (size_t)e] =
                        router_w[(size_t)h * (size_t)kGlobalExperts +
                                 (size_t)(rank * kLocalExperts + e)];
            CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
            CHECK_CUDA(cudaMalloc(&out->d_router_w_ep[L][rank],
                                  router_w_ep.size() * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(out->d_router_w_ep[L][rank], router_w_ep.data(),
                                  router_w_ep.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    }
    if (opt.model_router_allreduce_logits_gate) {
        const int shard_cols = kHidden / kGpus;
        std::vector<float> router_w_shard((size_t)shard_cols * (size_t)kGlobalExperts);
        for (int rank = 0; rank < kGpus; ++rank) {
            for (int local_h = 0; local_h < shard_cols; ++local_h) {
                const int global_h = rank * shard_cols + local_h;
                for (int expert = 0; expert < kGlobalExperts; ++expert)
                    router_w_shard[(size_t)local_h * (size_t)kGlobalExperts + (size_t)expert] =
                        router_w[(size_t)global_h * (size_t)kGlobalExperts + (size_t)expert];
            }
            CHECK_CUDA(cudaSetDevice(opt.devices[rank]));
            CHECK_CUDA(cudaMalloc(&out->d_router_w_shard[L][rank],
                                  router_w_shard.size() * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(out->d_router_w_shard[L][rank], router_w_shard.data(),
                                  router_w_shard.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    }
    std::printf("tp_ep_mtp_hc_layer43_load\tlayer\t43\trouter_bias\t%d\trank_dist\t%d\tPASS\n",
                have_bias ? 1 : 0, opt.tp_hc_current_allreduce_gate ? 1 : 0);
    std::fflush(stdout);
    return 0;
}

void close_shared_hc_controls(const Options &opt, SharedHcControls *out) {
    if (!out || !out->initialized) return;
    close_route_plan_host_workspace(&out->route_plan_ws);
    CHECK_CUDA(cudaSetDevice(opt.devices[0]));
    if (out->router_blas) {
        cublasStatus_t st = cublasDestroy(out->router_blas);
        if (st != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "router cublasDestroy failed status=%d\n", (int)st);
        }
    }
    for (int layer = 0; layer < 43; ++layer) {
        for (int rank = 1; rank < kGpus; ++rank) {
            if (out->d_attn_norm_weight_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_attn_norm_weight_rank[layer][rank]));
            }
            if (out->d_ffn_norm_weight_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_ffn_norm_weight_rank[layer][rank]));
            }
        }
        for (int rank = 0; rank < kGpus; ++rank) {
            if (out->d_attn_fn_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_attn_fn_rank[layer][rank]));
            }
            if (out->d_attn_base_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_attn_base_rank[layer][rank]));
            }
            if (out->d_attn_scale_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_attn_scale_rank[layer][rank]));
            }
            if (out->d_ffn_fn_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_ffn_fn_rank[layer][rank]));
            }
            if (out->d_ffn_base_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_ffn_base_rank[layer][rank]));
            }
            if (out->d_ffn_scale_rank[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_ffn_scale_rank[layer][rank]));
            }
        }
        if (out->d_router_hash[layer]) CHECK_CUDA(cudaFree(out->d_router_hash[layer]));
        if (out->d_router_bias[layer]) CHECK_CUDA(cudaFree(out->d_router_bias[layer]));
        for (int rank = 0; rank < kGpus; ++rank) {
            if (out->d_router_w_ep[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_router_w_ep[layer][rank]));
            }
            if (out->d_router_w_shard[layer][rank]) {
                CHECK_CUDA(cudaSetDevice(out->devices[rank]));
                CHECK_CUDA(cudaFree(out->d_router_w_shard[layer][rank]));
            }
        }
        CHECK_CUDA(cudaSetDevice(out->devices[0]));
        if (out->d_router_w[layer]) CHECK_CUDA(cudaFree(out->d_router_w[layer]));
        if (out->d_indexer_compress_norm[layer]) CHECK_CUDA(cudaFree(out->d_indexer_compress_norm[layer]));
        if (out->d_indexer_compress_ape[layer]) CHECK_CUDA(cudaFree(out->d_indexer_compress_ape[layer]));
        if (out->d_attn_compress_norm[layer]) CHECK_CUDA(cudaFree(out->d_attn_compress_norm[layer]));
        if (out->d_attn_compress_ape[layer]) CHECK_CUDA(cudaFree(out->d_attn_compress_ape[layer]));
        if (out->d_attn_sinks[layer]) CHECK_CUDA(cudaFree(out->d_attn_sinks[layer]));
        if (out->d_kv_a_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_kv_a_norm_weight[layer]));
        if (out->d_q_a_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_q_a_norm_weight[layer]));
        if (out->d_attn_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_attn_norm_weight[layer]));
        if (out->d_ffn_norm_weight[layer]) CHECK_CUDA(cudaFree(out->d_ffn_norm_weight[layer]));
        if (out->d_ffn_scale[layer]) CHECK_CUDA(cudaFree(out->d_ffn_scale[layer]));
        if (out->d_ffn_base[layer]) CHECK_CUDA(cudaFree(out->d_ffn_base[layer]));
        if (out->d_ffn_fn[layer]) CHECK_CUDA(cudaFree(out->d_ffn_fn[layer]));
        if (out->d_attn_scale[layer]) CHECK_CUDA(cudaFree(out->d_attn_scale[layer]));
        if (out->d_attn_base[layer]) CHECK_CUDA(cudaFree(out->d_attn_base[layer]));
        if (out->d_attn_fn[layer]) CHECK_CUDA(cudaFree(out->d_attn_fn[layer]));
    }
    if (out->d_router_weights) CHECK_CUDA(cudaFree(out->d_router_weights));
    if (out->d_router_selected) CHECK_CUDA(cudaFree(out->d_router_selected));
    if (out->d_router_logits) CHECK_CUDA(cudaFree(out->d_router_logits));
    if (out->d_router_tokens) CHECK_CUDA(cudaFree(out->d_router_tokens));
    if (out->d_router_active) CHECK_CUDA(cudaFree(out->d_router_active));
    if (out->d_index_comp_score_full) CHECK_CUDA(cudaFree(out->d_index_comp_score_full));
    if (out->d_index_comp_kv_full) CHECK_CUDA(cudaFree(out->d_index_comp_kv_full));
    if (out->d_indexer_w_full) CHECK_CUDA(cudaFree(out->d_indexer_w_full));
    if (out->d_indexer_q_full) CHECK_CUDA(cudaFree(out->d_indexer_q_full));
    if (out->d_attn_comp_score_full) CHECK_CUDA(cudaFree(out->d_attn_comp_score_full));
    if (out->d_attn_comp_kv_full) CHECK_CUDA(cudaFree(out->d_attn_comp_kv_full));
    if (out->d_ffn_normed) CHECK_CUDA(cudaFree(out->d_ffn_normed));
    if (out->d_kv_normed) CHECK_CUDA(cudaFree(out->d_kv_normed));
    if (out->d_kv_full) CHECK_CUDA(cudaFree(out->d_kv_full));
    if (out->d_q_a_normed) CHECK_CUDA(cudaFree(out->d_q_a_normed));
    if (out->d_q_a_full) CHECK_CUDA(cudaFree(out->d_q_a_full));
    if (out->d_attn_normed) CHECK_CUDA(cudaFree(out->d_attn_normed));
    if (out->d_current_full) CHECK_CUDA(cudaFree(out->d_current_full));
    if (out->d_split) CHECK_CUDA(cudaFree(out->d_split));
    if (out->d_mix) CHECK_CUDA(cudaFree(out->d_mix));
    if (out->d_hc_norm) CHECK_CUDA(cudaFree(out->d_hc_norm));
    if (out->d_hc) CHECK_CUDA(cudaFree(out->d_hc));
    *out = SharedHcControls{};
}

int upload_model_router_route_plan(const Options &opt,
                                   RankState ranks[kGpus],
                                   const std::vector<int> &selected,
                                   const std::vector<float> &weights);
int upload_model_router_route_plan_async(const Options &opt,
                                         RankState ranks[kGpus],
                                         const int *selected,
                                         const float *weights,
                                         RoutePlanHostWorkspace *ws);
int upload_model_router_route_plan_gpu(const Options &opt,
                                       SharedHcControls *hc,
                                       RankState ranks[kGpus]);
int enqueue_dense_wait_after_rank_stream(RankState ranks[kGpus]);
int enqueue_control_wait_after_rank_streams(const Options &opt,
                                            RankState ranks[kGpus],
                                            cudaStream_t control_stream);
int enqueue_control_wait_after_dense_streams(const Options &opt,
                                             RankState ranks[kGpus],
                                             cudaStream_t control_stream);

int next_graph_order_event_slot(RankState ranks[kGpus]) {
    const int slot = ranks[0].graph_event_cursor % kGraphOrderEventSlots;
    ranks[0].graph_event_cursor =
        (ranks[0].graph_event_cursor + 1) % kGraphOrderEventSlots;
    return slot;
}

cudaEvent_t graph_stream_done_event(RankState &r, int slot) {
    cudaEvent_t ev = r.graph_stream_done[slot % kGraphOrderEventSlots];
    return ev ? ev : r.stream_done;
}

cudaEvent_t graph_dense_done_event(RankState &r, int slot) {
    cudaEvent_t ev = r.graph_dense_done[slot % kGraphOrderEventSlots];
    return ev ? ev : r.dense_done;
}

/* ------------------------------------------------------------------------- */
/* Sprint 600: flag-gated root-cause probes (default off; with all probe     */
/* env vars unset no kernels are enqueued and no device state is allocated,  */
/* so the flag-off captured graph is byte-identical).                        */
/* ------------------------------------------------------------------------- */
enum S600DelaySite {
    kS600XchgTail = 0,
    kS600PreDown,
    kS600PostDown,
    kS600PrePack,
    kS600PreReturn,
    kS600PostReturn,
    kS600PostCompose,
    kS600DelaySiteCount
};
static const char *const kS600DelaySiteNames[kS600DelaySiteCount] = {
    "xchg_tail", "pre_down", "post_down", "pre_pack",
    "pre_return", "post_return", "post_compose"};

/* V100 SM clock ~1.5 GHz; approximate us->cycles is fine for injection. */
constexpr unsigned long long kS600CyclesPerUs = 1500ull;

struct S600ProbeState {
    bool initialized = false;
    bool any_delay = false;
    bool verify = false;
    bool return_verify = false;
    bool jitter = false;
    unsigned int jitter_min_us = 0;
    unsigned int jitter_max_us = 0;
    unsigned int jitter_state = 0;
    bool site_enabled[kS600DelaySiteCount] = {};
    unsigned long long site_us[kS600DelaySiteCount] = {};
    bool ag_verify = false;
    unsigned long long *d_delay[kGpus] = {};    /* kS600DelaySiteCount slots */
    unsigned long long *d_mismatch[kGpus] = {}; /* 4: count, src, idx, bits */
    unsigned long long *d_mismatch_ret[kGpus] = {}; /* 4: same, EP return */
    unsigned long long *d_mismatch_ag[kGpus] = {};  /* 4: same, post-attn AG */
};
static S600ProbeState g_s600;

/* Busy-wait whose duration is read from device memory at execution time, so
 * the host can retune/jitter it between graph replays without re-capture. */
__global__ void s600_delay_kernel(const unsigned long long *cycles_slot) {
    const unsigned long long target = *cycles_slot;
    if (!target) return;
    const long long start = clock64();
    while ((unsigned long long)(clock64() - start) < target) {
    }
}

/* Bit-compare the locally exchanged swiglu segment against a fresh remote
 * read of the source rank's segment. A nonzero mismatch count proves the
 * exchange consumed data the producer had not yet written (stale read). */
__global__ void s600_verify_seg_kernel(const float *dst, const float *src,
                                       uint32_t rows, uint32_t slots,
                                       uint32_t seg_off, int src_rank,
                                       unsigned long long *out) {
    const uint64_t n = (uint64_t)slots * rows;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)blockDim.x * gridDim.x) {
        const uint32_t slot = (uint32_t)(i / rows);
        const uint32_t r = (uint32_t)(i % rows);
        const uint64_t o = (uint64_t)slot * kMid + seg_off + r;
        const unsigned int a = __float_as_uint(dst[o]);
        const unsigned int b = __float_as_uint(src[o]);
        if (a != b) {
            if (atomicAdd(out, 1ull) == 0ull) {
                out[1] = (unsigned long long)src_rank;
                out[2] = o;
                out[3] = (unsigned long long)a | ((unsigned long long)b << 32);
            }
        }
    }
}

/* Flat bit-compare of two float arrays (used for the EP-return slices). */
__global__ void s600_verify_flat_kernel(const float *a_buf, const float *b_buf,
                                        uint64_t elems, int src_rank,
                                        unsigned long long *out) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < elems; i += (uint64_t)blockDim.x * gridDim.x) {
        const unsigned int a = __float_as_uint(a_buf[i]);
        const unsigned int b = __float_as_uint(b_buf[i]);
        if (a != b) {
            if (atomicAdd(out, 1ull) == 0ull) {
                out[1] = (unsigned long long)src_rank;
                out[2] = i;
                out[3] = (unsigned long long)a | ((unsigned long long)b << 32);
            }
        }
    }
}

int s600_probe_init(const Options &opt, RankState ranks[kGpus]) {
    if (g_s600.initialized) return 0;
    g_s600.initialized = true;
    if (!opt.s600_delay_spec && !opt.s600_swiglu_verify &&
        !opt.s600_return_verify && !opt.s600_ag_verify) {
        return 0;
    }
    if (opt.s600_delay_spec) {
        const char *p = opt.s600_delay_spec;
        while (p && *p) {
            while (*p == ',' || *p == ' ') ++p;
            if (!*p) break;
            const char *colon = std::strchr(p, ':');
            if (!colon) {
                std::fprintf(stderr,
                             "s600_delay_spec parse error near '%s'\n", p);
                return 1;
            }
            const size_t name_len = (size_t)(colon - p);
            int site = -1;
            for (int s = 0; s < kS600DelaySiteCount; ++s) {
                if (std::strlen(kS600DelaySiteNames[s]) == name_len &&
                    std::strncmp(kS600DelaySiteNames[s], p, name_len) == 0) {
                    site = s;
                    break;
                }
            }
            if (site < 0) {
                std::fprintf(stderr,
                             "s600_delay_spec unknown site near '%s'\n", p);
                return 1;
            }
            g_s600.site_enabled[site] = true;
            g_s600.site_us[site] = std::strtoull(colon + 1, nullptr, 10);
            g_s600.any_delay = true;
            p = std::strchr(colon + 1, ',');
        }
    }
    if (opt.s600_jitter_spec) {
        unsigned int mn = 0;
        unsigned int mx = 0;
        unsigned int seed = 1;
        if (std::sscanf(opt.s600_jitter_spec, "%u:%u:%u", &mn, &mx, &seed) < 2 ||
            mx < mn || !g_s600.any_delay) {
            std::fprintf(stderr,
                         "s600_jitter_spec invalid (need min:max[:seed] and "
                         "at least one delay site): %s\n",
                         opt.s600_jitter_spec);
            return 1;
        }
        g_s600.jitter = true;
        g_s600.jitter_min_us = mn;
        g_s600.jitter_max_us = mx;
        g_s600.jitter_state = seed ? seed : 1u;
    }
    g_s600.verify = opt.s600_swiglu_verify;
    g_s600.return_verify = opt.s600_return_verify;
    g_s600.ag_verify = opt.s600_ag_verify;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        if (g_s600.any_delay) {
            unsigned long long h[kS600DelaySiteCount] = {};
            for (int s = 0; s < kS600DelaySiteCount; ++s) {
                h[s] = g_s600.site_us[s] * kS600CyclesPerUs;
            }
            CHECK_CUDA(cudaMalloc(&g_s600.d_delay[rank], sizeof(h)));
            CHECK_CUDA(cudaMemcpy(g_s600.d_delay[rank], h, sizeof(h),
                                  cudaMemcpyHostToDevice));
        }
        if (g_s600.verify) {
            CHECK_CUDA(cudaMalloc(&g_s600.d_mismatch[rank],
                                  4 * sizeof(unsigned long long)));
            CHECK_CUDA(cudaMemset(g_s600.d_mismatch[rank], 0,
                                  4 * sizeof(unsigned long long)));
        }
        if (g_s600.return_verify) {
            CHECK_CUDA(cudaMalloc(&g_s600.d_mismatch_ret[rank],
                                  4 * sizeof(unsigned long long)));
            CHECK_CUDA(cudaMemset(g_s600.d_mismatch_ret[rank], 0,
                                  4 * sizeof(unsigned long long)));
        }
        if (g_s600.ag_verify) {
            CHECK_CUDA(cudaMalloc(&g_s600.d_mismatch_ag[rank],
                                  4 * sizeof(unsigned long long)));
            CHECK_CUDA(cudaMemset(g_s600.d_mismatch_ag[rank], 0,
                                  4 * sizeof(unsigned long long)));
        }
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
    std::printf("tp_ep_s600_probe_init\tdelay\t%s\tjitter\t%s\tverify\t%d\t"
                "return_verify\t%d\tag_verify\t%d\n",
                opt.s600_delay_spec ? opt.s600_delay_spec : "-",
                opt.s600_jitter_spec ? opt.s600_jitter_spec : "-",
                g_s600.verify ? 1 : 0, g_s600.return_verify ? 1 : 0,
                g_s600.ag_verify ? 1 : 0);
    std::fflush(stdout);
    return 0;
}

/* Verify the EP-return slices against a fresh remote read of the source
 * rank's contrib slice. Runs on the DENSE streams (idle during the return/
 * compose window) behind a rank->dense event edge, so the rank-stream
 * critical path is not lengthened (low-masking detection: the s600 rca5v
 * run proved rank-stream verifiers re-mask the race like the copy storm). */
void s600_return_verify_enqueue(RankState ranks[kGpus],
                                uint64_t src_stride_elems,
                                const uint64_t copy_elems_by_src[kGpus],
                                bool skip_self_copy) {
    if (!g_s600.return_verify) return;
    const int block = 256;
    const int slot = next_graph_order_event_slot(ranks);
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int dst = 0; dst < kGpus; ++dst) {
        RankState &r = ranks[dst];
        CHECK_CUDA(cudaSetDevice(r.device));
        cudaStream_t vstream = r.stream;
        if (r.dense_stream && r.dense_stream != r.stream) {
            cudaEvent_t ev = graph_stream_done_event(r, slot);
            if (ev) {
                CHECK_CUDA(cudaEventRecord(ev, r.stream));
                CHECK_CUDA(cudaStreamWaitEvent(r.dense_stream, ev, 0));
                vstream = r.dense_stream;
            }
        }
        for (int src = 0; src < kGpus; ++src) {
            if (skip_self_copy && src == dst) continue;
            const uint64_t copy_elems = copy_elems_by_src[src];
            if (!copy_elems || !r.d_ep_remote[src]) continue;
            const float *truth = ranks[src].d_ep_contrib_all +
                                 (uint64_t)dst * src_stride_elems;
            const unsigned int grid = (unsigned int)(
                (copy_elems + (uint64_t)block - 1) / (uint64_t)block);
            s600_verify_flat_kernel<<<grid, block, 0, vstream>>>(
                r.d_ep_remote[src], truth, copy_elems, src,
                g_s600.d_mismatch_ret[dst]);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
}

/* Late-verify the post-attention allgather: each rank-major segment src of
 * d_post_attn_full_rank_major must equal the source rank's d_post_attn_shard
 * (both stable until the next layer replay). Caller hooks this AFTER the 978
 * barrier on the dense streams, which are idle from there to the end of the
 * layer, so detection runs under the pack/return/compose shadow. */
void s600_ag_verify_enqueue(const Options &opt, RankState ranks[kGpus]) {
    if (!g_s600.ag_verify) return;
    const int block = 256;
    const uint64_t shard_elems =
        (uint64_t)opt.slots * (uint64_t)(kHidden / kGpus);
    const unsigned int grid = (unsigned int)(
        (shard_elems + (uint64_t)block - 1) / (uint64_t)block);
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int dst = 0; dst < kGpus; ++dst) {
        RankState &r = ranks[dst];
        if (!r.d_post_attn_full_rank_major) continue;
        CHECK_CUDA(cudaSetDevice(r.device));
        cudaStream_t vstream =
            (r.dense_stream && r.dense_stream != r.stream) ? r.dense_stream
                                                           : r.stream;
        for (int src = 0; src < kGpus; ++src) {
            if (src == dst) continue;
            if (!ranks[src].d_post_attn_shard) continue;
            s600_verify_flat_kernel<<<grid, block, 0, vstream>>>(
                r.d_post_attn_full_rank_major + (uint64_t)src * shard_elems,
                ranks[src].d_post_attn_shard, shard_elems, src,
                g_s600.d_mismatch_ag[dst]);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
}

void s600_delay_enqueue(RankState ranks[kGpus], int site, bool on_dense) {
    if (!g_s600.any_delay || !g_s600.site_enabled[site]) return;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        cudaStream_t stream =
            (on_dense && r.dense_stream) ? r.dense_stream : r.stream;
        s600_delay_kernel<<<1, 1, 0, stream>>>(g_s600.d_delay[rank] + site);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
}

/* Host-side, between replays only (streams idle at the call site). */
void s600_jitter_refresh(RankState ranks[kGpus]) {
    if (!g_s600.jitter || !g_s600.any_delay) return;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int rank = 0; rank < kGpus; ++rank) {
        unsigned long long h[kS600DelaySiteCount] = {};
        for (int s = 0; s < kS600DelaySiteCount; ++s) {
            if (!g_s600.site_enabled[s]) continue;
            unsigned int x = g_s600.jitter_state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            g_s600.jitter_state = x;
            const unsigned int span =
                g_s600.jitter_max_us - g_s600.jitter_min_us + 1u;
            h[s] = (g_s600.jitter_min_us + (x % span)) * kS600CyclesPerUs;
        }
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaMemcpy(g_s600.d_delay[rank], h, sizeof(h),
                              cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
}

/* ------------------------------------------------------------------------- */
/* Sprint 604 Phase A: rank<->dense hazard amplifier + Phase C fix edge.      */
/* Default off; with both envs unset no kernels/events are enqueued and no    */
/* device state is allocated, so the flag-off captured graph is byte-ident.   */
/* ------------------------------------------------------------------------- */
enum S604AmpSite {
    kS604PreDense = 0,   /* dense stream, before the EP/dense GEMMs */
    kS604PostDense,      /* dense stream, after the GEMMs (pre-954) */
    kS604PreDown,        /* dense stream, before shared-down GEMM */
    kS604PostDown,       /* dense stream, after shared-down (pre-978) */
    kS604PreCompose,     /* dense stream, right before compose */
    kS604AttnOutA,       /* dense stream, before the attn-output-A GEMM
                          * (widens the cross-rank dense->rank gap at
                          * attention_output.cu:48->87 - codex candidate 1) */
    kS604AttnOutB,       /* dense stream, before the attn-output-B GEMM */
    kS604AmpSiteCount
};
static const char *const kS604AmpSiteNames[kS604AmpSiteCount] = {
    "pre_dense", "post_dense", "pre_down", "post_down", "pre_compose",
    "attn_out_a", "attn_out_b"};

struct S604AmpState {
    bool initialized = false;
    bool any = false;
    bool site_enabled[kS604AmpSiteCount] = {};
    unsigned long long cycles = 0;          /* amp duration, same on all sites */
    unsigned long long *d_cycles[kGpus] = {}; /* one slot, device-read by kernel */
};
static S604AmpState g_s604;

int s604_amp_init(const Options &opt, RankState ranks[kGpus]) {
    if (g_s604.initialized) return 0;
    g_s604.initialized = true;
    if (opt.dense_hazard_amp_us <= 0) return 0;
    g_s604.cycles = (unsigned long long)opt.dense_hazard_amp_us * kS600CyclesPerUs;
    if (opt.dense_hazard_amp_site && *opt.dense_hazard_amp_site) {
        const char *p = opt.dense_hazard_amp_site;
        while (p && *p) {
            while (*p == ',' || *p == ' ') ++p;
            if (!*p) break;
            const char *end = p;
            while (*end && *end != ',' && *end != ' ') ++end;
            const size_t len = (size_t)(end - p);
            int site = -1;
            for (int s = 0; s < kS604AmpSiteCount; ++s) {
                if (std::strlen(kS604AmpSiteNames[s]) == len &&
                    std::strncmp(kS604AmpSiteNames[s], p, len) == 0) {
                    site = s;
                    break;
                }
            }
            if (site < 0) {
                std::fprintf(stderr,
                             "s604_amp_site unknown site near '%s'\n", p);
                return 1;
            }
            g_s604.site_enabled[site] = true;
            p = end;
        }
    } else {
        for (int s = 0; s < kS604AmpSiteCount; ++s) g_s604.site_enabled[s] = true;
    }
    g_s604.any = true;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaMalloc(&g_s604.d_cycles[rank], sizeof(unsigned long long)));
        CHECK_CUDA(cudaMemcpy(g_s604.d_cycles[rank], &g_s604.cycles,
                              sizeof(unsigned long long), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
    char sites[128] = {0};
    for (int s = 0; s < kS604AmpSiteCount; ++s) {
        if (!g_s604.site_enabled[s]) continue;
        std::strncat(sites, kS604AmpSiteNames[s], sizeof(sites) - std::strlen(sites) - 2);
        std::strncat(sites, ",", sizeof(sites) - std::strlen(sites) - 1);
    }
    std::printf("tp_ep_s604_amp_init\tus\t%d\tsites\t%s\tPASS\n",
                opt.dense_hazard_amp_us, sites[0] ? sites : "-");
    std::fflush(stdout);
    return 0;
}

/* Busy-wait on the DENSE stream (delays the dense producers, letting rank
 * consumers race ahead). on_dense=false targets the rank stream instead. */
void s604_amp_enqueue(RankState ranks[kGpus], int site, bool on_dense) {
    if (!g_s604.any || site < 0 || site >= kS604AmpSiteCount ||
        !g_s604.site_enabled[site]) {
        return;
    }
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        cudaStream_t stream =
            (on_dense && r.dense_stream) ? r.dense_stream : r.stream;
        s600_delay_kernel<<<1, 1, 0, stream>>>(g_s604.d_cycles[rank]);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
}

/* Sprint 604 Phase C: minimal CROSS-GPU dense<->rank ordering edge. The
 * rank-stream-only join (s602_rank_join / enqueue_rank_streams_wait_after_
 * dense_streams) orders only same-GPU dense<->rank or all-GPU rank<->rank;
 * it leaves a peer GPU's dense producer/consumer unordered against this
 * rank's stream. This records BOTH the rank and dense completion of every
 * GPU, then makes each rank stream AND each dense stream wait every peer's
 * rank and dense events - the dense involvement of the fb barrier, without
 * the redundant rank<->rank 8x8 join the default already supplies. Graph-
 * capturable, pre-allocated event slots. Flag-gated (opt.dense_fix). */
static int s604_dense_rank_edge(RankState ranks[kGpus]) {
    const int slot = next_graph_order_event_slot(ranks);
    for (int g = 0; g < kGpus; ++g) {
        RankState &r = ranks[g];
        CHECK_CUDA(cudaSetDevice(r.device));
        cudaEvent_t sev = graph_stream_done_event(r, slot);
        cudaEvent_t dev = graph_dense_done_event(r, slot);
        if (!sev || !dev) return 1;
        CHECK_CUDA(cudaEventRecord(sev, r.stream));
        CHECK_CUDA(cudaEventRecord(dev,
                                   r.dense_stream ? r.dense_stream : r.stream));
    }
    for (int g = 0; g < kGpus; ++g) {
        RankState &r = ranks[g];
        CHECK_CUDA(cudaSetDevice(r.device));
        for (int p = 0; p < kGpus; ++p) {
            /* rank stream waits every peer's dense completion (the missing
             * cross-GPU dense->rank edge). */
            CHECK_CUDA(cudaStreamWaitEvent(
                r.stream, graph_dense_done_event(ranks[p], slot), 0));
            /* dense stream waits every peer's rank completion (the reverse
             * edge: a peer's rank producer before this dense consumer). */
            if (r.dense_stream && r.dense_stream != r.stream) {
                CHECK_CUDA(cudaStreamWaitEvent(
                    r.dense_stream, graph_stream_done_event(ranks[p], slot), 0));
            }
        }
    }
    return 0;
}

int s604_dense_fix_enqueue(const Options &opt, RankState ranks[kGpus]) {
    if (opt.dense_fix <= 0) return 0;
    return s604_dense_rank_edge(ranks);
}

void s600_swiglu_verify_enqueue(const Options &opt, RankState ranks[kGpus],
                                const ResidentF8Dense &down, uint32_t rows) {
    if (!g_s600.verify) return;
    const int block = 256;
    const uint64_t seg_elems = (uint64_t)opt.slots * rows;
    const unsigned int grid =
        (unsigned int)((seg_elems + (uint64_t)block - 1) / (uint64_t)block);
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int dst = 0; dst < kGpus; ++dst) {
        RankState &r = ranks[dst];
        CHECK_CUDA(cudaSetDevice(r.device));
        for (int src = 0; src < kGpus; ++src) {
            if (src == dst) continue;
            s600_verify_seg_kernel<<<grid, block, 0, r.stream>>>(
                down.d_x[(size_t)dst], down.d_x[(size_t)src], rows,
                (uint32_t)opt.slots, (uint32_t)src * rows, src,
                g_s600.d_mismatch[dst]);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
}

/* Host-side, after a fully synced replay. Prints + resets on mismatch. */
void s600_collect_verify(int layer, int step, RankState ranks[kGpus]) {
    if (!g_s600.verify && !g_s600.return_verify && !g_s600.ag_verify) return;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        if (g_s600.verify) {
            unsigned long long h[4] = {};
            CHECK_CUDA(cudaMemcpy(h, g_s600.d_mismatch[rank], sizeof(h),
                                  cudaMemcpyDeviceToHost));
            if (h[0]) {
                std::printf("tp_ep_s600_swiglu_verify_mismatch\tstep\t%d\t"
                            "layer\t%d\tdst\t%d\tcount\t%llu\tsrc\t%llu\t"
                            "elem\t%llu\tdst_bits\t%08llx\tsrc_bits\t%08llx\n",
                            step, layer, rank, h[0], h[1], h[2],
                            h[3] & 0xffffffffull, h[3] >> 32);
                std::fflush(stdout);
                CHECK_CUDA(cudaMemset(g_s600.d_mismatch[rank], 0, sizeof(h)));
            }
        }
        if (g_s600.return_verify) {
            unsigned long long h[4] = {};
            CHECK_CUDA(cudaMemcpy(h, g_s600.d_mismatch_ret[rank], sizeof(h),
                                  cudaMemcpyDeviceToHost));
            if (h[0]) {
                std::printf("tp_ep_s600_return_verify_mismatch\tstep\t%d\t"
                            "layer\t%d\tdst\t%d\tcount\t%llu\tsrc\t%llu\t"
                            "elem\t%llu\tdst_bits\t%08llx\tsrc_bits\t%08llx\n",
                            step, layer, rank, h[0], h[1], h[2],
                            h[3] & 0xffffffffull, h[3] >> 32);
                std::fflush(stdout);
                CHECK_CUDA(cudaMemset(g_s600.d_mismatch_ret[rank], 0,
                                      sizeof(h)));
            }
        }
        if (g_s600.ag_verify) {
            unsigned long long h[4] = {};
            CHECK_CUDA(cudaMemcpy(h, g_s600.d_mismatch_ag[rank], sizeof(h),
                                  cudaMemcpyDeviceToHost));
            if (h[0]) {
                std::printf("tp_ep_s600_ag_verify_mismatch\tstep\t%d\t"
                            "layer\t%d\tdst\t%d\tcount\t%llu\tsrc\t%llu\t"
                            "elem\t%llu\tdst_bits\t%08llx\tsrc_bits\t%08llx\n",
                            step, layer, rank, h[0], h[1], h[2],
                            h[3] & 0xffffffffull, h[3] >> 32);
                std::fflush(stdout);
                CHECK_CUDA(cudaMemset(g_s600.d_mismatch_ag[rank], 0,
                                      sizeof(h)));
            }
        }
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
}

/* ------------------------------------------------------------------------- */
/* Sprint 599 C-A: shared swiglu_down input exchange via one grouped         */
/* ncclAllGather instead of the per-(dst,src,slot) UVA remote-load copies    */
/* (8 x 7 x slots tiny copy_f32 kernels per layer, 24/56 pairs crossing      */
/* SYS). Layout:                                                             */
/*   1. per-rank local swiglu writes the rank's strided segment of d_x       */
/*      (same kernel as materialize_shared_swiglu_down_input);               */
/*   2. one local 2D memcpy packs the segment contiguous;                    */
/*   3. one grouped ncclAllGather (seg = slots x rows floats per rank);      */
/*   4. seven local 2D memcpys unpack the other ranks' segments strided.     */
/* Scratch reuses d_ep_contrib_bcast_all (recv at offset 0: 8*seg; send at   */
/* offset 8*seg) -- same-stream ordering vs the EP-return broadcasts keeps   */
/* the reuse safe. Capture-safe: stream ops only; ends with the same         */
/* dense-waits-rank edge materialize ends with.                              */
/* ------------------------------------------------------------------------- */
int swiglu_down_exchange_nccl(const Options &opt,
                              RankState ranks[kGpus],
                              const ResidentF8Dense &gate,
                              const ResidentF8Dense &up,
                              const ResidentF8Dense &down) {
    if (gate.rows_per_gpu != kMid / kGpus ||
        up.rows_per_gpu != kMid / kGpus ||
        down.cols != kMid) {
        return 1;
    }
    const int block = 256;
    const uint32_t rows = (uint32_t)gate.rows_per_gpu;
    const uint32_t slots = (uint32_t)opt.slots;
    const uint64_t seg_elems = (uint64_t)slots * rows;
    const size_t seg_bytes = (size_t)seg_elems * sizeof(float);
    const uint64_t scratch_need = 9ull * seg_elems;
    const uint64_t scratch_have = (uint64_t)kGpus *
        (opt.compact_moe_decode_gate
             ? (uint64_t)opt.slots * (uint64_t)opt.top_k
             : (uint64_t)opt.slots) *
        (uint64_t)(kHidden / kGpus);
    if (scratch_need > scratch_have) return 2;
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!r.compose_nccl_initialized || !r.compose_nccl ||
            !r.d_ep_contrib_bcast_all ||
            !down.d_x[(size_t)rank] ||
            !gate.d_out[(size_t)rank] || !up.d_out[(size_t)rank]) {
            return 3;
        }
    }
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    /* 1. local swiglu into the rank's own strided segment + 2. pack. */
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        shared_swiglu_shard_to_float_kernel<<<
            (unsigned int)((seg_elems + block - 1) / block), block, 0,
            r.stream>>>(down.d_x[(size_t)rank], gate.d_out[(size_t)rank],
                        up.d_out[(size_t)rank], (uint32_t)rank, rows, slots,
                        kRoutedSwigluClamp);
        CHECK_CUDA(cudaGetLastError());
        float *send = r.d_ep_contrib_bcast_all + 8ull * seg_elems;
        CHECK_CUDA(cudaMemcpy2DAsync(
            send, (size_t)rows * sizeof(float),
            down.d_x[(size_t)rank] + (size_t)rank * rows,
            (size_t)kMid * sizeof(float), (size_t)rows * sizeof(float),
            slots, cudaMemcpyDeviceToDevice, r.stream));
    }
    /* 3. grouped allgather: recv = bcast scratch [8 * seg]. (Variant note:
     * a per-src grouped-broadcast variant diverged MORE under contention --
     * the divergence scales with the number of small captured collectives,
     * pointing at NCCL LL-protocol staleness across graph replays; the
     * single-allgather form plus NCCL_PROTO=Simple is the validated combo.) */
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        const float *send = r.d_ep_contrib_bcast_all + 8ull * seg_elems;
        CHECK_NCCL(ncclAllGather(send, r.d_ep_contrib_bcast_all,
                                 (size_t)seg_elems, ncclFloat,
                                 ds4_comm_epret(r), r.stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    /* 4. unpack the other ranks' segments strided into d_x. */
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        CHECK_CUDA(cudaSetDevice(r.device));
        for (int src = 0; src < kGpus; ++src) {
            if (src == rank) continue;
            CHECK_CUDA(cudaMemcpy2DAsync(
                down.d_x[(size_t)rank] + (size_t)src * rows,
                (size_t)kMid * sizeof(float),
                r.d_ep_contrib_bcast_all + (size_t)src * seg_elems,
                (size_t)rows * sizeof(float), (size_t)rows * sizeof(float),
                slots, cudaMemcpyDeviceToDevice, r.stream));
        }
    }
    (void)seg_bytes;
    /* s599 C-A2: cross-GPU barrier after the unpack. The allgather syncs the
     * exchange itself, but the rest of the layer (dense down GEMM, compose,
     * and the later EP-return broadcasts that reuse this scratch) ran under
     * materialize's stronger two-sync structure; restore equivalent ordering
     * before handing d_x to the dense stream. */
    if (enqueue_cross_gpu_stream_barrier(ranks, false) != 0) {
        CHECK_CUDA(cudaSetDevice(prior_device));
        return 5;
    }
    s600_swiglu_verify_enqueue(opt, ranks, down, (uint32_t)gate.rows_per_gpu);
    s600_delay_enqueue(ranks, kS600XchgTail, false);
    if (enqueue_dense_wait_after_rank_stream(ranks) != 0) {
        CHECK_CUDA(cudaSetDevice(prior_device));
        return 4;
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
    return 0;
}

/* s599 C-A4: swiglu_down input exchange via direct strided P2P 2D memcpys.
 * Replaces materialize's 8 x 7 x slots per-slot copy_f32 remote-load kernels
 * with one 2D DMA per (dst,src) pair (rows x slots strided). Pure copies of
 * the same bytes -- bit-exact vs the promoted path by construction; the
 * cross-rank ordering mirrors materialize (barrier after the local swiglu
 * kernels, dense-waits-rank at the end). */
int swiglu_down_exchange_memcpy2d(const Options &opt,
                                  RankState ranks[kGpus],
                                  const ResidentF8Dense &gate,
                                  const ResidentF8Dense &up,
                                  const ResidentF8Dense &down) {
    if (gate.rows_per_gpu != kMid / kGpus ||
        up.rows_per_gpu != kMid / kGpus ||
        down.cols != kMid) {
        return 1;
    }
    const int block = 256;
    const uint32_t rows = (uint32_t)gate.rows_per_gpu;
    const uint32_t slots = (uint32_t)opt.slots;
    const uint64_t seg_elems = (uint64_t)slots * rows;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!down.d_x[(size_t)rank] || !gate.d_out[(size_t)rank] ||
            !up.d_out[(size_t)rank]) {
            return 2;
        }
        CHECK_CUDA(cudaSetDevice(r.device));
        shared_swiglu_shard_to_float_kernel<<<
            (unsigned int)((seg_elems + block - 1) / block), block, 0,
            r.stream>>>(down.d_x[(size_t)rank], gate.d_out[(size_t)rank],
                        up.d_out[(size_t)rank], (uint32_t)rank, rows, slots,
                        kRoutedSwigluClamp);
        CHECK_CUDA(cudaGetLastError());
    }
    if (enqueue_cross_gpu_stream_barrier(ranks, false) != 0) {
        CHECK_CUDA(cudaSetDevice(prior_device));
        return 3;
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        RankState &r = ranks[dst];
        CHECK_CUDA(cudaSetDevice(r.device));
        for (int src = 0; src < kGpus; ++src) {
            if (src == dst) continue;
            CHECK_CUDA(cudaMemcpy2DAsync(
                down.d_x[(size_t)dst] + (size_t)src * rows,
                (size_t)kMid * sizeof(float),
                down.d_x[(size_t)src] + (size_t)src * rows,
                (size_t)kMid * sizeof(float),
                (size_t)rows * sizeof(float), slots,
                cudaMemcpyDeviceToDevice, r.stream));
        }
    }
    s600_swiglu_verify_enqueue(opt, ranks, down, rows);
    s600_delay_enqueue(ranks, kS600XchgTail, false);
    if (enqueue_dense_wait_after_rank_stream(ranks) != 0) {
        CHECK_CUDA(cudaSetDevice(prior_device));
        return 4;
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
    return 0;
}

/* s599 C-A5: batched swiglu_down exchange with the SAME UVA remote-load
 * kernel mechanics materialize uses (proven capture-safe on this path),
 * but one strided-gather kernel per (dst,src) pair instead of one
 * copy_f32_kernel per (dst,src,slot): 56 launches/layer instead of 1792.
 * Pure copies of the same bytes; ordering identical to materialize. */
__global__ void s599_strided_seg_copy_kernel(float *dst, const float *src,
                                             uint32_t rows, uint32_t slots,
                                             uint32_t seg_off) {
    const uint64_t n = (uint64_t)slots * rows;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)blockDim.x * gridDim.x) {
        const uint32_t slot = (uint32_t)(i / rows);
        const uint32_t r = (uint32_t)(i % rows);
        const uint64_t o = (uint64_t)slot * kMid + seg_off + r;
        dst[o] = src[o];
    }
}

int swiglu_down_exchange_batched(const Options &opt,
                                 RankState ranks[kGpus],
                                 const ResidentF8Dense &gate,
                                 const ResidentF8Dense &up,
                                 const ResidentF8Dense &down) {
    if (gate.rows_per_gpu != kMid / kGpus ||
        up.rows_per_gpu != kMid / kGpus ||
        down.cols != kMid) {
        return 1;
    }
    const int block = 256;
    const uint32_t rows = (uint32_t)gate.rows_per_gpu;
    const uint32_t slots = (uint32_t)opt.slots;
    const uint64_t seg_elems = (uint64_t)slots * rows;
    int prior_device = 0;
    CHECK_CUDA(cudaGetDevice(&prior_device));
    for (int rank = 0; rank < kGpus; ++rank) {
        RankState &r = ranks[rank];
        if (!down.d_x[(size_t)rank] || !gate.d_out[(size_t)rank] ||
            !up.d_out[(size_t)rank]) {
            return 2;
        }
        CHECK_CUDA(cudaSetDevice(r.device));
        shared_swiglu_shard_to_float_kernel<<<
            (unsigned int)((seg_elems + block - 1) / block), block, 0,
            r.stream>>>(down.d_x[(size_t)rank], gate.d_out[(size_t)rank],
                        up.d_out[(size_t)rank], (uint32_t)rank, rows, slots,
                        kRoutedSwigluClamp);
        CHECK_CUDA(cudaGetLastError());
    }
    if (enqueue_cross_gpu_stream_barrier(ranks, false) != 0) {
        CHECK_CUDA(cudaSetDevice(prior_device));
        return 3;
    }
    for (int dst = 0; dst < kGpus; ++dst) {
        RankState &r = ranks[dst];
        CHECK_CUDA(cudaSetDevice(r.device));
        for (int src = 0; src < kGpus; ++src) {
            if (src == dst) continue;
            s599_strided_seg_copy_kernel<<<
                (unsigned int)((seg_elems + block - 1) / block), block, 0,
                r.stream>>>(down.d_x[(size_t)dst], down.d_x[(size_t)src],
                            rows, slots, (uint32_t)src * rows);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    s600_swiglu_verify_enqueue(opt, ranks, down, rows);
    s600_delay_enqueue(ranks, kS600XchgTail, false);
    if (enqueue_dense_wait_after_rank_stream(ranks) != 0) {
        CHECK_CUDA(cudaSetDevice(prior_device));
        return 4;
    }
    CHECK_CUDA(cudaSetDevice(prior_device));
    return 0;
}
