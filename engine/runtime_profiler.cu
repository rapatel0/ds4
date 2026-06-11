void log_hc_current_full_rank_parity(const Options &opt,
                                     RankState ranks[kGpus],
                                     int layer,
                                     size_t elems);
int nccl_broadcast_f32_from_device0_to_current_full(
    const Options &opt,
    RankState ranks[kGpus],
    const float *src_device0,
    uint64_t elems,
    const char *label);

int enqueue_cross_gpu_stream_barrier(RankState ranks[kGpus],
                                     bool include_copy_streams);

void sync_typed_kv_boundary(const Options &opt, RankState ranks[kGpus]) {
    if (opt.decode_cudagraph_gate) {
        const int rc = enqueue_cross_gpu_stream_barrier(ranks, false);
        if (rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_typed_kv_graph_boundary_failed\trc\t%d\n",
                         rc);
            std::abort();
        }
        return;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        if (opt.true_ds4_attention_typed_kv_stream_sync_gate) {
            CHECK_CUDA(cudaStreamSynchronize(0));
        } else {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }
}

struct TensorF32Stats {
    int finite_bad = 0;
    size_t first_bad = (size_t)-1;
    float max_abs = 0.0f;
};

struct TensorF32DiffStats {
    int bad = 0;
    size_t first_bad = (size_t)-1;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
};

struct HalfInputDiffStats {
    unsigned long long compared = 0;
    unsigned long long mismatches = 0;
    int first_mismatch = -1;
    float max_abs = 0.0f;
};

TensorF32Stats collect_tensor_f32_stats(const float *ptr, size_t elems,
                                        cudaStream_t stream);
TensorF32Stats collect_raw_swa_row_stats(const float *ptr, uint32_t slots,
                                         uint32_t raw_rows, uint32_t raw_row,
                                         uint32_t head_dim,
                                         cudaStream_t stream);
TensorF32DiffStats collect_tensor_f32_diff_stats(const float *a, const float *b,
                                                 size_t elems,
                                                 cudaStream_t stream);
void merge_tensor_stats(TensorF32Stats *dst, const TensorF32Stats &src);
void log_tensor_f32_stats(const char *tag, int layer, int rank_id,
                          const float *ptr, size_t elems, cudaStream_t stream);
bool should_log_routed_semantic_stats(const Options &opt);
bool should_log_reference_hc_window(const Options &opt);

/* ------------------------------------------------------------------------- */
/* Sprint 597 Phase 2: flag-gated EP sub-stage profiler.                      */
/*                                                                           */
/* DS4_V100_TP_EP_EP_STAGE_PROFILE=1 (Options.ep_stage_profile) arms paired  */
/* markers at the EP sub-stage boundaries in engine/decode_loop.cu.          */
/*                                                                           */
/* Graph mode: CUDA rejects cudaEventElapsedTime on events recorded inside a */
/* captured graph (cudaErrorInvalidValue, verified on this driver), so the   */
/* graph-compatible equivalent of paired event records is used instead: a    */
/* 1-thread %globaltimer stamp kernel per boundary writing into a            */
/* pre-allocated per-rank device slot. Stamp kernels become graph nodes and  */
/* re-execute on every replay; the slots are read back AFTER the post-replay */
/* sync (no cudaMalloc and no D2H inside the captured region).               */
/* Eager mode: classic paired cudaEventRecord/cudaEventElapsedTime.          */
/* Flag-off: every entry point returns immediately; the promoted path is     */
/* untouched.                                                                */
/* ------------------------------------------------------------------------- */

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define DS4_EP_STAGE_PROF_NVTX 1
#else
#define DS4_EP_STAGE_PROF_NVTX 0
#endif

enum EpProfStage {
    kEpProfRoutePlanPack = 0,
    kEpProfGateUpGemm = 1,
    kEpProfDownGemm = 2,
    kEpProfDenseOverlap = 3,
    kEpProfSharedSwigluDown = 4,
    kEpProfContribPack = 5,
    kEpProfCompose = 6,
    kEpProfBarrier954 = 7,
    kEpProfBarrier978 = 8,
    kEpProfBarrier996 = 9,
    kEpProfBarrier1006 = 10,
    kEpProfBarrier1045 = 11,
    kEpProfBarrier1062 = 12,
    kEpProfBarrier1144 = 13,
    kEpProfBarrier1170 = 14,
    kEpProfBarrier1373 = 15,
    kEpProfCopySrcBase = 16, /* +src in [0,7] -> 16..23 */
    kEpProfEpReturnNccl = 24, /* s598 C1: grouped NCCL broadcast return */
    /* s599 Phase A: pre-EP prefix + final_hc decomposition */
    kEpProfHcCurrent = 25,
    kEpProfAttnProjection = 26,
    kEpProfCompressedKv = 27,
    kEpProfAttnState = 28,
    kEpProfTypedHistory = 29,
    kEpProfRawRead = 30,
    kEpProfAttnOutput = 31,
    kEpProfFinalHc = 32,
    kEpProfStageCount = 33,
};

static const char *ep_stage_prof_name(int stage) {
    switch (stage) {
    case kEpProfRoutePlanPack: return "route_plan_pack";
    case kEpProfGateUpGemm: return "gate_up_gemm";
    case kEpProfDownGemm: return "down_gemm";
    case kEpProfDenseOverlap: return "dense_overlap";
    case kEpProfSharedSwigluDown: return "shared_swiglu_down";
    case kEpProfContribPack: return "contrib_pack";
    case kEpProfCompose: return "compose";
    case kEpProfBarrier954: return "barrier_954_post_dense_launch";
    case kEpProfBarrier978: return "barrier_978_shared_down";
    case kEpProfBarrier996: return "barrier_996_pre_dense";
    case kEpProfBarrier1006: return "barrier_1006_shared_gate_up";
    case kEpProfBarrier1045: return "barrier_1045_shared_down_noov";
    case kEpProfBarrier1062: return "barrier_1062_dense";
    case kEpProfBarrier1144: return "barrier_1144_contrib_pack";
    case kEpProfBarrier1170: return "barrier_1170_nccl_rs";
    case kEpProfBarrier1373: return "barrier_1373_compose";
    case kEpProfEpReturnNccl: return "ep_return_nccl";
    case kEpProfHcCurrent: return "prefix_hc_current";
    case kEpProfAttnProjection: return "prefix_attn_projection";
    case kEpProfCompressedKv: return "prefix_compressed_kv";
    case kEpProfAttnState: return "prefix_attn_state";
    case kEpProfTypedHistory: return "prefix_typed_history";
    case kEpProfRawRead: return "prefix_raw_read";
    case kEpProfAttnOutput: return "prefix_attn_output";
    case kEpProfFinalHc: return "final_hc";
    default:
        if (stage >= kEpProfCopySrcBase &&
            stage < kEpProfCopySrcBase + kGpus) {
            static const char *copy_names[kGpus] = {
                "ep_copy_src0", "ep_copy_src1", "ep_copy_src2",
                "ep_copy_src3", "ep_copy_src4", "ep_copy_src5",
                "ep_copy_src6", "ep_copy_src7"};
            return copy_names[stage - kEpProfCopySrcBase];
        }
        return "unknown";
    }
}

__global__ void ep_stage_prof_stamp_kernel(unsigned long long *slot) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *slot = t;
}

struct EpStageProfilerState {
    static const int kMaxLayers = 44;
    /* graph mode: per-rank device stamp slots [stage][begin/end] (ns) */
    unsigned long long *d_stamps[kGpus] = {};
    bool device_ready = false;
    /* eager mode: per (layer, rank, stage) event pairs */
    cudaEvent_t ev_begin[kMaxLayers][kGpus][kEpProfStageCount] = {};
    cudaEvent_t ev_end[kMaxLayers][kGpus][kEpProfStageCount] = {};
    unsigned char armed[kMaxLayers][kGpus][kEpProfStageCount] = {};
    unsigned long long bytes[kMaxLayers][kGpus][kEpProfStageCount] = {};
    unsigned long long collect_seq = 0;
};

static EpStageProfilerState g_ep_stage_prof;

/* Pre-allocates the per-rank device stamp slots. Must be called OUTSIDE any
 * stream-capture region (decode_loop entry). Idempotent. */
static void ep_stage_prof_device_init(const Options &opt,
                                      RankState ranks[kGpus]) {
    if (!opt.ep_stage_profile) return;
    EpStageProfilerState &st = g_ep_stage_prof;
    if (st.device_ready) return;
    const size_t bytes = (size_t)kEpProfStageCount * 2u *
                         sizeof(unsigned long long);
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        CHECK_CUDA(cudaMalloc(&st.d_stamps[rank], bytes));
        CHECK_CUDA(cudaMemset(st.d_stamps[rank], 0, bytes));
    }
    st.device_ready = true;
}

static void ep_stage_prof_mark(const Options &opt, int rank, int device,
                               int stage, cudaStream_t stream, bool begin,
                               unsigned long long bytes) {
    if (!opt.ep_stage_profile) return;
    const int layer = opt.layer;
    if (layer < 0 || layer >= EpStageProfilerState::kMaxLayers) return;
    if (rank < 0 || rank >= kGpus) return;
    if (stage < 0 || stage >= kEpProfStageCount) return;
    if (!stream) return;
    EpStageProfilerState &st = g_ep_stage_prof;
    CHECK_CUDA(cudaSetDevice(device));
    if (opt.decode_cudagraph_gate) {
        if (!st.device_ready) return;
        unsigned long long *slot =
            st.d_stamps[rank] + (size_t)stage * 2u + (begin ? 0u : 1u);
        ep_stage_prof_stamp_kernel<<<1, 1, 0, stream>>>(slot);
        CHECK_CUDA(cudaGetLastError());
    } else {
        if (!st.ev_begin[layer][rank][stage]) {
            CHECK_CUDA(cudaEventCreate(&st.ev_begin[layer][rank][stage]));
            CHECK_CUDA(cudaEventCreate(&st.ev_end[layer][rank][stage]));
        }
        CHECK_CUDA(cudaEventRecord(begin ? st.ev_begin[layer][rank][stage]
                                         : st.ev_end[layer][rank][stage],
                                   stream));
    }
    if (begin) {
#if DS4_EP_STAGE_PROF_NVTX
        if (rank == 0) nvtxRangePushA(ep_stage_prof_name(stage));
#endif
    } else {
        st.armed[layer][rank][stage] = 1;
        st.bytes[layer][rank][stage] = bytes;
#if DS4_EP_STAGE_PROF_NVTX
        if (rank == 0) nvtxRangePop();
#endif
    }
}

/* Collection: only call after this layer-step's device work is complete
 * (post-replay sync / post-eager stream drain). Reads the per-rank actual
 * route totals (fixed-capacity plan) and the stage timings; emits one TSV
 * row per armed stage: layer, rank, stage, ms_event, rows, bytes, pct (of
 * the rank's armed-stage sum in this collect), plus a route-skew line. */
static void ep_stage_prof_collect(const Options &opt, RankState ranks[kGpus],
                                  const char *mode, int step_index) {
    if (!opt.ep_stage_profile) return;
    const int layer = opt.layer;
    if (layer < 0 || layer >= EpStageProfilerState::kMaxLayers) return;
    EpStageProfilerState &st = g_ep_stage_prof;
    st.collect_seq++;

    int route_totals[kGpus] = {};
    bool have_totals = false;
    if (ranks[0].d_route_totals) {
        CHECK_CUDA(cudaSetDevice(ranks[0].device));
        if (cudaMemcpy(route_totals, ranks[0].d_route_totals,
                       sizeof(route_totals),
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
            have_totals = true;
        } else {
            (void)cudaGetLastError();
        }
    }
    if (have_totals) {
        std::printf("tp_ep_ep_stage_routes\tseq\t%llu\tmode\t%s\tlayer\t%d\t"
                    "step\t%d\tposition\t%llu\tcapacity\t%d\t"
                    "routes\t%d,%d,%d,%d,%d,%d,%d,%d\n",
                    st.collect_seq, mode, layer, step_index,
                    (unsigned long long)opt.position,
                    ranks[0].route_capacity,
                    route_totals[0], route_totals[1], route_totals[2],
                    route_totals[3], route_totals[4], route_totals[5],
                    route_totals[6], route_totals[7]);
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        unsigned long long stamps[kEpProfStageCount * 2] = {};
        bool have_stamps = false;
        if (opt.decode_cudagraph_gate && st.device_ready &&
            st.d_stamps[rank]) {
            CHECK_CUDA(cudaSetDevice(ranks[rank].device));
            if (cudaMemcpy(stamps, st.d_stamps[rank], sizeof(stamps),
                           cudaMemcpyDeviceToHost) == cudaSuccess) {
                have_stamps = true;
            } else {
                (void)cudaGetLastError();
            }
        }
        double ms[kEpProfStageCount];
        double total = 0.0;
        for (int s = 0; s < kEpProfStageCount; ++s) {
            ms[s] = -1.0;
            if (!st.armed[layer][rank][s]) continue;
            if (opt.decode_cudagraph_gate) {
                if (!have_stamps) continue;
                const unsigned long long b = stamps[s * 2];
                const unsigned long long e = stamps[s * 2 + 1];
                if (!b || !e || e < b) continue;
                ms[s] = (double)(e - b) / 1.0e6;
            } else {
                float v = 0.0f;
                const cudaError_t rc = cudaEventElapsedTime(
                    &v, st.ev_begin[layer][rank][s],
                    st.ev_end[layer][rank][s]);
                if (rc != cudaSuccess) {
                    (void)cudaGetLastError();
                    continue;
                }
                ms[s] = (double)v;
            }
            total += ms[s];
        }
        const int rows = have_totals
            ? route_totals[rank]
            : ranks[rank].routes;
        for (int s = 0; s < kEpProfStageCount; ++s) {
            if (ms[s] < 0.0) continue;
            std::printf("tp_ep_ep_stage_profile\tseq\t%llu\tmode\t%s\t"
                        "layer\t%d\trank\t%d\tstage\t%s\tms_event\t%.4f\t"
                        "rows\t%d\tbytes\t%llu\tpct\t%.2f\t"
                        "step\t%d\tposition\t%llu\n",
                        st.collect_seq, mode, layer, rank,
                        ep_stage_prof_name(s), ms[s], rows,
                        st.bytes[layer][rank][s],
                        total > 0.0 ? 100.0 * ms[s] / total : 0.0,
                        step_index, (unsigned long long)opt.position);
        }
        /* Synthetic ep_window stage: rank-stream elapsed from the
         * route_plan_pack begin marker to the barrier_1373 end marker (the
         * contiguous EP region on this rank's stream). pct here reports the
         * named-stage COVERAGE of the window (sum/window); the residual
         * (100 - pct) is the unattributed other/overlap share. */
        double window_ms = -1.0;
        if (st.armed[layer][rank][kEpProfRoutePlanPack] &&
            st.armed[layer][rank][kEpProfBarrier1373]) {
            if (opt.decode_cudagraph_gate) {
                if (have_stamps) {
                    const unsigned long long b =
                        stamps[kEpProfRoutePlanPack * 2];
                    const unsigned long long e =
                        stamps[kEpProfBarrier1373 * 2 + 1];
                    if (b && e && e >= b) {
                        window_ms = (double)(e - b) / 1.0e6;
                    }
                }
            } else {
                float v = 0.0f;
                if (cudaEventElapsedTime(
                        &v, st.ev_begin[layer][rank][kEpProfRoutePlanPack],
                        st.ev_end[layer][rank][kEpProfBarrier1373]) ==
                    cudaSuccess) {
                    window_ms = (double)v;
                } else {
                    (void)cudaGetLastError();
                }
            }
        }
        if (window_ms >= 0.0) {
            std::printf("tp_ep_ep_stage_profile\tseq\t%llu\tmode\t%s\t"
                        "layer\t%d\trank\t%d\tstage\tep_window\t"
                        "ms_event\t%.4f\trows\t%d\tbytes\t0\tpct\t%.2f\t"
                        "step\t%d\tposition\t%llu\n",
                        st.collect_seq, mode, layer, rank, window_ms, rows,
                        window_ms > 0.0 ? 100.0 * total / window_ms : 0.0,
                        step_index, (unsigned long long)opt.position);
        }
    }
}
