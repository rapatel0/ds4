/* Sprint 597 Phase 2: EP sub-stage profiler gate. Plumbed from the launcher
 * via the DS4_V100_TP_EP_EP_STAGE_PROFILE environment variable; default off.
 * Flag-off leaves the promoted path byte-identical (every profiler entry
 * point returns immediately). */
static inline bool ds4_ep_stage_profile_env_default() {
    const char *v = std::getenv("DS4_V100_TP_EP_EP_STAGE_PROFILE");
    if (!v || !*v) return false;
    return !(v[0] == '0' && v[1] == '\0');
}

/* Sprint 598 B2-C: EP-return transport selector for the promoted
 * full-capture graph branch. DS4_V100_TP_EP_EP_RETURN_TRANSPORT=copy|nccl;
 * default copy (the s597 per-pair copy_f32 path, byte-identical). nccl
 * captures the grouped per-source NCCL broadcast return
 * (broadcast_ep_return_slices) inside the decode graph instead. */
static inline bool ds4_ep_return_transport_env_nccl() {
    const char *v = std::getenv("DS4_V100_TP_EP_EP_RETURN_TRANSPORT");
    return v && std::strcmp(v, "nccl") == 0;
}

/* Sprint 601 Phase B: NCCL-free EP return. relay = src-side peer-WRITE
 * kernels over NVLink; the 12 SYS pairs are forwarded one-hop through a
 * staging buffer on the relay GPU dst^4 (NVLink-adjacent to both ends per
 * the s597 Phase 1 relay table; each GPU relays exactly 3 directed pairs).
 * Pure byte moves (bit-exact by construction), fixed event order (the s597
 * 8x8 event barriers), graph-capturable, no SYS-path traffic. Combined
 * with SWIGLU_EXCHANGE=batched this removes 9 of 16 captured NCCL
 * collectives per rank-layer. Default off. */
static inline bool ds4_ep_return_transport_env_relay() {
    const char *v = std::getenv("DS4_V100_TP_EP_EP_RETURN_TRANSPORT");
    return v && std::strcmp(v, "relay") == 0;
}

/* Sprint 599 C-A: shared swiglu_down input exchange transport. copy =
 * per-(dst,src,slot) UVA remote-load copies inside
 * materialize_shared_swiglu_down_input (the s198-era path; crosses SYS).
 * nccl = local pack + one grouped ncclAllGather + local strided unpack
 * (no remote loads). Default copy. */
static inline bool ds4_swiglu_exchange_env_nccl() {
    const char *v = std::getenv("DS4_V100_TP_EP_SWIGLU_EXCHANGE");
    return v && std::strcmp(v, "nccl") == 0;
}

/* memcpy2d variant: one strided P2P 2D DMA per (dst,src) pair. */
static inline bool ds4_swiglu_exchange_env_memcpy2d() {
    const char *v = std::getenv("DS4_V100_TP_EP_SWIGLU_EXCHANGE");
    return v && std::strcmp(v, "memcpy2d") == 0;
}

/* batched variant: one strided UVA remote-load kernel per (dst,src). */
static inline bool ds4_swiglu_exchange_env_batched() {
    const char *v = std::getenv("DS4_V100_TP_EP_SWIGLU_EXCHANGE");
    return v && std::strcmp(v, "batched") == 0;
}

/* Sprint 600 root-cause probes (all default off; flag-off path is
 * byte-identical because no kernels are enqueued and no state allocated).
 *
 * DS4_V100_TP_EP_S600_DELAY="site:us[,site:us...]" inserts a flag-gated
 * busy-wait kernel at named points of the captured layer graph. The wait
 * duration is read from device memory at replay time, so the host can
 * retune (or jitter) it between replays without re-capturing. Sites:
 *   xchg_tail   on each rank stream after the swiglu exchange writes
 *   pre_down    on each dense stream before the shared-down GEMM
 *   post_down   on each dense stream after the shared-down GEMM
 *   pre_pack    on each rank stream before the EP contribution pack
 *   pre_return  on each rank stream before the EP-return broadcasts
 *   post_return on each rank stream after the EP-return broadcasts
 *   post_compose on each rank stream after compose
 *
 * DS4_V100_TP_EP_S600_JITTER="min_us:max_us:seed" randomizes every enabled
 * delay site's duration before each replay (perturbation stress).
 *
 * DS4_V100_TP_EP_S600_SWIGLU_VERIFY=1 enqueues, after the swiglu exchange,
 * one checker kernel per (dst,src) pair that re-reads the remote source
 * segment and bit-compares it with the locally exchanged copy; mismatch
 * counts are read back and printed after every replay (a nonzero count
 * proves the exchange consumed stale data).
 *
 * DS4_V100_TP_EP_GRAPH_DOT_DIR=/path dumps cudaGraphDebugDotPrint of the
 * captured layer graph (layer selected by DS4_V100_TP_EP_GRAPH_DOT_LAYER,
 * default 2) right after capture. */
static inline const char *ds4_s600_delay_spec_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_S600_DELAY");
    return (v && *v) ? v : nullptr;
}
static inline const char *ds4_s600_jitter_spec_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_S600_JITTER");
    return (v && *v) ? v : nullptr;
}
static inline bool ds4_s600_swiglu_verify_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_S600_SWIGLU_VERIFY");
    if (!v || !*v) return false;
    return !(v[0] == '0' && v[1] == '\0');
}
/* DS4_V100_TP_EP_S600_RETURN_VERIFY=1: after the EP-return broadcasts,
 * bit-compare each rank's received slice (d_ep_remote[src]) against a fresh
 * remote read of the source rank's d_ep_contrib_all dst-slice; a mismatch
 * proves the NCCL return delivered stale/incorrect bytes. */
static inline bool ds4_s600_return_verify_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_S600_RETURN_VERIFY");
    if (!v || !*v) return false;
    return !(v[0] == '0' && v[1] == '\0');
}
/* DS4_V100_TP_EP_S600_AG_VERIFY=1: late-verify the post-attention NCCL
 * allgather output (d_post_attn_full_rank_major) against a fresh remote read
 * of each source rank's d_post_attn_shard, on the dense streams after the
 * 978 barrier (the dense streams are idle there, so the verification runs
 * under the pack/return/compose shadow with near-zero timing perturbation). */
static inline bool ds4_s600_ag_verify_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_S600_AG_VERIFY");
    if (!v || !*v) return false;
    return !(v[0] == '0' && v[1] == '\0');
}
/* DS4_V100_TP_EP_S600_POSTSYNC=1: after every layer-graph replay (which the
 * replay loop already host-syncs via a root-stream event), additionally
 * cudaDeviceSynchronize every rank device. If this restores correctness, the
 * root-stream replay_stop event does NOT cover the whole multi-device graph
 * and successive replays overlap (the missing-edge mechanism). */
static inline bool ds4_s600_postsync_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_S600_POSTSYNC");
    if (!v || !*v) return false;
    return !(v[0] == '0' && v[1] == '\0');
}
static inline const char *ds4_s600_graph_dot_dir_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_GRAPH_DOT_DIR");
    return (v && *v) ? v : nullptr;
}
static inline int ds4_s600_graph_dot_layer_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_GRAPH_DOT_LAYER");
    return (v && *v) ? std::atoi(v) : 2;
}

/* Sprint 600 FIX: NCCL communicator isolation for the eager output head.
 * The decode layer graphs capture ~11 collectives per layer on the compose
 * communicator; the output head then issues ~6 EAGER collectives per step on
 * the SAME communicator. Interleaving captured-plan replays with eager
 * launches on one communicator is NCCL "graph mixing" (documented fragile;
 * multi-GPU-per-process capture is explicitly cautioned). S600 forensics
 * localized a rare, timing-dependent, step-wide logit corruption to this
 * interleaving. DS4_V100_TP_EP_HEAD_COMM=shared|dedicated selects the legacy
 * shared communicator or a dedicated output-head communicator (which makes
 * the captured communicator graph-only after capture). */
static inline bool ds4_head_comm_dedicated_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_HEAD_COMM");
    return v && std::strcmp(v, "dedicated") == 0;
}
/* head_comm=host: remove the eager output-head NCCL entirely. The five
 * reductions are tiny (32-128 floats per rank): D2H + host reduce + H2D.
 * The 64 KiB-per-rank allgather becomes UVA peer copies (byte moves,
 * order-free). The sum reductions run in fixed rank order 0..7, which is a
 * different float associativity than the NCCL ring -- the tolerance gate
 * adjudicates whether the token stream is preserved. This mode needs no
 * second communicator (the pod's 64 MiB /dev/shm cannot host one: NCCL
 * allocates a fixed 8 x 4 MiB of proxy pools per comm). */
static inline bool ds4_head_comm_host_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_HEAD_COMM");
    return v && std::strcmp(v, "host") == 0;
}

/* Sprint 601 Phase A: captured-collective communicator class split.
 * S600 localized a rare timing-dependent corruption to the ~16 captured
 * NCCL RING_LL collectives per rank-layer on the ONE 8-rank-per-process
 * communicator. DS4_V100_TP_EP_COMM_SPLIT moves whole collective CLASSES
 * onto dedicated communicators (created once at startup, before capture):
 *   none      (default) everything stays on the shared compose comm
 *   epret     the 8 EP-return broadcasts (+ swiglu nccl allgather + any
 *             compose reduce-scatter) move to a dedicated comm
 *   hc        every other captured collective (hc allreduces/allgather,
 *             router allreduce, full-current broadcast, post-attn
 *             allgather, attn-output allgather) moves to a dedicated comm
 *   epret+hc  both splits (three comms total incl. the main one)
 * Flag-off the per-class comm handles alias the compose comm, so the
 * captured graph is byte-identical. Each extra comm costs device memory
 * (s598: ~0.9 GiB/GPU at default buffsize) -- the init path prints free
 * VRAM before/after every comm it creates. */
static inline bool ds4_comm_split_epret_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_COMM_SPLIT");
    return v && std::strstr(v, "epret") != nullptr;
}
static inline bool ds4_comm_split_hc_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_COMM_SPLIT");
    return v && std::strstr(v, "hc") != nullptr;
}

/* Sprint 602: hc-class collective transport.
 * DS4_V100_TP_EP_HC_TRANSPORT=nccl|kernel (default nccl, byte-identical
 * off-path). kernel replaces the remaining captured NCCL collectives (the
 * s601-localized racing set) with peer-write/kernel-reduction equivalents
 * built on the s601 relay machinery:
 *   hc max+mix allreduces, hc sumsq allreduce, hc current allgather,
 *   ffn_normed full-current broadcast, router max/sumsq/logits allreduces,
 *   post-attention allgather.
 * Broadcast/allgathers are pure byte moves (bit-exact by construction;
 * dst^4 one-hop relay for SYS pairs). Allreduces are ring-order-exact
 * kernel folds reproducing NCCL's ring reduce-scatter accumulation order
 * (chunk schedule calibrated by tools/s602-fold-probe; parameters
 * overridable via the DS4_V100_TP_EP_S602_* envs below) so the s597
 * control anchor stays bit-valid. Combined with EP_RETURN_TRANSPORT=relay
 * + SWIGLU_EXCHANGE=batched the captured decode graph contains ZERO NCCL
 * ops. */
static inline bool ds4_hc_transport_env_kernel() {
    const char *v = std::getenv("DS4_V100_TP_EP_HC_TRANSPORT");
    return v && std::strcmp(v, "kernel") == 0;
}
/* DS4_V100_TP_EP_S602_VERIFY=1: bring-up bit-verifier (s600 pattern). The
 * NCCL collectives still run and feed the consumers (so the run's bits are
 * anchored); the kernel transport runs in parallel on shadow copies of the
 * same inputs and every result is bit-compared in-graph, with per-class
 * mismatch counters read back after each replay. */
static inline bool ds4_s602_verify_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_S602_VERIFY");
    if (!v || !*v) return false;
    return !(v[0] == '0' && v[1] == '\0');
}
/* DS4_V100_TP_EP_S602_KERNEL_MASK: per-collective bring-up mask (hex or
 * decimal; default 0xFF = all). Bits: 0 hc max+mix, 1 hc sumsq, 2 hc
 * allgather, 3 ffn_normed broadcast, 4 router max, 5 router sumsq,
 * 6 router logits, 7 post-attn allgather. */
static inline unsigned ds4_s602_kernel_mask_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_S602_KERNEL_MASK");
    if (!v || !*v) return 0xFFu;
    return (unsigned)std::strtoul(v, nullptr, 0);
}
/* Ring-order-exact fold parameters (defaults = s602 fold-probe findings;
 * envs allow recalibration without rebuild).
 *   DS4_V100_TP_EP_S602_RING       semicolon-separated per-channel rings
 *   DS4_V100_TP_EP_S602_FOLD_DELTA fold start = chunk + delta (mod 8)
 *   DS4_V100_TP_EP_S602_MIN_CHUNK  NCCL LL min chunk granularity (elems)
 *   DS4_V100_TP_EP_S602_NCHANNELS  channels the chunk loop assumes */
static inline const char *ds4_s602_ring_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_S602_RING");
    /* Default = NCCL's measured channel-0 auto ring on this pod under
     * NCCL_P2P_LEVEL=NVL (s602 fold-probe run2). NOT the s597 NO_SYS_RING
     * "0 3 2 1 5 7 6 4", which is only exported when ALLOW_VISIBLE_REMAP=1
     * (not the reference config). */
    return (v && *v) ? v : "0 3 2 1 5 6 7 4";
}
static inline int ds4_s602_fold_delta_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_S602_FOLD_DELTA");
    return (v && *v) ? std::atoi(v) : 1;
}
static inline int ds4_s602_min_chunk_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_S602_MIN_CHUNK");
    /* Default 0 = auto size rule calibrated by fold-probe run3:
     * mc(count) = 2 * clamp(pow2ceil(count/16), 96, 512) -- NCCL 2.19's
     * LL nthreads-by-size stepping (minChunk = nthreads*2 floats).
     * Matches every probed shape: 768->192, 8192->1024, 6144->1024,
     * 4096->512, 2048->256, <=1024->192. */
    return (v && *v) ? std::atoi(v) : 0;
}
static inline int ds4_s602_nchannels_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_S602_NCHANNELS");
    return (v && *v) ? std::atoi(v) : 1;
}

/* Sprint 603: site-synchronization mode for the s602 kernel collectives.
 * DS4_V100_TP_EP_S602_SYNC=join|edges (default join = the s602 all-rank
 * rank-stream join at both sync points of every site - the correctness
 * default; byte-identical captures). edges = per-collective producer ->
 * consumer dependency edges derived from the kernel read/write sets
 * (SPRINT-603-REPORT Phase A table): E0 each rank waits its 4 NVLink
 * peers (copy3/gather8/fold direct reads + relay-write WAR), E1 each fold
 * dst waits its mirror g^4 (the staged SYS forwards; peers at the
 * allgather/broadcast sites where E1 is also the site exit), E2 after the
 * AR folds launch each rank waits its 4 peers (the "my buffers are free
 * at collective completion" closure the falsified s602 pairwise set
 * lacked). Per-point bisect overrides, read only in edges mode:
 *   DS4_V100_TP_EP_S602_SYNC_E0=join|peers
 *   DS4_V100_TP_EP_S602_SYNC_E1=join|peers|mirror
 *   DS4_V100_TP_EP_S602_SYNC_E2=join|peers|none */
static inline bool ds4_s602_sync_env_edges() {
    const char *v = std::getenv("DS4_V100_TP_EP_S602_SYNC");
    return v && std::strcmp(v, "edges") == 0;
}
/* Sprint 603 Phase D: dense-WAR guard. The ffn_bcast site's writers
 * (ranks 0..4 own-copy; relays 1,2,3 cross-write d_current_full of
 * 5,6,7) overwrite a buffer whose PREVIOUS value is consumed on the
 * destination's DENSE streams; the s602 rank-stream sync (join or edges)
 * never orders those readers, and the s601 full rank+dense barrier -
 * which does - is the only configuration with a zero event census. The
 * guard records one event on every dense stream at the site's E0 and
 * makes each writer's rank stream wait the dense event of every GPU
 * whose buffer it writes (own + mirror).
 *   0 = off (s602 behavior), 1 = bcast site only (the derived hazard),
 *   2 = every s602 site (diagnostic superset). */
static inline int ds4_s602_dense_guard_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_S602_DENSE_GUARD");
    return (v && *v) ? std::atoi(v) : 0;
}
static inline const char *ds4_s602_sync_point_env(const char *name) {
    const char *v = std::getenv(name);
    return (v && *v) ? v : "";
}

/* Sprint 604 Phase A: deterministic rank<->dense hazard amplifier.
 * DS4_V100_TP_EP_DENSE_HAZARD_AMP=<us> (default 0, byte-identical off)
 * inserts a flag-gated busy-wait (the s600 device-memory-driven delay
 * kernel) on the DENSE stream at the EP/dense overlap hand-off points,
 * WIDENING the dense<->rank window so the s603-localized ordering hazard
 * fires on (nearly) every run. The s600 finding inverted: a delay at the
 * racing site restores order; a delay that holds the dense producers back
 * lets the rank-stream consumers race ahead onto stale buffers.
 *   DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE selects the injection site mask:
 *     pre_dense    on the dense stream before the EP/dense GEMMs (delays
 *                  attn/shared_gate/shared_up dense outputs vs swiglu read)
 *     post_dense   on the dense stream after the GEMMs, before the 954 join
 *     pre_down     on the dense stream before the shared-down GEMM
 *     post_down    on the dense stream after the shared-down GEMM (delays
 *                  shared_op->d_out vs the compose read)
 *     pre_compose  on the dense stream right before compose (catch-all)
 *   default = all sites. Comma-separated subset selects specific sites. */
static inline int ds4_dense_hazard_amp_us_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_DENSE_HAZARD_AMP");
    return (v && *v) ? std::atoi(v) : 0;
}
static inline const char *ds4_dense_hazard_amp_site_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE");
    return (v && *v) ? v : nullptr;
}

/* Sprint 604 Phase C: minimal dense->rank ordering edge (the fix).
 * DS4_V100_TP_EP_DENSE_FIX=0|1 (default 0, byte-identical off). When on,
 * the EP/dense overlap hand-off uses a CROSS-GPU dense<->rank edge (each
 * rank waits every peer's dense completion and vice-versa) at the points
 * where the rank-stream-only join leaves the dense producers/consumers
 * unordered across GPUs - the narrowest superset of the fb barrier's
 * dense involvement, without the full rank+dense 8x8 join. */
static inline int ds4_dense_fix_env() {
    /* Sprint 604: cross-rank dense->rank ordering edge for the attention-output
     * allgather hand-off (the carrier of the rank<->dense token-corruption
     * hazard). PROMOTED default-on: the 34-run soak showed fix-off corrupts
     * tokens in 7/17 runs (41%) while fix-on is 17/17 clean, at near-zero
     * perf cost. Set DS4_V100_TP_EP_DENSE_FIX=0 to roll back to the racy path
     * for diagnosis only. */
    const char *v = std::getenv("DS4_V100_TP_EP_DENSE_FIX");
    return (v && *v) ? std::atoi(v) : 1;
}

/* Sprint 605 Phase C: attention-output gather compaction.
 * DS4_V100_TP_EP_ATTN_OUT_GATHER8=0|1 (default 0, byte-identical off). When
 * on, the non-NCCL attention-output-A allgather replaces its 64 per-(dst,src)
 * cudaMemcpy2DAsync launches (8 per dst x 8 dst) with 8 single-launch gather8
 * kernels (one per dst, reading all 8 src d_out shards by pointer and writing
 * the dst's full buffer in the same [slot*Full + src*shard + h] layout). Pure
 * launch-count reduction (72->16 launches/layer for this stage); the dataflow
 * is byte-identical and the cross-GPU dense->rank ordering is unchanged (the
 * s604 DENSE_FIX edge enqueued before this block still covers the peer reads).
 * Off = the proven memcpy2D path. */
static inline int ds4_attn_out_gather8_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_ATTN_OUT_GATHER8");
    return (v && *v) ? std::atoi(v) : 0;
}

/* Sprint 606: rendezvous merge (no-graph-restructure step-floor lever).
 * DS4_V100_TP_EP_RDZV_MERGE=0|1 (default 0, byte-identical off). When on,
 * elides the final post-compose 8x8 cross-GPU event barrier (the 1373 site
 * in run_one_step) on the promoted decode path. SAFE BY DATAFLOW: the
 * compose kernels write r.d_next_hidden on r.stream; the next consumer
 * (expand_hidden_to_proxy_hc_shard_kernel in run_final_hc_carry) reads
 * r.d_next_hidden on the SAME r.stream (same-stream ordered, needs no
 * barrier), and the cross-GPU rendezvous that final_hc actually requires is
 * already supplied by the sync_all() INSIDE run_final_hc_carry (before the
 * hc final-expand allgather). So the 1373 barrier is redundant with that
 * internal sync_all. Eliding it removes ~16 event records + up to ~1280
 * cudaStreamWaitEvent enqueues per layer (the full include_copy_streams 8x8
 * barrier), a pure launch/sync reduction on the ~95%-launch/sync step.
 * Amplifier-gated (DENSE_HAZARD_AMP @ pre_compose / attn_out_a must stay
 * 1.0/1.0 with this on). Default off = the proven barrier path. Only takes
 * effect on the cudagraph decode path with final_hc_carry on. */
static inline int ds4_rdzv_merge_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_RDZV_MERGE");
    return (v && *v) ? std::atoi(v) : 0;
}

/* Sprint 599 C-B: enqueue the EP contribution pack + NCCL return right
 * after the routed GEMMs (before the dense/swiglu chain) and replace the
 * 8x8 cross-GPU barriers at the 954/978 sites with per-rank rank<->dense
 * event ordering, so the EP return overlaps the dense+swiglu work.
 * Requires the nccl EP-return transport. Default off. */
static inline bool ds4_ep_return_early_env() {
    const char *v = std::getenv("DS4_V100_TP_EP_EP_RETURN_EARLY");
    if (!v || !*v) return false;
    return !(v[0] == '0' && v[1] == '\0');
}

struct Options {
    const char *lib_path = "./build/turbomind-v100/libggml-turbomind.so";
    const char *pack_dir = nullptr;
    const char *contract_path = nullptr;
    const char *tm_index_path = nullptr;
    /* MTP (layer 43) dedicated weight source -- decoupled from the main pack
     * so the appliance loads 0-42 from --pack-dir and the MTP block from its
     * own EP-format pack (mxfp4/turbomind, fused gate_up, EP-split 32/rank). */
    const char *mtp_pack_dir = nullptr;
    const char *mtp_tm_index_path = nullptr;
    const char *mtp_contract_path = nullptr;
    const char *tokenizer_model_path = nullptr;
    int devices[kGpus] = {0, 1, 2, 3, 4, 5, 6, 7};
    int slots = 32;
    int top_k = 6;
    int layer = 2;
    int resident_profile_layer = -1;
    uint32_t kv_slot = 7;
    uint64_t position = 1024;
    int warmup = 5;
    int iters = 30;
    const char *dense_compute_tensor = nullptr;
    bool dense_compute_all_f8 = false;
    bool dense_compute_all_bf16 = false;
    bool compose_next_hidden = false;
    int decode_steps = 0;
    bool ep_return_fp16 = false;
    bool fuse_compose_sum = true;
    bool dense_hmma_compose = false;
    bool dense_f16_cublas_compose = true;
    bool dense_f16_cache_compose = true;
    bool all_layers = true;
    bool skip_descriptor_checks = true;
    bool skip_predecode_probes = true;
    bool share_tp_runtime = true;
    bool tp_runtime_explicit = false;
    bool tp_runtime_skip_unused_comp_state = true;
    uint64_t tp_runtime_scratch_mib = 1024;
    bool share_expert_bindings = true;
    bool parallel_expert_load_gate = true;
    bool overlap_ep_dense = true;
    bool direct_remote_compose = false;
    bool source_copy_schedule = true;
    bool copy_event_compose = true;
    bool compact_route_compose = true;
    bool token_major_all_layers = true;
    bool share_dense_ops = true;
    bool skip_self_compose_copy = true;
    bool multi_copy_streams = true;
    bool nccl_reduce_scatter_compose_gate = false;
    bool defer_nccl_init_gate = true;
    bool serving_bench = false;
    bool skip_decode_checksum = false;
    bool serve_http = false;
    const char *host = "127.0.0.1";
    int port = 18082;
    int max_requests = 0;
    int microbatch_wait_us = 5000;
    bool decode_cudagraph_gate = false;
    bool decode_cudagraph_replay_probe_gate = false;
    bool decode_cudagraph_persistent_replay_gate = false;
    bool decode_cudagraph_output_sync_gate = false;
    bool decode_cudagraph_hc_current_sync_gate = false;
    const char *decode_cudagraph_stage_sync = nullptr;
    const char *decode_cudagraph_suffix_stage = nullptr;
    bool decode_stage_checksum_gate = false;
    bool compact_moe_decode_gate = true;
    bool fused_gated_silu_gate = false;
    bool final_hc_carry_gate = true;
    bool diagnostic_output_head = true;
    bool diagnostic_output_head_lazy_gate = true;
    bool tp_hc_final_expand_gate = true;
    bool tp_hc_current_input_gate = true;
    bool tp_hc_current_input_peer_gather_gate = false;
    bool tp_hc_current_input_nccl_allgather_gate = true;
    bool tp_hc_current_allreduce_gate = true;
    bool tp_hc_current_input_stream_sync_gate = true;
    bool tp_hc_current_input_fused_fill_pack_gate = false;
    bool tp_hc_current_full_parity_gate = false;
    bool tp_hc_persist_state_gate = true;
    bool tp_peer_accounting_gate = false;
    bool tp_peer_reject_sys_gate = false;
    bool model_router_routes = true;
    bool router_cublas_gate = false;
    bool router_hash_fast_gate = false;
    bool gpu_route_plan_gate = true;
    bool route_plan_async_upload_gate = true;
    bool routed_ffn_norm_input_gate = true;
    bool routed_ffn_rank_major_input_gate = true;
    bool routed_ffn_rank_major_shared_input_gate = false;
    bool routed_ffn_rank_major_route_input_gate = false;
    bool routed_ffn_rank_major_input_parity_gate = false;
    bool post_attention_route_reuse_audit_gate = false;
    bool post_attention_fixed_capacity_route_plan_gate = true;
    bool post_attention_slot_major_ffn_norm_gate = false;
    bool model_router_rank_major_logits_gate = false;
    bool model_router_allreduce_logits_gate = true;
    bool true_shared_ffn_gate = true;
    bool tp_kv_all_slots_gate = false;
    bool reference_hc_reduce_gate = false;
    bool reference_hc_state_guard_gate = false;
    bool true_ds4_attention_residency_gate = true;
    bool true_ds4_attention_projection_gate = true;
    bool true_ds4_attention_projection_direct_input_fill_gate = false;
    bool true_ds4_attention_projection_rank_local_input_gate = false;
    bool true_ds4_attention_projection_rank_major_input_gate = true;
    bool true_ds4_attention_projection_input_parity_gate = false;
    bool true_ds4_attention_state_gate = true;
    bool true_ds4_attention_rope_gate = true;
    bool true_ds4_attention_saturation_audit_gate = false;
    bool true_ds4_attention_kv_norm_reference_gate = false;
    bool true_ds4_attention_raw_read_gate = true;
    bool true_ds4_attention_raw_window_gate = true;
    bool true_ds4_attention_typed_kv_raw_gate = true;
    bool true_ds4_attention_typed_kv_compressed_gate = true;
    bool true_ds4_attention_typed_kv_indexer_gate = true;
    bool true_ds4_attention_typed_kv_history_gate = true;
    bool true_ds4_attention_typed_kv_skip_current_load_gate = true;
    bool true_ds4_attention_typed_kv_skip_raw_store_gate = false;
    bool true_ds4_attention_typed_kv_skip_compressed_store_gate = false;
    bool true_ds4_attention_typed_kv_skip_indexer_store_gate = false;
    bool true_ds4_attention_typed_kv_quiet_gate = true;
    bool true_ds4_attention_typed_kv_batch_rows_gate = true;
    bool true_ds4_attention_typed_kv_stream_sync_gate = true;
    bool fp8_e5m2_kv_gate = false;
    bool true_ds4_attention_output_gate = true;
    bool true_ds4_attention_output_nccl_allgather_gate = false;
    bool true_ds4_post_attention_ffn_input_gate = true;
    bool true_ds4_semantic_skip_stats_gate = true;
    bool true_ds4_compressed_kv_gate = false;
    bool true_ds4_indexer_attention_gate = false;
    bool true_ds4_compressed_kv_direct_input_fill_gate = false;
    bool true_ds4_compressed_kv_dense_event_wait_gate = true;
    bool true_ds4_compressed_kv_skip_dense_stats_gate = true;
    bool true_ds4_compressed_kv_fused_attn_input_fill_gate = false;
    bool true_ds4_compressed_kv_fused_input_fill_gate = false;
    bool true_ds4_compressed_kv_fused_rope_round_gate = false;
    bool true_ds4_compressed_kv_fused_pool_norm_gate = true;
    bool true_ds4_compressed_kv_fused_pool_norm_rope_round_gate = false;
    bool true_ds4_compressed_reference_diff_gate = false;
    uint32_t true_ds4_attention_raw_valid_rows = 1;
    uint64_t vram_min_free_mib = 0;
    uint64_t nccl_min_free_mib = 0;
    /* Sprint 597 Phase 2: EP sub-stage profiler (default off). */
    bool ep_stage_profile = ds4_ep_stage_profile_env_default();
    /* Sprint 598 B2-C: EP-return transport (default copy = s597 path). */
    bool ep_return_nccl = ds4_ep_return_transport_env_nccl();
    /* Sprint 601 Phase B: NCCL-free peer-write relay EP return. */
    bool ep_return_relay = ds4_ep_return_transport_env_relay();
    /* Sprint 599 C-A / C-B (default off = s598 promoted path). */
    bool swiglu_exchange_nccl = ds4_swiglu_exchange_env_nccl();
    bool swiglu_exchange_memcpy2d = ds4_swiglu_exchange_env_memcpy2d();
    bool swiglu_exchange_batched = ds4_swiglu_exchange_env_batched();
    bool ep_return_early = ds4_ep_return_early_env();
    /* Sprint 600 probes (default off; see comments above). */
    const char *s600_delay_spec = ds4_s600_delay_spec_env();
    const char *s600_jitter_spec = ds4_s600_jitter_spec_env();
    bool s600_swiglu_verify = ds4_s600_swiglu_verify_env();
    bool s600_return_verify = ds4_s600_return_verify_env();
    bool s600_ag_verify = ds4_s600_ag_verify_env();
    bool s600_postsync = ds4_s600_postsync_env();
    /* Sprint 600 fix (see comment above): default shared until gated. */
    bool head_comm_dedicated = ds4_head_comm_dedicated_env();
    bool head_comm_host = ds4_head_comm_host_env();
    /* Sprint 601 Phase A (see comment above): default none. */
    bool comm_split_epret = ds4_comm_split_epret_env();
    bool comm_split_hc = ds4_comm_split_hc_env();
    /* Sprint 602: hc-class collective transport (default nccl). */
    bool hc_transport_kernel = ds4_hc_transport_env_kernel();
    bool s602_verify = ds4_s602_verify_env();
    unsigned s602_kernel_mask = ds4_s602_kernel_mask_env();
    const char *s602_ring_spec = ds4_s602_ring_env();
    int s602_fold_delta = ds4_s602_fold_delta_env();
    int s602_min_chunk = ds4_s602_min_chunk_env();
    int s602_nchannels = ds4_s602_nchannels_env();
    bool s602_sync_edges = ds4_s602_sync_env_edges();
    int s602_dense_guard = ds4_s602_dense_guard_env();
    const char *s602_sync_e0 =
        ds4_s602_sync_point_env("DS4_V100_TP_EP_S602_SYNC_E0");
    const char *s602_sync_e1 =
        ds4_s602_sync_point_env("DS4_V100_TP_EP_S602_SYNC_E1");
    const char *s602_sync_e2 =
        ds4_s602_sync_point_env("DS4_V100_TP_EP_S602_SYNC_E2");
    const char *s600_graph_dot_dir = ds4_s600_graph_dot_dir_env();
    int s600_graph_dot_layer = ds4_s600_graph_dot_layer_env();
    /* Sprint 604 Phase A amplifier + Phase C fix (default off). */
    int dense_hazard_amp_us = ds4_dense_hazard_amp_us_env();
    const char *dense_hazard_amp_site = ds4_dense_hazard_amp_site_env();
    int dense_fix = ds4_dense_fix_env();
    int attn_out_gather8 = ds4_attn_out_gather8_env();
    /* Sprint 606: rendezvous merge (elide redundant post-compose 1373
     * barrier). Default off = byte-identical proven barrier path. */
    int rdzv_merge = ds4_rdzv_merge_env();
};
