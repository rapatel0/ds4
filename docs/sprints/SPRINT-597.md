# Sprint 597 - EP-Overhead Decomposition, Transport Ground Truth, and B2 Decision Gate

Date: 2026-06-11
Status: planned

## Goal

Reopen the TP/EP throughput program on the B2 track with a measurement-first
sprint. Decompose the EP stage (Sprint 581: `9.419 ms`/layer eager, `65.2%` of
the decode domain) into named sub-stages on the **promoted full-capture
default**, audit the per-pair EP return transport against the V100 cube-mesh
topology, and use the measured costs — not the hypothesis — to pick the order
of the staged B2 implementation (Sprints 598+).

This sprint ships **no promoted hot-path change**. Its outputs are a
decomposition artifact, a transport ground-truth artifact, a written B2
stage-order decision, and the reopened track docs.

Per `VALIDATION_CONTROL_POLICY.md` this cycle **opts into perf measurement**
(named gate: decode tok/s at the reference shape) because its failure mode is
the canonical "structurally landed but the perf didn't transfer to serving."

## Why (the corrected code model)

The planning consensus (three drafts + three cross-critiques, all verified
against source) corrected the intent's model of the promoted path in three
load-bearing ways. The sprint measures *this* path:

1. **Route plan: no host readback on the promoted path.** With graph mode on,
   `post_attention_fixed_capacity_route_plan` is active
   (`engine/post_attention_ffn.cu:42-45`, gate default-true at
   `engine/runtime_options.cuh:94`) and sets `r.routes = r.route_capacity`
   (= `slots*top_k` = 192) with **no** D2H count copy
   (`engine/router_plan.cu:198-200`). The eager host readback
   (`engine/router_plan.cu:63-88`) is not the promoted cost. The promoted cost
   is the **padded executor envelope**: the TurboMind grouped GEMM runs 192
   rows/rank every layer vs p50 max-rank pressure 64, max 132 (Sprint 542) —
   a ~3x tile overcount.
2. **EP return transport: per-pair remote-copy kernels, not NCCL.** On the
   promoted branch (`source_copy_schedule && decode_cudagraph_gate`,
   `engine/decode_loop.cu:1174-1195`), the compact EP return launches
   `enqueue_graph_f32_copy_between_devices` →
   `copy_f32_kernel` (`engine/runtime_pack.cu:176-190`, device IDs ignored;
   UVA remote loads) for **every (dst, src) pair — 56 kernel launches per
   layer**. `broadcast_ep_return_slices` (8 sequential NCCL broadcasts) is the
   **non-graph** branch (`decode_loop.cu:1196-1233`); `ncclReduceScatter` is
   the non-compact branch. On the SXM2 hybrid cube mesh, 12 of 28 undirected
   pairs have no NVLink — those remote loads cross SYS (PCIe/QPI), and
   **nothing accounts for them**: `record_peer_copy()` is not wired into this
   path, so `peer_copy_sys_bytes=0` proves nothing about it. The promoted
   path may be silently violating the no-SYS policy on ~43% of pairs.
3. **Barriers.** `sync_all()` (`decode_loop.cu:174-192`) is unconditional at
   `decode_loop.cu:954, 996, 1062, 1144, 1170, 1373`; in graph mode it is the
   8×8 cross-rank event barrier `enqueue_cross_gpu_stream_barrier`
   (`engine/output_head.cu:1726-1778`) — every stream waits on every rank.
   (`decode_loop.cu:918/1238` are `sync_after_decode_stage`, opt-in only.)

Two priors are explicitly corrected in the record:

- **Sprint 396 showed NCCL allreduce 2.96x FASTER than custom peer-doubling**
  (`4.513` vs `13.366` ms at 32 tokens). Custom peer transport is not
  pre-proven; any 599 transport candidate must beat both the NCCL control and
  the current graph-copy path on measurement.
- The actual expert math is small: TurboMind MXFP4 gate/up `0.174 ms` + down
  `0.051 ms` at the 96-route compact shape (Sprint 200). The weight-streaming
  roofline (~17 active local experts × 12.6 MB × 43 layers ≈ 9 GB/rank/step ≈
  ~10 ms at HBM2) puts the hardware ceiling near ~1,000+ tok/s aggregate at
  32 slots vs the measured 26.8 — the gap is scaffolding, not FLOPs. This
  sprint either confirms that split with numbers or refutes it.

## Reference shape and gates

- Shape: `32` slots / `256K` context / `64` tok/req, deterministic
  (`temperature=0`, `top_p=1`), steady-state window, startup+warmup excluded,
  `128` req — exactly the Sprint 581 methodology.
- Correctness gate (default): tolerance — selected-token AND
  generated-sequence agreement ≥ `0.99` vs the promoted control
  (`tools/ds4-v100-http-response-tolerance.py`); control artifact reused from
  Sprint 580/581 (no invalidators apply).
- Perf gate (opt-in, named): aggregate decode tok/s at the reference shape on
  the promoted full-capture leg, plus the decode-domain attribution.
- Topology invariant: NCCL graph SYS edges stay zero as promoted. The kernel
  remote-load SYS classification produced by Phase 1 is a **finding**, not a
  regression gate (the path is already promoted).
- All builds + runs on the pod (gpu-01, 8x V100-SXM2-32GB); the laptop cannot
  validate this work. Use the global A/B lock; no concurrent V100 jobs.

## Plan

### Phase 0 - Reproduce the anchor (gate for everything else)

1. Rebuild the current tree on the pod; run the reference shape on the
   promoted full-capture default and on the pure-eager leg.
2. Confirm the Sprint 581 anchors still hold post-MTP churn: ~`26.8` tok/s
   aggregate (full capture) and the eager per-layer decode-domain attribution
   with EP ≈ `9.4 ms`/layer. If they do not, record the new anchor and use it
   as the denominator for everything downstream; investigate only if the
   drift exceeds ~15%.

### Phase 1 - Topology + transport ground truth (the SYS audit)

1. Archive `nvidia-smi topo -m` from gpu-01; derive the per-GPU NVLink
   adjacency, the 12 non-NVLink undirected pairs, and (for 599) each
   non-adjacent pair's candidate one-hop NVLink relays.
2. Standalone microbench (new tool, not in the serving hot path): time
   `copy_f32_kernel`-style UVA remote loads for all 56 directed pairs at the
   EP return payload sizes (compact rows × `kHidden/kGpus` fp32; sweep
   ~8 KB-512 KB). Report per-pair latency and bandwidth; classify
   NVLink vs SYS.
3. Cross-check one serving window with nsys: attribute the per-pair EP-return
   copy kernels by (dst, src) and confirm the microbench ranking holds in
   situ.
4. Deliverable: per-pair table + a one-page finding on whether the promoted
   EP return crosses SYS and what it costs per layer/step.

### Phase 2 - EP sub-stage instrumentation

1. Add `DS4_V100_TP_EP_EP_STAGE_PROFILE` (default off) plumbed through the
   launcher (`tools/ds4-v100-run-tp-ep-appliance.sh`) into `Options`
   (`engine/runtime_options.cuh`, `engine/runtime_types.cuh`).
2. Flag-on behavior: paired `cudaEventRecord` nodes per rank at the EP
   sub-stage boundaries in `engine/decode_loop.cu` — {route-plan kernels,
   routed-input pack, gate/up GEMM, down GEMM, dense-overlap wait,
   contribution pack/reduce, EP return copies (per (dst,src) pair),
   barrier-wait per `sync_all()` site (954/996/1062/1144/1170/1373), compose,
   other} — plus NVTX ranges; pre-allocated event pools only (no `cudaMalloc`,
   no D2H in the captured region); TSV emitter in
   `engine/runtime_profiler.cu`/`engine/diagnostics_support.cu`:
   `layer, rank, stage, ms_event, rows, bytes, pct`.
3. Extend the eager `std::chrono` splits to the same stage list for the
   reconciliation leg.
4. Flag-off behavior: byte-identical promoted path — verified by the
   tolerance gate vs the promoted control and by comparing captured-graph
   node counts.
5. Report the flag-on costs: graph node-count delta, capture/replay/cache
   behavior delta, and the flag-on vs flag-off decode tok/s delta
   (**must be ≤ 3%**, else the attribution is treated as perturbed and the
   nsys leg alone decides).

### Phase 3 - Reference-shape capture and reconciliation

1. **Authority leg:** nsys/NVTX on the *unmodified* promoted full-capture
   default (flag off) — kernel-timeline mapping to stages across 8 ranks.
2. **Table leg:** flag-on full-capture run — the per-rank per-layer TSV.
3. **Reconciliation leg:** eager run with the extended `std::chrono` splits,
   reconciled to the (possibly re-anchored) Sprint 581 EP bucket.
4. Closure: named sub-stages must sum to the EP-stage total within **≤ 10%
   unattributed residual** per layer-class; report the residual explicitly as
   "other/overlap" with rank-local elapsed vs step-critical-path
   distinguished. Do not force-fit.
5. Representativeness: record the per-layer route-skew distribution
   (p50/p95/max per-rank routes, zero-route rank occurrences) over the
   measured window; sample one sub-capacity ramp window; separate first
   capture vs replay vs persistent-cache-hit steps in the artifact.
6. Archive raw nsys captures, TSVs, and the assembled decomposition table to
   the pod workspace artifact dir; reference paths in the sprint report.

### Phase 4 - Decision gate

Adjudicate the ~5%/95% math-vs-scaffolding hypothesis with numbers (including
the padded-GEMM tax: eager ~64-row actual vs graph 192-row envelope), then
pick the 598 lead stage by measured ms, using this branch table:

| If the dominant measured cost is | Then 598 leads with |
|---|---|
| Per-pair EP return copies, esp. SYS pairs | B2-C transport: static one-hop no-SYS forwarding schedule (+ pair batching) |
| Barrier-wait at the `sync_all()` sites | B2-D per-pair event dependencies replacing the 8×8 barrier |
| Padded grouped GEMM (192 vs ~64 rows) | B2-A device-masked / route-blocked executor (no ABI change first) |
| Contribution pack/reduce + compose kernels | B2-B sparse fp16 row-indexed return + fused weighted compose |
| Expert GEMM < ⅓ of EP stage | do **not** lead with deeper GEMM fusion (B2-E stays last/optional) |

Write the decision (stage order + one-line ms justification per stage) into
the Sprint 597 report and `SPIKE_B_STEERING.md`. The full stage menu and
go/no-go gates live in `SPRINT-597-DEFERRED.md`.

### Phase 5 - Reopen the track in the durable docs

1. `README.md`: supersede the "abandoned / research archive" note — track
   reopened for the measured EP-overhead elimination cycle; MTP stays punted
   (`MTP_IMPLEMENTATION.md`); PP stays a frozen baseline.
2. `SPIKE_B_STEERING.md`: B2 backlog → active; record the (re-)anchored
   baseline, the decomposition pointer, and the corrected transport/Sprint-396
   facts.
3. `docs/sprints/STATUS.md` rollup + `docs/sprints/EXPERIMENT-STATUS.md`
   gate entry for `DS4_V100_TP_EP_EP_STAGE_PROFILE`.
4. `docs/sprints/VISION.md`: already updated by planning (revision 564);
   confirm the sequence entry matches the measured decision after Phase 4.

## Definition of Done

1. **Anchor reproduced** or re-anchored with the delta explained
   (Phase 0 artifact).
2. **Transport ground truth archived**: gpu-01 topology dump; per-pair
   (56 directed) remote-load latency/bandwidth table at EP payload sizes;
   NVLink-vs-SYS classification; in-situ nsys cross-check; a written finding
   on promoted-path SYS exposure.
3. **Decomposition exists and is reproducible**: one launcher flag reproduces
   the per-rank, per-layer EP sub-stage table on the full-capture leg; raw
   nsys + TSV artifacts archived. The decomposition covers the **actual
   promoted transport** (`copy_f32_kernel` per-pair branch); NCCL
   broadcast/ReduceScatter appear only as labeled non-graph/non-compact
   controls.
4. **Self-consistent**: sub-stages sum to the EP-stage total with ≤ 10%
   unattributed residual per layer-class; rank-local vs critical-path time
   distinguished; route-skew distribution, ramp-window sample, and
   capture-vs-replay separation included.
5. **Non-perturbing**: flag-off path byte-identical (tolerance gate ≥ 0.99 on
   selected-token AND generated-sequence vs the reused Sprint 580/581
   control; captured-graph node counts unchanged; NCCL graph SYS edges 0).
   Flag-on tok/s delta ≤ 3% and reported alongside node-count/cache deltas.
6. **Hypothesis adjudicated**: the ~5%/95% math-vs-scaffolding split
   confirmed or refuted with numbers, including the padded-GEMM tax
   (eager actual-rows vs graph 192-row envelope).
7. **Decision recorded**: `SPRINT-597.md` report section +
   `SPIKE_B_STEERING.md` name the 598+ stage order via the Phase 4 branch
   table, with measured ms per candidate.
8. **Track reopened**: README / steering / STATUS / EXPERIMENT-STATUS updated;
   MTP punt and PP hard cut intact; deferred items recorded in
   `SPRINT-597-DEFERRED.md`.
9. All repo changes committed, excluding user-owned
   `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Instrumentation perturbs the graph it measures (node/scheduling changes, cache invalidations) | Med | Med | nsys on the unmodified graph is the authority; flag-on is a separate capture; node-count/cache/tok/s deltas reported; ≤ 3% overhead gate |
| Eager and graph legs measure different executor shapes (64 vs 192 rows) | High | Low | Expected — documented as the padding-tax finding, not an error; the graph leg decides |
| Self-consistency does not close (stream overlap) | Med | Med | ≤ 10% residual bound; report "other/overlap" explicitly; distinguish rank-local vs critical path; do not force-fit |
| Sprint 581 anchor has drifted post-MTP churn | Med | Med | Phase 0 re-anchors first; all percentages quoted against the reproduced baseline |
| nsys overhead/artifact size on a 43-layer × 8-rank serving run | Med | Low | Short capture windows; one decode window suffices for attribution; archive on pod workspace, summarize in repo |
| Decomposition refutes the scaffolding thesis (e.g. padded GEMM dominates) | Low-Med | Low | That is a success — the Phase 4 branch table routes 598 accordingly |
| SYS finding implicates the promoted path | Med | Low (this sprint) | Finding, not regression — the path is already promoted; it feeds 598/599 priority |
| README reopen is outward-facing | — | Low | Status note only; explicitly preserves the MTP punt and PP hard cut |

## Security

No new external surface; no prompts/responses or secrets in new logs; the
profiler writes TSVs under the pod workspace. Honor the global A/B lock on
gpu-01.

## Dependencies

- gpu-01 pod (8x V100-SXM2-32GB), current TP/EP packs, TurboMind sidecar
  (`libggml-turbomind.so`), promoted launcher default
  (`DS4_V100_TP_EP_DECODE_GRAPH_MODE=full`).
- nsys/NVTX per `gpu-profiling-guidance.md`; NCU is not required.
- `tools/ds4-v100-http-response-tolerance.py` + the Sprint 580/581 promoted
  control artifact (reused).
- No dependency on 598+; this sprint produces their inputs.

## Design appendix — B2 stage contract for Sprints 598-600 (NOT 597 work)

Recorded here as the cycle contract; details and prerequisites in
`SPRINT-597-DEFERRED.md`. Order is finalized by Phase 4.

- **B2-A device-masked executor**: keep graph-visible
  `total_tokens = route_capacity`, early-exit inactive rows device-side (no
  ABI change). The `_device_total_tokens` ABI extension is built only if the
  padded GEMM dominates even after masking.
- **B2-B sparse fp16 row-indexed return + fused weighted compose**: replace
  the dense `d_ep_contrib_all` grid; note the graph branch currently rejects
  `ep_return_fp16` (`decode_loop.cu:1175`, `return 13`) — enabling it there
  is part of this stage. Sprint 241's lesson stands: fp16 return loses unless
  the conversion is fused into pack/compose.
- **B2-C static one-hop no-SYS forwarding schedule** for the 12 non-NVLink
  pairs, computed at init from the Phase 1 adjacency; fixed staging buffers,
  graph-capturable, no mixed NCCL-plus-peer transport inside one captured
  graph in the first candidate; must beat BOTH the NCCL broadcast control and
  the current graph-copy path (Sprint 396 caution).
- **B2-D per-pair event dependencies** replacing the 8×8
  `enqueue_cross_gpu_stream_barrier` at the EP `sync_all()` sites — a
  destination waits only on the sources it consumes.
- **B2-E full fused dispatch→grouped-GEMM→weighted-combine**: only if A-D
  leave material overhead.

Cycle targets (gating 598→599→600 go/no-go): EP stage ≤ ~2 ms/layer and
aggregate decode ≥ ~3x the (re-)anchored baseline (≥ ~80 tok/s if the anchor
holds at 26.8), each stage individually: tolerance gate pass, peer-SYS-valid
transport proof, and a step-time reduction vs the prior promoted default at
the reference shape.

## References

- `docs/sprints/drafts/SPRINT-597-INTENT.md` (brief; note the three
  corrections in "Why" above supersede its line map and Sprint-396 claim)
- `docs/sprints/drafts/SPRINT-597-{CLAUDE,CODEX,GEMINI}-DRAFT.md` + critiques
  + `SPRINT-597-MERGE-NOTES.md`
- `SPRINT-581.md` (baseline + attribution), `SPRINT-542.md` (route pressure),
  `SPRINT-550.md` (route-blocked pack), `SPRINT-396.md` (NCCL vs doubling),
  `SPRINT-200.md` (TurboMind kernel times)
- `SPIKE_B_STEERING.md`, `docs/sprints/VISION.md`,
  `docs/sprints/VALIDATION_CONTROL_POLICY.md`, `gpu-profiling-guidance.md`
