# Sprint 597 Deferred Items

Everything discussed, drafted, or critiqued during Sprint 597 planning that
was excluded from the 597 scope. Sprint 597 is measurement + decision + docs
only; all implementation below is 598+ and its **order is finalized by the
Sprint 597 Phase 4 decision gate**, not by this list's numbering.

## B2-A — Device-masked / route-blocked executor

- **What**: Recover the ~3x padded-envelope tax on the promoted full-capture
  path (executor runs `total_tokens = route_capacity = 192` rows/rank/layer
  vs p50 max-rank pressure 64, max 132 — Sprint 542). Keep the graph-visible
  host scalar at fixed capacity; add device-side early-exit/masking for
  inactive padded rows in the routed gate/up + down execution. No TurboMind
  ABI change in the first candidate. Handle the zero-active-rows-on-a-rank
  edge gracefully (no hang, no NaN).
- **Why deferred**: implementation; 597 is measurement-only (interview
  decision). Also contingent — only justified if the decomposition shows the
  padded GEMM is material (Sprint 550's route-blocked *pack* showed no
  steady-state win, so masking alone is not pre-proven).
- **Target sprint**: 598 candidate (Phase 4 branch table).
- **Prerequisites**: 597 decomposition (padded-GEMM bucket measured via the
  eager-actual vs graph-envelope delta).
- **Files**: `engine/ep_executor.cu`, `engine/turbomind_bindings.cu`,
  `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h` (read),
  `engine/router_plan.cu` (route metadata).

## B2-B — Sparse fp16 row-indexed EP return + fused weighted compose

- **What**: Replace the dense `d_ep_contrib_all` grid
  (`kGpus × slots × kHidden/kGpus` fp32, `engine/runtime_resources.cu:442`)
  with row-indexed fp16 contributions (≤ `slots*top_k` rows ×
  `kHidden/kGpus` + device row→slot index), composed by a fused weighted
  kernel. Includes enabling fp16 return on the graph branch — currently
  rejected outright (`engine/decode_loop.cu:1175`, `return 13`) even though
  the half buffers exist (`d_ep_remote_half`, `runtime_pack.cu:325-334`).
  Sprint 241 lesson: standalone fp16 return *lost* — the conversion must be
  fused into pack/compose, not a separate cast pass.
- **Why deferred**: implementation; encoding-vs-collective-shape question is
  exactly what the 597 decomposition answers first.
- **Target sprint**: 598/599 candidate per branch table.
- **Prerequisites**: 597 decomposition (pack/transfer/compose split);
  tolerance-gate headroom check for fp16 (≥ 0.99 agreement).
- **Files**: `engine/decode_loop.cu`, `engine/ep_compose.cu`,
  `engine/runtime_pack.cu`, `engine/runtime_resources.cu`,
  `kernels/v100/compose.cuh`.

## B2-C — Static one-hop no-SYS forwarding schedule

- **What**: For the 12 non-NVLink undirected pairs on the SXM2 cube mesh,
  replace direct UVA remote loads (potential silent SYS traffic, see 597
  Phase 1 finding) with a deterministic one-hop relay through an NVLink
  neighbor: static schedule computed at init from the archived adjacency,
  fixed staging buffers, fixed event order, fully graph-capturable. First
  candidate uses peer copies only — no mixed NCCL-plus-peer transport inside
  one captured graph (deadlock risk flagged by all three critiques). Must
  beat BOTH the NCCL broadcast control and the current graph-copy path
  (Sprint 396: NCCL was 2.96x faster than custom doubling — custom transport
  is not pre-proven).
- **Why deferred**: implementation; the "quick one-hop fix probe" option was
  explicitly declined in the interview to keep 597 measurement-only.
- **Target sprint**: 598 candidate — leads if per-pair SYS transfers dominate
  the decomposition.
- **Prerequisites**: 597 Phase 1 per-pair table + adjacency; a peer-SYS
  validation method valid for kernel remote loads (597 DoD 2).
- **Files**: `engine/runtime_pack.cu`, `engine/runtime_resources.cu`,
  `engine/decode_loop.cu`, possibly new `engine/ep_topology.cu`,
  `engine/runtime_types.cuh` (`v100_nvlink_count`, `PeerCopyAccounting`).

## B2-D — Per-pair event dependencies replacing the 8×8 barrier

- **What**: At the EP `sync_all()` sites (`decode_loop.cu:954, 996, 1062,
  1144, 1170, 1373`), replace `enqueue_cross_gpu_stream_barrier`
  (`engine/output_head.cu:1726-1778`; every stream waits on every rank) with
  per-(dst,src) event waits — a destination waits only on the sources whose
  contribution slices it consumes (post one-hop: only its NVLink neighbors).
  Pre-allocated event slots, fixed order, graph-capturable.
- **Why deferred**: implementation; barrier-wait share of the 9.4 ms is
  unmeasured until 597.
- **Target sprint**: 598 candidate — leads if barrier-wait dominates.
- **Prerequisites**: 597 decomposition (per-site barrier-wait attribution).
- **Files**: `engine/decode_loop.cu`, `engine/output_head.cu`.

## B2-E — Full fused dispatch→grouped-GEMM→weighted-combine kernel

- **What**: The README's "left on the table" idea: one/two persistent kernels
  per layer per rank covering pack → gate/up → SiLU → down → weighted
  scatter-add (the down+route-weight+reduce epilogue already exists in the
  ABI: `ggml_turbomind_ds4_mxfp4_down_*_reduce`, header lines ~557-626).
- **Why deferred**: largest-risk, last-resort stage; possibly unnecessary if
  A-D collapse the overhead (consensus across all three drafts).
- **Target sprint**: 600, only if 598/599 leave material overhead.
- **Prerequisites**: B2-A/B/C/D results.
- **Files**: `kernels/turbomind/ggml-turbomind/*`, `engine/ep_executor.cu`,
  `engine/decode_loop.cu`.

## TurboMind `_device_total_tokens` ABI extension

- **What**: New entry point reading the token count from a device scalar so
  the launched grid can shrink below fixed capacity without a host readback;
  or a persistent-kernel variant.
- **Why deferred**: Rejected as a 597/598 assumption — the promoted path
  already runs device-resident `expert_offsets` with a fixed host capacity
  (Gemini's "new ABI required" conclusion was rejected in merge). Build only
  if B2-A masking is insufficient and the padded GEMM still dominates.
- **Target sprint**: contingent follow-on to B2-A.
- **Prerequisites**: B2-A measured insufficient.
- **Files**: `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h`,
  `kernels/turbomind/ggml-turbomind/api.cc`, `engine/turbomind_bindings.cu`.

## Wire `record_peer_copy` accounting into graph remote-copy kernels

- **What**: Close the no-SYS accounting blind spot: graph-branch
  `copy_f32_kernel` remote loads are not counted by `PeerCopyAccounting`, so
  `peer_copy_sys_bytes=0` cannot prove the promoted path honors the no-SYS
  policy. Add per-pair byte accounting (or a startup adjacency guard) for the
  kernel remote-load path.
- **Why deferred**: 597 covers the question via the offline microbench +
  nsys attribution; hot-path accounting is a code change.
- **Target sprint**: alongside B2-C.
- **Files**: `engine/runtime_pack.cu`, `engine/runtime_types.cuh`.

## NCCL ring/topology pinning

- **What**: Pin/inspect NCCL ring construction against the cube mesh
  (deferred tuning lever recorded in Sprint 581).
- **Why deferred**: only matters if NCCL collectives remain on the hot path
  after the B2 transport decision; the promoted EP return doesn't use NCCL.
- **Target sprint**: future; revisit after 599.
- **Files**: `engine/runtime_resources.cu` (NCCL init).

## Slots × context envelope sweep + C4 KV spill (scale tokens/step)

- **What**: After the EP-overhead constant is fixed, raise tokens/step to
  amortize the ~10 ms/step weight-streaming floor: shape envelope sweep and
  the C4 KV-spill lever to admit > 32 slots (more tokens/expert → climb from
  ~hundreds toward the ~1,000+ tok/s roofline).
- **Why deferred**: multiplies the per-step constant; pointless until the
  constant is fixed (same reasoning that deprioritized MTP).
- **Target sprint**: post-cycle tuning sprint (601+).
- **Prerequisites**: B2 cycle target met (EP ≤ ~2 ms/layer).

## MTP (B1) — stays punted

- **What**: The (K+1)-wide specdec loop; draft acceptance stuck 0/71 with
  raw-SWA window / attention-output / FFN-activation semantics at layer 43
  still unsearched (Sprints 587-596, `MTP_IMPLEMENTATION.md`).
- **Why deferred**: user punt (2026-05-30) reaffirmed during this planning;
  not on the critical path for the structural win; only a multiplier once
  the EP constant is fixed.
- **Target sprint**: none — requires an explicit reopen decision.

## Declined during planning

- **One-off one-hop staging probe inside 597** (interview: keep 597
  measurement-only; the probe blurs into B2-C implementation).
- **Mega-sprint packaging** (instrumentation + first B2 stage in one sprint):
  violates the focused-sprint convention; couples measurement risk with
  hot-path rewrite risk.
- **Eager decomposition as the numerical gate** (Codex draft position):
  rejected in interview — eager runs ~64 actual rows vs the promoted 192-row
  envelope; it remains the reconciliation cross-check only.
- **Bit-exact validation gate**: tolerance policy default stands; nothing in
  597 justifies opt-in bit-exact.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| B2-A device-masked executor | 598 (candidate) | 597 decomposition |
| B2-B sparse fp16 return + fused compose | 598/599 (candidate) | 597 decomposition; fp16 tolerance headroom |
| B2-C one-hop no-SYS forwarding | 598 (candidate) | 597 Phase 1 per-pair table |
| B2-D per-pair event dependencies | 598 (candidate) | 597 barrier-wait attribution |
| B2-E full fused MoE kernel | 600 (optional) | B2-A/B/C/D residual overhead |
| `_device_total_tokens` ABI | contingent | B2-A insufficient |
| Peer-copy accounting for graph kernels | with B2-C | — |
| NCCL ring/topology pinning | future | NCCL still on hot path post-599 |
| Envelope sweep + C4 KV spill | 601+ | cycle target met |
| MTP (B1) | none | explicit reopen decision |
