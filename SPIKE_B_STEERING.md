# Spike B Decode-Optimization Steering (updated 2026-05-28 after Sprint 526)

Steering for the next TP/EP serving-throughput phase, off the de-confounded
steady-state reference (32 slots / 256K / 256 req / 64 tok/req, ~35.9 tok/s
server decode, ~889 ms decode domain).

## Unifying diagnosis — read this first

Both dominant domains are **overhead-bound, not compute- or bandwidth-bound:**

- **EP 53% (~473 ms):** at 32 slots × top-6 = 192 activations over 256 experts,
  each GPU's 32 local experts see **<1 token** → grouped GEMM is tile-underfill;
  the cost is dispatch / pack / reduce / all-to-all orchestration, not FLOPs.
- **HC-current 40% (~357 ms):** moves ~2 MiB → the time is launch-count + sync +
  **GPU0-centralization** × 43 layers, not data.

**Consequence:** wins come from STRUCTURE — de-centralize, fuse, fewer launches,
remove host synchronization, capture static sub-regions — **not** from faster
kernels. Optimizing the expert GEMM alone moves a fraction of 53% and none of
40%. MTP remains deferred support code until the base TP/EP path is stable and
optimized.

## Current reassessment after sprints 478-525

- **A1-A3 are done.** A1 RMS-norm rank-local is rolled into A2. A2 HC mix
  row-parallel all-reduce is promoted from Sprint 478. A3 router rank-local
  all-reduce is promoted from Sprint 480.
- **A4 is complete for the served TP/EP path.** Sprint 483's "A6 PATH 4" result
  was structurally A4 for the attention-projection consumer, and Sprint 526
  finished the post-attention FFN shared/routed consumers in
  `engine/post_attention_ffn.cu`. The promoted path now reports
  `rank_major_shared_input=1`, `rank_major_route_input=1`, and
  `slot_major_ffn_norm=0`.
- **True A6 is still open.** In this document, A6 means fusing HC/current
  computation into the attention-projection prologue. It does not mean the
  Sprint 483 rank-major attention-projection consumer conversion.
- **C1 is newly more plausible, but not next.** Sprint 479 removed promoted
  hot-path direct peer-copy transport in favor of NCCL, and the structural
  extraction made the surface readable. Still, C1 should wait until
  output-head A1, sync-point reduction, and compact EP compose reduce the
  capture surface.
- **Use previous promotions as the control.** Do not duplicate control runs
  solely because a new sprint starts. Refresh control only when the binary,
  launcher defaults, topology policy, validation harness, model path, or target
  shape changed enough to invalidate the previous promoted artifact.

## A. HC-current (40%) — de-centralize GPU0 (highest-ROI, eager-path win today)

Steps 2–6 and 9–10 run on GPU0 over the full hidden while 7 GPUs idle. Every
"global over hidden" op decomposes into rank-local partial + one tiny all-reduce:

- **A1 RMS-norm rank-local:** done as part of A2. Partial sum-of-squares on
  each `[slots,4,512]` shard → all-reduce `[slots,4]` scalar → normalize
  locally. No gather.
- **A2 HC mix as row-parallel GEMM:** done and promoted. `attn_fn[16384×24] @ hc[slots,16384]` —
  16384=4×4096 is the contraction, already sharded (2048/rank). Each rank computes
  partial `[slots,24]` → all-reduce `[slots,24]` (3 KiB). Replaces gather-to-0 +
  full-norm + mix + split + broadcast with 2 rank-local kernels + 1 tiny allreduce.
- **A3 Router rank-local:** `[slots,4096]·[4096,256]`, 4096 contraction → row-
  parallel → all-reduce `[slots,256]`. Done and promoted.
- **A4 Drop the full-current allgather (step 8):** complete for the served
  TP/EP path. Attention projection is done under the old "A6 PATH 4" name,
  router is covered by promoted A3, and Sprint 526 made post-attention FFN
  shared/route consumers rank-major by default while removing the promoted
  path's slot-major FFN norm dependency.
- **A5 Fuse the survivors:** after A1–A4, HC-current ≈ 3 rank-local kernels + 2
  tiny all-reduces/layer vs ~12 steps. Fuse norm+partial-mix; mix-apply+FFN-norm.
- **A6 Fuse HC into the attention-projection prologue** (compute `current_shard`
  inside the projection kernel; drop the intermediate buffer + a launch).

Net: GPU0-serial / ~12-launch / barrier-heavy -> 8-way parallel / ~3-launch / 2
tiny collectives. **A1-A4 are complete for the served TP/EP path; the next
bankable NCCL cleanup is the model-boundary output-head A1 pattern.**

## B. EP (53%) — orchestration + sub-1-token experts

- **B1 MTP is deferred.** EP is 53%
  *because* M<1/expert; MTP (verify K draft tokens) makes experts see (K+1)× tokens
  → tiles fill + fewer steps. That remains true, but TP/EP MTP is intentionally
  out of the active docket until the base TP/EP path is stable and optimized.
- **B2 Fuse dispatch + grouped-GEMM + weighted-combine** into 1–2 kernels,
  device-side offsets only (no host sync on route counts). Template: the fork's
  `awq_moe_single_token_sm70` compact path. Make compact-route-compose the
  default. First finish the open compact-route transport half: replace the served
  path's variable-size per-pair peer-copy-equivalent compose movement with grouped
  `ncclSend`/`ncclRecv` or a statically bucketed NCCL scheme. Sprint 480's
  `ncclReduceScatter` evidence covers only non-compact FP32 and is not proof for
  served compact traffic.
- **B3 TP-sharded experts vs EP A/B (the S-F question — now justified).** EP's 53%
  is dispatch/all-to-all. TP-experts have **no all-to-all** (reduce via the hidden
  all-reduce). For 13B-active/8-GPU/32-slot, test whether all-to-all overhead >
  TP-reduce cost. Strongest evidence yet to actually run it.
- **B4 Overlap shared (dense, rank-local) expert with the routed all-to-all
  dispatch** (independent until combine; re-apply sprint-261 overlap on serving).
- **B5 Correctness-preserving capacity balancing** (capacity-16 broke output in
  435; need a parity-clean fixed-shape scheme — also helps capture, see C1).

## C. Cross-cutting — and the highest-ceiling idea

- **C1 Piecewise graph capture (likely the biggest single win).** Full-serving-
  graph is blocked on parity + dynamism, but the per-layer compute region
  (HC-current → attention → EP → compose) is proven capturable in the smoke. Graph
  ONLY that static sub-region; run dynamic orchestration (request mgmt, output
  head, sampling) eager around it; feed routing via **persistent device buffers**
  (fixed worst-case `slots×top_k`; route values change per step, graph structure
  fixed). Bridges "works in smoke, blocked in serving" → could recover most of the
  2.27× without solving the whole loop. Do this after A4, output-head A1,
  sync-point reduction, and compact EP compose so the captured region is not
  polluted by avoidable host syncs or remaining non-NCCL compose movement.
- **C2 Fix the graph-in-serving parity bug directly.** Graph mode changes the
  first token = a finite set of missing sync→event dependencies (461 fixed one).
  Diff eager vs graph dependency graph; close them all. Debuggable, not fundamental.
- **C3 Launch-count reduction is the universal eager-path lever.** Every fusion
  above (A5/A6/B2) helps the eager path that serves today, graphs or not. Given
  graph-in-serving is blocked, this is the only lever lifting the served ~12% util
  right now.
- **C4 Run the spill check** (still not done) on the HC-mix kernel, head_dim-512
  attention, and rank-major consume kernels: `-Xptxas -v` (registers/smem/spill) +
  ncu (occupancy, long-scoreboard stalls). If they spill, the rank-local rewrites
  leave perf on the floor.
- **C5 Replace hot host syncs with device events.** The hot engine path still
  contains dozens of `cudaDeviceSynchronize()` / `cudaStreamSynchronize()` calls
  across HC-current, decode-loop, attention projection/read/output,
  post-attention FFN, and EP compose. Replace structurally unnecessary host
  round-trips with `cudaEventRecord()` / `cudaStreamWaitEvent()` dependencies.
  This is both an eager-path cleanup and a graph-capture prerequisite.

## D. Model-boundary NCCL cleanup

- **D1 Output-head A1 pattern.** `engine/output_head.cu` still has the same
  gather-to-GPU0 → centralized RMS/mix/weighted sum/final RMS pattern that A2
  fixed inside every layer, plus hard host synchronizations. Apply the same
  rank-local partial plus NCCL all-reduce template at the model boundary:
  partial sum-of-squares over each `[slots,4,512]` shard, all-reduce the per-slot
  sums, normalize locally, and compute the head mix row-parallel. Expected gain
  is smaller than per-layer A2 because it runs once per decode step, but it is a
  clean `1-2%` candidate and removes another GPU0-centralized boundary.

## Recommended priority

| # | Idea | Domain | Why | Risk |
|---|---|---|---|---|
| Done | A4 finish rank-major consumers | HC 40% | Sprint 526 completed the remaining post-attention FFN shared/route consumers for the served path | Low |
| 1 | D1 output-head A1 pattern | Model boundary | Same proven A2 template; removes GPU0-centralized output-head norm/mix and hard host syncs | Low |
| 2 | C5 sync-point reduction | both | Removes host round-trips and makes C1 graph capture structurally possible | Low-Med |
| 3 | B2 compact EP variable-size NCCL compose | EP 53% | Targets served compact traffic and removes remaining peer-copy-equivalent compose movement | Med |
| 4 | C1/C2 piecewise graph capture and serving parity | both | Highest ceiling, but only after the surface is simplified | Med-High |
| 5 | A5/A6 fusion | HC/attention | Converts rank-local structure into fewer launches | Low-Med |
| 6 | B3/B4/B5 EP structural bets | EP 53% | TP-expert A/B, routed/shared overlap, and correctness-preserving capacity balancing | Med |
| Deferred | B1 MTP | EP 53% | Useful later, but do not use it to hide base TP/EP bottlenecks | Med |

## Discipline (unchanged)

- A/B every idea on the steady-state reference above, **parity-gated** (first
  token unchanged + tolerance) — not on reduced/short shapes that flatter results.
- Report **server decode tok/s AND GPU util** for control vs candidate.
- No re-baselining; use the latest promoted artifact as the control unless a
  real invalidator makes it non-comparable.
- No micro-opt before the change is justified by the profile.
- Re-profile the domain table after each promoted change (shares shift).
- One-off smokes and temporary gates are for evaluation only. A promoted result
  moves into the main code/default path and deletes the smoke-only flag,
  rejected branch, or diagnostic scaffold in the same sprint unless there is a
  documented debugger-only reason to keep it default-off.

## One-line frame

With A4 complete for the served path, remove the remaining output-head,
host-sync, and compact EP compose friction before attempting C1. C1 is still
the biggest ceiling, but it should run on the simplest fully rank-major, mostly
NCCL, sync-reduced surface we can make. MTP stays deferred.
