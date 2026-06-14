# Sprint 606 - Microbatch Ping-Pong: Overlap the Transport-Bound Step

Date: 2026-06-14
Status: planned

## Goal

Cut the decode step floor by overlapping cross-GPU transport/collective work
with compute, via 2-microbatch ping-pong: split the S active slots into two
half-batches A and B, and run them through the layer pipeline phase-shifted so
microbatch A's transport-bound prefix (attention-output allgather) and EP
collective window overlap microbatch B's GEMM compute, and vice versa.

This is the campaign's highest-ceiling lever and — after s605 — clearly the
RIGHT one: s605 proved the step is ~95% launch/sync/transport and the heaviest
prefix stage is DMA-transport-bound (gather/launch compaction REGRESSED it).
A transport-bound stage already on the NVLink copy-engine fast path cannot be
made cheaper; it can only be HIDDEN behind compute. Microbatch is the
mechanism. VRAM feasibility was resolved in s605 (~180× headroom).

Target: step floor 199 ms (S=8) toward the ~40-60 ms MTP-reachable floor.
Microbatch's ceiling is ~2× if transport and compute are balanced; realistic
first-cut goal is a measured, correctness-clean step reduction (>+15% decode-
domain) — full balancing is iterative. This sprint does NOT have to reach the
floor; it has to prove microbatch works on this graph and bank its first gain.

## The hard part (why this is a dedicated sprint)

The decode layer is captured as one monolithic CUDA graph per layer
(full-capture). Microbatch ping-pong needs either:
- **(a) Graph-split**: capture the layer as two sub-graphs (prefix/transport
  vs compute) and interleave A/B replays with cross-microbatch event ordering,
  OR
- **(b) In-graph choreography**: capture both microbatches' streams in one
  graph with the phase-shift expressed as intra-graph event edges.

Plus: the router/route-count/route-plan is currently whole-batch — it must be
split per-microbatch (each half-batch routes independently; the EP expert
GEMMs then see S/2 tokens each). The relay/edges EP-return and the s604
dense→rank fix must be re-derived for the half-batch shapes.

Approach: prefer (b) if the capture machinery allows two stream-sets in one
graph (lower replay-loop disruption); fall to (a) if not. Either way the
ordering is the risk — every cross-microbatch and dense↔rank edge must be
explicit, and the amplifier is the gate.

## Plan

### Phase 0 — Re-verify base + the amplifier-as-gate

Recreate the pod if degraded (s605 ran ~23h; /workspace persists). Re-measure
the promoted edges+fix floor (S=8 199 ms / S=32 222 ms) as the in-band control.
Confirm the amplifier still fires on the un-fixed config and is clean on the
promoted one (sanity that the gate is live).

### Phase A — Microbatch plumbing (default-off, byte-identical)

`DS4_V100_TP_EP_MICROBATCH=1|2` (default 1 = current whole-batch). At 2:
- Split slots into A=[0..S/2), B=[S/2..S) at admission.
- Per-microbatch router/route-count/route-plan (the EP path already sizes by
  rank.routes; feed it the half-batch route set).
- Per-microbatch activation/staging buffers (the ~21 MiB/rank set, doubled —
  VRAM-confirmed).
- Bring-up at S=8→2×4 with the bit-verifier: microbatch=2 must produce
  byte-identical tokens to microbatch=1 (it's the same math, just partitioned)
  BEFORE any overlap is enabled. This is the correctness anchor.

### Phase B — The overlap (the actual lever)

With A/B correct but still sequential (Phase A), introduce the phase-shift:
A's prefix/transport stage runs while B's previous-stage compute runs, via
the chosen graph structure. Each new cross-stream/cross-microbatch edge is
amplifier-gated (1-run) the moment it's added — any missing edge reopens a
hazard, and the amplifier catches it deterministically.

### Phase C — Gate + measure

- Correctness: amplifier 1.0/1.0 at the carrier sites under microbatch=2+
  overlap; ≥20-run soak token+ck clean; tolerance 1.0/1.0 vs the s597 control.
- Perf: reference-shape A/B (microbatch=1 vs 2) at S=8/16/32; decode-domain +
  per-slot; the stage table showing transport now overlapped with compute.
- Promote microbatch=2 default only if correctness-clean AND >+15%
  decode-domain; else keep opt-in and record the gain + the remaining
  balancing work.

### Phase D — Restate

Floors + per-slot curve on the final config; updated ≥50/slot + required-MTP
multiplier; captured-vs-remaining accounting; what the next overlap/balancing
iteration (or route-plan shadow) should target.

## Definition of Done

1. Microbatch plumbing (default-off); microbatch=2 sequential proven
   byte-identical to =1 (the correctness anchor) before overlap.
2. Overlap implemented; every cross-microbatch/dense↔rank edge amplifier-gated.
3. Gate matrix: amplifier + soak + tolerance; A/B perf at S=8/16/32.
4. Promotion decision; if promoted, default flip + rollback; else opt-in +
   recorded gain.
5. Restated target math with captured-vs-remaining; next-lever recommendation.
6. Report, follow-ups, orchestrator docs/commits.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Monolithic per-layer capture resists splitting/choreography | High | High | Phase A proves the plumbing sequential-correct first; if neither (a) nor (b) is tractable this sprint, that itself is the finding — fall to route-plan shadow + rendezvous merge (the no-graph-restructure levers) and re-scope microbatch |
| New cross-microbatch edge reopens a hazard | Med-High | Med | The amplifier is the 1-run gate on every edge added; bring-up byte-identical before overlap |
| Overlap gain < expected (transport & compute imbalanced) | Med | Med | First-cut goal is >+15%, not 2×; balancing is iterative across 606/607 |
| Half-batch EP shapes hit a route-count edge case (zero-route half) | Med | Med | The fixed-capacity route plan already handles zero-route ranks; test the all-tokens-to-one-half skew explicitly |
| VRAM (doubled activation set) | Low | Low | s605 resolved: ~21 MiB/rank vs ~3.9 GiB free |

## Dependencies

- HEAD 2dd71a66 (edges+fix promoted, amplifier, gather8 opt-in, s605 report).
- s605 decomposition (the transport-bound finding + the floor controls).
- The s604 amplifier (DENSE_HAZARD_AMP) as the per-edge correctness gate; the
  bit-verifier for the sequential-correctness anchor.
- Pod (recreate if degraded; /workspace persists).
