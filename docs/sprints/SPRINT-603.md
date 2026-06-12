# Sprint 603 - Join Reclaim on the Zero-NCCL Graph

Date: 2026-06-13
Status: planned

## Goal

The zero-NCCL stack is now the correctness default (s602: token-race-zero,
bit-exact, 6/6 census) but pays ~1.5 ms/layer for its 16 all-rank
rank-stream joins — masking the relay+batched gains (153 vs 208
demonstrated, vs the racing 169 it replaced). This sprint reclaims that
cost with **per-collective producer-consumer dependency edges**: each
consumer kernel waits only on the producers of the buffers it actually
reads, instead of all-rank joins at every site.

Targets: ≥194 decode-domain (the s602 promotion bar) with token-race-zero
preserved; stretch 208+ (the s601 demonstrated ceiling, now correctness-
clean). Secondary: hunt the residual checksum-only late-step flicker; soak
decision for flipping the binary defaults; prefix-compaction scoping if
budget remains.

## The cautionary tale (read before designing edges)

s602's first weak-ordering attempt (pairwise NVLink-peers-only) produced a
NEW race (30x divergence mass, token-level under stress). The discipline:
- Derive each collective's TRUE read/write set from the kernel arguments
  (the s602 collectives are in `engine/runtime_pack.cu` — ours, fully
  known), not from topology intuition.
- Ring-order-exact folds have a sequential dependency chain along the ring;
  an edge set weaker than the fold order is wrong by construction.
- Gate EVERY ordering variant with the s602 census methodology before
  trusting its perf number: ≥3 pairwise 256-step Simple-stress runs + LL
  census ≥6 runs (token events must be ZERO; checksum-only events ≤ the
  s602 baseline 0.17/run) + tolerance 1.0/1.0 vs the s597 control.

## Plan

1. **Phase A — dependency-graph derivation**: per s602 collective class,
   write down the per-rank producer→consumer buffer map (what each fold
   step/copy reads and writes, in ring order). Output: an explicit edge
   table reviewed against the code, archived in the report. Then implement
   `DS4_V100_TP_EP_S602_SYNC=join|edges` (default join — the current
   correctness default stays until edges pass everything).
2. **Phase B — race gates on edges** (the s602 methodology, escalating):
   Simple-stress pairwise ×3, LL census ×6 + cross-compares, tolerance.
   Any token event = stop, bisect the missing edge with the jitter tools.
3. **Phase C — perf + promotion**: reference-shape A/B vs the join default;
   promote `edges` only on race-gates-clean + ≥+15% over the join baseline
   (~153 → ≥176; expect ≥194 if the full join cost reclaims). Stage tables
   confirm where the 1.5 ms went.
4. **Phase D — the flicker hunt** (budget permitting, same code region):
   full-barrier control with n≥6 (the n=1 zero had P≈0.18 of luck),
   per-site jitter bisect of the late-step checksum-only events.
5. **Phase E — program restatement**: S=1/8 step floors on the final
   config; updated ≥50/slot budget + required MTP multiplier; binary-
   default soak recommendation; prefix-compaction scoping measurement.

## Definition of Done

1. Edge table derived from code and archived; `edges` mode implemented
   behind the flag, default unchanged until promotion.
2. Race gates: zero token events at every stage (census evidence); flicker
   rate vs the 0.17/run baseline reported.
3. Perf verdict with stage tables; promotion decision per gates (launcher
   default flip + rollback if passed; evidence if not).
4. Flicker hunt result or explicit deferral with budget reason.
5. Updated ≥50/slot statement (step floors, MTP multiplier at the new
   floor); binary-default soak recommendation.
6. Report + follow-ups; orchestrator docs/commits.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Edge set subtly weaker than the fold order (new race) | Med | High | Derive from code, not topology; census-gate every variant; the join default stays until proven |
| Reclaim is partial (edges still serialize the ring chain) | Med | Med | The ring fold IS sequential — overlap comes from letting OTHER streams run during it, which edges enable and joins forbid; stage tables verify |
| Flicker worsens as pacing tightens | Med | Med | Census thresholds are hard gates; Phase D hunts it in the same sprint |

## Dependencies

- HEAD 9b261c9b (zero-NCCL stack as launcher default); pod environment
  intact; s602 census/jitter/verifier tooling; s602 report's per-class
  collective inventory.
