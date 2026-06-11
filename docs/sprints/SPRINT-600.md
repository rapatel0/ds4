# Sprint 600 - Root-Cause the Swiglu-Exchange Ordering Hazard; Re-Adjudicate 220

Date: 2026-06-11
Status: planned

## Goal

Sprint 599 proved a timing-dependent ordering hazard: every faster
shared-swiglu exchange (including bit-identical byte mechanics at 1/32 the
launches) fails the tolerance gate while gaining +11.5-17.9% decode-domain.
The promoted path is correct only because its 1,792-launch exchange is slow
enough to win an unprotected race. This sprint:

1. **Root-causes the hazard** — names the exact missing dependency edge (or
   buffer-reuse WAR/RAW hazard) with evidence.
2. **Fixes it in the promoted path** — this is a correctness fix
   independent of perf; the promoted path carries the same latent debt.
3. **Re-applies the demonstrated levers** (fast swiglu exchange, C-B
   restack, C-C route-plan shadow) on top of the fix.
4. **Re-adjudicates the 220 decode-domain stretch target** — the loop's
   confirmation question, round 2.

## Evidence base (Sprint 599)

- Fast-exchange tolerance failures: nccl allgather 0.781/0.935 @ 197.17
  tok/s; batched remote-load (identical mechanics, fewer launches)
  0.922/0.954 @ 186.33. Per-variant probes in
  `logs/from-cluster/sprint599/`.
- C-B (early return + per-rank ordering) is tolerance-clean — whatever
  races, races against the swiglu exchange specifically.
- The exchange lives in `materialize_shared_swiglu_down_input`
  (`engine/ep_dense.cu`); consumers are the shared-down dense launch and
  the compose chain (`engine/decode_loop.cu` overlap block).

## Root-cause methodology (Phase A)

In rough order of expected yield; stop when the edge is named and proven:

1. **Static dependency audit**: map every buffer the exchange writes
   (`d_shared_mid` staging et al.) and every reader/writer of those buffers
   in the same and next layer-step, per stream, in the captured graph.
   Candidate hazards: (a) consumer on another stream/rank reads the
   exchanged buffer without an event edge from the exchange ops; (b) WAR -
   the fast exchange overwrites staging the previous step/stage still
   reads; (c) cross-layer reuse without a fence.
2. **Graph topology diff**: `cudaGraphDebugDotPrint` of the captured graph
   under `copy` vs `batched` - diff the edge sets around the swiglu and
   shared-down nodes; a missing edge present in neither (but enforced by
   launch-order serialization in `copy`) is the smoking gun.
3. **Delay-injection bisect**: flag-gated busy-wait kernels inserted at
   candidate points in the FAST variant; the injection point that restores
   1.0/1.0 tolerance localizes the race window. (Probes default-off,
   s599-style discipline.)
4. **First-divergence localization**: tolerance tool says WHICH slots/steps
   diverge first; correlate with rank/slot ownership of the racing buffer.

## Fix + verification (Phase B)

- Add the missing edge (event or graph dependency) in a form that is
  correct for ALL exchange variants, including promoted `copy`.
- Gates for the fix itself: tolerance 1.0/1.0 on `copy` (unchanged
  behavior), AND on `batched`, AND on `nccl`; then a **timing-perturbation
  stress** - N runs (>= 5) of the fast variant with randomized flag-gated
  delay/jitter injection at the previously-racing points, all clean. The
  fix must not measurably regress the promoted leg (within run band).

## Re-stack + re-adjudication (Phase C)

1. Promote the best fast swiglu exchange (expect `nccl` allgather: removes
   the SYS remote loads too - fold a no-SYS spot-check into the gate).
2. Restack C-B (its barrier-edge win may matter in the new wait structure);
   attempt C-C route-plan shadowing (0.52 ms pool) if budget remains.
3. Final reference-shape measurement, same harness, in-band control.
4. **Verdict**: >= ~220 decode-domain confirmed, or the updated floor
   analysis with the explicit reachable/not-reachable call and what bounds
   it now.

## Definition of Done

1. The hazard is named: exact buffer + exact missing edge + mechanism,
   with the evidence artifact (audit/dot-diff/bisect) archived.
2. The fix lands in the promoted path; `copy` leg tolerance 1.0/1.0 and
   perf within run band.
3. Fast-exchange variants pass tolerance after the fix; perturbation
   stress (>=5 jitter runs) clean.
4. Promotions: best exchange variant flipped in the launcher only if all
   gates (tolerance, no-SYS for the stage, perf >= +8% over the fixed
   promoted leg) pass; rollback flags retained.
5. Final stack measured; **220 verdict round 2 stated explicitly** with
   floor analysis if short.
6. Report (SPRINT-600-REPORT.md), follow-ups, orchestrator doc updates,
   commits per convention.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| The hazard is multiple independent races | Med | Med | Bisect + first-divergence localization separate them; fix iteratively, gate each |
| The fix serializes what the overlap design needs concurrent | Med | Med | Express the edge at per-pair/per-buffer granularity, not a global barrier (the s599 C-B lesson) |
| Root-cause stalls (race not reproducible under instrumentation) | Low-Med | High | The fast variants reproduce it deterministically at the reference shape (0.78-0.95, not flaky-rare); jitter probes widen the window |
| Fix changes promoted-path token stream (it was winning the race - fixing order could legitimately change nothing; if it changes tokens, the OLD path was emitting racy output) | Low | High | If `copy` tokens change under the fix, treat as a correctness finding: the fixed output becomes the new control after CPU-reference spot-verification; document loudly |

## Dependencies

- s598/s599 environment intact on gpu-01 (pack, contract, control, fixed
  harness, profiler, all opt-in flags).
- `SPRINT-599-REPORT.md` + `logs/from-cluster/sprint599/` forensics.
- `SPRINT-599-FOLLOWUPS.md` items 1-4 (this sprint executes #1/#4, attempts
  #2, restacks #3).
