# Sprint 604 - Root-Cause and Fix the Rank↔Dense Ordering Hazard

Date: 2026-06-13
Status: planned

## Goal

Identify, control, and fix the rank↔dense CUDA-stream ordering hazard that
s603 isolated. This is the gating correctness bug for the whole throughput
program: it is the only thing keeping the fast configs (edges, relay+batched)
from promoting, and the only event-free config today is the full rank+dense
barrier (`fb`) at a 2.1x cost. A real fix makes ordering correctness free,
unlocks the demonstrated ~208 ceiling, and removes the "rising correctness
bar" that bounded the s598→603 gains curve.

This is a debugging sprint, not a perf sprint. Success = the hazard is named,
made deterministic, fixed with a minimal dense→rank edge (NOT a full barrier),
and proven under both a deterministic amplifier and a ≥50-run soak with pod
telemetry.

## What s603 established (the starting evidence)

- The hazard is a **rank↔dense (or copy↔dense) stream ordering gap**, NOT a
  rank-stream sync gap: event rate is monotone in step speed and indifferent
  to which rank-stream sync point is strengthened; only `fb` (joins the dense
  streams) is clean (6/6 bit-exact).
- It is almost certainly **pre-existing** (the s60x promoted-path "latent
  event" class fired on the NCCL path too) — exposed, not introduced, by the
  speedups. So the audit must cover the WHOLE captured layer, not only the
  s602 sites.
- **Falsified**: the ffn_bcast-site `d_current_full` dense-WAR
  (`DENSE_GUARD=1`). The carrier is a different interaction.
- **Two event classes**: late-step flicker (steps ~53-63, mostly
  checksum-only) and an early-locus token-level class (step 1, 19/32 slots;
  also seen step 42). Treat as possibly-distinct carriers.
- **Signature**: batch-wide (every event marks all 32 slots at one step → a
  shared/non-per-slot buffer), variable onset, rate scales with pacing.
- **Confound**: identical configs moved 3-5x in event rate across two days —
  pod thermal/clock/contention state matters; small-n gates are unreliable.

Existing tooling: `DS4_V100_TP_EP_S602_FULL_BARRIER=1` (the clean bracket),
`DS4_V100_TP_EP_S602_DENSE_GUARD=1|2`, per-point `_E0/_E1/_E2` overrides, the
s600/s602 jitter-injection + per-step checksum + first-divergence tools.

## Plan

### Phase A — CONTROL: a deterministic amplifier (do this FIRST)

The current detector fires ~0.17-0.83 events/256-step run, so every gate is a
multi-run soak and the day-to-day confound corrupts it. Before hunting, build
a knob that makes the hazard fire on (nearly) every run, converting the gate
from "≥6 runs, statistical" to "1 run, deterministic":

- Use the s600 finding inverted: delay injection at the racing site restores
  order; a delay that *widens* the dense↔rank window should make the event
  near-certain. Add `DS4_V100_TP_EP_DENSE_HAZARD_AMP=<us>` (default 0): a
  flag-gated busy-wait inserted on the dense stream (or rank stream) at
  candidate hand-off points, sized to maximize event rate.
- Validate the amplifier on the KNOWN-exposed config (edges, no fix): tune
  until the event fires ≥ ~90% of single runs at the reference shape, ideally
  separately maximizing each of the two event classes.
- Gate criterion for the rest of the sprint: a candidate fix is "clean" only
  if it drives the amplified event rate to zero AND passes a ≥50-run soak at
  the un-amplified reference shape.

If a robust amplifier cannot be built (the hazard resists widening), fall back
to the ≥50-run soak as the gate and say so — but the amplifier is the
sprint's force multiplier; spend real effort here.

### Phase B — IDENTIFY: localize the carrier

Two convergent methods; intersect them.

1. **Per-site full-barrier bisect** (empirical): `fb` joins dense streams at
   ALL s602 sites and is clean. Add per-site dense-join overrides (the fb
   bracket at single-site granularity — extend the existing `_E*` override
   pattern to the dense join). Under the amplifier, find the minimal set of
   sites whose dense-join closes the event. If NO s602-site combination
   closes it, the carrier is outside the s602 sites (expected, given the
   pre-existing-bug hypothesis) → method 2 dominates.
2. **Static cross-stream audit** (the orchestrator is running a parallel
   codex concurrency audit; its ranked candidate list will be delivered to
   you — incorporate it): every buffer touched by both a dense-stream and a
   rank/copy-stream kernel in one layer-step without an intervening event.
   Prime suspects from s603: the attention-output buffer consumed by the
   same step's FFN-input prefix; the shared-FFN swiglu intermediates
   (dense) vs the routed path; `d_next_hidden` / compose; `d_ep_sum` dense
   readers.

For each candidate, confirm with the amplifier (inject the widening delay
specifically at that hand-off; if the event rate spikes, that's the window)
and with first-divergence localization (which buffer's slots flip).

### Phase C — FIX: the minimal dense→rank edge

Once the carrier is named: add the narrowest correct ordering — one event
record on the producer stream + targeted waits on the consumer stream(s)
that touch the buffer — NOT a full barrier, NOT a rank-stream-only edge.
Graph-capturable, pre-allocated events, flag-gated
(`DS4_V100_TP_EP_DENSE_FIX=0|1`, default 0 until proven). If there are two
event classes, fix both (they may need two edges).

### Phase D — PROVE

1. Amplified rate → zero with the fix on (each event class).
2. ≥50-run soak at the reference shape, un-amplified, zero token events,
   checksum events at or below the fb baseline (0/6), WITH pod telemetry
   (clocks, temp, power, ECC, concurrent-process check) captured per run so
   the day-to-day confound is explained not just noted.
3. Tolerance 1.0/1.0 vs the s597 control.
4. The fix composes with edges + relay + batched: re-run that stack with the
   fix and confirm it is now event-clean at full speed.

### Phase E — PROMOTE + restate

If proven: promote the fix (default on); re-evaluate edges promotion now that
its correctness gate passes; flip the binary defaults to match the launcher
(s602 follow-up #4). Re-measure the reference shape and the ≥50/slot budget +
required-MTP-multiplier on the now-clean fast base. This is the number that
reopens the MTP gate.

## Definition of Done

1. Deterministic amplifier built and validated (or documented infeasible with
   the soak fallback justified).
2. The carrier named: buffer, the two kernels/streams, RAW/WAR/WAW, file:line,
   confirmed by both the bisect and the amplifier; both event classes
   addressed.
3. Minimal dense→rank fix implemented, flag-gated, graph-capturable; flag-off
   byte-identical.
4. Proof: amplified rate zero + ≥50-run soak zero-token (with telemetry) +
   tolerance 1.0/1.0 + composes with the fast stack.
5. Promotion decision with the gate matrix; if promoted, launcher AND binary
   defaults updated, rollbacks retained.
6. Re-measured floors + ≥50/slot + MTP-multiplier restatement on the clean
   base.
7. Report, follow-ups, orchestrator docs/commits.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| No robust amplifier (hazard won't widen deterministically) | Med | High | Soak fallback (≥50 runs) as the gate; still cheaper than guessing; the static audit narrows candidates so fewer soaks needed |
| Multiple independent carriers (the two event classes are different bugs) | Med | Med | Localize and fix each separately; the amplifier tuned per-class separates them |
| fb is clean by global masking, not by ordering a specific edge (no minimal fix exists) | Low-Med | High | Then the per-site bisect finds the minimal dense-join SET that suffices; even a 2-3 site dense-join is far cheaper than full fb |
| Day-to-day pod variance corrupts even the soak | Med | Med | Telemetry capture + alternating A/B (fix-on vs fix-off interleaved same session) so the confound hits both arms equally |
| The carrier is in a kernel outside the allowed edit surface | Low | Med | The fix is an ordering edge in the launch sequence (decode_loop / runtime_pack), not the kernel body — surface is correct |

## Dependencies

- HEAD f45e5aa2; pod environment intact (zero-NCCL launcher default,
  S602_SYNC=join, pack/contract/control on /workspace, 16 Gi shm).
- s603 tooling (fb bracket, DENSE_GUARD, per-point overrides, jitter +
  per-step checksum + first-divergence).
- The parallel codex concurrency audit (delivered by the orchestrator).
- s603 report Phase A edge table + the rank↔dense analysis (lines ~270-326).
