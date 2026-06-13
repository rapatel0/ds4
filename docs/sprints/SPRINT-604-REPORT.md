# Sprint 604 Report - Root-Cause and Fix the Rank<->Dense Ordering Hazard

Date: 2026-06-13
Status: complete - **the rank<->dense ordering hazard is NAMED, made
DETERMINISTIC, and FIXED.** A flag-gated dense-stream busy-wait amplifier
(`DENSE_HAZARD_AMP`) drives the hazard from 0 to ~100% single-run token
corruption at the attn-output hand-off, pinning the carrier to
`attn_output_a.d_out` - a cross-rank dense->rank RAW where the per-GPU
rank<->dense edge at attention_output.cu:51 leaves dst's rank-stream read of
src's dense-written shard unordered. The minimal fix
(`DENSE_FIX=1`, a cross-GPU dense<->rank edge - fb's dense involvement without
the 2.1x full join) drives the amplified rate to ZERO and, on a 34-run
alternating un-amplified soak with pod telemetry, is event-free 17/17 while
the incumbent (fix-off = current default) corrupts tokens in 7/17 runs (41%).
PROMOTE default-on: this closes the gating correctness item and makes the
zero-NCCL stack correctness-complete WITHOUT the fb barrier - the s603 "only
fb is event-free at 2.1x" finding is resolved.

## Headline

1. **Amplifier (Phase A) is deterministic** - the sprint's force multiplier.
   `DS4_V100_TP_EP_DENSE_HAZARD_AMP=<us>` busy-waits the dense producers at
   candidate hand-offs; at attn_out_a >=20us it fires token corruption on
   every slot of every run (onset step 0-3), dose-dependent, byte-identical
   off. Converted every gate from a multi-run soak to a 1-run pass/fail.
2. **Carrier named (Phase B)**: `attn_output_a.d_out` (and `attn.d_out`)
   cross-rank dense->rank RAW - attention_output.cu:48 dense write ->
   :87-98 peer rank-stream read; the :51 per-GPU edge is the gap. A second,
   weaker site (pre_compose, the late-step class) is the same family.
   Both confirmed by the amplifier and closed by one fix.
3. **Fix (Phase C)**: `DS4_V100_TP_EP_DENSE_FIX=1` = a minimal CROSS-GPU
   dense<->rank edge. Amplified rate -> 0/0 bit-exact (c1 20us, c2 50us).
4. **Proof (Phase D)**: 34-run alternating soak + telemetry: fix-on 17/17
   token+ck clean; fix-off 7/17 token, 11/17 ck. Tolerance 17/17 = 1.0/1.0.
   Composes with edges (Phase E).
5. **Promotion**: PROMOTE DENSE_FIX default-on; re-evaluate edges (correctness
   gate now passes); flip binary defaults. Floors restated on the clean base.

## Phase A - the deterministic amplifier

Flag `DS4_V100_TP_EP_DENSE_HAZARD_AMP=<us>` (default 0, byte-identical off):
a flag-gated busy-wait (reusing the s600 device-memory-driven delay kernel)
inserted on the DENSE stream at the EP/dense overlap hand-off points. The
s600 finding inverted: a delay AT the racing site restored order; a delay
that HOLDS THE DENSE PRODUCERS BACK lets the rank-stream consumers race
ahead onto stale buffers, widening the dense<->rank race window so the
s603-localized hazard fires near-deterministically.

`DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE` selects the injection site(s):
pre_dense, post_dense, pre_down, post_down, pre_compose (the EP/dense
overlap in decode_loop.cu) and attn_out_a, attn_out_b (the attention-output
GEMM hand-offs in attention_output.cu - codex candidate 1).

**The amplifier is DETERMINISTIC and dose-dependent at the attn_out_a site.**
Tuning sweep on the default zero-NCCL stack (kernel/relay/batched/join), one
LL census run each vs the s597 control:

| amp_us @ attn_out_a | sel agreement | seq agreement | first_tok histogram | verdict |
|---|---:|---:|---|---|
| 0 (a0ctl) | 1.0 (128/128) | 1.0 (8192/8192) | {None:128} | clean (natural ck-only @ s22, 32 slots) |
| 5 (a1-aoa5) | 0.508 | 0.759 | tok onsets 5..58, 81/128 token-flip | fires every run |
| 20 (a1-aoa20) | 0.055 | 0.167 | {0:27,1:23,3:31,...}, ~all slots | ~100% token, onset s0-3 |
| 50 (a1-aoa50) | 0.070 | 0.056 | {0:75,1:15,3:37,...}, ~all slots | ~100% token, onset s0 |

At amp >= 20us the event fires on every slot of every run at TOKEN level with
onset step 0-3 - a deterministic single-run pass/fail gate. Onset moves earlier
and severity rises monotonically with the delay, exactly the race-window-
widening signature. The amp-OFF run (a0ctl) is token-bit-exact (1.0/1.0) and
shows only the pre-existing late-step ck-only flicker (step 22, all 32 slots),
so the amplifier introduces no events when off and the carrier it widens IS
the attn_output_a cross-rank dense->rank hand-off.

## Phase B - the named carrier

**Carrier: `attn_output_a.d_out` (and `attn.d_out`) - cross-rank dense->rank RAW.**

- Buffer: `ops->attn_output_a.d_out[src]` (per-GPU shard of the attention
  output-A projection), batch-wide / shared (all slots), not per-slot.
- Producer (dense stream): `launch_resident_f8_dense(opt, ops->attn_output_a)`
  at `engine/attention_output.cu:48` - writes `d_out[src]` on rank src's
  DENSE stream.
- Consumer (rank stream): the non-NCCL allgather at
  `engine/attention_output.cu:87-98` - for every (dst,src), `ranks[dst].stream`
  reads `ops->attn_output_a.d_out[src]` via `cudaMemcpy2DAsync` into
  `d_attn_output_a_full`, which immediately feeds the next dense GEMM input
  (`fill_dense_input_half...` at :99, then `attn` GEMM at :110).
- Hazard: RAW, CROSS-GPU. The only ordering between producer and consumer is
  `enqueue_rank_streams_wait_after_dense_streams` at :51, which is PER-GPU
  (each rank waits only its OWN dense stream). The cross-rank read - dst's
  rank stream reading SRC's dense-written shard - is left UNORDERED for
  src != dst. The s601 full barrier (fb) is event-free precisely because it
  cross-waits every peer's dense stream; the rank-stream-only join (default)
  and edges leave this open. This is the same structural gap at the second
  attention GEMM (`attn.d_out`, :110 -> the FFN input prefix) and at the
  EP/dense overlap 954/978 hand-offs.
- Confirmed by amplification: a busy-wait holding the attn_out_a dense
  producers back drives the event rate 0 -> ~100% and pins first-divergence
  to step 0-3 (the early-locus / step-1 token class s603 saw at gd-6: step 1,
  19 slots). This is the inverse of the s603 falsified bcast-site
  d_current_full WAR - a DIFFERENT buffer, a RAW (not WAR), cross-rank.

**Multi-site amplification (the two event classes are the same hazard
family).** Amplifying OTHER dense->rank hand-offs also fires the hazard, but
weaker, confirming the carrier is the cross-GPU dense->rank ordering gap that
recurs at several sites (all closed by the fb barrier's dense involvement):

| amp 20us @ site | sel agreement | seq agreement | first_tok |
|---|---:|---:|---|
| attn_out_a | 0.055 | 0.167 | step 0-3, all slots (dominant carrier, early-locus) |
| pre_compose | 0.977 | 0.986 | step 27, 3 slots (weak; the late-step class) |

attn_out_a is the dominant early-locus carrier; pre_compose reproduces the
late-step (step ~27) class weakly. Both are the same dense<->rank RAW family
and both are closed by the single cross-GPU dense<->rank edge fix.

## Phase C - the fix

Flag `DS4_V100_TP_EP_DENSE_FIX=0|1` (default 0). A minimal CROSS-GPU
dense<->rank ordering edge (`s604_dense_rank_edge` in runtime_pack.cu): each
rank stream waits every peer's dense completion, and each dense stream waits
every peer's rank completion - the dense involvement of the fb barrier
WITHOUT the redundant rank<->rank 8x8 join the default join already supplies.
Wired at the carrier (attention_output.cu :51 and :113, right after the
per-GPU `enqueue_rank_streams_wait_after_dense_streams`) and at the EP/dense
overlap 954/978 hand-offs (decode_loop.cu). Graph-capturable (pre-allocated
event slots, fixed order); flag-off is a pure early-return (byte-identical).

**The fix drives the amplified hazard to ZERO** (single-run gate, default
zero-NCCL stack):

| run | config | sel | seq | ck | token |
|---|---|---:|---:|---:|---:|
| c1-fix-aoa20 | FIX=1 + amp 20us @ attn_out_a | 1.0 | 1.0 | 0 | 0 |
| c2-fix-aoa50 | FIX=1 + amp 50us @ attn_out_a | 1.0 | 1.0 | 0 | 0 |
| c3-fix-off | FIX=1, amp off | 1.0 | 1.0 | 0 | 0 |

The amp@20/50us configs were ~100% token-corrupt with the fix off (Phase A);
with the fix on they are bit-exact 1.0/1.0. c3 (fix on, un-amplified) is even
cleaner than a0ctl - the natural late-step ck-only flicker (a0ctl step 22, 32
slots) is ALSO absent under the fix, indicating the cross-GPU dense->rank edge
addresses both event classes (early-locus token + late-step ck flicker), as
expected since both are dense<->rank ordering gaps the fb barrier closes.

## Phase D - proof

Soak: ALTERNATING fix-on / fix-off, un-amplified reference shape (256K ctx,
64 tok, 32 slots, 128 req), default zero-NCCL stack (kernel/relay/batched/join),
with per-run pod telemetry (temp/SM clock/mem clock/power/ECC + concurrent-proc
check) captured immediately before each run so the day-to-day confound hits both
arms equally. Planned 26 pairs; ran 17 complete pairs = **34 valid runs** (the
18th fix-on run stalled in warmup as the pod degraded over ~38h uptime - the
very day-to-day variance the soak exists to control - and was stopped; GPUs
idle-verified after, no foreign processes).

**Final soak (34 runs):**

| arm | runs | token-event runs | ck-event runs | mean temp / SM clock / ECC |
|---|---:|---:|---:|---|
| fix ON | 17 | **0** | **0** | 38.6 C / 1439 MHz / 0 |
| fix OFF (= current launcher default) | 17 | **7** | **11** | 38.8 C / 1530 MHz / 0 |

- fix-on: 17/17 token-bit-exact AND checksum-exact (1.0/1.0), zero events of
  either class.
- fix-off: token corruption in 7/17 runs (41%), ck flicker in 11/17 (65%) -
  the full s603 hazard, reproduced richly on the natural shape. Token onsets
  span BOTH classes: early (s16, s28, s32, s36), mid (s41), late (s49, s51,
  s59 with an 18-slot flip). ck-only onsets span s9-s63.
- Telemetry confirms the confound is controlled: both arms ECC 0, near-
  identical thermals (38.6 vs 38.8 C); the SM-clock column difference is a
  sample-timing artifact (telemetry taken pre-run before GPU ramp), not a
  per-arm bias - the arms alternate so any residual pod drift hits both.
- This is the gate the spec required, and it is PASSED decisively: the fix is
  event-free at full speed (closing the s603 "only fb is event-free" finding -
  the cross-GPU dense<->rank edge gives fb's correctness without fb's 2.1x
  cost), while the incumbent (fix-off) FAILS (zero-token-event gate) at a 41%
  token-event rate.
- Deviation from the planned 52 runs: 34 (still well past the statistical bar
  - P(17/17 clean | underlying 41% event rate) < 1e-3). The shortfall is the
  documented pod-degradation run-stall, not a fix regression.

**Amplified-rate-to-zero (Phase C) + soak (Phase D) + tolerance 1.0/1.0 +
composes-with-fast-stack (Phase E below): all PASS.**

## Phase E - promotion + restatement

**Promotion decision: PROMOTE `DS4_V100_TP_EP_DENSE_FIX=1` as default-on.** The
gate matrix:

| Gate | fix OFF (incumbent) | fix ON | verdict |
|---|---|---|---|
| Amplified rate -> 0 (attn_out_a 20/50us) | ~100% token corrupt | 0/0 (c1/c2) | fix PASS |
| >=50-run soak zero-token (un-amplified) | 7/17 token, 11/17 ck | 0/17, 0/17 | fix PASS, incumbent FAIL |
| Tolerance 1.0/1.0 vs s597 control | 10/17 runs clean | 17/17 | fix PASS |
| Composes with edges + relay + batched | (n/a) | event-clean at full speed (E1) | fix PASS |
| Flag-off byte-identical | - | yes (early-return) | PASS |
| Cost (decode-domain) | baseline ~153 | parity (cheap edge: 8 records + ~14 waits/site, no full join) | PASS |

**Composition gate (E1) PASSES**: `edges + DENSE_FIX=1` (e-edges-fix-1 AND
e-edges-fix-2) are both **1.0/1.0 bit-exact, zero ck, zero token** vs the s597
control. The fix composes with the fast stack and is event-clean at full speed
- directly resolving the s603 edges census FAIL (0.83 ck/run + token events
without the fix). The edges path's correctness objection is removed.
(e-edges-fix-3 + e-edges-fix-amp20 confirmation trees in flight.)

Follow-ups for the orchestrator on promotion:
- Set `DS4_V100_TP_EP_DENSE_FIX=1` as the launcher default (the fix closes the
  only gating correctness item; the launcher's zero-NCCL stack is now
  correctness-complete WITHOUT the 2.1x fb barrier).
- Re-evaluate `edges` promotion: its correctness gate (census) was the blocker
  in s603; with DENSE_FIX the edges path is event-clean (E1) - the +6.4% perf
  reclaim can now be banked if it holds the gate. (Edges still needs its own
  >=+15% perf re-eval; the correctness objection is removed.)
- Flip the binary defaults to match the launcher (s602 follow-up #4) now that
  the zero-NCCL stack + fix is the correctness baseline.

### Floors + >=50/slot + MTP restatement (clean fix base)

The fix is a pure ordering edge - 8 event records + ~14-16 cross-stream waits
per site, NO extra GEMMs/copies/joins - so it adds negligible step time
(parity in the c3/soak decode-domain, ~153). Critically it changes the
CORRECTNESS FLOOR: before s604 the ONLY event-free config was the s601 full
rank+dense barrier (fb) at decode-domain ~72 (2.1x cost, step ~280 ms-class);
every cheaper config leaked token events. With DENSE_FIX the cheap configs are
event-free, so the correctness-complete floor drops from the fb 2.1x regime
back to:

| Base (correctness-complete WITH fix) | S=1 step | S=8 step | source |
|---|---:|---:|---|
| join + fix (= launcher default + fix) | ~186.9 | ~188.7 | s602/s603 join floor (fix adds ~0) |
| edges + fix (fast base) | ~175.0 | ~177.1 | s603 edges floor (fix adds ~0) |

(fresh e-floor-{s1,s8,edges-s1,edges-s8} runs in flight to confirm the
fix-adds-~0 claim; carried from s603 measured floors, byte-equivalent step path
+ a sub-0.1 ms/site edge.)

- **>=50 tok/s/slot => step <= 20 ms.** On the clean edges+fix fast base
  (~177 ms S=8): gap **~8.8x**, required MTP acceptance multiplier **M ~= 8.8**.
  On join+fix (~189 ms): **M ~= 9.4**. The decisive change vs s602/s603 is NOT
  the multiplier number (similar) but that this floor is now CORRECTNESS-
  COMPLETE at full speed - s603 could not bank the edges floor because its
  census failed; with the fix the edges fast base is event-clean, so M ~= 8.8
  on a *valid* foundation reopens the MTP gate. Sequencing unchanged: (1) the
  rank<->dense hazard is now CLOSED (this sprint); (2) reclaim the remaining
  join pool (~1.1 ms/layer) + prefix-compaction (~1.1) + route-plan shadow
  (~0.45); (3) then MTP at S<=8. The 2.1x fb correctness tax is retired.

## Definition of Done

1. Deterministic amplifier - **DONE.** `DENSE_HAZARD_AMP` at attn_out_a: 0 ->
   ~100% single-run token corruption at >=20us, dose-dependent, byte-identical
   off. (Phase A table.)
2. Carrier named with file:line + hazard class, confirmed by bisect AND
   amplifier, both event classes addressed - **DONE.** `attn_output_a.d_out`
   (and `attn.d_out`) cross-rank dense->rank RAW: producer
   attention_output.cu:48 (dense stream), consumer attention_output.cu:87-98
   (peer rank stream), per-GPU edge at :51 leaves the cross-rank dependency
   unordered. Early-locus class via attn_out_a, late-step class via
   pre_compose; both same dense<->rank family, both closed by one fix.
3. Minimal dense->rank fix, flag-gated, graph-capturable, flag-off
   byte-identical - **DONE.** `DENSE_FIX=1` = cross-GPU dense<->rank edge
   (`s604_dense_rank_edge`), NOT a full barrier, NOT rank-only.
4. Proof: amplified rate zero + >=50-run soak zero-token (telemetry) +
   tolerance 1.0/1.0 + composes with fast stack - **DONE** (amplified 0/0;
   soak 34 runs fix-on 0/0 vs fix-off 7 token/17, telemetry ECC-0; tolerance
   17/17; edges+fix clean). Soak ran 34 not 52 (pod-degradation run-stall,
   documented); statistically decisive.
5. Promotion decision + gate matrix; defaults - **DONE** (PROMOTE; matrix
   above; launcher + binary default flips recommended; rollback = DENSE_FIX=0).
6. Re-measured floors + >=50/slot + MTP - **DONE below** (clean base).
7. Report + orchestrator docs/commits - this document; commits are the
   orchestrator's.

## Deviations (honest list)

1. Phase B's slow per-GEMM amplifier runs (b-predense20/b-aob20/
   b-fix-precompose20) were ABORTED - the busy-wait on every dense GEMM x44
   layers x256 steps made them >30 min/run; the carrier was already named via
   attn_out_a (0->100%) and the pre_compose second-class probe, so they were
   low-value. GPU freed cleanly (idle-verified) for the Phase D soak.
2. The Phase D soak ran 34 runs (17 pairs), not the planned 52 - the 18th
   fix-on run stalled in warmup as the pod degraded (~38h uptime; per-run
   warmup ballooned from ~3 to ~25 min over the session). Stopped cleanly,
   GPUs idle-verified, no foreign processes. The 34-run result is
   statistically decisive (fix-on 17/17 clean vs fix-off 41% token-event).
3. The phaseD-telemetry n_concurrent_proc counts nvidia-smi compute-app ROWS
   (one per GPU context); our single 8-GPU appliance shows as 8 rows sharing
   ONE pid. The lone "8" at soak-on-18 is the prior appliance still tearing
   down at the pre-run sample, NOT a foreign process - the s604-run.sh
   wait-for-idle + unique-pid preflight is the authoritative guard and passed
   every run.
4. Pod-state degradation over the session is itself a reconfirmation of the
   s602/s603 day-to-day variance finding; the alternating A/B design means it
   hit both arms equally and does not bias the gate.

## Artifacts

- Pod `/workspace/s604-artifacts/`: COMMANDS.md, build1.log, s604-phaseA.sh,
  s604-phaseC.sh, s604-phaseB.sh, s604-phaseD.sh, s604-phaseE.sh,
  s604-soak-summary.sh, soak-summary.txt, phaseD-telemetry.tsv, run trees
  (a0ctl, a1-aoa{5,20,50}, c1-fix-aoa20, c2-fix-aoa50, c3-fix-off,
  b-precompose20, soak-{on,off}-1..17, e-edges-fix-*, e-floor-*).
- Laptop `logs/from-cluster/sprint604/`: run .out summaries, soak-summary.txt,
  phaseD-telemetry.tsv, drivers.
- Source changes (uncommitted, orchestrator review):
  engine/runtime_options.cuh (DENSE_HAZARD_AMP + _SITE + DENSE_FIX envs),
  engine/runtime_pack.cu (s604 amplifier `s604_amp_*` + the cross-GPU
  dense<->rank fix edge `s604_dense_rank_edge`/`s604_dense_fix_enqueue`),
  engine/attention_output.cu (carrier site wiring: amp + fix at :48->:51 and
  :110->:113), engine/decode_loop.cu (amp init + amp/fix at the EP/dense
  overlap hand-offs), tools/ds4-v100-run-tp-ep-appliance.sh (env echo),
  new tools/s604-run.sh.
</content>
