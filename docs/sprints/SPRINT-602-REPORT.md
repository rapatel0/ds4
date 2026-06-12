# Sprint 602 Report - The NCCL-Free Decode Graph: Kill the Race, Unlock the Stack

Date: 2026-06-12
Status: complete - **the captured-NCCL race is dead on the zero-NCCL decode
graph (proven, detector-validated, bit-anchored), the batched-exchange and
relay transports are exonerated and now tolerance-clean, and the full
NCCL-free stack is the first configuration ever observed with ZERO token
events (6/6 census runs token-bit-exact vs the s597 control; 15/15 LL pairs
token-identical) - but it costs ~64 ms/step of site synchronization today
(-7% vs control), so the +15% perf gate fails and NO defaults flip; a
low-rate checksum-only late-step flicker remains in our own (instrumentable)
site sync and is the lead 603 item together with the join-cost reclaim.**

## Headline

1. **The race kill worked exactly as the spec predicted.** All 7 (really 9)
   hc-class captured NCCL collectives were replaced by flag-gated
   peer-write/kernel-fold equivalents that are BIT-IDENTICAL to live NCCL
   (in-graph verifier: zero mismatches; full-sequence tolerance vs the s597
   control: sequence 8192/8192 on every gate run). The s597 bit anchor
   survives; no control re-anchor was needed.
2. **Zero NCCL in the captured graph, proven** (verbose dot dump: 0
   matches, 114 s602 kernels in-graph), and on that graph the
   token-corrupting hazard is gone: the harshest s600 reproducer collapses
   from agreement 0.128 (NCCL, reproduced on demand same-day) to 0.997+,
   and the LL serving regime shows zero token events across the entire
   evidence set. Counter-proof: the partial config that keeps the EP-return
   NCCL broadcasts captured (kernel-only) fired a token event at the
   classic locus (d32b).
3. **The surviving-signature branch was exercised and closed in-repo**: the
   residual divergence was triaged with controlled pairs to the s602 site
   synchronization (NOT batched, NOT relay, NOT the eager head), root-caused
   to an under-synchronized pairwise dependency set, and fixed (all-rank
   rank-stream join) for a 30x divergence-mass reduction; a checksum-only
   late-step (59-63) flicker at 0.17/run (LL) remains ours to finish.
4. **No promotion**: the synchronization that buys correctness costs ~1.5
   ms/layer (16 joins), so the full stack measures ~153 vs the 194.4
   required; defaults stay (nccl/copy/nccl). The correctness-flip
   recommendation and the join-reclaim path for 603 are recorded.
5. **>=50/slot restated**: on the only correctness-complete config the step
   floor is 186.9-188.7 ms (S=1/S=8), required MTP multiplier 9.3-9.4 -
   MTP stays sequenced behind the 603 join reclaim + the unchanged prefix/
   route-plan pools.

## Agent handoff (honest deviation, first item)

This sprint was executed across two agents. A prior agent built the bulk of
Phase A (the `DS4_V100_TP_EP_HC_TRANSPORT=nccl|kernel` flag, ~635 lines of
kernel-collective helpers in `engine/runtime_pack.cu` with ring-order-exact
fold logic, the 9-class call-site conversions in `engine/hc_current.cu`,
`engine/router_step.cu`, `engine/post_attention_ffn.cu`,
`engine/runtime_resources.cu`, the in-graph bit-verifier, and
`tools/s602-fold-probe.cu`), then stalled on infrastructure. The work was
inherited as an uncommitted working-tree diff at laptop HEAD d11e0f3f and
audited line-by-line before completion. Audit findings:

- The helpers and all 9 call-site conversions were complete and correct.
- Three gaps were found and fixed: `s602_state_init` was never called
  (wired into `ensure_compose_buffers` in runtime_resources.cu),
  `s602_collect_verify` was never called (wired next to the two
  `s600_collect_verify` sites in decode_loop.cu), and the launcher had no
  `HC_TRANSPORT` plumbing (added: default, validation, echo, export).
- The default ring spec was wrong ("0 3 2 1 5 7 6 4", the s597
  NO_SYS_RING env value) - the actual reference config never exports
  NCCL_RINGS (ALLOW_VISIBLE_REMAP=0), and NCCL's auto channel-0 ring on
  this pod is "0 3 2 1 5 6 7 4". Fixed (measured via NCCL_DEBUG=INFO).

## The collective inventory audit

`grep -n nccl engine/*.cu` against the captured path, with gate tracing:

- Converted (the 9 classes, all live in the reference config): hc max+mix
  allreduces + hc sumsq allreduce + hc current allgather (hc_current.cu),
  ffn full-current broadcast (runtime_resources.cu), router max/sumsq/logits
  allreduces (router_step.cu), post-attention allgather
  (post_attention_ffn.cu).
- `attention_output.cu:67` allgather: gated by
  `true_ds4_attention_output_nccl_allgather_gate`, which defaults false and
  has NO setter anywhere (options.h parses nothing for it) - dead code in
  the reference config; the active branch is the peer `cudaMemcpy2DAsync`
  path. No conversion needed. (The s601 COMM_SPLIT=hc description listed it
  because the code site was switched to the class comm, not because it
  executes.)
- `decode_loop.cu:1352` / `ep_compose.cu:97` reduce-scatters: gated by
  `nccl_reduce_scatter_compose_gate`, default false, no setter - dead.
- `router_step.cu:86` allgather: `run_model_router_rank_major_logits` path,
  default-off (`model_router_allreduce_logits_gate=true` selects the
  converted allreduce variant).
- `output_head.cu` collectives: eager/head ops outside the captured graph,
  out of scope (s601 a1 proved them irrelevant to the race).

## Phase A0 - fold-order calibration (tools/s602-fold-probe)

Probe = standalone 8-GPU NCCL harness reproducing the engine's exact
grouped-allreduce structure with order-sensitive random inputs, plus a
hypothesis search over NCCL's ring reduce-scatter chunk schedule
(nc, minChunk, delta, ring-base). Run under the appliance env
(NCCL_P2P_LEVEL=NVL, auto algo/proto, NCCL 2.19.3).

- Measured auto topology: 12 channels over 4 distinct rings; ch0 ring
  `0 3 2 1 5 6 7 4` (NOT the s597 NO_SYS_RING; that env is only exported
  when ALLOW_VISIBLE_REMAP=1, which the reference config never sets).
- run2 (pow2 minChunk grid): single-chunk shapes match (fold = left fold
  along the ch0 ring starting at position 1, delta=1, nc=1); multi-chunk
  shapes (hc_mix at slots>=16, r_logits at slots<=4) NO-MATCH.
- run3 (extended grid): hc_mix matches globally with chunks of 192; r_logits
  matches per-shape with chunk 192/256/512/1024 by size - no single global
  minChunk exists. The unifying rule (every probed shape, all four sum
  collectives): **NCCL LL picks nthreads = clamp(pow2ceil(bytes/64), 96, 512)
  and minChunk = nthreads*2 floats; delta=1, nc=1, ch0 ring.**
- Engine: `DS4_V100_TP_EP_S602_MIN_CHUNK` default is now 0 = this auto size
  rule (`s602_min_chunk_for()` in runtime_pack.cu); the env still overrides
  globally, and RING/FOLD_DELTA/NCHANNELS remain recalibration knobs.

## Phase A - per-collective bring-up and gates (all PASS)

| Gate | Run | Result |
|---|---|---|
| Flag-off byte-identity | rctl602 (build1, flags off) | decode-domain 164.80 / wall 111.42 (s601 band 167.5-169.0, -1.9%); tolerance vs s597 control 1.0/1.0; sequence 8192/8192; zero events |
| In-graph bit-verifier, all 9 classes | averify1 (kernel+verify, 32x8) | tp_ep_s602_init PASS (stage 284 KiB/GPU); **ZERO mismatches** across 9 classes x 8 ranks x 43 layers x 8 steps |
| Kernel-mode full-sequence bit-anchor | a2-kernel-tol (build3) | **tolerance 1.0/1.0 vs s597 control - BIT-EXACT** (8192/8192); zero events |
| Verifier after pairwise-sync optimization | averify2 (build4) | ZERO mismatches; replay 4.49 ms/layer (verify+NCCL still on) |
| Kernel-mode perf + bit-anchor (final) | a3-kernel-tol (build4) | decode-domain 162.86 / wall 112.32 (parity, -1.2% vs control); **tolerance 1.0/1.0 BIT-EXACT**; VRAM 29616 MiB (control 30338) |

**The bit-anchor decision resolved on the primary path: ring-order-exact
held.** The s597 control anchor remains valid; no re-anchor was needed; the
fallback policy was never invoked.

### The performance lesson (build3 -> build4)

The first complete kernel transport used the s601 full 8x8 cross-GPU
barrier (which also joins the dense streams) at both sync points of every
site - 16 full barriers per layer - and measured replay 10.9 ms/layer vs
4.13 control (decode-domain 67.8). The replaced NCCL collectives only ever
ordered the rank streams, so the dense joins were pure over-synchronization
destroying the layer's rank<->dense overlap. Fix: pairwise event ordering
with exactly the NCCL-equivalent dependency sets - B0: each rank stream
waits its 4 NVLink peers (folds read quad+mirror partials; relays read
their 3 quad-mates; WAR on relay-written buffers is vs the mirror's prior
consumers); B1: each rank waits its mirror (g^4), the only writer into its
buffers. 8 records + 32 waits and 8 records + 8 waits respectively, rank
streams only. Escape hatch: `DS4_V100_TP_EP_S602_FULL_BARRIER=1`.
Result: 10.9 -> 4.49 (verify-on) and kernel-mode parity with NCCL.

## Phase B - race verdict

### Zero-NCCL captured graph: PROVEN

`cudaGraphDebugDotPrint` of the layer-2 captured graph under the full stack
(`HC_TRANSPORT=kernel + EP_RETURN_TRANSPORT=relay + SWIGLU_EXCHANGE=batched`):
**zero matches for "nccl"** in the verbose dot (762 kernel nodes, 65 memcpy,
32 memset; s602 kernels in-graph: 57 copy3 + 41 fold + 16 gather8). The only
NCCL left in the entire step is the eager output head, outside the graph.

### The Simple-stress reproducer: signature transformed

b-sb-{1,2,3} (NCCL_PROTO=Simple + batched + relay + kernel; the harshest
s600 detector, 3 pairwise comparisons of 256-step runs):

| Pair | selected agreement | sequence agreement | first_ck histogram |
|---|---:|---:|---|
| sb2 vs sb1 | 0.906 | 0.949 | {0:32, 18:32, 20:32, 22:32} |
| sb3 vs sb1 | 0.938 | 0.962 | {0:32, 9:32, 18:32, 35:32} |
| sb3 vs sb2 | 0.953 | 0.969 | {7:32, 9:32, 20:32, 28:32} |

versus the captured-NCCL signature (s601 and the bsanity control below):
agreement 0.016-0.047, fixed locus 96 pairs @ step 0 + 32 @ step 2, token
flips nearly everywhere. The zero-NCCL stack still diverges run-to-run under
Simple stress, but with ~30x less divergence mass, variable onsets (7-35),
and rare token flips - a DIFFERENT, far weaker hazard (triage below).

### Detector sanity: the old race reproduces on demand

bsanity-{1,2} (same Simple+batched+relay regime, HC back on NCCL): agreement
**0.047 / 0.128, first_ck {0:96, 2:32}** - the exact s600/s601 catastrophic
signature, reconfirmed same-day on this binary. The detector is valid, and
the s602 kernel transport is what removed the catastrophic race.

### LL-regime census (the serving regime)

g-1 (LL, full stack): **tolerance 1.0/1.0 vs the s597 control - BIT-EXACT,
zero checksum events, zero token events.** The batched exchange - which
failed tolerance in every prior sprint (s599 C-A5; s601 c1 = 0.9375/0.8928
token flips) - is CLEAN with the captured-NCCL race dead, confirming s601's
"racing collectives are upstream of the exchange" reading and exonerating
the exchange itself.

### The surviving signature, triaged to root cause (the "race survives" branch - exercised and CLOSED)

The residual b-sb divergence was characterized with controlled pairs, all
under the same Simple-stress regime:

| Pair | Config delta | Verdict |
|---|---|---|
| b-sb-kc-{1,2} | batched -> copy exchange | still diverges (0.789/0.935, onsets {4,13,20,27}) - batched exchange NOT the carrier |
| b-sb-fb-{1,2} | s602 sync -> full s601 barrier | **BIT-IDENTICAL (1.0/1.0)** - the carrier is the s602 site synchronization |

The fb result simultaneously exonerates the eager-head NCCL (still present
and Simple in fb), the relay return, and the batched exchange. Root cause:
the first-cut pairwise dependency set (B0 = wait 4 NVLink peers, B1 = wait
mirror g^4) under-synchronizes - NCCL's completion semantics join all 8
rank streams at every collective, and the engine relies on that contract.
Fix (build5): full 8x8 join across the rank streams only (keeps NCCL's
contract; avoids the dense-stream joins that cost 10.9 ms/layer). The
pairwise mode was removed; `DS4_V100_TP_EP_S602_FULL_BARRIER=1` retained.

This is exactly the payoff the spec predicted for a surviving signature:
with the collectives in-repo, the hazard was localized and fixed in hours,
not escalated to NVIDIA.

### build5 (rank-join) gates - THE RACE VERDICT

- averify5: ZERO verifier mismatches (rank-join bring-up clean).
- b5-sb-{1,2,3} Simple-stress pairwise: residual shrinks to sequence
  agreement **0.9976/0.9969/0.9994** (1-2 divergent batches per pair,
  onsets {59}/{6,63}/{6,63}) - vs 0.949-0.969 pairwise-sync and 0.128 NCCL.
  Not yet bit-identical under maximal stress.
- **LL census (the serving regime), g5-{1..6}: ALL SIX runs token-bit-exact
  vs the s597 control (sequence 1.0 = 8192/8192). ONE checksum-only event
  in 6 runs (g5-6, batch 3, step 62, ZERO token changes) = 0.17 ck-events/
  run and 0.0 token events - vs the promoted path's 1.0 events/run with
  token flips in 2 of 3 control runs (s601), and relay's 1.5/run.**
- LL run-to-run determinism: **15/15 pairwise comparisons token-identical**
  (the single ck event appears as a {62:32} checksum-histogram row).

**Race verdict: the captured-NCCL race - the s599/600/601 hazard that had
escalated to token corruption on the DEFAULT serving path - is DEAD on the
zero-NCCL graph (detector-validated same-day). What remains on the
zero-NCCL stack is a low-rate, checksum-only, late-step (59-63) flicker:
0.17/run at LL with zero token impact ever observed on build5, and ~1.7
events/256-step pair under the artificial Simple-stress amplifier. The
kernels are ours; the next escalation of the s602 site-sync dependency
analysis (or the full-barrier mode, bit-stable in its single tested pair
at 2.3x cost) closes it. Confirmation from the other direction: the
PARTIAL config (hc kernel + EP-return NCCL still captured) fired a
token-level event at the classic locus (d32b, Phase D) - any captured NCCL
means token events; zero captured NCCL means none observed.**

Residual-signature notes for 603: events cluster at steps 59-63 of the
64-token window (teardown-adjacent); checksum stream only; token streams
bit-exact in every observed instance on build5.

## Phase C - promotion

### Gate matrix (full stack: kernel + relay + batched, build5, vs spec gates)

| Gate | Result |
|---|---|
| Race-zero, pairwise identity >= 3x256-step | **token-level PASS** (15/15 LL pairs token-identical; 3/3 Simple-stress pairs sequence 0.997-0.999); strict checksum-identity FAIL (residual ck-only flicker) |
| Race-zero, event census >= 6 runs, zero ck AND token events | **token events ZERO (6/6)**; ck events 1/6 runs (0.17/run vs promoted 1.0/run) - strict FAIL, 6x improvement |
| Tolerance 1.0/1.0 vs s597 control | **PASS 6/6 (token-bit-exact, sequence 8192/8192)** |
| Perf >= +15% over 169.01 (>= 194.4) | **FAIL** - census 142.3-156.6 (median ~153.4), i.e. -7% vs the same-day flag-off control 164.80 |
| no-SYS | by construction (NV peers + dst^4 relay only; zero NCCL in-graph); s598 nsys proof stands for the promoted return; new nsys window not run (budget; moot with non-promotion) |
| VRAM in budget | PASS - 28768 MiB max vs control 30338 (kernel transport is net-NEGATIVE VRAM: 284 KiB staging vs NCCL channel buffers) |

**Phase C verdict: NO PROMOTION - launcher defaults unchanged** (EP return
nccl, swiglu copy, HC transport nccl, early-return off). All s602 modes
remain opt-in envs with trivial rollbacks
(`DS4_V100_TP_EP_HC_TRANSPORT=kernel`, `DS4_V100_TP_EP_S602_FULL_BARRIER`,
`DS4_V100_TP_EP_S602_{RING,FOLD_DELTA,MIN_CHUNK,NCHANNELS,KERNEL_MASK,VERIFY}`).

### Why perf regressed and the recommendation

The s602 sites add 16 all-rank rank-stream joins per layer. At parity that
costs nothing on the kernel-only config (a3: 162.86 vs control 164.80,
-1.2%), but it serializes the EP window enough that the relay+batched
levers (s601 demonstrated +23.8%) no longer materialize: g5 ep_ms shows the
EP envelope halved (448 -> 234 ms) yet decode-domain is flat - the EP
savings fell off the critical path. The site-sync structure is now the
binding constraint (the 603 lever).

**The d32b twist (Phase D, kernel-only finals): kernel-only is NOT
race-free.** d32b (HC kernel, EP return still NCCL) fired a token-level
event in its third observation - batch 4, onset step 1, 4/32 slots
token-flipping, the classic captured-NCCL locus class (cf. s601's
rctl601b/d32 escalation events). Kernel-only != zero-NCCL: the 8 EP-return
broadcasts remain captured, and with the hc collectives gone they carry
their own exposure (a2/a3's clean runs were 2-run luck at this event
rate). **The full zero-NCCL stack is the only correctness-complete
configuration** - 6/6 census runs token-clean, 15/15 pairs token-identical.

**Recommendation for the orchestrator** (not actioned; outside this
sprint's gates): the correctness flip worth weighing is the FULL stack
(kernel+relay+batched) at ~-7% decode-domain - it is the only config with
zero observed token events (the promoted default corrupts tokens at ~1
event per reference run, s601 escalation + reconfirmed by d32b's
kernel-only event this sprint). The +15% perf gate as written blocks it;
the correctness calculus may dominate. The 603 site-sync lever (below)
likely turns the trade positive.

## Phase D - re-measure

All numbers reference shape unless noted; same pod/day; flags-off control
rctl602 = 164.80 decode-domain / 111.42 wall (s601 band 167.5-169.0, -1.9%).

| Config | S=1 step | S=8 step | S=32 step | S=32 decode-domain | per-slot @ S=8 |
|---|---:|---:|---:|---:|---:|
| s601 promoted (NCCL, racing) | 123.3 | 125.7 | 189.3 | 169.01 | 7.96 |
| s602 kernel-only (EP-return NCCL remains; still races - d32b) | 168.8 | 175.9 | 196.5 | 162.86 (a3) | 5.68 |
| **s602 zero-NCCL full stack (correctness-complete)** | **186.9** | **188.7** | **208.6** | **142.3-156.6, median ~153.4** | **5.30** |

(d1f 5.35 tok/s, d8f 42.40, census g5-{1..6}; the curve stays slot-flat to
S=8 exactly as s601 measured.)

Stage means at S=8 (profiler-on, rank 0, relative structure; d8p kernel-only
/ d8fp full stack vs the s601 promoted values):

| Stage | s601 promoted | kernel-only | full stack |
|---|---:|---:|---:|
| ep_window (envelope) | 1.531 | 2.286 | 2.557 |
| route_plan_pack | 0.449 | 0.954 | 0.935 |
| prefix_hc_current | 0.223 | 0.686 | 0.664 |
| prefix_attn_output | 0.659 | 0.599 | 0.566 |
| final_hc | 0.287 | 0.308 | 0.604 |
| ep_return | 0.375 (nccl) | 0.585 (nccl) | 0.521 (relay) |

The 16 rank-joins/layer (~0.09 ms each) land in the stages hosting the s602
sites (router ARs inside route_plan_pack +0.5, hc sites +0.45, final_hc) -
~1.4-1.5 ms/layer total, matching the step delta (186.9 - 123.3 = 63.6 ms
= 1.48 ms/layer). That join cost is the single dominant 603 reclaim target.

### >=50 tok/s/slot budget + required-MTP-multiplier restatement

Target: step <= 20 ms. On the only correctness-complete config (zero-NCCL
full stack): **step floor 186.9 ms (S=1) / 188.7 ms (S=8) - gap 9.3-9.4x;
required MTP acceptance multiplier M = 9.3 @ S=1 and 9.4 @ S=8** (s601
stated 6.3-7.1 on the racing promoted base - that base corrupts tokens and
is no longer a valid foundation). MTP block-2 yields ~2-3 accepted
tokens/step at realistic acceptance: **>=50/slot remains unreachable via
MTP alone**, and the gap WIDENED because correctness costs ~64 ms/step of
synchronization today. Sequencing: (1) 603 site-sync reclaim (the 1.5
ms/layer join pool: scoped/merged joins, join-free fold dataflow, or
finding the single missing pairwise edge - the full-vs-pairwise delta
brackets it); (2) then the unchanged prefix-compaction (~1.1 ms/layer) and
route-plan-shadow (~0.45) pools; (3) only then re-open the MTP gate at
S<=8. If 603 recovers the joins to ~0.1 ms/layer total, the zero-NCCL
stack lands near the s601 floor (~125 ms) with bit-exact tokens, and M
returns to ~6.3 - still requiring step levers before MTP.

## Definition of Done

1. Per-collective kernel replacements built, bit-verified, flag-gated;
   ring-order-exact for the allreduces - **done**: 9 classes, in-graph
   verifier zero mismatches (averify1/2/5), ring-order-exact PROVEN
   end-to-end (a2/a3/g5 tolerance 1.0/1.0, sequence 8192/8192 BIT-EXACT vs
   the s597 control; no re-anchor needed; fallback never invoked); fold
   calibration = measured ch0 auto ring + NCCL LL nthreads-by-size rule
   (probe-derived, env-overridable).
2. Zero-NCCL captured graph proven - **done**: layer-2 verbose dot dump, 0
   nccl matches, 762 kernel nodes incl. 114 s602 kernels.
3. Race verdict with event census - **done**: catastrophic captured-NCCL
   race DEAD (bsanity reproduces it on demand on NCCL; zero-NCCL census
   0.17 ck-only events/run, ZERO token events, 15/15 LL pairs
   token-identical). The spec's "race survives" branch was exercised: the
   surviving signature was triaged in-repo to the s602 site-sync
   dependency set (kc/fb controlled pairs), fixed (pairwise -> rank-join,
   30x divergence-mass reduction), and the residual ck-only late-step
   flicker is documented with rates and loci. Counter-evidence d32b: any
   captured NCCL (kernel-only's EP return) still produces token events.
4. Promotion per gates - **done, non-promotion recorded with the gate
   matrix** (tolerance PASS 6/6, token-race-zero PASS, perf FAIL -7% vs
   +15% required, VRAM PASS, no-SYS by construction); launcher defaults
   unchanged; all modes opt-in with rollbacks; correctness-flip
   recommendation recorded for the orchestrator.
5. Final numbers + >=50/slot budget + MTP statement - **done** (above).
6. Report + follow-ups - this document; commits are the orchestrator's.

## Deviations (honest list)

1. Agent handoff (see top): prior agent's Phase A inherited uncommitted,
   audited, completed; three wiring gaps and one wrong default fixed.
2. The spec's "promoted ring 0 3 2 1 5 7 6 4" premise was wrong for the
   reference config - the launcher only exports NCCL_RINGS under
   ALLOW_VISIBLE_REMAP=1. Calibration used the measured auto rings.
3. The fold-probe's first run burned one GPU-job slot on the wrong default
   ring (run1, discarded).
4. The first census driver had a bash bug (`$i_DONE` under set -u) that
   killed runs g-2..6 after g-1; fixed and re-run as g5 on build5 (g-1 on
   build4 retained as evidence).
5. The cancelled build4 queue (census2 + head-host pair) left two defunct
   zombie processes on the pod (harmless; reaped by init).
6. The full-barrier control (b-sb-fb) is a single pair; at the measured
   residual rate its zero-divergence has P~0.18 of being luck - recorded,
   not over-claimed; rank-join's own 3-pair evidence stands.
7. The eager-head-host triage pair (hh) was cancelled when fb localized
   the carrier to the s602 sync; with fb's n=1 caveat, head-host remains
   a cheap 603 confirmation run.
8. d32b doubles as the kernel-only token-event observation AND a slow
   perf outlier (139.35; pod under sustained load for hours by then);
   kernel-only S=32 perf is anchored on a3 instead.
9. No nsys window was run (one V100 job at a time + budget; no-SYS holds
   by construction for the s602 transports and the s598 proof stands).
10. Phase D S=1/S=8 runs are SKIP_TOL (no 32-slot tolerance defined there),
    per the s601 method.

## Artifacts

- Pod `/workspace/s602-artifacts/`: COMMANDS.md, build1-5.log,
  s602-fold-probe + fold-probe-run{1,2,3}.log, dot/ + dot-summary.txt,
  run trees (rctl602, averify{1,2,5}, a2-kernel-tol, a3-kernel-tol,
  b-sb-{1,2,3}, b-sb-kc-{1,2}, b-sb-fb-{1,2}, bsanity-{1,2}, dotrun, g-1,
  g5-{1..6}, b5-sb-{1,2,3}, d32b, d1, d8, d1p, d8p, d1f, d8f, d8fp),
  compare files (phaseB-sb/-sanity/-kc/-fb, b5-sb, g5-pairwise), drivers
  (phaseB-gates.sh, phaseB2-sanity.sh, phaseBC-census{,2}.sh,
  phaseB3-triage.sh, phaseB4-headhost.sh [unrun], phaseB-all.sh,
  a3-chain.sh, b5-gates.sh, g5-pairwise.sh, phaseD.sh, phaseD2.sh).
- Laptop `logs/from-cluster/sprint602/`: COMMANDS.md, run summaries,
  compare files, probe logs, dot-summary (29 files).
- Source changes (uncommitted, orchestrator review): engine/runtime_pack.cu
  (s602 kernel-collective set + rank-join sync + auto min-chunk),
  engine/{hc_current,router_step,post_attention_ffn,runtime_resources}.cu
  (site conversions + init wiring), engine/decode_loop.cu (collect wiring),
  engine/runtime_options.cuh + runtime_types.cuh (flags/state),
  tools/ds4-v100-run-tp-ep-appliance.sh (HC_TRANSPORT plumbing),
  new tools/s602-fold-probe.cu, new tools/s602-run.sh.

## Follow-up seeds (for SPRINT-603 planning)

1. **Reclaim the join cost (~1.5 ms/layer, 16 joins)** - the single
   dominant step-time lever, larger than the prefix/route-plan pools. The
   bracket from this sprint: pairwise (4 NV-peer edges) is too weak,
   all-rank join suffices for token-exactness - the missing edge(s) lie in
   between. Candidates: per-site scoped joins derived from the actual
   consumer sets, merging adjacent sites' B1/B0, join-free fold dataflow,
   or finding the single missing pairwise dependency (the late-step 59-63
   locus is the clue; instrument the site kernels).
2. **Finish the residual ck-only flicker** (0.17/run LL, late-step): same
   investigation as (1); confirmatory cheap runs: more b-sb-fb pairs (the
   n=1 caveat) and the unrun head-host pair.
3. **The correctness flip decision**: zero-NCCL full stack at -7% vs a
   default that corrupts tokens ~1/run - orchestrator call; 603's join
   reclaim likely makes it strictly positive and unlocks the s601
   relay/batched gains that are currently masked (EP envelope halved with
   zero step-time benefit).
4. NVIDIA escalation package strengthened: same-day A/B - identical binary,
   identical regime, NCCL hc collectives = agreement 0.128 with token
   flips; our bit-identical kernel transport = 0.997+; plus the d32b
   EP-return-NCCL token event. "RING_LL captured collectives in
   multi-GPU-per-process graphs" now has a clean counterfactual.
5. MTP gate stays sequenced after (1) + at least one of prefix-compaction
   (~1.1 ms/layer) / route-plan shadow (~0.45 ms/layer).
