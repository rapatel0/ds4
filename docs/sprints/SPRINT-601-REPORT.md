# Sprint 601 Report - Kill the NCCL Race (Comm Isolation / NCCL-Free EP), Measure Slot Scaling

Date: 2026-06-12
Status: complete - **the race survived both planned kill families (comm
isolation at every granularity; NCCL-free EP window) and is now localized to
the hc-class collectives; no promotion (the relay stack measures +11% but
raises the event rate); the promoted path's own exposure escalated to
token-level on the record; slot-scaling curve measured: the step is a
~123-130 ms launch/wait floor that is slot-flat to S=8, so >=50/slot needs
~6-7x acceptance from MTP - unreachable without first cutting the base step
2-3x further.**

Environment: pod `llm/llamacpp-build-8gpu` (gpu-01, 8x V100-SXM2-32GB, driver
580.126.20, NCCL 2.19.3), recreated with a **16 Gi memory-backed /dev/shm**
(the s600 blocker); /workspace persisted; apt re-provisioned per s597
COMMANDS.md. Pack `...-s597`, contract, s597 `phase0-full-control`, fixed
harness, idle+foreign-process preflight on every run. Tree at laptop HEAD
1ba7b528 + the s601 edits below.

## Phase 0 - shm unblock verified

- `/dev/shm` = 16 Gi tmpfs (verified in-pod).
- s600 twocomm test rebuilt and re-run: **both 8-rank comms init (rc=0)**,
  33 MiB /dev/shm each. Dedicated-comm creation is unblocked.
- Identity control `rctl601` (flags off): decode-domain 168.15 / wall 114.75
  (in band vs s600 170.18/117.08), tolerance 1.0/1.0, nodes 2825/layer.
  Evidence note: the known promoted-path latent event fired once
  (batch 1, step 2, checksum-only, zero token changes - the exact rctl600
  signature).

## New code (all default-off; flag-off byte-identical)

- `DS4_V100_TP_EP_COMM_SPLIT=none|epret|hc|epret+hc` (engine/runtime_options.cuh,
  runtime_types.cuh, runtime_resources.cu + call sites in runtime_pack.cu,
  hc_current.cu, router_step.cu, post_attention_ffn.cu, attention_output.cu,
  decode_loop.cu, ep_compose.cu):
  - `epret`: the 8 captured EP-return broadcasts (+ swiglu nccl allgather +
    compose reduce-scatter) move to a dedicated communicator (default buffsize).
  - `hc`: every other captured collective (hc 3 allreduces + allgather, router
    allreduce, full-current broadcast, post-attn allgather, attn-output
    allgather) moves to a dedicated communicator (1 MiB buffsize).
  - Measured VRAM cost: **490 MiB/GPU per comm** (single-ring NCCL_RINGS policy;
    well under the s598 ~0.9 GiB estimate). Both comms together fit the budget.
- `DS4_V100_TP_EP_EP_RETURN_TRANSPORT=relay` (Phase B, see below):
  engine/runtime_options.cuh, runtime_types.cuh (staging buffer),
  runtime_resources.cu (alloc/free), runtime_pack.cu (`ep_return_relay_graph`),
  runtime_profiler.cu (`ep_return_relay` stage), decode_loop.cu (dispatch).
- Launcher plumbing for `DS4_V100_TP_EP_COMM_SPLIT` + `DS4_V100_TP_EP_HEAD_COMM`
  + `relay` transport value.
- Bench harness: `--slots` now accepts {1,2,4,8,16,24,32} and the deterministic
  submission wave follows the slot count (32-slot behavior unchanged).
- `tools/s601-run.sh` run wrapper (preflight + bench + tolerance/first-divergence).

## Phase A - communicator-isolation matrix

Detector: the harshest s600 reproducer (NCCL_PROTO=Simple + batched exchange;
fires nearly every step) is **run-to-run nondeterministic** when the race is
alive, so the gate is pairwise bit-identity of `decode_step_checksums` across
>= 3 x 256-step runs (the LL-vs-control comparison is invalid under Simple's
allreduce regime).

| Candidate | Config | Runs | Verdict (event counts) |
|---|---|---|---|
| a0 reproducer sanity | none | 2 | FAIL as expected: 128/128 pairs diverge (96 @ step 0, 32 @ step 2); agreement 0.023/0.087 |
| a1 dedicated head comm | HEAD_COMM=dedicated | 2 | **FAIL**: identical signature (0.023/0.072) - eager-head isolation does not touch the captured-collective race |
| a3 full class split | COMM_SPLIT=epret+hc | 3 | **FAIL**: all 3 pairwise comparisons diverge, identical signature (0.016-0.031 / 96 pairs @ step 0 + 32 @ step 2) |
| a2 EP-return split | COMM_SPLIT=epret | 0 | not run: strictly weaker than a3, subsumed by its verdict |

**Phase A verdict: communicator isolation does not touch the race at any
granularity** (eager-head isolation, EP-return class, full class split with
the compose comm reduced to eager-only). The hazard is per-collective /
NCCL-internal - the spec's "intra-comm" risk row. Pivot to Phase B.

## Phase B - NCCL-free EP window

New transport `DS4_V100_TP_EP_EP_RETURN_TRANSPORT=relay` (default unchanged,
binary default `copy`, launcher default `nccl` until promotion):

- src-side **peer-write** copy kernels over NVLink for the 16 NV directed
  pairs per src; the 24 SYS directed pairs forward **one-hop** through a
  staging buffer on relay GPU `dst^4` - per the s597 relay table this GPU is
  NVLink-adjacent to both ends for every SYS pair, and the schedule is
  perfectly balanced (3 directed relays per GPU).
- Fixed order: stage W (src streams) -> 8x8 event barrier -> stage F (relay
  streams) -> 8x8 event barrier -> compose. Byte moves only (bit-exact by
  construction), graph-capturable, no SYS traffic.
- Staging cost: ~5.5 MiB/GPU at the reference shape.
- Combined with `SWIGLU_EXCHANGE=batched` this removes 9 of 16 captured NCCL
  collectives per rank-layer (8 EP-return broadcasts; the batched exchange
  replaces the copy storm without NCCL).

Results:

- **Transport correctness + speed (b1-relay-ctl, relay + copy exchange, LL)**:
  decode-domain **186.58 / wall 125.82** (+10.8% over the 170.18 baseline with
  the copy storm still in place), tolerance **1.0/1.0**, capture 2721
  nodes/layer (vs 2825). One checksum-only latent event (batch 2, step 33,
  zero token changes) - promoted-path-level exposure, unchanged.
- **Race gate (b-sb-1/2/3, Simple + batched + relay)**: **FAIL** - all three
  pairwise comparisons diverge with the identical signature (96 pairs @ step
  0, 32 @ step 2; agreement 0.016-0.047). Removing all 9 EP-window
  collectives does not move the divergence locus by one step.

**Phase B race verdict: the captured-NCCL race does not live in the EP
window.** It survives in the remaining hc-class collectives (hc 3 allreduces
+ allgather, router allreduce, full-current broadcast, post-attn allgather).
This answers the s600 follow-up-seed experiment: an NCCL-free EP window alone
is insufficient; the remaining in-repo path is a fully NCCL-free decode graph
(kernel reductions for hc/router - with the associativity/bit-anchor decision
that implies). Out of s601 scope; A and B are both exhausted with evidence.

Diagnostic side-fact: under Simple, every config's FIRST event is batch 1
step 2 - the same locus as the promoted path's rare LL event (rctl600 and
rctl601 both fired at batch 1 step 2, checksum-only).

## Phase C - promotion

Candidate stack measurements (LL regime, reference shape, vs the s597
control):

| Config | decode-domain | wall | tolerance | verdict |
|---|---:|---:|---|---|
| rctl601 nccl+copy (baseline, this binary family) | 168.15 | 114.75 | 1.0/1.0, 1 checksum-only event (b1 s2) | baseline |
| b1 relay+copy | **186.58 (+11.0%)** | 125.82 | **1.0/1.0**, 1 checksum-only event (b2 s33) | promotion candidate |
| c1 relay+batched | 208.18 (+23.8%) | 135.27 | FAIL 0.9375/0.8928 (token flips, onsets s2+s13) | demonstrated ceiling; batched stays opt-in |
| c2 relay+batched+profiler | 179.16 | 117.0 | FAIL (0.5/0.70) | stage-table evidence run |

Stage table (c2, per rank per layer): **ep_return_relay 0.24-0.26 ms** (the
NCCL return was 0.61), shared_swiglu_down 0.41-0.50 (batched),
route_plan_pack 0.40-0.44, prefix_attn_output 0.76-0.89, final_hc 0.43,
ep_window ~1.8 ms, barriers 0.02-0.24.

The batched swiglu exchange still trips token-level corruption at the
reference shape (s599 C-A5 behavior, unchanged by the relay return - the
racing collectives are upstream of the exchange). It remains opt-in.

### Promotion gate matrix for relay + copy

Event census (LL regime, this binary, vs the s597 control; an "event" = one
32-slot batch whose `decode_step_checksums` diverge; TOKEN = token flips at
the onset step):

| Config | Runs | Events | Token-flipping events | Tolerance 1.0/1.0 runs |
|---|---:|---:|---:|---:|
| nccl+copy (promoted) | 3 (rctl601, rctl601b, d32) | 3 (1.0/run) | 2 (8 and 4 slots @ b1 s2) | 1/3 |
| relay+copy | 6 (b1, g2-2..6) | 9 (1.5/run) | 5 (4 slots each) | 2/6 |

| Gate | Result |
|---|---|
| Race exposure not worsened vs promoted (>=3 paired runs) | **FAIL** - 1.5 events/run vs 1.0, 5/6 runs with events, 4/6 with token flips; mechanism-consistent (faster EP window = tighter intra-replay pacing = higher exposure, the s600 rate-vs-spacing curve reconfirmed) |
| Tolerance 1.0/1.0 | FAIL in 4/6 candidate runs (and 2/3 control runs - see escalation note) |
| Perf >= +8% over 168.15 (>=181.6) | PASS - 185.2-187.5 across 6 runs (+10.2-11.5%) |
| no-SYS spot-check | moot with non-promotion; relay avoids SYS by construction (NV-direct + dst^4 one-hop staging); the s598 no-SYS proof for the promoted NCCL return stands |

**Phase C verdict: NO PROMOTION.** Launcher defaults unchanged (`nccl`
return, `copy` exchange, shared comm, early-return off). `relay`, `batched`,
and `COMM_SPLIT` retained opt-in with documented rollbacks.

**Escalation fact (new, load-bearing):** the PROMOTED path itself fired
token-level events in 2 of 3 control runs this sprint (rctl601b: 8/32 slots
flip at batch 1 step 2, sequence 0.9395; d32: 4/32 slots, same locus) - the
first token-flipping events ever recorded on the promoted config (s600's
promoted-path events were checksum-only). The hc-collective race is now a
live token-correctness debt on the DEFAULT serving path at ~1 event per
4-batch reference run, not a checksum curiosity. The fully-NCCL-free decode
graph (kernel reductions for the hc/router collective set, ring-order-exact
if the bit anchor is to survive) is the lead item for the next sprint, and
the day-to-day instability of the control anchor is itself part of why no
default flip is defensible this sprint.

### C-B restack / C-C route-plan shadow

- **C-B (early return) restack: not attempted, with reasoning.** The relay
  return embeds two 8x8 event barriers; hoisting it into the dense-overlap
  window would fence the dense streams mid-overlap (the barrier waits on
  dense streams too), which is structurally anti-synergistic - and s599
  already measured the restack neutral-to-negative even with a barrier-free
  NCCL return. Moot for defaults given non-promotion.
- **C-C (route-plan shadow): not attempted** - budget went to the Phase A/B
  matrix and the Phase C event census (30 reference-shape runs this sprint).
  The pool is re-measured at 0.40-0.44 ms/layer (c2 stage table); the
  plan/pack split design from s599 stands for a future sprint.

## Phase D - slot-scaling curve (promoted stack, flags off)

Method: 256K ctx, 64 tok/req, deterministic harness with the submission wave
equal to the slot count (4 full waves per point; 8 at S=1), one bench
session, idle-verified between runs. S>=8 points on build2; S<8 points on
build3 (identical except the fixture-KV-slot clamp, a no-op for S>=8 -
see deviations). All waves coalesced into full S-slot batches
(coalesced_batch_max = S on every row).

| S | requests | decode-domain tok/s | wall tok/s | per-slot tok/s | step ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8.11 | 6.13 | 8.11 | 123.3 |
| 4 | 16 | 30.68 | 22.67 | 7.67 | 130.4 |
| 8 | 32 | 63.64 | 45.23 | 7.96 | 125.7 |
| 16 | 64 | 111.96 | 77.65 | 7.00 | 142.9 |
| 32 | 128 | 169.01 | 114.88 | 5.28 | 189.3 |

(Profiler-on companions: d1p 6.44 decode-domain / step 155.3; d8p 50.43 /
step 158.6 - the stage profiler costs ~20-26% at these speeds, so the stage
tables below are for relative structure, not absolute floors.)

**The curve is nearly flat from S=1 to S=8** (123-130 ms/step), then grows
(+17 ms to S=16, +64 ms to S=32). Per-slot tok/s peaks at S=1 with only
**8.11** - i.e. even a single slot pays ~123 ms/step. The decode step is a
launch/wait-latency floor, not a bandwidth problem, exactly as the s599
busy-time analysis predicted (2.94 ms GPU-busy vs 4.25 ms replay per layer).

Stage tables (mean ms/rank/layer over the steady tail; profiler-on):

| Stage | S=1 | S=8 | reads as |
|---|---:|---:|---|
| ep_window (envelope) | 1.586 | 1.531 | slot-flat (latency) |
| prefix_attn_output | 0.636 | 0.659 | slot-flat (launch-bound) |
| route_plan_pack | 0.423 | 0.449 | slot-flat (control latency) |
| ep_return_nccl | 0.350 | 0.375 | slot-flat (ring latency) |
| final_hc | 0.358 | 0.287 | ~flat |
| prefix_hc_current | 0.212 | 0.223 | slot-flat |
| prefix_attn_projection | 0.183 | 0.208 | ~flat |
| shared_swiglu_down | 0.051 | 0.241 | **scales with slots** (copy storm volume) |
| gate_up+down GEMMs | 0.115 | 0.242 | scales with routes |
| barriers (954/978/1144/1373) | 0.604 | 0.150 | idle-wait skew at S=1 |

The only stages that materially shrink with S are the volume-driven copies
and GEMMs; everything else is per-layer latency. That is why per-slot
throughput is nearly slot-count-flat below S=16.

### >=50 tok/s/slot budget analysis

Target: >=50 tok/s per slot <=> step <= 20 ms. Measured promoted-stack step:

- **Step floor measured this sprint: 123.3 ms (S=1)** = 2.87 ms/layer x 43.
  At S=8: 125.7 ms; S=16: 142.9 ms. The gap to 20 ms is **6.2-7.1x**.
- Itemized remaining per-layer step at S=8 (flags-off ~2.92 ms/layer):
  EP window ~1.53 (ep_return 0.37 + route_plan 0.45 + swiglu 0.24 + GEMMs
  0.24 + barriers 0.15, overlapped), pre-EP prefix ~1.1 (attn_output 0.66 +
  hc_current 0.22 + attn_projection 0.21 + attn_state/typed_history 0.18),
  final_hc 0.29.
- Demonstrated (race-blocked) levers from this sprint: relay return
  (0.61 -> 0.24 ms/layer) and batched exchange; full demonstrated stack =
  208.18 decode-domain at S=32 (step 153.7 ms, per-slot 6.51) - still 7.7x
  from the target.

**MTP acceptance multiplier required for >=50 tok/s/slot** (M = accepted
tokens per slot per step = 50 x step_time, assuming MTP leaves step time
unchanged - optimistic, since verify adds work):

| Base | step @ S=8 | M needed @ S=8 | step @ S=16 | M needed @ S=16 |
|---|---:|---:|---:|---:|
| Promoted (today) | 125.7 ms | **6.3** | 142.9 ms | **7.1** |
| Demonstrated relay+batched (blocked) | ~102-126 ms (est.) | ~5.1-6.3 | ~118-143 ms (est.) | ~5.9-7.1 |
| 80 tok/s/slot ideal | 125.7 ms | 10.1 | 142.9 ms | 11.4 |

**Explicit statement for the MTP gate:** MTP block-2 (the s216-s224
machinery) yields at most ~2-3 accepted tokens/step/slot at realistic
acceptance - **>=50/slot is NOT reachable via the MTP multiplier alone on
the current base step**. The required M of 6.3-7.1 means the base step must
first fall to ~40-60 ms (a further 2-3x beyond even the demonstrated
race-blocked stack) for MTP x2.5-3 to close the gap. The step is
launch/wait-bound, so that 2-3x lives in (a) the race fix unlocking the
EP-window stack (-0.6 to -0.9 ms/layer demonstrated), (b) prefix
launch-latency compaction (~1.1 ms/layer pool), (c) route-plan shadowing
(~0.45 ms/layer pool), and (d) cross-layer graph consolidation (the
remaining per-layer replay/launch overhead). The MTP-reopen gate should
therefore be sequenced AFTER the race fix + at least one major step-time
lever, and targeted at S<=8 where per-slot is highest.

### Final 32-slot reference numbers (promoted stack, this binary)

decode-domain 169.01 / wall 114.88 (d32; rctl601 168.15/114.75, rctl601b
167.53/114.06 - band 167.5-169.0 vs the s600 170.18/117.08, -0.7 to -1.6%).

## Definition of Done

1. shm unblock verified; dedicated-comm creation proven - **done** (Phase 0:
   16 Gi tmpfs, twocomm PASS, dedicated head comm + 2 class comms init at
   490 MiB/GPU each).
2. Race verdict per candidate with event counts - **done**: a0/a1/a3 and
   b-sb all FAIL with full event histograms (every Simple-stress pairing
   diverges 128/128 at the same loci); a2 subsumed by a3. **Both A and B
   exhausted with evidence**; no kill configuration exists within scope.
   New positive localization: the racing collectives are the hc-class set.
3. Stack promotion per gates - **done, non-promotion recorded with the gate
   matrix** (relay+copy: perf PASS +11%, race-exposure FAIL 1.5 vs 1.0
   events/run, tolerance FAIL 4/6 runs); launcher defaults unchanged;
   rollbacks trivially retained (all new modes are opt-in envs).
4. Scaling curve + budget analysis - **done** (S=1/4/8/16/32 measured,
   stage tables at S=1/8, itemized 20 ms-gap budget).
5. Report - this document; commits are the orchestrator's.
6. MTP-gate statement - **done, explicit**: measured base step floor
   123.3 ms (S=1) / 125.7 ms (S=8) / 142.9 ms (S=16); required MTP
   acceptance multiplier 6.3 @ S=8 and 7.1 @ S=16 (10.1/11.4 for the 80
   ideal) - not reachable by MTP alone; sequence the MTP gate after the
   race fix plus >=1 major step-time lever.

## Deviations (honest list)

1. a2 (COMM_SPLIT=epret alone) was not run - strictly weaker than the a3
   maximal split, whose FAIL subsumes it.
2. The c1/c2 relay+batched runs are tolerance-failing by construction (the
   batched exchange trips the race); they were run for perf/stage-table
   evidence only, exactly like the s599 stack run.
3. The jitter perturbation-stress runs (g3, staged) were not run - they gate
   a fix that did not land (same disposition as s600).
4. The nsys no-SYS spot-check for relay was not run - moot with
   non-promotion; relay avoids SYS by construction and the s598 proof for
   the promoted return stands.
5. The bench harness slot guard was relaxed to {1,2,4,8,16,24,32} and the
   deterministic submission wave now follows the slot count - 32-slot
   behavior is unchanged (wave=32), verified by rctl601 tolerance 1.0/1.0.
6. Phase D's S<32 points use 4 full waves (8 at S=1), i.e. shorter total
   windows than the 128-request reference; steady-state per-step times are
   the quantity of interest and are slot-count-flat per run.
7. ~25 minutes were lost to self-matching pgrep build-waiters (process
   hygiene; no GPU-run impact, no measurement impact).
7a. Phase D S<8 points initially failed: the fixture KV slot (Options
   default 7) is outside the slot count for S<8 (HTTP 500s). Engine fix:
   clamp to slots-1 in layer_runner.cu/layer_decode.cu (no-op for S>=8);
   S<8 points re-run on the fixed build3 while S>=8 points stand from
   build2 (behavior-identical at S>=8). The failed d4 server lingered
   holding the GPUs and aborted the d1/d8p preflights - killed by pid,
   idle re-verified, both re-run on build3.
8. The promoted-path control itself fired a token-level event (rctl601b) -
   recorded as an escalation fact; the s597 control anchor was NOT
   re-anchored and remains the tolerance reference.
9. One inadvertent laptop-side skill invocation (no effect on pod or repo).
10. The session was interrupted once by a laptop power loss; the pod kept
    running and all artifacts survived; c1/c2 were verified complete from
    their run trees before resuming.

## Artifacts

- Pod `/workspace/s601-artifacts/`: COMMANDS.md, provision.log, build1/2.log,
  phase0-twocomm.log, smoke-split/, phaseA-stage{1,2}-compare.txt,
  phaseB-compare.txt, run trees (rctl601, rctl601b, a0-sb-{1,2}, a1-sb-{1,2},
  a3-sb-{1,2,3}, b1-relay-ctl, b-sb-{1,2,3}, c1-relay-batched,
  c2-relay-batched-prof, g2-relay-copy-{2..6}, d1, d4, d8, d16, d32, d1p,
  d8p), drivers (s601-run.sh via /workspace/ds4/tools, phaseA-stage*.sh,
  phaseB-gates.sh, phaseC-*.sh, phaseD.sh, s601-nsys.sh staged unused).
- Laptop `logs/from-cluster/sprint601/`: COMMANDS.md, compare files,
  run-summaries.txt, per-run tolerance outputs, c2-stage-table-tail.txt,
  analyze-scaling.py, phase D stage tables.
- Source changes (uncommitted, orchestrator review): engine/runtime_options.cuh,
  runtime_types.cuh, runtime_resources.cu, runtime_pack.cu,
  runtime_profiler.cu, decode_loop.cu, hc_current.cu, router_step.cu,
  post_attention_ffn.cu, attention_output.cu, ep_compose.cu,
  tools/ds4-v100-run-tp-ep-appliance.sh, tools/ds4-v100-tp-ep-http-bench.sh,
  new tools/s601-run.sh, docs/sprints/SPRINT-601-REPORT.md.

## Follow-up seeds (for SPRINT-602 planning)

1. **Fully NCCL-free decode graph** (the only remaining in-repo race kill):
   replace the hc-class collectives (hc 3 allreduces + allgather, router
   allreduce, full-current broadcast, post-attn allgather) with kernel
   reductions/peer copies. The allgathers/broadcast are byte moves (bit-free
   wins); the allreduces are order-sensitive - a ring-order-exact kernel
   reduction (emulating the fixed NCCL_RINGS "0 3 2 1 5 7 6 4" chunk order)
   is required if the s597 bit anchor is to survive, else a control
   re-anchor decision. Unlocks the demonstrated 186.58 (relay+copy) and
   208.18 (relay+batched) stacks.
2. The promoted path's token-level exposure (rctl601b) raises the urgency:
   the debt now corrupts tokens on the default config.
3. NVIDIA escalation package: the s600 matrix plus this sprint's two new
   eliminations (comm isolation at all granularities; EP-window removal)
   sharpen the reproducer story to "RING_LL hc-class collectives in
   multi-GPU-per-process captured graphs".
4. C-C route-plan shadow (0.40-0.44 ms pool) and prefix launch-compaction
   (1.54 ms pool) remain the biggest non-MTP step-time levers after the
   race fix.
