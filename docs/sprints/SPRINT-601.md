# Sprint 601 - Kill the NCCL Race (Comm Isolation / NCCL-Free EP), Measure Slot Scaling

Date: 2026-06-12
Status: planned

## Goal

Three deliverables, in priority order:

1. **Fix or remove the captured-NCCL race** (Sprint 600 root cause: a
   timing-dependent race inside the captured collectives on the single
   8-rank-per-process communicator; fires on the promoted path ~1/256
   steps). The leading candidate — communicator isolation — was blocked
   only by the pod's 64 MiB /dev/shm k8s default, **now lifted to 16 Gi**
   (manifest updated, pod recreated). If isolation does not kill it, fall
   to the NCCL-free EP window.
2. **Unlock and promote the fast-exchange stack** the fix enables (s599
   demonstrated +11.5-17.9%; plus C-B restack and C-C route-plan shadow).
3. **Measure the empirical slot-scaling curve** (S=1/4/8/16/32) with the
   final stack — the program's ≥50 tok/s/slot trajectory is per-slot ≈
   1/step-time (slot-flat, measured 3x), so this curve grounds the MTP
   decision gate that follows this sprint.

## Program context (clarified target)

≥50 tok/s PER SLOT (80 ideal) ⇔ step ≤ 20 ms. Today 170.18 decode-domain at
32 slots = ~183 ms step (per-slot ~5.3). Bandwidth caps per-slot ~47 at 32
slots → the target lives at S ≤ 16. This sprint's job is to push step time
down and produce the measured scaling curve; the MTP gate (next) decides
the multiplier path.

## Phase 0 - Verify the shm unblock

- Confirm /dev/shm=16Gi in the recreated pod; rebuild if the pod image needs
  re-provisioning (apt installs do not survive recreation — re-run the
  Phase -1 provisioning from s597 COMMANDS.md; /workspace persisted).
- Re-run the s600 twocomm/splitShare standalone test: dedicated communicator
  creation must now succeed.

## Phase A - Communicator-isolation matrix (the fix candidate family)

Use the **fastest deterministic reproducer** from s600 (NCCL_PROTO=Simple
fires ~every step; batched exchange ~4/256; per-step first-divergence
checksums detect events exactly):

1. **Dedicated head comm** (the blocked s600 candidate): head collectives on
   their own communicator; EP/dense collectives on the main comm.
2. If the race persists: **split by collective class** (EP-return comm vs
   HC/allgather comm vs head comm) — NCCL's caution is about many
   collectives on ONE comm inside capture; isolation granularity is the
   experiment.
3. If isolation helps but incompletely: combine with pacing only where
   measured-necessary.

Race gate for any candidate: **zero corruption events across ≥ 3 ×
256-step runs under the harshest reproducer config** (Simple + batched
exchange), then zero events at the promoted config, then tolerance 1.0/1.0
vs the s597 control. VRAM check: each extra comm costs device memory
(s598 measured +848-944 MiB/GPU for one comm) — budget against the ~1.3 GiB
free; if two extra comms don't fit, prefer class-split over per-layer
splits.

## Phase B - If isolation fails: NCCL-free EP window

Remove 9/16 collectives per rank-layer from the captured window: batched
swiglu exchange (exists, flag) + EP return via the s597 relay-table
peer-write kernels (NVLink one-hop for the 12 SYS pairs — the s597 Phase 1
relay table is the input). Same race gate (the remaining collectives must
not fire), same tolerance gate. s600 projected ~195-205 decode-domain for
this configuration.

## Phase C - Promote the unlocked stack

With the race dead (A or B): promote the best swiglu exchange (≥ +8% over
the fixed baseline, no-SYS spot-check), restack C-B, attempt C-C route-plan
shadow. Final reference-shape measurement at 32 slots.

## Phase D - Slot-scaling curve (one bench session)

With the final stack: S = 1, 4, 8, 16, 32 at 256K, deterministic, same
harness — decode-domain, wall, per-slot tok/s, step time, and the s597
profiler stage table at S=1 and S=8 (to see which stages actually shrink).
Deliverable: the measured per-slot-vs-S curve and the updated step-time
budget table for the ≥50/slot program (what remains between the measured
step and 20 ms, itemized by stage).

## Definition of Done

1. shm unblock verified; dedicated-comm creation proven.
2. Race verdict per candidate with event counts (the checksum tool is the
   gate, not tolerance alone); the kill configuration named, or both A and
   B exhausted with evidence.
3. Stack promotion per gates (race-zero, tolerance, ≥+8%, no-SYS);
   launcher defaults flipped with rollbacks retained; or non-promotion
   evidence recorded.
4. Scaling curve measured and archived; per-slot table + stage tables at
   S=1/8; updated ≥50/slot budget analysis written.
5. Report (SPRINT-601-REPORT.md) with DoD checklist + deviations;
   follow-ups; orchestrator docs/commits.
6. Explicit statement for the MTP gate: the measured base step-time floor
   achieved, and the acceptance multiplier MTP would need for ≥50/slot at
   S=8 and S=16.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Comm isolation doesn't kill the race (it's intra-comm per-collective) | Med | Med | Phase B is the planned fallback, not an improvisation |
| Extra comms blow the VRAM budget | Med | Med | Class-split granularity; measure free VRAM per comm added |
| Relay-table peer-write return reintroduces ordering hazards | Med | High | It removes collectives but adds peer-write edges — gate with the checksum tool at Simple-stress level |
| Scaling curve confounded by ramp/admission at low S | Low | Low | Steady-state windows, full-batch waves per harness; report occupancy per run |

## Dependencies

- Pod recreated with 16 Gi /dev/shm (done, 2026-06-12); /workspace intact;
  re-provision apt packages per s597 COMMANDS.md.
- s600 reproducers, checksum tooling, twocomm test, dot dumps
  (`/workspace/s600-artifacts/`, `logs/from-cluster/sprint600/`).
- s597 relay table (`logs/from-cluster/sprint597-phase01/`).
- HEAD: everything through Sprint 600 committed (3e5a391d).
