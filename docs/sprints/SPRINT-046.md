# Sprint 046: Slot/Context Admission Envelope

## Status

Complete.

## Overview

Sprint 046 turns the current `aggregate_slot_context_envelope` blocker into a
runtime-enforced admission contract. The appliance should stop advertising only
an informal one-slot limit and instead expose a concrete, measured envelope for
slot/context tiers, reject requests outside that envelope, and record aggregate
timing evidence.

This sprint does not need to ship tensor-batched multi-slot execution in one
step. It must be honest about the current scheduler: one active decode at a
time unless a later active-microbatch sprint changes that. The goal is to make
memory fit, context tiers, configured slot modes, queue/reject behavior, and
MTP memory pressure explicit and gate-owned.

## Use Cases

- Operators can run an envelope gate for 1/2/4/8 configured slots at 128K,
  256K, 512K, and 1M context tiers.
- `/v100/status` reports configured slots, active microbatch capacity,
  queue/reject policy, context tokens, and whether tensor-batched slots are
  implemented.
- `/metrics` exposes slot/context admission fields and rejected request
  counters.
- Requests that exceed the configured context or ask for unsupported
  concurrency are rejected clearly instead of silently queueing or overfilling
  VRAM.
- The full gate no longer hides all remaining work behind the broad phrase
  `aggregate_slot_context_envelope`; it distinguishes admission/envelope from
  true active microbatch scheduling.

## Architecture

- Keep the resident replay runtime as the single execution owner for this
  sprint.
- Add a machine-readable planner/envelope output to the existing V100 planner
  rather than duplicating KV math in shell.
- Add replay service admission metadata:
  - `configured_slots`;
  - `active_microbatch`;
  - `queue_policy`;
  - `ctx_tokens`;
  - `scheduler_slots_ready`;
  - cumulative `rejected_requests`.
- Enforce admission in the HTTP path:
  - reject `tokens=0` or tokens above the existing hard max;
  - reject prompt plus generated tokens beyond configured context;
  - reject concurrent generation when `active_microbatch=1`;
  - keep health/status/metrics available while generation is busy.
- Add an aggregate envelope smoke that:
  - emits planner JSON/TSV for the required context tiers and slot targets;
  - starts the appliance in the conservative single-active mode;
  - sends repeated loopback requests and records aggregate tok/s and latency;
  - records one intentional rejection case.
- Keep the readiness ladder honest:
  - `slot_context_admission` can pass in this sprint;
  - true batched execution remains `active_microbatch_scheduler` until the
    scheduler runs multiple device-resident slots in one layer/stage pass.

## Parallel Work

Parallel sidecar agents should inspect:

- replay HTTP status/metrics/request-loop changes and reject semantics;
- planner/context memory math for the 1/2/4/8 slot and 128K/256K/512K/1M
  context matrix.

## Implementation

1. Extend `tools/ds4-v100-plan` with JSON or TSV envelope output for the
   required slot/context matrix, using the corrected DS4 compressed KV layout
   from `docs/architecture/DS4-V100-LAYOUT.md`.
2. Add replay CLI/env fields for configured slots, active microbatch, and
   queue policy. Defaults remain conservative: one configured slot, one active
   request, reject busy.
3. Extend `/v100/status` and `/metrics` with the new admission fields and
   rejection counters.
4. Implement reject-busy behavior around generation while leaving health,
   status, and metrics responsive.
5. Add `tools/ds4-v100-slot-context-envelope.sh` to produce planner artifacts,
   run the conservative serving smoke, and prove at least one rejection path.
6. Wire the new smoke into `tools/ds4-v100-gate.sh` as
   `slot_context_admission`.
7. Update operations docs, sprint report, follow-ups, and vision.

## Files Summary

- `tools/ds4-v100-plan.c`
- `tools/ds4-v100-replay.c`
- `tools/ds4-v100-slot-context-envelope.sh`
- `tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-gate.sh`
- `deploy/v100/ds4-v100-appliance.env.example`
- `docs/operations/DS4-V100-APPLIANCE.md`
- `docs/sprints/SPRINT-046-REPORT.md`
- `docs/sprints/SPRINT-046-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

## Definition of Done

- Local object compile passes for changed C files.
- Shell syntax checks pass.
- Planner JSON/TSV output covers 1/2/4/8 slots and 128K/256K/512K/1M context
  tiers.
- Service status and metrics expose configured slots, active microbatch,
  context, queue policy, and reject counters.
- A local or cluster smoke proves an over-admission request is rejected with a
  clear HTTP status and that normal generation still returns first token bytes
  `3136`.
- On the V100 cluster, the slot/context envelope smoke records planner
  artifacts, loopback aggregate timing, and rejection evidence.
- Full V100 gate includes `slot_context_admission PASS` and has no failures.
- Readiness reporting remains honest about the remaining gap to true
  tensor-batched active microbatch execution.
- Sprint report records commands, outputs, artifacts, timings, and remaining
  readiness blockers.

## Risks

- Planner admission is not the same as true tensor-batched execution. Status and
  readiness text must say this explicitly.
- Current runtime caches are sufficient for the short correctness fixture but
  do not yet prove a real 256K/512K/1M prompt replay. The envelope gate should
  separate memory admission from long-prompt correctness.
- Threaded or reject-busy serving must not allow two requests to mutate the
  same scheduler state at once.
- MTP sidecar memory on gpu7 must stay included in the MTP-on envelope.

## Security

No new external exposure. Serving remains loopback by default. Reject-busy and
over-context responses should not echo prompt text.

## Dependencies

- Sprint 045 MTP verify serving and metrics.
- `docs/architecture/DS4-V100-LAYOUT.md` corrected KV/cache estimates.
- Real base model, optional MTP sidecar, pack index, and 8x V100 cluster
  access for the final smoke.

## Outcome

Complete. `slot_context_admission` is now represented as a hard runtime contract with
explicit per-slot and per-context admission checks, queue/reject semantics, and status
gauge coverage.

The next unimplemented blocker remains `active_microbatch_scheduler`: requests are
admitted for multi-slot operation, but decode is not yet executed through true
tensor-resident slot batching.
