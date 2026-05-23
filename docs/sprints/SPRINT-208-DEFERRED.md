# Sprint 208 Deferred Items

Date: 2026-05-23

## Production TP8 Scheduler Integration

- **What**: Implement full TP8 model serving with token generation, request
  batching, sharded KV, attention, routed/shared FFN, output head, and
  launcher integration in a TP-only runtime path.
- **Why deferred**: Sprint 208 first needs hard evidence that TP8 collectives
  and sharded KV are viable at 32 slots and 128K-256K context. No scheduler
  implementation belongs in Sprint 208.
- **Target sprint**: Sprint 209+ if Sprint 208 gates pass.
- **Prerequisites**: Positive TP8 planner, collective, resident-boundary, and
  sharded-KV evidence.
- **Files**: `ds4_v100_tp_scheduler.*`, TP runtime entry, TP pack descriptors,
  TP tests.

## PP Scheduler Generalization

- **What**: Refactor `ds4_v100_scheduler.*` into a generic scheduler that can
  support both PP/layer and TP execution.
- **Why deferred**: The user explicitly rejected this direction. TP is
  sufficiently different and should not fight PP abstractions.
- **Target sprint**: Not planned.
- **Prerequisites**: None. This is intentionally not on the roadmap.
- **Files**: `ds4_v100_scheduler.*`, `ds4_v100_tp_scheduler.*`.

## MTP On TP8

- **What**: Add speculative/MTP draft verify/commit to a TP8 runtime.
- **Why deferred**: MTP can hide base-forward problems and should follow a
  coherent TP8 base forward. Sprint 208 only investigates topology viability.
- **Target sprint**: Future after TP8 base decode or after PP path MTP is
  production-ready.
- **Prerequisites**: TP8 base forward correctness and throughput evidence.
- **Files**: `ds4_v100_mtp.*`, TP scheduler/runtime files.

## Full TP8 Sharded Attention Implementation

- **What**: Implement production DS4 attention over TP-sharded KV/cache,
  including compressed KV row selection and ratio-4/ratio-128 behavior.
- **Why deferred**: Sprint 208 may define KV shard descriptors and memory gates,
  but the first evidence target is collective/scheduler viability.
- **Target sprint**: Sprint 209+ if TP8 boundary gates pass.
- **Prerequisites**: TP8 KV ownership descriptor and positive boundary results.
- **Files**: TP attention kernels, TP KV descriptors, TP scheduler files.

## TP Pack Converter

- **What**: Build a full offline converter that writes production TP8 shard
  files with split-axis metadata for dense, routed, shared, output, and KV
  ownership.
- **Why deferred**: The planner and boundary probes should first confirm which
  TP topology is worth packing for.
- **Target sprint**: Sprint 209+ after topology decision.
- **Prerequisites**: Selected TP topology and shard-axis policy.
- **Files**: new TP pack tool, manifest schema docs, pack tests.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| Production TP8 scheduler integration | Sprint 209+ | Sprint 208 gates |
| PP scheduler generalization | Not planned | Explicitly rejected direction |
| MTP on TP8 | Future | TP8 base forward |
| Full TP8 sharded attention | Sprint 209+ | KV ownership + boundary gates |
| TP pack converter | Sprint 209+ | Selected topology and split policy |
