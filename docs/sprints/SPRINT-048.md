# Sprint 048: Request-Loop Active Microbatch Integration

## Status

Complete.

## Overview

Sprint 047 shipped slot-aware scheduler primitives but HTTP generation still ran
request-by-request. Sprint 048 integrates active-microbatch execution into the
serve loop so concurrent requests can be coalesced through the scheduler batch
APIs.

## Goals

1. Wire request-loop generation to a pending-request batch path.
2. Use scheduler batch decode/handoff for non-MTP one-token requests.
3. Keep admission behavior (`reject-busy` vs `sequential`) explicit.
4. Preserve correctness-first fallback for non-batchable requests.

## Scope

- `tools/ds4-v100-replay.c`:
  - add pending request queue state and batch processing helpers;
  - route HTTP generation through pending-batch execution;
  - keep MTP-enabled or multi-token requests on per-request fallback path;
  - keep slot admission separate from generation critical section.
- `tests/cuda_v100_full_scheduler_smoke.c`:
  - warning cleanup for `--slots` path (`n_slots` initialization order).

## Out of Scope

- Cluster throughput claims for 2/4/8 slots.
- Tensor-batched fused kernels (`tensor_batched_slots` remains false).
- MTP plus active-microbatch co-scheduling claims.

## Definition of Done

- `tools/ds4-v100-replay.c` compiles with pending-batch serve integration.
- Active-microbatch admission no longer implies whole-request serialization.
- Local build checks are warning-clean for touched targets.
