# Sprint 047 Follow-Ups

## P0: Request-Loop Active Microbatch Integration

- Complete in Sprint 048 (`docs/sprints/SPRINT-048.md`).
- `tools/ds4-v100-replay --serve` now uses pending-request batch dispatch for
  non-MTP one-token requests.
- Remaining expansion is multi-token token-step batching (tracked in
  Sprint 048 follow-ups).

## P1: Slot Throughput Evidence

- Run slot/context sweeps (1/2/4/8 slots; 128K/256K/512K/1M where feasible).
- Record aggregate tok/s, p95/p99 latency, and per-stage timing.
- Compare `reject-busy` and `sequential` policies under identical load.

## P2: Batch-Kernel Uplift

- Identify highest-impact kernels for true tensor-batched slot execution.
- Keep `scheduler_slots_ready` true; advance `tensor_batched_slots` only when
  tensor-batched kernels are selected and validated.
