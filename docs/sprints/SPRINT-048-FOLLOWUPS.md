# Sprint 048 Follow-Ups

## P0: Cluster Functional Validation Under Load

- Run `tools/ds4-v100-replay --serve` on cluster with `--slots 2/4` and
  `--active-microbatch 2/4`.
- Fire concurrent one-token requests and confirm:
  - successful coalesced responses;
  - queue policy behavior (`reject-busy`, `sequential`);
  - no scheduler state corruption across repeated runs.

## P1: Throughput/Latency Evidence

- Measure aggregate tok/s and p95/p99 latency at:
  - slots `1/2/4/8`;
  - context tiers `128K/256K` first, then `512K/1M` where feasible.
- Capture per-stage decode and handoff timing deltas.

## P2: Broaden Batch Coverage

- Extend beyond first-token batching to token-step batching for multi-token
  continuations while preserving deterministic reset/checkpoint safety.
- Evaluate MTP scheduling interaction after base multi-token batching is stable.
