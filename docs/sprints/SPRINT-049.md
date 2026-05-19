# Sprint 049: Aggregate Throughput Envelope Evidence

## Status

Complete.

## Overview

Sprint 049 turns Level-6 work from code-only scaffolding into measured cluster
evidence. The sprint adds a dedicated aggregate-throughput harness and executes
it on the 8x V100 pod for representative slot/context tiers, then records the
results as sprint artifacts.

## Goals

1. Add a gate-compatible aggregate throughput script with concurrent load.
2. Measure p50/p95/p99 latency and aggregate tok/s under active-microbatch load.
3. Capture at least one focused MTP on/off throughput comparison.
4. Keep readiness reporting honest about uncovered tiers.

## Scope

- Add `tools/ds4-v100-aggregate-throughput.sh`.
- Wire a new gate rung `aggregate_slot_context_throughput` into
  `tools/ds4-v100-gate.sh`.
- Run cluster benchmarks from `/workspace/ds4-sprint049` on `llamacpp-build-8gpu`.
- Collect TSV/JSON/server logs under local `logs/from-cluster/sprint049*`.

## Out of Scope

- Full 1/2/4/8 slot matrix across all queue policies in one sprint.
- Long-prompt (true long prefill) context-tier quality/perf characterization.
- Multi-token token-step batch scheduling beyond first-token batch path.

## Definition of Done

- New throughput harness compiles/runs in cluster environment.
- Gate references the new rung.
- Cluster artifacts include successful multi-case runs with no request errors.
- Sprint report records measured numbers and remaining blockers explicitly.
