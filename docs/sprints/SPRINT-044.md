# Sprint 044: Throughput Optimization And Operating Envelope

## Status

Complete.

## Overview

Sprint 044 closes the current `throughput_optimization` blocker by turning
Sprint043 timing diagnostics into an explicit operating-envelope artifact and
shipping the first real optimization. The optimization target is fresh-process
stage open/upload, which currently dominates operator startup at roughly
289-345 seconds per service start.

The sprint should preserve the base one-slot correctness path and the MTP
correctness gates. It should not claim multi-slot throughput until admission
and scheduling are implemented.

## Use Cases

- Measure startup/upload and decode timing in a repeatable benchmark artifact.
- Compare serial stage open/upload against parallel stage open/upload on the
  same 8x V100 host.
- Keep the default appliance path optimized without removing a serial fallback.
- Report context/slot limits honestly, even where only admission analysis is
  available.
- Move the full gate from `missing=throughput_optimization` to the next honest
  blocker.

## Architecture

- Add parallel stage open/upload inside `ds4_v100_replay_open`:
  - tokenizer open, model mmap, and `ds4_gpu_set_model_fd` stay single-threaded;
  - each stage opens its own context, pack index, device arena, upload chunk,
    KV/cache tensors, and HC tensors in a worker thread;
  - each worker writes only its `scheds[stage]`, `open_ms[stage]`, and private
    error buffer;
  - all workers are joined before replay is returned;
  - serial open remains available as a fallback for debugging and before/after
    benchmarking.
- Add a replay `--open-only` path for cheap startup benchmarks that avoid
  prompt replay and generation after resident upload.
- Add a benchmark script that:
  - runs serial open-only;
  - runs default parallel open-only;
  - optionally runs one normal two-token replay;
  - emits a report with open totals, per-stage open timings, speedup,
    decode timing, and an explicit verdict.
- Wire the benchmark into the gate as `throughput_optimization`.

## Parallel Work

Parallel sidecar agents should inspect:

- data-race risk in CUDA globals and stage scheduler open;
- benchmark/report shape and honest readiness semantics.

## Implementation

1. Extend `ds4_v100_replay_options` and `tools/ds4-v100-replay`:
   - `--serial-open`;
   - `--open-only`;
   - JSON output for open-only timing.
2. Implement threaded stage open/upload in `ds4_v100_replay_open`.
3. Add `tools/ds4-v100-throughput-bench.sh`.
4. Add a focused gate step in `tools/ds4-v100-gate.sh` that runs the benchmark
   with the real model and pack index.
5. Update operations docs, sprint report, follow-ups, and vision.

## Files Summary

- `ds4_v100_replay.h`
- `ds4_v100_replay.c`
- `tools/ds4-v100-replay.c`
- `tools/ds4-v100-throughput-bench.sh`
- `tools/ds4-v100-gate.sh`
- `docs/operations/DS4-V100-APPLIANCE.md`
- `docs/sprints/SPRINT-044-REPORT.md`
- `docs/sprints/SPRINT-044-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

## Definition of Done

- Local object compile passes for changed C files.
- Shell syntax checks pass.
- `tools/ds4-v100-replay --help` documents `--open-only` and `--serial-open`.
- On the V100 cluster, serial open-only and parallel open-only benchmark runs
  both pass.
- Benchmark report records serial open total, parallel open total,
  per-stage timings, speedup ratio, and verdict.
- A normal two-token replay still returns first-token bytes `3136`.
- Full V100 gate includes `throughput_optimization PASS`, has no failures, and
  no longer reports `missing=throughput_optimization`.
- Sprint report records before/after timing, commands, artifacts, and remaining
  readiness blockers.

## Risks

- CUDA global model-cache state is process-global. Keep model mapping and file
  descriptor registration single-threaded before stage workers start, and avoid
  invoking optional model range caches in open workers.
- Concurrent arena allocation/upload may expose memory pressure or driver
  serialization. Keep `--serial-open` as a fallback and stop if correctness or
  reserve checks regress.
- A faster startup does not imply multi-slot throughput. The report must keep
  decode throughput, slot admission, and context tiers distinct.

## Security

No new external serving surface. Benchmark scripts run local loopback or CLI
paths only.

## Dependencies

- Sprint043 production deployment package.
- Real base model, pack index, and 8x V100 cluster access.

## Open Questions

- If parallel open/upload is unstable or slower on the V100 driver, should the
  sprint switch to a benchmark-only operating envelope and defer optimization?
- Should the next optimization after startup target output-head projection,
  MTP serving integration, or slot batching?
