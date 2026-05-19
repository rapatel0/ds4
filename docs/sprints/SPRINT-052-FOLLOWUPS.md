# Sprint 052 Follow-ups

## Continuous Multi-token Batching

- **Severity**: Critical
- **Target sprint**: Sprint 053
- **Files**: `tools/ds4-v100-replay.c`, `ds4_v100_replay.c`,
  `ds4_v100_replay.h`, `ds4_v100_scheduler.c`, `ds4_v100_scheduler.h`
- **Issue**: The sustained decode baseline still shows low GPU utilization
  because multi-token requests fall through the non-batched generation path.
  Request concurrency alone does not create enough effective work.
- **Evidence**:
  `logs/from-cluster/sprint052-sustained-baseline/sustained_decode.tsv`
  measured `3.304551` aggregate generated tok/s and `10.804%` average GPU
  utilization for 1M context, one slot, 16 tokens/request.
- **Next step**: Implement token-step batching that keeps active request state
  resident across continuation tokens.

## Per-GPU Utilization Reporting

- **Severity**: Useful
- **Target sprint**: Sprint 053 or next benchmark rerun
- **Files**: `tools/ds4-v100-sustained-decode-bench.sh`
- **Issue**: The first Sprint 052 artifact includes raw `gpu_util.csv` and
  combined GPU utilization summary. The script now also emits per-GPU summary
  for future runs, but the first copied result JSON predates that parser
  refinement.
- **Next step**: Rerun the sustained benchmark after Sprint 053 to capture
  per-GPU averages and maxima in JSON.
