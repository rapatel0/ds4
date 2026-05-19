# Sprint 053 Follow-ups

## Hot-Path Kernel Occupancy

- **Severity**: Critical
- **Target sprint**: Sprint 054
- **Files**: `ds4_v100_scheduler.c`, `ds4_v100_layer_execute.c`,
  `ds4_cuda.cu`, `tools/ds4-v100-replay.c`
- **Issue**: Same-length token-step batching is wired and proven, but
  throughput only improved by about `2.4%` from one to two active slots and GPU
  utilization stayed near `11%`.
- **Evidence**:
  `logs/from-cluster/sprint053-token-step-batching/sustained_decode.tsv`
  measured `3.371659` generated tok/s and `11.133%` average GPU utilization
  for the two-slot case, despite `tensor_batched_groups=1`,
  `tensor_batched_requests=2`, and `tensor_batched_tokens=32`.
- **Next step**: Replace or fuse the hottest routed/shared FFN and projection
  paths with the strongest available V100 low-bit kernels, then rerun the same
  sustained matrix.

## Larger Queue-Depth Batch Shape

- **Severity**: Important
- **Target sprint**: Sprint 054 or Sprint 055
- **Files**: `tools/ds4-v100-sustained-decode-bench.sh`,
  `tools/ds4-v100-replay.c`
- **Issue**: The Sprint 053 proof used `requests=4` and `slots=2`, which was
  enough to prove branch execution but not enough to characterize 4/8-slot
  behavior or queue-depth sensitivity.
- **Evidence**: Only one two-request batch group was captured in the two-slot
  run.
- **Next step**: After hot-path kernel work, rerun sustained profiles with
  `slots=2,4,8`, `requests>=16`, and short/medium context tiers to separate
  batching benefit from 1M-context bandwidth pressure.

## Mixed-Length Scheduling

- **Severity**: Important
- **Target sprint**: After hot-path improvement
- **Files**: `tools/ds4-v100-replay.c`, `ds4_v100_replay.c`,
  `ds4_v100_scheduler.c`
- **Issue**: Sprint 053 only batches requests with identical requested token
  counts. Practical serving will need live slots to retire independently while
  other slots continue.
- **Evidence**: The server intentionally falls back to serial generation when
  pending requests have mixed `tokens` values.
- **Next step**: Add an active-slot mask and per-slot remaining-token counts
  once kernel occupancy is high enough for this extra scheduler complexity to
  matter.

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Hot-path kernel occupancy | Critical | Sprint 054 | `ds4_v100_scheduler.c`, `ds4_v100_layer_execute.c`, `ds4_cuda.cu`, `tools/ds4-v100-replay.c` |
| Larger queue-depth batch shape | Important | Sprint 054 or Sprint 055 | `tools/ds4-v100-sustained-decode-bench.sh`, `tools/ds4-v100-replay.c` |
| Mixed-length scheduling | Important | After hot-path improvement | `tools/ds4-v100-replay.c`, `ds4_v100_replay.c`, `ds4_v100_scheduler.c` |
