# Sprint 047 Report: Active-Microbatch Scheduler Core

## Result

`SHIP`.

Scheduler-level multi-slot execution primitives are now implemented. The stage
scheduler owns per-slot KV/HC state and can run slot batches through stage
decode and stage handoff APIs.

## Implemented

- `ds4_v100_scheduler.h/.c`
  - Added `DS4_V100_SCHED_MAX_SLOTS`.
  - Added batch APIs:
    - `ds4_v100_stage_scheduler_decode_token_batch`
    - `ds4_v100_stage_scheduler_decode_hc_batch`
    - `ds4_v100_stage_scheduler_handoff_batch`
  - Refactored scheduler state to slot-aware storage:
    - cache slices by `[layer][slot]`
    - HC ping-pong buffers by slot
  - Kept existing single-slot APIs as wrappers.
  - Kept checkpoint/snapshot paths explicit single-slot only for now.

- Scheduler smoke coverage
  - `tests/cuda_v100_stage_scheduler_smoke.c`:
    - `--slots N`
    - exercises `decode_token_batch` for `N > 1`.
  - `tests/cuda_v100_full_scheduler_smoke.c`:
    - `--slots N`
    - exercises batched stage decode and batched cross-stage handoff.

- Gate
  - `tools/ds4-v100-gate.sh` now includes:
    - `active_microbatch_scheduler`
    - runs `tests/cuda_v100_full_scheduler_smoke ... --slots 2`
    - adds readiness missing key `active_microbatch_scheduler` when absent.

- Service status/metrics
  - `tools/ds4-v100-replay.c` now reports:
    - `"scheduler_slots_ready": true`
    - `ds4_v100_scheduler_slots_ready 1`
  - Still reports `"tensor_batched_slots": false`.

## Verification (local compile/syntax)

```bash
cc -fsyntax-only -I. ds4_v100_scheduler.c
cc -fsyntax-only -I. ds4_v100_replay.c
cc -fsyntax-only -I. tools/ds4-v100-replay.c
cc -fsyntax-only -I. tests/cuda_v100_stage_scheduler_smoke.c
cc -fsyntax-only -I. tests/cuda_v100_full_scheduler_smoke.c
bash -n tools/ds4-v100-gate.sh
```

## Remaining Gap

The HTTP request loop still serializes independent prompts. This sprint ships
the scheduler substrate; the next sprint should integrate active microbatch
scheduling in the request loop and produce multi-slot throughput evidence.
