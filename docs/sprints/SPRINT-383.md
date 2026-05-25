# Sprint 383: VRAM-Aware Active-Slot Baseline

## Overview

Re-baseline the current TP/EP serving path at the target `32` slot / `256K`
shape with both GPU utilization sampling and the new Sprint 382 VRAM telemetry.

This sprint is a measurement sprint. It does not optimize kernels. It creates
the authoritative before-state for the next launch/synchronization or kernel
fusion sprint.

## Rationale

`TEMP_THROUGHPUT_PROMPT.md` says the current TP/EP path is launch/sync-bound:
server decode and GPU utilization stay flat as active requests rise from `1`
to `32`. Sprint 382 added memory admission so future measurements also know
whether the target shape is operating safely inside 32GB.

Before changing the steady-state decode loop again, we need a fresh baseline
that records:

- active request count,
- coalesced batch size,
- client and server decode throughput,
- average/max GPU utilization,
- max observed GPU memory,
- CUDA allocator free-memory margin from `cudaMemGetInfo`.

## Scope

- Extend `tools/ds4-v100-tp-ep-active-slot-matrix.py` to carry VRAM summary
  fields from each profile run into the aggregate TSV/JSON.
- Add an explicit inter-case cooldown so repeated resident server startups do
  not immediately re-enter CUDA context creation after teardown.
- Run the active-slot matrix on gpu-01 at:
  `32` configured slots, `256K` context, `position=262080`, `32`
  generated tokens, active request cases `1,4,8,16,32`.
- Enable GPU sampling and VRAM report/admission for every case.
- Record the matrix artifact path and a concise table in the sprint outcome.

## Out Of Scope

- No PP/layer-split work.
- No kernel changes.
- No CUDA graph changes.
- No MTP changes.
- No promotion/rejection of performance gates; this sprint only establishes
  the post-Sprint-382 baseline.

## Definition Of Done

- Local syntax checks pass for the matrix/profile scripts.
- V100 matrix run completes at the target shape for all requested active cases.
- Aggregate matrix TSV/JSON include VRAM fields:
  `vram_min_free_mib`, `vram_max_used_mib`, `vram_threshold_mib`, and
  `vram_failures`.
- If a no-cooldown run fails during repeated startup, record that evidence and
  rerun the remaining cases with cooldown instead of hiding the failure.
- The sprint outcome records the throughput/utilization/memory table and
  identifies the next implementation gate.

## Risks

- The full HTTP matrix is slow because every case performs resident startup.
- A case may fail memory admission after Sprint 382; if so, the failure is
  itself useful and should be recorded instead of bypassed.
- GPU utilization sampling is coarse; use it for directional imbalance, not
  per-kernel attribution.

## Outcome

Implemented and validated.

Code changes:

- `tools/ds4-v100-tp-ep-active-slot-matrix.py` now carries per-case VRAM
  fields into `active_slot_matrix.tsv` and `active_slot_matrix.json`.
- Added `--case-cooldown-seconds N` to the matrix runner. This is needed
  because repeated resident TP/EP server startups can fail immediately after a
  previous teardown with CUDA OOM at context creation even when `nvidia-smi`
  later shows all GPUs free.

V100 evidence:

Artifacts:

```text
/workspace/logs/sprint383-vram-aware-matrix/
/workspace/logs/sprint383-vram-aware-matrix-retry/
/workspace/logs/sprint383-vram-aware-matrix-retry2/
/workspace/logs/sprint383-vram-aware-matrix-combined/
```

The first no-cooldown matrix completed `requests=1`, then failed starting
`requests=4` with:

```text
cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:3187: out of memory
```

The first retry completed `requests=4`, then failed starting `requests=8` with
the same startup OOM. The cooldown-enabled retry completed `8,16,32`.

Combined matrix:

| Active requests | HTTP 200 | Client tok/s | Server decode tok/s | Avg GPU util | Max GPU util | Max memory | Min free VRAM | VRAM failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 1.321769 | 97.749438 | 8.540625% | 40.0% | 32398 MiB | 1754 MiB | 0 |
| 4 | 4/4 | 5.502740 | 97.092452 | 8.600694% | 39.0% | 32398 MiB | 1754 MiB | 0 |
| 8 | 8/8 | 10.891140 | 96.330654 | 8.375000% | 39.0% | 32398 MiB | 1754 MiB | 0 |
| 16 | 16/16 | 20.026059 | 92.690622 | 8.238372% | 39.0% | 32398 MiB | 1754 MiB | 0 |
| 32 | 32/32 | 43.853691 | 97.076706 | 9.292763% | 40.0% | 32398 MiB | 1754 MiB | 0 |

## Decision

The current TP/EP bottleneck is still steady-state launch/synchronization and
GPU0-heavy orchestration, not active-request admission and not memory
capacity. Client throughput scales because more requests amortize the fixed
decode step, but server decode remains flat around `92-98` tok/s and average
GPU utilization remains below `10%`.

The repeated no-cooldown startup OOM is also real operational evidence. Keep
the matrix cooldown available and use VRAM admission in all future serving
baselines.

Next implementation sprint should attack steady-state sync/launch overhead
with a narrow gate and should preserve the new matrix as the before/after
measurement.
