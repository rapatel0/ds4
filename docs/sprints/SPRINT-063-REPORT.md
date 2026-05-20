# Sprint 063 Report: Stage Wavefront Scheduler Proof

## Result

`SHIP`.

Sprint 063 delivered the scheduler primitives needed for stage wavefronting and
proved the two-stage slot-lane mechanics on the V100 cluster.

## Code Changes

- `ds4_v100_scheduler.h` / `ds4_v100_scheduler.c`
  - Added `ds4_v100_stage_scheduler_decode_token_slot_span`.
  - Added `ds4_v100_stage_scheduler_decode_hc_slot_span`.
  - Added `ds4_v100_stage_scheduler_handoff_slot_span`.
  - Added `ds4_v100_stage_scheduler_read_hc_slot`.
  - Kept existing zero-based APIs as wrappers.
  - Added layer context to decode error messages.
- `ds4_cuda.cu`
  - Changed the CUDA temp scratch buffer from one global pointer to per-device
    scratch slots. This is required before independent GPU stages can have
    in-flight async work without clobbering scratch ownership.
- `tests/cuda_v100_stage_wavefront_smoke.c`
  - Added a two-stage V100 smoke that runs a serial reference, then advances
    slot lanes in wavefront order and compares final HC state.
- `Makefile`
  - Added the new CUDA smoke target.

## Validation

Local:

- `cc -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99 -I. -c -o /tmp/ds4_v100_scheduler.o ds4_v100_scheduler.c`
- `cc -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99 -I. -D_FILE_OFFSET_BITS=64 -c -o /tmp/cuda_v100_stage_wavefront_smoke.o tests/cuda_v100_stage_wavefront_smoke.c`
- `make ds4_v100_scheduler.o tests/cuda_v100_stage_wavefront_smoke.o`
- `git diff --check`

V100 pod:

- `CUDA_ARCH=sm_70 make tests/cuda_v100_stage_wavefront_smoke tests/cuda_v100_full_scheduler_smoke tests/cuda_v100_selected_token_smoke`
- `cuda_v100_stage_wavefront_smoke: token0=16 token1=926 max_abs_slot0=0 max_abs_slot1=0 ok`
- `cuda_source_dtypes_smoke: ok`
- `cuda_v100_full_scheduler_smoke --slots 2: ok`
- `cuda_v100_selected_token_smoke: selected=926 expected=3136 ok`

## Decision

Proceed to served wavefronting behind an opt-in flag.

The smoke does not yet prove throughput improvement. It proves the required
mechanics:

- nonzero slot lanes can be addressed directly,
- stage handoff can target a slot span,
- slot-specific HC can be read for parity,
- per-device CUDA scratch no longer prevents independent GPU-stage work,
- the serial reference and wavefront lane order match exactly for the tested
two-stage path.

Sprint 064 should wire the same primitives into the replay server's same-length
non-MTP batch path. The first target comparison should reuse the Sprint 062
bench shape:

- `ctx=1048576`, `slots=2`, `tokens=16`, `requests=4`
- `ctx=262144`, `slots=4`, `tokens=16`, `requests=4`

Success is not just correctness; the opt-in path should beat the current
`~3.7-3.8` generated tok/s profiled baseline before becoming the default.

## Artifacts

- `logs/from-cluster/sprint063-wavefront/wavefront_smoke.log`
