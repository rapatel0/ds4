# Sprint 120 - Single-Slot Shared SwiGLU Row-Pair Probe

Date: 2026-05-21

## Objective

Test whether the per-slot shared gate/up/SwiGLU fusion from Sprint 117 can be
rescued with a row-pair kernel. This is a narrow opt-in probe before attempting
a deeper SM70 software-pipelined F8 kernel.

## Implementation

1. Add `arena_f8_e4m3_b128_pair_swiglu_rows2_kernel`.
2. Dispatch it behind `DS4_CUDA_F8_PAIR_SWIGLU_SINGLE_ROWS2=1`.
3. Expose the launcher knob as
   `DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE_ROWS2=0`.
4. Extend `tests/cuda_f8_hmma_pair_swiglu_smoke` so the new single-token
   row-pair path is compared against the scalar single-token reference.

The path only has an effect when `DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE=1` is
also enabled.

## Results

Focused correctness:

- `tests/cuda_f8_hmma_pair_swiglu_smoke`: passed.
- `DS4_CUDA_F8_PAIR_SWIGLU_SINGLE_ROWS2=1 tests/cuda_f8_hmma_pair_swiglu_smoke`:
  passed.

Same-binary 8-slot/256K served A/B on the 8x V100 node:

| Mode | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---|
| Current default | `34.490294` | `32.334651` | 8/8 token match |
| Single scalar fusion | `34.689964` | `32.521841` | 8/8 token match |
| Single row-pair fusion | `34.380968` | `32.232157` | 8/8 token match |

Artifacts:

- `logs/from-cluster/sprint120-rows2/default/summary.json`
- `logs/from-cluster/sprint120-rows2/single-scalar/summary.json`
- `logs/from-cluster/sprint120-rows2/single-rows2/summary.json`

## Decision

Keep the row-pair single-fusion path opt-in/off. It is correct, but it does not
beat the current default. The scalar single-fusion result is slightly higher in
this run, but not enough to promote because Sprint 117 previously measured it
neutral and the current delta is inside the noise band.

Next kernel work should move to a real SM70 software-pipelined F8 mainloop or
TurboMind grouped expert scheduling. More row-count compaction is unlikely to
close the utilization gap.
