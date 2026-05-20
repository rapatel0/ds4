# Sprint 076: Parallel Output-Head Top-1 Reducer

## Status

Complete.

## Overview

Sprint 075 proved that device-resident greedy output selection is correct but
not useful when implemented as a single CUDA thread scanning all `129280`
vocabulary logits. That candidate regressed output-head timing from
`346.461 ms` to `423.818 ms`, so it stayed opt-in.

Sprint 076 turned that failed candidate into a real throughput probe by
replacing the serial scan with a deterministic parallel F32 top-1 reducer. The
V100 evidence met the default-change rule, so the parallel device top-1 path is
now the greedy `k == 1` default with `DS4_V100_DISABLE_OUTPUT_HEAD_FASTPATH=1`
as the rollback switch.

## Goals

1. Replace the serial `top1_f32_serial_kernel` with a parallel two-stage CUDA
   reducer.
2. Preserve deterministic tie handling: larger logit wins, exact ties keep the
   lower token id, matching the host scan order.
3. Preserve host-path fail-closed behavior by detecting any non-finite logit in
   the parallel reducer.
4. Keep `DS4_V100_ENABLE_OUTPUT_HEAD_FASTPATH=1` as the opt-in gate until V100
   A/B evidence justifies a default change, then switch to default-on with a
   disable fallback.
5. Preserve the full-logit host top-k path for `k > 1` and diagnostics.
6. Validate selected-token correctness (`3136`) on V100 for both default and
   opt-in paths.
7. Re-run the 1M context, 4-slot, per-step async sustained benchmark:
   - default host-logit path;
   - opt-in parallel device top-1 path.
8. Decide whether the opt-in path should become default. Decision: yes.

## Non-Goals

- Vocab-parallel output projection.
- Batched output-head projection across slots.
- Sampling or non-greedy top-k serving.
- Output-head dtype or quantization changes.
- Stream/event handoff changes.

## Implementation

1. Update `ds4_cuda.cu`:
   - add a block-level F32 top-1 kernel over fixed-size vocabulary chunks;
   - add a final reducer kernel over per-block candidates;
   - use shared-memory reduction and the same tie rule as the CPU path;
   - carry a non-finite status flag using raw float exponent-bit checks;
   - store only one token/logit pair in host-visible result copies.
2. Keep `ds4_gpu_top1_f32_tensor` API stable in `ds4_gpu.h`.
3. Reuse the existing scheduler path from Sprint 075:
   - `k == 1`;
   - default-on after the A/B decision;
   - `DS4_V100_DISABLE_OUTPUT_HEAD_FASTPATH=1` disables it;
   - `DS4_V100_ENABLE_OUTPUT_HEAD_FASTPATH=0` also disables it for
     compatibility with the Sprint 075 env gate.
4. Extend validation if needed:
   - bounded logits smoke checks device top-1 against GPU-readback top-1;
   - selected-token smoke checks default and opt-in serving paths;
   - sustained benchmark records output-head timing and aggregate tok/s.

## Definition of Done

- [x] Local compile passes for changed C/CUDA-facing objects where possible.
- [x] `git diff --check` passes.
- [x] V100 CUDA build passes for replay and relevant smokes.
- [x] `cuda_v100_bounded_logits_smoke` passes on V100.
- [x] `cuda_v100_selected_token_smoke` passes with the default host-logit path.
- [x] `cuda_v100_selected_token_smoke` passes with
  `DS4_V100_ENABLE_OUTPUT_HEAD_FASTPATH=1`.
- [x] Sustained V100 A/B at `ctx=1048576`, `slots=4`, `tokens=16`,
  `requests=4`, `async_pipeline_mode=per-step` records:
  generated tok/s, continuation tok/s, output-head ms, token matches, and GPU
  utilization.
- [x] Default-on selected-token smoke passes after the scheduler policy flip.
- [x] Disable-fallback selected-token smoke passes with
  `DS4_V100_DISABLE_OUTPUT_HEAD_FASTPATH=1`.
- [x] Sprint report records the default decision.
- [x] Vision document is updated.
- [x] Artifacts are committed.

## Outcome

The parallel reducer is correct and clears the default-change threshold.

At `ctx=1048576`, `slots=4`, `tokens=16`, `requests=4`, per-step async:

| Path | Generated tok/s | Continuation tok/s | Avg latency ms | Output-head ms | Avg GPU util |
|---|---:|---:|---:|---:|---:|
| host-logit control | `8.656498` | `8.115467` | `7391.521` | `324.953` | `19.075%` |
| parallel device top-1 | `9.031197` | `8.466747` | `7084.718` | `134.510` | `19.271%` |

Generated and continuation throughput improved by `+4.329%`. Output-head timing
dropped by `58.606%`. Because both metrics beat the Sprint 076 decision rule,
the parallel device top-1 path is now default for greedy `k == 1` selection.
`DS4_V100_DISABLE_OUTPUT_HEAD_FASTPATH=1` remains the rollback switch and was
validated by selected-token smoke.

## Decision Rule

- Make the parallel device top-1 path default only if it improves generated
  tok/s by at least `1%` or reduces output-head timing by at least `10%`
  without correctness regression.
- Keep it opt-in if the gain is smaller, noisy, or isolated to one metric.
- Disable or remove it if it regresses correctness.

## Risks

- Output projection may dominate the output-head path enough that top-1
  reduction/readback cleanup is still too small.
- A parallel reducer can mishandle exact ties if it does not preserve lower
  token ids.
- `--use_fast_math` and near-tie GPU/CPU accumulation differences mean tests
  should compare against GPU-readback logits where appropriate, not a separate
  CPU matmul oracle.

## Security

No new serving surface. The feature remains gated by an internal environment
variable unless the sprint evidence supports a default change.
