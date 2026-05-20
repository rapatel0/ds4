# Sprint 077: Batched Output-Head Selection

## Status

Complete. The batched output-head primitive is correct and opt-in, but it is
disabled by default because paired V100 throughput evidence showed a regression
against the Sprint 076 per-slot device top-1 path.

## Overview

Sprint 076 made greedy output selection faster by default, but replay still
selects each active slot one at a time after every batched decode step. In the
1M/4-slot per-step profile, output-head timing is down to `134.510 ms`, but
that time is still paid through four serial output-head calls per token step.

Sprint 077 should batch greedy `k == 1` output-head selection across active
slots on gpu7. The goal is to run HC collapse, output norm, BF16 vocabulary
projection, and top-1 over all active slots in one output-head pass, while
preserving the full-logit host path for diagnostics and rollback.

## Goals

1. Add a batched BF16 arena matmul for output projection:
   `n_slots x hidden -> n_slots x vocab`.
2. Add a batched F32 top-1 reducer for row-major logits:
   `n_slots x vocab -> n_slots token/logit`.
3. Add scheduler-owned output-head batch scratch for gpu7.
4. Add `ds4_v100_stage_scheduler_select_token_batch` for greedy `k == 1`
   serving.
5. Update `ds4_v100_replay_generate_batch` to select all active slots together
   after prompt replay and after each continuation decode step.
6. Preserve current single-slot and host-logit fallback behavior:
   - `DS4_V100_DISABLE_OUTPUT_HEAD_FASTPATH=1`;
   - `k > 1` diagnostics.
7. Validate V100 selected-token correctness and sustained 1M/4-slot throughput.

## Non-Goals

- Sampling or non-greedy top-k batching.
- Vocab-parallel output projection.
- Changing output weight dtype.
- Changing stage async scheduling or handoff.
- Changing MTP draft logits/top-k internals.

## Implementation

1. Extend `ds4_gpu.h` / `ds4_cuda.cu`:
   - `ds4_gpu_arena_bf16_matmul_f32_rows`;
   - `ds4_gpu_top1_f32_rows_tensor`;
   - reuse Sprint 076 tie handling and non-finite status semantics.
2. Extend `ds4_v100_scheduler.c`:
   - add output-head batch scratch tensors;
   - copy slot HC tensors into a contiguous batch scratch;
   - run row-wise HC RMS norm, batched HC-head matmul, output HC weights,
     weighted sum, output norm, batched BF16 vocab projection, and batched
     top-1;
   - fall back to existing per-slot selection when disabled or `n_slots == 1`.
3. Extend replay:
   - add `replay_select_token_batch`;
   - replace serial slot-selection loops in `ds4_v100_replay_generate_batch`
     with one batch call where `n_prompts > 1`;
   - preserve per-token text decoding and counters.
4. Benchmark:
   - baseline default from Sprint 076 behavior with batch selection disabled;
   - candidate batched output-head path with `DS4_V100_ENABLE_OUTPUT_HEAD_BATCH=1`;
   - same 1M context, 4 slots, 16 tokens/request, 4 measured requests,
     per-step async fixture.

## Definition of Done

- [x] Local compile passes for changed C/CUDA-facing objects where possible.
- [x] `git diff --check` passes.
- [x] V100 CUDA build passes for replay and relevant smokes.
- [x] `cuda_v100_bounded_logits_smoke` passes on V100.
- [x] V100 selected-token smoke passes for default and output-head fallback.
- [x] Sustained V100 A/B at `ctx=1048576`, `slots=4`, `tokens=16`,
  `requests=4`, `async_pipeline_mode=per-step` records generated tok/s,
  continuation tok/s, output-head ms, token matches, and GPU utilization.
- [x] Sprint report records whether batched output-head remains default.
- [x] Vision document is updated.
- [x] Artifacts are committed.

## Results

| Path | Generated tok/s | Continuation tok/s | Avg latency ms | Output-head ms | Avg GPU util | Token match |
|---|---:|---:|---:|---:|---:|---:|
| Batched output-head opt-in | `8.616841` | `8.078288` | `7425.525` | `139.750` | `18.269%` | `4/4` |
| Per-slot device top-1 control | `9.028544` | `8.464260` | `7086.929` | `135.080` | `19.855%` | `4/4` |
| Post-patch default | `9.011829` | `8.448590` | `7099.939` | `135.402` | `19.823%` | `4/4` |

The batched path is correct, but it regressed generated/continuation tok/s by
`4.56%`, increased output-head timing by `3.46%`, and increased average latency
by `4.78%` relative to the paired per-slot control. It therefore remains
available only with `DS4_V100_ENABLE_OUTPUT_HEAD_BATCH=1`; default serving stays
on the Sprint 076 per-slot parallel device top-1 selector.

## Decision Rule

- Keep batched output-head selection default if it improves generated tok/s by
  at least `1%` or reduces output-head timing by at least `15%` without
  correctness regression.
- Keep the implementation opt-in or disabled if it is correct but neutral.
- Disable it if selected-token correctness regresses.

## Risks

- Batched `hc_head_fn` may route through cuBLAS and change tiny numerical
  details; selected-token smoke and same-fixture token checks are the gate.
- Output-head projection may already be small enough after Sprint 076 that
  batching does not move aggregate tok/s materially.
- Copying per-slot HC tensors into contiguous scratch could erase the benefit if
  done synchronously or with excessive allocation churn.

## Security

No new external serving surface. The fallback remains an internal environment
switch.
