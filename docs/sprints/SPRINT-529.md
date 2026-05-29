# Sprint 529 - C5 Sync-Point Reduction Pass 2

Date: 2026-05-28

## Goal

Continue SPIKE B C5 by replacing served-path host stream synchronizations in
the attention-output projection stage with existing CUDA event dependencies.

This pass focuses on `engine/attention_output.cu`, which has the same repeated
rank-stream / dense-stream handoff pattern that graph mode already expresses
with events.

## Context

- Sprint 528 removed output-head device-wide waits for projection timing and
  top-1 readback.
- `SPIKE_B_STEERING.md` still lists C5 as open for decode-loop and per-stage
  stream waits.
- `engine/output_head.cu` already provides reusable event helpers:
  `enqueue_dense_wait_after_rank_stream()` and
  `enqueue_rank_streams_wait_after_dense_streams()`.
- The attention-output stage performs two dense projections:
  1. rank stream fills `attn_output_a` input, dense stream runs projection A,
  2. rank stream all-gathers/fills `attn` input, dense stream runs projection B.
  The eager path currently host-synchronizes at each handoff.

## Scope

1. Replace the eager attention-output handoff after input fill with
   `enqueue_dense_wait_after_rank_stream()`.
2. Replace the eager handoff after projection A with
   `enqueue_rank_streams_wait_after_dense_streams()`.
3. Replace the eager handoff after `attn` input fill with
   `enqueue_dense_wait_after_rank_stream()`.
4. Replace the eager handoff after projection B with
   `enqueue_rank_streams_wait_after_dense_streams()`.
5. Preserve graph-mode behavior and output stats.

## Non-Goals

- Do not touch MTP.
- Do not start C1 graph capture.
- Do not change attention math, output projection layout, or NCCL all-gather
  behavior.
- Do not add a runtime flag or one-off smoke scaffold.
- Do not sweep every remaining sync site mechanically; later C5 passes can
  handle attention projection, post-attention FFN, HC-current, and decode-loop
  audit syncs with per-site review.

## Validation

Required local checks:

- `git diff --check`
- active-code search confirms no new C5 feature flag or permanent smoke gate.

Local result:

- `git diff --check`: PASS.
- Active-code search: PASS. No new C5 flag, no one-off smoke scaffold, and no
  new permanent runtime switch.

Required remote checks:

- Remote CUDA build:
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- Selected-token gate at `32` requests / `32` slots / `256K` / `2` tokens.

Remote result:

- Remote workspace:
  `/localpool/ds4/workspace/s529-attn-output-sync`
- Artifact:
  `/localpool/ds4/workspace/s529-attn-output-sync-selected32`
- Build:
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`: PASS.
- Selected-token gate:
  - `http_200=32`
  - `output_head_first_token=128819`
  - `output_head_finite_bad=0`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`
  - `scaffold_sum_pre_ep_attention_output_ms=66.511281`
- Server attention-output logs:
  - `86` `tp_ep_true_attention_output_projection` lines.
  - `0` non-PASS lines.
  - This covers `43` layers across the two generated tokens.

Gate requirements:

- `http_200=32`
- output-head server first token remains compatible with the promoted control
  artifact for the same shape.
- `output_head_finite_bad=0`
- `peer_copy_ops=0`
- `peer_copy_sys_bytes=0`
- `nccl_graph_sys_edge_count=0`
- Server logs still show `tp_ep_true_attention_output ... PASS`.

## Definition of Done

- Attention-output eager stream handoffs use CUDA events instead of host stream
  synchronization.
- Graph-mode behavior is unchanged.
- No new flag or smoke scaffold is left behind.
- Local and remote builds pass.
- V100 selected-token gate passes or the sprint records a concrete blocker and
  leaves the promoted path unchanged.

## Decision

Promote.

The served attention-output eager handoffs now use the same CUDA event
dependencies previously used only under graph event ordering. This removes the
four host stream-synchronization branches from the main path without adding a
flag or retaining a diagnostic branch. The change is correctness-only for this
sprint and is primarily C1 readiness work.
