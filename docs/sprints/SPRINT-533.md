# Sprint 533 - C5 Attention-Projection Event Handoffs

Date: 2026-05-28

## Goal

Continue C5 sync-point reduction by removing promoted-path host waits from
`engine/attention_projection.cu`.

## Context

- Sprint 529 promoted attention-output event handoffs.
- Sprint 532 promoted post-attention FFN event handoffs.
- The next contained remaining C5 surface is attention projection: the graph
  path already uses stream/event helpers for the same dependencies that the
  eager promoted path still handles with host waits.
- This sprint is main-path cleanup only. No runtime flag, broad diagnostic,
  permanent smoke, or MTP work belongs here.

## Scope

1. Replace the non-graph host wait after attention-norm control work with
   rank-stream waits on the existing control event.
2. Replace rank-stream host waits after Q/KV input fills with dense-stream
   event waits.
3. Replace dense-stream host waits after Q/KV dense projections with a
   control-stream event wait before gather/norm control work.
4. Remove the unnecessary host wait between same-control-stream gather and
   Q/KV norm work.
5. Replace rank-stream host waits after Q/KV norm fill with dense-stream event
   waits.
6. Replace the final Q-B dense host wait with rank-stream event waits for the
   next stage.
7. Preserve diagnostic/stat paths that genuinely collect host-visible data.

## Validation

Required local checks:

- `git diff --check`
- active-code search confirms no new C5 runtime flag or permanent smoke
  scaffold.

Required remote checks:

- Remote CUDA build: passed in
  `/workspace/s533-attn-proj-events`.
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- Selected-token gate: passed in
  `/workspace/s533-attn-proj-events-selected32` at `32` requests / `32`
  slots / `256K` / `2` tokens.

Gate requirements:

- `http_200=32`: passed
- server output-head first generated token remains `128819`: passed
- `output_head_finite_bad=0`: passed
- `peer_copy_ops=0`: passed
- `peer_copy_sys_bytes=0`: passed
- `nccl_graph_sys_edge_count=0`: passed
- Server logs still show `tp_ep_true_attention_projection_prefix ... PASS`
  with `rank_major_input=1`.

Observed selected-token summary:

- `client_generated_tok_s=9.723544278120388`
- `scaffold_ms_per_token=1622.027197`
- `scaffold_sum_pre_ep_attention_projection_ms=52.677469`
- `scaffold_attention_projection_rank_major_input_gate=1`
- `scaffold_compact_moe_decode_gate=1`

Server log confirmation:

- `tp_ep_true_attention_projection_prefix` emitted PASS lines with
  `rank_local_input=1`, `rank_major_input=1`, and `current_source=rank0`.
- `tp_ep_diagnostic_output_head` emitted PASS with `first_token=128819`,
  `device_sync_count=0`, `stream_sync_count=8`, and `event_sync_count=8`.

## Decision

Promoted.

The promoted attention-projection path now uses device-event ordering for the
pure stream dependencies covered by this sprint:

1. attention-norm control work to rank-stream consumers;
2. Q/KV input fills to dense-stream consumers;
3. Q/KV dense projections back to control-stream gather/norm work;
4. Q/KV norm fill to Q-B dense consumers;
5. Q-B dense output back to rank-stream consumers.

The unnecessary host wait between same-control-stream gather and Q/KV norm
work was removed. The remaining `cudaDeviceSynchronize()` in
`engine/attention_projection.cu` is inside the explicit
`true_ds4_attention_kv_norm_reference_gate` diagnostic path.

## Definition of Done

- Promoted attention-projection path uses device events for pure stream
  ordering boundaries.
- Host waits remain only for diagnostics or genuine host-visible data.
- No one-off smoke, runtime flag, broad diagnostic branch, or MTP work is
  added.
- Local and remote builds pass.
- V100 selected-token gate passes or the sprint records the concrete blocker
  and leaves the promoted path unchanged.
