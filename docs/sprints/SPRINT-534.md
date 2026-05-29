# Sprint 534 - C5 Attention-Read Event Handoffs

Date: 2026-05-28

## Goal

Continue C5 sync-point reduction by removing promoted-path host waits from
`engine/attention_read.cu`.

## Context

- Sprint 529 promoted attention-output event handoffs.
- Sprint 532 promoted post-attention FFN event handoffs.
- Sprint 533 promoted attention-projection event handoffs.
- The promoted attention-read path writes `d_attn_heads` on each rank stream.
  The following attention-output stage reads `d_attn_heads` from the same rank
  stream before handing off to dense streams, so the post-read host waits are
  pure ordering barriers.
- This sprint is main-path cleanup only. No runtime flag, broad diagnostic,
  permanent smoke, or MTP work belongs here.

## Scope

1. Remove the non-graph host synchronization after `attention_raw_swa_one_row`
   in `run_true_ds4_attention_raw_read()`.
2. Remove the non-graph host synchronization after the raw-window attention
   kernels in `run_true_ds4_attention_raw_window()`.
3. Preserve diagnostic/stat paths that genuinely consume host-visible data.
   `log_tensor_f32_stats()` still synchronizes the relevant stream when it
   reads tensors for early-layer diagnostics.
4. Do not alter typed-history, indexer-topk, HC-current, decode-loop, or EP
   compose synchronization in this sprint.

## Validation

Required local checks:

- `git diff --check`
- active-code search confirms no new C5 runtime flag or permanent smoke
  scaffold.

Required remote checks:

- Remote CUDA build: passed in
  `/workspace/s534-attn-read-events`.
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- Selected-token gate: passed in
  `/workspace/s534-attn-read-events-selected32` at `32` requests / `32`
  slots / `256K` / `2` tokens.

Gate requirements:

- `http_200=32`: passed
- server output-head first generated token remains `128819`: passed
- `output_head_finite_bad=0`: passed
- `peer_copy_ops=0`: passed
- `peer_copy_sys_bytes=0`: passed
- `nccl_graph_sys_edge_count=0`: passed
- Server logs still show `tp_ep_true_attention_raw_window ... PASS`: passed

Observed selected-token summary:

- `client_generated_tok_s=10.620287093644198`
- `scaffold_ms_per_token=1594.528986`
- `scaffold_sum_pre_ep_raw_read_ms=46.676925`
- `scaffold_compact_moe_decode_gate=1`

Server log confirmation:

- `tp_ep_true_attention_raw_window` emitted PASS lines for the promoted path
  with `valid_rows=1`, `visible_compressed_rows=0`, and
  `selected_compressed_rows=0`.
- `tp_ep_diagnostic_output_head` emitted PASS with `first_token=128819`,
  `device_sync_count=0`, `stream_sync_count=8`, and `event_sync_count=8`.

## Decision

Promoted.

The promoted attention-read raw/window path no longer uses host
synchronization after producing `d_attn_heads` on rank streams. The following
attention-output stage consumes those heads on the same rank streams and then
uses device-event handoffs to dense streams. Diagnostic stat reads still call
`log_tensor_f32_stats()`, which synchronizes the stream when host-visible data
is actually consumed.

## Definition of Done

- Promoted attention-read raw/window path no longer host-synchronizes for pure
  rank-stream ordering.
- Diagnostics remain correct and host-visible stat reads still synchronize.
- No one-off smoke, runtime flag, broad diagnostic branch, or MTP work is
  added.
- Local and remote builds pass.
- V100 selected-token gate passes or the sprint records the concrete blocker
  and leaves the promoted path unchanged.
