# Sprint 532 - C5 Post-Attention FFN Event Handoffs

Date: 2026-05-28

## Goal

Continue C5 sync-point reduction by removing promoted-path host stream waits
from `engine/post_attention_ffn.cu`.

## Context

- Sprint 528 removed output-head device-wide waits.
- Sprint 529 removed attention-output rank/dense handoff waits.
- Sprint 531 completed the compact EP broadcast transport cleanup.
- The promoted post-attention FFN path is rank-major and skips semantic stats,
  so several host waits are now pure ordering barriers rather than host-visible
  data requirements.

## Scope

1. Do not sync rank streams after `d_post_attn_shard` production when semantic
   stats are skipped; the following NCCL all-gather is enqueued on the same
   rank streams.
2. Do not sync rank streams after the post-attention rank-major all-gather; the
   following rank-major input-fill kernels are enqueued on the same rank
   streams.
3. Replace the final rank-stream synchronization after shared/route FFN input
   fill with `enqueue_dense_wait_after_rank_stream()` so dense streams wait on
   device events instead of the host.
4. Preserve diagnostic/stat paths that genuinely consume host-visible data.
5. Do not add a runtime flag, broad sync diagnostic, one-off smoke, or MTP work.

## Validation

Required local checks:

- `git diff --check`
- active-code search confirms no new C5 runtime flag or permanent smoke
  scaffold.

Required remote checks:

- Remote CUDA build: passed in
  `/workspace/s532-post-ffn-events`.
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- Selected-token gate: passed in
  `/workspace/s532-post-ffn-events-selected32` at `32` requests / `32`
  slots / `256K` / `2` tokens.

Gate requirements:

- `http_200=32`: passed
- server output-head first generated token remains `128819`: passed
- `output_head_finite_bad=0`: passed
- `peer_copy_ops=0`: passed
- `peer_copy_sys_bytes=0`: passed
- `nccl_graph_sys_edge_count=0`: passed
- Server logs still show `tp_ep_post_attention_ffn_input ... PASS` with
  `rank_major_input=1` and `slot_major_ffn_norm=0`.

Observed selected-token summary:

- `client_generated_tok_s=9.667920052827988`
- `scaffold_ms_per_token=1662.384368`
- `scaffold_sum_pre_ep_post_attention_ffn_input_ms=63.421656`
- `scaffold_sum_compose_ms=50.219847`
- `scaffold_compact_moe_decode_gate=1`

Server log confirmation:

- `tp_ep_post_attention_ffn_input` emitted PASS lines with
  `stats_skipped=1`, `rank_major_input=1`,
  `rank_major_shared_input=1`, `rank_major_route_input=1`, and
  `slot_major_ffn_norm=0`.
- `tp_ep_diagnostic_output_head` emitted PASS with `first_token=128819`,
  `device_sync_count=0`, `stream_sync_count=8`, and `event_sync_count=8`.

## Decision

Promoted.

The promoted post-attention FFN input path no longer uses host stream
synchronization for the three pure ordering boundaries covered by this sprint:

1. after `d_post_attn_shard` production when semantic stats are skipped;
2. after post-attention rank-major all-gather before rank-major fill kernels;
3. after shared/route FFN input fill before dense-stream consumers.

No runtime flag, permanent smoke, broad diagnostic branch, or MTP work was
added. Remaining host syncs in `engine/post_attention_ffn.cu` are tied to
control/diagnostic route-plan paths and stay in the C5 audit surface.

## Definition of Done

- Promoted post-attention FFN input path no longer uses host synchronization
  for pure rank-stream ordering.
- Dense stream handoff uses device events.
- Diagnostic/stat paths remain correct.
- No one-off smoke, runtime flag, or diagnostic branch remains.
- Local and remote builds pass.
- V100 selected-token gate passes or the sprint records the concrete blocker
  and leaves the promoted path unchanged.
