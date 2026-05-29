# Sprint 535 - C5 HC-Current Fill Event Handoff

Date: 2026-05-29

## Goal

Continue C5 sync-point reduction by replacing the promoted HC-current final
fill/pack host wait with a device-event handoff.

## Context

- Sprint 532 promoted post-attention FFN event handoffs.
- Sprint 533 promoted attention-projection event handoffs.
- Sprint 534 promoted attention-read raw/window event handoffs.
- `run_shared_hc_current_input()` still host-synchronizes rank streams after
  filling dense inputs and route packs. The next dense consumers only need an
  ordering dependency from rank streams to dense streams.
- This sprint is main-path cleanup only. No runtime flag, broad diagnostic,
  permanent smoke, or MTP work belongs here.

## Scope

1. Replace the non-graph host synchronization after HC-current final fill/pack
   with `enqueue_dense_wait_after_rank_stream()`.
2. Preserve router readback/upload, parity, reference, and diagnostic
   synchronization paths.
3. Do not alter EP compose, decode-loop stage sync diagnostics, typed-indexer,
   or MTP code.

## Validation

Required local checks:

- `git diff --check`
- active-code search confirms no new C5 runtime flag or permanent smoke
  scaffold.

Required remote checks:

- Remote CUDA build:
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- Selected-token gate at `32` requests / `32` slots / `256K` / `2` tokens.

Gate requirements:

- `http_200=32`
- server output-head first generated token remains `128819`
- `output_head_finite_bad=0`
- `peer_copy_ops=0`
- `peer_copy_sys_bytes=0`
- `nccl_graph_sys_edge_count=0`
- Server logs still show normal HC-current/attention downstream PASS markers.

## Definition of Done

- Promoted HC-current final fill/pack path no longer host-synchronizes for the
  pure rank-stream-to-dense-stream ordering boundary.
- Diagnostics and host-visible router/control paths remain unchanged.
- No one-off smoke, runtime flag, broad diagnostic branch, or MTP work is
  added.
- Local and remote builds pass.
- V100 selected-token gate passes or the sprint records the concrete blocker
  and leaves the promoted path unchanged.

## Result

Promoted.

Implementation:

- Replaced the promoted HC-current final fill/pack eager host stream
  synchronization with the existing dense-stream device-event handoff:
  `enqueue_dense_wait_after_rank_stream()`.
- Left router readback/upload, parity/reference paths, diagnostics, EP compose,
  decode-loop stage sync, typed-indexer/top-k, and MTP untouched.
- Added no runtime flag, permanent smoke, or diagnostic branch.

Validation:

- `git diff --check`: pass.
- Active-code search: no new C5 runtime flag, permanent smoke, or MTP path.
- Remote V100 build: pass with
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`.
- Selected-token gate at `32` requests / `32` slots / `256K` / `2` tokens:
  `http_200=32`, `output_head_finite_bad=0`, `peer_copy_ops=0`,
  `peer_copy_sys_bytes=0`, `nccl_graph_sys_edge_count=0`,
  `scaffold_compact_moe_decode_gate=1`.
- Server log confirmed output-head first generated token `128819` on the first
  selected-token step. The JSON summary reports the later step token `68338`,
  matching the known multi-step summary nuance.
- Server logs showed HC-current and attention downstream PASS markers,
  including `tp_hc_current_input_gate=1`,
  `tp_hc_current_input_nccl_allgather=1`,
  `tp_hc_current_allreduce=1`, and
  `tp_ep_true_attention_projection_prefix ... rank_major_input=1 ... PASS`.
