# Sprint 531 - B2 Compact EP Broadcast Trim

Date: 2026-05-28

## Goal

Keep compact EP compose on the topology-compatible NCCL broadcast path, but
remove padded compact over-transfer.

## Context

- Sprint 530 rejected grouped all-pairs NCCL send/recv because NCCL routed some
  pairs through SHM and failed the container `/dev/shm` budget.
- The existing served compact compose path already uses NCCL broadcast and
  passed prior selected-token gates under the promoted no-SYS topology policy.
- The helper still broadcasts a full `kGpus * compact_segment_elems` buffer for
  every source rank, even when the source rank has zero compact rows or far
  fewer than `slots * top_k` compact rows.

## Scope

1. Keep the existing `broadcast_ep_return_slices()` movement primitive.
2. Skip zero-row source ranks before issuing NCCL broadcast.
3. For compact sources with fewer active rows than the padded segment stride,
   pack each destination slice into contiguous scratch with `cudaMemcpy2DAsync`
   on the source rank stream, then broadcast only the packed active rows.
4. Preserve non-compact FP32 and FP16 behavior.
5. Do not add a runtime flag, one-off smoke, all-pairs P2P path, or MTP work.

## Validation

Required local checks:

- `git diff --check`
- active-code search confirms no new B2 runtime flag or permanent smoke
  scaffold.

Local result:

- `git diff --check`: PASS.
- Active-code search: PASS. No new B2 runtime flag, no all-pairs send/recv
  path, and no permanent smoke scaffold.

Required remote checks:

- Remote CUDA build:
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- Selected-token gate at `32` requests / `32` slots / `256K` / `2` tokens.

Remote result:

- Remote workspace:
  `/localpool/ds4/workspace/s531-compact-bcast-trim`
- Artifact:
  `/localpool/ds4/workspace/s531-compact-bcast-trim-selected32`
- Build:
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`: PASS.
- Selected-token gate:
  - `http_200=32`
  - server output-head first generated token: `128819`
  - `output_head_finite_bad=0`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`
  - `scaffold_compact_moe_decode_gate=1`
  - compact route logs present, e.g. layer 0:
    `routes 0,64,0,0,0,0,64,64`
  - `scaffold_sum_compose_ms=49.542611`

Note: the profile summary's `output_head_first_token=68338` corresponds to the
second selected-token step. The authoritative server output-head line for the
first generated token reports `first_token 128819`, matching the promoted
control shape.

Gate requirements:

- `http_200=32`
- output-head server first token remains compatible with the promoted control
  artifact for the same shape.
- `output_head_finite_bad=0`
- `peer_copy_ops=0`
- `peer_copy_sys_bytes=0`
- `nccl_graph_sys_edge_count=0`
- Server logs still show compact route compose enabled.

## Definition of Done

- Compact EP compose return movement remains NCCL broadcast based.
- Zero-route source ranks do not issue padded broadcasts.
- Active compact rows are packed before reduced-size broadcast so destination
  compact row indexing remains unchanged.
- No one-off smoke, runtime flag, or diagnostic branch remains.
- Local and remote builds pass.
- V100 selected-token gate passes or the sprint records the concrete blocker
  and leaves the promoted path unchanged.

## Decision

Promote.

The served compact EP compose path remains on NCCL broadcast, which is
compatible with the promoted no-SYS topology policy. Zero-route source ranks no
longer issue padded broadcasts. Active compact rows are packed into contiguous
scratch on the source rank before broadcast, so the broadcast byte count follows
the active compact row count while the destination compact row indexing remains
unchanged.

B2 transport cleanup is complete enough for the current graph-readiness docket.
The larger B2 fusion item remains open, but the next sprint should return to
C5 sync-point reduction rather than retrying compact EP all-pairs transport.
