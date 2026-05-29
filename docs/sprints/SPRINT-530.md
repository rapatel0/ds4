# Sprint 530 - B2 Compact EP Compose NCCL Send/Recv

Date: 2026-05-28

## Goal

Replace the served compact-route EP compose return movement with NCCL
point-to-point messages while preserving the existing compact pack and compact
compose math.

## Context

- Sprint 480's `ncclReduceScatter` evidence applies only to non-compact FP32
  compose and does not prove the served compact-route default.
- The served path in `engine/decode_loop.cu` uses compact route rows and the
  `compose_next_hidden_compact8_multi_kernel` consumer.
- The compact path has variable route counts by source rank. Each source sends
  the same compact row count to every destination, with destination-specific
  data stored at `dst * compact_segment_elems`.
- This sprint should not introduce a long-lived experiment flag. If the path
  promotes, it becomes the main compact-route movement path. If it fails, the
  change is reverted or documented as a blocker.

## Scope

1. Add a reusable NCCL send/recv helper for EP return slices.
2. Use that helper for served compact-route FP32 compose movement in
   `engine/decode_loop.cu`.
3. Leave non-compact FP32 and FP16 compose on their existing paths.
4. Preserve existing compact pack and compact compose kernels.
5. Do not touch MTP.

## Validation

Required local checks:

- `git diff --check`
- active-code search confirms no new B2/C5 feature flag or permanent smoke
  scaffold.

Local result:

- `git diff --check`: PASS.
- Active-code search: PASS. The attempted candidate added no runtime flag or
  permanent smoke scaffold.

Required remote checks:

- Remote CUDA build:
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- Selected-token gate at `32` requests / `32` slots / `256K` / `2` tokens.

Remote result:

- Remote workspace:
  `/localpool/ds4/workspace/s530-compact-sendrecv`
- Artifact:
  `/localpool/ds4/workspace/s530-compact-sendrecv-selected32`
- Build:
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`: PASS.
- Selected-token gate: FAIL.
  - Profiler error: `Remote end closed connection without response`.
  - Failure summary:
    `/localpool/ds4/workspace/s530-compact-sendrecv-selected32/nvprof-gpu-trace-s530-compact-sendrecv-selected32/failure-summary.md`
  - Server stderr ended with:
    `nccl error ./engine/runtime_pack.cu:381: unhandled system error`.
  - NCCL stdout showed point-to-point channels such as
    `7[7] -> 0[0] via SHM/direct/direct`.
  - NCCL then failed creating shared-memory transport segments:
    `failed to extend /dev/shm/nccl-* to 9637892 bytes`.

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

- Served compact-route compose return movement uses NCCL grouped send/recv.
- No one-off smoke, runtime flag, or diagnostic branch remains in the promoted
  path.
- Local and remote builds pass.
- V100 selected-token gate passes or the sprint records the concrete blocker
  and leaves the promoted path unchanged.

## Decision

Reject grouped all-pairs NCCL send/recv for served compact compose.

The approach is not compatible with the current no-SYS/no-SHM topology policy:
NCCL point-to-point all-to-all routes at least some cross-rank pairs through
SHM and fails in the container's `/dev/shm` budget. The failed candidate code
was removed, leaving the promoted compact compose path unchanged.

B2 remains open only for a topology-compatible alternative, such as a ring
collective/bucketed scheme that preserves the no-SYS NCCL policy. Do not retry
all-pairs `ncclSend`/`ncclRecv` as a promotion path without first proving it
can stay on NVL/P2P and pass the shared-memory constraint.
