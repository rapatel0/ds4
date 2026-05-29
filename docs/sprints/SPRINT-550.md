# Sprint 550 - C1 Compact EP Pack Route-Block Kernel

Date: 2026-05-29

## Goal

Reduce fixed-capacity route padding overhead inside the promoted graph-stable
compact EP compose path without adding a new flag or changing route semantics.

## Implementation

Changed `ep_pack_route_dest_shards_kernel` from a flat
`routes * hidden` launch to a route-block launch:

- Grid shape is now one CUDA block per graph-visible route.
- Active routes still write the same packed values:
  `float(route_hidden[route, h]) * route_weight[route]`.
- Inactive padded routes return before looping across hidden dimensions.

The promoted fixed-capacity graph path still exposes a stable route envelope to
CUDA graph replay. This only reduces useless work inside that fixed envelope.

## Validation

Remote workspace:

- `/workspace/s550-ep-pack-route-block`

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- PASS

Short selected-token graph sanity:

- Artifact:
  `/workspace/s550-ep-pack-route-block-artifacts/none-s550-graph8x4-p262080/summary.json`
- Shape: `8` requests / `8` slots / `256K` context / `4` generated tokens
- Endpoint: `/v100/selected-token`
- Result:
  - `http_200=8`
  - `output_head_first_token=29361`, matching the prior default graph sanity
    record from Sprint 541
  - `graph_audit_blocker=none`
  - `graph_audit_persistent_cache_hits=43`
  - `graph_audit_persistent_invalidations=0`
  - `graph_audit_persistent_invalidate_position=0`
  - `graph_audit_replay_succeeded=43`
  - `graph_audit_replay_attempted=43`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`
  - `vram_failures=0`

This short run is correctness and transport evidence only. It is not a
steady-state performance gate because it does not use a long warm request
window.

## Decision

Promote the route-block compact EP pack kernel. It is a contained
fixed-envelope efficiency cleanup and keeps the graph-stable route shape
unchanged.

Continue C1 padding-efficiency work by looking for the next fixed-capacity
compact compose/executor site where inactive padded routes still do hidden-wide
work. If the remaining overhead is inside Turbomind grouped GEMM row bounds,
that likely requires a larger executor-level design rather than another kernel
launch rewrite.
