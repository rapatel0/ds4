# Sprint 555 - Rejected Served Full-Capture Position-Key Removal

Date: 2026-05-29

## Goal

Test whether full-capture persistent CUDA graph replay can drop the decode
position from its cache key for the served non-compressed path.

The hypothesis was that compressed-KV emitted topology is the remaining
position-keyed blocker, and the served appliance has compressed KV off.

## Candidate

Temporarily changed `engine/decode_loop.cu` so no-suffix full capture would not
invalidate a persistent graph across decode positions when
`true_ds4_compressed_kv_gate` was off.

Two follow-up fixes were tested during the bug loop:

- skip the dynamic suffix-prefix replay helper when no suffix stage is selected
- add a device-event barrier after updating `d_decode_position` before
  launching the full graph

Both fixes were removed after validation failed.

## Validation

Remote workspace:

- `/workspace/s555-full-capture-position-key`

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- PASS

Same-shape eager control:

- Artifact:
  `/workspace/s555-full-capture-position-key-artifacts/none-s555-eager8x4-p262080/summary.json`
- Shape: `8` requests / `8` slots / `256K` context / `4` generated tokens
- Endpoint: `/v100/selected-token`
- Result:
  - `http_200=8`
  - `output_head_first_token=29361`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`
  - `vram_failures=0`

Full-capture candidate without suffix stage:

- Artifact:
  `/workspace/s555-full-capture-position-key-artifacts/none-s555-fullgraph8x4-p262080-serverargs-h396a9fa7/summary.json`
- Structural result:
  - `graph_audit_blocker=none`
  - `graph_audit_persistent_cache_hits=43`
  - `graph_audit_persistent_invalidations=0`
  - `graph_audit_persistent_invalidate_position=0`
  - `graph_audit_replay_succeeded=43`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`
- Correctness failure:
  - `output_head_first_token=128819`
  - eager control first token was `29361`

Prefix-helper fix candidate:

- Artifact:
  `/workspace/s555-full-capture-position-key-artifacts/none-s555-fullgraph8x4-p262080-prefixfix-serverargs-h396a9fa7/summary.json`
- Structural result stayed clean with `43/43` replays and zero position
  invalidations.
- Correctness still failed:
  - `output_head_first_token=118235`

Position-copy barrier candidate:

- Artifact:
  `/workspace/s555-full-capture-position-key-artifacts/none-s555-fullgraph8x4-p262080-positionbarrier-serverargs-h396a9fa7/summary.json`
- Structural result stayed clean with `43/43` replays and zero position
  invalidations.
- Correctness still failed:
  - `output_head_first_token=118235`

## Decision

Reject the served full-capture position-key removal. Candidate code was removed.

The promoted `compose_eager_final_hc` graph suffix replay remains the correct
serving graph path. Full capture is still not safe to reuse across decode
positions, even with compressed KV off.

Next C1 work should localize the full-capture first divergence with stage
checksums instead of trying broader cache-key changes. The cache-key guard is a
correctness boundary, not just a stale leftover.
