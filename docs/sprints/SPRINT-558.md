# Sprint 558 - Fresh-State Full-Capture Replay Probe

Date: 2026-05-29

## Goal

Repair no-suffix full-capture replay validation so replay is tested from fresh
request state instead of by immediately replaying on buffers already advanced by
capture.

## Change

`engine/decode_loop.cu` now handles no-suffix replay-probe differently from the
promoted suffix replay path:

- If a matching full-capture graph already exists, replay it directly.
- If no matching graph exists, run the real eager decode step, then capture and
  instantiate the graph as an audit/cache artifact without launching it.
- Do not run the eager suffix-prefix helper for no-suffix full-capture replay.
- Keep the promoted suffix path behavior unchanged.

This gives the harness a real fresh-state full-capture replay check on the next
same-position request.

## Validation

Remote workspace:

- `/workspace/s558-fresh-state-replay`

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- PASS

Eager sequential control:

- Artifact:
  `/workspace/s558-fresh-state-replay-artifacts/none-s558-eager-seq2/summary.json`
- Shape: `2` sequential requests / `8` slots / `256K` context / `1` token
- Request concurrency: `1`
- Result:
  - `http_200=2`
  - response 0 selected token: `128819`
  - response 1 selected token: `128819`
  - response 0 checksum: `7238127778`
  - response 1 checksum: `5174931161`

No-suffix full-capture fresh-state replay:

- Artifact:
  `/workspace/s558-fresh-state-replay-artifacts/none-s558-fullgraph-fresh-replay-seq2c-serverargs-h396a9fa7/summary.json`
- Shape: `2` sequential requests / `8` slots / `256K` context / `1` token
- Request concurrency: `1`
- Result:
  - `http_200=2`
  - request 0 selected token: `128819`
  - request 1 selected token: `128819`
  - request 0 checksum: `7238127778`
  - request 1 checksum: `5002850195`
  - `graph_audit_capture_attempted=43`
  - `graph_audit_capture_succeeded=43`
  - `scaffold_decode_cudagraph_persistent_cache_hits=43`
  - `scaffold_decode_cudagraph_replay_attempted=43`
  - `scaffold_decode_cudagraph_replay_succeeded=43`
  - `scaffold_decode_cudagraph_persistent_invalidations=0`
  - `scaffold_decode_cudagraph_persistent_invalidate_position=0`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`

Promoted suffix sanity:

- Artifact:
  `/workspace/s558-fresh-state-replay-artifacts/none-s558-suffix-sanity/summary.json`
- Shape: `1` request / `8` slots / `256K` context / `1` token
- Result:
  - `http_200=1`
  - `scaffold_decode_cudagraph_capture_attempted=43`
  - `scaffold_decode_cudagraph_capture_succeeded=43`
  - `scaffold_decode_cudagraph_replay_attempted=43`
  - `scaffold_decode_cudagraph_replay_succeeded=43`
  - `scaffold_decode_cudagraph_persistent_invalidations=0`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`

## Decision

Promote as validation-harness repair, not as a full-capture serving default.

The fresh-state replay path now exists and preserves the deterministic selected
token in this reduced same-position probe. The second replay checksum still
differs from eager, so the next full-capture work should localize that checksum
drift before relaxing the full-capture position cache key or promoting full
capture beyond diagnostics.
