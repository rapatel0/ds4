# Sprint 559 - Full-Capture Replay Checksum Drift Localization

Date: 2026-05-29

## Goal

Localize the remaining no-suffix full-capture checksum drift found in Sprint
558 before relaxing the full-capture position cache key or promoting no-suffix
full capture beyond diagnostics.

## Context

Sprint 558 repaired the validation harness: on a no-suffix full-capture cache
miss, the served request runs eager and captures the graph only as audit/cache
state; a later same-position request replays the cached graph from fresh request
state. The reduced sequential probe matched eager selected tokens on both
requests, but the second replay checksum drifted:

- eager request 1 checksum: `5174931161`
- full-capture replay request 1 checksum: `5002850195`

This sprint is diagnostic-first. Do not add permanent flags, do not promote
full capture, and do not change the ordered plan until the first divergent
stage is understood.

## Plan

1. Rebuild the current appliance in a fresh remote workspace copied with
   `rsync`.
2. Run the exact reduced sequential shape from Sprint 558 with
   `--decode-stage-checksum-gate`:
   - eager control: `2` sequential requests / `8` slots / `256K` context /
     `1` token
   - full-capture probe: same shape plus no-suffix graph gates and persistent
     replay
3. Parse server logs for `tp_ep_decode_stage_checksum` records.
4. Compare request 1, layer by layer and stage/tensor/rank by stage/tensor/rank.
5. If the first drift points to a small mechanical bug, patch it and rerun the
   reduced probe plus promoted suffix sanity.
6. If the first drift points to larger emitted-topology or row-bookkeeping work,
   document the blocker and keep full capture diagnostic-only.

## Result

The first real drift was final-HC host state after full-capture persistent
replay.

Stage checksum probes:

- Eager artifact:
  `/workspace/s559-checksum-drift-artifacts/none-s559-eager-stage-checksum-seq2-serverargs-h91108f54/summary.json`
- Full-capture artifact:
  `/workspace/s559-checksum-drift-artifacts/none-s559-fullgraph-stage-checksum-seq2-serverargs-hf0408634/summary.json`

The reduced replay matched eager through `compose.next_hidden`, but
`final_hc_shard` was stale after replay. Root cause: full-capture replay
captures the final-HC kernels, but CUDA graphs cannot capture the host-side
`d_final_hc_shard` / `d_hc_scratch_shard` pointer swap performed by
`run_shared_hc_final_expand()`. The cache-hit replay path launched the graph
but did not mirror that host metadata update, so the new final-HC state lived
in the physical scratch buffer while host code still read the old final-HC
pointer.

`engine/decode_loop.cu` now mirrors that swap after a successful no-suffix
persistent full-capture replay, guarded by the same conditions that make eager
perform the swap:

- `final_hc_carry_gate`
- `tp_hc_final_expand_gate`
- no suffix stage

Claude bug-find review was run on the patch and found the missing
`tp_hc_final_expand_gate` guard before final validation. The guard is included
in the committed fix.

## Validation

Remote workspace:

- `/workspace/s559-checksum-drift`

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- PASS

Post-fix eager sequential control:

- Artifact:
  `/workspace/s559-checksum-drift-artifacts/none-s559-eager-seq2-postfix/summary.json`
- Shape: `2` sequential requests / `8` slots / `256K` context / `1` token
- Result:
  - response 0 selected token: `128819`
  - response 1 selected token: `128819`
  - response 0 checksum: `7238127778`
  - response 1 checksum: `5174931161`

Post-fix no-suffix full-capture fresh-state replay:

- Artifact:
  `/workspace/s559-checksum-drift-artifacts/none-s559-fullgraph-seq2-final-serverargs-h396a9fa7/summary.json`
- Same shape as eager.
- Result:
  - response 0 selected token: `128819`
  - response 1 selected token: `128819`
  - response 0 checksum: `7238127778`
  - response 1 checksum: `5174931161`
  - `graph_audit_blocker=none`
  - `graph_audit_capture_attempted=43`
  - `graph_audit_capture_succeeded=43`
  - `scaffold_decode_cudagraph_persistent_cache_hits=43`
  - `scaffold_decode_cudagraph_replay_attempted=43`
  - `scaffold_decode_cudagraph_replay_succeeded=43`
  - `scaffold_decode_cudagraph_persistent_invalidations=0`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`

Promoted suffix sanity:

- Artifact:
  `/workspace/s559-checksum-drift-artifacts/none-s559-suffix-sanity3-serverargs-h396a9fa7/summary.json`
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

Promote the host-swap mirror as a full-capture validation/correctness repair.
This is not a no-suffix full-capture serving-default promotion. The next C1
work is emitted topology and row-position metadata, because same-position
fresh-state full-capture replay is now parity-clean at the reduced diagnostic
shape while cross-position reuse remains intentionally guarded by the position
cache key.

## Definition of Done

- Remote build passes.
- Eager and full-capture probes complete with deterministic selected-token
  output.
- The first divergent stage is identified, or the existing checksum logging is
  shown insufficient with a concrete next instrumentation point.
- `SPIKE_B_STEERING.md` and `docs/sprints/VISION.md` are updated with the
  decision.
- No new permanent gate or smoke is left behind unless it has a documented
  debugger-only sunset.
