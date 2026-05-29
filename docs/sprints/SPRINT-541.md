# Sprint 541 - C1 Graph Audit Blocker Classification

Date: 2026-05-29

## Goal

Make the graph audit report the current promoted graph-order surface instead
of carrying stale helper-host-sync blockers from before the C5 handoff sprints.

## Starting Evidence

Sprint 540 promoted TP/EP graph suffix replay after a warmed selected-token
gate, but the summary still reported:

- `graph_audit_blocker=helper_host_synchronization`
- `graph_audit_helper_host_sync_blocker_classes=2`

The two counted classes are attention-output and post-attention FFN input.
Those were already converted to graph-order event handoffs in Sprints 529 and
532, and the Sprint 540 run showed:

- `graph_audit_stream_sync_count=0`
- `graph_audit_rank_stream_sync_count=0`
- `graph_audit_dense_stream_sync_count=0`
- `graph_audit_copy_stream_sync_count=0`
- `graph_audit_replay_succeeded=43/43`

So this is stale audit classification, not a serving correctness blocker.

## Scope

1. Update the helper-host-sync blocker accounting in `engine/token_major_loop.cu`
   so attention-output and post-attention FFN are no longer counted as blockers
   when graph ordering is active.
2. Keep graph suffix replay behavior unchanged.
3. Validate that the promoted launcher default still builds, serves, replays,
   and reports `graph_audit_blocker=none`.

## Non-goals

- No graph math changes.
- No MTP work.
- No new runtime flags.
- No default change beyond the already-promoted Sprint 540 launcher default.

## Execution

Workspace:

- Remote build workspace: `/workspace/s541-graph-audit`
- Artifact root: `/workspace/s541-graph-audit-artifacts`

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- Result: PASS

Implementation:

- `engine/token_major_loop.cu`
  - Attention-output and post-attention FFN input no longer count as
    helper-host-sync blocker classes when `decode_cudagraph_gate` is active.
  - No graph execution behavior changed.

Validation:

- Default launcher selected-token run, no explicit graph server args:
  `/workspace/s541-graph-audit-artifacts/none-s541-default-graph8x4-p262080`
- Eager control:
  `/workspace/s538-c2-parity/none-s538-eager8x4`
- Result:
  - `http_200=8`
  - first output-head token `29361`
  - all `8` generated token sequences and decode-step checksums matched eager.
  - `graph_audit_blocker=none`
  - `graph_audit_helper_host_sync_blocker_classes=0`
  - `graph_audit_capture_eligible=1`
  - `graph_audit_persistent_cache_hits=43`
  - `graph_audit_persistent_invalidate_position=0`
  - `graph_audit_replay_succeeded=43`
  - `graph_audit_replay_attempted=43`
  - `peer_copy_ops=0`, `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`

Decision:

- Promote the audit classification cleanup.
- Remaining C1 work should now focus on real capture-surface/padding work, not
  the stale helper-host-sync blocker label.
