# Sprint 539 - C1 Route-Stable Graph Suffix Replay

Date: 2026-05-29

## Goal

Recover cross-position persistent suffix graph reuse without replaying stale
routed FFN or EP-compose launch geometry.

Sprint 538 restored selected-token serving parity by making suffix graphs
position-keyed again. That was the correct safety repair, but it removed the
cache reuse that C1 needs for performance. Sprint 539 makes the routed suffix
graph-stable before removing the position key.

## Starting Evidence

- Sprint 537 direct-token-major suffix replay reused cache entries across
  positions and passed direct parity, but HTTP selected-token serving diverged
  after the first generated token.
- Sprint 538 repaired selected-token serving parity by invalidating persistent
  suffix graphs by decode position.
- Sprint 538 logs show the served graph path still used variable per-rank
  route geometry despite `post_attention_fixed_capacity_route_plan_gate=1`.
  Example layer 0 at 8 slots:
  `routes 0,16,0,0,0,0,16,16`, `total_routes 48`.
- Root cause in code: the fixed-capacity post-attention route plan was tied to
  `reuse_model_router_route_plan`. The persistent graph dynamic prefix runs
  post-attention routing with reuse disabled, so it still captured variable
  route counts.

## Scope

1. Make fixed-capacity post-attention route planning apply to graph-event-order
   execution, not to eager execution.
2. Keep eager serving on compact actual-route geometry.
3. Re-enable cross-position persistent suffix cache reuse only when the graph
   suffix is using fixed-capacity post-attention route geometry.
4. Validate selected-token parity at the Sprint 538 failing shapes.
5. Confirm graph cache hits are present and position invalidations are gone.
6. Run warmed performance only after selected-token parity is strict.

## Non-goals

- No MTP work.
- No new permanent smoke harness.
- No broad host synchronizes.
- No graph default promotion if strict selected-token parity fails.
- No performance claim from startup-dominated elapsed time.

## Validation

Correctness gate:

- Remote V100 build passes.
- Eager and graph selected-token generated token sequences match exactly at
  8 requests / 8 slots / 4 tokens and 8 requests / 8 slots / 8 tokens.
- `decode_cudagraph_persistent_cache_hits > 0`.
- `decode_cudagraph_persistent_invalidate_position == 0`.
- `decode_cudagraph_replay_succeeded == decode_cudagraph_replay_attempted`.
- `peer_copy_ops=0`, `peer_copy_sys_bytes=0`.
- `nccl_graph_sys_edge_count=0`.

Performance gate:

- Isolate initialization/startup time.
- Use warmup before measuring steady state.
- Prefer long deterministic selected-token or fixed-prompt generation windows.
- Compare request-window or steady-state fields, not full-run elapsed time.

## Execution

Workspace:

- Local repo: `/Users/ravi/repos/ds4`
- Remote build workspace: `/workspace/s539-route-stable`
- Artifact root: `/workspace/s539-route-stable-artifacts`

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- Result: PASS

Implementation:

- `engine/post_attention_ffn.cu`
  - Fixed-capacity post-attention route planning now applies when graph-event
    ordering is active, instead of being tied to
    `reuse_model_router_route_plan`.
  - Eager serving remains on compact actual-route geometry.
- `engine/decode_loop.cu`
  - Persistent graph cache reuse drops the decode-position key only for the
    `compose_eager_final_hc` suffix when compact MOE compose is active and
    fixed-capacity post-attention route geometry is enabled.
  - Other suffixes remain position-keyed.

Validation:

- Graph `8` requests / `8` slots / `256K` / `4` tokens:
  - Artifact:
    `/workspace/s539-route-stable-artifacts/none-s539-graph8x4-compose-stable-p262080-serverargs-h2180dc1d`
  - Eager control:
    `/workspace/s538-c2-parity/none-s538-eager8x4`
  - Result: all `8` generated token sequences and decode-step checksums
    matched eager exactly.
  - First output-head token: `29361`
  - `graph_audit_persistent_cache_hits=43`
  - `graph_audit_persistent_cache_misses=0`
  - `graph_audit_persistent_invalidate_position=0`
  - `graph_audit_replay_succeeded=43`
  - `graph_audit_replay_attempted=43`
  - `peer_copy_ops=0`, `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`
  - Layer 0 post-attention route geometry now reports `total_routes=384`,
    confirming fixed-capacity graph geometry.

- Graph `8` requests / `8` slots / `256K` / `8` tokens:
  - Artifact:
    `/workspace/s539-route-stable-artifacts/none-s539-graph8x8-compose-stable-p262080-serverargs-h2180dc1d`
  - Eager control:
    `/workspace/s538-c2-repair-artifacts/none-s538-repair-eager8x8`
  - Result: all `8` generated token sequences and decode-step checksums
    matched eager exactly.
  - First output-head token: `42395`
  - `graph_audit_persistent_cache_hits=43`
  - `graph_audit_persistent_cache_misses=0`
  - `graph_audit_persistent_invalidate_position=0`
  - `graph_audit_replay_succeeded=43`
  - `graph_audit_replay_attempted=43`
  - `peer_copy_ops=0`, `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`
  - Layer 0 post-attention route geometry reports `total_routes=384`.

Performance note:

- These reduced selected-token runs are correctness/cache evidence, not a
  promotion-grade throughput claim.
- The `8x4` graph run was slower than the eager control
  (`client_generated_tok_s 5.151956817` vs `5.635190281`,
  `scaffold_ms_per_token 430.363234` vs `378.062715`).
- The `8x8` graph run was roughly flat/mixed
  (`client_generated_tok_s 6.306026715` vs `6.218910130`,
  `scaffold_ms_per_token 391.569853` vs `382.825652`).
- Do not promote graph serving as a performance default from this evidence.
  The next graph sprint should either reduce the fixed-capacity padding cost
  or run a warmed long-generation/request-window test after a more efficient
  graph-stable geometry is available.

Decision:

- C1 route-stable suffix replay is correctness-clean for the selected-token
  served path and cache reuse is restored.
- Graph serving remains opt-in. No default promotion yet.
