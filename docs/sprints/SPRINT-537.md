# Sprint 537 - C1 Piecewise Graph Capture Stage 1

Date: 2026-05-29

## Goal

Restore and validate the smallest reusable TP/EP CUDA graph suffix path after
the structural cleanup, without promoting graph serving before parity.

## Scope

1. Re-expose the existing graph diagnostic controls on the TP/EP appliance CLI
   so C1 can be exercised from the current binary.
2. Make promoted HC-current and router NCCL all-reduce paths graph-order
   capable instead of rejecting graph mode.
3. Reuse suffix-only persistent graphs across decode positions when the
   position-dependent prefix runs eagerly.
4. Validate direct-token-major correctness and cache reuse before any HTTP
   selected-token probe.

## Implementation

- `appliance/options.h`
  - Added narrowly scoped graph diagnostic parser support:
    `--decode-cudagraph-gate`,
    `--decode-cudagraph-replay-probe-gate`,
    `--decode-cudagraph-persistent-replay-gate`,
    `--decode-cudagraph-suffix-stage`,
    `--decode-cudagraph-stage-sync`,
    `--decode-cudagraph-output-sync-gate`,
    `--decode-cudagraph-hc-current-sync-gate`, and
    `--decode-stage-checksum-gate`.
  - These only wire existing `Options` fields. They do not change defaults.
- `engine/hc_current.cu`
  - Removed the graph-mode rejection from the promoted HC-current NCCL
    all-reduce path. NCCL all-reduce is graph-capturable and already ordered by
    rank streams.
- `engine/router_step.cu`
  - Removed the graph-mode rejection from promoted router all-reduce.
  - Replaced the graph-mode tail host stream waits with the existing
    `enqueue_control_wait_after_rank_streams()` device-event handoff before
    copying rank-major router logits to the control buffer.
- `engine/decode_loop.cu`
  - Kept full persistent graphs position-keyed.
  - Made suffix-only persistent graphs reusable across decode positions because
    HC-current, attention/KV, route planning, and other position-dependent work
    run in the eager prefix for the suffix split.
- `docs/sprints/VISION.md`
  - Added the measurement rule: short direct/HTTP graph probes are
    correctness/cache-behavior evidence only. Performance claims require
    startup/initialization isolated out, startup warmup where supported, enough
    warmed work to reach steady state, and request-window or steady-state
    metrics rather than full-run elapsed time or full-run GPU averages.

## Artifacts

- Remote workspace: `/workspace/s537-c1-stage1`
- Direct profiles:
  - Eager `8` slots / `2` tokens:
    `/workspace/s537-c1-stage1-direct/none-direct-s537-direct-eager8x2`
  - Graph `8` slots / `2` tokens after graph all-reduce repair:
    `/workspace/s537-c1-stage1-direct-r3/none-direct-s537-direct-graph8x2-compose-r3-serverargs-h2180dc1d`
  - Eager `8` slots / `4` tokens:
    `/workspace/s537-c1-stage1-direct-r4/none-direct-s537-direct-eager8x4`
  - Graph `8` slots / `4` tokens:
    `/workspace/s537-c1-stage1-direct-r4/none-direct-s537-direct-graph8x4-compose-serverargs-h2180dc1d`
- HTTP selected-token probes:
  - Eager `8` requests / `8` slots / `4` tokens:
    `/workspace/s537-c1-stage1-http/none-s537-http-eager8x4`
  - Graph `8` requests / `8` slots / `4` tokens:
    `/workspace/s537-c1-stage1-http/none-s537-http-graph8x4-compose-serverargs-h2180dc1d`

## Validation

Remote build passed:

```text
CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance
```

### Direct-token-major

Initial graph probe failed before capture:

```text
tp_hc_current_input_failed layer 0 rc 11
```

Root cause: suffix replay runs the dynamic prefix eagerly, but the prefix still
received `decode_cudagraph_gate=true`. Promoted HC-current all-reduce treated
graph mode as unsupported and returned `11`.

After enabling graph-order NCCL all-reduces:

| Run | Result |
|---|---|
| Eager `8x2` | pass, first token `123327`, `scaffold_ms_per_token=436.720318` |
| Graph `8x2` before suffix cache key fix | pass, first token `123327`, `86/86` replays, `86` misses, `43` position invalidations |
| Graph `8x2` after suffix cache key fix | pass, first token `123327`, `86/86` replays, `43` misses, `43` hits, `0` invalidations |
| Eager `8x4` | pass, first token `123327`, `scaffold_ms_per_token=448.760927` |
| Graph `8x4` | pass, first token `123327`, `172/172` replays, `43` misses, `129` hits, `0` invalidations, `scaffold_ms_per_token=440.622602` |

The direct graph suffix is now correctness-clean and reusable across decode
positions. The `8x4` improvement is small and not promotion-grade by itself,
but it proves the graph path is actually replaying steady decode work.

### HTTP selected-token

Reduced serving probes were run only as correctness/parity checks. They are not
performance evidence because startup dominates the elapsed time and there was
no long warmed request window.

Eager `8` requests / `8` slots / `4` tokens:

- `http_200=8`
- request window: `5.849614s`
- startup/readiness: `102.181344s`
- first token: `29361`
- `peer_copy_ops=0`
- `peer_copy_sys_bytes=0`
- `nccl_graph_sys_edge_count=0`
- `scaffold_ms_per_token=384.489463`

Graph `8` requests / `8` slots / `4` tokens:

- `http_200=8`
- request window: `6.406554s`
- startup/readiness: `101.175443s`
- first token: `61012`
- `peer_copy_ops=0`
- `peer_copy_sys_bytes=0`
- `nccl_graph_sys_edge_count=0`
- `graph_audit_persistent_cache_hits=43`
- `graph_audit_persistent_cache_misses=0`
- `graph_audit_replay_succeeded=43`
- `scaffold_ms_per_token=361.237680`

Serving graph replay is active and topology-clean, but selected-token parity is
still wrong. This blocks promotion and moves the next work to C2 parity repair.

## Decision

Do not promote graph serving defaults.

Sprint 537 succeeds as C1 Stage 1 direct graph enablement:

- The current appliance can exercise graph diagnostics again.
- Promoted HC-current/router all-reduce paths no longer reject graph ordering.
- Direct suffix graph replay is correctness-clean and cache-reusable across
  decode positions.
- Serving selected-token still diverges, so C2 remains the next ordered sprint.

Performance interpretation:

- Short graph probes may show higher/spikier utilization and the graph path now
  clearly opens performance headroom.
- No throughput claim should be made from these short probes. Promotion-grade
  graph performance needs startup isolated, startup warmup enabled, enough
  warmed requests/tokens for steady state, and request-window or steady-state
  metrics.
