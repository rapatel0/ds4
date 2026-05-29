# Sprint 548 - C1 Post-KV Graph Boundary Probe

Date: 2026-05-29

## Goal

Evaluate the larger post-KV graph suffix boundary requested by Sprint 547:
run the dynamic prefix through `raw_read`, capture/replay
`attention_output -> post_attention_ffn_input -> routed_ffn -> dense -> compose`,
and keep final-HC carry eager as in the promoted `compose_eager_final_hc`
suffix.

## Implementation

Added the default-off diagnostic suffix stage
`post_kv_compose_eager_final_hc` behind the existing
`--decode-cudagraph-suffix-stage` option. No new runtime flag was added.

The decode loop now has a small stage map so prefix-only and suffix-only graph
execution can split at either the promoted routed-FFN boundary or the new
post-KV boundary.

This sprint also fixed a Sprint 546 lifecycle bug: shared appliance/token-major
rank buffers did not allocate `RankState::d_decode_position`, so graph probes
could return before capture when using shared buffers. `open_shared_rank_buffers`
now allocates and initializes the device position scalar, tracks its bytes, and
`close_shared_rank_buffers` frees it.

## Validation

Remote workspace:

- `/workspace/s548-post-kv-boundary`

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- PASS

Initial invalid probes:

- Runs without `--repo-dir /workspace/s548-post-kv-boundary` used the profiler's
  stale default repo directory and are ignored.
- After using the correct repo directory, both the promoted
  `compose_eager_final_hc` control and the new post-KV boundary failed before
  capture because shared `d_decode_position` was missing. That was the lifecycle
  bug fixed in this sprint.

Reduced direct-token-major checks after the fix:

| Run | Suffix | Tokens | Result | Captures | Cache hits | Replays | Position invalidations | Notes |
|---|---|---:|---|---:|---:|---:|---:|---|
| `/workspace/s548-control-compose-fixed-artifacts-r1/none-direct-s548-control-compose-fixed-direct1-p262080-r1-serverargs-h2180dc1d/summary.json` | `compose_eager_final_hc` | 1 | PASS | 43 | 0 | 43/43 | 0 | promoted control sanity check |
| `/workspace/s548-postkv-fixed-artifacts-r1/none-direct-s548-postkv-fixed-direct1-p262080-r1-serverargs-h3e6f3cbf/summary.json` | `post_kv_compose_eager_final_hc` | 1 | PASS | 43 | 0 | 43/43 | 0 | boundary is capturable |
| `/workspace/s548-postkv-fixed-artifacts-r2/none-direct-s548-postkv-fixed-direct4-p262080-r2-serverargs-h3e6f3cbf/summary.json` | `post_kv_compose_eager_final_hc` | 4 | PASS | 43 | 129 | 172/172 | 0 | cross-position cache reuse works in the direct scaffold |
| `/workspace/s548-compose-fixed-artifacts-r2/none-direct-s548-compose-fixed-direct4-p262080-r2-serverargs-h2180dc1d/summary.json` | `compose_eager_final_hc` | 4 | PASS | 43 | 129 | 172/172 | 0 | current control comparison |

Transport/graph invariants held in the post-KV multi-token probe:

- `nccl_graph_sys_edge_count=0`
- `vram_failures=0`
- `graph_audit_blocker=none`
- `graph_audit_replay_error_name=cudaSuccess`

## Decision

Do not promote the post-KV boundary as the default graph suffix.

It is correct in the reduced direct scaffold, including cross-position cache
reuse, but it is slower than the promoted suffix in the comparable 4-token
direct probe:

- Promoted compose suffix: `scaffold_projected_slot_step_tok_s=16.895996`,
  `graph_audit_sum_replay_ms=390.940352`, `graph_audit_capture_nodes=112832`
- Post-KV suffix: `scaffold_projected_slot_step_tok_s=15.156673`,
  `graph_audit_sum_replay_ms=530.209665`, `graph_audit_capture_nodes=173720`

The larger capture region adds attention-output and post-attention work to the
graph, but the extra graph size and replay cost outweigh the launch reduction
in this reduced probe. This is correctness/cache evidence, not a steady-state
serving performance gate.

## Follow-Up

Keep the shared `d_decode_position` lifecycle fix.

Treat `post_kv_compose_eager_final_hc` as a diagnostic C1 boundary only. The
next C1 performance work should not move the suffix earlier by itself; it should
reduce fixed-padding overhead inside the promoted graph-stable routed executor
and compose path, or return to full-capture device-state work if that becomes
the larger measured lever.
