# Sprint 557 - Guard Invalid Full-Capture Replay Probe

Date: 2026-05-29

## Goal

Prevent no-suffix full-capture replay-probe from returning misleading tokens by
replaying a captured full decode step on already-advanced live buffers.

## Change

`engine/decode_loop.cu` now rejects `--decode-cudagraph-replay-probe-gate` when
no CUDA graph suffix stage is selected.

The guard prints:

`tp_ep_decode_cudagraph_replay_probe_blocked ... reason full_capture_live_state_replay_requires_snapshot`

and returns `cudaErrorNotSupported` through the decode path.

The promoted `compose_eager_final_hc` suffix replay path is unchanged.

## Validation

Remote workspace:

- `/workspace/s557-replay-probe-guard`

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- PASS

Blocked no-suffix full-capture replay-probe:

- Artifact:
  `/workspace/s557-replay-probe-guard-artifacts/none-s557-fullgraph-replay-probe-block-serverargs-h396a9fa7/summary.json`
- Shape: `1` request / `8` slots / `256K` context / `1` token
- Result:
  - `http_200=0`
  - server log:
    `tp_ep_decode_cudagraph_replay_probe_blocked layer 0 reason full_capture_live_state_replay_requires_snapshot error_code 801 error_name cudaErrorNotSupported`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`

Promoted suffix replay sanity:

- Artifact:
  `/workspace/s557-replay-probe-guard-artifacts/none-s557-suffix-replay-sanity/summary.json`
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

Promote the guard as validation-harness repair.

No-suffix full-capture replay validation still requires a real fresh-state
replay mechanism: either snapshot/restore the device input state before replay,
or split cache-miss behavior so capture execution is kept and replay is tested
only on a later fresh-state cache hit.
