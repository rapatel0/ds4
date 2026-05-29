# Sprint 556 - Full-Capture Replay-Probe Divergence

Date: 2026-05-29

## Goal

Localize the Sprint 555 full-capture correctness failure before attempting any
new full-capture cache-key relaxation.

## Result

The failure is in no-suffix full-capture replay-probe semantics, not plain graph
capture and not transport topology.

No-suffix full graph capture without replay-probe preserves the same selected
token as eager at the exact validation shape. Adding immediate replay-probe
changes the token.

The code path explains the result: replay-probe captures by executing
`run_one_step()` on the live decode buffers, then immediately launches the
captured full graph against those already-advanced buffers. For a no-suffix full
graph, that is a second application of the decode layer state, not a clean replay
from the same input state. The promoted suffix graph can tolerate this better
because the suffix inputs are produced by an eager prefix and are not overwritten
the same way by the suffix replay.

## Validation

Remote workspace:

- `/workspace/s556-full-capture-divergence`

Build with temporary diagnostic cache-key relaxation:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- PASS

Same-shape eager control with stage checksums:

- Artifact:
  `/workspace/s556-full-capture-divergence-artifacts/none-s556-eager8x2-p262080-checksum-serverargs-h91108f54/summary.json`
- Shape: `8` requests / `8` slots / `256K` context / `2` generated tokens
- Result:
  - `http_200=8`
  - `output_head_first_token=128819`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`
  - `vram_failures=0`

No-suffix full-capture with immediate replay-probe:

- Artifact:
  `/workspace/s556-full-capture-divergence-artifacts/none-s556-fullgraph8x2-p262080-checksum-serverargs-hf721eeed/summary.json`
- Result:
  - `http_200=8`
  - `output_head_first_token=118235`
  - `graph_audit_blocker=none`
  - `graph_audit_replay_succeeded=43`
  - `graph_audit_replay_attempted=43`
  - `graph_audit_persistent_invalidations=0`
  - `graph_audit_persistent_invalidate_position=0`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`

No-suffix full-capture without replay-probe:

- Artifact:
  `/workspace/s556-full-capture-divergence-artifacts/none-s556-fullgraph8x2-p262080-noprobe-serverargs-h2c621b83/summary.json`
- Result:
  - `http_200=8`
  - `output_head_first_token=128819`
  - `graph_audit_blocker=none`
  - `graph_audit_capture_attempted=43`
  - `graph_audit_capture_succeeded=43`
  - `graph_audit_replay_attempted=0`
  - `graph_audit_replay_succeeded=0`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`

The stage-checksum run confirmed broad replay-state drift, but it is not a
valid first-stage locator for replay-probe because replay checksums are emitted
after the whole graph has replayed, not inside the captured graph at each stage.

## Decision

Do not use no-suffix replay-probe output as a full-capture parity signal until
the probe either snapshots/restores device input state before replay or is split
so full-capture cache-miss validation keeps the capture-executed result and
tests replay only on a fresh state.

The Sprint 555 cache-key relaxation remains rejected. The local diagnostic
cache-key patch was reverted.

## Next

Fix the full-capture replay validation harness before more full-capture cache-key
experiments:

- keep the existing promoted suffix replay path unchanged
- make no-suffix replay-probe explicitly avoid double-applying a captured full
  step on live buffers, or add a real device-state snapshot/restore probe
- only then resume full-capture persistent reuse or emitted-topology work
