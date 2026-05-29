# Sprint 551 - C1 Dynamic-Position Raw Typed KV

Date: 2026-05-29

## Goal

Remove one remaining host-position launch-argument dependency from the
full-capture C1 surface without adding a production flag.

Sprint 546 moved pure kernel position consumers to `RankState::d_decode_position`.
The raw typed-KV runtime path still computed KV row offsets on the host from
`opt.position`, then captured those offsets into store/load kernel arguments.
That is not replay-stable across decode positions.

## Implementation

Added graph-capturable TP runtime row APIs:

- `ds4_tp_runtime_kv_rows_store_f32_device_streams_at_position`
- `ds4_tp_runtime_kv_rows_load_f32_device_streams_at_position`

These APIs keep the layer/kind/slot shape fixed, but their CUDA kernels read
the current decode position from per-GPU device memory and compute the physical
KV row inside the kernel.

Wired graph-mode raw typed-KV store/load in
`engine/compressed_kv_step.cu` to use those dynamic-position APIs. The raw load
now also writes into the raw SWA destination row selected from device position,
instead of capturing a host-computed `raw_row` destination pointer.

No new appliance flag was added. Non-graph static-position runtime APIs remain
unchanged.

## Validation

Remote workspace:

- `/workspace/s551-typed-raw-position`

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance tools/ds4-v100-tp-runtime-smoke`
- PASS

Promoted graph sanity:

- Artifact:
  `/workspace/s551-typed-raw-position-artifacts/none-s551-graph8x4-p262080/summary.json`
- Shape: `8` requests / `8` slots / `256K` context / `4` generated tokens
- Endpoint: `/v100/selected-token`
- Result:
  - `http_200=8`
  - `output_head_first_token=29361`
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

This promoted graph sanity does not exercise raw typed-KV because
`true_ds4_compressed_kv_gate` is still off in the served appliance path.

Targeted TP runtime smoke:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/ds4-v100-tp-runtime-smoke \
  --ctx 262144 --slots 2 --kv-dtype f8 --device-kv-row-at-position \
  --kind attn_raw --layer 2 --slot 1 --position 262081
```

Result:

```text
tp_dynamic_position_kv_row ctx=262144 slots=2 layer=2 ratio=4 slot=1 position=262081 kind=attn_raw physical_row=65 logical_cols=512 bad_values=0 max_abs=0.000000000
```

The smoke compares the new dynamic-position store/load path against the
existing static-position device row path for the same row. Exact agreement at
`physical_row=65` validates both dynamic source-row selection and dynamic raw
SWA destination-row selection.

## Decision

Promote the dynamic-position raw typed-KV runtime path as C1 device-state
cleanup. This is correctness/capture-readiness work only, not a throughput
claim.

Full capture is still intentionally position-keyed. Remaining blockers are the
emitted compressed/indexer typed-KV paths and their host row bookkeeping:

- host `emitted` branch over `opt.position`
- host compressed/indexer row counters
- host row-position arrays used by typed-history reload
- static-position typed-KV store/load calls for emitted compressed/indexer rows

Next C1 work should extend the dynamic-position/device-state design to emitted
compressed/indexer rows as one coordinated change. Do not remove the full-capture
position cache key until those paths are replay-stable.
