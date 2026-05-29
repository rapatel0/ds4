# Sprint 552 - C1 Dynamic-Position Emitted Typed KV

Date: 2026-05-29

## Goal

Continue the full-capture device-position program by removing static host
`opt.position` from the graph-mode emitted compressed/indexer typed-KV runtime
store/load calls.

Sprint 551 made raw typed-KV store/load compute physical rows from
`RankState::d_decode_position`. The emitted compressed attention and indexer
paths still used the older graph-mode runtime APIs whose row view is computed
from host `opt.position` before capture. That leaves stale row offsets baked
into captured graphs.

## Implementation

Updated graph-mode batch store/load in `engine/compressed_kv_step.cu`:

- emitted compressed attention rows now call
  `ds4_tp_runtime_kv_rows_store_f32_device_streams_at_position` and
  `ds4_tp_runtime_kv_rows_load_f32_device_streams_at_position`
- emitted indexer rows now call the same dynamic-position APIs
- each call passes per-rank `RankState::d_decode_position` so the runtime
  kernels compute the physical KV row at replay time
- non-graph static-position calls remain unchanged
- no production flag was added

The bounded row inside `d_attn_comp_rows` / `d_index_comp_rows` is still selected
by host row bookkeeping. This sprint only removes static physical-KV-row
selection from the emitted typed-KV runtime calls.

## Validation

Remote workspace:

- `/workspace/s552-emitted-typed-position`

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance tools/ds4-v100-tp-runtime-smoke`
- PASS

Targeted TP runtime smokes:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/ds4-v100-tp-runtime-smoke \
  --ctx 262144 --slots 2 --kv-dtype f8 --device-kv-row-at-position \
  --kind attn --layer 2 --slot 1 --position 262083
```

```text
tp_dynamic_position_kv_row ctx=262144 slots=2 layer=2 ratio=4 slot=1 position=262083 kind=attn physical_row=65648 logical_cols=512 bad_values=0 max_abs=0.000000000
```

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/ds4-v100-tp-runtime-smoke \
  --ctx 262144 --slots 2 --kv-dtype f8 --device-kv-row-at-position \
  --kind indexer --layer 2 --slot 1 --position 262083
```

```text
tp_dynamic_position_kv_row ctx=262144 slots=2 layer=2 ratio=4 slot=1 position=262083 kind=indexer physical_row=65520 logical_cols=128 bad_values=0 max_abs=0.000000000
```

These smokes compare the dynamic-position row APIs against the existing
static-position device row path for the same logical rows. Exact agreement
validates physical-row computation for both emitted compressed attention rows
and emitted indexer rows.

Promoted graph sanity:

- Artifact:
  `/workspace/s552-emitted-typed-position-artifacts/none-s552-graph8x4-p262080/summary.json`
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

The promoted graph sanity does not exercise compressed/indexer typed-KV because
`true_ds4_compressed_kv_gate` is still off in the served appliance path. It is
included as a regression check for the promoted graph suffix path only.

## Decision

Promote the graph-mode emitted compressed/indexer typed-KV runtime calls to the
dynamic-position APIs. This is C1 capture-readiness work only, not a throughput
claim.

Full capture remains intentionally position-keyed. Remaining blockers:

- host `emitted` branch over `opt.position`
- host compressed/indexer bounded-row counters
- host row-position arrays used by typed-history reload
- captured source/destination bounded-row pointers for emitted rows

The next C1 sprint should make emitted-row topology and bounded-row selection
device-stable before attempting to remove the full-capture position cache key.
