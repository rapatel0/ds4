# Sprint 553 - C1 Dynamic Bounded Rows for Emitted Typed KV

Date: 2026-05-29

## Goal

Remove captured host-selected bounded-row source/destination pointers from the
graph-mode emitted compressed/indexer typed-KV runtime calls.

Sprint 552 made emitted compressed/indexer typed-KV store/load compute the
physical KV row from `RankState::d_decode_position`, but the graph still
captured pointers offset to the host-selected bounded row inside
`d_attn_comp_rows` and `d_index_comp_rows`. Full capture needs that bounded row
selection to be replay-updated too.

## Implementation

Added bounded dynamic-position TP runtime APIs:

- `ds4_tp_runtime_kv_rows_store_f32_device_streams_at_position_bounded`
- `ds4_tp_runtime_kv_rows_load_f32_device_streams_at_position_bounded`

These kernels compute both values from device decode position:

- physical typed-KV row in the runtime KV cache
- bounded compact row inside the source/destination scratch buffer

Updated graph-mode emitted typed-KV calls in `engine/compressed_kv_step.cu`:

- compressed attention store/load now passes the base `d_attn_comp_rows`
  pointer plus `kBoundedCompRows`
- indexer store/load now passes the base `d_index_comp_rows` pointer plus
  `kBoundedCompRows`
- non-graph static-position paths remain unchanged
- no production flag was added

Extended `smokes/tp-runtime-smoke.cu` with
`--device-kv-row-at-position-bounded` to compare the bounded dynamic path
against the existing static-position row path.

## Validation

Remote workspace:

- `/workspace/s553-bounded-typed-position`

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance tools/ds4-v100-tp-runtime-smoke`
- PASS

Targeted TP runtime smokes:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/ds4-v100-tp-runtime-smoke \
  --ctx 262144 --slots 2 --kv-dtype f8 --device-kv-row-at-position-bounded \
  --kind attn --layer 2 --slot 1 --position 262087
```

```text
tp_dynamic_bounded_position_kv_row ctx=262144 slots=2 layer=2 ratio=4 slot=1 position=262087 kind=attn physical_row=65649 bounded_row=1 logical_cols=512 bad_values=0 max_abs=0.000000000
```

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/ds4-v100-tp-runtime-smoke \
  --ctx 262144 --slots 2 --kv-dtype f8 --device-kv-row-at-position-bounded \
  --kind indexer --layer 2 --slot 1 --position 262087
```

```text
tp_dynamic_bounded_position_kv_row ctx=262144 slots=2 layer=2 ratio=4 slot=1 position=262087 kind=indexer physical_row=65521 bounded_row=1 logical_cols=128 bad_values=0 max_abs=0.000000000
```

The `position=262087` shape is intentionally an emitted ratio-4 position whose
bounded row is `1`, not row zero. Exact agreement validates dynamic physical-row
and dynamic bounded-row selection.

Promoted graph sanity:

- Artifact:
  `/workspace/s553-bounded-typed-position-artifacts/none-s553-graph8x4-p262080/summary.json`
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

Promote the bounded dynamic-position emitted typed-KV runtime path as C1
capture-readiness cleanup. This is not a throughput claim.

Full capture remains intentionally position-keyed. Remaining blockers:

- host `emitted` branch over `opt.position`
- host compressed/indexer row counters
- host row-position arrays used by typed-history reload
- typed-history reload still iterates host-visible row positions and calls
  static-position row loads

The next C1 sprint should move emitted-row topology and row-position metadata
toward device-stable state, or explicitly choose a full-capture boundary that
keeps typed-history outside replay.
