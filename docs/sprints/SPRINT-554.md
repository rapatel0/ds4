# Sprint 554 - C1 Dynamic Typed-History Row Loads

Date: 2026-05-29

## Goal

Remove static host row-position arguments from graph-mode typed-history reloads.

Sprints 551-553 made current typed-KV store/load physical and bounded rows
derive from `RankState::d_decode_position`, but history reload still iterated
host row-position arrays and passed the selected historical position into the TP
runtime. Full capture needs that source position to be replay-updated too.

## Implementation

Added a graph-capable TP runtime history-row load API:

- `ds4_tp_runtime_kv_rows_load_f32_device_streams_at_history_row`

The runtime kernel computes the historical source position from:

- current `d_decode_position`
- bounded history row
- layer compression ratio
- typed-KV row kind

Then it loads that physical typed-KV row into the matching bounded row of the
destination compact history buffer.

Updated graph-mode typed-history reload in `engine/attention_read.cu`:

- compressed attention history reloads pass base `d_attn_comp_rows`
- indexer history reloads pass base `d_index_comp_rows`
- non-graph static-position reloads remain unchanged
- no production flag was added

Extended `smokes/tp-runtime-smoke.cu` with `--device-kv-history-row` to compare
dynamic history-row load against the existing static-position row load.

## Validation

Remote workspace:

- `/workspace/s554-history-row-position`

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance tools/ds4-v100-tp-runtime-smoke`
- PASS

Targeted TP runtime smokes:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/ds4-v100-tp-runtime-smoke \
  --ctx 262144 --slots 2 --kv-dtype f8 --device-kv-history-row \
  --kind attn --layer 2 --slot 1 --position 262095 --history-row 1
```

```text
tp_dynamic_history_kv_row ctx=262144 slots=2 layer=2 ratio=4 slot=1 position=262095 history_position=262087 kind=attn physical_row=65649 bounded_row=1 logical_cols=512 bad_values=0 max_abs=0.000000000
```

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/ds4-v100-tp-runtime-smoke \
  --ctx 262144 --slots 2 --kv-dtype f8 --device-kv-history-row \
  --kind indexer --layer 2 --slot 1 --position 262095 --history-row 1
```

```text
tp_dynamic_history_kv_row ctx=262144 slots=2 layer=2 ratio=4 slot=1 position=262095 history_position=262087 kind=indexer physical_row=65521 bounded_row=1 logical_cols=128 bad_values=0 max_abs=0.000000000
```

Promoted graph sanity:

- Artifact:
  `/workspace/s554-history-row-position-artifacts/none-s554-graph8x4-p262080/summary.json`
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

The promoted graph sanity still does not exercise compressed/indexer typed-KV
because `true_ds4_compressed_kv_gate` is off in the served appliance path. It
is included as a regression check for the promoted graph suffix path only.

## Decision

Promote the dynamic typed-history row-load runtime path as C1
capture-readiness cleanup. This is not a throughput claim.

Full capture remains position-keyed. Remaining blockers:

- host `emitted` branch over `opt.position`
- host compressed/indexer row counters
- host row-position arrays for emitted/current row bookkeeping
- graph topology still differs between emitted and non-emitted positions

The next C1 sprint should make emitted/non-emitted graph topology stable, or
explicitly restrict full-capture reuse to positions where bounded compressed
history is saturated and the captured topology is valid.
