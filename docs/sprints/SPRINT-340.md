---
sprint: 340
title: TP/EP Skip Current Typed KV Reload
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 340 - TP/EP Skip Current Typed KV Reload

## Goal

Reduce typed-history serving overhead by removing same-step typed KV
store-then-load roundtrips for the current raw-SWA, compressed-attention, and
ratio-4 indexer rows.

## Scope

TP/EP only. No PP/layer-split work. No MTP. This sprint keeps strict
current-row roundtrip diagnostics available by making the skip an explicit
performance gate.

## Definition of Done

- [x] Add an opt-in binary gate for skipping current-row typed reloads.
- [x] Add an appliance env toggle and command emission for the gate.
- [x] Preserve production typed KV stores for raw, compressed, and indexer
      rows.
- [x] Keep immediate attention staging valid without same-step typed reloads.
- [x] Keep future history reload semantics available by not marking skipped
      current rows as typed-loaded cache entries.
- [x] Expose the gate in HTTP response/status/metrics.
- [x] Build on the V100 pod.
- [x] Run the same `32` request / `32` slot / `256K` / `8` token HTTP A/B and
      compare against Sprint 339.

## Outcome

Added:

- binary flag:
  `--true-ds4-attention-typed-kv-skip-current-load-gate`
- appliance env:
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_CURRENT_LOAD`

When enabled, typed raw-SWA, compressed-attention, and ratio-4 indexer paths
still store production typed KV rows, but they do not immediately load the same
row back from typed KV in the same layer step. Raw SWA staging is populated by
the existing local raw-SWA staging kernel; compressed/indexer staging keeps the
already-computed emitted row. Current rows skipped this way are not marked as
persistent typed-loaded history cache entries, so future non-current history
loads can still reload typed rows.

The HTTP server now reports the skip gate in `/status`, `/metrics`, and
generation response metadata.

## Validation

Local:

```text
bash -n tools/ds4-v100-run-appliance.sh
python3 -m py_compile tools/ds4-v100-tp-ep-http-ab.py
git diff --check
```

Result: PASS.

V100 build:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS.

Same-shape A/B:

```text
endpoint: /v1/chat/completions
requests: 32 concurrent
slots: 32
ctx: 262144
tokens/request: 8
diagnostic output head: on
HC persistent state: on
typed candidate: typed-history + skip-current-load
```

Summary:

```text
case           http  batch  server tok/s  decode tok/s  client tok/s
control        32/32 32     316.297621    735.600737    104.071414
typed-history  32/32 32      74.383163     86.322558     17.524156
```

Typed candidate evidence:

```text
typed_raw_lines 942
typed_compressed_lines 105
typed_indexer_lines 105
typed_history_lines 898
history_loaded_attn_rows_2 84
history_loaded_indexer_rows_2 84
history_reloaded_attn_rows_nonzero 105
history_reloaded_indexer_rows_nonzero 105
typed_current_load_0 1152
typed_current_load_1 0
```

Final GPU state was idle: all devices reported `0 MiB` used.

## Decision

The skip-current-load gate works and improved typed-history serving from
Sprint 339's `68.358523` server wall tok/s to `74.383163`, but the path is
still about `4.25x` slower than the same-run control and `8.5x` slower by
decode tok/s.

The remaining gap is probably dominated by typed KV store overhead and broad
per-slot/per-layer diagnostic store plumbing, not same-step current loads.
Next sprint should measure store-only cost by family and then either batch the
typed row stores or move the store operation into the producer kernels.

## Artifacts

- `logs/from-cluster/sprint340-skip-current-typed-load/cluster/summary.tsv`
- `logs/from-cluster/sprint340-skip-current-typed-load/cluster/control/`
- `logs/from-cluster/sprint340-skip-current-typed-load/cluster/typed-history/`
