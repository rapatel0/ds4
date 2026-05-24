---
sprint: 331
title: TP/EP Device F32 Typed KV Store Load
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 331 - TP/EP Device F32 Typed KV Store Load

## Goal

Expose production-facing device-pointer APIs for storing and loading f32 rows
through the TP runtime's typed row-sharded KV arena.

## Why This Sprint

Sprint 330 proved the runtime can address and validate typed KV rows, but that
roundtrip used an internal synthetic row and host gather. The full-layer smoke
needs a device-to-device primitive: take an f32 row already resident on each
rank, pack it into the F8 TP-sharded arena, then decode it back into device f32
buffers for attention-read validation.

## Scope

TP/EP only. No PP/layer-split variants. No MTP. This sprint adds the runtime
store/load primitive and a focused runtime smoke; it does not yet replace the
full-layer attention buffers.

## Definition of Done

- [x] Public runtime APIs exist for device f32 store and device f32 load.
- [x] The APIs use the same physical row view and F8 row-sharded arena as
      Sprints 329-330.
- [x] A smoke validates attention and indexer device roundtrips at
      `32` slots / `256K` on the V100 pod.
- [x] Existing runtime fixture and dense KV slice smokes still pass.
- [x] `VISION.md`, `TEMP_STATUS_REPORT_043.md`, cluster artifacts, and this
      sprint doc are updated and committed.

## Outcome

Added public runtime APIs for device f32 rows:

- `ds4_v100_tp_runtime_kv_row_store_f32_device`
- `ds4_v100_tp_runtime_kv_row_load_f32_device`
- `ds4_v100_tp_runtime_kv_row_device_roundtrip_f32`

The store API takes one device f32 row pointer per GPU, packs each row through
the F8 E4M3 block-128 path, and writes only that rank's physical shard into the
typed KV arena. The load API decodes the distributed row back to device f32
buffers on each rank using peer-access reads from the eight KV shards.

This is the primitive the full-layer smoke needs before replacing f32
diagnostic raw/compressed KV buffers.

## Validation

Build on `llm/llamacpp-build-8gpu`:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-runtime-smoke
```

Result: PASS.

Target `32` slots / `256K` / F8, layer `2`, slot `31`, position `262140`:

| Row kind | Physical row | Logical bytes | Row bytes/GPU | Bad values | Max abs | Verdict |
|---|---:|---:|---:|---:|---:|---|
| attention | `65663` | `516` | `65` | `0` | `0.000000000` | PASS |
| indexer | `65535` | `129` | `17` | `0` | `0.000000000` | PASS |

Regression checks:

- default runtime fixture: `fixture_max_abs=0.000000000`
- existing dense KV slice: `max_abs=0.000000000`

Artifacts:

- `logs/from-cluster/sprint331-device-kv-row/cluster/device-row-attn-layer2-slot31-pos262140.log`
- `logs/from-cluster/sprint331-device-kv-row/cluster/device-row-indexer-layer2-slot31-pos262140.log`
- `logs/from-cluster/sprint331-device-kv-row/cluster/runtime-fixture.log`
- `logs/from-cluster/sprint331-device-kv-row/cluster/runtime-dense-kv-slice.log`

## Next Step

Use `ds4_v100_tp_runtime_kv_row_store_f32_device` and
`ds4_v100_tp_runtime_kv_row_load_f32_device` in
`tools/ds4-v100-tp-ep-full-layer-smoke.cu` for the raw-SWA attention row first,
then extend the same path to compressed attention and indexer rows while
preserving compact-reference diff gates.
