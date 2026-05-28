# TEMP_STATUS_REPORT_043

Date: 2026-05-24

## Topline

Sprint 331 added device-to-device f32 store/load APIs for the TP runtime's
typed row-sharded KV arena.

This is the practical bridge from "typed KV row storage exists" to "full-layer
attention can use production typed KV instead of f32 diagnostic buffers."

## What Changed

Updated:

- `ds4_v100_tp_runtime.h`
- `ds4_v100_tp_runtime.cu`
- `tools/ds4-v100-tp-runtime-smoke.cu`

New APIs:

- `ds4_v100_tp_runtime_kv_row_store_f32_device`
- `ds4_v100_tp_runtime_kv_row_load_f32_device`
- `ds4_v100_tp_runtime_kv_row_device_roundtrip_f32`

The store API accepts one device f32 row pointer per GPU and writes that row
through the F8 E4M3 block-128 packer into the rank's physical KV shard. The
load API decodes the distributed row back into device f32 buffers on each GPU
using peer-access reads from all eight shards.

## V100 Validation

Build:

- Command: `make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-runtime-smoke`
- Result: PASS

Target device roundtrips at `32` slots / `256K` / F8:

| Config | Physical row | Logical bytes | Row bytes/GPU | Bad values | Max abs | Result |
|---|---:|---:|---:|---:|---:|---|
| attention, layer 2, slot 31, position 262140 | `65663` | `516` | `65` | `0` | `0.000000000` | PASS |
| indexer, layer 2, slot 31, position 262140 | `65535` | `129` | `17` | `0` | `0.000000000` | PASS |

Regression checks:

- default TP runtime fixture: `fixture_max_abs=0.000000000`
- existing dense KV slice: `max_abs=0.000000000`

## Current Gap

The runtime now has the device primitive the full-layer path needs. The next
step is to use it in `tools/ds4-v100-tp-ep-full-layer-smoke.cu`:

- start with raw-SWA row store/load
- keep the compact-reference diff gates active
- then extend to compressed attention rows and ratio-4 indexer rows

## Artifacts

- `logs/from-cluster/sprint331-device-kv-row/cluster/device-row-attn-layer2-slot31-pos262140.log`
- `logs/from-cluster/sprint331-device-kv-row/cluster/device-row-indexer-layer2-slot31-pos262140.log`
- `logs/from-cluster/sprint331-device-kv-row/cluster/runtime-fixture.log`
- `logs/from-cluster/sprint331-device-kv-row/cluster/runtime-dense-kv-slice.log`
