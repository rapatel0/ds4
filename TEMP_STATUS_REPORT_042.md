# TEMP_STATUS_REPORT_042

Date: 2026-05-24

## Topline

Sprint 330 added the first typed KV row storage primitive to the TP runtime.

This is the execution-side bridge needed after Sprint 329: the runtime can now
address an attention or ratio-4 indexer row in the physical row-sharded arena,
write an F8 E4M3 block-128 row into the distributed shards, gather those shards,
decode them, and validate the reconstructed f32 row.

## What Changed

Updated:

- `ds4_v100_tp_runtime.h`
- `ds4_v100_tp_runtime.cu`
- `tools/ds4-v100-tp-runtime-smoke.cu`

New APIs:

- `ds4_v100_tp_runtime_kv_row_view`
- `ds4_v100_tp_runtime_kv_row_roundtrip_f32`

The row view exposes:

- layer/ratio/slot/position
- row kind: attention or indexer
- logical columns and logical packed bytes
- physical row
- per-GPU offset and row byte count

## V100 Validation

Build:

- Command: `make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-runtime-smoke`
- Result: PASS

Target row tests:

| Config | Physical row | Logical bytes | Row bytes/GPU | Bad decoded values | Max abs | Result |
|---|---:|---:|---:|---:|---:|---|
| attention, layer 2, slot 31, position 262140 | `65663` | `516` | `65` | `0` | `0.000000000` | PASS |
| indexer, layer 2, slot 31, position 262140 | `65535` | `129` | `17` | `0` | `0.000000000` | PASS |

Regression checks:

- default TP runtime fixture: `fixture_max_abs=0.000000000`
- existing dense KV slice: `max_abs=0.000000000`

## Notes

The smoke reports byte mismatches against the host canonical pack, but those do
not fail the gate. Under nvcc `--use_fast_math`, the device packer can choose an
equivalent E8M0 scale and different q bytes while reconstructing the same f32
values. The integration contract for this sprint is decoded row equality.

## Current Gap

The primitive is ready, but the full-layer smoke still computes attention from
f32 diagnostic raw/compressed buffers. The next sprint should use the row view
and typed store primitive in `tools/ds4-v100-tp-ep-full-layer-smoke.cu`, while
keeping compact-reference diffs active.

## Artifacts

- `logs/from-cluster/sprint330-typed-kv-row/cluster/typed-row-attn-layer2-slot31-pos262140.log`
- `logs/from-cluster/sprint330-typed-kv-row/cluster/typed-row-indexer-layer2-slot31-pos262140.log`
- `logs/from-cluster/sprint330-typed-kv-row/cluster/runtime-fixture.log`
- `logs/from-cluster/sprint330-typed-kv-row/cluster/runtime-dense-kv-slice.log`
