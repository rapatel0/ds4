---
sprint: 330
title: TP/EP Typed KV Row Store Primitive
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 330 - TP/EP Typed KV Row Store Primitive

## Goal

Add the first production-oriented typed KV row primitive to
`ds4_v100_tp_runtime`, so the full-layer TP/EP path can begin replacing f32
diagnostic raw/compressed KV buffers with the physical row-sharded arena.

## Why This Sprint

Sprint 329 aligned memory accounting with the runtime's physical row-sharded
KV layout. The remaining integration gap is execution: the all-layer smoke
still computes attention over separate f32 diagnostic buffers.

Before changing the full attention path, the runtime needs a narrow,
validated primitive that can:

- address the same physical row layout used by the allocator
- write an F8 row shard into each rank's arena
- prove the distributed shards reconstruct the expected quantized row
- expose row offsets and byte counts for later full-layer kernels

## Scope

TP/EP only. No PP/layer-split variants. No MTP. This sprint adds a runtime
primitive and focused smoke validation; it does not yet replace the all-layer
attention read path.

## Definition of Done

- [x] `ds4_v100_tp_runtime` exposes a row view for attention and ratio-4
      indexer rows.
- [x] The runtime can write an F8 E4M3 block-128 row into physical TP shards.
- [x] A smoke validates shard gather/decode against the expected quantized row.
- [x] The smoke passes for `32` slots / `256K` on the V100 pod.
- [x] `VISION.md`, `TEMP_STATUS_REPORT_042.md`, cluster artifacts, and this
      sprint doc are updated and committed.

## Outcome

Added TP runtime APIs for production KV row addressing and typed row storage:

- `ds4_v100_tp_runtime_kv_row_view`
- `ds4_v100_tp_runtime_kv_row_roundtrip_f32`

The row view exposes layer, ratio, slot, position, row kind, logical column
count, logical packed bytes, physical row, per-GPU offsets, and per-GPU row
bytes. The roundtrip smoke writes an F8 E4M3 block-128 row into the physical
TP-sharded arena, gathers the shards, decodes the row, and checks the
reconstructed f32 values against the expected quantized row.

The smoke reports canonical host-byte mismatches separately. With nvcc
`--use_fast_math`, the device path can pick an equivalent E8M0 scale byte and
different q bytes while reconstructing the same f32 values. The pass condition
is therefore decoded-value equality, not byte identity against the host
canonical pack.

## Validation

Build on `llm/llamacpp-build-8gpu`:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-runtime-smoke
```

Result: PASS.

Target `32` slots / `256K` / F8, layer `2`, slot `31`, position `262140`:

| Row kind | Physical row | Logical bytes | Row bytes/GPU | Bad decoded values | Max abs | Verdict |
|---|---:|---:|---:|---:|---:|---|
| attention | `65663` | `516` | `65` | `0` | `0.000000000` | PASS |
| indexer | `65535` | `129` | `17` | `0` | `0.000000000` | PASS |

Regression checks:

- default runtime fixture: `fixture_max_abs=0.000000000`
- existing dense KV slice: `max_abs=0.000000000`

Artifacts:

- `logs/from-cluster/sprint330-typed-kv-row/cluster/typed-row-attn-layer2-slot31-pos262140.log`
- `logs/from-cluster/sprint330-typed-kv-row/cluster/typed-row-indexer-layer2-slot31-pos262140.log`
- `logs/from-cluster/sprint330-typed-kv-row/cluster/runtime-fixture.log`
- `logs/from-cluster/sprint330-typed-kv-row/cluster/runtime-dense-kv-slice.log`

## Next Step After This Sprint

Use the row view and typed row store primitive inside
`tools/ds4-v100-tp-ep-full-layer-smoke.cu` while keeping the compact-reference
diff gates enabled.
