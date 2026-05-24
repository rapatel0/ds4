---
sprint: 329
title: TP/EP Physical Row-Sharded KV Budget Alignment
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 329 - TP/EP Physical Row-Sharded KV Budget Alignment

## Goal

Make the TP planner and allocation smoke budget the same physical KV layout
that `ds4_v100_tp_runtime` already allocates.

## Why This Sprint

Sprint 328 proved the target memory shape is physically allocatable, but the
follow-up audit found a small accounting mismatch:

- planner/arena aggregate-byte TP shard at `32` slots / `256K`: `3.396 GiB/GPU`
- runtime physical row-sharded layout at `32` slots / `256K`: `3.453 GiB/GPU`
- difference: about `58.5 MiB/GPU`

This does not threaten fit, but it matters. The production appliance should
budget the layout it actually addresses: each row is sharded independently
across TP ranks, so per-row quantization/block padding must be included before
summing rows.

## Scope

TP/EP only. No PP/layer-split variants. No MTP. This sprint fixes memory
contract accuracy; it does not change serving semantics.

## Implementation Plan

1. Update `tools/ds4-v100-plan-tp.c`:

   - compute physical row-sharded persistent KV bytes per GPU
   - keep ideal aggregate-sharded bytes visible for comparison
   - use physical row-sharded bytes in resident budget/admission
   - emit the physical value in JSON

2. Update `tools/ds4-v100-tp-kv-arena-smoke.cu`:

   - allocate physical row-sharded KV bytes, matching
     `ds4_v100_tp_runtime`
   - print the ideal-vs-physical delta

3. Validate against the V100 pod:

   - planner target `32` slots / `256K` / F8 real pack
   - arena target `32` slots / `256K` / F8 real pack
   - runtime smoke report for `32` slots / `256K` / F8

## Definition of Done

- [x] Planner budget uses runtime-matching row-sharded KV bytes.
- [x] Arena smoke allocates runtime-matching row-sharded KV bytes.
- [x] The planner still reports the ideal aggregate-sharded value for context.
- [x] V100 planner and allocation gates pass at `32` slots / `256K`.
- [x] Runtime smoke KV byte count matches the planner physical value.
- [x] `VISION.md`, `TEMP_STATUS_REPORT_041.md`, cluster/local artifacts, and
      this sprint doc are updated and committed.

## Outcome

Updated `tools/ds4-v100-plan-tp.c` and
`tools/ds4-v100-tp-kv-arena-smoke.cu` so the resident budget uses the physical
row-sharded KV layout already allocated by `ds4_v100_tp_runtime`.

The planner still reports the ideal aggregate-sharded value as a lower-bound
comparison, but it no longer uses that lower bound for fit/admission decisions.

## Validation

Build on `llm/llamacpp-build-8gpu`:

```text
make -j80 CUDA_ARCH=sm_70 \
  tools/ds4-v100-plan-tp \
  tools/ds4-v100-tp-kv-arena-smoke \
  tools/ds4-v100-tp-runtime-smoke
```

Result: PASS.

Target `32` slots / `256K` / F8, real pack:

| Source | KV bytes / GPU | Notes |
|---|---:|---|
| planner physical row-sharded | `3707940864` | resident budget source |
| runtime smoke | `3707940864` | `ds4_v100_tp_runtime` allocation |
| arena smoke | `3707940864` | allocation proof |
| ideal aggregate shard | `3646642176` | lower-bound comparison only |

The physical row-shard overhead is `61298688` bytes, or about
`58.46 MiB/GPU`.

Updated target budget:

| Case | Per-GPU budget | No-reserve allocation | Free after allocation | Verdict |
|---|---:|---:|---:|---|
| `32` slots / `256K` / F8 | `27.058 GiB` | `25.058 GiB` | `6.366 GiB` | PASS |

Admission tiers with physical row-sharded KV:

| Context | Max slots | Per-GPU total at max |
|---:|---:|---:|
| `128K` | `125` | `31.97 GiB` |
| `256K` | `62` | `31.87 GiB` |
| `512K` | `31` | `31.86 GiB` |
| `1M` | `15` | `31.53 GiB` |

Artifacts:

- `logs/from-cluster/sprint329-row-sharded-kv/cluster/plan-slots32-ctx262144-f8.md`
- `logs/from-cluster/sprint329-row-sharded-kv/cluster/plan-slots32-ctx262144-f8.json`
- `logs/from-cluster/sprint329-row-sharded-kv/cluster/runtime-slots32-ctx262144-f8.log`
- `logs/from-cluster/sprint329-row-sharded-kv/cluster/arena-slots32-ctx262144-f8.log`

## Next Step

Wire the production typed KV arena into the all-layer TP/EP attention path:

- use the runtime row descriptors as the source of truth for raw SWA,
  compressed attention, and ratio-4 indexer rows
- add typed row store/load kernels for the full-layer smoke
- preserve compact-reference diff gates while replacing f32 diagnostic row
  buffers with production arena reads
- rerun the `32` slot / `256K` all-layer compressed-history gate

## Risks

- Admission tiers may move slightly because the physical layout is larger than
  the ideal aggregate-sharded estimate. That is expected; prefer accurate
  accounting over optimistic fit.
- This only aligns byte budgeting. The next runtime sprint still needs to make
  the full-layer path read/write the production typed arena instead of its
  f32 diagnostic arrays.
