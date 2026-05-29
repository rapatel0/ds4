# Sprint 547 - C1 Compressed-KV Topology Boundary

Date: 2026-05-29

## Goal

Review the next full-capture position stage after Sprint 546 and decide whether
compressed-KV emitted-row topology can be made graph-stable with a narrow
device-masked kernel pass.

## Finding

The emitted-row branch is not just a kernel launch condition.

In `engine/compressed_kv_step.cu`, `emitted` gates:

- compressed attention row emission kernels
- indexer compressed row emission kernels
- `attn_comp_rows_written_layers` and `index_comp_rows_written_layers`
- compressed-row position bookkeeping
- typed-KV compressed store/load runtime calls
- typed-KV indexer store/load runtime calls
- indexer scoring/top-k work after an emitted index row

Those effects cross the engine/runtime boundary. The typed-KV operations still
select physical rows through host runtime calls using `opt.position`, and the
host row counters define which bounded row later history reads should consider
visible.

## Decision

Do not make the emitted-row kernels always launch as a standalone change.

That would increase work on non-emitted positions, including the promoted suffix
graph path's eager prefix, without making full-capture persistent replay safe.
The graph would still be position-dependent through typed-KV runtime calls and
host row bookkeeping.

## Required Design Split

One of these must happen before full capture can drop the position cache key:

1. Keep compressed-KV typed runtime work outside the captured region and capture
   a later graph boundary, after dynamic KV topology has settled.
2. Refactor typed-KV row store/load/view into replay-stable device-indexed
   operations and move compressed-row visibility/bookkeeping into device state.

The first option is likely smaller and should be evaluated before a typed-KV
runtime refactor.

## Validation

No code changed in this sprint. Sprint 546's remote appliance build remains the
latest code validation:

- `/workspace/s546-device-position`
- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- PASS

## Follow-Up

Plan the next C1 sprint around capture-boundary selection rather than blindly
device-masking emitted-row kernels. The target question is: what is the largest
post-KV graph region that is replay-stable after Sprint 546 and still larger
than the currently promoted `compose_eager_final_hc` suffix?
