# Sprint 545 - C1 Full-Capture Position Dependency Map

Date: 2026-05-29

## Goal

Determine whether full graph capture can safely stop invalidating persistent
graphs by decode position with a narrow device-position scalar patch, or whether
the position dependency is broader and needs a staged design.

## Context

Sprint 544 proved that full capture is now correctness-clean on the current
surface:

- response/checksum multisets matched eager
- `graph_audit_blocker=none`
- `graph_audit_capture_succeeded=43`
- `graph_audit_replay_succeeded=43`
- peer-copy/SYS `0`
- NCCL graph SYS edges `0`

The remaining blocker was persistent reuse:

- `graph_audit_persistent_cache_hits=0`
- `graph_audit_persistent_cache_misses=43`
- `graph_audit_persistent_invalidate_position=43`

## Trace

The full-capture path still bakes `opt.position` into multiple semantic
surfaces:

1. RoPE kernel launch arguments in `engine/compressed_kv_step.cu`.
   Attention, compressed-attention, and indexer RoPE kernels receive
   `opt.position` or derived values such as `opt.position + 1 - ratio`.
2. Compressed-KV emission decisions in `engine/compressed_kv_step.cu`.
   `emitted` is computed on the host from `(opt.position + 1) % ratio`; it
   controls whether bounded compressed rows are produced, stored, and later
   read.
3. Typed-KV runtime row selection in `engine/compressed_kv_step.cu` and
   `engine/attention_read.cu`. Store/load/view calls pass `opt.position` or a
   recorded row position into the runtime to select the physical KV row.
4. Host bookkeeping arrays in `RankState`. The path records
   `attn_comp_row_position_layers`, `index_comp_row_position_layers`, and their
   loaded-position mirrors. Attention history load uses those values to decide
   whether a visible compressed row is already current or must be reloaded.
5. Raw SWA row selection in `engine/compressed_kv_step.cu` and
   `engine/attention_read.cu`. Raw update/read/window code uses
   `opt.position % kRawSwaRows`.
6. Persistent graph cache safety in `engine/decode_loop.cu`. Full capture is
   still position-keyed because the captured launch arguments and host-side
   branches above are not replay-stable across decode positions.

## Decision

Do not remove position from the full-capture persistent cache key in one patch.

A single device-resident scalar would not be sufficient because the full graph
does not only read position inside kernels. Position also selects host-side
runtime KV rows, controls emitted-row branches, mutates per-layer row metadata,
and changes raw-window addressing. Dropping the cache key now would recreate the
same class of stale cross-position replay bug repaired in Sprint 538.

## Required Stages

1. Introduce replay-updated device position state and convert pure kernel
   position consumers first, starting with RoPE/raw-row kernels whose only
   position dependency is a scalar argument.
2. Separate or device-stabilize compressed-KV emission decisions so capture
   topology does not depend on the host-computed emitted branch for a specific
   position.
3. Make typed-KV store/load/view semantics graph-safe. Either the runtime calls
   stay outside the captured region, or the captured region must use
   replay-stable device-indexed row selection.
4. Move or stabilize the host bookkeeping that tracks compressed-row positions
   and loaded-position mirrors.
5. Only after those surfaces are replay-stable, remove position from the
   full-capture persistent cache key and rerun the Sprint 544 full-capture
   correctness gate followed by a warmed long-generation performance gate.

## Follow-Up

Keep the promoted suffix replay path as the production graph win. The next C1
work should either start Stage 1 of replay-updated dynamic position, or pivot to
another measured launch/capture lever such as a safer A5/A6 fusion target or
EP full-shape masked executor work.
