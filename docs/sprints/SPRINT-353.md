---
sprint: 353
title: TP/EP Fused Compressed-KV Input Fill
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 353 - TP/EP Fused Compressed-KV Input Fill

## Overview

Sprint 352 showed that the emitted compressed-KV path at `32` slots / `256K`
spends `16.776362 ms` per one-token all-layer run just filling compressor and
indexer dense inputs from the same current-hidden vector. The current ratio-4
path launches separate fill kernels for attention compressor KV, attention
compressor gate, indexer projection, indexer compressor KV, and indexer
compressor gate.

This sprint adds an opt-in fused fill that reads `d_current_full` once per rank
and writes those five half-input buffers in one kernel. The goal is to reduce
launch and memory-read fragmentation without changing the default runtime path.

No PP/layer-split work. No MTP. No default semantic change unless V100 evidence
justifies promotion.

## Implementation

1. Add `--true-ds4-compressed-kv-fused-input-fill-gate` to the TP/EP full-layer
   smoke options.
2. Add a CUDA fused fill kernel for the ratio-4 compressor/indexer current-input
   buffers.
3. Use the fused path only when compressed KV, indexer attention, and ratio-4
   execution are active.
4. Add `--fused-compressed-input-fill` to the direct profile harness so the
   V100 A/B is repeatable.
5. Report whether the fused path was active in `tp_ep_compressed_kv_projection`
   rows and profiler summaries.

## Verification

- Local syntax checks pass.
- V100 `sm_70` build passes.
- V100 direct baseline passes at `32` slots / `256K` / emitted-row
  `position=262143`.
- V100 direct fused-fill candidate passes at the same shape.
- Compare generated decode tok/s and compressed-KV internal timings.

## Definition of Done

- [x] Fused input-fill gate is implemented and defaults off.
- [x] Direct profiler flag is implemented.
- [x] V100 build passes.
- [x] V100 baseline and fused-fill A/B runs pass.
- [x] Docs/status/temp report are updated with the decision.
- [x] Artifacts are copied into `logs/from-cluster/`.
- [x] Sprint artifacts are committed.

## Outcome

Implemented `--true-ds4-compressed-kv-fused-input-fill-gate` in
`tools/ds4-v100-tp-ep-full-layer-smoke.cu` and
`--fused-compressed-input-fill` in `tools/ds4-v100-tp-ep-profile.py`.

The fused path is selected only for ratio-4 layers with indexer attention
enabled. It reads each rank's `[slots,4096]` `d_current_full` vector once and
writes the five current-derived half-input buffers used by the attention
compressor and indexer compressor path:

- `attn_compress_kv`
- `attn_compress_gate`
- `indexer_proj`
- `indexer_compress_kv`
- `indexer_compress_gate`

V100 same-binary A/B, `32` slots / `256K`, emitted-row `position=262143`,
one decode step:

| Variant | Fused layers | Decode tok/s | Decode ms | Pre-EP compressed-KV ms | Compressed-KV sum ms | First token |
|---|---:|---:|---:|---:|---:|---:|
| control | `0` | `79.011931` | `405.002124` | `130.391665` | `129.840446` | `54639` |
| fused input fill | `21` | `80.534845` | `397.343535` | `129.781758` | `129.263171` | `54639` |

Internal fill timers:

| Variant | Attention input fill ms | Indexer input fill ms |
|---|---:|---:|
| control | `12.698553` | `3.918875` |
| fused input fill | `12.530838` | `4.838720` |

## Decision

The fused path is correct and gives a small positive topline signal
(`+1.9%` decode tok/s in this run), but the compressed-KV stage itself changes
by less than `1 ms`. This is not strong enough to promote as a default.

Keep the fused fill as an opt-in diagnostic. The next material lever should be
the larger compressor/indexer state and dense boundaries:

- attention state/emit
- indexer state/emit
- shared compressor state update plus row emit
- eventually HMMA/CUTLASS dense kernel shape tuning once state/emit overhead is
  better isolated

Artifacts:

```text
logs/from-cluster/sprint353-fused-compressed-input-fill/cluster/
```
