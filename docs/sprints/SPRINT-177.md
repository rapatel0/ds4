# Sprint 177 - FFN-Only Microbatch Wavefront

Date: 2026-05-22
Status: Completed

## Overview

Sprint 177 targets the scheduler gap left by Sprints 160, 166, 168, 175, and
176. Fixed slot chunking, ready-window chunking, whole-layer wavefront batching,
wrapper-level routed executor fusion, and per-layer TP/EP overlays are all
correct but slower. The remaining practical scheduling hypothesis is narrower:
batch only the FFN/routed-expert portion of a layer, while letting attention,
HC prep, stage handoff, and per-slot progress remain as overlapped as the
current per-step event-handoff path.

The sprint introduces a default-off diagnostic path:

```text
DS4_V100_ASYNC_FFN_WAVEFRONT=1
DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK=2|4
```

The goal is to create `n_slots * routes` routed-FFN work at 256K without the
known regressions from `DS4_V100_ASYNC_SLOT_CHUNK` or
`DS4_V100_ASYNC_LAYER_WAVEFRONT`.

## Non-Goals

- No default promotion unless served A/B clears the throughput gate.
- No new monolithic MXFP4 gate/up + down kernel in this sprint.
- No TP/EP topology changes.
- No MTP changes.
- No changes to the default per-step path when the new gate is unset.

## Architecture

The current winning 16-slot/256K path preserves stage overlap by advancing one
slot at a time through each stage. That keeps latency low but presents routed
FFN as six-route work. Whole-stage and whole-layer batching increase routed
shape density but stall handoff and lose more than they gain.

Sprint 177 splits one HC decode layer into three pieces:

```text
prepare_ffn:
  hidden_hc -> attention -> after_attn_hc -> ffn_cur -> ffn_norm

batch_ffn:
  ffn_norm[slots] -> existing execute_ffn_delta_batch()

finish_ffn:
  ffn_delta + after_attn_hc + ffn_split -> next_hidden_hc
```

The replay worker then tracks each slot inside a stage as:

```text
not_started -> ffn_ready(layer) -> layer_done -> next layer
```

Only contiguous slots at the same layer and in `ffn_ready` state are grouped.
This preserves per-stage event handoff semantics and isolates the exact lever
that has not yet been tested: dense routed FFN only.

## Implementation

### Phase 1 - Layer Executor Split

- Add a small prepared-FFN API to `ds4_v100_layer_execute.h`.
- Implement prepare and finish helpers in `ds4_v100_layer_execute.c` by factoring
  the existing `ds4_v100_layer_execute_hc_decode()` / batch code.
- Reuse `ds4_v100_layer_batch_scratch` tensors for `after_attn_hc`, `ffn_split`,
  `ffn_norm`, and `ffn_delta`.
- Reuse the existing static `execute_ffn_delta_batch()` for the dense FFN call.

### Phase 2 - Scheduler API

- Add one-stage one-layer FFN microbatch API in `ds4_v100_scheduler.c` that:
  - prepares a slot through FFN-ready state;
  - batches prepared FFN slots at the same layer;
  - finishes those slots and advances `cur_hc`.
- Preserve existing scheduler reports enough for route count, TurboMind use, and
  timing counters.

### Phase 3 - Replay Worker

- Add `DS4_V100_ASYNC_FFN_WAVEFRONT=1` worker mode beside the current per-step
  and layer-wavefront workers.
- Add `DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK` with default `2` and max scheduler
  slots.
- Keep event handoff and stage completion semantics unchanged.
- Record explicit evidence when FFN microbatching is active, including chunk
  size and observed grouped slots/routes.

### Phase 4 - Launcher And Validation

- Export/print the new env flags in `tools/ds4-v100-run-appliance.sh`.
- Add or extend a CUDA smoke that compares FFN microbatch output against the
  existing per-slot path for a small slot count.
- Run V100 build, smoke, selected-token/full-scheduler validation, and served
  16-slot/256K A/B.

## Files Summary

| File | Change |
|---|---|
| `ds4_v100_layer_execute.h` | Add prepared-FFN microbatch types/APIs |
| `ds4_v100_layer_execute.c` | Split HC layer decode around FFN and reuse batched FFN |
| `ds4_v100_scheduler.h` | Expose stage FFN microbatch layer API if needed |
| `ds4_v100_scheduler.c` | Add scheduler wrapper over prepare/batch/finish |
| `ds4_v100_replay.c` | Add FFN-only wavefront async worker |
| `tools/ds4-v100-run-appliance.sh` | Export/log new env flags |
| `tests/` | Add/extend CUDA smoke for FFN microbatch parity |
| `docs/sprints/VISION.md` | Record outcome |
| `logs/from-cluster/sprint177-ffn-wavefront/` | V100 evidence |

## Definition Of Done

- [x] Default path is unchanged when `DS4_V100_ASYNC_FFN_WAVEFRONT` is unset.
- [x] Layer executor can prepare FFN state, batch FFN, and finish HC output.
- [x] FFN microbatch path reuses existing batched routed-FFN/TurboMind code.
- [x] Replay worker supports `DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK=2` and `4`.
- [x] V100 build passes for affected targets.
- [x] CUDA/serving smoke proves FFN microbatch parity against the per-slot path.
- [x] Selected-token or full-scheduler smoke at 16-slot/256K preserves expected
      token `3136`.
- [x] Logs prove denser routed work, e.g. microbatch slots `2/4` and routes
      `12/24`, without `DS4_V100_ASYNC_SLOT_CHUNK` or whole-layer wavefront.
- [x] Served same-binary 16-slot/256K A/B records prompt, generated, and
      continuation tok/s separately with `16/16` token match.
- [x] Promote only if continuation tok/s is non-regressing and preferably at
      least `5%` better; otherwise keep diagnostic-only and document the result.

## Implementation Notes

- Added `ds4_v100_layer_execute_hc_prepare_ffn()` and
  `ds4_v100_layer_execute_hc_finish_ffn_batch()` so an HC layer can be split
  around the routed/shared FFN boundary.
- Added `ds4_v100_stage_scheduler_decode_hc_ffn_microbatch_layer()` as a
  default-off stage scheduler wrapper that prepares multiple slots, calls the
  existing `execute_ffn_delta_batch()`, then finishes the HC expansion.
- Added `DS4_V100_ASYNC_FFN_WAVEFRONT=1` and
  `DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK=N` in the replay per-step worker and the
  appliance launcher.
- Added `DS4_V100_ASYNC_FFN_WAVEFRONT_VERBOSE=1` for serving logs that show
  actual grouped FFN work.
- Fixed a serving-only correctness bug found during smoke: the first
  implementation used absolute request slots as FFN scratch slots. The existing
  batched shared-F8 path assumes compact scratch inputs `[0..n-1]` for its
  pointer table, so FFN microbatches now use compact scratch slots per group.

## V100 Validation

Build target:

```text
make ds4_v100_layer_execute.o ds4_v100_scheduler.o ds4_v100_replay.o \
  tools/ds4-v100-replay \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke \
  CUDA_ARCH=sm_70 -j80
```

Baseline correctness:

```text
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=926 ... expected=3136 ... ok
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=16 layers=43 tm_layers=43 ... ok
```

Direct replay smoke:

```text
ctx=32768 slots=4 active_microbatch=4 tokens=2
DS4_V100_ASYNC_FFN_WAVEFRONT=1
DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK=2
selected token: 3136
continuation_decode: 15.720289 tok/s
```

Serving smoke:

| Mode | Ctx | Slots | Tokens | Match | Generated tok/s | Continuation tok/s | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| FFN wavefront chunk 2 | 32K | 4 | 2 | 4/4 | 3.023867 | 1.511934 | logs show `slots=2 routes=12` |
| FFN wavefront chunk 4 | 32K | 4 | 2 | 4/4 | 1.634313 | 0.817156 | logs show `slots=4 routes=24` |

The chunk-2 smoke first failed before the compact-scratch fix with `2/4`
incorrect first tokens. After the fix it passed with `4/4` token match.

Same-binary served 16-slot/256K A/B:

| Mode | Generated tok/s | Continuation tok/s | Prompt tok/s | Match |
|---|---:|---:|---:|---:|
| Control | 70.943330 | 69.834841 | 19.952812 | 16/16 |
| FFN wavefront chunk 2 | 58.063198 | 57.155960 | 16.330274 | 16/16 |
| FFN wavefront chunk 4 | 35.577336 | 35.021440 | 10.006126 | 16/16 |

Artifacts:

```text
logs/from-cluster/sprint177-ffn-wavefront/
```

## Decision

Keep FFN-only wavefront batching diagnostic-only.

The implementation proves the scheduler can form denser routed FFN work inside
the production serving path without global slot chunking. However, the same
16-slot/256K fixture shows that waiting for FFN-ready groups loses more stage
overlap than the denser routed GEMM shape gains. Chunk 2 regressed continuation
throughput by about `18%`, and chunk 4 by about `50%`.

This closes the scheduler-side local batching hypothesis. The next material
work should be either:

- a true persistent/fused routed-FFN executor that removes the remaining global
  `mid_half`/FFN-boundary traffic inside one GPU; or
- a broader persistent TP/EP topology that keeps ownership across multiple
  routed layers instead of copying full hidden state back after each layer.

## Risks

- Splitting the layer executor may accidentally change HC residual ordering.
- FFN-only grouping may still wait long enough to reduce stage overlap.
- Reports/timing aggregation may undercount or double-count batched layers.
- If this regresses, scheduler-side batching is effectively exhausted and the
  next sprint should return to a true no-`mid_half` kernel boundary.
