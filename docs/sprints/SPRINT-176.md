# Sprint 176 - TP/EP Routed-FFN Span

Date: 2026-05-22
Status: Complete

## Overview

Sprint 176 moves the tensor/expert-parallel routed-FFN experiment from a
singleton layer overlay to a bounded span inside one layer-parallel stage.
Sprint 174 proved the TP2 math and one-layer scheduler hook are correct but
slower in served mode. Sprint 175 proved wrapper-level six-route fusion is also
not enough. The remaining topology question is whether peer ownership over a
small contiguous routed-FFN span can amortize the boundary setup and make the
TP/EP primitive useful in the real appliance loop.

The first target is intentionally small:

```text
DS4_V100_TP_EP_ROUTED_FFN=span
DS4_V100_TP_EP_LAYER_FIRST=3
DS4_V100_TP_EP_LAYER_COUNT=2
DS4_V100_TP_EP_PEER=3
```

This keeps all work inside stage 0, uses one owner GPU and one NVLink peer, and
reuses the existing TP2 half-mid routed primitive. It is materially different
from Sprint 174 because the scheduler now owns a TP/EP span contract rather
than a single `tp2_layer`.

## Non-Goals

- No 8-way TP/EP production topology in this sprint.
- No default promotion unless served A/B clears the throughput gate.
- No true single-kernel gate/up + down fusion.
- No MTP changes.
- No attention/shared-FFN TP in this sprint.

## Use Cases

1. A bounded TP split pack can contain adjacent routed layers for the same
   owner/peer pair.
2. The scheduler validates a TP/EP span and fails closed if any layer is missing
   TP2 bindings, crosses a stage boundary, has a different peer, or references
   different shard files.
3. Layer execution can select TP2 for any layer inside the configured span.
4. Stage/full scheduler smokes can report `tp2_layers=2`.
5. Served same-binary A/B decides whether span topology is worth expanding.

## Architecture

The stage scheduler owns one TP/EP span:

```text
stage owner GPU
  resident normal appliance weights
  TP2 owner split weights for layers [first, first + count)
  owner scratch and peer receive buffer

peer GPU
  TP2 peer split weights for the same layers
  shared peer input/route/output scratch

each layer in span
  copy hidden/routes to peer
  owner half routed FFN
  peer half routed FFN
  copy peer partial back
  sum into owner output
```

The span still pays per-layer hidden transfer because shared FFN and the next
layer require full owner-side hidden state. This sprint is a topology gate, not
a final TP implementation. If two adjacent layers are still slower, the next TP
work must change payload ownership more radically or stop.

## Implementation

### Phase 1 - Pack Bounded Layer Ranges

- Add `--layer-count N` to `tools/ds4-v100-appliance-pack`.
- Keep existing `--layer N` behavior as `first layer`, default count `1`.
- With `--skip-non-experts`, emit routed expert rows only for layers in the
  configured range.
- Add `--tp-split-only` so TP/EP overlays can carry only split TP rows rather
  than duplicate full owner-side routed expert weights. This is required to fit
  the two-layer overlay in 32 GiB V100 VRAM.

### Phase 2 - Scheduler Span Config

- Add `tp2_layer_count` to scheduler and layer-execute config.
- Parse `DS4_V100_TP_EP_LAYER_COUNT`.
- Validate `count >= 1` and `first + count <= 43`.
- Set up one TP/EP owner/peer arena pair for the intersection of the configured
  span and the current stage.
- Fail closed when any layer in the stage-local span lacks TP2 rows, has a peer
  mismatch, or uses different owner/peer shard files.

### Phase 3 - Layer Selection And Launcher

- Change `tp2_routed_enabled()` from singleton equality to span membership.
- Remove the launcher restriction that forced `LAYER_COUNT=1`.
- Keep all TP/EP flags default-off.

### Phase 4 - V100 Validation

- Build affected targets on the V100 pod.
- Generate a bounded layer-3/4 TP split pack.
- Create a combined appliance view with normal `pack-index.tsv`, normal shard
  files, and a TurboMind index that appends TP2 rows for layers 3 and 4.
- Run:
  - stage scheduler smoke expecting `tp2_layers=2`;
  - selected-token/full-scheduler smoke with expected token `3136`;
  - same-binary served 16-slot/256K A/B.

## Files Summary

| File | Change |
|---|---|
| `tools/ds4-v100-appliance-pack.cu` | Add `--layer-count` bounded TP split range and `--tp-split-only` overlay packs |
| `ds4_v100_scheduler.c` | Parse and validate TP/EP layer spans |
| `ds4_v100_layer_execute.h` | Carry TP/EP span count into layer execution |
| `ds4_v100_layer_execute.c` | Select TP2 for layers in the configured span |
| `tools/ds4-v100-run-appliance.sh` | Allow bounded `DS4_V100_TP_EP_LAYER_COUNT > 1` |
| `docs/sprints/VISION.md` | Record outcome |
| `logs/from-cluster/sprint176-tp-ep-span/` | V100 evidence |

## Definition Of Done

- [x] `--layer-count` exists and preserves existing one-layer pack behavior.
- [x] `--tp-split-only` emits a VRAM-fit overlay containing only split TP rows.
- [x] TP/EP scheduler span config is parsed and validated.
- [x] Layer execution selects TP2 for every layer inside the configured span.
- [x] Missing binding, invalid peer, and out-of-stage span errors fail closed.
- [x] Launcher accepts bounded `DS4_V100_TP_EP_LAYER_COUNT > 1`.
- [x] V100 build passes for affected targets.
- [x] V100 stage/full scheduler smoke passes with `tp2_layers=2`.
- [x] V100 selected-token smoke returns expected token `3136`.
- [x] Served 16-slot/256K A/B records prompt, generated, and continuation tok/s
      separately with `16/16` token match.
- [x] Promote only if continuation/decode tok/s improves by at least `10%`.
- [x] If correct but slower, keep diagnostic-only and pivot away from TP/EP
      overlays unless the next design eliminates per-layer hidden transfer.

## Results

Implemented and validated the two-layer TP/EP span over layers 3 and 4 on GPU0
with GPU3 as the NVLink peer.

Build on `llm/llamacpp-build-8gpu` passed:

```text
make tools/ds4-v100-appliance-pack \
  ds4_v100_layer_execute.o \
  ds4_v100_scheduler.o \
  tools/ds4-v100-replay \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke \
  CUDA_ARCH=sm_70 -j80
```

The first bounded pack attempt duplicated full routed weights and failed the
stage smoke with a GPU0 overlay arena OOM. `--tp-split-only` fixed the layout:

```text
tm_rows=8
tm_weight_bytes=6442450944
tm_scale_bytes=402653184
gpu0.weights bytes=3422552064
gpu3.weights bytes=3422552064
```

Correctness:

```text
cuda_v100_stage_scheduler_smoke: ... tm_layers=6 tp2_layers=2 ... ok
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=926 ... expected=3136 ... ok
cuda_v100_full_scheduler_smoke: stages=8 ... layers=43 tm_layers=43 ... ok
```

Served same-binary 16-slot/256K A/B, 16 requests x 64 generated tokens,
per-step async + event handoff:

| Mode | Prompt tok/s | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---:|---:|
| control | `20.110142` | `71.502728` | `70.385497` | `16/16` |
| TP2 span verbose | `18.420008` | `65.493363` | `64.470029` | `16/16` |
| TP2 span quiet | `18.644305` | `66.290863` | `65.255068` | `16/16` |

Decision: do not promote. The bounded TP/EP span is correct and now has a
proper VRAM-fit overlay pack format, but it regresses served continuation
throughput by about `7.3%` in the quiet run. The cost is not logging; the
per-layer full-hidden peer transfer/reduce boundary and duplicated layer-local
ownership still dominate. Keep this path diagnostic-only.

Next work should not expand the same two-GPU overlay pattern layer-by-layer.
Either the topology needs a persistent ownership boundary that avoids returning
full hidden state to the owner after each layer, or we should return to a true
in-GPU persistent routed-FFN executor.
