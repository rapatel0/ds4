# TEMP Status Report 001

Date: 2026-05-21

## Topline

- Latest committed sprint: `b85574b` (`scheduler: add layer-span decode primitive`).
- Current working sprint: Sprint 168, opt-in in-stage layer-wavefront replay diagnostic.
- Worktree currently has uncommitted Sprint 168 changes:
  - `ds4_v100_replay.c`
  - `docs/sprints/SPRINT-168.md`
- GPUs were clear after the latest diagnostic.

## Current Best Known Throughput

The best clean practical 16-slot/256K comparison remains the Sprint 166
per-step event-handoff baseline:

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Notes |
|---|---:|---:|---:|---:|---|
| per-step event handoff control | 256K | 16 | `26.346605` | `24.699942` | Production TurboMind appliance, MTP off |
| ready-window coalescing | 256K | 16 | `25.027658` | `23.463429` | Correct but slower |

Sprint 168 layer-wavefront has now been tested in a clean same-prompt A/B with
extra request headroom:

| Mode | Context | Slots | Status | Generated tok/s | Continuation tok/s |
|---|---:|---:|---:|---:|---:|
| per-step event handoff control | 256K | 16 | `16/16` OK | `32.906564` | `30.849903` |
| layer-wavefront chunk=2 | 256K | 16 | `16/16` OK | `26.126248` | `24.493358` |
| layer-wavefront chunk=4 | 256K | 16 | `16/16` OK | `19.175887` | `17.977394` |

Interpretation: layer-wavefront is correct but materially slower. Do not
promote it.

## Sprint 167 Completed

Sprint 167 added bounded layer-span scheduler APIs:

- `ds4_v100_stage_scheduler_decode_token_layer_span()`
- `ds4_v100_stage_scheduler_decode_hc_layer_span()`

Validation on the V100 pod:

- New smoke built: `tests/cuda_v100_stage_layer_span_smoke`.
- Production TurboMind segmented layer-span smoke passed:
  - `stage0=[0,5]`
  - `stage1=[6,11]`
  - `max_abs_slot0=0.01612854`
  - `max_abs_slot1=0.0221862793`
  - threshold `0.03`
- Full-vs-full repeat diagnostic passed in the same drift envelope:
  - `max_abs_slot0=0.016078949`
  - `max_abs_slot1=0.0161018372`
- Normal replay regression smoke passed:
  - prompt `Hello`
  - token id `19923`
  - text hex `48656c6c6f`

Cluster log: `logs/from-cluster/sprint167-layer-span/summary.log`.

## Sprint 168 Completed Locally, Pending Commit

Implemented so far:

- Env gate: `DS4_V100_ASYNC_LAYER_WAVEFRONT=1`.
- Chunk cap: `DS4_V100_ASYNC_LAYER_WAVEFRONT_CHUNK`.
- Per-stage worker tracks each slot's next local layer.
- It batches contiguous slots that are currently ready at the same layer.
- It only marks a slot ready for the next stage after the final local layer.
- Existing per-step path is unchanged when the env gate is unset.

Validation so far:

- Local syntax check passed for `ds4_v100_replay.c`.
- V100 build passed for `tools/ds4-v100-replay`.
- 4-slot/32K production-appliance smoke passed with layer-wavefront enabled:
  - prompt `Hello`
  - token id `19923`
  - text hex `48656c6c6f`
  - generated tok/s `0.672019`

Clean 16-slot/256K diagnostic:

- Server: production TurboMind appliance, MTP off.
- Env:
  - `DS4_V100_ASYNC_LAYER_WAVEFRONT=1`
  - `DS4_V100_ASYNC_LAYER_WAVEFRONT_CHUNK=2`
  - `DS4_V100_TURBOMIND_LIB=./build/turbomind-v100-s127/libggml-turbomind.so`
  - `DS4_V100_TURBOMIND_GATED_SILU=1`
  - `DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1`
- Control:
  - `16/16` successful responses
  - generated tok/s `32.906564`
  - continuation tok/s `30.849903`
  - avg latency `6690.311 ms`
- Layer-wavefront chunk 2:
  - `16/16` successful responses
  - generated tok/s `26.126248`
  - continuation tok/s `24.493358`
  - avg latency `7514.939 ms`
- Layer-wavefront chunk 4:
  - `16/16` successful responses
  - generated tok/s `19.175887`
  - continuation tok/s `17.977394`
  - avg latency `11613.848 ms`

## What Has Been Tried Recently

- Ready-window slot coalescing: correct, slower, default off.
- Layer-span scheduler primitive: correct, committed.
- Layer-wavefront replay worker: builds and smokes, first throughput run is
  neutral/slightly worse and not clean enough for a decision.
- TP2 overlay: correct as a primitive, slower as a synchronous one-layer
  overlay; future TP work should be persistent TP/EP, not per-layer copies.
- Host stream-per-expert software pipeline: active but slower.
- CUDA graph wrapper: blocked by legacy default-stream capture.
- Fixed-shape route executors: correct but not material in served path.

## Next Best Step

Commit Sprint 168 as a default-off diagnostic, then stop the layer-parallel
scheduling line. The next material path should be persistent TP/EP or a
persistent routed-FFN executor boundary.
