# Sprint 074: Async Peer Handoff Probe

## Status

Complete.

## Overview

Sprint 073 proved that improving host condition-variable scheduling is not
enough. The per-step async pipeline remains the best practical default at
`8.649395` generated tok/s for 1M/4 slots, while mailbox persistent workers
reach `8.053284`. The remaining visible costs include peer handoff and device
sync around each stage/slot lane.

Sprint 074 should move one level lower: add an opt-in async peer-copy handoff
for HC relay and measure it against the current blocking handoff. The first
probe should keep correctness and stream ordering conservative: queue the
peer copy on the destination device's default stream, then launch destination
decode on the same default stream, relying on stream order before the existing
end-of-stage device synchronize.

## Goals

1. Add a GPU API for queued device/device and peer tensor copy.
2. Add an opt-in scheduler handoff mode that uses the queued copy for HC relay.
3. Keep the existing blocking copy path as the default and as the A/B control.
4. Wire the opt-in through replay and benchmark tooling.
5. Preserve selected-token correctness and existing async pipeline modes.
6. Run V100 A/B evidence for:
   - blocking handoff + per-step async;
   - async handoff + per-step async;
   - optionally blocking/async handoff under mailbox if time permits.
7. Decide whether async handoff should become the appliance default or remain
   diagnostic.

## Non-Goals

- Rewriting attention, FFN, routed expert, or output-head kernels.
- Introducing custom CUDA streams per stage.
- Removing the existing per-stage device synchronize.
- Changing MTP behavior.
- Changing appliance `auto` unless the V100 matrix is clearly positive.

## Implementation

1. Extend `ds4_gpu.h` / `ds4_cuda.cu` with a queued tensor copy primitive:
   - same-device: `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, 0)`;
   - cross-device: `cudaSetDevice(dst->device)` then
     `cudaMemcpyPeerAsync(..., 0)`;
   - preserve bounds/device checks from `ds4_gpu_tensor_copy`.
2. Extend scheduler handoff APIs in `ds4_v100_scheduler.h/.c`:
   - add `ds4_v100_stage_scheduler_handoff_slot_span_async`;
   - add batch/single wrappers only if they keep call sites simple;
   - keep the current blocking APIs unchanged.
3. Extend replay options in `ds4_v100_replay.h/.c`:
   - `bool async_handoff`;
   - use async handoff inside serial, wavefront, per-step, persistent, and
     mailbox paths only when explicitly enabled.
4. Wire CLI and benchmark flags:
   - `tools/ds4-v100-replay --async-handoff`;
   - `tools/ds4-v100-sustained-decode-bench.sh --async-handoff`;
   - status JSON reports `async_handoff=true|false`.
5. Preserve appliance launcher behavior unless evidence supports changing it.
   Add env support only as opt-in, e.g. `DS4_V100_ASYNC_HANDOFF=1`.
6. Add Sprint 074 report and update `docs/sprints/VISION.md`.

## Definition of Done

- [x] Local compile passes for changed C files.
- [x] Shell syntax checks pass for changed scripts.
- [x] `git diff --check` passes.
- [x] Invalid/new CLI flags behave as expected.
- [x] V100 build passes for `tools/ds4-v100-replay` and scheduler smokes.
- [x] V100 wavefront and selected-token smokes pass.
- [x] A short mailbox or per-step sustained smoke returns token hex `3136`
  with `async_handoff=true`.
- [x] V100 A/B matrix records blocking vs async handoff at 1M/2 and 1M/4
  slots using per-step async.
- [x] Sprint report records timing deltas, handoff counters, and the default
  decision.
- [x] Vision document is updated.

## Outcome

`SHIP`, but keep async handoff opt-in.

The runtime now has a queued tensor-copy primitive and an opt-in
`--async-handoff` path for HC relay. V100 correctness stayed green, and the
per-step A/B matrix showed a small positive result:

| Handoff | 1M/2 generated tok/s | 1M/4 generated tok/s | Decision |
|---|---:|---:|---|
| blocking | `5.553165` | `8.605744` | current default |
| async | `5.591514` | `8.738546` | opt-in, below default threshold |

Async handoff improved 1M/4 generated tok/s by `1.543%`, below the `3%`
default-change threshold. Keep it available as an opt-in probe, but do not
change appliance `auto`. The next sprint should target custom CUDA stream/event
handoff or kernel-side work.

Artifacts:

- `logs/from-cluster/sprint074-async-handoff-smoke`
- `logs/from-cluster/sprint074-perstep-blocking`
- `logs/from-cluster/sprint074-perstep-async-handoff`
- `logs/from-cluster/sprint074-handoff-comparison`

## Decision Rule

- If async handoff improves 1M/4-slot per-step generated tok/s by at least
  `3%` without regressing 1M/2 slots, make it the practical appliance default.
- If it is within `+/-2%`, keep it opt-in and use the evidence to decide
  between custom streams/events and kernel-side work.
- If it regresses by more than `2%`, remove or quarantine the mode unless the
  timing counters reveal a fixable ordering issue.

## Risks

- Default-stream semantics may serialize more than expected or vary by CUDA
  runtime behavior.
- Async peer copies could expose missing ordering assumptions if destination
  decode launches on a different stream in some future code path.
- Handoff is visible but not the only large cost; a correct async copy probe may
  still be throughput-neutral.
- If the probe is neutral, the next sprint should pivot to kernel-side work
  rather than more host scheduling.

## Security

No new serving surface. Async handoff is an opt-in internal execution mode on
the existing loopback appliance path.
