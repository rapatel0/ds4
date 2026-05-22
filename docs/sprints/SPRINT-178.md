# Sprint 178 - TP/EP Parallel Halves Gate

Date: 2026-05-22
Status: Completed

## Overview

Sprint 177 ruled out scheduler-local FFN batching: it formed denser routed
work, but lost too much stage overlap. The next unresolved topology question is
whether the existing two-way TP/EP routed-FFN boundary is slow because TP is
wrong for this appliance, or because the current overlay still serializes too
much of the owner/peer execution.

Sprint 178 adds a default-off gate:

```text
DS4_V100_TP_EP_PARALLEL_HALVES=1
```

When the existing `DS4_V100_TP_EP_ROUTED_FFN` path is active, this gate launches
the owner half and peer half from separate host threads after peer input/route
copy is queued. The result is still diagnostic-only and still uses the current
copy-back/reduce boundary. The goal is to isolate the next question: can the
two half-width routed FFN computations overlap enough on the real V100 node to
justify a larger persistent TP/EP ownership sprint?

## Non-Goals

- No default promotion unless served A/B is correct and faster.
- No 8-way TP rewrite.
- No new TP pack format.
- No peer ownership across attention or residual state yet.
- No changes to the default path when the gate is unset.

## Architecture

Current TP/EP overlay:

```text
copy x/routes/weights -> peer
owner half routed FFN
peer half routed FFN
copy peer partial -> owner
sum owner + peer
```

Sprint 178 gated path:

```text
copy x/routes/weights -> peer
launch peer half routed FFN on peer host thread
run owner half routed FFN on caller thread
join peer thread
copy peer partial -> owner
sum owner + peer
```

This does not remove the boundary cost. It only tests whether owner/peer compute
overlap is available in the current runtime and how much it changes the served
16-slot/256K TP/EP overlay.

## Implementation

### Phase 1 - Runtime Gate

- Add `DS4_V100_TP_EP_PARALLEL_HALVES`.
- Keep default off.
- Accept the flag in `tools/ds4-v100-run-appliance.sh`.
- Export and record the flag in `startup.env`.

### Phase 2 - Layer Executor

- Add a small pthread worker for the owner/peer routed half call.
- In `execute_turbomind_tp2_routed()`, after input copies:
  - if gate off, keep existing sequential owner then peer calls;
  - if gate on, run peer on a worker thread and owner on the caller thread;
  - join and propagate both errors.
- Keep timing compatible with existing reports. In verbose mode, report that the
  parallel-halves path was used and use the combined launch/join time for the
  routed half bucket.

### Phase 3 - Validation

- Local object build.
- V100 CUDA build for scheduler/replay/smoke targets.
- TP/EP stage smoke with `DS4_V100_TP_EP_PARALLEL_HALVES=1`.
- Selected-token or full scheduler smoke returns expected token `3136`.
- Served same-binary 16-slot/256K A/B:
  - control: no TP/EP;
  - TP/EP sequential overlay if needed for baseline;
  - TP/EP parallel-halves candidate.

## Files Summary

| File | Change |
|---|---|
| `ds4_v100_layer_execute.c` | Add guarded owner/peer parallel TP2 execution |
| `tools/ds4-v100-run-appliance.sh` | Validate/export/log `DS4_V100_TP_EP_PARALLEL_HALVES` |
| `deploy/v100/ds4-v100-appliance.env.example` | Document default-off flag |
| `docs/sprints/VISION.md` | Record outcome |
| `logs/from-cluster/sprint178-tp-ep-parallel-halves/` | V100 evidence |

## Definition Of Done

- [x] Default path unchanged when `DS4_V100_TP_EP_PARALLEL_HALVES` is unset.
- [x] Launcher validates, exports, and logs the new flag.
- [x] V100 build passes for affected targets.
- [x] TP/EP smoke proves the parallel-halves candidate executes and preserves
      routed output correctness.
- [x] Full scheduler or selected-token smoke preserves expected token `3136`.
- [x] Served 16-slot/256K A/B records prompt, generated, and continuation tok/s
      separately with `16/16` token match.
- [x] Promote only if continuation tok/s improves; otherwise keep diagnostic
      and use the result to choose fused local executor versus broader TP/EP.

## Risks

- CUDA context use from an extra host thread may expose thread-safety issues in
  the current arena/TurboMind wrapper.
- If the underlying TurboMind wrapper performs hidden synchronizing work, host
  parallelism may not improve anything.
- Even if half compute overlaps, copy-back/reduce may still dominate enough to
  keep the TP/EP overlay slower than default.

## Results

Implemented `DS4_V100_TP_EP_PARALLEL_HALVES=1` as a guarded host-thread
parallel owner/peer path inside the existing TP2 routed-FFN overlay. The
launcher now validates, exports, and records the flag in `startup.env`.

V100 build on `llm/llamacpp-build-8gpu` passed:

```text
make ds4_v100_layer_execute.o ds4_v100_scheduler.o ds4_v100_replay.o \
  tools/ds4-v100-replay \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke \
  CUDA_ARCH=sm_70 -j80
```

Correctness:

```text
cuda_v100_stage_scheduler_smoke: ... tm_layers=6 tp2_layers=2 ... ok
cuda_v100_full_scheduler_smoke: stages=8 ... layers=43 tm_layers=43 ... ok
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=926 ... expected=3136 ... ok
```

The stage smoke also confirmed the new path executes:

```text
ds4: TP/EP routed boundary layer=3 ... async_input=1 parallel_halves=1 ...
ds4: TP/EP routed boundary layer=4 ... async_input=1 parallel_halves=1 ...
```

Served same-binary 16-slot/256K A/B, 16 requests x 64 generated tokens,
per-step async + event handoff:

| Mode | Prompt tok/s | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---:|---:|
| no TP/EP control | `20.053070` | `71.299803` | `70.185744` | `16/16` |
| TP2 span sequential | `18.624110` | `66.219059` | `65.184386` | `16/16` |
| TP2 span parallel halves | `18.670979` | `66.385703` | `65.348426` | `16/16` |

Parallel halves improved the existing TP2 span by only about `0.25%`, while
remaining about `6.9%` slower than the no-TP production path.

Decision: keep `DS4_V100_TP_EP_PARALLEL_HALVES` default-off and diagnostic-only.
Host-thread overlap is not the missing lever for the current copy-back/reduce
overlay. The TP result continues to argue against extending this per-layer
overlay pattern; the next material work should either design persistent TP/EP
ownership that does not return full hidden state after each layer, or implement
a true persistent/fused routed-FFN executor that removes the wrapper-level
intermediate traffic.

Evidence: `logs/from-cluster/sprint178-tp-ep-parallel-halves/`.
