# Sprint 168 - Opt-In In-Stage Layer Wavefront

Date: 2026-05-21

## Objective

Use the Sprint 167 layer-span APIs in the replay runtime. The goal is to test
whether the per-step async pipeline can recover some routed-FFN batch density
inside each stage without reverting to whole-stage slot chunking.

## Scope

- Add an explicit diagnostic env gate:
  `DS4_V100_ASYNC_LAYER_WAVEFRONT=1`.
- Add `DS4_V100_ASYNC_LAYER_WAVEFRONT_CHUNK` to cap contiguous same-layer slot
  batches.
- Keep existing per-step behavior unchanged unless the env gate is enabled.
- Track per-slot local layer progress inside each stage worker.
- Batch only contiguous slots that are ready at the same local layer.
- Mark inter-stage readiness only after a slot finishes the stage's final
  layer.
- Validate correctness and throughput on the production TurboMind appliance at
  a short smoke shape first, then a 16-slot/256K sustained diagnostic if smoke
  passes.

## Implementation

1. Add replay helpers for stage layer ranges and env parsing.
2. Add a layer-wavefront worker path inside the existing per-step async
   pipeline worker.
3. For each stage:
   - stage 0 starts slots with token embedding plus first local layer;
   - stages 1-7 hand off a slot once the previous stage marks it done, then
     execute the first local layer;
   - subsequent layers use `decode_hc_layer_span()`;
   - completed slots record the existing stage-ready event when event handoff
     is enabled, then mark the slot done.
4. Accumulate per-slot reports enough for existing counters to remain useful.
5. Document the result in the vision and cluster logs.

## Definition of Done

- [x] Env-gated layer-wavefront worker is implemented.
- [x] Default per-step path is unchanged when the gate is unset.
- [x] `tools/ds4-v100-replay` builds on the V100 pod.
- [x] Production TurboMind replay smoke passes with the gate enabled.
- [x] A 16-slot/256K served or sustained diagnostic is captured if the smoke
      passes.
- [x] Result is recorded in `docs/sprints/VISION.md`.
- [x] Changes are committed.

## Validation

Build:

```bash
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

Result: PASS on `llm/llamacpp-build-8gpu`.

Production TurboMind smoke with layer-wavefront enabled:

- `DS4_V100_ASYNC_LAYER_WAVEFRONT=1`
- `DS4_V100_ASYNC_LAYER_WAVEFRONT_CHUNK=2`
- `ctx=32768`
- `slots=4`
- `active_microbatch=4`
- prompt `Hello`
- output token id `19923`
- text hex `48656c6c6f`
- generated tok/s `0.672019`

Clean same-prompt 16-slot/256K A/B, MTP off:

| Mode | Status | Generated tok/s | Continuation tok/s | Avg latency | Max latency |
|---|---:|---:|---:|---:|---:|
| per-step event handoff control | 16/16 | `32.906564` | `30.849903` | `6690.311 ms` | `7770.703 ms` |
| layer-wavefront chunk 2 | 16/16 | `26.126248` | `24.493358` | `7514.939 ms` | `9792.627 ms` |
| layer-wavefront chunk 4 | 16/16 | `19.175887` | `17.977394` | `11613.848 ms` | `13349.293 ms` |

Full cluster output is recorded in
`logs/from-cluster/sprint168-layer-wavefront/summary.log`.

## Decision

Do not promote in-stage layer-wavefront scheduling. It is correct and remains
available as an explicit diagnostic, but it loses materially to the existing
per-step event-handoff baseline at the practical 16-slot/256K tier. Chunk 4 is
worse than chunk 2, so simply increasing same-layer slot batching is not the
missing throughput lever.

## Decision Gate

If this path does not materially improve continuation/decode throughput, stop
trying to recover density through layer-parallel scheduling. The next sprint
should move to a broader persistent TP/EP scheduler boundary or a DS4-specific
persistent routed-FFN executor.
