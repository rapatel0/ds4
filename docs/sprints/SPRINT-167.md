# Sprint 167 - Layer-Span Scheduler Primitive

Date: 2026-05-21

## Objective

Create the missing scheduler primitive for in-stage layer wavefront execution.
Recent serving experiments showed that queue-level slot coalescing is the wrong
place to recover routed-FFN density at the 16-slot/256K tier: whole-stage
chunking and ready-window chunking both expose wider kernels but lose too much
stage overlap. The next useful boundary is inside a stage: let the runtime
execute a bounded layer range for a slot span, so a later replay scheduler can
batch slots that are ready at the same local layer without forcing a whole
stage barrier.

## Scope

- Add public scheduler APIs for bounded layer ranges:
  - token-embedding stage seed plus `[first_layer,last_layer]`;
  - HC-input stage execution over `[first_layer,last_layer]`.
- Preserve all existing full-stage APIs and defaults.
- Reuse the current layer-batch path when `n_slots > 1`.
- Report the actual executed layer range in
  `ds4_v100_stage_scheduler_report`.
- Add CUDA smoke coverage proving segmented execution matches full-stage
  execution for a two-slot/two-stage path.
- Do not change HTTP serving defaults in this sprint.

## Implementation

1. Refactor `ds4_v100_stage_scheduler_decode_hc_slot_span()` through an
   internal helper that accepts an explicit layer range.
2. Add `ds4_v100_stage_scheduler_decode_hc_layer_span()`.
3. Add `ds4_v100_stage_scheduler_decode_token_layer_span()`, which seeds token
   embeddings for the requested slot span and then executes the requested
   layer range.
4. Add `tests/cuda_v100_stage_layer_span_smoke.c`:
   - run the existing full-stage batch path for two slots across stages 0 and 1;
   - run stage 0 and stage 1 as multiple bounded layer spans;
   - compare final HC slots against the full-stage reference;
   - assert report layer ranges and layer counts.
5. Add Makefile build/clean entries for the new smoke.

## Validation

Build locally where possible:

```bash
make tests/cuda_v100_stage_layer_span_smoke
```

Validate on the V100 pod:

```bash
make -j80 CUDA_ARCH=sm_70 tests/cuda_v100_stage_layer_span_smoke
tests/cuda_v100_stage_layer_span_smoke \
  --index /workspace/ds4-appliance-full-tm-gated-s127/pack-index.tsv \
  --turbomind-index /workspace/ds4-appliance-full-tm-gated-s127/turbomind-pack-index.tsv \
  --shard-dir /workspace/ds4-appliance-full-tm-gated-s127 \
  --model /models/DSv4-Flash-256e-fixed.gguf
```

The production TurboMind appliance needs both the base pack index and the
TurboMind pack index; using only `pack-index.tsv` fails at open because the
production pack intentionally omits raw expert descriptors.

Validation result:

- Build passed on `llm/llamacpp-build-8gpu`.
- Production TurboMind segmented layer-span smoke passed:
  `stage0=[0,5] stage1=[6,11]`, `max_abs_slot0=0.01612854`,
  `max_abs_slot1=0.0221862793`, threshold `0.03`.
- Full-vs-full repeat diagnostic passed with the same drift envelope:
  `max_abs_slot0=0.016078949`, `max_abs_slot1=0.0161018372`, threshold `0.03`.
- Normal replay regression smoke passed on the production appliance:
  prompt `Hello`, token id `19923`, text hex `48656c6c6f`.
- Full cluster output is recorded in
  `logs/from-cluster/sprint167-layer-span/summary.log`.

## Definition of Done

- [x] Layer-span APIs are declared and implemented.
- [x] Existing full-stage APIs still build and use the same semantics.
- [x] New layer-span smoke builds.
- [x] New layer-span smoke passes on V100.
- [x] Result is recorded in `docs/sprints/VISION.md`.
- [x] Changes are committed.

## Decision Gate

If the primitive is correct, the next sprint should wire an opt-in replay
diagnostic that keeps per-slot `(stage, local_layer)` progress and batches only
slots that are ready at the same local layer. If this still fails to improve
the practical 16-slot/256K decode tier, stop spending time on layer-parallel
slot scheduling and move to the broader persistent TP/EP scheduler boundary.
