# Sprint 209 Intent - Bounded TP8 One-Layer Prototype

Date: 2026-05-23

## Seed Prompt

Sprint 208 cleared the first TP8 investigation gate. Plan the next sprint: a
bounded one-layer TP8 prototype in completely separate TP-only files. Do not
abstract the PP/layer scheduler and do not add TP modes to the current
production scheduler.

## Orientation Summary

- Sprint 208 showed 32-slot/256K `PP1/TP8` fits only with sharded KV:
  `26.84 GiB` worst GPU with F8 KV sharding versus `50.63 GiB` with replicated
  KV.
- TP8 recursive-doubling FP16 hidden reductions passed on all eight V100s:
  `0.322599 ms` at 32 tokens, `0.372364 ms` at 64, and `0.436299 ms` at 128.
- The 43-layer, two-reduction/layer TP8 boundary measured `29.381000 ms` at
  32 tokens, `32.605223 ms` at 64, and `37.994584 ms` at 128.
- The evidence clears a topology investigation gate but does not prove TP8
  serving. The next question is whether a bounded layer with real resident work
  stays inside the TP8 boundary well enough to justify a runtime branch.
- The user explicitly rejected generic scheduler implementation. TP work must
  stay in new TP-only files.

## Relevant Codebase Areas

- `tools/ds4-v100-plan-tp.c`: TP memory/topology envelope.
- `tools/ds4-v100-tp8-collective-smoke.cu`: TP8 collective primitive evidence.
- `tools/ds4-v100-tp8-layer-proxy.cu`: TP8 resident-boundary proxy.
- `kernels/turbomind/ggml-turbomind/test_tp_split_4gpu.cpp`: reference for
  TP split FFN compute shape, but not a target to mutate into production.
- `kernels/turbomind/ggml-turbomind/test_tp4_resident_layer_slice.cu`:
  reference for resident layer-slice correctness and pitfalls.
- `ds4_v100_scheduler.*`: production PP scheduler, baseline/control only. Do
  not modify for TP.

## Constraints

- No generic scheduler.
- No PP scheduler modifications.
- No launcher default changes.
- No full-model serving claim.
- New TP code should live under new files such as `tools/ds4-v100-tp8-*`,
  `ds4_v100_tp8_*`, or `tests/tp8_*`.
- Reuse only low-level helpers, source format readers, TurboMind kernels, and
  measurement patterns.
- Preserve existing dirty Sprint 207 files; do not commit them with Sprint 209.

## Success Criteria

- A bounded TP8 one-layer executable exists in new TP-only files.
- It runs on all eight V100s and keeps hidden state resident across the layer
  boundary.
- It includes a sharded-KV descriptor/allocation smoke for a DS4 layer at
  128K/256K planning dimensions without allocating replicated KV.
- It includes at least one real compute component inside the TP8 boundary:
  either a synthetic dense projection plus routed-like split/reduce, or a
  TurboMind-backed routed FFN shard if feasible.
- It reports timing for 32, 64, and 128 token shapes and compares against the
  Sprint 208 boundary-only results.
- It records whether Sprint 210 should continue to a TP8 runtime branch, fall
  back to PP2/TP4, or pause TP work.

## Verification Strategy

- Build locally where possible; CUDA targets must build on the V100 pod with
  `CUDA_ARCH=sm_70`.
- Run on all eight GPUs with 32, 64, and 128 token payloads.
- Validate cross-device hidden equality after reductions.
- Validate KV shard byte counts and per-GPU ownership without allocating
  replicated KV.
- Store logs under `logs/from-cluster/sprint209-tp8-layer/`.

## Uncertainty Assessment

- Correctness: Medium-high. The prototype is bounded, but sharded KV ownership
  and resident all-GPU execution are new.
- Scope: Medium. It must stop at one bounded layer, not a runtime scheduler.
- Architecture: High. It creates the first concrete TP-only layer artifact and
  must not fight PP abstractions.

## Open Questions

- Should the first one-layer prototype use a synthetic dense/routed compute
  body or directly adapt TurboMind TP split helpers to TP8?
- What is the minimum KV shard descriptor needed to prove ownership without
  implementing full attention?
- Should the prototype model two reductions/layer or include additional
  reduction points for dense attention/output paths?

## Vision Context

The vision now points to practical serving through either larger fused kernels,
MTP, or a topology shift. Sprint 208 made TP8 worth one more bounded gate. This
sprint is that gate: prove whether a single TP8 layer with sharded ownership and
resident compute is plausible before any TP runtime branch is started.
