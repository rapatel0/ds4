# SPRINT-017 Intent: V100 Scheduler-Owned Layer State Gate

## Seed Prompt

Continue sprint-plan and sprint-execute loops until the DS4 V100 appliance
vision is realized, with actual implementation each sprint.

## Orientation Summary

- Sprint 016 shipped descriptor-bound router-selected FFN compute on the V100
  pod, but the composition still lives inside a standalone CUDA smoke.
- The readiness gate still reports `ready=false` with remaining gaps:
  full layer scheduler, attention/residual/norm, real-model selected token,
  public serving, MTP, and throughput benchmark.
- `docs/architecture/DS4-V100-LAYOUT.md` remains the architecture anchor. The
  baseline is 8-GPU layer sharding, source-faithful FP8/MXFP4/F32/I32 packs, no
  persistent dequantized weight copies, and F16 KV first.
- Relevant code is `ds4_v100_context.*`, `ds4_gpu.h`, `ds4_cuda.cu`,
  `tests/cuda_v100_descriptor_bound_ffn_smoke.c`,
  `tests/v100_layer_binding_smoke.c`, and `tools/ds4-v100-gate.sh`.
- Actionable Sprint 016 follow-up: introduce scheduler-owned layer state that
  carries bindings, source row views, route metadata, scratch ownership
  contracts, and arena references.

## Goal

Move descriptor-bound router/FFN execution from test-local binding code into a
reusable scheduler-owned layer state API, then prove the descriptor-bound FFN
smoke runs through that API on local builds and the V100 pod.

## Constraints

- Do not unlock serving.
- Do not claim full layer execution; attention/residual/norm remain deferred.
- Do not persistently dequantize source weights.
- Keep the API fail-closed on wrong layer, dtype, shape, owner GPU, or arena
  range.
- Preserve the Sprint 016 router-selected FFN correctness evidence.

## Success Criteria

- A public V100 layer-state module binds layer-local descriptors once.
- The layer state exposes router, shared expert, selected routed expert, and
  arena-span/source-row-view helpers.
- A CPU/local layer-state smoke validates layer-2 state from the real pack
  index.
- `tests/cuda_v100_descriptor_bound_ffn_smoke` uses the layer-state API instead
  of reimplementing descriptor shape and route view logic locally.
- The appliance gate includes and passes a `layer_state` check on the V100 pod.

## Verification Strategy

- Local:
  - `make tests/v100_layer_state_smoke`
  - `make tests/cuda_v100_descriptor_bound_ffn_smoke.o`
  - `bash -n tools/ds4-v100-gate.sh`
  - `git diff --check`
- Cluster:
  - Build `tests/v100_layer_state_smoke`.
  - Run the router-enabled descriptor-bound FFN smoke on layer 2.
  - Run full `tools/ds4-v100-gate.sh --build --pack-index ...`.

## Uncertainty

- Correctness: Low. Sprint 016 already proved the math path; this sprint
  mostly moves descriptor ownership into a reusable state object.
- Scope: Medium. The refactor must avoid turning into full attention/layer
  execution.
- Architecture: Medium. The layer-state API should be useful for attention and
  selected-token work without overcommitting to the final server scheduler.

## Open Questions

1. Should Sprint 018 extend this state to attention/residual/norm or push first
   to selected-token logits?
2. Should production arena reuse be pulled into Sprint 018, or wait until the
   layer output path is coherent?
