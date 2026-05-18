---
sprint: 014
title: V100 Real Pack-Index Layer Descriptor Gate
seed: continue DS4 V100 appliance loop after bounded MXFP4 MoE selected token
date: 2026-05-18
---

# SPRINT-014 Intent

## Seed Prompt

Continue `$sprint-plan` and `$sprint-execute` toward the DS4 V100 appliance
vision. Sprint 013 proved bounded synthetic MXFP4 MoE and selected-token
composition. The remaining readiness gap is binding those kernel surfaces to
real pack-index layer descriptors and the layer scheduler.

## Orientation Summary

- Source residency, F8 projection, attention/compressor, BF16 output-head, and
  MXFP4 routed MoE primitives now exist as V100 smokes.
- The appliance gate passes implemented checks but reports `ready=false`
  because the path is not bound to real pack-index descriptors.
- `ds4_pack.[ch]` already parses `pack-index.tsv` rows with semantic id,
  source dtype, runtime layout, owning GPU, layer id, kernel family, shard file,
  shard offset, and byte length.
- `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv` is a real-model pack-index
  artifact suitable for local descriptor validation.
- The first useful runtime integration artifact is a fail-closed layer
  descriptor gate that verifies the exact tensors a real layer needs before
  compute code consumes them.

## Vision Context

The North Star remains a narrow DS4 V100 appliance that runs the
high-intelligence source quantized model from pure device-resident packs. Sprint
014 should close the gap between synthetic kernel fixtures and real model
descriptors without prematurely unlocking serving.

## Relevant Codebase Areas

| Area | Role |
|---|---|
| `ds4_pack.[ch]` | Pack-index parser and lookup API |
| `tools/ds4-v100-gate.sh` | Appliance readiness gate |
| `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv` | Real-model pack-index fixture |
| `tools/ds4-v100-plan.c` | Tensor family naming and layer ownership assumptions |
| `docs/architecture/DS4-V100-LAYOUT.md` | Required layer tensor families |

## Constraints

- Do not load or execute full model weights in this sprint unless the descriptor
  gate first passes.
- Do not unlock source-layout serving.
- Descriptor validation must fail closed on missing tensors, wrong dtype,
  wrong layout, wrong kernel family, wrong layer id, wrong owning GPU, or
  invalid shard file.
- The result must be runnable locally against the committed pack-index fixture
  and on the cluster against a generated or existing pack index.

## Proposed Sprint Shape

1. Add `tools/ds4-v100-layer-descriptor-gate`, a pack-index validator for a
   real source-layout layer.
2. Validate required descriptors for the default ratio-4 layer 2, including
   attention, compressor, indexer, router, routed MXFP4 experts, shared F8
   experts, HC controls, and output head.
3. Extend the appliance gate to optionally run descriptor validation when a
   pack index path is supplied.
4. Archive local and V100/cluster logs.

## Success Criteria

- The descriptor gate passes on the committed real pack-index fixture for
  layer 2 and prints a stable descriptor report.
- Negative checks fail closed on an incomplete synthetic index.
- The appliance gate can run descriptor validation with `--pack-index FILE`.
- The sprint report identifies the next integration step from descriptors to
  real layer compute.

## Verification Strategy

- Local build and run of the descriptor gate against
  `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv`.
- Local negative synthetic index test.
- Cluster run of the descriptor gate and full appliance gate with
  `--pack-index`.
- `git diff --check`.

## Uncertainty Assessment

| Area | Risk | Notes |
|---|---|---|
| Correctness | Low | Descriptor validation uses existing pack parser and exact tensor names |
| Scope | Low | This sprint does not execute real layer math |
| Architecture | Medium | The descriptor contract must be strict enough for later runtime use |
| Performance | Low | This is validation/plumbing, not a hot path |

## Open Questions

- Should the next sprint consume this descriptor report directly or turn it
  into C structs owned by the V100 execution context?
- Should descriptor validation expand to layer 3 and layer 42 before real
  compute integration?
