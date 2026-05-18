# SPRINT-014 Follow-Ups

These are not blockers for the Sprint 014 `SHIP` verdict, but they block a
deployable appliance.

## Runtime Descriptor Table

- **What:** Convert descriptor-gate validation into runtime-owned descriptor
  structs attached to the V100 context.
- **Why:** Sprint 014 proves descriptor presence and policy, but compute code
  still needs stable typed bindings instead of repeated string lookup.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 015.
- **Files:** `ds4_pack.*`, V100 context/scheduler files, new descriptor module.

## Descriptor-Bound Layer Compute

- **What:** Run at least one real layer or bounded selected-token path from
  validated pack descriptors and resident shard bytes.
- **Why:** The appliance remains non-serving until real model bytes flow through
  the bounded attention, MoE, and logits primitives.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 015.
- **Files:** `ds4_cuda.cu`, `ds4_gpu.h`, scheduler/context files, CUDA smokes.

## Layer-Class Coverage

- **What:** Extend descriptor validation beyond layer 2 to representative
  layer classes: SWA-only layers 0-1, ratio-128 odd layers, later ratio-4
  layers, and final output ownership.
- **Why:** Layer 2 covers the richest early ratio-4/indexer/hash-router case,
  but not all model layer variants.
- **Severity:** Important.
- **Suggested sprint:** Sprint 015 or Sprint 016.
- **Files:** `tools/ds4-v100-layer-descriptor-gate.c`, appliance gate.

## Shared Expert Execution In Real Layer

- **What:** Include source-F8 shared expert gate/up/down composition in the
  first descriptor-bound layer compute path.
- **Why:** Sprint 013 covered routed MXFP4 experts synthetically; DS4 layer
  output also includes the shared expert path.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 015.
- **Files:** `ds4_cuda.cu`, `tests/cuda_v100_*`, scheduler/context files.

## Readiness Policy

- **What:** Move the appliance gate readiness string from a static missing-list
  toward feature-derived readiness as real layer compute and serving land.
- **Why:** The gate should keep reporting not-ready, but the reasons should
  track implemented milestones precisely.
- **Severity:** Important.
- **Suggested sprint:** Sprint 015+.
- **Files:** `tools/ds4-v100-gate.sh`, deployment scripts.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Runtime descriptor table | Critical | Sprint 015 | pack/context/scheduler |
| Descriptor-bound layer compute | Critical | Sprint 015 | CUDA/context/scheduler/tests |
| Layer-class coverage | Important | Sprint 015-016 | descriptor gate/gate script |
| Shared expert execution in real layer | Critical | Sprint 015 | CUDA/tests/scheduler |
| Readiness policy | Important | Sprint 015+ | gate/deployment scripts |
