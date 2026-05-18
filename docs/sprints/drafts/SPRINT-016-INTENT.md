---
sprint: 016
title: V100 Descriptor-Bound Router FFN Gate
date: 2026-05-18
seed: continue sprint-plan and sprint-execute loop until DS4 V100 appliance vision is realized
vision: ../VISION.md
architecture: ../../architecture/DS4-V100-LAYOUT.md
---

# Sprint 016 Intent: V100 Descriptor-Bound Router FFN Gate

## Orientation Summary

- Sprint 015 proved layer-2 descriptor-bound FFN compute from real source GGUF
  bytes at real pack offsets, but it used a fixed expert.
- Layer-2 has hash-router metadata (`ffn_gate_tid2eid`) and a source-F32 router
  projection (`ffn_gate_inp.weight`) in the real pack index.
- Existing CUDA router selection can consume logits and a model-map hash table,
  but there is no arena-backed source-F32 matmul for `ffn_gate_inp.weight`.
- The next useful readiness step is model-selected routed experts, not
  attention yet.

## Success Criteria

- Add a V100 arena source-F32 matrix-vector primitive for descriptor-bound
  router projection.
- Extend descriptor-bound FFN validation to load `ffn_gate_inp.weight` and
  `ffn_gate_tid2eid` from real source GGUF offsets.
- Compute router logits from deterministic hidden input on GPU and CPU.
- Select experts/weights through the real hash-router path.
- Execute all selected routed MXFP4 experts plus the shared F8 expert path and
  compare summed output against CPU source-format references.
- Keep the appliance gate passing and `ready=false`.

## Verification Strategy

- Local:
  - Build the new/changed objects.
  - Run layer binding smoke.
  - `bash -n tools/ds4-v100-gate.sh`
  - `git diff --check`
- Cluster:
  - Build with `CUDA_ARCH=sm_70`.
  - Run descriptor-bound FFN smoke with real router enabled.
  - Run full appliance gate with `--pack-index`.

## Deferred

- Attention/residual/norm, selected-token logits, serving, MTP, and throughput
  remain deferred.
