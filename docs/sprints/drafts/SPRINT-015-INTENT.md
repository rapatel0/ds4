---
sprint: 015
title: V100 Descriptor-Bound FFN Compute Gate
date: 2026-05-18
seed: continue sprint-plan and sprint-execute loop until DS4 V100 appliance vision is realized
vision: ../VISION.md
architecture: ../../architecture/DS4-V100-LAYOUT.md
---

# Sprint 015 Intent: V100 Descriptor-Bound FFN Compute Gate

## Orientation Summary

- Sprint 014 shipped a strict real pack-index descriptor gate for layer 2 and
  wired it into `tools/ds4-v100-gate.sh` behind `--pack-index`.
- `ds4_v100_context` already classifies and stores pack entries internally, but
  compute code has no public typed descriptor binding API.
- Existing V100 source-format CUDA primitives can execute bounded BF16, F8, and
  MXFP4 matrix-vector operations from `ds4_gpu_arena` offsets.
- The next deployability gap is not another report-only descriptor check; it is
  proving that real source-model bytes at real pack offsets can feed a compute
  slice.
- `docs/architecture/DS4-V100-LAYOUT.md` remains the topology and memory-layout
  anchor. The sprint should preserve pure device-resident pack semantics and
  keep BF16/FP8/MXFP4 as source formats on V100, not native tensor-core claims.

## Relevant Code Areas

- `ds4_v100_context.h`, `ds4_v100_context.c`
- `ds4_pack.h`, `ds4_pack.c`
- `ds4_gpu.h`, `ds4_cuda.cu`
- `ds4_source_formats.h`, `ds4_source_formats.c`
- `tools/ds4-v100-gate.sh`
- `tests/cuda_v100_mxfp4_moe_smoke.c`
- `tests/cuda_source_dtypes_smoke.c`

## Success Criteria

- Expose a fail-closed runtime descriptor binding API from `ds4_v100_context`.
- Provide typed shape/offset metadata sufficient to construct source row views
  for BF16/F8/MXFP4 compute surfaces.
- Add a local binding smoke that proves layer-2 and output-head descriptors can
  be materialized without CUDA.
- Add a CUDA descriptor-bound FFN smoke that reads real bytes from the source
  GGUF using pack-index source offsets, uploads them into a V100 arena at real
  shard offsets, and compares routed MXFP4 plus shared F8 FFN outputs against
  CPU source-format references.
- Add the new smoke to the appliance gate behind `--model`/`--pack-index`.
- Run local validation and the full V100 cluster gate.

## Verification Strategy

- Local:
  - `make tests/v100_layer_binding_smoke`
  - `./tests/v100_layer_binding_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --layer 2`
  - `bash -n tools/ds4-v100-gate.sh`
  - `git diff --check`
- Cluster:
  - Build the new CUDA smoke with `CUDA_ARCH=sm_70`.
  - Run the smoke against `/models/DSv4-Flash-256e-fixed.gguf` and the real pack
    index.
  - Run the full appliance gate with `--pack-index`.

## Uncertainty Assessment

- Correctness: Medium. Real source bytes and CPU reference reduce ambiguity,
  but this is still a bounded FFN slice with synthetic activation input.
- Scope: Medium. Descriptor binding is small; real-data CUDA smoke touches file
  IO, arena offsets, F8, MXFP4, and shared+routed FFN composition.
- Architecture: Low. The work follows existing pack/context/arena patterns.

## Open Questions

- Should the first descriptor-bound compute smoke include output-head logits?
  Default answer for this sprint: no, because that would require crossing to
  GPU 7 or duplicating output head bytes in the layer-2 arena.
- Should router selection be real in this sprint? Default answer: no. Use a
  fixed expert and deterministic activation to keep the first real-byte FFN
  gate bounded; router integration can follow once descriptor-bound compute is
  stable.

## Follow-Ups Pulled Forward

- From `docs/sprints/SPRINT-014-FOLLOWUPS.md`:
  - Runtime descriptor table: Critical, targeted here.
  - Descriptor-bound layer compute: Critical, targeted here as bounded FFN.
  - Shared expert execution in real layer: Critical, targeted here.

## Vision Context

The North Star is a DS4 V100 appliance that runs the source quantized model
from device-resident packs. Sprint 015 sits between descriptor validation and
full serving: it must prove that real source-model bytes bound by pack
descriptors can drive executable V100 compute.
