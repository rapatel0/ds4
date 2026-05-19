# Sprint 039: Resident MTP Logits and Top-K Parity

## Status

Complete.

## Overview

Sprint 039 closes the next concrete `missing=mtp_forward` gap by proving the
resident MTP sidecar can turn an MTP hidden-control state into draft logits and
a selected token using the native MTP output head path:

- MTP-specific HC head collapse (`mtp.0.hc_head_*`);
- MTP output norm (`mtp.0.norm.weight`);
- base model vocabulary projection (`output.weight`);
- CPU oracle top-k parity for deterministic input.

This sprint still does not clear full `mtp_forward`. It proves the final logits
surface needed before composing prefix + attention + FFN + logits into a full
one-token resident MTP block and adding draft verify/rollback semantics.

## Use Cases

- Validate that MTP draft logits use sidecar-resident HC head tensors rather
  than the base model output HC head.
- Verify the MTP output norm is loaded from the sidecar arena and feeds the
  base model vocabulary projection.
- Add a gate rung that distinguishes "MTP attention/FFN slices pass" from "MTP
  can produce a draft token candidate."

## Architecture

- Keep `docs/architecture/DS4-V100-LAYOUT.md` as the topology anchor:
  MTP sidecar tensors live in a compact gpu7 arena, while the base output head
  remains a gpu7-owned base-model tensor.
- Add a focused `tools/ds4-v100-mtp-logits-smoke.c` rather than overloading the
  integrated attention smoke. Logits/top-k need both the MTP sidecar GGUF and
  the base model pack index.
- Add a resident-arena HC-head weight helper for sidecar F32 scale/base tensors.
  The existing helper reads scale/base from mmap-backed model bytes; Sprint 039
  needs the same sigmoid-plus-epsilon transform using resident sidecar offsets.
- Upload only `output.weight` into a small gpu7 output-head arena for this
  smoke. That avoids depending on scheduler internals while keeping the logits
  path device-resident.
- Compare GPU top-k against a CPU oracle that reads the same sidecar bytes and
  base-model BF16 output rows.

## Implementation

1. Add `ds4_gpu_arena_output_hc_weights_tensor`.
2. Add `tools/ds4-v100-mtp-logits-smoke.c` with:
   - `--model FILE`
   - `--mtp-model FILE`
   - `--pack-index FILE`
   - `--gpu N`
   - `--require-gpus N`
   - `--reserve-mib N`
   - `--top-k N`
   - `--logit-tol F`
   - `--report FILE`
3. In the smoke:
   - open and upload the MTP sidecar on gpu7;
   - validate/bind `mtp.0.hc_head_fn.weight`,
     `mtp.0.hc_head_scale.weight`, `mtp.0.hc_head_base.weight`, and
     `mtp.0.norm.weight`;
   - parse the pack index and validate base `output.weight` as BF16
     `[4096 x vocab]`;
   - upload `output.weight` into a gpu7 arena;
   - run deterministic MTP HC through resident HC collapse, MTP norm, BF16
     output projection, read logits, and select top-k;
   - compute the same top-k with CPU sidecar/base bytes and compare.
4. Wire the new smoke into `Makefile` and `tools/ds4-v100-gate.sh`.
5. Keep readiness honest: passing `mtp_logits` should move the MTP missing rung
   forward, but readiness remains blocked until full one-token MTP forward and
   draft verify/rollback are implemented.
6. Update the sprint report, follow-ups, and `docs/sprints/VISION.md`.

## Files Summary

- `ds4_gpu.h`
- `ds4_cuda.cu`
- `ds4_gpu_arena_stub.c`
- `tools/ds4-v100-mtp-logits-smoke.c`
- `tools/ds4-v100-gate.sh`
- `Makefile`
- `docs/sprints/SPRINT-039-REPORT.md`
- `docs/sprints/SPRINT-039-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

## Definition of Done

- Local object/stub compile passes for the new smoke and GPU API declaration.
- CUDA build on the V100 cluster passes for
  `tools/ds4-v100-mtp-logits-smoke`.
- Focused MTP logits smoke passes on gpu7 with all 8 V100s visible, the real
  base model, the real MTP sidecar, and the real pack index.
- Full V100 gate includes `mtp_logits PASS`, has no failures, and still reports
  `ready=false` with the next MTP blocker rather than a completed readiness
  claim.
- Sprint report records commands, outputs, top-k tokens/logits, memory reserve,
  and remaining blockers.
- Vision document names Sprint 039 and updates the readiness ladder.

## Results

- Local object/stub compile passed:
  `make tools/ds4-v100-mtp-logits-smoke.o ds4_v100_mtp.o ds4_v100_context.o ds4_gpu_arena_stub.o ds4_cpu.o ds4_source_formats.o`
- Local gate shell check passed:
  `bash -n tools/ds4-v100-gate.sh`
- Cluster CUDA build passed:
  `CUDA_ARCH=sm_70 make tools/ds4-v100-mtp-logits-smoke`
- Focused cluster smoke passed:
  `mtp_logits_smoke: cpu_top1=65615 gpu_top1=65615 max_abs=9.53674316e-07 PASS`
- Focused top-k evidence:
  - rank 1 token `65615`, delta `0`
  - rank 2 token `8764`, delta `0`
  - rank 3 token `5865`, delta `0`
  - rank 4 token `2630`, delta `0`
  - rank 5 token `41163`, delta `9.53674316e-07`
- Focused memory evidence:
  - MTP sidecar resident arena `3,807,601,408` bytes
  - base `output.weight` upload `1,059,061,760` bytes
  - free after output upload `28,878,307,328` bytes
  - reserve requirement `4,294,967,296` bytes
- Full 8-GPU gate passed:
  `gate summary PASS failures=0 ready=false`
- Full gate now includes:
  `gate mtp_logits PASS`
- Readiness remains intentionally blocked on:
  `missing=mtp_forward`

## Risks

- The base output projection is BF16 in the current V100 pack index, while the
  native `ds4.c` MTP helper can use a quantized output projection depending on
  model format. This smoke should validate the fork's current base model format
  rather than claim universal MTP output support.
- CPU and V100 reduction order can produce small logit differences. Token
  parity should remain exact; logit tolerance should be explicit and reported.
- This proves deterministic hidden-state logits, not a complete MTP draft. The
  next sprint still has to compose prefix, attention, FFN, logits, and state
  handling in one path.

## Security

No public serving surface changes. This sprint adds local CUDA smoke coverage
and one gate step.

## Dependencies

- Sprint 031 MTP sidecar residency.
- Sprint 036 resident MTP FFN slice.
- Sprint 038 resident MTP integrated attention.
- Base model GGUF and MTP sidecar GGUF at the cluster paths used by the gate.
- Pack index with `output.weight` BF16 binding for gpu7.

## Open Questions

- Whether Sprint 040 should compose the full one-token resident MTP block first
  or immediately wire a draft/verify state smoke around the logits primitive.
- Whether the output-head arena should stay as a smoke-only upload or become a
  reusable runtime binding in the MTP forward executor.
