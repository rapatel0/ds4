# Sprint 244 - TP/EP Resident Dense Tensor-Core Ceiling

Date: 2026-05-23
Status: Planned

## Overview

Sprint 243 rejected the first naive HMMA dense kernel. The failure mode was
useful: simply adding WMMA is not enough if each tile repeatedly decodes and
stages packed F8 weights inefficiently. Before spending more time on custom
low-bit dense kernels, Sprint 244 measures a resident tensor-core ceiling for
the same TP/EP composition tensors.

This sprint adds an opt-in diagnostic path that expands the two F8 composition
tensors once during setup into resident FP16 device buffers, converts the
resident activations once into FP16, and uses cuBLAS FP16 Tensor Core GEMM with
FP32 output/accumulation during the decode loop. This is not the desired final
format, but it tells us how much dense-stage time is theoretically removable
when F8 decode and layout conversion are kept out of the per-step boundary.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Add an opt-in `--dense-f16-cublas-compose` path to the separate TP/EP
  full-layer smoke.
- Target only the two dense composition tensors used in the resident loop:
  - `blk.2.attn_output_b.weight`
  - `blk.2.ffn_down_shexp.weight`
- Preserve scalar dense and naive HMMA diagnostic paths for A/B.
- Benchmark at `32` slots / `256K` / `50` resident steps, MTP off:
  - scalar dense + fused compose/sum control;
  - resident FP16/cuBLAS dense + fused compose/sum candidate.
- Report dense backend selection and stage timings.

## Non-Goals

- No PP scheduler edits.
- No `ds4_v100_scheduler.*` changes.
- No MTP.
- No server/API integration.
- No claim that expanded FP16 weights fit or preserve the final memory budget
  across all model tensors.
- No final logits-equivalence claim.

## Design

For the two composition tensors only:

```text
setup:
  packed F8 rows -> GPU
  GPU expand F8 -> resident FP16 weight matrix
  deterministic input activations -> resident FP16 activation matrix

decode loop:
  cublasGemmEx(W_fp16^T, X_fp16) -> FP32 output shard
  EP return + fused compose/sum unchanged
```

This is a ceiling measurement. If it is materially faster, the next production
kernel should preserve packed low-bit residency but use the same compute shape
with software-pipelined F8 decode. If it is not faster, dense GEMM is not the
next practical serving lever and the sprint sequence should move toward
all-layer/server integration or remaining synchronization collapse.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | Resident FP16/cuBLAS dense ceiling path and reporting |
| `docs/sprints/SPRINT-244.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `docs/architecture/DS4-V100-TP-EP-LAYER2-COMMUNICATION.md` | Boundary evidence |
| `logs/from-cluster/sprint244-tp-ep-dense-f16-cublas/` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan is committed before implementation evidence.
- [ ] Implementation stays in the separate TP/EP codepath.
- [ ] No PP scheduler files are modified.
- [ ] `--dense-f16-cublas-compose` builds on the V100 pod.
- [ ] Scalar dense + fused compose/sum control still passes.
- [ ] FP16/cuBLAS dense + fused compose/sum candidate passes finite/repeat
      checks.
- [ ] A/B evidence records `ms_per_step`, `slot_step_tok_s`, dense stage time,
      compose stage time, checksum, and selected dense backend.
- [ ] Evidence is copied to
      `logs/from-cluster/sprint244-tp-ep-dense-f16-cublas/`.
- [ ] Status, vision, and architecture docs are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

## Risks

- The FP16 expanded path may be faster but memory-prohibitive as a final model
  format; that still makes it useful as a compute ceiling.
- cuBLAS launch overhead may dominate at 32 slots even when tensor cores are
  used.
- Numeric checksum will likely differ from scalar F8 FP32-dot output because
  inputs and weights are rounded to FP16.

## Decision

Pending.
