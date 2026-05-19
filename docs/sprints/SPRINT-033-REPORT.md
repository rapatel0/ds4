---
sprint: 033
title: V100 Resident MTP Q8 Projection Probe
status: completed
date: 2026-05-18
verdict: SHIP
---

# SPRINT-033 Report: V100 Resident MTP Q8 Projection Probe

## Summary

Sprint 033 shipped the first resident MTP compute primitive. The MTP sidecar is
no longer only a gpu7 residency object: Q8_0 tensors inside the compact sidecar
arena can now feed V100 CUDA matmul kernels directly through
`ds4_gpu_arena_q8_0_matmul_f32`.

The sprint validates the first MTP prefix projection tensors:

- `mtp.0.e_proj.weight`: Q8_0 `[4096,4096]`, `n_tok=1`
- `mtp.0.h_proj.weight`: Q8_0 `[4096,4096]`, `n_tok=4`

Both resident-arena outputs matched the existing Q8_0 CUDA path exactly on the
V100 pod.

## Implementation

- Added `ds4_gpu_arena_q8_0_matmul_f32` to the GPU arena API.
- Implemented the CUDA adapter by reusing the existing prequantized V100 Q8_0
  DP4A kernels against `arena + resident_offset`.
- Added a fail-closed non-CUDA stub.
- Added MTP sidecar map/size accessors and a Q8_0 tensor-view helper that
  validates dtype, shape, stride, source bounds, and resident bounds.
- Added `tools/ds4-v100-mtp-prefix-smoke`.
- Added `mtp_prefix` to `tools/ds4-v100-gate.sh`.
- Updated readiness semantics:
  - missing sidecar -> `mtp_sidecar`
  - missing residency -> `mtp_residency`
  - missing projection parity -> `mtp_prefix`
  - passing projection parity but no full draft -> `mtp_forward`

## Evidence

Focused gpu7 smoke:

Artifact: `docs/sprints/drafts/SPRINT-033-MTP-PREFIX/mtp_prefix.report`

Key results:

- MTP arena bytes: `3807601408`
- Uploaded tensors: `32`
- Uploaded bytes: `3807600108`
- Spot checks: `60`
- Free after upload: `29937369088`
- Required reserve: `4294967296`
- `mtp.0.e_proj.weight`: `max_abs=0`, `max_rel=0`, `PASS`
- `mtp.0.h_proj.weight`: `max_abs=0`, `max_rel=0`, `PASS`

Full 8x V100 gate:

Artifact directory: `docs/sprints/drafts/SPRINT-033-GATE-CLUSTER-8GPU/`

Result:

```text
gate mtp_prefix PASS
gate v100_replay_tool PASS
gate v100_appliance_http PASS
gate v100_appliance_http_long PASS
gate readiness NOT_READY missing=mtp_forward
gate summary PASS failures=0 ready=false
```

Selected replay timing from the full gate:

- `open_total_ms`: `251120.228`
- `prompt_replay_ms`: `3401.375`
- `continuation_decode_ms`: `145.478`
- generated tokens: `2`
- first token: id `926`, hex `3136`

Long HTTP smoke:

- request 1: `generated_tokens=2`, `first_hex=3136`, `continuation_ms=143.544`
- request 2: `generated_tokens=2`, `first_hex=3136`, `continuation_ms=143.077`

## Notes

The first focused run found that using the full sidecar mmap plus high
`source_offset` as the reference model-map path could fail a CUDA host-to-device
copy near the end of the sidecar file. The resident arena path itself had
already computed `e_proj` correctly. The final smoke compares resident output
against the existing `ds4_gpu_matmul_q8_0_tensor` path using a tensor-local host
copy of the same sidecar bytes. That keeps the comparison against the existing
Q8_0 kernel surface while avoiding high-offset mmap behavior in the test
harness.

## Remaining Gap

Level 3 is still not complete. Sprint 033 proves resident MTP Q8_0 projection
compute, but the appliance still lacks:

- F32 MTP prefix norms from resident sidecar tensors.
- Prefix add/repeat composition into `[4 x 4096]` HC state.
- Dense MTP block attention/FFN execution.
- Q4_K routed MTP expert execution from the sidecar arena.
- MTP output-head logits/top-k.
- Draft/verify/rollback state tests.
