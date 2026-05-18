---
sprint: 012
title: V100 Appliance Gate And Bounded Output-Head Logits
status: completed
date: 2026-05-18
verdict: SHIP
---

# SPRINT-012 Report: V100 Appliance Gate And Bounded Output-Head Logits

## Verdict

`SHIP`

Sprint 012 shipped a concrete logits-facing V100 primitive and a runnable
appliance gate. Source-BF16 output-head rows can now be read from a resident
`ds4_gpu_arena`, expanded explicitly, multiplied by a device F32 hidden vector,
and compared against a CPU BF16 reference with top-k agreement.

The appliance is still not ready for serving. The new gate reports
`ready=false` because full layer/MoE execution, full selected-token decode,
public serving, MTP, and throughput benchmarks remain missing.

## What Shipped

- Added `ds4_gpu_arena_bf16_matmul_f32`, a bounded source-BF16 matrix-vector
  diagnostic path from resident arena bytes to device F32 logits.
- Added a CUDA `sm_70` kernel for the BF16 source matmul and a fail-closed host
  stub so non-CUDA builds do not imply BF16 runtime support.
- Added `tests/cuda_v100_bounded_logits_smoke.c`, which builds a deterministic
  bounded output-head fixture, compares logits against a CPU BF16 reference,
  checks top-3 agreement, and rejects invalid source/output views.
- Added `tools/ds4-v100-gate.sh`, a V100 readiness gate that runs real-model
  source guards, Sprint 009-011 CUDA regressions, and the new bounded logits
  smoke.

## Evidence

Local validation:

- `make tests/cuda_v100_bounded_logits_smoke.o tests/cuda_bf16_probe.o`
- `make ds4_gpu_arena_stub.o tests/bf16_probe_smoke tests/source_dtypes_smoke`
- `./tests/bf16_probe_smoke`
- `./tests/source_dtypes_smoke`
- `bash -n tools/ds4-v100-gate.sh`
- `git diff --check`

Cluster validation:

- `docs/sprints/drafts/SPRINT-012-GATE-CLUSTER/gate-summary.log`
  - Source guards passed against `/models/DSv4-Flash-256e-fixed.gguf`.
  - Source dtype, BF16 probe, 1M single-slot context/KV, compressor bridge,
    prefill KV, HC relay, projection/attention, and bounded logits smokes
    passed on the 8x V100 pod.
  - Gate summary: `PASS`, `failures=0`, `ready=false`.
- `docs/sprints/drafts/SPRINT-012-GATE-CLUSTER/bounded_logits.log`
  - `cuda_v100_bounded_logits_smoke: ok`.

## Deviations

- The BF16 output-head primitive is diagnostic and scalar-reduction oriented,
  not the final fast output-head implementation.
- The bounded logits smoke uses a synthetic output-head fixture, not a full
  model selected-token path.
- Router, shared expert, routed expert, and full layer residual/HC scheduler
  integration remain outside this sprint.

## Handoff

Sprint 013 should target the real deployment blocker: a coherent V100
source-layout layer/MoE/selected-token gate. The new output-head primitive and
gate script are the substrate for that work.
