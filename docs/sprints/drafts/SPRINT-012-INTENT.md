---
sprint: 012
title: V100 Appliance Gate And Bounded Output-Head Logits
seed: continue toward the DS4 V100 appliance vision with real implementation
date: 2026-05-18
---

# SPRINT-012 Intent

## Seed Prompt

Continue the DS4 V100 appliance goal by running `$sprint-plan` and
`$sprint-execute` in a loop until the vision is realized. The next sprint must
implement something concrete, not only update planning documents.

## Orientation Summary

- Sprints 001-004 shipped source inventory, manifest, deterministic per-GPU
  pack shards, and full 8x V100 source-byte residency.
- Sprints 005-008 established source dtype correctness anchors: BF16 gather,
  execution-format policy, guarded CPU source oracle, source-layout guards,
  exact KV admission, and a bounded source-F8 CUDA decode probe.
- Sprints 009-011 moved into V100 execution: KV/state arena allocation,
  compressor recurrence, stage-owned KV updates, source-F8 projection
  diagnostics, and projection-fed ratio-4/ratio-128 attention/compressor
  smokes.
- Normal source-layout generation remains intentionally guarded. The missing
  deployment gates are coherent layer output, router/shared/routed expert
  execution, output-head logits/top-k, appliance health checks, and later
  performance work.
- `docs/architecture/DS4-V100-LAYOUT.md` is the architecture anchor for source
  dtypes, V100 execution dtypes, layer ownership, memory layout, and topology.

## Vision Context

The North Star is a narrow DS4 V100 appliance for an 8x V100-SXM2-32GB stack
that runs the high-intelligence source quantized DS4 Flash model from pure
device-resident packs by default.

Sprint 012 sits between the bounded projection/attention gate and deployment.
It should add a concrete logits-producing surface without pretending the full
MoE runtime or public serving path is ready.

## Relevant Codebase Areas

| Area | Role |
|---|---|
| `docs/architecture/DS4-V100-LAYOUT.md` | Source/runtime dtype and topology contract |
| `ds4_gpu.h` | CUDA arena/tensor API declarations |
| `ds4_cuda.cu` | CUDA source dtype, projection, router, and diagnostic kernels |
| `ds4_v100_context.[ch]` | Execution-format policy and layer/context descriptors |
| `ds4_v100_context_cuda.cu` | KV arena views and stage-owned KV update APIs |
| `ds4_source_formats.[ch]` | CPU source-format reference helpers |
| `tests/cuda_source_dtypes_smoke.c` | Source dtype CUDA smoke coverage |
| `tests/cuda_v100_projection_attention_smoke.c` | Latest projection/attention slice |
| `tools/ds4-source-oracle-vector.c` | Real-model source guard and CPU oracle entry point |
| `tools/ds4-v100-residency-smoke.c` | Real-pack residency and BF16 probe tool |

## Constraints

- V100 has no native BF16, FP8, or FP4 tensor-core execution.
- Source BF16 may be gathered/converted or used in bounded diagnostics, but no
  code should claim native BF16 compute on V100.
- Broad FP32 GEMM/matmul fallback is not acceptable as the production runtime
  path. FP32 is acceptable for control, reductions, CPU references, and bounded
  diagnostic comparisons.
- Large source weights must remain device-resident; no persistent dequantized
  full-model copies in VRAM or host/SSD offload as a default path.
- Public source-layout generation/server unlock remains guarded until a real
  correctness gate passes.
- The sprint must commit a runnable implementation artifact and V100 validation
  evidence.

## Prior Deferred/Follow-Up Items Now Actionable

- From `SPRINT-011-DEFERRED.md`: full logits-producing decode is targeted at
  Sprint 012+ after projection/attention correctness.
- From `SPRINT-011-FOLLOWUPS.md`: output head and selected-token comparison are
  critical for Sprint 012.
- From `SPRINT-011-FOLLOWUPS.md`: full layer output and router/expert
  correctness remain critical, but may be split if implementing them together
  would block a smaller shippable logits primitive.
- From `SPRINT-005-DEFERRED.md`: BF16 output-head compute surfaces are now
  relevant because the output head is a source BF16 tensor and V100 needs an
  explicit conversion/diagnostic boundary.

## Proposed Sprint Shape

This sprint should ship the first bounded output-head/logits gate on V100:

1. Add a CUDA BF16 source matrix matmul diagnostic that reads BF16 source rows
   from a `ds4_gpu_arena`, expands values to F32 in-kernel, computes logits for
   a bounded row count, and compares against a CPU BF16 reference.
2. Add a bounded logits smoke that uses the BF16 output-head primitive to
   produce logits and top-1/top-k evidence from a synthetic source-layout output
   head.
3. Add an appliance readiness gate that runs the current real-model guards and
   V100 smokes and emits a fail-closed readiness report. The gate should
   distinguish `ready=false because MoE/full selected-token serving is still
   missing` from actual test failures.
4. Preserve source-layout normal-generation guard behavior.

## Success Criteria

- A committed V100 CUDA primitive reads source BF16 matrix bytes from resident
  arena memory and produces bounded output logits on device.
- CUDA tests compare BF16 output-head logits against CPU references and verify
  top-1/top-k agreement for deterministic fixtures.
- A committed gate command can be run on the V100 pod to validate source guards,
  existing source/KV/projection smokes, the new output-head logits smoke, and
  current readiness status.
- V100 `sm_70` cluster logs are archived under `docs/sprints/drafts/`.
- The sprint report explicitly states whether the appliance is deployable and
  what blocks deployment.

## Verification Strategy

- Local build:
  - compile changed host objects and CUDA-smoke objects where possible;
  - run host model-less smokes;
  - run `git diff --check`.
- V100 cluster:
  - build with `CUDA_ARCH=sm_70`;
  - run source dtype, BF16 probe, context/KV, compressor, prefill, HC relay,
    projection/attention, and new bounded logits smokes;
  - run `tools/ds4-source-oracle-vector --guards-only` against the real model;
  - run the new appliance gate command.

## Uncertainty Assessment

| Area | Risk | Notes |
|---|---|---|
| Correctness | Medium | BF16 conversion is known, but output-head reduction and top-k need new CUDA coverage |
| Scope | Medium | Full MoE plus full logits is too large for one sprint; bounded logits must ship first |
| Architecture | Low | The architecture doc already defines source BF16 output head and V100 conversion policy |
| Deployment | Medium | The gate can ship before public serving, but it must fail closed honestly |

## Open Questions

- Should the BF16 output-head diagnostic eventually convert to FP16 and use
  HMMA tiles, or should Sprint 014 evaluate FP8/Q8/vocab-TP alternatives?
- What bounded real-model fixture should become the first full selected-token
  V100 comparison after the output-head primitive exists?
- How much of router/shared/routed expert execution should be wired before
  calling the appliance deployable?
