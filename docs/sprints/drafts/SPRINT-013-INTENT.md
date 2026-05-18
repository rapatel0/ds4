---
sprint: 013
title: V100 Source MXFP4 MoE And Selected-Token Gate
seed: continue DS4 V100 appliance loop after Sprint 012 readiness gate
date: 2026-05-18
---

# SPRINT-013 Intent

## Seed Prompt

Continue `$sprint-plan` and `$sprint-execute` in a loop until the DS4 V100
appliance vision is realized. Sprint 012 shipped the output-head/logits
primitive and gate, but the gate still reports `ready=false` because full
layer/MoE and selected-token execution are missing.

## Orientation Summary

- The repo now has source-byte residency, layer/KV context planning, source
  dtype guards, F8 projection diagnostics, projection-fed attention/compressor
  smokes, BF16 output-head logits, and a runnable V100 appliance gate.
- The next concrete blocker is the routed expert path. DS4 Flash routed
  gate/up/down experts are source MXFP4, while shared experts are source
  F8_E4M3_B128.
- `docs/architecture/DS4-V100-LAYOUT.md` identifies routed gate/up/down experts
  as the dominant layer path and the biggest format/topology fork.
- Existing router/top-k and SwiGLU device kernels can be reused for a bounded
  single-token MoE fixture; existing source-F8 and source-BF16 matmul
  diagnostics can cover shared expert and output head surfaces.
- Normal source-layout generation remains guarded until a real selected-token
  path exists.

## Vision Context

The North Star remains a narrow DS4 V100 appliance that runs the
high-intelligence source quantized model from pure device-resident packs.
Sprint 013 should turn the readiness gate from "output head works" toward
"bounded layer/MoE selected token works" without starting public serving or
throughput work prematurely.

## Relevant Codebase Areas

| Area | Role |
|---|---|
| `docs/architecture/DS4-V100-LAYOUT.md` | MoE/source dtype and topology anchor |
| `ds4_source_formats.[ch]` | CPU MXFP4 source reference helpers |
| `ds4_gpu.h`, `ds4_cuda.cu` | CUDA arena matmul, router, SwiGLU, add, and logits APIs |
| `tests/cuda_source_dtypes_smoke.c` | Source dtype CUDA diagnostic coverage |
| `tests/cuda_v100_bounded_logits_smoke.c` | Output-head/logits fixture substrate |
| `tools/ds4-v100-gate.sh` | Appliance readiness gate to extend |

## Constraints

- V100 has no native Blackwell FP4/MXFP4 execution. MXFP4 is a source/runtime
  packed format that must feed explicit low-bit/decode kernels.
- No broad FP32 production GEMM fallback and no persistent dequantized expert
  copies.
- The first MXFP4 path may be diagnostic and scalar-reduction oriented, but it
  must read resident packed bytes and compare to CPU source references.
- Public serving and MTP remain out of scope until bounded selected-token
  correctness exists.

## Proposed Sprint Shape

1. Add a bounded CUDA `ds4_gpu_arena_mxfp4_matmul_f32` source expert primitive.
2. Add a bounded single-token MoE fixture that uses router selection, source
   MXFP4 gate/up/down expert matmuls, SwiGLU, accumulation, and the existing
   BF16 output-head matmul to produce a selected token.
3. Extend the V100 gate to run the new MoE/selected-token smoke while still
   reporting `ready=false` until real model layer integration exists.

## Success Criteria

- Source-MXFP4 rows in a resident arena can be multiplied by a device F32
  vector and compared against CPU `ds4_src_mxfp4_row_dot`.
- A bounded V100 MoE fixture produces route ids, route weights, expert outputs,
  logits, and selected token matching CPU references.
- The appliance gate includes the new MoE selected-token smoke and passes on
  the V100 pod.
- No public serving unlock, persistent dequantization, or throughput claim is
  introduced.

## Verification Strategy

- Local object builds and CPU source dtype smokes.
- V100 `sm_70` build/run of source dtype, bounded logits, new MoE selected-token
  smoke, and the full appliance gate.
- Real-model source guard check remains fail-closed.

## Uncertainty Assessment

| Area | Risk | Notes |
|---|---|---|
| Correctness | Medium | MXFP4 semantics are known, but router/MoE composition adds more surfaces |
| Scope | Medium | Full real-model MoE scheduler may not fit; bounded selected-token fixture should |
| Architecture | Low | Existing layout doc already identifies MXFP4 routed experts as next gate |
| Performance | High | Diagnostic MXFP4 matmul is not expected to be final throughput kernel |

## Open Questions

- Should the first production routed expert path evolve from this MXFP4
  diagnostic, or should it switch to TurboMind/tc-grid once correctness is
  anchored?
- How much of the bounded MoE fixture can be reused when wiring real layer
  descriptors from the pack index?
