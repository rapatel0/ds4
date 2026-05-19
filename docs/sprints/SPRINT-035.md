---
created: 2026-05-18
status: complete
---

# Sprint 035: Resident MTP Q4_K Routed Expert Execution

## Overview

Sprint 035 turns the MTP sidecar's largest remaining resident tensor family into
an executable V100 path. Sprint 034 proved that gpu7 can hold the full MTP
sidecar and compose the F32/Q8_0 prefix chain into `mtp_input_hc`; this sprint
adds the Q4_K routed expert primitive that the MTP FFN block needs next.

This is still a narrow appliance step, not a generic GGUF feature. The target
layout is the `deepseek4_mtp_support_q4k_q8_0_f32` sidecar described by the
gate and the architecture anchor in
`docs/architecture/DS4-V100-LAYOUT.md`.

## Scope

- Add a typed resident view for 3D Q4_K expert tensors in the MTP sidecar.
- Expose an arena-resident one-token Q4_K routed-MoE API that reuses the
  existing `sm_70` Q4_K gate/up and direct six-expert down-sum kernels.
- Add a focused MTP Q4_K smoke that uploads the real sidecar to gpu7, runs a
  deterministic six-expert routed FFN slice, and compares CUDA output against a
  CPU reference over the selected expert slices.
- Wire the smoke into `tools/ds4-v100-gate.sh` so the readiness ladder can
  distinguish `mtp_prefix`, `mtp_q4k`, and full `mtp_forward`.
- Preserve the current base replay and HTTP appliance gates.

## Non-Goals

- Full MTP block forward with attention, shared expert, HC post, logits, or
  draft verification.
- Multi-slot or batched MTP expert scheduling.
- Changing the source model format or the base 43-layer scheduler.
- Claiming Level 3 readiness; the gate should still stop at `mtp_forward`.

## Architecture

The Q4_K expert tensors are already resident in the gpu7 MTP arena:

| Tensor | Native dtype | Shape | Row bytes | Expert bytes | Total bytes |
|---|---|---:|---:|---:|---:|
| `mtp.0.ffn_gate_exps.weight` | Q4_K | `[4096, 2048, 256]` | 2304 | 4,718,592 | 1,207,959,552 |
| `mtp.0.ffn_up_exps.weight` | Q4_K | `[4096, 2048, 256]` | 2304 | 4,718,592 | 1,207,959,552 |
| `mtp.0.ffn_down_exps.weight` | Q4_K | `[2048, 4096, 256]` | 1152 | 4,718,592 | 1,207,959,552 |

The runtime primitive should pass raw arena pointers into the same CUDA kernel
sequence currently used by descriptor-bound mapped Q4_K MoE:

1. Quantize the F32 input vector to Q8_K.
2. Run Q4_K gate/up dot products for six selected experts.
3. Clamp, SwiGLU, and apply router weights.
4. Quantize the six mid vectors to Q8_K.
5. Run the direct Q4_K six-expert down-sum kernel into a 4096-wide output.

The first version is intentionally one-token and six-expert only because that is
the decode path needed by MTP K=1. Batched MTP scheduling belongs after full MTP
forward correctness exists.

## Implementation Plan

1. Add `ds4_gpu_q4_k_expert_view` to `ds4_gpu.h`.
2. Add `ds4_v100_mtp_sidecar_q4_k_expert_view()` to validate the MTP tensor
   dtype, shape, row stride, expert stride, source range, and resident range.
3. Add `ds4_gpu_arena_q4_k_routed_moe_one_f32()` to `ds4_cuda.cu` and a
   fail-closed stub to `ds4_gpu_arena_stub.c`.
4. Add `tools/ds4-v100-mtp-q4k-smoke.c`:
   - opens the real MTP sidecar on gpu7,
   - binds the three Q4_K expert tensors,
   - builds deterministic F32 input, selected expert ids, and route weights,
   - runs the resident CUDA primitive,
   - computes a CPU reference over the selected slices only,
   - reports max absolute/relative error and timings.
5. Add Makefile and gate wiring:
   - build target `tools/ds4-v100-mtp-q4k-smoke`,
   - gate name `mtp_q4k`,
   - readiness order `mtp_sidecar -> mtp_residency -> mtp_prefix -> mtp_q4k -> mtp_forward`.
6. Run local compile checks, then run the focused smoke and full gate on the
   V100 pod.
7. Write a report/follow-ups, update `docs/sprints/VISION.md`, and commit.

## Definition of Done

- [x] Q4_K expert sidecar view rejects invalid dtype/shape/stride/range.
- [x] Resident arena Q4_K routed-MoE API runs without model-map range cache.
- [x] Focused MTP Q4_K smoke passes on a V100 with the real MTP sidecar.
- [x] Full 8-GPU gate passes all implemented checks and reports
  `ready=false missing=mtp_forward`.
- [x] Sprint report and vision readiness ladder are updated.
- [x] Changes are committed with only explicitly staged files.

## Result

`SHIP`. The V100 pod built `tools/ds4-v100-mtp-q4k-smoke` for `sm_70`, ran the
real MTP sidecar from gpu7 resident offsets, and matched the CPU Q4_K reference
with `max_abs=1.43051147e-06`. The full 8-GPU gate passed with `failures=0` and
`ready=false missing=mtp_forward`.

## Risks

- CPU and CUDA Q4_K references may not be bit-exact because CUDA uses warp
  reductions and Q8_K requantization. The smoke must use an explicit tolerance
  and report it.
- The Q4_K sidecar is large; the smoke should not allocate full host duplicates
  beyond the existing mapped sidecar. It should read only selected expert slices
  for CPU reference.
- The existing Q4_K CUDA path currently assumes one token and six experts. That
  is acceptable for this sprint, but it must be documented as a decode-only
  primitive.

## Expected Outcome

After this sprint, the MTP path will have resident prefix composition plus
resident Q4_K routed expert execution. The remaining Level 3 gap should narrow
to dense/shared MTP block execution, logits/top-k, and draft verify/rollback.
