# Sprint 036: Resident MTP FFN Slice

## Status

Complete.

## Objective

Move the MTP path from isolated prefix and routed-expert primitives to an
executable resident FFN block slice on gpu7. The slice must use sidecar-resident
MTP tensors for router logits, bias top-k routing, Q4_K routed experts, Q8_0
shared expert execution, and routed+shared accumulation.

This sprint is intentionally still a slice, not full `mtp_forward`: MTP raw
attention/cache update, logits, and draft verify/rollback remain follow-ups.

## Vision Context

`docs/sprints/VISION.md` names `missing=mtp_forward` as the remaining readiness
blocker after Sprint 035. `docs/architecture/DS4-V100-LAYOUT.md` remains the
architecture anchor: the MTP block is gpu7-resident, uses the sidecar arena, and
should avoid format churn or host staging once the sidecar upload is complete.

## Implementation Plan

1. Add a resident arena router-select API for one-token bias routing.
   - Input: `ffn_gate_inp.weight` F32 matmul logits and `exp_probs_b.bias`.
   - Output: selected expert ids, normalized route weights, and router probs.
   - Semantics must match the existing CUDA router kernels:
     `sqrt(softplus(logit))`, bias only for top-k score, unbiased probs for
     route weights, normalized then scaled by `1.5`.

2. Add a sidecar view helper for 2D F32 matrices.
   - Needed for `mtp.0.ffn_gate_inp.weight` with shape `[4096,256]`.
   - Keep the existing source row view convention: rows are output rows, cols
     are input columns.

3. Add a focused MTP FFN smoke tool.
   - Open and upload the real MTP sidecar on gpu7.
   - Build deterministic F32 `after_attn_hc` input.
   - Run resident HC FFN control, split, and FFN RMS norm.
   - Run resident F32 router matmul.
   - Run resident bias router selection.
   - Run resident Q4_K routed MoE using the Sprint 035 primitive.
   - Run resident Q8_0 shared gate/up/down matmuls and unclamped shared SwiGLU.
   - Add routed and shared outputs.
   - Expand back to `next_hc`.
   - Compare selected experts, route weights, routed output, shared output,
     final FFN output, and `next_hc` to a CPU reference using the same sidecar
     bytes.

4. Wire the smoke into the V100 gate.
   - Add build target.
   - Add `mtp_ffn` gate after `mtp_q4k`.
   - Update readiness ordering from `mtp_q4k` to `mtp_ffn` before the remaining
     `mtp_forward` blocker.

## Definition of Done

- `make` builds the new object on macOS with the CUDA stub path.
- CUDA build on the V100 cluster builds the new smoke for `sm_70`.
- Focused cluster smoke passes against
  `/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`.
- Full V100 gate passes with `mtp_ffn PASS` and still reports
  `ready=false missing=mtp_forward`.
- `docs/sprints/VISION.md` records the new readiness rung and Sprint 036 state.
- Sprint report and follow-ups are written.

## Results

- Local C/stub build passed:
  `make tools/ds4-v100-mtp-ffn-smoke.o ds4_v100_mtp.o ds4_gpu_arena_stub.o ds4_cpu.o`
- Local shell/diff checks passed:
  `bash -n tools/ds4-v100-gate.sh`, `git diff --check`
- Cluster CUDA build passed:
  `CUDA_ARCH=sm_70 make tools/ds4-v100-mtp-ffn-smoke`
- Focused cluster smoke passed:
  `mtp_ffn_smoke PASS`
- Full 8-GPU gate passed:
  `gate summary PASS failures=0 ready=false`
- Readiness remains intentionally blocked on:
  `missing=mtp_forward`

## Risks

- Shared expert CUDA paths use an optional clamped fused helper elsewhere, but
  the CPU/native shared FFN path is unclamped. This sprint uses separate Q8_0
  matmuls plus unclamped `ds4_gpu_swiglu_tensor(..., clamp=0.0f)` for parity.
- The smoke remains one-token decode only. Batched-slot scheduling is deferred
  to throughput work after full MTP correctness.
- Passing this sprint does not prove MTP attention/raw cache or draft-token
  rollback semantics.
