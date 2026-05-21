# Sprint 118 - Per-Slot Attention HMMA Probe

Date: 2026-05-21

## Objective

Test the smallest real tensor-core path that affects the current fast served
appliance. Sprint 117 showed that `async_pipeline_mode=per-step` preserves
stage overlap but dispatches the hot F8 projections with `n_tokens=1`, so the
Sprint 115/116 batch HMMA kernels are not active in the default HTTP path.

This sprint adds an opt-in single-token Volta HMMA dispatch for the hot
`4096 x 8192` per-slot F8 projection:

| Shape | Tensor family |
|---:|---|
| `4096 x 8192` | attention output-B / HC projection |

## Rationale

The scalar shared gate/up/SwiGLU fusion in Sprint 117 was correct but did not
increase throughput. The failure mode was useful: launch fusion alone is not
enough when the math still runs through scalar row-pair reductions.

The existing `arena_f8_e4m3_b128_matmul_batch_hmma_attn_kernel` is generic over
rows/cols and can run at `n_tokens=1`. That wastes the unused WMMA token
columns, but it also reduces CTA count dramatically and uses Volta tensor
cores. The `4096 x 8192` shape is a safer first target than smaller
projections because it has more K-side work per launch and was confirmed hot
in the Sprint 117 served trace.

## Implementation

1. Add `DS4_CUDA_F8_HMMA_SINGLE=1` in the CUDA F8 plain matmul wrapper.
2. Reuse `arena_f8_e4m3_b128_matmul_batch_hmma_attn_kernel` with
   `n_tokens=1` for `rows=4096, cols=8192`.
3. Add launcher/env plumbing as `DS4_V100_CUDA_F8_HMMA_SINGLE=0`.
4. Keep the path opt-in unless same-binary served A/B shows a clear win.
5. Capture wrapper trace evidence that the served path dispatches
   `plain/hmma_single` instead of scalar `plain/rows2` for the target shape.

## Definition of Done

- [x] CUDA source builds for `sm_70`.
- [x] `DS4_V100_CUDA_F8_HMMA_SINGLE=1` is available through the appliance
      launcher and recorded in startup logs.
- [x] Source-format smoke, full scheduler, and selected-token oracle pass.
- [x] 8-slot/256K served A/B is measured with `8/8` token match.
- [x] Trace confirms `plain/hmma_single` is actually used for `4096 x 8192` in
      the per-step served path.
- [x] Decision is recorded in `docs/sprints/EXPERIMENT-STATUS.md` and
      `docs/sprints/STATUS.md`.

## Promotion Rule

Promote only if the opt-in path improves the 8-slot/256K same-binary served
result by at least about 1% without token mismatches. If it regresses or stays
inside noise, keep it opt-in/off and proceed to a deeper software-pipelined
custom kernel or TurboMind persistent grouped expert work.

## Results

The path was correct and the trace confirmed it was active, but it regressed
served throughput badly:

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| Same-binary control | 262,144 | 8 | `33.502249` | `31.408359` | 8/8 token match |
| `DS4_V100_CUDA_F8_HMMA_SINGLE=1` | 262,144 | 8 | `16.083451` | `15.078235` | 8/8 token match |

Trace evidence:

- `plain/hmma_single` appeared for `4096 x 8192` on the served per-step path.
- The other hot F8 paths stayed on `plain/rows2` or grouped rows2, as intended.

Decision: keep this path opt-in/off. Single-token WMMA wastes too much work in
the 16-row token dimension, and the reduced CTA count does not compensate. The
next sprint should not broaden this to the other per-slot F8 shapes. It should
move to a real software-pipelined kernel or TurboMind persistent grouped expert
work where the tensor-core tile is filled by useful work.
