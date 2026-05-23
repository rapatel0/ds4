# Sprint 192 - Single-Slot Attention Output-A HMMA Probe

Date: 2026-05-22

## Objective

Test a tensor-core route for the largest filled-context attention sub-bucket:
single-slot `attention_output_a`.

## Rationale

Sprint 191 showed that filled-context direct replay is attention dominated.
For len-1024 / ctx-262144, attention consumed `37195.555 ms` out of
`66127.631 ms` profiled stage time. Within attention, output projection was the
largest sub-bucket (`15478.988 ms`, `41.62%` of attention). The existing
single-slot path calls the grouped scalar F8 row-pair kernel for
`attention_output_a`, while a DS4-shaped grouped-batch HMMA kernel already
exists for the 16-slot batch path.

This sprint adds an explicit default-off single-token route through the
existing HMMA grouped-batch kernel. The goal is to determine whether spending
extra inactive token lanes on Volta tensor cores beats the current scalar
row-pair kernel for the actual single-slot filled-context shape.

## Scope

- Add a default-off single-slot attention output-A HMMA selector.
- Reuse the existing grouped-batch HMMA kernel with `n_tokens=1`.
- Keep the existing row-pair grouped output path as the default.
- Validate correctness and timing on V100 direct synthetic len-256 and
  len-1024 tiers.
- Record whether the path should be promoted, kept diagnostic-only, or
  rejected.

## Non-Goals

- No default serving behavior change unless the A/B result is clearly positive.
- No attention output-B rewrite.
- No q/kv projection rewrite.
- No TP/EP topology change.
- No MTP serving change.

## Implementation Plan

1. Add CUDA support for `DS4_CUDA_F8_HMMA_GROUPED_ATTN_O_SINGLE=1` so
   `ds4_gpu_arena_f8_e4m3_b128_matmul_grouped_batch_f32()` can launch the
   DS4 HMMA grouped attention-output kernel with `n_tokens=1`.
2. Add runtime selector `DS4_V100_SINGLE_SLOT_ATTN_OUTPUT_A_HMMA=1` in the
   layer executor's single-slot grouped attention output-A path.
3. Wire launcher/env-example validation and export for the new flags.
4. Build on the V100 pod.
5. Run direct synthetic A/B:
   - len-256 / ctx-262144 / tokens=2
   - len-1024 / ctx-262144 / tokens=2
   - include `DS4_V100_PROFILE_ATTENTION_DETAIL=1` so `attn_output` is
     visible.

## Definition of Done

- [x] V100 build passes.
- [x] Control and HMMA candidate preserve output IDs for len-256 and len-1024.
- [x] Timing evidence is archived under `logs/from-cluster/`.
- [x] Sprint outcome records whether the candidate improves `attn_output` and
  continuation tok/s enough to promote.
- [x] Vision is updated.
- [x] Changes are committed.

## Implementation

- Added default-off `DS4_V100_SINGLE_SLOT_ATTN_OUTPUT_A_HMMA=1` in the
  single-slot grouped attention output-A path.
- Added default-off `DS4_CUDA_F8_HMMA_GROUPED_ATTN_O_SINGLE=1` so the existing
  DS4 grouped-batch HMMA attention-output kernel can be selected for
  `n_tokens=1`.
- Wired launcher and env-example validation/export for both flags.
- Kept the production default on the current grouped row-pair F8 path.

## V100 Evidence

Build:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

Direct synthetic A/B on the persistent Sprint 181 production pack, with
`DS4_V100_PROFILE_ATTENTION_DETAIL=1`:

| Prompt len | Mode | Prompt tok/s | Continuation tok/s | Output IDs | Attention ms | Attn output ms | Total profile ms |
|---:|---|---:|---:|---|---:|---:|---:|
| 256 | control | 13.925948 | 14.722744 | `3955, 361` | 9031.833 | 3876.448 | 16607.744 |
| 256 | HMMA single | 8.535002 | 8.842737 | `3955, 361` | 20700.255 | 15519.781 | 28283.033 |
| 1024 | control | 14.476492 | 14.577239 | `926, 926` | 37021.986 | 15473.716 | 65809.171 |
| 1024 | HMMA single | 8.710177 | 8.747221 | `926, 926` | 83467.613 | 61889.994 | 112344.903 |

Evidence:

- `logs/from-cluster/sprint192-attn-o-hmma/len256-control/result.json`
- `logs/from-cluster/sprint192-attn-o-hmma/len256-control/summary.json`
- `logs/from-cluster/sprint192-attn-o-hmma/len256-hmma/result.json`
- `logs/from-cluster/sprint192-attn-o-hmma/len256-hmma/summary.json`
- `logs/from-cluster/sprint192-attn-o-hmma/len1024-control/result.json`
- `logs/from-cluster/sprint192-attn-o-hmma/len1024-control/summary.json`
- `logs/from-cluster/sprint192-attn-o-hmma/len1024-hmma/result.json`
- `logs/from-cluster/sprint192-attn-o-hmma/len1024-hmma/summary.json`

## Outcome

Rejected for production. The candidate preserved output IDs but made the
single-slot attention-output bucket roughly `4x` slower and continuation
throughput roughly `40%` lower on both filled-context tiers.

The result is specific: the existing 16-slot grouped-batch HMMA kernel is the
wrong shape for single-token decode. It does not disprove tensor-core execution
or fusion generally. It does show that the next sprint should stop trying
single-kernel substitutions and instead implement a larger fused/persistent
attention boundary that avoids repeated F32 materialization and launch overhead.

## Follow-Up

Sprint 193 should target a larger attention projection/output boundary. The
candidate direction is a fused grouped attention-output path that keeps
low-precision source bytes in the kernel and expands inside GPU registers, or a
persistent TP/EP ownership prototype that changes the shape enough to make
tensor cores useful without per-layer full-hidden copy-back.
