# Sprint 038: Resident MTP Integrated Attention Slice

## Status

Complete.

## Overview

Sprint 038 closes the largest remaining gap inside `missing=mtp_forward` before
MTP logits/top-k: compose the resident MTP attention path from native sidecar
bytes instead of synthetic Q/KV inputs.

Sprint 037 proved raw/SWA attention and raw-cache wrap semantics from
sidecar-resident sinks. This sprint strengthens that gate into an integrated
attention slice:

- HC attention control (`hc_attn_fn`, `hc_attn_scale`, `hc_attn_base`);
- `attn_norm`;
- Q/KV projections (`attn_q_a`, `attn_q_b`, `attn_kv`);
- Q and KV RMS normalization;
- production raw-cache store;
- sink-aware attention decode;
- grouped attention output projection (`attn_output_a`, `attn_output_b`);
- HC expansion back to `[4 x 4096]`.

This still does not clear `mtp_forward`. It is the last attention-side
composition rung before adding MTP output norm/logits/top-k and draft
verify/rollback semantics.

## Use Cases

- Validate that the MTP sidecar can execute its native attention block from the
  compact gpu7 resident arena without mmap-backed sidecar tensor reads.
- Compare the resident-arena composition against the established CUDA
  mmap-backed sidecar path for the same real sidecar bytes.
- Keep the appliance gate honest: `mtp_attn` becomes an integrated attention
  proof while readiness remains blocked on `missing=mtp_forward`.

## Architecture

- Keep `docs/architecture/DS4-V100-LAYOUT.md` as the topology and memory-layout
  anchor. MTP remains a gpu7 sidecar arena, owned separately from the 8-stage
  base-model layer shards.
- Extend `tools/ds4-v100-mtp-attn-smoke.c` rather than add another narrow tool.
  The `mtp_attn` gate should prove both:
  - raw attention/cache-wrap parity from Sprint 037;
  - integrated attention projection/output parity from this sprint.
- Use two CUDA paths for integrated attention:
  - **Resident path**: arena-backed sidecar views and arena kernels.
  - **Reference path**: existing mmap-backed CUDA sidecar tensor APIs.
- Use position `0` for the integrated projection/output parity case. This keeps
  RoPE as a no-op for the first proof and isolates sidecar composition,
  resident offsets, grouped output projection, and HC expansion. Sprint 037
  already covers cache wrap semantics.
- Do not use BF16 production GEMM on V100. Source BF16 is metadata/storage only;
  this MTP sidecar path uses F32 control/norm tensors, Q8_0 projection/shared
  tensors, Q4_K routed tensors in the FFN path, FP8 raw-cache admission, and
  F16/F32 CUDA math where the existing V100 kernels require it.

## Implementation

1. Extend the MTP attention smoke with an integrated attention case.
2. Bind these real MTP sidecar tensors from the resident arena:
   - `mtp.0.hc_attn_fn.weight`
   - `mtp.0.hc_attn_scale.weight`
   - `mtp.0.hc_attn_base.weight`
   - `mtp.0.attn_norm.weight`
   - `mtp.0.attn_q_a.weight`
   - `mtp.0.attn_q_a_norm.weight`
   - `mtp.0.attn_q_b.weight`
   - `mtp.0.attn_kv_a_norm.weight`
   - `mtp.0.attn_kv.weight`
   - `mtp.0.attn_sinks.weight`
   - `mtp.0.attn_output_a.weight`
   - `mtp.0.attn_output_b.weight`
3. Run deterministic HC input through the resident attention path:
   - plain HC RMS norm;
   - HC attention function matmul;
   - HC split weighted sum;
   - attention RMS norm;
   - Q low-rank projection, Q low-rank norm, Q head projection;
   - KV projection and KV norm;
   - raw-cache store;
   - attention decode;
   - grouped output projection;
   - HC expansion.
4. Run the same deterministic HC input through the mmap-backed CUDA reference
   path and compare intermediate and final tensors with explicit tolerances.
5. Keep the full gate readiness order unchanged except that `mtp_attn` now
   means integrated attention is proven.
6. Update the sprint report, follow-ups, and `docs/sprints/VISION.md`.

## Files Summary

- `tools/ds4-v100-mtp-attn-smoke.c`
- `tools/ds4-v100-gate.sh`
- `Makefile`
- `docs/sprints/SPRINT-038-REPORT.md`
- `docs/sprints/SPRINT-038-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

## Definition of Done

- Local compile checks pass for the changed MTP attention smoke and CPU stubs.
- CUDA build on the V100 cluster passes for
  `tools/ds4-v100-mtp-attn-smoke`.
- Focused MTP attention smoke passes on gpu7 with all 8 V100s visible and
  reports both raw attention/cache-wrap parity and integrated attention parity.
- Full V100 gate includes `mtp_attn PASS`, has no failures, and still reports
  `ready=false missing=mtp_forward`.
- Sprint report records commands, outputs, tolerances, and remaining blockers.
- Vision document names Sprint 038 and updates the readiness ladder.

## Results

- Local C/stub build passed:
  `make tools/ds4-v100-mtp-attn-smoke.o ds4_v100_mtp.o ds4_gpu_arena_stub.o ds4_cpu.o`
- Local shell/diff checks passed:
  `bash -n tools/ds4-v100-gate.sh`, `git diff --check`
- Cluster CUDA build passed:
  `CUDA_ARCH=sm_70 make tools/ds4-v100-mtp-attn-smoke`
- Focused cluster smoke passed:
  `mtp_attn_smoke PASS`
- Focused integrated attention evidence:
  - `q_heads max_abs=2.14576721e-06`
  - `kv_row max_abs=0.000867605209`
  - `heads max_abs=2.14576721e-06`
  - `attn_out max_abs=0.258209229`
  - `next_hc max_abs=0.19461441`
- Full 8-GPU gate passed:
  `gate summary PASS failures=0 ready=false`
- Readiness remains intentionally blocked on:
  `missing=mtp_forward`

## Risks

- The integrated parity check compares resident arena composition to the
  established mmap-backed CUDA path, not a full CPU semantic oracle. This is
  appropriate for residency/composition risk, but logits/top-k must still use a
  trusted oracle in the next sprint.
- Grouped attention output can hide row-stride mistakes if dimensions are only
  checked at the final HC output. The smoke should compare at least `q`, `kv`,
  `heads`, `attn_out`, and `next_hc`.
- Position-0 RoPE isolation is deliberate. Later MTP forward work still needs
  native decode-position coverage once logits/top-k and draft verification are
  connected.

## Security

No public serving surface changes. This sprint only strengthens local CUDA
smoke coverage and the internal appliance gate.

## Dependencies

- Sprint 031 MTP sidecar residency.
- Sprint 034 resident prefix composition.
- Sprint 036 resident MTP FFN slice.
- Sprint 037 resident raw attention and cache-wrap proof.
- Real MTP sidecar model at the cluster path used by prior gates.

## Open Questions

- Whether Sprint 039 should clear `mtp_forward` by adding output norm/logits
  top-k first, or by first wiring prefix + attention + FFN into one resident
  MTP block state object.
- Whether draft verify/rollback should begin as a focused state-management
  smoke before MTP logits are connected to the replay server.
