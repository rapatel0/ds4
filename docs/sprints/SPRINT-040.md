# Sprint 040: Resident One-Token MTP Forward Composition

## Status

Complete.

## Overview

Sprint 040 should compose the MTP sidecar primitives from Sprints 034-039 into
one resident gpu7 forward path:

- prefix composition from deterministic embedding plus previous HC;
- integrated MTP attention with raw-cache update;
- resident MTP FFN;
- MTP output HC collapse, norm, base vocabulary projection, and top-k.

The goal is to prove a continuous one-token MTP forward surface, not yet
production speculative decoding. Draft verify/rollback against target-model
state remains the next sprint unless the implementation turns out to be small
enough to include without risking correctness.

## Use Cases

- Prove the resident MTP sidecar can execute a complete draft-candidate forward
  pass instead of only independent slices.
- Catch boundary errors between prefix output, attention output, FFN output,
  and logits/top-k.
- Advance the gate from primitive MTP coverage to a full `mtp_forward` proof
  while keeping readiness blocked on state-safe verification.

## Architecture

- Keep `docs/architecture/DS4-V100-LAYOUT.md` as the topology anchor. Sprint
  040 remains gpu7-sidecar focused and does not change base-model sharding.
- Add a focused `tools/ds4-v100-mtp-forward-smoke.c` that can own the continuous
  composition without destabilizing the smaller primitive smokes.
- Reuse the exact kernel families already validated:
  - F32 arena RMSNorm and HC control for prefix/attention/FFN/output head;
  - Q8_0 arena matmul for prefix projections, attention projections, shared
    FFN, grouped attention output, and output logits where applicable;
  - Q4_K arena routed MoE for MTP experts;
  - arena attention decode and production raw KV store for MTP SWA attention;
  - BF16 base output projection from the current V100 pack index.
- Use deterministic input first:
  - deterministic base embedding `[4096]`;
  - deterministic previous HC `[4 x 4096]`;
  - MTP raw cache starts empty at position 0.
  This isolates resident composition and CPU/GPU parity. Native prompt-token
  oracle wiring can follow once the continuous path is stable.
- CPU reference should be sidecar/base-byte based, not a synthetic proxy. It
  should compute the same prefix, attention, FFN, output norm, logits, and top-k.

## Parallel Work

This sprint is suitable for parallel subagents:

- Explorer A: map reusable code and helper extraction from existing MTP smokes.
- Explorer B: define the best native `ds4.c` oracle and where deterministic
  composition ends versus draft/verify semantics.
- Main implementation: add the forward smoke, wire the gate, and validate.

## Implementation

1. Add `tools/ds4-v100-mtp-forward-smoke.c`.
2. Bind all MTP tensors required by prefix, attention, FFN, and logits.
3. Upload the base BF16 `output.weight` into a gpu7 output-head arena, matching
   Sprint 039.
4. Run resident GPU composition:
   - prefix: `enorm/e_proj/repeat + hnorm/h_proj`;
   - attention: HC control, attention norm, Q/KV projections, raw-cache store,
     attention decode, grouped output, HC expand;
   - FFN: HC control, FFN norm, bias router, Q4_K routed experts, Q8_0 shared
     expert, routed+shared add, HC expand;
   - logits: MTP HC head, MTP output norm, base output projection, top-k.
5. Run CPU sidecar/base-byte reference for the same deterministic input.
6. Compare intermediate boundaries and final top-k:
   - `mtp_input_hc`;
   - `attn_next_hc`;
   - `ffn_next_hc`;
   - top-k draft candidates and selected logits.
7. Add `mtp_forward` to `tools/ds4-v100-gate.sh` after `mtp_logits`.
8. Update readiness to report the next blocker as `mtp_verify` only if the
   continuous forward smoke passes. Do not claim production readiness.
9. Update report, follow-ups, and vision.

## Files Summary

- `tools/ds4-v100-mtp-forward-smoke.c`
- `tools/ds4-v100-gate.sh`
- `Makefile`
- `docs/sprints/SPRINT-040-REPORT.md`
- `docs/sprints/SPRINT-040-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

## Definition of Done

- Local object/stub compile passes for the new forward smoke.
- CUDA build on the V100 cluster passes for
  `tools/ds4-v100-mtp-forward-smoke`.
- Focused MTP forward smoke passes on gpu7 with all 8 V100s visible, the real
  base model, the real MTP sidecar, and the real pack index.
- The smoke reports `mtp_input_hc`, `attn_next_hc`, `ffn_next_hc`, and final
  top-k parity against CPU reference.
- Full V100 gate includes `mtp_forward PASS`, has no failures, and reports the
  next honest blocker, expected to be `missing=mtp_verify`.
- Sprint report records commands, outputs, tolerances, top-k tokens/logits,
  memory reserve, and remaining blockers.
- Vision document updates Level 3 readiness and the sprint sequence.

## Risks

- This will duplicate helper logic unless we either extract shared MTP smoke
  helpers or accept one integration tool with copied CPU reference code. Favor
  correctness and explicitness first; refactor only after the continuous path
  is green.
- CPU serial accumulation and V100 warp reductions already required wider
  tolerances in grouped attention output. The forward smoke should report
  boundary-specific tolerances instead of hiding everything behind final top-k.
- A deterministic input forward smoke is not the same as speculative serving.
  It does not prove target-model verification, partial accept, rollback, or MTP
  raw-cache state safety.

## Security

No public serving surface changes. This sprint adds an internal CUDA smoke and
gate readiness rung.

## Dependencies

- Sprint 034 resident prefix composition.
- Sprint 036 resident MTP FFN slice.
- Sprint 038 resident integrated MTP attention.
- Sprint 039 resident MTP logits/top-k parity.
- Real base model, MTP sidecar, and pack index on the V100 cluster.

## Open Questions

- Resolved for this sprint: the first full forward smoke uses deterministic
  embedding plus deterministic previous HC. Native prompt-token draft/verify
  remains the next readiness rung.
- Resolved for this sprint: composition stays tool-local until verify/rollback
  semantics define the reusable runtime surface.

## Results

- Added `tools/ds4-v100-mtp-forward-smoke.c`, a resident gpu7 MTP forward
  composition smoke that runs prefix, attention, FFN, MTP output HC collapse,
  output norm, base BF16 output projection, and top-k.
- Wired `mtp_forward` into `tools/ds4-v100-gate.sh` after `mtp_logits`.
- Updated readiness so a passing composed forward smoke advances the MTP blocker
  to `missing=mtp_verify`, not `ready=true`.
- Focused V100 smoke passed on the real base model, real MTP sidecar, real pack
  index, gpu7, and all 8 visible V100s:
  - `prefix_hc max_abs=0.000127315521`
  - `attn_next_hc max_abs=0.327054977`
  - `ffn_next_hc max_abs=0.959003448`
  - exact top-5 token parity: `101365,40810,102216,7178,112542`
  - `logit_max_abs=0.0884904861`
- Full V100 gate passed with `failures=0 ready=false missing=mtp_verify`.
