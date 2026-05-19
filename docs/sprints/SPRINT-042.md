# Sprint 042: Native Prompt-Token MTP Verify

## Status

Complete.

## Overview

Sprint 042 moves the MTP path from deterministic resident forward plus
rollback safety to a native prompt-token verify gate. The MTP sidecar must
consume the actual just-committed target token embedding and the target
scheduler HC state after that token is committed, produce a real one-token
draft, and compare that draft to the target model's next top-1 token by exact
token equality.

This sprint does not optimize throughput or expose public speculative serving.
It proves the correctness boundary needed before those changes are safe.

## Use Cases

- Prove the resident MTP sidecar can draft from real target decode state rather
  than deterministic synthetic inputs.
- Prove exact token-equality verification against the target model is wired
  into the V100 gate.
- Keep target rollback/state safety from Sprint 041 intact for rejected drafts.
- Advance readiness from `missing=mtp_verify` to the next real blocker.

## Architecture

- Keep the base model topology from `docs/architecture/DS4-V100-LAYOUT.md`:
  layers remain contiguous and stage-owned across the 8 V100s, the final HC
  state and output head remain on gpu7, and the MTP sidecar remains resident on
  gpu7.
- Use the existing scheduler sequence to commit token `T`:
  - gpu0 decodes from `token_embd.weight`;
  - gpu1-gpu7 receive HC handoffs and decode stage-local layers;
  - gpu7 owns the post-commit target HC and output-head top-k.
- Add a narrow embedding read helper for the committed token:
  `token_embd.weight` BF16 row -> F32[4096]. This is a 16 KiB smoke-path host
  hop and is acceptable for the verify gate; production can later replace it
  with a device-local gather on gpu7.
- Reuse the resident one-token MTP forward composition from Sprint 040, but feed
  it real `embed(T)` and post-commit target HC. The MTP raw cache should follow
  DS4 semantics for the first draft: `n_raw = 1`, row `pos % 128`.
- Preserve Sprint 041 rollback checks. A rejected real draft must restore the
  target scheduler state exactly and leave only accepted MTP raw visibility.

## Parallel Work

This sprint is designed for parallel subagents:

- MTP forward extraction/reuse: identify the smallest safe way to reuse the
  resident MTP forward path with caller-provided embedding and HC.
- Scheduler-state boundary: verify HC timing and embedding source semantics
  against native `ds4.c`.
- Cluster validation: run focused one- or two-card compile/probe work in
  parallel where possible, then run the full 8-GPU gate once the local build is
  clean.

## Implementation

1. Add a scheduler/API helper to read a committed token embedding as F32 from
   the BF16 source table with descriptor validation.
2. Refactor or reuse the resident MTP forward smoke so the verify smoke can run
   one draft from caller-provided `embed[4096]`, `prev_hc[4][4096]`, and
   position.
3. Extend `tools/ds4-v100-mtp-verify-smoke.c`:
   - replay the prompt through the target scheduler;
   - select and commit target token `T`;
   - read `embed(T)` and gpu7 post-commit HC;
   - run resident MTP forward from those inputs;
   - compare MTP top-1 with target post-`T` top-1 by exact token equality;
   - exercise accept or reject transition without corrupting target state.
4. Split gate semantics:
   - keep `mtp_rollback` as the state-safety proof;
   - add `mtp_verify` as the native prompt-token draft/verify proof.
5. Update Makefile dependencies, gate readiness logic, Sprint 042 report,
   follow-ups, and vision.

## Files Summary

- `ds4_v100_scheduler.h`
- `ds4_v100_scheduler.c`
- `tools/ds4-v100-mtp-forward-smoke.c`
- `tools/ds4-v100-mtp-verify-smoke.c`
- `tools/ds4-v100-gate.sh`
- `Makefile`
- `docs/sprints/SPRINT-042-REPORT.md`
- `docs/sprints/SPRINT-042-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

## Definition of Done

- Local object/stub compile passes for the scheduler helper and MTP verify
  smoke.
- CUDA build on the V100 cluster passes for
  `tools/ds4-v100-mtp-verify-smoke`.
- Focused native `mtp_verify` smoke runs on the 8x V100 pod with the real base
  model, real MTP sidecar, and real pack index.
- The smoke report records prompt length, committed token, committed position,
  target top-1, MTP draft top-1, exact equality decision, MTP raw row, snapshot
  bytes, and rollback parity.
- Full V100 gate includes `mtp_verify PASS`, has no failures, and no longer
  reports `missing=mtp_verify`.
- Sprint report records commands, outputs, token decisions, and remaining
  readiness blockers.
- Vision document updates Sprint 042 and the readiness ladder.

## Risks

- The first real MTP draft may not equal the target top-1 for the chosen prompt.
  That is still a correctness-safe reject, but it may not prove a positive
  speculative accept. If this happens, record it honestly and add a follow-up
  to search for an accept fixture.
- The smoke-path embedding read uses host BF16 decode. This is fine for
  correctness evidence but not the production low-movement path.
- Reusing the forward smoke may leave some tool-level code duplication. A
  production MTP runtime object should be extracted after correctness is proven.

## Security

No new public serving surface. This sprint only adds internal CUDA smoke/gate
coverage and readiness reporting.

## Dependencies

- Sprint 040 resident one-token MTP forward composition.
- Sprint 041 scheduler snapshot/rollback safety.
- Real base model, MTP sidecar, pack index, and 8x V100 cluster access.

## Open Questions

- If the short prompt yields a real MTP reject, should the next sprint search
  for a positive accept fixture before public speculative serving?
- Should production MTP embedding gather be a scheduler API or a dedicated MTP
  runtime API on gpu7?
