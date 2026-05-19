# Sprint 041: MTP Rollback State Safety

## Status

Complete.

## Overview

Sprint 041 moved the V100 appliance from deterministic `mtp_forward` only to a
gated rollback/state-safety proof. It did not claim native prompt-token
`mtp_verify`; the full gate now passes `mtp_rollback` and still honestly
reports `missing=mtp_verify`.

The key semantic rule is unchanged from native DS4: MTP is only a drafter. The
target model decides acceptance by exact token equality. Numeric closeness is
diagnostic evidence only, never an accept condition.

## Use Cases

- Prove target scheduler state can be snapshotted and restored after a rejected
  speculative decode.
- Prove MTP raw-cache visibility can discard speculative rows and keep only the
  accepted prefix.
- Keep the readiness gate honest: rollback state safety is now proven, while
  native prompt-token MTP verification remains the next blocker.

## Architecture

- Keep `docs/architecture/DS4-V100-LAYOUT.md` as the topology anchor. This
  sprint does not change sharding or base model layer ownership.
- Add a narrow V100 scheduler snapshot API around mutable target state:
  - current HC tensor contents and active HC buffer identity;
  - per-layer raw KV tensors;
  - per-layer attention compressor state/cache tensors and counters;
  - per-layer indexer compressor state/cache/top-k tensors and counters.
- Keep immutable model weights and MTP sidecar weights out of snapshots.
- Add a tool-level MTP rollback smoke rather than a public serving change first.
  Production helper extraction should wait until this sprint proves the state
  ownership boundary.

## Semantics To Prove

1. Replay a real prompt through the base V100 scheduler.
2. Select the next target token `T`.
3. Commit `T` through the ordinary target scheduler path.
4. Record target top-1 for the post-`T` prefix.
5. Force a reject path with a deliberately wrong draft token.
6. Restore target and synthetic MTP raw-cache visibility state.
7. Continue ordinary target decode and prove it matches a clean replay of the
   same canonical token stream.

Native prompt-token MTP input and real MTP draft acceptance remain Sprint 042
scope. This sprint deliberately stopped at the state-safety primitive once the
proxy nature of the draft source was identified.

## Parallel Work

This sprint was planned with two read-only subagents:

- Replay/state explorer: mapped target scheduler state and the smallest
  snapshot/restore API.
- Native-semantics explorer: mapped DS4 speculative acceptance semantics and
  evidence requirements.

## Implementation

1. Add `ds4_v100_stage_scheduler_snapshot` APIs:
   - create/capture;
   - restore;
   - free;
   - optional report of captured byte count and layer count.
2. Add focused scheduler snapshot tests:
   - capture after a prompt-prefix replay;
   - mutate by decoding one extra token;
   - restore;
   - compare selected top-k or final HC against a clean replay.
3. Add `tools/ds4-v100-mtp-verify-smoke.c` as the `mtp_rollback` gate:
   - open base replay/runtime and MTP sidecar;
   - replay `short_reasoning_plain`;
   - commit the target token through the normal V100 path;
   - force reject/restore;
   - decode the canonical next token and compare against restored replay.
4. Report:
   - prompt token count;
   - committed target token;
   - rejected draft token;
   - target top-1;
   - accept/reject decision;
   - snapshot byte count;
   - per-stage restore status;
   - MTP raw visibility before/after;
   - clean-control token/logit parity.
5. Add `mtp_rollback` to `tools/ds4-v100-gate.sh` after `mtp_forward`.
6. Update readiness so a passing rollback smoke moves the blocker from
   `mtp_rollback` to the still-missing native `mtp_verify`.
7. Update report, follow-ups, and vision.

## Files Summary

- `ds4_v100_scheduler.h`
- `ds4_v100_scheduler.c`
- `tests/cuda_v100_scheduler_snapshot_smoke.c`
- `tools/ds4-v100-mtp-verify-smoke.c`
- `tools/ds4-v100-gate.sh`
- `Makefile`
- `docs/sprints/SPRINT-041-REPORT.md`
- `docs/sprints/SPRINT-041-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

## Definition of Done

- Local object/stub compile passes for the scheduler snapshot API and new smoke.
- CUDA build on the V100 cluster passes for the new snapshot test and
  `tools/ds4-v100-mtp-verify-smoke`.
- Focused scheduler snapshot smoke passes on the V100 cluster.
- Focused MTP rollback smoke passes on gpu7 with all 8 V100s visible, the real
  base model, real MTP sidecar, and real pack index.
- Full V100 gate includes `mtp_rollback PASS`, has no failures, and reports
  `missing=mtp_verify`.
- Sprint report records commands, outputs, snapshot bytes, token decisions,
  parity evidence, and remaining blockers.
- Vision document updates the Level 3 readiness ladder and sprint sequence.

## Risks

- Token-only rollback is insufficient. The target backend state includes raw KV,
  compressed rows, indexer rows, compressor state, and current HC.
- Snapshotting all mutable target tensors is heavy. This is acceptable for a
  correctness smoke but should not become the production fast path.
- Synthetic rejected-draft state proves rollback state safety but does not
  prove native prompt-token MTP quality. Native prompt-token wiring is Sprint
  042.
- Exact token equality is the only valid accept condition.

## Security

No public serving changes. This sprint adds internal CUDA snapshot/verify
smokes and gate readiness wiring.

## Dependencies

- Sprint 040 resident one-token MTP forward composition.
- Real base model, MTP sidecar, and pack index on the V100 cluster.

## Open Questions

- Does target scheduler snapshot/restore need a public runtime API immediately,
  or can it remain internal/test-oriented until production speculative serving?
- Can native prompt-token MTP input reuse the existing replay final HC directly,
  or does it require a separate MTP draft-session object first?
