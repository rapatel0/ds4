# Sprint 071: Exact MTP Commit Serving

## Status

Complete.

## Overview

Sprint 070 made the MTP forward runtime persistent, but MTP serving is still
diagnostic: it runs after normal target generation, compares the draft to a
target token that was already produced, and never changes the target replay
state.

Sprint 071 moves MTP to the generation path for the one-slot appliance mode.
The first version should remain exact-verified for safety: the target model
still selects the verification token, and an accepted MTP draft is emitted and
then fed as the next committed token. This establishes the state mutation and
serving contract needed before any unsafe/skip-verify or multi-token MTP
optimization.

## Goals

1. Add the smallest public replay hooks needed for a controlled one-slot
   generation loop:
   - feed a token at a position;
   - select the current target token;
   - initialize/finalize counters without bypassing reset semantics.
2. Add `--mtp-serving commit` alongside `off` and `verify`.
3. In commit mode, run one-slot generation as:
   - replay prompt tokens through the target model;
   - select the first target token normally;
   - feed each committed token through the target model;
   - run MTP from the committed token embedding plus post-commit target HC;
   - select the target verifier token;
   - if draft top-1 equals target top-1, emit the draft as committed;
   - otherwise emit the target token and count the rejection.
4. Keep commit mode exact-verified and one-slot only.
5. Surface commit-mode counters in JSON/status/metrics.
6. Validate on the V100 cluster with the real base model, real MTP sidecar, and
   real pack index.
7. Prove the post-commit output sequence matches the baseline verified target
   sequence on the fixture.

## Non-Goals

- Skipping target verification.
- Recursive or K>1 speculative drafting.
- Multi-slot MTP serving.
- Enabling MTP commit by default in the practical 4-slot appliance profile.
- Rewriting MTP kernels or changing draft logits.
- Claiming throughput uplift from exact-verify commit; this sprint establishes
  the state-mutating contract first.

## Implementation

1. Extend `ds4_v100_replay.h/.c` with narrow one-slot incremental APIs. Keep
   the existing `ds4_v100_replay_generate` and batch paths unchanged.
2. Factor the existing MTP draft helper in `tools/ds4-v100-replay.c` so both
   verify mode and commit mode can call the same draft primitive.
3. Add a commit-mode one-slot generator in `tools/ds4-v100-replay.c`.
4. Route CLI and HTTP one-slot requests through commit-mode generation when
   `--mtp-serving commit` is selected.
5. Keep status mode names explicit: `mtp_verify_one_slot` and
   `mtp_commit_one_slot`.
6. Update `tools/ds4-v100-mtp-serving-smoke.sh` or add a companion smoke so
   cluster validation can run both verify and commit mode.
7. Add Sprint 071 report and update `docs/sprints/VISION.md`.

## Parallel Work

- API review explorer: confirm the smallest public replay hooks and any state
  safety risks.
- Validation review explorer: identify the baseline-vs-commit evidence needed
  to prove exact commit correctness without treating identical first-token hex
  as sufficient.

## Definition of Done

- [x] Local compile passes for changed C files.
- [x] `git diff --check` passes.
- [x] `--mtp-serving commit` is accepted by CLI parsing and rejects
  `--active-microbatch != 1`.
- [x] `/v100/status` reports `mode="mtp_commit_one_slot"` and
  `mtp.serving_mode="commit"` in commit mode.
- [x] Commit-mode JSON reports attempted/accepted/rejected commit counters.
- [x] V100 build passes for `tools/ds4-v100-replay` and the MTP smoke targets.
- [x] V100 commit-mode serving smoke returns first-token hex `3136`, accepts
  the fixture draft, and reports at least one committed MTP token.
- [x] Baseline-vs-commit evidence proves the emitted token sequence is
  unchanged on the fixture.
- [x] Sprint report records commands, outputs, artifacts, timing, and the next
  throughput decision.
- [x] Vision document is updated.

## Outcome

`SHIP`.

`--mtp-serving commit` is now an exact-verified one-slot serving mode. It feeds
prompt tokens through the target model, emits the first target token normally,
then verifies each MTP draft against the target verifier token before emitting
the accepted draft as the committed output.

The V100 commit smoke accepted `2/2` drafts, reported
`mode="mtp_commit_one_slot"`, `mtp.serving_mode="commit"`, and
`mtp.committed=2`, and produced the same emitted token sequence as the verify
baseline: `[926, 1]`. This is correctness/state-mutation progress, not a
throughput win yet, because commit mode still runs exact target verification.

## Risks

- Exact verification may not improve throughput because it still computes the
  target verifier token. This sprint is about correctness and state mutation.
- Feeding an accepted draft at the wrong position would silently corrupt KV/HC
  state. Validation must check more than the first generated token.
- The current request scheduler resets replay state per request; commit mode
  should therefore focus on within-request state mutation, not cross-request
  continuation.
- MTP remains one-slot because the MTP service has shared scratch and one raw
  cache.

## Security

No new external exposure. Commit mode remains opt-in through the existing
loopback appliance endpoint and inherits the one-slot MTP guard.
