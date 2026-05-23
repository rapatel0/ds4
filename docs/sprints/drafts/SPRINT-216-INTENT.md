# Sprint 216 Intent - True MTP Speculative Verification Gate

Date: 2026-05-23

## Seed Prompt

Continue toward the high-throughput practical-serving vision after Sprint 215
showed the current best deployable mode is `32` slots at `128K`, while MTP is
not yet a speedup.

## Orientation Summary

- Sprint 215 measured production serving against the persistent TurboMind pack:
  `32` slots at `128K` is best today at `68.403129` continuation tok/s, and
  `16` slots at `256K` remains the maximum-context mode at `61.624766`
  continuation tok/s.
- `32` slots at `256K` still fails closed at the launcher cap.
- MTP verify is compatible but not useful for throughput: `0/16` accepts in the
  matrix and only `12.279921` continuation tok/s.
- One-slot MTP commit accepted `8/15` drafts, but still only reached
  `7.846341` continuation tok/s. That confirms the current commit path is not a
  real speculative speedup path.
- The next sprint should answer whether true target verification over drafted
  tokens can be implemented against the current replay/runtime APIs without a
  scheduler rewrite.

## Vision Context

The project needs a material multiplier, not another sub-10% wrapper tweak.
MTP remains the most plausible 2x-class lever if it can batch target
verification over drafted tokens and advance KV/cache state for accepted
drafts. Sprint 216 should build the smallest correct target-verification gate
before any broad production integration.

## Relevant Code Areas

- `tools/ds4-v100-replay.c`
- `ds4_v100_replay.c`
- `ds4_v100_replay.h`
- `ds4_v100_mtp.c`
- `tools/ds4-v100-sustained-decode-bench.sh`
- `tools/ds4-v100-practical-serving-matrix.sh`
- `docs/operations/DS4-V100-APPLIANCE.md`

## Constraints

- Preserve base token correctness.
- Keep MTP default off.
- Do not fake speedup by counting accepted drafts while still doing the same
  number of target forwards.
- Do not introduce a generic scheduler abstraction.
- If replay cannot verify multiple drafted tokens in one target batch, prove
  that with a focused harness and pivot.

## Success Criteria

- A focused MTP speculative verification harness exists.
- The harness reports:
  - draft tokens proposed;
  - target tokens verified;
  - accepted prefix length;
  - target forwards performed;
  - effective output tokens;
  - effective tok/s versus normal one-slot generation.
- If the current runtime cannot batch target verification over drafted tokens,
  the sprint records the exact API/runtime gap.
- If the gate passes, the next sprint can safely integrate MTP speculative mode
  into serving behind an explicit flag.

## Verification Strategy

- Build on V100 if C code changes.
- Run one-slot 256K focused tests with the production pack and MTP sidecar.
- Compare emitted token sequence against base replay for the same prompt and
  token count.
- Record prompt/prefill, continuation, draft, accepted-prefix, and effective
  tok/s separately.

## Uncertainty Assessment

- Correctness: High. Speculative decoding changes token/state semantics.
- Scope: Medium-high. The sprint should be a gate/harness first, not full
  production rollout.
- Architecture: Medium. The replay API may lack a multi-position target
  verification primitive.

## Open Questions

- Can `ds4_v100_replay` feed a drafted token block and verify target logits in
  one target pass, or only one token at a time?
- Does the MTP sidecar support drafting more than one token per target state, or
  only one next-token proposal?
- How should accepted drafted tokens advance scheduler/KV state without
  replaying the same work serially?
