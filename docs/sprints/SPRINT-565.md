# Sprint 565 - C1 Captured-Position Scalar Localization

Date: 2026-05-29

## Goal

Find the first remaining replay divergence after Sprint 564's final-HC pointer
cache key. Do not change the served default or relax the full-capture position
key until the first bad captured position dependency is identified.

## Context

Sprint 564 proved two things:

- no-suffix full-capture cache miss must keep serving eager because CUDA stream
  capture records the graph but does not materialize response tensors;
- final-HC pointer identity is necessary defensive graph-cache metadata, but a
  remote-only relaxed-position retry still diverged on request 3.

The next likely class is a replay-critical scalar `opt.position` or derived row
argument captured into a full graph instead of read from `RankState::d_decode_position`.
The compressed-KV-off served path has fewer candidates: raw attention read
kernels already consume `d_decode_position`, while remaining host-position uses
may be logging, host metadata, prechecks, or non-served compressed-KV branches.

## Plan

1. Rebuild a remote-only relaxed-position diagnostic copy after Sprint 564.
2. Enable `--decode-stage-checksum-gate` and run the three-request same-session
   selected-token shape.
3. Compare eager vs relaxed replay stage checksums for occurrence 2, where the
   Sprint 564 relaxed retry diverged.
4. Use the first bad stage/tensor to select the minimal production repair.

## Definition of Done

- Remote diagnostic build passes.
- Stage-checksum artifacts identify the first bad stage/tensor after Sprint 564.
- If the first bad position dependency is clear, implement the minimal repair
  and validate. Otherwise record the localization and keep full capture
  position-keyed.

## Result

Ran the diagnostic as planned with a remote-only relaxed-position build in
`/workspace/s564-cache-miss-state` and artifacts under
`/workspace/s565-stage-artifacts`.

Response-level result reproduced the Sprint 564 relaxed failure:

- eager request 1: `24426` / `128829740021`
- eager request 2: `2039` / `106648190597`
- eager request 3: `117465` / `17092309830`
- relaxed replay request 1: `24426` / `128829740021`
- relaxed replay request 2: `2039` / `106648190597`
- relaxed replay request 3: `128818` / `81184816026`

Stage checksum parsing:

- occurrence 0 matched.
- occurrence 1 first sequential diff was layer 0
  `hc_current.current_full_rank_major` rank 0:
  eager `1673392381`, relaxed replay `2096617547`.
- The immediately preceding layer 0 `hc_current.current_shard` rank 0 and
  `current_full` rank 0 records matched.
- Occurrence 2 later diverged broadly, with the first sequential diff at layer
  0 `post_attention_ffn_input.route_a` rank 0.

## Decision

No production code change in this sprint. Full capture remains position-keyed.
The next repair should focus on the graph-replayed rank-major current buffer
path in HC-current: why `d_current_full_rank_major` can differ while the
adjacent current shard / slot-major current checksum still matches, and why the
divergence becomes token-visible on the following request. Do not retry
cross-position cache-key relaxation until that rank-major current path is
clean.
