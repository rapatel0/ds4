# Sprint 563 - C1 Residual Captured-Position State Localization

Date: 2026-05-29

## Goal

Localize the first stage that diverges when no-suffix full-capture graphs are
reused across positions. Sprint 562 proved the cache-key relaxation is still
unsafe, so this sprint must identify the remaining captured position-derived
state before any further cache-key retry.

## Context

Sprint 562 tested no-suffix full-capture cross-position reuse after the
Sprint 561 topology work. A two-request same-session gate matched, but a
six-request same-session gate diverged on request three:

- eager: token/checksum `117465` / `17092309830`
- replay: token/checksum `2039` / `110810249310`

The candidate was removed. Full capture remains position-keyed. Claude
bug-find review flagged likely hidden captured position state such as
host-baked compressed-row indices or other position-derived kernel arguments.

## Constraints

- No permanent new CLI/env flag.
- Do not touch MTP.
- Do not re-land the failed cache-key relaxation as production code.
- Temporary remote-only diagnostic patches are allowed if they are not
  committed.
- Use stage checksums and existing diagnostics before adding new instrumentation.

## Plan

1. Create a temporary remote validation copy with the Sprint 562 relaxation
   re-applied only for diagnostics.
2. Enable `--decode-stage-checksum-gate` for both eager and relaxed full-capture
   same-session six-request runs.
3. Compare stage checksum streams by request occurrence, layer, stage, tensor,
   and rank.
4. Identify the earliest stage/tensor/rank where replay diverges from eager on
   the request-three failure.
5. Record the concrete blocker and the next implementation sprint. If the
   first divergent stage is a host-baked row/source argument with a narrow
   device-position fix, plan that fix next; otherwise keep localizing.

## Definition of Done

- [x] Temporary remote diagnostic build/run completes or produces a concrete
  blocker.
- [x] The first divergent stage is identified from authoritative logs, or the
  sprint records why the existing stage checksums are insufficient.
- [x] Promoted source code remains unchanged by the failed Sprint 562 candidate.
- [x] `SPIKE_B_STEERING.md` and `docs/sprints/VISION.md` are updated with the
  localization result and next ordered item.
- [x] No temporary diagnostic source, binary, or production flag is committed.

## Diagnostic Run

Temporary remote copy:

- `/workspace/s563-localize`
- The Sprint 562 cache-key relaxation was re-applied only in this remote copy.
- Local source kept the rejected candidate removed.

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`

Stage-checksum probe:

- Eager artifact:
  `/workspace/s563-localize-artifacts/s563-stage-eager`
- Relaxed full-capture artifact:
  `/workspace/s563-localize-artifacts/s563-stage-fullgraph`
- Both used same-session selected-token requests with
  `--decode-stage-checksum-gate`.

Response-level result:

- Request 1 matched:
  - token/checksum `24426` / `128829740021`
- Request 2 diverged under stage checksums:
  - eager token/checksum `2039` / `106648190597`
  - relaxed replay token/checksum `50845` / `114177715767`
- Request 3 also diverged:
  - eager token/checksum `117465` / `17092309830`
  - relaxed replay token/checksum `37102` / `78382880974`

Stage-checksum localization:

- Occurrence 0 matched across all logged stage tensors.
- Occurrence 1 first diverged at layer `1`, stage `hc_current`.
- Layer `0` had zero occurrence-1 checksum diffs.
- Layer `1` had immediate `hc_current` diffs across ranks, including:
  - `current_shard` rank 0: eager `255270109`, replay `252329227`
  - `current_full` rank 0: eager `2061267831`, replay `2071866390`
  - `current_full_rank_major` rank 0: eager `2041173666`, replay `2024541679`
  - `final_hc_shard` rank 0: eager `1025894790`, replay `500914459`

## Decision

Sprint 563 localizes the first observed divergence to layer-1 HC-current input
state after a layer-0 replay that still matches the logged stage checksums.
The next sprint should instrument or repair the inter-layer current/HC state
handoff for no-suffix full-capture cache-hit replay. Do not retry cache-key
relaxation until layer-1 HC-current starts from the same tensors as eager.
