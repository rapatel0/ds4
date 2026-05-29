# Sprint 572 - C1 Early Continuation Full-Capture Replay-State Repair

Date: 2026-05-29

## Goal

Repair or decisively narrow the no-suffix full-capture serving divergence at
continuation step `0 -> 1`.

Sprint 571 showed this is not a pure long-generation threshold. The recreated
Sprint 569 shape diverged for every measured response at continuation offset
`1`, while the longer prompt can diverge immediately at offset `0`.

## Working Hypothesis

The no-suffix full-capture cache-miss path may corrupt live serving state.

Current flow in `engine/decode_loop.cu`:

1. If full-capture replay-probe is active and the graph cache misses,
   `run_eager_decode_steps()` runs the step and serves that result.
2. `attempt_capture_probe(true)` then captures a full step on the same live
   buffers to populate the persistent graph cache.
3. The capture-only path restores host pointer state, but stream capture /
   graph setup may still leave device-side KV/current/history state different
   from the served eager result.

That shape matches the Sprint 571 result: token `0` can match because it is the
served eager result, while token `1` reads state after the capture-only pass and
diverges.

## Plan

1. Inspect the no-suffix full-capture cache-miss path and confirm whether the
   capture-only pass can mutate device state after the served eager step.
2. Add the narrowest safe repair:
   - Prefer a cache-miss flow that captures and then launches the captured graph
     exactly once as the served result, avoiding a separate eager-then-capture
     mutation.
   - If capture-as-result is still invalid, add same-logical-point diagnostics
     that prove which mutable state differs before choosing a broader snapshot
     design.
3. Validate with request-level generated token metadata only:
   - recreated `s569-shape`: `32` slots / `32` tokens / one warmup / one
     measured batch;
   - `s570-prompt-32`: same shape with the longer prompt.
4. Promote only if both cells match all generated token sequences and graph
   replay is active. Otherwise reject the code candidate and record the next
   blocker.

## Constraints

- Do not add permanent feature flags.
- Do not use broad tensor checksums unless they are taken at the same logical
  point in eager and replay.
- Do not promote no-suffix full capture unless request-level generated token
  sequences match.
- Preserve the current promoted suffix-control path.

## Definition of Done

- A local code candidate or diagnostic patch is implemented and built on the
  V100 node, or rejected with direct evidence.
- Validation artifacts are recorded.
- Recreated `s569-shape` and `s570-prompt-32` request-level generated sequences
  are compared across suffix-control and no-suffix full capture.
- Steering and vision are updated with the result.
- All repo changes from this sprint are committed, excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`.

## Execution

Implemented the narrow cache-miss candidate in a remote copy at
`/workspace/s572-early-continuation-repair`:

1. On no-suffix full-capture cache miss, skip the eager-before-capture serve.
2. Capture and instantiate the full graph.
3. Launch that captured graph once and use it as the served result.
4. Mirror the post-replay full-capture host swaps after the capture launch.

The appliance build passed:

```text
make appliance/ds4-v100-tp-ep-appliance
```

The build emitted an expected candidate-only warning: `full_capture_cache_hit`
was no longer referenced after removing the cache-miss eager branch. Because the
candidate failed validation, the local code trial was removed instead of
cleaned up.

## Validation

Artifacts:

```text
/workspace/s572-early-continuation-repair-artifacts
```

Validation used deterministic request-level generated token sequences as the
oracle. Startup/readiness and warmup requests were excluded from the measured
comparison. Both cells ran the promoted suffix-control leg against the
no-suffix full-capture candidate.

| Cell | Shape | Result |
| --- | --- | --- |
| `s569-shape` | Sprint 569 prompt, `32` slots, `32` tokens, one full-slot warmup, one measured batch, position `250000` | Failed: generated token sequences diverged for `32/32` measured responses; first diff offsets began `[2, 2, 2, 2, 2, 0, 2, 2, 2, 2]`. |
| `s570-prompt-32` | Sprint 570 prompt, `32` slots, `32` tokens, one full-slot warmup, one measured batch, position `250000` | Failed: generated token sequences diverged for `32/32` measured responses; first diff offset was `0` for the first ten mismatches. |

## Decision

Reject the cache-miss capture-as-served-result candidate.

The candidate moved the `s569-shape` first divergence from mostly continuation
offset `1` in Sprint 571 to mostly offset `2`, so it changed the failure shape,
but it did not repair serving parity. The `s570-prompt-32` cell still diverged
immediately at offset `0`.

The promoted tree has no code change from this sprint. The next C1 sprint
should instrument comparable same-logical-point request state around
continuation step `0 -> 1`, specifically:

- prompt-cache/coalescing metadata;
- slot assignment and request ordering;
- decode input token and selected-token handoff;
- full-capture HC/current rebase timing and post-replay host metadata.

Avoid timing-shifted tensor or scratch-buffer comparisons. Request-level
generated token sequences remain the serving parity oracle; lower-level logs
must be taken at the same logical point in both legs before they are treated as
evidence.
