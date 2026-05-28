# Sprint 443: Rank-Major Serving Harness

## Objective

Stay TP/EP only and move the serving harness toward the rank-major execution
shape.

The current anti-pattern is `gather full hidden to device 0 -> compute there ->
redistribute`. The first bounded step is to make HTTP A/Bs explicitly enable
the existing rank-major candidates so they can be tested with readiness,
response parity, GPU sampling, and VRAM checks.

## Implementation Plan

1. Fix HTTP-mode env wiring for rank-major router logits.
2. Add HTTP A/B harness flags for control/candidate:
   - rank-local/rank-major attention projection input;
   - rank-major FFN input;
   - rank-major router logits.
3. Keep fixed-capacity post-attention route planning as the current safe route
   metadata path.
4. Run an 8-request / 8-slot / 256K / 2-token HTTP A/B:
   - control: HC-current NCCL + post-attention FFN input + fixed-capacity route
     plan;
   - candidate: control plus rank-major attention/FFN/router inputs.
5. Promote only if readiness and response parity pass and server decode improves.

## Validation

- Local and remote Python syntax checks.
- V100 HTTP A/B artifact with readiness, response parity, summaries, and GPU
  utilization samples.
- No stale GPU compute processes after the run.

## Out Of Scope

- PP/layer-split variants.
- MTP.
- Actual-route-sync or other host-synchronized route-count production paths.

## Current Status

The harness changes are implemented and syntax-checked. The V100 A/B has not
yet produced a valid rank-major performance result.

Observed blockers:

- 8 requests / max 16 failed before serving during expert residency allocation
  even with scratch512/deferred NCCL:
  `cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:9643: out of memory`.
- The reduced 4 request / max 12 run was interrupted before readiness by an
  external root cleanup process that sent `SIGTERM` to the managed server.

Do not treat the interrupted `rc=-9` / `rc=-15` exits as rank-major runtime
evidence. The next run needs an exclusive node window.

## Current Outcome

Added the HTTP A/B harness graph controls and profile summary parsing.

The first V100 graph candidate exposed a capture blocker:

```text
cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:9817:
operation not permitted when stream is capturing
```

That line is `collect_tensor_f32_stats`, so graph serving must run with semantic
stats skipped. The follow-up run did emit `--true-ds4-semantic-skip-stats-gate`,
but the paired A/B became invalid because overlapping server processes reused
the same port. A single persistent-graph HTTP smoke then failed in the profile
wrapper with `server exited rc=-15` before a summary was written.

## Decision

Do not claim a graph serving result yet.

Before the next graph A/B, harden the HTTP profile wrapper:

- kill the whole process group on failure;
- verify no stale serving process remains before each case;
- preflight the chosen port;
- rerun a single persistent-graph HTTP smoke before paired A/B.
