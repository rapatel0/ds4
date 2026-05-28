# Sprint 444: HTTP Harness Process Hygiene

## Objective

Stay TP/EP only and make the HTTP profile/A-B harness safe enough to validate
persistent graph serving without stale server processes or port collisions.

Sprint 443 added graph A/B controls, but the run became invalid because
`ds4-v100-tp-ep-full-layer-smoke` serving children survived after wrapper
failure and reused ports in later cases.

## Implementation Plan

1. Add a profile-harness port preflight before starting the launcher.
2. Add explicit stale DS4 serving-process cleanup by port.
3. Make cleanup run before startup and in `finally`.
4. Wire the A/B harness to enable that cleanup for both control and candidate
   cases.
5. Re-run a single persistent-graph HTTP smoke on a clean port.

## Validation

- Local Python syntax checks.
- Local shell syntax check for the launcher.
- Remote Python syntax checks.
- V100 no-stale-process check before and after the run.
- Persistent-graph single HTTP smoke either produces a summary or a concrete
  model/runtime blocker with no stale GPU process left behind.

## Out Of Scope

- PP/layer-split variants.
- MTP.
- Claiming graph serving promotion without a valid readiness/parity A/B.

## Execution Summary

Implemented the process-hygiene pieces:

- `tools/ds4-v100-tp-ep-profile.py`
  - Added managed DS4 server discovery by `--serve-http --port`.
  - Added `--kill-stale-server` preflight/finally cleanup.
  - Added a hard preflight failure when the port is already open and cleanup is
    not enabled.
  - Fixed HTTP server request budgeting so `/health`, `/status`, and `/metrics`
    do not consume the entire configured generation request budget.
- `tools/ds4-v100-tp-ep-nccl-http-ab.py`
  - Enables `--kill-stale-server` for both control and candidate profile runs.

While validating persistent graph serving, also fixed concrete graph-path issues
found on the V100 node:

- Added `position` to the per-layer persistent graph cache key. Reusing a graph
  captured at an earlier decode position was fast but incorrect.
- Added graph-stream typed KV stores for compressed and indexer rows.
- Added graph-stream typed KV row loads in `ds4_v100_tp_runtime`.
- Added a fallback from contiguous expert allocation to per-expert allocation
  when the contiguous allocation fails.
- Replaced one graph-time indexer top-k `cudaMemcpyPeerAsync` with a captured
  device-copy kernel.

## Cluster Evidence

Artifacts:

- `/localpool/ds4/workspace/logs/s444-graph-single-fixed`
- `/localpool/ds4/workspace/logs/s444-graph-ab-fixed`
- `/localpool/ds4/workspace/logs/s444-graph-ab-position-key`
- `/localpool/ds4/workspace/logs/s444-graph-candidate-smoke3`
- `/localpool/ds4/workspace/logs/s444-graph-candidate-smoke4`

Valid positive smoke:

- Shape: `8` slots, `256K` context, `2` HTTP chat requests, `2` generated
  tokens/request.
- Persistent graph HTTP candidate returned `2/2` HTTP 200.
- Server generated decode: `41.291192 tok/s`.
- Server continuation decode: `41.323516 tok/s`.
- Average GPU util: `27.16%`.
- Max GPU util: `63%`.
- Max VRAM used: `28083 MiB`; minimum free: `4410 MiB`.

Invalid fast A/B:

- Shape: `8` requests, `8` slots, `256K`, `4` tokens/request.
- Control server generated decode: `19.817413 tok/s`.
- Persistent graph candidate generated decode: `41.035001 tok/s`.
- Speedup: `2.07x` server generated decode, `2.60x` average GPU util.
- Rejected because response parity failed `0/8`; the persistent graph cache was
  reusing graphs across decode positions.

Position-keyed graph follow-up:

- Adding `position` to the graph key prevents stale-position graph reuse, but it
  removes the useful persistent-cache behavior and exposes remaining capture
  safety/memory problems.
- Stream-aware compressed/indexer stores, stream-aware typed KV loads, and
  graph-time top-k device copies moved the blocker forward.
- The latest minimal candidate smoke still does not produce HTTP 200. It exits
  during a later position/layer-2 graph capture attempt after repeated
  per-position graph captures, with low utilization and no promotable serving
  result.

## Decision

The HTTP harness cleanup is complete and should stay.

Persistent graph serving is not promotable in its current form:

- The fast result was invalid because it reused position-dependent graphs.
- Correct position-keying makes each position recapture/reinstantiate, which is
  too expensive and still fragile.
- The real production design needs graph parameterization/device-side dynamic
  state, not a per-position graph cache.

## Next Work

1. Keep the harness cleanup and request-budget fix.
2. Do not enable persistent graph serving by default.
3. If graph work continues, replace position-baked kernel args with device
   scalar/state buffers so one captured graph can be replayed across positions
   safely.
4. Otherwise return to the TP/EP throughput path: rank-major serving A/B,
   32-slot memory headroom, and the full-shape device-side routed FFN executor.
